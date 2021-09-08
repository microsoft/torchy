// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

#include "tensor.h"
#include "dispatch.h"
#include "trace.h"
#include <ATen/RedispatchFunctions.h>
#include <torch/library.h>
#include <cassert>

#ifdef C10_DISABLE_TENSORIMPL_EXTENSIBILITY
# error Cannot disable C10_DISABLE_TENSORIMPL_EXTENSIBILITY
#endif

using namespace at;
using namespace std;

namespace {

/*thread_local*/ Trace trace;


class TorchyTensor final : public TensorImpl {
  unsigned trace_idx = -1u;
  bool has_shape_data = false;
#ifndef NDEBUG
  // True if an inplace operation may not preserve the shape.
  // A trace may have multiple ops over a same tensor. If one of those ops is
  // in-place, it may change the shape. As we only keep the inferred shape for
  // the last op, we cannot use the information for checking the shapes
  // of prev ops (for debugging purposes).
  bool has_multiple_shapes = false;
  uint8_t inferred_shape_dims;
  array<unsigned, 5> inferred_shape;
#endif

  bool& materialized_var() const {
    assert(storage_);
    return storage_.unsafeGetStorageImpl()->reserved();
  }

  bool materialized() const {
    return storage_ && materialized_var();
  }

  bool shared() const {
    return storage_ && !storage_.unique();
  }

  void check_inferred_shape() {
#ifndef NDEBUG
    if (!has_shape_data || has_multiple_shapes)
      return;

    auto real_shape = TensorImpl::sizes();

    auto error = [&]() {
      cerr << "Bad shape. Real: " << real_shape << " / Inferred: "
           << ArrayRef<unsigned>(inferred_shape.data(), inferred_shape_dims)
           << endl;
      assert(0);
    };

    if (real_shape.size() != inferred_shape_dims)
      error();

    for (unsigned i = 0; i < inferred_shape_dims; ++i) {
      if (real_shape[i] != inferred_shape[i])
        error();
    }
#endif
  }

  void store_shape() {
    has_shape_data = true;
#ifndef NDEBUG
    auto real_shape = TensorImpl::sizes();
    if (real_shape.size() > inferred_shape.size()) {
      has_shape_data = false;
      cerr << "WARN: Can't keep track of tensor with so many dimensions: "
           << real_shape.size() << endl;
      return;
    }
    inferred_shape_dims = real_shape.size();
    for (unsigned i = 0; i < inferred_shape_dims; ++i) {
      inferred_shape[i] = real_shape[i];
    }
#endif
  }

  void copy_torchy_data(const TorchyTensor *tt) {
    has_shape_data      = tt->has_shape_data;
#ifndef NDEBUG
    has_multiple_shapes = tt->has_multiple_shapes;
    inferred_shape_dims = tt->inferred_shape_dims;
    inferred_shape      = tt->inferred_shape;
#endif
  }

public:
  TorchyTensor(DispatchKeySet key_set, caffe2::TypeMeta dtype,
               c10::optional<c10::Device> device_opt)
    : TensorImpl(key_set.add(DISPATCHKEY), dtype, device_opt) {}

  TorchyTensor(Tensor &&t) : TensorImpl(t.key_set(), t.dtype(), t.device()) {
    set(t);

    // steal pyobj & friends
    auto *other = t.getIntrusivePtr().get();
    auto *interp = other->pyobj_interpreter();
    init_pyobj(interp, other->check_pyobj(interp).value_or(nullptr),
               c10::impl::PyInterpreterStatus::DEFINITELY_UNINITIALIZED);

    if (other->owns_pyobj()) {
      set_owns_pyobj(true);
      other->set_owns_pyobj(false);
    }
  }

  void set_materialized(bool val) {
    if (storage_)
      materialized_var() = val;
  }

  void set_idx(unsigned idx) {
    assert(trace_idx == -1u);
    trace_idx = idx;
  }

  void update_idx(unsigned idx) {
    if (trace_idx == -1u)
      trace_idx = idx;
  }

  void set_shape(IntArrayRef shape) {
    sizes_and_strides_.set_sizes(shape);
    store_shape();

    // must be run after setting has_shape_data as it calls sizes()
    refresh_numel();
  }

  void set_no_shape_info() {
    has_shape_data = false;
#ifndef NDEBUG
    has_multiple_shapes = true;
#endif
  }

#ifndef NDEBUG
  void resetMultipleShapes() {
    has_multiple_shapes = false;
  }
#endif

  unsigned getTraceIdx() const { return trace_idx; }
  bool hasShapeData() const { return has_shape_data; }

  void set(const Tensor &t) {
    assert(dtype() == t.dtype());
    assert(device() == t.device());

    trace_idx = -1u;

    auto *other = t.getIntrusivePtr().get();
    copy_tensor_metadata(other, this, other->version_counter(),
                         other->allow_tensor_metadata_change());
    // overriden by copy_tensor_metadata
    key_set_= key_set_.add(DISPATCHKEY);

    set_materialized(true);

    check_inferred_shape();
    store_shape();

    // must be run after materialized is set to true, as these call the
    // overriden methods below
    refresh_numel();
    refresh_contiguous();
  }

  void initInPlaceUpdate() {
    // The tensor is materialized; it just has an old value
    // which is needed to compute the in-place op. Let's pretend it's up-to-date
    // so the getters below work.
    set_materialized(true);
  }

  void endInPlaceUpdate() {
    trace_idx = -1u;
    set_materialized(true);

    check_inferred_shape();
    store_shape();
  }

  void ensure_materialized(STATS(FlushReason reason)) const {
    if (!trace.is_flushing() && !materialized()) {
      assert(trace_idx != -1u);
      trace.flush(STATS(reason));
      assert(!storage_ || materialized());
    }
  }

  void release_resources() override {
    if (trace_idx != -1u)
      trace.set_unobservable(trace_idx, (uintptr_t)this);
    TensorImpl::release_resources();
  }

  // NOTE: for the overrides below we use the default TensorImpl code
  // This means we don't support other implementations, such as sparse tensors
  // A way to fix would be to keep a pointer to the TensorImpl, but that adds
  // an extra indirection. Another way is to templatize these.

  IntArrayRef sizes() const override {
    if (!has_shape_data)
      ensure_materialized(STATS(FlushReason::SIZES));
    return TensorImpl::sizes();
  }

  IntArrayRef strides() const override {
    // TODO
    if (!has_shape_data || true)
      ensure_materialized(STATS(FlushReason::STRIDES));
    return TensorImpl::strides();
  }

  int64_t dim() const override {
    if (!has_shape_data)
      ensure_materialized(STATS(FlushReason::DIM));
    return TensorImpl::dim();
  }

  bool has_storage() const override {
    ensure_materialized(STATS(FlushReason::HAS_STORAGE));
    return TensorImpl::has_storage();
  }

  const Storage& storage() const override {
    ensure_materialized(STATS(FlushReason::STORAGE));
    return TensorImpl::storage();
  }

  int64_t numel() const override {
    if (!has_shape_data)
      ensure_materialized(STATS(FlushReason::NUMEL));
    return TensorImpl::numel();
  }

  bool is_contiguous(at::MemoryFormat memory_format) const override {
    // TODO
    if (!has_shape_data || true)
      ensure_materialized(STATS(FlushReason::IS_CONTIGUOUS));
    return TensorImpl::is_contiguous(memory_format);
  }

  int64_t storage_offset() const override {
    ensure_materialized(STATS(FlushReason::STORAGE_OFFSET));
    return TensorImpl::storage_offset();
  }

  const char* tensorimpl_type_name() const override {
    return "TorchyTensor";
  }

  void set_size(int64_t dim, int64_t new_size) override {
    ensure_materialized(STATS(FlushReason::SET_SIZE));
    TensorImpl::set_size(dim, new_size);
    store_shape();
  }

  void set_stride(int64_t dim, int64_t new_stride) override {
    ensure_materialized(STATS(FlushReason::SET_STRIDE));
    TensorImpl::set_stride(dim, new_stride);
  }

  void set_storage_offset(int64_t storage_offset) override {
    ensure_materialized(STATS(FlushReason::SET_STORAGE_OFFSET));
    TensorImpl::set_storage_offset(storage_offset);
  }

  int64_t size(int64_t d) const override {
    if (!has_shape_data)
      ensure_materialized(STATS(FlushReason::SIZE));
    return TensorImpl::size(d);
  }

  int64_t stride(int64_t d) const override {
    // TODO
    if (!has_shape_data || true)
      ensure_materialized(STATS(FlushReason::STRIDE));
    return TensorImpl::stride(d);
  }

  template <typename T>
  c10::intrusive_ptr<TensorImpl>
  my_shallow_copy_and_detach(T &&version_counter,
                             bool allow_tensor_metadata_change) const {
    auto copy
      = c10::make_intrusive<TorchyTensor>(key_set_, data_type_, device_opt_);

    copy_tensor_metadata(this, copy.get(), forward<T>(version_counter),
                         allow_tensor_metadata_change);

    if (trace_idx != -1u)
      trace.add_shared(trace_idx, (uintptr_t)copy.get());
    copy->trace_idx = trace_idx;

    // overriden by copy_tensor_metadata
    copy->key_set_= copy->key_set_.add(DISPATCHKEY);
    copy->numel_  = numel_;
    copy->copy_torchy_data(this);
    return copy;
  }

  c10::intrusive_ptr<TensorImpl>
  shallow_copy_and_detach(const c10::VariableVersion &version_counter,
                          bool allow_tensor_metadata_change) const override {
    return
      my_shallow_copy_and_detach(version_counter, allow_tensor_metadata_change);
  }

  c10::intrusive_ptr<TensorImpl>
  shallow_copy_and_detach(c10::VariableVersion &&version_counter,
                          bool allow_tensor_metadata_change) const override {
    return my_shallow_copy_and_detach(move(version_counter),
                                      allow_tensor_metadata_change);
  }

  void shallow_copy_from(const c10::intrusive_ptr<TensorImpl> &impl) override {
    if (trace_idx != -1u)
      trace.set_unobservable(trace_idx, (uintptr_t)this);
    trace_idx = -1u;

    TensorImpl::shallow_copy_from(impl);
    // overriden by copy_tensor_metadata
    key_set_ = key_set_.add(DISPATCHKEY);

    if (auto tt = dynamic_cast<TorchyTensor*>(impl.get())) {
      if (tt->trace_idx != -1u) {
        trace.add_shared(tt->trace_idx, (uintptr_t)this);
        trace_idx = tt->trace_idx;
      }
      copy_torchy_data(tt);
    }
  }
};
}


static TorchyTensor* is_torchy(const Tensor &t) {
  return dynamic_cast<TorchyTensor*>(t.getIntrusivePtr().get());
}

unsigned trace_idx(const Tensor &t) {
  if (auto tt = is_torchy(t))
    return tt->getTraceIdx();
  return -1u;
}

TensorImpl* is_impl(uintptr_t tt) {
  return tt && tt != DUMMY_TORCHY ? (TorchyTensor*)tt : nullptr;
}

void set(uintptr_t tt, const Tensor &t) {
  assert(tt != DUMMY_TORCHY);
  ((TorchyTensor*)tt)->set(t);
}

void init_update_in_place(uintptr_t tt) {
  if (tt != DUMMY_TORCHY)
    ((TorchyTensor*)tt)->initInPlaceUpdate();
}

void end_update_in_place(uintptr_t tt) {
  if (tt != DUMMY_TORCHY)
    ((TorchyTensor*)tt)->endInPlaceUpdate();
}

#ifndef NDEBUG
void finish_trace(uintptr_t tt) {
  if (tt != DUMMY_TORCHY)
    ((TorchyTensor*)tt)->resetMultipleShapes();
}
#endif

bool tensor_has_dtype(uintptr_t tt) {
  return tt != DUMMY_TORCHY;
}

ScalarType tensor_get_dtype(uintptr_t tt) {
  assert(tt != DUMMY_TORCHY);
  // TODO: compact once upstream catches up
  auto ty = ((TorchyTensor*)tt)->dtype();
  return ty.toScalarType();
}

bool tensor_has_shape(uintptr_t tt) {
  return tt != DUMMY_TORCHY && ((TorchyTensor*)tt)->hasShapeData();
}

IntArrayRef tensor_get_shape(uintptr_t tt) {
  assert(tt != DUMMY_TORCHY);
  return ((TorchyTensor*)tt)->sizes();
}


namespace {

#include "type_inference.h"

ScalarType optional_type(const c10::optional<Tensor> &t) {
  return t ? t->dtype().toScalarType() : ScalarType::Undefined;
}

#define PASS(t) \
  t.dtype().toScalarType(), [&]() { return t.dim() == 0; }

#define PASS_OPT(t) \
  optional_type(t), [&]() { return t && t->dim() == 0; }

ScalarType to_float2(const Tensor &t1, const Tensor &t2) {
  return to_float2(PASS(t1), PASS(t2));
}

ScalarType to_float3(const Tensor &t1, const Tensor &t2, const Tensor &t3) {
  return to_float3(PASS(t1), PASS(t2), PASS(t3));
}

ScalarType to_float4(const Tensor &t1, const Tensor &t2, const Tensor &t3,
                     const c10::optional<Tensor> &t4) {
  return to_float4(PASS(t1), PASS(t2), PASS(t3), PASS_OPT(t4));
}

ScalarType to_real2(const Tensor &t1, const Tensor &t2) {
  return to_real2(PASS(t1), PASS(t2));
}

ScalarType bool_to_int2(const Tensor &t1, const Tensor &t2) {
  return bool_to_int2(PASS(t1), PASS(t2));
}

optional<IntArrayRef> shape_of(const Tensor &t) {
  // best effort; don't trigger rematerialization
  if (auto *tt = is_torchy(t)) {
    if (!tt->hasShapeData())
      return {};
  }
  return t.sizes();
}

optional<IntArrayRef> shape_of(const optional<Tensor> &t) {
  return shape_of(*t);
}

optional<IntArrayRef> shape_of(IntArrayRef shape) {
  return shape;
}

bool empty_opt_tensor(const optional<Tensor> &t) {
  return !t;
}

template <typename T>
bool empty_opt_tensor(const T&) {
  return false;
}

static std::vector<int64_t> tmp_shape;

#include "shape_inference.h"

#define GET_SHAPE(v) \
  auto shape_##v = shape_of(v); \
  if (!shape_##v) return {}

optional<IntArrayRef> shape_std_promote(IntArrayRef shape) {
  return shape;
}

template <typename A, typename B, typename... Tail>
optional<IntArrayRef> shape_std_promote(A &a, B &b, Tail&&... tail) {
  GET_SHAPE(a);
  if (empty_opt_tensor(b))
    return shape_std_promote(*shape_a, forward<Tail>(tail)...);

  GET_SHAPE(b);
  tmp_shape = shape_std_promote(*shape_a, *shape_b);
  return shape_std_promote(tmp_shape, forward<Tail>(tail)...);
}

optional<IntArrayRef> shape_matmul(const Tensor &a, IntArrayRef shape_b) {
  GET_SHAPE(a);
  return tmp_shape = shape_matmul(*shape_a, shape_b);
}

optional<IntArrayRef> shape_matmul(const Tensor &a, const Tensor &b) {
  GET_SHAPE(b);
  return shape_matmul(a, *shape_b);
}

optional<IntArrayRef> shape_mul(const Tensor &a, const Tensor &b) {
  GET_SHAPE(a);
  GET_SHAPE(b);
  return tmp_shape = shape_mul(*shape_a, *shape_b);
}

optional<IntArrayRef> shape_mul(TensorList lst) {
  if (lst.empty())
    return {};

  auto shape = shape_of(lst[0]);
  if (!shape)
    return {};

  for (unsigned i = 1, e = lst.size(); i != e; ++i) {
    auto shape_2 = shape_of(lst[i]);
    if (!shape_2)
      return {};

    tmp_shape = shape_mul(*shape, *shape_2);
    shape = tmp_shape;
  }
  return shape;
}

optional<IntArrayRef> shape_mult(const Tensor &a, const Tensor &b) {
  GET_SHAPE(a);
  GET_SHAPE(b);
  return tmp_shape = shape_mult(*shape_a, *shape_b);
}

optional<IntArrayRef> shape_mul_last(const Tensor &a, IntArrayRef shape_b) {
  GET_SHAPE(a);
  return tmp_shape = shape_mul_last(*shape_a, shape_b);
}

optional<IntArrayRef> shape_mul_last(const Tensor &a, const Tensor &b) {
  GET_SHAPE(b);
  return shape_mul_last(a, *shape_b);
}

optional<IntArrayRef> shape_pick_1st(const Tensor &t) {
  GET_SHAPE(t);
  return IntArrayRef(shape_t->data(), 1);
}

optional<IntArrayRef> shape_join(const Tensor &a, const Tensor &b) {
  GET_SHAPE(a);
  GET_SHAPE(b);
  return tmp_shape = shape_join(*shape_a, *shape_b);
}

optional<IntArrayRef> shape_pad1(const Tensor &t) {
  GET_SHAPE(t);
  return tmp_shape = shape_pad1(*shape_t);
}

optional<IntArrayRef> shape_drop1(const Tensor &t) {
  GET_SHAPE(t);
  return IntArrayRef(shape_t->data(), shape_t->size()-1);
}

optional<IntArrayRef> shape_drop2(const Tensor &t) {
  GET_SHAPE(t);
  return IntArrayRef(shape_t->data(), shape_t->size()-2);
}

optional<IntArrayRef>
shape_transpose(const Tensor &t, int64_t dim1, int64_t dim2) {
  GET_SHAPE(t);
  return tmp_shape = shape_transpose(*shape_t, dim1, dim2);
}

optional<IntArrayRef> shape_reshape(const Tensor &t, IntArrayRef to) {
  GET_SHAPE(t);
  return tmp_shape = shape_reshape(*shape_t, to);
}

optional<IntArrayRef> shape_select(const Tensor &t, int64_t dim) {
  GET_SHAPE(t);
  return tmp_shape = shape_select(*shape_t, dim);
}

optional<IntArrayRef> shape_unsqueeze(const Tensor &t, int64_t dim) {
  GET_SHAPE(t);
  return tmp_shape = shape_unsqueeze(*shape_t, dim);
}

optional<IntArrayRef>
shape_flatten(const Tensor &t, int64_t start, int64_t end) {
  GET_SHAPE(t);
  return tmp_shape = shape_flatten(*shape_t, start, end);
}

optional<IntArrayRef> shape_arange(const Scalar &start, const Scalar &end,
                                   const Scalar &step) {
  return tmp_shape = shape_arange_vec(start, end, step);
}

optional<IntArrayRef> shape_embedding(const Tensor &w, const Tensor &idxs) {
  GET_SHAPE(w);
  GET_SHAPE(idxs);
  return tmp_shape = shape_embedding(*shape_w, *shape_idxs);
}

optional<IntArrayRef> shape_slice(const Tensor &t, int64_t dim,
                                  optional<int64_t> start_opt,
                                  optional<int64_t> end_opt, int64_t step) {
  GET_SHAPE(t);
  return tmp_shape = shape_slice(*shape_t, dim, start_opt, end_opt, step);
}

optional<IntArrayRef> shape_stack(TensorList lst, int64_t dim) {
  auto shape = shape_of(lst[0]);
  if (!shape)
    return {};
  return tmp_shape = shape_stack(*shape, lst.size(), dim);
}

optional<IntArrayRef>
shape_argmax(const Tensor &t, optional<int64_t> dim, bool keepdim) {
  GET_SHAPE(t);
  return tmp_shape = shape_argmax(*shape_t, dim, keepdim);
}

optional<IntArrayRef> shape_conv2d(const Tensor &in, IntArrayRef kernel,
                                   IntArrayRef stride, IntArrayRef pad,
                                   IntArrayRef dilation,
                                   optional<int64_t> out_opt = {}) {
  GET_SHAPE(in);
  int64_t out = out_opt.value_or((*shape_in)[1]);
  return
    tmp_shape = shape_conv2d(*shape_in, kernel, stride, pad, dilation, out);
}

optional<IntArrayRef> shape_conv2d(const Tensor &in, const Tensor &w,
                                   IntArrayRef stride, IntArrayRef pad,
                                   IntArrayRef dilation) {
  GET_SHAPE(w);
  return shape_conv2d(in, shape_w->slice(2, 2), stride, pad, dilation,
                      (*shape_w)[0]);
}

optional<IntArrayRef> shape_pool2d(const Tensor &in, IntArrayRef shape) {
  GET_SHAPE(in);
  return tmp_shape = shape_pool2d(*shape_in, shape);
}

bool eq_shapes(optional<IntArrayRef> s1, optional<IntArrayRef> s2) {
  if (!s1 || !s2)
    return false;
  return *s1 == *s2;
}

bool eq_shapes(const Tensor &t1, const Tensor &t2) {
  return eq_shapes(shape_of(t1), shape_of(t2));
}

bool eq_shapes(const Tensor &t1, optional<IntArrayRef> s2) {
  return eq_shapes(shape_of(t1), s2);
}

Tensor register_new_tensor(DispatchKeySet ks, TorchOp op,
                           caffe2::TypeMeta dtype, c10::Device device) {
  auto tt = at::detail::make_tensor<TorchyTensor>(ks, dtype, device);
  auto tt_ptr = tt.getIntrusivePtr().get();
  unsigned trace_idx = trace.register_tensor((uintptr_t)tt_ptr, op, ks);
  static_cast<TorchyTensor*>(tt_ptr)->set_idx(trace_idx);
  return tt;
}

Tensor register_new_tensor(DispatchKeySet ks, TorchOp op, ScalarType dtype,
                           c10::Device device) {
  return register_new_tensor(ks, op, scalarTypeToTypeMeta(dtype), move(device));
}

Tensor register_new_tensor(DispatchKeySet ks, TorchOp op,
                           caffe2::TypeMeta dtype,
                           c10::optional<at::Device> device) {
  // see build/aten/src/ATen/RegisterBackendSelect.cpp for redispatching logic
  auto dev = device.value_or(kCPU);
  return register_new_tensor(ks, op, dtype, dev);
}

Tensor register_new_tensor(DispatchKeySet ks, TorchOp op,
                           c10::optional<at::ScalarType> dtype,
                           c10::optional<at::Device> device) {
  auto dty = dtype ? scalarTypeToTypeMeta(*dtype) : at::get_default_dtype();
  return register_new_tensor(ks, op, dty, device);
}

Tensor register_new_tensor(DispatchKeySet ks, TorchOp op,
                           const TensorList &list) {
  if (list.empty())
    return register_new_tensor(ks, op, nullopt, nullopt);
  return register_new_tensor(ks, op, promote_tys(list), list.front().device());
}

void set_shape(Tensor &t, optional<IntArrayRef> shape) {
  if (!shape)
    return;

  TorchyTensor *tt = is_torchy(t);
  assert(tt);
  tt->set_shape(*shape);
}

void set_shape(Tensor &t, const Tensor &shape_t) {
  set_shape(t, shape_of(shape_t));
}

bool register_in_place(const Tensor &t0, TorchOp op, DispatchKeySet ks,
                       bool preserves_shape) {
  auto &t = const_cast<Tensor&>(t0);
  TorchyTensor *tt = is_torchy(t);

  // if the tensor's impl & storage aren't shared, replace them with
  // Torchy equivalents so we can intercept them in the future.
  if (!tt &&
      t.getIntrusivePtr().unique() &&
      t.getIntrusivePtr()->unique_version() &&
      (!t.has_storage() || t.storage().unique())) {
    t = at::detail::make_tensor<TorchyTensor>(move(t));
    tt = is_torchy(t);
    assert(tt);
  }

  auto idx = trace.register_tensor(tt ? (uintptr_t)tt : DUMMY_TORCHY, op, ks);
  if (tt) {
    tt->set_materialized(false);
    tt->update_idx(idx);
    if (!preserves_shape)
      tt->set_no_shape_info();
    return false;
  }

  // shared; needs flushing
  return true;
}

#include "autogen/dispatch_wrappers.h"

TORCH_LIBRARY_IMPL(_, DISPATCHKEY_NO_NS, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(aten, DISPATCHKEY_NO_NS, m) {
#include "autogen/torch_library_table.h"
}

TORCH_LIBRARY_IMPL(_, AUTOGRADDISPATCHKEY_NO_NS, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

}
