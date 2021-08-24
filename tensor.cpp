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
    if (!has_shape_data || shared())
      return;
    auto real_shape = TensorImpl::sizes();
    assert(real_shape.size() == inferred_shape_dims);
    for (unsigned i = 0; i < inferred_shape_dims; ++i) {
      assert(real_shape[i] == inferred_shape[i]);
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
    inferred_shape_dims = tt->inferred_shape_dims;
    inferred_shape      = tt->inferred_shape;
#endif
  }

public:
  TorchyTensor(DispatchKeySet key_set, caffe2::TypeMeta dtype,
               const c10::optional<c10::Device> &device_opt)
    : TensorImpl(key_set, dtype, device_opt) {}

  TorchyTensor(caffe2::TypeMeta dtype, c10::Device device)
    : TensorImpl(DISPATCHKEY, dtype, device) {}

  TorchyTensor(Tensor &&t) : TensorImpl(DISPATCHKEY, t.dtype(), t.device()) {
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

  void set_no_shape_info() {
    has_shape_data = false;
  }

  unsigned getTraceIdx() const { return trace_idx; }
  bool hasShapeData() const { return has_shape_data; }

  void set(const Tensor &t) {
    assert(dtype() == t.dtype());
    assert(device() == t.device());

    trace_idx = -1u;

    auto *other = t.getIntrusivePtr().get();
    copy_tensor_metadata(other, this, other->version_counter(),
                         other->allow_tensor_metadata_change());

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
    // TODO
    if (!has_shape_data || true)
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

    if (trace_idx != -1u)
      trace.add_shared(trace_idx, (uintptr_t)copy.get());
    copy->trace_idx = trace_idx;

    copy_tensor_metadata(this, copy.get(), forward<T>(version_counter),
                         allow_tensor_metadata_change);
    copy->numel_ = numel_;
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

    if (auto tt = dynamic_cast<TorchyTensor*>(impl.get())) {
      if (tt->trace_idx != -1u) {
        trace.add_shared(tt->trace_idx, (uintptr_t)this);
        trace_idx = tt->trace_idx;
      }
      copy_torchy_data(tt);
    }

    TensorImpl::shallow_copy_from(impl);
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

bool tensor_has_shape(uintptr_t tt) {
  return tt != DUMMY_TORCHY && ((TorchyTensor*)tt)->hasShapeData();
}

void tensor_print_shape(ostream &os, uintptr_t tt) {
  assert(tt != DUMMY_TORCHY);
  os << ((TorchyTensor*)tt)->sizes();
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

Tensor register_new_tensor(DispatchKeySet ks, TorchOp op,
                           caffe2::TypeMeta dtype, c10::Device device) {
  auto tt = at::detail::make_tensor<TorchyTensor>(dtype, device);
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
  auto dev = device ? *device : Device(kCPU);
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
