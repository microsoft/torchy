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

  bool& materialized_var() const {
    assert(storage_);
    return storage_.unsafeGetStorageImpl()->reserved();
  }

  bool materialized() const {
    return storage_ && materialized_var();
  }

#if 0
  bool shared() const {
    return storage_ && !storage_.unique();
  }
#endif

public:
  TorchyTensor(DispatchKeySet key_set, caffe2::TypeMeta dtype,
               const c10::optional<c10::Device> &device_opt)
    : TensorImpl(key_set, dtype, device_opt) {}

  TorchyTensor(caffe2::TypeMeta dtype, c10::Device device)
    : TensorImpl(DISPATCHKEY, dtype, device) {}

  TorchyTensor(Tensor &&t) : TensorImpl(DISPATCHKEY, t.dtype(), t.device()) {
    set(t);
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

  unsigned getTraceIdx() const { return trace_idx; }

  void set(const Tensor &t) {
    assert(dtype() == t.dtype());
    assert(device() == t.device());

    trace_idx = -1u;

    auto my_ks = key_set_;
    auto *other = t.getIntrusivePtr().get();
    copy_tensor_metadata(other, this, other->version_counter(),
                         other->allow_tensor_metadata_change());
    key_set_ = key_set_ | my_ks;

    set_materialized(true);

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
  }

  void ensure_materialized() const {
    if (!trace.is_flushing() && !materialized()) {
      assert(trace_idx != -1u);
      trace.flush();
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
    ensure_materialized();
    return TensorImpl::sizes();
  }

  IntArrayRef strides() const override {
    ensure_materialized();
    return TensorImpl::strides();
  }

  int64_t dim() const override {
    ensure_materialized();
    return TensorImpl::dim();
  }

  bool has_storage() const override {
    ensure_materialized();
    return TensorImpl::has_storage();
  }

  const Storage& storage() const override {
    ensure_materialized();
    return TensorImpl::storage();
  }

  int64_t numel() const override {
    ensure_materialized();
    return TensorImpl::numel();
  }

  bool is_contiguous(at::MemoryFormat memory_format) const override {
    ensure_materialized();
    return TensorImpl::is_contiguous(memory_format);
  }

  int64_t storage_offset() const override {
    ensure_materialized();
    return TensorImpl::storage_offset();
  }

  const char* tensorimpl_type_name() const override {
    return "TorchyTensor";
  }

  void set_size(int64_t dim, int64_t new_size) override {
    ensure_materialized();
    TensorImpl::set_size(dim, new_size);
  }

  void set_stride(int64_t dim, int64_t new_stride) override {
    ensure_materialized();
    TensorImpl::set_stride(dim, new_stride);
  }

  void set_storage_offset(int64_t storage_offset) override {
    ensure_materialized();
    TensorImpl::set_storage_offset(storage_offset);
  }

  int64_t size(int64_t d) const override {
    ensure_materialized();
    return TensorImpl::size(d);
  }

  int64_t stride(int64_t d) const override {
    ensure_materialized();
    return TensorImpl::stride(d);
  }

  template <typename T>
  c10::intrusive_ptr<TensorImpl>
  my_shallow_copy_and_detach(T version_counter,
                             bool allow_tensor_metadata_change) const {
    auto copy
      = c10::make_intrusive<TorchyTensor>(key_set_, data_type_, device_opt_);

    if (trace_idx != -1u)
      trace.add_shared(trace_idx, (uintptr_t)copy.get());
    copy->trace_idx = trace_idx;

    copy_tensor_metadata(this, copy.get(), forward<T>(version_counter),
                         allow_tensor_metadata_change);
    copy->numel_ = numel_;
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


namespace {

TorchyTensor* prepare_in_place(const Tensor &t0) {
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
  if (tt)
    tt->set_materialized(false);
  return tt;
}

void finish_in_place(TorchyTensor *tt, unsigned idx) {
  if (tt) {
    tt->update_idx(idx);
  } else {
    trace.flush();
  }
}

void ensure_materialized(const Tensor &t) {
  if (auto tt = is_torchy(t))
    tt->ensure_materialized();
}

void ensure_materialized(const optional<Tensor> &t) {
  if (t)
    ensure_materialized(*t);
}

template<typename T>
void ensure_materialized(const ArrayRef<T> &l) {
  for (const auto &elem : l) {
    ensure_materialized(elem);
  }
}

template<typename T>
void ensure_materialized(const List<T> &l) {
  for (const auto &it : l) {
    const T &elem = it;
    ensure_materialized(elem);
  }
}

// see build/aten/src/ATen/RegisterBackendSelect.cpp for redispatching logic
pair<caffe2::TypeMeta, Device> compute_dtype() {
  return { at::get_default_dtype(), Device(kCPU) };
}

pair<caffe2::TypeMeta, Device> compute_dtype(const TensorList &list) {
  if (list.empty())
    return compute_dtype();
  return { list.front().dtype(), list.front().device() };
}

#include "autogen/dispatch_wrappers.h"

TORCH_LIBRARY_IMPL(aten, DISPATCHKEY_NO_NS, m) {
#include "autogen/torch_library_table.h"
}

TORCH_LIBRARY_IMPL(_, AUTOGRADDISPATCHKEY_NO_NS, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

}
