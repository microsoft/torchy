// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

// TODO: deep copy input tensors modified in place
// TODO: lazy in-place modification for torchy tensors. copy otherwise

#undef NDEBUG
#include "tensor.h"
#include "dispatch.h"
#include "trace.h"
#include <ATen/RedispatchFunctions.h>
#include <torch/library.h>
#include <iostream>
#include <map>

#ifdef C10_DISABLE_TENSORIMPL_EXTENSIBILITY
# error Cannot disable C10_DISABLE_TENSORIMPL_EXTENSIBILITY
#endif

using namespace at;
using namespace std;

static thread_local Trace trace;


class TorchyTensor final : public TensorImpl {
  c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl> tensor;
  unsigned trace_idx;
  bool materialized = false;

public:
  template<typename... T>
  TorchyTensor(caffe2::TypeMeta dtype, c10::Device device, const T&... args)
    : TensorImpl(DISPATCHKEY, dtype, device) {
    trace_idx = trace.register_tensor(this, args...);
  }

  TorchyTensor(Tensor &&t) : TensorImpl(DISPATCHKEY, t.dtype(), t.device()) {
    set(move(t));
  }

  template<typename... T>
  void addInplace(const T&... args) {
    materialized = false;
    auto idx = trace.register_tensor(this, args...);
    if (trace_idx == -1u)
      trace_idx = idx;

    // if our data is shared, we need to flush straight away, otherwise
    // another tensor with the same data will miss that the tensor is not
    // current anymore.
    if (sharedImpl())
      trace.flush();
  }

  unsigned getTraceIdx() const { return trace_idx; }
  bool sharedImpl() const { return tensor && !tensor.unique(); }

  void set(Tensor &&t) {
    assert(!materialized && !tensor);
    assert(dtype() == t.dtype());
    assert(device() == t.device());

    trace_idx    = -1u;
    materialized = true;

    // we need to keep a reference-counted tensor because tensors may be shared
    // e.g. to() may return the input tensor
    tensor = t.unsafeReleaseIntrusivePtr();

    auto my_ks = key_set_;
    TensorImpl::shallow_copy_from(tensor);
    key_set_ = key_set_ | my_ks;
  }

  void initInPlaceUpdate() {
    // The tensor is materialized; it just has an old value
    // which is needed to compute the in-place op. Let's pretend it's up-to-date
    // so the getters below work.
    assert(tensor);
    materialized = true;
  }

  void endInPlaceUpdate() {
    trace_idx = -1u;
  }

  void ensure_materialized() const {
    if (!materialized) {
      trace.flush();
      assert(materialized);
    }
    assert(tensor);
  }

  friend ostream& operator<<(ostream &os, const TorchyTensor &tt) {
    return os << Tensor(tt.tensor);
  }

  void release_resources() override {
    if (trace_idx != -1u)
      trace.set_unobservable(trace_idx);
    tensor.reset();
    TensorImpl::release_resources();
  }

  IntArrayRef sizes() const override {
    ensure_materialized();
    return tensor->sizes();
  }

  IntArrayRef strides() const override {
    ensure_materialized();
    return tensor->strides();
  }

  int64_t dim() const override {
    ensure_materialized();
    return tensor->dim();
  }

  int64_t numel() const override {
    ensure_materialized();
    return tensor->numel();
  }

  bool is_contiguous(at::MemoryFormat memory_format) const override {
    ensure_materialized();
    return tensor->is_contiguous(memory_format);
  }

  int64_t storage_offset() const override {
    ensure_materialized();
    return tensor->storage_offset();
  }

  const char* tensorimpl_type_name() const override {
    return "TorchyTensor";
  }

  void set_size(int64_t dim, int64_t new_size) override {
    ensure_materialized();
    tensor->set_size(dim, new_size);
    TensorImpl::set_size(dim, new_size);
  }

  void set_stride(int64_t dim, int64_t new_stride) override {
    ensure_materialized();
    tensor->set_stride(dim, new_stride);
    TensorImpl::set_stride(dim, new_stride);
  }

  void set_storage_offset(int64_t storage_offset) override {
    ensure_materialized();
    tensor->set_storage_offset(storage_offset);
    TensorImpl::set_storage_offset(storage_offset);
  }

  int64_t size(int64_t d) const override {
    ensure_materialized();
    return tensor->size(d);
  }

  int64_t stride(int64_t d) const override {
    ensure_materialized();
    return tensor->stride(d);
  }

  c10::intrusive_ptr<TensorImpl>
  shallow_copy_and_detach(const c10::VariableVersion &version_counter,
                          bool allow_tensor_metadata_change) const override {
    assert(0 && "TorchyTensor::shallow_copy_and_detach(1)");
    return {};
  }

  c10::intrusive_ptr<TensorImpl>
  shallow_copy_and_detach(c10::VariableVersion &&version_counter,
                          bool allow_tensor_metadata_change) const override {
    assert(0 && "TorchyTensor::shallow_copy_and_detach(2)");
    return {};
  }

  void shallow_copy_from(const c10::intrusive_ptr<TensorImpl> &impl) override {
    assert(0 && "TorchyTensor::shallow_copy_from");
  }
};


static TorchyTensor* is_torchy(const Tensor &t) {
  return dynamic_cast<TorchyTensor*>(t.unsafeGetTensorImpl());
}

unsigned trace_idx(const Tensor &t) {
  if (auto tt = is_torchy(t))
    return tt->getTraceIdx();
  return -1u;
}

void set(TorchyTensor *tt, Tensor &&t) {
  tt->set(move(t));
}

void init_update_in_place(TorchyTensor *tt) {
  tt->initInPlaceUpdate();
}

void end_update_in_place(TorchyTensor *tt) {
  tt->endInPlaceUpdate();
}


namespace {
void ensure_materialized() {}

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

template<typename A, typename... T>
void ensure_materialized(const A &a, T&... args) {
  ensure_materialized(a);
  ensure_materialized(args...);
}

void will_override(const Tensor &t) {
  if (auto tt = is_torchy(t)) {
    if (tt->sharedImpl() && !trace.is_flushing())
      trace.flush();
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
