// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

// TODO: deep copy input tensors modified in place
// TODO: lazy in-place modification for torchy tensors. copy otherwise

#undef NDEBUG
#include "dispatch.h"
#include <ATen/NativeFunctions.h>
#include <ATen/RedispatchFunctions.h>
#include <ATen/Tensor.h>
#include <c10/util/variant.h>
#include <torch/library.h>
#include <iostream>
#include <map>

#ifdef C10_DISABLE_TENSORIMPL_EXTENSIBILITY
# error Cannot disable C10_DISABLE_TENSORIMPL_EXTENSIBILITY
#endif

using namespace at;
using namespace std;

namespace {

class TorchyTensor;
using TorchTensorImpl = c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>;

unsigned trace_idx(const Tensor &t);
void print(ostream &os, const TorchyTensor &tt);
void set(TorchyTensor *tt, Tensor &&t);
void init_update_in_place(TorchyTensor *tt);
void end_update_in_place(TorchyTensor *tt);


thread_local Trace trace;


class ScopedAutoFlush {
  bool was_flushing;
public:
  ScopedAutoFlush() : was_flushing(trace.is_flushing()) {
    if (!was_flushing) {
      trace.flush();
      trace.set_flushing(true);
    }
  }

  ~ScopedAutoFlush() {
    trace.set_flushing(was_flushing);
  }
};


class TorchyTensor final : public TensorImpl {
  TorchTensorImpl tensor;
  unsigned trace_idx;
  bool materialized = false;

public:
  template<typename... T>
  TorchyTensor(caffe2::TypeMeta dtype, c10::Device device, unsigned op_id,
               DispatchKeySet ks, const T&... args)
    : TensorImpl(DISPATCHKEY, dtype, device) {
    trace_idx = trace.register_tensor(this, op_id, ks, args...);
  }

  TorchyTensor(Tensor &&t) : TensorImpl(DISPATCHKEY, t.dtype(), t.device()) {
    set(move(t));
  }

  template<typename... T>
  void addInplace(unsigned op_id, DispatchKeySet ks, const T&... args) {
    materialized = false;
    auto idx = trace.register_tensor(this, op_id, ks, args...);
    if (trace_idx == -1u)
      trace_idx = idx;
  }

  unsigned getTraceIdx() const { return trace_idx; }

  void set(Tensor &&t) {
    assert(!materialized && !tensor);
    assert(dtype() == t.dtype());
    trace_idx    = -1u;
    materialized = true;
    tensor       = t.unsafeReleaseIntrusivePtr();

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
    if (tensor)
      tensor->release_resources();
    if (trace_idx != -1u)
      trace.set_unobservable(trace_idx);
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

TorchyTensor* is_torchy(const Tensor &t) {
  return dynamic_cast<TorchyTensor*>(t.unsafeGetTensorImpl());
}

unsigned trace_idx(const Tensor &t) {
  if (auto tt = is_torchy(t))
    return tt->getTraceIdx();
  return -1u;
}

void print(ostream &os, const TorchyTensor &tensor) {
  os << tensor;
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

void ensure_materialized() {}

template<typename... T>
void ensure_materialized(const Tensor &t, T&... args) {
  if (auto tt = is_torchy(t))
    tt->ensure_materialized();
  ensure_materialized(args...);
}

void will_override(const Tensor &t) {
  // TODO: refine this to account only for unprocessed references
  if (is_torchy(t)) {
    if (!trace.is_flushing())
      trace.flush();
  }
}


Tensor& mul__Tensor(c10::DispatchKeySet ks, Tensor &self, const Tensor &other) {
  ENTER("mul_");
  auto tt = is_torchy(self);
  if (tt && !trace.is_flushing()) {
    tt->addInplace(ks, "mul__Tensor", self, other);
    return self;
  }
  will_override(self);
  ensure_materialized(self, other);
  return at::redispatch::mul_(
    ks & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY),
    self, other);
}

Tensor sum(c10::DispatchKeySet ks, const Tensor &self,
           c10::optional<ScalarType> dtype) {
  ENTER("sum");
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return
      at::redispatch::sum(
         ks & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY),
         self, dtype);
  }
  auto ty = self.dtype();
  if (ty == kBool)
    ty = scalarTypeToTypeMeta(kLong);
  return MK_TORCHY(dtype ? scalarTypeToTypeMeta(*dtype) : ty, self.device(),
                   "sum", self, dtype);
}

#include "autogen/dispatch_wrappers.h"

TORCH_LIBRARY_IMPL(aten, DISPATCHKEY_NO_NS, m) {
#include "autogen/torch_library_table.h"
}

TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
  m.impl("isfinite", isfinite);
  m.impl("reshape", reshape);
  m.impl("to.device", to_device);
}

}
