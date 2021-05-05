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

#define DUMMY_TORCHY 0x1

using namespace at;
using namespace std;

namespace {

/*thread_local*/ Trace trace;

class TorchyStorage : public StorageImpl {
  bool materialized = false;

public:
  TorchyStorage(StorageImpl &&other)
    : StorageImpl({}, 0, {}, nullptr, false), materialized(true) {
    StorageImpl::operator=(move(other));
  }

  friend class TorchyTensor;
};


class TorchyTensor final : public TensorImpl {
  unsigned trace_idx = -1u;

  TorchyStorage& tstorage() const {
    assert(storage_);
    return *static_cast<TorchyStorage*>(storage_.unsafeGetStorageImpl());
  }

  bool materialized() const {
    return storage_ && tstorage().materialized;
  }

  bool shared() const {
    return storage_ && !storage_.unique();
  }

public:
  TorchyTensor(DispatchKeySet key_set, caffe2::TypeMeta data_type,
               const c10::optional<c10::Device> &device_opt)
    : TensorImpl(key_set, data_type, device_opt) {}

  template<typename... T>
  TorchyTensor(caffe2::TypeMeta dtype, c10::Device device, const T&... args)
    : TensorImpl(DISPATCHKEY, dtype, device) {
    trace_idx = trace.register_tensor((uintptr_t)this, args...);
  }

  TorchyTensor(Tensor &&t) : TensorImpl(DISPATCHKEY, t.dtype(), t.device()) {
    set(move(t));
  }

  template<typename... T>
  void addInplace(const T&... args) {
    if (storage_)
      tstorage().materialized = false;

    auto idx = trace.register_tensor((uintptr_t)this, args...);
    if (trace_idx == -1u)
      trace_idx = idx;

    // if our data is shared, we need to flush straight away, otherwise
    // another tensor with the same data will miss that the tensor is not
    // current anymore.
    if (shared())
      trace.flush();
  }

  unsigned getTraceIdx() const { return trace_idx; }

  void set(Tensor &&t) {
    assert(!materialized());
    assert(dtype() == t.dtype());
    assert(device() == t.device());

    trace_idx = -1u;

    auto my_ks = key_set_;
    TensorImpl::shallow_copy_from(t.getIntrusivePtr());
    key_set_ = key_set_ | my_ks;

    assert(storage_);
    auto storage_impl
      = intrusive_ptr<StorageImpl>::reclaim(
          storage_.unsafeReleaseStorageImpl());
    assert(storage_impl.unique());
    storage_ = Storage(c10::make_intrusive<TorchyStorage>(move(*storage_impl)));
    tstorage().materialized = true;
  }

  void initInPlaceUpdate() {
    // The tensor is materialized; it just has an old value
    // which is needed to compute the in-place op. Let's pretend it's up-to-date
    // so the getters below work.
    tstorage().materialized = true;
  }

  void endInPlaceUpdate() {
    trace_idx = -1u;
  }

  void ensure_materialized() const {
    if (!materialized()) {
      trace.flush();
      assert(materialized());
    }
  }

  void release_resources() override {
    if (trace_idx != -1u)
      trace.set_unobservable(trace_idx);
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

  c10::intrusive_ptr<TensorImpl>
  shallow_copy_and_detach(const c10::VariableVersion &version_counter,
                          bool allow_tensor_metadata_change) const override {
    auto copy
      = c10::make_intrusive<TorchyTensor>(key_set_, data_type_, device_opt_);

    copy_tensor_metadata(this, copy.get(), version_counter,
                         allow_tensor_metadata_change);
    copy->refresh_numel();
    copy->refresh_contiguous();
    return copy;
  }

  c10::intrusive_ptr<TensorImpl>
  shallow_copy_and_detach(c10::VariableVersion &&version_counter,
                          bool allow_tensor_metadata_change) const override {
    auto copy
      = c10::make_intrusive<TorchyTensor>(key_set_, data_type_, device_opt_);

    copy_tensor_metadata(this, copy.get(), move(version_counter),
                         allow_tensor_metadata_change);
    copy->refresh_numel();
    copy->refresh_contiguous();
    return copy;
  }

  void shallow_copy_from(const c10::intrusive_ptr<TensorImpl> &impl) override {
    TensorImpl::shallow_copy_from(impl);

    trace_idx = -1u;
    if (auto tt = dynamic_cast<TorchyTensor*>(impl.get()))
      trace_idx = tt->trace_idx;
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

void set(uintptr_t tt, Tensor &&t) {
  assert(tt != DUMMY_TORCHY);
  ((TorchyTensor*)tt)->set(move(t));
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

template<typename... T>
void compute_in_place(Tensor &t, const T&... args) {
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

  if (tt) {
    tt->addInplace(args...);
    return;
  }

  // If this happens in practice, then we could optimize it further
  assert(!dynamic_cast<TorchyStorage*>(t.storage().unsafeGetStorageImpl()));

  trace.register_tensor(DUMMY_TORCHY, args...);
  trace.flush();
}

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
