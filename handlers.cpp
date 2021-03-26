#include <ATen/NativeFunctions.h>
#include <ATen/Tensor.h>
#include <torch/library.h>
#include <iostream>
#include "dispatch.h"

#ifdef C10_DISABLE_TENSORIMPL_EXTENSIBILITY
#error Cannot disable C10_DISABLE_TENSORIMPL_EXTENSIBILITY
#endif

using namespace at;
using namespace std;

namespace {

struct TorchyTensor final : public TensorImpl {
  TorchyTensor(caffe2::TypeMeta dtype, c10::Device device)
    : TensorImpl(DISPATCHKEY, dtype, device) {}

  IntArrayRef sizes() const override {
    cout << "Called TorchyTensor::sizes()" << endl;
    return {};
  }

  IntArrayRef strides() const override {
    cout << "Called TorchyTensor::strides()" << endl;
    return {};
  }

  int64_t dim() const override {
    cout << "Called TorchyTensor::dim()" << endl;
    return {};
  }

  bool has_storage() const override {
    cout << "Called TorchyTensor::has_storage()" << endl;
    return {};
  }

  int64_t numel() const override {
    cout << "Called TorchyTensor::numel()" << endl;
    return {};
  }

  const char* tensorimpl_type_name() const override {
    return "TorchyTensor";
  }

  void set_size(int64_t dim, int64_t new_size) override {
    cout << "Called TorchyTensor::set_size()" << endl;
  }

  void set_stride(int64_t dim, int64_t new_stride) override {
    cout << "Called TorchyTensor::set_stride()" << endl;
  }

  void set_storage_offset(int64_t storage_offset) override {
    cout << "Called TorchyTensor::set_storage_offset()" << endl;
  }

  int64_t size(int64_t d) const override {
    cout << "Called TorchyTensor::size()" << endl;
    return {};
  }

  int64_t stride(int64_t d) const override {
    cout << "Called TorchyTensor::stride()" << endl;
    return {};
  }

  c10::intrusive_ptr<TensorImpl>
  shallow_copy_and_detach(const c10::VariableVersion &version_counter,
                          bool allow_tensor_metadata_change) const override {
    cout << "Called TorchyTensor::shallow_copy_from(1)" << endl;
    return {};
  }

  c10::intrusive_ptr<TensorImpl>
  shallow_copy_and_detach(c10::VariableVersion &&version_counter,
                          bool allow_tensor_metadata_change) const override {
    cout << "Called TorchyTensor::shallow_copy_from(2)" << endl;
    return {};
  }

  void shallow_copy_from(const c10::intrusive_ptr<TensorImpl> &impl) override {
    cout << "Called TorchyTensor::shallow_copy_from(3)" << endl;
  }
};


Tensor abs(const Tensor &self) {
  cout << "Called abs" << endl;
  return at::detail::make_tensor<TorchyTensor>(self.dtype(), self.device());
}

Tensor add_Tensor(const Tensor &self, const Tensor &other,
                  const Scalar &alpha) {
  cout << "Called add.Tensor" << endl;
  return at::detail::make_tensor<TorchyTensor>(self.dtype(), self.device());
}

Tensor as_strided(const Tensor &self, IntArrayRef size, IntArrayRef stride,
                  c10::optional<int64_t> storage_offset) {
  cout << "Called as_strided" << endl;
  return at::detail::make_tensor<TorchyTensor>(self.dtype(), self.device());
}

Tensor& bitwise_and_Tensor_out(const Tensor &self, const Tensor &other,
                               Tensor &out) {
  cout << "Called bitwise_and.Tensor_out" << endl;
  return out;
}

Tensor& ceil_out(const Tensor &self, Tensor &out) {
  cout << "Called ceil.out" << endl;
  return out;
}

Tensor& copy_(Tensor &self, const Tensor &src, bool non_blocking) {
  cout << "Called copy_" << endl;
  return self;
}

Tensor& detach_(Tensor &self) {
  cout << "Called detach_" << endl;
  return self;
}

Tensor empty_memory_format(IntArrayRef size, c10::optional<ScalarType> dtype,
                           c10::optional<Layout> layout,
                           c10::optional<Device> device,
                           c10::optional<bool> pin_memory,
                           c10::optional<MemoryFormat> memory_format) {
  cout << "Called empty.memory_format" << endl;
  return native::empty_cpu(size, dtype, layout, device, pin_memory,
                           memory_format);
}

Tensor empty_strided(IntArrayRef size, IntArrayRef stride,
                     c10::optional<ScalarType> dtype,
                     c10::optional<Layout> layout,
                     c10::optional<Device> device,
                     c10::optional<bool> pin_memory) {
  cout << "Called empty_strided" << endl;
  return
    native::empty_strided_cpu(size, stride, dtype, layout, device, pin_memory);
}

Tensor eq_Tensor(const Tensor &self, const Tensor &other) {
  cout << "Called eq.Tensor" << endl;
  return at::detail::make_tensor<TorchyTensor>(self.dtype(), self.device());
}

Tensor masked_select(const Tensor &self, const Tensor &mask) {
  cout << "Called masked_select" << endl;
  return at::detail::make_tensor<TorchyTensor>(self.dtype(), self.device());
}

Tensor max(const Tensor &self) {
  cout << "Called max" << endl;
  return at::detail::make_tensor<TorchyTensor>(self.dtype(), self.device());
}

Tensor min(const Tensor &self) {
  cout << "Called min" << endl;
  return at::detail::make_tensor<TorchyTensor>(self.dtype(), self.device());
}

Tensor mul_Tensor(const Tensor &self, const Tensor &other) {
  cout << "Called mul.Tensor" << endl;
  return at::detail::make_tensor<TorchyTensor>(self.dtype(), self.device());
}

Tensor ne_Scalar(const Tensor &self, const Scalar &other) {
  cout << "Called ne.Scalar" << endl;
  return at::detail::make_tensor<TorchyTensor>(self.dtype(), self.device());
}

Tensor ne_Tensor(const Tensor &self, const Tensor &other) {
  cout << "Called ne.Tensor" << endl;
  return at::detail::make_tensor<TorchyTensor>(self.dtype(), self.device());
}

Tensor reshape(const Tensor &self, IntArrayRef shape) {
  cout << "Called reshape" << endl;
  return at::detail::make_tensor<TorchyTensor>(self.dtype(), self.device());
}

Tensor to_device(const Tensor &self, Device device, ScalarType dtype,
                 bool non_blocking, bool copy,
                 c10::optional<MemoryFormat> memory_format) {
  cout << "Called to.device" << endl;
  return self;
}

Tensor view(const Tensor &self, IntArrayRef size) {
  cout << "Called view" << endl;
  return at::detail::make_tensor<TorchyTensor>(self.dtype(), self.device());
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("abs", abs);
  m.impl("add.Tensor", add_Tensor);
  m.impl("as_strided", as_strided);
  m.impl("bitwise_and.Tensor_out", bitwise_and_Tensor_out);
  m.impl("ceil.out", ceil_out);
  m.impl("copy_", copy_);
  m.impl("detach_", detach_); // FIXME: RegisterDefaultBackend
  m.impl("empty.memory_format", empty_memory_format); // FIXME: not called
  m.impl("empty_strided", empty_strided); // FIXME: not called
  m.impl("eq.Tensor", eq_Tensor);
  m.impl("masked_select", masked_select);
  m.impl("max", max);
  m.impl("min", min);
  m.impl("mul.Tensor", mul_Tensor);
  m.impl("ne.Scalar", ne_Scalar);
  m.impl("ne.Tensor", ne_Tensor);
  m.impl("reshape", reshape); // FIXME: RegisterMath
  m.impl("to.device", to_device); // FIXME: RegisterMath
  m.impl("view", view);
}

}
