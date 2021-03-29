// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

#undef NDEBUG
#include "dispatch.h"
#include <ATen/NativeFunctions.h>
#include <ATen/Tensor.h>
#include <c10/util/variant.h>
#include <torch/library.h>
#include <iostream>
#include <map>

#define MAX_TAPE_LENGTH 64

#if 1
#define DBG(x) x
#else
#define DBG(x)
#endif

#ifdef C10_DISABLE_TENSORIMPL_EXTENSIBILITY
# error Cannot disable C10_DISABLE_TENSORIMPL_EXTENSIBILITY
#endif

using namespace at;
using namespace std;

namespace {

unsigned tape_idx(const Tensor &t);


struct TensorOp {
  TensorImpl **tensor;
  const char *id;
  vector<c10::variant<const Tensor*, const c10::Scalar*>> args;
  unsigned refs;

  void incref() {
    assert(isObservable());
    ++refs;
  }

  void decref() {
    assert(refs > 0);
    --refs;
  }

  bool isObservable() const {
    return tensor;
  }

  bool needsComputing() const {
    return isObservable() || refs > 0;
  }

  void print(ostream &os, map<const Tensor*, unsigned> &inputs) const {
    if (!needsComputing()) {
      os << "[dead]";
      return;
    }

    os << id;
    bool first = true;
    for (auto &arg : args) {
      os << (first ? " " : ", ");
      first = false;
      if (auto t = get_if<const Tensor*>(&arg)) {
        auto idx = tape_idx(**t);
        if (idx != -1u) {
          os << '%' << idx;
        } else {
          auto I = inputs.emplace(*t, (unsigned)inputs.size()).first;
          os << "in<" << I->second << '>';
        }
      } else if (auto s = get_if<const Scalar*>(&arg)) {
        os << **s;
      } else {
        assert(false);
      }
    }

  if (refs > 0)
    os << " [refs=" << refs << ']';

  if (isObservable())
    os << " [output]";
  }
};


class Tape {
  TensorOp ops[MAX_TAPE_LENGTH];
  unsigned next_op = 0;

  void incref(const Scalar &s) {}
  void incref(const Tensor &t) {
    auto idx = tape_idx(t);
    if (idx != -1u)
      ops[idx].incref();
  }

  template<typename A, typename... T>
  void registerOpArgs(TensorOp &op, const A &arg, T&... args) {
    op.args.emplace_back(&arg);
    incref(arg);
    registerOpArgs(op, args...);
  }

  void registerOpArgs(TensorOp &op) {}

public:
  unsigned register_tensor(TensorImpl **tensor) {
    if (next_op == MAX_TAPE_LENGTH)
      flush();

    auto &op = ops[next_op];
    op.tensor = tensor;
    op.id = nullptr;
    op.args.clear();
    op.refs = 0;
    return next_op++;
  }

  template<typename... T>
  void registerOp(unsigned idx, const char *op_id, T&... args) {
    auto &op = ops[idx];
    op.id = op_id;
    registerOpArgs(op, args...);
  }

  void set_unobservable(unsigned idx) {
    auto &op = ops[idx];
    assert(op.tensor);
    op.tensor = nullptr;

    for (auto &arg : op.args) {
      if (auto t = get_if<const Tensor*>(&arg)) {
        auto idx = tape_idx(**t);
        if (idx != -1u)
          ops[idx].decref();
      }
    }
  }

  void flush() {
    DBG(cout << "Flush tape\n"; cout << *this;)
    // TODO
    next_op = 0;
  }

  friend ostream& operator<<(ostream &os, const Tape &t) {
    if (t.next_op == 0)
      return os << "empty tape";

    map<const Tensor*, unsigned> inputs_map;
    for (unsigned i = 0; i < t.next_op; ++i) {
      os << '%' << i << " = ";
      t.ops[i].print(os, inputs_map);
      os << '\n';
    }
    return os << '\n';
  }
};

thread_local Tape tape;


class TorchyTensor final : public TensorImpl {
  TensorImpl *tensor = nullptr;
  unsigned tape_idx;

  void ensure_tensor() const {
    if (!tensor) {
      tape.flush();
      assert(tensor);
    }
  }

public:
  TorchyTensor(caffe2::TypeMeta dtype, c10::Device device)
    : TensorImpl(DISPATCHKEY, dtype, device) {
    tape_idx = tape.register_tensor(&tensor);
  }

  template<typename... T>
  void registerOp(const char *op_id, const T&... args) {
    tape.registerOp(tape_idx, op_id, args...);
  }

  unsigned getTapeIdx() const { return tape_idx; }

  void release_resources() override {
    TensorImpl::release_resources();
    tape.set_unobservable(tape_idx);
  }

  IntArrayRef sizes() const override {
    ensure_tensor();
    return tensor->sizes();
  }

  IntArrayRef strides() const override {
    ensure_tensor();
    return tensor->strides();
  }

  int64_t dim() const override {
    ensure_tensor();
    return tensor->dim();
  }

  bool has_storage() const override {
    ensure_tensor();
    return tensor->has_storage();
  }

  int64_t numel() const override {
    ensure_tensor();
    return tensor->numel();
  }

  const char* tensorimpl_type_name() const override {
    return "TorchyTensor";
  }

  void set_size(int64_t dim, int64_t new_size) override {
    ensure_tensor();
    tensor->set_size(dim, new_size);
  }

  void set_stride(int64_t dim, int64_t new_stride) override {
    ensure_tensor();
    tensor->set_stride(dim, new_stride);
  }

  void set_storage_offset(int64_t storage_offset) override {
    ensure_tensor();
    tensor->set_storage_offset(storage_offset);
  }

  int64_t size(int64_t d) const override {
    ensure_tensor();
    return tensor->size(d);
  }

  int64_t stride(int64_t d) const override {
    ensure_tensor();
    return tensor->stride(d);
  }

  c10::intrusive_ptr<TensorImpl>
  shallow_copy_and_detach(const c10::VariableVersion &version_counter,
                          bool allow_tensor_metadata_change) const override {
    TORCH_CHECK_NOT_IMPLEMENTED(false,
                                "TorchyTensor::shallow_copy_and_detach(1)");
    return {};
  }

  c10::intrusive_ptr<TensorImpl>
  shallow_copy_and_detach(c10::VariableVersion &&version_counter,
                          bool allow_tensor_metadata_change) const override {
    TORCH_CHECK_NOT_IMPLEMENTED(false,
                                "TorchyTensor::shallow_copy_and_detach(2)");
    return {};
  }

  void shallow_copy_from(const c10::intrusive_ptr<TensorImpl> &impl) override {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "TorchyTensor::shallow_copy_from");
  }
};

bool is_torchy(const Tensor &t) {
  return t.key_set().has(DISPATCHKEY);
}

unsigned tape_idx(const Tensor &t) {
  if (is_torchy(t))
    return ((TorchyTensor*)t.unsafeGetTensorImpl())->getTapeIdx();
  return -1u;
}


Tensor abs(const Tensor &self) {
  cout << "Called abs" << endl;
  return at::detail::make_tensor<TorchyTensor>(self.dtype(), self.device());
}

Tensor add_Tensor(const Tensor &self, const Tensor &other,
                  const Scalar &alpha) {
  auto t = at::detail::make_tensor<TorchyTensor>(self.dtype(), self.device());
  ((TorchyTensor*)t.unsafeGetTensorImpl())->registerOp("add", self, other, alpha);
  return t;
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
