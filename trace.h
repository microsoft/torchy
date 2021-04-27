#pragma once

// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

#include <ATen/Tensor.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/util/variant.h>
#include <cassert>
#include <cstring>
#include <map>
#include <memory>
#include <vector>

#define MAX_TRACE_LENGTH 64

class TorchyTensor;

using UnionInputTys = c10::variant<
  bool,
  double,
  int64_t,
  at::Device,
  at::Dimname,
  at::DimnameList,
  at::ScalarType,
  at::Storage,
  at::Tensor,
  at::TensorList,
  c10::IntArrayRef,
  c10::List<c10::optional<at::Tensor>>,
  c10::optional<bool>,
  c10::optional<double>,
  c10::optional<int64_t>,
  c10::optional<at::ArrayRef<double>>,
  c10::optional<at::DimnameList>,
  c10::optional<at::Generator>,
  c10::optional<at::IntArrayRef>,
  c10::optional<at::MemoryFormat>,
  c10::optional<at::Scalar>,
  c10::optional<at::Tensor>,
  c10::optional<c10::Device>,
  c10::optional<c10::Layout>,
  c10::optional<c10::ScalarType>,
  c10::optional<std::string>,
  c10::Scalar,
  std::string
>;

struct TensorOp {
  TorchyTensor *tensor;
  std::vector<UnionInputTys> args;
  c10::DispatchKeySet dispatch_key;
  unsigned id;
  unsigned refs;

  void incref() {
    assert(isObservable());
    ++refs;
  }

  void decref(TensorOp *ops);

  bool isObservable() const {
    assert(!tensor || refs > 0);
    return tensor;
  }

  bool needsComputing() const {
    return refs > 0;
  }

  void print(std::ostream &os,
             std::map<const at::TensorImpl*, unsigned> &inputs) const;
};


class Trace {
  TensorOp ops[MAX_TRACE_LENGTH];
  unsigned next_op = 0;
  bool flushing = false;

  template <typename T>
  void incref(T t) {}

  void incref(const at::Tensor &t);
  void incref(const c10::optional<at::Tensor> &t);
  void incref(const at::TensorList &l);
  void incref(const c10::List<c10::optional<at::Tensor>> &l);

  std::vector<std::unique_ptr<unsigned char[]>> deep_copies;

  template<typename T>
  at::ArrayRef<T> deep_copy(at::ArrayRef<T> arr) {
    if (arr.empty())
      return arr;
    size_t size = arr.size() * sizeof(T);
    auto ptr = new unsigned char[size];
    memcpy(ptr, arr.data(), size);
    deep_copies.emplace_back(ptr);
    return { (T*)ptr, arr.size() };
  }

  template<typename A>
  void registerOpArg(TensorOp &op, A arg) {
    op.args.emplace_back(std::move(arg));
  }

  template<typename T>
  void registerOpArg(TensorOp &op, const at::ArrayRef<T> &arg) {
    op.args.emplace_back(deep_copy(arg));
  }

  template<typename T>
  void registerOpArg(TensorOp &op, const c10::optional<at::ArrayRef<T>> &arg) {
    c10::optional<at::ArrayRef<T>> copy;
    if (arg)
      copy = deep_copy(*arg);
    op.args.emplace_back(std::move(copy));
   }

  template<typename T>
  void registerOpArg(TensorOp &op, const c10::List<T> &arg) {
    op.args.emplace_back(arg.copy());
  }

  template<typename A, typename... T>
  void registerOpArgs(TensorOp &op, const A &arg, T&... args) {
    registerOpArg(op, arg);
    incref(arg);
    registerOpArgs(op, args...);
  }

  void registerOpArgs(TensorOp &op) {}

public:
  bool is_flushing() const { return flushing; }
  unsigned numOps() const { return next_op; }
  TensorOp* getOps() { return ops; }

  template<typename... T>
  unsigned register_tensor(TorchyTensor *tensor, unsigned op_id,
                           c10::DispatchKeySet ks, T&... args) {
    assert(!flushing);
    if (next_op == MAX_TRACE_LENGTH)
      flush();

    auto &op = ops[next_op];
    op.tensor = tensor;
    op.id = op_id;
    op.args.clear();
    registerOpArgs(op, args...);
    op.refs = 1;
    op.dispatch_key = ks;
    return next_op++;
  }

  void set_unobservable(unsigned idx) {
    auto &op = ops[idx];
    assert(op.tensor);
    op.tensor = nullptr;
    op.decref(ops);

    // reclaim slot if this was the last created tensor
    if (op.refs == 0 && idx+1 == next_op) {
      --next_op;
    }
  }

  void flush();

  friend std::ostream& operator<<(std::ostream &os, const Trace &t);
};
