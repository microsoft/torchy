// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

#pragma once

#include <ATen/Tensor.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/util/variant.h>
#include <cassert>
#include <map>
#include <memory>
#include <vector>

#define MAX_TRACE_LENGTH 64

class TorchyTensor;

using UnionInputTys = c10::variant<
  int64_t,
  c10::IntArrayRef,
  c10::optional<int64_t>,
  c10::optional<c10::ScalarType>,
  c10::Scalar,
  at::Tensor
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

  std::vector<std::unique_ptr<unsigned char[]>> deep_copies;
  c10::IntArrayRef deep_copy(c10::IntArrayRef arr);

  template<typename A>
  void registerOpArg(TensorOp &op, A arg) {
    op.args.emplace_back(std::move(arg));
  }

  void registerOpArg(TensorOp &op, c10::IntArrayRef arg) {
    op.args.emplace_back(deep_copy(arg));
  }

  template<typename A, typename... T>
  void registerOpArgs(TensorOp &op, const A &arg, T&... args) {
    registerOpArg(op, arg);
    incref(arg);
    registerOpArgs(op, args...);
  }

  void registerOpArgs(TensorOp &op) {}

public:
  bool is_flushing() const {
    return flushing;
  }

  void set_flushing(bool val) {
    flushing = val;
  }

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
