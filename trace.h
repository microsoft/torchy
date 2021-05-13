#pragma once

// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

#include "config.h"
#include "ops.h"
#include <ATen/Tensor.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/util/variant.h>
#include <array>
#include <cassert>
#include <cstring>
#include <map>
#include <memory>
#include <ostream>
#include <vector>

#define MAX_TRACE_LENGTH 64

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
  // TODO: measure typical amount of sharing
  std::array<uintptr_t, 3> tensors;
  // TODO: investigate if specializing this for the common case
  // e.g. 2 tensors makes sense (would save space + 1 mem alloc)
  std::vector<UnionInputTys> args;
  c10::DispatchKeySet dispatch_key;
  TorchOp id;
  uint16_t refs;
  bool observable;

  bool needsComputing() const {
    return refs > 0;
  }

  void print(std::ostream &os,
             std::map<const at::TensorImpl*, unsigned> &inputs) const;

private:
  void incref();
  void decref(TensorOp *ops);
  bool hasTensors() const;

  friend class Trace;
};


class Trace {
  TensorOp ops[MAX_TRACE_LENGTH];
  unsigned next_op = 0;
  bool flushing = false;
  bool destroyed = false;

  template <typename T>
  void incref(const T &t) {}

  void incref(const at::Tensor &t);
  void incref(const c10::optional<at::Tensor> &t);
  void incref(const at::TensorList &l);
  void incref(const c10::List<c10::optional<at::Tensor>> &l);

  std::vector<std::unique_ptr<unsigned char[]>> deep_copies;

  template<typename T>
  at::ArrayRef<T> deep_copy(const at::ArrayRef<T> &arr) {
    if (arr.empty())
      return arr;
    size_t size = arr.size() * sizeof(T);
    auto ptr = new unsigned char[size];
    memcpy(ptr, arr.data(), size);
    deep_copies.emplace_back(ptr);
    return { (T*)ptr, arr.size() };
  }

public:
  ~Trace();

  bool is_flushing() const { return flushing; }
  unsigned numOps() const { return next_op; }
  TensorOp* getOps() { return ops; }

  template<typename A>
  void append_arg(unsigned idx, A &&arg) {
    incref(arg);
    ops[idx].args.emplace_back(std::forward<A>(arg));
  }

  template<typename T>
  void append_arg(unsigned idx, at::ArrayRef<T> &&arg) {
    incref(arg);
    ops[idx].args.emplace_back(deep_copy(arg));
  }

  template<typename T>
  void append_arg(unsigned idx, c10::optional<at::ArrayRef<T>> &&arg) {
    incref(arg);
    c10::optional<at::ArrayRef<T>> copy;
    if (arg)
      copy = deep_copy(*arg);
    ops[idx].args.emplace_back(std::move(copy));
   }

  template<typename T>
  void append_arg(unsigned idx, const c10::List<T> &arg) {
    incref(arg);
    ops[idx].args.emplace_back(arg.copy());
  }

  unsigned register_tensor(uintptr_t tensor, TorchOp op_id,
                           c10::DispatchKeySet ks);

  void add_shared(unsigned idx, uintptr_t ptr);
  void set_unobservable(unsigned idx, uintptr_t ptr);
  void flush();

  friend std::ostream& operator<<(std::ostream &os, const Trace &t);
};
