// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

#include "dispatch.h"
#include "tensor.h"
#include "trace.h"
#include <ATen/core/List.h>
#include <ATen/RedispatchFunctions.h>
#include <type_traits>

//#define DEBUG_DISPATCH

#ifdef DEBUG_DISPATCH
# include <iostream>
#endif

using namespace at;

static void init_update_in_place(TensorOp &op) {
  for (auto tensor : op.tensors) {
    if (tensor != 0)
      init_update_in_place(tensor);
  }
}

static void end_update_in_place(TensorOp &op) {
  for (auto tensor : op.tensors) {
    if (tensor != 0)
      end_update_in_place(tensor);
  }
}

#ifndef NDEBUG
static void finish_trace(TensorOp &op) {
  for (auto tensor : op.tensors) {
    if (tensor != 0)
      finish_trace(tensor);
  }
}
#endif

static void set(TensorOp &op, const Tensor &t) {
  for (auto tensor : op.tensors) {
    if (tensor != 0)
      set(tensor, t);
  }
}

namespace {

template <typename T>
struct load {
  T operator()(UnionInputTy &arg) {
    return std::move(get<T>(arg));
  }
};

template <typename T>
struct load<T&> {
  T& operator()(UnionInputTy &arg) {
    return get<T>(arg);
  }
};

template <typename T>
struct load<ArrayRef<T>> {
  ArrayRef<T> operator()(UnionInputTy &arg) {
    return get<std::vector<T>>(arg);
  }
};

template <typename T>
struct load<c10::optional<ArrayRef<T>>> {
  c10::optional<ArrayRef<T>> operator()(UnionInputTy &arg) {
    auto &opt = get<c10::optional<std::vector<T>>>(arg);
    if (!opt)
      return c10::nullopt;
    return *opt;
  }
};

template <>
c10::string_view load<c10::string_view>::operator()(UnionInputTy &arg) {
  return get<std::string>(arg);
}

template <>
c10::optional<c10::string_view>
load<c10::optional<c10::string_view>>::operator()(UnionInputTy &arg) {
  auto &opt = get<c10::optional<std::string>>(arg);
  if (!opt)
    return c10::nullopt;
  return *opt;
}

#include "autogen/interpreter_redispatch_tables.h"

struct DispatchKeyComputer {
  c10::DispatchKeySet ks;

  DispatchKeyComputer(c10::DispatchKeySet ks) : ks(ks) {}

  template <typename T>
  void operator()(const T&) {}

  void operator()(const at::Tensor &t) {
    ks = ks | t.key_set();
  }

  void operator()(const at::Generator &gen) {
    if (gen.defined())
      ks = ks | gen.key_set();
  }

  template<typename T>
  void operator()(const c10::optional<T> &opt) {
    if (opt)
      (*this)(*opt);
  }

  template<typename T>
  void operator()(const std::vector<T> &l) {
    for (const auto &elem : l) {
      (*this)(elem);
    }
  }

  template<typename T>
  void operator()(const at::List<T> &l) {
    for (const auto &it : l) {
      const T &elem = it;
      (*this)(elem);
    }
  }
};

}


namespace interpreter {

void run(Trace &t) {
  auto *ops = t.getOps();

  for (unsigned i = 0, e = t.numOps(); i < e; ++i) {
    auto &op = ops[i];
    if (!op.needsComputing())
      continue;

#ifdef DEBUG_DISPATCH
    std::cerr << "Dispatch " << op.id << std::endl;
#endif

    DispatchKeyComputer visitor(op.dispatch_key);
    for (auto &arg : op.args) {
      visit(visitor, arg);
    }
    auto ks
      = visitor.ks & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);

    ThreadLocalState::setThreadLocalState(op.tls);

    if (op.id >= FIRST_INPLACE_OP)
      init_update_in_place(op);

    switch (op.id) {

#include "autogen/interpreter_redispatch.h"

      default:
        assert(0 && "Unhandled op");
    }

    // generated redispatch code only reaches here for in-place ops
    end_update_in_place(op);
  }

#ifndef NDEBUG
  for (unsigned i = 0, e = t.numOps(); i < e; ++i) {
    finish_trace(ops[i]);
  }
#endif

}

}
