// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

#include "dispatch.h"
#include "tensor.h"
#include "trace.h"
#include <ATen/core/List.h>
#include <ATen/RedispatchFunctions.h>
#include <type_traits>

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

}


namespace interpreter {

void run(Trace &t) {
  auto *ops = t.getOps();

  for (unsigned i = 0, e = t.numOps(); i < e; ++i) {
    auto &op = ops[i];
    if (!op.needsComputing())
      continue;

    auto ks = op.dispatch_key;
    for (auto &arg : op.args) {
      if (auto t = get_if<Tensor>(&arg)) {
        ks = ks | t->key_set();
      }
    }
    ks = ks & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);

    ThreadLocalState::setThreadLocalState(op.tls);

    switch (op.id) {

#include "autogen/interpreter_redispatch.h"

      default:
        assert(0 && "Unhandled op");
    }

    // generated redispatch code only reached here for in-place ops
    end_update_in_place(op);
  }
}

}
