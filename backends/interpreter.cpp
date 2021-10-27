// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

#include "common.h"
#include "dispatch.h"
#include <ATen/RedispatchFunctions.h>
#include <type_traits>

//#define DEBUG_DISPATCH

#ifdef DEBUG_DISPATCH
# include <iostream>
#endif

using namespace at;

namespace {

struct LoadState {
  InputData &inputs;
  Tensor *results;
  const TraceOpRunTimeData *data;
  Tensor tmp_tensors[MAX_NUM_INPUTS];
  unsigned next_tensor = 0;

  void reset() {
    next_tensor = 0;
  }
};

#define LOAD_ARGS UnionInputTy &arg, LoadState &load_state

template <typename T>
struct load {
  T operator()(LOAD_ARGS) {
    return std::move(get<T>(arg));
  }
};

template <typename T>
struct load<T&> {
  T& operator()(LOAD_ARGS) {
    return get<T>(arg);
  }
};

template <typename T>
struct load<ArrayRef<T>> {
  ArrayRef<T> operator()(LOAD_ARGS) {
    return get<std::vector<T>>(arg);
  }
};

template <typename T>
struct load<c10::optional<ArrayRef<T>>> {
  c10::optional<ArrayRef<T>> operator()(LOAD_ARGS) {
    auto &opt = get<c10::optional<std::vector<T>>>(arg);
    if (!opt)
      return c10::nullopt;
    return *opt;
  }
};

template <>
struct load<c10::string_view> {
  c10::string_view operator()(LOAD_ARGS) {
    return get<std::string>(arg);
  }
};

template <>
struct load<c10::optional<c10::string_view>> {
  c10::optional<c10::string_view> operator()(LOAD_ARGS) {
    auto &opt = get<c10::optional<std::string>>(arg);
    if (!opt)
      return c10::nullopt;
    return *opt;
  }
};

Tensor& get_tensor(InputIdx idx, LoadState &load_state) {
  if (idx.is_input())
    return load_state.inputs[idx.input_idx()].toTensor();

  auto ptr = (TensorImpl*)load_state.data[idx.trace_idx()].someTensor();
  // tensor not observable; so use intermediate value
  if (ptr == nullptr)
    return load_state.results[idx.trace_idx()];

  assert(load_state.next_tensor < MAX_NUM_INPUTS);
  return load_state.tmp_tensors[load_state.next_tensor++] =
          Tensor(intrusive_ptr<TensorImpl>::unsafe_reclaim_from_nonowning(ptr));
}

template <>
struct load<Tensor&> {
  Tensor& operator()(LOAD_ARGS) {
    return get_tensor(get<InputIdx>(arg), load_state);
  }
};

template <>
struct load<optional<Tensor>&> {
  optional<Tensor> operator()(LOAD_ARGS) {
    auto idx = get<optional<InputIdx>>(arg);
    if (!idx)
      return c10::nullopt;
    return get_tensor(*idx, load_state);
  }
};

template <>
struct load<TensorList> {
  std::vector<Tensor> operator()(LOAD_ARGS) {
    std::vector<Tensor> vect;
    for (auto idx : get<std::vector<InputIdx>>(arg)) {
      vect.emplace_back(get_tensor(idx, load_state));
    }
    return vect;
  }
};

template <>
struct load<c10::List<c10::optional<Tensor>>&> {
  c10::List<c10::optional<Tensor>> operator()(LOAD_ARGS) {
    c10::List<c10::optional<Tensor>> lst;
    for (auto idx : get<std::vector<c10::optional<InputIdx>>>(arg)) {
      lst.push_back(
        idx ? make_optional(get_tensor(*idx, load_state)) : c10::nullopt);
    }
    return lst;
  }
};

template <>
struct load<Storage> {
  Storage operator()(LOAD_ARGS) {
    // Storage input is never shared; can be moved
    return
      std::move(load_state.inputs[get<InputIdx>(arg).input_idx()]).toStorage();
  }
};

template <>
struct load<c10::optional<Generator>> {
  c10::optional<Generator> operator()(LOAD_ARGS) {
    auto idx = get<c10::optional<InputIdx>>(arg);
    if (!idx)
      return c10::nullopt;
    return load_state.inputs[idx->input_idx()].toGenerator();
  }
};

#include "autogen/interpreter_redispatch_tables.h"

struct DispatchKeyComputer {
  c10::DispatchKeySet ks;
  LoadState &load_state;

  template <typename T>
  void operator()(const T&) {}

  void operator()(InputIdx idx) {
    if (idx.is_input()) {
      auto &arg = load_state.inputs[idx.input_idx()];
      if (arg.isTensor()) {
        ks = ks | arg.toTensor().key_set();
      } else if (arg.isGenerator()) {
        const auto &gen = arg.toGenerator();
        if (gen.defined())
          ks = ks | gen.key_set();
      } else {
        assert(arg.isStorage());
      }
    } else {
      ks = ks | get_tensor(idx, load_state).key_set();
    }
  }

  template<typename T>
  void operator()(const c10::optional<T> &a) {
    if (a)
      (*this)(*a);
  }

  template<typename T>
  void operator()(const std::vector<T> &l) {
    for (const auto &elem : l) {
      (*this)(elem);
    }
  }
};

}


void Interpreter::run(const void *prog, Trace &t) {
  Tensor results[MAX_TRACE_LENGTH];
  LoadState load_state{t.getInputs(), results, t.getRuntimeData()};

  ThreadLocalState tls;
  auto *ops = t.getOps();
  auto *data = t.getRuntimeData();

  for (unsigned i = 0, e = t.numOps(); i < e; ++i) {
    auto &op = ops[i];
    auto &rdata = data[i];
    if (op.dead)
      continue;

#ifdef DEBUG_DISPATCH
    std::cerr << "Dispatch " << op.id << std::endl;
#endif

    DispatchKeyComputer visitor{rdata.dispatch_key, load_state};
    for (auto &arg : op.args) {
      visit(visitor, arg);
    }
    auto ks
      = visitor.ks & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    load_state.reset();

    ThreadLocalState::setThreadLocalState(*(ThreadLocalState*)&rdata.tls);

    switch (op.id) {

#include "autogen/interpreter_redispatch.h"

      default:
        assert(0 && "Unhandled op");
    }

    set(rdata, results[i]);
    load_state.reset();
  }

  for (unsigned i = 0, e = t.numOps(); i < e; ++i) {
    finish_trace(data[i]);
  }

  ThreadLocalState::setThreadLocalState(tls);
}
