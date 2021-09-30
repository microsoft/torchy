#pragma once

// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

#include "config.h"
#include "autogen/ops_data.h"
#include "backends/backends.h"
#include "ops.h"
#include <ATen/core/ivalue.h>
#include <ATen/Tensor.h>
#include <ATen/ThreadLocalState.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/util/variant.h>
#include <array>
#include <cassert>
#include <cstring>
#include <map>
#include <memory>
#include <ostream>
#include <unordered_map>
#include <vector>

#define MAX_TRACE_LENGTH 64

class InputIdx {
  int idx;
public:
  InputIdx(unsigned idx, bool input) : idx(input ? idx : ~idx) {}

  bool is_input() const { return idx >= 0; }
  bool is_trace() const { return idx < 0; }

  unsigned input_idx() const {
    assert(is_input());
    return idx;
  }

  unsigned trace_idx() const {
    assert(!is_input());
    return ~idx;
  }

  bool operator==(const InputIdx &rhs) const { return idx == rhs.idx; }
};

using UnionInputTy = c10::variant<
  bool,
  double,
  int64_t,
  InputIdx,
  at::Device,
  at::Dimname,
  at::MemoryFormat,
  at::ScalarType,
  c10::optional<bool>,
  c10::optional<double>,
  c10::optional<int64_t>,
  c10::optional<InputIdx>,
  c10::optional<at::MemoryFormat>,
  c10::optional<at::Scalar>,
  c10::optional<c10::Device>,
  c10::optional<c10::Layout>,
  c10::optional<c10::ScalarType>,
  c10::optional<std::string>,
  c10::Scalar,
  std::string,
  std::vector<long>,
  std::vector<InputIdx>,
  std::vector<c10::optional<InputIdx>>,
  std::vector<at::Dimname>,
  c10::optional<std::vector<double>>,
  c10::optional<std::vector<long>>,
  c10::optional<std::vector<at::Dimname>>
>;

struct TraceOpDef {
  std::vector<UnionInputTy> args;
  TorchOp id;
  bool observable;
  bool dead;

  bool inplace() const {
    return id >= FIRST_INPLACE_OP;
  }

  bool operator==(const TraceOpDef &rhs) const;

private:
  void destroy();

  friend class Trace;
};


struct TraceOpRunTimeData {
  std::array<uintptr_t, 3> tensors;
  at::ThreadLocalState tls;
  c10::DispatchKeySet dispatch_key;
  uint16_t refs;
  bool inplace;

  bool needsComputing() const {
    return refs > 0;
  }

  uintptr_t someTensor() const;

private:
  void destroy();
  bool hasTensors() const;
  unsigned numTensors() const;

  friend class Trace;
};


struct TraceCacheKey {
  std::unique_ptr<TraceOpDef[]> ops;
  unsigned num_ops;
  // TODO: some backends may want shape information of inputs to specialize code

  TraceCacheKey(TraceOpDef *ops, unsigned num_ops)
    : ops(ops), num_ops(num_ops) {}

  bool operator==(const TraceCacheKey &rhs) const;
};

struct TraceCacheKeyHash {
  size_t operator()(const TraceCacheKey &key) const;
};

struct TraceCacheData {
  void *program = nullptr;
  TorchyBackend *backend = nullptr;

  TraceCacheData(void *program, TorchyBackend *backend)
    : program(program), backend(backend) {}

  TraceCacheData(TraceCacheData &&other) {
    std::swap(program, other.program);
    std::swap(backend, other.backend);
  }

  ~TraceCacheData();
};

using InputData = std::vector<c10::IValue>;

class Trace {
  TraceOpDef          ops[MAX_TRACE_LENGTH];
  TraceOpRunTimeData data[MAX_TRACE_LENGTH];
  InputData inputs;

  std::unordered_map<TraceCacheKey, TraceCacheData, TraceCacheKeyHash> cache;
  unsigned next_op = 0;
  bool flushing = false;
  bool destroyed = false;

  void incref(unsigned idx);
  void decref(unsigned idx);
  InputIdx get_tensor_idx(const at::Tensor &t);
  TraceCacheKey mk_trace_key();

  void print(std::ostream &os, unsigned idx) const;

public:
  ~Trace();

  bool is_flushing() const { return flushing; }
  unsigned numOps() const { return next_op; }
  const TraceOpDef* getOps() const { return ops; }
  TraceOpDef* getOps() { return ops; }
  const TraceOpRunTimeData* getRuntimeData() const { return data; }
  InputData& getInputs() { return inputs; }
  const InputData& getInputs() const { return inputs; }

  template<typename A>
  void append_arg(A &&arg) {
    ops[next_op-1].args.emplace_back(std::forward<A>(arg));
  }

  template<typename T>
  void append_arg(at::ArrayRef<T> arg) {
    ops[next_op-1].args.emplace_back(arg.vec());
  }

  template<typename T>
  void append_arg(c10::optional<at::ArrayRef<T>> arg) {
    c10::optional<std::vector<T>> copy;
    if (arg)
      copy = arg->vec();
    ops[next_op-1].args.emplace_back(std::move(copy));
  }

  void append_arg(const at::Tensor &arg);
  void append_arg(at::Tensor &arg) {
    append_arg(const_cast<const at::Tensor&>(arg));
  }
  void append_arg(at::ArrayRef<at::Tensor> arg);
  void append_arg(c10::optional<at::Tensor> arg);
  void append_arg(const c10::List<c10::optional<at::Tensor>> &arg);
  void append_arg(at::Storage &&arg);
  void append_arg(c10::optional<at::Generator> &&arg);
  void append_arg(c10::string_view arg);
  void append_arg(c10::optional<c10::string_view> arg);

  unsigned register_tensor(uintptr_t tensor, TorchOp op_id,
                           c10::DispatchKeySet ks, unsigned idx_inplace = -1u);

  void add_shared(unsigned idx, uintptr_t ptr);
  void set_unobservable(unsigned idx, uintptr_t ptr);
  void flush(STATS(FlushReason reason));

  friend std::ostream& operator<<(std::ostream &os, const Trace &t);
  friend struct PretendFlushing;

  struct PretendFlushing {
    Trace &t;
    bool old_val;

    PretendFlushing(Trace &t) : t(t) {
      old_val = t.flushing;
      t.flushing = true;
    }

    ~PretendFlushing() {
      t.flushing = old_val;
    }
  };
};
