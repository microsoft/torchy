// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

#include "trace.h"
#include "stopwatch.h"
#include "tensor.h"
#include "utils.h"
#include <ATen/core/Formatting.h>
#include <ATen/core/List.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <unordered_map>

using namespace at;
using namespace std;

namespace interpreter { void run(Trace &t); }
namespace torchscript { bool run(Trace &t); }

void TensorOp::destroy() {
  args.clear();
  tls.~ThreadLocalState();
}

void TensorOp::incref() {
  assert(observable);
  ++refs;
  assert(refs != 0);
}

void TensorOp::decref(TensorOp *ops) {
  assert(refs > 0);
  --refs;

  if (refs == 0) {
    TensorVisitor visitor([&](const Tensor &t) {
      auto idx = trace_idx(t);
      if (idx != -1u)
        ops[idx].decref(ops);
    });
    for (auto &arg : args) {
      visit(visitor, arg);
    }
    assert(!observable && !hasTensors());
    destroy();
  }
}

bool TensorOp::hasTensors() const {
  return find_if(tensors.begin(), tensors.end(), [](auto t) { return t != 0; })
           != tensors.end();
}

unsigned TensorOp::numTensors() const {
  return count_if(tensors.begin(), tensors.end(), [](auto t) { return t!=0; });
}

uintptr_t TensorOp::someTensor() const {
  auto I = find_if(tensors.begin(), tensors.end(), [](auto t) { return t!=0; });
  return I == tensors.end() ? 0 : *I;
}

bool TensorOp::operator!=(const Tensor &t) const {
  auto *t_ptr = t.getIntrusivePtr().get();
  for (auto ptr : tensors) {
    if (ptr == (uintptr_t)t_ptr)
      return false;
  }
  return true;
}

namespace {
using InputMap = map<const TensorImpl*, unsigned>;

class printer {
  ostream &os;
  InputMap &inputs;
  const TensorOp &op;

public:
  printer(ostream &os, InputMap &inputs, const TensorOp &op)
    : os(os), inputs(inputs), op(op) {}

  template<typename T>
  ostream& operator()(const T &a) {
    return os << a;
  }

  ostream& operator()(const Tensor &t) {
    auto idx = trace_idx(t);
    if (idx != -1u && op != t)
      return os << '%' << idx;

    auto n = inputs.emplace(t.getIntrusivePtr().get(),
                            (unsigned)inputs.size()).first->second;
    return os << "in<" << n << '>';
  }

  template<typename T>
  ostream& operator()(const optional<T> &a) {
    if (!a)
      return os << "(null)";
    return (*this)(*a);
  }

  template<typename T>
  ostream& operator()(const vector<T> &l) {
    os << '[';
    bool first = true;
    for (const auto &elem : l) {
      if (!first) os << ", ";
      first = false;
      (*this)(elem);
    }
    return os << ']';
  }

  template<typename T>
  ostream& operator()(const List<T> &l) {
    os << '(';
    bool first = true;
    for (const auto &it : l) {
      if (!first) os << ", ";
      first = false;

      const T &elem = it;
      (*this)(elem);
    }
    return os << ')';
  }

  ostream& operator()(const Storage &s) {
    if (!s)
      return os << "storage(null)";
    return os << "storage(" << s.nbytes() << ')';
  }

  ostream& operator()(const Generator &g) {
    if (!g.defined())
      return os << "generator(null)";
    return os << "generator(" << g.current_seed() << ", " << g.device() << ')';
  }
};
}

void TensorOp::print(ostream &os, InputMap &inputs) const {
  auto t = someTensor();
  if (t && tensor_has_dtype(t))
    os << '<' << tensor_get_dtype(t) << "> ";
  os << id;

  if (!needsComputing()) {
    os << " [dead]";
    return;
  }

  bool first = true;
  for (auto &arg : args) {
    os << (first ? " " : ", ");
    first = false;

    visit(printer(os, inputs, *this), arg);
  }

  auto n_tensors = numTensors();
  assert(n_tensors >= observable);
  assert(refs >= n_tensors);

  if (refs > 0)
    os << " #refs E/I=" << n_tensors << '/' << (refs - n_tensors);

  if (observable)
    os << " #output";

  if (t && tensor_has_shape(t))
    os << " shape=" << tensor_get_shape(t);
}


Trace::~Trace() {
  destroyed = true;
}

void Trace::incref(const Tensor &t) {
  auto idx = trace_idx(t);
  if (idx != -1u) {
    assert(idx < next_op);
    ops[idx].incref();
  }
}

void Trace::incref(const optional<Tensor> &t) {
  if (t)
    incref(*t);
}

void Trace::incref(const TensorList &l) {
  for (auto &t : l) {
    incref(t);
  }
}

void Trace::incref(const List<optional<Tensor>> &l) {
  for (const auto &t : l) {
    const optional<Tensor> &opt = t;
    incref(opt);
  }
}

unsigned Trace::register_tensor(uintptr_t tensor, TorchOp op_id,
                                c10::DispatchKeySet ks) {
  assert(!flushing);
#ifndef TORCHY_RELEASE
  // FOR DEBUGGING ONLY. Can be used to binary search a trace that goes wrong
  static unsigned call_count = 0;
  ++call_count;
  if (auto *limit = getenv("TORCHY_FLUSH_BEFORE")) {
    if (call_count <= (unsigned)atoi(limit))
      flush(STATS(FlushReason::DEBUG));
  }
  if (auto *limit = getenv("TORCHY_FLUSH_AFTER")) {
    if (call_count > (unsigned)atoi(limit))
      flush(STATS(FlushReason::DEBUG));
  }
  if (auto *limit = getenv("TORCHY_MAX_TRACE_LENGTH")) {
    if (next_op == (unsigned)atoi(limit))
      flush(STATS(FlushReason::TRACE_MAX_LENGTH));
  }
#endif

  if (next_op == MAX_TRACE_LENGTH)
    flush(STATS(FlushReason::TRACE_MAX_LENGTH));

  auto &op = ops[next_op];
  op.tensors[0] = tensor;
  for (unsigned i = 1; i < op.tensors.size(); ++i) {
    op.tensors[i] = 0;
  }
  op.id = op_id;
  assert(op.args.empty());
  op.refs = 1;
  op.observable = true;
  op.tls = ThreadLocalState();
  op.dispatch_key = ks;
  return next_op++;
}

void Trace::add_shared(unsigned idx, uintptr_t ptr) {
  assert(idx < next_op);
  auto &op = ops[idx];

  for (auto &tensor : op.tensors) {
    if (tensor == 0) {
      tensor = ptr;
      op.incref();
      return;
    }
  }

  // no more space for additional observers; just flush
  flush(STATS(FlushReason::OVERFLOW_SHARED_LIST));
}

void Trace::set_unobservable(unsigned idx, uintptr_t ptr) {
  // technically this accesses memory that has been deallocated already
  // but since it's a global and it's just a bool, whatever..
  if (destroyed)
    return;

  assert(idx < next_op);
  auto &op = ops[idx];

  bool found = false;
  for (auto &tensor : op.tensors) {
    if (tensor == ptr) {
      tensor = 0;
      found = true;
      break;
    }
  }
  assert(found); (void)found;

  op.observable = op.hasTensors();
  op.decref(ops);

  // reclaim slot if this was the last created tensor
  if (op.refs == 0 && idx+1 == next_op) {
    --next_op;
  }
}

void Trace::flush(STATS(FlushReason reason)) {
  assert(!flushing);
  flushing = true;

  // trim set of observable tensors as the references in arguments keep the
  // tensors alive and therefore we aren't notified the user's program
  // can't observe these tensors anymore
  // TODO: benchmark: should we skip this when running the interpreter that
  // doesn't really benefit from this information?
  {
    // tensor impl -> (refs, trace idx)
    unordered_map<uintptr_t, pair<uint16_t, uint16_t>> refs;
    refs.reserve(next_op);

    TensorVisitor visitor([&](const Tensor &t) {
      auto I = refs.find((uintptr_t)t.getIntrusivePtr().get());
      // all refs are inputs -> not observable
      if (I != refs.end() && --I->second.first == 0) {
        refs.erase(I);
        auto &argop = ops[I->second.second];
        argop.observable = false;
        argop.decref(ops);
      }
    });

    for (unsigned i = 0; i < next_op; ++i) {
      auto &op = ops[i];

      if (i > 0) {
        for (auto &arg : op.args) {
          visit(visitor, arg);
        }
      }

      if (op.observable) {
        for (auto tensor : op.tensors) {
          if (tensor == 0 || tensor == DUMMY_TORCHY)
            continue;
          refs.emplace(tensor,
            // -1 as reclaim adds +1 ref
            make_pair(intrusive_ptr<TensorImpl>::unsafe_reclaim_from_nonowning(
                        (TensorImpl*)tensor).use_count()-1, i));
        }
      }
    }
  }

  stats_register_trace(*this, reason);

#ifdef TORCHY_PRINT_TRACE_ON_FLUSH
  cerr << "Flush trace\n" << *this << endl;
#endif

  STATS(StopWatch run_time);
  // try torchscript first; fallback to the interpreter if it can't handle this
  if (!torchscript::run(*this))
    interpreter::run(*this);

  STATS(run_time.stop());

  stats_register_trace_time(run_time);

  // reduce reference count on tensors s.t. they are deleted if possible
  for (unsigned i = 0; i < next_op; ++i) {
    ops[i].destroy();
  }

  next_op = 0;
  flushing = false;
}

ostream& operator<<(ostream &os, const Trace &t) {
  if (t.next_op == 0)
    return os << "(empty)\n";

  map<const TensorImpl*, unsigned> inputs_map;
  for (unsigned i = 0; i < t.next_op; ++i) {
    os << '%' << i << " = ";
    t.ops[i].print(os, inputs_map);
    os << '\n';
  }

  if (inputs_map.empty())
    return os;

  os << "\nInputs' shapes:\n";
  for (unsigned i = 0, e = inputs_map.size(); i != e; ++i) {
    auto I = find_if(inputs_map.begin(), inputs_map.end(),
                     [=](auto &p) { return p.second == i; });
    assert(I != inputs_map.end());
    os << "in<" << i << ">: " << I->first->sizes() << '\n';
  }
  return os;
}
