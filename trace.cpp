// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

#include "trace.h"
#include "tensor.h"
#include "utils.h"
#include <ATen/core/Formatting.h>
#include <ATen/core/List.h>
#include <unordered_map>

using namespace at;
using namespace std;

namespace interpreter { void run(Trace &t); }

void TensorOp::incref() {
  assert(observable);
  ++refs;
}

void TensorOp::decref(TensorOp *ops) {
  assert(refs > 0);
  --refs;

  if (refs == 0) {
    for (auto &arg : args) {
      visit(TensorVisitor([&](const Tensor &t) {
        auto idx = trace_idx(t);
        if (idx != -1u)
          ops[idx].decref(ops);
      }), arg);
    }
    args.clear();
    assert(!tensor);
  }
}

namespace {
using InputMap = map<const TensorImpl*, unsigned>;

class printer {
  ostream &os;
  InputMap &inputs;

public:
  printer(ostream &os, InputMap &inputs) : os(os), inputs(inputs) {}

  template<typename T>
  ostream& operator()(const T &a) {
    return os << a;
  }

  ostream& operator()(const Tensor &t) {
    auto idx = trace_idx(t);
    if (idx != -1u)
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
  ostream& operator()(const ArrayRef<T> &l) {
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

  ostream& operator()(const Generator &g) {
    if (!g.defined())
      return os << "generator(null)";
    return os << "generator(" << g.current_seed() << ", " << g.device() << ")";
  }
};
}

void TensorOp::print(ostream &os, InputMap &inputs) const {
    if (!needsComputing()) {
      os << "[dead]";
      return;
    }

    os << id;
    bool first = true;
    for (auto &arg : args) {
      os << (first ? " " : ", ");
      first = false;

      visit(printer(os, inputs), arg);
    }

  if (refs > 1)
    os << " #refs=" << (refs - observable);

  if (observable)
    os << " #output";
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

void Trace::set_unobservable(unsigned idx) {
  assert(idx < next_op);
  auto &op = ops[idx];

  assert(op.tensor);
  op.tensor = 0;
  op.observable = false;
  op.decref(ops);

  // reclaim slot if this was the last created tensor
  if (op.refs == 0 && idx+1 == next_op) {
    --next_op;
  }
}

void Trace::flush() {
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

    for (unsigned i = 0; i < next_op; ++i) {
      auto &op = ops[i];

      if (i > 0) {
        for (auto &arg : op.args) {
          visit(TensorVisitor([&](const Tensor &t) {
            auto I = refs.find((uintptr_t)t.getIntrusivePtr().get());
            // all refs are inputs -> not observable
            if (I != refs.end() && --I->second.first == 0) {
              refs.erase(I);
              ops[I->second.second].observable = false;
            }
          }), arg);
        }
      }

      if (op.observable && op.tensor != DUMMY_TORCHY)
        refs.emplace(
          op.tensor,
          // -1 as reclaim adds +1 ref
          make_pair(intrusive_ptr<TensorImpl>::unsafe_reclaim_from_nonowning(
                      (TensorImpl*)op.tensor).use_count()-1, i));
    }
  }

#ifdef TORCHY_PRINT_TRACE_ON_FLUSH
  cerr << "Flush trace\n" << *this;
#endif

  interpreter::run(*this);

  // reduce reference count on tensors s.t. they are deleted if possible
  for (unsigned i = 0; i < next_op; ++i) {
    ops[i].args.clear();
  }

  next_op = 0;
  flushing = false;
  deep_copies.clear();
}

ostream& operator<<(ostream &os, const Trace &t) {
  if (t.next_op == 0)
    os << "(empty)\n";

  map<const TensorImpl*, unsigned> inputs_map;
  for (unsigned i = 0; i < t.next_op; ++i) {
    os << '%' << i << " = ";
    t.ops[i].print(os, inputs_map);
    os << '\n';
  }
  return os << endl;
}
