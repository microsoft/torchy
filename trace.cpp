// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

#include "trace.h"
#include "tensor.h"
#include <ATen/core/Formatting.h>
#include <ATen/core/List.h>

using namespace at;
using namespace std;

namespace interpreter { void run(Trace &t); }

static void decref(TensorOp *ops, const Tensor &t) {
  auto idx = trace_idx(t);
  if (idx != -1u)
    ops[idx].decref(ops);
}

void TensorOp::decref(TensorOp *ops) {
  assert(refs > 0);
  --refs;

  if (refs == 0) {
    for (auto &arg : args) {
      if (auto t = get_if<Tensor>(&arg)) {
        ::decref(ops, *t);
      } else if (auto t = get_if<optional<Tensor>>(&arg)) {
        if (*t)
          ::decref(ops, **t);
      } else if (auto l = get_if<TensorList>(&arg)) {
        for (auto &t : *l) {
          ::decref(ops, t);
        }
      } else if (auto l = get_if<List<optional<Tensor>>>(&arg)) {
        for (const auto &t : *l) {
          const optional<Tensor> &opt = t;
          if (opt)
            ::decref(ops, *opt);
        }
      }
    }
  }
}

void TensorOp::print(ostream &os,
                     map<const TensorImpl*, unsigned> &inputs) const {
    if (!needsComputing()) {
      os << "[dead]";
      return;
    }

    os << id;
    bool first = true;
    for (auto &arg : args) {
      os << (first ? " " : ", ");
      first = false;
      if (auto t = get_if<Tensor>(&arg)) {
        auto idx = trace_idx(*t);
        if (idx != -1u) {
          os << '%' << idx;
        } else {
          auto n = inputs.emplace(t->unsafeGetTensorImpl(),
                                  (unsigned)inputs.size()).first->second;
          os << "in<" << n << '>';
        }

#define OPTIONAL(type)                                     \
      } else if (auto a = get_if<optional<type>>(&arg)) {  \
        if (*a) { os << **a; } else { os << "(null)"; }

      OPTIONAL(bool)
      OPTIONAL(double)
      OPTIONAL(int64_t)
      OPTIONAL(Scalar)
      OPTIONAL(ScalarType)
      OPTIONAL(string)

      } else if (auto a = get_if<IntArrayRef>(&arg)) {  os << *a;
      } else if (auto a = get_if<Scalar>(&arg)) {       os << *a;
      } else if (auto a = get_if<bool>(&arg)) {         os << *a;
      } else if (auto a = get_if<double>(&arg)) {       os << *a;
      } else if (auto a = get_if<int64_t>(&arg)) {      os << *a;
      } else if (auto a = get_if<string>(&arg)) {       os << *a;
      } else {
        assert(0 && "missing case in TensorOp::print");
      }
    }

  if (refs > 1)
    os << " #refs=" << (refs-1);

  if (isObservable())
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

void Trace::flush() {
#if 1
  cout << "Flush trace\n" << *this;
#endif

  assert(!flushing);
  flushing = true;

  interpreter::run(*this);

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
