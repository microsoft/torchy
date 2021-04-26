// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

#include "trace.h"
#include <cstring>

using namespace at;
using namespace std;

namespace interpreter { void run(Trace &t); }

void TensorOp::decref(TensorOp *ops) {
  assert(refs > 0);
  --refs;

  if (refs == 0) {
    for (auto &arg : args) {
      if (auto t = get_if<Tensor>(&arg)) {
        auto idx = trace_idx(*t);
        if (idx != -1u)
          ops[idx].decref(ops);
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
      } else if (auto s = get_if<Scalar>(&arg)) {
        os << *s;
      } else if (auto a = get_if<IntArrayRef>(&arg)) {
        os << *a;
      } else if (auto i = get_if<int64_t>(&arg)) {
        os << *i;
      } else if (auto o = get_if<c10::optional<int64_t>>(&arg)) {
        if (*o)
          os << **o;
        else
          os << "(null)";
      } else if (auto s = get_if<c10::optional<ScalarType>>(&arg)) {
        if (*s)
          os << **s;
        else
          os << "(null)";
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

IntArrayRef Trace::deep_copy(IntArrayRef arr) {
  size_t size = arr.size() * sizeof(int64_t);
  auto ptr = new unsigned char[size];
  memcpy(ptr, arr.data(), size);
  deep_copies.emplace_back(ptr);
  return { (int64_t*)ptr, arr.size() };
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
