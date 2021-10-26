// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

#include "trace.h"
#include "backends/backends.h"
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

static Interpreter interpreter;
static TorchScript torchscript;
static TorchyBackend *backend
  = getenv("TORCHY_FORCE_INTERPRETER")
      ? (TorchyBackend*)&interpreter : (TorchyBackend*)&torchscript;

bool TraceOpDef::operator==(const TraceOpDef &rhs) const {
  return id == rhs.id && observable == rhs.observable && dead == rhs.dead &&
         args == rhs.args;
}

void TraceOpDef::destroy() {
  args.clear();
}

void TraceOpRunTimeData::destroy() {
  tls.~ThreadLocalState();
}

void Trace::incref(unsigned idx) {
  assert(idx < next_op);
  auto &op    = ops[idx];
  auto &rdata = data[idx];
  (void)op;
  assert(op.observable && !op.dead && rdata.refs > 0);
  ++rdata.refs;
  assert(rdata.refs != 0);
}

void Trace::decref(unsigned idx) {
  assert(idx < next_op);
  auto &op    = ops[idx];
  auto &rdata = data[idx];
  assert(!op.dead && rdata.refs > 0);

  // We can't declare ops dead for sure because of aliasing. A non-inplace op
  // may return a new tensor, but sharing storage with another tensor
  // (e.g., view, reshape).
  // Just because the result of these ops is dead it doesn't mean there isn't
  // another tensor out there with the same storage.
  // See tests/unit/inplace_dead_alias.py
  //--rdata.refs;

  if (rdata.refs == 0) {
    TensorVisitor visitor([&](InputIdx idx) {
      if (idx.is_trace())
        decref(idx.trace_idx());
    }, inputs);
    for (auto &arg : op.args) {
      visit(visitor, arg);
    }
    assert(!op.observable && !rdata.hasTensors());
    op.dead = true;
    op.destroy();
    rdata.destroy();
  }
}

bool TraceOpRunTimeData::hasTensors() const {
  return find_if(tensors.begin(), tensors.end(), [](auto t) { return t != 0; })
           != tensors.end();
}

unsigned TraceOpRunTimeData::numTensors() const {
  return count_if(tensors.begin(), tensors.end(), [](auto t) { return t!=0; });
}

uintptr_t TraceOpRunTimeData::someTensor() const {
  auto I = find_if(tensors.begin(), tensors.end(),
                   [](auto t) { return t != 0 && t != DUMMY_TORCHY; });
  return I == tensors.end() ? 0 : *I;
}

bool TraceCacheKey::operator==(const TraceCacheKey &rhs) const {
  return num_ops == rhs.num_ops &&
         equal(ops.get(), &ops[num_ops], rhs.ops.get());
}

size_t TraceCacheKeyHash::operator()(const TraceCacheKey &key) const {
  size_t hash = 0;
  // we only hash the prefix of the trace ops
  // this enables us to discover quickly traces that need deoptimization
  // and prefix of traces for speculative execution
  unsigned bits = 11;
  unsigned max_rounds = (sizeof(size_t) * 8) / bits;
  for (unsigned i = 0; i < min(key.num_ops, max_rounds); ++i) {
    hash = (hash << bits) | key.ops[i].id;
  }
  return hash;
}

TraceCacheData::~TraceCacheData() {
  if (backend)
    backend->destroy(program);
}


namespace {

class printer {
  ostream &os;

public:
  printer(ostream &os) : os(os) {}

  template<typename T>
  ostream& operator()(const T &a) {
    return os << a;
  }

  ostream& operator()(const InputIdx &idx) {
    if (idx.is_input())
      return os << "in<" << idx.input_idx() << '>';
    return os << '%' << idx.trace_idx();
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
};
}

static void print_op(ostream &os, unsigned idx, const TraceOpDef &op,
                     const TraceOpRunTimeData *rdata) {
  os << '%' << idx << " = ";
  auto t = rdata ? rdata->someTensor() : 0;
  if (t && tensor_has_dtype(t))
    os << '<' << tensor_get_dtype(t) << "> ";
  os << op.id;

  if (op.dead) {
    os << " [dead]\n";
    return;
  }

  bool first = true;
  for (auto &arg : op.args) {
    os << (first ? " " : ", ");
    first = false;
    visit(printer(os), arg);
  }

  if (rdata) {
    auto n_tensors = rdata->numTensors();
    assert(n_tensors >= op.observable);
    assert(rdata->refs >= n_tensors);

    if (rdata->refs > 0)
      os << " #refs E/I=" << n_tensors << '/' << (rdata->refs - n_tensors);
  }

  if (op.observable)
    os << " #output";

  if (t && tensor_has_shape(t))
    os << " shape=" << tensor_get_shape(t);

  os << '\n';
}

void Trace::print(ostream &os, unsigned idx) const {
  assert(idx < next_op);
  print_op(os, idx, ops[idx], &data[idx]);
}

Trace::~Trace() {
  destroyed = true;
#if 0
  cerr << "NUM BUCKETS: " << cache.bucket_count() << '\n';
  unsigned collisions = 0;
  for (unsigned i = 0; i < cache.bucket_count(); ++i) {
    auto sz = cache.bucket_size(i);
    if (sz <= 1)
      continue;

    cerr << i << ": " << sz << '\n';
    collisions += sz;

    for (auto &p : cache) {
      auto &k = p.first;
      if (cache.bucket(k) == i) {
        cerr << "HASH: " << TraceCacheKeyHash()(k) << '\n';
        for (unsigned i = 0; i < k.num_ops; ++i) {
          print_op(cerr, i, k.ops[i], nullptr);
        }
      }
    }
  }
  cerr << "TOTAL COLLISIONS = " << collisions << endl;
#endif
}

bool Trace::is_input(const c10::TensorImpl &t) const {
  for (auto &in : inputs) {
    if (in.isTensor() && in.toTensor().unsafeGetTensorImpl() == &t)
      return true;
  }
  return false;
}

InputIdx Trace::get_tensor_idx(const Tensor &t) {
  auto idx = trace_idx(t);
  if (idx != -1u) {
    assert(idx < next_op);
    // check if this is also an input tensor
    // e.g. for inplace ops we have %0 = op %0 <-- but we want in<0> there
    if (idx != next_op-1) {
      incref(idx);
      return { idx, false };
    }
  }

  // check if tensor is already an input
  unsigned i = 0;
  for (auto &in : inputs) {
    if (in.isTensor() && in.toTensor().unsafeGetTensorImpl()
          == t.unsafeGetTensorImpl())
      return { i, true };
    ++i;
  }

  inputs.emplace_back(t);
  return { (unsigned)inputs.size()-1, true };
}

void Trace::append_arg(const Tensor &t) {
  ops[next_op-1].args.emplace_back(get_tensor_idx(t));
}

void Trace::append_arg(ArrayRef<Tensor> arg) {
  vector<InputIdx> val;
  for (auto &t : arg) {
    val.emplace_back(get_tensor_idx(t));
  }
  ops[next_op-1].args.emplace_back(move(val));
}

void Trace::append_arg(const optional<Tensor> &arg) {
  optional<InputIdx> val;
  if (arg)
    val = get_tensor_idx(*arg);
  ops[next_op-1].args.emplace_back(move(val));
}

void Trace::append_arg(const List<optional<Tensor>> &arg) {
  vector<optional<InputIdx>> val;
  for (const auto &it : arg) {
    const optional<Tensor> &in = it;
    optional<InputIdx> elem;
    if (in)
      elem = get_tensor_idx(*in);
    val.emplace_back(move(elem));
  }
  ops[next_op-1].args.emplace_back(move(val));
}

void Trace::append_arg(Storage &&arg) {
  ops[next_op-1].args.emplace_back(InputIdx(inputs.size(), true));
  inputs.emplace_back(move(arg));
}

void Trace::append_arg(optional<Generator> &&arg) {
  optional<InputIdx> val;
  if (arg) {
    val = InputIdx(inputs.size(), true);
    inputs.emplace_back(move(*arg));
  }
  ops[next_op-1].args.emplace_back(move(val));
}

void Trace::append_arg(string_view arg) {
  ops[next_op-1].args.emplace_back(string(arg.data(), arg.size()));
}

void Trace::append_arg(optional<string_view> arg) {
  optional<string> copy;
  if (arg)
    copy = string(arg->data(), arg->size());
  ops[next_op-1].args.emplace_back(move(copy));
}

unsigned Trace::register_tensor(uintptr_t tensor, TorchOp op_id,
                                c10::DispatchKeySet ks) {
  assert(!flushing);
#ifndef TORCHY_RELEASE
  // FOR DEBUGGING ONLY. Can be used to binary search a trace that goes wrong
  static unsigned call_count = 0;
  ++call_count;
  if (auto *limit = getenv("TORCHY_FLUSH_BEFORE")) {
    if (next_op != 0 && call_count <= (unsigned)atoi(limit))
      flush(STATS(FlushReason::DEBUG));
  }
  if (auto *limit = getenv("TORCHY_FLUSH_AFTER")) {
    if (next_op != 0 && call_count > (unsigned)atoi(limit))
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
  op.id = op_id;
  assert(op.args.empty());
  op.observable = true;
  op.dead = false;

  auto &rdata = data[next_op];
  rdata.tensors[0] = tensor;
  for (unsigned i = 1; i < rdata.tensors.size(); ++i) {
    rdata.tensors[i] = 0;
  }
  rdata.refs = 1;
  rdata.tls = ThreadLocalState();
  rdata.dispatch_key = ks;
  rdata.inplace = op.inplace();
  return next_op++;
}

void Trace::add_shared(unsigned idx, uintptr_t ptr) {
  assert(idx < next_op);
  for (auto &tensor : data[idx].tensors) {
    if (tensor == 0) {
      tensor = ptr;
      incref(idx);
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
  auto &op    = ops[idx];
  auto &rdata = data[idx];

  bool found = false;
  for (auto &tensor : rdata.tensors) {
    if (tensor == ptr) {
      tensor = 0;
      found = true;
      break;
    }
  }
  assert(found); (void)found;

  op.observable = rdata.hasTensors();
  decref(idx);

  // reclaim slot if this was the last created tensor
  if (rdata.refs == 0 && idx+1 == next_op) {
    op.destroy();
    rdata.destroy();
    --next_op;
  }
}

TraceCacheKey Trace::mk_trace_key() {
  TraceOpDef *new_ops = new TraceOpDef[next_op];
  // we can't move the args for the interpreter as it traverses the args
  // in run() rather than compile() -- a NOP
  std::uninitialized_copy_n(ops, next_op, new_ops);
  return TraceCacheKey(new_ops, next_op);
}

void Trace::flush(STATS(FlushReason reason)) {
  assert(!flushing);
  assert(next_op > 0);
  flushing = true;

  stats_register_trace(*this, reason);

#ifdef TORCHY_PRINT_TRACE_ON_FLUSH
  cerr << "Flush trace\n" << *this << endl;
#endif

  TraceCacheKey key = { ops, next_op };
  auto I = cache.find(key);
  key.ops.release(); // so the destructor doesn't kick in

  if (I == cache.end()) {
    STATS(StopWatch time);
    auto *program      = backend->compile(*this);
    auto *used_backend = backend;

    // fallback to the interpreter if the default backend can't handle this
    if (!program) {
      program      = interpreter.compile(*this);
      used_backend = &interpreter;
    }
    STATS(time.stop());
    stats_register_compile_time(time);

    I = cache.emplace(piecewise_construct, forward_as_tuple(mk_trace_key()),
                      forward_as_tuple(program, used_backend)).first;
  }

  STATS(StopWatch run_time);
  I->second.backend->run(I->second.program, *this);
  STATS(run_time.stop());
  stats_register_trace_time(run_time);

  // reduce reference count on tensors s.t. they are deleted if possible
  for (unsigned i = 0; i < next_op; ++i) {
    ops[i].destroy();
    data[i].destroy();
  }
  inputs.clear();

  next_op = 0;
  flushing = false;
}

ostream& operator<<(ostream &os, const Trace &t) {
  if (t.next_op == 0)
    return os << "(empty)\n";

  for (unsigned i = 0; i < t.next_op; ++i) {
    t.print(os, i);
  }

  if (t.inputs.empty())
    return os;

  os << "\nInputs:\n";
  unsigned i = 0;
  for (auto &in : t.inputs) {
    os << "in<" << i++ << ">: ";
    if (in.isTensor()) {
      const auto &t = in.toTensor();
      os << "tensor(" << t.scalar_type() << " : " << t.sizes() << ")\n";
    } else if (in.isGenerator()) {
      const auto &g = in.toGenerator();
      os << "generator(" << g.current_seed() << ", " << g.device() << ")\n";
    } else if (in.isStorage()) {
      os << "storage(" << in.toStorage().nbytes() << ")\n";
    } else {
      assert(0);
    }
  }
  return os;
}
