// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

#include "autogen/ops_data.h"
#include "common.h"
#include "trace.h"
#include <torch/csrc/jit/api/method.h>
#include <map>

using namespace at;
using namespace torch::jit;

#define DEBUG_GRAPH

#ifdef DEBUG_GRAPH
# include <iostream>
#endif

#define MAX_NUM_INPUTS 12

namespace {

std::string cut_overload(const char *fn) {
  auto *dot = strchr(fn, '.');
  return dot ? std::string(fn, dot - fn) : fn;
}

using ValueMap = std::map<const TensorImpl*, Value*>;

class ValGen {
  Graph &g;
  ValueMap &map;
  Stack &inputs;
  Value *none_val = nullptr;

public:
  ValGen(Graph &g, ValueMap &map, Stack &inputs)
    : g(g), map(map), inputs(inputs) {}

  Value* mk_none() {
    if (!none_val) {
      auto *n = g.createNone();
      g.appendNode(n);
      none_val = n->output();
    }
    return none_val;
  }

  template<typename T>
  Value* operator()(const T &a) {
    return g.insertConstant(a);
  }

  Value* operator()(const Tensor &t) {
    auto &v = map[t.getIntrusivePtr().get()];
    if (!v) {
      v = g.addInput();
      inputs.emplace_back(t);
    }
    return v;
  }

  template<typename T>
  Value* operator()(const optional<T> &a) {
    return a ? (*this)(*a) : mk_none();
  }

  template<typename T>
  Value* operator()(const std::vector<T> &l) {
    std::vector<Value*> vals;
    for (const auto &elem : l) {
      vals.emplace_back((*this)(elem));
    }
    auto *n = g.createList(vals[0]->type(), vals);
    g.appendNode(n);
    return n->output();
  }

  template<typename T>
  Value* operator()(const List<T> &l) {
    std::vector<Value*> vals;
    for (const auto &it : l) {
      const T &elem = it;
      vals.emplace_back((*this)(elem));
    }
    auto *n = g.createList(vals[0]->type(), vals);
    g.appendNode(n);
    return n->output();
  }

  // unsupported by TorchScript
  Value* operator()(const Storage&) {
    return nullptr;
  }
};
}


namespace torchscript {

bool run(Trace &t) {
  auto *ops = t.getOps();
  Value *outputs[MAX_TRACE_LENGTH];
  uint8_t output_ops[MAX_TRACE_LENGTH];
  unsigned num_outputs = 0;
  Stack stack;
  ValueMap val_map;
  Value *op_inputs[MAX_NUM_INPUTS];

  auto graph = std::make_shared<Graph>();
  ValGen val_gen(*graph, val_map, stack);

  for (unsigned i = 0, e = t.numOps(); i < e; ++i) {
    auto &op = ops[i];
    if (!op.needsComputing())
      continue;

    if (op.id >= FIRST_INPLACE_OP)
      init_update_in_place(op);

    unsigned num_inputs = 0;
    for (auto &arg : op.args) {
      auto *v = visit(val_gen, arg);
      if (!v) {
        stats_inc_torchscript_fail();
        return false;
      }
      op_inputs[num_inputs++] = v;
    }

    Node *n = graph->create(Symbol::aten(cut_overload(op_name(op.id))),
                            at::ArrayRef<Value*>(op_inputs, num_inputs));
    if (!n->maybeOperator()) {
      std::cerr << "Op not supported by TorchScript: " << op_name(op.id)
                << std::endl;
      // prints a nice error msg with supported overloads
      n->getOperator();
    }
    graph->appendNode(n);

    Value *v = n->output();
    if (op.observable && op.id < FIRST_INPLACE_OP) {
      output_ops[num_outputs] = i;
      outputs[num_outputs++] = v;
    }

    for (auto tt : op.tensors) {
      if (auto *t = is_impl(tt))
        val_map.emplace(t, v);
    }
  }

  if (num_outputs == 0) {
    graph->registerOutput(val_gen.mk_none());
  } else if (num_outputs == 1) {
    graph->registerOutput(outputs[0]);
  } else if (num_outputs > 1) {
    auto *t = graph->createTuple(at::ArrayRef<Value*>(outputs, num_outputs));
    graph->appendNode(t);
    graph->registerOutput(t->output());
  }

#ifdef DEBUG_GRAPH
  graph->print(std::cerr);
#endif

  assert((graph->lint(), true));

  GraphFunction fn("torchy", move(graph), {});

#ifdef DEBUG_GRAPH
  fn.optimized_graph()->print(std::cerr << "\nOptimized graph:\n");
  std::cerr << '\n';
#endif

  fn.run(stack);
  // inputs are consumed, and the output is passed back on the stack
  assert(stack.size() == 1);

  // patch tensors with the output
  if (num_outputs == 1) {
    assert(stack[0].isTensor());
    set(ops[output_ops[0]], std::move(stack[0]).toTensor());
  }
  else if (num_outputs > 1) {
    assert(stack[0].isTuple());
    auto t = std::move(stack[0]).toTuple();
    auto &elems = t->elements();
    assert(elems.size() == num_outputs);

    for (unsigned i = 0; i < num_outputs; ++i) {
      assert(elems[i].isTensor());
      set(ops[output_ops[i]], std::move(elems[i]).toTensor());
    }
  }

  for (unsigned i = 0, e = t.numOps(); i < e; ++i) {
    auto &op = ops[i];
    if (op.id >= FIRST_INPLACE_OP)
      end_update_in_place(op);
    finish_trace(op);
  }
  return true;
}

}
