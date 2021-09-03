// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

#include "tensor.h"
#include "trace.h"
#include <torch/csrc/jit/api/method.h>
#include <map>

using namespace at;
using namespace torch::jit;

//#define DEBUG_GRAPH

#ifdef DEBUG_GRAPH
# include <iostream>
#endif

#define MAX_NUM_INPUTS 12

namespace {

using ValueMap = std::map<const TensorImpl*, Value*>;

class ValGen {
  Graph &g;
  ValueMap &map;

public:
  ValGen(Graph &g, ValueMap &map) : g(g), map(map) {}

//insertConstant(const IValue& val <-- Scalar)

  template<typename T>
  Value* operator()(const T &a) {
    return nullptr; // TODO
  }

  Value* operator()(const Tensor &t) {
    auto &v = map[t.getIntrusivePtr().get()];
    if (!v)
      v = g.addInput();
    return v;
  }

  template<typename T>
  Value* operator()(const optional<T> &a) {
    if (!a)
      return g.createNone();
    return (*this)(*a);
  }

  template<typename T>
  Value* operator()(const std::vector<T> &l) {
    /*
    for (const auto &elem : l) {
    }
    */
    return nullptr; // TODO
  }

  template<typename T>
  Value* operator()(const List<T> &l) {
    /*
    for (const auto &it : l) {
    }
    */
   //createList(const TypePtr& elem_type, at::ArrayRef<Value*> values);
    return nullptr; // TODO
  }

  Value* operator()(const Storage &s) {
    return nullptr; // TODO
  }

  Value* operator()(const Generator &g) {
    return nullptr; // TODO
  }
};
}


namespace torchscript {

void run(Trace &t) {
  auto *ops = t.getOps();
  Value *outputs[MAX_TRACE_LENGTH];
  unsigned num_outputs = 0;
  Stack fn_inputs;
  ValueMap val_map;
  Value *op_inputs[MAX_NUM_INPUTS];

  auto graph = std::make_shared<Graph>();
  ValGen val_gen(*graph, val_map);

  for (unsigned i = 0, e = t.numOps(); i < e; ++i) {
    auto &op = ops[i];
    if (!op.needsComputing())
      continue;

    unsigned num_inputs = 0;
    for (auto &arg : op.args) {
      op_inputs[num_inputs++] = val_gen(arg);
    }

    Node *n = graph->create(Symbol::aten(op_name(op.id)),
                            at::ArrayRef<Value*>(op_inputs, num_inputs));
    if (op.observable)
      outputs[num_outputs++] = n->output();
  }

#ifdef DEBUG_GRAPH
  graph->print(std::cerr);
#endif

  assert((graph->lint(), true));

  if (num_outputs > 1) {
    auto *t = graph->createTuple(at::ArrayRef<Value*>(outputs, num_outputs));
    graph->registerOutput(t->output());
  } else {
    assert(num_outputs == 1);
    graph->registerOutput(outputs[0]);
  }

  GraphFunction fn("torchy", move(graph), {});
  fn.run(fn_inputs);
}

}
