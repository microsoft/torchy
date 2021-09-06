// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

#include "tensor.h"
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

using ValueMap = std::map<const TensorImpl*, Value*>;

class ValGen {
  Graph &g;
  ValueMap &map;

public:
  ValGen(Graph &g, ValueMap &map) : g(g), map(map) {}

  template<typename T>
  Value* operator()(const T &a) {
    return g.insertConstant(a);
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
      return g.createNone()->output();
    return (*this)(*a);
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

  Value* operator()(const Device &d) {
    std::cerr << "DEVICE" << std::endl;
    return nullptr; // TODO
  }

  Value* operator()(const Dimname &d) {
    std::cerr << "DIMNAME" << std::endl;
    return nullptr; // TODO
  }

  Value* operator()(const Generator &g) {
    std::cerr << "GENERATOR" << std::endl;
    return nullptr; // TODO
  }

  Value* operator()(const Layout &l) {
    std::cerr << "LAYOUT" << std::endl;
    return nullptr; // TODO
  }

  Value* operator()(const MemoryFormat &m) {
    std::cerr << "MEMORYFORMAT" << std::endl;
    return nullptr; // TODO
  }

  Value* operator()(const ScalarType &s) {
    std::cerr << "SCALARTYPE" << std::endl;
    return nullptr; // TODO
  }

  Value* operator()(const Storage &s) {
    std::cerr << "STORAGE" << std::endl;
    return nullptr; // TODO
  }

  Value* operator()(const std::string &s) {
    std::cerr << "STRING" << std::endl;
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
      op_inputs[num_inputs++] = visit(val_gen, arg);
    }

    Node *n = graph->create(Symbol::aten(op_name(op.id)),
                            at::ArrayRef<Value*>(op_inputs, num_inputs));
    graph->appendNode(n);

    Value *v = n->output();
    if (op.observable)
      outputs[num_outputs++] = v;

    for (auto tt : op.tensors) {
      if (auto *t = is_impl(tt))
        val_map.emplace(t, v);
    }
  }

  if (num_outputs > 1) {
    auto *t = graph->createTuple(at::ArrayRef<Value*>(outputs, num_outputs));
    graph->appendNode(t);
    graph->registerOutput(t->output());
  } else {
    assert(num_outputs == 1);
    graph->registerOutput(outputs[0]);
  }

#ifdef DEBUG_GRAPH
  graph->print(std::cerr);
#endif

  assert((graph->lint(), true));

  GraphFunction fn("torchy", move(graph), {});
  fn.run(fn_inputs);
}

}
