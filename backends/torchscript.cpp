// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

#include "common.h"
#include <torch/csrc/jit/api/method.h>
#include <map>

using namespace at;
using namespace torch::jit;

#define DEBUG_GRAPH

#ifdef DEBUG_GRAPH
# include <iostream>
#endif

namespace {

std::string cut_overload(const char *fn) {
  auto *dot = strchr(fn, '.');
  return dot ? std::string(fn, dot - fn) : fn;
}

class ValGen {
  Graph &g;
  Value **results;
  std::vector<Value*> inputs;
  Value *none_val = nullptr;

public:
  ValGen(Graph &g, Value **results, const Stack &in_stack)
    : g(g), results(results) {
    for (auto &in : in_stack) {
      auto *v = g.addInput();

      if (in.isTensor()) {
        v->setType(TensorType::create(in.toTensor()));
      } else if (in.isGenerator()) {
        v->setType(GeneratorType::get());
      } else {
        assert(in.isStorage());
        v->setType(StorageType::get());
      }
      inputs.push_back(v);
    }
  }

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

  Value* operator()(const InputIdx &in) {
    return in.is_input() ? inputs[in.input_idx()] : results[in.trace_idx()];
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
    if (vals.empty())
      return mk_none();

    auto ty = vals[0]->type()->cast<TensorType>() ? TensorType::get()
                                                  : vals[0]->type();
    auto *n = g.createList(ty, vals);
    g.appendNode(n);
    return n->output();
  }

  template<typename T>
  Value* operator()(const std::vector<optional<InputIdx>> &l) {
    std::vector<Value*> vals;
    for (const auto &elem : l) {
      vals.emplace_back((*this)(elem));
      assert(vals.back()->type()->cast<NoneType>() ||
             vals.back()->type()->cast<TensorType>());
    }
    auto *n = g.createList(OptionalType::ofTensor(), vals);
    g.appendNode(n);
    return n->output();
  }
};

struct CompiledProgram {
  std::unique_ptr<GraphFunction> fn;
  uint8_t output_ops[MAX_TRACE_LENGTH];
  unsigned num_outputs = 0;
};

}


void* TorchScript::compile(const Trace &t) {
  auto *ops = t.getOps();
  Value *results[MAX_TRACE_LENGTH];
  Value *outputs[MAX_TRACE_LENGTH];
  Value *op_inputs[MAX_NUM_INPUTS];

  auto prog = std::make_unique<CompiledProgram>();
  auto &output_ops  = prog->output_ops;
  auto &num_outputs = prog->num_outputs;

  auto graph = std::make_shared<Graph>();
  ValGen val_gen(*graph, results, t.getInputs());

  for (unsigned i = 0, e = t.numOps(); i < e; ++i) {
    auto &op = ops[i];
    if (op.dead)
      continue;

    unsigned num_inputs = 0;
    for (auto &arg : op.args) {
      auto *v = visit(val_gen, arg);
      if (!v) {
        stats_inc_torchscript_fail();
        return nullptr;
      }
      op_inputs[num_inputs++] = v;
    }

    Node *n = graph->create(Symbol::aten(cut_overload(op_name(op.id))),
                            at::ArrayRef<Value*>(op_inputs, num_inputs));
    if (!n->maybeOperator()) {
#ifdef DEBUG_GRAPH
      std::cerr << "Op not supported by TorchScript: " << op_name(op.id)
                << std::endl;
      // prints a nice error msg with supported overloads
      n->getOperator();
#endif
      stats_inc_torchscript_fail();
      return nullptr;
    }
    graph->appendNode(n);

    Value *v = n->output();
    results[i] = v;
    if (op.observable && !op.inplace()) {
      output_ops[num_outputs] = i;
      outputs[num_outputs++] = v;
    }

    // TODO: set type of output if available?
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

  prog->fn
    = std::make_unique<GraphFunction>("torchy", move(graph),
                                      std::function<void(GraphFunction&)>());
  // force optimization now
  prog->fn->optimized_graph();

#ifdef DEBUG_GRAPH
  prog->fn->optimized_graph()->print(std::cerr << "\nOptimized graph:\n");
  std::cerr << '\n';
#endif

  return prog.release();
}

void TorchScript::run(const void *ptr, Trace &t) {
  auto prog = (const CompiledProgram*)ptr;
  auto &output_ops = prog->output_ops;
  auto num_outputs = prog->num_outputs;
  auto *data = t.getRuntimeData();

  // FIXME: we don't take TLS or dispatch keys into account here
  // may break more complicated programs..

  for (unsigned i = 0, e = t.numOps(); i < e; ++i) {
    auto &op = data[i];
    if (op.needsComputing() && op.inplace)
      init_update_in_place(op);
  }

  auto &stack = t.getInputs();
  prog->fn->run(stack);
  // inputs are consumed, and the output is passed back on the stack
  assert(stack.size() == 1);

  // patch tensors with the output
  if (num_outputs == 1) {
    assert(stack[0].isTensor());
    set(data[output_ops[0]], std::move(stack[0]).toTensor());
  }
  else if (num_outputs > 1) {
    assert(stack[0].isTuple());
    auto t = std::move(stack[0]).toTuple();
    auto &elems = t->elements();
    assert(elems.size() == num_outputs);

    for (unsigned i = 0; i < num_outputs; ++i) {
      assert(elems[i].isTensor());
      set(data[output_ops[i]], std::move(elems[i]).toTensor());
    }
  }

  for (unsigned i = 0, e = t.numOps(); i < e; ++i) {
    auto &op = data[i];
    if (op.needsComputing() && op.inplace)
      end_update_in_place(op);
    finish_trace(op);
  }
}

void TorchScript::destroy(void *prog) {
  delete (CompiledProgram*)prog;
}
