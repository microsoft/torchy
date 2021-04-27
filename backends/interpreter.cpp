// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

#include "dispatch.h"
#include "ops.h"
#include "tensor.h"
#include "trace.h"
#include <ATen/core/List.h>
#include <ATen/RedispatchFunctions.h>

using namespace at;

namespace interpreter {

void run(Trace &t) {
  auto *ops = t.getOps();

  for (unsigned i = 0, e = t.numOps(); i < e; ++i) {
    auto &op = ops[i];
    if (!op.needsComputing())
      continue;

    auto ks = op.dispatch_key;
    for (auto &arg : op.args) {
      if (auto t = get_if<Tensor>(&arg)) {
        ks = ks | t->key_set();
      }
    }
    ks = ks & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);

    switch (op.id) {

#include "autogen/interpreter_redispatch.h"

      default:
        assert(0 && "Unhandled op");
    }
  }
}

}
