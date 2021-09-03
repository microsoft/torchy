// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

#include "ops.h"

using namespace std;

static const char *op_names[] = {
#include "autogen/ops_names.h"
};

const char* op_name(TorchOp op) {
  return op_names[op];
}

ostream& operator<<(ostream &os, TorchOp op) {
  return os << op_names[op];
}
