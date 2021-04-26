// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

#include "ops.h"

using namespace std;

static char *op_names[] = {
#include "autogen/ops_names.h"
};

ostream& operator<<(ostream &os, TorchOp op) {
  return os << op_names[op];
}
