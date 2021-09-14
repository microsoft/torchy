// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

#include "tensor.h"
#include "trace.h"

static void init_update_in_place(TensorOp &op) {
  for (auto tensor : op.tensors) {
    if (tensor != 0)
      init_update_in_place(tensor);
  }
}

static void end_update_in_place(TensorOp &op) {
  assert(op.tensors[0] != 0);
  end_update_in_place_first(op.tensors[0]);

  for (unsigned i = 1; i < op.tensors.size(); ++i) {
    if (op.tensors[i] != 0)
      end_update_in_place_copy(op.tensors[i], op.tensors[0]);
  }
}

#ifndef NDEBUG
static void finish_trace(TensorOp &op) {
  for (auto tensor : op.tensors) {
    if (tensor != 0)
      finish_trace(tensor);
  }
}
#else
# define finish_trace(op) (void)0
#endif

static void set(TensorOp &op, const at::Tensor &t) {
  for (auto tensor : op.tensors) {
    if (tensor != 0)
      set(tensor, t);
  }
}
