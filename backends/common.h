// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

#include "backends.h"
#include "tensor.h"
#include "trace.h"

#define MAX_NUM_INPUTS 12

#ifndef NDEBUG
static void finish_trace(const TraceOpRunTimeData &data) {
  for (auto tensor : data.tensors) {
    if (tensor != 0)
      finish_trace(tensor);
  }
}
#else
# define finish_trace(op) (void)0
#endif

static void set(const TraceOpRunTimeData &data, const at::Tensor &t) {
  for (auto tensor : data.tensors) {
    if (tensor != 0)
      set(tensor, t);
  }
}
