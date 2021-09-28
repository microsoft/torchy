// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

#include "backends.h"
#include "tensor.h"
#include "trace.h"

#define MAX_NUM_INPUTS 12

static void init_update_in_place(const TraceOpRunTimeData &data) {
  for (auto tensor : data.tensors) {
    if (tensor != 0)
      init_update_in_place(tensor);
  }
}

static void end_update_in_place(const TraceOpRunTimeData &data) {
  bool first = true;
  unsigned first_idx;

  for (unsigned i = 0; i < data.tensors.size(); ++i) {
    if (data.tensors[i] == 0)
      continue;
    if (first) {
      end_update_in_place_first(data.tensors[i]);
      first_idx = i;
    } else {
      end_update_in_place_copy(data.tensors[i], data.tensors[first_idx]);
    }
    first = false;
  }
}

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
