#pragma once

// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

#include "config.h"
#include <c10/core/ScalarType.h>
#include <c10/util/ArrayRef.h>
#include <cstdint>

namespace at { class Tensor; }

unsigned trace_idx(const at::Tensor &t);

void set(uintptr_t torchy, const at::Tensor &t);
void init_update_in_place(uintptr_t torchy);
void end_update_in_place(uintptr_t torchy);

at::ScalarType tensor_get_dtype(uintptr_t tt);
bool tensor_has_shape(uintptr_t tt);
at::IntArrayRef tensor_get_shape(uintptr_t tt);
