#pragma once

// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

#include "config.h"
#include <ostream>

namespace at { class Tensor; }

unsigned trace_idx(const at::Tensor &t);

void set(uintptr_t torchy, at::Tensor &&t);
void init_update_in_place(uintptr_t torchy);
void end_update_in_place(uintptr_t torchy);
