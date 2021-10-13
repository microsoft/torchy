#pragma once

// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

#include "config.h"
#include <c10/core/ScalarType.h>
#include <c10/util/ArrayRef.h>
#include <cstdint>

namespace at { class Tensor; }
namespace c10 { class TensorImpl; }

unsigned trace_idx(const at::Tensor &t);
c10::TensorImpl* is_impl(uintptr_t torchy);

void set(uintptr_t torchy, const at::Tensor &t);
#ifndef NDEBUG
void finish_trace(uintptr_t torchy);
#endif

bool tensor_has_dtype(uintptr_t tt);
at::ScalarType tensor_get_dtype(uintptr_t tt);
bool tensor_has_shape(uintptr_t tt);
at::IntArrayRef tensor_get_shape(uintptr_t tt);
