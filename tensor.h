#pragma once

// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

#include <ostream>

namespace at { class Tensor; }
class TorchyTensor;

unsigned trace_idx(const at::Tensor &t);
std::ostream& operator<<(std::ostream &os, const TorchyTensor &tt);

void set(TorchyTensor *tt, at::Tensor &&t);
void init_update_in_place(TorchyTensor *tt);
void end_update_in_place(TorchyTensor *tt);
