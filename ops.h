#pragma once

// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

#include <ostream>

enum TorchOp {
#include "autogen/ops_enum.h"
};

std::ostream& operator<<(std::ostream &os, TorchOp op);
