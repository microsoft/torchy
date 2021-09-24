#pragma once

// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

class InputIdx;

class TensorVisitor {
  std::function<void(InputIdx)> visit;
  InputData &inputs;

public:
  TensorVisitor(std::function<void(InputIdx)> &&visit, InputData &inputs)
   : visit(std::move(visit)), inputs(inputs) {}

  template <typename T>
  void operator()(const T&) {}

  void operator()(const InputIdx &idx) {
    if (idx.is_trace() || inputs[idx.input_idx()].isTensor())
      visit(idx);
  }

  template<typename T>
  void operator()(const c10::optional<T> &a) {
    if (a)
      (*this)(*a);
  }

  template<typename T>
  void operator()(const std::vector<T> &l) {
    for (const auto &elem : l) {
      (*this)(elem);
    }
  }
};
