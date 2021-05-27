#pragma once

// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

#include <ATen/Tensor.h>

class TensorVisitor {
  std::function<void(const at::Tensor&)> visit;

public:
  TensorVisitor(std::function<void(const at::Tensor&)> &&visit)
   : visit(std::move(visit)) {}

  template <typename T>
  void operator()(const T&) {}

  void operator()(const at::Tensor &t) {
    visit(t);
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

  template<typename T>
  void operator()(const at::List<T> &l) {
    for (const auto &it : l) {
      const T &elem = it;
      (*this)(elem);
    }
  }
};
