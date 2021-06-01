#pragma once

// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

#include <chrono>

class StopWatch {
  std::chrono::steady_clock::time_point start, end;
#ifndef NDEBUG
  bool stopped = false;
#endif

public:
  StopWatch();
  void stop();
  float seconds() const;
};
