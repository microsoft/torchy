// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

#include "stopwatch.h"
#include <cassert>

using namespace std;
using namespace std::chrono;

static auto now() {
  return steady_clock::now();
}

StopWatch::StopWatch() : start(now()) {}

void StopWatch::stop() {
  assert(!stopped);
  end = now();
  assert((stopped = true));
}

float StopWatch::seconds() const {
  assert(stopped);
  return duration_cast<duration<float>>(end - start).count();
}
