#pragma once

// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

// Please update stats.cpp string version of this when adding a new reason!
enum class FlushReason {
  DEBUG,
  DIM,
  HAS_STORAGE,
  INPLACE_SHARED,
  IS_CONTIGUOUS,
  NUMEL,
  OVERFLOW_SHARED_LIST,
  SET_SIZE,
  SET_STORAGE_OFFSET,
  SET_STRIDE,
  SIZE,
  SIZES,
  STORAGE,
  STORAGE_OFFSET,
  STRIDE,
  STRIDES,
  TRACE_MAX_LENGTH,
  UNSUPPORTED_OPERATION,
  NUM_REASONS
};


#ifdef TORCHY_ENABLE_STATS

class StopWatch;
class Trace;

#define STATS(x) x
void stats_register_trace(const Trace &t, FlushReason reason);
void stats_register_trace_time(const StopWatch &run_time);

#else

#define STATS(x)
#define stats_register_trace(t, reason) (void)0
#define stats_register_trace_time(run_time) (void)0

#endif
