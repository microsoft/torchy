#pragma once

// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

#ifdef TORCHY_RELEASE

#ifndef NDEBUG
#define NDEBUG
#endif

#else

#ifdef NDEBUG
#undef NDEBUG
#endif

#define TORCHY_PRINT_TRACE_ON_FLUSH

#endif
