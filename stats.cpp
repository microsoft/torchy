// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

#include "config.h"

#ifdef TORCHY_ENABLE_STATS
#include <algorithm>
#include <cstring>
#include <iostream>

using namespace std;

#define NUM_ELEMS(a) (sizeof(a) / sizeof(*a))

namespace {

const char* flush_reasons[] = {
  "debug",
  "dim",
  "has_storage",
  "inplace shared",
  "is_contiguous",
  "numel",
  "overflow shared list",
  "set_size",
  "set_storage_offset",
  "set_stride",
  "size",
  "sizes",
  "storage",
  "storage_offset",
  "stride",
  "strides",
  "trace max length",
  "unsupported operation",
};

static_assert(NUM_ELEMS(flush_reasons) == (unsigned)FlushReason::NUM_REASONS);

unsigned flush_reasons_count[(unsigned)FlushReason::NUM_REASONS] = {0};

struct PrintStats {
  ~PrintStats() {
    cerr << "\n\n------------ STATISTICS ------------\n";
    print_table("Trace Flush Reason", flush_reasons_count, flush_reasons,
                (unsigned)FlushReason::NUM_REASONS);
    cerr << endl;
  }

  void print_table(const char *header, unsigned *data, const char **labels,
                   unsigned size) {
    cerr << header << ":\n";

    unsigned max_label = 0;
    for (unsigned i = 0; i < size; ++i) {
      if (data[i] != 0)
        max_label = max(max_label, (unsigned)strlen(labels[i]));
    }

    for (unsigned i = 0; i < size; ++i) {
      if (data[i] == 0)
        continue;
      cerr << labels[i] << ": ";
      pad(labels[i], max_label);
      cerr << data[i] << '\n';
    }
    cerr << '\n';
  }

  void pad(const char *str, unsigned length) {
    for (unsigned i = strlen(str); i < length; ++i) {
      cerr << ' ';
    }
  }
};

PrintStats printer;

}

void inc_flush_reason(FlushReason reason) {
  ++flush_reasons_count[(unsigned)reason];
}

#endif
