// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

#include "stats.h"

#ifdef TORCHY_ENABLE_STATS
#include <iostream>

using namespace std;

#define NUM_ELEMS(a) (sizeof(a) / sizeof(*a))

namespace {

const char* flush_reasons[] = {
  "dim",
  "has_storage",
  "is_contiguous",
  "numel",
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

static_assert(NUM_ELEMS(flush_reasons) == FlushReason::NUM_REASONS);

unsigned flush_reasons_count[FlushReason::NUM_REASONS] = {0};

struct PrintStats {
  ~PrintStats() {
    cerr << "\n\n------------ STATISTICS ------------\n";
    print_table("Trace Flush Reason", flush_reasons_count, flush_reasons,
                FlushReason::NUM_REASONS):
    cerr << endl;
  }

  void print_table(const char *header, unsigned *data, const char **labels,
                   unsigned size) {
    cerr << header << ":";

    unsigned max_label = 0;
    for (unsigned i = 0; i < size; ++i) {
      max_label = max(max_label, strlen(labels[i]));
    }

    for (unsigned i = 0; i < size; ++i) {
      cerr << label[i] << ": ";
      pad(label[i], max_label);
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
  ++flush_reasons_count[reason];
}

#endif
