// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

#include "config.h"

#ifdef TORCHY_ENABLE_STATS
#include "stopwatch.h"
#include "trace.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

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
};

static_assert(NUM_ELEMS(flush_reasons) == (unsigned)FlushReason::NUM_REASONS);

float median(vector<float> &v) {
  sort(v.begin(), v.end());
  auto sz = v.size();
  if (sz % 2 == 0)
    return (v[sz/2 - 1] + v[sz/2]) / 2.0;
  return v[sz / 2];
}

void inc(vector<unsigned> &v, size_t idx, unsigned amount = 1) {
  if (idx >= v.size())
    v.resize(idx+1);
  v[idx] += amount;
}

string shrink_trace(const string &t) {
  istringstream ss(t);
  string out;
  unsigned lines = 0;

  for (string line; getline(ss, line); ) {
    if (line.find("[dead]") != string::npos)
      continue;

    if (++lines == 20) {
      out += "...\\l";
      break;
    }

    auto pos = line.find(" #refs=");
    if (pos != string::npos)
      line.resize(pos);

    pos = line.find(" #output");
    if (pos != string::npos)
      line.resize(pos);

    out += line.substr(0, 50);
    out += "\\l";
  }
  return out;
}

array<unsigned, (unsigned)FlushReason::NUM_REASONS> flush_reasons_count;
array<unsigned, MAX_TRACE_LENGTH+1> trace_size;
array<unsigned, MAX_TRACE_LENGTH+1> num_trace_outputs;
array<unsigned, MAX_TRACE_LENGTH+1> num_trace_deads;
vector<pair<float, string>> trace_compile_time;
unordered_map<string, vector<float>> trace_run_time;
unordered_map<string, unordered_map<string, unsigned>> trace_successors;
string first_trace, current_trace, last_trace;
unsigned unsupported_wrappers = 0;
unsigned torchscript_failures = 0;

struct PrintStats {
  ~PrintStats() {
    cerr << "\n\n------------ STATISTICS ------------\n\n";
    print_table("Trace Flush Reason", flush_reasons_count.data(), flush_reasons,
                flush_reasons_count.size());
    print_table("Trace Size", trace_size.data(), trace_size.size());
    print_table("Number of Outputs per Trace", num_trace_outputs.data(),
                num_trace_outputs.size());
    print_table("Number of Dead Ops per Trace", num_trace_deads.data(),
                num_trace_deads.size());

    vector<unsigned> successor_frequency;
    for (const auto &p : trace_successors) {
      inc(successor_frequency, p.second.size());
    }
    print_table("Number of Successors per Trace", successor_frequency.data(),
                successor_frequency.size());

    vector<unsigned> trace_freq_stats;
    unsigned total = 0;
    for (const auto &p : trace_run_time) {
      inc(trace_freq_stats, p.second.size());
      total += p.second.size();
    }
    print_table("Frequency per Trace", trace_freq_stats.data(),
                trace_freq_stats.size());

    int cutoff = trace_freq_stats.size()-1;
    unsigned num_traces = 0;
    for (; cutoff >= 0; --cutoff) {
      if (num_traces >= 20)
        break;
      num_traces += trace_freq_stats[cutoff];
    }

    vector<unsigned> trace_times;
    for (auto &p : trace_run_time) {
      inc(trace_times, unsigned(median(p.second) * 1000000.0), p.second.size());
    }
    print_table("Run-times per Trace (microseconds)", trace_times.data(),
                trace_times.size());

    print_header("Most Frequent Traces");
    for (auto &p : trace_run_time) {
      if (p.second.size() >= (unsigned)cutoff)
        cerr << "Trace executed " << p.second.size() << " times ("
             << unsigned(median(p.second) * 1000000.0) << " us)\n"
             << p.first << "\n\n";
    }

    print_header("Slowest Trace Compilation");
    sort(trace_compile_time.begin(), trace_compile_time.end());
    {
      auto I = trace_compile_time.rbegin(), E = trace_compile_time.rend();
      for (unsigned i = 0; i < 10 && I != E; ++i, ++I) {
        cerr << "Trace compiled in "
             << unsigned(I->first * 1000000.0) << " us\n"
             << I->second << "\n\n";
      }
    }

    cerr << "Number of Torchscript failures:\t" << torchscript_failures
         << "\nNumber of unsupported ops:\t" << unsupported_wrappers
         << "\nNumber of traces:\t" << total
         << "\nDistinct traces:\t" << trace_run_time.size() << '\n';

    cerr << endl;

    if (auto p = getenv("TORCHY_PRINT_DOT"))
      if (*p)
        print_dot();
  }

private:
  void print_dot() {
    {
      ofstream f("trace.dot");
      print_dot(f, false);
    }
    {
      ofstream f("trace_detailed.dot");
      print_dot(f, true);
    }
  }

  void print_dot(ofstream &os, bool label_trace) {
    unordered_map<string, unsigned> trace_map;
    trace_map[first_trace] = 0;

    auto get_id = [&](const string &t) -> unsigned {
      auto sz = trace_map.size();
      return trace_map.emplace(t, sz).first->second;
    };

    os << "digraph {\n"
          "n0 [label=Start]\n"
          "n0 -> T0\n";

    for (const auto &p : trace_successors) {
      unsigned src = get_id(p.first);

      if (label_trace)
        os << 'T' << src << " [label=\"" << shrink_trace(p.first)
           << "\", shape=box]\n";

      for (const auto &p : p.second) {
        os << 'T' << src << " -> T" << get_id(p.first) << '\n';
      }
    }
    os << "}\n";
  }

  void print_table(const char *header, unsigned *data, const char **labels,
                   size_t size) {
    print_header(header);

    size_t max_label = 0;
    for (unsigned i = 0; i < size; ++i) {
      if (data[i] != 0)
        max_label = max(max_label, strlen(labels[i]));
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

  void print_table(const char *header, unsigned *data, size_t size) {
    print_header(header);

    size_t max_label = to_string(size-1).size();

    for (unsigned i = 0; i < size; ++i) {
      if (data[i] == 0)
        continue;
      string label = to_string(i);
      cerr << label << ": ";
      pad(label.c_str(), max_label);
      cerr << data[i] << '\n';
    }
    cerr << '\n';
  }

  void print_header(const char *header) {
    cerr << header << '\n';
    repeat('=', strlen(header));
    cerr << '\n';
  }

  void pad(const char *str, size_t length) {
    repeat(' ', length - strlen(str));
  }

  void repeat(char ch, size_t length) {
    for (size_t i = 0; i < length; ++i) {
      cerr << ch;
    }
  }
};

PrintStats printer;

}

void stats_register_trace(const Trace &t, FlushReason reason) {
  ++flush_reasons_count[(unsigned)reason];

  unsigned num_ops = t.numOps();
  ++trace_size[num_ops];

  unsigned num_outputs = 0;
  unsigned num_deads = 0;

  auto *ops = t.getOps();
  for (unsigned i = 0; i < num_ops; ++i) {
    auto &op = ops[i];
    num_outputs += op.observable;
    num_deads   += op.dead;
  }
  assert(num_outputs > 0);
  ++num_trace_outputs[num_outputs];
  ++num_trace_deads[num_deads];

  // We need to get the trace's string representation before it is executed.
  // Tensor's trace_idx gets overwritten during materialization.
  stringstream trace_ss;
  trace_ss << t;
  current_trace = move(trace_ss).str();

  if (first_trace.empty())
    first_trace = current_trace;
}

void stats_register_compile_time(const StopWatch &run_time) {
  trace_compile_time.emplace_back(run_time.seconds(), current_trace);
}

void stats_register_trace_time(const StopWatch &run_time) {
  trace_run_time[current_trace].emplace_back(run_time.seconds());

  if (!last_trace.empty())
    ++trace_successors[last_trace][current_trace];
  last_trace = move(current_trace);
}

void stats_inc_unsupported_wrapper() {
  ++unsupported_wrappers;
}

void stats_inc_torchscript_fail() {
  ++torchscript_failures;
}

#endif
