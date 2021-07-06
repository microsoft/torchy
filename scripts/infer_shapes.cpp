// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

#include <ATen/core/List.h>
#include <ATen/NativeFunctions.h>
#include <ATen/RedispatchFunctions.h>
#include <array>
#include <cassert>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <unistd.h>

using namespace at;
using namespace c10;
using namespace std;

namespace {
array<unsigned char, 3> dim_sz = { 1, 2, 7 };
constexpr unsigned max_dim = 3;
unsigned num_test_shapes;

using Shape = vector<long>;
vector<Shape> all_shapes;
map<Shape, unsigned> map_shapes;

void fill_vector(Shape &shape, unsigned dims, unsigned i) {
  if (i == dims) {
    map_shapes[shape] = all_shapes.size();
    all_shapes.emplace_back(shape);
    return;
  }

  for (auto n : dim_sz) {
    shape.push_back(n);
    fill_vector(shape, dims, i+1);
    shape.pop_back();
  }
}

void init_shapes() {
  all_shapes.reserve(1024);

  Shape shape;
  for (unsigned dim = 0; dim <= max_dim; ++dim) {
    fill_vector(shape, dim, 0);
    assert(shape.empty());
  }
  num_test_shapes = all_shapes.size();
  assert(num_test_shapes == map_shapes.size());
}

template <typename T>
unsigned lookup_shape(T &&shape) {
  auto p = map_shapes.emplace(forward<T>(shape), all_shapes.size());
  if (p.second)
    all_shapes.emplace_back(p.first->first);
  return p.first->second;
}

unsigned lookup_shape(IntArrayRef shape) {
  return lookup_shape(shape.vec());
}

Tensor new_tensor(unsigned shape) {
  IntArrayRef ref(all_shapes[shape]);
  auto t = native::empty_cpu(ref, kFloat);
  native::ones_out(ref, t);
  return t;
}

unsigned standard_promote(unsigned a, unsigned b) {
  if (a == b)
    return a;

  auto &shape_a = all_shapes[a];
  auto &shape_b = all_shapes[b];
  if (shape_a.empty()) return b;
  if (shape_b.empty()) return a;

  bool is_a_ge = shape_a.size() >= shape_b.size();
  auto &ge = is_a_ge ? shape_a : shape_b;
  auto &lt = is_a_ge ? shape_b : shape_a;

  auto promoted = ge;
  unsigned j = lt.size()-1;
  for (int i = ge.size()-1; i >= 0; --i) {
    if (lt[j] > ge[i])
      promoted[i] = lt[j];

    if (j-- == 0)
      break;
  }
  return lookup_shape(move(promoted));
}

unsigned standard_promote(const vector<unsigned> &shapes) {
  unsigned shape = shapes[0];
  for (auto sh : shapes) {
    shape = standard_promote(shape, sh);
  }
  return shape;
}

unsigned pick_1st(unsigned s) {
  auto &shape = all_shapes[s];
  return shape.empty() ? -1u : lookup_shape({ shape[0] });
}

unsigned join(unsigned a, unsigned b) {
  auto res = all_shapes[a];
  auto &shape_b = all_shapes[b];
  res.insert(res.end(), shape_b.begin(), shape_b.end());
  return lookup_shape(move(res));
}

struct Result {
  vector<unsigned> inputs;
  unsigned output;
};

char *call_only = nullptr;
vector<unsigned> type_trail;
vector<Result> results;

void print_shape(unsigned id) {
  if (id == -1u) {
    cout << "null";
    return;
  }
  cout << '<';
  bool first = true;
  for (auto n : all_shapes[id]) {
    if (!first)
      cout << ", ";
    first = false;
    cout << n;
  }
  cout << '>';
}

void print(const Result &result) {
  bool first = true;
  for (auto input : result.inputs) {
    if (!first)
      cout << ", ";
    first = false;
    print_shape((unsigned)input);
  }
  cout << " -> ";
  print_shape(result.output);
}

struct C {
  const char *name;

  void call(function<Tensor()> fn) {
    try {
      auto shape = lookup_shape(fn().sizes());
      results.push_back({type_trail, shape});
#if 0
    } catch (const c10::Error &) {
      cout << "Exception!\n";
    } catch (const runtime_error &) {
      cout << "Runtime Exception!\n";
    }
#endif
    } catch (...) {}
  }

  template <typename... Tail>
  void call(function<Tensor(Tensor&, Tail...)> fn) {
    for (unsigned shape = 0; shape < num_test_shapes; ++shape) {
      type_trail.push_back(shape);
      call(function<Tensor(Tail&&...)>{[=](Tail&&... args) -> Tensor {
        auto t = new_tensor(shape);
        return fn(t, args...);
      }});
      type_trail.pop_back();
    }
  }

  template <typename T, typename... Tail>
  void call(function<Tensor(c10::optional<T>&, Tail...)> fn) {
    // call with a value
    call(function<Tensor(T&, Tail&&...)>{
      [=](T &val, Tail&&... args) -> Tensor {
        c10::optional<T> opt(val);
        return fn(opt, args...);
      }});

    // and call without a value
    type_trail.push_back(-1u);
    call(function<Tensor(Tail&&...)>{[=](Tail&&... args) -> Tensor {
      c10::optional<T> opt;
      return fn(opt, forward<Tail>(args)...);
    }});
    type_trail.pop_back();
  }

  template <typename... Tail>
  void call(function<Tensor(TensorList&, Tail...)> fn) {
    for (unsigned shape = 0; shape < num_test_shapes; ++shape) {
      type_trail.push_back(shape);
      for (unsigned shape2 = 0; shape2 < num_test_shapes; ++shape2) {
        type_trail.push_back(shape2);
        call(function<Tensor(Tail&&...)>{[=](Tail&&... args) -> Tensor {
          Tensor ts[2] = { new_tensor(shape),
                           new_tensor(shape2) };
          ArrayRef<Tensor> aref(ts);
          return fn(aref, args...);
        }});
        type_trail.pop_back();
      }
      type_trail.pop_back();
    }
  }

  template <typename... Tail>
  void call(function<Tensor(List<c10::optional<Tensor>>&, Tail...)> fn) {
    for (unsigned shape = 0; shape < num_test_shapes; ++shape) {
      type_trail.push_back(shape);
      for (unsigned shape2 = 0; shape2 < num_test_shapes; ++shape2) {
        type_trail.push_back(shape2);
        call(function<Tensor(Tail&&...)>{[=](Tail&&... args) -> Tensor {
          List<c10::optional<Tensor>> list({ new_tensor(shape),
                                             new_tensor(shape2) });
          return fn(list, args...);
        }});
        type_trail.pop_back();
      }
      type_trail.pop_back();
    }
  }

  template <typename... Tail>
  void call(function<Tensor(at::Scalar&, Tail...)> fn) {
    call(function<Tensor(Tail&&...)>{[=](Tail&&... args) -> Tensor {
      Scalar s(1);
      return fn(s, args...);
    }});
  }

  template <typename T>
  void analyze(function<T> fn) {
    if (call_only && strcmp(name, call_only))
      return;

    results.clear();
    call(move(fn));

    bool all_equal = true;
    bool eq_first = true;
    bool std_promote = true;
    bool pick_1st_2nd = true;
    bool join_2_3 = true;

    for (auto &result : results) {
      auto &trail = result.inputs;
      auto &out = result.output;

#define TEST(var, test) \
  var = var && test

#define DEBUG(what)                 \
  if (what != type) {               \
    cout << "WRONG: ";              \
    print(result);                  \
    cout << " vs " << what << endl; \
  }

      TEST(all_equal,    out == results[0].output);
      TEST(eq_first,     out == trail[0]);
      TEST(std_promote,  out == standard_promote(trail));
      TEST(pick_1st_2nd, trail.size() >= 2 && out == pick_1st(trail[1]));
      TEST(join_2_3,     trail.size() >= 3 && out == join(trail[1], trail[2]));
    }

    cout << name;

    if (results.empty()) {
      cout << ": NO_SAMPLES" << endl;
      return;
    }
    if (all_equal) {
      cout << ": ALL ";
      print_shape(results[0].output);
      cout << endl;
      return;
    }

#define PRINT(var, msg)         \
    if (var) {                  \
      cout << ": " msg << endl; \
      return;                   \
    }

    PRINT(eq_first, "EQ_FIRST")
    PRINT(std_promote, "STD_PROMOTE")
    PRINT(pick_1st_2nd, "PICK_1ST_2ND")
    PRINT(join_2_3, "JOIN_2_3")

    cout << ": NON_STANDARD:\n";

    for (auto &result : results) {
      print(result);
      if (standard_promote(result.inputs) != result.output) {
        cout << " [non-standard promotion]\n";
      } else {
        cout << '\n';
      }
    }
    cout << endl;
  }

};

}

int main(int argc, char **argv) {
  if (argc > 1) {
    call_only = argv[1];
  }
  results.reserve(1024 * 1024);
  init_shapes();

#include "call_pytorch_fns.h"

  _Exit(0);
  return 0;
}
