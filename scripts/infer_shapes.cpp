// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

#include <ATen/core/List.h>
#include <ATen/NativeFunctions.h>
#include <ATen/RedispatchFunctions.h>
#include <c10/util/Logging.h>
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

void fill_vector(Shape &shape, unsigned dims, unsigned i) {
  if (i == dims) {
    lookup_shape(shape);
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

  // add one more shape for layout tests
  lookup_shape({-1});
}

Tensor new_tensor(unsigned shape, ScalarType ty) {
  IntArrayRef ref(all_shapes[shape]);
  auto t = native::empty_cpu(ref, ty);
  native::ones_out(ref, t);
  return t;
}

#include "../shape_inference.h"

unsigned standard_promote(unsigned a, unsigned b) {
  if (a == -1u) return b;
  if (b == -1u) return a;
  return lookup_shape(shape_std_promote(all_shapes[a], all_shapes[b]));
}

unsigned standard_promote(unsigned a, unsigned b, unsigned c) {
  return standard_promote(standard_promote(a, b), c);
}

unsigned standard_promote(const vector<unsigned> &shapes) {
  unsigned shape = shapes[0];
  for (auto sh : shapes) {
    shape = standard_promote(shape, sh);
  }
  return shape;
}

unsigned pick_1st(unsigned s) {
  if (s == -1u) return -1u;
  auto &shape = all_shapes[s];
  return shape.empty() ? -1u : lookup_shape({ shape[0] });
}

unsigned matmul(unsigned a, unsigned b) {
  if (a == -1u || b == -1u) return -1u;
  auto &shape_a = all_shapes[a];
  auto &shape_b = all_shapes[b];
  if (shape_a.empty() || shape_b.empty())
    return -1u;
  return lookup_shape(shape_matmul(shape_a, shape_b));
}

unsigned mul(unsigned a, unsigned b) {
  if (a == -1u || b == -1u) return -1u;
  auto &shape_a = all_shapes[a];
  auto &shape_b = all_shapes[b];
  if (shape_a.empty() || shape_b.empty())
    return -1u;
  return lookup_shape(shape_mul(shape_a, shape_b));
}

unsigned mult(unsigned a, unsigned b) {
  if (a == -1u || b == -1u) return -1u;
  auto &shape_a = all_shapes[a];
  auto &shape_b = all_shapes[b];
  if (shape_a.empty() || shape_b.empty())
    return -1u;
  return lookup_shape(shape_mult(shape_a, shape_b));
}

unsigned mul_last(unsigned a, unsigned b) {
  if (a == -1u || b == -1u) return -1u;
  return lookup_shape(shape_mul_last(all_shapes[a], all_shapes[b]));
}

unsigned join(unsigned a, unsigned b) {
  if (a == -1u || b == -1u) return -1u;
  return lookup_shape(shape_join(all_shapes[a], all_shapes[b]));
}

unsigned pad1(unsigned s) {
  if (s == -1u) return -1u;
  return lookup_shape(shape_pad1(all_shapes[s]));
}

unsigned drop1(unsigned s) {
  if (s == -1u) return -1u;
  auto res = all_shapes[s];
  if (res.size() < 1)
    return -1u;
  res.pop_back();
  return lookup_shape(move(res));
}

unsigned drop2(unsigned s) {
  if (s == -1u) return -1u;
  auto res = all_shapes[s];
  if (res.size() < 2)
    return -1u;
  res.pop_back();
  res.pop_back();
  return lookup_shape(move(res));
}

unsigned reshape(unsigned s, unsigned to) {
  if (s == -1u || to == -1u) return -1u;
  return lookup_shape(shape_reshape(all_shapes[s], all_shapes[to]));
}

unsigned pool2d(unsigned in, unsigned shape) {
  if (in == -1u || shape == -1u) return -1u;
  auto &s_in = all_shapes[in];
  if (s_in.size() < 2) return -1u;
  return lookup_shape(shape_pool2d(s_in, all_shapes[shape]));
}

unsigned transpose2d(unsigned s) {
  if (s == -1u) return -1u;
  return lookup_shape(shape_transpose2d(all_shapes[s]));
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
    auto sz = results.size();
    for (auto ty : { kFloat, kShort }) {
      for (unsigned shape = 0; shape < num_test_shapes; ++shape) {
        type_trail.push_back(shape);
        call(function<Tensor(Tail...)>{[=](Tail... args) -> Tensor {
          auto t = new_tensor(shape, ty);
          return fn(t, args...);
        }});
        type_trail.pop_back();
      }
      if (results.size() != sz)
        break;
    }
  }

  template <typename T, typename... Tail>
  void call(function<Tensor(c10::optional<T>&, Tail...)> fn) {
    // call with a value
    call(function<Tensor(T&, Tail...)>{
      [=](T &val, Tail... args) -> Tensor {
        c10::optional<T> opt(val);
        return fn(opt, args...);
      }});

    // and call without a value
    type_trail.push_back(-1u);
    call(function<Tensor(Tail...)>{[=](Tail... args) -> Tensor {
      c10::optional<T> opt;
      return fn(opt, forward<Tail>(args)...);
    }});
    type_trail.pop_back();
  }

  template <typename T, typename... Tail>
  void call(function<Tensor(c10::optional<T>, Tail...)> fn) {
    // call with a value
    call(function<Tensor(T, Tail...)>{
      [=](T val, Tail... args) -> Tensor {
        return fn(val, args...);
      }});

    // and call without a value
    type_trail.push_back(-1u);
    call(function<Tensor(Tail...)>{[=](Tail... args) -> Tensor {
      return fn(c10::nullopt, forward<Tail>(args)...);
    }});
    type_trail.pop_back();
  }

  template <typename... Tail>
  void call(function<Tensor(TensorList&, Tail...)> fn) {
    auto sz = results.size();
    for (auto ty : { kFloat, kShort }) {
      for (unsigned shape = 0; shape < num_test_shapes; ++shape) {
        type_trail.push_back(shape);
        for (unsigned shape2 = 0; shape2 < num_test_shapes; ++shape2) {
          type_trail.push_back(shape2);
          call(function<Tensor(Tail...)>{[=](Tail... args) -> Tensor {
            Tensor ts[2] = { new_tensor(shape, ty),
                             new_tensor(shape2, ty) };
            ArrayRef<Tensor> aref(ts);
            return fn(aref, args...);
          }});
          type_trail.pop_back();
        }
        type_trail.pop_back();
      }
      if (results.size() != sz)
        break;
    }
  }

  template <typename... Tail>
  void call(function<Tensor(List<c10::optional<Tensor>>&, Tail...)> fn) {
    auto sz = results.size();
    for (auto ty : { kFloat, kShort }) {
      for (unsigned shape = 0; shape < num_test_shapes; ++shape) {
        type_trail.push_back(shape);
        for (unsigned shape2 = 0; shape2 < num_test_shapes; ++shape2) {
          type_trail.push_back(shape2);
          call(function<Tensor(Tail...)>{[=](Tail... args) -> Tensor {
            List<c10::optional<Tensor>> list({ new_tensor(shape, ty),
                                               new_tensor(shape2, ty) });
            return fn(list, args...);
          }});
          type_trail.pop_back();
        }
        type_trail.pop_back();
      }
      if (results.size() != sz)
        break;
    }
  }

  template <typename... Tail>
  void call(function<Tensor(at::Scalar&, Tail...)> fn) {
    call(function<Tensor(Tail...)>{[=](Tail... args) -> Tensor {
      Scalar s(1);
      return fn(s, args...);
    }});
  }

  template <typename... Tail>
  void call(function<Tensor(int64_t, Tail...)> fn) {
    //auto n = num_samples;
    for (int64_t v : {0, 1}) {
      call(function<Tensor(Tail...)>{[=](Tail... args) -> Tensor {
        return fn(v, args...);
      }});
      /* FIXME
      if (num_samples > n)
        break;
      */
    }
  }

  template <typename... Tail>
  void call(function<Tensor(IntArrayRef, Tail...)> fn) {
    // Note here we want <= to test one extra shape
    for (unsigned shape = 0; shape <= num_test_shapes; ++shape) {
      type_trail.push_back(shape);
      call(function<Tensor(Tail...)>{[=](Tail... args) -> Tensor {
        return fn(all_shapes[shape], args...);
      }});
      type_trail.pop_back();
    }
  }

  template <typename... Tail>
  void call(function<Tensor(ScalarType, Tail...)> fn) {
    call(function<Tensor(Tail...)>{[=](Tail... args) -> Tensor {
      return fn(kFloat, args...);
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
    bool eq_second = true;
    bool eq_third = true;
    bool std_promote = true;
    bool promote_1_2 = true;
    bool promote_1_2_3 = true;
    bool pick_1st_2nd = true;
    bool mul_1_2 = true;
    bool mult_1_2 = true;
    bool matmul_1_2 = true;
    bool matmul_2_3 = true;
    bool mullast_1_2 = true;
    bool join_2_3 = true;
    bool is_pad1 = true;
    bool is_drop1 = true;
    bool is_drop2 = true;
    bool is_reshape = true;
    bool is_pool2d = true;
    bool is_transpose2d = true;

    for (auto &result : results) {
      auto &trail = result.inputs;
      auto &out = result.output;

#define TEST(var, test) \
  var = var && test

#define DEBUG(what)                 \
  if (what != out) {                \
    cout << "WRONG: ";              \
    print(result);                  \
    cout << " vs ";                 \
    print_shape(what);              \
    cout << endl;                   \
  }

      TEST(all_equal,    out == results[0].output);
      TEST(eq_first,     out == trail[0]);
      TEST(eq_second,    trail.size() >= 2 && out == trail[1]);
      TEST(eq_third,     trail.size() >= 3 && out == trail[2]);
      TEST(std_promote,  out == standard_promote(trail));
      TEST(promote_1_2,  trail.size() >= 2 &&
                         out == standard_promote(trail[0], trail[1]));
      TEST(promote_1_2_3, trail.size() >= 3 &&
                          out == standard_promote(trail[0], trail[1],trail[2]));
      TEST(pick_1st_2nd, trail.size() >= 2 && out == pick_1st(trail[1]));
      TEST(mul_1_2,      trail.size() >= 2 && out == mul(trail[0], trail[1]));
      TEST(mult_1_2,     trail.size() >= 2 && out == mult(trail[0], trail[1]));
      TEST(matmul_1_2,   trail.size() >= 2 &&
                         out == matmul(trail[0], trail[1]));
      TEST(matmul_2_3,   trail.size() >= 3 &&
                         out == matmul(trail[1], trail[2]));
      TEST(mullast_1_2,  trail.size() >= 2 &&
                         out == mul_last(trail[0], trail[1]));
      TEST(join_2_3,     trail.size() >= 3 && out == join(trail[1], trail[2]));
      TEST(is_pad1,      out == pad1(trail[0]));
      TEST(is_drop1,     out == drop1(trail[0]));
      TEST(is_drop2,     out == drop2(trail[0]));
      TEST(is_reshape,   trail.size() >= 2 &&
                         out == reshape(trail[0], trail[1]));
      TEST(is_pool2d,    trail.size() >= 2 &&
                         out == pool2d(trail[0], trail[1]));
      TEST(is_transpose2d, out == transpose2d(trail[0]));
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
    PRINT(eq_second, "EQ_SECOND")
    PRINT(eq_third, "EQ_THIRD")
    PRINT(std_promote, "STD_PROMOTE")
    PRINT(promote_1_2, "PROMOTE_1_2")
    PRINT(promote_1_2_3, "PROMOTE_1_2_3")
    PRINT(pick_1st_2nd, "PICK_1ST_2ND")
    PRINT(mul_1_2, "MUL_1ST_2ND")
    PRINT(mult_1_2, "MULT_1ST_2ND")
    PRINT(matmul_1_2, "MATMUL_1ST_2ND")
    PRINT(matmul_2_3, "MATMUL_2ND_3RD")
    PRINT(mullast_1_2, "MULLAST_1ST_2ND")
    PRINT(join_2_3, "JOIN_2_3")
    PRINT(is_pad1, "PAD1")
    PRINT(is_drop1, "DROP1")
    PRINT(is_drop2, "DROP2")
    PRINT(is_reshape, "RESHAPE")
    PRINT(is_pool2d, "POOL2D")
    PRINT(is_transpose2d, "TRANSPOSE2D")

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

  SetStackTraceFetcher([]() { return string(); });

#include "call_pytorch_fns.h"

  _Exit(0);
  return 0;
}
