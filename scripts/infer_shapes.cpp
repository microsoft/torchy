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
#include <variant>

using namespace at;
using namespace c10;
using namespace std;

namespace {
array<unsigned char, 3> dim_sz = { 1, 2, 7 };
constexpr unsigned max_dim = 3;

using Shape = vector<int64_t>;
using ShapeRef = IntArrayRef;
vector<Shape> test_shapes;

void fill_vector(Shape &shape, unsigned dims, unsigned i) {
  if (i == dims) {
    test_shapes.push_back(shape);
    return;
  }

  for (auto n : dim_sz) {
    shape.push_back(n);
    fill_vector(shape, dims, i+1);
    shape.pop_back();
  }
}

void init_shapes() {
  Shape shape;
  for (unsigned dim = 0; dim <= max_dim; ++dim) {
    fill_vector(shape, dim, 0);
    assert(shape.empty());
  }

  // add one more shape for layout tests
  test_shapes.push_back({-1});
}

Tensor new_tensor(unsigned shape, ScalarType ty) {
  IntArrayRef ref(test_shapes[shape]);
  auto t = native::empty_cpu(ref, ty);
  native::ones_out(ref, t);
  return t;
}

#include "../shape_inference.h"

// uint8_t - idx in test_shapes
// bool - empty optional
using TrailElem = variant<uint8_t, bool>;

#define GET_SHAPE(v) \
  auto idx_##v = get_if<uint8_t>(&v); \
  if (!idx_##v) return {}; \
  const auto &shape_##v = test_shapes[*idx_##v];

std::optional<Shape> shape_std_promote(TrailElem a, TrailElem b, TrailElem c) {
  GET_SHAPE(a);
  GET_SHAPE(b);
  GET_SHAPE(c);
  return shape_std_promote(shape_std_promote(shape_a, shape_b), shape_c);
}

std::optional<Shape> shape_std_promote(const vector<TrailElem> &shapes) {
  std::optional<Shape> shape;
  for (auto elem : shapes) {
    if (auto *idx = get_if<uint8_t>(&elem)) {
      auto &sh = test_shapes[*idx];
      shape = shape ? shape_std_promote(*shape, sh) : sh;
    }
  }
  return shape;
}

std::optional<Shape> pick_1st(TrailElem a) {
  GET_SHAPE(a);
  return shape_a.empty() ? Shape() : Shape({ shape_a[0] });
}

std::optional<Shape> drop1(TrailElem a) {
  GET_SHAPE(a);
  if (shape_a.size() < 1)
    return shape_a;

  auto res = shape_a;
  res.pop_back();
  return res;
}

std::optional<Shape> drop2(TrailElem a) {
  GET_SHAPE(a);
  if (shape_a.size() < 1)
    return shape_a;

  auto res = shape_a;
  res.pop_back();
  res.pop_back();
  return res;
}

std::optional<Shape> get_shape(TrailElem a) {
  GET_SHAPE(a)
  return shape_a;
}

#define DECL_UNARY(name) \
std::optional<Shape> name(TrailElem a) { \
  GET_SHAPE(a); \
  return name(shape_a); \
}

#define DECL_BINARY(name, cond) \
std::optional<Shape> name(TrailElem a, TrailElem b) { \
  GET_SHAPE(a); \
  GET_SHAPE(b); \
  if (!(cond)) return {}; \
  return name(shape_a, shape_b); \
}

DECL_UNARY(shape_pad1)
DECL_UNARY(shape_transpose2d)
DECL_BINARY(shape_std_promote, true)
DECL_BINARY(shape_mul, !shape_a.empty() && !shape_b.empty())
DECL_BINARY(shape_mult, !shape_a.empty() && !shape_b.empty())
DECL_BINARY(shape_matmul, !shape_a.empty() && !shape_b.empty())
DECL_BINARY(shape_mul_last, true)
DECL_BINARY(shape_join, true)
DECL_BINARY(shape_reshape, true)
DECL_BINARY(shape_pool2d, shape_a.size() >= 2)

bool print_all = false;
char *call_only = nullptr;
vector<TrailElem> type_trail;

std::optional<Shape> all_shape;

array<tuple<const char*, unsigned, function<std::optional<Shape>()>>, 20>
is_shape_fn = {
  make_tuple("ALL", 1, [&]() { return all_shape; }),
  make_tuple("EQ_FIRST", 1, [&]() { return get_shape(type_trail[0]); }),
  make_tuple("EQ_SECOND", 2, [&]() { return get_shape(type_trail[1]); }),
  make_tuple("EQ_THIRD", 3, [&]() { return get_shape(type_trail[2]); }),
  make_tuple("STD_PROMOTE", 1, [&]() { return shape_std_promote(type_trail); }),
  make_tuple("PROMOTE_1_2", 2, [&]() { return shape_std_promote(type_trail[0], type_trail[1]); }),
  make_tuple("PROMOTE_1_2_3", 3, [&]() { return shape_std_promote(type_trail[0], type_trail[1], type_trail[2]); }),
  make_tuple("PICK_1ST_2ND", 2, [&]() { return pick_1st(type_trail[1]); }),
  make_tuple("MUL_1ST_2ND", 2, [&]() { return shape_mul(type_trail[0], type_trail[1]); }),
  make_tuple("MULT_1ST_2ND", 2, [&]() { return shape_mult(type_trail[0], type_trail[1]); }),
  make_tuple("MATMUL_1ST_2ND", 2, [&]() { return shape_matmul(type_trail[0], type_trail[1]); }),
  make_tuple("MATMUL_2ND_3RD", 3, [&]() { return shape_matmul(type_trail[1], type_trail[2]); }),
  make_tuple("MULLAST_1ST_2ND", 2, [&]() { return shape_mul_last(type_trail[0], type_trail[1]); }),
  make_tuple("JOIN_2_3", 3, [&]() { return shape_join(type_trail[1], type_trail[2]); }),
  make_tuple("PAD1", 1, [&]() { return shape_pad1(type_trail[0]); }),
  make_tuple("DROP1", 1, [&]() { return drop1(type_trail[0]); }),
  make_tuple("DROP2", 1, [&]() { return drop2(type_trail[0]); }),
  make_tuple("RESHAPE", 2, [&]() { return shape_reshape(type_trail[0], type_trail[1]); }),
  make_tuple("POOL2D", 2, [&]() { return shape_pool2d(type_trail[0], type_trail[1]); }),
  make_tuple("TRANSPOSE2D", 1, [&]() { return shape_transpose2d(type_trail[0]); }),
};

array<bool, is_shape_fn.size()> is_shape_flags;
unsigned num_samples = 0;

void print(ShapeRef output) {
  bool first = true;
  for (auto input : type_trail) {
    if (!first)
      cout << ", ";
    first = false;
    if (auto *idx = get_if<uint8_t>(&input)) {
      cout << ShapeRef(test_shapes[*idx]);
    } else if (get_if<bool>(&input)) {
      cout << "(null)";
    } else {
      assert(0);
    }
  }
  cout << " -> " << output;
}


void infer(ShapeRef output) {
  ++num_samples;

  if (!all_shape)
    all_shape = output.vec();

  auto num_args = type_trail.size();

  for (unsigned i = 0; i < is_shape_fn.size(); ++i) {
    auto &f = is_shape_flags[i];
    if (!f)
      continue;

    bool n_args = num_args >= get<1>(is_shape_fn[i]);
    auto fn_out = n_args ? get<2>(is_shape_fn[i])() : std::nullopt;
    f = n_args && fn_out && output == *fn_out;

    if (print_all && !f) {
      cout << "FAILED: " << get<0>(is_shape_fn[i]);
      if (n_args && fn_out) {
        cout << " / EXPECTING: " << ShapeRef(*fn_out);
      }
      cout << '\n';
    }
  }
}

struct C {
  const char *name;

  void call(function<Tensor()> fn) {
    try {
      auto tensor = fn();
      auto shape = tensor.sizes();

      if (print_all)
        print(shape);
      infer(shape);
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
    auto sz = num_samples;
    for (auto ty : { kFloat, kShort }) {
      for (uint8_t shape = 0; shape < test_shapes.size(); ++shape) {
        type_trail.emplace_back(shape);
        call(function<Tensor(Tail...)>{[=](Tail... args) -> Tensor {
          auto t = new_tensor(shape, ty);
          return fn(t, args...);
        }});
        type_trail.pop_back();
      }
      if (num_samples != sz)
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
    type_trail.emplace_back(false);
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
    type_trail.emplace_back(false);
    call(function<Tensor(Tail...)>{[=](Tail... args) -> Tensor {
      return fn(c10::nullopt, forward<Tail>(args)...);
    }});
    type_trail.pop_back();
  }

  template <typename... Tail>
  void call(function<Tensor(TensorList&, Tail...)> fn) {
    auto sz = num_samples;
    for (auto ty : { kFloat, kShort }) {
      for (uint8_t shape = 0; shape < test_shapes.size(); ++shape) {
        type_trail.emplace_back(shape);
        for (uint8_t shape2 = 0; shape2 < test_shapes.size(); ++shape2) {
          type_trail.emplace_back(shape2);
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
      if (num_samples != sz)
        break;
    }
  }

  template <typename... Tail>
  void call(function<Tensor(List<c10::optional<Tensor>>&, Tail...)> fn) {
    auto sz = num_samples;
    for (auto ty : { kFloat, kShort }) {
      for (uint8_t shape = 0; shape < test_shapes.size(); ++shape) {
        type_trail.emplace_back(shape);
        for (uint8_t shape2 = 0; shape2 < test_shapes.size(); ++shape2) {
          type_trail.emplace_back(shape2);
          call(function<Tensor(Tail...)>{[=](Tail... args) -> Tensor {
            List<c10::optional<Tensor>> list({ new_tensor(shape, ty),
                                               new_tensor(shape2, ty) });
            return fn(list, args...);
          }});
          type_trail.pop_back();
        }
        type_trail.pop_back();
      }
      if (num_samples != sz)
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
    // TODO
    for (int64_t v : {1}) {
      call(function<Tensor(Tail...)>{[=](Tail... args) -> Tensor {
        return fn(v, args...);
      }});
    }
  }

  template <typename... Tail>
  void call(function<Tensor(IntArrayRef, Tail...)> fn) {
    for (uint8_t shape = 0; shape < test_shapes.size(); ++shape) {
      type_trail.emplace_back(shape);
      call(function<Tensor(Tail...)>{[=](Tail... args) -> Tensor {
        return fn(test_shapes[shape], args...);
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

    for (auto &f : is_shape_flags) {
      f = true;
    }
    all_shape.reset();

    call(move(fn));

    cout << name;

    if (num_samples == 0) {
      cout << ": NO_SAMPLES\n";
      return;
    }
    if (is_shape_flags[0]) {
      cout << ": ALL " << ShapeRef(*all_shape) << '\n';
      return;
    }
    for (unsigned i = 1; i < is_shape_flags.size(); ++i) {
      if (is_shape_flags[i]) {
        cout << ": " << get<0>(is_shape_fn[i]) << '\n';
        return;
      }
    }
    cout << ": NON_STANDARD\n";
  }

};

}

int main(int argc, char **argv) {
  if (argc > 3) {
    cerr << "Usage: " << argv[0] << " <fn name> <verbose>\n";
    return -1;
  }
  if (argc >= 2)
    call_only = argv[1];
  if (argc == 3)
    print_all = true;

  init_shapes();

  SetStackTraceFetcher([]() { return string(); });

#include "call_pytorch_fns.h"

  return 0;
}
