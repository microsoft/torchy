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
#include "../shape_inference.h"
#include "../strides_inference.h"

using namespace at;
using namespace c10;
using namespace std;

namespace {

using Shape = vector<int64_t>;
using Strides = vector<int64_t>;
using ShapeStrides = pair<Shape, Strides>;
using ShapeRef = IntArrayRef;
using StridesRef = IntArrayRef;

vector<ShapeStrides> test_shapes;

void init_shapes() {
  test_shapes.emplace_back(Shape({1, 2}), Strides({2, 1}));
  test_shapes.emplace_back(Shape({1, 2}), Strides({1, 2}));
  test_shapes.emplace_back(Shape({1, 2}), Strides({1, 1}));
  test_shapes.emplace_back(Shape({2, 1}), Strides({1, 2}));
  test_shapes.emplace_back(Shape({2, 3, 4}), Strides({12, 4, 1}));
  test_shapes.emplace_back(Shape({2, 3, 4}), Strides({12, 1, 4}));
  test_shapes.emplace_back(Shape({2, 3, 4}), Strides({1, 4, 12}));
  test_shapes.emplace_back(Shape({2, 3, 4}), Strides({4, 1, 12}));
  test_shapes.emplace_back(Shape({2, 3, 4}), Strides({1, 12, 4}));
  test_shapes.emplace_back(Shape({2, 3, 4}), Strides({4, 12, 1}));
  test_shapes.emplace_back(Shape({2, 3, 4}), Strides({1, 1, 1}));
  test_shapes.emplace_back(Shape({2, 3, 4}), Strides({1, 2, 3}));
  test_shapes.emplace_back(Shape({2, 3, 4}), Strides({18, 1, 1}));
  test_shapes.emplace_back(Shape({2, 3, 4}), Strides({1, 16, 1}));
  test_shapes.emplace_back(Shape({1, 1, 2}), Strides({1, 1, 3}));
  test_shapes.emplace_back(Shape({1, 9, 3 ,2}), Strides({1, 2, 7, 5}));
  test_shapes.emplace_back(Shape({1, 9, 3, 2}), Strides({54, 6, 2, 5}));

  // for layout tests
  test_shapes.emplace_back(Shape({-1}), Strides({1}));
  test_shapes.emplace_back(Shape({1, 0}), Strides({1, 0}));
  test_shapes.emplace_back(Shape({1, 2, 0}), Strides({2, 1, 0}));
}

Tensor new_tensor(unsigned idx, ScalarType ty) {
  auto &[shape, strides] = test_shapes[idx];
  auto t = native::empty_strided_cpu(shape, strides, ty);
  native::ones_out(shape, t);
  return t;
}

// uint8_t - idx in test_shapes
// void* - empty optional
using TrailElem = variant<uint8_t, void*, bool, int64_t, at::MemoryFormat>;

#define GET_SHAPE(v) \
  auto idx_##v = get_if<uint8_t>(&v); \
  if (!idx_##v) return {}; \
  const auto &shape_##v = test_shapes[*idx_##v].first;

#define GET_STRIDES(v) \
  auto idx_##v = get_if<uint8_t>(&v); \
  if (!idx_##v) return {}; \
  const auto &strides_##v = test_shapes[*idx_##v].second;

#define GET_SHAPES_STRIDES(v) \
  auto idx_##v = get_if<uint8_t>(&v); \
  if (!idx_##v) return {}; \
  const auto &shape_##v = test_shapes[*idx_##v].first; \
  const auto &strides_##v = test_shapes[*idx_##v].second;

#define GET_BOOL(v) \
  auto val_ref_##v = get_if<bool>(&v); \
  if (!val_ref_##v) return {}; \
  bool bool_##v = *val_ref_##v;

#define GET_INT(v) \
  auto val_ref_##v = get_if<int64_t>(&v); \
  if (!val_ref_##v) return {}; \
  int64_t int_##v = *val_ref_##v;

#define GET_OPT_INT(v) \
  c10::optional<int64_t> int_##v; \
  if (!get_if<void*>(&v)) { \
    auto val_ref_##v = get_if<int64_t>(&v); \
    if (!val_ref_##v) return {}; \
    int_##v = *val_ref_##v; \
  }

#define GET_MEMFORMAT(v) \
  auto val_ref_##v = get_if<at::MemoryFormat>(&v); \
  if (!val_ref_##v) return {}; \
  at::MemoryFormat memformat_##v = *val_ref_##v;

#define GET_OPT_MEMFORMAT(v) \
  c10::optional<at::MemoryFormat> memformat_##v; \
  if (!get_if<void*>(&v)) { \
    auto val_ref_##v = get_if<at::MemoryFormat>(&v); \
    if (!val_ref_##v) return {}; \
    memformat_##v = *val_ref_##v; \
  }

#define GET_OPT_MEMFORMAT_OPT(v) \
  c10::optional<at::MemoryFormat> memformat_##v; \
  if (v && !get_if<void*>(&*v)) { \
    auto val_ref_##v = get_if<at::MemoryFormat>(&*v); \
    if (!val_ref_##v) return {}; \
    memformat_##v = *val_ref_##v; \
  }

c10::optional<Strides> strides_std_promote(const vector<TrailElem> &ops,
                                           ShapeRef out) {
  Strides ret;
  std::vector<std::pair<IntArrayRef, IntArrayRef>> ops_data;
  for (auto elem : ops) {
    if (auto *idx = get_if<uint8_t>(&elem)) {
      auto &[shape, strides] = test_shapes[*idx];
      if (shape.size() > out.size())
        return {};
      ops_data.emplace_back(shape, strides);
    }
  }
  return strides_std_promote(out, ops_data);
}

c10::optional<Strides> strides_contiguous_out(ShapeRef shape) {
  return strides_contiguous(shape);
}

c10::optional<Strides> strides_permute(TrailElem a, TrailElem b) {
  GET_STRIDES(a);
  GET_SHAPE(b);
  return shape_permute(strides_a, shape_b);
}

c10::optional<Strides> strides_view(TrailElem a, ShapeRef out) {
  GET_SHAPES_STRIDES(a);
  return strides_view(shape_a, strides_a, out);
}

c10::optional<Strides> strides_clone(TrailElem a,
                                     c10::optional<TrailElem> b = {}) {
  GET_SHAPES_STRIDES(a);
  GET_OPT_MEMFORMAT_OPT(b);
  return strides_clone(shape_a, strides_a, memformat_b, b.has_value());
}

c10::optional<Strides> strides_clone2(TrailElem a, TrailElem b, TrailElem c) {
  GET_SHAPES_STRIDES(a);
  GET_BOOL(b);
  GET_OPT_MEMFORMAT(c)
  return strides_clone2(shape_a, strides_a, memformat_c, bool_b);
}

c10::optional<Strides> strides_clone_bool(TrailElem a, TrailElem b) {
  GET_SHAPES_STRIDES(a);
  GET_BOOL(b);
  return strides_clone_bool(shape_a, strides_a, bool_b);
}

c10::optional<Strides>
strides_transpose(TrailElem a, TrailElem b, TrailElem c) {
  GET_STRIDES(a);
  GET_INT(b);
  GET_INT(c);
  return shape_transpose(strides_a, int_b, int_c);
}

c10::optional<Strides> strides_expand(TrailElem a, TrailElem b) {
  GET_SHAPES_STRIDES(a);
  GET_SHAPE(b);
  return strides_expand(shape_a, strides_a, shape_b);
}

c10::optional<Strides> strides_slice(TrailElem a, TrailElem b, TrailElem c,
                                     TrailElem d, TrailElem e) {
  GET_STRIDES(a);
  GET_INT(b);
  GET_INT(e);
  return strides_slice(strides_a, int_b, int_e);
}

c10::optional<Strides> strides_flatten(TrailElem a, IntArrayRef out) {
  GET_SHAPES_STRIDES(a);
  return strides_flatten(shape_a, strides_a, out);
}

c10::optional<Strides> strides_select(TrailElem a, TrailElem b) {
  GET_STRIDES(a);
  GET_INT(b);
  return shape_select(strides_a, int_b);
}

c10::optional<Strides> strides_unsqueeze(TrailElem a, TrailElem b) {
  GET_SHAPES_STRIDES(a);
  GET_INT(b);
  return strides_unsqueeze(shape_a, strides_a, int_b);
}

c10::optional<Strides> get_strides(TrailElem a) {
  GET_STRIDES(a);
  return strides_a;
}

#define DECL_UNARY(name) \
c10::optional<Strides> name(TrailElem a) { \
  GET_STRIDES(a); \
  return name(strides_a); \
}

DECL_UNARY(strides_transpose2d);

bool print_all = false;
char *call_only = nullptr;
vector<TrailElem> type_trail;

c10::optional<Strides> all_strides;

array<tuple<const char*, unsigned,
            function<c10::optional<Strides>(ShapeRef)>>, 20> is_strides_fn = {
  make_tuple("ALL", 0, [&](ShapeRef out) { return all_strides; }),
  make_tuple("EQ_FIRST", 1, [&](ShapeRef out) { return get_strides(type_trail[0]); }),
  make_tuple("EQ_SECOND", 2, [&](ShapeRef out) { return get_strides(type_trail[1]); }),
  make_tuple("EQ_THIRD", 3, [&](ShapeRef out) { return get_strides(type_trail[2]); }),
  make_tuple("CONTIGUOUS", 0, [&](ShapeRef out) { return strides_contiguous_out(out); }),
  make_tuple("STD_PROMOTE", 1, [&](ShapeRef out) { return strides_std_promote(type_trail, out); }),
  make_tuple("TRANSPOSE2D", 1, [&](ShapeRef out) { return strides_transpose2d(type_trail[0]); }),
  make_tuple("TRANSPOSE", 3, [&](ShapeRef out) { return strides_transpose(type_trail[0], type_trail[1], type_trail[2]); }),
  make_tuple("PERMUTE", 2, [&](ShapeRef out) { return strides_permute(type_trail[0], type_trail[1]); }),
  make_tuple("VIEW", 1, [&](ShapeRef out) { return strides_view(type_trail[0], out); }),
  make_tuple("CLONE", 2, [&](ShapeRef out) { return strides_clone(type_trail[0], type_trail[1]); }),
  make_tuple("CLONE1", 1, [&](ShapeRef out) { return strides_clone(type_trail[0]); }),
  make_tuple("CLONE2", 4, [&](ShapeRef out) { return strides_clone2(type_trail[0], type_trail[2], type_trail[3]); }),
  make_tuple("CLONE3", 3, [&](ShapeRef out) { return strides_clone2(type_trail[0], true, type_trail[2]); }),
  make_tuple("CLONE_BOOL", 2, [&](ShapeRef out) { return strides_clone_bool(type_trail[0], type_trail[1]); }),
  make_tuple("EXPAND", 2, [&](ShapeRef out) { return strides_expand(type_trail[0], type_trail[1]); }),
  make_tuple("SLICE", 5, [&](ShapeRef out) { return strides_slice(type_trail[0], type_trail[1], type_trail[2], type_trail[3], type_trail[4]); }),
  make_tuple("FLATTEN", 3, [&](ShapeRef out) { return strides_flatten(type_trail[0], out); }),
  make_tuple("SELECT", 2, [&](ShapeRef out) { return strides_select(type_trail[0], type_trail[1]); }),
  make_tuple("UNSQUEEZE", 2, [&](ShapeRef out) { return strides_unsqueeze(type_trail[0], type_trail[1]); }),
};

array<bool, is_strides_fn.size()> is_strides_flags;
unsigned num_samples = 0;

void print(ShapeRef shape, StridesRef strides) {
  bool first = true;
  for (auto input : type_trail) {
    if (!first)
      cout << ", ";
    first = false;
    if (auto *idx = get_if<uint8_t>(&input)) {
      auto &[shape, strides] = test_shapes[*idx];
      cout << ShapeRef(shape) << " / " << StridesRef(strides);
    } else if (auto *v = get_if<int64_t>(&input)) {
      cout << *v;
    } else if (auto *v = get_if<bool>(&input)) {
      cout << *v;
    } else if (auto *v = get_if<at::MemoryFormat>(&input)) {
      cout << *v;
    } else if (get_if<void*>(&input)) {
      cout << "(null)";
    } else {
      assert(0);
    }
  }
  cout << " -> " << shape << " / " << strides << endl;
}


void infer(ShapeRef shape, StridesRef strides) {
  ++num_samples;

  if (!all_strides)
    all_strides = strides.vec();

  auto num_args = type_trail.size();

  for (unsigned i = 0; i < is_strides_fn.size(); ++i) {
    auto &f = is_strides_flags[i];
    if (!f)
      continue;

    bool n_args = num_args >= get<1>(is_strides_fn[i]);
    auto fn_out = n_args ? get<2>(is_strides_fn[i])(shape) : c10::nullopt;
    f = n_args && fn_out && strides == *fn_out;

    if (print_all && !f) {
      cout << "FAILED: " << get<0>(is_strides_fn[i]);
      if (n_args && fn_out) {
        cout << " / EXPECTING: " << StridesRef(*fn_out);
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
      auto strides = tensor.strides();

      if (print_all)
        print(shape, strides);
      infer(shape, strides);
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
    for (auto ty : { kFloat, kShort, kBool }) {
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
    type_trail.emplace_back(nullptr);
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
    type_trail.emplace_back(nullptr);
    call(function<Tensor(Tail...)>{[=](Tail... args) -> Tensor {
      return fn(c10::nullopt, forward<Tail>(args)...);
    }});
    type_trail.pop_back();
  }

  template <typename... Tail>
  void call(function<Tensor(c10::optional<at::ScalarType>, Tail...)> fn) {
    // call with a value
    call(function<Tensor(at::ScalarType, Tail...)>{
      [=](at::ScalarType val, Tail... args) -> Tensor {
        return fn(val, args...);
      }});

    // and call without a value
    call(function<Tensor(Tail...)>{[=](Tail... args) -> Tensor {
      return fn(c10::nullopt, forward<Tail>(args)...);
    }});
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
    for (int64_t v : {0, 1, 2}) {
      type_trail.emplace_back(v);
      call(function<Tensor(Tail...)>{[=](Tail... args) -> Tensor {
        return fn(v, args...);
      }});
      type_trail.pop_back();
    }
  }

  template <typename... Tail>
  void call(function<Tensor(bool, Tail...)> fn) {
    for (bool v : {false, true}) {
      type_trail.emplace_back(v);
      call(function<Tensor(Tail...)>{[=](Tail... args) -> Tensor {
        return fn(v, args...);
      }});
      type_trail.pop_back();
    }
  }

  template <typename... Tail>
  void call(function<Tensor(IntArrayRef, Tail...)> fn) {
    for (uint8_t shape = 0; shape < test_shapes.size(); ++shape) {
      type_trail.emplace_back(shape);
      call(function<Tensor(Tail...)>{[=](Tail... args) -> Tensor {
        return fn(test_shapes[shape].first, args...);
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

  template <typename... Tail>
  void call(function<Tensor(at::MemoryFormat, Tail...)> fn) {
    for (auto v : {at::MemoryFormat::Preserve, at::MemoryFormat::Contiguous}) {
      type_trail.emplace_back(v);
      call(function<Tensor(Tail...)>{[=](Tail... args) -> Tensor {
        return fn(v, args...);
      }});
      type_trail.pop_back();
    }
  }

  template <typename T>
  void analyze(function<T> fn) {
    if (call_only && strcmp(name, call_only))
      return;

    for (auto &f : is_strides_flags) {
      f = true;
    }
    all_strides.reset();

    call(move(fn));

    cout << name;

    if (num_samples == 0) {
      cout << ": NO_SAMPLES\n";
      return;
    }
    if (is_strides_flags[0]) {
      cout << ": ALL " << ShapeRef(*all_strides) << '\n';
      return;
    }
    for (unsigned i = 1; i < is_strides_flags.size(); ++i) {
      if (is_strides_flags[i]) {
        cout << ": " << get<0>(is_strides_fn[i]) << '\n';
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
