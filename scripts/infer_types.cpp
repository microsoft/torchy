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

using namespace at;
using namespace c10;
using namespace std;

namespace {
array<ScalarType, NumScalarTypes-1> types = {
#define DECL(_, x) ScalarType::x,
  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DECL)
#undef DECL
};

struct InputEntry {
  ScalarType ty;
  bool zerodim;
  bool is_scalar = false;
};

bool print_all = false;
char *call_only = nullptr;
vector<InputEntry> type_trail;

Tensor new_tensor(IntArrayRef shape, ScalarType ty) {
  auto t = native::empty_cpu(shape, ty);
  native::ones_out(shape, t);
  return t;
}

void print(ScalarType default_dtype, ScalarType output) {
  bool first = true;
  for (auto input : type_trail) {
    if (!first)
      cout << ", ";
    cout << input.ty;
    if (input.zerodim)
      cout << " [z]";
    if (input.is_scalar)
      cout << " [s]";
    first = false;
  }
  cout << ", default=" << default_dtype << " -> " << output << endl;
}

#define ARG(n) args[n].ty, args[n].is_scalar,\
  function<bool()>([&]() { return args[n].zerodim; })

#define callVA(f, max_elems) \
  switch (min(args.size(), max_elems)) { \
  case 1:  return f(ARG(0));\
  case 2:  return f(ARG(0), ARG(1));\
  case 3:  return f(ARG(0), ARG(1), ARG(2));\
  case 4:  return f(ARG(0), ARG(1), ARG(2), ARG(3));\
  case 5:  return f(ARG(0), ARG(1), ARG(2), ARG(3), ARG(4));\
  default: cout << "ERROR: Too many args!" << endl; exit(-1); \
  }

#include "../type_inference.h"

ScalarType promoted_type_trail(const vector<InputEntry> &args) {
  callVA(promote_tys, (size_t)-1ull)
}

ScalarType promoted_type_trail_const(const vector<InputEntry> &args) {
  callVA(promote_const, (size_t)-1ull)
}

ScalarType promoted_type_trail_buggy(const vector<InputEntry> &args,
                                     size_t max_elems = -1ull) {
  callVA(promote_buggy, max_elems)
}

c10::optional<ScalarType> to_optional(ScalarType ty) {
  if (ty == ScalarType::Undefined)
    return {};
  return ty;
}

ScalarType optional_or_default(ScalarType ty) {
  if (ty == ScalarType::Undefined)
    return typeMetaToScalarType(at::get_default_dtype());
  return ty;
}

#define PASS(arg) arg.ty, [&]() { return arg.zerodim; }
#define PASSF(arg) arg.ty, arg.is_scalar, [&]() { return arg.zerodim; }

std::optional<ScalarType> all_type;

array<tuple<const char*, unsigned, function<ScalarType()>>, 30> is_type_fn = {
  make_tuple("ALL", 1, [&]() { return *all_type; }),
  make_tuple("EQ_FIRST", 1, [&]() { return type_trail[0].ty; }),
  make_tuple("EQ_SECOND", 2, [&]() { return type_trail[1].ty; }),
  make_tuple("EQ_THIRD", 3, [&]() { return type_trail[2].ty; }),
  make_tuple("EQ_FOURTH", 4, [&]() { return type_trail[3].ty; }),
  make_tuple("EQ_PROMOTED", 1, [&]() { return promoted_type_trail(type_trail); }),
  make_tuple("EQ_PROMOTED_CONST", 1, [&]() { return promoted_type_trail_const(type_trail); }),
  make_tuple("EQ_PROMOTED_BUGGY", 1, [&]() { return promoted_type_trail_buggy(type_trail); }),
  make_tuple("EQ_PROMOTED_BUGGY2", 2, [&]() { return promoted_type_trail_buggy(type_trail, 2); }),
  make_tuple("TO_VALUE_TYPE", 1, [&]() { return toValueType(type_trail[0].ty); }),
  make_tuple("TO_FLOAT", 1, [&]() { return to_float(type_trail[0].ty); }),
  make_tuple("TO_FLOAT2", 2, [&]() { return to_float2(PASSF(type_trail[0]), PASSF(type_trail[1])); }),
  make_tuple("TO_FLOAT3", 3, [&]() { return to_float3(PASS(type_trail[0]), PASS(type_trail[1]), PASS(type_trail[2])); }),
  make_tuple("TO_FLOAT4", 4, [&]() { return to_float4(PASS(type_trail[0]), PASS(type_trail[1]), PASS(type_trail[2]), PASS(type_trail[3])); }),
  make_tuple("TO_DOUBLE2", 2, [&]() { return to_double2(type_trail[0].ty, type_trail[1].ty); }),
  make_tuple("TO_FLOAT_DOUBLE", 2, [&]() { return to_float_double(type_trail[1].ty); }),
  make_tuple("TO_REAL_FLOAT", 1, [&]() { return to_real_float(type_trail[0].ty); }),
  make_tuple("TO_REAL2", 2, [&]() { return to_real2(PASS(type_trail[0]), PASS(type_trail[1])); }),
  make_tuple("TO_COMPLEX", 1, [&]() { return to_complex(type_trail[0].ty); }),
  make_tuple("BOOL2INT", 1, [&]() { return bool_to_int(type_trail[0].ty); }),
  make_tuple("BOOLBYTE", 1, [&]() { return bool_byte(type_trail[0].ty); }),
  make_tuple("INTEGRAL2INT", 1, [&]() { return integrals_to_int(type_trail[0].ty); }),
  make_tuple("TO_QINT", 1, [&]() { return toQIntType(type_trail[0].ty); }),
  make_tuple("OPTIONAL_OR21", 2, [&]() { return optional_or_else(to_optional(type_trail[1].ty), type_trail[0].ty); }),
  make_tuple("OPTIONAL_OR31", 3, [&]() { return optional_or_else(to_optional(type_trail[2].ty), type_trail[0].ty); }),
  make_tuple("OPTIONAL_O21LONG", 2, [&]() { return optional_or_longelse(to_optional(type_trail[1].ty), type_trail[0].ty); }),
  make_tuple("FIRST_OR_DEFAULT", 1, [&]() { return optional_or_default(type_trail[0].ty); }),
  make_tuple("SECOND_OR_DEFAULT", 2, [&]() { return optional_or_default(type_trail[1].ty); }),
  make_tuple("FIRST_OR_LONG", 1, [&]() { return optional_or_else(to_optional(type_trail[0].ty), kLong); }),
  make_tuple("SECOND_OR_LONG_DEFAULT", 2, [&]() { return optional_or_longdefault(to_optional(type_trail[1].ty), type_trail[0].ty); }),
};

array<bool, is_type_fn.size()> is_type_flags;
unsigned num_samples = 0;

void infer(ScalarType output) {
  ++num_samples;

  if (!all_type)
    all_type = output;

  auto num_args = type_trail.size();

  for (unsigned i = 0; i < is_type_fn.size(); ++i) {
    auto &f = is_type_flags[i];
    if (!f)
      continue;
    f = num_args >= get<1>(is_type_fn[i]) && get<2>(is_type_fn[i])() == output;
    if (print_all && !f) {
      cout << "FAILED: " << get<0>(is_type_fn[i]);
      if (num_args >= get<1>(is_type_fn[i]))
        cout << " / EXPECTING: " << get<2>(is_type_fn[i])();
      cout << '\n';
    }
  }
}

struct C {
  const char *name;

  void call(function<Tensor()> fn) {
    try {
      for (auto def : { kFloat, kDouble }) {
        at::set_default_dtype(scalarTypeToTypeMeta(def));
        auto output = typeMetaToScalarType(fn().dtype());

        if (print_all)
          print(def, output);
        infer(output);
      }
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
    ArrayRef<long> zero_sz = {};
    ArrayRef<long> nonzero_sz = {1};

    for (auto ty : types) {
      if (isQIntType(ty))
        continue;
      for (bool zerodim : { false, true }) {
        type_trail.push_back({ty, zerodim});
        call(function<Tensor(Tail...)>{[=](Tail... args) -> Tensor {
          auto t = new_tensor(zerodim ? zero_sz : nonzero_sz, ty);
          return fn(t, args...);
        }});
        type_trail.pop_back();
      }
    }
  }

  template <typename... Tail>
  void call(function<Tensor(ScalarType, Tail...)> fn) {
    for (auto ty : types) {
      if (isQIntType(ty))
        continue;
      type_trail.push_back({ty, false});
      call(function<Tensor(Tail...)>{[=](Tail... args) -> Tensor {
        return fn(ty, args...);
      }});
      type_trail.pop_back();
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
    type_trail.push_back({ScalarType::Undefined, false});
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
    type_trail.push_back({ScalarType::Undefined, false});
    call(function<Tensor(Tail...)>{[=](Tail... args) -> Tensor {
      return fn(c10::nullopt, forward<Tail>(args)...);
    }});
    type_trail.pop_back();
  }

  template <typename... Tail>
  void call(function<Tensor(TensorList&, Tail...)> fn) {
    for (auto ty1 : types) {
      if (isQIntType(ty1))
        continue;
      type_trail.push_back({ty1, false});

      for (auto ty2 : types) {
        if (isQIntType(ty2))
          continue;
        type_trail.push_back({ty2, false});
        call(function<Tensor(Tail...)>{[=](Tail... args) -> Tensor {
          Tensor ts[2] = { new_tensor({1}, ty1),
                           new_tensor({1}, ty2) };
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
    for (auto ty1 : types) {
      if (isQIntType(ty1))
        continue;
      type_trail.push_back({ty1, false});

      for (auto ty2 : types) {
        if (isQIntType(ty2))
          continue;
        type_trail.push_back({ty2, false});
        call(function<Tensor(Tail...)>{[=](Tail... args) -> Tensor {
          List<c10::optional<Tensor>> list({ new_tensor({1}, ty1),
                                             new_tensor({1}, ty2) });
          return fn(list, args...);
        }});
        type_trail.pop_back();
      }
      type_trail.pop_back();
    }
  }

  template <typename... Tail>
  void call(function<Tensor(at::Scalar&, Tail...)> fn) {
    auto test = [&](ScalarType ty, initializer_list<at::Scalar> tries) {
      type_trail.push_back({ty, false, true});
      auto n = num_samples;
      for (auto v : tries) {
        call(function<Tensor(Tail...)>{[=](Tail... args) -> Tensor {
          return fn(const_cast<Scalar&>(v), args...);
        }});
        if (num_samples > n)
          break;
      }
      type_trail.pop_back();
    };

    test(kBool, {false, true});
    test(kShort, {0, 1});
    test(kLong, {0, 1});
    test(kFloat, {0.0, 1.0});
    test(kDouble, {0.0, 1.0});
    test(kComplexFloat, {c10::complex<float>(0.0), c10::complex<float>(1.0)});
    test(kComplexDouble, {c10::complex<double>(0.0),c10::complex<double>(1.0)});
  }

  template <typename... Tail>
  void call(function<Tensor(int64_t, Tail...)> fn) {
    auto n = num_samples;
    for (int64_t v : {0, 1}) {
      call(function<Tensor(Tail...)>{[=](Tail... args) -> Tensor {
        return fn(v, args...);
      }});
      if (num_samples > n)
        break;
    }
  }

  template <typename... Tail>
  void call(function<Tensor(bool, Tail...)> fn) {
    auto n = num_samples;
    for (bool v : {false, true}) {
      call(function<Tensor(Tail...)>{[=](Tail... args) -> Tensor {
        return fn(v, args...);
      }});
      if (num_samples > n)
        break;
    }
  }

  template <typename... Tail>
  void call(function<Tensor(IntArrayRef, Tail...)> fn) {
    call(function<Tensor(Tail...)>{[=](Tail... args) -> Tensor {
      return fn({}, args...);
    }});
  }


  template <typename... Tail>
  void call(function<Tensor(c10::optional<at::MemoryFormat>, Tail...)> fn) {
    call(function<Tensor(Tail...)>{[=](Tail... args) -> Tensor {
      return fn(c10::nullopt, args...);
    }});
  }

  template <typename T>
  void analyze(function<T> fn) {
    if (call_only && strcmp(name, call_only))
      return;

    for (auto &f : is_type_flags) {
      f = true;
    }
    all_type.reset();

    call(move(fn));

    cout << name;

    if (num_samples == 0) {
      cout << ": NO_SAMPLES\n";
      return;
    }
    if (is_type_flags[0]) {
      cout << ": ALL " << *all_type << '\n';
      return;
    }
    for (unsigned i = 1; i < is_type_flags.size(); ++i) {
      if (is_type_flags[i]) {
        cout << ": " << get<0>(is_type_fn[i]) << '\n';
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

  SetStackTraceFetcher([]() { return string(); });

#include "call_pytorch_fns.h"

  return 0;
}
