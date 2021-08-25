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
#include <unistd.h>

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
};

struct Result {
  vector<InputEntry> inputs;
  ScalarType default_dtype;
  ScalarType output;
};

char *call_only = nullptr;
vector<InputEntry> type_trail;
vector<Result> results;

Tensor new_tensor(IntArrayRef shape, ScalarType ty) {
  auto t = native::empty_cpu(shape, ty);
  native::ones_out(shape, t);
  return t;
}

void print(const Result &result) {
  bool first = true;
  for (auto input : result.inputs) {
    if (!first)
      cout << ", ";
    cout << input.ty;
    if (input.zerodim)
      cout << " [z]";
    first = false;
  }
  cout << ", default=" << result.default_dtype << " -> " << result.output;
}

#define ARG(n) args[n].ty, [&]() { return args[n].zerodim; }

#define callVA(f) \
  switch (args.size()) { \
  case 1:  return f(ARG(0));\
  case 2:  return f(ARG(0), ARG(1));\
  case 3:  return f(ARG(0), ARG(1), ARG(2));\
  case 4:  return f(ARG(0), ARG(1), ARG(2), ARG(3));\
  case 5:  return f(ARG(0), ARG(1), ARG(2), ARG(3), ARG(4));\
  default: cout << "ERROR: Too many args!" << endl; _Exit(-1); \
  }

#include "../type_inference.h"

ScalarType promoted_type_trail(const vector<InputEntry> &args) {
  callVA(promote_tys)
}

ScalarType promoted_type_trail_const(const vector<InputEntry> &args) {
  callVA(promote_const)
}

ScalarType promoted_type_trail_buggy(const vector<InputEntry> &args) {
  callVA(promote_buggy)
}

optional<ScalarType> to_optional(ScalarType ty) {
  if (ty == ScalarType::Undefined)
    return {};
  return ty;
}

ScalarType optional_or_default(ScalarType ty) {
  if (ty == ScalarType::Undefined)
    return typeMetaToScalarType(at::get_default_dtype());
  return ty;
}

struct C {
  const char *name;

  void call(function<Tensor()> fn) {
    try {
      for (auto def : { kFloat, kDouble }) {
        at::set_default_dtype(scalarTypeToTypeMeta(def));
        auto output = typeMetaToScalarType(fn().dtype());
        results.push_back({type_trail, def, output});
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
        call(function<Tensor(Tail&&...)>{[=](Tail&&... args) -> Tensor {
          auto t = new_tensor(zerodim ? zero_sz : nonzero_sz, ty);
          return fn(t, args...);
        }});
        type_trail.pop_back();
      }
    }
  }

  template <typename... Tail>
  void call(function<Tensor(ScalarType&, Tail...)> fn) {
    for (auto ty : types) {
      if (isQIntType(ty))
        continue;
      type_trail.push_back({ty, false});
      call(function<Tensor(Tail&&...)>{[=](Tail&&... args) -> Tensor {
        auto ty_cpy = ty;
        return fn(ty_cpy, args...);
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
    type_trail.push_back({ScalarType::Undefined, false});
    call(function<Tensor(Tail&&...)>{[=](Tail&&... args) -> Tensor {
      c10::optional<T> opt;
      return fn(opt, forward<Tail>(args)...);
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
        call(function<Tensor(Tail&&...)>{[=](Tail&&... args) -> Tensor {
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
        call(function<Tensor(Tail&&...)>{[=](Tail&&... args) -> Tensor {
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
      type_trail.push_back({ty, false});
      auto n = results.size();
      for (auto v : tries) {
        call(function<Tensor(Tail&&...)>{[=](Tail&&... args) -> Tensor {
          return fn(const_cast<Scalar&>(v), args...);
        }});
        if (results.size() > n)
          break;
      }
      type_trail.pop_back();
    };

    test(kBool, {false, true});
    test(kLong, {0, 1});
    test(kDouble, {0.0, 1.0});
    test(kComplexDouble, {c10::complex<double>(0.0),c10::complex<double>(1.0)});
  }

  template <typename... Tail>
  void call(function<Tensor(IntArrayRef&, Tail...)> fn) {
    call(function<Tensor(Tail&&...)>{[=](Tail&&... args) -> Tensor {
      IntArrayRef s;
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
    bool eq_promoted = true;
    bool eq_promoted_const = true;
    bool eq_promoted_buggy = true;
    bool eq_first = true;
    bool eq_second = true;
    bool eq_third = true;
    bool eq_fourth = true;
    bool is_value = true;
    bool is_to_float = true;
    bool is_to_double = true;
    bool is_to_double2 = true;
    bool is_to_float2 = true;
    bool is_to_float3 = true;
    bool is_to_float4 = true;
    bool is_to_fdouble = true;
    bool is_to_real_float = true;
    bool is_to_real2 = true;
    bool is_to_complex = true;
    bool is_bool2int = true;
    bool is_bool2int2 = true;
    bool is_boolbyte = true;
    bool is_integral2int = true;
    bool is_to_qint = true;
    bool is_optional_or21 = true;
    bool is_optional_or21long = true;
    bool is_1_or_default = true;
    bool is_1_or_long = true;

    for (auto &result : results) {
      auto &type_trail = result.inputs;
      auto &type = result.output;
      at::set_default_dtype(scalarTypeToTypeMeta(result.default_dtype));

#define PASS(arg) \
  arg.ty, [&]() { return arg.zerodim; }

#define DEBUG(what)                 \
  if (what != type) {               \
    cout << "WRONG: ";              \
    print(result);                  \
    cout << " vs " << what << endl; \
  }

      all_equal        &= type == results[0].output;
      eq_promoted      &= type == promoted_type_trail(type_trail);
      eq_promoted_const&= type == promoted_type_trail_const(type_trail);
      eq_promoted_buggy&= type == promoted_type_trail_buggy(type_trail);
      eq_first         &= type == type_trail[0].ty;
      eq_second        &= type_trail.size() >= 2 && type == type_trail[1].ty;
      eq_third         &= type_trail.size() >= 3 && type == type_trail[2].ty;
      eq_fourth        &= type_trail.size() >= 4 && type == type_trail[3].ty;
      is_value         &= toValueType(type_trail[0].ty) == type;
      is_to_float      &= to_float(type_trail[0].ty) == type;
      is_to_double     &= to_double(type_trail[0].ty) == type;
      is_to_double2    &= type_trail.size() >= 2 &&
                          to_double2(type_trail[0].ty,
                                     type_trail[1].ty) == type;
      is_to_float2     &= type_trail.size() >= 2 &&
                          to_float2(PASS(type_trail[0]),
                                    PASS(type_trail[1])) == type;
      is_to_float3     &= type_trail.size() >= 3 &&
                          to_float3(PASS(type_trail[0]), PASS(type_trail[1]),
                                    PASS(type_trail[2])) == type;
      is_to_float4     &= type_trail.size() >= 4 &&
                          to_float4(PASS(type_trail[0]),
                                    PASS(type_trail[1]),
                                    PASS(type_trail[2]),
                                    PASS(type_trail[3])) == type;
      is_to_fdouble    &= type_trail.size() >= 2 &&
                          to_float_double(type_trail[1].ty) == type;
      is_to_real_float &= to_real_float(type_trail[0].ty) == type;
      is_to_real2      &= type_trail.size() >= 2 &&
                          to_real2(PASS(type_trail[0]),
                                   PASS(type_trail[1])) == type;
      is_to_complex    &= to_complex(type_trail[0].ty) == type;
      is_bool2int      &= bool_to_int(type_trail[0].ty) == type;
      is_bool2int2     &= type_trail.size() >= 2 &&
                          bool_to_int2(PASS(type_trail[0]),
                                       PASS(type_trail[1])) == type;
      is_boolbyte      &= bool_byte(type_trail[0].ty) == type;
      is_integral2int  &= integrals_to_int(type_trail[0].ty) == type;
      is_to_qint       &= toQIntType(type_trail[0].ty) == type;
      is_optional_or21 &= type_trail.size() >= 2 &&
                          optional_or_else(to_optional(type_trail[1].ty),
                                           type_trail[0].ty) == type;
      is_optional_or21long &= type_trail.size() >= 2 &&
                             optional_or_longelse(to_optional(type_trail[1].ty),
                                                  type_trail[0].ty) == type;
      is_1_or_default  &= optional_or_default(type_trail[0].ty) == type;
      is_1_or_long     &= optional_or_else(to_optional(type_trail[0].ty), kLong)
                            == type;
    }

    cout << name;

    if (results.empty()) {
      cout << ": NO_SAMPLES" << endl;
      return;
    }
    if (all_equal) {
      cout << ": ALL " << results[0].output << endl;
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
    PRINT(eq_fourth, "EQ_FOURTH")
    PRINT(eq_promoted, "EQ_PROMOTED")
    PRINT(eq_promoted_const, "EQ_PROMOTED_CONST")
    PRINT(eq_promoted_buggy, "EQ_PROMOTED_BUGGY")
    PRINT(is_value, "TO_VALUE_TYPE")
    PRINT(is_to_float, "TO_FLOAT")
    PRINT(is_to_double, "TO_DOUBLE")
    PRINT(is_to_double2, "TO_DOUBLE2")
    PRINT(is_to_float2, "TO_FLOAT2")
    PRINT(is_to_float3, "TO_FLOAT3")
    PRINT(is_to_float4, "TO_FLOAT4")
    PRINT(is_to_fdouble, "TO_FLOAT_DOUBLE")
    PRINT(is_to_real_float, "TO_REAL_FLOAT")
    PRINT(is_to_real2, "TO_REAL2")
    PRINT(is_to_complex, "TO_COMPLEX")
    PRINT(is_bool2int, "BOOL2INT")
    PRINT(is_bool2int2, "BOOL2INT2")
    PRINT(is_boolbyte, "BOOLBYTE")
    PRINT(is_integral2int, "INTEGRAL2INT")
    PRINT(is_to_qint, "TO_QINT")
    PRINT(is_optional_or21, "OPTIONAL_OR21")
    PRINT(is_optional_or21long, "OPTIONAL_O21LONG")
    PRINT(is_1_or_default, "FIRST_OR_DEFAULT")
    PRINT(is_1_or_long, "FIRST_OR_LONG")

    cout << ": NON_STANDARD:\n";

    for (auto &result : results) {
      print(result);

      if (promoted_type_trail(result.inputs) != result.output) {
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

#include "call_pytorch_fns.h"

  _Exit(0);
  return 0;
}
