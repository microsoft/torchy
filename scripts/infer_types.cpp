// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

#include <ATen/core/List.h>
#include <ATen/NativeFunctions.h>
#include <ATen/RedispatchFunctions.h>
#include <array>
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

char *call_only = nullptr;
vector<ScalarType> type_trail;
vector<pair<vector<ScalarType>, ScalarType>> results;

void print(const vector<ScalarType> &type_trail) {
  bool first = true;
  for (auto ty : type_trail) {
    if (!first)
      cout << ", ";
    cout << ty;
    first = false;
  }
}

ScalarType promoted_type_trail(const vector<ScalarType> &type_trail) {
  auto ty = type_trail[0];
  for (auto ty2 : type_trail) {
    if (ty2 != ScalarType::Undefined)
      ty = promoteTypes(ty, ty2);
  }
  return ty;
}

#include "../type_inference.h"

struct C {
  const char *name;

  void call(function<Tensor()> fn) {
    try {
      auto dtype = typeMetaToScalarType(fn().dtype());
      results.emplace_back(type_trail, dtype);
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
    for (auto ty : types) {
      if (isQIntType(ty))
        continue;
      type_trail.emplace_back(ty);
      call(function<Tensor(Tail&&...)>{[=](Tail&&... args) -> Tensor {
        auto t = native::empty_cpu({1}, ty);
        return fn(t, args...);
      }});
      type_trail.pop_back();
    }
  }

  template <typename T, typename... Tail>
  void call(function<Tensor(optional<T>&, Tail...)> fn) {
    // call with a value
    call(function<Tensor(T&, Tail&&...)>{
      [=](T &val, Tail&&... args) -> Tensor {
        optional<T> opt(val);
        return fn(opt, args...);
      }});

    // and call without a value
    type_trail.emplace_back(ScalarType::Undefined);
    call(function<Tensor(Tail&&...)>{[=](Tail&&... args) -> Tensor {
      optional<T> opt;
      return fn(opt, forward<Tail>(args)...);
    }});
    type_trail.pop_back();
  }

  template <typename... Tail>
  void call(function<Tensor(TensorList&, Tail...)> fn) {
    for (auto ty1 : types) {
      if (isQIntType(ty1))
        continue;
      type_trail.emplace_back(ty1);

      for (auto ty2 : types) {
        if (isQIntType(ty2))
          continue;
        type_trail.emplace_back(ty2);
        call(function<Tensor(Tail&&...)>{[=](Tail&&... args) -> Tensor {
          Tensor ts[2] = { native::empty_cpu({1}, ty1),
                           native::empty_cpu({1}, ty2) };
          ArrayRef<Tensor> aref(ts);
          return fn(aref, args...);
        }});
        type_trail.pop_back();
      }
      type_trail.pop_back();
    }
  }

  template <typename... Tail>
  void call(function<Tensor(List<optional<Tensor>>&, Tail...)> fn) {
    for (auto ty1 : types) {
      if (isQIntType(ty1))
        continue;
      type_trail.emplace_back(ty1);

      for (auto ty2 : types) {
        if (isQIntType(ty2))
          continue;
        type_trail.emplace_back(ty2);
        call(function<Tensor(Tail&&...)>{[=](Tail&&... args) -> Tensor {
          List<optional<Tensor>> list({ native::empty_cpu({1}, ty1),
                                        native::empty_cpu({1}, ty2) });
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
      type_trail.emplace_back(ty);
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

  template <typename T>
  void analyze(function<T> fn) {
    if (call_only && strcmp(name, call_only))
      return;

    results.clear();
    call(move(fn));

    bool all_equal = true;
    bool eq_promoted = true;
    bool eq_first = true;
    bool eq_second = true;
    bool eq_third = true;
    bool is_value = true;
    bool is_to_float = true;
    bool is_to_double = true;
    bool is_to_double2 = true;
    bool is_to_float2 = true;
    bool is_to_float2_2 = true;
    bool is_to_float2_3 = true;
    bool is_to_float2_4 = true;
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

    for (auto &[type_trail, type] : results) {
      all_equal        &= type == results[0].second;
      eq_promoted      &= type == promoted_type_trail(type_trail);
      eq_first         &= type == type_trail[0];
      eq_second        &= type_trail.size() >= 2 && type == type_trail[1];
      eq_third         &= type_trail.size() >= 3 && type == type_trail[2];
      is_value         &= toValueType(type_trail[0]) == type;
      is_to_float      &= to_float(type_trail[0]) == type;
      is_to_double     &= to_double(type_trail[0]) == type;
      is_to_double2    &= type_trail.size() >= 2 &&
                          to_double2(type_trail[0], type_trail[1]) == type;
      is_to_float2     &= type_trail.size() >= 2 &&
                          to_float2(type_trail[0], type_trail[1]) == type;
      is_to_float2_2   &= type_trail.size() >= 2 &&
                          to_float2_2(type_trail[0], type_trail[1]) == type;
      is_to_float2_3   &= type_trail.size() >= 2 &&
                          to_float2_3(type_trail[0], type_trail[1]) == type;
      is_to_float2_4   &= type_trail.size() >= 2 &&
                          to_float2_4(type_trail[0], type_trail[1]) == type;
      is_to_float3     &= type_trail.size() >= 3 &&
                          to_float3(type_trail[0], type_trail[1], type_trail[2])
                            == type;
      is_to_float4     &= type_trail.size() >= 4 &&
                          to_float4(type_trail[0], type_trail[1], type_trail[2],
                                    type_trail[3]) == type;
      is_to_fdouble    &= type_trail.size() >= 2 &&
                          to_float_double(type_trail[1]) == type;
      is_to_real_float &= to_real_float(type_trail[0]) == type;
      is_to_real2      &= type_trail.size() >= 2 &&
                          to_real2(type_trail[0], type_trail[1]) == type;
      is_to_complex    &= to_complex(type_trail[0]) == type;
      is_bool2int      &= bool_to_int(type_trail[0]) == type;
      is_bool2int2     &= type_trail.size() >= 2 &&
                          bool_to_int2(type_trail[0], type_trail[1]) == type;
      is_boolbyte      &= bool_byte(type_trail[0]) == type;
      is_integral2int  &= integrals_to_int(type_trail[0]) == type;
      is_to_qint       &= toQIntType(type_trail[0]) == type;
    }

    cout << name;

    if (results.empty()) {
      cout << ": NO_SAMPLES" << endl;
      return;
    }
    if (all_equal) {
      cout << ": ALL " << results[0].second << endl;
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
    PRINT(eq_promoted, "EQ_PROMOTED")
    PRINT(is_value, "TO_VALUE_TYPE")
    PRINT(is_to_float, "TO_FLOAT")
    PRINT(is_to_double, "TO_DOUBLE")
    PRINT(is_to_double2, "TO_DOUBLE2")
    PRINT(is_to_float2, "TO_FLOAT2")
    PRINT(is_to_float2_2, "TO_FLOAT2_2")
    PRINT(is_to_float2_3, "TO_FLOAT2_3")
    PRINT(is_to_float2_4, "TO_FLOAT2_4")
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

    cout << ": NON_STANDARD:" << endl;

    for (auto &[type_trail, type] : results) {
      print(type_trail);
      cout << " -> " << type;

      if (promoted_type_trail(type_trail) != type) {
        cout << " [non-standard promotion]\n";
      } else {
        cout << '\n';
      }
    }
    cout << '\n';
  }

};

}

int main(int argc, char **argv) {
  if (argc > 1) {
    call_only = argv[1];
  }
  results.reserve(1024 * 1024);

#include "call_pytorch_fns.h"

  return 0;
}
