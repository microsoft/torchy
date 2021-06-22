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

ScalarType promoted_type_trail(const vector<ScalarType> &type_trail) {
  auto ty = type_trail[0];
  for (auto ty2 : type_trail) {
    ty = promoteTypes(ty, ty2);
  }
  return ty;
}

ScalarType to_float(ScalarType ty) {
  switch (ty) {
  case ScalarType::Half:
  case ScalarType::Double:
  case ScalarType::ComplexFloat:
  case ScalarType::ComplexDouble:
  case ScalarType::BFloat16:
    return ty;
  default:
    return ScalarType::Float;
  }
}

ScalarType to_double(ScalarType ty) {
  if (ty == ScalarType::Float)
    return ty;
  return ScalarType::Double;
}

ScalarType to_float2(ScalarType ty, ScalarType ty2) {
  if (isIntegralType(ty, true) && isIntegralType(ty2, true))
    return ScalarType::Float;
  return promoteTypes(ty, ty2);
}

ScalarType to_real_float(ScalarType ty) {
  switch (ty) {
  case ScalarType::Half:
  case ScalarType::Double:
  case ScalarType::BFloat16:
    return ty;
  case ScalarType::ComplexDouble:
    return ScalarType::Double;
  default:
    return ScalarType::Float;
  }
}

ScalarType to_complex(ScalarType ty) {
  switch (ty) {
  case ScalarType::Half:
    return ScalarType::ComplexHalf;
  case ScalarType::Float:
    return ScalarType::ComplexFloat;
  case ScalarType::Double:
    return ScalarType::ComplexDouble;
  default:
    return ty;
  }
}

ScalarType bool_to_int(ScalarType ty) {
  if (ty == ScalarType::Bool)
    return ScalarType::Long;
  return ty;
}

struct C {
  const char *name;

  void call(function<Tensor()> fn) {
    try {
      auto dtype = typeMetaToScalarType(fn().dtype());
      results.emplace_back(type_trail, dtype);
    } catch (const c10::Error &) {
      //cout << "Exception!\n";
    } catch (const runtime_error &) {
      //cout << "Runtime Exception!\n";
    }
  }

  template <typename... Tail>
  void call(function<Tensor(Tensor&, Tail...)> fn) {
    for (auto ty : types) {
      if (isQIntType(ty))
        continue;
      type_trail.emplace_back(ty);
      call(function<Tensor(Tail&&...)>{[=](Tail&&... args) -> Tensor {
        auto t = native::empty_cpu({1}, ty);
        return fn(t, forward<Tail>(args)...);
      }});
      type_trail.pop_back();
    }
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
          return fn(aref, forward<Tail>(args)...);
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
          return fn(list, forward<Tail>(args)...);
        }});
        type_trail.pop_back();
      }
      type_trail.pop_back();
    }
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
    bool is_value = true;
    bool is_to_float = true;
    bool is_to_float2 = true;
    bool is_to_float3 = true;
    bool is_to_double2 = true;
    bool is_to_real_float = true;
    bool is_to_complex = true;
    bool is_bool2int = true;

    for (auto &[type_trail, type] : results) {
      all_equal        &= type == results[0].second;
      eq_promoted      &= type == promoted_type_trail(type_trail);
      eq_first         &= type == type_trail[0];
      eq_second        &= type_trail.size() >= 2 && type == type_trail[1];
      is_value         &= toValueType(type_trail[0]) == type;
      is_to_float      &= to_float(type_trail[0]) == type;
      is_to_float2     &= type_trail.size() >= 2 &&
                          to_float2(type_trail[0], type_trail[1]) == type;
      is_to_float3     &= type_trail.size() >= 3 &&
                          to_float2(type_trail[0], type_trail[2]) == type;
      is_to_double2    &= type_trail.size() > 1 &&
                          to_double(type_trail[1]) == type;
      is_to_real_float &= to_real_float(type_trail[0]) == type;
      is_to_complex    &= to_complex(type_trail[0]) == type;
      is_bool2int      &= bool_to_int(type_trail[0]) == type;
    }

    cout << name;

    if (all_equal) {
      cout << ": ALL " << results[0].second << endl;
      return;
    }
    if (eq_first) {
      cout << ": EQ_FIRST" << endl;
      return;
    }
    if (eq_second) {
      cout << ": EQ_SECOND" << endl;
      return;
    }
    if (eq_promoted) {
      cout << ": EQ_PROMOTED" << endl;
      return;
    }
    if (is_value) {
      cout << ": TO_VALUE_TYPE" << endl;
      return;
    }
    if (is_to_float) {
      cout << ": TO_FLOAT" << endl;
      return;
    }
    if (is_to_float2) {
      cout << ": TO_FLOAT2" << endl;
      return;
    }
    if (is_to_float3) {
      cout << ": TO_FLOAT3" << endl;
      return;
    }
    if (is_to_double2) {
      cout << ": TO_DOUBLE2" << endl;
      return;
    }
    if (is_to_real_float) {
      cout << ": TO_REAL_FLOAT" << endl;
      return;
    }
    if (is_to_complex) {
      cout << ": TO_COMPLEX" << endl;
      return;
    }
    if (is_bool2int) {
      cout << ": BOOL2INT" << endl;
      return;
    }

    cout << ": NON_STANDARD:" << endl;

    for (auto &[type_trail, type] : results) {
      bool first = true;
      for (auto ty : type_trail) {
        if (!first)
          cout << ", ";
        cout << ty;
        first = false;
      }
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
#include "call_pytorch_fns.h"
  return 0;
}
