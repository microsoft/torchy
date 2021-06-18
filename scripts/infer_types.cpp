// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

#include <ATen/core/List.h>
#include <ATen/NativeFunctions.h>
#include <ATen/RedispatchFunctions.h>
#include <array>
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

vector<ScalarType> type_trail;

ScalarType promoted_type_trail() {
  auto ty = type_trail[0];
  for (auto ty2 : type_trail) {
    ty = promoteTypes(ty, ty2);
  }
  return ty;
}

struct C {
  static void call(const char *name, function<Tensor()> fn) {
    cout << name << '(';
    bool first = true;
    for (auto ty : type_trail) {
      if (!first)
        cout << ", ";
      cout << ty;
      first = false;
    }
    cout << ") -> ";
    auto promoted_ty = promoted_type_trail();
    try {
      auto dtype = typeMetaToScalarType(fn().dtype());
      cout << dtype;
      if (dtype != promoted_ty)
        cout << " [not equal to promoted type: " << promoted_ty << ']';
      cout << '\n';

    } catch (const c10::Error &) {
      cout << "Exception! [promoted type: " << promoted_ty << "]\n";
    }
  }

  template <typename... Tail>
  static void call(const char *name, function<Tensor(Tensor&, Tail...)> fn) {
    for (auto ty : types) {
      if (isQIntType(ty))
        continue;
      type_trail.emplace_back(ty);
      call(name, function<Tensor(Tail&&...)>{[=](Tail&&... args) -> Tensor {
        auto t = native::empty_cpu({1}, ty);
        return fn(t, forward<Tail>(args)...);
      }});
      type_trail.pop_back();
    }
  }

  template <typename... Tail>
  static void call(const char *name, function<Tensor(TensorList&, Tail...)> fn) {
    for (auto ty1 : types) {
      if (isQIntType(ty1))
        continue;
      type_trail.emplace_back(ty1);

      for (auto ty2 : types) {
        if (isQIntType(ty2))
          continue;
        type_trail.emplace_back(ty2);
        call(name, function<Tensor(Tail&&...)>{[=](Tail&&... args) -> Tensor {
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
  static void call(const char *name,
                   function<Tensor(List<optional<Tensor>>&, Tail...)> fn) {
    for (auto ty1 : types) {
      if (isQIntType(ty1))
        continue;
      type_trail.emplace_back(ty1);

      for (auto ty2 : types) {
        if (isQIntType(ty2))
          continue;
        type_trail.emplace_back(ty2);
        call(name, function<Tensor(Tail&&...)>{[=](Tail&&... args) -> Tensor {
          List<optional<Tensor>> list({ native::empty_cpu({1}, ty1),
                                        native::empty_cpu({1}, ty2) });
          return fn(list, forward<Tail>(args)...);
        }});
        type_trail.pop_back();
      }
      type_trail.pop_back();
    }
  }

};

}

int main() {
#include "call_pytorch_fns.h"
  return 0;
}
