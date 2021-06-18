// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

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

void call(const char *name, function<Tensor()> fn) {
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
void call(const char *name, function<Tensor(const Tensor&, Tail...)> fn) {
  for (auto ty : types) {
    if (isQIntType(ty))
      continue;
    type_trail.emplace_back(ty);
    call(name, function<Tensor(Tail&&...)>{[=](Tail&&... args) -> Tensor {
      return fn(native::empty_cpu({1}, ty), forward<Tail>(args)...);
    }});
    type_trail.pop_back();
  }
}

}

int main() {
  call("add", function<Tensor(const Tensor &, const Tensor &)>{[](const Tensor &t0, const Tensor &t1) { return redispatch::add(DispatchKeySet(DispatchKey::CPU), t0, t1, Scalar(0)); }});
  return 0;
}
