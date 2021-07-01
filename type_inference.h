// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

// https://pytorch.org/docs/stable/tensor_attributes.html#type-promotion-doc
// complex > floating > integral > boolean > undefined
unsigned ty_to_num(ScalarType ty) {
  if (isComplexType(ty))
    return 4;
  if (isFloatingType(ty))
    return 3;
  if (isIntegralType(ty, false))
    return 2;
  if (ty == kBool)
    return 1;
  return 0;
}

void promote(ScalarType &ty_zero, ScalarType &ty_nonzero, ScalarType ty,
             const function<bool()> &z) {
  auto &target = z() ? ty_zero : ty_nonzero;
  if (target == ScalarType::Undefined)
    target = ty;
  else if (ty != ScalarType::Undefined)
    target = promoteTypes(target, ty);
}

// return zero_ty, nonzero_ty
template <typename... Args>
void promote(ScalarType &ty_zero, ScalarType &ty_nonzero, ScalarType ty,
             const function<bool()> &z, Args&&... args) {
  promote(ty_zero, ty_nonzero, ty, z);
  promote(ty_zero, ty_nonzero, args...);
}

// return zero_ty, nonzero_ty
template <typename... Args>
pair<ScalarType,ScalarType> promote(ScalarType ty, const function<bool()> &z,
                                    Args&&... args) {
  auto ty_zero = ScalarType::Undefined;
  auto ty_nonzero = ScalarType::Undefined;
  promote(ty_zero, ty_nonzero, ty, z, args...);
  return { ty_zero, ty_nonzero };
}

bool all_ty_eq(ScalarType ty) { return true; }

template <typename... Args>
bool all_ty_eq(ScalarType ty0, ScalarType ty, const function<bool()> &z,
               Args&&... args) {
  return ty0 == ty && all_ty_eq(ty, args...);
}

template <typename... Args>
bool all_ty_eq(ScalarType ty, const function<bool()> &z, Args&&... args) {
  return all_ty_eq(ty, args...);
}

template <typename... Args>
ScalarType pick_first_ty(ScalarType ty, const function<bool()> &z,
                         Args&&... args) {
  return ty;
}

template <typename... Args>
ScalarType promote_buggy(Args&&... args) {
  // avoid calling zerodim
  if (all_ty_eq(args...))
    return pick_first_ty(args...);

  auto p = promote(args...);
  auto ty_zero = p.first;
  auto ty_nonzero = p.second;

  // PyTorch bug: https://github.com/pytorch/pytorch/issues/60941
  if (ty_zero == kComplexFloat && ty_nonzero == kDouble)
    return kComplexDouble;

  return ty_to_num(ty_zero) > ty_to_num(ty_nonzero) ? ty_zero : ty_nonzero;
}

template <typename... Args>
ScalarType promote_const(Args&&... args) {
  // avoid calling zerodim
  if (all_ty_eq(args...))
    return pick_first_ty(args...);

  auto p = promote(args...);
  auto ty_zero = p.first;
  auto ty_nonzero = p.second;

  return ty_to_num(ty_zero) > ty_to_num(ty_nonzero) ? ty_zero : ty_nonzero;
}

ScalarType to_float(ScalarType ty) {
  if (isIntegralType(ty, true))
    return typeMetaToScalarType(at::get_default_dtype());
  return ty;
}

ScalarType to_float_double(ScalarType ty) {
  switch (ty) {
  case ScalarType::Float:
    return ty;
  case ScalarType::Undefined:
    return ScalarType::Long;
  default:
    return ScalarType::Double;
  }
}

ScalarType to_double(ScalarType ty) {
  if (isComplexType(ty))
    return ScalarType::ComplexDouble;
  return ScalarType::Double;
}

ScalarType to_double2(ScalarType ty, ScalarType ty2) {
  if (isComplexType(ty) || isComplexType(ty2))
    return ScalarType::ComplexDouble;
  return ScalarType::Double;
}

ScalarType to_float2(ScalarType ty1, const function<bool()> &zerodim1,
                     ScalarType ty2, const function<bool()> &zerodim2) {
  if (isIntegralType(ty1, true) && isIntegralType(ty2, true))
    return typeMetaToScalarType(at::get_default_dtype());

  return promote_buggy(ty1, zerodim1, ty2, zerodim2);
}

ScalarType to_float2_2(ScalarType ty1, const function<bool()> &zerodim1,
                       ScalarType ty2, const function<bool()> &zerodim2) {
  auto default_ty = typeMetaToScalarType(at::get_default_dtype());

  if (isIntegralType(ty1, true) && isIntegralType(ty2, true))
    return default_ty;

  if (isIntegralType(ty1, true))
    return promoteTypes(ty1, ty2);

  if (ty1 == kFloat && ty2 == kFloat)
    return ty1;

  if (default_ty == kFloat && ty1 == kDouble && zerodim1() && !zerodim2())
    return promoteTypes(default_ty, ty2);

  if (default_ty == kDouble &&
      isFloatingType(ty1) &&
      isIntegralType(ty2, true) &&
      (zerodim1() || !zerodim2()))
    return promoteTypes(ty1, default_ty);

  return promote_buggy(ty1, zerodim1, ty2, zerodim2);
}

ScalarType to_float2_4(ScalarType ty1, const function<bool()> &zerodim1,
                       ScalarType ty2, const function<bool()> &zerodim2) {
  auto res = promote_buggy(ty1, zerodim1, ty2, zerodim2);
  if (isIntegralType(ty1, true)) {
    auto default_ty = typeMetaToScalarType(at::get_default_dtype());
    if (default_ty == kDouble &&
        (isFloatingType(ty2) || isComplexType(ty2)) &&
        zerodim1() && !zerodim2())
      return ty2;
    if (ty2 == kDouble && !zerodim1() &&zerodim2())
      return default_ty;
    return promoteTypes(res, default_ty);
  }
  return res;
}

ScalarType to_float3(ScalarType ty1, const function<bool()> &zerodim1,
                     ScalarType ty2, const function<bool()> &zerodim2,
                     ScalarType ty3, const function<bool()> &zerodim3) {
  if (isIntegralType(ty1, true) &&
      isIntegralType(ty2, true) &&
      isIntegralType(ty3, true))
    return typeMetaToScalarType(at::get_default_dtype());

  if (zerodim1() && zerodim2())
    return promote_buggy(ty1, zerodim1, ty2, zerodim2, ty3, zerodim3);
  return promote_buggy(promote_buggy(ty1, zerodim1, ty2, zerodim2),
                       [](){ return false; }, ty3, zerodim3);
}

ScalarType to_float4(ScalarType ty1, const function<bool()> &zerodim1,
                     ScalarType ty2, const function<bool()> &zerodim2,
                     ScalarType ty3, const function<bool()> &zerodim3,
                     ScalarType ty4, const function<bool()> &zerodim4) {
  if (ty4 != ScalarType::Undefined)
    return promoteTypes(promote_buggy(ty3, zerodim3, ty4, zerodim4), kLong);

  return promote_buggy(to_float(ty2), zerodim2, ty3, zerodim3);
}

ScalarType to_real_float(ScalarType ty) {
  if (isFloatingType(ty))
    return ty;
  if (isComplexType(ty))
    return toValueType(ty);
  return typeMetaToScalarType(at::get_default_dtype());
}

ScalarType to_real2(ScalarType ty1, const function<bool()> &zerodim1,
                    ScalarType ty2, const function<bool()> &zerodim2) {
  auto promoted = promote_buggy(ty1, zerodim1, ty2, zerodim2);
  return
    isComplexType(ty1) || isComplexType(ty2) ? toValueType(promoted) : promoted;
}

ScalarType myToComplexType(ScalarType ty) {
  return ty == kBFloat16 ? kFloat : toComplexType(ty);
}

ScalarType to_complex(ScalarType ty) {
  if (isComplexType(ty))
    return ty;
  if (isFloatingType(ty))
    return myToComplexType(ty);
  return toComplexType(typeMetaToScalarType(at::get_default_dtype()));
}

ScalarType bool_to_int(ScalarType ty) {
  if (ty == ScalarType::Bool)
    return ScalarType::Long;
  return ty;
}

ScalarType bool_to_int2(ScalarType ty1, const function<bool()> &zerodim1,
                        ScalarType ty2, const function<bool()> &zerodim2) {
  if (isIntegralType(ty1, true) &&
      ty2 == ScalarType::Bool &&
      (zerodim1() || !zerodim2()))
    return ScalarType::Long;
  return promote_buggy(ty1, zerodim1, ty2, zerodim2);
}

ScalarType bool_byte(ScalarType ty) {
  if (ty == ScalarType::Byte)
    return ty;
  return ScalarType::Bool;
}

ScalarType integrals_to_int(ScalarType ty) {
  if (isIntegralType(ty, true))
    return ScalarType::Long;
  return ty;
}
