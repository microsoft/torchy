// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

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

// return zero_ty, nonzero_ty
pair<ScalarType,ScalarType>
promote2_spilt(ScalarType ty1, function<bool()> zerodim1,
               ScalarType ty2, function<bool()> zerodim2) {
  bool z1 = zerodim1();
  auto zero    = z1 ? ty1 : ScalarType::Undefined;
  auto nonzero = z1 ? ScalarType::Undefined : ty1;

  auto &ty2up = (zerodim2() ? zero : nonzero);
  ty2up = ty2up != ScalarType::Undefined ? promoteTypes(ty2up, ty2) : ty2;
  return { zero, nonzero };
}

ScalarType promote2_buggy(ScalarType ty1, function<bool()> zerodim1,
                          ScalarType ty2, function<bool()> zerodim2) {
  // avoid calling zerodim
  if (ty1 == ty2)
    return ty1;

  auto p = promote2_spilt(ty1, move(zerodim1), ty2, move(zerodim2));
  auto ty_zero = p.first;
  auto ty_nonzero = p.second;

  // PyTorch bug: https://github.com/pytorch/pytorch/issues/60941
  if (ty_zero == kComplexFloat && ty_nonzero == kDouble)
    return kComplexDouble;

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

ScalarType to_float2(ScalarType ty1, function<bool()> zerodim1, ScalarType ty2,
                     function<bool()> zerodim2) {
  if (isIntegralType(ty1, true) && isIntegralType(ty2, true))
    return typeMetaToScalarType(at::get_default_dtype());

  return promote2_buggy(ty1, move(zerodim1), ty2, move(zerodim2));
}

ScalarType to_float2_2(ScalarType ty1, function<bool()> zerodim1,
                       ScalarType ty2, function<bool()> zerodim2) {
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
      (ty1 == kFloat || ty1 == kComplexFloat) &&
      isIntegralType(ty2, true) &&
      (zerodim1() || !zerodim2()))
    return promoteTypes(ty1, default_ty);

  return promote2_buggy(ty1, move(zerodim1), ty2, move(zerodim2));
}

ScalarType to_float2_4(ScalarType ty, ScalarType ty2) {
  auto res = promoteTypes(ty, ty2);
  if (isIntegralType(ty, true))
    return promoteTypes(res, typeMetaToScalarType(at::get_default_dtype()));
  return res;
}

ScalarType to_float3(ScalarType ty, ScalarType ty2, ScalarType ty3) {
  if (isIntegralType(ty, true) &&
      isIntegralType(ty2, true) &&
      isIntegralType(ty3, true))
    return typeMetaToScalarType(at::get_default_dtype());
  return promoteTypes(promoteTypes(ty, ty2), ty3);
}

ScalarType to_float4(ScalarType ty1, ScalarType ty2, ScalarType ty3,
                     ScalarType ty4) {
  if (ty4 != ScalarType::Undefined)
    return promoteTypes(promoteTypes(ty3, ty4), kLong);

  return promoteTypes(to_float(ty2), ty3);
}

ScalarType to_real_float(ScalarType ty) {
  if (isFloatingType(ty))
    return ty;
  if (isComplexType(ty))
    return toValueType(ty);
  return typeMetaToScalarType(at::get_default_dtype());
}

ScalarType to_real2(ScalarType ty, ScalarType ty2) {
  if (isComplexType(ty) || isComplexType(ty2))
    return toValueType(promoteTypes(ty, ty2));
  return promoteTypes(ty, ty2);
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

ScalarType bool_to_int2(ScalarType ty, ScalarType ty2) {
  if (isIntegralType(ty, true) && ty2 == ScalarType::Bool)
    return ScalarType::Long;
  return promoteTypes(ty, ty2);
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
