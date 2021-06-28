// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

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

ScalarType to_float2(ScalarType ty, ScalarType ty2) {
  if (isIntegralType(ty, true) && isIntegralType(ty2, true))
    return typeMetaToScalarType(at::get_default_dtype());
  return promoteTypes(ty, ty2);
}

ScalarType to_float2_2(ScalarType ty, ScalarType ty2) {
  auto res = promoteTypes(ty, ty2);
  if (isFloatingType(ty2) || isComplexType(ty2))
    return res;
  return promoteTypes(res, typeMetaToScalarType(at::get_default_dtype()));
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
