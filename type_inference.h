// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

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

ScalarType to_float_double(ScalarType ty) {
  if (ty == ScalarType::Float)
    return ty;
  return ScalarType::Double;
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
    return ScalarType::Float;
  return promoteTypes(ty, ty2);
}

ScalarType to_float2_2(ScalarType ty, ScalarType ty2) {
  if (isComplexType(ty) ||
      ty == ScalarType::Double ||
      ty2 == ScalarType::Double ||
      ty2 == ScalarType::BFloat16)
    return promoteTypes(ty, ty2);

  return ScalarType::Float;
}

ScalarType to_float2_3(ScalarType ty, ScalarType ty2) {
  if ((isIntegralType(ty, true) ||
       ty == ScalarType::Half ||
       ty == ScalarType::BFloat16) && isIntegralType(ty2, true))
    return ScalarType::Float;
  return promoteTypes(ty, ty2);
}

ScalarType to_float2_4(ScalarType ty, ScalarType ty2) {
  if (isIntegralType(ty, true) &&
      (isIntegralType(ty2, true) ||
       ty2 == ScalarType::Half ||
       ty2 == ScalarType::BFloat16))
    return ScalarType::Float;
  return promoteTypes(ty, ty2);
}

ScalarType to_float3(ScalarType ty, ScalarType ty2, ScalarType ty3) {
  if (isIntegralType(ty, true) &&
      isIntegralType(ty2, true) &&
      isIntegralType(ty3, true))
    return ScalarType::Float;
  return promoteTypes(promoteTypes(ty, ty2), ty3);
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

ScalarType to_real2(ScalarType ty, ScalarType ty2) {
  if (isComplexType(ty) || isComplexType(ty2))
    return toValueType(promoteTypes(ty, ty2));
  return promoteTypes(ty, ty2);
}

ScalarType to_complex(ScalarType ty) {
  switch (ty) {
  case ScalarType::Half:
  case ScalarType::ComplexHalf:
    return ScalarType::ComplexHalf;
  case ScalarType::Double:
  case ScalarType::ComplexDouble:
    return ScalarType::ComplexDouble;
  default:
    return ScalarType::ComplexFloat;
  }
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
