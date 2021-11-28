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

ScalarType promote_tys_undef(ScalarType a, ScalarType b) {
  if (a == ScalarType::Undefined)
    return b;
  if (b == ScalarType::Undefined)
    return a;
  return promoteTypes(a, b);
}

void promote_(ScalarType &ty_scalar, ScalarType &ty_zero,
              ScalarType &ty_nonzero) {}

void promote_(ScalarType &ty_scalar, ScalarType &ty_zero,
              ScalarType &ty_nonzero, ScalarType ty,
              bool is_scalar, const function<bool()> &z) {
  auto &target = is_scalar ? ty_scalar : (z() ? ty_zero : ty_nonzero);
  target = promote_tys_undef(target, ty);
}

template <typename... Args>
void promote_(ScalarType &ty_scalar, ScalarType &ty_zero,
              ScalarType &ty_nonzero, ScalarType ty,
              bool is_scalar, const function<bool()> &z, Args&&... args) {
  promote_(ty_scalar, ty_zero, ty_nonzero, ty, is_scalar, z);
  promote_(ty_scalar, ty_zero, ty_nonzero, args...);
}

void promote_(ScalarType &ty_scalar, ScalarType &ty_zero,
              ScalarType &ty_nonzero, const Tensor &t) {
  promote_(ty_scalar, ty_zero, ty_nonzero, t.scalar_type(), false,
           [&]() { return t.dim() == 0; });
}

template <typename... Args>
void promote_(ScalarType &ty_scalar, ScalarType &ty_zero,
              ScalarType &ty_nonzero, const Tensor &t,
              Args&&... args) {
  promote_(ty_scalar, ty_zero, ty_nonzero, t);
  promote_(ty_scalar, ty_zero, ty_nonzero, args...);
}

template <typename... Args>
void promote_(ScalarType &ty_scalar, ScalarType &ty_zero,
              ScalarType &ty_nonzero,
              const TensorList &list, Args&&... args) {
  for (auto &elem : list) {
    promote_(ty_scalar, ty_zero, ty_nonzero, elem);
  }
  promote_(ty_scalar, ty_zero, ty_nonzero, args...);
}

template <typename... Args>
tuple<ScalarType,ScalarType,ScalarType> promote(Args&&... args) {
  auto ty_scalar = ScalarType::Undefined;
  auto ty_zero = ScalarType::Undefined;
  auto ty_nonzero = ScalarType::Undefined;
  promote_(ty_scalar, ty_zero, ty_nonzero, args...);
  return { ty_scalar, ty_zero, ty_nonzero };
}

bool all_ty_eq(ScalarType ty) { return true; }

template <typename... Args>
bool all_ty_eq(ScalarType ty0, ScalarType ty, bool is_scalar,
               const function<bool()> &z, Args&&... args) {
  return ty0 == ty && all_ty_eq(ty0, args...);
}

template <typename... Args>
bool all_ty_eq(ScalarType ty0, const Tensor &t, Args&&... args) {
  return ty0 == t.scalar_type() && all_ty_eq(ty0, args...);
}

template <typename... Args>
bool all_ty_eq(ScalarType ty, bool is_scalar, const function<bool()> &z,
               Args&&... args) {
  return all_ty_eq(ty, args...);
}

template <typename... Args>
bool all_ty_eq(const TensorList &list, Args&&... args) {
  if (list.empty())
    return true;

  auto ty = list.front().dtype();
  for (auto &elem : list) {
    if (elem.dtype() != ty)
      return false;
  }
  return all_ty_eq(ty.toScalarType(), args...);
}

template <typename... Args>
bool all_ty_eq(const Tensor &t, Args&&... args) {
  return all_ty_eq(t.scalar_type(), args...);
}

template <typename... Args>
ScalarType pick_first_ty(ScalarType ty, bool is_scalar,
                         const function<bool()> &z, Args&&... args) {
  return ty;
}

template <typename... Args>
ScalarType pick_first_ty(const TensorList &list, Args&&... args) {
  return list.empty() ? typeMetaToScalarType(at::get_default_dtype())
                      : list.front().scalar_type();
}

template <typename... Args>
ScalarType pick_first_ty(const Tensor &t, Args&&... args) {
  return t.scalar_type();
}

template <typename... Args>
ScalarType promote_tys(Args&&... args) {
  // avoid calling zerodim
  if (all_ty_eq(args...))
    return pick_first_ty(args...);

  auto p = promote(args...);
  return promote_tys_undef(get<0>(p), promote_tys_undef(get<1>(p), get<2>(p)));
}

template <typename... Args>
ScalarType promote_buggy(Args&&... args) {
  // avoid calling zerodim
  if (all_ty_eq(args...))
    return pick_first_ty(args...);

  auto p = promote(args...);
  auto ty_scalar = get<0>(p);
  auto ty_zero = get<1>(p);
  auto ty_nonzero = get<2>(p);

  // PyTorch bug: https://github.com/pytorch/pytorch/issues/60941
  if (((ty_scalar == kComplexFloat || ty_scalar == kComplexDouble) &&
       ((ty_zero == kDouble && ty_nonzero != kComplexFloat) ||
         ty_nonzero == kDouble)) ||
      (ty_zero == kComplexFloat && ty_nonzero == kDouble))
    return kComplexDouble;

  // 1. scalars of higher category win over tensors
  if (ty_to_num(ty_scalar) > ty_to_num(ty_zero) &&
      ty_to_num(ty_scalar) > ty_to_num(ty_nonzero)) {
    // ints -> long
    if (isIntegralType(ty_scalar, false))
      return kLong;

    // floating scalars revert to default dtype
    if (isFloatingType(ty_scalar))
      return typeMetaToScalarType(at::get_default_dtype());
    if (isComplexType(ty_scalar) && ty_nonzero != kDouble)
      return toComplexType(typeMetaToScalarType(at::get_default_dtype()));
    return ty_scalar;
  }

  // 2. zero-dim tensors win over non-zero
  return ty_to_num(ty_zero) > ty_to_num(ty_nonzero) ? ty_zero : ty_nonzero;
}

template <typename... Args>
ScalarType promote_const(Args&&... args) {
  // avoid calling zerodim
  if (all_ty_eq(args...))
    return pick_first_ty(args...);

  auto p = promote(args...);
  auto ty_scalar = get<0>(p);
  auto ty_zero = get<1>(p);
  auto ty_nonzero = get<2>(p);

  // 1. scalars of higher category win over tensors
  if (ty_to_num(ty_scalar) > ty_to_num(ty_zero) &&
      ty_to_num(ty_scalar) > ty_to_num(ty_nonzero)) {
    // ints -> long
    if (isIntegralType(ty_scalar, false))
      return kLong;

    // floating scalars revert to default dtype
    if (isFloatingType(ty_scalar))
      return typeMetaToScalarType(at::get_default_dtype());
    if (isComplexType(ty_scalar) && ty_nonzero != kDouble)
      return toComplexType(typeMetaToScalarType(at::get_default_dtype()));
    return ty_scalar;
  }

  // 2. zero-dim tensors win over non-zero
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

  return promote_buggy(ty1, false, zerodim1, ty2, false, zerodim2);
}

ScalarType to_float3(ScalarType ty1, const function<bool()> &zerodim1,
                     ScalarType ty2, const function<bool()> &zerodim2,
                     ScalarType ty3, const function<bool()> &zerodim3) {
  if (isIntegralType(ty1, true) &&
      isIntegralType(ty2, true) &&
      isIntegralType(ty3, true))
    return typeMetaToScalarType(at::get_default_dtype());

  if (zerodim1() && zerodim2())
    return promote_buggy(ty1, false, zerodim1, ty2, false, zerodim2, ty3, false,
                         zerodim3);
  return promote_buggy(promote_buggy(ty1, false, zerodim1, ty2, false,
                                     zerodim2), false, [](){ return false; },
                       ty3, false, zerodim3);
}

ScalarType to_float4(ScalarType ty1, const function<bool()> &zerodim1,
                     ScalarType ty2, const function<bool()> &zerodim2,
                     ScalarType ty3, const function<bool()> &zerodim3,
                     ScalarType ty4, const function<bool()> &zerodim4) {
  if (ty4 != ScalarType::Undefined)
    return
      promoteTypes(promote_buggy(ty3, false, zerodim3, ty4, false, zerodim4),
                   kLong);

  return promote_buggy(to_float(ty2), false, zerodim2, ty3, false, zerodim3);
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
  auto promoted = promote_buggy(ty1, false, zerodim1, ty2, false, zerodim2);
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
  return promote_buggy(ty1, false, zerodim1, ty2, false, zerodim2);
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

ScalarType optional_or_else(optional<ScalarType> opt, ScalarType ty) {
  return opt.value_or(ty);
}

ScalarType optional_or_longelse(optional<ScalarType> opt, ScalarType ty) {
  return optional_or_else(opt, integrals_to_int(ty));
}
