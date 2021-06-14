case H__CAST_BYTE:
case H__CAST_CHAR:
case H__CAST_DOUBLE:
case H__CAST_FLOAT:
case H__CAST_INT:
case H__CAST_LONG:
case H__CAST_SHORT:
case H__CAST_HALF:
case H_MATRIX_RANK:
case H_STD:
case H_VAR:
case H_NUCLEAR_NORM:
case H_CHOLESKY:
case H_CHOLESKY_INVERSE:
  set(op, redispatch_ptrs_0[op.id - H__CAST_BYTE](ks, load<const at::Tensor &>()(op.args[0]), load<bool>()(op.args[1])));
  break;

case H_DATA:
case H__SHAPE_AS_TENSOR:
case H_ABS:
case H_ABSOLUTE:
case H_ANGLE:
case H_VIEW_AS_REAL:
case H__VIEW_AS_REAL_PHYSICAL:
case H_VIEW_AS_COMPLEX:
case H_REAL:
case H_IMAG:
case H__CONJ:
case H_CONJ:
case H__CONJ_PHYSICAL:
case H_CONJ_PHYSICAL:
case H_RESOLVE_CONJ:
case H_ARCCOS:
case H_ARCCOSH:
case H_ARCSINH:
case H_ARCTANH:
case H_ASIN:
case H_ARCSIN:
case H_ARCTAN:
case H_ATLEAST_1D:
case H_ATLEAST_2D:
case H_ATLEAST_3D:
case H_LOGICAL_NOT:
case H_CEIL:
case H_FLOOR:
case H_INVERSE:
case H__INVERSE_HELPER:
case H_ISNAN:
case H_ISREAL:
case H_FBGEMM_PACK_GEMM_MATRIX_FP16:
case H_FBGEMM_PACK_QUANTIZED_MATRIX:
case H_LOG1P:
case H_LOGDET:
case H_MATRIX_EXP:
case H_MEDIAN:
case H_NANMEDIAN:
case H_MIOPEN_CONVOLUTION_BACKWARD_BIAS:
case H_NUMPY_T:
case H_PIN_MEMORY:
case H_RAD2DEG:
case H_DEG2RAD:
case H_RAVEL:
case H_NEG:
case H_NEGATIVE:
case H_RELU:
case H_RELU6:
case H_GELU:
case H_SELU:
case H_SILU:
case H_MISH:
case H_SIGMOID:
case H_DETACH:
case H_SQUEEZE:
case H_SQRT:
case H_SQUARE:
case H_T:
case H_TANH:
case H_FLIPLR:
case H_FLIPUD:
case H_TRUNC:
case H_FIX:
case H__SPARSE_SUM:
case H_FROBENIUS_NORM:
case H_POSITIVE:
case H_COALESCE:
case H__COALESCE:
case H__INDICES:
case H__VALUES:
case H_INDICES:
case H_VALUES:
case H_CROW_INDICES:
case H_COL_INDICES:
case H_TO_SPARSE:
case H_DEQUANTIZE_SELF:
case H_Q_PER_CHANNEL_SCALES:
case H_Q_PER_CHANNEL_ZERO_POINTS:
case H_INT_REPR:
case H__SATURATE_WEIGHT_TO_FP16:
case H_TRACE:
case H_NONZERO:
case H_SIGN:
case H_SIGNBIT:
case H_MIN:
case H_MAX:
case H_MSORT:
case H_ALL:
case H_ANY:
case H_ALIAS:
case H_HARDSIGMOID:
case H_HARDSWISH:
case H_LOG_SIGMOID:
case H_ISFINITE:
case H_ISINF:
case H_ISPOSINF:
case H_ISNEGINF:
case H_SPECIAL_EXPM1:
case H_SPECIAL_EXP2:
case H_SPECIAL_PSI:
case H_SPECIAL_DIGAMMA:
case H_SPECIAL_GAMMALN:
case H_SPECIAL_ERF:
case H_SPECIAL_ERFC:
case H_SPECIAL_ERFINV:
case H_SPECIAL_NDTR:
case H_SPECIAL_I0:
case H_SPECIAL_EXPIT:
case H_LINALG_CHOLESKY:
case H_LINALG_DET:
case H_DET:
case H_LINALG_EIGVALS:
case H_LINALG_INV:
case H_LINALG_SVDVALS:
  set(op, redispatch_ptrs_1[op.id - H_DATA](ks, load<const at::Tensor &>()(op.args[0])));
  break;

case H_REQUIRES_GRAD_:
case H__COALESCED_:
  {at::Tensor &(*ptr)(DispatchKeySet, at::Tensor &, bool) = at::redispatch::__dispatch_requires_grad_;
  if (op.id == H__COALESCED_) ptr = at::redispatch::_coalesced_;
  init_update_in_place(op);
  ptr(ks, load<at::Tensor &>()(op.args[0]), load<bool>()(op.args[1]));
  end_update_in_place(op);
  break;}

case H__FW_PRIMAL:
case H__DIM_ARANGE:
case H_DIAGFLAT:
case H__LOGCUMSUMEXP:
case H_LOGCUMSUMEXP:
case H_MATRIX_POWER:
case H_MVLGAMMA:
case H_PIXEL_SHUFFLE:
case H_PIXEL_UNSHUFFLE:
case H_CHANNEL_SHUFFLE:
case H_SQUEEZE_DIM:
case H_ONE_HOT:
case H_UNSQUEEZE:
case H_TO_SPARSE_SPARSE_DIM:
case H_DIAG:
case H_TRIU:
case H_TRIL:
case H__CUMSUM:
case H__CUMPROD:
case H_GLU:
case H_LINALG_TENSORINV:
case H_LINALG_MATRIX_POWER:
  set(op, redispatch_ptrs_2[op.id - H__FW_PRIMAL](ks, load<const at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1])));
  break;

case H__MAKE_DUAL:
case H_TRAPZ_X:
case H__WEIGHT_NORM:
case H_MSE_LOSS:
case H_L1_LOSS:
case H_MULTILABEL_MARGIN_LOSS:
case H_SOFT_MARGIN_LOSS:
case H_GLU_BACKWARD:
  set(op, redispatch_ptrs_3[op.id - H__MAKE_DUAL](ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<int64_t>()(op.args[2])));
  break;

case H_RENAME_:
  init_update_in_place(op);
  at::redispatch::rename_(ks, load<at::Tensor &>()(op.args[0]), load<c10::optional<at::DimnameList>>()(op.args[1]));
  end_update_in_place(op);
  break;

case H_RENAME:
  set(op, at::redispatch::rename(ks, load<const at::Tensor &>()(op.args[0]), load<c10::optional<at::DimnameList>>()(op.args[1])));
  break;

case H_ALIGN_TO:
case H_REFINE_NAMES:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, at::DimnameList) = at::redispatch::align_to;
  if (op.id == H_REFINE_NAMES) ptr = at::redispatch::refine_names;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<at::DimnameList>()(op.args[1])));
  break;}

case H_ALIGN_TO_ELLIPSIS_IDX:
  set(op, at::redispatch::align_to(ks, load<const at::Tensor &>()(op.args[0]), load<at::DimnameList>()(op.args[1]), load<int64_t>()(op.args[2])));
  break;

case H_ALIGN_AS:
case H__RESHAPE_FROM_TENSOR:
case H_LOGICAL_XOR:
case H_LOGICAL_AND:
case H_LOGICAL_OR:
case H_BMM:
case H_CLAMP_MAX_TENSOR:
case H_CLAMP_MIN_TENSOR:
case H_COMPLEX:
case H_POLAR:
case H_CUDNN_GRID_SAMPLER:
case H_DIV_TENSOR:
case H_DIVIDE_TENSOR:
case H_TRUE_DIVIDE_TENSOR:
case H_DOT:
case H_VDOT:
case H_EXPAND_AS:
case H_FLOOR_DIVIDE:
case H_KRON:
case H_LDEXP_TENSOR:
case H_LOGADDEXP:
case H_LOGADDEXP2:
case H_XLOGY_TENSOR:
case H_MATMUL:
case H_MATRIX_EXP_BACKWARD:
case H__COMPUTE_LINEAR_COMBINATION:
case H_MM:
case H__SPARSE_MM:
case H__SPARSE_SPARSE_MATMUL:
case H__SPARSE_MASK_HELPER:
case H_MUL_TENSOR:
case H_MULTIPLY_TENSOR:
case H_MV:
case H__EUCLIDEAN_DIST:
case H_RESHAPE_AS:
case H_PRELU:
case H_INFINITELY_DIFFERENTIABLE_GELU_BACKWARD:
case H_SILU_BACKWARD:
case H_MISH_BACKWARD:
case H_SMM:
case H_TYPE_AS:
case H_VIEW_AS:
case H__STANDARD_GAMMA_GRAD:
case H_SPARSE_MASK:
case H_TO_DENSE_BACKWARD:
case H_HSPMM:
case H_TO_MKLDNN_BACKWARD:
case H_FAKE_QUANTIZE_PER_TENSOR_AFFINE_CACHEMASK_BACKWARD:
case H_FAKE_QUANTIZE_PER_CHANNEL_AFFINE_CACHEMASK_BACKWARD:
case H_BITWISE_AND_TENSOR:
case H___AND___TENSOR:
case H_BITWISE_OR_TENSOR:
case H___OR___TENSOR:
case H_BITWISE_XOR_TENSOR:
case H___XOR___TENSOR:
case H___LSHIFT___TENSOR:
case H___RSHIFT___TENSOR:
case H_NE_TENSOR:
case H_NOT_EQUAL_TENSOR:
case H_EQ_TENSOR:
case H_GE_TENSOR:
case H_GREATER_EQUAL_TENSOR:
case H_LE_TENSOR:
case H_LESS_EQUAL_TENSOR:
case H_GT_TENSOR:
case H_GREATER_TENSOR:
case H_LT_TENSOR:
case H_LESS_TENSOR:
case H_TAKE:
case H_MASKED_SELECT:
case H_ORGQR:
case H_FMOD_TENSOR:
case H_MAX_OTHER:
case H_MIN_OTHER:
case H_FLOAT_POWER_TENSOR_TENSOR:
case H_HARDSWISH_BACKWARD:
case H_MKLDNN_ADAPTIVE_AVG_POOL2D_BACKWARD:
case H__ADAPTIVE_AVG_POOL2D_BACKWARD:
case H__ADAPTIVE_AVG_POOL3D_BACKWARD:
case H_SIGMOID_BACKWARD:
case H_TANH_BACKWARD:
case H_LINALG_HOUSEHOLDER_PRODUCT:
case H_INNER:
case H_OUTER:
case H_GER:
case H_LINALG_SOLVE:
  set(op, redispatch_ptrs_4[op.id - H_ALIGN_AS](ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1])));
  break;

case H__CUDNN_RNN_FLATTEN_WEIGHT:
  set(op, at::redispatch::_cudnn_rnn_flatten_weight(ks, load<at::TensorList>()(op.args[0]), load<int64_t>()(op.args[1]), load<int64_t>()(op.args[2]), load<int64_t>()(op.args[3]), load<int64_t>()(op.args[4]), load<int64_t>()(op.args[5]), load<int64_t>()(op.args[6]), load<bool>()(op.args[7]), load<bool>()(op.args[8])));
  break;

case H__CUDNN_INIT_DROPOUT_STATE:
  set(op, at::redispatch::_cudnn_init_dropout_state(ks, load<double>()(op.args[0]), load<bool>()(op.args[1]), load<int64_t>()(op.args[2]), load<c10::optional<at::ScalarType>>()(op.args[3]), load<c10::optional<at::Layout>>()(op.args[4]), load<c10::optional<at::Device>>()(op.args[5]), load<c10::optional<bool>>()(op.args[6])));
  break;

case H__MASKED_SCALE:
  set(op, at::redispatch::_masked_scale(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<double>()(op.args[2])));
  break;

case H__SOBOL_ENGINE_FF_:
  init_update_in_place(op);
  at::redispatch::_sobol_engine_ff_(ks, load<at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<int64_t>()(op.args[3]), load<int64_t>()(op.args[4]));
  end_update_in_place(op);
  break;

case H__SOBOL_ENGINE_SCRAMBLE_:
  init_update_in_place(op);
  at::redispatch::_sobol_engine_scramble_(ks, load<at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<int64_t>()(op.args[2]));
  end_update_in_place(op);
  break;

case H__SOBOL_ENGINE_INITIALIZE_STATE_:
case H_MVLGAMMA_:
case H_SQUEEZE__DIM:
case H_UNSQUEEZE_:
case H_TRIL_:
case H_TRIU_:
case H_POLYGAMMA_:
  init_update_in_place(op);
  redispatch_ptrs_5[op.id - H__SOBOL_ENGINE_INITIALIZE_STATE_](ks, load<at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1]));
  end_update_in_place(op);
  break;

case H_DROPOUT:
case H_FEATURE_DROPOUT:
case H_ALPHA_DROPOUT:
case H_FEATURE_ALPHA_DROPOUT:
case H_MATRIX_RANK_TOL:
case H_LINALG_PINV:
  set(op, redispatch_ptrs_6[op.id - H_DROPOUT](ks, load<const at::Tensor &>()(op.args[0]), load<double>()(op.args[1]), load<bool>()(op.args[2])));
  break;

case H_DROPOUT_:
case H_FEATURE_DROPOUT_:
case H_ALPHA_DROPOUT_:
case H_FEATURE_ALPHA_DROPOUT_:
  init_update_in_place(op);
  redispatch_ptrs_7[op.id - H_DROPOUT_](ks, load<at::Tensor &>()(op.args[0]), load<double>()(op.args[1]), load<bool>()(op.args[2]));
  end_update_in_place(op);
  break;

case H_ABS_:
case H_ABSOLUTE_:
case H_CONJ_PHYSICAL_:
case H_ARCCOS_:
case H_ARCCOSH_:
case H_ARCSINH_:
case H_ARCTANH_:
case H_ASIN_:
case H_ARCSIN_:
case H_ARCTAN_:
case H_LOGICAL_NOT_:
case H_CEIL_:
case H_FLOOR_:
case H_LOG1P_:
case H_RAD2DEG_:
case H_DEG2RAD_:
case H_NEG_:
case H_NEGATIVE_:
case H_RELU_:
case H_RELU6_:
case H_SELU_:
case H_SILU_:
case H_MISH_:
case H_SIGMOID_:
case H_DETACH_:
case H_SQUEEZE_:
case H_SQUARE_:
case H_T_:
case H_TANH_:
case H_TRUNC_:
case H_FIX_:
case H_ZERO_:
case H_SET_:
case H_SIGN_:
case H_HARDSWISH_:
  init_update_in_place(op);
  redispatch_ptrs_8[op.id - H_ABS_](ks, load<at::Tensor &>()(op.args[0]));
  end_update_in_place(op);
  break;

case H_ABS_OUT:
case H_ABSOLUTE_OUT:
case H_ANGLE_OUT:
case H_SGN_OUT:
case H_CONJ_PHYSICAL_OUT:
case H_ACOS_OUT:
case H_ARCCOS_OUT:
case H_ACOSH_OUT:
case H_ARCCOSH_OUT:
case H_ASINH_OUT:
case H_ARCSINH_OUT:
case H_ATANH_OUT:
case H_ARCTANH_OUT:
case H_ASIN_OUT:
case H_ARCSIN_OUT:
case H_ATAN_OUT:
case H_ARCTAN_OUT:
case H_BITWISE_NOT_OUT:
case H_LOGICAL_NOT_OUT:
case H_CEIL_OUT:
case H_COS_OUT:
case H_COSH_OUT:
case H_ERF_OUT:
case H_ERFC_OUT:
case H_EXP_OUT:
case H_EXP2_OUT:
case H_EXPM1_OUT:
case H_FLOOR_OUT:
case H_FRAC_OUT:
case H_INVERSE_OUT:
case H_LOG_OUT:
case H_LOG10_OUT:
case H_LOG1P_OUT:
case H_LOG2_OUT:
case H_RAD2DEG_OUT:
case H_DEG2RAD_OUT:
case H_RECIPROCAL_OUT:
case H_NEG_OUT:
case H_NEGATIVE_OUT:
case H_ROUND_OUT:
case H_GELU_OUT:
case H_RSQRT_OUT:
case H_SILU_OUT:
case H_MISH_OUT:
case H_SIGMOID_OUT:
case H_SIN_OUT:
case H_SINC_OUT:
case H_SINH_OUT:
case H_SQRT_OUT:
case H_SQUARE_OUT:
case H_TAN_OUT:
case H_TANH_OUT:
case H_TRUNC_OUT:
case H_FIX_OUT:
case H_NONZERO_OUT:
case H_LGAMMA_OUT:
case H_DIGAMMA_OUT:
case H_ERFINV_OUT:
case H_I0_OUT:
case H_SIGN_OUT:
case H_SIGNBIT_OUT:
case H_MSORT_OUT:
case H_HARDSIGMOID_OUT:
case H_HARDSWISH_OUT:
case H_LOG_SIGMOID_OUT:
case H_ISPOSINF_OUT:
case H_ISNEGINF_OUT:
case H_SPECIAL_ENTR_OUT:
case H_SPECIAL_EXPM1_OUT:
case H_SPECIAL_EXP2_OUT:
case H_SPECIAL_PSI_OUT:
case H_SPECIAL_DIGAMMA_OUT:
case H_SPECIAL_GAMMALN_OUT:
case H_SPECIAL_ERF_OUT:
case H_SPECIAL_ERFC_OUT:
case H_SPECIAL_ERFINV_OUT:
case H_SPECIAL_NDTR_OUT:
case H_SPECIAL_I0_OUT:
case H_SPECIAL_I0E_OUT:
case H_SPECIAL_I1_OUT:
case H_SPECIAL_I1E_OUT:
case H_SPECIAL_EXPIT_OUT:
case H_LINALG_CHOLESKY_OUT:
case H_LINALG_DET_OUT:
case H_LINALG_EIGVALS_OUT:
case H_LINALG_INV_OUT:
case H_LINALG_SVDVALS_OUT:
  init_update_in_place(op);
  redispatch_ptrs_9[op.id - H_ABS_OUT](ks, load<const at::Tensor &>()(op.args[0]), load<at::Tensor &>()(op.args[1]));
  end_update_in_place(op);
  break;

case H_AVG_POOL1D:
  set(op, at::redispatch::avg_pool1d(ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<at::IntArrayRef>()(op.args[3]), load<bool>()(op.args[4]), load<bool>()(op.args[5])));
  break;

case H_ADAPTIVE_AVG_POOL1D:
case H_BROADCAST_TO:
case H_COUNT_NONZERO_DIM_INTLIST:
case H_PERMUTE:
case H_REPEAT:
case H_RESHAPE:
case H__MKLDNN_RESHAPE:
case H_SUM_TO_SIZE:
case H_TILE:
case H_FLIP:
case H__UNSAFE_VIEW:
case H__SPARSE_SUM_DIM:
case H_VIEW:
case H_TRACE_BACKWARD:
case H_ADAPTIVE_AVG_POOL2D:
case H_MKLDNN_ADAPTIVE_AVG_POOL2D:
case H__ADAPTIVE_AVG_POOL2D:
case H_ADAPTIVE_AVG_POOL3D:
case H__ADAPTIVE_AVG_POOL3D:
case H_REFLECTION_PAD1D:
case H_REFLECTION_PAD2D:
  set(op, redispatch_ptrs_10[op.id - H_ADAPTIVE_AVG_POOL1D](ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1])));
  break;

case H_ADD_TENSOR:
case H__ADD_RELU_TENSOR:
case H_THRESHOLD_BACKWARD:
case H_WHERE_SCALAROTHER:
case H_SUB_TENSOR:
case H_SUBTRACT_TENSOR:
case H_RSUB_TENSOR:
case H_MASKED_FILL_SCALAR:
case H_DIST:
case H_LERP_SCALAR:
case H__TEST_SERIALIZATION_SUBCMUL:
  set(op, redispatch_ptrs_11[op.id - H_ADD_TENSOR](ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Scalar &>()(op.args[2])));
  break;

case H_ADD__TENSOR:
case H__ADD_RELU__TENSOR:
case H_SUB__TENSOR:
case H_SUBTRACT__TENSOR:
case H_MASKED_FILL__SCALAR:
case H_LERP__SCALAR:
  init_update_in_place(op);
  redispatch_ptrs_12[op.id - H_ADD__TENSOR](ks, load<at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Scalar &>()(op.args[2]));
  end_update_in_place(op);
  break;

case H_ADD_OUT:
case H__ADD_RELU_OUT:
case H_HARDSHRINK_BACKWARD_GRAD_INPUT:
case H_THRESHOLD_BACKWARD_GRAD_INPUT:
case H_SUB_OUT:
case H_SUBTRACT_OUT:
case H_LERP_SCALAR_OUT:
case H_SOFTSHRINK_BACKWARD_GRAD_INPUT:
  init_update_in_place(op);
  redispatch_ptrs_13[op.id - H_ADD_OUT](ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Scalar &>()(op.args[2]), load<at::Tensor &>()(op.args[3]));
  end_update_in_place(op);
  break;

case H_ADD_SCALAR:
case H_THRESHOLD:
case H_WHERE_SCALAR:
case H_SUB_SCALAR:
case H_SUBTRACT_SCALAR:
case H_RSUB_SCALAR:
case H_HARDTANH:
  set(op, redispatch_ptrs_14[op.id - H_ADD_SCALAR](ks, load<const at::Tensor &>()(op.args[0]), load<const at::Scalar &>()(op.args[1]), load<const at::Scalar &>()(op.args[2])));
  break;

case H_ADD__SCALAR:
case H_SUB__SCALAR:
case H_SUBTRACT__SCALAR:
case H_HARDTANH_:
  init_update_in_place(op);
  redispatch_ptrs_15[op.id - H_ADD__SCALAR](ks, load<at::Tensor &>()(op.args[0]), load<const at::Scalar &>()(op.args[1]), load<const at::Scalar &>()(op.args[2]));
  end_update_in_place(op);
  break;

case H_ADDMV_OUT:
case H_ADDR_OUT:
case H_BADDBMM_OUT:
case H_SSPADDMM_OUT:
case H_ADDMM_OUT:
case H_ADDBMM_OUT:
  init_update_in_place(op);
  redispatch_ptrs_16[op.id - H_ADDMV_OUT](ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<const at::Scalar &>()(op.args[3]), load<const at::Scalar &>()(op.args[4]), load<at::Tensor &>()(op.args[5]));
  end_update_in_place(op);
  break;

case H_ADDR:
case H_BADDBMM:
case H_SSPADDMM:
case H__SPARSE_ADDMM:
case H_ADDMM:
case H_ADDBMM:
  set(op, redispatch_ptrs_17[op.id - H_ADDR](ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<const at::Scalar &>()(op.args[3]), load<const at::Scalar &>()(op.args[4])));
  break;

case H_ADDR_:
case H_BADDBMM_:
case H__BADDBMM_MKL_:
case H_ADDMM_:
case H_ADDBMM_:
  init_update_in_place(op);
  redispatch_ptrs_18[op.id - H_ADDR_](ks, load<at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<const at::Scalar &>()(op.args[3]), load<const at::Scalar &>()(op.args[4]));
  end_update_in_place(op);
  break;

case H_AFFINE_GRID_GENERATOR:
case H_AFFINE_GRID_GENERATOR_BACKWARD:
case H_EXPAND:
case H_LOGSUMEXP:
case H_AMAX:
case H_AMIN:
case H_FROBENIUS_NORM_DIM:
case H_NUCLEAR_NORM_DIM:
  set(op, redispatch_ptrs_19[op.id - H_AFFINE_GRID_GENERATOR](ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<bool>()(op.args[2])));
  break;

case H_ALL_DIM:
case H_ANY_DIM:
case H__LOG_SOFTMAX:
case H__SOFTMAX:
case H__SPARSE_SOFTMAX:
case H__SPARSE_LOG_SOFTMAX:
case H_COMBINATIONS:
case H_ARGSORT:
  set(op, redispatch_ptrs_20[op.id - H_ALL_DIM](ks, load<const at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1]), load<bool>()(op.args[2])));
  break;

case H_ALL_OUT:
case H_ANY_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, int64_t, bool, at::Tensor &) = at::redispatch::all_outf;
  if (op.id == H_ANY_OUT) ptr = at::redispatch::any_outf;
  init_update_in_place(op);
  ptr(ks, load<const at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1]), load<bool>()(op.args[2]), load<at::Tensor &>()(op.args[3]));
  end_update_in_place(op);
  break;}

case H_ALL_DIMNAME:
case H_ANY_DIMNAME:
case H_ARGSORT_DIMNAME:
  set(op, redispatch_ptrs_21[op.id - H_ALL_DIMNAME](ks, load<const at::Tensor &>()(op.args[0]), load<at::Dimname>()(op.args[1]), load<bool>()(op.args[2])));
  break;

case H_ALL_DIMNAME_OUT:
case H_ANY_DIMNAME_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, at::Dimname, bool, at::Tensor &) = at::redispatch::all_outf;
  if (op.id == H_ANY_DIMNAME_OUT) ptr = at::redispatch::any_outf;
  init_update_in_place(op);
  ptr(ks, load<const at::Tensor &>()(op.args[0]), load<at::Dimname>()(op.args[1]), load<bool>()(op.args[2]), load<at::Tensor &>()(op.args[3]));
  end_update_in_place(op);
  break;}

case H_ARANGE:
case H_SCALAR_TENSOR:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Scalar &, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<at::Device>, c10::optional<bool>) = at::redispatch::arange;
  if (op.id == H_SCALAR_TENSOR) ptr = at::redispatch::scalar_tensor;
  set(op, ptr(ks, load<const at::Scalar &>()(op.args[0]), load<c10::optional<at::ScalarType>>()(op.args[1]), load<c10::optional<at::Layout>>()(op.args[2]), load<c10::optional<at::Device>>()(op.args[3]), load<c10::optional<bool>>()(op.args[4])));
  break;}

case H_ARANGE_START:
case H_RANGE:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Scalar &, const at::Scalar &, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<at::Device>, c10::optional<bool>) = at::redispatch::arange;
  if (op.id == H_RANGE) ptr = at::redispatch::range;
  set(op, ptr(ks, load<const at::Scalar &>()(op.args[0]), load<const at::Scalar &>()(op.args[1]), load<c10::optional<at::ScalarType>>()(op.args[2]), load<c10::optional<at::Layout>>()(op.args[3]), load<c10::optional<at::Device>>()(op.args[4]), load<c10::optional<bool>>()(op.args[5])));
  break;}

case H_ARANGE_START_STEP:
case H_RANGE_STEP:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Scalar &, const at::Scalar &, const at::Scalar &, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<at::Device>, c10::optional<bool>) = at::redispatch::arange;
  if (op.id == H_RANGE_STEP) ptr = at::redispatch::range;
  set(op, ptr(ks, load<const at::Scalar &>()(op.args[0]), load<const at::Scalar &>()(op.args[1]), load<const at::Scalar &>()(op.args[2]), load<c10::optional<at::ScalarType>>()(op.args[3]), load<c10::optional<at::Layout>>()(op.args[4]), load<c10::optional<at::Device>>()(op.args[5]), load<c10::optional<bool>>()(op.args[6])));
  break;}

case H_ARANGE_OUT:
  init_update_in_place(op);
  at::redispatch::arange_outf(ks, load<const at::Scalar &>()(op.args[0]), load<at::Tensor &>()(op.args[1]));
  end_update_in_place(op);
  break;

case H_ARANGE_START_OUT:
case H_RANGE_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Scalar &, const at::Scalar &, const at::Scalar &, at::Tensor &) = at::redispatch::arange_outf;
  if (op.id == H_RANGE_OUT) ptr = at::redispatch::range_outf;
  init_update_in_place(op);
  ptr(ks, load<const at::Scalar &>()(op.args[0]), load<const at::Scalar &>()(op.args[1]), load<const at::Scalar &>()(op.args[2]), load<at::Tensor &>()(op.args[3]));
  end_update_in_place(op);
  break;}

case H_ARGMAX:
case H_ARGMIN:
case H_VANDER:
  set(op, redispatch_ptrs_22[op.id - H_ARGMAX](ks, load<const at::Tensor &>()(op.args[0]), load<c10::optional<int64_t>>()(op.args[1]), load<bool>()(op.args[2])));
  break;

case H_ARGMAX_OUT:
case H_ARGMIN_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, c10::optional<int64_t>, bool, at::Tensor &) = at::redispatch::argmax_outf;
  if (op.id == H_ARGMIN_OUT) ptr = at::redispatch::argmin_outf;
  init_update_in_place(op);
  ptr(ks, load<const at::Tensor &>()(op.args[0]), load<c10::optional<int64_t>>()(op.args[1]), load<bool>()(op.args[2]), load<at::Tensor &>()(op.args[3]));
  end_update_in_place(op);
  break;}

case H_AS_STRIDED:
  set(op, at::redispatch::as_strided(ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<c10::optional<int64_t>>()(op.args[3])));
  break;

case H_AS_STRIDED_:
  init_update_in_place(op);
  at::redispatch::as_strided_(ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<c10::optional<int64_t>>()(op.args[3]));
  end_update_in_place(op);
  break;

case H_BARTLETT_WINDOW:
case H_BLACKMAN_WINDOW:
case H_EYE:
case H_HANN_WINDOW:
case H_HAMMING_WINDOW:
case H_KAISER_WINDOW:
case H_RANDPERM:
  set(op, redispatch_ptrs_23[op.id - H_BARTLETT_WINDOW](ks, load<int64_t>()(op.args[0]), load<c10::optional<at::ScalarType>>()(op.args[1]), load<c10::optional<at::Layout>>()(op.args[2]), load<c10::optional<at::Device>>()(op.args[3]), load<c10::optional<bool>>()(op.args[4])));
  break;

case H_BARTLETT_WINDOW_PERIODIC:
case H_BLACKMAN_WINDOW_PERIODIC:
case H_HANN_WINDOW_PERIODIC:
case H_HAMMING_WINDOW_PERIODIC:
case H_KAISER_WINDOW_PERIODIC:
  set(op, redispatch_ptrs_24[op.id - H_BARTLETT_WINDOW_PERIODIC](ks, load<int64_t>()(op.args[0]), load<bool>()(op.args[1]), load<c10::optional<at::ScalarType>>()(op.args[2]), load<c10::optional<at::Layout>>()(op.args[3]), load<c10::optional<at::Device>>()(op.args[4]), load<c10::optional<bool>>()(op.args[5])));
  break;

case H_BATCH_NORM:
case H_INSTANCE_NORM:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, bool, double, double, bool) = at::redispatch::batch_norm;
  if (op.id == H_INSTANCE_NORM) ptr = at::redispatch::instance_norm;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<const c10::optional<at::Tensor> &>()(op.args[1]), load<const c10::optional<at::Tensor> &>()(op.args[2]), load<const c10::optional<at::Tensor> &>()(op.args[3]), load<const c10::optional<at::Tensor> &>()(op.args[4]), load<bool>()(op.args[5]), load<double>()(op.args[6]), load<double>()(op.args[7]), load<bool>()(op.args[8])));
  break;}

case H_QUANTIZED_BATCH_NORM:
  set(op, at::redispatch::quantized_batch_norm(ks, load<const at::Tensor &>()(op.args[0]), load<const c10::optional<at::Tensor> &>()(op.args[1]), load<const c10::optional<at::Tensor> &>()(op.args[2]), load<const at::Tensor &>()(op.args[3]), load<const at::Tensor &>()(op.args[4]), load<double>()(op.args[5]), load<double>()(op.args[6]), load<int64_t>()(op.args[7])));
  break;

case H_BERNOULLI:
case H__STANDARD_GAMMA:
case H__SAMPLE_DIRICHLET:
case H_POISSON:
  set(op, redispatch_ptrs_25[op.id - H_BERNOULLI](ks, load<const at::Tensor &>()(op.args[0]), load<c10::optional<at::Generator>>()(op.args[1])));
  break;

case H_BERNOULLI_OUT:
  init_update_in_place(op);
  at::redispatch::bernoulli_outf(ks, load<const at::Tensor &>()(op.args[0]), load<c10::optional<at::Generator>>()(op.args[1]), load<at::Tensor &>()(op.args[2]));
  end_update_in_place(op);
  break;

case H_BERNOULLI__TENSOR:
  init_update_in_place(op);
  at::redispatch::bernoulli_(ks, load<at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<c10::optional<at::Generator>>()(op.args[2]));
  end_update_in_place(op);
  break;

case H_BERNOULLI__FLOAT:
case H_EXPONENTIAL_:
case H_GEOMETRIC_:
  init_update_in_place(op);
  redispatch_ptrs_26[op.id - H_BERNOULLI__FLOAT](ks, load<at::Tensor &>()(op.args[0]), load<double>()(op.args[1]), load<c10::optional<at::Generator>>()(op.args[2]));
  end_update_in_place(op);
  break;

case H_BERNOULLI_P:
case H_NORMAL_TENSOR_FLOAT:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, double, c10::optional<at::Generator>) = at::redispatch::bernoulli;
  if (op.id == H_NORMAL_TENSOR_FLOAT) ptr = at::redispatch::normal;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<double>()(op.args[1]), load<c10::optional<at::Generator>>()(op.args[2])));
  break;}

case H_BILINEAR:
  set(op, at::redispatch::bilinear(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<const c10::optional<at::Tensor> &>()(op.args[3])));
  break;

case H_BINARY_CROSS_ENTROPY:
  set(op, at::redispatch::binary_cross_entropy(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const c10::optional<at::Tensor> &>()(op.args[2]), load<int64_t>()(op.args[3])));
  break;

case H_BINARY_CROSS_ENTROPY_OUT:
  init_update_in_place(op);
  at::redispatch::binary_cross_entropy_outf(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const c10::optional<at::Tensor> &>()(op.args[2]), load<int64_t>()(op.args[3]), load<at::Tensor &>()(op.args[4]));
  end_update_in_place(op);
  break;

case H_BINARY_CROSS_ENTROPY_BACKWARD:
  set(op, at::redispatch::binary_cross_entropy_backward(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<const c10::optional<at::Tensor> &>()(op.args[3]), load<int64_t>()(op.args[4])));
  break;

case H_BINARY_CROSS_ENTROPY_BACKWARD_GRAD_INPUT:
  init_update_in_place(op);
  at::redispatch::binary_cross_entropy_backward_outf(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<const c10::optional<at::Tensor> &>()(op.args[3]), load<int64_t>()(op.args[4]), load<at::Tensor &>()(op.args[5]));
  end_update_in_place(op);
  break;

case H_BINARY_CROSS_ENTROPY_WITH_LOGITS:
  set(op, at::redispatch::binary_cross_entropy_with_logits(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const c10::optional<at::Tensor> &>()(op.args[2]), load<const c10::optional<at::Tensor> &>()(op.args[3]), load<int64_t>()(op.args[4])));
  break;

case H_BINARY_CROSS_ENTROPY_WITH_LOGITS_BACKWARD:
  set(op, at::redispatch::binary_cross_entropy_with_logits_backward(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<const c10::optional<at::Tensor> &>()(op.args[3]), load<const c10::optional<at::Tensor> &>()(op.args[4]), load<int64_t>()(op.args[5])));
  break;

case H_BINCOUNT:
  set(op, at::redispatch::bincount(ks, load<const at::Tensor &>()(op.args[0]), load<const c10::optional<at::Tensor> &>()(op.args[1]), load<int64_t>()(op.args[2])));
  break;

case H_COPYSIGN_OUT:
case H_LOGICAL_XOR_OUT:
case H_LOGICAL_AND_OUT:
case H_LOGICAL_OR_OUT:
case H_BMM_OUT:
case H_CLAMP_MAX_TENSOR_OUT:
case H_CLAMP_MIN_TENSOR_OUT:
case H_COMPLEX_OUT:
case H_POLAR_OUT:
case H_DIV_OUT:
case H_DIVIDE_OUT:
case H_TRUE_DIVIDE_OUT:
case H_DOT_OUT:
case H_VDOT_OUT:
case H_FLOOR_DIVIDE_OUT:
case H_GCD_OUT:
case H_LCM_OUT:
case H_KRON_OUT:
case H_LDEXP_OUT:
case H_LOGADDEXP_OUT:
case H_LOGADDEXP2_OUT:
case H_XLOGY_OUTTENSOR:
case H_MATMUL_OUT:
case H__COMPUTE_LINEAR_COMBINATION_OUT:
case H_MM_OUT:
case H_MUL_OUT:
case H_MULTIPLY_OUT:
case H_MV_OUT:
case H_GELU_BACKWARD_GRAD_INPUT:
case H_HEAVISIDE_OUT:
case H_HSPMM_OUT:
case H_BITWISE_AND_TENSOR_OUT:
case H_BITWISE_OR_TENSOR_OUT:
case H_BITWISE_XOR_TENSOR_OUT:
case H_NE_TENSOR_OUT:
case H_NOT_EQUAL_TENSOR_OUT:
case H_EQ_TENSOR_OUT:
case H_GE_TENSOR_OUT:
case H_GREATER_EQUAL_TENSOR_OUT:
case H_LE_TENSOR_OUT:
case H_LESS_EQUAL_TENSOR_OUT:
case H_GT_TENSOR_OUT:
case H_GREATER_TENSOR_OUT:
case H_LT_TENSOR_OUT:
case H_LESS_TENSOR_OUT:
case H_TAKE_OUT:
case H_MASKED_SELECT_OUT:
case H_ORGQR_OUT:
case H_ATAN2_OUT:
case H_FMOD_TENSOR_OUT:
case H_HYPOT_OUT:
case H_IGAMMA_OUT:
case H_IGAMMAC_OUT:
case H_NEXTAFTER_OUT:
case H_REMAINDER_TENSOR_OUT:
case H_FMIN_OUT:
case H_FMAX_OUT:
case H_MAXIMUM_OUT:
case H_MAX_OUT:
case H_MINIMUM_OUT:
case H_MIN_OUT:
case H_POW_TENSOR_TENSOR_OUT:
case H_FLOAT_POWER_TENSOR_TENSOR_OUT:
case H_HARDSIGMOID_BACKWARD_GRAD_INPUT:
case H_ADAPTIVE_AVG_POOL3D_BACKWARD_GRAD_INPUT:
case H_SIGMOID_BACKWARD_GRAD_INPUT:
case H_TANH_BACKWARD_GRAD_INPUT:
case H_SPECIAL_XLOG1PY_OUT:
case H_LINALG_HOUSEHOLDER_PRODUCT_OUT:
case H_INNER_OUT:
case H_OUTER_OUT:
case H_GER_OUT:
case H_LINALG_SOLVE_OUT:
  init_update_in_place(op);
  redispatch_ptrs_27[op.id - H_COPYSIGN_OUT](ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<at::Tensor &>()(op.args[2]));
  end_update_in_place(op);
  break;

case H_COPYSIGN_SCALAR:
case H_CLAMP_MAX:
case H_CLAMP_MIN:
case H_DIV_SCALAR:
case H_DIVIDE_SCALAR:
case H_TRUE_DIVIDE_SCALAR:
case H_FLOOR_DIVIDE_SCALAR:
case H_XLOGY_SCALAR_OTHER:
case H_MUL_SCALAR:
case H_MULTIPLY_SCALAR:
case H_CELU:
case H_NATIVE_NORM:
case H_NORM_SCALAR:
case H_BITWISE_AND_SCALAR:
case H___AND___SCALAR:
case H_BITWISE_OR_SCALAR:
case H___OR___SCALAR:
case H_BITWISE_XOR_SCALAR:
case H___XOR___SCALAR:
case H___LSHIFT___SCALAR:
case H___RSHIFT___SCALAR:
case H_NE_SCALAR:
case H_NOT_EQUAL_SCALAR:
case H_EQ_SCALAR:
case H_GE_SCALAR:
case H_GREATER_EQUAL_SCALAR:
case H_LE_SCALAR:
case H_LESS_EQUAL_SCALAR:
case H_GT_SCALAR:
case H_GREATER_SCALAR:
case H_LT_SCALAR:
case H_LESS_SCALAR:
case H_FMOD_SCALAR:
case H_REMAINDER_SCALAR:
case H_POW_TENSOR_SCALAR:
case H_FLOAT_POWER_TENSOR_SCALAR:
case H_LEAKY_RELU:
case H_SPECIAL_XLOG1PY_OTHER_SCALAR:
  set(op, redispatch_ptrs_28[op.id - H_COPYSIGN_SCALAR](ks, load<const at::Tensor &>()(op.args[0]), load<const at::Scalar &>()(op.args[1])));
  break;

case H_COPYSIGN__SCALAR:
case H_CLAMP_MAX_:
case H_CLAMP_MIN_:
case H_DIV__SCALAR:
case H_DIVIDE__SCALAR:
case H_TRUE_DIVIDE__SCALAR:
case H_FILL__SCALAR:
case H_FLOOR_DIVIDE__SCALAR:
case H_XLOGY__SCALAR_OTHER:
case H_MUL__SCALAR:
case H_MULTIPLY__SCALAR:
case H_CELU_:
case H_EQ__SCALAR:
case H_BITWISE_AND__SCALAR:
case H___IAND___SCALAR:
case H_BITWISE_OR__SCALAR:
case H___IOR___SCALAR:
case H_BITWISE_XOR__SCALAR:
case H___IXOR___SCALAR:
case H___ILSHIFT___SCALAR:
case H___IRSHIFT___SCALAR:
case H_FMOD__SCALAR:
case H_NE__SCALAR:
case H_NOT_EQUAL__SCALAR:
case H_GE__SCALAR:
case H_GREATER_EQUAL__SCALAR:
case H_LE__SCALAR:
case H_LESS_EQUAL__SCALAR:
case H_GT__SCALAR:
case H_GREATER__SCALAR:
case H_LT__SCALAR:
case H_LESS__SCALAR:
case H_REMAINDER__SCALAR:
case H_FLOAT_POWER__SCALAR:
case H_LEAKY_RELU_:
  init_update_in_place(op);
  redispatch_ptrs_29[op.id - H_COPYSIGN__SCALAR](ks, load<at::Tensor &>()(op.args[0]), load<const at::Scalar &>()(op.args[1]));
  end_update_in_place(op);
  break;

case H_COPYSIGN_SCALAR_OUT:
case H_CLAMP_MAX_OUT:
case H_CLAMP_MIN_OUT:
case H_XLOGY_OUTSCALAR_OTHER:
case H_HARDSHRINK_OUT:
case H_BITWISE_AND_SCALAR_OUT:
case H_BITWISE_OR_SCALAR_OUT:
case H_BITWISE_XOR_SCALAR_OUT:
case H_NE_SCALAR_OUT:
case H_NOT_EQUAL_SCALAR_OUT:
case H_EQ_SCALAR_OUT:
case H_GE_SCALAR_OUT:
case H_GREATER_EQUAL_SCALAR_OUT:
case H_LE_SCALAR_OUT:
case H_LESS_EQUAL_SCALAR_OUT:
case H_GT_SCALAR_OUT:
case H_GREATER_SCALAR_OUT:
case H_LT_SCALAR_OUT:
case H_LESS_SCALAR_OUT:
case H_FMOD_SCALAR_OUT:
case H_REMAINDER_SCALAR_OUT:
case H_POW_TENSOR_SCALAR_OUT:
case H_FLOAT_POWER_TENSOR_SCALAR_OUT:
case H_LEAKY_RELU_OUT:
case H_SOFTSHRINK_OUT:
case H_SPECIAL_XLOG1PY_OTHER_SCALAR_OUT:
  init_update_in_place(op);
  redispatch_ptrs_30[op.id - H_COPYSIGN_SCALAR_OUT](ks, load<const at::Tensor &>()(op.args[0]), load<const at::Scalar &>()(op.args[1]), load<at::Tensor &>()(op.args[2]));
  end_update_in_place(op);
  break;

case H_LOGICAL_XOR_:
case H_LOGICAL_AND_:
case H_LOGICAL_OR_:
case H_CLAMP_MAX__TENSOR:
case H_CLAMP_MIN__TENSOR:
case H_DIV__TENSOR:
case H_DIVIDE__TENSOR:
case H_TRUE_DIVIDE__TENSOR:
case H_FILL__TENSOR:
case H_FLOOR_DIVIDE__TENSOR:
case H_LDEXP_:
case H_XLOGY__TENSOR:
case H_MUL__TENSOR:
case H_MULTIPLY__TENSOR:
case H_SET__SOURCE_TENSOR:
case H_EQ__TENSOR:
case H_BITWISE_AND__TENSOR:
case H___IAND___TENSOR:
case H_BITWISE_OR__TENSOR:
case H___IOR___TENSOR:
case H_BITWISE_XOR__TENSOR:
case H___IXOR___TENSOR:
case H___ILSHIFT___TENSOR:
case H___IRSHIFT___TENSOR:
case H_FMOD__TENSOR:
case H_NE__TENSOR:
case H_NOT_EQUAL__TENSOR:
case H_GE__TENSOR:
case H_GREATER_EQUAL__TENSOR:
case H_LE__TENSOR:
case H_LESS_EQUAL__TENSOR:
case H_GT__TENSOR:
case H_GREATER__TENSOR:
case H_LT__TENSOR:
case H_LESS__TENSOR:
case H_HYPOT_:
case H_NEXTAFTER_:
case H_FLOAT_POWER__TENSOR:
  init_update_in_place(op);
  redispatch_ptrs_31[op.id - H_LOGICAL_XOR_](ks, load<at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]));
  end_update_in_place(op);
  break;

case H__BMM:
case H_CHOLESKY_SOLVE:
case H__CHOLESKY_SOLVE_HELPER:
case H_LINALG_PINV_RCOND_TENSOR:
case H_LINALG_MATRIX_RANK_TOL_TENSOR:
  set(op, redispatch_ptrs_32[op.id - H__BMM](ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<bool>()(op.args[2])));
  break;

case H__BMM_OUT:
case H_CHOLESKY_SOLVE_OUT:
case H_LINALG_PINV_OUT_RCOND_TENSOR:
case H_LINALG_MATRIX_RANK_OUT_TOL_TENSOR:
  init_update_in_place(op);
  redispatch_ptrs_33[op.id - H__BMM_OUT](ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<bool>()(op.args[2]), load<at::Tensor &>()(op.args[3]));
  end_update_in_place(op);
  break;

case H_CAT:
case H_STACK:
case H__STACK:
case H__CAT:
  set(op, redispatch_ptrs_34[op.id - H_CAT](ks, load<at::TensorList>()(op.args[0]), load<int64_t>()(op.args[1])));
  break;

case H_CAT_OUT:
case H_STACK_OUT:
case H__STACK_OUT:
case H__CAT_OUT:
  init_update_in_place(op);
  redispatch_ptrs_35[op.id - H_CAT_OUT](ks, load<at::TensorList>()(op.args[0]), load<int64_t>()(op.args[1]), load<at::Tensor &>()(op.args[2]));
  end_update_in_place(op);
  break;

case H_CAT_NAMES:
  set(op, at::redispatch::cat(ks, load<at::TensorList>()(op.args[0]), load<at::Dimname>()(op.args[1])));
  break;

case H_CAT_NAMES_OUT:
  init_update_in_place(op);
  at::redispatch::cat_outf(ks, load<at::TensorList>()(op.args[0]), load<at::Dimname>()(op.args[1]), load<at::Tensor &>()(op.args[2]));
  end_update_in_place(op);
  break;

case H_BLOCK_DIAG:
case H_CHAIN_MATMUL:
case H_ROW_STACK:
case H_HSTACK:
case H_VSTACK:
case H_DSTACK:
case H_CARTESIAN_PROD:
case H_COLUMN_STACK:
case H_LINALG_MULTI_DOT:
case H_FLATTEN_DENSE_TENSORS:
  set(op, redispatch_ptrs_36[op.id - H_BLOCK_DIAG](ks, load<at::TensorList>()(op.args[0])));
  break;

case H_CHAIN_MATMUL_OUT:
case H_ROW_STACK_OUT:
case H_HSTACK_OUT:
case H_VSTACK_OUT:
case H_DSTACK_OUT:
case H_COLUMN_STACK_OUT:
case H_LINALG_MULTI_DOT_OUT:
  init_update_in_place(op);
  redispatch_ptrs_37[op.id - H_CHAIN_MATMUL_OUT](ks, load<at::TensorList>()(op.args[0]), load<at::Tensor &>()(op.args[1]));
  end_update_in_place(op);
  break;

case H_CLAMP:
case H_CLIP:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const c10::optional<at::Scalar> &, const c10::optional<at::Scalar> &) = at::redispatch::clamp;
  if (op.id == H_CLIP) ptr = at::redispatch::clip;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<const c10::optional<at::Scalar> &>()(op.args[1]), load<const c10::optional<at::Scalar> &>()(op.args[2])));
  break;}

case H_CLAMP_TENSOR:
case H_CLIP_TENSOR:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &) = at::redispatch::clamp;
  if (op.id == H_CLIP_TENSOR) ptr = at::redispatch::clip;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<const c10::optional<at::Tensor> &>()(op.args[1]), load<const c10::optional<at::Tensor> &>()(op.args[2])));
  break;}

case H_CLAMP_:
case H_CLIP_:
  {at::Tensor &(*ptr)(DispatchKeySet, at::Tensor &, const c10::optional<at::Scalar> &, const c10::optional<at::Scalar> &) = at::redispatch::clamp_;
  if (op.id == H_CLIP_) ptr = at::redispatch::clip_;
  init_update_in_place(op);
  ptr(ks, load<at::Tensor &>()(op.args[0]), load<const c10::optional<at::Scalar> &>()(op.args[1]), load<const c10::optional<at::Scalar> &>()(op.args[2]));
  end_update_in_place(op);
  break;}

case H_CLAMP__TENSOR:
case H_CLIP__TENSOR:
  {at::Tensor &(*ptr)(DispatchKeySet, at::Tensor &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &) = at::redispatch::clamp_;
  if (op.id == H_CLIP__TENSOR) ptr = at::redispatch::clip_;
  init_update_in_place(op);
  ptr(ks, load<at::Tensor &>()(op.args[0]), load<const c10::optional<at::Tensor> &>()(op.args[1]), load<const c10::optional<at::Tensor> &>()(op.args[2]));
  end_update_in_place(op);
  break;}

case H_CLAMP_OUT:
case H_CLIP_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, const c10::optional<at::Scalar> &, const c10::optional<at::Scalar> &, at::Tensor &) = at::redispatch::clamp_outf;
  if (op.id == H_CLIP_OUT) ptr = at::redispatch::clip_outf;
  init_update_in_place(op);
  ptr(ks, load<const at::Tensor &>()(op.args[0]), load<const c10::optional<at::Scalar> &>()(op.args[1]), load<const c10::optional<at::Scalar> &>()(op.args[2]), load<at::Tensor &>()(op.args[3]));
  end_update_in_place(op);
  break;}

case H_CLAMP_TENSOR_OUT:
case H_CLIP_TENSOR_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, at::Tensor &) = at::redispatch::clamp_outf;
  if (op.id == H_CLIP_TENSOR_OUT) ptr = at::redispatch::clip_outf;
  init_update_in_place(op);
  ptr(ks, load<const at::Tensor &>()(op.args[0]), load<const c10::optional<at::Tensor> &>()(op.args[1]), load<const c10::optional<at::Tensor> &>()(op.args[2]), load<at::Tensor &>()(op.args[3]));
  end_update_in_place(op);
  break;}

case H_CONSTANT_PAD_ND:
  set(op, at::redispatch::constant_pad_nd(ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<const at::Scalar &>()(op.args[2])));
  break;

case H_CONTIGUOUS:
  set(op, at::redispatch::__dispatch_contiguous(ks, load<const at::Tensor &>()(op.args[0]), load<at::MemoryFormat>()(op.args[1])));
  break;

case H_CONVOLUTION:
case H_CONVOLUTION_OVERRIDEABLE:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, bool, at::IntArrayRef, int64_t) = at::redispatch::convolution;
  if (op.id == H_CONVOLUTION_OVERRIDEABLE) ptr = at::redispatch::convolution_overrideable;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const c10::optional<at::Tensor> &>()(op.args[2]), load<at::IntArrayRef>()(op.args[3]), load<at::IntArrayRef>()(op.args[4]), load<at::IntArrayRef>()(op.args[5]), load<bool>()(op.args[6]), load<at::IntArrayRef>()(op.args[7]), load<int64_t>()(op.args[8])));
  break;}

case H__CONVOLUTION:
  set(op, at::redispatch::_convolution(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const c10::optional<at::Tensor> &>()(op.args[2]), load<at::IntArrayRef>()(op.args[3]), load<at::IntArrayRef>()(op.args[4]), load<at::IntArrayRef>()(op.args[5]), load<bool>()(op.args[6]), load<at::IntArrayRef>()(op.args[7]), load<int64_t>()(op.args[8]), load<bool>()(op.args[9]), load<bool>()(op.args[10]), load<bool>()(op.args[11]), load<bool>()(op.args[12])));
  break;

case H__CONVOLUTION_DEPRECATED:
  set(op, at::redispatch::_convolution(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const c10::optional<at::Tensor> &>()(op.args[2]), load<at::IntArrayRef>()(op.args[3]), load<at::IntArrayRef>()(op.args[4]), load<at::IntArrayRef>()(op.args[5]), load<bool>()(op.args[6]), load<at::IntArrayRef>()(op.args[7]), load<int64_t>()(op.args[8]), load<bool>()(op.args[9]), load<bool>()(op.args[10]), load<bool>()(op.args[11])));
  break;

case H__CONVOLUTION_MODE:
case H_CONV1D_PADDING:
case H_CONV2D_PADDING:
case H_CONV3D_PADDING:
  set(op, redispatch_ptrs_38[op.id - H__CONVOLUTION_MODE](ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const c10::optional<at::Tensor> &>()(op.args[2]), load<at::IntArrayRef>()(op.args[3]), load<c10::string_view>()(op.args[4]), load<at::IntArrayRef>()(op.args[5]), load<int64_t>()(op.args[6])));
  break;

case H__CONVOLUTION_NOGROUP:
  set(op, at::redispatch::_convolution_nogroup(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const c10::optional<at::Tensor> &>()(op.args[2]), load<at::IntArrayRef>()(op.args[3]), load<at::IntArrayRef>()(op.args[4]), load<at::IntArrayRef>()(op.args[5]), load<bool>()(op.args[6]), load<at::IntArrayRef>()(op.args[7])));
  break;

case H_CONV1D:
case H_CONV2D:
case H_CONV3D:
case H_CUDNN_CONVOLUTION_RELU:
case H_MKLDNN_CONVOLUTION:
  set(op, redispatch_ptrs_39[op.id - H_CONV1D](ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const c10::optional<at::Tensor> &>()(op.args[2]), load<at::IntArrayRef>()(op.args[3]), load<at::IntArrayRef>()(op.args[4]), load<at::IntArrayRef>()(op.args[5]), load<int64_t>()(op.args[6])));
  break;

case H_CONV_TBC:
case H_CUMMAXMIN_BACKWARD:
case H__MAKE_PER_CHANNEL_QUANTIZED_TENSOR:
case H_MSE_LOSS_BACKWARD:
case H_L1_LOSS_BACKWARD:
case H_SOFT_MARGIN_LOSS_BACKWARD:
  set(op, redispatch_ptrs_40[op.id - H_CONV_TBC](ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<int64_t>()(op.args[3])));
  break;

case H_CONV_TRANSPOSE1D:
case H_CONV_TRANSPOSE2D_INPUT:
case H_CONV_TRANSPOSE3D_INPUT:
  set(op, redispatch_ptrs_41[op.id - H_CONV_TRANSPOSE1D](ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const c10::optional<at::Tensor> &>()(op.args[2]), load<at::IntArrayRef>()(op.args[3]), load<at::IntArrayRef>()(op.args[4]), load<at::IntArrayRef>()(op.args[5]), load<int64_t>()(op.args[6]), load<at::IntArrayRef>()(op.args[7])));
  break;

case H_COPY_:
case H_COPY_SPARSE_TO_SPARSE_:
  {at::Tensor &(*ptr)(DispatchKeySet, at::Tensor &, const at::Tensor &, bool) = at::redispatch::copy_;
  if (op.id == H_COPY_SPARSE_TO_SPARSE_) ptr = at::redispatch::copy_sparse_to_sparse_;
  init_update_in_place(op);
  ptr(ks, load<at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<bool>()(op.args[2]));
  end_update_in_place(op);
  break;}

case H_COSINE_EMBEDDING_LOSS:
case H_MARGIN_RANKING_LOSS:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, const at::Tensor &, double, int64_t) = at::redispatch::cosine_embedding_loss;
  if (op.id == H_MARGIN_RANKING_LOSS) ptr = at::redispatch::margin_ranking_loss;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<double>()(op.args[3]), load<int64_t>()(op.args[4])));
  break;}

case H_COUNT_NONZERO:
case H_REPEAT_INTERLEAVE_TENSOR:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, c10::optional<int64_t>) = at::redispatch::count_nonzero;
  if (op.id == H_REPEAT_INTERLEAVE_TENSOR) ptr = at::redispatch::repeat_interleave;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<c10::optional<int64_t>>()(op.args[1])));
  break;}

case H_CUDNN_AFFINE_GRID_GENERATOR:
case H_CUDNN_AFFINE_GRID_GENERATOR_BACKWARD:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, int64_t, int64_t, int64_t, int64_t) = at::redispatch::cudnn_affine_grid_generator;
  if (op.id == H_CUDNN_AFFINE_GRID_GENERATOR_BACKWARD) ptr = at::redispatch::cudnn_affine_grid_generator_backward;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1]), load<int64_t>()(op.args[2]), load<int64_t>()(op.args[3]), load<int64_t>()(op.args[4])));
  break;}

case H_CUDNN_CONVOLUTION_DEPRECATED:
case H_MIOPEN_CONVOLUTION:
case H_MIOPEN_DEPTHWISE_CONVOLUTION:
  set(op, redispatch_ptrs_42[op.id - H_CUDNN_CONVOLUTION_DEPRECATED](ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const c10::optional<at::Tensor> &>()(op.args[2]), load<at::IntArrayRef>()(op.args[3]), load<at::IntArrayRef>()(op.args[4]), load<at::IntArrayRef>()(op.args[5]), load<int64_t>()(op.args[6]), load<bool>()(op.args[7]), load<bool>()(op.args[8])));
  break;

case H_CUDNN_CONVOLUTION_DEPRECATED2:
case H_MIOPEN_CONVOLUTION_TRANSPOSE_BACKWARD_INPUT:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t, bool, bool) = at::redispatch::cudnn_convolution;
  if (op.id == H_MIOPEN_CONVOLUTION_TRANSPOSE_BACKWARD_INPUT) ptr = at::redispatch::miopen_convolution_transpose_backward_input;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<at::IntArrayRef>()(op.args[3]), load<at::IntArrayRef>()(op.args[4]), load<int64_t>()(op.args[5]), load<bool>()(op.args[6]), load<bool>()(op.args[7])));
  break;}

case H_CUDNN_CONVOLUTION:
case H_CUDNN_CONVOLUTION_TRANSPOSE_BACKWARD_INPUT:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t, bool, bool, bool) = at::redispatch::cudnn_convolution;
  if (op.id == H_CUDNN_CONVOLUTION_TRANSPOSE_BACKWARD_INPUT) ptr = at::redispatch::cudnn_convolution_transpose_backward_input;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<at::IntArrayRef>()(op.args[3]), load<at::IntArrayRef>()(op.args[4]), load<int64_t>()(op.args[5]), load<bool>()(op.args[6]), load<bool>()(op.args[7]), load<bool>()(op.args[8])));
  break;}

case H_CUDNN_CONVOLUTION_BACKWARD_INPUT:
case H_CUDNN_CONVOLUTION_BACKWARD_WEIGHT:
case H_CUDNN_CONVOLUTION_TRANSPOSE_BACKWARD_WEIGHT:
  set(op, redispatch_ptrs_43[op.id - H_CUDNN_CONVOLUTION_BACKWARD_INPUT](ks, load<at::IntArrayRef>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<at::IntArrayRef>()(op.args[3]), load<at::IntArrayRef>()(op.args[4]), load<at::IntArrayRef>()(op.args[5]), load<int64_t>()(op.args[6]), load<bool>()(op.args[7]), load<bool>()(op.args[8]), load<bool>()(op.args[9])));
  break;

case H_CUDNN_CONVOLUTION_TRANSPOSE_DEPRECATED:
case H_MIOPEN_CONVOLUTION_TRANSPOSE:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t, bool, bool) = at::redispatch::cudnn_convolution_transpose;
  if (op.id == H_MIOPEN_CONVOLUTION_TRANSPOSE) ptr = at::redispatch::miopen_convolution_transpose;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const c10::optional<at::Tensor> &>()(op.args[2]), load<at::IntArrayRef>()(op.args[3]), load<at::IntArrayRef>()(op.args[4]), load<at::IntArrayRef>()(op.args[5]), load<at::IntArrayRef>()(op.args[6]), load<int64_t>()(op.args[7]), load<bool>()(op.args[8]), load<bool>()(op.args[9])));
  break;}

case H_CUDNN_CONVOLUTION_TRANSPOSE_DEPRECATED2:
  set(op, at::redispatch::cudnn_convolution_transpose(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<at::IntArrayRef>()(op.args[3]), load<at::IntArrayRef>()(op.args[4]), load<at::IntArrayRef>()(op.args[5]), load<int64_t>()(op.args[6]), load<bool>()(op.args[7]), load<bool>()(op.args[8])));
  break;

case H_CUDNN_CONVOLUTION_TRANSPOSE:
  set(op, at::redispatch::cudnn_convolution_transpose(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<at::IntArrayRef>()(op.args[3]), load<at::IntArrayRef>()(op.args[4]), load<at::IntArrayRef>()(op.args[5]), load<int64_t>()(op.args[6]), load<bool>()(op.args[7]), load<bool>()(op.args[8]), load<bool>()(op.args[9])));
  break;

case H_CUDNN_CONVOLUTION_ADD_RELU:
  set(op, at::redispatch::cudnn_convolution_add_relu(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<const c10::optional<at::Scalar> &>()(op.args[3]), load<const c10::optional<at::Tensor> &>()(op.args[4]), load<at::IntArrayRef>()(op.args[5]), load<at::IntArrayRef>()(op.args[6]), load<at::IntArrayRef>()(op.args[7]), load<int64_t>()(op.args[8])));
  break;

case H_CUMPROD:
case H_CUMSUM:
case H_LOG_SOFTMAX_INT:
case H_SOFTMAX_INT:
case H__SPARSE_SOFTMAX_INT:
case H__SPARSE_LOG_SOFTMAX_INT:
  set(op, redispatch_ptrs_44[op.id - H_CUMPROD](ks, load<const at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1]), load<c10::optional<at::ScalarType>>()(op.args[2])));
  break;

case H_CUMPROD_:
case H_CUMSUM_:
  {at::Tensor &(*ptr)(DispatchKeySet, at::Tensor &, int64_t, c10::optional<at::ScalarType>) = at::redispatch::cumprod_;
  if (op.id == H_CUMSUM_) ptr = at::redispatch::cumsum_;
  init_update_in_place(op);
  ptr(ks, load<at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1]), load<c10::optional<at::ScalarType>>()(op.args[2]));
  end_update_in_place(op);
  break;}

case H_CUMPROD_OUT:
case H_CUMSUM_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, int64_t, c10::optional<at::ScalarType>, at::Tensor &) = at::redispatch::cumprod_outf;
  if (op.id == H_CUMSUM_OUT) ptr = at::redispatch::cumsum_outf;
  init_update_in_place(op);
  ptr(ks, load<const at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1]), load<c10::optional<at::ScalarType>>()(op.args[2]), load<at::Tensor &>()(op.args[3]));
  end_update_in_place(op);
  break;}

case H_CUMPROD_DIMNAME:
case H_CUMSUM_DIMNAME:
case H_LOG_SOFTMAX_DIMNAME:
case H_SOFTMAX_DIMNAME:
case H__SPARSE_SOFTMAX_DIMNAME:
case H__SPARSE_LOG_SOFTMAX_DIMNAME:
  set(op, redispatch_ptrs_45[op.id - H_CUMPROD_DIMNAME](ks, load<const at::Tensor &>()(op.args[0]), load<at::Dimname>()(op.args[1]), load<c10::optional<at::ScalarType>>()(op.args[2])));
  break;

case H_CUMPROD__DIMNAME:
case H_CUMSUM__DIMNAME:
  {at::Tensor &(*ptr)(DispatchKeySet, at::Tensor &, at::Dimname, c10::optional<at::ScalarType>) = at::redispatch::cumprod_;
  if (op.id == H_CUMSUM__DIMNAME) ptr = at::redispatch::cumsum_;
  init_update_in_place(op);
  ptr(ks, load<at::Tensor &>()(op.args[0]), load<at::Dimname>()(op.args[1]), load<c10::optional<at::ScalarType>>()(op.args[2]));
  end_update_in_place(op);
  break;}

case H_CUMPROD_DIMNAME_OUT:
case H_CUMSUM_DIMNAME_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, at::Dimname, c10::optional<at::ScalarType>, at::Tensor &) = at::redispatch::cumprod_outf;
  if (op.id == H_CUMSUM_DIMNAME_OUT) ptr = at::redispatch::cumsum_outf;
  init_update_in_place(op);
  ptr(ks, load<const at::Tensor &>()(op.args[0]), load<at::Dimname>()(op.args[1]), load<c10::optional<at::ScalarType>>()(op.args[2]), load<at::Tensor &>()(op.args[3]));
  end_update_in_place(op);
  break;}

case H_CUMPROD_BACKWARD:
case H__LOG_SOFTMAX_BACKWARD_DATA:
case H__SOFTMAX_BACKWARD_DATA:
case H__SPARSE_SOFTMAX_BACKWARD_DATA:
case H__SPARSE_LOG_SOFTMAX_BACKWARD_DATA:
  set(op, redispatch_ptrs_46[op.id - H_CUMPROD_BACKWARD](ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<int64_t>()(op.args[2]), load<const at::Tensor &>()(op.args[3])));
  break;

case H_CTC_LOSS_INTLIST:
  set(op, at::redispatch::ctc_loss(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<at::IntArrayRef>()(op.args[3]), load<int64_t>()(op.args[4]), load<int64_t>()(op.args[5]), load<bool>()(op.args[6])));
  break;

case H_CTC_LOSS_TENSOR:
  set(op, at::redispatch::ctc_loss(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<const at::Tensor &>()(op.args[3]), load<int64_t>()(op.args[4]), load<int64_t>()(op.args[5]), load<bool>()(op.args[6])));
  break;

case H__CTC_LOSS_BACKWARD:
  set(op, at::redispatch::_ctc_loss_backward(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<at::IntArrayRef>()(op.args[3]), load<at::IntArrayRef>()(op.args[4]), load<const at::Tensor &>()(op.args[5]), load<const at::Tensor &>()(op.args[6]), load<int64_t>()(op.args[7]), load<bool>()(op.args[8])));
  break;

case H_DIAG_EMBED:
case H_DIAGONAL:
case H_NARROW_COPY:
case H_NARROW:
case H_UNFOLD:
case H__REMOVE_BATCH_DIM:
  set(op, redispatch_ptrs_47[op.id - H_DIAG_EMBED](ks, load<const at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1]), load<int64_t>()(op.args[2]), load<int64_t>()(op.args[3])));
  break;

case H_DIAGONAL_DIMNAME:
  set(op, at::redispatch::diagonal(ks, load<const at::Tensor &>()(op.args[0]), load<at::Dimname>()(op.args[1]), load<at::Dimname>()(op.args[2]), load<at::Dimname>()(op.args[3]), load<int64_t>()(op.args[4])));
  break;

case H_DIAGONAL_BACKWARD:
case H_UNFOLD_BACKWARD:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, at::IntArrayRef, int64_t, int64_t, int64_t) = at::redispatch::diagonal_backward;
  if (op.id == H_UNFOLD_BACKWARD) ptr = at::redispatch::unfold_backward;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<int64_t>()(op.args[2]), load<int64_t>()(op.args[3]), load<int64_t>()(op.args[4])));
  break;}

case H_FILL_DIAGONAL_:
  init_update_in_place(op);
  at::redispatch::fill_diagonal_(ks, load<at::Tensor &>()(op.args[0]), load<const at::Scalar &>()(op.args[1]), load<bool>()(op.args[2]));
  end_update_in_place(op);
  break;

case H_DIFF:
  set(op, at::redispatch::diff(ks, load<const at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1]), load<int64_t>()(op.args[2]), load<const c10::optional<at::Tensor> &>()(op.args[3]), load<const c10::optional<at::Tensor> &>()(op.args[4])));
  break;

case H_DIFF_OUT:
  init_update_in_place(op);
  at::redispatch::diff_outf(ks, load<const at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1]), load<int64_t>()(op.args[2]), load<const c10::optional<at::Tensor> &>()(op.args[3]), load<const c10::optional<at::Tensor> &>()(op.args[4]), load<at::Tensor &>()(op.args[5]));
  end_update_in_place(op);
  break;

case H_DIV_TENSOR_MODE:
case H_DIVIDE_TENSOR_MODE:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, c10::optional<c10::string_view>) = at::redispatch::div;
  if (op.id == H_DIVIDE_TENSOR_MODE) ptr = at::redispatch::divide;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<c10::optional<c10::string_view>>()(op.args[2])));
  break;}

case H_DIV__TENSOR_MODE:
case H_DIVIDE__TENSOR_MODE:
  {at::Tensor &(*ptr)(DispatchKeySet, at::Tensor &, const at::Tensor &, c10::optional<c10::string_view>) = at::redispatch::div_;
  if (op.id == H_DIVIDE__TENSOR_MODE) ptr = at::redispatch::divide_;
  init_update_in_place(op);
  ptr(ks, load<at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<c10::optional<c10::string_view>>()(op.args[2]));
  end_update_in_place(op);
  break;}

case H_DIV_OUT_MODE:
case H_DIVIDE_OUT_MODE:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, c10::optional<c10::string_view>, at::Tensor &) = at::redispatch::div_outf;
  if (op.id == H_DIVIDE_OUT_MODE) ptr = at::redispatch::divide_outf;
  init_update_in_place(op);
  ptr(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<c10::optional<c10::string_view>>()(op.args[2]), load<at::Tensor &>()(op.args[3]));
  end_update_in_place(op);
  break;}

case H_DIV_SCALAR_MODE:
case H_DIVIDE_SCALAR_MODE:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Scalar &, c10::optional<c10::string_view>) = at::redispatch::div;
  if (op.id == H_DIVIDE_SCALAR_MODE) ptr = at::redispatch::divide;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Scalar &>()(op.args[1]), load<c10::optional<c10::string_view>>()(op.args[2])));
  break;}

case H_DIV__SCALAR_MODE:
case H_DIVIDE__SCALAR_MODE:
  {at::Tensor &(*ptr)(DispatchKeySet, at::Tensor &, const at::Scalar &, c10::optional<c10::string_view>) = at::redispatch::div_;
  if (op.id == H_DIVIDE__SCALAR_MODE) ptr = at::redispatch::divide_;
  init_update_in_place(op);
  ptr(ks, load<at::Tensor &>()(op.args[0]), load<const at::Scalar &>()(op.args[1]), load<c10::optional<c10::string_view>>()(op.args[2]));
  end_update_in_place(op);
  break;}

case H_EINSUM:
  set(op, at::redispatch::einsum(ks, load<c10::string_view>()(op.args[0]), load<at::TensorList>()(op.args[1])));
  break;

case H_EMBEDDING:
  set(op, at::redispatch::embedding(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<int64_t>()(op.args[2]), load<bool>()(op.args[3]), load<bool>()(op.args[4])));
  break;

case H_EMBEDDING_BACKWARD:
  set(op, at::redispatch::embedding_backward(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<int64_t>()(op.args[2]), load<int64_t>()(op.args[3]), load<bool>()(op.args[4]), load<bool>()(op.args[5])));
  break;

case H_EMBEDDING_DENSE_BACKWARD:
case H_EMBEDDING_SPARSE_BACKWARD:
case H_GRID_SAMPLER:
case H_GRID_SAMPLER_2D:
case H__GRID_SAMPLER_2D_CPU_FALLBACK:
case H_GRID_SAMPLER_3D:
  set(op, redispatch_ptrs_48[op.id - H_EMBEDDING_DENSE_BACKWARD](ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<int64_t>()(op.args[2]), load<int64_t>()(op.args[3]), load<bool>()(op.args[4])));
  break;

case H_EMBEDDING_RENORM_:
  init_update_in_place(op);
  at::redispatch::embedding_renorm_(ks, load<at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<double>()(op.args[2]), load<double>()(op.args[3]));
  end_update_in_place(op);
  break;

case H__EMBEDDING_BAG_BACKWARD:
  set(op, at::redispatch::_embedding_bag_backward(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<const at::Tensor &>()(op.args[3]), load<const at::Tensor &>()(op.args[4]), load<const at::Tensor &>()(op.args[5]), load<int64_t>()(op.args[6]), load<bool>()(op.args[7]), load<int64_t>()(op.args[8]), load<bool>()(op.args[9]), load<const c10::optional<at::Tensor> &>()(op.args[10]), load<int64_t>()(op.args[11])));
  break;

case H__EMBEDDING_BAG_SPARSE_BACKWARD:
case H__EMBEDDING_BAG_DENSE_BACKWARD:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t, bool, int64_t, const c10::optional<at::Tensor> &, int64_t) = at::redispatch::_embedding_bag_sparse_backward;
  if (op.id == H__EMBEDDING_BAG_DENSE_BACKWARD) ptr = at::redispatch::_embedding_bag_dense_backward;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<const at::Tensor &>()(op.args[3]), load<const at::Tensor &>()(op.args[4]), load<int64_t>()(op.args[5]), load<bool>()(op.args[6]), load<int64_t>()(op.args[7]), load<const c10::optional<at::Tensor> &>()(op.args[8]), load<int64_t>()(op.args[9])));
  break;}

case H__EMBEDDING_BAG_PER_SAMPLE_WEIGHTS_BACKWARD:
  set(op, at::redispatch::_embedding_bag_per_sample_weights_backward(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<const at::Tensor &>()(op.args[3]), load<const at::Tensor &>()(op.args[4]), load<int64_t>()(op.args[5]), load<int64_t>()(op.args[6])));
  break;

case H_EMPTY_NAMES:
  set(op, at::redispatch::empty(ks, load<at::IntArrayRef>()(op.args[0]), load<c10::optional<at::DimnameList>>()(op.args[1]), load<c10::optional<at::ScalarType>>()(op.args[2]), load<c10::optional<at::Layout>>()(op.args[3]), load<c10::optional<at::Device>>()(op.args[4]), load<c10::optional<bool>>()(op.args[5]), load<c10::optional<at::MemoryFormat>>()(op.args[6])));
  break;

case H_EMPTY_MEMORY_FORMAT:
  set(op, at::redispatch::empty(ks, load<at::IntArrayRef>()(op.args[0]), load<c10::optional<at::ScalarType>>()(op.args[1]), load<c10::optional<at::Layout>>()(op.args[2]), load<c10::optional<at::Device>>()(op.args[3]), load<c10::optional<bool>>()(op.args[4]), load<c10::optional<at::MemoryFormat>>()(op.args[5])));
  break;

case H_NEW_EMPTY:
case H_NEW_ZEROS:
case H_NEW_ONES:
  set(op, redispatch_ptrs_49[op.id - H_NEW_EMPTY](ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<c10::optional<at::ScalarType>>()(op.args[2]), load<c10::optional<at::Layout>>()(op.args[3]), load<c10::optional<at::Device>>()(op.args[4]), load<c10::optional<bool>>()(op.args[5])));
  break;

case H_NEW_EMPTY_STRIDED:
  set(op, at::redispatch::new_empty_strided(ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<c10::optional<at::ScalarType>>()(op.args[3]), load<c10::optional<at::Layout>>()(op.args[4]), load<c10::optional<at::Device>>()(op.args[5]), load<c10::optional<bool>>()(op.args[6])));
  break;

case H_NEW_FULL:
  set(op, at::redispatch::new_full(ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<const at::Scalar &>()(op.args[2]), load<c10::optional<at::ScalarType>>()(op.args[3]), load<c10::optional<at::Layout>>()(op.args[4]), load<c10::optional<at::Device>>()(op.args[5]), load<c10::optional<bool>>()(op.args[6])));
  break;

case H__EMPTY_AFFINE_QUANTIZED:
  set(op, at::redispatch::_empty_affine_quantized(ks, load<at::IntArrayRef>()(op.args[0]), load<c10::optional<at::ScalarType>>()(op.args[1]), load<c10::optional<at::Layout>>()(op.args[2]), load<c10::optional<at::Device>>()(op.args[3]), load<c10::optional<bool>>()(op.args[4]), load<double>()(op.args[5]), load<int64_t>()(op.args[6]), load<c10::optional<at::MemoryFormat>>()(op.args[7])));
  break;

case H__EMPTY_PER_CHANNEL_AFFINE_QUANTIZED:
  set(op, at::redispatch::_empty_per_channel_affine_quantized(ks, load<at::IntArrayRef>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<int64_t>()(op.args[3]), load<c10::optional<at::ScalarType>>()(op.args[4]), load<c10::optional<at::Layout>>()(op.args[5]), load<c10::optional<at::Device>>()(op.args[6]), load<c10::optional<bool>>()(op.args[7]), load<c10::optional<at::MemoryFormat>>()(op.args[8])));
  break;

case H_RESIZE_:
  init_update_in_place(op);
  at::redispatch::resize_(ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<c10::optional<at::MemoryFormat>>()(op.args[2]));
  end_update_in_place(op);
  break;

case H_EMPTY_QUANTIZED:
  set(op, at::redispatch::empty_quantized(ks, load<at::IntArrayRef>()(op.args[0]), load<const at::Tensor &>()(op.args[1])));
  break;

case H_EMPTY_OUT:
  init_update_in_place(op);
  at::redispatch::empty_outf(ks, load<at::IntArrayRef>()(op.args[0]), load<c10::optional<at::MemoryFormat>>()(op.args[1]), load<at::Tensor &>()(op.args[2]));
  end_update_in_place(op);
  break;

case H_EMPTY_LIKE:
case H_ONES_LIKE:
case H_RAND_LIKE:
case H_RANDN_LIKE:
case H_ZEROS_LIKE:
  set(op, redispatch_ptrs_50[op.id - H_EMPTY_LIKE](ks, load<const at::Tensor &>()(op.args[0]), load<c10::optional<at::ScalarType>>()(op.args[1]), load<c10::optional<at::Layout>>()(op.args[2]), load<c10::optional<at::Device>>()(op.args[3]), load<c10::optional<bool>>()(op.args[4]), load<c10::optional<at::MemoryFormat>>()(op.args[5])));
  break;

case H_EMPTY_STRIDED:
  set(op, at::redispatch::empty_strided(ks, load<at::IntArrayRef>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<c10::optional<at::ScalarType>>()(op.args[2]), load<c10::optional<at::Layout>>()(op.args[3]), load<c10::optional<at::Device>>()(op.args[4]), load<c10::optional<bool>>()(op.args[5])));
  break;

case H_EYE_M:
  set(op, at::redispatch::eye(ks, load<int64_t>()(op.args[0]), load<int64_t>()(op.args[1]), load<c10::optional<at::ScalarType>>()(op.args[2]), load<c10::optional<at::Layout>>()(op.args[3]), load<c10::optional<at::Device>>()(op.args[4]), load<c10::optional<bool>>()(op.args[5])));
  break;

case H_EYE_OUT:
case H_RANDPERM_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, int64_t, at::Tensor &) = at::redispatch::eye_outf;
  if (op.id == H_RANDPERM_OUT) ptr = at::redispatch::randperm_outf;
  init_update_in_place(op);
  ptr(ks, load<int64_t>()(op.args[0]), load<at::Tensor &>()(op.args[1]));
  end_update_in_place(op);
  break;}

case H_EYE_M_OUT:
  init_update_in_place(op);
  at::redispatch::eye_outf(ks, load<int64_t>()(op.args[0]), load<int64_t>()(op.args[1]), load<at::Tensor &>()(op.args[2]));
  end_update_in_place(op);
  break;

case H_FLATTEN_USING_INTS:
case H_FBGEMM_PACK_QUANTIZED_MATRIX_KN:
case H_MOVEDIM_INT:
case H_MOVEAXIS_INT:
case H_SELECT_INT:
case H_TRANSPOSE_INT:
case H__MKLDNN_TRANSPOSE:
case H_NORM_EXCEPT_DIM:
case H_SWAPAXES:
case H_SWAPDIMS:
case H__ADD_BATCH_DIM:
case H__TEST_AMBIGUOUS_DEFAULTS_A:
  set(op, redispatch_ptrs_51[op.id - H_FLATTEN_USING_INTS](ks, load<const at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1]), load<int64_t>()(op.args[2])));
  break;

case H_FLATTEN_NAMED_OUT_DIM:
  set(op, at::redispatch::flatten(ks, load<const at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1]), load<int64_t>()(op.args[2]), load<at::Dimname>()(op.args[3])));
  break;

case H_FLATTEN_USING_NAMES:
  set(op, at::redispatch::flatten(ks, load<const at::Tensor &>()(op.args[0]), load<at::Dimname>()(op.args[1]), load<at::Dimname>()(op.args[2]), load<at::Dimname>()(op.args[3])));
  break;

case H_FLATTEN_DIMNAMELIST:
  set(op, at::redispatch::flatten(ks, load<const at::Tensor &>()(op.args[0]), load<at::DimnameList>()(op.args[1]), load<at::Dimname>()(op.args[2])));
  break;

case H_UNFLATTEN_INT:
  set(op, at::redispatch::unflatten(ks, load<const at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<c10::optional<at::DimnameList>>()(op.args[3])));
  break;

case H_UNFLATTEN_DIMNAME:
  set(op, at::redispatch::unflatten(ks, load<const at::Tensor &>()(op.args[0]), load<at::Dimname>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<at::DimnameList>()(op.args[3])));
  break;

case H_FULL_NAMES:
  set(op, at::redispatch::full(ks, load<at::IntArrayRef>()(op.args[0]), load<const at::Scalar &>()(op.args[1]), load<c10::optional<at::DimnameList>>()(op.args[2]), load<c10::optional<at::ScalarType>>()(op.args[3]), load<c10::optional<at::Layout>>()(op.args[4]), load<c10::optional<at::Device>>()(op.args[5]), load<c10::optional<bool>>()(op.args[6])));
  break;

case H_FULL:
  set(op, at::redispatch::full(ks, load<at::IntArrayRef>()(op.args[0]), load<const at::Scalar &>()(op.args[1]), load<c10::optional<at::ScalarType>>()(op.args[2]), load<c10::optional<at::Layout>>()(op.args[3]), load<c10::optional<at::Device>>()(op.args[4]), load<c10::optional<bool>>()(op.args[5])));
  break;

case H_FULL_OUT:
  init_update_in_place(op);
  at::redispatch::full_outf(ks, load<at::IntArrayRef>()(op.args[0]), load<const at::Scalar &>()(op.args[1]), load<at::Tensor &>()(op.args[2]));
  end_update_in_place(op);
  break;

case H_FULL_LIKE:
  set(op, at::redispatch::full_like(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Scalar &>()(op.args[1]), load<c10::optional<at::ScalarType>>()(op.args[2]), load<c10::optional<at::Layout>>()(op.args[3]), load<c10::optional<at::Device>>()(op.args[4]), load<c10::optional<bool>>()(op.args[5]), load<c10::optional<at::MemoryFormat>>()(op.args[6])));
  break;

case H_FROM_FILE:
  set(op, at::redispatch::from_file(ks, load<c10::string_view>()(op.args[0]), load<c10::optional<bool>>()(op.args[1]), load<c10::optional<int64_t>>()(op.args[2]), load<c10::optional<at::ScalarType>>()(op.args[3]), load<c10::optional<at::Layout>>()(op.args[4]), load<c10::optional<at::Device>>()(op.args[5]), load<c10::optional<bool>>()(op.args[6])));
  break;

case H_HAMMING_WINDOW_PERIODIC_ALPHA:
case H_KAISER_WINDOW_BETA:
  {at::Tensor(*ptr)(DispatchKeySet, int64_t, bool, double, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<at::Device>, c10::optional<bool>) = at::redispatch::hamming_window;
  if (op.id == H_KAISER_WINDOW_BETA) ptr = at::redispatch::kaiser_window;
  set(op, ptr(ks, load<int64_t>()(op.args[0]), load<bool>()(op.args[1]), load<double>()(op.args[2]), load<c10::optional<at::ScalarType>>()(op.args[3]), load<c10::optional<at::Layout>>()(op.args[4]), load<c10::optional<at::Device>>()(op.args[5]), load<c10::optional<bool>>()(op.args[6])));
  break;}

case H_HAMMING_WINDOW_PERIODIC_ALPHA_BETA:
  set(op, at::redispatch::hamming_window(ks, load<int64_t>()(op.args[0]), load<bool>()(op.args[1]), load<double>()(op.args[2]), load<double>()(op.args[3]), load<c10::optional<at::ScalarType>>()(op.args[4]), load<c10::optional<at::Layout>>()(op.args[5]), load<c10::optional<at::Device>>()(op.args[6]), load<c10::optional<bool>>()(op.args[7])));
  break;

case H_HINGE_EMBEDDING_LOSS:
  set(op, at::redispatch::hinge_embedding_loss(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<double>()(op.args[2]), load<int64_t>()(op.args[3])));
  break;

case H_GROUP_NORM:
  set(op, at::redispatch::group_norm(ks, load<const at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1]), load<const c10::optional<at::Tensor> &>()(op.args[2]), load<const c10::optional<at::Tensor> &>()(op.args[3]), load<double>()(op.args[4]), load<bool>()(op.args[5])));
  break;

case H__FFT_R2C:
case H__FFT_C2C:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, at::IntArrayRef, int64_t, bool) = at::redispatch::_fft_r2c;
  if (op.id == H__FFT_C2C) ptr = at::redispatch::_fft_c2c;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<int64_t>()(op.args[2]), load<bool>()(op.args[3])));
  break;}

case H__FFT_R2C_OUT:
case H__FFT_C2C_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, at::IntArrayRef, int64_t, bool, at::Tensor &) = at::redispatch::_fft_r2c_outf;
  if (op.id == H__FFT_C2C_OUT) ptr = at::redispatch::_fft_c2c_outf;
  init_update_in_place(op);
  ptr(ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<int64_t>()(op.args[2]), load<bool>()(op.args[3]), load<at::Tensor &>()(op.args[4]));
  end_update_in_place(op);
  break;}

case H__FFT_C2R:
case H_SELECT_BACKWARD:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, at::IntArrayRef, int64_t, int64_t) = at::redispatch::_fft_c2r;
  if (op.id == H_SELECT_BACKWARD) ptr = at::redispatch::select_backward;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<int64_t>()(op.args[2]), load<int64_t>()(op.args[3])));
  break;}

case H__FFT_C2R_OUT:
  init_update_in_place(op);
  at::redispatch::_fft_c2r_outf(ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<int64_t>()(op.args[2]), load<int64_t>()(op.args[3]), load<at::Tensor &>()(op.args[4]));
  end_update_in_place(op);
  break;

case H_INDEX_TENSOR:
  set(op, at::redispatch::index(ks, load<const at::Tensor &>()(op.args[0]), load<const c10::List<c10::optional<at::Tensor>> &>()(op.args[1])));
  break;

case H_INDEX_COPY_:
case H_INDEX_ADD_:
case H_INDEX_FILL__INT_TENSOR:
case H__INDEX_COPY_:
  init_update_in_place(op);
  redispatch_ptrs_52[op.id - H_INDEX_COPY_](ks, load<at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<const at::Tensor &>()(op.args[3]));
  end_update_in_place(op);
  break;

case H_INDEX_COPY:
case H_INDEX_ADD:
case H_INDEX_FILL_INT_TENSOR:
case H__GATHER_SPARSE_BACKWARD:
  set(op, redispatch_ptrs_53[op.id - H_INDEX_COPY](ks, load<const at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<const at::Tensor &>()(op.args[3])));
  break;

case H_INDEX_COPY__DIMNAME:
case H_INDEX_FILL__DIMNAME_TENSOR:
  {at::Tensor &(*ptr)(DispatchKeySet, at::Tensor &, at::Dimname, const at::Tensor &, const at::Tensor &) = at::redispatch::index_copy_;
  if (op.id == H_INDEX_FILL__DIMNAME_TENSOR) ptr = at::redispatch::index_fill_;
  init_update_in_place(op);
  ptr(ks, load<at::Tensor &>()(op.args[0]), load<at::Dimname>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<const at::Tensor &>()(op.args[3]));
  end_update_in_place(op);
  break;}

case H_INDEX_COPY_DIMNAME:
case H_INDEX_FILL_DIMNAME_TENSOR:
case H_SCATTER_DIMNAME_SRC:
case H_SCATTER_ADD_DIMNAME:
  set(op, redispatch_ptrs_54[op.id - H_INDEX_COPY_DIMNAME](ks, load<const at::Tensor &>()(op.args[0]), load<at::Dimname>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<const at::Tensor &>()(op.args[3])));
  break;

case H_INDEX_PUT_:
  init_update_in_place(op);
  at::redispatch::index_put_(ks, load<at::Tensor &>()(op.args[0]), load<const c10::List<c10::optional<at::Tensor>> &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<bool>()(op.args[3]));
  end_update_in_place(op);
  break;

case H_INDEX_PUT:
  set(op, at::redispatch::index_put(ks, load<const at::Tensor &>()(op.args[0]), load<const c10::List<c10::optional<at::Tensor>> &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<bool>()(op.args[3])));
  break;

case H__INDEX_PUT_IMPL_:
  init_update_in_place(op);
  at::redispatch::_index_put_impl_(ks, load<at::Tensor &>()(op.args[0]), load<const c10::List<c10::optional<at::Tensor>> &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<bool>()(op.args[3]), load<bool>()(op.args[4]));
  end_update_in_place(op);
  break;

case H_ISCLOSE:
case H_PAIRWISE_DISTANCE:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, double, double, bool) = at::redispatch::isclose;
  if (op.id == H_PAIRWISE_DISTANCE) ptr = at::redispatch::pairwise_distance;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<double>()(op.args[2]), load<double>()(op.args[3]), load<bool>()(op.args[4])));
  break;}

case H_KL_DIV:
  set(op, at::redispatch::kl_div(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<int64_t>()(op.args[2]), load<bool>()(op.args[3])));
  break;

case H_KL_DIV_BACKWARD:
  set(op, at::redispatch::kl_div_backward(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<int64_t>()(op.args[3]), load<bool>()(op.args[4])));
  break;

case H_LAYER_NORM:
  set(op, at::redispatch::layer_norm(ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<const c10::optional<at::Tensor> &>()(op.args[2]), load<const c10::optional<at::Tensor> &>()(op.args[3]), load<double>()(op.args[4]), load<bool>()(op.args[5])));
  break;

case H_NAN_TO_NUM:
  set(op, at::redispatch::nan_to_num(ks, load<const at::Tensor &>()(op.args[0]), load<c10::optional<double>>()(op.args[1]), load<c10::optional<double>>()(op.args[2]), load<c10::optional<double>>()(op.args[3])));
  break;

case H_NAN_TO_NUM_:
  init_update_in_place(op);
  at::redispatch::nan_to_num_(ks, load<at::Tensor &>()(op.args[0]), load<c10::optional<double>>()(op.args[1]), load<c10::optional<double>>()(op.args[2]), load<c10::optional<double>>()(op.args[3]));
  end_update_in_place(op);
  break;

case H_NAN_TO_NUM_OUT:
  init_update_in_place(op);
  at::redispatch::nan_to_num_outf(ks, load<const at::Tensor &>()(op.args[0]), load<c10::optional<double>>()(op.args[1]), load<c10::optional<double>>()(op.args[2]), load<c10::optional<double>>()(op.args[3]), load<at::Tensor &>()(op.args[4]));
  end_update_in_place(op);
  break;

case H_LINEAR:
case H_MKLDNN_LINEAR:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &) = at::redispatch::linear;
  if (op.id == H_MKLDNN_LINEAR) ptr = at::redispatch::mkldnn_linear;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const c10::optional<at::Tensor> &>()(op.args[2])));
  break;}

case H_MKLDNN_LINEAR_BACKWARD_INPUT:
  set(op, at::redispatch::mkldnn_linear_backward_input(ks, load<at::IntArrayRef>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2])));
  break;

case H_FBGEMM_LINEAR_INT8_WEIGHT_FP32_ACTIVATION:
case H_FBGEMM_LINEAR_INT8_WEIGHT:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Scalar &, const at::Scalar &, const at::Tensor &) = at::redispatch::fbgemm_linear_int8_weight_fp32_activation;
  if (op.id == H_FBGEMM_LINEAR_INT8_WEIGHT) ptr = at::redispatch::fbgemm_linear_int8_weight;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<const at::Tensor &>()(op.args[3]), load<const at::Scalar &>()(op.args[4]), load<const at::Scalar &>()(op.args[5]), load<const at::Tensor &>()(op.args[6])));
  break;}

case H_FBGEMM_LINEAR_FP16_WEIGHT_FP32_ACTIVATION:
case H_FBGEMM_LINEAR_FP16_WEIGHT:
case H_WHERE_SELF:
case H__S_WHERE:
case H__DIRICHLET_GRAD:
case H_MASKED_FILL_TENSOR:
case H_MASKED_SCATTER:
case H_MASKED_SELECT_BACKWARD:
case H_LU_SOLVE:
case H_LERP_TENSOR:
case H_LOG_SIGMOID_BACKWARD:
  set(op, redispatch_ptrs_55[op.id - H_FBGEMM_LINEAR_FP16_WEIGHT_FP32_ACTIVATION](ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2])));
  break;

case H_LINSPACE:
  set(op, at::redispatch::linspace(ks, load<const at::Scalar &>()(op.args[0]), load<const at::Scalar &>()(op.args[1]), load<c10::optional<int64_t>>()(op.args[2]), load<c10::optional<at::ScalarType>>()(op.args[3]), load<c10::optional<at::Layout>>()(op.args[4]), load<c10::optional<at::Device>>()(op.args[5]), load<c10::optional<bool>>()(op.args[6])));
  break;

case H_LINSPACE_OUT:
  init_update_in_place(op);
  at::redispatch::linspace_outf(ks, load<const at::Scalar &>()(op.args[0]), load<const at::Scalar &>()(op.args[1]), load<c10::optional<int64_t>>()(op.args[2]), load<at::Tensor &>()(op.args[3]));
  end_update_in_place(op);
  break;

case H_XLOGY_SCALAR_SELF:
case H_REMAINDER_SCALAR_TENSOR:
case H_FLOAT_POWER_SCALAR:
case H_SPECIAL_XLOG1PY_SELF_SCALAR:
  set(op, redispatch_ptrs_56[op.id - H_XLOGY_SCALAR_SELF](ks, load<const at::Scalar &>()(op.args[0]), load<const at::Tensor &>()(op.args[1])));
  break;

case H_XLOGY_OUTSCALAR_SELF:
case H_POW_SCALAR_OUT:
case H_FLOAT_POWER_SCALAR_OUT:
case H_SPECIAL_XLOG1PY_SELF_SCALAR_OUT:
  init_update_in_place(op);
  redispatch_ptrs_57[op.id - H_XLOGY_OUTSCALAR_SELF](ks, load<const at::Scalar &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<at::Tensor &>()(op.args[2]));
  end_update_in_place(op);
  break;

case H_LOGSPACE:
  set(op, at::redispatch::logspace(ks, load<const at::Scalar &>()(op.args[0]), load<const at::Scalar &>()(op.args[1]), load<c10::optional<int64_t>>()(op.args[2]), load<double>()(op.args[3]), load<c10::optional<at::ScalarType>>()(op.args[4]), load<c10::optional<at::Layout>>()(op.args[5]), load<c10::optional<at::Device>>()(op.args[6]), load<c10::optional<bool>>()(op.args[7])));
  break;

case H_LOGSPACE_OUT:
  init_update_in_place(op);
  at::redispatch::logspace_outf(ks, load<const at::Scalar &>()(op.args[0]), load<const at::Scalar &>()(op.args[1]), load<c10::optional<int64_t>>()(op.args[2]), load<double>()(op.args[3]), load<at::Tensor &>()(op.args[4]));
  end_update_in_place(op);
  break;

case H__LOGCUMSUMEXP_OUT:
case H_LOGCUMSUMEXP_OUT:
case H_MATRIX_POWER_OUT:
case H_DIAG_OUT:
case H_TRIU_OUT:
case H_TRIL_OUT:
case H__CUMSUM_OUT:
case H__CUMPROD_OUT:
case H_GLU_OUT:
case H_LINALG_TENSORINV_OUT:
case H_LINALG_MATRIX_POWER_OUT:
  init_update_in_place(op);
  redispatch_ptrs_58[op.id - H__LOGCUMSUMEXP_OUT](ks, load<const at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1]), load<at::Tensor &>()(op.args[2]));
  end_update_in_place(op);
  break;

case H_LOGCUMSUMEXP_DIMNAME:
case H_SQUEEZE_DIMNAME:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, at::Dimname) = at::redispatch::logcumsumexp;
  if (op.id == H_SQUEEZE_DIMNAME) ptr = at::redispatch::squeeze;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<at::Dimname>()(op.args[1])));
  break;}

case H_LOGCUMSUMEXP_DIMNAME_OUT:
  init_update_in_place(op);
  at::redispatch::logcumsumexp_outf(ks, load<const at::Tensor &>()(op.args[0]), load<at::Dimname>()(op.args[1]), load<at::Tensor &>()(op.args[2]));
  end_update_in_place(op);
  break;

case H_LOGSUMEXP_OUT:
case H_AMAX_OUT:
case H_AMIN_OUT:
case H_FROBENIUS_NORM_OUT:
case H_NUCLEAR_NORM_DIM_OUT:
  init_update_in_place(op);
  redispatch_ptrs_59[op.id - H_LOGSUMEXP_OUT](ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<bool>()(op.args[2]), load<at::Tensor &>()(op.args[3]));
  end_update_in_place(op);
  break;

case H_LOGSUMEXP_NAMES:
  set(op, at::redispatch::logsumexp(ks, load<const at::Tensor &>()(op.args[0]), load<at::DimnameList>()(op.args[1]), load<bool>()(op.args[2])));
  break;

case H_LOGSUMEXP_NAMES_OUT:
  init_update_in_place(op);
  at::redispatch::logsumexp_outf(ks, load<const at::Tensor &>()(op.args[0]), load<at::DimnameList>()(op.args[1]), load<bool>()(op.args[2]), load<at::Tensor &>()(op.args[3]));
  end_update_in_place(op);
  break;

case H_VALUE_SELECTING_REDUCTION_BACKWARD:
  set(op, at::redispatch::value_selecting_reduction_backward(ks, load<const at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<at::IntArrayRef>()(op.args[3]), load<bool>()(op.args[4])));
  break;

case H_MAX_POOL1D:
case H_MAX_POOL2D:
case H_MKLDNN_MAX_POOL2D:
case H_MKLDNN_MAX_POOL3D:
case H_QUANTIZED_MAX_POOL1D:
case H_QUANTIZED_MAX_POOL2D:
case H_MAX_POOL3D:
  set(op, redispatch_ptrs_60[op.id - H_MAX_POOL1D](ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<at::IntArrayRef>()(op.args[3]), load<at::IntArrayRef>()(op.args[4]), load<bool>()(op.args[5])));
  break;

case H_MKLDNN_MAX_POOL2D_BACKWARD:
case H_MKLDNN_MAX_POOL3D_BACKWARD:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, bool) = at::redispatch::mkldnn_max_pool2d_backward;
  if (op.id == H_MKLDNN_MAX_POOL3D_BACKWARD) ptr = at::redispatch::mkldnn_max_pool3d_backward;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<at::IntArrayRef>()(op.args[3]), load<at::IntArrayRef>()(op.args[4]), load<at::IntArrayRef>()(op.args[5]), load<at::IntArrayRef>()(op.args[6]), load<bool>()(op.args[7])));
  break;}

case H_MEAN:
case H_SUM:
case H_NANSUM:
case H_PROD:
case H_TO_DENSE:
case H_TO_MKLDNN:
  set(op, redispatch_ptrs_61[op.id - H_MEAN](ks, load<const at::Tensor &>()(op.args[0]), load<c10::optional<at::ScalarType>>()(op.args[1])));
  break;

case H_MEAN_DIM:
case H_SUM_DIM_INTLIST:
case H_NANSUM_DIM_INTLIST:
  set(op, redispatch_ptrs_62[op.id - H_MEAN_DIM](ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<bool>()(op.args[2]), load<c10::optional<at::ScalarType>>()(op.args[3])));
  break;

case H_MEAN_OUT:
case H_SUM_INTLIST_OUT:
case H_NANSUM_INTLIST_OUT:
  init_update_in_place(op);
  redispatch_ptrs_63[op.id - H_MEAN_OUT](ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<bool>()(op.args[2]), load<c10::optional<at::ScalarType>>()(op.args[3]), load<at::Tensor &>()(op.args[4]));
  end_update_in_place(op);
  break;

case H_MEAN_NAMES_DIM:
case H_SUM_DIM_DIMNAMELIST:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, at::DimnameList, bool, c10::optional<at::ScalarType>) = at::redispatch::mean;
  if (op.id == H_SUM_DIM_DIMNAMELIST) ptr = at::redispatch::sum;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<at::DimnameList>()(op.args[1]), load<bool>()(op.args[2]), load<c10::optional<at::ScalarType>>()(op.args[3])));
  break;}

case H_MEAN_NAMES_OUT:
case H_SUM_DIMNAMELIST_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, at::DimnameList, bool, c10::optional<at::ScalarType>, at::Tensor &) = at::redispatch::mean_outf;
  if (op.id == H_SUM_DIMNAMELIST_OUT) ptr = at::redispatch::sum_outf;
  init_update_in_place(op);
  ptr(ks, load<const at::Tensor &>()(op.args[0]), load<at::DimnameList>()(op.args[1]), load<bool>()(op.args[2]), load<c10::optional<at::ScalarType>>()(op.args[3]), load<at::Tensor &>()(op.args[4]));
  end_update_in_place(op);
  break;}

case H_MKLDNN_CONVOLUTION_BACKWARD_INPUT:
  set(op, at::redispatch::mkldnn_convolution_backward_input(ks, load<at::IntArrayRef>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<at::IntArrayRef>()(op.args[3]), load<at::IntArrayRef>()(op.args[4]), load<at::IntArrayRef>()(op.args[5]), load<int64_t>()(op.args[6]), load<bool>()(op.args[7])));
  break;

case H_MIOPEN_CONVOLUTION_BACKWARD_INPUT:
case H_MIOPEN_CONVOLUTION_BACKWARD_WEIGHT:
case H_MIOPEN_CONVOLUTION_TRANSPOSE_BACKWARD_WEIGHT:
case H_MIOPEN_DEPTHWISE_CONVOLUTION_BACKWARD_INPUT:
case H_MIOPEN_DEPTHWISE_CONVOLUTION_BACKWARD_WEIGHT:
  set(op, redispatch_ptrs_64[op.id - H_MIOPEN_CONVOLUTION_BACKWARD_INPUT](ks, load<at::IntArrayRef>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<at::IntArrayRef>()(op.args[3]), load<at::IntArrayRef>()(op.args[4]), load<at::IntArrayRef>()(op.args[5]), load<int64_t>()(op.args[6]), load<bool>()(op.args[7]), load<bool>()(op.args[8])));
  break;

case H_NARROW_COPY_OUT:
  init_update_in_place(op);
  at::redispatch::narrow_copy_outf(ks, load<const at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1]), load<int64_t>()(op.args[2]), load<int64_t>()(op.args[3]), load<at::Tensor &>()(op.args[4]));
  end_update_in_place(op);
  break;

case H_NARROW_TENSOR:
  set(op, at::redispatch::narrow(ks, load<const at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<int64_t>()(op.args[3])));
  break;

case H_BATCH_NORM_ELEMT:
  set(op, at::redispatch::batch_norm_elemt(ks, load<const at::Tensor &>()(op.args[0]), load<const c10::optional<at::Tensor> &>()(op.args[1]), load<const c10::optional<at::Tensor> &>()(op.args[2]), load<const at::Tensor &>()(op.args[3]), load<const at::Tensor &>()(op.args[4]), load<double>()(op.args[5])));
  break;

case H_BATCH_NORM_ELEMT_OUT:
  init_update_in_place(op);
  at::redispatch::batch_norm_elemt_outf(ks, load<const at::Tensor &>()(op.args[0]), load<const c10::optional<at::Tensor> &>()(op.args[1]), load<const c10::optional<at::Tensor> &>()(op.args[2]), load<const at::Tensor &>()(op.args[3]), load<const at::Tensor &>()(op.args[4]), load<double>()(op.args[5]), load<at::Tensor &>()(op.args[6]));
  end_update_in_place(op);
  break;

case H_BATCH_NORM_BACKWARD_ELEMT:
  set(op, at::redispatch::batch_norm_backward_elemt(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<const at::Tensor &>()(op.args[3]), load<const c10::optional<at::Tensor> &>()(op.args[4]), load<const at::Tensor &>()(op.args[5]), load<const at::Tensor &>()(op.args[6]), load<const at::Tensor &>()(op.args[7])));
  break;

case H__NNPACK_SPATIAL_CONVOLUTION:
  set(op, at::redispatch::_nnpack_spatial_convolution(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const c10::optional<at::Tensor> &>()(op.args[2]), load<at::IntArrayRef>()(op.args[3]), load<at::IntArrayRef>()(op.args[4])));
  break;

case H__NNPACK_SPATIAL_CONVOLUTION_BACKWARD_INPUT:
case H_MAX_UNPOOL2D_BACKWARD:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, const at::Tensor &, at::IntArrayRef) = at::redispatch::_nnpack_spatial_convolution_backward_input;
  if (op.id == H_MAX_UNPOOL2D_BACKWARD) ptr = at::redispatch::max_unpool2d_backward;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<at::IntArrayRef>()(op.args[3])));
  break;}

case H__NNPACK_SPATIAL_CONVOLUTION_BACKWARD_WEIGHT:
  set(op, at::redispatch::_nnpack_spatial_convolution_backward_weight(ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<at::IntArrayRef>()(op.args[3])));
  break;

case H_ONES_NAMES:
case H_RAND_NAMES:
case H_RANDN_NAMES:
case H_ZEROS_NAMES:
  set(op, redispatch_ptrs_65[op.id - H_ONES_NAMES](ks, load<at::IntArrayRef>()(op.args[0]), load<c10::optional<at::DimnameList>>()(op.args[1]), load<c10::optional<at::ScalarType>>()(op.args[2]), load<c10::optional<at::Layout>>()(op.args[3]), load<c10::optional<at::Device>>()(op.args[4]), load<c10::optional<bool>>()(op.args[5])));
  break;

case H_ONES:
case H_RAND:
case H_RANDN:
case H_ZEROS:
case H_SPARSE_COO_TENSOR_SIZE:
  set(op, redispatch_ptrs_66[op.id - H_ONES](ks, load<at::IntArrayRef>()(op.args[0]), load<c10::optional<at::ScalarType>>()(op.args[1]), load<c10::optional<at::Layout>>()(op.args[2]), load<c10::optional<at::Device>>()(op.args[3]), load<c10::optional<bool>>()(op.args[4])));
  break;

case H_ONES_OUT:
case H_RAND_OUT:
case H_RANDN_OUT:
case H_ZEROS_OUT:
  init_update_in_place(op);
  redispatch_ptrs_67[op.id - H_ONES_OUT](ks, load<at::IntArrayRef>()(op.args[0]), load<at::Tensor &>()(op.args[1]));
  end_update_in_place(op);
  break;

case H_CDIST:
case H__CDIST_FORWARD:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, double, c10::optional<int64_t>) = at::redispatch::cdist;
  if (op.id == H__CDIST_FORWARD) ptr = at::redispatch::_cdist_forward;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<double>()(op.args[2]), load<c10::optional<int64_t>>()(op.args[3])));
  break;}

case H__CDIST_BACKWARD:
  set(op, at::redispatch::_cdist_backward(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<double>()(op.args[3]), load<const at::Tensor &>()(op.args[4])));
  break;

case H_PDIST:
case H__PDIST_FORWARD:
case H_PINVERSE:
  set(op, redispatch_ptrs_68[op.id - H_PDIST](ks, load<const at::Tensor &>()(op.args[0]), load<double>()(op.args[1])));
  break;

case H__PDIST_BACKWARD:
  set(op, at::redispatch::_pdist_backward(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<double>()(op.args[2]), load<const at::Tensor &>()(op.args[3])));
  break;

case H_COSINE_SIMILARITY:
case H_SMOOTH_L1_LOSS:
case H_HUBER_LOSS:
  set(op, redispatch_ptrs_69[op.id - H_COSINE_SIMILARITY](ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<int64_t>()(op.args[2]), load<double>()(op.args[3])));
  break;

case H_MOVEDIM_INTLIST:
case H_MOVEAXIS_INTLIST:
case H_ROLL:
  set(op, redispatch_ptrs_70[op.id - H_MOVEDIM_INTLIST](ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<at::IntArrayRef>()(op.args[2])));
  break;

case H_POISSON_NLL_LOSS:
  set(op, at::redispatch::poisson_nll_loss(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<bool>()(op.args[2]), load<bool>()(op.args[3]), load<double>()(op.args[4]), load<int64_t>()(op.args[5])));
  break;

case H_RAND_GENERATOR_WITH_NAMES:
case H_RANDN_GENERATOR_WITH_NAMES:
  {at::Tensor(*ptr)(DispatchKeySet, at::IntArrayRef, c10::optional<at::Generator>, c10::optional<at::DimnameList>, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<at::Device>, c10::optional<bool>) = at::redispatch::rand;
  if (op.id == H_RANDN_GENERATOR_WITH_NAMES) ptr = at::redispatch::randn;
  set(op, ptr(ks, load<at::IntArrayRef>()(op.args[0]), load<c10::optional<at::Generator>>()(op.args[1]), load<c10::optional<at::DimnameList>>()(op.args[2]), load<c10::optional<at::ScalarType>>()(op.args[3]), load<c10::optional<at::Layout>>()(op.args[4]), load<c10::optional<at::Device>>()(op.args[5]), load<c10::optional<bool>>()(op.args[6])));
  break;}

case H_RAND_GENERATOR:
case H_RANDN_GENERATOR:
  {at::Tensor(*ptr)(DispatchKeySet, at::IntArrayRef, c10::optional<at::Generator>, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<at::Device>, c10::optional<bool>) = at::redispatch::rand;
  if (op.id == H_RANDN_GENERATOR) ptr = at::redispatch::randn;
  set(op, ptr(ks, load<at::IntArrayRef>()(op.args[0]), load<c10::optional<at::Generator>>()(op.args[1]), load<c10::optional<at::ScalarType>>()(op.args[2]), load<c10::optional<at::Layout>>()(op.args[3]), load<c10::optional<at::Device>>()(op.args[4]), load<c10::optional<bool>>()(op.args[5])));
  break;}

case H_RAND_GENERATOR_OUT:
case H_RANDN_GENERATOR_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, at::IntArrayRef, c10::optional<at::Generator>, at::Tensor &) = at::redispatch::rand_outf;
  if (op.id == H_RANDN_GENERATOR_OUT) ptr = at::redispatch::randn_outf;
  init_update_in_place(op);
  ptr(ks, load<at::IntArrayRef>()(op.args[0]), load<c10::optional<at::Generator>>()(op.args[1]), load<at::Tensor &>()(op.args[2]));
  end_update_in_place(op);
  break;}

case H_RANDINT:
  set(op, at::redispatch::randint(ks, load<int64_t>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<c10::optional<at::ScalarType>>()(op.args[2]), load<c10::optional<at::Layout>>()(op.args[3]), load<c10::optional<at::Device>>()(op.args[4]), load<c10::optional<bool>>()(op.args[5])));
  break;

case H_RANDINT_GENERATOR:
  set(op, at::redispatch::randint(ks, load<int64_t>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<c10::optional<at::Generator>>()(op.args[2]), load<c10::optional<at::ScalarType>>()(op.args[3]), load<c10::optional<at::Layout>>()(op.args[4]), load<c10::optional<at::Device>>()(op.args[5]), load<c10::optional<bool>>()(op.args[6])));
  break;

case H_RANDINT_LOW:
case H__SPARSE_COO_TENSOR_WITH_DIMS:
  {at::Tensor(*ptr)(DispatchKeySet, int64_t, int64_t, at::IntArrayRef, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<at::Device>, c10::optional<bool>) = at::redispatch::randint;
  if (op.id == H__SPARSE_COO_TENSOR_WITH_DIMS) ptr = at::redispatch::_sparse_coo_tensor_with_dims;
  set(op, ptr(ks, load<int64_t>()(op.args[0]), load<int64_t>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<c10::optional<at::ScalarType>>()(op.args[3]), load<c10::optional<at::Layout>>()(op.args[4]), load<c10::optional<at::Device>>()(op.args[5]), load<c10::optional<bool>>()(op.args[6])));
  break;}

case H_RANDINT_LOW_GENERATOR:
  set(op, at::redispatch::randint(ks, load<int64_t>()(op.args[0]), load<int64_t>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<c10::optional<at::Generator>>()(op.args[3]), load<c10::optional<at::ScalarType>>()(op.args[4]), load<c10::optional<at::Layout>>()(op.args[5]), load<c10::optional<at::Device>>()(op.args[6]), load<c10::optional<bool>>()(op.args[7])));
  break;

case H_RANDINT_OUT:
  init_update_in_place(op);
  at::redispatch::randint_outf(ks, load<int64_t>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<at::Tensor &>()(op.args[2]));
  end_update_in_place(op);
  break;

case H_RANDINT_GENERATOR_OUT:
  init_update_in_place(op);
  at::redispatch::randint_outf(ks, load<int64_t>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<c10::optional<at::Generator>>()(op.args[2]), load<at::Tensor &>()(op.args[3]));
  end_update_in_place(op);
  break;

case H_RANDINT_LOW_OUT:
  init_update_in_place(op);
  at::redispatch::randint_outf(ks, load<int64_t>()(op.args[0]), load<int64_t>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<at::Tensor &>()(op.args[3]));
  end_update_in_place(op);
  break;

case H_RANDINT_LOW_GENERATOR_OUT:
  init_update_in_place(op);
  at::redispatch::randint_outf(ks, load<int64_t>()(op.args[0]), load<int64_t>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<c10::optional<at::Generator>>()(op.args[3]), load<at::Tensor &>()(op.args[4]));
  end_update_in_place(op);
  break;

case H_RANDINT_LIKE:
  set(op, at::redispatch::randint_like(ks, load<const at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1]), load<c10::optional<at::ScalarType>>()(op.args[2]), load<c10::optional<at::Layout>>()(op.args[3]), load<c10::optional<at::Device>>()(op.args[4]), load<c10::optional<bool>>()(op.args[5]), load<c10::optional<at::MemoryFormat>>()(op.args[6])));
  break;

case H_RANDINT_LIKE_LOW_DTYPE:
  set(op, at::redispatch::randint_like(ks, load<const at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1]), load<int64_t>()(op.args[2]), load<c10::optional<at::ScalarType>>()(op.args[3]), load<c10::optional<at::Layout>>()(op.args[4]), load<c10::optional<at::Device>>()(op.args[5]), load<c10::optional<bool>>()(op.args[6]), load<c10::optional<at::MemoryFormat>>()(op.args[7])));
  break;

case H_RANDPERM_GENERATOR:
  set(op, at::redispatch::randperm(ks, load<int64_t>()(op.args[0]), load<c10::optional<at::Generator>>()(op.args[1]), load<c10::optional<at::ScalarType>>()(op.args[2]), load<c10::optional<at::Layout>>()(op.args[3]), load<c10::optional<at::Device>>()(op.args[4]), load<c10::optional<bool>>()(op.args[5])));
  break;

case H_RANDPERM_GENERATOR_OUT:
  init_update_in_place(op);
  at::redispatch::randperm_outf(ks, load<int64_t>()(op.args[0]), load<c10::optional<at::Generator>>()(op.args[1]), load<at::Tensor &>()(op.args[2]));
  end_update_in_place(op);
  break;

case H_REPEAT_INTERLEAVE_SELF_TENSOR:
  set(op, at::redispatch::repeat_interleave(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<c10::optional<int64_t>>()(op.args[2]), load<c10::optional<int64_t>>()(op.args[3])));
  break;

case H_REPEAT_INTERLEAVE_SELF_INT:
  set(op, at::redispatch::repeat_interleave(ks, load<const at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1]), load<c10::optional<int64_t>>()(op.args[2]), load<c10::optional<int64_t>>()(op.args[3])));
  break;

case H_RRELU:
  set(op, at::redispatch::rrelu(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Scalar &>()(op.args[1]), load<const at::Scalar &>()(op.args[2]), load<bool>()(op.args[3]), load<c10::optional<at::Generator>>()(op.args[4])));
  break;

case H_RRELU_:
  init_update_in_place(op);
  at::redispatch::rrelu_(ks, load<at::Tensor &>()(op.args[0]), load<const at::Scalar &>()(op.args[1]), load<const at::Scalar &>()(op.args[2]), load<bool>()(op.args[3]), load<c10::optional<at::Generator>>()(op.args[4]));
  end_update_in_place(op);
  break;

case H_SELECT_DIMNAME:
  set(op, at::redispatch::select(ks, load<const at::Tensor &>()(op.args[0]), load<at::Dimname>()(op.args[1]), load<int64_t>()(op.args[2])));
  break;

case H_LOGIT:
case H_SPECIAL_LOGIT:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, c10::optional<double>) = at::redispatch::logit;
  if (op.id == H_SPECIAL_LOGIT) ptr = at::redispatch::special_logit;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<c10::optional<double>>()(op.args[1])));
  break;}

case H_LOGIT_:
  init_update_in_place(op);
  at::redispatch::logit_(ks, load<at::Tensor &>()(op.args[0]), load<c10::optional<double>>()(op.args[1]));
  end_update_in_place(op);
  break;

case H_LOGIT_OUT:
case H_SPECIAL_LOGIT_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, c10::optional<double>, at::Tensor &) = at::redispatch::logit_outf;
  if (op.id == H_SPECIAL_LOGIT_OUT) ptr = at::redispatch::special_logit_outf;
  init_update_in_place(op);
  ptr(ks, load<const at::Tensor &>()(op.args[0]), load<c10::optional<double>>()(op.args[1]), load<at::Tensor &>()(op.args[2]));
  end_update_in_place(op);
  break;}

case H_SLICE_TENSOR:
  set(op, at::redispatch::slice(ks, load<const at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1]), load<c10::optional<int64_t>>()(op.args[2]), load<c10::optional<int64_t>>()(op.args[3]), load<int64_t>()(op.args[4])));
  break;

case H_SLICE_BACKWARD:
  set(op, at::redispatch::slice_backward(ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<int64_t>()(op.args[2]), load<int64_t>()(op.args[3]), load<int64_t>()(op.args[4]), load<int64_t>()(op.args[5])));
  break;

case H_SQUEEZE__DIMNAME:
  init_update_in_place(op);
  at::redispatch::squeeze_(ks, load<at::Tensor &>()(op.args[0]), load<at::Dimname>()(op.args[1]));
  end_update_in_place(op);
  break;

case H_STFT:
  set(op, at::redispatch::stft(ks, load<const at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1]), load<c10::optional<int64_t>>()(op.args[2]), load<c10::optional<int64_t>>()(op.args[3]), load<const c10::optional<at::Tensor> &>()(op.args[4]), load<bool>()(op.args[5]), load<c10::optional<bool>>()(op.args[6]), load<c10::optional<bool>>()(op.args[7])));
  break;

case H_ISTFT:
  set(op, at::redispatch::istft(ks, load<const at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1]), load<c10::optional<int64_t>>()(op.args[2]), load<c10::optional<int64_t>>()(op.args[3]), load<const c10::optional<at::Tensor> &>()(op.args[4]), load<bool>()(op.args[5]), load<bool>()(op.args[6]), load<c10::optional<bool>>()(op.args[7]), load<c10::optional<int64_t>>()(op.args[8]), load<bool>()(op.args[9])));
  break;

case H_STD_DIM:
case H_VAR_DIM:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, at::IntArrayRef, bool, bool) = at::redispatch::std;
  if (op.id == H_VAR_DIM) ptr = at::redispatch::var;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<bool>()(op.args[2]), load<bool>()(op.args[3])));
  break;}

case H_STD_CORRECTION:
case H_VAR_CORRECTION:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, c10::optional<at::IntArrayRef>, c10::optional<int64_t>, bool) = at::redispatch::std;
  if (op.id == H_VAR_CORRECTION) ptr = at::redispatch::var;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<c10::optional<at::IntArrayRef>>()(op.args[1]), load<c10::optional<int64_t>>()(op.args[2]), load<bool>()(op.args[3])));
  break;}

case H_STD_OUT:
case H_VAR_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, at::IntArrayRef, bool, bool, at::Tensor &) = at::redispatch::std_outf;
  if (op.id == H_VAR_OUT) ptr = at::redispatch::var_outf;
  init_update_in_place(op);
  ptr(ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<bool>()(op.args[2]), load<bool>()(op.args[3]), load<at::Tensor &>()(op.args[4]));
  end_update_in_place(op);
  break;}

case H_STD_CORRECTION_OUT:
case H_VAR_CORRECTION_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, c10::optional<at::IntArrayRef>, c10::optional<int64_t>, bool, at::Tensor &) = at::redispatch::std_outf;
  if (op.id == H_VAR_CORRECTION_OUT) ptr = at::redispatch::var_outf;
  init_update_in_place(op);
  ptr(ks, load<const at::Tensor &>()(op.args[0]), load<c10::optional<at::IntArrayRef>>()(op.args[1]), load<c10::optional<int64_t>>()(op.args[2]), load<bool>()(op.args[3]), load<at::Tensor &>()(op.args[4]));
  end_update_in_place(op);
  break;}

case H_STD_NAMES_DIM:
case H_VAR_NAMES_DIM:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, at::DimnameList, bool, bool) = at::redispatch::std;
  if (op.id == H_VAR_NAMES_DIM) ptr = at::redispatch::var;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<at::DimnameList>()(op.args[1]), load<bool>()(op.args[2]), load<bool>()(op.args[3])));
  break;}

case H_STD_NAMES_OUT:
case H_VAR_NAMES_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, at::DimnameList, bool, bool, at::Tensor &) = at::redispatch::std_outf;
  if (op.id == H_VAR_NAMES_OUT) ptr = at::redispatch::var_outf;
  init_update_in_place(op);
  ptr(ks, load<const at::Tensor &>()(op.args[0]), load<at::DimnameList>()(op.args[1]), load<bool>()(op.args[2]), load<bool>()(op.args[3]), load<at::Tensor &>()(op.args[4]));
  end_update_in_place(op);
  break;}

case H_STD_CORRECTION_NAMES:
case H_VAR_CORRECTION_NAMES:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, at::DimnameList, c10::optional<int64_t>, bool) = at::redispatch::std;
  if (op.id == H_VAR_CORRECTION_NAMES) ptr = at::redispatch::var;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<at::DimnameList>()(op.args[1]), load<c10::optional<int64_t>>()(op.args[2]), load<bool>()(op.args[3])));
  break;}

case H_STD_CORRECTION_NAMES_OUT:
case H_VAR_CORRECTION_NAMES_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, at::DimnameList, c10::optional<int64_t>, bool, at::Tensor &) = at::redispatch::std_outf;
  if (op.id == H_VAR_CORRECTION_NAMES_OUT) ptr = at::redispatch::var_outf;
  init_update_in_place(op);
  ptr(ks, load<const at::Tensor &>()(op.args[0]), load<at::DimnameList>()(op.args[1]), load<c10::optional<int64_t>>()(op.args[2]), load<bool>()(op.args[3]), load<at::Tensor &>()(op.args[4]));
  end_update_in_place(op);
  break;}

case H_PROD_DIM_INT:
  set(op, at::redispatch::prod(ks, load<const at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1]), load<bool>()(op.args[2]), load<c10::optional<at::ScalarType>>()(op.args[3])));
  break;

case H_PROD_INT_OUT:
  init_update_in_place(op);
  at::redispatch::prod_outf(ks, load<const at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1]), load<bool>()(op.args[2]), load<c10::optional<at::ScalarType>>()(op.args[3]), load<at::Tensor &>()(op.args[4]));
  end_update_in_place(op);
  break;

case H_PROD_DIM_DIMNAME:
  set(op, at::redispatch::prod(ks, load<const at::Tensor &>()(op.args[0]), load<at::Dimname>()(op.args[1]), load<bool>()(op.args[2]), load<c10::optional<at::ScalarType>>()(op.args[3])));
  break;

case H_PROD_DIMNAME_OUT:
  init_update_in_place(op);
  at::redispatch::prod_outf(ks, load<const at::Tensor &>()(op.args[0]), load<at::Dimname>()(op.args[1]), load<bool>()(op.args[2]), load<c10::optional<at::ScalarType>>()(op.args[3]), load<at::Tensor &>()(op.args[4]));
  end_update_in_place(op);
  break;

case H_TENSORDOT:
  set(op, at::redispatch::tensordot(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<at::IntArrayRef>()(op.args[3])));
  break;

case H_TENSORDOT_OUT:
  init_update_in_place(op);
  at::redispatch::tensordot_outf(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<at::IntArrayRef>()(op.args[3]), load<at::Tensor &>()(op.args[4]));
  end_update_in_place(op);
  break;

case H_THRESHOLD_OUT:
case H_HARDTANH_OUT:
case H_SOFTPLUS_OUT:
  init_update_in_place(op);
  redispatch_ptrs_71[op.id - H_THRESHOLD_OUT](ks, load<const at::Tensor &>()(op.args[0]), load<const at::Scalar &>()(op.args[1]), load<const at::Scalar &>()(op.args[2]), load<at::Tensor &>()(op.args[3]));
  end_update_in_place(op);
  break;

case H_TRANSPOSE_DIMNAME:
  set(op, at::redispatch::transpose(ks, load<const at::Tensor &>()(op.args[0]), load<at::Dimname>()(op.args[1]), load<at::Dimname>()(op.args[2])));
  break;

case H_TRANSPOSE_:
case H__MKLDNN_TRANSPOSE_:
case H_SWAPAXES_:
case H_SWAPDIMS_:
  init_update_in_place(op);
  redispatch_ptrs_72[op.id - H_TRANSPOSE_](ks, load<at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1]), load<int64_t>()(op.args[2]));
  end_update_in_place(op);
  break;

case H_ROT90:
  set(op, at::redispatch::rot90(ks, load<const at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1]), load<at::IntArrayRef>()(op.args[2])));
  break;

case H_TRAPZ_DX:
case H__MAKE_PER_TENSOR_QUANTIZED_TENSOR:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, double, int64_t) = at::redispatch::trapz;
  if (op.id == H__MAKE_PER_TENSOR_QUANTIZED_TENSOR) ptr = at::redispatch::_make_per_tensor_quantized_tensor;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<double>()(op.args[1]), load<int64_t>()(op.args[2])));
  break;}

case H__TRILINEAR:
  set(op, at::redispatch::_trilinear(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<at::IntArrayRef>()(op.args[3]), load<at::IntArrayRef>()(op.args[4]), load<at::IntArrayRef>()(op.args[5]), load<at::IntArrayRef>()(op.args[6]), load<int64_t>()(op.args[7])));
  break;

case H_TRIPLET_MARGIN_LOSS:
  set(op, at::redispatch::triplet_margin_loss(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<double>()(op.args[3]), load<double>()(op.args[4]), load<double>()(op.args[5]), load<bool>()(op.args[6]), load<int64_t>()(op.args[7])));
  break;

case H_WHERE_SCALARSELF:
  set(op, at::redispatch::where(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Scalar &>()(op.args[1]), load<const at::Tensor &>()(op.args[2])));
  break;

case H_BINOMIAL:
case H_NORMAL_TENSOR_TENSOR:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, c10::optional<at::Generator>) = at::redispatch::binomial;
  if (op.id == H_NORMAL_TENSOR_TENSOR) ptr = at::redispatch::normal;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<c10::optional<at::Generator>>()(op.args[2])));
  break;}

case H_NATIVE_NORM_SCALAROPT_DIM_DTYPE:
  set(op, at::redispatch::native_norm(ks, load<const at::Tensor &>()(op.args[0]), load<const c10::optional<at::Scalar> &>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<bool>()(op.args[3]), load<c10::optional<at::ScalarType>>()(op.args[4])));
  break;

case H__SPARSE_SUM_DTYPE:
case H_VIEW_DTYPE:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, at::ScalarType) = at::redispatch::_sparse_sum;
  if (op.id == H_VIEW_DTYPE) ptr = at::redispatch::view;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<at::ScalarType>()(op.args[1])));
  break;}

case H__SPARSE_SUM_DIM_DTYPE:
  set(op, at::redispatch::_sparse_sum(ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<at::ScalarType>()(op.args[2])));
  break;

case H__SPARSE_SUM_BACKWARD:
case H_MAX_UNPOOL2D:
case H_REFLECTION_PAD2D_BACKWARD:
case H_REPLICATION_PAD2D_BACKWARD:
case H_REPLICATION_PAD3D_BACKWARD:
  set(op, redispatch_ptrs_73[op.id - H__SPARSE_SUM_BACKWARD](ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<at::IntArrayRef>()(op.args[2])));
  break;

case H_NORM_SCALAROPT_DTYPE:
  set(op, at::redispatch::norm(ks, load<const at::Tensor &>()(op.args[0]), load<const c10::optional<at::Scalar> &>()(op.args[1]), load<at::ScalarType>()(op.args[2])));
  break;

case H_NORM_SCALAROPT_DIM_DTYPE:
  set(op, at::redispatch::norm(ks, load<const at::Tensor &>()(op.args[0]), load<const c10::optional<at::Scalar> &>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<bool>()(op.args[3]), load<at::ScalarType>()(op.args[4])));
  break;

case H_NORM_SCALAROPT_DIM:
  set(op, at::redispatch::norm(ks, load<const at::Tensor &>()(op.args[0]), load<const c10::optional<at::Scalar> &>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<bool>()(op.args[3])));
  break;

case H_NORM_DTYPE_OUT:
  init_update_in_place(op);
  at::redispatch::norm_outf(ks, load<const at::Tensor &>()(op.args[0]), load<const c10::optional<at::Scalar> &>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<bool>()(op.args[3]), load<at::ScalarType>()(op.args[4]), load<at::Tensor &>()(op.args[5]));
  end_update_in_place(op);
  break;

case H_NORM_OUT:
  init_update_in_place(op);
  at::redispatch::norm_outf(ks, load<const at::Tensor &>()(op.args[0]), load<const c10::optional<at::Scalar> &>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<bool>()(op.args[3]), load<at::Tensor &>()(op.args[4]));
  end_update_in_place(op);
  break;

case H_NORM_NAMES_SCALAROPT_DIM_DTYPE:
  set(op, at::redispatch::norm(ks, load<const at::Tensor &>()(op.args[0]), load<const c10::optional<at::Scalar> &>()(op.args[1]), load<at::DimnameList>()(op.args[2]), load<bool>()(op.args[3]), load<at::ScalarType>()(op.args[4])));
  break;

case H_NORM_NAMES_SCALAROPT_DIM:
  set(op, at::redispatch::norm(ks, load<const at::Tensor &>()(op.args[0]), load<const c10::optional<at::Scalar> &>()(op.args[1]), load<at::DimnameList>()(op.args[2]), load<bool>()(op.args[3])));
  break;

case H_NORM_NAMES_DTYPE_OUT:
  init_update_in_place(op);
  at::redispatch::norm_outf(ks, load<const at::Tensor &>()(op.args[0]), load<const c10::optional<at::Scalar> &>()(op.args[1]), load<at::DimnameList>()(op.args[2]), load<bool>()(op.args[3]), load<at::ScalarType>()(op.args[4]), load<at::Tensor &>()(op.args[5]));
  end_update_in_place(op);
  break;

case H_NORM_NAMES_OUT:
  init_update_in_place(op);
  at::redispatch::norm_outf(ks, load<const at::Tensor &>()(op.args[0]), load<const c10::optional<at::Scalar> &>()(op.args[1]), load<at::DimnameList>()(op.args[2]), load<bool>()(op.args[3]), load<at::Tensor &>()(op.args[4]));
  end_update_in_place(op);
  break;

case H_NUCLEAR_NORM_OUT:
case H_CHOLESKY_OUT:
case H_CHOLESKY_INVERSE_OUT:
  init_update_in_place(op);
  redispatch_ptrs_74[op.id - H_NUCLEAR_NORM_OUT](ks, load<const at::Tensor &>()(op.args[0]), load<bool>()(op.args[1]), load<at::Tensor &>()(op.args[2]));
  end_update_in_place(op);
  break;

case H_CLONE:
  set(op, at::redispatch::clone(ks, load<const at::Tensor &>()(op.args[0]), load<c10::optional<at::MemoryFormat>>()(op.args[1])));
  break;

case H_RESIZE_AS_:
  init_update_in_place(op);
  at::redispatch::resize_as_(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<c10::optional<at::MemoryFormat>>()(op.args[2]));
  end_update_in_place(op);
  break;

case H_RESIZE_AS_SPARSE_:
  init_update_in_place(op);
  at::redispatch::resize_as_sparse_(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]));
  end_update_in_place(op);
  break;

case H_SPARSE_CSR_TENSOR_CROW_COL_VALUE_SIZE:
case H__SPARSE_CSR_TENSOR_UNSAFE:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, const at::Tensor &, at::IntArrayRef, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<at::Device>, c10::optional<bool>) = at::redispatch::sparse_csr_tensor;
  if (op.id == H__SPARSE_CSR_TENSOR_UNSAFE) ptr = at::redispatch::_sparse_csr_tensor_unsafe;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<at::IntArrayRef>()(op.args[3]), load<c10::optional<at::ScalarType>>()(op.args[4]), load<c10::optional<at::Layout>>()(op.args[5]), load<c10::optional<at::Device>>()(op.args[6]), load<c10::optional<bool>>()(op.args[7])));
  break;}

case H_SPARSE_CSR_TENSOR_CROW_COL_VALUE:
  set(op, at::redispatch::sparse_csr_tensor(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<c10::optional<at::ScalarType>>()(op.args[3]), load<c10::optional<at::Layout>>()(op.args[4]), load<c10::optional<at::Device>>()(op.args[5]), load<c10::optional<bool>>()(op.args[6])));
  break;

case H_SPARSE_COO_TENSOR_INDICES:
  set(op, at::redispatch::sparse_coo_tensor(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<c10::optional<at::ScalarType>>()(op.args[2]), load<c10::optional<at::Layout>>()(op.args[3]), load<c10::optional<at::Device>>()(op.args[4]), load<c10::optional<bool>>()(op.args[5])));
  break;

case H_SPARSE_COO_TENSOR_INDICES_SIZE:
case H__SPARSE_COO_TENSOR_UNSAFE:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, at::IntArrayRef, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<at::Device>, c10::optional<bool>) = at::redispatch::sparse_coo_tensor;
  if (op.id == H__SPARSE_COO_TENSOR_UNSAFE) ptr = at::redispatch::_sparse_coo_tensor_unsafe;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<c10::optional<at::ScalarType>>()(op.args[3]), load<c10::optional<at::Layout>>()(op.args[4]), load<c10::optional<at::Device>>()(op.args[5]), load<c10::optional<bool>>()(op.args[6])));
  break;}

case H__SPARSE_COO_TENSOR_WITH_DIMS_AND_TENSORS:
  set(op, at::redispatch::_sparse_coo_tensor_with_dims_and_tensors(ks, load<int64_t>()(op.args[0]), load<int64_t>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<const at::Tensor &>()(op.args[3]), load<const at::Tensor &>()(op.args[4]), load<c10::optional<at::ScalarType>>()(op.args[5]), load<c10::optional<at::Layout>>()(op.args[6]), load<c10::optional<at::Device>>()(op.args[7]), load<c10::optional<bool>>()(op.args[8])));
  break;

case H_SPARSE_RESIZE_:
case H_SPARSE_RESIZE_AND_CLEAR_:
  {const at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, at::IntArrayRef, int64_t, int64_t) = at::redispatch::sparse_resize_;
  if (op.id == H_SPARSE_RESIZE_AND_CLEAR_) ptr = at::redispatch::sparse_resize_and_clear_;
  init_update_in_place(op);
  ptr(ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<int64_t>()(op.args[2]), load<int64_t>()(op.args[3]));
  end_update_in_place(op);
  break;}

case H_MKLDNN_REORDER_CONV2D_WEIGHT:
case H_MKLDNN_REORDER_CONV3D_WEIGHT:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t) = at::redispatch::mkldnn_reorder_conv2d_weight;
  if (op.id == H_MKLDNN_REORDER_CONV3D_WEIGHT) ptr = at::redispatch::mkldnn_reorder_conv3d_weight;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<at::IntArrayRef>()(op.args[3]), load<int64_t>()(op.args[4])));
  break;}

case H_QUANTIZE_PER_TENSOR:
  set(op, at::redispatch::quantize_per_tensor(ks, load<const at::Tensor &>()(op.args[0]), load<double>()(op.args[1]), load<int64_t>()(op.args[2]), load<at::ScalarType>()(op.args[3])));
  break;

case H_QUANTIZE_PER_CHANNEL:
  set(op, at::redispatch::quantize_per_channel(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<int64_t>()(op.args[3]), load<at::ScalarType>()(op.args[4])));
  break;

case H_FAKE_QUANTIZE_PER_TENSOR_AFFINE:
  set(op, at::redispatch::fake_quantize_per_tensor_affine(ks, load<const at::Tensor &>()(op.args[0]), load<double>()(op.args[1]), load<int64_t>()(op.args[2]), load<int64_t>()(op.args[3]), load<int64_t>()(op.args[4])));
  break;

case H__FAKE_QUANTIZE_LEARNABLE_PER_TENSOR_AFFINE:
  set(op, at::redispatch::_fake_quantize_learnable_per_tensor_affine(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<int64_t>()(op.args[3]), load<int64_t>()(op.args[4]), load<double>()(op.args[5])));
  break;

case H_FAKE_QUANTIZE_PER_CHANNEL_AFFINE:
  set(op, at::redispatch::fake_quantize_per_channel_affine(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<int64_t>()(op.args[3]), load<int64_t>()(op.args[4]), load<int64_t>()(op.args[5])));
  break;

case H__FAKE_QUANTIZE_LEARNABLE_PER_CHANNEL_AFFINE:
  set(op, at::redispatch::_fake_quantize_learnable_per_channel_affine(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<int64_t>()(op.args[3]), load<int64_t>()(op.args[4]), load<int64_t>()(op.args[5]), load<double>()(op.args[6])));
  break;

case H_TO_DTYPE_LAYOUT:
  set(op, at::redispatch::to(ks, load<const at::Tensor &>()(op.args[0]), load<c10::optional<at::ScalarType>>()(op.args[1]), load<c10::optional<at::Layout>>()(op.args[2]), load<c10::optional<at::Device>>()(op.args[3]), load<c10::optional<bool>>()(op.args[4]), load<bool>()(op.args[5]), load<bool>()(op.args[6]), load<c10::optional<at::MemoryFormat>>()(op.args[7])));
  break;

case H_TO_DEVICE:
  set(op, at::redispatch::to(ks, load<const at::Tensor &>()(op.args[0]), load<at::Device>()(op.args[1]), load<at::ScalarType>()(op.args[2]), load<bool>()(op.args[3]), load<bool>()(op.args[4]), load<c10::optional<at::MemoryFormat>>()(op.args[5])));
  break;

case H_TO_DTYPE:
  set(op, at::redispatch::to(ks, load<const at::Tensor &>()(op.args[0]), load<at::ScalarType>()(op.args[1]), load<bool>()(op.args[2]), load<bool>()(op.args[3]), load<c10::optional<at::MemoryFormat>>()(op.args[4])));
  break;

case H_TO_OTHER:
  set(op, at::redispatch::to(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<bool>()(op.args[2]), load<bool>()(op.args[3]), load<c10::optional<at::MemoryFormat>>()(op.args[4])));
  break;

case H_GRU_CELL:
case H_RNN_TANH_CELL:
case H_RNN_RELU_CELL:
  set(op, redispatch_ptrs_75[op.id - H_GRU_CELL](ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<const at::Tensor &>()(op.args[3]), load<const c10::optional<at::Tensor> &>()(op.args[4]), load<const c10::optional<at::Tensor> &>()(op.args[5])));
  break;

case H_QUANTIZED_GRU_CELL:
case H_QUANTIZED_RNN_RELU_CELL:
case H_QUANTIZED_RNN_TANH_CELL:
  set(op, redispatch_ptrs_76[op.id - H_QUANTIZED_GRU_CELL](ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<const at::Tensor &>()(op.args[3]), load<const at::Tensor &>()(op.args[4]), load<const at::Tensor &>()(op.args[5]), load<const at::Tensor &>()(op.args[6]), load<const at::Tensor &>()(op.args[7]), load<const at::Tensor &>()(op.args[8]), load<const at::Tensor &>()(op.args[9]), load<const at::Scalar &>()(op.args[10]), load<const at::Scalar &>()(op.args[11]), load<const at::Scalar &>()(op.args[12]), load<const at::Scalar &>()(op.args[13])));
  break;

case H__PACK_PADDED_SEQUENCE_BACKWARD:
  set(op, at::redispatch::_pack_padded_sequence_backward(ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<bool>()(op.args[3])));
  break;

case H_SET__SOURCE_STORAGE:
  init_update_in_place(op);
  at::redispatch::set_(ks, load<at::Tensor &>()(op.args[0]), load<at::Storage>()(op.args[1]));
  end_update_in_place(op);
  break;

case H_SET__SOURCE_STORAGE_STORAGE_OFFSET:
  init_update_in_place(op);
  at::redispatch::set_(ks, load<at::Tensor &>()(op.args[0]), load<at::Storage>()(op.args[1]), load<int64_t>()(op.args[2]), load<at::IntArrayRef>()(op.args[3]), load<at::IntArrayRef>()(op.args[4]));
  end_update_in_place(op);
  break;

case H_MASKED_FILL__TENSOR:
case H_MASKED_SCATTER_:
case H_LERP__TENSOR:
  init_update_in_place(op);
  redispatch_ptrs_77[op.id - H_MASKED_FILL__TENSOR](ks, load<at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]));
  end_update_in_place(op);
  break;

case H_PUT_:
  init_update_in_place(op);
  at::redispatch::put_(ks, load<at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<bool>()(op.args[3]));
  end_update_in_place(op);
  break;

case H_PUT:
  set(op, at::redispatch::put(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<bool>()(op.args[3])));
  break;

case H_INDEX_ADD__ALPHA:
  init_update_in_place(op);
  at::redispatch::index_add_(ks, load<at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<const at::Tensor &>()(op.args[3]), load<const at::Scalar &>()(op.args[4]));
  end_update_in_place(op);
  break;

case H_INDEX_ADD_ALPHA:
  set(op, at::redispatch::index_add(ks, load<const at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<const at::Tensor &>()(op.args[3]), load<const at::Scalar &>()(op.args[4])));
  break;

case H_INDEX_ADD_DIMNAME:
  set(op, at::redispatch::index_add(ks, load<const at::Tensor &>()(op.args[0]), load<at::Dimname>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<const at::Tensor &>()(op.args[3]), load<const at::Scalar &>()(op.args[4])));
  break;

case H_INDEX_FILL__INT_SCALAR:
  init_update_in_place(op);
  at::redispatch::index_fill_(ks, load<at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<const at::Scalar &>()(op.args[3]));
  end_update_in_place(op);
  break;

case H_INDEX_FILL_INT_SCALAR:
  set(op, at::redispatch::index_fill(ks, load<const at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<const at::Scalar &>()(op.args[3])));
  break;

case H_INDEX_FILL__DIMNAME_SCALAR:
  init_update_in_place(op);
  at::redispatch::index_fill_(ks, load<at::Tensor &>()(op.args[0]), load<at::Dimname>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<const at::Scalar &>()(op.args[3]));
  end_update_in_place(op);
  break;

case H_INDEX_FILL_DIMNAME_SCALAR:
case H_SCATTER_DIMNAME_VALUE:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, at::Dimname, const at::Tensor &, const at::Scalar &) = at::redispatch::index_fill;
  if (op.id == H_SCATTER_DIMNAME_VALUE) ptr = at::redispatch::scatter;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<at::Dimname>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<const at::Scalar &>()(op.args[3])));
  break;}

case H_SCATTER_SRC_OUT:
case H_SCATTER_ADD_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, int64_t, const at::Tensor &, const at::Tensor &, at::Tensor &) = at::redispatch::scatter_outf;
  if (op.id == H_SCATTER_ADD_OUT) ptr = at::redispatch::scatter_add_outf;
  init_update_in_place(op);
  ptr(ks, load<const at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<const at::Tensor &>()(op.args[3]), load<at::Tensor &>()(op.args[4]));
  end_update_in_place(op);
  break;}

case H_SCATTER_VALUE_OUT:
  init_update_in_place(op);
  at::redispatch::scatter_outf(ks, load<const at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<const at::Scalar &>()(op.args[3]), load<at::Tensor &>()(op.args[4]));
  end_update_in_place(op);
  break;

case H_SCATTER_REDUCE_OUT:
  init_update_in_place(op);
  at::redispatch::scatter_outf(ks, load<const at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<const at::Tensor &>()(op.args[3]), load<c10::string_view>()(op.args[4]), load<at::Tensor &>()(op.args[5]));
  end_update_in_place(op);
  break;

case H_SCATTER_VALUE_REDUCE_OUT:
  init_update_in_place(op);
  at::redispatch::scatter_outf(ks, load<const at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<const at::Scalar &>()(op.args[3]), load<c10::string_view>()(op.args[4]), load<at::Tensor &>()(op.args[5]));
  end_update_in_place(op);
  break;

case H_ADDCDIV_:
case H_ADDCMUL_:
  {at::Tensor &(*ptr)(DispatchKeySet, at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Scalar &) = at::redispatch::addcdiv_;
  if (op.id == H_ADDCMUL_) ptr = at::redispatch::addcmul_;
  init_update_in_place(op);
  ptr(ks, load<at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<const at::Scalar &>()(op.args[3]));
  end_update_in_place(op);
  break;}

case H_RANDOM__FROM:
  init_update_in_place(op);
  at::redispatch::random_(ks, load<at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1]), load<c10::optional<int64_t>>()(op.args[2]), load<c10::optional<at::Generator>>()(op.args[3]));
  end_update_in_place(op);
  break;

case H_RANDOM__TO:
  init_update_in_place(op);
  at::redispatch::random_(ks, load<at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1]), load<c10::optional<at::Generator>>()(op.args[2]));
  end_update_in_place(op);
  break;

case H_RANDOM_:
  init_update_in_place(op);
  at::redispatch::random_(ks, load<at::Tensor &>()(op.args[0]), load<c10::optional<at::Generator>>()(op.args[1]));
  end_update_in_place(op);
  break;

case H_UNIFORM_:
case H_CAUCHY_:
case H_LOG_NORMAL_:
case H_NORMAL_:
  init_update_in_place(op);
  redispatch_ptrs_78[op.id - H_UNIFORM_](ks, load<at::Tensor &>()(op.args[0]), load<double>()(op.args[1]), load<double>()(op.args[2]), load<c10::optional<at::Generator>>()(op.args[3]));
  end_update_in_place(op);
  break;

case H_DIAG_BACKWARD:
  set(op, at::redispatch::diag_backward(ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<int64_t>()(op.args[2])));
  break;

case H_CROSS_OUT:
case H_TAKE_ALONG_DIM_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, c10::optional<int64_t>, at::Tensor &) = at::redispatch::cross_outf;
  if (op.id == H_TAKE_ALONG_DIM_OUT) ptr = at::redispatch::take_along_dim_outf;
  init_update_in_place(op);
  ptr(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<c10::optional<int64_t>>()(op.args[2]), load<at::Tensor &>()(op.args[3]));
  end_update_in_place(op);
  break;}

case H_CROSS:
case H_TAKE_ALONG_DIM:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, c10::optional<int64_t>) = at::redispatch::cross;
  if (op.id == H_TAKE_ALONG_DIM) ptr = at::redispatch::take_along_dim;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<c10::optional<int64_t>>()(op.args[2])));
  break;}

case H_TRIL_INDICES:
case H_TRIU_INDICES:
  {at::Tensor(*ptr)(DispatchKeySet, int64_t, int64_t, int64_t, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<at::Device>, c10::optional<bool>) = at::redispatch::tril_indices;
  if (op.id == H_TRIU_INDICES) ptr = at::redispatch::triu_indices;
  set(op, ptr(ks, load<int64_t>()(op.args[0]), load<int64_t>()(op.args[1]), load<int64_t>()(op.args[2]), load<c10::optional<at::ScalarType>>()(op.args[3]), load<c10::optional<at::Layout>>()(op.args[4]), load<c10::optional<at::Device>>()(op.args[5]), load<c10::optional<bool>>()(op.args[6])));
  break;}

case H_INDEX_SELECT_OUT:
  init_update_in_place(op);
  at::redispatch::index_select_outf(ks, load<const at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<at::Tensor &>()(op.args[3]));
  end_update_in_place(op);
  break;

case H_INDEX_SELECT:
  set(op, at::redispatch::index_select(ks, load<const at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1]), load<const at::Tensor &>()(op.args[2])));
  break;

case H_INDEX_SELECT_DIMNAME_OUT:
  init_update_in_place(op);
  at::redispatch::index_select_outf(ks, load<const at::Tensor &>()(op.args[0]), load<at::Dimname>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<at::Tensor &>()(op.args[3]));
  end_update_in_place(op);
  break;

case H_INDEX_SELECT_DIMNAME:
  set(op, at::redispatch::index_select(ks, load<const at::Tensor &>()(op.args[0]), load<at::Dimname>()(op.args[1]), load<const at::Tensor &>()(op.args[2])));
  break;

case H_INDEX_SELECT_BACKWARD:
  set(op, at::redispatch::index_select_backward(ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<int64_t>()(op.args[2]), load<const at::Tensor &>()(op.args[3])));
  break;

case H_GATHER_OUT:
  init_update_in_place(op);
  at::redispatch::gather_outf(ks, load<const at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<bool>()(op.args[3]), load<at::Tensor &>()(op.args[4]));
  end_update_in_place(op);
  break;

case H_GATHER:
  set(op, at::redispatch::gather(ks, load<const at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<bool>()(op.args[3])));
  break;

case H_GATHER_BACKWARD:
  set(op, at::redispatch::gather_backward(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<int64_t>()(op.args[2]), load<const at::Tensor &>()(op.args[3]), load<bool>()(op.args[4])));
  break;

case H_GATHER_DIMNAME_OUT:
  init_update_in_place(op);
  at::redispatch::gather_outf(ks, load<const at::Tensor &>()(op.args[0]), load<at::Dimname>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<bool>()(op.args[3]), load<at::Tensor &>()(op.args[4]));
  end_update_in_place(op);
  break;

case H_GATHER_DIMNAME:
  set(op, at::redispatch::gather(ks, load<const at::Tensor &>()(op.args[0]), load<at::Dimname>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<bool>()(op.args[3])));
  break;

case H_ADDCMUL_OUT:
case H_ADDCDIV_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Scalar &, at::Tensor &) = at::redispatch::addcmul_outf;
  if (op.id == H_ADDCDIV_OUT) ptr = at::redispatch::addcdiv_outf;
  init_update_in_place(op);
  ptr(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<const at::Scalar &>()(op.args[3]), load<at::Tensor &>()(op.args[4]));
  end_update_in_place(op);
  break;}

case H_ADDCMUL:
case H_ADDCDIV:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Scalar &) = at::redispatch::addcmul;
  if (op.id == H_ADDCDIV) ptr = at::redispatch::addcdiv;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<const at::Scalar &>()(op.args[3])));
  break;}

case H_CROSS_ENTROPY_LOSS:
case H_NLL_LOSS_ND:
case H_NLL_LOSS:
case H_NLL_LOSS2D:
  set(op, redispatch_ptrs_79[op.id - H_CROSS_ENTROPY_LOSS](ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const c10::optional<at::Tensor> &>()(op.args[2]), load<int64_t>()(op.args[3]), load<int64_t>()(op.args[4])));
  break;

case H_ORMQR_OUT:
  init_update_in_place(op);
  at::redispatch::ormqr_outf(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<bool>()(op.args[3]), load<bool>()(op.args[4]), load<at::Tensor &>()(op.args[5]));
  end_update_in_place(op);
  break;

case H_ORMQR:
  set(op, at::redispatch::ormqr(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<bool>()(op.args[3]), load<bool>()(op.args[4])));
  break;

case H_LU_SOLVE_OUT:
case H_LERP_TENSOR_OUT:
case H_LOG_SIGMOID_BACKWARD_GRAD_INPUT:
case H_ADAPTIVE_MAX_POOL2D_BACKWARD_GRAD_INPUT:
case H_ADAPTIVE_MAX_POOL3D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op);
  redispatch_ptrs_80[op.id - H_LU_SOLVE_OUT](ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<at::Tensor &>()(op.args[3]));
  end_update_in_place(op);
  break;

case H_MULTINOMIAL_OUT:
  init_update_in_place(op);
  at::redispatch::multinomial_outf(ks, load<const at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1]), load<bool>()(op.args[2]), load<c10::optional<at::Generator>>()(op.args[3]), load<at::Tensor &>()(op.args[4]));
  end_update_in_place(op);
  break;

case H_MULTINOMIAL:
  set(op, at::redispatch::multinomial(ks, load<const at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1]), load<bool>()(op.args[2]), load<c10::optional<at::Generator>>()(op.args[3])));
  break;

case H_POLYGAMMA_OUT:
  init_update_in_place(op);
  at::redispatch::polygamma_outf(ks, load<int64_t>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<at::Tensor &>()(op.args[2]));
  end_update_in_place(op);
  break;

case H_HISTC_OUT:
  init_update_in_place(op);
  at::redispatch::histc_outf(ks, load<const at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1]), load<const at::Scalar &>()(op.args[2]), load<const at::Scalar &>()(op.args[3]), load<at::Tensor &>()(op.args[4]));
  end_update_in_place(op);
  break;

case H_HISTC:
  set(op, at::redispatch::histc(ks, load<const at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1]), load<const at::Scalar &>()(op.args[2]), load<const at::Scalar &>()(op.args[3])));
  break;

case H_QUANTILE_SCALAR_OUT:
case H_NANQUANTILE_SCALAR_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, double, c10::optional<int64_t>, bool, at::Tensor &) = at::redispatch::quantile_outf;
  if (op.id == H_NANQUANTILE_SCALAR_OUT) ptr = at::redispatch::nanquantile_outf;
  init_update_in_place(op);
  ptr(ks, load<const at::Tensor &>()(op.args[0]), load<double>()(op.args[1]), load<c10::optional<int64_t>>()(op.args[2]), load<bool>()(op.args[3]), load<at::Tensor &>()(op.args[4]));
  end_update_in_place(op);
  break;}

case H_QUANTILE_SCALAR:
case H_NANQUANTILE_SCALAR:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, double, c10::optional<int64_t>, bool) = at::redispatch::quantile;
  if (op.id == H_NANQUANTILE_SCALAR) ptr = at::redispatch::nanquantile;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<double>()(op.args[1]), load<c10::optional<int64_t>>()(op.args[2]), load<bool>()(op.args[3])));
  break;}

case H_QUANTILE_OUT:
case H_NANQUANTILE_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, c10::optional<int64_t>, bool, at::Tensor &) = at::redispatch::quantile_outf;
  if (op.id == H_NANQUANTILE_OUT) ptr = at::redispatch::nanquantile_outf;
  init_update_in_place(op);
  ptr(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<c10::optional<int64_t>>()(op.args[2]), load<bool>()(op.args[3]), load<at::Tensor &>()(op.args[4]));
  end_update_in_place(op);
  break;}

case H_QUANTILE:
case H_NANQUANTILE:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, c10::optional<int64_t>, bool) = at::redispatch::quantile;
  if (op.id == H_NANQUANTILE) ptr = at::redispatch::nanquantile;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<c10::optional<int64_t>>()(op.args[2]), load<bool>()(op.args[3])));
  break;}

case H_QUANTILE_NEW_SCALAR_OUT:
case H_NANQUANTILE_NEW_SCALAR_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, double, c10::optional<int64_t>, bool, c10::string_view, at::Tensor &) = at::redispatch::quantile_outf;
  if (op.id == H_NANQUANTILE_NEW_SCALAR_OUT) ptr = at::redispatch::nanquantile_outf;
  init_update_in_place(op);
  ptr(ks, load<const at::Tensor &>()(op.args[0]), load<double>()(op.args[1]), load<c10::optional<int64_t>>()(op.args[2]), load<bool>()(op.args[3]), load<c10::string_view>()(op.args[4]), load<at::Tensor &>()(op.args[5]));
  end_update_in_place(op);
  break;}

case H_QUANTILE_NEW_SCALAR:
case H_NANQUANTILE_NEW_SCALAR:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, double, c10::optional<int64_t>, bool, c10::string_view) = at::redispatch::quantile;
  if (op.id == H_NANQUANTILE_NEW_SCALAR) ptr = at::redispatch::nanquantile;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<double>()(op.args[1]), load<c10::optional<int64_t>>()(op.args[2]), load<bool>()(op.args[3]), load<c10::string_view>()(op.args[4])));
  break;}

case H_QUANTILE_NEW_OUT:
case H_NANQUANTILE_NEW_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, c10::optional<int64_t>, bool, c10::string_view, at::Tensor &) = at::redispatch::quantile_outf;
  if (op.id == H_NANQUANTILE_NEW_OUT) ptr = at::redispatch::nanquantile_outf;
  init_update_in_place(op);
  ptr(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<c10::optional<int64_t>>()(op.args[2]), load<bool>()(op.args[3]), load<c10::string_view>()(op.args[4]), load<at::Tensor &>()(op.args[5]));
  end_update_in_place(op);
  break;}

case H_QUANTILE_NEW:
case H_NANQUANTILE_NEW:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, c10::optional<int64_t>, bool, c10::string_view) = at::redispatch::quantile;
  if (op.id == H_NANQUANTILE_NEW) ptr = at::redispatch::nanquantile;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<c10::optional<int64_t>>()(op.args[2]), load<bool>()(op.args[3]), load<c10::string_view>()(op.args[4])));
  break;}

case H_RENORM_OUT:
  init_update_in_place(op);
  at::redispatch::renorm_outf(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Scalar &>()(op.args[1]), load<int64_t>()(op.args[2]), load<const at::Scalar &>()(op.args[3]), load<at::Tensor &>()(op.args[4]));
  end_update_in_place(op);
  break;

case H_NORMAL_TENSOR_FLOAT_OUT:
  init_update_in_place(op);
  at::redispatch::normal_outf(ks, load<const at::Tensor &>()(op.args[0]), load<double>()(op.args[1]), load<c10::optional<at::Generator>>()(op.args[2]), load<at::Tensor &>()(op.args[3]));
  end_update_in_place(op);
  break;

case H_NORMAL_FLOAT_TENSOR_OUT:
  init_update_in_place(op);
  at::redispatch::normal_outf(ks, load<double>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<c10::optional<at::Generator>>()(op.args[2]), load<at::Tensor &>()(op.args[3]));
  end_update_in_place(op);
  break;

case H_NORMAL_FLOAT_TENSOR:
  set(op, at::redispatch::normal(ks, load<double>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<c10::optional<at::Generator>>()(op.args[2])));
  break;

case H_NORMAL_TENSOR_TENSOR_OUT:
  init_update_in_place(op);
  at::redispatch::normal_outf(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<c10::optional<at::Generator>>()(op.args[2]), load<at::Tensor &>()(op.args[3]));
  end_update_in_place(op);
  break;

case H_NORMAL_FLOAT_FLOAT:
  set(op, at::redispatch::normal(ks, load<double>()(op.args[0]), load<double>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<c10::optional<at::Generator>>()(op.args[3]), load<c10::optional<at::ScalarType>>()(op.args[4]), load<c10::optional<at::Layout>>()(op.args[5]), load<c10::optional<at::Device>>()(op.args[6]), load<c10::optional<bool>>()(op.args[7])));
  break;

case H_NORMAL_FLOAT_FLOAT_OUT:
  init_update_in_place(op);
  at::redispatch::normal_outf(ks, load<double>()(op.args[0]), load<double>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<c10::optional<at::Generator>>()(op.args[3]), load<at::Tensor &>()(op.args[4]));
  end_update_in_place(op);
  break;

case H__AMP_UPDATE_SCALE_:
  init_update_in_place(op);
  at::redispatch::_amp_update_scale_(ks, load<at::Tensor &>()(op.args[0]), load<at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<double>()(op.args[3]), load<double>()(op.args[4]), load<int64_t>()(op.args[5]));
  end_update_in_place(op);
  break;

case H_BUCKETIZE_TENSOR:
case H_SEARCHSORTED_TENSOR:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, bool, bool) = at::redispatch::bucketize;
  if (op.id == H_SEARCHSORTED_TENSOR) ptr = at::redispatch::searchsorted;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<bool>()(op.args[2]), load<bool>()(op.args[3])));
  break;}

case H_BUCKETIZE_TENSOR_OUT:
case H_SEARCHSORTED_TENSOR_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, bool, bool, at::Tensor &) = at::redispatch::bucketize_outf;
  if (op.id == H_SEARCHSORTED_TENSOR_OUT) ptr = at::redispatch::searchsorted_outf;
  init_update_in_place(op);
  ptr(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<bool>()(op.args[2]), load<bool>()(op.args[3]), load<at::Tensor &>()(op.args[4]));
  end_update_in_place(op);
  break;}

case H_BUCKETIZE_SCALAR:
  set(op, at::redispatch::bucketize(ks, load<const at::Scalar &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<bool>()(op.args[2]), load<bool>()(op.args[3])));
  break;

case H_SEARCHSORTED_SCALAR:
  set(op, at::redispatch::searchsorted(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Scalar &>()(op.args[1]), load<bool>()(op.args[2]), load<bool>()(op.args[3])));
  break;

case H_MSE_LOSS_OUT:
case H_L1_LOSS_OUT:
case H_MULTILABEL_MARGIN_LOSS_OUT:
case H_SOFT_MARGIN_LOSS_OUT:
case H_GLU_BACKWARD_GRAD_INPUT:
  init_update_in_place(op);
  redispatch_ptrs_81[op.id - H_MSE_LOSS_OUT](ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<int64_t>()(op.args[2]), load<at::Tensor &>()(op.args[3]));
  end_update_in_place(op);
  break;

case H_MSE_LOSS_BACKWARD_GRAD_INPUT:
case H_L1_LOSS_BACKWARD_GRAD_INPUT:
case H_SOFT_MARGIN_LOSS_BACKWARD_GRAD_INPUT:
  init_update_in_place(op);
  redispatch_ptrs_82[op.id - H_MSE_LOSS_BACKWARD_GRAD_INPUT](ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<int64_t>()(op.args[3]), load<at::Tensor &>()(op.args[4]));
  end_update_in_place(op);
  break;

case H_MULTI_MARGIN_LOSS_OUT:
  init_update_in_place(op);
  at::redispatch::multi_margin_loss_outf(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Scalar &>()(op.args[2]), load<const at::Scalar &>()(op.args[3]), load<const c10::optional<at::Tensor> &>()(op.args[4]), load<int64_t>()(op.args[5]), load<at::Tensor &>()(op.args[6]));
  end_update_in_place(op);
  break;

case H_MULTI_MARGIN_LOSS:
  set(op, at::redispatch::multi_margin_loss(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Scalar &>()(op.args[2]), load<const at::Scalar &>()(op.args[3]), load<const c10::optional<at::Tensor> &>()(op.args[4]), load<int64_t>()(op.args[5])));
  break;

case H_MULTI_MARGIN_LOSS_BACKWARD_GRAD_INPUT:
  init_update_in_place(op);
  at::redispatch::multi_margin_loss_backward_outf(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<const at::Scalar &>()(op.args[3]), load<const at::Scalar &>()(op.args[4]), load<const c10::optional<at::Tensor> &>()(op.args[5]), load<int64_t>()(op.args[6]), load<at::Tensor &>()(op.args[7]));
  end_update_in_place(op);
  break;

case H_MULTI_MARGIN_LOSS_BACKWARD:
  set(op, at::redispatch::multi_margin_loss_backward(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<const at::Scalar &>()(op.args[3]), load<const at::Scalar &>()(op.args[4]), load<const c10::optional<at::Tensor> &>()(op.args[5]), load<int64_t>()(op.args[6])));
  break;

case H_MULTILABEL_MARGIN_LOSS_BACKWARD_GRAD_INPUT:
  init_update_in_place(op);
  at::redispatch::multilabel_margin_loss_backward_outf(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<int64_t>()(op.args[3]), load<const at::Tensor &>()(op.args[4]), load<at::Tensor &>()(op.args[5]));
  end_update_in_place(op);
  break;

case H_MULTILABEL_MARGIN_LOSS_BACKWARD:
  set(op, at::redispatch::multilabel_margin_loss_backward(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<int64_t>()(op.args[3]), load<const at::Tensor &>()(op.args[4])));
  break;

case H_NLL_LOSS_OUT:
case H_NLL_LOSS2D_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, int64_t, int64_t, at::Tensor &) = at::redispatch::nll_loss_outf;
  if (op.id == H_NLL_LOSS2D_OUT) ptr = at::redispatch::nll_loss2d_outf;
  init_update_in_place(op);
  ptr(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const c10::optional<at::Tensor> &>()(op.args[2]), load<int64_t>()(op.args[3]), load<int64_t>()(op.args[4]), load<at::Tensor &>()(op.args[5]));
  end_update_in_place(op);
  break;}

case H_NLL_LOSS_BACKWARD_GRAD_INPUT:
case H_NLL_LOSS2D_BACKWARD_GRAD_INPUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, int64_t, int64_t, const at::Tensor &, at::Tensor &) = at::redispatch::nll_loss_backward_outf;
  if (op.id == H_NLL_LOSS2D_BACKWARD_GRAD_INPUT) ptr = at::redispatch::nll_loss2d_backward_outf;
  init_update_in_place(op);
  ptr(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<const c10::optional<at::Tensor> &>()(op.args[3]), load<int64_t>()(op.args[4]), load<int64_t>()(op.args[5]), load<const at::Tensor &>()(op.args[6]), load<at::Tensor &>()(op.args[7]));
  end_update_in_place(op);
  break;}

case H_NLL_LOSS_BACKWARD:
case H_NLL_LOSS2D_BACKWARD:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, int64_t, int64_t, const at::Tensor &) = at::redispatch::nll_loss_backward;
  if (op.id == H_NLL_LOSS2D_BACKWARD) ptr = at::redispatch::nll_loss2d_backward;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<const c10::optional<at::Tensor> &>()(op.args[3]), load<int64_t>()(op.args[4]), load<int64_t>()(op.args[5]), load<const at::Tensor &>()(op.args[6])));
  break;}

case H_SMOOTH_L1_LOSS_OUT:
case H_HUBER_LOSS_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, int64_t, double, at::Tensor &) = at::redispatch::smooth_l1_loss_outf;
  if (op.id == H_HUBER_LOSS_OUT) ptr = at::redispatch::huber_loss_outf;
  init_update_in_place(op);
  ptr(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<int64_t>()(op.args[2]), load<double>()(op.args[3]), load<at::Tensor &>()(op.args[4]));
  end_update_in_place(op);
  break;}

case H_SMOOTH_L1_LOSS_BACKWARD_GRAD_INPUT:
case H_HUBER_LOSS_BACKWARD_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t, double, at::Tensor &) = at::redispatch::smooth_l1_loss_backward_outf;
  if (op.id == H_HUBER_LOSS_BACKWARD_OUT) ptr = at::redispatch::huber_loss_backward_outf;
  init_update_in_place(op);
  ptr(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<int64_t>()(op.args[3]), load<double>()(op.args[4]), load<at::Tensor &>()(op.args[5]));
  end_update_in_place(op);
  break;}

case H_SMOOTH_L1_LOSS_BACKWARD:
case H_HUBER_LOSS_BACKWARD:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t, double) = at::redispatch::smooth_l1_loss_backward;
  if (op.id == H_HUBER_LOSS_BACKWARD) ptr = at::redispatch::huber_loss_backward;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<int64_t>()(op.args[3]), load<double>()(op.args[4])));
  break;}

case H_ELU_OUT:
  init_update_in_place(op);
  at::redispatch::elu_outf(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Scalar &>()(op.args[1]), load<const at::Scalar &>()(op.args[2]), load<const at::Scalar &>()(op.args[3]), load<at::Tensor &>()(op.args[4]));
  end_update_in_place(op);
  break;

case H_ELU_BACKWARD_GRAD_INPUT:
  init_update_in_place(op);
  at::redispatch::elu_backward_outf(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Scalar &>()(op.args[1]), load<const at::Scalar &>()(op.args[2]), load<const at::Scalar &>()(op.args[3]), load<bool>()(op.args[4]), load<const at::Tensor &>()(op.args[5]), load<at::Tensor &>()(op.args[6]));
  end_update_in_place(op);
  break;

case H_ELU_:
  init_update_in_place(op);
  at::redispatch::elu_(ks, load<at::Tensor &>()(op.args[0]), load<const at::Scalar &>()(op.args[1]), load<const at::Scalar &>()(op.args[2]), load<const at::Scalar &>()(op.args[3]));
  end_update_in_place(op);
  break;

case H_HARDTANH_BACKWARD_GRAD_INPUT:
  init_update_in_place(op);
  at::redispatch::hardtanh_backward_outf(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Scalar &>()(op.args[2]), load<const at::Scalar &>()(op.args[3]), load<at::Tensor &>()(op.args[4]));
  end_update_in_place(op);
  break;

case H_HARDTANH_BACKWARD:
  set(op, at::redispatch::hardtanh_backward(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Scalar &>()(op.args[2]), load<const at::Scalar &>()(op.args[3])));
  break;

case H_LEAKY_RELU_BACKWARD_GRAD_INPUT:
  init_update_in_place(op);
  at::redispatch::leaky_relu_backward_outf(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Scalar &>()(op.args[2]), load<bool>()(op.args[3]), load<at::Tensor &>()(op.args[4]));
  end_update_in_place(op);
  break;

case H_RRELU_WITH_NOISE_OUT:
  init_update_in_place(op);
  at::redispatch::rrelu_with_noise_outf(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Scalar &>()(op.args[2]), load<const at::Scalar &>()(op.args[3]), load<bool>()(op.args[4]), load<c10::optional<at::Generator>>()(op.args[5]), load<at::Tensor &>()(op.args[6]));
  end_update_in_place(op);
  break;

case H_RRELU_WITH_NOISE:
  set(op, at::redispatch::rrelu_with_noise(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Scalar &>()(op.args[2]), load<const at::Scalar &>()(op.args[3]), load<bool>()(op.args[4]), load<c10::optional<at::Generator>>()(op.args[5])));
  break;

case H_RRELU_WITH_NOISE_BACKWARD:
  set(op, at::redispatch::rrelu_with_noise_backward(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<const at::Scalar &>()(op.args[3]), load<const at::Scalar &>()(op.args[4]), load<bool>()(op.args[5]), load<bool>()(op.args[6])));
  break;

case H_RRELU_WITH_NOISE_:
  init_update_in_place(op);
  at::redispatch::rrelu_with_noise_(ks, load<at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Scalar &>()(op.args[2]), load<const at::Scalar &>()(op.args[3]), load<bool>()(op.args[4]), load<c10::optional<at::Generator>>()(op.args[5]));
  end_update_in_place(op);
  break;

case H_SOFTPLUS_BACKWARD_GRAD_INPUT:
  init_update_in_place(op);
  at::redispatch::softplus_backward_outf(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Scalar &>()(op.args[2]), load<const at::Scalar &>()(op.args[3]), load<const at::Tensor &>()(op.args[4]), load<at::Tensor &>()(op.args[5]));
  end_update_in_place(op);
  break;

case H_ADAPTIVE_AVG_POOL2D_OUT:
case H_ADAPTIVE_AVG_POOL3D_OUT:
case H_REFLECTION_PAD1D_OUT:
case H_REFLECTION_PAD2D_OUT:
case H_REPLICATION_PAD1D_OUT:
case H_REPLICATION_PAD2D_OUT:
case H_REPLICATION_PAD3D_OUT:
  init_update_in_place(op);
  redispatch_ptrs_83[op.id - H_ADAPTIVE_AVG_POOL2D_OUT](ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<at::Tensor &>()(op.args[2]));
  end_update_in_place(op);
  break;

case H_AVG_POOL2D_OUT:
case H_AVG_POOL3D_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, bool, bool, c10::optional<int64_t>, at::Tensor &) = at::redispatch::avg_pool2d_outf;
  if (op.id == H_AVG_POOL3D_OUT) ptr = at::redispatch::avg_pool3d_outf;
  init_update_in_place(op);
  ptr(ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<at::IntArrayRef>()(op.args[3]), load<bool>()(op.args[4]), load<bool>()(op.args[5]), load<c10::optional<int64_t>>()(op.args[6]), load<at::Tensor &>()(op.args[7]));
  end_update_in_place(op);
  break;}

case H_AVG_POOL2D:
case H_AVG_POOL3D:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, bool, bool, c10::optional<int64_t>) = at::redispatch::avg_pool2d;
  if (op.id == H_AVG_POOL3D) ptr = at::redispatch::avg_pool3d;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<at::IntArrayRef>()(op.args[3]), load<bool>()(op.args[4]), load<bool>()(op.args[5]), load<c10::optional<int64_t>>()(op.args[6])));
  break;}

case H_AVG_POOL2D_BACKWARD_GRAD_INPUT:
case H_AVG_POOL3D_BACKWARD_GRAD_INPUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, bool, bool, c10::optional<int64_t>, at::Tensor &) = at::redispatch::avg_pool2d_backward_outf;
  if (op.id == H_AVG_POOL3D_BACKWARD_GRAD_INPUT) ptr = at::redispatch::avg_pool3d_backward_outf;
  init_update_in_place(op);
  ptr(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<at::IntArrayRef>()(op.args[3]), load<at::IntArrayRef>()(op.args[4]), load<bool>()(op.args[5]), load<bool>()(op.args[6]), load<c10::optional<int64_t>>()(op.args[7]), load<at::Tensor &>()(op.args[8]));
  end_update_in_place(op);
  break;}

case H_AVG_POOL2D_BACKWARD:
case H_AVG_POOL3D_BACKWARD:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, bool, bool, c10::optional<int64_t>) = at::redispatch::avg_pool2d_backward;
  if (op.id == H_AVG_POOL3D_BACKWARD) ptr = at::redispatch::avg_pool3d_backward;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<at::IntArrayRef>()(op.args[3]), load<at::IntArrayRef>()(op.args[4]), load<bool>()(op.args[5]), load<bool>()(op.args[6]), load<c10::optional<int64_t>>()(op.args[7])));
  break;}

case H_FRACTIONAL_MAX_POOL2D_BACKWARD_GRAD_INPUT:
case H_FRACTIONAL_MAX_POOL3D_BACKWARD_GRAD_INPUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, const at::Tensor &, at::Tensor &) = at::redispatch::fractional_max_pool2d_backward_outf;
  if (op.id == H_FRACTIONAL_MAX_POOL3D_BACKWARD_GRAD_INPUT) ptr = at::redispatch::fractional_max_pool3d_backward_outf;
  init_update_in_place(op);
  ptr(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<at::IntArrayRef>()(op.args[3]), load<const at::Tensor &>()(op.args[4]), load<at::Tensor &>()(op.args[5]));
  end_update_in_place(op);
  break;}

case H_FRACTIONAL_MAX_POOL2D_BACKWARD:
case H_FRACTIONAL_MAX_POOL3D_BACKWARD:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, const at::Tensor &) = at::redispatch::fractional_max_pool2d_backward;
  if (op.id == H_FRACTIONAL_MAX_POOL3D_BACKWARD) ptr = at::redispatch::fractional_max_pool3d_backward;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<at::IntArrayRef>()(op.args[3]), load<const at::Tensor &>()(op.args[4])));
  break;}

case H_MAX_POOL2D_WITH_INDICES_BACKWARD_GRAD_INPUT:
case H_MAX_POOL3D_WITH_INDICES_BACKWARD_GRAD_INPUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, bool, const at::Tensor &, at::Tensor &) = at::redispatch::max_pool2d_with_indices_backward_outf;
  if (op.id == H_MAX_POOL3D_WITH_INDICES_BACKWARD_GRAD_INPUT) ptr = at::redispatch::max_pool3d_with_indices_backward_outf;
  init_update_in_place(op);
  ptr(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<at::IntArrayRef>()(op.args[3]), load<at::IntArrayRef>()(op.args[4]), load<at::IntArrayRef>()(op.args[5]), load<bool>()(op.args[6]), load<const at::Tensor &>()(op.args[7]), load<at::Tensor &>()(op.args[8]));
  end_update_in_place(op);
  break;}

case H_MAX_POOL3D_WITH_INDICES_BACKWARD:
  set(op, at::redispatch::max_pool3d_with_indices_backward(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<at::IntArrayRef>()(op.args[3]), load<at::IntArrayRef>()(op.args[4]), load<at::IntArrayRef>()(op.args[5]), load<bool>()(op.args[6]), load<const at::Tensor &>()(op.args[7])));
  break;

case H_MAX_UNPOOL2D_OUT:
case H_REFLECTION_PAD1D_BACKWARD_GRAD_INPUT:
case H_REFLECTION_PAD2D_BACKWARD_GRAD_INPUT:
case H_REPLICATION_PAD1D_BACKWARD_GRAD_INPUT:
case H_REPLICATION_PAD2D_BACKWARD_GRAD_INPUT:
case H_REPLICATION_PAD3D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op);
  redispatch_ptrs_84[op.id - H_MAX_UNPOOL2D_OUT](ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<at::Tensor &>()(op.args[3]));
  end_update_in_place(op);
  break;

case H_MAX_UNPOOL2D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op);
  at::redispatch::max_unpool2d_backward_outf(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<at::IntArrayRef>()(op.args[3]), load<at::Tensor &>()(op.args[4]));
  end_update_in_place(op);
  break;

case H_MAX_UNPOOL3D_OUT:
  init_update_in_place(op);
  at::redispatch::max_unpool3d_outf(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<at::IntArrayRef>()(op.args[3]), load<at::IntArrayRef>()(op.args[4]), load<at::Tensor &>()(op.args[5]));
  end_update_in_place(op);
  break;

case H_MAX_UNPOOL3D:
  set(op, at::redispatch::max_unpool3d(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<at::IntArrayRef>()(op.args[3]), load<at::IntArrayRef>()(op.args[4])));
  break;

case H_MAX_UNPOOL3D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op);
  at::redispatch::max_unpool3d_backward_outf(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<at::IntArrayRef>()(op.args[3]), load<at::IntArrayRef>()(op.args[4]), load<at::IntArrayRef>()(op.args[5]), load<at::Tensor &>()(op.args[6]));
  end_update_in_place(op);
  break;

case H_MAX_UNPOOL3D_BACKWARD:
  set(op, at::redispatch::max_unpool3d_backward(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<at::IntArrayRef>()(op.args[3]), load<at::IntArrayRef>()(op.args[4]), load<at::IntArrayRef>()(op.args[5])));
  break;

case H_UPSAMPLE_LINEAR1D_VEC:
case H_UPSAMPLE_BILINEAR2D_VEC:
case H_UPSAMPLE_TRILINEAR3D_VEC:
case H_UPSAMPLE_BICUBIC2D_VEC:
  set(op, redispatch_ptrs_85[op.id - H_UPSAMPLE_LINEAR1D_VEC](ks, load<const at::Tensor &>()(op.args[0]), load<c10::optional<at::IntArrayRef>>()(op.args[1]), load<bool>()(op.args[2]), load<c10::optional<at::ArrayRef<double>>>()(op.args[3])));
  break;

case H_UPSAMPLE_LINEAR1D_BACKWARD_VEC:
case H_UPSAMPLE_BILINEAR2D_BACKWARD_VEC:
case H_UPSAMPLE_TRILINEAR3D_BACKWARD_VEC:
case H_UPSAMPLE_BICUBIC2D_BACKWARD_VEC:
  set(op, redispatch_ptrs_86[op.id - H_UPSAMPLE_LINEAR1D_BACKWARD_VEC](ks, load<const at::Tensor &>()(op.args[0]), load<c10::optional<at::IntArrayRef>>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<bool>()(op.args[3]), load<c10::optional<at::ArrayRef<double>>>()(op.args[4])));
  break;

case H_UPSAMPLE_NEAREST1D_VEC:
case H_UPSAMPLE_NEAREST2D_VEC:
case H_UPSAMPLE_NEAREST3D_VEC:
  set(op, redispatch_ptrs_87[op.id - H_UPSAMPLE_NEAREST1D_VEC](ks, load<const at::Tensor &>()(op.args[0]), load<c10::optional<at::IntArrayRef>>()(op.args[1]), load<c10::optional<at::ArrayRef<double>>>()(op.args[2])));
  break;

case H_UPSAMPLE_NEAREST1D_BACKWARD_VEC:
case H_UPSAMPLE_NEAREST2D_BACKWARD_VEC:
case H_UPSAMPLE_NEAREST3D_BACKWARD_VEC:
  set(op, redispatch_ptrs_88[op.id - H_UPSAMPLE_NEAREST1D_BACKWARD_VEC](ks, load<const at::Tensor &>()(op.args[0]), load<c10::optional<at::IntArrayRef>>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<c10::optional<at::ArrayRef<double>>>()(op.args[3])));
  break;

case H_UPSAMPLE_LINEAR1D_OUT:
  init_update_in_place(op);
  at::redispatch::upsample_linear1d_outf(ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<bool>()(op.args[2]), load<c10::optional<double>>()(op.args[3]), load<at::Tensor &>()(op.args[4]));
  end_update_in_place(op);
  break;

case H_UPSAMPLE_LINEAR1D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op);
  at::redispatch::upsample_linear1d_backward_outf(ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<bool>()(op.args[3]), load<c10::optional<double>>()(op.args[4]), load<at::Tensor &>()(op.args[5]));
  end_update_in_place(op);
  break;

case H_UPSAMPLE_BILINEAR2D_OUT:
case H_UPSAMPLE_BICUBIC2D_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, at::IntArrayRef, bool, c10::optional<double>, c10::optional<double>, at::Tensor &) = at::redispatch::upsample_bilinear2d_outf;
  if (op.id == H_UPSAMPLE_BICUBIC2D_OUT) ptr = at::redispatch::upsample_bicubic2d_outf;
  init_update_in_place(op);
  ptr(ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<bool>()(op.args[2]), load<c10::optional<double>>()(op.args[3]), load<c10::optional<double>>()(op.args[4]), load<at::Tensor &>()(op.args[5]));
  end_update_in_place(op);
  break;}

case H_UPSAMPLE_BILINEAR2D:
  set(op, at::redispatch::upsample_bilinear2d(ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<bool>()(op.args[2]), load<c10::optional<double>>()(op.args[3]), load<c10::optional<double>>()(op.args[4])));
  break;

case H_UPSAMPLE_BILINEAR2D_BACKWARD_GRAD_INPUT:
case H_UPSAMPLE_BICUBIC2D_BACKWARD_GRAD_INPUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, bool, c10::optional<double>, c10::optional<double>, at::Tensor &) = at::redispatch::upsample_bilinear2d_backward_outf;
  if (op.id == H_UPSAMPLE_BICUBIC2D_BACKWARD_GRAD_INPUT) ptr = at::redispatch::upsample_bicubic2d_backward_outf;
  init_update_in_place(op);
  ptr(ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<bool>()(op.args[3]), load<c10::optional<double>>()(op.args[4]), load<c10::optional<double>>()(op.args[5]), load<at::Tensor &>()(op.args[6]));
  end_update_in_place(op);
  break;}

case H_UPSAMPLE_TRILINEAR3D_OUT:
  init_update_in_place(op);
  at::redispatch::upsample_trilinear3d_outf(ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<bool>()(op.args[2]), load<c10::optional<double>>()(op.args[3]), load<c10::optional<double>>()(op.args[4]), load<c10::optional<double>>()(op.args[5]), load<at::Tensor &>()(op.args[6]));
  end_update_in_place(op);
  break;

case H_UPSAMPLE_TRILINEAR3D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op);
  at::redispatch::upsample_trilinear3d_backward_outf(ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<bool>()(op.args[3]), load<c10::optional<double>>()(op.args[4]), load<c10::optional<double>>()(op.args[5]), load<c10::optional<double>>()(op.args[6]), load<at::Tensor &>()(op.args[7]));
  end_update_in_place(op);
  break;

case H_UPSAMPLE_NEAREST1D_OUT:
  init_update_in_place(op);
  at::redispatch::upsample_nearest1d_outf(ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<c10::optional<double>>()(op.args[2]), load<at::Tensor &>()(op.args[3]));
  end_update_in_place(op);
  break;

case H_UPSAMPLE_NEAREST1D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op);
  at::redispatch::upsample_nearest1d_backward_outf(ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<c10::optional<double>>()(op.args[3]), load<at::Tensor &>()(op.args[4]));
  end_update_in_place(op);
  break;

case H_UPSAMPLE_NEAREST2D_OUT:
  init_update_in_place(op);
  at::redispatch::upsample_nearest2d_outf(ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<c10::optional<double>>()(op.args[2]), load<c10::optional<double>>()(op.args[3]), load<at::Tensor &>()(op.args[4]));
  end_update_in_place(op);
  break;

case H_UPSAMPLE_NEAREST2D:
  set(op, at::redispatch::upsample_nearest2d(ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<c10::optional<double>>()(op.args[2]), load<c10::optional<double>>()(op.args[3])));
  break;

case H_UPSAMPLE_NEAREST2D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op);
  at::redispatch::upsample_nearest2d_backward_outf(ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<c10::optional<double>>()(op.args[3]), load<c10::optional<double>>()(op.args[4]), load<at::Tensor &>()(op.args[5]));
  end_update_in_place(op);
  break;

case H_UPSAMPLE_NEAREST3D_OUT:
  init_update_in_place(op);
  at::redispatch::upsample_nearest3d_outf(ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<c10::optional<double>>()(op.args[2]), load<c10::optional<double>>()(op.args[3]), load<c10::optional<double>>()(op.args[4]), load<at::Tensor &>()(op.args[5]));
  end_update_in_place(op);
  break;

case H_UPSAMPLE_NEAREST3D:
  set(op, at::redispatch::upsample_nearest3d(ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<c10::optional<double>>()(op.args[2]), load<c10::optional<double>>()(op.args[3]), load<c10::optional<double>>()(op.args[4])));
  break;

case H_UPSAMPLE_NEAREST3D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op);
  at::redispatch::upsample_nearest3d_backward_outf(ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<c10::optional<double>>()(op.args[3]), load<c10::optional<double>>()(op.args[4]), load<c10::optional<double>>()(op.args[5]), load<at::Tensor &>()(op.args[6]));
  end_update_in_place(op);
  break;

case H_LOGIT_BACKWARD_GRAD_INPUT:
  init_update_in_place(op);
  at::redispatch::logit_backward_outf(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<c10::optional<double>>()(op.args[2]), load<at::Tensor &>()(op.args[3]));
  end_update_in_place(op);
  break;

case H_LOGIT_BACKWARD:
  set(op, at::redispatch::logit_backward(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<c10::optional<double>>()(op.args[2])));
  break;

case H_SLOW_CONV_TRANSPOSE2D_OUT:
case H_SLOW_CONV_TRANSPOSE3D_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, at::IntArrayRef, const c10::optional<at::Tensor> &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::Tensor &) = at::redispatch::slow_conv_transpose2d_outf;
  if (op.id == H_SLOW_CONV_TRANSPOSE3D_OUT) ptr = at::redispatch::slow_conv_transpose3d_outf;
  init_update_in_place(op);
  ptr(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<const c10::optional<at::Tensor> &>()(op.args[3]), load<at::IntArrayRef>()(op.args[4]), load<at::IntArrayRef>()(op.args[5]), load<at::IntArrayRef>()(op.args[6]), load<at::IntArrayRef>()(op.args[7]), load<at::Tensor &>()(op.args[8]));
  end_update_in_place(op);
  break;}

case H_SLOW_CONV_TRANSPOSE2D:
case H_SLOW_CONV_TRANSPOSE3D:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, at::IntArrayRef, const c10::optional<at::Tensor> &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef) = at::redispatch::slow_conv_transpose2d;
  if (op.id == H_SLOW_CONV_TRANSPOSE3D) ptr = at::redispatch::slow_conv_transpose3d;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<const c10::optional<at::Tensor> &>()(op.args[3]), load<at::IntArrayRef>()(op.args[4]), load<at::IntArrayRef>()(op.args[5]), load<at::IntArrayRef>()(op.args[6]), load<at::IntArrayRef>()(op.args[7])));
  break;}

case H_THNN_CONV2D_OUT:
case H_SLOW_CONV3D_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, at::IntArrayRef, const c10::optional<at::Tensor> &, at::IntArrayRef, at::IntArrayRef, at::Tensor &) = at::redispatch::thnn_conv2d_outf;
  if (op.id == H_SLOW_CONV3D_OUT) ptr = at::redispatch::slow_conv3d_outf;
  init_update_in_place(op);
  ptr(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<const c10::optional<at::Tensor> &>()(op.args[3]), load<at::IntArrayRef>()(op.args[4]), load<at::IntArrayRef>()(op.args[5]), load<at::Tensor &>()(op.args[6]));
  end_update_in_place(op);
  break;}

case H_THNN_CONV2D:
case H_SLOW_CONV3D:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, at::IntArrayRef, const c10::optional<at::Tensor> &, at::IntArrayRef, at::IntArrayRef) = at::redispatch::thnn_conv2d;
  if (op.id == H_SLOW_CONV3D) ptr = at::redispatch::slow_conv3d;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<const c10::optional<at::Tensor> &>()(op.args[3]), load<at::IntArrayRef>()(op.args[4]), load<at::IntArrayRef>()(op.args[5])));
  break;}

case H_THNN_CONV_DEPTHWISE2D_OUT:
case H_THNN_CONV_DEPTHWISE2D_FORWARD_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, at::IntArrayRef, const c10::optional<at::Tensor> &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::Tensor &) = at::redispatch::thnn_conv_depthwise2d_outf;
  if (op.id == H_THNN_CONV_DEPTHWISE2D_FORWARD_OUT) ptr = at::redispatch::thnn_conv_depthwise2d_forward_outf;
  init_update_in_place(op);
  ptr(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<const c10::optional<at::Tensor> &>()(op.args[3]), load<at::IntArrayRef>()(op.args[4]), load<at::IntArrayRef>()(op.args[5]), load<at::IntArrayRef>()(op.args[6]), load<at::Tensor &>()(op.args[7]));
  end_update_in_place(op);
  break;}

case H_THNN_CONV_DEPTHWISE2D:
case H_THNN_CONV_DEPTHWISE2D_FORWARD:
case H_CONV_DEPTHWISE3D:
case H_SLOW_CONV_DILATED2D:
case H_SLOW_CONV_DILATED3D:
  set(op, redispatch_ptrs_89[op.id - H_THNN_CONV_DEPTHWISE2D](ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<const c10::optional<at::Tensor> &>()(op.args[3]), load<at::IntArrayRef>()(op.args[4]), load<at::IntArrayRef>()(op.args[5]), load<at::IntArrayRef>()(op.args[6])));
  break;

case H_COL2IM_OUT:
case H_IM2COL_BACKWARD_GRAD_INPUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::Tensor &) = at::redispatch::col2im_outf;
  if (op.id == H_IM2COL_BACKWARD_GRAD_INPUT) ptr = at::redispatch::im2col_backward_outf;
  init_update_in_place(op);
  ptr(ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<at::IntArrayRef>()(op.args[3]), load<at::IntArrayRef>()(op.args[4]), load<at::IntArrayRef>()(op.args[5]), load<at::Tensor &>()(op.args[6]));
  end_update_in_place(op);
  break;}

case H_COL2IM:
case H_IM2COL_BACKWARD:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef) = at::redispatch::col2im;
  if (op.id == H_IM2COL_BACKWARD) ptr = at::redispatch::im2col_backward;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<at::IntArrayRef>()(op.args[3]), load<at::IntArrayRef>()(op.args[4]), load<at::IntArrayRef>()(op.args[5])));
  break;}

case H_COL2IM_BACKWARD_GRAD_INPUT:
case H_IM2COL_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::Tensor &) = at::redispatch::col2im_backward_outf;
  if (op.id == H_IM2COL_OUT) ptr = at::redispatch::im2col_outf;
  init_update_in_place(op);
  ptr(ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<at::IntArrayRef>()(op.args[3]), load<at::IntArrayRef>()(op.args[4]), load<at::Tensor &>()(op.args[5]));
  end_update_in_place(op);
  break;}

case H_COL2IM_BACKWARD:
case H_IM2COL:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef) = at::redispatch::col2im_backward;
  if (op.id == H_IM2COL) ptr = at::redispatch::im2col;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<at::IntArrayRef>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<at::IntArrayRef>()(op.args[3]), load<at::IntArrayRef>()(op.args[4])));
  break;}

case H_FFT_FFT:
case H_FFT_IFFT:
case H_FFT_RFFT:
case H_FFT_IRFFT:
case H_FFT_HFFT:
case H_FFT_IHFFT:
  set(op, redispatch_ptrs_90[op.id - H_FFT_FFT](ks, load<const at::Tensor &>()(op.args[0]), load<c10::optional<int64_t>>()(op.args[1]), load<int64_t>()(op.args[2]), load<c10::optional<c10::string_view>>()(op.args[3])));
  break;

case H_FFT_FFT_OUT:
case H_FFT_IFFT_OUT:
case H_FFT_RFFT_OUT:
case H_FFT_IRFFT_OUT:
case H_FFT_HFFT_OUT:
case H_FFT_IHFFT_OUT:
  init_update_in_place(op);
  redispatch_ptrs_91[op.id - H_FFT_FFT_OUT](ks, load<const at::Tensor &>()(op.args[0]), load<c10::optional<int64_t>>()(op.args[1]), load<int64_t>()(op.args[2]), load<c10::optional<c10::string_view>>()(op.args[3]), load<at::Tensor &>()(op.args[4]));
  end_update_in_place(op);
  break;

case H_FFT_FFT2:
case H_FFT_IFFT2:
case H_FFT_RFFT2:
case H_FFT_IRFFT2:
  set(op, redispatch_ptrs_92[op.id - H_FFT_FFT2](ks, load<const at::Tensor &>()(op.args[0]), load<c10::optional<at::IntArrayRef>>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<c10::optional<c10::string_view>>()(op.args[3])));
  break;

case H_FFT_FFT2_OUT:
case H_FFT_IFFT2_OUT:
case H_FFT_RFFT2_OUT:
case H_FFT_IRFFT2_OUT:
  init_update_in_place(op);
  redispatch_ptrs_93[op.id - H_FFT_FFT2_OUT](ks, load<const at::Tensor &>()(op.args[0]), load<c10::optional<at::IntArrayRef>>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<c10::optional<c10::string_view>>()(op.args[3]), load<at::Tensor &>()(op.args[4]));
  end_update_in_place(op);
  break;

case H_FFT_FFTN:
case H_FFT_IFFTN:
case H_FFT_RFFTN:
case H_FFT_IRFFTN:
  set(op, redispatch_ptrs_94[op.id - H_FFT_FFTN](ks, load<const at::Tensor &>()(op.args[0]), load<c10::optional<at::IntArrayRef>>()(op.args[1]), load<c10::optional<at::IntArrayRef>>()(op.args[2]), load<c10::optional<c10::string_view>>()(op.args[3])));
  break;

case H_FFT_FFTN_OUT:
case H_FFT_IFFTN_OUT:
case H_FFT_RFFTN_OUT:
case H_FFT_IRFFTN_OUT:
  init_update_in_place(op);
  redispatch_ptrs_95[op.id - H_FFT_FFTN_OUT](ks, load<const at::Tensor &>()(op.args[0]), load<c10::optional<at::IntArrayRef>>()(op.args[1]), load<c10::optional<at::IntArrayRef>>()(op.args[2]), load<c10::optional<c10::string_view>>()(op.args[3]), load<at::Tensor &>()(op.args[4]));
  end_update_in_place(op);
  break;

case H_FFT_FFTFREQ:
case H_FFT_RFFTFREQ:
  {at::Tensor(*ptr)(DispatchKeySet, int64_t, double, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<at::Device>, c10::optional<bool>) = at::redispatch::fft_fftfreq;
  if (op.id == H_FFT_RFFTFREQ) ptr = at::redispatch::fft_rfftfreq;
  set(op, ptr(ks, load<int64_t>()(op.args[0]), load<double>()(op.args[1]), load<c10::optional<at::ScalarType>>()(op.args[2]), load<c10::optional<at::Layout>>()(op.args[3]), load<c10::optional<at::Device>>()(op.args[4]), load<c10::optional<bool>>()(op.args[5])));
  break;}

case H_FFT_FFTFREQ_OUT:
case H_FFT_RFFTFREQ_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, int64_t, double, at::Tensor &) = at::redispatch::fft_fftfreq_outf;
  if (op.id == H_FFT_RFFTFREQ_OUT) ptr = at::redispatch::fft_rfftfreq_outf;
  init_update_in_place(op);
  ptr(ks, load<int64_t>()(op.args[0]), load<double>()(op.args[1]), load<at::Tensor &>()(op.args[2]));
  end_update_in_place(op);
  break;}

case H_FFT_FFTSHIFT:
case H_FFT_IFFTSHIFT:
case H__TEST_OPTIONAL_INTLIST:
case H__TEST_OPTIONAL_FILLED_INTLIST:
  set(op, redispatch_ptrs_96[op.id - H_FFT_FFTSHIFT](ks, load<const at::Tensor &>()(op.args[0]), load<c10::optional<at::IntArrayRef>>()(op.args[1])));
  break;

case H_LINALG_EIGVALSH:
case H_LINALG_COND_P_STR:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, c10::string_view) = at::redispatch::linalg_eigvalsh;
  if (op.id == H_LINALG_COND_P_STR) ptr = at::redispatch::linalg_cond;
  set(op, ptr(ks, load<const at::Tensor &>()(op.args[0]), load<c10::string_view>()(op.args[1])));
  break;}

case H_LINALG_EIGVALSH_OUT:
case H_LINALG_COND_P_STR_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, c10::string_view, at::Tensor &) = at::redispatch::linalg_eigvalsh_outf;
  if (op.id == H_LINALG_COND_P_STR_OUT) ptr = at::redispatch::linalg_cond_outf;
  init_update_in_place(op);
  ptr(ks, load<const at::Tensor &>()(op.args[0]), load<c10::string_view>()(op.args[1]), load<at::Tensor &>()(op.args[2]));
  end_update_in_place(op);
  break;}

case H__LINALG_INV_OUT_HELPER_:
case H__LINALG_SOLVE_OUT_HELPER_:
  {at::Tensor &(*ptr)(DispatchKeySet, at::Tensor &, at::Tensor &, at::Tensor &) = at::redispatch::_linalg_inv_out_helper_;
  if (op.id == H__LINALG_SOLVE_OUT_HELPER_) ptr = at::redispatch::_linalg_solve_out_helper_;
  init_update_in_place(op);
  ptr(ks, load<at::Tensor &>()(op.args[0]), load<at::Tensor &>()(op.args[1]), load<at::Tensor &>()(op.args[2]));
  end_update_in_place(op);
  break;}

case H_LINALG_NORM:
  set(op, at::redispatch::linalg_norm(ks, load<const at::Tensor &>()(op.args[0]), load<const c10::optional<at::Scalar> &>()(op.args[1]), load<c10::optional<at::IntArrayRef>>()(op.args[2]), load<bool>()(op.args[3]), load<c10::optional<at::ScalarType>>()(op.args[4])));
  break;

case H_LINALG_NORM_ORD_STR:
  set(op, at::redispatch::linalg_norm(ks, load<const at::Tensor &>()(op.args[0]), load<c10::string_view>()(op.args[1]), load<c10::optional<at::IntArrayRef>>()(op.args[2]), load<bool>()(op.args[3]), load<c10::optional<at::ScalarType>>()(op.args[4])));
  break;

case H_LINALG_NORM_OUT:
  init_update_in_place(op);
  at::redispatch::linalg_norm_outf(ks, load<const at::Tensor &>()(op.args[0]), load<const c10::optional<at::Scalar> &>()(op.args[1]), load<c10::optional<at::IntArrayRef>>()(op.args[2]), load<bool>()(op.args[3]), load<c10::optional<at::ScalarType>>()(op.args[4]), load<at::Tensor &>()(op.args[5]));
  end_update_in_place(op);
  break;

case H_LINALG_NORM_ORD_STR_OUT:
  init_update_in_place(op);
  at::redispatch::linalg_norm_outf(ks, load<const at::Tensor &>()(op.args[0]), load<c10::string_view>()(op.args[1]), load<c10::optional<at::IntArrayRef>>()(op.args[2]), load<bool>()(op.args[3]), load<c10::optional<at::ScalarType>>()(op.args[4]), load<at::Tensor &>()(op.args[5]));
  end_update_in_place(op);
  break;

case H_LINALG_VECTOR_NORM:
  set(op, at::redispatch::linalg_vector_norm(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Scalar &>()(op.args[1]), load<c10::optional<at::IntArrayRef>>()(op.args[2]), load<bool>()(op.args[3]), load<c10::optional<at::ScalarType>>()(op.args[4])));
  break;

case H_LINALG_VECTOR_NORM_OUT:
  init_update_in_place(op);
  at::redispatch::linalg_vector_norm_outf(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Scalar &>()(op.args[1]), load<c10::optional<at::IntArrayRef>>()(op.args[2]), load<bool>()(op.args[3]), load<c10::optional<at::ScalarType>>()(op.args[4]), load<at::Tensor &>()(op.args[5]));
  end_update_in_place(op);
  break;

case H_LINALG_MATRIX_NORM:
  set(op, at::redispatch::linalg_matrix_norm(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Scalar &>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<bool>()(op.args[3]), load<c10::optional<at::ScalarType>>()(op.args[4])));
  break;

case H_LINALG_MATRIX_NORM_OUT:
  init_update_in_place(op);
  at::redispatch::linalg_matrix_norm_outf(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Scalar &>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<bool>()(op.args[3]), load<c10::optional<at::ScalarType>>()(op.args[4]), load<at::Tensor &>()(op.args[5]));
  end_update_in_place(op);
  break;

case H_LINALG_MATRIX_NORM_STR_ORD:
  set(op, at::redispatch::linalg_matrix_norm(ks, load<const at::Tensor &>()(op.args[0]), load<c10::string_view>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<bool>()(op.args[3]), load<c10::optional<at::ScalarType>>()(op.args[4])));
  break;

case H_LINALG_MATRIX_NORM_STR_ORD_OUT:
  init_update_in_place(op);
  at::redispatch::linalg_matrix_norm_outf(ks, load<const at::Tensor &>()(op.args[0]), load<c10::string_view>()(op.args[1]), load<at::IntArrayRef>()(op.args[2]), load<bool>()(op.args[3]), load<c10::optional<at::ScalarType>>()(op.args[4]), load<at::Tensor &>()(op.args[5]));
  end_update_in_place(op);
  break;

case H_LINALG_COND:
  set(op, at::redispatch::linalg_cond(ks, load<const at::Tensor &>()(op.args[0]), load<const c10::optional<at::Scalar> &>()(op.args[1])));
  break;

case H_LINALG_COND_OUT:
  init_update_in_place(op);
  at::redispatch::linalg_cond_outf(ks, load<const at::Tensor &>()(op.args[0]), load<const c10::optional<at::Scalar> &>()(op.args[1]), load<at::Tensor &>()(op.args[2]));
  end_update_in_place(op);
  break;

case H_LINALG_PINV_OUT:
  init_update_in_place(op);
  at::redispatch::linalg_pinv_outf(ks, load<const at::Tensor &>()(op.args[0]), load<double>()(op.args[1]), load<bool>()(op.args[2]), load<at::Tensor &>()(op.args[3]));
  end_update_in_place(op);
  break;

case H_LINALG_TENSORSOLVE:
  set(op, at::redispatch::linalg_tensorsolve(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<c10::optional<at::IntArrayRef>>()(op.args[2])));
  break;

case H_LINALG_TENSORSOLVE_OUT:
  init_update_in_place(op);
  at::redispatch::linalg_tensorsolve_outf(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<c10::optional<at::IntArrayRef>>()(op.args[2]), load<at::Tensor &>()(op.args[3]));
  end_update_in_place(op);
  break;

case H_LINALG_MATRIX_RANK:
  set(op, at::redispatch::linalg_matrix_rank(ks, load<const at::Tensor &>()(op.args[0]), load<c10::optional<double>>()(op.args[1]), load<bool>()(op.args[2])));
  break;

case H_LINALG_MATRIX_RANK_OUT:
  init_update_in_place(op);
  at::redispatch::linalg_matrix_rank_outf(ks, load<const at::Tensor &>()(op.args[0]), load<c10::optional<double>>()(op.args[1]), load<bool>()(op.args[2]), load<at::Tensor &>()(op.args[3]));
  end_update_in_place(op);
  break;

case H__TEST_OPTIONAL_FLOATLIST:
  set(op, at::redispatch::_test_optional_floatlist(ks, load<const at::Tensor &>()(op.args[0]), load<c10::optional<at::ArrayRef<double>>>()(op.args[1])));
  break;

case H__TEST_STRING_DEFAULT:
  set(op, at::redispatch::_test_string_default(ks, load<const at::Tensor &>()(op.args[0]), load<c10::string_view>()(op.args[1]), load<c10::string_view>()(op.args[2])));
  break;

case H__TEST_AMBIGUOUS_DEFAULTS_B:
  set(op, at::redispatch::_test_ambiguous_defaults(ks, load<const at::Tensor &>()(op.args[0]), load<int64_t>()(op.args[1]), load<c10::string_view>()(op.args[2])));
  break;

case H_SEGMENT_REDUCE:
  set(op, at::redispatch::segment_reduce(ks, load<const at::Tensor &>()(op.args[0]), load<c10::string_view>()(op.args[1]), load<const c10::optional<at::Tensor> &>()(op.args[2]), load<const c10::optional<at::Tensor> &>()(op.args[3]), load<int64_t>()(op.args[4]), load<bool>()(op.args[5]), load<const c10::optional<at::Scalar> &>()(op.args[6])));
  break;

case H__SEGMENT_REDUCE_BACKWARD:
  set(op, at::redispatch::_segment_reduce_backward(ks, load<const at::Tensor &>()(op.args[0]), load<const at::Tensor &>()(op.args[1]), load<const at::Tensor &>()(op.args[2]), load<c10::string_view>()(op.args[3]), load<const c10::optional<at::Tensor> &>()(op.args[4])));
  break;

case H_PAD_SEQUENCE:
  set(op, at::redispatch::pad_sequence(ks, load<at::TensorList>()(op.args[0]), load<bool>()(op.args[1]), load<double>()(op.args[2])));
  break;

