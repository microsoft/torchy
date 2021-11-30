case H_EMPTY_STRIDED:
  results[i] = at::redispatch::empty_strided(ks, load<at::IntArrayRef>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<c10::optional<at::ScalarType>>()(op.args[2], load_state), load<c10::optional<at::Layout>>()(op.args[3], load_state), load<c10::optional<at::Device>>()(op.args[4], load_state), load<c10::optional<bool>>()(op.args[5], load_state));
  break;

case H_FULL_NAMES:
  results[i] = at::redispatch::full(ks, load<at::IntArrayRef>()(op.args[0], load_state), load<at::Scalar &>()(op.args[1], load_state), load<c10::optional<at::DimnameList>>()(op.args[2], load_state), load<c10::optional<at::ScalarType>>()(op.args[3], load_state), load<c10::optional<at::Layout>>()(op.args[4], load_state), load<c10::optional<at::Device>>()(op.args[5], load_state), load<c10::optional<bool>>()(op.args[6], load_state));
  break;

case H_FULL:
  results[i] = at::redispatch::full(ks, load<at::IntArrayRef>()(op.args[0], load_state), load<at::Scalar &>()(op.args[1], load_state), load<c10::optional<at::ScalarType>>()(op.args[2], load_state), load<c10::optional<at::Layout>>()(op.args[3], load_state), load<c10::optional<at::Device>>()(op.args[4], load_state), load<c10::optional<bool>>()(op.args[5], load_state));
  break;

case H_MKLDNN_LINEAR_BACKWARD_INPUT:
  results[i] = at::redispatch::mkldnn_linear_backward_input(ks, load<at::IntArrayRef>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state));
  break;

case H_MKLDNN_CONVOLUTION_BACKWARD_INPUT:
  results[i] = at::redispatch::mkldnn_convolution_backward_input(ks, load<at::IntArrayRef>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::IntArrayRef>()(op.args[3], load_state), load<at::IntArrayRef>()(op.args[4], load_state), load<at::IntArrayRef>()(op.args[5], load_state), load<int64_t>()(op.args[6], load_state), load<bool>()(op.args[7], load_state));
  break;

case H_MIOPEN_CONVOLUTION_BACKWARD_INPUT:
case H_MIOPEN_CONVOLUTION_BACKWARD_WEIGHT:
case H_MIOPEN_CONVOLUTION_TRANSPOSE_BACKWARD_WEIGHT:
case H_MIOPEN_DEPTHWISE_CONVOLUTION_BACKWARD_INPUT:
case H_MIOPEN_DEPTHWISE_CONVOLUTION_BACKWARD_WEIGHT:
  results[i] = redispatch_ptrs_0[op.id - H_MIOPEN_CONVOLUTION_BACKWARD_INPUT](ks, load<at::IntArrayRef>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::IntArrayRef>()(op.args[3], load_state), load<at::IntArrayRef>()(op.args[4], load_state), load<at::IntArrayRef>()(op.args[5], load_state), load<int64_t>()(op.args[6], load_state), load<bool>()(op.args[7], load_state), load<bool>()(op.args[8], load_state));
  break;

case H_CUDNN_CONVOLUTION_BACKWARD_INPUT:
case H_CUDNN_CONVOLUTION_BACKWARD_WEIGHT:
case H_CUDNN_CONVOLUTION_TRANSPOSE_BACKWARD_WEIGHT:
  results[i] = redispatch_ptrs_1[op.id - H_CUDNN_CONVOLUTION_BACKWARD_INPUT](ks, load<at::IntArrayRef>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::IntArrayRef>()(op.args[3], load_state), load<at::IntArrayRef>()(op.args[4], load_state), load<at::IntArrayRef>()(op.args[5], load_state), load<int64_t>()(op.args[6], load_state), load<bool>()(op.args[7], load_state), load<bool>()(op.args[8], load_state), load<bool>()(op.args[9], load_state));
  break;

case H__EMPTY_PER_CHANNEL_AFFINE_QUANTIZED:
  results[i] = at::redispatch::_empty_per_channel_affine_quantized(ks, load<at::IntArrayRef>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<int64_t>()(op.args[3], load_state), load<c10::optional<at::ScalarType>>()(op.args[4], load_state), load<c10::optional<at::Layout>>()(op.args[5], load_state), load<c10::optional<at::Device>>()(op.args[6], load_state), load<c10::optional<bool>>()(op.args[7], load_state), load<c10::optional<at::MemoryFormat>>()(op.args[8], load_state));
  break;

case H_EMPTY_QUANTIZED:
  results[i] = at::redispatch::empty_quantized(ks, load<at::IntArrayRef>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<c10::optional<at::ScalarType>>()(op.args[2], load_state), load<c10::optional<at::Layout>>()(op.args[3], load_state), load<c10::optional<at::Device>>()(op.args[4], load_state), load<c10::optional<bool>>()(op.args[5], load_state), load<c10::optional<at::MemoryFormat>>()(op.args[6], load_state));
  break;

case H_ONES_NAMES:
case H_RAND_NAMES:
case H_RANDN_NAMES:
case H_ZEROS_NAMES:
  results[i] = redispatch_ptrs_2[op.id - H_ONES_NAMES](ks, load<at::IntArrayRef>()(op.args[0], load_state), load<c10::optional<at::DimnameList>>()(op.args[1], load_state), load<c10::optional<at::ScalarType>>()(op.args[2], load_state), load<c10::optional<at::Layout>>()(op.args[3], load_state), load<c10::optional<at::Device>>()(op.args[4], load_state), load<c10::optional<bool>>()(op.args[5], load_state));
  break;

case H_EMPTY_NAMES:
  results[i] = at::redispatch::empty(ks, load<at::IntArrayRef>()(op.args[0], load_state), load<c10::optional<at::DimnameList>>()(op.args[1], load_state), load<c10::optional<at::ScalarType>>()(op.args[2], load_state), load<c10::optional<at::Layout>>()(op.args[3], load_state), load<c10::optional<at::Device>>()(op.args[4], load_state), load<c10::optional<bool>>()(op.args[5], load_state), load<c10::optional<at::MemoryFormat>>()(op.args[6], load_state));
  break;

case H_RAND_GENERATOR_WITH_NAMES:
case H_RANDN_GENERATOR_WITH_NAMES:
  {at::Tensor(*ptr)(DispatchKeySet, at::IntArrayRef, c10::optional<at::Generator>, c10::optional<at::DimnameList>, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<at::Device>, c10::optional<bool>) = at::redispatch::rand;
  if (op.id == H_RANDN_GENERATOR_WITH_NAMES) ptr = at::redispatch::randn;
  results[i] = ptr(ks, load<at::IntArrayRef>()(op.args[0], load_state), load<c10::optional<at::Generator>>()(op.args[1], load_state), load<c10::optional<at::DimnameList>>()(op.args[2], load_state), load<c10::optional<at::ScalarType>>()(op.args[3], load_state), load<c10::optional<at::Layout>>()(op.args[4], load_state), load<c10::optional<at::Device>>()(op.args[5], load_state), load<c10::optional<bool>>()(op.args[6], load_state));
  break;}

case H_RAND_GENERATOR:
case H_RANDN_GENERATOR:
  {at::Tensor(*ptr)(DispatchKeySet, at::IntArrayRef, c10::optional<at::Generator>, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<at::Device>, c10::optional<bool>) = at::redispatch::rand;
  if (op.id == H_RANDN_GENERATOR) ptr = at::redispatch::randn;
  results[i] = ptr(ks, load<at::IntArrayRef>()(op.args[0], load_state), load<c10::optional<at::Generator>>()(op.args[1], load_state), load<c10::optional<at::ScalarType>>()(op.args[2], load_state), load<c10::optional<at::Layout>>()(op.args[3], load_state), load<c10::optional<at::Device>>()(op.args[4], load_state), load<c10::optional<bool>>()(op.args[5], load_state));
  break;}

case H_ONES:
case H_RAND:
case H_RANDN:
case H_ZEROS:
case H_SPARSE_COO_TENSOR_SIZE:
  results[i] = redispatch_ptrs_3[op.id - H_ONES](ks, load<at::IntArrayRef>()(op.args[0], load_state), load<c10::optional<at::ScalarType>>()(op.args[1], load_state), load<c10::optional<at::Layout>>()(op.args[2], load_state), load<c10::optional<at::Device>>()(op.args[3], load_state), load<c10::optional<bool>>()(op.args[4], load_state));
  break;

case H_EMPTY_MEMORY_FORMAT:
  results[i] = at::redispatch::empty(ks, load<at::IntArrayRef>()(op.args[0], load_state), load<c10::optional<at::ScalarType>>()(op.args[1], load_state), load<c10::optional<at::Layout>>()(op.args[2], load_state), load<c10::optional<at::Device>>()(op.args[3], load_state), load<c10::optional<bool>>()(op.args[4], load_state), load<c10::optional<at::MemoryFormat>>()(op.args[5], load_state));
  break;

case H__EMPTY_AFFINE_QUANTIZED:
  results[i] = at::redispatch::_empty_affine_quantized(ks, load<at::IntArrayRef>()(op.args[0], load_state), load<c10::optional<at::ScalarType>>()(op.args[1], load_state), load<c10::optional<at::Layout>>()(op.args[2], load_state), load<c10::optional<at::Device>>()(op.args[3], load_state), load<c10::optional<bool>>()(op.args[4], load_state), load<double>()(op.args[5], load_state), load<int64_t>()(op.args[6], load_state), load<c10::optional<at::MemoryFormat>>()(op.args[7], load_state));
  break;

case H_ARANGE_START_STEP:
case H_RANGE_STEP:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Scalar &, const at::Scalar &, const at::Scalar &, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<at::Device>, c10::optional<bool>) = at::redispatch::arange;
  if (op.id == H_RANGE_STEP) ptr = at::redispatch::range;
  results[i] = ptr(ks, load<at::Scalar &>()(op.args[0], load_state), load<at::Scalar &>()(op.args[1], load_state), load<at::Scalar &>()(op.args[2], load_state), load<c10::optional<at::ScalarType>>()(op.args[3], load_state), load<c10::optional<at::Layout>>()(op.args[4], load_state), load<c10::optional<at::Device>>()(op.args[5], load_state), load<c10::optional<bool>>()(op.args[6], load_state));
  break;}

case H_ARANGE_START:
case H_RANGE:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Scalar &, const at::Scalar &, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<at::Device>, c10::optional<bool>) = at::redispatch::arange;
  if (op.id == H_RANGE) ptr = at::redispatch::range;
  results[i] = ptr(ks, load<at::Scalar &>()(op.args[0], load_state), load<at::Scalar &>()(op.args[1], load_state), load<c10::optional<at::ScalarType>>()(op.args[2], load_state), load<c10::optional<at::Layout>>()(op.args[3], load_state), load<c10::optional<at::Device>>()(op.args[4], load_state), load<c10::optional<bool>>()(op.args[5], load_state));
  break;}

case H_LINSPACE:
  results[i] = at::redispatch::linspace(ks, load<at::Scalar &>()(op.args[0], load_state), load<at::Scalar &>()(op.args[1], load_state), load<c10::optional<int64_t>>()(op.args[2], load_state), load<c10::optional<at::ScalarType>>()(op.args[3], load_state), load<c10::optional<at::Layout>>()(op.args[4], load_state), load<c10::optional<at::Device>>()(op.args[5], load_state), load<c10::optional<bool>>()(op.args[6], load_state));
  break;

case H_LOGSPACE:
  results[i] = at::redispatch::logspace(ks, load<at::Scalar &>()(op.args[0], load_state), load<at::Scalar &>()(op.args[1], load_state), load<c10::optional<int64_t>>()(op.args[2], load_state), load<double>()(op.args[3], load_state), load<c10::optional<at::ScalarType>>()(op.args[4], load_state), load<c10::optional<at::Layout>>()(op.args[5], load_state), load<c10::optional<at::Device>>()(op.args[6], load_state), load<c10::optional<bool>>()(op.args[7], load_state));
  break;

case H_XLOGY_SCALAR_SELF:
case H_BITWISE_LEFT_SHIFT_SCALAR_TENSOR:
case H_BITWISE_RIGHT_SHIFT_SCALAR_TENSOR:
case H_REMAINDER_SCALAR_TENSOR:
case H_POW_SCALAR:
case H_FLOAT_POWER_SCALAR:
case H_SPECIAL_XLOG1PY_SELF_SCALAR:
case H_SPECIAL_XLOGY_SELF_SCALAR:
case H_SPECIAL_ZETA_SELF_SCALAR:
  results[i] = redispatch_ptrs_4[op.id - H_XLOGY_SCALAR_SELF](ks, load<at::Scalar &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state));
  break;

case H_ISIN_SCALAR_TENSOR:
case H_BUCKETIZE_SCALAR:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Scalar &, const at::Tensor &, bool, bool) = at::redispatch::isin;
  if (op.id == H_BUCKETIZE_SCALAR) ptr = at::redispatch::bucketize;
  results[i] = ptr(ks, load<at::Scalar &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<bool>()(op.args[2], load_state), load<bool>()(op.args[3], load_state));
  break;}

case H_ARANGE:
case H_SCALAR_TENSOR:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Scalar &, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<at::Device>, c10::optional<bool>) = at::redispatch::arange;
  if (op.id == H_SCALAR_TENSOR) ptr = at::redispatch::scalar_tensor;
  results[i] = ptr(ks, load<at::Scalar &>()(op.args[0], load_state), load<c10::optional<at::ScalarType>>()(op.args[1], load_state), load<c10::optional<at::Layout>>()(op.args[2], load_state), load<c10::optional<at::Device>>()(op.args[3], load_state), load<c10::optional<bool>>()(op.args[4], load_state));
  break;}

case H_DATA:
case H__SHAPE_AS_TENSOR:
case H_ABS:
case H_ABSOLUTE:
case H_ANGLE:
case H_VIEW_AS_REAL:
case H_VIEW_AS_COMPLEX:
case H_SGN:
case H_REAL:
case H_IMAG:
case H__CONJ:
case H_CONJ:
case H__CONJ_PHYSICAL:
case H_CONJ_PHYSICAL:
case H_RESOLVE_CONJ:
case H_RESOLVE_NEG:
case H__NEG_VIEW:
case H_ACOS:
case H_ARCCOS:
case H_ACOSH:
case H_ARCCOSH:
case H_ASINH:
case H_ARCSINH:
case H_ATANH:
case H_ARCTANH:
case H_ASIN:
case H_ARCSIN:
case H_ATAN:
case H_ARCTAN:
case H_ATLEAST_1D:
case H_ATLEAST_2D:
case H_ATLEAST_3D:
case H_BITWISE_NOT:
case H_LOGICAL_NOT:
case H_CEIL:
case H_COS:
case H_COSH:
case H_CORRCOEF:
case H_ERF:
case H_ERFC:
case H_EXP:
case H_EXP2:
case H_EXPM1:
case H_FLOOR:
case H_FRAC:
case H_INVERSE:
case H__INVERSE_HELPER:
case H_ISNAN:
case H_ISREAL:
case H_FBGEMM_PACK_GEMM_MATRIX_FP16:
case H_FBGEMM_PACK_QUANTIZED_MATRIX:
case H_LOG:
case H_LOG10:
case H_LOG1P:
case H_LOG2:
case H_LOGDET:
case H_MATRIX_EXP:
case H_MEDIAN:
case H_NANMEDIAN:
case H_MIOPEN_CONVOLUTION_BACKWARD_BIAS:
case H_NUMPY_T:
case H_MATRIX_H:
case H_MT:
case H_MH:
case H_ADJOINT:
case H_RAD2DEG:
case H_DEG2RAD:
case H_RAVEL:
case H_RECIPROCAL:
case H_NEG:
case H_NEGATIVE:
case H_ROUND:
case H_RELU:
case H_RELU6:
case H_GELU:
case H_RSQRT:
case H_SELU:
case H_SILU:
case H_MISH:
case H_SIGMOID:
case H_SIN:
case H_SINC:
case H_SINH:
case H_DETACH:
case H_SQUEEZE:
case H_SQRT:
case H_SQUARE:
case H_T:
case H_TAN:
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
case H_ARGWHERE:
case H_LGAMMA:
case H_DIGAMMA:
case H_ERFINV:
case H_I0:
case H_SIGN:
case H_SIGNBIT:
case H_MIN:
case H_MAX:
case H_MSORT:
case H_ALL:
case H_ANY:
case H_ALIAS:
case H__TORCH_CUDA_CU_LINKER_SYMBOL_OP:
case H_HARDSIGMOID:
case H_HARDSWISH:
case H_LOG_SIGMOID:
case H_ISFINITE:
case H_ISINF:
case H_ISPOSINF:
case H_ISNEGINF:
case H_SPECIAL_ENTR:
case H_SPECIAL_NDTRI:
case H_SPECIAL_EXPM1:
case H_SPECIAL_EXP2:
case H_SPECIAL_PSI:
case H_SPECIAL_DIGAMMA:
case H_SPECIAL_GAMMALN:
case H_SPECIAL_ERF:
case H_SPECIAL_ERFC:
case H_SPECIAL_ERFCX:
case H_SPECIAL_ERFINV:
case H_SPECIAL_NDTR:
case H_SPECIAL_I0:
case H_SPECIAL_I0E:
case H_SPECIAL_I1:
case H_SPECIAL_I1E:
case H_SPECIAL_EXPIT:
case H_SPECIAL_SINC:
case H_SPECIAL_ROUND:
case H_SPECIAL_LOG1P:
case H_LINALG_DET:
case H_DET:
case H_LINALG_MATRIX_EXP:
case H_LINALG_EIGVALS:
case H_LINALG_INV:
case H_LINALG_SVDVALS:
case H__TEST_WARN_IN_AUTOGRAD:
  results[i] = redispatch_ptrs_5[op.id - H_DATA](ks, load<at::Tensor &>()(op.args[0], load_state));
  break;

case H_TO_DEVICE:
  results[i] = at::redispatch::to(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Device>()(op.args[1], load_state), load<at::ScalarType>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<bool>()(op.args[4], load_state), load<c10::optional<at::MemoryFormat>>()(op.args[5], load_state));
  break;

case H_LOGCUMSUMEXP_DIMNAME:
case H_SQUEEZE_DIMNAME:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, at::Dimname) = at::redispatch::logcumsumexp;
  if (op.id == H_SQUEEZE_DIMNAME) ptr = at::redispatch::squeeze;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Dimname>()(op.args[1], load_state));
  break;}

case H_TRANSPOSE_DIMNAME:
  results[i] = at::redispatch::transpose(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Dimname>()(op.args[1], load_state), load<at::Dimname>()(op.args[2], load_state));
  break;

case H_FLATTEN_USING_NAMES:
  results[i] = at::redispatch::flatten(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Dimname>()(op.args[1], load_state), load<at::Dimname>()(op.args[2], load_state), load<at::Dimname>()(op.args[3], load_state));
  break;

case H_DIAGONAL_DIMNAME:
  results[i] = at::redispatch::diagonal(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Dimname>()(op.args[1], load_state), load<at::Dimname>()(op.args[2], load_state), load<at::Dimname>()(op.args[3], load_state), load<int64_t>()(op.args[4], load_state));
  break;

case H_UNFLATTEN_DIMNAME:
  results[i] = at::redispatch::unflatten(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Dimname>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<at::DimnameList>()(op.args[3], load_state));
  break;

case H_INDEX_SELECT_DIMNAME:
  results[i] = at::redispatch::index_select(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Dimname>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state));
  break;

case H_INDEX_FILL_DIMNAME_SCALAR:
case H_SCATTER_DIMNAME_VALUE:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, at::Dimname, const at::Tensor &, const at::Scalar &) = at::redispatch::index_fill;
  if (op.id == H_SCATTER_DIMNAME_VALUE) ptr = at::redispatch::scatter;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Dimname>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::Scalar &>()(op.args[3], load_state));
  break;}

case H_INDEX_COPY_DIMNAME:
case H_INDEX_FILL_DIMNAME_TENSOR:
case H_SCATTER_DIMNAME_SRC:
case H_SCATTER_ADD_DIMNAME:
  results[i] = redispatch_ptrs_6[op.id - H_INDEX_COPY_DIMNAME](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Dimname>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state));
  break;

case H_INDEX_ADD_DIMNAME:
  results[i] = at::redispatch::index_add(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Dimname>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state), load<at::Scalar &>()(op.args[4], load_state));
  break;

case H_GATHER_DIMNAME:
  results[i] = at::redispatch::gather(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Dimname>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<bool>()(op.args[3], load_state));
  break;

case H_ALL_DIMNAME:
case H_ANY_DIMNAME:
case H_ARGSORT_DIMNAME:
  results[i] = redispatch_ptrs_7[op.id - H_ALL_DIMNAME](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Dimname>()(op.args[1], load_state), load<bool>()(op.args[2], load_state));
  break;

case H_PROD_DIM_DIMNAME:
  results[i] = at::redispatch::prod(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Dimname>()(op.args[1], load_state), load<bool>()(op.args[2], load_state), load<c10::optional<at::ScalarType>>()(op.args[3], load_state));
  break;

case H_CUMPROD_DIMNAME:
case H_CUMSUM_DIMNAME:
case H_LOG_SOFTMAX_DIMNAME:
case H_SOFTMAX_DIMNAME:
case H__SPARSE_SOFTMAX_DIMNAME:
case H__SPARSE_LOG_SOFTMAX_DIMNAME:
  results[i] = redispatch_ptrs_8[op.id - H_CUMPROD_DIMNAME](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Dimname>()(op.args[1], load_state), load<c10::optional<at::ScalarType>>()(op.args[2], load_state));
  break;

case H_SELECT_DIMNAME:
  results[i] = at::redispatch::select(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Dimname>()(op.args[1], load_state), load<int64_t>()(op.args[2], load_state));
  break;

case H_ALIGN_TO:
case H_REFINE_NAMES:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, at::DimnameList) = at::redispatch::align_to;
  if (op.id == H_REFINE_NAMES) ptr = at::redispatch::refine_names;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::DimnameList>()(op.args[1], load_state));
  break;}

case H_FLATTEN_DIMNAMELIST:
  results[i] = at::redispatch::flatten(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::DimnameList>()(op.args[1], load_state), load<at::Dimname>()(op.args[2], load_state));
  break;

case H_LOGSUMEXP_NAMES:
  results[i] = at::redispatch::logsumexp(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::DimnameList>()(op.args[1], load_state), load<bool>()(op.args[2], load_state));
  break;

case H_STD_NAMES_DIM:
case H_VAR_NAMES_DIM:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, at::DimnameList, bool, bool) = at::redispatch::std;
  if (op.id == H_VAR_NAMES_DIM) ptr = at::redispatch::var;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::DimnameList>()(op.args[1], load_state), load<bool>()(op.args[2], load_state), load<bool>()(op.args[3], load_state));
  break;}

case H_MEAN_NAMES_DIM:
case H_SUM_DIM_DIMNAMELIST:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, at::DimnameList, bool, c10::optional<at::ScalarType>) = at::redispatch::mean;
  if (op.id == H_SUM_DIM_DIMNAMELIST) ptr = at::redispatch::sum;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::DimnameList>()(op.args[1], load_state), load<bool>()(op.args[2], load_state), load<c10::optional<at::ScalarType>>()(op.args[3], load_state));
  break;}

case H_STD_CORRECTION_NAMES:
case H_VAR_CORRECTION_NAMES:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, at::DimnameList, c10::optional<int64_t>, bool) = at::redispatch::std;
  if (op.id == H_VAR_CORRECTION_NAMES) ptr = at::redispatch::var;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::DimnameList>()(op.args[1], load_state), load<c10::optional<int64_t>>()(op.args[2], load_state), load<bool>()(op.args[3], load_state));
  break;}

case H_ALIGN_TO_ELLIPSIS_IDX:
  results[i] = at::redispatch::align_to(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::DimnameList>()(op.args[1], load_state), load<int64_t>()(op.args[2], load_state));
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
case H_REFLECTION_PAD3D:
case H_REPLICATION_PAD1D:
case H_REPLICATION_PAD2D:
case H_REPLICATION_PAD3D:
  results[i] = redispatch_ptrs_9[op.id - H_ADAPTIVE_AVG_POOL1D](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state));
  break;

case H_MOVEDIM_INTLIST:
case H_MOVEAXIS_INTLIST:
case H__RESHAPE_ALIAS:
case H_ROLL:
  results[i] = redispatch_ptrs_10[op.id - H_MOVEDIM_INTLIST](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state));
  break;

case H_COL2IM_BACKWARD:
case H_IM2COL:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef) = at::redispatch::col2im_backward;
  if (op.id == H_IM2COL) ptr = at::redispatch::im2col;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<at::IntArrayRef>()(op.args[3], load_state), load<at::IntArrayRef>()(op.args[4], load_state));
  break;}

case H_COL2IM:
case H_IM2COL_BACKWARD:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef) = at::redispatch::col2im;
  if (op.id == H_IM2COL_BACKWARD) ptr = at::redispatch::im2col_backward;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<at::IntArrayRef>()(op.args[3], load_state), load<at::IntArrayRef>()(op.args[4], load_state), load<at::IntArrayRef>()(op.args[5], load_state));
  break;}

case H_MAX_POOL1D:
case H_MAX_POOL2D:
case H_MKLDNN_MAX_POOL2D:
case H_MKLDNN_MAX_POOL3D:
case H_QUANTIZED_MAX_POOL1D:
case H_QUANTIZED_MAX_POOL2D:
case H_MAX_POOL3D:
  results[i] = redispatch_ptrs_11[op.id - H_MAX_POOL1D](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<at::IntArrayRef>()(op.args[3], load_state), load<at::IntArrayRef>()(op.args[4], load_state), load<bool>()(op.args[5], load_state));
  break;

case H_AVG_POOL1D:
  results[i] = at::redispatch::avg_pool1d(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<at::IntArrayRef>()(op.args[3], load_state), load<bool>()(op.args[4], load_state), load<bool>()(op.args[5], load_state));
  break;

case H_AVG_POOL2D:
case H_AVG_POOL3D:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, bool, bool, c10::optional<int64_t>) = at::redispatch::avg_pool2d;
  if (op.id == H_AVG_POOL3D) ptr = at::redispatch::avg_pool3d;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<at::IntArrayRef>()(op.args[3], load_state), load<bool>()(op.args[4], load_state), load<bool>()(op.args[5], load_state), load<c10::optional<int64_t>>()(op.args[6], load_state));
  break;}

case H_MKLDNN_REORDER_CONV2D_WEIGHT:
case H_MKLDNN_REORDER_CONV3D_WEIGHT:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t) = at::redispatch::mkldnn_reorder_conv2d_weight;
  if (op.id == H_MKLDNN_REORDER_CONV3D_WEIGHT) ptr = at::redispatch::mkldnn_reorder_conv3d_weight;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<at::IntArrayRef>()(op.args[3], load_state), load<int64_t>()(op.args[4], load_state));
  break;}

case H_UPSAMPLE_LINEAR1D_BACKWARD:
  results[i] = at::redispatch::upsample_linear1d_backward(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<c10::optional<double>>()(op.args[4], load_state));
  break;

case H_UPSAMPLE_BILINEAR2D_BACKWARD:
case H__UPSAMPLE_BILINEAR2D_AA_BACKWARD:
case H_UPSAMPLE_BICUBIC2D_BACKWARD:
  results[i] = redispatch_ptrs_12[op.id - H_UPSAMPLE_BILINEAR2D_BACKWARD](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<c10::optional<double>>()(op.args[4], load_state), load<c10::optional<double>>()(op.args[5], load_state));
  break;

case H_UPSAMPLE_TRILINEAR3D_BACKWARD:
  results[i] = at::redispatch::upsample_trilinear3d_backward(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<c10::optional<double>>()(op.args[4], load_state), load<c10::optional<double>>()(op.args[5], load_state), load<c10::optional<double>>()(op.args[6], load_state));
  break;

case H_NEW_EMPTY_STRIDED:
  results[i] = at::redispatch::new_empty_strided(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<c10::optional<at::ScalarType>>()(op.args[3], load_state), load<c10::optional<at::Layout>>()(op.args[4], load_state), load<c10::optional<at::Device>>()(op.args[5], load_state), load<c10::optional<bool>>()(op.args[6], load_state));
  break;

case H_UPSAMPLE_NEAREST1D_BACKWARD:
case H__UPSAMPLE_NEAREST_EXACT1D_BACKWARD:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, c10::optional<double>) = at::redispatch::upsample_nearest1d_backward;
  if (op.id == H__UPSAMPLE_NEAREST_EXACT1D_BACKWARD) ptr = at::redispatch::_upsample_nearest_exact1d_backward;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<c10::optional<double>>()(op.args[3], load_state));
  break;}

case H_UPSAMPLE_NEAREST2D_BACKWARD:
case H__UPSAMPLE_NEAREST_EXACT2D_BACKWARD:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, c10::optional<double>, c10::optional<double>) = at::redispatch::upsample_nearest2d_backward;
  if (op.id == H__UPSAMPLE_NEAREST_EXACT2D_BACKWARD) ptr = at::redispatch::_upsample_nearest_exact2d_backward;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<c10::optional<double>>()(op.args[3], load_state), load<c10::optional<double>>()(op.args[4], load_state));
  break;}

case H_UPSAMPLE_NEAREST3D_BACKWARD:
case H__UPSAMPLE_NEAREST_EXACT3D_BACKWARD:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, c10::optional<double>, c10::optional<double>, c10::optional<double>) = at::redispatch::upsample_nearest3d_backward;
  if (op.id == H__UPSAMPLE_NEAREST_EXACT3D_BACKWARD) ptr = at::redispatch::_upsample_nearest_exact3d_backward;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<c10::optional<double>>()(op.args[3], load_state), load<c10::optional<double>>()(op.args[4], load_state), load<c10::optional<double>>()(op.args[5], load_state));
  break;}

case H_AS_STRIDED:
  results[i] = at::redispatch::as_strided(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<c10::optional<int64_t>>()(op.args[3], load_state));
  break;

case H_CONSTANT_PAD_ND:
  results[i] = at::redispatch::constant_pad_nd(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<at::Scalar &>()(op.args[2], load_state));
  break;

case H_NEW_FULL:
  results[i] = at::redispatch::new_full(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<at::Scalar &>()(op.args[2], load_state), load<c10::optional<at::ScalarType>>()(op.args[3], load_state), load<c10::optional<at::Layout>>()(op.args[4], load_state), load<c10::optional<at::Device>>()(op.args[5], load_state), load<c10::optional<bool>>()(op.args[6], load_state));
  break;

case H__SPARSE_SUM_DIM_DTYPE:
  results[i] = at::redispatch::_sparse_sum(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<at::ScalarType>()(op.args[2], load_state));
  break;

case H__NNPACK_SPATIAL_CONVOLUTION_BACKWARD_WEIGHT:
  results[i] = at::redispatch::_nnpack_spatial_convolution_backward_weight(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::IntArrayRef>()(op.args[3], load_state));
  break;

case H__PACK_PADDED_SEQUENCE_BACKWARD:
  results[i] = at::redispatch::_pack_padded_sequence_backward(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<bool>()(op.args[3], load_state));
  break;

case H_AFFINE_GRID_GENERATOR:
case H_AFFINE_GRID_GENERATOR_BACKWARD:
case H_EXPAND:
case H_LOGSUMEXP:
case H_AMAX:
case H_AMIN:
case H_FROBENIUS_NORM_DIM:
case H_NUCLEAR_NORM_DIM:
case H_SPECIAL_LOGSUMEXP:
  results[i] = redispatch_ptrs_13[op.id - H_AFFINE_GRID_GENERATOR](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<bool>()(op.args[2], load_state));
  break;

case H_STD_DIM:
case H_VAR_DIM:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, at::IntArrayRef, bool, bool) = at::redispatch::std;
  if (op.id == H_VAR_DIM) ptr = at::redispatch::var;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<bool>()(op.args[2], load_state), load<bool>()(op.args[3], load_state));
  break;}

case H_MEAN_DIM:
case H_NANMEAN:
case H_SUM_DIM_INTLIST:
case H_NANSUM_DIM_INTLIST:
  results[i] = redispatch_ptrs_14[op.id - H_MEAN_DIM](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<bool>()(op.args[2], load_state), load<c10::optional<at::ScalarType>>()(op.args[3], load_state));
  break;

case H_UPSAMPLE_LINEAR1D:
  results[i] = at::redispatch::upsample_linear1d(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<bool>()(op.args[2], load_state), load<c10::optional<double>>()(op.args[3], load_state));
  break;

case H_UPSAMPLE_BILINEAR2D:
case H__UPSAMPLE_BILINEAR2D_AA:
case H_UPSAMPLE_BICUBIC2D:
  results[i] = redispatch_ptrs_15[op.id - H_UPSAMPLE_BILINEAR2D](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<bool>()(op.args[2], load_state), load<c10::optional<double>>()(op.args[3], load_state), load<c10::optional<double>>()(op.args[4], load_state));
  break;

case H_UPSAMPLE_TRILINEAR3D:
  results[i] = at::redispatch::upsample_trilinear3d(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<bool>()(op.args[2], load_state), load<c10::optional<double>>()(op.args[3], load_state), load<c10::optional<double>>()(op.args[4], load_state), load<c10::optional<double>>()(op.args[5], load_state));
  break;

case H__HISTOGRAMDD_FROM_BIN_CTS:
  results[i] = at::redispatch::_histogramdd_from_bin_cts(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<c10::optional<at::ArrayRef<double>>>()(op.args[2], load_state), load<c10::optional<at::Tensor> &>()(op.args[3], load_state), load<bool>()(op.args[4], load_state));
  break;

case H_NEW_EMPTY:
case H_NEW_ZEROS:
case H_NEW_ONES:
  results[i] = redispatch_ptrs_16[op.id - H_NEW_EMPTY](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<c10::optional<at::ScalarType>>()(op.args[2], load_state), load<c10::optional<at::Layout>>()(op.args[3], load_state), load<c10::optional<at::Device>>()(op.args[4], load_state), load<c10::optional<bool>>()(op.args[5], load_state));
  break;

case H_LAYER_NORM:
  results[i] = at::redispatch::layer_norm(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<c10::optional<at::Tensor> &>()(op.args[2], load_state), load<c10::optional<at::Tensor> &>()(op.args[3], load_state), load<double>()(op.args[4], load_state), load<bool>()(op.args[5], load_state));
  break;

case H_UPSAMPLE_NEAREST1D:
case H__UPSAMPLE_NEAREST_EXACT1D:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, at::IntArrayRef, c10::optional<double>) = at::redispatch::upsample_nearest1d;
  if (op.id == H__UPSAMPLE_NEAREST_EXACT1D) ptr = at::redispatch::_upsample_nearest_exact1d;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<c10::optional<double>>()(op.args[2], load_state));
  break;}

case H_UPSAMPLE_NEAREST2D:
case H__UPSAMPLE_NEAREST_EXACT2D:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, at::IntArrayRef, c10::optional<double>, c10::optional<double>) = at::redispatch::upsample_nearest2d;
  if (op.id == H__UPSAMPLE_NEAREST_EXACT2D) ptr = at::redispatch::_upsample_nearest_exact2d;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<c10::optional<double>>()(op.args[2], load_state), load<c10::optional<double>>()(op.args[3], load_state));
  break;}

case H_UPSAMPLE_NEAREST3D:
case H__UPSAMPLE_NEAREST_EXACT3D:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, at::IntArrayRef, c10::optional<double>, c10::optional<double>, c10::optional<double>) = at::redispatch::upsample_nearest3d;
  if (op.id == H__UPSAMPLE_NEAREST_EXACT3D) ptr = at::redispatch::_upsample_nearest_exact3d;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<c10::optional<double>>()(op.args[2], load_state), load<c10::optional<double>>()(op.args[3], load_state), load<c10::optional<double>>()(op.args[4], load_state));
  break;}

case H_DIAG_BACKWARD:
  results[i] = at::redispatch::diag_backward(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<int64_t>()(op.args[2], load_state));
  break;

case H_INDEX_SELECT_BACKWARD:
  results[i] = at::redispatch::index_select_backward(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<int64_t>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state));
  break;

case H__FFT_R2C:
case H__FFT_C2C:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, at::IntArrayRef, int64_t, bool) = at::redispatch::_fft_r2c;
  if (op.id == H__FFT_C2C) ptr = at::redispatch::_fft_c2c;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<int64_t>()(op.args[2], load_state), load<bool>()(op.args[3], load_state));
  break;}

case H__FFT_C2R:
case H_SELECT_BACKWARD:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, at::IntArrayRef, int64_t, int64_t) = at::redispatch::_fft_c2r;
  if (op.id == H_SELECT_BACKWARD) ptr = at::redispatch::select_backward;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<int64_t>()(op.args[2], load_state), load<int64_t>()(op.args[3], load_state));
  break;}

case H_DIAGONAL_BACKWARD:
case H_UNFOLD_BACKWARD:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, at::IntArrayRef, int64_t, int64_t, int64_t) = at::redispatch::diagonal_backward;
  if (op.id == H_UNFOLD_BACKWARD) ptr = at::redispatch::unfold_backward;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<int64_t>()(op.args[2], load_state), load<int64_t>()(op.args[3], load_state), load<int64_t>()(op.args[4], load_state));
  break;}

case H_SLICE_BACKWARD:
  results[i] = at::redispatch::slice_backward(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<int64_t>()(op.args[2], load_state), load<int64_t>()(op.args[3], load_state), load<int64_t>()(op.args[4], load_state), load<int64_t>()(op.args[5], load_state));
  break;

case H_CONTIGUOUS:
  results[i] = at::redispatch::__dispatch_contiguous(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::MemoryFormat>()(op.args[1], load_state));
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
case H_HARDSHRINK:
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
case H_BITWISE_LEFT_SHIFT_TENSOR_SCALAR:
case H___RSHIFT___SCALAR:
case H_BITWISE_RIGHT_SHIFT_TENSOR_SCALAR:
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
case H_SOFTSHRINK:
case H_SPECIAL_XLOG1PY_OTHER_SCALAR:
case H_SPECIAL_XLOGY_OTHER_SCALAR:
case H_SPECIAL_ZETA_OTHER_SCALAR:
  results[i] = redispatch_ptrs_17[op.id - H_COPYSIGN_SCALAR](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Scalar &>()(op.args[1], load_state));
  break;

case H_LINALG_MATRIX_NORM:
  results[i] = at::redispatch::linalg_matrix_norm(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Scalar &>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<c10::optional<at::ScalarType>>()(op.args[4], load_state));
  break;

case H__ADD_RELU_SCALAR:
case H_ADD_SCALAR:
case H_THRESHOLD:
case H_WHERE_SCALAR:
case H_SUB_SCALAR:
case H_SUBTRACT_SCALAR:
case H_RSUB_SCALAR:
case H_HARDTANH:
case H_SOFTPLUS:
  results[i] = redispatch_ptrs_18[op.id - H__ADD_RELU_SCALAR](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Scalar &>()(op.args[1], load_state), load<at::Scalar &>()(op.args[2], load_state));
  break;

case H_ELU:
  results[i] = at::redispatch::elu(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Scalar &>()(op.args[1], load_state), load<at::Scalar &>()(op.args[2], load_state), load<at::Scalar &>()(op.args[3], load_state));
  break;

case H_ELU_BACKWARD:
  results[i] = at::redispatch::elu_backward(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Scalar &>()(op.args[1], load_state), load<at::Scalar &>()(op.args[2], load_state), load<at::Scalar &>()(op.args[3], load_state), load<bool>()(op.args[4], load_state), load<at::Tensor &>()(op.args[5], load_state));
  break;

case H_RRELU:
  results[i] = at::redispatch::rrelu(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Scalar &>()(op.args[1], load_state), load<at::Scalar &>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<c10::optional<at::Generator>>()(op.args[4], load_state));
  break;

case H_WHERE_SCALARSELF:
  results[i] = at::redispatch::where(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Scalar &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state));
  break;

case H_ISIN_TENSOR_SCALAR:
  results[i] = at::redispatch::isin(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Scalar &>()(op.args[1], load_state), load<bool>()(op.args[2], load_state), load<bool>()(op.args[3], load_state));
  break;

case H_SEARCHSORTED_SCALAR:
  results[i] = at::redispatch::searchsorted(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Scalar &>()(op.args[1], load_state), load<bool>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<c10::optional<c10::string_view>>()(op.args[4], load_state), load<c10::optional<at::Tensor> &>()(op.args[5], load_state));
  break;

case H_LINALG_VECTOR_NORM:
  results[i] = at::redispatch::linalg_vector_norm(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Scalar &>()(op.args[1], load_state), load<c10::optional<at::IntArrayRef>>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<c10::optional<at::ScalarType>>()(op.args[4], load_state));
  break;

case H_FULL_LIKE:
  results[i] = at::redispatch::full_like(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Scalar &>()(op.args[1], load_state), load<c10::optional<at::ScalarType>>()(op.args[2], load_state), load<c10::optional<at::Layout>>()(op.args[3], load_state), load<c10::optional<at::Device>>()(op.args[4], load_state), load<c10::optional<bool>>()(op.args[5], load_state), load<c10::optional<at::MemoryFormat>>()(op.args[6], load_state));
  break;

case H_DIV_SCALAR_MODE:
case H_DIVIDE_SCALAR_MODE:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Scalar &, c10::optional<c10::string_view>) = at::redispatch::div;
  if (op.id == H_DIVIDE_SCALAR_MODE) ptr = at::redispatch::divide;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Scalar &>()(op.args[1], load_state), load<c10::optional<c10::string_view>>()(op.args[2], load_state));
  break;}

case H_CUMULATIVE_TRAPEZOID_DX:
case H_TRAPEZOID_DX:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Scalar &, int64_t) = at::redispatch::cumulative_trapezoid;
  if (op.id == H_TRAPEZOID_DX) ptr = at::redispatch::trapezoid;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Scalar &>()(op.args[1], load_state), load<int64_t>()(op.args[2], load_state));
  break;}

case H_RENORM:
  results[i] = at::redispatch::renorm(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Scalar &>()(op.args[1], load_state), load<int64_t>()(op.args[2], load_state), load<at::Scalar &>()(op.args[3], load_state));
  break;

case H__SPARSE_SUM_DTYPE:
case H_VIEW_DTYPE:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, at::ScalarType) = at::redispatch::_sparse_sum;
  if (op.id == H_VIEW_DTYPE) ptr = at::redispatch::view;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::ScalarType>()(op.args[1], load_state));
  break;}

case H_QUANTIZE_PER_TENSOR_DYNAMIC:
  results[i] = at::redispatch::quantize_per_tensor_dynamic(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::ScalarType>()(op.args[1], load_state), load<bool>()(op.args[2], load_state));
  break;

case H_TO_DTYPE:
  results[i] = at::redispatch::to(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::ScalarType>()(op.args[1], load_state), load<bool>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<c10::optional<at::MemoryFormat>>()(op.args[4], load_state));
  break;

case H_ALIGN_AS:
case H__RESHAPE_FROM_TENSOR:
case H_COPYSIGN_TENSOR:
case H_LOGICAL_XOR:
case H_LOGICAL_AND:
case H_LOGICAL_OR:
case H_BMM:
case H_CLAMP_MAX_TENSOR:
case H_CLAMP_MIN_TENSOR:
case H_COMPLEX:
case H_POLAR:
case H__COPY_FROM_AND_RESIZE:
case H_CUDNN_GRID_SAMPLER:
case H_DIV_TENSOR:
case H_DIVIDE_TENSOR:
case H_TRUE_DIVIDE_TENSOR:
case H_DOT:
case H_VDOT:
case H_EXPAND_AS:
case H_FLOOR_DIVIDE:
case H_GCD:
case H_LCM:
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
case H_GELU_BACKWARD:
case H_INFINITELY_DIFFERENTIABLE_GELU_BACKWARD:
case H_SILU_BACKWARD:
case H_MISH_BACKWARD:
case H_SMM:
case H_TYPE_AS:
case H_VIEW_AS:
case H__STANDARD_GAMMA_GRAD:
case H_HEAVISIDE:
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
case H_BITWISE_LEFT_SHIFT_TENSOR:
case H___RSHIFT___TENSOR:
case H_BITWISE_RIGHT_SHIFT_TENSOR:
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
case H_ATAN2:
case H_ARCTAN2:
case H_FMOD_TENSOR:
case H_HYPOT:
case H_IGAMMA:
case H_IGAMMAC:
case H_NEXTAFTER:
case H_REMAINDER_TENSOR:
case H_FMIN:
case H_FMAX:
case H_MAXIMUM:
case H_MAX_OTHER:
case H_MINIMUM:
case H_MIN_OTHER:
case H_POW_TENSOR_TENSOR:
case H_FLOAT_POWER_TENSOR_TENSOR:
case H_HARDSIGMOID_BACKWARD:
case H_HARDSWISH_BACKWARD:
case H_MKLDNN_ADAPTIVE_AVG_POOL2D_BACKWARD:
case H__ADAPTIVE_AVG_POOL2D_BACKWARD:
case H__ADAPTIVE_AVG_POOL3D_BACKWARD:
case H_SIGMOID_BACKWARD:
case H_TANH_BACKWARD:
case H_SPECIAL_XLOG1PY:
case H_SPECIAL_XLOGY:
case H_SPECIAL_ZETA:
case H_SPECIAL_GAMMAINC:
case H_SPECIAL_GAMMAINCC:
case H_LINALG_MATMUL:
case H_LINALG_HOUSEHOLDER_PRODUCT:
case H_INNER:
case H_OUTER:
case H_GER:
case H_LINALG_SOLVE:
  results[i] = redispatch_ptrs_19[op.id - H_ALIGN_AS](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state));
  break;

case H__SPARSE_SUM_BACKWARD:
case H_MAX_UNPOOL2D:
case H_REFLECTION_PAD1D_BACKWARD:
case H_REFLECTION_PAD2D_BACKWARD:
case H_REFLECTION_PAD3D_BACKWARD:
case H_REPLICATION_PAD1D_BACKWARD:
case H_REPLICATION_PAD2D_BACKWARD:
case H_REPLICATION_PAD3D_BACKWARD:
  results[i] = redispatch_ptrs_20[op.id - H__SPARSE_SUM_BACKWARD](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state));
  break;

case H_TENSORDOT:
  results[i] = at::redispatch::tensordot(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<at::IntArrayRef>()(op.args[3], load_state));
  break;

case H_MAX_UNPOOL3D:
  results[i] = at::redispatch::max_unpool3d(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<at::IntArrayRef>()(op.args[3], load_state), load<at::IntArrayRef>()(op.args[4], load_state));
  break;

case H_MAX_POOL2D_WITH_INDICES_BACKWARD:
case H_MAX_POOL3D_WITH_INDICES_BACKWARD:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, bool, const at::Tensor &) = at::redispatch::max_pool2d_with_indices_backward;
  if (op.id == H_MAX_POOL3D_WITH_INDICES_BACKWARD) ptr = at::redispatch::max_pool3d_with_indices_backward;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<at::IntArrayRef>()(op.args[3], load_state), load<at::IntArrayRef>()(op.args[4], load_state), load<at::IntArrayRef>()(op.args[5], load_state), load<bool>()(op.args[6], load_state), load<at::Tensor &>()(op.args[7], load_state));
  break;}

case H_CUDNN_CONVOLUTION_TRANSPOSE_DEPRECATED2:
  results[i] = at::redispatch::cudnn_convolution_transpose(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<at::IntArrayRef>()(op.args[3], load_state), load<at::IntArrayRef>()(op.args[4], load_state), load<at::IntArrayRef>()(op.args[5], load_state), load<int64_t>()(op.args[6], load_state), load<bool>()(op.args[7], load_state), load<bool>()(op.args[8], load_state));
  break;

case H_CUDNN_CONVOLUTION_TRANSPOSE:
  results[i] = at::redispatch::cudnn_convolution_transpose(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<at::IntArrayRef>()(op.args[3], load_state), load<at::IntArrayRef>()(op.args[4], load_state), load<at::IntArrayRef>()(op.args[5], load_state), load<int64_t>()(op.args[6], load_state), load<bool>()(op.args[7], load_state), load<bool>()(op.args[8], load_state), load<bool>()(op.args[9], load_state));
  break;

case H_AVG_POOL2D_BACKWARD:
case H_AVG_POOL3D_BACKWARD:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, bool, bool, c10::optional<int64_t>) = at::redispatch::avg_pool2d_backward;
  if (op.id == H_AVG_POOL3D_BACKWARD) ptr = at::redispatch::avg_pool3d_backward;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<at::IntArrayRef>()(op.args[3], load_state), load<at::IntArrayRef>()(op.args[4], load_state), load<bool>()(op.args[5], load_state), load<bool>()(op.args[6], load_state), load<c10::optional<int64_t>>()(op.args[7], load_state));
  break;}

case H_CUDNN_CONVOLUTION_DEPRECATED2:
case H_MIOPEN_CONVOLUTION_TRANSPOSE_BACKWARD_INPUT:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t, bool, bool) = at::redispatch::cudnn_convolution;
  if (op.id == H_MIOPEN_CONVOLUTION_TRANSPOSE_BACKWARD_INPUT) ptr = at::redispatch::miopen_convolution_transpose_backward_input;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<at::IntArrayRef>()(op.args[3], load_state), load<at::IntArrayRef>()(op.args[4], load_state), load<int64_t>()(op.args[5], load_state), load<bool>()(op.args[6], load_state), load<bool>()(op.args[7], load_state));
  break;}

case H_CUDNN_CONVOLUTION:
case H_CUDNN_CONVOLUTION_TRANSPOSE_BACKWARD_INPUT:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t, bool, bool, bool) = at::redispatch::cudnn_convolution;
  if (op.id == H_CUDNN_CONVOLUTION_TRANSPOSE_BACKWARD_INPUT) ptr = at::redispatch::cudnn_convolution_transpose_backward_input;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<at::IntArrayRef>()(op.args[3], load_state), load<at::IntArrayRef>()(op.args[4], load_state), load<int64_t>()(op.args[5], load_state), load<bool>()(op.args[6], load_state), load<bool>()(op.args[7], load_state), load<bool>()(op.args[8], load_state));
  break;}

case H_FRACTIONAL_MAX_POOL2D_BACKWARD:
case H_FRACTIONAL_MAX_POOL3D_BACKWARD:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, const at::Tensor &) = at::redispatch::fractional_max_pool2d_backward;
  if (op.id == H_FRACTIONAL_MAX_POOL3D_BACKWARD) ptr = at::redispatch::fractional_max_pool3d_backward;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<at::IntArrayRef>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;}

case H_CTC_LOSS_INTLIST:
  results[i] = at::redispatch::ctc_loss(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<at::IntArrayRef>()(op.args[3], load_state), load<int64_t>()(op.args[4], load_state), load<int64_t>()(op.args[5], load_state), load<bool>()(op.args[6], load_state));
  break;

case H_SPARSE_COO_TENSOR_INDICES_SIZE:
case H__SPARSE_COO_TENSOR_UNSAFE:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, at::IntArrayRef, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<at::Device>, c10::optional<bool>) = at::redispatch::sparse_coo_tensor;
  if (op.id == H__SPARSE_COO_TENSOR_UNSAFE) ptr = at::redispatch::_sparse_coo_tensor_unsafe;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<c10::optional<at::ScalarType>>()(op.args[3], load_state), load<c10::optional<at::Layout>>()(op.args[4], load_state), load<c10::optional<at::Device>>()(op.args[5], load_state), load<c10::optional<bool>>()(op.args[6], load_state));
  break;}

case H_THNN_CONV2D:
case H_SLOW_CONV3D:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, at::IntArrayRef, const c10::optional<at::Tensor> &, at::IntArrayRef, at::IntArrayRef) = at::redispatch::thnn_conv2d;
  if (op.id == H_SLOW_CONV3D) ptr = at::redispatch::slow_conv3d;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<c10::optional<at::Tensor> &>()(op.args[3], load_state), load<at::IntArrayRef>()(op.args[4], load_state), load<at::IntArrayRef>()(op.args[5], load_state));
  break;}

case H__CONV_DEPTHWISE2D:
case H_CONV_DEPTHWISE3D:
case H_SLOW_CONV_DILATED2D:
case H_SLOW_CONV_DILATED3D:
  results[i] = redispatch_ptrs_21[op.id - H__CONV_DEPTHWISE2D](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<c10::optional<at::Tensor> &>()(op.args[3], load_state), load<at::IntArrayRef>()(op.args[4], load_state), load<at::IntArrayRef>()(op.args[5], load_state), load<at::IntArrayRef>()(op.args[6], load_state));
  break;

case H_SLOW_CONV_TRANSPOSE2D:
case H_SLOW_CONV_TRANSPOSE3D:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, at::IntArrayRef, const c10::optional<at::Tensor> &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef) = at::redispatch::slow_conv_transpose2d;
  if (op.id == H_SLOW_CONV_TRANSPOSE3D) ptr = at::redispatch::slow_conv_transpose3d;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<c10::optional<at::Tensor> &>()(op.args[3], load_state), load<at::IntArrayRef>()(op.args[4], load_state), load<at::IntArrayRef>()(op.args[5], load_state), load<at::IntArrayRef>()(op.args[6], load_state), load<at::IntArrayRef>()(op.args[7], load_state));
  break;}

case H_ADD_TENSOR:
case H__ADD_RELU_TENSOR:
case H_HARDSHRINK_BACKWARD:
case H_THRESHOLD_BACKWARD:
case H_WHERE_SCALAROTHER:
case H_SUB_TENSOR:
case H_SUBTRACT_TENSOR:
case H_RSUB_TENSOR:
case H_MASKED_FILL_SCALAR:
case H_DIST:
case H_LERP_SCALAR:
case H_SOFTSHRINK_BACKWARD:
case H__TEST_SERIALIZATION_SUBCMUL:
  results[i] = redispatch_ptrs_22[op.id - H_ADD_TENSOR](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Scalar &>()(op.args[2], load_state));
  break;

case H_HARDTANH_BACKWARD:
  results[i] = at::redispatch::hardtanh_backward(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Scalar &>()(op.args[2], load_state), load<at::Scalar &>()(op.args[3], load_state));
  break;

case H_SOFTPLUS_BACKWARD:
  results[i] = at::redispatch::softplus_backward(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Scalar &>()(op.args[2], load_state), load<at::Scalar &>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;

case H_RRELU_WITH_NOISE:
  results[i] = at::redispatch::rrelu_with_noise(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Scalar &>()(op.args[2], load_state), load<at::Scalar &>()(op.args[3], load_state), load<bool>()(op.args[4], load_state), load<c10::optional<at::Generator>>()(op.args[5], load_state));
  break;

case H_MULTI_MARGIN_LOSS:
  results[i] = at::redispatch::multi_margin_loss(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Scalar &>()(op.args[2], load_state), load<at::Scalar &>()(op.args[3], load_state), load<c10::optional<at::Tensor> &>()(op.args[4], load_state), load<int64_t>()(op.args[5], load_state));
  break;

case H_LEAKY_RELU_BACKWARD:
  results[i] = at::redispatch::leaky_relu_backward(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Scalar &>()(op.args[2], load_state), load<bool>()(op.args[3], load_state));
  break;

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
case H_ADAPTIVE_MAX_POOL2D_BACKWARD:
case H_ADAPTIVE_MAX_POOL3D_BACKWARD:
  results[i] = redispatch_ptrs_23[op.id - H_FBGEMM_LINEAR_FP16_WEIGHT_FP32_ACTIVATION](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state));
  break;

case H__NNPACK_SPATIAL_CONVOLUTION_BACKWARD_INPUT:
case H_MAX_UNPOOL2D_BACKWARD:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, const at::Tensor &, at::IntArrayRef) = at::redispatch::_nnpack_spatial_convolution_backward_input;
  if (op.id == H_MAX_UNPOOL2D_BACKWARD) ptr = at::redispatch::max_unpool2d_backward;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::IntArrayRef>()(op.args[3], load_state));
  break;}

case H_MAX_UNPOOL3D_BACKWARD:
  results[i] = at::redispatch::max_unpool3d_backward(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::IntArrayRef>()(op.args[3], load_state), load<at::IntArrayRef>()(op.args[4], load_state), load<at::IntArrayRef>()(op.args[5], load_state));
  break;

case H_MKLDNN_MAX_POOL2D_BACKWARD:
case H_MKLDNN_MAX_POOL3D_BACKWARD:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, bool) = at::redispatch::mkldnn_max_pool2d_backward;
  if (op.id == H_MKLDNN_MAX_POOL3D_BACKWARD) ptr = at::redispatch::mkldnn_max_pool3d_backward;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::IntArrayRef>()(op.args[3], load_state), load<at::IntArrayRef>()(op.args[4], load_state), load<at::IntArrayRef>()(op.args[5], load_state), load<at::IntArrayRef>()(op.args[6], load_state), load<bool>()(op.args[7], load_state));
  break;}

case H__TRILINEAR:
  results[i] = at::redispatch::_trilinear(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::IntArrayRef>()(op.args[3], load_state), load<at::IntArrayRef>()(op.args[4], load_state), load<at::IntArrayRef>()(op.args[5], load_state), load<at::IntArrayRef>()(op.args[6], load_state), load<int64_t>()(op.args[7], load_state));
  break;

case H__CTC_LOSS_BACKWARD:
  results[i] = at::redispatch::_ctc_loss_backward(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::IntArrayRef>()(op.args[3], load_state), load<at::IntArrayRef>()(op.args[4], load_state), load<at::Tensor &>()(op.args[5], load_state), load<at::Tensor &>()(op.args[6], load_state), load<int64_t>()(op.args[7], load_state), load<bool>()(op.args[8], load_state));
  break;

case H_SPARSE_CSR_TENSOR_CROW_COL_VALUE_SIZE:
case H__SPARSE_CSR_TENSOR_UNSAFE:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, const at::Tensor &, at::IntArrayRef, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<at::Device>, c10::optional<bool>) = at::redispatch::sparse_csr_tensor;
  if (op.id == H__SPARSE_CSR_TENSOR_UNSAFE) ptr = at::redispatch::_sparse_csr_tensor_unsafe;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::IntArrayRef>()(op.args[3], load_state), load<c10::optional<at::ScalarType>>()(op.args[4], load_state), load<c10::optional<at::Layout>>()(op.args[5], load_state), load<c10::optional<at::Device>>()(op.args[6], load_state), load<c10::optional<bool>>()(op.args[7], load_state));
  break;}

case H_ADDCMUL:
case H_ADDCDIV:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Scalar &) = at::redispatch::addcmul;
  if (op.id == H_ADDCDIV) ptr = at::redispatch::addcdiv;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::Scalar &>()(op.args[3], load_state));
  break;}

case H_ADDMV:
case H_ADDR:
case H_BADDBMM:
case H_SSPADDMM:
case H__SPARSE_ADDMM:
case H_ADDMM:
case H_ADDBMM:
  results[i] = redispatch_ptrs_24[op.id - H_ADDMV](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::Scalar &>()(op.args[3], load_state), load<at::Scalar &>()(op.args[4], load_state));
  break;

case H_RRELU_WITH_NOISE_BACKWARD:
  results[i] = at::redispatch::rrelu_with_noise_backward(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::Scalar &>()(op.args[3], load_state), load<at::Scalar &>()(op.args[4], load_state), load<bool>()(op.args[5], load_state), load<bool>()(op.args[6], load_state));
  break;

case H_MULTI_MARGIN_LOSS_BACKWARD:
  results[i] = at::redispatch::multi_margin_loss_backward(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::Scalar &>()(op.args[3], load_state), load<at::Scalar &>()(op.args[4], load_state), load<c10::optional<at::Tensor> &>()(op.args[5], load_state), load<int64_t>()(op.args[6], load_state));
  break;

case H_QUANTIZE_PER_TENSOR_TENSOR_QPARAMS:
  results[i] = at::redispatch::quantize_per_tensor(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::ScalarType>()(op.args[3], load_state));
  break;

case H_FBGEMM_LINEAR_INT8_WEIGHT_FP32_ACTIVATION:
case H_FBGEMM_LINEAR_INT8_WEIGHT:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Scalar &, const at::Scalar &, const at::Tensor &) = at::redispatch::fbgemm_linear_int8_weight_fp32_activation;
  if (op.id == H_FBGEMM_LINEAR_INT8_WEIGHT) ptr = at::redispatch::fbgemm_linear_int8_weight;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state), load<at::Scalar &>()(op.args[4], load_state), load<at::Scalar &>()(op.args[5], load_state), load<at::Tensor &>()(op.args[6], load_state));
  break;}

case H__DET_LU_BASED_HELPER_BACKWARD_HELPER:
  results[i] = at::redispatch::_det_lu_based_helper_backward_helper(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;

case H_QUANTIZED_GRU_CELL:
case H_QUANTIZED_RNN_RELU_CELL:
case H_QUANTIZED_RNN_TANH_CELL:
  results[i] = redispatch_ptrs_25[op.id - H_QUANTIZED_GRU_CELL](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state), load<at::Tensor &>()(op.args[5], load_state), load<at::Tensor &>()(op.args[6], load_state), load<at::Tensor &>()(op.args[7], load_state), load<at::Tensor &>()(op.args[8], load_state), load<at::Tensor &>()(op.args[9], load_state), load<at::Scalar &>()(op.args[10], load_state), load<at::Scalar &>()(op.args[11], load_state), load<at::Scalar &>()(op.args[12], load_state), load<at::Scalar &>()(op.args[13], load_state));
  break;

case H_FUSED_MOVING_AVG_OBS_FAKE_QUANT:
  results[i] = at::redispatch::fused_moving_avg_obs_fake_quant(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state), load<at::Tensor &>()(op.args[5], load_state), load<at::Tensor &>()(op.args[6], load_state), load<double>()(op.args[7], load_state), load<int64_t>()(op.args[8], load_state), load<int64_t>()(op.args[9], load_state), load<int64_t>()(op.args[10], load_state), load<bool>()(op.args[11], load_state), load<bool>()(op.args[12], load_state));
  break;

case H__EMBEDDING_BAG_BACKWARD:
  results[i] = at::redispatch::_embedding_bag_backward(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state), load<at::Tensor &>()(op.args[5], load_state), load<int64_t>()(op.args[6], load_state), load<bool>()(op.args[7], load_state), load<int64_t>()(op.args[8], load_state), load<bool>()(op.args[9], load_state), load<c10::optional<at::Tensor> &>()(op.args[10], load_state), load<int64_t>()(op.args[11], load_state));
  break;

case H__EMBEDDING_BAG_SPARSE_BACKWARD:
case H__EMBEDDING_BAG_DENSE_BACKWARD:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t, bool, int64_t, const c10::optional<at::Tensor> &, int64_t) = at::redispatch::_embedding_bag_sparse_backward;
  if (op.id == H__EMBEDDING_BAG_DENSE_BACKWARD) ptr = at::redispatch::_embedding_bag_dense_backward;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state), load<int64_t>()(op.args[5], load_state), load<bool>()(op.args[6], load_state), load<int64_t>()(op.args[7], load_state), load<c10::optional<at::Tensor> &>()(op.args[8], load_state), load<int64_t>()(op.args[9], load_state));
  break;}

case H__EMBEDDING_BAG_PER_SAMPLE_WEIGHTS_BACKWARD:
  results[i] = at::redispatch::_embedding_bag_per_sample_weights_backward(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state), load<int64_t>()(op.args[5], load_state), load<int64_t>()(op.args[6], load_state));
  break;

case H_BATCH_NORM_BACKWARD_ELEMT:
  results[i] = at::redispatch::batch_norm_backward_elemt(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state), load<c10::optional<at::Tensor> &>()(op.args[4], load_state), load<at::Tensor &>()(op.args[5], load_state), load<at::Tensor &>()(op.args[6], load_state), load<at::Tensor &>()(op.args[7], load_state));
  break;

case H_GRU_CELL:
case H_RNN_TANH_CELL:
case H_RNN_RELU_CELL:
  results[i] = redispatch_ptrs_26[op.id - H_GRU_CELL](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state), load<c10::optional<at::Tensor> &>()(op.args[4], load_state), load<c10::optional<at::Tensor> &>()(op.args[5], load_state));
  break;

case H_CTC_LOSS_TENSOR:
  results[i] = at::redispatch::ctc_loss(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state), load<int64_t>()(op.args[4], load_state), load<int64_t>()(op.args[5], load_state), load<bool>()(op.args[6], load_state));
  break;

case H_PUT:
  results[i] = at::redispatch::put(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<bool>()(op.args[3], load_state));
  break;

case H_ORMQR:
  results[i] = at::redispatch::ormqr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<bool>()(op.args[4], load_state));
  break;

case H_CUDNN_CONVOLUTION_ADD_RELU:
  results[i] = at::redispatch::cudnn_convolution_add_relu(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<c10::optional<at::Scalar> &>()(op.args[3], load_state), load<c10::optional<at::Tensor> &>()(op.args[4], load_state), load<at::IntArrayRef>()(op.args[5], load_state), load<at::IntArrayRef>()(op.args[6], load_state), load<at::IntArrayRef>()(op.args[7], load_state), load<int64_t>()(op.args[8], load_state));
  break;

case H_SPARSE_CSR_TENSOR_CROW_COL_VALUE:
  results[i] = at::redispatch::sparse_csr_tensor(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<c10::optional<at::ScalarType>>()(op.args[3], load_state), load<c10::optional<at::Layout>>()(op.args[4], load_state), load<c10::optional<at::Device>>()(op.args[5], load_state), load<c10::optional<bool>>()(op.args[6], load_state));
  break;

case H_BILINEAR:
  results[i] = at::redispatch::bilinear(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<c10::optional<at::Tensor> &>()(op.args[3], load_state));
  break;

case H_BINARY_CROSS_ENTROPY_WITH_LOGITS_BACKWARD:
  results[i] = at::redispatch::binary_cross_entropy_with_logits_backward(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<c10::optional<at::Tensor> &>()(op.args[3], load_state), load<c10::optional<at::Tensor> &>()(op.args[4], load_state), load<int64_t>()(op.args[5], load_state));
  break;

case H_BINARY_CROSS_ENTROPY_BACKWARD:
  results[i] = at::redispatch::binary_cross_entropy_backward(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<c10::optional<at::Tensor> &>()(op.args[3], load_state), load<int64_t>()(op.args[4], load_state));
  break;

case H_NLL_LOSS_BACKWARD:
case H_NLL_LOSS2D_BACKWARD:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, int64_t, int64_t, const at::Tensor &) = at::redispatch::nll_loss_backward;
  if (op.id == H_NLL_LOSS2D_BACKWARD) ptr = at::redispatch::nll_loss2d_backward;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<c10::optional<at::Tensor> &>()(op.args[3], load_state), load<int64_t>()(op.args[4], load_state), load<int64_t>()(op.args[5], load_state), load<at::Tensor &>()(op.args[6], load_state));
  break;}

case H__SEGMENT_REDUCE_BACKWARD:
  results[i] = at::redispatch::_segment_reduce_backward(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<c10::string_view>()(op.args[3], load_state), load<c10::optional<at::Tensor> &>()(op.args[4], load_state), load<int64_t>()(op.args[5], load_state));
  break;

case H__CDIST_BACKWARD:
  results[i] = at::redispatch::_cdist_backward(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<double>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;

case H_TRIPLET_MARGIN_LOSS:
  results[i] = at::redispatch::triplet_margin_loss(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<double>()(op.args[3], load_state), load<double>()(op.args[4], load_state), load<double>()(op.args[5], load_state), load<bool>()(op.args[6], load_state), load<int64_t>()(op.args[7], load_state));
  break;

case H_COSINE_EMBEDDING_LOSS:
case H_MARGIN_RANKING_LOSS:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, const at::Tensor &, double, int64_t) = at::redispatch::cosine_embedding_loss;
  if (op.id == H_MARGIN_RANKING_LOSS) ptr = at::redispatch::margin_ranking_loss;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<double>()(op.args[3], load_state), load<int64_t>()(op.args[4], load_state));
  break;}

case H_CONV_TBC:
case H_CUMMAXMIN_BACKWARD:
case H__MAKE_PER_CHANNEL_QUANTIZED_TENSOR:
case H_MSE_LOSS_BACKWARD:
case H_L1_LOSS_BACKWARD:
case H_SOFT_MARGIN_LOSS_BACKWARD:
  results[i] = redispatch_ptrs_27[op.id - H_CONV_TBC](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<int64_t>()(op.args[3], load_state));
  break;

case H_QUANTIZE_PER_CHANNEL:
  results[i] = at::redispatch::quantize_per_channel(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<int64_t>()(op.args[3], load_state), load<at::ScalarType>()(op.args[4], load_state));
  break;

case H_MULTILABEL_MARGIN_LOSS_BACKWARD:
  results[i] = at::redispatch::multilabel_margin_loss_backward(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<int64_t>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;

case H_KL_DIV_BACKWARD:
  results[i] = at::redispatch::kl_div_backward(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<int64_t>()(op.args[3], load_state), load<bool>()(op.args[4], load_state));
  break;

case H_SMOOTH_L1_LOSS_BACKWARD:
case H_HUBER_LOSS_BACKWARD:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t, double) = at::redispatch::smooth_l1_loss_backward;
  if (op.id == H_HUBER_LOSS_BACKWARD) ptr = at::redispatch::huber_loss_backward;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<int64_t>()(op.args[3], load_state), load<double>()(op.args[4], load_state));
  break;}

case H_FAKE_QUANTIZE_PER_TENSOR_AFFINE_TENSOR_QPARAMS:
  results[i] = at::redispatch::fake_quantize_per_tensor_affine(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<int64_t>()(op.args[3], load_state), load<int64_t>()(op.args[4], load_state));
  break;

case H__FAKE_QUANTIZE_LEARNABLE_PER_TENSOR_AFFINE:
  results[i] = at::redispatch::_fake_quantize_learnable_per_tensor_affine(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<int64_t>()(op.args[3], load_state), load<int64_t>()(op.args[4], load_state), load<double>()(op.args[5], load_state));
  break;

case H_FAKE_QUANTIZE_PER_CHANNEL_AFFINE:
  results[i] = at::redispatch::fake_quantize_per_channel_affine(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<int64_t>()(op.args[3], load_state), load<int64_t>()(op.args[4], load_state), load<int64_t>()(op.args[5], load_state));
  break;

case H__FAKE_QUANTIZE_LEARNABLE_PER_CHANNEL_AFFINE:
  results[i] = at::redispatch::_fake_quantize_learnable_per_channel_affine(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<int64_t>()(op.args[3], load_state), load<int64_t>()(op.args[4], load_state), load<int64_t>()(op.args[5], load_state), load<double>()(op.args[6], load_state));
  break;

case H__COPY_FROM:
case H_CHOLESKY_SOLVE:
case H__CHOLESKY_SOLVE_HELPER:
case H__CONVERT_INDICES_FROM_CSR_TO_COO:
case H_LINALG_PINV_RCOND_TENSOR:
case H_LINALG_MATRIX_RANK_TOL_TENSOR:
  results[i] = redispatch_ptrs_28[op.id - H__COPY_FROM](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<bool>()(op.args[2], load_state));
  break;

case H_ISIN_TENSOR_TENSOR:
case H_BUCKETIZE_TENSOR:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, bool, bool) = at::redispatch::isin;
  if (op.id == H_BUCKETIZE_TENSOR) ptr = at::redispatch::bucketize;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<bool>()(op.args[2], load_state), load<bool>()(op.args[3], load_state));
  break;}

case H_LINALG_SOLVE_TRIANGULAR:
  results[i] = at::redispatch::linalg_solve_triangular(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<bool>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<bool>()(op.args[4], load_state));
  break;

case H_TO_OTHER:
  results[i] = at::redispatch::to(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<bool>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<c10::optional<at::MemoryFormat>>()(op.args[4], load_state));
  break;

case H_SEARCHSORTED_TENSOR:
  results[i] = at::redispatch::searchsorted(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<bool>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<c10::optional<c10::string_view>>()(op.args[4], load_state), load<c10::optional<at::Tensor> &>()(op.args[5], load_state));
  break;

case H_POISSON_NLL_LOSS:
  results[i] = at::redispatch::poisson_nll_loss(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<bool>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<double>()(op.args[4], load_state), load<int64_t>()(op.args[5], load_state));
  break;

case H_BINOMIAL:
case H_NORMAL_TENSOR_TENSOR:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, c10::optional<at::Generator>) = at::redispatch::binomial;
  if (op.id == H_NORMAL_TENSOR_TENSOR) ptr = at::redispatch::normal;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<c10::optional<at::Generator>>()(op.args[2], load_state));
  break;}

case H_LINALG_TENSORSOLVE:
  results[i] = at::redispatch::linalg_tensorsolve(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<c10::optional<at::IntArrayRef>>()(op.args[2], load_state));
  break;

case H_SPARSE_COO_TENSOR_INDICES:
  results[i] = at::redispatch::sparse_coo_tensor(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<c10::optional<at::ScalarType>>()(op.args[2], load_state), load<c10::optional<at::Layout>>()(op.args[3], load_state), load<c10::optional<at::Device>>()(op.args[4], load_state), load<c10::optional<bool>>()(op.args[5], load_state));
  break;

case H_LINEAR:
case H_MKLDNN_LINEAR:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &) = at::redispatch::linear;
  if (op.id == H_MKLDNN_LINEAR) ptr = at::redispatch::mkldnn_linear;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<c10::optional<at::Tensor> &>()(op.args[2], load_state));
  break;}

case H__NNPACK_SPATIAL_CONVOLUTION:
  results[i] = at::redispatch::_nnpack_spatial_convolution(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<c10::optional<at::Tensor> &>()(op.args[2], load_state), load<at::IntArrayRef>()(op.args[3], load_state), load<at::IntArrayRef>()(op.args[4], load_state));
  break;

case H_CUDNN_CONVOLUTION_TRANSPOSE_DEPRECATED:
case H_MIOPEN_CONVOLUTION_TRANSPOSE:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t, bool, bool) = at::redispatch::cudnn_convolution_transpose;
  if (op.id == H_MIOPEN_CONVOLUTION_TRANSPOSE) ptr = at::redispatch::miopen_convolution_transpose;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<c10::optional<at::Tensor> &>()(op.args[2], load_state), load<at::IntArrayRef>()(op.args[3], load_state), load<at::IntArrayRef>()(op.args[4], load_state), load<at::IntArrayRef>()(op.args[5], load_state), load<at::IntArrayRef>()(op.args[6], load_state), load<int64_t>()(op.args[7], load_state), load<bool>()(op.args[8], load_state), load<bool>()(op.args[9], load_state));
  break;}

case H__CONVOLUTION_NOGROUP:
  results[i] = at::redispatch::_convolution_nogroup(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<c10::optional<at::Tensor> &>()(op.args[2], load_state), load<at::IntArrayRef>()(op.args[3], load_state), load<at::IntArrayRef>()(op.args[4], load_state), load<at::IntArrayRef>()(op.args[5], load_state), load<bool>()(op.args[6], load_state), load<at::IntArrayRef>()(op.args[7], load_state));
  break;

case H_CONVOLUTION:
case H_CONVOLUTION_OVERRIDEABLE:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, bool, at::IntArrayRef, int64_t) = at::redispatch::convolution;
  if (op.id == H_CONVOLUTION_OVERRIDEABLE) ptr = at::redispatch::convolution_overrideable;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<c10::optional<at::Tensor> &>()(op.args[2], load_state), load<at::IntArrayRef>()(op.args[3], load_state), load<at::IntArrayRef>()(op.args[4], load_state), load<at::IntArrayRef>()(op.args[5], load_state), load<bool>()(op.args[6], load_state), load<at::IntArrayRef>()(op.args[7], load_state), load<int64_t>()(op.args[8], load_state));
  break;}

case H__CONVOLUTION_DEPRECATED:
  results[i] = at::redispatch::_convolution(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<c10::optional<at::Tensor> &>()(op.args[2], load_state), load<at::IntArrayRef>()(op.args[3], load_state), load<at::IntArrayRef>()(op.args[4], load_state), load<at::IntArrayRef>()(op.args[5], load_state), load<bool>()(op.args[6], load_state), load<at::IntArrayRef>()(op.args[7], load_state), load<int64_t>()(op.args[8], load_state), load<bool>()(op.args[9], load_state), load<bool>()(op.args[10], load_state), load<bool>()(op.args[11], load_state));
  break;

case H__CONVOLUTION:
  results[i] = at::redispatch::_convolution(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<c10::optional<at::Tensor> &>()(op.args[2], load_state), load<at::IntArrayRef>()(op.args[3], load_state), load<at::IntArrayRef>()(op.args[4], load_state), load<at::IntArrayRef>()(op.args[5], load_state), load<bool>()(op.args[6], load_state), load<at::IntArrayRef>()(op.args[7], load_state), load<int64_t>()(op.args[8], load_state), load<bool>()(op.args[9], load_state), load<bool>()(op.args[10], load_state), load<bool>()(op.args[11], load_state), load<bool>()(op.args[12], load_state));
  break;

case H_CONV1D:
case H_CONV2D:
case H_CONV3D:
case H_CUDNN_CONVOLUTION_RELU:
case H_MKLDNN_CONVOLUTION:
  results[i] = redispatch_ptrs_29[op.id - H_CONV1D](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<c10::optional<at::Tensor> &>()(op.args[2], load_state), load<at::IntArrayRef>()(op.args[3], load_state), load<at::IntArrayRef>()(op.args[4], load_state), load<at::IntArrayRef>()(op.args[5], load_state), load<int64_t>()(op.args[6], load_state));
  break;

case H_CONV_TRANSPOSE1D:
case H_CONV_TRANSPOSE2D_INPUT:
case H_CONV_TRANSPOSE3D_INPUT:
  results[i] = redispatch_ptrs_30[op.id - H_CONV_TRANSPOSE1D](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<c10::optional<at::Tensor> &>()(op.args[2], load_state), load<at::IntArrayRef>()(op.args[3], load_state), load<at::IntArrayRef>()(op.args[4], load_state), load<at::IntArrayRef>()(op.args[5], load_state), load<int64_t>()(op.args[6], load_state), load<at::IntArrayRef>()(op.args[7], load_state));
  break;

case H_CUDNN_CONVOLUTION_DEPRECATED:
case H_MIOPEN_CONVOLUTION:
case H_MIOPEN_DEPTHWISE_CONVOLUTION:
  results[i] = redispatch_ptrs_31[op.id - H_CUDNN_CONVOLUTION_DEPRECATED](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<c10::optional<at::Tensor> &>()(op.args[2], load_state), load<at::IntArrayRef>()(op.args[3], load_state), load<at::IntArrayRef>()(op.args[4], load_state), load<at::IntArrayRef>()(op.args[5], load_state), load<int64_t>()(op.args[6], load_state), load<bool>()(op.args[7], load_state), load<bool>()(op.args[8], load_state));
  break;

case H__CONVOLUTION_MODE:
case H_CONV1D_PADDING:
case H_CONV2D_PADDING:
case H_CONV3D_PADDING:
  results[i] = redispatch_ptrs_32[op.id - H__CONVOLUTION_MODE](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<c10::optional<at::Tensor> &>()(op.args[2], load_state), load<at::IntArrayRef>()(op.args[3], load_state), load<c10::string_view>()(op.args[4], load_state), load<at::IntArrayRef>()(op.args[5], load_state), load<int64_t>()(op.args[6], load_state));
  break;

case H_BINARY_CROSS_ENTROPY_WITH_LOGITS:
  results[i] = at::redispatch::binary_cross_entropy_with_logits(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<c10::optional<at::Tensor> &>()(op.args[2], load_state), load<c10::optional<at::Tensor> &>()(op.args[3], load_state), load<int64_t>()(op.args[4], load_state));
  break;

case H_BINARY_CROSS_ENTROPY:
  results[i] = at::redispatch::binary_cross_entropy(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<c10::optional<at::Tensor> &>()(op.args[2], load_state), load<int64_t>()(op.args[3], load_state));
  break;

case H_NLL_LOSS_ND:
case H_NLL_LOSS:
case H_NLL_LOSS2D:
  results[i] = redispatch_ptrs_33[op.id - H_NLL_LOSS_ND](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<c10::optional<at::Tensor> &>()(op.args[2], load_state), load<int64_t>()(op.args[3], load_state), load<int64_t>()(op.args[4], load_state));
  break;

case H_CROSS_ENTROPY_LOSS:
  results[i] = at::redispatch::cross_entropy_loss(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<c10::optional<at::Tensor> &>()(op.args[2], load_state), load<int64_t>()(op.args[3], load_state), load<int64_t>()(op.args[4], load_state), load<double>()(op.args[5], load_state));
  break;

case H_DIV_TENSOR_MODE:
case H_DIVIDE_TENSOR_MODE:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, c10::optional<c10::string_view>) = at::redispatch::div;
  if (op.id == H_DIVIDE_TENSOR_MODE) ptr = at::redispatch::divide;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<c10::optional<c10::string_view>>()(op.args[2], load_state));
  break;}

case H_LOGIT_BACKWARD:
  results[i] = at::redispatch::logit_backward(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<c10::optional<double>>()(op.args[2], load_state));
  break;

case H_CROSS:
case H_TAKE_ALONG_DIM:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, c10::optional<int64_t>) = at::redispatch::cross;
  if (op.id == H_TAKE_ALONG_DIM) ptr = at::redispatch::take_along_dim;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<c10::optional<int64_t>>()(op.args[2], load_state));
  break;}

case H_QUANTILE:
case H_NANQUANTILE:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, c10::optional<int64_t>, bool) = at::redispatch::quantile;
  if (op.id == H_NANQUANTILE) ptr = at::redispatch::nanquantile;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<c10::optional<int64_t>>()(op.args[2], load_state), load<bool>()(op.args[3], load_state));
  break;}

case H_QUANTILE_NEW:
case H_NANQUANTILE_NEW:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, c10::optional<int64_t>, bool, c10::string_view) = at::redispatch::quantile;
  if (op.id == H_NANQUANTILE_NEW) ptr = at::redispatch::nanquantile;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<c10::optional<int64_t>>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<c10::string_view>()(op.args[4], load_state));
  break;}

case H_REPEAT_INTERLEAVE_SELF_TENSOR:
  results[i] = at::redispatch::repeat_interleave(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<c10::optional<int64_t>>()(op.args[2], load_state), load<c10::optional<int64_t>>()(op.args[3], load_state));
  break;

case H__MASKED_SCALE:
case H_NATIVE_DROPOUT_BACKWARD:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, double) = at::redispatch::_masked_scale;
  if (op.id == H_NATIVE_DROPOUT_BACKWARD) ptr = at::redispatch::native_dropout_backward;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<double>()(op.args[2], load_state));
  break;}

case H__PDIST_BACKWARD:
  results[i] = at::redispatch::_pdist_backward(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<double>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state));
  break;

case H_CDIST:
case H__CDIST_FORWARD:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, double, c10::optional<int64_t>) = at::redispatch::cdist;
  if (op.id == H__CDIST_FORWARD) ptr = at::redispatch::_cdist_forward;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<double>()(op.args[2], load_state), load<c10::optional<int64_t>>()(op.args[3], load_state));
  break;}

case H_ISCLOSE:
case H_PAIRWISE_DISTANCE:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, double, double, bool) = at::redispatch::isclose;
  if (op.id == H_PAIRWISE_DISTANCE) ptr = at::redispatch::pairwise_distance;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<double>()(op.args[2], load_state), load<double>()(op.args[3], load_state), load<bool>()(op.args[4], load_state));
  break;}

case H_HINGE_EMBEDDING_LOSS:
  results[i] = at::redispatch::hinge_embedding_loss(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<double>()(op.args[2], load_state), load<int64_t>()(op.args[3], load_state));
  break;

case H__MAKE_DUAL:
case H__NEW_ZEROS_WITH_SAME_FEATURE_META:
case H_CUMULATIVE_TRAPEZOID_X:
case H_TRAPEZOID_X:
case H_TRAPZ_X:
case H__WEIGHT_NORM:
case H_MSE_LOSS:
case H_L1_LOSS:
case H_MULTILABEL_MARGIN_LOSS:
case H_SOFT_MARGIN_LOSS:
case H_GLU_BACKWARD:
case H_LINALG_CROSS:
  results[i] = redispatch_ptrs_34[op.id - H__MAKE_DUAL](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<int64_t>()(op.args[2], load_state));
  break;

case H__LOG_SOFTMAX_BACKWARD_DATA:
case H__SOFTMAX_BACKWARD_DATA:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, int64_t, at::ScalarType) = at::redispatch::_log_softmax_backward_data;
  if (op.id == H__SOFTMAX_BACKWARD_DATA) ptr = at::redispatch::_softmax_backward_data;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<int64_t>()(op.args[2], load_state), load<at::ScalarType>()(op.args[3], load_state));
  break;}

case H_CUMPROD_BACKWARD:
case H__SPARSE_SOFTMAX_BACKWARD_DATA:
case H__SPARSE_LOG_SOFTMAX_BACKWARD_DATA:
  results[i] = redispatch_ptrs_35[op.id - H_CUMPROD_BACKWARD](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<int64_t>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state));
  break;

case H_GATHER_BACKWARD:
  results[i] = at::redispatch::gather_backward(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<int64_t>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state), load<bool>()(op.args[4], load_state));
  break;

case H_KL_DIV:
  results[i] = at::redispatch::kl_div(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<int64_t>()(op.args[2], load_state), load<bool>()(op.args[3], load_state));
  break;

case H_EMBEDDING:
  results[i] = at::redispatch::embedding(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<int64_t>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<bool>()(op.args[4], load_state));
  break;

case H_SLICE_SCATTER:
  results[i] = at::redispatch::slice_scatter(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<int64_t>()(op.args[2], load_state), load<c10::optional<int64_t>>()(op.args[3], load_state), load<c10::optional<int64_t>>()(op.args[4], load_state), load<int64_t>()(op.args[5], load_state));
  break;

case H_COSINE_SIMILARITY:
case H_SMOOTH_L1_LOSS:
case H_HUBER_LOSS:
  results[i] = redispatch_ptrs_36[op.id - H_COSINE_SIMILARITY](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<int64_t>()(op.args[2], load_state), load<double>()(op.args[3], load_state));
  break;

case H_SELECT_SCATTER:
  results[i] = at::redispatch::select_scatter(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<int64_t>()(op.args[2], load_state), load<int64_t>()(op.args[3], load_state));
  break;

case H_EMBEDDING_DENSE_BACKWARD:
case H_EMBEDDING_SPARSE_BACKWARD:
case H_GRID_SAMPLER:
case H_GRID_SAMPLER_2D:
case H__GRID_SAMPLER_2D_CPU_FALLBACK:
case H_GRID_SAMPLER_3D:
  results[i] = redispatch_ptrs_37[op.id - H_EMBEDDING_DENSE_BACKWARD](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<int64_t>()(op.args[2], load_state), load<int64_t>()(op.args[3], load_state), load<bool>()(op.args[4], load_state));
  break;

case H_EMBEDDING_BACKWARD:
  results[i] = at::redispatch::embedding_backward(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<int64_t>()(op.args[2], load_state), load<int64_t>()(op.args[3], load_state), load<bool>()(op.args[4], load_state), load<bool>()(op.args[5], load_state));
  break;

case H_DIAGONAL_SCATTER:
  results[i] = at::redispatch::diagonal_scatter(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<int64_t>()(op.args[2], load_state), load<int64_t>()(op.args[3], load_state), load<int64_t>()(op.args[4], load_state));
  break;

case H__HISTOGRAMDD_FROM_BIN_TENSORS:
  results[i] = at::redispatch::_histogramdd_from_bin_tensors(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::TensorList>()(op.args[1], load_state), load<c10::optional<at::Tensor> &>()(op.args[2], load_state), load<bool>()(op.args[3], load_state));
  break;

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
case H_LINALG_CHOLESKY:
  results[i] = redispatch_ptrs_38[op.id - H__CAST_BYTE](ks, load<at::Tensor &>()(op.args[0], load_state), load<bool>()(op.args[1], load_state));
  break;

case H__AUTOCAST_TO_FULL_PRECISION:
  results[i] = at::redispatch::_autocast_to_full_precision(ks, load<at::Tensor &>()(op.args[0], load_state), load<bool>()(op.args[1], load_state), load<bool>()(op.args[2], load_state));
  break;

case H__AUTOCAST_TO_REDUCED_PRECISION:
  results[i] = at::redispatch::_autocast_to_reduced_precision(ks, load<at::Tensor &>()(op.args[0], load_state), load<bool>()(op.args[1], load_state), load<bool>()(op.args[2], load_state), load<at::ScalarType>()(op.args[3], load_state), load<at::ScalarType>()(op.args[4], load_state));
  break;

case H_INDEX_TENSOR:
  results[i] = at::redispatch::index(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::List<c10::optional<at::Tensor>> &>()(op.args[1], load_state));
  break;

case H_INDEX_PUT:
  results[i] = at::redispatch::index_put(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::List<c10::optional<at::Tensor>> &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<bool>()(op.args[3], load_state));
  break;

case H__TEST_OPTIONAL_FLOATLIST:
  results[i] = at::redispatch::_test_optional_floatlist(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::ArrayRef<double>>>()(op.args[1], load_state));
  break;

case H_PIN_MEMORY:
case H__PIN_MEMORY:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, c10::optional<at::Device>) = at::redispatch::pin_memory;
  if (op.id == H__PIN_MEMORY) ptr = at::redispatch::_pin_memory;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::Device>>()(op.args[1], load_state));
  break;}

case H_RENAME:
  results[i] = at::redispatch::rename(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::DimnameList>>()(op.args[1], load_state));
  break;

case H_BERNOULLI:
case H__STANDARD_GAMMA:
case H__SAMPLE_DIRICHLET:
case H_POISSON:
  results[i] = redispatch_ptrs_39[op.id - H_BERNOULLI](ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::Generator>>()(op.args[1], load_state));
  break;

case H_FFT_FFTSHIFT:
case H_FFT_IFFTSHIFT:
case H__TEST_OPTIONAL_INTLIST:
case H__TEST_OPTIONAL_FILLED_INTLIST:
  results[i] = redispatch_ptrs_40[op.id - H_FFT_FFTSHIFT](ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::IntArrayRef>>()(op.args[1], load_state));
  break;

case H_UPSAMPLE_LINEAR1D_BACKWARD_VEC:
case H_UPSAMPLE_BILINEAR2D_BACKWARD_VEC:
case H__UPSAMPLE_BILINEAR2D_AA_BACKWARD_VEC:
case H_UPSAMPLE_TRILINEAR3D_BACKWARD_VEC:
case H_UPSAMPLE_BICUBIC2D_BACKWARD_VEC:
  results[i] = redispatch_ptrs_41[op.id - H_UPSAMPLE_LINEAR1D_BACKWARD_VEC](ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::IntArrayRef>>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<c10::optional<at::ArrayRef<double>>>()(op.args[4], load_state));
  break;

case H_UPSAMPLE_NEAREST1D_BACKWARD_VEC:
case H__UPSAMPLE_NEAREST_EXACT1D_BACKWARD_VEC:
case H_UPSAMPLE_NEAREST2D_BACKWARD_VEC:
case H__UPSAMPLE_NEAREST_EXACT2D_BACKWARD_VEC:
case H_UPSAMPLE_NEAREST3D_BACKWARD_VEC:
case H__UPSAMPLE_NEAREST_EXACT3D_BACKWARD_VEC:
  results[i] = redispatch_ptrs_42[op.id - H_UPSAMPLE_NEAREST1D_BACKWARD_VEC](ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::IntArrayRef>>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<c10::optional<at::ArrayRef<double>>>()(op.args[3], load_state));
  break;

case H_FFT_FFT2:
case H_FFT_IFFT2:
case H_FFT_RFFT2:
case H_FFT_IRFFT2:
case H_FFT_HFFT2:
case H_FFT_IHFFT2:
  results[i] = redispatch_ptrs_43[op.id - H_FFT_FFT2](ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::IntArrayRef>>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<c10::optional<c10::string_view>>()(op.args[3], load_state));
  break;

case H_UPSAMPLE_LINEAR1D_VEC:
case H_UPSAMPLE_BILINEAR2D_VEC:
case H__UPSAMPLE_BILINEAR2D_AA_VEC:
case H_UPSAMPLE_TRILINEAR3D_VEC:
case H_UPSAMPLE_BICUBIC2D_VEC:
  results[i] = redispatch_ptrs_44[op.id - H_UPSAMPLE_LINEAR1D_VEC](ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::IntArrayRef>>()(op.args[1], load_state), load<bool>()(op.args[2], load_state), load<c10::optional<at::ArrayRef<double>>>()(op.args[3], load_state));
  break;

case H_UPSAMPLE_NEAREST1D_VEC:
case H__UPSAMPLE_NEAREST_EXACT1D_VEC:
case H_UPSAMPLE_NEAREST2D_VEC:
case H__UPSAMPLE_NEAREST_EXACT2D_VEC:
case H_UPSAMPLE_NEAREST3D_VEC:
case H__UPSAMPLE_NEAREST_EXACT3D_VEC:
  results[i] = redispatch_ptrs_45[op.id - H_UPSAMPLE_NEAREST1D_VEC](ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::IntArrayRef>>()(op.args[1], load_state), load<c10::optional<at::ArrayRef<double>>>()(op.args[2], load_state));
  break;

case H_FFT_FFTN:
case H_FFT_IFFTN:
case H_FFT_RFFTN:
case H_FFT_IRFFTN:
case H_FFT_HFFTN:
case H_FFT_IHFFTN:
  results[i] = redispatch_ptrs_46[op.id - H_FFT_FFTN](ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::IntArrayRef>>()(op.args[1], load_state), load<c10::optional<at::IntArrayRef>>()(op.args[2], load_state), load<c10::optional<c10::string_view>>()(op.args[3], load_state));
  break;

case H_STD_CORRECTION:
case H_VAR_CORRECTION:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, c10::optional<at::IntArrayRef>, c10::optional<int64_t>, bool) = at::redispatch::std;
  if (op.id == H_VAR_CORRECTION) ptr = at::redispatch::var;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::IntArrayRef>>()(op.args[1], load_state), load<c10::optional<int64_t>>()(op.args[2], load_state), load<bool>()(op.args[3], load_state));
  break;}

case H_CLONE:
  results[i] = at::redispatch::clone(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::MemoryFormat>>()(op.args[1], load_state));
  break;

case H_LINALG_COND:
  results[i] = at::redispatch::linalg_cond(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::Scalar> &>()(op.args[1], load_state));
  break;

case H_NORM_NAMES_SCALAROPT_DIM:
  results[i] = at::redispatch::norm(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::Scalar> &>()(op.args[1], load_state), load<at::DimnameList>()(op.args[2], load_state), load<bool>()(op.args[3], load_state));
  break;

case H_NORM_NAMES_SCALAROPT_DIM_DTYPE:
  results[i] = at::redispatch::norm(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::Scalar> &>()(op.args[1], load_state), load<at::DimnameList>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<at::ScalarType>()(op.args[4], load_state));
  break;

case H_NORM_SCALAROPT_DIM:
  results[i] = at::redispatch::norm(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::Scalar> &>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<bool>()(op.args[3], load_state));
  break;

case H_NORM_SCALAROPT_DIM_DTYPE:
  results[i] = at::redispatch::norm(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::Scalar> &>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<at::ScalarType>()(op.args[4], load_state));
  break;

case H_NATIVE_NORM_SCALAROPT_DIM_DTYPE:
  results[i] = at::redispatch::native_norm(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::Scalar> &>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<c10::optional<at::ScalarType>>()(op.args[4], load_state));
  break;

case H_NORM_SCALAROPT_DTYPE:
  results[i] = at::redispatch::norm(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::Scalar> &>()(op.args[1], load_state), load<at::ScalarType>()(op.args[2], load_state));
  break;

case H_LINALG_NORM:
  results[i] = at::redispatch::linalg_norm(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::Scalar> &>()(op.args[1], load_state), load<c10::optional<at::IntArrayRef>>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<c10::optional<at::ScalarType>>()(op.args[4], load_state));
  break;

case H_CLAMP:
case H_CLIP:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const c10::optional<at::Scalar> &, const c10::optional<at::Scalar> &) = at::redispatch::clamp;
  if (op.id == H_CLIP) ptr = at::redispatch::clip;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::Scalar> &>()(op.args[1], load_state), load<c10::optional<at::Scalar> &>()(op.args[2], load_state));
  break;}

case H_MEAN:
case H_SUM:
case H_NANSUM:
case H_PROD:
case H_TO_DENSE:
case H_TO_MKLDNN:
  results[i] = redispatch_ptrs_47[op.id - H_MEAN](ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::ScalarType>>()(op.args[1], load_state));
  break;

case H_TO_DTYPE_LAYOUT:
  results[i] = at::redispatch::to(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::ScalarType>>()(op.args[1], load_state), load<c10::optional<at::Layout>>()(op.args[2], load_state), load<c10::optional<at::Device>>()(op.args[3], load_state), load<c10::optional<bool>>()(op.args[4], load_state), load<bool>()(op.args[5], load_state), load<bool>()(op.args[6], load_state), load<c10::optional<at::MemoryFormat>>()(op.args[7], load_state));
  break;

case H__TO_COPY:
  results[i] = at::redispatch::_to_copy(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::ScalarType>>()(op.args[1], load_state), load<c10::optional<at::Layout>>()(op.args[2], load_state), load<c10::optional<at::Device>>()(op.args[3], load_state), load<c10::optional<bool>>()(op.args[4], load_state), load<bool>()(op.args[5], load_state), load<c10::optional<at::MemoryFormat>>()(op.args[6], load_state));
  break;

case H_EMPTY_LIKE:
case H_ONES_LIKE:
case H_RAND_LIKE:
case H_RANDN_LIKE:
case H_ZEROS_LIKE:
  results[i] = redispatch_ptrs_48[op.id - H_EMPTY_LIKE](ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::ScalarType>>()(op.args[1], load_state), load<c10::optional<at::Layout>>()(op.args[2], load_state), load<c10::optional<at::Device>>()(op.args[3], load_state), load<c10::optional<bool>>()(op.args[4], load_state), load<c10::optional<at::MemoryFormat>>()(op.args[5], load_state));
  break;

case H_CLAMP_TENSOR:
case H_CLIP_TENSOR:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &) = at::redispatch::clamp;
  if (op.id == H_CLIP_TENSOR) ptr = at::redispatch::clip;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::Tensor> &>()(op.args[1], load_state), load<c10::optional<at::Tensor> &>()(op.args[2], load_state));
  break;}

case H_BATCH_NORM_ELEMT:
  results[i] = at::redispatch::batch_norm_elemt(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::Tensor> &>()(op.args[1], load_state), load<c10::optional<at::Tensor> &>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state), load<double>()(op.args[5], load_state));
  break;

case H_QUANTIZED_BATCH_NORM:
  results[i] = at::redispatch::quantized_batch_norm(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::Tensor> &>()(op.args[1], load_state), load<c10::optional<at::Tensor> &>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state), load<double>()(op.args[5], load_state), load<double>()(op.args[6], load_state), load<int64_t>()(op.args[7], load_state));
  break;

case H_LINALG_PINV_ATOL_RTOL_TENSOR:
case H_LINALG_MATRIX_RANK_ATOL_RTOL_TENSOR:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, bool) = at::redispatch::linalg_pinv;
  if (op.id == H_LINALG_MATRIX_RANK_ATOL_RTOL_TENSOR) ptr = at::redispatch::linalg_matrix_rank;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::Tensor> &>()(op.args[1], load_state), load<c10::optional<at::Tensor> &>()(op.args[2], load_state), load<bool>()(op.args[3], load_state));
  break;}

case H_BATCH_NORM:
case H_INSTANCE_NORM:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, bool, double, double, bool) = at::redispatch::batch_norm;
  if (op.id == H_INSTANCE_NORM) ptr = at::redispatch::instance_norm;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::Tensor> &>()(op.args[1], load_state), load<c10::optional<at::Tensor> &>()(op.args[2], load_state), load<c10::optional<at::Tensor> &>()(op.args[3], load_state), load<c10::optional<at::Tensor> &>()(op.args[4], load_state), load<bool>()(op.args[5], load_state), load<double>()(op.args[6], load_state), load<double>()(op.args[7], load_state), load<bool>()(op.args[8], load_state));
  break;}

case H_BINCOUNT:
  results[i] = at::redispatch::bincount(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::Tensor> &>()(op.args[1], load_state), load<int64_t>()(op.args[2], load_state));
  break;

case H_LOGIT:
case H_SPECIAL_LOGIT:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, c10::optional<double>) = at::redispatch::logit;
  if (op.id == H_SPECIAL_LOGIT) ptr = at::redispatch::special_logit;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<double>>()(op.args[1], load_state));
  break;}

case H_LINALG_PINV_ATOL_RTOL_FLOAT:
case H_LINALG_MATRIX_RANK_ATOL_RTOL_FLOAT:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, c10::optional<double>, c10::optional<double>, bool) = at::redispatch::linalg_pinv;
  if (op.id == H_LINALG_MATRIX_RANK_ATOL_RTOL_FLOAT) ptr = at::redispatch::linalg_matrix_rank;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<double>>()(op.args[1], load_state), load<c10::optional<double>>()(op.args[2], load_state), load<bool>()(op.args[3], load_state));
  break;}

case H_NAN_TO_NUM:
  results[i] = at::redispatch::nan_to_num(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<double>>()(op.args[1], load_state), load<c10::optional<double>>()(op.args[2], load_state), load<c10::optional<double>>()(op.args[3], load_state));
  break;

case H_COUNT_NONZERO:
case H_REPEAT_INTERLEAVE_TENSOR:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, c10::optional<int64_t>) = at::redispatch::count_nonzero;
  if (op.id == H_REPEAT_INTERLEAVE_TENSOR) ptr = at::redispatch::repeat_interleave;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<int64_t>>()(op.args[1], load_state));
  break;}

case H_ARGMAX:
case H_ARGMIN:
case H_VANDER:
  results[i] = redispatch_ptrs_49[op.id - H_ARGMAX](ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<int64_t>>()(op.args[1], load_state), load<bool>()(op.args[2], load_state));
  break;

case H_FFT_FFT:
case H_FFT_IFFT:
case H_FFT_RFFT:
case H_FFT_IRFFT:
case H_FFT_HFFT:
case H_FFT_IHFFT:
  results[i] = redispatch_ptrs_50[op.id - H_FFT_FFT](ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<int64_t>>()(op.args[1], load_state), load<int64_t>()(op.args[2], load_state), load<c10::optional<c10::string_view>>()(op.args[3], load_state));
  break;

case H_LINALG_EIGVALSH:
case H_LINALG_COND_P_STR:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, c10::string_view) = at::redispatch::linalg_eigvalsh;
  if (op.id == H_LINALG_COND_P_STR) ptr = at::redispatch::linalg_cond;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::string_view>()(op.args[1], load_state));
  break;}

case H_LINALG_MATRIX_NORM_STR_ORD:
  results[i] = at::redispatch::linalg_matrix_norm(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::string_view>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<c10::optional<at::ScalarType>>()(op.args[4], load_state));
  break;

case H_LINALG_NORM_ORD_STR:
  results[i] = at::redispatch::linalg_norm(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::string_view>()(op.args[1], load_state), load<c10::optional<at::IntArrayRef>>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<c10::optional<at::ScalarType>>()(op.args[4], load_state));
  break;

case H_SEGMENT_REDUCE:
  results[i] = at::redispatch::segment_reduce(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::string_view>()(op.args[1], load_state), load<c10::optional<at::Tensor> &>()(op.args[2], load_state), load<c10::optional<at::Tensor> &>()(op.args[3], load_state), load<int64_t>()(op.args[4], load_state), load<bool>()(op.args[5], load_state), load<c10::optional<at::Scalar> &>()(op.args[6], load_state));
  break;

case H__TEST_STRING_DEFAULT:
  results[i] = at::redispatch::_test_string_default(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::string_view>()(op.args[1], load_state), load<c10::string_view>()(op.args[2], load_state));
  break;

case H_PDIST:
case H__PDIST_FORWARD:
case H_PINVERSE:
  results[i] = redispatch_ptrs_51[op.id - H_PDIST](ks, load<at::Tensor &>()(op.args[0], load_state), load<double>()(op.args[1], load_state));
  break;

case H_DROPOUT:
case H_FEATURE_DROPOUT:
case H_ALPHA_DROPOUT:
case H_FEATURE_ALPHA_DROPOUT:
case H_MATRIX_RANK_TOL:
case H_LINALG_PINV:
case H_LINALG_MATRIX_RANK:
  results[i] = redispatch_ptrs_52[op.id - H_DROPOUT](ks, load<at::Tensor &>()(op.args[0], load_state), load<double>()(op.args[1], load_state), load<bool>()(op.args[2], load_state));
  break;

case H_BERNOULLI_P:
case H_NORMAL_TENSOR_FLOAT:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, double, c10::optional<at::Generator>) = at::redispatch::bernoulli;
  if (op.id == H_NORMAL_TENSOR_FLOAT) ptr = at::redispatch::normal;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<double>()(op.args[1], load_state), load<c10::optional<at::Generator>>()(op.args[2], load_state));
  break;}

case H_QUANTILE_SCALAR:
case H_NANQUANTILE_SCALAR:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, double, c10::optional<int64_t>, bool) = at::redispatch::quantile;
  if (op.id == H_NANQUANTILE_SCALAR) ptr = at::redispatch::nanquantile;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<double>()(op.args[1], load_state), load<c10::optional<int64_t>>()(op.args[2], load_state), load<bool>()(op.args[3], load_state));
  break;}

case H_QUANTILE_NEW_SCALAR:
case H_NANQUANTILE_NEW_SCALAR:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, double, c10::optional<int64_t>, bool, c10::string_view) = at::redispatch::quantile;
  if (op.id == H_NANQUANTILE_NEW_SCALAR) ptr = at::redispatch::nanquantile;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<double>()(op.args[1], load_state), load<c10::optional<int64_t>>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<c10::string_view>()(op.args[4], load_state));
  break;}

case H_TRAPZ_DX:
case H__MAKE_PER_TENSOR_QUANTIZED_TENSOR:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, double, int64_t) = at::redispatch::trapz;
  if (op.id == H__MAKE_PER_TENSOR_QUANTIZED_TENSOR) ptr = at::redispatch::_make_per_tensor_quantized_tensor;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<double>()(op.args[1], load_state), load<int64_t>()(op.args[2], load_state));
  break;}

case H_QUANTIZE_PER_TENSOR:
  results[i] = at::redispatch::quantize_per_tensor(ks, load<at::Tensor &>()(op.args[0], load_state), load<double>()(op.args[1], load_state), load<int64_t>()(op.args[2], load_state), load<at::ScalarType>()(op.args[3], load_state));
  break;

case H_FAKE_QUANTIZE_PER_TENSOR_AFFINE:
  results[i] = at::redispatch::fake_quantize_per_tensor_affine(ks, load<at::Tensor &>()(op.args[0], load_state), load<double>()(op.args[1], load_state), load<int64_t>()(op.args[2], load_state), load<int64_t>()(op.args[3], load_state), load<int64_t>()(op.args[4], load_state));
  break;

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
case H_GLU:
case H_SPECIAL_MULTIGAMMALN:
case H_LINALG_TENSORINV:
case H_LINALG_MATRIX_POWER:
  results[i] = redispatch_ptrs_53[op.id - H__FW_PRIMAL](ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state));
  break;

case H_ROT90:
  results[i] = at::redispatch::rot90(ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state));
  break;

case H_UNFLATTEN_INT:
  results[i] = at::redispatch::unflatten(ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<c10::optional<at::DimnameList>>()(op.args[3], load_state));
  break;

case H_HISTC:
  results[i] = at::redispatch::histc(ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<at::Scalar &>()(op.args[2], load_state), load<at::Scalar &>()(op.args[3], load_state));
  break;

case H_INDEX_SELECT:
  results[i] = at::redispatch::index_select(ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state));
  break;

case H_VALUE_SELECTING_REDUCTION_BACKWARD:
  results[i] = at::redispatch::value_selecting_reduction_backward(ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::IntArrayRef>()(op.args[3], load_state), load<bool>()(op.args[4], load_state));
  break;

case H_INDEX_FILL_INT_SCALAR:
case H_SCATTER_VALUE:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, int64_t, const at::Tensor &, const at::Scalar &) = at::redispatch::index_fill;
  if (op.id == H_SCATTER_VALUE) ptr = at::redispatch::scatter;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::Scalar &>()(op.args[3], load_state));
  break;}

case H_SCATTER_VALUE_REDUCE:
  results[i] = at::redispatch::scatter(ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::Scalar &>()(op.args[3], load_state), load<c10::string_view>()(op.args[4], load_state));
  break;

case H_INDEX_COPY:
case H_INDEX_ADD:
case H_INDEX_FILL_INT_TENSOR:
case H_SCATTER_SRC:
case H_SCATTER_ADD:
case H__GATHER_SPARSE_BACKWARD:
  results[i] = redispatch_ptrs_54[op.id - H_INDEX_COPY](ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state));
  break;

case H_INDEX_ADD_ALPHA:
  results[i] = at::redispatch::index_add(ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state), load<at::Scalar &>()(op.args[4], load_state));
  break;

case H_SCATTER_REDUCE:
  results[i] = at::redispatch::scatter(ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state), load<c10::string_view>()(op.args[4], load_state));
  break;

case H_GATHER:
  results[i] = at::redispatch::gather(ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<bool>()(op.args[3], load_state));
  break;

case H__SCATTER_REDUCE:
  results[i] = at::redispatch::_scatter_reduce(ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<c10::string_view>()(op.args[3], load_state), load<c10::optional<int64_t>>()(op.args[4], load_state));
  break;

case H_NARROW_TENSOR:
  results[i] = at::redispatch::narrow(ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<int64_t>()(op.args[3], load_state));
  break;

case H_ALL_DIM:
case H_ANY_DIM:
case H__LOG_SOFTMAX:
case H__SOFTMAX:
case H__SPARSE_SOFTMAX:
case H__SPARSE_LOG_SOFTMAX:
case H_COMBINATIONS:
case H_ARGSORT:
case H__CONVERT_INDICES_FROM_COO_TO_CSR:
  results[i] = redispatch_ptrs_55[op.id - H_ALL_DIM](ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<bool>()(op.args[2], load_state));
  break;

case H_MULTINOMIAL:
  results[i] = at::redispatch::multinomial(ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<bool>()(op.args[2], load_state), load<c10::optional<at::Generator>>()(op.args[3], load_state));
  break;

case H_PROD_DIM_INT:
  results[i] = at::redispatch::prod(ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<bool>()(op.args[2], load_state), load<c10::optional<at::ScalarType>>()(op.args[3], load_state));
  break;

case H_CUMPROD:
case H_CUMSUM:
case H_LOG_SOFTMAX_INT:
case H_SOFTMAX_INT:
case H__SPARSE_SOFTMAX_INT:
case H__SPARSE_LOG_SOFTMAX_INT:
case H_SPECIAL_LOG_SOFTMAX:
case H_SPECIAL_SOFTMAX:
  results[i] = redispatch_ptrs_56[op.id - H_CUMPROD](ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<c10::optional<at::ScalarType>>()(op.args[2], load_state));
  break;

case H_RANDINT_LIKE:
  results[i] = at::redispatch::randint_like(ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<c10::optional<at::ScalarType>>()(op.args[2], load_state), load<c10::optional<at::Layout>>()(op.args[3], load_state), load<c10::optional<at::Device>>()(op.args[4], load_state), load<c10::optional<bool>>()(op.args[5], load_state), load<c10::optional<at::MemoryFormat>>()(op.args[6], load_state));
  break;

case H_COV:
  results[i] = at::redispatch::cov(ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<c10::optional<at::Tensor> &>()(op.args[2], load_state), load<c10::optional<at::Tensor> &>()(op.args[3], load_state));
  break;

case H_GROUP_NORM:
  results[i] = at::redispatch::group_norm(ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<c10::optional<at::Tensor> &>()(op.args[2], load_state), load<c10::optional<at::Tensor> &>()(op.args[3], load_state), load<double>()(op.args[4], load_state), load<bool>()(op.args[5], load_state));
  break;

case H_REPEAT_INTERLEAVE_SELF_INT:
  results[i] = at::redispatch::repeat_interleave(ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<c10::optional<int64_t>>()(op.args[2], load_state), load<c10::optional<int64_t>>()(op.args[3], load_state));
  break;

case H_ISTFT:
  results[i] = at::redispatch::istft(ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<c10::optional<int64_t>>()(op.args[2], load_state), load<c10::optional<int64_t>>()(op.args[3], load_state), load<c10::optional<at::Tensor> &>()(op.args[4], load_state), load<bool>()(op.args[5], load_state), load<bool>()(op.args[6], load_state), load<c10::optional<bool>>()(op.args[7], load_state), load<c10::optional<int64_t>>()(op.args[8], load_state), load<bool>()(op.args[9], load_state));
  break;

case H_STFT:
  results[i] = at::redispatch::stft(ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<c10::optional<int64_t>>()(op.args[2], load_state), load<c10::optional<int64_t>>()(op.args[3], load_state), load<c10::optional<at::Tensor> &>()(op.args[4], load_state), load<bool>()(op.args[5], load_state), load<c10::optional<bool>>()(op.args[6], load_state), load<c10::optional<bool>>()(op.args[7], load_state));
  break;

case H_SLICE_TENSOR:
  results[i] = at::redispatch::slice(ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<c10::optional<int64_t>>()(op.args[2], load_state), load<c10::optional<int64_t>>()(op.args[3], load_state), load<int64_t>()(op.args[4], load_state));
  break;

case H__TEST_AMBIGUOUS_DEFAULTS_B:
  results[i] = at::redispatch::_test_ambiguous_defaults(ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<c10::string_view>()(op.args[2], load_state));
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
  results[i] = redispatch_ptrs_57[op.id - H_FLATTEN_USING_INTS](ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<int64_t>()(op.args[2], load_state));
  break;

case H_FLATTEN_NAMED_OUT_DIM:
  results[i] = at::redispatch::flatten(ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<int64_t>()(op.args[2], load_state), load<at::Dimname>()(op.args[3], load_state));
  break;

case H_RANDINT_LIKE_LOW_DTYPE:
  results[i] = at::redispatch::randint_like(ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<int64_t>()(op.args[2], load_state), load<c10::optional<at::ScalarType>>()(op.args[3], load_state), load<c10::optional<at::Layout>>()(op.args[4], load_state), load<c10::optional<at::Device>>()(op.args[5], load_state), load<c10::optional<bool>>()(op.args[6], load_state), load<c10::optional<at::MemoryFormat>>()(op.args[7], load_state));
  break;

case H_DIFF:
  results[i] = at::redispatch::diff(ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<int64_t>()(op.args[2], load_state), load<c10::optional<at::Tensor> &>()(op.args[3], load_state), load<c10::optional<at::Tensor> &>()(op.args[4], load_state));
  break;

case H_DIAG_EMBED:
case H_DIAGONAL:
case H_NARROW_COPY:
case H_NARROW:
case H_UNFOLD:
case H__REMOVE_BATCH_DIM:
  results[i] = redispatch_ptrs_58[op.id - H_DIAG_EMBED](ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<int64_t>()(op.args[2], load_state), load<int64_t>()(op.args[3], load_state));
  break;

case H_CUDNN_AFFINE_GRID_GENERATOR:
case H_CUDNN_AFFINE_GRID_GENERATOR_BACKWARD:
  {at::Tensor(*ptr)(DispatchKeySet, const at::Tensor &, int64_t, int64_t, int64_t, int64_t) = at::redispatch::cudnn_affine_grid_generator;
  if (op.id == H_CUDNN_AFFINE_GRID_GENERATOR_BACKWARD) ptr = at::redispatch::cudnn_affine_grid_generator_backward;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<int64_t>()(op.args[2], load_state), load<int64_t>()(op.args[3], load_state), load<int64_t>()(op.args[4], load_state));
  break;}

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
  results[i] = redispatch_ptrs_59[op.id - H_BLOCK_DIAG](ks, load<at::TensorList>()(op.args[0], load_state));
  break;

case H_CAT_NAMES:
case H_CONCAT_NAMES:
  {at::Tensor(*ptr)(DispatchKeySet, at::TensorList, at::Dimname) = at::redispatch::cat;
  if (op.id == H_CONCAT_NAMES) ptr = at::redispatch::concat;
  results[i] = ptr(ks, load<at::TensorList>()(op.args[0], load_state), load<at::Dimname>()(op.args[1], load_state));
  break;}

case H_PAD_SEQUENCE:
  results[i] = at::redispatch::pad_sequence(ks, load<at::TensorList>()(op.args[0], load_state), load<bool>()(op.args[1], load_state), load<double>()(op.args[2], load_state));
  break;

case H_CAT:
case H_CONCAT:
case H_STACK:
case H__STACK:
case H__CAT:
  results[i] = redispatch_ptrs_60[op.id - H_CAT](ks, load<at::TensorList>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state));
  break;

case H__CUDNN_RNN_FLATTEN_WEIGHT:
  results[i] = at::redispatch::_cudnn_rnn_flatten_weight(ks, load<at::TensorList>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<int64_t>()(op.args[2], load_state), load<int64_t>()(op.args[3], load_state), load<int64_t>()(op.args[4], load_state), load<int64_t>()(op.args[5], load_state), load<int64_t>()(op.args[6], load_state), load<bool>()(op.args[7], load_state), load<bool>()(op.args[8], load_state));
  break;

case H_EINSUM:
  results[i] = at::redispatch::einsum(ks, load<c10::string_view>()(op.args[0], load_state), load<at::TensorList>()(op.args[1], load_state));
  break;

case H_FROM_FILE:
  results[i] = at::redispatch::from_file(ks, load<c10::string_view>()(op.args[0], load_state), load<c10::optional<bool>>()(op.args[1], load_state), load<c10::optional<int64_t>>()(op.args[2], load_state), load<c10::optional<at::ScalarType>>()(op.args[3], load_state), load<c10::optional<at::Layout>>()(op.args[4], load_state), load<c10::optional<at::Device>>()(op.args[5], load_state), load<c10::optional<bool>>()(op.args[6], load_state));
  break;

case H_NORMAL_FLOAT_TENSOR:
  results[i] = at::redispatch::normal(ks, load<double>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<c10::optional<at::Generator>>()(op.args[2], load_state));
  break;

case H__CUDNN_INIT_DROPOUT_STATE:
  results[i] = at::redispatch::_cudnn_init_dropout_state(ks, load<double>()(op.args[0], load_state), load<bool>()(op.args[1], load_state), load<int64_t>()(op.args[2], load_state), load<c10::optional<at::ScalarType>>()(op.args[3], load_state), load<c10::optional<at::Layout>>()(op.args[4], load_state), load<c10::optional<at::Device>>()(op.args[5], load_state), load<c10::optional<bool>>()(op.args[6], load_state));
  break;

case H_NORMAL_FLOAT_FLOAT:
  results[i] = at::redispatch::normal(ks, load<double>()(op.args[0], load_state), load<double>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<c10::optional<at::Generator>>()(op.args[3], load_state), load<c10::optional<at::ScalarType>>()(op.args[4], load_state), load<c10::optional<at::Layout>>()(op.args[5], load_state), load<c10::optional<at::Device>>()(op.args[6], load_state), load<c10::optional<bool>>()(op.args[7], load_state));
  break;

case H_RANDINT_GENERATOR:
  results[i] = at::redispatch::randint(ks, load<int64_t>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<c10::optional<at::Generator>>()(op.args[2], load_state), load<c10::optional<at::ScalarType>>()(op.args[3], load_state), load<c10::optional<at::Layout>>()(op.args[4], load_state), load<c10::optional<at::Device>>()(op.args[5], load_state), load<c10::optional<bool>>()(op.args[6], load_state));
  break;

case H_RANDINT:
  results[i] = at::redispatch::randint(ks, load<int64_t>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<c10::optional<at::ScalarType>>()(op.args[2], load_state), load<c10::optional<at::Layout>>()(op.args[3], load_state), load<c10::optional<at::Device>>()(op.args[4], load_state), load<c10::optional<bool>>()(op.args[5], load_state));
  break;

case H_POLYGAMMA:
case H_SPECIAL_POLYGAMMA:
  {at::Tensor(*ptr)(DispatchKeySet, int64_t, const at::Tensor &) = at::redispatch::polygamma;
  if (op.id == H_SPECIAL_POLYGAMMA) ptr = at::redispatch::special_polygamma;
  results[i] = ptr(ks, load<int64_t>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state));
  break;}

case H_BARTLETT_WINDOW_PERIODIC:
case H_BLACKMAN_WINDOW_PERIODIC:
case H_HANN_WINDOW_PERIODIC:
case H_HAMMING_WINDOW_PERIODIC:
case H_KAISER_WINDOW_PERIODIC:
  results[i] = redispatch_ptrs_61[op.id - H_BARTLETT_WINDOW_PERIODIC](ks, load<int64_t>()(op.args[0], load_state), load<bool>()(op.args[1], load_state), load<c10::optional<at::ScalarType>>()(op.args[2], load_state), load<c10::optional<at::Layout>>()(op.args[3], load_state), load<c10::optional<at::Device>>()(op.args[4], load_state), load<c10::optional<bool>>()(op.args[5], load_state));
  break;

case H_HAMMING_WINDOW_PERIODIC_ALPHA:
case H_KAISER_WINDOW_BETA:
  {at::Tensor(*ptr)(DispatchKeySet, int64_t, bool, double, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<at::Device>, c10::optional<bool>) = at::redispatch::hamming_window;
  if (op.id == H_KAISER_WINDOW_BETA) ptr = at::redispatch::kaiser_window;
  results[i] = ptr(ks, load<int64_t>()(op.args[0], load_state), load<bool>()(op.args[1], load_state), load<double>()(op.args[2], load_state), load<c10::optional<at::ScalarType>>()(op.args[3], load_state), load<c10::optional<at::Layout>>()(op.args[4], load_state), load<c10::optional<at::Device>>()(op.args[5], load_state), load<c10::optional<bool>>()(op.args[6], load_state));
  break;}

case H_HAMMING_WINDOW_PERIODIC_ALPHA_BETA:
  results[i] = at::redispatch::hamming_window(ks, load<int64_t>()(op.args[0], load_state), load<bool>()(op.args[1], load_state), load<double>()(op.args[2], load_state), load<double>()(op.args[3], load_state), load<c10::optional<at::ScalarType>>()(op.args[4], load_state), load<c10::optional<at::Layout>>()(op.args[5], load_state), load<c10::optional<at::Device>>()(op.args[6], load_state), load<c10::optional<bool>>()(op.args[7], load_state));
  break;

case H_RANDPERM_GENERATOR:
  results[i] = at::redispatch::randperm(ks, load<int64_t>()(op.args[0], load_state), load<c10::optional<at::Generator>>()(op.args[1], load_state), load<c10::optional<at::ScalarType>>()(op.args[2], load_state), load<c10::optional<at::Layout>>()(op.args[3], load_state), load<c10::optional<at::Device>>()(op.args[4], load_state), load<c10::optional<bool>>()(op.args[5], load_state));
  break;

case H_BARTLETT_WINDOW:
case H_BLACKMAN_WINDOW:
case H_EYE:
case H_HANN_WINDOW:
case H_HAMMING_WINDOW:
case H_KAISER_WINDOW:
case H_RANDPERM:
  results[i] = redispatch_ptrs_62[op.id - H_BARTLETT_WINDOW](ks, load<int64_t>()(op.args[0], load_state), load<c10::optional<at::ScalarType>>()(op.args[1], load_state), load<c10::optional<at::Layout>>()(op.args[2], load_state), load<c10::optional<at::Device>>()(op.args[3], load_state), load<c10::optional<bool>>()(op.args[4], load_state));
  break;

case H_FFT_FFTFREQ:
case H_FFT_RFFTFREQ:
  {at::Tensor(*ptr)(DispatchKeySet, int64_t, double, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<at::Device>, c10::optional<bool>) = at::redispatch::fft_fftfreq;
  if (op.id == H_FFT_RFFTFREQ) ptr = at::redispatch::fft_rfftfreq;
  results[i] = ptr(ks, load<int64_t>()(op.args[0], load_state), load<double>()(op.args[1], load_state), load<c10::optional<at::ScalarType>>()(op.args[2], load_state), load<c10::optional<at::Layout>>()(op.args[3], load_state), load<c10::optional<at::Device>>()(op.args[4], load_state), load<c10::optional<bool>>()(op.args[5], load_state));
  break;}

case H__SPARSE_COO_TENSOR_WITH_DIMS_AND_TENSORS:
  results[i] = at::redispatch::_sparse_coo_tensor_with_dims_and_tensors(ks, load<int64_t>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state), load<c10::optional<at::ScalarType>>()(op.args[5], load_state), load<c10::optional<at::Layout>>()(op.args[6], load_state), load<c10::optional<at::Device>>()(op.args[7], load_state), load<c10::optional<bool>>()(op.args[8], load_state));
  break;

case H_RANDINT_LOW_GENERATOR:
  results[i] = at::redispatch::randint(ks, load<int64_t>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<c10::optional<at::Generator>>()(op.args[3], load_state), load<c10::optional<at::ScalarType>>()(op.args[4], load_state), load<c10::optional<at::Layout>>()(op.args[5], load_state), load<c10::optional<at::Device>>()(op.args[6], load_state), load<c10::optional<bool>>()(op.args[7], load_state));
  break;

case H_RANDINT_LOW:
case H__SPARSE_COO_TENSOR_WITH_DIMS:
  {at::Tensor(*ptr)(DispatchKeySet, int64_t, int64_t, at::IntArrayRef, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<at::Device>, c10::optional<bool>) = at::redispatch::randint;
  if (op.id == H__SPARSE_COO_TENSOR_WITH_DIMS) ptr = at::redispatch::_sparse_coo_tensor_with_dims;
  results[i] = ptr(ks, load<int64_t>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<c10::optional<at::ScalarType>>()(op.args[3], load_state), load<c10::optional<at::Layout>>()(op.args[4], load_state), load<c10::optional<at::Device>>()(op.args[5], load_state), load<c10::optional<bool>>()(op.args[6], load_state));
  break;}

case H_EYE_M:
  results[i] = at::redispatch::eye(ks, load<int64_t>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<c10::optional<at::ScalarType>>()(op.args[2], load_state), load<c10::optional<at::Layout>>()(op.args[3], load_state), load<c10::optional<at::Device>>()(op.args[4], load_state), load<c10::optional<bool>>()(op.args[5], load_state));
  break;

case H_TRIL_INDICES:
case H_TRIU_INDICES:
  {at::Tensor(*ptr)(DispatchKeySet, int64_t, int64_t, int64_t, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<at::Device>, c10::optional<bool>) = at::redispatch::tril_indices;
  if (op.id == H_TRIU_INDICES) ptr = at::redispatch::triu_indices;
  results[i] = ptr(ks, load<int64_t>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<int64_t>()(op.args[2], load_state), load<c10::optional<at::ScalarType>>()(op.args[3], load_state), load<c10::optional<at::Layout>>()(op.args[4], load_state), load<c10::optional<at::Device>>()(op.args[5], load_state), load<c10::optional<bool>>()(op.args[6], load_state));
  break;}

case H_FULL_OUT:
  results[i] = at::redispatch::full_outf(ks, load<at::IntArrayRef>()(op.args[0], load_state), load<at::Scalar &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state));
  break;

case H_ONES_OUT:
case H_RAND_OUT:
case H_RANDN_OUT:
case H_ZEROS_OUT:
  results[i] = redispatch_ptrs_63[op.id - H_ONES_OUT](ks, load<at::IntArrayRef>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state));
  break;

case H_RAND_GENERATOR_OUT:
case H_RANDN_GENERATOR_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, at::IntArrayRef, c10::optional<at::Generator>, at::Tensor &) = at::redispatch::rand_outf;
  if (op.id == H_RANDN_GENERATOR_OUT) ptr = at::redispatch::randn_outf;
  results[i] = ptr(ks, load<at::IntArrayRef>()(op.args[0], load_state), load<c10::optional<at::Generator>>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state));
  break;}

case H_EMPTY_OUT:
  results[i] = at::redispatch::empty_outf(ks, load<at::IntArrayRef>()(op.args[0], load_state), load<c10::optional<at::MemoryFormat>>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state));
  break;

case H_ARANGE_START_OUT:
case H_RANGE_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Scalar &, const at::Scalar &, const at::Scalar &, at::Tensor &) = at::redispatch::arange_outf;
  if (op.id == H_RANGE_OUT) ptr = at::redispatch::range_outf;
  results[i] = ptr(ks, load<at::Scalar &>()(op.args[0], load_state), load<at::Scalar &>()(op.args[1], load_state), load<at::Scalar &>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state));
  break;}

case H_LINSPACE_OUT:
  results[i] = at::redispatch::linspace_outf(ks, load<at::Scalar &>()(op.args[0], load_state), load<at::Scalar &>()(op.args[1], load_state), load<c10::optional<int64_t>>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state));
  break;

case H_LOGSPACE_OUT:
  results[i] = at::redispatch::logspace_outf(ks, load<at::Scalar &>()(op.args[0], load_state), load<at::Scalar &>()(op.args[1], load_state), load<c10::optional<int64_t>>()(op.args[2], load_state), load<double>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;

case H_ARANGE_OUT:
  results[i] = at::redispatch::arange_outf(ks, load<at::Scalar &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state));
  break;

case H_XLOGY_OUTSCALAR_SELF:
case H_POW_SCALAR_OUT:
case H_FLOAT_POWER_SCALAR_OUT:
case H_SPECIAL_XLOG1PY_SELF_SCALAR_OUT:
case H_SPECIAL_XLOGY_SELF_SCALAR_OUT:
case H_SPECIAL_ZETA_SELF_SCALAR_OUT:
  results[i] = redispatch_ptrs_64[op.id - H_XLOGY_OUTSCALAR_SELF](ks, load<at::Scalar &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state));
  break;

case H_ISIN_SCALAR_TENSOR_OUT:
  results[i] = at::redispatch::isin_outf(ks, load<at::Scalar &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<bool>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;

case H_ABS_:
case H_ABSOLUTE_:
case H_SGN_:
case H_CONJ_PHYSICAL_:
case H_ACOS_:
case H_ARCCOS_:
case H_ACOSH_:
case H_ARCCOSH_:
case H_ASINH_:
case H_ARCSINH_:
case H_ATANH_:
case H_ARCTANH_:
case H_ASIN_:
case H_ARCSIN_:
case H_ATAN_:
case H_ARCTAN_:
case H_BITWISE_NOT_:
case H_LOGICAL_NOT_:
case H_CEIL_:
case H_COS_:
case H_COSH_:
case H_ERF_:
case H_ERFC_:
case H_EXP_:
case H_EXP2_:
case H_EXPM1_:
case H_FLOOR_:
case H_FRAC_:
case H_LOG_:
case H_LOG10_:
case H_LOG1P_:
case H_LOG2_:
case H_RAD2DEG_:
case H_DEG2RAD_:
case H_RECIPROCAL_:
case H_NEG_:
case H_NEGATIVE_:
case H_ROUND_:
case H_RELU_:
case H_RELU6_:
case H_RSQRT_:
case H_SELU_:
case H_SILU_:
case H_MISH_:
case H_SIGMOID_:
case H_SIN_:
case H_SINC_:
case H_SINH_:
case H_DETACH_:
case H_SQUEEZE_:
case H_SQRT_:
case H_SQUARE_:
case H_T_:
case H_TAN_:
case H_TANH_:
case H_TRUNC_:
case H_FIX_:
case H_ZERO_:
case H_SET_:
case H_DIGAMMA_:
case H_LGAMMA_:
case H_ERFINV_:
case H_I0_:
case H_SIGN_:
case H_HARDSIGMOID_:
case H_HARDSWISH_:
  results[i] = redispatch_ptrs_65[op.id - H_ABS_](ks, load<at::Tensor &>()(op.args[0], load_state));
  break;

case H_SQUEEZE__DIMNAME:
  results[i] = at::redispatch::squeeze_(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Dimname>()(op.args[1], load_state));
  break;

case H_LOGCUMSUMEXP_DIMNAME_OUT:
  results[i] = at::redispatch::logcumsumexp_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Dimname>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state));
  break;

case H_INDEX_FILL__DIMNAME_SCALAR:
  results[i] = at::redispatch::index_fill_(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Dimname>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::Scalar &>()(op.args[3], load_state));
  break;

case H_INDEX_COPY__DIMNAME:
case H_INDEX_FILL__DIMNAME_TENSOR:
  {at::Tensor &(*ptr)(DispatchKeySet, at::Tensor &, at::Dimname, const at::Tensor &, const at::Tensor &) = at::redispatch::index_copy_;
  if (op.id == H_INDEX_FILL__DIMNAME_TENSOR) ptr = at::redispatch::index_fill_;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Dimname>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state));
  break;}

case H_INDEX_SELECT_DIMNAME_OUT:
  results[i] = at::redispatch::index_select_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Dimname>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state));
  break;

case H_GATHER_DIMNAME_OUT:
  results[i] = at::redispatch::gather_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Dimname>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;

case H_ALL_DIMNAME_OUT:
case H_ANY_DIMNAME_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, at::Dimname, bool, at::Tensor &) = at::redispatch::all_outf;
  if (op.id == H_ANY_DIMNAME_OUT) ptr = at::redispatch::any_outf;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Dimname>()(op.args[1], load_state), load<bool>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state));
  break;}

case H_PROD_DIMNAME_OUT:
  results[i] = at::redispatch::prod_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Dimname>()(op.args[1], load_state), load<bool>()(op.args[2], load_state), load<c10::optional<at::ScalarType>>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;

case H_CUMPROD__DIMNAME:
case H_CUMSUM__DIMNAME:
  {at::Tensor &(*ptr)(DispatchKeySet, at::Tensor &, at::Dimname, c10::optional<at::ScalarType>) = at::redispatch::cumprod_;
  if (op.id == H_CUMSUM__DIMNAME) ptr = at::redispatch::cumsum_;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Dimname>()(op.args[1], load_state), load<c10::optional<at::ScalarType>>()(op.args[2], load_state));
  break;}

case H_CUMPROD_DIMNAME_OUT:
case H_CUMSUM_DIMNAME_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, at::Dimname, c10::optional<at::ScalarType>, at::Tensor &) = at::redispatch::cumprod_outf;
  if (op.id == H_CUMSUM_DIMNAME_OUT) ptr = at::redispatch::cumsum_outf;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Dimname>()(op.args[1], load_state), load<c10::optional<at::ScalarType>>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state));
  break;}

case H_LOGSUMEXP_NAMES_OUT:
  results[i] = at::redispatch::logsumexp_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::DimnameList>()(op.args[1], load_state), load<bool>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state));
  break;

case H_STD_NAMES_OUT:
case H_VAR_NAMES_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, at::DimnameList, bool, bool, at::Tensor &) = at::redispatch::std_outf;
  if (op.id == H_VAR_NAMES_OUT) ptr = at::redispatch::var_outf;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::DimnameList>()(op.args[1], load_state), load<bool>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;}

case H_MEAN_NAMES_OUT:
case H_SUM_DIMNAMELIST_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, at::DimnameList, bool, c10::optional<at::ScalarType>, at::Tensor &) = at::redispatch::mean_outf;
  if (op.id == H_SUM_DIMNAMELIST_OUT) ptr = at::redispatch::sum_outf;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::DimnameList>()(op.args[1], load_state), load<bool>()(op.args[2], load_state), load<c10::optional<at::ScalarType>>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;}

case H_STD_CORRECTION_NAMES_OUT:
case H_VAR_CORRECTION_NAMES_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, at::DimnameList, c10::optional<int64_t>, bool, at::Tensor &) = at::redispatch::std_outf;
  if (op.id == H_VAR_CORRECTION_NAMES_OUT) ptr = at::redispatch::var_outf;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::DimnameList>()(op.args[1], load_state), load<c10::optional<int64_t>>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;}

case H_COL2IM_OUT:
case H_IM2COL_BACKWARD_GRAD_INPUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::Tensor &) = at::redispatch::col2im_outf;
  if (op.id == H_IM2COL_BACKWARD_GRAD_INPUT) ptr = at::redispatch::im2col_backward_outf;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<at::IntArrayRef>()(op.args[3], load_state), load<at::IntArrayRef>()(op.args[4], load_state), load<at::IntArrayRef>()(op.args[5], load_state), load<at::Tensor &>()(op.args[6], load_state));
  break;}

case H_COL2IM_BACKWARD_GRAD_INPUT:
case H_IM2COL_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::Tensor &) = at::redispatch::col2im_backward_outf;
  if (op.id == H_IM2COL_OUT) ptr = at::redispatch::im2col_outf;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<at::IntArrayRef>()(op.args[3], load_state), load<at::IntArrayRef>()(op.args[4], load_state), load<at::Tensor &>()(op.args[5], load_state));
  break;}

case H_AVG_POOL2D_OUT:
case H_AVG_POOL3D_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, bool, bool, c10::optional<int64_t>, at::Tensor &) = at::redispatch::avg_pool2d_outf;
  if (op.id == H_AVG_POOL3D_OUT) ptr = at::redispatch::avg_pool3d_outf;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<at::IntArrayRef>()(op.args[3], load_state), load<bool>()(op.args[4], load_state), load<bool>()(op.args[5], load_state), load<c10::optional<int64_t>>()(op.args[6], load_state), load<at::Tensor &>()(op.args[7], load_state));
  break;}

case H_UPSAMPLE_LINEAR1D_BACKWARD_GRAD_INPUT:
  results[i] = at::redispatch::upsample_linear1d_backward_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<c10::optional<double>>()(op.args[4], load_state), load<at::Tensor &>()(op.args[5], load_state));
  break;

case H_UPSAMPLE_BILINEAR2D_BACKWARD_GRAD_INPUT:
case H__UPSAMPLE_BILINEAR2D_AA_BACKWARD_GRAD_INPUT:
case H_UPSAMPLE_BICUBIC2D_BACKWARD_GRAD_INPUT:
  results[i] = redispatch_ptrs_66[op.id - H_UPSAMPLE_BILINEAR2D_BACKWARD_GRAD_INPUT](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<c10::optional<double>>()(op.args[4], load_state), load<c10::optional<double>>()(op.args[5], load_state), load<at::Tensor &>()(op.args[6], load_state));
  break;

case H_UPSAMPLE_TRILINEAR3D_BACKWARD_GRAD_INPUT:
  results[i] = at::redispatch::upsample_trilinear3d_backward_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<c10::optional<double>>()(op.args[4], load_state), load<c10::optional<double>>()(op.args[5], load_state), load<c10::optional<double>>()(op.args[6], load_state), load<at::Tensor &>()(op.args[7], load_state));
  break;

case H_UPSAMPLE_NEAREST1D_BACKWARD_GRAD_INPUT:
case H__UPSAMPLE_NEAREST_EXACT1D_BACKWARD_GRAD_INPUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, c10::optional<double>, at::Tensor &) = at::redispatch::upsample_nearest1d_backward_outf;
  if (op.id == H__UPSAMPLE_NEAREST_EXACT1D_BACKWARD_GRAD_INPUT) ptr = at::redispatch::_upsample_nearest_exact1d_backward_outf;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<c10::optional<double>>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;}

case H_UPSAMPLE_NEAREST2D_BACKWARD_GRAD_INPUT:
case H__UPSAMPLE_NEAREST_EXACT2D_BACKWARD_GRAD_INPUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, c10::optional<double>, c10::optional<double>, at::Tensor &) = at::redispatch::upsample_nearest2d_backward_outf;
  if (op.id == H__UPSAMPLE_NEAREST_EXACT2D_BACKWARD_GRAD_INPUT) ptr = at::redispatch::_upsample_nearest_exact2d_backward_outf;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<c10::optional<double>>()(op.args[3], load_state), load<c10::optional<double>>()(op.args[4], load_state), load<at::Tensor &>()(op.args[5], load_state));
  break;}

case H_UPSAMPLE_NEAREST3D_BACKWARD_GRAD_INPUT:
case H__UPSAMPLE_NEAREST_EXACT3D_BACKWARD_GRAD_INPUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, c10::optional<double>, c10::optional<double>, c10::optional<double>, at::Tensor &) = at::redispatch::upsample_nearest3d_backward_outf;
  if (op.id == H__UPSAMPLE_NEAREST_EXACT3D_BACKWARD_GRAD_INPUT) ptr = at::redispatch::_upsample_nearest_exact3d_backward_outf;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<c10::optional<double>>()(op.args[3], load_state), load<c10::optional<double>>()(op.args[4], load_state), load<c10::optional<double>>()(op.args[5], load_state), load<at::Tensor &>()(op.args[6], load_state));
  break;}

case H_AS_STRIDED_:
  results[i] = at::redispatch::as_strided_(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<c10::optional<int64_t>>()(op.args[3], load_state));
  break;

case H_ADAPTIVE_AVG_POOL2D_OUT:
case H_ADAPTIVE_AVG_POOL3D_OUT:
case H_REFLECTION_PAD1D_OUT:
case H_REFLECTION_PAD2D_OUT:
case H_REFLECTION_PAD3D_OUT:
case H_REPLICATION_PAD1D_OUT:
case H_REPLICATION_PAD2D_OUT:
case H_REPLICATION_PAD3D_OUT:
  results[i] = redispatch_ptrs_67[op.id - H_ADAPTIVE_AVG_POOL2D_OUT](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state));
  break;

case H_LOGSUMEXP_OUT:
case H_AMAX_OUT:
case H_AMIN_OUT:
case H_FROBENIUS_NORM_OUT:
case H_NUCLEAR_NORM_DIM_OUT:
case H_SPECIAL_LOGSUMEXP_OUT:
  results[i] = redispatch_ptrs_68[op.id - H_LOGSUMEXP_OUT](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<bool>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state));
  break;

case H_STD_OUT:
case H_VAR_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, at::IntArrayRef, bool, bool, at::Tensor &) = at::redispatch::std_outf;
  if (op.id == H_VAR_OUT) ptr = at::redispatch::var_outf;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<bool>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;}

case H_MEAN_OUT:
case H_NANMEAN_OUT:
case H_SUM_INTLIST_OUT:
case H_NANSUM_INTLIST_OUT:
  results[i] = redispatch_ptrs_69[op.id - H_MEAN_OUT](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<bool>()(op.args[2], load_state), load<c10::optional<at::ScalarType>>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;

case H_UPSAMPLE_LINEAR1D_OUT:
  results[i] = at::redispatch::upsample_linear1d_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<bool>()(op.args[2], load_state), load<c10::optional<double>>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;

case H_UPSAMPLE_BILINEAR2D_OUT:
case H__UPSAMPLE_BILINEAR2D_AA_OUT:
case H_UPSAMPLE_BICUBIC2D_OUT:
  results[i] = redispatch_ptrs_70[op.id - H_UPSAMPLE_BILINEAR2D_OUT](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<bool>()(op.args[2], load_state), load<c10::optional<double>>()(op.args[3], load_state), load<c10::optional<double>>()(op.args[4], load_state), load<at::Tensor &>()(op.args[5], load_state));
  break;

case H_UPSAMPLE_TRILINEAR3D_OUT:
  results[i] = at::redispatch::upsample_trilinear3d_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<bool>()(op.args[2], load_state), load<c10::optional<double>>()(op.args[3], load_state), load<c10::optional<double>>()(op.args[4], load_state), load<c10::optional<double>>()(op.args[5], load_state), load<at::Tensor &>()(op.args[6], load_state));
  break;

case H_RESIZE_:
  results[i] = at::redispatch::resize_(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<c10::optional<at::MemoryFormat>>()(op.args[2], load_state));
  break;

case H_UPSAMPLE_NEAREST1D_OUT:
case H__UPSAMPLE_NEAREST_EXACT1D_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, at::IntArrayRef, c10::optional<double>, at::Tensor &) = at::redispatch::upsample_nearest1d_outf;
  if (op.id == H__UPSAMPLE_NEAREST_EXACT1D_OUT) ptr = at::redispatch::_upsample_nearest_exact1d_outf;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<c10::optional<double>>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state));
  break;}

case H_UPSAMPLE_NEAREST2D_OUT:
case H__UPSAMPLE_NEAREST_EXACT2D_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, at::IntArrayRef, c10::optional<double>, c10::optional<double>, at::Tensor &) = at::redispatch::upsample_nearest2d_outf;
  if (op.id == H__UPSAMPLE_NEAREST_EXACT2D_OUT) ptr = at::redispatch::_upsample_nearest_exact2d_outf;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<c10::optional<double>>()(op.args[2], load_state), load<c10::optional<double>>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;}

case H_UPSAMPLE_NEAREST3D_OUT:
case H__UPSAMPLE_NEAREST_EXACT3D_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, at::IntArrayRef, c10::optional<double>, c10::optional<double>, c10::optional<double>, at::Tensor &) = at::redispatch::upsample_nearest3d_outf;
  if (op.id == H__UPSAMPLE_NEAREST_EXACT3D_OUT) ptr = at::redispatch::_upsample_nearest_exact3d_outf;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<c10::optional<double>>()(op.args[2], load_state), load<c10::optional<double>>()(op.args[3], load_state), load<c10::optional<double>>()(op.args[4], load_state), load<at::Tensor &>()(op.args[5], load_state));
  break;}

case H__FFT_R2C_OUT:
case H__FFT_C2C_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, at::IntArrayRef, int64_t, bool, at::Tensor &) = at::redispatch::_fft_r2c_outf;
  if (op.id == H__FFT_C2C_OUT) ptr = at::redispatch::_fft_c2c_outf;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<int64_t>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;}

case H_SPARSE_RESIZE_:
case H_SPARSE_RESIZE_AND_CLEAR_:
  {const at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, at::IntArrayRef, int64_t, int64_t) = at::redispatch::sparse_resize_;
  if (op.id == H_SPARSE_RESIZE_AND_CLEAR_) ptr = at::redispatch::sparse_resize_and_clear_;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<int64_t>()(op.args[2], load_state), load<int64_t>()(op.args[3], load_state));
  break;}

case H__FFT_C2R_OUT:
  results[i] = at::redispatch::_fft_c2r_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<int64_t>()(op.args[2], load_state), load<int64_t>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
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
case H_BITWISE_LEFT_SHIFT__TENSOR_SCALAR:
case H___IRSHIFT___SCALAR:
case H_BITWISE_RIGHT_SHIFT__TENSOR_SCALAR:
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
case H_FMOD__SCALAR:
case H_REMAINDER__SCALAR:
case H_POW__SCALAR:
case H_FLOAT_POWER__SCALAR:
case H_LEAKY_RELU_:
  results[i] = redispatch_ptrs_71[op.id - H_COPYSIGN__SCALAR](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Scalar &>()(op.args[1], load_state));
  break;

case H_LINALG_MATRIX_NORM_OUT:
  results[i] = at::redispatch::linalg_matrix_norm_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Scalar &>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<c10::optional<at::ScalarType>>()(op.args[4], load_state), load<at::Tensor &>()(op.args[5], load_state));
  break;

case H__ADD_RELU__SCALAR:
case H_ADD__SCALAR:
case H_THRESHOLD_:
case H_SUB__SCALAR:
case H_SUBTRACT__SCALAR:
case H_HARDTANH_:
  results[i] = redispatch_ptrs_72[op.id - H__ADD_RELU__SCALAR](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Scalar &>()(op.args[1], load_state), load<at::Scalar &>()(op.args[2], load_state));
  break;

case H_ELU_:
  results[i] = at::redispatch::elu_(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Scalar &>()(op.args[1], load_state), load<at::Scalar &>()(op.args[2], load_state), load<at::Scalar &>()(op.args[3], load_state));
  break;

case H_ELU_OUT:
  results[i] = at::redispatch::elu_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Scalar &>()(op.args[1], load_state), load<at::Scalar &>()(op.args[2], load_state), load<at::Scalar &>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;

case H_ELU_BACKWARD_GRAD_INPUT:
  results[i] = at::redispatch::elu_backward_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Scalar &>()(op.args[1], load_state), load<at::Scalar &>()(op.args[2], load_state), load<at::Scalar &>()(op.args[3], load_state), load<bool>()(op.args[4], load_state), load<at::Tensor &>()(op.args[5], load_state), load<at::Tensor &>()(op.args[6], load_state));
  break;

case H_THRESHOLD_OUT:
case H_HARDTANH_OUT:
case H_SOFTPLUS_OUT:
  results[i] = redispatch_ptrs_73[op.id - H_THRESHOLD_OUT](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Scalar &>()(op.args[1], load_state), load<at::Scalar &>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state));
  break;

case H_RRELU_:
  results[i] = at::redispatch::rrelu_(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Scalar &>()(op.args[1], load_state), load<at::Scalar &>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<c10::optional<at::Generator>>()(op.args[4], load_state));
  break;

case H_COPYSIGN_SCALAR_OUT:
case H_CLAMP_MAX_OUT:
case H_CLAMP_MIN_OUT:
case H_XLOGY_OUTSCALAR_OTHER:
case H_HARDSHRINK_OUT:
case H_BITWISE_AND_SCALAR_OUT:
case H_BITWISE_OR_SCALAR_OUT:
case H_BITWISE_XOR_SCALAR_OUT:
case H_BITWISE_LEFT_SHIFT_TENSOR_SCALAR_OUT:
case H_BITWISE_RIGHT_SHIFT_TENSOR_SCALAR_OUT:
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
case H_SPECIAL_XLOGY_OTHER_SCALAR_OUT:
case H_SPECIAL_ZETA_OTHER_SCALAR_OUT:
  results[i] = redispatch_ptrs_74[op.id - H_COPYSIGN_SCALAR_OUT](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Scalar &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state));
  break;

case H_FILL_DIAGONAL_:
  results[i] = at::redispatch::fill_diagonal_(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Scalar &>()(op.args[1], load_state), load<bool>()(op.args[2], load_state));
  break;

case H_ISIN_TENSOR_SCALAR_OUT:
  results[i] = at::redispatch::isin_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Scalar &>()(op.args[1], load_state), load<bool>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;

case H_LINALG_VECTOR_NORM_OUT:
  results[i] = at::redispatch::linalg_vector_norm_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Scalar &>()(op.args[1], load_state), load<c10::optional<at::IntArrayRef>>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<c10::optional<at::ScalarType>>()(op.args[4], load_state), load<at::Tensor &>()(op.args[5], load_state));
  break;

case H_DIV__SCALAR_MODE:
case H_DIVIDE__SCALAR_MODE:
  {at::Tensor &(*ptr)(DispatchKeySet, at::Tensor &, const at::Scalar &, c10::optional<c10::string_view>) = at::redispatch::div_;
  if (op.id == H_DIVIDE__SCALAR_MODE) ptr = at::redispatch::divide_;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Scalar &>()(op.args[1], load_state), load<c10::optional<c10::string_view>>()(op.args[2], load_state));
  break;}

case H_RENORM_:
  results[i] = at::redispatch::renorm_(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Scalar &>()(op.args[1], load_state), load<int64_t>()(op.args[2], load_state), load<at::Scalar &>()(op.args[3], load_state));
  break;

case H_RENORM_OUT:
  results[i] = at::redispatch::renorm_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Scalar &>()(op.args[1], load_state), load<int64_t>()(op.args[2], load_state), load<at::Scalar &>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;

case H_SET__SOURCE_STORAGE:
  results[i] = at::redispatch::set_(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Storage>()(op.args[1], load_state));
  break;

case H_SET__SOURCE_STORAGE_STORAGE_OFFSET:
  results[i] = at::redispatch::set_(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Storage>()(op.args[1], load_state), load<int64_t>()(op.args[2], load_state), load<at::IntArrayRef>()(op.args[3], load_state), load<at::IntArrayRef>()(op.args[4], load_state));
  break;

case H_COPYSIGN__TENSOR:
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
case H_GCD_:
case H_LCM_:
case H_LDEXP_:
case H_XLOGY__TENSOR:
case H_MUL__TENSOR:
case H_MULTIPLY__TENSOR:
case H_HEAVISIDE_:
case H_SET__SOURCE_TENSOR:
case H_EQ__TENSOR:
case H_BITWISE_AND__TENSOR:
case H___IAND___TENSOR:
case H_BITWISE_OR__TENSOR:
case H___IOR___TENSOR:
case H_BITWISE_XOR__TENSOR:
case H___IXOR___TENSOR:
case H___ILSHIFT___TENSOR:
case H_BITWISE_LEFT_SHIFT__TENSOR:
case H___IRSHIFT___TENSOR:
case H_BITWISE_RIGHT_SHIFT__TENSOR:
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
case H_ATAN2_:
case H_ARCTAN2_:
case H_FMOD__TENSOR:
case H_HYPOT_:
case H_IGAMMA_:
case H_IGAMMAC_:
case H_NEXTAFTER_:
case H_REMAINDER__TENSOR:
case H_POW__TENSOR:
case H_FLOAT_POWER__TENSOR:
  results[i] = redispatch_ptrs_75[op.id - H_COPYSIGN__TENSOR](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state));
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
case H_ALL_ALL_OUT:
case H_ANY_ALL_OUT:
case H_HARDSIGMOID_OUT:
case H_HARDSWISH_OUT:
case H_LOG_SIGMOID_OUT:
case H_ISPOSINF_OUT:
case H_ISNEGINF_OUT:
case H_SPECIAL_ENTR_OUT:
case H_SPECIAL_NDTRI_OUT:
case H_SPECIAL_EXPM1_OUT:
case H_SPECIAL_EXP2_OUT:
case H_SPECIAL_PSI_OUT:
case H_SPECIAL_DIGAMMA_OUT:
case H_SPECIAL_GAMMALN_OUT:
case H_SPECIAL_ERF_OUT:
case H_SPECIAL_ERFC_OUT:
case H_SPECIAL_ERFCX_OUT:
case H_SPECIAL_ERFINV_OUT:
case H_SPECIAL_NDTR_OUT:
case H_SPECIAL_I0_OUT:
case H_SPECIAL_I0E_OUT:
case H_SPECIAL_I1_OUT:
case H_SPECIAL_I1E_OUT:
case H_SPECIAL_EXPIT_OUT:
case H_SPECIAL_SINC_OUT:
case H_SPECIAL_ROUND_OUT:
case H_SPECIAL_LOG1P_OUT:
case H_LINALG_DET_OUT:
case H_LINALG_EIGVALS_OUT:
case H_LINALG_INV_OUT:
case H_LINALG_SVDVALS_OUT:
  results[i] = redispatch_ptrs_76[op.id - H_ABS_OUT](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state));
  break;

case H_RESIZE_AS_SPARSE_:
  results[i] = at::redispatch::resize_as_sparse_(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state));
  break;

case H_MAX_POOL2D_WITH_INDICES_BACKWARD_GRAD_INPUT:
case H_MAX_POOL3D_WITH_INDICES_BACKWARD_GRAD_INPUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, bool, const at::Tensor &, at::Tensor &) = at::redispatch::max_pool2d_with_indices_backward_outf;
  if (op.id == H_MAX_POOL3D_WITH_INDICES_BACKWARD_GRAD_INPUT) ptr = at::redispatch::max_pool3d_with_indices_backward_outf;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<at::IntArrayRef>()(op.args[3], load_state), load<at::IntArrayRef>()(op.args[4], load_state), load<at::IntArrayRef>()(op.args[5], load_state), load<bool>()(op.args[6], load_state), load<at::Tensor &>()(op.args[7], load_state), load<at::Tensor &>()(op.args[8], load_state));
  break;}

case H_MAX_UNPOOL3D_OUT:
  results[i] = at::redispatch::max_unpool3d_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<at::IntArrayRef>()(op.args[3], load_state), load<at::IntArrayRef>()(op.args[4], load_state), load<at::Tensor &>()(op.args[5], load_state));
  break;

case H_AVG_POOL2D_BACKWARD_GRAD_INPUT:
case H_AVG_POOL3D_BACKWARD_GRAD_INPUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, bool, bool, c10::optional<int64_t>, at::Tensor &) = at::redispatch::avg_pool2d_backward_outf;
  if (op.id == H_AVG_POOL3D_BACKWARD_GRAD_INPUT) ptr = at::redispatch::avg_pool3d_backward_outf;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<at::IntArrayRef>()(op.args[3], load_state), load<at::IntArrayRef>()(op.args[4], load_state), load<bool>()(op.args[5], load_state), load<bool>()(op.args[6], load_state), load<c10::optional<int64_t>>()(op.args[7], load_state), load<at::Tensor &>()(op.args[8], load_state));
  break;}

case H_TENSORDOT_OUT:
  results[i] = at::redispatch::tensordot_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<at::IntArrayRef>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;

case H_FRACTIONAL_MAX_POOL2D_BACKWARD_GRAD_INPUT:
case H_FRACTIONAL_MAX_POOL3D_BACKWARD_GRAD_INPUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, const at::Tensor &, at::Tensor &) = at::redispatch::fractional_max_pool2d_backward_outf;
  if (op.id == H_FRACTIONAL_MAX_POOL3D_BACKWARD_GRAD_INPUT) ptr = at::redispatch::fractional_max_pool3d_backward_outf;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<at::IntArrayRef>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state), load<at::Tensor &>()(op.args[5], load_state));
  break;}

case H_MAX_UNPOOL2D_OUT:
case H_REFLECTION_PAD1D_BACKWARD_GRAD_INPUT:
case H_REFLECTION_PAD2D_BACKWARD_GRAD_INPUT:
case H_REFLECTION_PAD3D_BACKWARD_GRAD_INPUT:
case H_REPLICATION_PAD1D_BACKWARD_GRAD_INPUT:
case H_REPLICATION_PAD2D_BACKWARD_GRAD_INPUT:
case H_REPLICATION_PAD3D_BACKWARD_GRAD_INPUT:
  results[i] = redispatch_ptrs_77[op.id - H_MAX_UNPOOL2D_OUT](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state));
  break;

case H_SLOW_CONV_TRANSPOSE2D_OUT:
case H_SLOW_CONV_TRANSPOSE3D_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, at::IntArrayRef, const c10::optional<at::Tensor> &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::Tensor &) = at::redispatch::slow_conv_transpose2d_outf;
  if (op.id == H_SLOW_CONV_TRANSPOSE3D_OUT) ptr = at::redispatch::slow_conv_transpose3d_outf;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<c10::optional<at::Tensor> &>()(op.args[3], load_state), load<at::IntArrayRef>()(op.args[4], load_state), load<at::IntArrayRef>()(op.args[5], load_state), load<at::IntArrayRef>()(op.args[6], load_state), load<at::IntArrayRef>()(op.args[7], load_state), load<at::Tensor &>()(op.args[8], load_state));
  break;}

case H__CONV_DEPTHWISE2D_OUT:
  results[i] = at::redispatch::_conv_depthwise2d_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<c10::optional<at::Tensor> &>()(op.args[3], load_state), load<at::IntArrayRef>()(op.args[4], load_state), load<at::IntArrayRef>()(op.args[5], load_state), load<at::IntArrayRef>()(op.args[6], load_state), load<at::Tensor &>()(op.args[7], load_state));
  break;

case H_THNN_CONV2D_OUT:
case H_SLOW_CONV3D_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, at::IntArrayRef, const c10::optional<at::Tensor> &, at::IntArrayRef, at::IntArrayRef, at::Tensor &) = at::redispatch::thnn_conv2d_outf;
  if (op.id == H_SLOW_CONV3D_OUT) ptr = at::redispatch::slow_conv3d_outf;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<c10::optional<at::Tensor> &>()(op.args[3], load_state), load<at::IntArrayRef>()(op.args[4], load_state), load<at::IntArrayRef>()(op.args[5], load_state), load<at::Tensor &>()(op.args[6], load_state));
  break;}

case H_ADD__TENSOR:
case H__ADD_RELU__TENSOR:
case H_SUB__TENSOR:
case H_SUBTRACT__TENSOR:
case H_MASKED_FILL__SCALAR:
case H_LERP__SCALAR:
  results[i] = redispatch_ptrs_78[op.id - H_ADD__TENSOR](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Scalar &>()(op.args[2], load_state));
  break;

case H_HARDTANH_BACKWARD_GRAD_INPUT:
  results[i] = at::redispatch::hardtanh_backward_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Scalar &>()(op.args[2], load_state), load<at::Scalar &>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;

case H_SOFTPLUS_BACKWARD_GRAD_INPUT:
  results[i] = at::redispatch::softplus_backward_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Scalar &>()(op.args[2], load_state), load<at::Scalar &>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state), load<at::Tensor &>()(op.args[5], load_state));
  break;

case H_RRELU_WITH_NOISE_:
  results[i] = at::redispatch::rrelu_with_noise_(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Scalar &>()(op.args[2], load_state), load<at::Scalar &>()(op.args[3], load_state), load<bool>()(op.args[4], load_state), load<c10::optional<at::Generator>>()(op.args[5], load_state));
  break;

case H_RRELU_WITH_NOISE_OUT:
  results[i] = at::redispatch::rrelu_with_noise_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Scalar &>()(op.args[2], load_state), load<at::Scalar &>()(op.args[3], load_state), load<bool>()(op.args[4], load_state), load<c10::optional<at::Generator>>()(op.args[5], load_state), load<at::Tensor &>()(op.args[6], load_state));
  break;

case H_MULTI_MARGIN_LOSS_OUT:
  results[i] = at::redispatch::multi_margin_loss_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Scalar &>()(op.args[2], load_state), load<at::Scalar &>()(op.args[3], load_state), load<c10::optional<at::Tensor> &>()(op.args[4], load_state), load<int64_t>()(op.args[5], load_state), load<at::Tensor &>()(op.args[6], load_state));
  break;

case H_ADD_OUT:
case H__ADD_RELU_OUT:
case H_HARDSHRINK_BACKWARD_GRAD_INPUT:
case H_THRESHOLD_BACKWARD_GRAD_INPUT:
case H_SUB_OUT:
case H_SUBTRACT_OUT:
case H_LERP_SCALAR_OUT:
case H_SOFTSHRINK_BACKWARD_GRAD_INPUT:
  results[i] = redispatch_ptrs_79[op.id - H_ADD_OUT](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Scalar &>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state));
  break;

case H_LEAKY_RELU_BACKWARD_GRAD_INPUT:
  results[i] = at::redispatch::leaky_relu_backward_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Scalar &>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;

case H__LINALG_INV_OUT_HELPER_:
  results[i] = at::redispatch::_linalg_inv_out_helper_(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state));
  break;

case H_MASKED_FILL__TENSOR:
case H_MASKED_SCATTER_:
case H_LERP__TENSOR:
  results[i] = redispatch_ptrs_80[op.id - H_MASKED_FILL__TENSOR](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state));
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
case H_SILU_BACKWARD_GRAD_INPUT:
case H_HEAVISIDE_OUT:
case H_HSPMM_OUT:
case H_BITWISE_AND_TENSOR_OUT:
case H_BITWISE_OR_TENSOR_OUT:
case H_BITWISE_XOR_TENSOR_OUT:
case H_BITWISE_LEFT_SHIFT_TENSOR_OUT:
case H_BITWISE_RIGHT_SHIFT_TENSOR_OUT:
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
case H_ARCTAN2_OUT:
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
case H_SPECIAL_XLOGY_OUT:
case H_SPECIAL_ZETA_OUT:
case H_SPECIAL_GAMMAINC_OUT:
case H_SPECIAL_GAMMAINCC_OUT:
case H_LINALG_MATMUL_OUT:
case H_LINALG_HOUSEHOLDER_PRODUCT_OUT:
case H_INNER_OUT:
case H_OUTER_OUT:
case H_GER_OUT:
case H_LINALG_SOLVE_OUT:
  results[i] = redispatch_ptrs_81[op.id - H_COPYSIGN_OUT](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state));
  break;

case H_MAX_UNPOOL3D_BACKWARD_GRAD_INPUT:
  results[i] = at::redispatch::max_unpool3d_backward_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::IntArrayRef>()(op.args[3], load_state), load<at::IntArrayRef>()(op.args[4], load_state), load<at::IntArrayRef>()(op.args[5], load_state), load<at::Tensor &>()(op.args[6], load_state));
  break;

case H_MAX_UNPOOL2D_BACKWARD_GRAD_INPUT:
  results[i] = at::redispatch::max_unpool2d_backward_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::IntArrayRef>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;

case H_ADDCMUL_:
case H_ADDCDIV_:
  {at::Tensor &(*ptr)(DispatchKeySet, at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Scalar &) = at::redispatch::addcmul_;
  if (op.id == H_ADDCDIV_) ptr = at::redispatch::addcdiv_;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::Scalar &>()(op.args[3], load_state));
  break;}

case H_ADDMV_:
case H_ADDR_:
case H_BADDBMM_:
case H_ADDMM_:
case H_ADDBMM_:
  results[i] = redispatch_ptrs_82[op.id - H_ADDMV_](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::Scalar &>()(op.args[3], load_state), load<at::Scalar &>()(op.args[4], load_state));
  break;

case H_ADDMV_OUT:
case H_ADDR_OUT:
case H_BADDBMM_OUT:
case H_SSPADDMM_OUT:
case H_ADDMM_OUT:
case H_ADDBMM_OUT:
  results[i] = redispatch_ptrs_83[op.id - H_ADDMV_OUT](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::Scalar &>()(op.args[3], load_state), load<at::Scalar &>()(op.args[4], load_state), load<at::Tensor &>()(op.args[5], load_state));
  break;

case H_MULTI_MARGIN_LOSS_BACKWARD_GRAD_INPUT:
  results[i] = at::redispatch::multi_margin_loss_backward_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::Scalar &>()(op.args[3], load_state), load<at::Scalar &>()(op.args[4], load_state), load<c10::optional<at::Tensor> &>()(op.args[5], load_state), load<int64_t>()(op.args[6], load_state), load<at::Tensor &>()(op.args[7], load_state));
  break;

case H_ADDCMUL_OUT:
case H_ADDCDIV_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Scalar &, at::Tensor &) = at::redispatch::addcmul_outf;
  if (op.id == H_ADDCDIV_OUT) ptr = at::redispatch::addcdiv_outf;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::Scalar &>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;}

case H_LU_SOLVE_OUT:
case H_LERP_TENSOR_OUT:
case H_LOG_SIGMOID_BACKWARD_GRAD_INPUT:
case H_ADAPTIVE_MAX_POOL2D_BACKWARD_GRAD_INPUT:
case H_ADAPTIVE_MAX_POOL3D_BACKWARD_GRAD_INPUT:
  results[i] = redispatch_ptrs_84[op.id - H_LU_SOLVE_OUT](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state));
  break;

case H_PUT_:
  results[i] = at::redispatch::put_(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<bool>()(op.args[3], load_state));
  break;

case H_ORMQR_OUT:
  results[i] = at::redispatch::ormqr_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<bool>()(op.args[4], load_state), load<at::Tensor &>()(op.args[5], load_state));
  break;

case H_BINARY_CROSS_ENTROPY_BACKWARD_GRAD_INPUT:
  results[i] = at::redispatch::binary_cross_entropy_backward_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<c10::optional<at::Tensor> &>()(op.args[3], load_state), load<int64_t>()(op.args[4], load_state), load<at::Tensor &>()(op.args[5], load_state));
  break;

case H_NLL_LOSS_BACKWARD_GRAD_INPUT:
case H_NLL_LOSS2D_BACKWARD_GRAD_INPUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, int64_t, int64_t, const at::Tensor &, at::Tensor &) = at::redispatch::nll_loss_backward_outf;
  if (op.id == H_NLL_LOSS2D_BACKWARD_GRAD_INPUT) ptr = at::redispatch::nll_loss2d_backward_outf;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<c10::optional<at::Tensor> &>()(op.args[3], load_state), load<int64_t>()(op.args[4], load_state), load<int64_t>()(op.args[5], load_state), load<at::Tensor &>()(op.args[6], load_state), load<at::Tensor &>()(op.args[7], load_state));
  break;}

case H__AMP_UPDATE_SCALE_:
  results[i] = at::redispatch::_amp_update_scale_(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<double>()(op.args[3], load_state), load<double>()(op.args[4], load_state), load<int64_t>()(op.args[5], load_state));
  break;

case H_MSE_LOSS_BACKWARD_GRAD_INPUT:
case H_L1_LOSS_BACKWARD_GRAD_INPUT:
case H_SOFT_MARGIN_LOSS_BACKWARD_GRAD_INPUT:
  results[i] = redispatch_ptrs_85[op.id - H_MSE_LOSS_BACKWARD_GRAD_INPUT](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<int64_t>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;

case H_MULTILABEL_MARGIN_LOSS_BACKWARD_GRAD_INPUT:
  results[i] = at::redispatch::multilabel_margin_loss_backward_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<int64_t>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state), load<at::Tensor &>()(op.args[5], load_state));
  break;

case H_SMOOTH_L1_LOSS_BACKWARD_GRAD_INPUT:
case H_HUBER_LOSS_BACKWARD_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t, double, at::Tensor &) = at::redispatch::smooth_l1_loss_backward_outf;
  if (op.id == H_HUBER_LOSS_BACKWARD_OUT) ptr = at::redispatch::huber_loss_backward_outf;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<int64_t>()(op.args[3], load_state), load<double>()(op.args[4], load_state), load<at::Tensor &>()(op.args[5], load_state));
  break;}

case H_COPY_:
case H_COPY_SPARSE_TO_SPARSE_:
  {at::Tensor &(*ptr)(DispatchKeySet, at::Tensor &, const at::Tensor &, bool) = at::redispatch::copy_;
  if (op.id == H_COPY_SPARSE_TO_SPARSE_) ptr = at::redispatch::copy_sparse_to_sparse_;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<bool>()(op.args[2], load_state));
  break;}

case H_CHOLESKY_SOLVE_OUT:
case H__CONVERT_INDICES_FROM_CSR_TO_COO_OUT:
case H_LINALG_PINV_OUT_RCOND_TENSOR:
case H_LINALG_MATRIX_RANK_OUT_TOL_TENSOR:
  results[i] = redispatch_ptrs_86[op.id - H_CHOLESKY_SOLVE_OUT](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<bool>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state));
  break;

case H_ISIN_TENSOR_TENSOR_OUT:
case H_BUCKETIZE_TENSOR_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, bool, bool, at::Tensor &) = at::redispatch::isin_outf;
  if (op.id == H_BUCKETIZE_TENSOR_OUT) ptr = at::redispatch::bucketize_outf;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<bool>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;}

case H_LINALG_SOLVE_TRIANGULAR_OUT:
  results[i] = at::redispatch::linalg_solve_triangular_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<bool>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<bool>()(op.args[4], load_state), load<at::Tensor &>()(op.args[5], load_state));
  break;

case H_SEARCHSORTED_TENSOR_OUT:
  results[i] = at::redispatch::searchsorted_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<bool>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<c10::optional<c10::string_view>>()(op.args[4], load_state), load<c10::optional<at::Tensor> &>()(op.args[5], load_state), load<at::Tensor &>()(op.args[6], load_state));
  break;

case H_BERNOULLI__TENSOR:
  results[i] = at::redispatch::bernoulli_(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<c10::optional<at::Generator>>()(op.args[2], load_state));
  break;

case H_NORMAL_TENSOR_TENSOR_OUT:
  results[i] = at::redispatch::normal_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<c10::optional<at::Generator>>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state));
  break;

case H_LINALG_TENSORSOLVE_OUT:
  results[i] = at::redispatch::linalg_tensorsolve_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<c10::optional<at::IntArrayRef>>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state));
  break;

case H_RESIZE_AS_:
  results[i] = at::redispatch::resize_as_(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<c10::optional<at::MemoryFormat>>()(op.args[2], load_state));
  break;

case H_LINEAR_OUT:
  results[i] = at::redispatch::linear_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<c10::optional<at::Tensor> &>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state));
  break;

case H_BINARY_CROSS_ENTROPY_OUT:
  results[i] = at::redispatch::binary_cross_entropy_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<c10::optional<at::Tensor> &>()(op.args[2], load_state), load<int64_t>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;

case H_NLL_LOSS_OUT:
case H_NLL_LOSS2D_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, int64_t, int64_t, at::Tensor &) = at::redispatch::nll_loss_outf;
  if (op.id == H_NLL_LOSS2D_OUT) ptr = at::redispatch::nll_loss2d_outf;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<c10::optional<at::Tensor> &>()(op.args[2], load_state), load<int64_t>()(op.args[3], load_state), load<int64_t>()(op.args[4], load_state), load<at::Tensor &>()(op.args[5], load_state));
  break;}

case H_DIV__TENSOR_MODE:
case H_DIVIDE__TENSOR_MODE:
  {at::Tensor &(*ptr)(DispatchKeySet, at::Tensor &, const at::Tensor &, c10::optional<c10::string_view>) = at::redispatch::div_;
  if (op.id == H_DIVIDE__TENSOR_MODE) ptr = at::redispatch::divide_;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<c10::optional<c10::string_view>>()(op.args[2], load_state));
  break;}

case H_DIV_OUT_MODE:
case H_DIVIDE_OUT_MODE:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, c10::optional<c10::string_view>, at::Tensor &) = at::redispatch::div_outf;
  if (op.id == H_DIVIDE_OUT_MODE) ptr = at::redispatch::divide_outf;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<c10::optional<c10::string_view>>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state));
  break;}

case H_LOGIT_BACKWARD_GRAD_INPUT:
  results[i] = at::redispatch::logit_backward_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<c10::optional<double>>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state));
  break;

case H_CROSS_OUT:
case H_TAKE_ALONG_DIM_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, c10::optional<int64_t>, at::Tensor &) = at::redispatch::cross_outf;
  if (op.id == H_TAKE_ALONG_DIM_OUT) ptr = at::redispatch::take_along_dim_outf;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<c10::optional<int64_t>>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state));
  break;}

case H_QUANTILE_OUT:
case H_NANQUANTILE_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, c10::optional<int64_t>, bool, at::Tensor &) = at::redispatch::quantile_outf;
  if (op.id == H_NANQUANTILE_OUT) ptr = at::redispatch::nanquantile_outf;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<c10::optional<int64_t>>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;}

case H_QUANTILE_NEW_OUT:
case H_NANQUANTILE_NEW_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, c10::optional<int64_t>, bool, c10::string_view, at::Tensor &) = at::redispatch::quantile_outf;
  if (op.id == H_NANQUANTILE_NEW_OUT) ptr = at::redispatch::nanquantile_outf;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<c10::optional<int64_t>>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<c10::string_view>()(op.args[4], load_state), load<at::Tensor &>()(op.args[5], load_state));
  break;}

case H_EMBEDDING_RENORM_:
  results[i] = at::redispatch::embedding_renorm_(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<double>()(op.args[2], load_state), load<double>()(op.args[3], load_state));
  break;

case H__SOBOL_ENGINE_SCRAMBLE_:
  results[i] = at::redispatch::_sobol_engine_scramble_(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<int64_t>()(op.args[2], load_state));
  break;

case H__LOG_SOFTMAX_BACKWARD_DATA_OUT:
case H__SOFTMAX_BACKWARD_DATA_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, int64_t, at::ScalarType, at::Tensor &) = at::redispatch::_log_softmax_backward_data_outf;
  if (op.id == H__SOFTMAX_BACKWARD_DATA_OUT) ptr = at::redispatch::_softmax_backward_data_outf;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<int64_t>()(op.args[2], load_state), load<at::ScalarType>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;}

case H_MSE_LOSS_OUT:
case H_L1_LOSS_OUT:
case H_MULTILABEL_MARGIN_LOSS_OUT:
case H_SOFT_MARGIN_LOSS_OUT:
case H_GLU_BACKWARD_GRAD_INPUT:
case H_LINALG_CROSS_OUT:
  results[i] = redispatch_ptrs_87[op.id - H_MSE_LOSS_OUT](ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<int64_t>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state));
  break;

case H_SMOOTH_L1_LOSS_OUT:
case H_HUBER_LOSS_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, const at::Tensor &, int64_t, double, at::Tensor &) = at::redispatch::smooth_l1_loss_outf;
  if (op.id == H_HUBER_LOSS_OUT) ptr = at::redispatch::huber_loss_outf;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<int64_t>()(op.args[2], load_state), load<double>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;}

case H_REQUIRES_GRAD_:
case H__COALESCED_:
  {at::Tensor &(*ptr)(DispatchKeySet, at::Tensor &, bool) = at::redispatch::__dispatch_requires_grad_;
  if (op.id == H__COALESCED_) ptr = at::redispatch::_coalesced_;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<bool>()(op.args[1], load_state));
  break;}

case H_NUCLEAR_NORM_OUT:
case H_CHOLESKY_OUT:
case H_CHOLESKY_INVERSE_OUT:
case H_LINALG_CHOLESKY_OUT:
  results[i] = redispatch_ptrs_88[op.id - H_NUCLEAR_NORM_OUT](ks, load<at::Tensor &>()(op.args[0], load_state), load<bool>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state));
  break;

case H_INDEX_PUT_:
  results[i] = at::redispatch::index_put_(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::List<c10::optional<at::Tensor>> &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<bool>()(op.args[3], load_state));
  break;

case H__INDEX_PUT_IMPL_:
  results[i] = at::redispatch::_index_put_impl_(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::List<c10::optional<at::Tensor>> &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<bool>()(op.args[4], load_state));
  break;

case H_RENAME_:
  results[i] = at::redispatch::rename_(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::DimnameList>>()(op.args[1], load_state));
  break;

case H_RANDOM_:
  results[i] = at::redispatch::random_(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::Generator>>()(op.args[1], load_state));
  break;

case H_BERNOULLI_OUT:
  results[i] = at::redispatch::bernoulli_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::Generator>>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state));
  break;

case H_FFT_FFT2_OUT:
case H_FFT_IFFT2_OUT:
case H_FFT_RFFT2_OUT:
case H_FFT_IRFFT2_OUT:
  results[i] = redispatch_ptrs_89[op.id - H_FFT_FFT2_OUT](ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::IntArrayRef>>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<c10::optional<c10::string_view>>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;

case H_FFT_HFFT2_OUT:
case H_FFT_IHFFT2_OUT:
  {const at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, c10::optional<at::IntArrayRef>, at::IntArrayRef, c10::optional<c10::string_view>, const at::Tensor &) = at::redispatch::fft_hfft2_outf;
  if (op.id == H_FFT_IHFFT2_OUT) ptr = at::redispatch::fft_ihfft2_outf;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::IntArrayRef>>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<c10::optional<c10::string_view>>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;}

case H_FFT_FFTN_OUT:
case H_FFT_IFFTN_OUT:
case H_FFT_RFFTN_OUT:
case H_FFT_IRFFTN_OUT:
  results[i] = redispatch_ptrs_90[op.id - H_FFT_FFTN_OUT](ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::IntArrayRef>>()(op.args[1], load_state), load<c10::optional<at::IntArrayRef>>()(op.args[2], load_state), load<c10::optional<c10::string_view>>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;

case H_FFT_HFFTN_OUT:
case H_FFT_IHFFTN_OUT:
  {const at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, c10::optional<at::IntArrayRef>, c10::optional<at::IntArrayRef>, c10::optional<c10::string_view>, const at::Tensor &) = at::redispatch::fft_hfftn_outf;
  if (op.id == H_FFT_IHFFTN_OUT) ptr = at::redispatch::fft_ihfftn_outf;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::IntArrayRef>>()(op.args[1], load_state), load<c10::optional<at::IntArrayRef>>()(op.args[2], load_state), load<c10::optional<c10::string_view>>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;}

case H_STD_CORRECTION_OUT:
case H_VAR_CORRECTION_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, c10::optional<at::IntArrayRef>, c10::optional<int64_t>, bool, at::Tensor &) = at::redispatch::std_outf;
  if (op.id == H_VAR_CORRECTION_OUT) ptr = at::redispatch::var_outf;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::IntArrayRef>>()(op.args[1], load_state), load<c10::optional<int64_t>>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;}

case H_NORM_NAMES_DTYPE_OUT:
  results[i] = at::redispatch::norm_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::Scalar> &>()(op.args[1], load_state), load<at::DimnameList>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<at::ScalarType>()(op.args[4], load_state), load<at::Tensor &>()(op.args[5], load_state));
  break;

case H_NORM_NAMES_OUT:
  results[i] = at::redispatch::norm_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::Scalar> &>()(op.args[1], load_state), load<at::DimnameList>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;

case H_NORM_DTYPE_OUT:
  results[i] = at::redispatch::norm_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::Scalar> &>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<at::ScalarType>()(op.args[4], load_state), load<at::Tensor &>()(op.args[5], load_state));
  break;

case H_NORM_OUT:
  results[i] = at::redispatch::norm_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::Scalar> &>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;

case H_LINALG_COND_OUT:
  results[i] = at::redispatch::linalg_cond_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::Scalar> &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state));
  break;

case H_LINALG_NORM_OUT:
  results[i] = at::redispatch::linalg_norm_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::Scalar> &>()(op.args[1], load_state), load<c10::optional<at::IntArrayRef>>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<c10::optional<at::ScalarType>>()(op.args[4], load_state), load<at::Tensor &>()(op.args[5], load_state));
  break;

case H_CLAMP_:
case H_CLIP_:
  {at::Tensor &(*ptr)(DispatchKeySet, at::Tensor &, const c10::optional<at::Scalar> &, const c10::optional<at::Scalar> &) = at::redispatch::clamp_;
  if (op.id == H_CLIP_) ptr = at::redispatch::clip_;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::Scalar> &>()(op.args[1], load_state), load<c10::optional<at::Scalar> &>()(op.args[2], load_state));
  break;}

case H_CLAMP_OUT:
case H_CLIP_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, const c10::optional<at::Scalar> &, const c10::optional<at::Scalar> &, at::Tensor &) = at::redispatch::clamp_outf;
  if (op.id == H_CLIP_OUT) ptr = at::redispatch::clip_outf;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::Scalar> &>()(op.args[1], load_state), load<c10::optional<at::Scalar> &>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state));
  break;}

case H_CLAMP__TENSOR:
case H_CLIP__TENSOR:
  {at::Tensor &(*ptr)(DispatchKeySet, at::Tensor &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &) = at::redispatch::clamp_;
  if (op.id == H_CLIP__TENSOR) ptr = at::redispatch::clip_;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::Tensor> &>()(op.args[1], load_state), load<c10::optional<at::Tensor> &>()(op.args[2], load_state));
  break;}

case H_CLAMP_TENSOR_OUT:
case H_CLIP_TENSOR_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, at::Tensor &) = at::redispatch::clamp_outf;
  if (op.id == H_CLIP_TENSOR_OUT) ptr = at::redispatch::clip_outf;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::Tensor> &>()(op.args[1], load_state), load<c10::optional<at::Tensor> &>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state));
  break;}

case H_BATCH_NORM_ELEMT_OUT:
  results[i] = at::redispatch::batch_norm_elemt_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::Tensor> &>()(op.args[1], load_state), load<c10::optional<at::Tensor> &>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state), load<double>()(op.args[5], load_state), load<at::Tensor &>()(op.args[6], load_state));
  break;

case H_LINALG_PINV_ATOL_RTOL_TENSOR_OUT:
case H_LINALG_MATRIX_RANK_ATOL_RTOL_TENSOR_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, bool, at::Tensor &) = at::redispatch::linalg_pinv_outf;
  if (op.id == H_LINALG_MATRIX_RANK_ATOL_RTOL_TENSOR_OUT) ptr = at::redispatch::linalg_matrix_rank_outf;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<at::Tensor> &>()(op.args[1], load_state), load<c10::optional<at::Tensor> &>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;}

case H_LOGIT_:
  results[i] = at::redispatch::logit_(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<double>>()(op.args[1], load_state));
  break;

case H_LOGIT_OUT:
case H_SPECIAL_LOGIT_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, c10::optional<double>, at::Tensor &) = at::redispatch::logit_outf;
  if (op.id == H_SPECIAL_LOGIT_OUT) ptr = at::redispatch::special_logit_outf;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<double>>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state));
  break;}

case H_LINALG_PINV_ATOL_RTOL_FLOAT_OUT:
case H_LINALG_MATRIX_RANK_ATOL_RTOL_FLOAT_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, c10::optional<double>, c10::optional<double>, bool, at::Tensor &) = at::redispatch::linalg_pinv_outf;
  if (op.id == H_LINALG_MATRIX_RANK_ATOL_RTOL_FLOAT_OUT) ptr = at::redispatch::linalg_matrix_rank_outf;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<double>>()(op.args[1], load_state), load<c10::optional<double>>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;}

case H_NAN_TO_NUM_:
  results[i] = at::redispatch::nan_to_num_(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<double>>()(op.args[1], load_state), load<c10::optional<double>>()(op.args[2], load_state), load<c10::optional<double>>()(op.args[3], load_state));
  break;

case H_NAN_TO_NUM_OUT:
  results[i] = at::redispatch::nan_to_num_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<double>>()(op.args[1], load_state), load<c10::optional<double>>()(op.args[2], load_state), load<c10::optional<double>>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;

case H_ARGMAX_OUT:
case H_ARGMIN_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, c10::optional<int64_t>, bool, at::Tensor &) = at::redispatch::argmax_outf;
  if (op.id == H_ARGMIN_OUT) ptr = at::redispatch::argmin_outf;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<int64_t>>()(op.args[1], load_state), load<bool>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state));
  break;}

case H_FFT_FFT_OUT:
case H_FFT_IFFT_OUT:
case H_FFT_RFFT_OUT:
case H_FFT_IRFFT_OUT:
case H_FFT_HFFT_OUT:
case H_FFT_IHFFT_OUT:
  results[i] = redispatch_ptrs_91[op.id - H_FFT_FFT_OUT](ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::optional<int64_t>>()(op.args[1], load_state), load<int64_t>()(op.args[2], load_state), load<c10::optional<c10::string_view>>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;

case H_LINALG_MATRIX_NORM_STR_ORD_OUT:
  results[i] = at::redispatch::linalg_matrix_norm_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::string_view>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<c10::optional<at::ScalarType>>()(op.args[4], load_state), load<at::Tensor &>()(op.args[5], load_state));
  break;

case H_LINALG_EIGVALSH_OUT:
case H_LINALG_COND_P_STR_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, c10::string_view, at::Tensor &) = at::redispatch::linalg_eigvalsh_outf;
  if (op.id == H_LINALG_COND_P_STR_OUT) ptr = at::redispatch::linalg_cond_outf;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::string_view>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state));
  break;}

case H_LINALG_NORM_ORD_STR_OUT:
  results[i] = at::redispatch::linalg_norm_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<c10::string_view>()(op.args[1], load_state), load<c10::optional<at::IntArrayRef>>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<c10::optional<at::ScalarType>>()(op.args[4], load_state), load<at::Tensor &>()(op.args[5], load_state));
  break;

case H_DROPOUT_:
case H_FEATURE_DROPOUT_:
case H_ALPHA_DROPOUT_:
case H_FEATURE_ALPHA_DROPOUT_:
  results[i] = redispatch_ptrs_92[op.id - H_DROPOUT_](ks, load<at::Tensor &>()(op.args[0], load_state), load<double>()(op.args[1], load_state), load<bool>()(op.args[2], load_state));
  break;

case H_LINALG_PINV_OUT:
case H_LINALG_MATRIX_RANK_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, double, bool, at::Tensor &) = at::redispatch::linalg_pinv_outf;
  if (op.id == H_LINALG_MATRIX_RANK_OUT) ptr = at::redispatch::linalg_matrix_rank_outf;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<double>()(op.args[1], load_state), load<bool>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state));
  break;}

case H_BERNOULLI__FLOAT:
case H_EXPONENTIAL_:
case H_GEOMETRIC_:
  results[i] = redispatch_ptrs_93[op.id - H_BERNOULLI__FLOAT](ks, load<at::Tensor &>()(op.args[0], load_state), load<double>()(op.args[1], load_state), load<c10::optional<at::Generator>>()(op.args[2], load_state));
  break;

case H_NORMAL_TENSOR_FLOAT_OUT:
  results[i] = at::redispatch::normal_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<double>()(op.args[1], load_state), load<c10::optional<at::Generator>>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state));
  break;

case H_QUANTILE_SCALAR_OUT:
case H_NANQUANTILE_SCALAR_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, double, c10::optional<int64_t>, bool, at::Tensor &) = at::redispatch::quantile_outf;
  if (op.id == H_NANQUANTILE_SCALAR_OUT) ptr = at::redispatch::nanquantile_outf;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<double>()(op.args[1], load_state), load<c10::optional<int64_t>>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;}

case H_QUANTILE_NEW_SCALAR_OUT:
case H_NANQUANTILE_NEW_SCALAR_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, double, c10::optional<int64_t>, bool, c10::string_view, at::Tensor &) = at::redispatch::quantile_outf;
  if (op.id == H_NANQUANTILE_NEW_SCALAR_OUT) ptr = at::redispatch::nanquantile_outf;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<double>()(op.args[1], load_state), load<c10::optional<int64_t>>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<c10::string_view>()(op.args[4], load_state), load<at::Tensor &>()(op.args[5], load_state));
  break;}

case H_UNIFORM_:
case H_CAUCHY_:
case H_LOG_NORMAL_:
case H_NORMAL_:
  results[i] = redispatch_ptrs_94[op.id - H_UNIFORM_](ks, load<at::Tensor &>()(op.args[0], load_state), load<double>()(op.args[1], load_state), load<double>()(op.args[2], load_state), load<c10::optional<at::Generator>>()(op.args[3], load_state));
  break;

case H__SOBOL_ENGINE_INITIALIZE_STATE_:
case H_MVLGAMMA_:
case H_SQUEEZE__DIM:
case H_UNSQUEEZE_:
case H_TRIL_:
case H_TRIU_:
case H_POLYGAMMA_:
  results[i] = redispatch_ptrs_95[op.id - H__SOBOL_ENGINE_INITIALIZE_STATE_](ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state));
  break;

case H_HISTC_OUT:
  results[i] = at::redispatch::histc_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<at::Scalar &>()(op.args[2], load_state), load<at::Scalar &>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;

case H__LOGCUMSUMEXP_OUT:
case H_LOGCUMSUMEXP_OUT:
case H_MATRIX_POWER_OUT:
case H_MVLGAMMA_OUT:
case H_DIAG_OUT:
case H_TRIU_OUT:
case H_TRIL_OUT:
case H_GLU_OUT:
case H_SPECIAL_MULTIGAMMALN_OUT:
case H_LINALG_TENSORINV_OUT:
case H_LINALG_MATRIX_POWER_OUT:
  results[i] = redispatch_ptrs_96[op.id - H__LOGCUMSUMEXP_OUT](ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state));
  break;

case H_INDEX_FILL__INT_SCALAR:
case H_SCATTER__VALUE:
  {at::Tensor &(*ptr)(DispatchKeySet, at::Tensor &, int64_t, const at::Tensor &, const at::Scalar &) = at::redispatch::index_fill_;
  if (op.id == H_SCATTER__VALUE) ptr = at::redispatch::scatter_;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::Scalar &>()(op.args[3], load_state));
  break;}

case H_SCATTER_VALUE_OUT:
  results[i] = at::redispatch::scatter_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::Scalar &>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;

case H_SCATTER__VALUE_REDUCE:
  results[i] = at::redispatch::scatter_(ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::Scalar &>()(op.args[3], load_state), load<c10::string_view>()(op.args[4], load_state));
  break;

case H_SCATTER_VALUE_REDUCE_OUT:
  results[i] = at::redispatch::scatter_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::Scalar &>()(op.args[3], load_state), load<c10::string_view>()(op.args[4], load_state), load<at::Tensor &>()(op.args[5], load_state));
  break;

case H_INDEX_COPY_:
case H_INDEX_ADD_:
case H_INDEX_FILL__INT_TENSOR:
case H_SCATTER__SRC:
case H_SCATTER_ADD_:
case H__INDEX_COPY_:
  results[i] = redispatch_ptrs_97[op.id - H_INDEX_COPY_](ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state));
  break;

case H_INDEX_SELECT_OUT:
  results[i] = at::redispatch::index_select_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state));
  break;

case H_INDEX_ADD__ALPHA:
  results[i] = at::redispatch::index_add_(ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state), load<at::Scalar &>()(op.args[4], load_state));
  break;

case H_SCATTER_SRC_OUT:
case H_SCATTER_ADD_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, int64_t, const at::Tensor &, const at::Tensor &, at::Tensor &) = at::redispatch::scatter_outf;
  if (op.id == H_SCATTER_ADD_OUT) ptr = at::redispatch::scatter_add_outf;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;}

case H_SCATTER__REDUCE:
  results[i] = at::redispatch::scatter_(ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state), load<c10::string_view>()(op.args[4], load_state));
  break;

case H_SCATTER_REDUCE_OUT:
  results[i] = at::redispatch::scatter_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state), load<c10::string_view>()(op.args[4], load_state), load<at::Tensor &>()(op.args[5], load_state));
  break;

case H_GATHER_OUT:
  results[i] = at::redispatch::gather_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<bool>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;

case H__SCATTER_REDUCE_OUT:
  results[i] = at::redispatch::_scatter_reduce_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<c10::string_view>()(op.args[3], load_state), load<c10::optional<int64_t>>()(op.args[4], load_state), load<at::Tensor &>()(op.args[5], load_state));
  break;

case H__SOBOL_ENGINE_FF_:
  results[i] = at::redispatch::_sobol_engine_ff_(ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state), load<int64_t>()(op.args[3], load_state), load<int64_t>()(op.args[4], load_state));
  break;

case H_ALL_OUT:
case H_ANY_OUT:
case H__LOG_SOFTMAX_OUT:
case H__SOFTMAX_OUT:
case H__CONVERT_INDICES_FROM_COO_TO_CSR_OUT:
  results[i] = redispatch_ptrs_98[op.id - H_ALL_OUT](ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<bool>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state));
  break;

case H_MULTINOMIAL_OUT:
  results[i] = at::redispatch::multinomial_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<bool>()(op.args[2], load_state), load<c10::optional<at::Generator>>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;

case H_PROD_INT_OUT:
  results[i] = at::redispatch::prod_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<bool>()(op.args[2], load_state), load<c10::optional<at::ScalarType>>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;

case H_RANDOM__TO:
  results[i] = at::redispatch::random_(ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<c10::optional<at::Generator>>()(op.args[2], load_state));
  break;

case H_CUMPROD_:
case H_CUMSUM_:
  {at::Tensor &(*ptr)(DispatchKeySet, at::Tensor &, int64_t, c10::optional<at::ScalarType>) = at::redispatch::cumprod_;
  if (op.id == H_CUMSUM_) ptr = at::redispatch::cumsum_;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<c10::optional<at::ScalarType>>()(op.args[2], load_state));
  break;}

case H_CUMPROD_OUT:
case H_CUMSUM_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, const at::Tensor &, int64_t, c10::optional<at::ScalarType>, at::Tensor &) = at::redispatch::cumprod_outf;
  if (op.id == H_CUMSUM_OUT) ptr = at::redispatch::cumsum_outf;
  results[i] = ptr(ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<c10::optional<at::ScalarType>>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state));
  break;}

case H_RANDOM__FROM:
  results[i] = at::redispatch::random_(ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<c10::optional<int64_t>>()(op.args[2], load_state), load<c10::optional<at::Generator>>()(op.args[3], load_state));
  break;

case H_TRANSPOSE_:
case H__MKLDNN_TRANSPOSE_:
case H_SWAPAXES_:
case H_SWAPDIMS_:
  results[i] = redispatch_ptrs_99[op.id - H_TRANSPOSE_](ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<int64_t>()(op.args[2], load_state));
  break;

case H_DIFF_OUT:
  results[i] = at::redispatch::diff_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<int64_t>()(op.args[2], load_state), load<c10::optional<at::Tensor> &>()(op.args[3], load_state), load<c10::optional<at::Tensor> &>()(op.args[4], load_state), load<at::Tensor &>()(op.args[5], load_state));
  break;

case H_NARROW_COPY_OUT:
  results[i] = at::redispatch::narrow_copy_outf(ks, load<at::Tensor &>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<int64_t>()(op.args[2], load_state), load<int64_t>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;

case H_CAT_NAMES_OUT:
case H_CONCAT_NAMES_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, at::TensorList, at::Dimname, at::Tensor &) = at::redispatch::cat_outf;
  if (op.id == H_CONCAT_NAMES_OUT) ptr = at::redispatch::concat_outf;
  results[i] = ptr(ks, load<at::TensorList>()(op.args[0], load_state), load<at::Dimname>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state));
  break;}

case H_CHAIN_MATMUL_OUT:
case H_ROW_STACK_OUT:
case H_HSTACK_OUT:
case H_VSTACK_OUT:
case H_DSTACK_OUT:
case H_COLUMN_STACK_OUT:
case H_LINALG_MULTI_DOT_OUT:
  results[i] = redispatch_ptrs_100[op.id - H_CHAIN_MATMUL_OUT](ks, load<at::TensorList>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state));
  break;

case H_CAT_OUT:
case H_CONCAT_OUT:
case H_STACK_OUT:
case H__STACK_OUT:
case H__CAT_OUT:
  results[i] = redispatch_ptrs_101[op.id - H_CAT_OUT](ks, load<at::TensorList>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state));
  break;

case H_NORMAL_FLOAT_TENSOR_OUT:
  results[i] = at::redispatch::normal_outf(ks, load<double>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<c10::optional<at::Generator>>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state));
  break;

case H_NORMAL_FLOAT_FLOAT_OUT:
  results[i] = at::redispatch::normal_outf(ks, load<double>()(op.args[0], load_state), load<double>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<c10::optional<at::Generator>>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;

case H_RANDINT_OUT:
  results[i] = at::redispatch::randint_outf(ks, load<int64_t>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state));
  break;

case H_RANDINT_GENERATOR_OUT:
  results[i] = at::redispatch::randint_outf(ks, load<int64_t>()(op.args[0], load_state), load<at::IntArrayRef>()(op.args[1], load_state), load<c10::optional<at::Generator>>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state));
  break;

case H_EYE_OUT:
case H_RANDPERM_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, int64_t, at::Tensor &) = at::redispatch::eye_outf;
  if (op.id == H_RANDPERM_OUT) ptr = at::redispatch::randperm_outf;
  results[i] = ptr(ks, load<int64_t>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state));
  break;}

case H_POLYGAMMA_OUT:
case H_SPECIAL_POLYGAMMA_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, int64_t, const at::Tensor &, at::Tensor &) = at::redispatch::polygamma_outf;
  if (op.id == H_SPECIAL_POLYGAMMA_OUT) ptr = at::redispatch::special_polygamma_outf;
  results[i] = ptr(ks, load<int64_t>()(op.args[0], load_state), load<at::Tensor &>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state));
  break;}

case H_RANDPERM_GENERATOR_OUT:
  results[i] = at::redispatch::randperm_outf(ks, load<int64_t>()(op.args[0], load_state), load<c10::optional<at::Generator>>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state));
  break;

case H_FFT_FFTFREQ_OUT:
case H_FFT_RFFTFREQ_OUT:
  {at::Tensor &(*ptr)(DispatchKeySet, int64_t, double, at::Tensor &) = at::redispatch::fft_fftfreq_outf;
  if (op.id == H_FFT_RFFTFREQ_OUT) ptr = at::redispatch::fft_rfftfreq_outf;
  results[i] = ptr(ks, load<int64_t>()(op.args[0], load_state), load<double>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state));
  break;}

case H_RANDINT_LOW_OUT:
  results[i] = at::redispatch::randint_outf(ks, load<int64_t>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<at::Tensor &>()(op.args[3], load_state));
  break;

case H_RANDINT_LOW_GENERATOR_OUT:
  results[i] = at::redispatch::randint_outf(ks, load<int64_t>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<at::IntArrayRef>()(op.args[2], load_state), load<c10::optional<at::Generator>>()(op.args[3], load_state), load<at::Tensor &>()(op.args[4], load_state));
  break;

case H_EYE_M_OUT:
  results[i] = at::redispatch::eye_outf(ks, load<int64_t>()(op.args[0], load_state), load<int64_t>()(op.args[1], load_state), load<at::Tensor &>()(op.args[2], load_state));
  break;

