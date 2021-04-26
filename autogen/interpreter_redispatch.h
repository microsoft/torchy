case H__CAST_BYTE: set(op.tensor, at::redispatch::_cast_Byte(ks, )); break;
case H__CAST_CHAR: set(op.tensor, at::redispatch::_cast_Char(ks, )); break;
case H__CAST_DOUBLE: set(op.tensor, at::redispatch::_cast_Double(ks, )); break;
case H__CAST_FLOAT: set(op.tensor, at::redispatch::_cast_Float(ks, )); break;
case H__CAST_INT: set(op.tensor, at::redispatch::_cast_Int(ks, )); break;
case H__CAST_LONG: set(op.tensor, at::redispatch::_cast_Long(ks, )); break;
case H__CAST_SHORT: set(op.tensor, at::redispatch::_cast_Short(ks, )); break;
case H__CAST_HALF: set(op.tensor, at::redispatch::_cast_Half(ks, )); break;
case H__FW_PRIMAL: set(op.tensor, at::redispatch::_fw_primal(ks, )); break;
case H__MAKE_DUAL: set(op.tensor, at::redispatch::_make_dual(ks, )); break;
case H__UNPACK_DUAL: set(op.tensor, at::redispatch::_unpack_dual(ks, )); break;
case H_RENAME_: TODO
case H_RENAME: set(op.tensor, at::redispatch::rename(ks, )); break;
case H_ALIGN_TO: set(op.tensor, at::redispatch::align_to(ks, )); break;
case H_ALIGN_TO_ELLIPSIS_IDX: set(op.tensor, at::redispatch::align_to(ks, )); break;
case H_ALIGN_AS: set(op.tensor, at::redispatch::align_as(ks, )); break;
case H_ALIGN_TENSORS: set(op.tensor, at::redispatch::align_tensors(ks, )); break;
case H__ASSERT_ASYNC: set(op.tensor, at::redispatch::_assert_async(ks, )); break;
case H_REFINE_NAMES: set(op.tensor, at::redispatch::refine_names(ks, )); break;
case H__USE_CUDNN_CTC_LOSS: set(op.tensor, at::redispatch::_use_cudnn_ctc_loss(ks, )); break;
case H__CUDNN_CTC_LOSS: set(op.tensor, at::redispatch::_cudnn_ctc_loss(ks, )); break;
case H__USE_CUDNN_RNN_FLATTEN_WEIGHT: set(op.tensor, at::redispatch::_use_cudnn_rnn_flatten_weight(ks, )); break;
case H__CUDNN_RNN_FLATTEN_WEIGHT: set(op.tensor, at::redispatch::_cudnn_rnn_flatten_weight(ks, )); break;
case H__CUDNN_RNN: set(op.tensor, at::redispatch::_cudnn_rnn(ks, )); break;
case H__CUDNN_RNN_BACKWARD: set(op.tensor, at::redispatch::_cudnn_rnn_backward(ks, )); break;
case H__CUDNN_INIT_DROPOUT_STATE: set(op.tensor, at::redispatch::_cudnn_init_dropout_state(ks, )); break;
case H__DEBUG_HAS_INTERNAL_OVERLAP: set(op.tensor, at::redispatch::_debug_has_internal_overlap(ks, )); break;
case H__FUSED_DROPOUT: set(op.tensor, at::redispatch::_fused_dropout(ks, )); break;
case H__MASKED_SCALE: set(op.tensor, at::redispatch::_masked_scale(ks, )); break;
case H__SOBOL_ENGINE_DRAW: set(op.tensor, at::redispatch::_sobol_engine_draw(ks, )); break;
case H__SOBOL_ENGINE_FF_: TODO
case H__SOBOL_ENGINE_SCRAMBLE_: TODO
case H__SOBOL_ENGINE_INITIALIZE_STATE_: TODO
case H__RESHAPE_FROM_TENSOR: set(op.tensor, at::redispatch::_reshape_from_tensor(ks, )); break;
case H__SHAPE_AS_TENSOR: set(op.tensor, at::redispatch::_shape_as_tensor(ks, )); break;
case H_DROPOUT: set(op.tensor, at::redispatch::dropout(ks, )); break;
case H_DROPOUT_: TODO
case H_FEATURE_DROPOUT: set(op.tensor, at::redispatch::feature_dropout(ks, )); break;
case H_FEATURE_DROPOUT_: TODO
case H_ALPHA_DROPOUT: set(op.tensor, at::redispatch::alpha_dropout(ks, )); break;
case H_ALPHA_DROPOUT_: TODO
case H_FEATURE_ALPHA_DROPOUT: set(op.tensor, at::redispatch::feature_alpha_dropout(ks, )); break;
case H_FEATURE_ALPHA_DROPOUT_: TODO
case H_ABS: set(op.tensor, at::redispatch::abs(ks, )); break;
case H_ABS_: TODO
case H_ABS_OUT: TODO
case H_ABSOLUTE: set(op.tensor, at::redispatch::absolute(ks, )); break;
case H_ABSOLUTE_: TODO
case H_ABSOLUTE_OUT: TODO
case H_ANGLE: set(op.tensor, at::redispatch::angle(ks, )); break;
case H_ANGLE_OUT: TODO
case H_VIEW_AS_REAL: set(op.tensor, at::redispatch::view_as_real(ks, )); break;
case H_VIEW_AS_COMPLEX: set(op.tensor, at::redispatch::view_as_complex(ks, )); break;
case H_SGN: set(op.tensor, at::redispatch::sgn(ks, )); break;
case H_SGN_: TODO
case H_SGN_OUT: TODO
case H_REAL: set(op.tensor, at::redispatch::real(ks, )); break;
case H_IMAG: set(op.tensor, at::redispatch::imag(ks, )); break;
case H_CONJ: set(op.tensor, at::redispatch::conj(ks, )); break;
case H_CONJ_OUT: TODO
case H__CONJ: set(op.tensor, at::redispatch::_conj(ks, )); break;
case H_ACOS_OUT: TODO
case H_ARCCOS: set(op.tensor, at::redispatch::arccos(ks, )); break;
case H_ARCCOS_: TODO
case H_ARCCOS_OUT: TODO
case H_AVG_POOL1D: set(op.tensor, at::redispatch::avg_pool1d(ks, )); break;
case H_ADAPTIVE_AVG_POOL1D: set(op.tensor, at::redispatch::adaptive_avg_pool1d(ks, )); break;
case H_ADAPTIVE_MAX_POOL1D: set(op.tensor, at::redispatch::adaptive_max_pool1d(ks, )); break;
case H_ADD_TENSOR: set(op.tensor, at::redispatch::add(ks, )); break;
case H_ADD__TENSOR: TODO
case H_ADD_OUT: TODO
case H__ADD_RELU_TENSOR: set(op.tensor, at::redispatch::_add_relu(ks, )); break;
case H__ADD_RELU__TENSOR: TODO
case H__ADD_RELU_OUT: TODO
case H_ADD_SCALAR: set(op.tensor, at::redispatch::add(ks, )); break;
case H_ADD__SCALAR: TODO
case H_ADDMV_OUT: TODO
case H_ADDR: set(op.tensor, at::redispatch::addr(ks, )); break;
case H_ADDR_: TODO
case H_ADDR_OUT: TODO
case H_AFFINE_GRID_GENERATOR: set(op.tensor, at::redispatch::affine_grid_generator(ks, )); break;
case H_AFFINE_GRID_GENERATOR_BACKWARD: set(op.tensor, at::redispatch::affine_grid_generator_backward(ks, )); break;
case H_ALL_DIM: set(op.tensor, at::redispatch::all(ks, )); break;
case H_ALL_OUT: TODO
case H_ALL_DIMNAME: set(op.tensor, at::redispatch::all(ks, )); break;
case H_ALL_DIMNAME_OUT: TODO
case H_ALLCLOSE: set(op.tensor, at::redispatch::allclose(ks, )); break;
case H_ANY_DIM: set(op.tensor, at::redispatch::any(ks, )); break;
case H_ANY_OUT: TODO
case H_ANY_DIMNAME: set(op.tensor, at::redispatch::any(ks, )); break;
case H_ANY_DIMNAME_OUT: TODO
case H_ARANGE: set(op.tensor, at::redispatch::arange(ks, )); break;
case H_ARANGE_START: set(op.tensor, at::redispatch::arange(ks, )); break;
case H_ARANGE_START_STEP: set(op.tensor, at::redispatch::arange(ks, )); break;
case H_ARANGE_OUT: TODO
case H_ARANGE_START_OUT: TODO
case H__DIM_ARANGE: set(op.tensor, at::redispatch::_dim_arange(ks, )); break;
case H_ARGMAX: set(op.tensor, at::redispatch::argmax(ks, )); break;
case H_ARGMAX_OUT: TODO
case H_ARGMIN: set(op.tensor, at::redispatch::argmin(ks, )); break;
case H_ARGMIN_OUT: TODO
case H_ACOSH_OUT: TODO
case H_ARCCOSH: set(op.tensor, at::redispatch::arccosh(ks, )); break;
case H_ARCCOSH_: TODO
case H_ARCCOSH_OUT: TODO
case H_ASINH_OUT: TODO
case H_ARCSINH: set(op.tensor, at::redispatch::arcsinh(ks, )); break;
case H_ARCSINH_: TODO
case H_ARCSINH_OUT: TODO
case H_ATANH_OUT: TODO
case H_ARCTANH: set(op.tensor, at::redispatch::arctanh(ks, )); break;
case H_ARCTANH_: TODO
case H_ARCTANH_OUT: TODO
case H_AS_STRIDED: set(op.tensor, at::redispatch::as_strided(ks, )); break;
case H_AS_STRIDED_: TODO
case H_ASIN: set(op.tensor, at::redispatch::asin(ks, )); break;
case H_ASIN_: TODO
case H_ASIN_OUT: TODO
case H_ARCSIN: set(op.tensor, at::redispatch::arcsin(ks, )); break;
case H_ARCSIN_: TODO
case H_ARCSIN_OUT: TODO
case H_ATAN_OUT: TODO
case H_ARCTAN: set(op.tensor, at::redispatch::arctan(ks, )); break;
case H_ARCTAN_: TODO
case H_ARCTAN_OUT: TODO
case H_ATLEAST_1D: set(op.tensor, at::redispatch::atleast_1d(ks, )); break;
case H_ATLEAST_1D_SEQUENCE: set(op.tensor, at::redispatch::atleast_1d(ks, )); break;
case H_ATLEAST_2D: set(op.tensor, at::redispatch::atleast_2d(ks, )); break;
case H_ATLEAST_2D_SEQUENCE: set(op.tensor, at::redispatch::atleast_2d(ks, )); break;
case H_ATLEAST_3D: set(op.tensor, at::redispatch::atleast_3d(ks, )); break;
case H_ATLEAST_3D_SEQUENCE: set(op.tensor, at::redispatch::atleast_3d(ks, )); break;
case H_BADDBMM: set(op.tensor, at::redispatch::baddbmm(ks, )); break;
case H_BADDBMM_: TODO
case H__BADDBMM_MKL_: TODO
case H_BADDBMM_OUT: TODO
case H_BARTLETT_WINDOW: set(op.tensor, at::redispatch::bartlett_window(ks, )); break;
case H_BARTLETT_WINDOW_PERIODIC: set(op.tensor, at::redispatch::bartlett_window(ks, )); break;
case H_BATCH_NORM: set(op.tensor, at::redispatch::batch_norm(ks, )); break;
case H_QUANTIZED_BATCH_NORM: set(op.tensor, at::redispatch::quantized_batch_norm(ks, )); break;
case H__BATCH_NORM_IMPL_INDEX: set(op.tensor, at::redispatch::_batch_norm_impl_index(ks, )); break;
case H__BATCH_NORM_IMPL_INDEX_BACKWARD: set(op.tensor, at::redispatch::_batch_norm_impl_index_backward(ks, )); break;
case H_BERNOULLI: set(op.tensor, at::redispatch::bernoulli(ks, )); break;
case H_BERNOULLI_OUT: TODO
case H_BERNOULLI__TENSOR: TODO
case H_BERNOULLI__FLOAT: TODO
case H_BERNOULLI_P: set(op.tensor, at::redispatch::bernoulli(ks, )); break;
case H_BILINEAR: set(op.tensor, at::redispatch::bilinear(ks, )); break;
case H_BINARY_CROSS_ENTROPY: set(op.tensor, at::redispatch::binary_cross_entropy(ks, )); break;
case H_BINARY_CROSS_ENTROPY_OUT: TODO
case H_BINARY_CROSS_ENTROPY_BACKWARD: set(op.tensor, at::redispatch::binary_cross_entropy_backward(ks, )); break;
case H_BINARY_CROSS_ENTROPY_BACKWARD_GRAD_INPUT: TODO
case H_BINARY_CROSS_ENTROPY_WITH_LOGITS: set(op.tensor, at::redispatch::binary_cross_entropy_with_logits(ks, )); break;
case H_BINARY_CROSS_ENTROPY_WITH_LOGITS_BACKWARD: set(op.tensor, at::redispatch::binary_cross_entropy_with_logits_backward(ks, )); break;
case H_BINCOUNT: set(op.tensor, at::redispatch::bincount(ks, )); break;
case H_BITWISE_NOT: set(op.tensor, at::redispatch::bitwise_not(ks, )); break;
case H_BITWISE_NOT_: TODO
case H_BITWISE_NOT_OUT: TODO
case H_COPYSIGN_OUT: TODO
case H_COPYSIGN_SCALAR: set(op.tensor, at::redispatch::copysign(ks, )); break;
case H_COPYSIGN__SCALAR: TODO
case H_COPYSIGN_SCALAR_OUT: TODO
case H_LOGICAL_NOT: set(op.tensor, at::redispatch::logical_not(ks, )); break;
case H_LOGICAL_NOT_: TODO
case H_LOGICAL_NOT_OUT: TODO
case H_LOGICAL_XOR: set(op.tensor, at::redispatch::logical_xor(ks, )); break;
case H_LOGICAL_XOR_: TODO
case H_LOGICAL_XOR_OUT: TODO
case H_LOGICAL_AND: set(op.tensor, at::redispatch::logical_and(ks, )); break;
case H_LOGICAL_AND_: TODO
case H_LOGICAL_AND_OUT: TODO
case H_LOGICAL_OR: set(op.tensor, at::redispatch::logical_or(ks, )); break;
case H_LOGICAL_OR_: TODO
case H_LOGICAL_OR_OUT: TODO
case H_BLACKMAN_WINDOW: set(op.tensor, at::redispatch::blackman_window(ks, )); break;
case H_BLACKMAN_WINDOW_PERIODIC: set(op.tensor, at::redispatch::blackman_window(ks, )); break;
case H_BMM: set(op.tensor, at::redispatch::bmm(ks, )); break;
case H__BMM: set(op.tensor, at::redispatch::_bmm(ks, )); break;
case H_BMM_OUT: TODO
case H__BMM_OUT: TODO
case H_BROADCAST_TENSORS: set(op.tensor, at::redispatch::broadcast_tensors(ks, )); break;
case H_BROADCAST_TO: set(op.tensor, at::redispatch::broadcast_to(ks, )); break;
case H_CAT: set(op.tensor, at::redispatch::cat(ks, )); break;
case H_CAT_OUT: TODO
case H_CAT_NAMES: set(op.tensor, at::redispatch::cat(ks, )); break;
case H_CAT_NAMES_OUT: TODO
case H_BLOCK_DIAG: set(op.tensor, at::redispatch::block_diag(ks, )); break;
case H_CEIL: set(op.tensor, at::redispatch::ceil(ks, )); break;
case H_CEIL_: TODO
case H_CEIL_OUT: TODO
case H_CHAIN_MATMUL: set(op.tensor, at::redispatch::chain_matmul(ks, )); break;
case H_CHAIN_MATMUL_OUT: TODO
case H_UNSAFE_CHUNK: set(op.tensor, at::redispatch::unsafe_chunk(ks, )); break;
case H_CHUNK: set(op.tensor, at::redispatch::chunk(ks, )); break;
case H_TENSOR_SPLIT_SECTIONS: set(op.tensor, at::redispatch::tensor_split(ks, )); break;
case H_TENSOR_SPLIT_INDICES: set(op.tensor, at::redispatch::tensor_split(ks, )); break;
case H_TENSOR_SPLIT_TENSOR_INDICES_OR_SECTIONS: set(op.tensor, at::redispatch::tensor_split(ks, )); break;
case H_CLAMP: set(op.tensor, at::redispatch::clamp(ks, )); break;
case H_CLAMP_: TODO
case H_CLAMP_OUT: TODO
case H_CLAMP_MAX: set(op.tensor, at::redispatch::clamp_max(ks, )); break;
case H_CLAMP_MAX_: TODO
case H_CLAMP_MAX_OUT: TODO
case H_CLAMP_MIN: set(op.tensor, at::redispatch::clamp_min(ks, )); break;
case H_CLAMP_MIN_: TODO
case H_CLAMP_MIN_OUT: TODO
case H_CLIP: set(op.tensor, at::redispatch::clip(ks, )); break;
case H_CLIP_: TODO
case H_CLIP_OUT: TODO
case H_CUDNN_IS_ACCEPTABLE: set(op.tensor, at::redispatch::cudnn_is_acceptable(ks, )); break;
case H_COMPLEX: set(op.tensor, at::redispatch::complex(ks, )); break;
case H_COMPLEX_OUT: TODO
case H_POLAR: set(op.tensor, at::redispatch::polar(ks, )); break;
case H_POLAR_OUT: TODO
case H_CONSTANT_PAD_ND: set(op.tensor, at::redispatch::constant_pad_nd(ks, )); break;
case H_CONVOLUTION: set(op.tensor, at::redispatch::convolution(ks, )); break;
case H_CONVOLUTION_OVERRIDEABLE: set(op.tensor, at::redispatch::convolution_overrideable(ks, )); break;
case H_CONVOLUTION_BACKWARD_OVERRIDEABLE: set(op.tensor, at::redispatch::convolution_backward_overrideable(ks, )); break;
case H__CONVOLUTION: set(op.tensor, at::redispatch::_convolution(ks, )); break;
case H__CONVOLUTION_DEPRECATED: set(op.tensor, at::redispatch::_convolution(ks, )); break;
case H__CONVOLUTION_MODE: set(op.tensor, at::redispatch::_convolution_mode(ks, )); break;
case H__CONVOLUTION_NOGROUP: set(op.tensor, at::redispatch::_convolution_nogroup(ks, )); break;
case H__CONVOLUTION_DOUBLE_BACKWARD: set(op.tensor, at::redispatch::_convolution_double_backward(ks, )); break;
case H_CONV1D: set(op.tensor, at::redispatch::conv1d(ks, )); break;
case H_CONV2D: set(op.tensor, at::redispatch::conv2d(ks, )); break;
case H_CONV3D: set(op.tensor, at::redispatch::conv3d(ks, )); break;
case H_CONV1D_PADDING: set(op.tensor, at::redispatch::conv1d(ks, )); break;
case H_CONV2D_PADDING: set(op.tensor, at::redispatch::conv2d(ks, )); break;
case H_CONV3D_PADDING: set(op.tensor, at::redispatch::conv3d(ks, )); break;
case H_CONV_TBC: set(op.tensor, at::redispatch::conv_tbc(ks, )); break;
case H_CONV_TBC_BACKWARD: set(op.tensor, at::redispatch::conv_tbc_backward(ks, )); break;
case H_CONV_TRANSPOSE1D: set(op.tensor, at::redispatch::conv_transpose1d(ks, )); break;
case H_CONV_TRANSPOSE2D_INPUT: set(op.tensor, at::redispatch::conv_transpose2d(ks, )); break;
case H_CONV_TRANSPOSE3D_INPUT: set(op.tensor, at::redispatch::conv_transpose3d(ks, )); break;
case H_COPY_: TODO
case H_COS_OUT: TODO
case H_COSH_OUT: TODO
case H_COSINE_EMBEDDING_LOSS: set(op.tensor, at::redispatch::cosine_embedding_loss(ks, )); break;
case H_COUNT_NONZERO_DIM_INTLIST: set(op.tensor, at::redispatch::count_nonzero(ks, )); break;
case H_COUNT_NONZERO: set(op.tensor, at::redispatch::count_nonzero(ks, )); break;
case H_CUDNN_AFFINE_GRID_GENERATOR: set(op.tensor, at::redispatch::cudnn_affine_grid_generator(ks, )); break;
case H_CUDNN_AFFINE_GRID_GENERATOR_BACKWARD: set(op.tensor, at::redispatch::cudnn_affine_grid_generator_backward(ks, )); break;
case H_CUDNN_BATCH_NORM: set(op.tensor, at::redispatch::cudnn_batch_norm(ks, )); break;
case H_CUDNN_BATCH_NORM_BACKWARD: set(op.tensor, at::redispatch::cudnn_batch_norm_backward(ks, )); break;
case H_CUDNN_CONVOLUTION_DEPRECATED: set(op.tensor, at::redispatch::cudnn_convolution(ks, )); break;
case H_CUDNN_CONVOLUTION_DEPRECATED2: set(op.tensor, at::redispatch::cudnn_convolution(ks, )); break;
case H_CUDNN_CONVOLUTION: set(op.tensor, at::redispatch::cudnn_convolution(ks, )); break;
case H_CUDNN_CONVOLUTION_BACKWARD_INPUT: set(op.tensor, at::redispatch::cudnn_convolution_backward_input(ks, )); break;
case H_CUDNN_CONVOLUTION_BACKWARD: set(op.tensor, at::redispatch::cudnn_convolution_backward(ks, )); break;
case H_CUDNN_CONVOLUTION_BACKWARD_WEIGHT: set(op.tensor, at::redispatch::cudnn_convolution_backward_weight(ks, )); break;
case H_CUDNN_CONVOLUTION_TRANSPOSE_DEPRECATED: set(op.tensor, at::redispatch::cudnn_convolution_transpose(ks, )); break;
case H_CUDNN_CONVOLUTION_TRANSPOSE_DEPRECATED2: set(op.tensor, at::redispatch::cudnn_convolution_transpose(ks, )); break;
case H_CUDNN_CONVOLUTION_TRANSPOSE: set(op.tensor, at::redispatch::cudnn_convolution_transpose(ks, )); break;
case H_CUDNN_CONVOLUTION_TRANSPOSE_BACKWARD: set(op.tensor, at::redispatch::cudnn_convolution_transpose_backward(ks, )); break;
case H_CUDNN_CONVOLUTION_TRANSPOSE_BACKWARD_INPUT: set(op.tensor, at::redispatch::cudnn_convolution_transpose_backward_input(ks, )); break;
case H_CUDNN_CONVOLUTION_TRANSPOSE_BACKWARD_WEIGHT: set(op.tensor, at::redispatch::cudnn_convolution_transpose_backward_weight(ks, )); break;
case H_CUDNN_CONVOLUTION_RELU: set(op.tensor, at::redispatch::cudnn_convolution_relu(ks, )); break;
case H_CUDNN_CONVOLUTION_ADD_RELU: set(op.tensor, at::redispatch::cudnn_convolution_add_relu(ks, )); break;
case H_CUDNN_GRID_SAMPLER: set(op.tensor, at::redispatch::cudnn_grid_sampler(ks, )); break;
case H_CUDNN_GRID_SAMPLER_BACKWARD: set(op.tensor, at::redispatch::cudnn_grid_sampler_backward(ks, )); break;
case H_CUMMAX: set(op.tensor, at::redispatch::cummax(ks, )); break;
case H_CUMMAX_OUT: TODO
case H_CUMMAX_DIMNAME: set(op.tensor, at::redispatch::cummax(ks, )); break;
case H_CUMMAX_DIMNAME_OUT: TODO
case H__CUMMAX_HELPER: set(op.tensor, at::redispatch::_cummax_helper(ks, )); break;
case H_CUMMIN: set(op.tensor, at::redispatch::cummin(ks, )); break;
case H_CUMMIN_OUT: TODO
case H_CUMMIN_DIMNAME: set(op.tensor, at::redispatch::cummin(ks, )); break;
case H_CUMMIN_DIMNAME_OUT: TODO
case H__CUMMIN_HELPER: set(op.tensor, at::redispatch::_cummin_helper(ks, )); break;
case H_CUMMAXMIN_BACKWARD: set(op.tensor, at::redispatch::cummaxmin_backward(ks, )); break;
case H_CUMPROD: set(op.tensor, at::redispatch::cumprod(ks, )); break;
case H_CUMPROD_: TODO
case H_CUMPROD_OUT: TODO
case H_CUMPROD_DIMNAME: set(op.tensor, at::redispatch::cumprod(ks, )); break;
case H_CUMPROD__DIMNAME: TODO
case H_CUMPROD_DIMNAME_OUT: TODO
case H_CUMPROD_BACKWARD: set(op.tensor, at::redispatch::cumprod_backward(ks, )); break;
case H_CUMSUM: set(op.tensor, at::redispatch::cumsum(ks, )); break;
case H_CUMSUM_: TODO
case H_CUMSUM_OUT: TODO
case H_CUMSUM_DIMNAME: set(op.tensor, at::redispatch::cumsum(ks, )); break;
case H_CUMSUM__DIMNAME: TODO
case H_CUMSUM_DIMNAME_OUT: TODO
case H_CTC_LOSS_INTLIST: set(op.tensor, at::redispatch::ctc_loss(ks, )); break;
case H_CTC_LOSS_TENSOR: set(op.tensor, at::redispatch::ctc_loss(ks, )); break;
case H__CTC_LOSS: set(op.tensor, at::redispatch::_ctc_loss(ks, )); break;
case H__CTC_LOSS_BACKWARD: set(op.tensor, at::redispatch::_ctc_loss_backward(ks, )); break;
case H_DIAG_EMBED: set(op.tensor, at::redispatch::diag_embed(ks, )); break;
case H_DIAGFLAT: set(op.tensor, at::redispatch::diagflat(ks, )); break;
case H_DIAGONAL: set(op.tensor, at::redispatch::diagonal(ks, )); break;
case H_DIAGONAL_DIMNAME: set(op.tensor, at::redispatch::diagonal(ks, )); break;
case H_DIAGONAL_BACKWARD: set(op.tensor, at::redispatch::diagonal_backward(ks, )); break;
case H_FILL_DIAGONAL_: TODO
case H_DIFF: set(op.tensor, at::redispatch::diff(ks, )); break;
case H_DIFF_OUT: TODO
case H_DIV_TENSOR: set(op.tensor, at::redispatch::div(ks, )); break;
case H_DIV__TENSOR: TODO
case H_DIV_OUT: TODO
case H_DIV_OUT_MODE: TODO
case H_DIV_SCALAR: set(op.tensor, at::redispatch::div(ks, )); break;
case H_DIV__SCALAR: TODO
case H_DIV_SCALAR_MODE: set(op.tensor, at::redispatch::div(ks, )); break;
case H_DIV__SCALAR_MODE: TODO
case H_DIVIDE_TENSOR: set(op.tensor, at::redispatch::divide(ks, )); break;
case H_DIVIDE__TENSOR: TODO
case H_DIVIDE_OUT: TODO
case H_DIVIDE_SCALAR: set(op.tensor, at::redispatch::divide(ks, )); break;
case H_DIVIDE__SCALAR: TODO
case H_DIVIDE_TENSOR_MODE: set(op.tensor, at::redispatch::divide(ks, )); break;
case H_DIVIDE__TENSOR_MODE: TODO
case H_DIVIDE_OUT_MODE: TODO
case H_DIVIDE_SCALAR_MODE: set(op.tensor, at::redispatch::divide(ks, )); break;
case H_DIVIDE__SCALAR_MODE: TODO
case H_TRUE_DIVIDE_TENSOR: set(op.tensor, at::redispatch::true_divide(ks, )); break;
case H_TRUE_DIVIDE__TENSOR: TODO
case H_TRUE_DIVIDE_OUT: TODO
case H_TRUE_DIVIDE_SCALAR: set(op.tensor, at::redispatch::true_divide(ks, )); break;
case H_TRUE_DIVIDE__SCALAR: TODO
case H_DOT: set(op.tensor, at::redispatch::dot(ks, )); break;
case H_DOT_OUT: TODO
case H_VDOT: set(op.tensor, at::redispatch::vdot(ks, )); break;
case H_VDOT_OUT: TODO
case H_EINSUM: set(op.tensor, at::redispatch::einsum(ks, )); break;
case H_EMBEDDING: set(op.tensor, at::redispatch::embedding(ks, )); break;
case H_EMBEDDING_BACKWARD: set(op.tensor, at::redispatch::embedding_backward(ks, )); break;
case H_EMBEDDING_DENSE_BACKWARD: set(op.tensor, at::redispatch::embedding_dense_backward(ks, )); break;
case H_EMBEDDING_RENORM_: TODO
case H_EMBEDDING_SPARSE_BACKWARD: set(op.tensor, at::redispatch::embedding_sparse_backward(ks, )); break;
case H__EMBEDDING_BAG_FORWARD_ONLY: set(op.tensor, at::redispatch::_embedding_bag_forward_only(ks, )); break;
case H__ROWWISE_PRUNE: set(op.tensor, at::redispatch::_rowwise_prune(ks, )); break;
case H_ROW_STACK: set(op.tensor, at::redispatch::row_stack(ks, )); break;
case H_ROW_STACK_OUT: TODO
case H_EMBEDDING_BAG: set(op.tensor, at::redispatch::embedding_bag(ks, )); break;
case H_EMBEDDING_BAG_PADDING_IDX: set(op.tensor, at::redispatch::embedding_bag(ks, )); break;
case H__EMBEDDING_BAG: set(op.tensor, at::redispatch::_embedding_bag(ks, )); break;
case H__EMBEDDING_BAG_BACKWARD: set(op.tensor, at::redispatch::_embedding_bag_backward(ks, )); break;
case H__EMBEDDING_BAG_SPARSE_BACKWARD: set(op.tensor, at::redispatch::_embedding_bag_sparse_backward(ks, )); break;
case H__EMBEDDING_BAG_DENSE_BACKWARD: set(op.tensor, at::redispatch::_embedding_bag_dense_backward(ks, )); break;
case H__EMBEDDING_BAG_PER_SAMPLE_WEIGHTS_BACKWARD: set(op.tensor, at::redispatch::_embedding_bag_per_sample_weights_backward(ks, )); break;
case H_EMPTY_NAMES: set(op.tensor, at::redispatch::empty(ks, )); break;
case H_EMPTY_MEMORY_FORMAT: set(op.tensor, at::redispatch::empty(ks, )); break;
case H_NEW_EMPTY: set(op.tensor, at::redispatch::new_empty(ks, )); break;
case H_NEW_EMPTY_STRIDED: set(op.tensor, at::redispatch::new_empty_strided(ks, )); break;
case H_NEW_FULL: set(op.tensor, at::redispatch::new_full(ks, )); break;
case H_NEW_ZEROS: set(op.tensor, at::redispatch::new_zeros(ks, )); break;
case H__EMPTY_AFFINE_QUANTIZED: set(op.tensor, at::redispatch::_empty_affine_quantized(ks, )); break;
case H__EMPTY_PER_CHANNEL_AFFINE_QUANTIZED: set(op.tensor, at::redispatch::_empty_per_channel_affine_quantized(ks, )); break;
case H_RESIZE_: TODO
case H_EMPTY_QUANTIZED: set(op.tensor, at::redispatch::empty_quantized(ks, )); break;
case H_EMPTY_OUT: TODO
case H_EMPTY_LIKE: set(op.tensor, at::redispatch::empty_like(ks, )); break;
case H_EMPTY_STRIDED: set(op.tensor, at::redispatch::empty_strided(ks, )); break;
case H_ERF_OUT: TODO
case H_ERFC_OUT: TODO
case H_EXP_OUT: TODO
case H_EXP2_OUT: TODO
case H_EXPM1_OUT: TODO
case H_EXPAND: set(op.tensor, at::redispatch::expand(ks, )); break;
case H_EXPAND_AS: set(op.tensor, at::redispatch::expand_as(ks, )); break;
case H_EYE: set(op.tensor, at::redispatch::eye(ks, )); break;
case H_EYE_M: set(op.tensor, at::redispatch::eye(ks, )); break;
case H_EYE_OUT: TODO
case H_EYE_M_OUT: TODO
case H_FLATTEN_USING_INTS: set(op.tensor, at::redispatch::flatten(ks, )); break;
case H_FLATTEN_NAMED_OUT_DIM: set(op.tensor, at::redispatch::flatten(ks, )); break;
case H_FLATTEN_USING_NAMES: set(op.tensor, at::redispatch::flatten(ks, )); break;
case H_FLATTEN_DIMNAMELIST: set(op.tensor, at::redispatch::flatten(ks, )); break;
case H_UNFLATTEN_INT: set(op.tensor, at::redispatch::unflatten(ks, )); break;
case H_UNFLATTEN_DIMNAME: set(op.tensor, at::redispatch::unflatten(ks, )); break;
case H_FILL__SCALAR: TODO
case H_FILL__TENSOR: TODO
case H_FLOOR: set(op.tensor, at::redispatch::floor(ks, )); break;
case H_FLOOR_: TODO
case H_FLOOR_OUT: TODO
case H_FLOOR_DIVIDE: set(op.tensor, at::redispatch::floor_divide(ks, )); break;
case H_FLOOR_DIVIDE__TENSOR: TODO
case H_FLOOR_DIVIDE_OUT: TODO
case H_FLOOR_DIVIDE_SCALAR: set(op.tensor, at::redispatch::floor_divide(ks, )); break;
case H_FLOOR_DIVIDE__SCALAR: TODO
case H_FRAC: set(op.tensor, at::redispatch::frac(ks, )); break;
case H_FRAC_: TODO
case H_FRAC_OUT: TODO
case H_FULL_NAMES: set(op.tensor, at::redispatch::full(ks, )); break;
case H_FULL: set(op.tensor, at::redispatch::full(ks, )); break;
case H_FULL_OUT: TODO
case H_FULL_LIKE: set(op.tensor, at::redispatch::full_like(ks, )); break;
case H_FROM_FILE: set(op.tensor, at::redispatch::from_file(ks, )); break;
case H_GCD_OUT: TODO
case H_GCD: set(op.tensor, at::redispatch::gcd(ks, )); break;
case H_GCD_: TODO
case H_LCM_OUT: TODO
case H_LCM: set(op.tensor, at::redispatch::lcm(ks, )); break;
case H_LCM_: TODO
case H_GRID_SAMPLER: set(op.tensor, at::redispatch::grid_sampler(ks, )); break;
case H_GRID_SAMPLER_2D: set(op.tensor, at::redispatch::grid_sampler_2d(ks, )); break;
case H_GRID_SAMPLER_2D_BACKWARD: set(op.tensor, at::redispatch::grid_sampler_2d_backward(ks, )); break;
case H__GRID_SAMPLER_2D_CPU_FALLBACK: set(op.tensor, at::redispatch::_grid_sampler_2d_cpu_fallback(ks, )); break;
case H__GRID_SAMPLER_2D_CPU_FALLBACK_BACKWARD: set(op.tensor, at::redispatch::_grid_sampler_2d_cpu_fallback_backward(ks, )); break;
case H_GRID_SAMPLER_3D: set(op.tensor, at::redispatch::grid_sampler_3d(ks, )); break;
case H_GRID_SAMPLER_3D_BACKWARD: set(op.tensor, at::redispatch::grid_sampler_3d_backward(ks, )); break;
case H_HANN_WINDOW: set(op.tensor, at::redispatch::hann_window(ks, )); break;
case H_HANN_WINDOW_PERIODIC: set(op.tensor, at::redispatch::hann_window(ks, )); break;
case H_HAMMING_WINDOW: set(op.tensor, at::redispatch::hamming_window(ks, )); break;
case H_HAMMING_WINDOW_PERIODIC: set(op.tensor, at::redispatch::hamming_window(ks, )); break;
case H_HAMMING_WINDOW_PERIODIC_ALPHA: set(op.tensor, at::redispatch::hamming_window(ks, )); break;
case H_HAMMING_WINDOW_PERIODIC_ALPHA_BETA: set(op.tensor, at::redispatch::hamming_window(ks, )); break;
case H_KAISER_WINDOW: set(op.tensor, at::redispatch::kaiser_window(ks, )); break;
case H_KAISER_WINDOW_PERIODIC: set(op.tensor, at::redispatch::kaiser_window(ks, )); break;
case H_KAISER_WINDOW_BETA: set(op.tensor, at::redispatch::kaiser_window(ks, )); break;
case H_HINGE_EMBEDDING_LOSS: set(op.tensor, at::redispatch::hinge_embedding_loss(ks, )); break;
case H_GROUP_NORM: set(op.tensor, at::redispatch::group_norm(ks, )); break;
case H_NATIVE_GROUP_NORM: set(op.tensor, at::redispatch::native_group_norm(ks, )); break;
case H_NATIVE_GROUP_NORM_BACKWARD: set(op.tensor, at::redispatch::native_group_norm_backward(ks, )); break;
case H__FFT_R2C: set(op.tensor, at::redispatch::_fft_r2c(ks, )); break;
case H__FFT_R2C_OUT: TODO
case H__FFT_C2R: set(op.tensor, at::redispatch::_fft_c2r(ks, )); break;
case H__FFT_C2R_OUT: TODO
case H__FFT_C2C: set(op.tensor, at::redispatch::_fft_c2c(ks, )); break;
case H__FFT_C2C_OUT: TODO
case H__CUFFT_GET_PLAN_CACHE_SIZE: set(op.tensor, at::redispatch::_cufft_get_plan_cache_size(ks, )); break;
case H__CUFFT_GET_PLAN_CACHE_MAX_SIZE: set(op.tensor, at::redispatch::_cufft_get_plan_cache_max_size(ks, )); break;
case H__CUFFT_SET_PLAN_CACHE_MAX_SIZE: set(op.tensor, at::redispatch::_cufft_set_plan_cache_max_size(ks, )); break;
case H__CUFFT_CLEAR_PLAN_CACHE: set(op.tensor, at::redispatch::_cufft_clear_plan_cache(ks, )); break;
case H_INDEX_TENSOR: set(op.tensor, at::redispatch::index(ks, )); break;
case H_INDEX_COPY_: TODO
case H_INDEX_COPY: set(op.tensor, at::redispatch::index_copy(ks, )); break;
case H_INDEX_COPY__DIMNAME: TODO
case H_INDEX_COPY_DIMNAME: set(op.tensor, at::redispatch::index_copy(ks, )); break;
case H_INDEX_PUT_: TODO
case H_INDEX_PUT: set(op.tensor, at::redispatch::index_put(ks, )); break;
case H__INDEX_PUT_IMPL_: TODO
case H_INSTANCE_NORM: set(op.tensor, at::redispatch::instance_norm(ks, )); break;
case H_INVERSE: set(op.tensor, at::redispatch::inverse(ks, )); break;
case H_INVERSE_OUT: TODO
case H__INVERSE_HELPER: set(op.tensor, at::redispatch::_inverse_helper(ks, )); break;
case H_ISCLOSE: set(op.tensor, at::redispatch::isclose(ks, )); break;
case H_ISNAN: set(op.tensor, at::redispatch::isnan(ks, )); break;
case H_IS_DISTRIBUTED: set(op.tensor, at::redispatch::is_distributed(ks, )); break;
case H_ISREAL: set(op.tensor, at::redispatch::isreal(ks, )); break;
case H_IS_NONZERO: set(op.tensor, at::redispatch::is_nonzero(ks, )); break;
case H_IS_SAME_SIZE: set(op.tensor, at::redispatch::is_same_size(ks, )); break;
case H_KL_DIV: set(op.tensor, at::redispatch::kl_div(ks, )); break;
case H_KL_DIV_BACKWARD: set(op.tensor, at::redispatch::kl_div_backward(ks, )); break;
case H_KRON: set(op.tensor, at::redispatch::kron(ks, )); break;
case H_KRON_OUT: TODO
case H_KTHVALUE: set(op.tensor, at::redispatch::kthvalue(ks, )); break;
case H_KTHVALUE_VALUES: TODO
case H_KTHVALUE_DIMNAME: set(op.tensor, at::redispatch::kthvalue(ks, )); break;
case H_KTHVALUE_DIMNAME_OUT: TODO
case H_LAYER_NORM: set(op.tensor, at::redispatch::layer_norm(ks, )); break;
case H_NATIVE_LAYER_NORM: set(op.tensor, at::redispatch::native_layer_norm(ks, )); break;
case H_NATIVE_LAYER_NORM_BACKWARD: set(op.tensor, at::redispatch::native_layer_norm_backward(ks, )); break;
case H_NAN_TO_NUM: set(op.tensor, at::redispatch::nan_to_num(ks, )); break;
case H_NAN_TO_NUM_: TODO
case H_NAN_TO_NUM_OUT: TODO
case H_LINEAR: set(op.tensor, at::redispatch::linear(ks, )); break;
case H_MKLDNN_LINEAR: set(op.tensor, at::redispatch::mkldnn_linear(ks, )); break;
case H_MKLDNN_LINEAR_BACKWARD_INPUT: set(op.tensor, at::redispatch::mkldnn_linear_backward_input(ks, )); break;
case H_MKLDNN_LINEAR_BACKWARD_WEIGHTS: set(op.tensor, at::redispatch::mkldnn_linear_backward_weights(ks, )); break;
case H_MKLDNN_LINEAR_BACKWARD: set(op.tensor, at::redispatch::mkldnn_linear_backward(ks, )); break;
case H_FBGEMM_LINEAR_INT8_WEIGHT_FP32_ACTIVATION: set(op.tensor, at::redispatch::fbgemm_linear_int8_weight_fp32_activation(ks, )); break;
case H_FBGEMM_LINEAR_INT8_WEIGHT: set(op.tensor, at::redispatch::fbgemm_linear_int8_weight(ks, )); break;
case H_FBGEMM_LINEAR_QUANTIZE_WEIGHT: set(op.tensor, at::redispatch::fbgemm_linear_quantize_weight(ks, )); break;
case H_FBGEMM_PACK_GEMM_MATRIX_FP16: set(op.tensor, at::redispatch::fbgemm_pack_gemm_matrix_fp16(ks, )); break;
case H_FBGEMM_LINEAR_FP16_WEIGHT_FP32_ACTIVATION: set(op.tensor, at::redispatch::fbgemm_linear_fp16_weight_fp32_activation(ks, )); break;
case H_FBGEMM_LINEAR_FP16_WEIGHT: set(op.tensor, at::redispatch::fbgemm_linear_fp16_weight(ks, )); break;
case H_FBGEMM_PACK_QUANTIZED_MATRIX: set(op.tensor, at::redispatch::fbgemm_pack_quantized_matrix(ks, )); break;
case H_FBGEMM_PACK_QUANTIZED_MATRIX_KN: set(op.tensor, at::redispatch::fbgemm_pack_quantized_matrix(ks, )); break;
case H_LDEXP_TENSOR: set(op.tensor, at::redispatch::ldexp(ks, )); break;
case H_LDEXP_: TODO
case H_LDEXP_OUT: TODO
case H_LINSPACE: set(op.tensor, at::redispatch::linspace(ks, )); break;
case H_LINSPACE_OUT: TODO
case H_LOG_OUT: TODO
case H_LOG10_OUT: TODO
case H_LOG1P: set(op.tensor, at::redispatch::log1p(ks, )); break;
case H_LOG1P_: TODO
case H_LOG1P_OUT: TODO
case H_LOG2_OUT: TODO
case H_LOGADDEXP_OUT: TODO
case H_LOGADDEXP: set(op.tensor, at::redispatch::logaddexp(ks, )); break;
case H_LOGADDEXP2_OUT: TODO
case H_LOGADDEXP2: set(op.tensor, at::redispatch::logaddexp2(ks, )); break;
case H_XLOGY_TENSOR: set(op.tensor, at::redispatch::xlogy(ks, )); break;
case H_XLOGY_SCALAR_SELF: set(op.tensor, at::redispatch::xlogy(ks, )); break;
case H_XLOGY_SCALAR_OTHER: set(op.tensor, at::redispatch::xlogy(ks, )); break;
case H_XLOGY__TENSOR: TODO
case H_XLOGY__SCALAR_OTHER: TODO
case H_XLOGY_OUTTENSOR: TODO
case H_XLOGY_OUTSCALAR_SELF: TODO
case H_XLOGY_OUTSCALAR_OTHER: TODO
case H_LOGDET: set(op.tensor, at::redispatch::logdet(ks, )); break;
case H_LOGSPACE: set(op.tensor, at::redispatch::logspace(ks, )); break;
case H_LOGSPACE_OUT: TODO
case H_LOG_SOFTMAX_INT: set(op.tensor, at::redispatch::log_softmax(ks, )); break;
case H_LOG_SOFTMAX_DIMNAME: set(op.tensor, at::redispatch::log_softmax(ks, )); break;
case H__LOG_SOFTMAX: set(op.tensor, at::redispatch::_log_softmax(ks, )); break;
case H__LOG_SOFTMAX_BACKWARD_DATA: set(op.tensor, at::redispatch::_log_softmax_backward_data(ks, )); break;
case H__LOGCUMSUMEXP: set(op.tensor, at::redispatch::_logcumsumexp(ks, )); break;
case H__LOGCUMSUMEXP_OUT: TODO
case H_LOGCUMSUMEXP: set(op.tensor, at::redispatch::logcumsumexp(ks, )); break;
case H_LOGCUMSUMEXP_OUT: TODO
case H_LOGCUMSUMEXP_DIMNAME: set(op.tensor, at::redispatch::logcumsumexp(ks, )); break;
case H_LOGCUMSUMEXP_DIMNAME_OUT: TODO
case H_LOGSUMEXP: set(op.tensor, at::redispatch::logsumexp(ks, )); break;
case H_LOGSUMEXP_OUT: TODO
case H_LOGSUMEXP_NAMES: set(op.tensor, at::redispatch::logsumexp(ks, )); break;
case H_LOGSUMEXP_NAMES_OUT: TODO
case H_MARGIN_RANKING_LOSS: set(op.tensor, at::redispatch::margin_ranking_loss(ks, )); break;
case H_MATMUL: set(op.tensor, at::redispatch::matmul(ks, )); break;
case H_MATMUL_OUT: TODO
case H_MATRIX_RANK_TOL: set(op.tensor, at::redispatch::matrix_rank(ks, )); break;
case H_MATRIX_RANK: set(op.tensor, at::redispatch::matrix_rank(ks, )); break;
case H_MATRIX_POWER: set(op.tensor, at::redispatch::matrix_power(ks, )); break;
case H_MATRIX_POWER_OUT: TODO
case H_MATRIX_EXP: set(op.tensor, at::redispatch::matrix_exp(ks, )); break;
case H_MATRIX_EXP_BACKWARD: set(op.tensor, at::redispatch::matrix_exp_backward(ks, )); break;
case H__AMINMAX: set(op.tensor, at::redispatch::_aminmax(ks, )); break;
case H__AMINMAX_DIM: set(op.tensor, at::redispatch::_aminmax(ks, )); break;
case H__COMPUTE_LINEAR_COMBINATION: set(op.tensor, at::redispatch::_compute_linear_combination(ks, )); break;
case H__COMPUTE_LINEAR_COMBINATION_OUT: TODO
case H_MAX_DIM: set(op.tensor, at::redispatch::max(ks, )); break;
case H_MAX_DIM_MAX: TODO
case H_MAX_NAMES_DIM: set(op.tensor, at::redispatch::max(ks, )); break;
case H_MAX_NAMES_DIM_MAX: TODO
case H_VALUE_SELECTING_REDUCTION_BACKWARD: set(op.tensor, at::redispatch::value_selecting_reduction_backward(ks, )); break;
case H_AMAX: set(op.tensor, at::redispatch::amax(ks, )); break;
case H_AMAX_OUT: TODO
case H_MAX_POOL1D_WITH_INDICES: set(op.tensor, at::redispatch::max_pool1d_with_indices(ks, )); break;
case H_MAX_POOL1D: set(op.tensor, at::redispatch::max_pool1d(ks, )); break;
case H_MAX_POOL2D: set(op.tensor, at::redispatch::max_pool2d(ks, )); break;
case H_MKLDNN_MAX_POOL2D: set(op.tensor, at::redispatch::mkldnn_max_pool2d(ks, )); break;
case H_MKLDNN_MAX_POOL2D_BACKWARD: set(op.tensor, at::redispatch::mkldnn_max_pool2d_backward(ks, )); break;
case H_MKLDNN_MAX_POOL3D: set(op.tensor, at::redispatch::mkldnn_max_pool3d(ks, )); break;
case H_MKLDNN_MAX_POOL3D_BACKWARD: set(op.tensor, at::redispatch::mkldnn_max_pool3d_backward(ks, )); break;
case H_QUANTIZED_MAX_POOL1D: set(op.tensor, at::redispatch::quantized_max_pool1d(ks, )); break;
case H_QUANTIZED_MAX_POOL2D: set(op.tensor, at::redispatch::quantized_max_pool2d(ks, )); break;
case H_MAX_POOL3D: set(op.tensor, at::redispatch::max_pool3d(ks, )); break;
case H_MEAN: set(op.tensor, at::redispatch::mean(ks, )); break;
case H_MEAN_DIM: set(op.tensor, at::redispatch::mean(ks, )); break;
case H_MEAN_OUT: TODO
case H_MEAN_NAMES_DIM: set(op.tensor, at::redispatch::mean(ks, )); break;
case H_MEAN_NAMES_OUT: TODO
case H_MEDIAN: set(op.tensor, at::redispatch::median(ks, )); break;
case H_MEDIAN_DIM: set(op.tensor, at::redispatch::median(ks, )); break;
case H_MEDIAN_DIM_VALUES: TODO
case H_MEDIAN_NAMES_DIM: set(op.tensor, at::redispatch::median(ks, )); break;
case H_MEDIAN_NAMES_DIM_VALUES: TODO
case H_NANMEDIAN: set(op.tensor, at::redispatch::nanmedian(ks, )); break;
case H_NANMEDIAN_DIM: set(op.tensor, at::redispatch::nanmedian(ks, )); break;
case H_NANMEDIAN_DIM_VALUES: TODO
case H_NANMEDIAN_NAMES_DIM: set(op.tensor, at::redispatch::nanmedian(ks, )); break;
case H_NANMEDIAN_NAMES_DIM_VALUES: TODO
case H_MIN_DIM: set(op.tensor, at::redispatch::min(ks, )); break;
case H_MIN_DIM_MIN: TODO
case H_MIN_NAMES_DIM: set(op.tensor, at::redispatch::min(ks, )); break;
case H_MIN_NAMES_DIM_MIN: TODO
case H_AMIN: set(op.tensor, at::redispatch::amin(ks, )); break;
case H_AMIN_OUT: TODO
case H_MKLDNN_CONVOLUTION: set(op.tensor, at::redispatch::mkldnn_convolution(ks, )); break;
case H_MKLDNN_CONVOLUTION_BACKWARD_INPUT: set(op.tensor, at::redispatch::mkldnn_convolution_backward_input(ks, )); break;
case H_MKLDNN_CONVOLUTION_BACKWARD_WEIGHTS: set(op.tensor, at::redispatch::mkldnn_convolution_backward_weights(ks, )); break;
case H_MKLDNN_CONVOLUTION_BACKWARD: set(op.tensor, at::redispatch::mkldnn_convolution_backward(ks, )); break;
case H_MIOPEN_BATCH_NORM: set(op.tensor, at::redispatch::miopen_batch_norm(ks, )); break;
case H_MIOPEN_BATCH_NORM_BACKWARD: set(op.tensor, at::redispatch::miopen_batch_norm_backward(ks, )); break;
case H_MIOPEN_CONVOLUTION: set(op.tensor, at::redispatch::miopen_convolution(ks, )); break;
case H_MIOPEN_CONVOLUTION_BACKWARD_INPUT: set(op.tensor, at::redispatch::miopen_convolution_backward_input(ks, )); break;
case H_MIOPEN_CONVOLUTION_BACKWARD: set(op.tensor, at::redispatch::miopen_convolution_backward(ks, )); break;
case H_MIOPEN_CONVOLUTION_BACKWARD_BIAS: set(op.tensor, at::redispatch::miopen_convolution_backward_bias(ks, )); break;
case H_MIOPEN_CONVOLUTION_BACKWARD_WEIGHT: set(op.tensor, at::redispatch::miopen_convolution_backward_weight(ks, )); break;
case H_MIOPEN_CONVOLUTION_TRANSPOSE: set(op.tensor, at::redispatch::miopen_convolution_transpose(ks, )); break;
case H_MIOPEN_CONVOLUTION_TRANSPOSE_BACKWARD: set(op.tensor, at::redispatch::miopen_convolution_transpose_backward(ks, )); break;
case H_MIOPEN_CONVOLUTION_TRANSPOSE_BACKWARD_INPUT: set(op.tensor, at::redispatch::miopen_convolution_transpose_backward_input(ks, )); break;
case H_MIOPEN_CONVOLUTION_TRANSPOSE_BACKWARD_WEIGHT: set(op.tensor, at::redispatch::miopen_convolution_transpose_backward_weight(ks, )); break;
case H_MIOPEN_DEPTHWISE_CONVOLUTION: set(op.tensor, at::redispatch::miopen_depthwise_convolution(ks, )); break;
case H_MIOPEN_DEPTHWISE_CONVOLUTION_BACKWARD_INPUT: set(op.tensor, at::redispatch::miopen_depthwise_convolution_backward_input(ks, )); break;
case H_MIOPEN_DEPTHWISE_CONVOLUTION_BACKWARD: set(op.tensor, at::redispatch::miopen_depthwise_convolution_backward(ks, )); break;
case H_MIOPEN_DEPTHWISE_CONVOLUTION_BACKWARD_WEIGHT: set(op.tensor, at::redispatch::miopen_depthwise_convolution_backward_weight(ks, )); break;
case H_MIOPEN_RNN: set(op.tensor, at::redispatch::miopen_rnn(ks, )); break;
case H_MIOPEN_RNN_BACKWARD: set(op.tensor, at::redispatch::miopen_rnn_backward(ks, )); break;
case H_MM: set(op.tensor, at::redispatch::mm(ks, )); break;
case H_MM_OUT: TODO
case H__SPARSE_MM: set(op.tensor, at::redispatch::_sparse_mm(ks, )); break;
case H__SPARSE_SPARSE_MATMUL: set(op.tensor, at::redispatch::_sparse_sparse_matmul(ks, )); break;
case H__SPARSE_MASK_HELPER: set(op.tensor, at::redispatch::_sparse_mask_helper(ks, )); break;
case H_MODE: set(op.tensor, at::redispatch::mode(ks, )); break;
case H_MODE_VALUES: TODO
case H_MODE_DIMNAME: set(op.tensor, at::redispatch::mode(ks, )); break;
case H_MODE_DIMNAME_OUT: TODO
case H_MUL_TENSOR: set(op.tensor, at::redispatch::mul(ks, )); break;
case H_MUL__TENSOR: TODO
case H_MUL_OUT: TODO
case H_MUL_SCALAR: set(op.tensor, at::redispatch::mul(ks, )); break;
case H_MUL__SCALAR: TODO
case H_MULTIPLY_TENSOR: set(op.tensor, at::redispatch::multiply(ks, )); break;
case H_MULTIPLY__TENSOR: TODO
case H_MULTIPLY_OUT: TODO
case H_MULTIPLY_SCALAR: set(op.tensor, at::redispatch::multiply(ks, )); break;
case H_MULTIPLY__SCALAR: TODO
case H_MV: set(op.tensor, at::redispatch::mv(ks, )); break;
case H_MV_OUT: TODO
case H_MVLGAMMA: set(op.tensor, at::redispatch::mvlgamma(ks, )); break;
case H_MVLGAMMA_: TODO
case H_NARROW_COPY: set(op.tensor, at::redispatch::narrow_copy(ks, )); break;
case H_NARROW_COPY_OUT: TODO
case H_NARROW: set(op.tensor, at::redispatch::narrow(ks, )); break;
case H_NARROW_TENSOR: set(op.tensor, at::redispatch::narrow(ks, )); break;
case H_NATIVE_BATCH_NORM: set(op.tensor, at::redispatch::native_batch_norm(ks, )); break;
case H_NATIVE_BATCH_NORM_OUT: TODO
case H_BATCH_NORM_STATS: set(op.tensor, at::redispatch::batch_norm_stats(ks, )); break;
case H_BATCH_NORM_ELEMT: set(op.tensor, at::redispatch::batch_norm_elemt(ks, )); break;
case H_BATCH_NORM_ELEMT_OUT: TODO
case H_BATCH_NORM_GATHER_STATS: set(op.tensor, at::redispatch::batch_norm_gather_stats(ks, )); break;
case H_BATCH_NORM_GATHER_STATS_WITH_COUNTS: set(op.tensor, at::redispatch::batch_norm_gather_stats_with_counts(ks, )); break;
case H_NATIVE_BATCH_NORM_BACKWARD: set(op.tensor, at::redispatch::native_batch_norm_backward(ks, )); break;
case H_BATCH_NORM_BACKWARD_REDUCE: set(op.tensor, at::redispatch::batch_norm_backward_reduce(ks, )); break;
case H_BATCH_NORM_BACKWARD_ELEMT: set(op.tensor, at::redispatch::batch_norm_backward_elemt(ks, )); break;
case H_BATCH_NORM_UPDATE_STATS: set(op.tensor, at::redispatch::batch_norm_update_stats(ks, )); break;
case H_IS_VULKAN_AVAILABLE: set(op.tensor, at::redispatch::is_vulkan_available(ks, )); break;
case H__NNPACK_AVAILABLE: set(op.tensor, at::redispatch::_nnpack_available(ks, )); break;
case H__NNPACK_SPATIAL_CONVOLUTION: set(op.tensor, at::redispatch::_nnpack_spatial_convolution(ks, )); break;
case H__NNPACK_SPATIAL_CONVOLUTION_BACKWARD: set(op.tensor, at::redispatch::_nnpack_spatial_convolution_backward(ks, )); break;
case H__NNPACK_SPATIAL_CONVOLUTION_BACKWARD_INPUT: set(op.tensor, at::redispatch::_nnpack_spatial_convolution_backward_input(ks, )); break;
case H__NNPACK_SPATIAL_CONVOLUTION_BACKWARD_WEIGHT: set(op.tensor, at::redispatch::_nnpack_spatial_convolution_backward_weight(ks, )); break;
case H_ONES_NAMES: set(op.tensor, at::redispatch::ones(ks, )); break;
case H_ONES: set(op.tensor, at::redispatch::ones(ks, )); break;
case H_ONES_OUT: TODO
case H_ONES_LIKE: set(op.tensor, at::redispatch::ones_like(ks, )); break;
case H_PAIRWISE_DISTANCE: set(op.tensor, at::redispatch::pairwise_distance(ks, )); break;
case H_CDIST: set(op.tensor, at::redispatch::cdist(ks, )); break;
case H__EUCLIDEAN_DIST: set(op.tensor, at::redispatch::_euclidean_dist(ks, )); break;
case H__CDIST_FORWARD: set(op.tensor, at::redispatch::_cdist_forward(ks, )); break;
case H__CDIST_BACKWARD: set(op.tensor, at::redispatch::_cdist_backward(ks, )); break;
case H_PDIST: set(op.tensor, at::redispatch::pdist(ks, )); break;
case H__PDIST_FORWARD: set(op.tensor, at::redispatch::_pdist_forward(ks, )); break;
case H__PDIST_BACKWARD: set(op.tensor, at::redispatch::_pdist_backward(ks, )); break;
case H_COSINE_SIMILARITY: set(op.tensor, at::redispatch::cosine_similarity(ks, )); break;
case H_PERMUTE: set(op.tensor, at::redispatch::permute(ks, )); break;
case H_MOVEDIM_INTLIST: set(op.tensor, at::redispatch::movedim(ks, )); break;
case H_MOVEDIM_INT: set(op.tensor, at::redispatch::movedim(ks, )); break;
case H_MOVEAXIS_INTLIST: set(op.tensor, at::redispatch::moveaxis(ks, )); break;
case H_MOVEAXIS_INT: set(op.tensor, at::redispatch::moveaxis(ks, )); break;
case H_NUMPY_T: set(op.tensor, at::redispatch::numpy_T(ks, )); break;
case H_PIXEL_SHUFFLE: set(op.tensor, at::redispatch::pixel_shuffle(ks, )); break;
case H_PIXEL_UNSHUFFLE: set(op.tensor, at::redispatch::pixel_unshuffle(ks, )); break;
case H_CHANNEL_SHUFFLE: set(op.tensor, at::redispatch::channel_shuffle(ks, )); break;
case H_IS_PINNED: set(op.tensor, at::redispatch::is_pinned(ks, )); break;
case H_PIN_MEMORY: set(op.tensor, at::redispatch::pin_memory(ks, )); break;
case H_PINVERSE: set(op.tensor, at::redispatch::pinverse(ks, )); break;
case H_POISSON_NLL_LOSS: set(op.tensor, at::redispatch::poisson_nll_loss(ks, )); break;
case H_RAD2DEG: set(op.tensor, at::redispatch::rad2deg(ks, )); break;
case H_RAD2DEG_: TODO
case H_RAD2DEG_OUT: TODO
case H_DEG2RAD: set(op.tensor, at::redispatch::deg2rad(ks, )); break;
case H_DEG2RAD_: TODO
case H_DEG2RAD_OUT: TODO
case H_SCALAR_TENSOR: set(op.tensor, at::redispatch::scalar_tensor(ks, )); break;
case H_RAND_NAMES: set(op.tensor, at::redispatch::rand(ks, )); break;
case H_RAND_GENERATOR_WITH_NAMES: set(op.tensor, at::redispatch::rand(ks, )); break;
case H_RAND: set(op.tensor, at::redispatch::rand(ks, )); break;
case H_RAND_GENERATOR: set(op.tensor, at::redispatch::rand(ks, )); break;
case H_RAND_OUT: TODO
case H_RAND_GENERATOR_OUT: TODO
case H_RAND_LIKE: set(op.tensor, at::redispatch::rand_like(ks, )); break;
case H_RANDINT: set(op.tensor, at::redispatch::randint(ks, )); break;
case H_RANDINT_GENERATOR: set(op.tensor, at::redispatch::randint(ks, )); break;
case H_RANDINT_LOW: set(op.tensor, at::redispatch::randint(ks, )); break;
case H_RANDINT_LOW_GENERATOR: set(op.tensor, at::redispatch::randint(ks, )); break;
case H_RANDINT_OUT: TODO
case H_RANDINT_GENERATOR_OUT: TODO
case H_RANDINT_LOW_OUT: TODO
case H_RANDINT_LOW_GENERATOR_OUT: TODO
case H_RANDINT_LIKE: set(op.tensor, at::redispatch::randint_like(ks, )); break;
case H_RANDINT_LIKE_LOW_DTYPE: set(op.tensor, at::redispatch::randint_like(ks, )); break;
case H_RANDN: set(op.tensor, at::redispatch::randn(ks, )); break;
case H_RANDN_GENERATOR: set(op.tensor, at::redispatch::randn(ks, )); break;
case H_RANDN_NAMES: set(op.tensor, at::redispatch::randn(ks, )); break;
case H_RANDN_GENERATOR_WITH_NAMES: set(op.tensor, at::redispatch::randn(ks, )); break;
case H_RANDN_OUT: TODO
case H_RANDN_GENERATOR_OUT: TODO
case H_RANDN_LIKE: set(op.tensor, at::redispatch::randn_like(ks, )); break;
case H_RANDPERM: set(op.tensor, at::redispatch::randperm(ks, )); break;
case H_RANDPERM_GENERATOR: set(op.tensor, at::redispatch::randperm(ks, )); break;
case H_RANDPERM_OUT: TODO
case H_RANDPERM_GENERATOR_OUT: TODO
case H_RANGE_STEP: set(op.tensor, at::redispatch::range(ks, )); break;
case H_RANGE: set(op.tensor, at::redispatch::range(ks, )); break;
case H_RANGE_OUT: TODO
case H_RAVEL: set(op.tensor, at::redispatch::ravel(ks, )); break;
case H_RECIPROCAL_OUT: TODO
case H_NEG: set(op.tensor, at::redispatch::neg(ks, )); break;
case H_NEG_: TODO
case H_NEG_OUT: TODO
case H_NEGATIVE: set(op.tensor, at::redispatch::negative(ks, )); break;
case H_NEGATIVE_: TODO
case H_NEGATIVE_OUT: TODO
case H_REPEAT: set(op.tensor, at::redispatch::repeat(ks, )); break;
case H_REPEAT_INTERLEAVE_TENSOR: set(op.tensor, at::redispatch::repeat_interleave(ks, )); break;
case H_REPEAT_INTERLEAVE_SELF_TENSOR: set(op.tensor, at::redispatch::repeat_interleave(ks, )); break;
case H_REPEAT_INTERLEAVE_SELF_INT: set(op.tensor, at::redispatch::repeat_interleave(ks, )); break;
case H_RESHAPE: set(op.tensor, at::redispatch::reshape(ks, )); break;
case H__MKLDNN_RESHAPE: set(op.tensor, at::redispatch::_mkldnn_reshape(ks, )); break;
case H_RESHAPE_AS: set(op.tensor, at::redispatch::reshape_as(ks, )); break;
case H_ROUND: set(op.tensor, at::redispatch::round(ks, )); break;
case H_ROUND_: TODO
case H_ROUND_OUT: TODO
case H_RRELU: set(op.tensor, at::redispatch::rrelu(ks, )); break;
case H_RRELU_: TODO
case H_RELU: set(op.tensor, at::redispatch::relu(ks, )); break;
case H_RELU_: TODO
case H_RELU6: set(op.tensor, at::redispatch::relu6(ks, )); break;
case H_RELU6_: TODO
case H_PRELU: set(op.tensor, at::redispatch::prelu(ks, )); break;
case H_PRELU_BACKWARD: set(op.tensor, at::redispatch::prelu_backward(ks, )); break;
case H_GELU: set(op.tensor, at::redispatch::gelu(ks, )); break;
case H_GELU_BACKWARD: set(op.tensor, at::redispatch::gelu_backward(ks, )); break;
case H_INFINITELY_DIFFERENTIABLE_GELU_BACKWARD: set(op.tensor, at::redispatch::infinitely_differentiable_gelu_backward(ks, )); break;
case H_HARDSHRINK: set(op.tensor, at::redispatch::hardshrink(ks, )); break;
case H_HARDSHRINK_BACKWARD: set(op.tensor, at::redispatch::hardshrink_backward(ks, )); break;
case H_RSQRT: set(op.tensor, at::redispatch::rsqrt(ks, )); break;
case H_RSQRT_: TODO
case H_RSQRT_OUT: TODO
case H_SELECT_DIMNAME: set(op.tensor, at::redispatch::select(ks, )); break;
case H_SELECT_INT: set(op.tensor, at::redispatch::select(ks, )); break;
case H_SELECT_BACKWARD: set(op.tensor, at::redispatch::select_backward(ks, )); break;
case H_SELU: set(op.tensor, at::redispatch::selu(ks, )); break;
case H_SELU_: TODO
case H_CELU: set(op.tensor, at::redispatch::celu(ks, )); break;
case H_CELU_: TODO
case H_SILU: set(op.tensor, at::redispatch::silu(ks, )); break;
case H_SILU_: TODO
case H_SILU_OUT: TODO
case H_SILU_BACKWARD: set(op.tensor, at::redispatch::silu_backward(ks, )); break;
case H_SIGMOID: set(op.tensor, at::redispatch::sigmoid(ks, )); break;
case H_SIGMOID_: TODO
case H_SIGMOID_OUT: TODO
case H_LOGIT: set(op.tensor, at::redispatch::logit(ks, )); break;
case H_LOGIT_: TODO
case H_LOGIT_OUT: TODO
case H_SIN_OUT: TODO
case H_SINC_OUT: TODO
case H_SINH_OUT: TODO
case H_DETACH: set(op.tensor, at::redispatch::detach(ks, )); break;
case H_DETACH_: TODO
case H_SIZE_DIMNAME: set(op.tensor, at::redispatch::size(ks, )); break;
case H_SLICE_TENSOR: set(op.tensor, at::redispatch::slice(ks, )); break;
case H_SLICE_BACKWARD: set(op.tensor, at::redispatch::slice_backward(ks, )); break;
case H_SLOGDET: set(op.tensor, at::redispatch::slogdet(ks, )); break;
case H_SMM: set(op.tensor, at::redispatch::smm(ks, )); break;
case H_SOFTMAX_INT: set(op.tensor, at::redispatch::softmax(ks, )); break;
case H_SOFTMAX_DIMNAME: set(op.tensor, at::redispatch::softmax(ks, )); break;
case H__SOFTMAX: set(op.tensor, at::redispatch::_softmax(ks, )); break;
case H__SOFTMAX_BACKWARD_DATA: set(op.tensor, at::redispatch::_softmax_backward_data(ks, )); break;
case H_UNSAFE_SPLIT_TENSOR: set(op.tensor, at::redispatch::unsafe_split(ks, )); break;
case H_SPLIT_TENSOR: set(op.tensor, at::redispatch::split(ks, )); break;
case H_UNSAFE_SPLIT_WITH_SIZES: set(op.tensor, at::redispatch::unsafe_split_with_sizes(ks, )); break;
case H_SPLIT_WITH_SIZES: set(op.tensor, at::redispatch::split_with_sizes(ks, )); break;
case H_SQUEEZE: set(op.tensor, at::redispatch::squeeze(ks, )); break;
case H_SQUEEZE_DIM: set(op.tensor, at::redispatch::squeeze(ks, )); break;
case H_SQUEEZE_DIMNAME: set(op.tensor, at::redispatch::squeeze(ks, )); break;
case H_SQUEEZE_: TODO
case H_SQUEEZE__DIM: TODO
case H_SQUEEZE__DIMNAME: TODO
case H_SSPADDMM: set(op.tensor, at::redispatch::sspaddmm(ks, )); break;
case H_SSPADDMM_OUT: TODO
case H_STACK: set(op.tensor, at::redispatch::stack(ks, )); break;
case H_STACK_OUT: TODO
case H__STACK: set(op.tensor, at::redispatch::_stack(ks, )); break;
case H__STACK_OUT: TODO
case H_HSTACK: set(op.tensor, at::redispatch::hstack(ks, )); break;
case H_HSTACK_OUT: TODO
case H_VSTACK: set(op.tensor, at::redispatch::vstack(ks, )); break;
case H_VSTACK_OUT: TODO
case H_DSTACK: set(op.tensor, at::redispatch::dstack(ks, )); break;
case H_DSTACK_OUT: TODO
case H_STFT: set(op.tensor, at::redispatch::stft(ks, )); break;
case H_ISTFT: set(op.tensor, at::redispatch::istft(ks, )); break;
case H_STRIDE_DIMNAME: set(op.tensor, at::redispatch::stride(ks, )); break;
case H_SUM: set(op.tensor, at::redispatch::sum(ks, )); break;
case H_SUM_DIM_INTLIST: set(op.tensor, at::redispatch::sum(ks, )); break;
case H_SUM_DIM_DIMNAMELIST: set(op.tensor, at::redispatch::sum(ks, )); break;
case H_SUM_INTLIST_OUT: TODO
case H_SUM_DIMNAMELIST_OUT: TODO
case H_NANSUM: set(op.tensor, at::redispatch::nansum(ks, )); break;
case H_NANSUM_DIM_INTLIST: set(op.tensor, at::redispatch::nansum(ks, )); break;
case H_NANSUM_INTLIST_OUT: TODO
case H_SUM_TO_SIZE: set(op.tensor, at::redispatch::sum_to_size(ks, )); break;
case H_SQRT: set(op.tensor, at::redispatch::sqrt(ks, )); break;
case H_SQRT_OUT: TODO
case H_SQUARE: set(op.tensor, at::redispatch::square(ks, )); break;
case H_SQUARE_: TODO
case H_SQUARE_OUT: TODO
case H_STD: set(op.tensor, at::redispatch::std(ks, )); break;
case H_STD_DIM: set(op.tensor, at::redispatch::std(ks, )); break;
case H_STD_MEAN: set(op.tensor, at::redispatch::std_mean(ks, )); break;
case H_STD_MEAN_DIM: set(op.tensor, at::redispatch::std_mean(ks, )); break;
case H_STD_MEAN_NAMES_DIM: set(op.tensor, at::redispatch::std_mean(ks, )); break;
case H_STD_OUT: TODO
case H_STD_NAMES_DIM: set(op.tensor, at::redispatch::std(ks, )); break;
case H_STD_NAMES_OUT: TODO
case H_PROD: set(op.tensor, at::redispatch::prod(ks, )); break;
case H_PROD_DIM_INT: set(op.tensor, at::redispatch::prod(ks, )); break;
case H_PROD_INT_OUT: TODO
case H_PROD_DIM_DIMNAME: set(op.tensor, at::redispatch::prod(ks, )); break;
case H_PROD_DIMNAME_OUT: TODO
case H_T: set(op.tensor, at::redispatch::t(ks, )); break;
case H_T_: TODO
case H_TAN_OUT: TODO
case H_TANH: set(op.tensor, at::redispatch::tanh(ks, )); break;
case H_TANH_: TODO
case H_TANH_OUT: TODO
case H_TENSORDOT: set(op.tensor, at::redispatch::tensordot(ks, )); break;
case H_TENSORDOT_OUT: TODO
case H_THRESHOLD: set(op.tensor, at::redispatch::threshold(ks, )); break;
case H_THRESHOLD_: TODO
case H_THRESHOLD_OUT: TODO
case H_THRESHOLD_BACKWARD: set(op.tensor, at::redispatch::threshold_backward(ks, )); break;
case H_TILE: set(op.tensor, at::redispatch::tile(ks, )); break;
case H_TRANSPOSE_INT: set(op.tensor, at::redispatch::transpose(ks, )); break;
case H_TRANSPOSE_DIMNAME: set(op.tensor, at::redispatch::transpose(ks, )); break;
case H__MKLDNN_TRANSPOSE: set(op.tensor, at::redispatch::_mkldnn_transpose(ks, )); break;
case H_TRANSPOSE_: TODO
case H__MKLDNN_TRANSPOSE_: TODO
case H_ONE_HOT: set(op.tensor, at::redispatch::one_hot(ks, )); break;
case H_FLIP: set(op.tensor, at::redispatch::flip(ks, )); break;
case H_FLIPLR: set(op.tensor, at::redispatch::fliplr(ks, )); break;
case H_FLIPUD: set(op.tensor, at::redispatch::flipud(ks, )); break;
case H_ROLL: set(op.tensor, at::redispatch::roll(ks, )); break;
case H_ROT90: set(op.tensor, at::redispatch::rot90(ks, )); break;
case H_TRAPZ_X: set(op.tensor, at::redispatch::trapz(ks, )); break;
case H_TRAPZ_DX: set(op.tensor, at::redispatch::trapz(ks, )); break;
case H__TRILINEAR: set(op.tensor, at::redispatch::_trilinear(ks, )); break;
case H_TRIPLET_MARGIN_LOSS: set(op.tensor, at::redispatch::triplet_margin_loss(ks, )); break;
case H_TRUNC: set(op.tensor, at::redispatch::trunc(ks, )); break;
case H_TRUNC_: TODO
case H_TRUNC_OUT: TODO
case H_FIX: set(op.tensor, at::redispatch::fix(ks, )); break;
case H_FIX_: TODO
case H_FIX_OUT: TODO
case H_TYPE_AS: set(op.tensor, at::redispatch::type_as(ks, )); break;
case H__HAS_COMPATIBLE_SHALLOW_COPY_TYPE: set(op.tensor, at::redispatch::_has_compatible_shallow_copy_type(ks, )); break;
case H__UNIQUE: set(op.tensor, at::redispatch::_unique(ks, )); break;
case H_UNIQUE_DIM: set(op.tensor, at::redispatch::unique_dim(ks, )); break;
case H_UNIQUE_CONSECUTIVE: set(op.tensor, at::redispatch::unique_consecutive(ks, )); break;
case H_UNIQUE_DIM_CONSECUTIVE: set(op.tensor, at::redispatch::unique_dim_consecutive(ks, )); break;
case H__UNIQUE2: set(op.tensor, at::redispatch::_unique2(ks, )); break;
case H__UNSAFE_VIEW: set(op.tensor, at::redispatch::_unsafe_view(ks, )); break;
case H_UNSQUEEZE: set(op.tensor, at::redispatch::unsqueeze(ks, )); break;
case H_UNSQUEEZE_: TODO
case H_VANDER: set(op.tensor, at::redispatch::vander(ks, )); break;
case H_VAR: set(op.tensor, at::redispatch::var(ks, )); break;
case H_VAR_DIM: set(op.tensor, at::redispatch::var(ks, )); break;
case H_VAR_OUT: TODO
case H_VAR_NAMES_DIM: set(op.tensor, at::redispatch::var(ks, )); break;
case H_VAR_NAMES_OUT: TODO
case H_VAR_MEAN: set(op.tensor, at::redispatch::var_mean(ks, )); break;
case H_VAR_MEAN_DIM: set(op.tensor, at::redispatch::var_mean(ks, )); break;
case H_VAR_MEAN_NAMES_DIM: set(op.tensor, at::redispatch::var_mean(ks, )); break;
case H_VIEW_AS: set(op.tensor, at::redispatch::view_as(ks, )); break;
case H_WHERE_SELF: set(op.tensor, at::redispatch::where(ks, )); break;
case H_WHERE_SCALARSELF: set(op.tensor, at::redispatch::where(ks, )); break;
case H_WHERE_SCALAROTHER: set(op.tensor, at::redispatch::where(ks, )); break;
case H_WHERE_SCALAR: set(op.tensor, at::redispatch::where(ks, )); break;
case H_WHERE: set(op.tensor, at::redispatch::where(ks, )); break;
case H__S_WHERE: set(op.tensor, at::redispatch::_s_where(ks, )); break;
case H_NORM_EXCEPT_DIM: set(op.tensor, at::redispatch::norm_except_dim(ks, )); break;
case H__WEIGHT_NORM: set(op.tensor, at::redispatch::_weight_norm(ks, )); break;
case H__WEIGHT_NORM_CUDA_INTERFACE: set(op.tensor, at::redispatch::_weight_norm_cuda_interface(ks, )); break;
case H__WEIGHT_NORM_CUDA_INTERFACE_BACKWARD: set(op.tensor, at::redispatch::_weight_norm_cuda_interface_backward(ks, )); break;
case H__WEIGHT_NORM_DIFFERENTIABLE_BACKWARD: set(op.tensor, at::redispatch::_weight_norm_differentiable_backward(ks, )); break;
case H_ZEROS_NAMES: set(op.tensor, at::redispatch::zeros(ks, )); break;
case H_ZEROS: set(op.tensor, at::redispatch::zeros(ks, )); break;
case H_ZEROS_OUT: TODO
case H_ZEROS_LIKE: set(op.tensor, at::redispatch::zeros_like(ks, )); break;
case H__STANDARD_GAMMA_GRAD: set(op.tensor, at::redispatch::_standard_gamma_grad(ks, )); break;
case H__STANDARD_GAMMA: set(op.tensor, at::redispatch::_standard_gamma(ks, )); break;
case H__DIRICHLET_GRAD: set(op.tensor, at::redispatch::_dirichlet_grad(ks, )); break;
case H__SAMPLE_DIRICHLET: set(op.tensor, at::redispatch::_sample_dirichlet(ks, )); break;
case H_POISSON: set(op.tensor, at::redispatch::poisson(ks, )); break;
case H_BINOMIAL: set(op.tensor, at::redispatch::binomial(ks, )); break;
case H_NATIVE_NORM: set(op.tensor, at::redispatch::native_norm(ks, )); break;
case H_NATIVE_NORM_SCALAROPT_DIM_DTYPE: set(op.tensor, at::redispatch::native_norm(ks, )); break;
case H__SPARSE_SUM: set(op.tensor, at::redispatch::_sparse_sum(ks, )); break;
case H__SPARSE_SUM_DTYPE: set(op.tensor, at::redispatch::_sparse_sum(ks, )); break;
case H__SPARSE_SUM_DIM: set(op.tensor, at::redispatch::_sparse_sum(ks, )); break;
case H__SPARSE_SUM_DIM_DTYPE: set(op.tensor, at::redispatch::_sparse_sum(ks, )); break;
case H__SPARSE_SUM_BACKWARD: set(op.tensor, at::redispatch::_sparse_sum_backward(ks, )); break;
case H__SPARSE_SOFTMAX_INT: set(op.tensor, at::redispatch::_sparse_softmax(ks, )); break;
case H__SPARSE_SOFTMAX_DIMNAME: set(op.tensor, at::redispatch::_sparse_softmax(ks, )); break;
case H__SPARSE_SOFTMAX: set(op.tensor, at::redispatch::_sparse_softmax(ks, )); break;
case H__SPARSE_SOFTMAX_BACKWARD_DATA: set(op.tensor, at::redispatch::_sparse_softmax_backward_data(ks, )); break;
case H__SPARSE_LOG_SOFTMAX_INT: set(op.tensor, at::redispatch::_sparse_log_softmax(ks, )); break;
case H__SPARSE_LOG_SOFTMAX_DIMNAME: set(op.tensor, at::redispatch::_sparse_log_softmax(ks, )); break;
case H__SPARSE_LOG_SOFTMAX: set(op.tensor, at::redispatch::_sparse_log_softmax(ks, )); break;
case H__SPARSE_LOG_SOFTMAX_BACKWARD_DATA: set(op.tensor, at::redispatch::_sparse_log_softmax_backward_data(ks, )); break;
case H_NORM_SCALAROPT_DTYPE: set(op.tensor, at::redispatch::norm(ks, )); break;
case H_NORM_SCALAR: set(op.tensor, at::redispatch::norm(ks, )); break;
case H_NORM_SCALAROPT_DIM_DTYPE: set(op.tensor, at::redispatch::norm(ks, )); break;
case H_NORM_SCALAROPT_DIM: set(op.tensor, at::redispatch::norm(ks, )); break;
case H_NORM_DTYPE_OUT: TODO
case H_NORM_OUT: TODO
case H_NORM_NAMES_SCALAROPT_DIM_DTYPE: set(op.tensor, at::redispatch::norm(ks, )); break;
case H_NORM_NAMES_SCALAROPT_DIM: set(op.tensor, at::redispatch::norm(ks, )); break;
case H_NORM_NAMES_DTYPE_OUT: TODO
case H_NORM_NAMES_OUT: TODO
case H_FREXP_TENSOR: set(op.tensor, at::redispatch::frexp(ks, )); break;
case H_FREXP_TENSOR_OUT: TODO
case H_FROBENIUS_NORM: set(op.tensor, at::redispatch::frobenius_norm(ks, )); break;
case H_FROBENIUS_NORM_DIM: set(op.tensor, at::redispatch::frobenius_norm(ks, )); break;
case H_FROBENIUS_NORM_OUT: TODO
case H_NUCLEAR_NORM: set(op.tensor, at::redispatch::nuclear_norm(ks, )); break;
case H_NUCLEAR_NORM_OUT: TODO
case H_NUCLEAR_NORM_DIM: set(op.tensor, at::redispatch::nuclear_norm(ks, )); break;
case H_NUCLEAR_NORM_DIM_OUT: TODO
case H_CLONE: set(op.tensor, at::redispatch::clone(ks, )); break;
case H_RESIZE_AS_: TODO
case H_RESIZE_AS_SPARSE_: TODO
case H_ZERO_: TODO
case H_SUB_OUT: TODO
case H_SUB_TENSOR: set(op.tensor, at::redispatch::sub(ks, )); break;
case H_SUB__TENSOR: TODO
case H_SUB_SCALAR: set(op.tensor, at::redispatch::sub(ks, )); break;
case H_SUB__SCALAR: TODO
case H_SUBTRACT_OUT: TODO
case H_SUBTRACT_TENSOR: set(op.tensor, at::redispatch::subtract(ks, )); break;
case H_SUBTRACT__TENSOR: TODO
case H_SUBTRACT_SCALAR: set(op.tensor, at::redispatch::subtract(ks, )); break;
case H_SUBTRACT__SCALAR: TODO
case H_RSUB_TENSOR: set(op.tensor, at::redispatch::rsub(ks, )); break;
case H_HEAVISIDE_OUT: TODO
case H_HEAVISIDE: set(op.tensor, at::redispatch::heaviside(ks, )); break;
case H_HEAVISIDE_: TODO
case H_RSUB_SCALAR: set(op.tensor, at::redispatch::rsub(ks, )); break;
case H__SPARSE_ADDMM: set(op.tensor, at::redispatch::_sparse_addmm(ks, )); break;
case H_ADDMM_OUT: TODO
case H_ADDMM: set(op.tensor, at::redispatch::addmm(ks, )); break;
case H_ADDMM_: TODO
case H_SPARSE_CSR_TENSOR_CROW_COL_VALUE_SIZE: set(op.tensor, at::redispatch::sparse_csr_tensor(ks, )); break;
case H_SPARSE_CSR_TENSOR_CROW_COL_VALUE: set(op.tensor, at::redispatch::sparse_csr_tensor(ks, )); break;
case H_SPARSE_COO_TENSOR_SIZE: set(op.tensor, at::redispatch::sparse_coo_tensor(ks, )); break;
case H_SPARSE_COO_TENSOR_INDICES: set(op.tensor, at::redispatch::sparse_coo_tensor(ks, )); break;
case H_SPARSE_COO_TENSOR_INDICES_SIZE: set(op.tensor, at::redispatch::sparse_coo_tensor(ks, )); break;
case H__SPARSE_COO_TENSOR_UNSAFE: set(op.tensor, at::redispatch::_sparse_coo_tensor_unsafe(ks, )); break;
case H__VALIDATE_SPARSE_COO_TENSOR_ARGS: set(op.tensor, at::redispatch::_validate_sparse_coo_tensor_args(ks, )); break;
case H__SPARSE_COO_TENSOR_WITH_DIMS: set(op.tensor, at::redispatch::_sparse_coo_tensor_with_dims(ks, )); break;
case H__SPARSE_COO_TENSOR_WITH_DIMS_AND_TENSORS: set(op.tensor, at::redispatch::_sparse_coo_tensor_with_dims_and_tensors(ks, )); break;
case H_SPARSE_RESIZE_: TODO
case H_SPARSE_RESIZE_AND_CLEAR_: TODO
case H_SPARSE_MASK: set(op.tensor, at::redispatch::sparse_mask(ks, )); break;
case H_TO_DENSE: set(op.tensor, at::redispatch::to_dense(ks, )); break;
case H_TO_DENSE_BACKWARD: set(op.tensor, at::redispatch::to_dense_backward(ks, )); break;
case H_SPARSE_DIM: set(op.tensor, at::redispatch::sparse_dim(ks, )); break;
case H__DIMI: set(op.tensor, at::redispatch::_dimI(ks, )); break;
case H_DENSE_DIM: set(op.tensor, at::redispatch::dense_dim(ks, )); break;
case H__DIMV: set(op.tensor, at::redispatch::_dimV(ks, )); break;
case H__NNZ: set(op.tensor, at::redispatch::_nnz(ks, )); break;
case H_COALESCE: set(op.tensor, at::redispatch::coalesce(ks, )); break;
case H__COALESCE: set(op.tensor, at::redispatch::_coalesce(ks, )); break;
case H_IS_COALESCED: set(op.tensor, at::redispatch::is_coalesced(ks, )); break;
case H__INDICES: set(op.tensor, at::redispatch::_indices(ks, )); break;
case H__VALUES: set(op.tensor, at::redispatch::_values(ks, )); break;
case H__COALESCED_: TODO
case H_INDICES: set(op.tensor, at::redispatch::indices(ks, )); break;
case H_VALUES: set(op.tensor, at::redispatch::values(ks, )); break;
case H_CROW_INDICES: set(op.tensor, at::redispatch::crow_indices(ks, )); break;
case H_COL_INDICES: set(op.tensor, at::redispatch::col_indices(ks, )); break;
case H_HSPMM_OUT: TODO
case H_HSPMM: set(op.tensor, at::redispatch::hspmm(ks, )); break;
case H_COPY_SPARSE_TO_SPARSE_: TODO
case H_UNBIND_INT: set(op.tensor, at::redispatch::unbind(ks, )); break;
case H_UNBIND_DIMNAME: set(op.tensor, at::redispatch::unbind(ks, )); break;
case H_TO_SPARSE_SPARSE_DIM: set(op.tensor, at::redispatch::to_sparse(ks, )); break;
case H_TO_SPARSE: set(op.tensor, at::redispatch::to_sparse(ks, )); break;
case H_TO_MKLDNN: set(op.tensor, at::redispatch::to_mkldnn(ks, )); break;
case H_MKLDNN_REORDER_CONV2D_WEIGHT: set(op.tensor, at::redispatch::mkldnn_reorder_conv2d_weight(ks, )); break;
case H_MKLDNN_REORDER_CONV3D_WEIGHT: set(op.tensor, at::redispatch::mkldnn_reorder_conv3d_weight(ks, )); break;
case H_TO_MKLDNN_BACKWARD: set(op.tensor, at::redispatch::to_mkldnn_backward(ks, )); break;
case H_QUANTIZE_PER_TENSOR: set(op.tensor, at::redispatch::quantize_per_tensor(ks, )); break;
case H_QUANTIZE_PER_TENSOR_TENSORS: set(op.tensor, at::redispatch::quantize_per_tensor(ks, )); break;
case H_QUANTIZE_PER_CHANNEL: set(op.tensor, at::redispatch::quantize_per_channel(ks, )); break;
case H_DEQUANTIZE_SELF: set(op.tensor, at::redispatch::dequantize(ks, )); break;
case H_DEQUANTIZE_TENSORS: set(op.tensor, at::redispatch::dequantize(ks, )); break;
case H_Q_SCALE: set(op.tensor, at::redispatch::q_scale(ks, )); break;
case H_Q_ZERO_POINT: set(op.tensor, at::redispatch::q_zero_point(ks, )); break;
case H_Q_PER_CHANNEL_SCALES: set(op.tensor, at::redispatch::q_per_channel_scales(ks, )); break;
case H_Q_PER_CHANNEL_ZERO_POINTS: set(op.tensor, at::redispatch::q_per_channel_zero_points(ks, )); break;
case H_Q_PER_CHANNEL_AXIS: set(op.tensor, at::redispatch::q_per_channel_axis(ks, )); break;
case H_INT_REPR: set(op.tensor, at::redispatch::int_repr(ks, )); break;
case H__MAKE_PER_TENSOR_QUANTIZED_TENSOR: set(op.tensor, at::redispatch::_make_per_tensor_quantized_tensor(ks, )); break;
case H__MAKE_PER_CHANNEL_QUANTIZED_TENSOR: set(op.tensor, at::redispatch::_make_per_channel_quantized_tensor(ks, )); break;
case H_QSCHEME: set(op.tensor, at::redispatch::qscheme(ks, )); break;
case H_FAKE_QUANTIZE_PER_TENSOR_AFFINE: set(op.tensor, at::redispatch::fake_quantize_per_tensor_affine(ks, )); break;
case H_FAKE_QUANTIZE_PER_TENSOR_AFFINE_CACHEMASK: set(op.tensor, at::redispatch::fake_quantize_per_tensor_affine_cachemask(ks, )); break;
case H_FAKE_QUANTIZE_PER_TENSOR_AFFINE_CACHEMASK_BACKWARD: set(op.tensor, at::redispatch::fake_quantize_per_tensor_affine_cachemask_backward(ks, )); break;
case H__FAKE_QUANTIZE_LEARNABLE_PER_TENSOR_AFFINE: set(op.tensor, at::redispatch::_fake_quantize_learnable_per_tensor_affine(ks, )); break;
case H__FAKE_QUANTIZE_LEARNABLE_PER_TENSOR_AFFINE_BACKWARD: set(op.tensor, at::redispatch::_fake_quantize_learnable_per_tensor_affine_backward(ks, )); break;
case H_FAKE_QUANTIZE_PER_CHANNEL_AFFINE: set(op.tensor, at::redispatch::fake_quantize_per_channel_affine(ks, )); break;
case H_FAKE_QUANTIZE_PER_CHANNEL_AFFINE_CACHEMASK: set(op.tensor, at::redispatch::fake_quantize_per_channel_affine_cachemask(ks, )); break;
case H_FAKE_QUANTIZE_PER_CHANNEL_AFFINE_CACHEMASK_BACKWARD: set(op.tensor, at::redispatch::fake_quantize_per_channel_affine_cachemask_backward(ks, )); break;
case H__FAKE_QUANTIZE_LEARNABLE_PER_CHANNEL_AFFINE: set(op.tensor, at::redispatch::_fake_quantize_learnable_per_channel_affine(ks, )); break;
case H__FAKE_QUANTIZE_LEARNABLE_PER_CHANNEL_AFFINE_BACKWARD: set(op.tensor, at::redispatch::_fake_quantize_learnable_per_channel_affine_backward(ks, )); break;
case H__CHOOSE_QPARAMS_PER_TENSOR: set(op.tensor, at::redispatch::_choose_qparams_per_tensor(ks, )); break;
case H__SATURATE_WEIGHT_TO_FP16: set(op.tensor, at::redispatch::_saturate_weight_to_fp16(ks, )); break;
case H_CHOOSE_QPARAMS_OPTIMIZED: set(op.tensor, at::redispatch::choose_qparams_optimized(ks, )); break;
case H_TO_DTYPE_LAYOUT: set(op.tensor, at::redispatch::to(ks, )); break;
case H_TO_DEVICE: set(op.tensor, at::redispatch::to(ks, )); break;
case H_TO_DTYPE: set(op.tensor, at::redispatch::to(ks, )); break;
case H_TO_OTHER: set(op.tensor, at::redispatch::to(ks, )); break;
case H_MESHGRID: set(op.tensor, at::redispatch::meshgrid(ks, )); break;
case H_CARTESIAN_PROD: set(op.tensor, at::redispatch::cartesian_prod(ks, )); break;
case H_COMBINATIONS: set(op.tensor, at::redispatch::combinations(ks, )); break;
case H_ITEM: set(op.tensor, at::redispatch::item(ks, )); break;
case H_RESULT_TYPE_TENSOR: set(op.tensor, at::redispatch::result_type(ks, )); break;
case H_RESULT_TYPE_SCALAR: set(op.tensor, at::redispatch::result_type(ks, )); break;
case H_RESULT_TYPE_SCALAR_TENSOR: set(op.tensor, at::redispatch::result_type(ks, )); break;
case H_RESULT_TYPE_SCALAR_SCALAR: set(op.tensor, at::redispatch::result_type(ks, )); break;
case H_CAN_CAST: set(op.tensor, at::redispatch::can_cast(ks, )); break;
case H_PROMOTE_TYPES: set(op.tensor, at::redispatch::promote_types(ks, )); break;
case H__LOCAL_SCALAR_DENSE: set(op.tensor, at::redispatch::_local_scalar_dense(ks, )); break;
case H__THNN_FUSED_LSTM_CELL: set(op.tensor, at::redispatch::_thnn_fused_lstm_cell(ks, )); break;
case H__THNN_FUSED_LSTM_CELL_BACKWARD: set(op.tensor, at::redispatch::_thnn_fused_lstm_cell_backward(ks, )); break;
case H__THNN_DIFFERENTIABLE_LSTM_CELL_BACKWARD: set(op.tensor, at::redispatch::_thnn_differentiable_lstm_cell_backward(ks, )); break;
case H__THNN_FUSED_GRU_CELL: set(op.tensor, at::redispatch::_thnn_fused_gru_cell(ks, )); break;
case H__THNN_FUSED_GRU_CELL_BACKWARD: set(op.tensor, at::redispatch::_thnn_fused_gru_cell_backward(ks, )); break;
case H__THNN_DIFFERENTIABLE_GRU_CELL_BACKWARD: set(op.tensor, at::redispatch::_thnn_differentiable_gru_cell_backward(ks, )); break;
case H_LSTM_INPUT: set(op.tensor, at::redispatch::lstm(ks, )); break;
case H_LSTM_DATA: set(op.tensor, at::redispatch::lstm(ks, )); break;
case H_GRU_INPUT: set(op.tensor, at::redispatch::gru(ks, )); break;
case H_GRU_DATA: set(op.tensor, at::redispatch::gru(ks, )); break;
case H_RNN_TANH_INPUT: set(op.tensor, at::redispatch::rnn_tanh(ks, )); break;
case H_RNN_TANH_DATA: set(op.tensor, at::redispatch::rnn_tanh(ks, )); break;
case H_RNN_RELU_INPUT: set(op.tensor, at::redispatch::rnn_relu(ks, )); break;
case H_RNN_RELU_DATA: set(op.tensor, at::redispatch::rnn_relu(ks, )); break;
case H_LSTM_CELL: set(op.tensor, at::redispatch::lstm_cell(ks, )); break;
case H_GRU_CELL: set(op.tensor, at::redispatch::gru_cell(ks, )); break;
case H_RNN_TANH_CELL: set(op.tensor, at::redispatch::rnn_tanh_cell(ks, )); break;
case H_RNN_RELU_CELL: set(op.tensor, at::redispatch::rnn_relu_cell(ks, )); break;
case H_QUANTIZED_LSTM_CELL: set(op.tensor, at::redispatch::quantized_lstm_cell(ks, )); break;
case H_QUANTIZED_GRU_CELL: set(op.tensor, at::redispatch::quantized_gru_cell(ks, )); break;
case H_QUANTIZED_RNN_RELU_CELL: set(op.tensor, at::redispatch::quantized_rnn_relu_cell(ks, )); break;
case H_QUANTIZED_RNN_TANH_CELL: set(op.tensor, at::redispatch::quantized_rnn_tanh_cell(ks, )); break;
case H__PACK_PADDED_SEQUENCE: set(op.tensor, at::redispatch::_pack_padded_sequence(ks, )); break;
case H__PACK_PADDED_SEQUENCE_BACKWARD: set(op.tensor, at::redispatch::_pack_padded_sequence_backward(ks, )); break;
case H__PAD_PACKED_SEQUENCE: set(op.tensor, at::redispatch::_pad_packed_sequence(ks, )); break;
case H_SET__SOURCE_STORAGE: TODO
case H_SET__SOURCE_STORAGE_STORAGE_OFFSET: TODO
case H_SET__SOURCE_TENSOR: TODO
case H_SET_: TODO
case H_IS_SET_TO: set(op.tensor, at::redispatch::is_set_to(ks, )); break;
case H_MASKED_FILL__SCALAR: TODO
case H_MASKED_FILL_SCALAR: set(op.tensor, at::redispatch::masked_fill(ks, )); break;
case H_MASKED_FILL__TENSOR: TODO
case H_MASKED_FILL_TENSOR: set(op.tensor, at::redispatch::masked_fill(ks, )); break;
case H_MASKED_SCATTER_: TODO
case H_MASKED_SCATTER: set(op.tensor, at::redispatch::masked_scatter(ks, )); break;
case H_VIEW: set(op.tensor, at::redispatch::view(ks, )); break;
case H_VIEW_DTYPE: set(op.tensor, at::redispatch::view(ks, )); break;
case H_PUT_: TODO
case H_PUT: set(op.tensor, at::redispatch::put(ks, )); break;
case H_INDEX_ADD_: TODO
case H_INDEX_ADD__ALPHA: TODO
case H_INDEX_ADD: set(op.tensor, at::redispatch::index_add(ks, )); break;
case H_INDEX_ADD_ALPHA: set(op.tensor, at::redispatch::index_add(ks, )); break;
case H_INDEX_ADD_DIMNAME: set(op.tensor, at::redispatch::index_add(ks, )); break;
case H_INDEX_FILL__INT_SCALAR: TODO
case H_INDEX_FILL_INT_SCALAR: set(op.tensor, at::redispatch::index_fill(ks, )); break;
case H_INDEX_FILL__INT_TENSOR: TODO
case H_INDEX_FILL_INT_TENSOR: set(op.tensor, at::redispatch::index_fill(ks, )); break;
case H_INDEX_FILL__DIMNAME_SCALAR: TODO
case H_INDEX_FILL__DIMNAME_TENSOR: TODO
case H_INDEX_FILL_DIMNAME_SCALAR: set(op.tensor, at::redispatch::index_fill(ks, )); break;
case H_INDEX_FILL_DIMNAME_TENSOR: set(op.tensor, at::redispatch::index_fill(ks, )); break;
case H_SCATTER__SRC: TODO
case H_SCATTER_SRC: set(op.tensor, at::redispatch::scatter(ks, )); break;
case H_SCATTER__VALUE: TODO
case H_SCATTER_VALUE: set(op.tensor, at::redispatch::scatter(ks, )); break;
case H_SCATTER_DIMNAME_SRC: set(op.tensor, at::redispatch::scatter(ks, )); break;
case H_SCATTER_DIMNAME_VALUE: set(op.tensor, at::redispatch::scatter(ks, )); break;
case H_SCATTER__REDUCE: TODO
case H_SCATTER__VALUE_REDUCE: TODO
case H_SCATTER_ADD_: TODO
case H_SCATTER_ADD: set(op.tensor, at::redispatch::scatter_add(ks, )); break;
case H_SCATTER_ADD_DIMNAME: set(op.tensor, at::redispatch::scatter_add(ks, )); break;
case H_EQ__SCALAR: TODO
case H_EQ__TENSOR: TODO
case H_BITWISE_AND_TENSOR_OUT: TODO
case H_BITWISE_AND_SCALAR_OUT: TODO
case H_BITWISE_AND_SCALAR: set(op.tensor, at::redispatch::bitwise_and(ks, )); break;
case H_BITWISE_AND_TENSOR: set(op.tensor, at::redispatch::bitwise_and(ks, )); break;
case H_BITWISE_AND__SCALAR: TODO
case H_BITWISE_AND__TENSOR: TODO
case H___AND___SCALAR: set(op.tensor, at::redispatch::__and__(ks, )); break;
case H___AND___TENSOR: set(op.tensor, at::redispatch::__and__(ks, )); break;
case H___IAND___SCALAR: TODO
case H___IAND___TENSOR: TODO
case H_BITWISE_OR_TENSOR_OUT: TODO
case H_BITWISE_OR_SCALAR_OUT: TODO
case H_BITWISE_OR_SCALAR: set(op.tensor, at::redispatch::bitwise_or(ks, )); break;
case H_BITWISE_OR_TENSOR: set(op.tensor, at::redispatch::bitwise_or(ks, )); break;
case H_BITWISE_OR__SCALAR: TODO
case H_BITWISE_OR__TENSOR: TODO
case H___OR___SCALAR: set(op.tensor, at::redispatch::__or__(ks, )); break;
case H___OR___TENSOR: set(op.tensor, at::redispatch::__or__(ks, )); break;
case H___IOR___SCALAR: TODO
case H___IOR___TENSOR: TODO
case H_BITWISE_XOR_TENSOR_OUT: TODO
case H_BITWISE_XOR_SCALAR_OUT: TODO
case H_BITWISE_XOR_SCALAR: set(op.tensor, at::redispatch::bitwise_xor(ks, )); break;
case H_BITWISE_XOR_TENSOR: set(op.tensor, at::redispatch::bitwise_xor(ks, )); break;
case H_BITWISE_XOR__SCALAR: TODO
case H_BITWISE_XOR__TENSOR: TODO
case H___XOR___SCALAR: set(op.tensor, at::redispatch::__xor__(ks, )); break;
case H___XOR___TENSOR: set(op.tensor, at::redispatch::__xor__(ks, )); break;
case H___IXOR___SCALAR: TODO
case H___IXOR___TENSOR: TODO
case H___LSHIFT___SCALAR: set(op.tensor, at::redispatch::__lshift__(ks, )); break;
case H___LSHIFT___TENSOR: set(op.tensor, at::redispatch::__lshift__(ks, )); break;
case H___ILSHIFT___SCALAR: TODO
case H___ILSHIFT___TENSOR: TODO
case H___RSHIFT___SCALAR: set(op.tensor, at::redispatch::__rshift__(ks, )); break;
case H___RSHIFT___TENSOR: set(op.tensor, at::redispatch::__rshift__(ks, )); break;
case H___IRSHIFT___SCALAR: TODO
case H___IRSHIFT___TENSOR: TODO
case H_TRIL_: TODO
case H_TRIU_: TODO
case H_RENORM_: TODO
case H_LERP__SCALAR: TODO
case H_LERP__TENSOR: TODO
case H_FMOD__SCALAR: TODO
case H_FMOD__TENSOR: TODO
case H_REMAINDER__SCALAR: TODO
case H_REMAINDER__TENSOR: TODO
case H_ADDBMM_: TODO
case H_ADDBMM_OUT: TODO
case H_ADDBMM: set(op.tensor, at::redispatch::addbmm(ks, )); break;
case H_ADDCDIV_: TODO
case H_RANDOM__FROM: TODO
case H_RANDOM__TO: TODO
case H_RANDOM_: TODO
case H_UNIFORM_: TODO
case H_CAUCHY_: TODO
case H_LOG_NORMAL_: TODO
case H_EXPONENTIAL_: TODO
case H_GEOMETRIC_: TODO
case H_DIAG_OUT: TODO
case H_DIAG: set(op.tensor, at::redispatch::diag(ks, )); break;
case H_DIAG_BACKWARD: set(op.tensor, at::redispatch::diag_backward(ks, )); break;
case H_CROSS_OUT: TODO
case H_CROSS: set(op.tensor, at::redispatch::cross(ks, )); break;
case H_TRIU_OUT: TODO
case H_TRIU: set(op.tensor, at::redispatch::triu(ks, )); break;
case H_TRIL_OUT: TODO
case H_TRIL: set(op.tensor, at::redispatch::tril(ks, )); break;
case H_TRIL_INDICES: set(op.tensor, at::redispatch::tril_indices(ks, )); break;
case H_TRIU_INDICES: set(op.tensor, at::redispatch::triu_indices(ks, )); break;
case H_TRACE: set(op.tensor, at::redispatch::trace(ks, )); break;
case H_TRACE_BACKWARD: set(op.tensor, at::redispatch::trace_backward(ks, )); break;
case H_NE_SCALAR_OUT: TODO
case H_NE_SCALAR: set(op.tensor, at::redispatch::ne(ks, )); break;
case H_NE_TENSOR_OUT: TODO
case H_NE_TENSOR: set(op.tensor, at::redispatch::ne(ks, )); break;
case H_NE__SCALAR: TODO
case H_NE__TENSOR: TODO
case H_NOT_EQUAL_SCALAR_OUT: TODO
case H_NOT_EQUAL_SCALAR: set(op.tensor, at::redispatch::not_equal(ks, )); break;
case H_NOT_EQUAL_TENSOR_OUT: TODO
case H_NOT_EQUAL_TENSOR: set(op.tensor, at::redispatch::not_equal(ks, )); break;
case H_NOT_EQUAL__SCALAR: TODO
case H_NOT_EQUAL__TENSOR: TODO
case H_EQ_SCALAR_OUT: TODO
case H_EQ_SCALAR: set(op.tensor, at::redispatch::eq(ks, )); break;
case H_EQ_TENSOR_OUT: TODO
case H_EQ_TENSOR: set(op.tensor, at::redispatch::eq(ks, )); break;
case H_GE_SCALAR_OUT: TODO
case H_GE_SCALAR: set(op.tensor, at::redispatch::ge(ks, )); break;
case H_GE_TENSOR_OUT: TODO
case H_GE_TENSOR: set(op.tensor, at::redispatch::ge(ks, )); break;
case H_GE__SCALAR: TODO
case H_GE__TENSOR: TODO
case H_GREATER_EQUAL_SCALAR_OUT: TODO
case H_GREATER_EQUAL_SCALAR: set(op.tensor, at::redispatch::greater_equal(ks, )); break;
case H_GREATER_EQUAL_TENSOR_OUT: TODO
case H_GREATER_EQUAL_TENSOR: set(op.tensor, at::redispatch::greater_equal(ks, )); break;
case H_GREATER_EQUAL__SCALAR: TODO
case H_GREATER_EQUAL__TENSOR: TODO
case H_LE_SCALAR_OUT: TODO
case H_LE_SCALAR: set(op.tensor, at::redispatch::le(ks, )); break;
case H_LE_TENSOR_OUT: TODO
case H_LE_TENSOR: set(op.tensor, at::redispatch::le(ks, )); break;
case H_LE__SCALAR: TODO
case H_LE__TENSOR: TODO
case H_LESS_EQUAL_SCALAR_OUT: TODO
case H_LESS_EQUAL_SCALAR: set(op.tensor, at::redispatch::less_equal(ks, )); break;
case H_LESS_EQUAL_TENSOR_OUT: TODO
case H_LESS_EQUAL_TENSOR: set(op.tensor, at::redispatch::less_equal(ks, )); break;
case H_LESS_EQUAL__SCALAR: TODO
case H_LESS_EQUAL__TENSOR: TODO
case H_GT_SCALAR_OUT: TODO
case H_GT_SCALAR: set(op.tensor, at::redispatch::gt(ks, )); break;
case H_GT_TENSOR_OUT: TODO
case H_GT_TENSOR: set(op.tensor, at::redispatch::gt(ks, )); break;
case H_GT__SCALAR: TODO
case H_GT__TENSOR: TODO
case H_GREATER_SCALAR_OUT: TODO
case H_GREATER_SCALAR: set(op.tensor, at::redispatch::greater(ks, )); break;
case H_GREATER_TENSOR_OUT: TODO
case H_GREATER_TENSOR: set(op.tensor, at::redispatch::greater(ks, )); break;
case H_GREATER__SCALAR: TODO
case H_GREATER__TENSOR: TODO
case H_LT_SCALAR_OUT: TODO
case H_LT_SCALAR: set(op.tensor, at::redispatch::lt(ks, )); break;
case H_LT_TENSOR_OUT: TODO
case H_LT_TENSOR: set(op.tensor, at::redispatch::lt(ks, )); break;
case H_LT__SCALAR: TODO
case H_LT__TENSOR: TODO
case H_LESS_SCALAR_OUT: TODO
case H_LESS_SCALAR: set(op.tensor, at::redispatch::less(ks, )); break;
case H_LESS_TENSOR_OUT: TODO
case H_LESS_TENSOR: set(op.tensor, at::redispatch::less(ks, )); break;
case H_LESS__SCALAR: TODO
case H_LESS__TENSOR: TODO
case H_TAKE_OUT: TODO
case H_TAKE: set(op.tensor, at::redispatch::take(ks, )); break;
case H_TAKE_ALONG_DIM_OUT: TODO
case H_TAKE_ALONG_DIM: set(op.tensor, at::redispatch::take_along_dim(ks, )); break;
case H_INDEX_SELECT_OUT: TODO
case H_INDEX_SELECT: set(op.tensor, at::redispatch::index_select(ks, )); break;
case H_INDEX_SELECT_DIMNAME_OUT: TODO
case H_INDEX_SELECT_DIMNAME: set(op.tensor, at::redispatch::index_select(ks, )); break;
case H_INDEX_SELECT_BACKWARD: set(op.tensor, at::redispatch::index_select_backward(ks, )); break;
case H_MASKED_SELECT_OUT: TODO
case H_MASKED_SELECT: set(op.tensor, at::redispatch::masked_select(ks, )); break;
case H_MASKED_SELECT_BACKWARD: set(op.tensor, at::redispatch::masked_select_backward(ks, )); break;
case H_NONZERO_OUT: TODO
case H_NONZERO: set(op.tensor, at::redispatch::nonzero(ks, )); break;
case H_NONZERO_NUMPY: set(op.tensor, at::redispatch::nonzero_numpy(ks, )); break;
case H_GATHER_OUT: TODO
case H_GATHER: set(op.tensor, at::redispatch::gather(ks, )); break;
case H_GATHER_BACKWARD: set(op.tensor, at::redispatch::gather_backward(ks, )); break;
case H_GATHER_DIMNAME_OUT: TODO
case H_GATHER_DIMNAME: set(op.tensor, at::redispatch::gather(ks, )); break;
case H__GATHER_SPARSE_BACKWARD: set(op.tensor, at::redispatch::_gather_sparse_backward(ks, )); break;
case H_ADDCMUL_OUT: TODO
case H_ADDCMUL: set(op.tensor, at::redispatch::addcmul(ks, )); break;
case H_ADDCMUL_: TODO
case H_ADDCDIV_OUT: TODO
case H_ADDCDIV: set(op.tensor, at::redispatch::addcdiv(ks, )); break;
case H_CROSS_ENTROPY_LOSS: set(op.tensor, at::redispatch::cross_entropy_loss(ks, )); break;
case H_LSTSQ_X: TODO
case H_LSTSQ: set(op.tensor, at::redispatch::lstsq(ks, )); break;
case H_TRIANGULAR_SOLVE_X: TODO
case H_TRIANGULAR_SOLVE: set(op.tensor, at::redispatch::triangular_solve(ks, )); break;
case H_SYMEIG_E: TODO
case H_SYMEIG: set(op.tensor, at::redispatch::symeig(ks, )); break;
case H__SYMEIG_HELPER: set(op.tensor, at::redispatch::_symeig_helper(ks, )); break;
case H_EIG_E: TODO
case H_EIG: set(op.tensor, at::redispatch::eig(ks, )); break;
case H_SVD_U: TODO
case H_SVD: set(op.tensor, at::redispatch::svd(ks, )); break;
case H__SVD_HELPER: set(op.tensor, at::redispatch::_svd_helper(ks, )); break;
case H_SWAPAXES: set(op.tensor, at::redispatch::swapaxes(ks, )); break;
case H_SWAPAXES_: TODO
case H_SWAPDIMS: set(op.tensor, at::redispatch::swapdims(ks, )); break;
case H_SWAPDIMS_: TODO
case H_CHOLESKY_OUT: TODO
case H_CHOLESKY: set(op.tensor, at::redispatch::cholesky(ks, )); break;
case H__CHOLESKY_HELPER: set(op.tensor, at::redispatch::_cholesky_helper(ks, )); break;
case H_CHOLESKY_SOLVE_OUT: TODO
case H_CHOLESKY_SOLVE: set(op.tensor, at::redispatch::cholesky_solve(ks, )); break;
case H__CHOLESKY_SOLVE_HELPER: set(op.tensor, at::redispatch::_cholesky_solve_helper(ks, )); break;
case H_SOLVE: set(op.tensor, at::redispatch::solve(ks, )); break;
case H_SOLVE_SOLUTION: TODO
case H__SOLVE_HELPER: set(op.tensor, at::redispatch::_solve_helper(ks, )); break;
case H_CHOLESKY_INVERSE: set(op.tensor, at::redispatch::cholesky_inverse(ks, )); break;
case H_CHOLESKY_INVERSE_OUT: TODO
case H_QR_Q: TODO
case H_QR: set(op.tensor, at::redispatch::qr(ks, )); break;
case H_GEQRF_A: TODO
case H_GEQRF: set(op.tensor, at::redispatch::geqrf(ks, )); break;
case H_ORGQR: set(op.tensor, at::redispatch::orgqr(ks, )); break;
case H_ORGQR_OUT: TODO
case H_ORMQR_OUT: TODO
case H_ORMQR: set(op.tensor, at::redispatch::ormqr(ks, )); break;
case H__LU_WITH_INFO: set(op.tensor, at::redispatch::_lu_with_info(ks, )); break;
case H_LU_SOLVE_OUT: TODO
case H_LU_SOLVE: set(op.tensor, at::redispatch::lu_solve(ks, )); break;
case H__LU_SOLVE_HELPER: set(op.tensor, at::redispatch::_lu_solve_helper(ks, )); break;
case H_MULTINOMIAL_OUT: TODO
case H_MULTINOMIAL: set(op.tensor, at::redispatch::multinomial(ks, )); break;
case H_LGAMMA_OUT: TODO
case H_DIGAMMA_OUT: TODO
case H_POLYGAMMA_OUT: TODO
case H_POLYGAMMA: set(op.tensor, at::redispatch::polygamma(ks, )); break;
case H_POLYGAMMA_: TODO
case H_ERFINV_OUT: TODO
case H_I0: set(op.tensor, at::redispatch::i0(ks, )); break;
case H_I0_: TODO
case H_I0_OUT: TODO
case H_SIGN: set(op.tensor, at::redispatch::sign(ks, )); break;
case H_SIGN_: TODO
case H_SIGN_OUT: TODO
case H_SIGNBIT: set(op.tensor, at::redispatch::signbit(ks, )); break;
case H_SIGNBIT_OUT: TODO
case H_DIST: set(op.tensor, at::redispatch::dist(ks, )); break;
case H_ATAN2_OUT: TODO
case H_LERP_SCALAR_OUT: TODO
case H_LERP_TENSOR_OUT: TODO
case H_LERP_SCALAR: set(op.tensor, at::redispatch::lerp(ks, )); break;
case H_LERP_TENSOR: set(op.tensor, at::redispatch::lerp(ks, )); break;
case H_HISTC_OUT: TODO
case H_HISTC: set(op.tensor, at::redispatch::histc(ks, )); break;
case H_FMOD_SCALAR_OUT: TODO
case H_FMOD_SCALAR: set(op.tensor, at::redispatch::fmod(ks, )); break;
case H_FMOD_TENSOR_OUT: TODO
case H_FMOD_TENSOR: set(op.tensor, at::redispatch::fmod(ks, )); break;
case H_HYPOT_OUT: TODO
case H_HYPOT: set(op.tensor, at::redispatch::hypot(ks, )); break;
case H_HYPOT_: TODO
case H_IGAMMA_OUT: TODO
case H_IGAMMA: set(op.tensor, at::redispatch::igamma(ks, )); break;
case H_IGAMMA_: TODO
case H_IGAMMAC_OUT: TODO
case H_IGAMMAC: set(op.tensor, at::redispatch::igammac(ks, )); break;
case H_IGAMMAC_: TODO
case H_NEXTAFTER_OUT: TODO
case H_NEXTAFTER: set(op.tensor, at::redispatch::nextafter(ks, )); break;
case H_NEXTAFTER_: TODO
case H_REMAINDER_SCALAR_OUT: TODO
case H_REMAINDER_SCALAR: set(op.tensor, at::redispatch::remainder(ks, )); break;
case H_REMAINDER_TENSOR_OUT: TODO
case H_REMAINDER_TENSOR: set(op.tensor, at::redispatch::remainder(ks, )); break;
case H_MIN: set(op.tensor, at::redispatch::min(ks, )); break;
case H_FMIN: set(op.tensor, at::redispatch::fmin(ks, )); break;
case H_FMIN_OUT: TODO
case H_MAX: set(op.tensor, at::redispatch::max(ks, )); break;
case H_FMAX: set(op.tensor, at::redispatch::fmax(ks, )); break;
case H_FMAX_OUT: TODO
case H_MAXIMUM: set(op.tensor, at::redispatch::maximum(ks, )); break;
case H_MAXIMUM_OUT: TODO
case H_MAX_OTHER: set(op.tensor, at::redispatch::max(ks, )); break;
case H_MAX_OUT: TODO
case H_MINIMUM: set(op.tensor, at::redispatch::minimum(ks, )); break;
case H_MINIMUM_OUT: TODO
case H_MIN_OUT: TODO
case H_MIN_OTHER: set(op.tensor, at::redispatch::min(ks, )); break;
case H_QUANTILE_SCALAR_OUT: TODO
case H_QUANTILE_SCALAR: set(op.tensor, at::redispatch::quantile(ks, )); break;
case H_QUANTILE_OUT: TODO
case H_QUANTILE: set(op.tensor, at::redispatch::quantile(ks, )); break;
case H_NANQUANTILE_SCALAR_OUT: TODO
case H_NANQUANTILE_SCALAR: set(op.tensor, at::redispatch::nanquantile(ks, )); break;
case H_NANQUANTILE_OUT: TODO
case H_NANQUANTILE: set(op.tensor, at::redispatch::nanquantile(ks, )); break;
case H_QUANTILE_NEW_SCALAR_OUT: TODO
case H_QUANTILE_NEW_SCALAR: set(op.tensor, at::redispatch::quantile(ks, )); break;
case H_QUANTILE_NEW_OUT: TODO
case H_QUANTILE_NEW: set(op.tensor, at::redispatch::quantile(ks, )); break;
case H_NANQUANTILE_NEW_SCALAR_OUT: TODO
case H_NANQUANTILE_NEW_SCALAR: set(op.tensor, at::redispatch::nanquantile(ks, )); break;
case H_NANQUANTILE_NEW_OUT: TODO
case H_NANQUANTILE_NEW: set(op.tensor, at::redispatch::nanquantile(ks, )); break;
case H_SORT_VALUES: TODO
case H_SORT_VALUES_STABLE: TODO
case H_SORT: set(op.tensor, at::redispatch::sort(ks, )); break;
case H_SORT_STABLE: set(op.tensor, at::redispatch::sort(ks, )); break;
case H_SORT_DIMNAME_VALUES: TODO
case H_SORT_DIMNAME_VALUES_STABLE: TODO
case H_SORT_DIMNAME: set(op.tensor, at::redispatch::sort(ks, )); break;
case H_SORT_DIMNAME_STABLE: set(op.tensor, at::redispatch::sort(ks, )); break;
case H_MSORT_OUT: TODO
case H_MSORT: set(op.tensor, at::redispatch::msort(ks, )); break;
case H_ARGSORT: set(op.tensor, at::redispatch::argsort(ks, )); break;
case H_ARGSORT_DIMNAME: set(op.tensor, at::redispatch::argsort(ks, )); break;
case H_TOPK_VALUES: TODO
case H_TOPK: set(op.tensor, at::redispatch::topk(ks, )); break;
case H_ALL: set(op.tensor, at::redispatch::all(ks, )); break;
case H_ANY: set(op.tensor, at::redispatch::any(ks, )); break;
case H_RENORM_OUT: TODO
case H_RENORM: set(op.tensor, at::redispatch::renorm(ks, )); break;
case H_UNFOLD: set(op.tensor, at::redispatch::unfold(ks, )); break;
case H_UNFOLD_BACKWARD: set(op.tensor, at::redispatch::unfold_backward(ks, )); break;
case H_EQUAL: set(op.tensor, at::redispatch::equal(ks, )); break;
case H_POW_TENSOR_TENSOR_OUT: TODO
case H_POW_SCALAR_OUT: TODO
case H_POW_TENSOR_SCALAR_OUT: TODO
case H_POW_TENSOR_SCALAR: set(op.tensor, at::redispatch::pow(ks, )); break;
case H_FLOAT_POWER_TENSOR_TENSOR_OUT: TODO
case H_FLOAT_POWER_TENSOR_TENSOR: set(op.tensor, at::redispatch::float_power(ks, )); break;
case H_FLOAT_POWER_SCALAR_OUT: TODO
case H_FLOAT_POWER_SCALAR: set(op.tensor, at::redispatch::float_power(ks, )); break;
case H_FLOAT_POWER_TENSOR_SCALAR_OUT: TODO
case H_FLOAT_POWER_TENSOR_SCALAR: set(op.tensor, at::redispatch::float_power(ks, )); break;
case H_FLOAT_POWER__SCALAR: TODO
case H_FLOAT_POWER__TENSOR: TODO
case H_NORMAL_: TODO
case H_NORMAL_TENSOR_FLOAT_OUT: TODO
case H_NORMAL_TENSOR_FLOAT: set(op.tensor, at::redispatch::normal(ks, )); break;
case H_NORMAL_FLOAT_TENSOR_OUT: TODO
case H_NORMAL_FLOAT_TENSOR: set(op.tensor, at::redispatch::normal(ks, )); break;
case H_NORMAL_TENSOR_TENSOR_OUT: TODO
case H_NORMAL_TENSOR_TENSOR: set(op.tensor, at::redispatch::normal(ks, )); break;
case H_NORMAL_FLOAT_FLOAT: set(op.tensor, at::redispatch::normal(ks, )); break;
case H_NORMAL_FLOAT_FLOAT_OUT: TODO
case H_ALIAS: set(op.tensor, at::redispatch::alias(ks, )); break;
case H__INDEX_COPY_: TODO
case H__CUMSUM: set(op.tensor, at::redispatch::_cumsum(ks, )); break;
case H__CUMSUM_OUT: TODO
case H__CUMPROD: set(op.tensor, at::redispatch::_cumprod(ks, )); break;
case H__CUMPROD_OUT: TODO
case H__VAR: set(op.tensor, at::redispatch::_var(ks, )); break;
case H__STD: set(op.tensor, at::redispatch::_std(ks, )); break;
case H__AMP_FOREACH_NON_FINITE_CHECK_AND_UNSCALE_: set(op.tensor, at::redispatch::_amp_foreach_non_finite_check_and_unscale_(ks, )); break;
case H__AMP_UPDATE_SCALE: set(op.tensor, at::redispatch::_amp_update_scale(ks, )); break;
case H__CAT: set(op.tensor, at::redispatch::_cat(ks, )); break;
case H__CAT_OUT: TODO
case H__FOREACH_ADD_SCALAR: set(op.tensor, at::redispatch::_foreach_add(ks, )); break;
case H__FOREACH_ADD__SCALAR: set(op.tensor, at::redispatch::_foreach_add_(ks, )); break;
case H__FOREACH_SUB_SCALAR: set(op.tensor, at::redispatch::_foreach_sub(ks, )); break;
case H__FOREACH_SUB__SCALAR: set(op.tensor, at::redispatch::_foreach_sub_(ks, )); break;
case H__FOREACH_MUL_SCALAR: set(op.tensor, at::redispatch::_foreach_mul(ks, )); break;
case H__FOREACH_MUL__SCALAR: set(op.tensor, at::redispatch::_foreach_mul_(ks, )); break;
case H__FOREACH_DIV_SCALAR: set(op.tensor, at::redispatch::_foreach_div(ks, )); break;
case H__FOREACH_DIV__SCALAR: set(op.tensor, at::redispatch::_foreach_div_(ks, )); break;
case H__FOREACH_ADD_LIST: set(op.tensor, at::redispatch::_foreach_add(ks, )); break;
case H__FOREACH_ADD__LIST: set(op.tensor, at::redispatch::_foreach_add_(ks, )); break;
case H__FOREACH_SUB_LIST: set(op.tensor, at::redispatch::_foreach_sub(ks, )); break;
case H__FOREACH_SUB__LIST: set(op.tensor, at::redispatch::_foreach_sub_(ks, )); break;
case H__FOREACH_MUL_LIST: set(op.tensor, at::redispatch::_foreach_mul(ks, )); break;
case H__FOREACH_MUL__LIST: set(op.tensor, at::redispatch::_foreach_mul_(ks, )); break;
case H__FOREACH_DIV_LIST: set(op.tensor, at::redispatch::_foreach_div(ks, )); break;
case H__FOREACH_DIV__LIST: set(op.tensor, at::redispatch::_foreach_div_(ks, )); break;
case H__FOREACH_ADD_SCALARLIST: set(op.tensor, at::redispatch::_foreach_add(ks, )); break;
case H__FOREACH_ADD__SCALARLIST: set(op.tensor, at::redispatch::_foreach_add_(ks, )); break;
case H__FOREACH_SUB_SCALARLIST: set(op.tensor, at::redispatch::_foreach_sub(ks, )); break;
case H__FOREACH_SUB__SCALARLIST: set(op.tensor, at::redispatch::_foreach_sub_(ks, )); break;
case H__FOREACH_DIV_SCALARLIST: set(op.tensor, at::redispatch::_foreach_div(ks, )); break;
case H__FOREACH_DIV__SCALARLIST: set(op.tensor, at::redispatch::_foreach_div_(ks, )); break;
case H__FOREACH_MUL_SCALARLIST: set(op.tensor, at::redispatch::_foreach_mul(ks, )); break;
case H__FOREACH_MUL__SCALARLIST: set(op.tensor, at::redispatch::_foreach_mul_(ks, )); break;
case H__FOREACH_EXP: set(op.tensor, at::redispatch::_foreach_exp(ks, )); break;
case H__FOREACH_ZERO_: set(op.tensor, at::redispatch::_foreach_zero_(ks, )); break;
case H__FOREACH_EXP_: set(op.tensor, at::redispatch::_foreach_exp_(ks, )); break;
case H__FOREACH_SQRT: set(op.tensor, at::redispatch::_foreach_sqrt(ks, )); break;
case H__FOREACH_SQRT_: set(op.tensor, at::redispatch::_foreach_sqrt_(ks, )); break;
case H__FOREACH_ABS: set(op.tensor, at::redispatch::_foreach_abs(ks, )); break;
case H__FOREACH_ABS_: set(op.tensor, at::redispatch::_foreach_abs_(ks, )); break;
case H__FOREACH_ACOS: set(op.tensor, at::redispatch::_foreach_acos(ks, )); break;
case H__FOREACH_ACOS_: set(op.tensor, at::redispatch::_foreach_acos_(ks, )); break;
case H__FOREACH_ASIN: set(op.tensor, at::redispatch::_foreach_asin(ks, )); break;
case H__FOREACH_ASIN_: set(op.tensor, at::redispatch::_foreach_asin_(ks, )); break;
case H__FOREACH_ATAN: set(op.tensor, at::redispatch::_foreach_atan(ks, )); break;
case H__FOREACH_ATAN_: set(op.tensor, at::redispatch::_foreach_atan_(ks, )); break;
case H__FOREACH_CEIL: set(op.tensor, at::redispatch::_foreach_ceil(ks, )); break;
case H__FOREACH_CEIL_: set(op.tensor, at::redispatch::_foreach_ceil_(ks, )); break;
case H__FOREACH_COS: set(op.tensor, at::redispatch::_foreach_cos(ks, )); break;
case H__FOREACH_COS_: set(op.tensor, at::redispatch::_foreach_cos_(ks, )); break;
case H__FOREACH_COSH: set(op.tensor, at::redispatch::_foreach_cosh(ks, )); break;
case H__FOREACH_COSH_: set(op.tensor, at::redispatch::_foreach_cosh_(ks, )); break;
case H__FOREACH_ERF: set(op.tensor, at::redispatch::_foreach_erf(ks, )); break;
case H__FOREACH_ERF_: set(op.tensor, at::redispatch::_foreach_erf_(ks, )); break;
case H__FOREACH_ERFC: set(op.tensor, at::redispatch::_foreach_erfc(ks, )); break;
case H__FOREACH_ERFC_: set(op.tensor, at::redispatch::_foreach_erfc_(ks, )); break;
case H__FOREACH_EXPM1: set(op.tensor, at::redispatch::_foreach_expm1(ks, )); break;
case H__FOREACH_EXPM1_: set(op.tensor, at::redispatch::_foreach_expm1_(ks, )); break;
case H__FOREACH_FLOOR: set(op.tensor, at::redispatch::_foreach_floor(ks, )); break;
case H__FOREACH_FLOOR_: set(op.tensor, at::redispatch::_foreach_floor_(ks, )); break;
case H__FOREACH_LOG: set(op.tensor, at::redispatch::_foreach_log(ks, )); break;
case H__FOREACH_LOG_: set(op.tensor, at::redispatch::_foreach_log_(ks, )); break;
case H__FOREACH_LOG10: set(op.tensor, at::redispatch::_foreach_log10(ks, )); break;
case H__FOREACH_LOG10_: set(op.tensor, at::redispatch::_foreach_log10_(ks, )); break;
case H__FOREACH_LOG1P: set(op.tensor, at::redispatch::_foreach_log1p(ks, )); break;
case H__FOREACH_LOG1P_: set(op.tensor, at::redispatch::_foreach_log1p_(ks, )); break;
case H__FOREACH_LOG2: set(op.tensor, at::redispatch::_foreach_log2(ks, )); break;
case H__FOREACH_LOG2_: set(op.tensor, at::redispatch::_foreach_log2_(ks, )); break;
case H__FOREACH_NEG: set(op.tensor, at::redispatch::_foreach_neg(ks, )); break;
case H__FOREACH_NEG_: set(op.tensor, at::redispatch::_foreach_neg_(ks, )); break;
case H__FOREACH_TAN: set(op.tensor, at::redispatch::_foreach_tan(ks, )); break;
case H__FOREACH_TAN_: set(op.tensor, at::redispatch::_foreach_tan_(ks, )); break;
case H__FOREACH_TANH: set(op.tensor, at::redispatch::_foreach_tanh(ks, )); break;
case H__FOREACH_TANH_: set(op.tensor, at::redispatch::_foreach_tanh_(ks, )); break;
case H__FOREACH_SIN: set(op.tensor, at::redispatch::_foreach_sin(ks, )); break;
case H__FOREACH_SIN_: set(op.tensor, at::redispatch::_foreach_sin_(ks, )); break;
case H__FOREACH_SINH: set(op.tensor, at::redispatch::_foreach_sinh(ks, )); break;
case H__FOREACH_SINH_: set(op.tensor, at::redispatch::_foreach_sinh_(ks, )); break;
case H__FOREACH_ROUND: set(op.tensor, at::redispatch::_foreach_round(ks, )); break;
case H__FOREACH_ROUND_: set(op.tensor, at::redispatch::_foreach_round_(ks, )); break;
case H__FOREACH_LGAMMA: set(op.tensor, at::redispatch::_foreach_lgamma(ks, )); break;
case H__FOREACH_LGAMMA_: set(op.tensor, at::redispatch::_foreach_lgamma_(ks, )); break;
case H__FOREACH_FRAC: set(op.tensor, at::redispatch::_foreach_frac(ks, )); break;
case H__FOREACH_FRAC_: set(op.tensor, at::redispatch::_foreach_frac_(ks, )); break;
case H__FOREACH_RECIPROCAL: set(op.tensor, at::redispatch::_foreach_reciprocal(ks, )); break;
case H__FOREACH_RECIPROCAL_: set(op.tensor, at::redispatch::_foreach_reciprocal_(ks, )); break;
case H__FOREACH_SIGMOID: set(op.tensor, at::redispatch::_foreach_sigmoid(ks, )); break;
case H__FOREACH_SIGMOID_: set(op.tensor, at::redispatch::_foreach_sigmoid_(ks, )); break;
case H__FOREACH_TRUNC: set(op.tensor, at::redispatch::_foreach_trunc(ks, )); break;
case H__FOREACH_TRUNC_: set(op.tensor, at::redispatch::_foreach_trunc_(ks, )); break;
case H__FOREACH_ADDCDIV__SCALAR: set(op.tensor, at::redispatch::_foreach_addcdiv_(ks, )); break;
case H__FOREACH_ADDCMUL__SCALAR: set(op.tensor, at::redispatch::_foreach_addcmul_(ks, )); break;
case H__FOREACH_ADDCDIV__SCALARLIST: set(op.tensor, at::redispatch::_foreach_addcdiv_(ks, )); break;
case H__FOREACH_ADDCMUL__SCALARLIST: set(op.tensor, at::redispatch::_foreach_addcmul_(ks, )); break;
case H__FOREACH_ADDCDIV_SCALAR: set(op.tensor, at::redispatch::_foreach_addcdiv(ks, )); break;
case H__FOREACH_ADDCMUL_SCALAR: set(op.tensor, at::redispatch::_foreach_addcmul(ks, )); break;
case H__FOREACH_ADDCDIV_SCALARLIST: set(op.tensor, at::redispatch::_foreach_addcdiv(ks, )); break;
case H__FOREACH_ADDCMUL_SCALARLIST: set(op.tensor, at::redispatch::_foreach_addcmul(ks, )); break;
case H__FOREACH_MAXIMUM_LIST: set(op.tensor, at::redispatch::_foreach_maximum(ks, )); break;
case H__FOREACH_MINIMUM_LIST: set(op.tensor, at::redispatch::_foreach_minimum(ks, )); break;
case H_BUCKETIZE_TENSOR: set(op.tensor, at::redispatch::bucketize(ks, )); break;
case H_BUCKETIZE_TENSOR_OUT: TODO
case H_BUCKETIZE_SCALAR: set(op.tensor, at::redispatch::bucketize(ks, )); break;
case H_SEARCHSORTED_TENSOR: set(op.tensor, at::redispatch::searchsorted(ks, )); break;
case H_SEARCHSORTED_TENSOR_OUT: TODO
case H_SEARCHSORTED_SCALAR: set(op.tensor, at::redispatch::searchsorted(ks, )); break;
case H_MSE_LOSS_OUT: TODO
case H_MSE_LOSS: set(op.tensor, at::redispatch::mse_loss(ks, )); break;
case H_MSE_LOSS_BACKWARD_GRAD_INPUT: TODO
case H_MSE_LOSS_BACKWARD: set(op.tensor, at::redispatch::mse_loss_backward(ks, )); break;
case H_L1_LOSS_OUT: TODO
case H_L1_LOSS: set(op.tensor, at::redispatch::l1_loss(ks, )); break;
case H_L1_LOSS_BACKWARD_GRAD_INPUT: TODO
case H_L1_LOSS_BACKWARD: set(op.tensor, at::redispatch::l1_loss_backward(ks, )); break;
case H_MULTI_MARGIN_LOSS_OUT: TODO
case H_MULTI_MARGIN_LOSS: set(op.tensor, at::redispatch::multi_margin_loss(ks, )); break;
case H_MULTI_MARGIN_LOSS_BACKWARD_GRAD_INPUT: TODO
case H_MULTI_MARGIN_LOSS_BACKWARD: set(op.tensor, at::redispatch::multi_margin_loss_backward(ks, )); break;
case H_MULTILABEL_MARGIN_LOSS_OUT: TODO
case H_MULTILABEL_MARGIN_LOSS: set(op.tensor, at::redispatch::multilabel_margin_loss(ks, )); break;
case H_MULTILABEL_MARGIN_LOSS_FORWARD_OUTPUT: TODO
case H_MULTILABEL_MARGIN_LOSS_FORWARD: set(op.tensor, at::redispatch::multilabel_margin_loss_forward(ks, )); break;
case H_MULTILABEL_MARGIN_LOSS_BACKWARD_GRAD_INPUT: TODO
case H_MULTILABEL_MARGIN_LOSS_BACKWARD: set(op.tensor, at::redispatch::multilabel_margin_loss_backward(ks, )); break;
case H_NLL_LOSS_OUT: TODO
case H_NLL_LOSS_ND: set(op.tensor, at::redispatch::nll_loss_nd(ks, )); break;
case H_NLL_LOSS: set(op.tensor, at::redispatch::nll_loss(ks, )); break;
case H_NLL_LOSS_FORWARD_OUTPUT: TODO
case H_NLL_LOSS_FORWARD: set(op.tensor, at::redispatch::nll_loss_forward(ks, )); break;
case H_NLL_LOSS_BACKWARD_GRAD_INPUT: TODO
case H_NLL_LOSS_BACKWARD: set(op.tensor, at::redispatch::nll_loss_backward(ks, )); break;
case H_NLL_LOSS2D_OUT: TODO
case H_NLL_LOSS2D: set(op.tensor, at::redispatch::nll_loss2d(ks, )); break;
case H_NLL_LOSS2D_FORWARD_OUTPUT: TODO
case H_NLL_LOSS2D_FORWARD: set(op.tensor, at::redispatch::nll_loss2d_forward(ks, )); break;
case H_NLL_LOSS2D_BACKWARD_GRAD_INPUT: TODO
case H_NLL_LOSS2D_BACKWARD: set(op.tensor, at::redispatch::nll_loss2d_backward(ks, )); break;
case H_SMOOTH_L1_LOSS_OUT: TODO
case H_SMOOTH_L1_LOSS: set(op.tensor, at::redispatch::smooth_l1_loss(ks, )); break;
case H_SMOOTH_L1_LOSS_BACKWARD_GRAD_INPUT: TODO
case H_SMOOTH_L1_LOSS_BACKWARD: set(op.tensor, at::redispatch::smooth_l1_loss_backward(ks, )); break;
case H_HUBER_LOSS_OUT: TODO
case H_HUBER_LOSS: set(op.tensor, at::redispatch::huber_loss(ks, )); break;
case H_HUBER_LOSS_BACKWARD_OUT: TODO
case H_HUBER_LOSS_BACKWARD: set(op.tensor, at::redispatch::huber_loss_backward(ks, )); break;
case H_SOFT_MARGIN_LOSS_OUT: TODO
case H_SOFT_MARGIN_LOSS: set(op.tensor, at::redispatch::soft_margin_loss(ks, )); break;
case H_SOFT_MARGIN_LOSS_BACKWARD_GRAD_INPUT: TODO
case H_SOFT_MARGIN_LOSS_BACKWARD: set(op.tensor, at::redispatch::soft_margin_loss_backward(ks, )); break;
case H_ELU_OUT: TODO
case H_ELU: set(op.tensor, at::redispatch::elu(ks, )); break;
case H_ELU_BACKWARD: set(op.tensor, at::redispatch::elu_backward(ks, )); break;
case H_ELU_: TODO
case H_GLU_OUT: TODO
case H_GLU: set(op.tensor, at::redispatch::glu(ks, )); break;
case H_GLU_BACKWARD_GRAD_INPUT: TODO
case H_GLU_BACKWARD: set(op.tensor, at::redispatch::glu_backward(ks, )); break;
case H_HARDSIGMOID_OUT: TODO
case H_HARDSIGMOID: set(op.tensor, at::redispatch::hardsigmoid(ks, )); break;
case H_HARDSIGMOID_: TODO
case H_HARDSIGMOID_BACKWARD: set(op.tensor, at::redispatch::hardsigmoid_backward(ks, )); break;
case H_HARDTANH_OUT: TODO
case H_HARDTANH: set(op.tensor, at::redispatch::hardtanh(ks, )); break;
case H_HARDTANH_BACKWARD_GRAD_INPUT: TODO
case H_HARDTANH_BACKWARD: set(op.tensor, at::redispatch::hardtanh_backward(ks, )); break;
case H_HARDTANH_: TODO
case H_HARDSWISH_OUT: TODO
case H_HARDSWISH: set(op.tensor, at::redispatch::hardswish(ks, )); break;
case H_HARDSWISH_: TODO
case H_HARDSWISH_BACKWARD: set(op.tensor, at::redispatch::hardswish_backward(ks, )); break;
case H_LEAKY_RELU_OUT: TODO
case H_LEAKY_RELU: set(op.tensor, at::redispatch::leaky_relu(ks, )); break;
case H_LEAKY_RELU_BACKWARD: set(op.tensor, at::redispatch::leaky_relu_backward(ks, )); break;
case H_LEAKY_RELU_: TODO
case H_LOG_SIGMOID_OUT: TODO
case H_LOG_SIGMOID: set(op.tensor, at::redispatch::log_sigmoid(ks, )); break;
case H_LOG_SIGMOID_FORWARD_OUTPUT: TODO
case H_LOG_SIGMOID_FORWARD: set(op.tensor, at::redispatch::log_sigmoid_forward(ks, )); break;
case H_LOG_SIGMOID_BACKWARD_GRAD_INPUT: TODO
case H_LOG_SIGMOID_BACKWARD: set(op.tensor, at::redispatch::log_sigmoid_backward(ks, )); break;
case H_RRELU_WITH_NOISE_OUT: TODO
case H_RRELU_WITH_NOISE: set(op.tensor, at::redispatch::rrelu_with_noise(ks, )); break;
case H_RRELU_WITH_NOISE_BACKWARD: set(op.tensor, at::redispatch::rrelu_with_noise_backward(ks, )); break;
case H_RRELU_WITH_NOISE_: TODO
case H_SOFTPLUS_OUT: TODO
case H_SOFTPLUS: set(op.tensor, at::redispatch::softplus(ks, )); break;
case H_SOFTPLUS_BACKWARD_GRAD_INPUT: TODO
case H_SOFTPLUS_BACKWARD: set(op.tensor, at::redispatch::softplus_backward(ks, )); break;
case H_SOFTSHRINK_OUT: TODO
case H_SOFTSHRINK: set(op.tensor, at::redispatch::softshrink(ks, )); break;
case H_SOFTSHRINK_BACKWARD_GRAD_INPUT: TODO
case H_SOFTSHRINK_BACKWARD: set(op.tensor, at::redispatch::softshrink_backward(ks, )); break;
case H_ADAPTIVE_AVG_POOL2D_OUT: TODO
case H_ADAPTIVE_AVG_POOL2D: set(op.tensor, at::redispatch::adaptive_avg_pool2d(ks, )); break;
case H_MKLDNN_ADAPTIVE_AVG_POOL2D: set(op.tensor, at::redispatch::mkldnn_adaptive_avg_pool2d(ks, )); break;
case H_MKLDNN_ADAPTIVE_AVG_POOL2D_BACKWARD: set(op.tensor, at::redispatch::mkldnn_adaptive_avg_pool2d_backward(ks, )); break;
case H__ADAPTIVE_AVG_POOL2D: set(op.tensor, at::redispatch::_adaptive_avg_pool2d(ks, )); break;
case H__ADAPTIVE_AVG_POOL2D_BACKWARD: set(op.tensor, at::redispatch::_adaptive_avg_pool2d_backward(ks, )); break;
case H_ADAPTIVE_AVG_POOL3D_OUT: TODO
case H_ADAPTIVE_AVG_POOL3D: set(op.tensor, at::redispatch::adaptive_avg_pool3d(ks, )); break;
case H__ADAPTIVE_AVG_POOL3D: set(op.tensor, at::redispatch::_adaptive_avg_pool3d(ks, )); break;
case H_ADAPTIVE_AVG_POOL3D_BACKWARD_GRAD_INPUT: TODO
case H__ADAPTIVE_AVG_POOL3D_BACKWARD: set(op.tensor, at::redispatch::_adaptive_avg_pool3d_backward(ks, )); break;
case H_ADAPTIVE_MAX_POOL2D_OUT: TODO
case H_ADAPTIVE_MAX_POOL2D_BACKWARD_GRAD_INPUT: TODO
case H_ADAPTIVE_MAX_POOL2D_BACKWARD: set(op.tensor, at::redispatch::adaptive_max_pool2d_backward(ks, )); break;
case H_ADAPTIVE_MAX_POOL3D_OUT: TODO
case H_ADAPTIVE_MAX_POOL3D_BACKWARD_GRAD_INPUT: TODO
case H_ADAPTIVE_MAX_POOL3D_BACKWARD: set(op.tensor, at::redispatch::adaptive_max_pool3d_backward(ks, )); break;
case H_AVG_POOL2D_OUT: TODO
case H_AVG_POOL2D: set(op.tensor, at::redispatch::avg_pool2d(ks, )); break;
case H_AVG_POOL2D_BACKWARD_GRAD_INPUT: TODO
case H_AVG_POOL2D_BACKWARD: set(op.tensor, at::redispatch::avg_pool2d_backward(ks, )); break;
case H_AVG_POOL3D_OUT: TODO
case H_AVG_POOL3D: set(op.tensor, at::redispatch::avg_pool3d(ks, )); break;
case H_AVG_POOL3D_BACKWARD_GRAD_INPUT: TODO
case H_AVG_POOL3D_BACKWARD: set(op.tensor, at::redispatch::avg_pool3d_backward(ks, )); break;
case H_FRACTIONAL_MAX_POOL2D_OUTPUT: TODO
case H_FRACTIONAL_MAX_POOL2D_BACKWARD_GRAD_INPUT: TODO
case H_FRACTIONAL_MAX_POOL2D_BACKWARD: set(op.tensor, at::redispatch::fractional_max_pool2d_backward(ks, )); break;
case H_FRACTIONAL_MAX_POOL3D_OUTPUT: TODO
case H_FRACTIONAL_MAX_POOL3D: set(op.tensor, at::redispatch::fractional_max_pool3d(ks, )); break;
case H_FRACTIONAL_MAX_POOL3D_BACKWARD_GRAD_INPUT: TODO
case H_FRACTIONAL_MAX_POOL3D_BACKWARD: set(op.tensor, at::redispatch::fractional_max_pool3d_backward(ks, )); break;
case H_MAX_POOL2D_WITH_INDICES_OUT: TODO
case H_MAX_POOL2D_WITH_INDICES: set(op.tensor, at::redispatch::max_pool2d_with_indices(ks, )); break;
case H_MAX_POOL2D_WITH_INDICES_BACKWARD_GRAD_INPUT: TODO
case H_MAX_POOL2D_WITH_INDICES_BACKWARD: set(op.tensor, at::redispatch::max_pool2d_with_indices_backward(ks, )); break;
case H_MAX_POOL3D_WITH_INDICES_OUT: TODO
case H_MAX_POOL3D_WITH_INDICES: set(op.tensor, at::redispatch::max_pool3d_with_indices(ks, )); break;
case H_MAX_POOL3D_WITH_INDICES_BACKWARD_GRAD_INPUT: TODO
case H_MAX_POOL3D_WITH_INDICES_BACKWARD: set(op.tensor, at::redispatch::max_pool3d_with_indices_backward(ks, )); break;
case H_MAX_UNPOOL2D_OUT: TODO
case H_MAX_UNPOOL2D: set(op.tensor, at::redispatch::max_unpool2d(ks, )); break;
case H_MAX_UNPOOL2D_BACKWARD_GRAD_INPUT: TODO
case H_MAX_UNPOOL2D_BACKWARD: set(op.tensor, at::redispatch::max_unpool2d_backward(ks, )); break;
case H_MAX_UNPOOL3D_OUT: TODO
case H_MAX_UNPOOL3D: set(op.tensor, at::redispatch::max_unpool3d(ks, )); break;
case H_MAX_UNPOOL3D_BACKWARD_GRAD_INPUT: TODO
case H_MAX_UNPOOL3D_BACKWARD: set(op.tensor, at::redispatch::max_unpool3d_backward(ks, )); break;
case H_REFLECTION_PAD1D_OUT: TODO
case H_REFLECTION_PAD1D: set(op.tensor, at::redispatch::reflection_pad1d(ks, )); break;
case H_REFLECTION_PAD1D_BACKWARD_GRAD_INPUT: TODO
case H_REFLECTION_PAD1D_BACKWARD: set(op.tensor, at::redispatch::reflection_pad1d_backward(ks, )); break;
case H_REFLECTION_PAD2D_OUT: TODO
case H_REFLECTION_PAD2D: set(op.tensor, at::redispatch::reflection_pad2d(ks, )); break;
case H_REFLECTION_PAD2D_BACKWARD_GRAD_INPUT: TODO
case H_REFLECTION_PAD2D_BACKWARD: set(op.tensor, at::redispatch::reflection_pad2d_backward(ks, )); break;
case H_REPLICATION_PAD1D_OUT: TODO
case H_REPLICATION_PAD1D_BACKWARD_GRAD_INPUT: TODO
case H_REPLICATION_PAD2D_OUT: TODO
case H_REPLICATION_PAD2D_BACKWARD_GRAD_INPUT: TODO
case H_REPLICATION_PAD2D_BACKWARD: set(op.tensor, at::redispatch::replication_pad2d_backward(ks, )); break;
case H_REPLICATION_PAD3D_OUT: TODO
case H_REPLICATION_PAD3D_BACKWARD_GRAD_INPUT: TODO
case H_REPLICATION_PAD3D_BACKWARD: set(op.tensor, at::redispatch::replication_pad3d_backward(ks, )); break;
case H_UPSAMPLE_LINEAR1D_VEC: set(op.tensor, at::redispatch::upsample_linear1d(ks, )); break;
case H_UPSAMPLE_LINEAR1D_BACKWARD_VEC: set(op.tensor, at::redispatch::upsample_linear1d_backward(ks, )); break;
case H_UPSAMPLE_BILINEAR2D_VEC: set(op.tensor, at::redispatch::upsample_bilinear2d(ks, )); break;
case H_UPSAMPLE_BILINEAR2D_BACKWARD_VEC: set(op.tensor, at::redispatch::upsample_bilinear2d_backward(ks, )); break;
case H_UPSAMPLE_TRILINEAR3D_VEC: set(op.tensor, at::redispatch::upsample_trilinear3d(ks, )); break;
case H_UPSAMPLE_TRILINEAR3D_BACKWARD_VEC: set(op.tensor, at::redispatch::upsample_trilinear3d_backward(ks, )); break;
case H_UPSAMPLE_BICUBIC2D_VEC: set(op.tensor, at::redispatch::upsample_bicubic2d(ks, )); break;
case H_UPSAMPLE_BICUBIC2D_BACKWARD_VEC: set(op.tensor, at::redispatch::upsample_bicubic2d_backward(ks, )); break;
case H_UPSAMPLE_NEAREST1D_VEC: set(op.tensor, at::redispatch::upsample_nearest1d(ks, )); break;
case H_UPSAMPLE_NEAREST1D_BACKWARD_VEC: set(op.tensor, at::redispatch::upsample_nearest1d_backward(ks, )); break;
case H_UPSAMPLE_NEAREST2D_VEC: set(op.tensor, at::redispatch::upsample_nearest2d(ks, )); break;
case H_UPSAMPLE_NEAREST2D_BACKWARD_VEC: set(op.tensor, at::redispatch::upsample_nearest2d_backward(ks, )); break;
case H_UPSAMPLE_NEAREST3D_VEC: set(op.tensor, at::redispatch::upsample_nearest3d(ks, )); break;
case H_UPSAMPLE_NEAREST3D_BACKWARD_VEC: set(op.tensor, at::redispatch::upsample_nearest3d_backward(ks, )); break;
case H_UPSAMPLE_LINEAR1D_OUT: TODO
case H_UPSAMPLE_LINEAR1D_BACKWARD_GRAD_INPUT: TODO
case H_UPSAMPLE_BILINEAR2D_OUT: TODO
case H_UPSAMPLE_BILINEAR2D: set(op.tensor, at::redispatch::upsample_bilinear2d(ks, )); break;
case H_UPSAMPLE_BILINEAR2D_BACKWARD_GRAD_INPUT: TODO
case H_UPSAMPLE_BICUBIC2D_OUT: TODO
case H_UPSAMPLE_BICUBIC2D_BACKWARD_GRAD_INPUT: TODO
case H_UPSAMPLE_TRILINEAR3D_OUT: TODO
case H_UPSAMPLE_TRILINEAR3D_BACKWARD_GRAD_INPUT: TODO
case H_UPSAMPLE_NEAREST1D_OUT: TODO
case H_UPSAMPLE_NEAREST1D_BACKWARD_GRAD_INPUT: TODO
case H_UPSAMPLE_NEAREST2D_OUT: TODO
case H_UPSAMPLE_NEAREST2D: set(op.tensor, at::redispatch::upsample_nearest2d(ks, )); break;
case H_UPSAMPLE_NEAREST2D_BACKWARD_GRAD_INPUT: TODO
case H_UPSAMPLE_NEAREST3D_OUT: TODO
case H_UPSAMPLE_NEAREST3D: set(op.tensor, at::redispatch::upsample_nearest3d(ks, )); break;
case H_UPSAMPLE_NEAREST3D_BACKWARD_GRAD_INPUT: TODO
case H_SIGMOID_BACKWARD_GRAD_INPUT: TODO
case H_SIGMOID_BACKWARD: set(op.tensor, at::redispatch::sigmoid_backward(ks, )); break;
case H_LOGIT_BACKWARD_GRAD_INPUT: TODO
case H_LOGIT_BACKWARD: set(op.tensor, at::redispatch::logit_backward(ks, )); break;
case H_TANH_BACKWARD_GRAD_INPUT: TODO
case H_TANH_BACKWARD: set(op.tensor, at::redispatch::tanh_backward(ks, )); break;
case H_SLOW_CONV_TRANSPOSE2D_OUT: TODO
case H_SLOW_CONV_TRANSPOSE2D: set(op.tensor, at::redispatch::slow_conv_transpose2d(ks, )); break;
case H_SLOW_CONV_TRANSPOSE2D_BACKWARD_GRAD_OUTPUT: TODO
case H_SLOW_CONV_TRANSPOSE2D_BACKWARD_OUTPUT_MASK: set(op.tensor, at::redispatch::slow_conv_transpose2d_backward(ks, )); break;
case H_SLOW_CONV_TRANSPOSE3D_OUT: TODO
case H_SLOW_CONV_TRANSPOSE3D: set(op.tensor, at::redispatch::slow_conv_transpose3d(ks, )); break;
case H_SLOW_CONV_TRANSPOSE3D_BACKWARD_GRAD_OUTPUT: TODO
case H_SLOW_CONV_TRANSPOSE3D_BACKWARD_OUTPUT_MASK: set(op.tensor, at::redispatch::slow_conv_transpose3d_backward(ks, )); break;
case H_THNN_CONV2D_OUT: TODO
case H_THNN_CONV2D: set(op.tensor, at::redispatch::thnn_conv2d(ks, )); break;
case H_THNN_CONV2D_FORWARD_OUTPUT: TODO
case H_THNN_CONV2D_FORWARD: set(op.tensor, at::redispatch::thnn_conv2d_forward(ks, )); break;
case H_THNN_CONV2D_BACKWARD_GRAD_INPUT: TODO
case H_THNN_CONV2D_BACKWARD_OUTPUT_MASK: set(op.tensor, at::redispatch::thnn_conv2d_backward(ks, )); break;
case H_THNN_CONV_DEPTHWISE2D_OUT: TODO
case H_THNN_CONV_DEPTHWISE2D: set(op.tensor, at::redispatch::thnn_conv_depthwise2d(ks, )); break;
case H_THNN_CONV_DEPTHWISE2D_FORWARD_OUT: TODO
case H_THNN_CONV_DEPTHWISE2D_FORWARD: set(op.tensor, at::redispatch::thnn_conv_depthwise2d_forward(ks, )); break;
case H_THNN_CONV_DEPTHWISE2D_BACKWARD_GRAD_INPUT: TODO
case H_THNN_CONV_DEPTHWISE2D_BACKWARD_OUTPUT_MASK: set(op.tensor, at::redispatch::thnn_conv_depthwise2d_backward(ks, )); break;
case H_CONV_DEPTHWISE3D: set(op.tensor, at::redispatch::conv_depthwise3d(ks, )); break;
case H_CONV_DEPTHWISE3D_BACKWARD_GRAD_INPUT: TODO
case H_CONV_DEPTHWISE3D_BACKWARD_OUTPUT_MASK: set(op.tensor, at::redispatch::conv_depthwise3d_backward(ks, )); break;
case H_SLOW_CONV3D_OUT: TODO
case H_SLOW_CONV3D: set(op.tensor, at::redispatch::slow_conv3d(ks, )); break;
case H_SLOW_CONV3D_FORWARD_OUTPUT: TODO
case H_SLOW_CONV3D_FORWARD: set(op.tensor, at::redispatch::slow_conv3d_forward(ks, )); break;
case H_SLOW_CONV3D_BACKWARD_GRAD_INPUT: TODO
case H_SLOW_CONV3D_BACKWARD_OUTPUT_MASK: set(op.tensor, at::redispatch::slow_conv3d_backward(ks, )); break;
case H_SLOW_CONV_DILATED2D: set(op.tensor, at::redispatch::slow_conv_dilated2d(ks, )); break;
case H_SLOW_CONV_DILATED2D_BACKWARD: set(op.tensor, at::redispatch::slow_conv_dilated2d_backward(ks, )); break;
case H_SLOW_CONV_DILATED3D: set(op.tensor, at::redispatch::slow_conv_dilated3d(ks, )); break;
case H_SLOW_CONV_DILATED3D_BACKWARD: set(op.tensor, at::redispatch::slow_conv_dilated3d_backward(ks, )); break;
case H_COL2IM_OUT: TODO
case H_COL2IM: set(op.tensor, at::redispatch::col2im(ks, )); break;
case H_COL2IM_BACKWARD_GRAD_INPUT: TODO
case H_COL2IM_BACKWARD: set(op.tensor, at::redispatch::col2im_backward(ks, )); break;
case H_COLUMN_STACK: set(op.tensor, at::redispatch::column_stack(ks, )); break;
case H_COLUMN_STACK_OUT: TODO
case H_IM2COL_OUT: TODO
case H_IM2COL: set(op.tensor, at::redispatch::im2col(ks, )); break;
case H_IM2COL_BACKWARD_GRAD_INPUT: TODO
case H_IM2COL_BACKWARD: set(op.tensor, at::redispatch::im2col_backward(ks, )); break;
case H_ISFINITE: set(op.tensor, at::redispatch::isfinite(ks, )); break;
case H_ISINF: set(op.tensor, at::redispatch::isinf(ks, )); break;
case H_RECORD_STREAM: set(op.tensor, at::redispatch::record_stream(ks, )); break;
case H_ISPOSINF: set(op.tensor, at::redispatch::isposinf(ks, )); break;
case H_ISPOSINF_OUT: TODO
case H_ISNEGINF: set(op.tensor, at::redispatch::isneginf(ks, )); break;
case H_ISNEGINF_OUT: TODO
case H__ADD_BATCH_DIM: set(op.tensor, at::redispatch::_add_batch_dim(ks, )); break;
case H__REMOVE_BATCH_DIM: set(op.tensor, at::redispatch::_remove_batch_dim(ks, )); break;
case H_SPECIAL_ENTR_OUT: TODO
case H_SPECIAL_EXPM1: set(op.tensor, at::redispatch::special_expm1(ks, )); break;
case H_SPECIAL_EXPM1_OUT: TODO
case H_SPECIAL_EXP2: set(op.tensor, at::redispatch::special_exp2(ks, )); break;
case H_SPECIAL_EXP2_OUT: TODO
case H_SPECIAL_GAMMALN: set(op.tensor, at::redispatch::special_gammaln(ks, )); break;
case H_SPECIAL_GAMMALN_OUT: TODO
case H_SPECIAL_ERF: set(op.tensor, at::redispatch::special_erf(ks, )); break;
case H_SPECIAL_ERF_OUT: TODO
case H_SPECIAL_ERFC: set(op.tensor, at::redispatch::special_erfc(ks, )); break;
case H_SPECIAL_ERFC_OUT: TODO
case H_SPECIAL_ERFINV: set(op.tensor, at::redispatch::special_erfinv(ks, )); break;
case H_SPECIAL_ERFINV_OUT: TODO
case H_SPECIAL_I0E_OUT: TODO
case H_SPECIAL_LOGIT: set(op.tensor, at::redispatch::special_logit(ks, )); break;
case H_SPECIAL_LOGIT_OUT: TODO
case H_SPECIAL_EXPIT: set(op.tensor, at::redispatch::special_expit(ks, )); break;
case H_SPECIAL_EXPIT_OUT: TODO
case H_FFT_FFT: set(op.tensor, at::redispatch::fft_fft(ks, )); break;
case H_FFT_FFT_OUT: TODO
case H_FFT_IFFT: set(op.tensor, at::redispatch::fft_ifft(ks, )); break;
case H_FFT_IFFT_OUT: TODO
case H_FFT_RFFT: set(op.tensor, at::redispatch::fft_rfft(ks, )); break;
case H_FFT_RFFT_OUT: TODO
case H_FFT_IRFFT: set(op.tensor, at::redispatch::fft_irfft(ks, )); break;
case H_FFT_IRFFT_OUT: TODO
case H_FFT_HFFT: set(op.tensor, at::redispatch::fft_hfft(ks, )); break;
case H_FFT_HFFT_OUT: TODO
case H_FFT_IHFFT: set(op.tensor, at::redispatch::fft_ihfft(ks, )); break;
case H_FFT_IHFFT_OUT: TODO
case H_FFT_FFT2: set(op.tensor, at::redispatch::fft_fft2(ks, )); break;
case H_FFT_FFT2_OUT: TODO
case H_FFT_IFFT2: set(op.tensor, at::redispatch::fft_ifft2(ks, )); break;
case H_FFT_IFFT2_OUT: TODO
case H_FFT_RFFT2: set(op.tensor, at::redispatch::fft_rfft2(ks, )); break;
case H_FFT_RFFT2_OUT: TODO
case H_FFT_IRFFT2: set(op.tensor, at::redispatch::fft_irfft2(ks, )); break;
case H_FFT_IRFFT2_OUT: TODO
case H_FFT_FFTN: set(op.tensor, at::redispatch::fft_fftn(ks, )); break;
case H_FFT_FFTN_OUT: TODO
case H_FFT_IFFTN: set(op.tensor, at::redispatch::fft_ifftn(ks, )); break;
case H_FFT_IFFTN_OUT: TODO
case H_FFT_RFFTN: set(op.tensor, at::redispatch::fft_rfftn(ks, )); break;
case H_FFT_RFFTN_OUT: TODO
case H_FFT_IRFFTN: set(op.tensor, at::redispatch::fft_irfftn(ks, )); break;
case H_FFT_IRFFTN_OUT: TODO
case H_FFT_FFTFREQ: set(op.tensor, at::redispatch::fft_fftfreq(ks, )); break;
case H_FFT_FFTFREQ_OUT: TODO
case H_FFT_RFFTFREQ: set(op.tensor, at::redispatch::fft_rfftfreq(ks, )); break;
case H_FFT_RFFTFREQ_OUT: TODO
case H_FFT_FFTSHIFT: set(op.tensor, at::redispatch::fft_fftshift(ks, )); break;
case H_FFT_IFFTSHIFT: set(op.tensor, at::redispatch::fft_ifftshift(ks, )); break;
case H_LINALG_CHOLESKY: set(op.tensor, at::redispatch::linalg_cholesky(ks, )); break;
case H_LINALG_CHOLESKY_OUT: TODO
case H_LINALG_DET: set(op.tensor, at::redispatch::linalg_det(ks, )); break;
case H_LINALG_DET_OUT: TODO
case H_DET: set(op.tensor, at::redispatch::det(ks, )); break;
case H_LINALG_LSTSQ: set(op.tensor, at::redispatch::linalg_lstsq(ks, )); break;
case H_LINALG_LSTSQ_OUT: TODO
case H__LSTSQ_HELPER_: TODO
case H_LINALG_SLOGDET: set(op.tensor, at::redispatch::linalg_slogdet(ks, )); break;
case H_LINALG_SLOGDET_OUT: TODO
case H_LINALG_EIG: set(op.tensor, at::redispatch::linalg_eig(ks, )); break;
case H_LINALG_EIG_OUT: TODO
case H_LINALG_EIGVALS: set(op.tensor, at::redispatch::linalg_eigvals(ks, )); break;
case H_LINALG_EIGVALS_OUT: TODO
case H_LINALG_EIGH: set(op.tensor, at::redispatch::linalg_eigh(ks, )); break;
case H_LINALG_EIGH_EIGVALS: TODO
case H_LINALG_EIGVALSH: set(op.tensor, at::redispatch::linalg_eigvalsh(ks, )); break;
case H_LINALG_EIGVALSH_OUT: TODO
case H_LINALG_HOUSEHOLDER_PRODUCT: set(op.tensor, at::redispatch::linalg_householder_product(ks, )); break;
case H_LINALG_HOUSEHOLDER_PRODUCT_OUT: TODO
case H__LINALG_INV_OUT_HELPER_: TODO
case H_LINALG_INV: set(op.tensor, at::redispatch::linalg_inv(ks, )); break;
case H_LINALG_INV_OUT: TODO
case H_INNER: set(op.tensor, at::redispatch::inner(ks, )); break;
case H_INNER_OUT: TODO
case H_OUTER: set(op.tensor, at::redispatch::outer(ks, )); break;
case H_OUTER_OUT: TODO
case H_GER: set(op.tensor, at::redispatch::ger(ks, )); break;
case H_GER_OUT: TODO
case H_LINALG_NORM: set(op.tensor, at::redispatch::linalg_norm(ks, )); break;
case H_LINALG_NORM_ORD_STR: set(op.tensor, at::redispatch::linalg_norm(ks, )); break;
case H_LINALG_NORM_OUT: TODO
case H_LINALG_NORM_ORD_STR_OUT: TODO
case H_LINALG_VECTOR_NORM: set(op.tensor, at::redispatch::linalg_vector_norm(ks, )); break;
case H_LINALG_VECTOR_NORM_OUT: TODO
case H_LINALG_SVD_U: TODO
case H_LINALG_SVD: set(op.tensor, at::redispatch::linalg_svd(ks, )); break;
case H_LINALG_COND: set(op.tensor, at::redispatch::linalg_cond(ks, )); break;
case H_LINALG_COND_OUT: TODO
case H_LINALG_COND_P_STR: set(op.tensor, at::redispatch::linalg_cond(ks, )); break;
case H_LINALG_COND_P_STR_OUT: TODO
case H_LINALG_PINV: set(op.tensor, at::redispatch::linalg_pinv(ks, )); break;
case H_LINALG_PINV_RCOND_TENSOR: set(op.tensor, at::redispatch::linalg_pinv(ks, )); break;
case H_LINALG_PINV_OUT: TODO
case H_LINALG_PINV_OUT_RCOND_TENSOR: TODO
case H__LINALG_SOLVE_OUT_HELPER_: TODO
case H_LINALG_SOLVE: set(op.tensor, at::redispatch::linalg_solve(ks, )); break;
case H_LINALG_SOLVE_OUT: TODO
case H_LINALG_TENSORINV: set(op.tensor, at::redispatch::linalg_tensorinv(ks, )); break;
case H_LINALG_TENSORINV_OUT: TODO
case H_LINALG_TENSORSOLVE: set(op.tensor, at::redispatch::linalg_tensorsolve(ks, )); break;
case H_LINALG_TENSORSOLVE_OUT: TODO
case H_LINALG_QR: set(op.tensor, at::redispatch::linalg_qr(ks, )); break;
case H_LINALG_QR_OUT: TODO
case H__LINALG_QR_HELPER: set(op.tensor, at::redispatch::_linalg_qr_helper(ks, )); break;
case H_LINALG_MATRIX_POWER: set(op.tensor, at::redispatch::linalg_matrix_power(ks, )); break;
case H_LINALG_MATRIX_POWER_OUT: TODO
case H_LINALG_MATRIX_RANK: set(op.tensor, at::redispatch::linalg_matrix_rank(ks, )); break;
case H_LINALG_MATRIX_RANK_OUT: TODO
case H_LINALG_MULTI_DOT: set(op.tensor, at::redispatch::linalg_multi_dot(ks, )); break;
case H_LINALG_MULTI_DOT_OUT: TODO
case H__TEST_SERIALIZATION_SUBCMUL: set(op.tensor, at::redispatch::_test_serialization_subcmul(ks, )); break;
case H__TEST_OPTIONAL_INTLIST: set(op.tensor, at::redispatch::_test_optional_intlist(ks, )); break;
case H__TEST_OPTIONAL_FILLED_INTLIST: set(op.tensor, at::redispatch::_test_optional_filled_intlist(ks, )); break;
case H__TEST_OPTIONAL_FLOATLIST: set(op.tensor, at::redispatch::_test_optional_floatlist(ks, )); break;
case H__TEST_STRING_DEFAULT: set(op.tensor, at::redispatch::_test_string_default(ks, )); break;
case H__TEST_AMBIGUOUS_DEFAULTS_A: set(op.tensor, at::redispatch::_test_ambiguous_defaults(ks, )); break;
case H__TEST_AMBIGUOUS_DEFAULTS_B: set(op.tensor, at::redispatch::_test_ambiguous_defaults(ks, )); break;
case H_SEGMENT_REDUCE: set(op.tensor, at::redispatch::segment_reduce(ks, )); break;
