
case H__CAST_BYTE:
  set(op.tensor, at::redispatch::_cast_Byte(ks, self, non_blocking));
  break;


case H__CAST_CHAR:
  set(op.tensor, at::redispatch::_cast_Char(ks, self, non_blocking));
  break;


case H__CAST_DOUBLE:
  set(op.tensor, at::redispatch::_cast_Double(ks, self, non_blocking));
  break;


case H__CAST_FLOAT:
  set(op.tensor, at::redispatch::_cast_Float(ks, self, non_blocking));
  break;


case H__CAST_INT:
  set(op.tensor, at::redispatch::_cast_Int(ks, self, non_blocking));
  break;


case H__CAST_LONG:
  set(op.tensor, at::redispatch::_cast_Long(ks, self, non_blocking));
  break;


case H__CAST_SHORT:
  set(op.tensor, at::redispatch::_cast_Short(ks, self, non_blocking));
  break;


case H__CAST_HALF:
  set(op.tensor, at::redispatch::_cast_Half(ks, self, non_blocking));
  break;


case H__FW_PRIMAL:
  set(op.tensor, at::redispatch::_fw_primal(ks, self, level));
  break;


case H__MAKE_DUAL:
  set(op.tensor, at::redispatch::_make_dual(ks, primal, tangent, level));
  break;


case H_RENAME_:
  init_update_in_place(op.tensor);
  at::redispatch::rename_(ks, self, names)
  end_update_in_place(op.tensor);
  break;


case H_RENAME:
  set(op.tensor, at::redispatch::rename(ks, self, names));
  break;


case H_ALIGN_TO:
  set(op.tensor, at::redispatch::align_to(ks, self, names));
  break;


case H_ALIGN_TO_ELLIPSIS_IDX:
  set(op.tensor, at::redispatch::align_to(ks, self, order, ellipsis_idx));
  break;


case H_ALIGN_AS:
  set(op.tensor, at::redispatch::align_as(ks, self, other));
  break;




case H_REFINE_NAMES:
  set(op.tensor, at::redispatch::refine_names(ks, self, names));
  break;





case H__CUDNN_RNN_FLATTEN_WEIGHT:
  set(op.tensor, at::redispatch::_cudnn_rnn_flatten_weight(ks, weight_arr, weight_stride0, input_size, mode, hidden_size, proj_size, num_layers, batch_first, bidirectional));
  break;




case H__CUDNN_INIT_DROPOUT_STATE:
  set(op.tensor, at::redispatch::_cudnn_init_dropout_state(ks, dropout, train, dropout_seed, dtype, layout, device, pin_memory));
  break;




case H__MASKED_SCALE:
  set(op.tensor, at::redispatch::_masked_scale(ks, self, mask, scale));
  break;


case H__SOBOL_ENGINE_FF_:
  init_update_in_place(op.tensor);
  at::redispatch::_sobol_engine_ff_(ks, self, n, sobolstate, dimension, num_generated)
  end_update_in_place(op.tensor);
  break;

case H__SOBOL_ENGINE_SCRAMBLE_:
  init_update_in_place(op.tensor);
  at::redispatch::_sobol_engine_scramble_(ks, self, ltm, dimension)
  end_update_in_place(op.tensor);
  break;

case H__SOBOL_ENGINE_INITIALIZE_STATE_:
  init_update_in_place(op.tensor);
  at::redispatch::_sobol_engine_initialize_state_(ks, self, dimension)
  end_update_in_place(op.tensor);
  break;


case H__RESHAPE_FROM_TENSOR:
  set(op.tensor, at::redispatch::_reshape_from_tensor(ks, self, shape));
  break;


case H__SHAPE_AS_TENSOR:
  set(op.tensor, at::redispatch::_shape_as_tensor(ks, self));
  break;


case H_DROPOUT:
  set(op.tensor, at::redispatch::dropout(ks, input, p, train));
  break;

case H_DROPOUT_:
  init_update_in_place(op.tensor);
  at::redispatch::dropout_(ks, self, p, train)
  end_update_in_place(op.tensor);
  break;


case H_FEATURE_DROPOUT:
  set(op.tensor, at::redispatch::feature_dropout(ks, input, p, train));
  break;

case H_FEATURE_DROPOUT_:
  init_update_in_place(op.tensor);
  at::redispatch::feature_dropout_(ks, self, p, train)
  end_update_in_place(op.tensor);
  break;


case H_ALPHA_DROPOUT:
  set(op.tensor, at::redispatch::alpha_dropout(ks, input, p, train));
  break;

case H_ALPHA_DROPOUT_:
  init_update_in_place(op.tensor);
  at::redispatch::alpha_dropout_(ks, self, p, train)
  end_update_in_place(op.tensor);
  break;


case H_FEATURE_ALPHA_DROPOUT:
  set(op.tensor, at::redispatch::feature_alpha_dropout(ks, input, p, train));
  break;

case H_FEATURE_ALPHA_DROPOUT_:
  init_update_in_place(op.tensor);
  at::redispatch::feature_alpha_dropout_(ks, self, p, train)
  end_update_in_place(op.tensor);
  break;


case H_ABS:
  set(op.tensor, at::redispatch::abs(ks, self));
  break;

case H_ABS_:
  init_update_in_place(op.tensor);
  at::redispatch::abs_(ks, self)
  end_update_in_place(op.tensor);
  break;

case H_ABS_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::abs_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_ABSOLUTE:
  set(op.tensor, at::redispatch::absolute(ks, self));
  break;

case H_ABSOLUTE_:
  init_update_in_place(op.tensor);
  at::redispatch::absolute_(ks, self)
  end_update_in_place(op.tensor);
  break;

case H_ABSOLUTE_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::absolute_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_ANGLE:
  set(op.tensor, at::redispatch::angle(ks, self));
  break;

case H_ANGLE_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::angle_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_VIEW_AS_REAL:
  set(op.tensor, at::redispatch::view_as_real(ks, self));
  break;


case H_VIEW_AS_COMPLEX:
  set(op.tensor, at::redispatch::view_as_complex(ks, self));
  break;


case H_SGN:
  set(op.tensor, at::redispatch::sgn(ks, self));
  break;

case H_SGN_:
  init_update_in_place(op.tensor);
  at::redispatch::sgn_(ks, self)
  end_update_in_place(op.tensor);
  break;

case H_SGN_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::sgn_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_REAL:
  set(op.tensor, at::redispatch::real(ks, self));
  break;


case H_IMAG:
  set(op.tensor, at::redispatch::imag(ks, self));
  break;


case H_CONJ:
  set(op.tensor, at::redispatch::conj(ks, self));
  break;

case H_CONJ_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::conj_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H__CONJ:
  set(op.tensor, at::redispatch::_conj(ks, self));
  break;

case H_ACOS_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::acos_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_ARCCOS:
  set(op.tensor, at::redispatch::arccos(ks, self));
  break;

case H_ARCCOS_:
  init_update_in_place(op.tensor);
  at::redispatch::arccos_(ks, self)
  end_update_in_place(op.tensor);
  break;

case H_ARCCOS_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::arccos_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_AVG_POOL1D:
  set(op.tensor, at::redispatch::avg_pool1d(ks, self, kernel_size, stride, padding, ceil_mode, count_include_pad));
  break;


case H_ADAPTIVE_AVG_POOL1D:
  set(op.tensor, at::redispatch::adaptive_avg_pool1d(ks, self, output_size));
  break;



case H_ADD_TENSOR:
  set(op.tensor, at::redispatch::add(ks, self, other, alpha));
  break;

case H_ADD__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::add_(ks, self, other, alpha)
  end_update_in_place(op.tensor);
  break;

case H_ADD_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::add_outf(ks, self, other, alpha, out)
  end_update_in_place(op.tensor);
  break;


case H__ADD_RELU_TENSOR:
  set(op.tensor, at::redispatch::_add_relu(ks, self, other, alpha));
  break;

case H__ADD_RELU__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::_add_relu_(ks, self, other, alpha)
  end_update_in_place(op.tensor);
  break;

case H__ADD_RELU_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::_add_relu_outf(ks, self, other, alpha, out)
  end_update_in_place(op.tensor);
  break;


case H_ADD_SCALAR:
  set(op.tensor, at::redispatch::add(ks, self, other, alpha));
  break;

case H_ADD__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::add_(ks, self, other, alpha)
  end_update_in_place(op.tensor);
  break;

case H_ADDMV_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::addmv_outf(ks, self, mat, vec, beta, alpha, out)
  end_update_in_place(op.tensor);
  break;


case H_ADDR:
  set(op.tensor, at::redispatch::addr(ks, self, vec1, vec2, beta, alpha));
  break;

case H_ADDR_:
  init_update_in_place(op.tensor);
  at::redispatch::addr_(ks, self, vec1, vec2, beta, alpha)
  end_update_in_place(op.tensor);
  break;

case H_ADDR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::addr_outf(ks, self, vec1, vec2, beta, alpha, out)
  end_update_in_place(op.tensor);
  break;


case H_AFFINE_GRID_GENERATOR:
  set(op.tensor, at::redispatch::affine_grid_generator(ks, theta, size, align_corners));
  break;


case H_AFFINE_GRID_GENERATOR_BACKWARD:
  set(op.tensor, at::redispatch::affine_grid_generator_backward(ks, grad, size, align_corners));
  break;


case H_ALL_DIM:
  set(op.tensor, at::redispatch::all(ks, self, dim, keepdim));
  break;

case H_ALL_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::all_outf(ks, self, dim, keepdim, out)
  end_update_in_place(op.tensor);
  break;


case H_ALL_DIMNAME:
  set(op.tensor, at::redispatch::all(ks, self, dim, keepdim));
  break;

case H_ALL_DIMNAME_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::all_outf(ks, self, dim, keepdim, out)
  end_update_in_place(op.tensor);
  break;



case H_ANY_DIM:
  set(op.tensor, at::redispatch::any(ks, self, dim, keepdim));
  break;

case H_ANY_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::any_outf(ks, self, dim, keepdim, out)
  end_update_in_place(op.tensor);
  break;


case H_ANY_DIMNAME:
  set(op.tensor, at::redispatch::any(ks, self, dim, keepdim));
  break;

case H_ANY_DIMNAME_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::any_outf(ks, self, dim, keepdim, out)
  end_update_in_place(op.tensor);
  break;


case H_ARANGE:
  set(op.tensor, at::redispatch::arange(ks, end, dtype, layout, device, pin_memory));
  break;


case H_ARANGE_START:
  set(op.tensor, at::redispatch::arange(ks, start, end, dtype, layout, device, pin_memory));
  break;


case H_ARANGE_START_STEP:
  set(op.tensor, at::redispatch::arange(ks, start, end, step, dtype, layout, device, pin_memory));
  break;

case H_ARANGE_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::arange_outf(ks, end, out)
  end_update_in_place(op.tensor);
  break;

case H_ARANGE_START_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::arange_outf(ks, start, end, step, out)
  end_update_in_place(op.tensor);
  break;


case H__DIM_ARANGE:
  set(op.tensor, at::redispatch::_dim_arange(ks, like, dim));
  break;


case H_ARGMAX:
  set(op.tensor, at::redispatch::argmax(ks, self, dim, keepdim));
  break;

case H_ARGMAX_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::argmax_outf(ks, self, dim, keepdim, out)
  end_update_in_place(op.tensor);
  break;


case H_ARGMIN:
  set(op.tensor, at::redispatch::argmin(ks, self, dim, keepdim));
  break;

case H_ARGMIN_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::argmin_outf(ks, self, dim, keepdim, out)
  end_update_in_place(op.tensor);
  break;

case H_ACOSH_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::acosh_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_ARCCOSH:
  set(op.tensor, at::redispatch::arccosh(ks, self));
  break;

case H_ARCCOSH_:
  init_update_in_place(op.tensor);
  at::redispatch::arccosh_(ks, self)
  end_update_in_place(op.tensor);
  break;

case H_ARCCOSH_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::arccosh_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;

case H_ASINH_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::asinh_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_ARCSINH:
  set(op.tensor, at::redispatch::arcsinh(ks, self));
  break;

case H_ARCSINH_:
  init_update_in_place(op.tensor);
  at::redispatch::arcsinh_(ks, self)
  end_update_in_place(op.tensor);
  break;

case H_ARCSINH_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::arcsinh_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;

case H_ATANH_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::atanh_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_ARCTANH:
  set(op.tensor, at::redispatch::arctanh(ks, self));
  break;

case H_ARCTANH_:
  init_update_in_place(op.tensor);
  at::redispatch::arctanh_(ks, self)
  end_update_in_place(op.tensor);
  break;

case H_ARCTANH_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::arctanh_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_AS_STRIDED:
  set(op.tensor, at::redispatch::as_strided(ks, self, size, stride, storage_offset));
  break;

case H_AS_STRIDED_:
  init_update_in_place(op.tensor);
  at::redispatch::as_strided_(ks, self, size, stride, storage_offset)
  end_update_in_place(op.tensor);
  break;


case H_ASIN:
  set(op.tensor, at::redispatch::asin(ks, self));
  break;

case H_ASIN_:
  init_update_in_place(op.tensor);
  at::redispatch::asin_(ks, self)
  end_update_in_place(op.tensor);
  break;

case H_ASIN_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::asin_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_ARCSIN:
  set(op.tensor, at::redispatch::arcsin(ks, self));
  break;

case H_ARCSIN_:
  init_update_in_place(op.tensor);
  at::redispatch::arcsin_(ks, self)
  end_update_in_place(op.tensor);
  break;

case H_ARCSIN_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::arcsin_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;

case H_ATAN_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::atan_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_ARCTAN:
  set(op.tensor, at::redispatch::arctan(ks, self));
  break;

case H_ARCTAN_:
  init_update_in_place(op.tensor);
  at::redispatch::arctan_(ks, self)
  end_update_in_place(op.tensor);
  break;

case H_ARCTAN_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::arctan_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_ATLEAST_1D:
  set(op.tensor, at::redispatch::atleast_1d(ks, self));
  break;



case H_ATLEAST_2D:
  set(op.tensor, at::redispatch::atleast_2d(ks, self));
  break;



case H_ATLEAST_3D:
  set(op.tensor, at::redispatch::atleast_3d(ks, self));
  break;



case H_BADDBMM:
  set(op.tensor, at::redispatch::baddbmm(ks, self, batch1, batch2, beta, alpha));
  break;

case H_BADDBMM_:
  init_update_in_place(op.tensor);
  at::redispatch::baddbmm_(ks, self, batch1, batch2, beta, alpha)
  end_update_in_place(op.tensor);
  break;

case H__BADDBMM_MKL_:
  init_update_in_place(op.tensor);
  at::redispatch::_baddbmm_mkl_(ks, self, batch1, batch2, beta, alpha)
  end_update_in_place(op.tensor);
  break;

case H_BADDBMM_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::baddbmm_outf(ks, self, batch1, batch2, beta, alpha, out)
  end_update_in_place(op.tensor);
  break;


case H_BARTLETT_WINDOW:
  set(op.tensor, at::redispatch::bartlett_window(ks, window_length, dtype, layout, device, pin_memory));
  break;


case H_BARTLETT_WINDOW_PERIODIC:
  set(op.tensor, at::redispatch::bartlett_window(ks, window_length, periodic, dtype, layout, device, pin_memory));
  break;


case H_BATCH_NORM:
  set(op.tensor, at::redispatch::batch_norm(ks, input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled));
  break;


case H_QUANTIZED_BATCH_NORM:
  set(op.tensor, at::redispatch::quantized_batch_norm(ks, input, weight, bias, mean, var, eps, output_scale, output_zero_point));
  break;




case H_BERNOULLI:
  set(op.tensor, at::redispatch::bernoulli(ks, self, generator));
  break;

case H_BERNOULLI_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::bernoulli_outf(ks, self, generator, out)
  end_update_in_place(op.tensor);
  break;

case H_BERNOULLI__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::bernoulli_(ks, self, p, generator)
  end_update_in_place(op.tensor);
  break;

case H_BERNOULLI__FLOAT:
  init_update_in_place(op.tensor);
  at::redispatch::bernoulli_(ks, self, p, generator)
  end_update_in_place(op.tensor);
  break;


case H_BERNOULLI_P:
  set(op.tensor, at::redispatch::bernoulli(ks, self, p, generator));
  break;


case H_BILINEAR:
  set(op.tensor, at::redispatch::bilinear(ks, input1, input2, weight, bias));
  break;


case H_BINARY_CROSS_ENTROPY:
  set(op.tensor, at::redispatch::binary_cross_entropy(ks, self, target, weight, reduction));
  break;

case H_BINARY_CROSS_ENTROPY_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::binary_cross_entropy_outf(ks, self, target, weight, reduction, out)
  end_update_in_place(op.tensor);
  break;


case H_BINARY_CROSS_ENTROPY_BACKWARD:
  set(op.tensor, at::redispatch::binary_cross_entropy_backward(ks, grad_output, self, target, weight, reduction));
  break;

case H_BINARY_CROSS_ENTROPY_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::binary_cross_entropy_backward_outf(ks, grad_output, self, target, weight, reduction, grad_input)
  end_update_in_place(op.tensor);
  break;


case H_BINARY_CROSS_ENTROPY_WITH_LOGITS:
  set(op.tensor, at::redispatch::binary_cross_entropy_with_logits(ks, self, target, weight, pos_weight, reduction));
  break;


case H_BINARY_CROSS_ENTROPY_WITH_LOGITS_BACKWARD:
  set(op.tensor, at::redispatch::binary_cross_entropy_with_logits_backward(ks, grad_output, self, target, weight, pos_weight, reduction));
  break;


case H_BINCOUNT:
  set(op.tensor, at::redispatch::bincount(ks, self, weights, minlength));
  break;


case H_BITWISE_NOT:
  set(op.tensor, at::redispatch::bitwise_not(ks, self));
  break;

case H_BITWISE_NOT_:
  init_update_in_place(op.tensor);
  at::redispatch::bitwise_not_(ks, self)
  end_update_in_place(op.tensor);
  break;

case H_BITWISE_NOT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::bitwise_not_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;

case H_COPYSIGN_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::copysign_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_COPYSIGN_SCALAR:
  set(op.tensor, at::redispatch::copysign(ks, self, other));
  break;

case H_COPYSIGN__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::copysign_(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H_COPYSIGN_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::copysign_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_LOGICAL_NOT:
  set(op.tensor, at::redispatch::logical_not(ks, self));
  break;

case H_LOGICAL_NOT_:
  init_update_in_place(op.tensor);
  at::redispatch::logical_not_(ks, self)
  end_update_in_place(op.tensor);
  break;

case H_LOGICAL_NOT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::logical_not_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_LOGICAL_XOR:
  set(op.tensor, at::redispatch::logical_xor(ks, self, other));
  break;

case H_LOGICAL_XOR_:
  init_update_in_place(op.tensor);
  at::redispatch::logical_xor_(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H_LOGICAL_XOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::logical_xor_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_LOGICAL_AND:
  set(op.tensor, at::redispatch::logical_and(ks, self, other));
  break;

case H_LOGICAL_AND_:
  init_update_in_place(op.tensor);
  at::redispatch::logical_and_(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H_LOGICAL_AND_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::logical_and_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_LOGICAL_OR:
  set(op.tensor, at::redispatch::logical_or(ks, self, other));
  break;

case H_LOGICAL_OR_:
  init_update_in_place(op.tensor);
  at::redispatch::logical_or_(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H_LOGICAL_OR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::logical_or_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_BLACKMAN_WINDOW:
  set(op.tensor, at::redispatch::blackman_window(ks, window_length, dtype, layout, device, pin_memory));
  break;


case H_BLACKMAN_WINDOW_PERIODIC:
  set(op.tensor, at::redispatch::blackman_window(ks, window_length, periodic, dtype, layout, device, pin_memory));
  break;


case H_BMM:
  set(op.tensor, at::redispatch::bmm(ks, self, mat2));
  break;


case H__BMM:
  set(op.tensor, at::redispatch::_bmm(ks, self, mat2, deterministic));
  break;

case H_BMM_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::bmm_outf(ks, self, mat2, out)
  end_update_in_place(op.tensor);
  break;

case H__BMM_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::_bmm_outf(ks, self, mat2, deterministic, out)
  end_update_in_place(op.tensor);
  break;



case H_BROADCAST_TO:
  set(op.tensor, at::redispatch::broadcast_to(ks, self, size));
  break;


case H_CAT:
  set(op.tensor, at::redispatch::cat(ks, tensors, dim));
  break;

case H_CAT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::cat_outf(ks, tensors, dim, out)
  end_update_in_place(op.tensor);
  break;


case H_CAT_NAMES:
  set(op.tensor, at::redispatch::cat(ks, tensors, dim));
  break;

case H_CAT_NAMES_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::cat_outf(ks, tensors, dim, out)
  end_update_in_place(op.tensor);
  break;


case H_BLOCK_DIAG:
  set(op.tensor, at::redispatch::block_diag(ks, tensors));
  break;


case H_CEIL:
  set(op.tensor, at::redispatch::ceil(ks, self));
  break;

case H_CEIL_:
  init_update_in_place(op.tensor);
  at::redispatch::ceil_(ks, self)
  end_update_in_place(op.tensor);
  break;

case H_CEIL_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::ceil_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_CHAIN_MATMUL:
  set(op.tensor, at::redispatch::chain_matmul(ks, matrices));
  break;

case H_CHAIN_MATMUL_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::chain_matmul_outf(ks, matrices, out)
  end_update_in_place(op.tensor);
  break;







case H_CLAMP:
  set(op.tensor, at::redispatch::clamp(ks, self, min, max));
  break;

case H_CLAMP_:
  init_update_in_place(op.tensor);
  at::redispatch::clamp_(ks, self, min, max)
  end_update_in_place(op.tensor);
  break;

case H_CLAMP_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::clamp_outf(ks, self, min, max, out)
  end_update_in_place(op.tensor);
  break;


case H_CLAMP_MAX:
  set(op.tensor, at::redispatch::clamp_max(ks, self, max));
  break;

case H_CLAMP_MAX_:
  init_update_in_place(op.tensor);
  at::redispatch::clamp_max_(ks, self, max)
  end_update_in_place(op.tensor);
  break;

case H_CLAMP_MAX_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::clamp_max_outf(ks, self, max, out)
  end_update_in_place(op.tensor);
  break;


case H_CLAMP_MIN:
  set(op.tensor, at::redispatch::clamp_min(ks, self, min));
  break;

case H_CLAMP_MIN_:
  init_update_in_place(op.tensor);
  at::redispatch::clamp_min_(ks, self, min)
  end_update_in_place(op.tensor);
  break;

case H_CLAMP_MIN_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::clamp_min_outf(ks, self, min, out)
  end_update_in_place(op.tensor);
  break;


case H_CLIP:
  set(op.tensor, at::redispatch::clip(ks, self, min, max));
  break;

case H_CLIP_:
  init_update_in_place(op.tensor);
  at::redispatch::clip_(ks, self, min, max)
  end_update_in_place(op.tensor);
  break;

case H_CLIP_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::clip_outf(ks, self, min, max, out)
  end_update_in_place(op.tensor);
  break;



case H_COMPLEX:
  set(op.tensor, at::redispatch::complex(ks, real, imag));
  break;

case H_COMPLEX_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::complex_outf(ks, real, imag, out)
  end_update_in_place(op.tensor);
  break;


case H_POLAR:
  set(op.tensor, at::redispatch::polar(ks, abs, angle));
  break;

case H_POLAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::polar_outf(ks, abs, angle, out)
  end_update_in_place(op.tensor);
  break;


case H_CONSTANT_PAD_ND:
  set(op.tensor, at::redispatch::constant_pad_nd(ks, self, pad, value));
  break;


case H_CONVOLUTION:
  set(op.tensor, at::redispatch::convolution(ks, input, weight, bias, stride, padding, dilation, transposed, output_padding, groups));
  break;


case H_CONVOLUTION_OVERRIDEABLE:
  set(op.tensor, at::redispatch::convolution_overrideable(ks, input, weight, bias, stride, padding, dilation, transposed, output_padding, groups));
  break;



case H__CONVOLUTION:
  set(op.tensor, at::redispatch::_convolution(ks, input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, allow_tf32));
  break;


case H__CONVOLUTION_DEPRECATED:
  set(op.tensor, at::redispatch::_convolution(ks, input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled));
  break;


case H__CONVOLUTION_MODE:
  set(op.tensor, at::redispatch::_convolution_mode(ks, input, weight, bias, stride, padding, dilation, groups));
  break;


case H__CONVOLUTION_NOGROUP:
  set(op.tensor, at::redispatch::_convolution_nogroup(ks, input, weight, bias, stride, padding, dilation, transposed, output_padding));
  break;



case H_CONV1D:
  set(op.tensor, at::redispatch::conv1d(ks, input, weight, bias, stride, padding, dilation, groups));
  break;


case H_CONV2D:
  set(op.tensor, at::redispatch::conv2d(ks, input, weight, bias, stride, padding, dilation, groups));
  break;


case H_CONV3D:
  set(op.tensor, at::redispatch::conv3d(ks, input, weight, bias, stride, padding, dilation, groups));
  break;


case H_CONV1D_PADDING:
  set(op.tensor, at::redispatch::conv1d(ks, input, weight, bias, stride, padding, dilation, groups));
  break;


case H_CONV2D_PADDING:
  set(op.tensor, at::redispatch::conv2d(ks, input, weight, bias, stride, padding, dilation, groups));
  break;


case H_CONV3D_PADDING:
  set(op.tensor, at::redispatch::conv3d(ks, input, weight, bias, stride, padding, dilation, groups));
  break;


case H_CONV_TBC:
  set(op.tensor, at::redispatch::conv_tbc(ks, self, weight, bias, pad));
  break;



case H_CONV_TRANSPOSE1D:
  set(op.tensor, at::redispatch::conv_transpose1d(ks, input, weight, bias, stride, padding, output_padding, groups, dilation));
  break;


case H_CONV_TRANSPOSE2D_INPUT:
  set(op.tensor, at::redispatch::conv_transpose2d(ks, input, weight, bias, stride, padding, output_padding, groups, dilation));
  break;


case H_CONV_TRANSPOSE3D_INPUT:
  set(op.tensor, at::redispatch::conv_transpose3d(ks, input, weight, bias, stride, padding, output_padding, groups, dilation));
  break;

case H_COPY_:
  init_update_in_place(op.tensor);
  at::redispatch::copy_(ks, self, src, non_blocking)
  end_update_in_place(op.tensor);
  break;

case H_COS_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::cos_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;

case H_COSH_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::cosh_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_COSINE_EMBEDDING_LOSS:
  set(op.tensor, at::redispatch::cosine_embedding_loss(ks, input1, input2, target, margin, reduction));
  break;


case H_COUNT_NONZERO_DIM_INTLIST:
  set(op.tensor, at::redispatch::count_nonzero(ks, self, dim));
  break;


case H_COUNT_NONZERO:
  set(op.tensor, at::redispatch::count_nonzero(ks, self, dim));
  break;


case H_CUDNN_AFFINE_GRID_GENERATOR:
  set(op.tensor, at::redispatch::cudnn_affine_grid_generator(ks, theta, N, C, H, W));
  break;


case H_CUDNN_AFFINE_GRID_GENERATOR_BACKWARD:
  set(op.tensor, at::redispatch::cudnn_affine_grid_generator_backward(ks, grad, N, C, H, W));
  break;




case H_CUDNN_CONVOLUTION_DEPRECATED:
  set(op.tensor, at::redispatch::cudnn_convolution(ks, self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic));
  break;


case H_CUDNN_CONVOLUTION_DEPRECATED2:
  set(op.tensor, at::redispatch::cudnn_convolution(ks, self, weight, padding, stride, dilation, groups, benchmark, deterministic));
  break;


case H_CUDNN_CONVOLUTION:
  set(op.tensor, at::redispatch::cudnn_convolution(ks, self, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32));
  break;


case H_CUDNN_CONVOLUTION_BACKWARD_INPUT:
  set(op.tensor, at::redispatch::cudnn_convolution_backward_input(ks, self_size, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32));
  break;



case H_CUDNN_CONVOLUTION_BACKWARD_WEIGHT:
  set(op.tensor, at::redispatch::cudnn_convolution_backward_weight(ks, weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32));
  break;


case H_CUDNN_CONVOLUTION_TRANSPOSE_DEPRECATED:
  set(op.tensor, at::redispatch::cudnn_convolution_transpose(ks, self, weight, bias, padding, output_padding, stride, dilation, groups, benchmark, deterministic));
  break;


case H_CUDNN_CONVOLUTION_TRANSPOSE_DEPRECATED2:
  set(op.tensor, at::redispatch::cudnn_convolution_transpose(ks, self, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic));
  break;


case H_CUDNN_CONVOLUTION_TRANSPOSE:
  set(op.tensor, at::redispatch::cudnn_convolution_transpose(ks, self, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic, allow_tf32));
  break;



case H_CUDNN_CONVOLUTION_TRANSPOSE_BACKWARD_INPUT:
  set(op.tensor, at::redispatch::cudnn_convolution_transpose_backward_input(ks, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32));
  break;


case H_CUDNN_CONVOLUTION_TRANSPOSE_BACKWARD_WEIGHT:
  set(op.tensor, at::redispatch::cudnn_convolution_transpose_backward_weight(ks, weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32));
  break;


case H_CUDNN_CONVOLUTION_RELU:
  set(op.tensor, at::redispatch::cudnn_convolution_relu(ks, self, weight, bias, stride, padding, dilation, groups));
  break;


case H_CUDNN_CONVOLUTION_ADD_RELU:
  set(op.tensor, at::redispatch::cudnn_convolution_add_relu(ks, self, weight, z, alpha, bias, stride, padding, dilation, groups));
  break;


case H_CUDNN_GRID_SAMPLER:
  set(op.tensor, at::redispatch::cudnn_grid_sampler(ks, self, grid));
  break;













case H_CUMMAXMIN_BACKWARD:
  set(op.tensor, at::redispatch::cummaxmin_backward(ks, grad, input, indices, dim));
  break;


case H_CUMPROD:
  set(op.tensor, at::redispatch::cumprod(ks, self, dim, dtype));
  break;

case H_CUMPROD_:
  init_update_in_place(op.tensor);
  at::redispatch::cumprod_(ks, self, dim, dtype)
  end_update_in_place(op.tensor);
  break;

case H_CUMPROD_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::cumprod_outf(ks, self, dim, dtype, out)
  end_update_in_place(op.tensor);
  break;


case H_CUMPROD_DIMNAME:
  set(op.tensor, at::redispatch::cumprod(ks, self, dim, dtype));
  break;

case H_CUMPROD__DIMNAME:
  init_update_in_place(op.tensor);
  at::redispatch::cumprod_(ks, self, dim, dtype)
  end_update_in_place(op.tensor);
  break;

case H_CUMPROD_DIMNAME_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::cumprod_outf(ks, self, dim, dtype, out)
  end_update_in_place(op.tensor);
  break;


case H_CUMPROD_BACKWARD:
  set(op.tensor, at::redispatch::cumprod_backward(ks, grad, input, dim, output));
  break;


case H_CUMSUM:
  set(op.tensor, at::redispatch::cumsum(ks, self, dim, dtype));
  break;

case H_CUMSUM_:
  init_update_in_place(op.tensor);
  at::redispatch::cumsum_(ks, self, dim, dtype)
  end_update_in_place(op.tensor);
  break;

case H_CUMSUM_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::cumsum_outf(ks, self, dim, dtype, out)
  end_update_in_place(op.tensor);
  break;


case H_CUMSUM_DIMNAME:
  set(op.tensor, at::redispatch::cumsum(ks, self, dim, dtype));
  break;

case H_CUMSUM__DIMNAME:
  init_update_in_place(op.tensor);
  at::redispatch::cumsum_(ks, self, dim, dtype)
  end_update_in_place(op.tensor);
  break;

case H_CUMSUM_DIMNAME_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::cumsum_outf(ks, self, dim, dtype, out)
  end_update_in_place(op.tensor);
  break;


case H_CTC_LOSS_INTLIST:
  set(op.tensor, at::redispatch::ctc_loss(ks, log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity));
  break;


case H_CTC_LOSS_TENSOR:
  set(op.tensor, at::redispatch::ctc_loss(ks, log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity));
  break;



case H__CTC_LOSS_BACKWARD:
  set(op.tensor, at::redispatch::_ctc_loss_backward(ks, grad, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, blank, zero_infinity));
  break;


case H_DIAG_EMBED:
  set(op.tensor, at::redispatch::diag_embed(ks, self, offset, dim1, dim2));
  break;


case H_DIAGFLAT:
  set(op.tensor, at::redispatch::diagflat(ks, self, offset));
  break;


case H_DIAGONAL:
  set(op.tensor, at::redispatch::diagonal(ks, self, offset, dim1, dim2));
  break;


case H_DIAGONAL_DIMNAME:
  set(op.tensor, at::redispatch::diagonal(ks, self, outdim, dim1, dim2, offset));
  break;


case H_DIAGONAL_BACKWARD:
  set(op.tensor, at::redispatch::diagonal_backward(ks, grad, input_sizes, offset, dim1, dim2));
  break;

case H_FILL_DIAGONAL_:
  init_update_in_place(op.tensor);
  at::redispatch::fill_diagonal_(ks, self, fill_value, wrap)
  end_update_in_place(op.tensor);
  break;


case H_DIFF:
  set(op.tensor, at::redispatch::diff(ks, self, n, dim, prepend, append));
  break;

case H_DIFF_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::diff_outf(ks, self, n, dim, prepend, append, out)
  end_update_in_place(op.tensor);
  break;


case H_DIV_TENSOR:
  set(op.tensor, at::redispatch::div(ks, self, other));
  break;

case H_DIV__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::div_(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H_DIV_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::div_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;

case H_DIV_OUT_MODE:
  init_update_in_place(op.tensor);
  at::redispatch::div_outf(ks, self, other, rounding_mode, out)
  end_update_in_place(op.tensor);
  break;


case H_DIV_SCALAR:
  set(op.tensor, at::redispatch::div(ks, self, other));
  break;

case H_DIV__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::div_(ks, self, other)
  end_update_in_place(op.tensor);
  break;


case H_DIV_SCALAR_MODE:
  set(op.tensor, at::redispatch::div(ks, self, other, rounding_mode));
  break;

case H_DIV__SCALAR_MODE:
  init_update_in_place(op.tensor);
  at::redispatch::div_(ks, self, other, rounding_mode)
  end_update_in_place(op.tensor);
  break;


case H_DIVIDE_TENSOR:
  set(op.tensor, at::redispatch::divide(ks, self, other));
  break;

case H_DIVIDE__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::divide_(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H_DIVIDE_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::divide_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_DIVIDE_SCALAR:
  set(op.tensor, at::redispatch::divide(ks, self, other));
  break;

case H_DIVIDE__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::divide_(ks, self, other)
  end_update_in_place(op.tensor);
  break;


case H_DIVIDE_TENSOR_MODE:
  set(op.tensor, at::redispatch::divide(ks, self, other, rounding_mode));
  break;

case H_DIVIDE__TENSOR_MODE:
  init_update_in_place(op.tensor);
  at::redispatch::divide_(ks, self, other, rounding_mode)
  end_update_in_place(op.tensor);
  break;

case H_DIVIDE_OUT_MODE:
  init_update_in_place(op.tensor);
  at::redispatch::divide_outf(ks, self, other, rounding_mode, out)
  end_update_in_place(op.tensor);
  break;


case H_DIVIDE_SCALAR_MODE:
  set(op.tensor, at::redispatch::divide(ks, self, other, rounding_mode));
  break;

case H_DIVIDE__SCALAR_MODE:
  init_update_in_place(op.tensor);
  at::redispatch::divide_(ks, self, other, rounding_mode)
  end_update_in_place(op.tensor);
  break;


case H_TRUE_DIVIDE_TENSOR:
  set(op.tensor, at::redispatch::true_divide(ks, self, other));
  break;

case H_TRUE_DIVIDE__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::true_divide_(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H_TRUE_DIVIDE_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::true_divide_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_TRUE_DIVIDE_SCALAR:
  set(op.tensor, at::redispatch::true_divide(ks, self, other));
  break;

case H_TRUE_DIVIDE__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::true_divide_(ks, self, other)
  end_update_in_place(op.tensor);
  break;


case H_DOT:
  set(op.tensor, at::redispatch::dot(ks, self, tensor));
  break;

case H_DOT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::dot_outf(ks, self, tensor, out)
  end_update_in_place(op.tensor);
  break;


case H_VDOT:
  set(op.tensor, at::redispatch::vdot(ks, self, other));
  break;

case H_VDOT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::vdot_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_EINSUM:
  set(op.tensor, at::redispatch::einsum(ks, equation, tensors));
  break;


case H_EMBEDDING:
  set(op.tensor, at::redispatch::embedding(ks, weight, indices, padding_idx, scale_grad_by_freq, sparse));
  break;


case H_EMBEDDING_BACKWARD:
  set(op.tensor, at::redispatch::embedding_backward(ks, grad, indices, num_weights, padding_idx, scale_grad_by_freq, sparse));
  break;


case H_EMBEDDING_DENSE_BACKWARD:
  set(op.tensor, at::redispatch::embedding_dense_backward(ks, grad_output, indices, num_weights, padding_idx, scale_grad_by_freq));
  break;

case H_EMBEDDING_RENORM_:
  init_update_in_place(op.tensor);
  at::redispatch::embedding_renorm_(ks, self, indices, max_norm, norm_type)
  end_update_in_place(op.tensor);
  break;


case H_EMBEDDING_SPARSE_BACKWARD:
  set(op.tensor, at::redispatch::embedding_sparse_backward(ks, grad, indices, num_weights, padding_idx, scale_grad_by_freq));
  break;




case H_ROW_STACK:
  set(op.tensor, at::redispatch::row_stack(ks, tensors));
  break;

case H_ROW_STACK_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::row_stack_outf(ks, tensors, out)
  end_update_in_place(op.tensor);
  break;





case H__EMBEDDING_BAG_BACKWARD:
  set(op.tensor, at::redispatch::_embedding_bag_backward(ks, grad, indices, offsets, offset2bag, bag_size, maximum_indices, num_weights, scale_grad_by_freq, mode, sparse, per_sample_weights, padding_idx));
  break;


case H__EMBEDDING_BAG_SPARSE_BACKWARD:
  set(op.tensor, at::redispatch::_embedding_bag_sparse_backward(ks, grad, indices, offsets, offset2bag, bag_size, num_weights, scale_grad_by_freq, mode, per_sample_weights, padding_idx));
  break;


case H__EMBEDDING_BAG_DENSE_BACKWARD:
  set(op.tensor, at::redispatch::_embedding_bag_dense_backward(ks, grad, indices, offset2bag, bag_size, maximum_indices, num_weights, scale_grad_by_freq, mode, per_sample_weights, padding_idx));
  break;


case H__EMBEDDING_BAG_PER_SAMPLE_WEIGHTS_BACKWARD:
  set(op.tensor, at::redispatch::_embedding_bag_per_sample_weights_backward(ks, grad, weight, indices, offsets, offset2bag, mode, padding_idx));
  break;


case H_EMPTY_NAMES:
  set(op.tensor, at::redispatch::empty(ks, size, names, dtype, layout, device, pin_memory, memory_format));
  break;


case H_EMPTY_MEMORY_FORMAT:
  set(op.tensor, at::redispatch::empty(ks, size, dtype, layout, device, pin_memory, memory_format));
  break;


case H_NEW_EMPTY:
  set(op.tensor, at::redispatch::new_empty(ks, self, size, dtype, layout, device, pin_memory));
  break;


case H_NEW_EMPTY_STRIDED:
  set(op.tensor, at::redispatch::new_empty_strided(ks, self, size, stride, dtype, layout, device, pin_memory));
  break;


case H_NEW_FULL:
  set(op.tensor, at::redispatch::new_full(ks, self, size, fill_value, dtype, layout, device, pin_memory));
  break;


case H_NEW_ZEROS:
  set(op.tensor, at::redispatch::new_zeros(ks, self, size, dtype, layout, device, pin_memory));
  break;


case H__EMPTY_AFFINE_QUANTIZED:
  set(op.tensor, at::redispatch::_empty_affine_quantized(ks, size, dtype, layout, device, pin_memory, scale, zero_point, memory_format));
  break;


case H__EMPTY_PER_CHANNEL_AFFINE_QUANTIZED:
  set(op.tensor, at::redispatch::_empty_per_channel_affine_quantized(ks, size, scales, zero_points, axis, dtype, layout, device, pin_memory, memory_format));
  break;



case H_EMPTY_QUANTIZED:
  set(op.tensor, at::redispatch::empty_quantized(ks, size, qtensor));
  break;

case H_EMPTY_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::empty_outf(ks, size, memory_format, out)
  end_update_in_place(op.tensor);
  break;


case H_EMPTY_LIKE:
  set(op.tensor, at::redispatch::empty_like(ks, self, dtype, layout, device, pin_memory, memory_format));
  break;


case H_EMPTY_STRIDED:
  set(op.tensor, at::redispatch::empty_strided(ks, size, stride, dtype, layout, device, pin_memory));
  break;

case H_ERF_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::erf_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;

case H_ERFC_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::erfc_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;

case H_EXP_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::exp_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;

case H_EXP2_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::exp2_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;

case H_EXPM1_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::expm1_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_EXPAND:
  set(op.tensor, at::redispatch::expand(ks, self, size, implicit));
  break;


case H_EXPAND_AS:
  set(op.tensor, at::redispatch::expand_as(ks, self, other));
  break;


case H_EYE:
  set(op.tensor, at::redispatch::eye(ks, n, dtype, layout, device, pin_memory));
  break;


case H_EYE_M:
  set(op.tensor, at::redispatch::eye(ks, n, m, dtype, layout, device, pin_memory));
  break;

case H_EYE_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::eye_outf(ks, n, out)
  end_update_in_place(op.tensor);
  break;

case H_EYE_M_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::eye_outf(ks, n, m, out)
  end_update_in_place(op.tensor);
  break;


case H_FLATTEN_USING_INTS:
  set(op.tensor, at::redispatch::flatten(ks, self, start_dim, end_dim));
  break;


case H_FLATTEN_NAMED_OUT_DIM:
  set(op.tensor, at::redispatch::flatten(ks, self, start_dim, end_dim, out_dim));
  break;


case H_FLATTEN_USING_NAMES:
  set(op.tensor, at::redispatch::flatten(ks, self, start_dim, end_dim, out_dim));
  break;


case H_FLATTEN_DIMNAMELIST:
  set(op.tensor, at::redispatch::flatten(ks, self, dims, out_dim));
  break;


case H_UNFLATTEN_INT:
  set(op.tensor, at::redispatch::unflatten(ks, self, dim, sizes, names));
  break;


case H_UNFLATTEN_DIMNAME:
  set(op.tensor, at::redispatch::unflatten(ks, self, dim, sizes, names));
  break;

case H_FILL__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::fill_(ks, self, value)
  end_update_in_place(op.tensor);
  break;

case H_FILL__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::fill_(ks, self, value)
  end_update_in_place(op.tensor);
  break;


case H_FLOOR:
  set(op.tensor, at::redispatch::floor(ks, self));
  break;

case H_FLOOR_:
  init_update_in_place(op.tensor);
  at::redispatch::floor_(ks, self)
  end_update_in_place(op.tensor);
  break;

case H_FLOOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::floor_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_FLOOR_DIVIDE:
  set(op.tensor, at::redispatch::floor_divide(ks, self, other));
  break;

case H_FLOOR_DIVIDE__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::floor_divide_(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H_FLOOR_DIVIDE_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::floor_divide_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_FLOOR_DIVIDE_SCALAR:
  set(op.tensor, at::redispatch::floor_divide(ks, self, other));
  break;

case H_FLOOR_DIVIDE__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::floor_divide_(ks, self, other)
  end_update_in_place(op.tensor);
  break;


case H_FRAC:
  set(op.tensor, at::redispatch::frac(ks, self));
  break;

case H_FRAC_:
  init_update_in_place(op.tensor);
  at::redispatch::frac_(ks, self)
  end_update_in_place(op.tensor);
  break;

case H_FRAC_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::frac_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_FULL_NAMES:
  set(op.tensor, at::redispatch::full(ks, size, fill_value, names, dtype, layout, device, pin_memory));
  break;


case H_FULL:
  set(op.tensor, at::redispatch::full(ks, size, fill_value, dtype, layout, device, pin_memory));
  break;

case H_FULL_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::full_outf(ks, size, fill_value, out)
  end_update_in_place(op.tensor);
  break;


case H_FULL_LIKE:
  set(op.tensor, at::redispatch::full_like(ks, self, fill_value, dtype, layout, device, pin_memory, memory_format));
  break;


case H_FROM_FILE:
  set(op.tensor, at::redispatch::from_file(ks, filename, shared, size, dtype, layout, device, pin_memory));
  break;

case H_GCD_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::gcd_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_GCD:
  set(op.tensor, at::redispatch::gcd(ks, self, other));
  break;

case H_GCD_:
  init_update_in_place(op.tensor);
  at::redispatch::gcd_(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H_LCM_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::lcm_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_LCM:
  set(op.tensor, at::redispatch::lcm(ks, self, other));
  break;

case H_LCM_:
  init_update_in_place(op.tensor);
  at::redispatch::lcm_(ks, self, other)
  end_update_in_place(op.tensor);
  break;


case H_GRID_SAMPLER:
  set(op.tensor, at::redispatch::grid_sampler(ks, input, grid, interpolation_mode, padding_mode, align_corners));
  break;


case H_GRID_SAMPLER_2D:
  set(op.tensor, at::redispatch::grid_sampler_2d(ks, input, grid, interpolation_mode, padding_mode, align_corners));
  break;



case H__GRID_SAMPLER_2D_CPU_FALLBACK:
  set(op.tensor, at::redispatch::_grid_sampler_2d_cpu_fallback(ks, input, grid, interpolation_mode, padding_mode, align_corners));
  break;



case H_GRID_SAMPLER_3D:
  set(op.tensor, at::redispatch::grid_sampler_3d(ks, input, grid, interpolation_mode, padding_mode, align_corners));
  break;



case H_HANN_WINDOW:
  set(op.tensor, at::redispatch::hann_window(ks, window_length, dtype, layout, device, pin_memory));
  break;


case H_HANN_WINDOW_PERIODIC:
  set(op.tensor, at::redispatch::hann_window(ks, window_length, periodic, dtype, layout, device, pin_memory));
  break;


case H_HAMMING_WINDOW:
  set(op.tensor, at::redispatch::hamming_window(ks, window_length, dtype, layout, device, pin_memory));
  break;


case H_HAMMING_WINDOW_PERIODIC:
  set(op.tensor, at::redispatch::hamming_window(ks, window_length, periodic, dtype, layout, device, pin_memory));
  break;


case H_HAMMING_WINDOW_PERIODIC_ALPHA:
  set(op.tensor, at::redispatch::hamming_window(ks, window_length, periodic, alpha, dtype, layout, device, pin_memory));
  break;


case H_HAMMING_WINDOW_PERIODIC_ALPHA_BETA:
  set(op.tensor, at::redispatch::hamming_window(ks, window_length, periodic, alpha, beta, dtype, layout, device, pin_memory));
  break;


case H_KAISER_WINDOW:
  set(op.tensor, at::redispatch::kaiser_window(ks, window_length, dtype, layout, device, pin_memory));
  break;


case H_KAISER_WINDOW_PERIODIC:
  set(op.tensor, at::redispatch::kaiser_window(ks, window_length, periodic, dtype, layout, device, pin_memory));
  break;


case H_KAISER_WINDOW_BETA:
  set(op.tensor, at::redispatch::kaiser_window(ks, window_length, periodic, beta, dtype, layout, device, pin_memory));
  break;


case H_HINGE_EMBEDDING_LOSS:
  set(op.tensor, at::redispatch::hinge_embedding_loss(ks, self, target, margin, reduction));
  break;


case H_GROUP_NORM:
  set(op.tensor, at::redispatch::group_norm(ks, input, num_groups, weight, bias, eps, cudnn_enabled));
  break;




case H__FFT_R2C:
  set(op.tensor, at::redispatch::_fft_r2c(ks, self, dim, normalization, onesided));
  break;

case H__FFT_R2C_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::_fft_r2c_outf(ks, self, dim, normalization, onesided, out)
  end_update_in_place(op.tensor);
  break;


case H__FFT_C2R:
  set(op.tensor, at::redispatch::_fft_c2r(ks, self, dim, normalization, last_dim_size));
  break;

case H__FFT_C2R_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::_fft_c2r_outf(ks, self, dim, normalization, last_dim_size, out)
  end_update_in_place(op.tensor);
  break;


case H__FFT_C2C:
  set(op.tensor, at::redispatch::_fft_c2c(ks, self, dim, normalization, forward));
  break;

case H__FFT_C2C_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::_fft_c2c_outf(ks, self, dim, normalization, forward, out)
  end_update_in_place(op.tensor);
  break;






case H_INDEX_TENSOR:
  set(op.tensor, at::redispatch::index(ks, self, indices));
  break;

case H_INDEX_COPY_:
  init_update_in_place(op.tensor);
  at::redispatch::index_copy_(ks, self, dim, index, source)
  end_update_in_place(op.tensor);
  break;


case H_INDEX_COPY:
  set(op.tensor, at::redispatch::index_copy(ks, self, dim, index, source));
  break;

case H_INDEX_COPY__DIMNAME:
  init_update_in_place(op.tensor);
  at::redispatch::index_copy_(ks, self, dim, index, source)
  end_update_in_place(op.tensor);
  break;


case H_INDEX_COPY_DIMNAME:
  set(op.tensor, at::redispatch::index_copy(ks, self, dim, index, source));
  break;

case H_INDEX_PUT_:
  init_update_in_place(op.tensor);
  at::redispatch::index_put_(ks, self, indices, values, accumulate)
  end_update_in_place(op.tensor);
  break;


case H_INDEX_PUT:
  set(op.tensor, at::redispatch::index_put(ks, self, indices, values, accumulate));
  break;

case H__INDEX_PUT_IMPL_:
  init_update_in_place(op.tensor);
  at::redispatch::_index_put_impl_(ks, self, indices, values, accumulate, unsafe)
  end_update_in_place(op.tensor);
  break;


case H_INSTANCE_NORM:
  set(op.tensor, at::redispatch::instance_norm(ks, input, weight, bias, running_mean, running_var, use_input_stats, momentum, eps, cudnn_enabled));
  break;


case H_INVERSE:
  set(op.tensor, at::redispatch::inverse(ks, self));
  break;

case H_INVERSE_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::inverse_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H__INVERSE_HELPER:
  set(op.tensor, at::redispatch::_inverse_helper(ks, self));
  break;


case H_ISCLOSE:
  set(op.tensor, at::redispatch::isclose(ks, self, other, rtol, atol, equal_nan));
  break;


case H_ISNAN:
  set(op.tensor, at::redispatch::isnan(ks, self));
  break;



case H_ISREAL:
  set(op.tensor, at::redispatch::isreal(ks, self));
  break;




case H_KL_DIV:
  set(op.tensor, at::redispatch::kl_div(ks, self, target, reduction, log_target));
  break;


case H_KL_DIV_BACKWARD:
  set(op.tensor, at::redispatch::kl_div_backward(ks, grad_output, self, target, reduction, log_target));
  break;


case H_KRON:
  set(op.tensor, at::redispatch::kron(ks, self, other));
  break;

case H_KRON_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::kron_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;






case H_LAYER_NORM:
  set(op.tensor, at::redispatch::layer_norm(ks, input, normalized_shape, weight, bias, eps, cudnn_enable));
  break;




case H_NAN_TO_NUM:
  set(op.tensor, at::redispatch::nan_to_num(ks, self, nan, posinf, neginf));
  break;

case H_NAN_TO_NUM_:
  init_update_in_place(op.tensor);
  at::redispatch::nan_to_num_(ks, self, nan, posinf, neginf)
  end_update_in_place(op.tensor);
  break;

case H_NAN_TO_NUM_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::nan_to_num_outf(ks, self, nan, posinf, neginf, out)
  end_update_in_place(op.tensor);
  break;


case H_LINEAR:
  set(op.tensor, at::redispatch::linear(ks, input, weight, bias));
  break;


case H_MKLDNN_LINEAR:
  set(op.tensor, at::redispatch::mkldnn_linear(ks, self, weight, bias));
  break;


case H_MKLDNN_LINEAR_BACKWARD_INPUT:
  set(op.tensor, at::redispatch::mkldnn_linear_backward_input(ks, input_size, grad_output, weight));
  break;




case H_FBGEMM_LINEAR_INT8_WEIGHT_FP32_ACTIVATION:
  set(op.tensor, at::redispatch::fbgemm_linear_int8_weight_fp32_activation(ks, input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias));
  break;


case H_FBGEMM_LINEAR_INT8_WEIGHT:
  set(op.tensor, at::redispatch::fbgemm_linear_int8_weight(ks, input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias));
  break;



case H_FBGEMM_PACK_GEMM_MATRIX_FP16:
  set(op.tensor, at::redispatch::fbgemm_pack_gemm_matrix_fp16(ks, input));
  break;


case H_FBGEMM_LINEAR_FP16_WEIGHT_FP32_ACTIVATION:
  set(op.tensor, at::redispatch::fbgemm_linear_fp16_weight_fp32_activation(ks, input, packed_weight, bias));
  break;


case H_FBGEMM_LINEAR_FP16_WEIGHT:
  set(op.tensor, at::redispatch::fbgemm_linear_fp16_weight(ks, input, packed_weight, bias));
  break;


case H_FBGEMM_PACK_QUANTIZED_MATRIX:
  set(op.tensor, at::redispatch::fbgemm_pack_quantized_matrix(ks, input));
  break;


case H_FBGEMM_PACK_QUANTIZED_MATRIX_KN:
  set(op.tensor, at::redispatch::fbgemm_pack_quantized_matrix(ks, input, K, N));
  break;


case H_LDEXP_TENSOR:
  set(op.tensor, at::redispatch::ldexp(ks, self, other));
  break;

case H_LDEXP_:
  init_update_in_place(op.tensor);
  at::redispatch::ldexp_(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H_LDEXP_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::ldexp_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_LINSPACE:
  set(op.tensor, at::redispatch::linspace(ks, start, end, steps, dtype, layout, device, pin_memory));
  break;

case H_LINSPACE_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::linspace_outf(ks, start, end, steps, out)
  end_update_in_place(op.tensor);
  break;

case H_LOG_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::log_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;

case H_LOG10_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::log10_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_LOG1P:
  set(op.tensor, at::redispatch::log1p(ks, self));
  break;

case H_LOG1P_:
  init_update_in_place(op.tensor);
  at::redispatch::log1p_(ks, self)
  end_update_in_place(op.tensor);
  break;

case H_LOG1P_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::log1p_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;

case H_LOG2_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::log2_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;

case H_LOGADDEXP_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::logaddexp_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_LOGADDEXP:
  set(op.tensor, at::redispatch::logaddexp(ks, self, other));
  break;

case H_LOGADDEXP2_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::logaddexp2_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_LOGADDEXP2:
  set(op.tensor, at::redispatch::logaddexp2(ks, self, other));
  break;


case H_XLOGY_TENSOR:
  set(op.tensor, at::redispatch::xlogy(ks, self, other));
  break;


case H_XLOGY_SCALAR_SELF:
  set(op.tensor, at::redispatch::xlogy(ks, self, other));
  break;


case H_XLOGY_SCALAR_OTHER:
  set(op.tensor, at::redispatch::xlogy(ks, self, other));
  break;

case H_XLOGY__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::xlogy_(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H_XLOGY__SCALAR_OTHER:
  init_update_in_place(op.tensor);
  at::redispatch::xlogy_(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H_XLOGY_OUTTENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::xlogy_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;

case H_XLOGY_OUTSCALAR_SELF:
  init_update_in_place(op.tensor);
  at::redispatch::xlogy_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;

case H_XLOGY_OUTSCALAR_OTHER:
  init_update_in_place(op.tensor);
  at::redispatch::xlogy_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_LOGDET:
  set(op.tensor, at::redispatch::logdet(ks, self));
  break;


case H_LOGSPACE:
  set(op.tensor, at::redispatch::logspace(ks, start, end, steps, base, dtype, layout, device, pin_memory));
  break;

case H_LOGSPACE_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::logspace_outf(ks, start, end, steps, base, out)
  end_update_in_place(op.tensor);
  break;


case H_LOG_SOFTMAX_INT:
  set(op.tensor, at::redispatch::log_softmax(ks, self, dim, dtype));
  break;


case H_LOG_SOFTMAX_DIMNAME:
  set(op.tensor, at::redispatch::log_softmax(ks, self, dim, dtype));
  break;


case H__LOG_SOFTMAX:
  set(op.tensor, at::redispatch::_log_softmax(ks, self, dim, half_to_float));
  break;


case H__LOG_SOFTMAX_BACKWARD_DATA:
  set(op.tensor, at::redispatch::_log_softmax_backward_data(ks, grad_output, output, dim, self));
  break;


case H__LOGCUMSUMEXP:
  set(op.tensor, at::redispatch::_logcumsumexp(ks, self, dim));
  break;

case H__LOGCUMSUMEXP_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::_logcumsumexp_outf(ks, self, dim, out)
  end_update_in_place(op.tensor);
  break;


case H_LOGCUMSUMEXP:
  set(op.tensor, at::redispatch::logcumsumexp(ks, self, dim));
  break;

case H_LOGCUMSUMEXP_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::logcumsumexp_outf(ks, self, dim, out)
  end_update_in_place(op.tensor);
  break;


case H_LOGCUMSUMEXP_DIMNAME:
  set(op.tensor, at::redispatch::logcumsumexp(ks, self, dim));
  break;

case H_LOGCUMSUMEXP_DIMNAME_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::logcumsumexp_outf(ks, self, dim, out)
  end_update_in_place(op.tensor);
  break;


case H_LOGSUMEXP:
  set(op.tensor, at::redispatch::logsumexp(ks, self, dim, keepdim));
  break;

case H_LOGSUMEXP_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::logsumexp_outf(ks, self, dim, keepdim, out)
  end_update_in_place(op.tensor);
  break;


case H_LOGSUMEXP_NAMES:
  set(op.tensor, at::redispatch::logsumexp(ks, self, dim, keepdim));
  break;

case H_LOGSUMEXP_NAMES_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::logsumexp_outf(ks, self, dim, keepdim, out)
  end_update_in_place(op.tensor);
  break;


case H_MARGIN_RANKING_LOSS:
  set(op.tensor, at::redispatch::margin_ranking_loss(ks, input1, input2, target, margin, reduction));
  break;


case H_MATMUL:
  set(op.tensor, at::redispatch::matmul(ks, self, other));
  break;

case H_MATMUL_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::matmul_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_MATRIX_RANK_TOL:
  set(op.tensor, at::redispatch::matrix_rank(ks, self, tol, symmetric));
  break;


case H_MATRIX_RANK:
  set(op.tensor, at::redispatch::matrix_rank(ks, self, symmetric));
  break;


case H_MATRIX_POWER:
  set(op.tensor, at::redispatch::matrix_power(ks, self, n));
  break;

case H_MATRIX_POWER_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::matrix_power_outf(ks, self, n, out)
  end_update_in_place(op.tensor);
  break;


case H_MATRIX_EXP:
  set(op.tensor, at::redispatch::matrix_exp(ks, self));
  break;


case H_MATRIX_EXP_BACKWARD:
  set(op.tensor, at::redispatch::matrix_exp_backward(ks, self, grad));
  break;




case H__COMPUTE_LINEAR_COMBINATION:
  set(op.tensor, at::redispatch::_compute_linear_combination(ks, input, coefficients));
  break;

case H__COMPUTE_LINEAR_COMBINATION_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::_compute_linear_combination_outf(ks, input, coefficients, out)
  end_update_in_place(op.tensor);
  break;






case H_VALUE_SELECTING_REDUCTION_BACKWARD:
  set(op.tensor, at::redispatch::value_selecting_reduction_backward(ks, grad, dim, indices, sizes, keepdim));
  break;


case H_AMAX:
  set(op.tensor, at::redispatch::amax(ks, self, dim, keepdim));
  break;

case H_AMAX_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::amax_outf(ks, self, dim, keepdim, out)
  end_update_in_place(op.tensor);
  break;



case H_MAX_POOL1D:
  set(op.tensor, at::redispatch::max_pool1d(ks, self, kernel_size, stride, padding, dilation, ceil_mode));
  break;


case H_MAX_POOL2D:
  set(op.tensor, at::redispatch::max_pool2d(ks, self, kernel_size, stride, padding, dilation, ceil_mode));
  break;


case H_MKLDNN_MAX_POOL2D:
  set(op.tensor, at::redispatch::mkldnn_max_pool2d(ks, self, kernel_size, stride, padding, dilation, ceil_mode));
  break;


case H_MKLDNN_MAX_POOL2D_BACKWARD:
  set(op.tensor, at::redispatch::mkldnn_max_pool2d_backward(ks, grad_output, output, input, kernel_size, stride, padding, dilation, ceil_mode));
  break;


case H_MKLDNN_MAX_POOL3D:
  set(op.tensor, at::redispatch::mkldnn_max_pool3d(ks, self, kernel_size, stride, padding, dilation, ceil_mode));
  break;


case H_MKLDNN_MAX_POOL3D_BACKWARD:
  set(op.tensor, at::redispatch::mkldnn_max_pool3d_backward(ks, grad_output, output, input, kernel_size, stride, padding, dilation, ceil_mode));
  break;


case H_QUANTIZED_MAX_POOL1D:
  set(op.tensor, at::redispatch::quantized_max_pool1d(ks, self, kernel_size, stride, padding, dilation, ceil_mode));
  break;


case H_QUANTIZED_MAX_POOL2D:
  set(op.tensor, at::redispatch::quantized_max_pool2d(ks, self, kernel_size, stride, padding, dilation, ceil_mode));
  break;


case H_MAX_POOL3D:
  set(op.tensor, at::redispatch::max_pool3d(ks, self, kernel_size, stride, padding, dilation, ceil_mode));
  break;


case H_MEAN:
  set(op.tensor, at::redispatch::mean(ks, self, dtype));
  break;


case H_MEAN_DIM:
  set(op.tensor, at::redispatch::mean(ks, self, dim, keepdim, dtype));
  break;

case H_MEAN_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::mean_outf(ks, self, dim, keepdim, dtype, out)
  end_update_in_place(op.tensor);
  break;


case H_MEAN_NAMES_DIM:
  set(op.tensor, at::redispatch::mean(ks, self, dim, keepdim, dtype));
  break;

case H_MEAN_NAMES_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::mean_outf(ks, self, dim, keepdim, dtype, out)
  end_update_in_place(op.tensor);
  break;


case H_MEDIAN:
  set(op.tensor, at::redispatch::median(ks, self));
  break;






case H_NANMEDIAN:
  set(op.tensor, at::redispatch::nanmedian(ks, self));
  break;










case H_AMIN:
  set(op.tensor, at::redispatch::amin(ks, self, dim, keepdim));
  break;

case H_AMIN_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::amin_outf(ks, self, dim, keepdim, out)
  end_update_in_place(op.tensor);
  break;


case H_MKLDNN_CONVOLUTION:
  set(op.tensor, at::redispatch::mkldnn_convolution(ks, self, weight, bias, padding, stride, dilation, groups));
  break;


case H_MKLDNN_CONVOLUTION_BACKWARD_INPUT:
  set(op.tensor, at::redispatch::mkldnn_convolution_backward_input(ks, self_size, grad_output, weight, padding, stride, dilation, groups, bias_defined));
  break;






case H_MIOPEN_CONVOLUTION:
  set(op.tensor, at::redispatch::miopen_convolution(ks, self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic));
  break;


case H_MIOPEN_CONVOLUTION_BACKWARD_INPUT:
  set(op.tensor, at::redispatch::miopen_convolution_backward_input(ks, self_size, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic));
  break;



case H_MIOPEN_CONVOLUTION_BACKWARD_BIAS:
  set(op.tensor, at::redispatch::miopen_convolution_backward_bias(ks, grad_output));
  break;


case H_MIOPEN_CONVOLUTION_BACKWARD_WEIGHT:
  set(op.tensor, at::redispatch::miopen_convolution_backward_weight(ks, weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic));
  break;


case H_MIOPEN_CONVOLUTION_TRANSPOSE:
  set(op.tensor, at::redispatch::miopen_convolution_transpose(ks, self, weight, bias, padding, output_padding, stride, dilation, groups, benchmark, deterministic));
  break;



case H_MIOPEN_CONVOLUTION_TRANSPOSE_BACKWARD_INPUT:
  set(op.tensor, at::redispatch::miopen_convolution_transpose_backward_input(ks, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic));
  break;


case H_MIOPEN_CONVOLUTION_TRANSPOSE_BACKWARD_WEIGHT:
  set(op.tensor, at::redispatch::miopen_convolution_transpose_backward_weight(ks, weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic));
  break;


case H_MIOPEN_DEPTHWISE_CONVOLUTION:
  set(op.tensor, at::redispatch::miopen_depthwise_convolution(ks, self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic));
  break;


case H_MIOPEN_DEPTHWISE_CONVOLUTION_BACKWARD_INPUT:
  set(op.tensor, at::redispatch::miopen_depthwise_convolution_backward_input(ks, self_size, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic));
  break;



case H_MIOPEN_DEPTHWISE_CONVOLUTION_BACKWARD_WEIGHT:
  set(op.tensor, at::redispatch::miopen_depthwise_convolution_backward_weight(ks, weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic));
  break;




case H_MM:
  set(op.tensor, at::redispatch::mm(ks, self, mat2));
  break;

case H_MM_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::mm_outf(ks, self, mat2, out)
  end_update_in_place(op.tensor);
  break;


case H__SPARSE_MM:
  set(op.tensor, at::redispatch::_sparse_mm(ks, sparse, dense));
  break;


case H__SPARSE_SPARSE_MATMUL:
  set(op.tensor, at::redispatch::_sparse_sparse_matmul(ks, self, other));
  break;


case H__SPARSE_MASK_HELPER:
  set(op.tensor, at::redispatch::_sparse_mask_helper(ks, t, mask_indices));
  break;






case H_MUL_TENSOR:
  set(op.tensor, at::redispatch::mul(ks, self, other));
  break;

case H_MUL__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::mul_(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H_MUL_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::mul_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_MUL_SCALAR:
  set(op.tensor, at::redispatch::mul(ks, self, other));
  break;

case H_MUL__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::mul_(ks, self, other)
  end_update_in_place(op.tensor);
  break;


case H_MULTIPLY_TENSOR:
  set(op.tensor, at::redispatch::multiply(ks, self, other));
  break;

case H_MULTIPLY__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::multiply_(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H_MULTIPLY_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::multiply_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_MULTIPLY_SCALAR:
  set(op.tensor, at::redispatch::multiply(ks, self, other));
  break;

case H_MULTIPLY__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::multiply_(ks, self, other)
  end_update_in_place(op.tensor);
  break;


case H_MV:
  set(op.tensor, at::redispatch::mv(ks, self, vec));
  break;

case H_MV_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::mv_outf(ks, self, vec, out)
  end_update_in_place(op.tensor);
  break;


case H_MVLGAMMA:
  set(op.tensor, at::redispatch::mvlgamma(ks, self, p));
  break;

case H_MVLGAMMA_:
  init_update_in_place(op.tensor);
  at::redispatch::mvlgamma_(ks, self, p)
  end_update_in_place(op.tensor);
  break;


case H_NARROW_COPY:
  set(op.tensor, at::redispatch::narrow_copy(ks, self, dim, start, length));
  break;

case H_NARROW_COPY_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::narrow_copy_outf(ks, self, dim, start, length, out)
  end_update_in_place(op.tensor);
  break;


case H_NARROW:
  set(op.tensor, at::redispatch::narrow(ks, self, dim, start, length));
  break;


case H_NARROW_TENSOR:
  set(op.tensor, at::redispatch::narrow(ks, self, dim, start, length));
  break;





case H_BATCH_NORM_ELEMT:
  set(op.tensor, at::redispatch::batch_norm_elemt(ks, input, weight, bias, mean, invstd, eps));
  break;

case H_BATCH_NORM_ELEMT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::batch_norm_elemt_outf(ks, input, weight, bias, mean, invstd, eps, out)
  end_update_in_place(op.tensor);
  break;






case H_BATCH_NORM_BACKWARD_ELEMT:
  set(op.tensor, at::redispatch::batch_norm_backward_elemt(ks, grad_out, input, mean, invstd, weight, mean_dy, mean_dy_xmu, count));
  break;





case H__NNPACK_SPATIAL_CONVOLUTION:
  set(op.tensor, at::redispatch::_nnpack_spatial_convolution(ks, input, weight, bias, padding, stride));
  break;



case H__NNPACK_SPATIAL_CONVOLUTION_BACKWARD_INPUT:
  set(op.tensor, at::redispatch::_nnpack_spatial_convolution_backward_input(ks, input, grad_output, weight, padding));
  break;


case H__NNPACK_SPATIAL_CONVOLUTION_BACKWARD_WEIGHT:
  set(op.tensor, at::redispatch::_nnpack_spatial_convolution_backward_weight(ks, input, weightsize, grad_output, padding));
  break;


case H_ONES_NAMES:
  set(op.tensor, at::redispatch::ones(ks, size, names, dtype, layout, device, pin_memory));
  break;


case H_ONES:
  set(op.tensor, at::redispatch::ones(ks, size, dtype, layout, device, pin_memory));
  break;

case H_ONES_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::ones_outf(ks, size, out)
  end_update_in_place(op.tensor);
  break;


case H_ONES_LIKE:
  set(op.tensor, at::redispatch::ones_like(ks, self, dtype, layout, device, pin_memory, memory_format));
  break;


case H_PAIRWISE_DISTANCE:
  set(op.tensor, at::redispatch::pairwise_distance(ks, x1, x2, p, eps, keepdim));
  break;


case H_CDIST:
  set(op.tensor, at::redispatch::cdist(ks, x1, x2, p, compute_mode));
  break;


case H__EUCLIDEAN_DIST:
  set(op.tensor, at::redispatch::_euclidean_dist(ks, x1, x2));
  break;


case H__CDIST_FORWARD:
  set(op.tensor, at::redispatch::_cdist_forward(ks, x1, x2, p, compute_mode));
  break;


case H__CDIST_BACKWARD:
  set(op.tensor, at::redispatch::_cdist_backward(ks, grad, x1, x2, p, cdist));
  break;


case H_PDIST:
  set(op.tensor, at::redispatch::pdist(ks, self, p));
  break;


case H__PDIST_FORWARD:
  set(op.tensor, at::redispatch::_pdist_forward(ks, self, p));
  break;


case H__PDIST_BACKWARD:
  set(op.tensor, at::redispatch::_pdist_backward(ks, grad, self, p, pdist));
  break;


case H_COSINE_SIMILARITY:
  set(op.tensor, at::redispatch::cosine_similarity(ks, x1, x2, dim, eps));
  break;


case H_PERMUTE:
  set(op.tensor, at::redispatch::permute(ks, self, dims));
  break;


case H_MOVEDIM_INTLIST:
  set(op.tensor, at::redispatch::movedim(ks, self, source, destination));
  break;


case H_MOVEDIM_INT:
  set(op.tensor, at::redispatch::movedim(ks, self, source, destination));
  break;


case H_MOVEAXIS_INTLIST:
  set(op.tensor, at::redispatch::moveaxis(ks, self, source, destination));
  break;


case H_MOVEAXIS_INT:
  set(op.tensor, at::redispatch::moveaxis(ks, self, source, destination));
  break;


case H_NUMPY_T:
  set(op.tensor, at::redispatch::numpy_T(ks, self));
  break;


case H_PIXEL_SHUFFLE:
  set(op.tensor, at::redispatch::pixel_shuffle(ks, self, upscale_factor));
  break;


case H_PIXEL_UNSHUFFLE:
  set(op.tensor, at::redispatch::pixel_unshuffle(ks, self, downscale_factor));
  break;


case H_CHANNEL_SHUFFLE:
  set(op.tensor, at::redispatch::channel_shuffle(ks, self, groups));
  break;



case H_PIN_MEMORY:
  set(op.tensor, at::redispatch::pin_memory(ks, self));
  break;


case H_PINVERSE:
  set(op.tensor, at::redispatch::pinverse(ks, self, rcond));
  break;


case H_POISSON_NLL_LOSS:
  set(op.tensor, at::redispatch::poisson_nll_loss(ks, input, target, log_input, full, eps, reduction));
  break;


case H_RAD2DEG:
  set(op.tensor, at::redispatch::rad2deg(ks, self));
  break;

case H_RAD2DEG_:
  init_update_in_place(op.tensor);
  at::redispatch::rad2deg_(ks, self)
  end_update_in_place(op.tensor);
  break;

case H_RAD2DEG_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::rad2deg_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_DEG2RAD:
  set(op.tensor, at::redispatch::deg2rad(ks, self));
  break;

case H_DEG2RAD_:
  init_update_in_place(op.tensor);
  at::redispatch::deg2rad_(ks, self)
  end_update_in_place(op.tensor);
  break;

case H_DEG2RAD_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::deg2rad_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_SCALAR_TENSOR:
  set(op.tensor, at::redispatch::scalar_tensor(ks, s, dtype, layout, device, pin_memory));
  break;


case H_RAND_NAMES:
  set(op.tensor, at::redispatch::rand(ks, size, names, dtype, layout, device, pin_memory));
  break;


case H_RAND_GENERATOR_WITH_NAMES:
  set(op.tensor, at::redispatch::rand(ks, size, generator, names, dtype, layout, device, pin_memory));
  break;


case H_RAND:
  set(op.tensor, at::redispatch::rand(ks, size, dtype, layout, device, pin_memory));
  break;


case H_RAND_GENERATOR:
  set(op.tensor, at::redispatch::rand(ks, size, generator, dtype, layout, device, pin_memory));
  break;

case H_RAND_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::rand_outf(ks, size, out)
  end_update_in_place(op.tensor);
  break;

case H_RAND_GENERATOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::rand_outf(ks, size, generator, out)
  end_update_in_place(op.tensor);
  break;


case H_RAND_LIKE:
  set(op.tensor, at::redispatch::rand_like(ks, self, dtype, layout, device, pin_memory, memory_format));
  break;


case H_RANDINT:
  set(op.tensor, at::redispatch::randint(ks, high, size, dtype, layout, device, pin_memory));
  break;


case H_RANDINT_GENERATOR:
  set(op.tensor, at::redispatch::randint(ks, high, size, generator, dtype, layout, device, pin_memory));
  break;


case H_RANDINT_LOW:
  set(op.tensor, at::redispatch::randint(ks, low, high, size, dtype, layout, device, pin_memory));
  break;


case H_RANDINT_LOW_GENERATOR:
  set(op.tensor, at::redispatch::randint(ks, low, high, size, generator, dtype, layout, device, pin_memory));
  break;

case H_RANDINT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::randint_outf(ks, high, size, out)
  end_update_in_place(op.tensor);
  break;

case H_RANDINT_GENERATOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::randint_outf(ks, high, size, generator, out)
  end_update_in_place(op.tensor);
  break;

case H_RANDINT_LOW_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::randint_outf(ks, low, high, size, out)
  end_update_in_place(op.tensor);
  break;

case H_RANDINT_LOW_GENERATOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::randint_outf(ks, low, high, size, generator, out)
  end_update_in_place(op.tensor);
  break;


case H_RANDINT_LIKE:
  set(op.tensor, at::redispatch::randint_like(ks, self, high, dtype, layout, device, pin_memory, memory_format));
  break;


case H_RANDINT_LIKE_LOW_DTYPE:
  set(op.tensor, at::redispatch::randint_like(ks, self, low, high, dtype, layout, device, pin_memory, memory_format));
  break;


case H_RANDN:
  set(op.tensor, at::redispatch::randn(ks, size, dtype, layout, device, pin_memory));
  break;


case H_RANDN_GENERATOR:
  set(op.tensor, at::redispatch::randn(ks, size, generator, dtype, layout, device, pin_memory));
  break;


case H_RANDN_NAMES:
  set(op.tensor, at::redispatch::randn(ks, size, names, dtype, layout, device, pin_memory));
  break;


case H_RANDN_GENERATOR_WITH_NAMES:
  set(op.tensor, at::redispatch::randn(ks, size, generator, names, dtype, layout, device, pin_memory));
  break;

case H_RANDN_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::randn_outf(ks, size, out)
  end_update_in_place(op.tensor);
  break;

case H_RANDN_GENERATOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::randn_outf(ks, size, generator, out)
  end_update_in_place(op.tensor);
  break;


case H_RANDN_LIKE:
  set(op.tensor, at::redispatch::randn_like(ks, self, dtype, layout, device, pin_memory, memory_format));
  break;


case H_RANDPERM:
  set(op.tensor, at::redispatch::randperm(ks, n, dtype, layout, device, pin_memory));
  break;


case H_RANDPERM_GENERATOR:
  set(op.tensor, at::redispatch::randperm(ks, n, generator, dtype, layout, device, pin_memory));
  break;

case H_RANDPERM_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::randperm_outf(ks, n, out)
  end_update_in_place(op.tensor);
  break;

case H_RANDPERM_GENERATOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::randperm_outf(ks, n, generator, out)
  end_update_in_place(op.tensor);
  break;


case H_RANGE_STEP:
  set(op.tensor, at::redispatch::range(ks, start, end, step, dtype, layout, device, pin_memory));
  break;


case H_RANGE:
  set(op.tensor, at::redispatch::range(ks, start, end, dtype, layout, device, pin_memory));
  break;

case H_RANGE_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::range_outf(ks, start, end, step, out)
  end_update_in_place(op.tensor);
  break;


case H_RAVEL:
  set(op.tensor, at::redispatch::ravel(ks, self));
  break;

case H_RECIPROCAL_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::reciprocal_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_NEG:
  set(op.tensor, at::redispatch::neg(ks, self));
  break;

case H_NEG_:
  init_update_in_place(op.tensor);
  at::redispatch::neg_(ks, self)
  end_update_in_place(op.tensor);
  break;

case H_NEG_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::neg_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_NEGATIVE:
  set(op.tensor, at::redispatch::negative(ks, self));
  break;

case H_NEGATIVE_:
  init_update_in_place(op.tensor);
  at::redispatch::negative_(ks, self)
  end_update_in_place(op.tensor);
  break;

case H_NEGATIVE_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::negative_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_REPEAT:
  set(op.tensor, at::redispatch::repeat(ks, self, repeats));
  break;


case H_REPEAT_INTERLEAVE_TENSOR:
  set(op.tensor, at::redispatch::repeat_interleave(ks, repeats));
  break;


case H_REPEAT_INTERLEAVE_SELF_TENSOR:
  set(op.tensor, at::redispatch::repeat_interleave(ks, self, repeats, dim));
  break;


case H_REPEAT_INTERLEAVE_SELF_INT:
  set(op.tensor, at::redispatch::repeat_interleave(ks, self, repeats, dim));
  break;


case H_RESHAPE:
  set(op.tensor, at::redispatch::reshape(ks, self, shape));
  break;


case H__MKLDNN_RESHAPE:
  set(op.tensor, at::redispatch::_mkldnn_reshape(ks, self, shape));
  break;


case H_RESHAPE_AS:
  set(op.tensor, at::redispatch::reshape_as(ks, self, other));
  break;


case H_ROUND:
  set(op.tensor, at::redispatch::round(ks, self));
  break;

case H_ROUND_:
  init_update_in_place(op.tensor);
  at::redispatch::round_(ks, self)
  end_update_in_place(op.tensor);
  break;

case H_ROUND_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::round_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_RRELU:
  set(op.tensor, at::redispatch::rrelu(ks, self, lower, upper, training, generator));
  break;

case H_RRELU_:
  init_update_in_place(op.tensor);
  at::redispatch::rrelu_(ks, self, lower, upper, training, generator)
  end_update_in_place(op.tensor);
  break;


case H_RELU:
  set(op.tensor, at::redispatch::relu(ks, self));
  break;

case H_RELU_:
  init_update_in_place(op.tensor);
  at::redispatch::relu_(ks, self)
  end_update_in_place(op.tensor);
  break;


case H_RELU6:
  set(op.tensor, at::redispatch::relu6(ks, self));
  break;

case H_RELU6_:
  init_update_in_place(op.tensor);
  at::redispatch::relu6_(ks, self)
  end_update_in_place(op.tensor);
  break;


case H_PRELU:
  set(op.tensor, at::redispatch::prelu(ks, self, weight));
  break;



case H_GELU:
  set(op.tensor, at::redispatch::gelu(ks, self));
  break;


case H_GELU_BACKWARD:
  set(op.tensor, at::redispatch::gelu_backward(ks, grad, self));
  break;


case H_INFINITELY_DIFFERENTIABLE_GELU_BACKWARD:
  set(op.tensor, at::redispatch::infinitely_differentiable_gelu_backward(ks, grad, self));
  break;


case H_HARDSHRINK:
  set(op.tensor, at::redispatch::hardshrink(ks, self, lambd));
  break;


case H_HARDSHRINK_BACKWARD:
  set(op.tensor, at::redispatch::hardshrink_backward(ks, grad_out, self, lambd));
  break;


case H_RSQRT:
  set(op.tensor, at::redispatch::rsqrt(ks, self));
  break;

case H_RSQRT_:
  init_update_in_place(op.tensor);
  at::redispatch::rsqrt_(ks, self)
  end_update_in_place(op.tensor);
  break;

case H_RSQRT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::rsqrt_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_SELECT_DIMNAME:
  set(op.tensor, at::redispatch::select(ks, self, dim, index));
  break;


case H_SELECT_INT:
  set(op.tensor, at::redispatch::select(ks, self, dim, index));
  break;


case H_SELECT_BACKWARD:
  set(op.tensor, at::redispatch::select_backward(ks, grad, input_sizes, dim, index));
  break;


case H_SELU:
  set(op.tensor, at::redispatch::selu(ks, self));
  break;

case H_SELU_:
  init_update_in_place(op.tensor);
  at::redispatch::selu_(ks, self)
  end_update_in_place(op.tensor);
  break;


case H_CELU:
  set(op.tensor, at::redispatch::celu(ks, self, alpha));
  break;

case H_CELU_:
  init_update_in_place(op.tensor);
  at::redispatch::celu_(ks, self, alpha)
  end_update_in_place(op.tensor);
  break;


case H_SILU:
  set(op.tensor, at::redispatch::silu(ks, self));
  break;

case H_SILU_:
  init_update_in_place(op.tensor);
  at::redispatch::silu_(ks, self)
  end_update_in_place(op.tensor);
  break;

case H_SILU_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::silu_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_SILU_BACKWARD:
  set(op.tensor, at::redispatch::silu_backward(ks, grad_output, self));
  break;


case H_SIGMOID:
  set(op.tensor, at::redispatch::sigmoid(ks, self));
  break;

case H_SIGMOID_:
  init_update_in_place(op.tensor);
  at::redispatch::sigmoid_(ks, self)
  end_update_in_place(op.tensor);
  break;

case H_SIGMOID_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::sigmoid_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_LOGIT:
  set(op.tensor, at::redispatch::logit(ks, self, eps));
  break;

case H_LOGIT_:
  init_update_in_place(op.tensor);
  at::redispatch::logit_(ks, self, eps)
  end_update_in_place(op.tensor);
  break;

case H_LOGIT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::logit_outf(ks, self, eps, out)
  end_update_in_place(op.tensor);
  break;

case H_SIN_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::sin_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;

case H_SINC_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::sinc_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;

case H_SINH_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::sinh_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_DETACH:
  set(op.tensor, at::redispatch::detach(ks, self));
  break;

case H_DETACH_:
  init_update_in_place(op.tensor);
  at::redispatch::detach_(ks, self)
  end_update_in_place(op.tensor);
  break;



case H_SLICE_TENSOR:
  set(op.tensor, at::redispatch::slice(ks, self, dim, start, end, step));
  break;


case H_SLICE_BACKWARD:
  set(op.tensor, at::redispatch::slice_backward(ks, grad, input_sizes, dim, start, end, step));
  break;



case H_SMM:
  set(op.tensor, at::redispatch::smm(ks, self, mat2));
  break;


case H_SOFTMAX_INT:
  set(op.tensor, at::redispatch::softmax(ks, self, dim, dtype));
  break;


case H_SOFTMAX_DIMNAME:
  set(op.tensor, at::redispatch::softmax(ks, self, dim, dtype));
  break;


case H__SOFTMAX:
  set(op.tensor, at::redispatch::_softmax(ks, self, dim, half_to_float));
  break;


case H__SOFTMAX_BACKWARD_DATA:
  set(op.tensor, at::redispatch::_softmax_backward_data(ks, grad_output, output, dim, self));
  break;






case H_SQUEEZE:
  set(op.tensor, at::redispatch::squeeze(ks, self));
  break;


case H_SQUEEZE_DIM:
  set(op.tensor, at::redispatch::squeeze(ks, self, dim));
  break;


case H_SQUEEZE_DIMNAME:
  set(op.tensor, at::redispatch::squeeze(ks, self, dim));
  break;

case H_SQUEEZE_:
  init_update_in_place(op.tensor);
  at::redispatch::squeeze_(ks, self)
  end_update_in_place(op.tensor);
  break;

case H_SQUEEZE__DIM:
  init_update_in_place(op.tensor);
  at::redispatch::squeeze_(ks, self, dim)
  end_update_in_place(op.tensor);
  break;

case H_SQUEEZE__DIMNAME:
  init_update_in_place(op.tensor);
  at::redispatch::squeeze_(ks, self, dim)
  end_update_in_place(op.tensor);
  break;


case H_SSPADDMM:
  set(op.tensor, at::redispatch::sspaddmm(ks, self, mat1, mat2, beta, alpha));
  break;

case H_SSPADDMM_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::sspaddmm_outf(ks, self, mat1, mat2, beta, alpha, out)
  end_update_in_place(op.tensor);
  break;


case H_STACK:
  set(op.tensor, at::redispatch::stack(ks, tensors, dim));
  break;

case H_STACK_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::stack_outf(ks, tensors, dim, out)
  end_update_in_place(op.tensor);
  break;


case H__STACK:
  set(op.tensor, at::redispatch::_stack(ks, tensors, dim));
  break;

case H__STACK_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::_stack_outf(ks, tensors, dim, out)
  end_update_in_place(op.tensor);
  break;


case H_HSTACK:
  set(op.tensor, at::redispatch::hstack(ks, tensors));
  break;

case H_HSTACK_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::hstack_outf(ks, tensors, out)
  end_update_in_place(op.tensor);
  break;


case H_VSTACK:
  set(op.tensor, at::redispatch::vstack(ks, tensors));
  break;

case H_VSTACK_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::vstack_outf(ks, tensors, out)
  end_update_in_place(op.tensor);
  break;


case H_DSTACK:
  set(op.tensor, at::redispatch::dstack(ks, tensors));
  break;

case H_DSTACK_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::dstack_outf(ks, tensors, out)
  end_update_in_place(op.tensor);
  break;


case H_STFT:
  set(op.tensor, at::redispatch::stft(ks, self, n_fft, hop_length, win_length, window, normalized, onesided, return_complex));
  break;


case H_ISTFT:
  set(op.tensor, at::redispatch::istft(ks, self, n_fft, hop_length, win_length, window, center, normalized, onesided, length, return_complex));
  break;



case H_SUM:
  set(op.tensor, at::redispatch::sum(ks, self, dtype));
  break;


case H_SUM_DIM_INTLIST:
  set(op.tensor, at::redispatch::sum(ks, self, dim, keepdim, dtype));
  break;


case H_SUM_DIM_DIMNAMELIST:
  set(op.tensor, at::redispatch::sum(ks, self, dim, keepdim, dtype));
  break;

case H_SUM_INTLIST_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::sum_outf(ks, self, dim, keepdim, dtype, out)
  end_update_in_place(op.tensor);
  break;

case H_SUM_DIMNAMELIST_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::sum_outf(ks, self, dim, keepdim, dtype, out)
  end_update_in_place(op.tensor);
  break;


case H_NANSUM:
  set(op.tensor, at::redispatch::nansum(ks, self, dtype));
  break;


case H_NANSUM_DIM_INTLIST:
  set(op.tensor, at::redispatch::nansum(ks, self, dim, keepdim, dtype));
  break;

case H_NANSUM_INTLIST_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::nansum_outf(ks, self, dim, keepdim, dtype, out)
  end_update_in_place(op.tensor);
  break;


case H_SUM_TO_SIZE:
  set(op.tensor, at::redispatch::sum_to_size(ks, self, size));
  break;


case H_SQRT:
  set(op.tensor, at::redispatch::sqrt(ks, self));
  break;

case H_SQRT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::sqrt_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_SQUARE:
  set(op.tensor, at::redispatch::square(ks, self));
  break;

case H_SQUARE_:
  init_update_in_place(op.tensor);
  at::redispatch::square_(ks, self)
  end_update_in_place(op.tensor);
  break;

case H_SQUARE_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::square_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_STD:
  set(op.tensor, at::redispatch::std(ks, self, unbiased));
  break;


case H_STD_DIM:
  set(op.tensor, at::redispatch::std(ks, self, dim, unbiased, keepdim));
  break;




case H_STD_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::std_outf(ks, self, dim, unbiased, keepdim, out)
  end_update_in_place(op.tensor);
  break;


case H_STD_NAMES_DIM:
  set(op.tensor, at::redispatch::std(ks, self, dim, unbiased, keepdim));
  break;

case H_STD_NAMES_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::std_outf(ks, self, dim, unbiased, keepdim, out)
  end_update_in_place(op.tensor);
  break;


case H_PROD:
  set(op.tensor, at::redispatch::prod(ks, self, dtype));
  break;


case H_PROD_DIM_INT:
  set(op.tensor, at::redispatch::prod(ks, self, dim, keepdim, dtype));
  break;

case H_PROD_INT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::prod_outf(ks, self, dim, keepdim, dtype, out)
  end_update_in_place(op.tensor);
  break;


case H_PROD_DIM_DIMNAME:
  set(op.tensor, at::redispatch::prod(ks, self, dim, keepdim, dtype));
  break;

case H_PROD_DIMNAME_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::prod_outf(ks, self, dim, keepdim, dtype, out)
  end_update_in_place(op.tensor);
  break;


case H_T:
  set(op.tensor, at::redispatch::t(ks, self));
  break;

case H_T_:
  init_update_in_place(op.tensor);
  at::redispatch::t_(ks, self)
  end_update_in_place(op.tensor);
  break;

case H_TAN_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::tan_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_TANH:
  set(op.tensor, at::redispatch::tanh(ks, self));
  break;

case H_TANH_:
  init_update_in_place(op.tensor);
  at::redispatch::tanh_(ks, self)
  end_update_in_place(op.tensor);
  break;

case H_TANH_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::tanh_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_TENSORDOT:
  set(op.tensor, at::redispatch::tensordot(ks, self, other, dims_self, dims_other));
  break;

case H_TENSORDOT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::tensordot_outf(ks, self, other, dims_self, dims_other, out)
  end_update_in_place(op.tensor);
  break;


case H_THRESHOLD:
  set(op.tensor, at::redispatch::threshold(ks, self, threshold, value));
  break;

case H_THRESHOLD_:
  init_update_in_place(op.tensor);
  at::redispatch::threshold_(ks, self, threshold, value)
  end_update_in_place(op.tensor);
  break;

case H_THRESHOLD_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::threshold_outf(ks, self, threshold, value, out)
  end_update_in_place(op.tensor);
  break;


case H_THRESHOLD_BACKWARD:
  set(op.tensor, at::redispatch::threshold_backward(ks, grad_output, self, threshold));
  break;


case H_TILE:
  set(op.tensor, at::redispatch::tile(ks, self, dims));
  break;


case H_TRANSPOSE_INT:
  set(op.tensor, at::redispatch::transpose(ks, self, dim0, dim1));
  break;


case H_TRANSPOSE_DIMNAME:
  set(op.tensor, at::redispatch::transpose(ks, self, dim0, dim1));
  break;


case H__MKLDNN_TRANSPOSE:
  set(op.tensor, at::redispatch::_mkldnn_transpose(ks, self, dim0, dim1));
  break;

case H_TRANSPOSE_:
  init_update_in_place(op.tensor);
  at::redispatch::transpose_(ks, self, dim0, dim1)
  end_update_in_place(op.tensor);
  break;

case H__MKLDNN_TRANSPOSE_:
  init_update_in_place(op.tensor);
  at::redispatch::_mkldnn_transpose_(ks, self, dim0, dim1)
  end_update_in_place(op.tensor);
  break;


case H_ONE_HOT:
  set(op.tensor, at::redispatch::one_hot(ks, self, num_classes));
  break;


case H_FLIP:
  set(op.tensor, at::redispatch::flip(ks, self, dims));
  break;


case H_FLIPLR:
  set(op.tensor, at::redispatch::fliplr(ks, self));
  break;


case H_FLIPUD:
  set(op.tensor, at::redispatch::flipud(ks, self));
  break;


case H_ROLL:
  set(op.tensor, at::redispatch::roll(ks, self, shifts, dims));
  break;


case H_ROT90:
  set(op.tensor, at::redispatch::rot90(ks, self, k, dims));
  break;


case H_TRAPZ_X:
  set(op.tensor, at::redispatch::trapz(ks, y, x, dim));
  break;


case H_TRAPZ_DX:
  set(op.tensor, at::redispatch::trapz(ks, y, dx, dim));
  break;


case H__TRILINEAR:
  set(op.tensor, at::redispatch::_trilinear(ks, i1, i2, i3, expand1, expand2, expand3, sumdim, unroll_dim));
  break;


case H_TRIPLET_MARGIN_LOSS:
  set(op.tensor, at::redispatch::triplet_margin_loss(ks, anchor, positive, negative, margin, p, eps, swap, reduction));
  break;


case H_TRUNC:
  set(op.tensor, at::redispatch::trunc(ks, self));
  break;

case H_TRUNC_:
  init_update_in_place(op.tensor);
  at::redispatch::trunc_(ks, self)
  end_update_in_place(op.tensor);
  break;

case H_TRUNC_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::trunc_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_FIX:
  set(op.tensor, at::redispatch::fix(ks, self));
  break;

case H_FIX_:
  init_update_in_place(op.tensor);
  at::redispatch::fix_(ks, self)
  end_update_in_place(op.tensor);
  break;

case H_FIX_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::fix_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_TYPE_AS:
  set(op.tensor, at::redispatch::type_as(ks, self, other));
  break;








case H__UNSAFE_VIEW:
  set(op.tensor, at::redispatch::_unsafe_view(ks, self, size));
  break;


case H_UNSQUEEZE:
  set(op.tensor, at::redispatch::unsqueeze(ks, self, dim));
  break;

case H_UNSQUEEZE_:
  init_update_in_place(op.tensor);
  at::redispatch::unsqueeze_(ks, self, dim)
  end_update_in_place(op.tensor);
  break;


case H_VANDER:
  set(op.tensor, at::redispatch::vander(ks, x, N, increasing));
  break;


case H_VAR:
  set(op.tensor, at::redispatch::var(ks, self, unbiased));
  break;


case H_VAR_DIM:
  set(op.tensor, at::redispatch::var(ks, self, dim, unbiased, keepdim));
  break;

case H_VAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::var_outf(ks, self, dim, unbiased, keepdim, out)
  end_update_in_place(op.tensor);
  break;


case H_VAR_NAMES_DIM:
  set(op.tensor, at::redispatch::var(ks, self, dim, unbiased, keepdim));
  break;

case H_VAR_NAMES_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::var_outf(ks, self, dim, unbiased, keepdim, out)
  end_update_in_place(op.tensor);
  break;





case H_VIEW_AS:
  set(op.tensor, at::redispatch::view_as(ks, self, other));
  break;


case H_WHERE_SELF:
  set(op.tensor, at::redispatch::where(ks, condition, self, other));
  break;


case H_WHERE_SCALARSELF:
  set(op.tensor, at::redispatch::where(ks, condition, self, other));
  break;


case H_WHERE_SCALAROTHER:
  set(op.tensor, at::redispatch::where(ks, condition, self, other));
  break;


case H_WHERE_SCALAR:
  set(op.tensor, at::redispatch::where(ks, condition, self, other));
  break;



case H__S_WHERE:
  set(op.tensor, at::redispatch::_s_where(ks, condition, self, other));
  break;


case H_NORM_EXCEPT_DIM:
  set(op.tensor, at::redispatch::norm_except_dim(ks, v, pow, dim));
  break;


case H__WEIGHT_NORM:
  set(op.tensor, at::redispatch::_weight_norm(ks, v, g, dim));
  break;





case H_ZEROS_NAMES:
  set(op.tensor, at::redispatch::zeros(ks, size, names, dtype, layout, device, pin_memory));
  break;


case H_ZEROS:
  set(op.tensor, at::redispatch::zeros(ks, size, dtype, layout, device, pin_memory));
  break;

case H_ZEROS_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::zeros_outf(ks, size, out)
  end_update_in_place(op.tensor);
  break;


case H_ZEROS_LIKE:
  set(op.tensor, at::redispatch::zeros_like(ks, self, dtype, layout, device, pin_memory, memory_format));
  break;


case H__STANDARD_GAMMA_GRAD:
  set(op.tensor, at::redispatch::_standard_gamma_grad(ks, self, output));
  break;


case H__STANDARD_GAMMA:
  set(op.tensor, at::redispatch::_standard_gamma(ks, self, generator));
  break;


case H__DIRICHLET_GRAD:
  set(op.tensor, at::redispatch::_dirichlet_grad(ks, x, alpha, total));
  break;


case H__SAMPLE_DIRICHLET:
  set(op.tensor, at::redispatch::_sample_dirichlet(ks, self, generator));
  break;


case H_POISSON:
  set(op.tensor, at::redispatch::poisson(ks, self, generator));
  break;


case H_BINOMIAL:
  set(op.tensor, at::redispatch::binomial(ks, count, prob, generator));
  break;


case H_NATIVE_NORM:
  set(op.tensor, at::redispatch::native_norm(ks, self, p));
  break;


case H_NATIVE_NORM_SCALAROPT_DIM_DTYPE:
  set(op.tensor, at::redispatch::native_norm(ks, self, p, dim, keepdim, dtype));
  break;


case H__SPARSE_SUM:
  set(op.tensor, at::redispatch::_sparse_sum(ks, self));
  break;


case H__SPARSE_SUM_DTYPE:
  set(op.tensor, at::redispatch::_sparse_sum(ks, self, dtype));
  break;


case H__SPARSE_SUM_DIM:
  set(op.tensor, at::redispatch::_sparse_sum(ks, self, dim));
  break;


case H__SPARSE_SUM_DIM_DTYPE:
  set(op.tensor, at::redispatch::_sparse_sum(ks, self, dim, dtype));
  break;


case H__SPARSE_SUM_BACKWARD:
  set(op.tensor, at::redispatch::_sparse_sum_backward(ks, grad, self, dim));
  break;


case H__SPARSE_SOFTMAX_INT:
  set(op.tensor, at::redispatch::_sparse_softmax(ks, self, dim, dtype));
  break;


case H__SPARSE_SOFTMAX_DIMNAME:
  set(op.tensor, at::redispatch::_sparse_softmax(ks, self, dim, dtype));
  break;


case H__SPARSE_SOFTMAX:
  set(op.tensor, at::redispatch::_sparse_softmax(ks, self, dim, half_to_float));
  break;


case H__SPARSE_SOFTMAX_BACKWARD_DATA:
  set(op.tensor, at::redispatch::_sparse_softmax_backward_data(ks, grad_output, output, dim, self));
  break;


case H__SPARSE_LOG_SOFTMAX_INT:
  set(op.tensor, at::redispatch::_sparse_log_softmax(ks, self, dim, dtype));
  break;


case H__SPARSE_LOG_SOFTMAX_DIMNAME:
  set(op.tensor, at::redispatch::_sparse_log_softmax(ks, self, dim, dtype));
  break;


case H__SPARSE_LOG_SOFTMAX:
  set(op.tensor, at::redispatch::_sparse_log_softmax(ks, self, dim, half_to_float));
  break;


case H__SPARSE_LOG_SOFTMAX_BACKWARD_DATA:
  set(op.tensor, at::redispatch::_sparse_log_softmax_backward_data(ks, grad_output, output, dim, self));
  break;


case H_NORM_SCALAROPT_DTYPE:
  set(op.tensor, at::redispatch::norm(ks, self, p, dtype));
  break;


case H_NORM_SCALAR:
  set(op.tensor, at::redispatch::norm(ks, self, p));
  break;


case H_NORM_SCALAROPT_DIM_DTYPE:
  set(op.tensor, at::redispatch::norm(ks, self, p, dim, keepdim, dtype));
  break;


case H_NORM_SCALAROPT_DIM:
  set(op.tensor, at::redispatch::norm(ks, self, p, dim, keepdim));
  break;

case H_NORM_DTYPE_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::norm_outf(ks, self, p, dim, keepdim, dtype, out)
  end_update_in_place(op.tensor);
  break;

case H_NORM_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::norm_outf(ks, self, p, dim, keepdim, out)
  end_update_in_place(op.tensor);
  break;


case H_NORM_NAMES_SCALAROPT_DIM_DTYPE:
  set(op.tensor, at::redispatch::norm(ks, self, p, dim, keepdim, dtype));
  break;


case H_NORM_NAMES_SCALAROPT_DIM:
  set(op.tensor, at::redispatch::norm(ks, self, p, dim, keepdim));
  break;

case H_NORM_NAMES_DTYPE_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::norm_outf(ks, self, p, dim, keepdim, dtype, out)
  end_update_in_place(op.tensor);
  break;

case H_NORM_NAMES_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::norm_outf(ks, self, p, dim, keepdim, out)
  end_update_in_place(op.tensor);
  break;




case H_FROBENIUS_NORM:
  set(op.tensor, at::redispatch::frobenius_norm(ks, self));
  break;


case H_FROBENIUS_NORM_DIM:
  set(op.tensor, at::redispatch::frobenius_norm(ks, self, dim, keepdim));
  break;

case H_FROBENIUS_NORM_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::frobenius_norm_outf(ks, self, dim, keepdim, out)
  end_update_in_place(op.tensor);
  break;


case H_NUCLEAR_NORM:
  set(op.tensor, at::redispatch::nuclear_norm(ks, self, keepdim));
  break;

case H_NUCLEAR_NORM_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::nuclear_norm_outf(ks, self, keepdim, out)
  end_update_in_place(op.tensor);
  break;


case H_NUCLEAR_NORM_DIM:
  set(op.tensor, at::redispatch::nuclear_norm(ks, self, dim, keepdim));
  break;

case H_NUCLEAR_NORM_DIM_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::nuclear_norm_outf(ks, self, dim, keepdim, out)
  end_update_in_place(op.tensor);
  break;


case H_CLONE:
  set(op.tensor, at::redispatch::clone(ks, self, memory_format));
  break;



case H_ZERO_:
  init_update_in_place(op.tensor);
  at::redispatch::zero_(ks, self)
  end_update_in_place(op.tensor);
  break;

case H_SUB_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::sub_outf(ks, self, other, alpha, out)
  end_update_in_place(op.tensor);
  break;


case H_SUB_TENSOR:
  set(op.tensor, at::redispatch::sub(ks, self, other, alpha));
  break;

case H_SUB__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::sub_(ks, self, other, alpha)
  end_update_in_place(op.tensor);
  break;


case H_SUB_SCALAR:
  set(op.tensor, at::redispatch::sub(ks, self, other, alpha));
  break;

case H_SUB__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::sub_(ks, self, other, alpha)
  end_update_in_place(op.tensor);
  break;

case H_SUBTRACT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::subtract_outf(ks, self, other, alpha, out)
  end_update_in_place(op.tensor);
  break;


case H_SUBTRACT_TENSOR:
  set(op.tensor, at::redispatch::subtract(ks, self, other, alpha));
  break;

case H_SUBTRACT__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::subtract_(ks, self, other, alpha)
  end_update_in_place(op.tensor);
  break;


case H_SUBTRACT_SCALAR:
  set(op.tensor, at::redispatch::subtract(ks, self, other, alpha));
  break;

case H_SUBTRACT__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::subtract_(ks, self, other, alpha)
  end_update_in_place(op.tensor);
  break;


case H_RSUB_TENSOR:
  set(op.tensor, at::redispatch::rsub(ks, self, other, alpha));
  break;

case H_HEAVISIDE_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::heaviside_outf(ks, self, values, out)
  end_update_in_place(op.tensor);
  break;


case H_HEAVISIDE:
  set(op.tensor, at::redispatch::heaviside(ks, self, values));
  break;

case H_HEAVISIDE_:
  init_update_in_place(op.tensor);
  at::redispatch::heaviside_(ks, self, values)
  end_update_in_place(op.tensor);
  break;


case H_RSUB_SCALAR:
  set(op.tensor, at::redispatch::rsub(ks, self, other, alpha));
  break;


case H__SPARSE_ADDMM:
  set(op.tensor, at::redispatch::_sparse_addmm(ks, self, sparse, dense, beta, alpha));
  break;

case H_ADDMM_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::addmm_outf(ks, self, mat1, mat2, beta, alpha, out)
  end_update_in_place(op.tensor);
  break;


case H_ADDMM:
  set(op.tensor, at::redispatch::addmm(ks, self, mat1, mat2, beta, alpha));
  break;

case H_ADDMM_:
  init_update_in_place(op.tensor);
  at::redispatch::addmm_(ks, self, mat1, mat2, beta, alpha)
  end_update_in_place(op.tensor);
  break;


case H_SPARSE_CSR_TENSOR_CROW_COL_VALUE_SIZE:
  set(op.tensor, at::redispatch::sparse_csr_tensor(ks, crow_indices, col_indices, values, size, dtype, layout, device, pin_memory));
  break;


case H_SPARSE_CSR_TENSOR_CROW_COL_VALUE:
  set(op.tensor, at::redispatch::sparse_csr_tensor(ks, crow_indices, col_indices, values, dtype, layout, device, pin_memory));
  break;


case H_SPARSE_COO_TENSOR_SIZE:
  set(op.tensor, at::redispatch::sparse_coo_tensor(ks, size, dtype, layout, device, pin_memory));
  break;


case H_SPARSE_COO_TENSOR_INDICES:
  set(op.tensor, at::redispatch::sparse_coo_tensor(ks, indices, values, dtype, layout, device, pin_memory));
  break;


case H_SPARSE_COO_TENSOR_INDICES_SIZE:
  set(op.tensor, at::redispatch::sparse_coo_tensor(ks, indices, values, size, dtype, layout, device, pin_memory));
  break;


case H__SPARSE_COO_TENSOR_UNSAFE:
  set(op.tensor, at::redispatch::_sparse_coo_tensor_unsafe(ks, indices, values, size, dtype, layout, device, pin_memory));
  break;



case H__SPARSE_COO_TENSOR_WITH_DIMS:
  set(op.tensor, at::redispatch::_sparse_coo_tensor_with_dims(ks, sparse_dim, dense_dim, size, dtype, layout, device, pin_memory));
  break;


case H__SPARSE_COO_TENSOR_WITH_DIMS_AND_TENSORS:
  set(op.tensor, at::redispatch::_sparse_coo_tensor_with_dims_and_tensors(ks, sparse_dim, dense_dim, size, indices, values, dtype, layout, device, pin_memory));
  break;




case H_SPARSE_MASK:
  set(op.tensor, at::redispatch::sparse_mask(ks, self, mask));
  break;


case H_TO_DENSE:
  set(op.tensor, at::redispatch::to_dense(ks, self, dtype));
  break;


case H_TO_DENSE_BACKWARD:
  set(op.tensor, at::redispatch::to_dense_backward(ks, grad, input));
  break;







case H_COALESCE:
  set(op.tensor, at::redispatch::coalesce(ks, self));
  break;


case H__COALESCE:
  set(op.tensor, at::redispatch::_coalesce(ks, self));
  break;



case H__INDICES:
  set(op.tensor, at::redispatch::_indices(ks, self));
  break;


case H__VALUES:
  set(op.tensor, at::redispatch::_values(ks, self));
  break;

case H__COALESCED_:
  init_update_in_place(op.tensor);
  at::redispatch::_coalesced_(ks, self, coalesced)
  end_update_in_place(op.tensor);
  break;


case H_INDICES:
  set(op.tensor, at::redispatch::indices(ks, self));
  break;


case H_VALUES:
  set(op.tensor, at::redispatch::values(ks, self));
  break;


case H_CROW_INDICES:
  set(op.tensor, at::redispatch::crow_indices(ks, self));
  break;


case H_COL_INDICES:
  set(op.tensor, at::redispatch::col_indices(ks, self));
  break;

case H_HSPMM_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::hspmm_outf(ks, mat1, mat2, out)
  end_update_in_place(op.tensor);
  break;


case H_HSPMM:
  set(op.tensor, at::redispatch::hspmm(ks, mat1, mat2));
  break;

case H_COPY_SPARSE_TO_SPARSE_:
  init_update_in_place(op.tensor);
  at::redispatch::copy_sparse_to_sparse_(ks, self, src, non_blocking)
  end_update_in_place(op.tensor);
  break;




case H_TO_SPARSE_SPARSE_DIM:
  set(op.tensor, at::redispatch::to_sparse(ks, self, sparse_dim));
  break;


case H_TO_SPARSE:
  set(op.tensor, at::redispatch::to_sparse(ks, self));
  break;


case H_TO_MKLDNN:
  set(op.tensor, at::redispatch::to_mkldnn(ks, self, dtype));
  break;


case H_MKLDNN_REORDER_CONV2D_WEIGHT:
  set(op.tensor, at::redispatch::mkldnn_reorder_conv2d_weight(ks, self, padding, stride, dilation, groups));
  break;


case H_MKLDNN_REORDER_CONV3D_WEIGHT:
  set(op.tensor, at::redispatch::mkldnn_reorder_conv3d_weight(ks, self, padding, stride, dilation, groups));
  break;


case H_TO_MKLDNN_BACKWARD:
  set(op.tensor, at::redispatch::to_mkldnn_backward(ks, grad, input));
  break;


case H_QUANTIZE_PER_TENSOR:
  set(op.tensor, at::redispatch::quantize_per_tensor(ks, self, scale, zero_point, dtype));
  break;



case H_QUANTIZE_PER_CHANNEL:
  set(op.tensor, at::redispatch::quantize_per_channel(ks, self, scales, zero_points, axis, dtype));
  break;


case H_DEQUANTIZE_SELF:
  set(op.tensor, at::redispatch::dequantize(ks, self));
  break;





case H_Q_PER_CHANNEL_SCALES:
  set(op.tensor, at::redispatch::q_per_channel_scales(ks, self));
  break;


case H_Q_PER_CHANNEL_ZERO_POINTS:
  set(op.tensor, at::redispatch::q_per_channel_zero_points(ks, self));
  break;



case H_INT_REPR:
  set(op.tensor, at::redispatch::int_repr(ks, self));
  break;


case H__MAKE_PER_TENSOR_QUANTIZED_TENSOR:
  set(op.tensor, at::redispatch::_make_per_tensor_quantized_tensor(ks, self, scale, zero_point));
  break;


case H__MAKE_PER_CHANNEL_QUANTIZED_TENSOR:
  set(op.tensor, at::redispatch::_make_per_channel_quantized_tensor(ks, self, scale, zero_point, axis));
  break;



case H_FAKE_QUANTIZE_PER_TENSOR_AFFINE:
  set(op.tensor, at::redispatch::fake_quantize_per_tensor_affine(ks, self, scale, zero_point, quant_min, quant_max));
  break;



case H_FAKE_QUANTIZE_PER_TENSOR_AFFINE_CACHEMASK_BACKWARD:
  set(op.tensor, at::redispatch::fake_quantize_per_tensor_affine_cachemask_backward(ks, grad, mask));
  break;


case H__FAKE_QUANTIZE_LEARNABLE_PER_TENSOR_AFFINE:
  set(op.tensor, at::redispatch::_fake_quantize_learnable_per_tensor_affine(ks, self, scale, zero_point, quant_min, quant_max, grad_factor));
  break;



case H_FAKE_QUANTIZE_PER_CHANNEL_AFFINE:
  set(op.tensor, at::redispatch::fake_quantize_per_channel_affine(ks, self, scale, zero_point, axis, quant_min, quant_max));
  break;



case H_FAKE_QUANTIZE_PER_CHANNEL_AFFINE_CACHEMASK_BACKWARD:
  set(op.tensor, at::redispatch::fake_quantize_per_channel_affine_cachemask_backward(ks, grad, mask));
  break;


case H__FAKE_QUANTIZE_LEARNABLE_PER_CHANNEL_AFFINE:
  set(op.tensor, at::redispatch::_fake_quantize_learnable_per_channel_affine(ks, self, scale, zero_point, axis, quant_min, quant_max, grad_factor));
  break;




case H__SATURATE_WEIGHT_TO_FP16:
  set(op.tensor, at::redispatch::_saturate_weight_to_fp16(ks, weight));
  break;



case H_TO_DTYPE_LAYOUT:
  set(op.tensor, at::redispatch::to(ks, self, dtype, layout, device, pin_memory, non_blocking, copy, memory_format));
  break;


case H_TO_DEVICE:
  set(op.tensor, at::redispatch::to(ks, self, device, dtype, non_blocking, copy, memory_format));
  break;


case H_TO_DTYPE:
  set(op.tensor, at::redispatch::to(ks, self, dtype, non_blocking, copy, memory_format));
  break;


case H_TO_OTHER:
  set(op.tensor, at::redispatch::to(ks, self, other, non_blocking, copy, memory_format));
  break;



case H_CARTESIAN_PROD:
  set(op.tensor, at::redispatch::cartesian_prod(ks, tensors));
  break;


case H_COMBINATIONS:
  set(op.tensor, at::redispatch::combinations(ks, self, r, with_replacement));
  break;

























case H_GRU_CELL:
  set(op.tensor, at::redispatch::gru_cell(ks, input, hx, w_ih, w_hh, b_ih, b_hh));
  break;


case H_RNN_TANH_CELL:
  set(op.tensor, at::redispatch::rnn_tanh_cell(ks, input, hx, w_ih, w_hh, b_ih, b_hh));
  break;


case H_RNN_RELU_CELL:
  set(op.tensor, at::redispatch::rnn_relu_cell(ks, input, hx, w_ih, w_hh, b_ih, b_hh));
  break;



case H_QUANTIZED_GRU_CELL:
  set(op.tensor, at::redispatch::quantized_gru_cell(ks, input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh));
  break;


case H_QUANTIZED_RNN_RELU_CELL:
  set(op.tensor, at::redispatch::quantized_rnn_relu_cell(ks, input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh));
  break;


case H_QUANTIZED_RNN_TANH_CELL:
  set(op.tensor, at::redispatch::quantized_rnn_tanh_cell(ks, input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh));
  break;



case H__PACK_PADDED_SEQUENCE_BACKWARD:
  set(op.tensor, at::redispatch::_pack_padded_sequence_backward(ks, grad, input_size, batch_sizes, batch_first));
  break;


case H_SET__SOURCE_STORAGE:
  init_update_in_place(op.tensor);
  at::redispatch::set_(ks, self, source)
  end_update_in_place(op.tensor);
  break;

case H_SET__SOURCE_STORAGE_STORAGE_OFFSET:
  init_update_in_place(op.tensor);
  at::redispatch::set_(ks, self, source, storage_offset, size, stride)
  end_update_in_place(op.tensor);
  break;

case H_SET__SOURCE_TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::set_(ks, self, source)
  end_update_in_place(op.tensor);
  break;

case H_SET_:
  init_update_in_place(op.tensor);
  at::redispatch::set_(ks, self)
  end_update_in_place(op.tensor);
  break;


case H_MASKED_FILL__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::masked_fill_(ks, self, mask, value)
  end_update_in_place(op.tensor);
  break;


case H_MASKED_FILL_SCALAR:
  set(op.tensor, at::redispatch::masked_fill(ks, self, mask, value));
  break;

case H_MASKED_FILL__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::masked_fill_(ks, self, mask, value)
  end_update_in_place(op.tensor);
  break;


case H_MASKED_FILL_TENSOR:
  set(op.tensor, at::redispatch::masked_fill(ks, self, mask, value));
  break;

case H_MASKED_SCATTER_:
  init_update_in_place(op.tensor);
  at::redispatch::masked_scatter_(ks, self, mask, source)
  end_update_in_place(op.tensor);
  break;


case H_MASKED_SCATTER:
  set(op.tensor, at::redispatch::masked_scatter(ks, self, mask, source));
  break;


case H_VIEW:
  set(op.tensor, at::redispatch::view(ks, self, size));
  break;


case H_VIEW_DTYPE:
  set(op.tensor, at::redispatch::view(ks, self, dtype));
  break;

case H_PUT_:
  init_update_in_place(op.tensor);
  at::redispatch::put_(ks, self, index, source, accumulate)
  end_update_in_place(op.tensor);
  break;


case H_PUT:
  set(op.tensor, at::redispatch::put(ks, self, index, source, accumulate));
  break;

case H_INDEX_ADD_:
  init_update_in_place(op.tensor);
  at::redispatch::index_add_(ks, self, dim, index, source)
  end_update_in_place(op.tensor);
  break;

case H_INDEX_ADD__ALPHA:
  init_update_in_place(op.tensor);
  at::redispatch::index_add_(ks, self, dim, index, source, alpha)
  end_update_in_place(op.tensor);
  break;


case H_INDEX_ADD:
  set(op.tensor, at::redispatch::index_add(ks, self, dim, index, source));
  break;


case H_INDEX_ADD_ALPHA:
  set(op.tensor, at::redispatch::index_add(ks, self, dim, index, source, alpha));
  break;


case H_INDEX_ADD_DIMNAME:
  set(op.tensor, at::redispatch::index_add(ks, self, dim, index, source, alpha));
  break;

case H_INDEX_FILL__INT_SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::index_fill_(ks, self, dim, index, value)
  end_update_in_place(op.tensor);
  break;


case H_INDEX_FILL_INT_SCALAR:
  set(op.tensor, at::redispatch::index_fill(ks, self, dim, index, value));
  break;

case H_INDEX_FILL__INT_TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::index_fill_(ks, self, dim, index, value)
  end_update_in_place(op.tensor);
  break;


case H_INDEX_FILL_INT_TENSOR:
  set(op.tensor, at::redispatch::index_fill(ks, self, dim, index, value));
  break;

case H_INDEX_FILL__DIMNAME_SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::index_fill_(ks, self, dim, index, value)
  end_update_in_place(op.tensor);
  break;

case H_INDEX_FILL__DIMNAME_TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::index_fill_(ks, self, dim, index, value)
  end_update_in_place(op.tensor);
  break;


case H_INDEX_FILL_DIMNAME_SCALAR:
  set(op.tensor, at::redispatch::index_fill(ks, self, dim, index, value));
  break;


case H_INDEX_FILL_DIMNAME_TENSOR:
  set(op.tensor, at::redispatch::index_fill(ks, self, dim, index, value));
  break;

case H_SCATTER__SRC:
  init_update_in_place(op.tensor);
  at::redispatch::scatter_(ks, self, dim, index, src)
  end_update_in_place(op.tensor);
  break;


case H_SCATTER_SRC:
  set(op.tensor, at::redispatch::scatter(ks, self, dim, index, src));
  break;

case H_SCATTER__VALUE:
  init_update_in_place(op.tensor);
  at::redispatch::scatter_(ks, self, dim, index, value)
  end_update_in_place(op.tensor);
  break;


case H_SCATTER_VALUE:
  set(op.tensor, at::redispatch::scatter(ks, self, dim, index, value));
  break;


case H_SCATTER_DIMNAME_SRC:
  set(op.tensor, at::redispatch::scatter(ks, self, dim, index, src));
  break;


case H_SCATTER_DIMNAME_VALUE:
  set(op.tensor, at::redispatch::scatter(ks, self, dim, index, value));
  break;

case H_SCATTER__REDUCE:
  init_update_in_place(op.tensor);
  at::redispatch::scatter_(ks, self, dim, index, src, reduce)
  end_update_in_place(op.tensor);
  break;

case H_SCATTER__VALUE_REDUCE:
  init_update_in_place(op.tensor);
  at::redispatch::scatter_(ks, self, dim, index, value, reduce)
  end_update_in_place(op.tensor);
  break;

case H_SCATTER_ADD_:
  init_update_in_place(op.tensor);
  at::redispatch::scatter_add_(ks, self, dim, index, src)
  end_update_in_place(op.tensor);
  break;


case H_SCATTER_ADD:
  set(op.tensor, at::redispatch::scatter_add(ks, self, dim, index, src));
  break;


case H_SCATTER_ADD_DIMNAME:
  set(op.tensor, at::redispatch::scatter_add(ks, self, dim, index, src));
  break;

case H_EQ__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::eq_(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H_EQ__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::eq_(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H_BITWISE_AND_TENSOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::bitwise_and_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;

case H_BITWISE_AND_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::bitwise_and_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_BITWISE_AND_SCALAR:
  set(op.tensor, at::redispatch::bitwise_and(ks, self, other));
  break;


case H_BITWISE_AND_TENSOR:
  set(op.tensor, at::redispatch::bitwise_and(ks, self, other));
  break;

case H_BITWISE_AND__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::bitwise_and_(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H_BITWISE_AND__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::bitwise_and_(ks, self, other)
  end_update_in_place(op.tensor);
  break;


case H___AND___SCALAR:
  set(op.tensor, at::redispatch::__and__(ks, self, other));
  break;


case H___AND___TENSOR:
  set(op.tensor, at::redispatch::__and__(ks, self, other));
  break;

case H___IAND___SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::__iand__(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H___IAND___TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::__iand__(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H_BITWISE_OR_TENSOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::bitwise_or_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;

case H_BITWISE_OR_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::bitwise_or_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_BITWISE_OR_SCALAR:
  set(op.tensor, at::redispatch::bitwise_or(ks, self, other));
  break;


case H_BITWISE_OR_TENSOR:
  set(op.tensor, at::redispatch::bitwise_or(ks, self, other));
  break;

case H_BITWISE_OR__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::bitwise_or_(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H_BITWISE_OR__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::bitwise_or_(ks, self, other)
  end_update_in_place(op.tensor);
  break;


case H___OR___SCALAR:
  set(op.tensor, at::redispatch::__or__(ks, self, other));
  break;


case H___OR___TENSOR:
  set(op.tensor, at::redispatch::__or__(ks, self, other));
  break;

case H___IOR___SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::__ior__(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H___IOR___TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::__ior__(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H_BITWISE_XOR_TENSOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::bitwise_xor_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;

case H_BITWISE_XOR_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::bitwise_xor_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_BITWISE_XOR_SCALAR:
  set(op.tensor, at::redispatch::bitwise_xor(ks, self, other));
  break;


case H_BITWISE_XOR_TENSOR:
  set(op.tensor, at::redispatch::bitwise_xor(ks, self, other));
  break;

case H_BITWISE_XOR__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::bitwise_xor_(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H_BITWISE_XOR__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::bitwise_xor_(ks, self, other)
  end_update_in_place(op.tensor);
  break;


case H___XOR___SCALAR:
  set(op.tensor, at::redispatch::__xor__(ks, self, other));
  break;


case H___XOR___TENSOR:
  set(op.tensor, at::redispatch::__xor__(ks, self, other));
  break;

case H___IXOR___SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::__ixor__(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H___IXOR___TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::__ixor__(ks, self, other)
  end_update_in_place(op.tensor);
  break;


case H___LSHIFT___SCALAR:
  set(op.tensor, at::redispatch::__lshift__(ks, self, other));
  break;


case H___LSHIFT___TENSOR:
  set(op.tensor, at::redispatch::__lshift__(ks, self, other));
  break;

case H___ILSHIFT___SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::__ilshift__(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H___ILSHIFT___TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::__ilshift__(ks, self, other)
  end_update_in_place(op.tensor);
  break;


case H___RSHIFT___SCALAR:
  set(op.tensor, at::redispatch::__rshift__(ks, self, other));
  break;


case H___RSHIFT___TENSOR:
  set(op.tensor, at::redispatch::__rshift__(ks, self, other));
  break;

case H___IRSHIFT___SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::__irshift__(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H___IRSHIFT___TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::__irshift__(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H_TRIL_:
  init_update_in_place(op.tensor);
  at::redispatch::tril_(ks, self, diagonal)
  end_update_in_place(op.tensor);
  break;

case H_TRIU_:
  init_update_in_place(op.tensor);
  at::redispatch::triu_(ks, self, diagonal)
  end_update_in_place(op.tensor);
  break;

case H_RENORM_:
  init_update_in_place(op.tensor);
  at::redispatch::renorm_(ks, self, p, dim, maxnorm)
  end_update_in_place(op.tensor);
  break;

case H_LERP__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::lerp_(ks, self, end, weight)
  end_update_in_place(op.tensor);
  break;

case H_LERP__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::lerp_(ks, self, end, weight)
  end_update_in_place(op.tensor);
  break;

case H_FMOD__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::fmod_(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H_FMOD__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::fmod_(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H_REMAINDER__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::remainder_(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H_REMAINDER__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::remainder_(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H_ADDBMM_:
  init_update_in_place(op.tensor);
  at::redispatch::addbmm_(ks, self, batch1, batch2, beta, alpha)
  end_update_in_place(op.tensor);
  break;

case H_ADDBMM_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::addbmm_outf(ks, self, batch1, batch2, beta, alpha, out)
  end_update_in_place(op.tensor);
  break;


case H_ADDBMM:
  set(op.tensor, at::redispatch::addbmm(ks, self, batch1, batch2, beta, alpha));
  break;

case H_ADDCDIV_:
  init_update_in_place(op.tensor);
  at::redispatch::addcdiv_(ks, self, tensor1, tensor2, value)
  end_update_in_place(op.tensor);
  break;

case H_RANDOM__FROM:
  init_update_in_place(op.tensor);
  at::redispatch::random_(ks, self, from, to, generator)
  end_update_in_place(op.tensor);
  break;

case H_RANDOM__TO:
  init_update_in_place(op.tensor);
  at::redispatch::random_(ks, self, to, generator)
  end_update_in_place(op.tensor);
  break;

case H_RANDOM_:
  init_update_in_place(op.tensor);
  at::redispatch::random_(ks, self, generator)
  end_update_in_place(op.tensor);
  break;

case H_UNIFORM_:
  init_update_in_place(op.tensor);
  at::redispatch::uniform_(ks, self, from, to, generator)
  end_update_in_place(op.tensor);
  break;

case H_CAUCHY_:
  init_update_in_place(op.tensor);
  at::redispatch::cauchy_(ks, self, median, sigma, generator)
  end_update_in_place(op.tensor);
  break;

case H_LOG_NORMAL_:
  init_update_in_place(op.tensor);
  at::redispatch::log_normal_(ks, self, mean, std, generator)
  end_update_in_place(op.tensor);
  break;

case H_EXPONENTIAL_:
  init_update_in_place(op.tensor);
  at::redispatch::exponential_(ks, self, lambd, generator)
  end_update_in_place(op.tensor);
  break;

case H_GEOMETRIC_:
  init_update_in_place(op.tensor);
  at::redispatch::geometric_(ks, self, p, generator)
  end_update_in_place(op.tensor);
  break;

case H_DIAG_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::diag_outf(ks, self, diagonal, out)
  end_update_in_place(op.tensor);
  break;


case H_DIAG:
  set(op.tensor, at::redispatch::diag(ks, self, diagonal));
  break;


case H_DIAG_BACKWARD:
  set(op.tensor, at::redispatch::diag_backward(ks, grad, input_sizes, diagonal));
  break;

case H_CROSS_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::cross_outf(ks, self, other, dim, out)
  end_update_in_place(op.tensor);
  break;


case H_CROSS:
  set(op.tensor, at::redispatch::cross(ks, self, other, dim));
  break;

case H_TRIU_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::triu_outf(ks, self, diagonal, out)
  end_update_in_place(op.tensor);
  break;


case H_TRIU:
  set(op.tensor, at::redispatch::triu(ks, self, diagonal));
  break;

case H_TRIL_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::tril_outf(ks, self, diagonal, out)
  end_update_in_place(op.tensor);
  break;


case H_TRIL:
  set(op.tensor, at::redispatch::tril(ks, self, diagonal));
  break;


case H_TRIL_INDICES:
  set(op.tensor, at::redispatch::tril_indices(ks, row, col, offset, dtype, layout, device, pin_memory));
  break;


case H_TRIU_INDICES:
  set(op.tensor, at::redispatch::triu_indices(ks, row, col, offset, dtype, layout, device, pin_memory));
  break;


case H_TRACE:
  set(op.tensor, at::redispatch::trace(ks, self));
  break;


case H_TRACE_BACKWARD:
  set(op.tensor, at::redispatch::trace_backward(ks, grad, sizes));
  break;

case H_NE_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::ne_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_NE_SCALAR:
  set(op.tensor, at::redispatch::ne(ks, self, other));
  break;

case H_NE_TENSOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::ne_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_NE_TENSOR:
  set(op.tensor, at::redispatch::ne(ks, self, other));
  break;

case H_NE__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::ne_(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H_NE__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::ne_(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H_NOT_EQUAL_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::not_equal_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_NOT_EQUAL_SCALAR:
  set(op.tensor, at::redispatch::not_equal(ks, self, other));
  break;

case H_NOT_EQUAL_TENSOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::not_equal_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_NOT_EQUAL_TENSOR:
  set(op.tensor, at::redispatch::not_equal(ks, self, other));
  break;

case H_NOT_EQUAL__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::not_equal_(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H_NOT_EQUAL__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::not_equal_(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H_EQ_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::eq_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_EQ_SCALAR:
  set(op.tensor, at::redispatch::eq(ks, self, other));
  break;

case H_EQ_TENSOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::eq_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_EQ_TENSOR:
  set(op.tensor, at::redispatch::eq(ks, self, other));
  break;

case H_GE_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::ge_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_GE_SCALAR:
  set(op.tensor, at::redispatch::ge(ks, self, other));
  break;

case H_GE_TENSOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::ge_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_GE_TENSOR:
  set(op.tensor, at::redispatch::ge(ks, self, other));
  break;

case H_GE__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::ge_(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H_GE__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::ge_(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H_GREATER_EQUAL_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::greater_equal_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_GREATER_EQUAL_SCALAR:
  set(op.tensor, at::redispatch::greater_equal(ks, self, other));
  break;

case H_GREATER_EQUAL_TENSOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::greater_equal_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_GREATER_EQUAL_TENSOR:
  set(op.tensor, at::redispatch::greater_equal(ks, self, other));
  break;

case H_GREATER_EQUAL__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::greater_equal_(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H_GREATER_EQUAL__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::greater_equal_(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H_LE_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::le_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_LE_SCALAR:
  set(op.tensor, at::redispatch::le(ks, self, other));
  break;

case H_LE_TENSOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::le_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_LE_TENSOR:
  set(op.tensor, at::redispatch::le(ks, self, other));
  break;

case H_LE__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::le_(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H_LE__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::le_(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H_LESS_EQUAL_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::less_equal_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_LESS_EQUAL_SCALAR:
  set(op.tensor, at::redispatch::less_equal(ks, self, other));
  break;

case H_LESS_EQUAL_TENSOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::less_equal_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_LESS_EQUAL_TENSOR:
  set(op.tensor, at::redispatch::less_equal(ks, self, other));
  break;

case H_LESS_EQUAL__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::less_equal_(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H_LESS_EQUAL__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::less_equal_(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H_GT_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::gt_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_GT_SCALAR:
  set(op.tensor, at::redispatch::gt(ks, self, other));
  break;

case H_GT_TENSOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::gt_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_GT_TENSOR:
  set(op.tensor, at::redispatch::gt(ks, self, other));
  break;

case H_GT__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::gt_(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H_GT__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::gt_(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H_GREATER_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::greater_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_GREATER_SCALAR:
  set(op.tensor, at::redispatch::greater(ks, self, other));
  break;

case H_GREATER_TENSOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::greater_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_GREATER_TENSOR:
  set(op.tensor, at::redispatch::greater(ks, self, other));
  break;

case H_GREATER__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::greater_(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H_GREATER__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::greater_(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H_LT_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::lt_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_LT_SCALAR:
  set(op.tensor, at::redispatch::lt(ks, self, other));
  break;

case H_LT_TENSOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::lt_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_LT_TENSOR:
  set(op.tensor, at::redispatch::lt(ks, self, other));
  break;

case H_LT__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::lt_(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H_LT__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::lt_(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H_LESS_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::less_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_LESS_SCALAR:
  set(op.tensor, at::redispatch::less(ks, self, other));
  break;

case H_LESS_TENSOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::less_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_LESS_TENSOR:
  set(op.tensor, at::redispatch::less(ks, self, other));
  break;

case H_LESS__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::less_(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H_LESS__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::less_(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H_TAKE_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::take_outf(ks, self, index, out)
  end_update_in_place(op.tensor);
  break;


case H_TAKE:
  set(op.tensor, at::redispatch::take(ks, self, index));
  break;

case H_TAKE_ALONG_DIM_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::take_along_dim_outf(ks, self, indices, dim, out)
  end_update_in_place(op.tensor);
  break;


case H_TAKE_ALONG_DIM:
  set(op.tensor, at::redispatch::take_along_dim(ks, self, indices, dim));
  break;

case H_INDEX_SELECT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::index_select_outf(ks, self, dim, index, out)
  end_update_in_place(op.tensor);
  break;


case H_INDEX_SELECT:
  set(op.tensor, at::redispatch::index_select(ks, self, dim, index));
  break;

case H_INDEX_SELECT_DIMNAME_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::index_select_outf(ks, self, dim, index, out)
  end_update_in_place(op.tensor);
  break;


case H_INDEX_SELECT_DIMNAME:
  set(op.tensor, at::redispatch::index_select(ks, self, dim, index));
  break;


case H_INDEX_SELECT_BACKWARD:
  set(op.tensor, at::redispatch::index_select_backward(ks, grad, self_sizes, dim, index));
  break;

case H_MASKED_SELECT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::masked_select_outf(ks, self, mask, out)
  end_update_in_place(op.tensor);
  break;


case H_MASKED_SELECT:
  set(op.tensor, at::redispatch::masked_select(ks, self, mask));
  break;


case H_MASKED_SELECT_BACKWARD:
  set(op.tensor, at::redispatch::masked_select_backward(ks, grad, input, mask));
  break;

case H_NONZERO_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::nonzero_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_NONZERO:
  set(op.tensor, at::redispatch::nonzero(ks, self));
  break;


case H_GATHER_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::gather_outf(ks, self, dim, index, sparse_grad, out)
  end_update_in_place(op.tensor);
  break;


case H_GATHER:
  set(op.tensor, at::redispatch::gather(ks, self, dim, index, sparse_grad));
  break;


case H_GATHER_BACKWARD:
  set(op.tensor, at::redispatch::gather_backward(ks, grad, self, dim, index, sparse_grad));
  break;

case H_GATHER_DIMNAME_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::gather_outf(ks, self, dim, index, sparse_grad, out)
  end_update_in_place(op.tensor);
  break;


case H_GATHER_DIMNAME:
  set(op.tensor, at::redispatch::gather(ks, self, dim, index, sparse_grad));
  break;


case H__GATHER_SPARSE_BACKWARD:
  set(op.tensor, at::redispatch::_gather_sparse_backward(ks, self, dim, index, grad));
  break;

case H_ADDCMUL_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::addcmul_outf(ks, self, tensor1, tensor2, value, out)
  end_update_in_place(op.tensor);
  break;


case H_ADDCMUL:
  set(op.tensor, at::redispatch::addcmul(ks, self, tensor1, tensor2, value));
  break;

case H_ADDCMUL_:
  init_update_in_place(op.tensor);
  at::redispatch::addcmul_(ks, self, tensor1, tensor2, value)
  end_update_in_place(op.tensor);
  break;

case H_ADDCDIV_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::addcdiv_outf(ks, self, tensor1, tensor2, value, out)
  end_update_in_place(op.tensor);
  break;


case H_ADDCDIV:
  set(op.tensor, at::redispatch::addcdiv(ks, self, tensor1, tensor2, value));
  break;


case H_CROSS_ENTROPY_LOSS:
  set(op.tensor, at::redispatch::cross_entropy_loss(ks, self, target, weight, reduction, ignore_index));
  break;














case H_SWAPAXES:
  set(op.tensor, at::redispatch::swapaxes(ks, self, axis0, axis1));
  break;

case H_SWAPAXES_:
  init_update_in_place(op.tensor);
  at::redispatch::swapaxes_(ks, self, axis0, axis1)
  end_update_in_place(op.tensor);
  break;


case H_SWAPDIMS:
  set(op.tensor, at::redispatch::swapdims(ks, self, dim0, dim1));
  break;

case H_SWAPDIMS_:
  init_update_in_place(op.tensor);
  at::redispatch::swapdims_(ks, self, dim0, dim1)
  end_update_in_place(op.tensor);
  break;

case H_CHOLESKY_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::cholesky_outf(ks, self, upper, out)
  end_update_in_place(op.tensor);
  break;


case H_CHOLESKY:
  set(op.tensor, at::redispatch::cholesky(ks, self, upper));
  break;


case H__CHOLESKY_HELPER:
  set(op.tensor, at::redispatch::_cholesky_helper(ks, self, upper));
  break;

case H_CHOLESKY_SOLVE_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::cholesky_solve_outf(ks, self, input2, upper, out)
  end_update_in_place(op.tensor);
  break;


case H_CHOLESKY_SOLVE:
  set(op.tensor, at::redispatch::cholesky_solve(ks, self, input2, upper));
  break;


case H__CHOLESKY_SOLVE_HELPER:
  set(op.tensor, at::redispatch::_cholesky_solve_helper(ks, self, A, upper));
  break;





case H_CHOLESKY_INVERSE:
  set(op.tensor, at::redispatch::cholesky_inverse(ks, self, upper));
  break;

case H_CHOLESKY_INVERSE_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::cholesky_inverse_outf(ks, self, upper, out)
  end_update_in_place(op.tensor);
  break;






case H_ORGQR:
  set(op.tensor, at::redispatch::orgqr(ks, self, input2));
  break;

case H_ORGQR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::orgqr_outf(ks, self, input2, out)
  end_update_in_place(op.tensor);
  break;

case H_ORMQR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::ormqr_outf(ks, self, input2, input3, left, transpose, out)
  end_update_in_place(op.tensor);
  break;


case H_ORMQR:
  set(op.tensor, at::redispatch::ormqr(ks, self, input2, input3, left, transpose));
  break;


case H_LU_SOLVE_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::lu_solve_outf(ks, self, LU_data, LU_pivots, out)
  end_update_in_place(op.tensor);
  break;


case H_LU_SOLVE:
  set(op.tensor, at::redispatch::lu_solve(ks, self, LU_data, LU_pivots));
  break;


case H__LU_SOLVE_HELPER:
  set(op.tensor, at::redispatch::_lu_solve_helper(ks, self, LU_data, LU_pivots));
  break;

case H_MULTINOMIAL_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::multinomial_outf(ks, self, num_samples, replacement, generator, out)
  end_update_in_place(op.tensor);
  break;


case H_MULTINOMIAL:
  set(op.tensor, at::redispatch::multinomial(ks, self, num_samples, replacement, generator));
  break;

case H_LGAMMA_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::lgamma_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;

case H_DIGAMMA_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::digamma_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;

case H_POLYGAMMA_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::polygamma_outf(ks, n, self, out)
  end_update_in_place(op.tensor);
  break;


case H_POLYGAMMA:
  set(op.tensor, at::redispatch::polygamma(ks, n, self));
  break;

case H_POLYGAMMA_:
  init_update_in_place(op.tensor);
  at::redispatch::polygamma_(ks, self, n)
  end_update_in_place(op.tensor);
  break;

case H_ERFINV_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::erfinv_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_I0:
  set(op.tensor, at::redispatch::i0(ks, self));
  break;

case H_I0_:
  init_update_in_place(op.tensor);
  at::redispatch::i0_(ks, self)
  end_update_in_place(op.tensor);
  break;

case H_I0_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::i0_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_SIGN:
  set(op.tensor, at::redispatch::sign(ks, self));
  break;

case H_SIGN_:
  init_update_in_place(op.tensor);
  at::redispatch::sign_(ks, self)
  end_update_in_place(op.tensor);
  break;

case H_SIGN_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::sign_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_SIGNBIT:
  set(op.tensor, at::redispatch::signbit(ks, self));
  break;

case H_SIGNBIT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::signbit_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_DIST:
  set(op.tensor, at::redispatch::dist(ks, self, other, p));
  break;

case H_ATAN2_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::atan2_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;

case H_LERP_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::lerp_outf(ks, self, end, weight, out)
  end_update_in_place(op.tensor);
  break;

case H_LERP_TENSOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::lerp_outf(ks, self, end, weight, out)
  end_update_in_place(op.tensor);
  break;


case H_LERP_SCALAR:
  set(op.tensor, at::redispatch::lerp(ks, self, end, weight));
  break;


case H_LERP_TENSOR:
  set(op.tensor, at::redispatch::lerp(ks, self, end, weight));
  break;

case H_HISTC_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::histc_outf(ks, self, bins, min, max, out)
  end_update_in_place(op.tensor);
  break;


case H_HISTC:
  set(op.tensor, at::redispatch::histc(ks, self, bins, min, max));
  break;

case H_FMOD_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::fmod_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_FMOD_SCALAR:
  set(op.tensor, at::redispatch::fmod(ks, self, other));
  break;

case H_FMOD_TENSOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::fmod_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_FMOD_TENSOR:
  set(op.tensor, at::redispatch::fmod(ks, self, other));
  break;

case H_HYPOT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::hypot_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_HYPOT:
  set(op.tensor, at::redispatch::hypot(ks, self, other));
  break;

case H_HYPOT_:
  init_update_in_place(op.tensor);
  at::redispatch::hypot_(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H_IGAMMA_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::igamma_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_IGAMMA:
  set(op.tensor, at::redispatch::igamma(ks, self, other));
  break;

case H_IGAMMA_:
  init_update_in_place(op.tensor);
  at::redispatch::igamma_(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H_IGAMMAC_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::igammac_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_IGAMMAC:
  set(op.tensor, at::redispatch::igammac(ks, self, other));
  break;

case H_IGAMMAC_:
  init_update_in_place(op.tensor);
  at::redispatch::igammac_(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H_NEXTAFTER_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::nextafter_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_NEXTAFTER:
  set(op.tensor, at::redispatch::nextafter(ks, self, other));
  break;

case H_NEXTAFTER_:
  init_update_in_place(op.tensor);
  at::redispatch::nextafter_(ks, self, other)
  end_update_in_place(op.tensor);
  break;

case H_REMAINDER_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::remainder_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_REMAINDER_SCALAR:
  set(op.tensor, at::redispatch::remainder(ks, self, other));
  break;

case H_REMAINDER_TENSOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::remainder_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_REMAINDER_TENSOR:
  set(op.tensor, at::redispatch::remainder(ks, self, other));
  break;


case H_MIN:
  set(op.tensor, at::redispatch::min(ks, self));
  break;


case H_FMIN:
  set(op.tensor, at::redispatch::fmin(ks, self, other));
  break;

case H_FMIN_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::fmin_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_MAX:
  set(op.tensor, at::redispatch::max(ks, self));
  break;


case H_FMAX:
  set(op.tensor, at::redispatch::fmax(ks, self, other));
  break;

case H_FMAX_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::fmax_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_MAXIMUM:
  set(op.tensor, at::redispatch::maximum(ks, self, other));
  break;

case H_MAXIMUM_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::maximum_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_MAX_OTHER:
  set(op.tensor, at::redispatch::max(ks, self, other));
  break;

case H_MAX_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::max_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_MINIMUM:
  set(op.tensor, at::redispatch::minimum(ks, self, other));
  break;

case H_MINIMUM_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::minimum_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;

case H_MIN_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::min_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_MIN_OTHER:
  set(op.tensor, at::redispatch::min(ks, self, other));
  break;

case H_QUANTILE_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::quantile_outf(ks, self, q, dim, keepdim, out)
  end_update_in_place(op.tensor);
  break;


case H_QUANTILE_SCALAR:
  set(op.tensor, at::redispatch::quantile(ks, self, q, dim, keepdim));
  break;

case H_QUANTILE_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::quantile_outf(ks, self, q, dim, keepdim, out)
  end_update_in_place(op.tensor);
  break;


case H_QUANTILE:
  set(op.tensor, at::redispatch::quantile(ks, self, q, dim, keepdim));
  break;

case H_NANQUANTILE_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::nanquantile_outf(ks, self, q, dim, keepdim, out)
  end_update_in_place(op.tensor);
  break;


case H_NANQUANTILE_SCALAR:
  set(op.tensor, at::redispatch::nanquantile(ks, self, q, dim, keepdim));
  break;

case H_NANQUANTILE_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::nanquantile_outf(ks, self, q, dim, keepdim, out)
  end_update_in_place(op.tensor);
  break;


case H_NANQUANTILE:
  set(op.tensor, at::redispatch::nanquantile(ks, self, q, dim, keepdim));
  break;

case H_QUANTILE_NEW_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::quantile_outf(ks, self, q, dim, keepdim, interpolation, out)
  end_update_in_place(op.tensor);
  break;


case H_QUANTILE_NEW_SCALAR:
  set(op.tensor, at::redispatch::quantile(ks, self, q, dim, keepdim, interpolation));
  break;

case H_QUANTILE_NEW_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::quantile_outf(ks, self, q, dim, keepdim, interpolation, out)
  end_update_in_place(op.tensor);
  break;


case H_QUANTILE_NEW:
  set(op.tensor, at::redispatch::quantile(ks, self, q, dim, keepdim, interpolation));
  break;

case H_NANQUANTILE_NEW_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::nanquantile_outf(ks, self, q, dim, keepdim, interpolation, out)
  end_update_in_place(op.tensor);
  break;


case H_NANQUANTILE_NEW_SCALAR:
  set(op.tensor, at::redispatch::nanquantile(ks, self, q, dim, keepdim, interpolation));
  break;

case H_NANQUANTILE_NEW_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::nanquantile_outf(ks, self, q, dim, keepdim, interpolation, out)
  end_update_in_place(op.tensor);
  break;


case H_NANQUANTILE_NEW:
  set(op.tensor, at::redispatch::nanquantile(ks, self, q, dim, keepdim, interpolation));
  break;









case H_MSORT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::msort_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_MSORT:
  set(op.tensor, at::redispatch::msort(ks, self));
  break;


case H_ARGSORT:
  set(op.tensor, at::redispatch::argsort(ks, self, dim, descending));
  break;


case H_ARGSORT_DIMNAME:
  set(op.tensor, at::redispatch::argsort(ks, self, dim, descending));
  break;




case H_ALL:
  set(op.tensor, at::redispatch::all(ks, self));
  break;


case H_ANY:
  set(op.tensor, at::redispatch::any(ks, self));
  break;

case H_RENORM_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::renorm_outf(ks, self, p, dim, maxnorm, out)
  end_update_in_place(op.tensor);
  break;


case H_RENORM:
  set(op.tensor, at::redispatch::renorm(ks, self, p, dim, maxnorm));
  break;


case H_UNFOLD:
  set(op.tensor, at::redispatch::unfold(ks, self, dimension, size, step));
  break;


case H_UNFOLD_BACKWARD:
  set(op.tensor, at::redispatch::unfold_backward(ks, grad_in, input_sizes, dim, size, step));
  break;


case H_POW_TENSOR_TENSOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::pow_outf(ks, self, exponent, out)
  end_update_in_place(op.tensor);
  break;

case H_POW_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::pow_outf(ks, self, exponent, out)
  end_update_in_place(op.tensor);
  break;

case H_POW_TENSOR_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::pow_outf(ks, self, exponent, out)
  end_update_in_place(op.tensor);
  break;


case H_POW_TENSOR_SCALAR:
  set(op.tensor, at::redispatch::pow(ks, self, exponent));
  break;

case H_FLOAT_POWER_TENSOR_TENSOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::float_power_outf(ks, self, exponent, out)
  end_update_in_place(op.tensor);
  break;


case H_FLOAT_POWER_TENSOR_TENSOR:
  set(op.tensor, at::redispatch::float_power(ks, self, exponent));
  break;

case H_FLOAT_POWER_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::float_power_outf(ks, self, exponent, out)
  end_update_in_place(op.tensor);
  break;


case H_FLOAT_POWER_SCALAR:
  set(op.tensor, at::redispatch::float_power(ks, self, exponent));
  break;

case H_FLOAT_POWER_TENSOR_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::float_power_outf(ks, self, exponent, out)
  end_update_in_place(op.tensor);
  break;


case H_FLOAT_POWER_TENSOR_SCALAR:
  set(op.tensor, at::redispatch::float_power(ks, self, exponent));
  break;

case H_FLOAT_POWER__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::float_power_(ks, self, exponent)
  end_update_in_place(op.tensor);
  break;

case H_FLOAT_POWER__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::float_power_(ks, self, exponent)
  end_update_in_place(op.tensor);
  break;

case H_NORMAL_:
  init_update_in_place(op.tensor);
  at::redispatch::normal_(ks, self, mean, std, generator)
  end_update_in_place(op.tensor);
  break;

case H_NORMAL_TENSOR_FLOAT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::normal_outf(ks, mean, std, generator, out)
  end_update_in_place(op.tensor);
  break;


case H_NORMAL_TENSOR_FLOAT:
  set(op.tensor, at::redispatch::normal(ks, mean, std, generator));
  break;

case H_NORMAL_FLOAT_TENSOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::normal_outf(ks, mean, std, generator, out)
  end_update_in_place(op.tensor);
  break;


case H_NORMAL_FLOAT_TENSOR:
  set(op.tensor, at::redispatch::normal(ks, mean, std, generator));
  break;

case H_NORMAL_TENSOR_TENSOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::normal_outf(ks, mean, std, generator, out)
  end_update_in_place(op.tensor);
  break;


case H_NORMAL_TENSOR_TENSOR:
  set(op.tensor, at::redispatch::normal(ks, mean, std, generator));
  break;


case H_NORMAL_FLOAT_FLOAT:
  set(op.tensor, at::redispatch::normal(ks, mean, std, size, generator, dtype, layout, device, pin_memory));
  break;

case H_NORMAL_FLOAT_FLOAT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::normal_outf(ks, mean, std, size, generator, out)
  end_update_in_place(op.tensor);
  break;


case H_ALIAS:
  set(op.tensor, at::redispatch::alias(ks, self));
  break;

case H__INDEX_COPY_:
  init_update_in_place(op.tensor);
  at::redispatch::_index_copy_(ks, self, dim, index, source)
  end_update_in_place(op.tensor);
  break;


case H__CUMSUM:
  set(op.tensor, at::redispatch::_cumsum(ks, self, dim));
  break;

case H__CUMSUM_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::_cumsum_outf(ks, self, dim, out)
  end_update_in_place(op.tensor);
  break;


case H__CUMPROD:
  set(op.tensor, at::redispatch::_cumprod(ks, self, dim));
  break;

case H__CUMPROD_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::_cumprod_outf(ks, self, dim, out)
  end_update_in_place(op.tensor);
  break;


case H__VAR:
  set(op.tensor, at::redispatch::_var(ks, self, unbiased));
  break;


case H__STD:
  set(op.tensor, at::redispatch::_std(ks, self, unbiased));
  break;



case H__AMP_UPDATE_SCALE:
  set(op.tensor, at::redispatch::_amp_update_scale(ks, growth_tracker, current_scale, found_inf, scale_growth_factor, scale_backoff_factor, growth_interval));
  break;


case H__CAT:
  set(op.tensor, at::redispatch::_cat(ks, tensors, dim));
  break;

case H__CAT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::_cat_outf(ks, tensors, dim, out)
  end_update_in_place(op.tensor);
  break;





























































































case H_BUCKETIZE_TENSOR:
  set(op.tensor, at::redispatch::bucketize(ks, self, boundaries, out_int32, right));
  break;

case H_BUCKETIZE_TENSOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::bucketize_outf(ks, self, boundaries, out_int32, right, out)
  end_update_in_place(op.tensor);
  break;


case H_BUCKETIZE_SCALAR:
  set(op.tensor, at::redispatch::bucketize(ks, self, boundaries, out_int32, right));
  break;


case H_SEARCHSORTED_TENSOR:
  set(op.tensor, at::redispatch::searchsorted(ks, sorted_sequence, self, out_int32, right));
  break;

case H_SEARCHSORTED_TENSOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::searchsorted_outf(ks, sorted_sequence, self, out_int32, right, out)
  end_update_in_place(op.tensor);
  break;


case H_SEARCHSORTED_SCALAR:
  set(op.tensor, at::redispatch::searchsorted(ks, sorted_sequence, self, out_int32, right));
  break;

case H_MSE_LOSS_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::mse_loss_outf(ks, self, target, reduction, out)
  end_update_in_place(op.tensor);
  break;


case H_MSE_LOSS:
  set(op.tensor, at::redispatch::mse_loss(ks, self, target, reduction));
  break;

case H_MSE_LOSS_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::mse_loss_backward_outf(ks, grad_output, self, target, reduction, grad_input)
  end_update_in_place(op.tensor);
  break;


case H_MSE_LOSS_BACKWARD:
  set(op.tensor, at::redispatch::mse_loss_backward(ks, grad_output, self, target, reduction));
  break;

case H_L1_LOSS_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::l1_loss_outf(ks, self, target, reduction, out)
  end_update_in_place(op.tensor);
  break;


case H_L1_LOSS:
  set(op.tensor, at::redispatch::l1_loss(ks, self, target, reduction));
  break;

case H_L1_LOSS_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::l1_loss_backward_outf(ks, grad_output, self, target, reduction, grad_input)
  end_update_in_place(op.tensor);
  break;


case H_L1_LOSS_BACKWARD:
  set(op.tensor, at::redispatch::l1_loss_backward(ks, grad_output, self, target, reduction));
  break;

case H_MULTI_MARGIN_LOSS_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::multi_margin_loss_outf(ks, self, target, p, margin, weight, reduction, out)
  end_update_in_place(op.tensor);
  break;


case H_MULTI_MARGIN_LOSS:
  set(op.tensor, at::redispatch::multi_margin_loss(ks, self, target, p, margin, weight, reduction));
  break;

case H_MULTI_MARGIN_LOSS_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::multi_margin_loss_backward_outf(ks, grad_output, self, target, p, margin, weight, reduction, grad_input)
  end_update_in_place(op.tensor);
  break;


case H_MULTI_MARGIN_LOSS_BACKWARD:
  set(op.tensor, at::redispatch::multi_margin_loss_backward(ks, grad_output, self, target, p, margin, weight, reduction));
  break;

case H_MULTILABEL_MARGIN_LOSS_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::multilabel_margin_loss_outf(ks, self, target, reduction, out)
  end_update_in_place(op.tensor);
  break;


case H_MULTILABEL_MARGIN_LOSS:
  set(op.tensor, at::redispatch::multilabel_margin_loss(ks, self, target, reduction));
  break;



case H_MULTILABEL_MARGIN_LOSS_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::multilabel_margin_loss_backward_outf(ks, grad_output, self, target, reduction, is_target, grad_input)
  end_update_in_place(op.tensor);
  break;


case H_MULTILABEL_MARGIN_LOSS_BACKWARD:
  set(op.tensor, at::redispatch::multilabel_margin_loss_backward(ks, grad_output, self, target, reduction, is_target));
  break;

case H_NLL_LOSS_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::nll_loss_outf(ks, self, target, weight, reduction, ignore_index, out)
  end_update_in_place(op.tensor);
  break;


case H_NLL_LOSS_ND:
  set(op.tensor, at::redispatch::nll_loss_nd(ks, self, target, weight, reduction, ignore_index));
  break;


case H_NLL_LOSS:
  set(op.tensor, at::redispatch::nll_loss(ks, self, target, weight, reduction, ignore_index));
  break;



case H_NLL_LOSS_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::nll_loss_backward_outf(ks, grad_output, self, target, weight, reduction, ignore_index, total_weight, grad_input)
  end_update_in_place(op.tensor);
  break;


case H_NLL_LOSS_BACKWARD:
  set(op.tensor, at::redispatch::nll_loss_backward(ks, grad_output, self, target, weight, reduction, ignore_index, total_weight));
  break;

case H_NLL_LOSS2D_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::nll_loss2d_outf(ks, self, target, weight, reduction, ignore_index, out)
  end_update_in_place(op.tensor);
  break;


case H_NLL_LOSS2D:
  set(op.tensor, at::redispatch::nll_loss2d(ks, self, target, weight, reduction, ignore_index));
  break;



case H_NLL_LOSS2D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::nll_loss2d_backward_outf(ks, grad_output, self, target, weight, reduction, ignore_index, total_weight, grad_input)
  end_update_in_place(op.tensor);
  break;


case H_NLL_LOSS2D_BACKWARD:
  set(op.tensor, at::redispatch::nll_loss2d_backward(ks, grad_output, self, target, weight, reduction, ignore_index, total_weight));
  break;

case H_SMOOTH_L1_LOSS_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::smooth_l1_loss_outf(ks, self, target, reduction, beta, out)
  end_update_in_place(op.tensor);
  break;


case H_SMOOTH_L1_LOSS:
  set(op.tensor, at::redispatch::smooth_l1_loss(ks, self, target, reduction, beta));
  break;

case H_SMOOTH_L1_LOSS_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::smooth_l1_loss_backward_outf(ks, grad_output, self, target, reduction, beta, grad_input)
  end_update_in_place(op.tensor);
  break;


case H_SMOOTH_L1_LOSS_BACKWARD:
  set(op.tensor, at::redispatch::smooth_l1_loss_backward(ks, grad_output, self, target, reduction, beta));
  break;

case H_HUBER_LOSS_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::huber_loss_outf(ks, self, target, reduction, delta, out)
  end_update_in_place(op.tensor);
  break;


case H_HUBER_LOSS:
  set(op.tensor, at::redispatch::huber_loss(ks, self, target, reduction, delta));
  break;

case H_HUBER_LOSS_BACKWARD_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::huber_loss_backward_outf(ks, grad_output, self, target, reduction, delta, grad_input)
  end_update_in_place(op.tensor);
  break;


case H_HUBER_LOSS_BACKWARD:
  set(op.tensor, at::redispatch::huber_loss_backward(ks, grad_output, self, target, reduction, delta));
  break;

case H_SOFT_MARGIN_LOSS_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::soft_margin_loss_outf(ks, self, target, reduction, out)
  end_update_in_place(op.tensor);
  break;


case H_SOFT_MARGIN_LOSS:
  set(op.tensor, at::redispatch::soft_margin_loss(ks, self, target, reduction));
  break;

case H_SOFT_MARGIN_LOSS_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::soft_margin_loss_backward_outf(ks, grad_output, self, target, reduction, grad_input)
  end_update_in_place(op.tensor);
  break;


case H_SOFT_MARGIN_LOSS_BACKWARD:
  set(op.tensor, at::redispatch::soft_margin_loss_backward(ks, grad_output, self, target, reduction));
  break;

case H_ELU_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::elu_outf(ks, self, alpha, scale, input_scale, out)
  end_update_in_place(op.tensor);
  break;


case H_ELU:
  set(op.tensor, at::redispatch::elu(ks, self, alpha, scale, input_scale));
  break;


case H_ELU_BACKWARD:
  set(op.tensor, at::redispatch::elu_backward(ks, grad_output, alpha, scale, input_scale, is_result, self_or_result));
  break;

case H_ELU_:
  init_update_in_place(op.tensor);
  at::redispatch::elu_(ks, self, alpha, scale, input_scale)
  end_update_in_place(op.tensor);
  break;

case H_GLU_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::glu_outf(ks, self, dim, out)
  end_update_in_place(op.tensor);
  break;


case H_GLU:
  set(op.tensor, at::redispatch::glu(ks, self, dim));
  break;

case H_GLU_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::glu_backward_outf(ks, grad_output, self, dim, grad_input)
  end_update_in_place(op.tensor);
  break;


case H_GLU_BACKWARD:
  set(op.tensor, at::redispatch::glu_backward(ks, grad_output, self, dim));
  break;

case H_HARDSIGMOID_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::hardsigmoid_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_HARDSIGMOID:
  set(op.tensor, at::redispatch::hardsigmoid(ks, self));
  break;

case H_HARDSIGMOID_:
  init_update_in_place(op.tensor);
  at::redispatch::hardsigmoid_(ks, self)
  end_update_in_place(op.tensor);
  break;


case H_HARDSIGMOID_BACKWARD:
  set(op.tensor, at::redispatch::hardsigmoid_backward(ks, grad_output, self));
  break;

case H_HARDTANH_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::hardtanh_outf(ks, self, min_val, max_val, out)
  end_update_in_place(op.tensor);
  break;


case H_HARDTANH:
  set(op.tensor, at::redispatch::hardtanh(ks, self, min_val, max_val));
  break;

case H_HARDTANH_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::hardtanh_backward_outf(ks, grad_output, self, min_val, max_val, grad_input)
  end_update_in_place(op.tensor);
  break;


case H_HARDTANH_BACKWARD:
  set(op.tensor, at::redispatch::hardtanh_backward(ks, grad_output, self, min_val, max_val));
  break;

case H_HARDTANH_:
  init_update_in_place(op.tensor);
  at::redispatch::hardtanh_(ks, self, min_val, max_val)
  end_update_in_place(op.tensor);
  break;

case H_HARDSWISH_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::hardswish_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_HARDSWISH:
  set(op.tensor, at::redispatch::hardswish(ks, self));
  break;

case H_HARDSWISH_:
  init_update_in_place(op.tensor);
  at::redispatch::hardswish_(ks, self)
  end_update_in_place(op.tensor);
  break;


case H_HARDSWISH_BACKWARD:
  set(op.tensor, at::redispatch::hardswish_backward(ks, grad_output, self));
  break;

case H_LEAKY_RELU_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::leaky_relu_outf(ks, self, negative_slope, out)
  end_update_in_place(op.tensor);
  break;


case H_LEAKY_RELU:
  set(op.tensor, at::redispatch::leaky_relu(ks, self, negative_slope));
  break;


case H_LEAKY_RELU_BACKWARD:
  set(op.tensor, at::redispatch::leaky_relu_backward(ks, grad_output, self, negative_slope, self_is_result));
  break;

case H_LEAKY_RELU_:
  init_update_in_place(op.tensor);
  at::redispatch::leaky_relu_(ks, self, negative_slope)
  end_update_in_place(op.tensor);
  break;

case H_LOG_SIGMOID_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::log_sigmoid_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_LOG_SIGMOID:
  set(op.tensor, at::redispatch::log_sigmoid(ks, self));
  break;



case H_LOG_SIGMOID_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::log_sigmoid_backward_outf(ks, grad_output, self, buffer, grad_input)
  end_update_in_place(op.tensor);
  break;


case H_LOG_SIGMOID_BACKWARD:
  set(op.tensor, at::redispatch::log_sigmoid_backward(ks, grad_output, self, buffer));
  break;

case H_RRELU_WITH_NOISE_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::rrelu_with_noise_outf(ks, self, noise, lower, upper, training, generator, out)
  end_update_in_place(op.tensor);
  break;


case H_RRELU_WITH_NOISE:
  set(op.tensor, at::redispatch::rrelu_with_noise(ks, self, noise, lower, upper, training, generator));
  break;


case H_RRELU_WITH_NOISE_BACKWARD:
  set(op.tensor, at::redispatch::rrelu_with_noise_backward(ks, grad_output, self, noise, lower, upper, training, self_is_result));
  break;

case H_RRELU_WITH_NOISE_:
  init_update_in_place(op.tensor);
  at::redispatch::rrelu_with_noise_(ks, self, noise, lower, upper, training, generator)
  end_update_in_place(op.tensor);
  break;

case H_SOFTPLUS_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::softplus_outf(ks, self, beta, threshold, out)
  end_update_in_place(op.tensor);
  break;


case H_SOFTPLUS:
  set(op.tensor, at::redispatch::softplus(ks, self, beta, threshold));
  break;

case H_SOFTPLUS_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::softplus_backward_outf(ks, grad_output, self, beta, threshold, output, grad_input)
  end_update_in_place(op.tensor);
  break;


case H_SOFTPLUS_BACKWARD:
  set(op.tensor, at::redispatch::softplus_backward(ks, grad_output, self, beta, threshold, output));
  break;

case H_SOFTSHRINK_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::softshrink_outf(ks, self, lambd, out)
  end_update_in_place(op.tensor);
  break;


case H_SOFTSHRINK:
  set(op.tensor, at::redispatch::softshrink(ks, self, lambd));
  break;

case H_SOFTSHRINK_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::softshrink_backward_outf(ks, grad_output, self, lambd, grad_input)
  end_update_in_place(op.tensor);
  break;


case H_SOFTSHRINK_BACKWARD:
  set(op.tensor, at::redispatch::softshrink_backward(ks, grad_output, self, lambd));
  break;

case H_ADAPTIVE_AVG_POOL2D_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::adaptive_avg_pool2d_outf(ks, self, output_size, out)
  end_update_in_place(op.tensor);
  break;


case H_ADAPTIVE_AVG_POOL2D:
  set(op.tensor, at::redispatch::adaptive_avg_pool2d(ks, self, output_size));
  break;


case H_MKLDNN_ADAPTIVE_AVG_POOL2D:
  set(op.tensor, at::redispatch::mkldnn_adaptive_avg_pool2d(ks, self, output_size));
  break;


case H_MKLDNN_ADAPTIVE_AVG_POOL2D_BACKWARD:
  set(op.tensor, at::redispatch::mkldnn_adaptive_avg_pool2d_backward(ks, grad_output, self));
  break;


case H__ADAPTIVE_AVG_POOL2D:
  set(op.tensor, at::redispatch::_adaptive_avg_pool2d(ks, self, output_size));
  break;


case H__ADAPTIVE_AVG_POOL2D_BACKWARD:
  set(op.tensor, at::redispatch::_adaptive_avg_pool2d_backward(ks, grad_output, self));
  break;

case H_ADAPTIVE_AVG_POOL3D_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::adaptive_avg_pool3d_outf(ks, self, output_size, out)
  end_update_in_place(op.tensor);
  break;


case H_ADAPTIVE_AVG_POOL3D:
  set(op.tensor, at::redispatch::adaptive_avg_pool3d(ks, self, output_size));
  break;


case H__ADAPTIVE_AVG_POOL3D:
  set(op.tensor, at::redispatch::_adaptive_avg_pool3d(ks, self, output_size));
  break;

case H_ADAPTIVE_AVG_POOL3D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::adaptive_avg_pool3d_backward_outf(ks, grad_output, self, grad_input)
  end_update_in_place(op.tensor);
  break;


case H__ADAPTIVE_AVG_POOL3D_BACKWARD:
  set(op.tensor, at::redispatch::_adaptive_avg_pool3d_backward(ks, grad_output, self));
  break;


case H_ADAPTIVE_MAX_POOL2D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::adaptive_max_pool2d_backward_outf(ks, grad_output, self, indices, grad_input)
  end_update_in_place(op.tensor);
  break;


case H_ADAPTIVE_MAX_POOL2D_BACKWARD:
  set(op.tensor, at::redispatch::adaptive_max_pool2d_backward(ks, grad_output, self, indices));
  break;


case H_ADAPTIVE_MAX_POOL3D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::adaptive_max_pool3d_backward_outf(ks, grad_output, self, indices, grad_input)
  end_update_in_place(op.tensor);
  break;


case H_ADAPTIVE_MAX_POOL3D_BACKWARD:
  set(op.tensor, at::redispatch::adaptive_max_pool3d_backward(ks, grad_output, self, indices));
  break;

case H_AVG_POOL2D_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::avg_pool2d_outf(ks, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, out)
  end_update_in_place(op.tensor);
  break;


case H_AVG_POOL2D:
  set(op.tensor, at::redispatch::avg_pool2d(ks, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override));
  break;

case H_AVG_POOL2D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::avg_pool2d_backward_outf(ks, grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, grad_input)
  end_update_in_place(op.tensor);
  break;


case H_AVG_POOL2D_BACKWARD:
  set(op.tensor, at::redispatch::avg_pool2d_backward(ks, grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override));
  break;

case H_AVG_POOL3D_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::avg_pool3d_outf(ks, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, out)
  end_update_in_place(op.tensor);
  break;


case H_AVG_POOL3D:
  set(op.tensor, at::redispatch::avg_pool3d(ks, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override));
  break;

case H_AVG_POOL3D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::avg_pool3d_backward_outf(ks, grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, grad_input)
  end_update_in_place(op.tensor);
  break;


case H_AVG_POOL3D_BACKWARD:
  set(op.tensor, at::redispatch::avg_pool3d_backward(ks, grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override));
  break;


case H_FRACTIONAL_MAX_POOL2D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::fractional_max_pool2d_backward_outf(ks, grad_output, self, kernel_size, output_size, indices, grad_input)
  end_update_in_place(op.tensor);
  break;


case H_FRACTIONAL_MAX_POOL2D_BACKWARD:
  set(op.tensor, at::redispatch::fractional_max_pool2d_backward(ks, grad_output, self, kernel_size, output_size, indices));
  break;



case H_FRACTIONAL_MAX_POOL3D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::fractional_max_pool3d_backward_outf(ks, grad_output, self, kernel_size, output_size, indices, grad_input)
  end_update_in_place(op.tensor);
  break;


case H_FRACTIONAL_MAX_POOL3D_BACKWARD:
  set(op.tensor, at::redispatch::fractional_max_pool3d_backward(ks, grad_output, self, kernel_size, output_size, indices));
  break;



case H_MAX_POOL2D_WITH_INDICES_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::max_pool2d_with_indices_backward_outf(ks, grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices, grad_input)
  end_update_in_place(op.tensor);
  break;


case H_MAX_POOL2D_WITH_INDICES_BACKWARD:
  set(op.tensor, at::redispatch::max_pool2d_with_indices_backward(ks, grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices));
  break;



case H_MAX_POOL3D_WITH_INDICES_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::max_pool3d_with_indices_backward_outf(ks, grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices, grad_input)
  end_update_in_place(op.tensor);
  break;


case H_MAX_POOL3D_WITH_INDICES_BACKWARD:
  set(op.tensor, at::redispatch::max_pool3d_with_indices_backward(ks, grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices));
  break;

case H_MAX_UNPOOL2D_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::max_unpool2d_outf(ks, self, indices, output_size, out)
  end_update_in_place(op.tensor);
  break;


case H_MAX_UNPOOL2D:
  set(op.tensor, at::redispatch::max_unpool2d(ks, self, indices, output_size));
  break;

case H_MAX_UNPOOL2D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::max_unpool2d_backward_outf(ks, grad_output, self, indices, output_size, grad_input)
  end_update_in_place(op.tensor);
  break;


case H_MAX_UNPOOL2D_BACKWARD:
  set(op.tensor, at::redispatch::max_unpool2d_backward(ks, grad_output, self, indices, output_size));
  break;

case H_MAX_UNPOOL3D_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::max_unpool3d_outf(ks, self, indices, output_size, stride, padding, out)
  end_update_in_place(op.tensor);
  break;


case H_MAX_UNPOOL3D:
  set(op.tensor, at::redispatch::max_unpool3d(ks, self, indices, output_size, stride, padding));
  break;

case H_MAX_UNPOOL3D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::max_unpool3d_backward_outf(ks, grad_output, self, indices, output_size, stride, padding, grad_input)
  end_update_in_place(op.tensor);
  break;


case H_MAX_UNPOOL3D_BACKWARD:
  set(op.tensor, at::redispatch::max_unpool3d_backward(ks, grad_output, self, indices, output_size, stride, padding));
  break;

case H_REFLECTION_PAD1D_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::reflection_pad1d_outf(ks, self, padding, out)
  end_update_in_place(op.tensor);
  break;


case H_REFLECTION_PAD1D:
  set(op.tensor, at::redispatch::reflection_pad1d(ks, self, padding));
  break;

case H_REFLECTION_PAD1D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::reflection_pad1d_backward_outf(ks, grad_output, self, padding, grad_input)
  end_update_in_place(op.tensor);
  break;


case H_REFLECTION_PAD1D_BACKWARD:
  set(op.tensor, at::redispatch::reflection_pad1d_backward(ks, grad_output, self, padding));
  break;

case H_REFLECTION_PAD2D_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::reflection_pad2d_outf(ks, self, padding, out)
  end_update_in_place(op.tensor);
  break;


case H_REFLECTION_PAD2D:
  set(op.tensor, at::redispatch::reflection_pad2d(ks, self, padding));
  break;

case H_REFLECTION_PAD2D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::reflection_pad2d_backward_outf(ks, grad_output, self, padding, grad_input)
  end_update_in_place(op.tensor);
  break;


case H_REFLECTION_PAD2D_BACKWARD:
  set(op.tensor, at::redispatch::reflection_pad2d_backward(ks, grad_output, self, padding));
  break;

case H_REPLICATION_PAD1D_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::replication_pad1d_outf(ks, self, padding, out)
  end_update_in_place(op.tensor);
  break;

case H_REPLICATION_PAD1D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::replication_pad1d_backward_outf(ks, grad_output, self, padding, grad_input)
  end_update_in_place(op.tensor);
  break;

case H_REPLICATION_PAD2D_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::replication_pad2d_outf(ks, self, padding, out)
  end_update_in_place(op.tensor);
  break;

case H_REPLICATION_PAD2D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::replication_pad2d_backward_outf(ks, grad_output, self, padding, grad_input)
  end_update_in_place(op.tensor);
  break;


case H_REPLICATION_PAD2D_BACKWARD:
  set(op.tensor, at::redispatch::replication_pad2d_backward(ks, grad_output, self, padding));
  break;

case H_REPLICATION_PAD3D_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::replication_pad3d_outf(ks, self, padding, out)
  end_update_in_place(op.tensor);
  break;

case H_REPLICATION_PAD3D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::replication_pad3d_backward_outf(ks, grad_output, self, padding, grad_input)
  end_update_in_place(op.tensor);
  break;


case H_REPLICATION_PAD3D_BACKWARD:
  set(op.tensor, at::redispatch::replication_pad3d_backward(ks, grad_output, self, padding));
  break;


case H_UPSAMPLE_LINEAR1D_VEC:
  set(op.tensor, at::redispatch::upsample_linear1d(ks, input, output_size, align_corners, scale_factors));
  break;


case H_UPSAMPLE_LINEAR1D_BACKWARD_VEC:
  set(op.tensor, at::redispatch::upsample_linear1d_backward(ks, grad_output, output_size, input_size, align_corners, scale_factors));
  break;


case H_UPSAMPLE_BILINEAR2D_VEC:
  set(op.tensor, at::redispatch::upsample_bilinear2d(ks, input, output_size, align_corners, scale_factors));
  break;


case H_UPSAMPLE_BILINEAR2D_BACKWARD_VEC:
  set(op.tensor, at::redispatch::upsample_bilinear2d_backward(ks, grad_output, output_size, input_size, align_corners, scale_factors));
  break;


case H_UPSAMPLE_TRILINEAR3D_VEC:
  set(op.tensor, at::redispatch::upsample_trilinear3d(ks, input, output_size, align_corners, scale_factors));
  break;


case H_UPSAMPLE_TRILINEAR3D_BACKWARD_VEC:
  set(op.tensor, at::redispatch::upsample_trilinear3d_backward(ks, grad_output, output_size, input_size, align_corners, scale_factors));
  break;


case H_UPSAMPLE_BICUBIC2D_VEC:
  set(op.tensor, at::redispatch::upsample_bicubic2d(ks, input, output_size, align_corners, scale_factors));
  break;


case H_UPSAMPLE_BICUBIC2D_BACKWARD_VEC:
  set(op.tensor, at::redispatch::upsample_bicubic2d_backward(ks, grad_output, output_size, input_size, align_corners, scale_factors));
  break;


case H_UPSAMPLE_NEAREST1D_VEC:
  set(op.tensor, at::redispatch::upsample_nearest1d(ks, input, output_size, scale_factors));
  break;


case H_UPSAMPLE_NEAREST1D_BACKWARD_VEC:
  set(op.tensor, at::redispatch::upsample_nearest1d_backward(ks, grad_output, output_size, input_size, scale_factors));
  break;


case H_UPSAMPLE_NEAREST2D_VEC:
  set(op.tensor, at::redispatch::upsample_nearest2d(ks, input, output_size, scale_factors));
  break;


case H_UPSAMPLE_NEAREST2D_BACKWARD_VEC:
  set(op.tensor, at::redispatch::upsample_nearest2d_backward(ks, grad_output, output_size, input_size, scale_factors));
  break;


case H_UPSAMPLE_NEAREST3D_VEC:
  set(op.tensor, at::redispatch::upsample_nearest3d(ks, input, output_size, scale_factors));
  break;


case H_UPSAMPLE_NEAREST3D_BACKWARD_VEC:
  set(op.tensor, at::redispatch::upsample_nearest3d_backward(ks, grad_output, output_size, input_size, scale_factors));
  break;

case H_UPSAMPLE_LINEAR1D_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::upsample_linear1d_outf(ks, self, output_size, align_corners, scales, out)
  end_update_in_place(op.tensor);
  break;

case H_UPSAMPLE_LINEAR1D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::upsample_linear1d_backward_outf(ks, grad_output, output_size, input_size, align_corners, scales, grad_input)
  end_update_in_place(op.tensor);
  break;

case H_UPSAMPLE_BILINEAR2D_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::upsample_bilinear2d_outf(ks, self, output_size, align_corners, scales_h, scales_w, out)
  end_update_in_place(op.tensor);
  break;


case H_UPSAMPLE_BILINEAR2D:
  set(op.tensor, at::redispatch::upsample_bilinear2d(ks, self, output_size, align_corners, scales_h, scales_w));
  break;

case H_UPSAMPLE_BILINEAR2D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::upsample_bilinear2d_backward_outf(ks, grad_output, output_size, input_size, align_corners, scales_h, scales_w, grad_input)
  end_update_in_place(op.tensor);
  break;

case H_UPSAMPLE_BICUBIC2D_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::upsample_bicubic2d_outf(ks, self, output_size, align_corners, scales_h, scales_w, out)
  end_update_in_place(op.tensor);
  break;

case H_UPSAMPLE_BICUBIC2D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::upsample_bicubic2d_backward_outf(ks, grad_output, output_size, input_size, align_corners, scales_h, scales_w, grad_input)
  end_update_in_place(op.tensor);
  break;

case H_UPSAMPLE_TRILINEAR3D_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::upsample_trilinear3d_outf(ks, self, output_size, align_corners, scales_d, scales_h, scales_w, out)
  end_update_in_place(op.tensor);
  break;

case H_UPSAMPLE_TRILINEAR3D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::upsample_trilinear3d_backward_outf(ks, grad_output, output_size, input_size, align_corners, scales_d, scales_h, scales_w, grad_input)
  end_update_in_place(op.tensor);
  break;

case H_UPSAMPLE_NEAREST1D_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::upsample_nearest1d_outf(ks, self, output_size, scales, out)
  end_update_in_place(op.tensor);
  break;

case H_UPSAMPLE_NEAREST1D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::upsample_nearest1d_backward_outf(ks, grad_output, output_size, input_size, scales, grad_input)
  end_update_in_place(op.tensor);
  break;

case H_UPSAMPLE_NEAREST2D_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::upsample_nearest2d_outf(ks, self, output_size, scales_h, scales_w, out)
  end_update_in_place(op.tensor);
  break;


case H_UPSAMPLE_NEAREST2D:
  set(op.tensor, at::redispatch::upsample_nearest2d(ks, self, output_size, scales_h, scales_w));
  break;

case H_UPSAMPLE_NEAREST2D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::upsample_nearest2d_backward_outf(ks, grad_output, output_size, input_size, scales_h, scales_w, grad_input)
  end_update_in_place(op.tensor);
  break;

case H_UPSAMPLE_NEAREST3D_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::upsample_nearest3d_outf(ks, self, output_size, scales_d, scales_h, scales_w, out)
  end_update_in_place(op.tensor);
  break;


case H_UPSAMPLE_NEAREST3D:
  set(op.tensor, at::redispatch::upsample_nearest3d(ks, self, output_size, scales_d, scales_h, scales_w));
  break;

case H_UPSAMPLE_NEAREST3D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::upsample_nearest3d_backward_outf(ks, grad_output, output_size, input_size, scales_d, scales_h, scales_w, grad_input)
  end_update_in_place(op.tensor);
  break;

case H_SIGMOID_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::sigmoid_backward_outf(ks, grad_output, output, grad_input)
  end_update_in_place(op.tensor);
  break;


case H_SIGMOID_BACKWARD:
  set(op.tensor, at::redispatch::sigmoid_backward(ks, grad_output, output));
  break;

case H_LOGIT_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::logit_backward_outf(ks, grad_output, self, eps, grad_input)
  end_update_in_place(op.tensor);
  break;


case H_LOGIT_BACKWARD:
  set(op.tensor, at::redispatch::logit_backward(ks, grad_output, self, eps));
  break;

case H_TANH_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::tanh_backward_outf(ks, grad_output, output, grad_input)
  end_update_in_place(op.tensor);
  break;


case H_TANH_BACKWARD:
  set(op.tensor, at::redispatch::tanh_backward(ks, grad_output, output));
  break;

case H_SLOW_CONV_TRANSPOSE2D_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::slow_conv_transpose2d_outf(ks, self, weight, kernel_size, bias, stride, padding, output_padding, dilation, out)
  end_update_in_place(op.tensor);
  break;


case H_SLOW_CONV_TRANSPOSE2D:
  set(op.tensor, at::redispatch::slow_conv_transpose2d(ks, self, weight, kernel_size, bias, stride, padding, output_padding, dilation));
  break;



case H_SLOW_CONV_TRANSPOSE3D_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::slow_conv_transpose3d_outf(ks, self, weight, kernel_size, bias, stride, padding, output_padding, dilation, out)
  end_update_in_place(op.tensor);
  break;


case H_SLOW_CONV_TRANSPOSE3D:
  set(op.tensor, at::redispatch::slow_conv_transpose3d(ks, self, weight, kernel_size, bias, stride, padding, output_padding, dilation));
  break;



case H_THNN_CONV2D_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::thnn_conv2d_outf(ks, self, weight, kernel_size, bias, stride, padding, out)
  end_update_in_place(op.tensor);
  break;


case H_THNN_CONV2D:
  set(op.tensor, at::redispatch::thnn_conv2d(ks, self, weight, kernel_size, bias, stride, padding));
  break;





case H_THNN_CONV_DEPTHWISE2D_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::thnn_conv_depthwise2d_outf(ks, self, weight, kernel_size, bias, stride, padding, dilation, out)
  end_update_in_place(op.tensor);
  break;


case H_THNN_CONV_DEPTHWISE2D:
  set(op.tensor, at::redispatch::thnn_conv_depthwise2d(ks, self, weight, kernel_size, bias, stride, padding, dilation));
  break;

case H_THNN_CONV_DEPTHWISE2D_FORWARD_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::thnn_conv_depthwise2d_forward_outf(ks, self, weight, kernel_size, bias, stride, padding, dilation, out)
  end_update_in_place(op.tensor);
  break;


case H_THNN_CONV_DEPTHWISE2D_FORWARD:
  set(op.tensor, at::redispatch::thnn_conv_depthwise2d_forward(ks, self, weight, kernel_size, bias, stride, padding, dilation));
  break;




case H_CONV_DEPTHWISE3D:
  set(op.tensor, at::redispatch::conv_depthwise3d(ks, self, weight, kernel_size, bias, stride, padding, dilation));
  break;



case H_SLOW_CONV3D_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::slow_conv3d_outf(ks, self, weight, kernel_size, bias, stride, padding, out)
  end_update_in_place(op.tensor);
  break;


case H_SLOW_CONV3D:
  set(op.tensor, at::redispatch::slow_conv3d(ks, self, weight, kernel_size, bias, stride, padding));
  break;






case H_SLOW_CONV_DILATED2D:
  set(op.tensor, at::redispatch::slow_conv_dilated2d(ks, self, weight, kernel_size, bias, stride, padding, dilation));
  break;



case H_SLOW_CONV_DILATED3D:
  set(op.tensor, at::redispatch::slow_conv_dilated3d(ks, self, weight, kernel_size, bias, stride, padding, dilation));
  break;


case H_COL2IM_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::col2im_outf(ks, self, output_size, kernel_size, dilation, padding, stride, out)
  end_update_in_place(op.tensor);
  break;


case H_COL2IM:
  set(op.tensor, at::redispatch::col2im(ks, self, output_size, kernel_size, dilation, padding, stride));
  break;

case H_COL2IM_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::col2im_backward_outf(ks, grad_output, kernel_size, dilation, padding, stride, grad_input)
  end_update_in_place(op.tensor);
  break;


case H_COL2IM_BACKWARD:
  set(op.tensor, at::redispatch::col2im_backward(ks, grad_output, kernel_size, dilation, padding, stride));
  break;


case H_COLUMN_STACK:
  set(op.tensor, at::redispatch::column_stack(ks, tensors));
  break;

case H_COLUMN_STACK_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::column_stack_outf(ks, tensors, out)
  end_update_in_place(op.tensor);
  break;

case H_IM2COL_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::im2col_outf(ks, self, kernel_size, dilation, padding, stride, out)
  end_update_in_place(op.tensor);
  break;


case H_IM2COL:
  set(op.tensor, at::redispatch::im2col(ks, self, kernel_size, dilation, padding, stride));
  break;

case H_IM2COL_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::im2col_backward_outf(ks, grad_output, input_size, kernel_size, dilation, padding, stride, grad_input)
  end_update_in_place(op.tensor);
  break;


case H_IM2COL_BACKWARD:
  set(op.tensor, at::redispatch::im2col_backward(ks, grad_output, input_size, kernel_size, dilation, padding, stride));
  break;


case H_ISFINITE:
  set(op.tensor, at::redispatch::isfinite(ks, self));
  break;


case H_ISINF:
  set(op.tensor, at::redispatch::isinf(ks, self));
  break;



case H_ISPOSINF:
  set(op.tensor, at::redispatch::isposinf(ks, self));
  break;

case H_ISPOSINF_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::isposinf_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_ISNEGINF:
  set(op.tensor, at::redispatch::isneginf(ks, self));
  break;

case H_ISNEGINF_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::isneginf_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H__ADD_BATCH_DIM:
  set(op.tensor, at::redispatch::_add_batch_dim(ks, self, batch_dim, level));
  break;


case H__REMOVE_BATCH_DIM:
  set(op.tensor, at::redispatch::_remove_batch_dim(ks, self, level, batch_size, out_dim));
  break;

case H_SPECIAL_ENTR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::special_entr_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_SPECIAL_EXPM1:
  set(op.tensor, at::redispatch::special_expm1(ks, self));
  break;

case H_SPECIAL_EXPM1_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::special_expm1_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_SPECIAL_EXP2:
  set(op.tensor, at::redispatch::special_exp2(ks, self));
  break;

case H_SPECIAL_EXP2_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::special_exp2_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_SPECIAL_GAMMALN:
  set(op.tensor, at::redispatch::special_gammaln(ks, self));
  break;

case H_SPECIAL_GAMMALN_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::special_gammaln_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_SPECIAL_ERF:
  set(op.tensor, at::redispatch::special_erf(ks, self));
  break;

case H_SPECIAL_ERF_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::special_erf_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_SPECIAL_ERFC:
  set(op.tensor, at::redispatch::special_erfc(ks, self));
  break;

case H_SPECIAL_ERFC_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::special_erfc_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_SPECIAL_ERFINV:
  set(op.tensor, at::redispatch::special_erfinv(ks, self));
  break;

case H_SPECIAL_ERFINV_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::special_erfinv_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;

case H_SPECIAL_I0E_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::special_i0e_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_SPECIAL_LOGIT:
  set(op.tensor, at::redispatch::special_logit(ks, self, eps));
  break;

case H_SPECIAL_LOGIT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::special_logit_outf(ks, self, eps, out)
  end_update_in_place(op.tensor);
  break;


case H_SPECIAL_EXPIT:
  set(op.tensor, at::redispatch::special_expit(ks, self));
  break;

case H_SPECIAL_EXPIT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::special_expit_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_FFT_FFT:
  set(op.tensor, at::redispatch::fft_fft(ks, self, n, dim, norm));
  break;

case H_FFT_FFT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::fft_fft_outf(ks, self, n, dim, norm, out)
  end_update_in_place(op.tensor);
  break;


case H_FFT_IFFT:
  set(op.tensor, at::redispatch::fft_ifft(ks, self, n, dim, norm));
  break;

case H_FFT_IFFT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::fft_ifft_outf(ks, self, n, dim, norm, out)
  end_update_in_place(op.tensor);
  break;


case H_FFT_RFFT:
  set(op.tensor, at::redispatch::fft_rfft(ks, self, n, dim, norm));
  break;

case H_FFT_RFFT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::fft_rfft_outf(ks, self, n, dim, norm, out)
  end_update_in_place(op.tensor);
  break;


case H_FFT_IRFFT:
  set(op.tensor, at::redispatch::fft_irfft(ks, self, n, dim, norm));
  break;

case H_FFT_IRFFT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::fft_irfft_outf(ks, self, n, dim, norm, out)
  end_update_in_place(op.tensor);
  break;


case H_FFT_HFFT:
  set(op.tensor, at::redispatch::fft_hfft(ks, self, n, dim, norm));
  break;

case H_FFT_HFFT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::fft_hfft_outf(ks, self, n, dim, norm, out)
  end_update_in_place(op.tensor);
  break;


case H_FFT_IHFFT:
  set(op.tensor, at::redispatch::fft_ihfft(ks, self, n, dim, norm));
  break;

case H_FFT_IHFFT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::fft_ihfft_outf(ks, self, n, dim, norm, out)
  end_update_in_place(op.tensor);
  break;


case H_FFT_FFT2:
  set(op.tensor, at::redispatch::fft_fft2(ks, self, s, dim, norm));
  break;

case H_FFT_FFT2_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::fft_fft2_outf(ks, self, s, dim, norm, out)
  end_update_in_place(op.tensor);
  break;


case H_FFT_IFFT2:
  set(op.tensor, at::redispatch::fft_ifft2(ks, self, s, dim, norm));
  break;

case H_FFT_IFFT2_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::fft_ifft2_outf(ks, self, s, dim, norm, out)
  end_update_in_place(op.tensor);
  break;


case H_FFT_RFFT2:
  set(op.tensor, at::redispatch::fft_rfft2(ks, self, s, dim, norm));
  break;

case H_FFT_RFFT2_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::fft_rfft2_outf(ks, self, s, dim, norm, out)
  end_update_in_place(op.tensor);
  break;


case H_FFT_IRFFT2:
  set(op.tensor, at::redispatch::fft_irfft2(ks, self, s, dim, norm));
  break;

case H_FFT_IRFFT2_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::fft_irfft2_outf(ks, self, s, dim, norm, out)
  end_update_in_place(op.tensor);
  break;


case H_FFT_FFTN:
  set(op.tensor, at::redispatch::fft_fftn(ks, self, s, dim, norm));
  break;

case H_FFT_FFTN_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::fft_fftn_outf(ks, self, s, dim, norm, out)
  end_update_in_place(op.tensor);
  break;


case H_FFT_IFFTN:
  set(op.tensor, at::redispatch::fft_ifftn(ks, self, s, dim, norm));
  break;

case H_FFT_IFFTN_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::fft_ifftn_outf(ks, self, s, dim, norm, out)
  end_update_in_place(op.tensor);
  break;


case H_FFT_RFFTN:
  set(op.tensor, at::redispatch::fft_rfftn(ks, self, s, dim, norm));
  break;

case H_FFT_RFFTN_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::fft_rfftn_outf(ks, self, s, dim, norm, out)
  end_update_in_place(op.tensor);
  break;


case H_FFT_IRFFTN:
  set(op.tensor, at::redispatch::fft_irfftn(ks, self, s, dim, norm));
  break;

case H_FFT_IRFFTN_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::fft_irfftn_outf(ks, self, s, dim, norm, out)
  end_update_in_place(op.tensor);
  break;


case H_FFT_FFTFREQ:
  set(op.tensor, at::redispatch::fft_fftfreq(ks, n, d, dtype, layout, device, pin_memory));
  break;

case H_FFT_FFTFREQ_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::fft_fftfreq_outf(ks, n, d, out)
  end_update_in_place(op.tensor);
  break;


case H_FFT_RFFTFREQ:
  set(op.tensor, at::redispatch::fft_rfftfreq(ks, n, d, dtype, layout, device, pin_memory));
  break;

case H_FFT_RFFTFREQ_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::fft_rfftfreq_outf(ks, n, d, out)
  end_update_in_place(op.tensor);
  break;


case H_FFT_FFTSHIFT:
  set(op.tensor, at::redispatch::fft_fftshift(ks, self, dim));
  break;


case H_FFT_IFFTSHIFT:
  set(op.tensor, at::redispatch::fft_ifftshift(ks, self, dim));
  break;


case H_LINALG_CHOLESKY:
  set(op.tensor, at::redispatch::linalg_cholesky(ks, self));
  break;

case H_LINALG_CHOLESKY_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::linalg_cholesky_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_LINALG_DET:
  set(op.tensor, at::redispatch::linalg_det(ks, self));
  break;

case H_LINALG_DET_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::linalg_det_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_DET:
  set(op.tensor, at::redispatch::det(ks, self));
  break;



case H__LSTSQ_HELPER_:
  init_update_in_place(op.tensor);
  at::redispatch::_lstsq_helper_(ks, self, rank, singular_values, infos, a, cond, driver_name)
  end_update_in_place(op.tensor);
  break;






case H_LINALG_EIGVALS:
  set(op.tensor, at::redispatch::linalg_eigvals(ks, self));
  break;

case H_LINALG_EIGVALS_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::linalg_eigvals_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;




case H_LINALG_EIGVALSH:
  set(op.tensor, at::redispatch::linalg_eigvalsh(ks, self, UPLO));
  break;

case H_LINALG_EIGVALSH_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::linalg_eigvalsh_outf(ks, self, UPLO, out)
  end_update_in_place(op.tensor);
  break;


case H_LINALG_HOUSEHOLDER_PRODUCT:
  set(op.tensor, at::redispatch::linalg_householder_product(ks, input, tau));
  break;

case H_LINALG_HOUSEHOLDER_PRODUCT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::linalg_householder_product_outf(ks, input, tau, out)
  end_update_in_place(op.tensor);
  break;

case H__LINALG_INV_OUT_HELPER_:
  init_update_in_place(op.tensor);
  at::redispatch::_linalg_inv_out_helper_(ks, self, infos_lu, infos_getri)
  end_update_in_place(op.tensor);
  break;


case H_LINALG_INV:
  set(op.tensor, at::redispatch::linalg_inv(ks, self));
  break;

case H_LINALG_INV_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::linalg_inv_outf(ks, self, out)
  end_update_in_place(op.tensor);
  break;


case H_INNER:
  set(op.tensor, at::redispatch::inner(ks, self, other));
  break;

case H_INNER_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::inner_outf(ks, self, other, out)
  end_update_in_place(op.tensor);
  break;


case H_OUTER:
  set(op.tensor, at::redispatch::outer(ks, self, vec2));
  break;

case H_OUTER_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::outer_outf(ks, self, vec2, out)
  end_update_in_place(op.tensor);
  break;


case H_GER:
  set(op.tensor, at::redispatch::ger(ks, self, vec2));
  break;

case H_GER_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::ger_outf(ks, self, vec2, out)
  end_update_in_place(op.tensor);
  break;


case H_LINALG_NORM:
  set(op.tensor, at::redispatch::linalg_norm(ks, self, ord, dim, keepdim, dtype));
  break;


case H_LINALG_NORM_ORD_STR:
  set(op.tensor, at::redispatch::linalg_norm(ks, self, ord, dim, keepdim, dtype));
  break;

case H_LINALG_NORM_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::linalg_norm_outf(ks, self, ord, dim, keepdim, dtype, out)
  end_update_in_place(op.tensor);
  break;

case H_LINALG_NORM_ORD_STR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::linalg_norm_outf(ks, self, ord, dim, keepdim, dtype, out)
  end_update_in_place(op.tensor);
  break;


case H_LINALG_VECTOR_NORM:
  set(op.tensor, at::redispatch::linalg_vector_norm(ks, self, ord, dim, keepdim, dtype));
  break;

case H_LINALG_VECTOR_NORM_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::linalg_vector_norm_outf(ks, self, ord, dim, keepdim, dtype, out)
  end_update_in_place(op.tensor);
  break;




case H_LINALG_COND:
  set(op.tensor, at::redispatch::linalg_cond(ks, self, p));
  break;

case H_LINALG_COND_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::linalg_cond_outf(ks, self, p, out)
  end_update_in_place(op.tensor);
  break;


case H_LINALG_COND_P_STR:
  set(op.tensor, at::redispatch::linalg_cond(ks, self, p));
  break;

case H_LINALG_COND_P_STR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::linalg_cond_outf(ks, self, p, out)
  end_update_in_place(op.tensor);
  break;


case H_LINALG_PINV:
  set(op.tensor, at::redispatch::linalg_pinv(ks, self, rcond, hermitian));
  break;


case H_LINALG_PINV_RCOND_TENSOR:
  set(op.tensor, at::redispatch::linalg_pinv(ks, self, rcond, hermitian));
  break;

case H_LINALG_PINV_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::linalg_pinv_outf(ks, self, rcond, hermitian, out)
  end_update_in_place(op.tensor);
  break;

case H_LINALG_PINV_OUT_RCOND_TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::linalg_pinv_outf(ks, self, rcond, hermitian, out)
  end_update_in_place(op.tensor);
  break;

case H__LINALG_SOLVE_OUT_HELPER_:
  init_update_in_place(op.tensor);
  at::redispatch::_linalg_solve_out_helper_(ks, self, other, infos)
  end_update_in_place(op.tensor);
  break;


case H_LINALG_SOLVE:
  set(op.tensor, at::redispatch::linalg_solve(ks, input, other));
  break;

case H_LINALG_SOLVE_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::linalg_solve_outf(ks, input, other, out)
  end_update_in_place(op.tensor);
  break;


case H_LINALG_TENSORINV:
  set(op.tensor, at::redispatch::linalg_tensorinv(ks, self, ind));
  break;

case H_LINALG_TENSORINV_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::linalg_tensorinv_outf(ks, self, ind, out)
  end_update_in_place(op.tensor);
  break;


case H_LINALG_TENSORSOLVE:
  set(op.tensor, at::redispatch::linalg_tensorsolve(ks, self, other, dims));
  break;

case H_LINALG_TENSORSOLVE_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::linalg_tensorsolve_outf(ks, self, other, dims, out)
  end_update_in_place(op.tensor);
  break;





case H_LINALG_MATRIX_POWER:
  set(op.tensor, at::redispatch::linalg_matrix_power(ks, self, n));
  break;

case H_LINALG_MATRIX_POWER_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::linalg_matrix_power_outf(ks, self, n, out)
  end_update_in_place(op.tensor);
  break;


case H_LINALG_MATRIX_RANK:
  set(op.tensor, at::redispatch::linalg_matrix_rank(ks, self, tol, hermitian));
  break;

case H_LINALG_MATRIX_RANK_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::linalg_matrix_rank_outf(ks, self, tol, hermitian, out)
  end_update_in_place(op.tensor);
  break;


case H_LINALG_MULTI_DOT:
  set(op.tensor, at::redispatch::linalg_multi_dot(ks, tensors));
  break;

case H_LINALG_MULTI_DOT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::linalg_multi_dot_outf(ks, tensors, out)
  end_update_in_place(op.tensor);
  break;


case H__TEST_SERIALIZATION_SUBCMUL:
  set(op.tensor, at::redispatch::_test_serialization_subcmul(ks, self, other, alpha));
  break;


case H__TEST_OPTIONAL_INTLIST:
  set(op.tensor, at::redispatch::_test_optional_intlist(ks, values, addends));
  break;


case H__TEST_OPTIONAL_FILLED_INTLIST:
  set(op.tensor, at::redispatch::_test_optional_filled_intlist(ks, values, addends));
  break;


case H__TEST_OPTIONAL_FLOATLIST:
  set(op.tensor, at::redispatch::_test_optional_floatlist(ks, values, addends));
  break;


case H__TEST_STRING_DEFAULT:
  set(op.tensor, at::redispatch::_test_string_default(ks, dummy, a, b));
  break;


case H__TEST_AMBIGUOUS_DEFAULTS_A:
  set(op.tensor, at::redispatch::_test_ambiguous_defaults(ks, dummy, a, b));
  break;


case H__TEST_AMBIGUOUS_DEFAULTS_B:
  set(op.tensor, at::redispatch::_test_ambiguous_defaults(ks, dummy, a, b));
  break;


case H_SEGMENT_REDUCE:
  set(op.tensor, at::redispatch::segment_reduce(ks, data, reduce, lengths, indices, axis, unsafe));
  break;

