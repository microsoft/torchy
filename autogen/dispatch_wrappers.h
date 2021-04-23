
Tensor wrap__cast_Byte(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_cast_Byte(self, non_blocking);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__CAST_BYTE, self, non_blocking);
}

Tensor wrap__cast_Char(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_cast_Char(self, non_blocking);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__CAST_CHAR, self, non_blocking);
}

Tensor wrap__cast_Double(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_cast_Double(self, non_blocking);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__CAST_DOUBLE, self, non_blocking);
}

Tensor wrap__cast_Float(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_cast_Float(self, non_blocking);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__CAST_FLOAT, self, non_blocking);
}

Tensor wrap__cast_Int(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_cast_Int(self, non_blocking);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__CAST_INT, self, non_blocking);
}

Tensor wrap__cast_Long(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_cast_Long(self, non_blocking);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__CAST_LONG, self, non_blocking);
}

Tensor wrap__cast_Short(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_cast_Short(self, non_blocking);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__CAST_SHORT, self, non_blocking);
}

Tensor wrap__cast_Half(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_cast_Half(self, non_blocking);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__CAST_HALF, self, non_blocking);
}

Tensor wrap__fw_primal(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_fw_primal(self, level);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__FW_PRIMAL, self, level);
}

Tensor wrap__make_dual(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(primal, tangent);
    return at::redispatch::_make_dual(primal, tangent, level);
  }
  return MK_TORCHY(primal.dtype(), primal.device(), H__MAKE_DUAL, primal, tangent, level);
}

Tensor wrap__unpack_dual(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(dual);
    return at::redispatch::_unpack_dual(dual, level);
  }
  return MK_TORCHY(dual.dtype(), dual.device(), H__UNPACK_DUAL, dual, level);
}

Tensor wrap_rename_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::rename_(self, names);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RENAME_, self, names);
}

Tensor wrap_rename(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::rename(self, names);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RENAME, self, names);
}

Tensor wrap_align_to(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::align_to(self, names);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ALIGN_TO, self, names);
}

Tensor wrap_align_to_ellipsis_idx(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::align_to(self, order, ellipsis_idx);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ALIGN_TO_ELLIPSIS_IDX, self, order, ellipsis_idx);
}

Tensor wrap_align_as(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::align_as(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ALIGN_AS, self, other);
}

Tensor[] wrap_align_tensors(args...) {
  ensure_materialized();
  return at::redispatch::align_tensors(tensors);
}

void wrap__assert_async(args...) {
  ensure_materialized(self);
  return at::redispatch::_assert_async(self);
}

Tensor wrap_refine_names(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::refine_names(self, names);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_REFINE_NAMES, self, names);
}

bool wrap__use_cudnn_ctc_loss(args...) {
  ensure_materialized(log_probs, targets);
  return at::redispatch::_use_cudnn_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank);
}

Tensor wrap__cudnn_ctc_loss(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(log_probs, targets);
    return at::redispatch::_cudnn_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, deterministic, zero_infinity);
  }
  return MK_TORCHY(log_probs.dtype(), log_probs.device(), H__CUDNN_CTC_LOSS, log_probs, targets, input_lengths, target_lengths, blank, deterministic, zero_infinity);
}

bool wrap__use_cudnn_rnn_flatten_weight(args...) {
  ensure_materialized();
  return at::redispatch::_use_cudnn_rnn_flatten_weight();
}

Tensor wrap__cudnn_rnn_flatten_weight(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::_cudnn_rnn_flatten_weight(weight_arr, weight_stride0, input_size, mode, hidden_size, proj_size, num_layers, batch_first, bidirectional);
  }
  return MK_TORCHY(None, None, H__CUDNN_RNN_FLATTEN_WEIGHT, weight_arr, weight_stride0, input_size, mode, hidden_size, proj_size, num_layers, batch_first, bidirectional);
}

Tensor wrap__cudnn_rnn(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, hx);
    return at::redispatch::_cudnn_rnn(input, weight, weight_stride0, weight_buf, hx, cx, mode, hidden_size, proj_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state);
  }
  return MK_TORCHY(input.dtype(), input.device(), H__CUDNN_RNN, input, weight, weight_stride0, weight_buf, hx, cx, mode, hidden_size, proj_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state);
}

Tensor wrap__cudnn_rnn_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, weight_buf, hx, output, reserve);
    return at::redispatch::_cudnn_rnn_backward(input, weight, weight_stride0, weight_buf, hx, cx, output, grad_output, grad_hy, grad_cy, mode, hidden_size, proj_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state, reserve, output_mask);
  }
  return MK_TORCHY(input.dtype(), input.device(), H__CUDNN_RNN_BACKWARD, input, weight, weight_stride0, weight_buf, hx, cx, output, grad_output, grad_hy, grad_cy, mode, hidden_size, proj_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state, reserve, output_mask);
}

Tensor wrap__cudnn_init_dropout_state(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::_cudnn_init_dropout_state(dropout, train, dropout_seed, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H__CUDNN_INIT_DROPOUT_STATE, dropout, train, dropout_seed, dtype, layout, device, pin_memory);
}

int wrap__debug_has_internal_overlap(args...) {
  ensure_materialized(self);
  return at::redispatch::_debug_has_internal_overlap(self);
}

Tensor wrap__fused_dropout(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_fused_dropout(self, p, generator);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__FUSED_DROPOUT, self, p, generator);
}

Tensor wrap__masked_scale(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, mask);
    return at::redispatch::_masked_scale(self, mask, scale);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__MASKED_SCALE, self, mask, scale);
}

Tensor wrap__sobol_engine_draw(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(quasi, sobolstate);
    return at::redispatch::_sobol_engine_draw(quasi, n, sobolstate, dimension, num_generated, dtype);
  }
  return MK_TORCHY(quasi.dtype(), quasi.device(), H__SOBOL_ENGINE_DRAW, quasi, n, sobolstate, dimension, num_generated, dtype);
}

Tensor wrap__sobol_engine_ff_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, sobolstate);
    return at::redispatch::_sobol_engine_ff_(self, n, sobolstate, dimension, num_generated);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__SOBOL_ENGINE_FF_, self, n, sobolstate, dimension, num_generated);
}

Tensor wrap__sobol_engine_scramble_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, ltm);
    return at::redispatch::_sobol_engine_scramble_(self, ltm, dimension);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__SOBOL_ENGINE_SCRAMBLE_, self, ltm, dimension);
}

Tensor wrap__sobol_engine_initialize_state_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_sobol_engine_initialize_state_(self, dimension);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__SOBOL_ENGINE_INITIALIZE_STATE_, self, dimension);
}

Tensor wrap__reshape_from_tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, shape);
    return at::redispatch::_reshape_from_tensor(self, shape);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__RESHAPE_FROM_TENSOR, self, shape);
}

Tensor wrap__shape_as_tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_shape_as_tensor(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__SHAPE_AS_TENSOR, self);
}

Tensor wrap_dropout(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input);
    return at::redispatch::dropout(input, p, train);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_DROPOUT, input, p, train);
}

Tensor wrap_dropout_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::dropout_(self, p, train);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DROPOUT_, self, p, train);
}

Tensor wrap_feature_dropout(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input);
    return at::redispatch::feature_dropout(input, p, train);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_FEATURE_DROPOUT, input, p, train);
}

Tensor wrap_feature_dropout_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::feature_dropout_(self, p, train);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FEATURE_DROPOUT_, self, p, train);
}

Tensor wrap_alpha_dropout(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input);
    return at::redispatch::alpha_dropout(input, p, train);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_ALPHA_DROPOUT, input, p, train);
}

Tensor wrap_alpha_dropout_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::alpha_dropout_(self, p, train);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ALPHA_DROPOUT_, self, p, train);
}

Tensor wrap_feature_alpha_dropout(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input);
    return at::redispatch::feature_alpha_dropout(input, p, train);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_FEATURE_ALPHA_DROPOUT, input, p, train);
}

Tensor wrap_feature_alpha_dropout_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::feature_alpha_dropout_(self, p, train);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FEATURE_ALPHA_DROPOUT_, self, p, train);
}

Tensor wrap_abs(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::abs(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ABS, self);
}

Tensor wrap_abs_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::abs_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ABS_, self);
}

Tensor wrap_abs_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::abs(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ABS_OUT, out, self);
}

Tensor wrap_absolute(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::absolute(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ABSOLUTE, self);
}

Tensor wrap_absolute_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::absolute_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ABSOLUTE_, self);
}

Tensor wrap_absolute_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::absolute(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ABSOLUTE_OUT, out, self);
}

Tensor wrap_angle(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::angle(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ANGLE, self);
}

Tensor wrap_angle_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::angle(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ANGLE_OUT, out, self);
}

Tensor wrap_view_as_real(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::view_as_real(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_VIEW_AS_REAL, self);
}

Tensor wrap_view_as_complex(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::view_as_complex(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_VIEW_AS_COMPLEX, self);
}

Tensor wrap_sgn(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::sgn(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SGN, self);
}

Tensor wrap_sgn_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::sgn_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SGN_, self);
}

Tensor wrap_sgn_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::sgn(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SGN_OUT, out, self);
}

Tensor wrap_real(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::real(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_REAL, self);
}

Tensor wrap_imag(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::imag(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_IMAG, self);
}

Tensor wrap_conj(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::conj(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CONJ, self);
}

Tensor wrap_conj_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::conj(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_CONJ_OUT, out, self);
}

Tensor wrap__conj(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_conj(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__CONJ, self);
}

Tensor wrap_acos_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::acos(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ACOS_OUT, out, self);
}

Tensor wrap_arccos(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::arccos(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ARCCOS, self);
}

Tensor wrap_arccos_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::arccos_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ARCCOS_, self);
}

Tensor wrap_arccos_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::arccos(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ARCCOS_OUT, out, self);
}

Tensor wrap_avg_pool1d(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::avg_pool1d(self, kernel_size, stride, padding, ceil_mode, count_include_pad);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_AVG_POOL1D, self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}

Tensor wrap_adaptive_avg_pool1d(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::adaptive_avg_pool1d(self, output_size);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ADAPTIVE_AVG_POOL1D, self, output_size);
}

Tensor wrap_adaptive_max_pool1d(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::adaptive_max_pool1d(self, output_size);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ADAPTIVE_MAX_POOL1D, self, output_size);
}

Tensor wrap_add_Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::add(self, other, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ADD_TENSOR, self, other, alpha);
}

Tensor wrap_add__Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::add_(self, other, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ADD__TENSOR, self, other, alpha);
}

Tensor wrap_add_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::add(out, self, other, alpha);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ADD_OUT, out, self, other, alpha);
}

Tensor wrap__add_relu_Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::_add_relu(self, other, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__ADD_RELU_TENSOR, self, other, alpha);
}

Tensor wrap__add_relu__Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::_add_relu_(self, other, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__ADD_RELU__TENSOR, self, other, alpha);
}

Tensor wrap__add_relu_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::_add_relu(out, self, other, alpha);
  }
  return MK_TORCHY(out.dtype(), out.device(), H__ADD_RELU_OUT, out, self, other, alpha);
}

Tensor wrap_add_Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::add(self, other, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ADD_SCALAR, self, other, alpha);
}

Tensor wrap_add__Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::add_(self, other, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ADD__SCALAR, self, other, alpha);
}

Tensor wrap_addmv_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, mat, vec);
    return at::redispatch::addmv(out, self, mat, vec, beta, alpha);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ADDMV_OUT, out, self, mat, vec, beta, alpha);
}

Tensor wrap_addr(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, vec1, vec2);
    return at::redispatch::addr(self, vec1, vec2, beta, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ADDR, self, vec1, vec2, beta, alpha);
}

Tensor wrap_addr_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, vec1, vec2);
    return at::redispatch::addr_(self, vec1, vec2, beta, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ADDR_, self, vec1, vec2, beta, alpha);
}

Tensor wrap_addr_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, vec1, vec2);
    return at::redispatch::addr(out, self, vec1, vec2, beta, alpha);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ADDR_OUT, out, self, vec1, vec2, beta, alpha);
}

Tensor wrap_affine_grid_generator(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(theta);
    return at::redispatch::affine_grid_generator(theta, size, align_corners);
  }
  return MK_TORCHY(theta.dtype(), theta.device(), H_AFFINE_GRID_GENERATOR, theta, size, align_corners);
}

Tensor wrap_affine_grid_generator_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad);
    return at::redispatch::affine_grid_generator_backward(grad, size, align_corners);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H_AFFINE_GRID_GENERATOR_BACKWARD, grad, size, align_corners);
}

Tensor wrap_all_dim(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::all(self, dim, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ALL_DIM, self, dim, keepdim);
}

Tensor wrap_all_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::all(out, self, dim, keepdim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ALL_OUT, out, self, dim, keepdim);
}

Tensor wrap_all_dimname(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::all(self, dim, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ALL_DIMNAME, self, dim, keepdim);
}

Tensor wrap_all_dimname_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::all(out, self, dim, keepdim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ALL_DIMNAME_OUT, out, self, dim, keepdim);
}

bool wrap_allclose(args...) {
  ensure_materialized(self, other);
  return at::redispatch::allclose(self, other, rtol, atol, equal_nan);
}

Tensor wrap_any_dim(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::any(self, dim, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ANY_DIM, self, dim, keepdim);
}

Tensor wrap_any_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::any(out, self, dim, keepdim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ANY_OUT, out, self, dim, keepdim);
}

Tensor wrap_any_dimname(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::any(self, dim, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ANY_DIMNAME, self, dim, keepdim);
}

Tensor wrap_any_dimname_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::any(out, self, dim, keepdim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ANY_DIMNAME_OUT, out, self, dim, keepdim);
}

Tensor wrap_arange(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::arange(end, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_ARANGE, end, dtype, layout, device, pin_memory);
}

Tensor wrap_arange_start(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::arange(start, end, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_ARANGE_START, start, end, dtype, layout, device, pin_memory);
}

Tensor wrap_arange_start_step(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::arange(start, end, step, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_ARANGE_START_STEP, start, end, step, dtype, layout, device, pin_memory);
}

Tensor wrap_arange_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::arange(out, end);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ARANGE_OUT, out, end);
}

Tensor wrap_arange_start_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::arange(out, start, end, step);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ARANGE_START_OUT, out, start, end, step);
}

Tensor wrap__dim_arange(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(like);
    return at::redispatch::_dim_arange(like, dim);
  }
  return MK_TORCHY(like.dtype(), like.device(), H__DIM_ARANGE, like, dim);
}

Tensor wrap_argmax(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::argmax(self, dim, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ARGMAX, self, dim, keepdim);
}

Tensor wrap_argmax_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::argmax(out, self, dim, keepdim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ARGMAX_OUT, out, self, dim, keepdim);
}

Tensor wrap_argmin(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::argmin(self, dim, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ARGMIN, self, dim, keepdim);
}

Tensor wrap_argmin_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::argmin(out, self, dim, keepdim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ARGMIN_OUT, out, self, dim, keepdim);
}

Tensor wrap_acosh_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::acosh(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ACOSH_OUT, out, self);
}

Tensor wrap_arccosh(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::arccosh(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ARCCOSH, self);
}

Tensor wrap_arccosh_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::arccosh_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ARCCOSH_, self);
}

Tensor wrap_arccosh_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::arccosh(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ARCCOSH_OUT, out, self);
}

Tensor wrap_asinh_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::asinh(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ASINH_OUT, out, self);
}

Tensor wrap_arcsinh(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::arcsinh(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ARCSINH, self);
}

Tensor wrap_arcsinh_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::arcsinh_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ARCSINH_, self);
}

Tensor wrap_arcsinh_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::arcsinh(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ARCSINH_OUT, out, self);
}

Tensor wrap_atanh_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::atanh(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ATANH_OUT, out, self);
}

Tensor wrap_arctanh(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::arctanh(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ARCTANH, self);
}

Tensor wrap_arctanh_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::arctanh_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ARCTANH_, self);
}

Tensor wrap_arctanh_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::arctanh(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ARCTANH_OUT, out, self);
}

Tensor wrap_as_strided(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::as_strided(self, size, stride, storage_offset);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_AS_STRIDED, self, size, stride, storage_offset);
}

Tensor wrap_as_strided_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::as_strided_(self, size, stride, storage_offset);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_AS_STRIDED_, self, size, stride, storage_offset);
}

Tensor wrap_asin(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::asin(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ASIN, self);
}

Tensor wrap_asin_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::asin_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ASIN_, self);
}

Tensor wrap_asin_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::asin(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ASIN_OUT, out, self);
}

Tensor wrap_arcsin(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::arcsin(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ARCSIN, self);
}

Tensor wrap_arcsin_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::arcsin_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ARCSIN_, self);
}

Tensor wrap_arcsin_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::arcsin(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ARCSIN_OUT, out, self);
}

Tensor wrap_atan_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::atan(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ATAN_OUT, out, self);
}

Tensor wrap_arctan(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::arctan(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ARCTAN, self);
}

Tensor wrap_arctan_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::arctan_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ARCTAN_, self);
}

Tensor wrap_arctan_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::arctan(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ARCTAN_OUT, out, self);
}

Tensor wrap_atleast_1d(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::atleast_1d(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ATLEAST_1D, self);
}

Tensor[] wrap_atleast_1d_Sequence(args...) {
  ensure_materialized();
  return at::redispatch::atleast_1d(tensors);
}

Tensor wrap_atleast_2d(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::atleast_2d(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ATLEAST_2D, self);
}

Tensor[] wrap_atleast_2d_Sequence(args...) {
  ensure_materialized();
  return at::redispatch::atleast_2d(tensors);
}

Tensor wrap_atleast_3d(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::atleast_3d(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ATLEAST_3D, self);
}

Tensor[] wrap_atleast_3d_Sequence(args...) {
  ensure_materialized();
  return at::redispatch::atleast_3d(tensors);
}

Tensor wrap_baddbmm(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, batch1, batch2);
    return at::redispatch::baddbmm(self, batch1, batch2, beta, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BADDBMM, self, batch1, batch2, beta, alpha);
}

Tensor wrap_baddbmm_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, batch1, batch2);
    return at::redispatch::baddbmm_(self, batch1, batch2, beta, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BADDBMM_, self, batch1, batch2, beta, alpha);
}

Tensor wrap__baddbmm_mkl_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, batch1, batch2);
    return at::redispatch::_baddbmm_mkl_(self, batch1, batch2, beta, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__BADDBMM_MKL_, self, batch1, batch2, beta, alpha);
}

Tensor wrap_baddbmm_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, batch1, batch2);
    return at::redispatch::baddbmm(out, self, batch1, batch2, beta, alpha);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_BADDBMM_OUT, out, self, batch1, batch2, beta, alpha);
}

Tensor wrap_bartlett_window(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::bartlett_window(window_length, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_BARTLETT_WINDOW, window_length, dtype, layout, device, pin_memory);
}

Tensor wrap_bartlett_window_periodic(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::bartlett_window(window_length, periodic, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_BARTLETT_WINDOW_PERIODIC, window_length, periodic, dtype, layout, device, pin_memory);
}

Tensor wrap_batch_norm(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input);
    return at::redispatch::batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_BATCH_NORM, input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
}

Tensor wrap_quantized_batch_norm(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, mean, var);
    return at::redispatch::quantized_batch_norm(input, weight, bias, mean, var, eps, output_scale, output_zero_point);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_QUANTIZED_BATCH_NORM, input, weight, bias, mean, var, eps, output_scale, output_zero_point);
}

Tensor wrap__batch_norm_impl_index(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input);
    return at::redispatch::_batch_norm_impl_index(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
  }
  return MK_TORCHY(input.dtype(), input.device(), H__BATCH_NORM_IMPL_INDEX, input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
}

Tensor wrap__batch_norm_impl_index_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, grad_output, reservedSpace);
    return at::redispatch::_batch_norm_impl_index_backward(impl_index, input, grad_output, weight, running_mean, running_var, save_mean, save_var_transform, train, eps, output_mask, reservedSpace);
  }
  return MK_TORCHY(input.dtype(), input.device(), H__BATCH_NORM_IMPL_INDEX_BACKWARD, impl_index, input, grad_output, weight, running_mean, running_var, save_mean, save_var_transform, train, eps, output_mask, reservedSpace);
}

Tensor wrap_bernoulli(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::bernoulli(self, generator);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BERNOULLI, self, generator);
}

Tensor wrap_bernoulli_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::bernoulli(out, self, generator);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_BERNOULLI_OUT, out, self, generator);
}

Tensor wrap_bernoulli__Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, p);
    return at::redispatch::bernoulli_(self, p, generator);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BERNOULLI__TENSOR, self, p, generator);
}

Tensor wrap_bernoulli__float(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::bernoulli_(self, p, generator);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BERNOULLI__FLOAT, self, p, generator);
}

Tensor wrap_bernoulli_p(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::bernoulli(self, p, generator);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BERNOULLI_P, self, p, generator);
}

Tensor wrap_bilinear(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input1, input2, weight);
    return at::redispatch::bilinear(input1, input2, weight, bias);
  }
  return MK_TORCHY(input1.dtype(), input1.device(), H_BILINEAR, input1, input2, weight, bias);
}

Tensor wrap_binary_cross_entropy(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, target);
    return at::redispatch::binary_cross_entropy(self, target, weight, reduction);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BINARY_CROSS_ENTROPY, self, target, weight, reduction);
}

Tensor wrap_binary_cross_entropy_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, target);
    return at::redispatch::binary_cross_entropy(out, self, target, weight, reduction);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_BINARY_CROSS_ENTROPY_OUT, out, self, target, weight, reduction);
}

Tensor wrap_binary_cross_entropy_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, target);
    return at::redispatch::binary_cross_entropy_backward(grad_output, self, target, weight, reduction);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_BINARY_CROSS_ENTROPY_BACKWARD, grad_output, self, target, weight, reduction);
}

Tensor wrap_binary_cross_entropy_backward_grad_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self, target);
    return at::redispatch::binary_cross_entropy_backward(grad_input, grad_output, self, target, weight, reduction);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_BINARY_CROSS_ENTROPY_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, target, weight, reduction);
}

Tensor wrap_binary_cross_entropy_with_logits(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, target);
    return at::redispatch::binary_cross_entropy_with_logits(self, target, weight, pos_weight, reduction);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BINARY_CROSS_ENTROPY_WITH_LOGITS, self, target, weight, pos_weight, reduction);
}

Tensor wrap_binary_cross_entropy_with_logits_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, target);
    return at::redispatch::binary_cross_entropy_with_logits_backward(grad_output, self, target, weight, pos_weight, reduction);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_BINARY_CROSS_ENTROPY_WITH_LOGITS_BACKWARD, grad_output, self, target, weight, pos_weight, reduction);
}

Tensor wrap_bincount(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::bincount(self, weights, minlength);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BINCOUNT, self, weights, minlength);
}

Tensor wrap_bitwise_not(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::bitwise_not(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BITWISE_NOT, self);
}

Tensor wrap_bitwise_not_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::bitwise_not_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BITWISE_NOT_, self);
}

Tensor wrap_bitwise_not_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::bitwise_not(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_BITWISE_NOT_OUT, out, self);
}

Tensor wrap_copysign_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::copysign(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_COPYSIGN_OUT, out, self, other);
}

Tensor wrap_copysign_Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::copysign(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_COPYSIGN_SCALAR, self, other);
}

Tensor wrap_copysign__Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::copysign_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_COPYSIGN__SCALAR, self, other);
}

Tensor wrap_copysign_Scalar_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::copysign(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_COPYSIGN_SCALAR_OUT, out, self, other);
}

Tensor wrap_logical_not(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::logical_not(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LOGICAL_NOT, self);
}

Tensor wrap_logical_not_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::logical_not_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LOGICAL_NOT_, self);
}

Tensor wrap_logical_not_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::logical_not(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LOGICAL_NOT_OUT, out, self);
}

Tensor wrap_logical_xor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::logical_xor(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LOGICAL_XOR, self, other);
}

Tensor wrap_logical_xor_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::logical_xor_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LOGICAL_XOR_, self, other);
}

Tensor wrap_logical_xor_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::logical_xor(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LOGICAL_XOR_OUT, out, self, other);
}

Tensor wrap_logical_and(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::logical_and(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LOGICAL_AND, self, other);
}

Tensor wrap_logical_and_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::logical_and_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LOGICAL_AND_, self, other);
}

Tensor wrap_logical_and_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::logical_and(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LOGICAL_AND_OUT, out, self, other);
}

Tensor wrap_logical_or(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::logical_or(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LOGICAL_OR, self, other);
}

Tensor wrap_logical_or_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::logical_or_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LOGICAL_OR_, self, other);
}

Tensor wrap_logical_or_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::logical_or(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LOGICAL_OR_OUT, out, self, other);
}

Tensor wrap_blackman_window(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::blackman_window(window_length, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_BLACKMAN_WINDOW, window_length, dtype, layout, device, pin_memory);
}

Tensor wrap_blackman_window_periodic(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::blackman_window(window_length, periodic, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_BLACKMAN_WINDOW_PERIODIC, window_length, periodic, dtype, layout, device, pin_memory);
}

Tensor wrap_bmm(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, mat2);
    return at::redispatch::bmm(self, mat2);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BMM, self, mat2);
}

Tensor wrap__bmm(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, mat2);
    return at::redispatch::_bmm(self, mat2, deterministic);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__BMM, self, mat2, deterministic);
}

Tensor wrap_bmm_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, mat2);
    return at::redispatch::bmm(out, self, mat2);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_BMM_OUT, out, self, mat2);
}

Tensor wrap__bmm_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, mat2);
    return at::redispatch::_bmm(out, self, mat2, deterministic);
  }
  return MK_TORCHY(out.dtype(), out.device(), H__BMM_OUT, out, self, mat2, deterministic);
}

Tensor[] wrap_broadcast_tensors(args...) {
  ensure_materialized();
  return at::redispatch::broadcast_tensors(tensors);
}

Tensor wrap_broadcast_to(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::broadcast_to(self, size);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BROADCAST_TO, self, size);
}

Tensor wrap_cat(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::cat(tensors, dim);
  }
  return MK_TORCHY(None, None, H_CAT, tensors, dim);
}

Tensor wrap_cat_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::cat(out, tensors, dim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_CAT_OUT, out, tensors, dim);
}

Tensor wrap_cat_names(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::cat(tensors, dim);
  }
  return MK_TORCHY(None, None, H_CAT_NAMES, tensors, dim);
}

Tensor wrap_cat_names_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::cat(out, tensors, dim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_CAT_NAMES_OUT, out, tensors, dim);
}

Tensor wrap_block_diag(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::block_diag(tensors);
  }
  return MK_TORCHY(None, None, H_BLOCK_DIAG, tensors);
}

Tensor wrap_ceil(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::ceil(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CEIL, self);
}

Tensor wrap_ceil_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::ceil_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CEIL_, self);
}

Tensor wrap_ceil_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::ceil(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_CEIL_OUT, out, self);
}

Tensor wrap_chain_matmul(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::chain_matmul(matrices);
  }
  return MK_TORCHY(None, None, H_CHAIN_MATMUL, matrices);
}

Tensor wrap_chain_matmul_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::chain_matmul(out, matrices);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_CHAIN_MATMUL_OUT, out, matrices);
}

Tensor[] wrap_unsafe_chunk(args...) {
  ensure_materialized(self);
  return at::redispatch::unsafe_chunk(self, chunks, dim);
}

Tensor[] wrap_chunk(args...) {
  ensure_materialized(self);
  return at::redispatch::chunk(self, chunks, dim);
}

Tensor[] wrap_tensor_split_sections(args...) {
  ensure_materialized(self);
  return at::redispatch::tensor_split(self, sections, dim);
}

Tensor[] wrap_tensor_split_indices(args...) {
  ensure_materialized(self);
  return at::redispatch::tensor_split(self, indices, dim);
}

Tensor[] wrap_tensor_split_tensor_indices_or_sections(args...) {
  ensure_materialized(self, tensor_indices_or_sections);
  return at::redispatch::tensor_split(self, tensor_indices_or_sections, dim);
}

Tensor wrap_clamp(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::clamp(self, min, max);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CLAMP, self, min, max);
}

Tensor wrap_clamp_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::clamp_(self, min, max);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CLAMP_, self, min, max);
}

Tensor wrap_clamp_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::clamp(out, self, min, max);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_CLAMP_OUT, out, self, min, max);
}

Tensor wrap_clamp_max(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::clamp_max(self, max);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CLAMP_MAX, self, max);
}

Tensor wrap_clamp_max_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::clamp_max_(self, max);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CLAMP_MAX_, self, max);
}

Tensor wrap_clamp_max_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::clamp_max(out, self, max);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_CLAMP_MAX_OUT, out, self, max);
}

Tensor wrap_clamp_min(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::clamp_min(self, min);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CLAMP_MIN, self, min);
}

Tensor wrap_clamp_min_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::clamp_min_(self, min);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CLAMP_MIN_, self, min);
}

Tensor wrap_clamp_min_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::clamp_min(out, self, min);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_CLAMP_MIN_OUT, out, self, min);
}

Tensor wrap_clip(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::clip(self, min, max);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CLIP, self, min, max);
}

Tensor wrap_clip_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::clip_(self, min, max);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CLIP_, self, min, max);
}

Tensor wrap_clip_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::clip(out, self, min, max);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_CLIP_OUT, out, self, min, max);
}

bool wrap_cudnn_is_acceptable(args...) {
  ensure_materialized(self);
  return at::redispatch::cudnn_is_acceptable(self);
}

Tensor wrap_complex(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(real, imag);
    return at::redispatch::complex(real, imag);
  }
  return MK_TORCHY(real.dtype(), real.device(), H_COMPLEX, real, imag);
}

Tensor wrap_complex_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, real, imag);
    return at::redispatch::complex(out, real, imag);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_COMPLEX_OUT, out, real, imag);
}

Tensor wrap_polar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(abs, angle);
    return at::redispatch::polar(abs, angle);
  }
  return MK_TORCHY(abs.dtype(), abs.device(), H_POLAR, abs, angle);
}

Tensor wrap_polar_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, abs, angle);
    return at::redispatch::polar(out, abs, angle);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_POLAR_OUT, out, abs, angle);
}

Tensor wrap_constant_pad_nd(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::constant_pad_nd(self, pad, value);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CONSTANT_PAD_ND, self, pad, value);
}

Tensor wrap_convolution(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, weight);
    return at::redispatch::convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_CONVOLUTION, input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
}

Tensor wrap_convolution_overrideable(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, weight);
    return at::redispatch::convolution_overrideable(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_CONVOLUTION_OVERRIDEABLE, input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
}

Tensor wrap_convolution_backward_overrideable(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, input, weight);
    return at::redispatch::convolution_backward_overrideable(grad_output, input, weight, stride, padding, dilation, transposed, output_padding, groups, output_mask);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_CONVOLUTION_BACKWARD_OVERRIDEABLE, grad_output, input, weight, stride, padding, dilation, transposed, output_padding, groups, output_mask);
}

Tensor wrap__convolution(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, weight);
    return at::redispatch::_convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, allow_tf32);
  }
  return MK_TORCHY(input.dtype(), input.device(), H__CONVOLUTION, input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, allow_tf32);
}

Tensor wrap__convolution_deprecated(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, weight);
    return at::redispatch::_convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled);
  }
  return MK_TORCHY(input.dtype(), input.device(), H__CONVOLUTION_DEPRECATED, input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled);
}

Tensor wrap__convolution_mode(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, weight);
    return at::redispatch::_convolution_mode(input, weight, bias, stride, padding, dilation, groups);
  }
  return MK_TORCHY(input.dtype(), input.device(), H__CONVOLUTION_MODE, input, weight, bias, stride, padding, dilation, groups);
}

Tensor wrap__convolution_nogroup(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, weight);
    return at::redispatch::_convolution_nogroup(input, weight, bias, stride, padding, dilation, transposed, output_padding);
  }
  return MK_TORCHY(input.dtype(), input.device(), H__CONVOLUTION_NOGROUP, input, weight, bias, stride, padding, dilation, transposed, output_padding);
}

Tensor wrap__convolution_double_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(gO, weight, self);
    return at::redispatch::_convolution_double_backward(ggI, ggW, ggb, gO, weight, self, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, allow_tf32, output_mask);
  }
  return MK_TORCHY(gO.dtype(), gO.device(), H__CONVOLUTION_DOUBLE_BACKWARD, ggI, ggW, ggb, gO, weight, self, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, allow_tf32, output_mask);
}

Tensor wrap_conv1d(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, weight);
    return at::redispatch::conv1d(input, weight, bias, stride, padding, dilation, groups);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_CONV1D, input, weight, bias, stride, padding, dilation, groups);
}

Tensor wrap_conv2d(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, weight);
    return at::redispatch::conv2d(input, weight, bias, stride, padding, dilation, groups);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_CONV2D, input, weight, bias, stride, padding, dilation, groups);
}

Tensor wrap_conv3d(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, weight);
    return at::redispatch::conv3d(input, weight, bias, stride, padding, dilation, groups);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_CONV3D, input, weight, bias, stride, padding, dilation, groups);
}

Tensor wrap_conv1d_padding(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, weight);
    return at::redispatch::conv1d(input, weight, bias, stride, padding, dilation, groups);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_CONV1D_PADDING, input, weight, bias, stride, padding, dilation, groups);
}

Tensor wrap_conv2d_padding(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, weight);
    return at::redispatch::conv2d(input, weight, bias, stride, padding, dilation, groups);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_CONV2D_PADDING, input, weight, bias, stride, padding, dilation, groups);
}

Tensor wrap_conv3d_padding(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, weight);
    return at::redispatch::conv3d(input, weight, bias, stride, padding, dilation, groups);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_CONV3D_PADDING, input, weight, bias, stride, padding, dilation, groups);
}

Tensor wrap_conv_tbc(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, weight, bias);
    return at::redispatch::conv_tbc(self, weight, bias, pad);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CONV_TBC, self, weight, bias, pad);
}

Tensor wrap_conv_tbc_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, input, weight, bias);
    return at::redispatch::conv_tbc_backward(self, input, weight, bias, pad);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CONV_TBC_BACKWARD, self, input, weight, bias, pad);
}

Tensor wrap_conv_transpose1d(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, weight);
    return at::redispatch::conv_transpose1d(input, weight, bias, stride, padding, output_padding, groups, dilation);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_CONV_TRANSPOSE1D, input, weight, bias, stride, padding, output_padding, groups, dilation);
}

Tensor wrap_conv_transpose2d_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, weight);
    return at::redispatch::conv_transpose2d(input, weight, bias, stride, padding, output_padding, groups, dilation);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_CONV_TRANSPOSE2D_INPUT, input, weight, bias, stride, padding, output_padding, groups, dilation);
}

Tensor wrap_conv_transpose3d_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, weight);
    return at::redispatch::conv_transpose3d(input, weight, bias, stride, padding, output_padding, groups, dilation);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_CONV_TRANSPOSE3D_INPUT, input, weight, bias, stride, padding, output_padding, groups, dilation);
}

Tensor wrap_copy_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, src);
    return at::redispatch::copy_(self, src, non_blocking);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_COPY_, self, src, non_blocking);
}

Tensor wrap_cos_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::cos(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_COS_OUT, out, self);
}

Tensor wrap_cosh_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::cosh(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_COSH_OUT, out, self);
}

Tensor wrap_cosine_embedding_loss(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input1, input2, target);
    return at::redispatch::cosine_embedding_loss(input1, input2, target, margin, reduction);
  }
  return MK_TORCHY(input1.dtype(), input1.device(), H_COSINE_EMBEDDING_LOSS, input1, input2, target, margin, reduction);
}

Tensor wrap_count_nonzero_dim_IntList(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::count_nonzero(self, dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_COUNT_NONZERO_DIM_INTLIST, self, dim);
}

Tensor wrap_count_nonzero(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::count_nonzero(self, dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_COUNT_NONZERO, self, dim);
}

Tensor wrap_cudnn_affine_grid_generator(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(theta);
    return at::redispatch::cudnn_affine_grid_generator(theta, N, C, H, W);
  }
  return MK_TORCHY(theta.dtype(), theta.device(), H_CUDNN_AFFINE_GRID_GENERATOR, theta, N, C, H, W);
}

Tensor wrap_cudnn_affine_grid_generator_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad);
    return at::redispatch::cudnn_affine_grid_generator_backward(grad, N, C, H, W);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H_CUDNN_AFFINE_GRID_GENERATOR_BACKWARD, grad, N, C, H, W);
}

Tensor wrap_cudnn_batch_norm(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, weight);
    return at::redispatch::cudnn_batch_norm(input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_CUDNN_BATCH_NORM, input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon);
}

Tensor wrap_cudnn_batch_norm_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, grad_output, weight, reserveSpace);
    return at::redispatch::cudnn_batch_norm_backward(input, grad_output, weight, running_mean, running_var, save_mean, save_var, epsilon, reserveSpace);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_CUDNN_BATCH_NORM_BACKWARD, input, grad_output, weight, running_mean, running_var, save_mean, save_var, epsilon, reserveSpace);
}

Tensor wrap_cudnn_convolution_deprecated(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, weight);
    return at::redispatch::cudnn_convolution(self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CUDNN_CONVOLUTION_DEPRECATED, self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic);
}

Tensor wrap_cudnn_convolution_deprecated2(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, weight);
    return at::redispatch::cudnn_convolution(self, weight, padding, stride, dilation, groups, benchmark, deterministic);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CUDNN_CONVOLUTION_DEPRECATED2, self, weight, padding, stride, dilation, groups, benchmark, deterministic);
}

Tensor wrap_cudnn_convolution(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, weight);
    return at::redispatch::cudnn_convolution(self, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CUDNN_CONVOLUTION, self, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
}

Tensor wrap_cudnn_convolution_backward_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, weight);
    return at::redispatch::cudnn_convolution_backward_input(self_size, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_CUDNN_CONVOLUTION_BACKWARD_INPUT, self_size, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
}

Tensor wrap_cudnn_convolution_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, grad_output, weight);
    return at::redispatch::cudnn_convolution_backward(self, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32, output_mask);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CUDNN_CONVOLUTION_BACKWARD, self, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32, output_mask);
}

Tensor wrap_cudnn_convolution_backward_weight(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self);
    return at::redispatch::cudnn_convolution_backward_weight(weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_CUDNN_CONVOLUTION_BACKWARD_WEIGHT, weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
}

Tensor wrap_cudnn_convolution_transpose_deprecated(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, weight);
    return at::redispatch::cudnn_convolution_transpose(self, weight, bias, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CUDNN_CONVOLUTION_TRANSPOSE_DEPRECATED, self, weight, bias, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
}

Tensor wrap_cudnn_convolution_transpose_deprecated2(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, weight);
    return at::redispatch::cudnn_convolution_transpose(self, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CUDNN_CONVOLUTION_TRANSPOSE_DEPRECATED2, self, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
}

Tensor wrap_cudnn_convolution_transpose(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, weight);
    return at::redispatch::cudnn_convolution_transpose(self, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CUDNN_CONVOLUTION_TRANSPOSE, self, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
}

Tensor wrap_cudnn_convolution_transpose_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, grad_output, weight);
    return at::redispatch::cudnn_convolution_transpose_backward(self, grad_output, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic, allow_tf32, output_mask);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CUDNN_CONVOLUTION_TRANSPOSE_BACKWARD, self, grad_output, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic, allow_tf32, output_mask);
}

Tensor wrap_cudnn_convolution_transpose_backward_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, weight);
    return at::redispatch::cudnn_convolution_transpose_backward_input(grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_CUDNN_CONVOLUTION_TRANSPOSE_BACKWARD_INPUT, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
}

Tensor wrap_cudnn_convolution_transpose_backward_weight(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self);
    return at::redispatch::cudnn_convolution_transpose_backward_weight(weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_CUDNN_CONVOLUTION_TRANSPOSE_BACKWARD_WEIGHT, weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
}

Tensor wrap_cudnn_convolution_relu(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, weight);
    return at::redispatch::cudnn_convolution_relu(self, weight, bias, stride, padding, dilation, groups);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CUDNN_CONVOLUTION_RELU, self, weight, bias, stride, padding, dilation, groups);
}

Tensor wrap_cudnn_convolution_add_relu(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, weight, z);
    return at::redispatch::cudnn_convolution_add_relu(self, weight, z, alpha, bias, stride, padding, dilation, groups);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CUDNN_CONVOLUTION_ADD_RELU, self, weight, z, alpha, bias, stride, padding, dilation, groups);
}

Tensor wrap_cudnn_grid_sampler(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, grid);
    return at::redispatch::cudnn_grid_sampler(self, grid);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CUDNN_GRID_SAMPLER, self, grid);
}

Tensor wrap_cudnn_grid_sampler_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, grid, grad_output);
    return at::redispatch::cudnn_grid_sampler_backward(self, grid, grad_output);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CUDNN_GRID_SAMPLER_BACKWARD, self, grid, grad_output);
}

Tensor wrap_cummax(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::cummax(self, dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CUMMAX, self, dim);
}

Tensor wrap_cummax_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(values, indices, self);
    return at::redispatch::cummax(values, indices, self, dim);
  }
  return MK_TORCHY(values.dtype(), values.device(), H_CUMMAX_OUT, values, indices, self, dim);
}

Tensor wrap_cummax_dimname(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::cummax(self, dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CUMMAX_DIMNAME, self, dim);
}

Tensor wrap_cummax_dimname_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(values, indices, self);
    return at::redispatch::cummax(values, indices, self, dim);
  }
  return MK_TORCHY(values.dtype(), values.device(), H_CUMMAX_DIMNAME_OUT, values, indices, self, dim);
}

void wrap__cummax_helper(args...) {
  ensure_materialized(self, values, indices);
  return at::redispatch::_cummax_helper(self, values, indices, dim);
}

Tensor wrap_cummin(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::cummin(self, dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CUMMIN, self, dim);
}

Tensor wrap_cummin_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(values, indices, self);
    return at::redispatch::cummin(values, indices, self, dim);
  }
  return MK_TORCHY(values.dtype(), values.device(), H_CUMMIN_OUT, values, indices, self, dim);
}

Tensor wrap_cummin_dimname(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::cummin(self, dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CUMMIN_DIMNAME, self, dim);
}

Tensor wrap_cummin_dimname_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(values, indices, self);
    return at::redispatch::cummin(values, indices, self, dim);
  }
  return MK_TORCHY(values.dtype(), values.device(), H_CUMMIN_DIMNAME_OUT, values, indices, self, dim);
}

void wrap__cummin_helper(args...) {
  ensure_materialized(self, values, indices);
  return at::redispatch::_cummin_helper(self, values, indices, dim);
}

Tensor wrap_cummaxmin_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad, input, indices);
    return at::redispatch::cummaxmin_backward(grad, input, indices, dim);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H_CUMMAXMIN_BACKWARD, grad, input, indices, dim);
}

Tensor wrap_cumprod(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::cumprod(self, dim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CUMPROD, self, dim, dtype);
}

Tensor wrap_cumprod_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::cumprod_(self, dim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CUMPROD_, self, dim, dtype);
}

Tensor wrap_cumprod_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::cumprod(out, self, dim, dtype);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_CUMPROD_OUT, out, self, dim, dtype);
}

Tensor wrap_cumprod_dimname(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::cumprod(self, dim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CUMPROD_DIMNAME, self, dim, dtype);
}

Tensor wrap_cumprod__dimname(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::cumprod_(self, dim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CUMPROD__DIMNAME, self, dim, dtype);
}

Tensor wrap_cumprod_dimname_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::cumprod(out, self, dim, dtype);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_CUMPROD_DIMNAME_OUT, out, self, dim, dtype);
}

Tensor wrap_cumprod_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad, input, output);
    return at::redispatch::cumprod_backward(grad, input, dim, output);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H_CUMPROD_BACKWARD, grad, input, dim, output);
}

Tensor wrap_cumsum(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::cumsum(self, dim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CUMSUM, self, dim, dtype);
}

Tensor wrap_cumsum_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::cumsum_(self, dim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CUMSUM_, self, dim, dtype);
}

Tensor wrap_cumsum_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::cumsum(out, self, dim, dtype);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_CUMSUM_OUT, out, self, dim, dtype);
}

Tensor wrap_cumsum_dimname(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::cumsum(self, dim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CUMSUM_DIMNAME, self, dim, dtype);
}

Tensor wrap_cumsum__dimname(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::cumsum_(self, dim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CUMSUM__DIMNAME, self, dim, dtype);
}

Tensor wrap_cumsum_dimname_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::cumsum(out, self, dim, dtype);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_CUMSUM_DIMNAME_OUT, out, self, dim, dtype);
}

Tensor wrap_ctc_loss_IntList(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(log_probs, targets);
    return at::redispatch::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity);
  }
  return MK_TORCHY(log_probs.dtype(), log_probs.device(), H_CTC_LOSS_INTLIST, log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity);
}

Tensor wrap_ctc_loss_Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(log_probs, targets, input_lengths, target_lengths);
    return at::redispatch::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity);
  }
  return MK_TORCHY(log_probs.dtype(), log_probs.device(), H_CTC_LOSS_TENSOR, log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity);
}

Tensor wrap__ctc_loss(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(log_probs, targets);
    return at::redispatch::_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, zero_infinity);
  }
  return MK_TORCHY(log_probs.dtype(), log_probs.device(), H__CTC_LOSS, log_probs, targets, input_lengths, target_lengths, blank, zero_infinity);
}

Tensor wrap__ctc_loss_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad, log_probs, targets, neg_log_likelihood, log_alpha);
    return at::redispatch::_ctc_loss_backward(grad, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, blank, zero_infinity);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H__CTC_LOSS_BACKWARD, grad, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, blank, zero_infinity);
}

Tensor wrap_diag_embed(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::diag_embed(self, offset, dim1, dim2);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DIAG_EMBED, self, offset, dim1, dim2);
}

Tensor wrap_diagflat(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::diagflat(self, offset);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DIAGFLAT, self, offset);
}

Tensor wrap_diagonal(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::diagonal(self, offset, dim1, dim2);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DIAGONAL, self, offset, dim1, dim2);
}

Tensor wrap_diagonal_Dimname(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::diagonal(self, outdim, dim1, dim2, offset);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DIAGONAL_DIMNAME, self, outdim, dim1, dim2, offset);
}

Tensor wrap_diagonal_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad);
    return at::redispatch::diagonal_backward(grad, input_sizes, offset, dim1, dim2);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H_DIAGONAL_BACKWARD, grad, input_sizes, offset, dim1, dim2);
}

Tensor wrap_fill_diagonal_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::fill_diagonal_(self, fill_value, wrap);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FILL_DIAGONAL_, self, fill_value, wrap);
}

Tensor wrap_diff(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::diff(self, n, dim, prepend, append);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DIFF, self, n, dim, prepend, append);
}

Tensor wrap_diff_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::diff(out, self, n, dim, prepend, append);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_DIFF_OUT, out, self, n, dim, prepend, append);
}

Tensor wrap_div_Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::div(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DIV_TENSOR, self, other);
}

Tensor wrap_div__Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::div_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DIV__TENSOR, self, other);
}

Tensor wrap_div_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::div(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_DIV_OUT, out, self, other);
}

Tensor wrap_div_out_mode(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::div(out, self, other, rounding_mode);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_DIV_OUT_MODE, out, self, other, rounding_mode);
}

Tensor wrap_div_Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::div(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DIV_SCALAR, self, other);
}

Tensor wrap_div__Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::div_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DIV__SCALAR, self, other);
}

Tensor wrap_div_Scalar_mode(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::div(self, other, rounding_mode);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DIV_SCALAR_MODE, self, other, rounding_mode);
}

Tensor wrap_div__Scalar_mode(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::div_(self, other, rounding_mode);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DIV__SCALAR_MODE, self, other, rounding_mode);
}

Tensor wrap_divide_Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::divide(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DIVIDE_TENSOR, self, other);
}

Tensor wrap_divide__Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::divide_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DIVIDE__TENSOR, self, other);
}

Tensor wrap_divide_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::divide(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_DIVIDE_OUT, out, self, other);
}

Tensor wrap_divide_Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::divide(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DIVIDE_SCALAR, self, other);
}

Tensor wrap_divide__Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::divide_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DIVIDE__SCALAR, self, other);
}

Tensor wrap_divide_Tensor_mode(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::divide(self, other, rounding_mode);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DIVIDE_TENSOR_MODE, self, other, rounding_mode);
}

Tensor wrap_divide__Tensor_mode(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::divide_(self, other, rounding_mode);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DIVIDE__TENSOR_MODE, self, other, rounding_mode);
}

Tensor wrap_divide_out_mode(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::divide(out, self, other, rounding_mode);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_DIVIDE_OUT_MODE, out, self, other, rounding_mode);
}

Tensor wrap_divide_Scalar_mode(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::divide(self, other, rounding_mode);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DIVIDE_SCALAR_MODE, self, other, rounding_mode);
}

Tensor wrap_divide__Scalar_mode(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::divide_(self, other, rounding_mode);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DIVIDE__SCALAR_MODE, self, other, rounding_mode);
}

Tensor wrap_true_divide_Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::true_divide(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TRUE_DIVIDE_TENSOR, self, other);
}

Tensor wrap_true_divide__Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::true_divide_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TRUE_DIVIDE__TENSOR, self, other);
}

Tensor wrap_true_divide_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::true_divide(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_TRUE_DIVIDE_OUT, out, self, other);
}

Tensor wrap_true_divide_Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::true_divide(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TRUE_DIVIDE_SCALAR, self, other);
}

Tensor wrap_true_divide__Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::true_divide_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TRUE_DIVIDE__SCALAR, self, other);
}

Tensor wrap_dot(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, tensor);
    return at::redispatch::dot(self, tensor);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DOT, self, tensor);
}

Tensor wrap_dot_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, tensor);
    return at::redispatch::dot(out, self, tensor);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_DOT_OUT, out, self, tensor);
}

Tensor wrap_vdot(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::vdot(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_VDOT, self, other);
}

Tensor wrap_vdot_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::vdot(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_VDOT_OUT, out, self, other);
}

Tensor wrap_einsum(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::einsum(equation, tensors);
  }
  return MK_TORCHY(None, None, H_EINSUM, equation, tensors);
}

Tensor wrap_embedding(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(weight, indices);
    return at::redispatch::embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse);
  }
  return MK_TORCHY(weight.dtype(), weight.device(), H_EMBEDDING, weight, indices, padding_idx, scale_grad_by_freq, sparse);
}

Tensor wrap_embedding_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad, indices);
    return at::redispatch::embedding_backward(grad, indices, num_weights, padding_idx, scale_grad_by_freq, sparse);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H_EMBEDDING_BACKWARD, grad, indices, num_weights, padding_idx, scale_grad_by_freq, sparse);
}

Tensor wrap_embedding_dense_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, indices);
    return at::redispatch::embedding_dense_backward(grad_output, indices, num_weights, padding_idx, scale_grad_by_freq);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_EMBEDDING_DENSE_BACKWARD, grad_output, indices, num_weights, padding_idx, scale_grad_by_freq);
}

Tensor wrap_embedding_renorm_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, indices);
    return at::redispatch::embedding_renorm_(self, indices, max_norm, norm_type);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_EMBEDDING_RENORM_, self, indices, max_norm, norm_type);
}

Tensor wrap_embedding_sparse_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad, indices);
    return at::redispatch::embedding_sparse_backward(grad, indices, num_weights, padding_idx, scale_grad_by_freq);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H_EMBEDDING_SPARSE_BACKWARD, grad, indices, num_weights, padding_idx, scale_grad_by_freq);
}

Tensor wrap__embedding_bag_forward_only(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(weight, indices, offsets);
    return at::redispatch::_embedding_bag_forward_only(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx);
  }
  return MK_TORCHY(weight.dtype(), weight.device(), H__EMBEDDING_BAG_FORWARD_ONLY, weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx);
}

Tensor wrap__rowwise_prune(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(weight, mask);
    return at::redispatch::_rowwise_prune(weight, mask, compressed_indices_dtype);
  }
  return MK_TORCHY(weight.dtype(), weight.device(), H__ROWWISE_PRUNE, weight, mask, compressed_indices_dtype);
}

Tensor wrap_row_stack(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::row_stack(tensors);
  }
  return MK_TORCHY(None, None, H_ROW_STACK, tensors);
}

Tensor wrap_row_stack_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::row_stack(out, tensors);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ROW_STACK_OUT, out, tensors);
}

Tensor wrap_embedding_bag(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(weight, indices, offsets);
    return at::redispatch::embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset);
  }
  return MK_TORCHY(weight.dtype(), weight.device(), H_EMBEDDING_BAG, weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset);
}

Tensor wrap_embedding_bag_padding_idx(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(weight, indices, offsets);
    return at::redispatch::embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx);
  }
  return MK_TORCHY(weight.dtype(), weight.device(), H_EMBEDDING_BAG_PADDING_IDX, weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx);
}

Tensor wrap__embedding_bag(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(weight, indices, offsets);
    return at::redispatch::_embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx);
  }
  return MK_TORCHY(weight.dtype(), weight.device(), H__EMBEDDING_BAG, weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx);
}

Tensor wrap__embedding_bag_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad, indices, offsets, offset2bag, bag_size, maximum_indices);
    return at::redispatch::_embedding_bag_backward(grad, indices, offsets, offset2bag, bag_size, maximum_indices, num_weights, scale_grad_by_freq, mode, sparse, per_sample_weights, padding_idx);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H__EMBEDDING_BAG_BACKWARD, grad, indices, offsets, offset2bag, bag_size, maximum_indices, num_weights, scale_grad_by_freq, mode, sparse, per_sample_weights, padding_idx);
}

Tensor wrap__embedding_bag_sparse_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad, indices, offsets, offset2bag, bag_size);
    return at::redispatch::_embedding_bag_sparse_backward(grad, indices, offsets, offset2bag, bag_size, num_weights, scale_grad_by_freq, mode, per_sample_weights, padding_idx);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H__EMBEDDING_BAG_SPARSE_BACKWARD, grad, indices, offsets, offset2bag, bag_size, num_weights, scale_grad_by_freq, mode, per_sample_weights, padding_idx);
}

Tensor wrap__embedding_bag_dense_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad, indices, offset2bag, bag_size, maximum_indices);
    return at::redispatch::_embedding_bag_dense_backward(grad, indices, offset2bag, bag_size, maximum_indices, num_weights, scale_grad_by_freq, mode, per_sample_weights, padding_idx);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H__EMBEDDING_BAG_DENSE_BACKWARD, grad, indices, offset2bag, bag_size, maximum_indices, num_weights, scale_grad_by_freq, mode, per_sample_weights, padding_idx);
}

Tensor wrap__embedding_bag_per_sample_weights_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad, weight, indices, offsets, offset2bag);
    return at::redispatch::_embedding_bag_per_sample_weights_backward(grad, weight, indices, offsets, offset2bag, mode, padding_idx);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H__EMBEDDING_BAG_PER_SAMPLE_WEIGHTS_BACKWARD, grad, weight, indices, offsets, offset2bag, mode, padding_idx);
}

Tensor wrap_empty_names(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::empty(size, names, dtype, layout, device, pin_memory, memory_format);
  }
  return MK_TORCHY(None, None, H_EMPTY_NAMES, size, names, dtype, layout, device, pin_memory, memory_format);
}

Tensor wrap_empty_memory_format(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::empty(size, dtype, layout, device, pin_memory, memory_format);
  }
  return MK_TORCHY(None, None, H_EMPTY_MEMORY_FORMAT, size, dtype, layout, device, pin_memory, memory_format);
}

Tensor wrap_new_empty(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::new_empty(self, size, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NEW_EMPTY, self, size, dtype, layout, device, pin_memory);
}

Tensor wrap_new_empty_strided(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::new_empty_strided(self, size, stride, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NEW_EMPTY_STRIDED, self, size, stride, dtype, layout, device, pin_memory);
}

Tensor wrap_new_full(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::new_full(self, size, fill_value, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NEW_FULL, self, size, fill_value, dtype, layout, device, pin_memory);
}

Tensor wrap_new_zeros(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::new_zeros(self, size, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NEW_ZEROS, self, size, dtype, layout, device, pin_memory);
}

Tensor wrap__empty_affine_quantized(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::_empty_affine_quantized(size, dtype, layout, device, pin_memory, scale, zero_point, memory_format);
  }
  return MK_TORCHY(None, None, H__EMPTY_AFFINE_QUANTIZED, size, dtype, layout, device, pin_memory, scale, zero_point, memory_format);
}

Tensor wrap__empty_per_channel_affine_quantized(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(scales, zero_points);
    return at::redispatch::_empty_per_channel_affine_quantized(size, scales, zero_points, axis, dtype, layout, device, pin_memory, memory_format);
  }
  return MK_TORCHY(scales.dtype(), scales.device(), H__EMPTY_PER_CHANNEL_AFFINE_QUANTIZED, size, scales, zero_points, axis, dtype, layout, device, pin_memory, memory_format);
}

Tensor wrap_resize_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::resize_(self, size, memory_format);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RESIZE_, self, size, memory_format);
}

Tensor wrap_empty_quantized(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(qtensor);
    return at::redispatch::empty_quantized(size, qtensor);
  }
  return MK_TORCHY(qtensor.dtype(), qtensor.device(), H_EMPTY_QUANTIZED, size, qtensor);
}

Tensor wrap_empty_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::empty(out, size, memory_format);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_EMPTY_OUT, out, size, memory_format);
}

Tensor wrap_empty_like(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::empty_like(self, dtype, layout, device, pin_memory, memory_format);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_EMPTY_LIKE, self, dtype, layout, device, pin_memory, memory_format);
}

Tensor wrap_empty_strided(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::empty_strided(size, stride, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_EMPTY_STRIDED, size, stride, dtype, layout, device, pin_memory);
}

Tensor wrap_erf_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::erf(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ERF_OUT, out, self);
}

Tensor wrap_erfc_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::erfc(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ERFC_OUT, out, self);
}

Tensor wrap_exp_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::exp(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_EXP_OUT, out, self);
}

Tensor wrap_exp2_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::exp2(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_EXP2_OUT, out, self);
}

Tensor wrap_expm1_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::expm1(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_EXPM1_OUT, out, self);
}

Tensor wrap_expand(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::expand(self, size, implicit);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_EXPAND, self, size, implicit);
}

Tensor wrap_expand_as(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::expand_as(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_EXPAND_AS, self, other);
}

Tensor wrap_eye(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::eye(n, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_EYE, n, dtype, layout, device, pin_memory);
}

Tensor wrap_eye_m(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::eye(n, m, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_EYE_M, n, m, dtype, layout, device, pin_memory);
}

Tensor wrap_eye_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::eye(out, n);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_EYE_OUT, out, n);
}

Tensor wrap_eye_m_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::eye(out, n, m);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_EYE_M_OUT, out, n, m);
}

Tensor wrap_flatten_using_ints(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::flatten(self, start_dim, end_dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FLATTEN_USING_INTS, self, start_dim, end_dim);
}

Tensor wrap_flatten_named_out_dim(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::flatten(self, start_dim, end_dim, out_dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FLATTEN_NAMED_OUT_DIM, self, start_dim, end_dim, out_dim);
}

Tensor wrap_flatten_using_names(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::flatten(self, start_dim, end_dim, out_dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FLATTEN_USING_NAMES, self, start_dim, end_dim, out_dim);
}

Tensor wrap_flatten_DimnameList(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::flatten(self, dims, out_dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FLATTEN_DIMNAMELIST, self, dims, out_dim);
}

Tensor wrap_unflatten_int(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::unflatten(self, dim, sizes, names);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_UNFLATTEN_INT, self, dim, sizes, names);
}

Tensor wrap_unflatten_Dimname(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::unflatten(self, dim, sizes, names);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_UNFLATTEN_DIMNAME, self, dim, sizes, names);
}

Tensor wrap_fill__Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::fill_(self, value);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FILL__SCALAR, self, value);
}

Tensor wrap_fill__Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, value);
    return at::redispatch::fill_(self, value);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FILL__TENSOR, self, value);
}

Tensor wrap_floor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::floor(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FLOOR, self);
}

Tensor wrap_floor_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::floor_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FLOOR_, self);
}

Tensor wrap_floor_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::floor(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FLOOR_OUT, out, self);
}

Tensor wrap_floor_divide(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::floor_divide(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FLOOR_DIVIDE, self, other);
}

Tensor wrap_floor_divide__Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::floor_divide_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FLOOR_DIVIDE__TENSOR, self, other);
}

Tensor wrap_floor_divide_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::floor_divide(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FLOOR_DIVIDE_OUT, out, self, other);
}

Tensor wrap_floor_divide_Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::floor_divide(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FLOOR_DIVIDE_SCALAR, self, other);
}

Tensor wrap_floor_divide__Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::floor_divide_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FLOOR_DIVIDE__SCALAR, self, other);
}

Tensor wrap_frac(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::frac(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FRAC, self);
}

Tensor wrap_frac_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::frac_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FRAC_, self);
}

Tensor wrap_frac_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::frac(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FRAC_OUT, out, self);
}

Tensor wrap_full_names(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::full(size, fill_value, names, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_FULL_NAMES, size, fill_value, names, dtype, layout, device, pin_memory);
}

Tensor wrap_full(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::full(size, fill_value, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_FULL, size, fill_value, dtype, layout, device, pin_memory);
}

Tensor wrap_full_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::full(out, size, fill_value);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FULL_OUT, out, size, fill_value);
}

Tensor wrap_full_like(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::full_like(self, fill_value, dtype, layout, device, pin_memory, memory_format);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FULL_LIKE, self, fill_value, dtype, layout, device, pin_memory, memory_format);
}

Tensor wrap_from_file(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::from_file(filename, shared, size, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_FROM_FILE, filename, shared, size, dtype, layout, device, pin_memory);
}

Tensor wrap_gcd_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::gcd(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_GCD_OUT, out, self, other);
}

Tensor wrap_gcd(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::gcd(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_GCD, self, other);
}

Tensor wrap_gcd_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::gcd_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_GCD_, self, other);
}

Tensor wrap_lcm_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::lcm(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LCM_OUT, out, self, other);
}

Tensor wrap_lcm(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::lcm(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LCM, self, other);
}

Tensor wrap_lcm_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::lcm_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LCM_, self, other);
}

Tensor wrap_grid_sampler(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, grid);
    return at::redispatch::grid_sampler(input, grid, interpolation_mode, padding_mode, align_corners);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_GRID_SAMPLER, input, grid, interpolation_mode, padding_mode, align_corners);
}

Tensor wrap_grid_sampler_2d(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, grid);
    return at::redispatch::grid_sampler_2d(input, grid, interpolation_mode, padding_mode, align_corners);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_GRID_SAMPLER_2D, input, grid, interpolation_mode, padding_mode, align_corners);
}

Tensor wrap_grid_sampler_2d_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, input, grid);
    return at::redispatch::grid_sampler_2d_backward(grad_output, input, grid, interpolation_mode, padding_mode, align_corners);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_GRID_SAMPLER_2D_BACKWARD, grad_output, input, grid, interpolation_mode, padding_mode, align_corners);
}

Tensor wrap__grid_sampler_2d_cpu_fallback(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, grid);
    return at::redispatch::_grid_sampler_2d_cpu_fallback(input, grid, interpolation_mode, padding_mode, align_corners);
  }
  return MK_TORCHY(input.dtype(), input.device(), H__GRID_SAMPLER_2D_CPU_FALLBACK, input, grid, interpolation_mode, padding_mode, align_corners);
}

Tensor wrap__grid_sampler_2d_cpu_fallback_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, input, grid);
    return at::redispatch::_grid_sampler_2d_cpu_fallback_backward(grad_output, input, grid, interpolation_mode, padding_mode, align_corners);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H__GRID_SAMPLER_2D_CPU_FALLBACK_BACKWARD, grad_output, input, grid, interpolation_mode, padding_mode, align_corners);
}

Tensor wrap_grid_sampler_3d(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, grid);
    return at::redispatch::grid_sampler_3d(input, grid, interpolation_mode, padding_mode, align_corners);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_GRID_SAMPLER_3D, input, grid, interpolation_mode, padding_mode, align_corners);
}

Tensor wrap_grid_sampler_3d_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, input, grid);
    return at::redispatch::grid_sampler_3d_backward(grad_output, input, grid, interpolation_mode, padding_mode, align_corners);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_GRID_SAMPLER_3D_BACKWARD, grad_output, input, grid, interpolation_mode, padding_mode, align_corners);
}

Tensor wrap_hann_window(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::hann_window(window_length, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_HANN_WINDOW, window_length, dtype, layout, device, pin_memory);
}

Tensor wrap_hann_window_periodic(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::hann_window(window_length, periodic, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_HANN_WINDOW_PERIODIC, window_length, periodic, dtype, layout, device, pin_memory);
}

Tensor wrap_hamming_window(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::hamming_window(window_length, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_HAMMING_WINDOW, window_length, dtype, layout, device, pin_memory);
}

Tensor wrap_hamming_window_periodic(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::hamming_window(window_length, periodic, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_HAMMING_WINDOW_PERIODIC, window_length, periodic, dtype, layout, device, pin_memory);
}

Tensor wrap_hamming_window_periodic_alpha(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::hamming_window(window_length, periodic, alpha, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_HAMMING_WINDOW_PERIODIC_ALPHA, window_length, periodic, alpha, dtype, layout, device, pin_memory);
}

Tensor wrap_hamming_window_periodic_alpha_beta(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::hamming_window(window_length, periodic, alpha, beta, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_HAMMING_WINDOW_PERIODIC_ALPHA_BETA, window_length, periodic, alpha, beta, dtype, layout, device, pin_memory);
}

Tensor wrap_kaiser_window(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::kaiser_window(window_length, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_KAISER_WINDOW, window_length, dtype, layout, device, pin_memory);
}

Tensor wrap_kaiser_window_periodic(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::kaiser_window(window_length, periodic, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_KAISER_WINDOW_PERIODIC, window_length, periodic, dtype, layout, device, pin_memory);
}

Tensor wrap_kaiser_window_beta(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::kaiser_window(window_length, periodic, beta, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_KAISER_WINDOW_BETA, window_length, periodic, beta, dtype, layout, device, pin_memory);
}

Tensor wrap_hinge_embedding_loss(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, target);
    return at::redispatch::hinge_embedding_loss(self, target, margin, reduction);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_HINGE_EMBEDDING_LOSS, self, target, margin, reduction);
}

Tensor wrap_group_norm(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input);
    return at::redispatch::group_norm(input, num_groups, weight, bias, eps, cudnn_enabled);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_GROUP_NORM, input, num_groups, weight, bias, eps, cudnn_enabled);
}

Tensor wrap_native_group_norm(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input);
    return at::redispatch::native_group_norm(input, weight, bias, N, C, HxW, group, eps);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_NATIVE_GROUP_NORM, input, weight, bias, N, C, HxW, group, eps);
}

Tensor wrap_native_group_norm_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_out, input, mean, rstd);
    return at::redispatch::native_group_norm_backward(grad_out, input, mean, rstd, weight, N, C, HxW, group, output_mask);
  }
  return MK_TORCHY(grad_out.dtype(), grad_out.device(), H_NATIVE_GROUP_NORM_BACKWARD, grad_out, input, mean, rstd, weight, N, C, HxW, group, output_mask);
}

Tensor wrap__fft_r2c(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_fft_r2c(self, dim, normalization, onesided);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__FFT_R2C, self, dim, normalization, onesided);
}

Tensor wrap__fft_r2c_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::_fft_r2c(out, self, dim, normalization, onesided);
  }
  return MK_TORCHY(out.dtype(), out.device(), H__FFT_R2C_OUT, out, self, dim, normalization, onesided);
}

Tensor wrap__fft_c2r(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_fft_c2r(self, dim, normalization, last_dim_size);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__FFT_C2R, self, dim, normalization, last_dim_size);
}

Tensor wrap__fft_c2r_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::_fft_c2r(out, self, dim, normalization, last_dim_size);
  }
  return MK_TORCHY(out.dtype(), out.device(), H__FFT_C2R_OUT, out, self, dim, normalization, last_dim_size);
}

Tensor wrap__fft_c2c(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_fft_c2c(self, dim, normalization, forward);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__FFT_C2C, self, dim, normalization, forward);
}

Tensor wrap__fft_c2c_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::_fft_c2c(out, self, dim, normalization, forward);
  }
  return MK_TORCHY(out.dtype(), out.device(), H__FFT_C2C_OUT, out, self, dim, normalization, forward);
}

int wrap__cufft_get_plan_cache_size(args...) {
  ensure_materialized();
  return at::redispatch::_cufft_get_plan_cache_size(device_index);
}

int wrap__cufft_get_plan_cache_max_size(args...) {
  ensure_materialized();
  return at::redispatch::_cufft_get_plan_cache_max_size(device_index);
}

void wrap__cufft_set_plan_cache_max_size(args...) {
  ensure_materialized();
  return at::redispatch::_cufft_set_plan_cache_max_size(device_index, max_size);
}

void wrap__cufft_clear_plan_cache(args...) {
  ensure_materialized();
  return at::redispatch::_cufft_clear_plan_cache(device_index);
}

Tensor wrap_index_Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::index(self, indices);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INDEX_TENSOR, self, indices);
}

Tensor wrap_index_copy_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index, source);
    return at::redispatch::index_copy_(self, dim, index, source);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INDEX_COPY_, self, dim, index, source);
}

Tensor wrap_index_copy(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index, source);
    return at::redispatch::index_copy(self, dim, index, source);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INDEX_COPY, self, dim, index, source);
}

Tensor wrap_index_copy__dimname(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index, source);
    return at::redispatch::index_copy_(self, dim, index, source);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INDEX_COPY__DIMNAME, self, dim, index, source);
}

Tensor wrap_index_copy_dimname(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index, source);
    return at::redispatch::index_copy(self, dim, index, source);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INDEX_COPY_DIMNAME, self, dim, index, source);
}

Tensor wrap_index_put_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, values);
    return at::redispatch::index_put_(self, indices, values, accumulate);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INDEX_PUT_, self, indices, values, accumulate);
}

Tensor wrap_index_put(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, values);
    return at::redispatch::index_put(self, indices, values, accumulate);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INDEX_PUT, self, indices, values, accumulate);
}

Tensor wrap__index_put_impl_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, values);
    return at::redispatch::_index_put_impl_(self, indices, values, accumulate, unsafe);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__INDEX_PUT_IMPL_, self, indices, values, accumulate, unsafe);
}

Tensor wrap_instance_norm(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input);
    return at::redispatch::instance_norm(input, weight, bias, running_mean, running_var, use_input_stats, momentum, eps, cudnn_enabled);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_INSTANCE_NORM, input, weight, bias, running_mean, running_var, use_input_stats, momentum, eps, cudnn_enabled);
}

Tensor wrap_inverse(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::inverse(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INVERSE, self);
}

Tensor wrap_inverse_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::inverse(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_INVERSE_OUT, out, self);
}

Tensor wrap__inverse_helper(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_inverse_helper(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__INVERSE_HELPER, self);
}

Tensor wrap_isclose(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::isclose(self, other, rtol, atol, equal_nan);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ISCLOSE, self, other, rtol, atol, equal_nan);
}

Tensor wrap_isnan(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::isnan(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ISNAN, self);
}

bool wrap_is_distributed(args...) {
  ensure_materialized(self);
  return at::redispatch::is_distributed(self);
}

Tensor wrap_isreal(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::isreal(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ISREAL, self);
}

bool wrap_is_nonzero(args...) {
  ensure_materialized(self);
  return at::redispatch::is_nonzero(self);
}

bool wrap_is_same_size(args...) {
  ensure_materialized(self, other);
  return at::redispatch::is_same_size(self, other);
}

Tensor wrap_kl_div(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, target);
    return at::redispatch::kl_div(self, target, reduction, log_target);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_KL_DIV, self, target, reduction, log_target);
}

Tensor wrap_kl_div_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, target);
    return at::redispatch::kl_div_backward(grad_output, self, target, reduction, log_target);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_KL_DIV_BACKWARD, grad_output, self, target, reduction, log_target);
}

Tensor wrap_kron(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::kron(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_KRON, self, other);
}

Tensor wrap_kron_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::kron(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_KRON_OUT, out, self, other);
}

Tensor wrap_kthvalue(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::kthvalue(self, k, dim, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_KTHVALUE, self, k, dim, keepdim);
}

Tensor wrap_kthvalue_values(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(values, indices, self);
    return at::redispatch::kthvalue(values, indices, self, k, dim, keepdim);
  }
  return MK_TORCHY(values.dtype(), values.device(), H_KTHVALUE_VALUES, values, indices, self, k, dim, keepdim);
}

Tensor wrap_kthvalue_dimname(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::kthvalue(self, k, dim, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_KTHVALUE_DIMNAME, self, k, dim, keepdim);
}

Tensor wrap_kthvalue_dimname_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(values, indices, self);
    return at::redispatch::kthvalue(values, indices, self, k, dim, keepdim);
  }
  return MK_TORCHY(values.dtype(), values.device(), H_KTHVALUE_DIMNAME_OUT, values, indices, self, k, dim, keepdim);
}

Tensor wrap_layer_norm(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input);
    return at::redispatch::layer_norm(input, normalized_shape, weight, bias, eps, cudnn_enable);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_LAYER_NORM, input, normalized_shape, weight, bias, eps, cudnn_enable);
}

Tensor wrap_native_layer_norm(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input);
    return at::redispatch::native_layer_norm(input, normalized_shape, weight, bias, eps);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_NATIVE_LAYER_NORM, input, normalized_shape, weight, bias, eps);
}

Tensor wrap_native_layer_norm_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_out, input, mean, rstd);
    return at::redispatch::native_layer_norm_backward(grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask);
  }
  return MK_TORCHY(grad_out.dtype(), grad_out.device(), H_NATIVE_LAYER_NORM_BACKWARD, grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask);
}

Tensor wrap_nan_to_num(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::nan_to_num(self, nan, posinf, neginf);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NAN_TO_NUM, self, nan, posinf, neginf);
}

Tensor wrap_nan_to_num_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::nan_to_num_(self, nan, posinf, neginf);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NAN_TO_NUM_, self, nan, posinf, neginf);
}

Tensor wrap_nan_to_num_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::nan_to_num(out, self, nan, posinf, neginf);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NAN_TO_NUM_OUT, out, self, nan, posinf, neginf);
}

Tensor wrap_linear(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, weight);
    return at::redispatch::linear(input, weight, bias);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_LINEAR, input, weight, bias);
}

Tensor wrap_mkldnn_linear(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, weight);
    return at::redispatch::mkldnn_linear(self, weight, bias);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MKLDNN_LINEAR, self, weight, bias);
}

Tensor wrap_mkldnn_linear_backward_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, weight);
    return at::redispatch::mkldnn_linear_backward_input(input_size, grad_output, weight);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_MKLDNN_LINEAR_BACKWARD_INPUT, input_size, grad_output, weight);
}

Tensor wrap_mkldnn_linear_backward_weights(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, input, weight);
    return at::redispatch::mkldnn_linear_backward_weights(grad_output, input, weight, bias_defined);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_MKLDNN_LINEAR_BACKWARD_WEIGHTS, grad_output, input, weight, bias_defined);
}

Tensor wrap_mkldnn_linear_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, grad_output, weight);
    return at::redispatch::mkldnn_linear_backward(self, grad_output, weight, output_mask);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MKLDNN_LINEAR_BACKWARD, self, grad_output, weight, output_mask);
}

Tensor wrap_fbgemm_linear_int8_weight_fp32_activation(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, weight, packed, col_offsets, bias);
    return at::redispatch::fbgemm_linear_int8_weight_fp32_activation(input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_FBGEMM_LINEAR_INT8_WEIGHT_FP32_ACTIVATION, input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias);
}

Tensor wrap_fbgemm_linear_int8_weight(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, weight, packed, col_offsets, bias);
    return at::redispatch::fbgemm_linear_int8_weight(input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_FBGEMM_LINEAR_INT8_WEIGHT, input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias);
}

Tensor wrap_fbgemm_linear_quantize_weight(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input);
    return at::redispatch::fbgemm_linear_quantize_weight(input);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_FBGEMM_LINEAR_QUANTIZE_WEIGHT, input);
}

Tensor wrap_fbgemm_pack_gemm_matrix_fp16(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input);
    return at::redispatch::fbgemm_pack_gemm_matrix_fp16(input);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_FBGEMM_PACK_GEMM_MATRIX_FP16, input);
}

Tensor wrap_fbgemm_linear_fp16_weight_fp32_activation(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, packed_weight, bias);
    return at::redispatch::fbgemm_linear_fp16_weight_fp32_activation(input, packed_weight, bias);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_FBGEMM_LINEAR_FP16_WEIGHT_FP32_ACTIVATION, input, packed_weight, bias);
}

Tensor wrap_fbgemm_linear_fp16_weight(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, packed_weight, bias);
    return at::redispatch::fbgemm_linear_fp16_weight(input, packed_weight, bias);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_FBGEMM_LINEAR_FP16_WEIGHT, input, packed_weight, bias);
}

Tensor wrap_fbgemm_pack_quantized_matrix(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input);
    return at::redispatch::fbgemm_pack_quantized_matrix(input);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_FBGEMM_PACK_QUANTIZED_MATRIX, input);
}

Tensor wrap_fbgemm_pack_quantized_matrix_KN(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input);
    return at::redispatch::fbgemm_pack_quantized_matrix(input, K, N);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_FBGEMM_PACK_QUANTIZED_MATRIX_KN, input, K, N);
}

Tensor wrap_ldexp_Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::ldexp(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LDEXP_TENSOR, self, other);
}

Tensor wrap_ldexp_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::ldexp_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LDEXP_, self, other);
}

Tensor wrap_ldexp_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::ldexp(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LDEXP_OUT, out, self, other);
}

Tensor wrap_linspace(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::linspace(start, end, steps, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_LINSPACE, start, end, steps, dtype, layout, device, pin_memory);
}

Tensor wrap_linspace_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::linspace(out, start, end, steps);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LINSPACE_OUT, out, start, end, steps);
}

Tensor wrap_log_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::log(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LOG_OUT, out, self);
}

Tensor wrap_log10_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::log10(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LOG10_OUT, out, self);
}

Tensor wrap_log1p(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::log1p(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LOG1P, self);
}

Tensor wrap_log1p_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::log1p_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LOG1P_, self);
}

Tensor wrap_log1p_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::log1p(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LOG1P_OUT, out, self);
}

Tensor wrap_log2_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::log2(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LOG2_OUT, out, self);
}

Tensor wrap_logaddexp_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::logaddexp(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LOGADDEXP_OUT, out, self, other);
}

Tensor wrap_logaddexp(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::logaddexp(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LOGADDEXP, self, other);
}

Tensor wrap_logaddexp2_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::logaddexp2(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LOGADDEXP2_OUT, out, self, other);
}

Tensor wrap_logaddexp2(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::logaddexp2(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LOGADDEXP2, self, other);
}

Tensor wrap_xlogy_Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::xlogy(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_XLOGY_TENSOR, self, other);
}

Tensor wrap_xlogy_Scalar_Self(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(other);
    return at::redispatch::xlogy(self, other);
  }
  return MK_TORCHY(other.dtype(), other.device(), H_XLOGY_SCALAR_SELF, self, other);
}

Tensor wrap_xlogy_Scalar_Other(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::xlogy(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_XLOGY_SCALAR_OTHER, self, other);
}

Tensor wrap_xlogy__Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::xlogy_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_XLOGY__TENSOR, self, other);
}

Tensor wrap_xlogy__Scalar_Other(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::xlogy_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_XLOGY__SCALAR_OTHER, self, other);
}

Tensor wrap_xlogy_OutTensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::xlogy(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_XLOGY_OUTTENSOR, out, self, other);
}

Tensor wrap_xlogy_OutScalar_Self(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, other);
    return at::redispatch::xlogy(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_XLOGY_OUTSCALAR_SELF, out, self, other);
}

Tensor wrap_xlogy_OutScalar_Other(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::xlogy(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_XLOGY_OUTSCALAR_OTHER, out, self, other);
}

Tensor wrap_logdet(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::logdet(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LOGDET, self);
}

Tensor wrap_logspace(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::logspace(start, end, steps, base, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_LOGSPACE, start, end, steps, base, dtype, layout, device, pin_memory);
}

Tensor wrap_logspace_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::logspace(out, start, end, steps, base);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LOGSPACE_OUT, out, start, end, steps, base);
}

Tensor wrap_log_softmax_int(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::log_softmax(self, dim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LOG_SOFTMAX_INT, self, dim, dtype);
}

Tensor wrap_log_softmax_Dimname(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::log_softmax(self, dim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LOG_SOFTMAX_DIMNAME, self, dim, dtype);
}

Tensor wrap__log_softmax(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_log_softmax(self, dim, half_to_float);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__LOG_SOFTMAX, self, dim, half_to_float);
}

Tensor wrap__log_softmax_backward_data(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, output, self);
    return at::redispatch::_log_softmax_backward_data(grad_output, output, dim, self);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H__LOG_SOFTMAX_BACKWARD_DATA, grad_output, output, dim, self);
}

Tensor wrap__logcumsumexp(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_logcumsumexp(self, dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__LOGCUMSUMEXP, self, dim);
}

Tensor wrap__logcumsumexp_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::_logcumsumexp(out, self, dim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H__LOGCUMSUMEXP_OUT, out, self, dim);
}

Tensor wrap_logcumsumexp(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::logcumsumexp(self, dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LOGCUMSUMEXP, self, dim);
}

Tensor wrap_logcumsumexp_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::logcumsumexp(out, self, dim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LOGCUMSUMEXP_OUT, out, self, dim);
}

Tensor wrap_logcumsumexp_dimname(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::logcumsumexp(self, dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LOGCUMSUMEXP_DIMNAME, self, dim);
}

Tensor wrap_logcumsumexp_dimname_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::logcumsumexp(out, self, dim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LOGCUMSUMEXP_DIMNAME_OUT, out, self, dim);
}

Tensor wrap_logsumexp(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::logsumexp(self, dim, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LOGSUMEXP, self, dim, keepdim);
}

Tensor wrap_logsumexp_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::logsumexp(out, self, dim, keepdim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LOGSUMEXP_OUT, out, self, dim, keepdim);
}

Tensor wrap_logsumexp_names(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::logsumexp(self, dim, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LOGSUMEXP_NAMES, self, dim, keepdim);
}

Tensor wrap_logsumexp_names_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::logsumexp(out, self, dim, keepdim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LOGSUMEXP_NAMES_OUT, out, self, dim, keepdim);
}

Tensor wrap_margin_ranking_loss(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input1, input2, target);
    return at::redispatch::margin_ranking_loss(input1, input2, target, margin, reduction);
  }
  return MK_TORCHY(input1.dtype(), input1.device(), H_MARGIN_RANKING_LOSS, input1, input2, target, margin, reduction);
}

Tensor wrap_matmul(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::matmul(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MATMUL, self, other);
}

Tensor wrap_matmul_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::matmul(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_MATMUL_OUT, out, self, other);
}

Tensor wrap_matrix_rank_tol(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::matrix_rank(self, tol, symmetric);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MATRIX_RANK_TOL, self, tol, symmetric);
}

Tensor wrap_matrix_rank(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::matrix_rank(self, symmetric);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MATRIX_RANK, self, symmetric);
}

Tensor wrap_matrix_power(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::matrix_power(self, n);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MATRIX_POWER, self, n);
}

Tensor wrap_matrix_power_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::matrix_power(out, self, n);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_MATRIX_POWER_OUT, out, self, n);
}

Tensor wrap_matrix_exp(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::matrix_exp(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MATRIX_EXP, self);
}

Tensor wrap_matrix_exp_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, grad);
    return at::redispatch::matrix_exp_backward(self, grad);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MATRIX_EXP_BACKWARD, self, grad);
}

Tensor wrap__aminmax(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_aminmax(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__AMINMAX, self);
}

Tensor wrap__aminmax_dim(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_aminmax(self, dim, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__AMINMAX_DIM, self, dim, keepdim);
}

Tensor wrap__compute_linear_combination(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, coefficients);
    return at::redispatch::_compute_linear_combination(input, coefficients);
  }
  return MK_TORCHY(input.dtype(), input.device(), H__COMPUTE_LINEAR_COMBINATION, input, coefficients);
}

Tensor wrap__compute_linear_combination_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, input, coefficients);
    return at::redispatch::_compute_linear_combination(out, input, coefficients);
  }
  return MK_TORCHY(out.dtype(), out.device(), H__COMPUTE_LINEAR_COMBINATION_OUT, out, input, coefficients);
}

Tensor wrap_max_dim(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::max(self, dim, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MAX_DIM, self, dim, keepdim);
}

Tensor wrap_max_dim_max(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(max, max_values, self);
    return at::redispatch::max(max, max_values, self, dim, keepdim);
  }
  return MK_TORCHY(max.dtype(), max.device(), H_MAX_DIM_MAX, max, max_values, self, dim, keepdim);
}

Tensor wrap_max_names_dim(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::max(self, dim, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MAX_NAMES_DIM, self, dim, keepdim);
}

Tensor wrap_max_names_dim_max(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(max, max_values, self);
    return at::redispatch::max(max, max_values, self, dim, keepdim);
  }
  return MK_TORCHY(max.dtype(), max.device(), H_MAX_NAMES_DIM_MAX, max, max_values, self, dim, keepdim);
}

Tensor wrap_value_selecting_reduction_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad, indices);
    return at::redispatch::value_selecting_reduction_backward(grad, dim, indices, sizes, keepdim);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H_VALUE_SELECTING_REDUCTION_BACKWARD, grad, dim, indices, sizes, keepdim);
}

Tensor wrap_amax(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::amax(self, dim, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_AMAX, self, dim, keepdim);
}

Tensor wrap_amax_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::amax(out, self, dim, keepdim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_AMAX_OUT, out, self, dim, keepdim);
}

Tensor wrap_max_pool1d_with_indices(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::max_pool1d_with_indices(self, kernel_size, stride, padding, dilation, ceil_mode);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MAX_POOL1D_WITH_INDICES, self, kernel_size, stride, padding, dilation, ceil_mode);
}

Tensor wrap_max_pool1d(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::max_pool1d(self, kernel_size, stride, padding, dilation, ceil_mode);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MAX_POOL1D, self, kernel_size, stride, padding, dilation, ceil_mode);
}

Tensor wrap_max_pool2d(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::max_pool2d(self, kernel_size, stride, padding, dilation, ceil_mode);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MAX_POOL2D, self, kernel_size, stride, padding, dilation, ceil_mode);
}

Tensor wrap_mkldnn_max_pool2d(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::mkldnn_max_pool2d(self, kernel_size, stride, padding, dilation, ceil_mode);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MKLDNN_MAX_POOL2D, self, kernel_size, stride, padding, dilation, ceil_mode);
}

Tensor wrap_mkldnn_max_pool2d_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, output, input);
    return at::redispatch::mkldnn_max_pool2d_backward(grad_output, output, input, kernel_size, stride, padding, dilation, ceil_mode);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_MKLDNN_MAX_POOL2D_BACKWARD, grad_output, output, input, kernel_size, stride, padding, dilation, ceil_mode);
}

Tensor wrap_mkldnn_max_pool3d(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::mkldnn_max_pool3d(self, kernel_size, stride, padding, dilation, ceil_mode);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MKLDNN_MAX_POOL3D, self, kernel_size, stride, padding, dilation, ceil_mode);
}

Tensor wrap_mkldnn_max_pool3d_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, output, input);
    return at::redispatch::mkldnn_max_pool3d_backward(grad_output, output, input, kernel_size, stride, padding, dilation, ceil_mode);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_MKLDNN_MAX_POOL3D_BACKWARD, grad_output, output, input, kernel_size, stride, padding, dilation, ceil_mode);
}

Tensor wrap_quantized_max_pool1d(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::quantized_max_pool1d(self, kernel_size, stride, padding, dilation, ceil_mode);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_QUANTIZED_MAX_POOL1D, self, kernel_size, stride, padding, dilation, ceil_mode);
}

Tensor wrap_quantized_max_pool2d(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::quantized_max_pool2d(self, kernel_size, stride, padding, dilation, ceil_mode);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_QUANTIZED_MAX_POOL2D, self, kernel_size, stride, padding, dilation, ceil_mode);
}

Tensor wrap_max_pool3d(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::max_pool3d(self, kernel_size, stride, padding, dilation, ceil_mode);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MAX_POOL3D, self, kernel_size, stride, padding, dilation, ceil_mode);
}

Tensor wrap_mean(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::mean(self, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MEAN, self, dtype);
}

Tensor wrap_mean_dim(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::mean(self, dim, keepdim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MEAN_DIM, self, dim, keepdim, dtype);
}

Tensor wrap_mean_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::mean(out, self, dim, keepdim, dtype);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_MEAN_OUT, out, self, dim, keepdim, dtype);
}

Tensor wrap_mean_names_dim(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::mean(self, dim, keepdim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MEAN_NAMES_DIM, self, dim, keepdim, dtype);
}

Tensor wrap_mean_names_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::mean(out, self, dim, keepdim, dtype);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_MEAN_NAMES_OUT, out, self, dim, keepdim, dtype);
}

Tensor wrap_median(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::median(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MEDIAN, self);
}

Tensor wrap_median_dim(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::median(self, dim, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MEDIAN_DIM, self, dim, keepdim);
}

Tensor wrap_median_dim_values(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(values, indices, self);
    return at::redispatch::median(values, indices, self, dim, keepdim);
  }
  return MK_TORCHY(values.dtype(), values.device(), H_MEDIAN_DIM_VALUES, values, indices, self, dim, keepdim);
}

Tensor wrap_median_names_dim(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::median(self, dim, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MEDIAN_NAMES_DIM, self, dim, keepdim);
}

Tensor wrap_median_names_dim_values(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(values, indices, self);
    return at::redispatch::median(values, indices, self, dim, keepdim);
  }
  return MK_TORCHY(values.dtype(), values.device(), H_MEDIAN_NAMES_DIM_VALUES, values, indices, self, dim, keepdim);
}

Tensor wrap_nanmedian(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::nanmedian(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NANMEDIAN, self);
}

Tensor wrap_nanmedian_dim(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::nanmedian(self, dim, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NANMEDIAN_DIM, self, dim, keepdim);
}

Tensor wrap_nanmedian_dim_values(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(values, indices, self);
    return at::redispatch::nanmedian(values, indices, self, dim, keepdim);
  }
  return MK_TORCHY(values.dtype(), values.device(), H_NANMEDIAN_DIM_VALUES, values, indices, self, dim, keepdim);
}

Tensor wrap_nanmedian_names_dim(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::nanmedian(self, dim, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NANMEDIAN_NAMES_DIM, self, dim, keepdim);
}

Tensor wrap_nanmedian_names_dim_values(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(values, indices, self);
    return at::redispatch::nanmedian(values, indices, self, dim, keepdim);
  }
  return MK_TORCHY(values.dtype(), values.device(), H_NANMEDIAN_NAMES_DIM_VALUES, values, indices, self, dim, keepdim);
}

Tensor wrap_min_dim(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::min(self, dim, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MIN_DIM, self, dim, keepdim);
}

Tensor wrap_min_dim_min(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(min, min_indices, self);
    return at::redispatch::min(min, min_indices, self, dim, keepdim);
  }
  return MK_TORCHY(min.dtype(), min.device(), H_MIN_DIM_MIN, min, min_indices, self, dim, keepdim);
}

Tensor wrap_min_names_dim(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::min(self, dim, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MIN_NAMES_DIM, self, dim, keepdim);
}

Tensor wrap_min_names_dim_min(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(min, min_indices, self);
    return at::redispatch::min(min, min_indices, self, dim, keepdim);
  }
  return MK_TORCHY(min.dtype(), min.device(), H_MIN_NAMES_DIM_MIN, min, min_indices, self, dim, keepdim);
}

Tensor wrap_amin(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::amin(self, dim, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_AMIN, self, dim, keepdim);
}

Tensor wrap_amin_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::amin(out, self, dim, keepdim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_AMIN_OUT, out, self, dim, keepdim);
}

Tensor wrap_mkldnn_convolution(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, weight);
    return at::redispatch::mkldnn_convolution(self, weight, bias, padding, stride, dilation, groups);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MKLDNN_CONVOLUTION, self, weight, bias, padding, stride, dilation, groups);
}

Tensor wrap_mkldnn_convolution_backward_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, weight);
    return at::redispatch::mkldnn_convolution_backward_input(self_size, grad_output, weight, padding, stride, dilation, groups, bias_defined);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_MKLDNN_CONVOLUTION_BACKWARD_INPUT, self_size, grad_output, weight, padding, stride, dilation, groups, bias_defined);
}

Tensor wrap_mkldnn_convolution_backward_weights(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self);
    return at::redispatch::mkldnn_convolution_backward_weights(weight_size, grad_output, self, padding, stride, dilation, groups, bias_defined);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_MKLDNN_CONVOLUTION_BACKWARD_WEIGHTS, weight_size, grad_output, self, padding, stride, dilation, groups, bias_defined);
}

Tensor wrap_mkldnn_convolution_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, grad_output, weight);
    return at::redispatch::mkldnn_convolution_backward(self, grad_output, weight, padding, stride, dilation, groups, output_mask);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MKLDNN_CONVOLUTION_BACKWARD, self, grad_output, weight, padding, stride, dilation, groups, output_mask);
}

Tensor wrap_miopen_batch_norm(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, weight);
    return at::redispatch::miopen_batch_norm(input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_MIOPEN_BATCH_NORM, input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon);
}

Tensor wrap_miopen_batch_norm_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, grad_output, weight);
    return at::redispatch::miopen_batch_norm_backward(input, grad_output, weight, running_mean, running_var, save_mean, save_var, epsilon);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_MIOPEN_BATCH_NORM_BACKWARD, input, grad_output, weight, running_mean, running_var, save_mean, save_var, epsilon);
}

Tensor wrap_miopen_convolution(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, weight);
    return at::redispatch::miopen_convolution(self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MIOPEN_CONVOLUTION, self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic);
}

Tensor wrap_miopen_convolution_backward_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, weight);
    return at::redispatch::miopen_convolution_backward_input(self_size, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_MIOPEN_CONVOLUTION_BACKWARD_INPUT, self_size, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
}

Tensor wrap_miopen_convolution_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, grad_output, weight);
    return at::redispatch::miopen_convolution_backward(self, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, output_mask);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MIOPEN_CONVOLUTION_BACKWARD, self, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, output_mask);
}

Tensor wrap_miopen_convolution_backward_bias(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output);
    return at::redispatch::miopen_convolution_backward_bias(grad_output);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_MIOPEN_CONVOLUTION_BACKWARD_BIAS, grad_output);
}

Tensor wrap_miopen_convolution_backward_weight(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self);
    return at::redispatch::miopen_convolution_backward_weight(weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_MIOPEN_CONVOLUTION_BACKWARD_WEIGHT, weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
}

Tensor wrap_miopen_convolution_transpose(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, weight);
    return at::redispatch::miopen_convolution_transpose(self, weight, bias, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MIOPEN_CONVOLUTION_TRANSPOSE, self, weight, bias, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
}

Tensor wrap_miopen_convolution_transpose_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, grad_output, weight);
    return at::redispatch::miopen_convolution_transpose_backward(self, grad_output, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic, output_mask);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MIOPEN_CONVOLUTION_TRANSPOSE_BACKWARD, self, grad_output, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic, output_mask);
}

Tensor wrap_miopen_convolution_transpose_backward_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, weight);
    return at::redispatch::miopen_convolution_transpose_backward_input(grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_MIOPEN_CONVOLUTION_TRANSPOSE_BACKWARD_INPUT, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
}

Tensor wrap_miopen_convolution_transpose_backward_weight(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self);
    return at::redispatch::miopen_convolution_transpose_backward_weight(weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_MIOPEN_CONVOLUTION_TRANSPOSE_BACKWARD_WEIGHT, weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
}

Tensor wrap_miopen_depthwise_convolution(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, weight);
    return at::redispatch::miopen_depthwise_convolution(self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MIOPEN_DEPTHWISE_CONVOLUTION, self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic);
}

Tensor wrap_miopen_depthwise_convolution_backward_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, weight);
    return at::redispatch::miopen_depthwise_convolution_backward_input(self_size, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_MIOPEN_DEPTHWISE_CONVOLUTION_BACKWARD_INPUT, self_size, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
}

Tensor wrap_miopen_depthwise_convolution_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, grad_output, weight);
    return at::redispatch::miopen_depthwise_convolution_backward(self, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, output_mask);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MIOPEN_DEPTHWISE_CONVOLUTION_BACKWARD, self, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, output_mask);
}

Tensor wrap_miopen_depthwise_convolution_backward_weight(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self);
    return at::redispatch::miopen_depthwise_convolution_backward_weight(weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_MIOPEN_DEPTHWISE_CONVOLUTION_BACKWARD_WEIGHT, weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
}

Tensor wrap_miopen_rnn(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, hx);
    return at::redispatch::miopen_rnn(input, weight, weight_stride0, hx, cx, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_MIOPEN_RNN, input, weight, weight_stride0, hx, cx, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state);
}

Tensor wrap_miopen_rnn_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, weight_buf, hx, output, reserve);
    return at::redispatch::miopen_rnn_backward(input, weight, weight_stride0, weight_buf, hx, cx, output, grad_output, grad_hy, grad_cy, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state, reserve, output_mask);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_MIOPEN_RNN_BACKWARD, input, weight, weight_stride0, weight_buf, hx, cx, output, grad_output, grad_hy, grad_cy, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state, reserve, output_mask);
}

Tensor wrap_mm(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, mat2);
    return at::redispatch::mm(self, mat2);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MM, self, mat2);
}

Tensor wrap_mm_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, mat2);
    return at::redispatch::mm(out, self, mat2);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_MM_OUT, out, self, mat2);
}

Tensor wrap__sparse_mm(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(sparse, dense);
    return at::redispatch::_sparse_mm(sparse, dense);
  }
  return MK_TORCHY(sparse.dtype(), sparse.device(), H__SPARSE_MM, sparse, dense);
}

Tensor wrap__sparse_sparse_matmul(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::_sparse_sparse_matmul(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__SPARSE_SPARSE_MATMUL, self, other);
}

Tensor wrap__sparse_mask_helper(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(t, mask_indices);
    return at::redispatch::_sparse_mask_helper(t, mask_indices);
  }
  return MK_TORCHY(t.dtype(), t.device(), H__SPARSE_MASK_HELPER, t, mask_indices);
}

Tensor wrap_mode(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::mode(self, dim, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MODE, self, dim, keepdim);
}

Tensor wrap_mode_values(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(values, indices, self);
    return at::redispatch::mode(values, indices, self, dim, keepdim);
  }
  return MK_TORCHY(values.dtype(), values.device(), H_MODE_VALUES, values, indices, self, dim, keepdim);
}

Tensor wrap_mode_dimname(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::mode(self, dim, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MODE_DIMNAME, self, dim, keepdim);
}

Tensor wrap_mode_dimname_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(values, indices, self);
    return at::redispatch::mode(values, indices, self, dim, keepdim);
  }
  return MK_TORCHY(values.dtype(), values.device(), H_MODE_DIMNAME_OUT, values, indices, self, dim, keepdim);
}

Tensor wrap_mul_Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::mul(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MUL_TENSOR, self, other);
}

Tensor wrap_mul__Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::mul_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MUL__TENSOR, self, other);
}

Tensor wrap_mul_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::mul(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_MUL_OUT, out, self, other);
}

Tensor wrap_mul_Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::mul(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MUL_SCALAR, self, other);
}

Tensor wrap_mul__Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::mul_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MUL__SCALAR, self, other);
}

Tensor wrap_multiply_Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::multiply(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MULTIPLY_TENSOR, self, other);
}

Tensor wrap_multiply__Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::multiply_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MULTIPLY__TENSOR, self, other);
}

Tensor wrap_multiply_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::multiply(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_MULTIPLY_OUT, out, self, other);
}

Tensor wrap_multiply_Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::multiply(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MULTIPLY_SCALAR, self, other);
}

Tensor wrap_multiply__Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::multiply_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MULTIPLY__SCALAR, self, other);
}

Tensor wrap_mv(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, vec);
    return at::redispatch::mv(self, vec);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MV, self, vec);
}

Tensor wrap_mv_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, vec);
    return at::redispatch::mv(out, self, vec);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_MV_OUT, out, self, vec);
}

Tensor wrap_mvlgamma(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::mvlgamma(self, p);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MVLGAMMA, self, p);
}

Tensor wrap_mvlgamma_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::mvlgamma_(self, p);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MVLGAMMA_, self, p);
}

Tensor wrap_narrow_copy(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::narrow_copy(self, dim, start, length);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NARROW_COPY, self, dim, start, length);
}

Tensor wrap_narrow_copy_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::narrow_copy(out, self, dim, start, length);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NARROW_COPY_OUT, out, self, dim, start, length);
}

Tensor wrap_narrow(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::narrow(self, dim, start, length);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NARROW, self, dim, start, length);
}

Tensor wrap_narrow_Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, start);
    return at::redispatch::narrow(self, dim, start, length);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NARROW_TENSOR, self, dim, start, length);
}

Tensor wrap_native_batch_norm(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input);
    return at::redispatch::native_batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_NATIVE_BATCH_NORM, input, weight, bias, running_mean, running_var, training, momentum, eps);
}

Tensor wrap_native_batch_norm_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, save_mean, save_invstd, input);
    return at::redispatch::native_batch_norm(out, save_mean, save_invstd, input, weight, bias, running_mean, running_var, training, momentum, eps);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NATIVE_BATCH_NORM_OUT, out, save_mean, save_invstd, input, weight, bias, running_mean, running_var, training, momentum, eps);
}

Tensor wrap_batch_norm_stats(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input);
    return at::redispatch::batch_norm_stats(input, eps);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_BATCH_NORM_STATS, input, eps);
}

Tensor wrap_batch_norm_elemt(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, mean, invstd);
    return at::redispatch::batch_norm_elemt(input, weight, bias, mean, invstd, eps);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_BATCH_NORM_ELEMT, input, weight, bias, mean, invstd, eps);
}

Tensor wrap_batch_norm_elemt_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, input, mean, invstd);
    return at::redispatch::batch_norm_elemt(out, input, weight, bias, mean, invstd, eps);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_BATCH_NORM_ELEMT_OUT, out, input, weight, bias, mean, invstd, eps);
}

Tensor wrap_batch_norm_gather_stats(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, mean, invstd);
    return at::redispatch::batch_norm_gather_stats(input, mean, invstd, running_mean, running_var, momentum, eps, count);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_BATCH_NORM_GATHER_STATS, input, mean, invstd, running_mean, running_var, momentum, eps, count);
}

Tensor wrap_batch_norm_gather_stats_with_counts(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, mean, invstd, counts);
    return at::redispatch::batch_norm_gather_stats_with_counts(input, mean, invstd, running_mean, running_var, momentum, eps, counts);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_BATCH_NORM_GATHER_STATS_WITH_COUNTS, input, mean, invstd, running_mean, running_var, momentum, eps, counts);
}

Tensor wrap_native_batch_norm_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_out, input);
    return at::redispatch::native_batch_norm_backward(grad_out, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps, output_mask);
  }
  return MK_TORCHY(grad_out.dtype(), grad_out.device(), H_NATIVE_BATCH_NORM_BACKWARD, grad_out, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps, output_mask);
}

Tensor wrap_batch_norm_backward_reduce(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_out, input, mean, invstd);
    return at::redispatch::batch_norm_backward_reduce(grad_out, input, mean, invstd, weight, input_g, weight_g, bias_g);
  }
  return MK_TORCHY(grad_out.dtype(), grad_out.device(), H_BATCH_NORM_BACKWARD_REDUCE, grad_out, input, mean, invstd, weight, input_g, weight_g, bias_g);
}

Tensor wrap_batch_norm_backward_elemt(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_out, input, mean, invstd, mean_dy, mean_dy_xmu, count);
    return at::redispatch::batch_norm_backward_elemt(grad_out, input, mean, invstd, weight, mean_dy, mean_dy_xmu, count);
  }
  return MK_TORCHY(grad_out.dtype(), grad_out.device(), H_BATCH_NORM_BACKWARD_ELEMT, grad_out, input, mean, invstd, weight, mean_dy, mean_dy_xmu, count);
}

Tensor wrap_batch_norm_update_stats(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input);
    return at::redispatch::batch_norm_update_stats(input, running_mean, running_var, momentum);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_BATCH_NORM_UPDATE_STATS, input, running_mean, running_var, momentum);
}

bool wrap_is_vulkan_available(args...) {
  ensure_materialized();
  return at::redispatch::is_vulkan_available();
}

bool wrap__nnpack_available(args...) {
  ensure_materialized();
  return at::redispatch::_nnpack_available();
}

Tensor wrap__nnpack_spatial_convolution(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, weight);
    return at::redispatch::_nnpack_spatial_convolution(input, weight, bias, padding, stride);
  }
  return MK_TORCHY(input.dtype(), input.device(), H__NNPACK_SPATIAL_CONVOLUTION, input, weight, bias, padding, stride);
}

Tensor wrap__nnpack_spatial_convolution_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, grad_output, weight);
    return at::redispatch::_nnpack_spatial_convolution_backward(input, grad_output, weight, padding, output_mask);
  }
  return MK_TORCHY(input.dtype(), input.device(), H__NNPACK_SPATIAL_CONVOLUTION_BACKWARD, input, grad_output, weight, padding, output_mask);
}

Tensor wrap__nnpack_spatial_convolution_backward_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, grad_output, weight);
    return at::redispatch::_nnpack_spatial_convolution_backward_input(input, grad_output, weight, padding);
  }
  return MK_TORCHY(input.dtype(), input.device(), H__NNPACK_SPATIAL_CONVOLUTION_BACKWARD_INPUT, input, grad_output, weight, padding);
}

Tensor wrap__nnpack_spatial_convolution_backward_weight(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, grad_output);
    return at::redispatch::_nnpack_spatial_convolution_backward_weight(input, weightsize, grad_output, padding);
  }
  return MK_TORCHY(input.dtype(), input.device(), H__NNPACK_SPATIAL_CONVOLUTION_BACKWARD_WEIGHT, input, weightsize, grad_output, padding);
}

Tensor wrap_ones_names(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::ones(size, names, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_ONES_NAMES, size, names, dtype, layout, device, pin_memory);
}

Tensor wrap_ones(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::ones(size, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_ONES, size, dtype, layout, device, pin_memory);
}

Tensor wrap_ones_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::ones(out, size);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ONES_OUT, out, size);
}

Tensor wrap_ones_like(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::ones_like(self, dtype, layout, device, pin_memory, memory_format);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ONES_LIKE, self, dtype, layout, device, pin_memory, memory_format);
}

Tensor wrap_pairwise_distance(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(x1, x2);
    return at::redispatch::pairwise_distance(x1, x2, p, eps, keepdim);
  }
  return MK_TORCHY(x1.dtype(), x1.device(), H_PAIRWISE_DISTANCE, x1, x2, p, eps, keepdim);
}

Tensor wrap_cdist(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(x1, x2);
    return at::redispatch::cdist(x1, x2, p, compute_mode);
  }
  return MK_TORCHY(x1.dtype(), x1.device(), H_CDIST, x1, x2, p, compute_mode);
}

Tensor wrap__euclidean_dist(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(x1, x2);
    return at::redispatch::_euclidean_dist(x1, x2);
  }
  return MK_TORCHY(x1.dtype(), x1.device(), H__EUCLIDEAN_DIST, x1, x2);
}

Tensor wrap__cdist_forward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(x1, x2);
    return at::redispatch::_cdist_forward(x1, x2, p, compute_mode);
  }
  return MK_TORCHY(x1.dtype(), x1.device(), H__CDIST_FORWARD, x1, x2, p, compute_mode);
}

Tensor wrap__cdist_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad, x1, x2, cdist);
    return at::redispatch::_cdist_backward(grad, x1, x2, p, cdist);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H__CDIST_BACKWARD, grad, x1, x2, p, cdist);
}

Tensor wrap_pdist(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::pdist(self, p);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_PDIST, self, p);
}

Tensor wrap__pdist_forward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_pdist_forward(self, p);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__PDIST_FORWARD, self, p);
}

Tensor wrap__pdist_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad, self, pdist);
    return at::redispatch::_pdist_backward(grad, self, p, pdist);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H__PDIST_BACKWARD, grad, self, p, pdist);
}

Tensor wrap_cosine_similarity(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(x1, x2);
    return at::redispatch::cosine_similarity(x1, x2, dim, eps);
  }
  return MK_TORCHY(x1.dtype(), x1.device(), H_COSINE_SIMILARITY, x1, x2, dim, eps);
}

Tensor wrap_permute(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::permute(self, dims);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_PERMUTE, self, dims);
}

Tensor wrap_movedim_intlist(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::movedim(self, source, destination);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MOVEDIM_INTLIST, self, source, destination);
}

Tensor wrap_movedim_int(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::movedim(self, source, destination);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MOVEDIM_INT, self, source, destination);
}

Tensor wrap_moveaxis_intlist(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::moveaxis(self, source, destination);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MOVEAXIS_INTLIST, self, source, destination);
}

Tensor wrap_moveaxis_int(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::moveaxis(self, source, destination);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MOVEAXIS_INT, self, source, destination);
}

Tensor wrap_numpy_T(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::numpy_T(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NUMPY_T, self);
}

Tensor wrap_pixel_shuffle(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::pixel_shuffle(self, upscale_factor);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_PIXEL_SHUFFLE, self, upscale_factor);
}

Tensor wrap_pixel_unshuffle(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::pixel_unshuffle(self, downscale_factor);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_PIXEL_UNSHUFFLE, self, downscale_factor);
}

Tensor wrap_channel_shuffle(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::channel_shuffle(self, groups);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CHANNEL_SHUFFLE, self, groups);
}

bool wrap_is_pinned(args...) {
  ensure_materialized(self);
  return at::redispatch::is_pinned(self);
}

Tensor wrap_pin_memory(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::pin_memory(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_PIN_MEMORY, self);
}

Tensor wrap_pinverse(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::pinverse(self, rcond);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_PINVERSE, self, rcond);
}

Tensor wrap_poisson_nll_loss(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, target);
    return at::redispatch::poisson_nll_loss(input, target, log_input, full, eps, reduction);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_POISSON_NLL_LOSS, input, target, log_input, full, eps, reduction);
}

Tensor wrap_rad2deg(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::rad2deg(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RAD2DEG, self);
}

Tensor wrap_rad2deg_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::rad2deg_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RAD2DEG_, self);
}

Tensor wrap_rad2deg_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::rad2deg(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_RAD2DEG_OUT, out, self);
}

Tensor wrap_deg2rad(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::deg2rad(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DEG2RAD, self);
}

Tensor wrap_deg2rad_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::deg2rad_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DEG2RAD_, self);
}

Tensor wrap_deg2rad_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::deg2rad(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_DEG2RAD_OUT, out, self);
}

Tensor wrap_scalar_tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::scalar_tensor(s, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_SCALAR_TENSOR, s, dtype, layout, device, pin_memory);
}

Tensor wrap_rand_names(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::rand(size, names, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_RAND_NAMES, size, names, dtype, layout, device, pin_memory);
}

Tensor wrap_rand_generator_with_names(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::rand(size, generator, names, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_RAND_GENERATOR_WITH_NAMES, size, generator, names, dtype, layout, device, pin_memory);
}

Tensor wrap_rand(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::rand(size, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_RAND, size, dtype, layout, device, pin_memory);
}

Tensor wrap_rand_generator(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::rand(size, generator, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_RAND_GENERATOR, size, generator, dtype, layout, device, pin_memory);
}

Tensor wrap_rand_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::rand(out, size);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_RAND_OUT, out, size);
}

Tensor wrap_rand_generator_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::rand(out, size, generator);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_RAND_GENERATOR_OUT, out, size, generator);
}

Tensor wrap_rand_like(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::rand_like(self, dtype, layout, device, pin_memory, memory_format);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RAND_LIKE, self, dtype, layout, device, pin_memory, memory_format);
}

Tensor wrap_randint(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::randint(high, size, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_RANDINT, high, size, dtype, layout, device, pin_memory);
}

Tensor wrap_randint_generator(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::randint(high, size, generator, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_RANDINT_GENERATOR, high, size, generator, dtype, layout, device, pin_memory);
}

Tensor wrap_randint_low(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::randint(low, high, size, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_RANDINT_LOW, low, high, size, dtype, layout, device, pin_memory);
}

Tensor wrap_randint_low_generator(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::randint(low, high, size, generator, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_RANDINT_LOW_GENERATOR, low, high, size, generator, dtype, layout, device, pin_memory);
}

Tensor wrap_randint_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::randint(out, high, size);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_RANDINT_OUT, out, high, size);
}

Tensor wrap_randint_generator_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::randint(out, high, size, generator);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_RANDINT_GENERATOR_OUT, out, high, size, generator);
}

Tensor wrap_randint_low_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::randint(out, low, high, size);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_RANDINT_LOW_OUT, out, low, high, size);
}

Tensor wrap_randint_low_generator_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::randint(out, low, high, size, generator);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_RANDINT_LOW_GENERATOR_OUT, out, low, high, size, generator);
}

Tensor wrap_randint_like(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::randint_like(self, high, dtype, layout, device, pin_memory, memory_format);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RANDINT_LIKE, self, high, dtype, layout, device, pin_memory, memory_format);
}

Tensor wrap_randint_like_low_dtype(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::randint_like(self, low, high, dtype, layout, device, pin_memory, memory_format);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RANDINT_LIKE_LOW_DTYPE, self, low, high, dtype, layout, device, pin_memory, memory_format);
}

Tensor wrap_randn(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::randn(size, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_RANDN, size, dtype, layout, device, pin_memory);
}

Tensor wrap_randn_generator(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::randn(size, generator, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_RANDN_GENERATOR, size, generator, dtype, layout, device, pin_memory);
}

Tensor wrap_randn_names(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::randn(size, names, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_RANDN_NAMES, size, names, dtype, layout, device, pin_memory);
}

Tensor wrap_randn_generator_with_names(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::randn(size, generator, names, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_RANDN_GENERATOR_WITH_NAMES, size, generator, names, dtype, layout, device, pin_memory);
}

Tensor wrap_randn_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::randn(out, size);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_RANDN_OUT, out, size);
}

Tensor wrap_randn_generator_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::randn(out, size, generator);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_RANDN_GENERATOR_OUT, out, size, generator);
}

Tensor wrap_randn_like(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::randn_like(self, dtype, layout, device, pin_memory, memory_format);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RANDN_LIKE, self, dtype, layout, device, pin_memory, memory_format);
}

Tensor wrap_randperm(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::randperm(n, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_RANDPERM, n, dtype, layout, device, pin_memory);
}

Tensor wrap_randperm_generator(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::randperm(n, generator, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_RANDPERM_GENERATOR, n, generator, dtype, layout, device, pin_memory);
}

Tensor wrap_randperm_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::randperm(out, n);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_RANDPERM_OUT, out, n);
}

Tensor wrap_randperm_generator_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::randperm(out, n, generator);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_RANDPERM_GENERATOR_OUT, out, n, generator);
}

Tensor wrap_range_step(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::range(start, end, step, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_RANGE_STEP, start, end, step, dtype, layout, device, pin_memory);
}

Tensor wrap_range(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::range(start, end, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_RANGE, start, end, dtype, layout, device, pin_memory);
}

Tensor wrap_range_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::range(out, start, end, step);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_RANGE_OUT, out, start, end, step);
}

Tensor wrap_ravel(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::ravel(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RAVEL, self);
}

Tensor wrap_reciprocal_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::reciprocal(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_RECIPROCAL_OUT, out, self);
}

Tensor wrap_neg(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::neg(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NEG, self);
}

Tensor wrap_neg_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::neg_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NEG_, self);
}

Tensor wrap_neg_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::neg(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NEG_OUT, out, self);
}

Tensor wrap_negative(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::negative(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NEGATIVE, self);
}

Tensor wrap_negative_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::negative_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NEGATIVE_, self);
}

Tensor wrap_negative_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::negative(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NEGATIVE_OUT, out, self);
}

Tensor wrap_repeat(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::repeat(self, repeats);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_REPEAT, self, repeats);
}

Tensor wrap_repeat_interleave_Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(repeats);
    return at::redispatch::repeat_interleave(repeats);
  }
  return MK_TORCHY(repeats.dtype(), repeats.device(), H_REPEAT_INTERLEAVE_TENSOR, repeats);
}

Tensor wrap_repeat_interleave_self_Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, repeats);
    return at::redispatch::repeat_interleave(self, repeats, dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_REPEAT_INTERLEAVE_SELF_TENSOR, self, repeats, dim);
}

Tensor wrap_repeat_interleave_self_int(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::repeat_interleave(self, repeats, dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_REPEAT_INTERLEAVE_SELF_INT, self, repeats, dim);
}

Tensor wrap_reshape(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::reshape(self, shape);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RESHAPE, self, shape);
}

Tensor wrap__mkldnn_reshape(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_mkldnn_reshape(self, shape);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__MKLDNN_RESHAPE, self, shape);
}

Tensor wrap_reshape_as(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::reshape_as(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RESHAPE_AS, self, other);
}

Tensor wrap_round(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::round(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ROUND, self);
}

Tensor wrap_round_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::round_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ROUND_, self);
}

Tensor wrap_round_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::round(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ROUND_OUT, out, self);
}

Tensor wrap_rrelu(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::rrelu(self, lower, upper, training, generator);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RRELU, self, lower, upper, training, generator);
}

Tensor wrap_rrelu_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::rrelu_(self, lower, upper, training, generator);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RRELU_, self, lower, upper, training, generator);
}

Tensor wrap_relu(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::relu(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RELU, self);
}

Tensor wrap_relu_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::relu_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RELU_, self);
}

Tensor wrap_relu6(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::relu6(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RELU6, self);
}

Tensor wrap_relu6_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::relu6_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RELU6_, self);
}

Tensor wrap_prelu(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, weight);
    return at::redispatch::prelu(self, weight);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_PRELU, self, weight);
}

Tensor wrap_prelu_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, weight);
    return at::redispatch::prelu_backward(grad_output, self, weight);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_PRELU_BACKWARD, grad_output, self, weight);
}

Tensor wrap_gelu(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::gelu(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_GELU, self);
}

Tensor wrap_gelu_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad, self);
    return at::redispatch::gelu_backward(grad, self);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H_GELU_BACKWARD, grad, self);
}

Tensor wrap_infinitely_differentiable_gelu_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad, self);
    return at::redispatch::infinitely_differentiable_gelu_backward(grad, self);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H_INFINITELY_DIFFERENTIABLE_GELU_BACKWARD, grad, self);
}

Tensor wrap_hardshrink(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::hardshrink(self, lambd);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_HARDSHRINK, self, lambd);
}

Tensor wrap_hardshrink_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_out, self);
    return at::redispatch::hardshrink_backward(grad_out, self, lambd);
  }
  return MK_TORCHY(grad_out.dtype(), grad_out.device(), H_HARDSHRINK_BACKWARD, grad_out, self, lambd);
}

Tensor wrap_rsqrt(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::rsqrt(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RSQRT, self);
}

Tensor wrap_rsqrt_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::rsqrt_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RSQRT_, self);
}

Tensor wrap_rsqrt_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::rsqrt(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_RSQRT_OUT, out, self);
}

Tensor wrap_select_Dimname(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::select(self, dim, index);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SELECT_DIMNAME, self, dim, index);
}

Tensor wrap_select_int(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::select(self, dim, index);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SELECT_INT, self, dim, index);
}

Tensor wrap_select_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad);
    return at::redispatch::select_backward(grad, input_sizes, dim, index);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H_SELECT_BACKWARD, grad, input_sizes, dim, index);
}

Tensor wrap_selu(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::selu(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SELU, self);
}

Tensor wrap_selu_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::selu_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SELU_, self);
}

Tensor wrap_celu(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::celu(self, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CELU, self, alpha);
}

Tensor wrap_celu_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::celu_(self, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CELU_, self, alpha);
}

Tensor wrap_silu(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::silu(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SILU, self);
}

Tensor wrap_silu_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::silu_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SILU_, self);
}

Tensor wrap_silu_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::silu(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SILU_OUT, out, self);
}

Tensor wrap_silu_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self);
    return at::redispatch::silu_backward(grad_output, self);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_SILU_BACKWARD, grad_output, self);
}

Tensor wrap_sigmoid(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::sigmoid(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SIGMOID, self);
}

Tensor wrap_sigmoid_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::sigmoid_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SIGMOID_, self);
}

Tensor wrap_sigmoid_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::sigmoid(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SIGMOID_OUT, out, self);
}

Tensor wrap_logit(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::logit(self, eps);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LOGIT, self, eps);
}

Tensor wrap_logit_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::logit_(self, eps);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LOGIT_, self, eps);
}

Tensor wrap_logit_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::logit(out, self, eps);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LOGIT_OUT, out, self, eps);
}

Tensor wrap_sin_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::sin(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SIN_OUT, out, self);
}

Tensor wrap_sinc_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::sinc(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SINC_OUT, out, self);
}

Tensor wrap_sinh_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::sinh(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SINH_OUT, out, self);
}

Tensor wrap_detach(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::detach(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DETACH, self);
}

Tensor wrap_detach_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::detach_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DETACH_, self);
}

int wrap_size_Dimname(args...) {
  ensure_materialized(self);
  return at::redispatch::size(self, dim);
}

Tensor wrap_slice_Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::slice(self, dim, start, end, step);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SLICE_TENSOR, self, dim, start, end, step);
}

Tensor wrap_slice_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad);
    return at::redispatch::slice_backward(grad, input_sizes, dim, start, end, step);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H_SLICE_BACKWARD, grad, input_sizes, dim, start, end, step);
}

Tensor wrap_slogdet(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::slogdet(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SLOGDET, self);
}

Tensor wrap_smm(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, mat2);
    return at::redispatch::smm(self, mat2);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SMM, self, mat2);
}

Tensor wrap_softmax_int(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::softmax(self, dim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SOFTMAX_INT, self, dim, dtype);
}

Tensor wrap_softmax_Dimname(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::softmax(self, dim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SOFTMAX_DIMNAME, self, dim, dtype);
}

Tensor wrap__softmax(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_softmax(self, dim, half_to_float);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__SOFTMAX, self, dim, half_to_float);
}

Tensor wrap__softmax_backward_data(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, output, self);
    return at::redispatch::_softmax_backward_data(grad_output, output, dim, self);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H__SOFTMAX_BACKWARD_DATA, grad_output, output, dim, self);
}

Tensor[] wrap_unsafe_split_Tensor(args...) {
  ensure_materialized(self);
  return at::redispatch::unsafe_split(self, split_size, dim);
}

Tensor[] wrap_split_Tensor(args...) {
  ensure_materialized(self);
  return at::redispatch::split(self, split_size, dim);
}

Tensor[] wrap_unsafe_split_with_sizes(args...) {
  ensure_materialized(self);
  return at::redispatch::unsafe_split_with_sizes(self, split_sizes, dim);
}

Tensor[] wrap_split_with_sizes(args...) {
  ensure_materialized(self);
  return at::redispatch::split_with_sizes(self, split_sizes, dim);
}

Tensor wrap_squeeze(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::squeeze(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SQUEEZE, self);
}

Tensor wrap_squeeze_dim(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::squeeze(self, dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SQUEEZE_DIM, self, dim);
}

Tensor wrap_squeeze_dimname(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::squeeze(self, dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SQUEEZE_DIMNAME, self, dim);
}

Tensor wrap_squeeze_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::squeeze_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SQUEEZE_, self);
}

Tensor wrap_squeeze__dim(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::squeeze_(self, dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SQUEEZE__DIM, self, dim);
}

Tensor wrap_squeeze__dimname(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::squeeze_(self, dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SQUEEZE__DIMNAME, self, dim);
}

Tensor wrap_sspaddmm(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, mat1, mat2);
    return at::redispatch::sspaddmm(self, mat1, mat2, beta, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SSPADDMM, self, mat1, mat2, beta, alpha);
}

Tensor wrap_sspaddmm_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, mat1, mat2);
    return at::redispatch::sspaddmm(out, self, mat1, mat2, beta, alpha);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SSPADDMM_OUT, out, self, mat1, mat2, beta, alpha);
}

Tensor wrap_stack(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::stack(tensors, dim);
  }
  return MK_TORCHY(None, None, H_STACK, tensors, dim);
}

Tensor wrap_stack_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::stack(out, tensors, dim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_STACK_OUT, out, tensors, dim);
}

Tensor wrap__stack(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::_stack(tensors, dim);
  }
  return MK_TORCHY(None, None, H__STACK, tensors, dim);
}

Tensor wrap__stack_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::_stack(out, tensors, dim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H__STACK_OUT, out, tensors, dim);
}

Tensor wrap_hstack(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::hstack(tensors);
  }
  return MK_TORCHY(None, None, H_HSTACK, tensors);
}

Tensor wrap_hstack_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::hstack(out, tensors);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_HSTACK_OUT, out, tensors);
}

Tensor wrap_vstack(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::vstack(tensors);
  }
  return MK_TORCHY(None, None, H_VSTACK, tensors);
}

Tensor wrap_vstack_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::vstack(out, tensors);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_VSTACK_OUT, out, tensors);
}

Tensor wrap_dstack(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::dstack(tensors);
  }
  return MK_TORCHY(None, None, H_DSTACK, tensors);
}

Tensor wrap_dstack_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::dstack(out, tensors);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_DSTACK_OUT, out, tensors);
}

Tensor wrap_stft(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::stft(self, n_fft, hop_length, win_length, window, normalized, onesided, return_complex);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_STFT, self, n_fft, hop_length, win_length, window, normalized, onesided, return_complex);
}

Tensor wrap_istft(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::istft(self, n_fft, hop_length, win_length, window, center, normalized, onesided, length, return_complex);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ISTFT, self, n_fft, hop_length, win_length, window, center, normalized, onesided, length, return_complex);
}

int wrap_stride_Dimname(args...) {
  ensure_materialized(self);
  return at::redispatch::stride(self, dim);
}

Tensor wrap_sum(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::sum(self, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SUM, self, dtype);
}

Tensor wrap_sum_dim_IntList(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::sum(self, dim, keepdim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SUM_DIM_INTLIST, self, dim, keepdim, dtype);
}

Tensor wrap_sum_dim_DimnameList(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::sum(self, dim, keepdim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SUM_DIM_DIMNAMELIST, self, dim, keepdim, dtype);
}

Tensor wrap_sum_IntList_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::sum(out, self, dim, keepdim, dtype);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SUM_INTLIST_OUT, out, self, dim, keepdim, dtype);
}

Tensor wrap_sum_DimnameList_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::sum(out, self, dim, keepdim, dtype);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SUM_DIMNAMELIST_OUT, out, self, dim, keepdim, dtype);
}

Tensor wrap_nansum(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::nansum(self, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NANSUM, self, dtype);
}

Tensor wrap_nansum_dim_IntList(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::nansum(self, dim, keepdim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NANSUM_DIM_INTLIST, self, dim, keepdim, dtype);
}

Tensor wrap_nansum_IntList_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::nansum(out, self, dim, keepdim, dtype);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NANSUM_INTLIST_OUT, out, self, dim, keepdim, dtype);
}

Tensor wrap_sum_to_size(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::sum_to_size(self, size);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SUM_TO_SIZE, self, size);
}

Tensor wrap_sqrt(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::sqrt(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SQRT, self);
}

Tensor wrap_sqrt_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::sqrt(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SQRT_OUT, out, self);
}

Tensor wrap_square(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::square(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SQUARE, self);
}

Tensor wrap_square_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::square_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SQUARE_, self);
}

Tensor wrap_square_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::square(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SQUARE_OUT, out, self);
}

Tensor wrap_std(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::std(self, unbiased);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_STD, self, unbiased);
}

Tensor wrap_std_dim(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::std(self, dim, unbiased, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_STD_DIM, self, dim, unbiased, keepdim);
}

Tensor wrap_std_mean(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::std_mean(self, unbiased);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_STD_MEAN, self, unbiased);
}

Tensor wrap_std_mean_dim(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::std_mean(self, dim, unbiased, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_STD_MEAN_DIM, self, dim, unbiased, keepdim);
}

Tensor wrap_std_mean_names_dim(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::std_mean(self, dim, unbiased, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_STD_MEAN_NAMES_DIM, self, dim, unbiased, keepdim);
}

Tensor wrap_std_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::std(out, self, dim, unbiased, keepdim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_STD_OUT, out, self, dim, unbiased, keepdim);
}

Tensor wrap_std_names_dim(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::std(self, dim, unbiased, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_STD_NAMES_DIM, self, dim, unbiased, keepdim);
}

Tensor wrap_std_names_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::std(out, self, dim, unbiased, keepdim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_STD_NAMES_OUT, out, self, dim, unbiased, keepdim);
}

Tensor wrap_prod(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::prod(self, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_PROD, self, dtype);
}

Tensor wrap_prod_dim_int(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::prod(self, dim, keepdim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_PROD_DIM_INT, self, dim, keepdim, dtype);
}

Tensor wrap_prod_int_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::prod(out, self, dim, keepdim, dtype);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_PROD_INT_OUT, out, self, dim, keepdim, dtype);
}

Tensor wrap_prod_dim_Dimname(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::prod(self, dim, keepdim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_PROD_DIM_DIMNAME, self, dim, keepdim, dtype);
}

Tensor wrap_prod_Dimname_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::prod(out, self, dim, keepdim, dtype);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_PROD_DIMNAME_OUT, out, self, dim, keepdim, dtype);
}

Tensor wrap_t(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::t(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_T, self);
}

Tensor wrap_t_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::t_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_T_, self);
}

Tensor wrap_tan_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::tan(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_TAN_OUT, out, self);
}

Tensor wrap_tanh(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::tanh(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TANH, self);
}

Tensor wrap_tanh_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::tanh_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TANH_, self);
}

Tensor wrap_tanh_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::tanh(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_TANH_OUT, out, self);
}

Tensor wrap_tensordot(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::tensordot(self, other, dims_self, dims_other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TENSORDOT, self, other, dims_self, dims_other);
}

Tensor wrap_tensordot_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::tensordot(out, self, other, dims_self, dims_other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_TENSORDOT_OUT, out, self, other, dims_self, dims_other);
}

Tensor wrap_threshold(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::threshold(self, threshold, value);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_THRESHOLD, self, threshold, value);
}

Tensor wrap_threshold_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::threshold_(self, threshold, value);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_THRESHOLD_, self, threshold, value);
}

Tensor wrap_threshold_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::threshold(out, self, threshold, value);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_THRESHOLD_OUT, out, self, threshold, value);
}

Tensor wrap_threshold_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self);
    return at::redispatch::threshold_backward(grad_output, self, threshold);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_THRESHOLD_BACKWARD, grad_output, self, threshold);
}

Tensor wrap_tile(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::tile(self, dims);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TILE, self, dims);
}

Tensor wrap_transpose_int(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::transpose(self, dim0, dim1);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TRANSPOSE_INT, self, dim0, dim1);
}

Tensor wrap_transpose_Dimname(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::transpose(self, dim0, dim1);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TRANSPOSE_DIMNAME, self, dim0, dim1);
}

Tensor wrap__mkldnn_transpose(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_mkldnn_transpose(self, dim0, dim1);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__MKLDNN_TRANSPOSE, self, dim0, dim1);
}

Tensor wrap_transpose_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::transpose_(self, dim0, dim1);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TRANSPOSE_, self, dim0, dim1);
}

Tensor wrap__mkldnn_transpose_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_mkldnn_transpose_(self, dim0, dim1);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__MKLDNN_TRANSPOSE_, self, dim0, dim1);
}

Tensor wrap_one_hot(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::one_hot(self, num_classes);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ONE_HOT, self, num_classes);
}

Tensor wrap_flip(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::flip(self, dims);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FLIP, self, dims);
}

Tensor wrap_fliplr(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::fliplr(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FLIPLR, self);
}

Tensor wrap_flipud(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::flipud(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FLIPUD, self);
}

Tensor wrap_roll(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::roll(self, shifts, dims);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ROLL, self, shifts, dims);
}

Tensor wrap_rot90(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::rot90(self, k, dims);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ROT90, self, k, dims);
}

Tensor wrap_trapz_x(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(y, x);
    return at::redispatch::trapz(y, x, dim);
  }
  return MK_TORCHY(y.dtype(), y.device(), H_TRAPZ_X, y, x, dim);
}

Tensor wrap_trapz_dx(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(y);
    return at::redispatch::trapz(y, dx, dim);
  }
  return MK_TORCHY(y.dtype(), y.device(), H_TRAPZ_DX, y, dx, dim);
}

Tensor wrap__trilinear(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(i1, i2, i3);
    return at::redispatch::_trilinear(i1, i2, i3, expand1, expand2, expand3, sumdim, unroll_dim);
  }
  return MK_TORCHY(i1.dtype(), i1.device(), H__TRILINEAR, i1, i2, i3, expand1, expand2, expand3, sumdim, unroll_dim);
}

Tensor wrap_triplet_margin_loss(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(anchor, positive, negative);
    return at::redispatch::triplet_margin_loss(anchor, positive, negative, margin, p, eps, swap, reduction);
  }
  return MK_TORCHY(anchor.dtype(), anchor.device(), H_TRIPLET_MARGIN_LOSS, anchor, positive, negative, margin, p, eps, swap, reduction);
}

Tensor wrap_trunc(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::trunc(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TRUNC, self);
}

Tensor wrap_trunc_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::trunc_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TRUNC_, self);
}

Tensor wrap_trunc_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::trunc(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_TRUNC_OUT, out, self);
}

Tensor wrap_fix(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::fix(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FIX, self);
}

Tensor wrap_fix_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::fix_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FIX_, self);
}

Tensor wrap_fix_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::fix(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FIX_OUT, out, self);
}

Tensor wrap_type_as(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::type_as(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TYPE_AS, self, other);
}

bool wrap__has_compatible_shallow_copy_type(args...) {
  ensure_materialized(self, from);
  return at::redispatch::_has_compatible_shallow_copy_type(self, from);
}

Tensor wrap__unique(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_unique(self, sorted, return_inverse);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__UNIQUE, self, sorted, return_inverse);
}

Tensor wrap_unique_dim(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::unique_dim(self, dim, sorted, return_inverse, return_counts);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_UNIQUE_DIM, self, dim, sorted, return_inverse, return_counts);
}

Tensor wrap_unique_consecutive(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::unique_consecutive(self, return_inverse, return_counts, dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_UNIQUE_CONSECUTIVE, self, return_inverse, return_counts, dim);
}

Tensor wrap_unique_dim_consecutive(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::unique_dim_consecutive(self, dim, return_inverse, return_counts);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_UNIQUE_DIM_CONSECUTIVE, self, dim, return_inverse, return_counts);
}

Tensor wrap__unique2(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_unique2(self, sorted, return_inverse, return_counts);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__UNIQUE2, self, sorted, return_inverse, return_counts);
}

Tensor wrap__unsafe_view(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_unsafe_view(self, size);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__UNSAFE_VIEW, self, size);
}

Tensor wrap_unsqueeze(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::unsqueeze(self, dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_UNSQUEEZE, self, dim);
}

Tensor wrap_unsqueeze_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::unsqueeze_(self, dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_UNSQUEEZE_, self, dim);
}

Tensor wrap_vander(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(x);
    return at::redispatch::vander(x, N, increasing);
  }
  return MK_TORCHY(x.dtype(), x.device(), H_VANDER, x, N, increasing);
}

Tensor wrap_var(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::var(self, unbiased);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_VAR, self, unbiased);
}

Tensor wrap_var_dim(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::var(self, dim, unbiased, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_VAR_DIM, self, dim, unbiased, keepdim);
}

Tensor wrap_var_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::var(out, self, dim, unbiased, keepdim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_VAR_OUT, out, self, dim, unbiased, keepdim);
}

Tensor wrap_var_names_dim(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::var(self, dim, unbiased, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_VAR_NAMES_DIM, self, dim, unbiased, keepdim);
}

Tensor wrap_var_names_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::var(out, self, dim, unbiased, keepdim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_VAR_NAMES_OUT, out, self, dim, unbiased, keepdim);
}

Tensor wrap_var_mean(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::var_mean(self, unbiased);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_VAR_MEAN, self, unbiased);
}

Tensor wrap_var_mean_dim(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::var_mean(self, dim, unbiased, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_VAR_MEAN_DIM, self, dim, unbiased, keepdim);
}

Tensor wrap_var_mean_names_dim(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::var_mean(self, dim, unbiased, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_VAR_MEAN_NAMES_DIM, self, dim, unbiased, keepdim);
}

Tensor wrap_view_as(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::view_as(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_VIEW_AS, self, other);
}

Tensor wrap_where_self(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(condition, self, other);
    return at::redispatch::where(condition, self, other);
  }
  return MK_TORCHY(condition.dtype(), condition.device(), H_WHERE_SELF, condition, self, other);
}

Tensor wrap_where_ScalarSelf(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(condition, other);
    return at::redispatch::where(condition, self, other);
  }
  return MK_TORCHY(condition.dtype(), condition.device(), H_WHERE_SCALARSELF, condition, self, other);
}

Tensor wrap_where_ScalarOther(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(condition, self);
    return at::redispatch::where(condition, self, other);
  }
  return MK_TORCHY(condition.dtype(), condition.device(), H_WHERE_SCALAROTHER, condition, self, other);
}

Tensor wrap_where_Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(condition);
    return at::redispatch::where(condition, self, other);
  }
  return MK_TORCHY(condition.dtype(), condition.device(), H_WHERE_SCALAR, condition, self, other);
}

Tensor[] wrap_where(args...) {
  ensure_materialized(condition);
  return at::redispatch::where(condition);
}

Tensor wrap__s_where(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(condition, self, other);
    return at::redispatch::_s_where(condition, self, other);
  }
  return MK_TORCHY(condition.dtype(), condition.device(), H__S_WHERE, condition, self, other);
}

Tensor wrap_norm_except_dim(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(v);
    return at::redispatch::norm_except_dim(v, pow, dim);
  }
  return MK_TORCHY(v.dtype(), v.device(), H_NORM_EXCEPT_DIM, v, pow, dim);
}

Tensor wrap__weight_norm(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(v, g);
    return at::redispatch::_weight_norm(v, g, dim);
  }
  return MK_TORCHY(v.dtype(), v.device(), H__WEIGHT_NORM, v, g, dim);
}

Tensor wrap__weight_norm_cuda_interface(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(v, g);
    return at::redispatch::_weight_norm_cuda_interface(v, g, dim);
  }
  return MK_TORCHY(v.dtype(), v.device(), H__WEIGHT_NORM_CUDA_INTERFACE, v, g, dim);
}

Tensor wrap__weight_norm_cuda_interface_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_w, saved_v, saved_g, saved_norms);
    return at::redispatch::_weight_norm_cuda_interface_backward(grad_w, saved_v, saved_g, saved_norms, dim);
  }
  return MK_TORCHY(grad_w.dtype(), grad_w.device(), H__WEIGHT_NORM_CUDA_INTERFACE_BACKWARD, grad_w, saved_v, saved_g, saved_norms, dim);
}

Tensor wrap__weight_norm_differentiable_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_w, saved_v, saved_g, saved_norms);
    return at::redispatch::_weight_norm_differentiable_backward(grad_w, saved_v, saved_g, saved_norms, dim);
  }
  return MK_TORCHY(grad_w.dtype(), grad_w.device(), H__WEIGHT_NORM_DIFFERENTIABLE_BACKWARD, grad_w, saved_v, saved_g, saved_norms, dim);
}

Tensor wrap_zeros_names(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::zeros(size, names, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_ZEROS_NAMES, size, names, dtype, layout, device, pin_memory);
}

Tensor wrap_zeros(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::zeros(size, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_ZEROS, size, dtype, layout, device, pin_memory);
}

Tensor wrap_zeros_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::zeros(out, size);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ZEROS_OUT, out, size);
}

Tensor wrap_zeros_like(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::zeros_like(self, dtype, layout, device, pin_memory, memory_format);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ZEROS_LIKE, self, dtype, layout, device, pin_memory, memory_format);
}

Tensor wrap__standard_gamma_grad(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, output);
    return at::redispatch::_standard_gamma_grad(self, output);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__STANDARD_GAMMA_GRAD, self, output);
}

Tensor wrap__standard_gamma(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_standard_gamma(self, generator);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__STANDARD_GAMMA, self, generator);
}

Tensor wrap__dirichlet_grad(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(x, alpha, total);
    return at::redispatch::_dirichlet_grad(x, alpha, total);
  }
  return MK_TORCHY(x.dtype(), x.device(), H__DIRICHLET_GRAD, x, alpha, total);
}

Tensor wrap__sample_dirichlet(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_sample_dirichlet(self, generator);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__SAMPLE_DIRICHLET, self, generator);
}

Tensor wrap_poisson(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::poisson(self, generator);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_POISSON, self, generator);
}

Tensor wrap_binomial(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(count, prob);
    return at::redispatch::binomial(count, prob, generator);
  }
  return MK_TORCHY(count.dtype(), count.device(), H_BINOMIAL, count, prob, generator);
}

Tensor wrap_native_norm(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::native_norm(self, p);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NATIVE_NORM, self, p);
}

Tensor wrap_native_norm_ScalarOpt_dim_dtype(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::native_norm(self, p, dim, keepdim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NATIVE_NORM_SCALAROPT_DIM_DTYPE, self, p, dim, keepdim, dtype);
}

Tensor wrap__sparse_sum(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_sparse_sum(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__SPARSE_SUM, self);
}

Tensor wrap__sparse_sum_dtype(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_sparse_sum(self, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__SPARSE_SUM_DTYPE, self, dtype);
}

Tensor wrap__sparse_sum_dim(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_sparse_sum(self, dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__SPARSE_SUM_DIM, self, dim);
}

Tensor wrap__sparse_sum_dim_dtype(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_sparse_sum(self, dim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__SPARSE_SUM_DIM_DTYPE, self, dim, dtype);
}

Tensor wrap__sparse_sum_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad, self);
    return at::redispatch::_sparse_sum_backward(grad, self, dim);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H__SPARSE_SUM_BACKWARD, grad, self, dim);
}

Tensor wrap__sparse_softmax_int(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_sparse_softmax(self, dim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__SPARSE_SOFTMAX_INT, self, dim, dtype);
}

Tensor wrap__sparse_softmax_Dimname(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_sparse_softmax(self, dim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__SPARSE_SOFTMAX_DIMNAME, self, dim, dtype);
}

Tensor wrap__sparse_softmax(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_sparse_softmax(self, dim, half_to_float);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__SPARSE_SOFTMAX, self, dim, half_to_float);
}

Tensor wrap__sparse_softmax_backward_data(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, output, self);
    return at::redispatch::_sparse_softmax_backward_data(grad_output, output, dim, self);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H__SPARSE_SOFTMAX_BACKWARD_DATA, grad_output, output, dim, self);
}

Tensor wrap__sparse_log_softmax_int(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_sparse_log_softmax(self, dim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__SPARSE_LOG_SOFTMAX_INT, self, dim, dtype);
}

Tensor wrap__sparse_log_softmax_Dimname(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_sparse_log_softmax(self, dim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__SPARSE_LOG_SOFTMAX_DIMNAME, self, dim, dtype);
}

Tensor wrap__sparse_log_softmax(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_sparse_log_softmax(self, dim, half_to_float);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__SPARSE_LOG_SOFTMAX, self, dim, half_to_float);
}

Tensor wrap__sparse_log_softmax_backward_data(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, output, self);
    return at::redispatch::_sparse_log_softmax_backward_data(grad_output, output, dim, self);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H__SPARSE_LOG_SOFTMAX_BACKWARD_DATA, grad_output, output, dim, self);
}

Tensor wrap_norm_ScalarOpt_dtype(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::norm(self, p, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NORM_SCALAROPT_DTYPE, self, p, dtype);
}

Tensor wrap_norm_Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::norm(self, p);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NORM_SCALAR, self, p);
}

Tensor wrap_norm_ScalarOpt_dim_dtype(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::norm(self, p, dim, keepdim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NORM_SCALAROPT_DIM_DTYPE, self, p, dim, keepdim, dtype);
}

Tensor wrap_norm_ScalarOpt_dim(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::norm(self, p, dim, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NORM_SCALAROPT_DIM, self, p, dim, keepdim);
}

Tensor wrap_norm_dtype_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::norm(out, self, p, dim, keepdim, dtype);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NORM_DTYPE_OUT, out, self, p, dim, keepdim, dtype);
}

Tensor wrap_norm_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::norm(out, self, p, dim, keepdim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NORM_OUT, out, self, p, dim, keepdim);
}

Tensor wrap_norm_names_ScalarOpt_dim_dtype(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::norm(self, p, dim, keepdim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NORM_NAMES_SCALAROPT_DIM_DTYPE, self, p, dim, keepdim, dtype);
}

Tensor wrap_norm_names_ScalarOpt_dim(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::norm(self, p, dim, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NORM_NAMES_SCALAROPT_DIM, self, p, dim, keepdim);
}

Tensor wrap_norm_names_dtype_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::norm(out, self, p, dim, keepdim, dtype);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NORM_NAMES_DTYPE_OUT, out, self, p, dim, keepdim, dtype);
}

Tensor wrap_norm_names_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::norm(out, self, p, dim, keepdim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NORM_NAMES_OUT, out, self, p, dim, keepdim);
}

Tensor wrap_frexp_Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::frexp(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FREXP_TENSOR, self);
}

Tensor wrap_frexp_Tensor_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(mantissa, exponent, self);
    return at::redispatch::frexp(mantissa, exponent, self);
  }
  return MK_TORCHY(mantissa.dtype(), mantissa.device(), H_FREXP_TENSOR_OUT, mantissa, exponent, self);
}

Tensor wrap_frobenius_norm(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::frobenius_norm(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FROBENIUS_NORM, self);
}

Tensor wrap_frobenius_norm_dim(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::frobenius_norm(self, dim, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FROBENIUS_NORM_DIM, self, dim, keepdim);
}

Tensor wrap_frobenius_norm_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::frobenius_norm(out, self, dim, keepdim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FROBENIUS_NORM_OUT, out, self, dim, keepdim);
}

Tensor wrap_nuclear_norm(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::nuclear_norm(self, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NUCLEAR_NORM, self, keepdim);
}

Tensor wrap_nuclear_norm_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::nuclear_norm(out, self, keepdim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NUCLEAR_NORM_OUT, out, self, keepdim);
}

Tensor wrap_nuclear_norm_dim(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::nuclear_norm(self, dim, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NUCLEAR_NORM_DIM, self, dim, keepdim);
}

Tensor wrap_nuclear_norm_dim_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::nuclear_norm(out, self, dim, keepdim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NUCLEAR_NORM_DIM_OUT, out, self, dim, keepdim);
}

Tensor wrap_clone(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::clone(self, memory_format);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CLONE, self, memory_format);
}

Tensor wrap_resize_as_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, the_template);
    return at::redispatch::resize_as_(self, the_template, memory_format);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RESIZE_AS_, self, the_template, memory_format);
}

Tensor wrap_resize_as_sparse_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, the_template);
    return at::redispatch::resize_as_sparse_(self, the_template);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RESIZE_AS_SPARSE_, self, the_template);
}

Tensor wrap_zero_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::zero_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ZERO_, self);
}

Tensor wrap_sub_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::sub(out, self, other, alpha);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SUB_OUT, out, self, other, alpha);
}

Tensor wrap_sub_Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::sub(self, other, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SUB_TENSOR, self, other, alpha);
}

Tensor wrap_sub__Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::sub_(self, other, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SUB__TENSOR, self, other, alpha);
}

Tensor wrap_sub_Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::sub(self, other, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SUB_SCALAR, self, other, alpha);
}

Tensor wrap_sub__Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::sub_(self, other, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SUB__SCALAR, self, other, alpha);
}

Tensor wrap_subtract_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::subtract(out, self, other, alpha);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SUBTRACT_OUT, out, self, other, alpha);
}

Tensor wrap_subtract_Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::subtract(self, other, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SUBTRACT_TENSOR, self, other, alpha);
}

Tensor wrap_subtract__Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::subtract_(self, other, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SUBTRACT__TENSOR, self, other, alpha);
}

Tensor wrap_subtract_Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::subtract(self, other, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SUBTRACT_SCALAR, self, other, alpha);
}

Tensor wrap_subtract__Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::subtract_(self, other, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SUBTRACT__SCALAR, self, other, alpha);
}

Tensor wrap_rsub_Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::rsub(self, other, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RSUB_TENSOR, self, other, alpha);
}

Tensor wrap_heaviside_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, values);
    return at::redispatch::heaviside(out, self, values);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_HEAVISIDE_OUT, out, self, values);
}

Tensor wrap_heaviside(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, values);
    return at::redispatch::heaviside(self, values);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_HEAVISIDE, self, values);
}

Tensor wrap_heaviside_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, values);
    return at::redispatch::heaviside_(self, values);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_HEAVISIDE_, self, values);
}

Tensor wrap_rsub_Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::rsub(self, other, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RSUB_SCALAR, self, other, alpha);
}

Tensor wrap__sparse_addmm(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, sparse, dense);
    return at::redispatch::_sparse_addmm(self, sparse, dense, beta, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__SPARSE_ADDMM, self, sparse, dense, beta, alpha);
}

Tensor wrap_addmm_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, mat1, mat2);
    return at::redispatch::addmm(out, self, mat1, mat2, beta, alpha);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ADDMM_OUT, out, self, mat1, mat2, beta, alpha);
}

Tensor wrap_addmm(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, mat1, mat2);
    return at::redispatch::addmm(self, mat1, mat2, beta, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ADDMM, self, mat1, mat2, beta, alpha);
}

Tensor wrap_addmm_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, mat1, mat2);
    return at::redispatch::addmm_(self, mat1, mat2, beta, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ADDMM_, self, mat1, mat2, beta, alpha);
}

Tensor wrap_sparse_csr_tensor_crow_col_value_size(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(crow_indices, col_indices, values);
    return at::redispatch::sparse_csr_tensor(crow_indices, col_indices, values, size, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(crow_indices.dtype(), crow_indices.device(), H_SPARSE_CSR_TENSOR_CROW_COL_VALUE_SIZE, crow_indices, col_indices, values, size, dtype, layout, device, pin_memory);
}

Tensor wrap_sparse_csr_tensor_crow_col_value(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(crow_indices, col_indices, values);
    return at::redispatch::sparse_csr_tensor(crow_indices, col_indices, values, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(crow_indices.dtype(), crow_indices.device(), H_SPARSE_CSR_TENSOR_CROW_COL_VALUE, crow_indices, col_indices, values, dtype, layout, device, pin_memory);
}

Tensor wrap_sparse_coo_tensor_size(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::sparse_coo_tensor(size, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_SPARSE_COO_TENSOR_SIZE, size, dtype, layout, device, pin_memory);
}

Tensor wrap_sparse_coo_tensor_indices(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(indices, values);
    return at::redispatch::sparse_coo_tensor(indices, values, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(indices.dtype(), indices.device(), H_SPARSE_COO_TENSOR_INDICES, indices, values, dtype, layout, device, pin_memory);
}

Tensor wrap_sparse_coo_tensor_indices_size(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(indices, values);
    return at::redispatch::sparse_coo_tensor(indices, values, size, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(indices.dtype(), indices.device(), H_SPARSE_COO_TENSOR_INDICES_SIZE, indices, values, size, dtype, layout, device, pin_memory);
}

Tensor wrap__sparse_coo_tensor_unsafe(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(indices, values);
    return at::redispatch::_sparse_coo_tensor_unsafe(indices, values, size, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(indices.dtype(), indices.device(), H__SPARSE_COO_TENSOR_UNSAFE, indices, values, size, dtype, layout, device, pin_memory);
}

void wrap__validate_sparse_coo_tensor_args(args...) {
  ensure_materialized(indices, values);
  return at::redispatch::_validate_sparse_coo_tensor_args(indices, values, size);
}

Tensor wrap__sparse_coo_tensor_with_dims(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::_sparse_coo_tensor_with_dims(sparse_dim, dense_dim, size, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H__SPARSE_COO_TENSOR_WITH_DIMS, sparse_dim, dense_dim, size, dtype, layout, device, pin_memory);
}

Tensor wrap__sparse_coo_tensor_with_dims_and_tensors(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(indices, values);
    return at::redispatch::_sparse_coo_tensor_with_dims_and_tensors(sparse_dim, dense_dim, size, indices, values, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(indices.dtype(), indices.device(), H__SPARSE_COO_TENSOR_WITH_DIMS_AND_TENSORS, sparse_dim, dense_dim, size, indices, values, dtype, layout, device, pin_memory);
}

Tensor wrap_sparse_resize_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::sparse_resize_(self, size, sparse_dim, dense_dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SPARSE_RESIZE_, self, size, sparse_dim, dense_dim);
}

Tensor wrap_sparse_resize_and_clear_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::sparse_resize_and_clear_(self, size, sparse_dim, dense_dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SPARSE_RESIZE_AND_CLEAR_, self, size, sparse_dim, dense_dim);
}

Tensor wrap_sparse_mask(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, mask);
    return at::redispatch::sparse_mask(self, mask);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SPARSE_MASK, self, mask);
}

Tensor wrap_to_dense(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::to_dense(self, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TO_DENSE, self, dtype);
}

Tensor wrap_to_dense_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad, input);
    return at::redispatch::to_dense_backward(grad, input);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H_TO_DENSE_BACKWARD, grad, input);
}

int wrap_sparse_dim(args...) {
  ensure_materialized(self);
  return at::redispatch::sparse_dim(self);
}

int wrap__dimI(args...) {
  ensure_materialized(self);
  return at::redispatch::_dimI(self);
}

int wrap_dense_dim(args...) {
  ensure_materialized(self);
  return at::redispatch::dense_dim(self);
}

int wrap__dimV(args...) {
  ensure_materialized(self);
  return at::redispatch::_dimV(self);
}

int wrap__nnz(args...) {
  ensure_materialized(self);
  return at::redispatch::_nnz(self);
}

Tensor wrap_coalesce(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::coalesce(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_COALESCE, self);
}

Tensor wrap__coalesce(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_coalesce(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__COALESCE, self);
}

bool wrap_is_coalesced(args...) {
  ensure_materialized(self);
  return at::redispatch::is_coalesced(self);
}

Tensor wrap__indices(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_indices(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__INDICES, self);
}

Tensor wrap__values(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_values(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__VALUES, self);
}

Tensor wrap__coalesced_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_coalesced_(self, coalesced);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__COALESCED_, self, coalesced);
}

Tensor wrap_indices(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::indices(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INDICES, self);
}

Tensor wrap_values(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::values(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_VALUES, self);
}

Tensor wrap_crow_indices(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::crow_indices(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CROW_INDICES, self);
}

Tensor wrap_col_indices(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::col_indices(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_COL_INDICES, self);
}

Tensor wrap_hspmm_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, mat1, mat2);
    return at::redispatch::hspmm(out, mat1, mat2);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_HSPMM_OUT, out, mat1, mat2);
}

Tensor wrap_hspmm(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(mat1, mat2);
    return at::redispatch::hspmm(mat1, mat2);
  }
  return MK_TORCHY(mat1.dtype(), mat1.device(), H_HSPMM, mat1, mat2);
}

Tensor wrap_copy_sparse_to_sparse_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, src);
    return at::redispatch::copy_sparse_to_sparse_(self, src, non_blocking);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_COPY_SPARSE_TO_SPARSE_, self, src, non_blocking);
}

Tensor[] wrap_unbind_int(args...) {
  ensure_materialized(self);
  return at::redispatch::unbind(self, dim);
}

Tensor[] wrap_unbind_Dimname(args...) {
  ensure_materialized(self);
  return at::redispatch::unbind(self, dim);
}

Tensor wrap_to_sparse_sparse_dim(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::to_sparse(self, sparse_dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TO_SPARSE_SPARSE_DIM, self, sparse_dim);
}

Tensor wrap_to_sparse(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::to_sparse(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TO_SPARSE, self);
}

Tensor wrap_to_mkldnn(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::to_mkldnn(self, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TO_MKLDNN, self, dtype);
}

Tensor wrap_mkldnn_reorder_conv2d_weight(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::mkldnn_reorder_conv2d_weight(self, padding, stride, dilation, groups);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MKLDNN_REORDER_CONV2D_WEIGHT, self, padding, stride, dilation, groups);
}

Tensor wrap_mkldnn_reorder_conv3d_weight(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::mkldnn_reorder_conv3d_weight(self, padding, stride, dilation, groups);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MKLDNN_REORDER_CONV3D_WEIGHT, self, padding, stride, dilation, groups);
}

Tensor wrap_to_mkldnn_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad, input);
    return at::redispatch::to_mkldnn_backward(grad, input);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H_TO_MKLDNN_BACKWARD, grad, input);
}

Tensor wrap_quantize_per_tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::quantize_per_tensor(self, scale, zero_point, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_QUANTIZE_PER_TENSOR, self, scale, zero_point, dtype);
}

Tensor[] wrap_quantize_per_tensor_tensors(args...) {
  ensure_materialized(scales, zero_points);
  return at::redispatch::quantize_per_tensor(tensors, scales, zero_points, dtype);
}

Tensor wrap_quantize_per_channel(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, scales, zero_points);
    return at::redispatch::quantize_per_channel(self, scales, zero_points, axis, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_QUANTIZE_PER_CHANNEL, self, scales, zero_points, axis, dtype);
}

Tensor wrap_dequantize_self(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::dequantize(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DEQUANTIZE_SELF, self);
}

Tensor[] wrap_dequantize_tensors(args...) {
  ensure_materialized();
  return at::redispatch::dequantize(tensors);
}

float wrap_q_scale(args...) {
  ensure_materialized(self);
  return at::redispatch::q_scale(self);
}

int wrap_q_zero_point(args...) {
  ensure_materialized(self);
  return at::redispatch::q_zero_point(self);
}

Tensor wrap_q_per_channel_scales(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::q_per_channel_scales(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_Q_PER_CHANNEL_SCALES, self);
}

Tensor wrap_q_per_channel_zero_points(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::q_per_channel_zero_points(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_Q_PER_CHANNEL_ZERO_POINTS, self);
}

int wrap_q_per_channel_axis(args...) {
  ensure_materialized(self);
  return at::redispatch::q_per_channel_axis(self);
}

Tensor wrap_int_repr(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::int_repr(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INT_REPR, self);
}

Tensor wrap__make_per_tensor_quantized_tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_make_per_tensor_quantized_tensor(self, scale, zero_point);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__MAKE_PER_TENSOR_QUANTIZED_TENSOR, self, scale, zero_point);
}

Tensor wrap__make_per_channel_quantized_tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, scale, zero_point);
    return at::redispatch::_make_per_channel_quantized_tensor(self, scale, zero_point, axis);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__MAKE_PER_CHANNEL_QUANTIZED_TENSOR, self, scale, zero_point, axis);
}

QScheme wrap_qscheme(args...) {
  ensure_materialized(self);
  return at::redispatch::qscheme(self);
}

Tensor wrap_fake_quantize_per_tensor_affine(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::fake_quantize_per_tensor_affine(self, scale, zero_point, quant_min, quant_max);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FAKE_QUANTIZE_PER_TENSOR_AFFINE, self, scale, zero_point, quant_min, quant_max);
}

Tensor wrap_fake_quantize_per_tensor_affine_cachemask(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::fake_quantize_per_tensor_affine_cachemask(self, scale, zero_point, quant_min, quant_max);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FAKE_QUANTIZE_PER_TENSOR_AFFINE_CACHEMASK, self, scale, zero_point, quant_min, quant_max);
}

Tensor wrap_fake_quantize_per_tensor_affine_cachemask_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad, mask);
    return at::redispatch::fake_quantize_per_tensor_affine_cachemask_backward(grad, mask);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H_FAKE_QUANTIZE_PER_TENSOR_AFFINE_CACHEMASK_BACKWARD, grad, mask);
}

Tensor wrap__fake_quantize_learnable_per_tensor_affine(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, scale, zero_point);
    return at::redispatch::_fake_quantize_learnable_per_tensor_affine(self, scale, zero_point, quant_min, quant_max, grad_factor);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__FAKE_QUANTIZE_LEARNABLE_PER_TENSOR_AFFINE, self, scale, zero_point, quant_min, quant_max, grad_factor);
}

Tensor wrap__fake_quantize_learnable_per_tensor_affine_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad, self, scale, zero_point);
    return at::redispatch::_fake_quantize_learnable_per_tensor_affine_backward(grad, self, scale, zero_point, quant_min, quant_max, grad_factor);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H__FAKE_QUANTIZE_LEARNABLE_PER_TENSOR_AFFINE_BACKWARD, grad, self, scale, zero_point, quant_min, quant_max, grad_factor);
}

Tensor wrap_fake_quantize_per_channel_affine(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, scale, zero_point);
    return at::redispatch::fake_quantize_per_channel_affine(self, scale, zero_point, axis, quant_min, quant_max);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FAKE_QUANTIZE_PER_CHANNEL_AFFINE, self, scale, zero_point, axis, quant_min, quant_max);
}

Tensor wrap_fake_quantize_per_channel_affine_cachemask(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, scale, zero_point);
    return at::redispatch::fake_quantize_per_channel_affine_cachemask(self, scale, zero_point, axis, quant_min, quant_max);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FAKE_QUANTIZE_PER_CHANNEL_AFFINE_CACHEMASK, self, scale, zero_point, axis, quant_min, quant_max);
}

Tensor wrap_fake_quantize_per_channel_affine_cachemask_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad, mask);
    return at::redispatch::fake_quantize_per_channel_affine_cachemask_backward(grad, mask);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H_FAKE_QUANTIZE_PER_CHANNEL_AFFINE_CACHEMASK_BACKWARD, grad, mask);
}

Tensor wrap__fake_quantize_learnable_per_channel_affine(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, scale, zero_point);
    return at::redispatch::_fake_quantize_learnable_per_channel_affine(self, scale, zero_point, axis, quant_min, quant_max, grad_factor);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__FAKE_QUANTIZE_LEARNABLE_PER_CHANNEL_AFFINE, self, scale, zero_point, axis, quant_min, quant_max, grad_factor);
}

Tensor wrap__fake_quantize_learnable_per_channel_affine_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad, self, scale, zero_point);
    return at::redispatch::_fake_quantize_learnable_per_channel_affine_backward(grad, self, scale, zero_point, axis, quant_min, quant_max, grad_factor);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H__FAKE_QUANTIZE_LEARNABLE_PER_CHANNEL_AFFINE_BACKWARD, grad, self, scale, zero_point, axis, quant_min, quant_max, grad_factor);
}

float wrap__choose_qparams_per_tensor(args...) {
  ensure_materialized(self);
  return at::redispatch::_choose_qparams_per_tensor(self, reduce_range);
}

Tensor wrap__saturate_weight_to_fp16(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(weight);
    return at::redispatch::_saturate_weight_to_fp16(weight);
  }
  return MK_TORCHY(weight.dtype(), weight.device(), H__SATURATE_WEIGHT_TO_FP16, weight);
}

Tensor wrap_choose_qparams_optimized(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input);
    return at::redispatch::choose_qparams_optimized(input, numel, n_bins, ratio, bit_width);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_CHOOSE_QPARAMS_OPTIMIZED, input, numel, n_bins, ratio, bit_width);
}

Tensor wrap_to_dtype_layout(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::to(self, dtype, layout, device, pin_memory, non_blocking, copy, memory_format);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TO_DTYPE_LAYOUT, self, dtype, layout, device, pin_memory, non_blocking, copy, memory_format);
}

Tensor wrap_to_device(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::to(self, device, dtype, non_blocking, copy, memory_format);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TO_DEVICE, self, device, dtype, non_blocking, copy, memory_format);
}

Tensor wrap_to_dtype(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::to(self, dtype, non_blocking, copy, memory_format);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TO_DTYPE, self, dtype, non_blocking, copy, memory_format);
}

Tensor wrap_to_other(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::to(self, other, non_blocking, copy, memory_format);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TO_OTHER, self, other, non_blocking, copy, memory_format);
}

Tensor[] wrap_meshgrid(args...) {
  ensure_materialized();
  return at::redispatch::meshgrid(tensors);
}

Tensor wrap_cartesian_prod(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::cartesian_prod(tensors);
  }
  return MK_TORCHY(None, None, H_CARTESIAN_PROD, tensors);
}

Tensor wrap_combinations(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::combinations(self, r, with_replacement);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_COMBINATIONS, self, r, with_replacement);
}

Scalar wrap_item(args...) {
  ensure_materialized(self);
  return at::redispatch::item(self);
}

ScalarType wrap_result_type_Tensor(args...) {
  ensure_materialized(tensor, other);
  return at::redispatch::result_type(tensor, other);
}

ScalarType wrap_result_type_Scalar(args...) {
  ensure_materialized(tensor);
  return at::redispatch::result_type(tensor, other);
}

ScalarType wrap_result_type_Scalar_Tensor(args...) {
  ensure_materialized(tensor);
  return at::redispatch::result_type(scalar, tensor);
}

ScalarType wrap_result_type_Scalar_Scalar(args...) {
  ensure_materialized();
  return at::redispatch::result_type(scalar1, scalar2);
}

bool wrap_can_cast(args...) {
  ensure_materialized();
  return at::redispatch::can_cast(from, to);
}

ScalarType wrap_promote_types(args...) {
  ensure_materialized();
  return at::redispatch::promote_types(type1, type2);
}

Scalar wrap__local_scalar_dense(args...) {
  ensure_materialized(self);
  return at::redispatch::_local_scalar_dense(self);
}

Tensor wrap__thnn_fused_lstm_cell(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input_gates, hidden_gates, cx);
    return at::redispatch::_thnn_fused_lstm_cell(input_gates, hidden_gates, cx, input_bias, hidden_bias);
  }
  return MK_TORCHY(input_gates.dtype(), input_gates.device(), H__THNN_FUSED_LSTM_CELL, input_gates, hidden_gates, cx, input_bias, hidden_bias);
}

Tensor wrap__thnn_fused_lstm_cell_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(cx, cy, workspace);
    return at::redispatch::_thnn_fused_lstm_cell_backward(grad_hy, grad_cy, cx, cy, workspace, has_bias);
  }
  return MK_TORCHY(cx.dtype(), cx.device(), H__THNN_FUSED_LSTM_CELL_BACKWARD, grad_hy, grad_cy, cx, cy, workspace, has_bias);
}

Tensor wrap__thnn_differentiable_lstm_cell_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input_gates, hidden_gates, cx, cy);
    return at::redispatch::_thnn_differentiable_lstm_cell_backward(grad_hy, grad_cy, input_gates, hidden_gates, input_bias, hidden_bias, cx, cy);
  }
  return MK_TORCHY(input_gates.dtype(), input_gates.device(), H__THNN_DIFFERENTIABLE_LSTM_CELL_BACKWARD, grad_hy, grad_cy, input_gates, hidden_gates, input_bias, hidden_bias, cx, cy);
}

Tensor wrap__thnn_fused_gru_cell(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input_gates, hidden_gates, hx);
    return at::redispatch::_thnn_fused_gru_cell(input_gates, hidden_gates, hx, input_bias, hidden_bias);
  }
  return MK_TORCHY(input_gates.dtype(), input_gates.device(), H__THNN_FUSED_GRU_CELL, input_gates, hidden_gates, hx, input_bias, hidden_bias);
}

Tensor wrap__thnn_fused_gru_cell_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_hy, workspace);
    return at::redispatch::_thnn_fused_gru_cell_backward(grad_hy, workspace, has_bias);
  }
  return MK_TORCHY(grad_hy.dtype(), grad_hy.device(), H__THNN_FUSED_GRU_CELL_BACKWARD, grad_hy, workspace, has_bias);
}

Tensor wrap__thnn_differentiable_gru_cell_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_hy, input_gates, hidden_gates, hx);
    return at::redispatch::_thnn_differentiable_gru_cell_backward(grad_hy, input_gates, hidden_gates, hx, input_bias, hidden_bias);
  }
  return MK_TORCHY(grad_hy.dtype(), grad_hy.device(), H__THNN_DIFFERENTIABLE_GRU_CELL_BACKWARD, grad_hy, input_gates, hidden_gates, hx, input_bias, hidden_bias);
}

Tensor wrap_lstm_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input);
    return at::redispatch::lstm(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_LSTM_INPUT, input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
}

Tensor wrap_lstm_data(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(data, batch_sizes);
    return at::redispatch::lstm(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
  }
  return MK_TORCHY(data.dtype(), data.device(), H_LSTM_DATA, data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
}

Tensor wrap_gru_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, hx);
    return at::redispatch::gru(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_GRU_INPUT, input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
}

Tensor wrap_gru_data(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(data, batch_sizes, hx);
    return at::redispatch::gru(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
  }
  return MK_TORCHY(data.dtype(), data.device(), H_GRU_DATA, data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
}

Tensor wrap_rnn_tanh_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, hx);
    return at::redispatch::rnn_tanh(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_RNN_TANH_INPUT, input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
}

Tensor wrap_rnn_tanh_data(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(data, batch_sizes, hx);
    return at::redispatch::rnn_tanh(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
  }
  return MK_TORCHY(data.dtype(), data.device(), H_RNN_TANH_DATA, data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
}

Tensor wrap_rnn_relu_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, hx);
    return at::redispatch::rnn_relu(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_RNN_RELU_INPUT, input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
}

Tensor wrap_rnn_relu_data(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(data, batch_sizes, hx);
    return at::redispatch::rnn_relu(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
  }
  return MK_TORCHY(data.dtype(), data.device(), H_RNN_RELU_DATA, data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
}

Tensor wrap_lstm_cell(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, w_ih, w_hh);
    return at::redispatch::lstm_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_LSTM_CELL, input, hx, w_ih, w_hh, b_ih, b_hh);
}

Tensor wrap_gru_cell(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, hx, w_ih, w_hh);
    return at::redispatch::gru_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_GRU_CELL, input, hx, w_ih, w_hh, b_ih, b_hh);
}

Tensor wrap_rnn_tanh_cell(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, hx, w_ih, w_hh);
    return at::redispatch::rnn_tanh_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_RNN_TANH_CELL, input, hx, w_ih, w_hh, b_ih, b_hh);
}

Tensor wrap_rnn_relu_cell(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, hx, w_ih, w_hh);
    return at::redispatch::rnn_relu_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_RNN_RELU_CELL, input, hx, w_ih, w_hh, b_ih, b_hh);
}

Tensor wrap_quantized_lstm_cell(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh);
    return at::redispatch::quantized_lstm_cell(input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_QUANTIZED_LSTM_CELL, input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
}

Tensor wrap_quantized_gru_cell(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh);
    return at::redispatch::quantized_gru_cell(input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_QUANTIZED_GRU_CELL, input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
}

Tensor wrap_quantized_rnn_relu_cell(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh);
    return at::redispatch::quantized_rnn_relu_cell(input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_QUANTIZED_RNN_RELU_CELL, input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
}

Tensor wrap_quantized_rnn_tanh_cell(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh);
    return at::redispatch::quantized_rnn_tanh_cell(input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_QUANTIZED_RNN_TANH_CELL, input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
}

Tensor wrap__pack_padded_sequence(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, lengths);
    return at::redispatch::_pack_padded_sequence(input, lengths, batch_first);
  }
  return MK_TORCHY(input.dtype(), input.device(), H__PACK_PADDED_SEQUENCE, input, lengths, batch_first);
}

Tensor wrap__pack_padded_sequence_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad, batch_sizes);
    return at::redispatch::_pack_padded_sequence_backward(grad, input_size, batch_sizes, batch_first);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H__PACK_PADDED_SEQUENCE_BACKWARD, grad, input_size, batch_sizes, batch_first);
}

Tensor wrap__pad_packed_sequence(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(data, batch_sizes);
    return at::redispatch::_pad_packed_sequence(data, batch_sizes, batch_first, padding_value, total_length);
  }
  return MK_TORCHY(data.dtype(), data.device(), H__PAD_PACKED_SEQUENCE, data, batch_sizes, batch_first, padding_value, total_length);
}

Tensor wrap_set__source_Storage(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::set_(self, source);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SET__SOURCE_STORAGE, self, source);
}

Tensor wrap_set__source_Storage_storage_offset(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::set_(self, source, storage_offset, size, stride);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SET__SOURCE_STORAGE_STORAGE_OFFSET, self, source, storage_offset, size, stride);
}

Tensor wrap_set__source_Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, source);
    return at::redispatch::set_(self, source);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SET__SOURCE_TENSOR, self, source);
}

Tensor wrap_set_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::set_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SET_, self);
}

bool wrap_is_set_to(args...) {
  ensure_materialized(self, tensor);
  return at::redispatch::is_set_to(self, tensor);
}

Tensor wrap_masked_fill__Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, mask);
    return at::redispatch::masked_fill_(self, mask, value);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MASKED_FILL__SCALAR, self, mask, value);
}

Tensor wrap_masked_fill_Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, mask);
    return at::redispatch::masked_fill(self, mask, value);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MASKED_FILL_SCALAR, self, mask, value);
}

Tensor wrap_masked_fill__Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, mask, value);
    return at::redispatch::masked_fill_(self, mask, value);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MASKED_FILL__TENSOR, self, mask, value);
}

Tensor wrap_masked_fill_Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, mask, value);
    return at::redispatch::masked_fill(self, mask, value);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MASKED_FILL_TENSOR, self, mask, value);
}

Tensor wrap_masked_scatter_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, mask, source);
    return at::redispatch::masked_scatter_(self, mask, source);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MASKED_SCATTER_, self, mask, source);
}

Tensor wrap_masked_scatter(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, mask, source);
    return at::redispatch::masked_scatter(self, mask, source);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MASKED_SCATTER, self, mask, source);
}

Tensor wrap_view(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::view(self, size);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_VIEW, self, size);
}

Tensor wrap_view_dtype(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::view(self, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_VIEW_DTYPE, self, dtype);
}

Tensor wrap_put_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index, source);
    return at::redispatch::put_(self, index, source, accumulate);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_PUT_, self, index, source, accumulate);
}

Tensor wrap_put(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index, source);
    return at::redispatch::put(self, index, source, accumulate);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_PUT, self, index, source, accumulate);
}

Tensor wrap_index_add_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index, source);
    return at::redispatch::index_add_(self, dim, index, source);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INDEX_ADD_, self, dim, index, source);
}

Tensor wrap_index_add__alpha(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index, source);
    return at::redispatch::index_add_(self, dim, index, source, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INDEX_ADD__ALPHA, self, dim, index, source, alpha);
}

Tensor wrap_index_add(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index, source);
    return at::redispatch::index_add(self, dim, index, source);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INDEX_ADD, self, dim, index, source);
}

Tensor wrap_index_add_alpha(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index, source);
    return at::redispatch::index_add(self, dim, index, source, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INDEX_ADD_ALPHA, self, dim, index, source, alpha);
}

Tensor wrap_index_add_dimname(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index, source);
    return at::redispatch::index_add(self, dim, index, source, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INDEX_ADD_DIMNAME, self, dim, index, source, alpha);
}

Tensor wrap_index_fill__int_Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index);
    return at::redispatch::index_fill_(self, dim, index, value);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INDEX_FILL__INT_SCALAR, self, dim, index, value);
}

Tensor wrap_index_fill_int_Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index);
    return at::redispatch::index_fill(self, dim, index, value);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INDEX_FILL_INT_SCALAR, self, dim, index, value);
}

Tensor wrap_index_fill__int_Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index, value);
    return at::redispatch::index_fill_(self, dim, index, value);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INDEX_FILL__INT_TENSOR, self, dim, index, value);
}

Tensor wrap_index_fill_int_Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index, value);
    return at::redispatch::index_fill(self, dim, index, value);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INDEX_FILL_INT_TENSOR, self, dim, index, value);
}

Tensor wrap_index_fill__Dimname_Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index);
    return at::redispatch::index_fill_(self, dim, index, value);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INDEX_FILL__DIMNAME_SCALAR, self, dim, index, value);
}

Tensor wrap_index_fill__Dimname_Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index, value);
    return at::redispatch::index_fill_(self, dim, index, value);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INDEX_FILL__DIMNAME_TENSOR, self, dim, index, value);
}

Tensor wrap_index_fill_Dimname_Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index);
    return at::redispatch::index_fill(self, dim, index, value);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INDEX_FILL_DIMNAME_SCALAR, self, dim, index, value);
}

Tensor wrap_index_fill_Dimname_Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index, value);
    return at::redispatch::index_fill(self, dim, index, value);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INDEX_FILL_DIMNAME_TENSOR, self, dim, index, value);
}

Tensor wrap_scatter__src(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index, src);
    return at::redispatch::scatter_(self, dim, index, src);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SCATTER__SRC, self, dim, index, src);
}

Tensor wrap_scatter_src(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index, src);
    return at::redispatch::scatter(self, dim, index, src);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SCATTER_SRC, self, dim, index, src);
}

Tensor wrap_scatter__value(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index);
    return at::redispatch::scatter_(self, dim, index, value);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SCATTER__VALUE, self, dim, index, value);
}

Tensor wrap_scatter_value(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index);
    return at::redispatch::scatter(self, dim, index, value);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SCATTER_VALUE, self, dim, index, value);
}

Tensor wrap_scatter_dimname_src(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index, src);
    return at::redispatch::scatter(self, dim, index, src);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SCATTER_DIMNAME_SRC, self, dim, index, src);
}

Tensor wrap_scatter_dimname_value(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index);
    return at::redispatch::scatter(self, dim, index, value);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SCATTER_DIMNAME_VALUE, self, dim, index, value);
}

Tensor wrap_scatter__reduce(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index, src);
    return at::redispatch::scatter_(self, dim, index, src, reduce);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SCATTER__REDUCE, self, dim, index, src, reduce);
}

Tensor wrap_scatter__value_reduce(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index);
    return at::redispatch::scatter_(self, dim, index, value, reduce);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SCATTER__VALUE_REDUCE, self, dim, index, value, reduce);
}

Tensor wrap_scatter_add_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index, src);
    return at::redispatch::scatter_add_(self, dim, index, src);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SCATTER_ADD_, self, dim, index, src);
}

Tensor wrap_scatter_add(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index, src);
    return at::redispatch::scatter_add(self, dim, index, src);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SCATTER_ADD, self, dim, index, src);
}

Tensor wrap_scatter_add_dimname(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index, src);
    return at::redispatch::scatter_add(self, dim, index, src);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SCATTER_ADD_DIMNAME, self, dim, index, src);
}

Tensor wrap_eq__Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::eq_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_EQ__SCALAR, self, other);
}

Tensor wrap_eq__Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::eq_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_EQ__TENSOR, self, other);
}

Tensor wrap_bitwise_and_Tensor_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::bitwise_and(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_BITWISE_AND_TENSOR_OUT, out, self, other);
}

Tensor wrap_bitwise_and_Scalar_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::bitwise_and(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_BITWISE_AND_SCALAR_OUT, out, self, other);
}

Tensor wrap_bitwise_and_Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::bitwise_and(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BITWISE_AND_SCALAR, self, other);
}

Tensor wrap_bitwise_and_Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::bitwise_and(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BITWISE_AND_TENSOR, self, other);
}

Tensor wrap_bitwise_and__Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::bitwise_and_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BITWISE_AND__SCALAR, self, other);
}

Tensor wrap_bitwise_and__Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::bitwise_and_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BITWISE_AND__TENSOR, self, other);
}

Tensor wrap___and___Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::__and__(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H___AND___SCALAR, self, other);
}

Tensor wrap___and___Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::__and__(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H___AND___TENSOR, self, other);
}

Tensor wrap___iand___Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::__iand__(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H___IAND___SCALAR, self, other);
}

Tensor wrap___iand___Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::__iand__(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H___IAND___TENSOR, self, other);
}

Tensor wrap_bitwise_or_Tensor_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::bitwise_or(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_BITWISE_OR_TENSOR_OUT, out, self, other);
}

Tensor wrap_bitwise_or_Scalar_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::bitwise_or(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_BITWISE_OR_SCALAR_OUT, out, self, other);
}

Tensor wrap_bitwise_or_Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::bitwise_or(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BITWISE_OR_SCALAR, self, other);
}

Tensor wrap_bitwise_or_Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::bitwise_or(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BITWISE_OR_TENSOR, self, other);
}

Tensor wrap_bitwise_or__Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::bitwise_or_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BITWISE_OR__SCALAR, self, other);
}

Tensor wrap_bitwise_or__Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::bitwise_or_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BITWISE_OR__TENSOR, self, other);
}

Tensor wrap___or___Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::__or__(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H___OR___SCALAR, self, other);
}

Tensor wrap___or___Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::__or__(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H___OR___TENSOR, self, other);
}

Tensor wrap___ior___Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::__ior__(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H___IOR___SCALAR, self, other);
}

Tensor wrap___ior___Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::__ior__(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H___IOR___TENSOR, self, other);
}

Tensor wrap_bitwise_xor_Tensor_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::bitwise_xor(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_BITWISE_XOR_TENSOR_OUT, out, self, other);
}

Tensor wrap_bitwise_xor_Scalar_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::bitwise_xor(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_BITWISE_XOR_SCALAR_OUT, out, self, other);
}

Tensor wrap_bitwise_xor_Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::bitwise_xor(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BITWISE_XOR_SCALAR, self, other);
}

Tensor wrap_bitwise_xor_Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::bitwise_xor(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BITWISE_XOR_TENSOR, self, other);
}

Tensor wrap_bitwise_xor__Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::bitwise_xor_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BITWISE_XOR__SCALAR, self, other);
}

Tensor wrap_bitwise_xor__Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::bitwise_xor_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BITWISE_XOR__TENSOR, self, other);
}

Tensor wrap___xor___Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::__xor__(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H___XOR___SCALAR, self, other);
}

Tensor wrap___xor___Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::__xor__(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H___XOR___TENSOR, self, other);
}

Tensor wrap___ixor___Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::__ixor__(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H___IXOR___SCALAR, self, other);
}

Tensor wrap___ixor___Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::__ixor__(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H___IXOR___TENSOR, self, other);
}

Tensor wrap___lshift___Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::__lshift__(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H___LSHIFT___SCALAR, self, other);
}

Tensor wrap___lshift___Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::__lshift__(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H___LSHIFT___TENSOR, self, other);
}

Tensor wrap___ilshift___Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::__ilshift__(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H___ILSHIFT___SCALAR, self, other);
}

Tensor wrap___ilshift___Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::__ilshift__(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H___ILSHIFT___TENSOR, self, other);
}

Tensor wrap___rshift___Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::__rshift__(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H___RSHIFT___SCALAR, self, other);
}

Tensor wrap___rshift___Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::__rshift__(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H___RSHIFT___TENSOR, self, other);
}

Tensor wrap___irshift___Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::__irshift__(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H___IRSHIFT___SCALAR, self, other);
}

Tensor wrap___irshift___Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::__irshift__(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H___IRSHIFT___TENSOR, self, other);
}

Tensor wrap_tril_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::tril_(self, diagonal);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TRIL_, self, diagonal);
}

Tensor wrap_triu_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::triu_(self, diagonal);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TRIU_, self, diagonal);
}

Tensor wrap_renorm_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::renorm_(self, p, dim, maxnorm);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RENORM_, self, p, dim, maxnorm);
}

Tensor wrap_lerp__Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, end);
    return at::redispatch::lerp_(self, end, weight);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LERP__SCALAR, self, end, weight);
}

Tensor wrap_lerp__Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, end, weight);
    return at::redispatch::lerp_(self, end, weight);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LERP__TENSOR, self, end, weight);
}

Tensor wrap_fmod__Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::fmod_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FMOD__SCALAR, self, other);
}

Tensor wrap_fmod__Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::fmod_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FMOD__TENSOR, self, other);
}

Tensor wrap_remainder__Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::remainder_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_REMAINDER__SCALAR, self, other);
}

Tensor wrap_remainder__Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::remainder_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_REMAINDER__TENSOR, self, other);
}

Tensor wrap_addbmm_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, batch1, batch2);
    return at::redispatch::addbmm_(self, batch1, batch2, beta, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ADDBMM_, self, batch1, batch2, beta, alpha);
}

Tensor wrap_addbmm_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, batch1, batch2);
    return at::redispatch::addbmm(out, self, batch1, batch2, beta, alpha);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ADDBMM_OUT, out, self, batch1, batch2, beta, alpha);
}

Tensor wrap_addbmm(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, batch1, batch2);
    return at::redispatch::addbmm(self, batch1, batch2, beta, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ADDBMM, self, batch1, batch2, beta, alpha);
}

Tensor wrap_addcdiv_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, tensor1, tensor2);
    return at::redispatch::addcdiv_(self, tensor1, tensor2, value);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ADDCDIV_, self, tensor1, tensor2, value);
}

Tensor wrap_random__from(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::random_(self, from, to, generator);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RANDOM__FROM, self, from, to, generator);
}

Tensor wrap_random__to(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::random_(self, to, generator);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RANDOM__TO, self, to, generator);
}

Tensor wrap_random_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::random_(self, generator);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RANDOM_, self, generator);
}

Tensor wrap_uniform_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::uniform_(self, from, to, generator);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_UNIFORM_, self, from, to, generator);
}

Tensor wrap_cauchy_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::cauchy_(self, median, sigma, generator);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CAUCHY_, self, median, sigma, generator);
}

Tensor wrap_log_normal_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::log_normal_(self, mean, std, generator);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LOG_NORMAL_, self, mean, std, generator);
}

Tensor wrap_exponential_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::exponential_(self, lambd, generator);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_EXPONENTIAL_, self, lambd, generator);
}

Tensor wrap_geometric_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::geometric_(self, p, generator);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_GEOMETRIC_, self, p, generator);
}

Tensor wrap_diag_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::diag(out, self, diagonal);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_DIAG_OUT, out, self, diagonal);
}

Tensor wrap_diag(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::diag(self, diagonal);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DIAG, self, diagonal);
}

Tensor wrap_diag_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad);
    return at::redispatch::diag_backward(grad, input_sizes, diagonal);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H_DIAG_BACKWARD, grad, input_sizes, diagonal);
}

Tensor wrap_cross_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::cross(out, self, other, dim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_CROSS_OUT, out, self, other, dim);
}

Tensor wrap_cross(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::cross(self, other, dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CROSS, self, other, dim);
}

Tensor wrap_triu_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::triu(out, self, diagonal);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_TRIU_OUT, out, self, diagonal);
}

Tensor wrap_triu(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::triu(self, diagonal);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TRIU, self, diagonal);
}

Tensor wrap_tril_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::tril(out, self, diagonal);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_TRIL_OUT, out, self, diagonal);
}

Tensor wrap_tril(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::tril(self, diagonal);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TRIL, self, diagonal);
}

Tensor wrap_tril_indices(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::tril_indices(row, col, offset, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_TRIL_INDICES, row, col, offset, dtype, layout, device, pin_memory);
}

Tensor wrap_triu_indices(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::triu_indices(row, col, offset, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_TRIU_INDICES, row, col, offset, dtype, layout, device, pin_memory);
}

Tensor wrap_trace(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::trace(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TRACE, self);
}

Tensor wrap_trace_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad);
    return at::redispatch::trace_backward(grad, sizes);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H_TRACE_BACKWARD, grad, sizes);
}

Tensor wrap_ne_Scalar_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::ne(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NE_SCALAR_OUT, out, self, other);
}

Tensor wrap_ne_Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::ne(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NE_SCALAR, self, other);
}

Tensor wrap_ne_Tensor_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::ne(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NE_TENSOR_OUT, out, self, other);
}

Tensor wrap_ne_Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::ne(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NE_TENSOR, self, other);
}

Tensor wrap_ne__Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::ne_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NE__SCALAR, self, other);
}

Tensor wrap_ne__Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::ne_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NE__TENSOR, self, other);
}

Tensor wrap_not_equal_Scalar_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::not_equal(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NOT_EQUAL_SCALAR_OUT, out, self, other);
}

Tensor wrap_not_equal_Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::not_equal(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NOT_EQUAL_SCALAR, self, other);
}

Tensor wrap_not_equal_Tensor_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::not_equal(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NOT_EQUAL_TENSOR_OUT, out, self, other);
}

Tensor wrap_not_equal_Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::not_equal(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NOT_EQUAL_TENSOR, self, other);
}

Tensor wrap_not_equal__Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::not_equal_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NOT_EQUAL__SCALAR, self, other);
}

Tensor wrap_not_equal__Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::not_equal_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NOT_EQUAL__TENSOR, self, other);
}

Tensor wrap_eq_Scalar_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::eq(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_EQ_SCALAR_OUT, out, self, other);
}

Tensor wrap_eq_Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::eq(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_EQ_SCALAR, self, other);
}

Tensor wrap_eq_Tensor_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::eq(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_EQ_TENSOR_OUT, out, self, other);
}

Tensor wrap_eq_Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::eq(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_EQ_TENSOR, self, other);
}

Tensor wrap_ge_Scalar_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::ge(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_GE_SCALAR_OUT, out, self, other);
}

Tensor wrap_ge_Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::ge(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_GE_SCALAR, self, other);
}

Tensor wrap_ge_Tensor_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::ge(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_GE_TENSOR_OUT, out, self, other);
}

Tensor wrap_ge_Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::ge(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_GE_TENSOR, self, other);
}

Tensor wrap_ge__Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::ge_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_GE__SCALAR, self, other);
}

Tensor wrap_ge__Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::ge_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_GE__TENSOR, self, other);
}

Tensor wrap_greater_equal_Scalar_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::greater_equal(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_GREATER_EQUAL_SCALAR_OUT, out, self, other);
}

Tensor wrap_greater_equal_Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::greater_equal(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_GREATER_EQUAL_SCALAR, self, other);
}

Tensor wrap_greater_equal_Tensor_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::greater_equal(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_GREATER_EQUAL_TENSOR_OUT, out, self, other);
}

Tensor wrap_greater_equal_Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::greater_equal(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_GREATER_EQUAL_TENSOR, self, other);
}

Tensor wrap_greater_equal__Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::greater_equal_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_GREATER_EQUAL__SCALAR, self, other);
}

Tensor wrap_greater_equal__Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::greater_equal_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_GREATER_EQUAL__TENSOR, self, other);
}

Tensor wrap_le_Scalar_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::le(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LE_SCALAR_OUT, out, self, other);
}

Tensor wrap_le_Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::le(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LE_SCALAR, self, other);
}

Tensor wrap_le_Tensor_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::le(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LE_TENSOR_OUT, out, self, other);
}

Tensor wrap_le_Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::le(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LE_TENSOR, self, other);
}

Tensor wrap_le__Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::le_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LE__SCALAR, self, other);
}

Tensor wrap_le__Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::le_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LE__TENSOR, self, other);
}

Tensor wrap_less_equal_Scalar_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::less_equal(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LESS_EQUAL_SCALAR_OUT, out, self, other);
}

Tensor wrap_less_equal_Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::less_equal(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LESS_EQUAL_SCALAR, self, other);
}

Tensor wrap_less_equal_Tensor_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::less_equal(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LESS_EQUAL_TENSOR_OUT, out, self, other);
}

Tensor wrap_less_equal_Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::less_equal(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LESS_EQUAL_TENSOR, self, other);
}

Tensor wrap_less_equal__Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::less_equal_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LESS_EQUAL__SCALAR, self, other);
}

Tensor wrap_less_equal__Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::less_equal_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LESS_EQUAL__TENSOR, self, other);
}

Tensor wrap_gt_Scalar_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::gt(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_GT_SCALAR_OUT, out, self, other);
}

Tensor wrap_gt_Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::gt(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_GT_SCALAR, self, other);
}

Tensor wrap_gt_Tensor_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::gt(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_GT_TENSOR_OUT, out, self, other);
}

Tensor wrap_gt_Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::gt(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_GT_TENSOR, self, other);
}

Tensor wrap_gt__Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::gt_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_GT__SCALAR, self, other);
}

Tensor wrap_gt__Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::gt_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_GT__TENSOR, self, other);
}

Tensor wrap_greater_Scalar_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::greater(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_GREATER_SCALAR_OUT, out, self, other);
}

Tensor wrap_greater_Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::greater(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_GREATER_SCALAR, self, other);
}

Tensor wrap_greater_Tensor_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::greater(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_GREATER_TENSOR_OUT, out, self, other);
}

Tensor wrap_greater_Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::greater(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_GREATER_TENSOR, self, other);
}

Tensor wrap_greater__Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::greater_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_GREATER__SCALAR, self, other);
}

Tensor wrap_greater__Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::greater_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_GREATER__TENSOR, self, other);
}

Tensor wrap_lt_Scalar_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::lt(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LT_SCALAR_OUT, out, self, other);
}

Tensor wrap_lt_Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::lt(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LT_SCALAR, self, other);
}

Tensor wrap_lt_Tensor_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::lt(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LT_TENSOR_OUT, out, self, other);
}

Tensor wrap_lt_Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::lt(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LT_TENSOR, self, other);
}

Tensor wrap_lt__Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::lt_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LT__SCALAR, self, other);
}

Tensor wrap_lt__Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::lt_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LT__TENSOR, self, other);
}

Tensor wrap_less_Scalar_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::less(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LESS_SCALAR_OUT, out, self, other);
}

Tensor wrap_less_Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::less(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LESS_SCALAR, self, other);
}

Tensor wrap_less_Tensor_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::less(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LESS_TENSOR_OUT, out, self, other);
}

Tensor wrap_less_Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::less(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LESS_TENSOR, self, other);
}

Tensor wrap_less__Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::less_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LESS__SCALAR, self, other);
}

Tensor wrap_less__Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::less_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LESS__TENSOR, self, other);
}

Tensor wrap_take_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, index);
    return at::redispatch::take(out, self, index);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_TAKE_OUT, out, self, index);
}

Tensor wrap_take(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index);
    return at::redispatch::take(self, index);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TAKE, self, index);
}

Tensor wrap_take_along_dim_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, indices);
    return at::redispatch::take_along_dim(out, self, indices, dim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_TAKE_ALONG_DIM_OUT, out, self, indices, dim);
}

Tensor wrap_take_along_dim(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, indices);
    return at::redispatch::take_along_dim(self, indices, dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TAKE_ALONG_DIM, self, indices, dim);
}

Tensor wrap_index_select_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, index);
    return at::redispatch::index_select(out, self, dim, index);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_INDEX_SELECT_OUT, out, self, dim, index);
}

Tensor wrap_index_select(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index);
    return at::redispatch::index_select(self, dim, index);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INDEX_SELECT, self, dim, index);
}

Tensor wrap_index_select_dimname_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, index);
    return at::redispatch::index_select(out, self, dim, index);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_INDEX_SELECT_DIMNAME_OUT, out, self, dim, index);
}

Tensor wrap_index_select_dimname(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index);
    return at::redispatch::index_select(self, dim, index);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INDEX_SELECT_DIMNAME, self, dim, index);
}

Tensor wrap_index_select_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad, index);
    return at::redispatch::index_select_backward(grad, self_sizes, dim, index);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H_INDEX_SELECT_BACKWARD, grad, self_sizes, dim, index);
}

Tensor wrap_masked_select_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, mask);
    return at::redispatch::masked_select(out, self, mask);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_MASKED_SELECT_OUT, out, self, mask);
}

Tensor wrap_masked_select(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, mask);
    return at::redispatch::masked_select(self, mask);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MASKED_SELECT, self, mask);
}

Tensor wrap_masked_select_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad, input, mask);
    return at::redispatch::masked_select_backward(grad, input, mask);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H_MASKED_SELECT_BACKWARD, grad, input, mask);
}

Tensor wrap_nonzero_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::nonzero(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NONZERO_OUT, out, self);
}

Tensor wrap_nonzero(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::nonzero(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NONZERO, self);
}

Tensor[] wrap_nonzero_numpy(args...) {
  ensure_materialized(self);
  return at::redispatch::nonzero_numpy(self);
}

Tensor wrap_gather_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, index);
    return at::redispatch::gather(out, self, dim, index, sparse_grad);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_GATHER_OUT, out, self, dim, index, sparse_grad);
}

Tensor wrap_gather(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index);
    return at::redispatch::gather(self, dim, index, sparse_grad);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_GATHER, self, dim, index, sparse_grad);
}

Tensor wrap_gather_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad, self, index);
    return at::redispatch::gather_backward(grad, self, dim, index, sparse_grad);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H_GATHER_BACKWARD, grad, self, dim, index, sparse_grad);
}

Tensor wrap_gather_dimname_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, index);
    return at::redispatch::gather(out, self, dim, index, sparse_grad);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_GATHER_DIMNAME_OUT, out, self, dim, index, sparse_grad);
}

Tensor wrap_gather_dimname(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index);
    return at::redispatch::gather(self, dim, index, sparse_grad);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_GATHER_DIMNAME, self, dim, index, sparse_grad);
}

Tensor wrap__gather_sparse_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index, grad);
    return at::redispatch::_gather_sparse_backward(self, dim, index, grad);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__GATHER_SPARSE_BACKWARD, self, dim, index, grad);
}

Tensor wrap_addcmul_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, tensor1, tensor2);
    return at::redispatch::addcmul(out, self, tensor1, tensor2, value);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ADDCMUL_OUT, out, self, tensor1, tensor2, value);
}

Tensor wrap_addcmul(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, tensor1, tensor2);
    return at::redispatch::addcmul(self, tensor1, tensor2, value);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ADDCMUL, self, tensor1, tensor2, value);
}

Tensor wrap_addcmul_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, tensor1, tensor2);
    return at::redispatch::addcmul_(self, tensor1, tensor2, value);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ADDCMUL_, self, tensor1, tensor2, value);
}

Tensor wrap_addcdiv_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, tensor1, tensor2);
    return at::redispatch::addcdiv(out, self, tensor1, tensor2, value);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ADDCDIV_OUT, out, self, tensor1, tensor2, value);
}

Tensor wrap_addcdiv(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, tensor1, tensor2);
    return at::redispatch::addcdiv(self, tensor1, tensor2, value);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ADDCDIV, self, tensor1, tensor2, value);
}

Tensor wrap_cross_entropy_loss(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, target);
    return at::redispatch::cross_entropy_loss(self, target, weight, reduction, ignore_index);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CROSS_ENTROPY_LOSS, self, target, weight, reduction, ignore_index);
}

Tensor wrap_lstsq_X(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(X, qr, self, A);
    return at::redispatch::lstsq(X, qr, self, A);
  }
  return MK_TORCHY(X.dtype(), X.device(), H_LSTSQ_X, X, qr, self, A);
}

Tensor wrap_lstsq(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, A);
    return at::redispatch::lstsq(self, A);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LSTSQ, self, A);
}

Tensor wrap_triangular_solve_X(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(X, M, self, A);
    return at::redispatch::triangular_solve(X, M, self, A, upper, transpose, unitriangular);
  }
  return MK_TORCHY(X.dtype(), X.device(), H_TRIANGULAR_SOLVE_X, X, M, self, A, upper, transpose, unitriangular);
}

Tensor wrap_triangular_solve(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, A);
    return at::redispatch::triangular_solve(self, A, upper, transpose, unitriangular);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TRIANGULAR_SOLVE, self, A, upper, transpose, unitriangular);
}

Tensor wrap_symeig_e(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(e, V, self);
    return at::redispatch::symeig(e, V, self, eigenvectors, upper);
  }
  return MK_TORCHY(e.dtype(), e.device(), H_SYMEIG_E, e, V, self, eigenvectors, upper);
}

Tensor wrap_symeig(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::symeig(self, eigenvectors, upper);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SYMEIG, self, eigenvectors, upper);
}

Tensor wrap__symeig_helper(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_symeig_helper(self, eigenvectors, upper);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__SYMEIG_HELPER, self, eigenvectors, upper);
}

Tensor wrap_eig_e(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(e, v, self);
    return at::redispatch::eig(e, v, self, eigenvectors);
  }
  return MK_TORCHY(e.dtype(), e.device(), H_EIG_E, e, v, self, eigenvectors);
}

Tensor wrap_eig(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::eig(self, eigenvectors);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_EIG, self, eigenvectors);
}

Tensor wrap_svd_U(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(U, S, V, self);
    return at::redispatch::svd(U, S, V, self, some, compute_uv);
  }
  return MK_TORCHY(U.dtype(), U.device(), H_SVD_U, U, S, V, self, some, compute_uv);
}

Tensor wrap_svd(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::svd(self, some, compute_uv);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SVD, self, some, compute_uv);
}

Tensor wrap__svd_helper(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_svd_helper(self, some, compute_uv);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__SVD_HELPER, self, some, compute_uv);
}

Tensor wrap_swapaxes(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::swapaxes(self, axis0, axis1);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SWAPAXES, self, axis0, axis1);
}

Tensor wrap_swapaxes_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::swapaxes_(self, axis0, axis1);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SWAPAXES_, self, axis0, axis1);
}

Tensor wrap_swapdims(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::swapdims(self, dim0, dim1);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SWAPDIMS, self, dim0, dim1);
}

Tensor wrap_swapdims_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::swapdims_(self, dim0, dim1);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SWAPDIMS_, self, dim0, dim1);
}

Tensor wrap_cholesky_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::cholesky(out, self, upper);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_CHOLESKY_OUT, out, self, upper);
}

Tensor wrap_cholesky(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::cholesky(self, upper);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CHOLESKY, self, upper);
}

Tensor wrap__cholesky_helper(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_cholesky_helper(self, upper);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__CHOLESKY_HELPER, self, upper);
}

Tensor wrap_cholesky_solve_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, input2);
    return at::redispatch::cholesky_solve(out, self, input2, upper);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_CHOLESKY_SOLVE_OUT, out, self, input2, upper);
}

Tensor wrap_cholesky_solve(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, input2);
    return at::redispatch::cholesky_solve(self, input2, upper);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CHOLESKY_SOLVE, self, input2, upper);
}

Tensor wrap__cholesky_solve_helper(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, A);
    return at::redispatch::_cholesky_solve_helper(self, A, upper);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__CHOLESKY_SOLVE_HELPER, self, A, upper);
}

Tensor wrap_solve(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, A);
    return at::redispatch::solve(self, A);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SOLVE, self, A);
}

Tensor wrap_solve_solution(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(solution, lu, self, A);
    return at::redispatch::solve(solution, lu, self, A);
  }
  return MK_TORCHY(solution.dtype(), solution.device(), H_SOLVE_SOLUTION, solution, lu, self, A);
}

Tensor wrap__solve_helper(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, A);
    return at::redispatch::_solve_helper(self, A);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__SOLVE_HELPER, self, A);
}

Tensor wrap_cholesky_inverse(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::cholesky_inverse(self, upper);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CHOLESKY_INVERSE, self, upper);
}

Tensor wrap_cholesky_inverse_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::cholesky_inverse(out, self, upper);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_CHOLESKY_INVERSE_OUT, out, self, upper);
}

Tensor wrap_qr_Q(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(Q, R, self);
    return at::redispatch::qr(Q, R, self, some);
  }
  return MK_TORCHY(Q.dtype(), Q.device(), H_QR_Q, Q, R, self, some);
}

Tensor wrap_qr(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::qr(self, some);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_QR, self, some);
}

Tensor wrap_geqrf_a(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(a, tau, self);
    return at::redispatch::geqrf(a, tau, self);
  }
  return MK_TORCHY(a.dtype(), a.device(), H_GEQRF_A, a, tau, self);
}

Tensor wrap_geqrf(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::geqrf(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_GEQRF, self);
}

Tensor wrap_orgqr(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, input2);
    return at::redispatch::orgqr(self, input2);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ORGQR, self, input2);
}

Tensor wrap_orgqr_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, input2);
    return at::redispatch::orgqr(out, self, input2);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ORGQR_OUT, out, self, input2);
}

Tensor wrap_ormqr_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, input2, input3);
    return at::redispatch::ormqr(out, self, input2, input3, left, transpose);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ORMQR_OUT, out, self, input2, input3, left, transpose);
}

Tensor wrap_ormqr(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, input2, input3);
    return at::redispatch::ormqr(self, input2, input3, left, transpose);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ORMQR, self, input2, input3, left, transpose);
}

Tensor wrap__lu_with_info(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_lu_with_info(self, pivot, check_errors);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__LU_WITH_INFO, self, pivot, check_errors);
}

Tensor wrap_lu_solve_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, LU_data, LU_pivots);
    return at::redispatch::lu_solve(out, self, LU_data, LU_pivots);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LU_SOLVE_OUT, out, self, LU_data, LU_pivots);
}

Tensor wrap_lu_solve(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, LU_data, LU_pivots);
    return at::redispatch::lu_solve(self, LU_data, LU_pivots);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LU_SOLVE, self, LU_data, LU_pivots);
}

Tensor wrap__lu_solve_helper(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, LU_data, LU_pivots);
    return at::redispatch::_lu_solve_helper(self, LU_data, LU_pivots);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__LU_SOLVE_HELPER, self, LU_data, LU_pivots);
}

Tensor wrap_multinomial_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::multinomial(out, self, num_samples, replacement, generator);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_MULTINOMIAL_OUT, out, self, num_samples, replacement, generator);
}

Tensor wrap_multinomial(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::multinomial(self, num_samples, replacement, generator);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MULTINOMIAL, self, num_samples, replacement, generator);
}

Tensor wrap_lgamma_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::lgamma(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LGAMMA_OUT, out, self);
}

Tensor wrap_digamma_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::digamma(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_DIGAMMA_OUT, out, self);
}

Tensor wrap_polygamma_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::polygamma(out, n, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_POLYGAMMA_OUT, out, n, self);
}

Tensor wrap_polygamma(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::polygamma(n, self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_POLYGAMMA, n, self);
}

Tensor wrap_polygamma_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::polygamma_(self, n);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_POLYGAMMA_, self, n);
}

Tensor wrap_erfinv_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::erfinv(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ERFINV_OUT, out, self);
}

Tensor wrap_i0(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::i0(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_I0, self);
}

Tensor wrap_i0_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::i0_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_I0_, self);
}

Tensor wrap_i0_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::i0(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_I0_OUT, out, self);
}

Tensor wrap_sign(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::sign(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SIGN, self);
}

Tensor wrap_sign_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::sign_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SIGN_, self);
}

Tensor wrap_sign_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::sign(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SIGN_OUT, out, self);
}

Tensor wrap_signbit(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::signbit(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SIGNBIT, self);
}

Tensor wrap_signbit_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::signbit(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SIGNBIT_OUT, out, self);
}

Tensor wrap_dist(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::dist(self, other, p);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DIST, self, other, p);
}

Tensor wrap_atan2_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::atan2(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ATAN2_OUT, out, self, other);
}

Tensor wrap_lerp_Scalar_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, end);
    return at::redispatch::lerp(out, self, end, weight);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LERP_SCALAR_OUT, out, self, end, weight);
}

Tensor wrap_lerp_Tensor_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, end, weight);
    return at::redispatch::lerp(out, self, end, weight);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LERP_TENSOR_OUT, out, self, end, weight);
}

Tensor wrap_lerp_Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, end);
    return at::redispatch::lerp(self, end, weight);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LERP_SCALAR, self, end, weight);
}

Tensor wrap_lerp_Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, end, weight);
    return at::redispatch::lerp(self, end, weight);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LERP_TENSOR, self, end, weight);
}

Tensor wrap_histc_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::histc(out, self, bins, min, max);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_HISTC_OUT, out, self, bins, min, max);
}

Tensor wrap_histc(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::histc(self, bins, min, max);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_HISTC, self, bins, min, max);
}

Tensor wrap_fmod_Scalar_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::fmod(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FMOD_SCALAR_OUT, out, self, other);
}

Tensor wrap_fmod_Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::fmod(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FMOD_SCALAR, self, other);
}

Tensor wrap_fmod_Tensor_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::fmod(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FMOD_TENSOR_OUT, out, self, other);
}

Tensor wrap_fmod_Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::fmod(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FMOD_TENSOR, self, other);
}

Tensor wrap_hypot_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::hypot(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_HYPOT_OUT, out, self, other);
}

Tensor wrap_hypot(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::hypot(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_HYPOT, self, other);
}

Tensor wrap_hypot_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::hypot_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_HYPOT_, self, other);
}

Tensor wrap_igamma_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::igamma(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_IGAMMA_OUT, out, self, other);
}

Tensor wrap_igamma(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::igamma(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_IGAMMA, self, other);
}

Tensor wrap_igamma_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::igamma_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_IGAMMA_, self, other);
}

Tensor wrap_igammac_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::igammac(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_IGAMMAC_OUT, out, self, other);
}

Tensor wrap_igammac(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::igammac(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_IGAMMAC, self, other);
}

Tensor wrap_igammac_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::igammac_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_IGAMMAC_, self, other);
}

Tensor wrap_nextafter_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::nextafter(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NEXTAFTER_OUT, out, self, other);
}

Tensor wrap_nextafter(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::nextafter(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NEXTAFTER, self, other);
}

Tensor wrap_nextafter_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::nextafter_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NEXTAFTER_, self, other);
}

Tensor wrap_remainder_Scalar_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::remainder(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_REMAINDER_SCALAR_OUT, out, self, other);
}

Tensor wrap_remainder_Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::remainder(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_REMAINDER_SCALAR, self, other);
}

Tensor wrap_remainder_Tensor_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::remainder(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_REMAINDER_TENSOR_OUT, out, self, other);
}

Tensor wrap_remainder_Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::remainder(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_REMAINDER_TENSOR, self, other);
}

Tensor wrap_min(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::min(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MIN, self);
}

Tensor wrap_fmin(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::fmin(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FMIN, self, other);
}

Tensor wrap_fmin_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::fmin(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FMIN_OUT, out, self, other);
}

Tensor wrap_max(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::max(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MAX, self);
}

Tensor wrap_fmax(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::fmax(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FMAX, self, other);
}

Tensor wrap_fmax_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::fmax(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FMAX_OUT, out, self, other);
}

Tensor wrap_maximum(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::maximum(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MAXIMUM, self, other);
}

Tensor wrap_maximum_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::maximum(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_MAXIMUM_OUT, out, self, other);
}

Tensor wrap_max_other(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::max(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MAX_OTHER, self, other);
}

Tensor wrap_max_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::max(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_MAX_OUT, out, self, other);
}

Tensor wrap_minimum(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::minimum(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MINIMUM, self, other);
}

Tensor wrap_minimum_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::minimum(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_MINIMUM_OUT, out, self, other);
}

Tensor wrap_min_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::min(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_MIN_OUT, out, self, other);
}

Tensor wrap_min_other(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::min(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MIN_OTHER, self, other);
}

Tensor wrap_quantile_scalar_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::quantile(out, self, q, dim, keepdim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_QUANTILE_SCALAR_OUT, out, self, q, dim, keepdim);
}

Tensor wrap_quantile_scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::quantile(self, q, dim, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_QUANTILE_SCALAR, self, q, dim, keepdim);
}

Tensor wrap_quantile_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, q);
    return at::redispatch::quantile(out, self, q, dim, keepdim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_QUANTILE_OUT, out, self, q, dim, keepdim);
}

Tensor wrap_quantile(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, q);
    return at::redispatch::quantile(self, q, dim, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_QUANTILE, self, q, dim, keepdim);
}

Tensor wrap_nanquantile_scalar_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::nanquantile(out, self, q, dim, keepdim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NANQUANTILE_SCALAR_OUT, out, self, q, dim, keepdim);
}

Tensor wrap_nanquantile_scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::nanquantile(self, q, dim, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NANQUANTILE_SCALAR, self, q, dim, keepdim);
}

Tensor wrap_nanquantile_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, q);
    return at::redispatch::nanquantile(out, self, q, dim, keepdim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NANQUANTILE_OUT, out, self, q, dim, keepdim);
}

Tensor wrap_nanquantile(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, q);
    return at::redispatch::nanquantile(self, q, dim, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NANQUANTILE, self, q, dim, keepdim);
}

Tensor wrap_quantile_new_scalar_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::quantile(out, self, q, dim, keepdim, interpolation);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_QUANTILE_NEW_SCALAR_OUT, out, self, q, dim, keepdim, interpolation);
}

Tensor wrap_quantile_new_scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::quantile(self, q, dim, keepdim, interpolation);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_QUANTILE_NEW_SCALAR, self, q, dim, keepdim, interpolation);
}

Tensor wrap_quantile_new_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, q);
    return at::redispatch::quantile(out, self, q, dim, keepdim, interpolation);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_QUANTILE_NEW_OUT, out, self, q, dim, keepdim, interpolation);
}

Tensor wrap_quantile_new(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, q);
    return at::redispatch::quantile(self, q, dim, keepdim, interpolation);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_QUANTILE_NEW, self, q, dim, keepdim, interpolation);
}

Tensor wrap_nanquantile_new_scalar_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::nanquantile(out, self, q, dim, keepdim, interpolation);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NANQUANTILE_NEW_SCALAR_OUT, out, self, q, dim, keepdim, interpolation);
}

Tensor wrap_nanquantile_new_scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::nanquantile(self, q, dim, keepdim, interpolation);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NANQUANTILE_NEW_SCALAR, self, q, dim, keepdim, interpolation);
}

Tensor wrap_nanquantile_new_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, q);
    return at::redispatch::nanquantile(out, self, q, dim, keepdim, interpolation);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NANQUANTILE_NEW_OUT, out, self, q, dim, keepdim, interpolation);
}

Tensor wrap_nanquantile_new(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, q);
    return at::redispatch::nanquantile(self, q, dim, keepdim, interpolation);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NANQUANTILE_NEW, self, q, dim, keepdim, interpolation);
}

Tensor wrap_sort_values(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(values, indices, self);
    return at::redispatch::sort(values, indices, self, dim, descending);
  }
  return MK_TORCHY(values.dtype(), values.device(), H_SORT_VALUES, values, indices, self, dim, descending);
}

Tensor wrap_sort_values_stable(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(values, indices, self);
    return at::redispatch::sort(values, indices, self, stable, dim, descending);
  }
  return MK_TORCHY(values.dtype(), values.device(), H_SORT_VALUES_STABLE, values, indices, self, stable, dim, descending);
}

Tensor wrap_sort(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::sort(self, dim, descending);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SORT, self, dim, descending);
}

Tensor wrap_sort_stable(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::sort(self, stable, dim, descending);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SORT_STABLE, self, stable, dim, descending);
}

Tensor wrap_sort_dimname_values(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(values, indices, self);
    return at::redispatch::sort(values, indices, self, dim, descending);
  }
  return MK_TORCHY(values.dtype(), values.device(), H_SORT_DIMNAME_VALUES, values, indices, self, dim, descending);
}

Tensor wrap_sort_dimname_values_stable(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(values, indices, self);
    return at::redispatch::sort(values, indices, self, stable, dim, descending);
  }
  return MK_TORCHY(values.dtype(), values.device(), H_SORT_DIMNAME_VALUES_STABLE, values, indices, self, stable, dim, descending);
}

Tensor wrap_sort_dimname(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::sort(self, dim, descending);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SORT_DIMNAME, self, dim, descending);
}

Tensor wrap_sort_dimname_stable(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::sort(self, stable, dim, descending);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SORT_DIMNAME_STABLE, self, stable, dim, descending);
}

Tensor wrap_msort_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::msort(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_MSORT_OUT, out, self);
}

Tensor wrap_msort(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::msort(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MSORT, self);
}

Tensor wrap_argsort(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::argsort(self, dim, descending);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ARGSORT, self, dim, descending);
}

Tensor wrap_argsort_dimname(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::argsort(self, dim, descending);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ARGSORT_DIMNAME, self, dim, descending);
}

Tensor wrap_topk_values(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(values, indices, self);
    return at::redispatch::topk(values, indices, self, k, dim, largest, sorted);
  }
  return MK_TORCHY(values.dtype(), values.device(), H_TOPK_VALUES, values, indices, self, k, dim, largest, sorted);
}

Tensor wrap_topk(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::topk(self, k, dim, largest, sorted);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TOPK, self, k, dim, largest, sorted);
}

Tensor wrap_all(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::all(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ALL, self);
}

Tensor wrap_any(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::any(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ANY, self);
}

Tensor wrap_renorm_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::renorm(out, self, p, dim, maxnorm);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_RENORM_OUT, out, self, p, dim, maxnorm);
}

Tensor wrap_renorm(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::renorm(self, p, dim, maxnorm);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RENORM, self, p, dim, maxnorm);
}

Tensor wrap_unfold(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::unfold(self, dimension, size, step);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_UNFOLD, self, dimension, size, step);
}

Tensor wrap_unfold_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_in);
    return at::redispatch::unfold_backward(grad_in, input_sizes, dim, size, step);
  }
  return MK_TORCHY(grad_in.dtype(), grad_in.device(), H_UNFOLD_BACKWARD, grad_in, input_sizes, dim, size, step);
}

bool wrap_equal(args...) {
  ensure_materialized(self, other);
  return at::redispatch::equal(self, other);
}

Tensor wrap_pow_Tensor_Tensor_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, exponent);
    return at::redispatch::pow(out, self, exponent);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_POW_TENSOR_TENSOR_OUT, out, self, exponent);
}

Tensor wrap_pow_Scalar_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, exponent);
    return at::redispatch::pow(out, self, exponent);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_POW_SCALAR_OUT, out, self, exponent);
}

Tensor wrap_pow_Tensor_Scalar_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::pow(out, self, exponent);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_POW_TENSOR_SCALAR_OUT, out, self, exponent);
}

Tensor wrap_pow_Tensor_Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::pow(self, exponent);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_POW_TENSOR_SCALAR, self, exponent);
}

Tensor wrap_float_power_Tensor_Tensor_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, exponent);
    return at::redispatch::float_power(out, self, exponent);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FLOAT_POWER_TENSOR_TENSOR_OUT, out, self, exponent);
}

Tensor wrap_float_power_Tensor_Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, exponent);
    return at::redispatch::float_power(self, exponent);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FLOAT_POWER_TENSOR_TENSOR, self, exponent);
}

Tensor wrap_float_power_Scalar_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, exponent);
    return at::redispatch::float_power(out, self, exponent);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FLOAT_POWER_SCALAR_OUT, out, self, exponent);
}

Tensor wrap_float_power_Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(exponent);
    return at::redispatch::float_power(self, exponent);
  }
  return MK_TORCHY(exponent.dtype(), exponent.device(), H_FLOAT_POWER_SCALAR, self, exponent);
}

Tensor wrap_float_power_Tensor_Scalar_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::float_power(out, self, exponent);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FLOAT_POWER_TENSOR_SCALAR_OUT, out, self, exponent);
}

Tensor wrap_float_power_Tensor_Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::float_power(self, exponent);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FLOAT_POWER_TENSOR_SCALAR, self, exponent);
}

Tensor wrap_float_power__Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::float_power_(self, exponent);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FLOAT_POWER__SCALAR, self, exponent);
}

Tensor wrap_float_power__Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, exponent);
    return at::redispatch::float_power_(self, exponent);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FLOAT_POWER__TENSOR, self, exponent);
}

Tensor wrap_normal_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::normal_(self, mean, std, generator);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NORMAL_, self, mean, std, generator);
}

Tensor wrap_normal_Tensor_float_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, mean);
    return at::redispatch::normal(out, mean, std, generator);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NORMAL_TENSOR_FLOAT_OUT, out, mean, std, generator);
}

Tensor wrap_normal_Tensor_float(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(mean);
    return at::redispatch::normal(mean, std, generator);
  }
  return MK_TORCHY(mean.dtype(), mean.device(), H_NORMAL_TENSOR_FLOAT, mean, std, generator);
}

Tensor wrap_normal_float_Tensor_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, std);
    return at::redispatch::normal(out, mean, std, generator);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NORMAL_FLOAT_TENSOR_OUT, out, mean, std, generator);
}

Tensor wrap_normal_float_Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(std);
    return at::redispatch::normal(mean, std, generator);
  }
  return MK_TORCHY(std.dtype(), std.device(), H_NORMAL_FLOAT_TENSOR, mean, std, generator);
}

Tensor wrap_normal_Tensor_Tensor_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, mean, std);
    return at::redispatch::normal(out, mean, std, generator);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NORMAL_TENSOR_TENSOR_OUT, out, mean, std, generator);
}

Tensor wrap_normal_Tensor_Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(mean, std);
    return at::redispatch::normal(mean, std, generator);
  }
  return MK_TORCHY(mean.dtype(), mean.device(), H_NORMAL_TENSOR_TENSOR, mean, std, generator);
}

Tensor wrap_normal_float_float(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::normal(mean, std, size, generator, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_NORMAL_FLOAT_FLOAT, mean, std, size, generator, dtype, layout, device, pin_memory);
}

Tensor wrap_normal_float_float_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::normal(out, mean, std, size, generator);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NORMAL_FLOAT_FLOAT_OUT, out, mean, std, size, generator);
}

Tensor wrap_alias(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::alias(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ALIAS, self);
}

Tensor wrap__index_copy_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index, source);
    return at::redispatch::_index_copy_(self, dim, index, source);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__INDEX_COPY_, self, dim, index, source);
}

Tensor wrap__cumsum(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_cumsum(self, dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__CUMSUM, self, dim);
}

Tensor wrap__cumsum_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::_cumsum(out, self, dim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H__CUMSUM_OUT, out, self, dim);
}

Tensor wrap__cumprod(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_cumprod(self, dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__CUMPROD, self, dim);
}

Tensor wrap__cumprod_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::_cumprod(out, self, dim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H__CUMPROD_OUT, out, self, dim);
}

Tensor wrap__var(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_var(self, unbiased);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__VAR, self, unbiased);
}

Tensor wrap__std(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_std(self, unbiased);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__STD, self, unbiased);
}

void wrap__amp_foreach_non_finite_check_and_unscale_(args...) {
  ensure_materialized(found_inf, inv_scale);
  return at::redispatch::_amp_foreach_non_finite_check_and_unscale_(self, found_inf, inv_scale);
}

Tensor wrap__amp_update_scale(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(growth_tracker, current_scale, found_inf);
    return at::redispatch::_amp_update_scale(growth_tracker, current_scale, found_inf, scale_growth_factor, scale_backoff_factor, growth_interval);
  }
  return MK_TORCHY(growth_tracker.dtype(), growth_tracker.device(), H__AMP_UPDATE_SCALE, growth_tracker, current_scale, found_inf, scale_growth_factor, scale_backoff_factor, growth_interval);
}

Tensor wrap__cat(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::_cat(tensors, dim);
  }
  return MK_TORCHY(None, None, H__CAT, tensors, dim);
}

Tensor wrap__cat_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::_cat(out, tensors, dim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H__CAT_OUT, out, tensors, dim);
}

Tensor[] wrap__foreach_add_Scalar(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_add(tensors, scalar);
}

void wrap__foreach_add__Scalar(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_add_(self, scalar);
}

Tensor[] wrap__foreach_sub_Scalar(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_sub(tensors, scalar);
}

void wrap__foreach_sub__Scalar(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_sub_(self, scalar);
}

Tensor[] wrap__foreach_mul_Scalar(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_mul(tensors, scalar);
}

void wrap__foreach_mul__Scalar(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_mul_(self, scalar);
}

Tensor[] wrap__foreach_div_Scalar(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_div(tensors, scalar);
}

void wrap__foreach_div__Scalar(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_div_(self, scalar);
}

Tensor[] wrap__foreach_add_List(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_add(tensors1, tensors2, alpha);
}

void wrap__foreach_add__List(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_add_(self, other, alpha);
}

Tensor[] wrap__foreach_sub_List(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_sub(tensors1, tensors2, alpha);
}

void wrap__foreach_sub__List(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_sub_(self, other, alpha);
}

Tensor[] wrap__foreach_mul_List(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_mul(tensors1, tensors2);
}

void wrap__foreach_mul__List(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_mul_(self, other);
}

Tensor[] wrap__foreach_div_List(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_div(tensors1, tensors2);
}

void wrap__foreach_div__List(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_div_(self, other);
}

Tensor[] wrap__foreach_add_ScalarList(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_add(tensors, scalars);
}

void wrap__foreach_add__ScalarList(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_add_(self, scalars);
}

Tensor[] wrap__foreach_sub_ScalarList(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_sub(tensors, scalars);
}

void wrap__foreach_sub__ScalarList(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_sub_(self, scalars);
}

Tensor[] wrap__foreach_div_ScalarList(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_div(tensors, scalars);
}

void wrap__foreach_div__ScalarList(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_div_(self, scalars);
}

Tensor[] wrap__foreach_mul_ScalarList(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_mul(tensors, scalars);
}

void wrap__foreach_mul__ScalarList(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_mul_(self, scalars);
}

Tensor[] wrap__foreach_exp(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_exp(tensors);
}

void wrap__foreach_zero_(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_zero_(self);
}

void wrap__foreach_exp_(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_exp_(self);
}

Tensor[] wrap__foreach_sqrt(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_sqrt(tensors);
}

void wrap__foreach_sqrt_(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_sqrt_(self);
}

Tensor[] wrap__foreach_abs(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_abs(tensors);
}

void wrap__foreach_abs_(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_abs_(self);
}

Tensor[] wrap__foreach_acos(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_acos(tensors);
}

void wrap__foreach_acos_(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_acos_(self);
}

Tensor[] wrap__foreach_asin(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_asin(tensors);
}

void wrap__foreach_asin_(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_asin_(self);
}

Tensor[] wrap__foreach_atan(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_atan(tensors);
}

void wrap__foreach_atan_(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_atan_(self);
}

Tensor[] wrap__foreach_ceil(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_ceil(tensors);
}

void wrap__foreach_ceil_(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_ceil_(self);
}

Tensor[] wrap__foreach_cos(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_cos(tensors);
}

void wrap__foreach_cos_(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_cos_(self);
}

Tensor[] wrap__foreach_cosh(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_cosh(tensors);
}

void wrap__foreach_cosh_(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_cosh_(self);
}

Tensor[] wrap__foreach_erf(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_erf(tensors);
}

void wrap__foreach_erf_(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_erf_(self);
}

Tensor[] wrap__foreach_erfc(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_erfc(tensors);
}

void wrap__foreach_erfc_(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_erfc_(self);
}

Tensor[] wrap__foreach_expm1(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_expm1(tensors);
}

void wrap__foreach_expm1_(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_expm1_(self);
}

Tensor[] wrap__foreach_floor(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_floor(tensors);
}

void wrap__foreach_floor_(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_floor_(self);
}

Tensor[] wrap__foreach_log(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_log(tensors);
}

void wrap__foreach_log_(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_log_(self);
}

Tensor[] wrap__foreach_log10(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_log10(tensors);
}

void wrap__foreach_log10_(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_log10_(self);
}

Tensor[] wrap__foreach_log1p(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_log1p(tensors);
}

void wrap__foreach_log1p_(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_log1p_(self);
}

Tensor[] wrap__foreach_log2(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_log2(tensors);
}

void wrap__foreach_log2_(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_log2_(self);
}

Tensor[] wrap__foreach_neg(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_neg(tensors);
}

void wrap__foreach_neg_(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_neg_(self);
}

Tensor[] wrap__foreach_tan(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_tan(tensors);
}

void wrap__foreach_tan_(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_tan_(self);
}

Tensor[] wrap__foreach_tanh(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_tanh(tensors);
}

void wrap__foreach_tanh_(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_tanh_(self);
}

Tensor[] wrap__foreach_sin(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_sin(tensors);
}

void wrap__foreach_sin_(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_sin_(self);
}

Tensor[] wrap__foreach_sinh(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_sinh(tensors);
}

void wrap__foreach_sinh_(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_sinh_(self);
}

Tensor[] wrap__foreach_round(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_round(tensors);
}

void wrap__foreach_round_(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_round_(self);
}

Tensor[] wrap__foreach_lgamma(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_lgamma(tensors);
}

void wrap__foreach_lgamma_(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_lgamma_(self);
}

Tensor[] wrap__foreach_frac(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_frac(tensors);
}

void wrap__foreach_frac_(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_frac_(self);
}

Tensor[] wrap__foreach_reciprocal(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_reciprocal(tensors);
}

void wrap__foreach_reciprocal_(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_reciprocal_(self);
}

Tensor[] wrap__foreach_sigmoid(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_sigmoid(tensors);
}

void wrap__foreach_sigmoid_(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_sigmoid_(self);
}

Tensor[] wrap__foreach_trunc(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_trunc(tensors);
}

void wrap__foreach_trunc_(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_trunc_(self);
}

void wrap__foreach_addcdiv__Scalar(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_addcdiv_(self, tensor1, tensor2, value);
}

void wrap__foreach_addcmul__Scalar(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_addcmul_(self, tensor1, tensor2, value);
}

void wrap__foreach_addcdiv__ScalarList(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_addcdiv_(self, tensor1, tensor2, scalars);
}

void wrap__foreach_addcmul__ScalarList(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_addcmul_(self, tensor1, tensor2, scalars);
}

Tensor[] wrap__foreach_addcdiv_Scalar(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_addcdiv(input, tensor1, tensor2, value);
}

Tensor[] wrap__foreach_addcmul_Scalar(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_addcmul(input, tensor1, tensor2, value);
}

Tensor[] wrap__foreach_addcdiv_ScalarList(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_addcdiv(input, tensor1, tensor2, scalars);
}

Tensor[] wrap__foreach_addcmul_ScalarList(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_addcmul(input, tensor1, tensor2, scalars);
}

Tensor[] wrap__foreach_maximum_List(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_maximum(tensors1, tensors2);
}

Tensor[] wrap__foreach_minimum_List(args...) {
  ensure_materialized();
  return at::redispatch::_foreach_minimum(tensors1, tensors2);
}

Tensor wrap_bucketize_Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, boundaries);
    return at::redispatch::bucketize(self, boundaries, out_int32, right);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BUCKETIZE_TENSOR, self, boundaries, out_int32, right);
}

Tensor wrap_bucketize_Tensor_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, boundaries);
    return at::redispatch::bucketize(out, self, boundaries, out_int32, right);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_BUCKETIZE_TENSOR_OUT, out, self, boundaries, out_int32, right);
}

Tensor wrap_bucketize_Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(boundaries);
    return at::redispatch::bucketize(self, boundaries, out_int32, right);
  }
  return MK_TORCHY(boundaries.dtype(), boundaries.device(), H_BUCKETIZE_SCALAR, self, boundaries, out_int32, right);
}

Tensor wrap_searchsorted_Tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(sorted_sequence, self);
    return at::redispatch::searchsorted(sorted_sequence, self, out_int32, right);
  }
  return MK_TORCHY(sorted_sequence.dtype(), sorted_sequence.device(), H_SEARCHSORTED_TENSOR, sorted_sequence, self, out_int32, right);
}

Tensor wrap_searchsorted_Tensor_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, sorted_sequence, self);
    return at::redispatch::searchsorted(out, sorted_sequence, self, out_int32, right);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SEARCHSORTED_TENSOR_OUT, out, sorted_sequence, self, out_int32, right);
}

Tensor wrap_searchsorted_Scalar(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(sorted_sequence);
    return at::redispatch::searchsorted(sorted_sequence, self, out_int32, right);
  }
  return MK_TORCHY(sorted_sequence.dtype(), sorted_sequence.device(), H_SEARCHSORTED_SCALAR, sorted_sequence, self, out_int32, right);
}

Tensor wrap_mse_loss_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, target);
    return at::redispatch::mse_loss(out, self, target, reduction);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_MSE_LOSS_OUT, out, self, target, reduction);
}

Tensor wrap_mse_loss(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, target);
    return at::redispatch::mse_loss(self, target, reduction);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MSE_LOSS, self, target, reduction);
}

Tensor wrap_mse_loss_backward_grad_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self, target);
    return at::redispatch::mse_loss_backward(grad_input, grad_output, self, target, reduction);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_MSE_LOSS_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, target, reduction);
}

Tensor wrap_mse_loss_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, target);
    return at::redispatch::mse_loss_backward(grad_output, self, target, reduction);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_MSE_LOSS_BACKWARD, grad_output, self, target, reduction);
}

Tensor wrap_l1_loss_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, target);
    return at::redispatch::l1_loss(out, self, target, reduction);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_L1_LOSS_OUT, out, self, target, reduction);
}

Tensor wrap_l1_loss(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, target);
    return at::redispatch::l1_loss(self, target, reduction);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_L1_LOSS, self, target, reduction);
}

Tensor wrap_l1_loss_backward_grad_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self, target);
    return at::redispatch::l1_loss_backward(grad_input, grad_output, self, target, reduction);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_L1_LOSS_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, target, reduction);
}

Tensor wrap_l1_loss_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, target);
    return at::redispatch::l1_loss_backward(grad_output, self, target, reduction);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_L1_LOSS_BACKWARD, grad_output, self, target, reduction);
}

Tensor wrap_multi_margin_loss_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, target);
    return at::redispatch::multi_margin_loss(out, self, target, p, margin, weight, reduction);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_MULTI_MARGIN_LOSS_OUT, out, self, target, p, margin, weight, reduction);
}

Tensor wrap_multi_margin_loss(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, target);
    return at::redispatch::multi_margin_loss(self, target, p, margin, weight, reduction);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MULTI_MARGIN_LOSS, self, target, p, margin, weight, reduction);
}

Tensor wrap_multi_margin_loss_backward_grad_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self, target);
    return at::redispatch::multi_margin_loss_backward(grad_input, grad_output, self, target, p, margin, weight, reduction);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_MULTI_MARGIN_LOSS_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, target, p, margin, weight, reduction);
}

Tensor wrap_multi_margin_loss_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, target);
    return at::redispatch::multi_margin_loss_backward(grad_output, self, target, p, margin, weight, reduction);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_MULTI_MARGIN_LOSS_BACKWARD, grad_output, self, target, p, margin, weight, reduction);
}

Tensor wrap_multilabel_margin_loss_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, target);
    return at::redispatch::multilabel_margin_loss(out, self, target, reduction);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_MULTILABEL_MARGIN_LOSS_OUT, out, self, target, reduction);
}

Tensor wrap_multilabel_margin_loss(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, target);
    return at::redispatch::multilabel_margin_loss(self, target, reduction);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MULTILABEL_MARGIN_LOSS, self, target, reduction);
}

Tensor wrap_multilabel_margin_loss_forward_output(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(output, is_target, self, target);
    return at::redispatch::multilabel_margin_loss_forward(output, is_target, self, target, reduction);
  }
  return MK_TORCHY(output.dtype(), output.device(), H_MULTILABEL_MARGIN_LOSS_FORWARD_OUTPUT, output, is_target, self, target, reduction);
}

Tensor wrap_multilabel_margin_loss_forward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, target);
    return at::redispatch::multilabel_margin_loss_forward(self, target, reduction);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MULTILABEL_MARGIN_LOSS_FORWARD, self, target, reduction);
}

Tensor wrap_multilabel_margin_loss_backward_grad_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self, target, is_target);
    return at::redispatch::multilabel_margin_loss_backward(grad_input, grad_output, self, target, reduction, is_target);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_MULTILABEL_MARGIN_LOSS_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, target, reduction, is_target);
}

Tensor wrap_multilabel_margin_loss_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, target, is_target);
    return at::redispatch::multilabel_margin_loss_backward(grad_output, self, target, reduction, is_target);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_MULTILABEL_MARGIN_LOSS_BACKWARD, grad_output, self, target, reduction, is_target);
}

Tensor wrap_nll_loss_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, target);
    return at::redispatch::nll_loss(out, self, target, weight, reduction, ignore_index);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NLL_LOSS_OUT, out, self, target, weight, reduction, ignore_index);
}

Tensor wrap_nll_loss_nd(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, target);
    return at::redispatch::nll_loss_nd(self, target, weight, reduction, ignore_index);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NLL_LOSS_ND, self, target, weight, reduction, ignore_index);
}

Tensor wrap_nll_loss(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, target);
    return at::redispatch::nll_loss(self, target, weight, reduction, ignore_index);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NLL_LOSS, self, target, weight, reduction, ignore_index);
}

Tensor wrap_nll_loss_forward_output(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(output, total_weight, self, target);
    return at::redispatch::nll_loss_forward(output, total_weight, self, target, weight, reduction, ignore_index);
  }
  return MK_TORCHY(output.dtype(), output.device(), H_NLL_LOSS_FORWARD_OUTPUT, output, total_weight, self, target, weight, reduction, ignore_index);
}

Tensor wrap_nll_loss_forward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, target);
    return at::redispatch::nll_loss_forward(self, target, weight, reduction, ignore_index);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NLL_LOSS_FORWARD, self, target, weight, reduction, ignore_index);
}

Tensor wrap_nll_loss_backward_grad_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self, target, total_weight);
    return at::redispatch::nll_loss_backward(grad_input, grad_output, self, target, weight, reduction, ignore_index, total_weight);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_NLL_LOSS_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, target, weight, reduction, ignore_index, total_weight);
}

Tensor wrap_nll_loss_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, target, total_weight);
    return at::redispatch::nll_loss_backward(grad_output, self, target, weight, reduction, ignore_index, total_weight);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_NLL_LOSS_BACKWARD, grad_output, self, target, weight, reduction, ignore_index, total_weight);
}

Tensor wrap_nll_loss2d_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, target);
    return at::redispatch::nll_loss2d(out, self, target, weight, reduction, ignore_index);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NLL_LOSS2D_OUT, out, self, target, weight, reduction, ignore_index);
}

Tensor wrap_nll_loss2d(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, target);
    return at::redispatch::nll_loss2d(self, target, weight, reduction, ignore_index);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NLL_LOSS2D, self, target, weight, reduction, ignore_index);
}

Tensor wrap_nll_loss2d_forward_output(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(output, total_weight, self, target);
    return at::redispatch::nll_loss2d_forward(output, total_weight, self, target, weight, reduction, ignore_index);
  }
  return MK_TORCHY(output.dtype(), output.device(), H_NLL_LOSS2D_FORWARD_OUTPUT, output, total_weight, self, target, weight, reduction, ignore_index);
}

Tensor wrap_nll_loss2d_forward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, target);
    return at::redispatch::nll_loss2d_forward(self, target, weight, reduction, ignore_index);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NLL_LOSS2D_FORWARD, self, target, weight, reduction, ignore_index);
}

Tensor wrap_nll_loss2d_backward_grad_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self, target, total_weight);
    return at::redispatch::nll_loss2d_backward(grad_input, grad_output, self, target, weight, reduction, ignore_index, total_weight);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_NLL_LOSS2D_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, target, weight, reduction, ignore_index, total_weight);
}

Tensor wrap_nll_loss2d_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, target, total_weight);
    return at::redispatch::nll_loss2d_backward(grad_output, self, target, weight, reduction, ignore_index, total_weight);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_NLL_LOSS2D_BACKWARD, grad_output, self, target, weight, reduction, ignore_index, total_weight);
}

Tensor wrap_smooth_l1_loss_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, target);
    return at::redispatch::smooth_l1_loss(out, self, target, reduction, beta);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SMOOTH_L1_LOSS_OUT, out, self, target, reduction, beta);
}

Tensor wrap_smooth_l1_loss(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, target);
    return at::redispatch::smooth_l1_loss(self, target, reduction, beta);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SMOOTH_L1_LOSS, self, target, reduction, beta);
}

Tensor wrap_smooth_l1_loss_backward_grad_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self, target);
    return at::redispatch::smooth_l1_loss_backward(grad_input, grad_output, self, target, reduction, beta);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_SMOOTH_L1_LOSS_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, target, reduction, beta);
}

Tensor wrap_smooth_l1_loss_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, target);
    return at::redispatch::smooth_l1_loss_backward(grad_output, self, target, reduction, beta);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_SMOOTH_L1_LOSS_BACKWARD, grad_output, self, target, reduction, beta);
}

Tensor wrap_huber_loss_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, target);
    return at::redispatch::huber_loss(out, self, target, reduction, delta);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_HUBER_LOSS_OUT, out, self, target, reduction, delta);
}

Tensor wrap_huber_loss(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, target);
    return at::redispatch::huber_loss(self, target, reduction, delta);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_HUBER_LOSS, self, target, reduction, delta);
}

Tensor wrap_huber_loss_backward_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self, target);
    return at::redispatch::huber_loss_backward(grad_input, grad_output, self, target, reduction, delta);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_HUBER_LOSS_BACKWARD_OUT, grad_input, grad_output, self, target, reduction, delta);
}

Tensor wrap_huber_loss_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, target);
    return at::redispatch::huber_loss_backward(grad_output, self, target, reduction, delta);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_HUBER_LOSS_BACKWARD, grad_output, self, target, reduction, delta);
}

Tensor wrap_soft_margin_loss_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, target);
    return at::redispatch::soft_margin_loss(out, self, target, reduction);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SOFT_MARGIN_LOSS_OUT, out, self, target, reduction);
}

Tensor wrap_soft_margin_loss(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, target);
    return at::redispatch::soft_margin_loss(self, target, reduction);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SOFT_MARGIN_LOSS, self, target, reduction);
}

Tensor wrap_soft_margin_loss_backward_grad_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self, target);
    return at::redispatch::soft_margin_loss_backward(grad_input, grad_output, self, target, reduction);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_SOFT_MARGIN_LOSS_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, target, reduction);
}

Tensor wrap_soft_margin_loss_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, target);
    return at::redispatch::soft_margin_loss_backward(grad_output, self, target, reduction);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_SOFT_MARGIN_LOSS_BACKWARD, grad_output, self, target, reduction);
}

Tensor wrap_elu_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::elu(out, self, alpha, scale, input_scale);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ELU_OUT, out, self, alpha, scale, input_scale);
}

Tensor wrap_elu(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::elu(self, alpha, scale, input_scale);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ELU, self, alpha, scale, input_scale);
}

Tensor wrap_elu_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self_or_result);
    return at::redispatch::elu_backward(grad_output, alpha, scale, input_scale, is_result, self_or_result);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_ELU_BACKWARD, grad_output, alpha, scale, input_scale, is_result, self_or_result);
}

Tensor wrap_elu_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::elu_(self, alpha, scale, input_scale);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ELU_, self, alpha, scale, input_scale);
}

Tensor wrap_glu_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::glu(out, self, dim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_GLU_OUT, out, self, dim);
}

Tensor wrap_glu(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::glu(self, dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_GLU, self, dim);
}

Tensor wrap_glu_backward_grad_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self);
    return at::redispatch::glu_backward(grad_input, grad_output, self, dim);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_GLU_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, dim);
}

Tensor wrap_glu_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self);
    return at::redispatch::glu_backward(grad_output, self, dim);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_GLU_BACKWARD, grad_output, self, dim);
}

Tensor wrap_hardsigmoid_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::hardsigmoid(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_HARDSIGMOID_OUT, out, self);
}

Tensor wrap_hardsigmoid(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::hardsigmoid(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_HARDSIGMOID, self);
}

Tensor wrap_hardsigmoid_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::hardsigmoid_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_HARDSIGMOID_, self);
}

Tensor wrap_hardsigmoid_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self);
    return at::redispatch::hardsigmoid_backward(grad_output, self);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_HARDSIGMOID_BACKWARD, grad_output, self);
}

Tensor wrap_hardtanh_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::hardtanh(out, self, min_val, max_val);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_HARDTANH_OUT, out, self, min_val, max_val);
}

Tensor wrap_hardtanh(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::hardtanh(self, min_val, max_val);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_HARDTANH, self, min_val, max_val);
}

Tensor wrap_hardtanh_backward_grad_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self);
    return at::redispatch::hardtanh_backward(grad_input, grad_output, self, min_val, max_val);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_HARDTANH_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, min_val, max_val);
}

Tensor wrap_hardtanh_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self);
    return at::redispatch::hardtanh_backward(grad_output, self, min_val, max_val);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_HARDTANH_BACKWARD, grad_output, self, min_val, max_val);
}

Tensor wrap_hardtanh_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::hardtanh_(self, min_val, max_val);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_HARDTANH_, self, min_val, max_val);
}

Tensor wrap_hardswish_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::hardswish(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_HARDSWISH_OUT, out, self);
}

Tensor wrap_hardswish(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::hardswish(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_HARDSWISH, self);
}

Tensor wrap_hardswish_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::hardswish_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_HARDSWISH_, self);
}

Tensor wrap_hardswish_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self);
    return at::redispatch::hardswish_backward(grad_output, self);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_HARDSWISH_BACKWARD, grad_output, self);
}

Tensor wrap_leaky_relu_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::leaky_relu(out, self, negative_slope);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LEAKY_RELU_OUT, out, self, negative_slope);
}

Tensor wrap_leaky_relu(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::leaky_relu(self, negative_slope);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LEAKY_RELU, self, negative_slope);
}

Tensor wrap_leaky_relu_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self);
    return at::redispatch::leaky_relu_backward(grad_output, self, negative_slope, self_is_result);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_LEAKY_RELU_BACKWARD, grad_output, self, negative_slope, self_is_result);
}

Tensor wrap_leaky_relu_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::leaky_relu_(self, negative_slope);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LEAKY_RELU_, self, negative_slope);
}

Tensor wrap_log_sigmoid_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::log_sigmoid(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LOG_SIGMOID_OUT, out, self);
}

Tensor wrap_log_sigmoid(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::log_sigmoid(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LOG_SIGMOID, self);
}

Tensor wrap_log_sigmoid_forward_output(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(output, buffer, self);
    return at::redispatch::log_sigmoid_forward(output, buffer, self);
  }
  return MK_TORCHY(output.dtype(), output.device(), H_LOG_SIGMOID_FORWARD_OUTPUT, output, buffer, self);
}

Tensor wrap_log_sigmoid_forward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::log_sigmoid_forward(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LOG_SIGMOID_FORWARD, self);
}

Tensor wrap_log_sigmoid_backward_grad_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self, buffer);
    return at::redispatch::log_sigmoid_backward(grad_input, grad_output, self, buffer);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_LOG_SIGMOID_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, buffer);
}

Tensor wrap_log_sigmoid_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, buffer);
    return at::redispatch::log_sigmoid_backward(grad_output, self, buffer);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_LOG_SIGMOID_BACKWARD, grad_output, self, buffer);
}

Tensor wrap_rrelu_with_noise_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, noise);
    return at::redispatch::rrelu_with_noise(out, self, noise, lower, upper, training, generator);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_RRELU_WITH_NOISE_OUT, out, self, noise, lower, upper, training, generator);
}

Tensor wrap_rrelu_with_noise(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, noise);
    return at::redispatch::rrelu_with_noise(self, noise, lower, upper, training, generator);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RRELU_WITH_NOISE, self, noise, lower, upper, training, generator);
}

Tensor wrap_rrelu_with_noise_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, noise);
    return at::redispatch::rrelu_with_noise_backward(grad_output, self, noise, lower, upper, training, self_is_result);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_RRELU_WITH_NOISE_BACKWARD, grad_output, self, noise, lower, upper, training, self_is_result);
}

Tensor wrap_rrelu_with_noise_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, noise);
    return at::redispatch::rrelu_with_noise_(self, noise, lower, upper, training, generator);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RRELU_WITH_NOISE_, self, noise, lower, upper, training, generator);
}

Tensor wrap_softplus_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::softplus(out, self, beta, threshold);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SOFTPLUS_OUT, out, self, beta, threshold);
}

Tensor wrap_softplus(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::softplus(self, beta, threshold);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SOFTPLUS, self, beta, threshold);
}

Tensor wrap_softplus_backward_grad_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self, output);
    return at::redispatch::softplus_backward(grad_input, grad_output, self, beta, threshold, output);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_SOFTPLUS_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, beta, threshold, output);
}

Tensor wrap_softplus_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, output);
    return at::redispatch::softplus_backward(grad_output, self, beta, threshold, output);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_SOFTPLUS_BACKWARD, grad_output, self, beta, threshold, output);
}

Tensor wrap_softshrink_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::softshrink(out, self, lambd);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SOFTSHRINK_OUT, out, self, lambd);
}

Tensor wrap_softshrink(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::softshrink(self, lambd);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SOFTSHRINK, self, lambd);
}

Tensor wrap_softshrink_backward_grad_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self);
    return at::redispatch::softshrink_backward(grad_input, grad_output, self, lambd);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_SOFTSHRINK_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, lambd);
}

Tensor wrap_softshrink_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self);
    return at::redispatch::softshrink_backward(grad_output, self, lambd);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_SOFTSHRINK_BACKWARD, grad_output, self, lambd);
}

Tensor wrap_adaptive_avg_pool2d_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::adaptive_avg_pool2d(out, self, output_size);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ADAPTIVE_AVG_POOL2D_OUT, out, self, output_size);
}

Tensor wrap_adaptive_avg_pool2d(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::adaptive_avg_pool2d(self, output_size);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ADAPTIVE_AVG_POOL2D, self, output_size);
}

Tensor wrap_mkldnn_adaptive_avg_pool2d(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::mkldnn_adaptive_avg_pool2d(self, output_size);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MKLDNN_ADAPTIVE_AVG_POOL2D, self, output_size);
}

Tensor wrap_mkldnn_adaptive_avg_pool2d_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self);
    return at::redispatch::mkldnn_adaptive_avg_pool2d_backward(grad_output, self);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_MKLDNN_ADAPTIVE_AVG_POOL2D_BACKWARD, grad_output, self);
}

Tensor wrap__adaptive_avg_pool2d(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_adaptive_avg_pool2d(self, output_size);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__ADAPTIVE_AVG_POOL2D, self, output_size);
}

Tensor wrap__adaptive_avg_pool2d_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self);
    return at::redispatch::_adaptive_avg_pool2d_backward(grad_output, self);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H__ADAPTIVE_AVG_POOL2D_BACKWARD, grad_output, self);
}

Tensor wrap_adaptive_avg_pool3d_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::adaptive_avg_pool3d(out, self, output_size);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ADAPTIVE_AVG_POOL3D_OUT, out, self, output_size);
}

Tensor wrap_adaptive_avg_pool3d(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::adaptive_avg_pool3d(self, output_size);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ADAPTIVE_AVG_POOL3D, self, output_size);
}

Tensor wrap__adaptive_avg_pool3d(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_adaptive_avg_pool3d(self, output_size);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__ADAPTIVE_AVG_POOL3D, self, output_size);
}

Tensor wrap_adaptive_avg_pool3d_backward_grad_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self);
    return at::redispatch::adaptive_avg_pool3d_backward(grad_input, grad_output, self);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_ADAPTIVE_AVG_POOL3D_BACKWARD_GRAD_INPUT, grad_input, grad_output, self);
}

Tensor wrap__adaptive_avg_pool3d_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self);
    return at::redispatch::_adaptive_avg_pool3d_backward(grad_output, self);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H__ADAPTIVE_AVG_POOL3D_BACKWARD, grad_output, self);
}

Tensor wrap_adaptive_max_pool2d_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, indices, self);
    return at::redispatch::adaptive_max_pool2d(out, indices, self, output_size);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ADAPTIVE_MAX_POOL2D_OUT, out, indices, self, output_size);
}

Tensor wrap_adaptive_max_pool2d_backward_grad_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self, indices);
    return at::redispatch::adaptive_max_pool2d_backward(grad_input, grad_output, self, indices);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_ADAPTIVE_MAX_POOL2D_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, indices);
}

Tensor wrap_adaptive_max_pool2d_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, indices);
    return at::redispatch::adaptive_max_pool2d_backward(grad_output, self, indices);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_ADAPTIVE_MAX_POOL2D_BACKWARD, grad_output, self, indices);
}

Tensor wrap_adaptive_max_pool3d_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, indices, self);
    return at::redispatch::adaptive_max_pool3d(out, indices, self, output_size);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ADAPTIVE_MAX_POOL3D_OUT, out, indices, self, output_size);
}

Tensor wrap_adaptive_max_pool3d_backward_grad_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self, indices);
    return at::redispatch::adaptive_max_pool3d_backward(grad_input, grad_output, self, indices);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_ADAPTIVE_MAX_POOL3D_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, indices);
}

Tensor wrap_adaptive_max_pool3d_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, indices);
    return at::redispatch::adaptive_max_pool3d_backward(grad_output, self, indices);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_ADAPTIVE_MAX_POOL3D_BACKWARD, grad_output, self, indices);
}

Tensor wrap_avg_pool2d_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::avg_pool2d(out, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_AVG_POOL2D_OUT, out, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}

Tensor wrap_avg_pool2d(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::avg_pool2d(self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_AVG_POOL2D, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}

Tensor wrap_avg_pool2d_backward_grad_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self);
    return at::redispatch::avg_pool2d_backward(grad_input, grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_AVG_POOL2D_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}

Tensor wrap_avg_pool2d_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self);
    return at::redispatch::avg_pool2d_backward(grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_AVG_POOL2D_BACKWARD, grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}

Tensor wrap_avg_pool3d_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::avg_pool3d(out, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_AVG_POOL3D_OUT, out, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}

Tensor wrap_avg_pool3d(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::avg_pool3d(self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_AVG_POOL3D, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}

Tensor wrap_avg_pool3d_backward_grad_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self);
    return at::redispatch::avg_pool3d_backward(grad_input, grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_AVG_POOL3D_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}

Tensor wrap_avg_pool3d_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self);
    return at::redispatch::avg_pool3d_backward(grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_AVG_POOL3D_BACKWARD, grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}

Tensor wrap_fractional_max_pool2d_output(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(output, indices, self, random_samples);
    return at::redispatch::fractional_max_pool2d(output, indices, self, kernel_size, output_size, random_samples);
  }
  return MK_TORCHY(output.dtype(), output.device(), H_FRACTIONAL_MAX_POOL2D_OUTPUT, output, indices, self, kernel_size, output_size, random_samples);
}

Tensor wrap_fractional_max_pool2d_backward_grad_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self, indices);
    return at::redispatch::fractional_max_pool2d_backward(grad_input, grad_output, self, kernel_size, output_size, indices);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_FRACTIONAL_MAX_POOL2D_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, kernel_size, output_size, indices);
}

Tensor wrap_fractional_max_pool2d_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, indices);
    return at::redispatch::fractional_max_pool2d_backward(grad_output, self, kernel_size, output_size, indices);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_FRACTIONAL_MAX_POOL2D_BACKWARD, grad_output, self, kernel_size, output_size, indices);
}

Tensor wrap_fractional_max_pool3d_output(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(output, indices, self, random_samples);
    return at::redispatch::fractional_max_pool3d(output, indices, self, kernel_size, output_size, random_samples);
  }
  return MK_TORCHY(output.dtype(), output.device(), H_FRACTIONAL_MAX_POOL3D_OUTPUT, output, indices, self, kernel_size, output_size, random_samples);
}

Tensor wrap_fractional_max_pool3d(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, random_samples);
    return at::redispatch::fractional_max_pool3d(self, kernel_size, output_size, random_samples);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FRACTIONAL_MAX_POOL3D, self, kernel_size, output_size, random_samples);
}

Tensor wrap_fractional_max_pool3d_backward_grad_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self, indices);
    return at::redispatch::fractional_max_pool3d_backward(grad_input, grad_output, self, kernel_size, output_size, indices);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_FRACTIONAL_MAX_POOL3D_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, kernel_size, output_size, indices);
}

Tensor wrap_fractional_max_pool3d_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, indices);
    return at::redispatch::fractional_max_pool3d_backward(grad_output, self, kernel_size, output_size, indices);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_FRACTIONAL_MAX_POOL3D_BACKWARD, grad_output, self, kernel_size, output_size, indices);
}

Tensor wrap_max_pool2d_with_indices_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, indices, self);
    return at::redispatch::max_pool2d_with_indices(out, indices, self, kernel_size, stride, padding, dilation, ceil_mode);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_MAX_POOL2D_WITH_INDICES_OUT, out, indices, self, kernel_size, stride, padding, dilation, ceil_mode);
}

Tensor wrap_max_pool2d_with_indices(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::max_pool2d_with_indices(self, kernel_size, stride, padding, dilation, ceil_mode);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MAX_POOL2D_WITH_INDICES, self, kernel_size, stride, padding, dilation, ceil_mode);
}

Tensor wrap_max_pool2d_with_indices_backward_grad_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self, indices);
    return at::redispatch::max_pool2d_with_indices_backward(grad_input, grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_MAX_POOL2D_WITH_INDICES_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
}

Tensor wrap_max_pool2d_with_indices_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, indices);
    return at::redispatch::max_pool2d_with_indices_backward(grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_MAX_POOL2D_WITH_INDICES_BACKWARD, grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
}

Tensor wrap_max_pool3d_with_indices_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, indices, self);
    return at::redispatch::max_pool3d_with_indices(out, indices, self, kernel_size, stride, padding, dilation, ceil_mode);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_MAX_POOL3D_WITH_INDICES_OUT, out, indices, self, kernel_size, stride, padding, dilation, ceil_mode);
}

Tensor wrap_max_pool3d_with_indices(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::max_pool3d_with_indices(self, kernel_size, stride, padding, dilation, ceil_mode);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MAX_POOL3D_WITH_INDICES, self, kernel_size, stride, padding, dilation, ceil_mode);
}

Tensor wrap_max_pool3d_with_indices_backward_grad_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self, indices);
    return at::redispatch::max_pool3d_with_indices_backward(grad_input, grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_MAX_POOL3D_WITH_INDICES_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
}

Tensor wrap_max_pool3d_with_indices_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, indices);
    return at::redispatch::max_pool3d_with_indices_backward(grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_MAX_POOL3D_WITH_INDICES_BACKWARD, grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
}

Tensor wrap_max_unpool2d_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, indices);
    return at::redispatch::max_unpool2d(out, self, indices, output_size);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_MAX_UNPOOL2D_OUT, out, self, indices, output_size);
}

Tensor wrap_max_unpool2d(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, indices);
    return at::redispatch::max_unpool2d(self, indices, output_size);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MAX_UNPOOL2D, self, indices, output_size);
}

Tensor wrap_max_unpool2d_backward_grad_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self, indices);
    return at::redispatch::max_unpool2d_backward(grad_input, grad_output, self, indices, output_size);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_MAX_UNPOOL2D_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, indices, output_size);
}

Tensor wrap_max_unpool2d_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, indices);
    return at::redispatch::max_unpool2d_backward(grad_output, self, indices, output_size);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_MAX_UNPOOL2D_BACKWARD, grad_output, self, indices, output_size);
}

Tensor wrap_max_unpool3d_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, indices);
    return at::redispatch::max_unpool3d(out, self, indices, output_size, stride, padding);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_MAX_UNPOOL3D_OUT, out, self, indices, output_size, stride, padding);
}

Tensor wrap_max_unpool3d(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, indices);
    return at::redispatch::max_unpool3d(self, indices, output_size, stride, padding);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MAX_UNPOOL3D, self, indices, output_size, stride, padding);
}

Tensor wrap_max_unpool3d_backward_grad_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self, indices);
    return at::redispatch::max_unpool3d_backward(grad_input, grad_output, self, indices, output_size, stride, padding);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_MAX_UNPOOL3D_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, indices, output_size, stride, padding);
}

Tensor wrap_max_unpool3d_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, indices);
    return at::redispatch::max_unpool3d_backward(grad_output, self, indices, output_size, stride, padding);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_MAX_UNPOOL3D_BACKWARD, grad_output, self, indices, output_size, stride, padding);
}

Tensor wrap_reflection_pad1d_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::reflection_pad1d(out, self, padding);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_REFLECTION_PAD1D_OUT, out, self, padding);
}

Tensor wrap_reflection_pad1d(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::reflection_pad1d(self, padding);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_REFLECTION_PAD1D, self, padding);
}

Tensor wrap_reflection_pad1d_backward_grad_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self);
    return at::redispatch::reflection_pad1d_backward(grad_input, grad_output, self, padding);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_REFLECTION_PAD1D_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, padding);
}

Tensor wrap_reflection_pad1d_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self);
    return at::redispatch::reflection_pad1d_backward(grad_output, self, padding);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_REFLECTION_PAD1D_BACKWARD, grad_output, self, padding);
}

Tensor wrap_reflection_pad2d_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::reflection_pad2d(out, self, padding);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_REFLECTION_PAD2D_OUT, out, self, padding);
}

Tensor wrap_reflection_pad2d(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::reflection_pad2d(self, padding);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_REFLECTION_PAD2D, self, padding);
}

Tensor wrap_reflection_pad2d_backward_grad_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self);
    return at::redispatch::reflection_pad2d_backward(grad_input, grad_output, self, padding);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_REFLECTION_PAD2D_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, padding);
}

Tensor wrap_reflection_pad2d_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self);
    return at::redispatch::reflection_pad2d_backward(grad_output, self, padding);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_REFLECTION_PAD2D_BACKWARD, grad_output, self, padding);
}

Tensor wrap_replication_pad1d_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::replication_pad1d(out, self, padding);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_REPLICATION_PAD1D_OUT, out, self, padding);
}

Tensor wrap_replication_pad1d_backward_grad_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self);
    return at::redispatch::replication_pad1d_backward(grad_input, grad_output, self, padding);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_REPLICATION_PAD1D_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, padding);
}

Tensor wrap_replication_pad2d_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::replication_pad2d(out, self, padding);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_REPLICATION_PAD2D_OUT, out, self, padding);
}

Tensor wrap_replication_pad2d_backward_grad_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self);
    return at::redispatch::replication_pad2d_backward(grad_input, grad_output, self, padding);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_REPLICATION_PAD2D_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, padding);
}

Tensor wrap_replication_pad2d_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self);
    return at::redispatch::replication_pad2d_backward(grad_output, self, padding);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_REPLICATION_PAD2D_BACKWARD, grad_output, self, padding);
}

Tensor wrap_replication_pad3d_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::replication_pad3d(out, self, padding);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_REPLICATION_PAD3D_OUT, out, self, padding);
}

Tensor wrap_replication_pad3d_backward_grad_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self);
    return at::redispatch::replication_pad3d_backward(grad_input, grad_output, self, padding);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_REPLICATION_PAD3D_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, padding);
}

Tensor wrap_replication_pad3d_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self);
    return at::redispatch::replication_pad3d_backward(grad_output, self, padding);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_REPLICATION_PAD3D_BACKWARD, grad_output, self, padding);
}

Tensor wrap_upsample_linear1d_vec(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input);
    return at::redispatch::upsample_linear1d(input, output_size, align_corners, scale_factors);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_UPSAMPLE_LINEAR1D_VEC, input, output_size, align_corners, scale_factors);
}

Tensor wrap_upsample_linear1d_backward_vec(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output);
    return at::redispatch::upsample_linear1d_backward(grad_output, output_size, input_size, align_corners, scale_factors);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_UPSAMPLE_LINEAR1D_BACKWARD_VEC, grad_output, output_size, input_size, align_corners, scale_factors);
}

Tensor wrap_upsample_bilinear2d_vec(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input);
    return at::redispatch::upsample_bilinear2d(input, output_size, align_corners, scale_factors);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_UPSAMPLE_BILINEAR2D_VEC, input, output_size, align_corners, scale_factors);
}

Tensor wrap_upsample_bilinear2d_backward_vec(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output);
    return at::redispatch::upsample_bilinear2d_backward(grad_output, output_size, input_size, align_corners, scale_factors);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_UPSAMPLE_BILINEAR2D_BACKWARD_VEC, grad_output, output_size, input_size, align_corners, scale_factors);
}

Tensor wrap_upsample_trilinear3d_vec(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input);
    return at::redispatch::upsample_trilinear3d(input, output_size, align_corners, scale_factors);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_UPSAMPLE_TRILINEAR3D_VEC, input, output_size, align_corners, scale_factors);
}

Tensor wrap_upsample_trilinear3d_backward_vec(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output);
    return at::redispatch::upsample_trilinear3d_backward(grad_output, output_size, input_size, align_corners, scale_factors);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_UPSAMPLE_TRILINEAR3D_BACKWARD_VEC, grad_output, output_size, input_size, align_corners, scale_factors);
}

Tensor wrap_upsample_bicubic2d_vec(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input);
    return at::redispatch::upsample_bicubic2d(input, output_size, align_corners, scale_factors);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_UPSAMPLE_BICUBIC2D_VEC, input, output_size, align_corners, scale_factors);
}

Tensor wrap_upsample_bicubic2d_backward_vec(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output);
    return at::redispatch::upsample_bicubic2d_backward(grad_output, output_size, input_size, align_corners, scale_factors);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_UPSAMPLE_BICUBIC2D_BACKWARD_VEC, grad_output, output_size, input_size, align_corners, scale_factors);
}

Tensor wrap_upsample_nearest1d_vec(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input);
    return at::redispatch::upsample_nearest1d(input, output_size, scale_factors);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_UPSAMPLE_NEAREST1D_VEC, input, output_size, scale_factors);
}

Tensor wrap_upsample_nearest1d_backward_vec(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output);
    return at::redispatch::upsample_nearest1d_backward(grad_output, output_size, input_size, scale_factors);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_UPSAMPLE_NEAREST1D_BACKWARD_VEC, grad_output, output_size, input_size, scale_factors);
}

Tensor wrap_upsample_nearest2d_vec(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input);
    return at::redispatch::upsample_nearest2d(input, output_size, scale_factors);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_UPSAMPLE_NEAREST2D_VEC, input, output_size, scale_factors);
}

Tensor wrap_upsample_nearest2d_backward_vec(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output);
    return at::redispatch::upsample_nearest2d_backward(grad_output, output_size, input_size, scale_factors);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_UPSAMPLE_NEAREST2D_BACKWARD_VEC, grad_output, output_size, input_size, scale_factors);
}

Tensor wrap_upsample_nearest3d_vec(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input);
    return at::redispatch::upsample_nearest3d(input, output_size, scale_factors);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_UPSAMPLE_NEAREST3D_VEC, input, output_size, scale_factors);
}

Tensor wrap_upsample_nearest3d_backward_vec(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output);
    return at::redispatch::upsample_nearest3d_backward(grad_output, output_size, input_size, scale_factors);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_UPSAMPLE_NEAREST3D_BACKWARD_VEC, grad_output, output_size, input_size, scale_factors);
}

Tensor wrap_upsample_linear1d_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::upsample_linear1d(out, self, output_size, align_corners, scales);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_UPSAMPLE_LINEAR1D_OUT, out, self, output_size, align_corners, scales);
}

Tensor wrap_upsample_linear1d_backward_grad_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output);
    return at::redispatch::upsample_linear1d_backward(grad_input, grad_output, output_size, input_size, align_corners, scales);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_UPSAMPLE_LINEAR1D_BACKWARD_GRAD_INPUT, grad_input, grad_output, output_size, input_size, align_corners, scales);
}

Tensor wrap_upsample_bilinear2d_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::upsample_bilinear2d(out, self, output_size, align_corners, scales_h, scales_w);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_UPSAMPLE_BILINEAR2D_OUT, out, self, output_size, align_corners, scales_h, scales_w);
}

Tensor wrap_upsample_bilinear2d(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::upsample_bilinear2d(self, output_size, align_corners, scales_h, scales_w);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_UPSAMPLE_BILINEAR2D, self, output_size, align_corners, scales_h, scales_w);
}

Tensor wrap_upsample_bilinear2d_backward_grad_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output);
    return at::redispatch::upsample_bilinear2d_backward(grad_input, grad_output, output_size, input_size, align_corners, scales_h, scales_w);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_UPSAMPLE_BILINEAR2D_BACKWARD_GRAD_INPUT, grad_input, grad_output, output_size, input_size, align_corners, scales_h, scales_w);
}

Tensor wrap_upsample_bicubic2d_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::upsample_bicubic2d(out, self, output_size, align_corners, scales_h, scales_w);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_UPSAMPLE_BICUBIC2D_OUT, out, self, output_size, align_corners, scales_h, scales_w);
}

Tensor wrap_upsample_bicubic2d_backward_grad_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output);
    return at::redispatch::upsample_bicubic2d_backward(grad_input, grad_output, output_size, input_size, align_corners, scales_h, scales_w);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_UPSAMPLE_BICUBIC2D_BACKWARD_GRAD_INPUT, grad_input, grad_output, output_size, input_size, align_corners, scales_h, scales_w);
}

Tensor wrap_upsample_trilinear3d_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::upsample_trilinear3d(out, self, output_size, align_corners, scales_d, scales_h, scales_w);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_UPSAMPLE_TRILINEAR3D_OUT, out, self, output_size, align_corners, scales_d, scales_h, scales_w);
}

Tensor wrap_upsample_trilinear3d_backward_grad_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output);
    return at::redispatch::upsample_trilinear3d_backward(grad_input, grad_output, output_size, input_size, align_corners, scales_d, scales_h, scales_w);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_UPSAMPLE_TRILINEAR3D_BACKWARD_GRAD_INPUT, grad_input, grad_output, output_size, input_size, align_corners, scales_d, scales_h, scales_w);
}

Tensor wrap_upsample_nearest1d_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::upsample_nearest1d(out, self, output_size, scales);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_UPSAMPLE_NEAREST1D_OUT, out, self, output_size, scales);
}

Tensor wrap_upsample_nearest1d_backward_grad_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output);
    return at::redispatch::upsample_nearest1d_backward(grad_input, grad_output, output_size, input_size, scales);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_UPSAMPLE_NEAREST1D_BACKWARD_GRAD_INPUT, grad_input, grad_output, output_size, input_size, scales);
}

Tensor wrap_upsample_nearest2d_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::upsample_nearest2d(out, self, output_size, scales_h, scales_w);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_UPSAMPLE_NEAREST2D_OUT, out, self, output_size, scales_h, scales_w);
}

Tensor wrap_upsample_nearest2d(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::upsample_nearest2d(self, output_size, scales_h, scales_w);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_UPSAMPLE_NEAREST2D, self, output_size, scales_h, scales_w);
}

Tensor wrap_upsample_nearest2d_backward_grad_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output);
    return at::redispatch::upsample_nearest2d_backward(grad_input, grad_output, output_size, input_size, scales_h, scales_w);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_UPSAMPLE_NEAREST2D_BACKWARD_GRAD_INPUT, grad_input, grad_output, output_size, input_size, scales_h, scales_w);
}

Tensor wrap_upsample_nearest3d_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::upsample_nearest3d(out, self, output_size, scales_d, scales_h, scales_w);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_UPSAMPLE_NEAREST3D_OUT, out, self, output_size, scales_d, scales_h, scales_w);
}

Tensor wrap_upsample_nearest3d(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::upsample_nearest3d(self, output_size, scales_d, scales_h, scales_w);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_UPSAMPLE_NEAREST3D, self, output_size, scales_d, scales_h, scales_w);
}

Tensor wrap_upsample_nearest3d_backward_grad_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output);
    return at::redispatch::upsample_nearest3d_backward(grad_input, grad_output, output_size, input_size, scales_d, scales_h, scales_w);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_UPSAMPLE_NEAREST3D_BACKWARD_GRAD_INPUT, grad_input, grad_output, output_size, input_size, scales_d, scales_h, scales_w);
}

Tensor wrap_sigmoid_backward_grad_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, output);
    return at::redispatch::sigmoid_backward(grad_input, grad_output, output);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_SIGMOID_BACKWARD_GRAD_INPUT, grad_input, grad_output, output);
}

Tensor wrap_sigmoid_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, output);
    return at::redispatch::sigmoid_backward(grad_output, output);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_SIGMOID_BACKWARD, grad_output, output);
}

Tensor wrap_logit_backward_grad_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self);
    return at::redispatch::logit_backward(grad_input, grad_output, self, eps);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_LOGIT_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, eps);
}

Tensor wrap_logit_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self);
    return at::redispatch::logit_backward(grad_output, self, eps);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_LOGIT_BACKWARD, grad_output, self, eps);
}

Tensor wrap_tanh_backward_grad_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, output);
    return at::redispatch::tanh_backward(grad_input, grad_output, output);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_TANH_BACKWARD_GRAD_INPUT, grad_input, grad_output, output);
}

Tensor wrap_tanh_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, output);
    return at::redispatch::tanh_backward(grad_output, output);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_TANH_BACKWARD, grad_output, output);
}

Tensor wrap_slow_conv_transpose2d_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, weight);
    return at::redispatch::slow_conv_transpose2d(out, self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SLOW_CONV_TRANSPOSE2D_OUT, out, self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}

Tensor wrap_slow_conv_transpose2d(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, weight);
    return at::redispatch::slow_conv_transpose2d(self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SLOW_CONV_TRANSPOSE2D, self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}

Tensor wrap_slow_conv_transpose2d_backward_grad_output(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_weight, grad_bias, grad_output, self, weight, columns, ones);
    return at::redispatch::slow_conv_transpose2d_backward(grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, columns, ones);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_SLOW_CONV_TRANSPOSE2D_BACKWARD_GRAD_OUTPUT, grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, columns, ones);
}

Tensor wrap_slow_conv_transpose2d_backward_output_mask(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, weight, columns, ones);
    return at::redispatch::slow_conv_transpose2d_backward(grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, columns, ones, output_mask);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_SLOW_CONV_TRANSPOSE2D_BACKWARD_OUTPUT_MASK, grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, columns, ones, output_mask);
}

Tensor wrap_slow_conv_transpose3d_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, weight);
    return at::redispatch::slow_conv_transpose3d(out, self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SLOW_CONV_TRANSPOSE3D_OUT, out, self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}

Tensor wrap_slow_conv_transpose3d(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, weight);
    return at::redispatch::slow_conv_transpose3d(self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SLOW_CONV_TRANSPOSE3D, self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}

Tensor wrap_slow_conv_transpose3d_backward_grad_output(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_weight, grad_bias, grad_output, self, weight, finput, fgrad_input);
    return at::redispatch::slow_conv_transpose3d_backward(grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, finput, fgrad_input);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_SLOW_CONV_TRANSPOSE3D_BACKWARD_GRAD_OUTPUT, grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, finput, fgrad_input);
}

Tensor wrap_slow_conv_transpose3d_backward_output_mask(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, weight, finput, fgrad_input);
    return at::redispatch::slow_conv_transpose3d_backward(grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, finput, fgrad_input, output_mask);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_SLOW_CONV_TRANSPOSE3D_BACKWARD_OUTPUT_MASK, grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, finput, fgrad_input, output_mask);
}

Tensor wrap_thnn_conv2d_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, weight);
    return at::redispatch::thnn_conv2d(out, self, weight, kernel_size, bias, stride, padding);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_THNN_CONV2D_OUT, out, self, weight, kernel_size, bias, stride, padding);
}

Tensor wrap_thnn_conv2d(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, weight);
    return at::redispatch::thnn_conv2d(self, weight, kernel_size, bias, stride, padding);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_THNN_CONV2D, self, weight, kernel_size, bias, stride, padding);
}

Tensor wrap_thnn_conv2d_forward_output(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(output, finput, fgrad_input, self, weight);
    return at::redispatch::thnn_conv2d_forward(output, finput, fgrad_input, self, weight, kernel_size, bias, stride, padding);
  }
  return MK_TORCHY(output.dtype(), output.device(), H_THNN_CONV2D_FORWARD_OUTPUT, output, finput, fgrad_input, self, weight, kernel_size, bias, stride, padding);
}

Tensor wrap_thnn_conv2d_forward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, weight);
    return at::redispatch::thnn_conv2d_forward(self, weight, kernel_size, bias, stride, padding);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_THNN_CONV2D_FORWARD, self, weight, kernel_size, bias, stride, padding);
}

Tensor wrap_thnn_conv2d_backward_grad_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_weight, grad_bias, grad_output, self, weight, finput, fgrad_input);
    return at::redispatch::thnn_conv2d_backward(grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_THNN_CONV2D_BACKWARD_GRAD_INPUT, grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input);
}

Tensor wrap_thnn_conv2d_backward_output_mask(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, weight, finput, fgrad_input);
    return at::redispatch::thnn_conv2d_backward(grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input, output_mask);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_THNN_CONV2D_BACKWARD_OUTPUT_MASK, grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input, output_mask);
}

Tensor wrap_thnn_conv_depthwise2d_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, weight);
    return at::redispatch::thnn_conv_depthwise2d(out, self, weight, kernel_size, bias, stride, padding, dilation);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_THNN_CONV_DEPTHWISE2D_OUT, out, self, weight, kernel_size, bias, stride, padding, dilation);
}

Tensor wrap_thnn_conv_depthwise2d(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, weight);
    return at::redispatch::thnn_conv_depthwise2d(self, weight, kernel_size, bias, stride, padding, dilation);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_THNN_CONV_DEPTHWISE2D, self, weight, kernel_size, bias, stride, padding, dilation);
}

Tensor wrap_thnn_conv_depthwise2d_forward_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, weight);
    return at::redispatch::thnn_conv_depthwise2d_forward(out, self, weight, kernel_size, bias, stride, padding, dilation);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_THNN_CONV_DEPTHWISE2D_FORWARD_OUT, out, self, weight, kernel_size, bias, stride, padding, dilation);
}

Tensor wrap_thnn_conv_depthwise2d_forward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, weight);
    return at::redispatch::thnn_conv_depthwise2d_forward(self, weight, kernel_size, bias, stride, padding, dilation);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_THNN_CONV_DEPTHWISE2D_FORWARD, self, weight, kernel_size, bias, stride, padding, dilation);
}

Tensor wrap_thnn_conv_depthwise2d_backward_grad_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_weight, grad_output, self, weight);
    return at::redispatch::thnn_conv_depthwise2d_backward(grad_input, grad_weight, grad_output, self, weight, kernel_size, stride, padding, dilation);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_THNN_CONV_DEPTHWISE2D_BACKWARD_GRAD_INPUT, grad_input, grad_weight, grad_output, self, weight, kernel_size, stride, padding, dilation);
}

Tensor wrap_thnn_conv_depthwise2d_backward_output_mask(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, weight);
    return at::redispatch::thnn_conv_depthwise2d_backward(grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_THNN_CONV_DEPTHWISE2D_BACKWARD_OUTPUT_MASK, grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
}

Tensor wrap_conv_depthwise3d(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, weight);
    return at::redispatch::conv_depthwise3d(self, weight, kernel_size, bias, stride, padding, dilation);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CONV_DEPTHWISE3D, self, weight, kernel_size, bias, stride, padding, dilation);
}

Tensor wrap_conv_depthwise3d_backward_grad_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_weight, grad_bias, grad_output, self, weight);
    return at::redispatch::conv_depthwise3d_backward(grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, dilation);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_CONV_DEPTHWISE3D_BACKWARD_GRAD_INPUT, grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, dilation);
}

Tensor wrap_conv_depthwise3d_backward_output_mask(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, weight);
    return at::redispatch::conv_depthwise3d_backward(grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_CONV_DEPTHWISE3D_BACKWARD_OUTPUT_MASK, grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
}

Tensor wrap_slow_conv3d_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, weight);
    return at::redispatch::slow_conv3d(out, self, weight, kernel_size, bias, stride, padding);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SLOW_CONV3D_OUT, out, self, weight, kernel_size, bias, stride, padding);
}

Tensor wrap_slow_conv3d(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, weight);
    return at::redispatch::slow_conv3d(self, weight, kernel_size, bias, stride, padding);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SLOW_CONV3D, self, weight, kernel_size, bias, stride, padding);
}

Tensor wrap_slow_conv3d_forward_output(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(output, finput, fgrad_input, self, weight);
    return at::redispatch::slow_conv3d_forward(output, finput, fgrad_input, self, weight, kernel_size, bias, stride, padding);
  }
  return MK_TORCHY(output.dtype(), output.device(), H_SLOW_CONV3D_FORWARD_OUTPUT, output, finput, fgrad_input, self, weight, kernel_size, bias, stride, padding);
}

Tensor wrap_slow_conv3d_forward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, weight);
    return at::redispatch::slow_conv3d_forward(self, weight, kernel_size, bias, stride, padding);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SLOW_CONV3D_FORWARD, self, weight, kernel_size, bias, stride, padding);
}

Tensor wrap_slow_conv3d_backward_grad_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_weight, grad_bias, grad_output, self, weight, finput, fgrad_input);
    return at::redispatch::slow_conv3d_backward(grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_SLOW_CONV3D_BACKWARD_GRAD_INPUT, grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input);
}

Tensor wrap_slow_conv3d_backward_output_mask(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, weight, finput, fgrad_input);
    return at::redispatch::slow_conv3d_backward(grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input, output_mask);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_SLOW_CONV3D_BACKWARD_OUTPUT_MASK, grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input, output_mask);
}

Tensor wrap_slow_conv_dilated2d(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, weight);
    return at::redispatch::slow_conv_dilated2d(self, weight, kernel_size, bias, stride, padding, dilation);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SLOW_CONV_DILATED2D, self, weight, kernel_size, bias, stride, padding, dilation);
}

Tensor wrap_slow_conv_dilated2d_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, weight);
    return at::redispatch::slow_conv_dilated2d_backward(grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_SLOW_CONV_DILATED2D_BACKWARD, grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
}

Tensor wrap_slow_conv_dilated3d(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, weight);
    return at::redispatch::slow_conv_dilated3d(self, weight, kernel_size, bias, stride, padding, dilation);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SLOW_CONV_DILATED3D, self, weight, kernel_size, bias, stride, padding, dilation);
}

Tensor wrap_slow_conv_dilated3d_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, weight);
    return at::redispatch::slow_conv_dilated3d_backward(grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_SLOW_CONV_DILATED3D_BACKWARD, grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
}

Tensor wrap_col2im_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::col2im(out, self, output_size, kernel_size, dilation, padding, stride);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_COL2IM_OUT, out, self, output_size, kernel_size, dilation, padding, stride);
}

Tensor wrap_col2im(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::col2im(self, output_size, kernel_size, dilation, padding, stride);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_COL2IM, self, output_size, kernel_size, dilation, padding, stride);
}

Tensor wrap_col2im_backward_grad_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output);
    return at::redispatch::col2im_backward(grad_input, grad_output, kernel_size, dilation, padding, stride);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_COL2IM_BACKWARD_GRAD_INPUT, grad_input, grad_output, kernel_size, dilation, padding, stride);
}

Tensor wrap_col2im_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output);
    return at::redispatch::col2im_backward(grad_output, kernel_size, dilation, padding, stride);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_COL2IM_BACKWARD, grad_output, kernel_size, dilation, padding, stride);
}

Tensor wrap_column_stack(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::column_stack(tensors);
  }
  return MK_TORCHY(None, None, H_COLUMN_STACK, tensors);
}

Tensor wrap_column_stack_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::column_stack(out, tensors);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_COLUMN_STACK_OUT, out, tensors);
}

Tensor wrap_im2col_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::im2col(out, self, kernel_size, dilation, padding, stride);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_IM2COL_OUT, out, self, kernel_size, dilation, padding, stride);
}

Tensor wrap_im2col(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::im2col(self, kernel_size, dilation, padding, stride);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_IM2COL, self, kernel_size, dilation, padding, stride);
}

Tensor wrap_im2col_backward_grad_input(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output);
    return at::redispatch::im2col_backward(grad_input, grad_output, input_size, kernel_size, dilation, padding, stride);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_IM2COL_BACKWARD_GRAD_INPUT, grad_input, grad_output, input_size, kernel_size, dilation, padding, stride);
}

Tensor wrap_im2col_backward(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output);
    return at::redispatch::im2col_backward(grad_output, input_size, kernel_size, dilation, padding, stride);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_IM2COL_BACKWARD, grad_output, input_size, kernel_size, dilation, padding, stride);
}

Tensor wrap_isfinite(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::isfinite(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ISFINITE, self);
}

Tensor wrap_isinf(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::isinf(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ISINF, self);
}

void wrap_record_stream(args...) {
  ensure_materialized(self);
  return at::redispatch::record_stream(self, s);
}

Tensor wrap_isposinf(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::isposinf(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ISPOSINF, self);
}

Tensor wrap_isposinf_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::isposinf(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ISPOSINF_OUT, out, self);
}

Tensor wrap_isneginf(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::isneginf(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ISNEGINF, self);
}

Tensor wrap_isneginf_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::isneginf(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ISNEGINF_OUT, out, self);
}

Tensor wrap__add_batch_dim(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_add_batch_dim(self, batch_dim, level);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__ADD_BATCH_DIM, self, batch_dim, level);
}

Tensor wrap__remove_batch_dim(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_remove_batch_dim(self, level, batch_size, out_dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__REMOVE_BATCH_DIM, self, level, batch_size, out_dim);
}

Tensor wrap_special_entr_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::special_entr(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SPECIAL_ENTR_OUT, out, self);
}

Tensor wrap_special_expm1(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::special_expm1(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SPECIAL_EXPM1, self);
}

Tensor wrap_special_expm1_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::special_expm1(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SPECIAL_EXPM1_OUT, out, self);
}

Tensor wrap_special_exp2(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::special_exp2(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SPECIAL_EXP2, self);
}

Tensor wrap_special_exp2_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::special_exp2(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SPECIAL_EXP2_OUT, out, self);
}

Tensor wrap_special_gammaln(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::special_gammaln(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SPECIAL_GAMMALN, self);
}

Tensor wrap_special_gammaln_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::special_gammaln(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SPECIAL_GAMMALN_OUT, out, self);
}

Tensor wrap_special_erf(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::special_erf(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SPECIAL_ERF, self);
}

Tensor wrap_special_erf_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::special_erf(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SPECIAL_ERF_OUT, out, self);
}

Tensor wrap_special_erfc(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::special_erfc(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SPECIAL_ERFC, self);
}

Tensor wrap_special_erfc_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::special_erfc(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SPECIAL_ERFC_OUT, out, self);
}

Tensor wrap_special_erfinv(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::special_erfinv(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SPECIAL_ERFINV, self);
}

Tensor wrap_special_erfinv_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::special_erfinv(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SPECIAL_ERFINV_OUT, out, self);
}

Tensor wrap_special_i0e_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::special_i0e(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SPECIAL_I0E_OUT, out, self);
}

Tensor wrap_special_logit(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::special_logit(self, eps);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SPECIAL_LOGIT, self, eps);
}

Tensor wrap_special_logit_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::special_logit(out, self, eps);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SPECIAL_LOGIT_OUT, out, self, eps);
}

Tensor wrap_special_expit(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::special_expit(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SPECIAL_EXPIT, self);
}

Tensor wrap_special_expit_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::special_expit(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SPECIAL_EXPIT_OUT, out, self);
}

Tensor wrap_fft_fft(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::fft_fft(self, n, dim, norm);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FFT_FFT, self, n, dim, norm);
}

Tensor wrap_fft_fft_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::fft_fft(out, self, n, dim, norm);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FFT_FFT_OUT, out, self, n, dim, norm);
}

Tensor wrap_fft_ifft(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::fft_ifft(self, n, dim, norm);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FFT_IFFT, self, n, dim, norm);
}

Tensor wrap_fft_ifft_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::fft_ifft(out, self, n, dim, norm);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FFT_IFFT_OUT, out, self, n, dim, norm);
}

Tensor wrap_fft_rfft(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::fft_rfft(self, n, dim, norm);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FFT_RFFT, self, n, dim, norm);
}

Tensor wrap_fft_rfft_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::fft_rfft(out, self, n, dim, norm);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FFT_RFFT_OUT, out, self, n, dim, norm);
}

Tensor wrap_fft_irfft(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::fft_irfft(self, n, dim, norm);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FFT_IRFFT, self, n, dim, norm);
}

Tensor wrap_fft_irfft_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::fft_irfft(out, self, n, dim, norm);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FFT_IRFFT_OUT, out, self, n, dim, norm);
}

Tensor wrap_fft_hfft(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::fft_hfft(self, n, dim, norm);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FFT_HFFT, self, n, dim, norm);
}

Tensor wrap_fft_hfft_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::fft_hfft(out, self, n, dim, norm);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FFT_HFFT_OUT, out, self, n, dim, norm);
}

Tensor wrap_fft_ihfft(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::fft_ihfft(self, n, dim, norm);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FFT_IHFFT, self, n, dim, norm);
}

Tensor wrap_fft_ihfft_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::fft_ihfft(out, self, n, dim, norm);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FFT_IHFFT_OUT, out, self, n, dim, norm);
}

Tensor wrap_fft_fft2(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::fft_fft2(self, s, dim, norm);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FFT_FFT2, self, s, dim, norm);
}

Tensor wrap_fft_fft2_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::fft_fft2(out, self, s, dim, norm);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FFT_FFT2_OUT, out, self, s, dim, norm);
}

Tensor wrap_fft_ifft2(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::fft_ifft2(self, s, dim, norm);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FFT_IFFT2, self, s, dim, norm);
}

Tensor wrap_fft_ifft2_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::fft_ifft2(out, self, s, dim, norm);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FFT_IFFT2_OUT, out, self, s, dim, norm);
}

Tensor wrap_fft_rfft2(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::fft_rfft2(self, s, dim, norm);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FFT_RFFT2, self, s, dim, norm);
}

Tensor wrap_fft_rfft2_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::fft_rfft2(out, self, s, dim, norm);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FFT_RFFT2_OUT, out, self, s, dim, norm);
}

Tensor wrap_fft_irfft2(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::fft_irfft2(self, s, dim, norm);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FFT_IRFFT2, self, s, dim, norm);
}

Tensor wrap_fft_irfft2_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::fft_irfft2(out, self, s, dim, norm);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FFT_IRFFT2_OUT, out, self, s, dim, norm);
}

Tensor wrap_fft_fftn(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::fft_fftn(self, s, dim, norm);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FFT_FFTN, self, s, dim, norm);
}

Tensor wrap_fft_fftn_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::fft_fftn(out, self, s, dim, norm);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FFT_FFTN_OUT, out, self, s, dim, norm);
}

Tensor wrap_fft_ifftn(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::fft_ifftn(self, s, dim, norm);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FFT_IFFTN, self, s, dim, norm);
}

Tensor wrap_fft_ifftn_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::fft_ifftn(out, self, s, dim, norm);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FFT_IFFTN_OUT, out, self, s, dim, norm);
}

Tensor wrap_fft_rfftn(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::fft_rfftn(self, s, dim, norm);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FFT_RFFTN, self, s, dim, norm);
}

Tensor wrap_fft_rfftn_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::fft_rfftn(out, self, s, dim, norm);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FFT_RFFTN_OUT, out, self, s, dim, norm);
}

Tensor wrap_fft_irfftn(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::fft_irfftn(self, s, dim, norm);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FFT_IRFFTN, self, s, dim, norm);
}

Tensor wrap_fft_irfftn_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::fft_irfftn(out, self, s, dim, norm);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FFT_IRFFTN_OUT, out, self, s, dim, norm);
}

Tensor wrap_fft_fftfreq(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::fft_fftfreq(n, d, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_FFT_FFTFREQ, n, d, dtype, layout, device, pin_memory);
}

Tensor wrap_fft_fftfreq_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::fft_fftfreq(out, n, d);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FFT_FFTFREQ_OUT, out, n, d);
}

Tensor wrap_fft_rfftfreq(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::fft_rfftfreq(n, d, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(None, None, H_FFT_RFFTFREQ, n, d, dtype, layout, device, pin_memory);
}

Tensor wrap_fft_rfftfreq_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::fft_rfftfreq(out, n, d);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FFT_RFFTFREQ_OUT, out, n, d);
}

Tensor wrap_fft_fftshift(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::fft_fftshift(self, dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FFT_FFTSHIFT, self, dim);
}

Tensor wrap_fft_ifftshift(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::fft_ifftshift(self, dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FFT_IFFTSHIFT, self, dim);
}

Tensor wrap_linalg_cholesky(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::linalg_cholesky(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LINALG_CHOLESKY, self);
}

Tensor wrap_linalg_cholesky_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::linalg_cholesky(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LINALG_CHOLESKY_OUT, out, self);
}

Tensor wrap_linalg_det(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::linalg_det(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LINALG_DET, self);
}

Tensor wrap_linalg_det_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::linalg_det(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LINALG_DET_OUT, out, self);
}

Tensor wrap_det(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::det(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DET, self);
}

Tensor wrap_linalg_lstsq(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, b);
    return at::redispatch::linalg_lstsq(self, b, cond, driver);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LINALG_LSTSQ, self, b, cond, driver);
}

Tensor wrap_linalg_lstsq_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(solution, residuals, rank, singular_values, self, b);
    return at::redispatch::linalg_lstsq(solution, residuals, rank, singular_values, self, b, cond, driver);
  }
  return MK_TORCHY(solution.dtype(), solution.device(), H_LINALG_LSTSQ_OUT, solution, residuals, rank, singular_values, self, b, cond, driver);
}

Tensor wrap__lstsq_helper_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, rank, singular_values, infos, a);
    return at::redispatch::_lstsq_helper_(self, rank, singular_values, infos, a, cond, driver_name);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__LSTSQ_HELPER_, self, rank, singular_values, infos, a, cond, driver_name);
}

Tensor wrap_linalg_slogdet(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::linalg_slogdet(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LINALG_SLOGDET, self);
}

Tensor wrap_linalg_slogdet_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(sign, logabsdet, self);
    return at::redispatch::linalg_slogdet(sign, logabsdet, self);
  }
  return MK_TORCHY(sign.dtype(), sign.device(), H_LINALG_SLOGDET_OUT, sign, logabsdet, self);
}

Tensor wrap_linalg_eig(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::linalg_eig(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LINALG_EIG, self);
}

Tensor wrap_linalg_eig_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(eigenvalues, eigenvectors, self);
    return at::redispatch::linalg_eig(eigenvalues, eigenvectors, self);
  }
  return MK_TORCHY(eigenvalues.dtype(), eigenvalues.device(), H_LINALG_EIG_OUT, eigenvalues, eigenvectors, self);
}

Tensor wrap_linalg_eigvals(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::linalg_eigvals(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LINALG_EIGVALS, self);
}

Tensor wrap_linalg_eigvals_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::linalg_eigvals(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LINALG_EIGVALS_OUT, out, self);
}

Tensor wrap_linalg_eigh(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::linalg_eigh(self, UPLO);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LINALG_EIGH, self, UPLO);
}

Tensor wrap_linalg_eigh_eigvals(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(eigvals, eigvecs, self);
    return at::redispatch::linalg_eigh(eigvals, eigvecs, self, UPLO);
  }
  return MK_TORCHY(eigvals.dtype(), eigvals.device(), H_LINALG_EIGH_EIGVALS, eigvals, eigvecs, self, UPLO);
}

Tensor wrap_linalg_eigvalsh(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::linalg_eigvalsh(self, UPLO);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LINALG_EIGVALSH, self, UPLO);
}

Tensor wrap_linalg_eigvalsh_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::linalg_eigvalsh(out, self, UPLO);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LINALG_EIGVALSH_OUT, out, self, UPLO);
}

Tensor wrap_linalg_householder_product(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, tau);
    return at::redispatch::linalg_householder_product(input, tau);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_LINALG_HOUSEHOLDER_PRODUCT, input, tau);
}

Tensor wrap_linalg_householder_product_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, input, tau);
    return at::redispatch::linalg_householder_product(out, input, tau);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LINALG_HOUSEHOLDER_PRODUCT_OUT, out, input, tau);
}

Tensor wrap__linalg_inv_out_helper_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, infos_lu, infos_getri);
    return at::redispatch::_linalg_inv_out_helper_(self, infos_lu, infos_getri);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__LINALG_INV_OUT_HELPER_, self, infos_lu, infos_getri);
}

Tensor wrap_linalg_inv(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::linalg_inv(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LINALG_INV, self);
}

Tensor wrap_linalg_inv_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::linalg_inv(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LINALG_INV_OUT, out, self);
}

Tensor wrap_inner(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::inner(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INNER, self, other);
}

Tensor wrap_inner_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::inner(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_INNER_OUT, out, self, other);
}

Tensor wrap_outer(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, vec2);
    return at::redispatch::outer(self, vec2);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_OUTER, self, vec2);
}

Tensor wrap_outer_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, vec2);
    return at::redispatch::outer(out, self, vec2);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_OUTER_OUT, out, self, vec2);
}

Tensor wrap_ger(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, vec2);
    return at::redispatch::ger(self, vec2);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_GER, self, vec2);
}

Tensor wrap_ger_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, vec2);
    return at::redispatch::ger(out, self, vec2);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_GER_OUT, out, self, vec2);
}

Tensor wrap_linalg_norm(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::linalg_norm(self, ord, dim, keepdim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LINALG_NORM, self, ord, dim, keepdim, dtype);
}

Tensor wrap_linalg_norm_ord_str(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::linalg_norm(self, ord, dim, keepdim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LINALG_NORM_ORD_STR, self, ord, dim, keepdim, dtype);
}

Tensor wrap_linalg_norm_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::linalg_norm(out, self, ord, dim, keepdim, dtype);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LINALG_NORM_OUT, out, self, ord, dim, keepdim, dtype);
}

Tensor wrap_linalg_norm_ord_str_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::linalg_norm(out, self, ord, dim, keepdim, dtype);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LINALG_NORM_ORD_STR_OUT, out, self, ord, dim, keepdim, dtype);
}

Tensor wrap_linalg_vector_norm(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::linalg_vector_norm(self, ord, dim, keepdim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LINALG_VECTOR_NORM, self, ord, dim, keepdim, dtype);
}

Tensor wrap_linalg_vector_norm_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::linalg_vector_norm(out, self, ord, dim, keepdim, dtype);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LINALG_VECTOR_NORM_OUT, out, self, ord, dim, keepdim, dtype);
}

Tensor wrap_linalg_svd_U(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(U, S, V, self);
    return at::redispatch::linalg_svd(U, S, V, self, full_matrices, compute_uv);
  }
  return MK_TORCHY(U.dtype(), U.device(), H_LINALG_SVD_U, U, S, V, self, full_matrices, compute_uv);
}

Tensor wrap_linalg_svd(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::linalg_svd(self, full_matrices, compute_uv);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LINALG_SVD, self, full_matrices, compute_uv);
}

Tensor wrap_linalg_cond(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::linalg_cond(self, p);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LINALG_COND, self, p);
}

Tensor wrap_linalg_cond_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::linalg_cond(out, self, p);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LINALG_COND_OUT, out, self, p);
}

Tensor wrap_linalg_cond_p_str(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::linalg_cond(self, p);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LINALG_COND_P_STR, self, p);
}

Tensor wrap_linalg_cond_p_str_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::linalg_cond(out, self, p);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LINALG_COND_P_STR_OUT, out, self, p);
}

Tensor wrap_linalg_pinv(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::linalg_pinv(self, rcond, hermitian);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LINALG_PINV, self, rcond, hermitian);
}

Tensor wrap_linalg_pinv_rcond_tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, rcond);
    return at::redispatch::linalg_pinv(self, rcond, hermitian);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LINALG_PINV_RCOND_TENSOR, self, rcond, hermitian);
}

Tensor wrap_linalg_pinv_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::linalg_pinv(out, self, rcond, hermitian);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LINALG_PINV_OUT, out, self, rcond, hermitian);
}

Tensor wrap_linalg_pinv_out_rcond_tensor(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, rcond);
    return at::redispatch::linalg_pinv(out, self, rcond, hermitian);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LINALG_PINV_OUT_RCOND_TENSOR, out, self, rcond, hermitian);
}

Tensor wrap__linalg_solve_out_helper_(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other, infos);
    return at::redispatch::_linalg_solve_out_helper_(self, other, infos);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__LINALG_SOLVE_OUT_HELPER_, self, other, infos);
}

Tensor wrap_linalg_solve(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(input, other);
    return at::redispatch::linalg_solve(input, other);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_LINALG_SOLVE, input, other);
}

Tensor wrap_linalg_solve_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, input, other);
    return at::redispatch::linalg_solve(out, input, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LINALG_SOLVE_OUT, out, input, other);
}

Tensor wrap_linalg_tensorinv(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::linalg_tensorinv(self, ind);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LINALG_TENSORINV, self, ind);
}

Tensor wrap_linalg_tensorinv_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::linalg_tensorinv(out, self, ind);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LINALG_TENSORINV_OUT, out, self, ind);
}

Tensor wrap_linalg_tensorsolve(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::linalg_tensorsolve(self, other, dims);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LINALG_TENSORSOLVE, self, other, dims);
}

Tensor wrap_linalg_tensorsolve_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::linalg_tensorsolve(out, self, other, dims);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LINALG_TENSORSOLVE_OUT, out, self, other, dims);
}

Tensor wrap_linalg_qr(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::linalg_qr(self, mode);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LINALG_QR, self, mode);
}

Tensor wrap_linalg_qr_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(Q, R, self);
    return at::redispatch::linalg_qr(Q, R, self, mode);
  }
  return MK_TORCHY(Q.dtype(), Q.device(), H_LINALG_QR_OUT, Q, R, self, mode);
}

Tensor wrap__linalg_qr_helper(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_linalg_qr_helper(self, mode);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__LINALG_QR_HELPER, self, mode);
}

Tensor wrap_linalg_matrix_power(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::linalg_matrix_power(self, n);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LINALG_MATRIX_POWER, self, n);
}

Tensor wrap_linalg_matrix_power_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::linalg_matrix_power(out, self, n);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LINALG_MATRIX_POWER_OUT, out, self, n);
}

Tensor wrap_linalg_matrix_rank(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::linalg_matrix_rank(self, tol, hermitian);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LINALG_MATRIX_RANK, self, tol, hermitian);
}

Tensor wrap_linalg_matrix_rank_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::linalg_matrix_rank(out, self, tol, hermitian);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LINALG_MATRIX_RANK_OUT, out, self, tol, hermitian);
}

Tensor wrap_linalg_multi_dot(args...) {
  if (trace.is_flushing()) {
    ensure_materialized();
    return at::redispatch::linalg_multi_dot(tensors);
  }
  return MK_TORCHY(None, None, H_LINALG_MULTI_DOT, tensors);
}

Tensor wrap_linalg_multi_dot_out(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::linalg_multi_dot(out, tensors);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LINALG_MULTI_DOT_OUT, out, tensors);
}

Tensor wrap__test_serialization_subcmul(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::_test_serialization_subcmul(self, other, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__TEST_SERIALIZATION_SUBCMUL, self, other, alpha);
}

Tensor wrap__test_optional_intlist(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(values);
    return at::redispatch::_test_optional_intlist(values, addends);
  }
  return MK_TORCHY(values.dtype(), values.device(), H__TEST_OPTIONAL_INTLIST, values, addends);
}

Tensor wrap__test_optional_filled_intlist(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(values);
    return at::redispatch::_test_optional_filled_intlist(values, addends);
  }
  return MK_TORCHY(values.dtype(), values.device(), H__TEST_OPTIONAL_FILLED_INTLIST, values, addends);
}

Tensor wrap__test_optional_floatlist(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(values);
    return at::redispatch::_test_optional_floatlist(values, addends);
  }
  return MK_TORCHY(values.dtype(), values.device(), H__TEST_OPTIONAL_FLOATLIST, values, addends);
}

Tensor wrap__test_string_default(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(dummy);
    return at::redispatch::_test_string_default(dummy, a, b);
  }
  return MK_TORCHY(dummy.dtype(), dummy.device(), H__TEST_STRING_DEFAULT, dummy, a, b);
}

Tensor wrap__test_ambiguous_defaults_a(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(dummy);
    return at::redispatch::_test_ambiguous_defaults(dummy, a, b);
  }
  return MK_TORCHY(dummy.dtype(), dummy.device(), H__TEST_AMBIGUOUS_DEFAULTS_A, dummy, a, b);
}

Tensor wrap__test_ambiguous_defaults_b(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(dummy);
    return at::redispatch::_test_ambiguous_defaults(dummy, a, b);
  }
  return MK_TORCHY(dummy.dtype(), dummy.device(), H__TEST_AMBIGUOUS_DEFAULTS_B, dummy, a, b);
}

Tensor wrap_segment_reduce(args...) {
  if (trace.is_flushing()) {
    ensure_materialized(data);
    return at::redispatch::segment_reduce(data, reduce, lengths, indices, axis, unsafe);
  }
  return MK_TORCHY(data.dtype(), data.device(), H_SEGMENT_REDUCE, data, reduce, lengths, indices, axis, unsafe);
}
