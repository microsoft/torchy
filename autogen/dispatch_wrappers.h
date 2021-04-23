
at::Tensor wrap__cast_Byte(const at::Tensor & self, bool non_blocking) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_cast_Byte(self, non_blocking);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__CAST_BYTE, self, non_blocking);
}

at::Tensor wrap__cast_Char(const at::Tensor & self, bool non_blocking) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_cast_Char(self, non_blocking);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__CAST_CHAR, self, non_blocking);
}

at::Tensor wrap__cast_Double(const at::Tensor & self, bool non_blocking) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_cast_Double(self, non_blocking);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__CAST_DOUBLE, self, non_blocking);
}

at::Tensor wrap__cast_Float(const at::Tensor & self, bool non_blocking) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_cast_Float(self, non_blocking);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__CAST_FLOAT, self, non_blocking);
}

at::Tensor wrap__cast_Int(const at::Tensor & self, bool non_blocking) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_cast_Int(self, non_blocking);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__CAST_INT, self, non_blocking);
}

at::Tensor wrap__cast_Long(const at::Tensor & self, bool non_blocking) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_cast_Long(self, non_blocking);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__CAST_LONG, self, non_blocking);
}

at::Tensor wrap__cast_Short(const at::Tensor & self, bool non_blocking) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_cast_Short(self, non_blocking);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__CAST_SHORT, self, non_blocking);
}

at::Tensor wrap__cast_Half(const at::Tensor & self, bool non_blocking) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_cast_Half(self, non_blocking);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__CAST_HALF, self, non_blocking);
}

at::Tensor wrap__fw_primal(const at::Tensor & self, int64_t level) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_fw_primal(self, level);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__FW_PRIMAL, self, level);
}

at::Tensor wrap__make_dual(const at::Tensor & primal, const at::Tensor & tangent, int64_t level) {
  if (trace.is_flushing()) {
    ensure_materialized(primal, tangent);
    return at::redispatch::_make_dual(primal, tangent, level);
  }
  return MK_TORCHY(primal.dtype(), primal.device(), H__MAKE_DUAL, primal, tangent, level);
}

std::tuple<at::Tensor,at::Tensor> wrap__unpack_dual(const at::Tensor & dual, int64_t level) {
  ensure_materialized(dual);
  return at::redispatch::_unpack_dual(dual, level);
}

at::Tensor & wrap_rename_(at::Tensor & self, c10::optional<at::DimnameList> names) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::rename_(self, names);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RENAME_, self, names);
}

at::Tensor wrap_rename(const at::Tensor & self, c10::optional<at::DimnameList> names) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::rename(self, names);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RENAME, self, names);
}

at::Tensor wrap_align_to(const at::Tensor & self, at::DimnameList names) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::align_to(self, names);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ALIGN_TO, self, names);
}

at::Tensor wrap_align_to_ellipsis_idx(const at::Tensor & self, at::DimnameList order, int64_t ellipsis_idx) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::align_to(self, order, ellipsis_idx);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ALIGN_TO_ELLIPSIS_IDX, self, order, ellipsis_idx);
}

at::Tensor wrap_align_as(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::align_as(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ALIGN_AS, self, other);
}

std::vector<at::Tensor> wrap_align_tensors(at::TensorList tensors) {
  ensure_materialized();
  return at::redispatch::align_tensors(tensors);
}

void wrap__assert_async(const at::Tensor & self) {
  ensure_materialized(self);
  return at::redispatch::_assert_async(self);
}

at::Tensor wrap_refine_names(const at::Tensor & self, at::DimnameList names) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::refine_names(self, names);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_REFINE_NAMES, self, names);
}

bool wrap__use_cudnn_ctc_loss(const at::Tensor & log_probs, const at::Tensor & targets, at::IntArrayRef input_lengths, at::IntArrayRef target_lengths, int64_t blank) {
  ensure_materialized(log_probs, targets);
  return at::redispatch::_use_cudnn_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank);
}

std::tuple<at::Tensor,at::Tensor> wrap__cudnn_ctc_loss(const at::Tensor & log_probs, const at::Tensor & targets, at::IntArrayRef input_lengths, at::IntArrayRef target_lengths, int64_t blank, bool deterministic, bool zero_infinity) {
  ensure_materialized(log_probs, targets);
  return at::redispatch::_cudnn_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, deterministic, zero_infinity);
}

bool wrap__use_cudnn_rnn_flatten_weight() {
  ensure_materialized();
  return at::redispatch::_use_cudnn_rnn_flatten_weight();
}

at::Tensor wrap__cudnn_rnn_flatten_weight(at::TensorList weight_arr, int64_t weight_stride0, int64_t input_size, int64_t mode, int64_t hidden_size, int64_t proj_size, int64_t num_layers, bool batch_first, bool bidirectional) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::_cudnn_rnn_flatten_weight(weight_arr, weight_stride0, input_size, mode, hidden_size, proj_size, num_layers, batch_first, bidirectional));
}

std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> wrap__cudnn_rnn(const at::Tensor & input, at::TensorList weight, int64_t weight_stride0, const c10::optional<at::Tensor> & weight_buf, const at::Tensor & hx, const c10::optional<at::Tensor> & cx, int64_t mode, int64_t hidden_size, int64_t proj_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, at::IntArrayRef batch_sizes, const c10::optional<at::Tensor> & dropout_state) {
  ensure_materialized(input, hx);
  return at::redispatch::_cudnn_rnn(input, weight, weight_stride0, weight_buf, hx, cx, mode, hidden_size, proj_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor,std::vector<at::Tensor>> wrap__cudnn_rnn_backward(const at::Tensor & input, at::TensorList weight, int64_t weight_stride0, const at::Tensor & weight_buf, const at::Tensor & hx, const c10::optional<at::Tensor> & cx, const at::Tensor & output, const c10::optional<at::Tensor> & grad_output, const c10::optional<at::Tensor> & grad_hy, const c10::optional<at::Tensor> & grad_cy, int64_t mode, int64_t hidden_size, int64_t proj_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, at::IntArrayRef batch_sizes, const c10::optional<at::Tensor> & dropout_state, const at::Tensor & reserve, std::array<bool,4> output_mask) {
  ensure_materialized(input, weight_buf, hx, output, reserve);
  return at::redispatch::_cudnn_rnn_backward(input, weight, weight_stride0, weight_buf, hx, cx, output, grad_output, grad_hy, grad_cy, mode, hidden_size, proj_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state, reserve, output_mask);
}

at::Tensor wrap__cudnn_init_dropout_state(double dropout, bool train, int64_t dropout_seed, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::_cudnn_init_dropout_state(dropout, train, dropout_seed, dtype, layout, device, pin_memory));
}

int64_t wrap__debug_has_internal_overlap(const at::Tensor & self) {
  ensure_materialized(self);
  return at::redispatch::_debug_has_internal_overlap(self);
}

std::tuple<at::Tensor,at::Tensor> wrap__fused_dropout(const at::Tensor & self, double p, c10::optional<at::Generator> generator) {
  ensure_materialized(self);
  return at::redispatch::_fused_dropout(self, p, generator);
}

at::Tensor wrap__masked_scale(const at::Tensor & self, const at::Tensor & mask, double scale) {
  if (trace.is_flushing()) {
    ensure_materialized(self, mask);
    return at::redispatch::_masked_scale(self, mask, scale);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__MASKED_SCALE, self, mask, scale);
}

std::tuple<at::Tensor,at::Tensor> wrap__sobol_engine_draw(const at::Tensor & quasi, int64_t n, const at::Tensor & sobolstate, int64_t dimension, int64_t num_generated, c10::optional<at::ScalarType> dtype) {
  ensure_materialized(quasi, sobolstate);
  return at::redispatch::_sobol_engine_draw(quasi, n, sobolstate, dimension, num_generated, dtype);
}

at::Tensor & wrap__sobol_engine_ff_(at::Tensor & self, int64_t n, const at::Tensor & sobolstate, int64_t dimension, int64_t num_generated) {
  if (trace.is_flushing()) {
    ensure_materialized(self, sobolstate);
    return at::redispatch::_sobol_engine_ff_(self, n, sobolstate, dimension, num_generated);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__SOBOL_ENGINE_FF_, self, n, sobolstate, dimension, num_generated);
}

at::Tensor & wrap__sobol_engine_scramble_(at::Tensor & self, const at::Tensor & ltm, int64_t dimension) {
  if (trace.is_flushing()) {
    ensure_materialized(self, ltm);
    return at::redispatch::_sobol_engine_scramble_(self, ltm, dimension);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__SOBOL_ENGINE_SCRAMBLE_, self, ltm, dimension);
}

at::Tensor & wrap__sobol_engine_initialize_state_(at::Tensor & self, int64_t dimension) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_sobol_engine_initialize_state_(self, dimension);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__SOBOL_ENGINE_INITIALIZE_STATE_, self, dimension);
}

at::Tensor wrap__reshape_from_tensor(const at::Tensor & self, const at::Tensor & shape) {
  if (trace.is_flushing()) {
    ensure_materialized(self, shape);
    return at::redispatch::_reshape_from_tensor(self, shape);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__RESHAPE_FROM_TENSOR, self, shape);
}

at::Tensor wrap__shape_as_tensor(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_shape_as_tensor(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__SHAPE_AS_TENSOR, self);
}

at::Tensor wrap_dropout(const at::Tensor & input, double p, bool train) {
  if (trace.is_flushing()) {
    ensure_materialized(input);
    return at::redispatch::dropout(input, p, train);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_DROPOUT, input, p, train);
}

at::Tensor & wrap_dropout_(at::Tensor & self, double p, bool train) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::dropout_(self, p, train);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DROPOUT_, self, p, train);
}

at::Tensor wrap_feature_dropout(const at::Tensor & input, double p, bool train) {
  if (trace.is_flushing()) {
    ensure_materialized(input);
    return at::redispatch::feature_dropout(input, p, train);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_FEATURE_DROPOUT, input, p, train);
}

at::Tensor & wrap_feature_dropout_(at::Tensor & self, double p, bool train) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::feature_dropout_(self, p, train);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FEATURE_DROPOUT_, self, p, train);
}

at::Tensor wrap_alpha_dropout(const at::Tensor & input, double p, bool train) {
  if (trace.is_flushing()) {
    ensure_materialized(input);
    return at::redispatch::alpha_dropout(input, p, train);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_ALPHA_DROPOUT, input, p, train);
}

at::Tensor & wrap_alpha_dropout_(at::Tensor & self, double p, bool train) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::alpha_dropout_(self, p, train);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ALPHA_DROPOUT_, self, p, train);
}

at::Tensor wrap_feature_alpha_dropout(const at::Tensor & input, double p, bool train) {
  if (trace.is_flushing()) {
    ensure_materialized(input);
    return at::redispatch::feature_alpha_dropout(input, p, train);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_FEATURE_ALPHA_DROPOUT, input, p, train);
}

at::Tensor & wrap_feature_alpha_dropout_(at::Tensor & self, double p, bool train) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::feature_alpha_dropout_(self, p, train);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FEATURE_ALPHA_DROPOUT_, self, p, train);
}

at::Tensor wrap_abs(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::abs(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ABS, self);
}

at::Tensor & wrap_abs_(at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::abs_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ABS_, self);
}

at::Tensor & wrap_abs_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::abs(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ABS_OUT, out, self);
}

at::Tensor wrap_absolute(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::absolute(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ABSOLUTE, self);
}

at::Tensor & wrap_absolute_(at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::absolute_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ABSOLUTE_, self);
}

at::Tensor & wrap_absolute_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::absolute(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ABSOLUTE_OUT, out, self);
}

at::Tensor wrap_angle(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::angle(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ANGLE, self);
}

at::Tensor & wrap_angle_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::angle(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ANGLE_OUT, out, self);
}

at::Tensor wrap_view_as_real(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::view_as_real(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_VIEW_AS_REAL, self);
}

at::Tensor wrap_view_as_complex(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::view_as_complex(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_VIEW_AS_COMPLEX, self);
}

at::Tensor wrap_sgn(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::sgn(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SGN, self);
}

at::Tensor & wrap_sgn_(at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::sgn_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SGN_, self);
}

at::Tensor & wrap_sgn_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::sgn(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SGN_OUT, out, self);
}

at::Tensor wrap_real(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::real(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_REAL, self);
}

at::Tensor wrap_imag(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::imag(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_IMAG, self);
}

at::Tensor wrap_conj(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::conj(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CONJ, self);
}

at::Tensor & wrap_conj_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::conj(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_CONJ_OUT, out, self);
}

at::Tensor wrap__conj(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_conj(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__CONJ, self);
}

at::Tensor & wrap_acos_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::acos(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ACOS_OUT, out, self);
}

at::Tensor wrap_arccos(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::arccos(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ARCCOS, self);
}

at::Tensor & wrap_arccos_(at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::arccos_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ARCCOS_, self);
}

at::Tensor & wrap_arccos_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::arccos(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ARCCOS_OUT, out, self);
}

at::Tensor wrap_avg_pool1d(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::avg_pool1d(self, kernel_size, stride, padding, ceil_mode, count_include_pad);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_AVG_POOL1D, self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}

at::Tensor wrap_adaptive_avg_pool1d(const at::Tensor & self, at::IntArrayRef output_size) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::adaptive_avg_pool1d(self, output_size);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ADAPTIVE_AVG_POOL1D, self, output_size);
}

std::tuple<at::Tensor,at::Tensor> wrap_adaptive_max_pool1d(const at::Tensor & self, at::IntArrayRef output_size) {
  ensure_materialized(self);
  return at::redispatch::adaptive_max_pool1d(self, output_size);
}

at::Tensor wrap_add_Tensor(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::add(self, other, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ADD_TENSOR, self, other, alpha);
}

at::Tensor & wrap_add__Tensor(at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::add_(self, other, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ADD__TENSOR, self, other, alpha);
}

at::Tensor & wrap_add_out(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::add(out, self, other, alpha);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ADD_OUT, out, self, other, alpha);
}

at::Tensor wrap__add_relu_Tensor(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::_add_relu(self, other, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__ADD_RELU_TENSOR, self, other, alpha);
}

at::Tensor & wrap__add_relu__Tensor(at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::_add_relu_(self, other, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__ADD_RELU__TENSOR, self, other, alpha);
}

at::Tensor & wrap__add_relu_out(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::_add_relu(out, self, other, alpha);
  }
  return MK_TORCHY(out.dtype(), out.device(), H__ADD_RELU_OUT, out, self, other, alpha);
}

at::Tensor wrap_add_Scalar(const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::add(self, other, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ADD_SCALAR, self, other, alpha);
}

at::Tensor & wrap_add__Scalar(at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::add_(self, other, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ADD__SCALAR, self, other, alpha);
}

at::Tensor & wrap_addmv_out(const at::Tensor & self, const at::Tensor & mat, const at::Tensor & vec, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, mat, vec);
    return at::redispatch::addmv(out, self, mat, vec, beta, alpha);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ADDMV_OUT, out, self, mat, vec, beta, alpha);
}

at::Tensor wrap_addr(const at::Tensor & self, const at::Tensor & vec1, const at::Tensor & vec2, const at::Scalar & beta, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    ensure_materialized(self, vec1, vec2);
    return at::redispatch::addr(self, vec1, vec2, beta, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ADDR, self, vec1, vec2, beta, alpha);
}

at::Tensor & wrap_addr_(at::Tensor & self, const at::Tensor & vec1, const at::Tensor & vec2, const at::Scalar & beta, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    ensure_materialized(self, vec1, vec2);
    return at::redispatch::addr_(self, vec1, vec2, beta, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ADDR_, self, vec1, vec2, beta, alpha);
}

at::Tensor & wrap_addr_out(const at::Tensor & self, const at::Tensor & vec1, const at::Tensor & vec2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, vec1, vec2);
    return at::redispatch::addr(out, self, vec1, vec2, beta, alpha);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ADDR_OUT, out, self, vec1, vec2, beta, alpha);
}

at::Tensor wrap_affine_grid_generator(const at::Tensor & theta, at::IntArrayRef size, bool align_corners) {
  if (trace.is_flushing()) {
    ensure_materialized(theta);
    return at::redispatch::affine_grid_generator(theta, size, align_corners);
  }
  return MK_TORCHY(theta.dtype(), theta.device(), H_AFFINE_GRID_GENERATOR, theta, size, align_corners);
}

at::Tensor wrap_affine_grid_generator_backward(const at::Tensor & grad, at::IntArrayRef size, bool align_corners) {
  if (trace.is_flushing()) {
    ensure_materialized(grad);
    return at::redispatch::affine_grid_generator_backward(grad, size, align_corners);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H_AFFINE_GRID_GENERATOR_BACKWARD, grad, size, align_corners);
}

at::Tensor wrap_all_dim(const at::Tensor & self, int64_t dim, bool keepdim) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::all(self, dim, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ALL_DIM, self, dim, keepdim);
}

at::Tensor & wrap_all_out(const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::all(out, self, dim, keepdim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ALL_OUT, out, self, dim, keepdim);
}

at::Tensor wrap_all_dimname(const at::Tensor & self, at::Dimname dim, bool keepdim) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::all(self, dim, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ALL_DIMNAME, self, dim, keepdim);
}

at::Tensor & wrap_all_dimname_out(const at::Tensor & self, at::Dimname dim, bool keepdim, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::all(out, self, dim, keepdim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ALL_DIMNAME_OUT, out, self, dim, keepdim);
}

bool wrap_allclose(const at::Tensor & self, const at::Tensor & other, double rtol, double atol, bool equal_nan) {
  ensure_materialized(self, other);
  return at::redispatch::allclose(self, other, rtol, atol, equal_nan);
}

at::Tensor wrap_any_dim(const at::Tensor & self, int64_t dim, bool keepdim) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::any(self, dim, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ANY_DIM, self, dim, keepdim);
}

at::Tensor & wrap_any_out(const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::any(out, self, dim, keepdim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ANY_OUT, out, self, dim, keepdim);
}

at::Tensor wrap_any_dimname(const at::Tensor & self, at::Dimname dim, bool keepdim) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::any(self, dim, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ANY_DIMNAME, self, dim, keepdim);
}

at::Tensor & wrap_any_dimname_out(const at::Tensor & self, at::Dimname dim, bool keepdim, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::any(out, self, dim, keepdim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ANY_DIMNAME_OUT, out, self, dim, keepdim);
}

at::Tensor wrap_arange(const at::Scalar & end, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::arange(end, dtype, layout, device, pin_memory));
}

at::Tensor wrap_arange_start(const at::Scalar & start, const at::Scalar & end, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::arange(start, end, dtype, layout, device, pin_memory));
}

at::Tensor wrap_arange_start_step(const at::Scalar & start, const at::Scalar & end, const at::Scalar & step, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::arange(start, end, step, dtype, layout, device, pin_memory));
}

at::Tensor & wrap_arange_out(const at::Scalar & end, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::arange(out, end);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ARANGE_OUT, out, end);
}

at::Tensor & wrap_arange_start_out(const at::Scalar & start, const at::Scalar & end, const at::Scalar & step, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::arange(out, start, end, step);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ARANGE_START_OUT, out, start, end, step);
}

at::Tensor wrap__dim_arange(const at::Tensor & like, int64_t dim) {
  if (trace.is_flushing()) {
    ensure_materialized(like);
    return at::redispatch::_dim_arange(like, dim);
  }
  return MK_TORCHY(like.dtype(), like.device(), H__DIM_ARANGE, like, dim);
}

at::Tensor wrap_argmax(const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::argmax(self, dim, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ARGMAX, self, dim, keepdim);
}

at::Tensor & wrap_argmax_out(const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::argmax(out, self, dim, keepdim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ARGMAX_OUT, out, self, dim, keepdim);
}

at::Tensor wrap_argmin(const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::argmin(self, dim, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ARGMIN, self, dim, keepdim);
}

at::Tensor & wrap_argmin_out(const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::argmin(out, self, dim, keepdim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ARGMIN_OUT, out, self, dim, keepdim);
}

at::Tensor & wrap_acosh_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::acosh(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ACOSH_OUT, out, self);
}

at::Tensor wrap_arccosh(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::arccosh(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ARCCOSH, self);
}

at::Tensor & wrap_arccosh_(at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::arccosh_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ARCCOSH_, self);
}

at::Tensor & wrap_arccosh_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::arccosh(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ARCCOSH_OUT, out, self);
}

at::Tensor & wrap_asinh_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::asinh(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ASINH_OUT, out, self);
}

at::Tensor wrap_arcsinh(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::arcsinh(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ARCSINH, self);
}

at::Tensor & wrap_arcsinh_(at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::arcsinh_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ARCSINH_, self);
}

at::Tensor & wrap_arcsinh_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::arcsinh(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ARCSINH_OUT, out, self);
}

at::Tensor & wrap_atanh_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::atanh(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ATANH_OUT, out, self);
}

at::Tensor wrap_arctanh(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::arctanh(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ARCTANH, self);
}

at::Tensor & wrap_arctanh_(at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::arctanh_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ARCTANH_, self);
}

at::Tensor & wrap_arctanh_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::arctanh(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ARCTANH_OUT, out, self);
}

at::Tensor wrap_as_strided(const at::Tensor & self, at::IntArrayRef size, at::IntArrayRef stride, c10::optional<int64_t> storage_offset) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::as_strided(self, size, stride, storage_offset);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_AS_STRIDED, self, size, stride, storage_offset);
}

at::Tensor & wrap_as_strided_(at::Tensor & self, at::IntArrayRef size, at::IntArrayRef stride, c10::optional<int64_t> storage_offset) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::as_strided_(self, size, stride, storage_offset);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_AS_STRIDED_, self, size, stride, storage_offset);
}

at::Tensor wrap_asin(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::asin(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ASIN, self);
}

at::Tensor & wrap_asin_(at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::asin_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ASIN_, self);
}

at::Tensor & wrap_asin_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::asin(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ASIN_OUT, out, self);
}

at::Tensor wrap_arcsin(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::arcsin(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ARCSIN, self);
}

at::Tensor & wrap_arcsin_(at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::arcsin_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ARCSIN_, self);
}

at::Tensor & wrap_arcsin_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::arcsin(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ARCSIN_OUT, out, self);
}

at::Tensor & wrap_atan_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::atan(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ATAN_OUT, out, self);
}

at::Tensor wrap_arctan(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::arctan(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ARCTAN, self);
}

at::Tensor & wrap_arctan_(at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::arctan_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ARCTAN_, self);
}

at::Tensor & wrap_arctan_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::arctan(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ARCTAN_OUT, out, self);
}

at::Tensor wrap_atleast_1d(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::atleast_1d(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ATLEAST_1D, self);
}

std::vector<at::Tensor> wrap_atleast_1d_Sequence(at::TensorList tensors) {
  ensure_materialized();
  return at::redispatch::atleast_1d(tensors);
}

at::Tensor wrap_atleast_2d(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::atleast_2d(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ATLEAST_2D, self);
}

std::vector<at::Tensor> wrap_atleast_2d_Sequence(at::TensorList tensors) {
  ensure_materialized();
  return at::redispatch::atleast_2d(tensors);
}

at::Tensor wrap_atleast_3d(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::atleast_3d(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ATLEAST_3D, self);
}

std::vector<at::Tensor> wrap_atleast_3d_Sequence(at::TensorList tensors) {
  ensure_materialized();
  return at::redispatch::atleast_3d(tensors);
}

at::Tensor wrap_baddbmm(const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    ensure_materialized(self, batch1, batch2);
    return at::redispatch::baddbmm(self, batch1, batch2, beta, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BADDBMM, self, batch1, batch2, beta, alpha);
}

at::Tensor & wrap_baddbmm_(at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    ensure_materialized(self, batch1, batch2);
    return at::redispatch::baddbmm_(self, batch1, batch2, beta, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BADDBMM_, self, batch1, batch2, beta, alpha);
}

at::Tensor & wrap__baddbmm_mkl_(at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    ensure_materialized(self, batch1, batch2);
    return at::redispatch::_baddbmm_mkl_(self, batch1, batch2, beta, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__BADDBMM_MKL_, self, batch1, batch2, beta, alpha);
}

at::Tensor & wrap_baddbmm_out(const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, batch1, batch2);
    return at::redispatch::baddbmm(out, self, batch1, batch2, beta, alpha);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_BADDBMM_OUT, out, self, batch1, batch2, beta, alpha);
}

at::Tensor wrap_bartlett_window(int64_t window_length, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::bartlett_window(window_length, dtype, layout, device, pin_memory));
}

at::Tensor wrap_bartlett_window_periodic(int64_t window_length, bool periodic, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::bartlett_window(window_length, periodic, dtype, layout, device, pin_memory));
}

at::Tensor wrap_batch_norm(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, bool training, double momentum, double eps, bool cudnn_enabled) {
  if (trace.is_flushing()) {
    ensure_materialized(input);
    return at::redispatch::batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_BATCH_NORM, input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
}

at::Tensor wrap_quantized_batch_norm(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const at::Tensor & mean, const at::Tensor & var, double eps, double output_scale, int64_t output_zero_point) {
  if (trace.is_flushing()) {
    ensure_materialized(input, mean, var);
    return at::redispatch::quantized_batch_norm(input, weight, bias, mean, var, eps, output_scale, output_zero_point);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_QUANTIZED_BATCH_NORM, input, weight, bias, mean, var, eps, output_scale, output_zero_point);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t> wrap__batch_norm_impl_index(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, bool training, double momentum, double eps, bool cudnn_enabled) {
  ensure_materialized(input);
  return at::redispatch::_batch_norm_impl_index(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> wrap__batch_norm_impl_index_backward(int64_t impl_index, const at::Tensor & input, const at::Tensor & grad_output, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, const c10::optional<at::Tensor> & save_mean, const c10::optional<at::Tensor> & save_var_transform, bool train, double eps, std::array<bool,3> output_mask, const at::Tensor & reservedSpace) {
  ensure_materialized(input, grad_output, reservedSpace);
  return at::redispatch::_batch_norm_impl_index_backward(impl_index, input, grad_output, weight, running_mean, running_var, save_mean, save_var_transform, train, eps, output_mask, reservedSpace);
}

at::Tensor wrap_bernoulli(const at::Tensor & self, c10::optional<at::Generator> generator) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::bernoulli(self, generator);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BERNOULLI, self, generator);
}

at::Tensor & wrap_bernoulli_out(const at::Tensor & self, c10::optional<at::Generator> generator, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::bernoulli(out, self, generator);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_BERNOULLI_OUT, out, self, generator);
}

at::Tensor & wrap_bernoulli__Tensor(at::Tensor & self, const at::Tensor & p, c10::optional<at::Generator> generator) {
  if (trace.is_flushing()) {
    ensure_materialized(self, p);
    return at::redispatch::bernoulli_(self, p, generator);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BERNOULLI__TENSOR, self, p, generator);
}

at::Tensor & wrap_bernoulli__float(at::Tensor & self, double p, c10::optional<at::Generator> generator) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::bernoulli_(self, p, generator);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BERNOULLI__FLOAT, self, p, generator);
}

at::Tensor wrap_bernoulli_p(const at::Tensor & self, double p, c10::optional<at::Generator> generator) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::bernoulli(self, p, generator);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BERNOULLI_P, self, p, generator);
}

at::Tensor wrap_bilinear(const at::Tensor & input1, const at::Tensor & input2, const at::Tensor & weight, const c10::optional<at::Tensor> & bias) {
  if (trace.is_flushing()) {
    ensure_materialized(input1, input2, weight);
    return at::redispatch::bilinear(input1, input2, weight, bias);
  }
  return MK_TORCHY(input1.dtype(), input1.device(), H_BILINEAR, input1, input2, weight, bias);
}

at::Tensor wrap_binary_cross_entropy(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction) {
  if (trace.is_flushing()) {
    ensure_materialized(self, target);
    return at::redispatch::binary_cross_entropy(self, target, weight, reduction);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BINARY_CROSS_ENTROPY, self, target, weight, reduction);
}

at::Tensor & wrap_binary_cross_entropy_out(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, target);
    return at::redispatch::binary_cross_entropy(out, self, target, weight, reduction);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_BINARY_CROSS_ENTROPY_OUT, out, self, target, weight, reduction);
}

at::Tensor wrap_binary_cross_entropy_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, target);
    return at::redispatch::binary_cross_entropy_backward(grad_output, self, target, weight, reduction);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_BINARY_CROSS_ENTROPY_BACKWARD, grad_output, self, target, weight, reduction);
}

at::Tensor & wrap_binary_cross_entropy_backward_grad_input(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self, target);
    return at::redispatch::binary_cross_entropy_backward(grad_input, grad_output, self, target, weight, reduction);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_BINARY_CROSS_ENTROPY_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, target, weight, reduction);
}

at::Tensor wrap_binary_cross_entropy_with_logits(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & pos_weight, int64_t reduction) {
  if (trace.is_flushing()) {
    ensure_materialized(self, target);
    return at::redispatch::binary_cross_entropy_with_logits(self, target, weight, pos_weight, reduction);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BINARY_CROSS_ENTROPY_WITH_LOGITS, self, target, weight, pos_weight, reduction);
}

at::Tensor wrap_binary_cross_entropy_with_logits_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & pos_weight, int64_t reduction) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, target);
    return at::redispatch::binary_cross_entropy_with_logits_backward(grad_output, self, target, weight, pos_weight, reduction);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_BINARY_CROSS_ENTROPY_WITH_LOGITS_BACKWARD, grad_output, self, target, weight, pos_weight, reduction);
}

at::Tensor wrap_bincount(const at::Tensor & self, const c10::optional<at::Tensor> & weights, int64_t minlength) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::bincount(self, weights, minlength);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BINCOUNT, self, weights, minlength);
}

at::Tensor wrap_bitwise_not(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::bitwise_not(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BITWISE_NOT, self);
}

at::Tensor & wrap_bitwise_not_(at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::bitwise_not_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BITWISE_NOT_, self);
}

at::Tensor & wrap_bitwise_not_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::bitwise_not(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_BITWISE_NOT_OUT, out, self);
}

at::Tensor & wrap_copysign_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::copysign(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_COPYSIGN_OUT, out, self, other);
}

at::Tensor wrap_copysign_Scalar(const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::copysign(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_COPYSIGN_SCALAR, self, other);
}

at::Tensor & wrap_copysign__Scalar(at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::copysign_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_COPYSIGN__SCALAR, self, other);
}

at::Tensor & wrap_copysign_Scalar_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::copysign(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_COPYSIGN_SCALAR_OUT, out, self, other);
}

at::Tensor wrap_logical_not(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::logical_not(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LOGICAL_NOT, self);
}

at::Tensor & wrap_logical_not_(at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::logical_not_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LOGICAL_NOT_, self);
}

at::Tensor & wrap_logical_not_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::logical_not(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LOGICAL_NOT_OUT, out, self);
}

at::Tensor wrap_logical_xor(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::logical_xor(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LOGICAL_XOR, self, other);
}

at::Tensor & wrap_logical_xor_(at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::logical_xor_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LOGICAL_XOR_, self, other);
}

at::Tensor & wrap_logical_xor_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::logical_xor(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LOGICAL_XOR_OUT, out, self, other);
}

at::Tensor wrap_logical_and(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::logical_and(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LOGICAL_AND, self, other);
}

at::Tensor & wrap_logical_and_(at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::logical_and_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LOGICAL_AND_, self, other);
}

at::Tensor & wrap_logical_and_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::logical_and(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LOGICAL_AND_OUT, out, self, other);
}

at::Tensor wrap_logical_or(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::logical_or(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LOGICAL_OR, self, other);
}

at::Tensor & wrap_logical_or_(at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::logical_or_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LOGICAL_OR_, self, other);
}

at::Tensor & wrap_logical_or_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::logical_or(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LOGICAL_OR_OUT, out, self, other);
}

at::Tensor wrap_blackman_window(int64_t window_length, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::blackman_window(window_length, dtype, layout, device, pin_memory));
}

at::Tensor wrap_blackman_window_periodic(int64_t window_length, bool periodic, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::blackman_window(window_length, periodic, dtype, layout, device, pin_memory));
}

at::Tensor wrap_bmm(const at::Tensor & self, const at::Tensor & mat2) {
  if (trace.is_flushing()) {
    ensure_materialized(self, mat2);
    return at::redispatch::bmm(self, mat2);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BMM, self, mat2);
}

at::Tensor wrap__bmm(const at::Tensor & self, const at::Tensor & mat2, bool deterministic) {
  if (trace.is_flushing()) {
    ensure_materialized(self, mat2);
    return at::redispatch::_bmm(self, mat2, deterministic);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__BMM, self, mat2, deterministic);
}

at::Tensor & wrap_bmm_out(const at::Tensor & self, const at::Tensor & mat2, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, mat2);
    return at::redispatch::bmm(out, self, mat2);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_BMM_OUT, out, self, mat2);
}

at::Tensor & wrap__bmm_out(const at::Tensor & self, const at::Tensor & mat2, bool deterministic, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, mat2);
    return at::redispatch::_bmm(out, self, mat2, deterministic);
  }
  return MK_TORCHY(out.dtype(), out.device(), H__BMM_OUT, out, self, mat2, deterministic);
}

std::vector<at::Tensor> wrap_broadcast_tensors(at::TensorList tensors) {
  ensure_materialized();
  return at::redispatch::broadcast_tensors(tensors);
}

at::Tensor wrap_broadcast_to(const at::Tensor & self, at::IntArrayRef size) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::broadcast_to(self, size);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BROADCAST_TO, self, size);
}

at::Tensor wrap_cat(at::TensorList tensors, int64_t dim) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::cat(tensors, dim));
}

at::Tensor & wrap_cat_out(at::TensorList tensors, int64_t dim, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::cat(out, tensors, dim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_CAT_OUT, out, tensors, dim);
}

at::Tensor wrap_cat_names(at::TensorList tensors, at::Dimname dim) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::cat(tensors, dim));
}

at::Tensor & wrap_cat_names_out(at::TensorList tensors, at::Dimname dim, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::cat(out, tensors, dim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_CAT_NAMES_OUT, out, tensors, dim);
}

at::Tensor wrap_block_diag(at::TensorList tensors) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::block_diag(tensors));
}

at::Tensor wrap_ceil(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::ceil(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CEIL, self);
}

at::Tensor & wrap_ceil_(at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::ceil_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CEIL_, self);
}

at::Tensor & wrap_ceil_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::ceil(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_CEIL_OUT, out, self);
}

at::Tensor wrap_chain_matmul(at::TensorList matrices) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::chain_matmul(matrices));
}

at::Tensor & wrap_chain_matmul_out(at::TensorList matrices, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::chain_matmul(out, matrices);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_CHAIN_MATMUL_OUT, out, matrices);
}

std::vector<at::Tensor> wrap_unsafe_chunk(const at::Tensor & self, int64_t chunks, int64_t dim) {
  ensure_materialized(self);
  return at::redispatch::unsafe_chunk(self, chunks, dim);
}

std::vector<at::Tensor> wrap_chunk(const at::Tensor & self, int64_t chunks, int64_t dim) {
  ensure_materialized(self);
  return at::redispatch::chunk(self, chunks, dim);
}

std::vector<at::Tensor> wrap_tensor_split_sections(const at::Tensor & self, int64_t sections, int64_t dim) {
  ensure_materialized(self);
  return at::redispatch::tensor_split(self, sections, dim);
}

std::vector<at::Tensor> wrap_tensor_split_indices(const at::Tensor & self, at::IntArrayRef indices, int64_t dim) {
  ensure_materialized(self);
  return at::redispatch::tensor_split(self, indices, dim);
}

std::vector<at::Tensor> wrap_tensor_split_tensor_indices_or_sections(const at::Tensor & self, const at::Tensor & tensor_indices_or_sections, int64_t dim) {
  ensure_materialized(self, tensor_indices_or_sections);
  return at::redispatch::tensor_split(self, tensor_indices_or_sections, dim);
}

at::Tensor wrap_clamp(const at::Tensor & self, const c10::optional<at::Scalar> & min, const c10::optional<at::Scalar> & max) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::clamp(self, min, max);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CLAMP, self, min, max);
}

at::Tensor & wrap_clamp_(at::Tensor & self, const c10::optional<at::Scalar> & min, const c10::optional<at::Scalar> & max) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::clamp_(self, min, max);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CLAMP_, self, min, max);
}

at::Tensor & wrap_clamp_out(const at::Tensor & self, const c10::optional<at::Scalar> & min, const c10::optional<at::Scalar> & max, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::clamp(out, self, min, max);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_CLAMP_OUT, out, self, min, max);
}

at::Tensor wrap_clamp_max(const at::Tensor & self, const at::Scalar & max) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::clamp_max(self, max);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CLAMP_MAX, self, max);
}

at::Tensor & wrap_clamp_max_(at::Tensor & self, const at::Scalar & max) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::clamp_max_(self, max);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CLAMP_MAX_, self, max);
}

at::Tensor & wrap_clamp_max_out(const at::Tensor & self, const at::Scalar & max, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::clamp_max(out, self, max);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_CLAMP_MAX_OUT, out, self, max);
}

at::Tensor wrap_clamp_min(const at::Tensor & self, const at::Scalar & min) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::clamp_min(self, min);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CLAMP_MIN, self, min);
}

at::Tensor & wrap_clamp_min_(at::Tensor & self, const at::Scalar & min) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::clamp_min_(self, min);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CLAMP_MIN_, self, min);
}

at::Tensor & wrap_clamp_min_out(const at::Tensor & self, const at::Scalar & min, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::clamp_min(out, self, min);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_CLAMP_MIN_OUT, out, self, min);
}

at::Tensor wrap_clip(const at::Tensor & self, const c10::optional<at::Scalar> & min, const c10::optional<at::Scalar> & max) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::clip(self, min, max);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CLIP, self, min, max);
}

at::Tensor & wrap_clip_(at::Tensor & self, const c10::optional<at::Scalar> & min, const c10::optional<at::Scalar> & max) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::clip_(self, min, max);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CLIP_, self, min, max);
}

at::Tensor & wrap_clip_out(const at::Tensor & self, const c10::optional<at::Scalar> & min, const c10::optional<at::Scalar> & max, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::clip(out, self, min, max);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_CLIP_OUT, out, self, min, max);
}

bool wrap_cudnn_is_acceptable(const at::Tensor & self) {
  ensure_materialized(self);
  return at::redispatch::cudnn_is_acceptable(self);
}

at::Tensor wrap_complex(const at::Tensor & real, const at::Tensor & imag) {
  if (trace.is_flushing()) {
    ensure_materialized(real, imag);
    return at::redispatch::complex(real, imag);
  }
  return MK_TORCHY(real.dtype(), real.device(), H_COMPLEX, real, imag);
}

at::Tensor & wrap_complex_out(const at::Tensor & real, const at::Tensor & imag, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, real, imag);
    return at::redispatch::complex(out, real, imag);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_COMPLEX_OUT, out, real, imag);
}

at::Tensor wrap_polar(const at::Tensor & abs, const at::Tensor & angle) {
  if (trace.is_flushing()) {
    ensure_materialized(abs, angle);
    return at::redispatch::polar(abs, angle);
  }
  return MK_TORCHY(abs.dtype(), abs.device(), H_POLAR, abs, angle);
}

at::Tensor & wrap_polar_out(const at::Tensor & abs, const at::Tensor & angle, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, abs, angle);
    return at::redispatch::polar(out, abs, angle);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_POLAR_OUT, out, abs, angle);
}

at::Tensor wrap_constant_pad_nd(const at::Tensor & self, at::IntArrayRef pad, const at::Scalar & value) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::constant_pad_nd(self, pad, value);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CONSTANT_PAD_ND, self, pad, value);
}

at::Tensor wrap_convolution(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups) {
  if (trace.is_flushing()) {
    ensure_materialized(input, weight);
    return at::redispatch::convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_CONVOLUTION, input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
}

at::Tensor wrap_convolution_overrideable(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups) {
  if (trace.is_flushing()) {
    ensure_materialized(input, weight);
    return at::redispatch::convolution_overrideable(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_CONVOLUTION_OVERRIDEABLE, input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> wrap_convolution_backward_overrideable(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & weight, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups, std::array<bool,3> output_mask) {
  ensure_materialized(grad_output, input, weight);
  return at::redispatch::convolution_backward_overrideable(grad_output, input, weight, stride, padding, dilation, transposed, output_padding, groups, output_mask);
}

at::Tensor wrap__convolution(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32) {
  if (trace.is_flushing()) {
    ensure_materialized(input, weight);
    return at::redispatch::_convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, allow_tf32);
  }
  return MK_TORCHY(input.dtype(), input.device(), H__CONVOLUTION, input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, allow_tf32);
}

at::Tensor wrap__convolution_deprecated(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled) {
  if (trace.is_flushing()) {
    ensure_materialized(input, weight);
    return at::redispatch::_convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled);
  }
  return MK_TORCHY(input.dtype(), input.device(), H__CONVOLUTION_DEPRECATED, input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled);
}

at::Tensor wrap__convolution_mode(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, std::string padding, at::IntArrayRef dilation, int64_t groups) {
  if (trace.is_flushing()) {
    ensure_materialized(input, weight);
    return at::redispatch::_convolution_mode(input, weight, bias, stride, padding, dilation, groups);
  }
  return MK_TORCHY(input.dtype(), input.device(), H__CONVOLUTION_MODE, input, weight, bias, stride, padding, dilation, groups);
}

at::Tensor wrap__convolution_nogroup(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding) {
  if (trace.is_flushing()) {
    ensure_materialized(input, weight);
    return at::redispatch::_convolution_nogroup(input, weight, bias, stride, padding, dilation, transposed, output_padding);
  }
  return MK_TORCHY(input.dtype(), input.device(), H__CONVOLUTION_NOGROUP, input, weight, bias, stride, padding, dilation, transposed, output_padding);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> wrap__convolution_double_backward(const c10::optional<at::Tensor> & ggI, const c10::optional<at::Tensor> & ggW, const c10::optional<at::Tensor> & ggb, const at::Tensor & gO, const at::Tensor & weight, const at::Tensor & self, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32, std::array<bool,3> output_mask) {
  ensure_materialized(gO, weight, self);
  return at::redispatch::_convolution_double_backward(ggI, ggW, ggb, gO, weight, self, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, allow_tf32, output_mask);
}

at::Tensor wrap_conv1d(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups) {
  if (trace.is_flushing()) {
    ensure_materialized(input, weight);
    return at::redispatch::conv1d(input, weight, bias, stride, padding, dilation, groups);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_CONV1D, input, weight, bias, stride, padding, dilation, groups);
}

at::Tensor wrap_conv2d(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups) {
  if (trace.is_flushing()) {
    ensure_materialized(input, weight);
    return at::redispatch::conv2d(input, weight, bias, stride, padding, dilation, groups);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_CONV2D, input, weight, bias, stride, padding, dilation, groups);
}

at::Tensor wrap_conv3d(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups) {
  if (trace.is_flushing()) {
    ensure_materialized(input, weight);
    return at::redispatch::conv3d(input, weight, bias, stride, padding, dilation, groups);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_CONV3D, input, weight, bias, stride, padding, dilation, groups);
}

at::Tensor wrap_conv1d_padding(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, std::string padding, at::IntArrayRef dilation, int64_t groups) {
  if (trace.is_flushing()) {
    ensure_materialized(input, weight);
    return at::redispatch::conv1d(input, weight, bias, stride, padding, dilation, groups);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_CONV1D_PADDING, input, weight, bias, stride, padding, dilation, groups);
}

at::Tensor wrap_conv2d_padding(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, std::string padding, at::IntArrayRef dilation, int64_t groups) {
  if (trace.is_flushing()) {
    ensure_materialized(input, weight);
    return at::redispatch::conv2d(input, weight, bias, stride, padding, dilation, groups);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_CONV2D_PADDING, input, weight, bias, stride, padding, dilation, groups);
}

at::Tensor wrap_conv3d_padding(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, std::string padding, at::IntArrayRef dilation, int64_t groups) {
  if (trace.is_flushing()) {
    ensure_materialized(input, weight);
    return at::redispatch::conv3d(input, weight, bias, stride, padding, dilation, groups);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_CONV3D_PADDING, input, weight, bias, stride, padding, dilation, groups);
}

at::Tensor wrap_conv_tbc(const at::Tensor & self, const at::Tensor & weight, const at::Tensor & bias, int64_t pad) {
  if (trace.is_flushing()) {
    ensure_materialized(self, weight, bias);
    return at::redispatch::conv_tbc(self, weight, bias, pad);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CONV_TBC, self, weight, bias, pad);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> wrap_conv_tbc_backward(const at::Tensor & self, const at::Tensor & input, const at::Tensor & weight, const at::Tensor & bias, int64_t pad) {
  ensure_materialized(self, input, weight, bias);
  return at::redispatch::conv_tbc_backward(self, input, weight, bias, pad);
}

at::Tensor wrap_conv_transpose1d(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, int64_t groups, at::IntArrayRef dilation) {
  if (trace.is_flushing()) {
    ensure_materialized(input, weight);
    return at::redispatch::conv_transpose1d(input, weight, bias, stride, padding, output_padding, groups, dilation);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_CONV_TRANSPOSE1D, input, weight, bias, stride, padding, output_padding, groups, dilation);
}

at::Tensor wrap_conv_transpose2d_input(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, int64_t groups, at::IntArrayRef dilation) {
  if (trace.is_flushing()) {
    ensure_materialized(input, weight);
    return at::redispatch::conv_transpose2d(input, weight, bias, stride, padding, output_padding, groups, dilation);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_CONV_TRANSPOSE2D_INPUT, input, weight, bias, stride, padding, output_padding, groups, dilation);
}

at::Tensor wrap_conv_transpose3d_input(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, int64_t groups, at::IntArrayRef dilation) {
  if (trace.is_flushing()) {
    ensure_materialized(input, weight);
    return at::redispatch::conv_transpose3d(input, weight, bias, stride, padding, output_padding, groups, dilation);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_CONV_TRANSPOSE3D_INPUT, input, weight, bias, stride, padding, output_padding, groups, dilation);
}

at::Tensor & wrap_copy_(at::Tensor & self, const at::Tensor & src, bool non_blocking) {
  if (trace.is_flushing()) {
    ensure_materialized(self, src);
    return at::redispatch::copy_(self, src, non_blocking);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_COPY_, self, src, non_blocking);
}

at::Tensor & wrap_cos_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::cos(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_COS_OUT, out, self);
}

at::Tensor & wrap_cosh_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::cosh(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_COSH_OUT, out, self);
}

at::Tensor wrap_cosine_embedding_loss(const at::Tensor & input1, const at::Tensor & input2, const at::Tensor & target, double margin, int64_t reduction) {
  if (trace.is_flushing()) {
    ensure_materialized(input1, input2, target);
    return at::redispatch::cosine_embedding_loss(input1, input2, target, margin, reduction);
  }
  return MK_TORCHY(input1.dtype(), input1.device(), H_COSINE_EMBEDDING_LOSS, input1, input2, target, margin, reduction);
}

at::Tensor wrap_count_nonzero_dim_IntList(const at::Tensor & self, at::IntArrayRef dim) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::count_nonzero(self, dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_COUNT_NONZERO_DIM_INTLIST, self, dim);
}

at::Tensor wrap_count_nonzero(const at::Tensor & self, c10::optional<int64_t> dim) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::count_nonzero(self, dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_COUNT_NONZERO, self, dim);
}

at::Tensor wrap_cudnn_affine_grid_generator(const at::Tensor & theta, int64_t N, int64_t C, int64_t H, int64_t W) {
  if (trace.is_flushing()) {
    ensure_materialized(theta);
    return at::redispatch::cudnn_affine_grid_generator(theta, N, C, H, W);
  }
  return MK_TORCHY(theta.dtype(), theta.device(), H_CUDNN_AFFINE_GRID_GENERATOR, theta, N, C, H, W);
}

at::Tensor wrap_cudnn_affine_grid_generator_backward(const at::Tensor & grad, int64_t N, int64_t C, int64_t H, int64_t W) {
  if (trace.is_flushing()) {
    ensure_materialized(grad);
    return at::redispatch::cudnn_affine_grid_generator_backward(grad, N, C, H, W);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H_CUDNN_AFFINE_GRID_GENERATOR_BACKWARD, grad, N, C, H, W);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> wrap_cudnn_batch_norm(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, bool training, double exponential_average_factor, double epsilon) {
  ensure_materialized(input, weight);
  return at::redispatch::cudnn_batch_norm(input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> wrap_cudnn_batch_norm_backward(const at::Tensor & input, const at::Tensor & grad_output, const at::Tensor & weight, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, const c10::optional<at::Tensor> & save_mean, const c10::optional<at::Tensor> & save_var, double epsilon, const at::Tensor & reserveSpace) {
  ensure_materialized(input, grad_output, weight, reserveSpace);
  return at::redispatch::cudnn_batch_norm_backward(input, grad_output, weight, running_mean, running_var, save_mean, save_var, epsilon, reserveSpace);
}

at::Tensor wrap_cudnn_convolution_deprecated(const at::Tensor & self, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  if (trace.is_flushing()) {
    ensure_materialized(self, weight);
    return at::redispatch::cudnn_convolution(self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CUDNN_CONVOLUTION_DEPRECATED, self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic);
}

at::Tensor wrap_cudnn_convolution_deprecated2(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  if (trace.is_flushing()) {
    ensure_materialized(self, weight);
    return at::redispatch::cudnn_convolution(self, weight, padding, stride, dilation, groups, benchmark, deterministic);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CUDNN_CONVOLUTION_DEPRECATED2, self, weight, padding, stride, dilation, groups, benchmark, deterministic);
}

at::Tensor wrap_cudnn_convolution(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, bool allow_tf32) {
  if (trace.is_flushing()) {
    ensure_materialized(self, weight);
    return at::redispatch::cudnn_convolution(self, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CUDNN_CONVOLUTION, self, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
}

at::Tensor wrap_cudnn_convolution_backward_input(at::IntArrayRef self_size, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, bool allow_tf32) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, weight);
    return at::redispatch::cudnn_convolution_backward_input(self_size, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_CUDNN_CONVOLUTION_BACKWARD_INPUT, self_size, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
}

std::tuple<at::Tensor,at::Tensor> wrap_cudnn_convolution_backward(const at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, bool allow_tf32, std::array<bool,2> output_mask) {
  ensure_materialized(self, grad_output, weight);
  return at::redispatch::cudnn_convolution_backward(self, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32, output_mask);
}

at::Tensor wrap_cudnn_convolution_backward_weight(at::IntArrayRef weight_size, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, bool allow_tf32) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self);
    return at::redispatch::cudnn_convolution_backward_weight(weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_CUDNN_CONVOLUTION_BACKWARD_WEIGHT, weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
}

at::Tensor wrap_cudnn_convolution_transpose_deprecated(const at::Tensor & self, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  if (trace.is_flushing()) {
    ensure_materialized(self, weight);
    return at::redispatch::cudnn_convolution_transpose(self, weight, bias, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CUDNN_CONVOLUTION_TRANSPOSE_DEPRECATED, self, weight, bias, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
}

at::Tensor wrap_cudnn_convolution_transpose_deprecated2(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  if (trace.is_flushing()) {
    ensure_materialized(self, weight);
    return at::redispatch::cudnn_convolution_transpose(self, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CUDNN_CONVOLUTION_TRANSPOSE_DEPRECATED2, self, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
}

at::Tensor wrap_cudnn_convolution_transpose(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, bool allow_tf32) {
  if (trace.is_flushing()) {
    ensure_materialized(self, weight);
    return at::redispatch::cudnn_convolution_transpose(self, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CUDNN_CONVOLUTION_TRANSPOSE, self, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
}

std::tuple<at::Tensor,at::Tensor> wrap_cudnn_convolution_transpose_backward(const at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, bool allow_tf32, std::array<bool,2> output_mask) {
  ensure_materialized(self, grad_output, weight);
  return at::redispatch::cudnn_convolution_transpose_backward(self, grad_output, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic, allow_tf32, output_mask);
}

at::Tensor wrap_cudnn_convolution_transpose_backward_input(const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, bool allow_tf32) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, weight);
    return at::redispatch::cudnn_convolution_transpose_backward_input(grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_CUDNN_CONVOLUTION_TRANSPOSE_BACKWARD_INPUT, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
}

at::Tensor wrap_cudnn_convolution_transpose_backward_weight(at::IntArrayRef weight_size, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, bool allow_tf32) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self);
    return at::redispatch::cudnn_convolution_transpose_backward_weight(weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_CUDNN_CONVOLUTION_TRANSPOSE_BACKWARD_WEIGHT, weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
}

at::Tensor wrap_cudnn_convolution_relu(const at::Tensor & self, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups) {
  if (trace.is_flushing()) {
    ensure_materialized(self, weight);
    return at::redispatch::cudnn_convolution_relu(self, weight, bias, stride, padding, dilation, groups);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CUDNN_CONVOLUTION_RELU, self, weight, bias, stride, padding, dilation, groups);
}

at::Tensor wrap_cudnn_convolution_add_relu(const at::Tensor & self, const at::Tensor & weight, const at::Tensor & z, const c10::optional<at::Scalar> & alpha, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups) {
  if (trace.is_flushing()) {
    ensure_materialized(self, weight, z);
    return at::redispatch::cudnn_convolution_add_relu(self, weight, z, alpha, bias, stride, padding, dilation, groups);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CUDNN_CONVOLUTION_ADD_RELU, self, weight, z, alpha, bias, stride, padding, dilation, groups);
}

at::Tensor wrap_cudnn_grid_sampler(const at::Tensor & self, const at::Tensor & grid) {
  if (trace.is_flushing()) {
    ensure_materialized(self, grid);
    return at::redispatch::cudnn_grid_sampler(self, grid);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CUDNN_GRID_SAMPLER, self, grid);
}

std::tuple<at::Tensor,at::Tensor> wrap_cudnn_grid_sampler_backward(const at::Tensor & self, const at::Tensor & grid, const at::Tensor & grad_output) {
  ensure_materialized(self, grid, grad_output);
  return at::redispatch::cudnn_grid_sampler_backward(self, grid, grad_output);
}

std::tuple<at::Tensor,at::Tensor> wrap_cummax(const at::Tensor & self, int64_t dim) {
  ensure_materialized(self);
  return at::redispatch::cummax(self, dim);
}

std::tuple<at::Tensor &,at::Tensor &> wrap_cummax_out(const at::Tensor & self, int64_t dim, at::Tensor & values, at::Tensor & indices) {
  ensure_materialized(values, indices, self);
  return at::redispatch::cummax(values, indices, self, dim);
}

std::tuple<at::Tensor,at::Tensor> wrap_cummax_dimname(const at::Tensor & self, at::Dimname dim) {
  ensure_materialized(self);
  return at::redispatch::cummax(self, dim);
}

std::tuple<at::Tensor &,at::Tensor &> wrap_cummax_dimname_out(const at::Tensor & self, at::Dimname dim, at::Tensor & values, at::Tensor & indices) {
  ensure_materialized(values, indices, self);
  return at::redispatch::cummax(values, indices, self, dim);
}

void wrap__cummax_helper(const at::Tensor & self, at::Tensor & values, at::Tensor & indices, int64_t dim) {
  ensure_materialized(self, values, indices);
  return at::redispatch::_cummax_helper(self, values, indices, dim);
}

std::tuple<at::Tensor,at::Tensor> wrap_cummin(const at::Tensor & self, int64_t dim) {
  ensure_materialized(self);
  return at::redispatch::cummin(self, dim);
}

std::tuple<at::Tensor &,at::Tensor &> wrap_cummin_out(const at::Tensor & self, int64_t dim, at::Tensor & values, at::Tensor & indices) {
  ensure_materialized(values, indices, self);
  return at::redispatch::cummin(values, indices, self, dim);
}

std::tuple<at::Tensor,at::Tensor> wrap_cummin_dimname(const at::Tensor & self, at::Dimname dim) {
  ensure_materialized(self);
  return at::redispatch::cummin(self, dim);
}

std::tuple<at::Tensor &,at::Tensor &> wrap_cummin_dimname_out(const at::Tensor & self, at::Dimname dim, at::Tensor & values, at::Tensor & indices) {
  ensure_materialized(values, indices, self);
  return at::redispatch::cummin(values, indices, self, dim);
}

void wrap__cummin_helper(const at::Tensor & self, at::Tensor & values, at::Tensor & indices, int64_t dim) {
  ensure_materialized(self, values, indices);
  return at::redispatch::_cummin_helper(self, values, indices, dim);
}

at::Tensor wrap_cummaxmin_backward(const at::Tensor & grad, const at::Tensor & input, const at::Tensor & indices, int64_t dim) {
  if (trace.is_flushing()) {
    ensure_materialized(grad, input, indices);
    return at::redispatch::cummaxmin_backward(grad, input, indices, dim);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H_CUMMAXMIN_BACKWARD, grad, input, indices, dim);
}

at::Tensor wrap_cumprod(const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::cumprod(self, dim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CUMPROD, self, dim, dtype);
}

at::Tensor & wrap_cumprod_(at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::cumprod_(self, dim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CUMPROD_, self, dim, dtype);
}

at::Tensor & wrap_cumprod_out(const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::cumprod(out, self, dim, dtype);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_CUMPROD_OUT, out, self, dim, dtype);
}

at::Tensor wrap_cumprod_dimname(const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::cumprod(self, dim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CUMPROD_DIMNAME, self, dim, dtype);
}

at::Tensor & wrap_cumprod__dimname(at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::cumprod_(self, dim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CUMPROD__DIMNAME, self, dim, dtype);
}

at::Tensor & wrap_cumprod_dimname_out(const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::cumprod(out, self, dim, dtype);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_CUMPROD_DIMNAME_OUT, out, self, dim, dtype);
}

at::Tensor wrap_cumprod_backward(const at::Tensor & grad, const at::Tensor & input, int64_t dim, const at::Tensor & output) {
  if (trace.is_flushing()) {
    ensure_materialized(grad, input, output);
    return at::redispatch::cumprod_backward(grad, input, dim, output);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H_CUMPROD_BACKWARD, grad, input, dim, output);
}

at::Tensor wrap_cumsum(const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::cumsum(self, dim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CUMSUM, self, dim, dtype);
}

at::Tensor & wrap_cumsum_(at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::cumsum_(self, dim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CUMSUM_, self, dim, dtype);
}

at::Tensor & wrap_cumsum_out(const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::cumsum(out, self, dim, dtype);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_CUMSUM_OUT, out, self, dim, dtype);
}

at::Tensor wrap_cumsum_dimname(const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::cumsum(self, dim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CUMSUM_DIMNAME, self, dim, dtype);
}

at::Tensor & wrap_cumsum__dimname(at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::cumsum_(self, dim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CUMSUM__DIMNAME, self, dim, dtype);
}

at::Tensor & wrap_cumsum_dimname_out(const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::cumsum(out, self, dim, dtype);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_CUMSUM_DIMNAME_OUT, out, self, dim, dtype);
}

at::Tensor wrap_ctc_loss_IntList(const at::Tensor & log_probs, const at::Tensor & targets, at::IntArrayRef input_lengths, at::IntArrayRef target_lengths, int64_t blank, int64_t reduction, bool zero_infinity) {
  if (trace.is_flushing()) {
    ensure_materialized(log_probs, targets);
    return at::redispatch::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity);
  }
  return MK_TORCHY(log_probs.dtype(), log_probs.device(), H_CTC_LOSS_INTLIST, log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity);
}

at::Tensor wrap_ctc_loss_Tensor(const at::Tensor & log_probs, const at::Tensor & targets, const at::Tensor & input_lengths, const at::Tensor & target_lengths, int64_t blank, int64_t reduction, bool zero_infinity) {
  if (trace.is_flushing()) {
    ensure_materialized(log_probs, targets, input_lengths, target_lengths);
    return at::redispatch::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity);
  }
  return MK_TORCHY(log_probs.dtype(), log_probs.device(), H_CTC_LOSS_TENSOR, log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity);
}

std::tuple<at::Tensor,at::Tensor> wrap__ctc_loss(const at::Tensor & log_probs, const at::Tensor & targets, at::IntArrayRef input_lengths, at::IntArrayRef target_lengths, int64_t blank, bool zero_infinity) {
  ensure_materialized(log_probs, targets);
  return at::redispatch::_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, zero_infinity);
}

at::Tensor wrap__ctc_loss_backward(const at::Tensor & grad, const at::Tensor & log_probs, const at::Tensor & targets, at::IntArrayRef input_lengths, at::IntArrayRef target_lengths, const at::Tensor & neg_log_likelihood, const at::Tensor & log_alpha, int64_t blank, bool zero_infinity) {
  if (trace.is_flushing()) {
    ensure_materialized(grad, log_probs, targets, neg_log_likelihood, log_alpha);
    return at::redispatch::_ctc_loss_backward(grad, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, blank, zero_infinity);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H__CTC_LOSS_BACKWARD, grad, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, blank, zero_infinity);
}

at::Tensor wrap_diag_embed(const at::Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::diag_embed(self, offset, dim1, dim2);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DIAG_EMBED, self, offset, dim1, dim2);
}

at::Tensor wrap_diagflat(const at::Tensor & self, int64_t offset) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::diagflat(self, offset);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DIAGFLAT, self, offset);
}

at::Tensor wrap_diagonal(const at::Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::diagonal(self, offset, dim1, dim2);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DIAGONAL, self, offset, dim1, dim2);
}

at::Tensor wrap_diagonal_Dimname(const at::Tensor & self, at::Dimname outdim, at::Dimname dim1, at::Dimname dim2, int64_t offset) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::diagonal(self, outdim, dim1, dim2, offset);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DIAGONAL_DIMNAME, self, outdim, dim1, dim2, offset);
}

at::Tensor wrap_diagonal_backward(const at::Tensor & grad, at::IntArrayRef input_sizes, int64_t offset, int64_t dim1, int64_t dim2) {
  if (trace.is_flushing()) {
    ensure_materialized(grad);
    return at::redispatch::diagonal_backward(grad, input_sizes, offset, dim1, dim2);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H_DIAGONAL_BACKWARD, grad, input_sizes, offset, dim1, dim2);
}

at::Tensor & wrap_fill_diagonal_(at::Tensor & self, const at::Scalar & fill_value, bool wrap) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::fill_diagonal_(self, fill_value, wrap);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FILL_DIAGONAL_, self, fill_value, wrap);
}

at::Tensor wrap_diff(const at::Tensor & self, int64_t n, int64_t dim, const c10::optional<at::Tensor> & prepend, const c10::optional<at::Tensor> & append) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::diff(self, n, dim, prepend, append);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DIFF, self, n, dim, prepend, append);
}

at::Tensor & wrap_diff_out(const at::Tensor & self, int64_t n, int64_t dim, const c10::optional<at::Tensor> & prepend, const c10::optional<at::Tensor> & append, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::diff(out, self, n, dim, prepend, append);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_DIFF_OUT, out, self, n, dim, prepend, append);
}

at::Tensor wrap_div_Tensor(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::div(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DIV_TENSOR, self, other);
}

at::Tensor & wrap_div__Tensor(at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::div_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DIV__TENSOR, self, other);
}

at::Tensor & wrap_div_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::div(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_DIV_OUT, out, self, other);
}

at::Tensor & wrap_div_out_mode(const at::Tensor & self, const at::Tensor & other, c10::optional<std::string> rounding_mode, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::div(out, self, other, rounding_mode);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_DIV_OUT_MODE, out, self, other, rounding_mode);
}

at::Tensor wrap_div_Scalar(const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::div(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DIV_SCALAR, self, other);
}

at::Tensor & wrap_div__Scalar(at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::div_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DIV__SCALAR, self, other);
}

at::Tensor wrap_div_Scalar_mode(const at::Tensor & self, const at::Scalar & other, c10::optional<std::string> rounding_mode) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::div(self, other, rounding_mode);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DIV_SCALAR_MODE, self, other, rounding_mode);
}

at::Tensor & wrap_div__Scalar_mode(at::Tensor & self, const at::Scalar & other, c10::optional<std::string> rounding_mode) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::div_(self, other, rounding_mode);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DIV__SCALAR_MODE, self, other, rounding_mode);
}

at::Tensor wrap_divide_Tensor(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::divide(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DIVIDE_TENSOR, self, other);
}

at::Tensor & wrap_divide__Tensor(at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::divide_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DIVIDE__TENSOR, self, other);
}

at::Tensor & wrap_divide_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::divide(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_DIVIDE_OUT, out, self, other);
}

at::Tensor wrap_divide_Scalar(const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::divide(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DIVIDE_SCALAR, self, other);
}

at::Tensor & wrap_divide__Scalar(at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::divide_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DIVIDE__SCALAR, self, other);
}

at::Tensor wrap_divide_Tensor_mode(const at::Tensor & self, const at::Tensor & other, c10::optional<std::string> rounding_mode) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::divide(self, other, rounding_mode);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DIVIDE_TENSOR_MODE, self, other, rounding_mode);
}

at::Tensor & wrap_divide__Tensor_mode(at::Tensor & self, const at::Tensor & other, c10::optional<std::string> rounding_mode) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::divide_(self, other, rounding_mode);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DIVIDE__TENSOR_MODE, self, other, rounding_mode);
}

at::Tensor & wrap_divide_out_mode(const at::Tensor & self, const at::Tensor & other, c10::optional<std::string> rounding_mode, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::divide(out, self, other, rounding_mode);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_DIVIDE_OUT_MODE, out, self, other, rounding_mode);
}

at::Tensor wrap_divide_Scalar_mode(const at::Tensor & self, const at::Scalar & other, c10::optional<std::string> rounding_mode) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::divide(self, other, rounding_mode);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DIVIDE_SCALAR_MODE, self, other, rounding_mode);
}

at::Tensor & wrap_divide__Scalar_mode(at::Tensor & self, const at::Scalar & other, c10::optional<std::string> rounding_mode) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::divide_(self, other, rounding_mode);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DIVIDE__SCALAR_MODE, self, other, rounding_mode);
}

at::Tensor wrap_true_divide_Tensor(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::true_divide(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TRUE_DIVIDE_TENSOR, self, other);
}

at::Tensor & wrap_true_divide__Tensor(at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::true_divide_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TRUE_DIVIDE__TENSOR, self, other);
}

at::Tensor & wrap_true_divide_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::true_divide(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_TRUE_DIVIDE_OUT, out, self, other);
}

at::Tensor wrap_true_divide_Scalar(const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::true_divide(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TRUE_DIVIDE_SCALAR, self, other);
}

at::Tensor & wrap_true_divide__Scalar(at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::true_divide_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TRUE_DIVIDE__SCALAR, self, other);
}

at::Tensor wrap_dot(const at::Tensor & self, const at::Tensor & tensor) {
  if (trace.is_flushing()) {
    ensure_materialized(self, tensor);
    return at::redispatch::dot(self, tensor);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DOT, self, tensor);
}

at::Tensor & wrap_dot_out(const at::Tensor & self, const at::Tensor & tensor, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, tensor);
    return at::redispatch::dot(out, self, tensor);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_DOT_OUT, out, self, tensor);
}

at::Tensor wrap_vdot(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::vdot(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_VDOT, self, other);
}

at::Tensor & wrap_vdot_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::vdot(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_VDOT_OUT, out, self, other);
}

at::Tensor wrap_einsum(std::string equation, at::TensorList tensors) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::einsum(equation, tensors));
}

at::Tensor wrap_embedding(const at::Tensor & weight, const at::Tensor & indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
  if (trace.is_flushing()) {
    ensure_materialized(weight, indices);
    return at::redispatch::embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse);
  }
  return MK_TORCHY(weight.dtype(), weight.device(), H_EMBEDDING, weight, indices, padding_idx, scale_grad_by_freq, sparse);
}

at::Tensor wrap_embedding_backward(const at::Tensor & grad, const at::Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
  if (trace.is_flushing()) {
    ensure_materialized(grad, indices);
    return at::redispatch::embedding_backward(grad, indices, num_weights, padding_idx, scale_grad_by_freq, sparse);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H_EMBEDDING_BACKWARD, grad, indices, num_weights, padding_idx, scale_grad_by_freq, sparse);
}

at::Tensor wrap_embedding_dense_backward(const at::Tensor & grad_output, const at::Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, indices);
    return at::redispatch::embedding_dense_backward(grad_output, indices, num_weights, padding_idx, scale_grad_by_freq);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_EMBEDDING_DENSE_BACKWARD, grad_output, indices, num_weights, padding_idx, scale_grad_by_freq);
}

at::Tensor & wrap_embedding_renorm_(at::Tensor & self, const at::Tensor & indices, double max_norm, double norm_type) {
  if (trace.is_flushing()) {
    ensure_materialized(self, indices);
    return at::redispatch::embedding_renorm_(self, indices, max_norm, norm_type);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_EMBEDDING_RENORM_, self, indices, max_norm, norm_type);
}

at::Tensor wrap_embedding_sparse_backward(const at::Tensor & grad, const at::Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) {
  if (trace.is_flushing()) {
    ensure_materialized(grad, indices);
    return at::redispatch::embedding_sparse_backward(grad, indices, num_weights, padding_idx, scale_grad_by_freq);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H_EMBEDDING_SPARSE_BACKWARD, grad, indices, num_weights, padding_idx, scale_grad_by_freq);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> wrap__embedding_bag_forward_only(const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const c10::optional<at::Tensor> & per_sample_weights, bool include_last_offset, int64_t padding_idx) {
  ensure_materialized(weight, indices, offsets);
  return at::redispatch::_embedding_bag_forward_only(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx);
}

std::tuple<at::Tensor,at::Tensor> wrap__rowwise_prune(const at::Tensor & weight, const at::Tensor & mask, at::ScalarType compressed_indices_dtype) {
  ensure_materialized(weight, mask);
  return at::redispatch::_rowwise_prune(weight, mask, compressed_indices_dtype);
}

at::Tensor wrap_row_stack(at::TensorList tensors) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::row_stack(tensors));
}

at::Tensor & wrap_row_stack_out(at::TensorList tensors, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::row_stack(out, tensors);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ROW_STACK_OUT, out, tensors);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> wrap_embedding_bag(const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const c10::optional<at::Tensor> & per_sample_weights, bool include_last_offset) {
  ensure_materialized(weight, indices, offsets);
  return at::redispatch::embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> wrap_embedding_bag_padding_idx(const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const c10::optional<at::Tensor> & per_sample_weights, bool include_last_offset, c10::optional<int64_t> padding_idx) {
  ensure_materialized(weight, indices, offsets);
  return at::redispatch::embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> wrap__embedding_bag(const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const c10::optional<at::Tensor> & per_sample_weights, bool include_last_offset, int64_t padding_idx) {
  ensure_materialized(weight, indices, offsets);
  return at::redispatch::_embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx);
}

at::Tensor wrap__embedding_bag_backward(const at::Tensor & grad, const at::Tensor & indices, const at::Tensor & offsets, const at::Tensor & offset2bag, const at::Tensor & bag_size, const at::Tensor & maximum_indices, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, bool sparse, const c10::optional<at::Tensor> & per_sample_weights, int64_t padding_idx) {
  if (trace.is_flushing()) {
    ensure_materialized(grad, indices, offsets, offset2bag, bag_size, maximum_indices);
    return at::redispatch::_embedding_bag_backward(grad, indices, offsets, offset2bag, bag_size, maximum_indices, num_weights, scale_grad_by_freq, mode, sparse, per_sample_weights, padding_idx);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H__EMBEDDING_BAG_BACKWARD, grad, indices, offsets, offset2bag, bag_size, maximum_indices, num_weights, scale_grad_by_freq, mode, sparse, per_sample_weights, padding_idx);
}

at::Tensor wrap__embedding_bag_sparse_backward(const at::Tensor & grad, const at::Tensor & indices, const at::Tensor & offsets, const at::Tensor & offset2bag, const at::Tensor & bag_size, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, const c10::optional<at::Tensor> & per_sample_weights, int64_t padding_idx) {
  if (trace.is_flushing()) {
    ensure_materialized(grad, indices, offsets, offset2bag, bag_size);
    return at::redispatch::_embedding_bag_sparse_backward(grad, indices, offsets, offset2bag, bag_size, num_weights, scale_grad_by_freq, mode, per_sample_weights, padding_idx);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H__EMBEDDING_BAG_SPARSE_BACKWARD, grad, indices, offsets, offset2bag, bag_size, num_weights, scale_grad_by_freq, mode, per_sample_weights, padding_idx);
}

at::Tensor wrap__embedding_bag_dense_backward(const at::Tensor & grad, const at::Tensor & indices, const at::Tensor & offset2bag, const at::Tensor & bag_size, const at::Tensor & maximum_indices, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, const c10::optional<at::Tensor> & per_sample_weights, int64_t padding_idx) {
  if (trace.is_flushing()) {
    ensure_materialized(grad, indices, offset2bag, bag_size, maximum_indices);
    return at::redispatch::_embedding_bag_dense_backward(grad, indices, offset2bag, bag_size, maximum_indices, num_weights, scale_grad_by_freq, mode, per_sample_weights, padding_idx);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H__EMBEDDING_BAG_DENSE_BACKWARD, grad, indices, offset2bag, bag_size, maximum_indices, num_weights, scale_grad_by_freq, mode, per_sample_weights, padding_idx);
}

at::Tensor wrap__embedding_bag_per_sample_weights_backward(const at::Tensor & grad, const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, const at::Tensor & offset2bag, int64_t mode, int64_t padding_idx) {
  if (trace.is_flushing()) {
    ensure_materialized(grad, weight, indices, offsets, offset2bag);
    return at::redispatch::_embedding_bag_per_sample_weights_backward(grad, weight, indices, offsets, offset2bag, mode, padding_idx);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H__EMBEDDING_BAG_PER_SAMPLE_WEIGHTS_BACKWARD, grad, weight, indices, offsets, offset2bag, mode, padding_idx);
}

at::Tensor wrap_empty_names(at::IntArrayRef size, c10::optional<at::DimnameList> names, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::empty(size, names, dtype, layout, device, pin_memory, memory_format));
}

at::Tensor wrap_empty_memory_format(at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::empty(size, dtype, layout, device, pin_memory, memory_format));
}

at::Tensor wrap_new_empty(const at::Tensor & self, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::new_empty(self, size, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NEW_EMPTY, self, size, dtype, layout, device, pin_memory);
}

at::Tensor wrap_new_empty_strided(const at::Tensor & self, at::IntArrayRef size, at::IntArrayRef stride, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::new_empty_strided(self, size, stride, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NEW_EMPTY_STRIDED, self, size, stride, dtype, layout, device, pin_memory);
}

at::Tensor wrap_new_full(const at::Tensor & self, at::IntArrayRef size, const at::Scalar & fill_value, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::new_full(self, size, fill_value, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NEW_FULL, self, size, fill_value, dtype, layout, device, pin_memory);
}

at::Tensor wrap_new_zeros(const at::Tensor & self, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::new_zeros(self, size, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NEW_ZEROS, self, size, dtype, layout, device, pin_memory);
}

at::Tensor wrap__empty_affine_quantized(at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, double scale, int64_t zero_point, c10::optional<at::MemoryFormat> memory_format) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::_empty_affine_quantized(size, dtype, layout, device, pin_memory, scale, zero_point, memory_format));
}

at::Tensor wrap__empty_per_channel_affine_quantized(at::IntArrayRef size, const at::Tensor & scales, const at::Tensor & zero_points, int64_t axis, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
  if (trace.is_flushing()) {
    ensure_materialized(scales, zero_points);
    return at::redispatch::_empty_per_channel_affine_quantized(size, scales, zero_points, axis, dtype, layout, device, pin_memory, memory_format);
  }
  return MK_TORCHY(scales.dtype(), scales.device(), H__EMPTY_PER_CHANNEL_AFFINE_QUANTIZED, size, scales, zero_points, axis, dtype, layout, device, pin_memory, memory_format);
}

const at::Tensor & wrap_resize_(const at::Tensor & self, at::IntArrayRef size, c10::optional<at::MemoryFormat> memory_format) {
  ensure_materialized(self);
  return at::redispatch::resize_(self, size, memory_format);
}

at::Tensor wrap_empty_quantized(at::IntArrayRef size, const at::Tensor & qtensor) {
  if (trace.is_flushing()) {
    ensure_materialized(qtensor);
    return at::redispatch::empty_quantized(size, qtensor);
  }
  return MK_TORCHY(qtensor.dtype(), qtensor.device(), H_EMPTY_QUANTIZED, size, qtensor);
}

at::Tensor & wrap_empty_out(at::IntArrayRef size, c10::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::empty(out, size, memory_format);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_EMPTY_OUT, out, size, memory_format);
}

at::Tensor wrap_empty_like(const at::Tensor & self, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::empty_like(self, dtype, layout, device, pin_memory, memory_format);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_EMPTY_LIKE, self, dtype, layout, device, pin_memory, memory_format);
}

at::Tensor wrap_empty_strided(at::IntArrayRef size, at::IntArrayRef stride, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::empty_strided(size, stride, dtype, layout, device, pin_memory));
}

at::Tensor & wrap_erf_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::erf(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ERF_OUT, out, self);
}

at::Tensor & wrap_erfc_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::erfc(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ERFC_OUT, out, self);
}

at::Tensor & wrap_exp_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::exp(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_EXP_OUT, out, self);
}

at::Tensor & wrap_exp2_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::exp2(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_EXP2_OUT, out, self);
}

at::Tensor & wrap_expm1_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::expm1(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_EXPM1_OUT, out, self);
}

at::Tensor wrap_expand(const at::Tensor & self, at::IntArrayRef size, bool implicit) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::expand(self, size, implicit);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_EXPAND, self, size, implicit);
}

at::Tensor wrap_expand_as(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::expand_as(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_EXPAND_AS, self, other);
}

at::Tensor wrap_eye(int64_t n, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::eye(n, dtype, layout, device, pin_memory));
}

at::Tensor wrap_eye_m(int64_t n, int64_t m, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::eye(n, m, dtype, layout, device, pin_memory));
}

at::Tensor & wrap_eye_out(int64_t n, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::eye(out, n);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_EYE_OUT, out, n);
}

at::Tensor & wrap_eye_m_out(int64_t n, int64_t m, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::eye(out, n, m);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_EYE_M_OUT, out, n, m);
}

at::Tensor wrap_flatten_using_ints(const at::Tensor & self, int64_t start_dim, int64_t end_dim) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::flatten(self, start_dim, end_dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FLATTEN_USING_INTS, self, start_dim, end_dim);
}

at::Tensor wrap_flatten_named_out_dim(const at::Tensor & self, int64_t start_dim, int64_t end_dim, at::Dimname out_dim) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::flatten(self, start_dim, end_dim, out_dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FLATTEN_NAMED_OUT_DIM, self, start_dim, end_dim, out_dim);
}

at::Tensor wrap_flatten_using_names(const at::Tensor & self, at::Dimname start_dim, at::Dimname end_dim, at::Dimname out_dim) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::flatten(self, start_dim, end_dim, out_dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FLATTEN_USING_NAMES, self, start_dim, end_dim, out_dim);
}

at::Tensor wrap_flatten_DimnameList(const at::Tensor & self, at::DimnameList dims, at::Dimname out_dim) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::flatten(self, dims, out_dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FLATTEN_DIMNAMELIST, self, dims, out_dim);
}

at::Tensor wrap_unflatten_int(const at::Tensor & self, int64_t dim, at::IntArrayRef sizes, c10::optional<at::DimnameList> names) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::unflatten(self, dim, sizes, names);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_UNFLATTEN_INT, self, dim, sizes, names);
}

at::Tensor wrap_unflatten_Dimname(const at::Tensor & self, at::Dimname dim, at::IntArrayRef sizes, at::DimnameList names) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::unflatten(self, dim, sizes, names);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_UNFLATTEN_DIMNAME, self, dim, sizes, names);
}

at::Tensor & wrap_fill__Scalar(at::Tensor & self, const at::Scalar & value) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::fill_(self, value);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FILL__SCALAR, self, value);
}

at::Tensor & wrap_fill__Tensor(at::Tensor & self, const at::Tensor & value) {
  if (trace.is_flushing()) {
    ensure_materialized(self, value);
    return at::redispatch::fill_(self, value);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FILL__TENSOR, self, value);
}

at::Tensor wrap_floor(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::floor(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FLOOR, self);
}

at::Tensor & wrap_floor_(at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::floor_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FLOOR_, self);
}

at::Tensor & wrap_floor_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::floor(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FLOOR_OUT, out, self);
}

at::Tensor wrap_floor_divide(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::floor_divide(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FLOOR_DIVIDE, self, other);
}

at::Tensor & wrap_floor_divide__Tensor(at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::floor_divide_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FLOOR_DIVIDE__TENSOR, self, other);
}

at::Tensor & wrap_floor_divide_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::floor_divide(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FLOOR_DIVIDE_OUT, out, self, other);
}

at::Tensor wrap_floor_divide_Scalar(const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::floor_divide(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FLOOR_DIVIDE_SCALAR, self, other);
}

at::Tensor & wrap_floor_divide__Scalar(at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::floor_divide_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FLOOR_DIVIDE__SCALAR, self, other);
}

at::Tensor wrap_frac(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::frac(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FRAC, self);
}

at::Tensor & wrap_frac_(at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::frac_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FRAC_, self);
}

at::Tensor & wrap_frac_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::frac(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FRAC_OUT, out, self);
}

at::Tensor wrap_full_names(at::IntArrayRef size, const at::Scalar & fill_value, c10::optional<at::DimnameList> names, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::full(size, fill_value, names, dtype, layout, device, pin_memory));
}

at::Tensor wrap_full(at::IntArrayRef size, const at::Scalar & fill_value, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::full(size, fill_value, dtype, layout, device, pin_memory));
}

at::Tensor & wrap_full_out(at::IntArrayRef size, const at::Scalar & fill_value, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::full(out, size, fill_value);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FULL_OUT, out, size, fill_value);
}

at::Tensor wrap_full_like(const at::Tensor & self, const at::Scalar & fill_value, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::full_like(self, fill_value, dtype, layout, device, pin_memory, memory_format);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FULL_LIKE, self, fill_value, dtype, layout, device, pin_memory, memory_format);
}

at::Tensor wrap_from_file(std::string filename, c10::optional<bool> shared, c10::optional<int64_t> size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::from_file(filename, shared, size, dtype, layout, device, pin_memory));
}

at::Tensor & wrap_gcd_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::gcd(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_GCD_OUT, out, self, other);
}

at::Tensor wrap_gcd(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::gcd(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_GCD, self, other);
}

at::Tensor & wrap_gcd_(at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::gcd_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_GCD_, self, other);
}

at::Tensor & wrap_lcm_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::lcm(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LCM_OUT, out, self, other);
}

at::Tensor wrap_lcm(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::lcm(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LCM, self, other);
}

at::Tensor & wrap_lcm_(at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::lcm_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LCM_, self, other);
}

at::Tensor wrap_grid_sampler(const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  if (trace.is_flushing()) {
    ensure_materialized(input, grid);
    return at::redispatch::grid_sampler(input, grid, interpolation_mode, padding_mode, align_corners);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_GRID_SAMPLER, input, grid, interpolation_mode, padding_mode, align_corners);
}

at::Tensor wrap_grid_sampler_2d(const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  if (trace.is_flushing()) {
    ensure_materialized(input, grid);
    return at::redispatch::grid_sampler_2d(input, grid, interpolation_mode, padding_mode, align_corners);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_GRID_SAMPLER_2D, input, grid, interpolation_mode, padding_mode, align_corners);
}

std::tuple<at::Tensor,at::Tensor> wrap_grid_sampler_2d_backward(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  ensure_materialized(grad_output, input, grid);
  return at::redispatch::grid_sampler_2d_backward(grad_output, input, grid, interpolation_mode, padding_mode, align_corners);
}

at::Tensor wrap__grid_sampler_2d_cpu_fallback(const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  if (trace.is_flushing()) {
    ensure_materialized(input, grid);
    return at::redispatch::_grid_sampler_2d_cpu_fallback(input, grid, interpolation_mode, padding_mode, align_corners);
  }
  return MK_TORCHY(input.dtype(), input.device(), H__GRID_SAMPLER_2D_CPU_FALLBACK, input, grid, interpolation_mode, padding_mode, align_corners);
}

std::tuple<at::Tensor,at::Tensor> wrap__grid_sampler_2d_cpu_fallback_backward(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  ensure_materialized(grad_output, input, grid);
  return at::redispatch::_grid_sampler_2d_cpu_fallback_backward(grad_output, input, grid, interpolation_mode, padding_mode, align_corners);
}

at::Tensor wrap_grid_sampler_3d(const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  if (trace.is_flushing()) {
    ensure_materialized(input, grid);
    return at::redispatch::grid_sampler_3d(input, grid, interpolation_mode, padding_mode, align_corners);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_GRID_SAMPLER_3D, input, grid, interpolation_mode, padding_mode, align_corners);
}

std::tuple<at::Tensor,at::Tensor> wrap_grid_sampler_3d_backward(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  ensure_materialized(grad_output, input, grid);
  return at::redispatch::grid_sampler_3d_backward(grad_output, input, grid, interpolation_mode, padding_mode, align_corners);
}

at::Tensor wrap_hann_window(int64_t window_length, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::hann_window(window_length, dtype, layout, device, pin_memory));
}

at::Tensor wrap_hann_window_periodic(int64_t window_length, bool periodic, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::hann_window(window_length, periodic, dtype, layout, device, pin_memory));
}

at::Tensor wrap_hamming_window(int64_t window_length, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::hamming_window(window_length, dtype, layout, device, pin_memory));
}

at::Tensor wrap_hamming_window_periodic(int64_t window_length, bool periodic, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::hamming_window(window_length, periodic, dtype, layout, device, pin_memory));
}

at::Tensor wrap_hamming_window_periodic_alpha(int64_t window_length, bool periodic, double alpha, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::hamming_window(window_length, periodic, alpha, dtype, layout, device, pin_memory));
}

at::Tensor wrap_hamming_window_periodic_alpha_beta(int64_t window_length, bool periodic, double alpha, double beta, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::hamming_window(window_length, periodic, alpha, beta, dtype, layout, device, pin_memory));
}

at::Tensor wrap_kaiser_window(int64_t window_length, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::kaiser_window(window_length, dtype, layout, device, pin_memory));
}

at::Tensor wrap_kaiser_window_periodic(int64_t window_length, bool periodic, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::kaiser_window(window_length, periodic, dtype, layout, device, pin_memory));
}

at::Tensor wrap_kaiser_window_beta(int64_t window_length, bool periodic, double beta, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::kaiser_window(window_length, periodic, beta, dtype, layout, device, pin_memory));
}

at::Tensor wrap_hinge_embedding_loss(const at::Tensor & self, const at::Tensor & target, double margin, int64_t reduction) {
  if (trace.is_flushing()) {
    ensure_materialized(self, target);
    return at::redispatch::hinge_embedding_loss(self, target, margin, reduction);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_HINGE_EMBEDDING_LOSS, self, target, margin, reduction);
}

at::Tensor wrap_group_norm(const at::Tensor & input, int64_t num_groups, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, double eps, bool cudnn_enabled) {
  if (trace.is_flushing()) {
    ensure_materialized(input);
    return at::redispatch::group_norm(input, num_groups, weight, bias, eps, cudnn_enabled);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_GROUP_NORM, input, num_groups, weight, bias, eps, cudnn_enabled);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> wrap_native_group_norm(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, int64_t N, int64_t C, int64_t HxW, int64_t group, double eps) {
  ensure_materialized(input);
  return at::redispatch::native_group_norm(input, weight, bias, N, C, HxW, group, eps);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> wrap_native_group_norm_backward(const at::Tensor & grad_out, const at::Tensor & input, const at::Tensor & mean, const at::Tensor & rstd, const c10::optional<at::Tensor> & weight, int64_t N, int64_t C, int64_t HxW, int64_t group, std::array<bool,3> output_mask) {
  ensure_materialized(grad_out, input, mean, rstd);
  return at::redispatch::native_group_norm_backward(grad_out, input, mean, rstd, weight, N, C, HxW, group, output_mask);
}

at::Tensor wrap__fft_r2c(const at::Tensor & self, at::IntArrayRef dim, int64_t normalization, bool onesided) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_fft_r2c(self, dim, normalization, onesided);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__FFT_R2C, self, dim, normalization, onesided);
}

at::Tensor & wrap__fft_r2c_out(const at::Tensor & self, at::IntArrayRef dim, int64_t normalization, bool onesided, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::_fft_r2c(out, self, dim, normalization, onesided);
  }
  return MK_TORCHY(out.dtype(), out.device(), H__FFT_R2C_OUT, out, self, dim, normalization, onesided);
}

at::Tensor wrap__fft_c2r(const at::Tensor & self, at::IntArrayRef dim, int64_t normalization, int64_t last_dim_size) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_fft_c2r(self, dim, normalization, last_dim_size);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__FFT_C2R, self, dim, normalization, last_dim_size);
}

at::Tensor & wrap__fft_c2r_out(const at::Tensor & self, at::IntArrayRef dim, int64_t normalization, int64_t last_dim_size, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::_fft_c2r(out, self, dim, normalization, last_dim_size);
  }
  return MK_TORCHY(out.dtype(), out.device(), H__FFT_C2R_OUT, out, self, dim, normalization, last_dim_size);
}

at::Tensor wrap__fft_c2c(const at::Tensor & self, at::IntArrayRef dim, int64_t normalization, bool forward) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_fft_c2c(self, dim, normalization, forward);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__FFT_C2C, self, dim, normalization, forward);
}

at::Tensor & wrap__fft_c2c_out(const at::Tensor & self, at::IntArrayRef dim, int64_t normalization, bool forward, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::_fft_c2c(out, self, dim, normalization, forward);
  }
  return MK_TORCHY(out.dtype(), out.device(), H__FFT_C2C_OUT, out, self, dim, normalization, forward);
}

int64_t wrap__cufft_get_plan_cache_size(int64_t device_index) {
  ensure_materialized();
  return at::redispatch::_cufft_get_plan_cache_size(device_index);
}

int64_t wrap__cufft_get_plan_cache_max_size(int64_t device_index) {
  ensure_materialized();
  return at::redispatch::_cufft_get_plan_cache_max_size(device_index);
}

void wrap__cufft_set_plan_cache_max_size(int64_t device_index, int64_t max_size) {
  ensure_materialized();
  return at::redispatch::_cufft_set_plan_cache_max_size(device_index, max_size);
}

void wrap__cufft_clear_plan_cache(int64_t device_index) {
  ensure_materialized();
  return at::redispatch::_cufft_clear_plan_cache(device_index);
}

at::Tensor wrap_index_Tensor(const at::Tensor & self, const c10::List<c10::optional<at::Tensor>> & indices) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::index(self, indices);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INDEX_TENSOR, self, indices);
}

at::Tensor & wrap_index_copy_(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index, source);
    return at::redispatch::index_copy_(self, dim, index, source);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INDEX_COPY_, self, dim, index, source);
}

at::Tensor wrap_index_copy(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index, source);
    return at::redispatch::index_copy(self, dim, index, source);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INDEX_COPY, self, dim, index, source);
}

at::Tensor & wrap_index_copy__dimname(at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & source) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index, source);
    return at::redispatch::index_copy_(self, dim, index, source);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INDEX_COPY__DIMNAME, self, dim, index, source);
}

at::Tensor wrap_index_copy_dimname(const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & source) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index, source);
    return at::redispatch::index_copy(self, dim, index, source);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INDEX_COPY_DIMNAME, self, dim, index, source);
}

at::Tensor & wrap_index_put_(at::Tensor & self, const c10::List<c10::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate) {
  if (trace.is_flushing()) {
    ensure_materialized(self, values);
    return at::redispatch::index_put_(self, indices, values, accumulate);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INDEX_PUT_, self, indices, values, accumulate);
}

at::Tensor wrap_index_put(const at::Tensor & self, const c10::List<c10::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate) {
  if (trace.is_flushing()) {
    ensure_materialized(self, values);
    return at::redispatch::index_put(self, indices, values, accumulate);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INDEX_PUT, self, indices, values, accumulate);
}

at::Tensor & wrap__index_put_impl_(at::Tensor & self, const c10::List<c10::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate, bool unsafe) {
  if (trace.is_flushing()) {
    ensure_materialized(self, values);
    return at::redispatch::_index_put_impl_(self, indices, values, accumulate, unsafe);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__INDEX_PUT_IMPL_, self, indices, values, accumulate, unsafe);
}

at::Tensor wrap_instance_norm(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, bool use_input_stats, double momentum, double eps, bool cudnn_enabled) {
  if (trace.is_flushing()) {
    ensure_materialized(input);
    return at::redispatch::instance_norm(input, weight, bias, running_mean, running_var, use_input_stats, momentum, eps, cudnn_enabled);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_INSTANCE_NORM, input, weight, bias, running_mean, running_var, use_input_stats, momentum, eps, cudnn_enabled);
}

at::Tensor wrap_inverse(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::inverse(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INVERSE, self);
}

at::Tensor & wrap_inverse_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::inverse(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_INVERSE_OUT, out, self);
}

at::Tensor wrap__inverse_helper(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_inverse_helper(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__INVERSE_HELPER, self);
}

at::Tensor wrap_isclose(const at::Tensor & self, const at::Tensor & other, double rtol, double atol, bool equal_nan) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::isclose(self, other, rtol, atol, equal_nan);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ISCLOSE, self, other, rtol, atol, equal_nan);
}

at::Tensor wrap_isnan(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::isnan(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ISNAN, self);
}

bool wrap_is_distributed(const at::Tensor & self) {
  ensure_materialized(self);
  return at::redispatch::is_distributed(self);
}

at::Tensor wrap_isreal(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::isreal(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ISREAL, self);
}

bool wrap_is_nonzero(const at::Tensor & self) {
  ensure_materialized(self);
  return at::redispatch::is_nonzero(self);
}

bool wrap_is_same_size(const at::Tensor & self, const at::Tensor & other) {
  ensure_materialized(self, other);
  return at::redispatch::is_same_size(self, other);
}

at::Tensor wrap_kl_div(const at::Tensor & self, const at::Tensor & target, int64_t reduction, bool log_target) {
  if (trace.is_flushing()) {
    ensure_materialized(self, target);
    return at::redispatch::kl_div(self, target, reduction, log_target);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_KL_DIV, self, target, reduction, log_target);
}

at::Tensor wrap_kl_div_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, bool log_target) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, target);
    return at::redispatch::kl_div_backward(grad_output, self, target, reduction, log_target);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_KL_DIV_BACKWARD, grad_output, self, target, reduction, log_target);
}

at::Tensor wrap_kron(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::kron(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_KRON, self, other);
}

at::Tensor & wrap_kron_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::kron(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_KRON_OUT, out, self, other);
}

std::tuple<at::Tensor,at::Tensor> wrap_kthvalue(const at::Tensor & self, int64_t k, int64_t dim, bool keepdim) {
  ensure_materialized(self);
  return at::redispatch::kthvalue(self, k, dim, keepdim);
}

std::tuple<at::Tensor &,at::Tensor &> wrap_kthvalue_values(const at::Tensor & self, int64_t k, int64_t dim, bool keepdim, at::Tensor & values, at::Tensor & indices) {
  ensure_materialized(values, indices, self);
  return at::redispatch::kthvalue(values, indices, self, k, dim, keepdim);
}

std::tuple<at::Tensor,at::Tensor> wrap_kthvalue_dimname(const at::Tensor & self, int64_t k, at::Dimname dim, bool keepdim) {
  ensure_materialized(self);
  return at::redispatch::kthvalue(self, k, dim, keepdim);
}

std::tuple<at::Tensor &,at::Tensor &> wrap_kthvalue_dimname_out(const at::Tensor & self, int64_t k, at::Dimname dim, bool keepdim, at::Tensor & values, at::Tensor & indices) {
  ensure_materialized(values, indices, self);
  return at::redispatch::kthvalue(values, indices, self, k, dim, keepdim);
}

at::Tensor wrap_layer_norm(const at::Tensor & input, at::IntArrayRef normalized_shape, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, double eps, bool cudnn_enable) {
  if (trace.is_flushing()) {
    ensure_materialized(input);
    return at::redispatch::layer_norm(input, normalized_shape, weight, bias, eps, cudnn_enable);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_LAYER_NORM, input, normalized_shape, weight, bias, eps, cudnn_enable);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> wrap_native_layer_norm(const at::Tensor & input, at::IntArrayRef normalized_shape, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, double eps) {
  ensure_materialized(input);
  return at::redispatch::native_layer_norm(input, normalized_shape, weight, bias, eps);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> wrap_native_layer_norm_backward(const at::Tensor & grad_out, const at::Tensor & input, at::IntArrayRef normalized_shape, const at::Tensor & mean, const at::Tensor & rstd, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, std::array<bool,3> output_mask) {
  ensure_materialized(grad_out, input, mean, rstd);
  return at::redispatch::native_layer_norm_backward(grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask);
}

at::Tensor wrap_nan_to_num(const at::Tensor & self, c10::optional<double> nan, c10::optional<double> posinf, c10::optional<double> neginf) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::nan_to_num(self, nan, posinf, neginf);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NAN_TO_NUM, self, nan, posinf, neginf);
}

at::Tensor & wrap_nan_to_num_(at::Tensor & self, c10::optional<double> nan, c10::optional<double> posinf, c10::optional<double> neginf) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::nan_to_num_(self, nan, posinf, neginf);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NAN_TO_NUM_, self, nan, posinf, neginf);
}

at::Tensor & wrap_nan_to_num_out(const at::Tensor & self, c10::optional<double> nan, c10::optional<double> posinf, c10::optional<double> neginf, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::nan_to_num(out, self, nan, posinf, neginf);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NAN_TO_NUM_OUT, out, self, nan, posinf, neginf);
}

at::Tensor wrap_linear(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias) {
  if (trace.is_flushing()) {
    ensure_materialized(input, weight);
    return at::redispatch::linear(input, weight, bias);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_LINEAR, input, weight, bias);
}

at::Tensor wrap_mkldnn_linear(const at::Tensor & self, const at::Tensor & weight, const c10::optional<at::Tensor> & bias) {
  if (trace.is_flushing()) {
    ensure_materialized(self, weight);
    return at::redispatch::mkldnn_linear(self, weight, bias);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MKLDNN_LINEAR, self, weight, bias);
}

at::Tensor wrap_mkldnn_linear_backward_input(at::IntArrayRef input_size, const at::Tensor & grad_output, const at::Tensor & weight) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, weight);
    return at::redispatch::mkldnn_linear_backward_input(input_size, grad_output, weight);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_MKLDNN_LINEAR_BACKWARD_INPUT, input_size, grad_output, weight);
}

std::tuple<at::Tensor,at::Tensor> wrap_mkldnn_linear_backward_weights(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & weight, bool bias_defined) {
  ensure_materialized(grad_output, input, weight);
  return at::redispatch::mkldnn_linear_backward_weights(grad_output, input, weight, bias_defined);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> wrap_mkldnn_linear_backward(const at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & weight, std::array<bool,3> output_mask) {
  ensure_materialized(self, grad_output, weight);
  return at::redispatch::mkldnn_linear_backward(self, grad_output, weight, output_mask);
}

at::Tensor wrap_fbgemm_linear_int8_weight_fp32_activation(const at::Tensor & input, const at::Tensor & weight, const at::Tensor & packed, const at::Tensor & col_offsets, const at::Scalar & weight_scale, const at::Scalar & weight_zero_point, const at::Tensor & bias) {
  if (trace.is_flushing()) {
    ensure_materialized(input, weight, packed, col_offsets, bias);
    return at::redispatch::fbgemm_linear_int8_weight_fp32_activation(input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_FBGEMM_LINEAR_INT8_WEIGHT_FP32_ACTIVATION, input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias);
}

at::Tensor wrap_fbgemm_linear_int8_weight(const at::Tensor & input, const at::Tensor & weight, const at::Tensor & packed, const at::Tensor & col_offsets, const at::Scalar & weight_scale, const at::Scalar & weight_zero_point, const at::Tensor & bias) {
  if (trace.is_flushing()) {
    ensure_materialized(input, weight, packed, col_offsets, bias);
    return at::redispatch::fbgemm_linear_int8_weight(input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_FBGEMM_LINEAR_INT8_WEIGHT, input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias);
}

std::tuple<at::Tensor,at::Tensor,double,int64_t> wrap_fbgemm_linear_quantize_weight(const at::Tensor & input) {
  ensure_materialized(input);
  return at::redispatch::fbgemm_linear_quantize_weight(input);
}

at::Tensor wrap_fbgemm_pack_gemm_matrix_fp16(const at::Tensor & input) {
  if (trace.is_flushing()) {
    ensure_materialized(input);
    return at::redispatch::fbgemm_pack_gemm_matrix_fp16(input);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_FBGEMM_PACK_GEMM_MATRIX_FP16, input);
}

at::Tensor wrap_fbgemm_linear_fp16_weight_fp32_activation(const at::Tensor & input, const at::Tensor & packed_weight, const at::Tensor & bias) {
  if (trace.is_flushing()) {
    ensure_materialized(input, packed_weight, bias);
    return at::redispatch::fbgemm_linear_fp16_weight_fp32_activation(input, packed_weight, bias);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_FBGEMM_LINEAR_FP16_WEIGHT_FP32_ACTIVATION, input, packed_weight, bias);
}

at::Tensor wrap_fbgemm_linear_fp16_weight(const at::Tensor & input, const at::Tensor & packed_weight, const at::Tensor & bias) {
  if (trace.is_flushing()) {
    ensure_materialized(input, packed_weight, bias);
    return at::redispatch::fbgemm_linear_fp16_weight(input, packed_weight, bias);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_FBGEMM_LINEAR_FP16_WEIGHT, input, packed_weight, bias);
}

at::Tensor wrap_fbgemm_pack_quantized_matrix(const at::Tensor & input) {
  if (trace.is_flushing()) {
    ensure_materialized(input);
    return at::redispatch::fbgemm_pack_quantized_matrix(input);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_FBGEMM_PACK_QUANTIZED_MATRIX, input);
}

at::Tensor wrap_fbgemm_pack_quantized_matrix_KN(const at::Tensor & input, int64_t K, int64_t N) {
  if (trace.is_flushing()) {
    ensure_materialized(input);
    return at::redispatch::fbgemm_pack_quantized_matrix(input, K, N);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_FBGEMM_PACK_QUANTIZED_MATRIX_KN, input, K, N);
}

at::Tensor wrap_ldexp_Tensor(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::ldexp(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LDEXP_TENSOR, self, other);
}

at::Tensor & wrap_ldexp_(at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::ldexp_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LDEXP_, self, other);
}

at::Tensor & wrap_ldexp_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::ldexp(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LDEXP_OUT, out, self, other);
}

at::Tensor wrap_linspace(const at::Scalar & start, const at::Scalar & end, c10::optional<int64_t> steps, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::linspace(start, end, steps, dtype, layout, device, pin_memory));
}

at::Tensor & wrap_linspace_out(const at::Scalar & start, const at::Scalar & end, c10::optional<int64_t> steps, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::linspace(out, start, end, steps);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LINSPACE_OUT, out, start, end, steps);
}

at::Tensor & wrap_log_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::log(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LOG_OUT, out, self);
}

at::Tensor & wrap_log10_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::log10(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LOG10_OUT, out, self);
}

at::Tensor wrap_log1p(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::log1p(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LOG1P, self);
}

at::Tensor & wrap_log1p_(at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::log1p_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LOG1P_, self);
}

at::Tensor & wrap_log1p_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::log1p(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LOG1P_OUT, out, self);
}

at::Tensor & wrap_log2_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::log2(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LOG2_OUT, out, self);
}

at::Tensor & wrap_logaddexp_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::logaddexp(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LOGADDEXP_OUT, out, self, other);
}

at::Tensor wrap_logaddexp(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::logaddexp(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LOGADDEXP, self, other);
}

at::Tensor & wrap_logaddexp2_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::logaddexp2(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LOGADDEXP2_OUT, out, self, other);
}

at::Tensor wrap_logaddexp2(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::logaddexp2(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LOGADDEXP2, self, other);
}

at::Tensor wrap_xlogy_Tensor(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::xlogy(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_XLOGY_TENSOR, self, other);
}

at::Tensor wrap_xlogy_Scalar_Self(const at::Scalar & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(other);
    return at::redispatch::xlogy(self, other);
  }
  return MK_TORCHY(other.dtype(), other.device(), H_XLOGY_SCALAR_SELF, self, other);
}

at::Tensor wrap_xlogy_Scalar_Other(const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::xlogy(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_XLOGY_SCALAR_OTHER, self, other);
}

at::Tensor & wrap_xlogy__Tensor(at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::xlogy_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_XLOGY__TENSOR, self, other);
}

at::Tensor & wrap_xlogy__Scalar_Other(at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::xlogy_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_XLOGY__SCALAR_OTHER, self, other);
}

at::Tensor & wrap_xlogy_OutTensor(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::xlogy(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_XLOGY_OUTTENSOR, out, self, other);
}

at::Tensor & wrap_xlogy_OutScalar_Self(const at::Scalar & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, other);
    return at::redispatch::xlogy(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_XLOGY_OUTSCALAR_SELF, out, self, other);
}

at::Tensor & wrap_xlogy_OutScalar_Other(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::xlogy(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_XLOGY_OUTSCALAR_OTHER, out, self, other);
}

at::Tensor wrap_logdet(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::logdet(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LOGDET, self);
}

at::Tensor wrap_logspace(const at::Scalar & start, const at::Scalar & end, c10::optional<int64_t> steps, double base, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::logspace(start, end, steps, base, dtype, layout, device, pin_memory));
}

at::Tensor & wrap_logspace_out(const at::Scalar & start, const at::Scalar & end, c10::optional<int64_t> steps, double base, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::logspace(out, start, end, steps, base);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LOGSPACE_OUT, out, start, end, steps, base);
}

at::Tensor wrap_log_softmax_int(const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::log_softmax(self, dim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LOG_SOFTMAX_INT, self, dim, dtype);
}

at::Tensor wrap_log_softmax_Dimname(const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::log_softmax(self, dim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LOG_SOFTMAX_DIMNAME, self, dim, dtype);
}

at::Tensor wrap__log_softmax(const at::Tensor & self, int64_t dim, bool half_to_float) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_log_softmax(self, dim, half_to_float);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__LOG_SOFTMAX, self, dim, half_to_float);
}

at::Tensor wrap__log_softmax_backward_data(const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, output, self);
    return at::redispatch::_log_softmax_backward_data(grad_output, output, dim, self);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H__LOG_SOFTMAX_BACKWARD_DATA, grad_output, output, dim, self);
}

at::Tensor wrap__logcumsumexp(const at::Tensor & self, int64_t dim) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_logcumsumexp(self, dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__LOGCUMSUMEXP, self, dim);
}

at::Tensor & wrap__logcumsumexp_out(const at::Tensor & self, int64_t dim, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::_logcumsumexp(out, self, dim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H__LOGCUMSUMEXP_OUT, out, self, dim);
}

at::Tensor wrap_logcumsumexp(const at::Tensor & self, int64_t dim) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::logcumsumexp(self, dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LOGCUMSUMEXP, self, dim);
}

at::Tensor & wrap_logcumsumexp_out(const at::Tensor & self, int64_t dim, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::logcumsumexp(out, self, dim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LOGCUMSUMEXP_OUT, out, self, dim);
}

at::Tensor wrap_logcumsumexp_dimname(const at::Tensor & self, at::Dimname dim) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::logcumsumexp(self, dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LOGCUMSUMEXP_DIMNAME, self, dim);
}

at::Tensor & wrap_logcumsumexp_dimname_out(const at::Tensor & self, at::Dimname dim, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::logcumsumexp(out, self, dim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LOGCUMSUMEXP_DIMNAME_OUT, out, self, dim);
}

at::Tensor wrap_logsumexp(const at::Tensor & self, at::IntArrayRef dim, bool keepdim) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::logsumexp(self, dim, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LOGSUMEXP, self, dim, keepdim);
}

at::Tensor & wrap_logsumexp_out(const at::Tensor & self, at::IntArrayRef dim, bool keepdim, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::logsumexp(out, self, dim, keepdim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LOGSUMEXP_OUT, out, self, dim, keepdim);
}

at::Tensor wrap_logsumexp_names(const at::Tensor & self, at::DimnameList dim, bool keepdim) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::logsumexp(self, dim, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LOGSUMEXP_NAMES, self, dim, keepdim);
}

at::Tensor & wrap_logsumexp_names_out(const at::Tensor & self, at::DimnameList dim, bool keepdim, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::logsumexp(out, self, dim, keepdim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LOGSUMEXP_NAMES_OUT, out, self, dim, keepdim);
}

at::Tensor wrap_margin_ranking_loss(const at::Tensor & input1, const at::Tensor & input2, const at::Tensor & target, double margin, int64_t reduction) {
  if (trace.is_flushing()) {
    ensure_materialized(input1, input2, target);
    return at::redispatch::margin_ranking_loss(input1, input2, target, margin, reduction);
  }
  return MK_TORCHY(input1.dtype(), input1.device(), H_MARGIN_RANKING_LOSS, input1, input2, target, margin, reduction);
}

at::Tensor wrap_matmul(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::matmul(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MATMUL, self, other);
}

at::Tensor & wrap_matmul_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::matmul(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_MATMUL_OUT, out, self, other);
}

at::Tensor wrap_matrix_rank_tol(const at::Tensor & self, double tol, bool symmetric) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::matrix_rank(self, tol, symmetric);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MATRIX_RANK_TOL, self, tol, symmetric);
}

at::Tensor wrap_matrix_rank(const at::Tensor & self, bool symmetric) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::matrix_rank(self, symmetric);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MATRIX_RANK, self, symmetric);
}

at::Tensor wrap_matrix_power(const at::Tensor & self, int64_t n) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::matrix_power(self, n);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MATRIX_POWER, self, n);
}

at::Tensor & wrap_matrix_power_out(const at::Tensor & self, int64_t n, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::matrix_power(out, self, n);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_MATRIX_POWER_OUT, out, self, n);
}

at::Tensor wrap_matrix_exp(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::matrix_exp(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MATRIX_EXP, self);
}

at::Tensor wrap_matrix_exp_backward(const at::Tensor & self, const at::Tensor & grad) {
  if (trace.is_flushing()) {
    ensure_materialized(self, grad);
    return at::redispatch::matrix_exp_backward(self, grad);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MATRIX_EXP_BACKWARD, self, grad);
}

std::tuple<at::Tensor,at::Tensor> wrap__aminmax(const at::Tensor & self) {
  ensure_materialized(self);
  return at::redispatch::_aminmax(self);
}

std::tuple<at::Tensor,at::Tensor> wrap__aminmax_dim(const at::Tensor & self, int64_t dim, bool keepdim) {
  ensure_materialized(self);
  return at::redispatch::_aminmax(self, dim, keepdim);
}

at::Tensor wrap__compute_linear_combination(const at::Tensor & input, const at::Tensor & coefficients) {
  if (trace.is_flushing()) {
    ensure_materialized(input, coefficients);
    return at::redispatch::_compute_linear_combination(input, coefficients);
  }
  return MK_TORCHY(input.dtype(), input.device(), H__COMPUTE_LINEAR_COMBINATION, input, coefficients);
}

at::Tensor & wrap__compute_linear_combination_out(const at::Tensor & input, const at::Tensor & coefficients, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, input, coefficients);
    return at::redispatch::_compute_linear_combination(out, input, coefficients);
  }
  return MK_TORCHY(out.dtype(), out.device(), H__COMPUTE_LINEAR_COMBINATION_OUT, out, input, coefficients);
}

std::tuple<at::Tensor,at::Tensor> wrap_max_dim(const at::Tensor & self, int64_t dim, bool keepdim) {
  ensure_materialized(self);
  return at::redispatch::max(self, dim, keepdim);
}

std::tuple<at::Tensor &,at::Tensor &> wrap_max_dim_max(const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & max, at::Tensor & max_values) {
  ensure_materialized(max, max_values, self);
  return at::redispatch::max(max, max_values, self, dim, keepdim);
}

std::tuple<at::Tensor,at::Tensor> wrap_max_names_dim(const at::Tensor & self, at::Dimname dim, bool keepdim) {
  ensure_materialized(self);
  return at::redispatch::max(self, dim, keepdim);
}

std::tuple<at::Tensor &,at::Tensor &> wrap_max_names_dim_max(const at::Tensor & self, at::Dimname dim, bool keepdim, at::Tensor & max, at::Tensor & max_values) {
  ensure_materialized(max, max_values, self);
  return at::redispatch::max(max, max_values, self, dim, keepdim);
}

at::Tensor wrap_value_selecting_reduction_backward(const at::Tensor & grad, int64_t dim, const at::Tensor & indices, at::IntArrayRef sizes, bool keepdim) {
  if (trace.is_flushing()) {
    ensure_materialized(grad, indices);
    return at::redispatch::value_selecting_reduction_backward(grad, dim, indices, sizes, keepdim);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H_VALUE_SELECTING_REDUCTION_BACKWARD, grad, dim, indices, sizes, keepdim);
}

at::Tensor wrap_amax(const at::Tensor & self, at::IntArrayRef dim, bool keepdim) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::amax(self, dim, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_AMAX, self, dim, keepdim);
}

at::Tensor & wrap_amax_out(const at::Tensor & self, at::IntArrayRef dim, bool keepdim, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::amax(out, self, dim, keepdim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_AMAX_OUT, out, self, dim, keepdim);
}

std::tuple<at::Tensor,at::Tensor> wrap_max_pool1d_with_indices(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  ensure_materialized(self);
  return at::redispatch::max_pool1d_with_indices(self, kernel_size, stride, padding, dilation, ceil_mode);
}

at::Tensor wrap_max_pool1d(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::max_pool1d(self, kernel_size, stride, padding, dilation, ceil_mode);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MAX_POOL1D, self, kernel_size, stride, padding, dilation, ceil_mode);
}

at::Tensor wrap_max_pool2d(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::max_pool2d(self, kernel_size, stride, padding, dilation, ceil_mode);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MAX_POOL2D, self, kernel_size, stride, padding, dilation, ceil_mode);
}

at::Tensor wrap_mkldnn_max_pool2d(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::mkldnn_max_pool2d(self, kernel_size, stride, padding, dilation, ceil_mode);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MKLDNN_MAX_POOL2D, self, kernel_size, stride, padding, dilation, ceil_mode);
}

at::Tensor wrap_mkldnn_max_pool2d_backward(const at::Tensor & grad_output, const at::Tensor & output, const at::Tensor & input, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, output, input);
    return at::redispatch::mkldnn_max_pool2d_backward(grad_output, output, input, kernel_size, stride, padding, dilation, ceil_mode);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_MKLDNN_MAX_POOL2D_BACKWARD, grad_output, output, input, kernel_size, stride, padding, dilation, ceil_mode);
}

at::Tensor wrap_mkldnn_max_pool3d(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::mkldnn_max_pool3d(self, kernel_size, stride, padding, dilation, ceil_mode);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MKLDNN_MAX_POOL3D, self, kernel_size, stride, padding, dilation, ceil_mode);
}

at::Tensor wrap_mkldnn_max_pool3d_backward(const at::Tensor & grad_output, const at::Tensor & output, const at::Tensor & input, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, output, input);
    return at::redispatch::mkldnn_max_pool3d_backward(grad_output, output, input, kernel_size, stride, padding, dilation, ceil_mode);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_MKLDNN_MAX_POOL3D_BACKWARD, grad_output, output, input, kernel_size, stride, padding, dilation, ceil_mode);
}

at::Tensor wrap_quantized_max_pool1d(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::quantized_max_pool1d(self, kernel_size, stride, padding, dilation, ceil_mode);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_QUANTIZED_MAX_POOL1D, self, kernel_size, stride, padding, dilation, ceil_mode);
}

at::Tensor wrap_quantized_max_pool2d(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::quantized_max_pool2d(self, kernel_size, stride, padding, dilation, ceil_mode);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_QUANTIZED_MAX_POOL2D, self, kernel_size, stride, padding, dilation, ceil_mode);
}

at::Tensor wrap_max_pool3d(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::max_pool3d(self, kernel_size, stride, padding, dilation, ceil_mode);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MAX_POOL3D, self, kernel_size, stride, padding, dilation, ceil_mode);
}

at::Tensor wrap_mean(const at::Tensor & self, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::mean(self, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MEAN, self, dtype);
}

at::Tensor wrap_mean_dim(const at::Tensor & self, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::mean(self, dim, keepdim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MEAN_DIM, self, dim, keepdim, dtype);
}

at::Tensor & wrap_mean_out(const at::Tensor & self, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::mean(out, self, dim, keepdim, dtype);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_MEAN_OUT, out, self, dim, keepdim, dtype);
}

at::Tensor wrap_mean_names_dim(const at::Tensor & self, at::DimnameList dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::mean(self, dim, keepdim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MEAN_NAMES_DIM, self, dim, keepdim, dtype);
}

at::Tensor & wrap_mean_names_out(const at::Tensor & self, at::DimnameList dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::mean(out, self, dim, keepdim, dtype);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_MEAN_NAMES_OUT, out, self, dim, keepdim, dtype);
}

at::Tensor wrap_median(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::median(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MEDIAN, self);
}

std::tuple<at::Tensor,at::Tensor> wrap_median_dim(const at::Tensor & self, int64_t dim, bool keepdim) {
  ensure_materialized(self);
  return at::redispatch::median(self, dim, keepdim);
}

std::tuple<at::Tensor &,at::Tensor &> wrap_median_dim_values(const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & values, at::Tensor & indices) {
  ensure_materialized(values, indices, self);
  return at::redispatch::median(values, indices, self, dim, keepdim);
}

std::tuple<at::Tensor,at::Tensor> wrap_median_names_dim(const at::Tensor & self, at::Dimname dim, bool keepdim) {
  ensure_materialized(self);
  return at::redispatch::median(self, dim, keepdim);
}

std::tuple<at::Tensor &,at::Tensor &> wrap_median_names_dim_values(const at::Tensor & self, at::Dimname dim, bool keepdim, at::Tensor & values, at::Tensor & indices) {
  ensure_materialized(values, indices, self);
  return at::redispatch::median(values, indices, self, dim, keepdim);
}

at::Tensor wrap_nanmedian(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::nanmedian(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NANMEDIAN, self);
}

std::tuple<at::Tensor,at::Tensor> wrap_nanmedian_dim(const at::Tensor & self, int64_t dim, bool keepdim) {
  ensure_materialized(self);
  return at::redispatch::nanmedian(self, dim, keepdim);
}

std::tuple<at::Tensor &,at::Tensor &> wrap_nanmedian_dim_values(const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & values, at::Tensor & indices) {
  ensure_materialized(values, indices, self);
  return at::redispatch::nanmedian(values, indices, self, dim, keepdim);
}

std::tuple<at::Tensor,at::Tensor> wrap_nanmedian_names_dim(const at::Tensor & self, at::Dimname dim, bool keepdim) {
  ensure_materialized(self);
  return at::redispatch::nanmedian(self, dim, keepdim);
}

std::tuple<at::Tensor &,at::Tensor &> wrap_nanmedian_names_dim_values(const at::Tensor & self, at::Dimname dim, bool keepdim, at::Tensor & values, at::Tensor & indices) {
  ensure_materialized(values, indices, self);
  return at::redispatch::nanmedian(values, indices, self, dim, keepdim);
}

std::tuple<at::Tensor,at::Tensor> wrap_min_dim(const at::Tensor & self, int64_t dim, bool keepdim) {
  ensure_materialized(self);
  return at::redispatch::min(self, dim, keepdim);
}

std::tuple<at::Tensor &,at::Tensor &> wrap_min_dim_min(const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & min, at::Tensor & min_indices) {
  ensure_materialized(min, min_indices, self);
  return at::redispatch::min(min, min_indices, self, dim, keepdim);
}

std::tuple<at::Tensor,at::Tensor> wrap_min_names_dim(const at::Tensor & self, at::Dimname dim, bool keepdim) {
  ensure_materialized(self);
  return at::redispatch::min(self, dim, keepdim);
}

std::tuple<at::Tensor &,at::Tensor &> wrap_min_names_dim_min(const at::Tensor & self, at::Dimname dim, bool keepdim, at::Tensor & min, at::Tensor & min_indices) {
  ensure_materialized(min, min_indices, self);
  return at::redispatch::min(min, min_indices, self, dim, keepdim);
}

at::Tensor wrap_amin(const at::Tensor & self, at::IntArrayRef dim, bool keepdim) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::amin(self, dim, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_AMIN, self, dim, keepdim);
}

at::Tensor & wrap_amin_out(const at::Tensor & self, at::IntArrayRef dim, bool keepdim, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::amin(out, self, dim, keepdim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_AMIN_OUT, out, self, dim, keepdim);
}

at::Tensor wrap_mkldnn_convolution(const at::Tensor & self, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups) {
  if (trace.is_flushing()) {
    ensure_materialized(self, weight);
    return at::redispatch::mkldnn_convolution(self, weight, bias, padding, stride, dilation, groups);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MKLDNN_CONVOLUTION, self, weight, bias, padding, stride, dilation, groups);
}

at::Tensor wrap_mkldnn_convolution_backward_input(at::IntArrayRef self_size, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool bias_defined) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, weight);
    return at::redispatch::mkldnn_convolution_backward_input(self_size, grad_output, weight, padding, stride, dilation, groups, bias_defined);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_MKLDNN_CONVOLUTION_BACKWARD_INPUT, self_size, grad_output, weight, padding, stride, dilation, groups, bias_defined);
}

std::tuple<at::Tensor,at::Tensor> wrap_mkldnn_convolution_backward_weights(at::IntArrayRef weight_size, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool bias_defined) {
  ensure_materialized(grad_output, self);
  return at::redispatch::mkldnn_convolution_backward_weights(weight_size, grad_output, self, padding, stride, dilation, groups, bias_defined);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> wrap_mkldnn_convolution_backward(const at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, std::array<bool,3> output_mask) {
  ensure_materialized(self, grad_output, weight);
  return at::redispatch::mkldnn_convolution_backward(self, grad_output, weight, padding, stride, dilation, groups, output_mask);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> wrap_miopen_batch_norm(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, bool training, double exponential_average_factor, double epsilon) {
  ensure_materialized(input, weight);
  return at::redispatch::miopen_batch_norm(input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> wrap_miopen_batch_norm_backward(const at::Tensor & input, const at::Tensor & grad_output, const at::Tensor & weight, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, const c10::optional<at::Tensor> & save_mean, const c10::optional<at::Tensor> & save_var, double epsilon) {
  ensure_materialized(input, grad_output, weight);
  return at::redispatch::miopen_batch_norm_backward(input, grad_output, weight, running_mean, running_var, save_mean, save_var, epsilon);
}

at::Tensor wrap_miopen_convolution(const at::Tensor & self, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  if (trace.is_flushing()) {
    ensure_materialized(self, weight);
    return at::redispatch::miopen_convolution(self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MIOPEN_CONVOLUTION, self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic);
}

at::Tensor wrap_miopen_convolution_backward_input(at::IntArrayRef self_size, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, weight);
    return at::redispatch::miopen_convolution_backward_input(self_size, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_MIOPEN_CONVOLUTION_BACKWARD_INPUT, self_size, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> wrap_miopen_convolution_backward(const at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,3> output_mask) {
  ensure_materialized(self, grad_output, weight);
  return at::redispatch::miopen_convolution_backward(self, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, output_mask);
}

at::Tensor wrap_miopen_convolution_backward_bias(const at::Tensor & grad_output) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output);
    return at::redispatch::miopen_convolution_backward_bias(grad_output);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_MIOPEN_CONVOLUTION_BACKWARD_BIAS, grad_output);
}

at::Tensor wrap_miopen_convolution_backward_weight(at::IntArrayRef weight_size, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self);
    return at::redispatch::miopen_convolution_backward_weight(weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_MIOPEN_CONVOLUTION_BACKWARD_WEIGHT, weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
}

at::Tensor wrap_miopen_convolution_transpose(const at::Tensor & self, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  if (trace.is_flushing()) {
    ensure_materialized(self, weight);
    return at::redispatch::miopen_convolution_transpose(self, weight, bias, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MIOPEN_CONVOLUTION_TRANSPOSE, self, weight, bias, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> wrap_miopen_convolution_transpose_backward(const at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,3> output_mask) {
  ensure_materialized(self, grad_output, weight);
  return at::redispatch::miopen_convolution_transpose_backward(self, grad_output, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic, output_mask);
}

at::Tensor wrap_miopen_convolution_transpose_backward_input(const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, weight);
    return at::redispatch::miopen_convolution_transpose_backward_input(grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_MIOPEN_CONVOLUTION_TRANSPOSE_BACKWARD_INPUT, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
}

at::Tensor wrap_miopen_convolution_transpose_backward_weight(at::IntArrayRef weight_size, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self);
    return at::redispatch::miopen_convolution_transpose_backward_weight(weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_MIOPEN_CONVOLUTION_TRANSPOSE_BACKWARD_WEIGHT, weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
}

at::Tensor wrap_miopen_depthwise_convolution(const at::Tensor & self, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  if (trace.is_flushing()) {
    ensure_materialized(self, weight);
    return at::redispatch::miopen_depthwise_convolution(self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MIOPEN_DEPTHWISE_CONVOLUTION, self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic);
}

at::Tensor wrap_miopen_depthwise_convolution_backward_input(at::IntArrayRef self_size, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, weight);
    return at::redispatch::miopen_depthwise_convolution_backward_input(self_size, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_MIOPEN_DEPTHWISE_CONVOLUTION_BACKWARD_INPUT, self_size, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> wrap_miopen_depthwise_convolution_backward(const at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,3> output_mask) {
  ensure_materialized(self, grad_output, weight);
  return at::redispatch::miopen_depthwise_convolution_backward(self, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, output_mask);
}

at::Tensor wrap_miopen_depthwise_convolution_backward_weight(at::IntArrayRef weight_size, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self);
    return at::redispatch::miopen_depthwise_convolution_backward_weight(weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_MIOPEN_DEPTHWISE_CONVOLUTION_BACKWARD_WEIGHT, weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> wrap_miopen_rnn(const at::Tensor & input, at::TensorList weight, int64_t weight_stride0, const at::Tensor & hx, const c10::optional<at::Tensor> & cx, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, at::IntArrayRef batch_sizes, const c10::optional<at::Tensor> & dropout_state) {
  ensure_materialized(input, hx);
  return at::redispatch::miopen_rnn(input, weight, weight_stride0, hx, cx, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor,std::vector<at::Tensor>> wrap_miopen_rnn_backward(const at::Tensor & input, at::TensorList weight, int64_t weight_stride0, const at::Tensor & weight_buf, const at::Tensor & hx, const c10::optional<at::Tensor> & cx, const at::Tensor & output, const c10::optional<at::Tensor> & grad_output, const c10::optional<at::Tensor> & grad_hy, const c10::optional<at::Tensor> & grad_cy, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, at::IntArrayRef batch_sizes, const c10::optional<at::Tensor> & dropout_state, const at::Tensor & reserve, std::array<bool,4> output_mask) {
  ensure_materialized(input, weight_buf, hx, output, reserve);
  return at::redispatch::miopen_rnn_backward(input, weight, weight_stride0, weight_buf, hx, cx, output, grad_output, grad_hy, grad_cy, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state, reserve, output_mask);
}

at::Tensor wrap_mm(const at::Tensor & self, const at::Tensor & mat2) {
  if (trace.is_flushing()) {
    ensure_materialized(self, mat2);
    return at::redispatch::mm(self, mat2);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MM, self, mat2);
}

at::Tensor & wrap_mm_out(const at::Tensor & self, const at::Tensor & mat2, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, mat2);
    return at::redispatch::mm(out, self, mat2);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_MM_OUT, out, self, mat2);
}

at::Tensor wrap__sparse_mm(const at::Tensor & sparse, const at::Tensor & dense) {
  if (trace.is_flushing()) {
    ensure_materialized(sparse, dense);
    return at::redispatch::_sparse_mm(sparse, dense);
  }
  return MK_TORCHY(sparse.dtype(), sparse.device(), H__SPARSE_MM, sparse, dense);
}

at::Tensor wrap__sparse_sparse_matmul(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::_sparse_sparse_matmul(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__SPARSE_SPARSE_MATMUL, self, other);
}

at::Tensor wrap__sparse_mask_helper(const at::Tensor & t, const at::Tensor & mask_indices) {
  if (trace.is_flushing()) {
    ensure_materialized(t, mask_indices);
    return at::redispatch::_sparse_mask_helper(t, mask_indices);
  }
  return MK_TORCHY(t.dtype(), t.device(), H__SPARSE_MASK_HELPER, t, mask_indices);
}

std::tuple<at::Tensor,at::Tensor> wrap_mode(const at::Tensor & self, int64_t dim, bool keepdim) {
  ensure_materialized(self);
  return at::redispatch::mode(self, dim, keepdim);
}

std::tuple<at::Tensor &,at::Tensor &> wrap_mode_values(const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & values, at::Tensor & indices) {
  ensure_materialized(values, indices, self);
  return at::redispatch::mode(values, indices, self, dim, keepdim);
}

std::tuple<at::Tensor,at::Tensor> wrap_mode_dimname(const at::Tensor & self, at::Dimname dim, bool keepdim) {
  ensure_materialized(self);
  return at::redispatch::mode(self, dim, keepdim);
}

std::tuple<at::Tensor &,at::Tensor &> wrap_mode_dimname_out(const at::Tensor & self, at::Dimname dim, bool keepdim, at::Tensor & values, at::Tensor & indices) {
  ensure_materialized(values, indices, self);
  return at::redispatch::mode(values, indices, self, dim, keepdim);
}

at::Tensor wrap_mul_Tensor(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::mul(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MUL_TENSOR, self, other);
}

at::Tensor & wrap_mul__Tensor(at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::mul_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MUL__TENSOR, self, other);
}

at::Tensor & wrap_mul_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::mul(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_MUL_OUT, out, self, other);
}

at::Tensor wrap_mul_Scalar(const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::mul(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MUL_SCALAR, self, other);
}

at::Tensor & wrap_mul__Scalar(at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::mul_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MUL__SCALAR, self, other);
}

at::Tensor wrap_multiply_Tensor(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::multiply(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MULTIPLY_TENSOR, self, other);
}

at::Tensor & wrap_multiply__Tensor(at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::multiply_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MULTIPLY__TENSOR, self, other);
}

at::Tensor & wrap_multiply_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::multiply(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_MULTIPLY_OUT, out, self, other);
}

at::Tensor wrap_multiply_Scalar(const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::multiply(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MULTIPLY_SCALAR, self, other);
}

at::Tensor & wrap_multiply__Scalar(at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::multiply_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MULTIPLY__SCALAR, self, other);
}

at::Tensor wrap_mv(const at::Tensor & self, const at::Tensor & vec) {
  if (trace.is_flushing()) {
    ensure_materialized(self, vec);
    return at::redispatch::mv(self, vec);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MV, self, vec);
}

at::Tensor & wrap_mv_out(const at::Tensor & self, const at::Tensor & vec, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, vec);
    return at::redispatch::mv(out, self, vec);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_MV_OUT, out, self, vec);
}

at::Tensor wrap_mvlgamma(const at::Tensor & self, int64_t p) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::mvlgamma(self, p);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MVLGAMMA, self, p);
}

at::Tensor & wrap_mvlgamma_(at::Tensor & self, int64_t p) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::mvlgamma_(self, p);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MVLGAMMA_, self, p);
}

at::Tensor wrap_narrow_copy(const at::Tensor & self, int64_t dim, int64_t start, int64_t length) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::narrow_copy(self, dim, start, length);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NARROW_COPY, self, dim, start, length);
}

at::Tensor & wrap_narrow_copy_out(const at::Tensor & self, int64_t dim, int64_t start, int64_t length, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::narrow_copy(out, self, dim, start, length);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NARROW_COPY_OUT, out, self, dim, start, length);
}

at::Tensor wrap_narrow(const at::Tensor & self, int64_t dim, int64_t start, int64_t length) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::narrow(self, dim, start, length);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NARROW, self, dim, start, length);
}

at::Tensor wrap_narrow_Tensor(const at::Tensor & self, int64_t dim, const at::Tensor & start, int64_t length) {
  if (trace.is_flushing()) {
    ensure_materialized(self, start);
    return at::redispatch::narrow(self, dim, start, length);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NARROW_TENSOR, self, dim, start, length);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> wrap_native_batch_norm(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, bool training, double momentum, double eps) {
  ensure_materialized(input);
  return at::redispatch::native_batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps);
}

std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> wrap_native_batch_norm_out(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, bool training, double momentum, double eps, at::Tensor & out, at::Tensor & save_mean, at::Tensor & save_invstd) {
  ensure_materialized(out, save_mean, save_invstd, input);
  return at::redispatch::native_batch_norm(out, save_mean, save_invstd, input, weight, bias, running_mean, running_var, training, momentum, eps);
}

std::tuple<at::Tensor,at::Tensor> wrap_batch_norm_stats(const at::Tensor & input, double eps) {
  ensure_materialized(input);
  return at::redispatch::batch_norm_stats(input, eps);
}

at::Tensor wrap_batch_norm_elemt(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const at::Tensor & mean, const at::Tensor & invstd, double eps) {
  if (trace.is_flushing()) {
    ensure_materialized(input, mean, invstd);
    return at::redispatch::batch_norm_elemt(input, weight, bias, mean, invstd, eps);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_BATCH_NORM_ELEMT, input, weight, bias, mean, invstd, eps);
}

at::Tensor & wrap_batch_norm_elemt_out(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const at::Tensor & mean, const at::Tensor & invstd, double eps, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, input, mean, invstd);
    return at::redispatch::batch_norm_elemt(out, input, weight, bias, mean, invstd, eps);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_BATCH_NORM_ELEMT_OUT, out, input, weight, bias, mean, invstd, eps);
}

std::tuple<at::Tensor,at::Tensor> wrap_batch_norm_gather_stats(const at::Tensor & input, const at::Tensor & mean, const at::Tensor & invstd, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, double momentum, double eps, int64_t count) {
  ensure_materialized(input, mean, invstd);
  return at::redispatch::batch_norm_gather_stats(input, mean, invstd, running_mean, running_var, momentum, eps, count);
}

std::tuple<at::Tensor,at::Tensor> wrap_batch_norm_gather_stats_with_counts(const at::Tensor & input, const at::Tensor & mean, const at::Tensor & invstd, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, double momentum, double eps, const at::Tensor & counts) {
  ensure_materialized(input, mean, invstd, counts);
  return at::redispatch::batch_norm_gather_stats_with_counts(input, mean, invstd, running_mean, running_var, momentum, eps, counts);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> wrap_native_batch_norm_backward(const at::Tensor & grad_out, const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, const c10::optional<at::Tensor> & save_mean, const c10::optional<at::Tensor> & save_invstd, bool train, double eps, std::array<bool,3> output_mask) {
  ensure_materialized(grad_out, input);
  return at::redispatch::native_batch_norm_backward(grad_out, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps, output_mask);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> wrap_batch_norm_backward_reduce(const at::Tensor & grad_out, const at::Tensor & input, const at::Tensor & mean, const at::Tensor & invstd, const c10::optional<at::Tensor> & weight, bool input_g, bool weight_g, bool bias_g) {
  ensure_materialized(grad_out, input, mean, invstd);
  return at::redispatch::batch_norm_backward_reduce(grad_out, input, mean, invstd, weight, input_g, weight_g, bias_g);
}

at::Tensor wrap_batch_norm_backward_elemt(const at::Tensor & grad_out, const at::Tensor & input, const at::Tensor & mean, const at::Tensor & invstd, const c10::optional<at::Tensor> & weight, const at::Tensor & mean_dy, const at::Tensor & mean_dy_xmu, const at::Tensor & count) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_out, input, mean, invstd, mean_dy, mean_dy_xmu, count);
    return at::redispatch::batch_norm_backward_elemt(grad_out, input, mean, invstd, weight, mean_dy, mean_dy_xmu, count);
  }
  return MK_TORCHY(grad_out.dtype(), grad_out.device(), H_BATCH_NORM_BACKWARD_ELEMT, grad_out, input, mean, invstd, weight, mean_dy, mean_dy_xmu, count);
}

std::tuple<at::Tensor,at::Tensor> wrap_batch_norm_update_stats(const at::Tensor & input, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, double momentum) {
  ensure_materialized(input);
  return at::redispatch::batch_norm_update_stats(input, running_mean, running_var, momentum);
}

bool wrap_is_vulkan_available() {
  ensure_materialized();
  return at::redispatch::is_vulkan_available();
}

bool wrap__nnpack_available() {
  ensure_materialized();
  return at::redispatch::_nnpack_available();
}

at::Tensor wrap__nnpack_spatial_convolution(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef padding, at::IntArrayRef stride) {
  if (trace.is_flushing()) {
    ensure_materialized(input, weight);
    return at::redispatch::_nnpack_spatial_convolution(input, weight, bias, padding, stride);
  }
  return MK_TORCHY(input.dtype(), input.device(), H__NNPACK_SPATIAL_CONVOLUTION, input, weight, bias, padding, stride);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> wrap__nnpack_spatial_convolution_backward(const at::Tensor & input, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, std::array<bool,3> output_mask) {
  ensure_materialized(input, grad_output, weight);
  return at::redispatch::_nnpack_spatial_convolution_backward(input, grad_output, weight, padding, output_mask);
}

at::Tensor wrap__nnpack_spatial_convolution_backward_input(const at::Tensor & input, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding) {
  if (trace.is_flushing()) {
    ensure_materialized(input, grad_output, weight);
    return at::redispatch::_nnpack_spatial_convolution_backward_input(input, grad_output, weight, padding);
  }
  return MK_TORCHY(input.dtype(), input.device(), H__NNPACK_SPATIAL_CONVOLUTION_BACKWARD_INPUT, input, grad_output, weight, padding);
}

at::Tensor wrap__nnpack_spatial_convolution_backward_weight(const at::Tensor & input, at::IntArrayRef weightsize, const at::Tensor & grad_output, at::IntArrayRef padding) {
  if (trace.is_flushing()) {
    ensure_materialized(input, grad_output);
    return at::redispatch::_nnpack_spatial_convolution_backward_weight(input, weightsize, grad_output, padding);
  }
  return MK_TORCHY(input.dtype(), input.device(), H__NNPACK_SPATIAL_CONVOLUTION_BACKWARD_WEIGHT, input, weightsize, grad_output, padding);
}

at::Tensor wrap_ones_names(at::IntArrayRef size, c10::optional<at::DimnameList> names, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::ones(size, names, dtype, layout, device, pin_memory));
}

at::Tensor wrap_ones(at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::ones(size, dtype, layout, device, pin_memory));
}

at::Tensor & wrap_ones_out(at::IntArrayRef size, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::ones(out, size);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ONES_OUT, out, size);
}

at::Tensor wrap_ones_like(const at::Tensor & self, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::ones_like(self, dtype, layout, device, pin_memory, memory_format);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ONES_LIKE, self, dtype, layout, device, pin_memory, memory_format);
}

at::Tensor wrap_pairwise_distance(const at::Tensor & x1, const at::Tensor & x2, double p, double eps, bool keepdim) {
  if (trace.is_flushing()) {
    ensure_materialized(x1, x2);
    return at::redispatch::pairwise_distance(x1, x2, p, eps, keepdim);
  }
  return MK_TORCHY(x1.dtype(), x1.device(), H_PAIRWISE_DISTANCE, x1, x2, p, eps, keepdim);
}

at::Tensor wrap_cdist(const at::Tensor & x1, const at::Tensor & x2, double p, c10::optional<int64_t> compute_mode) {
  if (trace.is_flushing()) {
    ensure_materialized(x1, x2);
    return at::redispatch::cdist(x1, x2, p, compute_mode);
  }
  return MK_TORCHY(x1.dtype(), x1.device(), H_CDIST, x1, x2, p, compute_mode);
}

at::Tensor wrap__euclidean_dist(const at::Tensor & x1, const at::Tensor & x2) {
  if (trace.is_flushing()) {
    ensure_materialized(x1, x2);
    return at::redispatch::_euclidean_dist(x1, x2);
  }
  return MK_TORCHY(x1.dtype(), x1.device(), H__EUCLIDEAN_DIST, x1, x2);
}

at::Tensor wrap__cdist_forward(const at::Tensor & x1, const at::Tensor & x2, double p, c10::optional<int64_t> compute_mode) {
  if (trace.is_flushing()) {
    ensure_materialized(x1, x2);
    return at::redispatch::_cdist_forward(x1, x2, p, compute_mode);
  }
  return MK_TORCHY(x1.dtype(), x1.device(), H__CDIST_FORWARD, x1, x2, p, compute_mode);
}

at::Tensor wrap__cdist_backward(const at::Tensor & grad, const at::Tensor & x1, const at::Tensor & x2, double p, const at::Tensor & cdist) {
  if (trace.is_flushing()) {
    ensure_materialized(grad, x1, x2, cdist);
    return at::redispatch::_cdist_backward(grad, x1, x2, p, cdist);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H__CDIST_BACKWARD, grad, x1, x2, p, cdist);
}

at::Tensor wrap_pdist(const at::Tensor & self, double p) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::pdist(self, p);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_PDIST, self, p);
}

at::Tensor wrap__pdist_forward(const at::Tensor & self, double p) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_pdist_forward(self, p);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__PDIST_FORWARD, self, p);
}

at::Tensor wrap__pdist_backward(const at::Tensor & grad, const at::Tensor & self, double p, const at::Tensor & pdist) {
  if (trace.is_flushing()) {
    ensure_materialized(grad, self, pdist);
    return at::redispatch::_pdist_backward(grad, self, p, pdist);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H__PDIST_BACKWARD, grad, self, p, pdist);
}

at::Tensor wrap_cosine_similarity(const at::Tensor & x1, const at::Tensor & x2, int64_t dim, double eps) {
  if (trace.is_flushing()) {
    ensure_materialized(x1, x2);
    return at::redispatch::cosine_similarity(x1, x2, dim, eps);
  }
  return MK_TORCHY(x1.dtype(), x1.device(), H_COSINE_SIMILARITY, x1, x2, dim, eps);
}

at::Tensor wrap_permute(const at::Tensor & self, at::IntArrayRef dims) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::permute(self, dims);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_PERMUTE, self, dims);
}

at::Tensor wrap_movedim_intlist(const at::Tensor & self, at::IntArrayRef source, at::IntArrayRef destination) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::movedim(self, source, destination);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MOVEDIM_INTLIST, self, source, destination);
}

at::Tensor wrap_movedim_int(const at::Tensor & self, int64_t source, int64_t destination) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::movedim(self, source, destination);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MOVEDIM_INT, self, source, destination);
}

at::Tensor wrap_moveaxis_intlist(const at::Tensor & self, at::IntArrayRef source, at::IntArrayRef destination) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::moveaxis(self, source, destination);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MOVEAXIS_INTLIST, self, source, destination);
}

at::Tensor wrap_moveaxis_int(const at::Tensor & self, int64_t source, int64_t destination) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::moveaxis(self, source, destination);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MOVEAXIS_INT, self, source, destination);
}

at::Tensor wrap_numpy_T(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::numpy_T(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NUMPY_T, self);
}

at::Tensor wrap_pixel_shuffle(const at::Tensor & self, int64_t upscale_factor) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::pixel_shuffle(self, upscale_factor);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_PIXEL_SHUFFLE, self, upscale_factor);
}

at::Tensor wrap_pixel_unshuffle(const at::Tensor & self, int64_t downscale_factor) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::pixel_unshuffle(self, downscale_factor);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_PIXEL_UNSHUFFLE, self, downscale_factor);
}

at::Tensor wrap_channel_shuffle(const at::Tensor & self, int64_t groups) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::channel_shuffle(self, groups);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CHANNEL_SHUFFLE, self, groups);
}

bool wrap_is_pinned(const at::Tensor & self) {
  ensure_materialized(self);
  return at::redispatch::is_pinned(self);
}

at::Tensor wrap_pin_memory(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::pin_memory(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_PIN_MEMORY, self);
}

at::Tensor wrap_pinverse(const at::Tensor & self, double rcond) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::pinverse(self, rcond);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_PINVERSE, self, rcond);
}

at::Tensor wrap_poisson_nll_loss(const at::Tensor & input, const at::Tensor & target, bool log_input, bool full, double eps, int64_t reduction) {
  if (trace.is_flushing()) {
    ensure_materialized(input, target);
    return at::redispatch::poisson_nll_loss(input, target, log_input, full, eps, reduction);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_POISSON_NLL_LOSS, input, target, log_input, full, eps, reduction);
}

at::Tensor wrap_rad2deg(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::rad2deg(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RAD2DEG, self);
}

at::Tensor & wrap_rad2deg_(at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::rad2deg_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RAD2DEG_, self);
}

at::Tensor & wrap_rad2deg_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::rad2deg(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_RAD2DEG_OUT, out, self);
}

at::Tensor wrap_deg2rad(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::deg2rad(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DEG2RAD, self);
}

at::Tensor & wrap_deg2rad_(at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::deg2rad_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DEG2RAD_, self);
}

at::Tensor & wrap_deg2rad_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::deg2rad(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_DEG2RAD_OUT, out, self);
}

at::Tensor wrap_scalar_tensor(const at::Scalar & s, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::scalar_tensor(s, dtype, layout, device, pin_memory));
}

at::Tensor wrap_rand_names(at::IntArrayRef size, c10::optional<at::DimnameList> names, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::rand(size, names, dtype, layout, device, pin_memory));
}

at::Tensor wrap_rand_generator_with_names(at::IntArrayRef size, c10::optional<at::Generator> generator, c10::optional<at::DimnameList> names, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::rand(size, generator, names, dtype, layout, device, pin_memory));
}

at::Tensor wrap_rand(at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::rand(size, dtype, layout, device, pin_memory));
}

at::Tensor wrap_rand_generator(at::IntArrayRef size, c10::optional<at::Generator> generator, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::rand(size, generator, dtype, layout, device, pin_memory));
}

at::Tensor & wrap_rand_out(at::IntArrayRef size, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::rand(out, size);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_RAND_OUT, out, size);
}

at::Tensor & wrap_rand_generator_out(at::IntArrayRef size, c10::optional<at::Generator> generator, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::rand(out, size, generator);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_RAND_GENERATOR_OUT, out, size, generator);
}

at::Tensor wrap_rand_like(const at::Tensor & self, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::rand_like(self, dtype, layout, device, pin_memory, memory_format);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RAND_LIKE, self, dtype, layout, device, pin_memory, memory_format);
}

at::Tensor wrap_randint(int64_t high, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::randint(high, size, dtype, layout, device, pin_memory));
}

at::Tensor wrap_randint_generator(int64_t high, at::IntArrayRef size, c10::optional<at::Generator> generator, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::randint(high, size, generator, dtype, layout, device, pin_memory));
}

at::Tensor wrap_randint_low(int64_t low, int64_t high, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::randint(low, high, size, dtype, layout, device, pin_memory));
}

at::Tensor wrap_randint_low_generator(int64_t low, int64_t high, at::IntArrayRef size, c10::optional<at::Generator> generator, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::randint(low, high, size, generator, dtype, layout, device, pin_memory));
}

at::Tensor & wrap_randint_out(int64_t high, at::IntArrayRef size, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::randint(out, high, size);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_RANDINT_OUT, out, high, size);
}

at::Tensor & wrap_randint_generator_out(int64_t high, at::IntArrayRef size, c10::optional<at::Generator> generator, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::randint(out, high, size, generator);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_RANDINT_GENERATOR_OUT, out, high, size, generator);
}

at::Tensor & wrap_randint_low_out(int64_t low, int64_t high, at::IntArrayRef size, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::randint(out, low, high, size);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_RANDINT_LOW_OUT, out, low, high, size);
}

at::Tensor & wrap_randint_low_generator_out(int64_t low, int64_t high, at::IntArrayRef size, c10::optional<at::Generator> generator, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::randint(out, low, high, size, generator);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_RANDINT_LOW_GENERATOR_OUT, out, low, high, size, generator);
}

at::Tensor wrap_randint_like(const at::Tensor & self, int64_t high, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::randint_like(self, high, dtype, layout, device, pin_memory, memory_format);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RANDINT_LIKE, self, high, dtype, layout, device, pin_memory, memory_format);
}

at::Tensor wrap_randint_like_low_dtype(const at::Tensor & self, int64_t low, int64_t high, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::randint_like(self, low, high, dtype, layout, device, pin_memory, memory_format);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RANDINT_LIKE_LOW_DTYPE, self, low, high, dtype, layout, device, pin_memory, memory_format);
}

at::Tensor wrap_randn(at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::randn(size, dtype, layout, device, pin_memory));
}

at::Tensor wrap_randn_generator(at::IntArrayRef size, c10::optional<at::Generator> generator, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::randn(size, generator, dtype, layout, device, pin_memory));
}

at::Tensor wrap_randn_names(at::IntArrayRef size, c10::optional<at::DimnameList> names, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::randn(size, names, dtype, layout, device, pin_memory));
}

at::Tensor wrap_randn_generator_with_names(at::IntArrayRef size, c10::optional<at::Generator> generator, c10::optional<at::DimnameList> names, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::randn(size, generator, names, dtype, layout, device, pin_memory));
}

at::Tensor & wrap_randn_out(at::IntArrayRef size, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::randn(out, size);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_RANDN_OUT, out, size);
}

at::Tensor & wrap_randn_generator_out(at::IntArrayRef size, c10::optional<at::Generator> generator, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::randn(out, size, generator);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_RANDN_GENERATOR_OUT, out, size, generator);
}

at::Tensor wrap_randn_like(const at::Tensor & self, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::randn_like(self, dtype, layout, device, pin_memory, memory_format);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RANDN_LIKE, self, dtype, layout, device, pin_memory, memory_format);
}

at::Tensor wrap_randperm(int64_t n, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::randperm(n, dtype, layout, device, pin_memory));
}

at::Tensor wrap_randperm_generator(int64_t n, c10::optional<at::Generator> generator, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::randperm(n, generator, dtype, layout, device, pin_memory));
}

at::Tensor & wrap_randperm_out(int64_t n, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::randperm(out, n);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_RANDPERM_OUT, out, n);
}

at::Tensor & wrap_randperm_generator_out(int64_t n, c10::optional<at::Generator> generator, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::randperm(out, n, generator);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_RANDPERM_GENERATOR_OUT, out, n, generator);
}

at::Tensor wrap_range_step(const at::Scalar & start, const at::Scalar & end, const at::Scalar & step, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::range(start, end, step, dtype, layout, device, pin_memory));
}

at::Tensor wrap_range(const at::Scalar & start, const at::Scalar & end, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::range(start, end, dtype, layout, device, pin_memory));
}

at::Tensor & wrap_range_out(const at::Scalar & start, const at::Scalar & end, const at::Scalar & step, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::range(out, start, end, step);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_RANGE_OUT, out, start, end, step);
}

at::Tensor wrap_ravel(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::ravel(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RAVEL, self);
}

at::Tensor & wrap_reciprocal_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::reciprocal(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_RECIPROCAL_OUT, out, self);
}

at::Tensor wrap_neg(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::neg(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NEG, self);
}

at::Tensor & wrap_neg_(at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::neg_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NEG_, self);
}

at::Tensor & wrap_neg_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::neg(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NEG_OUT, out, self);
}

at::Tensor wrap_negative(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::negative(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NEGATIVE, self);
}

at::Tensor & wrap_negative_(at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::negative_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NEGATIVE_, self);
}

at::Tensor & wrap_negative_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::negative(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NEGATIVE_OUT, out, self);
}

at::Tensor wrap_repeat(const at::Tensor & self, at::IntArrayRef repeats) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::repeat(self, repeats);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_REPEAT, self, repeats);
}

at::Tensor wrap_repeat_interleave_Tensor(const at::Tensor & repeats) {
  if (trace.is_flushing()) {
    ensure_materialized(repeats);
    return at::redispatch::repeat_interleave(repeats);
  }
  return MK_TORCHY(repeats.dtype(), repeats.device(), H_REPEAT_INTERLEAVE_TENSOR, repeats);
}

at::Tensor wrap_repeat_interleave_self_Tensor(const at::Tensor & self, const at::Tensor & repeats, c10::optional<int64_t> dim) {
  if (trace.is_flushing()) {
    ensure_materialized(self, repeats);
    return at::redispatch::repeat_interleave(self, repeats, dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_REPEAT_INTERLEAVE_SELF_TENSOR, self, repeats, dim);
}

at::Tensor wrap_repeat_interleave_self_int(const at::Tensor & self, int64_t repeats, c10::optional<int64_t> dim) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::repeat_interleave(self, repeats, dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_REPEAT_INTERLEAVE_SELF_INT, self, repeats, dim);
}

at::Tensor wrap_reshape(const at::Tensor & self, at::IntArrayRef shape) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::reshape(self, shape);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RESHAPE, self, shape);
}

at::Tensor wrap__mkldnn_reshape(const at::Tensor & self, at::IntArrayRef shape) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_mkldnn_reshape(self, shape);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__MKLDNN_RESHAPE, self, shape);
}

at::Tensor wrap_reshape_as(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::reshape_as(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RESHAPE_AS, self, other);
}

at::Tensor wrap_round(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::round(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ROUND, self);
}

at::Tensor & wrap_round_(at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::round_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ROUND_, self);
}

at::Tensor & wrap_round_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::round(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ROUND_OUT, out, self);
}

at::Tensor wrap_rrelu(const at::Tensor & self, const at::Scalar & lower, const at::Scalar & upper, bool training, c10::optional<at::Generator> generator) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::rrelu(self, lower, upper, training, generator);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RRELU, self, lower, upper, training, generator);
}

at::Tensor & wrap_rrelu_(at::Tensor & self, const at::Scalar & lower, const at::Scalar & upper, bool training, c10::optional<at::Generator> generator) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::rrelu_(self, lower, upper, training, generator);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RRELU_, self, lower, upper, training, generator);
}

at::Tensor wrap_relu(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::relu(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RELU, self);
}

at::Tensor & wrap_relu_(at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::relu_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RELU_, self);
}

at::Tensor wrap_relu6(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::relu6(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RELU6, self);
}

at::Tensor & wrap_relu6_(at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::relu6_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RELU6_, self);
}

at::Tensor wrap_prelu(const at::Tensor & self, const at::Tensor & weight) {
  if (trace.is_flushing()) {
    ensure_materialized(self, weight);
    return at::redispatch::prelu(self, weight);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_PRELU, self, weight);
}

std::tuple<at::Tensor,at::Tensor> wrap_prelu_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight) {
  ensure_materialized(grad_output, self, weight);
  return at::redispatch::prelu_backward(grad_output, self, weight);
}

at::Tensor wrap_gelu(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::gelu(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_GELU, self);
}

at::Tensor wrap_gelu_backward(const at::Tensor & grad, const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(grad, self);
    return at::redispatch::gelu_backward(grad, self);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H_GELU_BACKWARD, grad, self);
}

at::Tensor wrap_infinitely_differentiable_gelu_backward(const at::Tensor & grad, const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(grad, self);
    return at::redispatch::infinitely_differentiable_gelu_backward(grad, self);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H_INFINITELY_DIFFERENTIABLE_GELU_BACKWARD, grad, self);
}

at::Tensor wrap_hardshrink(const at::Tensor & self, const at::Scalar & lambd) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::hardshrink(self, lambd);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_HARDSHRINK, self, lambd);
}

at::Tensor wrap_hardshrink_backward(const at::Tensor & grad_out, const at::Tensor & self, const at::Scalar & lambd) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_out, self);
    return at::redispatch::hardshrink_backward(grad_out, self, lambd);
  }
  return MK_TORCHY(grad_out.dtype(), grad_out.device(), H_HARDSHRINK_BACKWARD, grad_out, self, lambd);
}

at::Tensor wrap_rsqrt(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::rsqrt(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RSQRT, self);
}

at::Tensor & wrap_rsqrt_(at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::rsqrt_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RSQRT_, self);
}

at::Tensor & wrap_rsqrt_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::rsqrt(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_RSQRT_OUT, out, self);
}

at::Tensor wrap_select_Dimname(const at::Tensor & self, at::Dimname dim, int64_t index) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::select(self, dim, index);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SELECT_DIMNAME, self, dim, index);
}

at::Tensor wrap_select_int(const at::Tensor & self, int64_t dim, int64_t index) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::select(self, dim, index);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SELECT_INT, self, dim, index);
}

at::Tensor wrap_select_backward(const at::Tensor & grad, at::IntArrayRef input_sizes, int64_t dim, int64_t index) {
  if (trace.is_flushing()) {
    ensure_materialized(grad);
    return at::redispatch::select_backward(grad, input_sizes, dim, index);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H_SELECT_BACKWARD, grad, input_sizes, dim, index);
}

at::Tensor wrap_selu(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::selu(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SELU, self);
}

at::Tensor & wrap_selu_(at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::selu_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SELU_, self);
}

at::Tensor wrap_celu(const at::Tensor & self, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::celu(self, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CELU, self, alpha);
}

at::Tensor & wrap_celu_(at::Tensor & self, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::celu_(self, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CELU_, self, alpha);
}

at::Tensor wrap_silu(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::silu(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SILU, self);
}

at::Tensor & wrap_silu_(at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::silu_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SILU_, self);
}

at::Tensor & wrap_silu_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::silu(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SILU_OUT, out, self);
}

at::Tensor wrap_silu_backward(const at::Tensor & grad_output, const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self);
    return at::redispatch::silu_backward(grad_output, self);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_SILU_BACKWARD, grad_output, self);
}

at::Tensor wrap_sigmoid(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::sigmoid(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SIGMOID, self);
}

at::Tensor & wrap_sigmoid_(at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::sigmoid_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SIGMOID_, self);
}

at::Tensor & wrap_sigmoid_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::sigmoid(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SIGMOID_OUT, out, self);
}

at::Tensor wrap_logit(const at::Tensor & self, c10::optional<double> eps) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::logit(self, eps);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LOGIT, self, eps);
}

at::Tensor & wrap_logit_(at::Tensor & self, c10::optional<double> eps) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::logit_(self, eps);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LOGIT_, self, eps);
}

at::Tensor & wrap_logit_out(const at::Tensor & self, c10::optional<double> eps, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::logit(out, self, eps);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LOGIT_OUT, out, self, eps);
}

at::Tensor & wrap_sin_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::sin(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SIN_OUT, out, self);
}

at::Tensor & wrap_sinc_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::sinc(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SINC_OUT, out, self);
}

at::Tensor & wrap_sinh_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::sinh(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SINH_OUT, out, self);
}

at::Tensor wrap_detach(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::detach(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DETACH, self);
}

at::Tensor & wrap_detach_(at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::detach_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DETACH_, self);
}

int64_t wrap_size_Dimname(const at::Tensor & self, at::Dimname dim) {
  ensure_materialized(self);
  return at::redispatch::size(self, dim);
}

at::Tensor wrap_slice_Tensor(const at::Tensor & self, int64_t dim, c10::optional<int64_t> start, c10::optional<int64_t> end, int64_t step) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::slice(self, dim, start, end, step);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SLICE_TENSOR, self, dim, start, end, step);
}

at::Tensor wrap_slice_backward(const at::Tensor & grad, at::IntArrayRef input_sizes, int64_t dim, int64_t start, int64_t end, int64_t step) {
  if (trace.is_flushing()) {
    ensure_materialized(grad);
    return at::redispatch::slice_backward(grad, input_sizes, dim, start, end, step);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H_SLICE_BACKWARD, grad, input_sizes, dim, start, end, step);
}

std::tuple<at::Tensor,at::Tensor> wrap_slogdet(const at::Tensor & self) {
  ensure_materialized(self);
  return at::redispatch::slogdet(self);
}

at::Tensor wrap_smm(const at::Tensor & self, const at::Tensor & mat2) {
  if (trace.is_flushing()) {
    ensure_materialized(self, mat2);
    return at::redispatch::smm(self, mat2);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SMM, self, mat2);
}

at::Tensor wrap_softmax_int(const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::softmax(self, dim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SOFTMAX_INT, self, dim, dtype);
}

at::Tensor wrap_softmax_Dimname(const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::softmax(self, dim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SOFTMAX_DIMNAME, self, dim, dtype);
}

at::Tensor wrap__softmax(const at::Tensor & self, int64_t dim, bool half_to_float) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_softmax(self, dim, half_to_float);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__SOFTMAX, self, dim, half_to_float);
}

at::Tensor wrap__softmax_backward_data(const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, output, self);
    return at::redispatch::_softmax_backward_data(grad_output, output, dim, self);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H__SOFTMAX_BACKWARD_DATA, grad_output, output, dim, self);
}

std::vector<at::Tensor> wrap_unsafe_split_Tensor(const at::Tensor & self, int64_t split_size, int64_t dim) {
  ensure_materialized(self);
  return at::redispatch::unsafe_split(self, split_size, dim);
}

std::vector<at::Tensor> wrap_split_Tensor(const at::Tensor & self, int64_t split_size, int64_t dim) {
  ensure_materialized(self);
  return at::redispatch::split(self, split_size, dim);
}

std::vector<at::Tensor> wrap_unsafe_split_with_sizes(const at::Tensor & self, at::IntArrayRef split_sizes, int64_t dim) {
  ensure_materialized(self);
  return at::redispatch::unsafe_split_with_sizes(self, split_sizes, dim);
}

std::vector<at::Tensor> wrap_split_with_sizes(const at::Tensor & self, at::IntArrayRef split_sizes, int64_t dim) {
  ensure_materialized(self);
  return at::redispatch::split_with_sizes(self, split_sizes, dim);
}

at::Tensor wrap_squeeze(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::squeeze(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SQUEEZE, self);
}

at::Tensor wrap_squeeze_dim(const at::Tensor & self, int64_t dim) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::squeeze(self, dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SQUEEZE_DIM, self, dim);
}

at::Tensor wrap_squeeze_dimname(const at::Tensor & self, at::Dimname dim) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::squeeze(self, dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SQUEEZE_DIMNAME, self, dim);
}

at::Tensor & wrap_squeeze_(at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::squeeze_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SQUEEZE_, self);
}

at::Tensor & wrap_squeeze__dim(at::Tensor & self, int64_t dim) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::squeeze_(self, dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SQUEEZE__DIM, self, dim);
}

at::Tensor & wrap_squeeze__dimname(at::Tensor & self, at::Dimname dim) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::squeeze_(self, dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SQUEEZE__DIMNAME, self, dim);
}

at::Tensor wrap_sspaddmm(const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    ensure_materialized(self, mat1, mat2);
    return at::redispatch::sspaddmm(self, mat1, mat2, beta, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SSPADDMM, self, mat1, mat2, beta, alpha);
}

at::Tensor & wrap_sspaddmm_out(const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, mat1, mat2);
    return at::redispatch::sspaddmm(out, self, mat1, mat2, beta, alpha);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SSPADDMM_OUT, out, self, mat1, mat2, beta, alpha);
}

at::Tensor wrap_stack(at::TensorList tensors, int64_t dim) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::stack(tensors, dim));
}

at::Tensor & wrap_stack_out(at::TensorList tensors, int64_t dim, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::stack(out, tensors, dim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_STACK_OUT, out, tensors, dim);
}

at::Tensor wrap__stack(at::TensorList tensors, int64_t dim) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::_stack(tensors, dim));
}

at::Tensor & wrap__stack_out(at::TensorList tensors, int64_t dim, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::_stack(out, tensors, dim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H__STACK_OUT, out, tensors, dim);
}

at::Tensor wrap_hstack(at::TensorList tensors) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::hstack(tensors));
}

at::Tensor & wrap_hstack_out(at::TensorList tensors, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::hstack(out, tensors);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_HSTACK_OUT, out, tensors);
}

at::Tensor wrap_vstack(at::TensorList tensors) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::vstack(tensors));
}

at::Tensor & wrap_vstack_out(at::TensorList tensors, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::vstack(out, tensors);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_VSTACK_OUT, out, tensors);
}

at::Tensor wrap_dstack(at::TensorList tensors) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::dstack(tensors));
}

at::Tensor & wrap_dstack_out(at::TensorList tensors, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::dstack(out, tensors);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_DSTACK_OUT, out, tensors);
}

at::Tensor wrap_stft(const at::Tensor & self, int64_t n_fft, c10::optional<int64_t> hop_length, c10::optional<int64_t> win_length, const c10::optional<at::Tensor> & window, bool normalized, c10::optional<bool> onesided, c10::optional<bool> return_complex) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::stft(self, n_fft, hop_length, win_length, window, normalized, onesided, return_complex);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_STFT, self, n_fft, hop_length, win_length, window, normalized, onesided, return_complex);
}

at::Tensor wrap_istft(const at::Tensor & self, int64_t n_fft, c10::optional<int64_t> hop_length, c10::optional<int64_t> win_length, const c10::optional<at::Tensor> & window, bool center, bool normalized, c10::optional<bool> onesided, c10::optional<int64_t> length, bool return_complex) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::istft(self, n_fft, hop_length, win_length, window, center, normalized, onesided, length, return_complex);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ISTFT, self, n_fft, hop_length, win_length, window, center, normalized, onesided, length, return_complex);
}

int64_t wrap_stride_Dimname(const at::Tensor & self, at::Dimname dim) {
  ensure_materialized(self);
  return at::redispatch::stride(self, dim);
}

at::Tensor wrap_sum(const at::Tensor & self, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::sum(self, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SUM, self, dtype);
}

at::Tensor wrap_sum_dim_IntList(const at::Tensor & self, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::sum(self, dim, keepdim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SUM_DIM_INTLIST, self, dim, keepdim, dtype);
}

at::Tensor wrap_sum_dim_DimnameList(const at::Tensor & self, at::DimnameList dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::sum(self, dim, keepdim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SUM_DIM_DIMNAMELIST, self, dim, keepdim, dtype);
}

at::Tensor & wrap_sum_IntList_out(const at::Tensor & self, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::sum(out, self, dim, keepdim, dtype);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SUM_INTLIST_OUT, out, self, dim, keepdim, dtype);
}

at::Tensor & wrap_sum_DimnameList_out(const at::Tensor & self, at::DimnameList dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::sum(out, self, dim, keepdim, dtype);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SUM_DIMNAMELIST_OUT, out, self, dim, keepdim, dtype);
}

at::Tensor wrap_nansum(const at::Tensor & self, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::nansum(self, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NANSUM, self, dtype);
}

at::Tensor wrap_nansum_dim_IntList(const at::Tensor & self, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::nansum(self, dim, keepdim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NANSUM_DIM_INTLIST, self, dim, keepdim, dtype);
}

at::Tensor & wrap_nansum_IntList_out(const at::Tensor & self, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::nansum(out, self, dim, keepdim, dtype);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NANSUM_INTLIST_OUT, out, self, dim, keepdim, dtype);
}

at::Tensor wrap_sum_to_size(const at::Tensor & self, at::IntArrayRef size) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::sum_to_size(self, size);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SUM_TO_SIZE, self, size);
}

at::Tensor wrap_sqrt(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::sqrt(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SQRT, self);
}

at::Tensor & wrap_sqrt_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::sqrt(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SQRT_OUT, out, self);
}

at::Tensor wrap_square(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::square(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SQUARE, self);
}

at::Tensor & wrap_square_(at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::square_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SQUARE_, self);
}

at::Tensor & wrap_square_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::square(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SQUARE_OUT, out, self);
}

at::Tensor wrap_std(const at::Tensor & self, bool unbiased) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::std(self, unbiased);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_STD, self, unbiased);
}

at::Tensor wrap_std_dim(const at::Tensor & self, at::IntArrayRef dim, bool unbiased, bool keepdim) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::std(self, dim, unbiased, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_STD_DIM, self, dim, unbiased, keepdim);
}

std::tuple<at::Tensor,at::Tensor> wrap_std_mean(const at::Tensor & self, bool unbiased) {
  ensure_materialized(self);
  return at::redispatch::std_mean(self, unbiased);
}

std::tuple<at::Tensor,at::Tensor> wrap_std_mean_dim(const at::Tensor & self, at::IntArrayRef dim, bool unbiased, bool keepdim) {
  ensure_materialized(self);
  return at::redispatch::std_mean(self, dim, unbiased, keepdim);
}

std::tuple<at::Tensor,at::Tensor> wrap_std_mean_names_dim(const at::Tensor & self, at::DimnameList dim, bool unbiased, bool keepdim) {
  ensure_materialized(self);
  return at::redispatch::std_mean(self, dim, unbiased, keepdim);
}

at::Tensor & wrap_std_out(const at::Tensor & self, at::IntArrayRef dim, bool unbiased, bool keepdim, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::std(out, self, dim, unbiased, keepdim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_STD_OUT, out, self, dim, unbiased, keepdim);
}

at::Tensor wrap_std_names_dim(const at::Tensor & self, at::DimnameList dim, bool unbiased, bool keepdim) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::std(self, dim, unbiased, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_STD_NAMES_DIM, self, dim, unbiased, keepdim);
}

at::Tensor & wrap_std_names_out(const at::Tensor & self, at::DimnameList dim, bool unbiased, bool keepdim, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::std(out, self, dim, unbiased, keepdim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_STD_NAMES_OUT, out, self, dim, unbiased, keepdim);
}

at::Tensor wrap_prod(const at::Tensor & self, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::prod(self, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_PROD, self, dtype);
}

at::Tensor wrap_prod_dim_int(const at::Tensor & self, int64_t dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::prod(self, dim, keepdim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_PROD_DIM_INT, self, dim, keepdim, dtype);
}

at::Tensor & wrap_prod_int_out(const at::Tensor & self, int64_t dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::prod(out, self, dim, keepdim, dtype);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_PROD_INT_OUT, out, self, dim, keepdim, dtype);
}

at::Tensor wrap_prod_dim_Dimname(const at::Tensor & self, at::Dimname dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::prod(self, dim, keepdim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_PROD_DIM_DIMNAME, self, dim, keepdim, dtype);
}

at::Tensor & wrap_prod_Dimname_out(const at::Tensor & self, at::Dimname dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::prod(out, self, dim, keepdim, dtype);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_PROD_DIMNAME_OUT, out, self, dim, keepdim, dtype);
}

at::Tensor wrap_t(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::t(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_T, self);
}

at::Tensor & wrap_t_(at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::t_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_T_, self);
}

at::Tensor & wrap_tan_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::tan(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_TAN_OUT, out, self);
}

at::Tensor wrap_tanh(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::tanh(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TANH, self);
}

at::Tensor & wrap_tanh_(at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::tanh_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TANH_, self);
}

at::Tensor & wrap_tanh_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::tanh(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_TANH_OUT, out, self);
}

at::Tensor wrap_tensordot(const at::Tensor & self, const at::Tensor & other, at::IntArrayRef dims_self, at::IntArrayRef dims_other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::tensordot(self, other, dims_self, dims_other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TENSORDOT, self, other, dims_self, dims_other);
}

at::Tensor & wrap_tensordot_out(const at::Tensor & self, const at::Tensor & other, at::IntArrayRef dims_self, at::IntArrayRef dims_other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::tensordot(out, self, other, dims_self, dims_other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_TENSORDOT_OUT, out, self, other, dims_self, dims_other);
}

at::Tensor wrap_threshold(const at::Tensor & self, const at::Scalar & threshold, const at::Scalar & value) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::threshold(self, threshold, value);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_THRESHOLD, self, threshold, value);
}

at::Tensor & wrap_threshold_(at::Tensor & self, const at::Scalar & threshold, const at::Scalar & value) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::threshold_(self, threshold, value);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_THRESHOLD_, self, threshold, value);
}

at::Tensor & wrap_threshold_out(const at::Tensor & self, const at::Scalar & threshold, const at::Scalar & value, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::threshold(out, self, threshold, value);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_THRESHOLD_OUT, out, self, threshold, value);
}

at::Tensor wrap_threshold_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & threshold) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self);
    return at::redispatch::threshold_backward(grad_output, self, threshold);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_THRESHOLD_BACKWARD, grad_output, self, threshold);
}

at::Tensor wrap_tile(const at::Tensor & self, at::IntArrayRef dims) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::tile(self, dims);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TILE, self, dims);
}

at::Tensor wrap_transpose_int(const at::Tensor & self, int64_t dim0, int64_t dim1) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::transpose(self, dim0, dim1);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TRANSPOSE_INT, self, dim0, dim1);
}

at::Tensor wrap_transpose_Dimname(const at::Tensor & self, at::Dimname dim0, at::Dimname dim1) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::transpose(self, dim0, dim1);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TRANSPOSE_DIMNAME, self, dim0, dim1);
}

at::Tensor wrap__mkldnn_transpose(const at::Tensor & self, int64_t dim0, int64_t dim1) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_mkldnn_transpose(self, dim0, dim1);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__MKLDNN_TRANSPOSE, self, dim0, dim1);
}

at::Tensor & wrap_transpose_(at::Tensor & self, int64_t dim0, int64_t dim1) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::transpose_(self, dim0, dim1);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TRANSPOSE_, self, dim0, dim1);
}

at::Tensor & wrap__mkldnn_transpose_(at::Tensor & self, int64_t dim0, int64_t dim1) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_mkldnn_transpose_(self, dim0, dim1);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__MKLDNN_TRANSPOSE_, self, dim0, dim1);
}

at::Tensor wrap_one_hot(const at::Tensor & self, int64_t num_classes) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::one_hot(self, num_classes);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ONE_HOT, self, num_classes);
}

at::Tensor wrap_flip(const at::Tensor & self, at::IntArrayRef dims) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::flip(self, dims);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FLIP, self, dims);
}

at::Tensor wrap_fliplr(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::fliplr(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FLIPLR, self);
}

at::Tensor wrap_flipud(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::flipud(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FLIPUD, self);
}

at::Tensor wrap_roll(const at::Tensor & self, at::IntArrayRef shifts, at::IntArrayRef dims) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::roll(self, shifts, dims);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ROLL, self, shifts, dims);
}

at::Tensor wrap_rot90(const at::Tensor & self, int64_t k, at::IntArrayRef dims) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::rot90(self, k, dims);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ROT90, self, k, dims);
}

at::Tensor wrap_trapz_x(const at::Tensor & y, const at::Tensor & x, int64_t dim) {
  if (trace.is_flushing()) {
    ensure_materialized(y, x);
    return at::redispatch::trapz(y, x, dim);
  }
  return MK_TORCHY(y.dtype(), y.device(), H_TRAPZ_X, y, x, dim);
}

at::Tensor wrap_trapz_dx(const at::Tensor & y, double dx, int64_t dim) {
  if (trace.is_flushing()) {
    ensure_materialized(y);
    return at::redispatch::trapz(y, dx, dim);
  }
  return MK_TORCHY(y.dtype(), y.device(), H_TRAPZ_DX, y, dx, dim);
}

at::Tensor wrap__trilinear(const at::Tensor & i1, const at::Tensor & i2, const at::Tensor & i3, at::IntArrayRef expand1, at::IntArrayRef expand2, at::IntArrayRef expand3, at::IntArrayRef sumdim, int64_t unroll_dim) {
  if (trace.is_flushing()) {
    ensure_materialized(i1, i2, i3);
    return at::redispatch::_trilinear(i1, i2, i3, expand1, expand2, expand3, sumdim, unroll_dim);
  }
  return MK_TORCHY(i1.dtype(), i1.device(), H__TRILINEAR, i1, i2, i3, expand1, expand2, expand3, sumdim, unroll_dim);
}

at::Tensor wrap_triplet_margin_loss(const at::Tensor & anchor, const at::Tensor & positive, const at::Tensor & negative, double margin, double p, double eps, bool swap, int64_t reduction) {
  if (trace.is_flushing()) {
    ensure_materialized(anchor, positive, negative);
    return at::redispatch::triplet_margin_loss(anchor, positive, negative, margin, p, eps, swap, reduction);
  }
  return MK_TORCHY(anchor.dtype(), anchor.device(), H_TRIPLET_MARGIN_LOSS, anchor, positive, negative, margin, p, eps, swap, reduction);
}

at::Tensor wrap_trunc(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::trunc(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TRUNC, self);
}

at::Tensor & wrap_trunc_(at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::trunc_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TRUNC_, self);
}

at::Tensor & wrap_trunc_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::trunc(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_TRUNC_OUT, out, self);
}

at::Tensor wrap_fix(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::fix(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FIX, self);
}

at::Tensor & wrap_fix_(at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::fix_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FIX_, self);
}

at::Tensor & wrap_fix_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::fix(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FIX_OUT, out, self);
}

at::Tensor wrap_type_as(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::type_as(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TYPE_AS, self, other);
}

bool wrap__has_compatible_shallow_copy_type(const at::Tensor & self, const at::Tensor & from) {
  ensure_materialized(self, from);
  return at::redispatch::_has_compatible_shallow_copy_type(self, from);
}

std::tuple<at::Tensor,at::Tensor> wrap__unique(const at::Tensor & self, bool sorted, bool return_inverse) {
  ensure_materialized(self);
  return at::redispatch::_unique(self, sorted, return_inverse);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> wrap_unique_dim(const at::Tensor & self, int64_t dim, bool sorted, bool return_inverse, bool return_counts) {
  ensure_materialized(self);
  return at::redispatch::unique_dim(self, dim, sorted, return_inverse, return_counts);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> wrap_unique_consecutive(const at::Tensor & self, bool return_inverse, bool return_counts, c10::optional<int64_t> dim) {
  ensure_materialized(self);
  return at::redispatch::unique_consecutive(self, return_inverse, return_counts, dim);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> wrap_unique_dim_consecutive(const at::Tensor & self, int64_t dim, bool return_inverse, bool return_counts) {
  ensure_materialized(self);
  return at::redispatch::unique_dim_consecutive(self, dim, return_inverse, return_counts);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> wrap__unique2(const at::Tensor & self, bool sorted, bool return_inverse, bool return_counts) {
  ensure_materialized(self);
  return at::redispatch::_unique2(self, sorted, return_inverse, return_counts);
}

at::Tensor wrap__unsafe_view(const at::Tensor & self, at::IntArrayRef size) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_unsafe_view(self, size);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__UNSAFE_VIEW, self, size);
}

at::Tensor wrap_unsqueeze(const at::Tensor & self, int64_t dim) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::unsqueeze(self, dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_UNSQUEEZE, self, dim);
}

at::Tensor & wrap_unsqueeze_(at::Tensor & self, int64_t dim) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::unsqueeze_(self, dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_UNSQUEEZE_, self, dim);
}

at::Tensor wrap_vander(const at::Tensor & x, c10::optional<int64_t> N, bool increasing) {
  if (trace.is_flushing()) {
    ensure_materialized(x);
    return at::redispatch::vander(x, N, increasing);
  }
  return MK_TORCHY(x.dtype(), x.device(), H_VANDER, x, N, increasing);
}

at::Tensor wrap_var(const at::Tensor & self, bool unbiased) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::var(self, unbiased);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_VAR, self, unbiased);
}

at::Tensor wrap_var_dim(const at::Tensor & self, at::IntArrayRef dim, bool unbiased, bool keepdim) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::var(self, dim, unbiased, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_VAR_DIM, self, dim, unbiased, keepdim);
}

at::Tensor & wrap_var_out(const at::Tensor & self, at::IntArrayRef dim, bool unbiased, bool keepdim, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::var(out, self, dim, unbiased, keepdim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_VAR_OUT, out, self, dim, unbiased, keepdim);
}

at::Tensor wrap_var_names_dim(const at::Tensor & self, at::DimnameList dim, bool unbiased, bool keepdim) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::var(self, dim, unbiased, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_VAR_NAMES_DIM, self, dim, unbiased, keepdim);
}

at::Tensor & wrap_var_names_out(const at::Tensor & self, at::DimnameList dim, bool unbiased, bool keepdim, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::var(out, self, dim, unbiased, keepdim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_VAR_NAMES_OUT, out, self, dim, unbiased, keepdim);
}

std::tuple<at::Tensor,at::Tensor> wrap_var_mean(const at::Tensor & self, bool unbiased) {
  ensure_materialized(self);
  return at::redispatch::var_mean(self, unbiased);
}

std::tuple<at::Tensor,at::Tensor> wrap_var_mean_dim(const at::Tensor & self, at::IntArrayRef dim, bool unbiased, bool keepdim) {
  ensure_materialized(self);
  return at::redispatch::var_mean(self, dim, unbiased, keepdim);
}

std::tuple<at::Tensor,at::Tensor> wrap_var_mean_names_dim(const at::Tensor & self, at::DimnameList dim, bool unbiased, bool keepdim) {
  ensure_materialized(self);
  return at::redispatch::var_mean(self, dim, unbiased, keepdim);
}

at::Tensor wrap_view_as(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::view_as(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_VIEW_AS, self, other);
}

at::Tensor wrap_where_self(const at::Tensor & condition, const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(condition, self, other);
    return at::redispatch::where(condition, self, other);
  }
  return MK_TORCHY(condition.dtype(), condition.device(), H_WHERE_SELF, condition, self, other);
}

at::Tensor wrap_where_ScalarSelf(const at::Tensor & condition, const at::Scalar & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(condition, other);
    return at::redispatch::where(condition, self, other);
  }
  return MK_TORCHY(condition.dtype(), condition.device(), H_WHERE_SCALARSELF, condition, self, other);
}

at::Tensor wrap_where_ScalarOther(const at::Tensor & condition, const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(condition, self);
    return at::redispatch::where(condition, self, other);
  }
  return MK_TORCHY(condition.dtype(), condition.device(), H_WHERE_SCALAROTHER, condition, self, other);
}

at::Tensor wrap_where_Scalar(const at::Tensor & condition, const at::Scalar & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(condition);
    return at::redispatch::where(condition, self, other);
  }
  return MK_TORCHY(condition.dtype(), condition.device(), H_WHERE_SCALAR, condition, self, other);
}

std::vector<at::Tensor> wrap_where(const at::Tensor & condition) {
  ensure_materialized(condition);
  return at::redispatch::where(condition);
}

at::Tensor wrap__s_where(const at::Tensor & condition, const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(condition, self, other);
    return at::redispatch::_s_where(condition, self, other);
  }
  return MK_TORCHY(condition.dtype(), condition.device(), H__S_WHERE, condition, self, other);
}

at::Tensor wrap_norm_except_dim(const at::Tensor & v, int64_t pow, int64_t dim) {
  if (trace.is_flushing()) {
    ensure_materialized(v);
    return at::redispatch::norm_except_dim(v, pow, dim);
  }
  return MK_TORCHY(v.dtype(), v.device(), H_NORM_EXCEPT_DIM, v, pow, dim);
}

at::Tensor wrap__weight_norm(const at::Tensor & v, const at::Tensor & g, int64_t dim) {
  if (trace.is_flushing()) {
    ensure_materialized(v, g);
    return at::redispatch::_weight_norm(v, g, dim);
  }
  return MK_TORCHY(v.dtype(), v.device(), H__WEIGHT_NORM, v, g, dim);
}

std::tuple<at::Tensor,at::Tensor> wrap__weight_norm_cuda_interface(const at::Tensor & v, const at::Tensor & g, int64_t dim) {
  ensure_materialized(v, g);
  return at::redispatch::_weight_norm_cuda_interface(v, g, dim);
}

std::tuple<at::Tensor,at::Tensor> wrap__weight_norm_cuda_interface_backward(const at::Tensor & grad_w, const at::Tensor & saved_v, const at::Tensor & saved_g, const at::Tensor & saved_norms, int64_t dim) {
  ensure_materialized(grad_w, saved_v, saved_g, saved_norms);
  return at::redispatch::_weight_norm_cuda_interface_backward(grad_w, saved_v, saved_g, saved_norms, dim);
}

std::tuple<at::Tensor,at::Tensor> wrap__weight_norm_differentiable_backward(const at::Tensor & grad_w, const at::Tensor & saved_v, const at::Tensor & saved_g, const at::Tensor & saved_norms, int64_t dim) {
  ensure_materialized(grad_w, saved_v, saved_g, saved_norms);
  return at::redispatch::_weight_norm_differentiable_backward(grad_w, saved_v, saved_g, saved_norms, dim);
}

at::Tensor wrap_zeros_names(at::IntArrayRef size, c10::optional<at::DimnameList> names, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::zeros(size, names, dtype, layout, device, pin_memory));
}

at::Tensor wrap_zeros(at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::zeros(size, dtype, layout, device, pin_memory));
}

at::Tensor & wrap_zeros_out(at::IntArrayRef size, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::zeros(out, size);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ZEROS_OUT, out, size);
}

at::Tensor wrap_zeros_like(const at::Tensor & self, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::zeros_like(self, dtype, layout, device, pin_memory, memory_format);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ZEROS_LIKE, self, dtype, layout, device, pin_memory, memory_format);
}

at::Tensor wrap__standard_gamma_grad(const at::Tensor & self, const at::Tensor & output) {
  if (trace.is_flushing()) {
    ensure_materialized(self, output);
    return at::redispatch::_standard_gamma_grad(self, output);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__STANDARD_GAMMA_GRAD, self, output);
}

at::Tensor wrap__standard_gamma(const at::Tensor & self, c10::optional<at::Generator> generator) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_standard_gamma(self, generator);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__STANDARD_GAMMA, self, generator);
}

at::Tensor wrap__dirichlet_grad(const at::Tensor & x, const at::Tensor & alpha, const at::Tensor & total) {
  if (trace.is_flushing()) {
    ensure_materialized(x, alpha, total);
    return at::redispatch::_dirichlet_grad(x, alpha, total);
  }
  return MK_TORCHY(x.dtype(), x.device(), H__DIRICHLET_GRAD, x, alpha, total);
}

at::Tensor wrap__sample_dirichlet(const at::Tensor & self, c10::optional<at::Generator> generator) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_sample_dirichlet(self, generator);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__SAMPLE_DIRICHLET, self, generator);
}

at::Tensor wrap_poisson(const at::Tensor & self, c10::optional<at::Generator> generator) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::poisson(self, generator);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_POISSON, self, generator);
}

at::Tensor wrap_binomial(const at::Tensor & count, const at::Tensor & prob, c10::optional<at::Generator> generator) {
  if (trace.is_flushing()) {
    ensure_materialized(count, prob);
    return at::redispatch::binomial(count, prob, generator);
  }
  return MK_TORCHY(count.dtype(), count.device(), H_BINOMIAL, count, prob, generator);
}

at::Tensor wrap_native_norm(const at::Tensor & self, const at::Scalar & p) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::native_norm(self, p);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NATIVE_NORM, self, p);
}

at::Tensor wrap_native_norm_ScalarOpt_dim_dtype(const at::Tensor & self, const c10::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::native_norm(self, p, dim, keepdim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NATIVE_NORM_SCALAROPT_DIM_DTYPE, self, p, dim, keepdim, dtype);
}

at::Tensor wrap__sparse_sum(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_sparse_sum(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__SPARSE_SUM, self);
}

at::Tensor wrap__sparse_sum_dtype(const at::Tensor & self, at::ScalarType dtype) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_sparse_sum(self, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__SPARSE_SUM_DTYPE, self, dtype);
}

at::Tensor wrap__sparse_sum_dim(const at::Tensor & self, at::IntArrayRef dim) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_sparse_sum(self, dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__SPARSE_SUM_DIM, self, dim);
}

at::Tensor wrap__sparse_sum_dim_dtype(const at::Tensor & self, at::IntArrayRef dim, at::ScalarType dtype) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_sparse_sum(self, dim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__SPARSE_SUM_DIM_DTYPE, self, dim, dtype);
}

at::Tensor wrap__sparse_sum_backward(const at::Tensor & grad, const at::Tensor & self, at::IntArrayRef dim) {
  if (trace.is_flushing()) {
    ensure_materialized(grad, self);
    return at::redispatch::_sparse_sum_backward(grad, self, dim);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H__SPARSE_SUM_BACKWARD, grad, self, dim);
}

at::Tensor wrap__sparse_softmax_int(const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_sparse_softmax(self, dim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__SPARSE_SOFTMAX_INT, self, dim, dtype);
}

at::Tensor wrap__sparse_softmax_Dimname(const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_sparse_softmax(self, dim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__SPARSE_SOFTMAX_DIMNAME, self, dim, dtype);
}

at::Tensor wrap__sparse_softmax(const at::Tensor & self, int64_t dim, bool half_to_float) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_sparse_softmax(self, dim, half_to_float);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__SPARSE_SOFTMAX, self, dim, half_to_float);
}

at::Tensor wrap__sparse_softmax_backward_data(const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, output, self);
    return at::redispatch::_sparse_softmax_backward_data(grad_output, output, dim, self);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H__SPARSE_SOFTMAX_BACKWARD_DATA, grad_output, output, dim, self);
}

at::Tensor wrap__sparse_log_softmax_int(const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_sparse_log_softmax(self, dim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__SPARSE_LOG_SOFTMAX_INT, self, dim, dtype);
}

at::Tensor wrap__sparse_log_softmax_Dimname(const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_sparse_log_softmax(self, dim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__SPARSE_LOG_SOFTMAX_DIMNAME, self, dim, dtype);
}

at::Tensor wrap__sparse_log_softmax(const at::Tensor & self, int64_t dim, bool half_to_float) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_sparse_log_softmax(self, dim, half_to_float);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__SPARSE_LOG_SOFTMAX, self, dim, half_to_float);
}

at::Tensor wrap__sparse_log_softmax_backward_data(const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, output, self);
    return at::redispatch::_sparse_log_softmax_backward_data(grad_output, output, dim, self);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H__SPARSE_LOG_SOFTMAX_BACKWARD_DATA, grad_output, output, dim, self);
}

at::Tensor wrap_norm_ScalarOpt_dtype(const at::Tensor & self, const c10::optional<at::Scalar> & p, at::ScalarType dtype) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::norm(self, p, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NORM_SCALAROPT_DTYPE, self, p, dtype);
}

at::Tensor wrap_norm_Scalar(const at::Tensor & self, const at::Scalar & p) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::norm(self, p);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NORM_SCALAR, self, p);
}

at::Tensor wrap_norm_ScalarOpt_dim_dtype(const at::Tensor & self, const c10::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim, at::ScalarType dtype) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::norm(self, p, dim, keepdim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NORM_SCALAROPT_DIM_DTYPE, self, p, dim, keepdim, dtype);
}

at::Tensor wrap_norm_ScalarOpt_dim(const at::Tensor & self, const c10::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::norm(self, p, dim, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NORM_SCALAROPT_DIM, self, p, dim, keepdim);
}

at::Tensor & wrap_norm_dtype_out(const at::Tensor & self, const c10::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim, at::ScalarType dtype, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::norm(out, self, p, dim, keepdim, dtype);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NORM_DTYPE_OUT, out, self, p, dim, keepdim, dtype);
}

at::Tensor & wrap_norm_out(const at::Tensor & self, const c10::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::norm(out, self, p, dim, keepdim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NORM_OUT, out, self, p, dim, keepdim);
}

at::Tensor wrap_norm_names_ScalarOpt_dim_dtype(const at::Tensor & self, const c10::optional<at::Scalar> & p, at::DimnameList dim, bool keepdim, at::ScalarType dtype) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::norm(self, p, dim, keepdim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NORM_NAMES_SCALAROPT_DIM_DTYPE, self, p, dim, keepdim, dtype);
}

at::Tensor wrap_norm_names_ScalarOpt_dim(const at::Tensor & self, const c10::optional<at::Scalar> & p, at::DimnameList dim, bool keepdim) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::norm(self, p, dim, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NORM_NAMES_SCALAROPT_DIM, self, p, dim, keepdim);
}

at::Tensor & wrap_norm_names_dtype_out(const at::Tensor & self, const c10::optional<at::Scalar> & p, at::DimnameList dim, bool keepdim, at::ScalarType dtype, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::norm(out, self, p, dim, keepdim, dtype);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NORM_NAMES_DTYPE_OUT, out, self, p, dim, keepdim, dtype);
}

at::Tensor & wrap_norm_names_out(const at::Tensor & self, const c10::optional<at::Scalar> & p, at::DimnameList dim, bool keepdim, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::norm(out, self, p, dim, keepdim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NORM_NAMES_OUT, out, self, p, dim, keepdim);
}

std::tuple<at::Tensor,at::Tensor> wrap_frexp_Tensor(const at::Tensor & self) {
  ensure_materialized(self);
  return at::redispatch::frexp(self);
}

std::tuple<at::Tensor &,at::Tensor &> wrap_frexp_Tensor_out(const at::Tensor & self, at::Tensor & mantissa, at::Tensor & exponent) {
  ensure_materialized(mantissa, exponent, self);
  return at::redispatch::frexp(mantissa, exponent, self);
}

at::Tensor wrap_frobenius_norm(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::frobenius_norm(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FROBENIUS_NORM, self);
}

at::Tensor wrap_frobenius_norm_dim(const at::Tensor & self, at::IntArrayRef dim, bool keepdim) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::frobenius_norm(self, dim, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FROBENIUS_NORM_DIM, self, dim, keepdim);
}

at::Tensor & wrap_frobenius_norm_out(const at::Tensor & self, at::IntArrayRef dim, bool keepdim, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::frobenius_norm(out, self, dim, keepdim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FROBENIUS_NORM_OUT, out, self, dim, keepdim);
}

at::Tensor wrap_nuclear_norm(const at::Tensor & self, bool keepdim) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::nuclear_norm(self, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NUCLEAR_NORM, self, keepdim);
}

at::Tensor & wrap_nuclear_norm_out(const at::Tensor & self, bool keepdim, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::nuclear_norm(out, self, keepdim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NUCLEAR_NORM_OUT, out, self, keepdim);
}

at::Tensor wrap_nuclear_norm_dim(const at::Tensor & self, at::IntArrayRef dim, bool keepdim) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::nuclear_norm(self, dim, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NUCLEAR_NORM_DIM, self, dim, keepdim);
}

at::Tensor & wrap_nuclear_norm_dim_out(const at::Tensor & self, at::IntArrayRef dim, bool keepdim, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::nuclear_norm(out, self, dim, keepdim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NUCLEAR_NORM_DIM_OUT, out, self, dim, keepdim);
}

at::Tensor wrap_clone(const at::Tensor & self, c10::optional<at::MemoryFormat> memory_format) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::clone(self, memory_format);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CLONE, self, memory_format);
}

const at::Tensor & wrap_resize_as_(const at::Tensor & self, const at::Tensor & the_template, c10::optional<at::MemoryFormat> memory_format) {
  ensure_materialized(self, the_template);
  return at::redispatch::resize_as_(self, the_template, memory_format);
}

const at::Tensor & wrap_resize_as_sparse_(const at::Tensor & self, const at::Tensor & the_template) {
  ensure_materialized(self, the_template);
  return at::redispatch::resize_as_sparse_(self, the_template);
}

at::Tensor & wrap_zero_(at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::zero_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ZERO_, self);
}

at::Tensor & wrap_sub_out(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::sub(out, self, other, alpha);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SUB_OUT, out, self, other, alpha);
}

at::Tensor wrap_sub_Tensor(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::sub(self, other, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SUB_TENSOR, self, other, alpha);
}

at::Tensor & wrap_sub__Tensor(at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::sub_(self, other, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SUB__TENSOR, self, other, alpha);
}

at::Tensor wrap_sub_Scalar(const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::sub(self, other, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SUB_SCALAR, self, other, alpha);
}

at::Tensor & wrap_sub__Scalar(at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::sub_(self, other, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SUB__SCALAR, self, other, alpha);
}

at::Tensor & wrap_subtract_out(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::subtract(out, self, other, alpha);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SUBTRACT_OUT, out, self, other, alpha);
}

at::Tensor wrap_subtract_Tensor(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::subtract(self, other, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SUBTRACT_TENSOR, self, other, alpha);
}

at::Tensor & wrap_subtract__Tensor(at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::subtract_(self, other, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SUBTRACT__TENSOR, self, other, alpha);
}

at::Tensor wrap_subtract_Scalar(const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::subtract(self, other, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SUBTRACT_SCALAR, self, other, alpha);
}

at::Tensor & wrap_subtract__Scalar(at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::subtract_(self, other, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SUBTRACT__SCALAR, self, other, alpha);
}

at::Tensor wrap_rsub_Tensor(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::rsub(self, other, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RSUB_TENSOR, self, other, alpha);
}

at::Tensor & wrap_heaviside_out(const at::Tensor & self, const at::Tensor & values, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, values);
    return at::redispatch::heaviside(out, self, values);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_HEAVISIDE_OUT, out, self, values);
}

at::Tensor wrap_heaviside(const at::Tensor & self, const at::Tensor & values) {
  if (trace.is_flushing()) {
    ensure_materialized(self, values);
    return at::redispatch::heaviside(self, values);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_HEAVISIDE, self, values);
}

at::Tensor & wrap_heaviside_(at::Tensor & self, const at::Tensor & values) {
  if (trace.is_flushing()) {
    ensure_materialized(self, values);
    return at::redispatch::heaviside_(self, values);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_HEAVISIDE_, self, values);
}

at::Tensor wrap_rsub_Scalar(const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::rsub(self, other, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RSUB_SCALAR, self, other, alpha);
}

at::Tensor wrap__sparse_addmm(const at::Tensor & self, const at::Tensor & sparse, const at::Tensor & dense, const at::Scalar & beta, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    ensure_materialized(self, sparse, dense);
    return at::redispatch::_sparse_addmm(self, sparse, dense, beta, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__SPARSE_ADDMM, self, sparse, dense, beta, alpha);
}

at::Tensor & wrap_addmm_out(const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, mat1, mat2);
    return at::redispatch::addmm(out, self, mat1, mat2, beta, alpha);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ADDMM_OUT, out, self, mat1, mat2, beta, alpha);
}

at::Tensor wrap_addmm(const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    ensure_materialized(self, mat1, mat2);
    return at::redispatch::addmm(self, mat1, mat2, beta, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ADDMM, self, mat1, mat2, beta, alpha);
}

at::Tensor & wrap_addmm_(at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    ensure_materialized(self, mat1, mat2);
    return at::redispatch::addmm_(self, mat1, mat2, beta, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ADDMM_, self, mat1, mat2, beta, alpha);
}

at::Tensor wrap_sparse_csr_tensor_crow_col_value_size(const at::Tensor & crow_indices, const at::Tensor & col_indices, const at::Tensor & values, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    ensure_materialized(crow_indices, col_indices, values);
    return at::redispatch::sparse_csr_tensor(crow_indices, col_indices, values, size, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(crow_indices.dtype(), crow_indices.device(), H_SPARSE_CSR_TENSOR_CROW_COL_VALUE_SIZE, crow_indices, col_indices, values, size, dtype, layout, device, pin_memory);
}

at::Tensor wrap_sparse_csr_tensor_crow_col_value(const at::Tensor & crow_indices, const at::Tensor & col_indices, const at::Tensor & values, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    ensure_materialized(crow_indices, col_indices, values);
    return at::redispatch::sparse_csr_tensor(crow_indices, col_indices, values, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(crow_indices.dtype(), crow_indices.device(), H_SPARSE_CSR_TENSOR_CROW_COL_VALUE, crow_indices, col_indices, values, dtype, layout, device, pin_memory);
}

at::Tensor wrap_sparse_coo_tensor_size(at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::sparse_coo_tensor(size, dtype, layout, device, pin_memory));
}

at::Tensor wrap_sparse_coo_tensor_indices(const at::Tensor & indices, const at::Tensor & values, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    ensure_materialized(indices, values);
    return at::redispatch::sparse_coo_tensor(indices, values, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(indices.dtype(), indices.device(), H_SPARSE_COO_TENSOR_INDICES, indices, values, dtype, layout, device, pin_memory);
}

at::Tensor wrap_sparse_coo_tensor_indices_size(const at::Tensor & indices, const at::Tensor & values, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    ensure_materialized(indices, values);
    return at::redispatch::sparse_coo_tensor(indices, values, size, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(indices.dtype(), indices.device(), H_SPARSE_COO_TENSOR_INDICES_SIZE, indices, values, size, dtype, layout, device, pin_memory);
}

at::Tensor wrap__sparse_coo_tensor_unsafe(const at::Tensor & indices, const at::Tensor & values, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    ensure_materialized(indices, values);
    return at::redispatch::_sparse_coo_tensor_unsafe(indices, values, size, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(indices.dtype(), indices.device(), H__SPARSE_COO_TENSOR_UNSAFE, indices, values, size, dtype, layout, device, pin_memory);
}

void wrap__validate_sparse_coo_tensor_args(const at::Tensor & indices, const at::Tensor & values, at::IntArrayRef size) {
  ensure_materialized(indices, values);
  return at::redispatch::_validate_sparse_coo_tensor_args(indices, values, size);
}

at::Tensor wrap__sparse_coo_tensor_with_dims(int64_t sparse_dim, int64_t dense_dim, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::_sparse_coo_tensor_with_dims(sparse_dim, dense_dim, size, dtype, layout, device, pin_memory));
}

at::Tensor wrap__sparse_coo_tensor_with_dims_and_tensors(int64_t sparse_dim, int64_t dense_dim, at::IntArrayRef size, const at::Tensor & indices, const at::Tensor & values, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    ensure_materialized(indices, values);
    return at::redispatch::_sparse_coo_tensor_with_dims_and_tensors(sparse_dim, dense_dim, size, indices, values, dtype, layout, device, pin_memory);
  }
  return MK_TORCHY(indices.dtype(), indices.device(), H__SPARSE_COO_TENSOR_WITH_DIMS_AND_TENSORS, sparse_dim, dense_dim, size, indices, values, dtype, layout, device, pin_memory);
}

const at::Tensor & wrap_sparse_resize_(const at::Tensor & self, at::IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) {
  ensure_materialized(self);
  return at::redispatch::sparse_resize_(self, size, sparse_dim, dense_dim);
}

const at::Tensor & wrap_sparse_resize_and_clear_(const at::Tensor & self, at::IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) {
  ensure_materialized(self);
  return at::redispatch::sparse_resize_and_clear_(self, size, sparse_dim, dense_dim);
}

at::Tensor wrap_sparse_mask(const at::Tensor & self, const at::Tensor & mask) {
  if (trace.is_flushing()) {
    ensure_materialized(self, mask);
    return at::redispatch::sparse_mask(self, mask);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SPARSE_MASK, self, mask);
}

at::Tensor wrap_to_dense(const at::Tensor & self, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::to_dense(self, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TO_DENSE, self, dtype);
}

at::Tensor wrap_to_dense_backward(const at::Tensor & grad, const at::Tensor & input) {
  if (trace.is_flushing()) {
    ensure_materialized(grad, input);
    return at::redispatch::to_dense_backward(grad, input);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H_TO_DENSE_BACKWARD, grad, input);
}

int64_t wrap_sparse_dim(const at::Tensor & self) {
  ensure_materialized(self);
  return at::redispatch::sparse_dim(self);
}

int64_t wrap__dimI(const at::Tensor & self) {
  ensure_materialized(self);
  return at::redispatch::_dimI(self);
}

int64_t wrap_dense_dim(const at::Tensor & self) {
  ensure_materialized(self);
  return at::redispatch::dense_dim(self);
}

int64_t wrap__dimV(const at::Tensor & self) {
  ensure_materialized(self);
  return at::redispatch::_dimV(self);
}

int64_t wrap__nnz(const at::Tensor & self) {
  ensure_materialized(self);
  return at::redispatch::_nnz(self);
}

at::Tensor wrap_coalesce(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::coalesce(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_COALESCE, self);
}

at::Tensor wrap__coalesce(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_coalesce(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__COALESCE, self);
}

bool wrap_is_coalesced(const at::Tensor & self) {
  ensure_materialized(self);
  return at::redispatch::is_coalesced(self);
}

at::Tensor wrap__indices(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_indices(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__INDICES, self);
}

at::Tensor wrap__values(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_values(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__VALUES, self);
}

at::Tensor & wrap__coalesced_(at::Tensor & self, bool coalesced) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_coalesced_(self, coalesced);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__COALESCED_, self, coalesced);
}

at::Tensor wrap_indices(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::indices(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INDICES, self);
}

at::Tensor wrap_values(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::values(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_VALUES, self);
}

at::Tensor wrap_crow_indices(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::crow_indices(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CROW_INDICES, self);
}

at::Tensor wrap_col_indices(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::col_indices(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_COL_INDICES, self);
}

at::Tensor & wrap_hspmm_out(const at::Tensor & mat1, const at::Tensor & mat2, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, mat1, mat2);
    return at::redispatch::hspmm(out, mat1, mat2);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_HSPMM_OUT, out, mat1, mat2);
}

at::Tensor wrap_hspmm(const at::Tensor & mat1, const at::Tensor & mat2) {
  if (trace.is_flushing()) {
    ensure_materialized(mat1, mat2);
    return at::redispatch::hspmm(mat1, mat2);
  }
  return MK_TORCHY(mat1.dtype(), mat1.device(), H_HSPMM, mat1, mat2);
}

at::Tensor & wrap_copy_sparse_to_sparse_(at::Tensor & self, const at::Tensor & src, bool non_blocking) {
  if (trace.is_flushing()) {
    ensure_materialized(self, src);
    return at::redispatch::copy_sparse_to_sparse_(self, src, non_blocking);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_COPY_SPARSE_TO_SPARSE_, self, src, non_blocking);
}

std::vector<at::Tensor> wrap_unbind_int(const at::Tensor & self, int64_t dim) {
  ensure_materialized(self);
  return at::redispatch::unbind(self, dim);
}

std::vector<at::Tensor> wrap_unbind_Dimname(const at::Tensor & self, at::Dimname dim) {
  ensure_materialized(self);
  return at::redispatch::unbind(self, dim);
}

at::Tensor wrap_to_sparse_sparse_dim(const at::Tensor & self, int64_t sparse_dim) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::to_sparse(self, sparse_dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TO_SPARSE_SPARSE_DIM, self, sparse_dim);
}

at::Tensor wrap_to_sparse(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::to_sparse(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TO_SPARSE, self);
}

at::Tensor wrap_to_mkldnn(const at::Tensor & self, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::to_mkldnn(self, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TO_MKLDNN, self, dtype);
}

at::Tensor wrap_mkldnn_reorder_conv2d_weight(const at::Tensor & self, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::mkldnn_reorder_conv2d_weight(self, padding, stride, dilation, groups);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MKLDNN_REORDER_CONV2D_WEIGHT, self, padding, stride, dilation, groups);
}

at::Tensor wrap_mkldnn_reorder_conv3d_weight(const at::Tensor & self, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::mkldnn_reorder_conv3d_weight(self, padding, stride, dilation, groups);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MKLDNN_REORDER_CONV3D_WEIGHT, self, padding, stride, dilation, groups);
}

at::Tensor wrap_to_mkldnn_backward(const at::Tensor & grad, const at::Tensor & input) {
  if (trace.is_flushing()) {
    ensure_materialized(grad, input);
    return at::redispatch::to_mkldnn_backward(grad, input);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H_TO_MKLDNN_BACKWARD, grad, input);
}

at::Tensor wrap_quantize_per_tensor(const at::Tensor & self, double scale, int64_t zero_point, at::ScalarType dtype) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::quantize_per_tensor(self, scale, zero_point, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_QUANTIZE_PER_TENSOR, self, scale, zero_point, dtype);
}

std::vector<at::Tensor> wrap_quantize_per_tensor_tensors(at::TensorList tensors, const at::Tensor & scales, const at::Tensor & zero_points, at::ScalarType dtype) {
  ensure_materialized(scales, zero_points);
  return at::redispatch::quantize_per_tensor(tensors, scales, zero_points, dtype);
}

at::Tensor wrap_quantize_per_channel(const at::Tensor & self, const at::Tensor & scales, const at::Tensor & zero_points, int64_t axis, at::ScalarType dtype) {
  if (trace.is_flushing()) {
    ensure_materialized(self, scales, zero_points);
    return at::redispatch::quantize_per_channel(self, scales, zero_points, axis, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_QUANTIZE_PER_CHANNEL, self, scales, zero_points, axis, dtype);
}

at::Tensor wrap_dequantize_self(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::dequantize(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DEQUANTIZE_SELF, self);
}

std::vector<at::Tensor> wrap_dequantize_tensors(at::TensorList tensors) {
  ensure_materialized();
  return at::redispatch::dequantize(tensors);
}

double wrap_q_scale(const at::Tensor & self) {
  ensure_materialized(self);
  return at::redispatch::q_scale(self);
}

int64_t wrap_q_zero_point(const at::Tensor & self) {
  ensure_materialized(self);
  return at::redispatch::q_zero_point(self);
}

at::Tensor wrap_q_per_channel_scales(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::q_per_channel_scales(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_Q_PER_CHANNEL_SCALES, self);
}

at::Tensor wrap_q_per_channel_zero_points(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::q_per_channel_zero_points(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_Q_PER_CHANNEL_ZERO_POINTS, self);
}

int64_t wrap_q_per_channel_axis(const at::Tensor & self) {
  ensure_materialized(self);
  return at::redispatch::q_per_channel_axis(self);
}

at::Tensor wrap_int_repr(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::int_repr(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INT_REPR, self);
}

at::Tensor wrap__make_per_tensor_quantized_tensor(const at::Tensor & self, double scale, int64_t zero_point) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_make_per_tensor_quantized_tensor(self, scale, zero_point);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__MAKE_PER_TENSOR_QUANTIZED_TENSOR, self, scale, zero_point);
}

at::Tensor wrap__make_per_channel_quantized_tensor(const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t axis) {
  if (trace.is_flushing()) {
    ensure_materialized(self, scale, zero_point);
    return at::redispatch::_make_per_channel_quantized_tensor(self, scale, zero_point, axis);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__MAKE_PER_CHANNEL_QUANTIZED_TENSOR, self, scale, zero_point, axis);
}

at::QScheme wrap_qscheme(const at::Tensor & self) {
  ensure_materialized(self);
  return at::redispatch::qscheme(self);
}

at::Tensor wrap_fake_quantize_per_tensor_affine(const at::Tensor & self, double scale, int64_t zero_point, int64_t quant_min, int64_t quant_max) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::fake_quantize_per_tensor_affine(self, scale, zero_point, quant_min, quant_max);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FAKE_QUANTIZE_PER_TENSOR_AFFINE, self, scale, zero_point, quant_min, quant_max);
}

std::tuple<at::Tensor,at::Tensor> wrap_fake_quantize_per_tensor_affine_cachemask(const at::Tensor & self, double scale, int64_t zero_point, int64_t quant_min, int64_t quant_max) {
  ensure_materialized(self);
  return at::redispatch::fake_quantize_per_tensor_affine_cachemask(self, scale, zero_point, quant_min, quant_max);
}

at::Tensor wrap_fake_quantize_per_tensor_affine_cachemask_backward(const at::Tensor & grad, const at::Tensor & mask) {
  if (trace.is_flushing()) {
    ensure_materialized(grad, mask);
    return at::redispatch::fake_quantize_per_tensor_affine_cachemask_backward(grad, mask);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H_FAKE_QUANTIZE_PER_TENSOR_AFFINE_CACHEMASK_BACKWARD, grad, mask);
}

at::Tensor wrap__fake_quantize_learnable_per_tensor_affine(const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t quant_min, int64_t quant_max, double grad_factor) {
  if (trace.is_flushing()) {
    ensure_materialized(self, scale, zero_point);
    return at::redispatch::_fake_quantize_learnable_per_tensor_affine(self, scale, zero_point, quant_min, quant_max, grad_factor);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__FAKE_QUANTIZE_LEARNABLE_PER_TENSOR_AFFINE, self, scale, zero_point, quant_min, quant_max, grad_factor);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> wrap__fake_quantize_learnable_per_tensor_affine_backward(const at::Tensor & grad, const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t quant_min, int64_t quant_max, double grad_factor) {
  ensure_materialized(grad, self, scale, zero_point);
  return at::redispatch::_fake_quantize_learnable_per_tensor_affine_backward(grad, self, scale, zero_point, quant_min, quant_max, grad_factor);
}

at::Tensor wrap_fake_quantize_per_channel_affine(const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max) {
  if (trace.is_flushing()) {
    ensure_materialized(self, scale, zero_point);
    return at::redispatch::fake_quantize_per_channel_affine(self, scale, zero_point, axis, quant_min, quant_max);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FAKE_QUANTIZE_PER_CHANNEL_AFFINE, self, scale, zero_point, axis, quant_min, quant_max);
}

std::tuple<at::Tensor,at::Tensor> wrap_fake_quantize_per_channel_affine_cachemask(const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max) {
  ensure_materialized(self, scale, zero_point);
  return at::redispatch::fake_quantize_per_channel_affine_cachemask(self, scale, zero_point, axis, quant_min, quant_max);
}

at::Tensor wrap_fake_quantize_per_channel_affine_cachemask_backward(const at::Tensor & grad, const at::Tensor & mask) {
  if (trace.is_flushing()) {
    ensure_materialized(grad, mask);
    return at::redispatch::fake_quantize_per_channel_affine_cachemask_backward(grad, mask);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H_FAKE_QUANTIZE_PER_CHANNEL_AFFINE_CACHEMASK_BACKWARD, grad, mask);
}

at::Tensor wrap__fake_quantize_learnable_per_channel_affine(const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max, double grad_factor) {
  if (trace.is_flushing()) {
    ensure_materialized(self, scale, zero_point);
    return at::redispatch::_fake_quantize_learnable_per_channel_affine(self, scale, zero_point, axis, quant_min, quant_max, grad_factor);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__FAKE_QUANTIZE_LEARNABLE_PER_CHANNEL_AFFINE, self, scale, zero_point, axis, quant_min, quant_max, grad_factor);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> wrap__fake_quantize_learnable_per_channel_affine_backward(const at::Tensor & grad, const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max, double grad_factor) {
  ensure_materialized(grad, self, scale, zero_point);
  return at::redispatch::_fake_quantize_learnable_per_channel_affine_backward(grad, self, scale, zero_point, axis, quant_min, quant_max, grad_factor);
}

std::tuple<double,int64_t> wrap__choose_qparams_per_tensor(const at::Tensor & self, bool reduce_range) {
  ensure_materialized(self);
  return at::redispatch::_choose_qparams_per_tensor(self, reduce_range);
}

at::Tensor wrap__saturate_weight_to_fp16(const at::Tensor & weight) {
  if (trace.is_flushing()) {
    ensure_materialized(weight);
    return at::redispatch::_saturate_weight_to_fp16(weight);
  }
  return MK_TORCHY(weight.dtype(), weight.device(), H__SATURATE_WEIGHT_TO_FP16, weight);
}

std::tuple<at::Tensor,at::Tensor> wrap_choose_qparams_optimized(const at::Tensor & input, int64_t numel, int64_t n_bins, double ratio, int64_t bit_width) {
  ensure_materialized(input);
  return at::redispatch::choose_qparams_optimized(input, numel, n_bins, ratio, bit_width);
}

at::Tensor wrap_to_dtype_layout(const at::Tensor & self, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, bool non_blocking, bool copy, c10::optional<at::MemoryFormat> memory_format) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::to(self, dtype, layout, device, pin_memory, non_blocking, copy, memory_format);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TO_DTYPE_LAYOUT, self, dtype, layout, device, pin_memory, non_blocking, copy, memory_format);
}

at::Tensor wrap_to_device(const at::Tensor & self, at::Device device, at::ScalarType dtype, bool non_blocking, bool copy, c10::optional<at::MemoryFormat> memory_format) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::to(self, device, dtype, non_blocking, copy, memory_format);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TO_DEVICE, self, device, dtype, non_blocking, copy, memory_format);
}

at::Tensor wrap_to_dtype(const at::Tensor & self, at::ScalarType dtype, bool non_blocking, bool copy, c10::optional<at::MemoryFormat> memory_format) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::to(self, dtype, non_blocking, copy, memory_format);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TO_DTYPE, self, dtype, non_blocking, copy, memory_format);
}

at::Tensor wrap_to_other(const at::Tensor & self, const at::Tensor & other, bool non_blocking, bool copy, c10::optional<at::MemoryFormat> memory_format) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::to(self, other, non_blocking, copy, memory_format);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TO_OTHER, self, other, non_blocking, copy, memory_format);
}

std::vector<at::Tensor> wrap_meshgrid(at::TensorList tensors) {
  ensure_materialized();
  return at::redispatch::meshgrid(tensors);
}

at::Tensor wrap_cartesian_prod(at::TensorList tensors) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::cartesian_prod(tensors));
}

at::Tensor wrap_combinations(const at::Tensor & self, int64_t r, bool with_replacement) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::combinations(self, r, with_replacement);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_COMBINATIONS, self, r, with_replacement);
}

at::Scalar wrap_item(const at::Tensor & self) {
  ensure_materialized(self);
  return at::redispatch::item(self);
}

at::ScalarType wrap_result_type_Tensor(const at::Tensor & tensor, const at::Tensor & other) {
  ensure_materialized(tensor, other);
  return at::redispatch::result_type(tensor, other);
}

at::ScalarType wrap_result_type_Scalar(const at::Tensor & tensor, const at::Scalar & other) {
  ensure_materialized(tensor);
  return at::redispatch::result_type(tensor, other);
}

at::ScalarType wrap_result_type_Scalar_Tensor(const at::Scalar & scalar, const at::Tensor & tensor) {
  ensure_materialized(tensor);
  return at::redispatch::result_type(scalar, tensor);
}

at::ScalarType wrap_result_type_Scalar_Scalar(const at::Scalar & scalar1, const at::Scalar & scalar2) {
  ensure_materialized();
  return at::redispatch::result_type(scalar1, scalar2);
}

bool wrap_can_cast(at::ScalarType from, at::ScalarType to) {
  ensure_materialized();
  return at::redispatch::can_cast(from, to);
}

at::ScalarType wrap_promote_types(at::ScalarType type1, at::ScalarType type2) {
  ensure_materialized();
  return at::redispatch::promote_types(type1, type2);
}

at::Scalar wrap__local_scalar_dense(const at::Tensor & self) {
  ensure_materialized(self);
  return at::redispatch::_local_scalar_dense(self);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> wrap__thnn_fused_lstm_cell(const at::Tensor & input_gates, const at::Tensor & hidden_gates, const at::Tensor & cx, const c10::optional<at::Tensor> & input_bias, const c10::optional<at::Tensor> & hidden_bias) {
  ensure_materialized(input_gates, hidden_gates, cx);
  return at::redispatch::_thnn_fused_lstm_cell(input_gates, hidden_gates, cx, input_bias, hidden_bias);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> wrap__thnn_fused_lstm_cell_backward(const c10::optional<at::Tensor> & grad_hy, const c10::optional<at::Tensor> & grad_cy, const at::Tensor & cx, const at::Tensor & cy, const at::Tensor & workspace, bool has_bias) {
  ensure_materialized(cx, cy, workspace);
  return at::redispatch::_thnn_fused_lstm_cell_backward(grad_hy, grad_cy, cx, cy, workspace, has_bias);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> wrap__thnn_differentiable_lstm_cell_backward(const c10::optional<at::Tensor> & grad_hy, const c10::optional<at::Tensor> & grad_cy, const at::Tensor & input_gates, const at::Tensor & hidden_gates, const c10::optional<at::Tensor> & input_bias, const c10::optional<at::Tensor> & hidden_bias, const at::Tensor & cx, const at::Tensor & cy) {
  ensure_materialized(input_gates, hidden_gates, cx, cy);
  return at::redispatch::_thnn_differentiable_lstm_cell_backward(grad_hy, grad_cy, input_gates, hidden_gates, input_bias, hidden_bias, cx, cy);
}

std::tuple<at::Tensor,at::Tensor> wrap__thnn_fused_gru_cell(const at::Tensor & input_gates, const at::Tensor & hidden_gates, const at::Tensor & hx, const c10::optional<at::Tensor> & input_bias, const c10::optional<at::Tensor> & hidden_bias) {
  ensure_materialized(input_gates, hidden_gates, hx);
  return at::redispatch::_thnn_fused_gru_cell(input_gates, hidden_gates, hx, input_bias, hidden_bias);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> wrap__thnn_fused_gru_cell_backward(const at::Tensor & grad_hy, const at::Tensor & workspace, bool has_bias) {
  ensure_materialized(grad_hy, workspace);
  return at::redispatch::_thnn_fused_gru_cell_backward(grad_hy, workspace, has_bias);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> wrap__thnn_differentiable_gru_cell_backward(const at::Tensor & grad_hy, const at::Tensor & input_gates, const at::Tensor & hidden_gates, const at::Tensor & hx, const c10::optional<at::Tensor> & input_bias, const c10::optional<at::Tensor> & hidden_bias) {
  ensure_materialized(grad_hy, input_gates, hidden_gates, hx);
  return at::redispatch::_thnn_differentiable_gru_cell_backward(grad_hy, input_gates, hidden_gates, hx, input_bias, hidden_bias);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> wrap_lstm_input(const at::Tensor & input, at::TensorList hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) {
  ensure_materialized(input);
  return at::redispatch::lstm(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> wrap_lstm_data(const at::Tensor & data, const at::Tensor & batch_sizes, at::TensorList hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) {
  ensure_materialized(data, batch_sizes);
  return at::redispatch::lstm(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
}

std::tuple<at::Tensor,at::Tensor> wrap_gru_input(const at::Tensor & input, const at::Tensor & hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) {
  ensure_materialized(input, hx);
  return at::redispatch::gru(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
}

std::tuple<at::Tensor,at::Tensor> wrap_gru_data(const at::Tensor & data, const at::Tensor & batch_sizes, const at::Tensor & hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) {
  ensure_materialized(data, batch_sizes, hx);
  return at::redispatch::gru(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
}

std::tuple<at::Tensor,at::Tensor> wrap_rnn_tanh_input(const at::Tensor & input, const at::Tensor & hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) {
  ensure_materialized(input, hx);
  return at::redispatch::rnn_tanh(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
}

std::tuple<at::Tensor,at::Tensor> wrap_rnn_tanh_data(const at::Tensor & data, const at::Tensor & batch_sizes, const at::Tensor & hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) {
  ensure_materialized(data, batch_sizes, hx);
  return at::redispatch::rnn_tanh(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
}

std::tuple<at::Tensor,at::Tensor> wrap_rnn_relu_input(const at::Tensor & input, const at::Tensor & hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) {
  ensure_materialized(input, hx);
  return at::redispatch::rnn_relu(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
}

std::tuple<at::Tensor,at::Tensor> wrap_rnn_relu_data(const at::Tensor & data, const at::Tensor & batch_sizes, const at::Tensor & hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) {
  ensure_materialized(data, batch_sizes, hx);
  return at::redispatch::rnn_relu(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
}

std::tuple<at::Tensor,at::Tensor> wrap_lstm_cell(const at::Tensor & input, at::TensorList hx, const at::Tensor & w_ih, const at::Tensor & w_hh, const c10::optional<at::Tensor> & b_ih, const c10::optional<at::Tensor> & b_hh) {
  ensure_materialized(input, w_ih, w_hh);
  return at::redispatch::lstm_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
}

at::Tensor wrap_gru_cell(const at::Tensor & input, const at::Tensor & hx, const at::Tensor & w_ih, const at::Tensor & w_hh, const c10::optional<at::Tensor> & b_ih, const c10::optional<at::Tensor> & b_hh) {
  if (trace.is_flushing()) {
    ensure_materialized(input, hx, w_ih, w_hh);
    return at::redispatch::gru_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_GRU_CELL, input, hx, w_ih, w_hh, b_ih, b_hh);
}

at::Tensor wrap_rnn_tanh_cell(const at::Tensor & input, const at::Tensor & hx, const at::Tensor & w_ih, const at::Tensor & w_hh, const c10::optional<at::Tensor> & b_ih, const c10::optional<at::Tensor> & b_hh) {
  if (trace.is_flushing()) {
    ensure_materialized(input, hx, w_ih, w_hh);
    return at::redispatch::rnn_tanh_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_RNN_TANH_CELL, input, hx, w_ih, w_hh, b_ih, b_hh);
}

at::Tensor wrap_rnn_relu_cell(const at::Tensor & input, const at::Tensor & hx, const at::Tensor & w_ih, const at::Tensor & w_hh, const c10::optional<at::Tensor> & b_ih, const c10::optional<at::Tensor> & b_hh) {
  if (trace.is_flushing()) {
    ensure_materialized(input, hx, w_ih, w_hh);
    return at::redispatch::rnn_relu_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_RNN_RELU_CELL, input, hx, w_ih, w_hh, b_ih, b_hh);
}

std::tuple<at::Tensor,at::Tensor> wrap_quantized_lstm_cell(const at::Tensor & input, at::TensorList hx, const at::Tensor & w_ih, const at::Tensor & w_hh, const at::Tensor & b_ih, const at::Tensor & b_hh, const at::Tensor & packed_ih, const at::Tensor & packed_hh, const at::Tensor & col_offsets_ih, const at::Tensor & col_offsets_hh, const at::Scalar & scale_ih, const at::Scalar & scale_hh, const at::Scalar & zero_point_ih, const at::Scalar & zero_point_hh) {
  ensure_materialized(input, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh);
  return at::redispatch::quantized_lstm_cell(input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
}

at::Tensor wrap_quantized_gru_cell(const at::Tensor & input, const at::Tensor & hx, const at::Tensor & w_ih, const at::Tensor & w_hh, const at::Tensor & b_ih, const at::Tensor & b_hh, const at::Tensor & packed_ih, const at::Tensor & packed_hh, const at::Tensor & col_offsets_ih, const at::Tensor & col_offsets_hh, const at::Scalar & scale_ih, const at::Scalar & scale_hh, const at::Scalar & zero_point_ih, const at::Scalar & zero_point_hh) {
  if (trace.is_flushing()) {
    ensure_materialized(input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh);
    return at::redispatch::quantized_gru_cell(input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_QUANTIZED_GRU_CELL, input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
}

at::Tensor wrap_quantized_rnn_relu_cell(const at::Tensor & input, const at::Tensor & hx, const at::Tensor & w_ih, const at::Tensor & w_hh, const at::Tensor & b_ih, const at::Tensor & b_hh, const at::Tensor & packed_ih, const at::Tensor & packed_hh, const at::Tensor & col_offsets_ih, const at::Tensor & col_offsets_hh, const at::Scalar & scale_ih, const at::Scalar & scale_hh, const at::Scalar & zero_point_ih, const at::Scalar & zero_point_hh) {
  if (trace.is_flushing()) {
    ensure_materialized(input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh);
    return at::redispatch::quantized_rnn_relu_cell(input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_QUANTIZED_RNN_RELU_CELL, input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
}

at::Tensor wrap_quantized_rnn_tanh_cell(const at::Tensor & input, const at::Tensor & hx, const at::Tensor & w_ih, const at::Tensor & w_hh, const at::Tensor & b_ih, const at::Tensor & b_hh, const at::Tensor & packed_ih, const at::Tensor & packed_hh, const at::Tensor & col_offsets_ih, const at::Tensor & col_offsets_hh, const at::Scalar & scale_ih, const at::Scalar & scale_hh, const at::Scalar & zero_point_ih, const at::Scalar & zero_point_hh) {
  if (trace.is_flushing()) {
    ensure_materialized(input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh);
    return at::redispatch::quantized_rnn_tanh_cell(input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_QUANTIZED_RNN_TANH_CELL, input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
}

std::tuple<at::Tensor,at::Tensor> wrap__pack_padded_sequence(const at::Tensor & input, const at::Tensor & lengths, bool batch_first) {
  ensure_materialized(input, lengths);
  return at::redispatch::_pack_padded_sequence(input, lengths, batch_first);
}

at::Tensor wrap__pack_padded_sequence_backward(const at::Tensor & grad, at::IntArrayRef input_size, const at::Tensor & batch_sizes, bool batch_first) {
  if (trace.is_flushing()) {
    ensure_materialized(grad, batch_sizes);
    return at::redispatch::_pack_padded_sequence_backward(grad, input_size, batch_sizes, batch_first);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H__PACK_PADDED_SEQUENCE_BACKWARD, grad, input_size, batch_sizes, batch_first);
}

std::tuple<at::Tensor,at::Tensor> wrap__pad_packed_sequence(const at::Tensor & data, const at::Tensor & batch_sizes, bool batch_first, const at::Scalar & padding_value, int64_t total_length) {
  ensure_materialized(data, batch_sizes);
  return at::redispatch::_pad_packed_sequence(data, batch_sizes, batch_first, padding_value, total_length);
}

at::Tensor & wrap_set__source_Storage(at::Tensor & self, at::Storage source) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::set_(self, source);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SET__SOURCE_STORAGE, self, source);
}

at::Tensor & wrap_set__source_Storage_storage_offset(at::Tensor & self, at::Storage source, int64_t storage_offset, at::IntArrayRef size, at::IntArrayRef stride) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::set_(self, source, storage_offset, size, stride);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SET__SOURCE_STORAGE_STORAGE_OFFSET, self, source, storage_offset, size, stride);
}

at::Tensor & wrap_set__source_Tensor(at::Tensor & self, const at::Tensor & source) {
  if (trace.is_flushing()) {
    ensure_materialized(self, source);
    return at::redispatch::set_(self, source);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SET__SOURCE_TENSOR, self, source);
}

at::Tensor & wrap_set_(at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::set_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SET_, self);
}

bool wrap_is_set_to(const at::Tensor & self, const at::Tensor & tensor) {
  ensure_materialized(self, tensor);
  return at::redispatch::is_set_to(self, tensor);
}

at::Tensor & wrap_masked_fill__Scalar(at::Tensor & self, const at::Tensor & mask, const at::Scalar & value) {
  if (trace.is_flushing()) {
    ensure_materialized(self, mask);
    return at::redispatch::masked_fill_(self, mask, value);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MASKED_FILL__SCALAR, self, mask, value);
}

at::Tensor wrap_masked_fill_Scalar(const at::Tensor & self, const at::Tensor & mask, const at::Scalar & value) {
  if (trace.is_flushing()) {
    ensure_materialized(self, mask);
    return at::redispatch::masked_fill(self, mask, value);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MASKED_FILL_SCALAR, self, mask, value);
}

at::Tensor & wrap_masked_fill__Tensor(at::Tensor & self, const at::Tensor & mask, const at::Tensor & value) {
  if (trace.is_flushing()) {
    ensure_materialized(self, mask, value);
    return at::redispatch::masked_fill_(self, mask, value);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MASKED_FILL__TENSOR, self, mask, value);
}

at::Tensor wrap_masked_fill_Tensor(const at::Tensor & self, const at::Tensor & mask, const at::Tensor & value) {
  if (trace.is_flushing()) {
    ensure_materialized(self, mask, value);
    return at::redispatch::masked_fill(self, mask, value);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MASKED_FILL_TENSOR, self, mask, value);
}

at::Tensor & wrap_masked_scatter_(at::Tensor & self, const at::Tensor & mask, const at::Tensor & source) {
  if (trace.is_flushing()) {
    ensure_materialized(self, mask, source);
    return at::redispatch::masked_scatter_(self, mask, source);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MASKED_SCATTER_, self, mask, source);
}

at::Tensor wrap_masked_scatter(const at::Tensor & self, const at::Tensor & mask, const at::Tensor & source) {
  if (trace.is_flushing()) {
    ensure_materialized(self, mask, source);
    return at::redispatch::masked_scatter(self, mask, source);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MASKED_SCATTER, self, mask, source);
}

at::Tensor wrap_view(const at::Tensor & self, at::IntArrayRef size) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::view(self, size);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_VIEW, self, size);
}

at::Tensor wrap_view_dtype(const at::Tensor & self, at::ScalarType dtype) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::view(self, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_VIEW_DTYPE, self, dtype);
}

at::Tensor & wrap_put_(at::Tensor & self, const at::Tensor & index, const at::Tensor & source, bool accumulate) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index, source);
    return at::redispatch::put_(self, index, source, accumulate);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_PUT_, self, index, source, accumulate);
}

at::Tensor wrap_put(const at::Tensor & self, const at::Tensor & index, const at::Tensor & source, bool accumulate) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index, source);
    return at::redispatch::put(self, index, source, accumulate);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_PUT, self, index, source, accumulate);
}

at::Tensor & wrap_index_add_(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index, source);
    return at::redispatch::index_add_(self, dim, index, source);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INDEX_ADD_, self, dim, index, source);
}

at::Tensor & wrap_index_add__alpha(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index, source);
    return at::redispatch::index_add_(self, dim, index, source, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INDEX_ADD__ALPHA, self, dim, index, source, alpha);
}

at::Tensor wrap_index_add(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index, source);
    return at::redispatch::index_add(self, dim, index, source);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INDEX_ADD, self, dim, index, source);
}

at::Tensor wrap_index_add_alpha(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index, source);
    return at::redispatch::index_add(self, dim, index, source, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INDEX_ADD_ALPHA, self, dim, index, source, alpha);
}

at::Tensor wrap_index_add_dimname(const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & source, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index, source);
    return at::redispatch::index_add(self, dim, index, source, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INDEX_ADD_DIMNAME, self, dim, index, source, alpha);
}

at::Tensor & wrap_index_fill__int_Scalar(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index);
    return at::redispatch::index_fill_(self, dim, index, value);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INDEX_FILL__INT_SCALAR, self, dim, index, value);
}

at::Tensor wrap_index_fill_int_Scalar(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index);
    return at::redispatch::index_fill(self, dim, index, value);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INDEX_FILL_INT_SCALAR, self, dim, index, value);
}

at::Tensor & wrap_index_fill__int_Tensor(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & value) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index, value);
    return at::redispatch::index_fill_(self, dim, index, value);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INDEX_FILL__INT_TENSOR, self, dim, index, value);
}

at::Tensor wrap_index_fill_int_Tensor(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & value) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index, value);
    return at::redispatch::index_fill(self, dim, index, value);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INDEX_FILL_INT_TENSOR, self, dim, index, value);
}

at::Tensor & wrap_index_fill__Dimname_Scalar(at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Scalar & value) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index);
    return at::redispatch::index_fill_(self, dim, index, value);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INDEX_FILL__DIMNAME_SCALAR, self, dim, index, value);
}

at::Tensor & wrap_index_fill__Dimname_Tensor(at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & value) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index, value);
    return at::redispatch::index_fill_(self, dim, index, value);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INDEX_FILL__DIMNAME_TENSOR, self, dim, index, value);
}

at::Tensor wrap_index_fill_Dimname_Scalar(const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Scalar & value) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index);
    return at::redispatch::index_fill(self, dim, index, value);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INDEX_FILL_DIMNAME_SCALAR, self, dim, index, value);
}

at::Tensor wrap_index_fill_Dimname_Tensor(const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & value) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index, value);
    return at::redispatch::index_fill(self, dim, index, value);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INDEX_FILL_DIMNAME_TENSOR, self, dim, index, value);
}

at::Tensor & wrap_scatter__src(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index, src);
    return at::redispatch::scatter_(self, dim, index, src);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SCATTER__SRC, self, dim, index, src);
}

at::Tensor wrap_scatter_src(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index, src);
    return at::redispatch::scatter(self, dim, index, src);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SCATTER_SRC, self, dim, index, src);
}

at::Tensor & wrap_scatter__value(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index);
    return at::redispatch::scatter_(self, dim, index, value);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SCATTER__VALUE, self, dim, index, value);
}

at::Tensor wrap_scatter_value(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index);
    return at::redispatch::scatter(self, dim, index, value);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SCATTER_VALUE, self, dim, index, value);
}

at::Tensor wrap_scatter_dimname_src(const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & src) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index, src);
    return at::redispatch::scatter(self, dim, index, src);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SCATTER_DIMNAME_SRC, self, dim, index, src);
}

at::Tensor wrap_scatter_dimname_value(const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Scalar & value) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index);
    return at::redispatch::scatter(self, dim, index, value);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SCATTER_DIMNAME_VALUE, self, dim, index, value);
}

at::Tensor & wrap_scatter__reduce(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, std::string reduce) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index, src);
    return at::redispatch::scatter_(self, dim, index, src, reduce);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SCATTER__REDUCE, self, dim, index, src, reduce);
}

at::Tensor & wrap_scatter__value_reduce(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value, std::string reduce) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index);
    return at::redispatch::scatter_(self, dim, index, value, reduce);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SCATTER__VALUE_REDUCE, self, dim, index, value, reduce);
}

at::Tensor & wrap_scatter_add_(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index, src);
    return at::redispatch::scatter_add_(self, dim, index, src);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SCATTER_ADD_, self, dim, index, src);
}

at::Tensor wrap_scatter_add(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index, src);
    return at::redispatch::scatter_add(self, dim, index, src);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SCATTER_ADD, self, dim, index, src);
}

at::Tensor wrap_scatter_add_dimname(const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & src) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index, src);
    return at::redispatch::scatter_add(self, dim, index, src);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SCATTER_ADD_DIMNAME, self, dim, index, src);
}

at::Tensor & wrap_eq__Scalar(at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::eq_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_EQ__SCALAR, self, other);
}

at::Tensor & wrap_eq__Tensor(at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::eq_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_EQ__TENSOR, self, other);
}

at::Tensor & wrap_bitwise_and_Tensor_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::bitwise_and(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_BITWISE_AND_TENSOR_OUT, out, self, other);
}

at::Tensor & wrap_bitwise_and_Scalar_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::bitwise_and(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_BITWISE_AND_SCALAR_OUT, out, self, other);
}

at::Tensor wrap_bitwise_and_Scalar(const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::bitwise_and(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BITWISE_AND_SCALAR, self, other);
}

at::Tensor wrap_bitwise_and_Tensor(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::bitwise_and(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BITWISE_AND_TENSOR, self, other);
}

at::Tensor & wrap_bitwise_and__Scalar(at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::bitwise_and_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BITWISE_AND__SCALAR, self, other);
}

at::Tensor & wrap_bitwise_and__Tensor(at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::bitwise_and_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BITWISE_AND__TENSOR, self, other);
}

at::Tensor wrap___and___Scalar(const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::__and__(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H___AND___SCALAR, self, other);
}

at::Tensor wrap___and___Tensor(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::__and__(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H___AND___TENSOR, self, other);
}

at::Tensor & wrap___iand___Scalar(at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::__iand__(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H___IAND___SCALAR, self, other);
}

at::Tensor & wrap___iand___Tensor(at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::__iand__(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H___IAND___TENSOR, self, other);
}

at::Tensor & wrap_bitwise_or_Tensor_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::bitwise_or(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_BITWISE_OR_TENSOR_OUT, out, self, other);
}

at::Tensor & wrap_bitwise_or_Scalar_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::bitwise_or(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_BITWISE_OR_SCALAR_OUT, out, self, other);
}

at::Tensor wrap_bitwise_or_Scalar(const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::bitwise_or(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BITWISE_OR_SCALAR, self, other);
}

at::Tensor wrap_bitwise_or_Tensor(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::bitwise_or(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BITWISE_OR_TENSOR, self, other);
}

at::Tensor & wrap_bitwise_or__Scalar(at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::bitwise_or_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BITWISE_OR__SCALAR, self, other);
}

at::Tensor & wrap_bitwise_or__Tensor(at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::bitwise_or_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BITWISE_OR__TENSOR, self, other);
}

at::Tensor wrap___or___Scalar(const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::__or__(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H___OR___SCALAR, self, other);
}

at::Tensor wrap___or___Tensor(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::__or__(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H___OR___TENSOR, self, other);
}

at::Tensor & wrap___ior___Scalar(at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::__ior__(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H___IOR___SCALAR, self, other);
}

at::Tensor & wrap___ior___Tensor(at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::__ior__(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H___IOR___TENSOR, self, other);
}

at::Tensor & wrap_bitwise_xor_Tensor_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::bitwise_xor(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_BITWISE_XOR_TENSOR_OUT, out, self, other);
}

at::Tensor & wrap_bitwise_xor_Scalar_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::bitwise_xor(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_BITWISE_XOR_SCALAR_OUT, out, self, other);
}

at::Tensor wrap_bitwise_xor_Scalar(const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::bitwise_xor(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BITWISE_XOR_SCALAR, self, other);
}

at::Tensor wrap_bitwise_xor_Tensor(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::bitwise_xor(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BITWISE_XOR_TENSOR, self, other);
}

at::Tensor & wrap_bitwise_xor__Scalar(at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::bitwise_xor_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BITWISE_XOR__SCALAR, self, other);
}

at::Tensor & wrap_bitwise_xor__Tensor(at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::bitwise_xor_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BITWISE_XOR__TENSOR, self, other);
}

at::Tensor wrap___xor___Scalar(const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::__xor__(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H___XOR___SCALAR, self, other);
}

at::Tensor wrap___xor___Tensor(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::__xor__(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H___XOR___TENSOR, self, other);
}

at::Tensor & wrap___ixor___Scalar(at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::__ixor__(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H___IXOR___SCALAR, self, other);
}

at::Tensor & wrap___ixor___Tensor(at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::__ixor__(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H___IXOR___TENSOR, self, other);
}

at::Tensor wrap___lshift___Scalar(const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::__lshift__(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H___LSHIFT___SCALAR, self, other);
}

at::Tensor wrap___lshift___Tensor(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::__lshift__(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H___LSHIFT___TENSOR, self, other);
}

at::Tensor & wrap___ilshift___Scalar(at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::__ilshift__(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H___ILSHIFT___SCALAR, self, other);
}

at::Tensor & wrap___ilshift___Tensor(at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::__ilshift__(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H___ILSHIFT___TENSOR, self, other);
}

at::Tensor wrap___rshift___Scalar(const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::__rshift__(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H___RSHIFT___SCALAR, self, other);
}

at::Tensor wrap___rshift___Tensor(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::__rshift__(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H___RSHIFT___TENSOR, self, other);
}

at::Tensor & wrap___irshift___Scalar(at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::__irshift__(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H___IRSHIFT___SCALAR, self, other);
}

at::Tensor & wrap___irshift___Tensor(at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::__irshift__(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H___IRSHIFT___TENSOR, self, other);
}

at::Tensor & wrap_tril_(at::Tensor & self, int64_t diagonal) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::tril_(self, diagonal);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TRIL_, self, diagonal);
}

at::Tensor & wrap_triu_(at::Tensor & self, int64_t diagonal) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::triu_(self, diagonal);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TRIU_, self, diagonal);
}

at::Tensor & wrap_renorm_(at::Tensor & self, const at::Scalar & p, int64_t dim, const at::Scalar & maxnorm) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::renorm_(self, p, dim, maxnorm);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RENORM_, self, p, dim, maxnorm);
}

at::Tensor & wrap_lerp__Scalar(at::Tensor & self, const at::Tensor & end, const at::Scalar & weight) {
  if (trace.is_flushing()) {
    ensure_materialized(self, end);
    return at::redispatch::lerp_(self, end, weight);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LERP__SCALAR, self, end, weight);
}

at::Tensor & wrap_lerp__Tensor(at::Tensor & self, const at::Tensor & end, const at::Tensor & weight) {
  if (trace.is_flushing()) {
    ensure_materialized(self, end, weight);
    return at::redispatch::lerp_(self, end, weight);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LERP__TENSOR, self, end, weight);
}

at::Tensor & wrap_fmod__Scalar(at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::fmod_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FMOD__SCALAR, self, other);
}

at::Tensor & wrap_fmod__Tensor(at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::fmod_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FMOD__TENSOR, self, other);
}

at::Tensor & wrap_remainder__Scalar(at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::remainder_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_REMAINDER__SCALAR, self, other);
}

at::Tensor & wrap_remainder__Tensor(at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::remainder_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_REMAINDER__TENSOR, self, other);
}

at::Tensor & wrap_addbmm_(at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    ensure_materialized(self, batch1, batch2);
    return at::redispatch::addbmm_(self, batch1, batch2, beta, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ADDBMM_, self, batch1, batch2, beta, alpha);
}

at::Tensor & wrap_addbmm_out(const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, batch1, batch2);
    return at::redispatch::addbmm(out, self, batch1, batch2, beta, alpha);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ADDBMM_OUT, out, self, batch1, batch2, beta, alpha);
}

at::Tensor wrap_addbmm(const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    ensure_materialized(self, batch1, batch2);
    return at::redispatch::addbmm(self, batch1, batch2, beta, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ADDBMM, self, batch1, batch2, beta, alpha);
}

at::Tensor & wrap_addcdiv_(at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value) {
  if (trace.is_flushing()) {
    ensure_materialized(self, tensor1, tensor2);
    return at::redispatch::addcdiv_(self, tensor1, tensor2, value);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ADDCDIV_, self, tensor1, tensor2, value);
}

at::Tensor & wrap_random__from(at::Tensor & self, int64_t from, c10::optional<int64_t> to, c10::optional<at::Generator> generator) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::random_(self, from, to, generator);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RANDOM__FROM, self, from, to, generator);
}

at::Tensor & wrap_random__to(at::Tensor & self, int64_t to, c10::optional<at::Generator> generator) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::random_(self, to, generator);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RANDOM__TO, self, to, generator);
}

at::Tensor & wrap_random_(at::Tensor & self, c10::optional<at::Generator> generator) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::random_(self, generator);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RANDOM_, self, generator);
}

at::Tensor & wrap_uniform_(at::Tensor & self, double from, double to, c10::optional<at::Generator> generator) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::uniform_(self, from, to, generator);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_UNIFORM_, self, from, to, generator);
}

at::Tensor & wrap_cauchy_(at::Tensor & self, double median, double sigma, c10::optional<at::Generator> generator) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::cauchy_(self, median, sigma, generator);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CAUCHY_, self, median, sigma, generator);
}

at::Tensor & wrap_log_normal_(at::Tensor & self, double mean, double std, c10::optional<at::Generator> generator) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::log_normal_(self, mean, std, generator);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LOG_NORMAL_, self, mean, std, generator);
}

at::Tensor & wrap_exponential_(at::Tensor & self, double lambd, c10::optional<at::Generator> generator) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::exponential_(self, lambd, generator);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_EXPONENTIAL_, self, lambd, generator);
}

at::Tensor & wrap_geometric_(at::Tensor & self, double p, c10::optional<at::Generator> generator) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::geometric_(self, p, generator);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_GEOMETRIC_, self, p, generator);
}

at::Tensor & wrap_diag_out(const at::Tensor & self, int64_t diagonal, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::diag(out, self, diagonal);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_DIAG_OUT, out, self, diagonal);
}

at::Tensor wrap_diag(const at::Tensor & self, int64_t diagonal) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::diag(self, diagonal);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DIAG, self, diagonal);
}

at::Tensor wrap_diag_backward(const at::Tensor & grad, at::IntArrayRef input_sizes, int64_t diagonal) {
  if (trace.is_flushing()) {
    ensure_materialized(grad);
    return at::redispatch::diag_backward(grad, input_sizes, diagonal);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H_DIAG_BACKWARD, grad, input_sizes, diagonal);
}

at::Tensor & wrap_cross_out(const at::Tensor & self, const at::Tensor & other, c10::optional<int64_t> dim, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::cross(out, self, other, dim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_CROSS_OUT, out, self, other, dim);
}

at::Tensor wrap_cross(const at::Tensor & self, const at::Tensor & other, c10::optional<int64_t> dim) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::cross(self, other, dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CROSS, self, other, dim);
}

at::Tensor & wrap_triu_out(const at::Tensor & self, int64_t diagonal, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::triu(out, self, diagonal);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_TRIU_OUT, out, self, diagonal);
}

at::Tensor wrap_triu(const at::Tensor & self, int64_t diagonal) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::triu(self, diagonal);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TRIU, self, diagonal);
}

at::Tensor & wrap_tril_out(const at::Tensor & self, int64_t diagonal, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::tril(out, self, diagonal);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_TRIL_OUT, out, self, diagonal);
}

at::Tensor wrap_tril(const at::Tensor & self, int64_t diagonal) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::tril(self, diagonal);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TRIL, self, diagonal);
}

at::Tensor wrap_tril_indices(int64_t row, int64_t col, int64_t offset, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::tril_indices(row, col, offset, dtype, layout, device, pin_memory));
}

at::Tensor wrap_triu_indices(int64_t row, int64_t col, int64_t offset, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::triu_indices(row, col, offset, dtype, layout, device, pin_memory));
}

at::Tensor wrap_trace(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::trace(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TRACE, self);
}

at::Tensor wrap_trace_backward(const at::Tensor & grad, at::IntArrayRef sizes) {
  if (trace.is_flushing()) {
    ensure_materialized(grad);
    return at::redispatch::trace_backward(grad, sizes);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H_TRACE_BACKWARD, grad, sizes);
}

at::Tensor & wrap_ne_Scalar_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::ne(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NE_SCALAR_OUT, out, self, other);
}

at::Tensor wrap_ne_Scalar(const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::ne(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NE_SCALAR, self, other);
}

at::Tensor & wrap_ne_Tensor_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::ne(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NE_TENSOR_OUT, out, self, other);
}

at::Tensor wrap_ne_Tensor(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::ne(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NE_TENSOR, self, other);
}

at::Tensor & wrap_ne__Scalar(at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::ne_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NE__SCALAR, self, other);
}

at::Tensor & wrap_ne__Tensor(at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::ne_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NE__TENSOR, self, other);
}

at::Tensor & wrap_not_equal_Scalar_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::not_equal(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NOT_EQUAL_SCALAR_OUT, out, self, other);
}

at::Tensor wrap_not_equal_Scalar(const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::not_equal(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NOT_EQUAL_SCALAR, self, other);
}

at::Tensor & wrap_not_equal_Tensor_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::not_equal(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NOT_EQUAL_TENSOR_OUT, out, self, other);
}

at::Tensor wrap_not_equal_Tensor(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::not_equal(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NOT_EQUAL_TENSOR, self, other);
}

at::Tensor & wrap_not_equal__Scalar(at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::not_equal_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NOT_EQUAL__SCALAR, self, other);
}

at::Tensor & wrap_not_equal__Tensor(at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::not_equal_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NOT_EQUAL__TENSOR, self, other);
}

at::Tensor & wrap_eq_Scalar_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::eq(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_EQ_SCALAR_OUT, out, self, other);
}

at::Tensor wrap_eq_Scalar(const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::eq(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_EQ_SCALAR, self, other);
}

at::Tensor & wrap_eq_Tensor_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::eq(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_EQ_TENSOR_OUT, out, self, other);
}

at::Tensor wrap_eq_Tensor(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::eq(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_EQ_TENSOR, self, other);
}

at::Tensor & wrap_ge_Scalar_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::ge(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_GE_SCALAR_OUT, out, self, other);
}

at::Tensor wrap_ge_Scalar(const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::ge(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_GE_SCALAR, self, other);
}

at::Tensor & wrap_ge_Tensor_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::ge(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_GE_TENSOR_OUT, out, self, other);
}

at::Tensor wrap_ge_Tensor(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::ge(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_GE_TENSOR, self, other);
}

at::Tensor & wrap_ge__Scalar(at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::ge_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_GE__SCALAR, self, other);
}

at::Tensor & wrap_ge__Tensor(at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::ge_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_GE__TENSOR, self, other);
}

at::Tensor & wrap_greater_equal_Scalar_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::greater_equal(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_GREATER_EQUAL_SCALAR_OUT, out, self, other);
}

at::Tensor wrap_greater_equal_Scalar(const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::greater_equal(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_GREATER_EQUAL_SCALAR, self, other);
}

at::Tensor & wrap_greater_equal_Tensor_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::greater_equal(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_GREATER_EQUAL_TENSOR_OUT, out, self, other);
}

at::Tensor wrap_greater_equal_Tensor(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::greater_equal(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_GREATER_EQUAL_TENSOR, self, other);
}

at::Tensor & wrap_greater_equal__Scalar(at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::greater_equal_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_GREATER_EQUAL__SCALAR, self, other);
}

at::Tensor & wrap_greater_equal__Tensor(at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::greater_equal_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_GREATER_EQUAL__TENSOR, self, other);
}

at::Tensor & wrap_le_Scalar_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::le(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LE_SCALAR_OUT, out, self, other);
}

at::Tensor wrap_le_Scalar(const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::le(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LE_SCALAR, self, other);
}

at::Tensor & wrap_le_Tensor_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::le(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LE_TENSOR_OUT, out, self, other);
}

at::Tensor wrap_le_Tensor(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::le(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LE_TENSOR, self, other);
}

at::Tensor & wrap_le__Scalar(at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::le_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LE__SCALAR, self, other);
}

at::Tensor & wrap_le__Tensor(at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::le_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LE__TENSOR, self, other);
}

at::Tensor & wrap_less_equal_Scalar_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::less_equal(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LESS_EQUAL_SCALAR_OUT, out, self, other);
}

at::Tensor wrap_less_equal_Scalar(const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::less_equal(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LESS_EQUAL_SCALAR, self, other);
}

at::Tensor & wrap_less_equal_Tensor_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::less_equal(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LESS_EQUAL_TENSOR_OUT, out, self, other);
}

at::Tensor wrap_less_equal_Tensor(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::less_equal(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LESS_EQUAL_TENSOR, self, other);
}

at::Tensor & wrap_less_equal__Scalar(at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::less_equal_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LESS_EQUAL__SCALAR, self, other);
}

at::Tensor & wrap_less_equal__Tensor(at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::less_equal_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LESS_EQUAL__TENSOR, self, other);
}

at::Tensor & wrap_gt_Scalar_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::gt(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_GT_SCALAR_OUT, out, self, other);
}

at::Tensor wrap_gt_Scalar(const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::gt(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_GT_SCALAR, self, other);
}

at::Tensor & wrap_gt_Tensor_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::gt(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_GT_TENSOR_OUT, out, self, other);
}

at::Tensor wrap_gt_Tensor(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::gt(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_GT_TENSOR, self, other);
}

at::Tensor & wrap_gt__Scalar(at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::gt_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_GT__SCALAR, self, other);
}

at::Tensor & wrap_gt__Tensor(at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::gt_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_GT__TENSOR, self, other);
}

at::Tensor & wrap_greater_Scalar_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::greater(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_GREATER_SCALAR_OUT, out, self, other);
}

at::Tensor wrap_greater_Scalar(const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::greater(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_GREATER_SCALAR, self, other);
}

at::Tensor & wrap_greater_Tensor_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::greater(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_GREATER_TENSOR_OUT, out, self, other);
}

at::Tensor wrap_greater_Tensor(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::greater(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_GREATER_TENSOR, self, other);
}

at::Tensor & wrap_greater__Scalar(at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::greater_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_GREATER__SCALAR, self, other);
}

at::Tensor & wrap_greater__Tensor(at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::greater_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_GREATER__TENSOR, self, other);
}

at::Tensor & wrap_lt_Scalar_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::lt(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LT_SCALAR_OUT, out, self, other);
}

at::Tensor wrap_lt_Scalar(const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::lt(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LT_SCALAR, self, other);
}

at::Tensor & wrap_lt_Tensor_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::lt(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LT_TENSOR_OUT, out, self, other);
}

at::Tensor wrap_lt_Tensor(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::lt(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LT_TENSOR, self, other);
}

at::Tensor & wrap_lt__Scalar(at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::lt_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LT__SCALAR, self, other);
}

at::Tensor & wrap_lt__Tensor(at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::lt_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LT__TENSOR, self, other);
}

at::Tensor & wrap_less_Scalar_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::less(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LESS_SCALAR_OUT, out, self, other);
}

at::Tensor wrap_less_Scalar(const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::less(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LESS_SCALAR, self, other);
}

at::Tensor & wrap_less_Tensor_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::less(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LESS_TENSOR_OUT, out, self, other);
}

at::Tensor wrap_less_Tensor(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::less(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LESS_TENSOR, self, other);
}

at::Tensor & wrap_less__Scalar(at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::less_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LESS__SCALAR, self, other);
}

at::Tensor & wrap_less__Tensor(at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::less_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LESS__TENSOR, self, other);
}

at::Tensor & wrap_take_out(const at::Tensor & self, const at::Tensor & index, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, index);
    return at::redispatch::take(out, self, index);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_TAKE_OUT, out, self, index);
}

at::Tensor wrap_take(const at::Tensor & self, const at::Tensor & index) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index);
    return at::redispatch::take(self, index);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TAKE, self, index);
}

at::Tensor & wrap_take_along_dim_out(const at::Tensor & self, const at::Tensor & indices, c10::optional<int64_t> dim, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, indices);
    return at::redispatch::take_along_dim(out, self, indices, dim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_TAKE_ALONG_DIM_OUT, out, self, indices, dim);
}

at::Tensor wrap_take_along_dim(const at::Tensor & self, const at::Tensor & indices, c10::optional<int64_t> dim) {
  if (trace.is_flushing()) {
    ensure_materialized(self, indices);
    return at::redispatch::take_along_dim(self, indices, dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_TAKE_ALONG_DIM, self, indices, dim);
}

at::Tensor & wrap_index_select_out(const at::Tensor & self, int64_t dim, const at::Tensor & index, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, index);
    return at::redispatch::index_select(out, self, dim, index);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_INDEX_SELECT_OUT, out, self, dim, index);
}

at::Tensor wrap_index_select(const at::Tensor & self, int64_t dim, const at::Tensor & index) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index);
    return at::redispatch::index_select(self, dim, index);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INDEX_SELECT, self, dim, index);
}

at::Tensor & wrap_index_select_dimname_out(const at::Tensor & self, at::Dimname dim, const at::Tensor & index, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, index);
    return at::redispatch::index_select(out, self, dim, index);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_INDEX_SELECT_DIMNAME_OUT, out, self, dim, index);
}

at::Tensor wrap_index_select_dimname(const at::Tensor & self, at::Dimname dim, const at::Tensor & index) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index);
    return at::redispatch::index_select(self, dim, index);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INDEX_SELECT_DIMNAME, self, dim, index);
}

at::Tensor wrap_index_select_backward(const at::Tensor & grad, at::IntArrayRef self_sizes, int64_t dim, const at::Tensor & index) {
  if (trace.is_flushing()) {
    ensure_materialized(grad, index);
    return at::redispatch::index_select_backward(grad, self_sizes, dim, index);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H_INDEX_SELECT_BACKWARD, grad, self_sizes, dim, index);
}

at::Tensor & wrap_masked_select_out(const at::Tensor & self, const at::Tensor & mask, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, mask);
    return at::redispatch::masked_select(out, self, mask);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_MASKED_SELECT_OUT, out, self, mask);
}

at::Tensor wrap_masked_select(const at::Tensor & self, const at::Tensor & mask) {
  if (trace.is_flushing()) {
    ensure_materialized(self, mask);
    return at::redispatch::masked_select(self, mask);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MASKED_SELECT, self, mask);
}

at::Tensor wrap_masked_select_backward(const at::Tensor & grad, const at::Tensor & input, const at::Tensor & mask) {
  if (trace.is_flushing()) {
    ensure_materialized(grad, input, mask);
    return at::redispatch::masked_select_backward(grad, input, mask);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H_MASKED_SELECT_BACKWARD, grad, input, mask);
}

at::Tensor & wrap_nonzero_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::nonzero(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NONZERO_OUT, out, self);
}

at::Tensor wrap_nonzero(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::nonzero(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NONZERO, self);
}

std::vector<at::Tensor> wrap_nonzero_numpy(const at::Tensor & self) {
  ensure_materialized(self);
  return at::redispatch::nonzero_numpy(self);
}

at::Tensor & wrap_gather_out(const at::Tensor & self, int64_t dim, const at::Tensor & index, bool sparse_grad, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, index);
    return at::redispatch::gather(out, self, dim, index, sparse_grad);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_GATHER_OUT, out, self, dim, index, sparse_grad);
}

at::Tensor wrap_gather(const at::Tensor & self, int64_t dim, const at::Tensor & index, bool sparse_grad) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index);
    return at::redispatch::gather(self, dim, index, sparse_grad);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_GATHER, self, dim, index, sparse_grad);
}

at::Tensor wrap_gather_backward(const at::Tensor & grad, const at::Tensor & self, int64_t dim, const at::Tensor & index, bool sparse_grad) {
  if (trace.is_flushing()) {
    ensure_materialized(grad, self, index);
    return at::redispatch::gather_backward(grad, self, dim, index, sparse_grad);
  }
  return MK_TORCHY(grad.dtype(), grad.device(), H_GATHER_BACKWARD, grad, self, dim, index, sparse_grad);
}

at::Tensor & wrap_gather_dimname_out(const at::Tensor & self, at::Dimname dim, const at::Tensor & index, bool sparse_grad, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, index);
    return at::redispatch::gather(out, self, dim, index, sparse_grad);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_GATHER_DIMNAME_OUT, out, self, dim, index, sparse_grad);
}

at::Tensor wrap_gather_dimname(const at::Tensor & self, at::Dimname dim, const at::Tensor & index, bool sparse_grad) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index);
    return at::redispatch::gather(self, dim, index, sparse_grad);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_GATHER_DIMNAME, self, dim, index, sparse_grad);
}

at::Tensor wrap__gather_sparse_backward(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & grad) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index, grad);
    return at::redispatch::_gather_sparse_backward(self, dim, index, grad);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__GATHER_SPARSE_BACKWARD, self, dim, index, grad);
}

at::Tensor & wrap_addcmul_out(const at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, tensor1, tensor2);
    return at::redispatch::addcmul(out, self, tensor1, tensor2, value);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ADDCMUL_OUT, out, self, tensor1, tensor2, value);
}

at::Tensor wrap_addcmul(const at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value) {
  if (trace.is_flushing()) {
    ensure_materialized(self, tensor1, tensor2);
    return at::redispatch::addcmul(self, tensor1, tensor2, value);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ADDCMUL, self, tensor1, tensor2, value);
}

at::Tensor & wrap_addcmul_(at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value) {
  if (trace.is_flushing()) {
    ensure_materialized(self, tensor1, tensor2);
    return at::redispatch::addcmul_(self, tensor1, tensor2, value);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ADDCMUL_, self, tensor1, tensor2, value);
}

at::Tensor & wrap_addcdiv_out(const at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, tensor1, tensor2);
    return at::redispatch::addcdiv(out, self, tensor1, tensor2, value);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ADDCDIV_OUT, out, self, tensor1, tensor2, value);
}

at::Tensor wrap_addcdiv(const at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value) {
  if (trace.is_flushing()) {
    ensure_materialized(self, tensor1, tensor2);
    return at::redispatch::addcdiv(self, tensor1, tensor2, value);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ADDCDIV, self, tensor1, tensor2, value);
}

at::Tensor wrap_cross_entropy_loss(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index) {
  if (trace.is_flushing()) {
    ensure_materialized(self, target);
    return at::redispatch::cross_entropy_loss(self, target, weight, reduction, ignore_index);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CROSS_ENTROPY_LOSS, self, target, weight, reduction, ignore_index);
}

std::tuple<at::Tensor &,at::Tensor &> wrap_lstsq_X(const at::Tensor & self, const at::Tensor & A, at::Tensor & X, at::Tensor & qr) {
  ensure_materialized(X, qr, self, A);
  return at::redispatch::lstsq(X, qr, self, A);
}

std::tuple<at::Tensor,at::Tensor> wrap_lstsq(const at::Tensor & self, const at::Tensor & A) {
  ensure_materialized(self, A);
  return at::redispatch::lstsq(self, A);
}

std::tuple<at::Tensor &,at::Tensor &> wrap_triangular_solve_X(const at::Tensor & self, const at::Tensor & A, bool upper, bool transpose, bool unitriangular, at::Tensor & X, at::Tensor & M) {
  ensure_materialized(X, M, self, A);
  return at::redispatch::triangular_solve(X, M, self, A, upper, transpose, unitriangular);
}

std::tuple<at::Tensor,at::Tensor> wrap_triangular_solve(const at::Tensor & self, const at::Tensor & A, bool upper, bool transpose, bool unitriangular) {
  ensure_materialized(self, A);
  return at::redispatch::triangular_solve(self, A, upper, transpose, unitriangular);
}

std::tuple<at::Tensor &,at::Tensor &> wrap_symeig_e(const at::Tensor & self, bool eigenvectors, bool upper, at::Tensor & e, at::Tensor & V) {
  ensure_materialized(e, V, self);
  return at::redispatch::symeig(e, V, self, eigenvectors, upper);
}

std::tuple<at::Tensor,at::Tensor> wrap_symeig(const at::Tensor & self, bool eigenvectors, bool upper) {
  ensure_materialized(self);
  return at::redispatch::symeig(self, eigenvectors, upper);
}

std::tuple<at::Tensor,at::Tensor> wrap__symeig_helper(const at::Tensor & self, bool eigenvectors, bool upper) {
  ensure_materialized(self);
  return at::redispatch::_symeig_helper(self, eigenvectors, upper);
}

std::tuple<at::Tensor &,at::Tensor &> wrap_eig_e(const at::Tensor & self, bool eigenvectors, at::Tensor & e, at::Tensor & v) {
  ensure_materialized(e, v, self);
  return at::redispatch::eig(e, v, self, eigenvectors);
}

std::tuple<at::Tensor,at::Tensor> wrap_eig(const at::Tensor & self, bool eigenvectors) {
  ensure_materialized(self);
  return at::redispatch::eig(self, eigenvectors);
}

std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> wrap_svd_U(const at::Tensor & self, bool some, bool compute_uv, at::Tensor & U, at::Tensor & S, at::Tensor & V) {
  ensure_materialized(U, S, V, self);
  return at::redispatch::svd(U, S, V, self, some, compute_uv);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> wrap_svd(const at::Tensor & self, bool some, bool compute_uv) {
  ensure_materialized(self);
  return at::redispatch::svd(self, some, compute_uv);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> wrap__svd_helper(const at::Tensor & self, bool some, bool compute_uv) {
  ensure_materialized(self);
  return at::redispatch::_svd_helper(self, some, compute_uv);
}

at::Tensor wrap_swapaxes(const at::Tensor & self, int64_t axis0, int64_t axis1) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::swapaxes(self, axis0, axis1);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SWAPAXES, self, axis0, axis1);
}

at::Tensor & wrap_swapaxes_(at::Tensor & self, int64_t axis0, int64_t axis1) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::swapaxes_(self, axis0, axis1);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SWAPAXES_, self, axis0, axis1);
}

at::Tensor wrap_swapdims(const at::Tensor & self, int64_t dim0, int64_t dim1) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::swapdims(self, dim0, dim1);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SWAPDIMS, self, dim0, dim1);
}

at::Tensor & wrap_swapdims_(at::Tensor & self, int64_t dim0, int64_t dim1) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::swapdims_(self, dim0, dim1);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SWAPDIMS_, self, dim0, dim1);
}

at::Tensor & wrap_cholesky_out(const at::Tensor & self, bool upper, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::cholesky(out, self, upper);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_CHOLESKY_OUT, out, self, upper);
}

at::Tensor wrap_cholesky(const at::Tensor & self, bool upper) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::cholesky(self, upper);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CHOLESKY, self, upper);
}

at::Tensor wrap__cholesky_helper(const at::Tensor & self, bool upper) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_cholesky_helper(self, upper);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__CHOLESKY_HELPER, self, upper);
}

at::Tensor & wrap_cholesky_solve_out(const at::Tensor & self, const at::Tensor & input2, bool upper, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, input2);
    return at::redispatch::cholesky_solve(out, self, input2, upper);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_CHOLESKY_SOLVE_OUT, out, self, input2, upper);
}

at::Tensor wrap_cholesky_solve(const at::Tensor & self, const at::Tensor & input2, bool upper) {
  if (trace.is_flushing()) {
    ensure_materialized(self, input2);
    return at::redispatch::cholesky_solve(self, input2, upper);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CHOLESKY_SOLVE, self, input2, upper);
}

at::Tensor wrap__cholesky_solve_helper(const at::Tensor & self, const at::Tensor & A, bool upper) {
  if (trace.is_flushing()) {
    ensure_materialized(self, A);
    return at::redispatch::_cholesky_solve_helper(self, A, upper);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__CHOLESKY_SOLVE_HELPER, self, A, upper);
}

std::tuple<at::Tensor,at::Tensor> wrap_solve(const at::Tensor & self, const at::Tensor & A) {
  ensure_materialized(self, A);
  return at::redispatch::solve(self, A);
}

std::tuple<at::Tensor &,at::Tensor &> wrap_solve_solution(const at::Tensor & self, const at::Tensor & A, at::Tensor & solution, at::Tensor & lu) {
  ensure_materialized(solution, lu, self, A);
  return at::redispatch::solve(solution, lu, self, A);
}

std::tuple<at::Tensor,at::Tensor> wrap__solve_helper(const at::Tensor & self, const at::Tensor & A) {
  ensure_materialized(self, A);
  return at::redispatch::_solve_helper(self, A);
}

at::Tensor wrap_cholesky_inverse(const at::Tensor & self, bool upper) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::cholesky_inverse(self, upper);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CHOLESKY_INVERSE, self, upper);
}

at::Tensor & wrap_cholesky_inverse_out(const at::Tensor & self, bool upper, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::cholesky_inverse(out, self, upper);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_CHOLESKY_INVERSE_OUT, out, self, upper);
}

std::tuple<at::Tensor &,at::Tensor &> wrap_qr_Q(const at::Tensor & self, bool some, at::Tensor & Q, at::Tensor & R) {
  ensure_materialized(Q, R, self);
  return at::redispatch::qr(Q, R, self, some);
}

std::tuple<at::Tensor,at::Tensor> wrap_qr(const at::Tensor & self, bool some) {
  ensure_materialized(self);
  return at::redispatch::qr(self, some);
}

std::tuple<at::Tensor &,at::Tensor &> wrap_geqrf_a(const at::Tensor & self, at::Tensor & a, at::Tensor & tau) {
  ensure_materialized(a, tau, self);
  return at::redispatch::geqrf(a, tau, self);
}

std::tuple<at::Tensor,at::Tensor> wrap_geqrf(const at::Tensor & self) {
  ensure_materialized(self);
  return at::redispatch::geqrf(self);
}

at::Tensor wrap_orgqr(const at::Tensor & self, const at::Tensor & input2) {
  if (trace.is_flushing()) {
    ensure_materialized(self, input2);
    return at::redispatch::orgqr(self, input2);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ORGQR, self, input2);
}

at::Tensor & wrap_orgqr_out(const at::Tensor & self, const at::Tensor & input2, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, input2);
    return at::redispatch::orgqr(out, self, input2);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ORGQR_OUT, out, self, input2);
}

at::Tensor & wrap_ormqr_out(const at::Tensor & self, const at::Tensor & input2, const at::Tensor & input3, bool left, bool transpose, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, input2, input3);
    return at::redispatch::ormqr(out, self, input2, input3, left, transpose);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ORMQR_OUT, out, self, input2, input3, left, transpose);
}

at::Tensor wrap_ormqr(const at::Tensor & self, const at::Tensor & input2, const at::Tensor & input3, bool left, bool transpose) {
  if (trace.is_flushing()) {
    ensure_materialized(self, input2, input3);
    return at::redispatch::ormqr(self, input2, input3, left, transpose);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ORMQR, self, input2, input3, left, transpose);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> wrap__lu_with_info(const at::Tensor & self, bool pivot, bool check_errors) {
  ensure_materialized(self);
  return at::redispatch::_lu_with_info(self, pivot, check_errors);
}

at::Tensor & wrap_lu_solve_out(const at::Tensor & self, const at::Tensor & LU_data, const at::Tensor & LU_pivots, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, LU_data, LU_pivots);
    return at::redispatch::lu_solve(out, self, LU_data, LU_pivots);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LU_SOLVE_OUT, out, self, LU_data, LU_pivots);
}

at::Tensor wrap_lu_solve(const at::Tensor & self, const at::Tensor & LU_data, const at::Tensor & LU_pivots) {
  if (trace.is_flushing()) {
    ensure_materialized(self, LU_data, LU_pivots);
    return at::redispatch::lu_solve(self, LU_data, LU_pivots);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LU_SOLVE, self, LU_data, LU_pivots);
}

at::Tensor wrap__lu_solve_helper(const at::Tensor & self, const at::Tensor & LU_data, const at::Tensor & LU_pivots) {
  if (trace.is_flushing()) {
    ensure_materialized(self, LU_data, LU_pivots);
    return at::redispatch::_lu_solve_helper(self, LU_data, LU_pivots);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__LU_SOLVE_HELPER, self, LU_data, LU_pivots);
}

at::Tensor & wrap_multinomial_out(const at::Tensor & self, int64_t num_samples, bool replacement, c10::optional<at::Generator> generator, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::multinomial(out, self, num_samples, replacement, generator);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_MULTINOMIAL_OUT, out, self, num_samples, replacement, generator);
}

at::Tensor wrap_multinomial(const at::Tensor & self, int64_t num_samples, bool replacement, c10::optional<at::Generator> generator) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::multinomial(self, num_samples, replacement, generator);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MULTINOMIAL, self, num_samples, replacement, generator);
}

at::Tensor & wrap_lgamma_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::lgamma(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LGAMMA_OUT, out, self);
}

at::Tensor & wrap_digamma_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::digamma(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_DIGAMMA_OUT, out, self);
}

at::Tensor & wrap_polygamma_out(int64_t n, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::polygamma(out, n, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_POLYGAMMA_OUT, out, n, self);
}

at::Tensor wrap_polygamma(int64_t n, const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::polygamma(n, self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_POLYGAMMA, n, self);
}

at::Tensor & wrap_polygamma_(at::Tensor & self, int64_t n) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::polygamma_(self, n);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_POLYGAMMA_, self, n);
}

at::Tensor & wrap_erfinv_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::erfinv(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ERFINV_OUT, out, self);
}

at::Tensor wrap_i0(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::i0(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_I0, self);
}

at::Tensor & wrap_i0_(at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::i0_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_I0_, self);
}

at::Tensor & wrap_i0_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::i0(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_I0_OUT, out, self);
}

at::Tensor wrap_sign(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::sign(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SIGN, self);
}

at::Tensor & wrap_sign_(at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::sign_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SIGN_, self);
}

at::Tensor & wrap_sign_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::sign(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SIGN_OUT, out, self);
}

at::Tensor wrap_signbit(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::signbit(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SIGNBIT, self);
}

at::Tensor & wrap_signbit_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::signbit(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SIGNBIT_OUT, out, self);
}

at::Tensor wrap_dist(const at::Tensor & self, const at::Tensor & other, const at::Scalar & p) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::dist(self, other, p);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DIST, self, other, p);
}

at::Tensor & wrap_atan2_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::atan2(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ATAN2_OUT, out, self, other);
}

at::Tensor & wrap_lerp_Scalar_out(const at::Tensor & self, const at::Tensor & end, const at::Scalar & weight, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, end);
    return at::redispatch::lerp(out, self, end, weight);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LERP_SCALAR_OUT, out, self, end, weight);
}

at::Tensor & wrap_lerp_Tensor_out(const at::Tensor & self, const at::Tensor & end, const at::Tensor & weight, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, end, weight);
    return at::redispatch::lerp(out, self, end, weight);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LERP_TENSOR_OUT, out, self, end, weight);
}

at::Tensor wrap_lerp_Scalar(const at::Tensor & self, const at::Tensor & end, const at::Scalar & weight) {
  if (trace.is_flushing()) {
    ensure_materialized(self, end);
    return at::redispatch::lerp(self, end, weight);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LERP_SCALAR, self, end, weight);
}

at::Tensor wrap_lerp_Tensor(const at::Tensor & self, const at::Tensor & end, const at::Tensor & weight) {
  if (trace.is_flushing()) {
    ensure_materialized(self, end, weight);
    return at::redispatch::lerp(self, end, weight);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LERP_TENSOR, self, end, weight);
}

at::Tensor & wrap_histc_out(const at::Tensor & self, int64_t bins, const at::Scalar & min, const at::Scalar & max, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::histc(out, self, bins, min, max);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_HISTC_OUT, out, self, bins, min, max);
}

at::Tensor wrap_histc(const at::Tensor & self, int64_t bins, const at::Scalar & min, const at::Scalar & max) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::histc(self, bins, min, max);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_HISTC, self, bins, min, max);
}

at::Tensor & wrap_fmod_Scalar_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::fmod(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FMOD_SCALAR_OUT, out, self, other);
}

at::Tensor wrap_fmod_Scalar(const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::fmod(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FMOD_SCALAR, self, other);
}

at::Tensor & wrap_fmod_Tensor_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::fmod(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FMOD_TENSOR_OUT, out, self, other);
}

at::Tensor wrap_fmod_Tensor(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::fmod(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FMOD_TENSOR, self, other);
}

at::Tensor & wrap_hypot_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::hypot(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_HYPOT_OUT, out, self, other);
}

at::Tensor wrap_hypot(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::hypot(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_HYPOT, self, other);
}

at::Tensor & wrap_hypot_(at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::hypot_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_HYPOT_, self, other);
}

at::Tensor & wrap_igamma_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::igamma(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_IGAMMA_OUT, out, self, other);
}

at::Tensor wrap_igamma(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::igamma(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_IGAMMA, self, other);
}

at::Tensor & wrap_igamma_(at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::igamma_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_IGAMMA_, self, other);
}

at::Tensor & wrap_igammac_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::igammac(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_IGAMMAC_OUT, out, self, other);
}

at::Tensor wrap_igammac(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::igammac(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_IGAMMAC, self, other);
}

at::Tensor & wrap_igammac_(at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::igammac_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_IGAMMAC_, self, other);
}

at::Tensor & wrap_nextafter_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::nextafter(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NEXTAFTER_OUT, out, self, other);
}

at::Tensor wrap_nextafter(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::nextafter(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NEXTAFTER, self, other);
}

at::Tensor & wrap_nextafter_(at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::nextafter_(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NEXTAFTER_, self, other);
}

at::Tensor & wrap_remainder_Scalar_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::remainder(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_REMAINDER_SCALAR_OUT, out, self, other);
}

at::Tensor wrap_remainder_Scalar(const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::remainder(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_REMAINDER_SCALAR, self, other);
}

at::Tensor & wrap_remainder_Tensor_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::remainder(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_REMAINDER_TENSOR_OUT, out, self, other);
}

at::Tensor wrap_remainder_Tensor(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::remainder(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_REMAINDER_TENSOR, self, other);
}

at::Tensor wrap_min(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::min(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MIN, self);
}

at::Tensor wrap_fmin(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::fmin(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FMIN, self, other);
}

at::Tensor & wrap_fmin_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::fmin(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FMIN_OUT, out, self, other);
}

at::Tensor wrap_max(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::max(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MAX, self);
}

at::Tensor wrap_fmax(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::fmax(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FMAX, self, other);
}

at::Tensor & wrap_fmax_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::fmax(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FMAX_OUT, out, self, other);
}

at::Tensor wrap_maximum(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::maximum(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MAXIMUM, self, other);
}

at::Tensor & wrap_maximum_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::maximum(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_MAXIMUM_OUT, out, self, other);
}

at::Tensor wrap_max_other(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::max(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MAX_OTHER, self, other);
}

at::Tensor & wrap_max_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::max(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_MAX_OUT, out, self, other);
}

at::Tensor wrap_minimum(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::minimum(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MINIMUM, self, other);
}

at::Tensor & wrap_minimum_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::minimum(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_MINIMUM_OUT, out, self, other);
}

at::Tensor & wrap_min_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::min(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_MIN_OUT, out, self, other);
}

at::Tensor wrap_min_other(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::min(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MIN_OTHER, self, other);
}

at::Tensor & wrap_quantile_scalar_out(const at::Tensor & self, double q, c10::optional<int64_t> dim, bool keepdim, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::quantile(out, self, q, dim, keepdim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_QUANTILE_SCALAR_OUT, out, self, q, dim, keepdim);
}

at::Tensor wrap_quantile_scalar(const at::Tensor & self, double q, c10::optional<int64_t> dim, bool keepdim) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::quantile(self, q, dim, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_QUANTILE_SCALAR, self, q, dim, keepdim);
}

at::Tensor & wrap_quantile_out(const at::Tensor & self, const at::Tensor & q, c10::optional<int64_t> dim, bool keepdim, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, q);
    return at::redispatch::quantile(out, self, q, dim, keepdim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_QUANTILE_OUT, out, self, q, dim, keepdim);
}

at::Tensor wrap_quantile(const at::Tensor & self, const at::Tensor & q, c10::optional<int64_t> dim, bool keepdim) {
  if (trace.is_flushing()) {
    ensure_materialized(self, q);
    return at::redispatch::quantile(self, q, dim, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_QUANTILE, self, q, dim, keepdim);
}

at::Tensor & wrap_nanquantile_scalar_out(const at::Tensor & self, double q, c10::optional<int64_t> dim, bool keepdim, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::nanquantile(out, self, q, dim, keepdim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NANQUANTILE_SCALAR_OUT, out, self, q, dim, keepdim);
}

at::Tensor wrap_nanquantile_scalar(const at::Tensor & self, double q, c10::optional<int64_t> dim, bool keepdim) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::nanquantile(self, q, dim, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NANQUANTILE_SCALAR, self, q, dim, keepdim);
}

at::Tensor & wrap_nanquantile_out(const at::Tensor & self, const at::Tensor & q, c10::optional<int64_t> dim, bool keepdim, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, q);
    return at::redispatch::nanquantile(out, self, q, dim, keepdim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NANQUANTILE_OUT, out, self, q, dim, keepdim);
}

at::Tensor wrap_nanquantile(const at::Tensor & self, const at::Tensor & q, c10::optional<int64_t> dim, bool keepdim) {
  if (trace.is_flushing()) {
    ensure_materialized(self, q);
    return at::redispatch::nanquantile(self, q, dim, keepdim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NANQUANTILE, self, q, dim, keepdim);
}

at::Tensor & wrap_quantile_new_scalar_out(const at::Tensor & self, double q, c10::optional<int64_t> dim, bool keepdim, std::string interpolation, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::quantile(out, self, q, dim, keepdim, interpolation);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_QUANTILE_NEW_SCALAR_OUT, out, self, q, dim, keepdim, interpolation);
}

at::Tensor wrap_quantile_new_scalar(const at::Tensor & self, double q, c10::optional<int64_t> dim, bool keepdim, std::string interpolation) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::quantile(self, q, dim, keepdim, interpolation);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_QUANTILE_NEW_SCALAR, self, q, dim, keepdim, interpolation);
}

at::Tensor & wrap_quantile_new_out(const at::Tensor & self, const at::Tensor & q, c10::optional<int64_t> dim, bool keepdim, std::string interpolation, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, q);
    return at::redispatch::quantile(out, self, q, dim, keepdim, interpolation);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_QUANTILE_NEW_OUT, out, self, q, dim, keepdim, interpolation);
}

at::Tensor wrap_quantile_new(const at::Tensor & self, const at::Tensor & q, c10::optional<int64_t> dim, bool keepdim, std::string interpolation) {
  if (trace.is_flushing()) {
    ensure_materialized(self, q);
    return at::redispatch::quantile(self, q, dim, keepdim, interpolation);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_QUANTILE_NEW, self, q, dim, keepdim, interpolation);
}

at::Tensor & wrap_nanquantile_new_scalar_out(const at::Tensor & self, double q, c10::optional<int64_t> dim, bool keepdim, std::string interpolation, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::nanquantile(out, self, q, dim, keepdim, interpolation);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NANQUANTILE_NEW_SCALAR_OUT, out, self, q, dim, keepdim, interpolation);
}

at::Tensor wrap_nanquantile_new_scalar(const at::Tensor & self, double q, c10::optional<int64_t> dim, bool keepdim, std::string interpolation) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::nanquantile(self, q, dim, keepdim, interpolation);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NANQUANTILE_NEW_SCALAR, self, q, dim, keepdim, interpolation);
}

at::Tensor & wrap_nanquantile_new_out(const at::Tensor & self, const at::Tensor & q, c10::optional<int64_t> dim, bool keepdim, std::string interpolation, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, q);
    return at::redispatch::nanquantile(out, self, q, dim, keepdim, interpolation);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NANQUANTILE_NEW_OUT, out, self, q, dim, keepdim, interpolation);
}

at::Tensor wrap_nanquantile_new(const at::Tensor & self, const at::Tensor & q, c10::optional<int64_t> dim, bool keepdim, std::string interpolation) {
  if (trace.is_flushing()) {
    ensure_materialized(self, q);
    return at::redispatch::nanquantile(self, q, dim, keepdim, interpolation);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NANQUANTILE_NEW, self, q, dim, keepdim, interpolation);
}

std::tuple<at::Tensor &,at::Tensor &> wrap_sort_values(const at::Tensor & self, int64_t dim, bool descending, at::Tensor & values, at::Tensor & indices) {
  ensure_materialized(values, indices, self);
  return at::redispatch::sort(values, indices, self, dim, descending);
}

std::tuple<at::Tensor &,at::Tensor &> wrap_sort_values_stable(const at::Tensor & self, c10::optional<bool> stable, int64_t dim, bool descending, at::Tensor & values, at::Tensor & indices) {
  ensure_materialized(values, indices, self);
  return at::redispatch::sort(values, indices, self, stable, dim, descending);
}

std::tuple<at::Tensor,at::Tensor> wrap_sort(const at::Tensor & self, int64_t dim, bool descending) {
  ensure_materialized(self);
  return at::redispatch::sort(self, dim, descending);
}

std::tuple<at::Tensor,at::Tensor> wrap_sort_stable(const at::Tensor & self, c10::optional<bool> stable, int64_t dim, bool descending) {
  ensure_materialized(self);
  return at::redispatch::sort(self, stable, dim, descending);
}

std::tuple<at::Tensor &,at::Tensor &> wrap_sort_dimname_values(const at::Tensor & self, at::Dimname dim, bool descending, at::Tensor & values, at::Tensor & indices) {
  ensure_materialized(values, indices, self);
  return at::redispatch::sort(values, indices, self, dim, descending);
}

std::tuple<at::Tensor &,at::Tensor &> wrap_sort_dimname_values_stable(const at::Tensor & self, c10::optional<bool> stable, at::Dimname dim, bool descending, at::Tensor & values, at::Tensor & indices) {
  ensure_materialized(values, indices, self);
  return at::redispatch::sort(values, indices, self, stable, dim, descending);
}

std::tuple<at::Tensor,at::Tensor> wrap_sort_dimname(const at::Tensor & self, at::Dimname dim, bool descending) {
  ensure_materialized(self);
  return at::redispatch::sort(self, dim, descending);
}

std::tuple<at::Tensor,at::Tensor> wrap_sort_dimname_stable(const at::Tensor & self, c10::optional<bool> stable, at::Dimname dim, bool descending) {
  ensure_materialized(self);
  return at::redispatch::sort(self, stable, dim, descending);
}

at::Tensor & wrap_msort_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::msort(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_MSORT_OUT, out, self);
}

at::Tensor wrap_msort(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::msort(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MSORT, self);
}

at::Tensor wrap_argsort(const at::Tensor & self, int64_t dim, bool descending) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::argsort(self, dim, descending);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ARGSORT, self, dim, descending);
}

at::Tensor wrap_argsort_dimname(const at::Tensor & self, at::Dimname dim, bool descending) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::argsort(self, dim, descending);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ARGSORT_DIMNAME, self, dim, descending);
}

std::tuple<at::Tensor &,at::Tensor &> wrap_topk_values(const at::Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted, at::Tensor & values, at::Tensor & indices) {
  ensure_materialized(values, indices, self);
  return at::redispatch::topk(values, indices, self, k, dim, largest, sorted);
}

std::tuple<at::Tensor,at::Tensor> wrap_topk(const at::Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) {
  ensure_materialized(self);
  return at::redispatch::topk(self, k, dim, largest, sorted);
}

at::Tensor wrap_all(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::all(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ALL, self);
}

at::Tensor wrap_any(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::any(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ANY, self);
}

at::Tensor & wrap_renorm_out(const at::Tensor & self, const at::Scalar & p, int64_t dim, const at::Scalar & maxnorm, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::renorm(out, self, p, dim, maxnorm);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_RENORM_OUT, out, self, p, dim, maxnorm);
}

at::Tensor wrap_renorm(const at::Tensor & self, const at::Scalar & p, int64_t dim, const at::Scalar & maxnorm) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::renorm(self, p, dim, maxnorm);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RENORM, self, p, dim, maxnorm);
}

at::Tensor wrap_unfold(const at::Tensor & self, int64_t dimension, int64_t size, int64_t step) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::unfold(self, dimension, size, step);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_UNFOLD, self, dimension, size, step);
}

at::Tensor wrap_unfold_backward(const at::Tensor & grad_in, at::IntArrayRef input_sizes, int64_t dim, int64_t size, int64_t step) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_in);
    return at::redispatch::unfold_backward(grad_in, input_sizes, dim, size, step);
  }
  return MK_TORCHY(grad_in.dtype(), grad_in.device(), H_UNFOLD_BACKWARD, grad_in, input_sizes, dim, size, step);
}

bool wrap_equal(const at::Tensor & self, const at::Tensor & other) {
  ensure_materialized(self, other);
  return at::redispatch::equal(self, other);
}

at::Tensor & wrap_pow_Tensor_Tensor_out(const at::Tensor & self, const at::Tensor & exponent, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, exponent);
    return at::redispatch::pow(out, self, exponent);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_POW_TENSOR_TENSOR_OUT, out, self, exponent);
}

at::Tensor & wrap_pow_Scalar_out(const at::Scalar & self, const at::Tensor & exponent, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, exponent);
    return at::redispatch::pow(out, self, exponent);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_POW_SCALAR_OUT, out, self, exponent);
}

at::Tensor & wrap_pow_Tensor_Scalar_out(const at::Tensor & self, const at::Scalar & exponent, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::pow(out, self, exponent);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_POW_TENSOR_SCALAR_OUT, out, self, exponent);
}

at::Tensor wrap_pow_Tensor_Scalar(const at::Tensor & self, const at::Scalar & exponent) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::pow(self, exponent);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_POW_TENSOR_SCALAR, self, exponent);
}

at::Tensor & wrap_float_power_Tensor_Tensor_out(const at::Tensor & self, const at::Tensor & exponent, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, exponent);
    return at::redispatch::float_power(out, self, exponent);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FLOAT_POWER_TENSOR_TENSOR_OUT, out, self, exponent);
}

at::Tensor wrap_float_power_Tensor_Tensor(const at::Tensor & self, const at::Tensor & exponent) {
  if (trace.is_flushing()) {
    ensure_materialized(self, exponent);
    return at::redispatch::float_power(self, exponent);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FLOAT_POWER_TENSOR_TENSOR, self, exponent);
}

at::Tensor & wrap_float_power_Scalar_out(const at::Scalar & self, const at::Tensor & exponent, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, exponent);
    return at::redispatch::float_power(out, self, exponent);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FLOAT_POWER_SCALAR_OUT, out, self, exponent);
}

at::Tensor wrap_float_power_Scalar(const at::Scalar & self, const at::Tensor & exponent) {
  if (trace.is_flushing()) {
    ensure_materialized(exponent);
    return at::redispatch::float_power(self, exponent);
  }
  return MK_TORCHY(exponent.dtype(), exponent.device(), H_FLOAT_POWER_SCALAR, self, exponent);
}

at::Tensor & wrap_float_power_Tensor_Scalar_out(const at::Tensor & self, const at::Scalar & exponent, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::float_power(out, self, exponent);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FLOAT_POWER_TENSOR_SCALAR_OUT, out, self, exponent);
}

at::Tensor wrap_float_power_Tensor_Scalar(const at::Tensor & self, const at::Scalar & exponent) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::float_power(self, exponent);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FLOAT_POWER_TENSOR_SCALAR, self, exponent);
}

at::Tensor & wrap_float_power__Scalar(at::Tensor & self, const at::Scalar & exponent) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::float_power_(self, exponent);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FLOAT_POWER__SCALAR, self, exponent);
}

at::Tensor & wrap_float_power__Tensor(at::Tensor & self, const at::Tensor & exponent) {
  if (trace.is_flushing()) {
    ensure_materialized(self, exponent);
    return at::redispatch::float_power_(self, exponent);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FLOAT_POWER__TENSOR, self, exponent);
}

at::Tensor & wrap_normal_(at::Tensor & self, double mean, double std, c10::optional<at::Generator> generator) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::normal_(self, mean, std, generator);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NORMAL_, self, mean, std, generator);
}

at::Tensor & wrap_normal_Tensor_float_out(const at::Tensor & mean, double std, c10::optional<at::Generator> generator, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, mean);
    return at::redispatch::normal(out, mean, std, generator);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NORMAL_TENSOR_FLOAT_OUT, out, mean, std, generator);
}

at::Tensor wrap_normal_Tensor_float(const at::Tensor & mean, double std, c10::optional<at::Generator> generator) {
  if (trace.is_flushing()) {
    ensure_materialized(mean);
    return at::redispatch::normal(mean, std, generator);
  }
  return MK_TORCHY(mean.dtype(), mean.device(), H_NORMAL_TENSOR_FLOAT, mean, std, generator);
}

at::Tensor & wrap_normal_float_Tensor_out(double mean, const at::Tensor & std, c10::optional<at::Generator> generator, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, std);
    return at::redispatch::normal(out, mean, std, generator);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NORMAL_FLOAT_TENSOR_OUT, out, mean, std, generator);
}

at::Tensor wrap_normal_float_Tensor(double mean, const at::Tensor & std, c10::optional<at::Generator> generator) {
  if (trace.is_flushing()) {
    ensure_materialized(std);
    return at::redispatch::normal(mean, std, generator);
  }
  return MK_TORCHY(std.dtype(), std.device(), H_NORMAL_FLOAT_TENSOR, mean, std, generator);
}

at::Tensor & wrap_normal_Tensor_Tensor_out(const at::Tensor & mean, const at::Tensor & std, c10::optional<at::Generator> generator, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, mean, std);
    return at::redispatch::normal(out, mean, std, generator);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NORMAL_TENSOR_TENSOR_OUT, out, mean, std, generator);
}

at::Tensor wrap_normal_Tensor_Tensor(const at::Tensor & mean, const at::Tensor & std, c10::optional<at::Generator> generator) {
  if (trace.is_flushing()) {
    ensure_materialized(mean, std);
    return at::redispatch::normal(mean, std, generator);
  }
  return MK_TORCHY(mean.dtype(), mean.device(), H_NORMAL_TENSOR_TENSOR, mean, std, generator);
}

at::Tensor wrap_normal_float_float(double mean, double std, at::IntArrayRef size, c10::optional<at::Generator> generator, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::normal(mean, std, size, generator, dtype, layout, device, pin_memory));
}

at::Tensor & wrap_normal_float_float_out(double mean, double std, at::IntArrayRef size, c10::optional<at::Generator> generator, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::normal(out, mean, std, size, generator);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NORMAL_FLOAT_FLOAT_OUT, out, mean, std, size, generator);
}

at::Tensor wrap_alias(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::alias(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ALIAS, self);
}

at::Tensor & wrap__index_copy_(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source) {
  if (trace.is_flushing()) {
    ensure_materialized(self, index, source);
    return at::redispatch::_index_copy_(self, dim, index, source);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__INDEX_COPY_, self, dim, index, source);
}

at::Tensor wrap__cumsum(const at::Tensor & self, int64_t dim) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_cumsum(self, dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__CUMSUM, self, dim);
}

at::Tensor & wrap__cumsum_out(const at::Tensor & self, int64_t dim, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::_cumsum(out, self, dim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H__CUMSUM_OUT, out, self, dim);
}

at::Tensor wrap__cumprod(const at::Tensor & self, int64_t dim) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_cumprod(self, dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__CUMPROD, self, dim);
}

at::Tensor & wrap__cumprod_out(const at::Tensor & self, int64_t dim, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::_cumprod(out, self, dim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H__CUMPROD_OUT, out, self, dim);
}

at::Tensor wrap__var(const at::Tensor & self, bool unbiased) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_var(self, unbiased);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__VAR, self, unbiased);
}

at::Tensor wrap__std(const at::Tensor & self, bool unbiased) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_std(self, unbiased);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__STD, self, unbiased);
}

void wrap__amp_foreach_non_finite_check_and_unscale_(at::TensorList self, at::Tensor & found_inf, const at::Tensor & inv_scale) {
  ensure_materialized(found_inf, inv_scale);
  return at::redispatch::_amp_foreach_non_finite_check_and_unscale_(self, found_inf, inv_scale);
}

at::Tensor wrap__amp_update_scale(at::Tensor & growth_tracker, const at::Tensor & current_scale, const at::Tensor & found_inf, double scale_growth_factor, double scale_backoff_factor, int64_t growth_interval) {
  if (trace.is_flushing()) {
    ensure_materialized(growth_tracker, current_scale, found_inf);
    return at::redispatch::_amp_update_scale(growth_tracker, current_scale, found_inf, scale_growth_factor, scale_backoff_factor, growth_interval);
  }
  return MK_TORCHY(growth_tracker.dtype(), growth_tracker.device(), H__AMP_UPDATE_SCALE, growth_tracker, current_scale, found_inf, scale_growth_factor, scale_backoff_factor, growth_interval);
}

at::Tensor wrap__cat(at::TensorList tensors, int64_t dim) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::_cat(tensors, dim));
}

at::Tensor & wrap__cat_out(at::TensorList tensors, int64_t dim, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::_cat(out, tensors, dim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H__CAT_OUT, out, tensors, dim);
}

std::vector<at::Tensor> wrap__foreach_add_Scalar(at::TensorList tensors, const at::Scalar & scalar) {
  ensure_materialized();
  return at::redispatch::_foreach_add(tensors, scalar);
}

void wrap__foreach_add__Scalar(at::TensorList self, const at::Scalar & scalar) {
  ensure_materialized();
  return at::redispatch::_foreach_add_(self, scalar);
}

std::vector<at::Tensor> wrap__foreach_sub_Scalar(at::TensorList tensors, const at::Scalar & scalar) {
  ensure_materialized();
  return at::redispatch::_foreach_sub(tensors, scalar);
}

void wrap__foreach_sub__Scalar(at::TensorList self, const at::Scalar & scalar) {
  ensure_materialized();
  return at::redispatch::_foreach_sub_(self, scalar);
}

std::vector<at::Tensor> wrap__foreach_mul_Scalar(at::TensorList tensors, const at::Scalar & scalar) {
  ensure_materialized();
  return at::redispatch::_foreach_mul(tensors, scalar);
}

void wrap__foreach_mul__Scalar(at::TensorList self, const at::Scalar & scalar) {
  ensure_materialized();
  return at::redispatch::_foreach_mul_(self, scalar);
}

std::vector<at::Tensor> wrap__foreach_div_Scalar(at::TensorList tensors, const at::Scalar & scalar) {
  ensure_materialized();
  return at::redispatch::_foreach_div(tensors, scalar);
}

void wrap__foreach_div__Scalar(at::TensorList self, const at::Scalar & scalar) {
  ensure_materialized();
  return at::redispatch::_foreach_div_(self, scalar);
}

std::vector<at::Tensor> wrap__foreach_add_List(at::TensorList tensors1, at::TensorList tensors2, const at::Scalar & alpha) {
  ensure_materialized();
  return at::redispatch::_foreach_add(tensors1, tensors2, alpha);
}

void wrap__foreach_add__List(at::TensorList self, at::TensorList other, const at::Scalar & alpha) {
  ensure_materialized();
  return at::redispatch::_foreach_add_(self, other, alpha);
}

std::vector<at::Tensor> wrap__foreach_sub_List(at::TensorList tensors1, at::TensorList tensors2, const at::Scalar & alpha) {
  ensure_materialized();
  return at::redispatch::_foreach_sub(tensors1, tensors2, alpha);
}

void wrap__foreach_sub__List(at::TensorList self, at::TensorList other, const at::Scalar & alpha) {
  ensure_materialized();
  return at::redispatch::_foreach_sub_(self, other, alpha);
}

std::vector<at::Tensor> wrap__foreach_mul_List(at::TensorList tensors1, at::TensorList tensors2) {
  ensure_materialized();
  return at::redispatch::_foreach_mul(tensors1, tensors2);
}

void wrap__foreach_mul__List(at::TensorList self, at::TensorList other) {
  ensure_materialized();
  return at::redispatch::_foreach_mul_(self, other);
}

std::vector<at::Tensor> wrap__foreach_div_List(at::TensorList tensors1, at::TensorList tensors2) {
  ensure_materialized();
  return at::redispatch::_foreach_div(tensors1, tensors2);
}

void wrap__foreach_div__List(at::TensorList self, at::TensorList other) {
  ensure_materialized();
  return at::redispatch::_foreach_div_(self, other);
}

std::vector<at::Tensor> wrap__foreach_add_ScalarList(at::TensorList tensors, at::ArrayRef<at::Scalar> scalars) {
  ensure_materialized();
  return at::redispatch::_foreach_add(tensors, scalars);
}

void wrap__foreach_add__ScalarList(at::TensorList self, at::ArrayRef<at::Scalar> scalars) {
  ensure_materialized();
  return at::redispatch::_foreach_add_(self, scalars);
}

std::vector<at::Tensor> wrap__foreach_sub_ScalarList(at::TensorList tensors, at::ArrayRef<at::Scalar> scalars) {
  ensure_materialized();
  return at::redispatch::_foreach_sub(tensors, scalars);
}

void wrap__foreach_sub__ScalarList(at::TensorList self, at::ArrayRef<at::Scalar> scalars) {
  ensure_materialized();
  return at::redispatch::_foreach_sub_(self, scalars);
}

std::vector<at::Tensor> wrap__foreach_div_ScalarList(at::TensorList tensors, at::ArrayRef<at::Scalar> scalars) {
  ensure_materialized();
  return at::redispatch::_foreach_div(tensors, scalars);
}

void wrap__foreach_div__ScalarList(at::TensorList self, at::ArrayRef<at::Scalar> scalars) {
  ensure_materialized();
  return at::redispatch::_foreach_div_(self, scalars);
}

std::vector<at::Tensor> wrap__foreach_mul_ScalarList(at::TensorList tensors, at::ArrayRef<at::Scalar> scalars) {
  ensure_materialized();
  return at::redispatch::_foreach_mul(tensors, scalars);
}

void wrap__foreach_mul__ScalarList(at::TensorList self, at::ArrayRef<at::Scalar> scalars) {
  ensure_materialized();
  return at::redispatch::_foreach_mul_(self, scalars);
}

std::vector<at::Tensor> wrap__foreach_exp(at::TensorList tensors) {
  ensure_materialized();
  return at::redispatch::_foreach_exp(tensors);
}

void wrap__foreach_zero_(at::TensorList self) {
  ensure_materialized();
  return at::redispatch::_foreach_zero_(self);
}

void wrap__foreach_exp_(at::TensorList self) {
  ensure_materialized();
  return at::redispatch::_foreach_exp_(self);
}

std::vector<at::Tensor> wrap__foreach_sqrt(at::TensorList tensors) {
  ensure_materialized();
  return at::redispatch::_foreach_sqrt(tensors);
}

void wrap__foreach_sqrt_(at::TensorList self) {
  ensure_materialized();
  return at::redispatch::_foreach_sqrt_(self);
}

std::vector<at::Tensor> wrap__foreach_abs(at::TensorList tensors) {
  ensure_materialized();
  return at::redispatch::_foreach_abs(tensors);
}

void wrap__foreach_abs_(at::TensorList self) {
  ensure_materialized();
  return at::redispatch::_foreach_abs_(self);
}

std::vector<at::Tensor> wrap__foreach_acos(at::TensorList tensors) {
  ensure_materialized();
  return at::redispatch::_foreach_acos(tensors);
}

void wrap__foreach_acos_(at::TensorList self) {
  ensure_materialized();
  return at::redispatch::_foreach_acos_(self);
}

std::vector<at::Tensor> wrap__foreach_asin(at::TensorList tensors) {
  ensure_materialized();
  return at::redispatch::_foreach_asin(tensors);
}

void wrap__foreach_asin_(at::TensorList self) {
  ensure_materialized();
  return at::redispatch::_foreach_asin_(self);
}

std::vector<at::Tensor> wrap__foreach_atan(at::TensorList tensors) {
  ensure_materialized();
  return at::redispatch::_foreach_atan(tensors);
}

void wrap__foreach_atan_(at::TensorList self) {
  ensure_materialized();
  return at::redispatch::_foreach_atan_(self);
}

std::vector<at::Tensor> wrap__foreach_ceil(at::TensorList tensors) {
  ensure_materialized();
  return at::redispatch::_foreach_ceil(tensors);
}

void wrap__foreach_ceil_(at::TensorList self) {
  ensure_materialized();
  return at::redispatch::_foreach_ceil_(self);
}

std::vector<at::Tensor> wrap__foreach_cos(at::TensorList tensors) {
  ensure_materialized();
  return at::redispatch::_foreach_cos(tensors);
}

void wrap__foreach_cos_(at::TensorList self) {
  ensure_materialized();
  return at::redispatch::_foreach_cos_(self);
}

std::vector<at::Tensor> wrap__foreach_cosh(at::TensorList tensors) {
  ensure_materialized();
  return at::redispatch::_foreach_cosh(tensors);
}

void wrap__foreach_cosh_(at::TensorList self) {
  ensure_materialized();
  return at::redispatch::_foreach_cosh_(self);
}

std::vector<at::Tensor> wrap__foreach_erf(at::TensorList tensors) {
  ensure_materialized();
  return at::redispatch::_foreach_erf(tensors);
}

void wrap__foreach_erf_(at::TensorList self) {
  ensure_materialized();
  return at::redispatch::_foreach_erf_(self);
}

std::vector<at::Tensor> wrap__foreach_erfc(at::TensorList tensors) {
  ensure_materialized();
  return at::redispatch::_foreach_erfc(tensors);
}

void wrap__foreach_erfc_(at::TensorList self) {
  ensure_materialized();
  return at::redispatch::_foreach_erfc_(self);
}

std::vector<at::Tensor> wrap__foreach_expm1(at::TensorList tensors) {
  ensure_materialized();
  return at::redispatch::_foreach_expm1(tensors);
}

void wrap__foreach_expm1_(at::TensorList self) {
  ensure_materialized();
  return at::redispatch::_foreach_expm1_(self);
}

std::vector<at::Tensor> wrap__foreach_floor(at::TensorList tensors) {
  ensure_materialized();
  return at::redispatch::_foreach_floor(tensors);
}

void wrap__foreach_floor_(at::TensorList self) {
  ensure_materialized();
  return at::redispatch::_foreach_floor_(self);
}

std::vector<at::Tensor> wrap__foreach_log(at::TensorList tensors) {
  ensure_materialized();
  return at::redispatch::_foreach_log(tensors);
}

void wrap__foreach_log_(at::TensorList self) {
  ensure_materialized();
  return at::redispatch::_foreach_log_(self);
}

std::vector<at::Tensor> wrap__foreach_log10(at::TensorList tensors) {
  ensure_materialized();
  return at::redispatch::_foreach_log10(tensors);
}

void wrap__foreach_log10_(at::TensorList self) {
  ensure_materialized();
  return at::redispatch::_foreach_log10_(self);
}

std::vector<at::Tensor> wrap__foreach_log1p(at::TensorList tensors) {
  ensure_materialized();
  return at::redispatch::_foreach_log1p(tensors);
}

void wrap__foreach_log1p_(at::TensorList self) {
  ensure_materialized();
  return at::redispatch::_foreach_log1p_(self);
}

std::vector<at::Tensor> wrap__foreach_log2(at::TensorList tensors) {
  ensure_materialized();
  return at::redispatch::_foreach_log2(tensors);
}

void wrap__foreach_log2_(at::TensorList self) {
  ensure_materialized();
  return at::redispatch::_foreach_log2_(self);
}

std::vector<at::Tensor> wrap__foreach_neg(at::TensorList tensors) {
  ensure_materialized();
  return at::redispatch::_foreach_neg(tensors);
}

void wrap__foreach_neg_(at::TensorList self) {
  ensure_materialized();
  return at::redispatch::_foreach_neg_(self);
}

std::vector<at::Tensor> wrap__foreach_tan(at::TensorList tensors) {
  ensure_materialized();
  return at::redispatch::_foreach_tan(tensors);
}

void wrap__foreach_tan_(at::TensorList self) {
  ensure_materialized();
  return at::redispatch::_foreach_tan_(self);
}

std::vector<at::Tensor> wrap__foreach_tanh(at::TensorList tensors) {
  ensure_materialized();
  return at::redispatch::_foreach_tanh(tensors);
}

void wrap__foreach_tanh_(at::TensorList self) {
  ensure_materialized();
  return at::redispatch::_foreach_tanh_(self);
}

std::vector<at::Tensor> wrap__foreach_sin(at::TensorList tensors) {
  ensure_materialized();
  return at::redispatch::_foreach_sin(tensors);
}

void wrap__foreach_sin_(at::TensorList self) {
  ensure_materialized();
  return at::redispatch::_foreach_sin_(self);
}

std::vector<at::Tensor> wrap__foreach_sinh(at::TensorList tensors) {
  ensure_materialized();
  return at::redispatch::_foreach_sinh(tensors);
}

void wrap__foreach_sinh_(at::TensorList self) {
  ensure_materialized();
  return at::redispatch::_foreach_sinh_(self);
}

std::vector<at::Tensor> wrap__foreach_round(at::TensorList tensors) {
  ensure_materialized();
  return at::redispatch::_foreach_round(tensors);
}

void wrap__foreach_round_(at::TensorList self) {
  ensure_materialized();
  return at::redispatch::_foreach_round_(self);
}

std::vector<at::Tensor> wrap__foreach_lgamma(at::TensorList tensors) {
  ensure_materialized();
  return at::redispatch::_foreach_lgamma(tensors);
}

void wrap__foreach_lgamma_(at::TensorList self) {
  ensure_materialized();
  return at::redispatch::_foreach_lgamma_(self);
}

std::vector<at::Tensor> wrap__foreach_frac(at::TensorList tensors) {
  ensure_materialized();
  return at::redispatch::_foreach_frac(tensors);
}

void wrap__foreach_frac_(at::TensorList self) {
  ensure_materialized();
  return at::redispatch::_foreach_frac_(self);
}

std::vector<at::Tensor> wrap__foreach_reciprocal(at::TensorList tensors) {
  ensure_materialized();
  return at::redispatch::_foreach_reciprocal(tensors);
}

void wrap__foreach_reciprocal_(at::TensorList self) {
  ensure_materialized();
  return at::redispatch::_foreach_reciprocal_(self);
}

std::vector<at::Tensor> wrap__foreach_sigmoid(at::TensorList tensors) {
  ensure_materialized();
  return at::redispatch::_foreach_sigmoid(tensors);
}

void wrap__foreach_sigmoid_(at::TensorList self) {
  ensure_materialized();
  return at::redispatch::_foreach_sigmoid_(self);
}

std::vector<at::Tensor> wrap__foreach_trunc(at::TensorList tensors) {
  ensure_materialized();
  return at::redispatch::_foreach_trunc(tensors);
}

void wrap__foreach_trunc_(at::TensorList self) {
  ensure_materialized();
  return at::redispatch::_foreach_trunc_(self);
}

void wrap__foreach_addcdiv__Scalar(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Scalar & value) {
  ensure_materialized();
  return at::redispatch::_foreach_addcdiv_(self, tensor1, tensor2, value);
}

void wrap__foreach_addcmul__Scalar(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Scalar & value) {
  ensure_materialized();
  return at::redispatch::_foreach_addcmul_(self, tensor1, tensor2, value);
}

void wrap__foreach_addcdiv__ScalarList(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, at::ArrayRef<at::Scalar> scalars) {
  ensure_materialized();
  return at::redispatch::_foreach_addcdiv_(self, tensor1, tensor2, scalars);
}

void wrap__foreach_addcmul__ScalarList(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, at::ArrayRef<at::Scalar> scalars) {
  ensure_materialized();
  return at::redispatch::_foreach_addcmul_(self, tensor1, tensor2, scalars);
}

std::vector<at::Tensor> wrap__foreach_addcdiv_Scalar(at::TensorList input, at::TensorList tensor1, at::TensorList tensor2, const at::Scalar & value) {
  ensure_materialized();
  return at::redispatch::_foreach_addcdiv(input, tensor1, tensor2, value);
}

std::vector<at::Tensor> wrap__foreach_addcmul_Scalar(at::TensorList input, at::TensorList tensor1, at::TensorList tensor2, const at::Scalar & value) {
  ensure_materialized();
  return at::redispatch::_foreach_addcmul(input, tensor1, tensor2, value);
}

std::vector<at::Tensor> wrap__foreach_addcdiv_ScalarList(at::TensorList input, at::TensorList tensor1, at::TensorList tensor2, at::ArrayRef<at::Scalar> scalars) {
  ensure_materialized();
  return at::redispatch::_foreach_addcdiv(input, tensor1, tensor2, scalars);
}

std::vector<at::Tensor> wrap__foreach_addcmul_ScalarList(at::TensorList input, at::TensorList tensor1, at::TensorList tensor2, at::ArrayRef<at::Scalar> scalars) {
  ensure_materialized();
  return at::redispatch::_foreach_addcmul(input, tensor1, tensor2, scalars);
}

std::vector<at::Tensor> wrap__foreach_maximum_List(at::TensorList tensors1, at::TensorList tensors2) {
  ensure_materialized();
  return at::redispatch::_foreach_maximum(tensors1, tensors2);
}

std::vector<at::Tensor> wrap__foreach_minimum_List(at::TensorList tensors1, at::TensorList tensors2) {
  ensure_materialized();
  return at::redispatch::_foreach_minimum(tensors1, tensors2);
}

at::Tensor wrap_bucketize_Tensor(const at::Tensor & self, const at::Tensor & boundaries, bool out_int32, bool right) {
  if (trace.is_flushing()) {
    ensure_materialized(self, boundaries);
    return at::redispatch::bucketize(self, boundaries, out_int32, right);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_BUCKETIZE_TENSOR, self, boundaries, out_int32, right);
}

at::Tensor & wrap_bucketize_Tensor_out(const at::Tensor & self, const at::Tensor & boundaries, bool out_int32, bool right, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, boundaries);
    return at::redispatch::bucketize(out, self, boundaries, out_int32, right);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_BUCKETIZE_TENSOR_OUT, out, self, boundaries, out_int32, right);
}

at::Tensor wrap_bucketize_Scalar(const at::Scalar & self, const at::Tensor & boundaries, bool out_int32, bool right) {
  if (trace.is_flushing()) {
    ensure_materialized(boundaries);
    return at::redispatch::bucketize(self, boundaries, out_int32, right);
  }
  return MK_TORCHY(boundaries.dtype(), boundaries.device(), H_BUCKETIZE_SCALAR, self, boundaries, out_int32, right);
}

at::Tensor wrap_searchsorted_Tensor(const at::Tensor & sorted_sequence, const at::Tensor & self, bool out_int32, bool right) {
  if (trace.is_flushing()) {
    ensure_materialized(sorted_sequence, self);
    return at::redispatch::searchsorted(sorted_sequence, self, out_int32, right);
  }
  return MK_TORCHY(sorted_sequence.dtype(), sorted_sequence.device(), H_SEARCHSORTED_TENSOR, sorted_sequence, self, out_int32, right);
}

at::Tensor & wrap_searchsorted_Tensor_out(const at::Tensor & sorted_sequence, const at::Tensor & self, bool out_int32, bool right, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, sorted_sequence, self);
    return at::redispatch::searchsorted(out, sorted_sequence, self, out_int32, right);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SEARCHSORTED_TENSOR_OUT, out, sorted_sequence, self, out_int32, right);
}

at::Tensor wrap_searchsorted_Scalar(const at::Tensor & sorted_sequence, const at::Scalar & self, bool out_int32, bool right) {
  if (trace.is_flushing()) {
    ensure_materialized(sorted_sequence);
    return at::redispatch::searchsorted(sorted_sequence, self, out_int32, right);
  }
  return MK_TORCHY(sorted_sequence.dtype(), sorted_sequence.device(), H_SEARCHSORTED_SCALAR, sorted_sequence, self, out_int32, right);
}

at::Tensor & wrap_mse_loss_out(const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, target);
    return at::redispatch::mse_loss(out, self, target, reduction);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_MSE_LOSS_OUT, out, self, target, reduction);
}

at::Tensor wrap_mse_loss(const at::Tensor & self, const at::Tensor & target, int64_t reduction) {
  if (trace.is_flushing()) {
    ensure_materialized(self, target);
    return at::redispatch::mse_loss(self, target, reduction);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MSE_LOSS, self, target, reduction);
}

at::Tensor & wrap_mse_loss_backward_grad_input(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self, target);
    return at::redispatch::mse_loss_backward(grad_input, grad_output, self, target, reduction);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_MSE_LOSS_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, target, reduction);
}

at::Tensor wrap_mse_loss_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, target);
    return at::redispatch::mse_loss_backward(grad_output, self, target, reduction);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_MSE_LOSS_BACKWARD, grad_output, self, target, reduction);
}

at::Tensor & wrap_l1_loss_out(const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, target);
    return at::redispatch::l1_loss(out, self, target, reduction);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_L1_LOSS_OUT, out, self, target, reduction);
}

at::Tensor wrap_l1_loss(const at::Tensor & self, const at::Tensor & target, int64_t reduction) {
  if (trace.is_flushing()) {
    ensure_materialized(self, target);
    return at::redispatch::l1_loss(self, target, reduction);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_L1_LOSS, self, target, reduction);
}

at::Tensor & wrap_l1_loss_backward_grad_input(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self, target);
    return at::redispatch::l1_loss_backward(grad_input, grad_output, self, target, reduction);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_L1_LOSS_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, target, reduction);
}

at::Tensor wrap_l1_loss_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, target);
    return at::redispatch::l1_loss_backward(grad_output, self, target, reduction);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_L1_LOSS_BACKWARD, grad_output, self, target, reduction);
}

at::Tensor & wrap_multi_margin_loss_out(const at::Tensor & self, const at::Tensor & target, const at::Scalar & p, const at::Scalar & margin, const c10::optional<at::Tensor> & weight, int64_t reduction, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, target);
    return at::redispatch::multi_margin_loss(out, self, target, p, margin, weight, reduction);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_MULTI_MARGIN_LOSS_OUT, out, self, target, p, margin, weight, reduction);
}

at::Tensor wrap_multi_margin_loss(const at::Tensor & self, const at::Tensor & target, const at::Scalar & p, const at::Scalar & margin, const c10::optional<at::Tensor> & weight, int64_t reduction) {
  if (trace.is_flushing()) {
    ensure_materialized(self, target);
    return at::redispatch::multi_margin_loss(self, target, p, margin, weight, reduction);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MULTI_MARGIN_LOSS, self, target, p, margin, weight, reduction);
}

at::Tensor & wrap_multi_margin_loss_backward_grad_input(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const at::Scalar & p, const at::Scalar & margin, const c10::optional<at::Tensor> & weight, int64_t reduction, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self, target);
    return at::redispatch::multi_margin_loss_backward(grad_input, grad_output, self, target, p, margin, weight, reduction);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_MULTI_MARGIN_LOSS_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, target, p, margin, weight, reduction);
}

at::Tensor wrap_multi_margin_loss_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const at::Scalar & p, const at::Scalar & margin, const c10::optional<at::Tensor> & weight, int64_t reduction) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, target);
    return at::redispatch::multi_margin_loss_backward(grad_output, self, target, p, margin, weight, reduction);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_MULTI_MARGIN_LOSS_BACKWARD, grad_output, self, target, p, margin, weight, reduction);
}

at::Tensor & wrap_multilabel_margin_loss_out(const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, target);
    return at::redispatch::multilabel_margin_loss(out, self, target, reduction);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_MULTILABEL_MARGIN_LOSS_OUT, out, self, target, reduction);
}

at::Tensor wrap_multilabel_margin_loss(const at::Tensor & self, const at::Tensor & target, int64_t reduction) {
  if (trace.is_flushing()) {
    ensure_materialized(self, target);
    return at::redispatch::multilabel_margin_loss(self, target, reduction);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MULTILABEL_MARGIN_LOSS, self, target, reduction);
}

std::tuple<at::Tensor &,at::Tensor &> wrap_multilabel_margin_loss_forward_output(const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & output, at::Tensor & is_target) {
  ensure_materialized(output, is_target, self, target);
  return at::redispatch::multilabel_margin_loss_forward(output, is_target, self, target, reduction);
}

std::tuple<at::Tensor,at::Tensor> wrap_multilabel_margin_loss_forward(const at::Tensor & self, const at::Tensor & target, int64_t reduction) {
  ensure_materialized(self, target);
  return at::redispatch::multilabel_margin_loss_forward(self, target, reduction);
}

at::Tensor & wrap_multilabel_margin_loss_backward_grad_input(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, const at::Tensor & is_target, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self, target, is_target);
    return at::redispatch::multilabel_margin_loss_backward(grad_input, grad_output, self, target, reduction, is_target);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_MULTILABEL_MARGIN_LOSS_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, target, reduction, is_target);
}

at::Tensor wrap_multilabel_margin_loss_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, const at::Tensor & is_target) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, target, is_target);
    return at::redispatch::multilabel_margin_loss_backward(grad_output, self, target, reduction, is_target);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_MULTILABEL_MARGIN_LOSS_BACKWARD, grad_output, self, target, reduction, is_target);
}

at::Tensor & wrap_nll_loss_out(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, target);
    return at::redispatch::nll_loss(out, self, target, weight, reduction, ignore_index);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NLL_LOSS_OUT, out, self, target, weight, reduction, ignore_index);
}

at::Tensor wrap_nll_loss_nd(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index) {
  if (trace.is_flushing()) {
    ensure_materialized(self, target);
    return at::redispatch::nll_loss_nd(self, target, weight, reduction, ignore_index);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NLL_LOSS_ND, self, target, weight, reduction, ignore_index);
}

at::Tensor wrap_nll_loss(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index) {
  if (trace.is_flushing()) {
    ensure_materialized(self, target);
    return at::redispatch::nll_loss(self, target, weight, reduction, ignore_index);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NLL_LOSS, self, target, weight, reduction, ignore_index);
}

std::tuple<at::Tensor &,at::Tensor &> wrap_nll_loss_forward_output(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, at::Tensor & output, at::Tensor & total_weight) {
  ensure_materialized(output, total_weight, self, target);
  return at::redispatch::nll_loss_forward(output, total_weight, self, target, weight, reduction, ignore_index);
}

std::tuple<at::Tensor,at::Tensor> wrap_nll_loss_forward(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index) {
  ensure_materialized(self, target);
  return at::redispatch::nll_loss_forward(self, target, weight, reduction, ignore_index);
}

at::Tensor & wrap_nll_loss_backward_grad_input(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, const at::Tensor & total_weight, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self, target, total_weight);
    return at::redispatch::nll_loss_backward(grad_input, grad_output, self, target, weight, reduction, ignore_index, total_weight);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_NLL_LOSS_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, target, weight, reduction, ignore_index, total_weight);
}

at::Tensor wrap_nll_loss_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, const at::Tensor & total_weight) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, target, total_weight);
    return at::redispatch::nll_loss_backward(grad_output, self, target, weight, reduction, ignore_index, total_weight);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_NLL_LOSS_BACKWARD, grad_output, self, target, weight, reduction, ignore_index, total_weight);
}

at::Tensor & wrap_nll_loss2d_out(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, target);
    return at::redispatch::nll_loss2d(out, self, target, weight, reduction, ignore_index);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_NLL_LOSS2D_OUT, out, self, target, weight, reduction, ignore_index);
}

at::Tensor wrap_nll_loss2d(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index) {
  if (trace.is_flushing()) {
    ensure_materialized(self, target);
    return at::redispatch::nll_loss2d(self, target, weight, reduction, ignore_index);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_NLL_LOSS2D, self, target, weight, reduction, ignore_index);
}

std::tuple<at::Tensor &,at::Tensor &> wrap_nll_loss2d_forward_output(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, at::Tensor & output, at::Tensor & total_weight) {
  ensure_materialized(output, total_weight, self, target);
  return at::redispatch::nll_loss2d_forward(output, total_weight, self, target, weight, reduction, ignore_index);
}

std::tuple<at::Tensor,at::Tensor> wrap_nll_loss2d_forward(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index) {
  ensure_materialized(self, target);
  return at::redispatch::nll_loss2d_forward(self, target, weight, reduction, ignore_index);
}

at::Tensor & wrap_nll_loss2d_backward_grad_input(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, const at::Tensor & total_weight, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self, target, total_weight);
    return at::redispatch::nll_loss2d_backward(grad_input, grad_output, self, target, weight, reduction, ignore_index, total_weight);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_NLL_LOSS2D_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, target, weight, reduction, ignore_index, total_weight);
}

at::Tensor wrap_nll_loss2d_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, const at::Tensor & total_weight) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, target, total_weight);
    return at::redispatch::nll_loss2d_backward(grad_output, self, target, weight, reduction, ignore_index, total_weight);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_NLL_LOSS2D_BACKWARD, grad_output, self, target, weight, reduction, ignore_index, total_weight);
}

at::Tensor & wrap_smooth_l1_loss_out(const at::Tensor & self, const at::Tensor & target, int64_t reduction, double beta, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, target);
    return at::redispatch::smooth_l1_loss(out, self, target, reduction, beta);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SMOOTH_L1_LOSS_OUT, out, self, target, reduction, beta);
}

at::Tensor wrap_smooth_l1_loss(const at::Tensor & self, const at::Tensor & target, int64_t reduction, double beta) {
  if (trace.is_flushing()) {
    ensure_materialized(self, target);
    return at::redispatch::smooth_l1_loss(self, target, reduction, beta);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SMOOTH_L1_LOSS, self, target, reduction, beta);
}

at::Tensor & wrap_smooth_l1_loss_backward_grad_input(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, double beta, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self, target);
    return at::redispatch::smooth_l1_loss_backward(grad_input, grad_output, self, target, reduction, beta);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_SMOOTH_L1_LOSS_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, target, reduction, beta);
}

at::Tensor wrap_smooth_l1_loss_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, double beta) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, target);
    return at::redispatch::smooth_l1_loss_backward(grad_output, self, target, reduction, beta);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_SMOOTH_L1_LOSS_BACKWARD, grad_output, self, target, reduction, beta);
}

at::Tensor & wrap_huber_loss_out(const at::Tensor & self, const at::Tensor & target, int64_t reduction, double delta, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, target);
    return at::redispatch::huber_loss(out, self, target, reduction, delta);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_HUBER_LOSS_OUT, out, self, target, reduction, delta);
}

at::Tensor wrap_huber_loss(const at::Tensor & self, const at::Tensor & target, int64_t reduction, double delta) {
  if (trace.is_flushing()) {
    ensure_materialized(self, target);
    return at::redispatch::huber_loss(self, target, reduction, delta);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_HUBER_LOSS, self, target, reduction, delta);
}

at::Tensor & wrap_huber_loss_backward_out(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, double delta, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self, target);
    return at::redispatch::huber_loss_backward(grad_input, grad_output, self, target, reduction, delta);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_HUBER_LOSS_BACKWARD_OUT, grad_input, grad_output, self, target, reduction, delta);
}

at::Tensor wrap_huber_loss_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, double delta) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, target);
    return at::redispatch::huber_loss_backward(grad_output, self, target, reduction, delta);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_HUBER_LOSS_BACKWARD, grad_output, self, target, reduction, delta);
}

at::Tensor & wrap_soft_margin_loss_out(const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, target);
    return at::redispatch::soft_margin_loss(out, self, target, reduction);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SOFT_MARGIN_LOSS_OUT, out, self, target, reduction);
}

at::Tensor wrap_soft_margin_loss(const at::Tensor & self, const at::Tensor & target, int64_t reduction) {
  if (trace.is_flushing()) {
    ensure_materialized(self, target);
    return at::redispatch::soft_margin_loss(self, target, reduction);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SOFT_MARGIN_LOSS, self, target, reduction);
}

at::Tensor & wrap_soft_margin_loss_backward_grad_input(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self, target);
    return at::redispatch::soft_margin_loss_backward(grad_input, grad_output, self, target, reduction);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_SOFT_MARGIN_LOSS_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, target, reduction);
}

at::Tensor wrap_soft_margin_loss_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, target);
    return at::redispatch::soft_margin_loss_backward(grad_output, self, target, reduction);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_SOFT_MARGIN_LOSS_BACKWARD, grad_output, self, target, reduction);
}

at::Tensor & wrap_elu_out(const at::Tensor & self, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::elu(out, self, alpha, scale, input_scale);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ELU_OUT, out, self, alpha, scale, input_scale);
}

at::Tensor wrap_elu(const at::Tensor & self, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::elu(self, alpha, scale, input_scale);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ELU, self, alpha, scale, input_scale);
}

at::Tensor wrap_elu_backward(const at::Tensor & grad_output, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale, bool is_result, const at::Tensor & self_or_result) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self_or_result);
    return at::redispatch::elu_backward(grad_output, alpha, scale, input_scale, is_result, self_or_result);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_ELU_BACKWARD, grad_output, alpha, scale, input_scale, is_result, self_or_result);
}

at::Tensor & wrap_elu_(at::Tensor & self, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::elu_(self, alpha, scale, input_scale);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ELU_, self, alpha, scale, input_scale);
}

at::Tensor & wrap_glu_out(const at::Tensor & self, int64_t dim, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::glu(out, self, dim);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_GLU_OUT, out, self, dim);
}

at::Tensor wrap_glu(const at::Tensor & self, int64_t dim) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::glu(self, dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_GLU, self, dim);
}

at::Tensor & wrap_glu_backward_grad_input(const at::Tensor & grad_output, const at::Tensor & self, int64_t dim, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self);
    return at::redispatch::glu_backward(grad_input, grad_output, self, dim);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_GLU_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, dim);
}

at::Tensor wrap_glu_backward(const at::Tensor & grad_output, const at::Tensor & self, int64_t dim) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self);
    return at::redispatch::glu_backward(grad_output, self, dim);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_GLU_BACKWARD, grad_output, self, dim);
}

at::Tensor & wrap_hardsigmoid_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::hardsigmoid(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_HARDSIGMOID_OUT, out, self);
}

at::Tensor wrap_hardsigmoid(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::hardsigmoid(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_HARDSIGMOID, self);
}

at::Tensor & wrap_hardsigmoid_(at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::hardsigmoid_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_HARDSIGMOID_, self);
}

at::Tensor wrap_hardsigmoid_backward(const at::Tensor & grad_output, const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self);
    return at::redispatch::hardsigmoid_backward(grad_output, self);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_HARDSIGMOID_BACKWARD, grad_output, self);
}

at::Tensor & wrap_hardtanh_out(const at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::hardtanh(out, self, min_val, max_val);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_HARDTANH_OUT, out, self, min_val, max_val);
}

at::Tensor wrap_hardtanh(const at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::hardtanh(self, min_val, max_val);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_HARDTANH, self, min_val, max_val);
}

at::Tensor & wrap_hardtanh_backward_grad_input(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self);
    return at::redispatch::hardtanh_backward(grad_input, grad_output, self, min_val, max_val);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_HARDTANH_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, min_val, max_val);
}

at::Tensor wrap_hardtanh_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self);
    return at::redispatch::hardtanh_backward(grad_output, self, min_val, max_val);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_HARDTANH_BACKWARD, grad_output, self, min_val, max_val);
}

at::Tensor & wrap_hardtanh_(at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::hardtanh_(self, min_val, max_val);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_HARDTANH_, self, min_val, max_val);
}

at::Tensor & wrap_hardswish_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::hardswish(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_HARDSWISH_OUT, out, self);
}

at::Tensor wrap_hardswish(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::hardswish(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_HARDSWISH, self);
}

at::Tensor & wrap_hardswish_(at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::hardswish_(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_HARDSWISH_, self);
}

at::Tensor wrap_hardswish_backward(const at::Tensor & grad_output, const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self);
    return at::redispatch::hardswish_backward(grad_output, self);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_HARDSWISH_BACKWARD, grad_output, self);
}

at::Tensor & wrap_leaky_relu_out(const at::Tensor & self, const at::Scalar & negative_slope, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::leaky_relu(out, self, negative_slope);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LEAKY_RELU_OUT, out, self, negative_slope);
}

at::Tensor wrap_leaky_relu(const at::Tensor & self, const at::Scalar & negative_slope) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::leaky_relu(self, negative_slope);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LEAKY_RELU, self, negative_slope);
}

at::Tensor wrap_leaky_relu_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & negative_slope, bool self_is_result) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self);
    return at::redispatch::leaky_relu_backward(grad_output, self, negative_slope, self_is_result);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_LEAKY_RELU_BACKWARD, grad_output, self, negative_slope, self_is_result);
}

at::Tensor & wrap_leaky_relu_(at::Tensor & self, const at::Scalar & negative_slope) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::leaky_relu_(self, negative_slope);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LEAKY_RELU_, self, negative_slope);
}

at::Tensor & wrap_log_sigmoid_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::log_sigmoid(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LOG_SIGMOID_OUT, out, self);
}

at::Tensor wrap_log_sigmoid(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::log_sigmoid(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LOG_SIGMOID, self);
}

std::tuple<at::Tensor &,at::Tensor &> wrap_log_sigmoid_forward_output(const at::Tensor & self, at::Tensor & output, at::Tensor & buffer) {
  ensure_materialized(output, buffer, self);
  return at::redispatch::log_sigmoid_forward(output, buffer, self);
}

std::tuple<at::Tensor,at::Tensor> wrap_log_sigmoid_forward(const at::Tensor & self) {
  ensure_materialized(self);
  return at::redispatch::log_sigmoid_forward(self);
}

at::Tensor & wrap_log_sigmoid_backward_grad_input(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & buffer, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self, buffer);
    return at::redispatch::log_sigmoid_backward(grad_input, grad_output, self, buffer);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_LOG_SIGMOID_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, buffer);
}

at::Tensor wrap_log_sigmoid_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & buffer) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, buffer);
    return at::redispatch::log_sigmoid_backward(grad_output, self, buffer);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_LOG_SIGMOID_BACKWARD, grad_output, self, buffer);
}

at::Tensor & wrap_rrelu_with_noise_out(const at::Tensor & self, const at::Tensor & noise, const at::Scalar & lower, const at::Scalar & upper, bool training, c10::optional<at::Generator> generator, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, noise);
    return at::redispatch::rrelu_with_noise(out, self, noise, lower, upper, training, generator);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_RRELU_WITH_NOISE_OUT, out, self, noise, lower, upper, training, generator);
}

at::Tensor wrap_rrelu_with_noise(const at::Tensor & self, const at::Tensor & noise, const at::Scalar & lower, const at::Scalar & upper, bool training, c10::optional<at::Generator> generator) {
  if (trace.is_flushing()) {
    ensure_materialized(self, noise);
    return at::redispatch::rrelu_with_noise(self, noise, lower, upper, training, generator);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RRELU_WITH_NOISE, self, noise, lower, upper, training, generator);
}

at::Tensor wrap_rrelu_with_noise_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & noise, const at::Scalar & lower, const at::Scalar & upper, bool training, bool self_is_result) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, noise);
    return at::redispatch::rrelu_with_noise_backward(grad_output, self, noise, lower, upper, training, self_is_result);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_RRELU_WITH_NOISE_BACKWARD, grad_output, self, noise, lower, upper, training, self_is_result);
}

at::Tensor & wrap_rrelu_with_noise_(at::Tensor & self, const at::Tensor & noise, const at::Scalar & lower, const at::Scalar & upper, bool training, c10::optional<at::Generator> generator) {
  if (trace.is_flushing()) {
    ensure_materialized(self, noise);
    return at::redispatch::rrelu_with_noise_(self, noise, lower, upper, training, generator);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_RRELU_WITH_NOISE_, self, noise, lower, upper, training, generator);
}

at::Tensor & wrap_softplus_out(const at::Tensor & self, const at::Scalar & beta, const at::Scalar & threshold, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::softplus(out, self, beta, threshold);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SOFTPLUS_OUT, out, self, beta, threshold);
}

at::Tensor wrap_softplus(const at::Tensor & self, const at::Scalar & beta, const at::Scalar & threshold) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::softplus(self, beta, threshold);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SOFTPLUS, self, beta, threshold);
}

at::Tensor & wrap_softplus_backward_grad_input(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & beta, const at::Scalar & threshold, const at::Tensor & output, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self, output);
    return at::redispatch::softplus_backward(grad_input, grad_output, self, beta, threshold, output);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_SOFTPLUS_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, beta, threshold, output);
}

at::Tensor wrap_softplus_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & beta, const at::Scalar & threshold, const at::Tensor & output) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, output);
    return at::redispatch::softplus_backward(grad_output, self, beta, threshold, output);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_SOFTPLUS_BACKWARD, grad_output, self, beta, threshold, output);
}

at::Tensor & wrap_softshrink_out(const at::Tensor & self, const at::Scalar & lambd, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::softshrink(out, self, lambd);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SOFTSHRINK_OUT, out, self, lambd);
}

at::Tensor wrap_softshrink(const at::Tensor & self, const at::Scalar & lambd) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::softshrink(self, lambd);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SOFTSHRINK, self, lambd);
}

at::Tensor & wrap_softshrink_backward_grad_input(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & lambd, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self);
    return at::redispatch::softshrink_backward(grad_input, grad_output, self, lambd);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_SOFTSHRINK_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, lambd);
}

at::Tensor wrap_softshrink_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & lambd) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self);
    return at::redispatch::softshrink_backward(grad_output, self, lambd);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_SOFTSHRINK_BACKWARD, grad_output, self, lambd);
}

at::Tensor & wrap_adaptive_avg_pool2d_out(const at::Tensor & self, at::IntArrayRef output_size, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::adaptive_avg_pool2d(out, self, output_size);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ADAPTIVE_AVG_POOL2D_OUT, out, self, output_size);
}

at::Tensor wrap_adaptive_avg_pool2d(const at::Tensor & self, at::IntArrayRef output_size) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::adaptive_avg_pool2d(self, output_size);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ADAPTIVE_AVG_POOL2D, self, output_size);
}

at::Tensor wrap_mkldnn_adaptive_avg_pool2d(const at::Tensor & self, at::IntArrayRef output_size) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::mkldnn_adaptive_avg_pool2d(self, output_size);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MKLDNN_ADAPTIVE_AVG_POOL2D, self, output_size);
}

at::Tensor wrap_mkldnn_adaptive_avg_pool2d_backward(const at::Tensor & grad_output, const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self);
    return at::redispatch::mkldnn_adaptive_avg_pool2d_backward(grad_output, self);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_MKLDNN_ADAPTIVE_AVG_POOL2D_BACKWARD, grad_output, self);
}

at::Tensor wrap__adaptive_avg_pool2d(const at::Tensor & self, at::IntArrayRef output_size) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_adaptive_avg_pool2d(self, output_size);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__ADAPTIVE_AVG_POOL2D, self, output_size);
}

at::Tensor wrap__adaptive_avg_pool2d_backward(const at::Tensor & grad_output, const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self);
    return at::redispatch::_adaptive_avg_pool2d_backward(grad_output, self);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H__ADAPTIVE_AVG_POOL2D_BACKWARD, grad_output, self);
}

at::Tensor & wrap_adaptive_avg_pool3d_out(const at::Tensor & self, at::IntArrayRef output_size, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::adaptive_avg_pool3d(out, self, output_size);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ADAPTIVE_AVG_POOL3D_OUT, out, self, output_size);
}

at::Tensor wrap_adaptive_avg_pool3d(const at::Tensor & self, at::IntArrayRef output_size) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::adaptive_avg_pool3d(self, output_size);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ADAPTIVE_AVG_POOL3D, self, output_size);
}

at::Tensor wrap__adaptive_avg_pool3d(const at::Tensor & self, at::IntArrayRef output_size) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_adaptive_avg_pool3d(self, output_size);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__ADAPTIVE_AVG_POOL3D, self, output_size);
}

at::Tensor & wrap_adaptive_avg_pool3d_backward_grad_input(const at::Tensor & grad_output, const at::Tensor & self, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self);
    return at::redispatch::adaptive_avg_pool3d_backward(grad_input, grad_output, self);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_ADAPTIVE_AVG_POOL3D_BACKWARD_GRAD_INPUT, grad_input, grad_output, self);
}

at::Tensor wrap__adaptive_avg_pool3d_backward(const at::Tensor & grad_output, const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self);
    return at::redispatch::_adaptive_avg_pool3d_backward(grad_output, self);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H__ADAPTIVE_AVG_POOL3D_BACKWARD, grad_output, self);
}

std::tuple<at::Tensor &,at::Tensor &> wrap_adaptive_max_pool2d_out(const at::Tensor & self, at::IntArrayRef output_size, at::Tensor & out, at::Tensor & indices) {
  ensure_materialized(out, indices, self);
  return at::redispatch::adaptive_max_pool2d(out, indices, self, output_size);
}

at::Tensor & wrap_adaptive_max_pool2d_backward_grad_input(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self, indices);
    return at::redispatch::adaptive_max_pool2d_backward(grad_input, grad_output, self, indices);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_ADAPTIVE_MAX_POOL2D_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, indices);
}

at::Tensor wrap_adaptive_max_pool2d_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, indices);
    return at::redispatch::adaptive_max_pool2d_backward(grad_output, self, indices);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_ADAPTIVE_MAX_POOL2D_BACKWARD, grad_output, self, indices);
}

std::tuple<at::Tensor &,at::Tensor &> wrap_adaptive_max_pool3d_out(const at::Tensor & self, at::IntArrayRef output_size, at::Tensor & out, at::Tensor & indices) {
  ensure_materialized(out, indices, self);
  return at::redispatch::adaptive_max_pool3d(out, indices, self, output_size);
}

at::Tensor & wrap_adaptive_max_pool3d_backward_grad_input(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self, indices);
    return at::redispatch::adaptive_max_pool3d_backward(grad_input, grad_output, self, indices);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_ADAPTIVE_MAX_POOL3D_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, indices);
}

at::Tensor wrap_adaptive_max_pool3d_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, indices);
    return at::redispatch::adaptive_max_pool3d_backward(grad_output, self, indices);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_ADAPTIVE_MAX_POOL3D_BACKWARD, grad_output, self, indices);
}

at::Tensor & wrap_avg_pool2d_out(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::avg_pool2d(out, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_AVG_POOL2D_OUT, out, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}

at::Tensor wrap_avg_pool2d(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::avg_pool2d(self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_AVG_POOL2D, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}

at::Tensor & wrap_avg_pool2d_backward_grad_input(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self);
    return at::redispatch::avg_pool2d_backward(grad_input, grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_AVG_POOL2D_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}

at::Tensor wrap_avg_pool2d_backward(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self);
    return at::redispatch::avg_pool2d_backward(grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_AVG_POOL2D_BACKWARD, grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}

at::Tensor & wrap_avg_pool3d_out(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::avg_pool3d(out, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_AVG_POOL3D_OUT, out, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}

at::Tensor wrap_avg_pool3d(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::avg_pool3d(self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_AVG_POOL3D, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}

at::Tensor & wrap_avg_pool3d_backward_grad_input(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self);
    return at::redispatch::avg_pool3d_backward(grad_input, grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_AVG_POOL3D_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}

at::Tensor wrap_avg_pool3d_backward(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self);
    return at::redispatch::avg_pool3d_backward(grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_AVG_POOL3D_BACKWARD, grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}

std::tuple<at::Tensor &,at::Tensor &> wrap_fractional_max_pool2d_output(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef output_size, const at::Tensor & random_samples, at::Tensor & output, at::Tensor & indices) {
  ensure_materialized(output, indices, self, random_samples);
  return at::redispatch::fractional_max_pool2d(output, indices, self, kernel_size, output_size, random_samples);
}

at::Tensor & wrap_fractional_max_pool2d_backward_grad_input(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef output_size, const at::Tensor & indices, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self, indices);
    return at::redispatch::fractional_max_pool2d_backward(grad_input, grad_output, self, kernel_size, output_size, indices);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_FRACTIONAL_MAX_POOL2D_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, kernel_size, output_size, indices);
}

at::Tensor wrap_fractional_max_pool2d_backward(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef output_size, const at::Tensor & indices) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, indices);
    return at::redispatch::fractional_max_pool2d_backward(grad_output, self, kernel_size, output_size, indices);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_FRACTIONAL_MAX_POOL2D_BACKWARD, grad_output, self, kernel_size, output_size, indices);
}

std::tuple<at::Tensor &,at::Tensor &> wrap_fractional_max_pool3d_output(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef output_size, const at::Tensor & random_samples, at::Tensor & output, at::Tensor & indices) {
  ensure_materialized(output, indices, self, random_samples);
  return at::redispatch::fractional_max_pool3d(output, indices, self, kernel_size, output_size, random_samples);
}

std::tuple<at::Tensor,at::Tensor> wrap_fractional_max_pool3d(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef output_size, const at::Tensor & random_samples) {
  ensure_materialized(self, random_samples);
  return at::redispatch::fractional_max_pool3d(self, kernel_size, output_size, random_samples);
}

at::Tensor & wrap_fractional_max_pool3d_backward_grad_input(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef output_size, const at::Tensor & indices, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self, indices);
    return at::redispatch::fractional_max_pool3d_backward(grad_input, grad_output, self, kernel_size, output_size, indices);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_FRACTIONAL_MAX_POOL3D_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, kernel_size, output_size, indices);
}

at::Tensor wrap_fractional_max_pool3d_backward(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef output_size, const at::Tensor & indices) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, indices);
    return at::redispatch::fractional_max_pool3d_backward(grad_output, self, kernel_size, output_size, indices);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_FRACTIONAL_MAX_POOL3D_BACKWARD, grad_output, self, kernel_size, output_size, indices);
}

std::tuple<at::Tensor &,at::Tensor &> wrap_max_pool2d_with_indices_out(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, at::Tensor & out, at::Tensor & indices) {
  ensure_materialized(out, indices, self);
  return at::redispatch::max_pool2d_with_indices(out, indices, self, kernel_size, stride, padding, dilation, ceil_mode);
}

std::tuple<at::Tensor,at::Tensor> wrap_max_pool2d_with_indices(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  ensure_materialized(self);
  return at::redispatch::max_pool2d_with_indices(self, kernel_size, stride, padding, dilation, ceil_mode);
}

at::Tensor & wrap_max_pool2d_with_indices_backward_grad_input(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, const at::Tensor & indices, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self, indices);
    return at::redispatch::max_pool2d_with_indices_backward(grad_input, grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_MAX_POOL2D_WITH_INDICES_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
}

at::Tensor wrap_max_pool2d_with_indices_backward(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, const at::Tensor & indices) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, indices);
    return at::redispatch::max_pool2d_with_indices_backward(grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_MAX_POOL2D_WITH_INDICES_BACKWARD, grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
}

std::tuple<at::Tensor &,at::Tensor &> wrap_max_pool3d_with_indices_out(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, at::Tensor & out, at::Tensor & indices) {
  ensure_materialized(out, indices, self);
  return at::redispatch::max_pool3d_with_indices(out, indices, self, kernel_size, stride, padding, dilation, ceil_mode);
}

std::tuple<at::Tensor,at::Tensor> wrap_max_pool3d_with_indices(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  ensure_materialized(self);
  return at::redispatch::max_pool3d_with_indices(self, kernel_size, stride, padding, dilation, ceil_mode);
}

at::Tensor & wrap_max_pool3d_with_indices_backward_grad_input(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, const at::Tensor & indices, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self, indices);
    return at::redispatch::max_pool3d_with_indices_backward(grad_input, grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_MAX_POOL3D_WITH_INDICES_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
}

at::Tensor wrap_max_pool3d_with_indices_backward(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, const at::Tensor & indices) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, indices);
    return at::redispatch::max_pool3d_with_indices_backward(grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_MAX_POOL3D_WITH_INDICES_BACKWARD, grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
}

at::Tensor & wrap_max_unpool2d_out(const at::Tensor & self, const at::Tensor & indices, at::IntArrayRef output_size, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, indices);
    return at::redispatch::max_unpool2d(out, self, indices, output_size);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_MAX_UNPOOL2D_OUT, out, self, indices, output_size);
}

at::Tensor wrap_max_unpool2d(const at::Tensor & self, const at::Tensor & indices, at::IntArrayRef output_size) {
  if (trace.is_flushing()) {
    ensure_materialized(self, indices);
    return at::redispatch::max_unpool2d(self, indices, output_size);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MAX_UNPOOL2D, self, indices, output_size);
}

at::Tensor & wrap_max_unpool2d_backward_grad_input(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices, at::IntArrayRef output_size, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self, indices);
    return at::redispatch::max_unpool2d_backward(grad_input, grad_output, self, indices, output_size);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_MAX_UNPOOL2D_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, indices, output_size);
}

at::Tensor wrap_max_unpool2d_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices, at::IntArrayRef output_size) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, indices);
    return at::redispatch::max_unpool2d_backward(grad_output, self, indices, output_size);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_MAX_UNPOOL2D_BACKWARD, grad_output, self, indices, output_size);
}

at::Tensor & wrap_max_unpool3d_out(const at::Tensor & self, const at::Tensor & indices, at::IntArrayRef output_size, at::IntArrayRef stride, at::IntArrayRef padding, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, indices);
    return at::redispatch::max_unpool3d(out, self, indices, output_size, stride, padding);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_MAX_UNPOOL3D_OUT, out, self, indices, output_size, stride, padding);
}

at::Tensor wrap_max_unpool3d(const at::Tensor & self, const at::Tensor & indices, at::IntArrayRef output_size, at::IntArrayRef stride, at::IntArrayRef padding) {
  if (trace.is_flushing()) {
    ensure_materialized(self, indices);
    return at::redispatch::max_unpool3d(self, indices, output_size, stride, padding);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_MAX_UNPOOL3D, self, indices, output_size, stride, padding);
}

at::Tensor & wrap_max_unpool3d_backward_grad_input(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices, at::IntArrayRef output_size, at::IntArrayRef stride, at::IntArrayRef padding, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self, indices);
    return at::redispatch::max_unpool3d_backward(grad_input, grad_output, self, indices, output_size, stride, padding);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_MAX_UNPOOL3D_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, indices, output_size, stride, padding);
}

at::Tensor wrap_max_unpool3d_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices, at::IntArrayRef output_size, at::IntArrayRef stride, at::IntArrayRef padding) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self, indices);
    return at::redispatch::max_unpool3d_backward(grad_output, self, indices, output_size, stride, padding);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_MAX_UNPOOL3D_BACKWARD, grad_output, self, indices, output_size, stride, padding);
}

at::Tensor & wrap_reflection_pad1d_out(const at::Tensor & self, at::IntArrayRef padding, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::reflection_pad1d(out, self, padding);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_REFLECTION_PAD1D_OUT, out, self, padding);
}

at::Tensor wrap_reflection_pad1d(const at::Tensor & self, at::IntArrayRef padding) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::reflection_pad1d(self, padding);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_REFLECTION_PAD1D, self, padding);
}

at::Tensor & wrap_reflection_pad1d_backward_grad_input(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self);
    return at::redispatch::reflection_pad1d_backward(grad_input, grad_output, self, padding);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_REFLECTION_PAD1D_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, padding);
}

at::Tensor wrap_reflection_pad1d_backward(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self);
    return at::redispatch::reflection_pad1d_backward(grad_output, self, padding);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_REFLECTION_PAD1D_BACKWARD, grad_output, self, padding);
}

at::Tensor & wrap_reflection_pad2d_out(const at::Tensor & self, at::IntArrayRef padding, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::reflection_pad2d(out, self, padding);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_REFLECTION_PAD2D_OUT, out, self, padding);
}

at::Tensor wrap_reflection_pad2d(const at::Tensor & self, at::IntArrayRef padding) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::reflection_pad2d(self, padding);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_REFLECTION_PAD2D, self, padding);
}

at::Tensor & wrap_reflection_pad2d_backward_grad_input(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self);
    return at::redispatch::reflection_pad2d_backward(grad_input, grad_output, self, padding);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_REFLECTION_PAD2D_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, padding);
}

at::Tensor wrap_reflection_pad2d_backward(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self);
    return at::redispatch::reflection_pad2d_backward(grad_output, self, padding);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_REFLECTION_PAD2D_BACKWARD, grad_output, self, padding);
}

at::Tensor & wrap_replication_pad1d_out(const at::Tensor & self, at::IntArrayRef padding, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::replication_pad1d(out, self, padding);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_REPLICATION_PAD1D_OUT, out, self, padding);
}

at::Tensor & wrap_replication_pad1d_backward_grad_input(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self);
    return at::redispatch::replication_pad1d_backward(grad_input, grad_output, self, padding);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_REPLICATION_PAD1D_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, padding);
}

at::Tensor & wrap_replication_pad2d_out(const at::Tensor & self, at::IntArrayRef padding, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::replication_pad2d(out, self, padding);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_REPLICATION_PAD2D_OUT, out, self, padding);
}

at::Tensor & wrap_replication_pad2d_backward_grad_input(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self);
    return at::redispatch::replication_pad2d_backward(grad_input, grad_output, self, padding);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_REPLICATION_PAD2D_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, padding);
}

at::Tensor wrap_replication_pad2d_backward(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self);
    return at::redispatch::replication_pad2d_backward(grad_output, self, padding);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_REPLICATION_PAD2D_BACKWARD, grad_output, self, padding);
}

at::Tensor & wrap_replication_pad3d_out(const at::Tensor & self, at::IntArrayRef padding, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::replication_pad3d(out, self, padding);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_REPLICATION_PAD3D_OUT, out, self, padding);
}

at::Tensor & wrap_replication_pad3d_backward_grad_input(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self);
    return at::redispatch::replication_pad3d_backward(grad_input, grad_output, self, padding);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_REPLICATION_PAD3D_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, padding);
}

at::Tensor wrap_replication_pad3d_backward(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self);
    return at::redispatch::replication_pad3d_backward(grad_output, self, padding);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_REPLICATION_PAD3D_BACKWARD, grad_output, self, padding);
}

at::Tensor wrap_upsample_linear1d_vec(const at::Tensor & input, c10::optional<at::IntArrayRef> output_size, bool align_corners, c10::optional<at::ArrayRef<double>> scale_factors) {
  if (trace.is_flushing()) {
    ensure_materialized(input);
    return at::redispatch::upsample_linear1d(input, output_size, align_corners, scale_factors);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_UPSAMPLE_LINEAR1D_VEC, input, output_size, align_corners, scale_factors);
}

at::Tensor wrap_upsample_linear1d_backward_vec(const at::Tensor & grad_output, c10::optional<at::IntArrayRef> output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<at::ArrayRef<double>> scale_factors) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output);
    return at::redispatch::upsample_linear1d_backward(grad_output, output_size, input_size, align_corners, scale_factors);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_UPSAMPLE_LINEAR1D_BACKWARD_VEC, grad_output, output_size, input_size, align_corners, scale_factors);
}

at::Tensor wrap_upsample_bilinear2d_vec(const at::Tensor & input, c10::optional<at::IntArrayRef> output_size, bool align_corners, c10::optional<at::ArrayRef<double>> scale_factors) {
  if (trace.is_flushing()) {
    ensure_materialized(input);
    return at::redispatch::upsample_bilinear2d(input, output_size, align_corners, scale_factors);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_UPSAMPLE_BILINEAR2D_VEC, input, output_size, align_corners, scale_factors);
}

at::Tensor wrap_upsample_bilinear2d_backward_vec(const at::Tensor & grad_output, c10::optional<at::IntArrayRef> output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<at::ArrayRef<double>> scale_factors) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output);
    return at::redispatch::upsample_bilinear2d_backward(grad_output, output_size, input_size, align_corners, scale_factors);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_UPSAMPLE_BILINEAR2D_BACKWARD_VEC, grad_output, output_size, input_size, align_corners, scale_factors);
}

at::Tensor wrap_upsample_trilinear3d_vec(const at::Tensor & input, c10::optional<at::IntArrayRef> output_size, bool align_corners, c10::optional<at::ArrayRef<double>> scale_factors) {
  if (trace.is_flushing()) {
    ensure_materialized(input);
    return at::redispatch::upsample_trilinear3d(input, output_size, align_corners, scale_factors);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_UPSAMPLE_TRILINEAR3D_VEC, input, output_size, align_corners, scale_factors);
}

at::Tensor wrap_upsample_trilinear3d_backward_vec(const at::Tensor & grad_output, c10::optional<at::IntArrayRef> output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<at::ArrayRef<double>> scale_factors) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output);
    return at::redispatch::upsample_trilinear3d_backward(grad_output, output_size, input_size, align_corners, scale_factors);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_UPSAMPLE_TRILINEAR3D_BACKWARD_VEC, grad_output, output_size, input_size, align_corners, scale_factors);
}

at::Tensor wrap_upsample_bicubic2d_vec(const at::Tensor & input, c10::optional<at::IntArrayRef> output_size, bool align_corners, c10::optional<at::ArrayRef<double>> scale_factors) {
  if (trace.is_flushing()) {
    ensure_materialized(input);
    return at::redispatch::upsample_bicubic2d(input, output_size, align_corners, scale_factors);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_UPSAMPLE_BICUBIC2D_VEC, input, output_size, align_corners, scale_factors);
}

at::Tensor wrap_upsample_bicubic2d_backward_vec(const at::Tensor & grad_output, c10::optional<at::IntArrayRef> output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<at::ArrayRef<double>> scale_factors) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output);
    return at::redispatch::upsample_bicubic2d_backward(grad_output, output_size, input_size, align_corners, scale_factors);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_UPSAMPLE_BICUBIC2D_BACKWARD_VEC, grad_output, output_size, input_size, align_corners, scale_factors);
}

at::Tensor wrap_upsample_nearest1d_vec(const at::Tensor & input, c10::optional<at::IntArrayRef> output_size, c10::optional<at::ArrayRef<double>> scale_factors) {
  if (trace.is_flushing()) {
    ensure_materialized(input);
    return at::redispatch::upsample_nearest1d(input, output_size, scale_factors);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_UPSAMPLE_NEAREST1D_VEC, input, output_size, scale_factors);
}

at::Tensor wrap_upsample_nearest1d_backward_vec(const at::Tensor & grad_output, c10::optional<at::IntArrayRef> output_size, at::IntArrayRef input_size, c10::optional<at::ArrayRef<double>> scale_factors) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output);
    return at::redispatch::upsample_nearest1d_backward(grad_output, output_size, input_size, scale_factors);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_UPSAMPLE_NEAREST1D_BACKWARD_VEC, grad_output, output_size, input_size, scale_factors);
}

at::Tensor wrap_upsample_nearest2d_vec(const at::Tensor & input, c10::optional<at::IntArrayRef> output_size, c10::optional<at::ArrayRef<double>> scale_factors) {
  if (trace.is_flushing()) {
    ensure_materialized(input);
    return at::redispatch::upsample_nearest2d(input, output_size, scale_factors);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_UPSAMPLE_NEAREST2D_VEC, input, output_size, scale_factors);
}

at::Tensor wrap_upsample_nearest2d_backward_vec(const at::Tensor & grad_output, c10::optional<at::IntArrayRef> output_size, at::IntArrayRef input_size, c10::optional<at::ArrayRef<double>> scale_factors) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output);
    return at::redispatch::upsample_nearest2d_backward(grad_output, output_size, input_size, scale_factors);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_UPSAMPLE_NEAREST2D_BACKWARD_VEC, grad_output, output_size, input_size, scale_factors);
}

at::Tensor wrap_upsample_nearest3d_vec(const at::Tensor & input, c10::optional<at::IntArrayRef> output_size, c10::optional<at::ArrayRef<double>> scale_factors) {
  if (trace.is_flushing()) {
    ensure_materialized(input);
    return at::redispatch::upsample_nearest3d(input, output_size, scale_factors);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_UPSAMPLE_NEAREST3D_VEC, input, output_size, scale_factors);
}

at::Tensor wrap_upsample_nearest3d_backward_vec(const at::Tensor & grad_output, c10::optional<at::IntArrayRef> output_size, at::IntArrayRef input_size, c10::optional<at::ArrayRef<double>> scale_factors) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output);
    return at::redispatch::upsample_nearest3d_backward(grad_output, output_size, input_size, scale_factors);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_UPSAMPLE_NEAREST3D_BACKWARD_VEC, grad_output, output_size, input_size, scale_factors);
}

at::Tensor & wrap_upsample_linear1d_out(const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::upsample_linear1d(out, self, output_size, align_corners, scales);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_UPSAMPLE_LINEAR1D_OUT, out, self, output_size, align_corners, scales);
}

at::Tensor & wrap_upsample_linear1d_backward_grad_input(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output);
    return at::redispatch::upsample_linear1d_backward(grad_input, grad_output, output_size, input_size, align_corners, scales);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_UPSAMPLE_LINEAR1D_BACKWARD_GRAD_INPUT, grad_input, grad_output, output_size, input_size, align_corners, scales);
}

at::Tensor & wrap_upsample_bilinear2d_out(const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::upsample_bilinear2d(out, self, output_size, align_corners, scales_h, scales_w);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_UPSAMPLE_BILINEAR2D_OUT, out, self, output_size, align_corners, scales_h, scales_w);
}

at::Tensor wrap_upsample_bilinear2d(const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::upsample_bilinear2d(self, output_size, align_corners, scales_h, scales_w);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_UPSAMPLE_BILINEAR2D, self, output_size, align_corners, scales_h, scales_w);
}

at::Tensor & wrap_upsample_bilinear2d_backward_grad_input(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output);
    return at::redispatch::upsample_bilinear2d_backward(grad_input, grad_output, output_size, input_size, align_corners, scales_h, scales_w);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_UPSAMPLE_BILINEAR2D_BACKWARD_GRAD_INPUT, grad_input, grad_output, output_size, input_size, align_corners, scales_h, scales_w);
}

at::Tensor & wrap_upsample_bicubic2d_out(const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::upsample_bicubic2d(out, self, output_size, align_corners, scales_h, scales_w);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_UPSAMPLE_BICUBIC2D_OUT, out, self, output_size, align_corners, scales_h, scales_w);
}

at::Tensor & wrap_upsample_bicubic2d_backward_grad_input(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output);
    return at::redispatch::upsample_bicubic2d_backward(grad_input, grad_output, output_size, input_size, align_corners, scales_h, scales_w);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_UPSAMPLE_BICUBIC2D_BACKWARD_GRAD_INPUT, grad_input, grad_output, output_size, input_size, align_corners, scales_h, scales_w);
}

at::Tensor & wrap_upsample_trilinear3d_out(const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::upsample_trilinear3d(out, self, output_size, align_corners, scales_d, scales_h, scales_w);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_UPSAMPLE_TRILINEAR3D_OUT, out, self, output_size, align_corners, scales_d, scales_h, scales_w);
}

at::Tensor & wrap_upsample_trilinear3d_backward_grad_input(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output);
    return at::redispatch::upsample_trilinear3d_backward(grad_input, grad_output, output_size, input_size, align_corners, scales_d, scales_h, scales_w);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_UPSAMPLE_TRILINEAR3D_BACKWARD_GRAD_INPUT, grad_input, grad_output, output_size, input_size, align_corners, scales_d, scales_h, scales_w);
}

at::Tensor & wrap_upsample_nearest1d_out(const at::Tensor & self, at::IntArrayRef output_size, c10::optional<double> scales, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::upsample_nearest1d(out, self, output_size, scales);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_UPSAMPLE_NEAREST1D_OUT, out, self, output_size, scales);
}

at::Tensor & wrap_upsample_nearest1d_backward_grad_input(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, c10::optional<double> scales, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output);
    return at::redispatch::upsample_nearest1d_backward(grad_input, grad_output, output_size, input_size, scales);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_UPSAMPLE_NEAREST1D_BACKWARD_GRAD_INPUT, grad_input, grad_output, output_size, input_size, scales);
}

at::Tensor & wrap_upsample_nearest2d_out(const at::Tensor & self, at::IntArrayRef output_size, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::upsample_nearest2d(out, self, output_size, scales_h, scales_w);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_UPSAMPLE_NEAREST2D_OUT, out, self, output_size, scales_h, scales_w);
}

at::Tensor wrap_upsample_nearest2d(const at::Tensor & self, at::IntArrayRef output_size, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::upsample_nearest2d(self, output_size, scales_h, scales_w);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_UPSAMPLE_NEAREST2D, self, output_size, scales_h, scales_w);
}

at::Tensor & wrap_upsample_nearest2d_backward_grad_input(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output);
    return at::redispatch::upsample_nearest2d_backward(grad_input, grad_output, output_size, input_size, scales_h, scales_w);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_UPSAMPLE_NEAREST2D_BACKWARD_GRAD_INPUT, grad_input, grad_output, output_size, input_size, scales_h, scales_w);
}

at::Tensor & wrap_upsample_nearest3d_out(const at::Tensor & self, at::IntArrayRef output_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::upsample_nearest3d(out, self, output_size, scales_d, scales_h, scales_w);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_UPSAMPLE_NEAREST3D_OUT, out, self, output_size, scales_d, scales_h, scales_w);
}

at::Tensor wrap_upsample_nearest3d(const at::Tensor & self, at::IntArrayRef output_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::upsample_nearest3d(self, output_size, scales_d, scales_h, scales_w);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_UPSAMPLE_NEAREST3D, self, output_size, scales_d, scales_h, scales_w);
}

at::Tensor & wrap_upsample_nearest3d_backward_grad_input(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output);
    return at::redispatch::upsample_nearest3d_backward(grad_input, grad_output, output_size, input_size, scales_d, scales_h, scales_w);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_UPSAMPLE_NEAREST3D_BACKWARD_GRAD_INPUT, grad_input, grad_output, output_size, input_size, scales_d, scales_h, scales_w);
}

at::Tensor & wrap_sigmoid_backward_grad_input(const at::Tensor & grad_output, const at::Tensor & output, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, output);
    return at::redispatch::sigmoid_backward(grad_input, grad_output, output);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_SIGMOID_BACKWARD_GRAD_INPUT, grad_input, grad_output, output);
}

at::Tensor wrap_sigmoid_backward(const at::Tensor & grad_output, const at::Tensor & output) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, output);
    return at::redispatch::sigmoid_backward(grad_output, output);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_SIGMOID_BACKWARD, grad_output, output);
}

at::Tensor & wrap_logit_backward_grad_input(const at::Tensor & grad_output, const at::Tensor & self, c10::optional<double> eps, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, self);
    return at::redispatch::logit_backward(grad_input, grad_output, self, eps);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_LOGIT_BACKWARD_GRAD_INPUT, grad_input, grad_output, self, eps);
}

at::Tensor wrap_logit_backward(const at::Tensor & grad_output, const at::Tensor & self, c10::optional<double> eps) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, self);
    return at::redispatch::logit_backward(grad_output, self, eps);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_LOGIT_BACKWARD, grad_output, self, eps);
}

at::Tensor & wrap_tanh_backward_grad_input(const at::Tensor & grad_output, const at::Tensor & output, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output, output);
    return at::redispatch::tanh_backward(grad_input, grad_output, output);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_TANH_BACKWARD_GRAD_INPUT, grad_input, grad_output, output);
}

at::Tensor wrap_tanh_backward(const at::Tensor & grad_output, const at::Tensor & output) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output, output);
    return at::redispatch::tanh_backward(grad_output, output);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_TANH_BACKWARD, grad_output, output);
}

at::Tensor & wrap_slow_conv_transpose2d_out(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, weight);
    return at::redispatch::slow_conv_transpose2d(out, self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SLOW_CONV_TRANSPOSE2D_OUT, out, self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}

at::Tensor wrap_slow_conv_transpose2d(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation) {
  if (trace.is_flushing()) {
    ensure_materialized(self, weight);
    return at::redispatch::slow_conv_transpose2d(self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SLOW_CONV_TRANSPOSE2D, self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}

std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> wrap_slow_conv_transpose2d_backward_grad_output(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation, const at::Tensor & columns, const at::Tensor & ones, at::Tensor & grad_input, at::Tensor & grad_weight, at::Tensor & grad_bias) {
  ensure_materialized(grad_input, grad_weight, grad_bias, grad_output, self, weight, columns, ones);
  return at::redispatch::slow_conv_transpose2d_backward(grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, columns, ones);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> wrap_slow_conv_transpose2d_backward_output_mask(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation, const at::Tensor & columns, const at::Tensor & ones, std::array<bool,3> output_mask) {
  ensure_materialized(grad_output, self, weight, columns, ones);
  return at::redispatch::slow_conv_transpose2d_backward(grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, columns, ones, output_mask);
}

at::Tensor & wrap_slow_conv_transpose3d_out(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, weight);
    return at::redispatch::slow_conv_transpose3d(out, self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SLOW_CONV_TRANSPOSE3D_OUT, out, self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}

at::Tensor wrap_slow_conv_transpose3d(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation) {
  if (trace.is_flushing()) {
    ensure_materialized(self, weight);
    return at::redispatch::slow_conv_transpose3d(self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SLOW_CONV_TRANSPOSE3D, self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}

std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> wrap_slow_conv_transpose3d_backward_grad_output(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation, const at::Tensor & finput, const at::Tensor & fgrad_input, at::Tensor & grad_input, at::Tensor & grad_weight, at::Tensor & grad_bias) {
  ensure_materialized(grad_input, grad_weight, grad_bias, grad_output, self, weight, finput, fgrad_input);
  return at::redispatch::slow_conv_transpose3d_backward(grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, finput, fgrad_input);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> wrap_slow_conv_transpose3d_backward_output_mask(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation, const at::Tensor & finput, const at::Tensor & fgrad_input, std::array<bool,3> output_mask) {
  ensure_materialized(grad_output, self, weight, finput, fgrad_input);
  return at::redispatch::slow_conv_transpose3d_backward(grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, finput, fgrad_input, output_mask);
}

at::Tensor & wrap_thnn_conv2d_out(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, weight);
    return at::redispatch::thnn_conv2d(out, self, weight, kernel_size, bias, stride, padding);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_THNN_CONV2D_OUT, out, self, weight, kernel_size, bias, stride, padding);
}

at::Tensor wrap_thnn_conv2d(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding) {
  if (trace.is_flushing()) {
    ensure_materialized(self, weight);
    return at::redispatch::thnn_conv2d(self, weight, kernel_size, bias, stride, padding);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_THNN_CONV2D, self, weight, kernel_size, bias, stride, padding);
}

std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> wrap_thnn_conv2d_forward_output(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::Tensor & output, at::Tensor & finput, at::Tensor & fgrad_input) {
  ensure_materialized(output, finput, fgrad_input, self, weight);
  return at::redispatch::thnn_conv2d_forward(output, finput, fgrad_input, self, weight, kernel_size, bias, stride, padding);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> wrap_thnn_conv2d_forward(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding) {
  ensure_materialized(self, weight);
  return at::redispatch::thnn_conv2d_forward(self, weight, kernel_size, bias, stride, padding);
}

std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> wrap_thnn_conv2d_backward_grad_input(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, const at::Tensor & finput, const at::Tensor & fgrad_input, at::Tensor & grad_input, at::Tensor & grad_weight, at::Tensor & grad_bias) {
  ensure_materialized(grad_input, grad_weight, grad_bias, grad_output, self, weight, finput, fgrad_input);
  return at::redispatch::thnn_conv2d_backward(grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> wrap_thnn_conv2d_backward_output_mask(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, const at::Tensor & finput, const at::Tensor & fgrad_input, std::array<bool,3> output_mask) {
  ensure_materialized(grad_output, self, weight, finput, fgrad_input);
  return at::redispatch::thnn_conv2d_backward(grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input, output_mask);
}

at::Tensor & wrap_thnn_conv_depthwise2d_out(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, weight);
    return at::redispatch::thnn_conv_depthwise2d(out, self, weight, kernel_size, bias, stride, padding, dilation);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_THNN_CONV_DEPTHWISE2D_OUT, out, self, weight, kernel_size, bias, stride, padding, dilation);
}

at::Tensor wrap_thnn_conv_depthwise2d(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation) {
  if (trace.is_flushing()) {
    ensure_materialized(self, weight);
    return at::redispatch::thnn_conv_depthwise2d(self, weight, kernel_size, bias, stride, padding, dilation);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_THNN_CONV_DEPTHWISE2D, self, weight, kernel_size, bias, stride, padding, dilation);
}

at::Tensor & wrap_thnn_conv_depthwise2d_forward_out(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, weight);
    return at::redispatch::thnn_conv_depthwise2d_forward(out, self, weight, kernel_size, bias, stride, padding, dilation);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_THNN_CONV_DEPTHWISE2D_FORWARD_OUT, out, self, weight, kernel_size, bias, stride, padding, dilation);
}

at::Tensor wrap_thnn_conv_depthwise2d_forward(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation) {
  if (trace.is_flushing()) {
    ensure_materialized(self, weight);
    return at::redispatch::thnn_conv_depthwise2d_forward(self, weight, kernel_size, bias, stride, padding, dilation);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_THNN_CONV_DEPTHWISE2D_FORWARD, self, weight, kernel_size, bias, stride, padding, dilation);
}

std::tuple<at::Tensor &,at::Tensor &> wrap_thnn_conv_depthwise2d_backward_grad_input(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, at::Tensor & grad_input, at::Tensor & grad_weight) {
  ensure_materialized(grad_input, grad_weight, grad_output, self, weight);
  return at::redispatch::thnn_conv_depthwise2d_backward(grad_input, grad_weight, grad_output, self, weight, kernel_size, stride, padding, dilation);
}

std::tuple<at::Tensor,at::Tensor> wrap_thnn_conv_depthwise2d_backward_output_mask(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, std::array<bool,2> output_mask) {
  ensure_materialized(grad_output, self, weight);
  return at::redispatch::thnn_conv_depthwise2d_backward(grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
}

at::Tensor wrap_conv_depthwise3d(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation) {
  if (trace.is_flushing()) {
    ensure_materialized(self, weight);
    return at::redispatch::conv_depthwise3d(self, weight, kernel_size, bias, stride, padding, dilation);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_CONV_DEPTHWISE3D, self, weight, kernel_size, bias, stride, padding, dilation);
}

std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> wrap_conv_depthwise3d_backward_grad_input(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, at::Tensor & grad_input, at::Tensor & grad_weight, at::Tensor & grad_bias) {
  ensure_materialized(grad_input, grad_weight, grad_bias, grad_output, self, weight);
  return at::redispatch::conv_depthwise3d_backward(grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, dilation);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> wrap_conv_depthwise3d_backward_output_mask(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, std::array<bool,3> output_mask) {
  ensure_materialized(grad_output, self, weight);
  return at::redispatch::conv_depthwise3d_backward(grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
}

at::Tensor & wrap_slow_conv3d_out(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, weight);
    return at::redispatch::slow_conv3d(out, self, weight, kernel_size, bias, stride, padding);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SLOW_CONV3D_OUT, out, self, weight, kernel_size, bias, stride, padding);
}

at::Tensor wrap_slow_conv3d(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding) {
  if (trace.is_flushing()) {
    ensure_materialized(self, weight);
    return at::redispatch::slow_conv3d(self, weight, kernel_size, bias, stride, padding);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SLOW_CONV3D, self, weight, kernel_size, bias, stride, padding);
}

std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> wrap_slow_conv3d_forward_output(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::Tensor & output, at::Tensor & finput, at::Tensor & fgrad_input) {
  ensure_materialized(output, finput, fgrad_input, self, weight);
  return at::redispatch::slow_conv3d_forward(output, finput, fgrad_input, self, weight, kernel_size, bias, stride, padding);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> wrap_slow_conv3d_forward(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding) {
  ensure_materialized(self, weight);
  return at::redispatch::slow_conv3d_forward(self, weight, kernel_size, bias, stride, padding);
}

std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> wrap_slow_conv3d_backward_grad_input(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, const at::Tensor & finput, const at::Tensor & fgrad_input, at::Tensor & grad_input, at::Tensor & grad_weight, at::Tensor & grad_bias) {
  ensure_materialized(grad_input, grad_weight, grad_bias, grad_output, self, weight, finput, fgrad_input);
  return at::redispatch::slow_conv3d_backward(grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> wrap_slow_conv3d_backward_output_mask(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, const at::Tensor & finput, const at::Tensor & fgrad_input, std::array<bool,3> output_mask) {
  ensure_materialized(grad_output, self, weight, finput, fgrad_input);
  return at::redispatch::slow_conv3d_backward(grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input, output_mask);
}

at::Tensor wrap_slow_conv_dilated2d(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation) {
  if (trace.is_flushing()) {
    ensure_materialized(self, weight);
    return at::redispatch::slow_conv_dilated2d(self, weight, kernel_size, bias, stride, padding, dilation);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SLOW_CONV_DILATED2D, self, weight, kernel_size, bias, stride, padding, dilation);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> wrap_slow_conv_dilated2d_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, std::array<bool,3> output_mask) {
  ensure_materialized(grad_output, self, weight);
  return at::redispatch::slow_conv_dilated2d_backward(grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
}

at::Tensor wrap_slow_conv_dilated3d(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation) {
  if (trace.is_flushing()) {
    ensure_materialized(self, weight);
    return at::redispatch::slow_conv_dilated3d(self, weight, kernel_size, bias, stride, padding, dilation);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SLOW_CONV_DILATED3D, self, weight, kernel_size, bias, stride, padding, dilation);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> wrap_slow_conv_dilated3d_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, std::array<bool,3> output_mask) {
  ensure_materialized(grad_output, self, weight);
  return at::redispatch::slow_conv_dilated3d_backward(grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
}

at::Tensor & wrap_col2im_out(const at::Tensor & self, at::IntArrayRef output_size, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::col2im(out, self, output_size, kernel_size, dilation, padding, stride);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_COL2IM_OUT, out, self, output_size, kernel_size, dilation, padding, stride);
}

at::Tensor wrap_col2im(const at::Tensor & self, at::IntArrayRef output_size, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::col2im(self, output_size, kernel_size, dilation, padding, stride);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_COL2IM, self, output_size, kernel_size, dilation, padding, stride);
}

at::Tensor & wrap_col2im_backward_grad_input(const at::Tensor & grad_output, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output);
    return at::redispatch::col2im_backward(grad_input, grad_output, kernel_size, dilation, padding, stride);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_COL2IM_BACKWARD_GRAD_INPUT, grad_input, grad_output, kernel_size, dilation, padding, stride);
}

at::Tensor wrap_col2im_backward(const at::Tensor & grad_output, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output);
    return at::redispatch::col2im_backward(grad_output, kernel_size, dilation, padding, stride);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_COL2IM_BACKWARD, grad_output, kernel_size, dilation, padding, stride);
}

at::Tensor wrap_column_stack(at::TensorList tensors) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::column_stack(tensors));
}

at::Tensor & wrap_column_stack_out(at::TensorList tensors, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::column_stack(out, tensors);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_COLUMN_STACK_OUT, out, tensors);
}

at::Tensor & wrap_im2col_out(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::im2col(out, self, kernel_size, dilation, padding, stride);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_IM2COL_OUT, out, self, kernel_size, dilation, padding, stride);
}

at::Tensor wrap_im2col(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::im2col(self, kernel_size, dilation, padding, stride);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_IM2COL, self, kernel_size, dilation, padding, stride);
}

at::Tensor & wrap_im2col_backward_grad_input(const at::Tensor & grad_output, at::IntArrayRef input_size, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_input, grad_output);
    return at::redispatch::im2col_backward(grad_input, grad_output, input_size, kernel_size, dilation, padding, stride);
  }
  return MK_TORCHY(grad_input.dtype(), grad_input.device(), H_IM2COL_BACKWARD_GRAD_INPUT, grad_input, grad_output, input_size, kernel_size, dilation, padding, stride);
}

at::Tensor wrap_im2col_backward(const at::Tensor & grad_output, at::IntArrayRef input_size, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride) {
  if (trace.is_flushing()) {
    ensure_materialized(grad_output);
    return at::redispatch::im2col_backward(grad_output, input_size, kernel_size, dilation, padding, stride);
  }
  return MK_TORCHY(grad_output.dtype(), grad_output.device(), H_IM2COL_BACKWARD, grad_output, input_size, kernel_size, dilation, padding, stride);
}

at::Tensor wrap_isfinite(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::isfinite(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ISFINITE, self);
}

at::Tensor wrap_isinf(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::isinf(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ISINF, self);
}

void wrap_record_stream(at::Tensor & self, at::Stream s) {
  ensure_materialized(self);
  return at::redispatch::record_stream(self, s);
}

at::Tensor wrap_isposinf(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::isposinf(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ISPOSINF, self);
}

at::Tensor & wrap_isposinf_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::isposinf(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ISPOSINF_OUT, out, self);
}

at::Tensor wrap_isneginf(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::isneginf(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_ISNEGINF, self);
}

at::Tensor & wrap_isneginf_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::isneginf(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_ISNEGINF_OUT, out, self);
}

at::Tensor wrap__add_batch_dim(const at::Tensor & self, int64_t batch_dim, int64_t level) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_add_batch_dim(self, batch_dim, level);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__ADD_BATCH_DIM, self, batch_dim, level);
}

at::Tensor wrap__remove_batch_dim(const at::Tensor & self, int64_t level, int64_t batch_size, int64_t out_dim) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::_remove_batch_dim(self, level, batch_size, out_dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__REMOVE_BATCH_DIM, self, level, batch_size, out_dim);
}

at::Tensor & wrap_special_entr_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::special_entr(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SPECIAL_ENTR_OUT, out, self);
}

at::Tensor wrap_special_expm1(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::special_expm1(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SPECIAL_EXPM1, self);
}

at::Tensor & wrap_special_expm1_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::special_expm1(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SPECIAL_EXPM1_OUT, out, self);
}

at::Tensor wrap_special_exp2(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::special_exp2(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SPECIAL_EXP2, self);
}

at::Tensor & wrap_special_exp2_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::special_exp2(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SPECIAL_EXP2_OUT, out, self);
}

at::Tensor wrap_special_gammaln(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::special_gammaln(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SPECIAL_GAMMALN, self);
}

at::Tensor & wrap_special_gammaln_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::special_gammaln(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SPECIAL_GAMMALN_OUT, out, self);
}

at::Tensor wrap_special_erf(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::special_erf(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SPECIAL_ERF, self);
}

at::Tensor & wrap_special_erf_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::special_erf(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SPECIAL_ERF_OUT, out, self);
}

at::Tensor wrap_special_erfc(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::special_erfc(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SPECIAL_ERFC, self);
}

at::Tensor & wrap_special_erfc_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::special_erfc(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SPECIAL_ERFC_OUT, out, self);
}

at::Tensor wrap_special_erfinv(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::special_erfinv(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SPECIAL_ERFINV, self);
}

at::Tensor & wrap_special_erfinv_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::special_erfinv(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SPECIAL_ERFINV_OUT, out, self);
}

at::Tensor & wrap_special_i0e_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::special_i0e(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SPECIAL_I0E_OUT, out, self);
}

at::Tensor wrap_special_logit(const at::Tensor & self, c10::optional<double> eps) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::special_logit(self, eps);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SPECIAL_LOGIT, self, eps);
}

at::Tensor & wrap_special_logit_out(const at::Tensor & self, c10::optional<double> eps, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::special_logit(out, self, eps);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SPECIAL_LOGIT_OUT, out, self, eps);
}

at::Tensor wrap_special_expit(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::special_expit(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_SPECIAL_EXPIT, self);
}

at::Tensor & wrap_special_expit_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::special_expit(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_SPECIAL_EXPIT_OUT, out, self);
}

at::Tensor wrap_fft_fft(const at::Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<std::string> norm) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::fft_fft(self, n, dim, norm);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FFT_FFT, self, n, dim, norm);
}

at::Tensor & wrap_fft_fft_out(const at::Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<std::string> norm, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::fft_fft(out, self, n, dim, norm);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FFT_FFT_OUT, out, self, n, dim, norm);
}

at::Tensor wrap_fft_ifft(const at::Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<std::string> norm) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::fft_ifft(self, n, dim, norm);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FFT_IFFT, self, n, dim, norm);
}

at::Tensor & wrap_fft_ifft_out(const at::Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<std::string> norm, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::fft_ifft(out, self, n, dim, norm);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FFT_IFFT_OUT, out, self, n, dim, norm);
}

at::Tensor wrap_fft_rfft(const at::Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<std::string> norm) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::fft_rfft(self, n, dim, norm);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FFT_RFFT, self, n, dim, norm);
}

at::Tensor & wrap_fft_rfft_out(const at::Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<std::string> norm, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::fft_rfft(out, self, n, dim, norm);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FFT_RFFT_OUT, out, self, n, dim, norm);
}

at::Tensor wrap_fft_irfft(const at::Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<std::string> norm) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::fft_irfft(self, n, dim, norm);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FFT_IRFFT, self, n, dim, norm);
}

at::Tensor & wrap_fft_irfft_out(const at::Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<std::string> norm, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::fft_irfft(out, self, n, dim, norm);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FFT_IRFFT_OUT, out, self, n, dim, norm);
}

at::Tensor wrap_fft_hfft(const at::Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<std::string> norm) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::fft_hfft(self, n, dim, norm);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FFT_HFFT, self, n, dim, norm);
}

at::Tensor & wrap_fft_hfft_out(const at::Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<std::string> norm, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::fft_hfft(out, self, n, dim, norm);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FFT_HFFT_OUT, out, self, n, dim, norm);
}

at::Tensor wrap_fft_ihfft(const at::Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<std::string> norm) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::fft_ihfft(self, n, dim, norm);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FFT_IHFFT, self, n, dim, norm);
}

at::Tensor & wrap_fft_ihfft_out(const at::Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<std::string> norm, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::fft_ihfft(out, self, n, dim, norm);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FFT_IHFFT_OUT, out, self, n, dim, norm);
}

at::Tensor wrap_fft_fft2(const at::Tensor & self, c10::optional<at::IntArrayRef> s, at::IntArrayRef dim, c10::optional<std::string> norm) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::fft_fft2(self, s, dim, norm);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FFT_FFT2, self, s, dim, norm);
}

at::Tensor & wrap_fft_fft2_out(const at::Tensor & self, c10::optional<at::IntArrayRef> s, at::IntArrayRef dim, c10::optional<std::string> norm, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::fft_fft2(out, self, s, dim, norm);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FFT_FFT2_OUT, out, self, s, dim, norm);
}

at::Tensor wrap_fft_ifft2(const at::Tensor & self, c10::optional<at::IntArrayRef> s, at::IntArrayRef dim, c10::optional<std::string> norm) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::fft_ifft2(self, s, dim, norm);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FFT_IFFT2, self, s, dim, norm);
}

at::Tensor & wrap_fft_ifft2_out(const at::Tensor & self, c10::optional<at::IntArrayRef> s, at::IntArrayRef dim, c10::optional<std::string> norm, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::fft_ifft2(out, self, s, dim, norm);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FFT_IFFT2_OUT, out, self, s, dim, norm);
}

at::Tensor wrap_fft_rfft2(const at::Tensor & self, c10::optional<at::IntArrayRef> s, at::IntArrayRef dim, c10::optional<std::string> norm) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::fft_rfft2(self, s, dim, norm);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FFT_RFFT2, self, s, dim, norm);
}

at::Tensor & wrap_fft_rfft2_out(const at::Tensor & self, c10::optional<at::IntArrayRef> s, at::IntArrayRef dim, c10::optional<std::string> norm, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::fft_rfft2(out, self, s, dim, norm);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FFT_RFFT2_OUT, out, self, s, dim, norm);
}

at::Tensor wrap_fft_irfft2(const at::Tensor & self, c10::optional<at::IntArrayRef> s, at::IntArrayRef dim, c10::optional<std::string> norm) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::fft_irfft2(self, s, dim, norm);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FFT_IRFFT2, self, s, dim, norm);
}

at::Tensor & wrap_fft_irfft2_out(const at::Tensor & self, c10::optional<at::IntArrayRef> s, at::IntArrayRef dim, c10::optional<std::string> norm, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::fft_irfft2(out, self, s, dim, norm);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FFT_IRFFT2_OUT, out, self, s, dim, norm);
}

at::Tensor wrap_fft_fftn(const at::Tensor & self, c10::optional<at::IntArrayRef> s, c10::optional<at::IntArrayRef> dim, c10::optional<std::string> norm) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::fft_fftn(self, s, dim, norm);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FFT_FFTN, self, s, dim, norm);
}

at::Tensor & wrap_fft_fftn_out(const at::Tensor & self, c10::optional<at::IntArrayRef> s, c10::optional<at::IntArrayRef> dim, c10::optional<std::string> norm, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::fft_fftn(out, self, s, dim, norm);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FFT_FFTN_OUT, out, self, s, dim, norm);
}

at::Tensor wrap_fft_ifftn(const at::Tensor & self, c10::optional<at::IntArrayRef> s, c10::optional<at::IntArrayRef> dim, c10::optional<std::string> norm) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::fft_ifftn(self, s, dim, norm);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FFT_IFFTN, self, s, dim, norm);
}

at::Tensor & wrap_fft_ifftn_out(const at::Tensor & self, c10::optional<at::IntArrayRef> s, c10::optional<at::IntArrayRef> dim, c10::optional<std::string> norm, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::fft_ifftn(out, self, s, dim, norm);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FFT_IFFTN_OUT, out, self, s, dim, norm);
}

at::Tensor wrap_fft_rfftn(const at::Tensor & self, c10::optional<at::IntArrayRef> s, c10::optional<at::IntArrayRef> dim, c10::optional<std::string> norm) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::fft_rfftn(self, s, dim, norm);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FFT_RFFTN, self, s, dim, norm);
}

at::Tensor & wrap_fft_rfftn_out(const at::Tensor & self, c10::optional<at::IntArrayRef> s, c10::optional<at::IntArrayRef> dim, c10::optional<std::string> norm, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::fft_rfftn(out, self, s, dim, norm);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FFT_RFFTN_OUT, out, self, s, dim, norm);
}

at::Tensor wrap_fft_irfftn(const at::Tensor & self, c10::optional<at::IntArrayRef> s, c10::optional<at::IntArrayRef> dim, c10::optional<std::string> norm) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::fft_irfftn(self, s, dim, norm);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FFT_IRFFTN, self, s, dim, norm);
}

at::Tensor & wrap_fft_irfftn_out(const at::Tensor & self, c10::optional<at::IntArrayRef> s, c10::optional<at::IntArrayRef> dim, c10::optional<std::string> norm, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::fft_irfftn(out, self, s, dim, norm);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FFT_IRFFTN_OUT, out, self, s, dim, norm);
}

at::Tensor wrap_fft_fftfreq(int64_t n, double d, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::fft_fftfreq(n, d, dtype, layout, device, pin_memory));
}

at::Tensor & wrap_fft_fftfreq_out(int64_t n, double d, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::fft_fftfreq(out, n, d);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FFT_FFTFREQ_OUT, out, n, d);
}

at::Tensor wrap_fft_rfftfreq(int64_t n, double d, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::fft_rfftfreq(n, d, dtype, layout, device, pin_memory));
}

at::Tensor & wrap_fft_rfftfreq_out(int64_t n, double d, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::fft_rfftfreq(out, n, d);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_FFT_RFFTFREQ_OUT, out, n, d);
}

at::Tensor wrap_fft_fftshift(const at::Tensor & self, c10::optional<at::IntArrayRef> dim) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::fft_fftshift(self, dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FFT_FFTSHIFT, self, dim);
}

at::Tensor wrap_fft_ifftshift(const at::Tensor & self, c10::optional<at::IntArrayRef> dim) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::fft_ifftshift(self, dim);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_FFT_IFFTSHIFT, self, dim);
}

at::Tensor wrap_linalg_cholesky(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::linalg_cholesky(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LINALG_CHOLESKY, self);
}

at::Tensor & wrap_linalg_cholesky_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::linalg_cholesky(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LINALG_CHOLESKY_OUT, out, self);
}

at::Tensor wrap_linalg_det(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::linalg_det(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LINALG_DET, self);
}

at::Tensor & wrap_linalg_det_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::linalg_det(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LINALG_DET_OUT, out, self);
}

at::Tensor wrap_det(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::det(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_DET, self);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> wrap_linalg_lstsq(const at::Tensor & self, const at::Tensor & b, c10::optional<double> cond, c10::optional<std::string> driver) {
  ensure_materialized(self, b);
  return at::redispatch::linalg_lstsq(self, b, cond, driver);
}

std::tuple<at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &> wrap_linalg_lstsq_out(const at::Tensor & self, const at::Tensor & b, c10::optional<double> cond, c10::optional<std::string> driver, at::Tensor & solution, at::Tensor & residuals, at::Tensor & rank, at::Tensor & singular_values) {
  ensure_materialized(solution, residuals, rank, singular_values, self, b);
  return at::redispatch::linalg_lstsq(solution, residuals, rank, singular_values, self, b, cond, driver);
}

at::Tensor & wrap__lstsq_helper_(at::Tensor & self, at::Tensor & rank, at::Tensor & singular_values, at::Tensor & infos, const at::Tensor & a, double cond, std::string driver_name) {
  if (trace.is_flushing()) {
    ensure_materialized(self, rank, singular_values, infos, a);
    return at::redispatch::_lstsq_helper_(self, rank, singular_values, infos, a, cond, driver_name);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__LSTSQ_HELPER_, self, rank, singular_values, infos, a, cond, driver_name);
}

std::tuple<at::Tensor,at::Tensor> wrap_linalg_slogdet(const at::Tensor & self) {
  ensure_materialized(self);
  return at::redispatch::linalg_slogdet(self);
}

std::tuple<at::Tensor &,at::Tensor &> wrap_linalg_slogdet_out(const at::Tensor & self, at::Tensor & sign, at::Tensor & logabsdet) {
  ensure_materialized(sign, logabsdet, self);
  return at::redispatch::linalg_slogdet(sign, logabsdet, self);
}

std::tuple<at::Tensor,at::Tensor> wrap_linalg_eig(const at::Tensor & self) {
  ensure_materialized(self);
  return at::redispatch::linalg_eig(self);
}

std::tuple<at::Tensor &,at::Tensor &> wrap_linalg_eig_out(const at::Tensor & self, at::Tensor & eigenvalues, at::Tensor & eigenvectors) {
  ensure_materialized(eigenvalues, eigenvectors, self);
  return at::redispatch::linalg_eig(eigenvalues, eigenvectors, self);
}

at::Tensor wrap_linalg_eigvals(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::linalg_eigvals(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LINALG_EIGVALS, self);
}

at::Tensor & wrap_linalg_eigvals_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::linalg_eigvals(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LINALG_EIGVALS_OUT, out, self);
}

std::tuple<at::Tensor,at::Tensor> wrap_linalg_eigh(const at::Tensor & self, std::string UPLO) {
  ensure_materialized(self);
  return at::redispatch::linalg_eigh(self, UPLO);
}

std::tuple<at::Tensor &,at::Tensor &> wrap_linalg_eigh_eigvals(const at::Tensor & self, std::string UPLO, at::Tensor & eigvals, at::Tensor & eigvecs) {
  ensure_materialized(eigvals, eigvecs, self);
  return at::redispatch::linalg_eigh(eigvals, eigvecs, self, UPLO);
}

at::Tensor wrap_linalg_eigvalsh(const at::Tensor & self, std::string UPLO) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::linalg_eigvalsh(self, UPLO);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LINALG_EIGVALSH, self, UPLO);
}

at::Tensor & wrap_linalg_eigvalsh_out(const at::Tensor & self, std::string UPLO, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::linalg_eigvalsh(out, self, UPLO);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LINALG_EIGVALSH_OUT, out, self, UPLO);
}

at::Tensor wrap_linalg_householder_product(const at::Tensor & input, const at::Tensor & tau) {
  if (trace.is_flushing()) {
    ensure_materialized(input, tau);
    return at::redispatch::linalg_householder_product(input, tau);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_LINALG_HOUSEHOLDER_PRODUCT, input, tau);
}

at::Tensor & wrap_linalg_householder_product_out(const at::Tensor & input, const at::Tensor & tau, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, input, tau);
    return at::redispatch::linalg_householder_product(out, input, tau);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LINALG_HOUSEHOLDER_PRODUCT_OUT, out, input, tau);
}

at::Tensor & wrap__linalg_inv_out_helper_(at::Tensor & self, at::Tensor & infos_lu, at::Tensor & infos_getri) {
  if (trace.is_flushing()) {
    ensure_materialized(self, infos_lu, infos_getri);
    return at::redispatch::_linalg_inv_out_helper_(self, infos_lu, infos_getri);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__LINALG_INV_OUT_HELPER_, self, infos_lu, infos_getri);
}

at::Tensor wrap_linalg_inv(const at::Tensor & self) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::linalg_inv(self);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LINALG_INV, self);
}

at::Tensor & wrap_linalg_inv_out(const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::linalg_inv(out, self);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LINALG_INV_OUT, out, self);
}

at::Tensor wrap_inner(const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::inner(self, other);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_INNER, self, other);
}

at::Tensor & wrap_inner_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::inner(out, self, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_INNER_OUT, out, self, other);
}

at::Tensor wrap_outer(const at::Tensor & self, const at::Tensor & vec2) {
  if (trace.is_flushing()) {
    ensure_materialized(self, vec2);
    return at::redispatch::outer(self, vec2);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_OUTER, self, vec2);
}

at::Tensor & wrap_outer_out(const at::Tensor & self, const at::Tensor & vec2, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, vec2);
    return at::redispatch::outer(out, self, vec2);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_OUTER_OUT, out, self, vec2);
}

at::Tensor wrap_ger(const at::Tensor & self, const at::Tensor & vec2) {
  if (trace.is_flushing()) {
    ensure_materialized(self, vec2);
    return at::redispatch::ger(self, vec2);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_GER, self, vec2);
}

at::Tensor & wrap_ger_out(const at::Tensor & self, const at::Tensor & vec2, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, vec2);
    return at::redispatch::ger(out, self, vec2);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_GER_OUT, out, self, vec2);
}

at::Tensor wrap_linalg_norm(const at::Tensor & self, const c10::optional<at::Scalar> & ord, c10::optional<at::IntArrayRef> dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::linalg_norm(self, ord, dim, keepdim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LINALG_NORM, self, ord, dim, keepdim, dtype);
}

at::Tensor wrap_linalg_norm_ord_str(const at::Tensor & self, std::string ord, c10::optional<at::IntArrayRef> dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::linalg_norm(self, ord, dim, keepdim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LINALG_NORM_ORD_STR, self, ord, dim, keepdim, dtype);
}

at::Tensor & wrap_linalg_norm_out(const at::Tensor & self, const c10::optional<at::Scalar> & ord, c10::optional<at::IntArrayRef> dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::linalg_norm(out, self, ord, dim, keepdim, dtype);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LINALG_NORM_OUT, out, self, ord, dim, keepdim, dtype);
}

at::Tensor & wrap_linalg_norm_ord_str_out(const at::Tensor & self, std::string ord, c10::optional<at::IntArrayRef> dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::linalg_norm(out, self, ord, dim, keepdim, dtype);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LINALG_NORM_ORD_STR_OUT, out, self, ord, dim, keepdim, dtype);
}

at::Tensor wrap_linalg_vector_norm(const at::Tensor & self, const c10::optional<at::Scalar> & ord, c10::optional<at::IntArrayRef> dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::linalg_vector_norm(self, ord, dim, keepdim, dtype);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LINALG_VECTOR_NORM, self, ord, dim, keepdim, dtype);
}

at::Tensor & wrap_linalg_vector_norm_out(const at::Tensor & self, const c10::optional<at::Scalar> & ord, c10::optional<at::IntArrayRef> dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::linalg_vector_norm(out, self, ord, dim, keepdim, dtype);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LINALG_VECTOR_NORM_OUT, out, self, ord, dim, keepdim, dtype);
}

std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> wrap_linalg_svd_U(const at::Tensor & self, bool full_matrices, bool compute_uv, at::Tensor & U, at::Tensor & S, at::Tensor & V) {
  ensure_materialized(U, S, V, self);
  return at::redispatch::linalg_svd(U, S, V, self, full_matrices, compute_uv);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> wrap_linalg_svd(const at::Tensor & self, bool full_matrices, bool compute_uv) {
  ensure_materialized(self);
  return at::redispatch::linalg_svd(self, full_matrices, compute_uv);
}

at::Tensor wrap_linalg_cond(const at::Tensor & self, const c10::optional<at::Scalar> & p) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::linalg_cond(self, p);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LINALG_COND, self, p);
}

at::Tensor & wrap_linalg_cond_out(const at::Tensor & self, const c10::optional<at::Scalar> & p, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::linalg_cond(out, self, p);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LINALG_COND_OUT, out, self, p);
}

at::Tensor wrap_linalg_cond_p_str(const at::Tensor & self, std::string p) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::linalg_cond(self, p);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LINALG_COND_P_STR, self, p);
}

at::Tensor & wrap_linalg_cond_p_str_out(const at::Tensor & self, std::string p, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::linalg_cond(out, self, p);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LINALG_COND_P_STR_OUT, out, self, p);
}

at::Tensor wrap_linalg_pinv(const at::Tensor & self, double rcond, bool hermitian) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::linalg_pinv(self, rcond, hermitian);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LINALG_PINV, self, rcond, hermitian);
}

at::Tensor wrap_linalg_pinv_rcond_tensor(const at::Tensor & self, const at::Tensor & rcond, bool hermitian) {
  if (trace.is_flushing()) {
    ensure_materialized(self, rcond);
    return at::redispatch::linalg_pinv(self, rcond, hermitian);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LINALG_PINV_RCOND_TENSOR, self, rcond, hermitian);
}

at::Tensor & wrap_linalg_pinv_out(const at::Tensor & self, double rcond, bool hermitian, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::linalg_pinv(out, self, rcond, hermitian);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LINALG_PINV_OUT, out, self, rcond, hermitian);
}

at::Tensor & wrap_linalg_pinv_out_rcond_tensor(const at::Tensor & self, const at::Tensor & rcond, bool hermitian, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, rcond);
    return at::redispatch::linalg_pinv(out, self, rcond, hermitian);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LINALG_PINV_OUT_RCOND_TENSOR, out, self, rcond, hermitian);
}

at::Tensor & wrap__linalg_solve_out_helper_(at::Tensor & self, at::Tensor & other, at::Tensor & infos) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other, infos);
    return at::redispatch::_linalg_solve_out_helper_(self, other, infos);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__LINALG_SOLVE_OUT_HELPER_, self, other, infos);
}

at::Tensor wrap_linalg_solve(const at::Tensor & input, const at::Tensor & other) {
  if (trace.is_flushing()) {
    ensure_materialized(input, other);
    return at::redispatch::linalg_solve(input, other);
  }
  return MK_TORCHY(input.dtype(), input.device(), H_LINALG_SOLVE, input, other);
}

at::Tensor & wrap_linalg_solve_out(const at::Tensor & input, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, input, other);
    return at::redispatch::linalg_solve(out, input, other);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LINALG_SOLVE_OUT, out, input, other);
}

at::Tensor wrap_linalg_tensorinv(const at::Tensor & self, int64_t ind) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::linalg_tensorinv(self, ind);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LINALG_TENSORINV, self, ind);
}

at::Tensor & wrap_linalg_tensorinv_out(const at::Tensor & self, int64_t ind, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::linalg_tensorinv(out, self, ind);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LINALG_TENSORINV_OUT, out, self, ind);
}

at::Tensor wrap_linalg_tensorsolve(const at::Tensor & self, const at::Tensor & other, c10::optional<at::IntArrayRef> dims) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::linalg_tensorsolve(self, other, dims);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LINALG_TENSORSOLVE, self, other, dims);
}

at::Tensor & wrap_linalg_tensorsolve_out(const at::Tensor & self, const at::Tensor & other, c10::optional<at::IntArrayRef> dims, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self, other);
    return at::redispatch::linalg_tensorsolve(out, self, other, dims);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LINALG_TENSORSOLVE_OUT, out, self, other, dims);
}

std::tuple<at::Tensor,at::Tensor> wrap_linalg_qr(const at::Tensor & self, std::string mode) {
  ensure_materialized(self);
  return at::redispatch::linalg_qr(self, mode);
}

std::tuple<at::Tensor &,at::Tensor &> wrap_linalg_qr_out(const at::Tensor & self, std::string mode, at::Tensor & Q, at::Tensor & R) {
  ensure_materialized(Q, R, self);
  return at::redispatch::linalg_qr(Q, R, self, mode);
}

std::tuple<at::Tensor,at::Tensor> wrap__linalg_qr_helper(const at::Tensor & self, std::string mode) {
  ensure_materialized(self);
  return at::redispatch::_linalg_qr_helper(self, mode);
}

at::Tensor wrap_linalg_matrix_power(const at::Tensor & self, int64_t n) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::linalg_matrix_power(self, n);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LINALG_MATRIX_POWER, self, n);
}

at::Tensor & wrap_linalg_matrix_power_out(const at::Tensor & self, int64_t n, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::linalg_matrix_power(out, self, n);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LINALG_MATRIX_POWER_OUT, out, self, n);
}

at::Tensor wrap_linalg_matrix_rank(const at::Tensor & self, c10::optional<double> tol, bool hermitian) {
  if (trace.is_flushing()) {
    ensure_materialized(self);
    return at::redispatch::linalg_matrix_rank(self, tol, hermitian);
  }
  return MK_TORCHY(self.dtype(), self.device(), H_LINALG_MATRIX_RANK, self, tol, hermitian);
}

at::Tensor & wrap_linalg_matrix_rank_out(const at::Tensor & self, c10::optional<double> tol, bool hermitian, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out, self);
    return at::redispatch::linalg_matrix_rank(out, self, tol, hermitian);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LINALG_MATRIX_RANK_OUT, out, self, tol, hermitian);
}

at::Tensor wrap_linalg_multi_dot(at::TensorList tensors) {
  ensure_materialized();
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::linalg_multi_dot(tensors));
}

at::Tensor & wrap_linalg_multi_dot_out(at::TensorList tensors, at::Tensor & out) {
  if (trace.is_flushing()) {
    ensure_materialized(out);
    return at::redispatch::linalg_multi_dot(out, tensors);
  }
  return MK_TORCHY(out.dtype(), out.device(), H_LINALG_MULTI_DOT_OUT, out, tensors);
}

at::Tensor wrap__test_serialization_subcmul(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    ensure_materialized(self, other);
    return at::redispatch::_test_serialization_subcmul(self, other, alpha);
  }
  return MK_TORCHY(self.dtype(), self.device(), H__TEST_SERIALIZATION_SUBCMUL, self, other, alpha);
}

at::Tensor wrap__test_optional_intlist(const at::Tensor & values, c10::optional<at::IntArrayRef> addends) {
  if (trace.is_flushing()) {
    ensure_materialized(values);
    return at::redispatch::_test_optional_intlist(values, addends);
  }
  return MK_TORCHY(values.dtype(), values.device(), H__TEST_OPTIONAL_INTLIST, values, addends);
}

at::Tensor wrap__test_optional_filled_intlist(const at::Tensor & values, c10::optional<at::IntArrayRef> addends) {
  if (trace.is_flushing()) {
    ensure_materialized(values);
    return at::redispatch::_test_optional_filled_intlist(values, addends);
  }
  return MK_TORCHY(values.dtype(), values.device(), H__TEST_OPTIONAL_FILLED_INTLIST, values, addends);
}

at::Tensor wrap__test_optional_floatlist(const at::Tensor & values, c10::optional<at::ArrayRef<double>> addends) {
  if (trace.is_flushing()) {
    ensure_materialized(values);
    return at::redispatch::_test_optional_floatlist(values, addends);
  }
  return MK_TORCHY(values.dtype(), values.device(), H__TEST_OPTIONAL_FLOATLIST, values, addends);
}

at::Tensor wrap__test_string_default(const at::Tensor & dummy, std::string a, std::string b) {
  if (trace.is_flushing()) {
    ensure_materialized(dummy);
    return at::redispatch::_test_string_default(dummy, a, b);
  }
  return MK_TORCHY(dummy.dtype(), dummy.device(), H__TEST_STRING_DEFAULT, dummy, a, b);
}

at::Tensor wrap__test_ambiguous_defaults_a(const at::Tensor & dummy, int64_t a, int64_t b) {
  if (trace.is_flushing()) {
    ensure_materialized(dummy);
    return at::redispatch::_test_ambiguous_defaults(dummy, a, b);
  }
  return MK_TORCHY(dummy.dtype(), dummy.device(), H__TEST_AMBIGUOUS_DEFAULTS_A, dummy, a, b);
}

at::Tensor wrap__test_ambiguous_defaults_b(const at::Tensor & dummy, int64_t a, std::string b) {
  if (trace.is_flushing()) {
    ensure_materialized(dummy);
    return at::redispatch::_test_ambiguous_defaults(dummy, a, b);
  }
  return MK_TORCHY(dummy.dtype(), dummy.device(), H__TEST_AMBIGUOUS_DEFAULTS_B, dummy, a, b);
}

at::Tensor wrap_segment_reduce(const at::Tensor & data, std::string reduce, const c10::optional<at::Tensor> & lengths, const c10::optional<at::Tensor> & indices, int64_t axis, bool unsafe) {
  if (trace.is_flushing()) {
    ensure_materialized(data);
    return at::redispatch::segment_reduce(data, reduce, lengths, indices, axis, unsafe);
  }
  return MK_TORCHY(data.dtype(), data.device(), H_SEGMENT_REDUCE, data, reduce, lengths, indices, axis, unsafe);
}
