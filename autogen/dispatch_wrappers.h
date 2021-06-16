
at::Tensor wrap__cast_Byte(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool non_blocking) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_cast_Byte(dispatchKeySet, self, non_blocking);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__CAST_BYTE, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(non_blocking);
  return tt;
}

at::Tensor wrap__cast_Char(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool non_blocking) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_cast_Char(dispatchKeySet, self, non_blocking);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__CAST_CHAR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(non_blocking);
  return tt;
}

at::Tensor wrap__cast_Double(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool non_blocking) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_cast_Double(dispatchKeySet, self, non_blocking);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__CAST_DOUBLE, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(non_blocking);
  return tt;
}

at::Tensor wrap__cast_Float(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool non_blocking) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_cast_Float(dispatchKeySet, self, non_blocking);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__CAST_FLOAT, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(non_blocking);
  return tt;
}

at::Tensor wrap__cast_Int(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool non_blocking) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_cast_Int(dispatchKeySet, self, non_blocking);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__CAST_INT, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(non_blocking);
  return tt;
}

at::Tensor wrap__cast_Long(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool non_blocking) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_cast_Long(dispatchKeySet, self, non_blocking);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__CAST_LONG, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(non_blocking);
  return tt;
}

at::Tensor wrap__cast_Short(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool non_blocking) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_cast_Short(dispatchKeySet, self, non_blocking);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__CAST_SHORT, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(non_blocking);
  return tt;
}

at::Tensor wrap__cast_Half(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool non_blocking) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_cast_Half(dispatchKeySet, self, non_blocking);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__CAST_HALF, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(non_blocking);
  return tt;
}

at::Tensor wrap_data(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::__dispatch_data(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_DATA, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_requires_grad_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, bool requires_grad) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::__dispatch_requires_grad_(dispatchKeySet, self, requires_grad);
  }
  bool flush = register_in_place(self, H_REQUIRES_GRAD_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(requires_grad);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap__fw_primal(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t level) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_fw_primal(dispatchKeySet, self, level);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__FW_PRIMAL, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(level);
  return tt;
}

at::Tensor wrap__make_dual(c10::DispatchKeySet dispatchKeySet, const at::Tensor & primal, const at::Tensor & tangent, int64_t level) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_make_dual(dispatchKeySet, primal, tangent, level);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__MAKE_DUAL, primal.dtype(), primal.device());
  trace.append_arg(primal);trace.append_arg(tangent);trace.append_arg(level);
  return tt;
}

at::Tensor & wrap_rename_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, c10::optional<at::DimnameList> names) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::rename_(dispatchKeySet, self, std::move(names));
  }
  bool flush = register_in_place(self, H_RENAME_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(names));
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_rename(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::DimnameList> names) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::rename(dispatchKeySet, self, std::move(names));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_RENAME, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(names));
  return tt;
}

at::Tensor wrap_align_to(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::DimnameList names) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::align_to(dispatchKeySet, self, std::move(names));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ALIGN_TO, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(names));
  return tt;
}

at::Tensor wrap_align_to_ellipsis_idx(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::DimnameList order, int64_t ellipsis_idx) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::align_to(dispatchKeySet, self, std::move(order), ellipsis_idx);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ALIGN_TO_ELLIPSIS_IDX, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(order));trace.append_arg(ellipsis_idx);
  return tt;
}

at::Tensor wrap_align_as(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::align_as(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ALIGN_AS, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor wrap_refine_names(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::DimnameList names) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::refine_names(dispatchKeySet, self, std::move(names));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_REFINE_NAMES, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(names));
  return tt;
}

at::Tensor wrap__cudnn_rnn_flatten_weight(c10::DispatchKeySet dispatchKeySet, at::TensorList weight_arr, int64_t weight_stride0, int64_t input_size, int64_t mode, int64_t hidden_size, int64_t proj_size, int64_t num_layers, bool batch_first, bool bidirectional) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_cudnn_rnn_flatten_weight(dispatchKeySet, std::move(weight_arr), weight_stride0, input_size, mode, hidden_size, proj_size, num_layers, batch_first, bidirectional);
  }
  auto defaults = compute_dtype(weight_arr);
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H__CUDNN_RNN_FLATTEN_WEIGHT, default_dtype, default_device);
  trace.append_arg(std::move(weight_arr));trace.append_arg(weight_stride0);trace.append_arg(input_size);trace.append_arg(mode);trace.append_arg(hidden_size);trace.append_arg(proj_size);trace.append_arg(num_layers);trace.append_arg(batch_first);trace.append_arg(bidirectional);
  return tt;
}

at::Tensor wrap__cudnn_init_dropout_state(c10::DispatchKeySet dispatchKeySet, double dropout, bool train, int64_t dropout_seed, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_cudnn_init_dropout_state(dispatchKeySet, dropout, train, dropout_seed, std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H__CUDNN_INIT_DROPOUT_STATE, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(dropout);trace.append_arg(train);trace.append_arg(dropout_seed);trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor wrap__masked_scale(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & mask, double scale) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_masked_scale(dispatchKeySet, self, mask, scale);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__MASKED_SCALE, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(mask);trace.append_arg(scale);
  return tt;
}

at::Tensor & wrap__sobol_engine_ff_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, int64_t n, const at::Tensor & sobolstate, int64_t dimension, int64_t num_generated) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_sobol_engine_ff_(dispatchKeySet, self, n, sobolstate, dimension, num_generated);
  }
  bool flush = register_in_place(self, H__SOBOL_ENGINE_FF_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(n);trace.append_arg(sobolstate);trace.append_arg(dimension);trace.append_arg(num_generated);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap__sobol_engine_scramble_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & ltm, int64_t dimension) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_sobol_engine_scramble_(dispatchKeySet, self, ltm, dimension);
  }
  bool flush = register_in_place(self, H__SOBOL_ENGINE_SCRAMBLE_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(ltm);trace.append_arg(dimension);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap__sobol_engine_initialize_state_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, int64_t dimension) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_sobol_engine_initialize_state_(dispatchKeySet, self, dimension);
  }
  bool flush = register_in_place(self, H__SOBOL_ENGINE_INITIALIZE_STATE_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(dimension);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap__reshape_from_tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & shape) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_reshape_from_tensor(dispatchKeySet, self, shape);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__RESHAPE_FROM_TENSOR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(shape);
  return tt;
}

at::Tensor wrap__shape_as_tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_shape_as_tensor(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__SHAPE_AS_TENSOR, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor wrap_dropout(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, double p, bool train) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::dropout(dispatchKeySet, input, p, train);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_DROPOUT, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(p);trace.append_arg(train);
  return tt;
}

at::Tensor & wrap_dropout_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, double p, bool train) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::dropout_(dispatchKeySet, self, p, train);
  }
  bool flush = register_in_place(self, H_DROPOUT_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(p);trace.append_arg(train);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_feature_dropout(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, double p, bool train) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::feature_dropout(dispatchKeySet, input, p, train);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_FEATURE_DROPOUT, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(p);trace.append_arg(train);
  return tt;
}

at::Tensor & wrap_feature_dropout_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, double p, bool train) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::feature_dropout_(dispatchKeySet, self, p, train);
  }
  bool flush = register_in_place(self, H_FEATURE_DROPOUT_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(p);trace.append_arg(train);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_alpha_dropout(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, double p, bool train) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::alpha_dropout(dispatchKeySet, input, p, train);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ALPHA_DROPOUT, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(p);trace.append_arg(train);
  return tt;
}

at::Tensor & wrap_alpha_dropout_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, double p, bool train) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::alpha_dropout_(dispatchKeySet, self, p, train);
  }
  bool flush = register_in_place(self, H_ALPHA_DROPOUT_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(p);trace.append_arg(train);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_feature_alpha_dropout(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, double p, bool train) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::feature_alpha_dropout(dispatchKeySet, input, p, train);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_FEATURE_ALPHA_DROPOUT, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(p);trace.append_arg(train);
  return tt;
}

at::Tensor & wrap_feature_alpha_dropout_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, double p, bool train) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::feature_alpha_dropout_(dispatchKeySet, self, p, train);
  }
  bool flush = register_in_place(self, H_FEATURE_ALPHA_DROPOUT_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(p);trace.append_arg(train);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_abs(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::abs(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ABS, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_abs_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::abs_(dispatchKeySet, self);
  }
  bool flush = register_in_place(self, H_ABS_, dispatchKeySet);
  trace.append_arg(self);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_abs_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::abs_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_ABS_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_absolute(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::absolute(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ABSOLUTE, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_absolute_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::absolute_(dispatchKeySet, self);
  }
  bool flush = register_in_place(self, H_ABSOLUTE_, dispatchKeySet);
  trace.append_arg(self);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_absolute_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::absolute_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_ABSOLUTE_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_angle(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::angle(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ANGLE, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_angle_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::angle_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_ANGLE_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_view_as_real(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::view_as_real(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_VIEW_AS_REAL, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor wrap__view_as_real_physical(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_view_as_real_physical(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__VIEW_AS_REAL_PHYSICAL, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor wrap_view_as_complex(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::view_as_complex(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_VIEW_AS_COMPLEX, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_sgn_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::sgn_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_SGN_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_real(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::real(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_REAL, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor wrap_imag(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::imag(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_IMAG, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor wrap__conj(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_conj(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__CONJ, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor wrap_conj(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::__dispatch_conj(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CONJ, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor wrap__conj_physical(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_conj_physical(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__CONJ_PHYSICAL, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor wrap_conj_physical(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::conj_physical(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CONJ_PHYSICAL, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_conj_physical_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::conj_physical_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_CONJ_PHYSICAL_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_conj_physical_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::conj_physical_(dispatchKeySet, self);
  }
  bool flush = register_in_place(self, H_CONJ_PHYSICAL_, dispatchKeySet);
  trace.append_arg(self);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_resolve_conj(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::resolve_conj(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_RESOLVE_CONJ, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_acos_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::acos_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_ACOS_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_arccos(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::arccos(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ARCCOS, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_arccos_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::arccos_(dispatchKeySet, self);
  }
  bool flush = register_in_place(self, H_ARCCOS_, dispatchKeySet);
  trace.append_arg(self);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_arccos_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::arccos_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_ARCCOS_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_avg_pool1d(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::avg_pool1d(dispatchKeySet, self, std::move(kernel_size), std::move(stride), std::move(padding), ceil_mode, count_include_pad);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_AVG_POOL1D, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(kernel_size));trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(ceil_mode);trace.append_arg(count_include_pad);
  return tt;
}

at::Tensor wrap_adaptive_avg_pool1d(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::adaptive_avg_pool1d(dispatchKeySet, self, std::move(output_size));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ADAPTIVE_AVG_POOL1D, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(output_size));
  return tt;
}

at::Tensor wrap_add_Tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::add(dispatchKeySet, self, other, alpha);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ADD_TENSOR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(alpha);
  return tt;
}

at::Tensor & wrap_add__Tensor(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::add_(dispatchKeySet, self, other, alpha);
  }
  bool flush = register_in_place(self, H_ADD__TENSOR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(alpha);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_add_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::add_outf(dispatchKeySet, self, other, alpha, out);
  }
  bool flush = register_in_place(out, H_ADD_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(alpha);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap__add_relu_Tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_add_relu(dispatchKeySet, self, other, alpha);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__ADD_RELU_TENSOR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(alpha);
  return tt;
}

at::Tensor & wrap__add_relu__Tensor(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_add_relu_(dispatchKeySet, self, other, alpha);
  }
  bool flush = register_in_place(self, H__ADD_RELU__TENSOR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(alpha);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap__add_relu_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_add_relu_outf(dispatchKeySet, self, other, alpha, out);
  }
  bool flush = register_in_place(out, H__ADD_RELU_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(alpha);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_add_Scalar(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::add(dispatchKeySet, self, other, alpha);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ADD_SCALAR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(alpha);
  return tt;
}

at::Tensor & wrap_add__Scalar(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::add_(dispatchKeySet, self, other, alpha);
  }
  bool flush = register_in_place(self, H_ADD__SCALAR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(alpha);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_addmv_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & mat, const at::Tensor & vec, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::addmv_outf(dispatchKeySet, self, mat, vec, beta, alpha, out);
  }
  bool flush = register_in_place(out, H_ADDMV_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(mat);trace.append_arg(vec);trace.append_arg(beta);trace.append_arg(alpha);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_addr(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & vec1, const at::Tensor & vec2, const at::Scalar & beta, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::addr(dispatchKeySet, self, vec1, vec2, beta, alpha);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ADDR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(vec1);trace.append_arg(vec2);trace.append_arg(beta);trace.append_arg(alpha);
  return tt;
}

at::Tensor & wrap_addr_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & vec1, const at::Tensor & vec2, const at::Scalar & beta, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::addr_(dispatchKeySet, self, vec1, vec2, beta, alpha);
  }
  bool flush = register_in_place(self, H_ADDR_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(vec1);trace.append_arg(vec2);trace.append_arg(beta);trace.append_arg(alpha);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_addr_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & vec1, const at::Tensor & vec2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::addr_outf(dispatchKeySet, self, vec1, vec2, beta, alpha, out);
  }
  bool flush = register_in_place(out, H_ADDR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(vec1);trace.append_arg(vec2);trace.append_arg(beta);trace.append_arg(alpha);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_affine_grid_generator(c10::DispatchKeySet dispatchKeySet, const at::Tensor & theta, at::IntArrayRef size, bool align_corners) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::affine_grid_generator(dispatchKeySet, theta, std::move(size), align_corners);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_AFFINE_GRID_GENERATOR, theta.dtype(), theta.device());
  trace.append_arg(theta);trace.append_arg(std::move(size));trace.append_arg(align_corners);
  return tt;
}

at::Tensor wrap_affine_grid_generator_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, at::IntArrayRef size, bool align_corners) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::affine_grid_generator_backward(dispatchKeySet, grad, std::move(size), align_corners);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_AFFINE_GRID_GENERATOR_BACKWARD, grad.dtype(), grad.device());
  trace.append_arg(grad);trace.append_arg(std::move(size));trace.append_arg(align_corners);
  return tt;
}

at::Tensor wrap_all_dim(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, bool keepdim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::all(dispatchKeySet, self, dim, keepdim);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ALL_DIM, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(keepdim);
  return tt;
}

at::Tensor & wrap_all_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::all_outf(dispatchKeySet, self, dim, keepdim, out);
  }
  bool flush = register_in_place(out, H_ALL_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(keepdim);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_all_dimname(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, bool keepdim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::all(dispatchKeySet, self, std::move(dim), keepdim);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ALL_DIMNAME, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(keepdim);
  return tt;
}

at::Tensor & wrap_all_dimname_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, bool keepdim, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::all_outf(dispatchKeySet, self, std::move(dim), keepdim, out);
  }
  bool flush = register_in_place(out, H_ALL_DIMNAME_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(keepdim);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_any_dim(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, bool keepdim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::any(dispatchKeySet, self, dim, keepdim);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ANY_DIM, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(keepdim);
  return tt;
}

at::Tensor & wrap_any_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::any_outf(dispatchKeySet, self, dim, keepdim, out);
  }
  bool flush = register_in_place(out, H_ANY_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(keepdim);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_any_dimname(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, bool keepdim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::any(dispatchKeySet, self, std::move(dim), keepdim);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ANY_DIMNAME, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(keepdim);
  return tt;
}

at::Tensor & wrap_any_dimname_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, bool keepdim, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::any_outf(dispatchKeySet, self, std::move(dim), keepdim, out);
  }
  bool flush = register_in_place(out, H_ANY_DIMNAME_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(keepdim);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_arange(c10::DispatchKeySet dispatchKeySet, const at::Scalar & end, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::arange(dispatchKeySet, end, std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_ARANGE, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(end);trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor wrap_arange_start(c10::DispatchKeySet dispatchKeySet, const at::Scalar & start, const at::Scalar & end, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::arange(dispatchKeySet, start, end, std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_ARANGE_START, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(start);trace.append_arg(end);trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor wrap_arange_start_step(c10::DispatchKeySet dispatchKeySet, const at::Scalar & start, const at::Scalar & end, const at::Scalar & step, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::arange(dispatchKeySet, start, end, step, std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_ARANGE_START_STEP, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(start);trace.append_arg(end);trace.append_arg(step);trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor & wrap_arange_out(c10::DispatchKeySet dispatchKeySet, const at::Scalar & end, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::arange_outf(dispatchKeySet, end, out);
  }
  bool flush = register_in_place(out, H_ARANGE_OUT, dispatchKeySet);
  trace.append_arg(end);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_arange_start_out(c10::DispatchKeySet dispatchKeySet, const at::Scalar & start, const at::Scalar & end, const at::Scalar & step, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::arange_outf(dispatchKeySet, start, end, step, out);
  }
  bool flush = register_in_place(out, H_ARANGE_START_OUT, dispatchKeySet);
  trace.append_arg(start);trace.append_arg(end);trace.append_arg(step);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap__dim_arange(c10::DispatchKeySet dispatchKeySet, const at::Tensor & like, int64_t dim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_dim_arange(dispatchKeySet, like, dim);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__DIM_ARANGE, like.dtype(), like.device());
  trace.append_arg(like);trace.append_arg(dim);
  return tt;
}

at::Tensor wrap_argmax(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::argmax(dispatchKeySet, self, dim, keepdim);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ARGMAX, scalarTypeToTypeMeta(kLong), self.device());
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(keepdim);
  return tt;
}

at::Tensor & wrap_argmax_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::argmax_outf(dispatchKeySet, self, dim, keepdim, out);
  }
  bool flush = register_in_place(out, H_ARGMAX_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(keepdim);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_argmin(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::argmin(dispatchKeySet, self, dim, keepdim);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ARGMIN, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(keepdim);
  return tt;
}

at::Tensor & wrap_argmin_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::argmin_outf(dispatchKeySet, self, dim, keepdim, out);
  }
  bool flush = register_in_place(out, H_ARGMIN_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(keepdim);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_acosh_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::acosh_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_ACOSH_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_arccosh(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::arccosh(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ARCCOSH, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_arccosh_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::arccosh_(dispatchKeySet, self);
  }
  bool flush = register_in_place(self, H_ARCCOSH_, dispatchKeySet);
  trace.append_arg(self);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_arccosh_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::arccosh_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_ARCCOSH_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_asinh_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::asinh_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_ASINH_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_arcsinh(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::arcsinh(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ARCSINH, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_arcsinh_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::arcsinh_(dispatchKeySet, self);
  }
  bool flush = register_in_place(self, H_ARCSINH_, dispatchKeySet);
  trace.append_arg(self);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_arcsinh_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::arcsinh_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_ARCSINH_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_atanh_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::atanh_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_ATANH_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_arctanh(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::arctanh(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ARCTANH, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_arctanh_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::arctanh_(dispatchKeySet, self);
  }
  bool flush = register_in_place(self, H_ARCTANH_, dispatchKeySet);
  trace.append_arg(self);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_arctanh_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::arctanh_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_ARCTANH_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_as_strided(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef size, at::IntArrayRef stride, c10::optional<int64_t> storage_offset) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::as_strided(dispatchKeySet, self, std::move(size), std::move(stride), storage_offset);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_AS_STRIDED, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(size));trace.append_arg(std::move(stride));trace.append_arg(storage_offset);
  return tt;
}

const at::Tensor & wrap_as_strided_(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef size, at::IntArrayRef stride, c10::optional<int64_t> storage_offset) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::as_strided_(dispatchKeySet, self, std::move(size), std::move(stride), storage_offset);
  }
  bool flush = register_in_place(self, H_AS_STRIDED_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(size));trace.append_arg(std::move(stride));trace.append_arg(storage_offset);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_asin(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::asin(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ASIN, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_asin_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::asin_(dispatchKeySet, self);
  }
  bool flush = register_in_place(self, H_ASIN_, dispatchKeySet);
  trace.append_arg(self);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_asin_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::asin_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_ASIN_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_arcsin(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::arcsin(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ARCSIN, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_arcsin_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::arcsin_(dispatchKeySet, self);
  }
  bool flush = register_in_place(self, H_ARCSIN_, dispatchKeySet);
  trace.append_arg(self);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_arcsin_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::arcsin_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_ARCSIN_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_atan_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::atan_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_ATAN_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_arctan(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::arctan(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ARCTAN, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_arctan_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::arctan_(dispatchKeySet, self);
  }
  bool flush = register_in_place(self, H_ARCTAN_, dispatchKeySet);
  trace.append_arg(self);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_arctan_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::arctan_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_ARCTAN_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_atleast_1d(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::atleast_1d(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ATLEAST_1D, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor wrap_atleast_2d(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::atleast_2d(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ATLEAST_2D, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor wrap_atleast_3d(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::atleast_3d(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ATLEAST_3D, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor wrap_baddbmm(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::baddbmm(dispatchKeySet, self, batch1, batch2, beta, alpha);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_BADDBMM, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(batch1);trace.append_arg(batch2);trace.append_arg(beta);trace.append_arg(alpha);
  return tt;
}

at::Tensor & wrap_baddbmm_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::baddbmm_(dispatchKeySet, self, batch1, batch2, beta, alpha);
  }
  bool flush = register_in_place(self, H_BADDBMM_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(batch1);trace.append_arg(batch2);trace.append_arg(beta);trace.append_arg(alpha);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap__baddbmm_mkl_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_baddbmm_mkl_(dispatchKeySet, self, batch1, batch2, beta, alpha);
  }
  bool flush = register_in_place(self, H__BADDBMM_MKL_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(batch1);trace.append_arg(batch2);trace.append_arg(beta);trace.append_arg(alpha);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_baddbmm_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::baddbmm_outf(dispatchKeySet, self, batch1, batch2, beta, alpha, out);
  }
  bool flush = register_in_place(out, H_BADDBMM_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(batch1);trace.append_arg(batch2);trace.append_arg(beta);trace.append_arg(alpha);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_bartlett_window(c10::DispatchKeySet dispatchKeySet, int64_t window_length, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::bartlett_window(dispatchKeySet, window_length, std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_BARTLETT_WINDOW, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(window_length);trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor wrap_bartlett_window_periodic(c10::DispatchKeySet dispatchKeySet, int64_t window_length, bool periodic, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::bartlett_window(dispatchKeySet, window_length, periodic, std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_BARTLETT_WINDOW_PERIODIC, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(window_length);trace.append_arg(periodic);trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor wrap_batch_norm(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, bool training, double momentum, double eps, bool cudnn_enabled) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::batch_norm(dispatchKeySet, input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_BATCH_NORM, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(weight);trace.append_arg(bias);trace.append_arg(running_mean);trace.append_arg(running_var);trace.append_arg(training);trace.append_arg(momentum);trace.append_arg(eps);trace.append_arg(cudnn_enabled);
  return tt;
}

at::Tensor wrap_quantized_batch_norm(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const at::Tensor & mean, const at::Tensor & var, double eps, double output_scale, int64_t output_zero_point) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::quantized_batch_norm(dispatchKeySet, input, weight, bias, mean, var, eps, output_scale, output_zero_point);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_QUANTIZED_BATCH_NORM, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(weight);trace.append_arg(bias);trace.append_arg(mean);trace.append_arg(var);trace.append_arg(eps);trace.append_arg(output_scale);trace.append_arg(output_zero_point);
  return tt;
}

at::Tensor wrap_bernoulli(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::Generator> generator) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::bernoulli(dispatchKeySet, self, std::move(generator));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_BERNOULLI, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(generator));
  return tt;
}

at::Tensor & wrap_bernoulli_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::Generator> generator, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::bernoulli_outf(dispatchKeySet, self, std::move(generator), out);
  }
  bool flush = register_in_place(out, H_BERNOULLI_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(generator));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_bernoulli__Tensor(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & p, c10::optional<at::Generator> generator) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::bernoulli_(dispatchKeySet, self, p, std::move(generator));
  }
  bool flush = register_in_place(self, H_BERNOULLI__TENSOR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(p);trace.append_arg(std::move(generator));
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_bernoulli__float(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, double p, c10::optional<at::Generator> generator) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::bernoulli_(dispatchKeySet, self, p, std::move(generator));
  }
  bool flush = register_in_place(self, H_BERNOULLI__FLOAT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(p);trace.append_arg(std::move(generator));
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_bernoulli_p(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, double p, c10::optional<at::Generator> generator) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::bernoulli(dispatchKeySet, self, p, std::move(generator));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_BERNOULLI_P, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(p);trace.append_arg(std::move(generator));
  return tt;
}

at::Tensor wrap_bilinear(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input1, const at::Tensor & input2, const at::Tensor & weight, const c10::optional<at::Tensor> & bias) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::bilinear(dispatchKeySet, input1, input2, weight, bias);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_BILINEAR, input1.dtype(), input1.device());
  trace.append_arg(input1);trace.append_arg(input2);trace.append_arg(weight);trace.append_arg(bias);
  return tt;
}

at::Tensor wrap_binary_cross_entropy(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::binary_cross_entropy(dispatchKeySet, self, target, weight, reduction);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_BINARY_CROSS_ENTROPY, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(target);trace.append_arg(weight);trace.append_arg(reduction);
  return tt;
}

at::Tensor & wrap_binary_cross_entropy_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::binary_cross_entropy_outf(dispatchKeySet, self, target, weight, reduction, out);
  }
  bool flush = register_in_place(out, H_BINARY_CROSS_ENTROPY_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(target);trace.append_arg(weight);trace.append_arg(reduction);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_binary_cross_entropy_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::binary_cross_entropy_backward(dispatchKeySet, grad_output, self, target, weight, reduction);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_BINARY_CROSS_ENTROPY_BACKWARD, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(target);trace.append_arg(weight);trace.append_arg(reduction);
  return tt;
}

at::Tensor & wrap_binary_cross_entropy_backward_grad_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::binary_cross_entropy_backward_outf(dispatchKeySet, grad_output, self, target, weight, reduction, grad_input);
  }
  bool flush = register_in_place(grad_input, H_BINARY_CROSS_ENTROPY_BACKWARD_GRAD_INPUT, dispatchKeySet);
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(target);trace.append_arg(weight);trace.append_arg(reduction);trace.append_arg(grad_input);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return grad_input;
}

at::Tensor wrap_binary_cross_entropy_with_logits(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & pos_weight, int64_t reduction) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::binary_cross_entropy_with_logits(dispatchKeySet, self, target, weight, pos_weight, reduction);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_BINARY_CROSS_ENTROPY_WITH_LOGITS, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(target);trace.append_arg(weight);trace.append_arg(pos_weight);trace.append_arg(reduction);
  return tt;
}

at::Tensor wrap_binary_cross_entropy_with_logits_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & pos_weight, int64_t reduction) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::binary_cross_entropy_with_logits_backward(dispatchKeySet, grad_output, self, target, weight, pos_weight, reduction);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_BINARY_CROSS_ENTROPY_WITH_LOGITS_BACKWARD, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(target);trace.append_arg(weight);trace.append_arg(pos_weight);trace.append_arg(reduction);
  return tt;
}

at::Tensor wrap_bincount(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::optional<at::Tensor> & weights, int64_t minlength) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::bincount(dispatchKeySet, self, weights, minlength);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_BINCOUNT, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(weights);trace.append_arg(minlength);
  return tt;
}

at::Tensor & wrap_bitwise_not_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::bitwise_not_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_BITWISE_NOT_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_copysign_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::copysign_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_COPYSIGN_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_copysign_Scalar(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::copysign(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_COPYSIGN_SCALAR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_copysign__Scalar(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::copysign_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_COPYSIGN__SCALAR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_copysign_Scalar_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::copysign_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_COPYSIGN_SCALAR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_logical_not(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::logical_not(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LOGICAL_NOT, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_logical_not_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::logical_not_(dispatchKeySet, self);
  }
  bool flush = register_in_place(self, H_LOGICAL_NOT_, dispatchKeySet);
  trace.append_arg(self);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_logical_not_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::logical_not_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_LOGICAL_NOT_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_logical_xor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::logical_xor(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LOGICAL_XOR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_logical_xor_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::logical_xor_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_LOGICAL_XOR_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_logical_xor_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::logical_xor_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_LOGICAL_XOR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_logical_and(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::logical_and(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LOGICAL_AND, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_logical_and_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::logical_and_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_LOGICAL_AND_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_logical_and_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::logical_and_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_LOGICAL_AND_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_logical_or(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::logical_or(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LOGICAL_OR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_logical_or_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::logical_or_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_LOGICAL_OR_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_logical_or_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::logical_or_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_LOGICAL_OR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_blackman_window(c10::DispatchKeySet dispatchKeySet, int64_t window_length, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::blackman_window(dispatchKeySet, window_length, std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_BLACKMAN_WINDOW, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(window_length);trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor wrap_blackman_window_periodic(c10::DispatchKeySet dispatchKeySet, int64_t window_length, bool periodic, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::blackman_window(dispatchKeySet, window_length, periodic, std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_BLACKMAN_WINDOW_PERIODIC, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(window_length);trace.append_arg(periodic);trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor wrap_bmm(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & mat2) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::bmm(dispatchKeySet, self, mat2);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_BMM, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(mat2);
  return tt;
}

at::Tensor wrap__bmm(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & mat2, bool deterministic) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_bmm(dispatchKeySet, self, mat2, deterministic);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__BMM, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(mat2);trace.append_arg(deterministic);
  return tt;
}

at::Tensor & wrap_bmm_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & mat2, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::bmm_outf(dispatchKeySet, self, mat2, out);
  }
  bool flush = register_in_place(out, H_BMM_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(mat2);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap__bmm_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & mat2, bool deterministic, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_bmm_outf(dispatchKeySet, self, mat2, deterministic, out);
  }
  bool flush = register_in_place(out, H__BMM_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(mat2);trace.append_arg(deterministic);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_broadcast_to(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef size) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::broadcast_to(dispatchKeySet, self, std::move(size));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_BROADCAST_TO, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(size));
  return tt;
}

at::Tensor wrap_cat(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors, int64_t dim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::cat(dispatchKeySet, std::move(tensors), dim);
  }
  auto defaults = compute_dtype(tensors);
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_CAT, default_dtype, default_device);
  trace.append_arg(std::move(tensors));trace.append_arg(dim);
  return tt;
}

at::Tensor & wrap_cat_out(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors, int64_t dim, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::cat_outf(dispatchKeySet, std::move(tensors), dim, out);
  }
  bool flush = register_in_place(out, H_CAT_OUT, dispatchKeySet);
  trace.append_arg(std::move(tensors));trace.append_arg(dim);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_cat_names(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors, at::Dimname dim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::cat(dispatchKeySet, std::move(tensors), std::move(dim));
  }
  auto defaults = compute_dtype(tensors);
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_CAT_NAMES, default_dtype, default_device);
  trace.append_arg(std::move(tensors));trace.append_arg(std::move(dim));
  return tt;
}

at::Tensor & wrap_cat_names_out(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors, at::Dimname dim, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::cat_outf(dispatchKeySet, std::move(tensors), std::move(dim), out);
  }
  bool flush = register_in_place(out, H_CAT_NAMES_OUT, dispatchKeySet);
  trace.append_arg(std::move(tensors));trace.append_arg(std::move(dim));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_block_diag(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::block_diag(dispatchKeySet, std::move(tensors));
  }
  auto defaults = compute_dtype(tensors);
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_BLOCK_DIAG, default_dtype, default_device);
  trace.append_arg(std::move(tensors));
  return tt;
}

at::Tensor wrap_ceil(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::ceil(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CEIL, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_ceil_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::ceil_(dispatchKeySet, self);
  }
  bool flush = register_in_place(self, H_CEIL_, dispatchKeySet);
  trace.append_arg(self);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_ceil_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::ceil_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_CEIL_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_chain_matmul(c10::DispatchKeySet dispatchKeySet, at::TensorList matrices) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::chain_matmul(dispatchKeySet, std::move(matrices));
  }
  auto defaults = compute_dtype(matrices);
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_CHAIN_MATMUL, default_dtype, default_device);
  trace.append_arg(std::move(matrices));
  return tt;
}

at::Tensor & wrap_chain_matmul_out(c10::DispatchKeySet dispatchKeySet, at::TensorList matrices, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::chain_matmul_outf(dispatchKeySet, std::move(matrices), out);
  }
  bool flush = register_in_place(out, H_CHAIN_MATMUL_OUT, dispatchKeySet);
  trace.append_arg(std::move(matrices));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_clamp(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::optional<at::Scalar> & min, const c10::optional<at::Scalar> & max) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::clamp(dispatchKeySet, self, min, max);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CLAMP, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(min);trace.append_arg(max);
  return tt;
}

at::Tensor wrap_clamp_Tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::optional<at::Tensor> & min, const c10::optional<at::Tensor> & max) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::clamp(dispatchKeySet, self, min, max);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CLAMP_TENSOR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(min);trace.append_arg(max);
  return tt;
}

at::Tensor & wrap_clamp_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const c10::optional<at::Scalar> & min, const c10::optional<at::Scalar> & max) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::clamp_(dispatchKeySet, self, min, max);
  }
  bool flush = register_in_place(self, H_CLAMP_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(min);trace.append_arg(max);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_clamp__Tensor(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const c10::optional<at::Tensor> & min, const c10::optional<at::Tensor> & max) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::clamp_(dispatchKeySet, self, min, max);
  }
  bool flush = register_in_place(self, H_CLAMP__TENSOR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(min);trace.append_arg(max);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_clamp_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::optional<at::Scalar> & min, const c10::optional<at::Scalar> & max, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::clamp_outf(dispatchKeySet, self, min, max, out);
  }
  bool flush = register_in_place(out, H_CLAMP_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(min);trace.append_arg(max);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_clamp_Tensor_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::optional<at::Tensor> & min, const c10::optional<at::Tensor> & max, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::clamp_outf(dispatchKeySet, self, min, max, out);
  }
  bool flush = register_in_place(out, H_CLAMP_TENSOR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(min);trace.append_arg(max);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_clamp_max(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & max) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::clamp_max(dispatchKeySet, self, max);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CLAMP_MAX, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(max);
  return tt;
}

at::Tensor wrap_clamp_max_Tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & max) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::clamp_max(dispatchKeySet, self, max);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CLAMP_MAX_TENSOR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(max);
  return tt;
}

at::Tensor & wrap_clamp_max_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & max) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::clamp_max_(dispatchKeySet, self, max);
  }
  bool flush = register_in_place(self, H_CLAMP_MAX_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(max);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_clamp_max__Tensor(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & max) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::clamp_max_(dispatchKeySet, self, max);
  }
  bool flush = register_in_place(self, H_CLAMP_MAX__TENSOR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(max);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_clamp_max_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & max, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::clamp_max_outf(dispatchKeySet, self, max, out);
  }
  bool flush = register_in_place(out, H_CLAMP_MAX_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(max);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_clamp_max_Tensor_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & max, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::clamp_max_outf(dispatchKeySet, self, max, out);
  }
  bool flush = register_in_place(out, H_CLAMP_MAX_TENSOR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(max);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_clamp_min(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & min) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::clamp_min(dispatchKeySet, self, min);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CLAMP_MIN, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(min);
  return tt;
}

at::Tensor wrap_clamp_min_Tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & min) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::clamp_min(dispatchKeySet, self, min);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CLAMP_MIN_TENSOR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(min);
  return tt;
}

at::Tensor & wrap_clamp_min_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & min) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::clamp_min_(dispatchKeySet, self, min);
  }
  bool flush = register_in_place(self, H_CLAMP_MIN_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(min);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_clamp_min__Tensor(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & min) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::clamp_min_(dispatchKeySet, self, min);
  }
  bool flush = register_in_place(self, H_CLAMP_MIN__TENSOR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(min);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_clamp_min_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & min, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::clamp_min_outf(dispatchKeySet, self, min, out);
  }
  bool flush = register_in_place(out, H_CLAMP_MIN_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(min);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_clamp_min_Tensor_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & min, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::clamp_min_outf(dispatchKeySet, self, min, out);
  }
  bool flush = register_in_place(out, H_CLAMP_MIN_TENSOR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(min);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_clip(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::optional<at::Scalar> & min, const c10::optional<at::Scalar> & max) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::clip(dispatchKeySet, self, min, max);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CLIP, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(min);trace.append_arg(max);
  return tt;
}

at::Tensor wrap_clip_Tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::optional<at::Tensor> & min, const c10::optional<at::Tensor> & max) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::clip(dispatchKeySet, self, min, max);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CLIP_TENSOR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(min);trace.append_arg(max);
  return tt;
}

at::Tensor & wrap_clip_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const c10::optional<at::Scalar> & min, const c10::optional<at::Scalar> & max) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::clip_(dispatchKeySet, self, min, max);
  }
  bool flush = register_in_place(self, H_CLIP_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(min);trace.append_arg(max);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_clip__Tensor(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const c10::optional<at::Tensor> & min, const c10::optional<at::Tensor> & max) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::clip_(dispatchKeySet, self, min, max);
  }
  bool flush = register_in_place(self, H_CLIP__TENSOR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(min);trace.append_arg(max);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_clip_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::optional<at::Scalar> & min, const c10::optional<at::Scalar> & max, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::clip_outf(dispatchKeySet, self, min, max, out);
  }
  bool flush = register_in_place(out, H_CLIP_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(min);trace.append_arg(max);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_clip_Tensor_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::optional<at::Tensor> & min, const c10::optional<at::Tensor> & max, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::clip_outf(dispatchKeySet, self, min, max, out);
  }
  bool flush = register_in_place(out, H_CLIP_TENSOR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(min);trace.append_arg(max);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_complex(c10::DispatchKeySet dispatchKeySet, const at::Tensor & real, const at::Tensor & imag) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::complex(dispatchKeySet, real, imag);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_COMPLEX, real.dtype(), real.device());
  trace.append_arg(real);trace.append_arg(imag);
  return tt;
}

at::Tensor & wrap_complex_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & real, const at::Tensor & imag, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::complex_outf(dispatchKeySet, real, imag, out);
  }
  bool flush = register_in_place(out, H_COMPLEX_OUT, dispatchKeySet);
  trace.append_arg(real);trace.append_arg(imag);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_polar(c10::DispatchKeySet dispatchKeySet, const at::Tensor & abs, const at::Tensor & angle) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::polar(dispatchKeySet, abs, angle);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_POLAR, abs.dtype(), abs.device());
  trace.append_arg(abs);trace.append_arg(angle);
  return tt;
}

at::Tensor & wrap_polar_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & abs, const at::Tensor & angle, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::polar_outf(dispatchKeySet, abs, angle, out);
  }
  bool flush = register_in_place(out, H_POLAR_OUT, dispatchKeySet);
  trace.append_arg(abs);trace.append_arg(angle);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_constant_pad_nd(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef pad, const at::Scalar & value) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::constant_pad_nd(dispatchKeySet, self, std::move(pad), value);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CONSTANT_PAD_ND, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(pad));trace.append_arg(value);
  return tt;
}

at::Tensor wrap_contiguous(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::MemoryFormat memory_format) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::__dispatch_contiguous(dispatchKeySet, self, std::move(memory_format));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CONTIGUOUS, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(memory_format));
  return tt;
}

at::Tensor wrap_convolution(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::convolution(dispatchKeySet, input, weight, bias, std::move(stride), std::move(padding), std::move(dilation), transposed, std::move(output_padding), groups);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CONVOLUTION, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(weight);trace.append_arg(bias);trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(std::move(dilation));trace.append_arg(transposed);trace.append_arg(std::move(output_padding));trace.append_arg(groups);
  return tt;
}

at::Tensor wrap_convolution_overrideable(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::convolution_overrideable(dispatchKeySet, input, weight, bias, std::move(stride), std::move(padding), std::move(dilation), transposed, std::move(output_padding), groups);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CONVOLUTION_OVERRIDEABLE, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(weight);trace.append_arg(bias);trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(std::move(dilation));trace.append_arg(transposed);trace.append_arg(std::move(output_padding));trace.append_arg(groups);
  return tt;
}

at::Tensor wrap__convolution(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_convolution(dispatchKeySet, input, weight, bias, std::move(stride), std::move(padding), std::move(dilation), transposed, std::move(output_padding), groups, benchmark, deterministic, cudnn_enabled, allow_tf32);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__CONVOLUTION, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(weight);trace.append_arg(bias);trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(std::move(dilation));trace.append_arg(transposed);trace.append_arg(std::move(output_padding));trace.append_arg(groups);trace.append_arg(benchmark);trace.append_arg(deterministic);trace.append_arg(cudnn_enabled);trace.append_arg(allow_tf32);
  return tt;
}

at::Tensor wrap__convolution_deprecated(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_convolution(dispatchKeySet, input, weight, bias, std::move(stride), std::move(padding), std::move(dilation), transposed, std::move(output_padding), groups, benchmark, deterministic, cudnn_enabled);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__CONVOLUTION_DEPRECATED, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(weight);trace.append_arg(bias);trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(std::move(dilation));trace.append_arg(transposed);trace.append_arg(std::move(output_padding));trace.append_arg(groups);trace.append_arg(benchmark);trace.append_arg(deterministic);trace.append_arg(cudnn_enabled);
  return tt;
}

at::Tensor wrap__convolution_mode(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, c10::string_view padding, at::IntArrayRef dilation, int64_t groups) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_convolution_mode(dispatchKeySet, input, weight, bias, std::move(stride), std::move(padding), std::move(dilation), groups);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__CONVOLUTION_MODE, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(weight);trace.append_arg(bias);trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(std::move(dilation));trace.append_arg(groups);
  return tt;
}

at::Tensor wrap__convolution_nogroup(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_convolution_nogroup(dispatchKeySet, input, weight, bias, std::move(stride), std::move(padding), std::move(dilation), transposed, std::move(output_padding));
  }
  auto tt = register_new_tensor(dispatchKeySet, H__CONVOLUTION_NOGROUP, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(weight);trace.append_arg(bias);trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(std::move(dilation));trace.append_arg(transposed);trace.append_arg(std::move(output_padding));
  return tt;
}

at::Tensor wrap_conv1d(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::conv1d(dispatchKeySet, input, weight, bias, std::move(stride), std::move(padding), std::move(dilation), groups);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CONV1D, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(weight);trace.append_arg(bias);trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(std::move(dilation));trace.append_arg(groups);
  return tt;
}

at::Tensor wrap_conv2d(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::conv2d(dispatchKeySet, input, weight, bias, std::move(stride), std::move(padding), std::move(dilation), groups);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CONV2D, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(weight);trace.append_arg(bias);trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(std::move(dilation));trace.append_arg(groups);
  return tt;
}

at::Tensor wrap_conv3d(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::conv3d(dispatchKeySet, input, weight, bias, std::move(stride), std::move(padding), std::move(dilation), groups);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CONV3D, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(weight);trace.append_arg(bias);trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(std::move(dilation));trace.append_arg(groups);
  return tt;
}

at::Tensor wrap_conv1d_padding(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, c10::string_view padding, at::IntArrayRef dilation, int64_t groups) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::conv1d(dispatchKeySet, input, weight, bias, std::move(stride), std::move(padding), std::move(dilation), groups);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CONV1D_PADDING, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(weight);trace.append_arg(bias);trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(std::move(dilation));trace.append_arg(groups);
  return tt;
}

at::Tensor wrap_conv2d_padding(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, c10::string_view padding, at::IntArrayRef dilation, int64_t groups) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::conv2d(dispatchKeySet, input, weight, bias, std::move(stride), std::move(padding), std::move(dilation), groups);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CONV2D_PADDING, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(weight);trace.append_arg(bias);trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(std::move(dilation));trace.append_arg(groups);
  return tt;
}

at::Tensor wrap_conv3d_padding(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, c10::string_view padding, at::IntArrayRef dilation, int64_t groups) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::conv3d(dispatchKeySet, input, weight, bias, std::move(stride), std::move(padding), std::move(dilation), groups);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CONV3D_PADDING, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(weight);trace.append_arg(bias);trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(std::move(dilation));trace.append_arg(groups);
  return tt;
}

at::Tensor wrap_conv_tbc(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, const at::Tensor & bias, int64_t pad) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::conv_tbc(dispatchKeySet, self, weight, bias, pad);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CONV_TBC, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(weight);trace.append_arg(bias);trace.append_arg(pad);
  return tt;
}

at::Tensor wrap_conv_transpose1d(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, int64_t groups, at::IntArrayRef dilation) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::conv_transpose1d(dispatchKeySet, input, weight, bias, std::move(stride), std::move(padding), std::move(output_padding), groups, std::move(dilation));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CONV_TRANSPOSE1D, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(weight);trace.append_arg(bias);trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(std::move(output_padding));trace.append_arg(groups);trace.append_arg(std::move(dilation));
  return tt;
}

at::Tensor wrap_conv_transpose2d_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, int64_t groups, at::IntArrayRef dilation) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::conv_transpose2d(dispatchKeySet, input, weight, bias, std::move(stride), std::move(padding), std::move(output_padding), groups, std::move(dilation));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CONV_TRANSPOSE2D_INPUT, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(weight);trace.append_arg(bias);trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(std::move(output_padding));trace.append_arg(groups);trace.append_arg(std::move(dilation));
  return tt;
}

at::Tensor wrap_conv_transpose3d_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, int64_t groups, at::IntArrayRef dilation) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::conv_transpose3d(dispatchKeySet, input, weight, bias, std::move(stride), std::move(padding), std::move(output_padding), groups, std::move(dilation));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CONV_TRANSPOSE3D_INPUT, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(weight);trace.append_arg(bias);trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(std::move(output_padding));trace.append_arg(groups);trace.append_arg(std::move(dilation));
  return tt;
}

at::Tensor & wrap_copy_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & src, bool non_blocking) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::copy_(dispatchKeySet, self, src, non_blocking);
  }
  bool flush = register_in_place(self, H_COPY_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(src);trace.append_arg(non_blocking);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_cos_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::cos_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_COS_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_cosh_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::cosh_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_COSH_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_cosine_embedding_loss(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input1, const at::Tensor & input2, const at::Tensor & target, double margin, int64_t reduction) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::cosine_embedding_loss(dispatchKeySet, input1, input2, target, margin, reduction);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_COSINE_EMBEDDING_LOSS, input1.dtype(), input1.device());
  trace.append_arg(input1);trace.append_arg(input2);trace.append_arg(target);trace.append_arg(margin);trace.append_arg(reduction);
  return tt;
}

at::Tensor wrap_count_nonzero_dim_IntList(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::count_nonzero(dispatchKeySet, self, std::move(dim));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_COUNT_NONZERO_DIM_INTLIST, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim));
  return tt;
}

at::Tensor wrap_count_nonzero(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<int64_t> dim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::count_nonzero(dispatchKeySet, self, dim);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_COUNT_NONZERO, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(dim);
  return tt;
}

at::Tensor wrap_cudnn_affine_grid_generator(c10::DispatchKeySet dispatchKeySet, const at::Tensor & theta, int64_t N, int64_t C, int64_t H, int64_t W) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::cudnn_affine_grid_generator(dispatchKeySet, theta, N, C, H, W);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CUDNN_AFFINE_GRID_GENERATOR, theta.dtype(), theta.device());
  trace.append_arg(theta);trace.append_arg(N);trace.append_arg(C);trace.append_arg(H);trace.append_arg(W);
  return tt;
}

at::Tensor wrap_cudnn_affine_grid_generator_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, int64_t N, int64_t C, int64_t H, int64_t W) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::cudnn_affine_grid_generator_backward(dispatchKeySet, grad, N, C, H, W);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CUDNN_AFFINE_GRID_GENERATOR_BACKWARD, grad.dtype(), grad.device());
  trace.append_arg(grad);trace.append_arg(N);trace.append_arg(C);trace.append_arg(H);trace.append_arg(W);
  return tt;
}

at::Tensor wrap_cudnn_convolution_deprecated(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::cudnn_convolution(dispatchKeySet, self, weight, bias, std::move(padding), std::move(stride), std::move(dilation), groups, benchmark, deterministic);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CUDNN_CONVOLUTION_DEPRECATED, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(weight);trace.append_arg(bias);trace.append_arg(std::move(padding));trace.append_arg(std::move(stride));trace.append_arg(std::move(dilation));trace.append_arg(groups);trace.append_arg(benchmark);trace.append_arg(deterministic);
  return tt;
}

at::Tensor wrap_cudnn_convolution_deprecated2(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::cudnn_convolution(dispatchKeySet, self, weight, std::move(padding), std::move(stride), std::move(dilation), groups, benchmark, deterministic);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CUDNN_CONVOLUTION_DEPRECATED2, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(weight);trace.append_arg(std::move(padding));trace.append_arg(std::move(stride));trace.append_arg(std::move(dilation));trace.append_arg(groups);trace.append_arg(benchmark);trace.append_arg(deterministic);
  return tt;
}

at::Tensor wrap_cudnn_convolution(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, bool allow_tf32) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::cudnn_convolution(dispatchKeySet, self, weight, std::move(padding), std::move(stride), std::move(dilation), groups, benchmark, deterministic, allow_tf32);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CUDNN_CONVOLUTION, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(weight);trace.append_arg(std::move(padding));trace.append_arg(std::move(stride));trace.append_arg(std::move(dilation));trace.append_arg(groups);trace.append_arg(benchmark);trace.append_arg(deterministic);trace.append_arg(allow_tf32);
  return tt;
}

at::Tensor wrap_cudnn_convolution_backward_input(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef self_size, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, bool allow_tf32) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::cudnn_convolution_backward_input(dispatchKeySet, std::move(self_size), grad_output, weight, std::move(padding), std::move(stride), std::move(dilation), groups, benchmark, deterministic, allow_tf32);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CUDNN_CONVOLUTION_BACKWARD_INPUT, grad_output.dtype(), grad_output.device());
  trace.append_arg(std::move(self_size));trace.append_arg(grad_output);trace.append_arg(weight);trace.append_arg(std::move(padding));trace.append_arg(std::move(stride));trace.append_arg(std::move(dilation));trace.append_arg(groups);trace.append_arg(benchmark);trace.append_arg(deterministic);trace.append_arg(allow_tf32);
  return tt;
}

at::Tensor wrap_cudnn_convolution_backward_weight(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef weight_size, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, bool allow_tf32) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::cudnn_convolution_backward_weight(dispatchKeySet, std::move(weight_size), grad_output, self, std::move(padding), std::move(stride), std::move(dilation), groups, benchmark, deterministic, allow_tf32);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CUDNN_CONVOLUTION_BACKWARD_WEIGHT, grad_output.dtype(), grad_output.device());
  trace.append_arg(std::move(weight_size));trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(std::move(padding));trace.append_arg(std::move(stride));trace.append_arg(std::move(dilation));trace.append_arg(groups);trace.append_arg(benchmark);trace.append_arg(deterministic);trace.append_arg(allow_tf32);
  return tt;
}

at::Tensor wrap_cudnn_convolution_transpose_deprecated(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::cudnn_convolution_transpose(dispatchKeySet, self, weight, bias, std::move(padding), std::move(output_padding), std::move(stride), std::move(dilation), groups, benchmark, deterministic);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CUDNN_CONVOLUTION_TRANSPOSE_DEPRECATED, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(weight);trace.append_arg(bias);trace.append_arg(std::move(padding));trace.append_arg(std::move(output_padding));trace.append_arg(std::move(stride));trace.append_arg(std::move(dilation));trace.append_arg(groups);trace.append_arg(benchmark);trace.append_arg(deterministic);
  return tt;
}

at::Tensor wrap_cudnn_convolution_transpose_deprecated2(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::cudnn_convolution_transpose(dispatchKeySet, self, weight, std::move(padding), std::move(output_padding), std::move(stride), std::move(dilation), groups, benchmark, deterministic);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CUDNN_CONVOLUTION_TRANSPOSE_DEPRECATED2, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(weight);trace.append_arg(std::move(padding));trace.append_arg(std::move(output_padding));trace.append_arg(std::move(stride));trace.append_arg(std::move(dilation));trace.append_arg(groups);trace.append_arg(benchmark);trace.append_arg(deterministic);
  return tt;
}

at::Tensor wrap_cudnn_convolution_transpose(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, bool allow_tf32) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::cudnn_convolution_transpose(dispatchKeySet, self, weight, std::move(padding), std::move(output_padding), std::move(stride), std::move(dilation), groups, benchmark, deterministic, allow_tf32);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CUDNN_CONVOLUTION_TRANSPOSE, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(weight);trace.append_arg(std::move(padding));trace.append_arg(std::move(output_padding));trace.append_arg(std::move(stride));trace.append_arg(std::move(dilation));trace.append_arg(groups);trace.append_arg(benchmark);trace.append_arg(deterministic);trace.append_arg(allow_tf32);
  return tt;
}

at::Tensor wrap_cudnn_convolution_transpose_backward_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, bool allow_tf32) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::cudnn_convolution_transpose_backward_input(dispatchKeySet, grad_output, weight, std::move(padding), std::move(stride), std::move(dilation), groups, benchmark, deterministic, allow_tf32);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CUDNN_CONVOLUTION_TRANSPOSE_BACKWARD_INPUT, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(weight);trace.append_arg(std::move(padding));trace.append_arg(std::move(stride));trace.append_arg(std::move(dilation));trace.append_arg(groups);trace.append_arg(benchmark);trace.append_arg(deterministic);trace.append_arg(allow_tf32);
  return tt;
}

at::Tensor wrap_cudnn_convolution_transpose_backward_weight(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef weight_size, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, bool allow_tf32) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::cudnn_convolution_transpose_backward_weight(dispatchKeySet, std::move(weight_size), grad_output, self, std::move(padding), std::move(stride), std::move(dilation), groups, benchmark, deterministic, allow_tf32);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CUDNN_CONVOLUTION_TRANSPOSE_BACKWARD_WEIGHT, grad_output.dtype(), grad_output.device());
  trace.append_arg(std::move(weight_size));trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(std::move(padding));trace.append_arg(std::move(stride));trace.append_arg(std::move(dilation));trace.append_arg(groups);trace.append_arg(benchmark);trace.append_arg(deterministic);trace.append_arg(allow_tf32);
  return tt;
}

at::Tensor wrap_cudnn_convolution_relu(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::cudnn_convolution_relu(dispatchKeySet, self, weight, bias, std::move(stride), std::move(padding), std::move(dilation), groups);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CUDNN_CONVOLUTION_RELU, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(weight);trace.append_arg(bias);trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(std::move(dilation));trace.append_arg(groups);
  return tt;
}

at::Tensor wrap_cudnn_convolution_add_relu(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, const at::Tensor & z, const c10::optional<at::Scalar> & alpha, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::cudnn_convolution_add_relu(dispatchKeySet, self, weight, z, alpha, bias, std::move(stride), std::move(padding), std::move(dilation), groups);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CUDNN_CONVOLUTION_ADD_RELU, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(weight);trace.append_arg(z);trace.append_arg(alpha);trace.append_arg(bias);trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(std::move(dilation));trace.append_arg(groups);
  return tt;
}

at::Tensor wrap_cudnn_grid_sampler(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & grid) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::cudnn_grid_sampler(dispatchKeySet, self, grid);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CUDNN_GRID_SAMPLER, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(grid);
  return tt;
}

at::Tensor wrap_cummaxmin_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, const at::Tensor & input, const at::Tensor & indices, int64_t dim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::cummaxmin_backward(dispatchKeySet, grad, input, indices, dim);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CUMMAXMIN_BACKWARD, grad.dtype(), grad.device());
  trace.append_arg(grad);trace.append_arg(input);trace.append_arg(indices);trace.append_arg(dim);
  return tt;
}

at::Tensor wrap_cumprod(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::cumprod(dispatchKeySet, self, dim, std::move(dtype));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CUMPROD, dtype ? scalarTypeToTypeMeta(*dtype) : self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(std::move(dtype));
  return tt;
}

at::Tensor & wrap_cumprod_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::cumprod_(dispatchKeySet, self, dim, std::move(dtype));
  }
  bool flush = register_in_place(self, H_CUMPROD_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(std::move(dtype));
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_cumprod_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::cumprod_outf(dispatchKeySet, self, dim, std::move(dtype), out);
  }
  bool flush = register_in_place(out, H_CUMPROD_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(std::move(dtype));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_cumprod_dimname(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::cumprod(dispatchKeySet, self, std::move(dim), std::move(dtype));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CUMPROD_DIMNAME, dtype ? scalarTypeToTypeMeta(*dtype) : self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(std::move(dtype));
  return tt;
}

at::Tensor & wrap_cumprod__dimname(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::cumprod_(dispatchKeySet, self, std::move(dim), std::move(dtype));
  }
  bool flush = register_in_place(self, H_CUMPROD__DIMNAME, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(std::move(dtype));
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_cumprod_dimname_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::cumprod_outf(dispatchKeySet, self, std::move(dim), std::move(dtype), out);
  }
  bool flush = register_in_place(out, H_CUMPROD_DIMNAME_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(std::move(dtype));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_cumprod_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, const at::Tensor & input, int64_t dim, const at::Tensor & output) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::cumprod_backward(dispatchKeySet, grad, input, dim, output);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CUMPROD_BACKWARD, grad.dtype(), grad.device());
  trace.append_arg(grad);trace.append_arg(input);trace.append_arg(dim);trace.append_arg(output);
  return tt;
}

at::Tensor wrap_cumsum(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::cumsum(dispatchKeySet, self, dim, std::move(dtype));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CUMSUM, dtype ? scalarTypeToTypeMeta(*dtype) : self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(std::move(dtype));
  return tt;
}

at::Tensor & wrap_cumsum_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::cumsum_(dispatchKeySet, self, dim, std::move(dtype));
  }
  bool flush = register_in_place(self, H_CUMSUM_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(std::move(dtype));
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_cumsum_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::cumsum_outf(dispatchKeySet, self, dim, std::move(dtype), out);
  }
  bool flush = register_in_place(out, H_CUMSUM_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(std::move(dtype));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_cumsum_dimname(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::cumsum(dispatchKeySet, self, std::move(dim), std::move(dtype));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CUMSUM_DIMNAME, dtype ? scalarTypeToTypeMeta(*dtype) : self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(std::move(dtype));
  return tt;
}

at::Tensor & wrap_cumsum__dimname(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::cumsum_(dispatchKeySet, self, std::move(dim), std::move(dtype));
  }
  bool flush = register_in_place(self, H_CUMSUM__DIMNAME, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(std::move(dtype));
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_cumsum_dimname_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::cumsum_outf(dispatchKeySet, self, std::move(dim), std::move(dtype), out);
  }
  bool flush = register_in_place(out, H_CUMSUM_DIMNAME_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(std::move(dtype));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_ctc_loss_IntList(c10::DispatchKeySet dispatchKeySet, const at::Tensor & log_probs, const at::Tensor & targets, at::IntArrayRef input_lengths, at::IntArrayRef target_lengths, int64_t blank, int64_t reduction, bool zero_infinity) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::ctc_loss(dispatchKeySet, log_probs, targets, std::move(input_lengths), std::move(target_lengths), blank, reduction, zero_infinity);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CTC_LOSS_INTLIST, log_probs.dtype(), log_probs.device());
  trace.append_arg(log_probs);trace.append_arg(targets);trace.append_arg(std::move(input_lengths));trace.append_arg(std::move(target_lengths));trace.append_arg(blank);trace.append_arg(reduction);trace.append_arg(zero_infinity);
  return tt;
}

at::Tensor wrap_ctc_loss_Tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & log_probs, const at::Tensor & targets, const at::Tensor & input_lengths, const at::Tensor & target_lengths, int64_t blank, int64_t reduction, bool zero_infinity) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::ctc_loss(dispatchKeySet, log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CTC_LOSS_TENSOR, log_probs.dtype(), log_probs.device());
  trace.append_arg(log_probs);trace.append_arg(targets);trace.append_arg(input_lengths);trace.append_arg(target_lengths);trace.append_arg(blank);trace.append_arg(reduction);trace.append_arg(zero_infinity);
  return tt;
}

at::Tensor wrap__ctc_loss_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, const at::Tensor & log_probs, const at::Tensor & targets, at::IntArrayRef input_lengths, at::IntArrayRef target_lengths, const at::Tensor & neg_log_likelihood, const at::Tensor & log_alpha, int64_t blank, bool zero_infinity) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_ctc_loss_backward(dispatchKeySet, grad, log_probs, targets, std::move(input_lengths), std::move(target_lengths), neg_log_likelihood, log_alpha, blank, zero_infinity);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__CTC_LOSS_BACKWARD, grad.dtype(), grad.device());
  trace.append_arg(grad);trace.append_arg(log_probs);trace.append_arg(targets);trace.append_arg(std::move(input_lengths));trace.append_arg(std::move(target_lengths));trace.append_arg(neg_log_likelihood);trace.append_arg(log_alpha);trace.append_arg(blank);trace.append_arg(zero_infinity);
  return tt;
}

at::Tensor wrap_diag_embed(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::diag_embed(dispatchKeySet, self, offset, dim1, dim2);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_DIAG_EMBED, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(offset);trace.append_arg(dim1);trace.append_arg(dim2);
  return tt;
}

at::Tensor wrap_diagflat(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t offset) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::diagflat(dispatchKeySet, self, offset);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_DIAGFLAT, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(offset);
  return tt;
}

at::Tensor wrap_diagonal(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::diagonal(dispatchKeySet, self, offset, dim1, dim2);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_DIAGONAL, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(offset);trace.append_arg(dim1);trace.append_arg(dim2);
  return tt;
}

at::Tensor wrap_diagonal_Dimname(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname outdim, at::Dimname dim1, at::Dimname dim2, int64_t offset) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::diagonal(dispatchKeySet, self, std::move(outdim), std::move(dim1), std::move(dim2), offset);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_DIAGONAL_DIMNAME, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(outdim));trace.append_arg(std::move(dim1));trace.append_arg(std::move(dim2));trace.append_arg(offset);
  return tt;
}

at::Tensor wrap_diagonal_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, at::IntArrayRef input_sizes, int64_t offset, int64_t dim1, int64_t dim2) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::diagonal_backward(dispatchKeySet, grad, std::move(input_sizes), offset, dim1, dim2);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_DIAGONAL_BACKWARD, grad.dtype(), grad.device());
  trace.append_arg(grad);trace.append_arg(std::move(input_sizes));trace.append_arg(offset);trace.append_arg(dim1);trace.append_arg(dim2);
  return tt;
}

at::Tensor & wrap_fill_diagonal_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & fill_value, bool wrap) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fill_diagonal_(dispatchKeySet, self, fill_value, wrap);
  }
  bool flush = register_in_place(self, H_FILL_DIAGONAL_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(fill_value);trace.append_arg(wrap);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_diff(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t n, int64_t dim, const c10::optional<at::Tensor> & prepend, const c10::optional<at::Tensor> & append) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::diff(dispatchKeySet, self, n, dim, prepend, append);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_DIFF, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(n);trace.append_arg(dim);trace.append_arg(prepend);trace.append_arg(append);
  return tt;
}

at::Tensor & wrap_diff_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t n, int64_t dim, const c10::optional<at::Tensor> & prepend, const c10::optional<at::Tensor> & append, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::diff_outf(dispatchKeySet, self, n, dim, prepend, append, out);
  }
  bool flush = register_in_place(out, H_DIFF_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(n);trace.append_arg(dim);trace.append_arg(prepend);trace.append_arg(append);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_div_Tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::div(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_DIV_TENSOR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_div__Tensor(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::div_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_DIV__TENSOR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_div_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::div_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_DIV_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_div_Tensor_mode(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, c10::optional<c10::string_view> rounding_mode) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::div(dispatchKeySet, self, other, std::move(rounding_mode));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_DIV_TENSOR_MODE, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(std::move(rounding_mode));
  return tt;
}

at::Tensor & wrap_div__Tensor_mode(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other, c10::optional<c10::string_view> rounding_mode) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::div_(dispatchKeySet, self, other, std::move(rounding_mode));
  }
  bool flush = register_in_place(self, H_DIV__TENSOR_MODE, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(std::move(rounding_mode));
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_div_out_mode(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, c10::optional<c10::string_view> rounding_mode, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::div_outf(dispatchKeySet, self, other, std::move(rounding_mode), out);
  }
  bool flush = register_in_place(out, H_DIV_OUT_MODE, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(std::move(rounding_mode));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_div_Scalar(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::div(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_DIV_SCALAR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_div__Scalar(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::div_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_DIV__SCALAR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_div_Scalar_mode(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, c10::optional<c10::string_view> rounding_mode) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::div(dispatchKeySet, self, other, std::move(rounding_mode));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_DIV_SCALAR_MODE, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(std::move(rounding_mode));
  return tt;
}

at::Tensor & wrap_div__Scalar_mode(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other, c10::optional<c10::string_view> rounding_mode) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::div_(dispatchKeySet, self, other, std::move(rounding_mode));
  }
  bool flush = register_in_place(self, H_DIV__SCALAR_MODE, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(std::move(rounding_mode));
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_divide_Tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::divide(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_DIVIDE_TENSOR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_divide__Tensor(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::divide_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_DIVIDE__TENSOR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_divide_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::divide_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_DIVIDE_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_divide_Scalar(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::divide(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_DIVIDE_SCALAR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_divide__Scalar(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::divide_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_DIVIDE__SCALAR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_divide_Tensor_mode(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, c10::optional<c10::string_view> rounding_mode) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::divide(dispatchKeySet, self, other, std::move(rounding_mode));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_DIVIDE_TENSOR_MODE, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(std::move(rounding_mode));
  return tt;
}

at::Tensor & wrap_divide__Tensor_mode(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other, c10::optional<c10::string_view> rounding_mode) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::divide_(dispatchKeySet, self, other, std::move(rounding_mode));
  }
  bool flush = register_in_place(self, H_DIVIDE__TENSOR_MODE, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(std::move(rounding_mode));
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_divide_out_mode(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, c10::optional<c10::string_view> rounding_mode, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::divide_outf(dispatchKeySet, self, other, std::move(rounding_mode), out);
  }
  bool flush = register_in_place(out, H_DIVIDE_OUT_MODE, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(std::move(rounding_mode));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_divide_Scalar_mode(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, c10::optional<c10::string_view> rounding_mode) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::divide(dispatchKeySet, self, other, std::move(rounding_mode));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_DIVIDE_SCALAR_MODE, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(std::move(rounding_mode));
  return tt;
}

at::Tensor & wrap_divide__Scalar_mode(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other, c10::optional<c10::string_view> rounding_mode) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::divide_(dispatchKeySet, self, other, std::move(rounding_mode));
  }
  bool flush = register_in_place(self, H_DIVIDE__SCALAR_MODE, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(std::move(rounding_mode));
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_true_divide_Tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::true_divide(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_TRUE_DIVIDE_TENSOR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_true_divide__Tensor(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::true_divide_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_TRUE_DIVIDE__TENSOR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_true_divide_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::true_divide_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_TRUE_DIVIDE_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_true_divide_Scalar(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::true_divide(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_TRUE_DIVIDE_SCALAR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_true_divide__Scalar(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::true_divide_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_TRUE_DIVIDE__SCALAR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_dot(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & tensor) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::dot(dispatchKeySet, self, tensor);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_DOT, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(tensor);
  return tt;
}

at::Tensor & wrap_dot_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & tensor, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::dot_outf(dispatchKeySet, self, tensor, out);
  }
  bool flush = register_in_place(out, H_DOT_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(tensor);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_vdot(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::vdot(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_VDOT, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_vdot_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::vdot_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_VDOT_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_einsum(c10::DispatchKeySet dispatchKeySet, c10::string_view equation, at::TensorList tensors) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::einsum(dispatchKeySet, std::move(equation), std::move(tensors));
  }
  auto defaults = compute_dtype(tensors);
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_EINSUM, default_dtype, default_device);
  trace.append_arg(std::move(equation));trace.append_arg(std::move(tensors));
  return tt;
}

at::Tensor wrap_embedding(c10::DispatchKeySet dispatchKeySet, const at::Tensor & weight, const at::Tensor & indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::embedding(dispatchKeySet, weight, indices, padding_idx, scale_grad_by_freq, sparse);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_EMBEDDING, weight.dtype(), weight.device());
  trace.append_arg(weight);trace.append_arg(indices);trace.append_arg(padding_idx);trace.append_arg(scale_grad_by_freq);trace.append_arg(sparse);
  return tt;
}

at::Tensor wrap_embedding_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, const at::Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::embedding_backward(dispatchKeySet, grad, indices, num_weights, padding_idx, scale_grad_by_freq, sparse);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_EMBEDDING_BACKWARD, grad.dtype(), grad.device());
  trace.append_arg(grad);trace.append_arg(indices);trace.append_arg(num_weights);trace.append_arg(padding_idx);trace.append_arg(scale_grad_by_freq);trace.append_arg(sparse);
  return tt;
}

at::Tensor wrap_embedding_dense_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::embedding_dense_backward(dispatchKeySet, grad_output, indices, num_weights, padding_idx, scale_grad_by_freq);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_EMBEDDING_DENSE_BACKWARD, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(indices);trace.append_arg(num_weights);trace.append_arg(padding_idx);trace.append_arg(scale_grad_by_freq);
  return tt;
}

at::Tensor & wrap_embedding_renorm_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & indices, double max_norm, double norm_type) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::embedding_renorm_(dispatchKeySet, self, indices, max_norm, norm_type);
  }
  bool flush = register_in_place(self, H_EMBEDDING_RENORM_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(indices);trace.append_arg(max_norm);trace.append_arg(norm_type);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_embedding_sparse_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, const at::Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::embedding_sparse_backward(dispatchKeySet, grad, indices, num_weights, padding_idx, scale_grad_by_freq);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_EMBEDDING_SPARSE_BACKWARD, grad.dtype(), grad.device());
  trace.append_arg(grad);trace.append_arg(indices);trace.append_arg(num_weights);trace.append_arg(padding_idx);trace.append_arg(scale_grad_by_freq);
  return tt;
}

at::Tensor wrap_row_stack(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::row_stack(dispatchKeySet, std::move(tensors));
  }
  auto defaults = compute_dtype(tensors);
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_ROW_STACK, default_dtype, default_device);
  trace.append_arg(std::move(tensors));
  return tt;
}

at::Tensor & wrap_row_stack_out(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::row_stack_outf(dispatchKeySet, std::move(tensors), out);
  }
  bool flush = register_in_place(out, H_ROW_STACK_OUT, dispatchKeySet);
  trace.append_arg(std::move(tensors));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap__embedding_bag_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, const at::Tensor & indices, const at::Tensor & offsets, const at::Tensor & offset2bag, const at::Tensor & bag_size, const at::Tensor & maximum_indices, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, bool sparse, const c10::optional<at::Tensor> & per_sample_weights, int64_t padding_idx) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_embedding_bag_backward(dispatchKeySet, grad, indices, offsets, offset2bag, bag_size, maximum_indices, num_weights, scale_grad_by_freq, mode, sparse, per_sample_weights, padding_idx);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__EMBEDDING_BAG_BACKWARD, grad.dtype(), grad.device());
  trace.append_arg(grad);trace.append_arg(indices);trace.append_arg(offsets);trace.append_arg(offset2bag);trace.append_arg(bag_size);trace.append_arg(maximum_indices);trace.append_arg(num_weights);trace.append_arg(scale_grad_by_freq);trace.append_arg(mode);trace.append_arg(sparse);trace.append_arg(per_sample_weights);trace.append_arg(padding_idx);
  return tt;
}

at::Tensor wrap__embedding_bag_sparse_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, const at::Tensor & indices, const at::Tensor & offsets, const at::Tensor & offset2bag, const at::Tensor & bag_size, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, const c10::optional<at::Tensor> & per_sample_weights, int64_t padding_idx) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_embedding_bag_sparse_backward(dispatchKeySet, grad, indices, offsets, offset2bag, bag_size, num_weights, scale_grad_by_freq, mode, per_sample_weights, padding_idx);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__EMBEDDING_BAG_SPARSE_BACKWARD, grad.dtype(), grad.device());
  trace.append_arg(grad);trace.append_arg(indices);trace.append_arg(offsets);trace.append_arg(offset2bag);trace.append_arg(bag_size);trace.append_arg(num_weights);trace.append_arg(scale_grad_by_freq);trace.append_arg(mode);trace.append_arg(per_sample_weights);trace.append_arg(padding_idx);
  return tt;
}

at::Tensor wrap__embedding_bag_dense_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, const at::Tensor & indices, const at::Tensor & offset2bag, const at::Tensor & bag_size, const at::Tensor & maximum_indices, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, const c10::optional<at::Tensor> & per_sample_weights, int64_t padding_idx) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_embedding_bag_dense_backward(dispatchKeySet, grad, indices, offset2bag, bag_size, maximum_indices, num_weights, scale_grad_by_freq, mode, per_sample_weights, padding_idx);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__EMBEDDING_BAG_DENSE_BACKWARD, grad.dtype(), grad.device());
  trace.append_arg(grad);trace.append_arg(indices);trace.append_arg(offset2bag);trace.append_arg(bag_size);trace.append_arg(maximum_indices);trace.append_arg(num_weights);trace.append_arg(scale_grad_by_freq);trace.append_arg(mode);trace.append_arg(per_sample_weights);trace.append_arg(padding_idx);
  return tt;
}

at::Tensor wrap__embedding_bag_per_sample_weights_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, const at::Tensor & offset2bag, int64_t mode, int64_t padding_idx) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_embedding_bag_per_sample_weights_backward(dispatchKeySet, grad, weight, indices, offsets, offset2bag, mode, padding_idx);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__EMBEDDING_BAG_PER_SAMPLE_WEIGHTS_BACKWARD, grad.dtype(), grad.device());
  trace.append_arg(grad);trace.append_arg(weight);trace.append_arg(indices);trace.append_arg(offsets);trace.append_arg(offset2bag);trace.append_arg(mode);trace.append_arg(padding_idx);
  return tt;
}

at::Tensor wrap_empty_names(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, c10::optional<at::DimnameList> names, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::empty(dispatchKeySet, std::move(size), std::move(names), std::move(dtype), std::move(layout), std::move(device), pin_memory, std::move(memory_format));
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_EMPTY_NAMES, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(std::move(size));trace.append_arg(std::move(names));trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);trace.append_arg(std::move(memory_format));
  return tt;
}

at::Tensor wrap_empty_memory_format(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::empty(dispatchKeySet, std::move(size), std::move(dtype), std::move(layout), std::move(device), pin_memory, std::move(memory_format));
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_EMPTY_MEMORY_FORMAT, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(std::move(size));trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);trace.append_arg(std::move(memory_format));
  return tt;
}

at::Tensor wrap_new_empty(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::new_empty(dispatchKeySet, self, std::move(size), std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_NEW_EMPTY, dtype ? scalarTypeToTypeMeta(*dtype) : self.dtype(), device ? *device : self.device());
  trace.append_arg(self);trace.append_arg(std::move(size));trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor wrap_new_empty_strided(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef size, at::IntArrayRef stride, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::new_empty_strided(dispatchKeySet, self, std::move(size), std::move(stride), std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_NEW_EMPTY_STRIDED, dtype ? scalarTypeToTypeMeta(*dtype) : self.dtype(), device ? *device : self.device());
  trace.append_arg(self);trace.append_arg(std::move(size));trace.append_arg(std::move(stride));trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor wrap_new_full(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef size, const at::Scalar & fill_value, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::new_full(dispatchKeySet, self, std::move(size), fill_value, std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_NEW_FULL, dtype ? scalarTypeToTypeMeta(*dtype) : self.dtype(), device ? *device : self.device());
  trace.append_arg(self);trace.append_arg(std::move(size));trace.append_arg(fill_value);trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor wrap_new_zeros(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::new_zeros(dispatchKeySet, self, std::move(size), std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_NEW_ZEROS, dtype ? scalarTypeToTypeMeta(*dtype) : self.dtype(), device ? *device : self.device());
  trace.append_arg(self);trace.append_arg(std::move(size));trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor wrap_new_ones(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::new_ones(dispatchKeySet, self, std::move(size), std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_NEW_ONES, dtype ? scalarTypeToTypeMeta(*dtype) : self.dtype(), device ? *device : self.device());
  trace.append_arg(self);trace.append_arg(std::move(size));trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor wrap__empty_affine_quantized(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, double scale, int64_t zero_point, c10::optional<at::MemoryFormat> memory_format) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_empty_affine_quantized(dispatchKeySet, std::move(size), std::move(dtype), std::move(layout), std::move(device), pin_memory, scale, zero_point, std::move(memory_format));
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H__EMPTY_AFFINE_QUANTIZED, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(std::move(size));trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);trace.append_arg(scale);trace.append_arg(zero_point);trace.append_arg(std::move(memory_format));
  return tt;
}

at::Tensor wrap__empty_per_channel_affine_quantized(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, const at::Tensor & scales, const at::Tensor & zero_points, int64_t axis, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_empty_per_channel_affine_quantized(dispatchKeySet, std::move(size), scales, zero_points, axis, std::move(dtype), std::move(layout), std::move(device), pin_memory, std::move(memory_format));
  }
  auto tt = register_new_tensor(dispatchKeySet, H__EMPTY_PER_CHANNEL_AFFINE_QUANTIZED, dtype ? scalarTypeToTypeMeta(*dtype) : scales.dtype(), device ? *device : scales.device());
  trace.append_arg(std::move(size));trace.append_arg(scales);trace.append_arg(zero_points);trace.append_arg(axis);trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);trace.append_arg(std::move(memory_format));
  return tt;
}

const at::Tensor & wrap_resize_(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef size, c10::optional<at::MemoryFormat> memory_format) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::resize_(dispatchKeySet, self, std::move(size), std::move(memory_format));
  }
  bool flush = register_in_place(self, H_RESIZE_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(size));trace.append_arg(std::move(memory_format));
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_empty_quantized(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, const at::Tensor & qtensor) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::empty_quantized(dispatchKeySet, std::move(size), qtensor);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_EMPTY_QUANTIZED, qtensor.dtype(), qtensor.device());
  trace.append_arg(std::move(size));trace.append_arg(qtensor);
  return tt;
}

at::Tensor & wrap_empty_out(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, c10::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::empty_outf(dispatchKeySet, std::move(size), std::move(memory_format), out);
  }
  bool flush = register_in_place(out, H_EMPTY_OUT, dispatchKeySet);
  trace.append_arg(std::move(size));trace.append_arg(std::move(memory_format));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_empty_like(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::empty_like(dispatchKeySet, self, std::move(dtype), std::move(layout), std::move(device), pin_memory, std::move(memory_format));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_EMPTY_LIKE, dtype ? scalarTypeToTypeMeta(*dtype) : self.dtype(), device ? *device : self.device());
  trace.append_arg(self);trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);trace.append_arg(std::move(memory_format));
  return tt;
}

at::Tensor wrap_empty_strided(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, at::IntArrayRef stride, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::empty_strided(dispatchKeySet, std::move(size), std::move(stride), std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_EMPTY_STRIDED, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(std::move(size));trace.append_arg(std::move(stride));trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor & wrap_erf_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::erf_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_ERF_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_erfc_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::erfc_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_ERFC_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_exp_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::exp_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_EXP_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_exp2_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::exp2_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_EXP2_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_expm1_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::expm1_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_EXPM1_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_expand(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef size, bool implicit) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::expand(dispatchKeySet, self, std::move(size), implicit);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_EXPAND, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(size));trace.append_arg(implicit);
  return tt;
}

at::Tensor wrap_expand_as(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::expand_as(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_EXPAND_AS, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor wrap_eye(c10::DispatchKeySet dispatchKeySet, int64_t n, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::eye(dispatchKeySet, n, std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_EYE, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(n);trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor wrap_eye_m(c10::DispatchKeySet dispatchKeySet, int64_t n, int64_t m, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::eye(dispatchKeySet, n, m, std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_EYE_M, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(n);trace.append_arg(m);trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor & wrap_eye_out(c10::DispatchKeySet dispatchKeySet, int64_t n, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::eye_outf(dispatchKeySet, n, out);
  }
  bool flush = register_in_place(out, H_EYE_OUT, dispatchKeySet);
  trace.append_arg(n);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_eye_m_out(c10::DispatchKeySet dispatchKeySet, int64_t n, int64_t m, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::eye_outf(dispatchKeySet, n, m, out);
  }
  bool flush = register_in_place(out, H_EYE_M_OUT, dispatchKeySet);
  trace.append_arg(n);trace.append_arg(m);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_flatten_using_ints(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t start_dim, int64_t end_dim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::flatten(dispatchKeySet, self, start_dim, end_dim);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_FLATTEN_USING_INTS, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(start_dim);trace.append_arg(end_dim);
  return tt;
}

at::Tensor wrap_flatten_named_out_dim(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t start_dim, int64_t end_dim, at::Dimname out_dim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::flatten(dispatchKeySet, self, start_dim, end_dim, std::move(out_dim));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_FLATTEN_NAMED_OUT_DIM, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(start_dim);trace.append_arg(end_dim);trace.append_arg(std::move(out_dim));
  return tt;
}

at::Tensor wrap_flatten_using_names(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname start_dim, at::Dimname end_dim, at::Dimname out_dim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::flatten(dispatchKeySet, self, std::move(start_dim), std::move(end_dim), std::move(out_dim));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_FLATTEN_USING_NAMES, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(start_dim));trace.append_arg(std::move(end_dim));trace.append_arg(std::move(out_dim));
  return tt;
}

at::Tensor wrap_flatten_DimnameList(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::DimnameList dims, at::Dimname out_dim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::flatten(dispatchKeySet, self, std::move(dims), std::move(out_dim));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_FLATTEN_DIMNAMELIST, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dims));trace.append_arg(std::move(out_dim));
  return tt;
}

at::Tensor wrap_unflatten_int(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, at::IntArrayRef sizes, c10::optional<at::DimnameList> names) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::unflatten(dispatchKeySet, self, dim, std::move(sizes), std::move(names));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_UNFLATTEN_INT, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(std::move(sizes));trace.append_arg(std::move(names));
  return tt;
}

at::Tensor wrap_unflatten_Dimname(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, at::IntArrayRef sizes, at::DimnameList names) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::unflatten(dispatchKeySet, self, std::move(dim), std::move(sizes), std::move(names));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_UNFLATTEN_DIMNAME, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(std::move(sizes));trace.append_arg(std::move(names));
  return tt;
}

at::Tensor & wrap_fill__Scalar(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & value) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fill_(dispatchKeySet, self, value);
  }
  bool flush = register_in_place(self, H_FILL__SCALAR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(value);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_fill__Tensor(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & value) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fill_(dispatchKeySet, self, value);
  }
  bool flush = register_in_place(self, H_FILL__TENSOR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(value);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_floor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::floor(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_FLOOR, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_floor_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::floor_(dispatchKeySet, self);
  }
  bool flush = register_in_place(self, H_FLOOR_, dispatchKeySet);
  trace.append_arg(self);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_floor_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::floor_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_FLOOR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_floor_divide(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::floor_divide(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_FLOOR_DIVIDE, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_floor_divide__Tensor(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::floor_divide_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_FLOOR_DIVIDE__TENSOR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_floor_divide_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::floor_divide_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_FLOOR_DIVIDE_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_floor_divide_Scalar(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::floor_divide(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_FLOOR_DIVIDE_SCALAR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_floor_divide__Scalar(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::floor_divide_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_FLOOR_DIVIDE__SCALAR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_frac_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::frac_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_FRAC_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_full_names(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, const at::Scalar & fill_value, c10::optional<at::DimnameList> names, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::full(dispatchKeySet, std::move(size), fill_value, std::move(names), std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_FULL_NAMES, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(std::move(size));trace.append_arg(fill_value);trace.append_arg(std::move(names));trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor wrap_full(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, const at::Scalar & fill_value, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::full(dispatchKeySet, std::move(size), fill_value, std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_FULL, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(std::move(size));trace.append_arg(fill_value);trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor & wrap_full_out(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, const at::Scalar & fill_value, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::full_outf(dispatchKeySet, std::move(size), fill_value, out);
  }
  bool flush = register_in_place(out, H_FULL_OUT, dispatchKeySet);
  trace.append_arg(std::move(size));trace.append_arg(fill_value);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_full_like(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & fill_value, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::full_like(dispatchKeySet, self, fill_value, std::move(dtype), std::move(layout), std::move(device), pin_memory, std::move(memory_format));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_FULL_LIKE, dtype ? scalarTypeToTypeMeta(*dtype) : self.dtype(), device ? *device : self.device());
  trace.append_arg(self);trace.append_arg(fill_value);trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);trace.append_arg(std::move(memory_format));
  return tt;
}

at::Tensor wrap_from_file(c10::DispatchKeySet dispatchKeySet, c10::string_view filename, c10::optional<bool> shared, c10::optional<int64_t> size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::from_file(dispatchKeySet, std::move(filename), shared, size, std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_FROM_FILE, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(std::move(filename));trace.append_arg(shared);trace.append_arg(size);trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor & wrap_gcd_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::gcd_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_GCD_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_lcm_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::lcm_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_LCM_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_grid_sampler(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::grid_sampler(dispatchKeySet, input, grid, interpolation_mode, padding_mode, align_corners);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_GRID_SAMPLER, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(grid);trace.append_arg(interpolation_mode);trace.append_arg(padding_mode);trace.append_arg(align_corners);
  return tt;
}

at::Tensor wrap_grid_sampler_2d(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::grid_sampler_2d(dispatchKeySet, input, grid, interpolation_mode, padding_mode, align_corners);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_GRID_SAMPLER_2D, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(grid);trace.append_arg(interpolation_mode);trace.append_arg(padding_mode);trace.append_arg(align_corners);
  return tt;
}

at::Tensor wrap__grid_sampler_2d_cpu_fallback(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_grid_sampler_2d_cpu_fallback(dispatchKeySet, input, grid, interpolation_mode, padding_mode, align_corners);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__GRID_SAMPLER_2D_CPU_FALLBACK, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(grid);trace.append_arg(interpolation_mode);trace.append_arg(padding_mode);trace.append_arg(align_corners);
  return tt;
}

at::Tensor wrap_grid_sampler_3d(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::grid_sampler_3d(dispatchKeySet, input, grid, interpolation_mode, padding_mode, align_corners);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_GRID_SAMPLER_3D, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(grid);trace.append_arg(interpolation_mode);trace.append_arg(padding_mode);trace.append_arg(align_corners);
  return tt;
}

at::Tensor wrap_hann_window(c10::DispatchKeySet dispatchKeySet, int64_t window_length, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::hann_window(dispatchKeySet, window_length, std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_HANN_WINDOW, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(window_length);trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor wrap_hann_window_periodic(c10::DispatchKeySet dispatchKeySet, int64_t window_length, bool periodic, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::hann_window(dispatchKeySet, window_length, periodic, std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_HANN_WINDOW_PERIODIC, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(window_length);trace.append_arg(periodic);trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor wrap_hamming_window(c10::DispatchKeySet dispatchKeySet, int64_t window_length, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::hamming_window(dispatchKeySet, window_length, std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_HAMMING_WINDOW, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(window_length);trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor wrap_hamming_window_periodic(c10::DispatchKeySet dispatchKeySet, int64_t window_length, bool periodic, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::hamming_window(dispatchKeySet, window_length, periodic, std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_HAMMING_WINDOW_PERIODIC, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(window_length);trace.append_arg(periodic);trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor wrap_hamming_window_periodic_alpha(c10::DispatchKeySet dispatchKeySet, int64_t window_length, bool periodic, double alpha, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::hamming_window(dispatchKeySet, window_length, periodic, alpha, std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_HAMMING_WINDOW_PERIODIC_ALPHA, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(window_length);trace.append_arg(periodic);trace.append_arg(alpha);trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor wrap_hamming_window_periodic_alpha_beta(c10::DispatchKeySet dispatchKeySet, int64_t window_length, bool periodic, double alpha, double beta, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::hamming_window(dispatchKeySet, window_length, periodic, alpha, beta, std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_HAMMING_WINDOW_PERIODIC_ALPHA_BETA, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(window_length);trace.append_arg(periodic);trace.append_arg(alpha);trace.append_arg(beta);trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor wrap_kaiser_window(c10::DispatchKeySet dispatchKeySet, int64_t window_length, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::kaiser_window(dispatchKeySet, window_length, std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_KAISER_WINDOW, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(window_length);trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor wrap_kaiser_window_periodic(c10::DispatchKeySet dispatchKeySet, int64_t window_length, bool periodic, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::kaiser_window(dispatchKeySet, window_length, periodic, std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_KAISER_WINDOW_PERIODIC, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(window_length);trace.append_arg(periodic);trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor wrap_kaiser_window_beta(c10::DispatchKeySet dispatchKeySet, int64_t window_length, bool periodic, double beta, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::kaiser_window(dispatchKeySet, window_length, periodic, beta, std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_KAISER_WINDOW_BETA, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(window_length);trace.append_arg(periodic);trace.append_arg(beta);trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor wrap_hinge_embedding_loss(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, double margin, int64_t reduction) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::hinge_embedding_loss(dispatchKeySet, self, target, margin, reduction);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_HINGE_EMBEDDING_LOSS, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(target);trace.append_arg(margin);trace.append_arg(reduction);
  return tt;
}

at::Tensor wrap_group_norm(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, int64_t num_groups, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, double eps, bool cudnn_enabled) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::group_norm(dispatchKeySet, input, num_groups, weight, bias, eps, cudnn_enabled);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_GROUP_NORM, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(num_groups);trace.append_arg(weight);trace.append_arg(bias);trace.append_arg(eps);trace.append_arg(cudnn_enabled);
  return tt;
}

at::Tensor wrap__fft_r2c(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, int64_t normalization, bool onesided) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_fft_r2c(dispatchKeySet, self, std::move(dim), normalization, onesided);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__FFT_R2C, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(normalization);trace.append_arg(onesided);
  return tt;
}

at::Tensor & wrap__fft_r2c_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, int64_t normalization, bool onesided, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_fft_r2c_outf(dispatchKeySet, self, std::move(dim), normalization, onesided, out);
  }
  bool flush = register_in_place(out, H__FFT_R2C_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(normalization);trace.append_arg(onesided);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap__fft_c2r(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, int64_t normalization, int64_t last_dim_size) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_fft_c2r(dispatchKeySet, self, std::move(dim), normalization, last_dim_size);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__FFT_C2R, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(normalization);trace.append_arg(last_dim_size);
  return tt;
}

at::Tensor & wrap__fft_c2r_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, int64_t normalization, int64_t last_dim_size, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_fft_c2r_outf(dispatchKeySet, self, std::move(dim), normalization, last_dim_size, out);
  }
  bool flush = register_in_place(out, H__FFT_C2R_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(normalization);trace.append_arg(last_dim_size);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap__fft_c2c(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, int64_t normalization, bool forward) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_fft_c2c(dispatchKeySet, self, std::move(dim), normalization, forward);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__FFT_C2C, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(normalization);trace.append_arg(forward);
  return tt;
}

at::Tensor & wrap__fft_c2c_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, int64_t normalization, bool forward, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_fft_c2c_outf(dispatchKeySet, self, std::move(dim), normalization, forward, out);
  }
  bool flush = register_in_place(out, H__FFT_C2C_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(normalization);trace.append_arg(forward);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_index_Tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::List<c10::optional<at::Tensor>> & indices) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::index(dispatchKeySet, self, indices);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_INDEX_TENSOR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(indices);
  return tt;
}

at::Tensor & wrap_index_copy_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::index_copy_(dispatchKeySet, self, dim, index, source);
  }
  bool flush = register_in_place(self, H_INDEX_COPY_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(index);trace.append_arg(source);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_index_copy(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::index_copy(dispatchKeySet, self, dim, index, source);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_INDEX_COPY, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(index);trace.append_arg(source);
  return tt;
}

at::Tensor & wrap_index_copy__dimname(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & source) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::index_copy_(dispatchKeySet, self, std::move(dim), index, source);
  }
  bool flush = register_in_place(self, H_INDEX_COPY__DIMNAME, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(index);trace.append_arg(source);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_index_copy_dimname(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & source) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::index_copy(dispatchKeySet, self, std::move(dim), index, source);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_INDEX_COPY_DIMNAME, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(index);trace.append_arg(source);
  return tt;
}

at::Tensor & wrap_index_put_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const c10::List<c10::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::index_put_(dispatchKeySet, self, indices, values, accumulate);
  }
  bool flush = register_in_place(self, H_INDEX_PUT_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(indices);trace.append_arg(values);trace.append_arg(accumulate);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_index_put(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::List<c10::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::index_put(dispatchKeySet, self, indices, values, accumulate);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_INDEX_PUT, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(indices);trace.append_arg(values);trace.append_arg(accumulate);
  return tt;
}

at::Tensor & wrap__index_put_impl_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const c10::List<c10::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate, bool unsafe) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_index_put_impl_(dispatchKeySet, self, indices, values, accumulate, unsafe);
  }
  bool flush = register_in_place(self, H__INDEX_PUT_IMPL_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(indices);trace.append_arg(values);trace.append_arg(accumulate);trace.append_arg(unsafe);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_instance_norm(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, bool use_input_stats, double momentum, double eps, bool cudnn_enabled) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::instance_norm(dispatchKeySet, input, weight, bias, running_mean, running_var, use_input_stats, momentum, eps, cudnn_enabled);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_INSTANCE_NORM, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(weight);trace.append_arg(bias);trace.append_arg(running_mean);trace.append_arg(running_var);trace.append_arg(use_input_stats);trace.append_arg(momentum);trace.append_arg(eps);trace.append_arg(cudnn_enabled);
  return tt;
}

at::Tensor wrap_inverse(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::inverse(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_INVERSE, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_inverse_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::inverse_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_INVERSE_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap__inverse_helper(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_inverse_helper(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__INVERSE_HELPER, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor wrap_isclose(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, double rtol, double atol, bool equal_nan) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::isclose(dispatchKeySet, self, other, rtol, atol, equal_nan);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ISCLOSE, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(rtol);trace.append_arg(atol);trace.append_arg(equal_nan);
  return tt;
}

at::Tensor & wrap_isin_Tensor_Tensor_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & elements, const at::Tensor & test_elements, bool assume_unique, bool invert, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::isin_outf(dispatchKeySet, elements, test_elements, assume_unique, invert, out);
  }
  bool flush = register_in_place(out, H_ISIN_TENSOR_TENSOR_OUT, dispatchKeySet);
  trace.append_arg(elements);trace.append_arg(test_elements);trace.append_arg(assume_unique);trace.append_arg(invert);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_isin_Tensor_Scalar_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & elements, const at::Scalar & test_element, bool assume_unique, bool invert, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::isin_outf(dispatchKeySet, elements, test_element, assume_unique, invert, out);
  }
  bool flush = register_in_place(out, H_ISIN_TENSOR_SCALAR_OUT, dispatchKeySet);
  trace.append_arg(elements);trace.append_arg(test_element);trace.append_arg(assume_unique);trace.append_arg(invert);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_isin_Scalar_Tensor_out(c10::DispatchKeySet dispatchKeySet, const at::Scalar & element, const at::Tensor & test_elements, bool assume_unique, bool invert, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::isin_outf(dispatchKeySet, element, test_elements, assume_unique, invert, out);
  }
  bool flush = register_in_place(out, H_ISIN_SCALAR_TENSOR_OUT, dispatchKeySet);
  trace.append_arg(element);trace.append_arg(test_elements);trace.append_arg(assume_unique);trace.append_arg(invert);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_isnan(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::isnan(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ISNAN, scalarTypeToTypeMeta(kBool), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor wrap_isreal(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::isreal(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ISREAL, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor wrap_kl_div(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, int64_t reduction, bool log_target) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::kl_div(dispatchKeySet, self, target, reduction, log_target);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_KL_DIV, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(target);trace.append_arg(reduction);trace.append_arg(log_target);
  return tt;
}

at::Tensor wrap_kl_div_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, bool log_target) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::kl_div_backward(dispatchKeySet, grad_output, self, target, reduction, log_target);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_KL_DIV_BACKWARD, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(target);trace.append_arg(reduction);trace.append_arg(log_target);
  return tt;
}

at::Tensor wrap_kron(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::kron(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_KRON, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_kron_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::kron_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_KRON_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_layer_norm(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, at::IntArrayRef normalized_shape, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, double eps, bool cudnn_enable) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::layer_norm(dispatchKeySet, input, std::move(normalized_shape), weight, bias, eps, cudnn_enable);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LAYER_NORM, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(std::move(normalized_shape));trace.append_arg(weight);trace.append_arg(bias);trace.append_arg(eps);trace.append_arg(cudnn_enable);
  return tt;
}

at::Tensor wrap_nan_to_num(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<double> nan, c10::optional<double> posinf, c10::optional<double> neginf) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::nan_to_num(dispatchKeySet, self, nan, posinf, neginf);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_NAN_TO_NUM, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(nan);trace.append_arg(posinf);trace.append_arg(neginf);
  return tt;
}

at::Tensor & wrap_nan_to_num_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, c10::optional<double> nan, c10::optional<double> posinf, c10::optional<double> neginf) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::nan_to_num_(dispatchKeySet, self, nan, posinf, neginf);
  }
  bool flush = register_in_place(self, H_NAN_TO_NUM_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(nan);trace.append_arg(posinf);trace.append_arg(neginf);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_nan_to_num_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<double> nan, c10::optional<double> posinf, c10::optional<double> neginf, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::nan_to_num_outf(dispatchKeySet, self, nan, posinf, neginf, out);
  }
  bool flush = register_in_place(out, H_NAN_TO_NUM_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(nan);trace.append_arg(posinf);trace.append_arg(neginf);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_linear(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::linear(dispatchKeySet, input, weight, bias);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LINEAR, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(weight);trace.append_arg(bias);
  return tt;
}

at::Tensor wrap_mkldnn_linear(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, const c10::optional<at::Tensor> & bias) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::mkldnn_linear(dispatchKeySet, self, weight, bias);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MKLDNN_LINEAR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(weight);trace.append_arg(bias);
  return tt;
}

at::Tensor wrap_mkldnn_linear_backward_input(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef input_size, const at::Tensor & grad_output, const at::Tensor & weight) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::mkldnn_linear_backward_input(dispatchKeySet, std::move(input_size), grad_output, weight);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MKLDNN_LINEAR_BACKWARD_INPUT, grad_output.dtype(), grad_output.device());
  trace.append_arg(std::move(input_size));trace.append_arg(grad_output);trace.append_arg(weight);
  return tt;
}

at::Tensor wrap_fbgemm_linear_int8_weight_fp32_activation(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & weight, const at::Tensor & packed, const at::Tensor & col_offsets, const at::Scalar & weight_scale, const at::Scalar & weight_zero_point, const at::Tensor & bias) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fbgemm_linear_int8_weight_fp32_activation(dispatchKeySet, input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_FBGEMM_LINEAR_INT8_WEIGHT_FP32_ACTIVATION, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(weight);trace.append_arg(packed);trace.append_arg(col_offsets);trace.append_arg(weight_scale);trace.append_arg(weight_zero_point);trace.append_arg(bias);
  return tt;
}

at::Tensor wrap_fbgemm_linear_int8_weight(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & weight, const at::Tensor & packed, const at::Tensor & col_offsets, const at::Scalar & weight_scale, const at::Scalar & weight_zero_point, const at::Tensor & bias) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fbgemm_linear_int8_weight(dispatchKeySet, input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_FBGEMM_LINEAR_INT8_WEIGHT, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(weight);trace.append_arg(packed);trace.append_arg(col_offsets);trace.append_arg(weight_scale);trace.append_arg(weight_zero_point);trace.append_arg(bias);
  return tt;
}

at::Tensor wrap_fbgemm_pack_gemm_matrix_fp16(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fbgemm_pack_gemm_matrix_fp16(dispatchKeySet, input);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_FBGEMM_PACK_GEMM_MATRIX_FP16, input.dtype(), input.device());
  trace.append_arg(input);
  return tt;
}

at::Tensor wrap_fbgemm_linear_fp16_weight_fp32_activation(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & packed_weight, const at::Tensor & bias) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fbgemm_linear_fp16_weight_fp32_activation(dispatchKeySet, input, packed_weight, bias);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_FBGEMM_LINEAR_FP16_WEIGHT_FP32_ACTIVATION, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(packed_weight);trace.append_arg(bias);
  return tt;
}

at::Tensor wrap_fbgemm_linear_fp16_weight(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & packed_weight, const at::Tensor & bias) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fbgemm_linear_fp16_weight(dispatchKeySet, input, packed_weight, bias);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_FBGEMM_LINEAR_FP16_WEIGHT, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(packed_weight);trace.append_arg(bias);
  return tt;
}

at::Tensor wrap_fbgemm_pack_quantized_matrix(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fbgemm_pack_quantized_matrix(dispatchKeySet, input);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_FBGEMM_PACK_QUANTIZED_MATRIX, input.dtype(), input.device());
  trace.append_arg(input);
  return tt;
}

at::Tensor wrap_fbgemm_pack_quantized_matrix_KN(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, int64_t K, int64_t N) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fbgemm_pack_quantized_matrix(dispatchKeySet, input, K, N);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_FBGEMM_PACK_QUANTIZED_MATRIX_KN, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(K);trace.append_arg(N);
  return tt;
}

at::Tensor wrap_ldexp_Tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::ldexp(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LDEXP_TENSOR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_ldexp_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::ldexp_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_LDEXP_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_ldexp_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::ldexp_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_LDEXP_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_linspace(c10::DispatchKeySet dispatchKeySet, const at::Scalar & start, const at::Scalar & end, c10::optional<int64_t> steps, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::linspace(dispatchKeySet, start, end, steps, std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_LINSPACE, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(start);trace.append_arg(end);trace.append_arg(steps);trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor & wrap_linspace_out(c10::DispatchKeySet dispatchKeySet, const at::Scalar & start, const at::Scalar & end, c10::optional<int64_t> steps, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::linspace_outf(dispatchKeySet, start, end, steps, out);
  }
  bool flush = register_in_place(out, H_LINSPACE_OUT, dispatchKeySet);
  trace.append_arg(start);trace.append_arg(end);trace.append_arg(steps);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_log_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::log_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_LOG_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_log10_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::log10_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_LOG10_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_log1p(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::log1p(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LOG1P, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_log1p_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::log1p_(dispatchKeySet, self);
  }
  bool flush = register_in_place(self, H_LOG1P_, dispatchKeySet);
  trace.append_arg(self);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_log1p_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::log1p_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_LOG1P_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_log2_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::log2_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_LOG2_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_logaddexp_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::logaddexp_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_LOGADDEXP_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_logaddexp(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::logaddexp(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LOGADDEXP, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_logaddexp2_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::logaddexp2_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_LOGADDEXP2_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_logaddexp2(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::logaddexp2(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LOGADDEXP2, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor wrap_xlogy_Tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::xlogy(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_XLOGY_TENSOR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor wrap_xlogy_Scalar_Self(c10::DispatchKeySet dispatchKeySet, const at::Scalar & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::xlogy(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_XLOGY_SCALAR_SELF, other.dtype(), other.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor wrap_xlogy_Scalar_Other(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::xlogy(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_XLOGY_SCALAR_OTHER, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_xlogy__Tensor(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::xlogy_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_XLOGY__TENSOR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_xlogy__Scalar_Other(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::xlogy_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_XLOGY__SCALAR_OTHER, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_xlogy_OutTensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::xlogy_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_XLOGY_OUTTENSOR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_xlogy_OutScalar_Self(c10::DispatchKeySet dispatchKeySet, const at::Scalar & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::xlogy_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_XLOGY_OUTSCALAR_SELF, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_xlogy_OutScalar_Other(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::xlogy_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_XLOGY_OUTSCALAR_OTHER, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_logdet(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::logdet(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LOGDET, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor wrap_logspace(c10::DispatchKeySet dispatchKeySet, const at::Scalar & start, const at::Scalar & end, c10::optional<int64_t> steps, double base, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::logspace(dispatchKeySet, start, end, steps, base, std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_LOGSPACE, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(start);trace.append_arg(end);trace.append_arg(steps);trace.append_arg(base);trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor & wrap_logspace_out(c10::DispatchKeySet dispatchKeySet, const at::Scalar & start, const at::Scalar & end, c10::optional<int64_t> steps, double base, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::logspace_outf(dispatchKeySet, start, end, steps, base, out);
  }
  bool flush = register_in_place(out, H_LOGSPACE_OUT, dispatchKeySet);
  trace.append_arg(start);trace.append_arg(end);trace.append_arg(steps);trace.append_arg(base);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_log_softmax_int(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::log_softmax(dispatchKeySet, self, dim, std::move(dtype));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LOG_SOFTMAX_INT, dtype ? scalarTypeToTypeMeta(*dtype) : self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(std::move(dtype));
  return tt;
}

at::Tensor wrap_log_softmax_Dimname(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::log_softmax(dispatchKeySet, self, std::move(dim), std::move(dtype));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LOG_SOFTMAX_DIMNAME, dtype ? scalarTypeToTypeMeta(*dtype) : self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(std::move(dtype));
  return tt;
}

at::Tensor wrap__log_softmax(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, bool half_to_float) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_log_softmax(dispatchKeySet, self, dim, half_to_float);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__LOG_SOFTMAX, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(half_to_float);
  return tt;
}

at::Tensor wrap__log_softmax_backward_data(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_log_softmax_backward_data(dispatchKeySet, grad_output, output, dim, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__LOG_SOFTMAX_BACKWARD_DATA, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(output);trace.append_arg(dim);trace.append_arg(self);
  return tt;
}

at::Tensor wrap__logcumsumexp(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_logcumsumexp(dispatchKeySet, self, dim);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__LOGCUMSUMEXP, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(dim);
  return tt;
}

at::Tensor & wrap__logcumsumexp_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_logcumsumexp_outf(dispatchKeySet, self, dim, out);
  }
  bool flush = register_in_place(out, H__LOGCUMSUMEXP_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_logcumsumexp(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::logcumsumexp(dispatchKeySet, self, dim);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LOGCUMSUMEXP, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(dim);
  return tt;
}

at::Tensor & wrap_logcumsumexp_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::logcumsumexp_outf(dispatchKeySet, self, dim, out);
  }
  bool flush = register_in_place(out, H_LOGCUMSUMEXP_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_logcumsumexp_dimname(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::logcumsumexp(dispatchKeySet, self, std::move(dim));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LOGCUMSUMEXP_DIMNAME, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim));
  return tt;
}

at::Tensor & wrap_logcumsumexp_dimname_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::logcumsumexp_outf(dispatchKeySet, self, std::move(dim), out);
  }
  bool flush = register_in_place(out, H_LOGCUMSUMEXP_DIMNAME_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_logsumexp(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, bool keepdim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::logsumexp(dispatchKeySet, self, std::move(dim), keepdim);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LOGSUMEXP, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(keepdim);
  return tt;
}

at::Tensor & wrap_logsumexp_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, bool keepdim, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::logsumexp_outf(dispatchKeySet, self, std::move(dim), keepdim, out);
  }
  bool flush = register_in_place(out, H_LOGSUMEXP_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(keepdim);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_logsumexp_names(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::DimnameList dim, bool keepdim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::logsumexp(dispatchKeySet, self, std::move(dim), keepdim);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LOGSUMEXP_NAMES, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(keepdim);
  return tt;
}

at::Tensor & wrap_logsumexp_names_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::DimnameList dim, bool keepdim, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::logsumexp_outf(dispatchKeySet, self, std::move(dim), keepdim, out);
  }
  bool flush = register_in_place(out, H_LOGSUMEXP_NAMES_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(keepdim);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_margin_ranking_loss(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input1, const at::Tensor & input2, const at::Tensor & target, double margin, int64_t reduction) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::margin_ranking_loss(dispatchKeySet, input1, input2, target, margin, reduction);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MARGIN_RANKING_LOSS, input1.dtype(), input1.device());
  trace.append_arg(input1);trace.append_arg(input2);trace.append_arg(target);trace.append_arg(margin);trace.append_arg(reduction);
  return tt;
}

at::Tensor wrap_matmul(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::matmul(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MATMUL, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_matmul_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::matmul_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_MATMUL_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_matrix_rank_tol(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, double tol, bool symmetric) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::matrix_rank(dispatchKeySet, self, tol, symmetric);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MATRIX_RANK_TOL, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(tol);trace.append_arg(symmetric);
  return tt;
}

at::Tensor wrap_matrix_rank(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool symmetric) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::matrix_rank(dispatchKeySet, self, symmetric);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MATRIX_RANK, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(symmetric);
  return tt;
}

at::Tensor wrap_matrix_power(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t n) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::matrix_power(dispatchKeySet, self, n);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MATRIX_POWER, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(n);
  return tt;
}

at::Tensor & wrap_matrix_power_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t n, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::matrix_power_outf(dispatchKeySet, self, n, out);
  }
  bool flush = register_in_place(out, H_MATRIX_POWER_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(n);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_matrix_exp(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::matrix_exp(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MATRIX_EXP, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor wrap_matrix_exp_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & grad) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::matrix_exp_backward(dispatchKeySet, self, grad);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MATRIX_EXP_BACKWARD, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(grad);
  return tt;
}

at::Tensor wrap__compute_linear_combination(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & coefficients) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_compute_linear_combination(dispatchKeySet, input, coefficients);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__COMPUTE_LINEAR_COMBINATION, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(coefficients);
  return tt;
}

at::Tensor & wrap__compute_linear_combination_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & coefficients, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_compute_linear_combination_outf(dispatchKeySet, input, coefficients, out);
  }
  bool flush = register_in_place(out, H__COMPUTE_LINEAR_COMBINATION_OUT, dispatchKeySet);
  trace.append_arg(input);trace.append_arg(coefficients);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_value_selecting_reduction_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, int64_t dim, const at::Tensor & indices, at::IntArrayRef sizes, bool keepdim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::value_selecting_reduction_backward(dispatchKeySet, grad, dim, indices, std::move(sizes), keepdim);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_VALUE_SELECTING_REDUCTION_BACKWARD, grad.dtype(), grad.device());
  trace.append_arg(grad);trace.append_arg(dim);trace.append_arg(indices);trace.append_arg(std::move(sizes));trace.append_arg(keepdim);
  return tt;
}

at::Tensor wrap_amax(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, bool keepdim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::amax(dispatchKeySet, self, std::move(dim), keepdim);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_AMAX, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(keepdim);
  return tt;
}

at::Tensor & wrap_amax_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, bool keepdim, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::amax_outf(dispatchKeySet, self, std::move(dim), keepdim, out);
  }
  bool flush = register_in_place(out, H_AMAX_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(keepdim);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_max_pool1d(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::max_pool1d(dispatchKeySet, self, std::move(kernel_size), std::move(stride), std::move(padding), std::move(dilation), ceil_mode);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MAX_POOL1D, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(kernel_size));trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(std::move(dilation));trace.append_arg(ceil_mode);
  return tt;
}

at::Tensor wrap_max_pool2d(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::max_pool2d(dispatchKeySet, self, std::move(kernel_size), std::move(stride), std::move(padding), std::move(dilation), ceil_mode);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MAX_POOL2D, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(kernel_size));trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(std::move(dilation));trace.append_arg(ceil_mode);
  return tt;
}

at::Tensor wrap_mkldnn_max_pool2d(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::mkldnn_max_pool2d(dispatchKeySet, self, std::move(kernel_size), std::move(stride), std::move(padding), std::move(dilation), ceil_mode);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MKLDNN_MAX_POOL2D, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(kernel_size));trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(std::move(dilation));trace.append_arg(ceil_mode);
  return tt;
}

at::Tensor wrap_mkldnn_max_pool2d_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & output, const at::Tensor & input, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::mkldnn_max_pool2d_backward(dispatchKeySet, grad_output, output, input, std::move(kernel_size), std::move(stride), std::move(padding), std::move(dilation), ceil_mode);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MKLDNN_MAX_POOL2D_BACKWARD, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(output);trace.append_arg(input);trace.append_arg(std::move(kernel_size));trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(std::move(dilation));trace.append_arg(ceil_mode);
  return tt;
}

at::Tensor wrap_mkldnn_max_pool3d(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::mkldnn_max_pool3d(dispatchKeySet, self, std::move(kernel_size), std::move(stride), std::move(padding), std::move(dilation), ceil_mode);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MKLDNN_MAX_POOL3D, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(kernel_size));trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(std::move(dilation));trace.append_arg(ceil_mode);
  return tt;
}

at::Tensor wrap_mkldnn_max_pool3d_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & output, const at::Tensor & input, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::mkldnn_max_pool3d_backward(dispatchKeySet, grad_output, output, input, std::move(kernel_size), std::move(stride), std::move(padding), std::move(dilation), ceil_mode);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MKLDNN_MAX_POOL3D_BACKWARD, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(output);trace.append_arg(input);trace.append_arg(std::move(kernel_size));trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(std::move(dilation));trace.append_arg(ceil_mode);
  return tt;
}

at::Tensor wrap_quantized_max_pool1d(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::quantized_max_pool1d(dispatchKeySet, self, std::move(kernel_size), std::move(stride), std::move(padding), std::move(dilation), ceil_mode);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_QUANTIZED_MAX_POOL1D, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(kernel_size));trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(std::move(dilation));trace.append_arg(ceil_mode);
  return tt;
}

at::Tensor wrap_quantized_max_pool2d(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::quantized_max_pool2d(dispatchKeySet, self, std::move(kernel_size), std::move(stride), std::move(padding), std::move(dilation), ceil_mode);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_QUANTIZED_MAX_POOL2D, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(kernel_size));trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(std::move(dilation));trace.append_arg(ceil_mode);
  return tt;
}

at::Tensor wrap_max_pool3d(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::max_pool3d(dispatchKeySet, self, std::move(kernel_size), std::move(stride), std::move(padding), std::move(dilation), ceil_mode);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MAX_POOL3D, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(kernel_size));trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(std::move(dilation));trace.append_arg(ceil_mode);
  return tt;
}

at::Tensor wrap_mean(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::mean(dispatchKeySet, self, std::move(dtype));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MEAN, dtype ? scalarTypeToTypeMeta(*dtype) : self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dtype));
  return tt;
}

at::Tensor wrap_mean_dim(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::mean(dispatchKeySet, self, std::move(dim), keepdim, std::move(dtype));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MEAN_DIM, dtype ? scalarTypeToTypeMeta(*dtype) : self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(keepdim);trace.append_arg(std::move(dtype));
  return tt;
}

at::Tensor & wrap_mean_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::mean_outf(dispatchKeySet, self, std::move(dim), keepdim, std::move(dtype), out);
  }
  bool flush = register_in_place(out, H_MEAN_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(keepdim);trace.append_arg(std::move(dtype));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_mean_names_dim(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::DimnameList dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::mean(dispatchKeySet, self, std::move(dim), keepdim, std::move(dtype));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MEAN_NAMES_DIM, dtype ? scalarTypeToTypeMeta(*dtype) : self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(keepdim);trace.append_arg(std::move(dtype));
  return tt;
}

at::Tensor & wrap_mean_names_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::DimnameList dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::mean_outf(dispatchKeySet, self, std::move(dim), keepdim, std::move(dtype), out);
  }
  bool flush = register_in_place(out, H_MEAN_NAMES_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(keepdim);trace.append_arg(std::move(dtype));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_median(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::median(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MEDIAN, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor wrap_nanmedian(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::nanmedian(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_NANMEDIAN, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor wrap_amin(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, bool keepdim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::amin(dispatchKeySet, self, std::move(dim), keepdim);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_AMIN, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(keepdim);
  return tt;
}

at::Tensor & wrap_amin_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, bool keepdim, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::amin_outf(dispatchKeySet, self, std::move(dim), keepdim, out);
  }
  bool flush = register_in_place(out, H_AMIN_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(keepdim);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_mkldnn_convolution(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::mkldnn_convolution(dispatchKeySet, self, weight, bias, std::move(padding), std::move(stride), std::move(dilation), groups);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MKLDNN_CONVOLUTION, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(weight);trace.append_arg(bias);trace.append_arg(std::move(padding));trace.append_arg(std::move(stride));trace.append_arg(std::move(dilation));trace.append_arg(groups);
  return tt;
}

at::Tensor wrap_mkldnn_convolution_backward_input(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef self_size, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool bias_defined) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::mkldnn_convolution_backward_input(dispatchKeySet, std::move(self_size), grad_output, weight, std::move(padding), std::move(stride), std::move(dilation), groups, bias_defined);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MKLDNN_CONVOLUTION_BACKWARD_INPUT, grad_output.dtype(), grad_output.device());
  trace.append_arg(std::move(self_size));trace.append_arg(grad_output);trace.append_arg(weight);trace.append_arg(std::move(padding));trace.append_arg(std::move(stride));trace.append_arg(std::move(dilation));trace.append_arg(groups);trace.append_arg(bias_defined);
  return tt;
}

at::Tensor wrap_miopen_convolution(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::miopen_convolution(dispatchKeySet, self, weight, bias, std::move(padding), std::move(stride), std::move(dilation), groups, benchmark, deterministic);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MIOPEN_CONVOLUTION, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(weight);trace.append_arg(bias);trace.append_arg(std::move(padding));trace.append_arg(std::move(stride));trace.append_arg(std::move(dilation));trace.append_arg(groups);trace.append_arg(benchmark);trace.append_arg(deterministic);
  return tt;
}

at::Tensor wrap_miopen_convolution_backward_input(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef self_size, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::miopen_convolution_backward_input(dispatchKeySet, std::move(self_size), grad_output, weight, std::move(padding), std::move(stride), std::move(dilation), groups, benchmark, deterministic);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MIOPEN_CONVOLUTION_BACKWARD_INPUT, grad_output.dtype(), grad_output.device());
  trace.append_arg(std::move(self_size));trace.append_arg(grad_output);trace.append_arg(weight);trace.append_arg(std::move(padding));trace.append_arg(std::move(stride));trace.append_arg(std::move(dilation));trace.append_arg(groups);trace.append_arg(benchmark);trace.append_arg(deterministic);
  return tt;
}

at::Tensor wrap_miopen_convolution_backward_bias(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::miopen_convolution_backward_bias(dispatchKeySet, grad_output);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MIOPEN_CONVOLUTION_BACKWARD_BIAS, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);
  return tt;
}

at::Tensor wrap_miopen_convolution_backward_weight(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef weight_size, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::miopen_convolution_backward_weight(dispatchKeySet, std::move(weight_size), grad_output, self, std::move(padding), std::move(stride), std::move(dilation), groups, benchmark, deterministic);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MIOPEN_CONVOLUTION_BACKWARD_WEIGHT, grad_output.dtype(), grad_output.device());
  trace.append_arg(std::move(weight_size));trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(std::move(padding));trace.append_arg(std::move(stride));trace.append_arg(std::move(dilation));trace.append_arg(groups);trace.append_arg(benchmark);trace.append_arg(deterministic);
  return tt;
}

at::Tensor wrap_miopen_convolution_transpose(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::miopen_convolution_transpose(dispatchKeySet, self, weight, bias, std::move(padding), std::move(output_padding), std::move(stride), std::move(dilation), groups, benchmark, deterministic);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MIOPEN_CONVOLUTION_TRANSPOSE, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(weight);trace.append_arg(bias);trace.append_arg(std::move(padding));trace.append_arg(std::move(output_padding));trace.append_arg(std::move(stride));trace.append_arg(std::move(dilation));trace.append_arg(groups);trace.append_arg(benchmark);trace.append_arg(deterministic);
  return tt;
}

at::Tensor wrap_miopen_convolution_transpose_backward_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::miopen_convolution_transpose_backward_input(dispatchKeySet, grad_output, weight, std::move(padding), std::move(stride), std::move(dilation), groups, benchmark, deterministic);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MIOPEN_CONVOLUTION_TRANSPOSE_BACKWARD_INPUT, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(weight);trace.append_arg(std::move(padding));trace.append_arg(std::move(stride));trace.append_arg(std::move(dilation));trace.append_arg(groups);trace.append_arg(benchmark);trace.append_arg(deterministic);
  return tt;
}

at::Tensor wrap_miopen_convolution_transpose_backward_weight(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef weight_size, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::miopen_convolution_transpose_backward_weight(dispatchKeySet, std::move(weight_size), grad_output, self, std::move(padding), std::move(stride), std::move(dilation), groups, benchmark, deterministic);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MIOPEN_CONVOLUTION_TRANSPOSE_BACKWARD_WEIGHT, grad_output.dtype(), grad_output.device());
  trace.append_arg(std::move(weight_size));trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(std::move(padding));trace.append_arg(std::move(stride));trace.append_arg(std::move(dilation));trace.append_arg(groups);trace.append_arg(benchmark);trace.append_arg(deterministic);
  return tt;
}

at::Tensor wrap_miopen_depthwise_convolution(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::miopen_depthwise_convolution(dispatchKeySet, self, weight, bias, std::move(padding), std::move(stride), std::move(dilation), groups, benchmark, deterministic);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MIOPEN_DEPTHWISE_CONVOLUTION, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(weight);trace.append_arg(bias);trace.append_arg(std::move(padding));trace.append_arg(std::move(stride));trace.append_arg(std::move(dilation));trace.append_arg(groups);trace.append_arg(benchmark);trace.append_arg(deterministic);
  return tt;
}

at::Tensor wrap_miopen_depthwise_convolution_backward_input(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef self_size, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::miopen_depthwise_convolution_backward_input(dispatchKeySet, std::move(self_size), grad_output, weight, std::move(padding), std::move(stride), std::move(dilation), groups, benchmark, deterministic);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MIOPEN_DEPTHWISE_CONVOLUTION_BACKWARD_INPUT, grad_output.dtype(), grad_output.device());
  trace.append_arg(std::move(self_size));trace.append_arg(grad_output);trace.append_arg(weight);trace.append_arg(std::move(padding));trace.append_arg(std::move(stride));trace.append_arg(std::move(dilation));trace.append_arg(groups);trace.append_arg(benchmark);trace.append_arg(deterministic);
  return tt;
}

at::Tensor wrap_miopen_depthwise_convolution_backward_weight(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef weight_size, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::miopen_depthwise_convolution_backward_weight(dispatchKeySet, std::move(weight_size), grad_output, self, std::move(padding), std::move(stride), std::move(dilation), groups, benchmark, deterministic);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MIOPEN_DEPTHWISE_CONVOLUTION_BACKWARD_WEIGHT, grad_output.dtype(), grad_output.device());
  trace.append_arg(std::move(weight_size));trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(std::move(padding));trace.append_arg(std::move(stride));trace.append_arg(std::move(dilation));trace.append_arg(groups);trace.append_arg(benchmark);trace.append_arg(deterministic);
  return tt;
}

at::Tensor wrap_mm(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & mat2) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::mm(dispatchKeySet, self, mat2);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MM, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(mat2);
  return tt;
}

at::Tensor & wrap_mm_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & mat2, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::mm_outf(dispatchKeySet, self, mat2, out);
  }
  bool flush = register_in_place(out, H_MM_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(mat2);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap__sparse_mm(c10::DispatchKeySet dispatchKeySet, const at::Tensor & sparse, const at::Tensor & dense) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_sparse_mm(dispatchKeySet, sparse, dense);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__SPARSE_MM, sparse.dtype(), sparse.device());
  trace.append_arg(sparse);trace.append_arg(dense);
  return tt;
}

at::Tensor wrap__sparse_sparse_matmul(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_sparse_sparse_matmul(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__SPARSE_SPARSE_MATMUL, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor wrap__sparse_mask_helper(c10::DispatchKeySet dispatchKeySet, const at::Tensor & t, const at::Tensor & mask_indices) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_sparse_mask_helper(dispatchKeySet, t, mask_indices);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__SPARSE_MASK_HELPER, t.dtype(), t.device());
  trace.append_arg(t);trace.append_arg(mask_indices);
  return tt;
}

at::Tensor wrap_mul_Tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::mul(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MUL_TENSOR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_mul__Tensor(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::mul_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_MUL__TENSOR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_mul_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::mul_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_MUL_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_mul_Scalar(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::mul(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MUL_SCALAR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_mul__Scalar(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::mul_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_MUL__SCALAR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_multiply_Tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::multiply(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MULTIPLY_TENSOR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_multiply__Tensor(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::multiply_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_MULTIPLY__TENSOR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_multiply_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::multiply_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_MULTIPLY_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_multiply_Scalar(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::multiply(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MULTIPLY_SCALAR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_multiply__Scalar(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::multiply_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_MULTIPLY__SCALAR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_mv(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & vec) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::mv(dispatchKeySet, self, vec);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MV, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(vec);
  return tt;
}

at::Tensor & wrap_mv_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & vec, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::mv_outf(dispatchKeySet, self, vec, out);
  }
  bool flush = register_in_place(out, H_MV_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(vec);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_mvlgamma(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t p) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::mvlgamma(dispatchKeySet, self, p);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MVLGAMMA, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(p);
  return tt;
}

at::Tensor & wrap_mvlgamma_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, int64_t p) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::mvlgamma_(dispatchKeySet, self, p);
  }
  bool flush = register_in_place(self, H_MVLGAMMA_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(p);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_narrow_copy(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, int64_t start, int64_t length) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::narrow_copy(dispatchKeySet, self, dim, start, length);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_NARROW_COPY, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(start);trace.append_arg(length);
  return tt;
}

at::Tensor & wrap_narrow_copy_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, int64_t start, int64_t length, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::narrow_copy_outf(dispatchKeySet, self, dim, start, length, out);
  }
  bool flush = register_in_place(out, H_NARROW_COPY_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(start);trace.append_arg(length);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_narrow(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, int64_t start, int64_t length) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::narrow(dispatchKeySet, self, dim, start, length);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_NARROW, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(start);trace.append_arg(length);
  return tt;
}

at::Tensor wrap_narrow_Tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, const at::Tensor & start, int64_t length) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::narrow(dispatchKeySet, self, dim, start, length);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_NARROW_TENSOR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(start);trace.append_arg(length);
  return tt;
}

at::Tensor wrap_batch_norm_elemt(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const at::Tensor & mean, const at::Tensor & invstd, double eps) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::batch_norm_elemt(dispatchKeySet, input, weight, bias, mean, invstd, eps);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_BATCH_NORM_ELEMT, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(weight);trace.append_arg(bias);trace.append_arg(mean);trace.append_arg(invstd);trace.append_arg(eps);
  return tt;
}

at::Tensor & wrap_batch_norm_elemt_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const at::Tensor & mean, const at::Tensor & invstd, double eps, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::batch_norm_elemt_outf(dispatchKeySet, input, weight, bias, mean, invstd, eps, out);
  }
  bool flush = register_in_place(out, H_BATCH_NORM_ELEMT_OUT, dispatchKeySet);
  trace.append_arg(input);trace.append_arg(weight);trace.append_arg(bias);trace.append_arg(mean);trace.append_arg(invstd);trace.append_arg(eps);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_batch_norm_backward_elemt(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_out, const at::Tensor & input, const at::Tensor & mean, const at::Tensor & invstd, const c10::optional<at::Tensor> & weight, const at::Tensor & mean_dy, const at::Tensor & mean_dy_xmu, const at::Tensor & count) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::batch_norm_backward_elemt(dispatchKeySet, grad_out, input, mean, invstd, weight, mean_dy, mean_dy_xmu, count);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_BATCH_NORM_BACKWARD_ELEMT, grad_out.dtype(), grad_out.device());
  trace.append_arg(grad_out);trace.append_arg(input);trace.append_arg(mean);trace.append_arg(invstd);trace.append_arg(weight);trace.append_arg(mean_dy);trace.append_arg(mean_dy_xmu);trace.append_arg(count);
  return tt;
}

at::Tensor wrap__nnpack_spatial_convolution(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef padding, at::IntArrayRef stride) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_nnpack_spatial_convolution(dispatchKeySet, input, weight, bias, std::move(padding), std::move(stride));
  }
  auto tt = register_new_tensor(dispatchKeySet, H__NNPACK_SPATIAL_CONVOLUTION, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(weight);trace.append_arg(bias);trace.append_arg(std::move(padding));trace.append_arg(std::move(stride));
  return tt;
}

at::Tensor wrap__nnpack_spatial_convolution_backward_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_nnpack_spatial_convolution_backward_input(dispatchKeySet, input, grad_output, weight, std::move(padding));
  }
  auto tt = register_new_tensor(dispatchKeySet, H__NNPACK_SPATIAL_CONVOLUTION_BACKWARD_INPUT, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(grad_output);trace.append_arg(weight);trace.append_arg(std::move(padding));
  return tt;
}

at::Tensor wrap__nnpack_spatial_convolution_backward_weight(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, at::IntArrayRef weightsize, const at::Tensor & grad_output, at::IntArrayRef padding) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_nnpack_spatial_convolution_backward_weight(dispatchKeySet, input, std::move(weightsize), grad_output, std::move(padding));
  }
  auto tt = register_new_tensor(dispatchKeySet, H__NNPACK_SPATIAL_CONVOLUTION_BACKWARD_WEIGHT, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(std::move(weightsize));trace.append_arg(grad_output);trace.append_arg(std::move(padding));
  return tt;
}

at::Tensor wrap_ones_names(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, c10::optional<at::DimnameList> names, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::ones(dispatchKeySet, std::move(size), std::move(names), std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_ONES_NAMES, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(std::move(size));trace.append_arg(std::move(names));trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor wrap_ones(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::ones(dispatchKeySet, std::move(size), std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_ONES, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(std::move(size));trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor & wrap_ones_out(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::ones_outf(dispatchKeySet, std::move(size), out);
  }
  bool flush = register_in_place(out, H_ONES_OUT, dispatchKeySet);
  trace.append_arg(std::move(size));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_ones_like(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::ones_like(dispatchKeySet, self, std::move(dtype), std::move(layout), std::move(device), pin_memory, std::move(memory_format));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ONES_LIKE, dtype ? scalarTypeToTypeMeta(*dtype) : self.dtype(), device ? *device : self.device());
  trace.append_arg(self);trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);trace.append_arg(std::move(memory_format));
  return tt;
}

at::Tensor wrap_pairwise_distance(c10::DispatchKeySet dispatchKeySet, const at::Tensor & x1, const at::Tensor & x2, double p, double eps, bool keepdim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::pairwise_distance(dispatchKeySet, x1, x2, p, eps, keepdim);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_PAIRWISE_DISTANCE, x1.dtype(), x1.device());
  trace.append_arg(x1);trace.append_arg(x2);trace.append_arg(p);trace.append_arg(eps);trace.append_arg(keepdim);
  return tt;
}

at::Tensor wrap_cdist(c10::DispatchKeySet dispatchKeySet, const at::Tensor & x1, const at::Tensor & x2, double p, c10::optional<int64_t> compute_mode) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::cdist(dispatchKeySet, x1, x2, p, compute_mode);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CDIST, x1.dtype(), x1.device());
  trace.append_arg(x1);trace.append_arg(x2);trace.append_arg(p);trace.append_arg(compute_mode);
  return tt;
}

at::Tensor wrap__euclidean_dist(c10::DispatchKeySet dispatchKeySet, const at::Tensor & x1, const at::Tensor & x2) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_euclidean_dist(dispatchKeySet, x1, x2);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__EUCLIDEAN_DIST, x1.dtype(), x1.device());
  trace.append_arg(x1);trace.append_arg(x2);
  return tt;
}

at::Tensor wrap__cdist_forward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & x1, const at::Tensor & x2, double p, c10::optional<int64_t> compute_mode) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_cdist_forward(dispatchKeySet, x1, x2, p, compute_mode);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__CDIST_FORWARD, x1.dtype(), x1.device());
  trace.append_arg(x1);trace.append_arg(x2);trace.append_arg(p);trace.append_arg(compute_mode);
  return tt;
}

at::Tensor wrap__cdist_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, const at::Tensor & x1, const at::Tensor & x2, double p, const at::Tensor & cdist) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_cdist_backward(dispatchKeySet, grad, x1, x2, p, cdist);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__CDIST_BACKWARD, grad.dtype(), grad.device());
  trace.append_arg(grad);trace.append_arg(x1);trace.append_arg(x2);trace.append_arg(p);trace.append_arg(cdist);
  return tt;
}

at::Tensor wrap_pdist(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, double p) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::pdist(dispatchKeySet, self, p);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_PDIST, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(p);
  return tt;
}

at::Tensor wrap__pdist_forward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, double p) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_pdist_forward(dispatchKeySet, self, p);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__PDIST_FORWARD, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(p);
  return tt;
}

at::Tensor wrap__pdist_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, const at::Tensor & self, double p, const at::Tensor & pdist) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_pdist_backward(dispatchKeySet, grad, self, p, pdist);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__PDIST_BACKWARD, grad.dtype(), grad.device());
  trace.append_arg(grad);trace.append_arg(self);trace.append_arg(p);trace.append_arg(pdist);
  return tt;
}

at::Tensor wrap_cosine_similarity(c10::DispatchKeySet dispatchKeySet, const at::Tensor & x1, const at::Tensor & x2, int64_t dim, double eps) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::cosine_similarity(dispatchKeySet, x1, x2, dim, eps);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_COSINE_SIMILARITY, x1.dtype(), x1.device());
  trace.append_arg(x1);trace.append_arg(x2);trace.append_arg(dim);trace.append_arg(eps);
  return tt;
}

at::Tensor wrap_permute(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dims) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::permute(dispatchKeySet, self, std::move(dims));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_PERMUTE, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dims));
  return tt;
}

at::Tensor wrap_movedim_intlist(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef source, at::IntArrayRef destination) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::movedim(dispatchKeySet, self, std::move(source), std::move(destination));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MOVEDIM_INTLIST, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(source));trace.append_arg(std::move(destination));
  return tt;
}

at::Tensor wrap_movedim_int(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t source, int64_t destination) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::movedim(dispatchKeySet, self, source, destination);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MOVEDIM_INT, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(source);trace.append_arg(destination);
  return tt;
}

at::Tensor wrap_moveaxis_intlist(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef source, at::IntArrayRef destination) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::moveaxis(dispatchKeySet, self, std::move(source), std::move(destination));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MOVEAXIS_INTLIST, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(source));trace.append_arg(std::move(destination));
  return tt;
}

at::Tensor wrap_moveaxis_int(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t source, int64_t destination) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::moveaxis(dispatchKeySet, self, source, destination);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MOVEAXIS_INT, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(source);trace.append_arg(destination);
  return tt;
}

at::Tensor wrap_numpy_T(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::numpy_T(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_NUMPY_T, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor wrap_pixel_shuffle(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t upscale_factor) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::pixel_shuffle(dispatchKeySet, self, upscale_factor);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_PIXEL_SHUFFLE, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(upscale_factor);
  return tt;
}

at::Tensor wrap_pixel_unshuffle(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t downscale_factor) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::pixel_unshuffle(dispatchKeySet, self, downscale_factor);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_PIXEL_UNSHUFFLE, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(downscale_factor);
  return tt;
}

at::Tensor wrap_channel_shuffle(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t groups) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::channel_shuffle(dispatchKeySet, self, groups);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CHANNEL_SHUFFLE, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(groups);
  return tt;
}

at::Tensor wrap_pin_memory(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::pin_memory(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_PIN_MEMORY, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor wrap_pinverse(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, double rcond) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::pinverse(dispatchKeySet, self, rcond);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_PINVERSE, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(rcond);
  return tt;
}

at::Tensor wrap_poisson_nll_loss(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & target, bool log_input, bool full, double eps, int64_t reduction) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::poisson_nll_loss(dispatchKeySet, input, target, log_input, full, eps, reduction);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_POISSON_NLL_LOSS, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(target);trace.append_arg(log_input);trace.append_arg(full);trace.append_arg(eps);trace.append_arg(reduction);
  return tt;
}

at::Tensor wrap_rad2deg(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::rad2deg(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_RAD2DEG, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_rad2deg_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::rad2deg_(dispatchKeySet, self);
  }
  bool flush = register_in_place(self, H_RAD2DEG_, dispatchKeySet);
  trace.append_arg(self);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_rad2deg_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::rad2deg_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_RAD2DEG_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_deg2rad(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::deg2rad(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_DEG2RAD, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_deg2rad_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::deg2rad_(dispatchKeySet, self);
  }
  bool flush = register_in_place(self, H_DEG2RAD_, dispatchKeySet);
  trace.append_arg(self);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_deg2rad_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::deg2rad_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_DEG2RAD_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_scalar_tensor(c10::DispatchKeySet dispatchKeySet, const at::Scalar & s, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::scalar_tensor(dispatchKeySet, s, std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_SCALAR_TENSOR, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(s);trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor wrap_rand_names(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, c10::optional<at::DimnameList> names, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::rand(dispatchKeySet, std::move(size), std::move(names), std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_RAND_NAMES, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(std::move(size));trace.append_arg(std::move(names));trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor wrap_rand_generator_with_names(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, c10::optional<at::Generator> generator, c10::optional<at::DimnameList> names, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::rand(dispatchKeySet, std::move(size), std::move(generator), std::move(names), std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_RAND_GENERATOR_WITH_NAMES, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(std::move(size));trace.append_arg(std::move(generator));trace.append_arg(std::move(names));trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor wrap_rand(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::rand(dispatchKeySet, std::move(size), std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_RAND, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(std::move(size));trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor wrap_rand_generator(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, c10::optional<at::Generator> generator, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::rand(dispatchKeySet, std::move(size), std::move(generator), std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_RAND_GENERATOR, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(std::move(size));trace.append_arg(std::move(generator));trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor & wrap_rand_out(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::rand_outf(dispatchKeySet, std::move(size), out);
  }
  bool flush = register_in_place(out, H_RAND_OUT, dispatchKeySet);
  trace.append_arg(std::move(size));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_rand_generator_out(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, c10::optional<at::Generator> generator, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::rand_outf(dispatchKeySet, std::move(size), std::move(generator), out);
  }
  bool flush = register_in_place(out, H_RAND_GENERATOR_OUT, dispatchKeySet);
  trace.append_arg(std::move(size));trace.append_arg(std::move(generator));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_rand_like(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::rand_like(dispatchKeySet, self, std::move(dtype), std::move(layout), std::move(device), pin_memory, std::move(memory_format));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_RAND_LIKE, dtype ? scalarTypeToTypeMeta(*dtype) : self.dtype(), device ? *device : self.device());
  trace.append_arg(self);trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);trace.append_arg(std::move(memory_format));
  return tt;
}

at::Tensor wrap_randint(c10::DispatchKeySet dispatchKeySet, int64_t high, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::randint(dispatchKeySet, high, std::move(size), std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_RANDINT, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(high);trace.append_arg(std::move(size));trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor wrap_randint_generator(c10::DispatchKeySet dispatchKeySet, int64_t high, at::IntArrayRef size, c10::optional<at::Generator> generator, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::randint(dispatchKeySet, high, std::move(size), std::move(generator), std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_RANDINT_GENERATOR, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(high);trace.append_arg(std::move(size));trace.append_arg(std::move(generator));trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor wrap_randint_low(c10::DispatchKeySet dispatchKeySet, int64_t low, int64_t high, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::randint(dispatchKeySet, low, high, std::move(size), std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_RANDINT_LOW, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(low);trace.append_arg(high);trace.append_arg(std::move(size));trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor wrap_randint_low_generator(c10::DispatchKeySet dispatchKeySet, int64_t low, int64_t high, at::IntArrayRef size, c10::optional<at::Generator> generator, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::randint(dispatchKeySet, low, high, std::move(size), std::move(generator), std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_RANDINT_LOW_GENERATOR, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(low);trace.append_arg(high);trace.append_arg(std::move(size));trace.append_arg(std::move(generator));trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor & wrap_randint_out(c10::DispatchKeySet dispatchKeySet, int64_t high, at::IntArrayRef size, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::randint_outf(dispatchKeySet, high, std::move(size), out);
  }
  bool flush = register_in_place(out, H_RANDINT_OUT, dispatchKeySet);
  trace.append_arg(high);trace.append_arg(std::move(size));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_randint_generator_out(c10::DispatchKeySet dispatchKeySet, int64_t high, at::IntArrayRef size, c10::optional<at::Generator> generator, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::randint_outf(dispatchKeySet, high, std::move(size), std::move(generator), out);
  }
  bool flush = register_in_place(out, H_RANDINT_GENERATOR_OUT, dispatchKeySet);
  trace.append_arg(high);trace.append_arg(std::move(size));trace.append_arg(std::move(generator));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_randint_low_out(c10::DispatchKeySet dispatchKeySet, int64_t low, int64_t high, at::IntArrayRef size, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::randint_outf(dispatchKeySet, low, high, std::move(size), out);
  }
  bool flush = register_in_place(out, H_RANDINT_LOW_OUT, dispatchKeySet);
  trace.append_arg(low);trace.append_arg(high);trace.append_arg(std::move(size));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_randint_low_generator_out(c10::DispatchKeySet dispatchKeySet, int64_t low, int64_t high, at::IntArrayRef size, c10::optional<at::Generator> generator, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::randint_outf(dispatchKeySet, low, high, std::move(size), std::move(generator), out);
  }
  bool flush = register_in_place(out, H_RANDINT_LOW_GENERATOR_OUT, dispatchKeySet);
  trace.append_arg(low);trace.append_arg(high);trace.append_arg(std::move(size));trace.append_arg(std::move(generator));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_randint_like(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t high, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::randint_like(dispatchKeySet, self, high, std::move(dtype), std::move(layout), std::move(device), pin_memory, std::move(memory_format));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_RANDINT_LIKE, dtype ? scalarTypeToTypeMeta(*dtype) : self.dtype(), device ? *device : self.device());
  trace.append_arg(self);trace.append_arg(high);trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);trace.append_arg(std::move(memory_format));
  return tt;
}

at::Tensor wrap_randint_like_low_dtype(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t low, int64_t high, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::randint_like(dispatchKeySet, self, low, high, std::move(dtype), std::move(layout), std::move(device), pin_memory, std::move(memory_format));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_RANDINT_LIKE_LOW_DTYPE, dtype ? scalarTypeToTypeMeta(*dtype) : self.dtype(), device ? *device : self.device());
  trace.append_arg(self);trace.append_arg(low);trace.append_arg(high);trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);trace.append_arg(std::move(memory_format));
  return tt;
}

at::Tensor wrap_randn(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::randn(dispatchKeySet, std::move(size), std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_RANDN, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(std::move(size));trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor wrap_randn_generator(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, c10::optional<at::Generator> generator, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::randn(dispatchKeySet, std::move(size), std::move(generator), std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_RANDN_GENERATOR, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(std::move(size));trace.append_arg(std::move(generator));trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor wrap_randn_names(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, c10::optional<at::DimnameList> names, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::randn(dispatchKeySet, std::move(size), std::move(names), std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_RANDN_NAMES, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(std::move(size));trace.append_arg(std::move(names));trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor wrap_randn_generator_with_names(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, c10::optional<at::Generator> generator, c10::optional<at::DimnameList> names, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::randn(dispatchKeySet, std::move(size), std::move(generator), std::move(names), std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_RANDN_GENERATOR_WITH_NAMES, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(std::move(size));trace.append_arg(std::move(generator));trace.append_arg(std::move(names));trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor & wrap_randn_out(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::randn_outf(dispatchKeySet, std::move(size), out);
  }
  bool flush = register_in_place(out, H_RANDN_OUT, dispatchKeySet);
  trace.append_arg(std::move(size));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_randn_generator_out(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, c10::optional<at::Generator> generator, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::randn_outf(dispatchKeySet, std::move(size), std::move(generator), out);
  }
  bool flush = register_in_place(out, H_RANDN_GENERATOR_OUT, dispatchKeySet);
  trace.append_arg(std::move(size));trace.append_arg(std::move(generator));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_randn_like(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::randn_like(dispatchKeySet, self, std::move(dtype), std::move(layout), std::move(device), pin_memory, std::move(memory_format));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_RANDN_LIKE, dtype ? scalarTypeToTypeMeta(*dtype) : self.dtype(), device ? *device : self.device());
  trace.append_arg(self);trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);trace.append_arg(std::move(memory_format));
  return tt;
}

at::Tensor wrap_randperm(c10::DispatchKeySet dispatchKeySet, int64_t n, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::randperm(dispatchKeySet, n, std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_RANDPERM, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(n);trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor wrap_randperm_generator(c10::DispatchKeySet dispatchKeySet, int64_t n, c10::optional<at::Generator> generator, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::randperm(dispatchKeySet, n, std::move(generator), std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_RANDPERM_GENERATOR, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(n);trace.append_arg(std::move(generator));trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor & wrap_randperm_out(c10::DispatchKeySet dispatchKeySet, int64_t n, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::randperm_outf(dispatchKeySet, n, out);
  }
  bool flush = register_in_place(out, H_RANDPERM_OUT, dispatchKeySet);
  trace.append_arg(n);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_randperm_generator_out(c10::DispatchKeySet dispatchKeySet, int64_t n, c10::optional<at::Generator> generator, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::randperm_outf(dispatchKeySet, n, std::move(generator), out);
  }
  bool flush = register_in_place(out, H_RANDPERM_GENERATOR_OUT, dispatchKeySet);
  trace.append_arg(n);trace.append_arg(std::move(generator));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_range_step(c10::DispatchKeySet dispatchKeySet, const at::Scalar & start, const at::Scalar & end, const at::Scalar & step, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::range(dispatchKeySet, start, end, step, std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_RANGE_STEP, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(start);trace.append_arg(end);trace.append_arg(step);trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor wrap_range(c10::DispatchKeySet dispatchKeySet, const at::Scalar & start, const at::Scalar & end, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::range(dispatchKeySet, start, end, std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_RANGE, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(start);trace.append_arg(end);trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor & wrap_range_out(c10::DispatchKeySet dispatchKeySet, const at::Scalar & start, const at::Scalar & end, const at::Scalar & step, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::range_outf(dispatchKeySet, start, end, step, out);
  }
  bool flush = register_in_place(out, H_RANGE_OUT, dispatchKeySet);
  trace.append_arg(start);trace.append_arg(end);trace.append_arg(step);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_ravel(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::ravel(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_RAVEL, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_reciprocal_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::reciprocal_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_RECIPROCAL_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_neg(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::neg(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_NEG, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_neg_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::neg_(dispatchKeySet, self);
  }
  bool flush = register_in_place(self, H_NEG_, dispatchKeySet);
  trace.append_arg(self);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_neg_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::neg_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_NEG_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_negative(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::negative(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_NEGATIVE, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_negative_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::negative_(dispatchKeySet, self);
  }
  bool flush = register_in_place(self, H_NEGATIVE_, dispatchKeySet);
  trace.append_arg(self);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_negative_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::negative_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_NEGATIVE_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_repeat(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef repeats) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::repeat(dispatchKeySet, self, std::move(repeats));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_REPEAT, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(repeats));
  return tt;
}

at::Tensor wrap_repeat_interleave_Tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & repeats, c10::optional<int64_t> output_size) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::repeat_interleave(dispatchKeySet, repeats, output_size);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_REPEAT_INTERLEAVE_TENSOR, repeats.dtype(), repeats.device());
  trace.append_arg(repeats);trace.append_arg(output_size);
  return tt;
}

at::Tensor wrap_repeat_interleave_self_Tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & repeats, c10::optional<int64_t> dim, c10::optional<int64_t> output_size) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::repeat_interleave(dispatchKeySet, self, repeats, dim, output_size);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_REPEAT_INTERLEAVE_SELF_TENSOR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(repeats);trace.append_arg(dim);trace.append_arg(output_size);
  return tt;
}

at::Tensor wrap_repeat_interleave_self_int(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t repeats, c10::optional<int64_t> dim, c10::optional<int64_t> output_size) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::repeat_interleave(dispatchKeySet, self, repeats, dim, output_size);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_REPEAT_INTERLEAVE_SELF_INT, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(repeats);trace.append_arg(dim);trace.append_arg(output_size);
  return tt;
}

at::Tensor wrap_reshape(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef shape) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::reshape(dispatchKeySet, self, std::move(shape));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_RESHAPE, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(shape));
  return tt;
}

at::Tensor wrap__mkldnn_reshape(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef shape) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_mkldnn_reshape(dispatchKeySet, self, std::move(shape));
  }
  auto tt = register_new_tensor(dispatchKeySet, H__MKLDNN_RESHAPE, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(shape));
  return tt;
}

at::Tensor wrap_reshape_as(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::reshape_as(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_RESHAPE_AS, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_round_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::round_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_ROUND_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_rrelu(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & lower, const at::Scalar & upper, bool training, c10::optional<at::Generator> generator) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::rrelu(dispatchKeySet, self, lower, upper, training, std::move(generator));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_RRELU, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(lower);trace.append_arg(upper);trace.append_arg(training);trace.append_arg(std::move(generator));
  return tt;
}

at::Tensor & wrap_rrelu_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & lower, const at::Scalar & upper, bool training, c10::optional<at::Generator> generator) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::rrelu_(dispatchKeySet, self, lower, upper, training, std::move(generator));
  }
  bool flush = register_in_place(self, H_RRELU_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(lower);trace.append_arg(upper);trace.append_arg(training);trace.append_arg(std::move(generator));
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_relu(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::relu(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_RELU, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_relu_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::relu_(dispatchKeySet, self);
  }
  bool flush = register_in_place(self, H_RELU_, dispatchKeySet);
  trace.append_arg(self);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_relu6(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::relu6(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_RELU6, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_relu6_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::relu6_(dispatchKeySet, self);
  }
  bool flush = register_in_place(self, H_RELU6_, dispatchKeySet);
  trace.append_arg(self);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_prelu(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::prelu(dispatchKeySet, self, weight);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_PRELU, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(weight);
  return tt;
}

at::Tensor & wrap_gelu_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::gelu_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_GELU_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_gelu(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::gelu(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_GELU, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_gelu_backward_grad_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, const at::Tensor & self, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::gelu_backward_outf(dispatchKeySet, grad, self, grad_input);
  }
  bool flush = register_in_place(grad_input, H_GELU_BACKWARD_GRAD_INPUT, dispatchKeySet);
  trace.append_arg(grad);trace.append_arg(self);trace.append_arg(grad_input);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return grad_input;
}

at::Tensor wrap_infinitely_differentiable_gelu_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::infinitely_differentiable_gelu_backward(dispatchKeySet, grad, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_INFINITELY_DIFFERENTIABLE_GELU_BACKWARD, grad.dtype(), grad.device());
  trace.append_arg(grad);trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_hardshrink_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & lambd, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::hardshrink_outf(dispatchKeySet, self, lambd, out);
  }
  bool flush = register_in_place(out, H_HARDSHRINK_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(lambd);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_hardshrink_backward_grad_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_out, const at::Tensor & self, const at::Scalar & lambd, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::hardshrink_backward_outf(dispatchKeySet, grad_out, self, lambd, grad_input);
  }
  bool flush = register_in_place(grad_input, H_HARDSHRINK_BACKWARD_GRAD_INPUT, dispatchKeySet);
  trace.append_arg(grad_out);trace.append_arg(self);trace.append_arg(lambd);trace.append_arg(grad_input);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return grad_input;
}

at::Tensor & wrap_rsqrt_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::rsqrt_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_RSQRT_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_select_Dimname(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, int64_t index) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::select(dispatchKeySet, self, std::move(dim), index);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SELECT_DIMNAME, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(index);
  return tt;
}

at::Tensor wrap_select_int(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, int64_t index) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::select(dispatchKeySet, self, dim, index);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SELECT_INT, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(index);
  return tt;
}

at::Tensor wrap_select_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, at::IntArrayRef input_sizes, int64_t dim, int64_t index) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::select_backward(dispatchKeySet, grad, std::move(input_sizes), dim, index);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SELECT_BACKWARD, grad.dtype(), grad.device());
  trace.append_arg(grad);trace.append_arg(std::move(input_sizes));trace.append_arg(dim);trace.append_arg(index);
  return tt;
}

at::Tensor wrap_selu(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::selu(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SELU, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_selu_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::selu_(dispatchKeySet, self);
  }
  bool flush = register_in_place(self, H_SELU_, dispatchKeySet);
  trace.append_arg(self);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_celu(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::celu(dispatchKeySet, self, alpha);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CELU, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(alpha);
  return tt;
}

at::Tensor & wrap_celu_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::celu_(dispatchKeySet, self, alpha);
  }
  bool flush = register_in_place(self, H_CELU_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(alpha);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_silu(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::silu(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SILU, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_silu_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::silu_(dispatchKeySet, self);
  }
  bool flush = register_in_place(self, H_SILU_, dispatchKeySet);
  trace.append_arg(self);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_silu_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::silu_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_SILU_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_silu_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::silu_backward(dispatchKeySet, grad_output, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SILU_BACKWARD, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(self);
  return tt;
}

at::Tensor wrap_mish(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::mish(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MISH, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_mish_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::mish_(dispatchKeySet, self);
  }
  bool flush = register_in_place(self, H_MISH_, dispatchKeySet);
  trace.append_arg(self);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_mish_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::mish_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_MISH_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_mish_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::mish_backward(dispatchKeySet, grad_output, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MISH_BACKWARD, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(self);
  return tt;
}

at::Tensor wrap_sigmoid(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::sigmoid(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SIGMOID, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_sigmoid_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::sigmoid_(dispatchKeySet, self);
  }
  bool flush = register_in_place(self, H_SIGMOID_, dispatchKeySet);
  trace.append_arg(self);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_sigmoid_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::sigmoid_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_SIGMOID_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_logit(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<double> eps) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::logit(dispatchKeySet, self, eps);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LOGIT, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(eps);
  return tt;
}

at::Tensor & wrap_logit_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, c10::optional<double> eps) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::logit_(dispatchKeySet, self, eps);
  }
  bool flush = register_in_place(self, H_LOGIT_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(eps);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_logit_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<double> eps, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::logit_outf(dispatchKeySet, self, eps, out);
  }
  bool flush = register_in_place(out, H_LOGIT_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(eps);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_sin_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::sin_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_SIN_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_sinc_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::sinc_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_SINC_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_sinh_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::sinh_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_SINH_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_detach(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::detach(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_DETACH, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_detach_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::detach_(dispatchKeySet, self);
  }
  bool flush = register_in_place(self, H_DETACH_, dispatchKeySet);
  trace.append_arg(self);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_slice_Tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, c10::optional<int64_t> start, c10::optional<int64_t> end, int64_t step) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::slice(dispatchKeySet, self, dim, start, end, step);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SLICE_TENSOR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(start);trace.append_arg(end);trace.append_arg(step);
  return tt;
}

at::Tensor wrap_slice_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, at::IntArrayRef input_sizes, int64_t dim, int64_t start, int64_t end, int64_t step) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::slice_backward(dispatchKeySet, grad, std::move(input_sizes), dim, start, end, step);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SLICE_BACKWARD, grad.dtype(), grad.device());
  trace.append_arg(grad);trace.append_arg(std::move(input_sizes));trace.append_arg(dim);trace.append_arg(start);trace.append_arg(end);trace.append_arg(step);
  return tt;
}

at::Tensor wrap_smm(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & mat2) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::smm(dispatchKeySet, self, mat2);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SMM, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(mat2);
  return tt;
}

at::Tensor wrap_softmax_int(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::softmax(dispatchKeySet, self, dim, std::move(dtype));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SOFTMAX_INT, dtype ? scalarTypeToTypeMeta(*dtype) : self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(std::move(dtype));
  return tt;
}

at::Tensor wrap_softmax_Dimname(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::softmax(dispatchKeySet, self, std::move(dim), std::move(dtype));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SOFTMAX_DIMNAME, dtype ? scalarTypeToTypeMeta(*dtype) : self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(std::move(dtype));
  return tt;
}

at::Tensor wrap__softmax(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, bool half_to_float) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_softmax(dispatchKeySet, self, dim, half_to_float);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__SOFTMAX, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(half_to_float);
  return tt;
}

at::Tensor wrap__softmax_backward_data(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_softmax_backward_data(dispatchKeySet, grad_output, output, dim, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__SOFTMAX_BACKWARD_DATA, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(output);trace.append_arg(dim);trace.append_arg(self);
  return tt;
}

at::Tensor wrap_squeeze(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::squeeze(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SQUEEZE, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor wrap_squeeze_dim(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::squeeze(dispatchKeySet, self, dim);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SQUEEZE_DIM, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(dim);
  return tt;
}

at::Tensor wrap_squeeze_dimname(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::squeeze(dispatchKeySet, self, std::move(dim));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SQUEEZE_DIMNAME, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim));
  return tt;
}

at::Tensor & wrap_squeeze_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::squeeze_(dispatchKeySet, self);
  }
  bool flush = register_in_place(self, H_SQUEEZE_, dispatchKeySet);
  trace.append_arg(self);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_squeeze__dim(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, int64_t dim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::squeeze_(dispatchKeySet, self, dim);
  }
  bool flush = register_in_place(self, H_SQUEEZE__DIM, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(dim);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_squeeze__dimname(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, at::Dimname dim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::squeeze_(dispatchKeySet, self, std::move(dim));
  }
  bool flush = register_in_place(self, H_SQUEEZE__DIMNAME, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(dim));
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_sspaddmm(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::sspaddmm(dispatchKeySet, self, mat1, mat2, beta, alpha);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SSPADDMM, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(mat1);trace.append_arg(mat2);trace.append_arg(beta);trace.append_arg(alpha);
  return tt;
}

at::Tensor & wrap_sspaddmm_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::sspaddmm_outf(dispatchKeySet, self, mat1, mat2, beta, alpha, out);
  }
  bool flush = register_in_place(out, H_SSPADDMM_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(mat1);trace.append_arg(mat2);trace.append_arg(beta);trace.append_arg(alpha);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_stack(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors, int64_t dim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::stack(dispatchKeySet, std::move(tensors), dim);
  }
  auto defaults = compute_dtype(tensors);
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_STACK, default_dtype, default_device);
  trace.append_arg(std::move(tensors));trace.append_arg(dim);
  return tt;
}

at::Tensor & wrap_stack_out(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors, int64_t dim, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::stack_outf(dispatchKeySet, std::move(tensors), dim, out);
  }
  bool flush = register_in_place(out, H_STACK_OUT, dispatchKeySet);
  trace.append_arg(std::move(tensors));trace.append_arg(dim);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap__stack(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors, int64_t dim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_stack(dispatchKeySet, std::move(tensors), dim);
  }
  auto defaults = compute_dtype(tensors);
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H__STACK, default_dtype, default_device);
  trace.append_arg(std::move(tensors));trace.append_arg(dim);
  return tt;
}

at::Tensor & wrap__stack_out(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors, int64_t dim, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_stack_outf(dispatchKeySet, std::move(tensors), dim, out);
  }
  bool flush = register_in_place(out, H__STACK_OUT, dispatchKeySet);
  trace.append_arg(std::move(tensors));trace.append_arg(dim);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_hstack(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::hstack(dispatchKeySet, std::move(tensors));
  }
  auto defaults = compute_dtype(tensors);
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_HSTACK, default_dtype, default_device);
  trace.append_arg(std::move(tensors));
  return tt;
}

at::Tensor & wrap_hstack_out(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::hstack_outf(dispatchKeySet, std::move(tensors), out);
  }
  bool flush = register_in_place(out, H_HSTACK_OUT, dispatchKeySet);
  trace.append_arg(std::move(tensors));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_vstack(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::vstack(dispatchKeySet, std::move(tensors));
  }
  auto defaults = compute_dtype(tensors);
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_VSTACK, default_dtype, default_device);
  trace.append_arg(std::move(tensors));
  return tt;
}

at::Tensor & wrap_vstack_out(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::vstack_outf(dispatchKeySet, std::move(tensors), out);
  }
  bool flush = register_in_place(out, H_VSTACK_OUT, dispatchKeySet);
  trace.append_arg(std::move(tensors));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_dstack(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::dstack(dispatchKeySet, std::move(tensors));
  }
  auto defaults = compute_dtype(tensors);
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_DSTACK, default_dtype, default_device);
  trace.append_arg(std::move(tensors));
  return tt;
}

at::Tensor & wrap_dstack_out(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::dstack_outf(dispatchKeySet, std::move(tensors), out);
  }
  bool flush = register_in_place(out, H_DSTACK_OUT, dispatchKeySet);
  trace.append_arg(std::move(tensors));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_stft(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t n_fft, c10::optional<int64_t> hop_length, c10::optional<int64_t> win_length, const c10::optional<at::Tensor> & window, bool normalized, c10::optional<bool> onesided, c10::optional<bool> return_complex) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::stft(dispatchKeySet, self, n_fft, hop_length, win_length, window, normalized, onesided, return_complex);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_STFT, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(n_fft);trace.append_arg(hop_length);trace.append_arg(win_length);trace.append_arg(window);trace.append_arg(normalized);trace.append_arg(onesided);trace.append_arg(return_complex);
  return tt;
}

at::Tensor wrap_istft(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t n_fft, c10::optional<int64_t> hop_length, c10::optional<int64_t> win_length, const c10::optional<at::Tensor> & window, bool center, bool normalized, c10::optional<bool> onesided, c10::optional<int64_t> length, bool return_complex) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::istft(dispatchKeySet, self, n_fft, hop_length, win_length, window, center, normalized, onesided, length, return_complex);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ISTFT, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(n_fft);trace.append_arg(hop_length);trace.append_arg(win_length);trace.append_arg(window);trace.append_arg(center);trace.append_arg(normalized);trace.append_arg(onesided);trace.append_arg(length);trace.append_arg(return_complex);
  return tt;
}

at::Tensor wrap_sum(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::sum(dispatchKeySet, self, std::move(dtype));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SUM, dtype ? scalarTypeToTypeMeta(*dtype) : self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dtype));
  return tt;
}

at::Tensor wrap_sum_dim_IntList(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::sum(dispatchKeySet, self, std::move(dim), keepdim, std::move(dtype));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SUM_DIM_INTLIST, dtype ? scalarTypeToTypeMeta(*dtype) : self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(keepdim);trace.append_arg(std::move(dtype));
  return tt;
}

at::Tensor wrap_sum_dim_DimnameList(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::DimnameList dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::sum(dispatchKeySet, self, std::move(dim), keepdim, std::move(dtype));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SUM_DIM_DIMNAMELIST, dtype ? scalarTypeToTypeMeta(*dtype) : self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(keepdim);trace.append_arg(std::move(dtype));
  return tt;
}

at::Tensor & wrap_sum_IntList_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::sum_outf(dispatchKeySet, self, std::move(dim), keepdim, std::move(dtype), out);
  }
  bool flush = register_in_place(out, H_SUM_INTLIST_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(keepdim);trace.append_arg(std::move(dtype));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_sum_DimnameList_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::DimnameList dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::sum_outf(dispatchKeySet, self, std::move(dim), keepdim, std::move(dtype), out);
  }
  bool flush = register_in_place(out, H_SUM_DIMNAMELIST_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(keepdim);trace.append_arg(std::move(dtype));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_nansum(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::nansum(dispatchKeySet, self, std::move(dtype));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_NANSUM, dtype ? scalarTypeToTypeMeta(*dtype) : self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dtype));
  return tt;
}

at::Tensor wrap_nansum_dim_IntList(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::nansum(dispatchKeySet, self, std::move(dim), keepdim, std::move(dtype));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_NANSUM_DIM_INTLIST, dtype ? scalarTypeToTypeMeta(*dtype) : self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(keepdim);trace.append_arg(std::move(dtype));
  return tt;
}

at::Tensor & wrap_nansum_IntList_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::nansum_outf(dispatchKeySet, self, std::move(dim), keepdim, std::move(dtype), out);
  }
  bool flush = register_in_place(out, H_NANSUM_INTLIST_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(keepdim);trace.append_arg(std::move(dtype));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_sum_to_size(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef size) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::sum_to_size(dispatchKeySet, self, std::move(size));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SUM_TO_SIZE, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(size));
  return tt;
}

at::Tensor wrap_sqrt(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::sqrt(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SQRT, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_sqrt_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::sqrt_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_SQRT_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_square(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::square(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SQUARE, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_square_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::square_(dispatchKeySet, self);
  }
  bool flush = register_in_place(self, H_SQUARE_, dispatchKeySet);
  trace.append_arg(self);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_square_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::square_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_SQUARE_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_std(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool unbiased) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::std(dispatchKeySet, self, unbiased);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_STD, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(unbiased);
  return tt;
}

at::Tensor wrap_std_dim(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, bool unbiased, bool keepdim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::std(dispatchKeySet, self, std::move(dim), unbiased, keepdim);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_STD_DIM, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(unbiased);trace.append_arg(keepdim);
  return tt;
}

at::Tensor wrap_std_correction(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::IntArrayRef> dim, c10::optional<int64_t> correction, bool keepdim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::std(dispatchKeySet, self, std::move(dim), correction, keepdim);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_STD_CORRECTION, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(correction);trace.append_arg(keepdim);
  return tt;
}

at::Tensor & wrap_std_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, bool unbiased, bool keepdim, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::std_outf(dispatchKeySet, self, std::move(dim), unbiased, keepdim, out);
  }
  bool flush = register_in_place(out, H_STD_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(unbiased);trace.append_arg(keepdim);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_std_correction_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::IntArrayRef> dim, c10::optional<int64_t> correction, bool keepdim, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::std_outf(dispatchKeySet, self, std::move(dim), correction, keepdim, out);
  }
  bool flush = register_in_place(out, H_STD_CORRECTION_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(correction);trace.append_arg(keepdim);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_std_names_dim(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::DimnameList dim, bool unbiased, bool keepdim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::std(dispatchKeySet, self, std::move(dim), unbiased, keepdim);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_STD_NAMES_DIM, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(unbiased);trace.append_arg(keepdim);
  return tt;
}

at::Tensor & wrap_std_names_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::DimnameList dim, bool unbiased, bool keepdim, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::std_outf(dispatchKeySet, self, std::move(dim), unbiased, keepdim, out);
  }
  bool flush = register_in_place(out, H_STD_NAMES_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(unbiased);trace.append_arg(keepdim);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_std_correction_names(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::DimnameList dim, c10::optional<int64_t> correction, bool keepdim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::std(dispatchKeySet, self, std::move(dim), correction, keepdim);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_STD_CORRECTION_NAMES, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(correction);trace.append_arg(keepdim);
  return tt;
}

at::Tensor & wrap_std_correction_names_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::DimnameList dim, c10::optional<int64_t> correction, bool keepdim, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::std_outf(dispatchKeySet, self, std::move(dim), correction, keepdim, out);
  }
  bool flush = register_in_place(out, H_STD_CORRECTION_NAMES_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(correction);trace.append_arg(keepdim);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_prod(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::prod(dispatchKeySet, self, std::move(dtype));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_PROD, dtype ? scalarTypeToTypeMeta(*dtype) : self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dtype));
  return tt;
}

at::Tensor wrap_prod_dim_int(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::prod(dispatchKeySet, self, dim, keepdim, std::move(dtype));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_PROD_DIM_INT, dtype ? scalarTypeToTypeMeta(*dtype) : self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(keepdim);trace.append_arg(std::move(dtype));
  return tt;
}

at::Tensor & wrap_prod_int_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::prod_outf(dispatchKeySet, self, dim, keepdim, std::move(dtype), out);
  }
  bool flush = register_in_place(out, H_PROD_INT_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(keepdim);trace.append_arg(std::move(dtype));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_prod_dim_Dimname(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::prod(dispatchKeySet, self, std::move(dim), keepdim, std::move(dtype));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_PROD_DIM_DIMNAME, dtype ? scalarTypeToTypeMeta(*dtype) : self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(keepdim);trace.append_arg(std::move(dtype));
  return tt;
}

at::Tensor & wrap_prod_Dimname_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::prod_outf(dispatchKeySet, self, std::move(dim), keepdim, std::move(dtype), out);
  }
  bool flush = register_in_place(out, H_PROD_DIMNAME_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(keepdim);trace.append_arg(std::move(dtype));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_t(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::t(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_T, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_t_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::t_(dispatchKeySet, self);
  }
  bool flush = register_in_place(self, H_T_, dispatchKeySet);
  trace.append_arg(self);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_tan_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::tan_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_TAN_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_tanh(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::tanh(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_TANH, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_tanh_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::tanh_(dispatchKeySet, self);
  }
  bool flush = register_in_place(self, H_TANH_, dispatchKeySet);
  trace.append_arg(self);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_tanh_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::tanh_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_TANH_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_tensordot(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::IntArrayRef dims_self, at::IntArrayRef dims_other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::tensordot(dispatchKeySet, self, other, std::move(dims_self), std::move(dims_other));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_TENSORDOT, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(std::move(dims_self));trace.append_arg(std::move(dims_other));
  return tt;
}

at::Tensor & wrap_tensordot_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::IntArrayRef dims_self, at::IntArrayRef dims_other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::tensordot_outf(dispatchKeySet, self, other, std::move(dims_self), std::move(dims_other), out);
  }
  bool flush = register_in_place(out, H_TENSORDOT_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(std::move(dims_self));trace.append_arg(std::move(dims_other));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_threshold(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & threshold, const at::Scalar & value) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::threshold(dispatchKeySet, self, threshold, value);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_THRESHOLD, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(threshold);trace.append_arg(value);
  return tt;
}

at::Tensor & wrap_threshold_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & threshold, const at::Scalar & value, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::threshold_outf(dispatchKeySet, self, threshold, value, out);
  }
  bool flush = register_in_place(out, H_THRESHOLD_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(threshold);trace.append_arg(value);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_threshold_backward_grad_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & threshold, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::threshold_backward_outf(dispatchKeySet, grad_output, self, threshold, grad_input);
  }
  bool flush = register_in_place(grad_input, H_THRESHOLD_BACKWARD_GRAD_INPUT, dispatchKeySet);
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(threshold);trace.append_arg(grad_input);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return grad_input;
}

at::Tensor wrap_threshold_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & threshold) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::threshold_backward(dispatchKeySet, grad_output, self, threshold);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_THRESHOLD_BACKWARD, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(threshold);
  return tt;
}

at::Tensor wrap_tile(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dims) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::tile(dispatchKeySet, self, std::move(dims));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_TILE, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dims));
  return tt;
}

at::Tensor wrap_transpose_int(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim0, int64_t dim1) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::transpose(dispatchKeySet, self, dim0, dim1);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_TRANSPOSE_INT, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(dim0);trace.append_arg(dim1);
  return tt;
}

at::Tensor wrap_transpose_Dimname(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim0, at::Dimname dim1) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::transpose(dispatchKeySet, self, std::move(dim0), std::move(dim1));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_TRANSPOSE_DIMNAME, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim0));trace.append_arg(std::move(dim1));
  return tt;
}

at::Tensor wrap__mkldnn_transpose(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim0, int64_t dim1) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_mkldnn_transpose(dispatchKeySet, self, dim0, dim1);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__MKLDNN_TRANSPOSE, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(dim0);trace.append_arg(dim1);
  return tt;
}

at::Tensor & wrap_transpose_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, int64_t dim0, int64_t dim1) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::transpose_(dispatchKeySet, self, dim0, dim1);
  }
  bool flush = register_in_place(self, H_TRANSPOSE_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(dim0);trace.append_arg(dim1);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap__mkldnn_transpose_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, int64_t dim0, int64_t dim1) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_mkldnn_transpose_(dispatchKeySet, self, dim0, dim1);
  }
  bool flush = register_in_place(self, H__MKLDNN_TRANSPOSE_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(dim0);trace.append_arg(dim1);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_one_hot(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t num_classes) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::one_hot(dispatchKeySet, self, num_classes);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ONE_HOT, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(num_classes);
  return tt;
}

at::Tensor wrap_flip(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dims) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::flip(dispatchKeySet, self, std::move(dims));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_FLIP, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dims));
  return tt;
}

at::Tensor wrap_fliplr(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fliplr(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_FLIPLR, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor wrap_flipud(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::flipud(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_FLIPUD, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor wrap_roll(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef shifts, at::IntArrayRef dims) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::roll(dispatchKeySet, self, std::move(shifts), std::move(dims));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ROLL, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(shifts));trace.append_arg(std::move(dims));
  return tt;
}

at::Tensor wrap_rot90(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t k, at::IntArrayRef dims) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::rot90(dispatchKeySet, self, k, std::move(dims));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ROT90, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(k);trace.append_arg(std::move(dims));
  return tt;
}

at::Tensor wrap_trapz_x(c10::DispatchKeySet dispatchKeySet, const at::Tensor & y, const at::Tensor & x, int64_t dim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::trapz(dispatchKeySet, y, x, dim);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_TRAPZ_X, y.dtype(), y.device());
  trace.append_arg(y);trace.append_arg(x);trace.append_arg(dim);
  return tt;
}

at::Tensor wrap_trapz_dx(c10::DispatchKeySet dispatchKeySet, const at::Tensor & y, double dx, int64_t dim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::trapz(dispatchKeySet, y, dx, dim);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_TRAPZ_DX, y.dtype(), y.device());
  trace.append_arg(y);trace.append_arg(dx);trace.append_arg(dim);
  return tt;
}

at::Tensor wrap__trilinear(c10::DispatchKeySet dispatchKeySet, const at::Tensor & i1, const at::Tensor & i2, const at::Tensor & i3, at::IntArrayRef expand1, at::IntArrayRef expand2, at::IntArrayRef expand3, at::IntArrayRef sumdim, int64_t unroll_dim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_trilinear(dispatchKeySet, i1, i2, i3, std::move(expand1), std::move(expand2), std::move(expand3), std::move(sumdim), unroll_dim);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__TRILINEAR, i1.dtype(), i1.device());
  trace.append_arg(i1);trace.append_arg(i2);trace.append_arg(i3);trace.append_arg(std::move(expand1));trace.append_arg(std::move(expand2));trace.append_arg(std::move(expand3));trace.append_arg(std::move(sumdim));trace.append_arg(unroll_dim);
  return tt;
}

at::Tensor wrap_triplet_margin_loss(c10::DispatchKeySet dispatchKeySet, const at::Tensor & anchor, const at::Tensor & positive, const at::Tensor & negative, double margin, double p, double eps, bool swap, int64_t reduction) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::triplet_margin_loss(dispatchKeySet, anchor, positive, negative, margin, p, eps, swap, reduction);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_TRIPLET_MARGIN_LOSS, anchor.dtype(), anchor.device());
  trace.append_arg(anchor);trace.append_arg(positive);trace.append_arg(negative);trace.append_arg(margin);trace.append_arg(p);trace.append_arg(eps);trace.append_arg(swap);trace.append_arg(reduction);
  return tt;
}

at::Tensor wrap_trunc(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::trunc(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_TRUNC, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_trunc_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::trunc_(dispatchKeySet, self);
  }
  bool flush = register_in_place(self, H_TRUNC_, dispatchKeySet);
  trace.append_arg(self);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_trunc_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::trunc_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_TRUNC_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_fix(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fix(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_FIX, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_fix_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fix_(dispatchKeySet, self);
  }
  bool flush = register_in_place(self, H_FIX_, dispatchKeySet);
  trace.append_arg(self);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_fix_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fix_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_FIX_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_type_as(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::type_as(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_TYPE_AS, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor wrap__unsafe_view(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef size) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_unsafe_view(dispatchKeySet, self, std::move(size));
  }
  auto tt = register_new_tensor(dispatchKeySet, H__UNSAFE_VIEW, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(size));
  return tt;
}

at::Tensor wrap_unsqueeze(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::unsqueeze(dispatchKeySet, self, dim);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_UNSQUEEZE, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(dim);
  return tt;
}

at::Tensor & wrap_unsqueeze_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, int64_t dim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::unsqueeze_(dispatchKeySet, self, dim);
  }
  bool flush = register_in_place(self, H_UNSQUEEZE_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(dim);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_vander(c10::DispatchKeySet dispatchKeySet, const at::Tensor & x, c10::optional<int64_t> N, bool increasing) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::vander(dispatchKeySet, x, N, increasing);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_VANDER, x.dtype(), x.device());
  trace.append_arg(x);trace.append_arg(N);trace.append_arg(increasing);
  return tt;
}

at::Tensor wrap_var(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool unbiased) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::var(dispatchKeySet, self, unbiased);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_VAR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(unbiased);
  return tt;
}

at::Tensor wrap_var_dim(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, bool unbiased, bool keepdim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::var(dispatchKeySet, self, std::move(dim), unbiased, keepdim);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_VAR_DIM, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(unbiased);trace.append_arg(keepdim);
  return tt;
}

at::Tensor wrap_var_correction(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::IntArrayRef> dim, c10::optional<int64_t> correction, bool keepdim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::var(dispatchKeySet, self, std::move(dim), correction, keepdim);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_VAR_CORRECTION, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(correction);trace.append_arg(keepdim);
  return tt;
}

at::Tensor & wrap_var_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, bool unbiased, bool keepdim, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::var_outf(dispatchKeySet, self, std::move(dim), unbiased, keepdim, out);
  }
  bool flush = register_in_place(out, H_VAR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(unbiased);trace.append_arg(keepdim);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_var_correction_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::IntArrayRef> dim, c10::optional<int64_t> correction, bool keepdim, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::var_outf(dispatchKeySet, self, std::move(dim), correction, keepdim, out);
  }
  bool flush = register_in_place(out, H_VAR_CORRECTION_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(correction);trace.append_arg(keepdim);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_var_names_dim(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::DimnameList dim, bool unbiased, bool keepdim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::var(dispatchKeySet, self, std::move(dim), unbiased, keepdim);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_VAR_NAMES_DIM, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(unbiased);trace.append_arg(keepdim);
  return tt;
}

at::Tensor & wrap_var_names_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::DimnameList dim, bool unbiased, bool keepdim, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::var_outf(dispatchKeySet, self, std::move(dim), unbiased, keepdim, out);
  }
  bool flush = register_in_place(out, H_VAR_NAMES_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(unbiased);trace.append_arg(keepdim);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_var_correction_names(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::DimnameList dim, c10::optional<int64_t> correction, bool keepdim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::var(dispatchKeySet, self, std::move(dim), correction, keepdim);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_VAR_CORRECTION_NAMES, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(correction);trace.append_arg(keepdim);
  return tt;
}

at::Tensor & wrap_var_correction_names_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::DimnameList dim, c10::optional<int64_t> correction, bool keepdim, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::var_outf(dispatchKeySet, self, std::move(dim), correction, keepdim, out);
  }
  bool flush = register_in_place(out, H_VAR_CORRECTION_NAMES_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(correction);trace.append_arg(keepdim);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_view_as(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::view_as(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_VIEW_AS, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor wrap_where_self(c10::DispatchKeySet dispatchKeySet, const at::Tensor & condition, const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::where(dispatchKeySet, condition, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_WHERE_SELF, condition.dtype(), condition.device());
  trace.append_arg(condition);trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor wrap_where_ScalarSelf(c10::DispatchKeySet dispatchKeySet, const at::Tensor & condition, const at::Scalar & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::where(dispatchKeySet, condition, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_WHERE_SCALARSELF, condition.dtype(), condition.device());
  trace.append_arg(condition);trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor wrap_where_ScalarOther(c10::DispatchKeySet dispatchKeySet, const at::Tensor & condition, const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::where(dispatchKeySet, condition, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_WHERE_SCALAROTHER, condition.dtype(), condition.device());
  trace.append_arg(condition);trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor wrap_where_Scalar(c10::DispatchKeySet dispatchKeySet, const at::Tensor & condition, const at::Scalar & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::where(dispatchKeySet, condition, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_WHERE_SCALAR, condition.dtype(), condition.device());
  trace.append_arg(condition);trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor wrap__s_where(c10::DispatchKeySet dispatchKeySet, const at::Tensor & condition, const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_s_where(dispatchKeySet, condition, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__S_WHERE, condition.dtype(), condition.device());
  trace.append_arg(condition);trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor wrap_norm_except_dim(c10::DispatchKeySet dispatchKeySet, const at::Tensor & v, int64_t pow, int64_t dim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::norm_except_dim(dispatchKeySet, v, pow, dim);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_NORM_EXCEPT_DIM, v.dtype(), v.device());
  trace.append_arg(v);trace.append_arg(pow);trace.append_arg(dim);
  return tt;
}

at::Tensor wrap__weight_norm(c10::DispatchKeySet dispatchKeySet, const at::Tensor & v, const at::Tensor & g, int64_t dim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_weight_norm(dispatchKeySet, v, g, dim);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__WEIGHT_NORM, v.dtype(), v.device());
  trace.append_arg(v);trace.append_arg(g);trace.append_arg(dim);
  return tt;
}

at::Tensor wrap_zeros_names(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, c10::optional<at::DimnameList> names, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::zeros(dispatchKeySet, std::move(size), std::move(names), std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_ZEROS_NAMES, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(std::move(size));trace.append_arg(std::move(names));trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor wrap_zeros(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::zeros(dispatchKeySet, std::move(size), std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_ZEROS, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(std::move(size));trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor & wrap_zeros_out(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::zeros_outf(dispatchKeySet, std::move(size), out);
  }
  bool flush = register_in_place(out, H_ZEROS_OUT, dispatchKeySet);
  trace.append_arg(std::move(size));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_zeros_like(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::zeros_like(dispatchKeySet, self, std::move(dtype), std::move(layout), std::move(device), pin_memory, std::move(memory_format));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ZEROS_LIKE, dtype ? scalarTypeToTypeMeta(*dtype) : self.dtype(), device ? *device : self.device());
  trace.append_arg(self);trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);trace.append_arg(std::move(memory_format));
  return tt;
}

at::Tensor wrap__standard_gamma_grad(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & output) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_standard_gamma_grad(dispatchKeySet, self, output);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__STANDARD_GAMMA_GRAD, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(output);
  return tt;
}

at::Tensor wrap__standard_gamma(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::Generator> generator) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_standard_gamma(dispatchKeySet, self, std::move(generator));
  }
  auto tt = register_new_tensor(dispatchKeySet, H__STANDARD_GAMMA, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(generator));
  return tt;
}

at::Tensor wrap__dirichlet_grad(c10::DispatchKeySet dispatchKeySet, const at::Tensor & x, const at::Tensor & alpha, const at::Tensor & total) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_dirichlet_grad(dispatchKeySet, x, alpha, total);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__DIRICHLET_GRAD, x.dtype(), x.device());
  trace.append_arg(x);trace.append_arg(alpha);trace.append_arg(total);
  return tt;
}

at::Tensor wrap__sample_dirichlet(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::Generator> generator) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_sample_dirichlet(dispatchKeySet, self, std::move(generator));
  }
  auto tt = register_new_tensor(dispatchKeySet, H__SAMPLE_DIRICHLET, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(generator));
  return tt;
}

at::Tensor wrap_poisson(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::Generator> generator) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::poisson(dispatchKeySet, self, std::move(generator));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_POISSON, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(generator));
  return tt;
}

at::Tensor wrap_binomial(c10::DispatchKeySet dispatchKeySet, const at::Tensor & count, const at::Tensor & prob, c10::optional<at::Generator> generator) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::binomial(dispatchKeySet, count, prob, std::move(generator));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_BINOMIAL, count.dtype(), count.device());
  trace.append_arg(count);trace.append_arg(prob);trace.append_arg(std::move(generator));
  return tt;
}

at::Tensor wrap_native_norm(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & p) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::native_norm(dispatchKeySet, self, p);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_NATIVE_NORM, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(p);
  return tt;
}

at::Tensor wrap_native_norm_ScalarOpt_dim_dtype(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::native_norm(dispatchKeySet, self, p, std::move(dim), keepdim, std::move(dtype));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_NATIVE_NORM_SCALAROPT_DIM_DTYPE, dtype ? scalarTypeToTypeMeta(*dtype) : self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(p);trace.append_arg(std::move(dim));trace.append_arg(keepdim);trace.append_arg(std::move(dtype));
  return tt;
}

at::Tensor wrap__sparse_sum(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_sparse_sum(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__SPARSE_SUM, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor wrap__sparse_sum_dtype(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::ScalarType dtype) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_sparse_sum(dispatchKeySet, self, std::move(dtype));
  }
  auto tt = register_new_tensor(dispatchKeySet, H__SPARSE_SUM_DTYPE, scalarTypeToTypeMeta(dtype), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dtype));
  return tt;
}

at::Tensor wrap__sparse_sum_dim(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_sparse_sum(dispatchKeySet, self, std::move(dim));
  }
  auto tt = register_new_tensor(dispatchKeySet, H__SPARSE_SUM_DIM, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim));
  return tt;
}

at::Tensor wrap__sparse_sum_dim_dtype(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, at::ScalarType dtype) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_sparse_sum(dispatchKeySet, self, std::move(dim), std::move(dtype));
  }
  auto tt = register_new_tensor(dispatchKeySet, H__SPARSE_SUM_DIM_DTYPE, scalarTypeToTypeMeta(dtype), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(std::move(dtype));
  return tt;
}

at::Tensor wrap__sparse_sum_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, const at::Tensor & self, at::IntArrayRef dim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_sparse_sum_backward(dispatchKeySet, grad, self, std::move(dim));
  }
  auto tt = register_new_tensor(dispatchKeySet, H__SPARSE_SUM_BACKWARD, grad.dtype(), grad.device());
  trace.append_arg(grad);trace.append_arg(self);trace.append_arg(std::move(dim));
  return tt;
}

at::Tensor wrap__sparse_softmax_int(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_sparse_softmax(dispatchKeySet, self, dim, std::move(dtype));
  }
  auto tt = register_new_tensor(dispatchKeySet, H__SPARSE_SOFTMAX_INT, dtype ? scalarTypeToTypeMeta(*dtype) : self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(std::move(dtype));
  return tt;
}

at::Tensor wrap__sparse_softmax_Dimname(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_sparse_softmax(dispatchKeySet, self, std::move(dim), std::move(dtype));
  }
  auto tt = register_new_tensor(dispatchKeySet, H__SPARSE_SOFTMAX_DIMNAME, dtype ? scalarTypeToTypeMeta(*dtype) : self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(std::move(dtype));
  return tt;
}

at::Tensor wrap__sparse_softmax(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, bool half_to_float) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_sparse_softmax(dispatchKeySet, self, dim, half_to_float);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__SPARSE_SOFTMAX, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(half_to_float);
  return tt;
}

at::Tensor wrap__sparse_softmax_backward_data(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_sparse_softmax_backward_data(dispatchKeySet, grad_output, output, dim, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__SPARSE_SOFTMAX_BACKWARD_DATA, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(output);trace.append_arg(dim);trace.append_arg(self);
  return tt;
}

at::Tensor wrap__sparse_log_softmax_int(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_sparse_log_softmax(dispatchKeySet, self, dim, std::move(dtype));
  }
  auto tt = register_new_tensor(dispatchKeySet, H__SPARSE_LOG_SOFTMAX_INT, dtype ? scalarTypeToTypeMeta(*dtype) : self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(std::move(dtype));
  return tt;
}

at::Tensor wrap__sparse_log_softmax_Dimname(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_sparse_log_softmax(dispatchKeySet, self, std::move(dim), std::move(dtype));
  }
  auto tt = register_new_tensor(dispatchKeySet, H__SPARSE_LOG_SOFTMAX_DIMNAME, dtype ? scalarTypeToTypeMeta(*dtype) : self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(std::move(dtype));
  return tt;
}

at::Tensor wrap__sparse_log_softmax(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, bool half_to_float) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_sparse_log_softmax(dispatchKeySet, self, dim, half_to_float);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__SPARSE_LOG_SOFTMAX, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(half_to_float);
  return tt;
}

at::Tensor wrap__sparse_log_softmax_backward_data(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_sparse_log_softmax_backward_data(dispatchKeySet, grad_output, output, dim, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__SPARSE_LOG_SOFTMAX_BACKWARD_DATA, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(output);trace.append_arg(dim);trace.append_arg(self);
  return tt;
}

at::Tensor wrap_norm_ScalarOpt_dtype(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::optional<at::Scalar> & p, at::ScalarType dtype) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::norm(dispatchKeySet, self, p, std::move(dtype));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_NORM_SCALAROPT_DTYPE, scalarTypeToTypeMeta(dtype), self.device());
  trace.append_arg(self);trace.append_arg(p);trace.append_arg(std::move(dtype));
  return tt;
}

at::Tensor wrap_norm_Scalar(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & p) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::norm(dispatchKeySet, self, p);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_NORM_SCALAR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(p);
  return tt;
}

at::Tensor wrap_norm_ScalarOpt_dim_dtype(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim, at::ScalarType dtype) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::norm(dispatchKeySet, self, p, std::move(dim), keepdim, std::move(dtype));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_NORM_SCALAROPT_DIM_DTYPE, scalarTypeToTypeMeta(dtype), self.device());
  trace.append_arg(self);trace.append_arg(p);trace.append_arg(std::move(dim));trace.append_arg(keepdim);trace.append_arg(std::move(dtype));
  return tt;
}

at::Tensor wrap_norm_ScalarOpt_dim(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::norm(dispatchKeySet, self, p, std::move(dim), keepdim);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_NORM_SCALAROPT_DIM, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(p);trace.append_arg(std::move(dim));trace.append_arg(keepdim);
  return tt;
}

at::Tensor & wrap_norm_dtype_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim, at::ScalarType dtype, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::norm_outf(dispatchKeySet, self, p, std::move(dim), keepdim, std::move(dtype), out);
  }
  bool flush = register_in_place(out, H_NORM_DTYPE_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(p);trace.append_arg(std::move(dim));trace.append_arg(keepdim);trace.append_arg(std::move(dtype));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_norm_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::norm_outf(dispatchKeySet, self, p, std::move(dim), keepdim, out);
  }
  bool flush = register_in_place(out, H_NORM_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(p);trace.append_arg(std::move(dim));trace.append_arg(keepdim);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_norm_names_ScalarOpt_dim_dtype(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::optional<at::Scalar> & p, at::DimnameList dim, bool keepdim, at::ScalarType dtype) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::norm(dispatchKeySet, self, p, std::move(dim), keepdim, std::move(dtype));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_NORM_NAMES_SCALAROPT_DIM_DTYPE, scalarTypeToTypeMeta(dtype), self.device());
  trace.append_arg(self);trace.append_arg(p);trace.append_arg(std::move(dim));trace.append_arg(keepdim);trace.append_arg(std::move(dtype));
  return tt;
}

at::Tensor wrap_norm_names_ScalarOpt_dim(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::optional<at::Scalar> & p, at::DimnameList dim, bool keepdim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::norm(dispatchKeySet, self, p, std::move(dim), keepdim);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_NORM_NAMES_SCALAROPT_DIM, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(p);trace.append_arg(std::move(dim));trace.append_arg(keepdim);
  return tt;
}

at::Tensor & wrap_norm_names_dtype_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::optional<at::Scalar> & p, at::DimnameList dim, bool keepdim, at::ScalarType dtype, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::norm_outf(dispatchKeySet, self, p, std::move(dim), keepdim, std::move(dtype), out);
  }
  bool flush = register_in_place(out, H_NORM_NAMES_DTYPE_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(p);trace.append_arg(std::move(dim));trace.append_arg(keepdim);trace.append_arg(std::move(dtype));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_norm_names_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::optional<at::Scalar> & p, at::DimnameList dim, bool keepdim, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::norm_outf(dispatchKeySet, self, p, std::move(dim), keepdim, out);
  }
  bool flush = register_in_place(out, H_NORM_NAMES_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(p);trace.append_arg(std::move(dim));trace.append_arg(keepdim);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_frobenius_norm(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::frobenius_norm(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_FROBENIUS_NORM, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor wrap_frobenius_norm_dim(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, bool keepdim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::frobenius_norm(dispatchKeySet, self, std::move(dim), keepdim);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_FROBENIUS_NORM_DIM, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(keepdim);
  return tt;
}

at::Tensor & wrap_frobenius_norm_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, bool keepdim, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::frobenius_norm_outf(dispatchKeySet, self, std::move(dim), keepdim, out);
  }
  bool flush = register_in_place(out, H_FROBENIUS_NORM_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(keepdim);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_nuclear_norm(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool keepdim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::nuclear_norm(dispatchKeySet, self, keepdim);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_NUCLEAR_NORM, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(keepdim);
  return tt;
}

at::Tensor & wrap_nuclear_norm_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool keepdim, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::nuclear_norm_outf(dispatchKeySet, self, keepdim, out);
  }
  bool flush = register_in_place(out, H_NUCLEAR_NORM_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(keepdim);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_nuclear_norm_dim(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, bool keepdim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::nuclear_norm(dispatchKeySet, self, std::move(dim), keepdim);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_NUCLEAR_NORM_DIM, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(keepdim);
  return tt;
}

at::Tensor & wrap_nuclear_norm_dim_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, bool keepdim, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::nuclear_norm_outf(dispatchKeySet, self, std::move(dim), keepdim, out);
  }
  bool flush = register_in_place(out, H_NUCLEAR_NORM_DIM_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(keepdim);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_clone(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::MemoryFormat> memory_format) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::clone(dispatchKeySet, self, std::move(memory_format));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CLONE, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(memory_format));
  return tt;
}

at::Tensor wrap_positive(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::positive(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_POSITIVE, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

const at::Tensor & wrap_resize_as_(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & the_template, c10::optional<at::MemoryFormat> memory_format) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::resize_as_(dispatchKeySet, self, the_template, std::move(memory_format));
  }
  bool flush = register_in_place(self, H_RESIZE_AS_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(the_template);trace.append_arg(std::move(memory_format));
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

const at::Tensor & wrap_resize_as_sparse_(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & the_template) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::resize_as_sparse_(dispatchKeySet, self, the_template);
  }
  bool flush = register_in_place(self, H_RESIZE_AS_SPARSE_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(the_template);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_zero_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::zero_(dispatchKeySet, self);
  }
  bool flush = register_in_place(self, H_ZERO_, dispatchKeySet);
  trace.append_arg(self);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_sub_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::sub_outf(dispatchKeySet, self, other, alpha, out);
  }
  bool flush = register_in_place(out, H_SUB_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(alpha);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_sub_Tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::sub(dispatchKeySet, self, other, alpha);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SUB_TENSOR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(alpha);
  return tt;
}

at::Tensor & wrap_sub__Tensor(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::sub_(dispatchKeySet, self, other, alpha);
  }
  bool flush = register_in_place(self, H_SUB__TENSOR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(alpha);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_sub_Scalar(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::sub(dispatchKeySet, self, other, alpha);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SUB_SCALAR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(alpha);
  return tt;
}

at::Tensor & wrap_sub__Scalar(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::sub_(dispatchKeySet, self, other, alpha);
  }
  bool flush = register_in_place(self, H_SUB__SCALAR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(alpha);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_subtract_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::subtract_outf(dispatchKeySet, self, other, alpha, out);
  }
  bool flush = register_in_place(out, H_SUBTRACT_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(alpha);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_subtract_Tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::subtract(dispatchKeySet, self, other, alpha);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SUBTRACT_TENSOR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(alpha);
  return tt;
}

at::Tensor & wrap_subtract__Tensor(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::subtract_(dispatchKeySet, self, other, alpha);
  }
  bool flush = register_in_place(self, H_SUBTRACT__TENSOR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(alpha);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_subtract_Scalar(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::subtract(dispatchKeySet, self, other, alpha);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SUBTRACT_SCALAR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(alpha);
  return tt;
}

at::Tensor & wrap_subtract__Scalar(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::subtract_(dispatchKeySet, self, other, alpha);
  }
  bool flush = register_in_place(self, H_SUBTRACT__SCALAR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(alpha);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_rsub_Tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::rsub(dispatchKeySet, self, other, alpha);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_RSUB_TENSOR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(alpha);
  return tt;
}

at::Tensor & wrap_heaviside_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & values, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::heaviside_outf(dispatchKeySet, self, values, out);
  }
  bool flush = register_in_place(out, H_HEAVISIDE_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(values);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_rsub_Scalar(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::rsub(dispatchKeySet, self, other, alpha);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_RSUB_SCALAR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(alpha);
  return tt;
}

at::Tensor wrap__sparse_addmm(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & sparse, const at::Tensor & dense, const at::Scalar & beta, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_sparse_addmm(dispatchKeySet, self, sparse, dense, beta, alpha);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__SPARSE_ADDMM, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(sparse);trace.append_arg(dense);trace.append_arg(beta);trace.append_arg(alpha);
  return tt;
}

at::Tensor & wrap_addmm_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::addmm_outf(dispatchKeySet, self, mat1, mat2, beta, alpha, out);
  }
  bool flush = register_in_place(out, H_ADDMM_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(mat1);trace.append_arg(mat2);trace.append_arg(beta);trace.append_arg(alpha);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_addmm(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::addmm(dispatchKeySet, self, mat1, mat2, beta, alpha);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ADDMM, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(mat1);trace.append_arg(mat2);trace.append_arg(beta);trace.append_arg(alpha);
  return tt;
}

at::Tensor & wrap_addmm_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::addmm_(dispatchKeySet, self, mat1, mat2, beta, alpha);
  }
  bool flush = register_in_place(self, H_ADDMM_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(mat1);trace.append_arg(mat2);trace.append_arg(beta);trace.append_arg(alpha);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_sparse_csr_tensor_crow_col_value_size(c10::DispatchKeySet dispatchKeySet, const at::Tensor & crow_indices, const at::Tensor & col_indices, const at::Tensor & values, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::sparse_csr_tensor(dispatchKeySet, crow_indices, col_indices, values, std::move(size), std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SPARSE_CSR_TENSOR_CROW_COL_VALUE_SIZE, dtype ? scalarTypeToTypeMeta(*dtype) : crow_indices.dtype(), device ? *device : crow_indices.device());
  trace.append_arg(crow_indices);trace.append_arg(col_indices);trace.append_arg(values);trace.append_arg(std::move(size));trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor wrap_sparse_csr_tensor_crow_col_value(c10::DispatchKeySet dispatchKeySet, const at::Tensor & crow_indices, const at::Tensor & col_indices, const at::Tensor & values, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::sparse_csr_tensor(dispatchKeySet, crow_indices, col_indices, values, std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SPARSE_CSR_TENSOR_CROW_COL_VALUE, dtype ? scalarTypeToTypeMeta(*dtype) : crow_indices.dtype(), device ? *device : crow_indices.device());
  trace.append_arg(crow_indices);trace.append_arg(col_indices);trace.append_arg(values);trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor wrap__sparse_csr_tensor_unsafe(c10::DispatchKeySet dispatchKeySet, const at::Tensor & crow_indices, const at::Tensor & col_indices, const at::Tensor & values, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_sparse_csr_tensor_unsafe(dispatchKeySet, crow_indices, col_indices, values, std::move(size), std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__SPARSE_CSR_TENSOR_UNSAFE, dtype ? scalarTypeToTypeMeta(*dtype) : crow_indices.dtype(), device ? *device : crow_indices.device());
  trace.append_arg(crow_indices);trace.append_arg(col_indices);trace.append_arg(values);trace.append_arg(std::move(size));trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor wrap_sparse_coo_tensor_size(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::sparse_coo_tensor(dispatchKeySet, std::move(size), std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_SPARSE_COO_TENSOR_SIZE, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(std::move(size));trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor wrap_sparse_coo_tensor_indices(c10::DispatchKeySet dispatchKeySet, const at::Tensor & indices, const at::Tensor & values, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::sparse_coo_tensor(dispatchKeySet, indices, values, std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SPARSE_COO_TENSOR_INDICES, dtype ? scalarTypeToTypeMeta(*dtype) : indices.dtype(), device ? *device : indices.device());
  trace.append_arg(indices);trace.append_arg(values);trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor wrap_sparse_coo_tensor_indices_size(c10::DispatchKeySet dispatchKeySet, const at::Tensor & indices, const at::Tensor & values, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::sparse_coo_tensor(dispatchKeySet, indices, values, std::move(size), std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SPARSE_COO_TENSOR_INDICES_SIZE, dtype ? scalarTypeToTypeMeta(*dtype) : indices.dtype(), device ? *device : indices.device());
  trace.append_arg(indices);trace.append_arg(values);trace.append_arg(std::move(size));trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor wrap__sparse_coo_tensor_unsafe(c10::DispatchKeySet dispatchKeySet, const at::Tensor & indices, const at::Tensor & values, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_sparse_coo_tensor_unsafe(dispatchKeySet, indices, values, std::move(size), std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__SPARSE_COO_TENSOR_UNSAFE, dtype ? scalarTypeToTypeMeta(*dtype) : indices.dtype(), device ? *device : indices.device());
  trace.append_arg(indices);trace.append_arg(values);trace.append_arg(std::move(size));trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor wrap__sparse_coo_tensor_with_dims(c10::DispatchKeySet dispatchKeySet, int64_t sparse_dim, int64_t dense_dim, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_sparse_coo_tensor_with_dims(dispatchKeySet, sparse_dim, dense_dim, std::move(size), std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H__SPARSE_COO_TENSOR_WITH_DIMS, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(sparse_dim);trace.append_arg(dense_dim);trace.append_arg(std::move(size));trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor wrap__sparse_coo_tensor_with_dims_and_tensors(c10::DispatchKeySet dispatchKeySet, int64_t sparse_dim, int64_t dense_dim, at::IntArrayRef size, const at::Tensor & indices, const at::Tensor & values, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_sparse_coo_tensor_with_dims_and_tensors(dispatchKeySet, sparse_dim, dense_dim, std::move(size), indices, values, std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__SPARSE_COO_TENSOR_WITH_DIMS_AND_TENSORS, dtype ? scalarTypeToTypeMeta(*dtype) : indices.dtype(), device ? *device : indices.device());
  trace.append_arg(sparse_dim);trace.append_arg(dense_dim);trace.append_arg(std::move(size));trace.append_arg(indices);trace.append_arg(values);trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

const at::Tensor & wrap_sparse_resize_(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::sparse_resize_(dispatchKeySet, self, std::move(size), sparse_dim, dense_dim);
  }
  bool flush = register_in_place(self, H_SPARSE_RESIZE_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(size));trace.append_arg(sparse_dim);trace.append_arg(dense_dim);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

const at::Tensor & wrap_sparse_resize_and_clear_(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::sparse_resize_and_clear_(dispatchKeySet, self, std::move(size), sparse_dim, dense_dim);
  }
  bool flush = register_in_place(self, H_SPARSE_RESIZE_AND_CLEAR_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(size));trace.append_arg(sparse_dim);trace.append_arg(dense_dim);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_sparse_mask(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & mask) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::sparse_mask(dispatchKeySet, self, mask);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SPARSE_MASK, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(mask);
  return tt;
}

at::Tensor wrap_to_dense(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::to_dense(dispatchKeySet, self, std::move(dtype));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_TO_DENSE, dtype ? scalarTypeToTypeMeta(*dtype) : self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dtype));
  return tt;
}

at::Tensor wrap_to_dense_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, const at::Tensor & input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::to_dense_backward(dispatchKeySet, grad, input);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_TO_DENSE_BACKWARD, grad.dtype(), grad.device());
  trace.append_arg(grad);trace.append_arg(input);
  return tt;
}

at::Tensor wrap_coalesce(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::coalesce(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_COALESCE, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor wrap__coalesce(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_coalesce(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__COALESCE, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor wrap__indices(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_indices(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__INDICES, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor wrap__values(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_values(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__VALUES, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap__coalesced_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, bool coalesced) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_coalesced_(dispatchKeySet, self, coalesced);
  }
  bool flush = register_in_place(self, H__COALESCED_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(coalesced);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_indices(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::indices(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_INDICES, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor wrap_values(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::values(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_VALUES, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor wrap_crow_indices(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::crow_indices(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CROW_INDICES, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor wrap_col_indices(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::col_indices(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_COL_INDICES, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_hspmm_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & mat1, const at::Tensor & mat2, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::hspmm_outf(dispatchKeySet, mat1, mat2, out);
  }
  bool flush = register_in_place(out, H_HSPMM_OUT, dispatchKeySet);
  trace.append_arg(mat1);trace.append_arg(mat2);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_hspmm(c10::DispatchKeySet dispatchKeySet, const at::Tensor & mat1, const at::Tensor & mat2) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::hspmm(dispatchKeySet, mat1, mat2);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_HSPMM, mat1.dtype(), mat1.device());
  trace.append_arg(mat1);trace.append_arg(mat2);
  return tt;
}

at::Tensor & wrap_copy_sparse_to_sparse_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & src, bool non_blocking) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::copy_sparse_to_sparse_(dispatchKeySet, self, src, non_blocking);
  }
  bool flush = register_in_place(self, H_COPY_SPARSE_TO_SPARSE_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(src);trace.append_arg(non_blocking);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_to_sparse_sparse_dim(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t sparse_dim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::to_sparse(dispatchKeySet, self, sparse_dim);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_TO_SPARSE_SPARSE_DIM, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(sparse_dim);
  return tt;
}

at::Tensor wrap_to_sparse(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::to_sparse(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_TO_SPARSE, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor wrap_to_mkldnn(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::to_mkldnn(dispatchKeySet, self, std::move(dtype));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_TO_MKLDNN, dtype ? scalarTypeToTypeMeta(*dtype) : self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dtype));
  return tt;
}

at::Tensor wrap_mkldnn_reorder_conv2d_weight(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::mkldnn_reorder_conv2d_weight(dispatchKeySet, self, std::move(padding), std::move(stride), std::move(dilation), groups);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MKLDNN_REORDER_CONV2D_WEIGHT, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(padding));trace.append_arg(std::move(stride));trace.append_arg(std::move(dilation));trace.append_arg(groups);
  return tt;
}

at::Tensor wrap_mkldnn_reorder_conv3d_weight(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::mkldnn_reorder_conv3d_weight(dispatchKeySet, self, std::move(padding), std::move(stride), std::move(dilation), groups);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MKLDNN_REORDER_CONV3D_WEIGHT, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(padding));trace.append_arg(std::move(stride));trace.append_arg(std::move(dilation));trace.append_arg(groups);
  return tt;
}

at::Tensor wrap_to_mkldnn_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, const at::Tensor & input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::to_mkldnn_backward(dispatchKeySet, grad, input);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_TO_MKLDNN_BACKWARD, grad.dtype(), grad.device());
  trace.append_arg(grad);trace.append_arg(input);
  return tt;
}

at::Tensor wrap_quantize_per_tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, double scale, int64_t zero_point, at::ScalarType dtype) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::quantize_per_tensor(dispatchKeySet, self, scale, zero_point, std::move(dtype));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_QUANTIZE_PER_TENSOR, scalarTypeToTypeMeta(dtype), self.device());
  trace.append_arg(self);trace.append_arg(scale);trace.append_arg(zero_point);trace.append_arg(std::move(dtype));
  return tt;
}

at::Tensor wrap_quantize_per_channel(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & scales, const at::Tensor & zero_points, int64_t axis, at::ScalarType dtype) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::quantize_per_channel(dispatchKeySet, self, scales, zero_points, axis, std::move(dtype));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_QUANTIZE_PER_CHANNEL, scalarTypeToTypeMeta(dtype), self.device());
  trace.append_arg(self);trace.append_arg(scales);trace.append_arg(zero_points);trace.append_arg(axis);trace.append_arg(std::move(dtype));
  return tt;
}

at::Tensor wrap_dequantize_self(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::dequantize(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_DEQUANTIZE_SELF, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor wrap_q_per_channel_scales(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::q_per_channel_scales(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_Q_PER_CHANNEL_SCALES, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor wrap_q_per_channel_zero_points(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::q_per_channel_zero_points(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_Q_PER_CHANNEL_ZERO_POINTS, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor wrap_int_repr(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::int_repr(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_INT_REPR, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor wrap__make_per_tensor_quantized_tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, double scale, int64_t zero_point) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_make_per_tensor_quantized_tensor(dispatchKeySet, self, scale, zero_point);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__MAKE_PER_TENSOR_QUANTIZED_TENSOR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(scale);trace.append_arg(zero_point);
  return tt;
}

at::Tensor wrap__make_per_channel_quantized_tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t axis) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_make_per_channel_quantized_tensor(dispatchKeySet, self, scale, zero_point, axis);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__MAKE_PER_CHANNEL_QUANTIZED_TENSOR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(scale);trace.append_arg(zero_point);trace.append_arg(axis);
  return tt;
}

at::Tensor wrap_fake_quantize_per_tensor_affine(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, double scale, int64_t zero_point, int64_t quant_min, int64_t quant_max) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fake_quantize_per_tensor_affine(dispatchKeySet, self, scale, zero_point, quant_min, quant_max);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_FAKE_QUANTIZE_PER_TENSOR_AFFINE, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(scale);trace.append_arg(zero_point);trace.append_arg(quant_min);trace.append_arg(quant_max);
  return tt;
}

at::Tensor wrap_fake_quantize_per_tensor_affine_cachemask_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, const at::Tensor & mask) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fake_quantize_per_tensor_affine_cachemask_backward(dispatchKeySet, grad, mask);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_FAKE_QUANTIZE_PER_TENSOR_AFFINE_CACHEMASK_BACKWARD, grad.dtype(), grad.device());
  trace.append_arg(grad);trace.append_arg(mask);
  return tt;
}

at::Tensor wrap__fake_quantize_learnable_per_tensor_affine(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t quant_min, int64_t quant_max, double grad_factor) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_fake_quantize_learnable_per_tensor_affine(dispatchKeySet, self, scale, zero_point, quant_min, quant_max, grad_factor);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__FAKE_QUANTIZE_LEARNABLE_PER_TENSOR_AFFINE, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(scale);trace.append_arg(zero_point);trace.append_arg(quant_min);trace.append_arg(quant_max);trace.append_arg(grad_factor);
  return tt;
}

at::Tensor wrap_fake_quantize_per_channel_affine(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fake_quantize_per_channel_affine(dispatchKeySet, self, scale, zero_point, axis, quant_min, quant_max);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_FAKE_QUANTIZE_PER_CHANNEL_AFFINE, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(scale);trace.append_arg(zero_point);trace.append_arg(axis);trace.append_arg(quant_min);trace.append_arg(quant_max);
  return tt;
}

at::Tensor wrap_fake_quantize_per_channel_affine_cachemask_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, const at::Tensor & mask) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fake_quantize_per_channel_affine_cachemask_backward(dispatchKeySet, grad, mask);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_FAKE_QUANTIZE_PER_CHANNEL_AFFINE_CACHEMASK_BACKWARD, grad.dtype(), grad.device());
  trace.append_arg(grad);trace.append_arg(mask);
  return tt;
}

at::Tensor wrap__fake_quantize_learnable_per_channel_affine(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max, double grad_factor) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_fake_quantize_learnable_per_channel_affine(dispatchKeySet, self, scale, zero_point, axis, quant_min, quant_max, grad_factor);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__FAKE_QUANTIZE_LEARNABLE_PER_CHANNEL_AFFINE, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(scale);trace.append_arg(zero_point);trace.append_arg(axis);trace.append_arg(quant_min);trace.append_arg(quant_max);trace.append_arg(grad_factor);
  return tt;
}

at::Tensor wrap__saturate_weight_to_fp16(c10::DispatchKeySet dispatchKeySet, const at::Tensor & weight) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_saturate_weight_to_fp16(dispatchKeySet, weight);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__SATURATE_WEIGHT_TO_FP16, weight.dtype(), weight.device());
  trace.append_arg(weight);
  return tt;
}

at::Tensor wrap_to_dtype_layout(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, bool non_blocking, bool copy, c10::optional<at::MemoryFormat> memory_format) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::to(dispatchKeySet, self, std::move(dtype), std::move(layout), std::move(device), pin_memory, non_blocking, copy, std::move(memory_format));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_TO_DTYPE_LAYOUT, dtype ? scalarTypeToTypeMeta(*dtype) : self.dtype(), device ? *device : self.device());
  trace.append_arg(self);trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);trace.append_arg(non_blocking);trace.append_arg(copy);trace.append_arg(std::move(memory_format));
  return tt;
}

at::Tensor wrap_to_device(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Device device, at::ScalarType dtype, bool non_blocking, bool copy, c10::optional<at::MemoryFormat> memory_format) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::to(dispatchKeySet, self, std::move(device), std::move(dtype), non_blocking, copy, std::move(memory_format));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_TO_DEVICE, scalarTypeToTypeMeta(dtype), device);
  trace.append_arg(self);trace.append_arg(std::move(device));trace.append_arg(std::move(dtype));trace.append_arg(non_blocking);trace.append_arg(copy);trace.append_arg(std::move(memory_format));
  return tt;
}

at::Tensor wrap_to_dtype(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::ScalarType dtype, bool non_blocking, bool copy, c10::optional<at::MemoryFormat> memory_format) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::to(dispatchKeySet, self, std::move(dtype), non_blocking, copy, std::move(memory_format));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_TO_DTYPE, scalarTypeToTypeMeta(dtype), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dtype));trace.append_arg(non_blocking);trace.append_arg(copy);trace.append_arg(std::move(memory_format));
  return tt;
}

at::Tensor wrap_to_other(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, bool non_blocking, bool copy, c10::optional<at::MemoryFormat> memory_format) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::to(dispatchKeySet, self, other, non_blocking, copy, std::move(memory_format));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_TO_OTHER, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(non_blocking);trace.append_arg(copy);trace.append_arg(std::move(memory_format));
  return tt;
}

at::Tensor wrap_cartesian_prod(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::cartesian_prod(dispatchKeySet, std::move(tensors));
  }
  auto defaults = compute_dtype(tensors);
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_CARTESIAN_PROD, default_dtype, default_device);
  trace.append_arg(std::move(tensors));
  return tt;
}

at::Tensor wrap_combinations(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t r, bool with_replacement) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::combinations(dispatchKeySet, self, r, with_replacement);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_COMBINATIONS, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(r);trace.append_arg(with_replacement);
  return tt;
}

at::Tensor wrap_gru_cell(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & hx, const at::Tensor & w_ih, const at::Tensor & w_hh, const c10::optional<at::Tensor> & b_ih, const c10::optional<at::Tensor> & b_hh) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::gru_cell(dispatchKeySet, input, hx, w_ih, w_hh, b_ih, b_hh);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_GRU_CELL, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(hx);trace.append_arg(w_ih);trace.append_arg(w_hh);trace.append_arg(b_ih);trace.append_arg(b_hh);
  return tt;
}

at::Tensor wrap_rnn_tanh_cell(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & hx, const at::Tensor & w_ih, const at::Tensor & w_hh, const c10::optional<at::Tensor> & b_ih, const c10::optional<at::Tensor> & b_hh) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::rnn_tanh_cell(dispatchKeySet, input, hx, w_ih, w_hh, b_ih, b_hh);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_RNN_TANH_CELL, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(hx);trace.append_arg(w_ih);trace.append_arg(w_hh);trace.append_arg(b_ih);trace.append_arg(b_hh);
  return tt;
}

at::Tensor wrap_rnn_relu_cell(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & hx, const at::Tensor & w_ih, const at::Tensor & w_hh, const c10::optional<at::Tensor> & b_ih, const c10::optional<at::Tensor> & b_hh) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::rnn_relu_cell(dispatchKeySet, input, hx, w_ih, w_hh, b_ih, b_hh);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_RNN_RELU_CELL, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(hx);trace.append_arg(w_ih);trace.append_arg(w_hh);trace.append_arg(b_ih);trace.append_arg(b_hh);
  return tt;
}

at::Tensor wrap_quantized_gru_cell(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & hx, const at::Tensor & w_ih, const at::Tensor & w_hh, const at::Tensor & b_ih, const at::Tensor & b_hh, const at::Tensor & packed_ih, const at::Tensor & packed_hh, const at::Tensor & col_offsets_ih, const at::Tensor & col_offsets_hh, const at::Scalar & scale_ih, const at::Scalar & scale_hh, const at::Scalar & zero_point_ih, const at::Scalar & zero_point_hh) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::quantized_gru_cell(dispatchKeySet, input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_QUANTIZED_GRU_CELL, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(hx);trace.append_arg(w_ih);trace.append_arg(w_hh);trace.append_arg(b_ih);trace.append_arg(b_hh);trace.append_arg(packed_ih);trace.append_arg(packed_hh);trace.append_arg(col_offsets_ih);trace.append_arg(col_offsets_hh);trace.append_arg(scale_ih);trace.append_arg(scale_hh);trace.append_arg(zero_point_ih);trace.append_arg(zero_point_hh);
  return tt;
}

at::Tensor wrap_quantized_rnn_relu_cell(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & hx, const at::Tensor & w_ih, const at::Tensor & w_hh, const at::Tensor & b_ih, const at::Tensor & b_hh, const at::Tensor & packed_ih, const at::Tensor & packed_hh, const at::Tensor & col_offsets_ih, const at::Tensor & col_offsets_hh, const at::Scalar & scale_ih, const at::Scalar & scale_hh, const at::Scalar & zero_point_ih, const at::Scalar & zero_point_hh) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::quantized_rnn_relu_cell(dispatchKeySet, input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_QUANTIZED_RNN_RELU_CELL, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(hx);trace.append_arg(w_ih);trace.append_arg(w_hh);trace.append_arg(b_ih);trace.append_arg(b_hh);trace.append_arg(packed_ih);trace.append_arg(packed_hh);trace.append_arg(col_offsets_ih);trace.append_arg(col_offsets_hh);trace.append_arg(scale_ih);trace.append_arg(scale_hh);trace.append_arg(zero_point_ih);trace.append_arg(zero_point_hh);
  return tt;
}

at::Tensor wrap_quantized_rnn_tanh_cell(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & hx, const at::Tensor & w_ih, const at::Tensor & w_hh, const at::Tensor & b_ih, const at::Tensor & b_hh, const at::Tensor & packed_ih, const at::Tensor & packed_hh, const at::Tensor & col_offsets_ih, const at::Tensor & col_offsets_hh, const at::Scalar & scale_ih, const at::Scalar & scale_hh, const at::Scalar & zero_point_ih, const at::Scalar & zero_point_hh) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::quantized_rnn_tanh_cell(dispatchKeySet, input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_QUANTIZED_RNN_TANH_CELL, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(hx);trace.append_arg(w_ih);trace.append_arg(w_hh);trace.append_arg(b_ih);trace.append_arg(b_hh);trace.append_arg(packed_ih);trace.append_arg(packed_hh);trace.append_arg(col_offsets_ih);trace.append_arg(col_offsets_hh);trace.append_arg(scale_ih);trace.append_arg(scale_hh);trace.append_arg(zero_point_ih);trace.append_arg(zero_point_hh);
  return tt;
}

at::Tensor wrap__pack_padded_sequence_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, at::IntArrayRef input_size, const at::Tensor & batch_sizes, bool batch_first) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_pack_padded_sequence_backward(dispatchKeySet, grad, std::move(input_size), batch_sizes, batch_first);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__PACK_PADDED_SEQUENCE_BACKWARD, grad.dtype(), grad.device());
  trace.append_arg(grad);trace.append_arg(std::move(input_size));trace.append_arg(batch_sizes);trace.append_arg(batch_first);
  return tt;
}

at::Tensor & wrap_set__source_Storage(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, at::Storage source) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::set_(dispatchKeySet, self, std::move(source));
  }
  bool flush = register_in_place(self, H_SET__SOURCE_STORAGE, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(source));
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_set__source_Storage_storage_offset(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, at::Storage source, int64_t storage_offset, at::IntArrayRef size, at::IntArrayRef stride) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::set_(dispatchKeySet, self, std::move(source), storage_offset, std::move(size), std::move(stride));
  }
  bool flush = register_in_place(self, H_SET__SOURCE_STORAGE_STORAGE_OFFSET, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(source));trace.append_arg(storage_offset);trace.append_arg(std::move(size));trace.append_arg(std::move(stride));
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_set__source_Tensor(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & source) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::set_(dispatchKeySet, self, source);
  }
  bool flush = register_in_place(self, H_SET__SOURCE_TENSOR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(source);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_set_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::set_(dispatchKeySet, self);
  }
  bool flush = register_in_place(self, H_SET_, dispatchKeySet);
  trace.append_arg(self);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_masked_fill__Scalar(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & mask, const at::Scalar & value) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::masked_fill_(dispatchKeySet, self, mask, value);
  }
  bool flush = register_in_place(self, H_MASKED_FILL__SCALAR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(mask);trace.append_arg(value);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_masked_fill_Scalar(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & mask, const at::Scalar & value) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::masked_fill(dispatchKeySet, self, mask, value);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MASKED_FILL_SCALAR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(mask);trace.append_arg(value);
  return tt;
}

at::Tensor & wrap_masked_fill__Tensor(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & mask, const at::Tensor & value) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::masked_fill_(dispatchKeySet, self, mask, value);
  }
  bool flush = register_in_place(self, H_MASKED_FILL__TENSOR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(mask);trace.append_arg(value);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_masked_fill_Tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & mask, const at::Tensor & value) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::masked_fill(dispatchKeySet, self, mask, value);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MASKED_FILL_TENSOR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(mask);trace.append_arg(value);
  return tt;
}

at::Tensor & wrap_masked_scatter_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & mask, const at::Tensor & source) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::masked_scatter_(dispatchKeySet, self, mask, source);
  }
  bool flush = register_in_place(self, H_MASKED_SCATTER_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(mask);trace.append_arg(source);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_masked_scatter(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & mask, const at::Tensor & source) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::masked_scatter(dispatchKeySet, self, mask, source);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MASKED_SCATTER, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(mask);trace.append_arg(source);
  return tt;
}

at::Tensor wrap_view(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef size) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::view(dispatchKeySet, self, std::move(size));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_VIEW, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(size));
  return tt;
}

at::Tensor wrap_view_dtype(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::ScalarType dtype) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::view(dispatchKeySet, self, std::move(dtype));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_VIEW_DTYPE, scalarTypeToTypeMeta(dtype), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dtype));
  return tt;
}

at::Tensor & wrap_put_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & index, const at::Tensor & source, bool accumulate) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::put_(dispatchKeySet, self, index, source, accumulate);
  }
  bool flush = register_in_place(self, H_PUT_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(index);trace.append_arg(source);trace.append_arg(accumulate);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_put(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & index, const at::Tensor & source, bool accumulate) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::put(dispatchKeySet, self, index, source, accumulate);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_PUT, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(index);trace.append_arg(source);trace.append_arg(accumulate);
  return tt;
}

at::Tensor & wrap_index_add_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::index_add_(dispatchKeySet, self, dim, index, source);
  }
  bool flush = register_in_place(self, H_INDEX_ADD_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(index);trace.append_arg(source);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_index_add__alpha(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::index_add_(dispatchKeySet, self, dim, index, source, alpha);
  }
  bool flush = register_in_place(self, H_INDEX_ADD__ALPHA, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(index);trace.append_arg(source);trace.append_arg(alpha);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_index_add(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::index_add(dispatchKeySet, self, dim, index, source);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_INDEX_ADD, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(index);trace.append_arg(source);
  return tt;
}

at::Tensor wrap_index_add_alpha(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::index_add(dispatchKeySet, self, dim, index, source, alpha);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_INDEX_ADD_ALPHA, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(index);trace.append_arg(source);trace.append_arg(alpha);
  return tt;
}

at::Tensor wrap_index_add_dimname(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & source, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::index_add(dispatchKeySet, self, std::move(dim), index, source, alpha);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_INDEX_ADD_DIMNAME, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(index);trace.append_arg(source);trace.append_arg(alpha);
  return tt;
}

at::Tensor & wrap_index_fill__int_Scalar(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::index_fill_(dispatchKeySet, self, dim, index, value);
  }
  bool flush = register_in_place(self, H_INDEX_FILL__INT_SCALAR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(index);trace.append_arg(value);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_index_fill_int_Scalar(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::index_fill(dispatchKeySet, self, dim, index, value);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_INDEX_FILL_INT_SCALAR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(index);trace.append_arg(value);
  return tt;
}

at::Tensor & wrap_index_fill__int_Tensor(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & value) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::index_fill_(dispatchKeySet, self, dim, index, value);
  }
  bool flush = register_in_place(self, H_INDEX_FILL__INT_TENSOR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(index);trace.append_arg(value);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_index_fill_int_Tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & value) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::index_fill(dispatchKeySet, self, dim, index, value);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_INDEX_FILL_INT_TENSOR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(index);trace.append_arg(value);
  return tt;
}

at::Tensor & wrap_index_fill__Dimname_Scalar(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Scalar & value) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::index_fill_(dispatchKeySet, self, std::move(dim), index, value);
  }
  bool flush = register_in_place(self, H_INDEX_FILL__DIMNAME_SCALAR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(index);trace.append_arg(value);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_index_fill__Dimname_Tensor(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & value) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::index_fill_(dispatchKeySet, self, std::move(dim), index, value);
  }
  bool flush = register_in_place(self, H_INDEX_FILL__DIMNAME_TENSOR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(index);trace.append_arg(value);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_index_fill_Dimname_Scalar(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Scalar & value) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::index_fill(dispatchKeySet, self, std::move(dim), index, value);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_INDEX_FILL_DIMNAME_SCALAR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(index);trace.append_arg(value);
  return tt;
}

at::Tensor wrap_index_fill_Dimname_Tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & value) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::index_fill(dispatchKeySet, self, std::move(dim), index, value);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_INDEX_FILL_DIMNAME_TENSOR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(index);trace.append_arg(value);
  return tt;
}

at::Tensor & wrap_scatter_src_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::scatter_outf(dispatchKeySet, self, dim, index, src, out);
  }
  bool flush = register_in_place(out, H_SCATTER_SRC_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(index);trace.append_arg(src);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_scatter_value_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::scatter_outf(dispatchKeySet, self, dim, index, value, out);
  }
  bool flush = register_in_place(out, H_SCATTER_VALUE_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(index);trace.append_arg(value);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_scatter_reduce_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, c10::string_view reduce, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::scatter_outf(dispatchKeySet, self, dim, index, src, std::move(reduce), out);
  }
  bool flush = register_in_place(out, H_SCATTER_REDUCE_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(index);trace.append_arg(src);trace.append_arg(std::move(reduce));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_scatter_value_reduce_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value, c10::string_view reduce, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::scatter_outf(dispatchKeySet, self, dim, index, value, std::move(reduce), out);
  }
  bool flush = register_in_place(out, H_SCATTER_VALUE_REDUCE_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(index);trace.append_arg(value);trace.append_arg(std::move(reduce));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_scatter_dimname_src(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & src) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::scatter(dispatchKeySet, self, std::move(dim), index, src);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SCATTER_DIMNAME_SRC, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(index);trace.append_arg(src);
  return tt;
}

at::Tensor wrap_scatter_dimname_value(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Scalar & value) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::scatter(dispatchKeySet, self, std::move(dim), index, value);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SCATTER_DIMNAME_VALUE, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(index);trace.append_arg(value);
  return tt;
}

at::Tensor & wrap_scatter_add_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::scatter_add_outf(dispatchKeySet, self, dim, index, src, out);
  }
  bool flush = register_in_place(out, H_SCATTER_ADD_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(index);trace.append_arg(src);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_scatter_add_dimname(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & src) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::scatter_add(dispatchKeySet, self, std::move(dim), index, src);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SCATTER_ADD_DIMNAME, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(index);trace.append_arg(src);
  return tt;
}

at::Tensor & wrap_eq__Scalar(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::eq_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_EQ__SCALAR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_eq__Tensor(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::eq_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_EQ__TENSOR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_bitwise_and_Tensor_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::bitwise_and_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_BITWISE_AND_TENSOR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_bitwise_and_Scalar_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::bitwise_and_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_BITWISE_AND_SCALAR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_bitwise_and_Scalar(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::bitwise_and(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_BITWISE_AND_SCALAR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor wrap_bitwise_and_Tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::bitwise_and(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_BITWISE_AND_TENSOR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_bitwise_and__Scalar(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::bitwise_and_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_BITWISE_AND__SCALAR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_bitwise_and__Tensor(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::bitwise_and_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_BITWISE_AND__TENSOR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap___and___Scalar(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::__and__(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H___AND___SCALAR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor wrap___and___Tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::__and__(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H___AND___TENSOR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap___iand___Scalar(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::__iand__(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H___IAND___SCALAR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap___iand___Tensor(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::__iand__(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H___IAND___TENSOR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_bitwise_or_Tensor_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::bitwise_or_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_BITWISE_OR_TENSOR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_bitwise_or_Scalar_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::bitwise_or_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_BITWISE_OR_SCALAR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_bitwise_or_Scalar(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::bitwise_or(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_BITWISE_OR_SCALAR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor wrap_bitwise_or_Tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::bitwise_or(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_BITWISE_OR_TENSOR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_bitwise_or__Scalar(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::bitwise_or_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_BITWISE_OR__SCALAR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_bitwise_or__Tensor(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::bitwise_or_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_BITWISE_OR__TENSOR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap___or___Scalar(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::__or__(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H___OR___SCALAR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor wrap___or___Tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::__or__(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H___OR___TENSOR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap___ior___Scalar(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::__ior__(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H___IOR___SCALAR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap___ior___Tensor(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::__ior__(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H___IOR___TENSOR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_bitwise_xor_Tensor_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::bitwise_xor_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_BITWISE_XOR_TENSOR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_bitwise_xor_Scalar_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::bitwise_xor_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_BITWISE_XOR_SCALAR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_bitwise_xor_Scalar(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::bitwise_xor(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_BITWISE_XOR_SCALAR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor wrap_bitwise_xor_Tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::bitwise_xor(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_BITWISE_XOR_TENSOR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_bitwise_xor__Scalar(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::bitwise_xor_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_BITWISE_XOR__SCALAR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_bitwise_xor__Tensor(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::bitwise_xor_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_BITWISE_XOR__TENSOR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap___xor___Scalar(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::__xor__(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H___XOR___SCALAR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor wrap___xor___Tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::__xor__(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H___XOR___TENSOR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap___ixor___Scalar(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::__ixor__(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H___IXOR___SCALAR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap___ixor___Tensor(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::__ixor__(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H___IXOR___TENSOR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap___lshift___Scalar(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::__lshift__(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H___LSHIFT___SCALAR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor wrap___lshift___Tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::__lshift__(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H___LSHIFT___TENSOR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap___ilshift___Scalar(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::__ilshift__(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H___ILSHIFT___SCALAR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap___ilshift___Tensor(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::__ilshift__(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H___ILSHIFT___TENSOR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap___rshift___Scalar(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::__rshift__(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H___RSHIFT___SCALAR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor wrap___rshift___Tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::__rshift__(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H___RSHIFT___TENSOR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap___irshift___Scalar(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::__irshift__(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H___IRSHIFT___SCALAR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap___irshift___Tensor(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::__irshift__(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H___IRSHIFT___TENSOR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_tril_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, int64_t diagonal) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::tril_(dispatchKeySet, self, diagonal);
  }
  bool flush = register_in_place(self, H_TRIL_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(diagonal);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_triu_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, int64_t diagonal) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::triu_(dispatchKeySet, self, diagonal);
  }
  bool flush = register_in_place(self, H_TRIU_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(diagonal);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_lerp__Scalar(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & end, const at::Scalar & weight) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::lerp_(dispatchKeySet, self, end, weight);
  }
  bool flush = register_in_place(self, H_LERP__SCALAR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(end);trace.append_arg(weight);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_lerp__Tensor(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & end, const at::Tensor & weight) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::lerp_(dispatchKeySet, self, end, weight);
  }
  bool flush = register_in_place(self, H_LERP__TENSOR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(end);trace.append_arg(weight);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_fmod__Scalar(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fmod_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_FMOD__SCALAR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_fmod__Tensor(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fmod_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_FMOD__TENSOR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_addbmm_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::addbmm_(dispatchKeySet, self, batch1, batch2, beta, alpha);
  }
  bool flush = register_in_place(self, H_ADDBMM_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(batch1);trace.append_arg(batch2);trace.append_arg(beta);trace.append_arg(alpha);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_addbmm_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::addbmm_outf(dispatchKeySet, self, batch1, batch2, beta, alpha, out);
  }
  bool flush = register_in_place(out, H_ADDBMM_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(batch1);trace.append_arg(batch2);trace.append_arg(beta);trace.append_arg(alpha);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_addbmm(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::addbmm(dispatchKeySet, self, batch1, batch2, beta, alpha);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ADDBMM, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(batch1);trace.append_arg(batch2);trace.append_arg(beta);trace.append_arg(alpha);
  return tt;
}

at::Tensor & wrap_addcdiv_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::addcdiv_(dispatchKeySet, self, tensor1, tensor2, value);
  }
  bool flush = register_in_place(self, H_ADDCDIV_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(tensor1);trace.append_arg(tensor2);trace.append_arg(value);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_random__from(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, int64_t from, c10::optional<int64_t> to, c10::optional<at::Generator> generator) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::random_(dispatchKeySet, self, from, to, std::move(generator));
  }
  bool flush = register_in_place(self, H_RANDOM__FROM, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(from);trace.append_arg(to);trace.append_arg(std::move(generator));
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_random__to(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, int64_t to, c10::optional<at::Generator> generator) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::random_(dispatchKeySet, self, to, std::move(generator));
  }
  bool flush = register_in_place(self, H_RANDOM__TO, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(to);trace.append_arg(std::move(generator));
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_random_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, c10::optional<at::Generator> generator) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::random_(dispatchKeySet, self, std::move(generator));
  }
  bool flush = register_in_place(self, H_RANDOM_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(generator));
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_uniform_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, double from, double to, c10::optional<at::Generator> generator) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::uniform_(dispatchKeySet, self, from, to, std::move(generator));
  }
  bool flush = register_in_place(self, H_UNIFORM_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(from);trace.append_arg(to);trace.append_arg(std::move(generator));
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_cauchy_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, double median, double sigma, c10::optional<at::Generator> generator) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::cauchy_(dispatchKeySet, self, median, sigma, std::move(generator));
  }
  bool flush = register_in_place(self, H_CAUCHY_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(median);trace.append_arg(sigma);trace.append_arg(std::move(generator));
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_log_normal_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, double mean, double std, c10::optional<at::Generator> generator) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::log_normal_(dispatchKeySet, self, mean, std, std::move(generator));
  }
  bool flush = register_in_place(self, H_LOG_NORMAL_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(mean);trace.append_arg(std);trace.append_arg(std::move(generator));
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_exponential_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, double lambd, c10::optional<at::Generator> generator) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::exponential_(dispatchKeySet, self, lambd, std::move(generator));
  }
  bool flush = register_in_place(self, H_EXPONENTIAL_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(lambd);trace.append_arg(std::move(generator));
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_geometric_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, double p, c10::optional<at::Generator> generator) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::geometric_(dispatchKeySet, self, p, std::move(generator));
  }
  bool flush = register_in_place(self, H_GEOMETRIC_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(p);trace.append_arg(std::move(generator));
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_diag_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t diagonal, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::diag_outf(dispatchKeySet, self, diagonal, out);
  }
  bool flush = register_in_place(out, H_DIAG_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(diagonal);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_diag(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t diagonal) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::diag(dispatchKeySet, self, diagonal);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_DIAG, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(diagonal);
  return tt;
}

at::Tensor wrap_diag_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, at::IntArrayRef input_sizes, int64_t diagonal) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::diag_backward(dispatchKeySet, grad, std::move(input_sizes), diagonal);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_DIAG_BACKWARD, grad.dtype(), grad.device());
  trace.append_arg(grad);trace.append_arg(std::move(input_sizes));trace.append_arg(diagonal);
  return tt;
}

at::Tensor & wrap_cross_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, c10::optional<int64_t> dim, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::cross_outf(dispatchKeySet, self, other, dim, out);
  }
  bool flush = register_in_place(out, H_CROSS_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(dim);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_cross(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, c10::optional<int64_t> dim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::cross(dispatchKeySet, self, other, dim);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CROSS, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(dim);
  return tt;
}

at::Tensor & wrap_triu_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t diagonal, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::triu_outf(dispatchKeySet, self, diagonal, out);
  }
  bool flush = register_in_place(out, H_TRIU_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(diagonal);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_triu(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t diagonal) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::triu(dispatchKeySet, self, diagonal);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_TRIU, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(diagonal);
  return tt;
}

at::Tensor & wrap_tril_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t diagonal, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::tril_outf(dispatchKeySet, self, diagonal, out);
  }
  bool flush = register_in_place(out, H_TRIL_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(diagonal);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_tril(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t diagonal) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::tril(dispatchKeySet, self, diagonal);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_TRIL, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(diagonal);
  return tt;
}

at::Tensor wrap_tril_indices(c10::DispatchKeySet dispatchKeySet, int64_t row, int64_t col, int64_t offset, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::tril_indices(dispatchKeySet, row, col, offset, std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_TRIL_INDICES, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(row);trace.append_arg(col);trace.append_arg(offset);trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor wrap_triu_indices(c10::DispatchKeySet dispatchKeySet, int64_t row, int64_t col, int64_t offset, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::triu_indices(dispatchKeySet, row, col, offset, std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_TRIU_INDICES, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(row);trace.append_arg(col);trace.append_arg(offset);trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor wrap_trace(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::trace(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_TRACE, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor wrap_trace_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, at::IntArrayRef sizes) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::trace_backward(dispatchKeySet, grad, std::move(sizes));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_TRACE_BACKWARD, grad.dtype(), grad.device());
  trace.append_arg(grad);trace.append_arg(std::move(sizes));
  return tt;
}

at::Tensor & wrap_ne_Scalar_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::ne_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_NE_SCALAR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_ne_Scalar(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::ne(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_NE_SCALAR, scalarTypeToTypeMeta(kBool), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_ne_Tensor_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::ne_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_NE_TENSOR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_ne_Tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::ne(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_NE_TENSOR, scalarTypeToTypeMeta(kBool), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_ne__Scalar(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::ne_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_NE__SCALAR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_ne__Tensor(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::ne_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_NE__TENSOR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_not_equal_Scalar_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::not_equal_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_NOT_EQUAL_SCALAR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_not_equal_Scalar(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::not_equal(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_NOT_EQUAL_SCALAR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_not_equal_Tensor_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::not_equal_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_NOT_EQUAL_TENSOR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_not_equal_Tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::not_equal(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_NOT_EQUAL_TENSOR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_not_equal__Scalar(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::not_equal_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_NOT_EQUAL__SCALAR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_not_equal__Tensor(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::not_equal_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_NOT_EQUAL__TENSOR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_eq_Scalar_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::eq_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_EQ_SCALAR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_eq_Scalar(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::eq(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_EQ_SCALAR, scalarTypeToTypeMeta(kBool), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_eq_Tensor_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::eq_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_EQ_TENSOR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_eq_Tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::eq(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_EQ_TENSOR, scalarTypeToTypeMeta(kBool), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_ge_Scalar_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::ge_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_GE_SCALAR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_ge_Scalar(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::ge(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_GE_SCALAR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_ge_Tensor_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::ge_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_GE_TENSOR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_ge_Tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::ge(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_GE_TENSOR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_ge__Scalar(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::ge_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_GE__SCALAR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_ge__Tensor(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::ge_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_GE__TENSOR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_greater_equal_Scalar_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::greater_equal_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_GREATER_EQUAL_SCALAR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_greater_equal_Scalar(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::greater_equal(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_GREATER_EQUAL_SCALAR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_greater_equal_Tensor_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::greater_equal_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_GREATER_EQUAL_TENSOR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_greater_equal_Tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::greater_equal(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_GREATER_EQUAL_TENSOR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_greater_equal__Scalar(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::greater_equal_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_GREATER_EQUAL__SCALAR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_greater_equal__Tensor(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::greater_equal_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_GREATER_EQUAL__TENSOR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_le_Scalar_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::le_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_LE_SCALAR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_le_Scalar(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::le(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LE_SCALAR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_le_Tensor_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::le_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_LE_TENSOR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_le_Tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::le(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LE_TENSOR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_le__Scalar(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::le_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_LE__SCALAR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_le__Tensor(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::le_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_LE__TENSOR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_less_equal_Scalar_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::less_equal_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_LESS_EQUAL_SCALAR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_less_equal_Scalar(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::less_equal(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LESS_EQUAL_SCALAR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_less_equal_Tensor_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::less_equal_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_LESS_EQUAL_TENSOR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_less_equal_Tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::less_equal(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LESS_EQUAL_TENSOR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_less_equal__Scalar(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::less_equal_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_LESS_EQUAL__SCALAR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_less_equal__Tensor(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::less_equal_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_LESS_EQUAL__TENSOR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_gt_Scalar_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::gt_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_GT_SCALAR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_gt_Scalar(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::gt(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_GT_SCALAR, scalarTypeToTypeMeta(kBool), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_gt_Tensor_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::gt_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_GT_TENSOR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_gt_Tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::gt(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_GT_TENSOR, scalarTypeToTypeMeta(kBool), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_gt__Scalar(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::gt_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_GT__SCALAR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_gt__Tensor(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::gt_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_GT__TENSOR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_greater_Scalar_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::greater_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_GREATER_SCALAR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_greater_Scalar(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::greater(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_GREATER_SCALAR, scalarTypeToTypeMeta(kBool), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_greater_Tensor_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::greater_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_GREATER_TENSOR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_greater_Tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::greater(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_GREATER_TENSOR, scalarTypeToTypeMeta(kBool), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_greater__Scalar(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::greater_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_GREATER__SCALAR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_greater__Tensor(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::greater_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_GREATER__TENSOR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_lt_Scalar_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::lt_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_LT_SCALAR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_lt_Scalar(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::lt(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LT_SCALAR, scalarTypeToTypeMeta(kBool), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_lt_Tensor_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::lt_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_LT_TENSOR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_lt_Tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::lt(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LT_TENSOR, scalarTypeToTypeMeta(kBool), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_lt__Scalar(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::lt_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_LT__SCALAR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_lt__Tensor(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::lt_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_LT__TENSOR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_less_Scalar_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::less_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_LESS_SCALAR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_less_Scalar(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::less(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LESS_SCALAR, scalarTypeToTypeMeta(kBool), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_less_Tensor_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::less_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_LESS_TENSOR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_less_Tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::less(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LESS_TENSOR, scalarTypeToTypeMeta(kBool), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_less__Scalar(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::less_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_LESS__SCALAR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_less__Tensor(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::less_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_LESS__TENSOR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_take_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & index, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::take_outf(dispatchKeySet, self, index, out);
  }
  bool flush = register_in_place(out, H_TAKE_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(index);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_take(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & index) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::take(dispatchKeySet, self, index);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_TAKE, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(index);
  return tt;
}

at::Tensor & wrap_take_along_dim_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & indices, c10::optional<int64_t> dim, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::take_along_dim_outf(dispatchKeySet, self, indices, dim, out);
  }
  bool flush = register_in_place(out, H_TAKE_ALONG_DIM_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(indices);trace.append_arg(dim);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_take_along_dim(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & indices, c10::optional<int64_t> dim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::take_along_dim(dispatchKeySet, self, indices, dim);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_TAKE_ALONG_DIM, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(indices);trace.append_arg(dim);
  return tt;
}

at::Tensor & wrap_index_select_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, const at::Tensor & index, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::index_select_outf(dispatchKeySet, self, dim, index, out);
  }
  bool flush = register_in_place(out, H_INDEX_SELECT_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(index);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_index_select(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, const at::Tensor & index) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::index_select(dispatchKeySet, self, dim, index);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_INDEX_SELECT, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(index);
  return tt;
}

at::Tensor & wrap_index_select_dimname_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, const at::Tensor & index, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::index_select_outf(dispatchKeySet, self, std::move(dim), index, out);
  }
  bool flush = register_in_place(out, H_INDEX_SELECT_DIMNAME_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(index);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_index_select_dimname(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, const at::Tensor & index) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::index_select(dispatchKeySet, self, std::move(dim), index);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_INDEX_SELECT_DIMNAME, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(index);
  return tt;
}

at::Tensor wrap_index_select_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, at::IntArrayRef self_sizes, int64_t dim, const at::Tensor & index) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::index_select_backward(dispatchKeySet, grad, std::move(self_sizes), dim, index);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_INDEX_SELECT_BACKWARD, grad.dtype(), grad.device());
  trace.append_arg(grad);trace.append_arg(std::move(self_sizes));trace.append_arg(dim);trace.append_arg(index);
  return tt;
}

at::Tensor & wrap_masked_select_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & mask, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::masked_select_outf(dispatchKeySet, self, mask, out);
  }
  bool flush = register_in_place(out, H_MASKED_SELECT_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(mask);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_masked_select(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & mask) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::masked_select(dispatchKeySet, self, mask);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MASKED_SELECT, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(mask);
  return tt;
}

at::Tensor wrap_masked_select_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, const at::Tensor & input, const at::Tensor & mask) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::masked_select_backward(dispatchKeySet, grad, input, mask);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MASKED_SELECT_BACKWARD, grad.dtype(), grad.device());
  trace.append_arg(grad);trace.append_arg(input);trace.append_arg(mask);
  return tt;
}

at::Tensor & wrap_nonzero_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::nonzero_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_NONZERO_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_nonzero(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::nonzero(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_NONZERO, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_gather_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, const at::Tensor & index, bool sparse_grad, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::gather_outf(dispatchKeySet, self, dim, index, sparse_grad, out);
  }
  bool flush = register_in_place(out, H_GATHER_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(index);trace.append_arg(sparse_grad);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_gather(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, const at::Tensor & index, bool sparse_grad) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::gather(dispatchKeySet, self, dim, index, sparse_grad);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_GATHER, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(index);trace.append_arg(sparse_grad);
  return tt;
}

at::Tensor wrap_gather_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, const at::Tensor & self, int64_t dim, const at::Tensor & index, bool sparse_grad) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::gather_backward(dispatchKeySet, grad, self, dim, index, sparse_grad);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_GATHER_BACKWARD, grad.dtype(), grad.device());
  trace.append_arg(grad);trace.append_arg(self);trace.append_arg(dim);trace.append_arg(index);trace.append_arg(sparse_grad);
  return tt;
}

at::Tensor & wrap_gather_dimname_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, const at::Tensor & index, bool sparse_grad, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::gather_outf(dispatchKeySet, self, std::move(dim), index, sparse_grad, out);
  }
  bool flush = register_in_place(out, H_GATHER_DIMNAME_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(index);trace.append_arg(sparse_grad);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_gather_dimname(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, const at::Tensor & index, bool sparse_grad) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::gather(dispatchKeySet, self, std::move(dim), index, sparse_grad);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_GATHER_DIMNAME, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(index);trace.append_arg(sparse_grad);
  return tt;
}

at::Tensor wrap__gather_sparse_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & grad) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_gather_sparse_backward(dispatchKeySet, self, dim, index, grad);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__GATHER_SPARSE_BACKWARD, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(index);trace.append_arg(grad);
  return tt;
}

at::Tensor & wrap_addcmul_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::addcmul_outf(dispatchKeySet, self, tensor1, tensor2, value, out);
  }
  bool flush = register_in_place(out, H_ADDCMUL_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(tensor1);trace.append_arg(tensor2);trace.append_arg(value);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_addcmul(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::addcmul(dispatchKeySet, self, tensor1, tensor2, value);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ADDCMUL, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(tensor1);trace.append_arg(tensor2);trace.append_arg(value);
  return tt;
}

at::Tensor & wrap_addcmul_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::addcmul_(dispatchKeySet, self, tensor1, tensor2, value);
  }
  bool flush = register_in_place(self, H_ADDCMUL_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(tensor1);trace.append_arg(tensor2);trace.append_arg(value);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_addcdiv_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::addcdiv_outf(dispatchKeySet, self, tensor1, tensor2, value, out);
  }
  bool flush = register_in_place(out, H_ADDCDIV_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(tensor1);trace.append_arg(tensor2);trace.append_arg(value);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_addcdiv(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::addcdiv(dispatchKeySet, self, tensor1, tensor2, value);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ADDCDIV, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(tensor1);trace.append_arg(tensor2);trace.append_arg(value);
  return tt;
}

at::Tensor wrap_cross_entropy_loss(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::cross_entropy_loss(dispatchKeySet, self, target, weight, reduction, ignore_index);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CROSS_ENTROPY_LOSS, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(target);trace.append_arg(weight);trace.append_arg(reduction);trace.append_arg(ignore_index);
  return tt;
}

at::Tensor wrap_swapaxes(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t axis0, int64_t axis1) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::swapaxes(dispatchKeySet, self, axis0, axis1);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SWAPAXES, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(axis0);trace.append_arg(axis1);
  return tt;
}

at::Tensor & wrap_swapaxes_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, int64_t axis0, int64_t axis1) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::swapaxes_(dispatchKeySet, self, axis0, axis1);
  }
  bool flush = register_in_place(self, H_SWAPAXES_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(axis0);trace.append_arg(axis1);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_swapdims(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim0, int64_t dim1) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::swapdims(dispatchKeySet, self, dim0, dim1);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SWAPDIMS, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(dim0);trace.append_arg(dim1);
  return tt;
}

at::Tensor & wrap_swapdims_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, int64_t dim0, int64_t dim1) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::swapdims_(dispatchKeySet, self, dim0, dim1);
  }
  bool flush = register_in_place(self, H_SWAPDIMS_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(dim0);trace.append_arg(dim1);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_cholesky_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool upper, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::cholesky_outf(dispatchKeySet, self, upper, out);
  }
  bool flush = register_in_place(out, H_CHOLESKY_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(upper);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_cholesky(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool upper) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::cholesky(dispatchKeySet, self, upper);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CHOLESKY, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(upper);
  return tt;
}

at::Tensor & wrap_cholesky_solve_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & input2, bool upper, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::cholesky_solve_outf(dispatchKeySet, self, input2, upper, out);
  }
  bool flush = register_in_place(out, H_CHOLESKY_SOLVE_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(input2);trace.append_arg(upper);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_cholesky_solve(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & input2, bool upper) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::cholesky_solve(dispatchKeySet, self, input2, upper);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CHOLESKY_SOLVE, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(input2);trace.append_arg(upper);
  return tt;
}

at::Tensor wrap__cholesky_solve_helper(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & A, bool upper) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_cholesky_solve_helper(dispatchKeySet, self, A, upper);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__CHOLESKY_SOLVE_HELPER, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(A);trace.append_arg(upper);
  return tt;
}

at::Tensor wrap_cholesky_inverse(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool upper) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::cholesky_inverse(dispatchKeySet, self, upper);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CHOLESKY_INVERSE, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(upper);
  return tt;
}

at::Tensor & wrap_cholesky_inverse_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool upper, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::cholesky_inverse_outf(dispatchKeySet, self, upper, out);
  }
  bool flush = register_in_place(out, H_CHOLESKY_INVERSE_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(upper);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_orgqr(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & input2) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::orgqr(dispatchKeySet, self, input2);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ORGQR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(input2);
  return tt;
}

at::Tensor & wrap_orgqr_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & input2, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::orgqr_outf(dispatchKeySet, self, input2, out);
  }
  bool flush = register_in_place(out, H_ORGQR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(input2);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_ormqr_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & input2, const at::Tensor & input3, bool left, bool transpose, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::ormqr_outf(dispatchKeySet, self, input2, input3, left, transpose, out);
  }
  bool flush = register_in_place(out, H_ORMQR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(input2);trace.append_arg(input3);trace.append_arg(left);trace.append_arg(transpose);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_ormqr(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & input2, const at::Tensor & input3, bool left, bool transpose) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::ormqr(dispatchKeySet, self, input2, input3, left, transpose);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ORMQR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(input2);trace.append_arg(input3);trace.append_arg(left);trace.append_arg(transpose);
  return tt;
}

at::Tensor & wrap_lu_solve_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & LU_data, const at::Tensor & LU_pivots, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::lu_solve_outf(dispatchKeySet, self, LU_data, LU_pivots, out);
  }
  bool flush = register_in_place(out, H_LU_SOLVE_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(LU_data);trace.append_arg(LU_pivots);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_lu_solve(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & LU_data, const at::Tensor & LU_pivots) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::lu_solve(dispatchKeySet, self, LU_data, LU_pivots);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LU_SOLVE, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(LU_data);trace.append_arg(LU_pivots);
  return tt;
}

at::Tensor & wrap_multinomial_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t num_samples, bool replacement, c10::optional<at::Generator> generator, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::multinomial_outf(dispatchKeySet, self, num_samples, replacement, std::move(generator), out);
  }
  bool flush = register_in_place(out, H_MULTINOMIAL_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(num_samples);trace.append_arg(replacement);trace.append_arg(std::move(generator));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_multinomial(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t num_samples, bool replacement, c10::optional<at::Generator> generator) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::multinomial(dispatchKeySet, self, num_samples, replacement, std::move(generator));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MULTINOMIAL, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(num_samples);trace.append_arg(replacement);trace.append_arg(std::move(generator));
  return tt;
}

at::Tensor & wrap_lgamma_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::lgamma_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_LGAMMA_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_digamma_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::digamma_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_DIGAMMA_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_polygamma_out(c10::DispatchKeySet dispatchKeySet, int64_t n, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::polygamma_outf(dispatchKeySet, n, self, out);
  }
  bool flush = register_in_place(out, H_POLYGAMMA_OUT, dispatchKeySet);
  trace.append_arg(n);trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_polygamma_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, int64_t n) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::polygamma_(dispatchKeySet, self, n);
  }
  bool flush = register_in_place(self, H_POLYGAMMA_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(n);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_erfinv_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::erfinv_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_ERFINV_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_i0_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::i0_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_I0_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_sign(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::sign(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SIGN, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_sign_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::sign_(dispatchKeySet, self);
  }
  bool flush = register_in_place(self, H_SIGN_, dispatchKeySet);
  trace.append_arg(self);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_sign_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::sign_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_SIGN_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_signbit(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::signbit(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SIGNBIT, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_signbit_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::signbit_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_SIGNBIT_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_dist(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, const at::Scalar & p) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::dist(dispatchKeySet, self, other, p);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_DIST, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(p);
  return tt;
}

at::Tensor & wrap_atan2_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::atan2_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_ATAN2_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_lerp_Scalar_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & end, const at::Scalar & weight, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::lerp_outf(dispatchKeySet, self, end, weight, out);
  }
  bool flush = register_in_place(out, H_LERP_SCALAR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(end);trace.append_arg(weight);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_lerp_Tensor_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & end, const at::Tensor & weight, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::lerp_outf(dispatchKeySet, self, end, weight, out);
  }
  bool flush = register_in_place(out, H_LERP_TENSOR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(end);trace.append_arg(weight);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_lerp_Scalar(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & end, const at::Scalar & weight) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::lerp(dispatchKeySet, self, end, weight);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LERP_SCALAR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(end);trace.append_arg(weight);
  return tt;
}

at::Tensor wrap_lerp_Tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & end, const at::Tensor & weight) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::lerp(dispatchKeySet, self, end, weight);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LERP_TENSOR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(end);trace.append_arg(weight);
  return tt;
}

at::Tensor & wrap_histc_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t bins, const at::Scalar & min, const at::Scalar & max, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::histc_outf(dispatchKeySet, self, bins, min, max, out);
  }
  bool flush = register_in_place(out, H_HISTC_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(bins);trace.append_arg(min);trace.append_arg(max);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_histc(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t bins, const at::Scalar & min, const at::Scalar & max) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::histc(dispatchKeySet, self, bins, min, max);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_HISTC, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(bins);trace.append_arg(min);trace.append_arg(max);
  return tt;
}

at::Tensor & wrap_fmod_Scalar_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fmod_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_FMOD_SCALAR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_fmod_Scalar(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fmod(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_FMOD_SCALAR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_fmod_Tensor_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fmod_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_FMOD_TENSOR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_fmod_Tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fmod(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_FMOD_TENSOR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_hypot_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::hypot_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_HYPOT_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_hypot_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::hypot_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_HYPOT_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_igamma_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::igamma_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_IGAMMA_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_igammac_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::igammac_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_IGAMMAC_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_nextafter_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::nextafter_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_NEXTAFTER_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_nextafter_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::nextafter_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_NEXTAFTER_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_remainder_Scalar_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::remainder_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_REMAINDER_SCALAR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_remainder_Scalar(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::remainder(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_REMAINDER_SCALAR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_remainder__Scalar(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::remainder_(dispatchKeySet, self, other);
  }
  bool flush = register_in_place(self, H_REMAINDER__SCALAR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_remainder_Tensor_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::remainder_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_REMAINDER_TENSOR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_remainder_Scalar_Tensor(c10::DispatchKeySet dispatchKeySet, const at::Scalar & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::remainder(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_REMAINDER_SCALAR_TENSOR, other.dtype(), other.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor wrap_min(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::min(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MIN, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_fmin_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fmin_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_FMIN_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_max(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::max(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MAX, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_fmax_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fmax_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_FMAX_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_maximum_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::maximum_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_MAXIMUM_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_max_other(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::max(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MAX_OTHER, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_max_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::max_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_MAX_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_minimum_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::minimum_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_MINIMUM_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_min_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::min_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_MIN_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_min_other(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::min(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MIN_OTHER, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_quantile_scalar_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, double q, c10::optional<int64_t> dim, bool keepdim, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::quantile_outf(dispatchKeySet, self, q, dim, keepdim, out);
  }
  bool flush = register_in_place(out, H_QUANTILE_SCALAR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(q);trace.append_arg(dim);trace.append_arg(keepdim);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_quantile_scalar(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, double q, c10::optional<int64_t> dim, bool keepdim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::quantile(dispatchKeySet, self, q, dim, keepdim);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_QUANTILE_SCALAR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(q);trace.append_arg(dim);trace.append_arg(keepdim);
  return tt;
}

at::Tensor & wrap_quantile_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & q, c10::optional<int64_t> dim, bool keepdim, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::quantile_outf(dispatchKeySet, self, q, dim, keepdim, out);
  }
  bool flush = register_in_place(out, H_QUANTILE_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(q);trace.append_arg(dim);trace.append_arg(keepdim);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_quantile(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & q, c10::optional<int64_t> dim, bool keepdim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::quantile(dispatchKeySet, self, q, dim, keepdim);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_QUANTILE, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(q);trace.append_arg(dim);trace.append_arg(keepdim);
  return tt;
}

at::Tensor & wrap_nanquantile_scalar_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, double q, c10::optional<int64_t> dim, bool keepdim, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::nanquantile_outf(dispatchKeySet, self, q, dim, keepdim, out);
  }
  bool flush = register_in_place(out, H_NANQUANTILE_SCALAR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(q);trace.append_arg(dim);trace.append_arg(keepdim);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_nanquantile_scalar(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, double q, c10::optional<int64_t> dim, bool keepdim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::nanquantile(dispatchKeySet, self, q, dim, keepdim);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_NANQUANTILE_SCALAR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(q);trace.append_arg(dim);trace.append_arg(keepdim);
  return tt;
}

at::Tensor & wrap_nanquantile_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & q, c10::optional<int64_t> dim, bool keepdim, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::nanquantile_outf(dispatchKeySet, self, q, dim, keepdim, out);
  }
  bool flush = register_in_place(out, H_NANQUANTILE_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(q);trace.append_arg(dim);trace.append_arg(keepdim);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_nanquantile(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & q, c10::optional<int64_t> dim, bool keepdim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::nanquantile(dispatchKeySet, self, q, dim, keepdim);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_NANQUANTILE, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(q);trace.append_arg(dim);trace.append_arg(keepdim);
  return tt;
}

at::Tensor & wrap_quantile_new_scalar_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, double q, c10::optional<int64_t> dim, bool keepdim, c10::string_view interpolation, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::quantile_outf(dispatchKeySet, self, q, dim, keepdim, std::move(interpolation), out);
  }
  bool flush = register_in_place(out, H_QUANTILE_NEW_SCALAR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(q);trace.append_arg(dim);trace.append_arg(keepdim);trace.append_arg(std::move(interpolation));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_quantile_new_scalar(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, double q, c10::optional<int64_t> dim, bool keepdim, c10::string_view interpolation) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::quantile(dispatchKeySet, self, q, dim, keepdim, std::move(interpolation));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_QUANTILE_NEW_SCALAR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(q);trace.append_arg(dim);trace.append_arg(keepdim);trace.append_arg(std::move(interpolation));
  return tt;
}

at::Tensor & wrap_quantile_new_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & q, c10::optional<int64_t> dim, bool keepdim, c10::string_view interpolation, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::quantile_outf(dispatchKeySet, self, q, dim, keepdim, std::move(interpolation), out);
  }
  bool flush = register_in_place(out, H_QUANTILE_NEW_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(q);trace.append_arg(dim);trace.append_arg(keepdim);trace.append_arg(std::move(interpolation));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_quantile_new(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & q, c10::optional<int64_t> dim, bool keepdim, c10::string_view interpolation) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::quantile(dispatchKeySet, self, q, dim, keepdim, std::move(interpolation));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_QUANTILE_NEW, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(q);trace.append_arg(dim);trace.append_arg(keepdim);trace.append_arg(std::move(interpolation));
  return tt;
}

at::Tensor & wrap_nanquantile_new_scalar_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, double q, c10::optional<int64_t> dim, bool keepdim, c10::string_view interpolation, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::nanquantile_outf(dispatchKeySet, self, q, dim, keepdim, std::move(interpolation), out);
  }
  bool flush = register_in_place(out, H_NANQUANTILE_NEW_SCALAR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(q);trace.append_arg(dim);trace.append_arg(keepdim);trace.append_arg(std::move(interpolation));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_nanquantile_new_scalar(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, double q, c10::optional<int64_t> dim, bool keepdim, c10::string_view interpolation) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::nanquantile(dispatchKeySet, self, q, dim, keepdim, std::move(interpolation));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_NANQUANTILE_NEW_SCALAR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(q);trace.append_arg(dim);trace.append_arg(keepdim);trace.append_arg(std::move(interpolation));
  return tt;
}

at::Tensor & wrap_nanquantile_new_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & q, c10::optional<int64_t> dim, bool keepdim, c10::string_view interpolation, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::nanquantile_outf(dispatchKeySet, self, q, dim, keepdim, std::move(interpolation), out);
  }
  bool flush = register_in_place(out, H_NANQUANTILE_NEW_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(q);trace.append_arg(dim);trace.append_arg(keepdim);trace.append_arg(std::move(interpolation));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_nanquantile_new(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & q, c10::optional<int64_t> dim, bool keepdim, c10::string_view interpolation) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::nanquantile(dispatchKeySet, self, q, dim, keepdim, std::move(interpolation));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_NANQUANTILE_NEW, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(q);trace.append_arg(dim);trace.append_arg(keepdim);trace.append_arg(std::move(interpolation));
  return tt;
}

at::Tensor & wrap_msort_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::msort_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_MSORT_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_msort(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::msort(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MSORT, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor wrap_argsort(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, bool descending) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::argsort(dispatchKeySet, self, dim, descending);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ARGSORT, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(descending);
  return tt;
}

at::Tensor wrap_argsort_dimname(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, bool descending) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::argsort(dispatchKeySet, self, std::move(dim), descending);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ARGSORT_DIMNAME, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim));trace.append_arg(descending);
  return tt;
}

at::Tensor wrap_all(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::all(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ALL, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor wrap_any(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::any(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ANY, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_renorm_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & p, int64_t dim, const at::Scalar & maxnorm, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::renorm_outf(dispatchKeySet, self, p, dim, maxnorm, out);
  }
  bool flush = register_in_place(out, H_RENORM_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(p);trace.append_arg(dim);trace.append_arg(maxnorm);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_unfold(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dimension, int64_t size, int64_t step) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::unfold(dispatchKeySet, self, dimension, size, step);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_UNFOLD, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(dimension);trace.append_arg(size);trace.append_arg(step);
  return tt;
}

at::Tensor wrap_unfold_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_in, at::IntArrayRef input_sizes, int64_t dim, int64_t size, int64_t step) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::unfold_backward(dispatchKeySet, grad_in, std::move(input_sizes), dim, size, step);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_UNFOLD_BACKWARD, grad_in.dtype(), grad_in.device());
  trace.append_arg(grad_in);trace.append_arg(std::move(input_sizes));trace.append_arg(dim);trace.append_arg(size);trace.append_arg(step);
  return tt;
}

at::Tensor & wrap_pow_Tensor_Tensor_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & exponent, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::pow_outf(dispatchKeySet, self, exponent, out);
  }
  bool flush = register_in_place(out, H_POW_TENSOR_TENSOR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(exponent);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_pow_Scalar_out(c10::DispatchKeySet dispatchKeySet, const at::Scalar & self, const at::Tensor & exponent, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::pow_outf(dispatchKeySet, self, exponent, out);
  }
  bool flush = register_in_place(out, H_POW_SCALAR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(exponent);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_pow_Tensor_Scalar_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & exponent, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::pow_outf(dispatchKeySet, self, exponent, out);
  }
  bool flush = register_in_place(out, H_POW_TENSOR_SCALAR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(exponent);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_pow_Tensor_Scalar(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & exponent) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::pow(dispatchKeySet, self, exponent);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_POW_TENSOR_SCALAR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(exponent);
  return tt;
}

at::Tensor & wrap_float_power_Tensor_Tensor_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & exponent, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::float_power_outf(dispatchKeySet, self, exponent, out);
  }
  bool flush = register_in_place(out, H_FLOAT_POWER_TENSOR_TENSOR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(exponent);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_float_power_Tensor_Tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & exponent) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::float_power(dispatchKeySet, self, exponent);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_FLOAT_POWER_TENSOR_TENSOR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(exponent);
  return tt;
}

at::Tensor & wrap_float_power_Scalar_out(c10::DispatchKeySet dispatchKeySet, const at::Scalar & self, const at::Tensor & exponent, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::float_power_outf(dispatchKeySet, self, exponent, out);
  }
  bool flush = register_in_place(out, H_FLOAT_POWER_SCALAR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(exponent);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_float_power_Scalar(c10::DispatchKeySet dispatchKeySet, const at::Scalar & self, const at::Tensor & exponent) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::float_power(dispatchKeySet, self, exponent);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_FLOAT_POWER_SCALAR, exponent.dtype(), exponent.device());
  trace.append_arg(self);trace.append_arg(exponent);
  return tt;
}

at::Tensor & wrap_float_power_Tensor_Scalar_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & exponent, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::float_power_outf(dispatchKeySet, self, exponent, out);
  }
  bool flush = register_in_place(out, H_FLOAT_POWER_TENSOR_SCALAR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(exponent);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_float_power_Tensor_Scalar(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & exponent) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::float_power(dispatchKeySet, self, exponent);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_FLOAT_POWER_TENSOR_SCALAR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(exponent);
  return tt;
}

at::Tensor & wrap_float_power__Scalar(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & exponent) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::float_power_(dispatchKeySet, self, exponent);
  }
  bool flush = register_in_place(self, H_FLOAT_POWER__SCALAR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(exponent);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_float_power__Tensor(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & exponent) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::float_power_(dispatchKeySet, self, exponent);
  }
  bool flush = register_in_place(self, H_FLOAT_POWER__TENSOR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(exponent);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_normal_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, double mean, double std, c10::optional<at::Generator> generator) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::normal_(dispatchKeySet, self, mean, std, std::move(generator));
  }
  bool flush = register_in_place(self, H_NORMAL_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(mean);trace.append_arg(std);trace.append_arg(std::move(generator));
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_normal_Tensor_float_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & mean, double std, c10::optional<at::Generator> generator, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::normal_outf(dispatchKeySet, mean, std, std::move(generator), out);
  }
  bool flush = register_in_place(out, H_NORMAL_TENSOR_FLOAT_OUT, dispatchKeySet);
  trace.append_arg(mean);trace.append_arg(std);trace.append_arg(std::move(generator));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_normal_Tensor_float(c10::DispatchKeySet dispatchKeySet, const at::Tensor & mean, double std, c10::optional<at::Generator> generator) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::normal(dispatchKeySet, mean, std, std::move(generator));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_NORMAL_TENSOR_FLOAT, mean.dtype(), mean.device());
  trace.append_arg(mean);trace.append_arg(std);trace.append_arg(std::move(generator));
  return tt;
}

at::Tensor & wrap_normal_float_Tensor_out(c10::DispatchKeySet dispatchKeySet, double mean, const at::Tensor & std, c10::optional<at::Generator> generator, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::normal_outf(dispatchKeySet, mean, std, std::move(generator), out);
  }
  bool flush = register_in_place(out, H_NORMAL_FLOAT_TENSOR_OUT, dispatchKeySet);
  trace.append_arg(mean);trace.append_arg(std);trace.append_arg(std::move(generator));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_normal_float_Tensor(c10::DispatchKeySet dispatchKeySet, double mean, const at::Tensor & std, c10::optional<at::Generator> generator) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::normal(dispatchKeySet, mean, std, std::move(generator));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_NORMAL_FLOAT_TENSOR, std.dtype(), std.device());
  trace.append_arg(mean);trace.append_arg(std);trace.append_arg(std::move(generator));
  return tt;
}

at::Tensor & wrap_normal_Tensor_Tensor_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & mean, const at::Tensor & std, c10::optional<at::Generator> generator, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::normal_outf(dispatchKeySet, mean, std, std::move(generator), out);
  }
  bool flush = register_in_place(out, H_NORMAL_TENSOR_TENSOR_OUT, dispatchKeySet);
  trace.append_arg(mean);trace.append_arg(std);trace.append_arg(std::move(generator));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_normal_Tensor_Tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & mean, const at::Tensor & std, c10::optional<at::Generator> generator) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::normal(dispatchKeySet, mean, std, std::move(generator));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_NORMAL_TENSOR_TENSOR, mean.dtype(), mean.device());
  trace.append_arg(mean);trace.append_arg(std);trace.append_arg(std::move(generator));
  return tt;
}

at::Tensor wrap_normal_float_float(c10::DispatchKeySet dispatchKeySet, double mean, double std, at::IntArrayRef size, c10::optional<at::Generator> generator, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::normal(dispatchKeySet, mean, std, std::move(size), std::move(generator), std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_NORMAL_FLOAT_FLOAT, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(mean);trace.append_arg(std);trace.append_arg(std::move(size));trace.append_arg(std::move(generator));trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor & wrap_normal_float_float_out(c10::DispatchKeySet dispatchKeySet, double mean, double std, at::IntArrayRef size, c10::optional<at::Generator> generator, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::normal_outf(dispatchKeySet, mean, std, std::move(size), std::move(generator), out);
  }
  bool flush = register_in_place(out, H_NORMAL_FLOAT_FLOAT_OUT, dispatchKeySet);
  trace.append_arg(mean);trace.append_arg(std);trace.append_arg(std::move(size));trace.append_arg(std::move(generator));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_alias(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::alias(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ALIAS, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap__index_copy_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_index_copy_(dispatchKeySet, self, dim, index, source);
  }
  bool flush = register_in_place(self, H__INDEX_COPY_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(index);trace.append_arg(source);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap__cumsum(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_cumsum(dispatchKeySet, self, dim);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__CUMSUM, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(dim);
  return tt;
}

at::Tensor & wrap__cumsum_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_cumsum_outf(dispatchKeySet, self, dim, out);
  }
  bool flush = register_in_place(out, H__CUMSUM_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap__cumprod(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_cumprod(dispatchKeySet, self, dim);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__CUMPROD, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(dim);
  return tt;
}

at::Tensor & wrap__cumprod_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_cumprod_outf(dispatchKeySet, self, dim, out);
  }
  bool flush = register_in_place(out, H__CUMPROD_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap__amp_update_scale_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, at::Tensor & growth_tracker, const at::Tensor & found_inf, double scale_growth_factor, double scale_backoff_factor, int64_t growth_interval) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_amp_update_scale_(dispatchKeySet, self, growth_tracker, found_inf, scale_growth_factor, scale_backoff_factor, growth_interval);
  }
  bool flush = register_in_place(self, H__AMP_UPDATE_SCALE_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(growth_tracker);trace.append_arg(found_inf);trace.append_arg(scale_growth_factor);trace.append_arg(scale_backoff_factor);trace.append_arg(growth_interval);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap__cat(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors, int64_t dim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_cat(dispatchKeySet, std::move(tensors), dim);
  }
  auto defaults = compute_dtype(tensors);
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H__CAT, default_dtype, default_device);
  trace.append_arg(std::move(tensors));trace.append_arg(dim);
  return tt;
}

at::Tensor & wrap__cat_out(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors, int64_t dim, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_cat_outf(dispatchKeySet, std::move(tensors), dim, out);
  }
  bool flush = register_in_place(out, H__CAT_OUT, dispatchKeySet);
  trace.append_arg(std::move(tensors));trace.append_arg(dim);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_bucketize_Tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & boundaries, bool out_int32, bool right) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::bucketize(dispatchKeySet, self, boundaries, out_int32, right);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_BUCKETIZE_TENSOR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(boundaries);trace.append_arg(out_int32);trace.append_arg(right);
  return tt;
}

at::Tensor & wrap_bucketize_Tensor_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & boundaries, bool out_int32, bool right, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::bucketize_outf(dispatchKeySet, self, boundaries, out_int32, right, out);
  }
  bool flush = register_in_place(out, H_BUCKETIZE_TENSOR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(boundaries);trace.append_arg(out_int32);trace.append_arg(right);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_bucketize_Scalar(c10::DispatchKeySet dispatchKeySet, const at::Scalar & self, const at::Tensor & boundaries, bool out_int32, bool right) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::bucketize(dispatchKeySet, self, boundaries, out_int32, right);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_BUCKETIZE_SCALAR, boundaries.dtype(), boundaries.device());
  trace.append_arg(self);trace.append_arg(boundaries);trace.append_arg(out_int32);trace.append_arg(right);
  return tt;
}

at::Tensor wrap_searchsorted_Tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & sorted_sequence, const at::Tensor & self, bool out_int32, bool right) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::searchsorted(dispatchKeySet, sorted_sequence, self, out_int32, right);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SEARCHSORTED_TENSOR, sorted_sequence.dtype(), sorted_sequence.device());
  trace.append_arg(sorted_sequence);trace.append_arg(self);trace.append_arg(out_int32);trace.append_arg(right);
  return tt;
}

at::Tensor & wrap_searchsorted_Tensor_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & sorted_sequence, const at::Tensor & self, bool out_int32, bool right, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::searchsorted_outf(dispatchKeySet, sorted_sequence, self, out_int32, right, out);
  }
  bool flush = register_in_place(out, H_SEARCHSORTED_TENSOR_OUT, dispatchKeySet);
  trace.append_arg(sorted_sequence);trace.append_arg(self);trace.append_arg(out_int32);trace.append_arg(right);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_searchsorted_Scalar(c10::DispatchKeySet dispatchKeySet, const at::Tensor & sorted_sequence, const at::Scalar & self, bool out_int32, bool right) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::searchsorted(dispatchKeySet, sorted_sequence, self, out_int32, right);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SEARCHSORTED_SCALAR, sorted_sequence.dtype(), sorted_sequence.device());
  trace.append_arg(sorted_sequence);trace.append_arg(self);trace.append_arg(out_int32);trace.append_arg(right);
  return tt;
}

at::Tensor & wrap_mse_loss_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::mse_loss_outf(dispatchKeySet, self, target, reduction, out);
  }
  bool flush = register_in_place(out, H_MSE_LOSS_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(target);trace.append_arg(reduction);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_mse_loss(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, int64_t reduction) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::mse_loss(dispatchKeySet, self, target, reduction);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MSE_LOSS, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(target);trace.append_arg(reduction);
  return tt;
}

at::Tensor & wrap_mse_loss_backward_grad_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::mse_loss_backward_outf(dispatchKeySet, grad_output, self, target, reduction, grad_input);
  }
  bool flush = register_in_place(grad_input, H_MSE_LOSS_BACKWARD_GRAD_INPUT, dispatchKeySet);
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(target);trace.append_arg(reduction);trace.append_arg(grad_input);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return grad_input;
}

at::Tensor wrap_mse_loss_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::mse_loss_backward(dispatchKeySet, grad_output, self, target, reduction);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MSE_LOSS_BACKWARD, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(target);trace.append_arg(reduction);
  return tt;
}

at::Tensor & wrap_l1_loss_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::l1_loss_outf(dispatchKeySet, self, target, reduction, out);
  }
  bool flush = register_in_place(out, H_L1_LOSS_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(target);trace.append_arg(reduction);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_l1_loss(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, int64_t reduction) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::l1_loss(dispatchKeySet, self, target, reduction);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_L1_LOSS, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(target);trace.append_arg(reduction);
  return tt;
}

at::Tensor & wrap_l1_loss_backward_grad_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::l1_loss_backward_outf(dispatchKeySet, grad_output, self, target, reduction, grad_input);
  }
  bool flush = register_in_place(grad_input, H_L1_LOSS_BACKWARD_GRAD_INPUT, dispatchKeySet);
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(target);trace.append_arg(reduction);trace.append_arg(grad_input);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return grad_input;
}

at::Tensor wrap_l1_loss_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::l1_loss_backward(dispatchKeySet, grad_output, self, target, reduction);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_L1_LOSS_BACKWARD, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(target);trace.append_arg(reduction);
  return tt;
}

at::Tensor & wrap_multi_margin_loss_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, const at::Scalar & p, const at::Scalar & margin, const c10::optional<at::Tensor> & weight, int64_t reduction, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::multi_margin_loss_outf(dispatchKeySet, self, target, p, margin, weight, reduction, out);
  }
  bool flush = register_in_place(out, H_MULTI_MARGIN_LOSS_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(target);trace.append_arg(p);trace.append_arg(margin);trace.append_arg(weight);trace.append_arg(reduction);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_multi_margin_loss(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, const at::Scalar & p, const at::Scalar & margin, const c10::optional<at::Tensor> & weight, int64_t reduction) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::multi_margin_loss(dispatchKeySet, self, target, p, margin, weight, reduction);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MULTI_MARGIN_LOSS, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(target);trace.append_arg(p);trace.append_arg(margin);trace.append_arg(weight);trace.append_arg(reduction);
  return tt;
}

at::Tensor & wrap_multi_margin_loss_backward_grad_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const at::Scalar & p, const at::Scalar & margin, const c10::optional<at::Tensor> & weight, int64_t reduction, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::multi_margin_loss_backward_outf(dispatchKeySet, grad_output, self, target, p, margin, weight, reduction, grad_input);
  }
  bool flush = register_in_place(grad_input, H_MULTI_MARGIN_LOSS_BACKWARD_GRAD_INPUT, dispatchKeySet);
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(target);trace.append_arg(p);trace.append_arg(margin);trace.append_arg(weight);trace.append_arg(reduction);trace.append_arg(grad_input);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return grad_input;
}

at::Tensor wrap_multi_margin_loss_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const at::Scalar & p, const at::Scalar & margin, const c10::optional<at::Tensor> & weight, int64_t reduction) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::multi_margin_loss_backward(dispatchKeySet, grad_output, self, target, p, margin, weight, reduction);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MULTI_MARGIN_LOSS_BACKWARD, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(target);trace.append_arg(p);trace.append_arg(margin);trace.append_arg(weight);trace.append_arg(reduction);
  return tt;
}

at::Tensor & wrap_multilabel_margin_loss_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::multilabel_margin_loss_outf(dispatchKeySet, self, target, reduction, out);
  }
  bool flush = register_in_place(out, H_MULTILABEL_MARGIN_LOSS_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(target);trace.append_arg(reduction);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_multilabel_margin_loss(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, int64_t reduction) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::multilabel_margin_loss(dispatchKeySet, self, target, reduction);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MULTILABEL_MARGIN_LOSS, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(target);trace.append_arg(reduction);
  return tt;
}

at::Tensor & wrap_multilabel_margin_loss_backward_grad_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, const at::Tensor & is_target, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::multilabel_margin_loss_backward_outf(dispatchKeySet, grad_output, self, target, reduction, is_target, grad_input);
  }
  bool flush = register_in_place(grad_input, H_MULTILABEL_MARGIN_LOSS_BACKWARD_GRAD_INPUT, dispatchKeySet);
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(target);trace.append_arg(reduction);trace.append_arg(is_target);trace.append_arg(grad_input);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return grad_input;
}

at::Tensor wrap_multilabel_margin_loss_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, const at::Tensor & is_target) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::multilabel_margin_loss_backward(dispatchKeySet, grad_output, self, target, reduction, is_target);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MULTILABEL_MARGIN_LOSS_BACKWARD, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(target);trace.append_arg(reduction);trace.append_arg(is_target);
  return tt;
}

at::Tensor & wrap_nll_loss_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::nll_loss_outf(dispatchKeySet, self, target, weight, reduction, ignore_index, out);
  }
  bool flush = register_in_place(out, H_NLL_LOSS_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(target);trace.append_arg(weight);trace.append_arg(reduction);trace.append_arg(ignore_index);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_nll_loss_nd(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::nll_loss_nd(dispatchKeySet, self, target, weight, reduction, ignore_index);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_NLL_LOSS_ND, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(target);trace.append_arg(weight);trace.append_arg(reduction);trace.append_arg(ignore_index);
  return tt;
}

at::Tensor wrap_nll_loss(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::nll_loss(dispatchKeySet, self, target, weight, reduction, ignore_index);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_NLL_LOSS, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(target);trace.append_arg(weight);trace.append_arg(reduction);trace.append_arg(ignore_index);
  return tt;
}

at::Tensor & wrap_nll_loss_backward_grad_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, const at::Tensor & total_weight, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::nll_loss_backward_outf(dispatchKeySet, grad_output, self, target, weight, reduction, ignore_index, total_weight, grad_input);
  }
  bool flush = register_in_place(grad_input, H_NLL_LOSS_BACKWARD_GRAD_INPUT, dispatchKeySet);
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(target);trace.append_arg(weight);trace.append_arg(reduction);trace.append_arg(ignore_index);trace.append_arg(total_weight);trace.append_arg(grad_input);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return grad_input;
}

at::Tensor wrap_nll_loss_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, const at::Tensor & total_weight) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::nll_loss_backward(dispatchKeySet, grad_output, self, target, weight, reduction, ignore_index, total_weight);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_NLL_LOSS_BACKWARD, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(target);trace.append_arg(weight);trace.append_arg(reduction);trace.append_arg(ignore_index);trace.append_arg(total_weight);
  return tt;
}

at::Tensor & wrap_nll_loss2d_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::nll_loss2d_outf(dispatchKeySet, self, target, weight, reduction, ignore_index, out);
  }
  bool flush = register_in_place(out, H_NLL_LOSS2D_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(target);trace.append_arg(weight);trace.append_arg(reduction);trace.append_arg(ignore_index);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_nll_loss2d(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::nll_loss2d(dispatchKeySet, self, target, weight, reduction, ignore_index);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_NLL_LOSS2D, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(target);trace.append_arg(weight);trace.append_arg(reduction);trace.append_arg(ignore_index);
  return tt;
}

at::Tensor & wrap_nll_loss2d_backward_grad_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, const at::Tensor & total_weight, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::nll_loss2d_backward_outf(dispatchKeySet, grad_output, self, target, weight, reduction, ignore_index, total_weight, grad_input);
  }
  bool flush = register_in_place(grad_input, H_NLL_LOSS2D_BACKWARD_GRAD_INPUT, dispatchKeySet);
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(target);trace.append_arg(weight);trace.append_arg(reduction);trace.append_arg(ignore_index);trace.append_arg(total_weight);trace.append_arg(grad_input);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return grad_input;
}

at::Tensor wrap_nll_loss2d_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, const at::Tensor & total_weight) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::nll_loss2d_backward(dispatchKeySet, grad_output, self, target, weight, reduction, ignore_index, total_weight);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_NLL_LOSS2D_BACKWARD, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(target);trace.append_arg(weight);trace.append_arg(reduction);trace.append_arg(ignore_index);trace.append_arg(total_weight);
  return tt;
}

at::Tensor & wrap_smooth_l1_loss_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, int64_t reduction, double beta, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::smooth_l1_loss_outf(dispatchKeySet, self, target, reduction, beta, out);
  }
  bool flush = register_in_place(out, H_SMOOTH_L1_LOSS_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(target);trace.append_arg(reduction);trace.append_arg(beta);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_smooth_l1_loss(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, int64_t reduction, double beta) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::smooth_l1_loss(dispatchKeySet, self, target, reduction, beta);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SMOOTH_L1_LOSS, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(target);trace.append_arg(reduction);trace.append_arg(beta);
  return tt;
}

at::Tensor & wrap_smooth_l1_loss_backward_grad_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, double beta, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::smooth_l1_loss_backward_outf(dispatchKeySet, grad_output, self, target, reduction, beta, grad_input);
  }
  bool flush = register_in_place(grad_input, H_SMOOTH_L1_LOSS_BACKWARD_GRAD_INPUT, dispatchKeySet);
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(target);trace.append_arg(reduction);trace.append_arg(beta);trace.append_arg(grad_input);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return grad_input;
}

at::Tensor wrap_smooth_l1_loss_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, double beta) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::smooth_l1_loss_backward(dispatchKeySet, grad_output, self, target, reduction, beta);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SMOOTH_L1_LOSS_BACKWARD, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(target);trace.append_arg(reduction);trace.append_arg(beta);
  return tt;
}

at::Tensor & wrap_huber_loss_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, int64_t reduction, double delta, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::huber_loss_outf(dispatchKeySet, self, target, reduction, delta, out);
  }
  bool flush = register_in_place(out, H_HUBER_LOSS_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(target);trace.append_arg(reduction);trace.append_arg(delta);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_huber_loss(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, int64_t reduction, double delta) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::huber_loss(dispatchKeySet, self, target, reduction, delta);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_HUBER_LOSS, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(target);trace.append_arg(reduction);trace.append_arg(delta);
  return tt;
}

at::Tensor & wrap_huber_loss_backward_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, double delta, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::huber_loss_backward_outf(dispatchKeySet, grad_output, self, target, reduction, delta, grad_input);
  }
  bool flush = register_in_place(grad_input, H_HUBER_LOSS_BACKWARD_OUT, dispatchKeySet);
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(target);trace.append_arg(reduction);trace.append_arg(delta);trace.append_arg(grad_input);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return grad_input;
}

at::Tensor wrap_huber_loss_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, double delta) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::huber_loss_backward(dispatchKeySet, grad_output, self, target, reduction, delta);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_HUBER_LOSS_BACKWARD, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(target);trace.append_arg(reduction);trace.append_arg(delta);
  return tt;
}

at::Tensor & wrap_soft_margin_loss_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::soft_margin_loss_outf(dispatchKeySet, self, target, reduction, out);
  }
  bool flush = register_in_place(out, H_SOFT_MARGIN_LOSS_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(target);trace.append_arg(reduction);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_soft_margin_loss(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, int64_t reduction) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::soft_margin_loss(dispatchKeySet, self, target, reduction);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SOFT_MARGIN_LOSS, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(target);trace.append_arg(reduction);
  return tt;
}

at::Tensor & wrap_soft_margin_loss_backward_grad_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::soft_margin_loss_backward_outf(dispatchKeySet, grad_output, self, target, reduction, grad_input);
  }
  bool flush = register_in_place(grad_input, H_SOFT_MARGIN_LOSS_BACKWARD_GRAD_INPUT, dispatchKeySet);
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(target);trace.append_arg(reduction);trace.append_arg(grad_input);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return grad_input;
}

at::Tensor wrap_soft_margin_loss_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::soft_margin_loss_backward(dispatchKeySet, grad_output, self, target, reduction);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SOFT_MARGIN_LOSS_BACKWARD, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(target);trace.append_arg(reduction);
  return tt;
}

at::Tensor & wrap_elu_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::elu_outf(dispatchKeySet, self, alpha, scale, input_scale, out);
  }
  bool flush = register_in_place(out, H_ELU_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(alpha);trace.append_arg(scale);trace.append_arg(input_scale);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_elu_backward_grad_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale, bool is_result, const at::Tensor & self_or_result, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::elu_backward_outf(dispatchKeySet, grad_output, alpha, scale, input_scale, is_result, self_or_result, grad_input);
  }
  bool flush = register_in_place(grad_input, H_ELU_BACKWARD_GRAD_INPUT, dispatchKeySet);
  trace.append_arg(grad_output);trace.append_arg(alpha);trace.append_arg(scale);trace.append_arg(input_scale);trace.append_arg(is_result);trace.append_arg(self_or_result);trace.append_arg(grad_input);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return grad_input;
}

at::Tensor & wrap_elu_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::elu_(dispatchKeySet, self, alpha, scale, input_scale);
  }
  bool flush = register_in_place(self, H_ELU_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(alpha);trace.append_arg(scale);trace.append_arg(input_scale);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_glu_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::glu_outf(dispatchKeySet, self, dim, out);
  }
  bool flush = register_in_place(out, H_GLU_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(dim);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_glu(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::glu(dispatchKeySet, self, dim);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_GLU, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(dim);
  return tt;
}

at::Tensor & wrap_glu_backward_grad_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, int64_t dim, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::glu_backward_outf(dispatchKeySet, grad_output, self, dim, grad_input);
  }
  bool flush = register_in_place(grad_input, H_GLU_BACKWARD_GRAD_INPUT, dispatchKeySet);
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(dim);trace.append_arg(grad_input);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return grad_input;
}

at::Tensor wrap_glu_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, int64_t dim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::glu_backward(dispatchKeySet, grad_output, self, dim);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_GLU_BACKWARD, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(dim);
  return tt;
}

at::Tensor & wrap_hardsigmoid_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::hardsigmoid_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_HARDSIGMOID_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_hardsigmoid(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::hardsigmoid(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_HARDSIGMOID, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_hardsigmoid_backward_grad_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::hardsigmoid_backward_outf(dispatchKeySet, grad_output, self, grad_input);
  }
  bool flush = register_in_place(grad_input, H_HARDSIGMOID_BACKWARD_GRAD_INPUT, dispatchKeySet);
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(grad_input);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return grad_input;
}

at::Tensor & wrap_hardtanh_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::hardtanh_outf(dispatchKeySet, self, min_val, max_val, out);
  }
  bool flush = register_in_place(out, H_HARDTANH_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(min_val);trace.append_arg(max_val);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_hardtanh(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::hardtanh(dispatchKeySet, self, min_val, max_val);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_HARDTANH, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(min_val);trace.append_arg(max_val);
  return tt;
}

at::Tensor & wrap_hardtanh_backward_grad_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::hardtanh_backward_outf(dispatchKeySet, grad_output, self, min_val, max_val, grad_input);
  }
  bool flush = register_in_place(grad_input, H_HARDTANH_BACKWARD_GRAD_INPUT, dispatchKeySet);
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(min_val);trace.append_arg(max_val);trace.append_arg(grad_input);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return grad_input;
}

at::Tensor wrap_hardtanh_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::hardtanh_backward(dispatchKeySet, grad_output, self, min_val, max_val);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_HARDTANH_BACKWARD, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(min_val);trace.append_arg(max_val);
  return tt;
}

at::Tensor & wrap_hardtanh_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::hardtanh_(dispatchKeySet, self, min_val, max_val);
  }
  bool flush = register_in_place(self, H_HARDTANH_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(min_val);trace.append_arg(max_val);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_hardswish_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::hardswish_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_HARDSWISH_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_hardswish(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::hardswish(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_HARDSWISH, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_hardswish_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::hardswish_(dispatchKeySet, self);
  }
  bool flush = register_in_place(self, H_HARDSWISH_, dispatchKeySet);
  trace.append_arg(self);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_hardswish_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::hardswish_backward(dispatchKeySet, grad_output, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_HARDSWISH_BACKWARD, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_leaky_relu_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & negative_slope, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::leaky_relu_outf(dispatchKeySet, self, negative_slope, out);
  }
  bool flush = register_in_place(out, H_LEAKY_RELU_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(negative_slope);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_leaky_relu(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & negative_slope) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::leaky_relu(dispatchKeySet, self, negative_slope);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LEAKY_RELU, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(negative_slope);
  return tt;
}

at::Tensor & wrap_leaky_relu_backward_grad_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & negative_slope, bool self_is_result, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::leaky_relu_backward_outf(dispatchKeySet, grad_output, self, negative_slope, self_is_result, grad_input);
  }
  bool flush = register_in_place(grad_input, H_LEAKY_RELU_BACKWARD_GRAD_INPUT, dispatchKeySet);
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(negative_slope);trace.append_arg(self_is_result);trace.append_arg(grad_input);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return grad_input;
}

at::Tensor & wrap_leaky_relu_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & negative_slope) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::leaky_relu_(dispatchKeySet, self, negative_slope);
  }
  bool flush = register_in_place(self, H_LEAKY_RELU_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(negative_slope);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_log_sigmoid_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::log_sigmoid_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_LOG_SIGMOID_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_log_sigmoid(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::log_sigmoid(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LOG_SIGMOID, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_log_sigmoid_backward_grad_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & buffer, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::log_sigmoid_backward_outf(dispatchKeySet, grad_output, self, buffer, grad_input);
  }
  bool flush = register_in_place(grad_input, H_LOG_SIGMOID_BACKWARD_GRAD_INPUT, dispatchKeySet);
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(buffer);trace.append_arg(grad_input);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return grad_input;
}

at::Tensor wrap_log_sigmoid_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & buffer) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::log_sigmoid_backward(dispatchKeySet, grad_output, self, buffer);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LOG_SIGMOID_BACKWARD, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(buffer);
  return tt;
}

at::Tensor & wrap_rrelu_with_noise_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & noise, const at::Scalar & lower, const at::Scalar & upper, bool training, c10::optional<at::Generator> generator, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::rrelu_with_noise_outf(dispatchKeySet, self, noise, lower, upper, training, std::move(generator), out);
  }
  bool flush = register_in_place(out, H_RRELU_WITH_NOISE_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(noise);trace.append_arg(lower);trace.append_arg(upper);trace.append_arg(training);trace.append_arg(std::move(generator));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_rrelu_with_noise(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & noise, const at::Scalar & lower, const at::Scalar & upper, bool training, c10::optional<at::Generator> generator) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::rrelu_with_noise(dispatchKeySet, self, noise, lower, upper, training, std::move(generator));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_RRELU_WITH_NOISE, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(noise);trace.append_arg(lower);trace.append_arg(upper);trace.append_arg(training);trace.append_arg(std::move(generator));
  return tt;
}

at::Tensor wrap_rrelu_with_noise_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & noise, const at::Scalar & lower, const at::Scalar & upper, bool training, bool self_is_result) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::rrelu_with_noise_backward(dispatchKeySet, grad_output, self, noise, lower, upper, training, self_is_result);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_RRELU_WITH_NOISE_BACKWARD, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(noise);trace.append_arg(lower);trace.append_arg(upper);trace.append_arg(training);trace.append_arg(self_is_result);
  return tt;
}

at::Tensor & wrap_rrelu_with_noise_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & noise, const at::Scalar & lower, const at::Scalar & upper, bool training, c10::optional<at::Generator> generator) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::rrelu_with_noise_(dispatchKeySet, self, noise, lower, upper, training, std::move(generator));
  }
  bool flush = register_in_place(self, H_RRELU_WITH_NOISE_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(noise);trace.append_arg(lower);trace.append_arg(upper);trace.append_arg(training);trace.append_arg(std::move(generator));
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor & wrap_softplus_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & beta, const at::Scalar & threshold, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::softplus_outf(dispatchKeySet, self, beta, threshold, out);
  }
  bool flush = register_in_place(out, H_SOFTPLUS_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(beta);trace.append_arg(threshold);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_softplus_backward_grad_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & beta, const at::Scalar & threshold, const at::Tensor & output, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::softplus_backward_outf(dispatchKeySet, grad_output, self, beta, threshold, output, grad_input);
  }
  bool flush = register_in_place(grad_input, H_SOFTPLUS_BACKWARD_GRAD_INPUT, dispatchKeySet);
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(beta);trace.append_arg(threshold);trace.append_arg(output);trace.append_arg(grad_input);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return grad_input;
}

at::Tensor & wrap_softshrink_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & lambd, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::softshrink_outf(dispatchKeySet, self, lambd, out);
  }
  bool flush = register_in_place(out, H_SOFTSHRINK_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(lambd);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_softshrink_backward_grad_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & lambd, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::softshrink_backward_outf(dispatchKeySet, grad_output, self, lambd, grad_input);
  }
  bool flush = register_in_place(grad_input, H_SOFTSHRINK_BACKWARD_GRAD_INPUT, dispatchKeySet);
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(lambd);trace.append_arg(grad_input);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return grad_input;
}

at::Tensor & wrap_adaptive_avg_pool2d_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::adaptive_avg_pool2d_outf(dispatchKeySet, self, std::move(output_size), out);
  }
  bool flush = register_in_place(out, H_ADAPTIVE_AVG_POOL2D_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(output_size));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_adaptive_avg_pool2d(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::adaptive_avg_pool2d(dispatchKeySet, self, std::move(output_size));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ADAPTIVE_AVG_POOL2D, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(output_size));
  return tt;
}

at::Tensor wrap_mkldnn_adaptive_avg_pool2d(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::mkldnn_adaptive_avg_pool2d(dispatchKeySet, self, std::move(output_size));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MKLDNN_ADAPTIVE_AVG_POOL2D, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(output_size));
  return tt;
}

at::Tensor wrap_mkldnn_adaptive_avg_pool2d_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::mkldnn_adaptive_avg_pool2d_backward(dispatchKeySet, grad_output, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MKLDNN_ADAPTIVE_AVG_POOL2D_BACKWARD, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(self);
  return tt;
}

at::Tensor wrap__adaptive_avg_pool2d(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_adaptive_avg_pool2d(dispatchKeySet, self, std::move(output_size));
  }
  auto tt = register_new_tensor(dispatchKeySet, H__ADAPTIVE_AVG_POOL2D, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(output_size));
  return tt;
}

at::Tensor wrap__adaptive_avg_pool2d_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_adaptive_avg_pool2d_backward(dispatchKeySet, grad_output, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__ADAPTIVE_AVG_POOL2D_BACKWARD, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_adaptive_avg_pool3d_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::adaptive_avg_pool3d_outf(dispatchKeySet, self, std::move(output_size), out);
  }
  bool flush = register_in_place(out, H_ADAPTIVE_AVG_POOL3D_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(output_size));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_adaptive_avg_pool3d(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::adaptive_avg_pool3d(dispatchKeySet, self, std::move(output_size));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ADAPTIVE_AVG_POOL3D, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(output_size));
  return tt;
}

at::Tensor wrap__adaptive_avg_pool3d(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_adaptive_avg_pool3d(dispatchKeySet, self, std::move(output_size));
  }
  auto tt = register_new_tensor(dispatchKeySet, H__ADAPTIVE_AVG_POOL3D, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(output_size));
  return tt;
}

at::Tensor & wrap_adaptive_avg_pool3d_backward_grad_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::adaptive_avg_pool3d_backward_outf(dispatchKeySet, grad_output, self, grad_input);
  }
  bool flush = register_in_place(grad_input, H_ADAPTIVE_AVG_POOL3D_BACKWARD_GRAD_INPUT, dispatchKeySet);
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(grad_input);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return grad_input;
}

at::Tensor wrap__adaptive_avg_pool3d_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_adaptive_avg_pool3d_backward(dispatchKeySet, grad_output, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__ADAPTIVE_AVG_POOL3D_BACKWARD, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_adaptive_max_pool2d_backward_grad_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::adaptive_max_pool2d_backward_outf(dispatchKeySet, grad_output, self, indices, grad_input);
  }
  bool flush = register_in_place(grad_input, H_ADAPTIVE_MAX_POOL2D_BACKWARD_GRAD_INPUT, dispatchKeySet);
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(indices);trace.append_arg(grad_input);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return grad_input;
}

at::Tensor & wrap_adaptive_max_pool3d_backward_grad_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::adaptive_max_pool3d_backward_outf(dispatchKeySet, grad_output, self, indices, grad_input);
  }
  bool flush = register_in_place(grad_input, H_ADAPTIVE_MAX_POOL3D_BACKWARD_GRAD_INPUT, dispatchKeySet);
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(indices);trace.append_arg(grad_input);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return grad_input;
}

at::Tensor & wrap_avg_pool2d_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::avg_pool2d_outf(dispatchKeySet, self, std::move(kernel_size), std::move(stride), std::move(padding), ceil_mode, count_include_pad, divisor_override, out);
  }
  bool flush = register_in_place(out, H_AVG_POOL2D_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(kernel_size));trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(ceil_mode);trace.append_arg(count_include_pad);trace.append_arg(divisor_override);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_avg_pool2d(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::avg_pool2d(dispatchKeySet, self, std::move(kernel_size), std::move(stride), std::move(padding), ceil_mode, count_include_pad, divisor_override);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_AVG_POOL2D, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(kernel_size));trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(ceil_mode);trace.append_arg(count_include_pad);trace.append_arg(divisor_override);
  return tt;
}

at::Tensor & wrap_avg_pool2d_backward_grad_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::avg_pool2d_backward_outf(dispatchKeySet, grad_output, self, std::move(kernel_size), std::move(stride), std::move(padding), ceil_mode, count_include_pad, divisor_override, grad_input);
  }
  bool flush = register_in_place(grad_input, H_AVG_POOL2D_BACKWARD_GRAD_INPUT, dispatchKeySet);
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(std::move(kernel_size));trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(ceil_mode);trace.append_arg(count_include_pad);trace.append_arg(divisor_override);trace.append_arg(grad_input);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return grad_input;
}

at::Tensor wrap_avg_pool2d_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::avg_pool2d_backward(dispatchKeySet, grad_output, self, std::move(kernel_size), std::move(stride), std::move(padding), ceil_mode, count_include_pad, divisor_override);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_AVG_POOL2D_BACKWARD, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(std::move(kernel_size));trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(ceil_mode);trace.append_arg(count_include_pad);trace.append_arg(divisor_override);
  return tt;
}

at::Tensor & wrap_avg_pool3d_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::avg_pool3d_outf(dispatchKeySet, self, std::move(kernel_size), std::move(stride), std::move(padding), ceil_mode, count_include_pad, divisor_override, out);
  }
  bool flush = register_in_place(out, H_AVG_POOL3D_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(kernel_size));trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(ceil_mode);trace.append_arg(count_include_pad);trace.append_arg(divisor_override);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_avg_pool3d(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::avg_pool3d(dispatchKeySet, self, std::move(kernel_size), std::move(stride), std::move(padding), ceil_mode, count_include_pad, divisor_override);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_AVG_POOL3D, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(kernel_size));trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(ceil_mode);trace.append_arg(count_include_pad);trace.append_arg(divisor_override);
  return tt;
}

at::Tensor & wrap_avg_pool3d_backward_grad_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::avg_pool3d_backward_outf(dispatchKeySet, grad_output, self, std::move(kernel_size), std::move(stride), std::move(padding), ceil_mode, count_include_pad, divisor_override, grad_input);
  }
  bool flush = register_in_place(grad_input, H_AVG_POOL3D_BACKWARD_GRAD_INPUT, dispatchKeySet);
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(std::move(kernel_size));trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(ceil_mode);trace.append_arg(count_include_pad);trace.append_arg(divisor_override);trace.append_arg(grad_input);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return grad_input;
}

at::Tensor wrap_avg_pool3d_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::avg_pool3d_backward(dispatchKeySet, grad_output, self, std::move(kernel_size), std::move(stride), std::move(padding), ceil_mode, count_include_pad, divisor_override);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_AVG_POOL3D_BACKWARD, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(std::move(kernel_size));trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(ceil_mode);trace.append_arg(count_include_pad);trace.append_arg(divisor_override);
  return tt;
}

at::Tensor & wrap_fractional_max_pool2d_backward_grad_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef output_size, const at::Tensor & indices, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fractional_max_pool2d_backward_outf(dispatchKeySet, grad_output, self, std::move(kernel_size), std::move(output_size), indices, grad_input);
  }
  bool flush = register_in_place(grad_input, H_FRACTIONAL_MAX_POOL2D_BACKWARD_GRAD_INPUT, dispatchKeySet);
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(std::move(kernel_size));trace.append_arg(std::move(output_size));trace.append_arg(indices);trace.append_arg(grad_input);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return grad_input;
}

at::Tensor wrap_fractional_max_pool2d_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef output_size, const at::Tensor & indices) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fractional_max_pool2d_backward(dispatchKeySet, grad_output, self, std::move(kernel_size), std::move(output_size), indices);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_FRACTIONAL_MAX_POOL2D_BACKWARD, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(std::move(kernel_size));trace.append_arg(std::move(output_size));trace.append_arg(indices);
  return tt;
}

at::Tensor & wrap_fractional_max_pool3d_backward_grad_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef output_size, const at::Tensor & indices, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fractional_max_pool3d_backward_outf(dispatchKeySet, grad_output, self, std::move(kernel_size), std::move(output_size), indices, grad_input);
  }
  bool flush = register_in_place(grad_input, H_FRACTIONAL_MAX_POOL3D_BACKWARD_GRAD_INPUT, dispatchKeySet);
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(std::move(kernel_size));trace.append_arg(std::move(output_size));trace.append_arg(indices);trace.append_arg(grad_input);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return grad_input;
}

at::Tensor wrap_fractional_max_pool3d_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef output_size, const at::Tensor & indices) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fractional_max_pool3d_backward(dispatchKeySet, grad_output, self, std::move(kernel_size), std::move(output_size), indices);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_FRACTIONAL_MAX_POOL3D_BACKWARD, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(std::move(kernel_size));trace.append_arg(std::move(output_size));trace.append_arg(indices);
  return tt;
}

at::Tensor & wrap_max_pool2d_with_indices_backward_grad_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, const at::Tensor & indices, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::max_pool2d_with_indices_backward_outf(dispatchKeySet, grad_output, self, std::move(kernel_size), std::move(stride), std::move(padding), std::move(dilation), ceil_mode, indices, grad_input);
  }
  bool flush = register_in_place(grad_input, H_MAX_POOL2D_WITH_INDICES_BACKWARD_GRAD_INPUT, dispatchKeySet);
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(std::move(kernel_size));trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(std::move(dilation));trace.append_arg(ceil_mode);trace.append_arg(indices);trace.append_arg(grad_input);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return grad_input;
}

at::Tensor & wrap_max_pool3d_with_indices_backward_grad_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, const at::Tensor & indices, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::max_pool3d_with_indices_backward_outf(dispatchKeySet, grad_output, self, std::move(kernel_size), std::move(stride), std::move(padding), std::move(dilation), ceil_mode, indices, grad_input);
  }
  bool flush = register_in_place(grad_input, H_MAX_POOL3D_WITH_INDICES_BACKWARD_GRAD_INPUT, dispatchKeySet);
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(std::move(kernel_size));trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(std::move(dilation));trace.append_arg(ceil_mode);trace.append_arg(indices);trace.append_arg(grad_input);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return grad_input;
}

at::Tensor wrap_max_pool3d_with_indices_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, const at::Tensor & indices) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::max_pool3d_with_indices_backward(dispatchKeySet, grad_output, self, std::move(kernel_size), std::move(stride), std::move(padding), std::move(dilation), ceil_mode, indices);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MAX_POOL3D_WITH_INDICES_BACKWARD, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(std::move(kernel_size));trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(std::move(dilation));trace.append_arg(ceil_mode);trace.append_arg(indices);
  return tt;
}

at::Tensor & wrap_max_unpool2d_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & indices, at::IntArrayRef output_size, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::max_unpool2d_outf(dispatchKeySet, self, indices, std::move(output_size), out);
  }
  bool flush = register_in_place(out, H_MAX_UNPOOL2D_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(indices);trace.append_arg(std::move(output_size));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_max_unpool2d(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & indices, at::IntArrayRef output_size) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::max_unpool2d(dispatchKeySet, self, indices, std::move(output_size));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MAX_UNPOOL2D, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(indices);trace.append_arg(std::move(output_size));
  return tt;
}

at::Tensor & wrap_max_unpool2d_backward_grad_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices, at::IntArrayRef output_size, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::max_unpool2d_backward_outf(dispatchKeySet, grad_output, self, indices, std::move(output_size), grad_input);
  }
  bool flush = register_in_place(grad_input, H_MAX_UNPOOL2D_BACKWARD_GRAD_INPUT, dispatchKeySet);
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(indices);trace.append_arg(std::move(output_size));trace.append_arg(grad_input);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return grad_input;
}

at::Tensor wrap_max_unpool2d_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices, at::IntArrayRef output_size) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::max_unpool2d_backward(dispatchKeySet, grad_output, self, indices, std::move(output_size));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MAX_UNPOOL2D_BACKWARD, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(indices);trace.append_arg(std::move(output_size));
  return tt;
}

at::Tensor & wrap_max_unpool3d_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & indices, at::IntArrayRef output_size, at::IntArrayRef stride, at::IntArrayRef padding, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::max_unpool3d_outf(dispatchKeySet, self, indices, std::move(output_size), std::move(stride), std::move(padding), out);
  }
  bool flush = register_in_place(out, H_MAX_UNPOOL3D_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(indices);trace.append_arg(std::move(output_size));trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_max_unpool3d(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & indices, at::IntArrayRef output_size, at::IntArrayRef stride, at::IntArrayRef padding) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::max_unpool3d(dispatchKeySet, self, indices, std::move(output_size), std::move(stride), std::move(padding));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MAX_UNPOOL3D, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(indices);trace.append_arg(std::move(output_size));trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));
  return tt;
}

at::Tensor & wrap_max_unpool3d_backward_grad_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices, at::IntArrayRef output_size, at::IntArrayRef stride, at::IntArrayRef padding, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::max_unpool3d_backward_outf(dispatchKeySet, grad_output, self, indices, std::move(output_size), std::move(stride), std::move(padding), grad_input);
  }
  bool flush = register_in_place(grad_input, H_MAX_UNPOOL3D_BACKWARD_GRAD_INPUT, dispatchKeySet);
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(indices);trace.append_arg(std::move(output_size));trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(grad_input);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return grad_input;
}

at::Tensor wrap_max_unpool3d_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices, at::IntArrayRef output_size, at::IntArrayRef stride, at::IntArrayRef padding) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::max_unpool3d_backward(dispatchKeySet, grad_output, self, indices, std::move(output_size), std::move(stride), std::move(padding));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_MAX_UNPOOL3D_BACKWARD, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(indices);trace.append_arg(std::move(output_size));trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));
  return tt;
}

at::Tensor & wrap_reflection_pad1d_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::reflection_pad1d_outf(dispatchKeySet, self, std::move(padding), out);
  }
  bool flush = register_in_place(out, H_REFLECTION_PAD1D_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(padding));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_reflection_pad1d(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef padding) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::reflection_pad1d(dispatchKeySet, self, std::move(padding));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_REFLECTION_PAD1D, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(padding));
  return tt;
}

at::Tensor & wrap_reflection_pad1d_backward_grad_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::reflection_pad1d_backward_outf(dispatchKeySet, grad_output, self, std::move(padding), grad_input);
  }
  bool flush = register_in_place(grad_input, H_REFLECTION_PAD1D_BACKWARD_GRAD_INPUT, dispatchKeySet);
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(std::move(padding));trace.append_arg(grad_input);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return grad_input;
}

at::Tensor & wrap_reflection_pad2d_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::reflection_pad2d_outf(dispatchKeySet, self, std::move(padding), out);
  }
  bool flush = register_in_place(out, H_REFLECTION_PAD2D_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(padding));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_reflection_pad2d(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef padding) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::reflection_pad2d(dispatchKeySet, self, std::move(padding));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_REFLECTION_PAD2D, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(padding));
  return tt;
}

at::Tensor & wrap_reflection_pad2d_backward_grad_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::reflection_pad2d_backward_outf(dispatchKeySet, grad_output, self, std::move(padding), grad_input);
  }
  bool flush = register_in_place(grad_input, H_REFLECTION_PAD2D_BACKWARD_GRAD_INPUT, dispatchKeySet);
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(std::move(padding));trace.append_arg(grad_input);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return grad_input;
}

at::Tensor wrap_reflection_pad2d_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::reflection_pad2d_backward(dispatchKeySet, grad_output, self, std::move(padding));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_REFLECTION_PAD2D_BACKWARD, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(std::move(padding));
  return tt;
}

at::Tensor & wrap_replication_pad1d_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::replication_pad1d_outf(dispatchKeySet, self, std::move(padding), out);
  }
  bool flush = register_in_place(out, H_REPLICATION_PAD1D_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(padding));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_replication_pad1d_backward_grad_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::replication_pad1d_backward_outf(dispatchKeySet, grad_output, self, std::move(padding), grad_input);
  }
  bool flush = register_in_place(grad_input, H_REPLICATION_PAD1D_BACKWARD_GRAD_INPUT, dispatchKeySet);
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(std::move(padding));trace.append_arg(grad_input);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return grad_input;
}

at::Tensor & wrap_replication_pad2d_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::replication_pad2d_outf(dispatchKeySet, self, std::move(padding), out);
  }
  bool flush = register_in_place(out, H_REPLICATION_PAD2D_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(padding));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_replication_pad2d_backward_grad_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::replication_pad2d_backward_outf(dispatchKeySet, grad_output, self, std::move(padding), grad_input);
  }
  bool flush = register_in_place(grad_input, H_REPLICATION_PAD2D_BACKWARD_GRAD_INPUT, dispatchKeySet);
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(std::move(padding));trace.append_arg(grad_input);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return grad_input;
}

at::Tensor wrap_replication_pad2d_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::replication_pad2d_backward(dispatchKeySet, grad_output, self, std::move(padding));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_REPLICATION_PAD2D_BACKWARD, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(std::move(padding));
  return tt;
}

at::Tensor & wrap_replication_pad3d_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::replication_pad3d_outf(dispatchKeySet, self, std::move(padding), out);
  }
  bool flush = register_in_place(out, H_REPLICATION_PAD3D_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(padding));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_replication_pad3d_backward_grad_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::replication_pad3d_backward_outf(dispatchKeySet, grad_output, self, std::move(padding), grad_input);
  }
  bool flush = register_in_place(grad_input, H_REPLICATION_PAD3D_BACKWARD_GRAD_INPUT, dispatchKeySet);
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(std::move(padding));trace.append_arg(grad_input);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return grad_input;
}

at::Tensor wrap_replication_pad3d_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::replication_pad3d_backward(dispatchKeySet, grad_output, self, std::move(padding));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_REPLICATION_PAD3D_BACKWARD, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(std::move(padding));
  return tt;
}

at::Tensor wrap_upsample_linear1d_vec(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, c10::optional<at::IntArrayRef> output_size, bool align_corners, c10::optional<at::ArrayRef<double>> scale_factors) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::upsample_linear1d(dispatchKeySet, input, std::move(output_size), align_corners, scale_factors);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_UPSAMPLE_LINEAR1D_VEC, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(std::move(output_size));trace.append_arg(align_corners);trace.append_arg(scale_factors);
  return tt;
}

at::Tensor wrap_upsample_linear1d_backward_vec(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, c10::optional<at::IntArrayRef> output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<at::ArrayRef<double>> scale_factors) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::upsample_linear1d_backward(dispatchKeySet, grad_output, std::move(output_size), std::move(input_size), align_corners, scale_factors);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_UPSAMPLE_LINEAR1D_BACKWARD_VEC, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(std::move(output_size));trace.append_arg(std::move(input_size));trace.append_arg(align_corners);trace.append_arg(scale_factors);
  return tt;
}

at::Tensor wrap_upsample_bilinear2d_vec(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, c10::optional<at::IntArrayRef> output_size, bool align_corners, c10::optional<at::ArrayRef<double>> scale_factors) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::upsample_bilinear2d(dispatchKeySet, input, std::move(output_size), align_corners, scale_factors);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_UPSAMPLE_BILINEAR2D_VEC, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(std::move(output_size));trace.append_arg(align_corners);trace.append_arg(scale_factors);
  return tt;
}

at::Tensor wrap_upsample_bilinear2d_backward_vec(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, c10::optional<at::IntArrayRef> output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<at::ArrayRef<double>> scale_factors) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::upsample_bilinear2d_backward(dispatchKeySet, grad_output, std::move(output_size), std::move(input_size), align_corners, scale_factors);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_UPSAMPLE_BILINEAR2D_BACKWARD_VEC, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(std::move(output_size));trace.append_arg(std::move(input_size));trace.append_arg(align_corners);trace.append_arg(scale_factors);
  return tt;
}

at::Tensor wrap_upsample_trilinear3d_vec(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, c10::optional<at::IntArrayRef> output_size, bool align_corners, c10::optional<at::ArrayRef<double>> scale_factors) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::upsample_trilinear3d(dispatchKeySet, input, std::move(output_size), align_corners, scale_factors);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_UPSAMPLE_TRILINEAR3D_VEC, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(std::move(output_size));trace.append_arg(align_corners);trace.append_arg(scale_factors);
  return tt;
}

at::Tensor wrap_upsample_trilinear3d_backward_vec(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, c10::optional<at::IntArrayRef> output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<at::ArrayRef<double>> scale_factors) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::upsample_trilinear3d_backward(dispatchKeySet, grad_output, std::move(output_size), std::move(input_size), align_corners, scale_factors);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_UPSAMPLE_TRILINEAR3D_BACKWARD_VEC, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(std::move(output_size));trace.append_arg(std::move(input_size));trace.append_arg(align_corners);trace.append_arg(scale_factors);
  return tt;
}

at::Tensor wrap_upsample_bicubic2d_vec(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, c10::optional<at::IntArrayRef> output_size, bool align_corners, c10::optional<at::ArrayRef<double>> scale_factors) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::upsample_bicubic2d(dispatchKeySet, input, std::move(output_size), align_corners, scale_factors);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_UPSAMPLE_BICUBIC2D_VEC, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(std::move(output_size));trace.append_arg(align_corners);trace.append_arg(scale_factors);
  return tt;
}

at::Tensor wrap_upsample_bicubic2d_backward_vec(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, c10::optional<at::IntArrayRef> output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<at::ArrayRef<double>> scale_factors) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::upsample_bicubic2d_backward(dispatchKeySet, grad_output, std::move(output_size), std::move(input_size), align_corners, scale_factors);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_UPSAMPLE_BICUBIC2D_BACKWARD_VEC, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(std::move(output_size));trace.append_arg(std::move(input_size));trace.append_arg(align_corners);trace.append_arg(scale_factors);
  return tt;
}

at::Tensor wrap_upsample_nearest1d_vec(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, c10::optional<at::IntArrayRef> output_size, c10::optional<at::ArrayRef<double>> scale_factors) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::upsample_nearest1d(dispatchKeySet, input, std::move(output_size), scale_factors);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_UPSAMPLE_NEAREST1D_VEC, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(std::move(output_size));trace.append_arg(scale_factors);
  return tt;
}

at::Tensor wrap_upsample_nearest1d_backward_vec(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, c10::optional<at::IntArrayRef> output_size, at::IntArrayRef input_size, c10::optional<at::ArrayRef<double>> scale_factors) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::upsample_nearest1d_backward(dispatchKeySet, grad_output, std::move(output_size), std::move(input_size), scale_factors);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_UPSAMPLE_NEAREST1D_BACKWARD_VEC, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(std::move(output_size));trace.append_arg(std::move(input_size));trace.append_arg(scale_factors);
  return tt;
}

at::Tensor wrap_upsample_nearest2d_vec(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, c10::optional<at::IntArrayRef> output_size, c10::optional<at::ArrayRef<double>> scale_factors) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::upsample_nearest2d(dispatchKeySet, input, std::move(output_size), scale_factors);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_UPSAMPLE_NEAREST2D_VEC, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(std::move(output_size));trace.append_arg(scale_factors);
  return tt;
}

at::Tensor wrap_upsample_nearest2d_backward_vec(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, c10::optional<at::IntArrayRef> output_size, at::IntArrayRef input_size, c10::optional<at::ArrayRef<double>> scale_factors) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::upsample_nearest2d_backward(dispatchKeySet, grad_output, std::move(output_size), std::move(input_size), scale_factors);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_UPSAMPLE_NEAREST2D_BACKWARD_VEC, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(std::move(output_size));trace.append_arg(std::move(input_size));trace.append_arg(scale_factors);
  return tt;
}

at::Tensor wrap_upsample_nearest3d_vec(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, c10::optional<at::IntArrayRef> output_size, c10::optional<at::ArrayRef<double>> scale_factors) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::upsample_nearest3d(dispatchKeySet, input, std::move(output_size), scale_factors);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_UPSAMPLE_NEAREST3D_VEC, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(std::move(output_size));trace.append_arg(scale_factors);
  return tt;
}

at::Tensor wrap_upsample_nearest3d_backward_vec(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, c10::optional<at::IntArrayRef> output_size, at::IntArrayRef input_size, c10::optional<at::ArrayRef<double>> scale_factors) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::upsample_nearest3d_backward(dispatchKeySet, grad_output, std::move(output_size), std::move(input_size), scale_factors);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_UPSAMPLE_NEAREST3D_BACKWARD_VEC, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(std::move(output_size));trace.append_arg(std::move(input_size));trace.append_arg(scale_factors);
  return tt;
}

at::Tensor & wrap_upsample_linear1d_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::upsample_linear1d_outf(dispatchKeySet, self, std::move(output_size), align_corners, scales, out);
  }
  bool flush = register_in_place(out, H_UPSAMPLE_LINEAR1D_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(output_size));trace.append_arg(align_corners);trace.append_arg(scales);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_upsample_linear1d_backward_grad_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::upsample_linear1d_backward_outf(dispatchKeySet, grad_output, std::move(output_size), std::move(input_size), align_corners, scales, grad_input);
  }
  bool flush = register_in_place(grad_input, H_UPSAMPLE_LINEAR1D_BACKWARD_GRAD_INPUT, dispatchKeySet);
  trace.append_arg(grad_output);trace.append_arg(std::move(output_size));trace.append_arg(std::move(input_size));trace.append_arg(align_corners);trace.append_arg(scales);trace.append_arg(grad_input);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return grad_input;
}

at::Tensor & wrap_upsample_bilinear2d_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::upsample_bilinear2d_outf(dispatchKeySet, self, std::move(output_size), align_corners, scales_h, scales_w, out);
  }
  bool flush = register_in_place(out, H_UPSAMPLE_BILINEAR2D_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(output_size));trace.append_arg(align_corners);trace.append_arg(scales_h);trace.append_arg(scales_w);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_upsample_bilinear2d(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::upsample_bilinear2d(dispatchKeySet, self, std::move(output_size), align_corners, scales_h, scales_w);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_UPSAMPLE_BILINEAR2D, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(output_size));trace.append_arg(align_corners);trace.append_arg(scales_h);trace.append_arg(scales_w);
  return tt;
}

at::Tensor & wrap_upsample_bilinear2d_backward_grad_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::upsample_bilinear2d_backward_outf(dispatchKeySet, grad_output, std::move(output_size), std::move(input_size), align_corners, scales_h, scales_w, grad_input);
  }
  bool flush = register_in_place(grad_input, H_UPSAMPLE_BILINEAR2D_BACKWARD_GRAD_INPUT, dispatchKeySet);
  trace.append_arg(grad_output);trace.append_arg(std::move(output_size));trace.append_arg(std::move(input_size));trace.append_arg(align_corners);trace.append_arg(scales_h);trace.append_arg(scales_w);trace.append_arg(grad_input);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return grad_input;
}

at::Tensor & wrap_upsample_bicubic2d_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::upsample_bicubic2d_outf(dispatchKeySet, self, std::move(output_size), align_corners, scales_h, scales_w, out);
  }
  bool flush = register_in_place(out, H_UPSAMPLE_BICUBIC2D_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(output_size));trace.append_arg(align_corners);trace.append_arg(scales_h);trace.append_arg(scales_w);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_upsample_bicubic2d_backward_grad_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::upsample_bicubic2d_backward_outf(dispatchKeySet, grad_output, std::move(output_size), std::move(input_size), align_corners, scales_h, scales_w, grad_input);
  }
  bool flush = register_in_place(grad_input, H_UPSAMPLE_BICUBIC2D_BACKWARD_GRAD_INPUT, dispatchKeySet);
  trace.append_arg(grad_output);trace.append_arg(std::move(output_size));trace.append_arg(std::move(input_size));trace.append_arg(align_corners);trace.append_arg(scales_h);trace.append_arg(scales_w);trace.append_arg(grad_input);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return grad_input;
}

at::Tensor & wrap_upsample_trilinear3d_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::upsample_trilinear3d_outf(dispatchKeySet, self, std::move(output_size), align_corners, scales_d, scales_h, scales_w, out);
  }
  bool flush = register_in_place(out, H_UPSAMPLE_TRILINEAR3D_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(output_size));trace.append_arg(align_corners);trace.append_arg(scales_d);trace.append_arg(scales_h);trace.append_arg(scales_w);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_upsample_trilinear3d_backward_grad_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::upsample_trilinear3d_backward_outf(dispatchKeySet, grad_output, std::move(output_size), std::move(input_size), align_corners, scales_d, scales_h, scales_w, grad_input);
  }
  bool flush = register_in_place(grad_input, H_UPSAMPLE_TRILINEAR3D_BACKWARD_GRAD_INPUT, dispatchKeySet);
  trace.append_arg(grad_output);trace.append_arg(std::move(output_size));trace.append_arg(std::move(input_size));trace.append_arg(align_corners);trace.append_arg(scales_d);trace.append_arg(scales_h);trace.append_arg(scales_w);trace.append_arg(grad_input);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return grad_input;
}

at::Tensor & wrap_upsample_nearest1d_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size, c10::optional<double> scales, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::upsample_nearest1d_outf(dispatchKeySet, self, std::move(output_size), scales, out);
  }
  bool flush = register_in_place(out, H_UPSAMPLE_NEAREST1D_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(output_size));trace.append_arg(scales);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_upsample_nearest1d_backward_grad_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, c10::optional<double> scales, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::upsample_nearest1d_backward_outf(dispatchKeySet, grad_output, std::move(output_size), std::move(input_size), scales, grad_input);
  }
  bool flush = register_in_place(grad_input, H_UPSAMPLE_NEAREST1D_BACKWARD_GRAD_INPUT, dispatchKeySet);
  trace.append_arg(grad_output);trace.append_arg(std::move(output_size));trace.append_arg(std::move(input_size));trace.append_arg(scales);trace.append_arg(grad_input);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return grad_input;
}

at::Tensor & wrap_upsample_nearest2d_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::upsample_nearest2d_outf(dispatchKeySet, self, std::move(output_size), scales_h, scales_w, out);
  }
  bool flush = register_in_place(out, H_UPSAMPLE_NEAREST2D_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(output_size));trace.append_arg(scales_h);trace.append_arg(scales_w);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_upsample_nearest2d(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::upsample_nearest2d(dispatchKeySet, self, std::move(output_size), scales_h, scales_w);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_UPSAMPLE_NEAREST2D, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(output_size));trace.append_arg(scales_h);trace.append_arg(scales_w);
  return tt;
}

at::Tensor & wrap_upsample_nearest2d_backward_grad_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::upsample_nearest2d_backward_outf(dispatchKeySet, grad_output, std::move(output_size), std::move(input_size), scales_h, scales_w, grad_input);
  }
  bool flush = register_in_place(grad_input, H_UPSAMPLE_NEAREST2D_BACKWARD_GRAD_INPUT, dispatchKeySet);
  trace.append_arg(grad_output);trace.append_arg(std::move(output_size));trace.append_arg(std::move(input_size));trace.append_arg(scales_h);trace.append_arg(scales_w);trace.append_arg(grad_input);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return grad_input;
}

at::Tensor & wrap_upsample_nearest3d_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::upsample_nearest3d_outf(dispatchKeySet, self, std::move(output_size), scales_d, scales_h, scales_w, out);
  }
  bool flush = register_in_place(out, H_UPSAMPLE_NEAREST3D_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(output_size));trace.append_arg(scales_d);trace.append_arg(scales_h);trace.append_arg(scales_w);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_upsample_nearest3d(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::upsample_nearest3d(dispatchKeySet, self, std::move(output_size), scales_d, scales_h, scales_w);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_UPSAMPLE_NEAREST3D, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(output_size));trace.append_arg(scales_d);trace.append_arg(scales_h);trace.append_arg(scales_w);
  return tt;
}

at::Tensor & wrap_upsample_nearest3d_backward_grad_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::upsample_nearest3d_backward_outf(dispatchKeySet, grad_output, std::move(output_size), std::move(input_size), scales_d, scales_h, scales_w, grad_input);
  }
  bool flush = register_in_place(grad_input, H_UPSAMPLE_NEAREST3D_BACKWARD_GRAD_INPUT, dispatchKeySet);
  trace.append_arg(grad_output);trace.append_arg(std::move(output_size));trace.append_arg(std::move(input_size));trace.append_arg(scales_d);trace.append_arg(scales_h);trace.append_arg(scales_w);trace.append_arg(grad_input);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return grad_input;
}

at::Tensor & wrap_sigmoid_backward_grad_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & output, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::sigmoid_backward_outf(dispatchKeySet, grad_output, output, grad_input);
  }
  bool flush = register_in_place(grad_input, H_SIGMOID_BACKWARD_GRAD_INPUT, dispatchKeySet);
  trace.append_arg(grad_output);trace.append_arg(output);trace.append_arg(grad_input);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return grad_input;
}

at::Tensor wrap_sigmoid_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & output) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::sigmoid_backward(dispatchKeySet, grad_output, output);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SIGMOID_BACKWARD, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(output);
  return tt;
}

at::Tensor & wrap_logit_backward_grad_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, c10::optional<double> eps, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::logit_backward_outf(dispatchKeySet, grad_output, self, eps, grad_input);
  }
  bool flush = register_in_place(grad_input, H_LOGIT_BACKWARD_GRAD_INPUT, dispatchKeySet);
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(eps);trace.append_arg(grad_input);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return grad_input;
}

at::Tensor wrap_logit_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, c10::optional<double> eps) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::logit_backward(dispatchKeySet, grad_output, self, eps);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LOGIT_BACKWARD, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(self);trace.append_arg(eps);
  return tt;
}

at::Tensor & wrap_tanh_backward_grad_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & output, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::tanh_backward_outf(dispatchKeySet, grad_output, output, grad_input);
  }
  bool flush = register_in_place(grad_input, H_TANH_BACKWARD_GRAD_INPUT, dispatchKeySet);
  trace.append_arg(grad_output);trace.append_arg(output);trace.append_arg(grad_input);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return grad_input;
}

at::Tensor wrap_tanh_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & output) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::tanh_backward(dispatchKeySet, grad_output, output);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_TANH_BACKWARD, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(output);
  return tt;
}

at::Tensor & wrap_slow_conv_transpose2d_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::slow_conv_transpose2d_outf(dispatchKeySet, self, weight, std::move(kernel_size), bias, std::move(stride), std::move(padding), std::move(output_padding), std::move(dilation), out);
  }
  bool flush = register_in_place(out, H_SLOW_CONV_TRANSPOSE2D_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(weight);trace.append_arg(std::move(kernel_size));trace.append_arg(bias);trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(std::move(output_padding));trace.append_arg(std::move(dilation));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_slow_conv_transpose2d(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::slow_conv_transpose2d(dispatchKeySet, self, weight, std::move(kernel_size), bias, std::move(stride), std::move(padding), std::move(output_padding), std::move(dilation));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SLOW_CONV_TRANSPOSE2D, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(weight);trace.append_arg(std::move(kernel_size));trace.append_arg(bias);trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(std::move(output_padding));trace.append_arg(std::move(dilation));
  return tt;
}

at::Tensor & wrap_slow_conv_transpose3d_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::slow_conv_transpose3d_outf(dispatchKeySet, self, weight, std::move(kernel_size), bias, std::move(stride), std::move(padding), std::move(output_padding), std::move(dilation), out);
  }
  bool flush = register_in_place(out, H_SLOW_CONV_TRANSPOSE3D_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(weight);trace.append_arg(std::move(kernel_size));trace.append_arg(bias);trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(std::move(output_padding));trace.append_arg(std::move(dilation));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_slow_conv_transpose3d(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::slow_conv_transpose3d(dispatchKeySet, self, weight, std::move(kernel_size), bias, std::move(stride), std::move(padding), std::move(output_padding), std::move(dilation));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SLOW_CONV_TRANSPOSE3D, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(weight);trace.append_arg(std::move(kernel_size));trace.append_arg(bias);trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(std::move(output_padding));trace.append_arg(std::move(dilation));
  return tt;
}

at::Tensor & wrap_thnn_conv2d_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::thnn_conv2d_outf(dispatchKeySet, self, weight, std::move(kernel_size), bias, std::move(stride), std::move(padding), out);
  }
  bool flush = register_in_place(out, H_THNN_CONV2D_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(weight);trace.append_arg(std::move(kernel_size));trace.append_arg(bias);trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_thnn_conv2d(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::thnn_conv2d(dispatchKeySet, self, weight, std::move(kernel_size), bias, std::move(stride), std::move(padding));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_THNN_CONV2D, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(weight);trace.append_arg(std::move(kernel_size));trace.append_arg(bias);trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));
  return tt;
}

at::Tensor & wrap_thnn_conv_depthwise2d_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::thnn_conv_depthwise2d_outf(dispatchKeySet, self, weight, std::move(kernel_size), bias, std::move(stride), std::move(padding), std::move(dilation), out);
  }
  bool flush = register_in_place(out, H_THNN_CONV_DEPTHWISE2D_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(weight);trace.append_arg(std::move(kernel_size));trace.append_arg(bias);trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(std::move(dilation));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_thnn_conv_depthwise2d(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::thnn_conv_depthwise2d(dispatchKeySet, self, weight, std::move(kernel_size), bias, std::move(stride), std::move(padding), std::move(dilation));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_THNN_CONV_DEPTHWISE2D, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(weight);trace.append_arg(std::move(kernel_size));trace.append_arg(bias);trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(std::move(dilation));
  return tt;
}

at::Tensor & wrap_thnn_conv_depthwise2d_forward_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::thnn_conv_depthwise2d_forward_outf(dispatchKeySet, self, weight, std::move(kernel_size), bias, std::move(stride), std::move(padding), std::move(dilation), out);
  }
  bool flush = register_in_place(out, H_THNN_CONV_DEPTHWISE2D_FORWARD_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(weight);trace.append_arg(std::move(kernel_size));trace.append_arg(bias);trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(std::move(dilation));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_thnn_conv_depthwise2d_forward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::thnn_conv_depthwise2d_forward(dispatchKeySet, self, weight, std::move(kernel_size), bias, std::move(stride), std::move(padding), std::move(dilation));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_THNN_CONV_DEPTHWISE2D_FORWARD, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(weight);trace.append_arg(std::move(kernel_size));trace.append_arg(bias);trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(std::move(dilation));
  return tt;
}

at::Tensor wrap_conv_depthwise3d(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::conv_depthwise3d(dispatchKeySet, self, weight, std::move(kernel_size), bias, std::move(stride), std::move(padding), std::move(dilation));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_CONV_DEPTHWISE3D, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(weight);trace.append_arg(std::move(kernel_size));trace.append_arg(bias);trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(std::move(dilation));
  return tt;
}

at::Tensor & wrap_slow_conv3d_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::slow_conv3d_outf(dispatchKeySet, self, weight, std::move(kernel_size), bias, std::move(stride), std::move(padding), out);
  }
  bool flush = register_in_place(out, H_SLOW_CONV3D_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(weight);trace.append_arg(std::move(kernel_size));trace.append_arg(bias);trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_slow_conv3d(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::slow_conv3d(dispatchKeySet, self, weight, std::move(kernel_size), bias, std::move(stride), std::move(padding));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SLOW_CONV3D, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(weight);trace.append_arg(std::move(kernel_size));trace.append_arg(bias);trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));
  return tt;
}

at::Tensor wrap_slow_conv_dilated2d(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::slow_conv_dilated2d(dispatchKeySet, self, weight, std::move(kernel_size), bias, std::move(stride), std::move(padding), std::move(dilation));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SLOW_CONV_DILATED2D, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(weight);trace.append_arg(std::move(kernel_size));trace.append_arg(bias);trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(std::move(dilation));
  return tt;
}

at::Tensor wrap_slow_conv_dilated3d(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::slow_conv_dilated3d(dispatchKeySet, self, weight, std::move(kernel_size), bias, std::move(stride), std::move(padding), std::move(dilation));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SLOW_CONV_DILATED3D, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(weight);trace.append_arg(std::move(kernel_size));trace.append_arg(bias);trace.append_arg(std::move(stride));trace.append_arg(std::move(padding));trace.append_arg(std::move(dilation));
  return tt;
}

at::Tensor & wrap_col2im_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::col2im_outf(dispatchKeySet, self, std::move(output_size), std::move(kernel_size), std::move(dilation), std::move(padding), std::move(stride), out);
  }
  bool flush = register_in_place(out, H_COL2IM_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(output_size));trace.append_arg(std::move(kernel_size));trace.append_arg(std::move(dilation));trace.append_arg(std::move(padding));trace.append_arg(std::move(stride));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_col2im(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::col2im(dispatchKeySet, self, std::move(output_size), std::move(kernel_size), std::move(dilation), std::move(padding), std::move(stride));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_COL2IM, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(output_size));trace.append_arg(std::move(kernel_size));trace.append_arg(std::move(dilation));trace.append_arg(std::move(padding));trace.append_arg(std::move(stride));
  return tt;
}

at::Tensor & wrap_col2im_backward_grad_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::col2im_backward_outf(dispatchKeySet, grad_output, std::move(kernel_size), std::move(dilation), std::move(padding), std::move(stride), grad_input);
  }
  bool flush = register_in_place(grad_input, H_COL2IM_BACKWARD_GRAD_INPUT, dispatchKeySet);
  trace.append_arg(grad_output);trace.append_arg(std::move(kernel_size));trace.append_arg(std::move(dilation));trace.append_arg(std::move(padding));trace.append_arg(std::move(stride));trace.append_arg(grad_input);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return grad_input;
}

at::Tensor wrap_col2im_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::col2im_backward(dispatchKeySet, grad_output, std::move(kernel_size), std::move(dilation), std::move(padding), std::move(stride));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_COL2IM_BACKWARD, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(std::move(kernel_size));trace.append_arg(std::move(dilation));trace.append_arg(std::move(padding));trace.append_arg(std::move(stride));
  return tt;
}

at::Tensor wrap_column_stack(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::column_stack(dispatchKeySet, std::move(tensors));
  }
  auto defaults = compute_dtype(tensors);
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_COLUMN_STACK, default_dtype, default_device);
  trace.append_arg(std::move(tensors));
  return tt;
}

at::Tensor & wrap_column_stack_out(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::column_stack_outf(dispatchKeySet, std::move(tensors), out);
  }
  bool flush = register_in_place(out, H_COLUMN_STACK_OUT, dispatchKeySet);
  trace.append_arg(std::move(tensors));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_im2col_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::im2col_outf(dispatchKeySet, self, std::move(kernel_size), std::move(dilation), std::move(padding), std::move(stride), out);
  }
  bool flush = register_in_place(out, H_IM2COL_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(kernel_size));trace.append_arg(std::move(dilation));trace.append_arg(std::move(padding));trace.append_arg(std::move(stride));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_im2col(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::im2col(dispatchKeySet, self, std::move(kernel_size), std::move(dilation), std::move(padding), std::move(stride));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_IM2COL, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(kernel_size));trace.append_arg(std::move(dilation));trace.append_arg(std::move(padding));trace.append_arg(std::move(stride));
  return tt;
}

at::Tensor & wrap_im2col_backward_grad_input(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, at::IntArrayRef input_size, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride, at::Tensor & grad_input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::im2col_backward_outf(dispatchKeySet, grad_output, std::move(input_size), std::move(kernel_size), std::move(dilation), std::move(padding), std::move(stride), grad_input);
  }
  bool flush = register_in_place(grad_input, H_IM2COL_BACKWARD_GRAD_INPUT, dispatchKeySet);
  trace.append_arg(grad_output);trace.append_arg(std::move(input_size));trace.append_arg(std::move(kernel_size));trace.append_arg(std::move(dilation));trace.append_arg(std::move(padding));trace.append_arg(std::move(stride));trace.append_arg(grad_input);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return grad_input;
}

at::Tensor wrap_im2col_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, at::IntArrayRef input_size, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::im2col_backward(dispatchKeySet, grad_output, std::move(input_size), std::move(kernel_size), std::move(dilation), std::move(padding), std::move(stride));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_IM2COL_BACKWARD, grad_output.dtype(), grad_output.device());
  trace.append_arg(grad_output);trace.append_arg(std::move(input_size));trace.append_arg(std::move(kernel_size));trace.append_arg(std::move(dilation));trace.append_arg(std::move(padding));trace.append_arg(std::move(stride));
  return tt;
}

at::Tensor wrap_isfinite(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::isfinite(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ISFINITE, scalarTypeToTypeMeta(kBool), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor wrap_isinf(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::isinf(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ISINF, scalarTypeToTypeMeta(kBool), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor wrap_isposinf(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::isposinf(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ISPOSINF, scalarTypeToTypeMeta(kBool), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_isposinf_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::isposinf_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_ISPOSINF_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_isneginf(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::isneginf(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_ISNEGINF, scalarTypeToTypeMeta(kBool), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_isneginf_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::isneginf_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_ISNEGINF_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap__add_batch_dim(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t batch_dim, int64_t level) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_add_batch_dim(dispatchKeySet, self, batch_dim, level);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__ADD_BATCH_DIM, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(batch_dim);trace.append_arg(level);
  return tt;
}

at::Tensor wrap__remove_batch_dim(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t level, int64_t batch_size, int64_t out_dim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_remove_batch_dim(dispatchKeySet, self, level, batch_size, out_dim);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__REMOVE_BATCH_DIM, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(level);trace.append_arg(batch_size);trace.append_arg(out_dim);
  return tt;
}

at::Tensor & wrap_special_entr_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::special_entr_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_SPECIAL_ENTR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_special_expm1(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::special_expm1(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SPECIAL_EXPM1, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_special_expm1_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::special_expm1_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_SPECIAL_EXPM1_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_special_exp2(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::special_exp2(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SPECIAL_EXP2, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_special_exp2_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::special_exp2_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_SPECIAL_EXP2_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_special_psi(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::special_psi(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SPECIAL_PSI, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_special_psi_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::special_psi_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_SPECIAL_PSI_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_special_digamma(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::special_digamma(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SPECIAL_DIGAMMA, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_special_digamma_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::special_digamma_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_SPECIAL_DIGAMMA_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_special_gammaln(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::special_gammaln(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SPECIAL_GAMMALN, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_special_gammaln_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::special_gammaln_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_SPECIAL_GAMMALN_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_special_erf(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::special_erf(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SPECIAL_ERF, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_special_erf_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::special_erf_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_SPECIAL_ERF_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_special_erfc(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::special_erfc(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SPECIAL_ERFC, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_special_erfc_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::special_erfc_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_SPECIAL_ERFC_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_special_erfinv(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::special_erfinv(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SPECIAL_ERFINV, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_special_erfinv_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::special_erfinv_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_SPECIAL_ERFINV_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_special_ndtr(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::special_ndtr(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SPECIAL_NDTR, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_special_ndtr_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::special_ndtr_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_SPECIAL_NDTR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_special_xlog1py_self_scalar(c10::DispatchKeySet dispatchKeySet, const at::Scalar & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::special_xlog1py(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SPECIAL_XLOG1PY_SELF_SCALAR, other.dtype(), other.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor wrap_special_xlog1py_other_scalar(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::special_xlog1py(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SPECIAL_XLOG1PY_OTHER_SCALAR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_special_xlog1py_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::special_xlog1py_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_SPECIAL_XLOG1PY_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_special_xlog1py_self_scalar_out(c10::DispatchKeySet dispatchKeySet, const at::Scalar & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::special_xlog1py_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_SPECIAL_XLOG1PY_SELF_SCALAR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_special_xlog1py_other_scalar_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::special_xlog1py_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_SPECIAL_XLOG1PY_OTHER_SCALAR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_special_i0(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::special_i0(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SPECIAL_I0, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_special_i0_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::special_i0_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_SPECIAL_I0_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_special_i0e_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::special_i0e_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_SPECIAL_I0E_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_special_i1_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::special_i1_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_SPECIAL_I1_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_special_i1e_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::special_i1e_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_SPECIAL_I1E_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_special_logit(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<double> eps) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::special_logit(dispatchKeySet, self, eps);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SPECIAL_LOGIT, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(eps);
  return tt;
}

at::Tensor & wrap_special_logit_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<double> eps, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::special_logit_outf(dispatchKeySet, self, eps, out);
  }
  bool flush = register_in_place(out, H_SPECIAL_LOGIT_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(eps);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_special_expit(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::special_expit(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SPECIAL_EXPIT, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_special_expit_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::special_expit_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_SPECIAL_EXPIT_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_fft_fft(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<c10::string_view> norm) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fft_fft(dispatchKeySet, self, n, dim, std::move(norm));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_FFT_FFT, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(n);trace.append_arg(dim);trace.append_arg(std::move(norm));
  return tt;
}

at::Tensor & wrap_fft_fft_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<c10::string_view> norm, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fft_fft_outf(dispatchKeySet, self, n, dim, std::move(norm), out);
  }
  bool flush = register_in_place(out, H_FFT_FFT_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(n);trace.append_arg(dim);trace.append_arg(std::move(norm));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_fft_ifft(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<c10::string_view> norm) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fft_ifft(dispatchKeySet, self, n, dim, std::move(norm));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_FFT_IFFT, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(n);trace.append_arg(dim);trace.append_arg(std::move(norm));
  return tt;
}

at::Tensor & wrap_fft_ifft_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<c10::string_view> norm, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fft_ifft_outf(dispatchKeySet, self, n, dim, std::move(norm), out);
  }
  bool flush = register_in_place(out, H_FFT_IFFT_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(n);trace.append_arg(dim);trace.append_arg(std::move(norm));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_fft_rfft(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<c10::string_view> norm) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fft_rfft(dispatchKeySet, self, n, dim, std::move(norm));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_FFT_RFFT, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(n);trace.append_arg(dim);trace.append_arg(std::move(norm));
  return tt;
}

at::Tensor & wrap_fft_rfft_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<c10::string_view> norm, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fft_rfft_outf(dispatchKeySet, self, n, dim, std::move(norm), out);
  }
  bool flush = register_in_place(out, H_FFT_RFFT_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(n);trace.append_arg(dim);trace.append_arg(std::move(norm));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_fft_irfft(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<c10::string_view> norm) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fft_irfft(dispatchKeySet, self, n, dim, std::move(norm));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_FFT_IRFFT, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(n);trace.append_arg(dim);trace.append_arg(std::move(norm));
  return tt;
}

at::Tensor & wrap_fft_irfft_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<c10::string_view> norm, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fft_irfft_outf(dispatchKeySet, self, n, dim, std::move(norm), out);
  }
  bool flush = register_in_place(out, H_FFT_IRFFT_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(n);trace.append_arg(dim);trace.append_arg(std::move(norm));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_fft_hfft(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<c10::string_view> norm) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fft_hfft(dispatchKeySet, self, n, dim, std::move(norm));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_FFT_HFFT, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(n);trace.append_arg(dim);trace.append_arg(std::move(norm));
  return tt;
}

at::Tensor & wrap_fft_hfft_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<c10::string_view> norm, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fft_hfft_outf(dispatchKeySet, self, n, dim, std::move(norm), out);
  }
  bool flush = register_in_place(out, H_FFT_HFFT_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(n);trace.append_arg(dim);trace.append_arg(std::move(norm));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_fft_ihfft(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<c10::string_view> norm) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fft_ihfft(dispatchKeySet, self, n, dim, std::move(norm));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_FFT_IHFFT, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(n);trace.append_arg(dim);trace.append_arg(std::move(norm));
  return tt;
}

at::Tensor & wrap_fft_ihfft_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<c10::string_view> norm, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fft_ihfft_outf(dispatchKeySet, self, n, dim, std::move(norm), out);
  }
  bool flush = register_in_place(out, H_FFT_IHFFT_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(n);trace.append_arg(dim);trace.append_arg(std::move(norm));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_fft_fft2(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::IntArrayRef> s, at::IntArrayRef dim, c10::optional<c10::string_view> norm) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fft_fft2(dispatchKeySet, self, std::move(s), std::move(dim), std::move(norm));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_FFT_FFT2, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(s));trace.append_arg(std::move(dim));trace.append_arg(std::move(norm));
  return tt;
}

at::Tensor & wrap_fft_fft2_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::IntArrayRef> s, at::IntArrayRef dim, c10::optional<c10::string_view> norm, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fft_fft2_outf(dispatchKeySet, self, std::move(s), std::move(dim), std::move(norm), out);
  }
  bool flush = register_in_place(out, H_FFT_FFT2_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(s));trace.append_arg(std::move(dim));trace.append_arg(std::move(norm));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_fft_ifft2(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::IntArrayRef> s, at::IntArrayRef dim, c10::optional<c10::string_view> norm) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fft_ifft2(dispatchKeySet, self, std::move(s), std::move(dim), std::move(norm));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_FFT_IFFT2, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(s));trace.append_arg(std::move(dim));trace.append_arg(std::move(norm));
  return tt;
}

at::Tensor & wrap_fft_ifft2_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::IntArrayRef> s, at::IntArrayRef dim, c10::optional<c10::string_view> norm, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fft_ifft2_outf(dispatchKeySet, self, std::move(s), std::move(dim), std::move(norm), out);
  }
  bool flush = register_in_place(out, H_FFT_IFFT2_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(s));trace.append_arg(std::move(dim));trace.append_arg(std::move(norm));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_fft_rfft2(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::IntArrayRef> s, at::IntArrayRef dim, c10::optional<c10::string_view> norm) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fft_rfft2(dispatchKeySet, self, std::move(s), std::move(dim), std::move(norm));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_FFT_RFFT2, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(s));trace.append_arg(std::move(dim));trace.append_arg(std::move(norm));
  return tt;
}

at::Tensor & wrap_fft_rfft2_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::IntArrayRef> s, at::IntArrayRef dim, c10::optional<c10::string_view> norm, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fft_rfft2_outf(dispatchKeySet, self, std::move(s), std::move(dim), std::move(norm), out);
  }
  bool flush = register_in_place(out, H_FFT_RFFT2_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(s));trace.append_arg(std::move(dim));trace.append_arg(std::move(norm));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_fft_irfft2(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::IntArrayRef> s, at::IntArrayRef dim, c10::optional<c10::string_view> norm) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fft_irfft2(dispatchKeySet, self, std::move(s), std::move(dim), std::move(norm));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_FFT_IRFFT2, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(s));trace.append_arg(std::move(dim));trace.append_arg(std::move(norm));
  return tt;
}

at::Tensor & wrap_fft_irfft2_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::IntArrayRef> s, at::IntArrayRef dim, c10::optional<c10::string_view> norm, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fft_irfft2_outf(dispatchKeySet, self, std::move(s), std::move(dim), std::move(norm), out);
  }
  bool flush = register_in_place(out, H_FFT_IRFFT2_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(s));trace.append_arg(std::move(dim));trace.append_arg(std::move(norm));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_fft_fftn(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::IntArrayRef> s, c10::optional<at::IntArrayRef> dim, c10::optional<c10::string_view> norm) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fft_fftn(dispatchKeySet, self, std::move(s), std::move(dim), std::move(norm));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_FFT_FFTN, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(s));trace.append_arg(std::move(dim));trace.append_arg(std::move(norm));
  return tt;
}

at::Tensor & wrap_fft_fftn_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::IntArrayRef> s, c10::optional<at::IntArrayRef> dim, c10::optional<c10::string_view> norm, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fft_fftn_outf(dispatchKeySet, self, std::move(s), std::move(dim), std::move(norm), out);
  }
  bool flush = register_in_place(out, H_FFT_FFTN_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(s));trace.append_arg(std::move(dim));trace.append_arg(std::move(norm));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_fft_ifftn(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::IntArrayRef> s, c10::optional<at::IntArrayRef> dim, c10::optional<c10::string_view> norm) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fft_ifftn(dispatchKeySet, self, std::move(s), std::move(dim), std::move(norm));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_FFT_IFFTN, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(s));trace.append_arg(std::move(dim));trace.append_arg(std::move(norm));
  return tt;
}

at::Tensor & wrap_fft_ifftn_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::IntArrayRef> s, c10::optional<at::IntArrayRef> dim, c10::optional<c10::string_view> norm, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fft_ifftn_outf(dispatchKeySet, self, std::move(s), std::move(dim), std::move(norm), out);
  }
  bool flush = register_in_place(out, H_FFT_IFFTN_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(s));trace.append_arg(std::move(dim));trace.append_arg(std::move(norm));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_fft_rfftn(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::IntArrayRef> s, c10::optional<at::IntArrayRef> dim, c10::optional<c10::string_view> norm) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fft_rfftn(dispatchKeySet, self, std::move(s), std::move(dim), std::move(norm));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_FFT_RFFTN, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(s));trace.append_arg(std::move(dim));trace.append_arg(std::move(norm));
  return tt;
}

at::Tensor & wrap_fft_rfftn_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::IntArrayRef> s, c10::optional<at::IntArrayRef> dim, c10::optional<c10::string_view> norm, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fft_rfftn_outf(dispatchKeySet, self, std::move(s), std::move(dim), std::move(norm), out);
  }
  bool flush = register_in_place(out, H_FFT_RFFTN_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(s));trace.append_arg(std::move(dim));trace.append_arg(std::move(norm));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_fft_irfftn(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::IntArrayRef> s, c10::optional<at::IntArrayRef> dim, c10::optional<c10::string_view> norm) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fft_irfftn(dispatchKeySet, self, std::move(s), std::move(dim), std::move(norm));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_FFT_IRFFTN, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(s));trace.append_arg(std::move(dim));trace.append_arg(std::move(norm));
  return tt;
}

at::Tensor & wrap_fft_irfftn_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::IntArrayRef> s, c10::optional<at::IntArrayRef> dim, c10::optional<c10::string_view> norm, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fft_irfftn_outf(dispatchKeySet, self, std::move(s), std::move(dim), std::move(norm), out);
  }
  bool flush = register_in_place(out, H_FFT_IRFFTN_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(s));trace.append_arg(std::move(dim));trace.append_arg(std::move(norm));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_fft_fftfreq(c10::DispatchKeySet dispatchKeySet, int64_t n, double d, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fft_fftfreq(dispatchKeySet, n, d, std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_FFT_FFTFREQ, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(n);trace.append_arg(d);trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor & wrap_fft_fftfreq_out(c10::DispatchKeySet dispatchKeySet, int64_t n, double d, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fft_fftfreq_outf(dispatchKeySet, n, d, out);
  }
  bool flush = register_in_place(out, H_FFT_FFTFREQ_OUT, dispatchKeySet);
  trace.append_arg(n);trace.append_arg(d);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_fft_rfftfreq(c10::DispatchKeySet dispatchKeySet, int64_t n, double d, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fft_rfftfreq(dispatchKeySet, n, d, std::move(dtype), std::move(layout), std::move(device), pin_memory);
  }
  auto defaults = compute_dtype();
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_FFT_RFFTFREQ, dtype ? scalarTypeToTypeMeta(*dtype) : default_dtype, device ? *device : default_device);
  trace.append_arg(n);trace.append_arg(d);trace.append_arg(std::move(dtype));trace.append_arg(std::move(layout));trace.append_arg(std::move(device));trace.append_arg(pin_memory);
  return tt;
}

at::Tensor & wrap_fft_rfftfreq_out(c10::DispatchKeySet dispatchKeySet, int64_t n, double d, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fft_rfftfreq_outf(dispatchKeySet, n, d, out);
  }
  bool flush = register_in_place(out, H_FFT_RFFTFREQ_OUT, dispatchKeySet);
  trace.append_arg(n);trace.append_arg(d);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_fft_fftshift(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::IntArrayRef> dim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fft_fftshift(dispatchKeySet, self, std::move(dim));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_FFT_FFTSHIFT, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim));
  return tt;
}

at::Tensor wrap_fft_ifftshift(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::IntArrayRef> dim) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::fft_ifftshift(dispatchKeySet, self, std::move(dim));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_FFT_IFFTSHIFT, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(dim));
  return tt;
}

at::Tensor wrap_linalg_cholesky(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::linalg_cholesky(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LINALG_CHOLESKY, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_linalg_cholesky_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::linalg_cholesky_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_LINALG_CHOLESKY_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_linalg_det(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::linalg_det(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LINALG_DET, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_linalg_det_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::linalg_det_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_LINALG_DET_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_det(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::det(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_DET, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor wrap_linalg_eigvals(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::linalg_eigvals(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LINALG_EIGVALS, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_linalg_eigvals_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::linalg_eigvals_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_LINALG_EIGVALS_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_linalg_eigvalsh(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::string_view UPLO) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::linalg_eigvalsh(dispatchKeySet, self, std::move(UPLO));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LINALG_EIGVALSH, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(UPLO));
  return tt;
}

at::Tensor & wrap_linalg_eigvalsh_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::string_view UPLO, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::linalg_eigvalsh_outf(dispatchKeySet, self, std::move(UPLO), out);
  }
  bool flush = register_in_place(out, H_LINALG_EIGVALSH_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(UPLO));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_linalg_householder_product(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & tau) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::linalg_householder_product(dispatchKeySet, input, tau);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LINALG_HOUSEHOLDER_PRODUCT, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(tau);
  return tt;
}

at::Tensor & wrap_linalg_householder_product_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & tau, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::linalg_householder_product_outf(dispatchKeySet, input, tau, out);
  }
  bool flush = register_in_place(out, H_LINALG_HOUSEHOLDER_PRODUCT_OUT, dispatchKeySet);
  trace.append_arg(input);trace.append_arg(tau);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap__linalg_inv_out_helper_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, at::Tensor & infos_lu, at::Tensor & infos_getri) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_linalg_inv_out_helper_(dispatchKeySet, self, infos_lu, infos_getri);
  }
  bool flush = register_in_place(self, H__LINALG_INV_OUT_HELPER_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(infos_lu);trace.append_arg(infos_getri);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_linalg_inv(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::linalg_inv(dispatchKeySet, self);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LINALG_INV, self.dtype(), self.device());
  trace.append_arg(self);
  return tt;
}

at::Tensor & wrap_linalg_inv_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::linalg_inv_outf(dispatchKeySet, self, out);
  }
  bool flush = register_in_place(out, H_LINALG_INV_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_inner(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::inner(dispatchKeySet, self, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_INNER, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_inner_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::inner_outf(dispatchKeySet, self, other, out);
  }
  bool flush = register_in_place(out, H_INNER_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_outer(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & vec2) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::outer(dispatchKeySet, self, vec2);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_OUTER, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(vec2);
  return tt;
}

at::Tensor & wrap_outer_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & vec2, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::outer_outf(dispatchKeySet, self, vec2, out);
  }
  bool flush = register_in_place(out, H_OUTER_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(vec2);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_ger(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & vec2) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::ger(dispatchKeySet, self, vec2);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_GER, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(vec2);
  return tt;
}

at::Tensor & wrap_ger_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & vec2, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::ger_outf(dispatchKeySet, self, vec2, out);
  }
  bool flush = register_in_place(out, H_GER_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(vec2);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_linalg_norm(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::optional<at::Scalar> & ord, c10::optional<at::IntArrayRef> dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::linalg_norm(dispatchKeySet, self, ord, std::move(dim), keepdim, std::move(dtype));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LINALG_NORM, dtype ? scalarTypeToTypeMeta(*dtype) : self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(ord);trace.append_arg(std::move(dim));trace.append_arg(keepdim);trace.append_arg(std::move(dtype));
  return tt;
}

at::Tensor wrap_linalg_norm_ord_str(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::string_view ord, c10::optional<at::IntArrayRef> dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::linalg_norm(dispatchKeySet, self, std::move(ord), std::move(dim), keepdim, std::move(dtype));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LINALG_NORM_ORD_STR, dtype ? scalarTypeToTypeMeta(*dtype) : self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(ord));trace.append_arg(std::move(dim));trace.append_arg(keepdim);trace.append_arg(std::move(dtype));
  return tt;
}

at::Tensor & wrap_linalg_norm_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::optional<at::Scalar> & ord, c10::optional<at::IntArrayRef> dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::linalg_norm_outf(dispatchKeySet, self, ord, std::move(dim), keepdim, std::move(dtype), out);
  }
  bool flush = register_in_place(out, H_LINALG_NORM_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(ord);trace.append_arg(std::move(dim));trace.append_arg(keepdim);trace.append_arg(std::move(dtype));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_linalg_norm_ord_str_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::string_view ord, c10::optional<at::IntArrayRef> dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::linalg_norm_outf(dispatchKeySet, self, std::move(ord), std::move(dim), keepdim, std::move(dtype), out);
  }
  bool flush = register_in_place(out, H_LINALG_NORM_ORD_STR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(ord));trace.append_arg(std::move(dim));trace.append_arg(keepdim);trace.append_arg(std::move(dtype));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_linalg_vector_norm(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & ord, c10::optional<at::IntArrayRef> dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::linalg_vector_norm(dispatchKeySet, self, ord, std::move(dim), keepdim, std::move(dtype));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LINALG_VECTOR_NORM, dtype ? scalarTypeToTypeMeta(*dtype) : self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(ord);trace.append_arg(std::move(dim));trace.append_arg(keepdim);trace.append_arg(std::move(dtype));
  return tt;
}

at::Tensor & wrap_linalg_vector_norm_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & ord, c10::optional<at::IntArrayRef> dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::linalg_vector_norm_outf(dispatchKeySet, self, ord, std::move(dim), keepdim, std::move(dtype), out);
  }
  bool flush = register_in_place(out, H_LINALG_VECTOR_NORM_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(ord);trace.append_arg(std::move(dim));trace.append_arg(keepdim);trace.append_arg(std::move(dtype));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_linalg_matrix_norm(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & ord, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::linalg_matrix_norm(dispatchKeySet, self, ord, std::move(dim), keepdim, std::move(dtype));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LINALG_MATRIX_NORM, dtype ? scalarTypeToTypeMeta(*dtype) : self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(ord);trace.append_arg(std::move(dim));trace.append_arg(keepdim);trace.append_arg(std::move(dtype));
  return tt;
}

at::Tensor & wrap_linalg_matrix_norm_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & ord, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::linalg_matrix_norm_outf(dispatchKeySet, self, ord, std::move(dim), keepdim, std::move(dtype), out);
  }
  bool flush = register_in_place(out, H_LINALG_MATRIX_NORM_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(ord);trace.append_arg(std::move(dim));trace.append_arg(keepdim);trace.append_arg(std::move(dtype));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_linalg_matrix_norm_str_ord(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::string_view ord, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::linalg_matrix_norm(dispatchKeySet, self, std::move(ord), std::move(dim), keepdim, std::move(dtype));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LINALG_MATRIX_NORM_STR_ORD, dtype ? scalarTypeToTypeMeta(*dtype) : self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(ord));trace.append_arg(std::move(dim));trace.append_arg(keepdim);trace.append_arg(std::move(dtype));
  return tt;
}

at::Tensor & wrap_linalg_matrix_norm_str_ord_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::string_view ord, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::linalg_matrix_norm_outf(dispatchKeySet, self, std::move(ord), std::move(dim), keepdim, std::move(dtype), out);
  }
  bool flush = register_in_place(out, H_LINALG_MATRIX_NORM_STR_ORD_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(ord));trace.append_arg(std::move(dim));trace.append_arg(keepdim);trace.append_arg(std::move(dtype));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_linalg_svdvals(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::linalg_svdvals(dispatchKeySet, input);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LINALG_SVDVALS, input.dtype(), input.device());
  trace.append_arg(input);
  return tt;
}

at::Tensor & wrap_linalg_svdvals_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::linalg_svdvals_outf(dispatchKeySet, input, out);
  }
  bool flush = register_in_place(out, H_LINALG_SVDVALS_OUT, dispatchKeySet);
  trace.append_arg(input);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_linalg_cond(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::optional<at::Scalar> & p) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::linalg_cond(dispatchKeySet, self, p);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LINALG_COND, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(p);
  return tt;
}

at::Tensor & wrap_linalg_cond_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::optional<at::Scalar> & p, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::linalg_cond_outf(dispatchKeySet, self, p, out);
  }
  bool flush = register_in_place(out, H_LINALG_COND_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(p);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_linalg_cond_p_str(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::string_view p) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::linalg_cond(dispatchKeySet, self, std::move(p));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LINALG_COND_P_STR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(std::move(p));
  return tt;
}

at::Tensor & wrap_linalg_cond_p_str_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::string_view p, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::linalg_cond_outf(dispatchKeySet, self, std::move(p), out);
  }
  bool flush = register_in_place(out, H_LINALG_COND_P_STR_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(std::move(p));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_linalg_pinv(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, double rcond, bool hermitian) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::linalg_pinv(dispatchKeySet, self, rcond, hermitian);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LINALG_PINV, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(rcond);trace.append_arg(hermitian);
  return tt;
}

at::Tensor wrap_linalg_pinv_rcond_tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & rcond, bool hermitian) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::linalg_pinv(dispatchKeySet, self, rcond, hermitian);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LINALG_PINV_RCOND_TENSOR, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(rcond);trace.append_arg(hermitian);
  return tt;
}

at::Tensor & wrap_linalg_pinv_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, double rcond, bool hermitian, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::linalg_pinv_outf(dispatchKeySet, self, rcond, hermitian, out);
  }
  bool flush = register_in_place(out, H_LINALG_PINV_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(rcond);trace.append_arg(hermitian);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap_linalg_pinv_out_rcond_tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & rcond, bool hermitian, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::linalg_pinv_outf(dispatchKeySet, self, rcond, hermitian, out);
  }
  bool flush = register_in_place(out, H_LINALG_PINV_OUT_RCOND_TENSOR, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(rcond);trace.append_arg(hermitian);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor & wrap__linalg_solve_out_helper_(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, at::Tensor & other, at::Tensor & infos) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_linalg_solve_out_helper_(dispatchKeySet, self, other, infos);
  }
  bool flush = register_in_place(self, H__LINALG_SOLVE_OUT_HELPER_, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(infos);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return self;
}

at::Tensor wrap_linalg_solve(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & other) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::linalg_solve(dispatchKeySet, input, other);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LINALG_SOLVE, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(other);
  return tt;
}

at::Tensor & wrap_linalg_solve_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & other, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::linalg_solve_outf(dispatchKeySet, input, other, out);
  }
  bool flush = register_in_place(out, H_LINALG_SOLVE_OUT, dispatchKeySet);
  trace.append_arg(input);trace.append_arg(other);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_linalg_tensorinv(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t ind) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::linalg_tensorinv(dispatchKeySet, self, ind);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LINALG_TENSORINV, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(ind);
  return tt;
}

at::Tensor & wrap_linalg_tensorinv_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t ind, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::linalg_tensorinv_outf(dispatchKeySet, self, ind, out);
  }
  bool flush = register_in_place(out, H_LINALG_TENSORINV_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(ind);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_linalg_tensorsolve(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, c10::optional<at::IntArrayRef> dims) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::linalg_tensorsolve(dispatchKeySet, self, other, std::move(dims));
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LINALG_TENSORSOLVE, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(std::move(dims));
  return tt;
}

at::Tensor & wrap_linalg_tensorsolve_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, c10::optional<at::IntArrayRef> dims, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::linalg_tensorsolve_outf(dispatchKeySet, self, other, std::move(dims), out);
  }
  bool flush = register_in_place(out, H_LINALG_TENSORSOLVE_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(std::move(dims));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_linalg_matrix_power(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t n) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::linalg_matrix_power(dispatchKeySet, self, n);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LINALG_MATRIX_POWER, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(n);
  return tt;
}

at::Tensor & wrap_linalg_matrix_power_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t n, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::linalg_matrix_power_outf(dispatchKeySet, self, n, out);
  }
  bool flush = register_in_place(out, H_LINALG_MATRIX_POWER_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(n);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_linalg_matrix_rank(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<double> tol, bool hermitian) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::linalg_matrix_rank(dispatchKeySet, self, tol, hermitian);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LINALG_MATRIX_RANK, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(tol);trace.append_arg(hermitian);
  return tt;
}

at::Tensor & wrap_linalg_matrix_rank_out(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<double> tol, bool hermitian, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::linalg_matrix_rank_outf(dispatchKeySet, self, tol, hermitian, out);
  }
  bool flush = register_in_place(out, H_LINALG_MATRIX_RANK_OUT, dispatchKeySet);
  trace.append_arg(self);trace.append_arg(tol);trace.append_arg(hermitian);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_linalg_matrix_rank_tol_tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & tol, bool hermitian) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::linalg_matrix_rank(dispatchKeySet, input, tol, hermitian);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_LINALG_MATRIX_RANK_TOL_TENSOR, input.dtype(), input.device());
  trace.append_arg(input);trace.append_arg(tol);trace.append_arg(hermitian);
  return tt;
}

at::Tensor & wrap_linalg_matrix_rank_out_tol_tensor(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & tol, bool hermitian, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::linalg_matrix_rank_outf(dispatchKeySet, input, tol, hermitian, out);
  }
  bool flush = register_in_place(out, H_LINALG_MATRIX_RANK_OUT_TOL_TENSOR, dispatchKeySet);
  trace.append_arg(input);trace.append_arg(tol);trace.append_arg(hermitian);trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap_linalg_multi_dot(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::linalg_multi_dot(dispatchKeySet, std::move(tensors));
  }
  auto defaults = compute_dtype(tensors);
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_LINALG_MULTI_DOT, default_dtype, default_device);
  trace.append_arg(std::move(tensors));
  return tt;
}

at::Tensor & wrap_linalg_multi_dot_out(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors, at::Tensor & out) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::linalg_multi_dot_outf(dispatchKeySet, std::move(tensors), out);
  }
  bool flush = register_in_place(out, H_LINALG_MULTI_DOT_OUT, dispatchKeySet);
  trace.append_arg(std::move(tensors));trace.append_arg(out);
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return out;
}

at::Tensor wrap__test_serialization_subcmul(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_test_serialization_subcmul(dispatchKeySet, self, other, alpha);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__TEST_SERIALIZATION_SUBCMUL, self.dtype(), self.device());
  trace.append_arg(self);trace.append_arg(other);trace.append_arg(alpha);
  return tt;
}

at::Tensor wrap__test_optional_intlist(c10::DispatchKeySet dispatchKeySet, const at::Tensor & values, c10::optional<at::IntArrayRef> addends) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_test_optional_intlist(dispatchKeySet, values, std::move(addends));
  }
  auto tt = register_new_tensor(dispatchKeySet, H__TEST_OPTIONAL_INTLIST, values.dtype(), values.device());
  trace.append_arg(values);trace.append_arg(std::move(addends));
  return tt;
}

at::Tensor wrap__test_optional_filled_intlist(c10::DispatchKeySet dispatchKeySet, const at::Tensor & values, c10::optional<at::IntArrayRef> addends) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_test_optional_filled_intlist(dispatchKeySet, values, std::move(addends));
  }
  auto tt = register_new_tensor(dispatchKeySet, H__TEST_OPTIONAL_FILLED_INTLIST, values.dtype(), values.device());
  trace.append_arg(values);trace.append_arg(std::move(addends));
  return tt;
}

at::Tensor wrap__test_optional_floatlist(c10::DispatchKeySet dispatchKeySet, const at::Tensor & values, c10::optional<at::ArrayRef<double>> addends) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_test_optional_floatlist(dispatchKeySet, values, addends);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__TEST_OPTIONAL_FLOATLIST, values.dtype(), values.device());
  trace.append_arg(values);trace.append_arg(addends);
  return tt;
}

at::Tensor wrap__test_string_default(c10::DispatchKeySet dispatchKeySet, const at::Tensor & dummy, c10::string_view a, c10::string_view b) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_test_string_default(dispatchKeySet, dummy, std::move(a), std::move(b));
  }
  auto tt = register_new_tensor(dispatchKeySet, H__TEST_STRING_DEFAULT, dummy.dtype(), dummy.device());
  trace.append_arg(dummy);trace.append_arg(std::move(a));trace.append_arg(std::move(b));
  return tt;
}

at::Tensor wrap__test_ambiguous_defaults_a(c10::DispatchKeySet dispatchKeySet, const at::Tensor & dummy, int64_t a, int64_t b) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_test_ambiguous_defaults(dispatchKeySet, dummy, a, b);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__TEST_AMBIGUOUS_DEFAULTS_A, dummy.dtype(), dummy.device());
  trace.append_arg(dummy);trace.append_arg(a);trace.append_arg(b);
  return tt;
}

at::Tensor wrap__test_ambiguous_defaults_b(c10::DispatchKeySet dispatchKeySet, const at::Tensor & dummy, int64_t a, c10::string_view b) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_test_ambiguous_defaults(dispatchKeySet, dummy, a, std::move(b));
  }
  auto tt = register_new_tensor(dispatchKeySet, H__TEST_AMBIGUOUS_DEFAULTS_B, dummy.dtype(), dummy.device());
  trace.append_arg(dummy);trace.append_arg(a);trace.append_arg(std::move(b));
  return tt;
}

at::Tensor wrap_segment_reduce(c10::DispatchKeySet dispatchKeySet, const at::Tensor & data, c10::string_view reduce, const c10::optional<at::Tensor> & lengths, const c10::optional<at::Tensor> & indices, int64_t axis, bool unsafe, const c10::optional<at::Scalar> & initial) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::segment_reduce(dispatchKeySet, data, std::move(reduce), lengths, indices, axis, unsafe, initial);
  }
  auto tt = register_new_tensor(dispatchKeySet, H_SEGMENT_REDUCE, data.dtype(), data.device());
  trace.append_arg(data);trace.append_arg(std::move(reduce));trace.append_arg(lengths);trace.append_arg(indices);trace.append_arg(axis);trace.append_arg(unsafe);trace.append_arg(initial);
  return tt;
}

at::Tensor wrap__segment_reduce_backward(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, const at::Tensor & output, const at::Tensor & data, c10::string_view reduce, const c10::optional<at::Tensor> & lengths) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::_segment_reduce_backward(dispatchKeySet, grad, output, data, std::move(reduce), lengths);
  }
  auto tt = register_new_tensor(dispatchKeySet, H__SEGMENT_REDUCE_BACKWARD, grad.dtype(), grad.device());
  trace.append_arg(grad);trace.append_arg(output);trace.append_arg(data);trace.append_arg(std::move(reduce));trace.append_arg(lengths);
  return tt;
}

at::Tensor wrap_pad_sequence(c10::DispatchKeySet dispatchKeySet, at::TensorList sequences, bool batch_first, double padding_value) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::pad_sequence(dispatchKeySet, std::move(sequences), batch_first, padding_value);
  }
  auto defaults = compute_dtype(sequences);
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_PAD_SEQUENCE, default_dtype, default_device);
  trace.append_arg(std::move(sequences));trace.append_arg(batch_first);trace.append_arg(padding_value);
  return tt;
}

at::Tensor wrap_flatten_dense_tensors(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
  if (trace.is_flushing()) {
    dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);
    return at::redispatch::flatten_dense_tensors(dispatchKeySet, std::move(tensors));
  }
  auto defaults = compute_dtype(tensors);
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;
  auto tt = register_new_tensor(dispatchKeySet, H_FLATTEN_DENSE_TENSORS, default_dtype, default_device);
  trace.append_arg(std::move(tensors));
  return tt;
}
