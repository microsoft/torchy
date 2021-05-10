case H__CAST_BYTE:
  set(op.tensor, at::redispatch::_cast_Byte(ks, get<at::Tensor>(op.args[0]), get<bool>(op.args[1])));
  break;

case H__CAST_CHAR:
  set(op.tensor, at::redispatch::_cast_Char(ks, get<at::Tensor>(op.args[0]), get<bool>(op.args[1])));
  break;

case H__CAST_DOUBLE:
  set(op.tensor, at::redispatch::_cast_Double(ks, get<at::Tensor>(op.args[0]), get<bool>(op.args[1])));
  break;

case H__CAST_FLOAT:
  set(op.tensor, at::redispatch::_cast_Float(ks, get<at::Tensor>(op.args[0]), get<bool>(op.args[1])));
  break;

case H__CAST_INT:
  set(op.tensor, at::redispatch::_cast_Int(ks, get<at::Tensor>(op.args[0]), get<bool>(op.args[1])));
  break;

case H__CAST_LONG:
  set(op.tensor, at::redispatch::_cast_Long(ks, get<at::Tensor>(op.args[0]), get<bool>(op.args[1])));
  break;

case H__CAST_SHORT:
  set(op.tensor, at::redispatch::_cast_Short(ks, get<at::Tensor>(op.args[0]), get<bool>(op.args[1])));
  break;

case H__CAST_HALF:
  set(op.tensor, at::redispatch::_cast_Half(ks, get<at::Tensor>(op.args[0]), get<bool>(op.args[1])));
  break;

case H__FW_PRIMAL:
  set(op.tensor, at::redispatch::_fw_primal(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1])));
  break;

case H__MAKE_DUAL:
  set(op.tensor, at::redispatch::_make_dual(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<int64_t>(op.args[2])));
  break;

// skip std::tuple<at::Tensor,at::Tensor> _unpack_dual(const at::Tensor & dual, int64_t level)

case H_RENAME_:
  init_update_in_place(op.tensor);
  at::redispatch::rename_(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::DimnameList>>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_RENAME:
  set(op.tensor, at::redispatch::rename(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::DimnameList>>(op.args[1])));
  break;

case H_ALIGN_TO:
  set(op.tensor, at::redispatch::align_to(ks, get<at::Tensor>(op.args[0]), get<at::DimnameList>(op.args[1])));
  break;

case H_ALIGN_TO_ELLIPSIS_IDX:
  set(op.tensor, at::redispatch::align_to(ks, get<at::Tensor>(op.args[0]), get<at::DimnameList>(op.args[1]), get<int64_t>(op.args[2])));
  break;

case H_ALIGN_AS:
  set(op.tensor, at::redispatch::align_as(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

// skip std::vector<at::Tensor> align_tensors(at::TensorList tensors)

// skip void _assert_async(const at::Tensor & self)

case H_REFINE_NAMES:
  set(op.tensor, at::redispatch::refine_names(ks, get<at::Tensor>(op.args[0]), get<at::DimnameList>(op.args[1])));
  break;

// skip bool _use_cudnn_ctc_loss(const at::Tensor & log_probs, const at::Tensor & targets, at::IntArrayRef input_lengths, at::IntArrayRef target_lengths, int64_t blank)

// skip std::tuple<at::Tensor,at::Tensor> _cudnn_ctc_loss(const at::Tensor & log_probs, const at::Tensor & targets, at::IntArrayRef input_lengths, at::IntArrayRef target_lengths, int64_t blank, bool deterministic, bool zero_infinity)

// skip bool _use_cudnn_rnn_flatten_weight()

case H__CUDNN_RNN_FLATTEN_WEIGHT:
  set(op.tensor, at::redispatch::_cudnn_rnn_flatten_weight(ks, get<at::TensorList>(op.args[0]), get<int64_t>(op.args[1]), get<int64_t>(op.args[2]), get<int64_t>(op.args[3]), get<int64_t>(op.args[4]), get<int64_t>(op.args[5]), get<int64_t>(op.args[6]), get<bool>(op.args[7]), get<bool>(op.args[8])));
  break;

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> _cudnn_rnn(const at::Tensor & input, at::TensorList weight, int64_t weight_stride0, const c10::optional<at::Tensor> & weight_buf, const at::Tensor & hx, const c10::optional<at::Tensor> & cx, int64_t mode, int64_t hidden_size, int64_t proj_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, at::IntArrayRef batch_sizes, const c10::optional<at::Tensor> & dropout_state)

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor,std::vector<at::Tensor>> _cudnn_rnn_backward(const at::Tensor & input, at::TensorList weight, int64_t weight_stride0, const at::Tensor & weight_buf, const at::Tensor & hx, const c10::optional<at::Tensor> & cx, const at::Tensor & output, const c10::optional<at::Tensor> & grad_output, const c10::optional<at::Tensor> & grad_hy, const c10::optional<at::Tensor> & grad_cy, int64_t mode, int64_t hidden_size, int64_t proj_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, at::IntArrayRef batch_sizes, const c10::optional<at::Tensor> & dropout_state, const at::Tensor & reserve, std::array<bool,4> output_mask)

case H__CUDNN_INIT_DROPOUT_STATE:
  set(op.tensor, at::redispatch::_cudnn_init_dropout_state(ks, get<double>(op.args[0]), get<bool>(op.args[1]), get<int64_t>(op.args[2]), get<c10::optional<at::ScalarType>>(op.args[3]), get<c10::optional<at::Layout>>(op.args[4]), get<c10::optional<at::Device>>(op.args[5]), get<c10::optional<bool>>(op.args[6])));
  break;

// skip int64_t _debug_has_internal_overlap(const at::Tensor & self)

// skip std::tuple<at::Tensor,at::Tensor> _fused_dropout(const at::Tensor & self, double p, c10::optional<at::Generator> generator)

case H__MASKED_SCALE:
  set(op.tensor, at::redispatch::_masked_scale(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<double>(op.args[2])));
  break;

// skip std::tuple<at::Tensor,at::Tensor> _sobol_engine_draw(const at::Tensor & quasi, int64_t n, const at::Tensor & sobolstate, int64_t dimension, int64_t num_generated, c10::optional<at::ScalarType> dtype)

case H__SOBOL_ENGINE_FF_:
  init_update_in_place(op.tensor);
  at::redispatch::_sobol_engine_ff_(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<at::Tensor>(op.args[2]), get<int64_t>(op.args[3]), get<int64_t>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H__SOBOL_ENGINE_SCRAMBLE_:
  init_update_in_place(op.tensor);
  at::redispatch::_sobol_engine_scramble_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<int64_t>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H__SOBOL_ENGINE_INITIALIZE_STATE_:
  init_update_in_place(op.tensor);
  at::redispatch::_sobol_engine_initialize_state_(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H__RESHAPE_FROM_TENSOR:
  set(op.tensor, at::redispatch::_reshape_from_tensor(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H__SHAPE_AS_TENSOR:
  set(op.tensor, at::redispatch::_shape_as_tensor(ks, get<at::Tensor>(op.args[0])));
  break;

case H_DROPOUT:
  set(op.tensor, at::redispatch::dropout(ks, get<at::Tensor>(op.args[0]), get<double>(op.args[1]), get<bool>(op.args[2])));
  break;

case H_DROPOUT_:
  init_update_in_place(op.tensor);
  at::redispatch::dropout_(ks, get<at::Tensor>(op.args[0]), get<double>(op.args[1]), get<bool>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_FEATURE_DROPOUT:
  set(op.tensor, at::redispatch::feature_dropout(ks, get<at::Tensor>(op.args[0]), get<double>(op.args[1]), get<bool>(op.args[2])));
  break;

case H_FEATURE_DROPOUT_:
  init_update_in_place(op.tensor);
  at::redispatch::feature_dropout_(ks, get<at::Tensor>(op.args[0]), get<double>(op.args[1]), get<bool>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_ALPHA_DROPOUT:
  set(op.tensor, at::redispatch::alpha_dropout(ks, get<at::Tensor>(op.args[0]), get<double>(op.args[1]), get<bool>(op.args[2])));
  break;

case H_ALPHA_DROPOUT_:
  init_update_in_place(op.tensor);
  at::redispatch::alpha_dropout_(ks, get<at::Tensor>(op.args[0]), get<double>(op.args[1]), get<bool>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_FEATURE_ALPHA_DROPOUT:
  set(op.tensor, at::redispatch::feature_alpha_dropout(ks, get<at::Tensor>(op.args[0]), get<double>(op.args[1]), get<bool>(op.args[2])));
  break;

case H_FEATURE_ALPHA_DROPOUT_:
  init_update_in_place(op.tensor);
  at::redispatch::feature_alpha_dropout_(ks, get<at::Tensor>(op.args[0]), get<double>(op.args[1]), get<bool>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_ABS:
  set(op.tensor, at::redispatch::abs(ks, get<at::Tensor>(op.args[0])));
  break;

case H_ABS_:
  init_update_in_place(op.tensor);
  at::redispatch::abs_(ks, get<at::Tensor>(op.args[0]));
  end_update_in_place(op.tensor);
  break;

case H_ABS_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::abs_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_ABSOLUTE:
  set(op.tensor, at::redispatch::absolute(ks, get<at::Tensor>(op.args[0])));
  break;

case H_ABSOLUTE_:
  init_update_in_place(op.tensor);
  at::redispatch::absolute_(ks, get<at::Tensor>(op.args[0]));
  end_update_in_place(op.tensor);
  break;

case H_ABSOLUTE_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::absolute_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_ANGLE:
  set(op.tensor, at::redispatch::angle(ks, get<at::Tensor>(op.args[0])));
  break;

case H_ANGLE_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::angle_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_VIEW_AS_REAL:
  set(op.tensor, at::redispatch::view_as_real(ks, get<at::Tensor>(op.args[0])));
  break;

case H_VIEW_AS_COMPLEX:
  set(op.tensor, at::redispatch::view_as_complex(ks, get<at::Tensor>(op.args[0])));
  break;

case H_SGN:
  set(op.tensor, at::redispatch::sgn(ks, get<at::Tensor>(op.args[0])));
  break;

case H_SGN_:
  init_update_in_place(op.tensor);
  at::redispatch::sgn_(ks, get<at::Tensor>(op.args[0]));
  end_update_in_place(op.tensor);
  break;

case H_SGN_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::sgn_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_REAL:
  set(op.tensor, at::redispatch::real(ks, get<at::Tensor>(op.args[0])));
  break;

case H_IMAG:
  set(op.tensor, at::redispatch::imag(ks, get<at::Tensor>(op.args[0])));
  break;

case H_CONJ:
  set(op.tensor, at::redispatch::conj(ks, get<at::Tensor>(op.args[0])));
  break;

case H_CONJ_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::conj_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H__CONJ:
  set(op.tensor, at::redispatch::_conj(ks, get<at::Tensor>(op.args[0])));
  break;

case H_ACOS_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::acos_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_ARCCOS:
  set(op.tensor, at::redispatch::arccos(ks, get<at::Tensor>(op.args[0])));
  break;

case H_ARCCOS_:
  init_update_in_place(op.tensor);
  at::redispatch::arccos_(ks, get<at::Tensor>(op.args[0]));
  end_update_in_place(op.tensor);
  break;

case H_ARCCOS_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::arccos_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_AVG_POOL1D:
  set(op.tensor, at::redispatch::avg_pool1d(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<bool>(op.args[4]), get<bool>(op.args[5])));
  break;

case H_ADAPTIVE_AVG_POOL1D:
  set(op.tensor, at::redispatch::adaptive_avg_pool1d(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1])));
  break;

// skip std::tuple<at::Tensor,at::Tensor> adaptive_max_pool1d(const at::Tensor & self, at::IntArrayRef output_size)

case H_ADD_TENSOR:
  set(op.tensor, at::redispatch::add(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Scalar>(op.args[2])));
  break;

case H_ADD__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::add_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Scalar>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_ADD_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::add_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Scalar>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H__ADD_RELU_TENSOR:
  set(op.tensor, at::redispatch::_add_relu(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Scalar>(op.args[2])));
  break;

case H__ADD_RELU__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::_add_relu_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Scalar>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H__ADD_RELU_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::_add_relu_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Scalar>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_ADD_SCALAR:
  set(op.tensor, at::redispatch::add(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Scalar>(op.args[2])));
  break;

case H_ADD__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::add_(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Scalar>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_ADDMV_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::addmv_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Scalar>(op.args[3]), get<at::Scalar>(op.args[4]), get<at::Tensor>(op.args[5]));
  end_update_in_place(op.tensor);
  break;

case H_ADDR:
  set(op.tensor, at::redispatch::addr(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Scalar>(op.args[3]), get<at::Scalar>(op.args[4])));
  break;

case H_ADDR_:
  init_update_in_place(op.tensor);
  at::redispatch::addr_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Scalar>(op.args[3]), get<at::Scalar>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_ADDR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::addr_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Scalar>(op.args[3]), get<at::Scalar>(op.args[4]), get<at::Tensor>(op.args[5]));
  end_update_in_place(op.tensor);
  break;

case H_AFFINE_GRID_GENERATOR:
  set(op.tensor, at::redispatch::affine_grid_generator(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<bool>(op.args[2])));
  break;

case H_AFFINE_GRID_GENERATOR_BACKWARD:
  set(op.tensor, at::redispatch::affine_grid_generator_backward(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<bool>(op.args[2])));
  break;

case H_ALL_DIM:
  set(op.tensor, at::redispatch::all(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<bool>(op.args[2])));
  break;

case H_ALL_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::all_outf(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<bool>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_ALL_DIMNAME:
  set(op.tensor, at::redispatch::all(ks, get<at::Tensor>(op.args[0]), get<at::Dimname>(op.args[1]), get<bool>(op.args[2])));
  break;

case H_ALL_DIMNAME_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::all_outf(ks, get<at::Tensor>(op.args[0]), get<at::Dimname>(op.args[1]), get<bool>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

// skip bool allclose(const at::Tensor & self, const at::Tensor & other, double rtol, double atol, bool equal_nan)

case H_ANY_DIM:
  set(op.tensor, at::redispatch::any(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<bool>(op.args[2])));
  break;

case H_ANY_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::any_outf(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<bool>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_ANY_DIMNAME:
  set(op.tensor, at::redispatch::any(ks, get<at::Tensor>(op.args[0]), get<at::Dimname>(op.args[1]), get<bool>(op.args[2])));
  break;

case H_ANY_DIMNAME_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::any_outf(ks, get<at::Tensor>(op.args[0]), get<at::Dimname>(op.args[1]), get<bool>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_ARANGE:
  set(op.tensor, at::redispatch::arange(ks, get<at::Scalar>(op.args[0]), get<c10::optional<at::ScalarType>>(op.args[1]), get<c10::optional<at::Layout>>(op.args[2]), get<c10::optional<at::Device>>(op.args[3]), get<c10::optional<bool>>(op.args[4])));
  break;

case H_ARANGE_START:
  set(op.tensor, at::redispatch::arange(ks, get<at::Scalar>(op.args[0]), get<at::Scalar>(op.args[1]), get<c10::optional<at::ScalarType>>(op.args[2]), get<c10::optional<at::Layout>>(op.args[3]), get<c10::optional<at::Device>>(op.args[4]), get<c10::optional<bool>>(op.args[5])));
  break;

case H_ARANGE_START_STEP:
  set(op.tensor, at::redispatch::arange(ks, get<at::Scalar>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Scalar>(op.args[2]), get<c10::optional<at::ScalarType>>(op.args[3]), get<c10::optional<at::Layout>>(op.args[4]), get<c10::optional<at::Device>>(op.args[5]), get<c10::optional<bool>>(op.args[6])));
  break;

case H_ARANGE_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::arange_outf(ks, get<at::Scalar>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_ARANGE_START_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::arange_outf(ks, get<at::Scalar>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Scalar>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H__DIM_ARANGE:
  set(op.tensor, at::redispatch::_dim_arange(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1])));
  break;

case H_ARGMAX:
  set(op.tensor, at::redispatch::argmax(ks, get<at::Tensor>(op.args[0]), get<c10::optional<int64_t>>(op.args[1]), get<bool>(op.args[2])));
  break;

case H_ARGMAX_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::argmax_outf(ks, get<at::Tensor>(op.args[0]), get<c10::optional<int64_t>>(op.args[1]), get<bool>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_ARGMIN:
  set(op.tensor, at::redispatch::argmin(ks, get<at::Tensor>(op.args[0]), get<c10::optional<int64_t>>(op.args[1]), get<bool>(op.args[2])));
  break;

case H_ARGMIN_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::argmin_outf(ks, get<at::Tensor>(op.args[0]), get<c10::optional<int64_t>>(op.args[1]), get<bool>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_ACOSH_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::acosh_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_ARCCOSH:
  set(op.tensor, at::redispatch::arccosh(ks, get<at::Tensor>(op.args[0])));
  break;

case H_ARCCOSH_:
  init_update_in_place(op.tensor);
  at::redispatch::arccosh_(ks, get<at::Tensor>(op.args[0]));
  end_update_in_place(op.tensor);
  break;

case H_ARCCOSH_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::arccosh_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_ASINH_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::asinh_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_ARCSINH:
  set(op.tensor, at::redispatch::arcsinh(ks, get<at::Tensor>(op.args[0])));
  break;

case H_ARCSINH_:
  init_update_in_place(op.tensor);
  at::redispatch::arcsinh_(ks, get<at::Tensor>(op.args[0]));
  end_update_in_place(op.tensor);
  break;

case H_ARCSINH_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::arcsinh_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_ATANH_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::atanh_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_ARCTANH:
  set(op.tensor, at::redispatch::arctanh(ks, get<at::Tensor>(op.args[0])));
  break;

case H_ARCTANH_:
  init_update_in_place(op.tensor);
  at::redispatch::arctanh_(ks, get<at::Tensor>(op.args[0]));
  end_update_in_place(op.tensor);
  break;

case H_ARCTANH_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::arctanh_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_AS_STRIDED:
  set(op.tensor, at::redispatch::as_strided(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<c10::optional<int64_t>>(op.args[3])));
  break;

case H_AS_STRIDED_:
  init_update_in_place(op.tensor);
  at::redispatch::as_strided_(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<c10::optional<int64_t>>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_ASIN:
  set(op.tensor, at::redispatch::asin(ks, get<at::Tensor>(op.args[0])));
  break;

case H_ASIN_:
  init_update_in_place(op.tensor);
  at::redispatch::asin_(ks, get<at::Tensor>(op.args[0]));
  end_update_in_place(op.tensor);
  break;

case H_ASIN_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::asin_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_ARCSIN:
  set(op.tensor, at::redispatch::arcsin(ks, get<at::Tensor>(op.args[0])));
  break;

case H_ARCSIN_:
  init_update_in_place(op.tensor);
  at::redispatch::arcsin_(ks, get<at::Tensor>(op.args[0]));
  end_update_in_place(op.tensor);
  break;

case H_ARCSIN_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::arcsin_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_ATAN_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::atan_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_ARCTAN:
  set(op.tensor, at::redispatch::arctan(ks, get<at::Tensor>(op.args[0])));
  break;

case H_ARCTAN_:
  init_update_in_place(op.tensor);
  at::redispatch::arctan_(ks, get<at::Tensor>(op.args[0]));
  end_update_in_place(op.tensor);
  break;

case H_ARCTAN_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::arctan_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_ATLEAST_1D:
  set(op.tensor, at::redispatch::atleast_1d(ks, get<at::Tensor>(op.args[0])));
  break;

// skip std::vector<at::Tensor> atleast_1d(at::TensorList tensors)

case H_ATLEAST_2D:
  set(op.tensor, at::redispatch::atleast_2d(ks, get<at::Tensor>(op.args[0])));
  break;

// skip std::vector<at::Tensor> atleast_2d(at::TensorList tensors)

case H_ATLEAST_3D:
  set(op.tensor, at::redispatch::atleast_3d(ks, get<at::Tensor>(op.args[0])));
  break;

// skip std::vector<at::Tensor> atleast_3d(at::TensorList tensors)

case H_BADDBMM:
  set(op.tensor, at::redispatch::baddbmm(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Scalar>(op.args[3]), get<at::Scalar>(op.args[4])));
  break;

case H_BADDBMM_:
  init_update_in_place(op.tensor);
  at::redispatch::baddbmm_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Scalar>(op.args[3]), get<at::Scalar>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H__BADDBMM_MKL_:
  init_update_in_place(op.tensor);
  at::redispatch::_baddbmm_mkl_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Scalar>(op.args[3]), get<at::Scalar>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_BADDBMM_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::baddbmm_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Scalar>(op.args[3]), get<at::Scalar>(op.args[4]), get<at::Tensor>(op.args[5]));
  end_update_in_place(op.tensor);
  break;

case H_BARTLETT_WINDOW:
  set(op.tensor, at::redispatch::bartlett_window(ks, get<int64_t>(op.args[0]), get<c10::optional<at::ScalarType>>(op.args[1]), get<c10::optional<at::Layout>>(op.args[2]), get<c10::optional<at::Device>>(op.args[3]), get<c10::optional<bool>>(op.args[4])));
  break;

case H_BARTLETT_WINDOW_PERIODIC:
  set(op.tensor, at::redispatch::bartlett_window(ks, get<int64_t>(op.args[0]), get<bool>(op.args[1]), get<c10::optional<at::ScalarType>>(op.args[2]), get<c10::optional<at::Layout>>(op.args[3]), get<c10::optional<at::Device>>(op.args[4]), get<c10::optional<bool>>(op.args[5])));
  break;

case H_BATCH_NORM:
  set(op.tensor, at::redispatch::batch_norm(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::Tensor>>(op.args[1]), get<c10::optional<at::Tensor>>(op.args[2]), get<c10::optional<at::Tensor>>(op.args[3]), get<c10::optional<at::Tensor>>(op.args[4]), get<bool>(op.args[5]), get<double>(op.args[6]), get<double>(op.args[7]), get<bool>(op.args[8])));
  break;

case H_QUANTIZED_BATCH_NORM:
  set(op.tensor, at::redispatch::quantized_batch_norm(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::Tensor>>(op.args[1]), get<c10::optional<at::Tensor>>(op.args[2]), get<at::Tensor>(op.args[3]), get<at::Tensor>(op.args[4]), get<double>(op.args[5]), get<double>(op.args[6]), get<int64_t>(op.args[7])));
  break;

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t> _batch_norm_impl_index(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, bool training, double momentum, double eps, bool cudnn_enabled)

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor> _batch_norm_impl_index_backward(int64_t impl_index, const at::Tensor & input, const at::Tensor & grad_output, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, const c10::optional<at::Tensor> & save_mean, const c10::optional<at::Tensor> & save_var_transform, bool train, double eps, std::array<bool,3> output_mask, const at::Tensor & reservedSpace)

case H_BERNOULLI:
  set(op.tensor, at::redispatch::bernoulli(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::Generator>>(op.args[1])));
  break;

case H_BERNOULLI_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::bernoulli_outf(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::Generator>>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_BERNOULLI__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::bernoulli_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<at::Generator>>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_BERNOULLI__FLOAT:
  init_update_in_place(op.tensor);
  at::redispatch::bernoulli_(ks, get<at::Tensor>(op.args[0]), get<double>(op.args[1]), get<c10::optional<at::Generator>>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_BERNOULLI_P:
  set(op.tensor, at::redispatch::bernoulli(ks, get<at::Tensor>(op.args[0]), get<double>(op.args[1]), get<c10::optional<at::Generator>>(op.args[2])));
  break;

case H_BILINEAR:
  set(op.tensor, at::redispatch::bilinear(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<c10::optional<at::Tensor>>(op.args[3])));
  break;

case H_BINARY_CROSS_ENTROPY:
  set(op.tensor, at::redispatch::binary_cross_entropy(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<at::Tensor>>(op.args[2]), get<int64_t>(op.args[3])));
  break;

case H_BINARY_CROSS_ENTROPY_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::binary_cross_entropy_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<at::Tensor>>(op.args[2]), get<int64_t>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_BINARY_CROSS_ENTROPY_BACKWARD:
  set(op.tensor, at::redispatch::binary_cross_entropy_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<c10::optional<at::Tensor>>(op.args[3]), get<int64_t>(op.args[4])));
  break;

case H_BINARY_CROSS_ENTROPY_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::binary_cross_entropy_backward_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<c10::optional<at::Tensor>>(op.args[3]), get<int64_t>(op.args[4]), get<at::Tensor>(op.args[5]));
  end_update_in_place(op.tensor);
  break;

case H_BINARY_CROSS_ENTROPY_WITH_LOGITS:
  set(op.tensor, at::redispatch::binary_cross_entropy_with_logits(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<at::Tensor>>(op.args[2]), get<c10::optional<at::Tensor>>(op.args[3]), get<int64_t>(op.args[4])));
  break;

case H_BINARY_CROSS_ENTROPY_WITH_LOGITS_BACKWARD:
  set(op.tensor, at::redispatch::binary_cross_entropy_with_logits_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<c10::optional<at::Tensor>>(op.args[3]), get<c10::optional<at::Tensor>>(op.args[4]), get<int64_t>(op.args[5])));
  break;

case H_BINCOUNT:
  set(op.tensor, at::redispatch::bincount(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::Tensor>>(op.args[1]), get<int64_t>(op.args[2])));
  break;

case H_BITWISE_NOT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::bitwise_not_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_COPYSIGN_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::copysign_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_COPYSIGN_SCALAR:
  set(op.tensor, at::redispatch::copysign(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1])));
  break;

case H_COPYSIGN__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::copysign_(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_COPYSIGN_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::copysign_outf(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_LOGICAL_NOT:
  set(op.tensor, at::redispatch::logical_not(ks, get<at::Tensor>(op.args[0])));
  break;

case H_LOGICAL_NOT_:
  init_update_in_place(op.tensor);
  at::redispatch::logical_not_(ks, get<at::Tensor>(op.args[0]));
  end_update_in_place(op.tensor);
  break;

case H_LOGICAL_NOT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::logical_not_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_LOGICAL_XOR:
  set(op.tensor, at::redispatch::logical_xor(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_LOGICAL_XOR_:
  init_update_in_place(op.tensor);
  at::redispatch::logical_xor_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_LOGICAL_XOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::logical_xor_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_LOGICAL_AND:
  set(op.tensor, at::redispatch::logical_and(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_LOGICAL_AND_:
  init_update_in_place(op.tensor);
  at::redispatch::logical_and_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_LOGICAL_AND_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::logical_and_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_LOGICAL_OR:
  set(op.tensor, at::redispatch::logical_or(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_LOGICAL_OR_:
  init_update_in_place(op.tensor);
  at::redispatch::logical_or_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_LOGICAL_OR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::logical_or_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_BLACKMAN_WINDOW:
  set(op.tensor, at::redispatch::blackman_window(ks, get<int64_t>(op.args[0]), get<c10::optional<at::ScalarType>>(op.args[1]), get<c10::optional<at::Layout>>(op.args[2]), get<c10::optional<at::Device>>(op.args[3]), get<c10::optional<bool>>(op.args[4])));
  break;

case H_BLACKMAN_WINDOW_PERIODIC:
  set(op.tensor, at::redispatch::blackman_window(ks, get<int64_t>(op.args[0]), get<bool>(op.args[1]), get<c10::optional<at::ScalarType>>(op.args[2]), get<c10::optional<at::Layout>>(op.args[3]), get<c10::optional<at::Device>>(op.args[4]), get<c10::optional<bool>>(op.args[5])));
  break;

case H_BMM:
  set(op.tensor, at::redispatch::bmm(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H__BMM:
  set(op.tensor, at::redispatch::_bmm(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<bool>(op.args[2])));
  break;

case H_BMM_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::bmm_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H__BMM_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::_bmm_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<bool>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

// skip std::vector<at::Tensor> broadcast_tensors(at::TensorList tensors)

case H_BROADCAST_TO:
  set(op.tensor, at::redispatch::broadcast_to(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1])));
  break;

case H_CAT:
  set(op.tensor, at::redispatch::cat(ks, get<at::TensorList>(op.args[0]), get<int64_t>(op.args[1])));
  break;

case H_CAT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::cat_outf(ks, get<at::TensorList>(op.args[0]), get<int64_t>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_CAT_NAMES:
  set(op.tensor, at::redispatch::cat(ks, get<at::TensorList>(op.args[0]), get<at::Dimname>(op.args[1])));
  break;

case H_CAT_NAMES_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::cat_outf(ks, get<at::TensorList>(op.args[0]), get<at::Dimname>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_BLOCK_DIAG:
  set(op.tensor, at::redispatch::block_diag(ks, get<at::TensorList>(op.args[0])));
  break;

case H_CEIL:
  set(op.tensor, at::redispatch::ceil(ks, get<at::Tensor>(op.args[0])));
  break;

case H_CEIL_:
  init_update_in_place(op.tensor);
  at::redispatch::ceil_(ks, get<at::Tensor>(op.args[0]));
  end_update_in_place(op.tensor);
  break;

case H_CEIL_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::ceil_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_CHAIN_MATMUL:
  set(op.tensor, at::redispatch::chain_matmul(ks, get<at::TensorList>(op.args[0])));
  break;

case H_CHAIN_MATMUL_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::chain_matmul_outf(ks, get<at::TensorList>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

// skip std::vector<at::Tensor> unsafe_chunk(const at::Tensor & self, int64_t chunks, int64_t dim)

// skip std::vector<at::Tensor> chunk(const at::Tensor & self, int64_t chunks, int64_t dim)

// skip std::vector<at::Tensor> tensor_split(const at::Tensor & self, int64_t sections, int64_t dim)

// skip std::vector<at::Tensor> tensor_split(const at::Tensor & self, at::IntArrayRef indices, int64_t dim)

// skip std::vector<at::Tensor> tensor_split(const at::Tensor & self, const at::Tensor & tensor_indices_or_sections, int64_t dim)

case H_CLAMP:
  set(op.tensor, at::redispatch::clamp(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::Scalar>>(op.args[1]), get<c10::optional<at::Scalar>>(op.args[2])));
  break;

case H_CLAMP_TENSOR:
  set(op.tensor, at::redispatch::clamp(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::Tensor>>(op.args[1]), get<c10::optional<at::Tensor>>(op.args[2])));
  break;

case H_CLAMP_:
  init_update_in_place(op.tensor);
  at::redispatch::clamp_(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::Scalar>>(op.args[1]), get<c10::optional<at::Scalar>>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_CLAMP__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::clamp_(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::Tensor>>(op.args[1]), get<c10::optional<at::Tensor>>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_CLAMP_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::clamp_outf(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::Scalar>>(op.args[1]), get<c10::optional<at::Scalar>>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_CLAMP_TENSOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::clamp_outf(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::Tensor>>(op.args[1]), get<c10::optional<at::Tensor>>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_CLAMP_MAX:
  set(op.tensor, at::redispatch::clamp_max(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1])));
  break;

case H_CLAMP_MAX_TENSOR:
  set(op.tensor, at::redispatch::clamp_max(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_CLAMP_MAX_:
  init_update_in_place(op.tensor);
  at::redispatch::clamp_max_(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_CLAMP_MAX__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::clamp_max_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_CLAMP_MAX_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::clamp_max_outf(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_CLAMP_MAX_TENSOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::clamp_max_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_CLAMP_MIN:
  set(op.tensor, at::redispatch::clamp_min(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1])));
  break;

case H_CLAMP_MIN_TENSOR:
  set(op.tensor, at::redispatch::clamp_min(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_CLAMP_MIN_:
  init_update_in_place(op.tensor);
  at::redispatch::clamp_min_(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_CLAMP_MIN__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::clamp_min_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_CLAMP_MIN_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::clamp_min_outf(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_CLAMP_MIN_TENSOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::clamp_min_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_CLIP:
  set(op.tensor, at::redispatch::clip(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::Scalar>>(op.args[1]), get<c10::optional<at::Scalar>>(op.args[2])));
  break;

case H_CLIP_TENSOR:
  set(op.tensor, at::redispatch::clip(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::Tensor>>(op.args[1]), get<c10::optional<at::Tensor>>(op.args[2])));
  break;

case H_CLIP_:
  init_update_in_place(op.tensor);
  at::redispatch::clip_(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::Scalar>>(op.args[1]), get<c10::optional<at::Scalar>>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_CLIP__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::clip_(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::Tensor>>(op.args[1]), get<c10::optional<at::Tensor>>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_CLIP_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::clip_outf(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::Scalar>>(op.args[1]), get<c10::optional<at::Scalar>>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_CLIP_TENSOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::clip_outf(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::Tensor>>(op.args[1]), get<c10::optional<at::Tensor>>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

// skip bool cudnn_is_acceptable(const at::Tensor & self)

case H_COMPLEX:
  set(op.tensor, at::redispatch::complex(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_COMPLEX_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::complex_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_POLAR:
  set(op.tensor, at::redispatch::polar(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_POLAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::polar_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_CONSTANT_PAD_ND:
  set(op.tensor, at::redispatch::constant_pad_nd(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<at::Scalar>(op.args[2])));
  break;

case H_CONVOLUTION:
  set(op.tensor, at::redispatch::convolution(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<at::Tensor>>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<bool>(op.args[6]), get<at::IntArrayRef>(op.args[7]), get<int64_t>(op.args[8])));
  break;

case H_CONVOLUTION_OVERRIDEABLE:
  set(op.tensor, at::redispatch::convolution_overrideable(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<at::Tensor>>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<bool>(op.args[6]), get<at::IntArrayRef>(op.args[7]), get<int64_t>(op.args[8])));
  break;

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor> convolution_backward_overrideable(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & weight, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups, std::array<bool,3> output_mask)

case H__CONVOLUTION:
  set(op.tensor, at::redispatch::_convolution(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<at::Tensor>>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<bool>(op.args[6]), get<at::IntArrayRef>(op.args[7]), get<int64_t>(op.args[8]), get<bool>(op.args[9]), get<bool>(op.args[10]), get<bool>(op.args[11]), get<bool>(op.args[12])));
  break;

case H__CONVOLUTION_DEPRECATED:
  set(op.tensor, at::redispatch::_convolution(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<at::Tensor>>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<bool>(op.args[6]), get<at::IntArrayRef>(op.args[7]), get<int64_t>(op.args[8]), get<bool>(op.args[9]), get<bool>(op.args[10]), get<bool>(op.args[11])));
  break;

case H__CONVOLUTION_MODE:
  set(op.tensor, at::redispatch::_convolution_mode(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<at::Tensor>>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<std::string>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<int64_t>(op.args[6])));
  break;

case H__CONVOLUTION_NOGROUP:
  set(op.tensor, at::redispatch::_convolution_nogroup(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<at::Tensor>>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<bool>(op.args[6]), get<at::IntArrayRef>(op.args[7])));
  break;

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor> _convolution_double_backward(const c10::optional<at::Tensor> & ggI, const c10::optional<at::Tensor> & ggW, const c10::optional<at::Tensor> & ggb, const at::Tensor & gO, const at::Tensor & weight, const at::Tensor & self, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32, std::array<bool,3> output_mask)

case H_CONV1D:
  set(op.tensor, at::redispatch::conv1d(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<at::Tensor>>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<int64_t>(op.args[6])));
  break;

case H_CONV2D:
  set(op.tensor, at::redispatch::conv2d(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<at::Tensor>>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<int64_t>(op.args[6])));
  break;

case H_CONV3D:
  set(op.tensor, at::redispatch::conv3d(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<at::Tensor>>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<int64_t>(op.args[6])));
  break;

case H_CONV1D_PADDING:
  set(op.tensor, at::redispatch::conv1d(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<at::Tensor>>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<std::string>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<int64_t>(op.args[6])));
  break;

case H_CONV2D_PADDING:
  set(op.tensor, at::redispatch::conv2d(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<at::Tensor>>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<std::string>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<int64_t>(op.args[6])));
  break;

case H_CONV3D_PADDING:
  set(op.tensor, at::redispatch::conv3d(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<at::Tensor>>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<std::string>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<int64_t>(op.args[6])));
  break;

case H_CONV_TBC:
  set(op.tensor, at::redispatch::conv_tbc(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<int64_t>(op.args[3])));
  break;

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor> conv_tbc_backward(const at::Tensor & self, const at::Tensor & input, const at::Tensor & weight, const at::Tensor & bias, int64_t pad)

case H_CONV_TRANSPOSE1D:
  set(op.tensor, at::redispatch::conv_transpose1d(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<at::Tensor>>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<int64_t>(op.args[6]), get<at::IntArrayRef>(op.args[7])));
  break;

case H_CONV_TRANSPOSE2D_INPUT:
  set(op.tensor, at::redispatch::conv_transpose2d(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<at::Tensor>>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<int64_t>(op.args[6]), get<at::IntArrayRef>(op.args[7])));
  break;

case H_CONV_TRANSPOSE3D_INPUT:
  set(op.tensor, at::redispatch::conv_transpose3d(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<at::Tensor>>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<int64_t>(op.args[6]), get<at::IntArrayRef>(op.args[7])));
  break;

case H_COPY_:
  init_update_in_place(op.tensor);
  at::redispatch::copy_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<bool>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_COS_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::cos_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_COSH_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::cosh_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_COSINE_EMBEDDING_LOSS:
  set(op.tensor, at::redispatch::cosine_embedding_loss(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<double>(op.args[3]), get<int64_t>(op.args[4])));
  break;

case H_COUNT_NONZERO_DIM_INTLIST:
  set(op.tensor, at::redispatch::count_nonzero(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1])));
  break;

case H_COUNT_NONZERO:
  set(op.tensor, at::redispatch::count_nonzero(ks, get<at::Tensor>(op.args[0]), get<c10::optional<int64_t>>(op.args[1])));
  break;

case H_CUDNN_AFFINE_GRID_GENERATOR:
  set(op.tensor, at::redispatch::cudnn_affine_grid_generator(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<int64_t>(op.args[2]), get<int64_t>(op.args[3]), get<int64_t>(op.args[4])));
  break;

case H_CUDNN_AFFINE_GRID_GENERATOR_BACKWARD:
  set(op.tensor, at::redispatch::cudnn_affine_grid_generator_backward(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<int64_t>(op.args[2]), get<int64_t>(op.args[3]), get<int64_t>(op.args[4])));
  break;

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> cudnn_batch_norm(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, bool training, double exponential_average_factor, double epsilon)

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor> cudnn_batch_norm_backward(const at::Tensor & input, const at::Tensor & grad_output, const at::Tensor & weight, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, const c10::optional<at::Tensor> & save_mean, const c10::optional<at::Tensor> & save_var, double epsilon, const at::Tensor & reserveSpace)

case H_CUDNN_CONVOLUTION_DEPRECATED:
  set(op.tensor, at::redispatch::cudnn_convolution(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<at::Tensor>>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<int64_t>(op.args[6]), get<bool>(op.args[7]), get<bool>(op.args[8])));
  break;

case H_CUDNN_CONVOLUTION_DEPRECATED2:
  set(op.tensor, at::redispatch::cudnn_convolution(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<int64_t>(op.args[5]), get<bool>(op.args[6]), get<bool>(op.args[7])));
  break;

case H_CUDNN_CONVOLUTION:
  set(op.tensor, at::redispatch::cudnn_convolution(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<int64_t>(op.args[5]), get<bool>(op.args[6]), get<bool>(op.args[7]), get<bool>(op.args[8])));
  break;

case H_CUDNN_CONVOLUTION_BACKWARD_INPUT:
  set(op.tensor, at::redispatch::cudnn_convolution_backward_input(ks, get<at::IntArrayRef>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<int64_t>(op.args[6]), get<bool>(op.args[7]), get<bool>(op.args[8]), get<bool>(op.args[9])));
  break;

// skip std::tuple<at::Tensor,at::Tensor> cudnn_convolution_backward(const at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, bool allow_tf32, std::array<bool,2> output_mask)

case H_CUDNN_CONVOLUTION_BACKWARD_WEIGHT:
  set(op.tensor, at::redispatch::cudnn_convolution_backward_weight(ks, get<at::IntArrayRef>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<int64_t>(op.args[6]), get<bool>(op.args[7]), get<bool>(op.args[8]), get<bool>(op.args[9])));
  break;

case H_CUDNN_CONVOLUTION_TRANSPOSE_DEPRECATED:
  set(op.tensor, at::redispatch::cudnn_convolution_transpose(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<at::Tensor>>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<at::IntArrayRef>(op.args[6]), get<int64_t>(op.args[7]), get<bool>(op.args[8]), get<bool>(op.args[9])));
  break;

case H_CUDNN_CONVOLUTION_TRANSPOSE_DEPRECATED2:
  set(op.tensor, at::redispatch::cudnn_convolution_transpose(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<int64_t>(op.args[6]), get<bool>(op.args[7]), get<bool>(op.args[8])));
  break;

case H_CUDNN_CONVOLUTION_TRANSPOSE:
  set(op.tensor, at::redispatch::cudnn_convolution_transpose(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<int64_t>(op.args[6]), get<bool>(op.args[7]), get<bool>(op.args[8]), get<bool>(op.args[9])));
  break;

// skip std::tuple<at::Tensor,at::Tensor> cudnn_convolution_transpose_backward(const at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, bool allow_tf32, std::array<bool,2> output_mask)

case H_CUDNN_CONVOLUTION_TRANSPOSE_BACKWARD_INPUT:
  set(op.tensor, at::redispatch::cudnn_convolution_transpose_backward_input(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<int64_t>(op.args[5]), get<bool>(op.args[6]), get<bool>(op.args[7]), get<bool>(op.args[8])));
  break;

case H_CUDNN_CONVOLUTION_TRANSPOSE_BACKWARD_WEIGHT:
  set(op.tensor, at::redispatch::cudnn_convolution_transpose_backward_weight(ks, get<at::IntArrayRef>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<int64_t>(op.args[6]), get<bool>(op.args[7]), get<bool>(op.args[8]), get<bool>(op.args[9])));
  break;

case H_CUDNN_CONVOLUTION_RELU:
  set(op.tensor, at::redispatch::cudnn_convolution_relu(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<at::Tensor>>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<int64_t>(op.args[6])));
  break;

case H_CUDNN_CONVOLUTION_ADD_RELU:
  set(op.tensor, at::redispatch::cudnn_convolution_add_relu(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<c10::optional<at::Scalar>>(op.args[3]), get<c10::optional<at::Tensor>>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<at::IntArrayRef>(op.args[6]), get<at::IntArrayRef>(op.args[7]), get<int64_t>(op.args[8])));
  break;

case H_CUDNN_GRID_SAMPLER:
  set(op.tensor, at::redispatch::cudnn_grid_sampler(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

// skip std::tuple<at::Tensor,at::Tensor> cudnn_grid_sampler_backward(const at::Tensor & self, const at::Tensor & grid, const at::Tensor & grad_output)

// skip std::tuple<at::Tensor,at::Tensor> cummax(const at::Tensor & self, int64_t dim)

// skip std::tuple<at::Tensor &,at::Tensor &> cummax_outf(const at::Tensor & self, int64_t dim, at::Tensor & values, at::Tensor & indices)

// skip std::tuple<at::Tensor,at::Tensor> cummax(const at::Tensor & self, at::Dimname dim)

// skip std::tuple<at::Tensor &,at::Tensor &> cummax_outf(const at::Tensor & self, at::Dimname dim, at::Tensor & values, at::Tensor & indices)

// skip void _cummax_helper(const at::Tensor & self, at::Tensor & values, at::Tensor & indices, int64_t dim)

// skip std::tuple<at::Tensor,at::Tensor> cummin(const at::Tensor & self, int64_t dim)

// skip std::tuple<at::Tensor &,at::Tensor &> cummin_outf(const at::Tensor & self, int64_t dim, at::Tensor & values, at::Tensor & indices)

// skip std::tuple<at::Tensor,at::Tensor> cummin(const at::Tensor & self, at::Dimname dim)

// skip std::tuple<at::Tensor &,at::Tensor &> cummin_outf(const at::Tensor & self, at::Dimname dim, at::Tensor & values, at::Tensor & indices)

// skip void _cummin_helper(const at::Tensor & self, at::Tensor & values, at::Tensor & indices, int64_t dim)

case H_CUMMAXMIN_BACKWARD:
  set(op.tensor, at::redispatch::cummaxmin_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<int64_t>(op.args[3])));
  break;

case H_CUMPROD:
  set(op.tensor, at::redispatch::cumprod(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<c10::optional<at::ScalarType>>(op.args[2])));
  break;

case H_CUMPROD_:
  init_update_in_place(op.tensor);
  at::redispatch::cumprod_(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<c10::optional<at::ScalarType>>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_CUMPROD_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::cumprod_outf(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<c10::optional<at::ScalarType>>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_CUMPROD_DIMNAME:
  set(op.tensor, at::redispatch::cumprod(ks, get<at::Tensor>(op.args[0]), get<at::Dimname>(op.args[1]), get<c10::optional<at::ScalarType>>(op.args[2])));
  break;

case H_CUMPROD__DIMNAME:
  init_update_in_place(op.tensor);
  at::redispatch::cumprod_(ks, get<at::Tensor>(op.args[0]), get<at::Dimname>(op.args[1]), get<c10::optional<at::ScalarType>>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_CUMPROD_DIMNAME_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::cumprod_outf(ks, get<at::Tensor>(op.args[0]), get<at::Dimname>(op.args[1]), get<c10::optional<at::ScalarType>>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_CUMPROD_BACKWARD:
  set(op.tensor, at::redispatch::cumprod_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<int64_t>(op.args[2]), get<at::Tensor>(op.args[3])));
  break;

case H_CUMSUM:
  set(op.tensor, at::redispatch::cumsum(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<c10::optional<at::ScalarType>>(op.args[2])));
  break;

case H_CUMSUM_:
  init_update_in_place(op.tensor);
  at::redispatch::cumsum_(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<c10::optional<at::ScalarType>>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_CUMSUM_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::cumsum_outf(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<c10::optional<at::ScalarType>>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_CUMSUM_DIMNAME:
  set(op.tensor, at::redispatch::cumsum(ks, get<at::Tensor>(op.args[0]), get<at::Dimname>(op.args[1]), get<c10::optional<at::ScalarType>>(op.args[2])));
  break;

case H_CUMSUM__DIMNAME:
  init_update_in_place(op.tensor);
  at::redispatch::cumsum_(ks, get<at::Tensor>(op.args[0]), get<at::Dimname>(op.args[1]), get<c10::optional<at::ScalarType>>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_CUMSUM_DIMNAME_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::cumsum_outf(ks, get<at::Tensor>(op.args[0]), get<at::Dimname>(op.args[1]), get<c10::optional<at::ScalarType>>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_CTC_LOSS_INTLIST:
  set(op.tensor, at::redispatch::ctc_loss(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<int64_t>(op.args[4]), get<int64_t>(op.args[5]), get<bool>(op.args[6])));
  break;

case H_CTC_LOSS_TENSOR:
  set(op.tensor, at::redispatch::ctc_loss(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Tensor>(op.args[3]), get<int64_t>(op.args[4]), get<int64_t>(op.args[5]), get<bool>(op.args[6])));
  break;

// skip std::tuple<at::Tensor,at::Tensor> _ctc_loss(const at::Tensor & log_probs, const at::Tensor & targets, at::IntArrayRef input_lengths, at::IntArrayRef target_lengths, int64_t blank, bool zero_infinity)

case H__CTC_LOSS_BACKWARD:
  set(op.tensor, at::redispatch::_ctc_loss_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::Tensor>(op.args[5]), get<at::Tensor>(op.args[6]), get<int64_t>(op.args[7]), get<bool>(op.args[8])));
  break;

case H_DIAG_EMBED:
  set(op.tensor, at::redispatch::diag_embed(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<int64_t>(op.args[2]), get<int64_t>(op.args[3])));
  break;

case H_DIAGFLAT:
  set(op.tensor, at::redispatch::diagflat(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1])));
  break;

case H_DIAGONAL:
  set(op.tensor, at::redispatch::diagonal(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<int64_t>(op.args[2]), get<int64_t>(op.args[3])));
  break;

case H_DIAGONAL_DIMNAME:
  set(op.tensor, at::redispatch::diagonal(ks, get<at::Tensor>(op.args[0]), get<at::Dimname>(op.args[1]), get<at::Dimname>(op.args[2]), get<at::Dimname>(op.args[3]), get<int64_t>(op.args[4])));
  break;

case H_DIAGONAL_BACKWARD:
  set(op.tensor, at::redispatch::diagonal_backward(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<int64_t>(op.args[2]), get<int64_t>(op.args[3]), get<int64_t>(op.args[4])));
  break;

case H_FILL_DIAGONAL_:
  init_update_in_place(op.tensor);
  at::redispatch::fill_diagonal_(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<bool>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_DIFF:
  set(op.tensor, at::redispatch::diff(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<int64_t>(op.args[2]), get<c10::optional<at::Tensor>>(op.args[3]), get<c10::optional<at::Tensor>>(op.args[4])));
  break;

case H_DIFF_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::diff_outf(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<int64_t>(op.args[2]), get<c10::optional<at::Tensor>>(op.args[3]), get<c10::optional<at::Tensor>>(op.args[4]), get<at::Tensor>(op.args[5]));
  end_update_in_place(op.tensor);
  break;

case H_DIV_TENSOR:
  set(op.tensor, at::redispatch::div(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_DIV__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::div_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_DIV_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::div_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_DIV_TENSOR_MODE:
  set(op.tensor, at::redispatch::div(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<std::string>>(op.args[2])));
  break;

case H_DIV__TENSOR_MODE:
  init_update_in_place(op.tensor);
  at::redispatch::div_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<std::string>>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_DIV_OUT_MODE:
  init_update_in_place(op.tensor);
  at::redispatch::div_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<std::string>>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_DIV_SCALAR:
  set(op.tensor, at::redispatch::div(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1])));
  break;

case H_DIV__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::div_(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_DIV_SCALAR_MODE:
  set(op.tensor, at::redispatch::div(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<c10::optional<std::string>>(op.args[2])));
  break;

case H_DIV__SCALAR_MODE:
  init_update_in_place(op.tensor);
  at::redispatch::div_(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<c10::optional<std::string>>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_DIVIDE_TENSOR:
  set(op.tensor, at::redispatch::divide(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_DIVIDE__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::divide_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_DIVIDE_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::divide_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_DIVIDE_SCALAR:
  set(op.tensor, at::redispatch::divide(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1])));
  break;

case H_DIVIDE__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::divide_(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_DIVIDE_TENSOR_MODE:
  set(op.tensor, at::redispatch::divide(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<std::string>>(op.args[2])));
  break;

case H_DIVIDE__TENSOR_MODE:
  init_update_in_place(op.tensor);
  at::redispatch::divide_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<std::string>>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_DIVIDE_OUT_MODE:
  init_update_in_place(op.tensor);
  at::redispatch::divide_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<std::string>>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_DIVIDE_SCALAR_MODE:
  set(op.tensor, at::redispatch::divide(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<c10::optional<std::string>>(op.args[2])));
  break;

case H_DIVIDE__SCALAR_MODE:
  init_update_in_place(op.tensor);
  at::redispatch::divide_(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<c10::optional<std::string>>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_TRUE_DIVIDE_TENSOR:
  set(op.tensor, at::redispatch::true_divide(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_TRUE_DIVIDE__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::true_divide_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_TRUE_DIVIDE_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::true_divide_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_TRUE_DIVIDE_SCALAR:
  set(op.tensor, at::redispatch::true_divide(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1])));
  break;

case H_TRUE_DIVIDE__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::true_divide_(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_DOT:
  set(op.tensor, at::redispatch::dot(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_DOT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::dot_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_VDOT:
  set(op.tensor, at::redispatch::vdot(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_VDOT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::vdot_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_EINSUM:
  set(op.tensor, at::redispatch::einsum(ks, get<std::string>(op.args[0]), get<at::TensorList>(op.args[1])));
  break;

case H_EMBEDDING:
  set(op.tensor, at::redispatch::embedding(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<int64_t>(op.args[2]), get<bool>(op.args[3]), get<bool>(op.args[4])));
  break;

case H_EMBEDDING_BACKWARD:
  set(op.tensor, at::redispatch::embedding_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<int64_t>(op.args[2]), get<int64_t>(op.args[3]), get<bool>(op.args[4]), get<bool>(op.args[5])));
  break;

case H_EMBEDDING_DENSE_BACKWARD:
  set(op.tensor, at::redispatch::embedding_dense_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<int64_t>(op.args[2]), get<int64_t>(op.args[3]), get<bool>(op.args[4])));
  break;

case H_EMBEDDING_RENORM_:
  init_update_in_place(op.tensor);
  at::redispatch::embedding_renorm_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<double>(op.args[2]), get<double>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_EMBEDDING_SPARSE_BACKWARD:
  set(op.tensor, at::redispatch::embedding_sparse_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<int64_t>(op.args[2]), get<int64_t>(op.args[3]), get<bool>(op.args[4])));
  break;

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> _embedding_bag_forward_only(const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const c10::optional<at::Tensor> & per_sample_weights, bool include_last_offset, int64_t padding_idx)

// skip std::tuple<at::Tensor,at::Tensor> _rowwise_prune(const at::Tensor & weight, const at::Tensor & mask, at::ScalarType compressed_indices_dtype)

case H_ROW_STACK:
  set(op.tensor, at::redispatch::row_stack(ks, get<at::TensorList>(op.args[0])));
  break;

case H_ROW_STACK_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::row_stack_outf(ks, get<at::TensorList>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> embedding_bag(const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const c10::optional<at::Tensor> & per_sample_weights, bool include_last_offset)

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> embedding_bag(const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const c10::optional<at::Tensor> & per_sample_weights, bool include_last_offset, c10::optional<int64_t> padding_idx)

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> _embedding_bag(const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const c10::optional<at::Tensor> & per_sample_weights, bool include_last_offset, int64_t padding_idx)

case H__EMBEDDING_BAG_BACKWARD:
  set(op.tensor, at::redispatch::_embedding_bag_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Tensor>(op.args[3]), get<at::Tensor>(op.args[4]), get<at::Tensor>(op.args[5]), get<int64_t>(op.args[6]), get<bool>(op.args[7]), get<int64_t>(op.args[8]), get<bool>(op.args[9]), get<c10::optional<at::Tensor>>(op.args[10]), get<int64_t>(op.args[11])));
  break;

case H__EMBEDDING_BAG_SPARSE_BACKWARD:
  set(op.tensor, at::redispatch::_embedding_bag_sparse_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Tensor>(op.args[3]), get<at::Tensor>(op.args[4]), get<int64_t>(op.args[5]), get<bool>(op.args[6]), get<int64_t>(op.args[7]), get<c10::optional<at::Tensor>>(op.args[8]), get<int64_t>(op.args[9])));
  break;

case H__EMBEDDING_BAG_DENSE_BACKWARD:
  set(op.tensor, at::redispatch::_embedding_bag_dense_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Tensor>(op.args[3]), get<at::Tensor>(op.args[4]), get<int64_t>(op.args[5]), get<bool>(op.args[6]), get<int64_t>(op.args[7]), get<c10::optional<at::Tensor>>(op.args[8]), get<int64_t>(op.args[9])));
  break;

case H__EMBEDDING_BAG_PER_SAMPLE_WEIGHTS_BACKWARD:
  set(op.tensor, at::redispatch::_embedding_bag_per_sample_weights_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Tensor>(op.args[3]), get<at::Tensor>(op.args[4]), get<int64_t>(op.args[5]), get<int64_t>(op.args[6])));
  break;

case H_EMPTY_NAMES:
  set(op.tensor, at::redispatch::empty(ks, get<at::IntArrayRef>(op.args[0]), get<c10::optional<at::DimnameList>>(op.args[1]), get<c10::optional<at::ScalarType>>(op.args[2]), get<c10::optional<at::Layout>>(op.args[3]), get<c10::optional<at::Device>>(op.args[4]), get<c10::optional<bool>>(op.args[5]), get<c10::optional<at::MemoryFormat>>(op.args[6])));
  break;

case H_EMPTY_MEMORY_FORMAT:
  set(op.tensor, at::redispatch::empty(ks, get<at::IntArrayRef>(op.args[0]), get<c10::optional<at::ScalarType>>(op.args[1]), get<c10::optional<at::Layout>>(op.args[2]), get<c10::optional<at::Device>>(op.args[3]), get<c10::optional<bool>>(op.args[4]), get<c10::optional<at::MemoryFormat>>(op.args[5])));
  break;

case H_NEW_EMPTY:
  set(op.tensor, at::redispatch::new_empty(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<c10::optional<at::ScalarType>>(op.args[2]), get<c10::optional<at::Layout>>(op.args[3]), get<c10::optional<at::Device>>(op.args[4]), get<c10::optional<bool>>(op.args[5])));
  break;

case H_NEW_EMPTY_STRIDED:
  set(op.tensor, at::redispatch::new_empty_strided(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<c10::optional<at::ScalarType>>(op.args[3]), get<c10::optional<at::Layout>>(op.args[4]), get<c10::optional<at::Device>>(op.args[5]), get<c10::optional<bool>>(op.args[6])));
  break;

case H_NEW_FULL:
  set(op.tensor, at::redispatch::new_full(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<at::Scalar>(op.args[2]), get<c10::optional<at::ScalarType>>(op.args[3]), get<c10::optional<at::Layout>>(op.args[4]), get<c10::optional<at::Device>>(op.args[5]), get<c10::optional<bool>>(op.args[6])));
  break;

case H_NEW_ZEROS:
  set(op.tensor, at::redispatch::new_zeros(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<c10::optional<at::ScalarType>>(op.args[2]), get<c10::optional<at::Layout>>(op.args[3]), get<c10::optional<at::Device>>(op.args[4]), get<c10::optional<bool>>(op.args[5])));
  break;

case H__EMPTY_AFFINE_QUANTIZED:
  set(op.tensor, at::redispatch::_empty_affine_quantized(ks, get<at::IntArrayRef>(op.args[0]), get<c10::optional<at::ScalarType>>(op.args[1]), get<c10::optional<at::Layout>>(op.args[2]), get<c10::optional<at::Device>>(op.args[3]), get<c10::optional<bool>>(op.args[4]), get<double>(op.args[5]), get<int64_t>(op.args[6]), get<c10::optional<at::MemoryFormat>>(op.args[7])));
  break;

case H__EMPTY_PER_CHANNEL_AFFINE_QUANTIZED:
  set(op.tensor, at::redispatch::_empty_per_channel_affine_quantized(ks, get<at::IntArrayRef>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<int64_t>(op.args[3]), get<c10::optional<at::ScalarType>>(op.args[4]), get<c10::optional<at::Layout>>(op.args[5]), get<c10::optional<at::Device>>(op.args[6]), get<c10::optional<bool>>(op.args[7]), get<c10::optional<at::MemoryFormat>>(op.args[8])));
  break;

case H_RESIZE_:
  init_update_in_place(op.tensor);
  at::redispatch::resize_(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<c10::optional<at::MemoryFormat>>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_EMPTY_QUANTIZED:
  set(op.tensor, at::redispatch::empty_quantized(ks, get<at::IntArrayRef>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_EMPTY_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::empty_outf(ks, get<at::IntArrayRef>(op.args[0]), get<c10::optional<at::MemoryFormat>>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_EMPTY_LIKE:
  set(op.tensor, at::redispatch::empty_like(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::ScalarType>>(op.args[1]), get<c10::optional<at::Layout>>(op.args[2]), get<c10::optional<at::Device>>(op.args[3]), get<c10::optional<bool>>(op.args[4]), get<c10::optional<at::MemoryFormat>>(op.args[5])));
  break;

case H_EMPTY_STRIDED:
  set(op.tensor, at::redispatch::empty_strided(ks, get<at::IntArrayRef>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<c10::optional<at::ScalarType>>(op.args[2]), get<c10::optional<at::Layout>>(op.args[3]), get<c10::optional<at::Device>>(op.args[4]), get<c10::optional<bool>>(op.args[5])));
  break;

case H_ERF_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::erf_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_ERFC_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::erfc_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_EXP_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::exp_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_EXP2_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::exp2_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_EXPM1_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::expm1_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_EXPAND:
  set(op.tensor, at::redispatch::expand(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<bool>(op.args[2])));
  break;

case H_EXPAND_AS:
  set(op.tensor, at::redispatch::expand_as(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_EYE:
  set(op.tensor, at::redispatch::eye(ks, get<int64_t>(op.args[0]), get<c10::optional<at::ScalarType>>(op.args[1]), get<c10::optional<at::Layout>>(op.args[2]), get<c10::optional<at::Device>>(op.args[3]), get<c10::optional<bool>>(op.args[4])));
  break;

case H_EYE_M:
  set(op.tensor, at::redispatch::eye(ks, get<int64_t>(op.args[0]), get<int64_t>(op.args[1]), get<c10::optional<at::ScalarType>>(op.args[2]), get<c10::optional<at::Layout>>(op.args[3]), get<c10::optional<at::Device>>(op.args[4]), get<c10::optional<bool>>(op.args[5])));
  break;

case H_EYE_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::eye_outf(ks, get<int64_t>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_EYE_M_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::eye_outf(ks, get<int64_t>(op.args[0]), get<int64_t>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_FLATTEN_USING_INTS:
  set(op.tensor, at::redispatch::flatten(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<int64_t>(op.args[2])));
  break;

case H_FLATTEN_NAMED_OUT_DIM:
  set(op.tensor, at::redispatch::flatten(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<int64_t>(op.args[2]), get<at::Dimname>(op.args[3])));
  break;

case H_FLATTEN_USING_NAMES:
  set(op.tensor, at::redispatch::flatten(ks, get<at::Tensor>(op.args[0]), get<at::Dimname>(op.args[1]), get<at::Dimname>(op.args[2]), get<at::Dimname>(op.args[3])));
  break;

case H_FLATTEN_DIMNAMELIST:
  set(op.tensor, at::redispatch::flatten(ks, get<at::Tensor>(op.args[0]), get<at::DimnameList>(op.args[1]), get<at::Dimname>(op.args[2])));
  break;

case H_UNFLATTEN_INT:
  set(op.tensor, at::redispatch::unflatten(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<c10::optional<at::DimnameList>>(op.args[3])));
  break;

case H_UNFLATTEN_DIMNAME:
  set(op.tensor, at::redispatch::unflatten(ks, get<at::Tensor>(op.args[0]), get<at::Dimname>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::DimnameList>(op.args[3])));
  break;

case H_FILL__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::fill_(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_FILL__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::fill_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_FLOOR:
  set(op.tensor, at::redispatch::floor(ks, get<at::Tensor>(op.args[0])));
  break;

case H_FLOOR_:
  init_update_in_place(op.tensor);
  at::redispatch::floor_(ks, get<at::Tensor>(op.args[0]));
  end_update_in_place(op.tensor);
  break;

case H_FLOOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::floor_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_FLOOR_DIVIDE:
  set(op.tensor, at::redispatch::floor_divide(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_FLOOR_DIVIDE__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::floor_divide_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_FLOOR_DIVIDE_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::floor_divide_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_FLOOR_DIVIDE_SCALAR:
  set(op.tensor, at::redispatch::floor_divide(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1])));
  break;

case H_FLOOR_DIVIDE__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::floor_divide_(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_FRAC_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::frac_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_FULL_NAMES:
  set(op.tensor, at::redispatch::full(ks, get<at::IntArrayRef>(op.args[0]), get<at::Scalar>(op.args[1]), get<c10::optional<at::DimnameList>>(op.args[2]), get<c10::optional<at::ScalarType>>(op.args[3]), get<c10::optional<at::Layout>>(op.args[4]), get<c10::optional<at::Device>>(op.args[5]), get<c10::optional<bool>>(op.args[6])));
  break;

case H_FULL:
  set(op.tensor, at::redispatch::full(ks, get<at::IntArrayRef>(op.args[0]), get<at::Scalar>(op.args[1]), get<c10::optional<at::ScalarType>>(op.args[2]), get<c10::optional<at::Layout>>(op.args[3]), get<c10::optional<at::Device>>(op.args[4]), get<c10::optional<bool>>(op.args[5])));
  break;

case H_FULL_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::full_outf(ks, get<at::IntArrayRef>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_FULL_LIKE:
  set(op.tensor, at::redispatch::full_like(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<c10::optional<at::ScalarType>>(op.args[2]), get<c10::optional<at::Layout>>(op.args[3]), get<c10::optional<at::Device>>(op.args[4]), get<c10::optional<bool>>(op.args[5]), get<c10::optional<at::MemoryFormat>>(op.args[6])));
  break;

case H_FROM_FILE:
  set(op.tensor, at::redispatch::from_file(ks, get<std::string>(op.args[0]), get<c10::optional<bool>>(op.args[1]), get<c10::optional<int64_t>>(op.args[2]), get<c10::optional<at::ScalarType>>(op.args[3]), get<c10::optional<at::Layout>>(op.args[4]), get<c10::optional<at::Device>>(op.args[5]), get<c10::optional<bool>>(op.args[6])));
  break;

case H_GCD_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::gcd_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_LCM_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::lcm_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_GRID_SAMPLER:
  set(op.tensor, at::redispatch::grid_sampler(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<int64_t>(op.args[2]), get<int64_t>(op.args[3]), get<bool>(op.args[4])));
  break;

case H_GRID_SAMPLER_2D:
  set(op.tensor, at::redispatch::grid_sampler_2d(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<int64_t>(op.args[2]), get<int64_t>(op.args[3]), get<bool>(op.args[4])));
  break;

// skip std::tuple<at::Tensor,at::Tensor> grid_sampler_2d_backward(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners)

case H__GRID_SAMPLER_2D_CPU_FALLBACK:
  set(op.tensor, at::redispatch::_grid_sampler_2d_cpu_fallback(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<int64_t>(op.args[2]), get<int64_t>(op.args[3]), get<bool>(op.args[4])));
  break;

// skip std::tuple<at::Tensor,at::Tensor> _grid_sampler_2d_cpu_fallback_backward(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners)

case H_GRID_SAMPLER_3D:
  set(op.tensor, at::redispatch::grid_sampler_3d(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<int64_t>(op.args[2]), get<int64_t>(op.args[3]), get<bool>(op.args[4])));
  break;

// skip std::tuple<at::Tensor,at::Tensor> grid_sampler_3d_backward(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners)

case H_HANN_WINDOW:
  set(op.tensor, at::redispatch::hann_window(ks, get<int64_t>(op.args[0]), get<c10::optional<at::ScalarType>>(op.args[1]), get<c10::optional<at::Layout>>(op.args[2]), get<c10::optional<at::Device>>(op.args[3]), get<c10::optional<bool>>(op.args[4])));
  break;

case H_HANN_WINDOW_PERIODIC:
  set(op.tensor, at::redispatch::hann_window(ks, get<int64_t>(op.args[0]), get<bool>(op.args[1]), get<c10::optional<at::ScalarType>>(op.args[2]), get<c10::optional<at::Layout>>(op.args[3]), get<c10::optional<at::Device>>(op.args[4]), get<c10::optional<bool>>(op.args[5])));
  break;

case H_HAMMING_WINDOW:
  set(op.tensor, at::redispatch::hamming_window(ks, get<int64_t>(op.args[0]), get<c10::optional<at::ScalarType>>(op.args[1]), get<c10::optional<at::Layout>>(op.args[2]), get<c10::optional<at::Device>>(op.args[3]), get<c10::optional<bool>>(op.args[4])));
  break;

case H_HAMMING_WINDOW_PERIODIC:
  set(op.tensor, at::redispatch::hamming_window(ks, get<int64_t>(op.args[0]), get<bool>(op.args[1]), get<c10::optional<at::ScalarType>>(op.args[2]), get<c10::optional<at::Layout>>(op.args[3]), get<c10::optional<at::Device>>(op.args[4]), get<c10::optional<bool>>(op.args[5])));
  break;

case H_HAMMING_WINDOW_PERIODIC_ALPHA:
  set(op.tensor, at::redispatch::hamming_window(ks, get<int64_t>(op.args[0]), get<bool>(op.args[1]), get<double>(op.args[2]), get<c10::optional<at::ScalarType>>(op.args[3]), get<c10::optional<at::Layout>>(op.args[4]), get<c10::optional<at::Device>>(op.args[5]), get<c10::optional<bool>>(op.args[6])));
  break;

case H_HAMMING_WINDOW_PERIODIC_ALPHA_BETA:
  set(op.tensor, at::redispatch::hamming_window(ks, get<int64_t>(op.args[0]), get<bool>(op.args[1]), get<double>(op.args[2]), get<double>(op.args[3]), get<c10::optional<at::ScalarType>>(op.args[4]), get<c10::optional<at::Layout>>(op.args[5]), get<c10::optional<at::Device>>(op.args[6]), get<c10::optional<bool>>(op.args[7])));
  break;

case H_KAISER_WINDOW:
  set(op.tensor, at::redispatch::kaiser_window(ks, get<int64_t>(op.args[0]), get<c10::optional<at::ScalarType>>(op.args[1]), get<c10::optional<at::Layout>>(op.args[2]), get<c10::optional<at::Device>>(op.args[3]), get<c10::optional<bool>>(op.args[4])));
  break;

case H_KAISER_WINDOW_PERIODIC:
  set(op.tensor, at::redispatch::kaiser_window(ks, get<int64_t>(op.args[0]), get<bool>(op.args[1]), get<c10::optional<at::ScalarType>>(op.args[2]), get<c10::optional<at::Layout>>(op.args[3]), get<c10::optional<at::Device>>(op.args[4]), get<c10::optional<bool>>(op.args[5])));
  break;

case H_KAISER_WINDOW_BETA:
  set(op.tensor, at::redispatch::kaiser_window(ks, get<int64_t>(op.args[0]), get<bool>(op.args[1]), get<double>(op.args[2]), get<c10::optional<at::ScalarType>>(op.args[3]), get<c10::optional<at::Layout>>(op.args[4]), get<c10::optional<at::Device>>(op.args[5]), get<c10::optional<bool>>(op.args[6])));
  break;

case H_HINGE_EMBEDDING_LOSS:
  set(op.tensor, at::redispatch::hinge_embedding_loss(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<double>(op.args[2]), get<int64_t>(op.args[3])));
  break;

case H_GROUP_NORM:
  set(op.tensor, at::redispatch::group_norm(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<c10::optional<at::Tensor>>(op.args[2]), get<c10::optional<at::Tensor>>(op.args[3]), get<double>(op.args[4]), get<bool>(op.args[5])));
  break;

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor> native_group_norm(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, int64_t N, int64_t C, int64_t HxW, int64_t group, double eps)

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor> native_group_norm_backward(const at::Tensor & grad_out, const at::Tensor & input, const at::Tensor & mean, const at::Tensor & rstd, const c10::optional<at::Tensor> & weight, int64_t N, int64_t C, int64_t HxW, int64_t group, std::array<bool,3> output_mask)

case H__FFT_R2C:
  set(op.tensor, at::redispatch::_fft_r2c(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<int64_t>(op.args[2]), get<bool>(op.args[3])));
  break;

case H__FFT_R2C_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::_fft_r2c_outf(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<int64_t>(op.args[2]), get<bool>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H__FFT_C2R:
  set(op.tensor, at::redispatch::_fft_c2r(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<int64_t>(op.args[2]), get<int64_t>(op.args[3])));
  break;

case H__FFT_C2R_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::_fft_c2r_outf(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<int64_t>(op.args[2]), get<int64_t>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H__FFT_C2C:
  set(op.tensor, at::redispatch::_fft_c2c(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<int64_t>(op.args[2]), get<bool>(op.args[3])));
  break;

case H__FFT_C2C_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::_fft_c2c_outf(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<int64_t>(op.args[2]), get<bool>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

// skip int64_t _cufft_get_plan_cache_size(int64_t device_index)

// skip int64_t _cufft_get_plan_cache_max_size(int64_t device_index)

// skip void _cufft_set_plan_cache_max_size(int64_t device_index, int64_t max_size)

// skip void _cufft_clear_plan_cache(int64_t device_index)

case H_INDEX_TENSOR:
  set(op.tensor, at::redispatch::index(ks, get<at::Tensor>(op.args[0]), get<c10::List<c10::optional<at::Tensor>>>(op.args[1])));
  break;

case H_INDEX_COPY_:
  init_update_in_place(op.tensor);
  at::redispatch::index_copy_(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_INDEX_COPY:
  set(op.tensor, at::redispatch::index_copy(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Tensor>(op.args[3])));
  break;

case H_INDEX_COPY__DIMNAME:
  init_update_in_place(op.tensor);
  at::redispatch::index_copy_(ks, get<at::Tensor>(op.args[0]), get<at::Dimname>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_INDEX_COPY_DIMNAME:
  set(op.tensor, at::redispatch::index_copy(ks, get<at::Tensor>(op.args[0]), get<at::Dimname>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Tensor>(op.args[3])));
  break;

case H_INDEX_PUT_:
  init_update_in_place(op.tensor);
  at::redispatch::index_put_(ks, get<at::Tensor>(op.args[0]), get<c10::List<c10::optional<at::Tensor>>>(op.args[1]), get<at::Tensor>(op.args[2]), get<bool>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_INDEX_PUT:
  set(op.tensor, at::redispatch::index_put(ks, get<at::Tensor>(op.args[0]), get<c10::List<c10::optional<at::Tensor>>>(op.args[1]), get<at::Tensor>(op.args[2]), get<bool>(op.args[3])));
  break;

case H__INDEX_PUT_IMPL_:
  init_update_in_place(op.tensor);
  at::redispatch::_index_put_impl_(ks, get<at::Tensor>(op.args[0]), get<c10::List<c10::optional<at::Tensor>>>(op.args[1]), get<at::Tensor>(op.args[2]), get<bool>(op.args[3]), get<bool>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_INSTANCE_NORM:
  set(op.tensor, at::redispatch::instance_norm(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::Tensor>>(op.args[1]), get<c10::optional<at::Tensor>>(op.args[2]), get<c10::optional<at::Tensor>>(op.args[3]), get<c10::optional<at::Tensor>>(op.args[4]), get<bool>(op.args[5]), get<double>(op.args[6]), get<double>(op.args[7]), get<bool>(op.args[8])));
  break;

case H_INVERSE:
  set(op.tensor, at::redispatch::inverse(ks, get<at::Tensor>(op.args[0])));
  break;

case H_INVERSE_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::inverse_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H__INVERSE_HELPER:
  set(op.tensor, at::redispatch::_inverse_helper(ks, get<at::Tensor>(op.args[0])));
  break;

case H_ISCLOSE:
  set(op.tensor, at::redispatch::isclose(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<double>(op.args[2]), get<double>(op.args[3]), get<bool>(op.args[4])));
  break;

case H_ISNAN:
  set(op.tensor, at::redispatch::isnan(ks, get<at::Tensor>(op.args[0])));
  break;

// skip bool is_distributed(const at::Tensor & self)

case H_ISREAL:
  set(op.tensor, at::redispatch::isreal(ks, get<at::Tensor>(op.args[0])));
  break;

// skip bool is_nonzero(const at::Tensor & self)

// skip bool is_same_size(const at::Tensor & self, const at::Tensor & other)

case H_KL_DIV:
  set(op.tensor, at::redispatch::kl_div(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<int64_t>(op.args[2]), get<bool>(op.args[3])));
  break;

case H_KL_DIV_BACKWARD:
  set(op.tensor, at::redispatch::kl_div_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<int64_t>(op.args[3]), get<bool>(op.args[4])));
  break;

case H_KRON:
  set(op.tensor, at::redispatch::kron(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_KRON_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::kron_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

// skip std::tuple<at::Tensor,at::Tensor> kthvalue(const at::Tensor & self, int64_t k, int64_t dim, bool keepdim)

// skip std::tuple<at::Tensor &,at::Tensor &> kthvalue_outf(const at::Tensor & self, int64_t k, int64_t dim, bool keepdim, at::Tensor & values, at::Tensor & indices)

// skip std::tuple<at::Tensor,at::Tensor> kthvalue(const at::Tensor & self, int64_t k, at::Dimname dim, bool keepdim)

// skip std::tuple<at::Tensor &,at::Tensor &> kthvalue_outf(const at::Tensor & self, int64_t k, at::Dimname dim, bool keepdim, at::Tensor & values, at::Tensor & indices)

case H_LAYER_NORM:
  set(op.tensor, at::redispatch::layer_norm(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<c10::optional<at::Tensor>>(op.args[2]), get<c10::optional<at::Tensor>>(op.args[3]), get<double>(op.args[4]), get<bool>(op.args[5])));
  break;

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor> native_layer_norm(const at::Tensor & input, at::IntArrayRef normalized_shape, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, double eps)

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor> native_layer_norm_backward(const at::Tensor & grad_out, const at::Tensor & input, at::IntArrayRef normalized_shape, const at::Tensor & mean, const at::Tensor & rstd, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, std::array<bool,3> output_mask)

case H_NAN_TO_NUM:
  set(op.tensor, at::redispatch::nan_to_num(ks, get<at::Tensor>(op.args[0]), get<c10::optional<double>>(op.args[1]), get<c10::optional<double>>(op.args[2]), get<c10::optional<double>>(op.args[3])));
  break;

case H_NAN_TO_NUM_:
  init_update_in_place(op.tensor);
  at::redispatch::nan_to_num_(ks, get<at::Tensor>(op.args[0]), get<c10::optional<double>>(op.args[1]), get<c10::optional<double>>(op.args[2]), get<c10::optional<double>>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_NAN_TO_NUM_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::nan_to_num_outf(ks, get<at::Tensor>(op.args[0]), get<c10::optional<double>>(op.args[1]), get<c10::optional<double>>(op.args[2]), get<c10::optional<double>>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_LINEAR:
  set(op.tensor, at::redispatch::linear(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<at::Tensor>>(op.args[2])));
  break;

case H_MKLDNN_LINEAR:
  set(op.tensor, at::redispatch::mkldnn_linear(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<at::Tensor>>(op.args[2])));
  break;

case H_MKLDNN_LINEAR_BACKWARD_INPUT:
  set(op.tensor, at::redispatch::mkldnn_linear_backward_input(ks, get<at::IntArrayRef>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2])));
  break;

// skip std::tuple<at::Tensor,at::Tensor> mkldnn_linear_backward_weights(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & weight, bool bias_defined)

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor> mkldnn_linear_backward(const at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & weight, std::array<bool,3> output_mask)

case H_FBGEMM_LINEAR_INT8_WEIGHT_FP32_ACTIVATION:
  set(op.tensor, at::redispatch::fbgemm_linear_int8_weight_fp32_activation(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Tensor>(op.args[3]), get<at::Scalar>(op.args[4]), get<at::Scalar>(op.args[5]), get<at::Tensor>(op.args[6])));
  break;

case H_FBGEMM_LINEAR_INT8_WEIGHT:
  set(op.tensor, at::redispatch::fbgemm_linear_int8_weight(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Tensor>(op.args[3]), get<at::Scalar>(op.args[4]), get<at::Scalar>(op.args[5]), get<at::Tensor>(op.args[6])));
  break;

// skip std::tuple<at::Tensor,at::Tensor,double,int64_t> fbgemm_linear_quantize_weight(const at::Tensor & input)

case H_FBGEMM_PACK_GEMM_MATRIX_FP16:
  set(op.tensor, at::redispatch::fbgemm_pack_gemm_matrix_fp16(ks, get<at::Tensor>(op.args[0])));
  break;

case H_FBGEMM_LINEAR_FP16_WEIGHT_FP32_ACTIVATION:
  set(op.tensor, at::redispatch::fbgemm_linear_fp16_weight_fp32_activation(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2])));
  break;

case H_FBGEMM_LINEAR_FP16_WEIGHT:
  set(op.tensor, at::redispatch::fbgemm_linear_fp16_weight(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2])));
  break;

case H_FBGEMM_PACK_QUANTIZED_MATRIX:
  set(op.tensor, at::redispatch::fbgemm_pack_quantized_matrix(ks, get<at::Tensor>(op.args[0])));
  break;

case H_FBGEMM_PACK_QUANTIZED_MATRIX_KN:
  set(op.tensor, at::redispatch::fbgemm_pack_quantized_matrix(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<int64_t>(op.args[2])));
  break;

case H_LDEXP_TENSOR:
  set(op.tensor, at::redispatch::ldexp(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_LDEXP_:
  init_update_in_place(op.tensor);
  at::redispatch::ldexp_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_LDEXP_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::ldexp_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_LINSPACE:
  set(op.tensor, at::redispatch::linspace(ks, get<at::Scalar>(op.args[0]), get<at::Scalar>(op.args[1]), get<c10::optional<int64_t>>(op.args[2]), get<c10::optional<at::ScalarType>>(op.args[3]), get<c10::optional<at::Layout>>(op.args[4]), get<c10::optional<at::Device>>(op.args[5]), get<c10::optional<bool>>(op.args[6])));
  break;

case H_LINSPACE_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::linspace_outf(ks, get<at::Scalar>(op.args[0]), get<at::Scalar>(op.args[1]), get<c10::optional<int64_t>>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_LOG_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::log_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_LOG10_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::log10_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_LOG1P:
  set(op.tensor, at::redispatch::log1p(ks, get<at::Tensor>(op.args[0])));
  break;

case H_LOG1P_:
  init_update_in_place(op.tensor);
  at::redispatch::log1p_(ks, get<at::Tensor>(op.args[0]));
  end_update_in_place(op.tensor);
  break;

case H_LOG1P_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::log1p_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_LOG2_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::log2_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_LOGADDEXP_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::logaddexp_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_LOGADDEXP:
  set(op.tensor, at::redispatch::logaddexp(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_LOGADDEXP2_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::logaddexp2_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_LOGADDEXP2:
  set(op.tensor, at::redispatch::logaddexp2(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_XLOGY_TENSOR:
  set(op.tensor, at::redispatch::xlogy(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_XLOGY_SCALAR_SELF:
  set(op.tensor, at::redispatch::xlogy(ks, get<at::Scalar>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_XLOGY_SCALAR_OTHER:
  set(op.tensor, at::redispatch::xlogy(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1])));
  break;

case H_XLOGY__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::xlogy_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_XLOGY__SCALAR_OTHER:
  init_update_in_place(op.tensor);
  at::redispatch::xlogy_(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_XLOGY_OUTTENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::xlogy_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_XLOGY_OUTSCALAR_SELF:
  init_update_in_place(op.tensor);
  at::redispatch::xlogy_outf(ks, get<at::Scalar>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_XLOGY_OUTSCALAR_OTHER:
  init_update_in_place(op.tensor);
  at::redispatch::xlogy_outf(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_LOGDET:
  set(op.tensor, at::redispatch::logdet(ks, get<at::Tensor>(op.args[0])));
  break;

case H_LOGSPACE:
  set(op.tensor, at::redispatch::logspace(ks, get<at::Scalar>(op.args[0]), get<at::Scalar>(op.args[1]), get<c10::optional<int64_t>>(op.args[2]), get<double>(op.args[3]), get<c10::optional<at::ScalarType>>(op.args[4]), get<c10::optional<at::Layout>>(op.args[5]), get<c10::optional<at::Device>>(op.args[6]), get<c10::optional<bool>>(op.args[7])));
  break;

case H_LOGSPACE_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::logspace_outf(ks, get<at::Scalar>(op.args[0]), get<at::Scalar>(op.args[1]), get<c10::optional<int64_t>>(op.args[2]), get<double>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_LOG_SOFTMAX_INT:
  set(op.tensor, at::redispatch::log_softmax(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<c10::optional<at::ScalarType>>(op.args[2])));
  break;

case H_LOG_SOFTMAX_DIMNAME:
  set(op.tensor, at::redispatch::log_softmax(ks, get<at::Tensor>(op.args[0]), get<at::Dimname>(op.args[1]), get<c10::optional<at::ScalarType>>(op.args[2])));
  break;

case H__LOG_SOFTMAX:
  set(op.tensor, at::redispatch::_log_softmax(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<bool>(op.args[2])));
  break;

case H__LOG_SOFTMAX_BACKWARD_DATA:
  set(op.tensor, at::redispatch::_log_softmax_backward_data(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<int64_t>(op.args[2]), get<at::Tensor>(op.args[3])));
  break;

case H__LOGCUMSUMEXP:
  set(op.tensor, at::redispatch::_logcumsumexp(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1])));
  break;

case H__LOGCUMSUMEXP_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::_logcumsumexp_outf(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_LOGCUMSUMEXP:
  set(op.tensor, at::redispatch::logcumsumexp(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1])));
  break;

case H_LOGCUMSUMEXP_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::logcumsumexp_outf(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_LOGCUMSUMEXP_DIMNAME:
  set(op.tensor, at::redispatch::logcumsumexp(ks, get<at::Tensor>(op.args[0]), get<at::Dimname>(op.args[1])));
  break;

case H_LOGCUMSUMEXP_DIMNAME_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::logcumsumexp_outf(ks, get<at::Tensor>(op.args[0]), get<at::Dimname>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_LOGSUMEXP:
  set(op.tensor, at::redispatch::logsumexp(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<bool>(op.args[2])));
  break;

case H_LOGSUMEXP_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::logsumexp_outf(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<bool>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_LOGSUMEXP_NAMES:
  set(op.tensor, at::redispatch::logsumexp(ks, get<at::Tensor>(op.args[0]), get<at::DimnameList>(op.args[1]), get<bool>(op.args[2])));
  break;

case H_LOGSUMEXP_NAMES_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::logsumexp_outf(ks, get<at::Tensor>(op.args[0]), get<at::DimnameList>(op.args[1]), get<bool>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_MARGIN_RANKING_LOSS:
  set(op.tensor, at::redispatch::margin_ranking_loss(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<double>(op.args[3]), get<int64_t>(op.args[4])));
  break;

case H_MATMUL:
  set(op.tensor, at::redispatch::matmul(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_MATMUL_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::matmul_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_MATRIX_RANK_TOL:
  set(op.tensor, at::redispatch::matrix_rank(ks, get<at::Tensor>(op.args[0]), get<double>(op.args[1]), get<bool>(op.args[2])));
  break;

case H_MATRIX_RANK:
  set(op.tensor, at::redispatch::matrix_rank(ks, get<at::Tensor>(op.args[0]), get<bool>(op.args[1])));
  break;

case H_MATRIX_POWER:
  set(op.tensor, at::redispatch::matrix_power(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1])));
  break;

case H_MATRIX_POWER_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::matrix_power_outf(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_MATRIX_EXP:
  set(op.tensor, at::redispatch::matrix_exp(ks, get<at::Tensor>(op.args[0])));
  break;

case H_MATRIX_EXP_BACKWARD:
  set(op.tensor, at::redispatch::matrix_exp_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

// skip std::tuple<at::Tensor,at::Tensor> _aminmax(const at::Tensor & self)

// skip std::tuple<at::Tensor,at::Tensor> _aminmax(const at::Tensor & self, int64_t dim, bool keepdim)

case H__COMPUTE_LINEAR_COMBINATION:
  set(op.tensor, at::redispatch::_compute_linear_combination(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H__COMPUTE_LINEAR_COMBINATION_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::_compute_linear_combination_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

// skip std::tuple<at::Tensor,at::Tensor> max(const at::Tensor & self, int64_t dim, bool keepdim)

// skip std::tuple<at::Tensor &,at::Tensor &> max_outf(const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & max, at::Tensor & max_values)

// skip std::tuple<at::Tensor,at::Tensor> max(const at::Tensor & self, at::Dimname dim, bool keepdim)

// skip std::tuple<at::Tensor &,at::Tensor &> max_outf(const at::Tensor & self, at::Dimname dim, bool keepdim, at::Tensor & max, at::Tensor & max_values)

case H_VALUE_SELECTING_REDUCTION_BACKWARD:
  set(op.tensor, at::redispatch::value_selecting_reduction_backward(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<bool>(op.args[4])));
  break;

case H_AMAX:
  set(op.tensor, at::redispatch::amax(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<bool>(op.args[2])));
  break;

case H_AMAX_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::amax_outf(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<bool>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

// skip std::tuple<at::Tensor,at::Tensor> max_pool1d_with_indices(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode)

case H_MAX_POOL1D:
  set(op.tensor, at::redispatch::max_pool1d(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<bool>(op.args[5])));
  break;

case H_MAX_POOL2D:
  set(op.tensor, at::redispatch::max_pool2d(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<bool>(op.args[5])));
  break;

case H_MKLDNN_MAX_POOL2D:
  set(op.tensor, at::redispatch::mkldnn_max_pool2d(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<bool>(op.args[5])));
  break;

case H_MKLDNN_MAX_POOL2D_BACKWARD:
  set(op.tensor, at::redispatch::mkldnn_max_pool2d_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<at::IntArrayRef>(op.args[6]), get<bool>(op.args[7])));
  break;

case H_MKLDNN_MAX_POOL3D:
  set(op.tensor, at::redispatch::mkldnn_max_pool3d(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<bool>(op.args[5])));
  break;

case H_MKLDNN_MAX_POOL3D_BACKWARD:
  set(op.tensor, at::redispatch::mkldnn_max_pool3d_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<at::IntArrayRef>(op.args[6]), get<bool>(op.args[7])));
  break;

case H_QUANTIZED_MAX_POOL1D:
  set(op.tensor, at::redispatch::quantized_max_pool1d(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<bool>(op.args[5])));
  break;

case H_QUANTIZED_MAX_POOL2D:
  set(op.tensor, at::redispatch::quantized_max_pool2d(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<bool>(op.args[5])));
  break;

case H_MAX_POOL3D:
  set(op.tensor, at::redispatch::max_pool3d(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<bool>(op.args[5])));
  break;

case H_MEAN:
  set(op.tensor, at::redispatch::mean(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::ScalarType>>(op.args[1])));
  break;

case H_MEAN_DIM:
  set(op.tensor, at::redispatch::mean(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<bool>(op.args[2]), get<c10::optional<at::ScalarType>>(op.args[3])));
  break;

case H_MEAN_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::mean_outf(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<bool>(op.args[2]), get<c10::optional<at::ScalarType>>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_MEAN_NAMES_DIM:
  set(op.tensor, at::redispatch::mean(ks, get<at::Tensor>(op.args[0]), get<at::DimnameList>(op.args[1]), get<bool>(op.args[2]), get<c10::optional<at::ScalarType>>(op.args[3])));
  break;

case H_MEAN_NAMES_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::mean_outf(ks, get<at::Tensor>(op.args[0]), get<at::DimnameList>(op.args[1]), get<bool>(op.args[2]), get<c10::optional<at::ScalarType>>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_MEDIAN:
  set(op.tensor, at::redispatch::median(ks, get<at::Tensor>(op.args[0])));
  break;

// skip std::tuple<at::Tensor,at::Tensor> median(const at::Tensor & self, int64_t dim, bool keepdim)

// skip std::tuple<at::Tensor &,at::Tensor &> median_outf(const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & values, at::Tensor & indices)

// skip std::tuple<at::Tensor,at::Tensor> median(const at::Tensor & self, at::Dimname dim, bool keepdim)

// skip std::tuple<at::Tensor &,at::Tensor &> median_outf(const at::Tensor & self, at::Dimname dim, bool keepdim, at::Tensor & values, at::Tensor & indices)

case H_NANMEDIAN:
  set(op.tensor, at::redispatch::nanmedian(ks, get<at::Tensor>(op.args[0])));
  break;

// skip std::tuple<at::Tensor,at::Tensor> nanmedian(const at::Tensor & self, int64_t dim, bool keepdim)

// skip std::tuple<at::Tensor &,at::Tensor &> nanmedian_outf(const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & values, at::Tensor & indices)

// skip std::tuple<at::Tensor,at::Tensor> nanmedian(const at::Tensor & self, at::Dimname dim, bool keepdim)

// skip std::tuple<at::Tensor &,at::Tensor &> nanmedian_outf(const at::Tensor & self, at::Dimname dim, bool keepdim, at::Tensor & values, at::Tensor & indices)

// skip std::tuple<at::Tensor,at::Tensor> min(const at::Tensor & self, int64_t dim, bool keepdim)

// skip std::tuple<at::Tensor &,at::Tensor &> min_outf(const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & min, at::Tensor & min_indices)

// skip std::tuple<at::Tensor,at::Tensor> min(const at::Tensor & self, at::Dimname dim, bool keepdim)

// skip std::tuple<at::Tensor &,at::Tensor &> min_outf(const at::Tensor & self, at::Dimname dim, bool keepdim, at::Tensor & min, at::Tensor & min_indices)

case H_AMIN:
  set(op.tensor, at::redispatch::amin(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<bool>(op.args[2])));
  break;

case H_AMIN_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::amin_outf(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<bool>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_MKLDNN_CONVOLUTION:
  set(op.tensor, at::redispatch::mkldnn_convolution(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<at::Tensor>>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<int64_t>(op.args[6])));
  break;

case H_MKLDNN_CONVOLUTION_BACKWARD_INPUT:
  set(op.tensor, at::redispatch::mkldnn_convolution_backward_input(ks, get<at::IntArrayRef>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<int64_t>(op.args[6]), get<bool>(op.args[7])));
  break;

// skip std::tuple<at::Tensor,at::Tensor> mkldnn_convolution_backward_weights(at::IntArrayRef weight_size, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool bias_defined)

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor> mkldnn_convolution_backward(const at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, std::array<bool,3> output_mask)

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor> miopen_batch_norm(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, bool training, double exponential_average_factor, double epsilon)

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor> miopen_batch_norm_backward(const at::Tensor & input, const at::Tensor & grad_output, const at::Tensor & weight, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, const c10::optional<at::Tensor> & save_mean, const c10::optional<at::Tensor> & save_var, double epsilon)

case H_MIOPEN_CONVOLUTION:
  set(op.tensor, at::redispatch::miopen_convolution(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<at::Tensor>>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<int64_t>(op.args[6]), get<bool>(op.args[7]), get<bool>(op.args[8])));
  break;

case H_MIOPEN_CONVOLUTION_BACKWARD_INPUT:
  set(op.tensor, at::redispatch::miopen_convolution_backward_input(ks, get<at::IntArrayRef>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<int64_t>(op.args[6]), get<bool>(op.args[7]), get<bool>(op.args[8])));
  break;

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor> miopen_convolution_backward(const at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,3> output_mask)

case H_MIOPEN_CONVOLUTION_BACKWARD_BIAS:
  set(op.tensor, at::redispatch::miopen_convolution_backward_bias(ks, get<at::Tensor>(op.args[0])));
  break;

case H_MIOPEN_CONVOLUTION_BACKWARD_WEIGHT:
  set(op.tensor, at::redispatch::miopen_convolution_backward_weight(ks, get<at::IntArrayRef>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<int64_t>(op.args[6]), get<bool>(op.args[7]), get<bool>(op.args[8])));
  break;

case H_MIOPEN_CONVOLUTION_TRANSPOSE:
  set(op.tensor, at::redispatch::miopen_convolution_transpose(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<at::Tensor>>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<at::IntArrayRef>(op.args[6]), get<int64_t>(op.args[7]), get<bool>(op.args[8]), get<bool>(op.args[9])));
  break;

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor> miopen_convolution_transpose_backward(const at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,3> output_mask)

case H_MIOPEN_CONVOLUTION_TRANSPOSE_BACKWARD_INPUT:
  set(op.tensor, at::redispatch::miopen_convolution_transpose_backward_input(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<int64_t>(op.args[5]), get<bool>(op.args[6]), get<bool>(op.args[7])));
  break;

case H_MIOPEN_CONVOLUTION_TRANSPOSE_BACKWARD_WEIGHT:
  set(op.tensor, at::redispatch::miopen_convolution_transpose_backward_weight(ks, get<at::IntArrayRef>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<int64_t>(op.args[6]), get<bool>(op.args[7]), get<bool>(op.args[8])));
  break;

case H_MIOPEN_DEPTHWISE_CONVOLUTION:
  set(op.tensor, at::redispatch::miopen_depthwise_convolution(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<at::Tensor>>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<int64_t>(op.args[6]), get<bool>(op.args[7]), get<bool>(op.args[8])));
  break;

case H_MIOPEN_DEPTHWISE_CONVOLUTION_BACKWARD_INPUT:
  set(op.tensor, at::redispatch::miopen_depthwise_convolution_backward_input(ks, get<at::IntArrayRef>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<int64_t>(op.args[6]), get<bool>(op.args[7]), get<bool>(op.args[8])));
  break;

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor> miopen_depthwise_convolution_backward(const at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,3> output_mask)

case H_MIOPEN_DEPTHWISE_CONVOLUTION_BACKWARD_WEIGHT:
  set(op.tensor, at::redispatch::miopen_depthwise_convolution_backward_weight(ks, get<at::IntArrayRef>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<int64_t>(op.args[6]), get<bool>(op.args[7]), get<bool>(op.args[8])));
  break;

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> miopen_rnn(const at::Tensor & input, at::TensorList weight, int64_t weight_stride0, const at::Tensor & hx, const c10::optional<at::Tensor> & cx, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, at::IntArrayRef batch_sizes, const c10::optional<at::Tensor> & dropout_state)

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor,std::vector<at::Tensor>> miopen_rnn_backward(const at::Tensor & input, at::TensorList weight, int64_t weight_stride0, const at::Tensor & weight_buf, const at::Tensor & hx, const c10::optional<at::Tensor> & cx, const at::Tensor & output, const c10::optional<at::Tensor> & grad_output, const c10::optional<at::Tensor> & grad_hy, const c10::optional<at::Tensor> & grad_cy, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, at::IntArrayRef batch_sizes, const c10::optional<at::Tensor> & dropout_state, const at::Tensor & reserve, std::array<bool,4> output_mask)

case H_MM:
  set(op.tensor, at::redispatch::mm(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_MM_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::mm_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H__SPARSE_MM:
  set(op.tensor, at::redispatch::_sparse_mm(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H__SPARSE_SPARSE_MATMUL:
  set(op.tensor, at::redispatch::_sparse_sparse_matmul(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H__SPARSE_MASK_HELPER:
  set(op.tensor, at::redispatch::_sparse_mask_helper(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

// skip std::tuple<at::Tensor,at::Tensor> mode(const at::Tensor & self, int64_t dim, bool keepdim)

// skip std::tuple<at::Tensor &,at::Tensor &> mode_outf(const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & values, at::Tensor & indices)

// skip std::tuple<at::Tensor,at::Tensor> mode(const at::Tensor & self, at::Dimname dim, bool keepdim)

// skip std::tuple<at::Tensor &,at::Tensor &> mode_outf(const at::Tensor & self, at::Dimname dim, bool keepdim, at::Tensor & values, at::Tensor & indices)

case H_MUL_TENSOR:
  set(op.tensor, at::redispatch::mul(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_MUL__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::mul_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_MUL_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::mul_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_MUL_SCALAR:
  set(op.tensor, at::redispatch::mul(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1])));
  break;

case H_MUL__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::mul_(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_MULTIPLY_TENSOR:
  set(op.tensor, at::redispatch::multiply(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_MULTIPLY__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::multiply_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_MULTIPLY_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::multiply_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_MULTIPLY_SCALAR:
  set(op.tensor, at::redispatch::multiply(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1])));
  break;

case H_MULTIPLY__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::multiply_(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_MV:
  set(op.tensor, at::redispatch::mv(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_MV_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::mv_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_MVLGAMMA:
  set(op.tensor, at::redispatch::mvlgamma(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1])));
  break;

case H_MVLGAMMA_:
  init_update_in_place(op.tensor);
  at::redispatch::mvlgamma_(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_NARROW_COPY:
  set(op.tensor, at::redispatch::narrow_copy(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<int64_t>(op.args[2]), get<int64_t>(op.args[3])));
  break;

case H_NARROW_COPY_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::narrow_copy_outf(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<int64_t>(op.args[2]), get<int64_t>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_NARROW:
  set(op.tensor, at::redispatch::narrow(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<int64_t>(op.args[2]), get<int64_t>(op.args[3])));
  break;

case H_NARROW_TENSOR:
  set(op.tensor, at::redispatch::narrow(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<at::Tensor>(op.args[2]), get<int64_t>(op.args[3])));
  break;

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor> native_batch_norm(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, bool training, double momentum, double eps)

// skip std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> native_batch_norm_outf(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, bool training, double momentum, double eps, at::Tensor & out, at::Tensor & save_mean, at::Tensor & save_invstd)

// skip std::tuple<at::Tensor,at::Tensor> batch_norm_stats(const at::Tensor & input, double eps)

case H_BATCH_NORM_ELEMT:
  set(op.tensor, at::redispatch::batch_norm_elemt(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::Tensor>>(op.args[1]), get<c10::optional<at::Tensor>>(op.args[2]), get<at::Tensor>(op.args[3]), get<at::Tensor>(op.args[4]), get<double>(op.args[5])));
  break;

case H_BATCH_NORM_ELEMT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::batch_norm_elemt_outf(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::Tensor>>(op.args[1]), get<c10::optional<at::Tensor>>(op.args[2]), get<at::Tensor>(op.args[3]), get<at::Tensor>(op.args[4]), get<double>(op.args[5]), get<at::Tensor>(op.args[6]));
  end_update_in_place(op.tensor);
  break;

// skip std::tuple<at::Tensor,at::Tensor> batch_norm_gather_stats(const at::Tensor & input, const at::Tensor & mean, const at::Tensor & invstd, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, double momentum, double eps, int64_t count)

// skip std::tuple<at::Tensor,at::Tensor> batch_norm_gather_stats_with_counts(const at::Tensor & input, const at::Tensor & mean, const at::Tensor & invstd, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, double momentum, double eps, const at::Tensor & counts)

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor> native_batch_norm_backward(const at::Tensor & grad_out, const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, const c10::optional<at::Tensor> & save_mean, const c10::optional<at::Tensor> & save_invstd, bool train, double eps, std::array<bool,3> output_mask)

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> batch_norm_backward_reduce(const at::Tensor & grad_out, const at::Tensor & input, const at::Tensor & mean, const at::Tensor & invstd, const c10::optional<at::Tensor> & weight, bool input_g, bool weight_g, bool bias_g)

case H_BATCH_NORM_BACKWARD_ELEMT:
  set(op.tensor, at::redispatch::batch_norm_backward_elemt(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Tensor>(op.args[3]), get<c10::optional<at::Tensor>>(op.args[4]), get<at::Tensor>(op.args[5]), get<at::Tensor>(op.args[6]), get<at::Tensor>(op.args[7])));
  break;

// skip std::tuple<at::Tensor,at::Tensor> batch_norm_update_stats(const at::Tensor & input, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, double momentum)

// skip bool is_vulkan_available()

// skip bool _nnpack_available()

case H__NNPACK_SPATIAL_CONVOLUTION:
  set(op.tensor, at::redispatch::_nnpack_spatial_convolution(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<at::Tensor>>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4])));
  break;

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor> _nnpack_spatial_convolution_backward(const at::Tensor & input, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, std::array<bool,3> output_mask)

case H__NNPACK_SPATIAL_CONVOLUTION_BACKWARD_INPUT:
  set(op.tensor, at::redispatch::_nnpack_spatial_convolution_backward_input(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::IntArrayRef>(op.args[3])));
  break;

case H__NNPACK_SPATIAL_CONVOLUTION_BACKWARD_WEIGHT:
  set(op.tensor, at::redispatch::_nnpack_spatial_convolution_backward_weight(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::IntArrayRef>(op.args[3])));
  break;

case H_ONES_NAMES:
  set(op.tensor, at::redispatch::ones(ks, get<at::IntArrayRef>(op.args[0]), get<c10::optional<at::DimnameList>>(op.args[1]), get<c10::optional<at::ScalarType>>(op.args[2]), get<c10::optional<at::Layout>>(op.args[3]), get<c10::optional<at::Device>>(op.args[4]), get<c10::optional<bool>>(op.args[5])));
  break;

case H_ONES:
  set(op.tensor, at::redispatch::ones(ks, get<at::IntArrayRef>(op.args[0]), get<c10::optional<at::ScalarType>>(op.args[1]), get<c10::optional<at::Layout>>(op.args[2]), get<c10::optional<at::Device>>(op.args[3]), get<c10::optional<bool>>(op.args[4])));
  break;

case H_ONES_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::ones_outf(ks, get<at::IntArrayRef>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_ONES_LIKE:
  set(op.tensor, at::redispatch::ones_like(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::ScalarType>>(op.args[1]), get<c10::optional<at::Layout>>(op.args[2]), get<c10::optional<at::Device>>(op.args[3]), get<c10::optional<bool>>(op.args[4]), get<c10::optional<at::MemoryFormat>>(op.args[5])));
  break;

case H_PAIRWISE_DISTANCE:
  set(op.tensor, at::redispatch::pairwise_distance(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<double>(op.args[2]), get<double>(op.args[3]), get<bool>(op.args[4])));
  break;

case H_CDIST:
  set(op.tensor, at::redispatch::cdist(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<double>(op.args[2]), get<c10::optional<int64_t>>(op.args[3])));
  break;

case H__EUCLIDEAN_DIST:
  set(op.tensor, at::redispatch::_euclidean_dist(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H__CDIST_FORWARD:
  set(op.tensor, at::redispatch::_cdist_forward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<double>(op.args[2]), get<c10::optional<int64_t>>(op.args[3])));
  break;

case H__CDIST_BACKWARD:
  set(op.tensor, at::redispatch::_cdist_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<double>(op.args[3]), get<at::Tensor>(op.args[4])));
  break;

case H_PDIST:
  set(op.tensor, at::redispatch::pdist(ks, get<at::Tensor>(op.args[0]), get<double>(op.args[1])));
  break;

case H__PDIST_FORWARD:
  set(op.tensor, at::redispatch::_pdist_forward(ks, get<at::Tensor>(op.args[0]), get<double>(op.args[1])));
  break;

case H__PDIST_BACKWARD:
  set(op.tensor, at::redispatch::_pdist_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<double>(op.args[2]), get<at::Tensor>(op.args[3])));
  break;

case H_COSINE_SIMILARITY:
  set(op.tensor, at::redispatch::cosine_similarity(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<int64_t>(op.args[2]), get<double>(op.args[3])));
  break;

case H_PERMUTE:
  set(op.tensor, at::redispatch::permute(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1])));
  break;

case H_MOVEDIM_INTLIST:
  set(op.tensor, at::redispatch::movedim(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<at::IntArrayRef>(op.args[2])));
  break;

case H_MOVEDIM_INT:
  set(op.tensor, at::redispatch::movedim(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<int64_t>(op.args[2])));
  break;

case H_MOVEAXIS_INTLIST:
  set(op.tensor, at::redispatch::moveaxis(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<at::IntArrayRef>(op.args[2])));
  break;

case H_MOVEAXIS_INT:
  set(op.tensor, at::redispatch::moveaxis(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<int64_t>(op.args[2])));
  break;

case H_NUMPY_T:
  set(op.tensor, at::redispatch::numpy_T(ks, get<at::Tensor>(op.args[0])));
  break;

case H_PIXEL_SHUFFLE:
  set(op.tensor, at::redispatch::pixel_shuffle(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1])));
  break;

case H_PIXEL_UNSHUFFLE:
  set(op.tensor, at::redispatch::pixel_unshuffle(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1])));
  break;

case H_CHANNEL_SHUFFLE:
  set(op.tensor, at::redispatch::channel_shuffle(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1])));
  break;

// skip bool is_pinned(const at::Tensor & self)

case H_PIN_MEMORY:
  set(op.tensor, at::redispatch::pin_memory(ks, get<at::Tensor>(op.args[0])));
  break;

case H_PINVERSE:
  set(op.tensor, at::redispatch::pinverse(ks, get<at::Tensor>(op.args[0]), get<double>(op.args[1])));
  break;

case H_POISSON_NLL_LOSS:
  set(op.tensor, at::redispatch::poisson_nll_loss(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<bool>(op.args[2]), get<bool>(op.args[3]), get<double>(op.args[4]), get<int64_t>(op.args[5])));
  break;

case H_RAD2DEG:
  set(op.tensor, at::redispatch::rad2deg(ks, get<at::Tensor>(op.args[0])));
  break;

case H_RAD2DEG_:
  init_update_in_place(op.tensor);
  at::redispatch::rad2deg_(ks, get<at::Tensor>(op.args[0]));
  end_update_in_place(op.tensor);
  break;

case H_RAD2DEG_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::rad2deg_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_DEG2RAD:
  set(op.tensor, at::redispatch::deg2rad(ks, get<at::Tensor>(op.args[0])));
  break;

case H_DEG2RAD_:
  init_update_in_place(op.tensor);
  at::redispatch::deg2rad_(ks, get<at::Tensor>(op.args[0]));
  end_update_in_place(op.tensor);
  break;

case H_DEG2RAD_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::deg2rad_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_SCALAR_TENSOR:
  set(op.tensor, at::redispatch::scalar_tensor(ks, get<at::Scalar>(op.args[0]), get<c10::optional<at::ScalarType>>(op.args[1]), get<c10::optional<at::Layout>>(op.args[2]), get<c10::optional<at::Device>>(op.args[3]), get<c10::optional<bool>>(op.args[4])));
  break;

case H_RAND_NAMES:
  set(op.tensor, at::redispatch::rand(ks, get<at::IntArrayRef>(op.args[0]), get<c10::optional<at::DimnameList>>(op.args[1]), get<c10::optional<at::ScalarType>>(op.args[2]), get<c10::optional<at::Layout>>(op.args[3]), get<c10::optional<at::Device>>(op.args[4]), get<c10::optional<bool>>(op.args[5])));
  break;

case H_RAND_GENERATOR_WITH_NAMES:
  set(op.tensor, at::redispatch::rand(ks, get<at::IntArrayRef>(op.args[0]), get<c10::optional<at::Generator>>(op.args[1]), get<c10::optional<at::DimnameList>>(op.args[2]), get<c10::optional<at::ScalarType>>(op.args[3]), get<c10::optional<at::Layout>>(op.args[4]), get<c10::optional<at::Device>>(op.args[5]), get<c10::optional<bool>>(op.args[6])));
  break;

case H_RAND:
  set(op.tensor, at::redispatch::rand(ks, get<at::IntArrayRef>(op.args[0]), get<c10::optional<at::ScalarType>>(op.args[1]), get<c10::optional<at::Layout>>(op.args[2]), get<c10::optional<at::Device>>(op.args[3]), get<c10::optional<bool>>(op.args[4])));
  break;

case H_RAND_GENERATOR:
  set(op.tensor, at::redispatch::rand(ks, get<at::IntArrayRef>(op.args[0]), get<c10::optional<at::Generator>>(op.args[1]), get<c10::optional<at::ScalarType>>(op.args[2]), get<c10::optional<at::Layout>>(op.args[3]), get<c10::optional<at::Device>>(op.args[4]), get<c10::optional<bool>>(op.args[5])));
  break;

case H_RAND_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::rand_outf(ks, get<at::IntArrayRef>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_RAND_GENERATOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::rand_outf(ks, get<at::IntArrayRef>(op.args[0]), get<c10::optional<at::Generator>>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_RAND_LIKE:
  set(op.tensor, at::redispatch::rand_like(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::ScalarType>>(op.args[1]), get<c10::optional<at::Layout>>(op.args[2]), get<c10::optional<at::Device>>(op.args[3]), get<c10::optional<bool>>(op.args[4]), get<c10::optional<at::MemoryFormat>>(op.args[5])));
  break;

case H_RANDINT:
  set(op.tensor, at::redispatch::randint(ks, get<int64_t>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<c10::optional<at::ScalarType>>(op.args[2]), get<c10::optional<at::Layout>>(op.args[3]), get<c10::optional<at::Device>>(op.args[4]), get<c10::optional<bool>>(op.args[5])));
  break;

case H_RANDINT_GENERATOR:
  set(op.tensor, at::redispatch::randint(ks, get<int64_t>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<c10::optional<at::Generator>>(op.args[2]), get<c10::optional<at::ScalarType>>(op.args[3]), get<c10::optional<at::Layout>>(op.args[4]), get<c10::optional<at::Device>>(op.args[5]), get<c10::optional<bool>>(op.args[6])));
  break;

case H_RANDINT_LOW:
  set(op.tensor, at::redispatch::randint(ks, get<int64_t>(op.args[0]), get<int64_t>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<c10::optional<at::ScalarType>>(op.args[3]), get<c10::optional<at::Layout>>(op.args[4]), get<c10::optional<at::Device>>(op.args[5]), get<c10::optional<bool>>(op.args[6])));
  break;

case H_RANDINT_LOW_GENERATOR:
  set(op.tensor, at::redispatch::randint(ks, get<int64_t>(op.args[0]), get<int64_t>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<c10::optional<at::Generator>>(op.args[3]), get<c10::optional<at::ScalarType>>(op.args[4]), get<c10::optional<at::Layout>>(op.args[5]), get<c10::optional<at::Device>>(op.args[6]), get<c10::optional<bool>>(op.args[7])));
  break;

case H_RANDINT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::randint_outf(ks, get<int64_t>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_RANDINT_GENERATOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::randint_outf(ks, get<int64_t>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<c10::optional<at::Generator>>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_RANDINT_LOW_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::randint_outf(ks, get<int64_t>(op.args[0]), get<int64_t>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_RANDINT_LOW_GENERATOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::randint_outf(ks, get<int64_t>(op.args[0]), get<int64_t>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<c10::optional<at::Generator>>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_RANDINT_LIKE:
  set(op.tensor, at::redispatch::randint_like(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<c10::optional<at::ScalarType>>(op.args[2]), get<c10::optional<at::Layout>>(op.args[3]), get<c10::optional<at::Device>>(op.args[4]), get<c10::optional<bool>>(op.args[5]), get<c10::optional<at::MemoryFormat>>(op.args[6])));
  break;

case H_RANDINT_LIKE_LOW_DTYPE:
  set(op.tensor, at::redispatch::randint_like(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<int64_t>(op.args[2]), get<c10::optional<at::ScalarType>>(op.args[3]), get<c10::optional<at::Layout>>(op.args[4]), get<c10::optional<at::Device>>(op.args[5]), get<c10::optional<bool>>(op.args[6]), get<c10::optional<at::MemoryFormat>>(op.args[7])));
  break;

case H_RANDN:
  set(op.tensor, at::redispatch::randn(ks, get<at::IntArrayRef>(op.args[0]), get<c10::optional<at::ScalarType>>(op.args[1]), get<c10::optional<at::Layout>>(op.args[2]), get<c10::optional<at::Device>>(op.args[3]), get<c10::optional<bool>>(op.args[4])));
  break;

case H_RANDN_GENERATOR:
  set(op.tensor, at::redispatch::randn(ks, get<at::IntArrayRef>(op.args[0]), get<c10::optional<at::Generator>>(op.args[1]), get<c10::optional<at::ScalarType>>(op.args[2]), get<c10::optional<at::Layout>>(op.args[3]), get<c10::optional<at::Device>>(op.args[4]), get<c10::optional<bool>>(op.args[5])));
  break;

case H_RANDN_NAMES:
  set(op.tensor, at::redispatch::randn(ks, get<at::IntArrayRef>(op.args[0]), get<c10::optional<at::DimnameList>>(op.args[1]), get<c10::optional<at::ScalarType>>(op.args[2]), get<c10::optional<at::Layout>>(op.args[3]), get<c10::optional<at::Device>>(op.args[4]), get<c10::optional<bool>>(op.args[5])));
  break;

case H_RANDN_GENERATOR_WITH_NAMES:
  set(op.tensor, at::redispatch::randn(ks, get<at::IntArrayRef>(op.args[0]), get<c10::optional<at::Generator>>(op.args[1]), get<c10::optional<at::DimnameList>>(op.args[2]), get<c10::optional<at::ScalarType>>(op.args[3]), get<c10::optional<at::Layout>>(op.args[4]), get<c10::optional<at::Device>>(op.args[5]), get<c10::optional<bool>>(op.args[6])));
  break;

case H_RANDN_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::randn_outf(ks, get<at::IntArrayRef>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_RANDN_GENERATOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::randn_outf(ks, get<at::IntArrayRef>(op.args[0]), get<c10::optional<at::Generator>>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_RANDN_LIKE:
  set(op.tensor, at::redispatch::randn_like(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::ScalarType>>(op.args[1]), get<c10::optional<at::Layout>>(op.args[2]), get<c10::optional<at::Device>>(op.args[3]), get<c10::optional<bool>>(op.args[4]), get<c10::optional<at::MemoryFormat>>(op.args[5])));
  break;

case H_RANDPERM:
  set(op.tensor, at::redispatch::randperm(ks, get<int64_t>(op.args[0]), get<c10::optional<at::ScalarType>>(op.args[1]), get<c10::optional<at::Layout>>(op.args[2]), get<c10::optional<at::Device>>(op.args[3]), get<c10::optional<bool>>(op.args[4])));
  break;

case H_RANDPERM_GENERATOR:
  set(op.tensor, at::redispatch::randperm(ks, get<int64_t>(op.args[0]), get<c10::optional<at::Generator>>(op.args[1]), get<c10::optional<at::ScalarType>>(op.args[2]), get<c10::optional<at::Layout>>(op.args[3]), get<c10::optional<at::Device>>(op.args[4]), get<c10::optional<bool>>(op.args[5])));
  break;

case H_RANDPERM_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::randperm_outf(ks, get<int64_t>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_RANDPERM_GENERATOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::randperm_outf(ks, get<int64_t>(op.args[0]), get<c10::optional<at::Generator>>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_RANGE_STEP:
  set(op.tensor, at::redispatch::range(ks, get<at::Scalar>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Scalar>(op.args[2]), get<c10::optional<at::ScalarType>>(op.args[3]), get<c10::optional<at::Layout>>(op.args[4]), get<c10::optional<at::Device>>(op.args[5]), get<c10::optional<bool>>(op.args[6])));
  break;

case H_RANGE:
  set(op.tensor, at::redispatch::range(ks, get<at::Scalar>(op.args[0]), get<at::Scalar>(op.args[1]), get<c10::optional<at::ScalarType>>(op.args[2]), get<c10::optional<at::Layout>>(op.args[3]), get<c10::optional<at::Device>>(op.args[4]), get<c10::optional<bool>>(op.args[5])));
  break;

case H_RANGE_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::range_outf(ks, get<at::Scalar>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Scalar>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_RAVEL:
  set(op.tensor, at::redispatch::ravel(ks, get<at::Tensor>(op.args[0])));
  break;

case H_RECIPROCAL_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::reciprocal_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_NEG:
  set(op.tensor, at::redispatch::neg(ks, get<at::Tensor>(op.args[0])));
  break;

case H_NEG_:
  init_update_in_place(op.tensor);
  at::redispatch::neg_(ks, get<at::Tensor>(op.args[0]));
  end_update_in_place(op.tensor);
  break;

case H_NEG_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::neg_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_NEGATIVE:
  set(op.tensor, at::redispatch::negative(ks, get<at::Tensor>(op.args[0])));
  break;

case H_NEGATIVE_:
  init_update_in_place(op.tensor);
  at::redispatch::negative_(ks, get<at::Tensor>(op.args[0]));
  end_update_in_place(op.tensor);
  break;

case H_NEGATIVE_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::negative_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_REPEAT:
  set(op.tensor, at::redispatch::repeat(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1])));
  break;

case H_REPEAT_INTERLEAVE_TENSOR:
  set(op.tensor, at::redispatch::repeat_interleave(ks, get<at::Tensor>(op.args[0])));
  break;

case H_REPEAT_INTERLEAVE_SELF_TENSOR:
  set(op.tensor, at::redispatch::repeat_interleave(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<int64_t>>(op.args[2])));
  break;

case H_REPEAT_INTERLEAVE_SELF_INT:
  set(op.tensor, at::redispatch::repeat_interleave(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<c10::optional<int64_t>>(op.args[2])));
  break;

case H_RESHAPE:
  set(op.tensor, at::redispatch::reshape(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1])));
  break;

case H__MKLDNN_RESHAPE:
  set(op.tensor, at::redispatch::_mkldnn_reshape(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1])));
  break;

case H_RESHAPE_AS:
  set(op.tensor, at::redispatch::reshape_as(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_ROUND_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::round_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_RRELU:
  set(op.tensor, at::redispatch::rrelu(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Scalar>(op.args[2]), get<bool>(op.args[3]), get<c10::optional<at::Generator>>(op.args[4])));
  break;

case H_RRELU_:
  init_update_in_place(op.tensor);
  at::redispatch::rrelu_(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Scalar>(op.args[2]), get<bool>(op.args[3]), get<c10::optional<at::Generator>>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_RELU:
  set(op.tensor, at::redispatch::relu(ks, get<at::Tensor>(op.args[0])));
  break;

case H_RELU_:
  init_update_in_place(op.tensor);
  at::redispatch::relu_(ks, get<at::Tensor>(op.args[0]));
  end_update_in_place(op.tensor);
  break;

case H_RELU6:
  set(op.tensor, at::redispatch::relu6(ks, get<at::Tensor>(op.args[0])));
  break;

case H_RELU6_:
  init_update_in_place(op.tensor);
  at::redispatch::relu6_(ks, get<at::Tensor>(op.args[0]));
  end_update_in_place(op.tensor);
  break;

case H_PRELU:
  set(op.tensor, at::redispatch::prelu(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

// skip std::tuple<at::Tensor,at::Tensor> prelu_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight)

case H_GELU:
  set(op.tensor, at::redispatch::gelu(ks, get<at::Tensor>(op.args[0])));
  break;

case H_GELU_BACKWARD:
  set(op.tensor, at::redispatch::gelu_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_INFINITELY_DIFFERENTIABLE_GELU_BACKWARD:
  set(op.tensor, at::redispatch::infinitely_differentiable_gelu_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_HARDSHRINK:
  set(op.tensor, at::redispatch::hardshrink(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1])));
  break;

case H_HARDSHRINK_BACKWARD:
  set(op.tensor, at::redispatch::hardshrink_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Scalar>(op.args[2])));
  break;

case H_RSQRT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::rsqrt_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_SELECT_DIMNAME:
  set(op.tensor, at::redispatch::select(ks, get<at::Tensor>(op.args[0]), get<at::Dimname>(op.args[1]), get<int64_t>(op.args[2])));
  break;

case H_SELECT_INT:
  set(op.tensor, at::redispatch::select(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<int64_t>(op.args[2])));
  break;

case H_SELECT_BACKWARD:
  set(op.tensor, at::redispatch::select_backward(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<int64_t>(op.args[2]), get<int64_t>(op.args[3])));
  break;

case H_SELU:
  set(op.tensor, at::redispatch::selu(ks, get<at::Tensor>(op.args[0])));
  break;

case H_SELU_:
  init_update_in_place(op.tensor);
  at::redispatch::selu_(ks, get<at::Tensor>(op.args[0]));
  end_update_in_place(op.tensor);
  break;

case H_CELU:
  set(op.tensor, at::redispatch::celu(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1])));
  break;

case H_CELU_:
  init_update_in_place(op.tensor);
  at::redispatch::celu_(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_SILU:
  set(op.tensor, at::redispatch::silu(ks, get<at::Tensor>(op.args[0])));
  break;

case H_SILU_:
  init_update_in_place(op.tensor);
  at::redispatch::silu_(ks, get<at::Tensor>(op.args[0]));
  end_update_in_place(op.tensor);
  break;

case H_SILU_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::silu_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_SILU_BACKWARD:
  set(op.tensor, at::redispatch::silu_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_SIGMOID:
  set(op.tensor, at::redispatch::sigmoid(ks, get<at::Tensor>(op.args[0])));
  break;

case H_SIGMOID_:
  init_update_in_place(op.tensor);
  at::redispatch::sigmoid_(ks, get<at::Tensor>(op.args[0]));
  end_update_in_place(op.tensor);
  break;

case H_SIGMOID_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::sigmoid_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_LOGIT:
  set(op.tensor, at::redispatch::logit(ks, get<at::Tensor>(op.args[0]), get<c10::optional<double>>(op.args[1])));
  break;

case H_LOGIT_:
  init_update_in_place(op.tensor);
  at::redispatch::logit_(ks, get<at::Tensor>(op.args[0]), get<c10::optional<double>>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_LOGIT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::logit_outf(ks, get<at::Tensor>(op.args[0]), get<c10::optional<double>>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_SIN_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::sin_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_SINC_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::sinc_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_SINH_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::sinh_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_DETACH:
  set(op.tensor, at::redispatch::detach(ks, get<at::Tensor>(op.args[0])));
  break;

case H_DETACH_:
  init_update_in_place(op.tensor);
  at::redispatch::detach_(ks, get<at::Tensor>(op.args[0]));
  end_update_in_place(op.tensor);
  break;

// skip int64_t size(const at::Tensor & self, at::Dimname dim)

case H_SLICE_TENSOR:
  set(op.tensor, at::redispatch::slice(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<c10::optional<int64_t>>(op.args[2]), get<c10::optional<int64_t>>(op.args[3]), get<int64_t>(op.args[4])));
  break;

case H_SLICE_BACKWARD:
  set(op.tensor, at::redispatch::slice_backward(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<int64_t>(op.args[2]), get<int64_t>(op.args[3]), get<int64_t>(op.args[4]), get<int64_t>(op.args[5])));
  break;

// skip std::tuple<at::Tensor,at::Tensor> slogdet(const at::Tensor & self)

case H_SMM:
  set(op.tensor, at::redispatch::smm(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_SOFTMAX_INT:
  set(op.tensor, at::redispatch::softmax(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<c10::optional<at::ScalarType>>(op.args[2])));
  break;

case H_SOFTMAX_DIMNAME:
  set(op.tensor, at::redispatch::softmax(ks, get<at::Tensor>(op.args[0]), get<at::Dimname>(op.args[1]), get<c10::optional<at::ScalarType>>(op.args[2])));
  break;

case H__SOFTMAX:
  set(op.tensor, at::redispatch::_softmax(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<bool>(op.args[2])));
  break;

case H__SOFTMAX_BACKWARD_DATA:
  set(op.tensor, at::redispatch::_softmax_backward_data(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<int64_t>(op.args[2]), get<at::Tensor>(op.args[3])));
  break;

// skip std::vector<at::Tensor> unsafe_split(const at::Tensor & self, int64_t split_size, int64_t dim)

// skip std::vector<at::Tensor> split(const at::Tensor & self, int64_t split_size, int64_t dim)

// skip std::vector<at::Tensor> unsafe_split_with_sizes(const at::Tensor & self, at::IntArrayRef split_sizes, int64_t dim)

// skip std::vector<at::Tensor> split_with_sizes(const at::Tensor & self, at::IntArrayRef split_sizes, int64_t dim)

// skip std::vector<at::Tensor> hsplit(const at::Tensor & self, int64_t sections)

// skip std::vector<at::Tensor> hsplit(const at::Tensor & self, at::IntArrayRef indices)

// skip std::vector<at::Tensor> vsplit(const at::Tensor & self, int64_t sections)

// skip std::vector<at::Tensor> vsplit(const at::Tensor & self, at::IntArrayRef indices)

// skip std::vector<at::Tensor> dsplit(const at::Tensor & self, int64_t sections)

// skip std::vector<at::Tensor> dsplit(const at::Tensor & self, at::IntArrayRef indices)

case H_SQUEEZE:
  set(op.tensor, at::redispatch::squeeze(ks, get<at::Tensor>(op.args[0])));
  break;

case H_SQUEEZE_DIM:
  set(op.tensor, at::redispatch::squeeze(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1])));
  break;

case H_SQUEEZE_DIMNAME:
  set(op.tensor, at::redispatch::squeeze(ks, get<at::Tensor>(op.args[0]), get<at::Dimname>(op.args[1])));
  break;

case H_SQUEEZE_:
  init_update_in_place(op.tensor);
  at::redispatch::squeeze_(ks, get<at::Tensor>(op.args[0]));
  end_update_in_place(op.tensor);
  break;

case H_SQUEEZE__DIM:
  init_update_in_place(op.tensor);
  at::redispatch::squeeze_(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_SQUEEZE__DIMNAME:
  init_update_in_place(op.tensor);
  at::redispatch::squeeze_(ks, get<at::Tensor>(op.args[0]), get<at::Dimname>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_SSPADDMM:
  set(op.tensor, at::redispatch::sspaddmm(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Scalar>(op.args[3]), get<at::Scalar>(op.args[4])));
  break;

case H_SSPADDMM_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::sspaddmm_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Scalar>(op.args[3]), get<at::Scalar>(op.args[4]), get<at::Tensor>(op.args[5]));
  end_update_in_place(op.tensor);
  break;

case H_STACK:
  set(op.tensor, at::redispatch::stack(ks, get<at::TensorList>(op.args[0]), get<int64_t>(op.args[1])));
  break;

case H_STACK_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::stack_outf(ks, get<at::TensorList>(op.args[0]), get<int64_t>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H__STACK:
  set(op.tensor, at::redispatch::_stack(ks, get<at::TensorList>(op.args[0]), get<int64_t>(op.args[1])));
  break;

case H__STACK_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::_stack_outf(ks, get<at::TensorList>(op.args[0]), get<int64_t>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_HSTACK:
  set(op.tensor, at::redispatch::hstack(ks, get<at::TensorList>(op.args[0])));
  break;

case H_HSTACK_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::hstack_outf(ks, get<at::TensorList>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_VSTACK:
  set(op.tensor, at::redispatch::vstack(ks, get<at::TensorList>(op.args[0])));
  break;

case H_VSTACK_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::vstack_outf(ks, get<at::TensorList>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_DSTACK:
  set(op.tensor, at::redispatch::dstack(ks, get<at::TensorList>(op.args[0])));
  break;

case H_DSTACK_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::dstack_outf(ks, get<at::TensorList>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_STFT:
  set(op.tensor, at::redispatch::stft(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<c10::optional<int64_t>>(op.args[2]), get<c10::optional<int64_t>>(op.args[3]), get<c10::optional<at::Tensor>>(op.args[4]), get<bool>(op.args[5]), get<c10::optional<bool>>(op.args[6]), get<c10::optional<bool>>(op.args[7])));
  break;

case H_ISTFT:
  set(op.tensor, at::redispatch::istft(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<c10::optional<int64_t>>(op.args[2]), get<c10::optional<int64_t>>(op.args[3]), get<c10::optional<at::Tensor>>(op.args[4]), get<bool>(op.args[5]), get<bool>(op.args[6]), get<c10::optional<bool>>(op.args[7]), get<c10::optional<int64_t>>(op.args[8]), get<bool>(op.args[9])));
  break;

// skip int64_t stride(const at::Tensor & self, at::Dimname dim)

case H_SUM:
  set(op.tensor, at::redispatch::sum(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::ScalarType>>(op.args[1])));
  break;

case H_SUM_DIM_INTLIST:
  set(op.tensor, at::redispatch::sum(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<bool>(op.args[2]), get<c10::optional<at::ScalarType>>(op.args[3])));
  break;

case H_SUM_DIM_DIMNAMELIST:
  set(op.tensor, at::redispatch::sum(ks, get<at::Tensor>(op.args[0]), get<at::DimnameList>(op.args[1]), get<bool>(op.args[2]), get<c10::optional<at::ScalarType>>(op.args[3])));
  break;

case H_SUM_INTLIST_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::sum_outf(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<bool>(op.args[2]), get<c10::optional<at::ScalarType>>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_SUM_DIMNAMELIST_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::sum_outf(ks, get<at::Tensor>(op.args[0]), get<at::DimnameList>(op.args[1]), get<bool>(op.args[2]), get<c10::optional<at::ScalarType>>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_NANSUM:
  set(op.tensor, at::redispatch::nansum(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::ScalarType>>(op.args[1])));
  break;

case H_NANSUM_DIM_INTLIST:
  set(op.tensor, at::redispatch::nansum(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<bool>(op.args[2]), get<c10::optional<at::ScalarType>>(op.args[3])));
  break;

case H_NANSUM_INTLIST_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::nansum_outf(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<bool>(op.args[2]), get<c10::optional<at::ScalarType>>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_SUM_TO_SIZE:
  set(op.tensor, at::redispatch::sum_to_size(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1])));
  break;

case H_SQRT:
  set(op.tensor, at::redispatch::sqrt(ks, get<at::Tensor>(op.args[0])));
  break;

case H_SQRT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::sqrt_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_SQUARE:
  set(op.tensor, at::redispatch::square(ks, get<at::Tensor>(op.args[0])));
  break;

case H_SQUARE_:
  init_update_in_place(op.tensor);
  at::redispatch::square_(ks, get<at::Tensor>(op.args[0]));
  end_update_in_place(op.tensor);
  break;

case H_SQUARE_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::square_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_STD:
  set(op.tensor, at::redispatch::std(ks, get<at::Tensor>(op.args[0]), get<bool>(op.args[1])));
  break;

case H_STD_DIM:
  set(op.tensor, at::redispatch::std(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<bool>(op.args[2]), get<bool>(op.args[3])));
  break;

case H_STD_CORRECTION:
  set(op.tensor, at::redispatch::std(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::IntArrayRef>>(op.args[1]), get<c10::optional<int64_t>>(op.args[2]), get<bool>(op.args[3])));
  break;

// skip std::tuple<at::Tensor,at::Tensor> std_mean(const at::Tensor & self, bool unbiased)

// skip std::tuple<at::Tensor,at::Tensor> std_mean(const at::Tensor & self, at::IntArrayRef dim, bool unbiased, bool keepdim)

// skip std::tuple<at::Tensor,at::Tensor> std_mean(const at::Tensor & self, c10::optional<at::IntArrayRef> dim, c10::optional<int64_t> correction, bool keepdim)

// skip std::tuple<at::Tensor,at::Tensor> std_mean(const at::Tensor & self, at::DimnameList dim, bool unbiased, bool keepdim)

// skip std::tuple<at::Tensor,at::Tensor> std_mean(const at::Tensor & self, at::DimnameList dim, c10::optional<int64_t> correction, bool keepdim)

case H_STD_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::std_outf(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<bool>(op.args[2]), get<bool>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_STD_CORRECTION_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::std_outf(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::IntArrayRef>>(op.args[1]), get<c10::optional<int64_t>>(op.args[2]), get<bool>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_STD_NAMES_DIM:
  set(op.tensor, at::redispatch::std(ks, get<at::Tensor>(op.args[0]), get<at::DimnameList>(op.args[1]), get<bool>(op.args[2]), get<bool>(op.args[3])));
  break;

case H_STD_NAMES_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::std_outf(ks, get<at::Tensor>(op.args[0]), get<at::DimnameList>(op.args[1]), get<bool>(op.args[2]), get<bool>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_STD_CORRECTION_NAMES:
  set(op.tensor, at::redispatch::std(ks, get<at::Tensor>(op.args[0]), get<at::DimnameList>(op.args[1]), get<c10::optional<int64_t>>(op.args[2]), get<bool>(op.args[3])));
  break;

case H_STD_CORRECTION_NAMES_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::std_outf(ks, get<at::Tensor>(op.args[0]), get<at::DimnameList>(op.args[1]), get<c10::optional<int64_t>>(op.args[2]), get<bool>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_PROD:
  set(op.tensor, at::redispatch::prod(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::ScalarType>>(op.args[1])));
  break;

case H_PROD_DIM_INT:
  set(op.tensor, at::redispatch::prod(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<bool>(op.args[2]), get<c10::optional<at::ScalarType>>(op.args[3])));
  break;

case H_PROD_INT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::prod_outf(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<bool>(op.args[2]), get<c10::optional<at::ScalarType>>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_PROD_DIM_DIMNAME:
  set(op.tensor, at::redispatch::prod(ks, get<at::Tensor>(op.args[0]), get<at::Dimname>(op.args[1]), get<bool>(op.args[2]), get<c10::optional<at::ScalarType>>(op.args[3])));
  break;

case H_PROD_DIMNAME_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::prod_outf(ks, get<at::Tensor>(op.args[0]), get<at::Dimname>(op.args[1]), get<bool>(op.args[2]), get<c10::optional<at::ScalarType>>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_T:
  set(op.tensor, at::redispatch::t(ks, get<at::Tensor>(op.args[0])));
  break;

case H_T_:
  init_update_in_place(op.tensor);
  at::redispatch::t_(ks, get<at::Tensor>(op.args[0]));
  end_update_in_place(op.tensor);
  break;

case H_TAN_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::tan_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_TANH:
  set(op.tensor, at::redispatch::tanh(ks, get<at::Tensor>(op.args[0])));
  break;

case H_TANH_:
  init_update_in_place(op.tensor);
  at::redispatch::tanh_(ks, get<at::Tensor>(op.args[0]));
  end_update_in_place(op.tensor);
  break;

case H_TANH_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::tanh_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_TENSORDOT:
  set(op.tensor, at::redispatch::tensordot(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::IntArrayRef>(op.args[3])));
  break;

case H_TENSORDOT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::tensordot_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_THRESHOLD:
  set(op.tensor, at::redispatch::threshold(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Scalar>(op.args[2])));
  break;

case H_THRESHOLD_:
  init_update_in_place(op.tensor);
  at::redispatch::threshold_(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Scalar>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_THRESHOLD_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::threshold_outf(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Scalar>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_THRESHOLD_BACKWARD:
  set(op.tensor, at::redispatch::threshold_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Scalar>(op.args[2])));
  break;

case H_TILE:
  set(op.tensor, at::redispatch::tile(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1])));
  break;

case H_TRANSPOSE_INT:
  set(op.tensor, at::redispatch::transpose(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<int64_t>(op.args[2])));
  break;

case H_TRANSPOSE_DIMNAME:
  set(op.tensor, at::redispatch::transpose(ks, get<at::Tensor>(op.args[0]), get<at::Dimname>(op.args[1]), get<at::Dimname>(op.args[2])));
  break;

case H__MKLDNN_TRANSPOSE:
  set(op.tensor, at::redispatch::_mkldnn_transpose(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<int64_t>(op.args[2])));
  break;

case H_TRANSPOSE_:
  init_update_in_place(op.tensor);
  at::redispatch::transpose_(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<int64_t>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H__MKLDNN_TRANSPOSE_:
  init_update_in_place(op.tensor);
  at::redispatch::_mkldnn_transpose_(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<int64_t>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_ONE_HOT:
  set(op.tensor, at::redispatch::one_hot(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1])));
  break;

case H_FLIP:
  set(op.tensor, at::redispatch::flip(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1])));
  break;

case H_FLIPLR:
  set(op.tensor, at::redispatch::fliplr(ks, get<at::Tensor>(op.args[0])));
  break;

case H_FLIPUD:
  set(op.tensor, at::redispatch::flipud(ks, get<at::Tensor>(op.args[0])));
  break;

case H_ROLL:
  set(op.tensor, at::redispatch::roll(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<at::IntArrayRef>(op.args[2])));
  break;

case H_ROT90:
  set(op.tensor, at::redispatch::rot90(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<at::IntArrayRef>(op.args[2])));
  break;

case H_TRAPZ_X:
  set(op.tensor, at::redispatch::trapz(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<int64_t>(op.args[2])));
  break;

case H_TRAPZ_DX:
  set(op.tensor, at::redispatch::trapz(ks, get<at::Tensor>(op.args[0]), get<double>(op.args[1]), get<int64_t>(op.args[2])));
  break;

case H__TRILINEAR:
  set(op.tensor, at::redispatch::_trilinear(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<at::IntArrayRef>(op.args[6]), get<int64_t>(op.args[7])));
  break;

case H_TRIPLET_MARGIN_LOSS:
  set(op.tensor, at::redispatch::triplet_margin_loss(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<double>(op.args[3]), get<double>(op.args[4]), get<double>(op.args[5]), get<bool>(op.args[6]), get<int64_t>(op.args[7])));
  break;

case H_TRUNC:
  set(op.tensor, at::redispatch::trunc(ks, get<at::Tensor>(op.args[0])));
  break;

case H_TRUNC_:
  init_update_in_place(op.tensor);
  at::redispatch::trunc_(ks, get<at::Tensor>(op.args[0]));
  end_update_in_place(op.tensor);
  break;

case H_TRUNC_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::trunc_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_FIX:
  set(op.tensor, at::redispatch::fix(ks, get<at::Tensor>(op.args[0])));
  break;

case H_FIX_:
  init_update_in_place(op.tensor);
  at::redispatch::fix_(ks, get<at::Tensor>(op.args[0]));
  end_update_in_place(op.tensor);
  break;

case H_FIX_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::fix_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_TYPE_AS:
  set(op.tensor, at::redispatch::type_as(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

// skip bool _has_compatible_shallow_copy_type(const at::Tensor & self, const at::Tensor & from)

// skip std::tuple<at::Tensor,at::Tensor> _unique(const at::Tensor & self, bool sorted, bool return_inverse)

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor> unique_dim(const at::Tensor & self, int64_t dim, bool sorted, bool return_inverse, bool return_counts)

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor> unique_consecutive(const at::Tensor & self, bool return_inverse, bool return_counts, c10::optional<int64_t> dim)

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor> unique_dim_consecutive(const at::Tensor & self, int64_t dim, bool return_inverse, bool return_counts)

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor> _unique2(const at::Tensor & self, bool sorted, bool return_inverse, bool return_counts)

case H__UNSAFE_VIEW:
  set(op.tensor, at::redispatch::_unsafe_view(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1])));
  break;

case H_UNSQUEEZE:
  set(op.tensor, at::redispatch::unsqueeze(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1])));
  break;

case H_UNSQUEEZE_:
  init_update_in_place(op.tensor);
  at::redispatch::unsqueeze_(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_VANDER:
  set(op.tensor, at::redispatch::vander(ks, get<at::Tensor>(op.args[0]), get<c10::optional<int64_t>>(op.args[1]), get<bool>(op.args[2])));
  break;

case H_VAR:
  set(op.tensor, at::redispatch::var(ks, get<at::Tensor>(op.args[0]), get<bool>(op.args[1])));
  break;

case H_VAR_DIM:
  set(op.tensor, at::redispatch::var(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<bool>(op.args[2]), get<bool>(op.args[3])));
  break;

case H_VAR_CORRECTION:
  set(op.tensor, at::redispatch::var(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::IntArrayRef>>(op.args[1]), get<c10::optional<int64_t>>(op.args[2]), get<bool>(op.args[3])));
  break;

case H_VAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::var_outf(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<bool>(op.args[2]), get<bool>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_VAR_CORRECTION_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::var_outf(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::IntArrayRef>>(op.args[1]), get<c10::optional<int64_t>>(op.args[2]), get<bool>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_VAR_NAMES_DIM:
  set(op.tensor, at::redispatch::var(ks, get<at::Tensor>(op.args[0]), get<at::DimnameList>(op.args[1]), get<bool>(op.args[2]), get<bool>(op.args[3])));
  break;

case H_VAR_NAMES_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::var_outf(ks, get<at::Tensor>(op.args[0]), get<at::DimnameList>(op.args[1]), get<bool>(op.args[2]), get<bool>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_VAR_CORRECTION_NAMES:
  set(op.tensor, at::redispatch::var(ks, get<at::Tensor>(op.args[0]), get<at::DimnameList>(op.args[1]), get<c10::optional<int64_t>>(op.args[2]), get<bool>(op.args[3])));
  break;

case H_VAR_CORRECTION_NAMES_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::var_outf(ks, get<at::Tensor>(op.args[0]), get<at::DimnameList>(op.args[1]), get<c10::optional<int64_t>>(op.args[2]), get<bool>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

// skip std::tuple<at::Tensor,at::Tensor> var_mean(const at::Tensor & self, bool unbiased)

// skip std::tuple<at::Tensor,at::Tensor> var_mean(const at::Tensor & self, at::IntArrayRef dim, bool unbiased, bool keepdim)

// skip std::tuple<at::Tensor,at::Tensor> var_mean(const at::Tensor & self, c10::optional<at::IntArrayRef> dim, c10::optional<int64_t> correction, bool keepdim)

// skip std::tuple<at::Tensor,at::Tensor> var_mean(const at::Tensor & self, at::DimnameList dim, bool unbiased, bool keepdim)

// skip std::tuple<at::Tensor,at::Tensor> var_mean(const at::Tensor & self, at::DimnameList dim, c10::optional<int64_t> correction, bool keepdim)

case H_VIEW_AS:
  set(op.tensor, at::redispatch::view_as(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_WHERE_SELF:
  set(op.tensor, at::redispatch::where(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2])));
  break;

case H_WHERE_SCALARSELF:
  set(op.tensor, at::redispatch::where(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Tensor>(op.args[2])));
  break;

case H_WHERE_SCALAROTHER:
  set(op.tensor, at::redispatch::where(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Scalar>(op.args[2])));
  break;

case H_WHERE_SCALAR:
  set(op.tensor, at::redispatch::where(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Scalar>(op.args[2])));
  break;

// skip std::vector<at::Tensor> where(const at::Tensor & condition)

case H__S_WHERE:
  set(op.tensor, at::redispatch::_s_where(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2])));
  break;

case H_NORM_EXCEPT_DIM:
  set(op.tensor, at::redispatch::norm_except_dim(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<int64_t>(op.args[2])));
  break;

case H__WEIGHT_NORM:
  set(op.tensor, at::redispatch::_weight_norm(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<int64_t>(op.args[2])));
  break;

// skip std::tuple<at::Tensor,at::Tensor> _weight_norm_cuda_interface(const at::Tensor & v, const at::Tensor & g, int64_t dim)

// skip std::tuple<at::Tensor,at::Tensor> _weight_norm_cuda_interface_backward(const at::Tensor & grad_w, const at::Tensor & saved_v, const at::Tensor & saved_g, const at::Tensor & saved_norms, int64_t dim)

// skip std::tuple<at::Tensor,at::Tensor> _weight_norm_differentiable_backward(const at::Tensor & grad_w, const at::Tensor & saved_v, const at::Tensor & saved_g, const at::Tensor & saved_norms, int64_t dim)

case H_ZEROS_NAMES:
  set(op.tensor, at::redispatch::zeros(ks, get<at::IntArrayRef>(op.args[0]), get<c10::optional<at::DimnameList>>(op.args[1]), get<c10::optional<at::ScalarType>>(op.args[2]), get<c10::optional<at::Layout>>(op.args[3]), get<c10::optional<at::Device>>(op.args[4]), get<c10::optional<bool>>(op.args[5])));
  break;

case H_ZEROS:
  set(op.tensor, at::redispatch::zeros(ks, get<at::IntArrayRef>(op.args[0]), get<c10::optional<at::ScalarType>>(op.args[1]), get<c10::optional<at::Layout>>(op.args[2]), get<c10::optional<at::Device>>(op.args[3]), get<c10::optional<bool>>(op.args[4])));
  break;

case H_ZEROS_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::zeros_outf(ks, get<at::IntArrayRef>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_ZEROS_LIKE:
  set(op.tensor, at::redispatch::zeros_like(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::ScalarType>>(op.args[1]), get<c10::optional<at::Layout>>(op.args[2]), get<c10::optional<at::Device>>(op.args[3]), get<c10::optional<bool>>(op.args[4]), get<c10::optional<at::MemoryFormat>>(op.args[5])));
  break;

case H__STANDARD_GAMMA_GRAD:
  set(op.tensor, at::redispatch::_standard_gamma_grad(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H__STANDARD_GAMMA:
  set(op.tensor, at::redispatch::_standard_gamma(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::Generator>>(op.args[1])));
  break;

case H__DIRICHLET_GRAD:
  set(op.tensor, at::redispatch::_dirichlet_grad(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2])));
  break;

case H__SAMPLE_DIRICHLET:
  set(op.tensor, at::redispatch::_sample_dirichlet(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::Generator>>(op.args[1])));
  break;

case H_POISSON:
  set(op.tensor, at::redispatch::poisson(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::Generator>>(op.args[1])));
  break;

case H_BINOMIAL:
  set(op.tensor, at::redispatch::binomial(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<at::Generator>>(op.args[2])));
  break;

case H_NATIVE_NORM:
  set(op.tensor, at::redispatch::native_norm(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1])));
  break;

case H_NATIVE_NORM_SCALAROPT_DIM_DTYPE:
  set(op.tensor, at::redispatch::native_norm(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::Scalar>>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<bool>(op.args[3]), get<c10::optional<at::ScalarType>>(op.args[4])));
  break;

case H__SPARSE_SUM:
  set(op.tensor, at::redispatch::_sparse_sum(ks, get<at::Tensor>(op.args[0])));
  break;

case H__SPARSE_SUM_DTYPE:
  set(op.tensor, at::redispatch::_sparse_sum(ks, get<at::Tensor>(op.args[0]), get<at::ScalarType>(op.args[1])));
  break;

case H__SPARSE_SUM_DIM:
  set(op.tensor, at::redispatch::_sparse_sum(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1])));
  break;

case H__SPARSE_SUM_DIM_DTYPE:
  set(op.tensor, at::redispatch::_sparse_sum(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<at::ScalarType>(op.args[2])));
  break;

case H__SPARSE_SUM_BACKWARD:
  set(op.tensor, at::redispatch::_sparse_sum_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2])));
  break;

case H__SPARSE_SOFTMAX_INT:
  set(op.tensor, at::redispatch::_sparse_softmax(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<c10::optional<at::ScalarType>>(op.args[2])));
  break;

case H__SPARSE_SOFTMAX_DIMNAME:
  set(op.tensor, at::redispatch::_sparse_softmax(ks, get<at::Tensor>(op.args[0]), get<at::Dimname>(op.args[1]), get<c10::optional<at::ScalarType>>(op.args[2])));
  break;

case H__SPARSE_SOFTMAX:
  set(op.tensor, at::redispatch::_sparse_softmax(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<bool>(op.args[2])));
  break;

case H__SPARSE_SOFTMAX_BACKWARD_DATA:
  set(op.tensor, at::redispatch::_sparse_softmax_backward_data(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<int64_t>(op.args[2]), get<at::Tensor>(op.args[3])));
  break;

case H__SPARSE_LOG_SOFTMAX_INT:
  set(op.tensor, at::redispatch::_sparse_log_softmax(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<c10::optional<at::ScalarType>>(op.args[2])));
  break;

case H__SPARSE_LOG_SOFTMAX_DIMNAME:
  set(op.tensor, at::redispatch::_sparse_log_softmax(ks, get<at::Tensor>(op.args[0]), get<at::Dimname>(op.args[1]), get<c10::optional<at::ScalarType>>(op.args[2])));
  break;

case H__SPARSE_LOG_SOFTMAX:
  set(op.tensor, at::redispatch::_sparse_log_softmax(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<bool>(op.args[2])));
  break;

case H__SPARSE_LOG_SOFTMAX_BACKWARD_DATA:
  set(op.tensor, at::redispatch::_sparse_log_softmax_backward_data(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<int64_t>(op.args[2]), get<at::Tensor>(op.args[3])));
  break;

case H_NORM_SCALAROPT_DTYPE:
  set(op.tensor, at::redispatch::norm(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::Scalar>>(op.args[1]), get<at::ScalarType>(op.args[2])));
  break;

case H_NORM_SCALAR:
  set(op.tensor, at::redispatch::norm(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1])));
  break;

case H_NORM_SCALAROPT_DIM_DTYPE:
  set(op.tensor, at::redispatch::norm(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::Scalar>>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<bool>(op.args[3]), get<at::ScalarType>(op.args[4])));
  break;

case H_NORM_SCALAROPT_DIM:
  set(op.tensor, at::redispatch::norm(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::Scalar>>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<bool>(op.args[3])));
  break;

case H_NORM_DTYPE_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::norm_outf(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::Scalar>>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<bool>(op.args[3]), get<at::ScalarType>(op.args[4]), get<at::Tensor>(op.args[5]));
  end_update_in_place(op.tensor);
  break;

case H_NORM_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::norm_outf(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::Scalar>>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<bool>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_NORM_NAMES_SCALAROPT_DIM_DTYPE:
  set(op.tensor, at::redispatch::norm(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::Scalar>>(op.args[1]), get<at::DimnameList>(op.args[2]), get<bool>(op.args[3]), get<at::ScalarType>(op.args[4])));
  break;

case H_NORM_NAMES_SCALAROPT_DIM:
  set(op.tensor, at::redispatch::norm(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::Scalar>>(op.args[1]), get<at::DimnameList>(op.args[2]), get<bool>(op.args[3])));
  break;

case H_NORM_NAMES_DTYPE_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::norm_outf(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::Scalar>>(op.args[1]), get<at::DimnameList>(op.args[2]), get<bool>(op.args[3]), get<at::ScalarType>(op.args[4]), get<at::Tensor>(op.args[5]));
  end_update_in_place(op.tensor);
  break;

case H_NORM_NAMES_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::norm_outf(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::Scalar>>(op.args[1]), get<at::DimnameList>(op.args[2]), get<bool>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

// skip std::tuple<at::Tensor,at::Tensor> frexp(const at::Tensor & self)

// skip std::tuple<at::Tensor &,at::Tensor &> frexp_outf(const at::Tensor & self, at::Tensor & mantissa, at::Tensor & exponent)

case H_FROBENIUS_NORM:
  set(op.tensor, at::redispatch::frobenius_norm(ks, get<at::Tensor>(op.args[0])));
  break;

case H_FROBENIUS_NORM_DIM:
  set(op.tensor, at::redispatch::frobenius_norm(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<bool>(op.args[2])));
  break;

case H_FROBENIUS_NORM_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::frobenius_norm_outf(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<bool>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_NUCLEAR_NORM:
  set(op.tensor, at::redispatch::nuclear_norm(ks, get<at::Tensor>(op.args[0]), get<bool>(op.args[1])));
  break;

case H_NUCLEAR_NORM_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::nuclear_norm_outf(ks, get<at::Tensor>(op.args[0]), get<bool>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_NUCLEAR_NORM_DIM:
  set(op.tensor, at::redispatch::nuclear_norm(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<bool>(op.args[2])));
  break;

case H_NUCLEAR_NORM_DIM_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::nuclear_norm_outf(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<bool>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_CLONE:
  set(op.tensor, at::redispatch::clone(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::MemoryFormat>>(op.args[1])));
  break;

case H_POSITIVE:
  set(op.tensor, at::redispatch::positive(ks, get<at::Tensor>(op.args[0])));
  break;

case H_RESIZE_AS_:
  init_update_in_place(op.tensor);
  at::redispatch::resize_as_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<at::MemoryFormat>>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_RESIZE_AS_SPARSE_:
  init_update_in_place(op.tensor);
  at::redispatch::resize_as_sparse_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_ZERO_:
  init_update_in_place(op.tensor);
  at::redispatch::zero_(ks, get<at::Tensor>(op.args[0]));
  end_update_in_place(op.tensor);
  break;

case H_SUB_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::sub_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Scalar>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_SUB_TENSOR:
  set(op.tensor, at::redispatch::sub(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Scalar>(op.args[2])));
  break;

case H_SUB__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::sub_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Scalar>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_SUB_SCALAR:
  set(op.tensor, at::redispatch::sub(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Scalar>(op.args[2])));
  break;

case H_SUB__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::sub_(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Scalar>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_SUBTRACT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::subtract_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Scalar>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_SUBTRACT_TENSOR:
  set(op.tensor, at::redispatch::subtract(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Scalar>(op.args[2])));
  break;

case H_SUBTRACT__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::subtract_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Scalar>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_SUBTRACT_SCALAR:
  set(op.tensor, at::redispatch::subtract(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Scalar>(op.args[2])));
  break;

case H_SUBTRACT__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::subtract_(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Scalar>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_RSUB_TENSOR:
  set(op.tensor, at::redispatch::rsub(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Scalar>(op.args[2])));
  break;

case H_HEAVISIDE_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::heaviside_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_HEAVISIDE:
  set(op.tensor, at::redispatch::heaviside(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_HEAVISIDE_:
  init_update_in_place(op.tensor);
  at::redispatch::heaviside_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_RSUB_SCALAR:
  set(op.tensor, at::redispatch::rsub(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Scalar>(op.args[2])));
  break;

case H__SPARSE_ADDMM:
  set(op.tensor, at::redispatch::_sparse_addmm(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Scalar>(op.args[3]), get<at::Scalar>(op.args[4])));
  break;

case H_ADDMM_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::addmm_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Scalar>(op.args[3]), get<at::Scalar>(op.args[4]), get<at::Tensor>(op.args[5]));
  end_update_in_place(op.tensor);
  break;

case H_ADDMM:
  set(op.tensor, at::redispatch::addmm(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Scalar>(op.args[3]), get<at::Scalar>(op.args[4])));
  break;

case H_ADDMM_:
  init_update_in_place(op.tensor);
  at::redispatch::addmm_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Scalar>(op.args[3]), get<at::Scalar>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_SPARSE_CSR_TENSOR_CROW_COL_VALUE_SIZE:
  set(op.tensor, at::redispatch::sparse_csr_tensor(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<c10::optional<at::ScalarType>>(op.args[4]), get<c10::optional<at::Layout>>(op.args[5]), get<c10::optional<at::Device>>(op.args[6]), get<c10::optional<bool>>(op.args[7])));
  break;

case H_SPARSE_CSR_TENSOR_CROW_COL_VALUE:
  set(op.tensor, at::redispatch::sparse_csr_tensor(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<c10::optional<at::ScalarType>>(op.args[3]), get<c10::optional<at::Layout>>(op.args[4]), get<c10::optional<at::Device>>(op.args[5]), get<c10::optional<bool>>(op.args[6])));
  break;

case H_SPARSE_COO_TENSOR_SIZE:
  set(op.tensor, at::redispatch::sparse_coo_tensor(ks, get<at::IntArrayRef>(op.args[0]), get<c10::optional<at::ScalarType>>(op.args[1]), get<c10::optional<at::Layout>>(op.args[2]), get<c10::optional<at::Device>>(op.args[3]), get<c10::optional<bool>>(op.args[4])));
  break;

case H_SPARSE_COO_TENSOR_INDICES:
  set(op.tensor, at::redispatch::sparse_coo_tensor(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<at::ScalarType>>(op.args[2]), get<c10::optional<at::Layout>>(op.args[3]), get<c10::optional<at::Device>>(op.args[4]), get<c10::optional<bool>>(op.args[5])));
  break;

case H_SPARSE_COO_TENSOR_INDICES_SIZE:
  set(op.tensor, at::redispatch::sparse_coo_tensor(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<c10::optional<at::ScalarType>>(op.args[3]), get<c10::optional<at::Layout>>(op.args[4]), get<c10::optional<at::Device>>(op.args[5]), get<c10::optional<bool>>(op.args[6])));
  break;

case H__SPARSE_COO_TENSOR_UNSAFE:
  set(op.tensor, at::redispatch::_sparse_coo_tensor_unsafe(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<c10::optional<at::ScalarType>>(op.args[3]), get<c10::optional<at::Layout>>(op.args[4]), get<c10::optional<at::Device>>(op.args[5]), get<c10::optional<bool>>(op.args[6])));
  break;

// skip void _validate_sparse_coo_tensor_args(const at::Tensor & indices, const at::Tensor & values, at::IntArrayRef size)

case H__SPARSE_COO_TENSOR_WITH_DIMS:
  set(op.tensor, at::redispatch::_sparse_coo_tensor_with_dims(ks, get<int64_t>(op.args[0]), get<int64_t>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<c10::optional<at::ScalarType>>(op.args[3]), get<c10::optional<at::Layout>>(op.args[4]), get<c10::optional<at::Device>>(op.args[5]), get<c10::optional<bool>>(op.args[6])));
  break;

case H__SPARSE_COO_TENSOR_WITH_DIMS_AND_TENSORS:
  set(op.tensor, at::redispatch::_sparse_coo_tensor_with_dims_and_tensors(ks, get<int64_t>(op.args[0]), get<int64_t>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::Tensor>(op.args[3]), get<at::Tensor>(op.args[4]), get<c10::optional<at::ScalarType>>(op.args[5]), get<c10::optional<at::Layout>>(op.args[6]), get<c10::optional<at::Device>>(op.args[7]), get<c10::optional<bool>>(op.args[8])));
  break;

case H_SPARSE_RESIZE_:
  init_update_in_place(op.tensor);
  at::redispatch::sparse_resize_(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<int64_t>(op.args[2]), get<int64_t>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_SPARSE_RESIZE_AND_CLEAR_:
  init_update_in_place(op.tensor);
  at::redispatch::sparse_resize_and_clear_(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<int64_t>(op.args[2]), get<int64_t>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_SPARSE_MASK:
  set(op.tensor, at::redispatch::sparse_mask(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_TO_DENSE:
  set(op.tensor, at::redispatch::to_dense(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::ScalarType>>(op.args[1])));
  break;

case H_TO_DENSE_BACKWARD:
  set(op.tensor, at::redispatch::to_dense_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

// skip int64_t sparse_dim(const at::Tensor & self)

// skip int64_t _dimI(const at::Tensor & self)

// skip int64_t dense_dim(const at::Tensor & self)

// skip int64_t _dimV(const at::Tensor & self)

// skip int64_t _nnz(const at::Tensor & self)

case H_COALESCE:
  set(op.tensor, at::redispatch::coalesce(ks, get<at::Tensor>(op.args[0])));
  break;

case H__COALESCE:
  set(op.tensor, at::redispatch::_coalesce(ks, get<at::Tensor>(op.args[0])));
  break;

// skip bool is_coalesced(const at::Tensor & self)

case H__INDICES:
  set(op.tensor, at::redispatch::_indices(ks, get<at::Tensor>(op.args[0])));
  break;

case H__VALUES:
  set(op.tensor, at::redispatch::_values(ks, get<at::Tensor>(op.args[0])));
  break;

case H__COALESCED_:
  init_update_in_place(op.tensor);
  at::redispatch::_coalesced_(ks, get<at::Tensor>(op.args[0]), get<bool>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_INDICES:
  set(op.tensor, at::redispatch::indices(ks, get<at::Tensor>(op.args[0])));
  break;

case H_VALUES:
  set(op.tensor, at::redispatch::values(ks, get<at::Tensor>(op.args[0])));
  break;

case H_CROW_INDICES:
  set(op.tensor, at::redispatch::crow_indices(ks, get<at::Tensor>(op.args[0])));
  break;

case H_COL_INDICES:
  set(op.tensor, at::redispatch::col_indices(ks, get<at::Tensor>(op.args[0])));
  break;

case H_HSPMM_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::hspmm_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_HSPMM:
  set(op.tensor, at::redispatch::hspmm(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_COPY_SPARSE_TO_SPARSE_:
  init_update_in_place(op.tensor);
  at::redispatch::copy_sparse_to_sparse_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<bool>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

// skip std::vector<at::Tensor> unbind(const at::Tensor & self, int64_t dim)

// skip std::vector<at::Tensor> unbind(const at::Tensor & self, at::Dimname dim)

case H_TO_SPARSE_SPARSE_DIM:
  set(op.tensor, at::redispatch::to_sparse(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1])));
  break;

case H_TO_SPARSE:
  set(op.tensor, at::redispatch::to_sparse(ks, get<at::Tensor>(op.args[0])));
  break;

case H_TO_MKLDNN:
  set(op.tensor, at::redispatch::to_mkldnn(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::ScalarType>>(op.args[1])));
  break;

case H_MKLDNN_REORDER_CONV2D_WEIGHT:
  set(op.tensor, at::redispatch::mkldnn_reorder_conv2d_weight(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<int64_t>(op.args[4])));
  break;

case H_MKLDNN_REORDER_CONV3D_WEIGHT:
  set(op.tensor, at::redispatch::mkldnn_reorder_conv3d_weight(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<int64_t>(op.args[4])));
  break;

case H_TO_MKLDNN_BACKWARD:
  set(op.tensor, at::redispatch::to_mkldnn_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_QUANTIZE_PER_TENSOR:
  set(op.tensor, at::redispatch::quantize_per_tensor(ks, get<at::Tensor>(op.args[0]), get<double>(op.args[1]), get<int64_t>(op.args[2]), get<at::ScalarType>(op.args[3])));
  break;

// skip std::vector<at::Tensor> quantize_per_tensor(at::TensorList tensors, const at::Tensor & scales, const at::Tensor & zero_points, at::ScalarType dtype)

case H_QUANTIZE_PER_CHANNEL:
  set(op.tensor, at::redispatch::quantize_per_channel(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<int64_t>(op.args[3]), get<at::ScalarType>(op.args[4])));
  break;

case H_DEQUANTIZE_SELF:
  set(op.tensor, at::redispatch::dequantize(ks, get<at::Tensor>(op.args[0])));
  break;

// skip std::vector<at::Tensor> dequantize(at::TensorList tensors)

// skip double q_scale(const at::Tensor & self)

// skip int64_t q_zero_point(const at::Tensor & self)

case H_Q_PER_CHANNEL_SCALES:
  set(op.tensor, at::redispatch::q_per_channel_scales(ks, get<at::Tensor>(op.args[0])));
  break;

case H_Q_PER_CHANNEL_ZERO_POINTS:
  set(op.tensor, at::redispatch::q_per_channel_zero_points(ks, get<at::Tensor>(op.args[0])));
  break;

// skip int64_t q_per_channel_axis(const at::Tensor & self)

case H_INT_REPR:
  set(op.tensor, at::redispatch::int_repr(ks, get<at::Tensor>(op.args[0])));
  break;

case H__MAKE_PER_TENSOR_QUANTIZED_TENSOR:
  set(op.tensor, at::redispatch::_make_per_tensor_quantized_tensor(ks, get<at::Tensor>(op.args[0]), get<double>(op.args[1]), get<int64_t>(op.args[2])));
  break;

case H__MAKE_PER_CHANNEL_QUANTIZED_TENSOR:
  set(op.tensor, at::redispatch::_make_per_channel_quantized_tensor(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<int64_t>(op.args[3])));
  break;

// skip at::QScheme qscheme(const at::Tensor & self)

case H_FAKE_QUANTIZE_PER_TENSOR_AFFINE:
  set(op.tensor, at::redispatch::fake_quantize_per_tensor_affine(ks, get<at::Tensor>(op.args[0]), get<double>(op.args[1]), get<int64_t>(op.args[2]), get<int64_t>(op.args[3]), get<int64_t>(op.args[4])));
  break;

// skip std::tuple<at::Tensor,at::Tensor> fake_quantize_per_tensor_affine_cachemask(const at::Tensor & self, double scale, int64_t zero_point, int64_t quant_min, int64_t quant_max)

case H_FAKE_QUANTIZE_PER_TENSOR_AFFINE_CACHEMASK_BACKWARD:
  set(op.tensor, at::redispatch::fake_quantize_per_tensor_affine_cachemask_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H__FAKE_QUANTIZE_LEARNABLE_PER_TENSOR_AFFINE:
  set(op.tensor, at::redispatch::_fake_quantize_learnable_per_tensor_affine(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<int64_t>(op.args[3]), get<int64_t>(op.args[4]), get<double>(op.args[5])));
  break;

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor> _fake_quantize_learnable_per_tensor_affine_backward(const at::Tensor & grad, const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t quant_min, int64_t quant_max, double grad_factor)

case H_FAKE_QUANTIZE_PER_CHANNEL_AFFINE:
  set(op.tensor, at::redispatch::fake_quantize_per_channel_affine(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<int64_t>(op.args[3]), get<int64_t>(op.args[4]), get<int64_t>(op.args[5])));
  break;

// skip std::tuple<at::Tensor,at::Tensor> fake_quantize_per_channel_affine_cachemask(const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max)

case H_FAKE_QUANTIZE_PER_CHANNEL_AFFINE_CACHEMASK_BACKWARD:
  set(op.tensor, at::redispatch::fake_quantize_per_channel_affine_cachemask_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H__FAKE_QUANTIZE_LEARNABLE_PER_CHANNEL_AFFINE:
  set(op.tensor, at::redispatch::_fake_quantize_learnable_per_channel_affine(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<int64_t>(op.args[3]), get<int64_t>(op.args[4]), get<int64_t>(op.args[5]), get<double>(op.args[6])));
  break;

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor> _fake_quantize_learnable_per_channel_affine_backward(const at::Tensor & grad, const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max, double grad_factor)

// skip std::tuple<double,int64_t> _choose_qparams_per_tensor(const at::Tensor & self, bool reduce_range)

case H__SATURATE_WEIGHT_TO_FP16:
  set(op.tensor, at::redispatch::_saturate_weight_to_fp16(ks, get<at::Tensor>(op.args[0])));
  break;

// skip std::tuple<at::Tensor,at::Tensor> choose_qparams_optimized(const at::Tensor & input, int64_t numel, int64_t n_bins, double ratio, int64_t bit_width)

case H_TO_DTYPE_LAYOUT:
  set(op.tensor, at::redispatch::to(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::ScalarType>>(op.args[1]), get<c10::optional<at::Layout>>(op.args[2]), get<c10::optional<at::Device>>(op.args[3]), get<c10::optional<bool>>(op.args[4]), get<bool>(op.args[5]), get<bool>(op.args[6]), get<c10::optional<at::MemoryFormat>>(op.args[7])));
  break;

case H_TO_DEVICE:
  set(op.tensor, at::redispatch::to(ks, get<at::Tensor>(op.args[0]), get<at::Device>(op.args[1]), get<at::ScalarType>(op.args[2]), get<bool>(op.args[3]), get<bool>(op.args[4]), get<c10::optional<at::MemoryFormat>>(op.args[5])));
  break;

case H_TO_DTYPE:
  set(op.tensor, at::redispatch::to(ks, get<at::Tensor>(op.args[0]), get<at::ScalarType>(op.args[1]), get<bool>(op.args[2]), get<bool>(op.args[3]), get<c10::optional<at::MemoryFormat>>(op.args[4])));
  break;

case H_TO_OTHER:
  set(op.tensor, at::redispatch::to(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<bool>(op.args[2]), get<bool>(op.args[3]), get<c10::optional<at::MemoryFormat>>(op.args[4])));
  break;

// skip std::vector<at::Tensor> meshgrid(at::TensorList tensors)

case H_CARTESIAN_PROD:
  set(op.tensor, at::redispatch::cartesian_prod(ks, get<at::TensorList>(op.args[0])));
  break;

case H_COMBINATIONS:
  set(op.tensor, at::redispatch::combinations(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<bool>(op.args[2])));
  break;

// skip at::Scalar item(const at::Tensor & self)

// skip at::ScalarType result_type(const at::Tensor & tensor, const at::Tensor & other)

// skip at::ScalarType result_type(const at::Tensor & tensor, const at::Scalar & other)

// skip at::ScalarType result_type(const at::Scalar & scalar, const at::Tensor & tensor)

// skip at::ScalarType result_type(const at::Scalar & scalar1, const at::Scalar & scalar2)

// skip bool can_cast(at::ScalarType from, at::ScalarType to)

// skip at::ScalarType promote_types(at::ScalarType type1, at::ScalarType type2)

// skip at::Scalar _local_scalar_dense(const at::Tensor & self)

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor> _thnn_fused_lstm_cell(const at::Tensor & input_gates, const at::Tensor & hidden_gates, const at::Tensor & cx, const c10::optional<at::Tensor> & input_bias, const c10::optional<at::Tensor> & hidden_bias)

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> _thnn_fused_lstm_cell_backward(const c10::optional<at::Tensor> & grad_hy, const c10::optional<at::Tensor> & grad_cy, const at::Tensor & cx, const at::Tensor & cy, const at::Tensor & workspace, bool has_bias)

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> _thnn_differentiable_lstm_cell_backward(const c10::optional<at::Tensor> & grad_hy, const c10::optional<at::Tensor> & grad_cy, const at::Tensor & input_gates, const at::Tensor & hidden_gates, const c10::optional<at::Tensor> & input_bias, const c10::optional<at::Tensor> & hidden_bias, const at::Tensor & cx, const at::Tensor & cy)

// skip std::tuple<at::Tensor,at::Tensor> _thnn_fused_gru_cell(const at::Tensor & input_gates, const at::Tensor & hidden_gates, const at::Tensor & hx, const c10::optional<at::Tensor> & input_bias, const c10::optional<at::Tensor> & hidden_bias)

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> _thnn_fused_gru_cell_backward(const at::Tensor & grad_hy, const at::Tensor & workspace, bool has_bias)

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> _thnn_differentiable_gru_cell_backward(const at::Tensor & grad_hy, const at::Tensor & input_gates, const at::Tensor & hidden_gates, const at::Tensor & hx, const c10::optional<at::Tensor> & input_bias, const c10::optional<at::Tensor> & hidden_bias)

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor> lstm(const at::Tensor & input, at::TensorList hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first)

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor> lstm(const at::Tensor & data, const at::Tensor & batch_sizes, at::TensorList hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional)

// skip std::tuple<at::Tensor,at::Tensor> gru(const at::Tensor & input, const at::Tensor & hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first)

// skip std::tuple<at::Tensor,at::Tensor> gru(const at::Tensor & data, const at::Tensor & batch_sizes, const at::Tensor & hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional)

// skip std::tuple<at::Tensor,at::Tensor> rnn_tanh(const at::Tensor & input, const at::Tensor & hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first)

// skip std::tuple<at::Tensor,at::Tensor> rnn_tanh(const at::Tensor & data, const at::Tensor & batch_sizes, const at::Tensor & hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional)

// skip std::tuple<at::Tensor,at::Tensor> rnn_relu(const at::Tensor & input, const at::Tensor & hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first)

// skip std::tuple<at::Tensor,at::Tensor> rnn_relu(const at::Tensor & data, const at::Tensor & batch_sizes, const at::Tensor & hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional)

// skip std::tuple<at::Tensor,at::Tensor> lstm_cell(const at::Tensor & input, at::TensorList hx, const at::Tensor & w_ih, const at::Tensor & w_hh, const c10::optional<at::Tensor> & b_ih, const c10::optional<at::Tensor> & b_hh)

case H_GRU_CELL:
  set(op.tensor, at::redispatch::gru_cell(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Tensor>(op.args[3]), get<c10::optional<at::Tensor>>(op.args[4]), get<c10::optional<at::Tensor>>(op.args[5])));
  break;

case H_RNN_TANH_CELL:
  set(op.tensor, at::redispatch::rnn_tanh_cell(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Tensor>(op.args[3]), get<c10::optional<at::Tensor>>(op.args[4]), get<c10::optional<at::Tensor>>(op.args[5])));
  break;

case H_RNN_RELU_CELL:
  set(op.tensor, at::redispatch::rnn_relu_cell(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Tensor>(op.args[3]), get<c10::optional<at::Tensor>>(op.args[4]), get<c10::optional<at::Tensor>>(op.args[5])));
  break;

// skip std::tuple<at::Tensor,at::Tensor> quantized_lstm_cell(const at::Tensor & input, at::TensorList hx, const at::Tensor & w_ih, const at::Tensor & w_hh, const at::Tensor & b_ih, const at::Tensor & b_hh, const at::Tensor & packed_ih, const at::Tensor & packed_hh, const at::Tensor & col_offsets_ih, const at::Tensor & col_offsets_hh, const at::Scalar & scale_ih, const at::Scalar & scale_hh, const at::Scalar & zero_point_ih, const at::Scalar & zero_point_hh)

case H_QUANTIZED_GRU_CELL:
  set(op.tensor, at::redispatch::quantized_gru_cell(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Tensor>(op.args[3]), get<at::Tensor>(op.args[4]), get<at::Tensor>(op.args[5]), get<at::Tensor>(op.args[6]), get<at::Tensor>(op.args[7]), get<at::Tensor>(op.args[8]), get<at::Tensor>(op.args[9]), get<at::Scalar>(op.args[10]), get<at::Scalar>(op.args[11]), get<at::Scalar>(op.args[12]), get<at::Scalar>(op.args[13])));
  break;

case H_QUANTIZED_RNN_RELU_CELL:
  set(op.tensor, at::redispatch::quantized_rnn_relu_cell(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Tensor>(op.args[3]), get<at::Tensor>(op.args[4]), get<at::Tensor>(op.args[5]), get<at::Tensor>(op.args[6]), get<at::Tensor>(op.args[7]), get<at::Tensor>(op.args[8]), get<at::Tensor>(op.args[9]), get<at::Scalar>(op.args[10]), get<at::Scalar>(op.args[11]), get<at::Scalar>(op.args[12]), get<at::Scalar>(op.args[13])));
  break;

case H_QUANTIZED_RNN_TANH_CELL:
  set(op.tensor, at::redispatch::quantized_rnn_tanh_cell(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Tensor>(op.args[3]), get<at::Tensor>(op.args[4]), get<at::Tensor>(op.args[5]), get<at::Tensor>(op.args[6]), get<at::Tensor>(op.args[7]), get<at::Tensor>(op.args[8]), get<at::Tensor>(op.args[9]), get<at::Scalar>(op.args[10]), get<at::Scalar>(op.args[11]), get<at::Scalar>(op.args[12]), get<at::Scalar>(op.args[13])));
  break;

// skip std::tuple<at::Tensor,at::Tensor> _pack_padded_sequence(const at::Tensor & input, const at::Tensor & lengths, bool batch_first)

case H__PACK_PADDED_SEQUENCE_BACKWARD:
  set(op.tensor, at::redispatch::_pack_padded_sequence_backward(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<at::Tensor>(op.args[2]), get<bool>(op.args[3])));
  break;

// skip std::tuple<at::Tensor,at::Tensor> _pad_packed_sequence(const at::Tensor & data, const at::Tensor & batch_sizes, bool batch_first, const at::Scalar & padding_value, int64_t total_length)

case H_SET__SOURCE_STORAGE:
  init_update_in_place(op.tensor);
  at::redispatch::set_(ks, get<at::Tensor>(op.args[0]), get<at::Storage>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_SET__SOURCE_STORAGE_STORAGE_OFFSET:
  init_update_in_place(op.tensor);
  at::redispatch::set_(ks, get<at::Tensor>(op.args[0]), get<at::Storage>(op.args[1]), get<int64_t>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_SET__SOURCE_TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::set_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_SET_:
  init_update_in_place(op.tensor);
  at::redispatch::set_(ks, get<at::Tensor>(op.args[0]));
  end_update_in_place(op.tensor);
  break;

// skip bool is_set_to(const at::Tensor & self, const at::Tensor & tensor)

case H_MASKED_FILL__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::masked_fill_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Scalar>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_MASKED_FILL_SCALAR:
  set(op.tensor, at::redispatch::masked_fill(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Scalar>(op.args[2])));
  break;

case H_MASKED_FILL__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::masked_fill_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_MASKED_FILL_TENSOR:
  set(op.tensor, at::redispatch::masked_fill(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2])));
  break;

case H_MASKED_SCATTER_:
  init_update_in_place(op.tensor);
  at::redispatch::masked_scatter_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_MASKED_SCATTER:
  set(op.tensor, at::redispatch::masked_scatter(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2])));
  break;

case H_VIEW:
  set(op.tensor, at::redispatch::view(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1])));
  break;

case H_VIEW_DTYPE:
  set(op.tensor, at::redispatch::view(ks, get<at::Tensor>(op.args[0]), get<at::ScalarType>(op.args[1])));
  break;

case H_PUT_:
  init_update_in_place(op.tensor);
  at::redispatch::put_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<bool>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_PUT:
  set(op.tensor, at::redispatch::put(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<bool>(op.args[3])));
  break;

case H_INDEX_ADD_:
  init_update_in_place(op.tensor);
  at::redispatch::index_add_(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_INDEX_ADD__ALPHA:
  init_update_in_place(op.tensor);
  at::redispatch::index_add_(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Tensor>(op.args[3]), get<at::Scalar>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_INDEX_ADD:
  set(op.tensor, at::redispatch::index_add(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Tensor>(op.args[3])));
  break;

case H_INDEX_ADD_ALPHA:
  set(op.tensor, at::redispatch::index_add(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Tensor>(op.args[3]), get<at::Scalar>(op.args[4])));
  break;

case H_INDEX_ADD_DIMNAME:
  set(op.tensor, at::redispatch::index_add(ks, get<at::Tensor>(op.args[0]), get<at::Dimname>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Tensor>(op.args[3]), get<at::Scalar>(op.args[4])));
  break;

case H_INDEX_FILL__INT_SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::index_fill_(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Scalar>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_INDEX_FILL_INT_SCALAR:
  set(op.tensor, at::redispatch::index_fill(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Scalar>(op.args[3])));
  break;

case H_INDEX_FILL__INT_TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::index_fill_(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_INDEX_FILL_INT_TENSOR:
  set(op.tensor, at::redispatch::index_fill(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Tensor>(op.args[3])));
  break;

case H_INDEX_FILL__DIMNAME_SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::index_fill_(ks, get<at::Tensor>(op.args[0]), get<at::Dimname>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Scalar>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_INDEX_FILL__DIMNAME_TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::index_fill_(ks, get<at::Tensor>(op.args[0]), get<at::Dimname>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_INDEX_FILL_DIMNAME_SCALAR:
  set(op.tensor, at::redispatch::index_fill(ks, get<at::Tensor>(op.args[0]), get<at::Dimname>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Scalar>(op.args[3])));
  break;

case H_INDEX_FILL_DIMNAME_TENSOR:
  set(op.tensor, at::redispatch::index_fill(ks, get<at::Tensor>(op.args[0]), get<at::Dimname>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Tensor>(op.args[3])));
  break;

case H_SCATTER__SRC:
  init_update_in_place(op.tensor);
  at::redispatch::scatter_(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_SCATTER_SRC:
  set(op.tensor, at::redispatch::scatter(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Tensor>(op.args[3])));
  break;

case H_SCATTER__VALUE:
  init_update_in_place(op.tensor);
  at::redispatch::scatter_(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Scalar>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_SCATTER_VALUE:
  set(op.tensor, at::redispatch::scatter(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Scalar>(op.args[3])));
  break;

case H_SCATTER_DIMNAME_SRC:
  set(op.tensor, at::redispatch::scatter(ks, get<at::Tensor>(op.args[0]), get<at::Dimname>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Tensor>(op.args[3])));
  break;

case H_SCATTER_DIMNAME_VALUE:
  set(op.tensor, at::redispatch::scatter(ks, get<at::Tensor>(op.args[0]), get<at::Dimname>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Scalar>(op.args[3])));
  break;

case H_SCATTER__REDUCE:
  init_update_in_place(op.tensor);
  at::redispatch::scatter_(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Tensor>(op.args[3]), get<std::string>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_SCATTER__VALUE_REDUCE:
  init_update_in_place(op.tensor);
  at::redispatch::scatter_(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Scalar>(op.args[3]), get<std::string>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_SCATTER_ADD_:
  init_update_in_place(op.tensor);
  at::redispatch::scatter_add_(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_SCATTER_ADD:
  set(op.tensor, at::redispatch::scatter_add(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Tensor>(op.args[3])));
  break;

case H_SCATTER_ADD_DIMNAME:
  set(op.tensor, at::redispatch::scatter_add(ks, get<at::Tensor>(op.args[0]), get<at::Dimname>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Tensor>(op.args[3])));
  break;

case H_EQ__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::eq_(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_EQ__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::eq_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_BITWISE_AND_TENSOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::bitwise_and_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_BITWISE_AND_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::bitwise_and_outf(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_BITWISE_AND_SCALAR:
  set(op.tensor, at::redispatch::bitwise_and(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1])));
  break;

case H_BITWISE_AND_TENSOR:
  set(op.tensor, at::redispatch::bitwise_and(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_BITWISE_AND__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::bitwise_and_(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_BITWISE_AND__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::bitwise_and_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H___AND___SCALAR:
  set(op.tensor, at::redispatch::__and__(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1])));
  break;

case H___AND___TENSOR:
  set(op.tensor, at::redispatch::__and__(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H___IAND___SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::__iand__(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H___IAND___TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::__iand__(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_BITWISE_OR_TENSOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::bitwise_or_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_BITWISE_OR_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::bitwise_or_outf(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_BITWISE_OR_SCALAR:
  set(op.tensor, at::redispatch::bitwise_or(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1])));
  break;

case H_BITWISE_OR_TENSOR:
  set(op.tensor, at::redispatch::bitwise_or(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_BITWISE_OR__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::bitwise_or_(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_BITWISE_OR__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::bitwise_or_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H___OR___SCALAR:
  set(op.tensor, at::redispatch::__or__(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1])));
  break;

case H___OR___TENSOR:
  set(op.tensor, at::redispatch::__or__(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H___IOR___SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::__ior__(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H___IOR___TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::__ior__(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_BITWISE_XOR_TENSOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::bitwise_xor_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_BITWISE_XOR_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::bitwise_xor_outf(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_BITWISE_XOR_SCALAR:
  set(op.tensor, at::redispatch::bitwise_xor(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1])));
  break;

case H_BITWISE_XOR_TENSOR:
  set(op.tensor, at::redispatch::bitwise_xor(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_BITWISE_XOR__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::bitwise_xor_(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_BITWISE_XOR__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::bitwise_xor_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H___XOR___SCALAR:
  set(op.tensor, at::redispatch::__xor__(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1])));
  break;

case H___XOR___TENSOR:
  set(op.tensor, at::redispatch::__xor__(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H___IXOR___SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::__ixor__(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H___IXOR___TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::__ixor__(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H___LSHIFT___SCALAR:
  set(op.tensor, at::redispatch::__lshift__(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1])));
  break;

case H___LSHIFT___TENSOR:
  set(op.tensor, at::redispatch::__lshift__(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H___ILSHIFT___SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::__ilshift__(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H___ILSHIFT___TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::__ilshift__(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H___RSHIFT___SCALAR:
  set(op.tensor, at::redispatch::__rshift__(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1])));
  break;

case H___RSHIFT___TENSOR:
  set(op.tensor, at::redispatch::__rshift__(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H___IRSHIFT___SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::__irshift__(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H___IRSHIFT___TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::__irshift__(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_TRIL_:
  init_update_in_place(op.tensor);
  at::redispatch::tril_(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_TRIU_:
  init_update_in_place(op.tensor);
  at::redispatch::triu_(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_RENORM_:
  init_update_in_place(op.tensor);
  at::redispatch::renorm_(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<int64_t>(op.args[2]), get<at::Scalar>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_LERP__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::lerp_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Scalar>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_LERP__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::lerp_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_FMOD__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::fmod_(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_FMOD__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::fmod_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_REMAINDER__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::remainder_(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_REMAINDER__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::remainder_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_ADDBMM_:
  init_update_in_place(op.tensor);
  at::redispatch::addbmm_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Scalar>(op.args[3]), get<at::Scalar>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_ADDBMM_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::addbmm_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Scalar>(op.args[3]), get<at::Scalar>(op.args[4]), get<at::Tensor>(op.args[5]));
  end_update_in_place(op.tensor);
  break;

case H_ADDBMM:
  set(op.tensor, at::redispatch::addbmm(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Scalar>(op.args[3]), get<at::Scalar>(op.args[4])));
  break;

case H_ADDCDIV_:
  init_update_in_place(op.tensor);
  at::redispatch::addcdiv_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Scalar>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_RANDOM__FROM:
  init_update_in_place(op.tensor);
  at::redispatch::random_(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<c10::optional<int64_t>>(op.args[2]), get<c10::optional<at::Generator>>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_RANDOM__TO:
  init_update_in_place(op.tensor);
  at::redispatch::random_(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<c10::optional<at::Generator>>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_RANDOM_:
  init_update_in_place(op.tensor);
  at::redispatch::random_(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::Generator>>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_UNIFORM_:
  init_update_in_place(op.tensor);
  at::redispatch::uniform_(ks, get<at::Tensor>(op.args[0]), get<double>(op.args[1]), get<double>(op.args[2]), get<c10::optional<at::Generator>>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_CAUCHY_:
  init_update_in_place(op.tensor);
  at::redispatch::cauchy_(ks, get<at::Tensor>(op.args[0]), get<double>(op.args[1]), get<double>(op.args[2]), get<c10::optional<at::Generator>>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_LOG_NORMAL_:
  init_update_in_place(op.tensor);
  at::redispatch::log_normal_(ks, get<at::Tensor>(op.args[0]), get<double>(op.args[1]), get<double>(op.args[2]), get<c10::optional<at::Generator>>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_EXPONENTIAL_:
  init_update_in_place(op.tensor);
  at::redispatch::exponential_(ks, get<at::Tensor>(op.args[0]), get<double>(op.args[1]), get<c10::optional<at::Generator>>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_GEOMETRIC_:
  init_update_in_place(op.tensor);
  at::redispatch::geometric_(ks, get<at::Tensor>(op.args[0]), get<double>(op.args[1]), get<c10::optional<at::Generator>>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_DIAG_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::diag_outf(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_DIAG:
  set(op.tensor, at::redispatch::diag(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1])));
  break;

case H_DIAG_BACKWARD:
  set(op.tensor, at::redispatch::diag_backward(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<int64_t>(op.args[2])));
  break;

case H_CROSS_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::cross_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<int64_t>>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_CROSS:
  set(op.tensor, at::redispatch::cross(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<int64_t>>(op.args[2])));
  break;

case H_TRIU_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::triu_outf(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_TRIU:
  set(op.tensor, at::redispatch::triu(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1])));
  break;

case H_TRIL_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::tril_outf(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_TRIL:
  set(op.tensor, at::redispatch::tril(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1])));
  break;

case H_TRIL_INDICES:
  set(op.tensor, at::redispatch::tril_indices(ks, get<int64_t>(op.args[0]), get<int64_t>(op.args[1]), get<int64_t>(op.args[2]), get<c10::optional<at::ScalarType>>(op.args[3]), get<c10::optional<at::Layout>>(op.args[4]), get<c10::optional<at::Device>>(op.args[5]), get<c10::optional<bool>>(op.args[6])));
  break;

case H_TRIU_INDICES:
  set(op.tensor, at::redispatch::triu_indices(ks, get<int64_t>(op.args[0]), get<int64_t>(op.args[1]), get<int64_t>(op.args[2]), get<c10::optional<at::ScalarType>>(op.args[3]), get<c10::optional<at::Layout>>(op.args[4]), get<c10::optional<at::Device>>(op.args[5]), get<c10::optional<bool>>(op.args[6])));
  break;

case H_TRACE:
  set(op.tensor, at::redispatch::trace(ks, get<at::Tensor>(op.args[0])));
  break;

case H_TRACE_BACKWARD:
  set(op.tensor, at::redispatch::trace_backward(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1])));
  break;

case H_NE_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::ne_outf(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_NE_SCALAR:
  set(op.tensor, at::redispatch::ne(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1])));
  break;

case H_NE_TENSOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::ne_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_NE_TENSOR:
  set(op.tensor, at::redispatch::ne(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_NE__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::ne_(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_NE__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::ne_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_NOT_EQUAL_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::not_equal_outf(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_NOT_EQUAL_SCALAR:
  set(op.tensor, at::redispatch::not_equal(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1])));
  break;

case H_NOT_EQUAL_TENSOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::not_equal_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_NOT_EQUAL_TENSOR:
  set(op.tensor, at::redispatch::not_equal(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_NOT_EQUAL__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::not_equal_(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_NOT_EQUAL__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::not_equal_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_EQ_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::eq_outf(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_EQ_SCALAR:
  set(op.tensor, at::redispatch::eq(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1])));
  break;

case H_EQ_TENSOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::eq_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_EQ_TENSOR:
  set(op.tensor, at::redispatch::eq(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_GE_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::ge_outf(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_GE_SCALAR:
  set(op.tensor, at::redispatch::ge(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1])));
  break;

case H_GE_TENSOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::ge_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_GE_TENSOR:
  set(op.tensor, at::redispatch::ge(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_GE__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::ge_(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_GE__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::ge_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_GREATER_EQUAL_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::greater_equal_outf(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_GREATER_EQUAL_SCALAR:
  set(op.tensor, at::redispatch::greater_equal(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1])));
  break;

case H_GREATER_EQUAL_TENSOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::greater_equal_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_GREATER_EQUAL_TENSOR:
  set(op.tensor, at::redispatch::greater_equal(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_GREATER_EQUAL__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::greater_equal_(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_GREATER_EQUAL__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::greater_equal_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_LE_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::le_outf(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_LE_SCALAR:
  set(op.tensor, at::redispatch::le(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1])));
  break;

case H_LE_TENSOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::le_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_LE_TENSOR:
  set(op.tensor, at::redispatch::le(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_LE__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::le_(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_LE__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::le_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_LESS_EQUAL_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::less_equal_outf(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_LESS_EQUAL_SCALAR:
  set(op.tensor, at::redispatch::less_equal(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1])));
  break;

case H_LESS_EQUAL_TENSOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::less_equal_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_LESS_EQUAL_TENSOR:
  set(op.tensor, at::redispatch::less_equal(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_LESS_EQUAL__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::less_equal_(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_LESS_EQUAL__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::less_equal_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_GT_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::gt_outf(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_GT_SCALAR:
  set(op.tensor, at::redispatch::gt(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1])));
  break;

case H_GT_TENSOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::gt_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_GT_TENSOR:
  set(op.tensor, at::redispatch::gt(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_GT__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::gt_(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_GT__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::gt_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_GREATER_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::greater_outf(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_GREATER_SCALAR:
  set(op.tensor, at::redispatch::greater(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1])));
  break;

case H_GREATER_TENSOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::greater_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_GREATER_TENSOR:
  set(op.tensor, at::redispatch::greater(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_GREATER__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::greater_(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_GREATER__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::greater_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_LT_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::lt_outf(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_LT_SCALAR:
  set(op.tensor, at::redispatch::lt(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1])));
  break;

case H_LT_TENSOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::lt_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_LT_TENSOR:
  set(op.tensor, at::redispatch::lt(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_LT__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::lt_(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_LT__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::lt_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_LESS_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::less_outf(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_LESS_SCALAR:
  set(op.tensor, at::redispatch::less(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1])));
  break;

case H_LESS_TENSOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::less_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_LESS_TENSOR:
  set(op.tensor, at::redispatch::less(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_LESS__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::less_(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_LESS__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::less_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_TAKE_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::take_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_TAKE:
  set(op.tensor, at::redispatch::take(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_TAKE_ALONG_DIM_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::take_along_dim_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<int64_t>>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_TAKE_ALONG_DIM:
  set(op.tensor, at::redispatch::take_along_dim(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<int64_t>>(op.args[2])));
  break;

case H_INDEX_SELECT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::index_select_outf(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_INDEX_SELECT:
  set(op.tensor, at::redispatch::index_select(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<at::Tensor>(op.args[2])));
  break;

case H_INDEX_SELECT_DIMNAME_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::index_select_outf(ks, get<at::Tensor>(op.args[0]), get<at::Dimname>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_INDEX_SELECT_DIMNAME:
  set(op.tensor, at::redispatch::index_select(ks, get<at::Tensor>(op.args[0]), get<at::Dimname>(op.args[1]), get<at::Tensor>(op.args[2])));
  break;

case H_INDEX_SELECT_BACKWARD:
  set(op.tensor, at::redispatch::index_select_backward(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<int64_t>(op.args[2]), get<at::Tensor>(op.args[3])));
  break;

case H_MASKED_SELECT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::masked_select_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_MASKED_SELECT:
  set(op.tensor, at::redispatch::masked_select(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_MASKED_SELECT_BACKWARD:
  set(op.tensor, at::redispatch::masked_select_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2])));
  break;

case H_NONZERO_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::nonzero_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_NONZERO:
  set(op.tensor, at::redispatch::nonzero(ks, get<at::Tensor>(op.args[0])));
  break;

// skip std::vector<at::Tensor> nonzero_numpy(const at::Tensor & self)

case H_GATHER_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::gather_outf(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<at::Tensor>(op.args[2]), get<bool>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_GATHER:
  set(op.tensor, at::redispatch::gather(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<at::Tensor>(op.args[2]), get<bool>(op.args[3])));
  break;

case H_GATHER_BACKWARD:
  set(op.tensor, at::redispatch::gather_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<int64_t>(op.args[2]), get<at::Tensor>(op.args[3]), get<bool>(op.args[4])));
  break;

case H_GATHER_DIMNAME_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::gather_outf(ks, get<at::Tensor>(op.args[0]), get<at::Dimname>(op.args[1]), get<at::Tensor>(op.args[2]), get<bool>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_GATHER_DIMNAME:
  set(op.tensor, at::redispatch::gather(ks, get<at::Tensor>(op.args[0]), get<at::Dimname>(op.args[1]), get<at::Tensor>(op.args[2]), get<bool>(op.args[3])));
  break;

case H__GATHER_SPARSE_BACKWARD:
  set(op.tensor, at::redispatch::_gather_sparse_backward(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Tensor>(op.args[3])));
  break;

case H_ADDCMUL_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::addcmul_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Scalar>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_ADDCMUL:
  set(op.tensor, at::redispatch::addcmul(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Scalar>(op.args[3])));
  break;

case H_ADDCMUL_:
  init_update_in_place(op.tensor);
  at::redispatch::addcmul_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Scalar>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_ADDCDIV_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::addcdiv_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Scalar>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_ADDCDIV:
  set(op.tensor, at::redispatch::addcdiv(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Scalar>(op.args[3])));
  break;

case H_CROSS_ENTROPY_LOSS:
  set(op.tensor, at::redispatch::cross_entropy_loss(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<at::Tensor>>(op.args[2]), get<int64_t>(op.args[3]), get<int64_t>(op.args[4])));
  break;

// skip std::tuple<at::Tensor &,at::Tensor &> lstsq_outf(const at::Tensor & self, const at::Tensor & A, at::Tensor & X, at::Tensor & qr)

// skip std::tuple<at::Tensor,at::Tensor> lstsq(const at::Tensor & self, const at::Tensor & A)

// skip std::tuple<at::Tensor &,at::Tensor &> triangular_solve_outf(const at::Tensor & self, const at::Tensor & A, bool upper, bool transpose, bool unitriangular, at::Tensor & X, at::Tensor & M)

// skip std::tuple<at::Tensor,at::Tensor> triangular_solve(const at::Tensor & self, const at::Tensor & A, bool upper, bool transpose, bool unitriangular)

// skip std::tuple<at::Tensor &,at::Tensor &> symeig_outf(const at::Tensor & self, bool eigenvectors, bool upper, at::Tensor & e, at::Tensor & V)

// skip std::tuple<at::Tensor,at::Tensor> symeig(const at::Tensor & self, bool eigenvectors, bool upper)

// skip std::tuple<at::Tensor,at::Tensor> _symeig_helper(const at::Tensor & self, bool eigenvectors, bool upper)

// skip std::tuple<at::Tensor &,at::Tensor &> eig_outf(const at::Tensor & self, bool eigenvectors, at::Tensor & e, at::Tensor & v)

// skip std::tuple<at::Tensor,at::Tensor> eig(const at::Tensor & self, bool eigenvectors)

// skip std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> svd_outf(const at::Tensor & self, bool some, bool compute_uv, at::Tensor & U, at::Tensor & S, at::Tensor & V)

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor> svd(const at::Tensor & self, bool some, bool compute_uv)

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor> _svd_helper(const at::Tensor & self, bool some, bool compute_uv)

case H_SWAPAXES:
  set(op.tensor, at::redispatch::swapaxes(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<int64_t>(op.args[2])));
  break;

case H_SWAPAXES_:
  init_update_in_place(op.tensor);
  at::redispatch::swapaxes_(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<int64_t>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_SWAPDIMS:
  set(op.tensor, at::redispatch::swapdims(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<int64_t>(op.args[2])));
  break;

case H_SWAPDIMS_:
  init_update_in_place(op.tensor);
  at::redispatch::swapdims_(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<int64_t>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_CHOLESKY_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::cholesky_outf(ks, get<at::Tensor>(op.args[0]), get<bool>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_CHOLESKY:
  set(op.tensor, at::redispatch::cholesky(ks, get<at::Tensor>(op.args[0]), get<bool>(op.args[1])));
  break;

case H_CHOLESKY_SOLVE_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::cholesky_solve_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<bool>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_CHOLESKY_SOLVE:
  set(op.tensor, at::redispatch::cholesky_solve(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<bool>(op.args[2])));
  break;

case H__CHOLESKY_SOLVE_HELPER:
  set(op.tensor, at::redispatch::_cholesky_solve_helper(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<bool>(op.args[2])));
  break;

// skip std::tuple<at::Tensor,at::Tensor> solve(const at::Tensor & self, const at::Tensor & A)

// skip std::tuple<at::Tensor &,at::Tensor &> solve_outf(const at::Tensor & self, const at::Tensor & A, at::Tensor & solution, at::Tensor & lu)

// skip std::tuple<at::Tensor,at::Tensor> _solve_helper(const at::Tensor & self, const at::Tensor & A)

case H_CHOLESKY_INVERSE:
  set(op.tensor, at::redispatch::cholesky_inverse(ks, get<at::Tensor>(op.args[0]), get<bool>(op.args[1])));
  break;

case H_CHOLESKY_INVERSE_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::cholesky_inverse_outf(ks, get<at::Tensor>(op.args[0]), get<bool>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

// skip std::tuple<at::Tensor &,at::Tensor &> qr_outf(const at::Tensor & self, bool some, at::Tensor & Q, at::Tensor & R)

// skip std::tuple<at::Tensor,at::Tensor> qr(const at::Tensor & self, bool some)

// skip std::tuple<at::Tensor &,at::Tensor &> geqrf_outf(const at::Tensor & self, at::Tensor & a, at::Tensor & tau)

// skip std::tuple<at::Tensor,at::Tensor> geqrf(const at::Tensor & self)

case H_ORGQR:
  set(op.tensor, at::redispatch::orgqr(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_ORGQR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::orgqr_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_ORMQR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::ormqr_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<bool>(op.args[3]), get<bool>(op.args[4]), get<at::Tensor>(op.args[5]));
  end_update_in_place(op.tensor);
  break;

case H_ORMQR:
  set(op.tensor, at::redispatch::ormqr(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<bool>(op.args[3]), get<bool>(op.args[4])));
  break;

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor> _lu_with_info(const at::Tensor & self, bool pivot, bool check_errors)

case H_LU_SOLVE_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::lu_solve_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_LU_SOLVE:
  set(op.tensor, at::redispatch::lu_solve(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2])));
  break;

case H_MULTINOMIAL_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::multinomial_outf(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<bool>(op.args[2]), get<c10::optional<at::Generator>>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_MULTINOMIAL:
  set(op.tensor, at::redispatch::multinomial(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<bool>(op.args[2]), get<c10::optional<at::Generator>>(op.args[3])));
  break;

case H_LGAMMA_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::lgamma_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_DIGAMMA_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::digamma_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_POLYGAMMA_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::polygamma_outf(ks, get<int64_t>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_POLYGAMMA_:
  init_update_in_place(op.tensor);
  at::redispatch::polygamma_(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_ERFINV_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::erfinv_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_I0_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::i0_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_SIGN:
  set(op.tensor, at::redispatch::sign(ks, get<at::Tensor>(op.args[0])));
  break;

case H_SIGN_:
  init_update_in_place(op.tensor);
  at::redispatch::sign_(ks, get<at::Tensor>(op.args[0]));
  end_update_in_place(op.tensor);
  break;

case H_SIGN_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::sign_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_SIGNBIT:
  set(op.tensor, at::redispatch::signbit(ks, get<at::Tensor>(op.args[0])));
  break;

case H_SIGNBIT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::signbit_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_DIST:
  set(op.tensor, at::redispatch::dist(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Scalar>(op.args[2])));
  break;

case H_ATAN2_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::atan2_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_LERP_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::lerp_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Scalar>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_LERP_TENSOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::lerp_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_LERP_SCALAR:
  set(op.tensor, at::redispatch::lerp(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Scalar>(op.args[2])));
  break;

case H_LERP_TENSOR:
  set(op.tensor, at::redispatch::lerp(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2])));
  break;

case H_HISTC_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::histc_outf(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<at::Scalar>(op.args[2]), get<at::Scalar>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_HISTC:
  set(op.tensor, at::redispatch::histc(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<at::Scalar>(op.args[2]), get<at::Scalar>(op.args[3])));
  break;

case H_FMOD_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::fmod_outf(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_FMOD_SCALAR:
  set(op.tensor, at::redispatch::fmod(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1])));
  break;

case H_FMOD_TENSOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::fmod_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_FMOD_TENSOR:
  set(op.tensor, at::redispatch::fmod(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_HYPOT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::hypot_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_HYPOT_:
  init_update_in_place(op.tensor);
  at::redispatch::hypot_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_IGAMMA_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::igamma_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_IGAMMAC_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::igammac_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_NEXTAFTER_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::nextafter_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_NEXTAFTER_:
  init_update_in_place(op.tensor);
  at::redispatch::nextafter_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_REMAINDER_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::remainder_outf(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_REMAINDER_SCALAR:
  set(op.tensor, at::redispatch::remainder(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1])));
  break;

case H_REMAINDER_TENSOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::remainder_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_REMAINDER_TENSOR:
  set(op.tensor, at::redispatch::remainder(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_MIN:
  set(op.tensor, at::redispatch::min(ks, get<at::Tensor>(op.args[0])));
  break;

case H_FMIN:
  set(op.tensor, at::redispatch::fmin(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_FMIN_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::fmin_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_MAX:
  set(op.tensor, at::redispatch::max(ks, get<at::Tensor>(op.args[0])));
  break;

case H_FMAX:
  set(op.tensor, at::redispatch::fmax(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_FMAX_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::fmax_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_MAXIMUM:
  set(op.tensor, at::redispatch::maximum(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_MAXIMUM_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::maximum_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_MAX_OTHER:
  set(op.tensor, at::redispatch::max(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_MAX_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::max_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_MINIMUM:
  set(op.tensor, at::redispatch::minimum(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_MINIMUM_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::minimum_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_MIN_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::min_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_MIN_OTHER:
  set(op.tensor, at::redispatch::min(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_QUANTILE_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::quantile_outf(ks, get<at::Tensor>(op.args[0]), get<double>(op.args[1]), get<c10::optional<int64_t>>(op.args[2]), get<bool>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_QUANTILE_SCALAR:
  set(op.tensor, at::redispatch::quantile(ks, get<at::Tensor>(op.args[0]), get<double>(op.args[1]), get<c10::optional<int64_t>>(op.args[2]), get<bool>(op.args[3])));
  break;

case H_QUANTILE_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::quantile_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<int64_t>>(op.args[2]), get<bool>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_QUANTILE:
  set(op.tensor, at::redispatch::quantile(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<int64_t>>(op.args[2]), get<bool>(op.args[3])));
  break;

case H_NANQUANTILE_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::nanquantile_outf(ks, get<at::Tensor>(op.args[0]), get<double>(op.args[1]), get<c10::optional<int64_t>>(op.args[2]), get<bool>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_NANQUANTILE_SCALAR:
  set(op.tensor, at::redispatch::nanquantile(ks, get<at::Tensor>(op.args[0]), get<double>(op.args[1]), get<c10::optional<int64_t>>(op.args[2]), get<bool>(op.args[3])));
  break;

case H_NANQUANTILE_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::nanquantile_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<int64_t>>(op.args[2]), get<bool>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_NANQUANTILE:
  set(op.tensor, at::redispatch::nanquantile(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<int64_t>>(op.args[2]), get<bool>(op.args[3])));
  break;

case H_QUANTILE_NEW_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::quantile_outf(ks, get<at::Tensor>(op.args[0]), get<double>(op.args[1]), get<c10::optional<int64_t>>(op.args[2]), get<bool>(op.args[3]), get<std::string>(op.args[4]), get<at::Tensor>(op.args[5]));
  end_update_in_place(op.tensor);
  break;

case H_QUANTILE_NEW_SCALAR:
  set(op.tensor, at::redispatch::quantile(ks, get<at::Tensor>(op.args[0]), get<double>(op.args[1]), get<c10::optional<int64_t>>(op.args[2]), get<bool>(op.args[3]), get<std::string>(op.args[4])));
  break;

case H_QUANTILE_NEW_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::quantile_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<int64_t>>(op.args[2]), get<bool>(op.args[3]), get<std::string>(op.args[4]), get<at::Tensor>(op.args[5]));
  end_update_in_place(op.tensor);
  break;

case H_QUANTILE_NEW:
  set(op.tensor, at::redispatch::quantile(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<int64_t>>(op.args[2]), get<bool>(op.args[3]), get<std::string>(op.args[4])));
  break;

case H_NANQUANTILE_NEW_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::nanquantile_outf(ks, get<at::Tensor>(op.args[0]), get<double>(op.args[1]), get<c10::optional<int64_t>>(op.args[2]), get<bool>(op.args[3]), get<std::string>(op.args[4]), get<at::Tensor>(op.args[5]));
  end_update_in_place(op.tensor);
  break;

case H_NANQUANTILE_NEW_SCALAR:
  set(op.tensor, at::redispatch::nanquantile(ks, get<at::Tensor>(op.args[0]), get<double>(op.args[1]), get<c10::optional<int64_t>>(op.args[2]), get<bool>(op.args[3]), get<std::string>(op.args[4])));
  break;

case H_NANQUANTILE_NEW_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::nanquantile_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<int64_t>>(op.args[2]), get<bool>(op.args[3]), get<std::string>(op.args[4]), get<at::Tensor>(op.args[5]));
  end_update_in_place(op.tensor);
  break;

case H_NANQUANTILE_NEW:
  set(op.tensor, at::redispatch::nanquantile(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<int64_t>>(op.args[2]), get<bool>(op.args[3]), get<std::string>(op.args[4])));
  break;

// skip std::tuple<at::Tensor &,at::Tensor &> sort_outf(const at::Tensor & self, int64_t dim, bool descending, at::Tensor & values, at::Tensor & indices)

// skip std::tuple<at::Tensor &,at::Tensor &> sort_outf(const at::Tensor & self, c10::optional<bool> stable, int64_t dim, bool descending, at::Tensor & values, at::Tensor & indices)

// skip std::tuple<at::Tensor,at::Tensor> sort(const at::Tensor & self, int64_t dim, bool descending)

// skip std::tuple<at::Tensor,at::Tensor> sort(const at::Tensor & self, c10::optional<bool> stable, int64_t dim, bool descending)

// skip std::tuple<at::Tensor &,at::Tensor &> sort_outf(const at::Tensor & self, at::Dimname dim, bool descending, at::Tensor & values, at::Tensor & indices)

// skip std::tuple<at::Tensor &,at::Tensor &> sort_outf(const at::Tensor & self, c10::optional<bool> stable, at::Dimname dim, bool descending, at::Tensor & values, at::Tensor & indices)

// skip std::tuple<at::Tensor,at::Tensor> sort(const at::Tensor & self, at::Dimname dim, bool descending)

// skip std::tuple<at::Tensor,at::Tensor> sort(const at::Tensor & self, c10::optional<bool> stable, at::Dimname dim, bool descending)

case H_MSORT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::msort_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_MSORT:
  set(op.tensor, at::redispatch::msort(ks, get<at::Tensor>(op.args[0])));
  break;

case H_ARGSORT:
  set(op.tensor, at::redispatch::argsort(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<bool>(op.args[2])));
  break;

case H_ARGSORT_DIMNAME:
  set(op.tensor, at::redispatch::argsort(ks, get<at::Tensor>(op.args[0]), get<at::Dimname>(op.args[1]), get<bool>(op.args[2])));
  break;

// skip std::tuple<at::Tensor &,at::Tensor &> topk_outf(const at::Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted, at::Tensor & values, at::Tensor & indices)

// skip std::tuple<at::Tensor,at::Tensor> topk(const at::Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted)

case H_ALL:
  set(op.tensor, at::redispatch::all(ks, get<at::Tensor>(op.args[0])));
  break;

case H_ANY:
  set(op.tensor, at::redispatch::any(ks, get<at::Tensor>(op.args[0])));
  break;

case H_RENORM_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::renorm_outf(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<int64_t>(op.args[2]), get<at::Scalar>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_RENORM:
  set(op.tensor, at::redispatch::renorm(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<int64_t>(op.args[2]), get<at::Scalar>(op.args[3])));
  break;

case H_UNFOLD:
  set(op.tensor, at::redispatch::unfold(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<int64_t>(op.args[2]), get<int64_t>(op.args[3])));
  break;

case H_UNFOLD_BACKWARD:
  set(op.tensor, at::redispatch::unfold_backward(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<int64_t>(op.args[2]), get<int64_t>(op.args[3]), get<int64_t>(op.args[4])));
  break;

// skip bool equal(const at::Tensor & self, const at::Tensor & other)

case H_POW_TENSOR_TENSOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::pow_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_POW_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::pow_outf(ks, get<at::Scalar>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_POW_TENSOR_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::pow_outf(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_POW_TENSOR_SCALAR:
  set(op.tensor, at::redispatch::pow(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1])));
  break;

case H_FLOAT_POWER_TENSOR_TENSOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::float_power_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_FLOAT_POWER_TENSOR_TENSOR:
  set(op.tensor, at::redispatch::float_power(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_FLOAT_POWER_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::float_power_outf(ks, get<at::Scalar>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_FLOAT_POWER_SCALAR:
  set(op.tensor, at::redispatch::float_power(ks, get<at::Scalar>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_FLOAT_POWER_TENSOR_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::float_power_outf(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_FLOAT_POWER_TENSOR_SCALAR:
  set(op.tensor, at::redispatch::float_power(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1])));
  break;

case H_FLOAT_POWER__SCALAR:
  init_update_in_place(op.tensor);
  at::redispatch::float_power_(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_FLOAT_POWER__TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::float_power_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_NORMAL_:
  init_update_in_place(op.tensor);
  at::redispatch::normal_(ks, get<at::Tensor>(op.args[0]), get<double>(op.args[1]), get<double>(op.args[2]), get<c10::optional<at::Generator>>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_NORMAL_TENSOR_FLOAT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::normal_outf(ks, get<at::Tensor>(op.args[0]), get<double>(op.args[1]), get<c10::optional<at::Generator>>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_NORMAL_TENSOR_FLOAT:
  set(op.tensor, at::redispatch::normal(ks, get<at::Tensor>(op.args[0]), get<double>(op.args[1]), get<c10::optional<at::Generator>>(op.args[2])));
  break;

case H_NORMAL_FLOAT_TENSOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::normal_outf(ks, get<double>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<at::Generator>>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_NORMAL_FLOAT_TENSOR:
  set(op.tensor, at::redispatch::normal(ks, get<double>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<at::Generator>>(op.args[2])));
  break;

case H_NORMAL_TENSOR_TENSOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::normal_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<at::Generator>>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_NORMAL_TENSOR_TENSOR:
  set(op.tensor, at::redispatch::normal(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<at::Generator>>(op.args[2])));
  break;

case H_NORMAL_FLOAT_FLOAT:
  set(op.tensor, at::redispatch::normal(ks, get<double>(op.args[0]), get<double>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<c10::optional<at::Generator>>(op.args[3]), get<c10::optional<at::ScalarType>>(op.args[4]), get<c10::optional<at::Layout>>(op.args[5]), get<c10::optional<at::Device>>(op.args[6]), get<c10::optional<bool>>(op.args[7])));
  break;

case H_NORMAL_FLOAT_FLOAT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::normal_outf(ks, get<double>(op.args[0]), get<double>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<c10::optional<at::Generator>>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_ALIAS:
  set(op.tensor, at::redispatch::alias(ks, get<at::Tensor>(op.args[0])));
  break;

case H__INDEX_COPY_:
  init_update_in_place(op.tensor);
  at::redispatch::_index_copy_(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H__CUMSUM:
  set(op.tensor, at::redispatch::_cumsum(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1])));
  break;

case H__CUMSUM_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::_cumsum_outf(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H__CUMPROD:
  set(op.tensor, at::redispatch::_cumprod(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1])));
  break;

case H__CUMPROD_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::_cumprod_outf(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

// skip void _amp_foreach_non_finite_check_and_unscale_(at::TensorList self, at::Tensor & found_inf, const at::Tensor & inv_scale)

case H__AMP_UPDATE_SCALE_:
  init_update_in_place(op.tensor);
  at::redispatch::_amp_update_scale_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<double>(op.args[3]), get<double>(op.args[4]), get<int64_t>(op.args[5]));
  end_update_in_place(op.tensor);
  break;

case H__CAT:
  set(op.tensor, at::redispatch::_cat(ks, get<at::TensorList>(op.args[0]), get<int64_t>(op.args[1])));
  break;

case H__CAT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::_cat_outf(ks, get<at::TensorList>(op.args[0]), get<int64_t>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

// skip std::vector<at::Tensor> _foreach_add(at::TensorList tensors, const at::Scalar & scalar)

// skip void _foreach_add_(at::TensorList self, const at::Scalar & scalar)

// skip std::vector<at::Tensor> _foreach_sub(at::TensorList tensors, const at::Scalar & scalar)

// skip void _foreach_sub_(at::TensorList self, const at::Scalar & scalar)

// skip std::vector<at::Tensor> _foreach_mul(at::TensorList tensors, const at::Scalar & scalar)

// skip void _foreach_mul_(at::TensorList self, const at::Scalar & scalar)

// skip std::vector<at::Tensor> _foreach_div(at::TensorList tensors, const at::Scalar & scalar)

// skip void _foreach_div_(at::TensorList self, const at::Scalar & scalar)

// skip std::vector<at::Tensor> _foreach_add(at::TensorList tensors1, at::TensorList tensors2, const at::Scalar & alpha)

// skip void _foreach_add_(at::TensorList self, at::TensorList other, const at::Scalar & alpha)

// skip std::vector<at::Tensor> _foreach_sub(at::TensorList tensors1, at::TensorList tensors2, const at::Scalar & alpha)

// skip void _foreach_sub_(at::TensorList self, at::TensorList other, const at::Scalar & alpha)

// skip std::vector<at::Tensor> _foreach_mul(at::TensorList tensors1, at::TensorList tensors2)

// skip void _foreach_mul_(at::TensorList self, at::TensorList other)

// skip std::vector<at::Tensor> _foreach_div(at::TensorList tensors1, at::TensorList tensors2)

// skip void _foreach_div_(at::TensorList self, at::TensorList other)

// skip std::vector<at::Tensor> _foreach_add(at::TensorList tensors, at::ArrayRef<at::Scalar> scalars)

// skip void _foreach_add_(at::TensorList self, at::ArrayRef<at::Scalar> scalars)

// skip std::vector<at::Tensor> _foreach_sub(at::TensorList tensors, at::ArrayRef<at::Scalar> scalars)

// skip void _foreach_sub_(at::TensorList self, at::ArrayRef<at::Scalar> scalars)

// skip std::vector<at::Tensor> _foreach_div(at::TensorList tensors, at::ArrayRef<at::Scalar> scalars)

// skip void _foreach_div_(at::TensorList self, at::ArrayRef<at::Scalar> scalars)

// skip std::vector<at::Tensor> _foreach_mul(at::TensorList tensors, at::ArrayRef<at::Scalar> scalars)

// skip void _foreach_mul_(at::TensorList self, at::ArrayRef<at::Scalar> scalars)

// skip std::vector<at::Tensor> _foreach_exp(at::TensorList tensors)

// skip void _foreach_zero_(at::TensorList self)

// skip void _foreach_exp_(at::TensorList self)

// skip std::vector<at::Tensor> _foreach_sqrt(at::TensorList tensors)

// skip void _foreach_sqrt_(at::TensorList self)

// skip std::vector<at::Tensor> _foreach_abs(at::TensorList tensors)

// skip void _foreach_abs_(at::TensorList self)

// skip std::vector<at::Tensor> _foreach_acos(at::TensorList tensors)

// skip void _foreach_acos_(at::TensorList self)

// skip std::vector<at::Tensor> _foreach_asin(at::TensorList tensors)

// skip void _foreach_asin_(at::TensorList self)

// skip std::vector<at::Tensor> _foreach_atan(at::TensorList tensors)

// skip void _foreach_atan_(at::TensorList self)

// skip std::vector<at::Tensor> _foreach_ceil(at::TensorList tensors)

// skip void _foreach_ceil_(at::TensorList self)

// skip std::vector<at::Tensor> _foreach_cos(at::TensorList tensors)

// skip void _foreach_cos_(at::TensorList self)

// skip std::vector<at::Tensor> _foreach_cosh(at::TensorList tensors)

// skip void _foreach_cosh_(at::TensorList self)

// skip std::vector<at::Tensor> _foreach_erf(at::TensorList tensors)

// skip void _foreach_erf_(at::TensorList self)

// skip std::vector<at::Tensor> _foreach_erfc(at::TensorList tensors)

// skip void _foreach_erfc_(at::TensorList self)

// skip std::vector<at::Tensor> _foreach_expm1(at::TensorList tensors)

// skip void _foreach_expm1_(at::TensorList self)

// skip std::vector<at::Tensor> _foreach_floor(at::TensorList tensors)

// skip void _foreach_floor_(at::TensorList self)

// skip std::vector<at::Tensor> _foreach_log(at::TensorList tensors)

// skip void _foreach_log_(at::TensorList self)

// skip std::vector<at::Tensor> _foreach_log10(at::TensorList tensors)

// skip void _foreach_log10_(at::TensorList self)

// skip std::vector<at::Tensor> _foreach_log1p(at::TensorList tensors)

// skip void _foreach_log1p_(at::TensorList self)

// skip std::vector<at::Tensor> _foreach_log2(at::TensorList tensors)

// skip void _foreach_log2_(at::TensorList self)

// skip std::vector<at::Tensor> _foreach_neg(at::TensorList tensors)

// skip void _foreach_neg_(at::TensorList self)

// skip std::vector<at::Tensor> _foreach_tan(at::TensorList tensors)

// skip void _foreach_tan_(at::TensorList self)

// skip std::vector<at::Tensor> _foreach_tanh(at::TensorList tensors)

// skip void _foreach_tanh_(at::TensorList self)

// skip std::vector<at::Tensor> _foreach_sin(at::TensorList tensors)

// skip void _foreach_sin_(at::TensorList self)

// skip std::vector<at::Tensor> _foreach_sinh(at::TensorList tensors)

// skip void _foreach_sinh_(at::TensorList self)

// skip std::vector<at::Tensor> _foreach_round(at::TensorList tensors)

// skip void _foreach_round_(at::TensorList self)

// skip std::vector<at::Tensor> _foreach_lgamma(at::TensorList tensors)

// skip void _foreach_lgamma_(at::TensorList self)

// skip std::vector<at::Tensor> _foreach_frac(at::TensorList tensors)

// skip void _foreach_frac_(at::TensorList self)

// skip std::vector<at::Tensor> _foreach_reciprocal(at::TensorList tensors)

// skip void _foreach_reciprocal_(at::TensorList self)

// skip std::vector<at::Tensor> _foreach_sigmoid(at::TensorList tensors)

// skip void _foreach_sigmoid_(at::TensorList self)

// skip std::vector<at::Tensor> _foreach_trunc(at::TensorList tensors)

// skip void _foreach_trunc_(at::TensorList self)

// skip void _foreach_addcdiv_(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Scalar & value)

// skip void _foreach_addcmul_(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Scalar & value)

// skip void _foreach_addcdiv_(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, at::ArrayRef<at::Scalar> scalars)

// skip void _foreach_addcmul_(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, at::ArrayRef<at::Scalar> scalars)

// skip std::vector<at::Tensor> _foreach_addcdiv(at::TensorList input, at::TensorList tensor1, at::TensorList tensor2, const at::Scalar & value)

// skip std::vector<at::Tensor> _foreach_addcmul(at::TensorList input, at::TensorList tensor1, at::TensorList tensor2, const at::Scalar & value)

// skip std::vector<at::Tensor> _foreach_addcdiv(at::TensorList input, at::TensorList tensor1, at::TensorList tensor2, at::ArrayRef<at::Scalar> scalars)

// skip std::vector<at::Tensor> _foreach_addcmul(at::TensorList input, at::TensorList tensor1, at::TensorList tensor2, at::ArrayRef<at::Scalar> scalars)

// skip std::vector<at::Tensor> _foreach_maximum(at::TensorList tensors1, at::TensorList tensors2)

// skip std::vector<at::Tensor> _foreach_minimum(at::TensorList tensors1, at::TensorList tensors2)

case H_BUCKETIZE_TENSOR:
  set(op.tensor, at::redispatch::bucketize(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<bool>(op.args[2]), get<bool>(op.args[3])));
  break;

case H_BUCKETIZE_TENSOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::bucketize_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<bool>(op.args[2]), get<bool>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_BUCKETIZE_SCALAR:
  set(op.tensor, at::redispatch::bucketize(ks, get<at::Scalar>(op.args[0]), get<at::Tensor>(op.args[1]), get<bool>(op.args[2]), get<bool>(op.args[3])));
  break;

case H_SEARCHSORTED_TENSOR:
  set(op.tensor, at::redispatch::searchsorted(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<bool>(op.args[2]), get<bool>(op.args[3])));
  break;

case H_SEARCHSORTED_TENSOR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::searchsorted_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<bool>(op.args[2]), get<bool>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_SEARCHSORTED_SCALAR:
  set(op.tensor, at::redispatch::searchsorted(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<bool>(op.args[2]), get<bool>(op.args[3])));
  break;

case H_MSE_LOSS_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::mse_loss_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<int64_t>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_MSE_LOSS:
  set(op.tensor, at::redispatch::mse_loss(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<int64_t>(op.args[2])));
  break;

case H_MSE_LOSS_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::mse_loss_backward_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<int64_t>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_MSE_LOSS_BACKWARD:
  set(op.tensor, at::redispatch::mse_loss_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<int64_t>(op.args[3])));
  break;

case H_L1_LOSS_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::l1_loss_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<int64_t>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_L1_LOSS:
  set(op.tensor, at::redispatch::l1_loss(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<int64_t>(op.args[2])));
  break;

case H_L1_LOSS_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::l1_loss_backward_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<int64_t>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_L1_LOSS_BACKWARD:
  set(op.tensor, at::redispatch::l1_loss_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<int64_t>(op.args[3])));
  break;

case H_MULTI_MARGIN_LOSS_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::multi_margin_loss_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Scalar>(op.args[2]), get<at::Scalar>(op.args[3]), get<c10::optional<at::Tensor>>(op.args[4]), get<int64_t>(op.args[5]), get<at::Tensor>(op.args[6]));
  end_update_in_place(op.tensor);
  break;

case H_MULTI_MARGIN_LOSS:
  set(op.tensor, at::redispatch::multi_margin_loss(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Scalar>(op.args[2]), get<at::Scalar>(op.args[3]), get<c10::optional<at::Tensor>>(op.args[4]), get<int64_t>(op.args[5])));
  break;

case H_MULTI_MARGIN_LOSS_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::multi_margin_loss_backward_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Scalar>(op.args[3]), get<at::Scalar>(op.args[4]), get<c10::optional<at::Tensor>>(op.args[5]), get<int64_t>(op.args[6]), get<at::Tensor>(op.args[7]));
  end_update_in_place(op.tensor);
  break;

case H_MULTI_MARGIN_LOSS_BACKWARD:
  set(op.tensor, at::redispatch::multi_margin_loss_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Scalar>(op.args[3]), get<at::Scalar>(op.args[4]), get<c10::optional<at::Tensor>>(op.args[5]), get<int64_t>(op.args[6])));
  break;

case H_MULTILABEL_MARGIN_LOSS_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::multilabel_margin_loss_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<int64_t>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_MULTILABEL_MARGIN_LOSS:
  set(op.tensor, at::redispatch::multilabel_margin_loss(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<int64_t>(op.args[2])));
  break;

// skip std::tuple<at::Tensor &,at::Tensor &> multilabel_margin_loss_forward_outf(const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & output, at::Tensor & is_target)

// skip std::tuple<at::Tensor,at::Tensor> multilabel_margin_loss_forward(const at::Tensor & self, const at::Tensor & target, int64_t reduction)

case H_MULTILABEL_MARGIN_LOSS_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::multilabel_margin_loss_backward_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<int64_t>(op.args[3]), get<at::Tensor>(op.args[4]), get<at::Tensor>(op.args[5]));
  end_update_in_place(op.tensor);
  break;

case H_MULTILABEL_MARGIN_LOSS_BACKWARD:
  set(op.tensor, at::redispatch::multilabel_margin_loss_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<int64_t>(op.args[3]), get<at::Tensor>(op.args[4])));
  break;

case H_NLL_LOSS_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::nll_loss_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<at::Tensor>>(op.args[2]), get<int64_t>(op.args[3]), get<int64_t>(op.args[4]), get<at::Tensor>(op.args[5]));
  end_update_in_place(op.tensor);
  break;

case H_NLL_LOSS_ND:
  set(op.tensor, at::redispatch::nll_loss_nd(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<at::Tensor>>(op.args[2]), get<int64_t>(op.args[3]), get<int64_t>(op.args[4])));
  break;

case H_NLL_LOSS:
  set(op.tensor, at::redispatch::nll_loss(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<at::Tensor>>(op.args[2]), get<int64_t>(op.args[3]), get<int64_t>(op.args[4])));
  break;

// skip std::tuple<at::Tensor &,at::Tensor &> nll_loss_forward_outf(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, at::Tensor & output, at::Tensor & total_weight)

// skip std::tuple<at::Tensor,at::Tensor> nll_loss_forward(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index)

case H_NLL_LOSS_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::nll_loss_backward_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<c10::optional<at::Tensor>>(op.args[3]), get<int64_t>(op.args[4]), get<int64_t>(op.args[5]), get<at::Tensor>(op.args[6]), get<at::Tensor>(op.args[7]));
  end_update_in_place(op.tensor);
  break;

case H_NLL_LOSS_BACKWARD:
  set(op.tensor, at::redispatch::nll_loss_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<c10::optional<at::Tensor>>(op.args[3]), get<int64_t>(op.args[4]), get<int64_t>(op.args[5]), get<at::Tensor>(op.args[6])));
  break;

case H_NLL_LOSS2D_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::nll_loss2d_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<at::Tensor>>(op.args[2]), get<int64_t>(op.args[3]), get<int64_t>(op.args[4]), get<at::Tensor>(op.args[5]));
  end_update_in_place(op.tensor);
  break;

case H_NLL_LOSS2D:
  set(op.tensor, at::redispatch::nll_loss2d(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<at::Tensor>>(op.args[2]), get<int64_t>(op.args[3]), get<int64_t>(op.args[4])));
  break;

// skip std::tuple<at::Tensor &,at::Tensor &> nll_loss2d_forward_outf(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, at::Tensor & output, at::Tensor & total_weight)

// skip std::tuple<at::Tensor,at::Tensor> nll_loss2d_forward(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index)

case H_NLL_LOSS2D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::nll_loss2d_backward_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<c10::optional<at::Tensor>>(op.args[3]), get<int64_t>(op.args[4]), get<int64_t>(op.args[5]), get<at::Tensor>(op.args[6]), get<at::Tensor>(op.args[7]));
  end_update_in_place(op.tensor);
  break;

case H_NLL_LOSS2D_BACKWARD:
  set(op.tensor, at::redispatch::nll_loss2d_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<c10::optional<at::Tensor>>(op.args[3]), get<int64_t>(op.args[4]), get<int64_t>(op.args[5]), get<at::Tensor>(op.args[6])));
  break;

case H_SMOOTH_L1_LOSS_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::smooth_l1_loss_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<int64_t>(op.args[2]), get<double>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_SMOOTH_L1_LOSS:
  set(op.tensor, at::redispatch::smooth_l1_loss(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<int64_t>(op.args[2]), get<double>(op.args[3])));
  break;

case H_SMOOTH_L1_LOSS_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::smooth_l1_loss_backward_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<int64_t>(op.args[3]), get<double>(op.args[4]), get<at::Tensor>(op.args[5]));
  end_update_in_place(op.tensor);
  break;

case H_SMOOTH_L1_LOSS_BACKWARD:
  set(op.tensor, at::redispatch::smooth_l1_loss_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<int64_t>(op.args[3]), get<double>(op.args[4])));
  break;

case H_HUBER_LOSS_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::huber_loss_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<int64_t>(op.args[2]), get<double>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_HUBER_LOSS:
  set(op.tensor, at::redispatch::huber_loss(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<int64_t>(op.args[2]), get<double>(op.args[3])));
  break;

case H_HUBER_LOSS_BACKWARD_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::huber_loss_backward_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<int64_t>(op.args[3]), get<double>(op.args[4]), get<at::Tensor>(op.args[5]));
  end_update_in_place(op.tensor);
  break;

case H_HUBER_LOSS_BACKWARD:
  set(op.tensor, at::redispatch::huber_loss_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<int64_t>(op.args[3]), get<double>(op.args[4])));
  break;

case H_SOFT_MARGIN_LOSS_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::soft_margin_loss_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<int64_t>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_SOFT_MARGIN_LOSS:
  set(op.tensor, at::redispatch::soft_margin_loss(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<int64_t>(op.args[2])));
  break;

case H_SOFT_MARGIN_LOSS_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::soft_margin_loss_backward_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<int64_t>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_SOFT_MARGIN_LOSS_BACKWARD:
  set(op.tensor, at::redispatch::soft_margin_loss_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<int64_t>(op.args[3])));
  break;

case H_ELU_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::elu_outf(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Scalar>(op.args[2]), get<at::Scalar>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_ELU:
  set(op.tensor, at::redispatch::elu(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Scalar>(op.args[2]), get<at::Scalar>(op.args[3])));
  break;

case H_ELU_BACKWARD:
  set(op.tensor, at::redispatch::elu_backward(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Scalar>(op.args[2]), get<at::Scalar>(op.args[3]), get<bool>(op.args[4]), get<at::Tensor>(op.args[5])));
  break;

case H_ELU_:
  init_update_in_place(op.tensor);
  at::redispatch::elu_(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Scalar>(op.args[2]), get<at::Scalar>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_GLU_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::glu_outf(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_GLU:
  set(op.tensor, at::redispatch::glu(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1])));
  break;

case H_GLU_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::glu_backward_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<int64_t>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_GLU_BACKWARD:
  set(op.tensor, at::redispatch::glu_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<int64_t>(op.args[2])));
  break;

case H_HARDSIGMOID_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::hardsigmoid_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_HARDSIGMOID:
  set(op.tensor, at::redispatch::hardsigmoid(ks, get<at::Tensor>(op.args[0])));
  break;

case H_HARDSIGMOID_:
  init_update_in_place(op.tensor);
  at::redispatch::hardsigmoid_(ks, get<at::Tensor>(op.args[0]));
  end_update_in_place(op.tensor);
  break;

case H_HARDSIGMOID_BACKWARD:
  set(op.tensor, at::redispatch::hardsigmoid_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_HARDTANH_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::hardtanh_outf(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Scalar>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_HARDTANH:
  set(op.tensor, at::redispatch::hardtanh(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Scalar>(op.args[2])));
  break;

case H_HARDTANH_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::hardtanh_backward_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Scalar>(op.args[2]), get<at::Scalar>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_HARDTANH_BACKWARD:
  set(op.tensor, at::redispatch::hardtanh_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Scalar>(op.args[2]), get<at::Scalar>(op.args[3])));
  break;

case H_HARDTANH_:
  init_update_in_place(op.tensor);
  at::redispatch::hardtanh_(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Scalar>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_HARDSWISH_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::hardswish_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_HARDSWISH:
  set(op.tensor, at::redispatch::hardswish(ks, get<at::Tensor>(op.args[0])));
  break;

case H_HARDSWISH_:
  init_update_in_place(op.tensor);
  at::redispatch::hardswish_(ks, get<at::Tensor>(op.args[0]));
  end_update_in_place(op.tensor);
  break;

case H_HARDSWISH_BACKWARD:
  set(op.tensor, at::redispatch::hardswish_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_LEAKY_RELU_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::leaky_relu_outf(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_LEAKY_RELU:
  set(op.tensor, at::redispatch::leaky_relu(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1])));
  break;

case H_LEAKY_RELU_BACKWARD:
  set(op.tensor, at::redispatch::leaky_relu_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Scalar>(op.args[2]), get<bool>(op.args[3])));
  break;

case H_LEAKY_RELU_:
  init_update_in_place(op.tensor);
  at::redispatch::leaky_relu_(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_LOG_SIGMOID_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::log_sigmoid_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_LOG_SIGMOID:
  set(op.tensor, at::redispatch::log_sigmoid(ks, get<at::Tensor>(op.args[0])));
  break;

// skip std::tuple<at::Tensor &,at::Tensor &> log_sigmoid_forward_outf(const at::Tensor & self, at::Tensor & output, at::Tensor & buffer)

// skip std::tuple<at::Tensor,at::Tensor> log_sigmoid_forward(const at::Tensor & self)

case H_LOG_SIGMOID_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::log_sigmoid_backward_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_LOG_SIGMOID_BACKWARD:
  set(op.tensor, at::redispatch::log_sigmoid_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2])));
  break;

case H_RRELU_WITH_NOISE_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::rrelu_with_noise_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Scalar>(op.args[2]), get<at::Scalar>(op.args[3]), get<bool>(op.args[4]), get<c10::optional<at::Generator>>(op.args[5]), get<at::Tensor>(op.args[6]));
  end_update_in_place(op.tensor);
  break;

case H_RRELU_WITH_NOISE:
  set(op.tensor, at::redispatch::rrelu_with_noise(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Scalar>(op.args[2]), get<at::Scalar>(op.args[3]), get<bool>(op.args[4]), get<c10::optional<at::Generator>>(op.args[5])));
  break;

case H_RRELU_WITH_NOISE_BACKWARD:
  set(op.tensor, at::redispatch::rrelu_with_noise_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Scalar>(op.args[3]), get<at::Scalar>(op.args[4]), get<bool>(op.args[5]), get<bool>(op.args[6])));
  break;

case H_RRELU_WITH_NOISE_:
  init_update_in_place(op.tensor);
  at::redispatch::rrelu_with_noise_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Scalar>(op.args[2]), get<at::Scalar>(op.args[3]), get<bool>(op.args[4]), get<c10::optional<at::Generator>>(op.args[5]));
  end_update_in_place(op.tensor);
  break;

case H_SOFTPLUS_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::softplus_outf(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Scalar>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_SOFTPLUS:
  set(op.tensor, at::redispatch::softplus(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Scalar>(op.args[2])));
  break;

case H_SOFTPLUS_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::softplus_backward_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Scalar>(op.args[2]), get<at::Scalar>(op.args[3]), get<at::Tensor>(op.args[4]), get<at::Tensor>(op.args[5]));
  end_update_in_place(op.tensor);
  break;

case H_SOFTPLUS_BACKWARD:
  set(op.tensor, at::redispatch::softplus_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Scalar>(op.args[2]), get<at::Scalar>(op.args[3]), get<at::Tensor>(op.args[4])));
  break;

case H_SOFTSHRINK_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::softshrink_outf(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_SOFTSHRINK:
  set(op.tensor, at::redispatch::softshrink(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1])));
  break;

case H_SOFTSHRINK_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::softshrink_backward_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Scalar>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_SOFTSHRINK_BACKWARD:
  set(op.tensor, at::redispatch::softshrink_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Scalar>(op.args[2])));
  break;

case H_ADAPTIVE_AVG_POOL2D_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::adaptive_avg_pool2d_outf(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_ADAPTIVE_AVG_POOL2D:
  set(op.tensor, at::redispatch::adaptive_avg_pool2d(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1])));
  break;

case H_MKLDNN_ADAPTIVE_AVG_POOL2D:
  set(op.tensor, at::redispatch::mkldnn_adaptive_avg_pool2d(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1])));
  break;

case H_MKLDNN_ADAPTIVE_AVG_POOL2D_BACKWARD:
  set(op.tensor, at::redispatch::mkldnn_adaptive_avg_pool2d_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H__ADAPTIVE_AVG_POOL2D:
  set(op.tensor, at::redispatch::_adaptive_avg_pool2d(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1])));
  break;

case H__ADAPTIVE_AVG_POOL2D_BACKWARD:
  set(op.tensor, at::redispatch::_adaptive_avg_pool2d_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_ADAPTIVE_AVG_POOL3D_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::adaptive_avg_pool3d_outf(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_ADAPTIVE_AVG_POOL3D:
  set(op.tensor, at::redispatch::adaptive_avg_pool3d(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1])));
  break;

case H__ADAPTIVE_AVG_POOL3D:
  set(op.tensor, at::redispatch::_adaptive_avg_pool3d(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1])));
  break;

case H_ADAPTIVE_AVG_POOL3D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::adaptive_avg_pool3d_backward_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H__ADAPTIVE_AVG_POOL3D_BACKWARD:
  set(op.tensor, at::redispatch::_adaptive_avg_pool3d_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

// skip std::tuple<at::Tensor &,at::Tensor &> adaptive_max_pool2d_outf(const at::Tensor & self, at::IntArrayRef output_size, at::Tensor & out, at::Tensor & indices)

case H_ADAPTIVE_MAX_POOL2D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::adaptive_max_pool2d_backward_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

// skip std::tuple<at::Tensor &,at::Tensor &> adaptive_max_pool3d_outf(const at::Tensor & self, at::IntArrayRef output_size, at::Tensor & out, at::Tensor & indices)

case H_ADAPTIVE_MAX_POOL3D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::adaptive_max_pool3d_backward_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_AVG_POOL2D_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::avg_pool2d_outf(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<bool>(op.args[4]), get<bool>(op.args[5]), get<c10::optional<int64_t>>(op.args[6]), get<at::Tensor>(op.args[7]));
  end_update_in_place(op.tensor);
  break;

case H_AVG_POOL2D:
  set(op.tensor, at::redispatch::avg_pool2d(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<bool>(op.args[4]), get<bool>(op.args[5]), get<c10::optional<int64_t>>(op.args[6])));
  break;

case H_AVG_POOL2D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::avg_pool2d_backward_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<bool>(op.args[5]), get<bool>(op.args[6]), get<c10::optional<int64_t>>(op.args[7]), get<at::Tensor>(op.args[8]));
  end_update_in_place(op.tensor);
  break;

case H_AVG_POOL2D_BACKWARD:
  set(op.tensor, at::redispatch::avg_pool2d_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<bool>(op.args[5]), get<bool>(op.args[6]), get<c10::optional<int64_t>>(op.args[7])));
  break;

case H_AVG_POOL3D_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::avg_pool3d_outf(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<bool>(op.args[4]), get<bool>(op.args[5]), get<c10::optional<int64_t>>(op.args[6]), get<at::Tensor>(op.args[7]));
  end_update_in_place(op.tensor);
  break;

case H_AVG_POOL3D:
  set(op.tensor, at::redispatch::avg_pool3d(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<bool>(op.args[4]), get<bool>(op.args[5]), get<c10::optional<int64_t>>(op.args[6])));
  break;

case H_AVG_POOL3D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::avg_pool3d_backward_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<bool>(op.args[5]), get<bool>(op.args[6]), get<c10::optional<int64_t>>(op.args[7]), get<at::Tensor>(op.args[8]));
  end_update_in_place(op.tensor);
  break;

case H_AVG_POOL3D_BACKWARD:
  set(op.tensor, at::redispatch::avg_pool3d_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<bool>(op.args[5]), get<bool>(op.args[6]), get<c10::optional<int64_t>>(op.args[7])));
  break;

// skip std::tuple<at::Tensor &,at::Tensor &> fractional_max_pool2d_outf(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef output_size, const at::Tensor & random_samples, at::Tensor & output, at::Tensor & indices)

case H_FRACTIONAL_MAX_POOL2D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::fractional_max_pool2d_backward_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::Tensor>(op.args[4]), get<at::Tensor>(op.args[5]));
  end_update_in_place(op.tensor);
  break;

case H_FRACTIONAL_MAX_POOL2D_BACKWARD:
  set(op.tensor, at::redispatch::fractional_max_pool2d_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::Tensor>(op.args[4])));
  break;

// skip std::tuple<at::Tensor &,at::Tensor &> fractional_max_pool3d_outf(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef output_size, const at::Tensor & random_samples, at::Tensor & output, at::Tensor & indices)

// skip std::tuple<at::Tensor,at::Tensor> fractional_max_pool3d(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef output_size, const at::Tensor & random_samples)

case H_FRACTIONAL_MAX_POOL3D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::fractional_max_pool3d_backward_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::Tensor>(op.args[4]), get<at::Tensor>(op.args[5]));
  end_update_in_place(op.tensor);
  break;

case H_FRACTIONAL_MAX_POOL3D_BACKWARD:
  set(op.tensor, at::redispatch::fractional_max_pool3d_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::Tensor>(op.args[4])));
  break;

// skip std::tuple<at::Tensor &,at::Tensor &> max_pool2d_with_indices_outf(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, at::Tensor & out, at::Tensor & indices)

case H_MAX_POOL2D_WITH_INDICES_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::max_pool2d_with_indices_backward_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<bool>(op.args[6]), get<at::Tensor>(op.args[7]), get<at::Tensor>(op.args[8]));
  end_update_in_place(op.tensor);
  break;

case H_MAX_POOL2D_WITH_INDICES_BACKWARD:
  set(op.tensor, at::redispatch::max_pool2d_with_indices_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<bool>(op.args[6]), get<at::Tensor>(op.args[7])));
  break;

// skip std::tuple<at::Tensor &,at::Tensor &> max_pool3d_with_indices_outf(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, at::Tensor & out, at::Tensor & indices)

// skip std::tuple<at::Tensor,at::Tensor> max_pool3d_with_indices(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode)

case H_MAX_POOL3D_WITH_INDICES_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::max_pool3d_with_indices_backward_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<bool>(op.args[6]), get<at::Tensor>(op.args[7]), get<at::Tensor>(op.args[8]));
  end_update_in_place(op.tensor);
  break;

case H_MAX_POOL3D_WITH_INDICES_BACKWARD:
  set(op.tensor, at::redispatch::max_pool3d_with_indices_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<bool>(op.args[6]), get<at::Tensor>(op.args[7])));
  break;

case H_MAX_UNPOOL2D_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::max_unpool2d_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_MAX_UNPOOL2D:
  set(op.tensor, at::redispatch::max_unpool2d(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2])));
  break;

case H_MAX_UNPOOL2D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::max_unpool2d_backward_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_MAX_UNPOOL2D_BACKWARD:
  set(op.tensor, at::redispatch::max_unpool2d_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::IntArrayRef>(op.args[3])));
  break;

case H_MAX_UNPOOL3D_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::max_unpool3d_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::Tensor>(op.args[5]));
  end_update_in_place(op.tensor);
  break;

case H_MAX_UNPOOL3D:
  set(op.tensor, at::redispatch::max_unpool3d(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4])));
  break;

case H_MAX_UNPOOL3D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::max_unpool3d_backward_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<at::Tensor>(op.args[6]));
  end_update_in_place(op.tensor);
  break;

case H_MAX_UNPOOL3D_BACKWARD:
  set(op.tensor, at::redispatch::max_unpool3d_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5])));
  break;

case H_REFLECTION_PAD1D_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::reflection_pad1d_outf(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_REFLECTION_PAD1D:
  set(op.tensor, at::redispatch::reflection_pad1d(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1])));
  break;

case H_REFLECTION_PAD1D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::reflection_pad1d_backward_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_REFLECTION_PAD1D_BACKWARD:
  set(op.tensor, at::redispatch::reflection_pad1d_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2])));
  break;

case H_REFLECTION_PAD2D_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::reflection_pad2d_outf(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_REFLECTION_PAD2D:
  set(op.tensor, at::redispatch::reflection_pad2d(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1])));
  break;

case H_REFLECTION_PAD2D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::reflection_pad2d_backward_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_REFLECTION_PAD2D_BACKWARD:
  set(op.tensor, at::redispatch::reflection_pad2d_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2])));
  break;

case H_REPLICATION_PAD1D_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::replication_pad1d_outf(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_REPLICATION_PAD1D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::replication_pad1d_backward_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_REPLICATION_PAD2D_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::replication_pad2d_outf(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_REPLICATION_PAD2D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::replication_pad2d_backward_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_REPLICATION_PAD2D_BACKWARD:
  set(op.tensor, at::redispatch::replication_pad2d_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2])));
  break;

case H_REPLICATION_PAD3D_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::replication_pad3d_outf(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_REPLICATION_PAD3D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::replication_pad3d_backward_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_REPLICATION_PAD3D_BACKWARD:
  set(op.tensor, at::redispatch::replication_pad3d_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2])));
  break;

case H_UPSAMPLE_LINEAR1D_VEC:
  set(op.tensor, at::redispatch::upsample_linear1d(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::IntArrayRef>>(op.args[1]), get<bool>(op.args[2]), get<c10::optional<at::ArrayRef<double>>>(op.args[3])));
  break;

case H_UPSAMPLE_LINEAR1D_BACKWARD_VEC:
  set(op.tensor, at::redispatch::upsample_linear1d_backward(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::IntArrayRef>>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<bool>(op.args[3]), get<c10::optional<at::ArrayRef<double>>>(op.args[4])));
  break;

case H_UPSAMPLE_BILINEAR2D_VEC:
  set(op.tensor, at::redispatch::upsample_bilinear2d(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::IntArrayRef>>(op.args[1]), get<bool>(op.args[2]), get<c10::optional<at::ArrayRef<double>>>(op.args[3])));
  break;

case H_UPSAMPLE_BILINEAR2D_BACKWARD_VEC:
  set(op.tensor, at::redispatch::upsample_bilinear2d_backward(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::IntArrayRef>>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<bool>(op.args[3]), get<c10::optional<at::ArrayRef<double>>>(op.args[4])));
  break;

case H_UPSAMPLE_TRILINEAR3D_VEC:
  set(op.tensor, at::redispatch::upsample_trilinear3d(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::IntArrayRef>>(op.args[1]), get<bool>(op.args[2]), get<c10::optional<at::ArrayRef<double>>>(op.args[3])));
  break;

case H_UPSAMPLE_TRILINEAR3D_BACKWARD_VEC:
  set(op.tensor, at::redispatch::upsample_trilinear3d_backward(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::IntArrayRef>>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<bool>(op.args[3]), get<c10::optional<at::ArrayRef<double>>>(op.args[4])));
  break;

case H_UPSAMPLE_BICUBIC2D_VEC:
  set(op.tensor, at::redispatch::upsample_bicubic2d(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::IntArrayRef>>(op.args[1]), get<bool>(op.args[2]), get<c10::optional<at::ArrayRef<double>>>(op.args[3])));
  break;

case H_UPSAMPLE_BICUBIC2D_BACKWARD_VEC:
  set(op.tensor, at::redispatch::upsample_bicubic2d_backward(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::IntArrayRef>>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<bool>(op.args[3]), get<c10::optional<at::ArrayRef<double>>>(op.args[4])));
  break;

case H_UPSAMPLE_NEAREST1D_VEC:
  set(op.tensor, at::redispatch::upsample_nearest1d(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::IntArrayRef>>(op.args[1]), get<c10::optional<at::ArrayRef<double>>>(op.args[2])));
  break;

case H_UPSAMPLE_NEAREST1D_BACKWARD_VEC:
  set(op.tensor, at::redispatch::upsample_nearest1d_backward(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::IntArrayRef>>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<c10::optional<at::ArrayRef<double>>>(op.args[3])));
  break;

case H_UPSAMPLE_NEAREST2D_VEC:
  set(op.tensor, at::redispatch::upsample_nearest2d(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::IntArrayRef>>(op.args[1]), get<c10::optional<at::ArrayRef<double>>>(op.args[2])));
  break;

case H_UPSAMPLE_NEAREST2D_BACKWARD_VEC:
  set(op.tensor, at::redispatch::upsample_nearest2d_backward(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::IntArrayRef>>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<c10::optional<at::ArrayRef<double>>>(op.args[3])));
  break;

case H_UPSAMPLE_NEAREST3D_VEC:
  set(op.tensor, at::redispatch::upsample_nearest3d(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::IntArrayRef>>(op.args[1]), get<c10::optional<at::ArrayRef<double>>>(op.args[2])));
  break;

case H_UPSAMPLE_NEAREST3D_BACKWARD_VEC:
  set(op.tensor, at::redispatch::upsample_nearest3d_backward(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::IntArrayRef>>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<c10::optional<at::ArrayRef<double>>>(op.args[3])));
  break;

case H_UPSAMPLE_LINEAR1D_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::upsample_linear1d_outf(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<bool>(op.args[2]), get<c10::optional<double>>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_UPSAMPLE_LINEAR1D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::upsample_linear1d_backward_outf(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<bool>(op.args[3]), get<c10::optional<double>>(op.args[4]), get<at::Tensor>(op.args[5]));
  end_update_in_place(op.tensor);
  break;

case H_UPSAMPLE_BILINEAR2D_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::upsample_bilinear2d_outf(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<bool>(op.args[2]), get<c10::optional<double>>(op.args[3]), get<c10::optional<double>>(op.args[4]), get<at::Tensor>(op.args[5]));
  end_update_in_place(op.tensor);
  break;

case H_UPSAMPLE_BILINEAR2D:
  set(op.tensor, at::redispatch::upsample_bilinear2d(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<bool>(op.args[2]), get<c10::optional<double>>(op.args[3]), get<c10::optional<double>>(op.args[4])));
  break;

case H_UPSAMPLE_BILINEAR2D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::upsample_bilinear2d_backward_outf(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<bool>(op.args[3]), get<c10::optional<double>>(op.args[4]), get<c10::optional<double>>(op.args[5]), get<at::Tensor>(op.args[6]));
  end_update_in_place(op.tensor);
  break;

case H_UPSAMPLE_BICUBIC2D_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::upsample_bicubic2d_outf(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<bool>(op.args[2]), get<c10::optional<double>>(op.args[3]), get<c10::optional<double>>(op.args[4]), get<at::Tensor>(op.args[5]));
  end_update_in_place(op.tensor);
  break;

case H_UPSAMPLE_BICUBIC2D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::upsample_bicubic2d_backward_outf(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<bool>(op.args[3]), get<c10::optional<double>>(op.args[4]), get<c10::optional<double>>(op.args[5]), get<at::Tensor>(op.args[6]));
  end_update_in_place(op.tensor);
  break;

case H_UPSAMPLE_TRILINEAR3D_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::upsample_trilinear3d_outf(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<bool>(op.args[2]), get<c10::optional<double>>(op.args[3]), get<c10::optional<double>>(op.args[4]), get<c10::optional<double>>(op.args[5]), get<at::Tensor>(op.args[6]));
  end_update_in_place(op.tensor);
  break;

case H_UPSAMPLE_TRILINEAR3D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::upsample_trilinear3d_backward_outf(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<bool>(op.args[3]), get<c10::optional<double>>(op.args[4]), get<c10::optional<double>>(op.args[5]), get<c10::optional<double>>(op.args[6]), get<at::Tensor>(op.args[7]));
  end_update_in_place(op.tensor);
  break;

case H_UPSAMPLE_NEAREST1D_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::upsample_nearest1d_outf(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<c10::optional<double>>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_UPSAMPLE_NEAREST1D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::upsample_nearest1d_backward_outf(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<c10::optional<double>>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_UPSAMPLE_NEAREST2D_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::upsample_nearest2d_outf(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<c10::optional<double>>(op.args[2]), get<c10::optional<double>>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_UPSAMPLE_NEAREST2D:
  set(op.tensor, at::redispatch::upsample_nearest2d(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<c10::optional<double>>(op.args[2]), get<c10::optional<double>>(op.args[3])));
  break;

case H_UPSAMPLE_NEAREST2D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::upsample_nearest2d_backward_outf(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<c10::optional<double>>(op.args[3]), get<c10::optional<double>>(op.args[4]), get<at::Tensor>(op.args[5]));
  end_update_in_place(op.tensor);
  break;

case H_UPSAMPLE_NEAREST3D_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::upsample_nearest3d_outf(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<c10::optional<double>>(op.args[2]), get<c10::optional<double>>(op.args[3]), get<c10::optional<double>>(op.args[4]), get<at::Tensor>(op.args[5]));
  end_update_in_place(op.tensor);
  break;

case H_UPSAMPLE_NEAREST3D:
  set(op.tensor, at::redispatch::upsample_nearest3d(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<c10::optional<double>>(op.args[2]), get<c10::optional<double>>(op.args[3]), get<c10::optional<double>>(op.args[4])));
  break;

case H_UPSAMPLE_NEAREST3D_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::upsample_nearest3d_backward_outf(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<c10::optional<double>>(op.args[3]), get<c10::optional<double>>(op.args[4]), get<c10::optional<double>>(op.args[5]), get<at::Tensor>(op.args[6]));
  end_update_in_place(op.tensor);
  break;

case H_SIGMOID_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::sigmoid_backward_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_SIGMOID_BACKWARD:
  set(op.tensor, at::redispatch::sigmoid_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_LOGIT_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::logit_backward_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<double>>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_LOGIT_BACKWARD:
  set(op.tensor, at::redispatch::logit_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<double>>(op.args[2])));
  break;

case H_TANH_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::tanh_backward_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_TANH_BACKWARD:
  set(op.tensor, at::redispatch::tanh_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_SLOW_CONV_TRANSPOSE2D_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::slow_conv_transpose2d_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<c10::optional<at::Tensor>>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<at::IntArrayRef>(op.args[6]), get<at::IntArrayRef>(op.args[7]), get<at::Tensor>(op.args[8]));
  end_update_in_place(op.tensor);
  break;

case H_SLOW_CONV_TRANSPOSE2D:
  set(op.tensor, at::redispatch::slow_conv_transpose2d(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<c10::optional<at::Tensor>>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<at::IntArrayRef>(op.args[6]), get<at::IntArrayRef>(op.args[7])));
  break;

// skip std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> slow_conv_transpose2d_backward_outf(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation, const at::Tensor & columns, const at::Tensor & ones, at::Tensor & grad_input, at::Tensor & grad_weight, at::Tensor & grad_bias)

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor> slow_conv_transpose2d_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation, const at::Tensor & columns, const at::Tensor & ones, std::array<bool,3> output_mask)

case H_SLOW_CONV_TRANSPOSE3D_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::slow_conv_transpose3d_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<c10::optional<at::Tensor>>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<at::IntArrayRef>(op.args[6]), get<at::IntArrayRef>(op.args[7]), get<at::Tensor>(op.args[8]));
  end_update_in_place(op.tensor);
  break;

case H_SLOW_CONV_TRANSPOSE3D:
  set(op.tensor, at::redispatch::slow_conv_transpose3d(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<c10::optional<at::Tensor>>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<at::IntArrayRef>(op.args[6]), get<at::IntArrayRef>(op.args[7])));
  break;

// skip std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> slow_conv_transpose3d_backward_outf(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation, const at::Tensor & finput, const at::Tensor & fgrad_input, at::Tensor & grad_input, at::Tensor & grad_weight, at::Tensor & grad_bias)

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor> slow_conv_transpose3d_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation, const at::Tensor & finput, const at::Tensor & fgrad_input, std::array<bool,3> output_mask)

case H_THNN_CONV2D_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::thnn_conv2d_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<c10::optional<at::Tensor>>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<at::Tensor>(op.args[6]));
  end_update_in_place(op.tensor);
  break;

case H_THNN_CONV2D:
  set(op.tensor, at::redispatch::thnn_conv2d(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<c10::optional<at::Tensor>>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5])));
  break;

// skip std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> thnn_conv2d_forward_outf(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::Tensor & output, at::Tensor & finput, at::Tensor & fgrad_input)

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor> thnn_conv2d_forward(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding)

// skip std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> thnn_conv2d_backward_outf(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, const at::Tensor & finput, const at::Tensor & fgrad_input, at::Tensor & grad_input, at::Tensor & grad_weight, at::Tensor & grad_bias)

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor> thnn_conv2d_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, const at::Tensor & finput, const at::Tensor & fgrad_input, std::array<bool,3> output_mask)

case H_THNN_CONV_DEPTHWISE2D_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::thnn_conv_depthwise2d_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<c10::optional<at::Tensor>>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<at::IntArrayRef>(op.args[6]), get<at::Tensor>(op.args[7]));
  end_update_in_place(op.tensor);
  break;

case H_THNN_CONV_DEPTHWISE2D:
  set(op.tensor, at::redispatch::thnn_conv_depthwise2d(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<c10::optional<at::Tensor>>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<at::IntArrayRef>(op.args[6])));
  break;

case H_THNN_CONV_DEPTHWISE2D_FORWARD_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::thnn_conv_depthwise2d_forward_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<c10::optional<at::Tensor>>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<at::IntArrayRef>(op.args[6]), get<at::Tensor>(op.args[7]));
  end_update_in_place(op.tensor);
  break;

case H_THNN_CONV_DEPTHWISE2D_FORWARD:
  set(op.tensor, at::redispatch::thnn_conv_depthwise2d_forward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<c10::optional<at::Tensor>>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<at::IntArrayRef>(op.args[6])));
  break;

// skip std::tuple<at::Tensor &,at::Tensor &> thnn_conv_depthwise2d_backward_outf(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, at::Tensor & grad_input, at::Tensor & grad_weight)

// skip std::tuple<at::Tensor,at::Tensor> thnn_conv_depthwise2d_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, std::array<bool,2> output_mask)

case H_CONV_DEPTHWISE3D:
  set(op.tensor, at::redispatch::conv_depthwise3d(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<c10::optional<at::Tensor>>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<at::IntArrayRef>(op.args[6])));
  break;

// skip std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> conv_depthwise3d_backward_outf(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, at::Tensor & grad_input, at::Tensor & grad_weight, at::Tensor & grad_bias)

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor> conv_depthwise3d_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, std::array<bool,3> output_mask)

case H_SLOW_CONV3D_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::slow_conv3d_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<c10::optional<at::Tensor>>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<at::Tensor>(op.args[6]));
  end_update_in_place(op.tensor);
  break;

case H_SLOW_CONV3D:
  set(op.tensor, at::redispatch::slow_conv3d(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<c10::optional<at::Tensor>>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5])));
  break;

// skip std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> slow_conv3d_forward_outf(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::Tensor & output, at::Tensor & finput, at::Tensor & fgrad_input)

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor> slow_conv3d_forward(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding)

// skip std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> slow_conv3d_backward_outf(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, const at::Tensor & finput, const at::Tensor & fgrad_input, at::Tensor & grad_input, at::Tensor & grad_weight, at::Tensor & grad_bias)

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor> slow_conv3d_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, const at::Tensor & finput, const at::Tensor & fgrad_input, std::array<bool,3> output_mask)

case H_SLOW_CONV_DILATED2D:
  set(op.tensor, at::redispatch::slow_conv_dilated2d(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<c10::optional<at::Tensor>>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<at::IntArrayRef>(op.args[6])));
  break;

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor> slow_conv_dilated2d_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, std::array<bool,3> output_mask)

case H_SLOW_CONV_DILATED3D:
  set(op.tensor, at::redispatch::slow_conv_dilated3d(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<c10::optional<at::Tensor>>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<at::IntArrayRef>(op.args[6])));
  break;

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor> slow_conv_dilated3d_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, std::array<bool,3> output_mask)

case H_COL2IM_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::col2im_outf(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<at::Tensor>(op.args[6]));
  end_update_in_place(op.tensor);
  break;

case H_COL2IM:
  set(op.tensor, at::redispatch::col2im(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5])));
  break;

case H_COL2IM_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::col2im_backward_outf(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::Tensor>(op.args[5]));
  end_update_in_place(op.tensor);
  break;

case H_COL2IM_BACKWARD:
  set(op.tensor, at::redispatch::col2im_backward(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4])));
  break;

case H_COLUMN_STACK:
  set(op.tensor, at::redispatch::column_stack(ks, get<at::TensorList>(op.args[0])));
  break;

case H_COLUMN_STACK_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::column_stack_outf(ks, get<at::TensorList>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_IM2COL_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::im2col_outf(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::Tensor>(op.args[5]));
  end_update_in_place(op.tensor);
  break;

case H_IM2COL:
  set(op.tensor, at::redispatch::im2col(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4])));
  break;

case H_IM2COL_BACKWARD_GRAD_INPUT:
  init_update_in_place(op.tensor);
  at::redispatch::im2col_backward_outf(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5]), get<at::Tensor>(op.args[6]));
  end_update_in_place(op.tensor);
  break;

case H_IM2COL_BACKWARD:
  set(op.tensor, at::redispatch::im2col_backward(ks, get<at::Tensor>(op.args[0]), get<at::IntArrayRef>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<at::IntArrayRef>(op.args[3]), get<at::IntArrayRef>(op.args[4]), get<at::IntArrayRef>(op.args[5])));
  break;

case H_ISFINITE:
  set(op.tensor, at::redispatch::isfinite(ks, get<at::Tensor>(op.args[0])));
  break;

case H_ISINF:
  set(op.tensor, at::redispatch::isinf(ks, get<at::Tensor>(op.args[0])));
  break;

// skip void record_stream(at::Tensor & self, at::Stream s)

case H_ISPOSINF:
  set(op.tensor, at::redispatch::isposinf(ks, get<at::Tensor>(op.args[0])));
  break;

case H_ISPOSINF_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::isposinf_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_ISNEGINF:
  set(op.tensor, at::redispatch::isneginf(ks, get<at::Tensor>(op.args[0])));
  break;

case H_ISNEGINF_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::isneginf_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H__ADD_BATCH_DIM:
  set(op.tensor, at::redispatch::_add_batch_dim(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<int64_t>(op.args[2])));
  break;

case H__REMOVE_BATCH_DIM:
  set(op.tensor, at::redispatch::_remove_batch_dim(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<int64_t>(op.args[2]), get<int64_t>(op.args[3])));
  break;

case H_SPECIAL_ENTR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::special_entr_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_SPECIAL_EXPM1:
  set(op.tensor, at::redispatch::special_expm1(ks, get<at::Tensor>(op.args[0])));
  break;

case H_SPECIAL_EXPM1_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::special_expm1_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_SPECIAL_EXP2:
  set(op.tensor, at::redispatch::special_exp2(ks, get<at::Tensor>(op.args[0])));
  break;

case H_SPECIAL_EXP2_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::special_exp2_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_SPECIAL_GAMMALN:
  set(op.tensor, at::redispatch::special_gammaln(ks, get<at::Tensor>(op.args[0])));
  break;

case H_SPECIAL_GAMMALN_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::special_gammaln_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_SPECIAL_ERF:
  set(op.tensor, at::redispatch::special_erf(ks, get<at::Tensor>(op.args[0])));
  break;

case H_SPECIAL_ERF_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::special_erf_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_SPECIAL_ERFC:
  set(op.tensor, at::redispatch::special_erfc(ks, get<at::Tensor>(op.args[0])));
  break;

case H_SPECIAL_ERFC_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::special_erfc_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_SPECIAL_ERFINV:
  set(op.tensor, at::redispatch::special_erfinv(ks, get<at::Tensor>(op.args[0])));
  break;

case H_SPECIAL_ERFINV_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::special_erfinv_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_SPECIAL_XLOG1PY_SELF_SCALAR:
  set(op.tensor, at::redispatch::special_xlog1py(ks, get<at::Scalar>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_SPECIAL_XLOG1PY_OTHER_SCALAR:
  set(op.tensor, at::redispatch::special_xlog1py(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1])));
  break;

case H_SPECIAL_XLOG1PY_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::special_xlog1py_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_SPECIAL_XLOG1PY_SELF_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::special_xlog1py_outf(ks, get<at::Scalar>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_SPECIAL_XLOG1PY_OTHER_SCALAR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::special_xlog1py_outf(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_SPECIAL_I0E_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::special_i0e_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_SPECIAL_LOGIT:
  set(op.tensor, at::redispatch::special_logit(ks, get<at::Tensor>(op.args[0]), get<c10::optional<double>>(op.args[1])));
  break;

case H_SPECIAL_LOGIT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::special_logit_outf(ks, get<at::Tensor>(op.args[0]), get<c10::optional<double>>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_SPECIAL_EXPIT:
  set(op.tensor, at::redispatch::special_expit(ks, get<at::Tensor>(op.args[0])));
  break;

case H_SPECIAL_EXPIT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::special_expit_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_FFT_FFT:
  set(op.tensor, at::redispatch::fft_fft(ks, get<at::Tensor>(op.args[0]), get<c10::optional<int64_t>>(op.args[1]), get<int64_t>(op.args[2]), get<c10::optional<std::string>>(op.args[3])));
  break;

case H_FFT_FFT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::fft_fft_outf(ks, get<at::Tensor>(op.args[0]), get<c10::optional<int64_t>>(op.args[1]), get<int64_t>(op.args[2]), get<c10::optional<std::string>>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_FFT_IFFT:
  set(op.tensor, at::redispatch::fft_ifft(ks, get<at::Tensor>(op.args[0]), get<c10::optional<int64_t>>(op.args[1]), get<int64_t>(op.args[2]), get<c10::optional<std::string>>(op.args[3])));
  break;

case H_FFT_IFFT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::fft_ifft_outf(ks, get<at::Tensor>(op.args[0]), get<c10::optional<int64_t>>(op.args[1]), get<int64_t>(op.args[2]), get<c10::optional<std::string>>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_FFT_RFFT:
  set(op.tensor, at::redispatch::fft_rfft(ks, get<at::Tensor>(op.args[0]), get<c10::optional<int64_t>>(op.args[1]), get<int64_t>(op.args[2]), get<c10::optional<std::string>>(op.args[3])));
  break;

case H_FFT_RFFT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::fft_rfft_outf(ks, get<at::Tensor>(op.args[0]), get<c10::optional<int64_t>>(op.args[1]), get<int64_t>(op.args[2]), get<c10::optional<std::string>>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_FFT_IRFFT:
  set(op.tensor, at::redispatch::fft_irfft(ks, get<at::Tensor>(op.args[0]), get<c10::optional<int64_t>>(op.args[1]), get<int64_t>(op.args[2]), get<c10::optional<std::string>>(op.args[3])));
  break;

case H_FFT_IRFFT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::fft_irfft_outf(ks, get<at::Tensor>(op.args[0]), get<c10::optional<int64_t>>(op.args[1]), get<int64_t>(op.args[2]), get<c10::optional<std::string>>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_FFT_HFFT:
  set(op.tensor, at::redispatch::fft_hfft(ks, get<at::Tensor>(op.args[0]), get<c10::optional<int64_t>>(op.args[1]), get<int64_t>(op.args[2]), get<c10::optional<std::string>>(op.args[3])));
  break;

case H_FFT_HFFT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::fft_hfft_outf(ks, get<at::Tensor>(op.args[0]), get<c10::optional<int64_t>>(op.args[1]), get<int64_t>(op.args[2]), get<c10::optional<std::string>>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_FFT_IHFFT:
  set(op.tensor, at::redispatch::fft_ihfft(ks, get<at::Tensor>(op.args[0]), get<c10::optional<int64_t>>(op.args[1]), get<int64_t>(op.args[2]), get<c10::optional<std::string>>(op.args[3])));
  break;

case H_FFT_IHFFT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::fft_ihfft_outf(ks, get<at::Tensor>(op.args[0]), get<c10::optional<int64_t>>(op.args[1]), get<int64_t>(op.args[2]), get<c10::optional<std::string>>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_FFT_FFT2:
  set(op.tensor, at::redispatch::fft_fft2(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::IntArrayRef>>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<c10::optional<std::string>>(op.args[3])));
  break;

case H_FFT_FFT2_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::fft_fft2_outf(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::IntArrayRef>>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<c10::optional<std::string>>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_FFT_IFFT2:
  set(op.tensor, at::redispatch::fft_ifft2(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::IntArrayRef>>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<c10::optional<std::string>>(op.args[3])));
  break;

case H_FFT_IFFT2_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::fft_ifft2_outf(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::IntArrayRef>>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<c10::optional<std::string>>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_FFT_RFFT2:
  set(op.tensor, at::redispatch::fft_rfft2(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::IntArrayRef>>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<c10::optional<std::string>>(op.args[3])));
  break;

case H_FFT_RFFT2_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::fft_rfft2_outf(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::IntArrayRef>>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<c10::optional<std::string>>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_FFT_IRFFT2:
  set(op.tensor, at::redispatch::fft_irfft2(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::IntArrayRef>>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<c10::optional<std::string>>(op.args[3])));
  break;

case H_FFT_IRFFT2_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::fft_irfft2_outf(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::IntArrayRef>>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<c10::optional<std::string>>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_FFT_FFTN:
  set(op.tensor, at::redispatch::fft_fftn(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::IntArrayRef>>(op.args[1]), get<c10::optional<at::IntArrayRef>>(op.args[2]), get<c10::optional<std::string>>(op.args[3])));
  break;

case H_FFT_FFTN_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::fft_fftn_outf(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::IntArrayRef>>(op.args[1]), get<c10::optional<at::IntArrayRef>>(op.args[2]), get<c10::optional<std::string>>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_FFT_IFFTN:
  set(op.tensor, at::redispatch::fft_ifftn(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::IntArrayRef>>(op.args[1]), get<c10::optional<at::IntArrayRef>>(op.args[2]), get<c10::optional<std::string>>(op.args[3])));
  break;

case H_FFT_IFFTN_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::fft_ifftn_outf(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::IntArrayRef>>(op.args[1]), get<c10::optional<at::IntArrayRef>>(op.args[2]), get<c10::optional<std::string>>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_FFT_RFFTN:
  set(op.tensor, at::redispatch::fft_rfftn(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::IntArrayRef>>(op.args[1]), get<c10::optional<at::IntArrayRef>>(op.args[2]), get<c10::optional<std::string>>(op.args[3])));
  break;

case H_FFT_RFFTN_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::fft_rfftn_outf(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::IntArrayRef>>(op.args[1]), get<c10::optional<at::IntArrayRef>>(op.args[2]), get<c10::optional<std::string>>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_FFT_IRFFTN:
  set(op.tensor, at::redispatch::fft_irfftn(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::IntArrayRef>>(op.args[1]), get<c10::optional<at::IntArrayRef>>(op.args[2]), get<c10::optional<std::string>>(op.args[3])));
  break;

case H_FFT_IRFFTN_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::fft_irfftn_outf(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::IntArrayRef>>(op.args[1]), get<c10::optional<at::IntArrayRef>>(op.args[2]), get<c10::optional<std::string>>(op.args[3]), get<at::Tensor>(op.args[4]));
  end_update_in_place(op.tensor);
  break;

case H_FFT_FFTFREQ:
  set(op.tensor, at::redispatch::fft_fftfreq(ks, get<int64_t>(op.args[0]), get<double>(op.args[1]), get<c10::optional<at::ScalarType>>(op.args[2]), get<c10::optional<at::Layout>>(op.args[3]), get<c10::optional<at::Device>>(op.args[4]), get<c10::optional<bool>>(op.args[5])));
  break;

case H_FFT_FFTFREQ_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::fft_fftfreq_outf(ks, get<int64_t>(op.args[0]), get<double>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_FFT_RFFTFREQ:
  set(op.tensor, at::redispatch::fft_rfftfreq(ks, get<int64_t>(op.args[0]), get<double>(op.args[1]), get<c10::optional<at::ScalarType>>(op.args[2]), get<c10::optional<at::Layout>>(op.args[3]), get<c10::optional<at::Device>>(op.args[4]), get<c10::optional<bool>>(op.args[5])));
  break;

case H_FFT_RFFTFREQ_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::fft_rfftfreq_outf(ks, get<int64_t>(op.args[0]), get<double>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_FFT_FFTSHIFT:
  set(op.tensor, at::redispatch::fft_fftshift(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::IntArrayRef>>(op.args[1])));
  break;

case H_FFT_IFFTSHIFT:
  set(op.tensor, at::redispatch::fft_ifftshift(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::IntArrayRef>>(op.args[1])));
  break;

// skip std::tuple<at::Tensor,at::Tensor> linalg_cholesky_ex(const at::Tensor & self, bool check_errors)

// skip std::tuple<at::Tensor &,at::Tensor &> linalg_cholesky_ex_outf(const at::Tensor & self, bool check_errors, at::Tensor & L, at::Tensor & info)

case H_LINALG_CHOLESKY:
  set(op.tensor, at::redispatch::linalg_cholesky(ks, get<at::Tensor>(op.args[0])));
  break;

case H_LINALG_CHOLESKY_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::linalg_cholesky_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_LINALG_DET:
  set(op.tensor, at::redispatch::linalg_det(ks, get<at::Tensor>(op.args[0])));
  break;

case H_LINALG_DET_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::linalg_det_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_DET:
  set(op.tensor, at::redispatch::det(ks, get<at::Tensor>(op.args[0])));
  break;

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> linalg_lstsq(const at::Tensor & self, const at::Tensor & b, c10::optional<double> rcond, c10::optional<std::string> driver)

// skip std::tuple<at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &> linalg_lstsq_outf(const at::Tensor & self, const at::Tensor & b, c10::optional<double> rcond, c10::optional<std::string> driver, at::Tensor & solution, at::Tensor & residuals, at::Tensor & rank, at::Tensor & singular_values)

// skip std::tuple<at::Tensor,at::Tensor> linalg_slogdet(const at::Tensor & self)

// skip std::tuple<at::Tensor &,at::Tensor &> linalg_slogdet_outf(const at::Tensor & self, at::Tensor & sign, at::Tensor & logabsdet)

// skip std::tuple<at::Tensor,at::Tensor> linalg_eig(const at::Tensor & self)

// skip std::tuple<at::Tensor &,at::Tensor &> linalg_eig_outf(const at::Tensor & self, at::Tensor & eigenvalues, at::Tensor & eigenvectors)

case H_LINALG_EIGVALS:
  set(op.tensor, at::redispatch::linalg_eigvals(ks, get<at::Tensor>(op.args[0])));
  break;

case H_LINALG_EIGVALS_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::linalg_eigvals_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

// skip std::tuple<at::Tensor,at::Tensor> linalg_eigh(const at::Tensor & self, std::string UPLO)

// skip std::tuple<at::Tensor &,at::Tensor &> linalg_eigh_outf(const at::Tensor & self, std::string UPLO, at::Tensor & eigvals, at::Tensor & eigvecs)

case H_LINALG_EIGVALSH:
  set(op.tensor, at::redispatch::linalg_eigvalsh(ks, get<at::Tensor>(op.args[0]), get<std::string>(op.args[1])));
  break;

case H_LINALG_EIGVALSH_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::linalg_eigvalsh_outf(ks, get<at::Tensor>(op.args[0]), get<std::string>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_LINALG_HOUSEHOLDER_PRODUCT:
  set(op.tensor, at::redispatch::linalg_householder_product(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_LINALG_HOUSEHOLDER_PRODUCT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::linalg_householder_product_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H__LINALG_INV_OUT_HELPER_:
  init_update_in_place(op.tensor);
  at::redispatch::_linalg_inv_out_helper_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_LINALG_INV:
  set(op.tensor, at::redispatch::linalg_inv(ks, get<at::Tensor>(op.args[0])));
  break;

case H_LINALG_INV_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::linalg_inv_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_INNER:
  set(op.tensor, at::redispatch::inner(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_INNER_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::inner_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_OUTER:
  set(op.tensor, at::redispatch::outer(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_OUTER_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::outer_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_GER:
  set(op.tensor, at::redispatch::ger(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_GER_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::ger_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_LINALG_NORM:
  set(op.tensor, at::redispatch::linalg_norm(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::Scalar>>(op.args[1]), get<c10::optional<at::IntArrayRef>>(op.args[2]), get<bool>(op.args[3]), get<c10::optional<at::ScalarType>>(op.args[4])));
  break;

case H_LINALG_NORM_ORD_STR:
  set(op.tensor, at::redispatch::linalg_norm(ks, get<at::Tensor>(op.args[0]), get<std::string>(op.args[1]), get<c10::optional<at::IntArrayRef>>(op.args[2]), get<bool>(op.args[3]), get<c10::optional<at::ScalarType>>(op.args[4])));
  break;

case H_LINALG_NORM_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::linalg_norm_outf(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::Scalar>>(op.args[1]), get<c10::optional<at::IntArrayRef>>(op.args[2]), get<bool>(op.args[3]), get<c10::optional<at::ScalarType>>(op.args[4]), get<at::Tensor>(op.args[5]));
  end_update_in_place(op.tensor);
  break;

case H_LINALG_NORM_ORD_STR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::linalg_norm_outf(ks, get<at::Tensor>(op.args[0]), get<std::string>(op.args[1]), get<c10::optional<at::IntArrayRef>>(op.args[2]), get<bool>(op.args[3]), get<c10::optional<at::ScalarType>>(op.args[4]), get<at::Tensor>(op.args[5]));
  end_update_in_place(op.tensor);
  break;

case H_LINALG_VECTOR_NORM:
  set(op.tensor, at::redispatch::linalg_vector_norm(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<c10::optional<at::IntArrayRef>>(op.args[2]), get<bool>(op.args[3]), get<c10::optional<at::ScalarType>>(op.args[4])));
  break;

case H_LINALG_VECTOR_NORM_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::linalg_vector_norm_outf(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<c10::optional<at::IntArrayRef>>(op.args[2]), get<bool>(op.args[3]), get<c10::optional<at::ScalarType>>(op.args[4]), get<at::Tensor>(op.args[5]));
  end_update_in_place(op.tensor);
  break;

case H_LINALG_MATRIX_NORM:
  set(op.tensor, at::redispatch::linalg_matrix_norm(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<bool>(op.args[3]), get<c10::optional<at::ScalarType>>(op.args[4])));
  break;

case H_LINALG_MATRIX_NORM_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::linalg_matrix_norm_outf(ks, get<at::Tensor>(op.args[0]), get<at::Scalar>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<bool>(op.args[3]), get<c10::optional<at::ScalarType>>(op.args[4]), get<at::Tensor>(op.args[5]));
  end_update_in_place(op.tensor);
  break;

case H_LINALG_MATRIX_NORM_STR_ORD:
  set(op.tensor, at::redispatch::linalg_matrix_norm(ks, get<at::Tensor>(op.args[0]), get<std::string>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<bool>(op.args[3]), get<c10::optional<at::ScalarType>>(op.args[4])));
  break;

case H_LINALG_MATRIX_NORM_STR_ORD_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::linalg_matrix_norm_outf(ks, get<at::Tensor>(op.args[0]), get<std::string>(op.args[1]), get<at::IntArrayRef>(op.args[2]), get<bool>(op.args[3]), get<c10::optional<at::ScalarType>>(op.args[4]), get<at::Tensor>(op.args[5]));
  end_update_in_place(op.tensor);
  break;

// skip std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> linalg_svd_outf(const at::Tensor & self, bool full_matrices, at::Tensor & U, at::Tensor & S, at::Tensor & Vh)

// skip std::tuple<at::Tensor,at::Tensor,at::Tensor> linalg_svd(const at::Tensor & self, bool full_matrices)

case H_LINALG_SVDVALS:
  set(op.tensor, at::redispatch::linalg_svdvals(ks, get<at::Tensor>(op.args[0])));
  break;

case H_LINALG_SVDVALS_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::linalg_svdvals_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H_LINALG_COND:
  set(op.tensor, at::redispatch::linalg_cond(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::Scalar>>(op.args[1])));
  break;

case H_LINALG_COND_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::linalg_cond_outf(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::Scalar>>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_LINALG_COND_P_STR:
  set(op.tensor, at::redispatch::linalg_cond(ks, get<at::Tensor>(op.args[0]), get<std::string>(op.args[1])));
  break;

case H_LINALG_COND_P_STR_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::linalg_cond_outf(ks, get<at::Tensor>(op.args[0]), get<std::string>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_LINALG_PINV:
  set(op.tensor, at::redispatch::linalg_pinv(ks, get<at::Tensor>(op.args[0]), get<double>(op.args[1]), get<bool>(op.args[2])));
  break;

case H_LINALG_PINV_RCOND_TENSOR:
  set(op.tensor, at::redispatch::linalg_pinv(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<bool>(op.args[2])));
  break;

case H_LINALG_PINV_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::linalg_pinv_outf(ks, get<at::Tensor>(op.args[0]), get<double>(op.args[1]), get<bool>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_LINALG_PINV_OUT_RCOND_TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::linalg_pinv_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<bool>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H__LINALG_SOLVE_OUT_HELPER_:
  init_update_in_place(op.tensor);
  at::redispatch::_linalg_solve_out_helper_(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_LINALG_SOLVE:
  set(op.tensor, at::redispatch::linalg_solve(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1])));
  break;

case H_LINALG_SOLVE_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::linalg_solve_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_LINALG_TENSORINV:
  set(op.tensor, at::redispatch::linalg_tensorinv(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1])));
  break;

case H_LINALG_TENSORINV_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::linalg_tensorinv_outf(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_LINALG_TENSORSOLVE:
  set(op.tensor, at::redispatch::linalg_tensorsolve(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<at::IntArrayRef>>(op.args[2])));
  break;

case H_LINALG_TENSORSOLVE_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::linalg_tensorsolve_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<c10::optional<at::IntArrayRef>>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

// skip std::tuple<at::Tensor,at::Tensor> linalg_qr(const at::Tensor & self, std::string mode)

// skip std::tuple<at::Tensor &,at::Tensor &> linalg_qr_outf(const at::Tensor & self, std::string mode, at::Tensor & Q, at::Tensor & R)

// skip std::tuple<at::Tensor,at::Tensor> _linalg_qr_helper(const at::Tensor & self, std::string mode)

case H_LINALG_MATRIX_POWER:
  set(op.tensor, at::redispatch::linalg_matrix_power(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1])));
  break;

case H_LINALG_MATRIX_POWER_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::linalg_matrix_power_outf(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<at::Tensor>(op.args[2]));
  end_update_in_place(op.tensor);
  break;

case H_LINALG_MATRIX_RANK:
  set(op.tensor, at::redispatch::linalg_matrix_rank(ks, get<at::Tensor>(op.args[0]), get<c10::optional<double>>(op.args[1]), get<bool>(op.args[2])));
  break;

case H_LINALG_MATRIX_RANK_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::linalg_matrix_rank_outf(ks, get<at::Tensor>(op.args[0]), get<c10::optional<double>>(op.args[1]), get<bool>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_LINALG_MATRIX_RANK_TOL_TENSOR:
  set(op.tensor, at::redispatch::linalg_matrix_rank(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<bool>(op.args[2])));
  break;

case H_LINALG_MATRIX_RANK_OUT_TOL_TENSOR:
  init_update_in_place(op.tensor);
  at::redispatch::linalg_matrix_rank_outf(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<bool>(op.args[2]), get<at::Tensor>(op.args[3]));
  end_update_in_place(op.tensor);
  break;

case H_LINALG_MULTI_DOT:
  set(op.tensor, at::redispatch::linalg_multi_dot(ks, get<at::TensorList>(op.args[0])));
  break;

case H_LINALG_MULTI_DOT_OUT:
  init_update_in_place(op.tensor);
  at::redispatch::linalg_multi_dot_outf(ks, get<at::TensorList>(op.args[0]), get<at::Tensor>(op.args[1]));
  end_update_in_place(op.tensor);
  break;

case H__TEST_SERIALIZATION_SUBCMUL:
  set(op.tensor, at::redispatch::_test_serialization_subcmul(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Scalar>(op.args[2])));
  break;

case H__TEST_OPTIONAL_INTLIST:
  set(op.tensor, at::redispatch::_test_optional_intlist(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::IntArrayRef>>(op.args[1])));
  break;

case H__TEST_OPTIONAL_FILLED_INTLIST:
  set(op.tensor, at::redispatch::_test_optional_filled_intlist(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::IntArrayRef>>(op.args[1])));
  break;

case H__TEST_OPTIONAL_FLOATLIST:
  set(op.tensor, at::redispatch::_test_optional_floatlist(ks, get<at::Tensor>(op.args[0]), get<c10::optional<at::ArrayRef<double>>>(op.args[1])));
  break;

case H__TEST_STRING_DEFAULT:
  set(op.tensor, at::redispatch::_test_string_default(ks, get<at::Tensor>(op.args[0]), get<std::string>(op.args[1]), get<std::string>(op.args[2])));
  break;

case H__TEST_AMBIGUOUS_DEFAULTS_A:
  set(op.tensor, at::redispatch::_test_ambiguous_defaults(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<int64_t>(op.args[2])));
  break;

case H__TEST_AMBIGUOUS_DEFAULTS_B:
  set(op.tensor, at::redispatch::_test_ambiguous_defaults(ks, get<at::Tensor>(op.args[0]), get<int64_t>(op.args[1]), get<std::string>(op.args[2])));
  break;

case H_SEGMENT_REDUCE:
  set(op.tensor, at::redispatch::segment_reduce(ks, get<at::Tensor>(op.args[0]), get<std::string>(op.args[1]), get<c10::optional<at::Tensor>>(op.args[2]), get<c10::optional<at::Tensor>>(op.args[3]), get<int64_t>(op.args[4]), get<bool>(op.args[5]), get<c10::optional<at::Scalar>>(op.args[6])));
  break;

case H_SEGMENT_REDUCE_BACKWARD:
  set(op.tensor, at::redispatch::segment_reduce_backward(ks, get<at::Tensor>(op.args[0]), get<at::Tensor>(op.args[1]), get<at::Tensor>(op.args[2]), get<c10::optional<at::Tensor>>(op.args[3])));
  break;

