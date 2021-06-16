at::Tensor(*const redispatch_ptrs_0[])(DispatchKeySet, const at::Tensor &, bool) = {
  at::redispatch::_cast_Byte,
  at::redispatch::_cast_Char,
  at::redispatch::_cast_Double,
  at::redispatch::_cast_Float,
  at::redispatch::_cast_Int,
  at::redispatch::_cast_Long,
  at::redispatch::_cast_Short,
  at::redispatch::_cast_Half,
  at::redispatch::matrix_rank,
  at::redispatch::std,
  at::redispatch::var,
  at::redispatch::nuclear_norm,
  at::redispatch::cholesky,
  at::redispatch::cholesky_inverse,
};

at::Tensor(*const redispatch_ptrs_1[])(DispatchKeySet, const at::Tensor &) = {
  at::redispatch::__dispatch_data,
  at::redispatch::_shape_as_tensor,
  at::redispatch::abs,
  at::redispatch::absolute,
  at::redispatch::angle,
  at::redispatch::view_as_real,
  at::redispatch::_view_as_real_physical,
  at::redispatch::view_as_complex,
  at::redispatch::real,
  at::redispatch::imag,
  at::redispatch::_conj,
  at::redispatch::__dispatch_conj,
  at::redispatch::_conj_physical,
  at::redispatch::conj_physical,
  at::redispatch::resolve_conj,
  at::redispatch::arccos,
  at::redispatch::arccosh,
  at::redispatch::arcsinh,
  at::redispatch::arctanh,
  at::redispatch::asin,
  at::redispatch::arcsin,
  at::redispatch::arctan,
  at::redispatch::atleast_1d,
  at::redispatch::atleast_2d,
  at::redispatch::atleast_3d,
  at::redispatch::logical_not,
  at::redispatch::ceil,
  at::redispatch::floor,
  at::redispatch::inverse,
  at::redispatch::_inverse_helper,
  at::redispatch::isnan,
  at::redispatch::isreal,
  at::redispatch::fbgemm_pack_gemm_matrix_fp16,
  at::redispatch::fbgemm_pack_quantized_matrix,
  at::redispatch::log1p,
  at::redispatch::logdet,
  at::redispatch::matrix_exp,
  at::redispatch::median,
  at::redispatch::nanmedian,
  at::redispatch::miopen_convolution_backward_bias,
  at::redispatch::numpy_T,
  at::redispatch::pin_memory,
  at::redispatch::rad2deg,
  at::redispatch::deg2rad,
  at::redispatch::ravel,
  at::redispatch::neg,
  at::redispatch::negative,
  at::redispatch::relu,
  at::redispatch::relu6,
  at::redispatch::gelu,
  at::redispatch::selu,
  at::redispatch::silu,
  at::redispatch::mish,
  at::redispatch::sigmoid,
  at::redispatch::detach,
  at::redispatch::squeeze,
  at::redispatch::sqrt,
  at::redispatch::square,
  at::redispatch::t,
  at::redispatch::tanh,
  at::redispatch::fliplr,
  at::redispatch::flipud,
  at::redispatch::trunc,
  at::redispatch::fix,
  at::redispatch::_sparse_sum,
  at::redispatch::frobenius_norm,
  at::redispatch::positive,
  at::redispatch::coalesce,
  at::redispatch::_coalesce,
  at::redispatch::_indices,
  at::redispatch::_values,
  at::redispatch::indices,
  at::redispatch::values,
  at::redispatch::crow_indices,
  at::redispatch::col_indices,
  at::redispatch::to_sparse,
  at::redispatch::dequantize,
  at::redispatch::q_per_channel_scales,
  at::redispatch::q_per_channel_zero_points,
  at::redispatch::int_repr,
  at::redispatch::_saturate_weight_to_fp16,
  at::redispatch::trace,
  at::redispatch::nonzero,
  at::redispatch::sign,
  at::redispatch::signbit,
  at::redispatch::min,
  at::redispatch::max,
  at::redispatch::msort,
  at::redispatch::all,
  at::redispatch::any,
  at::redispatch::alias,
  at::redispatch::hardsigmoid,
  at::redispatch::hardswish,
  at::redispatch::log_sigmoid,
  at::redispatch::isfinite,
  at::redispatch::isinf,
  at::redispatch::isposinf,
  at::redispatch::isneginf,
  at::redispatch::special_expm1,
  at::redispatch::special_exp2,
  at::redispatch::special_psi,
  at::redispatch::special_digamma,
  at::redispatch::special_gammaln,
  at::redispatch::special_erf,
  at::redispatch::special_erfc,
  at::redispatch::special_erfinv,
  at::redispatch::special_ndtr,
  at::redispatch::special_i0,
  at::redispatch::special_expit,
  at::redispatch::linalg_cholesky,
  at::redispatch::linalg_det,
  at::redispatch::det,
  at::redispatch::linalg_eigvals,
  at::redispatch::linalg_inv,
  at::redispatch::linalg_svdvals,
};

at::Tensor(*const redispatch_ptrs_2[])(DispatchKeySet, const at::Tensor &, int64_t) = {
  at::redispatch::_fw_primal,
  at::redispatch::_dim_arange,
  at::redispatch::diagflat,
  at::redispatch::_logcumsumexp,
  at::redispatch::logcumsumexp,
  at::redispatch::matrix_power,
  at::redispatch::mvlgamma,
  at::redispatch::pixel_shuffle,
  at::redispatch::pixel_unshuffle,
  at::redispatch::channel_shuffle,
  at::redispatch::squeeze,
  at::redispatch::one_hot,
  at::redispatch::unsqueeze,
  at::redispatch::to_sparse,
  at::redispatch::diag,
  at::redispatch::triu,
  at::redispatch::tril,
  at::redispatch::_cumsum,
  at::redispatch::_cumprod,
  at::redispatch::glu,
  at::redispatch::linalg_tensorinv,
  at::redispatch::linalg_matrix_power,
};

at::Tensor(*const redispatch_ptrs_3[])(DispatchKeySet, const at::Tensor &, const at::Tensor &, int64_t) = {
  at::redispatch::_make_dual,
  at::redispatch::trapz,
  at::redispatch::_weight_norm,
  at::redispatch::mse_loss,
  at::redispatch::l1_loss,
  at::redispatch::multilabel_margin_loss,
  at::redispatch::soft_margin_loss,
  at::redispatch::glu_backward,
};

at::Tensor(*const redispatch_ptrs_4[])(DispatchKeySet, const at::Tensor &, const at::Tensor &) = {
  at::redispatch::align_as,
  at::redispatch::_reshape_from_tensor,
  at::redispatch::logical_xor,
  at::redispatch::logical_and,
  at::redispatch::logical_or,
  at::redispatch::bmm,
  at::redispatch::clamp_max,
  at::redispatch::clamp_min,
  at::redispatch::complex,
  at::redispatch::polar,
  at::redispatch::cudnn_grid_sampler,
  at::redispatch::div,
  at::redispatch::divide,
  at::redispatch::true_divide,
  at::redispatch::dot,
  at::redispatch::vdot,
  at::redispatch::expand_as,
  at::redispatch::floor_divide,
  at::redispatch::kron,
  at::redispatch::ldexp,
  at::redispatch::logaddexp,
  at::redispatch::logaddexp2,
  at::redispatch::xlogy,
  at::redispatch::matmul,
  at::redispatch::matrix_exp_backward,
  at::redispatch::_compute_linear_combination,
  at::redispatch::mm,
  at::redispatch::_sparse_mm,
  at::redispatch::_sparse_sparse_matmul,
  at::redispatch::_sparse_mask_helper,
  at::redispatch::mul,
  at::redispatch::multiply,
  at::redispatch::mv,
  at::redispatch::_euclidean_dist,
  at::redispatch::reshape_as,
  at::redispatch::prelu,
  at::redispatch::infinitely_differentiable_gelu_backward,
  at::redispatch::silu_backward,
  at::redispatch::mish_backward,
  at::redispatch::smm,
  at::redispatch::type_as,
  at::redispatch::view_as,
  at::redispatch::_standard_gamma_grad,
  at::redispatch::sparse_mask,
  at::redispatch::to_dense_backward,
  at::redispatch::hspmm,
  at::redispatch::to_mkldnn_backward,
  at::redispatch::fake_quantize_per_tensor_affine_cachemask_backward,
  at::redispatch::fake_quantize_per_channel_affine_cachemask_backward,
  at::redispatch::bitwise_and,
  at::redispatch::__and__,
  at::redispatch::bitwise_or,
  at::redispatch::__or__,
  at::redispatch::bitwise_xor,
  at::redispatch::__xor__,
  at::redispatch::__lshift__,
  at::redispatch::__rshift__,
  at::redispatch::ne,
  at::redispatch::not_equal,
  at::redispatch::eq,
  at::redispatch::ge,
  at::redispatch::greater_equal,
  at::redispatch::le,
  at::redispatch::less_equal,
  at::redispatch::gt,
  at::redispatch::greater,
  at::redispatch::lt,
  at::redispatch::less,
  at::redispatch::take,
  at::redispatch::masked_select,
  at::redispatch::orgqr,
  at::redispatch::fmod,
  at::redispatch::max,
  at::redispatch::min,
  at::redispatch::float_power,
  at::redispatch::hardswish_backward,
  at::redispatch::mkldnn_adaptive_avg_pool2d_backward,
  at::redispatch::_adaptive_avg_pool2d_backward,
  at::redispatch::_adaptive_avg_pool3d_backward,
  at::redispatch::sigmoid_backward,
  at::redispatch::tanh_backward,
  at::redispatch::linalg_householder_product,
  at::redispatch::inner,
  at::redispatch::outer,
  at::redispatch::ger,
  at::redispatch::linalg_solve,
};

at::Tensor &(*const redispatch_ptrs_5[])(DispatchKeySet, at::Tensor &, int64_t) = {
  at::redispatch::_sobol_engine_initialize_state_,
  at::redispatch::mvlgamma_,
  at::redispatch::squeeze_,
  at::redispatch::unsqueeze_,
  at::redispatch::tril_,
  at::redispatch::triu_,
  at::redispatch::polygamma_,
};

at::Tensor(*const redispatch_ptrs_6[])(DispatchKeySet, const at::Tensor &, double, bool) = {
  at::redispatch::dropout,
  at::redispatch::feature_dropout,
  at::redispatch::alpha_dropout,
  at::redispatch::feature_alpha_dropout,
  at::redispatch::matrix_rank,
  at::redispatch::linalg_pinv,
};

at::Tensor &(*const redispatch_ptrs_7[])(DispatchKeySet, at::Tensor &, double, bool) = {
  at::redispatch::dropout_,
  at::redispatch::feature_dropout_,
  at::redispatch::alpha_dropout_,
  at::redispatch::feature_alpha_dropout_,
};

at::Tensor &(*const redispatch_ptrs_8[])(DispatchKeySet, at::Tensor &) = {
  at::redispatch::abs_,
  at::redispatch::absolute_,
  at::redispatch::conj_physical_,
  at::redispatch::arccos_,
  at::redispatch::arccosh_,
  at::redispatch::arcsinh_,
  at::redispatch::arctanh_,
  at::redispatch::asin_,
  at::redispatch::arcsin_,
  at::redispatch::arctan_,
  at::redispatch::logical_not_,
  at::redispatch::ceil_,
  at::redispatch::floor_,
  at::redispatch::log1p_,
  at::redispatch::rad2deg_,
  at::redispatch::deg2rad_,
  at::redispatch::neg_,
  at::redispatch::negative_,
  at::redispatch::relu_,
  at::redispatch::relu6_,
  at::redispatch::selu_,
  at::redispatch::silu_,
  at::redispatch::mish_,
  at::redispatch::sigmoid_,
  at::redispatch::detach_,
  at::redispatch::squeeze_,
  at::redispatch::square_,
  at::redispatch::t_,
  at::redispatch::tanh_,
  at::redispatch::trunc_,
  at::redispatch::fix_,
  at::redispatch::zero_,
  at::redispatch::set_,
  at::redispatch::sign_,
  at::redispatch::hardswish_,
};

at::Tensor &(*const redispatch_ptrs_9[])(DispatchKeySet, const at::Tensor &, at::Tensor &) = {
  at::redispatch::abs_outf,
  at::redispatch::absolute_outf,
  at::redispatch::angle_outf,
  at::redispatch::sgn_outf,
  at::redispatch::conj_physical_outf,
  at::redispatch::acos_outf,
  at::redispatch::arccos_outf,
  at::redispatch::acosh_outf,
  at::redispatch::arccosh_outf,
  at::redispatch::asinh_outf,
  at::redispatch::arcsinh_outf,
  at::redispatch::atanh_outf,
  at::redispatch::arctanh_outf,
  at::redispatch::asin_outf,
  at::redispatch::arcsin_outf,
  at::redispatch::atan_outf,
  at::redispatch::arctan_outf,
  at::redispatch::bitwise_not_outf,
  at::redispatch::logical_not_outf,
  at::redispatch::ceil_outf,
  at::redispatch::cos_outf,
  at::redispatch::cosh_outf,
  at::redispatch::erf_outf,
  at::redispatch::erfc_outf,
  at::redispatch::exp_outf,
  at::redispatch::exp2_outf,
  at::redispatch::expm1_outf,
  at::redispatch::floor_outf,
  at::redispatch::frac_outf,
  at::redispatch::inverse_outf,
  at::redispatch::log_outf,
  at::redispatch::log10_outf,
  at::redispatch::log1p_outf,
  at::redispatch::log2_outf,
  at::redispatch::rad2deg_outf,
  at::redispatch::deg2rad_outf,
  at::redispatch::reciprocal_outf,
  at::redispatch::neg_outf,
  at::redispatch::negative_outf,
  at::redispatch::round_outf,
  at::redispatch::gelu_outf,
  at::redispatch::rsqrt_outf,
  at::redispatch::silu_outf,
  at::redispatch::mish_outf,
  at::redispatch::sigmoid_outf,
  at::redispatch::sin_outf,
  at::redispatch::sinc_outf,
  at::redispatch::sinh_outf,
  at::redispatch::sqrt_outf,
  at::redispatch::square_outf,
  at::redispatch::tan_outf,
  at::redispatch::tanh_outf,
  at::redispatch::trunc_outf,
  at::redispatch::fix_outf,
  at::redispatch::nonzero_outf,
  at::redispatch::lgamma_outf,
  at::redispatch::digamma_outf,
  at::redispatch::erfinv_outf,
  at::redispatch::i0_outf,
  at::redispatch::sign_outf,
  at::redispatch::signbit_outf,
  at::redispatch::msort_outf,
  at::redispatch::hardsigmoid_outf,
  at::redispatch::hardswish_outf,
  at::redispatch::log_sigmoid_outf,
  at::redispatch::isposinf_outf,
  at::redispatch::isneginf_outf,
  at::redispatch::special_entr_outf,
  at::redispatch::special_expm1_outf,
  at::redispatch::special_exp2_outf,
  at::redispatch::special_psi_outf,
  at::redispatch::special_digamma_outf,
  at::redispatch::special_gammaln_outf,
  at::redispatch::special_erf_outf,
  at::redispatch::special_erfc_outf,
  at::redispatch::special_erfinv_outf,
  at::redispatch::special_ndtr_outf,
  at::redispatch::special_i0_outf,
  at::redispatch::special_i0e_outf,
  at::redispatch::special_i1_outf,
  at::redispatch::special_i1e_outf,
  at::redispatch::special_expit_outf,
  at::redispatch::linalg_cholesky_outf,
  at::redispatch::linalg_det_outf,
  at::redispatch::linalg_eigvals_outf,
  at::redispatch::linalg_inv_outf,
  at::redispatch::linalg_svdvals_outf,
};

at::Tensor(*const redispatch_ptrs_10[])(DispatchKeySet, const at::Tensor &, at::IntArrayRef) = {
  at::redispatch::adaptive_avg_pool1d,
  at::redispatch::broadcast_to,
  at::redispatch::count_nonzero,
  at::redispatch::permute,
  at::redispatch::repeat,
  at::redispatch::reshape,
  at::redispatch::_mkldnn_reshape,
  at::redispatch::sum_to_size,
  at::redispatch::tile,
  at::redispatch::flip,
  at::redispatch::_unsafe_view,
  at::redispatch::_sparse_sum,
  at::redispatch::view,
  at::redispatch::trace_backward,
  at::redispatch::adaptive_avg_pool2d,
  at::redispatch::mkldnn_adaptive_avg_pool2d,
  at::redispatch::_adaptive_avg_pool2d,
  at::redispatch::adaptive_avg_pool3d,
  at::redispatch::_adaptive_avg_pool3d,
  at::redispatch::reflection_pad1d,
  at::redispatch::reflection_pad2d,
};

at::Tensor(*const redispatch_ptrs_11[])(DispatchKeySet, const at::Tensor &, const at::Tensor &, const at::Scalar &) = {
  at::redispatch::add,
  at::redispatch::_add_relu,
  at::redispatch::threshold_backward,
  at::redispatch::where,
  at::redispatch::sub,
  at::redispatch::subtract,
  at::redispatch::rsub,
  at::redispatch::masked_fill,
  at::redispatch::dist,
  at::redispatch::lerp,
  at::redispatch::_test_serialization_subcmul,
};

at::Tensor &(*const redispatch_ptrs_12[])(DispatchKeySet, at::Tensor &, const at::Tensor &, const at::Scalar &) = {
  at::redispatch::add_,
  at::redispatch::_add_relu_,
  at::redispatch::sub_,
  at::redispatch::subtract_,
  at::redispatch::masked_fill_,
  at::redispatch::lerp_,
};

at::Tensor &(*const redispatch_ptrs_13[])(DispatchKeySet, const at::Tensor &, const at::Tensor &, const at::Scalar &, at::Tensor &) = {
  at::redispatch::add_outf,
  at::redispatch::_add_relu_outf,
  at::redispatch::hardshrink_backward_outf,
  at::redispatch::threshold_backward_outf,
  at::redispatch::sub_outf,
  at::redispatch::subtract_outf,
  at::redispatch::lerp_outf,
  at::redispatch::softshrink_backward_outf,
};

at::Tensor(*const redispatch_ptrs_14[])(DispatchKeySet, const at::Tensor &, const at::Scalar &, const at::Scalar &) = {
  at::redispatch::add,
  at::redispatch::threshold,
  at::redispatch::where,
  at::redispatch::sub,
  at::redispatch::subtract,
  at::redispatch::rsub,
  at::redispatch::hardtanh,
};

at::Tensor &(*const redispatch_ptrs_15[])(DispatchKeySet, at::Tensor &, const at::Scalar &, const at::Scalar &) = {
  at::redispatch::add_,
  at::redispatch::sub_,
  at::redispatch::subtract_,
  at::redispatch::hardtanh_,
};

at::Tensor &(*const redispatch_ptrs_16[])(DispatchKeySet, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Scalar &, const at::Scalar &, at::Tensor &) = {
  at::redispatch::addmv_outf,
  at::redispatch::addr_outf,
  at::redispatch::baddbmm_outf,
  at::redispatch::sspaddmm_outf,
  at::redispatch::addmm_outf,
  at::redispatch::addbmm_outf,
};

at::Tensor(*const redispatch_ptrs_17[])(DispatchKeySet, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Scalar &, const at::Scalar &) = {
  at::redispatch::addr,
  at::redispatch::baddbmm,
  at::redispatch::sspaddmm,
  at::redispatch::_sparse_addmm,
  at::redispatch::addmm,
  at::redispatch::addbmm,
};

at::Tensor &(*const redispatch_ptrs_18[])(DispatchKeySet, at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Scalar &, const at::Scalar &) = {
  at::redispatch::addr_,
  at::redispatch::baddbmm_,
  at::redispatch::_baddbmm_mkl_,
  at::redispatch::addmm_,
  at::redispatch::addbmm_,
};

at::Tensor(*const redispatch_ptrs_19[])(DispatchKeySet, const at::Tensor &, at::IntArrayRef, bool) = {
  at::redispatch::affine_grid_generator,
  at::redispatch::affine_grid_generator_backward,
  at::redispatch::expand,
  at::redispatch::logsumexp,
  at::redispatch::amax,
  at::redispatch::amin,
  at::redispatch::frobenius_norm,
  at::redispatch::nuclear_norm,
};

at::Tensor(*const redispatch_ptrs_20[])(DispatchKeySet, const at::Tensor &, int64_t, bool) = {
  at::redispatch::all,
  at::redispatch::any,
  at::redispatch::_log_softmax,
  at::redispatch::_softmax,
  at::redispatch::_sparse_softmax,
  at::redispatch::_sparse_log_softmax,
  at::redispatch::combinations,
  at::redispatch::argsort,
};

at::Tensor(*const redispatch_ptrs_21[])(DispatchKeySet, const at::Tensor &, at::Dimname, bool) = {
  at::redispatch::all,
  at::redispatch::any,
  at::redispatch::argsort,
};

at::Tensor(*const redispatch_ptrs_22[])(DispatchKeySet, const at::Tensor &, c10::optional<int64_t>, bool) = {
  at::redispatch::argmax,
  at::redispatch::argmin,
  at::redispatch::vander,
};

at::Tensor(*const redispatch_ptrs_23[])(DispatchKeySet, int64_t, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<at::Device>, c10::optional<bool>) = {
  at::redispatch::bartlett_window,
  at::redispatch::blackman_window,
  at::redispatch::eye,
  at::redispatch::hann_window,
  at::redispatch::hamming_window,
  at::redispatch::kaiser_window,
  at::redispatch::randperm,
};

at::Tensor(*const redispatch_ptrs_24[])(DispatchKeySet, int64_t, bool, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<at::Device>, c10::optional<bool>) = {
  at::redispatch::bartlett_window,
  at::redispatch::blackman_window,
  at::redispatch::hann_window,
  at::redispatch::hamming_window,
  at::redispatch::kaiser_window,
};

at::Tensor(*const redispatch_ptrs_25[])(DispatchKeySet, const at::Tensor &, c10::optional<at::Generator>) = {
  at::redispatch::bernoulli,
  at::redispatch::_standard_gamma,
  at::redispatch::_sample_dirichlet,
  at::redispatch::poisson,
};

at::Tensor &(*const redispatch_ptrs_26[])(DispatchKeySet, at::Tensor &, double, c10::optional<at::Generator>) = {
  at::redispatch::bernoulli_,
  at::redispatch::exponential_,
  at::redispatch::geometric_,
};

at::Tensor &(*const redispatch_ptrs_27[])(DispatchKeySet, const at::Tensor &, const at::Tensor &, at::Tensor &) = {
  at::redispatch::copysign_outf,
  at::redispatch::logical_xor_outf,
  at::redispatch::logical_and_outf,
  at::redispatch::logical_or_outf,
  at::redispatch::bmm_outf,
  at::redispatch::clamp_max_outf,
  at::redispatch::clamp_min_outf,
  at::redispatch::complex_outf,
  at::redispatch::polar_outf,
  at::redispatch::div_outf,
  at::redispatch::divide_outf,
  at::redispatch::true_divide_outf,
  at::redispatch::dot_outf,
  at::redispatch::vdot_outf,
  at::redispatch::floor_divide_outf,
  at::redispatch::gcd_outf,
  at::redispatch::lcm_outf,
  at::redispatch::kron_outf,
  at::redispatch::ldexp_outf,
  at::redispatch::logaddexp_outf,
  at::redispatch::logaddexp2_outf,
  at::redispatch::xlogy_outf,
  at::redispatch::matmul_outf,
  at::redispatch::_compute_linear_combination_outf,
  at::redispatch::mm_outf,
  at::redispatch::mul_outf,
  at::redispatch::multiply_outf,
  at::redispatch::mv_outf,
  at::redispatch::gelu_backward_outf,
  at::redispatch::heaviside_outf,
  at::redispatch::hspmm_outf,
  at::redispatch::bitwise_and_outf,
  at::redispatch::bitwise_or_outf,
  at::redispatch::bitwise_xor_outf,
  at::redispatch::ne_outf,
  at::redispatch::not_equal_outf,
  at::redispatch::eq_outf,
  at::redispatch::ge_outf,
  at::redispatch::greater_equal_outf,
  at::redispatch::le_outf,
  at::redispatch::less_equal_outf,
  at::redispatch::gt_outf,
  at::redispatch::greater_outf,
  at::redispatch::lt_outf,
  at::redispatch::less_outf,
  at::redispatch::take_outf,
  at::redispatch::masked_select_outf,
  at::redispatch::orgqr_outf,
  at::redispatch::atan2_outf,
  at::redispatch::fmod_outf,
  at::redispatch::hypot_outf,
  at::redispatch::igamma_outf,
  at::redispatch::igammac_outf,
  at::redispatch::nextafter_outf,
  at::redispatch::remainder_outf,
  at::redispatch::fmin_outf,
  at::redispatch::fmax_outf,
  at::redispatch::maximum_outf,
  at::redispatch::max_outf,
  at::redispatch::minimum_outf,
  at::redispatch::min_outf,
  at::redispatch::pow_outf,
  at::redispatch::float_power_outf,
  at::redispatch::hardsigmoid_backward_outf,
  at::redispatch::adaptive_avg_pool3d_backward_outf,
  at::redispatch::sigmoid_backward_outf,
  at::redispatch::tanh_backward_outf,
  at::redispatch::special_xlog1py_outf,
  at::redispatch::linalg_householder_product_outf,
  at::redispatch::inner_outf,
  at::redispatch::outer_outf,
  at::redispatch::ger_outf,
  at::redispatch::linalg_solve_outf,
};

at::Tensor(*const redispatch_ptrs_28[])(DispatchKeySet, const at::Tensor &, const at::Scalar &) = {
  at::redispatch::copysign,
  at::redispatch::clamp_max,
  at::redispatch::clamp_min,
  at::redispatch::div,
  at::redispatch::divide,
  at::redispatch::true_divide,
  at::redispatch::floor_divide,
  at::redispatch::xlogy,
  at::redispatch::mul,
  at::redispatch::multiply,
  at::redispatch::celu,
  at::redispatch::native_norm,
  at::redispatch::norm,
  at::redispatch::bitwise_and,
  at::redispatch::__and__,
  at::redispatch::bitwise_or,
  at::redispatch::__or__,
  at::redispatch::bitwise_xor,
  at::redispatch::__xor__,
  at::redispatch::__lshift__,
  at::redispatch::__rshift__,
  at::redispatch::ne,
  at::redispatch::not_equal,
  at::redispatch::eq,
  at::redispatch::ge,
  at::redispatch::greater_equal,
  at::redispatch::le,
  at::redispatch::less_equal,
  at::redispatch::gt,
  at::redispatch::greater,
  at::redispatch::lt,
  at::redispatch::less,
  at::redispatch::fmod,
  at::redispatch::remainder,
  at::redispatch::pow,
  at::redispatch::float_power,
  at::redispatch::leaky_relu,
  at::redispatch::special_xlog1py,
};

at::Tensor &(*const redispatch_ptrs_29[])(DispatchKeySet, at::Tensor &, const at::Scalar &) = {
  at::redispatch::copysign_,
  at::redispatch::clamp_max_,
  at::redispatch::clamp_min_,
  at::redispatch::div_,
  at::redispatch::divide_,
  at::redispatch::true_divide_,
  at::redispatch::fill_,
  at::redispatch::floor_divide_,
  at::redispatch::xlogy_,
  at::redispatch::mul_,
  at::redispatch::multiply_,
  at::redispatch::celu_,
  at::redispatch::eq_,
  at::redispatch::bitwise_and_,
  at::redispatch::__iand__,
  at::redispatch::bitwise_or_,
  at::redispatch::__ior__,
  at::redispatch::bitwise_xor_,
  at::redispatch::__ixor__,
  at::redispatch::__ilshift__,
  at::redispatch::__irshift__,
  at::redispatch::fmod_,
  at::redispatch::ne_,
  at::redispatch::not_equal_,
  at::redispatch::ge_,
  at::redispatch::greater_equal_,
  at::redispatch::le_,
  at::redispatch::less_equal_,
  at::redispatch::gt_,
  at::redispatch::greater_,
  at::redispatch::lt_,
  at::redispatch::less_,
  at::redispatch::remainder_,
  at::redispatch::float_power_,
  at::redispatch::leaky_relu_,
};

at::Tensor &(*const redispatch_ptrs_30[])(DispatchKeySet, const at::Tensor &, const at::Scalar &, at::Tensor &) = {
  at::redispatch::copysign_outf,
  at::redispatch::clamp_max_outf,
  at::redispatch::clamp_min_outf,
  at::redispatch::xlogy_outf,
  at::redispatch::hardshrink_outf,
  at::redispatch::bitwise_and_outf,
  at::redispatch::bitwise_or_outf,
  at::redispatch::bitwise_xor_outf,
  at::redispatch::ne_outf,
  at::redispatch::not_equal_outf,
  at::redispatch::eq_outf,
  at::redispatch::ge_outf,
  at::redispatch::greater_equal_outf,
  at::redispatch::le_outf,
  at::redispatch::less_equal_outf,
  at::redispatch::gt_outf,
  at::redispatch::greater_outf,
  at::redispatch::lt_outf,
  at::redispatch::less_outf,
  at::redispatch::fmod_outf,
  at::redispatch::remainder_outf,
  at::redispatch::pow_outf,
  at::redispatch::float_power_outf,
  at::redispatch::leaky_relu_outf,
  at::redispatch::softshrink_outf,
  at::redispatch::special_xlog1py_outf,
};

at::Tensor &(*const redispatch_ptrs_31[])(DispatchKeySet, at::Tensor &, const at::Tensor &) = {
  at::redispatch::logical_xor_,
  at::redispatch::logical_and_,
  at::redispatch::logical_or_,
  at::redispatch::clamp_max_,
  at::redispatch::clamp_min_,
  at::redispatch::div_,
  at::redispatch::divide_,
  at::redispatch::true_divide_,
  at::redispatch::fill_,
  at::redispatch::floor_divide_,
  at::redispatch::ldexp_,
  at::redispatch::xlogy_,
  at::redispatch::mul_,
  at::redispatch::multiply_,
  at::redispatch::set_,
  at::redispatch::eq_,
  at::redispatch::bitwise_and_,
  at::redispatch::__iand__,
  at::redispatch::bitwise_or_,
  at::redispatch::__ior__,
  at::redispatch::bitwise_xor_,
  at::redispatch::__ixor__,
  at::redispatch::__ilshift__,
  at::redispatch::__irshift__,
  at::redispatch::fmod_,
  at::redispatch::ne_,
  at::redispatch::not_equal_,
  at::redispatch::ge_,
  at::redispatch::greater_equal_,
  at::redispatch::le_,
  at::redispatch::less_equal_,
  at::redispatch::gt_,
  at::redispatch::greater_,
  at::redispatch::lt_,
  at::redispatch::less_,
  at::redispatch::hypot_,
  at::redispatch::nextafter_,
  at::redispatch::float_power_,
};

at::Tensor(*const redispatch_ptrs_32[])(DispatchKeySet, const at::Tensor &, const at::Tensor &, bool) = {
  at::redispatch::_bmm,
  at::redispatch::cholesky_solve,
  at::redispatch::_cholesky_solve_helper,
  at::redispatch::linalg_pinv,
  at::redispatch::linalg_matrix_rank,
};

at::Tensor &(*const redispatch_ptrs_33[])(DispatchKeySet, const at::Tensor &, const at::Tensor &, bool, at::Tensor &) = {
  at::redispatch::_bmm_outf,
  at::redispatch::cholesky_solve_outf,
  at::redispatch::linalg_pinv_outf,
  at::redispatch::linalg_matrix_rank_outf,
};

at::Tensor(*const redispatch_ptrs_34[])(DispatchKeySet, at::TensorList, int64_t) = {
  at::redispatch::cat,
  at::redispatch::stack,
  at::redispatch::_stack,
  at::redispatch::_cat,
};

at::Tensor &(*const redispatch_ptrs_35[])(DispatchKeySet, at::TensorList, int64_t, at::Tensor &) = {
  at::redispatch::cat_outf,
  at::redispatch::stack_outf,
  at::redispatch::_stack_outf,
  at::redispatch::_cat_outf,
};

at::Tensor(*const redispatch_ptrs_36[])(DispatchKeySet, at::TensorList) = {
  at::redispatch::block_diag,
  at::redispatch::chain_matmul,
  at::redispatch::row_stack,
  at::redispatch::hstack,
  at::redispatch::vstack,
  at::redispatch::dstack,
  at::redispatch::cartesian_prod,
  at::redispatch::column_stack,
  at::redispatch::linalg_multi_dot,
  at::redispatch::flatten_dense_tensors,
};

at::Tensor &(*const redispatch_ptrs_37[])(DispatchKeySet, at::TensorList, at::Tensor &) = {
  at::redispatch::chain_matmul_outf,
  at::redispatch::row_stack_outf,
  at::redispatch::hstack_outf,
  at::redispatch::vstack_outf,
  at::redispatch::dstack_outf,
  at::redispatch::column_stack_outf,
  at::redispatch::linalg_multi_dot_outf,
};

at::Tensor(*const redispatch_ptrs_38[])(DispatchKeySet, const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, at::IntArrayRef, c10::string_view, at::IntArrayRef, int64_t) = {
  at::redispatch::_convolution_mode,
  at::redispatch::conv1d,
  at::redispatch::conv2d,
  at::redispatch::conv3d,
};

at::Tensor(*const redispatch_ptrs_39[])(DispatchKeySet, const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t) = {
  at::redispatch::conv1d,
  at::redispatch::conv2d,
  at::redispatch::conv3d,
  at::redispatch::cudnn_convolution_relu,
  at::redispatch::mkldnn_convolution,
};

at::Tensor(*const redispatch_ptrs_40[])(DispatchKeySet, const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t) = {
  at::redispatch::conv_tbc,
  at::redispatch::cummaxmin_backward,
  at::redispatch::_make_per_channel_quantized_tensor,
  at::redispatch::mse_loss_backward,
  at::redispatch::l1_loss_backward,
  at::redispatch::soft_margin_loss_backward,
};

at::Tensor(*const redispatch_ptrs_41[])(DispatchKeySet, const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t, at::IntArrayRef) = {
  at::redispatch::conv_transpose1d,
  at::redispatch::conv_transpose2d,
  at::redispatch::conv_transpose3d,
};

at::Tensor(*const redispatch_ptrs_42[])(DispatchKeySet, const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t, bool, bool) = {
  at::redispatch::cudnn_convolution,
  at::redispatch::miopen_convolution,
  at::redispatch::miopen_depthwise_convolution,
};

at::Tensor(*const redispatch_ptrs_43[])(DispatchKeySet, at::IntArrayRef, const at::Tensor &, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t, bool, bool, bool) = {
  at::redispatch::cudnn_convolution_backward_input,
  at::redispatch::cudnn_convolution_backward_weight,
  at::redispatch::cudnn_convolution_transpose_backward_weight,
};

at::Tensor(*const redispatch_ptrs_44[])(DispatchKeySet, const at::Tensor &, int64_t, c10::optional<at::ScalarType>) = {
  at::redispatch::cumprod,
  at::redispatch::cumsum,
  at::redispatch::log_softmax,
  at::redispatch::softmax,
  at::redispatch::_sparse_softmax,
  at::redispatch::_sparse_log_softmax,
};

at::Tensor(*const redispatch_ptrs_45[])(DispatchKeySet, const at::Tensor &, at::Dimname, c10::optional<at::ScalarType>) = {
  at::redispatch::cumprod,
  at::redispatch::cumsum,
  at::redispatch::log_softmax,
  at::redispatch::softmax,
  at::redispatch::_sparse_softmax,
  at::redispatch::_sparse_log_softmax,
};

at::Tensor(*const redispatch_ptrs_46[])(DispatchKeySet, const at::Tensor &, const at::Tensor &, int64_t, const at::Tensor &) = {
  at::redispatch::cumprod_backward,
  at::redispatch::_log_softmax_backward_data,
  at::redispatch::_softmax_backward_data,
  at::redispatch::_sparse_softmax_backward_data,
  at::redispatch::_sparse_log_softmax_backward_data,
};

at::Tensor(*const redispatch_ptrs_47[])(DispatchKeySet, const at::Tensor &, int64_t, int64_t, int64_t) = {
  at::redispatch::diag_embed,
  at::redispatch::diagonal,
  at::redispatch::narrow_copy,
  at::redispatch::narrow,
  at::redispatch::unfold,
  at::redispatch::_remove_batch_dim,
};

at::Tensor(*const redispatch_ptrs_48[])(DispatchKeySet, const at::Tensor &, const at::Tensor &, int64_t, int64_t, bool) = {
  at::redispatch::embedding_dense_backward,
  at::redispatch::embedding_sparse_backward,
  at::redispatch::grid_sampler,
  at::redispatch::grid_sampler_2d,
  at::redispatch::_grid_sampler_2d_cpu_fallback,
  at::redispatch::grid_sampler_3d,
};

at::Tensor(*const redispatch_ptrs_49[])(DispatchKeySet, const at::Tensor &, at::IntArrayRef, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<at::Device>, c10::optional<bool>) = {
  at::redispatch::new_empty,
  at::redispatch::new_zeros,
  at::redispatch::new_ones,
};

at::Tensor(*const redispatch_ptrs_50[])(DispatchKeySet, const at::Tensor &, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<at::Device>, c10::optional<bool>, c10::optional<at::MemoryFormat>) = {
  at::redispatch::empty_like,
  at::redispatch::ones_like,
  at::redispatch::rand_like,
  at::redispatch::randn_like,
  at::redispatch::zeros_like,
};

at::Tensor(*const redispatch_ptrs_51[])(DispatchKeySet, const at::Tensor &, int64_t, int64_t) = {
  at::redispatch::flatten,
  at::redispatch::fbgemm_pack_quantized_matrix,
  at::redispatch::movedim,
  at::redispatch::moveaxis,
  at::redispatch::select,
  at::redispatch::transpose,
  at::redispatch::_mkldnn_transpose,
  at::redispatch::norm_except_dim,
  at::redispatch::swapaxes,
  at::redispatch::swapdims,
  at::redispatch::_add_batch_dim,
  at::redispatch::_test_ambiguous_defaults,
};

at::Tensor &(*const redispatch_ptrs_52[])(DispatchKeySet, at::Tensor &, int64_t, const at::Tensor &, const at::Tensor &) = {
  at::redispatch::index_copy_,
  at::redispatch::index_add_,
  at::redispatch::index_fill_,
  at::redispatch::_index_copy_,
};

at::Tensor(*const redispatch_ptrs_53[])(DispatchKeySet, const at::Tensor &, int64_t, const at::Tensor &, const at::Tensor &) = {
  at::redispatch::index_copy,
  at::redispatch::index_add,
  at::redispatch::index_fill,
  at::redispatch::_gather_sparse_backward,
};

at::Tensor(*const redispatch_ptrs_54[])(DispatchKeySet, const at::Tensor &, at::Dimname, const at::Tensor &, const at::Tensor &) = {
  at::redispatch::index_copy,
  at::redispatch::index_fill,
  at::redispatch::scatter,
  at::redispatch::scatter_add,
};

at::Tensor(*const redispatch_ptrs_55[])(DispatchKeySet, const at::Tensor &, const at::Tensor &, const at::Tensor &) = {
  at::redispatch::fbgemm_linear_fp16_weight_fp32_activation,
  at::redispatch::fbgemm_linear_fp16_weight,
  at::redispatch::where,
  at::redispatch::_s_where,
  at::redispatch::_dirichlet_grad,
  at::redispatch::masked_fill,
  at::redispatch::masked_scatter,
  at::redispatch::masked_select_backward,
  at::redispatch::lu_solve,
  at::redispatch::lerp,
  at::redispatch::log_sigmoid_backward,
};

at::Tensor(*const redispatch_ptrs_56[])(DispatchKeySet, const at::Scalar &, const at::Tensor &) = {
  at::redispatch::xlogy,
  at::redispatch::remainder,
  at::redispatch::float_power,
  at::redispatch::special_xlog1py,
};

at::Tensor &(*const redispatch_ptrs_57[])(DispatchKeySet, const at::Scalar &, const at::Tensor &, at::Tensor &) = {
  at::redispatch::xlogy_outf,
  at::redispatch::pow_outf,
  at::redispatch::float_power_outf,
  at::redispatch::special_xlog1py_outf,
};

at::Tensor &(*const redispatch_ptrs_58[])(DispatchKeySet, const at::Tensor &, int64_t, at::Tensor &) = {
  at::redispatch::_logcumsumexp_outf,
  at::redispatch::logcumsumexp_outf,
  at::redispatch::matrix_power_outf,
  at::redispatch::diag_outf,
  at::redispatch::triu_outf,
  at::redispatch::tril_outf,
  at::redispatch::_cumsum_outf,
  at::redispatch::_cumprod_outf,
  at::redispatch::glu_outf,
  at::redispatch::linalg_tensorinv_outf,
  at::redispatch::linalg_matrix_power_outf,
};

at::Tensor &(*const redispatch_ptrs_59[])(DispatchKeySet, const at::Tensor &, at::IntArrayRef, bool, at::Tensor &) = {
  at::redispatch::logsumexp_outf,
  at::redispatch::amax_outf,
  at::redispatch::amin_outf,
  at::redispatch::frobenius_norm_outf,
  at::redispatch::nuclear_norm_outf,
};

at::Tensor(*const redispatch_ptrs_60[])(DispatchKeySet, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, bool) = {
  at::redispatch::max_pool1d,
  at::redispatch::max_pool2d,
  at::redispatch::mkldnn_max_pool2d,
  at::redispatch::mkldnn_max_pool3d,
  at::redispatch::quantized_max_pool1d,
  at::redispatch::quantized_max_pool2d,
  at::redispatch::max_pool3d,
};

at::Tensor(*const redispatch_ptrs_61[])(DispatchKeySet, const at::Tensor &, c10::optional<at::ScalarType>) = {
  at::redispatch::mean,
  at::redispatch::sum,
  at::redispatch::nansum,
  at::redispatch::prod,
  at::redispatch::to_dense,
  at::redispatch::to_mkldnn,
};

at::Tensor(*const redispatch_ptrs_62[])(DispatchKeySet, const at::Tensor &, at::IntArrayRef, bool, c10::optional<at::ScalarType>) = {
  at::redispatch::mean,
  at::redispatch::sum,
  at::redispatch::nansum,
};

at::Tensor &(*const redispatch_ptrs_63[])(DispatchKeySet, const at::Tensor &, at::IntArrayRef, bool, c10::optional<at::ScalarType>, at::Tensor &) = {
  at::redispatch::mean_outf,
  at::redispatch::sum_outf,
  at::redispatch::nansum_outf,
};

at::Tensor(*const redispatch_ptrs_64[])(DispatchKeySet, at::IntArrayRef, const at::Tensor &, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t, bool, bool) = {
  at::redispatch::miopen_convolution_backward_input,
  at::redispatch::miopen_convolution_backward_weight,
  at::redispatch::miopen_convolution_transpose_backward_weight,
  at::redispatch::miopen_depthwise_convolution_backward_input,
  at::redispatch::miopen_depthwise_convolution_backward_weight,
};

at::Tensor(*const redispatch_ptrs_65[])(DispatchKeySet, at::IntArrayRef, c10::optional<at::DimnameList>, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<at::Device>, c10::optional<bool>) = {
  at::redispatch::ones,
  at::redispatch::rand,
  at::redispatch::randn,
  at::redispatch::zeros,
};

at::Tensor(*const redispatch_ptrs_66[])(DispatchKeySet, at::IntArrayRef, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<at::Device>, c10::optional<bool>) = {
  at::redispatch::ones,
  at::redispatch::rand,
  at::redispatch::randn,
  at::redispatch::zeros,
  at::redispatch::sparse_coo_tensor,
};

at::Tensor &(*const redispatch_ptrs_67[])(DispatchKeySet, at::IntArrayRef, at::Tensor &) = {
  at::redispatch::ones_outf,
  at::redispatch::rand_outf,
  at::redispatch::randn_outf,
  at::redispatch::zeros_outf,
};

at::Tensor(*const redispatch_ptrs_68[])(DispatchKeySet, const at::Tensor &, double) = {
  at::redispatch::pdist,
  at::redispatch::_pdist_forward,
  at::redispatch::pinverse,
};

at::Tensor(*const redispatch_ptrs_69[])(DispatchKeySet, const at::Tensor &, const at::Tensor &, int64_t, double) = {
  at::redispatch::cosine_similarity,
  at::redispatch::smooth_l1_loss,
  at::redispatch::huber_loss,
};

at::Tensor(*const redispatch_ptrs_70[])(DispatchKeySet, const at::Tensor &, at::IntArrayRef, at::IntArrayRef) = {
  at::redispatch::movedim,
  at::redispatch::moveaxis,
  at::redispatch::roll,
};

at::Tensor &(*const redispatch_ptrs_71[])(DispatchKeySet, const at::Tensor &, const at::Scalar &, const at::Scalar &, at::Tensor &) = {
  at::redispatch::threshold_outf,
  at::redispatch::hardtanh_outf,
  at::redispatch::softplus_outf,
};

at::Tensor &(*const redispatch_ptrs_72[])(DispatchKeySet, at::Tensor &, int64_t, int64_t) = {
  at::redispatch::transpose_,
  at::redispatch::_mkldnn_transpose_,
  at::redispatch::swapaxes_,
  at::redispatch::swapdims_,
};

at::Tensor(*const redispatch_ptrs_73[])(DispatchKeySet, const at::Tensor &, const at::Tensor &, at::IntArrayRef) = {
  at::redispatch::_sparse_sum_backward,
  at::redispatch::max_unpool2d,
  at::redispatch::reflection_pad2d_backward,
  at::redispatch::replication_pad2d_backward,
  at::redispatch::replication_pad3d_backward,
};

at::Tensor &(*const redispatch_ptrs_74[])(DispatchKeySet, const at::Tensor &, bool, at::Tensor &) = {
  at::redispatch::nuclear_norm_outf,
  at::redispatch::cholesky_outf,
  at::redispatch::cholesky_inverse_outf,
};

at::Tensor(*const redispatch_ptrs_75[])(DispatchKeySet, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &) = {
  at::redispatch::gru_cell,
  at::redispatch::rnn_tanh_cell,
  at::redispatch::rnn_relu_cell,
};

at::Tensor(*const redispatch_ptrs_76[])(DispatchKeySet, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Scalar &, const at::Scalar &, const at::Scalar &, const at::Scalar &) = {
  at::redispatch::quantized_gru_cell,
  at::redispatch::quantized_rnn_relu_cell,
  at::redispatch::quantized_rnn_tanh_cell,
};

at::Tensor &(*const redispatch_ptrs_77[])(DispatchKeySet, at::Tensor &, const at::Tensor &, const at::Tensor &) = {
  at::redispatch::masked_fill_,
  at::redispatch::masked_scatter_,
  at::redispatch::lerp_,
};

at::Tensor &(*const redispatch_ptrs_78[])(DispatchKeySet, at::Tensor &, double, double, c10::optional<at::Generator>) = {
  at::redispatch::uniform_,
  at::redispatch::cauchy_,
  at::redispatch::log_normal_,
  at::redispatch::normal_,
};

at::Tensor(*const redispatch_ptrs_79[])(DispatchKeySet, const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, int64_t, int64_t) = {
  at::redispatch::cross_entropy_loss,
  at::redispatch::nll_loss_nd,
  at::redispatch::nll_loss,
  at::redispatch::nll_loss2d,
};

at::Tensor &(*const redispatch_ptrs_80[])(DispatchKeySet, const at::Tensor &, const at::Tensor &, const at::Tensor &, at::Tensor &) = {
  at::redispatch::lu_solve_outf,
  at::redispatch::lerp_outf,
  at::redispatch::log_sigmoid_backward_outf,
  at::redispatch::adaptive_max_pool2d_backward_outf,
  at::redispatch::adaptive_max_pool3d_backward_outf,
};

at::Tensor &(*const redispatch_ptrs_81[])(DispatchKeySet, const at::Tensor &, const at::Tensor &, int64_t, at::Tensor &) = {
  at::redispatch::mse_loss_outf,
  at::redispatch::l1_loss_outf,
  at::redispatch::multilabel_margin_loss_outf,
  at::redispatch::soft_margin_loss_outf,
  at::redispatch::glu_backward_outf,
};

at::Tensor &(*const redispatch_ptrs_82[])(DispatchKeySet, const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t, at::Tensor &) = {
  at::redispatch::mse_loss_backward_outf,
  at::redispatch::l1_loss_backward_outf,
  at::redispatch::soft_margin_loss_backward_outf,
};

at::Tensor &(*const redispatch_ptrs_83[])(DispatchKeySet, const at::Tensor &, at::IntArrayRef, at::Tensor &) = {
  at::redispatch::adaptive_avg_pool2d_outf,
  at::redispatch::adaptive_avg_pool3d_outf,
  at::redispatch::reflection_pad1d_outf,
  at::redispatch::reflection_pad2d_outf,
  at::redispatch::replication_pad1d_outf,
  at::redispatch::replication_pad2d_outf,
  at::redispatch::replication_pad3d_outf,
};

at::Tensor &(*const redispatch_ptrs_84[])(DispatchKeySet, const at::Tensor &, const at::Tensor &, at::IntArrayRef, at::Tensor &) = {
  at::redispatch::max_unpool2d_outf,
  at::redispatch::reflection_pad1d_backward_outf,
  at::redispatch::reflection_pad2d_backward_outf,
  at::redispatch::replication_pad1d_backward_outf,
  at::redispatch::replication_pad2d_backward_outf,
  at::redispatch::replication_pad3d_backward_outf,
};

at::Tensor(*const redispatch_ptrs_85[])(DispatchKeySet, const at::Tensor &, c10::optional<at::IntArrayRef>, bool, c10::optional<at::ArrayRef<double>>) = {
  at::redispatch::upsample_linear1d,
  at::redispatch::upsample_bilinear2d,
  at::redispatch::upsample_trilinear3d,
  at::redispatch::upsample_bicubic2d,
};

at::Tensor(*const redispatch_ptrs_86[])(DispatchKeySet, const at::Tensor &, c10::optional<at::IntArrayRef>, at::IntArrayRef, bool, c10::optional<at::ArrayRef<double>>) = {
  at::redispatch::upsample_linear1d_backward,
  at::redispatch::upsample_bilinear2d_backward,
  at::redispatch::upsample_trilinear3d_backward,
  at::redispatch::upsample_bicubic2d_backward,
};

at::Tensor(*const redispatch_ptrs_87[])(DispatchKeySet, const at::Tensor &, c10::optional<at::IntArrayRef>, c10::optional<at::ArrayRef<double>>) = {
  at::redispatch::upsample_nearest1d,
  at::redispatch::upsample_nearest2d,
  at::redispatch::upsample_nearest3d,
};

at::Tensor(*const redispatch_ptrs_88[])(DispatchKeySet, const at::Tensor &, c10::optional<at::IntArrayRef>, at::IntArrayRef, c10::optional<at::ArrayRef<double>>) = {
  at::redispatch::upsample_nearest1d_backward,
  at::redispatch::upsample_nearest2d_backward,
  at::redispatch::upsample_nearest3d_backward,
};

at::Tensor(*const redispatch_ptrs_89[])(DispatchKeySet, const at::Tensor &, const at::Tensor &, at::IntArrayRef, const c10::optional<at::Tensor> &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef) = {
  at::redispatch::thnn_conv_depthwise2d,
  at::redispatch::thnn_conv_depthwise2d_forward,
  at::redispatch::conv_depthwise3d,
  at::redispatch::slow_conv_dilated2d,
  at::redispatch::slow_conv_dilated3d,
};

at::Tensor(*const redispatch_ptrs_90[])(DispatchKeySet, const at::Tensor &, c10::optional<int64_t>, int64_t, c10::optional<c10::string_view>) = {
  at::redispatch::fft_fft,
  at::redispatch::fft_ifft,
  at::redispatch::fft_rfft,
  at::redispatch::fft_irfft,
  at::redispatch::fft_hfft,
  at::redispatch::fft_ihfft,
};

at::Tensor &(*const redispatch_ptrs_91[])(DispatchKeySet, const at::Tensor &, c10::optional<int64_t>, int64_t, c10::optional<c10::string_view>, at::Tensor &) = {
  at::redispatch::fft_fft_outf,
  at::redispatch::fft_ifft_outf,
  at::redispatch::fft_rfft_outf,
  at::redispatch::fft_irfft_outf,
  at::redispatch::fft_hfft_outf,
  at::redispatch::fft_ihfft_outf,
};

at::Tensor(*const redispatch_ptrs_92[])(DispatchKeySet, const at::Tensor &, c10::optional<at::IntArrayRef>, at::IntArrayRef, c10::optional<c10::string_view>) = {
  at::redispatch::fft_fft2,
  at::redispatch::fft_ifft2,
  at::redispatch::fft_rfft2,
  at::redispatch::fft_irfft2,
};

at::Tensor &(*const redispatch_ptrs_93[])(DispatchKeySet, const at::Tensor &, c10::optional<at::IntArrayRef>, at::IntArrayRef, c10::optional<c10::string_view>, at::Tensor &) = {
  at::redispatch::fft_fft2_outf,
  at::redispatch::fft_ifft2_outf,
  at::redispatch::fft_rfft2_outf,
  at::redispatch::fft_irfft2_outf,
};

at::Tensor(*const redispatch_ptrs_94[])(DispatchKeySet, const at::Tensor &, c10::optional<at::IntArrayRef>, c10::optional<at::IntArrayRef>, c10::optional<c10::string_view>) = {
  at::redispatch::fft_fftn,
  at::redispatch::fft_ifftn,
  at::redispatch::fft_rfftn,
  at::redispatch::fft_irfftn,
};

at::Tensor &(*const redispatch_ptrs_95[])(DispatchKeySet, const at::Tensor &, c10::optional<at::IntArrayRef>, c10::optional<at::IntArrayRef>, c10::optional<c10::string_view>, at::Tensor &) = {
  at::redispatch::fft_fftn_outf,
  at::redispatch::fft_ifftn_outf,
  at::redispatch::fft_rfftn_outf,
  at::redispatch::fft_irfftn_outf,
};

at::Tensor(*const redispatch_ptrs_96[])(DispatchKeySet, const at::Tensor &, c10::optional<at::IntArrayRef>) = {
  at::redispatch::fft_fftshift,
  at::redispatch::fft_ifftshift,
  at::redispatch::_test_optional_intlist,
  at::redispatch::_test_optional_filled_intlist,
};
