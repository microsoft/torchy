// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

#include <ATen/ExpandUtils.h>
#include <ATen/TensorUtils.h>

using IntArrayRef = at::IntArrayRef;

namespace {

using ShapeStridesVec = std::vector<std::pair<IntArrayRef, IntArrayRef>>;

// Adapted from TensorIteratorBase::compute_strides
std::vector<std::vector<int64_t>>
compute_strides_it(IntArrayRef shape_out, const ShapeStridesVec &ops) {
  std::vector<std::vector<int64_t>> ret;
  auto ndim = shape_out.size();

  for (auto op : ops) {
    auto &shape = op.first;
    auto &strides = op.second;
    ret.emplace_back(ndim, 0);
    auto &new_strides = ret.back();
    auto offset = ndim - shape.size();

    for (const auto i : c10::irange(shape.size())) {
      if (shape[i] != 1 || shape_out[offset + i] == 1)
        new_strides[offset + i] = strides[i];
    }
  }
  return ret;
}

// Adapted from TensorIteratorBase::reorder_dimensions()
std::vector<unsigned>
reorder_dimensions_it(std::vector<std::vector<int64_t>> &strides,
                      IntArrayRef shape_out, const ShapeStridesVec &ops) {
  auto ndim = shape_out.size();
  std::vector<unsigned> perm(ndim);
  if (ndim == 1) {
    perm[0] = 0;
    return perm;
  }

  std::iota(perm.rbegin(), perm.rend(), 0);

  auto should_swap = [&](size_t dim0, size_t dim1) {
    for (auto &op_strides : strides) {
      int64_t stride0 = op_strides[dim0];
      int64_t stride1 = op_strides[dim1];
      if (stride0 == 0 || stride1 == 0) {
        continue;
      } else if (stride0 < stride1) {
        return -1;
      } else  if (stride0 > stride1) {
        return 1;
      } else {
         auto t_dim0 = shape_out[dim0];
         auto t_dim1 = shape_out[dim1];
         if (t_dim0 > t_dim1) {
             return 1;
         }
      }
    }
    return 0;
  };

  for (const auto i : c10::irange(1, ndim)) {
    int dim1 = i;
    for (int dim0 = i - 1; dim0 >= 0; dim0--) {
      int comparison = should_swap(perm[dim0], perm[dim1]);
      if (comparison > 0) {
        std::swap(perm[dim0], perm[dim1]);
        dim1 = dim0;
      } else if (comparison < 0) {
        break;
      }
    }
  }
  return perm;
}

// Adapted from TensorIteratorBase::compatible_stride
std::vector<int64_t> compatible_stride(IntArrayRef shape) {
  std::vector<int64_t> stride;
  int64_t next_stride = 1;
  for (auto sz : shape) {
    stride.push_back(next_stride);
    next_stride *= sz;
  }
  return stride;
}

// Adapted from TensorIteratorBase::permute_dimensions
std::vector<int64_t> permute_dimensions(IntArrayRef shape,
                                        const std::vector<unsigned> &perm) {
  std::vector<int64_t> res(shape.size(), 0);
  for (const auto i : c10::irange(perm.size())) {
    res[i] = shape[perm[i]];
  }
  return res;
}

// Adapted from TensorIteratorBase::invert_perm
std::vector<int64_t> invert_perm(IntArrayRef input,
                                 const std::vector<unsigned> &perm) {
  std::vector<int64_t> res(input.size());
  for (const auto dim : c10::irange(perm.size())) {
    res[perm[dim]] = input[dim];
  }
  return res;
}

std::vector<int64_t> strides_contiguous(IntArrayRef shape) {
  std::vector<int64_t> ret(shape.size());
  int64_t stride = 1;
  for(size_t i = shape.size(); i > 0; --i) {
    ret[i-1] = stride;
    stride *= shape[i-1] == 0 ? 1 : shape[i-1];
  }
  return ret;
}

bool all_shapes_eq(const ShapeStridesVec &ops) {
  for (auto &op : ops) {
    if (op.first != ops[0].first)
      return false;
  }
  return true;
}

std::vector<int64_t>
strides_std_promote(IntArrayRef shape_out, const ShapeStridesVec &ops) {
  if (all_shapes_eq(ops)) {
    bool all_contiguous = true;
    for (auto op : ops) {
      all_contiguous &= at::geometry_is_contiguous(op.first, op.second);
    }
    if (all_contiguous)
      return strides_contiguous(shape_out);
  }

  auto op_strides = compute_strides_it(shape_out, ops);
  auto perm = reorder_dimensions_it(op_strides, shape_out, ops);

  auto ndim = shape_out.size();
  bool inverted = true;
  for (const auto j : c10::irange(ndim)) {
    if (perm[j] != ndim - j - 1) {
      inverted = false;
      break;
    }
  }
  if (inverted)
    return strides_contiguous(shape_out);

  auto shape_perm = permute_dimensions(shape_out, perm);
  auto strides = compatible_stride(shape_perm);
  return invert_perm(strides, perm);
}

std::vector<int64_t> strides_contiguous(IntArrayRef shape,
                                        IntArrayRef strides,
                                        bool preserve = false) {
  if (preserve && at::geometry_is_contiguous(shape, strides))
    return strides.vec();
  return strides_contiguous(shape);
}

std::vector<int64_t> strides_transpose2d(IntArrayRef shape) {
  auto ret = shape.vec();
  reverse(ret.begin(), ret.end());
  return ret;
}

c10::optional<std::vector<int64_t>> strides_view(IntArrayRef shape_oldt,
                                                 IntArrayRef strides_oldt,
                                                 IntArrayRef shape_newt) {
  return at::detail::computeStride(shape_oldt, strides_oldt, shape_newt);
}

// adapted from TensorImpl::compute_non_overlapping_and_dense()
bool is_non_overlapping_and_dense(IntArrayRef sizes, IntArrayRef strides) {
  auto dim = sizes.size();
  if (sizes.size() == 1) {
    return sizes[0] < 2 || sizes[0] == 1;
  }
  c10::SmallVector<int64_t, 5> perm;
  perm.resize(dim);
  for (size_t i = 0; i < dim; i++) {
    perm[i] = i;
  }
  // Sort by strides, leaving 0 and 1 sized dims at the end of the array
  std::sort(perm.begin(), perm.end(), [&](int64_t a, int64_t b) {
    if (sizes[a] < 2) {
      return false;
    } else if (sizes[b] < 2) {
      return true;
    }
    return strides[a] < strides[b];
  });
  auto require_stride = 1;
  for (size_t i = 0; i < dim; i++) {
    const auto size_perm_i = sizes[perm[i]];
    if (size_perm_i < 2) {
      return true;
    }
    if (strides[perm[i]] != require_stride) {
      return false;
    }
    require_stride *= size_perm_i;
  }
  return true;
}

c10::optional<std::vector<int64_t>>
strides_clone(IntArrayRef shape, IntArrayRef strides,
              c10::optional<at::MemoryFormat> format, bool full_preseve) {
  switch (format.value_or(at::MemoryFormat::Preserve)) {
  case at::MemoryFormat::Preserve:
    return full_preseve && is_non_overlapping_and_dense(shape, strides)
             ? strides.vec()
             : at::infer_dense_strides(shape, strides);
  case at::MemoryFormat::Contiguous:
    return strides_contiguous(shape);
  default: // TODO
    return {};
  }
}

c10::optional<std::vector<int64_t>>
strides_clone2(IntArrayRef shape, IntArrayRef strides,
               c10::optional<at::MemoryFormat> format0, bool copy) {
  if (!copy)
    return strides.vec();

  switch (format0.value_or(at::MemoryFormat::Preserve)) {
  case at::MemoryFormat::Preserve:
    return strides_contiguous(shape, strides, true);
  case at::MemoryFormat::Contiguous:
    return strides_contiguous(shape);
  default: // TODO
    return {};
  }
}

c10::optional<std::vector<int64_t>>
strides_clone_bool(IntArrayRef shape, IntArrayRef strides, bool copy) {
  if (copy)
    return strides_clone(shape, strides, {}, true);
  return strides.vec();
}

std::vector<int64_t>
strides_expand(IntArrayRef shape, IntArrayRef strides, IntArrayRef tgt) {
  auto src_sz = shape.size();
  auto tgt_sz = tgt.size();
  std::vector<int64_t> ret(tgt_sz, 0);

  if (tgt_sz < src_sz)
    return {};

  auto expanded = [&](size_t i) {
    return shape[src_sz-i] != tgt[tgt_sz-i] && tgt[tgt_sz-i] != -1;
  };

  for (auto i = src_sz; i > 0; --i) {
    ret[tgt_sz-i] = expanded(i) ? 0 : strides[src_sz-i];
  }

  if (!expanded(src_sz)) {
    for (auto i = src_sz+1; i <= tgt_sz; ++i) {
      if (tgt[tgt_sz-i] != 1)
        break;
      ret[tgt_sz-i] = shape[0] * strides[0];
    }
  }
  return ret;
}

std::vector<int64_t> strides_slice(IntArrayRef strides, int64_t dim,
                                   int64_t step) {
  auto ret = strides.vec();
  ret[dim] *= step;
  return ret;
}

std::vector<int64_t>
strides_flatten(IntArrayRef shape, IntArrayRef strides, IntArrayRef shape_out) {
  if (shape == shape_out)
    return strides.vec();

  if (shape.empty())
    return { 1 };

  if (auto op = at::detail::computeStride(shape, strides, shape_out))
    return *op;
  return strides_contiguous(shape_out);
}

std::vector<int64_t>
strides_unsqueeze(IntArrayRef shape, IntArrayRef strides, int64_t dim) {
  auto res = strides.vec();
  if (dim < 0)
    dim += res.size() + 1;
  auto val = (size_t)dim < res.size() ? shape[dim] * strides[dim] : 1;
  if ((size_t)dim <= res.size())
    res.insert(res.begin() + dim, val);
  return res;
}

}
