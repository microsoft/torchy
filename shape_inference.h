// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

using IntArrayRef = at::IntArrayRef;
using namespace std;

namespace {

std::vector<int64_t> shape_std_promote(IntArrayRef a, IntArrayRef b) {
  if (a == b)
    return a.vec();

  if (a.empty()) return b.vec();
  if (b.empty()) return a.vec();

  bool is_a_ge = a.size() >= b.size();
  auto &ge = is_a_ge ? a : b;
  auto &lt = is_a_ge ? b : a;

  auto promoted = ge.vec();
  unsigned j = lt.size()-1;
  for (int i = ge.size()-1; i >= 0; --i) {
    if ((lt[j] > ge[i] && ge[i] != 0) || lt[j] == 0)
      promoted[i] = lt[j];

    if (j-- == 0)
      break;
  }
  return promoted;
}

std::vector<int64_t> shape_matmul(IntArrayRef a, IntArrayRef b) {
  auto res = a.vec();
  res.back() = b.back();
  return res;
}

// https://pytorch.org/docs/stable/generated/torch.matmul.html
std::vector<int64_t> shape_mul(IntArrayRef a, IntArrayRef b) {
  auto size_a = a.size();
  auto size_b = b.size();
  if (size_a == 1 && size_b == 1)
    return {};

  if (size_a <= 2 && size_b == 2)
    return shape_matmul(a, b);

  if (size_a == 2 && size_b == 1)
    return {a[0]};

  if (size_a == 1 && size_b >= 2) {
    auto res = b.vec();
    auto last = res.back();
    res.pop_back();
    res.back() = last;
    return res;
  }

  if (size_b == 1) {
    auto res = a.vec();
    res.pop_back();
    return res;
  }

  auto res = shape_std_promote(a, b);
  res[res.size()-2] = a[a.size()-2];
  res.back() = b.back();
  return res;
}

std::vector<int64_t> shape_mult(IntArrayRef a, IntArrayRef b) {
  if (b.size() < 2)
    return shape_mul(a, b);

  auto newb = b.vec();
  swap(newb.back(), newb[newb.size()-2]);
  return shape_mul(a, newb);
}

std::vector<int64_t> shape_mul_last(IntArrayRef a, IntArrayRef b) {
  if (a.empty()) return b.vec();
  if (b.empty()) return a.vec();

  bool is_a_ge = a.size() >= b.size();
  auto &ge = is_a_ge ? a : b;
  auto &lt = is_a_ge ? b : a;

  auto promoted = ge.vec();
  unsigned j = lt.size()-1;
  for (int i = ge.size()-1; i >= 0; --i) {
    promoted[i] *= lt[j];

    if (j-- == 0)
      break;
  }
  return promoted;
}

std::vector<int64_t> shape_join(IntArrayRef a, IntArrayRef b) {
  auto res = a.vec();
  res.insert(res.end(), b.begin(), b.end());
  return res;
}

std::vector<int64_t> shape_pad1(IntArrayRef s) {
  auto res = s.vec();
  res.push_back(1);
  return res;
}

std::vector<int64_t>
shape_transpose(IntArrayRef s, int64_t dim1, int64_t dim2) {
  auto res = s.vec();
  if (dim1 < 0)
    dim1 += res.size();
  if (dim2 < 0)
    dim2 += res.size();
  if ((size_t)dim1 < res.size() && (size_t)dim2 < res.size())
    swap(res[dim1], res[dim2]);
  return res;
}

std::vector<int64_t> shape_transpose2d(IntArrayRef s) {
  auto res = s.vec();
  if (res.size() >= 2)
    swap(res[0], res[1]);
  return res;
}

std::vector<int64_t> shape_reshape(IntArrayRef s, IntArrayRef to) {
  auto res = to.vec();

  // A single -1 value should be filled with the remaining implied num of elems
  int64_t other = 1;
  int64_t *minus1 = nullptr;

  for (auto &i : res) {
    if (i == -1) {
      minus1 = &i;
    } else {
      other *= i;
    }
  }
  if (minus1) {
    auto nelems = accumulate(s.begin(), s.end(), 1, multiplies<int64_t>());
    *minus1 = nelems / other;
  }

  return res;
}

std::vector<int64_t> shape_select(IntArrayRef s, int64_t dim) {
  auto res = s.vec();
  if (dim < 0)
    dim += res.size();
  if ((size_t)dim < res.size())
    res.erase(res.begin() + dim);
  return res;
}

std::vector<int64_t> shape_unsqueeze(IntArrayRef s, int64_t dim) {
  auto res = s.vec();
  if (dim < 0)
    dim += res.size() + 1;
  if ((size_t)dim <= res.size())
    res.insert(res.begin() + dim, 1);
  return res;
}

std::vector<int64_t> shape_flatten(IntArrayRef s, int64_t start, int64_t end) {
  if (s.empty())
    return { 1 };

  if (end < 0)
    end += s.size();

  auto res = s.vec();
  int64_t n = 1;
  for (int64_t i = start; i <= end; ++i) {
    n *= res[i];
  }
  res[start] = n;
  res.erase(res.begin() + start + 1, res.begin() + end + 1);
  return res;
}

std::vector<int64_t> shape_arange_vec(const at::Scalar &start,
                                      const at::Scalar &end,
                                      const at::Scalar &step) {
  int64_t res = 0;
  if (start.isIntegral(true)) {
    assert(end.isIntegral(true) && step.isIntegral(true));
    auto s = step.to<int64_t>();
    res = (end.to<int64_t>() - start.to<int64_t>() + s - 1) / s;
  } else if (start.isFloatingPoint()) {
    res = ceil((end.to<double>() - start.to<double>()) / step.to<double>());
  } else {
    assert(false);
  }
  return { res };
}

std::vector<int64_t> shape_embedding(IntArrayRef w, IntArrayRef idxs) {
  auto res = idxs.vec();
  res.push_back(w.back());
  return res;
}

std::vector<int64_t> shape_slice(IntArrayRef s, int64_t dim,
                                 c10::optional<int64_t> start_opt,
                                 c10::optional<int64_t> end_opt,
                                 int64_t step) {
  auto res = s.vec();
  if (dim < 0)
    dim += s.size();
  if ((size_t)dim >= res.size())
    return res;

  auto limit = s[dim];
  int64_t start = start_opt.value_or(0);
  int64_t end   = min(end_opt.value_or(limit), limit);
  if (start < 0)
    start += limit;
  res[dim] = end >= start ? ((end - start + step - 1) / step) : 0;
  return res;
}

std::vector<int64_t> shape_stack(IntArrayRef s, unsigned n, int64_t dim) {
  auto res = s.vec();
  if (dim < 0)
    dim += s.size();
  res.insert(res.begin() + dim, n);
  return res;
}

std::vector<int64_t> shape_cat(const vector<IntArrayRef> &shapes, int64_t dim) {
  if (shapes.empty())
    return {};

  auto res = shapes[0].vec();
  if (dim < 0)
    dim += res.size();

  res[dim] = 0;
  for (auto sh : shapes) {
    res[dim] += sh[dim];
  }
  return res;
}

std::vector<int64_t>
shape_argmax(IntArrayRef s, c10::optional<int64_t> opt_dim, bool keepdim) {
  if (!opt_dim) {
    if (!keepdim)
      return {};
    return std::vector<int64_t>(s.size(), 1);
  }

  auto res = s.vec();
  int64_t dim = *opt_dim;
  if (dim < 0)
    dim += s.size();

  if ((size_t)dim < res.size()) {
    if (keepdim)
      res[dim] = 1;
    else
      res.erase(res.begin() + dim);
  }
  return res;
}

std::vector<int64_t> shape_conv2d(IntArrayRef input, IntArrayRef kernel,
                                  IntArrayRef stride, IntArrayRef padding,
                                  IntArrayRef dilation, int64_t out_channels) {
  int64_t h_out = ((input[2] + 2*padding[0] - dilation[0] * (kernel[0]-1) - 1) /
                   stride[0]) + 1;
  int64_t w_out = ((input[3] + 2*padding[1] - dilation[1] * (kernel[1]-1) - 1) /
                   stride[1]) + 1;
  return { input[0], out_channels, h_out, w_out};
}

std::vector<int64_t> shape_pool2d(IntArrayRef in, IntArrayRef shape) {
  std::vector<int64_t> res(in.begin(), in.end()-2);
  res.insert(res.end(), shape.begin(), shape.end());
  return res;
}

std::vector<int64_t>
shape_reduce(IntArrayRef s, IntArrayRef dims0, bool keepdim) {
  if (dims0.empty())
    return std::vector<int64_t>(keepdim ? s.size() : 0, 1);

  auto dims = dims0.vec();
  for (auto &dim : dims) {
    if (dim < 0)
      dim += s.size();
  }
  sort(dims.begin(), dims.end());

  auto res = s.vec();
  unsigned i = 0;
  for (auto dim : dims) {
    if ((size_t)dim < s.size()) {
      if (keepdim)
        res[dim] = 1;
      else
        res.erase(res.begin() + dim - i++);
    }
  }
  return res;
}

std::vector<int64_t> shape_permute(IntArrayRef s, IntArrayRef dims) {
  std::vector<int64_t> res;
  for (auto dim : dims) {
    if (dim < 0)
      dim += s.size();
    if ((size_t)dim < s.size())
      res.emplace_back(s[dim]);
  }
  return res;
}

std::vector<int64_t> shape_unfold(IntArrayRef s, int64_t dim, int64_t size,
                                  int64_t step) {
  auto res = s.vec();
  res.emplace_back(size);
  if (dim < 0)
    dim += s.size();
  if ((size_t)dim < s.size() && step != 0)
    res[dim] = (s[dim] - size) / step + 1;
  return res;
}

std::vector<int64_t> shape_narrow(IntArrayRef s, int64_t dim, int64_t start,
                                  int64_t length) {

  auto res = s.vec();
  if (dim < 0)
    dim += s.size();
  if ((size_t)dim < s.size()) {
    res[dim] = length;
  }
  return res;
}

}
