// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

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
    if (lt[j] > ge[i])
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
  assert((unsigned)dim1 < res.size() && (unsigned)dim2 < res.size());
  swap(res[dim1], res[dim2]);
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
