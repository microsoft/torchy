#include <ATen/Tensor.h>
#include <torch/library.h>
#include <iostream>

using namespace at;
using namespace std;

namespace {

Tensor add_Tensor(const Tensor &self, const Tensor &other, const Scalar &alpha) {
  cout << "Called add" << endl;
  return self;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("add.Tensor", add_Tensor);
}

}
