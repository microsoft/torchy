// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

#include "tensor.h"
#include <torch/csrc/autograd/variable.h>
#include <torch/library.h>

using namespace at;
using namespace std;

namespace {

using AG = torch::autograd::AutogradMeta;

struct TorchyAutograd final : public AG {
  void set_requires_grad(bool requires_grad, TensorImpl *self_impl) override {
    ensure_materialized(self_impl STATS_ARG(FlushReason::AUTOGRAD));
    AG::set_requires_grad(requires_grad, self_impl);
  }

  // No need to override this one as all the other methods already trigger
  // materialization.
  //bool requires_grad() const override;

  Tensor& mutable_grad() override {
    ensure_materialized(nullptr STATS_ARG(FlushReason::AUTOGRAD));
    return AG::mutable_grad();
  }

  const Tensor& grad() const override {
    ensure_materialized(nullptr STATS_ARG(FlushReason::AUTOGRAD));
    return AG::grad();
  }

  const Tensor& fw_grad(uint64_t level, const TensorBase &self) const override {
    ensure_materialized(nullptr STATS_ARG(FlushReason::AUTOGRAD));
    return AG::fw_grad(level, self);
  }

  void set_fw_grad(const TensorBase &new_grad, const TensorBase &self,
                   uint64_t level, bool is_inplace_op) override {
    ensure_materialized(nullptr STATS_ARG(FlushReason::AUTOGRAD));
    AG::set_fw_grad(new_grad, self, level, is_inplace_op);
  }
};


Tensor singleton_undefined_tensor;

struct TorchyFactory : public c10::impl::AutogradMetaFactory {
  unique_ptr<AutogradMetaInterface> make() const override {
    return make_unique<TorchyAutograd>();
  }

  const Tensor& undefined_tensor() const override {
    return singleton_undefined_tensor;
  }
};


TorchyFactory factory;
c10::impl::AutogradMetaFactoryRegisterer reg(&factory);

}
