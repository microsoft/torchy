#include <Python.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <torch/csrc/Exceptions.h>

#define DISPATCHKEY DispatchKey::PrivateUse1

using namespace at;
using namespace std;

namespace {

void set_torchy_enabled(bool enable) {
  c10::impl::tls_set_dispatch_key_included(DISPATCHKEY, enable);
}

PYBIND11_MODULE(_TORCHY, m) {
  m.def("enable", set_torchy_enabled);
}

}
