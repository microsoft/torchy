# Copyright (c) 2021-present The Torchy Authors.
# Distributed under the MIT license that can be found in the LICENSE file.

PYTORCH = '../pytorch'

import sys
sys.path.append(PYTORCH)
from tools.codegen.gen import *
from tools.codegen.api import types
import torch

yaml_path = PYTORCH + '/aten/src/ATen/native/native_functions.yaml'
native_functions = parse_native_yaml(yaml_path)


def mk_arg(arg, tensors):
  type = arg.type.cpp_type()
  if 'Tensor' in type:
    t = f't{len(tensors)}'
    tensors.append(t)
    return t
  return '{}'


@with_native_function
def gen(fn):
  sig_group = CppSignatureGroup.from_native_function(fn, method=False, fallback_binding=fn.manual_cpp_binding)
  sig = sig_group.faithful_signature if sig_group.faithful_signature else sig_group.signature
  dispatcher_sig = DispatcherSignature.from_schema(fn.func)

  args = translate(sig.arguments(), dispatcher_sig.arguments())
  tensors = []
  args = [mk_arg(arg, tensors) for arg in args]

  fn = f'auto result = at::redispatch::{sig.name()}({", ".join(args)})'

  print(fn)
  exit()


for fn in native_functions.native_functions:
  if len(fn.func.returns) != 1 or str(fn.func.returns[0].type) != 'Tensor':
    continue

  gen(fn)
