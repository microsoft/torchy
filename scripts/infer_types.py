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
  if 'TensorList' in type:
    t = f't{len(tensors)}'
    tensors.append(('TensorList &', t))
    return t
  if type == 'const c10::List<c10::optional<at::Tensor>> &':
    t = f't{len(tensors)}'
    tensors.append(('List<optional<Tensor>> &', t))
    return t
  if 'Tensor' in type:
    t = f't{len(tensors)}'
    tensors.append(('Tensor &', t))
    return t
  if type == 'const at::Scalar &' or type == 'const c10::optional<at::Scalar> &':
    return 'Scalar(0)'
  if type == 'c10::string_view':
    return '"foo"'
  if type == 'int64_t':
    return '1'
  if type == 'double':
    return '1.0'
  if type == 'bool':
    return 'false'
  if type == 'at::IntArrayRef':
    return 'IntArrayRef{0}'
  if 'optional<' in type:
    return 'nullopt'
  if type == 'at::Dimname':
    return 'Dimname::wildcard()'
  if type == 'at::DimnameList':
    return '{Dimname::wildcard()}'
  return '{}'


@with_native_function
def gen(fn):
  sig_group = CppSignatureGroup.from_native_function(fn, method=False, fallback_binding=fn.manual_cpp_binding)
  sig = sig_group.faithful_signature if sig_group.faithful_signature else sig_group.signature
  dispatcher_sig = DispatcherSignature.from_schema(fn.func)

  args = translate(sig.arguments(), dispatcher_sig.arguments())
  tensors = []
  args = [mk_arg(arg, tensors) for arg in args]

  if not tensors:
    return f'// skip {fn.func.name}'

  key = 'DispatchKeySet(DispatchKey::CPU)'
  types = ', '.join(ty for ty,name in tensors)
  types_names = ', '.join(f'{ty} {name}' for ty,name in tensors)
  return f'C::call("{fn.func.name}", function<Tensor({types})>{{[]({types_names}) {{' +\
         f' return at::redispatch::{sig.name()}({key}, {", ".join(args)}); }}}});'


fd = open('scripts/call_pytorch_fns.h', 'w')
for fn in native_functions.native_functions:
  if len(fn.func.returns) != 1 or str(fn.func.returns[0].type) != 'Tensor':
    continue

  print(gen(fn), file=fd)
