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


def get(tensors, type):
  # try to reuse a tensor to avoid exponential explosion
  if len(tensors) >= 4:
    for ty, t in tensors:
      if ty == type:
        return t
  t = f't{len(tensors)}'
  tensors.append((type, t))
  return t

def mk_arg(arg, tensors):
  type = arg.type.remove_const_ref().cpp_type()
  dispatch_types = {
    'bool' : False,
    'int64_t' : False,
    'at::Scalar' : True,
    'at::ScalarType' : False,
    'at::IntArrayRef' : False,
    'c10::optional<int64_t>' : False,
    'c10::optional<at::MemoryFormat>' : False,
    'c10::optional<at::Scalar>' : True,
    'c10::optional<at::ScalarType>' : False,
  }
  if 'Tensor' in type:
    return get(tensors, type + ' &')
  if type in dispatch_types:
    if dispatch_types[type]:
      type += ' &'
    return get(tensors, type)
  if type == 'c10::string_view':
    return '"foo"'
  if type == 'double':
    return '1'
  if type == 'at::Dimname':
    return 'Dimname::wildcard()'
  if type == 'at::DimnameList':
    return '{Dimname::wildcard()}'
  if type == 'at::Device':
    return 'Device("cpu")'
  return '{}'


all_functions = []

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

  all_functions.append((str(fn.func.name), len(tensors)))
  ptr_cast = dispatcher_sig.type().replace(' (', '(*)(DispatchKeySet, ')

  key = 'DispatchKeySet(DispatchKey::CPU)'
  types = ', '.join(ty for ty,name in tensors)
  types_names = ', '.join(f'{ty} {name}' for ty,name in tensors)
  return f'C{{"{fn.func.name}"}}.analyze(function<Tensor({types})>{{[]({types_names}) {{' +\
         f' return static_cast<{ptr_cast}>(at::redispatch::{sig.name()})({key}, {", ".join(args)}); }}}});'

fd = open('scripts/call_pytorch_fns.h', 'w')
for fn in native_functions.native_functions:
  if len(fn.func.returns) != 1 or str(fn.func.returns[0].type) != 'Tensor':
    continue

  print(gen(fn), file=fd)


all_functions = sorted(all_functions, key=lambda p : -p[1])
timeout = 60  # minutes

fd = open('build.ninja', 'w')
print(f'''
rule type
  command = bash -c "timeout -s9 {timeout}m ./infer_types $in > $out 2> /dev/null || true"
  description = type $in

rule shape
  command = bash -c "timeout -s9 {timeout}m ./infer_shapes $in > $out 2> /dev/null || true"
  description = shape $in

rule strides
  command = bash -c "timeout -s9 {timeout}m ./infer_strides $in > $out 2> /dev/null || true"
  description = strides $in

rule merge
  command = bash -c "cat $in | sort > $out"
  description = Assemble final $out file

build types.txt: merge {" ".join(f'types/{fn}.txt' for fn,sz in all_functions)}
build shapes.txt: merge {" ".join(f'shapes/{fn}.txt' for fn,sz in all_functions)}
build strides.txt: merge {" ".join(f'strides/{fn}.txt' for fn,sz in all_functions)}
''', file=fd)

for fn,sz in all_functions:
  print(f'build types/{fn}.txt: type {fn}', file=fd)
  print(f'build shapes/{fn}.txt: shape {fn}', file=fd)
  print(f'build strides/{fn}.txt: strides {fn}', file=fd)
  print(f'build {fn}: phony', file=fd)
