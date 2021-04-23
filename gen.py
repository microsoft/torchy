# Copyright (c) 2021-present The Torchy Authors.
# Distributed under the MIT license that can be found in the LICENSE file.

PYTORCH = '../pytorch'

import sys
sys.path.append(PYTORCH)
from tools.codegen.gen import *

yaml_path = PYTORCH + '/aten/src/ATen/native/native_functions.yaml'
native_functions = parse_native_yaml(yaml_path)

for f in native_functions:
  if f.func.name.name.base == 'mul':
    print(f.func)
    print(f.func.name)
    print(f.func.arguments)
    for a in f.func.arguments.flat_non_out + list(f.func.arguments.out):
      a = a.argument if isinstance(a, SelfArgument) else a
      print(' arg', a, a.annotation.is_write if a.annotation else '')
    print(f.func.returns[0].type.name)
    print(f.func.returns[0].is_write)
    if f.func.returns[0].annotation:
      print(f.func.returns[0].annotation.alias_set[0])
    print()

def wrapper_name(fn):
  return 'wrap_' + str(fn.func.name).replace('.', '_')

def fn_enum(fn):
  return 'H_' + str(fn.func.name).replace('.', '_').upper()


def gen_dispatch_wrappers(f):
  for fn in native_functions:
    if not fn.dispatch or fn.manual_cpp_binding:
      continue

    # collect arguments for redispatch
    rargs         = []
    rargs_tensors = []
    dtype         = None
    device        = None
    for a in list(fn.func.arguments.out) + fn.func.arguments.flat_non_out:
      a = a.argument if isinstance(a, SelfArgument) else a
      rargs.append(a.name)
      if str(a.type) == 'Tensor':
        rargs_tensors.append(a.name)
        if not dtype:
          dtype  = f'{a.name}.dtype()'
          device = f'{a.name}.device()'

    rettype = str(fn.func.returns[0].type) if fn.func.returns else 'void'

    # TODO: Tensor[] and void?
    if rettype != 'Tensor':
      print(f'''
{rettype} {wrapper_name(fn)}(args...) {{
  ensure_materialized({', '.join(rargs_tensors)});
  return at::redispatch::{fn.func.name.name}({', '.join(rargs)});
}}''', file=f)
    else:
      print(f'''
{rettype} {wrapper_name(fn)}(args...) {{
  if (trace.is_flushing()) {{
    ensure_materialized({', '.join(rargs_tensors)});
    return at::redispatch::{fn.func.name.name}({', '.join(rargs)});
  }}
  return MK_TORCHY({dtype}, {device}, {fn_enum(fn)}, {', '.join(rargs)});
}}''', file=f)


def gen_torch_library_table(f):
  print('TORCH_LIBRARY_IMPL(aten, DISPATCHKEY_NO_NS, m) {', file=f)
  for fn in native_functions:
    if fn.dispatch and not fn.manual_cpp_binding:
      print(f'  m.impl("{fn.func.name}", {wrapper_name(fn)});', file=f)
  print('}', file=f)


with open('autogen/dispatch_wrappers.h', 'w') as f:
  gen_dispatch_wrappers(f)

with open('autogen/torch_library_table.h', 'w') as f:
  gen_torch_library_table(f)
