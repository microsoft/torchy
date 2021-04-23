# Copyright (c) 2021-present The Torchy Authors.
# Distributed under the MIT license that can be found in the LICENSE file.

PYTORCH = '../pytorch'

import sys
sys.path.append(PYTORCH)
from tools.codegen.gen import *

yaml_path = PYTORCH + '/aten/src/ATen/native/native_functions.yaml'
native_functions = parse_native_yaml(yaml_path)

for f in native_functions:
  if f.func.name.name.base == 'empty_strided':
    print(f.func)
    print(f.func.name)
    for a in f.func.arguments.flat_non_out + list(f.func.arguments.out):
      a = a.argument if isinstance(a, SelfArgument) else a
      print(' arg', a, a.annotation.is_write if a.annotation else '')
    print(f.func.returns[0].type.name)
    print(f.func.returns[0].is_write)
    if f.func.returns[0].annotation:
      print(f.func.returns[0].annotation.alias_set[0])
    print()

def skip_fn(fn):
  return not fn.dispatch or fn.manual_cpp_binding

def wrapper_name(fn):
  return 'wrap_' + str(fn.func.name).replace('.', '_')

def fn_enum(fn):
  return 'H_' + str(fn.func.name).replace('.', '_').upper()


@with_native_function
def gen_dispatch_wrapper(fn):
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

    sig = DispatcherSignature.from_schema(fn.func)
    rettype = sig.returns_type().cpp_type_registration_declarations()
    fndecl = sig.defn(name=wrapper_name(fn))

    # TODO: Tensor[] and void?
    if rettype != 'Tensor' and rettype != 'Tensor &':
      return f'''
{fndecl} {{
  ensure_materialized({', '.join(rargs_tensors)});
  return at::redispatch::{fn.func.name.name}({', '.join(rargs)});
}}'''

    elif not rargs_tensors:
      return f'''
{fndecl} {{
  ensure_materialized({', '.join(rargs_tensors)});
  return at::detail::make_tensor<TorchyTensor>(
           at::redispatch::{fn.func.name.name}({', '.join(rargs)}));
}}'''

    else:
      return f'''
{fndecl} {{
  if (trace.is_flushing()) {{
    ensure_materialized({', '.join(rargs_tensors)});
    return at::redispatch::{fn.func.name.name}({', '.join(rargs)});
  }}
  return MK_TORCHY({dtype}, {device}, {fn_enum(fn)}, {', '.join(rargs)});
}}'''


@with_native_function
def gen_torch_library_table(fn):
  return f'  m.impl("{fn.func.name}", {wrapper_name(fn)});'


fd1 = open('autogen/dispatch_wrappers.h', 'w')

fd2 = open('autogen/torch_library_table.h', 'w')
print('TORCH_LIBRARY_IMPL(aten, DISPATCHKEY_NO_NS, m) {', file=fd2)

for fn in native_functions:
  if skip_fn(fn):
    continue
  print(gen_dispatch_wrapper(fn), file=fd1)
  print(gen_torch_library_table(fn), file=fd2)

print('}', file=fd2)
