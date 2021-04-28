# Copyright (c) 2021-present The Torchy Authors.
# Distributed under the MIT license that can be found in the LICENSE file.

PYTORCH = '../pytorch'

from special_fns import *
import sys
sys.path.append(PYTORCH)
from tools.codegen.gen import *

yaml_path = PYTORCH + '/aten/src/ATen/native/native_functions.yaml'
native_functions = parse_native_yaml(yaml_path)

def skip_fn(fn):
  return not fn.dispatch or fn.manual_cpp_binding

def wrapper_name(fn):
  return 'wrap_' + str(fn.func.name).replace('.', '_')

def fn_enum(fn):
  return 'H_' + str(fn.func.name).replace('.', '_').upper()

def get_arg_of_type(args, type):
  for arg in args:
    if arg.type.cpp_type(strip_ref=True) == type:
      return arg
  return None

def maybe_tensor(type):
  types = {
    'at::Tensor',
    'at::TensorList',
    'c10::List<c10::optional<at::Tensor>>',
    'c10::optional<at::Tensor>',
  }
  return type.remove_const_ref().cpp_type() in types

# returns (dtype, device, init code)
def get_dtype_arg(args, dtype, device):
  uses_default = False
  tensors = [a.expr for a in args if a.type.remove_const_ref().cpp_type() == 'at::Tensor']
  if tensors:
    dtype_default  = f'{tensors[0]}.dtype()'
    device_default = f'{tensors[0]}.device()'
    needs_init_code = False
  else:
    dtype_default  = 'default_dtype'
    device_default = 'default_device'
    needs_init_code = True

  if not dtype:
    dtype = dtype_default
    uses_default = True
  if not device:
    device = device_default
    uses_default = True

  def fix_types(t, default, cast = None):
    if isinstance(t, str):
      return t, False
    if 'c10::optional' in t.type.remove_const_ref().cpp_type():
      if cast:
        return f'{t.expr} ? {cast}(*{t.expr}) : {default}', True
      return f'{t.expr} ? *{t.expr} : {default}', True
    return f'{cast}({t.expr})' if cast else t.expr, False

  dtype, tdef  = fix_types(dtype, dtype_default, 'scalarTypeToTypeMeta')
  device, ddef = fix_types(device, device_default)
  uses_default |= tdef or ddef

  init_code = ''
  if uses_default and needs_init_code:
    args = ', '.join([a.expr for a in args])
    init_code = f'''
  auto defaults = compute_dtype({args});
  auto &default_dtype = defaults.first;
  auto &default_device = defaults.second;'''

  return dtype, device, init_code


@with_native_function
def gen_dispatch_wrapper(fn):
  sig_group = CppSignatureGroup.from_native_function(fn, method=False, fallback_binding=fn.manual_cpp_binding)
  sig = sig_group.faithful_signature if sig_group.faithful_signature else sig_group.signature
  dispatcher_sig = DispatcherSignature.from_schema(fn.func)

  rettype = dispatcher_sig.returns_type().cpp_type()
  fndecl = sig.defn(prefix='wrap_', is_redispatching_fn=True)
  fndecl = fndecl.replace('wrap_' + sig.name(), wrapper_name(fn))

  args = translate(sig.arguments(), dispatcher_sig.arguments())
  rargs = ', '.join(['dispatchKeySet'] + [a.expr for a in args])
  redispatch = f'at::redispatch::{sig.name()}({rargs})'

  tensor_args = [a for a in args if maybe_tensor(a.type)]
  materialize = f"ensure_materialized({', '.join([a.expr for a in tensor_args])});"

  dispatchkey = "dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);"

  # returns a tensor and takes tensors as arguments
  # e.g. add(x, y)
  if rettype == 'at::Tensor':
    dtype = get_arg_of_type(args, 'at::ScalarType')
    if not dtype:
      dtype = get_arg_of_type(args, 'c10::optional<at::ScalarType>')
    if fn.func.name.name.base in always_returns_bool:
      dtype = 'scalarTypeToTypeMeta(kBool)'

    device = get_arg_of_type(args, 'at::Device')
    if not device:
      device = get_arg_of_type(args, 'c10::optional<at::Device>')

    dtype, device, dtype_init = get_dtype_arg(tensor_args, dtype, device)

    return f'''
{fndecl} {{
  if (trace.is_flushing()) {{
    {materialize}
    {dispatchkey}
    return {redispatch};
  }}{dtype_init}
  return at::detail::make_tensor<TorchyTensor>({dtype}, {device}, {fn_enum(fn)}, {rargs});
}}'''

  # in-place op. returns one of the arguments
  # e.g. mul_ or mul_out
  if rettype == 'at::Tensor &':
    assert tensor_args
    if fn.func.arguments.out:
      assert len(fn.func.arguments.out) == 1
      ret = fn.func.arguments.out[0].name
    else:
      assert fn.func.arguments.self_arg.argument.is_write
      ret = fn.func.arguments.self_arg.argument.name

    # TODO: we can also make it lazy if tensor is non-torchy but ref count == 1
    return f'''
{fndecl} {{
  auto tt = is_torchy({ret});
  if (tt && !trace.is_flushing()) {{
    tt->addInplace({fn_enum(fn)}, {rargs});
    return {ret};
  }}
  will_override({ret});
  {materialize}
  {dispatchkey}
  return {redispatch};
}}'''

  # returns e.g. a scalar. must materialize right away
  return f'''
{fndecl} {{
  {materialize}
  {dispatchkey}
  return {redispatch};
}}'''


@with_native_function
def gen_torch_library_table(fn):
  return f'm.impl("{fn.func.name}", {wrapper_name(fn)});'

@with_native_function
def gen_ops_enum(fn):
  return f'{fn_enum(fn)},'

@with_native_function
def gen_ops_names(fn):
  return f'"{fn.func.name}",'

@with_native_function
def gen_interpreter_redispatch(fn):
  sig_group = CppSignatureGroup.from_native_function(fn, method=False, fallback_binding=fn.manual_cpp_binding)
  sig = sig_group.faithful_signature if sig_group.faithful_signature else sig_group.signature
  dispatcher_sig = DispatcherSignature.from_schema(fn.func)

  dispatcher_exprs = translate(sig.arguments(), dispatcher_sig.arguments())
  args = []
  i = 0
  for arg in dispatcher_exprs:
    type = arg.type.cpp_type(strip_ref=True)
    args.append(f'get<{type}>(op.args[{i}])')
    i += 1

  redispatch = f'at::redispatch::{sig.name()}(ks, {", ".join(args)})'

  rettype = dispatcher_sig.returns_type().cpp_type()
  case = f'case {fn_enum(fn)}:'

  if rettype == 'at::Tensor':
    return f'''{case}
  set(op.tensor, {redispatch});
  break;
'''

  # in-place op
  if rettype == 'at::Tensor &':
    return f'''{case}
  init_update_in_place(op.tensor);
  {redispatch};
  end_update_in_place(op.tensor);
  break;
'''

  # nothing else gets interpreted
  return f'// skip {sig.defn()}\n'


fd1 = open('autogen/dispatch_wrappers.h', 'w')
fd2 = open('autogen/torch_library_table.h', 'w')
fd3 = open('autogen/ops_enum.h', 'w')
fd4 = open('autogen/ops_names.h', 'w')
fd5 = open('autogen/interpreter_redispatch.h', 'w')

for fn in native_functions:
  if skip_fn(fn):
    continue
  print(gen_dispatch_wrapper(fn), file=fd1)
  print(gen_torch_library_table(fn), file=fd2)
  print(gen_ops_enum(fn), file=fd3)
  print(gen_ops_names(fn), file=fd4)
  print(gen_interpreter_redispatch(fn), file=fd5)