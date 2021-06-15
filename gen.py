# Copyright (c) 2021-present The Torchy Authors.
# Distributed under the MIT license that can be found in the LICENSE file.

PYTORCH = '../pytorch'

from special_fns import *
import sys
sys.path.append(PYTORCH)
from tools.codegen.gen import *
from tools.codegen.api import types

yaml_path = PYTORCH + '/aten/src/ATen/native/native_functions.yaml'
native_functions = parse_native_yaml(yaml_path)

def skip_fn(fn):
  # TODO: benchmark if we really want to skip these
  # plus check if we want to skip is_generic_dispatch_key(key) as well
  # as those don't have real kernels, just autograd
  return not any(d.has_kernel(fn) for d in native_functions.backend_indices.values())

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


def fn_output(fn):
  if fn.func.arguments.out:
    assert len(fn.func.arguments.out) == 1
    return fn.func.arguments.out[0].name
  else:
    assert fn.func.arguments.self_arg.argument.is_write
    return fn.func.arguments.self_arg.argument.name


def move_if_needed(str, arg):
  basic_types = {
    'bool',
    'int64_t',
    'double',
  }
  free_copy_types = (
    types.ArrayRefCType,
    types.ConstRefCType,
    types.MutRefCType,
  )
  def free(type):
    return isinstance(type, free_copy_types) or \
           type.cpp_type() in basic_types

  if free(arg.type.type) or \
     (isinstance(arg.type.type, types.OptionalCType) and free(arg.type.type.elem)):
    return str
  return f'std::move({str})'


@with_native_function
def gen_dispatch_wrapper(fn):
  sig_group = CppSignatureGroup.from_native_function(fn, method=False, fallback_binding=fn.manual_cpp_binding)
  sig = sig_group.faithful_signature if sig_group.faithful_signature else sig_group.signature
  dispatcher_sig = DispatcherSignature.from_schema(fn.func)

  rettype = dispatcher_sig.returns_type().cpp_type()
  fndecl = sig.defn(prefix='wrap_', is_redispatching_fn=True)
  fndecl = fndecl.replace('wrap_' + sig.name(), wrapper_name(fn))

  args = translate(sig.arguments(), dispatcher_sig.arguments())
  register_args = ''.join([f'trace.append_arg({move_if_needed(a.expr, a)});' for a in args])

  rargs = ', '.join(['dispatchKeySet'] + [move_if_needed(a.expr, a) for a in args])
  redispatch = f'at::redispatch::{sig.name()}({rargs})'

  tensor_args = [a for a in args if maybe_tensor(a.type)]
  materialize = ''.join([f'ensure_materialized({a.expr});' for a in tensor_args])

  dispatchkey = "dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);"

  # returns a tensor and takes tensors as arguments
  # e.g. add(x, y)
  if rettype == 'at::Tensor':
    dtype = get_arg_of_type(args, 'at::ScalarType')
    if not dtype:
      dtype = get_arg_of_type(args, 'c10::optional<at::ScalarType>')

    fixed = fix_return_type.get(fn.func.name.name.base)
    if fixed:
      dtype = f'scalarTypeToTypeMeta({fixed})'

    device = get_arg_of_type(args, 'at::Device')
    if not device:
      device = get_arg_of_type(args, 'c10::optional<at::Device>')

    dtype, device, dtype_init = get_dtype_arg(tensor_args, dtype, device)

    return f'''
{fndecl} {{
  if (trace.is_flushing()) {{
    {dispatchkey}
    return {redispatch};
  }}{dtype_init}
  auto tt = register_new_tensor(dispatchKeySet, {fn_enum(fn)}, {dtype}, {device});
  {register_args}
  return tt;
}}'''

  # in-place op. returns one of the arguments
  # e.g. mul_ or mul_out
  if rettype == 'at::Tensor &' or\
     (fn.use_const_ref_for_mutable_tensors and rettype == 'const at::Tensor &'):
    assert tensor_args
    ret = fn_output(fn)

    # TODO: we can also make it lazy if tensor is non-torchy but ref count == 1
    return f'''
{fndecl} {{
  if (trace.is_flushing()) {{
    {dispatchkey}
    return {redispatch};
  }}
  TorchyTensor *tt = prepare_in_place({ret});
  unsigned trace_idx = trace.register_tensor(tt ? (uintptr_t)tt : DUMMY_TORCHY, {fn_enum(fn)}, dispatchKeySet);
  {register_args}
  finish_in_place(tt, trace_idx);
  return {ret};
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

enum_names = {}
@with_native_function
def gen_ops_names(fn):
  enum_names[fn_enum(fn)] = fn.func.name


# (code, redispatch_signature) -> (enum, fn_ptr)*
interpreter_code = {}

@with_native_function
def gen_interpreter_redispatch(fn):
  global interpreter_code
  sig_group = CppSignatureGroup.from_native_function(fn, method=False, fallback_binding=fn.manual_cpp_binding)
  sig = sig_group.faithful_signature if sig_group.faithful_signature else sig_group.signature
  dispatcher_sig = DispatcherSignature.from_schema(fn.func)

  dispatcher_exprs = translate(sig.arguments(), dispatcher_sig.arguments())
  args = []
  for i, arg in enumerate(dispatcher_exprs):
    type = arg.type.cpp_type(strip_ref=False)
    type = type.replace('const ', '')
    args.append(f'load<{type}>()(op.args[{i}])')

  redispatch = f'<FN>(ks, {", ".join(args)})'
  rettype = dispatcher_sig.returns_type().cpp_type()

  if rettype == 'at::Tensor':
    code = f'set(op, {redispatch});\n  continue;'

  # in-place op
  elif rettype == 'at::Tensor &' or\
       (fn.use_const_ref_for_mutable_tensors and rettype == 'const at::Tensor &'):
    code = f'''init_update_in_place(op);
  {redispatch};
  break;'''
  else:
    # nothing else gets interpreted
    return

  signature = dispatcher_sig.type()
  fn_ptr = f'at::redispatch::{sig.name()}'
  key = code, signature
  interpreter_code.setdefault(key, [])
  interpreter_code[key].append((fn_enum(fn), fn_ptr))


fd1 = open('autogen/dispatch_wrappers.h', 'w')
fd2 = open('autogen/torch_library_table.h', 'w')
fd3 = open('autogen/ops_enum.h', 'w')
fd4 = open('autogen/ops_names.h', 'w')
fd5 = open('autogen/interpreter_redispatch.h', 'w')
fd6 = open('autogen/interpreter_redispatch_tables.h', 'w')

total = 0
for fn in native_functions.native_functions:
  if skip_fn(fn):
    continue
  total += 1
  print(gen_dispatch_wrapper(fn), file=fd1)
  print(gen_torch_library_table(fn), file=fd2)
  gen_ops_names(fn)
  gen_interpreter_redispatch(fn)

print(f'Total redispatched functions: {total}')
print(f'Distinct signatures: {len(interpreter_code)}')

table_id = 0
for ((code, sig), entries) in interpreter_code.items():
  for (enum, ptr) in entries:
    print(f'case {enum}:', file=fd5)
    print(f'{enum},', file=fd3)
    print(f'"{enum_names[enum]}",', file=fd4)

  if len(entries) == 1:
    code = code.replace('<FN>', entries[0][1])
    print(f'  {code}\n', file=fd5)
  elif len(entries) == 2:
    ptr = sig.replace(' (', f'(*ptr)(DispatchKeySet, ')
    print(f'  {{{ptr} = {entries[0][1]};', file=fd5)
    print(f'  if (op.id == {entries[1][0]}) ptr = {entries[1][1]};', file=fd5)
    code = code.replace('<FN>', f'ptr')
    print(f'  {code}}}\n', file=fd5)
  else:
    table = f'redispatch_ptrs_{table_id}'
    table_id += 1
    code = code.replace('<FN>', f'{table}[op.id - {entries[0][0]}]')
    print(f'  {code}\n', file=fd5)

    table = sig.replace(' (', f'(*const {table}[])(DispatchKeySet, ')
    print(f'{table} = {{', file=fd6)
    for (enum, ptr) in entries:
      print(f'  {ptr},', file=fd6)
    print(f'}};\n', file=fd6)
