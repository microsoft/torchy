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

@with_native_function
def skip_fn(fn):
  allowed_ret_types = {
    'at::Tensor',
    'at::Tensor &',
    'const at::Tensor &',
  }
  dispatcher_sig = DispatcherSignature.from_schema(fn.func)
  rettype = dispatcher_sig.returns_type().cpp_type()
  if rettype not in allowed_ret_types:
    return True

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


def mk_dtype_infer(type, tensors):
  if type[0:3] == 'ALL':
    return f'scalarTypeToTypeMeta(k{type[4:]})'
  if type == 'BOOL2INT':
    return f'scalarTypeToTypeMeta(bool_to_int({tensors[0].expr}.dtype().toScalarType()))'
  if type == 'EQ_PROMOTED':
    # save code size by using the default function
    if len(tensors) == 1 and get_arg_of_type(tensors, 'at::TensorList'):
      return 'nullopt'
    return f'scalarTypeToTypeMeta(promote_tys({", ".join(t.expr for t in tensors)}))'
  if type == 'EQ_SECOND':
    return f'{tensors[1].expr}.dtype()'
  if type == 'EQ_THIRD':
    return f'{tensors[2].expr}.dtype()'
  if type == 'BOOLBYTE':
    return f'scalarTypeToTypeMeta(bool_byte({tensors[0].expr}.dtype().toScalarType()))'
  if type == 'BOOL2INT':
    return f'scalarTypeToTypeMeta(bool_to_int({tensors[0].expr}.dtype().toScalarType()))'
  if type == 'BOOL2INT2':
    return f'scalarTypeToTypeMeta(bool_to_int2({tensors[0].expr}.dtype().toScalarType(), {tensors[1].expr}.dtype().toScalarType()))'
  if type == 'INTEGRAL2INT':
    return f'scalarTypeToTypeMeta(integrals_to_int({tensors[0].expr}.dtype().toScalarType()))'
  if type == 'TO_COMPLEX':
    return f'scalarTypeToTypeMeta(to_complex({tensors[0].expr}.dtype().toScalarType()))'
  if type == 'TO_DOUBLE':
    return f'scalarTypeToTypeMeta(to_double({tensors[0].expr}.dtype().toScalarType()))'
  if type == 'TO_DOUBLE2':
    return f'scalarTypeToTypeMeta(to_double2({tensors[0].expr}.dtype().toScalarType(), {tensors[1].expr}.dtype().toScalarType()))'
  if type == 'TO_FLOAT':
    return f'scalarTypeToTypeMeta(to_float({tensors[0].expr}.dtype().toScalarType()))'
  if type == 'TO_FLOAT_DOUBLE':
    return f'scalarTypeToTypeMeta(to_float_double({tensors[0].expr}.dtype().toScalarType()))'
  if type == 'TO_FLOAT2':
    return f'scalarTypeToTypeMeta(to_float2({tensors[0].expr}.dtype().toScalarType(), {tensors[1].expr}.dtype().toScalarType()))'
  if type == 'TO_FLOAT2_2':
    return f'scalarTypeToTypeMeta(to_float2_2({tensors[0].expr}.dtype().toScalarType(), {tensors[1].expr}.dtype().toScalarType()))'
  if type == 'TO_FLOAT2_4':
    return f'scalarTypeToTypeMeta(to_float2_4({tensors[0].expr}.dtype().toScalarType(), {tensors[1].expr}.dtype().toScalarType()))'
  if type == 'TO_FLOAT3':
    return f'scalarTypeToTypeMeta(to_float3({tensors[0].expr}.dtype().toScalarType(), {tensors[1].expr}.dtype().toScalarType(), {tensors[2].expr}.dtype().toScalarType()))'
  if type == 'TO_FLOAT4':
    return f'scalarTypeToTypeMeta(to_float4({tensors[0].expr}.dtype().toScalarType(), {tensors[1].expr}.dtype().toScalarType(), {tensors[2].expr}.dtype().toScalarType(), optional_type({tensors[3].expr})))'
  if type == 'TO_QINT':
    return f'scalarTypeToTypeMeta(toQIntType({tensors[0].expr}.dtype().toScalarType()))'
  if type == 'TO_REAL2':
    return f'scalarTypeToTypeMeta(to_real2({tensors[0].expr}.dtype().toScalarType(), {tensors[1].expr}.dtype().toScalarType()))'
  if type == 'TO_REAL_FLOAT':
    return f'scalarTypeToTypeMeta(to_real_float({tensors[0].expr}.dtype().toScalarType()))'
  if type == 'TO_VALUE_TYPE':
    return f'scalarTypeToTypeMeta(toValueType({tensors[0].expr}.dtype().toScalarType()))'
  print('mk_dtype_infer', type)
  exit()


def get_dtype_arg(all_tensors, args, name):
  tensors = [a.expr for a in all_tensors if a.type.remove_const_ref().cpp_type() == 'at::Tensor']
  dtype = 'nullopt'
  device = 'nullopt'
  if tensors:
    dtype  = f'{tensors[0]}.dtype()'
    device = f'{tensors[0]}.device()'

  name = str(name)
  if name in type_inference:
    dtype = mk_dtype_infer(type_inference[name], all_tensors)

  dtype_arg = get_arg_of_type(args, 'at::ScalarType')
  if dtype_arg:
    device_arg = get_arg_of_type(args, 'at::Device')
    device_arg = device_arg.expr if device_arg else device
    dtype_arg = dtype_arg.expr
    return f'{dtype_arg}, {device_arg}'

  dtype_arg = get_arg_of_type(args, 'c10::optional<at::ScalarType>')
  if dtype_arg:
    device_arg = get_arg_of_type(args, 'c10::optional<at::Device>')
    dtype_arg = dtype_arg.expr
    if tensors:
      device_arg = device_arg.expr if device_arg else 'nullopt'
      return f'{tensors[0]}, {dtype_arg}, {device_arg}'

    device_arg = device_arg.expr if device_arg else device
    return f'{dtype_arg}, {device_arg}'

  tensor_arg = get_arg_of_type(args, 'at::TensorList')
  if dtype == 'nullopt' and tensor_arg:
    return tensor_arg.expr

  return f'{dtype}, {device}'


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

  dispatchkey = "dispatchKeySet = dispatchKeySet & DispatchKeySet(DispatchKeySet::FULL_AFTER, DISPATCHKEY);"

  # returns a tensor and takes tensors as arguments
  # e.g. add(x, y)
  if rettype == 'at::Tensor':
    dtype_device = get_dtype_arg(tensor_args, args, fn.func.name)

    return f'''
{fndecl} {{
  if (trace.is_flushing()) {{
    {dispatchkey}
    return {redispatch};
  }}
  auto tt = register_new_tensor(dispatchKeySet, {fn_enum(fn)}, {dtype_device});
  {register_args}
  return tt;
}}'''

  # in-place op. returns one of the arguments
  # e.g. mul_ or mul_out
  assert rettype == 'at::Tensor &' or rettype == 'const at::Tensor &'
  assert tensor_args
  ret = fn_output(fn)

  return f'''
{fndecl} {{
  if (trace.is_flushing()) {{
    {dispatchkey}
    return {redispatch};
  }}
  bool flush = register_in_place({ret}, {fn_enum(fn)}, dispatchKeySet);
  {register_args}
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  return {ret};
}}'''


@with_native_function
def gen_torch_library_table(fn):
  return f'm.impl("{fn.func.name}", {wrapper_name(fn)});'

enum_names = {}
@with_native_function
def gen_ops_names(fn):
  enum_names[fn_enum(fn)] = fn.func.name


# (inplace, code, redispatch_signature) -> (enum, fn_ptr)*
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
    inplace = False

  # in-place op
  else:
    assert rettype == 'at::Tensor &' or rettype == 'const at::Tensor &'
    inplace = True
    code = f'{redispatch};\n  break;'

  signature = dispatcher_sig.type()
  fn_ptr = f'at::redispatch::{sig.name()}'
  key = inplace, code, signature
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
# put all inplaces last
interpreter_code = sorted(interpreter_code.items())
is_first_inplace = True

for ((inplace, code, sig), entries) in interpreter_code:
  if inplace and is_first_inplace:
    is_first_inplace = False
    print(f'#define FIRST_INPLACE_OP {entries[0][0]}\n', file=fd6)

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
