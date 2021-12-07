# Copyright (c) 2021-present The Torchy Authors.
# Distributed under the MIT license that can be found in the LICENSE file.

PYTORCH = '../pytorch'

from typings_data import *
import sys
sys.path.append(PYTORCH)
from tools.codegen.gen import *
from tools.codegen.api import types

yaml_path = PYTORCH + '/aten/src/ATen/native/native_functions.yaml'
native_functions = parse_native_yaml(yaml_path)

dtype_exceptions = {
}

shape_exceptions = {
  'arange.start_out'  : 'ARANGE',
  'arange.start_step' : 'ARANGE',
  'cat'               : 'CAT',
  'conv2d'            : 'CONV2D',
  'embedding'         : 'EMBEDDING',
  'max_pool2d'        : 'CONV2D',
  'mkldnn_convolution': 'CONV2D2',
  'slice.Tensor'      : 'SLICE',
  'stack'             : 'STACK',
  'stack.out'         : 'STACK',
  'transpose_'        : '',
}

strides_exceptions = {
  'clone': 'CLONE',
  'embedding': 'CONTIGUOUS',
}

def get_dtype_infer_fn(fn):
  name = str(fn.func.name)
  return dtype_exceptions.get(name, type_inference.get(name))

def get_shape_infer_fn(fn):
  name = str(fn.func.name)
  return shape_exceptions.get(name, shape_inference.get(name))

def get_strides_infer_fn(fn):
  name = str(fn.func.name)
  return strides_exceptions.get(name, strides_inference.get(name))


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
  return False

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

def to_scalar_type(v):
  ty = v.type.remove_const_ref().cpp_type()
  if ty == 'at::Tensor':
    return f'{v.expr}.scalar_type()'
  if ty == 'c10::optional<at::ScalarType>':
    return f'{v.expr}.value_or(ScalarType::Undefined)'
  print('to_scalar_type', ty)
  exit(-1)

def is_type_arg(arg):
  type = arg.type.remove_const_ref().cpp_type()
  dispatch_types = [
    'at::Scalar',
    'at::ScalarType',
    'c10::optional<at::Scalar>',
    'c10::optional<at::ScalarType>',
  ]
  return 'Tensor' in type or type in dispatch_types

def to_dtype(arg):
  type = arg.type.cpp_type()
  if type == 'at::ScalarType':
    return arg.expr
  return f'{arg.expr}.dtype()'

def to_scalartype(arg):
  type = arg.type.remove_const_ref().cpp_type()
  if type == 'at::ScalarType':
    return arg.expr
  if type == 'at::Scalar':
    return f'{arg.expr}.type()'
  return f'{arg.expr}.scalar_type()'


def mk_dtype_infer(type, all_args):
  args = [arg for arg in all_args if is_type_arg(arg)]

  if type[0:3] == 'ALL':
    return f'k{type[4:]}'
  if type == 'BOOL2INT':
    return f'bool_to_int({args[0].expr}.scalar_type())'
  if type == 'EQ_PROMOTED':
    return f'promote_tys({", ".join(t.expr for t in args)})'
  if type == 'EQ_PROMOTED_BUGGY':
    return f'promote_buggy({", ".join(t.expr for t in args)})'
  if type == 'EQ_PROMOTED_BUGGY2':
    return f'promote_buggy({args[0].expr}, {args[1].expr})'
  if type == 'EQ_PROMOTED_CONST':
    return f'promote_const({", ".join(t.expr for t in args)})'
  if type == 'EQ_SECOND':
    return to_dtype(args[1])
  if type == 'EQ_THIRD':
    return to_dtype(args[2])
  if type == 'EQ_FOURTH':
    return to_dtype(args[3])
  if type == 'BOOLBYTE':
    return f'bool_byte({args[0].expr}.scalar_type())'
  if type == 'BOOL2INT':
    return f'bool_to_int({args[0].expr}.scalar_type())'
  if type == 'INTEGRAL2INT':
    return f'integrals_to_int({args[0].expr}.scalar_type())'
  if type == 'TO_COMPLEX':
    return f'to_complex({args[0].expr}.scalar_type())'
  if type == 'TO_DOUBLE2':
    return f'to_double2({to_scalartype(args[0])}, {to_scalartype(args[1])})'
  if type == 'TO_FLOAT':
    return f'to_float({args[0].expr}.scalar_type())'
  if type == 'TO_FLOAT_DOUBLE':
    return f'to_float_double({args[0].expr}.scalar_type())'
  if type == 'TO_FLOAT2':
    return f'to_float2({args[0].expr}, {args[1].expr})'
  if type == 'TO_FLOAT3':
    return f'to_float3({args[0].expr}, {args[1].expr}, {args[2].expr})'
  if type == 'TO_FLOAT4':
    return f'to_float4({args[0].expr}, {args[1].expr}, {args[2].expr}, {args[3].expr})'
  if type == 'TO_QINT':
    return f'toQIntType({args[0].expr}.scalar_type())'
  if type == 'TO_REAL2':
    return f'to_real2({args[0].expr}, {args[1].expr})'
  if type == 'TO_REAL_FLOAT':
    return f'to_real_float({to_scalar_type(args[0])})'
  if type == 'TO_VALUE_TYPE':
    return f'toValueType({args[0].expr}.scalar_type())'
  if type == 'OPTIONAL_OR21':
    return f'optional_or_else({args[1].expr}, {args[0].expr}.scalar_type())'
  if type == 'OPTIONAL_OR31':
    return f'optional_or_else({args[2].expr}, {args[0].expr}.scalar_type())'
  if type == 'OPTIONAL_O21LONG':
    return f'optional_or_longelse({args[1].expr}, {args[0].expr}.scalar_type())'
  if type == 'FIRST_OR_DEFAULT':
    return args[0].expr
  if type == 'SECOND_OR_DEFAULT':
    return args[1].expr
  if type == 'FIRST_OR_LONG':
    return f'optional_or_else({args[0].expr}, kLong)'
  if type == 'SECOND_OR_LONG_DEFAULT':
    return f'optional_or_longdefault({args[1].expr}, {args[0].expr}.type())'
  print('mk_dtype_infer', type)
  exit()


def get_dtype_arg(all_tensors, args, name):
  tensors = [a.expr for a in all_tensors if a.type.remove_const_ref().cpp_type() == 'at::Tensor']
  tensor_lst = [a.expr for a in all_tensors if a.type.remove_const_ref().cpp_type() == 'at::TensorList']
  dtype = 'nullopt'
  device = 'nullopt'
  if tensors:
    dtype  = f'{tensors[0]}.dtype()'
    device = f'{tensors[0]}.device()'
  elif tensor_lst:
    device = f'device_of({tensor_lst[0]})'

  device_arg = get_arg_of_type(args, 'at::Device')
  if device_arg:
    device = device_arg.expr

  device_arg = get_arg_of_type(args, 'c10::optional<at::Device>')
  if device_arg:
    device = device_arg.expr

  dtype_fn = get_dtype_infer_fn(fn)
  if dtype_fn:
    dtype = mk_dtype_infer(dtype_fn, args)
  else:
    dtype_arg = get_arg_of_type(args, 'at::ScalarType')
    if dtype_arg:
      dtype = dtype_arg.expr

    dtype_arg = get_arg_of_type(args, 'c10::optional<at::ScalarType>')
    if dtype_arg:
      dtype = dtype_arg.expr

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
    'at::Device',
    'at::Dimname',
    'at::DimnameList',
    'at::IntArrayRef',
    'at::MemoryFormat',
    'at::ScalarType',
    'at::Layout',
    'at::TensorList',
    'c10::string_view',
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

def is_shape_arg(arg):
  type = arg.type.cpp_type()
  dispatch_types = [
    'bool',
    'int64_t',
    'at::IntArrayRef',
    'c10::optional<int64_t>',
    'c10::optional<at::MemoryFormat>',
  ]
  return 'Tensor' in type or type in dispatch_types

def mk_shape_infer(shape, all_args):
  args = [arg for arg in all_args if is_shape_arg(arg)]

  if shape == 'ALL []':
    return 'IntArrayRef()'
  if shape == 'ALL [0]':
    return 'IntArrayRef(0)'
  if shape == 'ALL [1]':
    return 'IntArrayRef(1)'
  if shape == 'EQ_FIRST':
    return args[0].expr
  if shape == 'EQ_SECOND':
    return args[1].expr
  if shape == 'EQ_THIRD':
    return args[2].expr
  if shape == 'STD_PROMOTE':
    args = [arg.expr for arg in all_args if 'Tensor' in arg.type.cpp_type() or 'at::IntArrayRef' in arg.type.cpp_type()]
    return f'shape_std_promote({", ".join(args)})'
  if shape == 'PROMOTE_1_2':
    return f'shape_std_promote({args[0].expr}, {args[1].expr})'
  if shape == 'PROMOTE_1_2_3':
    return f'shape_std_promote({args[0].expr}, {args[1].expr}, {args[2].expr})'
  if shape == 'MATMUL_1ST_2ND':
    return f'shape_matmul({args[0].expr}, {args[1].expr})'
  if shape == 'MATMUL_2ND_3RD':
    return f'shape_matmul({args[1].expr}, {args[2].expr})'
  if shape == 'MUL_1ST_2ND':
    if args[0].type.cpp_type() == 'at::TensorList':
      return f'shape_mul({args[0].expr})'
    return f'shape_mul({args[0].expr}, {args[1].expr})'
  if shape == 'MULT_1ST_2ND':
    return f'shape_mult({args[0].expr}, {args[1].expr})'
  if shape == 'MULLAST_1ST_2ND':
    return f'shape_mul_last({args[0].expr}, {args[1].expr})'
  if shape == 'PICK_1ST_2ND':
    return f'shape_pick_1st({args[1].expr})'
  if shape == 'JOIN_2_3':
    return f'shape_join({args[1].expr}, {args[2].expr})'
  if shape == 'PAD1':
    return f'shape_pad1({args[0].expr})'
  if shape == 'DROP1':
    return f'shape_drop1({args[0].expr})'
  if shape == 'DROP2':
    return f'shape_drop2({args[0].expr})'
  if shape == 'TRANSPOSE':
    return f'shape_transpose({args[0].expr}, {all_args[1].expr}, {all_args[2].expr})'
  if shape == 'RESHAPE':
    return f'shape_reshape({args[0].expr}, {args[1].expr})'
  if shape == 'SELECT':
    return f'shape_select({args[0].expr}, {args[1].expr})'
  if shape == 'UNSQUEEZE':
    return f'shape_unsqueeze({args[0].expr}, {all_args[1].expr})'
  if shape == 'FLATTEN':
    return f'shape_flatten({args[0].expr}, {all_args[1].expr}, {all_args[2].expr})'
  if shape == 'ARANGE':
    return f'shape_arange({all_args[0].expr}, {all_args[1].expr}, {all_args[2].expr})'
  if shape == 'EMBEDDING':
    return f'shape_embedding({args[0].expr}, {args[1].expr})'
  if shape == 'SLICE':
    return f'shape_slice({args[0].expr}, {all_args[1].expr}, {all_args[2].expr}, {all_args[3].expr}, {all_args[4].expr})'
  if shape == 'STACK':
    return f'shape_stack({args[0].expr}, {all_args[1].expr})'
  if shape == 'CAT':
    return f'shape_cat({args[0].expr}, {args[1].expr})'
  if shape == 'ARGMAX':
    return f'shape_argmax({args[0].expr}, {args[1].expr}, {args[2].expr})'
  if shape == 'CONV2D':
    off = 0 if args[2].type.cpp_type() == 'at::IntArrayRef' else 1
    return f'shape_conv2d({args[0].expr}, {args[1].expr}, {args[2+off].expr}, {args[3+off].expr}, {args[4+off].expr})'
  if shape == 'CONV2D2':
    return f'shape_conv2d({args[0].expr}, {args[1].expr}, {args[4].expr}, {args[3].expr}, {args[5].expr})'
  if shape == 'POOL2D':
    return f'shape_pool2d({args[0].expr}, {args[1].expr})'
  if shape == 'TRANSPOSE2D':
    return f'shape_transpose2d({args[0].expr})'
  if shape == 'REDUCE':
    return f'shape_reduce({args[0].expr}, {args[1].expr}, {args[2].expr})'
  if shape == 'PERMUTE':
    return f'shape_permute({args[0].expr}, {args[1].expr})'
  if shape == 'UNFOLD':
    return f'shape_unfold({args[0].expr}, {all_args[1].expr}, {all_args[2].expr}, {all_args[3].expr})'
  if shape == 'NARROW':
    return f'shape_narrow({args[0].expr}, {args[1].expr}, {args[2].expr}, {args[3].expr})'

  print('mk_shape_infer', shape)
  return 'nullopt'
  #exit()


def mk_strides_infer(fn, all_args, ret):
  args = [arg for arg in all_args if is_shape_arg(arg)]

  if fn == 'ALL []':
    return 'IntArrayRef()'
  if fn == 'ALL [0]':
    return 'IntArrayRef(0)'
  if fn == 'EQ_FIRST':
    return args[0].expr
  if fn == 'EQ_SECOND':
    return args[1].expr
  if fn == 'CONTIGUOUS':
    return f'strides_contiguous({ret})'
  if fn == 'STD_PROMOTE':
    args = [arg.expr for arg in all_args if 'Tensor' in arg.type.cpp_type() or 'at::IntArrayRef' in arg.type.cpp_type()]
    return f'strides_std_promote({ret}, {", ".join(args)})'
  if fn == 'VIEW':
    return f'strides_view({args[0].expr}, {ret})'
  if fn == 'TRANSPOSE':
    return f'strides_transpose({args[0].expr})'
  if fn == 'CLONE':
    return f'strides_clone({args[0].expr}, {args[1].expr})'
  if fn == 'CLONE1':
    return f'strides_clone({args[0].expr})'
  if fn == 'CLONE2':
    return f'strides_clone2({args[0].expr}, {args[2].expr}, {args[3].expr})'
  if fn == 'CLONE_BOOL':
    return f'strides_clone_bool({args[0].expr}, {args[1].expr})'
  if fn == 'PERMUTE':
    return f'strides_permute({args[0].expr}, {args[1].expr})'

  print('mk_strides_infer', fn)
  return 'nullopt'


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

  shape_fn = get_shape_infer_fn(fn)
  strides_fn = get_strides_infer_fn(fn)

  # emit pass-through wrapper for unsupported functions
  if skip_fn(fn):
    return f'''
{fndecl} {{
  stats_inc_unsupported_wrapper();
  {dispatchkey}
  return {redispatch};
}}'''

  # returns a tensor and takes tensors as arguments
  # e.g. add(x, y)
  if rettype == 'at::Tensor':
    dtype_device = get_dtype_arg(tensor_args, args, fn.func.name)

    set_shape = ''
    if shape_fn:
      set_shape = f'set_shape(tt, {mk_shape_infer(shape_fn, args)});\n  '

    set_strides = ''
    if strides_fn:
      set_strides = f'set_strides(tt, {mk_strides_infer(strides_fn, args, "tt")});\n  '

    return f'''
{fndecl} {{
  if (trace.is_flushing()) {{
    {dispatchkey}
    return {redispatch};
  }}
  auto tt = register_new_tensor(dispatchKeySet, {fn_enum(fn)}, {dtype_device});
  {set_shape}{set_strides}{register_args}
  return tt;
}}'''

  # in-place op. returns one of the arguments
  # e.g. mul_ or mul_out
  assert rettype == 'at::Tensor &' or rettype == 'const at::Tensor &'
  assert tensor_args
  ret = fn_output(fn)

  keeps_shape = 'false'
  if (shape_fn == 'EQ_FIRST' and len(tensor_args) >= 1 and tensor_args[0].expr == ret) or\
     (shape_fn == 'EQ_SECOND' and len(tensor_args) >= 2 and tensor_args[1].expr == ret) or\
     (shape_fn == 'EQ_THIRD' and len(tensor_args) >= 3 and tensor_args[2].expr == ret):
    keeps_shape = 'true'
  elif shape_fn:
    keeps_shape = f'eq_shapes({ret}, {mk_shape_infer(shape_fn, args)})'

  keeps_strides = 'false'
  if (strides_fn == 'EQ_FIRST' and len(tensor_args) >= 1 and tensor_args[0].expr == ret) or\
     (strides_fn == 'EQ_SECOND' and len(tensor_args) >= 2 and tensor_args[1].expr == ret) or\
     (strides_fn == 'EQ_THIRD' and len(tensor_args) >= 3 and tensor_args[2].expr == ret):
    keeps_strides = 'true'
  elif strides_fn:
    keeps_strides = f'eq_shapes({ret}, {mk_strides_infer(strides_fn, args, ret)})'

  return f'''
{fndecl} {{
  if (trace.is_flushing()) {{
    {dispatchkey}
    return {redispatch};
  }}
  bool flush = register_in_place({ret}, {fn_enum(fn)}, dispatchKeySet, {keeps_shape}, {keeps_strides});
  {register_args}
  if (flush)
    trace.flush(STATS(FlushReason::INPLACE_SHARED));
  else
    update_trace_idx({ret});
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
    args.append(f'load<{type}>()(op.args[{i}], load_state)')

  redispatch = f'<FN>(ks, {", ".join(args)})'
  rettype = dispatcher_sig.returns_type().cpp_type()

  if rettype == 'at::Tensor':
    code = f'results[i] = {redispatch};\n  break;'
    inplace = False

  # in-place op
  else:
    assert rettype == 'at::Tensor &' or rettype == 'const at::Tensor &'
    inplace = True
    code = f'results[i] = {redispatch};\n  break;'

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
fd7 = open('autogen/ops_data.h', 'w')

total = 0
for fn in native_functions.native_functions:
  total += 1
  print(gen_dispatch_wrapper(fn), file=fd1)
  print(gen_torch_library_table(fn), file=fd2)

  if skip_fn(fn):
    continue
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
    print(f'#define FIRST_INPLACE_OP {entries[0][0]}', file=fd7)

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
