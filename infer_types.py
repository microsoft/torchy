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

dtypes = [
  torch.bfloat16,
  torch.bool,
  torch.complex128,
  torch.float16,
  torch.float32,
  torch.float64,
  torch.uint8,
  torch.int8,
  torch.int16,
  torch.int32,
  torch.int64,
]

errors_seen = set()
def error(name, error):
  msg = f'{error} on {name}'
  if msg in errors_seen:
    return
  errors_seen.add(msg)
  print(msg)


def mk_value(type):
  type = str(type)
  if '?' in type:
    return None
  if type == 'bool':
    return False
  if type == 'int' or type == 'Scalar':
    return 0
  if type == 'float':
    return 0.0
  if type == 'Dimname' or type == 'str':
    return '"foo"'
  if type == 'Dimname[]':
    return ['"foo"']
  if 'int[' in type:
    return [0]
  print(type)
  assert False


def test_types(name, call, tensors, i):
  if i == 0:
    try:
      eval(call)
    except AttributeError:
      error(name, 'AttributeError')
    except IndexError:
      error(name, 'IndexError')
    except RuntimeError:
      error(name, 'RuntimeError')
    except TypeError:
      error(name, 'TypeError')
    return

  i -= 1
  for type in dtypes:
    tensors[i] = torch.zeros((1), dtype=type)
    test_types(name, call, tensors, i)

#TODO: support _out functions


@with_native_function
def infer(fn):
  args = []
  prefix = 'torch'
  tensor_idx = 0
  is_method = fn.variants == {Variant.method}

  if is_method:
    prefix = f'tensors[{tensor_idx}]'
    tensor_idx += 1

  for arg in fn.func.arguments.flat_positional:
    if is_method and arg == fn.func.arguments.self_arg.argument:
      continue
    if arg.default:
      break
    if str(arg.type) == 'Tensor':
      args.append(f'tensors[{tensor_idx}]')
      tensor_idx += 1
    elif str(arg.type) == 'Tensor[]':
      args.append(f'[tensors[{tensor_idx}]]')
      tensor_idx += 1
    else:
      args.append(str(mk_value(arg.type)))

  if fn.func.arguments.out:
    args.append(f'out=tensors[{tensor_idx}]')
    tensor_idx += 1
    # not useful; output type == out tensor type
    return

  if tensor_idx == 0:
    return

  call = f'{prefix}.{fn.func.name.name}({", ".join(args)})'
  test_types(fn.func.name, call, [0] * tensor_idx, tensor_idx)


for fn in native_functions.native_functions:
  if len(fn.func.returns) != 1 or str(fn.func.returns[0].type) != 'Tensor':
    continue

  infer(fn)
  print(f'Done {fn.func.name}')
