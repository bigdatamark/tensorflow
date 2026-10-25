# [MODULE DOC]
# Dispatch support that lets standard elementwise TensorFlow ops accept ragged tensors, plus a
# generator of documentation text listing ops that support ragged tensors. Import-time side
# effect: registers a unary elementwise dispatcher for ragged tensors with the dispatch
# framework. Also keeps a deprecated binary elementwise handler and private helpers for
# formatting op signatures used by the op-list generator.

# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Operator dispatch for RaggedTensors."""

from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_shape
from tensorflow.python.util import dispatch
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_export
from tensorflow.python.util import tf_inspect


@dispatch.dispatch_for_unary_elementwise_apis(ragged_tensor.Ragged)
def ragged_unary_elementwise_op(op, x):
  # [FUNCTION DOC]
  # Dispatch handler applying a unary elementwise op to a ragged tensor. Inputs: op (the
  # original unary op) and x (ragged tensor or convertible value). Converts x to a tensor or
  # ragged tensor, applies the op to its values, and returns a ragged tensor of the same
  # structure with the mapped values. Registered at import time via the dispatch decorator.
  """Unary elementwise api handler for RaggedTensors."""
  x = ragged_tensor.convert_to_tensor_or_ragged_tensor(x)
  return x.with_values(op(x.values))


# TODO(martinz): This is deprecated. Delete.
def ragged_binary_elementwise_op(op, x, y):
  # [FUNCTION DOC]
  # KNOWN ISSUE: marked deprecated by a TODO (to be deleted) but still present.
  # Applies a binary elementwise op where either or both operands may be ragged. Inputs: op
  # (callable on two tensors) and x, y (ragged tensors or values convertible to tensors; each
  # converts using the other's dtype as a preferred dtype). If both are ragged their row-splits
  # dtypes are matched. When broadcasting is needed (both ragged, or the ragged one's flat
  # values rank does not exceed the other's rank) both are broadcast to a common dynamic shape
  # without broadcasting inner dimensions. The op is applied to flat values; if it returns a
  # plain boolean (the tensor equality special case) that is returned directly, otherwise the
  # result is wrapped with the ragged structure of x (or y if x is not ragged).
  """Binary elementwise api handler for RaggedTensors."""
  x_is_ragged = ragged_tensor.is_ragged(x)
  y_is_ragged = ragged_tensor.is_ragged(y)

  # Convert args to tensors.
  x = ragged_tensor.convert_to_tensor_or_ragged_tensor(
      x, preferred_dtype=(y.dtype if y_is_ragged else None))
  y = ragged_tensor.convert_to_tensor_or_ragged_tensor(
      y, preferred_dtype=x.dtype)

  if x_is_ragged and y_is_ragged:
    x, y = ragged_tensor.match_row_splits_dtypes(x, y)

  # Perform broadcasting, when appropriate
  if ((x_is_ragged and y_is_ragged) or
      (x_is_ragged and x.flat_values.shape.ndims <= y.shape.ndims) or
      (y_is_ragged and y.flat_values.shape.ndims <= x.shape.ndims)):
    # If both x and y are ragged, they must have the same row_splits_dtype now.
    if x_is_ragged:
      dim_size_dtype = x.row_splits.dtype
    else:
      dim_size_dtype = y.row_splits.dtype

    shape_x = ragged_tensor_shape.RaggedTensorDynamicShape.from_tensor(
        x, dim_size_dtype=dim_size_dtype)
    shape_y = ragged_tensor_shape.RaggedTensorDynamicShape.from_tensor(
        y, dim_size_dtype=dim_size_dtype)
    bcast_shape = ragged_tensor_shape.broadcast_dynamic_shape(shape_x, shape_y)
    x = ragged_tensor_shape.broadcast_to(
        x, bcast_shape, broadcast_inner_dimensions=False)
    y = ragged_tensor_shape.broadcast_to(
        y, bcast_shape, broadcast_inner_dimensions=False)

  x_values = x.flat_values if ragged_tensor.is_ragged(x) else x
  y_values = y.flat_values if ragged_tensor.is_ragged(y) else y
  mapped_values = op(x_values, y_values)
  if isinstance(mapped_values, bool):
    return mapped_values  # Special case for tensor_equals.
  if ragged_tensor.is_ragged(x):
    return x.with_flat_values(mapped_values)
  else:
    return y.with_flat_values(mapped_values)


# TODO(edloper): Update the documentation generation tools to automatically
# build lists of which types are supported by which ops (and then delete all
# the following code).


# We don't need to register a separate delegation handler for these v1 ops,
# since they delegate to the v2 ops (which already have a handler).  But we
# still want to include them in the ragged_op_list() output.
_V2_OPS_THAT_ARE_DELEGATED_TO_FROM_V1_OPS = [
    math_ops.reduce_sum,
    math_ops.reduce_prod,
    math_ops.reduce_min,
    math_ops.reduce_max,
    math_ops.reduce_mean,
    math_ops.reduce_variance,
    math_ops.reduce_std,
    math_ops.reduce_any,
    math_ops.reduce_all,
    string_ops.string_to_number,
    string_ops.string_to_hash_bucket,
    string_ops.reduce_join_v2,
]


def _ragged_op_signature(op, ragged_args, ragged_varargs=False):
  # [FUNCTION DOC]
  # HELPER: formatting helper for ragged_op_list documentation output.
  # Builds a markdown bullet describing an op's public signature. Inputs: op (callable),
  # ragged_args (collection of positional indices whose arguments accept ragged tensors, shown
  # in bold) and ragged_varargs (whether the variable positional argument also accepts ragged
  # tensors, default false). Appends defaults, variable positional and keyword arguments.
  # Returns the formatted string using the op's canonical exported name.
  """Returns a signature for the given op, marking ragged args in bold."""
  op_name = tf_export.get_canonical_name_for_symbol(op)
  argspec = tf_inspect.getfullargspec(op)
  arg_names = argspec.args

  # Mark ragged arguments in bold.
  for pos in ragged_args:
    arg_names[pos] = '**' + arg_names[pos] + '**'

  # Add argument defaults.
  if argspec.defaults is not None:
    for pos in range(-1, -len(argspec.defaults) - 1, -1):
      arg_names[pos] += '=`{!r}`'.format(argspec.defaults[pos])

  # Add varargs and keyword args
  if argspec.varargs:
    if ragged_varargs:
      arg_names.append('***' + argspec.varargs + '**')
    else:
      arg_names.append('*' + argspec.varargs)
  if argspec.varkw:
    arg_names.append('**' + argspec.varkw)

  return '* `tf.{}`({})'.format(op_name, ', '.join(arg_names))


def _op_is_in_tf_version(op, version):
  # [FUNCTION DOC]
  # HELPER: filter for ragged_op_list. Checks whether an op is exported in a TensorFlow API
  # version. Inputs: op (callable) and version (1 or 2). For version 1 returns truthy if the
  # unwrapped op has version-1 export names or is in the list of version-2 ops that version-1
  # ops delegate to; for version 2 returns truthy if it has version-2 export names. Raises a
  # value error for any other version.
  if version == 1:
    return (tf_export.get_v1_names(tf_decorator.unwrap(op)[1]) or
            op in _V2_OPS_THAT_ARE_DELEGATED_TO_FROM_V1_OPS)
  elif version == 2:
    return tf_export.get_v2_names(tf_decorator.unwrap(op)[1])
  else:
    raise ValueError('Expected version 1 or 2.')


def ragged_op_list(tf_version=2):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the returned string ends with a literal letter n (missing backslash) rather
  # than a newline; the docstring also misspells dispatchers.
  # Produces markdown documentation of ops that dispatch on ragged tensors. Input: tf_version
  # (1 or 2, default 2) selecting which exported API names are included. Collects the
  # type-based dispatch signatures registered for ragged tensors, marks ragged argument
  # positions, keeps ops exported in the chosen version, and adds the print op with ragged
  # varargs. Returns a header plus the sorted bullet lines. Raises a value error for other
  # version numbers.
  """Returns a string listing operations that have dispathers registered."""
  lines = []
  api_signatures = dispatch.type_based_dispatch_signatures_for(
      ragged_tensor.RaggedTensor)
  for api, signatures in api_signatures.items():
    arg_names = tf_inspect.getargspec(api).args
    ragged_args = set()
    for signature in signatures:
      for arg in signature:
        ragged_args.add(arg if isinstance(arg, int) else arg_names.index(arg))
    if _op_is_in_tf_version(api, tf_version):
      lines.append(_ragged_op_signature(api, ragged_args))

  lines.append(
      _ragged_op_signature(logging_ops.print_v2, [], ragged_varargs=True))
  return ('\n\n### Additional ops that support `RaggedTensor`\n\n'
          'Arguments that accept `RaggedTensor`s are marked in **bold**.\n\n' +
          '\n'.join(sorted(lines)) + 'n')
