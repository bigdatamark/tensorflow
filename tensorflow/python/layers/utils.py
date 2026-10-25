# [MODULE DOC]
# HELPER: shared validation and shape arithmetic utilities for the layers package.
# Provides conversion of a channels-first/last format name into a rank-specific layout string,
# normalization and validation of tuple, data format and padding arguments, computation of output
# or input lengths for convolutions and transposed convolutions, and conditional helpers that
# evaluate one of two callables depending on a possibly constant boolean predicate.
# No import-time side effects beyond imports.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================

"""Contains layer utilities for input validation and format conversion."""
from tensorflow.python.framework import smart_cond as smart_module
from tensorflow.python.ops import cond
from tensorflow.python.ops import variables


def convert_data_format(data_format, ndim):
  # [FUNCTION DOC]
  # Maps a data format name and tensor rank to the layout string used by TensorFlow ops.
  # Inputs: data_format, either channels_last or channels_first; ndim, the tensor rank (3, 4 or 5).
  # Returns NWC, NHWC or NDHWC for channels_last and NCW, NCHW or NCDHW for channels_first.
  # Raises a value error if the rank is not 3, 4 or 5, or if the format name is neither of the two.
  # No side effects.
  if data_format == 'channels_last':
    if ndim == 3:
      return 'NWC'
    elif ndim == 4:
      return 'NHWC'
    elif ndim == 5:
      return 'NDHWC'
    else:
      raise ValueError(f'Input rank: {ndim} not supported. We only support '
                       'input rank 3, 4 or 5.')
  elif data_format == 'channels_first':
    if ndim == 3:
      return 'NCW'
    elif ndim == 4:
      return 'NCHW'
    elif ndim == 5:
      return 'NCDHW'
    else:
      raise ValueError(f'Input rank: {ndim} not supported. We only support '
                       'input rank 3, 4 or 5.')
  else:
    raise ValueError(f'Invalid data_format: {data_format}. We only support '
                     '"channels_first" or "channels_last"')


def normalize_tuple(value, n, name):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the integer check only tests convertibility, so non-integer numbers such as floats
  # are accepted and returned unchanged, although the docstring promises a tuple of integers.
  # Validates an argument and turns it into a tuple of integers of a fixed length.
  # Inputs: value, a single integer (repeated n times) or an iterable of n integer-convertible items;
  # n, the required tuple size; name, the argument name used only in error messages.
  # Returns the resulting tuple. Raises a value error if value is not iterable, has the wrong length,
  # or contains an element that cannot be converted to an integer.
  """Transforms a single integer or iterable of integers into an integer tuple.

  Args:
    value: The value to validate and convert. Could an int, or any iterable
      of ints.
    n: The size of the tuple to be returned.
    name: The name of the argument being validated, e.g. "strides" or
      "kernel_size". This is only used to format error messages.

  Returns:
    A tuple of n integers.

  Raises:
    ValueError: If something else than an int/long or iterable thereof was
      passed.
  """
  if isinstance(value, int):
    return (value,) * n
  else:
    try:
      value_tuple = tuple(value)
    except TypeError:
      raise ValueError(f'Argument `{name}` must be a tuple of {str(n)} '
                       f'integers. Received: {str(value)}')
    if len(value_tuple) != n:
      raise ValueError(f'Argument `{name}` must be a tuple of {str(n)} '
                       f'integers. Received: {str(value)}')
    for single_value in value_tuple:
      try:
        int(single_value)
      except (ValueError, TypeError):
        raise ValueError(f'Argument `{name}` must be a tuple of {str(n)} '
                         f'integers. Received: {str(value)} including element '
                         f'{str(single_value)} of type '
                         f'{str(type(single_value))}')
    return value_tuple


def normalize_data_format(value):
  # [FUNCTION DOC]
  # Lowercases a data format string and validates it. Input: value, a string. Returns the lowercase
  # form, which is channels_first or channels_last. Raises a value error for any other string.
  # A non-string input fails with an attribute error from the lowercase call.
  data_format = value.lower()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('The `data_format` argument must be one of '
                     '"channels_first", "channels_last". Received: '
                     f'{str(value)}.')
  return data_format


def normalize_padding(value):
  # [FUNCTION DOC]
  # Lowercases a padding string and validates it. Input: value, a string. Returns valid or same.
  # Raises a value error for any other string. A non-string input fails with an attribute error.
  padding = value.lower()
  if padding not in {'valid', 'same'}:
    raise ValueError('The `padding` argument must be one of "valid", "same". '
                     f'Received: {str(padding)}.')
  return padding


def conv_output_length(input_length, filter_size, padding, stride, dilation=1):
  # [FUNCTION DOC]
  # Computes the output length of a convolution along one dimension. Inputs: input_length, integer
  # or nothing; filter_size; padding, one of same, valid, full; stride; dilation, default 1.
  # Returns nothing if input_length is nothing; otherwise the dilated-filter output length divided
  # by the stride, rounded up. Uses an assertion (an assertion error, not a value error) for an
  # unsupported padding string. No side effects.
  """Determines output length of a convolution given input length.

  Args:
      input_length: integer.
      filter_size: integer.
      padding: one of "same", "valid", "full".
      stride: integer.
      dilation: dilation rate, integer.

  Returns:
      The output length (integer).
  """
  if input_length is None:
    return None
  assert padding in {'same', 'valid', 'full'}
  dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
  if padding == 'same':
    output_length = input_length
  elif padding == 'valid':
    output_length = input_length - dilated_filter_size + 1
  elif padding == 'full':
    output_length = input_length + dilated_filter_size - 1
  return (output_length + stride - 1) // stride


def conv_input_length(output_length, filter_size, padding, stride):
  # [FUNCTION DOC]
  # Computes the input length that a convolution would need to produce a given output length.
  # Inputs: output_length (integer or nothing), filter_size, padding (same, valid or full), stride.
  # Returns nothing if output_length is nothing, else the input length. Assertion error on a bad
  # padding string. No side effects.
  """Determines input length of a convolution given output length.

  Args:
      output_length: integer.
      filter_size: integer.
      padding: one of "same", "valid", "full".
      stride: integer.

  Returns:
      The input length (integer).
  """
  if output_length is None:
    return None
  assert padding in {'same', 'valid', 'full'}
  if padding == 'same':
    pad = filter_size // 2
  elif padding == 'valid':
    pad = 0
  elif padding == 'full':
    pad = filter_size - 1
  return (output_length - 1) * stride - 2 * pad + filter_size


def deconv_output_length(input_length, filter_size, padding, stride):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says padding is one of same, valid, full, but no validation is done:
  # any other string silently behaves like same padding.
  # Computes the output length of a transposed convolution along one dimension. Inputs: input_length
  # (integer or nothing), filter_size, padding, stride. Returns nothing if input_length is nothing;
  # otherwise the length scaled by stride, plus the filter overhang for valid padding or minus the
  # overlap for full padding.
  """Determines output length of a transposed convolution given input length.

  Args:
      input_length: integer.
      filter_size: integer.
      padding: one of "same", "valid", "full".
      stride: integer.

  Returns:
      The output length (integer).
  """
  if input_length is None:
    return None
  input_length *= stride
  if padding == 'valid':
    input_length += max(filter_size - stride, 0)
  elif padding == 'full':
    input_length -= (stride + filter_size - 2)
  return input_length


def smart_cond(pred, true_fn=None, false_fn=None, name=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: true_fn and false_fn default to nothing, so calling without them fails later.
  # Chooses between two callables by a predicate. Inputs: pred, a boolean, tensor or variable;
  # true_fn and false_fn, callables; name, optional name for the conditional op. If pred is a
  # variable, always builds a graph conditional op. Otherwise delegates to the framework smart
  # conditional, which calls just one callable when pred is a constant and builds a conditional op
  # when it is dynamic. Returns the chosen callable's result. Raises a type error (from the
  # delegate) if a callable is not callable.
  """Return either `true_fn()` if predicate `pred` is true else `false_fn()`.

  If `pred` is a bool or has a constant value, we return either `true_fn()`
  or `false_fn()`, otherwise we use `tf.cond` to dynamically route to both.

  Args:
    pred: A scalar determining whether to return the result of `true_fn` or
      `false_fn`.
    true_fn: The callable to be performed if pred is true.
    false_fn: The callable to be performed if pred is false.
    name: Optional name prefix when using `tf.cond`.

  Returns:
    Tensors returned by the call to either `true_fn` or `false_fn`.

  Raises:
    TypeError: If `true_fn` or `false_fn` is not callable.
  """
  if isinstance(pred, variables.Variable):
    return cond.cond(
        pred, true_fn=true_fn, false_fn=false_fn, name=name)
  return smart_module.smart_cond(
      pred, true_fn=true_fn, false_fn=false_fn, name=name)


def constant_value(pred):
  # [FUNCTION DOC]
  # Returns the boolean value of a predicate when it is statically known. Input: pred, a boolean,
  # the integer 1 or 0 (converted to true or false), a tensor or a variable. Returns true or false
  # for constants, and nothing for variables or tensors with dynamic values. Raises a type error
  # (from the delegate) for unsupported types.
  """Return the bool value for `pred`, or None if `pred` had a dynamic value.

    Args:
      pred: A scalar, either a Python bool or a TensorFlow boolean variable
        or tensor, or the Python integer 1 or 0.

    Returns:
      True or False if `pred` has a constant boolean value, None otherwise.

    Raises:
      TypeError: If `pred` is not a Variable, Tensor or bool, or Python
        integer 1 or 0.
    """
  # Allow integer booleans.
  if isinstance(pred, int):
    if pred == 1:
      pred = True
    elif pred == 0:
      pred = False

  if isinstance(pred, variables.Variable):
    return None
  return smart_module.smart_constant_value(pred)
