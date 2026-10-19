# [MODULE DOC]
# HELPER: helper module for convolution layers.
# Argument normalization and validation (data format, padding, tuples), output/input length
# arithmetic for regular and transposed convolutions, kernel connectivity masks and index
# generators for locally connected layers, and a batch-dimension squeeze wrapper. No import-time
# side effects.
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
# ==============================================================================
"""Utilities used by convolution layers."""

import itertools

import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.ops import array_ops


def convert_data_format(data_format, ndim):
  # [FUNCTION DOC]
  # HELPER: translates Keras data format names into the layout strings of TensorFlow ops.
  # Inputs: data_format, either channels-last or channels-first; ndim, tensor rank (3, 4 or 5).
  # Returns the matching layout string (width, height-width or depth-height-width layouts with
  # channels last or first). Raises a value error for a rank other than 3, 4, 5 or for an
  # unrecognized data format. No side effects.
  if data_format == 'channels_last':
    if ndim == 3:
      return 'NWC'
    elif ndim == 4:
      return 'NHWC'
    elif ndim == 5:
      return 'NDHWC'
    else:
      raise ValueError('Input rank not supported:', ndim)
  elif data_format == 'channels_first':
    if ndim == 3:
      return 'NCW'
    elif ndim == 4:
      return 'NCHW'
    elif ndim == 5:
      return 'NCDHW'
    else:
      raise ValueError('Input rank not supported:', ndim)
  else:
    raise ValueError('Invalid data_format:', data_format)


def normalize_tuple(value, n, name):
  # [FUNCTION DOC]
  # HELPER: argument validation for layer parameters such as strides and kernel size.
  # KNOWN ISSUE: the docstring says it returns a tuple of integers, but for an iterable the
  # elements are only checked to be convertible to integers and are returned unconverted.
  # A single integer is repeated n times. Otherwise value is turned into a tuple that must have
  # length n and elements convertible to integers. Inputs: value (integer or iterable), n (needed
  # length), name (argument name used only in messages). Raises a value error on a non-iterable,
  # wrong length or non-integer element. No side effects.
  """Transforms a single integer or iterable of integers into an integer tuple.

  Args:
    value: The value to validate and convert. Could an int, or any iterable of
      ints.
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
      raise ValueError('The `' + name + '` argument must be a tuple of ' +
                       str(n) + ' integers. Received: ' + str(value))
    if len(value_tuple) != n:
      raise ValueError('The `' + name + '` argument must be a tuple of ' +
                       str(n) + ' integers. Received: ' + str(value))
    for single_value in value_tuple:
      try:
        int(single_value)
      except (ValueError, TypeError):
        raise ValueError('The `' + name + '` argument must be a tuple of ' +
                         str(n) + ' integers. Received: ' + str(value) + ' '
                         'including element ' + str(single_value) + ' of type' +
                         ' ' + str(type(single_value)))
    return value_tuple


def conv_output_length(input_length, filter_size, padding, stride, dilation=1):
  # [FUNCTION DOC]
  # HELPER: shape arithmetic for convolution layers.
  # Computes the output length along one dimension. Inputs: input_length (integer or nothing),
  # filter_size, padding (same, valid, full or causal), stride, dilation (default 1). Returns
  # nothing when input_length is nothing; otherwise the dilated-filter output length divided by
  # stride rounded up. Asserts (assertion error, not a value error) that padding is allowed.
  """Determines output length of a convolution given input length.

  Args:
      input_length: integer.
      filter_size: integer.
      padding: one of "same", "valid", "full", "causal"
      stride: integer.
      dilation: dilation rate, integer.

  Returns:
      The output length (integer).
  """
  if input_length is None:
    return None
  assert padding in {'same', 'valid', 'full', 'causal'}
  dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
  if padding in ['same', 'causal']:
    output_length = input_length
  elif padding == 'valid':
    output_length = input_length - dilated_filter_size + 1
  elif padding == 'full':
    output_length = input_length + dilated_filter_size - 1
  return (output_length + stride - 1) // stride


def conv_input_length(output_length, filter_size, padding, stride):
  # [FUNCTION DOC]
  # HELPER: shape arithmetic for convolution layers (inverse direction).
  # Computes the input length that would produce the given output length. Inputs: output_length
  # (integer or nothing), filter_size, padding (same, valid or full), stride. Returns nothing if
  # output_length is nothing. Asserts the padding value is allowed. No side effects.
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


def deconv_output_length(input_length,
                         filter_size,
                         padding,
                         output_padding=None,
                         stride=0,
                         dilation=1):
  # [FUNCTION DOC]
  # HELPER: shape arithmetic for transposed convolution layers.
  # KNOWN ISSUE: stride defaults to 0, which is not a meaningful stride and yields a wrong
  # length; callers must always pass it. Padding is asserted before the nothing check.
  # Computes the output length of a transposed convolution along one dimension. Inputs:
  # input_length (integer or nothing), filter_size, padding (same, valid or full), output_padding
  # (optional; nothing means infer the length), stride, dilation. Returns nothing if input_length
  # is nothing. Asserts the padding value is allowed. No side effects.
  """Determines output length of a transposed convolution given input length.

  Args:
      input_length: Integer.
      filter_size: Integer.
      padding: one of `"same"`, `"valid"`, `"full"`.
      output_padding: Integer, amount of padding along the output dimension. Can
        be set to `None` in which case the output length is inferred.
      stride: Integer.
      dilation: Integer.

  Returns:
      The output length (integer).
  """
  assert padding in {'same', 'valid', 'full'}
  if input_length is None:
    return None

  # Get the dilated kernel size
  filter_size = filter_size + (filter_size - 1) * (dilation - 1)

  # Infer length if output padding is None, else compute the exact length
  if output_padding is None:
    if padding == 'valid':
      length = input_length * stride + max(filter_size - stride, 0)
    elif padding == 'full':
      length = input_length * stride - (stride + filter_size - 2)
    elif padding == 'same':
      length = input_length * stride

  else:
    if padding == 'same':
      pad = filter_size // 2
    elif padding == 'valid':
      pad = 0
    elif padding == 'full':
      pad = filter_size - 1

    length = ((input_length - 1) * stride + filter_size - 2 * pad +
              output_padding)
  return length


def normalize_data_format(value):
  # [FUNCTION DOC]
  # HELPER: argument validation for layers taking a data format.
  # Input: value, a string or nothing; nothing means use the Keras backend default image data
  # format. Returns the lowercase format, which must be channels-first or channels-last; raises a
  # value error otherwise. Reads backend global configuration; no writes.
  if value is None:
    value = backend.image_data_format()
  data_format = value.lower()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('The `data_format` argument must be one of '
                     '"channels_first", "channels_last". Received: ' +
                     str(value))
  return data_format


def normalize_padding(value):
  # [FUNCTION DOC]
  # HELPER: argument validation for layers taking a padding mode.
  # A list or tuple is returned unchanged (explicit padding amounts, unvalidated). A string is
  # lowercased and must be valid, same or causal, else a value error is raised. Returns the
  # normalized value. No side effects.
  if isinstance(value, (list, tuple)):
    return value
  padding = value.lower()
  if padding not in {'valid', 'same', 'causal'}:
    raise ValueError('The `padding` argument must be a list/tuple or one of '
                     '"valid", "same" (or "causal", only for `Conv1D). '
                     'Received: ' + str(padding))
  return padding


def conv_kernel_mask(input_shape, kernel_shape, strides, padding):
  # [FUNCTION DOC]
  # HELPER: builds connectivity masks for locally connected layers.
  # Creates a boolean array whose axes are the input spatial dims followed by the output spatial
  # dims; an entry is true when that input location feeds that output location. Inputs:
  # input_shape tuple; kernel_shape and strides, tuples or single integers expanded to every
  # dimension; padding, same or valid. Raises not-implemented for other padding and a value error
  # when dimension counts differ. Loops over every output position, so it is slow for large shapes.
  """Compute a mask representing the connectivity of a convolution operation.

  Assume a convolution with given parameters is applied to an input having N
  spatial dimensions with `input_shape = (d_in1, ..., d_inN)` to produce an
  output with shape `(d_out1, ..., d_outN)`. This method returns a boolean array
  of shape `(d_in1, ..., d_inN, d_out1, ..., d_outN)` with `True` entries
  indicating pairs of input and output locations that are connected by a weight.

  Example:

    >>> input_shape = (4,)
    >>> kernel_shape = (2,)
    >>> strides = (1,)
    >>> padding = "valid"
    >>> conv_kernel_mask(input_shape, kernel_shape, strides, padding)
    array([[ True, False, False],
           [ True,  True, False],
           [False,  True,  True],
           [False, False,  True]])

    where rows and columns correspond to inputs and outputs respectively.


  Args:
    input_shape: tuple of size N: `(d_in1, ..., d_inN)`, spatial shape of the
      input.
    kernel_shape: tuple of size N, spatial shape of the convolutional kernel /
      receptive field.
    strides: tuple of size N, strides along each spatial dimension.
    padding: type of padding, string `"same"` or `"valid"`.
      `"valid"` means no padding. `"same"` results in padding evenly to
      the left/right or up/down of the input such that output has the same
      height/width dimension as the input.

  Returns:
    A boolean 2N-D `np.ndarray` of shape
    `(d_in1, ..., d_inN, d_out1, ..., d_outN)`, where `(d_out1, ..., d_outN)`
    is the spatial shape of the output. `True` entries in the mask represent
    pairs of input-output locations that are connected by a weight.

  Raises:
    ValueError: if `input_shape`, `kernel_shape` and `strides` don't have the
        same number of dimensions.
    NotImplementedError: if `padding` is not in {`"same"`, `"valid"`}.
  """
  if padding not in {'same', 'valid'}:
    raise NotImplementedError('Padding type %s not supported. '
                              'Only "valid" and "same" '
                              'are implemented.' % padding)

  in_dims = len(input_shape)
  if isinstance(kernel_shape, int):
    kernel_shape = (kernel_shape,) * in_dims
  if isinstance(strides, int):
    strides = (strides,) * in_dims

  kernel_dims = len(kernel_shape)
  stride_dims = len(strides)
  if kernel_dims != in_dims or stride_dims != in_dims:
    raise ValueError('Number of strides, input and kernel dimensions must all '
                     'match. Received: %d, %d, %d.' %
                     (stride_dims, in_dims, kernel_dims))

  output_shape = conv_output_shape(input_shape, kernel_shape, strides, padding)

  mask_shape = input_shape + output_shape
  mask = np.zeros(mask_shape, np.bool_)

  output_axes_ticks = [range(dim) for dim in output_shape]
  for output_position in itertools.product(*output_axes_ticks):
    input_axes_ticks = conv_connected_inputs(input_shape, kernel_shape,
                                             output_position, strides, padding)
    for input_position in itertools.product(*input_axes_ticks):
      mask[input_position + output_position] = True

  return mask


def conv_kernel_idxs(input_shape, kernel_shape, strides, padding, filters_in,
                     filters_out, data_format):
  # [FUNCTION DOC]
  # HELPER: index generator for sparse-kernel construction in locally connected layers.
  # Generator yielding pairs of flattened output index and flattened input index for every input
  # and output position connected through the kernel, crossed with all input and output filters.
  # Inputs: input_shape, kernel_shape, strides (integers are expanded), padding (same or valid),
  # filters_in, filters_out, data_format (channels-first or channels-last, which controls where
  # the filter axis goes). Errors are raised lazily on first iteration: not-implemented for other
  # padding, value error for mismatched dimension counts or unknown data format.
  """Yields output-input tuples of indices in a CNN layer.

  The generator iterates over all `(output_idx, input_idx)` tuples, where
    `output_idx` is an integer index in a flattened tensor representing a single
    output image of a convolutional layer that is connected (via the layer
    weights) to the respective single input image at `input_idx`

  Example:

    >>> input_shape = (2, 2)
    >>> kernel_shape = (2, 1)
    >>> strides = (1, 1)
    >>> padding = "valid"
    >>> filters_in = 1
    >>> filters_out = 1
    >>> data_format = "channels_last"
    >>> list(conv_kernel_idxs(input_shape, kernel_shape, strides, padding,
    ...                       filters_in, filters_out, data_format))
    [(0, 0), (0, 2), (1, 1), (1, 3)]

  Args:
    input_shape: tuple of size N: `(d_in1, ..., d_inN)`, spatial shape of the
      input.
    kernel_shape: tuple of size N, spatial shape of the convolutional kernel /
      receptive field.
    strides: tuple of size N, strides along each spatial dimension.
    padding: type of padding, string `"same"` or `"valid"`.
      `"valid"` means no padding. `"same"` results in padding evenly to
      the left/right or up/down of the input such that output has the same
      height/width dimension as the input.
    filters_in: `int`, number if filters in the input to the layer.
    filters_out: `int', number if filters in the output of the layer.
    data_format: string, "channels_first" or "channels_last".

  Yields:
    The next tuple `(output_idx, input_idx)`, where
    `output_idx` is an integer index in a flattened tensor representing a single
    output image of a convolutional layer that is connected (via the layer
    weights) to the respective single input image at `input_idx`.

  Raises:
      ValueError: if `data_format` is neither
      `"channels_last"` nor `"channels_first"`, or if number of strides, input,
      and kernel number of dimensions do not match.

      NotImplementedError: if `padding` is neither `"same"` nor `"valid"`.
  """
  if padding not in ('same', 'valid'):
    raise NotImplementedError('Padding type %s not supported. '
                              'Only "valid" and "same" '
                              'are implemented.' % padding)

  in_dims = len(input_shape)
  if isinstance(kernel_shape, int):
    kernel_shape = (kernel_shape,) * in_dims
  if isinstance(strides, int):
    strides = (strides,) * in_dims

  kernel_dims = len(kernel_shape)
  stride_dims = len(strides)
  if kernel_dims != in_dims or stride_dims != in_dims:
    raise ValueError('Number of strides, input and kernel dimensions must all '
                     'match. Received: %d, %d, %d.' %
                     (stride_dims, in_dims, kernel_dims))

  output_shape = conv_output_shape(input_shape, kernel_shape, strides, padding)
  output_axes_ticks = [range(dim) for dim in output_shape]

  if data_format == 'channels_first':
    concat_idxs = lambda spatial_idx, filter_idx: (filter_idx,) + spatial_idx
  elif data_format == 'channels_last':
    concat_idxs = lambda spatial_idx, filter_idx: spatial_idx + (filter_idx,)
  else:
    raise ValueError('Data format %s not recognized.'
                     '`data_format` must be "channels_first" or '
                     '"channels_last".' % data_format)

  for output_position in itertools.product(*output_axes_ticks):
    input_axes_ticks = conv_connected_inputs(input_shape, kernel_shape,
                                             output_position, strides, padding)
    for input_position in itertools.product(*input_axes_ticks):
      for f_in in range(filters_in):
        for f_out in range(filters_out):
          out_idx = np.ravel_multi_index(
              multi_index=concat_idxs(output_position, f_out),
              dims=concat_idxs(output_shape, filters_out))
          in_idx = np.ravel_multi_index(
              multi_index=concat_idxs(input_position, f_in),
              dims=concat_idxs(input_shape, filters_in))
          yield (out_idx, in_idx)


def conv_connected_inputs(input_shape, kernel_shape, output_position, strides,
                          padding):
  # [FUNCTION DOC]
  # HELPER: shared by conv_kernel_mask and conv_kernel_idxs.
  # KNOWN ISSUE: any padding other than valid is treated like same (no validation).
  # For one output position, computes per spatial dimension the range of input indices covered by
  # the kernel, clipped to the input bounds. Inputs: input_shape, kernel_shape, output_position,
  # strides (tuples of equal length), padding. Returns a list of range objects, one per dimension.
  """Return locations of the input connected to an output position.

  Assume a convolution with given parameters is applied to an input having N
  spatial dimensions with `input_shape = (d_in1, ..., d_inN)`. This method
  returns N ranges specifying the input region that was convolved with the
  kernel to produce the output at position
  `output_position = (p_out1, ..., p_outN)`.

  Example:

    >>> input_shape = (4, 4)
    >>> kernel_shape = (2, 1)
    >>> output_position = (1, 1)
    >>> strides = (1, 1)
    >>> padding = "valid"
    >>> conv_connected_inputs(input_shape, kernel_shape, output_position,
    ...                       strides, padding)
    [range(1, 3), range(1, 2)]

  Args:
    input_shape: tuple of size N: `(d_in1, ..., d_inN)`, spatial shape of the
      input.
    kernel_shape: tuple of size N, spatial shape of the convolutional kernel /
      receptive field.
    output_position: tuple of size N: `(p_out1, ..., p_outN)`, a single position
      in the output of the convolution.
    strides: tuple of size N, strides along each spatial dimension.
    padding: type of padding, string `"same"` or `"valid"`.
      `"valid"` means no padding. `"same"` results in padding evenly to
      the left/right or up/down of the input such that output has the same
      height/width dimension as the input.

  Returns:
    N ranges `[[p_in_left1, ..., p_in_right1], ...,
              [p_in_leftN, ..., p_in_rightN]]` specifying the region in the
    input connected to output_position.
  """
  ranges = []

  ndims = len(input_shape)
  for d in range(ndims):
    left_shift = int(kernel_shape[d] / 2)
    right_shift = kernel_shape[d] - left_shift

    center = output_position[d] * strides[d]

    if padding == 'valid':
      center += left_shift

    start = max(0, center - left_shift)
    end = min(input_shape[d], center + right_shift)

    ranges.append(range(start, end))

  return ranges


def conv_output_shape(input_shape, kernel_shape, strides, padding):
  # [FUNCTION DOC]
  # HELPER: shape arithmetic used by the mask and index helpers.
  # Applies the per-dimension output length computation across all spatial dimensions, then forces
  # any dimension whose input size is 0 to stay 0. Inputs: input_shape, kernel_shape, strides
  # (tuples) and padding. Returns a tuple of output sizes. Dimension count follows kernel_shape.
  """Return the output shape of an N-D convolution.

  Forces dimensions where input is empty (size 0) to remain empty.

  Args:
    input_shape: tuple of size N: `(d_in1, ..., d_inN)`, spatial shape of the
      input.
    kernel_shape: tuple of size N, spatial shape of the convolutional kernel /
      receptive field.
    strides: tuple of size N, strides along each spatial dimension.
    padding: type of padding, string `"same"` or `"valid"`.
      `"valid"` means no padding. `"same"` results in padding evenly to
      the left/right or up/down of the input such that output has the same
      height/width dimension as the input.

  Returns:
    tuple of size N: `(d_out1, ..., d_outN)`, spatial shape of the output.
  """
  dims = range(len(kernel_shape))
  output_shape = [
      conv_output_length(input_shape[d], kernel_shape[d], padding, strides[d])
      for d in dims
  ]
  output_shape = tuple(
      [0 if input_shape[d] == 0 else output_shape[d] for d in dims])
  return output_shape


def squeeze_batch_dims(inp, op, inner_rank):
  # [FUNCTION DOC]
  # HELPER: lets convolution ops that accept one batch dimension handle several.
  # Flattens all leading batch dimensions of inp into one, applies op, then restores the batch
  # dimensions on the result. Uses static shapes where fully known, dynamic shape tensors
  # otherwise, and sets the static shape of the output. Inputs: inp, tensor whose last inner_rank
  # dims are the op's input dims; op, callable from one tensor to one tensor; inner_rank, integer.
  # Returns the reshaped output tensor. Creates graph ops inside a name scope.
  """Returns `unsqueeze_batch(op(squeeze_batch(inp)))`.

  Where `squeeze_batch` reshapes `inp` to shape
  `[prod(inp.shape[:-inner_rank])] + inp.shape[-inner_rank:]`
  and `unsqueeze_batch` does the reverse reshape but on the output.

  Args:
    inp: A tensor with dims `batch_shape + inner_shape` where `inner_shape`
      is length `inner_rank`.
    op: A callable that takes a single input tensor and returns a single.
      output tensor.
    inner_rank: A python integer.

  Returns:
    `unsqueeze_batch_op(squeeze_batch(inp))`.
  """
  with ops.name_scope_v2('squeeze_batch_dims'):
    shape = inp.shape

    inner_shape = shape[-inner_rank:]
    if not inner_shape.is_fully_defined():
      inner_shape = array_ops.shape(inp)[-inner_rank:]

    batch_shape = shape[:-inner_rank]
    if not batch_shape.is_fully_defined():
      batch_shape = array_ops.shape(inp)[:-inner_rank]

    if isinstance(inner_shape, tensor_shape.TensorShape):
      inp_reshaped = array_ops.reshape(inp, [-1] + inner_shape.as_list())
    else:
      inp_reshaped = array_ops.reshape(
          inp, array_ops.concat(([-1], inner_shape), axis=-1))

    out_reshaped = op(inp_reshaped)

    out_inner_shape = out_reshaped.shape[-inner_rank:]
    if not out_inner_shape.is_fully_defined():
      out_inner_shape = array_ops.shape(out_reshaped)[-inner_rank:]

    out = array_ops.reshape(
        out_reshaped, array_ops.concat((batch_shape, out_inner_shape), axis=-1))

    out.set_shape(inp.shape[:-inner_rank] + out.shape[-inner_rank:])
    return out
