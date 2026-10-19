# [MODULE DOC]
# Merge layers of the legacy Keras layers package: layers that combine a list of tensors into one
# (Add, Subtract, Multiply, Average, Maximum, Minimum, Concatenate, Dot), a private shared base
# class handling broadcasting and masking, and lowercase functional wrappers that build the layer
# and immediately apply it to the inputs. No import-time side effects.
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
# pylint: disable=not-callable
# pylint: disable=redefined-builtin
"""Layers that can merge several inputs into one."""

from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn


class _Merge(Layer):
  # [CLASS DOC]
  # HELPER: private base class shared by all merge layers in this module.
  # Base layer for element-wise merges of a list of two or more tensors. Validates shapes in
  # build, aligns inputs of different rank (by expanding or transposing dimensions) in call, computes
  # the broadcast output shape and combines masks with a logical and. Subclasses implement the
  # merge function. State: a flag recording whether inputs need reshaping to match ranks.
  """Generic merge layer for elementwise merge functions.

  Used to implement `Sum`, `Average`, etc.
  """

  def __init__(self, **kwargs):
    # [FUNCTION DOC]
    # Constructor: forwards standard layer keyword arguments to the base layer and enables mask
    # support. Returns nothing.
    """Initializes a Merge layer.

    Args:
      **kwargs: standard layer keyword arguments.
    """
    super(_Merge, self).__init__(**kwargs)
    self.supports_masking = True

  def _merge_function(self, inputs):
    # [FUNCTION DOC]
    # HELPER: abstract hook that subclasses override to combine the (rank-aligned) input tensors.
    # The base version always raises a not-implemented error.
    raise NotImplementedError

  def _compute_elemwise_op_output_shape(self, shape1, shape2):
    # [FUNCTION DOC]
    # HELPER: shape computation for build and compute_output_shape.
    # Broadcasts two shapes (tuples without batch dimension, or null if unknown). Returns null if
    # either is null. If the first is shorter the arguments are swapped. Leading dimensions of the
    # longer shape are kept; trailing ones are matched pairwise: unknown stays unknown, a size of
    # one takes the other size, equal sizes are kept. Raises a value error on incompatible sizes.
    # Returns a tuple. No side effects.
    """Computes the shape of the resultant of an elementwise operation.

    Args:
        shape1: tuple or None. Shape of the first tensor
        shape2: tuple or None. Shape of the second tensor

    Returns:
        expected output shape when an element-wise operation is
        carried out on 2 tensors with shapes shape1 and shape2.
        tuple or None.

    Raises:
        ValueError: if shape1 and shape2 are not compatible for
            element-wise operations.
    """
    if None in [shape1, shape2]:
      return None
    elif len(shape1) < len(shape2):
      return self._compute_elemwise_op_output_shape(shape2, shape1)
    elif not shape2:
      return shape1
    output_shape = list(shape1[:-len(shape2)])
    for i, j in zip(shape1[-len(shape2):], shape2):
      if i is None or j is None:
        output_shape.append(None)
      elif i == 1:
        output_shape.append(j)
      elif j == 1:
        output_shape.append(i)
      else:
        if i != j:
          raise ValueError(
              'Operands could not be broadcast '
              'together with shapes ' + str(shape1) + ' ' + str(shape2))
        output_shape.append(i)
    return tuple(output_shape)

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    # [FUNCTION DOC]
    # Validates the list of input shapes (nothing is created). Raises a value error if the shapes
    # are not a list of tuples, if fewer than two inputs are given, or if the known batch sizes
    # differ. Folds all shapes through the broadcast computation (raising on incompatibility) and
    # records on the layer whether inputs have differing ranks (or unknown shapes), so that call
    # must reshape them. Returns nothing.
    # Used purely for shape validation.
    if not isinstance(input_shape[0], tuple):
      raise ValueError('A merge layer should be called on a list of inputs.')
    if len(input_shape) < 2:
      raise ValueError('A merge layer should be called '
                       'on a list of at least 2 inputs. '
                       'Got ' + str(len(input_shape)) + ' inputs.')
    batch_sizes = {s[0] for s in input_shape if s} - {None}
    if len(batch_sizes) > 1:
      raise ValueError(
          'Can not merge tensors with different '
          'batch sizes. Got tensors with shapes : ' + str(input_shape))
    if input_shape[0] is None:
      output_shape = None
    else:
      output_shape = input_shape[0][1:]
    for i in range(1, len(input_shape)):
      if input_shape[i] is None:
        shape = None
      else:
        shape = input_shape[i][1:]
      output_shape = self._compute_elemwise_op_output_shape(output_shape, shape)
    # If the inputs have different ranks, we have to reshape them
    # to make them broadcastable.
    if None not in input_shape and len(set(map(len, input_shape))) == 1:
      self._reshape_required = False
    else:
      self._reshape_required = True

  def call(self, inputs):
    # [FUNCTION DOC]
    # Forward pass: merges a list or tuple of tensors (raises a value error for any other type).
    # If no reshape is required, applies the merge function directly. Otherwise, if all ranks are
    # known, lower-rank inputs get extra axes inserted after the batch axis until ranks match.
    # If some rank is unknown, inputs are transposed so the batch axis is last (using dynamic
    # shapes when needed), merged, and the result is transposed back so batch is first again.
    # Returns the merged tensor.
    if not isinstance(inputs, (list, tuple)):
      raise ValueError('A merge layer should be called on a list of inputs.')
    if self._reshape_required:
      reshaped_inputs = []
      input_ndims = list(map(backend.ndim, inputs))
      if None not in input_ndims:
        # If ranks of all inputs are available,
        # we simply expand each of them at axis=1
        # until all of them have the same rank.
        max_ndim = max(input_ndims)
        for x in inputs:
          x_ndim = backend.ndim(x)
          for _ in range(max_ndim - x_ndim):
            x = array_ops.expand_dims(x, axis=1)
          reshaped_inputs.append(x)
        return self._merge_function(reshaped_inputs)
      else:
        # Transpose all inputs so that batch size is the last dimension.
        # (batch_size, dim1, dim2, ... ) -> (dim1, dim2, ... , batch_size)
        transposed = False
        for x in inputs:
          x_ndim = backend.ndim(x)
          if x_ndim is None:
            x_shape = array_ops.shape(x)
            batch_size = x_shape[0]
            new_shape = backend.concatenate(
                [x_shape[1:],
                 array_ops.expand_dims(batch_size, axis=-1)])
            x_transposed = array_ops.reshape(
                x,
                array_ops_stack.stack(
                    [batch_size, math_ops.reduce_prod(x_shape[1:])], axis=0))
            x_transposed = array_ops.transpose(x_transposed, perm=(1, 0))
            x_transposed = array_ops.reshape(x_transposed, new_shape)
            reshaped_inputs.append(x_transposed)
            transposed = True
          elif x_ndim > 1:
            dims = list(range(1, x_ndim)) + [0]
            reshaped_inputs.append(array_ops.transpose(x, perm=dims))
            transposed = True
          else:
            # We don't transpose inputs if they are 1D vectors or scalars.
            reshaped_inputs.append(x)
        y = self._merge_function(reshaped_inputs)
        y_ndim = backend.ndim(y)
        if transposed:
          # If inputs have been transposed, we have to transpose the output too.
          if y_ndim is None:
            y_shape = array_ops.shape(y)
            y_ndim = array_ops.shape(y_shape)[0]
            batch_size = y_shape[y_ndim - 1]
            new_shape = backend.concatenate([
                array_ops.expand_dims(batch_size, axis=-1), y_shape[:y_ndim - 1]
            ])
            y = array_ops.reshape(y, (-1, batch_size))
            y = array_ops.transpose(y, perm=(1, 0))
            y = array_ops.reshape(y, new_shape)
          elif y_ndim > 1:
            dims = [y_ndim - 1] + list(range(y_ndim - 1))
            y = array_ops.transpose(y, perm=dims)
        return y
    else:
      return self._merge_function(inputs)

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    # [FUNCTION DOC]
    # Computes the broadcast output shape from a list of input shapes (batch first). Folds the
    # non-batch parts through the element-wise broadcast helper (raising a value error when
    # incompatible) and prepends the common known batch size, or an unknown batch size if there is
    # not exactly one distinct known value. Returns a tuple.
    if input_shape[0] is None:
      output_shape = None
    else:
      output_shape = input_shape[0][1:]
    for i in range(1, len(input_shape)):
      if input_shape[i] is None:
        shape = None
      else:
        shape = input_shape[i][1:]
      output_shape = self._compute_elemwise_op_output_shape(output_shape, shape)
    batch_sizes = {s[0] for s in input_shape if s is not None} - {None}
    if len(batch_sizes) == 1:
      output_shape = (list(batch_sizes)[0],) + output_shape
    else:
      output_shape = (None,) + output_shape
    return output_shape

  def compute_mask(self, inputs, mask=None):
    # [FUNCTION DOC]
    # Combines input masks. Returns null if mask is null or every entry is null. Raises a value
    # error if mask or inputs is not a list/tuple, or their lengths differ. Otherwise stacks the
    # non-null masks and returns their logical and across the stack (a boolean tensor).
    if mask is None:
      return None
    if not isinstance(mask, (tuple, list)):
      raise ValueError('`mask` should be a list.')
    if not isinstance(inputs, (tuple, list)):
      raise ValueError('`inputs` should be a list.')
    if len(mask) != len(inputs):
      raise ValueError('The lists `inputs` and `mask` '
                       'should have the same length.')
    if all(m is None for m in mask):
      return None
    masks = [array_ops.expand_dims(m, axis=0) for m in mask if m is not None]
    return backend.all(
        backend.concatenate(masks, axis=0), axis=0, keepdims=False)


class Add(_Merge):
  # [CLASS DOC]
  # Merge layer that sums a list of same-shaped (or broadcastable) tensors element-wise.
  # Stateless; created with standard layer arguments and called on a list of tensors.
  """Layer that adds a list of inputs.

  It takes as input a list of tensors,
  all of the same shape, and returns
  a single tensor (also of the same shape).

  Examples:

  >>> input_shape = (2, 3, 4)
  >>> x1 = tf.random.normal(input_shape)
  >>> x2 = tf.random.normal(input_shape)
  >>> y = tf.keras.layers.Add()([x1, x2])
  >>> print(y.shape)
  (2, 3, 4)

  Used in a functional model:

  >>> input1 = tf.keras.layers.Input(shape=(16,))
  >>> x1 = tf.keras.layers.Dense(8, activation='relu')(input1)
  >>> input2 = tf.keras.layers.Input(shape=(32,))
  >>> x2 = tf.keras.layers.Dense(8, activation='relu')(input2)
  >>> # equivalent to `added = tf.keras.layers.add([x1, x2])`
  >>> added = tf.keras.layers.Add()([x1, x2])
  >>> out = tf.keras.layers.Dense(4)(added)
  >>> model = tf.keras.models.Model(inputs=[input1, input2], outputs=out)

  """

  def _merge_function(self, inputs):
    # [FUNCTION DOC]
    # HELPER: merge hook for Add. Accumulates all inputs by repeated addition and returns the sum
    # tensor. Takes a list of tensors; no side effects beyond what in-place addition implies.
    output = inputs[0]
    for i in range(1, len(inputs)):
      output += inputs[i]
    return output


class Subtract(_Merge):
  # [CLASS DOC]
  # Merge layer that subtracts the second of exactly two tensors from the first, element-wise.
  # Stateless; enforces the two-input requirement at build and merge time.
  """Layer that subtracts two inputs.

  It takes as input a list of tensors of size 2,
  both of the same shape, and returns a single tensor, (inputs[0] - inputs[1]),
  also of the same shape.

  Examples:

  ```python
      import keras

      input1 = keras.layers.Input(shape=(16,))
      x1 = keras.layers.Dense(8, activation='relu')(input1)
      input2 = keras.layers.Input(shape=(32,))
      x2 = keras.layers.Dense(8, activation='relu')(input2)
      # Equivalent to subtracted = keras.layers.subtract([x1, x2])
      subtracted = keras.layers.Subtract()([x1, x2])

      out = keras.layers.Dense(4)(subtracted)
      model = keras.models.Model(inputs=[input1, input2], outputs=out)
  ```
  """

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    # [FUNCTION DOC]
    # Runs the base shape validation, then raises a value error unless exactly two input shapes
    # were given. Creates no weights; returns nothing.
    super(Subtract, self).build(input_shape)
    if len(input_shape) != 2:
      raise ValueError('A `Subtract` layer should be called '
                       'on exactly 2 inputs')

  def _merge_function(self, inputs):
    # [FUNCTION DOC]
    # HELPER: merge hook for Subtract. Raises a value error unless exactly two inputs are given;
    # returns the first input minus the second.
    if len(inputs) != 2:
      raise ValueError('A `Subtract` layer should be called '
                       'on exactly 2 inputs')
    return inputs[0] - inputs[1]


class Multiply(_Merge):
  # [CLASS DOC]
  # Merge layer that multiplies a list of tensors element-wise (with broadcasting). Stateless.
  """Layer that multiplies (element-wise) a list of inputs.

  It takes as input a list of tensors, all of the same shape, and returns
  a single tensor (also of the same shape).

  >>> tf.keras.layers.Multiply()([np.arange(5).reshape(5, 1),
  ...                             np.arange(5, 10).reshape(5, 1)])
  <tf.Tensor: shape=(5, 1), dtype=int64, numpy=
  array([[ 0],
       [ 6],
       [14],
       [24],
       [36]])>

  >>> x1 = tf.keras.layers.Dense(8)(np.arange(10).reshape(5, 2))
  >>> x2 = tf.keras.layers.Dense(8)(np.arange(10, 20).reshape(5, 2))
  >>> multiplied = tf.keras.layers.Multiply()([x1, x2])
  >>> multiplied.shape
  TensorShape([5, 8])
  """

  def _merge_function(self, inputs):
    # [FUNCTION DOC]
    # HELPER: merge hook for Multiply. Returns the element-wise product of all inputs, computed
    # by repeated multiplication (not in place).
    output = inputs[0]
    for i in range(1, len(inputs)):
      output = output * inputs[i]
    return output


class Average(_Merge):
  # [CLASS DOC]
  # Merge layer that returns the element-wise mean of a list of tensors. Stateless.
  """Layer that averages a list of inputs element-wise.

  It takes as input a list of tensors, all of the same shape, and returns
  a single tensor (also of the same shape).

  Example:

  >>> x1 = np.ones((2, 2))
  >>> x2 = np.zeros((2, 2))
  >>> y = tf.keras.layers.Average()([x1, x2])
  >>> y.numpy().tolist()
  [[0.5, 0.5], [0.5, 0.5]]

  Usage in a functional model:

  >>> input1 = tf.keras.layers.Input(shape=(16,))
  >>> x1 = tf.keras.layers.Dense(8, activation='relu')(input1)
  >>> input2 = tf.keras.layers.Input(shape=(32,))
  >>> x2 = tf.keras.layers.Dense(8, activation='relu')(input2)
  >>> avg = tf.keras.layers.Average()([x1, x2])
  >>> out = tf.keras.layers.Dense(4)(avg)
  >>> model = tf.keras.models.Model(inputs=[input1, input2], outputs=out)

  Raises:
    ValueError: If there is a shape mismatch between the inputs and the shapes
      cannot be broadcasted to match.
  """

  def _merge_function(self, inputs):
    # [FUNCTION DOC]
    # HELPER: merge hook for Average. Sums all inputs and divides by the number of inputs;
    # returns the mean tensor.
    output = inputs[0]
    for i in range(1, len(inputs)):
      output += inputs[i]
    return output / len(inputs)


class Maximum(_Merge):
  # [CLASS DOC]
  # Merge layer that returns the element-wise maximum of a list of tensors. Stateless.
  """Layer that computes the maximum (element-wise) a list of inputs.

  It takes as input a list of tensors, all of the same shape, and returns
  a single tensor (also of the same shape).

  >>> tf.keras.layers.Maximum()([np.arange(5).reshape(5, 1),
  ...                            np.arange(5, 10).reshape(5, 1)])
  <tf.Tensor: shape=(5, 1), dtype=int64, numpy=
  array([[5],
       [6],
       [7],
       [8],
       [9]])>

  >>> x1 = tf.keras.layers.Dense(8)(np.arange(10).reshape(5, 2))
  >>> x2 = tf.keras.layers.Dense(8)(np.arange(10, 20).reshape(5, 2))
  >>> maxed = tf.keras.layers.Maximum()([x1, x2])
  >>> maxed.shape
  TensorShape([5, 8])
  """

  def _merge_function(self, inputs):
    # [FUNCTION DOC]
    # HELPER: merge hook for Maximum. Folds the inputs with the element-wise maximum op and
    # returns the resulting tensor.
    output = inputs[0]
    for i in range(1, len(inputs)):
      output = math_ops.maximum(output, inputs[i])
    return output


class Minimum(_Merge):
  # [CLASS DOC]
  # Merge layer that returns the element-wise minimum of a list of tensors. Stateless.
  """Layer that computes the minimum (element-wise) a list of inputs.

  It takes as input a list of tensors, all of the same shape, and returns
  a single tensor (also of the same shape).

  >>> tf.keras.layers.Minimum()([np.arange(5).reshape(5, 1),
  ...                            np.arange(5, 10).reshape(5, 1)])
  <tf.Tensor: shape=(5, 1), dtype=int64, numpy=
  array([[0],
       [1],
       [2],
       [3],
       [4]])>

  >>> x1 = tf.keras.layers.Dense(8)(np.arange(10).reshape(5, 2))
  >>> x2 = tf.keras.layers.Dense(8)(np.arange(10, 20).reshape(5, 2))
  >>> minned = tf.keras.layers.Minimum()([x1, x2])
  >>> minned.shape
  TensorShape([5, 8])
  """

  def _merge_function(self, inputs):
    # [FUNCTION DOC]
    # HELPER: merge hook for Minimum. Folds the inputs with the element-wise minimum op and
    # returns the resulting tensor.
    output = inputs[0]
    for i in range(1, len(inputs)):
      output = math_ops.minimum(output, inputs[i])
    return output


class Concatenate(_Merge):
  # [CLASS DOC]
  # Merge layer that joins a list of tensors along a chosen axis; all other dimensions must match.
  # State: the concatenation axis. Overrides build, merge, output shape, masking and config of the
  # base merge layer, and never requires rank-alignment reshaping.
  """Layer that concatenates a list of inputs.

  It takes as input a list of tensors, all of the same shape except
  for the concatenation axis, and returns a single tensor that is the
  concatenation of all inputs.

  >>> x = np.arange(20).reshape(2, 2, 5)
  >>> print(x)
  [[[ 0  1  2  3  4]
    [ 5  6  7  8  9]]
   [[10 11 12 13 14]
    [15 16 17 18 19]]]
  >>> y = np.arange(20, 30).reshape(2, 1, 5)
  >>> print(y)
  [[[20 21 22 23 24]]
   [[25 26 27 28 29]]]
  >>> tf.keras.layers.Concatenate(axis=1)([x, y])
  <tf.Tensor: shape=(2, 3, 5), dtype=int64, numpy=
  array([[[ 0,  1,  2,  3,  4],
          [ 5,  6,  7,  8,  9],
          [20, 21, 22, 23, 24]],
         [[10, 11, 12, 13, 14],
          [15, 16, 17, 18, 19],
          [25, 26, 27, 28, 29]]])>

  >>> x1 = tf.keras.layers.Dense(8)(np.arange(10).reshape(5, 2))
  >>> x2 = tf.keras.layers.Dense(8)(np.arange(10, 20).reshape(5, 2))
  >>> concatted = tf.keras.layers.Concatenate()([x1, x2])
  >>> concatted.shape
  TensorShape([5, 16])

  """

  def __init__(self, axis=-1, **kwargs):
    # [FUNCTION DOC]
    # Constructor. axis (integer, default last axis) is the concatenation axis; other keyword
    # arguments go to the base layer. Stores the axis, enables mask support and disables the
    # rank-alignment reshaping. Returns nothing.
    """Instantiates a Concatenate layer.

    >>> x = np.arange(20).reshape(2, 2, 5)
    >>> print(x)
    [[[ 0  1  2  3  4]
      [ 5  6  7  8  9]]
     [[10 11 12 13 14]
      [15 16 17 18 19]]]
    >>> y = np.arange(20, 30).reshape(2, 1, 5)
    >>> print(y)
    [[[20 21 22 23 24]]
     [[25 26 27 28 29]]]
    >>> tf.keras.layers.Concatenate(axis=1)([x, y])
    <tf.Tensor: shape=(2, 3, 5), dtype=int64, numpy=
    array([[[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9],
            [20, 21, 22, 23, 24]],
           [[10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
            [25, 26, 27, 28, 29]]])>

    Args:
      axis: Axis along which to concatenate.
      **kwargs: standard layer keyword arguments.
    """
    super(Concatenate, self).__init__(**kwargs)
    self.axis = axis
    self.supports_masking = True
    self._reshape_required = False

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the length check for "at least 1 input" can never fire, because the first
    # element is indexed before it; an empty list fails earlier with an index error instead.
    # Validates the list of input shapes without creating weights. Raises a value error if the
    # shapes are not a list of tuples. If all shapes are unknown, returns silently. Otherwise
    # removes the concatenation axis from each shape and requires the remainders to match: ranks
    # must be equal and every known dimension must agree (unknown dimensions are ignored), else a
    # value error is raised. Returns nothing.
    # Used purely for shape validation.
    if not isinstance(input_shape[0], tuple) or len(input_shape) < 1:
      raise ValueError('A `Concatenate` layer should be called '
                       'on a list of at least 1 input.')
    if all(shape is None for shape in input_shape):
      return
    reduced_inputs_shapes = [list(shape) for shape in input_shape]
    shape_set = set()
    for i in range(len(reduced_inputs_shapes)):
      del reduced_inputs_shapes[i][self.axis]
      shape_set.add(tuple(reduced_inputs_shapes[i]))

    if len(shape_set) != 1:
      err_msg = ('A `Concatenate` layer requires inputs with matching shapes '
                 'except for the concat axis. Got inputs shapes: %s' %
                 input_shape)
      # Make sure all the shapes have same ranks.
      ranks = set(len(shape) for shape in shape_set)
      if len(ranks) != 1:
        raise ValueError(err_msg)
      # Get the only rank for the set.
      (rank,) = ranks
      for axis in range(rank):
        # Skip the Nones in the shape since they are dynamic, also the axis for
        # concat has been removed above.
        unique_dims = set(
            shape[axis] for shape in shape_set if shape[axis] is not None)
        if len(unique_dims) > 1:
          raise ValueError(err_msg)

  def _merge_function(self, inputs):
    # [FUNCTION DOC]
    # HELPER: merge hook for Concatenate. Joins the input tensors along the stored axis and
    # returns the single concatenated tensor.
    return backend.concatenate(inputs, axis=self.axis)

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    # [FUNCTION DOC]
    # Computes the shape after concatenation from a list of shape tuples. Raises a value error if
    # the argument is not a list/tuple of shape sequences. The result equals the first shape with
    # the size on the concatenation axis replaced by the sum of all sizes on that axis, or unknown
    # if any of them is unknown. Returns a tuple.
    if ((not isinstance(input_shape, (tuple, list))) or
        (not isinstance(input_shape[0], (tuple, list)))):
      # The tf_utils.shape_type_conversion decorator turns tensorshapes
      # into tuples, so we need to verify that `input_shape` is a list/tuple,
      # *and* that the individual elements are themselves shape tuples.
      raise ValueError('A `Concatenate` layer should be called '
                       'on a list of inputs.')
    input_shapes = input_shape
    output_shape = list(input_shapes[0])
    for shape in input_shapes[1:]:
      if output_shape[self.axis] is None or shape[self.axis] is None:
        output_shape[self.axis] = None
        break
      output_shape[self.axis] += shape[self.axis]
    return tuple(output_shape)

  def compute_mask(self, inputs, mask=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the concatenated masks are reduced with a logical and over the last axis rather
    # than being returned as a mask concatenated along the layer axis, so the result's shape
    # differs from the concatenated output's leading dimensions unless the axis is the last one.
    # Builds the output mask. Returns null if mask is null or all entries are null; raises a value
    # error if mask or inputs is not a list/tuple or lengths differ. Missing masks become all-true
    # masks shaped like their input; masks of lower rank than their input get a trailing axis
    # added. All are concatenated along the layer axis and reduced with logical and (last axis).
    if mask is None:
      return None
    if not isinstance(mask, (tuple, list)):
      raise ValueError('`mask` should be a list.')
    if not isinstance(inputs, (tuple, list)):
      raise ValueError('`inputs` should be a list.')
    if len(mask) != len(inputs):
      raise ValueError('The lists `inputs` and `mask` '
                       'should have the same length.')
    if all(m is None for m in mask):
      return None
    # Make a list of masks while making sure
    # the dimensionality of each mask
    # is the same as the corresponding input.
    masks = []
    for input_i, mask_i in zip(inputs, mask):
      if mask_i is None:
        # Input is unmasked. Append all 1s to masks,
        masks.append(array_ops.ones_like(input_i, dtype='bool'))
      elif backend.ndim(mask_i) < backend.ndim(input_i):
        # Mask is smaller than the input, expand it
        masks.append(array_ops.expand_dims(mask_i, axis=-1))
      else:
        masks.append(mask_i)
    concatenated = backend.concatenate(masks, axis=self.axis)
    return backend.all(concatenated, axis=-1, keepdims=False)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the serializable configuration: the base layer entries plus the concatenation axis.
    config = {
        'axis': self.axis,
    }
    base_config = super(Concatenate, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class Dot(_Merge):
  # [CLASS DOC]
  # Merge layer computing a batched dot product between exactly two tensors along chosen axes,
  # optionally after L2 normalization (giving cosine similarity). State: the axes (one integer or
  # a pair of integers), the normalize flag. Produces no output mask.
  """Layer that computes a dot product between samples in two tensors.

  E.g. if applied to a list of two tensors `a` and `b` of shape
  `(batch_size, n)`, the output will be a tensor of shape `(batch_size, 1)`
  where each entry `i` will be the dot product between
  `a[i]` and `b[i]`.

  >>> x = np.arange(10).reshape(1, 5, 2)
  >>> print(x)
  [[[0 1]
    [2 3]
    [4 5]
    [6 7]
    [8 9]]]
  >>> y = np.arange(10, 20).reshape(1, 2, 5)
  >>> print(y)
  [[[10 11 12 13 14]
    [15 16 17 18 19]]]
  >>> tf.keras.layers.Dot(axes=(1, 2))([x, y])
  <tf.Tensor: shape=(1, 2, 2), dtype=int64, numpy=
  array([[[260, 360],
          [320, 445]]])>

  >>> x1 = tf.keras.layers.Dense(8)(np.arange(10).reshape(5, 2))
  >>> x2 = tf.keras.layers.Dense(8)(np.arange(10, 20).reshape(5, 2))
  >>> dotted = tf.keras.layers.Dot(axes=1)([x1, x2])
  >>> dotted.shape
  TensorShape([5, 1])


  """

  def __init__(self, axes, normalize=False, **kwargs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring calls this an "element-wise" dot product; it is a batched dot
    # product contracting the chosen axes.
    # Constructor. axes: an integer or a list/tuple of exactly two integers (axis of first and of
    # second input). normalize: boolean, default false. Raises a type error if axes is neither an
    # integer nor list/tuple, and a value error if a sequence does not have two integer elements.
    # Stores axes and normalize, enables mask support, disables rank-alignment reshaping.
    """Initializes a layer that computes the element-wise dot product.

      >>> x = np.arange(10).reshape(1, 5, 2)
      >>> print(x)
      [[[0 1]
        [2 3]
        [4 5]
        [6 7]
        [8 9]]]
      >>> y = np.arange(10, 20).reshape(1, 2, 5)
      >>> print(y)
      [[[10 11 12 13 14]
        [15 16 17 18 19]]]
      >>> tf.keras.layers.Dot(axes=(1, 2))([x, y])
      <tf.Tensor: shape=(1, 2, 2), dtype=int64, numpy=
      array([[[260, 360],
              [320, 445]]])>

    Args:
      axes: Integer or tuple of integers,
        axis or axes along which to take the dot product. If a tuple, should
        be two integers corresponding to the desired axis from the first input
        and the desired axis from the second input, respectively. Note that the
        size of the two selected axes must match.
      normalize: Whether to L2-normalize samples along the
        dot product axis before taking the dot product.
        If set to True, then the output of the dot product
        is the cosine proximity between the two samples.
      **kwargs: Standard layer keyword arguments.
    """
    super(Dot, self).__init__(**kwargs)
    if not isinstance(axes, int):
      if not isinstance(axes, (list, tuple)):
        raise TypeError('Invalid type for `axes` - '
                        'should be a list or an int.')
      if len(axes) != 2:
        raise ValueError('Invalid format for `axes` - '
                         'should contain two elements.')
      if not isinstance(axes[0], int) or not isinstance(axes[1], int):
        raise ValueError('Invalid format for `axes` - '
                         'list elements should be "int".')
    self.axes = axes
    self.normalize = normalize
    self.supports_masking = True
    self._reshape_required = False

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    # [FUNCTION DOC]
    # Validates input shapes only (no weights). Raises a value error unless given a list of
    # exactly two shape tuples. Unknown shapes return silently. Resolves negative integer axes
    # per input rank and raises a value error if the sizes of the two chosen axes differ.
    # Returns nothing.
    # Used purely for shape validation.
    if not isinstance(input_shape[0], tuple) or len(input_shape) != 2:
      raise ValueError('A `Dot` layer should be called '
                       'on a list of 2 inputs.')
    shape1 = input_shape[0]
    shape2 = input_shape[1]
    if shape1 is None or shape2 is None:
      return
    if isinstance(self.axes, int):
      if self.axes < 0:
        axes = [self.axes % len(shape1), self.axes % len(shape2)]
      else:
        axes = [self.axes] * 2
    else:
      axes = self.axes
    if shape1[axes[0]] != shape2[axes[1]]:
      raise ValueError('Dimension incompatibility '
                       '%s != %s. ' % (shape1[axes[0]], shape2[axes[1]]) +
                       'Layer shapes: %s, %s. ' % (shape1, shape2) +
                       'Chosen axes: %s, %s' % (axes[0], axes[1]))

  def _merge_function(self, inputs):
    # [FUNCTION DOC]
    # HELPER: merge hook for Dot. Rejects ragged inputs (via a shared utility) and raises a value
    # error unless exactly two inputs are given. Normalizes negative axes using each input's rank,
    # optionally L2-normalizes each input along its axis, and returns the batched dot product of
    # the two tensors along the axes.
    base_layer_utils.no_ragged_support(inputs, self.name)
    if len(inputs) != 2:
      raise ValueError('A `Dot` layer should be called on exactly 2 inputs')
    x1 = inputs[0]
    x2 = inputs[1]
    if isinstance(self.axes, int):
      if self.axes < 0:
        axes = [self.axes % backend.ndim(x1), self.axes % backend.ndim(x2)]
      else:
        axes = [self.axes] * 2
    else:
      axes = []
      for i in range(len(self.axes)):
        if self.axes[i] < 0:
          axes.append(self.axes[i] % backend.ndim(inputs[i]))
        else:
          axes.append(self.axes[i])
    if self.normalize:
      x1 = nn.l2_normalize(x1, axis=axes[0])
      x2 = nn.l2_normalize(x2, axis=axes[1])
    output = backend.batch_dot(x1, x2, axes)
    return output

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    # [FUNCTION DOC]
    # Computes the dot-product output shape from exactly two input shapes (raises a value error
    # otherwise). Removes the contracted axis from each shape and the batch axis from the second,
    # concatenates the remainders, and appends a trailing size of one if only one dimension is
    # left. Returns a tuple.
    if not isinstance(input_shape, (tuple, list)) or len(input_shape) != 2:
      raise ValueError('A `Dot` layer should be called '
                       'on a list of 2 inputs.')
    shape1 = list(input_shape[0])
    shape2 = list(input_shape[1])
    if isinstance(self.axes, int):
      if self.axes < 0:
        axes = [self.axes % len(shape1), self.axes % len(shape2)]
      else:
        axes = [self.axes] * 2
    else:
      axes = self.axes
    shape1.pop(axes[0])
    shape2.pop(axes[1])
    shape2.pop(0)
    output_shape = shape1 + shape2
    if len(output_shape) == 1:
      output_shape += [1]
    return tuple(output_shape)

  def compute_mask(self, inputs, mask=None):
    # [FUNCTION DOC]
    # Always returns null: the dot product layer does not propagate masks, ignoring both arguments.
    return None

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the serializable configuration: base layer entries plus the axes and normalize flag.
    config = {
        'axes': self.axes,
        'normalize': self.normalize,
    }
    base_config = super(Dot, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


def add(inputs, **kwargs):
  # [FUNCTION DOC]
  # Functional wrapper: creates an Add layer from the keyword arguments and applies it to the
  # list of input tensors. Returns the summed tensor. Errors come from the layer (shape mismatch).
  """Functional interface to the `tf.keras.layers.Add` layer.

  Args:
      inputs: A list of input tensors (at least 2) with the same shape.
      **kwargs: Standard layer keyword arguments.

  Returns:
      A tensor as the sum of the inputs. It has the same shape as the inputs.

  Examples:

  >>> input_shape = (2, 3, 4)
  >>> x1 = tf.random.normal(input_shape)
  >>> x2 = tf.random.normal(input_shape)
  >>> y = tf.keras.layers.add([x1, x2])
  >>> print(y.shape)
  (2, 3, 4)

  Used in a functional model:

  >>> input1 = tf.keras.layers.Input(shape=(16,))
  >>> x1 = tf.keras.layers.Dense(8, activation='relu')(input1)
  >>> input2 = tf.keras.layers.Input(shape=(32,))
  >>> x2 = tf.keras.layers.Dense(8, activation='relu')(input2)
  >>> added = tf.keras.layers.add([x1, x2])
  >>> out = tf.keras.layers.Dense(4)(added)
  >>> model = tf.keras.models.Model(inputs=[input1, input2], outputs=out)

  """
  return Add(**kwargs)(inputs)


def subtract(inputs, **kwargs):
  # [FUNCTION DOC]
  # Functional wrapper: creates a Subtract layer from the keyword arguments and applies it to
  # exactly two input tensors. Returns the first minus the second; the layer raises a value error
  # if the count is not two.
  """Functional interface to the `Subtract` layer.

  Args:
      inputs: A list of input tensors (exactly 2).
      **kwargs: Standard layer keyword arguments.

  Returns:
      A tensor, the difference of the inputs.

  Examples:

  ```python
      import keras

      input1 = keras.layers.Input(shape=(16,))
      x1 = keras.layers.Dense(8, activation='relu')(input1)
      input2 = keras.layers.Input(shape=(32,))
      x2 = keras.layers.Dense(8, activation='relu')(input2)
      subtracted = keras.layers.subtract([x1, x2])

      out = keras.layers.Dense(4)(subtracted)
      model = keras.models.Model(inputs=[input1, input2], outputs=out)
  ```
  """
  return Subtract(**kwargs)(inputs)


def multiply(inputs, **kwargs):
  # [FUNCTION DOC]
  # Functional wrapper: creates a Multiply layer from the keyword arguments and applies it to the
  # list of input tensors. Returns the element-wise product tensor.
  """Functional interface to the `Multiply` layer.

  Example:

  >>> x1 = np.arange(3.0)
  >>> x2 = np.arange(3.0)
  >>> tf.keras.layers.multiply([x1, x2])
  <tf.Tensor: shape=(3,), dtype=float32, numpy=array([0., 1., 4.], ...)>

  Usage in a functional model:

  >>> input1 = tf.keras.layers.Input(shape=(16,))
  >>> x1 = tf.keras.layers.Dense(8, activation='relu')(input1) #shape=(None, 8)
  >>> input2 = tf.keras.layers.Input(shape=(32,))
  >>> x2 = tf.keras.layers.Dense(8, activation='relu')(input2) #shape=(None, 8)
  >>> out = tf.keras.layers.multiply([x1,x2]) #shape=(None, 8)
  >>> out = tf.keras.layers.Dense(4)(out)
  >>> model = tf.keras.models.Model(inputs=[input1, input2], outputs=out)

  Args:
      inputs: A list of input tensors (at least 2).
      **kwargs: Standard layer keyword arguments.

  Returns:
      A tensor, the element-wise product of the inputs.
  """
  return Multiply(**kwargs)(inputs)


def average(inputs, **kwargs):
  # [FUNCTION DOC]
  # Functional wrapper: creates an Average layer from the keyword arguments and applies it to the
  # list of input tensors. Returns the element-wise mean tensor.
  """Functional interface to the `tf.keras.layers.Average` layer.

  Example:

  >>> x1 = np.ones((2, 2))
  >>> x2 = np.zeros((2, 2))
  >>> y = tf.keras.layers.Average()([x1, x2])
  >>> y.numpy().tolist()
  [[0.5, 0.5], [0.5, 0.5]]

  Usage in a functional model:

  >>> input1 = tf.keras.layers.Input(shape=(16,))
  >>> x1 = tf.keras.layers.Dense(8, activation='relu')(input1)
  >>> input2 = tf.keras.layers.Input(shape=(32,))
  >>> x2 = tf.keras.layers.Dense(8, activation='relu')(input2)
  >>> avg = tf.keras.layers.Average()([x1, x2])
  >>> out = tf.keras.layers.Dense(4)(avg)
  >>> model = tf.keras.models.Model(inputs=[input1, input2], outputs=out)

  Args:
      inputs: A list of input tensors (at least 2).
      **kwargs: Standard layer keyword arguments.

  Returns:
      A tensor, the average of the inputs.

  Raises:
    ValueError: If there is a shape mismatch between the inputs and the shapes
      cannot be broadcasted to match.
  """
  return Average(**kwargs)(inputs)


def maximum(inputs, **kwargs):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says a value error is raised for inputs of different shape, but the
  # layer broadcasts compatible shapes and only raises when shapes cannot be broadcast together.
  # Functional wrapper: creates a Maximum layer from the keyword arguments and applies it to the
  # list of input tensors. Returns the element-wise maximum tensor.
  """Functional interface to compute maximum (element-wise) list of `inputs`.

  This is equivalent to the `tf.keras.layers.Maximum` layer.

  For example:

  ```python
  input1 = tf.keras.layers.Input(shape=(16,))
  x1 = tf.keras.layers.Dense(8, activation='relu')(input1) #shape=(None, 8)
  input2 = tf.keras.layers.Input(shape=(32,))
  x2 = tf.keras.layers.Dense(8, activation='relu')(input2) #shape=(None, 8)
  max_inp=tf.keras.layers.maximum([x1,x2]) #shape=(None, 8)
  out = tf.keras.layers.Dense(4)(max_inp)
  model = tf.keras.models.Model(inputs=[input1, input2], outputs=out)
  ```

  Args:
      inputs: A list of input tensors (at least 2) of same shape.
      **kwargs: Standard layer keyword arguments.

  Returns:
      A tensor (of same shape as input tensor) with the element-wise
      maximum of the inputs.

  Raises:
      ValueError: If input tensors are of different shape.
  """
  return Maximum(**kwargs)(inputs)


def minimum(inputs, **kwargs):
  # [FUNCTION DOC]
  # Functional wrapper: creates a Minimum layer from the keyword arguments and applies it to the
  # list of input tensors. Returns the element-wise minimum tensor.
  """Functional interface to the `Minimum` layer.

  Args:
      inputs: A list of input tensors (at least 2).
      **kwargs: Standard layer keyword arguments.

  Returns:
      A tensor, the element-wise minimum of the inputs.
  """
  return Minimum(**kwargs)(inputs)


def concatenate(inputs, axis=-1, **kwargs):
  # [FUNCTION DOC]
  # Functional wrapper: creates a Concatenate layer with the given axis (default last) and extra
  # keyword arguments, and applies it to the list of input tensors. Returns the joined tensor.
  """Functional interface to the `Concatenate` layer.

  >>> x = np.arange(20).reshape(2, 2, 5)
  >>> print(x)
  [[[ 0  1  2  3  4]
    [ 5  6  7  8  9]]
   [[10 11 12 13 14]
    [15 16 17 18 19]]]
  >>> y = np.arange(20, 30).reshape(2, 1, 5)
  >>> print(y)
  [[[20 21 22 23 24]]
   [[25 26 27 28 29]]]
  >>> tf.keras.layers.concatenate([x, y],
  ...                             axis=1)
  <tf.Tensor: shape=(2, 3, 5), dtype=int64, numpy=
  array([[[ 0,  1,  2,  3,  4],
        [ 5,  6,  7,  8,  9],
        [20, 21, 22, 23, 24]],
       [[10, 11, 12, 13, 14],
        [15, 16, 17, 18, 19],
        [25, 26, 27, 28, 29]]])>

  Args:
      inputs: A list of input tensors (at least 2).
      axis: Concatenation axis.
      **kwargs: Standard layer keyword arguments.

  Returns:
      A tensor, the concatenation of the inputs alongside axis `axis`.
  """
  return Concatenate(axis=axis, **kwargs)(inputs)


def dot(inputs, axes, normalize=False, **kwargs):
  # [FUNCTION DOC]
  # Functional wrapper: creates a Dot layer with the given axes and normalize flag (default
  # false) plus extra keyword arguments, and applies it to the two input tensors. Returns the
  # batched dot product tensor.
  """Functional interface to the `Dot` layer.

  Args:
      inputs: A list of input tensors (at least 2).
      axes: Integer or tuple of integers,
          axis or axes along which to take the dot product.
      normalize: Whether to L2-normalize samples along the
          dot product axis before taking the dot product.
          If set to True, then the output of the dot product
          is the cosine proximity between the two samples.
      **kwargs: Standard layer keyword arguments.

  Returns:
      A tensor, the dot product of the samples from the inputs.
  """
  return Dot(axes=axes, normalize=normalize, **kwargs)(inputs)
