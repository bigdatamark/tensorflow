# [MODULE DOC]
# Weight constraints for Keras layers: callable objects that project a weight tensor (for example
# by clipping or normalizing) after an optimizer update. Defines the Constraint base class, the
# MaxNorm, NonNeg, UnitNorm, MinMaxNorm and RadialConstraint implementations, lowercase and
# legacy aliases for them, and the serialize, deserialize and get helpers that convert between
# constraint objects, configuration mappings and names. Deserialization looks names up in this
# module's global namespace. No import-time side effects beyond defining names and aliases.
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
# pylint: disable=invalid-name
# pylint: disable=g-classes-have-attributes
"""Constraints: functions that impose constraints on weight values."""

from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import while_loop
from tensorflow.tools.docs import doc_controls


class Constraint:
  # [CLASS DOC]
  # Base class for weight constraints. A constraint is a stateless callable that takes a weight
  # tensor and returns a projected tensor of the same kind; subclasses override the call method
  # and, if they have parameters, the config method. Passed to layers through kernel or bias
  # constraint arguments.
  """Base class for weight constraints.

  A `Constraint` instance works like a stateless function.
  Users who subclass this
  class should override the `__call__` method, which takes a single
  weight parameter and return a projected version of that parameter
  (e.g. normalized or clipped). Constraints can be used with various Keras
  layers via the `kernel_constraint` or `bias_constraint` arguments.

  Here's a simple example of a non-negative weight constraint:

  >>> class NonNegative(tf.keras.constraints.Constraint):
  ...
  ...  def __call__(self, w):
  ...    return w * tf.cast(tf.math.greater_equal(w, 0.), w.dtype)

  >>> weight = tf.constant((-1.0, 1.0))
  >>> NonNegative()(weight)
  <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.,  1.], dtype=float32)>

  >>> tf.keras.layers.Dense(4, kernel_constraint=NonNegative())
  """

  def __call__(self, w):
    # [FUNCTION DOC]
    # Default projection: returns the given weight variable unchanged. Subclasses override this
    # to implement their own projection. Input: w, the weight tensor or variable. No side effects.
    """Applies the constraint to the input weight variable.

    By default, the inputs weight variable is not modified.
    Users should override this method to implement their own projection
    function.

    Args:
      w: Input weight variable.

    Returns:
      Projected variable (by default, returns unmodified inputs).
    """
    return w

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the configuration of the constraint as a serializable mapping; the base class has
    # no parameters so it returns an empty mapping. Subclasses return their constructor arguments.
    """Returns a Python dict of the object config.

    A constraint config is a Python dictionary (JSON-serializable) that can
    be used to reinstantiate the same object.

    Returns:
      Python dict containing the configuration of the constraint object.
    """
    return {}


class MaxNorm(Constraint):
  # [CLASS DOC]
  # Constraint that rescales weights so the norm taken along a chosen axis (the weights incident
  # to each unit) does not exceed a maximum value. Holds max_value and axis. Alias: max_norm.
  """MaxNorm weight constraint.

  Constrains the weights incident to each hidden unit
  to have a norm less than or equal to a desired value.

  Also available via the shortcut function `tf.keras.constraints.max_norm`.

  Args:
    max_value: the maximum norm value for the incoming weights.
    axis: integer, axis along which to calculate weight norms.
      For instance, in a `Dense` layer the weight matrix
      has shape `(input_dim, output_dim)`,
      set `axis` to `0` to constrain each weight vector
      of length `(input_dim,)`.
      In a `Conv2D` layer with `data_format="channels_last"`,
      the weight tensor has shape
      `(rows, cols, input_depth, output_depth)`,
      set `axis` to `[0, 1, 2]`
      to constrain the weights of each filter tensor of size
      `(rows, cols, input_depth)`.

  """

  def __init__(self, max_value=2, axis=0):
    # [FUNCTION DOC]
    # Stores the maximum allowed norm (max_value, default 2) and the axis along which norms are
    # computed (axis, default 0). Returns nothing.
    self.max_value = max_value
    self.axis = axis

  @doc_controls.do_not_generate_docs
  def __call__(self, w):
    # [FUNCTION DOC]
    # Computes the norm of w along the configured axis (keeping dimensions), clips it between zero
    # and max_value, and returns w scaled by the ratio of clipped norm to the original norm (with a
    # small epsilon in the denominator to avoid division by zero). Input: w, a weight tensor.
    # Returns a tensor of the same shape; no side effects.
    norms = backend.sqrt(
        math_ops.reduce_sum(math_ops.square(w), axis=self.axis, keepdims=True))
    desired = backend.clip(norms, 0, self.max_value)
    return w * (desired / (backend.epsilon() + norms))

  @doc_controls.do_not_generate_docs
  def get_config(self):
    # [FUNCTION DOC]
    # Returns a mapping with the max_value and axis settings, suitable for re-creating the
    # constraint.
    return {'max_value': self.max_value, 'axis': self.axis}


class NonNeg(Constraint):
  # [CLASS DOC]
  # Constraint that forces weights to be non-negative by zeroing every negative entry. Has no
  # parameters and no state. Alias: non_neg.
  """Constrains the weights to be non-negative.

  Also available via the shortcut function `tf.keras.constraints.non_neg`.
  """

  def __call__(self, w):
    # [FUNCTION DOC]
    # Returns w multiplied elementwise by a 0/1 mask (cast to the backend default float type)
    # that is one where w is greater than or equal to zero, so negative entries become zero.
    # Input: w, a weight tensor. No side effects.
    return w * math_ops.cast(math_ops.greater_equal(w, 0.), backend.floatx())


class UnitNorm(Constraint):
  # [CLASS DOC]
  # Constraint that rescales weights so the norm along a chosen axis equals one. Holds the axis.
  # Alias: unit_norm.
  """Constrains the weights incident to each hidden unit to have unit norm.

  Also available via the shortcut function `tf.keras.constraints.unit_norm`.

  Args:
    axis: integer, axis along which to calculate weight norms.
      For instance, in a `Dense` layer the weight matrix
      has shape `(input_dim, output_dim)`,
      set `axis` to `0` to constrain each weight vector
      of length `(input_dim,)`.
      In a `Conv2D` layer with `data_format="channels_last"`,
      the weight tensor has shape
      `(rows, cols, input_depth, output_depth)`,
      set `axis` to `[0, 1, 2]`
      to constrain the weights of each filter tensor of size
      `(rows, cols, input_depth)`.
  """

  def __init__(self, axis=0):
    # [FUNCTION DOC]
    # Stores the axis (default 0) along which the norm is computed. Returns nothing.
    self.axis = axis

  @doc_controls.do_not_generate_docs
  def __call__(self, w):
    # [FUNCTION DOC]
    # Divides w by its norm along the configured axis (square root of the sum of squares, kept as
    # a dimension) plus a small epsilon, giving unit norm. Input: w, a weight tensor. Returns a
    # tensor of the same shape; no side effects.
    return w / (
        backend.epsilon() + backend.sqrt(
            math_ops.reduce_sum(
                math_ops.square(w), axis=self.axis, keepdims=True)))

  @doc_controls.do_not_generate_docs
  def get_config(self):
    # [FUNCTION DOC]
    # Returns a mapping containing the axis setting.
    return {'axis': self.axis}


class MinMaxNorm(Constraint):
  # [CLASS DOC]
  # Constraint that moves the norm of weights along an axis toward the interval between a minimum
  # and a maximum value, with a rate controlling how strictly (1.0 means fully) the interval is
  # enforced. Holds min_value, max_value, rate and axis. Alias: min_max_norm.
  """MinMaxNorm weight constraint.

  Constrains the weights incident to each hidden unit
  to have the norm between a lower bound and an upper bound.

  Also available via the shortcut function `tf.keras.constraints.min_max_norm`.

  Args:
    min_value: the minimum norm for the incoming weights.
    max_value: the maximum norm for the incoming weights.
    rate: rate for enforcing the constraint: weights will be
      rescaled to yield
      `(1 - rate) * norm + rate * norm.clip(min_value, max_value)`.
      Effectively, this means that rate=1.0 stands for strict
      enforcement of the constraint, while rate<1.0 means that
      weights will be rescaled at each step to slowly move
      towards a value inside the desired interval.
    axis: integer, axis along which to calculate weight norms.
      For instance, in a `Dense` layer the weight matrix
      has shape `(input_dim, output_dim)`,
      set `axis` to `0` to constrain each weight vector
      of length `(input_dim,)`.
      In a `Conv2D` layer with `data_format="channels_last"`,
      the weight tensor has shape
      `(rows, cols, input_depth, output_depth)`,
      set `axis` to `[0, 1, 2]`
      to constrain the weights of each filter tensor of size
      `(rows, cols, input_depth)`.
  """

  def __init__(self, min_value=0.0, max_value=1.0, rate=1.0, axis=0):
    # [FUNCTION DOC]
    # Stores min_value (default 0.0), max_value (default 1.0), rate (default 1.0, the enforcement
    # strength) and axis (default 0). Returns nothing; no validation of the values is done.
    self.min_value = min_value
    self.max_value = max_value
    self.rate = rate
    self.axis = axis

  @doc_controls.do_not_generate_docs
  def __call__(self, w):
    # [FUNCTION DOC]
    # Computes the norm of w along the axis, blends it with the version clipped to the interval
    # (rate weighting the clipped value and one minus rate the original), and returns w scaled by
    # the ratio of this desired norm to the original norm (epsilon guards division by zero).
    # Input: w, a weight tensor. Returns a tensor of the same shape; no side effects.
    norms = backend.sqrt(
        math_ops.reduce_sum(math_ops.square(w), axis=self.axis, keepdims=True))
    desired = (
        self.rate * backend.clip(norms, self.min_value, self.max_value) +
        (1 - self.rate) * norms)
    return w * (desired / (backend.epsilon() + norms))

  @doc_controls.do_not_generate_docs
  def get_config(self):
    # [FUNCTION DOC]
    # Returns a mapping with min_value, max_value, rate and axis.
    return {
        'min_value': self.min_value,
        'max_value': self.max_value,
        'rate': self.rate,
        'axis': self.axis
    }


class RadialConstraint(Constraint):
  # [CLASS DOC]
  # Constraint for four-dimensional convolution kernels (rows, columns, input depth, output
  # depth) that makes every entry at the same radial ring around the kernel center equal, so the
  # kernel is radially symmetric. Has no parameters. Alias: radial_constraint.
  """Constrains `Conv2D` kernel weights to be the same for each radius.

  Also available via the shortcut function
  `tf.keras.constraints.radial_constraint`.

  For example, the desired output for the following 4-by-4 kernel:

  ```
      kernel = [[v_00, v_01, v_02, v_03],
                [v_10, v_11, v_12, v_13],
                [v_20, v_21, v_22, v_23],
                [v_30, v_31, v_32, v_33]]
  ```

  is this::

  ```
      kernel = [[v_11, v_11, v_11, v_11],
                [v_11, v_33, v_33, v_11],
                [v_11, v_33, v_33, v_11],
                [v_11, v_11, v_11, v_11]]
  ```

  This constraint can be applied to any `Conv2D` layer version, including
  `Conv2DTranspose` and `SeparableConv2D`, and with either `"channels_last"` or
  `"channels_first"` data format. The method assumes the weight tensor is of
  shape `(rows, cols, input_depth, output_depth)`.
  """

  @doc_controls.do_not_generate_docs
  def __call__(self, w):
    # [FUNCTION DOC]
    # Makes a four-dimensional kernel radially symmetric. Input: w, a tensor of rank four (rows,
    # columns, input depth, output depth). Raises a value error if the rank is unknown or not four.
    # Merges the depth dimensions, applies the per-slice radial constraint with a map over slices,
    # and restores the original shape. Returns the constrained tensor; no side effects.
    w_shape = w.shape
    if w_shape.rank is None or w_shape.rank != 4:
      raise ValueError(
          'The weight tensor must be of rank 4, but is of shape: %s' % w_shape)

    height, width, channels, kernels = w_shape
    w = backend.reshape(w, (height, width, channels * kernels))
    # TODO(cpeter): Switch map_fn for a faster tf.vectorized_map once
    # backend.switch is supported.
    w = backend.map_fn(
        self._kernel_constraint,
        backend.stack(array_ops_stack.unstack(w, axis=-1), axis=0))
    return backend.reshape(
        backend.stack(array_ops_stack.unstack(w, axis=0), axis=-1),
        (height, width, channels, kernels))

  def _kernel_constraint(self, kernel):
    # [FUNCTION DOC]
    # HELPER: per-slice worker for the radial constraint call method.
    # Takes one kernel slice (height, width, channels) and builds a new one in which each ring
    # around the center is filled with the value found on the diagonal at that ring: it starts from
    # the center element (or a two-by-two block for even sizes) and repeatedly pads the array by one
    # on each side with a diagonal value, using a while loop with shape invariants. Returns the
    # rebuilt kernel. The loop assumes square kernels (only the first dimension size is used).
    """Radially constraints a kernel with shape (height, width, channels)."""
    padding = backend.constant([[1, 1], [1, 1]], dtype='int32')

    kernel_shape = backend.shape(kernel)[0]
    start = backend.cast(kernel_shape / 2, 'int32')

    kernel_new = backend.switch(
        backend.cast(math_ops.floormod(kernel_shape, 2), 'bool'),
        lambda: kernel[start - 1:start, start - 1:start],
        lambda: kernel[start - 1:start, start - 1:start] + backend.zeros(  # pylint: disable=g-long-lambda
            (2, 2), dtype=kernel.dtype))
    index = backend.switch(
        backend.cast(math_ops.floormod(kernel_shape, 2), 'bool'),
        lambda: backend.constant(0, dtype='int32'),
        lambda: backend.constant(1, dtype='int32'))
    while_condition = lambda index, *args: backend.less(index, start)

    def body_fn(i, array):
      # [FUNCTION DOC]
      # HELPER: loop body for the radial kernel construction. Takes the loop counter i and the
      # current array, pads the array by one on every side using the diagonal kernel value at the
      # offset given by the start position plus i, and returns the incremented counter together
      # with the padded array.
      return i + 1, array_ops.pad(
          array,
          padding,
          constant_values=kernel[start + i, start + i])

    _, kernel_new = while_loop.while_loop(
        while_condition,
        body_fn, [index, kernel_new],
        shape_invariants=[
            index.get_shape(),
            tensor_shape.TensorShape([None, None])
        ])
    return kernel_new


# Aliases.

max_norm = MaxNorm
non_neg = NonNeg
unit_norm = UnitNorm
min_max_norm = MinMaxNorm
radial_constraint = RadialConstraint

# Legacy aliases.
maxnorm = max_norm
nonneg = non_neg
unitnorm = unit_norm


def serialize(constraint):
  # [FUNCTION DOC]
  # Converts a constraint object into its serializable representation (class name plus config
  # mapping) using the Keras generic serialization helper. Input: constraint, a constraint
  # object. Returns the serialized form.
  return serialize_keras_object(constraint)


def deserialize(config, custom_objects=None):
  # [FUNCTION DOC]
  # Rebuilds a constraint from a serialized configuration. Inputs: config, a mapping with a class
  # name and config; custom_objects, an optional mapping of extra names to classes. Class names
  # are looked up in this module's global names first, then custom objects. Returns the new
  # constraint object. Raises whatever the generic deserializer raises for unknown names.
  return deserialize_keras_object(
      config,
      module_objects=globals(),
      custom_objects=custom_objects,
      printable_module_name='constraint')


def get(identifier):
  # [FUNCTION DOC]
  # Resolves a constraint identifier into a constraint. Input: identifier, which may be nothing
  # (returns nothing), a configuration mapping (deserialized), a string class name (deserialized
  # with empty config), or a callable (returned as is). Raises a value error for any other type.
  if identifier is None:
    return None
  if isinstance(identifier, dict):
    return deserialize(identifier)
  elif isinstance(identifier, str):
    config = {'class_name': str(identifier), 'config': {}}
    return deserialize(config)
  elif callable(identifier):
    return identifier
  else:
    raise ValueError('Could not interpret constraint identifier: ' +
                     str(identifier))
