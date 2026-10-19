# [MODULE DOC]
# Keras layers that apply activation functions as layers: LeakyReLU, PReLU, ELU, ThresholdedReLU,
# Softmax and ReLU, plus a private helper giving the large negative value used for masking in
# Softmax and a function returning this module's global namespace. All layers are element-wise
# (except Softmax along an axis), keep the input shape, and support masking. Part of the layers
# package, whose registry exposes them. No import-time side effects beyond defining names.
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
"""Layers that act as activation functions."""
# pylint: disable=g-classes-have-attributes

from tensorflow.python.framework import dtypes
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import math_ops


def get_globals():
  # [FUNCTION DOC]
  # HELPER: gives other code (such as the layer serialization registry) access to this module's
  # names. Takes no inputs and returns the module's mapping of global names to values (the live
  # mapping, not a copy).
  return globals()


class LeakyReLU(Layer):
  # [CLASS DOC]
  # Layer applying the leaky rectified linear unit: inputs below zero are multiplied by a small
  # slope, others pass through. Holds the slope (alpha) as a backend-float value. Supports masking.
  """Leaky version of a Rectified Linear Unit.

  It allows a small gradient when the unit is not active:

  ```
    f(x) = alpha * x if x < 0
    f(x) = x if x >= 0
  ```

  Usage:

  >>> layer = tf.keras.layers.LeakyReLU()
  >>> output = layer([-3.0, -1.0, 0.0, 2.0])
  >>> list(output.numpy())
  [-0.9, -0.3, 0.0, 2.0]
  >>> layer = tf.keras.layers.LeakyReLU(alpha=0.1)
  >>> output = layer([-3.0, -1.0, 0.0, 2.0])
  >>> list(output.numpy())
  [-0.3, -0.1, 0.0, 2.0]

  Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the batch axis)
    when using this layer as the first layer in a model.

  Output shape:
    Same shape as the input.

  Args:
    alpha: Float >= 0. Negative slope coefficient. Default to 0.3.

  """

  def __init__(self, alpha=0.3, **kwargs):
    # [FUNCTION DOC]
    # Constructor. Inputs: alpha, the negative slope (default 0.3; must not be null); extra
    # keyword arguments go to the base layer. Raises a value error if alpha is null. Marks the
    # layer as supporting masking and stores alpha cast to the backend float type.
    super(LeakyReLU, self).__init__(**kwargs)
    if alpha is None:
      raise ValueError('The alpha value of a Leaky ReLU layer '
                       'cannot be None, needs a float. '
                       'Got %s' % alpha)
    self.supports_masking = True
    self.alpha = backend.cast_to_floatx(alpha)

  def call(self, inputs):
    # [FUNCTION DOC]
    # Applies the leaky rectifier to the input tensor with the stored slope and returns a tensor
    # of the same shape.
    return backend.relu(inputs, alpha=self.alpha)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the base layer configuration merged with the alpha value as a plain float.
    config = {'alpha': float(self.alpha)}
    base_config = super(LeakyReLU, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    # [FUNCTION DOC]
    # Output shape equals the input shape (shape conversion decorator allows shape objects).
    return input_shape


class PReLU(Layer):
  # [CLASS DOC]
  # Parametric rectified linear unit: like a leaky rectifier but the negative slope is a trainable
  # weight array with one entry per non-batch input position, optionally shared along chosen
  # axes. Holds the initializer, regularizer, constraint, the list of shared axes and, after
  # building, the alpha weight. Supports masking.
  """Parametric Rectified Linear Unit.

  It follows:

  ```
    f(x) = alpha * x for x < 0
    f(x) = x for x >= 0
  ```

  where `alpha` is a learned array with the same shape as x.

  Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the samples axis)
    when using this layer as the first layer in a model.

  Output shape:
    Same shape as the input.

  Args:
    alpha_initializer: Initializer function for the weights.
    alpha_regularizer: Regularizer for the weights.
    alpha_constraint: Constraint for the weights.
    shared_axes: The axes along which to share learnable
      parameters for the activation function.
      For example, if the incoming feature maps
      are from a 2D convolution
      with output shape `(batch, height, width, channels)`,
      and you wish to share parameters across space
      so that each filter only has one set of parameters,
      set `shared_axes=[1, 2]`.
  """

  def __init__(self,
               alpha_initializer='zeros',
               alpha_regularizer=None,
               alpha_constraint=None,
               shared_axes=None,
               **kwargs):
    # [FUNCTION DOC]
    # Constructor. Inputs: initializer for alpha (name or object, default zeros), optional
    # regularizer, optional constraint, shared axes (none, a single axis or a list/tuple of
    # axes, counted with the batch axis as zero), and extra keyword arguments for the base
    # layer. Resolves the initializer, regularizer and constraint to objects, normalizes shared
    # axes to a list or none, and marks the layer as supporting masking.
    super(PReLU, self).__init__(**kwargs)
    self.supports_masking = True
    self.alpha_initializer = initializers.get(alpha_initializer)
    self.alpha_regularizer = regularizers.get(alpha_regularizer)
    self.alpha_constraint = constraints.get(alpha_constraint)
    if shared_axes is None:
      self.shared_axes = None
    elif not isinstance(shared_axes, (list, tuple)):
      self.shared_axes = [shared_axes]
    else:
      self.shared_axes = list(shared_axes)

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    # [FUNCTION DOC]
    # Creates the alpha weight. Input: the full input shape including batch. The weight shape is
    # the input shape without the batch dimension, with size one on each shared axis. Registers
    # the weight with the initializer, regularizer and constraint, sets an input spec requiring
    # the same rank and fixing the sizes of non-shared axes, and marks the layer built.
    param_shape = list(input_shape[1:])
    if self.shared_axes is not None:
      for i in self.shared_axes:
        param_shape[i - 1] = 1
    self.alpha = self.add_weight(
        shape=param_shape,
        name='alpha',
        initializer=self.alpha_initializer,
        regularizer=self.alpha_regularizer,
        constraint=self.alpha_constraint)
    # Set input spec
    axes = {}
    if self.shared_axes:
      for i in range(1, len(input_shape)):
        if i not in self.shared_axes:
          axes[i] = input_shape[i]
    self.input_spec = InputSpec(ndim=len(input_shape), axes=axes)
    self.built = True

  def call(self, inputs):
    # [FUNCTION DOC]
    # Computes the positive part of the input plus alpha times the negative part, broadcasting
    # alpha over the shared axes. Returns a tensor of the input shape.
    pos = backend.relu(inputs)
    neg = -self.alpha * backend.relu(-inputs)
    return pos + neg

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the base configuration merged with serialized initializer, regularizer and
    # constraint, and the shared axes list.
    config = {
        'alpha_initializer': initializers.serialize(self.alpha_initializer),
        'alpha_regularizer': regularizers.serialize(self.alpha_regularizer),
        'alpha_constraint': constraints.serialize(self.alpha_constraint),
        'shared_axes': self.shared_axes
    }
    base_config = super(PReLU, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    # [FUNCTION DOC]
    # Output shape equals the input shape.
    return input_shape


class ELU(Layer):
  # [CLASS DOC]
  # Exponential linear unit layer: negative inputs map to alpha times (exp(x) minus one), others
  # pass through. Holds alpha as a backend-float value. Supports masking.
  """Exponential Linear Unit.

  It follows:

  ```
    f(x) =  alpha * (exp(x) - 1.) for x < 0
    f(x) = x for x >= 0
  ```

  Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the samples axis)
    when using this layer as the first layer in a model.

  Output shape:
    Same shape as the input.

  Args:
    alpha: Scale for the negative factor.
  """

  def __init__(self, alpha=1.0, **kwargs):
    # [FUNCTION DOC]
    # Constructor. Inputs: alpha scale for negatives (default 1.0; must not be null) and extra
    # keyword arguments for the base layer. Raises a value error if alpha is null. Marks the
    # layer as supporting masking and stores alpha cast to the backend float type.
    super(ELU, self).__init__(**kwargs)
    if alpha is None:
      raise ValueError('Alpha of an ELU layer cannot be None, '
                       'requires a float. Got %s' % alpha)
    self.supports_masking = True
    self.alpha = backend.cast_to_floatx(alpha)

  def call(self, inputs):
    # [FUNCTION DOC]
    # Applies the backend exponential linear unit with the stored alpha; same shape as input.
    return backend.elu(inputs, self.alpha)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the base layer configuration merged with alpha as a plain float.
    config = {'alpha': float(self.alpha)}
    base_config = super(ELU, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    # [FUNCTION DOC]
    # Output shape equals the input shape.
    return input_shape


class ThresholdedReLU(Layer):
  # [CLASS DOC]
  # Layer that passes inputs strictly greater than a threshold (theta) and outputs zero
  # otherwise. Holds theta as a backend-float value. Supports masking.
  """Thresholded Rectified Linear Unit.

  It follows:

  ```
    f(x) = x for x > theta
    f(x) = 0 otherwise`
  ```

  Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the samples axis)
    when using this layer as the first layer in a model.

  Output shape:
    Same shape as the input.

  Args:
    theta: Float >= 0. Threshold location of activation.
  """

  def __init__(self, theta=1.0, **kwargs):
    # [FUNCTION DOC]
    # Constructor. Inputs: theta, the threshold (default 1.0) and extra keyword arguments for the
    # base layer. Raises a value error if theta is null or negative. Marks the layer as
    # supporting masking and stores theta cast to the backend float type.
    super(ThresholdedReLU, self).__init__(**kwargs)
    if theta is None:
      raise ValueError('Theta of a Thresholded ReLU layer cannot be '
                       'None, requires a float. Got %s' % theta)
    if theta < 0:
      raise ValueError('The theta value of a Thresholded ReLU layer '
                       'should be >=0, got %s' % theta)
    self.supports_masking = True
    self.theta = backend.cast_to_floatx(theta)

  def call(self, inputs):
    # [FUNCTION DOC]
    # Casts theta to the input dtype and returns the input multiplied by a mask of positions
    # strictly greater than theta, so other positions become zero. Same shape as input.
    theta = math_ops.cast(self.theta, inputs.dtype)
    return inputs * math_ops.cast(math_ops.greater(inputs, theta), inputs.dtype)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the base layer configuration merged with theta as a plain float.
    config = {'theta': float(self.theta)}
    base_config = super(ThresholdedReLU, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    # [FUNCTION DOC]
    # Output shape equals the input shape.
    return input_shape


def _large_compatible_negative(tensor_type):
  # [FUNCTION DOC]
  # HELPER: provides the masking offset for the Softmax layer in this module.
  # KNOWN ISSUE: the docstring says it returns a Tensor, but it returns a plain number (the
  # float16 minimum value, or minus one billion for other types).
  # Input: a data type. Returns the lowest float16 value for float16, because minus one billion
  # does not fit in that type, and minus one billion otherwise.
  """Large negative number as Tensor.

  This function is necessary because the standard value for epsilon
  in this module (-1e9) cannot be represented using tf.float16

  Args:
    tensor_type: a dtype to determine the type.

  Returns:
    a large negative number.
  """
  if tensor_type == dtypes.float16:
    return dtypes.float16.min
  return -1e9


class Softmax(Layer):
  # [CLASS DOC]
  # Softmax activation layer, normalizing along one axis or several axes, with optional masking
  # at call time. Holds the axis setting (default last axis). Supports masking.
  """Softmax activation function.

  Example without mask:

  >>> inp = np.asarray([1., 2., 1.])
  >>> layer = tf.keras.layers.Softmax()
  >>> layer(inp).numpy()
  array([0.21194157, 0.5761169 , 0.21194157], dtype=float32)
  >>> mask = np.asarray([True, False, True], dtype=bool)
  >>> layer(inp, mask).numpy()
  array([0.5, 0. , 0.5], dtype=float32)

  Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the samples axis)
    when using this layer as the first layer in a model.

  Output shape:
    Same shape as the input.

  Args:
    axis: Integer, or list of Integers, axis along which the softmax
      normalization is applied.
  Call arguments:
    inputs: The inputs, or logits to the softmax layer.
    mask: A boolean mask of the same shape as `inputs`. Defaults to `None`. The
      mask specifies 1 to keep and 0 to mask.

  Returns:
    softmaxed output with the same shape as `inputs`.
  """

  def __init__(self, axis=-1, **kwargs):
    # [FUNCTION DOC]
    # Constructor. Inputs: axis, an integer or list/tuple of integers (default minus one) and
    # extra keyword arguments for the base layer. Marks the layer as supporting masking and
    # stores the axis.
    super(Softmax, self).__init__(**kwargs)
    self.supports_masking = True
    self.axis = axis

  def call(self, inputs, mask=None):
    # [FUNCTION DOC]
    # Inputs: the logits tensor and an optional mask (nonzero keeps, zero masks, same shape).
    # If a mask is given, a large negative number is added at masked positions so they get
    # essentially zero probability. If the axis is a list or tuple with more than one entry,
    # computes the exponential of the inputs minus their log-sum-exp over those axes; with a
    # single entry or an integer it uses the backend softmax on that axis. Returns a tensor of
    # the input shape.
    if mask is not None:
      # Since mask is 1.0 for positions we want to keep and 0.0 for
      # masked positions, this operation will create a tensor which is 0.0 for
      # positions we want to attend and -1e.9 for masked positions.
      adder = (1.0 - math_ops.cast(mask, inputs.dtype)) * (
          _large_compatible_negative(inputs.dtype))

      # Since we are adding it to the raw scores before the softmax, this is
      # effectively the same as removing these entirely.
      inputs += adder
    if isinstance(self.axis, (tuple, list)):
      if len(self.axis) > 1:
        return math_ops.exp(inputs - math_ops.reduce_logsumexp(
            inputs, axis=self.axis, keepdims=True))
      else:
        return backend.softmax(inputs, axis=self.axis[0])
    return backend.softmax(inputs, axis=self.axis)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the base layer configuration merged with the axis setting.
    config = {'axis': self.axis}
    base_config = super(Softmax, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    # [FUNCTION DOC]
    # Output shape equals the input shape.
    return input_shape


class ReLU(Layer):
  # [CLASS DOC]
  # Rectified linear unit layer with optional maximum value, negative slope and threshold.
  # Holds these three settings as backend-float values (the maximum may be null for no cap).
  # Supports masking.
  """Rectified Linear Unit activation function.

  With default values, it returns element-wise `max(x, 0)`.

  Otherwise, it follows:

  ```
    f(x) = max_value if x >= max_value
    f(x) = x if threshold <= x < max_value
    f(x) = negative_slope * (x - threshold) otherwise
  ```

  Usage:

  >>> layer = tf.keras.layers.ReLU()
  >>> output = layer([-3.0, -1.0, 0.0, 2.0])
  >>> list(output.numpy())
  [0.0, 0.0, 0.0, 2.0]
  >>> layer = tf.keras.layers.ReLU(max_value=1.0)
  >>> output = layer([-3.0, -1.0, 0.0, 2.0])
  >>> list(output.numpy())
  [0.0, 0.0, 0.0, 1.0]
  >>> layer = tf.keras.layers.ReLU(negative_slope=1.0)
  >>> output = layer([-3.0, -1.0, 0.0, 2.0])
  >>> list(output.numpy())
  [-3.0, -1.0, 0.0, 2.0]
  >>> layer = tf.keras.layers.ReLU(threshold=1.5)
  >>> output = layer([-3.0, -1.0, 1.0, 2.0])
  >>> list(output.numpy())
  [0.0, 0.0, 0.0, 2.0]

  Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the batch axis)
    when using this layer as the first layer in a model.

  Output shape:
    Same shape as the input.

  Args:
    max_value: Float >= 0. Maximum activation value. Default to None, which
      means unlimited.
    negative_slope: Float >= 0. Negative slope coefficient. Default to 0.
    threshold: Float >= 0. Threshold value for thresholded activation. Default
      to 0.
  """

  def __init__(self, max_value=None, negative_slope=0, threshold=0, **kwargs):
    # [FUNCTION DOC]
    # Constructor. Inputs: max_value (null for unlimited, default; otherwise not negative),
    # negative_slope (default 0, not null or negative), threshold (default 0, not null or
    # negative), and extra keyword arguments for the base layer. Raises a value error for any
    # violated constraint. Marks the layer as supporting masking and stores the values cast to
    # the backend float type.
    super(ReLU, self).__init__(**kwargs)
    if max_value is not None and max_value < 0.:
      raise ValueError('max_value of a ReLU layer cannot be a negative '
                       'value. Got: %s' % max_value)
    if negative_slope is None or negative_slope < 0.:
      raise ValueError('negative_slope of a ReLU layer cannot be a negative '
                       'value. Got: %s' % negative_slope)
    if threshold is None or threshold < 0.:
      raise ValueError('threshold of a ReLU layer cannot be a negative '
                       'value. Got: %s' % threshold)

    self.supports_masking = True
    if max_value is not None:
      max_value = backend.cast_to_floatx(max_value)
    self.max_value = max_value
    self.negative_slope = backend.cast_to_floatx(negative_slope)
    self.threshold = backend.cast_to_floatx(threshold)

  def call(self, inputs):
    # [FUNCTION DOC]
    # Applies the backend rectifier with the stored negative slope, maximum and threshold and
    # returns a tensor of the input shape.
    # alpha is used for leaky relu slope in activations instead of
    # negative_slope.
    return backend.relu(inputs,
                        alpha=self.negative_slope,
                        max_value=self.max_value,
                        threshold=self.threshold)

  def get_config(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: unlike the other layers in this module, the stored values are returned as
    # they are (backend-float numpy values), not converted to plain floats, so the configuration
    # may not be directly serializable to text formats.
    # Returns the base layer configuration merged with maximum value, negative slope and threshold.
    config = {
        'max_value': self.max_value,
        'negative_slope': self.negative_slope,
        'threshold': self.threshold
    }
    base_config = super(ReLU, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    # [FUNCTION DOC]
    # Output shape equals the input shape.
    return input_shape
