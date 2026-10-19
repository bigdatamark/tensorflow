# [MODULE DOC]
# Core Keras layers (legacy tf.python.keras copy). Defines the basic building-block layers:
# Masking, Dropout and its spatial 1D/2D/3D variants, Activation, Reshape, Permute, Flatten,
# RepeatVector, Lambda, Dense and ActivityRegularization.
# It also defines the machinery that lets plain TensorFlow ops be applied to symbolic Keras tensors
# during functional-model construction: TFOpLambda, SlicingOpLambda, InstanceProperty,
# InstanceMethod, ClassMethod, and the dispatcher classes that intercept TF ops and class methods.
# Import-time side effects: registers a global op dispatcher (KerasOpDispatcher), registers slicing
# dispatchers for tensor indexing, boolean masking and ragged indexing, registers class-method
# dispatchers for ragged tensor constructors, and attaches delegating properties and methods to the
# ragged and sparse symbolic tensor classes.
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
"""Core Keras layers."""

import copy
import functools
import operator
import sys
import textwrap
import types as python_types
import warnings

import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine import keras_tensor
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import tensor_getitem_override
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.ragged import ragged_getitem
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.tf_export import get_canonical_name_for_symbol
from tensorflow.python.util.tf_export import get_symbol_from_name


# pylint: disable=g-classes-have-attributes
class Masking(Layer):
  # [CLASS DOC]
  # Layer that produces a boolean timestep mask: a timestep is masked when every feature at that
  # timestep equals the configured mask value. Holds the mask value; has no weights. The output
  # equals the input with masked timesteps zeroed, and the mask is attached to the output so that
  # downstream mask-aware layers skip those steps. Supports masking itself.
  """Masks a sequence by using a mask value to skip timesteps.

  For each timestep in the input tensor (dimension #1 in the tensor),
  if all values in the input tensor at that timestep
  are equal to `mask_value`, then the timestep will be masked (skipped)
  in all downstream layers (as long as they support masking).

  If any downstream layer does not support masking yet receives such
  an input mask, an exception will be raised.

  Example:

  Consider a Numpy data array `x` of shape `(samples, timesteps, features)`,
  to be fed to an LSTM layer. You want to mask timestep #3 and #5 because you
  lack data for these timesteps. You can:

  - Set `x[:, 3, :] = 0.` and `x[:, 5, :] = 0.`
  - Insert a `Masking` layer with `mask_value=0.` before the LSTM layer:

  ```python
  samples, timesteps, features = 32, 10, 8
  inputs = np.random.random([samples, timesteps, features]).astype(np.float32)
  inputs[:, 3, :] = 0.
  inputs[:, 5, :] = 0.

  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Masking(mask_value=0.,
                                    input_shape=(timesteps, features)))
  model.add(tf.keras.layers.LSTM(32))

  output = model(inputs)
  # The time step 3 and 5 will be skipped from LSTM calculation.
  ```

  See [the masking and padding guide](
    https://www.tensorflow.org/guide/keras/masking_and_padding)
  for more details.
  """

  def __init__(self, mask_value=0., **kwargs):
    # [FUNCTION DOC]
    # Constructor. Takes mask_value (number, default zero): the value that marks a padded timestep,
    # plus generic layer keyword arguments passed to the base layer. Marks the layer as supporting
    # masking and as computing outputs and mask jointly. Returns nothing.
    super(Masking, self).__init__(**kwargs)
    self.supports_masking = True
    self.mask_value = mask_value
    self._compute_output_and_mask_jointly = True

  def compute_mask(self, inputs, mask=None):
    # [FUNCTION DOC]
    # Returns a boolean tensor, one entry per (sample, timestep), that is true when at least one
    # feature of that timestep differs from the mask value. Input mask argument is ignored
    # (existing upstream masks are not combined). No side effects.
    return K.any(math_ops.not_equal(inputs, self.mask_value), axis=-1)

  def call(self, inputs):
    # [FUNCTION DOC]
    # Forward pass. Computes the per-timestep boolean mask, multiplies the inputs by the mask
    # (cast to the input type) so masked timesteps become zero, and stores the squeezed mask on the
    # output tensor as its Keras mask attribute (side effect on the returned tensor).
    # Returns the masked tensor, same shape as the input.
    boolean_mask = K.any(
        math_ops.not_equal(inputs, self.mask_value), axis=-1, keepdims=True)
    outputs = inputs * math_ops.cast(boolean_mask, inputs.dtype)
    # Compute the mask and outputs simultaneously.
    outputs._keras_mask = array_ops.squeeze(boolean_mask, axis=-1)  # pylint: disable=protected-access
    return outputs

  def compute_output_shape(self, input_shape):
    # [FUNCTION DOC]
    # Output shape equals the input shape; returns the given shape unchanged.
    return input_shape

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a serializable mapping: the base layer configuration plus the mask value.
    config = {'mask_value': self.mask_value}
    base_config = super(Masking, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class Dropout(Layer):
  # [CLASS DOC]
  # Regularization layer that, in training mode only, randomly zeroes a fraction (rate) of input
  # units and rescales the rest by one over one minus the rate. In inference mode it is the
  # identity. Holds rate, optional noise shape (shape of the random mask, allowing mask sharing
  # across axes) and optional random seed. Subclasses override the noise-shape hook (the spatial
  # dropout layers do). Has no weights and supports masking.
  """Applies Dropout to the input.

  The Dropout layer randomly sets input units to 0 with a frequency of `rate`
  at each step during training time, which helps prevent overfitting.
  Inputs not set to 0 are scaled up by 1/(1 - rate) such that the sum over
  all inputs is unchanged.

  Note that the Dropout layer only applies when `training` is set to True
  such that no values are dropped during inference. When using `model.fit`,
  `training` will be appropriately set to True automatically, and in other
  contexts, you can set the kwarg explicitly to True when calling the layer.

  (This is in contrast to setting `trainable=False` for a Dropout layer.
  `trainable` does not affect the layer's behavior, as Dropout does
  not have any variables/weights that can be frozen during training.)

  >>> tf.random.set_seed(0)
  >>> layer = tf.keras.layers.Dropout(.2, input_shape=(2,))
  >>> data = np.arange(10).reshape(5, 2).astype(np.float32)
  >>> print(data)
  [[0. 1.]
   [2. 3.]
   [4. 5.]
   [6. 7.]
   [8. 9.]]
  >>> outputs = layer(data, training=True)
  >>> print(outputs)
  tf.Tensor(
  [[ 0.    1.25]
   [ 2.5   3.75]
   [ 5.    6.25]
   [ 7.5   8.75]
   [10.    0.  ]], shape=(5, 2), dtype=float32)

  Args:
    rate: Float between 0 and 1. Fraction of the input units to drop.
    noise_shape: 1D integer tensor representing the shape of the
      binary dropout mask that will be multiplied with the input.
      For instance, if your inputs have shape
      `(batch_size, timesteps, features)` and
      you want the dropout mask to be the same for all timesteps,
      you can use `noise_shape=(batch_size, 1, features)`.
    seed: A Python integer to use as random seed.

  Call arguments:
    inputs: Input tensor (of any rank).
    training: Python boolean indicating whether the layer should behave in
      training mode (adding dropout) or in inference mode (doing nothing).
  """

  def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
    # [FUNCTION DOC]
    # Constructor. rate: fraction of units to drop. noise_shape: optional sequence of integers or
    # nothing values describing the dropout mask shape (nothing entries are filled from the actual
    # input shape at call time). seed: optional integer random seed. Extra keyword arguments go to
    # the base layer. Raises a value error when rate is a plain number outside zero to one
    # (non-numeric rates such as tensors are not validated). Enables masking support.
    super(Dropout, self).__init__(**kwargs)
    if isinstance(rate, (int, float)) and not 0 <= rate <= 1:
      raise ValueError(f'Invalid value {rate} received for '
                       f'`rate`, expected a value between 0 and 1.')
    self.rate = rate
    self.noise_shape = noise_shape
    self.seed = seed
    self.supports_masking = True

  def _get_noise_shape(self, inputs):
    # [FUNCTION DOC]
    # HELPER: Dropout.call helper that builds the mask shape; overridden by spatial dropout layers.
    # Returns nothing/null value when no noise shape was configured. Otherwise returns a tensor
    # shape where each null entry of the configured noise shape is replaced by the corresponding
    # dynamic dimension of the given inputs.
    # Subclasses of `Dropout` may implement `_get_noise_shape(self, inputs)`,
    # which will override `self.noise_shape`, and allows for custom noise
    # shapes with dynamically sized inputs.
    if self.noise_shape is None:
      return None

    concrete_inputs_shape = array_ops.shape(inputs)
    noise_shape = []
    for i, value in enumerate(self.noise_shape):
      noise_shape.append(concrete_inputs_shape[i] if value is None else value)
    return tensor_conversion.convert_to_tensor_v2_with_dispatch(noise_shape)

  def call(self, inputs, training=None):
    # [FUNCTION DOC]
    # Forward pass. inputs: tensor of any rank. training: boolean (or tensor) selecting training
    # mode; when nothing it falls back to the global Keras learning phase. In training mode applies
    # dropout with the layer's rate, noise shape and seed; otherwise returns an identity copy.
    # Returns a tensor shaped like the input. Uses random ops (consumes randomness) in training.
    if training is None:
      training = K.learning_phase()

    def dropped_inputs():
      # [FUNCTION DOC]
      # HELPER: training-branch callback for the conditional in Dropout.call.
      # Applies dropout to the enclosing inputs using the layer's noise shape, seed and rate and
      # returns the resulting tensor.
      return nn.dropout(
          inputs,
          noise_shape=self._get_noise_shape(inputs),
          seed=self.seed,
          rate=self.rate)

    output = control_flow_util.smart_cond(training, dropped_inputs,
                                          lambda: array_ops.identity(inputs))
    return output

  def compute_output_shape(self, input_shape):
    # [FUNCTION DOC]
    # Output shape equals the input shape; returns the given shape unchanged.
    return input_shape

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a serializable mapping: the base layer configuration plus rate, noise shape and seed.
    config = {
        'rate': self.rate,
        'noise_shape': self.noise_shape,
        'seed': self.seed
    }
    base_config = super(Dropout, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class SpatialDropout1D(Dropout):
  # [CLASS DOC]
  # Dropout variant for three-dimensional inputs (samples, timesteps, channels) that drops whole
  # channels: the random mask is shared across the timestep axis. Subclass of Dropout overriding
  # only the noise-shape computation; restricts input rank to exactly three.
  """Spatial 1D version of Dropout.

  This version performs the same function as Dropout, however, it drops
  entire 1D feature maps instead of individual elements. If adjacent frames
  within feature maps are strongly correlated (as is normally the case in
  early convolution layers) then regular dropout will not regularize the
  activations and will otherwise just result in an effective learning rate
  decrease. In this case, SpatialDropout1D will help promote independence
  between feature maps and should be used instead.

  Args:
    rate: Float between 0 and 1. Fraction of the input units to drop.

  Call arguments:
    inputs: A 3D tensor.
    training: Python boolean indicating whether the layer should behave in
      training mode (adding dropout) or in inference mode (doing nothing).

  Input shape:
    3D tensor with shape:
    `(samples, timesteps, channels)`

  Output shape:
    Same as input.

  References:
    - [Efficient Object Localization Using Convolutional
      Networks](https://arxiv.org/abs/1411.4280)
  """

  def __init__(self, rate, **kwargs):
    # [FUNCTION DOC]
    # Constructor. rate: fraction of channels to drop (validated by the Dropout base constructor).
    # Extra keyword arguments go to the base layer. Sets an input specification requiring rank three.
    super(SpatialDropout1D, self).__init__(rate, **kwargs)
    self.input_spec = InputSpec(ndim=3)

  def _get_noise_shape(self, inputs):
    # [FUNCTION DOC]
    # HELPER: overrides the Dropout mask-shape hook. Returns a tuple of dynamic batch size, one,
    # and dynamic channel count so the mask is shared over timesteps.
    input_shape = array_ops.shape(inputs)
    noise_shape = (input_shape[0], 1, input_shape[2])
    return noise_shape


class SpatialDropout2D(Dropout):
  # [CLASS DOC]
  # Dropout variant for four-dimensional image-like inputs that drops entire two-dimensional
  # feature maps (channels) by sharing the mask across the spatial axes. Holds the data format
  # (channels first or last; defaults to the global Keras image data format). Restricts input rank
  # to four. Overrides the noise-shape hook of Dropout.
  """Spatial 2D version of Dropout.

  This version performs the same function as Dropout, however, it drops
  entire 2D feature maps instead of individual elements. If adjacent pixels
  within feature maps are strongly correlated (as is normally the case in
  early convolution layers) then regular dropout will not regularize the
  activations and will otherwise just result in an effective learning rate
  decrease. In this case, SpatialDropout2D will help promote independence
  between feature maps and should be used instead.

  Args:
    rate: Float between 0 and 1. Fraction of the input units to drop.
    data_format: 'channels_first' or 'channels_last'.
      In 'channels_first' mode, the channels dimension
      (the depth) is at index 1,
      in 'channels_last' mode is it at index 3.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".

  Call arguments:
    inputs: A 4D tensor.
    training: Python boolean indicating whether the layer should behave in
      training mode (adding dropout) or in inference mode (doing nothing).

  Input shape:
    4D tensor with shape:
    `(samples, channels, rows, cols)` if data_format='channels_first'
    or 4D tensor with shape:
    `(samples, rows, cols, channels)` if data_format='channels_last'.

  Output shape:
    Same as input.

  References:
    - [Efficient Object Localization Using Convolutional
      Networks](https://arxiv.org/abs/1411.4280)
  """

  def __init__(self, rate, data_format=None, **kwargs):
    # [FUNCTION DOC]
    # Constructor. rate: fraction of feature maps to drop. data_format: channels-first or
    # channels-last string, or nothing to use the global Keras default. Extra keyword arguments go
    # to the base layer. Raises a value error for any other data format string. Sets an input
    # specification requiring rank four.
    super(SpatialDropout2D, self).__init__(rate, **kwargs)
    if data_format is None:
      data_format = K.image_data_format()
    if data_format not in {'channels_last', 'channels_first'}:
      raise ValueError('data_format must be in '
                       '{"channels_last", "channels_first"}')
    self.data_format = data_format
    self.input_spec = InputSpec(ndim=4)

  def _get_noise_shape(self, inputs):
    # [FUNCTION DOC]
    # HELPER: overrides the Dropout mask-shape hook. Returns a tuple of dynamic batch size and
    # channel count with ones on the two spatial axes (position of channels depends on the data
    # format), so the mask is shared across rows and columns.
    input_shape = array_ops.shape(inputs)
    if self.data_format == 'channels_first':
      return (input_shape[0], input_shape[1], 1, 1)
    elif self.data_format == 'channels_last':
      return (input_shape[0], 1, 1, input_shape[3])


class SpatialDropout3D(Dropout):
  # [CLASS DOC]
  # Dropout variant for five-dimensional volumetric inputs that drops entire three-dimensional
  # feature maps by sharing the mask across the three spatial axes. Holds the data format (channels
  # first or last, default from global Keras config). Restricts input rank to five. Overrides the
  # noise-shape hook of Dropout.
  """Spatial 3D version of Dropout.

  This version performs the same function as Dropout, however, it drops
  entire 3D feature maps instead of individual elements. If adjacent voxels
  within feature maps are strongly correlated (as is normally the case in
  early convolution layers) then regular dropout will not regularize the
  activations and will otherwise just result in an effective learning rate
  decrease. In this case, SpatialDropout3D will help promote independence
  between feature maps and should be used instead.

  Args:
    rate: Float between 0 and 1. Fraction of the input units to drop.
    data_format: 'channels_first' or 'channels_last'.
        In 'channels_first' mode, the channels dimension (the depth)
        is at index 1, in 'channels_last' mode is it at index 4.
        It defaults to the `image_data_format` value found in your
        Keras config file at `~/.keras/keras.json`.
        If you never set it, then it will be "channels_last".

  Call arguments:
    inputs: A 5D tensor.
    training: Python boolean indicating whether the layer should behave in
      training mode (adding dropout) or in inference mode (doing nothing).

  Input shape:
    5D tensor with shape:
    `(samples, channels, dim1, dim2, dim3)` if data_format='channels_first'
    or 5D tensor with shape:
    `(samples, dim1, dim2, dim3, channels)` if data_format='channels_last'.

  Output shape:
    Same as input.

  References:
    - [Efficient Object Localization Using Convolutional
      Networks](https://arxiv.org/abs/1411.4280)
  """

  def __init__(self, rate, data_format=None, **kwargs):
    # [FUNCTION DOC]
    # Constructor. rate: fraction of feature maps to drop. data_format: channels-first or
    # channels-last string, or nothing to use the global Keras default. Extra keyword arguments go
    # to the base layer. Raises a value error for any other data format string. Sets an input
    # specification requiring rank five.
    super(SpatialDropout3D, self).__init__(rate, **kwargs)
    if data_format is None:
      data_format = K.image_data_format()
    if data_format not in {'channels_last', 'channels_first'}:
      raise ValueError('data_format must be in '
                       '{"channels_last", "channels_first"}')
    self.data_format = data_format
    self.input_spec = InputSpec(ndim=5)

  def _get_noise_shape(self, inputs):
    # [FUNCTION DOC]
    # HELPER: overrides the Dropout mask-shape hook. Returns a tuple of dynamic batch size and
    # channel count with ones on the three spatial axes (channel position depends on the data
    # format), so the mask is shared across all spatial positions.
    input_shape = array_ops.shape(inputs)
    if self.data_format == 'channels_first':
      return (input_shape[0], input_shape[1], 1, 1, 1)
    elif self.data_format == 'channels_last':
      return (input_shape[0], 1, 1, 1, input_shape[4])


class Activation(Layer):
  # [CLASS DOC]
  # Stateless layer applying an element-wise activation function to its input. Holds the resolved
  # activation callable (looked up from a name or accepted as a callable). Supports masking and
  # preserves shape.
  """Applies an activation function to an output.

  Args:
    activation: Activation function, such as `tf.nn.relu`, or string name of
      built-in activation function, such as "relu".

  Usage:

  >>> layer = tf.keras.layers.Activation('relu')
  >>> output = layer([-3.0, -1.0, 0.0, 2.0])
  >>> list(output.numpy())
  [0.0, 0.0, 0.0, 2.0]
  >>> layer = tf.keras.layers.Activation(tf.nn.relu)
  >>> output = layer([-3.0, -1.0, 0.0, 2.0])
  >>> list(output.numpy())
  [0.0, 0.0, 0.0, 2.0]

  Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the batch axis)
    when using this layer as the first layer in a model.

  Output shape:
    Same shape as input.
  """

  def __init__(self, activation, **kwargs):
    # [FUNCTION DOC]
    # Constructor. activation: activation function or its registered string name; resolved through
    # the activations registry (unknown names raise an error there). Extra keyword arguments go to
    # the base layer. Enables masking support.
    super(Activation, self).__init__(**kwargs)
    self.supports_masking = True
    self.activation = activations.get(activation)

  def call(self, inputs):
    # [FUNCTION DOC]
    # Forward pass: applies the stored activation function to the inputs and returns the result.
    return self.activation(inputs)

  def compute_output_shape(self, input_shape):
    # [FUNCTION DOC]
    # Output shape equals the input shape; returns the given shape unchanged.
    return input_shape

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a serializable mapping: the base layer configuration plus the serialized activation.
    config = {'activation': activations.serialize(self.activation)}
    base_config = super(Activation, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class Reshape(Layer):
  # [CLASS DOC]
  # Layer that reshapes each sample to a fixed target shape (batch dimension excluded). One target
  # dimension may be negative to be inferred from the input size. Holds the target shape as a
  # tuple. Has no weights.
  """Layer that reshapes inputs into the given shape.

  Input shape:
    Arbitrary, although all dimensions in the input shape must be known/fixed.
    Use the keyword argument `input_shape` (tuple of integers, does not include
    the samples/batch size axis) when using this layer as the first layer
    in a model.

  Output shape:
    `(batch_size,) + target_shape`

  Example:

  >>> # as first layer in a Sequential model
  >>> model = tf.keras.Sequential()
  >>> model.add(tf.keras.layers.Reshape((3, 4), input_shape=(12,)))
  >>> # model.output_shape == (None, 3, 4), `None` is the batch size.
  >>> model.output_shape
  (None, 3, 4)

  >>> # as intermediate layer in a Sequential model
  >>> model.add(tf.keras.layers.Reshape((6, 2)))
  >>> model.output_shape
  (None, 6, 2)

  >>> # also supports shape inference using `-1` as dimension
  >>> model.add(tf.keras.layers.Reshape((-1, 2, 2)))
  >>> model.output_shape
  (None, 3, 2, 2)
  """

  def __init__(self, target_shape, **kwargs):
    # [FUNCTION DOC]
    # Constructor. target_shape: sequence of integers giving the per-sample output shape (no batch
    # axis); converted to a tuple and stored. Extra keyword arguments go to the base layer.
    """Creates a `tf.keras.layers.Reshape`  layer instance.

    Args:
      target_shape: Target shape. Tuple of integers, does not include the
        samples dimension (batch size).
      **kwargs: Any additional layer keyword arguments.
    """
    super(Reshape, self).__init__(**kwargs)
    self.target_shape = tuple(target_shape)

  def _fix_unknown_dimension(self, input_shape, output_shape):
    # [FUNCTION DOC]
    # HELPER: shape inference helper for Reshape.compute_output_shape.
    # KNOWN ISSUE: the docstring says the unknown dimension is marked by -1, but the code treats any
    # negative value as unknown (and raises if two negatives are present), so e.g. -2 is accepted.
    # Inputs: input_shape, the known per-sample input dimensions; output_shape, desired dimensions
    # with at most one negative entry. Returns a list where the negative entry is replaced by the
    # input element count divided by the product of the other dimensions. Raises a value error if
    # more than one unknown dimension exists, or if the sizes are inconsistent (including when the
    # known product is zero or does not divide the input size).
    """Find and replace a missing dimension in an output shape.

    This is a near direct port of the internal Numpy function
    `_fix_unknown_dimension` in `numpy/core/src/multiarray/shape.c`

    Args:
      input_shape: Shape of array being reshaped
      output_shape: Desired shape of the array with at most
        a single -1 which indicates a dimension that should be
        derived from the input shape.

    Returns:
      The new output shape with a -1 replaced with its computed value.

    Raises:
      ValueError: If the total array size of the output_shape is
      different than the input_shape, or more than one unknown dimension
      is specified.
    """
    output_shape = list(output_shape)
    msg = ('total size of new array must be unchanged, '
           'input_shape = {}, output_shape = {}'
           .format(input_shape, output_shape))

    known, unknown = 1, None
    for index, dim in enumerate(output_shape):
      if dim < 0:
        if unknown is None:
          unknown = index
        else:
          raise ValueError('Can only specify one unknown dimension.')
      else:
        known *= dim

    original = np.prod(input_shape, dtype=int)
    if unknown is not None:
      if known == 0 or original % known != 0:
        raise ValueError(msg)
      output_shape[unknown] = original // known
    elif original != known:
      raise ValueError(msg)
    return output_shape

  def compute_output_shape(self, input_shape):
    # [FUNCTION DOC]
    # Computes the full output shape (batch dimension kept as in the input). If any non-batch input
    # dimension is unknown, negative target entries become unknown (nothing) dimensions; otherwise
    # the negative entry is resolved via the unknown-dimension helper, which may raise a value
    # error on size mismatch. Returns a tensor shape object.
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if None in input_shape[1:]:
      output_shape = [input_shape[0]]
      # input shape (partially) unknown? replace -1's with None's
      output_shape += tuple(s if s != -1 else None for s in self.target_shape)
    else:
      output_shape = [input_shape[0]]
      output_shape += self._fix_unknown_dimension(input_shape[1:],
                                                  self.target_shape)
    return tensor_shape.TensorShape(output_shape)

  def call(self, inputs):
    # [FUNCTION DOC]
    # Forward pass. Reshapes the inputs to the dynamic batch size followed by the target shape. In
    # graph (non-eager) mode it also re-applies the statically computed output shape to the result
    # so partially unknown dimensions are inferred. Returns the reshaped tensor.
    result = array_ops.reshape(
        inputs, (array_ops.shape(inputs)[0],) + self.target_shape)
    if not context.executing_eagerly():
      # Set the static shape for the result since it might lost during array_ops
      # reshape, eg, some `None` dim in the result could be inferred.
      result.set_shape(self.compute_output_shape(inputs.shape))
    return result

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a serializable mapping: the base layer configuration plus the target shape.
    config = {'target_shape': self.target_shape}
    base_config = super(Reshape, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class Permute(Layer):
  # [CLASS DOC]
  # Layer that reorders the non-batch axes of its input according to a fixed one-based permutation
  # pattern. Holds the pattern as a tuple and requires inputs of rank equal to pattern length plus
  # one. No weights.
  """Permutes the dimensions of the input according to a given pattern.

  Useful e.g. connecting RNNs and convnets.

  Example:

  ```python
  model = Sequential()
  model.add(Permute((2, 1), input_shape=(10, 64)))
  # now: model.output_shape == (None, 64, 10)
  # note: `None` is the batch dimension
  ```

  Args:
    dims: Tuple of integers. Permutation pattern does not include the
      samples dimension. Indexing starts at 1.
      For instance, `(2, 1)` permutes the first and second dimensions
      of the input.

  Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the samples axis)
    when using this layer as the first layer in a model.

  Output shape:
    Same as the input shape, but with the dimensions re-ordered according
    to the specified pattern.
  """

  def __init__(self, dims, **kwargs):
    # [FUNCTION DOC]
    # Constructor. dims: sequence of integers that must be a permutation of one through its length
    # (batch axis excluded). Extra keyword arguments go to the base layer. Raises a value error when
    # dims is not such a permutation. Sets an input specification requiring rank of length plus one.
    super(Permute, self).__init__(**kwargs)
    self.dims = tuple(dims)
    if sorted(dims) != list(range(1, len(dims) + 1)):
      raise ValueError(
          'Invalid permutation `dims` for Permute Layer: %s. '
          'The set of indices in `dims` must be consecutive and start from 1.' %
          (dims,))
    self.input_spec = InputSpec(ndim=len(self.dims) + 1)

  def compute_output_shape(self, input_shape):
    # [FUNCTION DOC]
    # Returns a tensor shape in which the non-batch dimensions of the input shape are reordered by
    # the permutation pattern (batch dimension untouched). Input is any shape-like value.
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    output_shape = copy.copy(input_shape)
    for i, dim in enumerate(self.dims):
      target_dim = input_shape[dim]
      output_shape[i + 1] = target_dim
    return tensor_shape.TensorShape(output_shape)

  def call(self, inputs):
    # [FUNCTION DOC]
    # Forward pass: transposes the inputs keeping the batch axis first and reordering the others by
    # the stored pattern. Returns the transposed tensor.
    return array_ops.transpose(inputs, perm=(0,) + self.dims)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a serializable mapping: the base layer configuration plus the permutation pattern.
    config = {'dims': self.dims}
    base_config = super(Permute, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class Flatten(Layer):
  # [CLASS DOC]
  # Layer that collapses all non-batch dimensions into one, keeping the batch size. Rank-one inputs
  # get an added trailing dimension of size one. Holds the normalized data format; for channels
  # first inputs the channel axis is moved last before flattening so ordering is consistent.
  # Requires at least rank one. No weights.
  """Flattens the input. Does not affect the batch size.

  Note: If inputs are shaped `(batch,)` without a feature axis, then
  flattening adds an extra channel dimension and output shape is `(batch, 1)`.

  Args:
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, ..., channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, ...)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".

  Example:

  >>> model = tf.keras.Sequential()
  >>> model.add(tf.keras.layers.Conv2D(64, 3, 3, input_shape=(3, 32, 32)))
  >>> model.output_shape
  (None, 1, 10, 64)

  >>> model.add(Flatten())
  >>> model.output_shape
  (None, 640)

  """

  def __init__(self, data_format=None, **kwargs):
    # [FUNCTION DOC]
    # Constructor. data_format: channels-first or channels-last string, or nothing for the global
    # Keras default; normalized (invalid values raise an error in the normalizer). Extra keyword
    # arguments go to the base layer. Sets input specification of minimum rank one and records
    # whether the format is channels first.
    super(Flatten, self).__init__(**kwargs)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.input_spec = InputSpec(min_ndim=1)
    self._channels_first = self.data_format == 'channels_first'

  def call(self, inputs):
    # [FUNCTION DOC]
    # Forward pass. For channels-first data of rank above one, first moves the channel axis to the
    # end. Then, in eager mode, reshapes to batch size by minus one; in graph mode adds a trailing
    # axis for rank-one input, or reshapes to the most statically-defined two-dimensional shape
    # available (fully known non-batch size, else known batch size, else fully dynamic).
    # Returns a rank-two tensor (batch, features).
    if self._channels_first:
      rank = inputs.shape.rank
      if rank and rank > 1:
        # Switch to channels-last format.
        permutation = [0]
        permutation.extend(range(2, rank))
        permutation.append(1)
        inputs = array_ops.transpose(inputs, perm=permutation)

    if context.executing_eagerly():
      # Full static shape is guaranteed to be available.
      # Performance: Using `constant_op` is much faster than passing a list.
      flattened_shape = constant_op.constant([inputs.shape[0], -1])
      return array_ops.reshape(inputs, flattened_shape)
    else:
      input_shape = inputs.shape
      rank = input_shape.rank
      if rank == 1:
        return array_ops.expand_dims_v2(inputs, axis=1)
      else:
        batch_dim = tensor_shape.dimension_value(input_shape[0])
        non_batch_dims = input_shape[1:]
        # Reshape in a way that preserves as much shape info as possible.
        if non_batch_dims.is_fully_defined():
          last_dim = int(functools.reduce(operator.mul, non_batch_dims))
          flattened_shape = constant_op.constant([-1, last_dim])
        elif batch_dim is not None:
          flattened_shape = constant_op.constant([int(batch_dim), -1])
        else:
          flattened_shape = [array_ops.shape_v2(inputs)[0], -1]
        return array_ops.reshape(inputs, flattened_shape)

  def compute_output_shape(self, input_shape):
    # [FUNCTION DOC]
    # Returns a tensor shape of batch size followed by the product of the remaining dimensions, or
    # an unknown dimension if any of them is unknown or zero (a zero dimension is treated like an
    # unknown one by the truthiness test). An empty input shape yields shape of one element.
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if not input_shape:
      output_shape = tensor_shape.TensorShape([1])
    else:
      output_shape = [input_shape[0]]
    if np.all(input_shape[1:]):
      output_shape += [np.prod(input_shape[1:], dtype=int)]
    else:
      output_shape += [None]
    return tensor_shape.TensorShape(output_shape)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the base layer configuration mapping with the data format added.
    config = super(Flatten, self).get_config()
    config.update({'data_format': self.data_format})
    return config


class RepeatVector(Layer):
  # [CLASS DOC]
  # Layer that repeats a two-dimensional (samples, features) input n times along a new axis,
  # producing (samples, n, features). Holds the repetition count; requires rank-two input.
  """Repeats the input n times.

  Example:

  ```python
  model = Sequential()
  model.add(Dense(32, input_dim=32))
  # now: model.output_shape == (None, 32)
  # note: `None` is the batch dimension

  model.add(RepeatVector(3))
  # now: model.output_shape == (None, 3, 32)
  ```

  Args:
    n: Integer, repetition factor.

  Input shape:
    2D tensor of shape `(num_samples, features)`.

  Output shape:
    3D tensor of shape `(num_samples, n, features)`.
  """

  def __init__(self, n, **kwargs):
    # [FUNCTION DOC]
    # Constructor. n: integer repetition factor. Extra keyword arguments go to the base layer.
    # Raises a type error if n is not an integer (the value is stored before the check, so a
    # partially built object exists only if the error is swallowed). Requires rank-two inputs.
    super(RepeatVector, self).__init__(**kwargs)
    self.n = n
    if not isinstance(n, int):
      raise TypeError(f'Expected an integer value for `n`, got {type(n)}.')
    self.input_spec = InputSpec(ndim=2)

  def compute_output_shape(self, input_shape):
    # [FUNCTION DOC]
    # Returns a tensor shape of batch size, repetition count, and feature count taken from the
    # two-dimensional input shape.
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    return tensor_shape.TensorShape([input_shape[0], self.n, input_shape[1]])

  def call(self, inputs):
    # [FUNCTION DOC]
    # Forward pass: repeats each sample's feature vector n times via the backend repeat operation
    # and returns a rank-three tensor.
    return K.repeat(inputs, self.n)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a serializable mapping: the base layer configuration plus the repetition count.
    config = {'n': self.n}
    base_config = super(RepeatVector, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class Lambda(Layer):
  # [CLASS DOC]
  # Layer wrapping an arbitrary user-supplied function (or lambda) so it can be used in Sequential
  # and functional models. Holds the function, optional output shape (tuple, callable or nothing),
  # optional mask (constant or callable) and extra keyword arguments passed to the function, and
  # whether the function accepts training and mask arguments. During calls it detects variables
  # created or touched inside the function and raises or warns, because such variables are not
  # tracked by the layer. Serialization stores function bytecode for lambdas (non-portable) and
  # names for named functions; deserialization of lambdas is unsafe for untrusted configs.
  """Wraps arbitrary expressions as a `Layer` object.

  The `Lambda` layer exists so that arbitrary expressions can be used
  as a `Layer` when constructing `Sequential`
  and Functional API models. `Lambda` layers are best suited for simple
  operations or quick experimentation. For more advanced use cases, follow
  [this guide](https://www.tensorflow.org/guide/keras/custom_layers_and_models)
  for subclassing `tf.keras.layers.Layer`.

  WARNING: `tf.keras.layers.Lambda` layers have (de)serialization limitations!

  The main reason to subclass `tf.keras.layers.Layer` instead of using a
  `Lambda` layer is saving and inspecting a Model. `Lambda` layers
  are saved by serializing the Python bytecode, which is fundamentally
  non-portable. They should only be loaded in the same environment where
  they were saved. Subclassed layers can be saved in a more portable way
  by overriding their `get_config` method. Models that rely on
  subclassed Layers are also often easier to visualize and reason about.

  Examples:

  ```python
  # add a x -> x^2 layer
  model.add(Lambda(lambda x: x ** 2))
  ```
  ```python
  # add a layer that returns the concatenation
  # of the positive part of the input and
  # the opposite of the negative part

  def antirectifier(x):
      x -= K.mean(x, axis=1, keepdims=True)
      x = K.l2_normalize(x, axis=1)
      pos = K.relu(x)
      neg = K.relu(-x)
      return K.concatenate([pos, neg], axis=1)

  model.add(Lambda(antirectifier))
  ```

  Variables:
    While it is possible to use Variables with Lambda layers, this practice is
    discouraged as it can easily lead to bugs. For instance, consider the
    following layer:

    ```python
      scale = tf.Variable(1.)
      scale_layer = tf.keras.layers.Lambda(lambda x: x * scale)
    ```

    Because scale_layer does not directly track the `scale` variable, it will
    not appear in `scale_layer.trainable_weights` and will therefore not be
    trained if `scale_layer` is used in a Model.

    A better pattern is to write a subclassed Layer:

    ```python
      class ScaleLayer(tf.keras.layers.Layer):
        def __init__(self):
          super(ScaleLayer, self).__init__()
          self.scale = tf.Variable(1.)

        def call(self, inputs):
          return inputs * self.scale
    ```

    In general, Lambda layers can be convenient for simple stateless
    computation, but anything more complex should use a subclass Layer instead.

  Args:
    function: The function to be evaluated. Takes input tensor as first
      argument.
    output_shape: Expected output shape from function. This argument can be
      inferred if not explicitly provided. Can be a tuple or function. If a
      tuple, it only specifies the first dimension onward;
      sample dimension is assumed either the same as the input: `output_shape =
        (input_shape[0], ) + output_shape` or, the input is `None` and
      the sample dimension is also `None`: `output_shape = (None, ) +
        output_shape` If a function, it specifies the entire shape as a function
        of the
      input shape: `output_shape = f(input_shape)`
    mask: Either None (indicating no masking) or a callable with the same
      signature as the `compute_mask` layer method, or a tensor that will be
      returned as output mask regardless of what the input is.
    arguments: Optional dictionary of keyword arguments to be passed to the
      function.

  Input shape:
    Arbitrary. Use the keyword argument input_shape (tuple of
    integers, does not include the samples axis) when using this layer as the
    first layer in a model.

  Output shape:
    Specified by `output_shape` argument
  """

  @trackable.no_automatic_dependency_tracking
  def __init__(self, function, output_shape=None, mask=None, arguments=None,
               **kwargs):
    # [FUNCTION DOC]
    # Constructor (decorated so attribute assignments are not tracked as trackable dependencies).
    # function: callable applied to the inputs. output_shape: optional tuple (without batch axis),
    # callable from input shape to full output shape, or nothing to infer automatically. mask:
    # optional constant mask or callable with the compute-mask signature; if given, the layer
    # declares masking support. arguments: optional mapping of extra keyword arguments passed to
    # the function on every call. Extra keyword arguments go to the base layer. Inspects the
    # function's parameter names to learn whether it accepts training and mask arguments.
    # Initializes the "already warned" flag used to warn only once about untracked variables.
    super(Lambda, self).__init__(**kwargs)

    self.arguments = arguments or {}
    self.function = function

    if mask is not None:
      self.supports_masking = True
    self.mask = mask
    self._output_shape = output_shape

    # Warning on every invocation will be quite irksome in Eager mode.
    self._already_warned = False

    function_args = tf_inspect.getfullargspec(function).args
    self._fn_expects_training_arg = 'training' in function_args
    self._fn_expects_mask_arg = 'mask' in function_args

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    # [FUNCTION DOC]
    # Computes the layer's output shape (decorated to convert shapes to and from tuples). If no
    # output shape was configured, tries the base-class automatic inference in eager mode and turns
    # a not-implemented failure into a not-implemented error asking the user to supply an output
    # shape. If the configured output shape is callable it is called with the input shape; otherwise
    # the configured shape is treated as excluding the batch axis and the batch size of the first
    # input is prepended (nothing when input shape is empty). Returns a tensor shape or nested
    # structure of them.
    if self._output_shape is None:
      # Make use of existing autocomputation but provide Lambda-specific
      # error message. This is always safe to run even when the outer context
      # is Graph mode because Lambda layers don't have side effects such as
      # `add_loss`.
      with context.eager_mode():
        try:
          return super(Lambda, self).compute_output_shape(input_shape)
        except NotImplementedError:
          raise NotImplementedError(
              'We could not automatically infer the shape of the Lambda\'s '
              'output. Please specify `output_shape` for this Lambda.')

    if callable(self._output_shape):
      output_shapes = self._output_shape(input_shape)
      return tf_utils.convert_shapes(output_shapes, to_tuples=False)

    # Output shapes are passed directly and don't include batch dimension.
    input_tensor_shape = tf_utils.convert_shapes(input_shape, to_tuples=False)
    batch_size = nest.flatten(input_tensor_shape)[0][0] if input_shape else None

    def _add_batch(shape):
      # [FUNCTION DOC]
      # HELPER: prepends the enclosing batch size to a per-sample tensor shape and returns the
      # new tensor shape; used with structure mapping in compute_output_shape.
      return tensor_shape.TensorShape([batch_size] + shape.as_list())

    output_shapes = tf_utils.convert_shapes(self._output_shape, to_tuples=False)
    return nest.map_structure(_add_batch, output_shapes)

  def call(self, inputs, mask=None, training=None):
    # [FUNCTION DOC]
    # Forward pass. Copies the stored extra arguments, adds the mask and training arguments only if
    # the wrapped function declared parameters with those names, then calls the function on the
    # inputs inside a gradient tape and a variable-creator scope that record variables created or
    # accessed. Afterwards validates them with the variable check helper (which may raise a value
    # error or log a warning). Returns whatever the wrapped function returns.
    # We must copy for thread safety, but it only needs to be a shallow copy.
    kwargs = {k: v for k, v in self.arguments.items()}
    if self._fn_expects_mask_arg:
      kwargs['mask'] = mask
    if self._fn_expects_training_arg:
      kwargs['training'] = training

    created_variables = []
    def _variable_creator(next_creator, **kwargs):
      # [FUNCTION DOC]
      # HELPER: variable-creator hook for Lambda.call. Creates the variable through the next
      # creator in the chain, records it in the enclosing list of created variables (side effect)
      # and returns it.
      var = next_creator(**kwargs)
      created_variables.append(var)
      return var

    with backprop.GradientTape(watch_accessed_variables=True) as tape,\
        variable_scope.variable_creator_scope(_variable_creator):
      result = self.function(inputs, **kwargs)
    self._check_variables(created_variables, tape.watched_variables())
    return result

  def _check_variables(self, created_variables, accessed_variables):
    # [FUNCTION DOC]
    # HELPER: validation helper for Lambda.call.
    # Inputs: created_variables, variables created while the function ran; accessed_variables,
    # variables the gradient tape saw being read. Returns immediately when both are empty.
    # Raises a value error if any created variable is not among the layer's tracked weights.
    # For accessed variables that are not tracked, logs a one-time warning (via the warn helper)
    # and sets the already-warned flag. Returns nothing.
    if not created_variables and not accessed_variables:
      # In the common case that a Lambda layer does not touch a Variable, we
      # don't want to incur the runtime cost of assembling any state used for
      # checking only to immediately discard it.
      return

    tracked_weights = set(v.ref() for v in self.weights)
    untracked_new_vars = [
        v for v in created_variables if v.ref() not in tracked_weights
    ]
    if untracked_new_vars:
      variable_str = '\n'.join('  {}'.format(i) for i in untracked_new_vars)
      error_str = textwrap.dedent("""
          The following Variables were created within a Lambda layer ({name})
          but are not tracked by said layer:
          {variable_str}
          The layer cannot safely ensure proper Variable reuse across multiple
          calls, and consequently this behavior is disallowed for safety. Lambda
          layers are not well suited to stateful computation; instead, writing a
          subclassed Layer is the recommend way to define layers with
          Variables.""").format(name=self.name, variable_str=variable_str)
      raise ValueError(error_str)

    untracked_used_vars = [
        v for v in accessed_variables if v.ref() not in tracked_weights
    ]
    if untracked_used_vars and not self._already_warned:
      variable_str = '\n'.join('  {}'.format(i) for i in untracked_used_vars)
      self._warn(textwrap.dedent(
          '''
          The following Variables were used a Lambda layer's call ({name}), but
          are not present in its tracked objects:
          {variable_str}
          It is possible that this is intended behavior, but it is more likely
          an omission. This is a strong indication that this layer should be
          formulated as a subclassed Layer rather than a Lambda layer.'''
      ).format(name=self.name, variable_str=variable_str))
      self._already_warned = True

  def _warn(self, msg):
    # [FUNCTION DOC]
    # HELPER: warning sink for the variable check; a seam that tests override to raise instead.
    # Logs the given message through the TensorFlow logger at warning level.
    # This method will be overridden in a unit test to raise an error, because
    # self.assertWarns is not universally implemented.
    return tf_logging.warning(msg)

  def compute_mask(self, inputs, mask=None):
    # [FUNCTION DOC]
    # Returns the output mask. If the configured mask is callable it is called with the inputs and
    # the incoming mask; otherwise the configured constant (possibly nothing) is returned as is.
    if callable(self.mask):
      return self.mask(inputs, mask)
    return self.mask

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a serializable mapping: the base layer configuration plus, for the function, output
    # shape and (if set) mask, each a triple of serialized value, kind (lambda, function or raw)
    # and defining module, and the extra arguments mapping. Lambdas are serialized as bytecode.
    # Raises a value error (via the serializer) if the function or mask is not callable.
    function_config = self._serialize_function_to_config(self.function)
    output_shape_config = self._serialize_function_to_config(self._output_shape,
                                                             allow_raw=True)
    config = {
        'function': function_config[0],
        'function_type': function_config[1],
        'module': function_config[2],
        'output_shape': output_shape_config[0],
        'output_shape_type': output_shape_config[1],
        'output_shape_module': output_shape_config[2],
    }
    if self.mask is not None:
      mask_config = self._serialize_function_to_config(self.mask)
      config.update({
          'mask': mask_config[0],
          'mask_type': mask_config[1],
          'mask_module': mask_config[2]
      })
    config['arguments'] = self.arguments

    base_config = super(Lambda, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def _serialize_function_to_config(self, inputs, allow_raw=False):
    # [FUNCTION DOC]
    # HELPER: serialization helper for Lambda.get_config.
    # KNOWN ISSUE: the check for the lambda type matches every plain Python function (lambdas and
    # def-functions share one type), so named functions are also dumped as bytecode and the
    # name-based branch below it is only reached for other callables such as builtins or objects.
    # Inputs: inputs, a function, lambda, or (when allow_raw is true) any raw value such as a shape
    # tuple; allow_raw defaults to false. Returns a triple of the serialized value, a kind label
    # (lambda, function or raw), and the module name (nothing for raw values). Raises a value error
    # for non-callable input when raw values are not allowed.
    if isinstance(inputs, python_types.LambdaType):
      output = generic_utils.func_dump(inputs)
      output_type = 'lambda'
      module = inputs.__module__
    elif callable(inputs):
      output = inputs.__name__
      output_type = 'function'
      module = inputs.__module__
    elif allow_raw:
      output = inputs
      output_type = 'raw'
      module = None
    else:
      raise ValueError(
          'Invalid input for serialization, type: %s ' % type(inputs))

    return output, output_type, module

  @classmethod
  def from_config(cls, config, custom_objects=None):
    # [FUNCTION DOC]
    # Class method that rebuilds a Lambda layer from a configuration mapping produced by
    # get_config. config: the mapping (copied, caller's is not modified at top level, though nested
    # argument entries are rewritten in place). custom_objects: optional mapping of names to
    # objects used to resolve functions. Deserializes function, output shape and optional mask,
    # converts argument entries that were saved as arrays back into numpy arrays, and returns a new
    # layer. Deserializing lambdas executes stored bytecode, so only load trusted configs.
    config = config.copy()
    function = cls._parse_function_from_config(
        config, custom_objects, 'function', 'module', 'function_type')

    output_shape = cls._parse_function_from_config(
        config, custom_objects, 'output_shape', 'output_shape_module',
        'output_shape_type')
    if 'mask' in config:
      mask = cls._parse_function_from_config(
          config, custom_objects, 'mask', 'mask_module', 'mask_type')
    else:
      mask = None

    config['function'] = function
    config['output_shape'] = output_shape
    config['mask'] = mask

    # If arguments were numpy array, they have been saved as
    # list. We need to recover the ndarray
    if 'arguments' in config:
      for key in config['arguments']:
        if isinstance(config['arguments'][key], dict):
          arg_dict = config['arguments'][key]
          if 'type' in arg_dict and arg_dict['type'] == 'ndarray':
            # Overwrite the argument with its numpy translation
            config['arguments'][key] = np.array(arg_dict['value'])

    return cls(**config)

  @classmethod
  def _parse_function_from_config(
      cls, config, custom_objects, func_attr_name, module_attr_name,
      func_type_attr_name):
    # [FUNCTION DOC]
    # HELPER: deserialization helper for Lambda.from_config (class method).
    # Inputs: config mapping (mutated: the module and type entries are removed), custom_objects
    # optional mapping, and the three key names locating the serialized value, its module and its
    # kind in the configuration. Builds a globals namespace from this module plus the function's
    # module (if already imported; otherwise warns that the module is not loaded) plus custom
    # objects. Kind function: looks the name up via the generic deserializer; kind lambda: loads
    # bytecode with that namespace; kind raw: returns the stored value. Raises a type error for an
    # unknown kind. Returns the reconstructed function or raw value.
    globs = globals().copy()
    module = config.pop(module_attr_name, None)
    if module in sys.modules:
      globs.update(sys.modules[module].__dict__)
    elif module is not None:
      # Note: we don't know the name of the function if it's a lambda.
      warnings.warn('{} is not loaded, but a Lambda layer uses it. '
                    'It may cause errors.'.format(module)
                    , UserWarning)
    if custom_objects:
      globs.update(custom_objects)
    function_type = config.pop(func_type_attr_name)
    if function_type == 'function':
      # Simple lookup in custom objects
      function = generic_utils.deserialize_keras_object(
          config[func_attr_name],
          custom_objects=custom_objects,
          printable_module_name='function in Lambda layer')
    elif function_type == 'lambda':
      # Unsafe deserialization from bytecode
      function = generic_utils.func_load(
          config[func_attr_name], globs=globs)
    elif function_type == 'raw':
      function = config[func_attr_name]
    else:
      raise TypeError('Unknown function type:', function_type)
    return function


class Dense(Layer):
  # [CLASS DOC]
  # Fully connected layer computing an activation of the input times a kernel matrix plus optional
  # bias. Holds the number of units, activation, bias flag and the initializer, regularizer and
  # constraint objects for kernel and bias; creates kernel and bias weights lazily in build once the
  # last input dimension is known. Handles dense and sparse rank-two inputs and higher-rank inputs
  # (contracting the last axis). Supports masking and requires rank of at least two.
  """Just your regular densely-connected NN layer.

  `Dense` implements the operation:
  `output = activation(dot(input, kernel) + bias)`
  where `activation` is the element-wise activation function
  passed as the `activation` argument, `kernel` is a weights matrix
  created by the layer, and `bias` is a bias vector created by the layer
  (only applicable if `use_bias` is `True`). These are all attributes of
  `Dense`.

  Note: If the input to the layer has a rank greater than 2, then `Dense`
  computes the dot product between the `inputs` and the `kernel` along the
  last axis of the `inputs` and axis 0 of the `kernel` (using `tf.tensordot`).
  For example, if input has dimensions `(batch_size, d0, d1)`,
  then we create a `kernel` with shape `(d1, units)`, and the `kernel` operates
  along axis 2 of the `input`, on every sub-tensor of shape `(1, 1, d1)`
  (there are `batch_size * d0` such sub-tensors).
  The output in this case will have shape `(batch_size, d0, units)`.

  Besides, layer attributes cannot be modified after the layer has been called
  once (except the `trainable` attribute).
  When a popular kwarg `input_shape` is passed, then keras will create
  an input layer to insert before the current layer. This can be treated
  equivalent to explicitly defining an `InputLayer`.

  Example:

  >>> # Create a `Sequential` model and add a Dense layer as the first layer.
  >>> model = tf.keras.models.Sequential()
  >>> model.add(tf.keras.Input(shape=(16,)))
  >>> model.add(tf.keras.layers.Dense(32, activation='relu'))
  >>> # Now the model will take as input arrays of shape (None, 16)
  >>> # and output arrays of shape (None, 32).
  >>> # Note that after the first layer, you don't need to specify
  >>> # the size of the input anymore:
  >>> model.add(tf.keras.layers.Dense(32))
  >>> model.output_shape
  (None, 32)

  Args:
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use.
      If you don't specify anything, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix.
    bias_initializer: Initializer for the bias vector.
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to
      the output of the layer (its "activation").
    kernel_constraint: Constraint function applied to
      the `kernel` weights matrix.
    bias_constraint: Constraint function applied to the bias vector.

  Input shape:
    N-D tensor with shape: `(batch_size, ..., input_dim)`.
    The most common situation would be
    a 2D input with shape `(batch_size, input_dim)`.

  Output shape:
    N-D tensor with shape: `(batch_size, ..., units)`.
    For instance, for a 2D input with shape `(batch_size, input_dim)`,
    the output would have shape `(batch_size, units)`.
  """

  def __init__(self,
               units,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the error message says a positive integer is expected, but the check only
    # rejects negative values, so zero units is accepted.
    # Constructor. units: output dimensionality (converted to an integer if it is not one).
    # activation: function or name, nothing means linear. use_bias: whether to add a bias vector.
    # kernel_initializer, bias_initializer: initializer objects or names (defaults: Glorot uniform
    # and zeros). kernel_regularizer, bias_regularizer, activity_regularizer: optional regularizers
    # (the activity one is passed to the base layer). kernel_constraint, bias_constraint: optional
    # constraints. Extra keyword arguments go to the base layer. Raises a value error for negative
    # units. All string identifiers are resolved via their registries. Sets input specification of
    # minimum rank two and enables masking support.
    super(Dense, self).__init__(
        activity_regularizer=activity_regularizer, **kwargs)

    self.units = int(units) if not isinstance(units, int) else units
    if self.units < 0:
      raise ValueError(f'Received an invalid value for `units`, expected '
                       f'a positive integer, got {units}.')
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

    self.input_spec = InputSpec(min_ndim=2)
    self.supports_masking = True

  def build(self, input_shape):
    # [FUNCTION DOC]
    # Creates the layer's weights. input_shape: shape of the inputs (any shape-like value). Raises a
    # type error if the layer dtype is neither floating nor complex, and a value error if the last
    # input dimension is unknown. Updates the input specification to pin the last axis size, adds a
    # trainable kernel of shape last dimension by units and, if bias is enabled, a trainable bias of
    # length units (otherwise the bias attribute is nothing). Marks the layer as built. Returns
    # nothing.
    dtype = dtypes.as_dtype(self.dtype or K.floatx())
    if not (dtype.is_floating or dtype.is_complex):
      raise TypeError('Unable to build `Dense` layer with non-floating point '
                      'dtype %s' % (dtype,))

    input_shape = tensor_shape.TensorShape(input_shape)
    last_dim = tensor_shape.dimension_value(input_shape[-1])
    if last_dim is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})
    self.kernel = self.add_weight(
        'kernel',
        shape=[last_dim, self.units],
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        dtype=self.dtype,
        trainable=True)
    if self.use_bias:
      self.bias = self.add_weight(
          'bias',
          shape=[self.units,],
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          dtype=self.dtype,
          trainable=True)
    else:
      self.bias = None
    self.built = True

  def call(self, inputs):
    # [FUNCTION DOC]
    # Forward pass. Casts inputs to the layer's compute dtype if different. For rank-two (or
    # unknown rank) inputs, multiplies by the kernel; sparse inputs use an embedding-lookup based
    # sparse multiplication (after filling empty rows) to obtain sparse gradients, dense inputs use
    # a plain matrix multiply. For higher ranks, contracts the last input axis with the kernel and,
    # in graph mode, restores the static output shape. Then adds the bias if enabled and applies the
    # activation if set. Returns the output tensor with last dimension equal to units.
    if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
      inputs = math_ops.cast(inputs, dtype=self._compute_dtype_object)

    rank = inputs.shape.rank
    if rank == 2 or rank is None:
      # We use embedding_lookup_sparse as a more efficient matmul operation for
      # large sparse input tensors. The op will result in a sparse gradient, as
      # opposed to sparse_ops.sparse_tensor_dense_matmul which results in dense
      # gradients. This can lead to sigfinicant speedups, see b/171762937.
      if isinstance(inputs, sparse_tensor.SparseTensor):
        # We need to fill empty rows, as the op assumes at least one id per row.
        inputs, _ = sparse_ops.sparse_fill_empty_rows(inputs, 0)
        # We need to do some munging of our input to use the embedding lookup as
        # a matrix multiply. We split our input matrix into separate ids and
        # weights tensors. The values of the ids tensor should be the column
        # indices of our input matrix and the values of the weights tensor
        # can continue to the actual matrix weights.
        # The column arrangement of ids and weights
        # will be summed over and does not matter. See the documentation for
        # sparse_ops.sparse_tensor_dense_matmul a more detailed explanation
        # of the inputs to both ops.
        ids = sparse_tensor.SparseTensor(
            indices=inputs.indices,
            values=inputs.indices[:, 1],
            dense_shape=inputs.dense_shape)
        weights = inputs
        outputs = embedding_ops.embedding_lookup_sparse_v2(
            self.kernel, ids, weights, combiner='sum')
      else:
        outputs = gen_math_ops.MatMul(a=inputs, b=self.kernel)
    # Broadcast kernel to inputs.
    else:
      outputs = standard_ops.tensordot(inputs, self.kernel, [[rank - 1], [0]])
      # Reshape the output back to the original ndim of the input.
      if not context.executing_eagerly():
        shape = inputs.shape.as_list()
        output_shape = shape[:-1] + [self.kernel.shape[-1]]
        outputs.set_shape(output_shape)

    if self.use_bias:
      outputs = nn_ops.bias_add(outputs, self.bias)

    if self.activation is not None:
      outputs = self.activation(outputs)
    return outputs

  def compute_output_shape(self, input_shape):
    # [FUNCTION DOC]
    # Returns the input shape with its last dimension replaced by the number of units. Raises a
    # value error if the input has fewer than two dimensions (rank check) or if the innermost
    # dimension is undefined.
    input_shape = tensor_shape.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if tensor_shape.dimension_value(input_shape[-1]) is None:
      raise ValueError(
          'The innermost dimension of input_shape must be defined, but saw: %s'
          % (input_shape,))
    return input_shape[:-1].concatenate(self.units)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the base layer configuration mapping updated with units, serialized activation, bias
    # flag, and serialized initializers, regularizers (including activity) and constraints.
    config = super(Dense, self).get_config()
    config.update({
        'units': self.units,
        'activation': activations.serialize(self.activation),
        'use_bias': self.use_bias,
        'kernel_initializer': initializers.serialize(self.kernel_initializer),
        'bias_initializer': initializers.serialize(self.bias_initializer),
        'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint': constraints.serialize(self.kernel_constraint),
        'bias_constraint': constraints.serialize(self.bias_constraint)
    })
    return config


class ActivityRegularization(Layer):
  # [CLASS DOC]
  # Identity layer that adds an L1/L2 penalty on its input activity to the model loss via the
  # base layer's activity regularizer mechanism. Holds the two factors. Supports masking and
  # leaves shape unchanged.
  """Layer that applies an update to the cost function based input activity.

  Args:
    l1: L1 regularization factor (positive float).
    l2: L2 regularization factor (positive float).

  Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the samples axis)
    when using this layer as the first layer in a model.

  Output shape:
    Same shape as input.
  """

  def __init__(self, l1=0., l2=0., **kwargs):
    # [FUNCTION DOC]
    # Constructor. l1, l2: non-negative regularization factors (default zero). Builds a combined
    # L1/L2 regularizer from them and passes it to the base layer as the activity regularizer;
    # extra keyword arguments also go to the base layer. Stores the factors and enables masking.
    super(ActivityRegularization, self).__init__(
        activity_regularizer=regularizers.L1L2(l1=l1, l2=l2), **kwargs)
    self.supports_masking = True
    self.l1 = l1
    self.l2 = l2

  def compute_output_shape(self, input_shape):
    # [FUNCTION DOC]
    # Output shape equals the input shape; returns the given shape unchanged.
    return input_shape

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a serializable mapping: the base layer configuration plus the L1 and L2 factors.
    config = {'l1': self.l1, 'l2': self.l2}
    base_config = super(ActivityRegularization, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class TFOpLambda(Layer):
  # [CLASS DOC]
  # Layer that wraps a TensorFlow API function so it can be applied to symbolic Keras tensors in
  # functional model construction. Holds the wrapped function and its canonical API symbol name
  # (used for serialization). Its call method is replaced per instance with a decorated wrapper that
  # runs the function while tracking variables created or used (warning or raising like Lambda).
  # Disables autocasting and input-structure unnesting in configs, and must be restored from config
  # rather than traced in a SavedModel. Created automatically by the global op dispatcher.
  """Wraps TF API symbols in a `Layer` object.

  It is inserted by the Functional API construction whenever users call
  a supported TF symbol on KerasTensors.

  Like Lambda layers, this layer tries to raise warnings when it detects users
  explicitly use variables in the call. (To let them know
  that the layer will not capture the variables).

  This is useful in the case where users do something like:
  x = keras.Input(...)
  y = tf.Variable(...)
  out = x * tf_variable
  """

  @trackable.no_automatic_dependency_tracking
  def __init__(self, function, **kwargs):
    # [FUNCTION DOC]
    # Constructor (decorated to avoid automatic dependency tracking). function: the TensorFlow
    # callable to wrap. Extra keyword arguments go to the base layer. Looks up the function's
    # canonical public API name (trying the default API then the keras API); if no name keyword was
    # given, generates a unique layer name from the symbol (or the function's own name) that avoids
    # already observed names. Forces autocasting off, installs a decorated call method that routes
    # to the wrapper method, marks the layer as needing restoration from config, and then calls the
    # base constructor. Finally initializes the one-time warning flag and clears training/mask
    # argument expectations.
    self.function = function
    self.symbol = (
        get_canonical_name_for_symbol(self.function,
                                      add_prefix_to_v1_names=True) or
        get_canonical_name_for_symbol(self.function,
                                      api_name='keras',
                                      add_prefix_to_v1_names=True))
    if 'name' not in kwargs:
      # Generate a name.
      # TFOpLambda layers avoid already-observed names,
      # because users cannot easily control the generated names.
      # Without this avoidance, users would be more likely to run
      # into unavoidable duplicate layer name collisions.
      # (For standard layers users could just set `name` when creating the
      # layer to work around a collision, but they can't do that for
      # auto-generated layers)
      if self.symbol:
        name = 'tf.' + self.symbol
      else:
        name = self.function.__name__
      kwargs['name'] = K.unique_object_name(
          name, zero_based=True, avoid_observed_names=True)
    kwargs['autocast'] = False

    # Decorate the function to produce this layer's call method
    def _call_wrapper(*args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: thin forwarding function that becomes this layer's call method; passes all
      # positional and keyword arguments to the layer's wrapper method and returns its result.
      return self._call_wrapper(*args, **kwargs)
    self.call = tf_decorator.make_decorator(function, _call_wrapper)

    # Do not individually trace op layers in the SavedModel.
    self._must_restore_from_config = True

    super(TFOpLambda, self).__init__(**kwargs)

    # Preserve all argument data structures when saving/loading a config
    # (e.g., don't unnest lists that contain one element)
    self._preserve_input_structure_in_config = True

    # Warning on every invocation will be quite irksome in Eager mode.
    self._already_warned = False

    self._expects_training_arg = False
    self._expects_mask_arg = False

  def _call_wrapper(self, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: implementation behind TFOpLambda's call method.
    # Calls the wrapped function with the given positional and keyword arguments, after dropping any
    # name keyword (to avoid duplicate op names when the layer is reused), inside a gradient tape
    # and variable-creator scope that records created and accessed variables. Then validates them
    # with the variable check helper (may raise a value error or log a warning). Returns the
    # function's result.
    created_variables = []
    def _variable_creator(next_creator, **creator_kwargs):
      # [FUNCTION DOC]
      # HELPER: variable-creator hook; creates the variable via the next creator, appends it to the
      # enclosing list of created variables (side effect) and returns it.
      var = next_creator(**creator_kwargs)
      created_variables.append(var)
      return var

    with backprop.GradientTape(watch_accessed_variables=True) as tape, \
        variable_scope.variable_creator_scope(_variable_creator):
      # We explicitly drop `name` arguments here,
      # to guard against the case where an op explicitly has a
      # `name` passed (which is susceptible to producing
      # multiple ops w/ the same name when the layer is reused)
      kwargs.pop('name', None)
      result = self.function(*args, **kwargs)
    self._check_variables(created_variables, tape.watched_variables())
    return result

  def _check_variables(self, created_variables, accessed_variables):
    # [FUNCTION DOC]
    # HELPER: validation helper for the wrapper method; duplicate of the Lambda layer's check.
    # Inputs: variables created while the op ran, and variables accessed (seen by the tape).
    # Returns immediately if both are empty. Raises a value error if a created variable is not a
    # tracked weight of the layer. Logs a one-time warning for accessed untracked variables and sets
    # the already-warned flag. The messages talk about a Lambda layer even though this is an op
    # layer. Returns nothing.
    if not created_variables and not accessed_variables:
      # In the common case that a Lambda layer does not touch a Variable, we
      # don't want to incur the runtime cost of assembling any state used for
      # checking only to immediately discard it.
      return

    tracked_weights = set(v.ref() for v in self.weights)
    untracked_new_vars = [
        v for v in created_variables if v.ref() not in tracked_weights
    ]
    if untracked_new_vars:
      variable_str = '\n'.join('  {}'.format(i) for i in untracked_new_vars)
      error_str = textwrap.dedent("""
          The following Variables were created within a Lambda layer ({name})
          but are not tracked by said layer:
          {variable_str}
          The layer cannot safely ensure proper Variable reuse across multiple
          calls, and consequently this behavior is disallowed for safety. Lambda
          layers are not well suited to stateful computation; instead, writing a
          subclassed Layer is the recommend way to define layers with
          Variables.""").format(name=self.name, variable_str=variable_str)
      raise ValueError(error_str)

    untracked_used_vars = [
        v for v in accessed_variables if v.ref() not in tracked_weights
    ]
    if untracked_used_vars and not self._already_warned:
      variable_str = '\n'.join('  {}'.format(i) for i in untracked_used_vars)
      self._warn(textwrap.dedent(
          '''
          The following Variables were used a Lambda layer's call ({name}), but
          are not present in its tracked objects:
          {variable_str}
          It is possible that this is intended behavior, but it is more likely
          an omission. This is a strong indication that this layer should be
          formulated as a subclassed Layer rather than a Lambda layer.'''
      ).format(name=self.name, variable_str=variable_str))
      self._already_warned = True

  def _warn(self, msg):
    # [FUNCTION DOC]
    # HELPER: warning sink for the variable check; a seam that tests override to raise instead.
    # Logs the given message through the TensorFlow logger at warning level.
    # This method will be overridden in a unit test to raise an error, because
    # self.assertWarns is not universally implemented.
    return tf_logging.warning(msg)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a serializable mapping: the base layer configuration plus the wrapped function's
    # canonical API symbol name. Raises a value error if the function has no public API symbol
    # (for example, it was wrapped by a dispatch decorator), since it cannot be serialized.
    if not self.symbol:
      raise ValueError('This Keras op layer was generated from %s, a method '
                       'that is not an exposed in the TensorFlow API. This '
                       'may have happened if the method was explicitly '
                       'decorated to add dispatching support, and it was used '
                       'during Functional model construction. '
                       'To ensure cross-version compatibility of Keras models '
                       'that use op layers, only op layers produced from '
                       'exported TF API symbols can be serialized.'
                       % self.function)
    config = {
        'function': self.symbol
    }

    base_config = super(TFOpLambda, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    # [FUNCTION DOC]
    # Class method that rebuilds the layer from a configuration mapping. config: mapping whose
    # function entry is an API symbol name (the mapping is copied). custom_objects: accepted but
    # unused. Resolves the symbol to the actual TensorFlow callable and returns a new layer. Raises
    # a value error if the symbol cannot be found.
    config = config.copy()
    symbol_name = config['function']
    function = get_symbol_from_name(symbol_name)
    if not function:
      raise ValueError(
          'TF symbol `tf.%s` could not be found.' % symbol_name)

    config['function'] = function

    return cls(**config)


class KerasOpDispatcher(dispatch.GlobalOpDispatcher):
  # [CLASS DOC]
  # Global TensorFlow op dispatcher registered at import time. When any TF op is called with a
  # symbolic Keras tensor among its arguments, it wraps the op in a TFOpLambda layer and applies it,
  # so functional models can be built from plain TF ops. Otherwise it declines to handle the call.
  """A global dispatcher that allows building a functional model with TF Ops."""

  def handle(self, op, args, kwargs):
    # [FUNCTION DOC]
    # Dispatcher hook. op: the TF function being called; args, kwargs: its positional and keyword
    # arguments. If any argument (searched through nested structures) is a symbolic Keras tensor,
    # creates a TFOpLambda for the op and returns the result of calling it on the arguments (a
    # symbolic tensor; a layer node is added to the functional graph). Otherwise returns the
    # sentinel meaning the operation is not supported by this dispatcher.
    """Handle the specified operation with the specified arguments."""
    if any(
        isinstance(x, keras_tensor.KerasTensor)
        for x in nest.flatten([args, kwargs])):
      return TFOpLambda(op)(*args, **kwargs)
    else:
      return self.NOT_SUPPORTED

KerasOpDispatcher().register()


def _slice_to_dict(x):
  # [FUNCTION DOC]
  # HELPER: converts slice objects into mapping form so nested-structure utilities treat them as
  # structures (they otherwise are leaves). Slice input becomes a mapping of start, stop and step;
  # anything else is returned unchanged.
  if isinstance(x, slice):
    return {'start': x.start, 'stop': x.stop, 'step': x.step}
  return x


def _dict_to_slice(x):
  # [FUNCTION DOC]
  # HELPER: inverse of the slice-to-mapping helper. A mapping with start, stop and step entries is
  # converted back into a slice (missing keys raise a key error); anything else is returned as is.
  if isinstance(x, dict):
    return slice(x['start'], x['stop'], x['step'])
  return x


class SlicingOpLambda(TFOpLambda):
  # [CLASS DOC]
  # KNOWN ISSUE: the docstring is copied from TFOpLambda and does not mention slicing.
  # TFOpLambda variant used for slicing and boolean-mask ops. Because slice objects cannot be
  # stored as layer-call arguments in node structures, they are passed around as mappings; this
  # class wraps its call method to convert such mappings (also inside lists and tuples) back to
  # slice objects before invoking the op. Created by the slicing dispatcher.
  """Wraps TF API symbols in a `Layer` object.

  It is inserted by the Functional API construction whenever users call
  a supported TF symbol on KerasTensors.

  Like Lambda layers, this layer tries to raise warnings when it detects users
  explicitly use variables in the call. (To let them know
  that the layer will not capture the variables).

  This is useful in the case where users do something like:
  x = keras.Input(...)
  y = tf.Variable(...)
  out = x * tf_variable
  """

  @trackable.no_automatic_dependency_tracking
  def __init__(self, function, **kwargs):
    # [FUNCTION DOC]
    # Constructor (decorated to avoid automatic dependency tracking). function: the TensorFlow
    # slicing-like callable; extra keyword arguments go to the base constructor. After base setup,
    # replaces the layer's call method with a decorated wrapper that first converts slice-mappings
    # in the arguments back to slice objects, then invokes the original call method.
    super(SlicingOpLambda, self).__init__(function, **kwargs)

    original_call = self.call
    # Decorate the function to produce this layer's call method
    def _call_wrapper(*args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: becomes the layer's call method. Converts each positional argument, and each
      # element of positional arguments that are lists or tuples (one level deep), from slice
      # mapping back to a slice object; does the same for keyword argument values. Nested deeper
      # structures are not converted. Returns the original call method's result.
      # Turn any slice dicts in the args back into `slice` objects.
      # This conversion cannot use nest.flatten/map_structure,
      # because dicts are flattened by nest while slices aren't.
      # So, map_structure would only see the individual elements in the
      # dict.
      # This can't use map_structure_up_to either because the 'shallowness' of
      # the shallow tree would have to vary depending on if only one dim or
      # multiple are being sliced.
      new_args = []
      for arg in args:
        arg = _dict_to_slice(arg)
        if isinstance(arg, (list, tuple)):
          new_arg = []
          for sub_arg in arg:
            new_arg.append(_dict_to_slice(sub_arg))
          arg = new_arg
        new_args.append(arg)

      # Handle the kwargs too.
      new_kwargs = {}
      for key, value in kwargs.items():
        value = _dict_to_slice(value)
        if isinstance(value, (list, tuple)):
          new_value = []
          for v in value:
            new_value.append(_dict_to_slice(v))
          value = new_value
        new_kwargs[key] = value

      return original_call(*new_args, **new_kwargs)
    self.call = tf_decorator.make_decorator(original_call, _call_wrapper)


class TFSlicingOpDispatcher(dispatch.OpDispatcher):
  # [CLASS DOC]
  # Per-op dispatcher registered (at import time) for the slicing-style ops: tensor indexing,
  # boolean mask (v1 and v2) and ragged tensor indexing. Holds the op it dispatches for. When called
  # with symbolic Keras tensors it builds a SlicingOpLambda layer; otherwise it declines.
  """A global dispatcher that allows building a functional model with TF Ops."""

  def __init__(self, op):
    # [FUNCTION DOC]
    # Constructor. op: the TensorFlow function this dispatcher intercepts; stored for later use.
    self.op = op

  def handle(self, args, kwargs):
    # [FUNCTION DOC]
    # Dispatcher hook. args, kwargs: arguments of the intercepted op. Slice objects within them are
    # first turned into mappings (so structure utilities can traverse them). If any argument is a
    # symbolic Keras tensor, returns the result of applying a SlicingOpLambda for the op to the
    # converted arguments; otherwise returns the sentinel meaning not supported.
    """Handle the specified operation with the specified arguments."""
    args = nest.map_structure(_slice_to_dict, args)
    kwargs = nest.map_structure(_slice_to_dict, kwargs)
    if any(
        isinstance(x, keras_tensor.KerasTensor)
        for x in nest.flatten([args, kwargs])):
      return SlicingOpLambda(self.op)(*args, **kwargs)
    else:
      return self.NOT_SUPPORTED

for slicing_op in [
    tensor_getitem_override._slice_helper,  # pylint: disable=protected-access
    array_ops.boolean_mask,
    array_ops.boolean_mask_v2,
    ragged_getitem.ragged_tensor_getitem
]:
  TFSlicingOpDispatcher(slicing_op).register(slicing_op)


class InstanceProperty(Layer):
  # [CLASS DOC]
  # Layer that returns a named attribute of the object it is called on. Used by specialized
  # symbolic tensors (ragged, sparse) to represent dynamic property access in functional models.
  # Holds the attribute name; disables autocasting; must be restored from config.
  """Wraps an instance property access (e.g. `x.foo`) in a Keras Layer.

  This layer takes an attribute name `attr_name` in the constructor and,
  when called on input tensor `obj` returns `obj.attr_name`.

  KerasTensors specialized for specific extension types use it to
  represent instance property accesses on the represented object in the
  case where the property needs to be dynamically accessed as opposed to
  being statically computed from the typespec, e.g.

  x = keras.Input(..., ragged=True)
  out = x.flat_values
  """

  @trackable.no_automatic_dependency_tracking
  def __init__(self, attr_name, **kwargs):
    # [FUNCTION DOC]
    # Constructor (decorated to avoid automatic dependency tracking). attr_name: name of the
    # attribute to fetch. Extra keyword arguments go to the base layer. If no name keyword is
    # given, generates a unique layer name from the attribute name. Forces autocasting off, marks the
    # layer as restored from config rather than traced, and preserves input structure in configs.
    self.attr_name = attr_name

    if 'name' not in kwargs:
      kwargs['name'] = K.unique_object_name(
          'input.' + self.attr_name, zero_based=True, avoid_observed_names=True)
    kwargs['autocast'] = False

    # Do not individually trace op layers in the SavedModel.
    self._must_restore_from_config = True

    super(InstanceProperty, self).__init__(**kwargs)

    # Preserve all argument data structures when saving/loading a config
    # (e.g., don't unnest lists that contain one element)
    self._preserve_input_structure_in_config = True

  def call(self, obj):
    # [FUNCTION DOC]
    # Forward pass: returns the attribute with the stored name from the given object. Raises an
    # attribute error if the object lacks it.
    return getattr(obj, self.attr_name)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a serializable mapping: the base layer configuration plus the attribute name.
    config = {
        'attr_name': self.attr_name
    }
    base_config = super(InstanceProperty, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    # [FUNCTION DOC]
    # Class method that creates a layer from a configuration mapping by passing all entries to the
    # constructor. custom_objects is accepted but unused.
    return cls(**config)


class InstanceMethod(InstanceProperty):
  # [CLASS DOC]
  # InstanceProperty variant that calls the named method on the object with extra positional and
  # keyword arguments and returns the result. Used by specialized symbolic tensors for dynamic
  # method calls in functional models.
  """Wraps an instance method access (e.g. `x.foo(arg)` in a Keras Layer.

  This layer takes an attribute name `attr_name` in the constructor and,
  when called on input tensor `obj` with additional arguments `args` and
  `kwargs` returns `obj.attr_name(*args, **kwargs)`.

  KerasTensors specialized for specific extension types use it to
  represent dynamic instance method calls on the represented object, e.g.

  x = keras.Input(..., ragged=True)
  new_values = keras.Input(...)
  out = x.with_values(new_values)
  """

  def call(self, obj, args, kwargs):
    # [FUNCTION DOC]
    # Forward pass. obj: object owning the method. args: sequence of positional arguments. kwargs:
    # mapping of keyword arguments. Looks up the named method on the object, calls it with those
    # arguments and returns its result. Raises an attribute error if the method is missing.
    method = getattr(obj, self.attr_name)
    return method(*args, **kwargs)


def _delegate_property(keras_tensor_cls, property_name):  # pylint: disable=invalid-name
  # [FUNCTION DOC]
  # HELPER: import-time glue that exposes a property on a symbolic tensor class. Adds a read-only
  # property named property_name to keras_tensor_cls (mutating the class) whose getter builds an
  # InstanceProperty layer for that name and applies it to the instance. Returns nothing. Repeated
  # calls simply overwrite the attribute.
  """Register property on a KerasTensor class.

  Calling this multiple times with the same arguments should be a no-op.

  This method exposes a property on the KerasTensor class that will use an
  `InstanceProperty` layer to access the property on the represented
  intermediate values in the model.

  Args:
    keras_tensor_cls: The KerasTensor subclass that should expose the property.
    property_name: The name of the property to expose and delegate to the
      represented (Composite)Tensor.
  """
  # We use a lambda because we can't create a Keras layer at import time
  # due to dynamic layer class versioning.
  property_access = property(lambda self: InstanceProperty(property_name)(self))  # pylint: disable=unnecessary-lambda
  setattr(keras_tensor_cls, property_name, property_access)


def _delegate_method(keras_tensor_cls, method_name):  # pylint: disable=invalid-name
  # [FUNCTION DOC]
  # HELPER: import-time glue that exposes a method on a symbolic tensor class. Adds a method named
  # method_name to keras_tensor_cls (mutating the class) which builds an InstanceMethod layer for
  # that name and applies it to the instance with the call's arguments. Returns nothing.
  """Register method on a KerasTensor class.

  Calling this function times with the same arguments should be a no-op.

  This method exposes an instance method on the KerasTensor class that will use
  an `InstanceMethod` layer to run the desired method on the represented
  intermediate values in the model.

  Args:
    keras_tensor_cls: The KerasTensor subclass that should expose the property.
    method_name: The name of the method to expose and delegate to the
      represented (Composite)Tensor.
  """
  def delegate(self, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: the method installed on the symbolic tensor class. Takes any positional and keyword
    # arguments, wraps them in an InstanceMethod layer call on the tensor and returns the resulting
    # symbolic tensor.
    return InstanceMethod(method_name)(self, args, kwargs)
  setattr(keras_tensor_cls, method_name, delegate)

# We do not support the `uniform_row_length` property because it
# returns either `None` or an int tensor, and code that relies on it tends
# to check `is None` directly. Delegating it here would always return a
# `KerasTensor`, regardless of what can be statically inferred. This would
# never equal `None`, breaking code that expects it to be partially-static
# in unpredictable ways.
for ragged_property in [
    'values',
    'flat_values',
    'row_splits',
    'nested_row_splits'
]:
  _delegate_property(keras_tensor.RaggedKerasTensor, ragged_property)

for ragged_method_name in [
    'value_rowids',
    'nested_value_rowids',
    'nrows',
    'row_starts',
    'row_limits',
    'row_lengths',
    'nested_row_lengths',
    'bounding_shape',
    'with_values',
    'with_flat_values',
    'with_row_splits_dtype',
    'merge_dims',
    'to_tensor',
    'to_sparse',
]:
  _delegate_method(keras_tensor.RaggedKerasTensor, ragged_method_name)

for sparse_property in [
    'indices',
    'values',
]:
  _delegate_property(keras_tensor.SparseKerasTensor, sparse_property)

for sparse_method in [
    'with_values',
]:
  _delegate_method(keras_tensor.SparseKerasTensor, sparse_method)


class ClassMethod(Layer):
  # [CLASS DOC]
  # Layer that calls a named class method of a TensorFlow class (for example a ragged tensor
  # constructor) when its arguments are symbolic Keras tensors. Holds the class reference, method
  # name and the class's canonical API symbol (used to serialize). Disables autocasting, must be
  # restored from config, and does not expect training or mask arguments.
  """Wraps a TF API Class's class method  in a `Layer` object.

  It is inserted by the Functional API construction whenever users call
  a supported TF Class's class method on KerasTensors.

  This is useful in the case where users do something like:
  x = keras.Input(...)
  y = keras.Input(...)
  out = tf.RaggedTensor.from_row_splits(x, y)
  """

  @trackable.no_automatic_dependency_tracking
  def __init__(self, cls_ref, method_name, **kwargs):
    # [FUNCTION DOC]
    # Constructor (decorated to avoid automatic dependency tracking). cls_ref: the TensorFlow class
    # owning the method. method_name: name of the class method. Extra keyword arguments go to the
    # base layer. Looks up the class's canonical API symbol (default API then keras API). If no
    # name keyword is given, generates a unique name from the symbol and method name (if the symbol
    # is not found this concatenation fails with a type error). Forces autocasting off, marks the
    # layer as restored from config, preserves input structure and clears training/mask expectations.
    self.cls_ref = cls_ref
    self.method_name = method_name
    self.cls_symbol = (
        get_canonical_name_for_symbol(self.cls_ref,
                                      add_prefix_to_v1_names=True) or
        get_canonical_name_for_symbol(self.cls_ref,
                                      api_name='keras',
                                      add_prefix_to_v1_names=True))
    if 'name' not in kwargs:
      kwargs['name'] = K.unique_object_name(
          'tf.' + self.cls_symbol + '.' + self.method_name, zero_based=True,
          avoid_observed_names=True)
    kwargs['autocast'] = False

    # Do not individually trace op layers in the SavedModel.
    self._must_restore_from_config = True

    super(ClassMethod, self).__init__(**kwargs)

    # Preserve all argument data structures when saving/loading a config
    # (e.g., don't unnest lists that contain one element)
    self._preserve_input_structure_in_config = True

    self._expects_training_arg = False
    self._expects_mask_arg = False

  def call(self, args, kwargs):
    # [FUNCTION DOC]
    # Forward pass. args: sequence of positional arguments; kwargs: mapping of keyword arguments.
    # Calls the named class method on the stored class with them and returns the result.
    return getattr(self.cls_ref, self.method_name)(*args, **kwargs)

  def get_config(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the error message formats the class symbol, which is empty exactly when this
    # error is raised, so the message names nothing useful instead of the offending class.
    # Returns a serializable mapping: the base layer configuration plus the class's API symbol and
    # the method name. Raises a value error if the class has no public API symbol.
    if not self.cls_symbol:
      raise ValueError('This Keras class method conversion tried to convert '
                       'a method belonging to class %s, a class '
                       'that is not an exposed in the TensorFlow API. '
                       'To ensure cross-version compatibility of Keras models '
                       'that use op layers, only op layers produced from '
                       'exported TF API symbols can be serialized.'
                       % self.cls_symbol)
    config = {
        'cls_symbol': self.cls_symbol,
        'method_name': self.method_name
    }

    base_config = super(ClassMethod, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    # [FUNCTION DOC]
    # Class method that rebuilds the layer from a configuration mapping (copied; the class symbol
    # entry is removed from the copy). custom_objects is accepted but unused. Resolves the symbol to
    # the TensorFlow class and returns a new layer. Raises a value error if the symbol is not found.
    config = config.copy()
    symbol_name = config.pop('cls_symbol')
    cls_ref = get_symbol_from_name(symbol_name)
    if not cls_ref:
      raise ValueError(
          'TF symbol `tf.%s` could not be found.' % symbol_name)

    config['cls_ref'] = cls_ref

    return cls(**config)


class TFClassMethodDispatcher(dispatch.OpDispatcher):
  # [CLASS DOC]
  # Per-method dispatcher registered at import time for the ragged tensor constructor class
  # methods. Holds the class and method name. When called with symbolic Keras tensors it builds a
  # ClassMethod layer and applies it; otherwise it declines.
  """A class method dispatcher that allows building a functional model with TF class methods."""

  def __init__(self, cls, method_name):
    # [FUNCTION DOC]
    # Constructor. cls: the TensorFlow class owning the method; method_name: name of the class
    # method. Both are stored.
    self.cls = cls
    self.method_name = method_name

  def handle(self, args, kwargs):
    # [FUNCTION DOC]
    # Dispatcher hook. args, kwargs: arguments of the intercepted call. If any is a symbolic Keras
    # tensor, applies a ClassMethod layer to the arguments with the first positional argument (the
    # class itself) dropped, and returns the symbolic result; otherwise returns the sentinel meaning
    # not supported.
    """Handle the specified operation with the specified arguments."""
    if any(
        isinstance(x, keras_tensor.KerasTensor)
        for x in nest.flatten([args, kwargs])):
      return ClassMethod(self.cls, self.method_name)(args[1:], kwargs)
    else:
      return self.NOT_SUPPORTED

for ragged_class_method in [
    'from_value_rowids',
    'from_row_splits',
    'from_row_lengths',
    'from_row_starts',
    'from_row_limits',
    'from_uniform_row_length',
    'from_nested_value_rowids',
    'from_nested_row_splits',
    'from_nested_row_lengths',
    'from_tensor',
    'from_sparse',
]:
  TFClassMethodDispatcher(
      ragged_tensor.RaggedTensor, ragged_class_method).register(
          getattr(ragged_tensor.RaggedTensor, ragged_class_method))
