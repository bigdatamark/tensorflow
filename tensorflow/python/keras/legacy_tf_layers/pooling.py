# [MODULE DOC]
# Legacy tf.layers pooling API: average and max pooling classes for 1D, 2D and 3D inputs (thin
# subclasses of the Keras pooling layers that also derive from the legacy base layer) and the
# deprecated functional wrappers that build a layer and apply it to a tensor. Also defines the
# aliases AvgPool2D, MaxPool2D, max_pool2d and avg_pool2d. No import-time side effects other than
# those aliases.
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
# pylint: disable=g-classes-have-attributes
"""Contains the pooling layer classes and their functional aliases."""
import warnings

from tensorflow.python.keras import layers as keras_layers
from tensorflow.python.keras.legacy_tf_layers import base


class AveragePooling1D(keras_layers.AveragePooling1D, base.Layer):
  # [CLASS DOC]
  # Legacy tf.layers average pooling layer for 1D inputs. Combines the Keras AveragePooling1D layer with
  # the legacy tf.layers base layer class; all pooling behavior is inherited. It only differs from the
  # Keras layer in that the strides argument is required. Instances are created directly or by the
  # functional alias in this module.
  """Average Pooling layer for 1D inputs.

  Args:
    pool_size: An integer or tuple/list of a single integer,
      representing the size of the pooling window.
    strides: An integer or tuple/list of a single integer, specifying the
      strides of the pooling operation.
    padding: A string. The padding method, either 'valid' or 'same'.
      Case-insensitive.
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, length, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, length)`.
    name: A string, the name of the layer.
  """

  def __init__(self, pool_size, strides,
               padding='valid', data_format='channels_last',
               name=None, **kwargs):
    # [FUNCTION DOC]
    # Constructs the layer. Inputs: pool_size, integer or tuple of window sizes; strides, integer or
    # tuple of strides (required: raises a value error if nothing is given); padding, valid or same
    # (default valid); data_format, channels_last (default) or channels_first; name, optional layer
    # name; extra keyword arguments are passed to the parent constructor. Returns nothing.
    if strides is None:
      raise ValueError('Argument `strides` must not be None.')
    super(AveragePooling1D, self).__init__(
        pool_size=pool_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        name=name,
        **kwargs)


def average_pooling1d(inputs, pool_size, strides,
                      padding='valid', data_format='channels_last',
                      name=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says it raises a value error if eager execution is enabled, but this
  # function has no explicit check; any such error would come from applying the legacy layer.
  # Deprecated functional interface to the average pooling 1D layer. Inputs: inputs, tensor of rank 3 (batch, length, channels);
  # pool_size, strides (required), padding (default valid), data_format (default channels_last),
  # name (optional). Emits a deprecation warning, builds the matching layer class and applies it to
  # the inputs. Returns the output tensor. The value error for a missing strides comes from the
  # layer constructor.
  """Average Pooling layer for 1D inputs.

  Args:
    inputs: The tensor over which to pool. Must have rank 3.
    pool_size: An integer or tuple/list of a single integer,
      representing the size of the pooling window.
    strides: An integer or tuple/list of a single integer, specifying the
      strides of the pooling operation.
    padding: A string. The padding method, either 'valid' or 'same'.
      Case-insensitive.
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, length, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, length)`.
    name: A string, the name of the layer.

  Returns:
    The output tensor, of rank 3.

  Raises:
    ValueError: if eager execution is enabled.
  """
  warnings.warn('`tf.layers.average_pooling1d` is deprecated and '
                'will be removed in a future version. '
                'Please use `tf.keras.layers.AveragePooling1D` instead.')
  layer = AveragePooling1D(pool_size=pool_size,
                           strides=strides,
                           padding=padding,
                           data_format=data_format,
                           name=name)
  return layer.apply(inputs)


class MaxPooling1D(keras_layers.MaxPooling1D, base.Layer):
  # [CLASS DOC]
  # Legacy tf.layers max pooling layer for 1D inputs. Combines the Keras MaxPooling1D layer with
  # the legacy tf.layers base layer class; all pooling behavior is inherited. It only differs from the
  # Keras layer in that the strides argument is required. Instances are created directly or by the
  # functional alias in this module.
  """Max Pooling layer for 1D inputs.

  Args:
    pool_size: An integer or tuple/list of a single integer,
      representing the size of the pooling window.
    strides: An integer or tuple/list of a single integer, specifying the
      strides of the pooling operation.
    padding: A string. The padding method, either 'valid' or 'same'.
      Case-insensitive.
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, length, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, length)`.
    name: A string, the name of the layer.
  """

  def __init__(self, pool_size, strides,
               padding='valid', data_format='channels_last',
               name=None, **kwargs):
    # [FUNCTION DOC]
    # Constructs the layer. Inputs: pool_size, integer or tuple of window sizes; strides, integer or
    # tuple of strides (required: raises a value error if nothing is given); padding, valid or same
    # (default valid); data_format, channels_last (default) or channels_first; name, optional layer
    # name; extra keyword arguments are passed to the parent constructor. Returns nothing.
    if strides is None:
      raise ValueError('Argument `strides` must not be None.')
    super(MaxPooling1D, self).__init__(
        pool_size=pool_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        name=name,
        **kwargs)


def max_pooling1d(inputs, pool_size, strides,
                  padding='valid', data_format='channels_last',
                  name=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says it raises a value error if eager execution is enabled, but this
  # function has no explicit check; any such error would come from applying the legacy layer.
  # Deprecated functional interface to the max pooling 1D layer. Inputs: inputs, tensor of rank 3 (batch, length, channels);
  # pool_size, strides (required), padding (default valid), data_format (default channels_last),
  # name (optional). Emits a deprecation warning, builds the matching layer class and applies it to
  # the inputs. Returns the output tensor. The value error for a missing strides comes from the
  # layer constructor.
  """Max Pooling layer for 1D inputs.

  Args:
    inputs: The tensor over which to pool. Must have rank 3.
    pool_size: An integer or tuple/list of a single integer,
      representing the size of the pooling window.
    strides: An integer or tuple/list of a single integer, specifying the
      strides of the pooling operation.
    padding: A string. The padding method, either 'valid' or 'same'.
      Case-insensitive.
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, length, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, length)`.
    name: A string, the name of the layer.

  Returns:
    The output tensor, of rank 3.

  Raises:
    ValueError: if eager execution is enabled.
  """
  warnings.warn('`tf.layers.max_pooling1d` is deprecated and '
                'will be removed in a future version. '
                'Please use `tf.keras.layers.MaxPooling1D` instead.')
  layer = MaxPooling1D(pool_size=pool_size,
                       strides=strides,
                       padding=padding,
                       data_format=data_format,
                       name=name)
  return layer.apply(inputs)


class AveragePooling2D(keras_layers.AveragePooling2D, base.Layer):
  # [CLASS DOC]
  # Legacy tf.layers average pooling layer for 2D inputs. Combines the Keras AveragePooling2D layer with
  # the legacy tf.layers base layer class; all pooling behavior is inherited. It only differs from the
  # Keras layer in that the strides argument is required. Instances are created directly or by the
  # functional alias in this module.
  """Average pooling layer for 2D inputs (e.g. images).

  Args:
    pool_size: An integer or tuple/list of 2 integers: (pool_height, pool_width)
      specifying the size of the pooling window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 2 integers,
      specifying the strides of the pooling operation.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    padding: A string. The padding method, either 'valid' or 'same'.
      Case-insensitive.
    data_format: A string. The ordering of the dimensions in the inputs.
      `channels_last` (default) and `channels_first` are supported.
      `channels_last` corresponds to inputs with shape
      `(batch, height, width, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, height, width)`.
    name: A string, the name of the layer.
  """

  def __init__(self, pool_size, strides,
               padding='valid', data_format='channels_last',
               name=None, **kwargs):
    # [FUNCTION DOC]
    # Constructs the layer. Inputs: pool_size, integer or tuple of window sizes; strides, integer or
    # tuple of strides (required: raises a value error if nothing is given); padding, valid or same
    # (default valid); data_format, channels_last (default) or channels_first; name, optional layer
    # name; extra keyword arguments are passed to the parent constructor. Returns nothing.
    if strides is None:
      raise ValueError('Argument `strides` must not be None.')
    super(AveragePooling2D, self).__init__(
        pool_size=pool_size, strides=strides,
        padding=padding, data_format=data_format, name=name, **kwargs)


def average_pooling2d(inputs,
                      pool_size, strides,
                      padding='valid', data_format='channels_last',
                      name=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says it raises a value error if eager execution is enabled, but this
  # function has no explicit check; any such error would come from applying the legacy layer.
  # Deprecated functional interface to the average pooling 2D layer. Inputs: inputs, tensor of rank 4 images;
  # pool_size, strides (required), padding (default valid), data_format (default channels_last),
  # name (optional). Emits a deprecation warning, builds the matching layer class and applies it to
  # the inputs. Returns the output tensor. The value error for a missing strides comes from the
  # layer constructor.
  """Average pooling layer for 2D inputs (e.g. images).

  Args:
    inputs: The tensor over which to pool. Must have rank 4.
    pool_size: An integer or tuple/list of 2 integers: (pool_height, pool_width)
      specifying the size of the pooling window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 2 integers,
      specifying the strides of the pooling operation.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    padding: A string. The padding method, either 'valid' or 'same'.
      Case-insensitive.
    data_format: A string. The ordering of the dimensions in the inputs.
      `channels_last` (default) and `channels_first` are supported.
      `channels_last` corresponds to inputs with shape
      `(batch, height, width, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, height, width)`.
    name: A string, the name of the layer.

  Returns:
    Output tensor.

  Raises:
    ValueError: if eager execution is enabled.
  """
  warnings.warn('`tf.layers.average_pooling2d` is deprecated and '
                'will be removed in a future version. '
                'Please use `tf.keras.layers.AveragePooling2D` instead.')
  layer = AveragePooling2D(pool_size=pool_size, strides=strides,
                           padding=padding, data_format=data_format,
                           name=name)
  return layer.apply(inputs)


class MaxPooling2D(keras_layers.MaxPooling2D, base.Layer):
  # [CLASS DOC]
  # Legacy tf.layers max pooling layer for 2D inputs. Combines the Keras MaxPooling2D layer with
  # the legacy tf.layers base layer class; all pooling behavior is inherited. It only differs from the
  # Keras layer in that the strides argument is required. Instances are created directly or by the
  # functional alias in this module.
  """Max pooling layer for 2D inputs (e.g. images).

  Args:
    pool_size: An integer or tuple/list of 2 integers: (pool_height, pool_width)
      specifying the size of the pooling window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 2 integers,
      specifying the strides of the pooling operation.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    padding: A string. The padding method, either 'valid' or 'same'.
      Case-insensitive.
    data_format: A string. The ordering of the dimensions in the inputs.
      `channels_last` (default) and `channels_first` are supported.
      `channels_last` corresponds to inputs with shape
      `(batch, height, width, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, height, width)`.
    name: A string, the name of the layer.
  """

  def __init__(self, pool_size, strides,
               padding='valid', data_format='channels_last',
               name=None, **kwargs):
    # [FUNCTION DOC]
    # Constructs the layer. Inputs: pool_size, integer or tuple of window sizes; strides, integer or
    # tuple of strides (required: raises a value error if nothing is given); padding, valid or same
    # (default valid); data_format, channels_last (default) or channels_first; name, optional layer
    # name; extra keyword arguments are passed to the parent constructor. Returns nothing.
    if strides is None:
      raise ValueError('Argument `strides` must not be None.')
    super(MaxPooling2D, self).__init__(
        pool_size=pool_size, strides=strides,
        padding=padding, data_format=data_format, name=name, **kwargs)


def max_pooling2d(inputs,
                  pool_size, strides,
                  padding='valid', data_format='channels_last',
                  name=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says it raises a value error if eager execution is enabled, but this
  # function has no explicit check; any such error would come from applying the legacy layer.
  # Deprecated functional interface to the max pooling 2D layer. Inputs: inputs, tensor of rank 4 images;
  # pool_size, strides (required), padding (default valid), data_format (default channels_last),
  # name (optional). Emits a deprecation warning, builds the matching layer class and applies it to
  # the inputs. Returns the output tensor. The value error for a missing strides comes from the
  # layer constructor.
  """Max pooling layer for 2D inputs (e.g. images).

  Args:
    inputs: The tensor over which to pool. Must have rank 4.
    pool_size: An integer or tuple/list of 2 integers: (pool_height, pool_width)
      specifying the size of the pooling window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 2 integers,
      specifying the strides of the pooling operation.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    padding: A string. The padding method, either 'valid' or 'same'.
      Case-insensitive.
    data_format: A string. The ordering of the dimensions in the inputs.
      `channels_last` (default) and `channels_first` are supported.
      `channels_last` corresponds to inputs with shape
      `(batch, height, width, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, height, width)`.
    name: A string, the name of the layer.

  Returns:
    Output tensor.

  Raises:
    ValueError: if eager execution is enabled.
  """
  warnings.warn('`tf.layers.max_pooling2d` is deprecated and '
                'will be removed in a future version. '
                'Please use `tf.keras.layers.MaxPooling2D` instead.')
  layer = MaxPooling2D(pool_size=pool_size, strides=strides,
                       padding=padding, data_format=data_format,
                       name=name)
  return layer.apply(inputs)


class AveragePooling3D(keras_layers.AveragePooling3D, base.Layer):
  # [CLASS DOC]
  # Legacy tf.layers average pooling layer for 3D inputs. Combines the Keras AveragePooling3D layer with
  # the legacy tf.layers base layer class; all pooling behavior is inherited. It only differs from the
  # Keras layer in that the strides argument is required. Instances are created directly or by the
  # functional alias in this module.
  """Average pooling layer for 3D inputs (e.g. volumes).

  Args:
    pool_size: An integer or tuple/list of 3 integers:
      (pool_depth, pool_height, pool_width)
      specifying the size of the pooling window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 3 integers,
      specifying the strides of the pooling operation.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    padding: A string. The padding method, either 'valid' or 'same'.
      Case-insensitive.
    data_format: A string. The ordering of the dimensions in the inputs.
      `channels_last` (default) and `channels_first` are supported.
      `channels_last` corresponds to inputs with shape
      `(batch, depth, height, width, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch, channels, depth, height, width)`.
    name: A string, the name of the layer.
  """

  def __init__(self, pool_size, strides,
               padding='valid', data_format='channels_last',
               name=None, **kwargs):
    # [FUNCTION DOC]
    # Constructs the layer. Inputs: pool_size, integer or tuple of window sizes; strides, integer or
    # tuple of strides (required: raises a value error if nothing is given); padding, valid or same
    # (default valid); data_format, channels_last (default) or channels_first; name, optional layer
    # name; extra keyword arguments are passed to the parent constructor. Returns nothing.
    if strides is None:
      raise ValueError('Argument `strides` must not be None.')
    super(AveragePooling3D, self).__init__(
        pool_size=pool_size, strides=strides,
        padding=padding, data_format=data_format, name=name, **kwargs)


def average_pooling3d(inputs,
                      pool_size, strides,
                      padding='valid', data_format='channels_last',
                      name=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says it raises a value error if eager execution is enabled, but this
  # function has no explicit check; any such error would come from applying the legacy layer.
  # Deprecated functional interface to the average pooling 3D layer. Inputs: inputs, tensor of rank 5 volumes;
  # pool_size, strides (required), padding (default valid), data_format (default channels_last),
  # name (optional). Emits a deprecation warning, builds the matching layer class and applies it to
  # the inputs. Returns the output tensor. The value error for a missing strides comes from the
  # layer constructor.
  """Average pooling layer for 3D inputs (e.g. volumes).

  Args:
    inputs: The tensor over which to pool. Must have rank 5.
    pool_size: An integer or tuple/list of 3 integers:
      (pool_depth, pool_height, pool_width)
      specifying the size of the pooling window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 3 integers,
      specifying the strides of the pooling operation.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    padding: A string. The padding method, either 'valid' or 'same'.
      Case-insensitive.
    data_format: A string. The ordering of the dimensions in the inputs.
      `channels_last` (default) and `channels_first` are supported.
      `channels_last` corresponds to inputs with shape
      `(batch, depth, height, width, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch, channels, depth, height, width)`.
    name: A string, the name of the layer.

  Returns:
    Output tensor.

  Raises:
    ValueError: if eager execution is enabled.
  """
  warnings.warn('`tf.layers.average_pooling3d` is deprecated and '
                'will be removed in a future version. '
                'Please use `tf.keras.layers.AveragePooling3D` instead.')
  layer = AveragePooling3D(pool_size=pool_size, strides=strides,
                           padding=padding, data_format=data_format,
                           name=name)
  return layer.apply(inputs)


class MaxPooling3D(keras_layers.MaxPooling3D, base.Layer):
  # [CLASS DOC]
  # Legacy tf.layers max pooling layer for 3D inputs. Combines the Keras MaxPooling3D layer with
  # the legacy tf.layers base layer class; all pooling behavior is inherited. It only differs from the
  # Keras layer in that the strides argument is required. Instances are created directly or by the
  # functional alias in this module.
  """Max pooling layer for 3D inputs (e.g. volumes).

  Args:
    pool_size: An integer or tuple/list of 3 integers:
      (pool_depth, pool_height, pool_width)
      specifying the size of the pooling window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 3 integers,
      specifying the strides of the pooling operation.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    padding: A string. The padding method, either 'valid' or 'same'.
      Case-insensitive.
    data_format: A string. The ordering of the dimensions in the inputs.
      `channels_last` (default) and `channels_first` are supported.
      `channels_last` corresponds to inputs with shape
      `(batch, depth, height, width, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch, channels, depth, height, width)`.
    name: A string, the name of the layer.
  """

  def __init__(self, pool_size, strides,
               padding='valid', data_format='channels_last',
               name=None, **kwargs):
    # [FUNCTION DOC]
    # Constructs the layer. Inputs: pool_size, integer or tuple of window sizes; strides, integer or
    # tuple of strides (required: raises a value error if nothing is given); padding, valid or same
    # (default valid); data_format, channels_last (default) or channels_first; name, optional layer
    # name; extra keyword arguments are passed to the parent constructor. Returns nothing.
    if strides is None:
      raise ValueError('Argument `strides` must not be None.')
    super(MaxPooling3D, self).__init__(
        pool_size=pool_size, strides=strides,
        padding=padding, data_format=data_format, name=name, **kwargs)


def max_pooling3d(inputs,
                  pool_size, strides,
                  padding='valid', data_format='channels_last',
                  name=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says it raises a value error if eager execution is enabled, but this
  # function has no explicit check; any such error would come from applying the legacy layer.
  # KNOWN ISSUE: the docstring summary is split mid-sentence (the first line ends at an unfinished
  # parenthesis and the word volumes sits in a separate paragraph).
  # Deprecated functional interface to the max pooling 3D layer. Inputs: inputs, tensor of rank 5 volumes;
  # pool_size, strides (required), padding (default valid), data_format (default channels_last),
  # name (optional). Emits a deprecation warning, builds the matching layer class and applies it to
  # the inputs. Returns the output tensor. The value error for a missing strides comes from the
  # layer constructor.
  """Max pooling layer for 3D inputs (e.g.

  volumes).

  Args:
    inputs: The tensor over which to pool. Must have rank 5.
    pool_size: An integer or tuple/list of 3 integers: (pool_depth, pool_height,
      pool_width) specifying the size of the pooling window. Can be a single
      integer to specify the same value for all spatial dimensions.
    strides: An integer or tuple/list of 3 integers, specifying the strides of
      the pooling operation. Can be a single integer to specify the same value
      for all spatial dimensions.
    padding: A string. The padding method, either 'valid' or 'same'.
      Case-insensitive.
    data_format: A string. The ordering of the dimensions in the inputs.
      `channels_last` (default) and `channels_first` are supported.
      `channels_last` corresponds to inputs with shape `(batch, depth, height,
      width, channels)` while `channels_first` corresponds to inputs with shape
      `(batch, channels, depth, height, width)`.
    name: A string, the name of the layer.

  Returns:
    Output tensor.

  Raises:
    ValueError: if eager execution is enabled.
  """
  warnings.warn('`tf.layers.max_pooling3d` is deprecated and '
                'will be removed in a future version. '
                'Please use `tf.keras.layers.MaxPooling3D` instead.')
  layer = MaxPooling3D(pool_size=pool_size, strides=strides,
                       padding=padding, data_format=data_format,
                       name=name)
  return layer.apply(inputs)

# Aliases

AvgPool2D = AveragePooling2D
MaxPool2D = MaxPooling2D
max_pool2d = max_pooling2d
avg_pool2d = average_pooling2d
