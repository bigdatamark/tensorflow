# [MODULE DOC]
# Legacy tf.layers core layers: Dense (fully connected), Dropout and Flatten. Each class
# combines the corresponding Keras layer with the legacy base layer (variable-scope based
# variable handling), and each has a deprecated functional wrapper that warns, builds the
# layer and applies it to an input tensor. Aliases at the end bind the old fully-connected
# names to the dense layer class and function.
# Import-time side effects: only name bindings; the default bias initializer is a single
# zeros initializer created once. Not a helper module; part of the deprecated tf.layers API.
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
"""Contains the core layers: Dense, Dropout.

Also contains their functional aliases.
"""
import warnings

from tensorflow.python.keras import layers as keras_layers
from tensorflow.python.keras.legacy_tf_layers import base
from tensorflow.python.ops import init_ops


class Dense(keras_layers.Dense, base.Layer):
  # [CLASS DOC]
  # Legacy densely-connected layer: output is the activation of the input times a kernel plus
  # an optional bias. Combines the Keras dense layer (computation, weights) with the legacy
  # base layer (variable scope and reuse). No extra state; the constructor forwards arguments.
  """Densely-connected layer class.

  This layer implements the operation:
  `outputs = activation(inputs * kernel + bias)`
  Where `activation` is the activation function passed as the `activation`
  argument (if not `None`), `kernel` is a weights matrix created by the layer,
  and `bias` is a bias vector created by the layer
  (only if `use_bias` is `True`).

  Args:
    units: Integer or Long, dimensionality of the output space.
    activation: Activation function (callable). Set it to None to maintain a
      linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: Initializer function for the weight matrix.
      If `None` (default), weights are initialized using the default
      initializer used by `tf.compat.v1.get_variable`.
    bias_initializer: Initializer function for the bias.
    kernel_regularizer: Regularizer function for the weight matrix.
    bias_regularizer: Regularizer function for the bias.
    activity_regularizer: Regularizer function for the output.
    kernel_constraint: An optional projection function to be applied to the
        kernel after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
    bias_constraint: An optional projection function to be applied to the
        bias after being updated by an `Optimizer`.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: String, the name of the layer. Layers with the same name will
      share weights, but to avoid mistakes we require reuse=True in such cases.
    _reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.

  Properties:
    units: Python integer, dimensionality of the output space.
    activation: Activation function (callable).
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: Initializer instance (or name) for the kernel matrix.
    bias_initializer: Initializer instance (or name) for the bias.
    kernel_regularizer: Regularizer instance for the kernel matrix (callable)
    bias_regularizer: Regularizer instance for the bias (callable).
    activity_regularizer: Regularizer instance for the output (callable)
    kernel_constraint: Constraint function for the kernel matrix.
    bias_constraint: Constraint function for the bias.
    kernel: Weight matrix (TensorFlow variable or tensor).
    bias: Bias vector, if applicable (TensorFlow variable or tensor).
  """

  def __init__(self, units,
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=init_ops.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               **kwargs):
    # [FUNCTION DOC]
    # Constructs a dense layer, forwarding every argument to the parent constructors. Inputs:
    # units (output dimensionality), activation (optional callable), use_bias (default true),
    # kernel initializer (optional), bias initializer (default zeros), optional kernel, bias and
    # activity regularizers, optional kernel and bias constraints, trainable flag (default true),
    # optional name, and extra keyword arguments (including the private scope and reuse ones
    # used by the legacy base layer). Returns nothing.
    super(Dense, self).__init__(units=units,
                                activation=activation,
                                use_bias=use_bias,
                                kernel_initializer=kernel_initializer,
                                bias_initializer=bias_initializer,
                                kernel_regularizer=kernel_regularizer,
                                bias_regularizer=bias_regularizer,
                                activity_regularizer=activity_regularizer,
                                kernel_constraint=kernel_constraint,
                                bias_constraint=bias_constraint,
                                trainable=trainable,
                                name=name,
                                **kwargs)


def dense(
    inputs, units,
    activation=None,
    use_bias=True,
    kernel_initializer=None,
    bias_initializer=init_ops.zeros_initializer(),
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    trainable=True,
    name=None,
    reuse=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says a value error is raised if eager execution is enabled, but
  # this wrapper has no such check; it only emits a deprecation warning.
  # Deprecated functional form of the dense layer. Inputs: input tensor, units, and the same
  # options as the layer class, plus a name (used as the variable scope) and a reuse flag.
  # Emits a deprecation warning, builds the layer, applies it to the inputs and returns the
  # output tensor, whose last dimension equals units. Side effect: creates or reuses
  # variables in the named scope.
  """Functional interface for the densely-connected layer.

  This layer implements the operation:
  `outputs = activation(inputs * kernel + bias)`
  where `activation` is the activation function passed as the `activation`
  argument (if not `None`), `kernel` is a weights matrix created by the layer,
  and `bias` is a bias vector created by the layer
  (only if `use_bias` is `True`).

  Args:
    inputs: Tensor input.
    units: Integer or Long, dimensionality of the output space.
    activation: Activation function (callable). Set it to None to maintain a
      linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: Initializer function for the weight matrix.
      If `None` (default), weights are initialized using the default
      initializer used by `tf.compat.v1.get_variable`.
    bias_initializer: Initializer function for the bias.
    kernel_regularizer: Regularizer function for the weight matrix.
    bias_regularizer: Regularizer function for the bias.
    activity_regularizer: Regularizer function for the output.
    kernel_constraint: An optional projection function to be applied to the
        kernel after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
    bias_constraint: An optional projection function to be applied to the
        bias after being updated by an `Optimizer`.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: String, the name of the layer.
    reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.

  Returns:
    Output tensor the same shape as `inputs` except the last dimension is of
    size `units`.

  Raises:
    ValueError: if eager execution is enabled.
  """
  warnings.warn('`tf.layers.dense` is deprecated and '
                'will be removed in a future version. '
                'Please use `tf.keras.layers.Dense` instead.')
  layer = Dense(units,
                activation=activation,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                kernel_constraint=kernel_constraint,
                bias_constraint=bias_constraint,
                trainable=trainable,
                name=name,
                _scope=name,
                _reuse=reuse)
  return layer.apply(inputs)


class Dropout(keras_layers.Dropout, base.Layer):
  # [CLASS DOC]
  # Legacy dropout layer: during training randomly zeroes a fraction of input units and scales
  # the kept ones. Combines the Keras dropout layer with the legacy base layer. Overrides the
  # call method only to give the training argument a default of false.
  """Applies Dropout to the input.

  Dropout consists in randomly setting a fraction `rate` of input units to 0
  at each update during training time, which helps prevent overfitting.
  The units that are kept are scaled by `1 / (1 - rate)`, so that their
  sum is unchanged at training time and inference time.

  Args:
    rate: The dropout rate, between 0 and 1. E.g. `rate=0.1` would drop out
      10% of input units.
    noise_shape: 1D tensor of type `int32` representing the shape of the
      binary dropout mask that will be multiplied with the input.
      For instance, if your inputs have shape
      `(batch_size, timesteps, features)`, and you want the dropout mask
      to be the same for all timesteps, you can use
      `noise_shape=[batch_size, 1, features]`.
    seed: A Python integer. Used to create random seeds. See
      `tf.compat.v1.set_random_seed`.
      for behavior.
    name: The name of the layer (string).
  """

  def __init__(self, rate=0.5,
               noise_shape=None,
               seed=None,
               name=None,
               **kwargs):
    # [FUNCTION DOC]
    # Constructs a dropout layer, forwarding arguments to the parent constructors. Inputs: rate
    # (fraction of units dropped, default 0.5), noise_shape (optional shape of the dropout mask),
    # seed (optional integer for randomness), name (optional), and extra keyword arguments.
    # Returns nothing.
    super(Dropout, self).__init__(rate=rate,
                                  noise_shape=noise_shape,
                                  seed=seed,
                                  name=name,
                                  **kwargs)

  def call(self, inputs, training=False):
    # [FUNCTION DOC]
    # Applies dropout by delegating to the Keras dropout layer. Inputs: the input tensor and
    # training (boolean or boolean tensor, default false, meaning inference mode returns the
    # input unchanged). Returns the output tensor. Randomness is used only when training.
    return super(Dropout, self).call(inputs, training=training)


def dropout(inputs,
            rate=0.5,
            noise_shape=None,
            seed=None,
            training=False,
            name=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says a value error is raised if eager execution is enabled, but
  # this wrapper has no such check. Unlike the other wrappers it does not pass the name as a
  # variable scope and has no reuse argument.
  # Deprecated functional form of dropout. Inputs: input tensor, rate (default 0.5), optional
  # noise shape, optional seed, training flag (boolean or boolean tensor, default false), and
  # name. Emits a deprecation warning, builds the layer and applies it with the training flag,
  # returning the output tensor.
  """Applies Dropout to the input.

  Dropout consists in randomly setting a fraction `rate` of input units to 0
  at each update during training time, which helps prevent overfitting.
  The units that are kept are scaled by `1 / (1 - rate)`, so that their
  sum is unchanged at training time and inference time.

  Args:
    inputs: Tensor input.
    rate: The dropout rate, between 0 and 1. E.g. "rate=0.1" would drop out
      10% of input units.
    noise_shape: 1D tensor of type `int32` representing the shape of the
      binary dropout mask that will be multiplied with the input.
      For instance, if your inputs have shape
      `(batch_size, timesteps, features)`, and you want the dropout mask
      to be the same for all timesteps, you can use
      `noise_shape=[batch_size, 1, features]`.
    seed: A Python integer. Used to create random seeds. See
      `tf.compat.v1.set_random_seed`
      for behavior.
    training: Either a Python boolean, or a TensorFlow boolean scalar tensor
      (e.g. a placeholder). Whether to return the output in training mode
      (apply dropout) or in inference mode (return the input untouched).
    name: The name of the layer (string).

  Returns:
    Output tensor.

  Raises:
    ValueError: if eager execution is enabled.
  """
  warnings.warn('`tf.layers.dropout` is deprecated and '
                'will be removed in a future version. '
                'Please use `tf.keras.layers.Dropout` instead.')
  layer = Dropout(rate, noise_shape=noise_shape, seed=seed, name=name)
  return layer.apply(inputs, training=training)


class Flatten(keras_layers.Flatten, base.Layer):
  # [CLASS DOC]
  # Legacy flatten layer: collapses all dimensions except the batch axis. Combines the Keras
  # flatten layer with the legacy base layer; adds no behavior of its own (empty body).
  """Flattens an input tensor while preserving the batch axis (axis 0).

  Args:
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, ..., channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, ...)`.

  Examples:

  ```
    x = tf.compat.v1.placeholder(shape=(None, 4, 4), dtype='float32')
    y = Flatten()(x)
    # now `y` has shape `(None, 16)`

    x = tf.compat.v1.placeholder(shape=(None, 3, None), dtype='float32')
    y = Flatten()(x)
    # now `y` has shape `(None, None)`
  ```
  """
  pass


def flatten(inputs, name=None, data_format='channels_last'):
  # [FUNCTION DOC]
  # Deprecated functional form of flatten. Inputs: input tensor, optional layer name, and data
  # format (channels last by default, or channels first). Emits a deprecation warning, builds
  # the flatten layer, applies it and returns the reshaped tensor that keeps the batch axis.
  # Unlike other wrappers it does not set a variable scope; no variables are created anyway.
  """Flattens an input tensor while preserving the batch axis (axis 0).

  Args:
    inputs: Tensor input.
    name: The name of the layer (string).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, height, width, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, height, width)`.

  Returns:
    Reshaped tensor.

  Examples:

  ```
    x = tf.compat.v1.placeholder(shape=(None, 4, 4), dtype='float32')
    y = flatten(x)
    # now `y` has shape `(None, 16)`

    x = tf.compat.v1.placeholder(shape=(None, 3, None), dtype='float32')
    y = flatten(x)
    # now `y` has shape `(None, None)`
  ```
  """
  warnings.warn('`tf.layers.flatten` is deprecated and '
                'will be removed in a future version. '
                'Please use `tf.keras.layers.Flatten` instead.')
  layer = Flatten(name=name, data_format=data_format)
  return layer.apply(inputs)


# Aliases

FullyConnected = Dense
fully_connected = dense
