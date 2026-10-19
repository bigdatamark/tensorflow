# [MODULE DOC]
# Legacy Keras layer implementations for convolution and simple image or sequence reshaping.
# Exposes the generic Conv base, the public Conv1D, Conv2D and Conv3D layers, the transposed
# variants (1D, 2D, 3D), the separable base and its 1D and 2D layers, the depthwise 2D layer,
# the upsampling, zero-padding and cropping layers for 1D, 2D and 3D, and a set of backward
# compatible alias names (Convolution*, SeparableConvolution*, Deconvolution* and Deconv*). It
# also re-exports the average and max pooling layers from the pooling module purely for
# backwards namespace compatibility. Layers depend on shared helpers for argument normalization
# and output length arithmetic from the convolution utilities module. Importing has no side
# effects beyond defining classes and aliases.
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
"""Keras convolution layers and image transformation layers."""

import functools

from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
# imports for backwards namespace compatibility
# pylint: disable=unused-import
from tensorflow.python.keras.layers.pooling import AveragePooling1D
from tensorflow.python.keras.layers.pooling import AveragePooling2D
from tensorflow.python.keras.layers.pooling import AveragePooling3D
from tensorflow.python.keras.layers.pooling import MaxPooling1D
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.python.keras.layers.pooling import MaxPooling3D
# pylint: enable=unused-import
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
# pylint: disable=g-classes-have-attributes


class Conv(Layer):
  # [CLASS DOC]
  # HELPER: abstract implementation base shared by Conv1D, Conv2D, Conv3D and their subclasses.
  # Generic N-dimensional convolution layer. It stores the normalized hyperparameters (rank,
  # filters, kernel size, strides, padding, data format, dilation rate, groups, activation, bias
  # flag, initializers, regularizers, constraints), creates a kernel weight and an optional bias
  # weight when built, and applies a native convolution op plus bias and activation when called.
  # Subclasses fix the rank and expose the public constructor arguments. Causal padding is only
  # allowed for the 1D layers. The input specification requires at least rank plus two dimensions
  # and, after build, pins the channel dimension to the size seen at build time.
  """Abstract N-D convolution layer (private, used as implementation base).

  This layer creates a convolution kernel that is convolved
  (actually cross-correlated) with the layer input to produce a tensor of
  outputs. If `use_bias` is True (and a `bias_initializer` is provided),
  a bias vector is created and added to the outputs. Finally, if
  `activation` is not `None`, it is applied to the outputs as well.

  Note: layer attributes cannot be modified after the layer has been called
  once (except the `trainable` attribute).

  Args:
    rank: An integer, the rank of the convolution, e.g. "2" for 2D convolution.
    filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution). Could be "None", eg in the case of
      depth wise convolution.
    kernel_size: An integer or tuple/list of n integers, specifying the
      length of the convolution window.
    strides: An integer or tuple/list of n integers,
      specifying the stride length of the convolution.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: One of `"valid"`,  `"same"`, or `"causal"` (case-insensitive).
      `"valid"` means no padding. `"same"` results in padding with zeros
      evenly to the left/right or up/down of the input such that output has the
      same height/width dimension as the input. `"causal"` results in causal
      (dilated) convolutions, e.g. `output[t]` does not depend on `input[t+1:]`.
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, ..., channels)` while `channels_first` corresponds to
      inputs with shape `(batch_size, channels, ...)`.
      Note: `channels_first` is only available on GPUs.
    dilation_rate: An integer or tuple/list of n integers, specifying
      the dilation rate to use for dilated convolution.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any `strides` value != 1.
    groups: A positive integer specifying the number of groups in which the
      input is split along the channel axis. Each group is convolved
      separately with `filters / groups` filters. The output is the
      concatenation of all the `groups` results along the channel axis.
      Input channels and `filters` must both be divisible by `groups`.
    activation: Activation function to use.
      If you don't specify anything, no activation is applied.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: An initializer for the convolution kernel. If None, the 
      default initializer (glorot_uniform) will be used. 
    bias_initializer: An initializer for the bias vector. If None, the default
      initializer (zeros) will be used.
    kernel_regularizer: Optional regularizer for the convolution kernel.
    bias_regularizer: Optional regularizer for the bias vector.
    activity_regularizer: Optional regularizer function for the output.
    kernel_constraint: Optional projection function to be applied to the
        kernel after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
    bias_constraint: Optional projection function to be applied to the
        bias after being updated by an `Optimizer`.
  """

  def __init__(self,
               rank,
               filters,
               kernel_size,
               strides=1,
               padding='valid',
               data_format=None,
               dilation_rate=1,
               groups=1,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               conv_op=None,
               **kwargs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the conv_op parameter is accepted but never read or stored; it has no effect.
    # KNOWN ISSUE: the negative-filters error message is built from two adjacent literals with
    # no space between them, so it reads "`filters`.Was expecting".
    # Constructs the generic convolution layer. Inputs: rank (integer number of spatial
    # dimensions), filters (output channel count; a float is truncated to an integer; may be
    # null), kernel_size, strides and dilation_rate (an integer or sequence, normalized to a tuple
    # of length rank), padding (valid, same or causal, case-insensitive), data_format (channels
    # last or first; null means the global default), groups (null or zero means one), activation,
    # use_bias, initializers, regularizers and constraints (each resolved from a name, config or
    # object), activity_regularizer, trainable, name and extra base-layer keyword arguments.
    # Stores all normalized values as attributes, sets an input spec of minimum rank plus two
    # dimensions, runs the validation helper and caches whether padding is causal, whether the
    # layout is channels first, and the native data format string. Raises a value error when
    # filters is negative, filters is not divisible by groups, a kernel size or stride contains
    # zero, or causal padding is used by a layer that is not a 1D (separable) convolution. Returns
    # nothing.
    super(Conv, self).__init__(
        trainable=trainable,
        name=name,
        activity_regularizer=regularizers.get(activity_regularizer),
        **kwargs)
    self.rank = rank

    if isinstance(filters, float):
      filters = int(filters)
    if filters is not None and filters < 0:
      raise ValueError(f'Received a negative value for `filters`.'
                       f'Was expecting a positive value, got {filters}.')
    self.filters = filters
    self.groups = groups or 1
    self.kernel_size = conv_utils.normalize_tuple(
        kernel_size, rank, 'kernel_size')
    self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
    self.padding = conv_utils.normalize_padding(padding)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.dilation_rate = conv_utils.normalize_tuple(
        dilation_rate, rank, 'dilation_rate')

    self.activation = activations.get(activation)
    self.use_bias = use_bias

    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.bias_constraint = constraints.get(bias_constraint)
    self.input_spec = InputSpec(min_ndim=self.rank + 2)

    self._validate_init()
    self._is_causal = self.padding == 'causal'
    self._channels_first = self.data_format == 'channels_first'
    self._tf_data_format = conv_utils.convert_data_format(
        self.data_format, self.rank + 2)

  def _validate_init(self):
    # [FUNCTION DOC]
    # HELPER: argument validation step called at the end of the constructor.
    # Checks the already-normalized hyperparameters. Raises a value error if filters is not
    # divisible by groups, if any kernel size or stride entry is zero, or if causal padding is
    # requested on anything other than the 1D convolution or 1D separable convolution layers.
    # Returns nothing and has no side effects.
    if self.filters is not None and self.filters % self.groups != 0:
      raise ValueError(
          'The number of filters must be evenly divisible by the number of '
          'groups. Received: groups={}, filters={}'.format(
              self.groups, self.filters))

    if not all(self.kernel_size):
      raise ValueError('The argument `kernel_size` cannot contain 0(s). '
                       'Received: %s' % (self.kernel_size,))

    if not all(self.strides):
      raise ValueError('The argument `strides` cannot contains 0(s). '
                       'Received: %s' % (self.strides,))

    if (self.padding == 'causal' and not isinstance(self,
                                                    (Conv1D, SeparableConv1D))):
      raise ValueError('Causal padding is only supported for `Conv1D`'
                       'and `SeparableConv1D`.')

  def build(self, input_shape):
    # [FUNCTION DOC]
    # Creates the layer weights from the input shape. Input: input_shape, any shape-like value
    # whose channel dimension must be known. Reads the channel count, requires it to be divisible
    # by groups (otherwise raises a value error; an undefined channel dimension also raises a value
    # error), creates the kernel weight (kernel size, then input channels per group, then filters)
    # and, when use_bias is set, a bias weight of length filters, otherwise sets bias to null.
    # Updates the input spec to pin the channel axis to the observed size, translates padding to the
    # native uppercase form (causal becomes valid, since causal padding is applied in call), and
    # stores a partially applied native convolution function that uses the strides, padding,
    # dilations, data format and a name derived from the class name (Conv1D is named conv1d for
    # backward compatibility). Marks the layer as built. Returns nothing.
    input_shape = tensor_shape.TensorShape(input_shape)
    input_channel = self._get_input_channel(input_shape)
    if input_channel % self.groups != 0:
      raise ValueError(
          'The number of input channels must be evenly divisible by the number '
          'of groups. Received groups={}, but the input has {} channels '
          '(full input shape is {}).'.format(self.groups, input_channel,
                                             input_shape))
    kernel_shape = self.kernel_size + (input_channel // self.groups,
                                       self.filters)

    self.kernel = self.add_weight(
        name='kernel',
        shape=kernel_shape,
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        trainable=True,
        dtype=self.dtype)
    if self.use_bias:
      self.bias = self.add_weight(
          name='bias',
          shape=(self.filters,),
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          trainable=True,
          dtype=self.dtype)
    else:
      self.bias = None
    channel_axis = self._get_channel_axis()
    self.input_spec = InputSpec(min_ndim=self.rank + 2,
                                axes={channel_axis: input_channel})

    # Convert Keras formats to TF native formats.
    if self.padding == 'causal':
      tf_padding = 'VALID'  # Causal padding handled in `call`.
    elif isinstance(self.padding, str):
      tf_padding = self.padding.upper()
    else:
      tf_padding = self.padding
    tf_dilations = list(self.dilation_rate)
    tf_strides = list(self.strides)

    tf_op_name = self.__class__.__name__
    if tf_op_name == 'Conv1D':
      tf_op_name = 'conv1d'  # Backwards compat.

    self._convolution_op = functools.partial(
        nn_ops.convolution_v2,
        strides=tf_strides,
        padding=tf_padding,
        dilations=tf_dilations,
        data_format=self._tf_data_format,
        name=tf_op_name)
    self.built = True

  def call(self, inputs):
    # [FUNCTION DOC]
    # Forward pass. Input: a tensor of rank at least rank plus two (extra leading batch
    # dimensions are allowed). For causal padding it first pads the left side of the time axis.
    # It then applies the stored convolution op with the kernel, adds the bias if enabled (a
    # one-dimensional channels-first case reshapes the bias manually; inputs with several batch
    # dimensions are handled by collapsing the batch dimensions around the bias addition), sets the
    # static output shape when running in graph mode, and applies the activation if any. Returns the
    # output tensor. Has no other side effects.
    input_shape = inputs.shape

    if self._is_causal:  # Apply causal padding to inputs for Conv1D.
      inputs = array_ops.pad(inputs, self._compute_causal_padding(inputs))

    outputs = self._convolution_op(inputs, self.kernel)

    if self.use_bias:
      output_rank = outputs.shape.rank
      if self.rank == 1 and self._channels_first:
        # nn.bias_add does not accept a 1D input tensor.
        bias = array_ops.reshape(self.bias, (1, self.filters, 1))
        outputs += bias
      else:
        # Handle multiple batch dimensions.
        if output_rank is not None and output_rank > 2 + self.rank:

          def _apply_fn(o):
            # [FUNCTION DOC]
            # HELPER: callback for call that adds the bias to a batch-collapsed output tensor.
            # Input: o, an output tensor with a single batch dimension. Returns the tensor with
            # the layer bias added along the channel axis using the native data format.
            return nn.bias_add(o, self.bias, data_format=self._tf_data_format)

          outputs = conv_utils.squeeze_batch_dims(
              outputs, _apply_fn, inner_rank=self.rank + 1)
        else:
          outputs = nn.bias_add(
              outputs, self.bias, data_format=self._tf_data_format)

    if not context.executing_eagerly():
      # Infer the static output shape:
      out_shape = self.compute_output_shape(input_shape)
      outputs.set_shape(out_shape)

    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def _spatial_output_shape(self, spatial_input_shape):
    # [FUNCTION DOC]
    # HELPER: shape computation for compute_output_shape.
    # Input: a list of spatial lengths (one per spatial dimension). Returns a list of output
    # lengths computed per dimension from the kernel size, padding, stride and dilation rate.
    # Unknown (null) lengths stay unknown. No side effects.
    return [
        conv_utils.conv_output_length(
            length,
            self.kernel_size[i],
            padding=self.padding,
            stride=self.strides[i],
            dilation=self.dilation_rate[i])
        for i, length in enumerate(spatial_input_shape)
    ]

  def compute_output_shape(self, input_shape):
    # [FUNCTION DOC]
    # Computes the output shape for a given input shape without running the layer. Input: a
    # shape-like value with any number of leading batch dimensions. Keeps the batch dimensions,
    # replaces the spatial dimensions by their convolved lengths and the channel dimension by
    # filters, placing the channels last or first according to the data format. Returns a tensor
    # shape. No side effects.
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    batch_rank = len(input_shape) - self.rank - 1
    if self.data_format == 'channels_last':
      return tensor_shape.TensorShape(
          input_shape[:batch_rank]
          + self._spatial_output_shape(input_shape[batch_rank:-1])
          + [self.filters])
    else:
      return tensor_shape.TensorShape(
          input_shape[:batch_rank] + [self.filters] +
          self._spatial_output_shape(input_shape[batch_rank + 1:]))

  def _recreate_conv_op(self, inputs):  # pylint: disable=unused-argument
    # [FUNCTION DOC]
    # HELPER: hook for subclasses that may need to rebuild the convolution op for new inputs.
    # The base implementation ignores its input and always returns false, meaning no re-creation
    # is needed. No side effects.
    return False

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a serializable mapping of the constructor arguments: filters, kernel size, strides,
    # padding, data format, dilation rate, groups, and the serialized activation, initializers,
    # regularizers (including the activity regularizer) and constraints, plus use_bias, merged
    # with the base layer configuration (the layer-specific entries win on conflict). Rank is not
    # included because subclasses fix it. No side effects.
    config = {
        'filters':
            self.filters,
        'kernel_size':
            self.kernel_size,
        'strides':
            self.strides,
        'padding':
            self.padding,
        'data_format':
            self.data_format,
        'dilation_rate':
            self.dilation_rate,
        'groups':
            self.groups,
        'activation':
            activations.serialize(self.activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint)
    }
    base_config = super(Conv, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def _compute_causal_padding(self, inputs):
    # [FUNCTION DOC]
    # HELPER: padding computation for causal 1D convolution in call.
    # Input: the input tensor (only its rank is used; unknown rank is treated as one batch
    # dimension). Returns a list of before/after padding pairs: no padding for batch dimensions,
    # dilation rate times (kernel size minus one) zeros on the left of the time axis only, and none
    # on the channel axis, ordered according to the data format. No side effects.
    """Calculates padding for 'causal' option for 1-d conv layers."""
    left_pad = self.dilation_rate[0] * (self.kernel_size[0] - 1)
    if getattr(inputs.shape, 'ndims', None) is None:
      batch_rank = 1
    else:
      batch_rank = len(inputs.shape) - 2
    if self.data_format == 'channels_last':
      causal_padding = [[0, 0]] * batch_rank + [[left_pad, 0], [0, 0]]
    else:
      causal_padding = [[0, 0]] * batch_rank + [[0, 0], [left_pad, 0]]
    return causal_padding

  def _get_channel_axis(self):
    # [FUNCTION DOC]
    # HELPER: locates the channel dimension for build and input-spec logic.
    # Returns a negative axis index: minus one for channels last, or minus (rank plus one) for
    # channels first. No side effects.
    if self.data_format == 'channels_first':
      return -1 - self.rank
    else:
      return -1

  def _get_input_channel(self, input_shape):
    # [FUNCTION DOC]
    # HELPER: reads the input channel count during build.
    # Input: a tensor shape. Returns the integer size of the channel dimension. Raises a value
    # error if that dimension is undefined. No side effects.
    channel_axis = self._get_channel_axis()
    if input_shape.dims[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    return int(input_shape[channel_axis])

  def _get_padding_op(self):
    # [FUNCTION DOC]
    # HELPER: padding translation utility for native ops.
    # KNOWN ISSUE: nothing in this module calls this method; the transpose layers compute their
    # own padding string, so it appears to be dead code.
    # Returns the padding mode for the native op: causal becomes valid, and string modes are
    # uppercased; list or tuple paddings are returned unchanged. No side effects.
    if self.padding == 'causal':
      op_padding = 'valid'
    else:
      op_padding = self.padding
    if not isinstance(op_padding, (list, tuple)):
      op_padding = op_padding.upper()
    return op_padding


class Conv1D(Conv):
  # [CLASS DOC]
  # Public 1D (temporal) convolution layer. A thin subclass of the generic convolution layer
  # with the rank fixed at one; all computation lives in the base class. Supports valid, same and
  # causal padding, channels last (default) or channels first, dilation, groups, bias, activation,
  # and the usual initializer, regularizer and constraint options. Typical input is a tensor of
  # batch, steps, channels; extra leading batch dimensions are allowed.
  """1D convolution layer (e.g. temporal convolution).

  This layer creates a convolution kernel that is convolved
  with the layer input over a single spatial (or temporal) dimension
  to produce a tensor of outputs.
  If `use_bias` is True, a bias vector is created and added to the outputs.
  Finally, if `activation` is not `None`,
  it is applied to the outputs as well.

  When using this layer as the first layer in a model,
  provide an `input_shape` argument
  (tuple of integers or `None`, e.g.
  `(10, 128)` for sequences of 10 vectors of 128-dimensional vectors,
  or `(None, 128)` for variable-length sequences of 128-dimensional vectors.

  Examples:

  >>> # The inputs are 128-length vectors with 10 timesteps, and the batch size
  >>> # is 4.
  >>> input_shape = (4, 10, 128)
  >>> x = tf.random.normal(input_shape)
  >>> y = tf.keras.layers.Conv1D(
  ... 32, 3, activation='relu',input_shape=input_shape[1:])(x)
  >>> print(y.shape)
  (4, 8, 32)

  >>> # With extended batch shape [4, 7] (e.g. weather data where batch
  >>> # dimensions correspond to spatial location and the third dimension
  >>> # corresponds to time.)
  >>> input_shape = (4, 7, 10, 128)
  >>> x = tf.random.normal(input_shape)
  >>> y = tf.keras.layers.Conv1D(
  ... 32, 3, activation='relu', input_shape=input_shape[2:])(x)
  >>> print(y.shape)
  (4, 7, 8, 32)

  Args:
    filters: Integer, the dimensionality of the output space
      (i.e. the number of output filters in the convolution).
    kernel_size: An integer or tuple/list of a single integer,
      specifying the length of the 1D convolution window.
    strides: An integer or tuple/list of a single integer,
      specifying the stride length of the convolution.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: One of `"valid"`, `"same"` or `"causal"` (case-insensitive).
      `"valid"` means no padding. `"same"` results in padding with zeros evenly
      to the left/right or up/down of the input such that output has the same
      height/width dimension as the input.
      `"causal"` results in causal (dilated) convolutions, e.g. `output[t]`
      does not depend on `input[t+1:]`. Useful when modeling temporal data
      where the model should not violate the temporal order.
      See [WaveNet: A Generative Model for Raw Audio, section
        2.1](https://arxiv.org/abs/1609.03499).
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
    dilation_rate: an integer or tuple/list of a single integer, specifying
      the dilation rate to use for dilated convolution.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any `strides` value != 1.
    groups: A positive integer specifying the number of groups in which the
      input is split along the channel axis. Each group is convolved
      separately with `filters / groups` filters. The output is the
      concatenation of all the `groups` results along the channel axis.
      Input channels and `filters` must both be divisible by `groups`.
    activation: Activation function to use.
      If you don't specify anything, no activation is applied (
      see `keras.activations`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix (
      see `keras.initializers`). Defaults to 'glorot_uniform'.
    bias_initializer: Initializer for the bias vector (
      see `keras.initializers`). Defaults to 'zeros'.
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix (see `keras.regularizers`).
    bias_regularizer: Regularizer function applied to the bias vector (
      see `keras.regularizers`).
    activity_regularizer: Regularizer function applied to
      the output of the layer (its "activation") (
      see `keras.regularizers`).
    kernel_constraint: Constraint function applied to the kernel matrix (
      see `keras.constraints`).
    bias_constraint: Constraint function applied to the bias vector (
      see `keras.constraints`).

  Input shape:
    3+D tensor with shape: `batch_shape + (steps, input_dim)`

  Output shape:
    3+D tensor with shape: `batch_shape + (new_steps, filters)`
      `steps` value might have changed due to padding or strides.

  Returns:
    A tensor of rank 3 representing
    `activation(conv1d(inputs, kernel) + bias)`.

  Raises:
    ValueError: when both `strides > 1` and `dilation_rate > 1`.
  """

  def __init__(self,
               filters,
               kernel_size,
               strides=1,
               padding='valid',
               data_format='channels_last',
               dilation_rate=1,
               groups=1,
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
    # Constructs a 1D convolution layer by forwarding to the generic constructor with rank one.
    # Inputs: filters (output channel count), kernel_size (integer or one-element sequence),
    # strides (default one), padding (valid by default; same or causal also accepted),
    # data_format (channels last by default), dilation_rate (default one), groups (default one),
    # activation, use_bias (default true), initializers (kernel glorot uniform, bias zeros),
    # regularizers, constraints, plus extra base-layer keyword arguments such as name. The
    # activation, initializers, regularizers and constraints are resolved from names or configs
    # before forwarding. Raises a value error under the same conditions as the base constructor.
    # Returns nothing.
    super(Conv1D, self).__init__(
        rank=1,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        groups=groups,
        activation=activations.get(activation),
        use_bias=use_bias,
        kernel_initializer=initializers.get(kernel_initializer),
        bias_initializer=initializers.get(bias_initializer),
        kernel_regularizer=regularizers.get(kernel_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        kernel_constraint=constraints.get(kernel_constraint),
        bias_constraint=constraints.get(bias_constraint),
        **kwargs)


class Conv2D(Conv):
  # [CLASS DOC]
  # Public 2D convolution layer (for example over images). A thin subclass of the generic
  # convolution layer with the rank fixed at two; all computation lives in the base class.
  # Supports valid and same padding, channels last or first (default taken from the global
  # setting), dilation, groups, bias, activation, and initializer, regularizer and constraint
  # options. Typical input is a tensor of batch, rows, columns, channels.
  """2D convolution layer (e.g. spatial convolution over images).

  This layer creates a convolution kernel that is convolved
  with the layer input to produce a tensor of
  outputs. If `use_bias` is True,
  a bias vector is created and added to the outputs. Finally, if
  `activation` is not `None`, it is applied to the outputs as well.

  When using this layer as the first layer in a model,
  provide the keyword argument `input_shape`
  (tuple of integers or `None`, does not include the sample axis),
  e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
  in `data_format="channels_last"`. You can use `None` when
  a dimension has variable size.

  Examples:

  >>> # The inputs are 28x28 RGB images with `channels_last` and the batch
  >>> # size is 4.
  >>> input_shape = (4, 28, 28, 3)
  >>> x = tf.random.normal(input_shape)
  >>> y = tf.keras.layers.Conv2D(
  ... 2, 3, activation='relu', input_shape=input_shape[1:])(x)
  >>> print(y.shape)
  (4, 26, 26, 2)

  >>> # With `dilation_rate` as 2.
  >>> input_shape = (4, 28, 28, 3)
  >>> x = tf.random.normal(input_shape)
  >>> y = tf.keras.layers.Conv2D(
  ... 2, 3, activation='relu', dilation_rate=2, input_shape=input_shape[1:])(x)
  >>> print(y.shape)
  (4, 24, 24, 2)

  >>> # With `padding` as "same".
  >>> input_shape = (4, 28, 28, 3)
  >>> x = tf.random.normal(input_shape)
  >>> y = tf.keras.layers.Conv2D(
  ... 2, 3, activation='relu', padding="same", input_shape=input_shape[1:])(x)
  >>> print(y.shape)
  (4, 28, 28, 2)

  >>> # With extended batch shape [4, 7]:
  >>> input_shape = (4, 7, 28, 28, 3)
  >>> x = tf.random.normal(input_shape)
  >>> y = tf.keras.layers.Conv2D(
  ... 2, 3, activation='relu', input_shape=input_shape[2:])(x)
  >>> print(y.shape)
  (4, 7, 26, 26, 2)


  Args:
    filters: Integer, the dimensionality of the output space (i.e. the number of
      output filters in the convolution).
    kernel_size: An integer or tuple/list of 2 integers, specifying the height
      and width of the 2D convolution window. Can be a single integer to specify
      the same value for all spatial dimensions.
    strides: An integer or tuple/list of 2 integers, specifying the strides of
      the convolution along the height and width. Can be a single integer to
      specify the same value for all spatial dimensions. Specifying any stride
      value != 1 is incompatible with specifying any `dilation_rate` value != 1.
    padding: one of `"valid"` or `"same"` (case-insensitive).
      `"valid"` means no padding. `"same"` results in padding with zeros evenly
      to the left/right or up/down of the input such that output has the same
      height/width dimension as the input.
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs. `channels_last` corresponds
      to inputs with shape `(batch_size, height, width, channels)` while
      `channels_first` corresponds to inputs with shape `(batch_size, channels,
      height, width)`. It defaults to the `image_data_format` value found in
      your Keras config file at `~/.keras/keras.json`. If you never set it, then
      it will be `channels_last`.
    dilation_rate: an integer or tuple/list of 2 integers, specifying the
      dilation rate to use for dilated convolution. Can be a single integer to
      specify the same value for all spatial dimensions. Currently, specifying
      any `dilation_rate` value != 1 is incompatible with specifying any stride
      value != 1.
    groups: A positive integer specifying the number of groups in which the
      input is split along the channel axis. Each group is convolved separately
      with `filters / groups` filters. The output is the concatenation of all
      the `groups` results along the channel axis. Input channels and `filters`
      must both be divisible by `groups`.
    activation: Activation function to use. If you don't specify anything, no
      activation is applied (see `keras.activations`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix (see
      `keras.initializers`). Defaults to 'glorot_uniform'.
    bias_initializer: Initializer for the bias vector (see
      `keras.initializers`). Defaults to 'zeros'.
    kernel_regularizer: Regularizer function applied to the `kernel` weights
      matrix (see `keras.regularizers`). 
    bias_regularizer: Regularizer function applied to the bias vector (see
      `keras.regularizers`). 
    activity_regularizer: Regularizer function applied to the output of the
      layer (its "activation") (see `keras.regularizers`).
    kernel_constraint: Constraint function applied to the kernel matrix (see
      `keras.constraints`).
    bias_constraint: Constraint function applied to the bias vector (see
      `keras.constraints`).
  Input shape:
    4+D tensor with shape: `batch_shape + (channels, rows, cols)` if
      `data_format='channels_first'`
    or 4+D tensor with shape: `batch_shape + (rows, cols, channels)` if
      `data_format='channels_last'`.
  Output shape:
    4+D tensor with shape: `batch_shape + (filters, new_rows, new_cols)` if
    `data_format='channels_first'` or 4+D tensor with shape: `batch_shape +
      (new_rows, new_cols, filters)` if `data_format='channels_last'`.  `rows`
      and `cols` values might have changed due to padding.

  Returns:
    A tensor of rank 4+ representing
    `activation(conv2d(inputs, kernel) + bias)`.

  Raises:
    ValueError: if `padding` is `"causal"`.
    ValueError: when both `strides > 1` and `dilation_rate > 1`.
  """

  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1),
               groups=1,
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
    # Constructs a 2D convolution layer by forwarding to the generic constructor with rank two.
    # Inputs: filters (output channel count), kernel_size (integer or pair), strides (default one
    # in each direction), padding (valid by default, or same), data_format (null means the global
    # default), dilation_rate (default one in each direction), groups (default one), activation,
    # use_bias (default true), initializers (kernel glorot uniform, bias zeros), regularizers,
    # constraints, plus extra base-layer keyword arguments such as name. Names or configs are
    # resolved to objects before forwarding. Raises a value error under the same conditions as the
    # base constructor (causal padding is rejected here). Returns nothing.
    super(Conv2D, self).__init__(
        rank=2,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        groups=groups,
        activation=activations.get(activation),
        use_bias=use_bias,
        kernel_initializer=initializers.get(kernel_initializer),
        bias_initializer=initializers.get(bias_initializer),
        kernel_regularizer=regularizers.get(kernel_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        kernel_constraint=constraints.get(kernel_constraint),
        bias_constraint=constraints.get(bias_constraint),
        **kwargs)


class Conv3D(Conv):
  # [CLASS DOC]
  # Public 3D convolution layer (for example over volumes or video). A thin subclass of the
  # generic convolution layer with the rank fixed at three; all computation lives in the base
  # class. Supports valid and same padding, channels last or first, dilation, groups, bias,
  # activation, and initializer, regularizer and constraint options. Typical input is a tensor of
  # batch, three spatial dimensions, channels.
  """3D convolution layer (e.g. spatial convolution over volumes).

  This layer creates a convolution kernel that is convolved
  with the layer input to produce a tensor of
  outputs. If `use_bias` is True,
  a bias vector is created and added to the outputs. Finally, if
  `activation` is not `None`, it is applied to the outputs as well.

  When using this layer as the first layer in a model,
  provide the keyword argument `input_shape`
  (tuple of integers or `None`, does not include the sample axis),
  e.g. `input_shape=(128, 128, 128, 1)` for 128x128x128 volumes
  with a single channel,
  in `data_format="channels_last"`.

  Examples:

  >>> # The inputs are 28x28x28 volumes with a single channel, and the
  >>> # batch size is 4
  >>> input_shape =(4, 28, 28, 28, 1)
  >>> x = tf.random.normal(input_shape)
  >>> y = tf.keras.layers.Conv3D(
  ... 2, 3, activation='relu', input_shape=input_shape[1:])(x)
  >>> print(y.shape)
  (4, 26, 26, 26, 2)

  >>> # With extended batch shape [4, 7], e.g. a batch of 4 videos of 3D frames,
  >>> # with 7 frames per video.
  >>> input_shape = (4, 7, 28, 28, 28, 1)
  >>> x = tf.random.normal(input_shape)
  >>> y = tf.keras.layers.Conv3D(
  ... 2, 3, activation='relu', input_shape=input_shape[2:])(x)
  >>> print(y.shape)
  (4, 7, 26, 26, 26, 2)

  Args:
    filters: Integer, the dimensionality of the output space (i.e. the number of
      output filters in the convolution).
    kernel_size: An integer or tuple/list of 3 integers, specifying the depth,
      height and width of the 3D convolution window. Can be a single integer to
      specify the same value for all spatial dimensions.
    strides: An integer or tuple/list of 3 integers, specifying the strides of
      the convolution along each spatial dimension. Can be a single integer to
      specify the same value for all spatial dimensions. Specifying any stride
      value != 1 is incompatible with specifying any `dilation_rate` value != 1.
    padding: one of `"valid"` or `"same"` (case-insensitive).
      `"valid"` means no padding. `"same"` results in padding with zeros evenly
      to the left/right or up/down of the input such that output has the same
      height/width dimension as the input.
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs. `channels_last` corresponds
      to inputs with shape `batch_shape + (spatial_dim1, spatial_dim2,
      spatial_dim3, channels)` while `channels_first` corresponds to inputs with
      shape `batch_shape + (channels, spatial_dim1, spatial_dim2,
      spatial_dim3)`. It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`. If you never set it, then it
      will be "channels_last".
    dilation_rate: an integer or tuple/list of 3 integers, specifying the
      dilation rate to use for dilated convolution. Can be a single integer to
      specify the same value for all spatial dimensions. Currently, specifying
      any `dilation_rate` value != 1 is incompatible with specifying any stride
      value != 1.
    groups: A positive integer specifying the number of groups in which the
      input is split along the channel axis. Each group is convolved separately
      with `filters / groups` filters. The output is the concatenation of all
      the `groups` results along the channel axis. Input channels and `filters`
      must both be divisible by `groups`.
    activation: Activation function to use. If you don't specify anything, no
      activation is applied (see `keras.activations`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix (see
      `keras.initializers`). Defaults to 'glorot_uniform'.
    bias_initializer: Initializer for the bias vector (see
      `keras.initializers`). Defaults to 'zeros'.
    kernel_regularizer: Regularizer function applied to the `kernel` weights
      matrix (see `keras.regularizers`).
    bias_regularizer: Regularizer function applied to the bias vector (see
      `keras.regularizers`).
    activity_regularizer: Regularizer function applied to the output of the
      layer (its "activation") (see `keras.regularizers`).
    kernel_constraint: Constraint function applied to the kernel matrix (see
      `keras.constraints`).
    bias_constraint: Constraint function applied to the bias vector (see
      `keras.constraints`).
  Input shape:
    5+D tensor with shape: `batch_shape + (channels, conv_dim1, conv_dim2,
      conv_dim3)` if data_format='channels_first'
    or 5+D tensor with shape: `batch_shape + (conv_dim1, conv_dim2, conv_dim3,
      channels)` if data_format='channels_last'.
  Output shape:
    5+D tensor with shape: `batch_shape + (filters, new_conv_dim1,
      new_conv_dim2, new_conv_dim3)` if data_format='channels_first'
    or 5+D tensor with shape: `batch_shape + (new_conv_dim1, new_conv_dim2,
      new_conv_dim3, filters)` if data_format='channels_last'. `new_conv_dim1`,
      `new_conv_dim2` and `new_conv_dim3` values might have changed due to
      padding.

  Returns:
    A tensor of rank 5+ representing
    `activation(conv3d(inputs, kernel) + bias)`.

  Raises:
    ValueError: if `padding` is "causal".
    ValueError: when both `strides > 1` and `dilation_rate > 1`.
  """

  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1, 1),
               groups=1,
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
    # Constructs a 3D convolution layer by forwarding to the generic constructor with rank three.
    # Inputs: filters (output channel count), kernel_size (integer or triple), strides and
    # dilation_rate (default one in each direction), padding (valid by default, or same),
    # data_format (null means the global default), groups (default one), activation, use_bias
    # (default true), initializers (kernel glorot uniform, bias zeros), regularizers, constraints,
    # plus extra base-layer keyword arguments such as name. Names or configs are resolved to
    # objects before forwarding. Raises a value error under the same conditions as the base
    # constructor (causal padding is rejected here). Returns nothing.
    super(Conv3D, self).__init__(
        rank=3,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        groups=groups,
        activation=activations.get(activation),
        use_bias=use_bias,
        kernel_initializer=initializers.get(kernel_initializer),
        bias_initializer=initializers.get(bias_initializer),
        kernel_regularizer=regularizers.get(kernel_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        kernel_constraint=constraints.get(kernel_constraint),
        bias_constraint=constraints.get(bias_constraint),
        **kwargs)


class Conv1DTranspose(Conv1D):
  # [CLASS DOC]
  # KNOWN ISSUE: the docstring says a value error is raised for causal padding or for strides
  # and dilation both above one. Neither is checked here; causal padding passes the inherited
  # 1D validation and only fails later inside the native op (padding string becomes causal), and
  # the stride/dilation combination is left to the native op.
  # Transposed (fractionally strided) 1D convolution layer, subclassing the 1D convolution layer.
  # It reuses the parent's attribute handling but overrides build (kernel stored with filters
  # before input channels), call (uses the native 1D transpose op with a dynamically computed
  # output shape), compute_output_shape and get_config. Adds an optional output_padding attribute
  # (one-element tuple or null) that must be smaller than the stride. The groups option is not
  # exposed and is fixed at one. Input must be rank three.
  """Transposed convolution layer (sometimes called Deconvolution).

  The need for transposed convolutions generally arises
  from the desire to use a transformation going in the opposite direction
  of a normal convolution, i.e., from something that has the shape of the
  output of some convolution to something that has the shape of its input
  while maintaining a connectivity pattern that is compatible with
  said convolution.

  When using this layer as the first layer in a model,
  provide the keyword argument `input_shape`
  (tuple of integers or `None`, does not include the sample axis),
  e.g. `input_shape=(128, 3)` for data with 128 time steps and 3 channels.

  Args:
    filters: Integer, the dimensionality of the output space
      (i.e. the number of output filters in the convolution).
    kernel_size: An integer length of the 1D convolution window.
    strides: An integer specifying the stride of the convolution along the
      time dimension. Specifying a stride value != 1 is incompatible with
      specifying a `dilation_rate` value != 1. Defaults to 1.
    padding: one of `"valid"` or `"same"` (case-insensitive).
      `"valid"` means no padding. `"same"` results in padding with zeros evenly
      to the left/right or up/down of the input such that output has the same
      height/width dimension as the input.
    output_padding: An integer specifying the amount of padding along
      the time dimension of the output tensor.
      The amount of output padding must be lower than the stride.
      If set to `None` (default), the output shape is inferred.
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, length, channels)` while `channels_first` corresponds to
      inputs with shape `(batch_size, channels, length)`.
    dilation_rate: an integer, specifying
      the dilation rate to use for dilated convolution.
      Currently, specifying a `dilation_rate` value != 1 is
      incompatible with specifying a stride value != 1.
      Also dilation rate larger than 1 is not currently supported.
    activation: Activation function to use.
      If you don't specify anything, no activation is applied (
      see `keras.activations`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix (
      see `keras.initializers`). Defaults to 'glorot_uniform'.
    bias_initializer: Initializer for the bias vector (
      see `keras.initializers`). Defaults to 'zeros'.
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix (see `keras.regularizers`).
    bias_regularizer: Regularizer function applied to the bias vector (
      see `keras.regularizers`).
    activity_regularizer: Regularizer function applied to
      the output of the layer (its "activation") (see `keras.regularizers`).
    kernel_constraint: Constraint function applied to the kernel matrix (
      see `keras.constraints`).
    bias_constraint: Constraint function applied to the bias vector (
      see `keras.constraints`).

  Input shape:
    3D tensor with shape:
    `(batch_size, steps, channels)`

  Output shape:
    3D tensor with shape:
    `(batch_size, new_steps, filters)`
    If `output_padding` is specified:
    ```
    new_timesteps = ((timesteps - 1) * strides + kernel_size -
    2 * padding + output_padding)
    ```

  Returns:
    A tensor of rank 3 representing
    `activation(conv1dtranspose(inputs, kernel) + bias)`.

  Raises:
    ValueError: if `padding` is "causal".
    ValueError: when both `strides` > 1 and `dilation_rate` > 1.

  References:
    - [A guide to convolution arithmetic for deep learning](
      https://arxiv.org/abs/1603.07285v1)
    - [Deconvolutional Networks](
      https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf)
  """

  def __init__(self,
               filters,
               kernel_size,
               strides=1,
               padding='valid',
               output_padding=None,
               data_format=None,
               dilation_rate=1,
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
    # Constructs the transposed 1D convolution layer. Inputs: filters, kernel_size, strides
    # (default one), padding (valid default, or same), output_padding (null to infer the output
    # length, otherwise an integer or one-element sequence), data_format (null means the global
    # default), dilation_rate, activation, use_bias, initializers, regularizers, constraints and
    # extra base-layer keyword arguments. Forwards everything except output_padding to the parent
    # constructor (groups stays at its default of one), then normalizes output_padding to a tuple
    # and stores it. Raises a value error if any output padding value is not strictly smaller than
    # the corresponding stride, plus any error from the parent constructor. Returns nothing.
    super(Conv1DTranspose, self).__init__(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activations.get(activation),
        use_bias=use_bias,
        kernel_initializer=initializers.get(kernel_initializer),
        bias_initializer=initializers.get(bias_initializer),
        kernel_regularizer=regularizers.get(kernel_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        kernel_constraint=constraints.get(kernel_constraint),
        bias_constraint=constraints.get(bias_constraint),
        **kwargs)

    self.output_padding = output_padding
    if self.output_padding is not None:
      self.output_padding = conv_utils.normalize_tuple(
          self.output_padding, 1, 'output_padding')
      for stride, out_pad in zip(self.strides, self.output_padding):
        if out_pad >= stride:
          raise ValueError('Stride ' + str(self.strides) + ' must be '
                           'greater than output padding ' +
                           str(self.output_padding))

  def build(self, input_shape):
    # [FUNCTION DOC]
    # Creates the weights for the transposed 1D convolution. Input: a shape-like value that must
    # have exactly three dimensions with a defined channel dimension; otherwise raises a value
    # error. Sets an input spec requiring rank three and the observed channel size, creates a
    # kernel weight shaped as kernel size, filters, input channels (note the swapped order versus
    # the forward convolution), and a bias weight of length filters when use_bias is set (otherwise
    # bias is null). Marks the layer as built. Returns nothing.
    input_shape = tensor_shape.TensorShape(input_shape)
    if len(input_shape) != 3:
      raise ValueError('Inputs should have rank 3. Received input shape: ' +
                       str(input_shape))
    channel_axis = self._get_channel_axis()
    if input_shape.dims[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = int(input_shape[channel_axis])
    self.input_spec = InputSpec(ndim=3, axes={channel_axis: input_dim})
    kernel_shape = self.kernel_size + (self.filters, input_dim)

    self.kernel = self.add_weight(
        name='kernel',
        shape=kernel_shape,
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        trainable=True,
        dtype=self.dtype)
    if self.use_bias:
      self.bias = self.add_weight(
          name='bias',
          shape=(self.filters,),
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          trainable=True,
          dtype=self.dtype)
    else:
      self.bias = None
    self.built = True

  def call(self, inputs):
    # [FUNCTION DOC]
    # Forward pass of the transposed 1D convolution. Input: a rank-three tensor. Reads the dynamic
    # batch size and time length, computes the output length from the kernel size, padding, output
    # padding, stride and dilation, builds the dynamic output shape tensor (channels first or last),
    # and applies the native 1D transpose convolution with the uppercase padding string. In graph
    # mode it also sets the static output shape. Then adds the bias (if enabled) and applies the
    # activation (if any). Returns the output tensor. No other side effects.
    inputs_shape = array_ops.shape(inputs)
    batch_size = inputs_shape[0]
    if self.data_format == 'channels_first':
      t_axis = 2
    else:
      t_axis = 1

    length = inputs_shape[t_axis]
    if self.output_padding is None:
      output_padding = None
    else:
      output_padding = self.output_padding[0]

    # Infer the dynamic output shape:
    out_length = conv_utils.deconv_output_length(
        length, self.kernel_size[0], padding=self.padding,
        output_padding=output_padding, stride=self.strides[0],
        dilation=self.dilation_rate[0])
    if self.data_format == 'channels_first':
      output_shape = (batch_size, self.filters, out_length)
    else:
      output_shape = (batch_size, out_length, self.filters)
    data_format = conv_utils.convert_data_format(self.data_format, ndim=3)

    output_shape_tensor = array_ops_stack.stack(output_shape)
    outputs = nn_ops.conv1d_transpose(
        inputs,
        self.kernel,
        output_shape_tensor,
        strides=self.strides,
        padding=self.padding.upper(),
        data_format=data_format,
        dilations=self.dilation_rate)

    if not context.executing_eagerly():
      # Infer the static output shape:
      out_shape = self.compute_output_shape(inputs.shape)
      outputs.set_shape(out_shape)

    if self.use_bias:
      outputs = nn.bias_add(
          outputs,
          self.bias,
          data_format=data_format)

    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def compute_output_shape(self, input_shape):
    # [FUNCTION DOC]
    # Computes the output shape of the transposed 1D convolution. Input: a shape-like value of
    # rank three. Copies it, replaces the channel dimension by filters and the time dimension by
    # the transposed-convolution output length (derived from kernel size, padding, output padding,
    # stride and dilation; unknown stays unknown). Returns a tensor shape. No side effects.
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    output_shape = list(input_shape)
    if self.data_format == 'channels_first':
      c_axis, t_axis = 1, 2
    else:
      c_axis, t_axis = 2, 1

    if self.output_padding is None:
      output_padding = None
    else:
      output_padding = self.output_padding[0]
    output_shape[c_axis] = self.filters
    output_shape[t_axis] = conv_utils.deconv_output_length(
        output_shape[t_axis],
        self.kernel_size[0],
        padding=self.padding,
        output_padding=output_padding,
        stride=self.strides[0],
        dilation=self.dilation_rate[0])
    return tensor_shape.TensorShape(output_shape)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the parent layer configuration mapping with an extra entry for output_padding
    # (the normalized tuple or null). No side effects.
    config = super(Conv1DTranspose, self).get_config()
    config['output_padding'] = self.output_padding
    return config


class Conv2DTranspose(Conv2D):
  # [CLASS DOC]
  # KNOWN ISSUE: the docstring says a value error is raised for causal padding or when strides
  # and dilation are both above one. Causal padding is rejected by the 2D parent validation, but
  # the stride/dilation combination is not checked here.
  # Transposed (fractionally strided) 2D convolution layer, subclassing the 2D convolution layer.
  # Overrides build (kernel stored as kernel height, width, filters, input channels), call (uses
  # the backend 2D transpose op with a computed output shape, preferring static height and width
  # when known), compute_output_shape and get_config. Adds an optional output_padding attribute
  # (pair or null) that must be smaller than the stride. The groups option is not exposed.
  # Input must be rank four.
  """Transposed convolution layer (sometimes called Deconvolution).

  The need for transposed convolutions generally arises
  from the desire to use a transformation going in the opposite direction
  of a normal convolution, i.e., from something that has the shape of the
  output of some convolution to something that has the shape of its input
  while maintaining a connectivity pattern that is compatible with
  said convolution.

  When using this layer as the first layer in a model,
  provide the keyword argument `input_shape`
  (tuple of integers or `None`, does not include the sample axis),
  e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
  in `data_format="channels_last"`.

  Args:
    filters: Integer, the dimensionality of the output space
      (i.e. the number of output filters in the convolution).
    kernel_size: An integer or tuple/list of 2 integers, specifying the
      height and width of the 2D convolution window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 2 integers,
      specifying the strides of the convolution along the height and width.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: one of `"valid"` or `"same"` (case-insensitive).
      `"valid"` means no padding. `"same"` results in padding with zeros evenly
      to the left/right or up/down of the input such that output has the same
      height/width dimension as the input.
    output_padding: An integer or tuple/list of 2 integers,
      specifying the amount of padding along the height and width
      of the output tensor.
      Can be a single integer to specify the same value for all
      spatial dimensions.
      The amount of output padding along a given dimension must be
      lower than the stride along that same dimension.
      If set to `None` (default), the output shape is inferred.
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, height, width, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch_size, channels, height, width)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".
    dilation_rate: an integer or tuple/list of 2 integers, specifying
      the dilation rate to use for dilated convolution.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any stride value != 1.
    activation: Activation function to use.
      If you don't specify anything, no activation is applied (
      see `keras.activations`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix (
      see `keras.initializers`). Defaults to 'glorot_uniform'.
    bias_initializer: Initializer for the bias vector (
      see `keras.initializers`). Defaults to 'zeros'.
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix (see `keras.regularizers`).
    bias_regularizer: Regularizer function applied to the bias vector (
      see `keras.regularizers`).
    activity_regularizer: Regularizer function applied to
      the output of the layer (its "activation") (see `keras.regularizers`).
    kernel_constraint: Constraint function applied to the kernel matrix (
      see `keras.constraints`).
    bias_constraint: Constraint function applied to the bias vector (
      see `keras.constraints`).

  Input shape:
    4D tensor with shape:
    `(batch_size, channels, rows, cols)` if data_format='channels_first'
    or 4D tensor with shape:
    `(batch_size, rows, cols, channels)` if data_format='channels_last'.

  Output shape:
    4D tensor with shape:
    `(batch_size, filters, new_rows, new_cols)` if data_format='channels_first'
    or 4D tensor with shape:
    `(batch_size, new_rows, new_cols, filters)` if data_format='channels_last'.
    `rows` and `cols` values might have changed due to padding.
    If `output_padding` is specified:
    ```
    new_rows = ((rows - 1) * strides[0] + kernel_size[0] - 2 * padding[0] +
    output_padding[0])
    new_cols = ((cols - 1) * strides[1] + kernel_size[1] - 2 * padding[1] +
    output_padding[1])
    ```

  Returns:
    A tensor of rank 4 representing
    `activation(conv2dtranspose(inputs, kernel) + bias)`.

  Raises:
    ValueError: if `padding` is "causal".
    ValueError: when both `strides` > 1 and `dilation_rate` > 1.

  References:
    - [A guide to convolution arithmetic for deep
      learning](https://arxiv.org/abs/1603.07285v1)
    - [Deconvolutional
      Networks](https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf)
  """

  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               output_padding=None,
               data_format=None,
               dilation_rate=(1, 1),
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
    # Constructs the transposed 2D convolution layer. Inputs: filters, kernel_size, strides
    # (default one per direction), padding (valid default, or same), output_padding (null to
    # infer the output size, otherwise an integer or pair), data_format (null means the global
    # default), dilation_rate, activation, use_bias, initializers, regularizers, constraints and
    # extra base-layer keyword arguments. Forwards everything except output_padding to the parent
    # constructor, then normalizes output_padding to a pair and stores it. Raises a value error
    # if any output padding value is not strictly smaller than the matching stride, plus any
    # error from the parent constructor. Returns nothing.
    super(Conv2DTranspose, self).__init__(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activations.get(activation),
        use_bias=use_bias,
        kernel_initializer=initializers.get(kernel_initializer),
        bias_initializer=initializers.get(bias_initializer),
        kernel_regularizer=regularizers.get(kernel_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        kernel_constraint=constraints.get(kernel_constraint),
        bias_constraint=constraints.get(bias_constraint),
        **kwargs)

    self.output_padding = output_padding
    if self.output_padding is not None:
      self.output_padding = conv_utils.normalize_tuple(
          self.output_padding, 2, 'output_padding')
      for stride, out_pad in zip(self.strides, self.output_padding):
        if out_pad >= stride:
          raise ValueError('Stride ' + str(self.strides) + ' must be '
                           'greater than output padding ' +
                           str(self.output_padding))

  def build(self, input_shape):
    # [FUNCTION DOC]
    # Creates the weights for the transposed 2D convolution. Input: a shape-like value that must
    # have exactly four dimensions with a defined channel dimension; otherwise raises a value
    # error. Sets an input spec requiring rank four and the observed channel size, creates a
    # kernel weight shaped as kernel height, kernel width, filters, input channels, and a bias
    # weight of length filters when use_bias is set (otherwise bias is null). Marks the layer as
    # built. Returns nothing.
    input_shape = tensor_shape.TensorShape(input_shape)
    if len(input_shape) != 4:
      raise ValueError('Inputs should have rank 4. Received input '
                       'shape: ' + str(input_shape))
    channel_axis = self._get_channel_axis()
    if input_shape.dims[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = int(input_shape[channel_axis])
    self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
    kernel_shape = self.kernel_size + (self.filters, input_dim)

    self.kernel = self.add_weight(
        name='kernel',
        shape=kernel_shape,
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        trainable=True,
        dtype=self.dtype)
    if self.use_bias:
      self.bias = self.add_weight(
          name='bias',
          shape=(self.filters,),
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          trainable=True,
          dtype=self.dtype)
    else:
      self.bias = None
    self.built = True

  def call(self, inputs):
    # [FUNCTION DOC]
    # Forward pass of the transposed 2D convolution. Input: a rank-four tensor. Takes the
    # height and width from the static shape when known, otherwise from the dynamic shape; computes
    # the output height and width from kernel size, padding, output padding, stride and dilation;
    # builds the dynamic output shape tensor; applies the backend 2D transpose convolution; in
    # graph mode sets the static output shape; adds the bias (if enabled) and applies the
    # activation (if any). Returns the output tensor. No other side effects.
    inputs_shape = array_ops.shape(inputs)
    batch_size = inputs_shape[0]
    if self.data_format == 'channels_first':
      h_axis, w_axis = 2, 3
    else:
      h_axis, w_axis = 1, 2

    # Use the constant height and weight when possible.
    # TODO(scottzhu): Extract this into a utility function that can be applied
    # to all convolutional layers, which currently lost the static shape
    # information due to tf.shape().
    height, width = None, None
    if inputs.shape.rank is not None:
      dims = inputs.shape.as_list()
      height = dims[h_axis]
      width = dims[w_axis]
    height = height if height is not None else inputs_shape[h_axis]
    width = width if width is not None else inputs_shape[w_axis]

    kernel_h, kernel_w = self.kernel_size
    stride_h, stride_w = self.strides

    if self.output_padding is None:
      out_pad_h = out_pad_w = None
    else:
      out_pad_h, out_pad_w = self.output_padding

    # Infer the dynamic output shape:
    out_height = conv_utils.deconv_output_length(height,
                                                 kernel_h,
                                                 padding=self.padding,
                                                 output_padding=out_pad_h,
                                                 stride=stride_h,
                                                 dilation=self.dilation_rate[0])
    out_width = conv_utils.deconv_output_length(width,
                                                kernel_w,
                                                padding=self.padding,
                                                output_padding=out_pad_w,
                                                stride=stride_w,
                                                dilation=self.dilation_rate[1])
    if self.data_format == 'channels_first':
      output_shape = (batch_size, self.filters, out_height, out_width)
    else:
      output_shape = (batch_size, out_height, out_width, self.filters)

    output_shape_tensor = array_ops_stack.stack(output_shape)
    outputs = backend.conv2d_transpose(
        inputs,
        self.kernel,
        output_shape_tensor,
        strides=self.strides,
        padding=self.padding,
        data_format=self.data_format,
        dilation_rate=self.dilation_rate)

    if not context.executing_eagerly():
      # Infer the static output shape:
      out_shape = self.compute_output_shape(inputs.shape)
      outputs.set_shape(out_shape)

    if self.use_bias:
      outputs = nn.bias_add(
          outputs,
          self.bias,
          data_format=conv_utils.convert_data_format(self.data_format, ndim=4))

    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def compute_output_shape(self, input_shape):
    # [FUNCTION DOC]
    # Computes the output shape of the transposed 2D convolution. Input: a shape-like value of
    # rank four. Copies it, sets the channel dimension to filters and replaces height and width
    # by the transposed-convolution output lengths (from kernel size, padding, output padding,
    # stride and dilation; unknown stays unknown). Returns a tensor shape. No side effects.
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    output_shape = list(input_shape)
    if self.data_format == 'channels_first':
      c_axis, h_axis, w_axis = 1, 2, 3
    else:
      c_axis, h_axis, w_axis = 3, 1, 2

    kernel_h, kernel_w = self.kernel_size
    stride_h, stride_w = self.strides

    if self.output_padding is None:
      out_pad_h = out_pad_w = None
    else:
      out_pad_h, out_pad_w = self.output_padding

    output_shape[c_axis] = self.filters
    output_shape[h_axis] = conv_utils.deconv_output_length(
        output_shape[h_axis],
        kernel_h,
        padding=self.padding,
        output_padding=out_pad_h,
        stride=stride_h,
        dilation=self.dilation_rate[0])
    output_shape[w_axis] = conv_utils.deconv_output_length(
        output_shape[w_axis],
        kernel_w,
        padding=self.padding,
        output_padding=out_pad_w,
        stride=stride_w,
        dilation=self.dilation_rate[1])
    return tensor_shape.TensorShape(output_shape)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the parent layer configuration mapping with an extra entry for output_padding
    # (the normalized pair or null). No side effects.
    config = super(Conv2DTranspose, self).get_config()
    config['output_padding'] = self.output_padding
    return config


class Conv3DTranspose(Conv3D):
  # [CLASS DOC]
  # KNOWN ISSUE: the constructor accepts dilation_rate, but call and compute_output_shape never
  # use it when computing output sizes or in the native op, and get_config removes it, so
  # dilation is silently ignored and lost on serialization round trips.
  # KNOWN ISSUE: the docstring says a value error is raised when strides and dilation are both
  # above one; no such check exists in this class.
  # Transposed (fractionally strided) 3D convolution layer, subclassing the 3D convolution layer.
  # Overrides build (kernel stored as kernel depth, height, width, filters, input channels), call
  # (uses the native 3D transpose op with a dynamically computed output shape), compute_output_shape
  # and get_config. Adds an optional output_padding attribute (triple or null) that must be smaller
  # than the stride. The groups option is not exposed. Input must be rank five.
  """Transposed convolution layer (sometimes called Deconvolution).

  The need for transposed convolutions generally arises
  from the desire to use a transformation going in the opposite direction
  of a normal convolution, i.e., from something that has the shape of the
  output of some convolution to something that has the shape of its input
  while maintaining a connectivity pattern that is compatible with
  said convolution.

  When using this layer as the first layer in a model,
  provide the keyword argument `input_shape`
  (tuple of integers or `None`, does not include the sample axis),
  e.g. `input_shape=(128, 128, 128, 3)` for a 128x128x128 volume with 3 channels
  if `data_format="channels_last"`.

  Args:
    filters: Integer, the dimensionality of the output space
      (i.e. the number of output filters in the convolution).
    kernel_size: An integer or tuple/list of 3 integers, specifying the
      depth, height and width of the 3D convolution window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 3 integers,
      specifying the strides of the convolution along the depth, height
        and width.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: one of `"valid"` or `"same"` (case-insensitive).
      `"valid"` means no padding. `"same"` results in padding with zeros evenly
      to the left/right or up/down of the input such that output has the same
      height/width dimension as the input.
    output_padding: An integer or tuple/list of 3 integers,
      specifying the amount of padding along the depth, height, and
      width.
      Can be a single integer to specify the same value for all
      spatial dimensions.
      The amount of output padding along a given dimension must be
      lower than the stride along that same dimension.
      If set to `None` (default), the output shape is inferred.
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, depth, height, width, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch_size, channels, depth, height, width)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".
    dilation_rate: an integer or tuple/list of 3 integers, specifying
      the dilation rate to use for dilated convolution.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any stride value != 1.
    activation: Activation function to use.
      If you don't specify anything, no activation is applied (
      see `keras.activations`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix (
      see `keras.initializers`). Defaults to 'glorot_uniform'.
    bias_initializer: Initializer for the bias vector (
      see `keras.initializers`). Defaults to 'zeros'.
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix (
      see `keras.regularizers`).
    bias_regularizer: Regularizer function applied to the bias vector (
      see `keras.regularizers`).
    activity_regularizer: Regularizer function applied to
      the output of the layer (its "activation") (
      see `keras.regularizers`).
    kernel_constraint: Constraint function applied to the kernel matrix (
      see `keras.constraints`).
    bias_constraint: Constraint function applied to the bias vector (
      see `keras.constraints`).

  Input shape:
    5D tensor with shape:
    `(batch_size, channels, depth, rows, cols)` if data_format='channels_first'
    or 5D tensor with shape:
    `(batch_size, depth, rows, cols, channels)` if data_format='channels_last'.

  Output shape:
    5D tensor with shape:
    `(batch_size, filters, new_depth, new_rows, new_cols)` if
      data_format='channels_first'
    or 5D tensor with shape:
    `(batch_size, new_depth, new_rows, new_cols, filters)` if
      data_format='channels_last'.
    `depth` and `rows` and `cols` values might have changed due to padding.
    If `output_padding` is specified::
    ```
    new_depth = ((depth - 1) * strides[0] + kernel_size[0] - 2 * padding[0] +
    output_padding[0])
    new_rows = ((rows - 1) * strides[1] + kernel_size[1] - 2 * padding[1] +
    output_padding[1])
    new_cols = ((cols - 1) * strides[2] + kernel_size[2] - 2 * padding[2] +
    output_padding[2])
    ```

  Returns:
    A tensor of rank 5 representing
    `activation(conv3dtranspose(inputs, kernel) + bias)`.

  Raises:
    ValueError: if `padding` is "causal".
    ValueError: when both `strides` > 1 and `dilation_rate` > 1.

  References:
    - [A guide to convolution arithmetic for deep
      learning](https://arxiv.org/abs/1603.07285v1)
    - [Deconvolutional
      Networks](https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf)
  """

  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1, 1),
               padding='valid',
               output_padding=None,
               data_format=None,
               dilation_rate=(1, 1, 1),
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
    # Constructs the transposed 3D convolution layer. Inputs: filters, kernel_size, strides
    # (default one per direction), padding (valid default, or same), output_padding (null to
    # infer the output size, otherwise an integer or triple), data_format (null means the global
    # default), dilation_rate (accepted but later ignored), activation, use_bias, initializers,
    # regularizers, constraints and extra base-layer keyword arguments. Forwards everything except
    # output_padding to the parent constructor, then normalizes output_padding to a triple and
    # stores it. Raises a value error if any output padding value is not strictly smaller than the
    # matching stride, plus any error from the parent constructor. Returns nothing.
    super(Conv3DTranspose, self).__init__(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activations.get(activation),
        use_bias=use_bias,
        kernel_initializer=initializers.get(kernel_initializer),
        bias_initializer=initializers.get(bias_initializer),
        kernel_regularizer=regularizers.get(kernel_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        kernel_constraint=constraints.get(kernel_constraint),
        bias_constraint=constraints.get(bias_constraint),
        **kwargs)

    self.output_padding = output_padding
    if self.output_padding is not None:
      self.output_padding = conv_utils.normalize_tuple(
          self.output_padding, 3, 'output_padding')
      for stride, out_pad in zip(self.strides, self.output_padding):
        if out_pad >= stride:
          raise ValueError('Stride ' + str(self.strides) + ' must be '
                           'greater than output padding ' +
                           str(self.output_padding))

  def build(self, input_shape):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the rank error passes the shape as a second argument to the exception instead
    # of formatting it into the message, so the message is a tuple-like pair rather than one string.
    # Creates the weights for the transposed 3D convolution. Input: a shape-like value that must
    # have exactly five dimensions with a defined channel dimension; otherwise raises a value
    # error. Creates a kernel weight shaped as kernel depth, height, width, filters, input
    # channels, sets an input spec requiring rank five and the observed channel size, and creates a
    # bias weight of length filters when use_bias is set (otherwise bias is null). Marks the layer
    # as built. Returns nothing.
    input_shape = tensor_shape.TensorShape(input_shape)
    if len(input_shape) != 5:
      raise ValueError('Inputs should have rank 5, received input shape:',
                       str(input_shape))
    channel_axis = self._get_channel_axis()
    if input_shape.dims[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined, found None: ' + str(input_shape))
    input_dim = int(input_shape[channel_axis])
    kernel_shape = self.kernel_size + (self.filters, input_dim)
    self.input_spec = InputSpec(ndim=5, axes={channel_axis: input_dim})

    self.kernel = self.add_weight(
        'kernel',
        shape=kernel_shape,
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        trainable=True,
        dtype=self.dtype)
    if self.use_bias:
      self.bias = self.add_weight(
          'bias',
          shape=(self.filters,),
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          trainable=True,
          dtype=self.dtype)
    else:
      self.bias = None
    self.built = True

  def call(self, inputs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: dilation_rate is not applied to the output size or the native op, and the bias
    # is added using a four-dimensional data format string even though the tensor has rank five.
    # Forward pass of the transposed 3D convolution. Input: a rank-five tensor. Reads the dynamic
    # depth, height and width, computes the output sizes from kernel size, padding, output padding
    # and stride, builds the dynamic output shape tensor and the five-element stride list, and
    # applies the native 3D transpose convolution with the uppercase padding string. In graph mode
    # sets the static output shape. Adds the bias (if enabled) and applies the activation (if any).
    # Returns the output tensor. No other side effects.
    inputs_shape = array_ops.shape(inputs)
    batch_size = inputs_shape[0]
    if self.data_format == 'channels_first':
      d_axis, h_axis, w_axis = 2, 3, 4
    else:
      d_axis, h_axis, w_axis = 1, 2, 3

    depth = inputs_shape[d_axis]
    height = inputs_shape[h_axis]
    width = inputs_shape[w_axis]

    kernel_d, kernel_h, kernel_w = self.kernel_size
    stride_d, stride_h, stride_w = self.strides

    if self.output_padding is None:
      out_pad_d = out_pad_h = out_pad_w = None
    else:
      out_pad_d, out_pad_h, out_pad_w = self.output_padding

    # Infer the dynamic output shape:
    out_depth = conv_utils.deconv_output_length(depth,
                                                kernel_d,
                                                padding=self.padding,
                                                output_padding=out_pad_d,
                                                stride=stride_d)
    out_height = conv_utils.deconv_output_length(height,
                                                 kernel_h,
                                                 padding=self.padding,
                                                 output_padding=out_pad_h,
                                                 stride=stride_h)
    out_width = conv_utils.deconv_output_length(width,
                                                kernel_w,
                                                padding=self.padding,
                                                output_padding=out_pad_w,
                                                stride=stride_w)
    if self.data_format == 'channels_first':
      output_shape = (batch_size, self.filters, out_depth, out_height,
                      out_width)
      strides = (1, 1, stride_d, stride_h, stride_w)
    else:
      output_shape = (batch_size, out_depth, out_height, out_width,
                      self.filters)
      strides = (1, stride_d, stride_h, stride_w, 1)

    output_shape_tensor = array_ops_stack.stack(output_shape)
    outputs = nn.conv3d_transpose(
        inputs,
        self.kernel,
        output_shape_tensor,
        strides,
        data_format=conv_utils.convert_data_format(self.data_format, ndim=5),
        padding=self.padding.upper())

    if not context.executing_eagerly():
      # Infer the static output shape:
      out_shape = self.compute_output_shape(inputs.shape)
      outputs.set_shape(out_shape)

    if self.use_bias:
      outputs = nn.bias_add(
          outputs,
          self.bias,
          data_format=conv_utils.convert_data_format(self.data_format, ndim=4))

    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def compute_output_shape(self, input_shape):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the dilation rate is ignored when computing the output lengths.
    # Computes the output shape of the transposed 3D convolution. Input: a shape-like value of
    # rank five. Copies it, sets the channel dimension to filters and replaces depth, height and
    # width by the transposed-convolution output lengths (from kernel size, padding, output
    # padding and stride; unknown stays unknown). Returns a tensor shape. No side effects.
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    output_shape = list(input_shape)
    if self.data_format == 'channels_first':
      c_axis, d_axis, h_axis, w_axis = 1, 2, 3, 4
    else:
      c_axis, d_axis, h_axis, w_axis = 4, 1, 2, 3

    kernel_d, kernel_h, kernel_w = self.kernel_size
    stride_d, stride_h, stride_w = self.strides

    if self.output_padding is None:
      out_pad_d = out_pad_h = out_pad_w = None
    else:
      out_pad_d, out_pad_h, out_pad_w = self.output_padding

    output_shape[c_axis] = self.filters
    output_shape[d_axis] = conv_utils.deconv_output_length(
        output_shape[d_axis],
        kernel_d,
        padding=self.padding,
        output_padding=out_pad_d,
        stride=stride_d)
    output_shape[h_axis] = conv_utils.deconv_output_length(
        output_shape[h_axis],
        kernel_h,
        padding=self.padding,
        output_padding=out_pad_h,
        stride=stride_h)
    output_shape[w_axis] = conv_utils.deconv_output_length(
        output_shape[w_axis],
        kernel_w,
        padding=self.padding,
        output_padding=out_pad_w,
        stride=stride_w)
    return tensor_shape.TensorShape(output_shape)

  def get_config(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: removes dilation_rate from the configuration even though the constructor
    # accepts it, so a layer rebuilt from this configuration always uses the default dilation.
    # Returns the parent layer configuration mapping without the dilation rate entry and with an
    # extra entry for output_padding (the normalized triple or null). No side effects.
    config = super(Conv3DTranspose, self).get_config()
    config.pop('dilation_rate')
    config['output_padding'] = self.output_padding
    return config


class SeparableConv(Conv):
  # [CLASS DOC]
  # HELPER: abstract base shared by SeparableConv1D and SeparableConv2D.
  # KNOWN ISSUE: it inherits the parent kernel attributes (kernel initializer, regularizer,
  # constraint) which stay unused; weights are held in separate depthwise and pointwise kernels
  # and the plain kernel weight is never created. The groups option is not exposed.
  # Generic depthwise separable convolution layer: a per-channel (depthwise) spatial convolution
  # with a depth multiplier followed by a one-by-one (pointwise) convolution that mixes channels
  # into the requested number of filters, plus optional bias and activation. It adds depthwise
  # and pointwise initializers, regularizers and constraints and the depth multiplier on top of the
  # generic convolution state. The forward pass is left to subclasses (call raises not-implemented
  # here).
  """Abstract base layer for separable nD convolution.

  This layer performs a depthwise convolution that acts separately on
  channels, followed by a pointwise convolution that mixes channels.
  If `use_bias` is True and a bias initializer is provided,
  it adds a bias vector to the output.
  It then optionally applies an activation function to produce the final output.

  Args:
    rank: An integer, the rank of the convolution, e.g. "2" for 2D convolution.
    filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
    kernel_size: A tuple or list of integers specifying the spatial
      dimensions of the filters. Can be a single integer to specify the same
      value for all spatial dimensions.
    strides: A tuple or list of integers specifying the strides
      of the convolution. Can be a single integer to specify the same value for
      all spatial dimensions.
      Specifying any `stride` value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: One of `"valid"` or `"same"` (case-insensitive).
      `"valid"` means no padding. `"same"` results in padding with zeros evenly
      to the left/right or up/down of the input such that output has the same
      height/width dimension as the input.
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, ..., channels)` while `channels_first` corresponds to
      inputs with shape `(batch_size, channels, ...)`.
    dilation_rate: An integer or tuple/list of 2 integers, specifying
      the dilation rate to use for dilated convolution.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any stride value != 1.
    depth_multiplier: The number of depthwise convolution output channels for
      each input channel. The total number of depthwise convolution output
      channels will be equal to `num_filters_in * depth_multiplier`.
    activation: Activation function to use.
      If you don't specify anything, no activation is applied (
      see `keras.activations`).
    use_bias: Boolean, whether the layer uses a bias.
    depthwise_initializer: An initializer for the depthwise convolution kernel (
      see `keras.initializers`). If None, then the default initializer (
      'glorot_uniform') will be used.
    pointwise_initializer: An initializer for the pointwise convolution kernel (
      see `keras.initializers`). If None, then the default initializer 
      ('glorot_uniform') will be used.
    bias_initializer: An initializer for the bias vector. If None, the default
      initializer ('zeros') will be used (see `keras.initializers`).
    depthwise_regularizer: Optional regularizer for the depthwise
      convolution kernel.
    pointwise_regularizer: Optional regularizer for the pointwise
      convolution kernel.
    bias_regularizer: Optional regularizer for the bias vector.
    activity_regularizer: Optional regularizer function for the output.
    depthwise_constraint: Optional projection function to be applied to the
      depthwise kernel after being updated by an `Optimizer` (e.g. used for
      norm constraints or value constraints for layer weights). The function
      must take as input the unprojected variable and must return the
      projected variable (which must have the same shape). Constraints are
      not safe to use when doing asynchronous distributed training.
    pointwise_constraint: Optional projection function to be applied to the
      pointwise kernel after being updated by an `Optimizer`.
    bias_constraint: Optional projection function to be applied to the
      bias after being updated by an `Optimizer`.
    trainable: Boolean, if `True` the weights of this layer will be marked as
      trainable (and listed in `layer.trainable_weights`).
  """

  def __init__(self,
               rank,
               filters,
               kernel_size,
               strides=1,
               padding='valid',
               data_format=None,
               dilation_rate=1,
               depth_multiplier=1,
               activation=None,
               use_bias=True,
               depthwise_initializer='glorot_uniform',
               pointwise_initializer='glorot_uniform',
               bias_initializer='zeros',
               depthwise_regularizer=None,
               pointwise_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               depthwise_constraint=None,
               pointwise_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               **kwargs):
    # [FUNCTION DOC]
    # Constructs the generic separable convolution layer. Inputs: rank, filters, kernel_size,
    # strides, padding, data_format, dilation_rate (all forwarded to the generic constructor),
    # depth_multiplier (number of depthwise output channels per input channel, default one),
    # activation, use_bias, depthwise, pointwise and bias initializers (glorot uniform for the
    # first two, zeros for the bias), the corresponding regularizers and constraints, an activity
    # regularizer, trainable, name and extra base-layer keyword arguments. The parent receives the
    # bias and activity settings but not kernel settings; the depthwise and pointwise initializers,
    # regularizers and constraints are resolved here and stored as attributes together with
    # depth_multiplier. Raises a value error under the same conditions as the parent constructor.
    # Returns nothing.
    super(SeparableConv, self).__init__(
        rank=rank,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activations.get(activation),
        use_bias=use_bias,
        bias_initializer=initializers.get(bias_initializer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        bias_constraint=bias_constraint,
        trainable=trainable,
        name=name,
        **kwargs)
    self.depth_multiplier = depth_multiplier
    self.depthwise_initializer = initializers.get(depthwise_initializer)
    self.pointwise_initializer = initializers.get(pointwise_initializer)
    self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
    self.pointwise_regularizer = regularizers.get(pointwise_regularizer)
    self.depthwise_constraint = constraints.get(depthwise_constraint)
    self.pointwise_constraint = constraints.get(pointwise_constraint)

  def build(self, input_shape):
    # [FUNCTION DOC]
    # Creates the weights for the separable convolution. Input: a shape-like value whose channel
    # dimension must be defined (otherwise raises a value error). Sets an input spec requiring
    # exactly rank plus two dimensions and the observed channel size. Creates a depthwise kernel
    # (kernel size, input channels, depth multiplier), a pointwise kernel (ones for each spatial
    # dimension, input channels times depth multiplier, filters) and, when use_bias is set, a bias
    # of length filters (otherwise bias is null). Marks the layer as built. Returns nothing.
    input_shape = tensor_shape.TensorShape(input_shape)
    channel_axis = self._get_channel_axis()
    if input_shape.dims[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = int(input_shape[channel_axis])
    self.input_spec = InputSpec(ndim=self.rank + 2,
                                axes={channel_axis: input_dim})
    depthwise_kernel_shape = self.kernel_size + (input_dim,
                                                 self.depth_multiplier)
    pointwise_kernel_shape = (
        1,) * self.rank + (self.depth_multiplier * input_dim, self.filters)

    self.depthwise_kernel = self.add_weight(
        name='depthwise_kernel',
        shape=depthwise_kernel_shape,
        initializer=self.depthwise_initializer,
        regularizer=self.depthwise_regularizer,
        constraint=self.depthwise_constraint,
        trainable=True,
        dtype=self.dtype)
    self.pointwise_kernel = self.add_weight(
        name='pointwise_kernel',
        shape=pointwise_kernel_shape,
        initializer=self.pointwise_initializer,
        regularizer=self.pointwise_regularizer,
        constraint=self.pointwise_constraint,
        trainable=True,
        dtype=self.dtype)
    if self.use_bias:
      self.bias = self.add_weight(
          name='bias',
          shape=(self.filters,),
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          trainable=True,
          dtype=self.dtype)
    else:
      self.bias = None
    self.built = True

  def call(self, inputs):
    # [FUNCTION DOC]
    # Abstract forward pass. Always raises a not-implemented error; concrete subclasses override
    # it. Input is ignored. No side effects.
    raise NotImplementedError

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a serializable mapping of the constructor arguments: filters, kernel size, strides,
    # padding, data format, depth multiplier, dilation rate, use_bias, the serialized activation,
    # depthwise, pointwise and bias initializers, regularizers (including activity) and
    # constraints, merged with the base layer configuration (layer-specific entries win on
    # conflict). The parent convolution configuration is not used, so kernel-specific and groups
    # entries are absent. No side effects.
    config = {
        'filters':
            self.filters,
        'kernel_size':
            self.kernel_size,
        'strides':
            self.strides,
        'padding':
            self.padding,
        'data_format':
            self.data_format,
        'depth_multiplier':
            self.depth_multiplier,
        'dilation_rate':
            self.dilation_rate,
        'activation':
            activations.serialize(self.activation),
        'use_bias':
            self.use_bias,
        'depthwise_initializer':
            initializers.serialize(self.depthwise_initializer),
        'pointwise_initializer':
            initializers.serialize(self.pointwise_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'depthwise_regularizer':
            regularizers.serialize(self.depthwise_regularizer),
        'pointwise_regularizer':
            regularizers.serialize(self.pointwise_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'depthwise_constraint':
            constraints.serialize(self.depthwise_constraint),
        'pointwise_constraint':
            constraints.serialize(self.pointwise_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint)
    }
    base_config = super(SeparableConv, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class SeparableConv1D(SeparableConv):
  # [CLASS DOC]
  # Public depthwise separable 1D convolution layer (for example over sequences). Fixes the rank
  # at one and implements the forward pass by temporarily adding a dummy spatial dimension and
  # running the native 2D separable convolution. Supports valid, same and causal padding. Typical
  # input is batch, steps, channels (channels last) or batch, channels, steps.
  """Depthwise separable 1D convolution.

  This layer performs a depthwise convolution that acts separately on
  channels, followed by a pointwise convolution that mixes channels.
  If `use_bias` is True and a bias initializer is provided,
  it adds a bias vector to the output.
  It then optionally applies an activation function to produce the final output.

  Args:
    filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
    kernel_size: A single integer specifying the spatial
      dimensions of the filters.
    strides: A single integer specifying the strides
      of the convolution.
      Specifying any `stride` value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: One of `"valid"`, `"same"`, or `"causal"` (case-insensitive).
      `"valid"` means no padding. `"same"` results in padding with zeros evenly
      to the left/right or up/down of the input such that output has the same
      height/width dimension as the input. `"causal"` results in causal
      (dilated) convolutions, e.g. `output[t]` does not depend on `input[t+1:]`.
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, length, channels)` while `channels_first` corresponds to
      inputs with shape `(batch_size, channels, length)`.
    dilation_rate: A single integer, specifying
      the dilation rate to use for dilated convolution.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any stride value != 1.
    depth_multiplier: The number of depthwise convolution output channels for
      each input channel. The total number of depthwise convolution output
      channels will be equal to `num_filters_in * depth_multiplier`.
    activation: Activation function to use.
      If you don't specify anything, no activation is applied (
      see `keras.activations`).
    use_bias: Boolean, whether the layer uses a bias.
    depthwise_initializer: An initializer for the depthwise convolution kernel (
      see `keras.initializers`). If None, then the default initializer (
      'glorot_uniform') will be used.
    pointwise_initializer: An initializer for the pointwise convolution kernel (
      see `keras.initializers`). If None, then the default initializer 
      ('glorot_uniform') will be used.
    bias_initializer: An initializer for the bias vector. If None, the default
      initializer ('zeros') will be used (see `keras.initializers`).
    depthwise_regularizer: Optional regularizer for the depthwise
      convolution kernel (see `keras.regularizers`).
    pointwise_regularizer: Optional regularizer for the pointwise
      convolution kernel (see `keras.regularizers`).
    bias_regularizer: Optional regularizer for the bias vector (
      see `keras.regularizers`).
    activity_regularizer: Optional regularizer function for the output (
      see `keras.regularizers`).
    depthwise_constraint: Optional projection function to be applied to the
      depthwise kernel after being updated by an `Optimizer` (e.g. used for
      norm constraints or value constraints for layer weights). The function
      must take as input the unprojected variable and must return the
      projected variable (which must have the same shape). Constraints are
      not safe to use when doing asynchronous distributed training (
      see `keras.constraints`).
    pointwise_constraint: Optional projection function to be applied to the
      pointwise kernel after being updated by an `Optimizer` (
      see `keras.constraints`).
    bias_constraint: Optional projection function to be applied to the
      bias after being updated by an `Optimizer` (
      see `keras.constraints`).
    trainable: Boolean, if `True` the weights of this layer will be marked as
      trainable (and listed in `layer.trainable_weights`).

  Input shape:
    3D tensor with shape:
    `(batch_size, channels, steps)` if data_format='channels_first'
    or 5D tensor with shape:
    `(batch_size, steps, channels)` if data_format='channels_last'.

  Output shape:
    3D tensor with shape:
    `(batch_size, filters, new_steps)` if data_format='channels_first'
    or 3D tensor with shape:
    `(batch_size,  new_steps, filters)` if data_format='channels_last'.
    `new_steps` value might have changed due to padding or strides.

  Returns:
    A tensor of rank 3 representing
    `activation(separableconv1d(inputs, kernel) + bias)`.

  Raises:
    ValueError: when both `strides` > 1 and `dilation_rate` > 1.
  """

  def __init__(self,
               filters,
               kernel_size,
               strides=1,
               padding='valid',
               data_format=None,
               dilation_rate=1,
               depth_multiplier=1,
               activation=None,
               use_bias=True,
               depthwise_initializer='glorot_uniform',
               pointwise_initializer='glorot_uniform',
               bias_initializer='zeros',
               depthwise_regularizer=None,
               pointwise_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               depthwise_constraint=None,
               pointwise_constraint=None,
               bias_constraint=None,
               **kwargs):
    # [FUNCTION DOC]
    # Constructs a 1D separable convolution layer by forwarding to the generic separable
    # constructor with rank one. Inputs: filters, kernel_size (integer or one-element sequence),
    # strides, padding (valid by default; same or causal), data_format (null means the global
    # default), dilation_rate, depth_multiplier (default one), activation, use_bias, depthwise,
    # pointwise and bias initializers, regularizers and constraints, an activity regularizer, and
    # extra base-layer keyword arguments. Names or configs are resolved to objects first. Raises a
    # value error under the same conditions as the parent constructor. Returns nothing.
    super(SeparableConv1D, self).__init__(
        rank=1,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        depth_multiplier=depth_multiplier,
        activation=activations.get(activation),
        use_bias=use_bias,
        depthwise_initializer=initializers.get(depthwise_initializer),
        pointwise_initializer=initializers.get(pointwise_initializer),
        bias_initializer=initializers.get(bias_initializer),
        depthwise_regularizer=regularizers.get(depthwise_regularizer),
        pointwise_regularizer=regularizers.get(pointwise_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        depthwise_constraint=constraints.get(depthwise_constraint),
        pointwise_constraint=constraints.get(pointwise_constraint),
        bias_constraint=constraints.get(bias_constraint),
        **kwargs)

  def call(self, inputs):
    # [FUNCTION DOC]
    # Forward pass of the 1D separable convolution. Input: a rank-three tensor. For causal
    # padding, left-pads the time axis first and uses valid padding in the native op. Expands the
    # input and both kernels with an extra spatial dimension of size one (and a dilation of one
    # along it), applies the native 2D separable convolution with the duplicated stride, adds the
    # bias if enabled, removes the extra dimension, and applies the activation if any. Returns the
    # output tensor. No other side effects.
    if self.padding == 'causal':
      inputs = array_ops.pad(inputs, self._compute_causal_padding(inputs))
    if self.data_format == 'channels_last':
      strides = (1,) + self.strides * 2 + (1,)
      spatial_start_dim = 1
    else:
      strides = (1, 1) + self.strides * 2
      spatial_start_dim = 2

    # Explicitly broadcast inputs and kernels to 4D.
    # TODO(fchollet): refactor when a native separable_conv1d op is available.
    inputs = array_ops.expand_dims(inputs, spatial_start_dim)
    depthwise_kernel = array_ops.expand_dims(self.depthwise_kernel, 0)
    pointwise_kernel = array_ops.expand_dims(self.pointwise_kernel, 0)
    dilation_rate = (1,) + self.dilation_rate

    if self.padding == 'causal':
      op_padding = 'valid'
    else:
      op_padding = self.padding
    outputs = nn.separable_conv2d(
        inputs,
        depthwise_kernel,
        pointwise_kernel,
        strides=strides,
        padding=op_padding.upper(),
        rate=dilation_rate,
        data_format=conv_utils.convert_data_format(self.data_format, ndim=4))

    if self.use_bias:
      outputs = nn.bias_add(
          outputs,
          self.bias,
          data_format=conv_utils.convert_data_format(self.data_format, ndim=4))

    outputs = array_ops.squeeze(outputs, [spatial_start_dim])

    if self.activation is not None:
      return self.activation(outputs)
    return outputs


class SeparableConv2D(SeparableConv):
  # [CLASS DOC]
  # Public depthwise separable 2D convolution layer (for example over images). Fixes the rank at
  # two and implements the forward pass with the native 2D separable convolution op. Supports
  # valid and same padding, channels last or first. Typical input is batch, rows, columns,
  # channels.
  """Depthwise separable 2D convolution.

  Separable convolutions consist of first performing
  a depthwise spatial convolution
  (which acts on each input channel separately)
  followed by a pointwise convolution which mixes the resulting
  output channels. The `depth_multiplier` argument controls how many
  output channels are generated per input channel in the depthwise step.

  Intuitively, separable convolutions can be understood as
  a way to factorize a convolution kernel into two smaller kernels,
  or as an extreme version of an Inception block.

  Args:
    filters: Integer, the dimensionality of the output space
      (i.e. the number of output filters in the convolution).
    kernel_size: An integer or tuple/list of 2 integers, specifying the
      height and width of the 2D convolution window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 2 integers,
      specifying the strides of the convolution along the height and width.
      Can be a single integer to specify the same value for
      all spatial dimensions. Current implementation only supports equal 
      length strides in the row and column dimensions.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: one of `"valid"` or `"same"` (case-insensitive).
      `"valid"` means no padding. `"same"` results in padding with zeros evenly
      to the left/right or up/down of the input such that output has the same
      height/width dimension as the input.
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, height, width, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch_size, channels, height, width)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".
    dilation_rate: An integer or tuple/list of 2 integers, specifying
      the dilation rate to use for dilated convolution.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any `strides` value != 1.
    depth_multiplier: The number of depthwise convolution output channels
      for each input channel.
      The total number of depthwise convolution output
      channels will be equal to `filters_in * depth_multiplier`.
    activation: Activation function to use.
      If you don't specify anything, no activation is applied (
      see `keras.activations`).
    use_bias: Boolean, whether the layer uses a bias vector.
    depthwise_initializer: An initializer for the depthwise convolution kernel (
      see `keras.initializers`). If None, then the default initializer (
      'glorot_uniform') will be used.
    pointwise_initializer: An initializer for the pointwise convolution kernel (
      see `keras.initializers`). If None, then the default initializer 
      ('glorot_uniform') will be used.
    bias_initializer: An initializer for the bias vector. If None, the default
      initializer ('zeros') will be used (see `keras.initializers`).
    depthwise_regularizer: Regularizer function applied to
      the depthwise kernel matrix (see `keras.regularizers`).
    pointwise_regularizer: Regularizer function applied to
      the pointwise kernel matrix (see `keras.regularizers`).
    bias_regularizer: Regularizer function applied to the bias vector (
      see `keras.regularizers`).
    activity_regularizer: Regularizer function applied to
      the output of the layer (its "activation") (
      see `keras.regularizers`).
    depthwise_constraint: Constraint function applied to
      the depthwise kernel matrix (
      see `keras.constraints`).
    pointwise_constraint: Constraint function applied to
      the pointwise kernel matrix (
      see `keras.constraints`).
    bias_constraint: Constraint function applied to the bias vector (
      see `keras.constraints`).

  Input shape:
    4D tensor with shape:
    `(batch_size, channels, rows, cols)` if data_format='channels_first'
    or 4D tensor with shape:
    `(batch_size, rows, cols, channels)` if data_format='channels_last'.

  Output shape:
    4D tensor with shape:
    `(batch_size, filters, new_rows, new_cols)` if data_format='channels_first'
    or 4D tensor with shape:
    `(batch_size, new_rows, new_cols, filters)` if data_format='channels_last'.
    `rows` and `cols` values might have changed due to padding.

  Returns:
    A tensor of rank 4 representing
    `activation(separableconv2d(inputs, kernel) + bias)`.

  Raises:
    ValueError: if `padding` is "causal".
    ValueError: when both `strides` > 1 and `dilation_rate` > 1.
  """

  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1),
               depth_multiplier=1,
               activation=None,
               use_bias=True,
               depthwise_initializer='glorot_uniform',
               pointwise_initializer='glorot_uniform',
               bias_initializer='zeros',
               depthwise_regularizer=None,
               pointwise_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               depthwise_constraint=None,
               pointwise_constraint=None,
               bias_constraint=None,
               **kwargs):
    # [FUNCTION DOC]
    # Constructs a 2D separable convolution layer by forwarding to the generic separable
    # constructor with rank two. Inputs: filters, kernel_size (integer or pair), strides and
    # dilation_rate (default one per direction), padding (valid by default, or same), data_format
    # (null means the global default), depth_multiplier (default one), activation, use_bias,
    # depthwise, pointwise and bias initializers, regularizers and constraints, an activity
    # regularizer, and extra base-layer keyword arguments. Names or configs are resolved to objects
    # first. Raises a value error under the same conditions as the parent constructor (causal
    # padding is rejected). Returns nothing.
    super(SeparableConv2D, self).__init__(
        rank=2,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        depth_multiplier=depth_multiplier,
        activation=activations.get(activation),
        use_bias=use_bias,
        depthwise_initializer=initializers.get(depthwise_initializer),
        pointwise_initializer=initializers.get(pointwise_initializer),
        bias_initializer=initializers.get(bias_initializer),
        depthwise_regularizer=regularizers.get(depthwise_regularizer),
        pointwise_regularizer=regularizers.get(pointwise_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        depthwise_constraint=constraints.get(depthwise_constraint),
        pointwise_constraint=constraints.get(pointwise_constraint),
        bias_constraint=constraints.get(bias_constraint),
        **kwargs)

  def call(self, inputs):
    # [FUNCTION DOC]
    # Forward pass of the 2D separable convolution. Input: a rank-four tensor. Builds the four
    # element stride for the data format, applies the native separable convolution with the
    # depthwise and pointwise kernels, uppercase padding and the dilation rate, adds the bias if
    # enabled, and applies the activation if any. Returns the output tensor. No other side effects.
    # Apply the actual ops.
    if self.data_format == 'channels_last':
      strides = (1,) + self.strides + (1,)
    else:
      strides = (1, 1) + self.strides
    outputs = nn.separable_conv2d(
        inputs,
        self.depthwise_kernel,
        self.pointwise_kernel,
        strides=strides,
        padding=self.padding.upper(),
        rate=self.dilation_rate,
        data_format=conv_utils.convert_data_format(self.data_format, ndim=4))

    if self.use_bias:
      outputs = nn.bias_add(
          outputs,
          self.bias,
          data_format=conv_utils.convert_data_format(self.data_format, ndim=4))

    if self.activation is not None:
      return self.activation(outputs)
    return outputs


class DepthwiseConv2D(Conv2D):
  # [CLASS DOC]
  # KNOWN ISSUE: the docstring says a value error is raised when strides and dilation are both
  # above one; no such check exists here (causal padding is rejected by the 2D parent validation).
  # Depthwise 2D convolution layer: applies a separate spatial filter set to each input channel
  # (without the pointwise mixing step), producing input channels times depth multiplier output
  # channels. Subclasses the 2D convolution layer with filters set to null, and replaces the
  # kernel with a depthwise kernel (with its own initializer, regularizer and constraint). Overrides
  # build, call, compute_output_shape and get_config; the forward pass uses the backend depthwise
  # convolution.
  """Depthwise 2D convolution.

  Depthwise convolution is a type of convolution in which a single convolutional
  filter is apply to each input channel (i.e. in a depthwise way).
  You can understand depthwise convolution as being
  the first step in a depthwise separable convolution.

  It is implemented via the following steps:

  - Split the input into individual channels.
  - Convolve each input with the layer's kernel (called a depthwise kernel).
  - Stack the convolved outputs together (along the channels axis).

  Unlike a regular 2D convolution, depthwise convolution does not mix
  information across different input channels.

  The `depth_multiplier` argument controls how many
  output channels are generated per input channel in the depthwise step.

  Args:
    kernel_size: An integer or tuple/list of 2 integers, specifying the
      height and width of the 2D convolution window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 2 integers,
      specifying the strides of the convolution along the height and width.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: one of `'valid'` or `'same'` (case-insensitive).
      `"valid"` means no padding. `"same"` results in padding with zeros evenly
      to the left/right or up/down of the input such that output has the same
      height/width dimension as the input.
    depth_multiplier: The number of depthwise convolution output channels
      for each input channel.
      The total number of depthwise convolution output
      channels will be equal to `filters_in * depth_multiplier`.
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, height, width, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch_size, channels, height, width)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be 'channels_last'.
    dilation_rate: An integer or tuple/list of 2 integers, specifying
      the dilation rate to use for dilated convolution.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any `strides` value != 1.
    activation: Activation function to use.
      If you don't specify anything, no activation is applied (
      see `keras.activations`).
    use_bias: Boolean, whether the layer uses a bias vector.
    depthwise_initializer: Initializer for the depthwise kernel matrix (
      see `keras.initializers`). If None, the default initializer (
      'glorot_uniform') will be used.
    bias_initializer: Initializer for the bias vector (
      see `keras.initializers`). If None, the default initializer (
      'zeros') will bs used.
    depthwise_regularizer: Regularizer function applied to
      the depthwise kernel matrix (see `keras.regularizers`).
    bias_regularizer: Regularizer function applied to the bias vector (
      see `keras.regularizers`).
    activity_regularizer: Regularizer function applied to
      the output of the layer (its 'activation') (
      see `keras.regularizers`).
    depthwise_constraint: Constraint function applied to
      the depthwise kernel matrix (
      see `keras.constraints`).
    bias_constraint: Constraint function applied to the bias vector (
      see `keras.constraints`).

  Input shape:
    4D tensor with shape:
    `[batch_size, channels, rows, cols]` if data_format='channels_first'
    or 4D tensor with shape:
    `[batch_size, rows, cols, channels]` if data_format='channels_last'.

  Output shape:
    4D tensor with shape:
    `[batch_size, channels * depth_multiplier, new_rows, new_cols]` if
    data_format='channels_first' or 4D tensor with shape:
    `[batch_size, new_rows, new_cols, channels * depth_multiplier]` if
    data_format='channels_last'. `rows` and `cols` values might have
    changed due to padding.

  Returns:
    A tensor of rank 4 representing
    `activation(depthwiseconv2d(inputs, kernel) + bias)`.

  Raises:
    ValueError: if `padding` is "causal".
    ValueError: when both `strides` > 1 and `dilation_rate` > 1.
  """

  def __init__(self,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               depth_multiplier=1,
               data_format=None,
               dilation_rate=(1, 1),
               activation=None,
               use_bias=True,
               depthwise_initializer='glorot_uniform',
               bias_initializer='zeros',
               depthwise_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               depthwise_constraint=None,
               bias_constraint=None,
               **kwargs):
    # [FUNCTION DOC]
    # Constructs a depthwise 2D convolution layer. Inputs: kernel_size (integer or pair), strides
    # (default one per direction), padding (valid by default, or same), depth_multiplier (output
    # channels per input channel, default one), data_format (null means the global default),
    # dilation_rate, activation, use_bias, depthwise and bias initializers, depthwise and bias
    # regularizers, an activity regularizer, depthwise and bias constraints, and extra base-layer
    # keyword arguments. Calls the 2D parent constructor with filters set to null (the kernel
    # initializer, regularizer and constraint are left at the parent defaults and never used), then
    # stores depth_multiplier and the resolved depthwise initializer, regularizer and constraint,
    # and overrides the bias initializer with the resolved argument. Raises a value error under the
    # parent constructor conditions. Returns nothing.
    super(DepthwiseConv2D, self).__init__(
        filters=None,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        bias_constraint=bias_constraint,
        **kwargs)
    self.depth_multiplier = depth_multiplier
    self.depthwise_initializer = initializers.get(depthwise_initializer)
    self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
    self.depthwise_constraint = constraints.get(depthwise_constraint)
    self.bias_initializer = initializers.get(bias_initializer)

  def build(self, input_shape):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the rank check only rejects shapes with fewer than four dimensions (while the
    # input spec later demands exactly four), and the error passes the shape as a second exception
    # argument rather than formatting it into the message.
    # Creates the weights for the depthwise convolution. Input: a shape with at least four
    # dimensions and a defined channel dimension; otherwise raises a value error. Creates a
    # depthwise kernel (kernel height, kernel width, input channels, depth multiplier) and, when
    # use_bias is set, a bias of length input channels times depth multiplier (otherwise bias is
    # null). Sets the input spec to rank four with the observed channel size and marks the layer as
    # built. Returns nothing.
    if len(input_shape) < 4:
      raise ValueError('Inputs to `DepthwiseConv2D` should have rank 4. '
                       'Received input shape:', str(input_shape))
    input_shape = tensor_shape.TensorShape(input_shape)
    channel_axis = self._get_channel_axis()
    if input_shape.dims[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs to '
                       '`DepthwiseConv2D` '
                       'should be defined. Found `None`.')
    input_dim = int(input_shape[channel_axis])
    depthwise_kernel_shape = (self.kernel_size[0],
                              self.kernel_size[1],
                              input_dim,
                              self.depth_multiplier)

    self.depthwise_kernel = self.add_weight(
        shape=depthwise_kernel_shape,
        initializer=self.depthwise_initializer,
        name='depthwise_kernel',
        regularizer=self.depthwise_regularizer,
        constraint=self.depthwise_constraint)

    if self.use_bias:
      self.bias = self.add_weight(shape=(input_dim * self.depth_multiplier,),
                                  initializer=self.bias_initializer,
                                  name='bias',
                                  regularizer=self.bias_regularizer,
                                  constraint=self.bias_constraint)
    else:
      self.bias = None
    # Set input spec.
    self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
    self.built = True

  def call(self, inputs):
    # [FUNCTION DOC]
    # Forward pass of the depthwise convolution. Input: a rank-four tensor. Applies the backend
    # depthwise 2D convolution with the layer strides, padding, dilation rate and data format,
    # adds the bias if enabled, and applies the activation if any. Returns the output tensor.
    # No other side effects.
    outputs = backend.depthwise_conv2d(
        inputs,
        self.depthwise_kernel,
        strides=self.strides,
        padding=self.padding,
        dilation_rate=self.dilation_rate,
        data_format=self.data_format)

    if self.use_bias:
      outputs = backend.bias_add(
          outputs,
          self.bias,
          data_format=self.data_format)

    if self.activation is not None:
      return self.activation(outputs)

    return outputs

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    # [FUNCTION DOC]
    # Computes the output shape for a four-dimensional input shape (a decorator converts between
    # tensor shape objects and plain tuples). Output channels are input channels times depth
    # multiplier; rows and columns are replaced by their convolved lengths using kernel size,
    # padding, stride and dilation. Returns a four-element shape in the layer data format
    # (batch size is kept unchanged). No side effects.
    if self.data_format == 'channels_first':
      rows = input_shape[2]
      cols = input_shape[3]
      out_filters = input_shape[1] * self.depth_multiplier
    elif self.data_format == 'channels_last':
      rows = input_shape[1]
      cols = input_shape[2]
      out_filters = input_shape[3] * self.depth_multiplier

    rows = conv_utils.conv_output_length(rows, self.kernel_size[0],
                                         self.padding,
                                         self.strides[0],
                                         self.dilation_rate[0])
    cols = conv_utils.conv_output_length(cols, self.kernel_size[1],
                                         self.padding,
                                         self.strides[1],
                                         self.dilation_rate[1])
    if self.data_format == 'channels_first':
      return (input_shape[0], out_filters, rows, cols)
    elif self.data_format == 'channels_last':
      return (input_shape[0], rows, cols, out_filters)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the parent 2D convolution configuration without the filters and kernel initializer,
    # regularizer and constraint entries, and with added entries for depth_multiplier and the
    # serialized depthwise initializer, regularizer and constraint. No side effects.
    config = super(DepthwiseConv2D, self).get_config()
    config.pop('filters')
    config.pop('kernel_initializer')
    config.pop('kernel_regularizer')
    config.pop('kernel_constraint')
    config['depth_multiplier'] = self.depth_multiplier
    config['depthwise_initializer'] = initializers.serialize(
        self.depthwise_initializer)
    config['depthwise_regularizer'] = regularizers.serialize(
        self.depthwise_regularizer)
    config['depthwise_constraint'] = constraints.serialize(
        self.depthwise_constraint)
    return config


class UpSampling1D(Layer):
  # [CLASS DOC]
  # Parameter-free layer that upsamples a sequence by repeating each time step a fixed integer
  # number of times along the time axis. Holds one attribute, the integer upsampling factor, and
  # requires rank-three input (batch, steps, features).
  """Upsampling layer for 1D inputs.

  Repeats each temporal step `size` times along the time axis.

  Examples:

  >>> input_shape = (2, 2, 3)
  >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
  >>> print(x)
  [[[ 0  1  2]
    [ 3  4  5]]
   [[ 6  7  8]
    [ 9 10 11]]]
  >>> y = tf.keras.layers.UpSampling1D(size=2)(x)
  >>> print(y)
  tf.Tensor(
    [[[ 0  1  2]
      [ 0  1  2]
      [ 3  4  5]
      [ 3  4  5]]
     [[ 6  7  8]
      [ 6  7  8]
      [ 9 10 11]
      [ 9 10 11]]], shape=(2, 4, 3), dtype=int64)

  Args:
    size: Integer. Upsampling factor.

  Input shape:
    3D tensor with shape: `(batch_size, steps, features)`.

  Output shape:
    3D tensor with shape: `(batch_size, upsampled_steps, features)`.
  """

  def __init__(self, size=2, **kwargs):
    # [FUNCTION DOC]
    # Constructs the layer. Inputs: size (integer upsampling factor, default two; converted to an
    # integer) and base-layer keyword arguments such as name. Stores the factor and an input spec
    # requiring rank three. Returns nothing.
    super(UpSampling1D, self).__init__(**kwargs)
    self.size = int(size)
    self.input_spec = InputSpec(ndim=3)

  def compute_output_shape(self, input_shape):
    # [FUNCTION DOC]
    # Computes the output shape for a rank-three input shape: batch and feature sizes are kept
    # and the step count is multiplied by the upsampling factor (unknown stays unknown). Returns a
    # tensor shape. No side effects.
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    size = self.size * input_shape[1] if input_shape[1] is not None else None
    return tensor_shape.TensorShape([input_shape[0], size, input_shape[2]])

  def call(self, inputs):
    # [FUNCTION DOC]
    # Forward pass. Input: a rank-three tensor. Repeats each element along the time axis (axis
    # one) the configured number of times. Returns the upsampled tensor. No side effects.
    output = backend.repeat_elements(inputs, self.size, axis=1)
    return output

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the base layer configuration mapping extended with the upsampling factor. No side
    # effects.
    config = {'size': self.size}
    base_config = super(UpSampling1D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class UpSampling2D(Layer):
  # [CLASS DOC]
  # Parameter-free layer that upsamples images by an integer factor per spatial axis, using
  # either nearest-neighbor repetition or bilinear interpolation. Holds the size pair, the data
  # format and the interpolation mode, and requires rank-four input.
  """Upsampling layer for 2D inputs.

  Repeats the rows and columns of the data
  by `size[0]` and `size[1]` respectively.

  Examples:

  >>> input_shape = (2, 2, 1, 3)
  >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
  >>> print(x)
  [[[[ 0  1  2]]
    [[ 3  4  5]]]
   [[[ 6  7  8]]
    [[ 9 10 11]]]]
  >>> y = tf.keras.layers.UpSampling2D(size=(1, 2))(x)
  >>> print(y)
  tf.Tensor(
    [[[[ 0  1  2]
       [ 0  1  2]]
      [[ 3  4  5]
       [ 3  4  5]]]
     [[[ 6  7  8]
       [ 6  7  8]]
      [[ 9 10 11]
       [ 9 10 11]]]], shape=(2, 2, 2, 3), dtype=int64)

  Args:
    size: Int, or tuple of 2 integers.
      The upsampling factors for rows and columns.
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, height, width, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch_size, channels, height, width)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".
    interpolation: A string, one of `nearest` or `bilinear`.

  Input shape:
    4D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, rows, cols, channels)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, channels, rows, cols)`

  Output shape:
    4D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, upsampled_rows, upsampled_cols, channels)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, channels, upsampled_rows, upsampled_cols)`
  """

  def __init__(self,
               size=(2, 2),
               data_format=None,
               interpolation='nearest',
               **kwargs):
    # [FUNCTION DOC]
    # Constructs the layer. Inputs: size (integer or pair of integers, default two by two),
    # data_format (channels last or first; null means the global default), interpolation (either
    # nearest or bilinear, default nearest) and base-layer keyword arguments. Normalizes and stores
    # the values and sets an input spec requiring rank four. Raises a value error if interpolation
    # is not one of the two allowed names, or if size or data_format cannot be normalized. Returns
    # nothing.
    super(UpSampling2D, self).__init__(**kwargs)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.size = conv_utils.normalize_tuple(size, 2, 'size')
    if interpolation not in {'nearest', 'bilinear'}:
      raise ValueError('`interpolation` argument should be one of `"nearest"` '
                       'or `"bilinear"`.')
    self.interpolation = interpolation
    self.input_spec = InputSpec(ndim=4)

  def compute_output_shape(self, input_shape):
    # [FUNCTION DOC]
    # Computes the output shape for a rank-four input shape: batch and channel sizes are kept,
    # and height and width are multiplied by the respective upsampling factors (unknown stays
    # unknown), with the spatial axes located according to the data format. Returns a tensor
    # shape. No side effects.
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_first':
      height = self.size[0] * input_shape[
          2] if input_shape[2] is not None else None
      width = self.size[1] * input_shape[
          3] if input_shape[3] is not None else None
      return tensor_shape.TensorShape(
          [input_shape[0], input_shape[1], height, width])
    else:
      height = self.size[0] * input_shape[
          1] if input_shape[1] is not None else None
      width = self.size[1] * input_shape[
          2] if input_shape[2] is not None else None
      return tensor_shape.TensorShape(
          [input_shape[0], height, width, input_shape[3]])

  def call(self, inputs):
    # [FUNCTION DOC]
    # Forward pass. Input: a rank-four tensor. Resizes the images with the backend image resizing
    # helper using the row and column factors, the data format and the interpolation mode. Returns
    # the upsampled tensor. No side effects.
    return backend.resize_images(
        inputs, self.size[0], self.size[1], self.data_format,
        interpolation=self.interpolation)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the base layer configuration mapping extended with size, data format and
    # interpolation mode. No side effects.
    config = {
        'size': self.size,
        'data_format': self.data_format,
        'interpolation': self.interpolation
    }
    base_config = super(UpSampling2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class UpSampling3D(Layer):
  # [CLASS DOC]
  # Parameter-free layer that upsamples volumetric data by repeating along each of the three
  # spatial axes by an integer factor. Holds the size triple and the data format, and requires
  # rank-five input.
  """Upsampling layer for 3D inputs.

  Repeats the 1st, 2nd and 3rd dimensions
  of the data by `size[0]`, `size[1]` and `size[2]` respectively.

  Examples:

  >>> input_shape = (2, 1, 2, 1, 3)
  >>> x = tf.constant(1, shape=input_shape)
  >>> y = tf.keras.layers.UpSampling3D(size=2)(x)
  >>> print(y.shape)
  (2, 2, 4, 2, 3)

  Args:
    size: Int, or tuple of 3 integers.
      The upsampling factors for dim1, dim2 and dim3.
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
      while `channels_first` corresponds to inputs with shape
      `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".

  Input shape:
    5D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, dim1, dim2, dim3, channels)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, channels, dim1, dim2, dim3)`

  Output shape:
    5D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, upsampled_dim1, upsampled_dim2, upsampled_dim3, channels)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, channels, upsampled_dim1, upsampled_dim2, upsampled_dim3)`
  """

  def __init__(self, size=(2, 2, 2), data_format=None, **kwargs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the attributes are set before the base layer constructor runs, unlike the
    # sibling layers, which call it first.
    # Constructs the layer. Inputs: size (integer or triple of integers, default two in each
    # direction), data_format (null means the global default) and base-layer keyword arguments.
    # Normalizes and stores size and data format, sets an input spec requiring rank five, then
    # runs the base constructor. Raises a value error if size or data_format cannot be normalized.
    # Returns nothing.
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.size = conv_utils.normalize_tuple(size, 3, 'size')
    self.input_spec = InputSpec(ndim=5)
    super(UpSampling3D, self).__init__(**kwargs)

  def compute_output_shape(self, input_shape):
    # [FUNCTION DOC]
    # Computes the output shape for a rank-five input shape: batch and channel sizes are kept,
    # and the three spatial sizes are multiplied by the respective factors (unknown stays
    # unknown), with the spatial axes located according to the data format. Returns a tensor
    # shape. No side effects.
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_first':
      dim1 = self.size[0] * input_shape[
          2] if input_shape[2] is not None else None
      dim2 = self.size[1] * input_shape[
          3] if input_shape[3] is not None else None
      dim3 = self.size[2] * input_shape[
          4] if input_shape[4] is not None else None
      return tensor_shape.TensorShape(
          [input_shape[0], input_shape[1], dim1, dim2, dim3])
    else:
      dim1 = self.size[0] * input_shape[
          1] if input_shape[1] is not None else None
      dim2 = self.size[1] * input_shape[
          2] if input_shape[2] is not None else None
      dim3 = self.size[2] * input_shape[
          3] if input_shape[3] is not None else None
      return tensor_shape.TensorShape(
          [input_shape[0], dim1, dim2, dim3, input_shape[4]])

  def call(self, inputs):
    # [FUNCTION DOC]
    # Forward pass. Input: a rank-five tensor. Repeats the data along each spatial axis using the
    # backend volume resizing helper with the three factors and the data format. Returns the
    # upsampled tensor. No side effects.
    return backend.resize_volumes(
        inputs, self.size[0], self.size[1], self.size[2], self.data_format)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the base layer configuration mapping extended with size and data format. No side
    # effects.
    config = {'size': self.size, 'data_format': self.data_format}
    base_config = super(UpSampling3D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class ZeroPadding1D(Layer):
  # [CLASS DOC]
  # Parameter-free layer that pads the time axis of a rank-three input with zeros at the start
  # and end. Holds the normalized pair of left and right padding amounts.
  """Zero-padding layer for 1D input (e.g. temporal sequence).

  Examples:

  >>> input_shape = (2, 2, 3)
  >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
  >>> print(x)
  [[[ 0  1  2]
    [ 3  4  5]]
   [[ 6  7  8]
    [ 9 10 11]]]
  >>> y = tf.keras.layers.ZeroPadding1D(padding=2)(x)
  >>> print(y)
  tf.Tensor(
    [[[ 0  0  0]
      [ 0  0  0]
      [ 0  1  2]
      [ 3  4  5]
      [ 0  0  0]
      [ 0  0  0]]
     [[ 0  0  0]
      [ 0  0  0]
      [ 6  7  8]
      [ 9 10 11]
      [ 0  0  0]
      [ 0  0  0]]], shape=(2, 6, 3), dtype=int64)

  Args:
      padding: Int, or tuple of int (length 2), or dictionary.
          - If int:
          How many zeros to add at the beginning and end of
          the padding dimension (axis 1).
          - If tuple of int (length 2):
          How many zeros to add at the beginning and the end of
          the padding dimension (`(left_pad, right_pad)`).

  Input shape:
      3D tensor with shape `(batch_size, axis_to_pad, features)`

  Output shape:
      3D tensor with shape `(batch_size, padded_axis, features)`
  """

  def __init__(self, padding=1, **kwargs):
    # [FUNCTION DOC]
    # Constructs the layer. Inputs: padding (an integer applied to both ends, or a pair of left
    # and right amounts; default one) and base-layer keyword arguments. Normalizes the padding to
    # a pair, stores it, and sets an input spec requiring rank three. Raises a value error if the
    # padding cannot be normalized. Returns nothing.
    super(ZeroPadding1D, self).__init__(**kwargs)
    self.padding = conv_utils.normalize_tuple(padding, 2, 'padding')
    self.input_spec = InputSpec(ndim=3)

  def compute_output_shape(self, input_shape):
    # [FUNCTION DOC]
    # Computes the output shape for a rank-three input shape: batch and feature sizes are kept
    # and the step count grows by the sum of both padding amounts (unknown stays unknown). The
    # input is indexed directly without conversion, unlike the sibling layers. Returns a tensor
    # shape. No side effects.
    if input_shape[1] is not None:
      length = input_shape[1] + self.padding[0] + self.padding[1]
    else:
      length = None
    return tensor_shape.TensorShape([input_shape[0], length, input_shape[2]])

  def call(self, inputs):
    # [FUNCTION DOC]
    # Forward pass. Input: a rank-three tensor. Adds zeros before and after the time axis using
    # the backend temporal padding helper. Returns the padded tensor. No side effects.
    return backend.temporal_padding(inputs, padding=self.padding)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the base layer configuration mapping extended with the padding pair. No side
    # effects.
    config = {'padding': self.padding}
    base_config = super(ZeroPadding1D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class ZeroPadding2D(Layer):
  # [CLASS DOC]
  # Parameter-free layer that pads the two spatial axes of an image batch with zeros. Holds the
  # normalized padding as a pair of (before, after) pairs for rows and columns, plus the data
  # format, and requires rank-four input.
  """Zero-padding layer for 2D input (e.g. picture).

  This layer can add rows and columns of zeros
  at the top, bottom, left and right side of an image tensor.

  Examples:

  >>> input_shape = (1, 1, 2, 2)
  >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
  >>> print(x)
  [[[[0 1]
     [2 3]]]]
  >>> y = tf.keras.layers.ZeroPadding2D(padding=1)(x)
  >>> print(y)
  tf.Tensor(
    [[[[0 0]
       [0 0]
       [0 0]
       [0 0]]
      [[0 0]
       [0 1]
       [2 3]
       [0 0]]
      [[0 0]
       [0 0]
       [0 0]
       [0 0]]]], shape=(1, 3, 4, 2), dtype=int64)

  Args:
    padding: Int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
      - If int: the same symmetric padding
        is applied to height and width.
      - If tuple of 2 ints:
        interpreted as two different
        symmetric padding values for height and width:
        `(symmetric_height_pad, symmetric_width_pad)`.
      - If tuple of 2 tuples of 2 ints:
        interpreted as
        `((top_pad, bottom_pad), (left_pad, right_pad))`
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, height, width, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch_size, channels, height, width)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".

  Input shape:
    4D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, rows, cols, channels)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, channels, rows, cols)`

  Output shape:
    4D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, padded_rows, padded_cols, channels)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, channels, padded_rows, padded_cols)`
  """

  def __init__(self, padding=(1, 1), data_format=None, **kwargs):
    # [FUNCTION DOC]
    # Constructs the layer. Inputs: padding (an integer for symmetric padding everywhere, a pair
    # of integers for symmetric row and column padding, or a pair of pairs giving before and after
    # amounts per axis; default one each), data_format (null means the global default) and
    # base-layer keyword arguments. Normalizes padding to a pair of pairs and sets an input spec
    # requiring rank four. Raises a value error if padding is a sequence whose length is not two,
    # or is neither an integer nor a sequence. Returns nothing.
    super(ZeroPadding2D, self).__init__(**kwargs)
    self.data_format = conv_utils.normalize_data_format(data_format)
    if isinstance(padding, int):
      self.padding = ((padding, padding), (padding, padding))
    elif hasattr(padding, '__len__'):
      if len(padding) != 2:
        raise ValueError('`padding` should have two elements. '
                         'Found: ' + str(padding))
      height_padding = conv_utils.normalize_tuple(padding[0], 2,
                                                  '1st entry of padding')
      width_padding = conv_utils.normalize_tuple(padding[1], 2,
                                                 '2nd entry of padding')
      self.padding = (height_padding, width_padding)
    else:
      raise ValueError('`padding` should be either an int, '
                       'a tuple of 2 ints '
                       '(symmetric_height_pad, symmetric_width_pad), '
                       'or a tuple of 2 tuples of 2 ints '
                       '((top_pad, bottom_pad), (left_pad, right_pad)). '
                       'Found: ' + str(padding))
    self.input_spec = InputSpec(ndim=4)

  def compute_output_shape(self, input_shape):
    # [FUNCTION DOC]
    # Computes the output shape for a rank-four input shape: batch and channel sizes are kept,
    # and rows and columns grow by the sum of their before and after padding (unknown stays
    # unknown), with axes located by the data format. Returns a tensor shape. No side effects.
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_first':
      if input_shape[2] is not None:
        rows = input_shape[2] + self.padding[0][0] + self.padding[0][1]
      else:
        rows = None
      if input_shape[3] is not None:
        cols = input_shape[3] + self.padding[1][0] + self.padding[1][1]
      else:
        cols = None
      return tensor_shape.TensorShape(
          [input_shape[0], input_shape[1], rows, cols])
    elif self.data_format == 'channels_last':
      if input_shape[1] is not None:
        rows = input_shape[1] + self.padding[0][0] + self.padding[0][1]
      else:
        rows = None
      if input_shape[2] is not None:
        cols = input_shape[2] + self.padding[1][0] + self.padding[1][1]
      else:
        cols = None
      return tensor_shape.TensorShape(
          [input_shape[0], rows, cols, input_shape[3]])

  def call(self, inputs):
    # [FUNCTION DOC]
    # Forward pass. Input: a rank-four tensor. Adds zero rows and columns using the backend 2D
    # spatial padding helper with the stored padding and data format. Returns the padded tensor.
    # No side effects.
    return backend.spatial_2d_padding(
        inputs, padding=self.padding, data_format=self.data_format)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the base layer configuration mapping extended with padding and data format. No
    # side effects.
    config = {'padding': self.padding, 'data_format': self.data_format}
    base_config = super(ZeroPadding2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class ZeroPadding3D(Layer):
  # [CLASS DOC]
  # Parameter-free layer that pads the three spatial axes of a volumetric batch with zeros.
  # Holds the normalized padding as three (before, after) pairs, plus the data format, and
  # requires rank-five input.
  """Zero-padding layer for 3D data (spatial or spatio-temporal).

  Examples:

  >>> input_shape = (1, 1, 2, 2, 3)
  >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
  >>> y = tf.keras.layers.ZeroPadding3D(padding=2)(x)
  >>> print(y.shape)
  (1, 5, 6, 6, 3)

  Args:
    padding: Int, or tuple of 3 ints, or tuple of 3 tuples of 2 ints.
      - If int: the same symmetric padding
        is applied to height and width.
      - If tuple of 3 ints:
        interpreted as two different
        symmetric padding values for height and width:
        `(symmetric_dim1_pad, symmetric_dim2_pad, symmetric_dim3_pad)`.
      - If tuple of 3 tuples of 2 ints:
        interpreted as
        `((left_dim1_pad, right_dim1_pad), (left_dim2_pad,
          right_dim2_pad), (left_dim3_pad, right_dim3_pad))`
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
      while `channels_first` corresponds to inputs with shape
      `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".

  Input shape:
    5D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, first_axis_to_pad, second_axis_to_pad, third_axis_to_pad,
          depth)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, depth, first_axis_to_pad, second_axis_to_pad,
          third_axis_to_pad)`

  Output shape:
    5D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, first_padded_axis, second_padded_axis, third_axis_to_pad,
          depth)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, depth, first_padded_axis, second_padded_axis,
          third_axis_to_pad)`
  """

  def __init__(self, padding=(1, 1, 1), data_format=None, **kwargs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the error text for an invalid padding value names the last axis's right pad
    # as the second dimension's right pad (a typo in the message only; behavior is unaffected).
    # Constructs the layer. Inputs: padding (an integer for symmetric padding everywhere, a
    # triple of integers for symmetric per-axis padding, or a triple of pairs giving before and
    # after amounts per axis; default one each), data_format (null means the global default) and
    # base-layer keyword arguments. Normalizes padding to three pairs and sets an input spec
    # requiring rank five. Raises a value error if padding is a sequence whose length is not three,
    # or is neither an integer nor a sequence. Returns nothing.
    super(ZeroPadding3D, self).__init__(**kwargs)
    self.data_format = conv_utils.normalize_data_format(data_format)
    if isinstance(padding, int):
      self.padding = ((padding, padding), (padding, padding), (padding,
                                                               padding))
    elif hasattr(padding, '__len__'):
      if len(padding) != 3:
        raise ValueError('`padding` should have 3 elements. '
                         'Found: ' + str(padding))
      dim1_padding = conv_utils.normalize_tuple(padding[0], 2,
                                                '1st entry of padding')
      dim2_padding = conv_utils.normalize_tuple(padding[1], 2,
                                                '2nd entry of padding')
      dim3_padding = conv_utils.normalize_tuple(padding[2], 2,
                                                '3rd entry of padding')
      self.padding = (dim1_padding, dim2_padding, dim3_padding)
    else:
      raise ValueError(
          '`padding` should be either an int, '
          'a tuple of 3 ints '
          '(symmetric_dim1_pad, symmetric_dim2_pad, symmetric_dim3_pad), '
          'or a tuple of 3 tuples of 2 ints '
          '((left_dim1_pad, right_dim1_pad),'
          ' (left_dim2_pad, right_dim2_pad),'
          ' (left_dim3_pad, right_dim2_pad)). '
          'Found: ' + str(padding))
    self.input_spec = InputSpec(ndim=5)

  def compute_output_shape(self, input_shape):
    # [FUNCTION DOC]
    # Computes the output shape for a rank-five input shape: batch and channel sizes are kept,
    # and each of the three spatial sizes grows by the sum of its before and after padding
    # (unknown stays unknown), with axes located by the data format. Returns a tensor shape. No
    # side effects.
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_first':
      if input_shape[2] is not None:
        dim1 = input_shape[2] + self.padding[0][0] + self.padding[0][1]
      else:
        dim1 = None
      if input_shape[3] is not None:
        dim2 = input_shape[3] + self.padding[1][0] + self.padding[1][1]
      else:
        dim2 = None
      if input_shape[4] is not None:
        dim3 = input_shape[4] + self.padding[2][0] + self.padding[2][1]
      else:
        dim3 = None
      return tensor_shape.TensorShape(
          [input_shape[0], input_shape[1], dim1, dim2, dim3])
    elif self.data_format == 'channels_last':
      if input_shape[1] is not None:
        dim1 = input_shape[1] + self.padding[0][0] + self.padding[0][1]
      else:
        dim1 = None
      if input_shape[2] is not None:
        dim2 = input_shape[2] + self.padding[1][0] + self.padding[1][1]
      else:
        dim2 = None
      if input_shape[3] is not None:
        dim3 = input_shape[3] + self.padding[2][0] + self.padding[2][1]
      else:
        dim3 = None
      return tensor_shape.TensorShape(
          [input_shape[0], dim1, dim2, dim3, input_shape[4]])

  def call(self, inputs):
    # [FUNCTION DOC]
    # Forward pass. Input: a rank-five tensor. Adds zeros along the three spatial axes using the
    # backend 3D spatial padding helper with the stored padding and data format. Returns the
    # padded tensor. No side effects.
    return backend.spatial_3d_padding(
        inputs, padding=self.padding, data_format=self.data_format)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the base layer configuration mapping extended with padding and data format. No
    # side effects.
    config = {'padding': self.padding, 'data_format': self.data_format}
    base_config = super(ZeroPadding3D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class Cropping1D(Layer):
  # [CLASS DOC]
  # Parameter-free layer that removes entries from the start and end of the time axis of a
  # rank-three input. Holds the normalized pair of (start, end) crop amounts.
  """Cropping layer for 1D input (e.g. temporal sequence).

  It crops along the time dimension (axis 1).

  Examples:

  >>> input_shape = (2, 3, 2)
  >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
  >>> print(x)
  [[[ 0  1]
    [ 2  3]
    [ 4  5]]
   [[ 6  7]
    [ 8  9]
    [10 11]]]
  >>> y = tf.keras.layers.Cropping1D(cropping=1)(x)
  >>> print(y)
  tf.Tensor(
    [[[2 3]]
     [[8 9]]], shape=(2, 1, 2), dtype=int64)

  Args:
    cropping: Int or tuple of int (length 2)
      How many units should be trimmed off at the beginning and end of
      the cropping dimension (axis 1).
      If a single int is provided, the same value will be used for both.

  Input shape:
    3D tensor with shape `(batch_size, axis_to_crop, features)`

  Output shape:
    3D tensor with shape `(batch_size, cropped_axis, features)`
  """

  def __init__(self, cropping=(1, 1), **kwargs):
    # [FUNCTION DOC]
    # Constructs the layer. Inputs: cropping (an integer applied to both ends, or a pair of start
    # and end amounts; default one each) and base-layer keyword arguments. Normalizes the value
    # to a pair, stores it, and sets an input spec requiring rank three. Raises a value error if
    # the cropping cannot be normalized. Returns nothing.
    super(Cropping1D, self).__init__(**kwargs)
    self.cropping = conv_utils.normalize_tuple(cropping, 2, 'cropping')
    self.input_spec = InputSpec(ndim=3)

  def compute_output_shape(self, input_shape):
    # [FUNCTION DOC]
    # Computes the output shape for a rank-three input shape: batch and feature sizes are kept
    # and the step count shrinks by the sum of both crop amounts (unknown stays unknown). Returns a
    # tensor shape. No side effects.
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if input_shape[1] is not None:
      length = input_shape[1] - self.cropping[0] - self.cropping[1]
    else:
      length = None
    return tensor_shape.TensorShape([input_shape[0], length, input_shape[2]])

  def call(self, inputs):
    # [FUNCTION DOC]
    # Forward pass. Input: a rank-three tensor. Slices the time axis, dropping the leading crop
    # amount and the trailing crop amount; a trailing amount of zero is handled separately so
    # that nothing is dropped from the end. Returns the sliced tensor. No side effects.
    if self.cropping[1] == 0:
      return inputs[:, self.cropping[0]:, :]
    else:
      return inputs[:, self.cropping[0]:-self.cropping[1], :]

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the base layer configuration mapping extended with the crop pair. No side effects.
    config = {'cropping': self.cropping}
    base_config = super(Cropping1D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class Cropping2D(Layer):
  # [CLASS DOC]
  # Parameter-free layer that crops rows and columns from the borders of an image batch. Holds
  # the normalized crop amounts as a pair of (before, after) pairs plus the data format, and
  # requires rank-four input.
  """Cropping layer for 2D input (e.g. picture).

  It crops along spatial dimensions, i.e. height and width.

  Examples:

  >>> input_shape = (2, 28, 28, 3)
  >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
  >>> y = tf.keras.layers.Cropping2D(cropping=((2, 2), (4, 4)))(x)
  >>> print(y.shape)
  (2, 24, 20, 3)

  Args:
    cropping: Int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
      - If int: the same symmetric cropping
        is applied to height and width.
      - If tuple of 2 ints:
        interpreted as two different
        symmetric cropping values for height and width:
        `(symmetric_height_crop, symmetric_width_crop)`.
      - If tuple of 2 tuples of 2 ints:
        interpreted as
        `((top_crop, bottom_crop), (left_crop, right_crop))`
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, height, width, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch_size, channels, height, width)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".

  Input shape:
    4D tensor with shape:
    - If `data_format` is `"channels_last"`:
      `(batch_size, rows, cols, channels)`
    - If `data_format` is `"channels_first"`:
      `(batch_size, channels, rows, cols)`

  Output shape:
    4D tensor with shape:
    - If `data_format` is `"channels_last"`:
      `(batch_size, cropped_rows, cropped_cols, channels)`
    - If `data_format` is `"channels_first"`:
      `(batch_size, channels, cropped_rows, cropped_cols)`
  """

  def __init__(self, cropping=((0, 0), (0, 0)), data_format=None, **kwargs):
    # [FUNCTION DOC]
    # Constructs the layer. Inputs: cropping (an integer for symmetric cropping, a pair of
    # integers for symmetric row and column cropping, or a pair of pairs giving before and after
    # amounts per axis; default none), data_format (null means the global default) and base-layer
    # keyword arguments. Normalizes cropping to a pair of pairs and sets an input spec requiring
    # rank four. Raises a value error if cropping is a sequence whose length is not two, or is
    # neither an integer nor a sequence. Returns nothing.
    super(Cropping2D, self).__init__(**kwargs)
    self.data_format = conv_utils.normalize_data_format(data_format)
    if isinstance(cropping, int):
      self.cropping = ((cropping, cropping), (cropping, cropping))
    elif hasattr(cropping, '__len__'):
      if len(cropping) != 2:
        raise ValueError('`cropping` should have two elements. '
                         'Found: ' + str(cropping))
      height_cropping = conv_utils.normalize_tuple(cropping[0], 2,
                                                   '1st entry of cropping')
      width_cropping = conv_utils.normalize_tuple(cropping[1], 2,
                                                  '2nd entry of cropping')
      self.cropping = (height_cropping, width_cropping)
    else:
      raise ValueError('`cropping` should be either an int, '
                       'a tuple of 2 ints '
                       '(symmetric_height_crop, symmetric_width_crop), '
                       'or a tuple of 2 tuples of 2 ints '
                       '((top_crop, bottom_crop), (left_crop, right_crop)). '
                       'Found: ' + str(cropping))
    self.input_spec = InputSpec(ndim=4)

  def compute_output_shape(self, input_shape):
    # [FUNCTION DOC]
    # KNOWN ISSUE: unknown-size checks use truthiness rather than a null check, so a known spatial
    # size of zero is reported as unknown instead of being reduced by the crop amounts.
    # Computes the output shape for a rank-four input shape: batch and channel sizes are kept,
    # and rows and columns shrink by the sum of their before and after crop amounts (unknown
    # stays unknown), with axes located by the data format. Returns a tensor shape. No side
    # effects.
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    # pylint: disable=invalid-unary-operand-type
    if self.data_format == 'channels_first':
      return tensor_shape.TensorShape([
          input_shape[0], input_shape[1],
          input_shape[2] - self.cropping[0][0] - self.cropping[0][1]
          if input_shape[2] else None,
          input_shape[3] - self.cropping[1][0] - self.cropping[1][1]
          if input_shape[3] else None
      ])
    else:
      return tensor_shape.TensorShape([
          input_shape[0],
          input_shape[1] - self.cropping[0][0] - self.cropping[0][1]
          if input_shape[1] else None,
          input_shape[2] - self.cropping[1][0] - self.cropping[1][1]
          if input_shape[2] else None, input_shape[3]
      ])
    # pylint: enable=invalid-unary-operand-type

  def call(self, inputs):
    # [FUNCTION DOC]
    # Forward pass. Input: a rank-four tensor. Slices the two spatial axes (located by the data
    # format), dropping the leading and trailing crop amounts; trailing amounts of zero get
    # their own branches so that nothing is dropped from the end of that axis. Returns the sliced
    # tensor. No side effects.
    # pylint: disable=invalid-unary-operand-type
    if self.data_format == 'channels_first':
      if self.cropping[0][1] == self.cropping[1][1] == 0:
        return inputs[:, :, self.cropping[0][0]:, self.cropping[1][0]:]
      elif self.cropping[0][1] == 0:
        return inputs[:, :, self.cropping[0][0]:, self.cropping[1][0]:
                      -self.cropping[1][1]]
      elif self.cropping[1][1] == 0:
        return inputs[:, :, self.cropping[0][0]:-self.cropping[0][1],
                      self.cropping[1][0]:]
      return inputs[:, :, self.cropping[0][0]:-self.cropping[0][1],
                    self.cropping[1][0]:-self.cropping[1][1]]
    else:
      if self.cropping[0][1] == self.cropping[1][1] == 0:
        return inputs[:, self.cropping[0][0]:, self.cropping[1][0]:, :]
      elif self.cropping[0][1] == 0:
        return inputs[:, self.cropping[0][0]:, self.cropping[1][0]:
                      -self.cropping[1][1], :]
      elif self.cropping[1][1] == 0:
        return inputs[:, self.cropping[0][0]:-self.cropping[0][1],
                      self.cropping[1][0]:, :]
      return inputs[:, self.cropping[0][0]:-self.cropping[0][1], self.cropping[
          1][0]:-self.cropping[1][1], :]  # pylint: disable=invalid-unary-operand-type
    # pylint: enable=invalid-unary-operand-type

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the base layer configuration mapping extended with cropping and data format. No
    # side effects.
    config = {'cropping': self.cropping, 'data_format': self.data_format}
    base_config = super(Cropping2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class Cropping3D(Layer):
  # [CLASS DOC]
  # Parameter-free layer that crops the three spatial axes of a volumetric batch. Holds the
  # normalized crop amounts as three (before, after) pairs plus the data format, and requires
  # rank-five input.
  """Cropping layer for 3D data (e.g. spatial or spatio-temporal).

    Examples:

  >>> input_shape = (2, 28, 28, 10, 3)
  >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
  >>> y = tf.keras.layers.Cropping3D(cropping=(2, 4, 2))(x)
  >>> print(y.shape)
  (2, 24, 20, 6, 3)

  Args:
    cropping: Int, or tuple of 3 ints, or tuple of 3 tuples of 2 ints.
      - If int: the same symmetric cropping
        is applied to depth, height, and width.
      - If tuple of 3 ints: interpreted as two different
        symmetric cropping values for depth, height, and width:
        `(symmetric_dim1_crop, symmetric_dim2_crop, symmetric_dim3_crop)`.
      - If tuple of 3 tuples of 2 ints: interpreted as
        `((left_dim1_crop, right_dim1_crop), (left_dim2_crop,
          right_dim2_crop), (left_dim3_crop, right_dim3_crop))`
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
      while `channels_first` corresponds to inputs with shape
      `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".

  Input shape:
    5D tensor with shape:
    - If `data_format` is `"channels_last"`:
      `(batch_size, first_axis_to_crop, second_axis_to_crop, third_axis_to_crop,
        depth)`
    - If `data_format` is `"channels_first"`:
      `(batch_size, depth, first_axis_to_crop, second_axis_to_crop,
        third_axis_to_crop)`

  Output shape:
    5D tensor with shape:
    - If `data_format` is `"channels_last"`:
      `(batch_size, first_cropped_axis, second_cropped_axis, third_cropped_axis,
        depth)`
    - If `data_format` is `"channels_first"`:
      `(batch_size, depth, first_cropped_axis, second_cropped_axis,
        third_cropped_axis)`
  """

  def __init__(self,
               cropping=((1, 1), (1, 1), (1, 1)),
               data_format=None,
               **kwargs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the error text for an invalid cropping value names the last axis's right crop
    # as the second dimension's right crop (a typo in the message only).
    # Constructs the layer. Inputs: cropping (an integer for symmetric cropping, a triple of
    # integers for symmetric per-axis cropping, or a triple of pairs giving before and after
    # amounts per axis; default one each), data_format (null means the global default) and
    # base-layer keyword arguments. Normalizes cropping to three pairs and sets an input spec
    # requiring rank five. Raises a value error if cropping is a sequence whose length is not
    # three, or is neither an integer nor a sequence. Returns nothing.
    super(Cropping3D, self).__init__(**kwargs)
    self.data_format = conv_utils.normalize_data_format(data_format)
    if isinstance(cropping, int):
      self.cropping = ((cropping, cropping), (cropping, cropping), (cropping,
                                                                    cropping))
    elif hasattr(cropping, '__len__'):
      if len(cropping) != 3:
        raise ValueError('`cropping` should have 3 elements. '
                         'Found: ' + str(cropping))
      dim1_cropping = conv_utils.normalize_tuple(cropping[0], 2,
                                                 '1st entry of cropping')
      dim2_cropping = conv_utils.normalize_tuple(cropping[1], 2,
                                                 '2nd entry of cropping')
      dim3_cropping = conv_utils.normalize_tuple(cropping[2], 2,
                                                 '3rd entry of cropping')
      self.cropping = (dim1_cropping, dim2_cropping, dim3_cropping)
    else:
      raise ValueError(
          '`cropping` should be either an int, '
          'a tuple of 3 ints '
          '(symmetric_dim1_crop, symmetric_dim2_crop, symmetric_dim3_crop), '
          'or a tuple of 3 tuples of 2 ints '
          '((left_dim1_crop, right_dim1_crop),'
          ' (left_dim2_crop, right_dim2_crop),'
          ' (left_dim3_crop, right_dim2_crop)). '
          'Found: ' + str(cropping))
    self.input_spec = InputSpec(ndim=5)

  def compute_output_shape(self, input_shape):
    # [FUNCTION DOC]
    # Computes the output shape for a rank-five input shape: batch and channel sizes are kept,
    # and each spatial size shrinks by the sum of its before and after crop amounts (unknown stays
    # unknown), with axes located by the data format. Returns a tensor shape. No side effects.
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    # pylint: disable=invalid-unary-operand-type
    if self.data_format == 'channels_first':
      if input_shape[2] is not None:
        dim1 = input_shape[2] - self.cropping[0][0] - self.cropping[0][1]
      else:
        dim1 = None
      if input_shape[3] is not None:
        dim2 = input_shape[3] - self.cropping[1][0] - self.cropping[1][1]
      else:
        dim2 = None
      if input_shape[4] is not None:
        dim3 = input_shape[4] - self.cropping[2][0] - self.cropping[2][1]
      else:
        dim3 = None
      return tensor_shape.TensorShape(
          [input_shape[0], input_shape[1], dim1, dim2, dim3])
    elif self.data_format == 'channels_last':
      if input_shape[1] is not None:
        dim1 = input_shape[1] - self.cropping[0][0] - self.cropping[0][1]
      else:
        dim1 = None
      if input_shape[2] is not None:
        dim2 = input_shape[2] - self.cropping[1][0] - self.cropping[1][1]
      else:
        dim2 = None
      if input_shape[3] is not None:
        dim3 = input_shape[3] - self.cropping[2][0] - self.cropping[2][1]
      else:
        dim3 = None
      return tensor_shape.TensorShape(
          [input_shape[0], dim1, dim2, dim3, input_shape[4]])
    # pylint: enable=invalid-unary-operand-type

  def call(self, inputs):
    # [FUNCTION DOC]
    # Forward pass. Input: a rank-five tensor. Slices the three spatial axes (located by the data
    # format), dropping the leading and trailing crop amounts; every combination of zero trailing
    # amounts has its own branch so that nothing is dropped from the end of those axes. Returns
    # the sliced tensor. No side effects.
    # pylint: disable=invalid-unary-operand-type
    if self.data_format == 'channels_first':
      if self.cropping[0][1] == self.cropping[1][1] == self.cropping[2][1] == 0:
        return inputs[:, :, self.cropping[0][0]:, self.cropping[1][0]:,
                      self.cropping[2][0]:]
      elif self.cropping[0][1] == self.cropping[1][1] == 0:
        return inputs[:, :, self.cropping[0][0]:, self.cropping[1][0]:,
                      self.cropping[2][0]:-self.cropping[2][1]]
      elif self.cropping[1][1] == self.cropping[2][1] == 0:
        return inputs[:, :, self.cropping[0][0]:-self.cropping[0][1],
                      self.cropping[1][0]:, self.cropping[2][0]:]
      elif self.cropping[0][1] == self.cropping[2][1] == 0:
        return inputs[:, :, self.cropping[0][0]:, self.cropping[1][0]:
                      -self.cropping[1][1], self.cropping[2][0]:]
      elif self.cropping[0][1] == 0:
        return inputs[:, :, self.cropping[0][0]:, self.cropping[1][
            0]:-self.cropping[1][1], self.cropping[2][0]:-self.cropping[2][1]]
      elif self.cropping[1][1] == 0:
        return inputs[:, :, self.cropping[0][0]:-self.cropping[0][1], self.
                      cropping[1][0]:, self.cropping[2][0]:-self.cropping[2][1]]
      elif self.cropping[2][1] == 0:
        return inputs[:, :, self.cropping[0][0]:-self.cropping[0][1], self.
                      cropping[1][0]:-self.cropping[1][1], self.cropping[2][0]:]
      return inputs[:, :, self.cropping[0][0]:-self.cropping[0][1],
                    self.cropping[1][0]:-self.cropping[1][1], self.cropping[2][
                        0]:-self.cropping[2][1]]
    else:
      if self.cropping[0][1] == self.cropping[1][1] == self.cropping[2][1] == 0:
        return inputs[:, self.cropping[0][0]:, self.cropping[1][0]:,
                      self.cropping[2][0]:, :]
      elif self.cropping[0][1] == self.cropping[1][1] == 0:
        return inputs[:, self.cropping[0][0]:, self.cropping[1][0]:,
                      self.cropping[2][0]:-self.cropping[2][1], :]
      elif self.cropping[1][1] == self.cropping[2][1] == 0:
        return inputs[:, self.cropping[0][0]:-self.cropping[0][1],
                      self.cropping[1][0]:, self.cropping[2][0]:, :]
      elif self.cropping[0][1] == self.cropping[2][1] == 0:
        return inputs[:, self.cropping[0][0]:, self.cropping[1][0]:
                      -self.cropping[1][1], self.cropping[2][0]:, :]
      elif self.cropping[0][1] == 0:
        return inputs[:, self.cropping[0][0]:, self.cropping[1][
            0]:-self.cropping[1][1], self.cropping[2][0]:
                      -self.cropping[2][1], :]
      elif self.cropping[1][1] == 0:
        return inputs[:, self.cropping[0][
            0]:-self.cropping[0][1], self.cropping[1][0]:, self.cropping[2][0]:
                      -self.cropping[2][1], :]
      elif self.cropping[2][1] == 0:
        return inputs[:, self.cropping[0][0]:-self.cropping[0][1],
                      self.cropping[1][0]:-self.cropping[1][1], self.cropping[
                          2][0]:, :]
      return inputs[:, self.cropping[0][0]:-self.cropping[0][1], self.cropping[
          1][0]:-self.cropping[1][1], self.cropping[2][0]:  # pylint: disable=invalid-unary-operand-type
                    -self.cropping[2][1], :]  # pylint: disable=invalid-unary-operand-type
    # pylint: enable=invalid-unary-operand-type

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the base layer configuration mapping extended with cropping and data format. No
    # side effects.
    config = {'cropping': self.cropping, 'data_format': self.data_format}
    base_config = super(Cropping3D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


# Aliases

Convolution1D = Conv1D
Convolution2D = Conv2D
Convolution3D = Conv3D
SeparableConvolution1D = SeparableConv1D
SeparableConvolution2D = SeparableConv2D
Convolution2DTranspose = Conv2DTranspose
Convolution3DTranspose = Conv3DTranspose
Deconvolution2D = Deconv2D = Conv2DTranspose
Deconvolution3D = Deconv3D = Conv3DTranspose
