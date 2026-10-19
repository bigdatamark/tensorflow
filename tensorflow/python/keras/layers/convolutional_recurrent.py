# [MODULE DOC]
# Convolutional-recurrent Keras layers (legacy tensorflow.python.keras implementation).
# Defines three public classes: a generic base layer for recurrent networks whose cell applies
# two-dimensional convolutions (ConvRNN2D), a convolutional LSTM cell (ConvLSTM2DCell) and the
# layer that wraps that cell (ConvLSTM2D). Inputs are five-dimensional tensors (batch, time,
# and a two-dimensional image with channels); states are four-dimensional image-shaped tensors.
# These classes build on the generic recurrent layer machinery (RNN, dropout mixin) and on the
# backend helpers for convolution and stepping over time. Importing the module has no side
# effects beyond defining the classes. It is a regular layer module, not a helper module.
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
# pylint: disable=protected-access
# pylint: disable=g-classes-have-attributes
"""Convolutional-recurrent layers."""

import numpy as np

from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin
from tensorflow.python.keras.layers.recurrent import RNN
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops


class ConvRNN2D(RNN):
  # [CLASS DOC]
  # Base class for recurrent layers whose cell works on image-shaped data (convolutional cells).
  # Subclass of the generic recurrent layer; it requires five-dimensional inputs (samples,
  # time, rows, cols, channels or the channels-first variant) and keeps image-shaped states.
  # Holds the cell, an input specification fixed at five dimensions, a list of state variables
  # (None until stateful states are created) and the number of external constants (None until
  # set by input processing). It overrides shape inference, building, initial state creation,
  # the call over time and state reset. Unrolling and stacked cells are not supported.
  """Base class for convolutional-recurrent layers.

  Args:
    cell: A RNN cell instance. A RNN cell is a class that has:
      - a `call(input_at_t, states_at_t)` method, returning
        `(output_at_t, states_at_t_plus_1)`. The call method of the
        cell can also take the optional argument `constants`, see
        section "Note on passing external constants" below.
      - a `state_size` attribute. This can be a single integer
        (single state) in which case it is
        the number of channels of the recurrent state
        (which should be the same as the number of channels of the cell
        output). This can also be a list/tuple of integers
        (one size per state). In this case, the first entry
        (`state_size[0]`) should be the same as
        the size of the cell output.
    return_sequences: Boolean. Whether to return the last output.
      in the output sequence, or the full sequence.
    return_state: Boolean. Whether to return the last state
      in addition to the output.
    go_backwards: Boolean (default False).
      If True, process the input sequence backwards and return the
      reversed sequence.
    stateful: Boolean (default False). If True, the last state
      for each sample at index i in a batch will be used as initial
      state for the sample of index i in the following batch.
    input_shape: Use this argument to specify the shape of the
      input when this layer is the first one in a model.

  Call arguments:
    inputs: A 5D tensor.
    mask: Binary tensor of shape `(samples, timesteps)` indicating whether
      a given timestep should be masked.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. This argument is passed to the cell
      when calling it. This is for use with cells that use dropout.
    initial_state: List of initial state tensors to be passed to the first
      call of the cell.
    constants: List of constant tensors to be passed to the cell at each
      timestep.

  Input shape:
    5D tensor with shape:
    `(samples, timesteps, channels, rows, cols)`
    if data_format='channels_first' or 5D tensor with shape:
    `(samples, timesteps, rows, cols, channels)`
    if data_format='channels_last'.

  Output shape:
    - If `return_state`: a list of tensors. The first tensor is
      the output. The remaining tensors are the last states,
      each 4D tensor with shape:
      `(samples, filters, new_rows, new_cols)`
      if data_format='channels_first'
      or 4D tensor with shape:
      `(samples, new_rows, new_cols, filters)`
      if data_format='channels_last'.
      `rows` and `cols` values might have changed due to padding.
    - If `return_sequences`: 5D tensor with shape:
      `(samples, timesteps, filters, new_rows, new_cols)`
      if data_format='channels_first'
      or 5D tensor with shape:
      `(samples, timesteps, new_rows, new_cols, filters)`
      if data_format='channels_last'.
    - Else, 4D tensor with shape:
      `(samples, filters, new_rows, new_cols)`
      if data_format='channels_first'
      or 4D tensor with shape:
      `(samples, new_rows, new_cols, filters)`
      if data_format='channels_last'.

  Masking:
    This layer supports masking for input data with a variable number
    of timesteps.

  Note on using statefulness in RNNs:
    You can set RNN layers to be 'stateful', which means that the states
    computed for the samples in one batch will be reused as initial states
    for the samples in the next batch. This assumes a one-to-one mapping
    between samples in different successive batches.
    To enable statefulness:
      - Specify `stateful=True` in the layer constructor.
      - Specify a fixed batch size for your model, by passing
         - If sequential model:
            `batch_input_shape=(...)` to the first layer in your model.
         - If functional model with 1 or more Input layers:
            `batch_shape=(...)` to all the first layers in your model.
            This is the expected shape of your inputs
            *including the batch size*.
            It should be a tuple of integers,
            e.g. `(32, 10, 100, 100, 32)`.
            Note that the number of rows and columns should be specified
            too.
      - Specify `shuffle=False` when calling fit().
    To reset the states of your model, call `.reset_states()` on either
    a specific layer, or on your entire model.

  Note on specifying the initial state of RNNs:
    You can specify the initial state of RNN layers symbolically by
    calling them with the keyword argument `initial_state`. The value of
    `initial_state` should be a tensor or list of tensors representing
    the initial state of the RNN layer.
    You can specify the initial state of RNN layers numerically by
    calling `reset_states` with the keyword argument `states`. The value of
    `states` should be a numpy array or list of numpy arrays representing
    the initial state of the RNN layer.

  Note on passing external constants to RNNs:
    You can pass "external" constants to the cell using the `constants`
    keyword argument of `RNN.__call__` (as well as `RNN.call`) method. This
    requires that the `cell.call` method accepts the same keyword argument
    `constants`. Such constants can be used to condition the cell
    transformation on additional static inputs (not changing over time),
    a.k.a. an attention mechanism.
  """

  def __init__(self,
               cell,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               unroll=False,
               **kwargs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the error message for stacked cells concatenates two string pieces without a
    # space, so it reads "possible at the momentto stack"; also the cell list check only rejects
    # lists/tuples of cells.
    # Constructor. Inputs: cell (a convolutional RNN cell object), return_sequences (bool,
    # default false, return full sequence instead of last output), return_state (bool, default
    # false, also return final states), go_backwards (bool, process time in reverse),
    # stateful (bool, reuse final states as next batch initial states), unroll (must stay
    # false) and extra keyword arguments forwarded to the base recurrent layer.
    # Raises a type error if unroll is true or if the cell is a list/tuple. Otherwise calls the
    # base constructor, sets the input spec to five dimensions, clears the states and the
    # constants count. Returns nothing.
    if unroll:
      raise TypeError('Unrolling isn\'t possible with '
                      'convolutional RNNs.')
    if isinstance(cell, (list, tuple)):
      # The StackedConvRNN2DCells isn't implemented yet.
      raise TypeError('It is not possible at the moment to'
                      'stack convolutional cells.')
    super(ConvRNN2D, self).__init__(cell,
                                    return_sequences,
                                    return_state,
                                    go_backwards,
                                    stateful,
                                    unroll,
                                    **kwargs)
    self.input_spec = [InputSpec(ndim=5)]
    self.states = None
    self._num_constants = None

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    # [FUNCTION DOC]
    # KNOWN ISSUE: when return_state is true it always appends exactly two state shapes, which
    # is only right for cells with two states (as the LSTM cell); other cells would be wrong.
    # Computes the output shape from the input shape. input_shape is a five-dimensional shape,
    # or a list of shapes (then only the first is used). Spatial sizes are reduced using the
    # cell's kernel size, padding, strides and dilation; channels become the cell's filters.
    # If return_sequences is false the time axis is dropped. If return_state is true the result
    # is a list: the output shape followed by two state shapes (batch, spatial, filters).
    # Depends on the cell's data format; any other format leaves variables unset and fails.
    # No side effects.
    if isinstance(input_shape, list):
      input_shape = input_shape[0]

    cell = self.cell
    if cell.data_format == 'channels_first':
      rows = input_shape[3]
      cols = input_shape[4]
    elif cell.data_format == 'channels_last':
      rows = input_shape[2]
      cols = input_shape[3]
    rows = conv_utils.conv_output_length(rows,
                                         cell.kernel_size[0],
                                         padding=cell.padding,
                                         stride=cell.strides[0],
                                         dilation=cell.dilation_rate[0])
    cols = conv_utils.conv_output_length(cols,
                                         cell.kernel_size[1],
                                         padding=cell.padding,
                                         stride=cell.strides[1],
                                         dilation=cell.dilation_rate[1])

    if cell.data_format == 'channels_first':
      output_shape = input_shape[:2] + (cell.filters, rows, cols)
    elif cell.data_format == 'channels_last':
      output_shape = input_shape[:2] + (rows, cols, cell.filters)

    if not self.return_sequences:
      output_shape = output_shape[:1] + output_shape[2:]

    if self.return_state:
      output_shape = [output_shape]
      if cell.data_format == 'channels_first':
        output_shape += [(input_shape[0], cell.filters, rows, cols)
                         for _ in range(2)]
      elif cell.data_format == 'channels_last':
        output_shape += [(input_shape[0], rows, cols, cell.filters)
                         for _ in range(2)]
    return output_shape

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    # [FUNCTION DOC]
    # Creates the layer's variables and specs. input_shape is the five-dimensional input
    # shape, or a list holding it followed by shapes of initial states and constants (the
    # last entries being constants when a constants count is set).
    # Sets the input spec (batch size kept only when stateful, time axis free), builds the cell
    # first if the cell is itself a layer (passing constants shapes if present), then either
    # validates a pre-existing state spec against the cell's state sizes or creates one
    # (channel size per state, other dims free). Raises a value error when a provided initial
    # state's channel counts do not match the cell state sizes. If stateful, resets states.
    # Marks the layer as built. Returns nothing.
    # Note input_shape will be list of shapes of initial states and
    # constants if these are passed in __call__.
    if self._num_constants is not None:
      constants_shape = input_shape[-self._num_constants:]  # pylint: disable=E1130
    else:
      constants_shape = None

    if isinstance(input_shape, list):
      input_shape = input_shape[0]

    batch_size = input_shape[0] if self.stateful else None
    self.input_spec[0] = InputSpec(shape=(batch_size, None) + input_shape[2:5])

    # allow cell (if layer) to build before we set or validate state_spec
    if isinstance(self.cell, Layer):
      step_input_shape = (input_shape[0],) + input_shape[2:]
      if constants_shape is not None:
        self.cell.build([step_input_shape] + constants_shape)
      else:
        self.cell.build(step_input_shape)

    # set or validate state_spec
    if hasattr(self.cell.state_size, '__len__'):
      state_size = list(self.cell.state_size)
    else:
      state_size = [self.cell.state_size]

    if self.state_spec is not None:
      # initial_state was passed in call, check compatibility
      if self.cell.data_format == 'channels_first':
        ch_dim = 1
      elif self.cell.data_format == 'channels_last':
        ch_dim = 3
      if [spec.shape[ch_dim] for spec in self.state_spec] != state_size:
        raise ValueError(
            'An initial_state was passed that is not compatible with '
            '`cell.state_size`. Received `state_spec`={}; '
            'However `cell.state_size` is '
            '{}'.format([spec.shape for spec in self.state_spec],
                        self.cell.state_size))
    else:
      if self.cell.data_format == 'channels_first':
        self.state_spec = [InputSpec(shape=(None, dim, None, None))
                           for dim in state_size]
      elif self.cell.data_format == 'channels_last':
        self.state_spec = [InputSpec(shape=(None, None, None, dim))
                           for dim in state_size]
    if self.stateful:
      self.reset_states()
    self.built = True

  def get_initial_state(self, inputs):
    # [FUNCTION DOC]
    # Builds the all-zero initial states. inputs is the five-dimensional input tensor. A zero
    # tensor shaped like the inputs is summed over time, then passed through the cell's input
    # convolution with an all-zero kernel so the result has the correct spatial size and filter
    # count. Returns a list with one such zero state per entry of the cell's state size (one
    # entry if the state size is a single number). No side effects.
    # (samples, timesteps, rows, cols, filters)
    initial_state = backend.zeros_like(inputs)
    # (samples, rows, cols, filters)
    initial_state = backend.sum(initial_state, axis=1)
    shape = list(self.cell.kernel_shape)
    shape[-1] = self.cell.filters
    initial_state = self.cell.input_conv(initial_state,
                                         array_ops.zeros(tuple(shape),
                                                         initial_state.dtype),
                                         padding=self.cell.padding)

    if hasattr(self.cell.state_size, '__len__'):
      return [initial_state for _ in self.cell.state_size]
    else:
      return [initial_state]

  def call(self,
           inputs,
           mask=None,
           training=None,
           initial_state=None,
           constants=None):
    # [FUNCTION DOC]
    # Runs the cell over every timestep of the input. Inputs: inputs (5D tensor, or a structure
    # that is split into inputs, initial state and constants), mask (optional tensor or list;
    # a list uses its first element), training (optional bool passed to the cell only if the
    # cell's call accepts it), initial_state (optional list of tensors) and constants
    # (optional list of tensors given to the cell at each step).
    # Raises a value error if constants are given but the cell's call does not accept them.
    # Uses the backend rnn loop, honoring go_backwards and the mask. When stateful, adds update
    # operations assigning the new states into the stored state variables (side effect).
    # Returns the last output or full sequence of outputs depending on return_sequences; if
    # return_state is set, returns a list of the output followed by the final states.
    # note that the .build() method of subclasses MUST define
    # self.input_spec and self.state_spec with complete input shapes.
    inputs, initial_state, constants = self._process_inputs(
        inputs, initial_state, constants)

    if isinstance(mask, list):
      mask = mask[0]
    timesteps = backend.int_shape(inputs)[1]

    kwargs = {}
    if generic_utils.has_arg(self.cell.call, 'training'):
      kwargs['training'] = training

    if constants:
      if not generic_utils.has_arg(self.cell.call, 'constants'):
        raise ValueError('RNN cell does not support constants')

      def step(inputs, states):
        # [FUNCTION DOC]
        # HELPER: per-timestep function handed to the backend rnn loop inside call.
        # Variant used when constants exist: splits the trailing constants off the state list,
        # then calls the cell with inputs, the remaining states, the constants and the
        # captured training keyword. Returns the cell's output and new states.
        constants = states[-self._num_constants:]  # pylint: disable=invalid-unary-operand-type
        states = states[:-self._num_constants]  # pylint: disable=invalid-unary-operand-type
        return self.cell.call(inputs, states, constants=constants, **kwargs)
    else:
      def step(inputs, states):
        # [FUNCTION DOC]
        # HELPER: per-timestep function handed to the backend rnn loop inside call.
        # Variant used without constants: calls the cell with the step inputs, the states and
        # the captured keyword arguments (training if supported). Returns the cell's output
        # and list of new states.
        return self.cell.call(inputs, states, **kwargs)

    last_output, outputs, states = backend.rnn(step,
                                               inputs,
                                               initial_state,
                                               constants=constants,
                                               go_backwards=self.go_backwards,
                                               mask=mask,
                                               input_length=timesteps)
    if self.stateful:
      updates = [
          backend.update(self_state, state)
          for self_state, state in zip(self.states, states)
      ]
      self.add_update(updates)

    if self.return_sequences:
      output = outputs
    else:
      output = last_output

    if self.return_state:
      if not isinstance(states, (list, tuple)):
        states = [states]
      else:
        states = list(states)
      return [output] + states
    else:
      return output

  def reset_states(self, states=None):
    # [FUNCTION DOC]
    # Resets (or sets) the stored state variables of a stateful layer. states is optional: a
    # tensor/array or list of them giving new values; if omitted, states are zeroed.
    # Raises an attribute error if the layer is not stateful; raises a value error if the batch
    # size or spatial dims are not fully known, if the number of given states differs from the
    # layer's, or if a given state's shape differs from the expected one.
    # Side effects: if no state variables exist yet, creates zero-filled ones; otherwise
    # assigns zeros or the provided values into the existing variables via the backend.
    # Returns nothing.
    if not self.stateful:
      raise AttributeError('Layer must be stateful.')
    input_shape = self.input_spec[0].shape
    state_shape = self.compute_output_shape(input_shape)
    if self.return_state:
      state_shape = state_shape[0]
    if self.return_sequences:
      state_shape = state_shape[:1].concatenate(state_shape[2:])
    if None in state_shape:
      raise ValueError('If a RNN is stateful, it needs to know '
                       'its batch size. Specify the batch size '
                       'of your input tensors: \n'
                       '- If using a Sequential model, '
                       'specify the batch size by passing '
                       'a `batch_input_shape` '
                       'argument to your first layer.\n'
                       '- If using the functional API, specify '
                       'the time dimension by passing a '
                       '`batch_shape` argument to your Input layer.\n'
                       'The same thing goes for the number of rows and '
                       'columns.')

    # helper function
    def get_tuple_shape(nb_channels):
      # [FUNCTION DOC]
      # HELPER: shape builder for reset_states.
      # Returns the state shape with its channel dimension replaced by the given channel count
      # (axis 1 for channels-first, axis 3 for channels-last). Raises a key error for an
      # unknown data format.
      result = list(state_shape)
      if self.cell.data_format == 'channels_first':
        result[1] = nb_channels
      elif self.cell.data_format == 'channels_last':
        result[3] = nb_channels
      else:
        raise KeyError
      return tuple(result)

    # initialize state if None
    if self.states[0] is None:
      if hasattr(self.cell.state_size, '__len__'):
        self.states = [backend.zeros(get_tuple_shape(dim))
                       for dim in self.cell.state_size]
      else:
        self.states = [backend.zeros(get_tuple_shape(self.cell.state_size))]
    elif states is None:
      if hasattr(self.cell.state_size, '__len__'):
        for state, dim in zip(self.states, self.cell.state_size):
          backend.set_value(state, np.zeros(get_tuple_shape(dim)))
      else:
        backend.set_value(self.states[0],
                          np.zeros(get_tuple_shape(self.cell.state_size)))
    else:
      if not isinstance(states, (list, tuple)):
        states = [states]
      if len(states) != len(self.states):
        raise ValueError('Layer ' + self.name + ' expects ' +
                         str(len(self.states)) + ' states, ' +
                         'but it received ' + str(len(states)) +
                         ' state values. Input received: ' + str(states))
      for index, (value, state) in enumerate(zip(states, self.states)):
        if hasattr(self.cell.state_size, '__len__'):
          dim = self.cell.state_size[index]
        else:
          dim = self.cell.state_size
        if value.shape != get_tuple_shape(dim):
          raise ValueError('State ' + str(index) +
                           ' is incompatible with layer ' +
                           self.name + ': expected shape=' +
                           str(get_tuple_shape(dim)) +
                           ', found shape=' + str(value.shape))
        # TODO(anjalisridhar): consider batch calls to `set_value`.
        backend.set_value(state, value)


class ConvLSTM2DCell(DropoutRNNCellMixin, Layer):
  # [CLASS DOC]
  # Single-timestep cell of a convolutional LSTM, used by ConvLSTM2D. Combines the dropout
  # mask mixin with the base layer. Holds normalized convolution settings (kernel size,
  # strides, padding, data format, dilation), activations, initializers, regularizers,
  # constraints, clipped dropout rates, and a state size of two entries (hidden and carry,
  # each filters channels). After build it also holds the input kernel, recurrent kernel,
  # optional bias and the input kernel shape. Created by ConvLSTM2D, not usually directly.
  """Cell class for the ConvLSTM2D layer.

  Args:
    filters: Integer, the dimensionality of the output space
      (i.e. the number of output filters in the convolution).
    kernel_size: An integer or tuple/list of n integers, specifying the
      dimensions of the convolution window.
    strides: An integer or tuple/list of n integers,
      specifying the strides of the convolution.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: One of `"valid"` or `"same"` (case-insensitive).
      `"valid"` means no padding. `"same"` results in padding evenly to 
      the left/right or up/down of the input such that output has the same 
      height/width dimension as the input.
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".
    dilation_rate: An integer or tuple/list of n integers, specifying
      the dilation rate to use for dilated convolution.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any `strides` value != 1.
    activation: Activation function to use.
      If you don't specify anything, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    recurrent_activation: Activation function to use
      for the recurrent step.
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix,
      used for the linear transformation of the inputs.
    recurrent_initializer: Initializer for the `recurrent_kernel`
      weights matrix,
      used for the linear transformation of the recurrent state.
    bias_initializer: Initializer for the bias vector.
    unit_forget_bias: Boolean.
      If True, add 1 to the bias of the forget gate at initialization.
      Use in combination with `bias_initializer="zeros"`.
      This is recommended in [Jozefowicz et al., 2015](
        http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix.
    recurrent_regularizer: Regularizer function applied to
      the `recurrent_kernel` weights matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    kernel_constraint: Constraint function applied to
      the `kernel` weights matrix.
    recurrent_constraint: Constraint function applied to
      the `recurrent_kernel` weights matrix.
    bias_constraint: Constraint function applied to the bias vector.
    dropout: Float between 0 and 1.
      Fraction of the units to drop for
      the linear transformation of the inputs.
    recurrent_dropout: Float between 0 and 1.
      Fraction of the units to drop for
      the linear transformation of the recurrent state.

  Call arguments:
    inputs: A 4D tensor.
    states:  List of state tensors corresponding to the previous timestep.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. Only relevant when `dropout` or
      `recurrent_dropout` is used.
  """

  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1),
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               **kwargs):
    # [FUNCTION DOC]
    # Constructor. Inputs: filters (int, output channels), kernel_size, strides, dilation_rate
    # (int or pair, normalized to pairs), padding (valid or same), data_format (channels first
    # or last, default from global config), activation and recurrent_activation (names or
    # callables), use_bias, unit_forget_bias, initializers, regularizers and constraints (names,
    # objects or none), dropout and recurrent_dropout (floats clipped into the zero-to-one
    # range) and extra keyword arguments passed to the base layer.
    # Raises a value error from the normalization helpers on invalid sizes, padding or format.
    # Stores everything as attributes and sets the state size to two entries of filters.
    # Returns nothing.
    super(ConvLSTM2DCell, self).__init__(**kwargs)
    self.filters = filters
    self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
    self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
    self.padding = conv_utils.normalize_padding(padding)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2,
                                                    'dilation_rate')
    self.activation = activations.get(activation)
    self.recurrent_activation = activations.get(recurrent_activation)
    self.use_bias = use_bias

    self.kernel_initializer = initializers.get(kernel_initializer)
    self.recurrent_initializer = initializers.get(recurrent_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.unit_forget_bias = unit_forget_bias

    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)

    self.kernel_constraint = constraints.get(kernel_constraint)
    self.recurrent_constraint = constraints.get(recurrent_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

    self.dropout = min(1., max(0., dropout))
    self.recurrent_dropout = min(1., max(0., recurrent_dropout))
    self.state_size = (self.filters, self.filters)

  def build(self, input_shape):
    # [FUNCTION DOC]
    # Creates the cell's weights. input_shape is the per-step four-dimensional input shape; its
    # channel axis (first-after-batch or last, per data format) must be known, otherwise a
    # value error is raised. Creates the input kernel (four gates stacked on the output axis),
    # the recurrent kernel and, if use_bias, a bias of four times filters; when unit_forget_bias
    # is set the forget-gate slice of the bias is initialized to ones and the other slices use
    # the configured bias initializer. Records the kernel shape, sets bias to none when unused
    # and marks the cell built. Returns nothing.

    if self.data_format == 'channels_first':
      channel_axis = 1
    else:
      channel_axis = -1
    if input_shape[channel_axis] is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = input_shape[channel_axis]
    kernel_shape = self.kernel_size + (input_dim, self.filters * 4)
    self.kernel_shape = kernel_shape
    recurrent_kernel_shape = self.kernel_size + (self.filters, self.filters * 4)

    self.kernel = self.add_weight(shape=kernel_shape,
                                  initializer=self.kernel_initializer,
                                  name='kernel',
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
    self.recurrent_kernel = self.add_weight(
        shape=recurrent_kernel_shape,
        initializer=self.recurrent_initializer,
        name='recurrent_kernel',
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint)

    if self.use_bias:
      if self.unit_forget_bias:

        def bias_initializer(_, *args, **kwargs):
          # [FUNCTION DOC]
          # HELPER: custom bias initializer used by build when unit_forget_bias is enabled.
          # Ignores the requested shape argument and returns a concatenation of: configured
          # initial values for the input gate, ones for the forget gate, and configured values
          # for the cell and output gates (twice the filters), forming a vector of four times
          # filters. Extra positional and keyword arguments are forwarded to the initializers.
          return backend.concatenate([
              self.bias_initializer((self.filters,), *args, **kwargs),
              initializers.get('ones')((self.filters,), *args, **kwargs),
              self.bias_initializer((self.filters * 2,), *args, **kwargs),
          ])
      else:
        bias_initializer = self.bias_initializer
      self.bias = self.add_weight(
          shape=(self.filters * 4,),
          name='bias',
          initializer=bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint)
    else:
      self.bias = None
    self.built = True

  def call(self, inputs, states, training=None):
    # [FUNCTION DOC]
    # Computes one LSTM step with convolutions. inputs: 4D tensor for the current timestep.
    # states: list of two tensors, the previous hidden state and previous carry state.
    # training: optional bool controlling dropout masks. Applies per-gate dropout to the inputs
    # and to the hidden state when the corresponding rate is strictly between zero and one,
    # splits kernels and bias into the input, forget, cell and output gate parts, convolves
    # inputs with the input kernels (plus bias) and the hidden state with the recurrent kernels
    # (same padding, stride one), then computes the gates with the recurrent activation and
    # the new carry and hidden values. Returns the new hidden state and a list of hidden and
    # carry. Side effect: may create and cache dropout masks via the mixin.
    h_tm1 = states[0]  # previous memory state
    c_tm1 = states[1]  # previous carry state

    # dropout matrices for input units
    dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
    # dropout matrices for recurrent units
    rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
        h_tm1, training, count=4)

    if 0 < self.dropout < 1.:
      inputs_i = inputs * dp_mask[0]
      inputs_f = inputs * dp_mask[1]
      inputs_c = inputs * dp_mask[2]
      inputs_o = inputs * dp_mask[3]
    else:
      inputs_i = inputs
      inputs_f = inputs
      inputs_c = inputs
      inputs_o = inputs

    if 0 < self.recurrent_dropout < 1.:
      h_tm1_i = h_tm1 * rec_dp_mask[0]
      h_tm1_f = h_tm1 * rec_dp_mask[1]
      h_tm1_c = h_tm1 * rec_dp_mask[2]
      h_tm1_o = h_tm1 * rec_dp_mask[3]
    else:
      h_tm1_i = h_tm1
      h_tm1_f = h_tm1
      h_tm1_c = h_tm1
      h_tm1_o = h_tm1

    (kernel_i, kernel_f,
     kernel_c, kernel_o) = array_ops.split(self.kernel, 4, axis=3)
    (recurrent_kernel_i,
     recurrent_kernel_f,
     recurrent_kernel_c,
     recurrent_kernel_o) = array_ops.split(self.recurrent_kernel, 4, axis=3)

    if self.use_bias:
      bias_i, bias_f, bias_c, bias_o = array_ops.split(self.bias, 4)
    else:
      bias_i, bias_f, bias_c, bias_o = None, None, None, None

    x_i = self.input_conv(inputs_i, kernel_i, bias_i, padding=self.padding)
    x_f = self.input_conv(inputs_f, kernel_f, bias_f, padding=self.padding)
    x_c = self.input_conv(inputs_c, kernel_c, bias_c, padding=self.padding)
    x_o = self.input_conv(inputs_o, kernel_o, bias_o, padding=self.padding)
    h_i = self.recurrent_conv(h_tm1_i, recurrent_kernel_i)
    h_f = self.recurrent_conv(h_tm1_f, recurrent_kernel_f)
    h_c = self.recurrent_conv(h_tm1_c, recurrent_kernel_c)
    h_o = self.recurrent_conv(h_tm1_o, recurrent_kernel_o)

    i = self.recurrent_activation(x_i + h_i)
    f = self.recurrent_activation(x_f + h_f)
    c = f * c_tm1 + i * self.activation(x_c + h_c)
    o = self.recurrent_activation(x_o + h_o)
    h = o * self.activation(c)
    return h, [h, c]

  def input_conv(self, x, w, b=None, padding='valid'):
    # [FUNCTION DOC]
    # HELPER: convolution of the cell inputs, used by call and by the layer's initial state.
    # x: input tensor; w: kernel; b: optional bias (added along the channel axis when given);
    # padding: valid or same (default valid). Uses the cell's strides, data format and
    # dilation. Returns the convolved tensor. No side effects.
    conv_out = backend.conv2d(x, w, strides=self.strides,
                              padding=padding,
                              data_format=self.data_format,
                              dilation_rate=self.dilation_rate)
    if b is not None:
      conv_out = backend.bias_add(conv_out, b,
                                  data_format=self.data_format)
    return conv_out

  def recurrent_conv(self, x, w):
    # [FUNCTION DOC]
    # HELPER: convolution of the recurrent hidden state used by call. x: hidden state tensor;
    # w: recurrent kernel. Always uses stride one and same padding (so spatial size is kept),
    # with the cell's data format; no dilation, no bias. Returns the convolved tensor.
    conv_out = backend.conv2d(x, w, strides=(1, 1),
                              padding='same',
                              data_format=self.data_format)
    return conv_out

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a serializable mapping of the cell's configuration: convolution settings, the
    # serialized activations, initializers, regularizers and constraints, use_bias,
    # unit_forget_bias and the dropout rates, merged with the base layer configuration (the
    # cell's own entries win on key clashes). No side effects.
    config = {'filters': self.filters,
              'kernel_size': self.kernel_size,
              'strides': self.strides,
              'padding': self.padding,
              'data_format': self.data_format,
              'dilation_rate': self.dilation_rate,
              'activation': activations.serialize(self.activation),
              'recurrent_activation': activations.serialize(
                  self.recurrent_activation),
              'use_bias': self.use_bias,
              'kernel_initializer': initializers.serialize(
                  self.kernel_initializer),
              'recurrent_initializer': initializers.serialize(
                  self.recurrent_initializer),
              'bias_initializer': initializers.serialize(self.bias_initializer),
              'unit_forget_bias': self.unit_forget_bias,
              'kernel_regularizer': regularizers.serialize(
                  self.kernel_regularizer),
              'recurrent_regularizer': regularizers.serialize(
                  self.recurrent_regularizer),
              'bias_regularizer': regularizers.serialize(self.bias_regularizer),
              'kernel_constraint': constraints.serialize(
                  self.kernel_constraint),
              'recurrent_constraint': constraints.serialize(
                  self.recurrent_constraint),
              'bias_constraint': constraints.serialize(self.bias_constraint),
              'dropout': self.dropout,
              'recurrent_dropout': self.recurrent_dropout}
    base_config = super(ConvLSTM2DCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class ConvLSTM2D(ConvRNN2D):
  # [CLASS DOC]
  # Public two-dimensional convolutional LSTM layer. Subclass of ConvRNN2D that creates a
  # ConvLSTM2DCell from its constructor arguments and delegates all convolution, activation,
  # initializer, regularizer, constraint and dropout settings to it through read-only
  # properties. Additionally holds an activity regularizer. Input is a five-dimensional tensor
  # of image sequences; output follows the return_sequences and return_state options.
  """2D Convolutional LSTM layer.

  A convolutional LSTM is similar to an LSTM, but the input transformations
  and recurrent transformations are both convolutional. This layer is typically
  used to process timeseries of images (i.e. video-like data).

  It is known to perform well for weather data forecasting,
  using inputs that are timeseries of 2D grids of sensor values.
  It isn't usually applied to regular video data, due to its high computational
  cost.

  Args:
    filters: Integer, the dimensionality of the output space
      (i.e. the number of output filters in the convolution).
    kernel_size: An integer or tuple/list of n integers, specifying the
      dimensions of the convolution window.
    strides: An integer or tuple/list of n integers,
      specifying the strides of the convolution.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: One of `"valid"` or `"same"` (case-insensitive).
      `"valid"` means no padding. `"same"` results in padding evenly to
      the left/right or up/down of the input such that output has the same
      height/width dimension as the input.
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, time, ..., channels)`
      while `channels_first` corresponds to
      inputs with shape `(batch, time, channels, ...)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".
    dilation_rate: An integer or tuple/list of n integers, specifying
      the dilation rate to use for dilated convolution.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any `strides` value != 1.
    activation: Activation function to use.
      By default hyperbolic tangent activation function is applied
      (`tanh(x)`).
    recurrent_activation: Activation function to use
      for the recurrent step.
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix,
      used for the linear transformation of the inputs.
    recurrent_initializer: Initializer for the `recurrent_kernel`
      weights matrix,
      used for the linear transformation of the recurrent state.
    bias_initializer: Initializer for the bias vector.
    unit_forget_bias: Boolean.
      If True, add 1 to the bias of the forget gate at initialization.
      Use in combination with `bias_initializer="zeros"`.
      This is recommended in [Jozefowicz et al., 2015](
        http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix.
    recurrent_regularizer: Regularizer function applied to
      the `recurrent_kernel` weights matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to.
    kernel_constraint: Constraint function applied to
      the `kernel` weights matrix.
    recurrent_constraint: Constraint function applied to
      the `recurrent_kernel` weights matrix.
    bias_constraint: Constraint function applied to the bias vector.
    return_sequences: Boolean. Whether to return the last output
      in the output sequence, or the full sequence. (default False)
    return_state: Boolean Whether to return the last state
      in addition to the output. (default False)
    go_backwards: Boolean (default False).
      If True, process the input sequence backwards.
    stateful: Boolean (default False). If True, the last state
      for each sample at index i in a batch will be used as initial
      state for the sample of index i in the following batch.
    dropout: Float between 0 and 1.
      Fraction of the units to drop for
      the linear transformation of the inputs.
    recurrent_dropout: Float between 0 and 1.
      Fraction of the units to drop for
      the linear transformation of the recurrent state.

  Call arguments:
    inputs: A 5D float tensor (see input shape description below).
    mask: Binary tensor of shape `(samples, timesteps)` indicating whether
      a given timestep should be masked.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. This argument is passed to the cell
      when calling it. This is only relevant if `dropout` or `recurrent_dropout`
      are set.
    initial_state: List of initial state tensors to be passed to the first
      call of the cell.

  Input shape:
    - If data_format='channels_first'
        5D tensor with shape:
        `(samples, time, channels, rows, cols)`
    - If data_format='channels_last'
        5D tensor with shape:
        `(samples, time, rows, cols, channels)`

  Output shape:
    - If `return_state`: a list of tensors. The first tensor is
      the output. The remaining tensors are the last states,
      each 4D tensor with shape:
      `(samples, filters, new_rows, new_cols)`
      if data_format='channels_first'
      or 4D tensor with shape:
      `(samples, new_rows, new_cols, filters)`
      if data_format='channels_last'.
      `rows` and `cols` values might have changed due to padding.
    - If `return_sequences`: 5D tensor with shape:
      `(samples, timesteps, filters, new_rows, new_cols)`
      if data_format='channels_first'
      or 5D tensor with shape:
      `(samples, timesteps, new_rows, new_cols, filters)`
      if data_format='channels_last'.
    - Else, 4D tensor with shape:
      `(samples, filters, new_rows, new_cols)`
      if data_format='channels_first'
      or 4D tensor with shape:
      `(samples, new_rows, new_cols, filters)`
      if data_format='channels_last'.

  Raises:
    ValueError: in case of invalid constructor arguments.

  References:
    - [Shi et al., 2015](http://arxiv.org/abs/1506.04214v1)
    (the current implementation does not include the feedback loop on the
    cells output).

  Example:

  ```python
  steps = 10
  height = 32
  width = 32
  input_channels = 3
  output_channels = 6

  inputs = tf.keras.Input(shape=(steps, height, width, input_channels))
  layer = tf.keras.layers.ConvLSTM2D(filters=output_channels, kernel_size=3)
  outputs = layer(inputs)
  ```
  """

  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1),
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               dropout=0.,
               recurrent_dropout=0.,
               **kwargs):
    # [FUNCTION DOC]
    # Constructor. Takes the same cell options as ConvLSTM2DCell (filters, kernel_size,
    # strides, padding, data_format, dilation_rate, activations, bias and initializer,
    # regularizer and constraint settings, dropout rates), plus activity_regularizer,
    # return_sequences, return_state, go_backwards, stateful and extra keyword arguments
    # passed to the base layer. Creates the cell (forwarding only the dtype keyword, if any),
    # calls the base constructor and stores the resolved activity regularizer.
    # Raises a value error on invalid cell arguments. Returns nothing.
    cell = ConvLSTM2DCell(filters=filters,
                          kernel_size=kernel_size,
                          strides=strides,
                          padding=padding,
                          data_format=data_format,
                          dilation_rate=dilation_rate,
                          activation=activation,
                          recurrent_activation=recurrent_activation,
                          use_bias=use_bias,
                          kernel_initializer=kernel_initializer,
                          recurrent_initializer=recurrent_initializer,
                          bias_initializer=bias_initializer,
                          unit_forget_bias=unit_forget_bias,
                          kernel_regularizer=kernel_regularizer,
                          recurrent_regularizer=recurrent_regularizer,
                          bias_regularizer=bias_regularizer,
                          kernel_constraint=kernel_constraint,
                          recurrent_constraint=recurrent_constraint,
                          bias_constraint=bias_constraint,
                          dropout=dropout,
                          recurrent_dropout=recurrent_dropout,
                          dtype=kwargs.get('dtype'))
    super(ConvLSTM2D, self).__init__(cell,
                                     return_sequences=return_sequences,
                                     return_state=return_state,
                                     go_backwards=go_backwards,
                                     stateful=stateful,
                                     **kwargs)
    self.activity_regularizer = regularizers.get(activity_regularizer)

  def call(self, inputs, mask=None, training=None, initial_state=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: does not accept or forward the constants argument of the base class, so
    # external constants cannot be passed through this layer's call.
    # Thin override that forwards inputs, mask, training flag and initial_state to the base
    # class call and returns its result (output, sequence, or list with final states).
    return super(ConvLSTM2D, self).call(inputs,
                                        mask=mask,
                                        training=training,
                                        initial_state=initial_state)

  @property
  def filters(self):
    # [FUNCTION DOC]
    # Read-only property delegating to the underlying cell: returns the cell's filters setting.
    # No side effects.
    return self.cell.filters

  @property
  def kernel_size(self):
    # [FUNCTION DOC]
    # Read-only property delegating to the underlying cell: returns the cell's kernel_size setting.
    # No side effects.
    return self.cell.kernel_size

  @property
  def strides(self):
    # [FUNCTION DOC]
    # Read-only property delegating to the underlying cell: returns the cell's strides setting.
    # No side effects.
    return self.cell.strides

  @property
  def padding(self):
    # [FUNCTION DOC]
    # Read-only property delegating to the underlying cell: returns the cell's padding setting.
    # No side effects.
    return self.cell.padding

  @property
  def data_format(self):
    # [FUNCTION DOC]
    # Read-only property delegating to the underlying cell: returns the cell's data_format setting.
    # No side effects.
    return self.cell.data_format

  @property
  def dilation_rate(self):
    # [FUNCTION DOC]
    # Read-only property delegating to the underlying cell: returns the cell's dilation_rate setting.
    # No side effects.
    return self.cell.dilation_rate

  @property
  def activation(self):
    # [FUNCTION DOC]
    # Read-only property delegating to the underlying cell: returns the cell's activation setting.
    # No side effects.
    return self.cell.activation

  @property
  def recurrent_activation(self):
    # [FUNCTION DOC]
    # Read-only property delegating to the underlying cell: returns the cell's recurrent_activation setting.
    # No side effects.
    return self.cell.recurrent_activation

  @property
  def use_bias(self):
    # [FUNCTION DOC]
    # Read-only property delegating to the underlying cell: returns the cell's use_bias setting.
    # No side effects.
    return self.cell.use_bias

  @property
  def kernel_initializer(self):
    # [FUNCTION DOC]
    # Read-only property delegating to the underlying cell: returns the cell's kernel_initializer setting.
    # No side effects.
    return self.cell.kernel_initializer

  @property
  def recurrent_initializer(self):
    # [FUNCTION DOC]
    # Read-only property delegating to the underlying cell: returns the cell's recurrent_initializer setting.
    # No side effects.
    return self.cell.recurrent_initializer

  @property
  def bias_initializer(self):
    # [FUNCTION DOC]
    # Read-only property delegating to the underlying cell: returns the cell's bias_initializer setting.
    # No side effects.
    return self.cell.bias_initializer

  @property
  def unit_forget_bias(self):
    # [FUNCTION DOC]
    # Read-only property delegating to the underlying cell: returns the cell's unit_forget_bias setting.
    # No side effects.
    return self.cell.unit_forget_bias

  @property
  def kernel_regularizer(self):
    # [FUNCTION DOC]
    # Read-only property delegating to the underlying cell: returns the cell's kernel_regularizer setting.
    # No side effects.
    return self.cell.kernel_regularizer

  @property
  def recurrent_regularizer(self):
    # [FUNCTION DOC]
    # Read-only property delegating to the underlying cell: returns the cell's recurrent_regularizer setting.
    # No side effects.
    return self.cell.recurrent_regularizer

  @property
  def bias_regularizer(self):
    # [FUNCTION DOC]
    # Read-only property delegating to the underlying cell: returns the cell's bias_regularizer setting.
    # No side effects.
    return self.cell.bias_regularizer

  @property
  def kernel_constraint(self):
    # [FUNCTION DOC]
    # Read-only property delegating to the underlying cell: returns the cell's kernel_constraint setting.
    # No side effects.
    return self.cell.kernel_constraint

  @property
  def recurrent_constraint(self):
    # [FUNCTION DOC]
    # Read-only property delegating to the underlying cell: returns the cell's recurrent_constraint setting.
    # No side effects.
    return self.cell.recurrent_constraint

  @property
  def bias_constraint(self):
    # [FUNCTION DOC]
    # Read-only property delegating to the underlying cell: returns the cell's bias_constraint setting.
    # No side effects.
    return self.cell.bias_constraint

  @property
  def dropout(self):
    # [FUNCTION DOC]
    # Read-only property delegating to the underlying cell: returns the cell's dropout setting.
    # No side effects.
    return self.cell.dropout

  @property
  def recurrent_dropout(self):
    # [FUNCTION DOC]
    # Read-only property delegating to the underlying cell: returns the cell's recurrent_dropout setting.
    # No side effects.
    return self.cell.recurrent_dropout

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a serializable mapping of the layer's configuration: the delegated cell settings
    # (serialized activations, initializers, regularizers, constraints), the serialized activity
    # regularizer and the dropout rates, merged with the base configuration after removing its
    # cell entry (so the config can rebuild the layer from flat arguments). No side effects.
    config = {'filters': self.filters,
              'kernel_size': self.kernel_size,
              'strides': self.strides,
              'padding': self.padding,
              'data_format': self.data_format,
              'dilation_rate': self.dilation_rate,
              'activation': activations.serialize(self.activation),
              'recurrent_activation': activations.serialize(
                  self.recurrent_activation),
              'use_bias': self.use_bias,
              'kernel_initializer': initializers.serialize(
                  self.kernel_initializer),
              'recurrent_initializer': initializers.serialize(
                  self.recurrent_initializer),
              'bias_initializer': initializers.serialize(self.bias_initializer),
              'unit_forget_bias': self.unit_forget_bias,
              'kernel_regularizer': regularizers.serialize(
                  self.kernel_regularizer),
              'recurrent_regularizer': regularizers.serialize(
                  self.recurrent_regularizer),
              'bias_regularizer': regularizers.serialize(self.bias_regularizer),
              'activity_regularizer': regularizers.serialize(
                  self.activity_regularizer),
              'kernel_constraint': constraints.serialize(
                  self.kernel_constraint),
              'recurrent_constraint': constraints.serialize(
                  self.recurrent_constraint),
              'bias_constraint': constraints.serialize(self.bias_constraint),
              'dropout': self.dropout,
              'recurrent_dropout': self.recurrent_dropout}
    base_config = super(ConvLSTM2D, self).get_config()
    del base_config['cell']
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    # [FUNCTION DOC]
    # Class method that rebuilds a layer from a configuration mapping produced by get_config,
    # by passing every entry as a constructor argument. Returns the new layer instance.
    return cls(**config)
