# [MODULE DOC]
# Legacy (tf.python.keras) recurrent layer implementations. Defines the generic RNN layer that
# wraps a cell and unrolls it over time through the backend rnn loop, the StackedRNNCells wrapper
# that chains several cells into one, the abstract cell base class, a mixin that caches dropout
# masks for cells, the concrete cells and layers for simple RNN, GRU and LSTM, a peephole LSTM
# cell, and a handful of private helpers (dropout mask generation, argument standardization,
# zero-filled initial state creation, caching device selection and config helpers for the
# caching-device option).
# Layers in this module are consumed by the keras layers package, which re-exports the public
# classes, and by the v2 recurrent implementations that subclass these.
# Import-time side effects: none beyond importing dependencies and defining one module-level
# warning message string used when recurrent dropout forces the first implementation.
# This module is not a helper module; it is a primary layer-definition module.
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
"""Recurrent layers and their base classes."""

import collections
import warnings

import numpy as np

from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.saving.saved_model import layer_serialization
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import nest
from tensorflow.tools.docs import doc_controls


RECURRENT_DROPOUT_WARNING_MSG = (
    'RNN `implementation=2` is not supported when `recurrent_dropout` is set. '
    'Using `implementation=1`.')


class StackedRNNCells(Layer):
  # [CLASS DOC]
  # Layer that wraps a list of RNN cells so that the whole stack behaves like a single cell.
  # Holds the ordered list of cells and a flag saying whether the reported state structure is in
  # reverse cell order (kept for backward compatibility). Instances are created with a list of
  # cells (each must expose a call method and a state size) and are normally passed to the RNN
  # layer, which does this automatically when handed a list of cells. The output of each cell
  # feeds the next one and the states of all cells are gathered into one nested structure.
  """Wrapper allowing a stack of RNN cells to behave as a single cell.

  Used to implement efficient stacked RNNs.

  Args:
    cells: List of RNN cell instances.

  Examples:

  ```python
  batch_size = 3
  sentence_max_length = 5
  n_features = 2
  new_shape = (batch_size, sentence_max_length, n_features)
  x = tf.constant(np.reshape(np.arange(30), new_shape), dtype = tf.float32)

  rnn_cells = [tf.keras.layers.LSTMCell(128) for _ in range(2)]
  stacked_lstm = tf.keras.layers.StackedRNNCells(rnn_cells)
  lstm_layer = tf.keras.layers.RNN(stacked_lstm)

  result = lstm_layer(x)
  ```
  """

  def __init__(self, cells, **kwargs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the validation errors pass the cells list as a second argument to the error
    # constructor instead of formatting it into the message, so the message is a two-element
    # argument tuple rather than one readable sentence.
    # Constructor. Takes a list of cell objects and an optional keyword reverse_state_order
    # (boolean, default false, removed from the keyword arguments before they go to the base
    # layer). Raises a value error if any cell lacks a call method or a state size attribute.
    # Stores the cells and the flag; logs a warning when reverse_state_order is true because that
    # option is slated for deprecation. Remaining keyword arguments go to the base Layer.
    for cell in cells:
      if not 'call' in dir(cell):
        raise ValueError('All cells must have a `call` method. '
                         'received cells:', cells)
      if not 'state_size' in dir(cell):
        raise ValueError('All cells must have a '
                         '`state_size` attribute. '
                         'received cells:', cells)
    self.cells = cells
    # reverse_state_order determines whether the state size will be in a reverse
    # order of the cells' state. User might want to set this to True to keep the
    # existing behavior. This is only useful when use RNN(return_state=True)
    # since the state will be returned as the same order of state_size.
    self.reverse_state_order = kwargs.pop('reverse_state_order', False)
    if self.reverse_state_order:
      logging.warning('reverse_state_order=True in StackedRNNCells will soon '
                      'be deprecated. Please update the code to work with the '
                      'natural order of states if you rely on the RNN states, '
                      'eg RNN(return_state=True).')
    super(StackedRNNCells, self).__init__(**kwargs)

  @property
  def state_size(self):
    # [FUNCTION DOC]
    # Read-only property. Returns a tuple holding the state size of every wrapped cell, in cell
    # order (or reversed cell order when reverse_state_order is set).
    return tuple(c.state_size for c in
                 (self.cells[::-1] if self.reverse_state_order else self.cells))

  @property
  def output_size(self):
    # [FUNCTION DOC]
    # Read-only property giving the output size of the stack, i.e. that of the last cell. Uses the
    # last cell's output size when it has one, else the first element of its state size when the
    # state size has several entries, else the state size itself.
    if getattr(self.cells[-1], 'output_size', None) is not None:
      return self.cells[-1].output_size
    elif _is_multiple_state(self.cells[-1].state_size):
      return self.cells[-1].state_size[0]
    else:
      return self.cells[-1].state_size

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    # [FUNCTION DOC]
    # Builds the initial states for the whole stack. Inputs (optional tensor), batch_size
    # (optional scalar) and dtype (optional) describe the batch; either inputs or the
    # batch_size and dtype pair must be usable. For each cell (reversed if reverse_state_order is
    # set) it asks the cell for its own initial state when the cell supports that, else creates a
    # zero-filled state from the cell's state size. Returns a tuple with one entry per cell.
    initial_states = []
    for cell in self.cells[::-1] if self.reverse_state_order else self.cells:
      get_initial_state_fn = getattr(cell, 'get_initial_state', None)
      if get_initial_state_fn:
        initial_states.append(get_initial_state_fn(
            inputs=inputs, batch_size=batch_size, dtype=dtype))
      else:
        initial_states.append(_generate_zero_filled_state_for_cell(
            cell, inputs, batch_size, dtype))

    return tuple(initial_states)

  def call(self, inputs, states, constants=None, training=None, **kwargs):
    # [FUNCTION DOC]
    # Runs one time step through every cell in order. Inputs is the step input; states is the
    # flat or nested collection of all cell states; constants is an optional list of constant
    # tensors passed only to cells whose call accepts them; training is forwarded only to cells
    # whose call accepts it; other keyword arguments are forwarded to each cell. The states are
    # regrouped per cell, each cell consumes the previous cell's output, and tensor-style TF cells
    # with a single state get it unwrapped. Returns a pair: the last cell's output and the new
    # states packed in the same structure as the state size. The keyword dictionary is mutated
    # (training key added or removed) while looping.
    # Recover per-cell states.
    state_size = (self.state_size[::-1]
                  if self.reverse_state_order else self.state_size)
    nested_states = nest.pack_sequence_as(state_size, nest.flatten(states))

    # Call the cells in order and store the returned states.
    new_nested_states = []
    for cell, states in zip(self.cells, nested_states):
      states = states if nest.is_nested(states) else [states]
      # TF cell does not wrap the state into list when there is only one state.
      is_tf_rnn_cell = getattr(cell, '_is_tf_rnn_cell', None) is not None
      states = states[0] if len(states) == 1 and is_tf_rnn_cell else states
      if generic_utils.has_arg(cell.call, 'training'):
        kwargs['training'] = training
      else:
        kwargs.pop('training', None)
      # Use the __call__ function for callable objects, eg layers, so that it
      # will have the proper name scopes for the ops, etc.
      cell_call_fn = cell.__call__ if callable(cell) else cell.call
      if generic_utils.has_arg(cell.call, 'constants'):
        inputs, states = cell_call_fn(inputs, states,
                                      constants=constants, **kwargs)
      else:
        inputs, states = cell_call_fn(inputs, states, **kwargs)
      new_nested_states.append(states)

    return inputs, nest.pack_sequence_as(state_size,
                                         nest.flatten(new_nested_states))

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    # [FUNCTION DOC]
    # Builds every cell that is a layer and not yet built, inside a name scope named after the
    # cell. Input_shape is the shape of the step input (a list is reduced to its first element).
    # After each cell the shape for the next cell is derived from that cell's output size (or the
    # first state size when it has none). Marks this layer as built. Side effects: creates the
    # cells' weights and sets their built flags.
    if isinstance(input_shape, list):
      input_shape = input_shape[0]
    for cell in self.cells:
      if isinstance(cell, Layer) and not cell.built:
        with backend.name_scope(cell.name):
          cell.build(input_shape)
          cell.built = True
      if getattr(cell, 'output_size', None) is not None:
        output_dim = cell.output_size
      elif _is_multiple_state(cell.state_size):
        output_dim = cell.state_size[0]
      else:
        output_dim = cell.state_size
      input_shape = tuple([input_shape[0]] +
                          tensor_shape.TensorShape(output_dim).as_list())
    self.built = True

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a serializable configuration mapping: the base layer config plus a list with the
    # serialized configuration of each cell. Note that reverse_state_order is not included, so
    # it is lost on a round trip through the configuration.
    cells = []
    for cell in self.cells:
      cells.append(generic_utils.serialize_keras_object(cell))
    config = {'cells': cells}
    base_config = super(StackedRNNCells, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    # [FUNCTION DOC]
    # Class method that rebuilds a stacked-cell layer from a configuration mapping. The cell list
    # entry is removed from the given mapping (it is mutated), each cell configuration is
    # deserialized into a layer (optionally resolving names through the custom objects mapping),
    # and the class is constructed with the cells and the remaining configuration entries.
    # Returns the new instance.
    from tensorflow.python.keras.layers import deserialize as deserialize_layer  # pylint: disable=g-import-not-at-top
    cells = []
    for cell_config in config.pop('cells'):
      cells.append(
          deserialize_layer(cell_config, custom_objects=custom_objects))
    return cls(cells, **config)


class RNN(Layer):
  # [CLASS DOC]
  # Generic recurrent layer: wraps a cell (or a list of cells, which is stacked automatically)
  # and applies it across the time dimension of its input through the backend rnn loop. Holds the
  # cell, flags for returning sequences and states, backwards processing, statefulness,
  # unrolling, time-major layout and zero output for masked steps, plus the input, state and
  # constants specifications and, for stateful layers, the variables storing the last states.
  # Subclassed by SimpleRNN, GRU and LSTM here and by other recurrent layers elsewhere. Supports
  # masking, optional initial states and constants passed at call time, and saved-model
  # serialization through a dedicated saver.
  """Base class for recurrent layers.

  See [the Keras RNN API guide](https://www.tensorflow.org/guide/keras/rnn)
  for details about the usage of RNN API.

  Args:
    cell: A RNN cell instance or a list of RNN cell instances.
      A RNN cell is a class that has:
      - A `call(input_at_t, states_at_t)` method, returning
        `(output_at_t, states_at_t_plus_1)`. The call method of the
        cell can also take the optional argument `constants`, see
        section "Note on passing external constants" below.
      - A `state_size` attribute. This can be a single integer
        (single state) in which case it is the size of the recurrent
        state. This can also be a list/tuple of integers (one size per state).
        The `state_size` can also be TensorShape or tuple/list of
        TensorShape, to represent high dimension state.
      - A `output_size` attribute. This can be a single integer or a
        TensorShape, which represent the shape of the output. For backward
        compatible reason, if this attribute is not available for the
        cell, the value will be inferred by the first element of the
        `state_size`.
      - A `get_initial_state(inputs=None, batch_size=None, dtype=None)`
        method that creates a tensor meant to be fed to `call()` as the
        initial state, if the user didn't specify any initial state via other
        means. The returned initial state should have a shape of
        [batch_size, cell.state_size]. The cell might choose to create a
        tensor full of zeros, or full of other values based on the cell's
        implementation.
        `inputs` is the input tensor to the RNN layer, which should
        contain the batch size as its shape[0], and also dtype. Note that
        the shape[0] might be `None` during the graph construction. Either
        the `inputs` or the pair of `batch_size` and `dtype` are provided.
        `batch_size` is a scalar tensor that represents the batch size
        of the inputs. `dtype` is `tf.DType` that represents the dtype of
        the inputs.
        For backward compatibility, if this method is not implemented
        by the cell, the RNN layer will create a zero filled tensor with the
        size of [batch_size, cell.state_size].
      In the case that `cell` is a list of RNN cell instances, the cells
      will be stacked on top of each other in the RNN, resulting in an
      efficient stacked RNN.
    return_sequences: Boolean (default `False`). Whether to return the last
      output in the output sequence, or the full sequence.
    return_state: Boolean (default `False`). Whether to return the last state
      in addition to the output.
    go_backwards: Boolean (default `False`).
      If True, process the input sequence backwards and return the
      reversed sequence.
    stateful: Boolean (default `False`). If True, the last state
      for each sample at index i in a batch will be used as initial
      state for the sample of index i in the following batch.
    unroll: Boolean (default `False`).
      If True, the network will be unrolled, else a symbolic loop will be used.
      Unrolling can speed-up a RNN, although it tends to be more
      memory-intensive. Unrolling is only suitable for short sequences.
    time_major: The shape format of the `inputs` and `outputs` tensors.
      If True, the inputs and outputs will be in shape
      `(timesteps, batch, ...)`, whereas in the False case, it will be
      `(batch, timesteps, ...)`. Using `time_major = True` is a bit more
      efficient because it avoids transposes at the beginning and end of the
      RNN calculation. However, most TensorFlow data is batch-major, so by
      default this function accepts input and emits output in batch-major
      form.
    zero_output_for_mask: Boolean (default `False`).
      Whether the output should use zeros for the masked timesteps. Note that
      this field is only used when `return_sequences` is True and mask is
      provided. It can useful if you want to reuse the raw output sequence of
      the RNN without interference from the masked timesteps, eg, merging
      bidirectional RNNs.

  Call arguments:
    inputs: Input tensor.
    mask: Binary tensor of shape `[batch_size, timesteps]` indicating whether
      a given timestep should be masked. An individual `True` entry indicates
      that the corresponding timestep should be utilized, while a `False`
      entry indicates that the corresponding timestep should be ignored.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. This argument is passed to the cell
      when calling it. This is for use with cells that use dropout.
    initial_state: List of initial state tensors to be passed to the first
      call of the cell.
    constants: List of constant tensors to be passed to the cell at each
      timestep.

  Input shape:
    N-D tensor with shape `[batch_size, timesteps, ...]` or
    `[timesteps, batch_size, ...]` when time_major is True.

  Output shape:
    - If `return_state`: a list of tensors. The first tensor is
      the output. The remaining tensors are the last states,
      each with shape `[batch_size, state_size]`, where `state_size` could
      be a high dimension tensor shape.
    - If `return_sequences`: N-D tensor with shape
      `[batch_size, timesteps, output_size]`, where `output_size` could
      be a high dimension tensor shape, or
      `[timesteps, batch_size, output_size]` when `time_major` is True.
    - Else, N-D tensor with shape `[batch_size, output_size]`, where
      `output_size` could be a high dimension tensor shape.

  Masking:
    This layer supports masking for input data with a variable number
    of timesteps. To introduce masks to your data,
    use an [tf.keras.layers.Embedding] layer with the `mask_zero` parameter
    set to `True`.

  Note on using statefulness in RNNs:
    You can set RNN layers to be 'stateful', which means that the states
    computed for the samples in one batch will be reused as initial states
    for the samples in the next batch. This assumes a one-to-one mapping
    between samples in different successive batches.

    To enable statefulness:
      - Specify `stateful=True` in the layer constructor.
      - Specify a fixed batch size for your model, by passing
        If sequential model:
          `batch_input_shape=(...)` to the first layer in your model.
        Else for functional model with 1 or more Input layers:
          `batch_shape=(...)` to all the first layers in your model.
        This is the expected shape of your inputs
        *including the batch size*.
        It should be a tuple of integers, e.g. `(32, 10, 100)`.
      - Specify `shuffle=False` when calling `fit()`.

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

  Examples:

  ```python
  # First, let's define a RNN Cell, as a layer subclass.

  class MinimalRNNCell(keras.layers.Layer):

      def __init__(self, units, **kwargs):
          self.units = units
          self.state_size = units
          super(MinimalRNNCell, self).__init__(**kwargs)

      def build(self, input_shape):
          self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                        initializer='uniform',
                                        name='kernel')
          self.recurrent_kernel = self.add_weight(
              shape=(self.units, self.units),
              initializer='uniform',
              name='recurrent_kernel')
          self.built = True

      def call(self, inputs, states):
          prev_output = states[0]
          h = backend.dot(inputs, self.kernel)
          output = h + backend.dot(prev_output, self.recurrent_kernel)
          return output, [output]

  # Let's use this cell in a RNN layer:

  cell = MinimalRNNCell(32)
  x = keras.Input((None, 5))
  layer = RNN(cell)
  y = layer(x)

  # Here's how to use the cell to build a stacked RNN:

  cells = [MinimalRNNCell(32), MinimalRNNCell(64)]
  x = keras.Input((None, 5))
  layer = RNN(cells)
  y = layer(x)
  ```
  """

  def __init__(self,
               cell,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               unroll=False,
               time_major=False,
               **kwargs):
    # [FUNCTION DOC]
    # Constructor. Inputs: cell (a cell object, or a list or tuple of cells which is wrapped in a
    # stacked-cells layer), return_sequences, return_state, go_backwards, stateful, unroll and
    # time_major (all booleans, default false), and keyword arguments for the base layer. The
    # keyword zero_output_for_mask (boolean, default false) is removed from the keywords. If
    # only input_dim or input_length keywords are given they are converted into an input shape
    # pair. Raises a value error when the cell lacks a call method or state size attribute, or
    # when stateful is requested while a distribution strategy is active. Initializes the input,
    # state and constants specifications to empty, enables masking support and zero constants.
    if isinstance(cell, (list, tuple)):
      cell = StackedRNNCells(cell)
    if not 'call' in dir(cell):
      raise ValueError('`cell` should have a `call` method. '
                       'The RNN was passed:', cell)
    if not 'state_size' in dir(cell):
      raise ValueError('The RNN cell should have '
                       'an attribute `state_size` '
                       '(tuple of integers, '
                       'one integer per RNN state).')
    # If True, the output for masked timestep will be zeros, whereas in the
    # False case, output from previous timestep is returned for masked timestep.
    self.zero_output_for_mask = kwargs.pop('zero_output_for_mask', False)

    if 'input_shape' not in kwargs and (
        'input_dim' in kwargs or 'input_length' in kwargs):
      input_shape = (kwargs.pop('input_length', None),
                     kwargs.pop('input_dim', None))
      kwargs['input_shape'] = input_shape

    super(RNN, self).__init__(**kwargs)
    self.cell = cell
    self.return_sequences = return_sequences
    self.return_state = return_state
    self.go_backwards = go_backwards
    self.stateful = stateful
    self.unroll = unroll
    self.time_major = time_major

    self.supports_masking = True
    # The input shape is unknown yet, it could have nested tensor inputs, and
    # the input spec will be the list of specs for nested inputs, the structure
    # of the input_spec will be the same as the input.
    self.input_spec = None
    self.state_spec = None
    self._states = None
    self.constants_spec = None
    self._num_constants = 0

    if stateful:
      if distribute_lib.has_strategy():
        raise ValueError('RNNs with stateful=True not yet supported with '
                         'tf.distribute.Strategy.')

  @property
  def _use_input_spec_as_call_signature(self):
    # [FUNCTION DOC]
    # HELPER: used by the saved-model export logic to decide how to derive call signatures.
    # Read-only property. Returns false when the layer is unrolled (the time dimension must be
    # concrete, which the input spec cannot express); otherwise defers to the base layer's answer.
    if self.unroll:
      # When the RNN layer is unrolled, the time step shape cannot be unknown.
      # The input spec does not define the time step (because this layer can be
      # called with any time step value, as long as it is not None), so it
      # cannot be used as the call function signature when saving to SavedModel.
      return False
    return super(RNN, self)._use_input_spec_as_call_signature

  @property
  def states(self):
    # [FUNCTION DOC]
    # Read-only view of the recorded states. When no states have been created yet, returns a
    # list of empty placeholders matching the structure of the cell's state size (a bare
    # placeholder is wrapped in a list when the state size is not nested); otherwise returns the
    # stored states (variables of a stateful layer).
    if self._states is None:
      state = nest.map_structure(lambda _: None, self.cell.state_size)
      return state if nest.is_nested(self.cell.state_size) else [state]
    return self._states

  @states.setter
  # Automatic tracking catches "self._states" which adds an extra weight and
  # breaks HDF5 checkpoints.
  @trackable.no_automatic_dependency_tracking
  def states(self, states):
    # [FUNCTION DOC]
    # Setter for the recorded states. Stores the given states without registering them for
    # automatic dependency tracking (tracking would add an extra weight and break legacy HDF5
    # checkpoints). Returns nothing.
    self._states = states

  def compute_output_shape(self, input_shape):
    # [FUNCTION DOC]
    # Computes the static output shape(s) without running the layer. Input_shape is a shape (or
    # nested structure of shapes; a list is reduced to its first element). The result depends on
    # the cell output size and on return_sequences (adds the time axis), time_major (axis order)
    # and return_state. Returns a single shape, or a list of shapes (output first, then one
    # shape per state) when return_state is set. Does not change any state.
    if isinstance(input_shape, list):
      input_shape = input_shape[0]
    # Check whether the input shape contains any nested shapes. It could be
    # (tensor_shape(1, 2), tensor_shape(3, 4)) or (1, 2, 3) which is from numpy
    # inputs.
    try:
      input_shape = tensor_shape.TensorShape(input_shape)
    except (ValueError, TypeError):
      # A nested tensor input
      input_shape = nest.flatten(input_shape)[0]

    batch = input_shape[0]
    time_step = input_shape[1]
    if self.time_major:
      batch, time_step = time_step, batch

    if _is_multiple_state(self.cell.state_size):
      state_size = self.cell.state_size
    else:
      state_size = [self.cell.state_size]

    def _get_output_shape(flat_output_size):
      # [FUNCTION DOC]
      # HELPER: nested helper for the output shape computation.
      # Turns one cell output size into a full output shape by adding the batch axis, and the
      # time axis when sequences are returned, in the order given by the time-major setting.
      output_dim = tensor_shape.TensorShape(flat_output_size).as_list()
      if self.return_sequences:
        if self.time_major:
          output_shape = tensor_shape.TensorShape(
              [time_step, batch] + output_dim)
        else:
          output_shape = tensor_shape.TensorShape(
              [batch, time_step] + output_dim)
      else:
        output_shape = tensor_shape.TensorShape([batch] + output_dim)
      return output_shape

    if getattr(self.cell, 'output_size', None) is not None:
      # cell.output_size could be nested structure.
      output_shape = nest.flatten(nest.map_structure(
          _get_output_shape, self.cell.output_size))
      output_shape = output_shape[0] if len(output_shape) == 1 else output_shape
    else:
      # Note that state_size[0] could be a tensor_shape or int.
      output_shape = _get_output_shape(state_size[0])

    if self.return_state:
      def _get_state_shape(flat_state):
        # [FUNCTION DOC]
        # HELPER: nested helper for the output shape computation.
        # Converts one state size into a state shape by prepending the batch dimension.
        state_shape = [batch] + tensor_shape.TensorShape(flat_state).as_list()
        return tensor_shape.TensorShape(state_shape)
      state_shape = nest.map_structure(_get_state_shape, state_size)
      return generic_utils.to_list(output_shape) + nest.flatten(state_shape)
    else:
      return output_shape

  def compute_mask(self, inputs, mask):
    # [FUNCTION DOC]
    # Computes the mask to propagate downstream. Only the first mask of a possibly nested mask
    # structure is used. The output mask is that mask when return_sequences is set, else nothing.
    # When return_state is set, returns a list with the output mask followed by an empty entry per
    # state; otherwise returns the output mask alone. Inputs is unused.
    # Time step masks must be the same for each input.
    # This is because the mask for an RNN is of size [batch, time_steps, 1],
    # and specifies which time steps should be skipped, and a time step
    # must be skipped for all inputs.
    # TODO(scottzhu): Should we accept multiple different masks?
    mask = nest.flatten(mask)[0]
    output_mask = mask if self.return_sequences else None
    if self.return_state:
      state_mask = [None for _ in self.states]
      return [output_mask] + state_mask
    else:
      return output_mask

  def build(self, input_shape):
    # [FUNCTION DOC]
    # Builds the layer for a given input shape (single shape or nested structure; a list is
    # reduced to its first element). Creates or updates the input specification (batch axis left
    # open unless stateful, time axis left open), derives the per-step input shape, builds the
    # cell if it is an unbuilt layer (inside a name scope with the cell name), then either
    # validates an existing state specification (from a call-time initial state) or creates one
    # from the cell state sizes. For stateful layers it also creates the state variables via
    # reset_states. Marks the layer built. Can raise a value error through state validation or
    # through reset_states when the batch size is unknown.
    if isinstance(input_shape, list):
      input_shape = input_shape[0]
      # The input_shape here could be a nest structure.

    # do the tensor_shape to shapes here. The input could be single tensor, or a
    # nested structure of tensors.
    def get_input_spec(shape):
      # [FUNCTION DOC]
      # HELPER: nested helper of the build method.
      # Converts one input shape into an input specification, blanking the time axis and, for
      # non-stateful layers, the batch axis, honoring the time-major setting. Returns the spec.
      """Convert input shape to InputSpec."""
      if isinstance(shape, tensor_shape.TensorShape):
        input_spec_shape = shape.as_list()
      else:
        input_spec_shape = list(shape)
      batch_index, time_step_index = (1, 0) if self.time_major else (0, 1)
      if not self.stateful:
        input_spec_shape[batch_index] = None
      input_spec_shape[time_step_index] = None
      return InputSpec(shape=tuple(input_spec_shape))

    def get_step_input_shape(shape):
      # [FUNCTION DOC]
      # HELPER: nested helper of the build method.
      # Removes the time axis from an input shape (accepting shape objects or tuples) to get the
      # shape of a single step, honoring the time-major layout. Returns a tuple.
      if isinstance(shape, tensor_shape.TensorShape):
        shape = tuple(shape.as_list())
      # remove the timestep from the input_shape
      return shape[1:] if self.time_major else (shape[0],) + shape[2:]

    # Check whether the input shape contains any nested shapes. It could be
    # (tensor_shape(1, 2), tensor_shape(3, 4)) or (1, 2, 3) which is from numpy
    # inputs.
    try:
      input_shape = tensor_shape.TensorShape(input_shape)
    except (ValueError, TypeError):
      # A nested tensor input
      pass

    if not nest.is_nested(input_shape):
      # This indicates the there is only one input.
      if self.input_spec is not None:
        self.input_spec[0] = get_input_spec(input_shape)
      else:
        self.input_spec = [get_input_spec(input_shape)]
      step_input_shape = get_step_input_shape(input_shape)
    else:
      if self.input_spec is not None:
        self.input_spec[0] = nest.map_structure(get_input_spec, input_shape)
      else:
        self.input_spec = generic_utils.to_list(
            nest.map_structure(get_input_spec, input_shape))
      step_input_shape = nest.map_structure(get_step_input_shape, input_shape)

    # allow cell (if layer) to build before we set or validate state_spec.
    if isinstance(self.cell, Layer) and not self.cell.built:
      with backend.name_scope(self.cell.name):
        self.cell.build(step_input_shape)
        self.cell.built = True

    # set or validate state_spec
    if _is_multiple_state(self.cell.state_size):
      state_size = list(self.cell.state_size)
    else:
      state_size = [self.cell.state_size]

    if self.state_spec is not None:
      # initial_state was passed in call, check compatibility
      self._validate_state_spec(state_size, self.state_spec)
    else:
      self.state_spec = [
          InputSpec(shape=[None] + tensor_shape.TensorShape(dim).as_list())
          for dim in state_size
      ]
    if self.stateful:
      self.reset_states()
    self.built = True

  @staticmethod
  def _validate_state_spec(cell_state_sizes, init_state_specs):
    # [FUNCTION DOC]
    # HELPER: internal validation step for the build method.
    # Static method. Checks that the state specs derived from a user-provided initial state agree
    # with the cell's state sizes: same number of flattened entries and, ignoring the leading
    # batch axis, shapes compatible with each size. Raises a value error describing both
    # structures on any mismatch. Returns nothing.
    """Validate the state spec between the initial_state and the state_size.

    Args:
      cell_state_sizes: list, the `state_size` attribute from the cell.
      init_state_specs: list, the `state_spec` from the initial_state that is
        passed in `call()`.

    Raises:
      ValueError: When initial state spec is not compatible with the state size.
    """
    validation_error = ValueError(
        'An `initial_state` was passed that is not compatible with '
        '`cell.state_size`. Received `state_spec`={}; '
        'however `cell.state_size` is '
        '{}'.format(init_state_specs, cell_state_sizes))
    flat_cell_state_sizes = nest.flatten(cell_state_sizes)
    flat_state_specs = nest.flatten(init_state_specs)

    if len(flat_cell_state_sizes) != len(flat_state_specs):
      raise validation_error
    for cell_state_spec, cell_state_size in zip(flat_state_specs,
                                                flat_cell_state_sizes):
      if not tensor_shape.TensorShape(
          # Ignore the first axis for init_state which is for batch
          cell_state_spec.shape[1:]).is_compatible_with(
              tensor_shape.TensorShape(cell_state_size)):
        raise validation_error

  @doc_controls.do_not_doc_inheritable
  def get_initial_state(self, inputs):
    # [FUNCTION DOC]
    # Creates the default initial states for a batch. Inputs is a tensor or nested structure of
    # tensors (the first one gives batch size and dtype, honoring the time-major layout). Uses
    # the cell's own initial-state method when present, else zero-filled states from the cell's
    # state size. Returns a list of state tensors (a single state is wrapped into a list, and
    # tuple-like states are converted into a list).
    get_initial_state_fn = getattr(self.cell, 'get_initial_state', None)

    if nest.is_nested(inputs):
      # The input are nested sequences. Use the first element in the seq to get
      # batch size and dtype.
      inputs = nest.flatten(inputs)[0]

    input_shape = array_ops.shape(inputs)
    batch_size = input_shape[1] if self.time_major else input_shape[0]
    dtype = inputs.dtype
    if get_initial_state_fn:
      init_state = get_initial_state_fn(
          inputs=None, batch_size=batch_size, dtype=dtype)
    else:
      init_state = _generate_zero_filled_state(batch_size, self.cell.state_size,
                                               dtype)
    # Keras RNN expect the states in a list, even if it's a single state tensor.
    if not nest.is_nested(init_state):
      init_state = [init_state]
    # Force the state to be a list in case it is a namedtuple eg LSTMStateTuple.
    return list(init_state)

  def __call__(self, inputs, initial_state=None, constants=None, **kwargs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: after a symbolic call the temporary input specification is trimmed by slicing
    # off the last N entries where N is the number of additional specs; if that number is zero
    # (empty initial state or constants lists) the slice removes every entry and wipes the
    # input specification instead of leaving it untouched.
    # Entry point when the layer is invoked. Takes the main inputs, an optional initial state
    # (tensor or list) and optional constants (tensor or list); other keywords go to the base
    # call. Arguments are first standardized (they may arrive packed inside the inputs list).
    # With neither initial state nor constants it simply defers to the base layer. Otherwise the
    # state spec and constants spec are recorded (overwriting earlier ones), extra tensors must
    # be all symbolic Keras tensors or all not (mixing raises a value error). Symbolic extras are
    # appended to the inputs and the input spec is temporarily widened for the base call, then
    # trimmed; non-symbolic extras are passed on as keyword arguments. Returns the layer output.
    inputs, initial_state, constants = _standardize_args(inputs,
                                                         initial_state,
                                                         constants,
                                                         self._num_constants)

    if initial_state is None and constants is None:
      return super(RNN, self).__call__(inputs, **kwargs)

    # If any of `initial_state` or `constants` are specified and are Keras
    # tensors, then add them to the inputs and temporarily modify the
    # input_spec to include them.

    additional_inputs = []
    additional_specs = []
    if initial_state is not None:
      additional_inputs += initial_state
      self.state_spec = nest.map_structure(
          lambda s: InputSpec(shape=backend.int_shape(s)), initial_state)
      additional_specs += self.state_spec
    if constants is not None:
      additional_inputs += constants
      self.constants_spec = [
          InputSpec(shape=backend.int_shape(constant)) for constant in constants
      ]
      self._num_constants = len(constants)
      additional_specs += self.constants_spec
    # additional_inputs can be empty if initial_state or constants are provided
    # but empty (e.g. the cell is stateless).
    flat_additional_inputs = nest.flatten(additional_inputs)
    is_keras_tensor = backend.is_keras_tensor(
        flat_additional_inputs[0]) if flat_additional_inputs else True
    for tensor in flat_additional_inputs:
      if backend.is_keras_tensor(tensor) != is_keras_tensor:
        raise ValueError('The initial state or constants of an RNN'
                         ' layer cannot be specified with a mix of'
                         ' Keras tensors and non-Keras tensors'
                         ' (a "Keras tensor" is a tensor that was'
                         ' returned by a Keras layer, or by `Input`)')

    if is_keras_tensor:
      # Compute the full input spec, including state and constants
      full_input = [inputs] + additional_inputs
      if self.built:
        # Keep the input_spec since it has been populated in build() method.
        full_input_spec = self.input_spec + additional_specs
      else:
        # The original input_spec is None since there could be a nested tensor
        # input. Update the input_spec to match the inputs.
        full_input_spec = generic_utils.to_list(
            nest.map_structure(lambda _: None, inputs)) + additional_specs
      # Perform the call with temporarily replaced input_spec
      self.input_spec = full_input_spec
      output = super(RNN, self).__call__(full_input, **kwargs)
      # Remove the additional_specs from input spec and keep the rest. It is
      # important to keep since the input spec was populated by build(), and
      # will be reused in the stateful=True.
      self.input_spec = self.input_spec[:-len(additional_specs)]
      return output
    else:
      if initial_state is not None:
        kwargs['initial_state'] = initial_state
      if constants is not None:
        kwargs['constants'] = constants
      return super(RNN, self).__call__(inputs, **kwargs)

  def call(self,
           inputs,
           mask=None,
           training=None,
           initial_state=None,
           constants=None):
    # [FUNCTION DOC]
    # Forward computation. Inputs: a tensor, nested tensors or ragged tensor (padded and turned
    # into row lengths internally); mask (optional binary tensor, rejected for ragged inputs);
    # training flag (forwarded to the cell only if its call accepts it); initial_state (optional
    # list of tensors) and constants (optional list, requires the cell to accept constants).
    # Resets cached dropout masks of the cell(s), builds a per-step function wrapping the cell,
    # and runs the backend rnn loop with the layer's settings (backwards, unroll, time-major,
    # zero output for mask). For stateful layers the final states are assigned into the stored
    # state variables via layer updates. Returns the last output, or the full sequence when
    # return_sequences is set (converted back to ragged if the input was ragged), followed by the
    # final states as a list when return_state is set. Raises a value error when unrolling with an
    # unknown time dimension or when constants are given to a cell that does not accept them.
    # The input should be dense, padded with zeros. If a ragged input is fed
    # into the layer, it is padded and the row lengths are used for masking.
    inputs, row_lengths = backend.convert_inputs_if_ragged(inputs)
    is_ragged_input = (row_lengths is not None)
    self._validate_args_if_ragged(is_ragged_input, mask)

    inputs, initial_state, constants = self._process_inputs(
        inputs, initial_state, constants)

    self._maybe_reset_cell_dropout_mask(self.cell)
    if isinstance(self.cell, StackedRNNCells):
      for cell in self.cell.cells:
        self._maybe_reset_cell_dropout_mask(cell)

    if mask is not None:
      # Time step masks must be the same for each input.
      # TODO(scottzhu): Should we accept multiple different masks?
      mask = nest.flatten(mask)[0]

    if nest.is_nested(inputs):
      # In the case of nested input, use the first element for shape check.
      input_shape = backend.int_shape(nest.flatten(inputs)[0])
    else:
      input_shape = backend.int_shape(inputs)
    timesteps = input_shape[0] if self.time_major else input_shape[1]
    if self.unroll and timesteps is None:
      raise ValueError('Cannot unroll a RNN if the '
                       'time dimension is undefined. \n'
                       '- If using a Sequential model, '
                       'specify the time dimension by passing '
                       'an `input_shape` or `batch_input_shape` '
                       'argument to your first layer. If your '
                       'first layer is an Embedding, you can '
                       'also use the `input_length` argument.\n'
                       '- If using the functional API, specify '
                       'the time dimension by passing a `shape` '
                       'or `batch_shape` argument to your Input layer.')

    kwargs = {}
    if generic_utils.has_arg(self.cell.call, 'training'):
      kwargs['training'] = training

    # TF RNN cells expect single tensor as state instead of list wrapped tensor.
    is_tf_rnn_cell = getattr(self.cell, '_is_tf_rnn_cell', None) is not None
    # Use the __call__ function for callable objects, eg layers, so that it
    # will have the proper name scopes for the ops, etc.
    cell_call_fn = self.cell.__call__ if callable(self.cell) else self.cell.call
    if constants:
      if not generic_utils.has_arg(self.cell.call, 'constants'):
        raise ValueError('RNN cell does not support constants')

      def step(inputs, states):
        # [FUNCTION DOC]
        # HELPER: per-timestep function handed to the backend rnn loop (variant with constants).
        # Splits the trailing constants off the flat state list, unwraps a single state for
        # TF-style cells, calls the cell with the constants and the captured keyword arguments,
        # and returns the cell output with the new states guaranteed to be a list.
        constants = states[-self._num_constants:]  # pylint: disable=invalid-unary-operand-type
        states = states[:-self._num_constants]  # pylint: disable=invalid-unary-operand-type

        states = states[0] if len(states) == 1 and is_tf_rnn_cell else states
        output, new_states = cell_call_fn(
            inputs, states, constants=constants, **kwargs)
        if not nest.is_nested(new_states):
          new_states = [new_states]
        return output, new_states
    else:

      def step(inputs, states):
        # [FUNCTION DOC]
        # HELPER: per-timestep function handed to the backend rnn loop (variant without
        # constants). Unwraps a single state for TF-style cells, calls the cell with the captured
        # keyword arguments, and returns the cell output with the new states forced to a list.
        states = states[0] if len(states) == 1 and is_tf_rnn_cell else states
        output, new_states = cell_call_fn(inputs, states, **kwargs)
        if not nest.is_nested(new_states):
          new_states = [new_states]
        return output, new_states
    last_output, outputs, states = backend.rnn(
        step,
        inputs,
        initial_state,
        constants=constants,
        go_backwards=self.go_backwards,
        mask=mask,
        unroll=self.unroll,
        input_length=row_lengths if row_lengths is not None else timesteps,
        time_major=self.time_major,
        zero_output_for_mask=self.zero_output_for_mask)

    if self.stateful:
      updates = [
          state_ops.assign(self_state, state) for self_state, state in zip(
              nest.flatten(self.states), nest.flatten(states))
      ]
      self.add_update(updates)

    if self.return_sequences:
      output = backend.maybe_convert_to_ragged(
          is_ragged_input, outputs, row_lengths, go_backwards=self.go_backwards)
    else:
      output = last_output

    if self.return_state:
      if not isinstance(states, (list, tuple)):
        states = [states]
      else:
        states = list(states)
      return generic_utils.to_list(output) + states
    else:
      return output

  def _process_inputs(self, inputs, initial_state, constants):
    # [FUNCTION DOC]
    # HELPER: argument preparation step of the forward computation.
    # Takes the inputs (possibly a list packing the initial states and constants after the real
    # input), initial_state and constants. If the inputs are a non-tuple sequence they are split
    # back into real input, initial states and constants (using the stored constants count). For
    # stateful layers it uses the stored states, but when an initial state is also given it picks
    # the stored states if any of them is non-zero and the given initial state otherwise; for
    # non-stateful layers with no initial state it creates default states. Raises a value error if
    # the number of initial states differs from the number of states of the layer. Returns the
    # triple of inputs, initial states and constants.
    # input shape: `(samples, time (padded with zeros), input_dim)`
    # note that the .build() method of subclasses MUST define
    # self.input_spec and self.state_spec with complete input shapes.
    if (isinstance(inputs, collections.abc.Sequence)
        and not isinstance(inputs, tuple)):
      # get initial_state from full input spec
      # as they could be copied to multiple GPU.
      if not self._num_constants:
        initial_state = inputs[1:]
      else:
        initial_state = inputs[1:-self._num_constants]
        constants = inputs[-self._num_constants:]
      if len(initial_state) == 0:
        initial_state = None
      inputs = inputs[0]

    if self.stateful:
      if initial_state is not None:
        # When layer is stateful and initial_state is provided, check if the
        # recorded state is same as the default value (zeros). Use the recorded
        # state if it is not same as the default.
        non_zero_count = math_ops.add_n([math_ops.count_nonzero_v2(s)
                                         for s in nest.flatten(self.states)])
        # Set strict = True to keep the original structure of the state.
        initial_state = cond.cond(non_zero_count > 0,
                                  true_fn=lambda: self.states,
                                  false_fn=lambda: initial_state,
                                  strict=True)
      else:
        initial_state = self.states
    elif initial_state is None:
      initial_state = self.get_initial_state(inputs)

    if len(initial_state) != len(self.states):
      raise ValueError('Layer has ' + str(len(self.states)) +
                       ' states but was passed ' + str(len(initial_state)) +
                       ' initial states.')
    return inputs, initial_state, constants

  def _validate_args_if_ragged(self, is_ragged_input, mask):
    # [FUNCTION DOC]
    # HELPER: input validation step of the forward computation.
    # Does nothing for dense inputs. For ragged inputs raises a value error if a mask was
    # supplied (ragged inputs derive their own mask) or if the layer is unrolled. Returns nothing.
    if not is_ragged_input:
      return

    if mask is not None:
      raise ValueError('The mask that was passed in was ' + str(mask) +
                       ' and cannot be applied to RaggedTensor inputs. Please '
                       'make sure that there is no mask passed in by upstream '
                       'layers.')
    if self.unroll:
      raise ValueError('The input received contains RaggedTensors and does '
                       'not support unrolling. Disable unrolling by passing '
                       '`unroll=False` in the RNN Layer constructor.')

  def _maybe_reset_cell_dropout_mask(self, cell):
    # [FUNCTION DOC]
    # HELPER: called at the start of each forward computation.
    # If the given cell uses the dropout-mask mixin, clears its cached input and recurrent dropout
    # masks so that new masks are drawn for this batch. Other cells are left alone.
    if isinstance(cell, DropoutRNNCellMixin):
      cell.reset_dropout_mask()
      cell.reset_recurrent_dropout_mask()

  def reset_states(self, states=None):
    # [FUNCTION DOC]
    # Resets or initializes the stored states of a stateful layer. The optional states argument
    # is an array or list of arrays (numpy style) with the values to install; when nothing is
    # given, zeros are used. Raises an attribute error if the layer is not stateful and a value
    # error if the batch size is unknown, if the number of provided state values is wrong, or if a
    # provided value's shape differs from its state variable's shape. If the state variables do
    # not exist yet they are created as backend variables from the cell's initial state (the
    # provided values are ignored in that first case); otherwise existing variables are updated in
    # place with zeros or the provided values. Side effect: modifies variable values.
    """Reset the recorded states for the stateful RNN layer.

    Can only be used when RNN layer is constructed with `stateful` = `True`.
    Args:
      states: Numpy arrays that contains the value for the initial state, which
        will be feed to cell at the first time step. When the value is None,
        zero filled numpy array will be created based on the cell state size.

    Raises:
      AttributeError: When the RNN layer is not stateful.
      ValueError: When the batch size of the RNN layer is unknown.
      ValueError: When the input numpy array is not compatible with the RNN
        layer state, either size wise or dtype wise.
    """
    if not self.stateful:
      raise AttributeError('Layer must be stateful.')
    spec_shape = None
    if self.input_spec is not None:
      spec_shape = nest.flatten(self.input_spec[0])[0].shape
    if spec_shape is None:
      # It is possible to have spec shape to be None, eg when construct a RNN
      # with a custom cell, or standard RNN layers (LSTM/GRU) which we only know
      # it has 3 dim input, but not its full shape spec before build().
      batch_size = None
    else:
      batch_size = spec_shape[1] if self.time_major else spec_shape[0]
    if not batch_size:
      raise ValueError('If a RNN is stateful, it needs to know '
                       'its batch size. Specify the batch size '
                       'of your input tensors: \n'
                       '- If using a Sequential model, '
                       'specify the batch size by passing '
                       'a `batch_input_shape` '
                       'argument to your first layer.\n'
                       '- If using the functional API, specify '
                       'the batch size by passing a '
                       '`batch_shape` argument to your Input layer.')
    # initialize state if None
    if nest.flatten(self.states)[0] is None:
      if getattr(self.cell, 'get_initial_state', None):
        flat_init_state_values = nest.flatten(self.cell.get_initial_state(
            inputs=None, batch_size=batch_size,
            dtype=self.dtype or backend.floatx()))
      else:
        flat_init_state_values = nest.flatten(_generate_zero_filled_state(
            batch_size, self.cell.state_size, self.dtype or backend.floatx()))
      flat_states_variables = nest.map_structure(
          backend.variable, flat_init_state_values)
      self.states = nest.pack_sequence_as(self.cell.state_size,
                                          flat_states_variables)
      if not nest.is_nested(self.states):
        self.states = [self.states]
    elif states is None:
      for state, size in zip(nest.flatten(self.states),
                             nest.flatten(self.cell.state_size)):
        backend.set_value(
            state,
            np.zeros([batch_size] + tensor_shape.TensorShape(size).as_list()))
    else:
      flat_states = nest.flatten(self.states)
      flat_input_states = nest.flatten(states)
      if len(flat_input_states) != len(flat_states):
        raise ValueError('Layer ' + self.name + ' expects ' +
                         str(len(flat_states)) + ' states, '
                         'but it received ' + str(len(flat_input_states)) +
                         ' state values. Input received: ' + str(states))
      set_value_tuples = []
      for i, (value, state) in enumerate(zip(flat_input_states,
                                             flat_states)):
        if value.shape != state.shape:
          raise ValueError(
              'State ' + str(i) + ' is incompatible with layer ' +
              self.name + ': expected shape=' + str(
                  (batch_size, state)) + ', found shape=' + str(value.shape))
        set_value_tuples.append((state, value))
      backend.batch_set_value(set_value_tuples)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the serializable configuration: the base layer config plus the return and
    # direction flags, stateful, unroll, time-major, the serialized cell, and, only when
    # non-default, the constants count and the zero-output-for-mask flag.
    config = {
        'return_sequences': self.return_sequences,
        'return_state': self.return_state,
        'go_backwards': self.go_backwards,
        'stateful': self.stateful,
        'unroll': self.unroll,
        'time_major': self.time_major
    }
    if self._num_constants:
      config['num_constants'] = self._num_constants
    if self.zero_output_for_mask:
      config['zero_output_for_mask'] = self.zero_output_for_mask

    config['cell'] = generic_utils.serialize_keras_object(self.cell)
    base_config = super(RNN, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    # [FUNCTION DOC]
    # Class method that rebuilds an RNN layer from a configuration mapping (which is mutated: the
    # cell and constants-count entries are removed). Deserializes the cell, constructs the class
    # with it and the remaining entries, then restores the stored constants count on the new
    # layer. The optional custom objects mapping resolves user-defined classes. Returns the layer.
    from tensorflow.python.keras.layers import deserialize as deserialize_layer  # pylint: disable=g-import-not-at-top
    cell = deserialize_layer(config.pop('cell'), custom_objects=custom_objects)
    num_constants = config.pop('num_constants', 0)
    layer = cls(cell, **config)
    layer._num_constants = num_constants
    return layer

  @property
  def _trackable_saved_model_saver(self):
    # [FUNCTION DOC]
    # HELPER: hook used by the saved-model export machinery.
    # Read-only property returning the saver object specialized for recurrent layers.
    return layer_serialization.RNNSavedModelSaver(self)


class AbstractRNNCell(Layer):
  # [CLASS DOC]
  # Abstract base class for custom RNN cells. Subclasses must implement the one-step call, the
  # state size property and the output size property. It supplies a default initial state
  # method that creates zero-filled states. Used by passing instances to the RNN layer.
  """Abstract object representing an RNN cell.

  See [the Keras RNN API guide](https://www.tensorflow.org/guide/keras/rnn)
  for details about the usage of RNN API.

  This is the base class for implementing RNN cells with custom behavior.

  Every `RNNCell` must have the properties below and implement `call` with
  the signature `(output, next_state) = call(input, state)`.

  Examples:

  ```python
    class MinimalRNNCell(AbstractRNNCell):

      def __init__(self, units, **kwargs):
        self.units = units
        super(MinimalRNNCell, self).__init__(**kwargs)

      @property
      def state_size(self):
        return self.units

      def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

      def call(self, inputs, states):
        prev_output = states[0]
        h = backend.dot(inputs, self.kernel)
        output = h + backend.dot(prev_output, self.recurrent_kernel)
        return output, output
  ```

  This definition of cell differs from the definition used in the literature.
  In the literature, 'cell' refers to an object with a single scalar output.
  This definition refers to a horizontal array of such units.

  An RNN cell, in the most abstract setting, is anything that has
  a state and performs some operation that takes a matrix of inputs.
  This operation results in an output matrix with `self.output_size` columns.
  If `self.state_size` is an integer, this operation also results in a new
  state matrix with `self.state_size` columns.  If `self.state_size` is a
  (possibly nested tuple of) TensorShape object(s), then it should return a
  matching structure of Tensors having shape `[batch_size].concatenate(s)`
  for each `s` in `self.batch_size`.
  """

  def call(self, inputs, states):
    # [FUNCTION DOC]
    # Abstract one-step computation; subclasses return a pair of output and next state. Inputs is
    # the step input tensor and states the previous state(s). This base version always raises a
    # not-implemented error.
    """The function that contains the logic for one RNN step calculation.

    Args:
      inputs: the input tensor, which is a slide from the overall RNN input by
        the time dimension (usually the second dimension).
      states: the state tensor from previous step, which has the same shape
        as `(batch, state_size)`. In the case of timestep 0, it will be the
        initial state user specified, or zero filled tensor otherwise.

    Returns:
      A tuple of two tensors:
        1. output tensor for the current timestep, with size `output_size`.
        2. state tensor for next step, which has the shape of `state_size`.
    """
    raise NotImplementedError('Abstract method')

  @property
  def state_size(self):
    # [FUNCTION DOC]
    # Abstract read-only property: size or sizes of the cell's states (integer, shape, or tuple
    # of those). This base version always raises a not-implemented error.
    """size(s) of state(s) used by this cell.

    It can be represented by an Integer, a TensorShape or a tuple of Integers
    or TensorShapes.
    """
    raise NotImplementedError('Abstract method')

  @property
  def output_size(self):
    # [FUNCTION DOC]
    # Abstract read-only property: size of the output produced at each step. This base version
    # always raises a not-implemented error.
    """Integer or TensorShape: size of outputs produced by this cell."""
    raise NotImplementedError('Abstract method')

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    # [FUNCTION DOC]
    # Default initial state: zero-filled tensors shaped from the cell's state size. Takes an
    # optional input tensor (its batch size and dtype are then used) or an explicit batch size
    # and dtype. Returns the zero state structure; raises a value error if the batch size or dtype
    # cannot be determined.
    return _generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype)


@doc_controls.do_not_generate_docs
class DropoutRNNCellMixin(object):
  # [CLASS DOC]
  # Mixin for RNN cells that adds dropout-mask creation and caching. The host cell must define
  # dropout and recurrent dropout rate attributes. Holds two context-aware caches (one for
  # input masks and one for recurrent-state masks) so the same mask is reused across the time
  # steps of one batch. The RNN layer clears them before every batch. Caches are excluded when
  # the object is copied or pickled and recreated afterwards.
  """Object that hold dropout related fields for RNN Cell.

  This class is not a standalone RNN cell. It suppose to be used with a RNN cell
  by multiple inheritance. Any cell that mix with class should have following
  fields:
    dropout: a float number within range [0, 1). The ratio that the input
      tensor need to dropout.
    recurrent_dropout: a float number within range [0, 1). The ratio that the
      recurrent state weights need to dropout.
  This object will create and cache created dropout masks, and reuse them for
  the incoming data, so that the same mask is used for every batch input.
  """

  def __init__(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Constructor. Creates the untracked mask caches first, then forwards all positional and
    # keyword arguments to the next class in the inheritance chain (the layer base class).
    self._create_non_trackable_mask_cache()
    super(DropoutRNNCellMixin, self).__init__(*args, **kwargs)

  @trackable.no_automatic_dependency_tracking
  def _create_non_trackable_mask_cache(self):
    # [FUNCTION DOC]
    # HELPER: constructor helper of the dropout mask mixin.
    # KNOWN ISSUE: the docstring talks about eager-mode mask tensors that are generated
    # differently, but the code only creates two context-keyed value caches (one per mask kind).
    # Creates the input-mask cache and the recurrent-mask cache, each keyed by execution context
    # and filled lazily by the corresponding mask-creation method. Created without dependency
    # tracking so they are not saved or deep-copied. Returns nothing.
    """Create the cache for dropout and recurrent dropout mask.

    Note that the following two masks will be used in "graph function" mode,
    e.g. these masks are symbolic tensors. In eager mode, the `eager_*_mask`
    tensors will be generated differently than in the "graph function" case,
    and they will be cached.

    Also note that in graph mode, we still cache those masks only because the
    RNN could be created with `unroll=True`. In that case, the `cell.call()`
    function will be invoked multiple times, and we want to ensure same mask
    is used every time.

    Also the caches are created without tracking. Since they are not picklable
    by python when deepcopy, we don't want `layer._obj_reference_counts_dict`
    to track it by default.
    """
    self._dropout_mask_cache = backend.ContextValueCache(
        self._create_dropout_mask)
    self._recurrent_dropout_mask_cache = backend.ContextValueCache(
        self._create_recurrent_dropout_mask)

  def reset_dropout_mask(self):
    # [FUNCTION DOC]
    # Clears the cached input dropout masks so the next batch draws fresh ones. Called by the RNN
    # layer before each forward computation. Returns nothing.
    """Reset the cached dropout masks if any.

    This is important for the RNN layer to invoke this in it `call()` method so
    that the cached mask is cleared before calling the `cell.call()`. The mask
    should be cached across the timestep within the same batch, but shouldn't
    be cached between batches. Otherwise it will introduce unreasonable bias
    against certain index of data within the batch.
    """
    self._dropout_mask_cache.clear()

  def reset_recurrent_dropout_mask(self):
    # [FUNCTION DOC]
    # Clears the cached recurrent-state dropout masks so the next batch draws fresh ones. Called
    # by the RNN layer before each forward computation. Returns nothing.
    """Reset the cached recurrent dropout masks if any.

    This is important for the RNN layer to invoke this in it call() method so
    that the cached mask is cleared before calling the cell.call(). The mask
    should be cached across the timestep within the same batch, but shouldn't
    be cached between batches. Otherwise it will introduce unreasonable bias
    against certain index of data within the batch.
    """
    self._recurrent_dropout_mask_cache.clear()

  def _create_dropout_mask(self, inputs, training, count=1):
    # [FUNCTION DOC]
    # HELPER: cache factory for input dropout masks.
    # Builds dropout mask(s) shaped like the inputs using the cell's dropout rate. Count says how
    # many independent masks to make (one per fused gate). Returns one mask or a list of masks;
    # masks are all ones when not training.
    return _generate_dropout_mask(
        array_ops.ones_like(inputs),
        self.dropout,
        training=training,
        count=count)

  def _create_recurrent_dropout_mask(self, inputs, training, count=1):
    # [FUNCTION DOC]
    # HELPER: cache factory for recurrent-state dropout masks.
    # Same as the input mask factory but uses the cell's recurrent dropout rate and is given the
    # previous state as the shape template. Returns one mask or a list of count masks.
    return _generate_dropout_mask(
        array_ops.ones_like(inputs),
        self.recurrent_dropout,
        training=training,
        count=count)

  def get_dropout_mask_for_cell(self, inputs, training, count=1):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring promises a list of masks, but the result is nothing at all when
    # the dropout rate is zero, and a single mask tensor (not a list) when count is one.
    # Returns the cached input dropout mask(s) for the current context, creating them on first
    # use. Inputs gives the mask shape, training says whether dropout is active, count is the
    # number of masks (default one). Side effect: fills the input mask cache.
    """Get the dropout mask for RNN cell's input.

    It will create mask based on context if there isn't any existing cached
    mask. If a new mask is generated, it will update the cache in the cell.

    Args:
      inputs: The input tensor whose shape will be used to generate dropout
        mask.
      training: Boolean tensor, whether its in training mode, dropout will be
        ignored in non-training mode.
      count: Int, how many dropout mask will be generated. It is useful for cell
        that has internal weights fused together.
    Returns:
      List of mask tensor, generated or cached mask based on context.
    """
    if self.dropout == 0:
      return None
    init_kwargs = dict(inputs=inputs, training=training, count=count)
    return self._dropout_mask_cache.setdefault(kwargs=init_kwargs)

  def get_recurrent_dropout_mask_for_cell(self, inputs, training, count=1):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring promises a list of masks, but the result is nothing at all when
    # the recurrent dropout rate is zero, and a single mask tensor when count is one.
    # Returns the cached recurrent-state dropout mask(s) for the current context, creating them on
    # first use. Inputs (the previous state) gives the mask shape, training says whether dropout
    # is active, count is the number of masks. Side effect: fills the recurrent mask cache.
    """Get the recurrent dropout mask for RNN cell.

    It will create mask based on context if there isn't any existing cached
    mask. If a new mask is generated, it will update the cache in the cell.

    Args:
      inputs: The input tensor whose shape will be used to generate dropout
        mask.
      training: Boolean tensor, whether its in training mode, dropout will be
        ignored in non-training mode.
      count: Int, how many dropout mask will be generated. It is useful for cell
        that has internal weights fused together.
    Returns:
      List of mask tensor, generated or cached mask based on context.
    """
    if self.recurrent_dropout == 0:
      return None
    init_kwargs = dict(inputs=inputs, training=training, count=count)
    return self._recurrent_dropout_mask_cache.setdefault(kwargs=init_kwargs)

  def __getstate__(self):
    # [FUNCTION DOC]
    # Copy and pickle support. Takes the parent's state mapping and removes both mask caches
    # (they hold tensors and graphs that cannot be pickled). Returns the reduced state mapping.
    # Used for deepcopy. The caching can't be pickled by python, since it will
    # contain tensor and graph.
    state = super(DropoutRNNCellMixin, self).__getstate__()
    state.pop('_dropout_mask_cache', None)
    state.pop('_recurrent_dropout_mask_cache', None)
    return state

  def __setstate__(self, state):
    # [FUNCTION DOC]
    # Restore support after copy or unpickling. Adds freshly created, empty mask caches to the
    # given state mapping (mutating it) and then lets the parent restore the object. Returns
    # nothing.
    state['_dropout_mask_cache'] = backend.ContextValueCache(
        self._create_dropout_mask)
    state['_recurrent_dropout_mask_cache'] = backend.ContextValueCache(
        self._create_recurrent_dropout_mask)
    super(DropoutRNNCellMixin, self).__setstate__(state)


class SimpleRNNCell(DropoutRNNCellMixin, Layer):
  # [CLASS DOC]
  # Single-step cell of a fully connected recurrent network: the new output is an activation of
  # the input times a kernel plus the previous output times a recurrent kernel plus a bias, and
  # it is also the new state. Holds the configuration (units, activation, bias flag,
  # initializers, regularizers, constraints, dropout rates), the weights created at build time
  # (kernel, recurrent kernel, optional bias) and the state and output size, both equal to units.
  # Used inside the RNN layer (or directly in a stack); dropout masks come from the mixin.
  """Cell class for SimpleRNN.

  See [the Keras RNN API guide](https://www.tensorflow.org/guide/keras/rnn)
  for details about the usage of RNN API.

  This class processes one step within the whole time sequence input, whereas
  `tf.keras.layer.SimpleRNN` processes the whole sequence.

  Args:
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use.
      Default: hyperbolic tangent (`tanh`).
      If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, (default `True`), whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix,
      used for the linear transformation of the inputs. Default:
      `glorot_uniform`.
    recurrent_initializer: Initializer for the `recurrent_kernel`
      weights matrix, used for the linear transformation of the recurrent state.
      Default: `orthogonal`.
    bias_initializer: Initializer for the bias vector. Default: `zeros`.
    kernel_regularizer: Regularizer function applied to the `kernel` weights
      matrix. Default: `None`.
    recurrent_regularizer: Regularizer function applied to the
      `recurrent_kernel` weights matrix. Default: `None`.
    bias_regularizer: Regularizer function applied to the bias vector. Default:
      `None`.
    kernel_constraint: Constraint function applied to the `kernel` weights
      matrix. Default: `None`.
    recurrent_constraint: Constraint function applied to the `recurrent_kernel`
      weights matrix. Default: `None`.
    bias_constraint: Constraint function applied to the bias vector. Default:
      `None`.
    dropout: Float between 0 and 1. Fraction of the units to drop for the linear
      transformation of the inputs. Default: 0.
    recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for
      the linear transformation of the recurrent state. Default: 0.

  Call arguments:
    inputs: A 2D tensor, with shape of `[batch, feature]`.
    states: A 2D tensor with shape of `[batch, units]`, which is the state from
      the previous time step. For timestep 0, the initial state provided by user
      will be feed to cell.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. Only relevant when `dropout` or
      `recurrent_dropout` is used.

  Examples:

  ```python
  inputs = np.random.random([32, 10, 8]).astype(np.float32)
  rnn = tf.keras.layers.RNN(tf.keras.layers.SimpleRNNCell(4))

  output = rnn(inputs)  # The output has shape `[32, 4]`.

  rnn = tf.keras.layers.RNN(
      tf.keras.layers.SimpleRNNCell(4),
      return_sequences=True,
      return_state=True)

  # whole_sequence_output has shape `[32, 10, 4]`.
  # final_state has shape `[32, 4]`.
  whole_sequence_output, final_state = rnn(inputs)
  ```
  """

  def __init__(self,
               units,
               activation='tanh',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
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
    # KNOWN ISSUE: the error message demands a positive integer, but only negative values are
    # rejected; zero units is silently accepted.
    # Constructor. Takes units (output dimensionality), activation (name or callable, default
    # tanh, nothing means linear), use_bias (default true), initializer, regularizer and
    # constraint specifications for the kernel, recurrent kernel and bias (names, configs or
    # objects, resolved through the standard lookups), and dropout and recurrent_dropout rates
    # (clipped into the zero to one range). The keyword enable_caching_device is removed from the
    # keywords and defaults to true under the v2 behavior and false otherwise. Raises a value error
    # for negative units. Sets state size and output size to units.
    if units < 0:
      raise ValueError(f'Received an invalid value for units, expected '
                       f'a positive integer, got {units}.')
    # By default use cached variable under v2 mode, see b/143699808.
    if ops.executing_eagerly_outside_functions():
      self._enable_caching_device = kwargs.pop('enable_caching_device', True)
    else:
      self._enable_caching_device = kwargs.pop('enable_caching_device', False)
    super(SimpleRNNCell, self).__init__(**kwargs)
    self.units = units
    self.activation = activations.get(activation)
    self.use_bias = use_bias

    self.kernel_initializer = initializers.get(kernel_initializer)
    self.recurrent_initializer = initializers.get(recurrent_initializer)
    self.bias_initializer = initializers.get(bias_initializer)

    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)

    self.kernel_constraint = constraints.get(kernel_constraint)
    self.recurrent_constraint = constraints.get(recurrent_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

    self.dropout = min(1., max(0., dropout))
    self.recurrent_dropout = min(1., max(0., recurrent_dropout))
    self.state_size = self.units
    self.output_size = self.units

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    # [FUNCTION DOC]
    # Creates the weights. Input_shape is the step input shape; its last entry is the feature
    # size. Adds a kernel (features by units), a recurrent kernel (units by units) and, if
    # use_bias is set, a bias vector of length units, each with the configured initializer,
    # regularizer, constraint and optional device caching. Without bias the bias attribute is
    # nothing. Marks the cell built.
    default_caching_device = _caching_device(self)
    self.kernel = self.add_weight(
        shape=(input_shape[-1], self.units),
        name='kernel',
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        caching_device=default_caching_device)
    self.recurrent_kernel = self.add_weight(
        shape=(self.units, self.units),
        name='recurrent_kernel',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint,
        caching_device=default_caching_device)
    if self.use_bias:
      self.bias = self.add_weight(
          shape=(self.units,),
          name='bias',
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          caching_device=default_caching_device)
    else:
      self.bias = None
    self.built = True

  def call(self, inputs, states, training=None):
    # [FUNCTION DOC]
    # One recurrent step. Inputs is a batch by features tensor; states is the previous output
    # (a single tensor, or a list holding it); training says whether dropout is active. Applies
    # input dropout (if configured) before the kernel product, adds the bias, applies recurrent
    # dropout to the previous output, adds its product with the recurrent kernel and applies the
    # activation. Returns the output and the new state; the state is wrapped in a list exactly when
    # the given states were a list.
    prev_output = states[0] if nest.is_nested(states) else states
    dp_mask = self.get_dropout_mask_for_cell(inputs, training)
    rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
        prev_output, training)

    if dp_mask is not None:
      h = backend.dot(inputs * dp_mask, self.kernel)
    else:
      h = backend.dot(inputs, self.kernel)
    if self.bias is not None:
      h = backend.bias_add(h, self.bias)

    if rec_dp_mask is not None:
      prev_output = prev_output * rec_dp_mask
    output = h + backend.dot(prev_output, self.recurrent_kernel)
    if self.activation is not None:
      output = self.activation(output)

    new_state = [output] if nest.is_nested(states) else output
    return output, new_state

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    # [FUNCTION DOC]
    # Returns the zero-filled initial state (one tensor of batch by units). Takes an optional
    # inputs tensor (batch size and dtype are read from it) or explicit batch size and dtype.
    # Raises a value error if batch size or dtype cannot be determined.
    return _generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the serializable configuration: all constructor settings with activation,
    # initializers, regularizers and constraints serialized, merged with the base layer config and
    # the caching-device option when it differs from its default.
    config = {
        'units':
            self.units,
        'activation':
            activations.serialize(self.activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint),
        'dropout':
            self.dropout,
        'recurrent_dropout':
            self.recurrent_dropout
    }
    config.update(_config_for_enable_caching_device(self))
    base_config = super(SimpleRNNCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class SimpleRNN(RNN):
  # [CLASS DOC]
  # Layer version of a fully connected recurrent network: an RNN layer that owns a single
  # SimpleRNNCell. Accepts the cell's configuration plus the sequence options, expects three
  # dimensional inputs (batch, time, features), and exposes the cell's settings as read-only
  # properties. Also supports an activity regularizer applied to its output.
  """Fully-connected RNN where the output is to be fed back to input.

  See [the Keras RNN API guide](https://www.tensorflow.org/guide/keras/rnn)
  for details about the usage of RNN API.

  Args:
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use.
      Default: hyperbolic tangent (`tanh`).
      If you pass None, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, (default `True`), whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix,
      used for the linear transformation of the inputs. Default:
      `glorot_uniform`.
    recurrent_initializer: Initializer for the `recurrent_kernel`
      weights matrix, used for the linear transformation of the recurrent state.
      Default: `orthogonal`.
    bias_initializer: Initializer for the bias vector. Default: `zeros`.
    kernel_regularizer: Regularizer function applied to the `kernel` weights
      matrix. Default: `None`.
    recurrent_regularizer: Regularizer function applied to the
      `recurrent_kernel` weights matrix. Default: `None`.
    bias_regularizer: Regularizer function applied to the bias vector. Default:
      `None`.
    activity_regularizer: Regularizer function applied to the output of the
      layer (its "activation"). Default: `None`.
    kernel_constraint: Constraint function applied to the `kernel` weights
      matrix. Default: `None`.
    recurrent_constraint: Constraint function applied to the `recurrent_kernel`
      weights matrix.  Default: `None`.
    bias_constraint: Constraint function applied to the bias vector. Default:
      `None`.
    dropout: Float between 0 and 1.
      Fraction of the units to drop for the linear transformation of the inputs.
      Default: 0.
    recurrent_dropout: Float between 0 and 1.
      Fraction of the units to drop for the linear transformation of the
      recurrent state. Default: 0.
    return_sequences: Boolean. Whether to return the last output
      in the output sequence, or the full sequence. Default: `False`.
    return_state: Boolean. Whether to return the last state
      in addition to the output. Default: `False`
    go_backwards: Boolean (default False).
      If True, process the input sequence backwards and return the
      reversed sequence.
    stateful: Boolean (default False). If True, the last state
      for each sample at index i in a batch will be used as initial
      state for the sample of index i in the following batch.
    unroll: Boolean (default False).
      If True, the network will be unrolled,
      else a symbolic loop will be used.
      Unrolling can speed-up a RNN,
      although it tends to be more memory-intensive.
      Unrolling is only suitable for short sequences.

  Call arguments:
    inputs: A 3D tensor, with shape `[batch, timesteps, feature]`.
    mask: Binary tensor of shape `[batch, timesteps]` indicating whether
      a given timestep should be masked. An individual `True` entry indicates
      that the corresponding timestep should be utilized, while a `False` entry
      indicates that the corresponding timestep should be ignored.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. This argument is passed to the cell
      when calling it. This is only relevant if `dropout` or
      `recurrent_dropout` is used.
    initial_state: List of initial state tensors to be passed to the first
      call of the cell.

  Examples:

  ```python
  inputs = np.random.random([32, 10, 8]).astype(np.float32)
  simple_rnn = tf.keras.layers.SimpleRNN(4)

  output = simple_rnn(inputs)  # The output has shape `[32, 4]`.

  simple_rnn = tf.keras.layers.SimpleRNN(
      4, return_sequences=True, return_state=True)

  # whole_sequence_output has shape `[32, 10, 4]`.
  # final_state has shape `[32, 4]`.
  whole_sequence_output, final_state = simple_rnn(inputs)
  ```
  """

  def __init__(self,
               units,
               activation='tanh',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               unroll=False,
               **kwargs):
    # [FUNCTION DOC]
    # Constructor. Accepts the same cell settings as the simple cell (units, activation, bias flag,
    # initializers, regularizers, constraints, dropout rates), an activity regularizer, and the
    # sequence flags return_sequences, return_state, go_backwards, stateful and unroll. Extra
    # keywords go to the base RNN layer; the deprecated implementation keyword is dropped with a
    # logged warning, and enable_caching_device is forwarded to the cell instead. Dtype and
    # trainable keywords are also copied to the cell. Creates the cell, calls the base constructor,
    # sets the activity regularizer and fixes the input spec to three dimensions.
    if 'implementation' in kwargs:
      kwargs.pop('implementation')
      logging.warning('The `implementation` argument '
                      'in `SimpleRNN` has been deprecated. '
                      'Please remove it from your layer call.')
    if 'enable_caching_device' in kwargs:
      cell_kwargs = {'enable_caching_device':
                     kwargs.pop('enable_caching_device')}
    else:
      cell_kwargs = {}
    cell = SimpleRNNCell(
        units,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=bias_regularizer,
        kernel_constraint=kernel_constraint,
        recurrent_constraint=recurrent_constraint,
        bias_constraint=bias_constraint,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        dtype=kwargs.get('dtype'),
        trainable=kwargs.get('trainable', True),
        **cell_kwargs)
    super(SimpleRNN, self).__init__(
        cell,
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        unroll=unroll,
        **kwargs)
    self.activity_regularizer = regularizers.get(activity_regularizer)
    self.input_spec = [InputSpec(ndim=3)]

  def call(self, inputs, mask=None, training=None, initial_state=None):
    # [FUNCTION DOC]
    # Forward computation; takes inputs, an optional mask, a training flag and an optional initial
    # state list, and delegates to the base RNN computation. It narrows the base signature by
    # not accepting constants. Returns the output(s) of the base version.
    return super(SimpleRNN, self).call(
        inputs, mask=mask, training=training, initial_state=initial_state)

  @property
  def units(self):
    # [FUNCTION DOC]
    # HELPER: read-only pass-through to the wrapped cell.
    # Returns the number of units (output dimensionality) of the cell.
    return self.cell.units

  @property
  def activation(self):
    # [FUNCTION DOC]
    # HELPER: read-only pass-through to the wrapped cell.
    # Returns the activation function used by the cell.
    return self.cell.activation

  @property
  def use_bias(self):
    # [FUNCTION DOC]
    # HELPER: read-only pass-through to the wrapped cell.
    # Returns whether the cell has a bias vector.
    return self.cell.use_bias

  @property
  def kernel_initializer(self):
    # [FUNCTION DOC]
    # HELPER: read-only pass-through to the wrapped cell.
    # Returns the initializer of the input kernel.
    return self.cell.kernel_initializer

  @property
  def recurrent_initializer(self):
    # [FUNCTION DOC]
    # HELPER: read-only pass-through to the wrapped cell.
    # Returns the initializer of the recurrent kernel.
    return self.cell.recurrent_initializer

  @property
  def bias_initializer(self):
    # [FUNCTION DOC]
    # HELPER: read-only pass-through to the wrapped cell.
    # Returns the initializer of the bias vector.
    return self.cell.bias_initializer

  @property
  def kernel_regularizer(self):
    # [FUNCTION DOC]
    # HELPER: read-only pass-through to the wrapped cell.
    # Returns the regularizer of the input kernel.
    return self.cell.kernel_regularizer

  @property
  def recurrent_regularizer(self):
    # [FUNCTION DOC]
    # HELPER: read-only pass-through to the wrapped cell.
    # Returns the regularizer of the recurrent kernel.
    return self.cell.recurrent_regularizer

  @property
  def bias_regularizer(self):
    # [FUNCTION DOC]
    # HELPER: read-only pass-through to the wrapped cell.
    # Returns the regularizer of the bias vector.
    return self.cell.bias_regularizer

  @property
  def kernel_constraint(self):
    # [FUNCTION DOC]
    # HELPER: read-only pass-through to the wrapped cell.
    # Returns the constraint of the input kernel.
    return self.cell.kernel_constraint

  @property
  def recurrent_constraint(self):
    # [FUNCTION DOC]
    # HELPER: read-only pass-through to the wrapped cell.
    # Returns the constraint of the recurrent kernel.
    return self.cell.recurrent_constraint

  @property
  def bias_constraint(self):
    # [FUNCTION DOC]
    # HELPER: read-only pass-through to the wrapped cell.
    # Returns the constraint of the bias vector.
    return self.cell.bias_constraint

  @property
  def dropout(self):
    # [FUNCTION DOC]
    # HELPER: read-only pass-through to the wrapped cell.
    # Returns the input dropout rate of the cell.
    return self.cell.dropout

  @property
  def recurrent_dropout(self):
    # [FUNCTION DOC]
    # HELPER: read-only pass-through to the wrapped cell.
    # Returns the recurrent-state dropout rate of the cell.
    return self.cell.recurrent_dropout

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the serializable configuration: all cell settings (serialized), the activity
    # regularizer, the caching-device option if non-default, merged with the base RNN config from
    # which the nested cell entry is removed (the settings are flattened into the layer config).
    config = {
        'units':
            self.units,
        'activation':
            activations.serialize(self.activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint),
        'dropout':
            self.dropout,
        'recurrent_dropout':
            self.recurrent_dropout
    }
    base_config = super(SimpleRNN, self).get_config()
    config.update(_config_for_enable_caching_device(self.cell))
    del base_config['cell']
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    # [FUNCTION DOC]
    # Class method rebuilding the layer from a configuration mapping. Removes the deprecated
    # implementation entry if present (mutating the mapping) and constructs the class with the
    # remaining entries as keywords. Returns the new layer.
    if 'implementation' in config:
      config.pop('implementation')
    return cls(**config)


class GRUCell(DropoutRNNCellMixin, Layer):
  # [CLASS DOC]
  # Single-step gated recurrent unit cell. Computes update and reset gates and a candidate state
  # from the input and previous state, then mixes previous and candidate states with the update
  # gate. Holds the configuration, fused kernel and recurrent kernel weights (three gate blocks
  # each) and an optional bias (single or separate input and recurrent biases when the reset
  # gate is applied after the matrix product). Supports two implementations: split per-gate
  # products (1) and fused products (otherwise). State size and output size equal units. Used
  # inside the RNN layer.
  """Cell class for the GRU layer.

  Args:
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use.
      Default: hyperbolic tangent (`tanh`).
      If you pass None, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    recurrent_activation: Activation function to use
      for the recurrent step.
      Default: hard sigmoid (`hard_sigmoid`).
      If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix,
      used for the linear transformation of the inputs.
    recurrent_initializer: Initializer for the `recurrent_kernel`
      weights matrix,
      used for the linear transformation of the recurrent state.
    bias_initializer: Initializer for the bias vector.
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
      Fraction of the units to drop for the linear transformation of the inputs.
    recurrent_dropout: Float between 0 and 1.
      Fraction of the units to drop for
      the linear transformation of the recurrent state.
    reset_after: GRU convention (whether to apply reset gate after or
      before matrix multiplication). False = "before" (default),
      True = "after" (CuDNN compatible).

  Call arguments:
    inputs: A 2D tensor.
    states: List of state tensors corresponding to the previous timestep.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. Only relevant when `dropout` or
      `recurrent_dropout` is used.
  """

  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               reset_after=False,
               **kwargs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the error message demands a positive integer, but only negative values are
    # rejected; zero units is accepted.
    # KNOWN ISSUE: the implementation keyword is read from the keywords only after the base layer
    # constructor has already received them, so the removal comes too late to matter.
    # Constructor. Takes units, activation (default tanh), recurrent_activation (default hard
    # sigmoid), use_bias, initializer, regularizer and constraint specifications, dropout rates
    # (clipped to the zero to one range), and reset_after (apply the reset gate after the matrix
    # product, default false). Keywords enable_caching_device (default depends on v2 behavior) and
    # implementation (default one) are honored; if recurrent dropout is nonzero and the
    # implementation is not one, it is forced to one with a debug log. Raises a value error for
    # negative units. Sets state size and output size to units.
    if units < 0:
      raise ValueError(f'Received an invalid value for units, expected '
                       f'a positive integer, got {units}.')
    # By default use cached variable under v2 mode, see b/143699808.
    if ops.executing_eagerly_outside_functions():
      self._enable_caching_device = kwargs.pop('enable_caching_device', True)
    else:
      self._enable_caching_device = kwargs.pop('enable_caching_device', False)
    super(GRUCell, self).__init__(**kwargs)
    self.units = units
    self.activation = activations.get(activation)
    self.recurrent_activation = activations.get(recurrent_activation)
    self.use_bias = use_bias

    self.kernel_initializer = initializers.get(kernel_initializer)
    self.recurrent_initializer = initializers.get(recurrent_initializer)
    self.bias_initializer = initializers.get(bias_initializer)

    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)

    self.kernel_constraint = constraints.get(kernel_constraint)
    self.recurrent_constraint = constraints.get(recurrent_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

    self.dropout = min(1., max(0., dropout))
    self.recurrent_dropout = min(1., max(0., recurrent_dropout))

    implementation = kwargs.pop('implementation', 1)
    if self.recurrent_dropout != 0 and implementation != 1:
      logging.debug(RECURRENT_DROPOUT_WARNING_MSG)
      self.implementation = 1
    else:
      self.implementation = implementation
    self.reset_after = reset_after
    self.state_size = self.units
    self.output_size = self.units

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    # [FUNCTION DOC]
    # Creates the weights. Input_shape's last entry is the feature size. Adds a kernel of
    # features by three times units, a recurrent kernel of units by three times units and, with
    # use_bias, a bias: a single vector of three times units, or when reset_after is set a
    # two-row matrix (input bias and recurrent bias) whose shape intentionally differs from the
    # GPU-fused implementation so saved weights can be told apart. Marks the cell built.
    input_dim = input_shape[-1]
    default_caching_device = _caching_device(self)
    self.kernel = self.add_weight(
        shape=(input_dim, self.units * 3),
        name='kernel',
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        caching_device=default_caching_device)
    self.recurrent_kernel = self.add_weight(
        shape=(self.units, self.units * 3),
        name='recurrent_kernel',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint,
        caching_device=default_caching_device)

    if self.use_bias:
      if not self.reset_after:
        bias_shape = (3 * self.units,)
      else:
        # separate biases for input and recurrent kernels
        # Note: the shape is intentionally different from CuDNNGRU biases
        # `(2 * 3 * self.units,)`, so that we can distinguish the classes
        # when loading and converting saved weights.
        bias_shape = (2, 3 * self.units)
      self.bias = self.add_weight(shape=bias_shape,
                                  name='bias',
                                  initializer=self.bias_initializer,
                                  regularizer=self.bias_regularizer,
                                  constraint=self.bias_constraint,
                                  caching_device=default_caching_device)
    else:
      self.bias = None
    self.built = True

  def call(self, inputs, states, training=None):
    # [FUNCTION DOC]
    # One GRU step. Inputs is batch by features; states is the previous hidden state (a tensor or
    # a list holding it); training says whether dropout is active. Applies input and recurrent
    # dropout masks (three per kind), then computes the update gate, reset gate and candidate
    # state, either with separate per-gate products (implementation one) or with fused products
    # (otherwise), applying the reset gate before or after the recurrent product depending on
    # reset_after, and handling the single or split bias. The new hidden state mixes the previous
    # state and candidate through the update gate. Returns the hidden state as output and as new
    # state (wrapped in a list when the given states were a list).
    h_tm1 = states[0] if nest.is_nested(states) else states  # previous memory

    dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=3)
    rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
        h_tm1, training, count=3)

    if self.use_bias:
      if not self.reset_after:
        input_bias, recurrent_bias = self.bias, None
      else:
        input_bias, recurrent_bias = array_ops_stack.unstack(self.bias)

    if self.implementation == 1:
      if 0. < self.dropout < 1.:
        inputs_z = inputs * dp_mask[0]
        inputs_r = inputs * dp_mask[1]
        inputs_h = inputs * dp_mask[2]
      else:
        inputs_z = inputs
        inputs_r = inputs
        inputs_h = inputs

      x_z = backend.dot(inputs_z, self.kernel[:, :self.units])
      x_r = backend.dot(inputs_r, self.kernel[:, self.units:self.units * 2])
      x_h = backend.dot(inputs_h, self.kernel[:, self.units * 2:])

      if self.use_bias:
        x_z = backend.bias_add(x_z, input_bias[:self.units])
        x_r = backend.bias_add(x_r, input_bias[self.units: self.units * 2])
        x_h = backend.bias_add(x_h, input_bias[self.units * 2:])

      if 0. < self.recurrent_dropout < 1.:
        h_tm1_z = h_tm1 * rec_dp_mask[0]
        h_tm1_r = h_tm1 * rec_dp_mask[1]
        h_tm1_h = h_tm1 * rec_dp_mask[2]
      else:
        h_tm1_z = h_tm1
        h_tm1_r = h_tm1
        h_tm1_h = h_tm1

      recurrent_z = backend.dot(h_tm1_z, self.recurrent_kernel[:, :self.units])
      recurrent_r = backend.dot(
          h_tm1_r, self.recurrent_kernel[:, self.units:self.units * 2])
      if self.reset_after and self.use_bias:
        recurrent_z = backend.bias_add(recurrent_z, recurrent_bias[:self.units])
        recurrent_r = backend.bias_add(
            recurrent_r, recurrent_bias[self.units:self.units * 2])

      z = self.recurrent_activation(x_z + recurrent_z)
      r = self.recurrent_activation(x_r + recurrent_r)

      # reset gate applied after/before matrix multiplication
      if self.reset_after:
        recurrent_h = backend.dot(
            h_tm1_h, self.recurrent_kernel[:, self.units * 2:])
        if self.use_bias:
          recurrent_h = backend.bias_add(
              recurrent_h, recurrent_bias[self.units * 2:])
        recurrent_h = r * recurrent_h
      else:
        recurrent_h = backend.dot(
            r * h_tm1_h, self.recurrent_kernel[:, self.units * 2:])

      hh = self.activation(x_h + recurrent_h)
    else:
      if 0. < self.dropout < 1.:
        inputs = inputs * dp_mask[0]

      # inputs projected by all gate matrices at once
      matrix_x = backend.dot(inputs, self.kernel)
      if self.use_bias:
        # biases: bias_z_i, bias_r_i, bias_h_i
        matrix_x = backend.bias_add(matrix_x, input_bias)

      x_z, x_r, x_h = array_ops.split(matrix_x, 3, axis=-1)

      if self.reset_after:
        # hidden state projected by all gate matrices at once
        matrix_inner = backend.dot(h_tm1, self.recurrent_kernel)
        if self.use_bias:
          matrix_inner = backend.bias_add(matrix_inner, recurrent_bias)
      else:
        # hidden state projected separately for update/reset and new
        matrix_inner = backend.dot(
            h_tm1, self.recurrent_kernel[:, :2 * self.units])

      recurrent_z, recurrent_r, recurrent_h = array_ops.split(
          matrix_inner, [self.units, self.units, -1], axis=-1)

      z = self.recurrent_activation(x_z + recurrent_z)
      r = self.recurrent_activation(x_r + recurrent_r)

      if self.reset_after:
        recurrent_h = r * recurrent_h
      else:
        recurrent_h = backend.dot(
            r * h_tm1, self.recurrent_kernel[:, 2 * self.units:])

      hh = self.activation(x_h + recurrent_h)
    # previous and candidate state mixed by update gate
    h = z * h_tm1 + (1 - z) * hh
    new_state = [h] if nest.is_nested(states) else h
    return h, new_state

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the serializable configuration: all constructor settings (serialized), the
    # implementation and reset_after values, the caching-device option when non-default, merged
    # with the base layer config.
    config = {
        'units': self.units,
        'activation': activations.serialize(self.activation),
        'recurrent_activation':
            activations.serialize(self.recurrent_activation),
        'use_bias': self.use_bias,
        'kernel_initializer': initializers.serialize(self.kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'bias_initializer': initializers.serialize(self.bias_initializer),
        'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
        'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        'kernel_constraint': constraints.serialize(self.kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'bias_constraint': constraints.serialize(self.bias_constraint),
        'dropout': self.dropout,
        'recurrent_dropout': self.recurrent_dropout,
        'implementation': self.implementation,
        'reset_after': self.reset_after
    }
    config.update(_config_for_enable_caching_device(self))
    base_config = super(GRUCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    # [FUNCTION DOC]
    # Returns the zero-filled initial hidden state (batch by units). Takes an optional inputs
    # tensor or an explicit batch size and dtype; raises a value error if these cannot be
    # determined.
    return _generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype)


class GRU(RNN):
  # [CLASS DOC]
  # Layer version of the gated recurrent unit: an RNN layer owning one GRU cell. Takes the cell
  # settings plus sequence options, expects three dimensional inputs, exposes cell settings as
  # read-only properties and supports an activity regularizer. Includes the reset_after option
  # for the GPU-compatible variant.
  """Gated Recurrent Unit - Cho et al. 2014.

  There are two variants. The default one is based on 1406.1078v3 and
  has reset gate applied to hidden state before matrix multiplication. The
  other one is based on original 1406.1078v1 and has the order reversed.

  The second variant is compatible with CuDNNGRU (GPU-only) and allows
  inference on CPU. Thus it has separate biases for `kernel` and
  `recurrent_kernel`. Use `'reset_after'=True` and
  `recurrent_activation='sigmoid'`.

  Args:
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use.
      Default: hyperbolic tangent (`tanh`).
      If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    recurrent_activation: Activation function to use
      for the recurrent step.
      Default: hard sigmoid (`hard_sigmoid`).
      If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix,
      used for the linear transformation of the inputs.
    recurrent_initializer: Initializer for the `recurrent_kernel`
      weights matrix, used for the linear transformation of the recurrent state.
    bias_initializer: Initializer for the bias vector.
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix.
    recurrent_regularizer: Regularizer function applied to
      the `recurrent_kernel` weights matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to
      the output of the layer (its "activation")..
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
    return_sequences: Boolean. Whether to return the last output
      in the output sequence, or the full sequence.
    return_state: Boolean. Whether to return the last state
      in addition to the output.
    go_backwards: Boolean (default False).
      If True, process the input sequence backwards and return the
      reversed sequence.
    stateful: Boolean (default False). If True, the last state
      for each sample at index i in a batch will be used as initial
      state for the sample of index i in the following batch.
    unroll: Boolean (default False).
      If True, the network will be unrolled,
      else a symbolic loop will be used.
      Unrolling can speed-up a RNN,
      although it tends to be more memory-intensive.
      Unrolling is only suitable for short sequences.
    time_major: The shape format of the `inputs` and `outputs` tensors.
      If True, the inputs and outputs will be in shape
      `(timesteps, batch, ...)`, whereas in the False case, it will be
      `(batch, timesteps, ...)`. Using `time_major = True` is a bit more
      efficient because it avoids transposes at the beginning and end of the
      RNN calculation. However, most TensorFlow data is batch-major, so by
      default this function accepts input and emits output in batch-major
      form.
    reset_after: GRU convention (whether to apply reset gate after or
      before matrix multiplication). False = "before" (default),
      True = "after" (CuDNN compatible).

  Call arguments:
    inputs: A 3D tensor.
    mask: Binary tensor of shape `(samples, timesteps)` indicating whether
      a given timestep should be masked. An individual `True` entry indicates
      that the corresponding timestep should be utilized, while a `False`
      entry indicates that the corresponding timestep should be ignored.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. This argument is passed to the cell
      when calling it. This is only relevant if `dropout` or
      `recurrent_dropout` is used.
    initial_state: List of initial state tensors to be passed to the first
      call of the cell.
  """

  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               unroll=False,
               reset_after=False,
               **kwargs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: for implementation zero the warning says it now defaults to one, but the value
    # zero is still passed to the cell; the cell only treats exactly one as the split variant, so
    # zero actually runs the fused variant.
    # Constructor. Takes the GRU cell settings (units, activations, bias flag, initializers,
    # regularizers, constraints, dropout rates, reset_after), an activity regularizer and the
    # sequence flags (return_sequences, return_state, go_backwards, stateful, unroll). The
    # keyword implementation (default one) is removed and passed to the cell, with a logged
    # warning when zero; enable_caching_device is forwarded to the cell; dtype and trainable
    # keywords are copied to the cell; other keywords go to the base RNN layer. Sets the activity
    # regularizer and fixes the input spec to three dimensions.
    implementation = kwargs.pop('implementation', 1)
    if implementation == 0:
      logging.warning('`implementation=0` has been deprecated, '
                      'and now defaults to `implementation=1`.'
                      'Please update your layer call.')
    if 'enable_caching_device' in kwargs:
      cell_kwargs = {'enable_caching_device':
                     kwargs.pop('enable_caching_device')}
    else:
      cell_kwargs = {}
    cell = GRUCell(
        units,
        activation=activation,
        recurrent_activation=recurrent_activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=bias_regularizer,
        kernel_constraint=kernel_constraint,
        recurrent_constraint=recurrent_constraint,
        bias_constraint=bias_constraint,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        implementation=implementation,
        reset_after=reset_after,
        dtype=kwargs.get('dtype'),
        trainable=kwargs.get('trainable', True),
        **cell_kwargs)
    super(GRU, self).__init__(
        cell,
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        unroll=unroll,
        **kwargs)
    self.activity_regularizer = regularizers.get(activity_regularizer)
    self.input_spec = [InputSpec(ndim=3)]

  def call(self, inputs, mask=None, training=None, initial_state=None):
    # [FUNCTION DOC]
    # Forward computation; takes inputs, an optional mask, a training flag and an optional initial
    # state list, and delegates to the base RNN computation (no constants accepted). Returns the
    # base version's output(s).
    return super(GRU, self).call(
        inputs, mask=mask, training=training, initial_state=initial_state)

  @property
  def units(self):
    # [FUNCTION DOC]
    # HELPER: read-only pass-through to the wrapped cell.
    # Returns the number of units of the cell.
    return self.cell.units

  @property
  def activation(self):
    # [FUNCTION DOC]
    # HELPER: read-only pass-through to the wrapped cell.
    # Returns the activation function of the cell.
    return self.cell.activation

  @property
  def recurrent_activation(self):
    # [FUNCTION DOC]
    # HELPER: read-only pass-through to the wrapped cell.
    # Returns the gate (recurrent) activation function of the cell.
    return self.cell.recurrent_activation

  @property
  def use_bias(self):
    # [FUNCTION DOC]
    # HELPER: read-only pass-through to the wrapped cell.
    # Returns whether the cell uses a bias.
    return self.cell.use_bias

  @property
  def kernel_initializer(self):
    # [FUNCTION DOC]
    # HELPER: read-only pass-through to the wrapped cell.
    # Returns the initializer of the input kernel.
    return self.cell.kernel_initializer

  @property
  def recurrent_initializer(self):
    # [FUNCTION DOC]
    # HELPER: read-only pass-through to the wrapped cell.
    # Returns the initializer of the recurrent kernel.
    return self.cell.recurrent_initializer

  @property
  def bias_initializer(self):
    # [FUNCTION DOC]
    # HELPER: read-only pass-through to the wrapped cell.
    # Returns the initializer of the bias.
    return self.cell.bias_initializer

  @property
  def kernel_regularizer(self):
    # [FUNCTION DOC]
    # HELPER: read-only pass-through to the wrapped cell.
    # Returns the regularizer of the input kernel.
    return self.cell.kernel_regularizer

  @property
  def recurrent_regularizer(self):
    # [FUNCTION DOC]
    # HELPER: read-only pass-through to the wrapped cell.
    # Returns the regularizer of the recurrent kernel.
    return self.cell.recurrent_regularizer

  @property
  def bias_regularizer(self):
    # [FUNCTION DOC]
    # HELPER: read-only pass-through to the wrapped cell.
    # Returns the regularizer of the bias.
    return self.cell.bias_regularizer

  @property
  def kernel_constraint(self):
    # [FUNCTION DOC]
    # HELPER: read-only pass-through to the wrapped cell.
    # Returns the constraint of the input kernel.
    return self.cell.kernel_constraint

  @property
  def recurrent_constraint(self):
    # [FUNCTION DOC]
    # HELPER: read-only pass-through to the wrapped cell.
    # Returns the constraint of the recurrent kernel.
    return self.cell.recurrent_constraint

  @property
  def bias_constraint(self):
    # [FUNCTION DOC]
    # HELPER: read-only pass-through to the wrapped cell.
    # Returns the constraint of the bias.
    return self.cell.bias_constraint

  @property
  def dropout(self):
    # [FUNCTION DOC]
    # HELPER: read-only pass-through to the wrapped cell.
    # Returns the input dropout rate of the cell.
    return self.cell.dropout

  @property
  def recurrent_dropout(self):
    # [FUNCTION DOC]
    # HELPER: read-only pass-through to the wrapped cell.
    # Returns the recurrent dropout rate of the cell.
    return self.cell.recurrent_dropout

  @property
  def implementation(self):
    # [FUNCTION DOC]
    # HELPER: read-only pass-through to the wrapped cell.
    # Returns the implementation mode actually used by the cell.
    return self.cell.implementation

  @property
  def reset_after(self):
    # [FUNCTION DOC]
    # HELPER: read-only pass-through to the wrapped cell.
    # Returns whether the reset gate is applied after the matrix product.
    return self.cell.reset_after

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the serializable configuration: all cell settings (serialized), activity
    # regularizer, implementation, reset_after and the caching-device option if non-default,
    # merged with the base RNN config minus its nested cell entry.
    config = {
        'units':
            self.units,
        'activation':
            activations.serialize(self.activation),
        'recurrent_activation':
            activations.serialize(self.recurrent_activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint),
        'dropout':
            self.dropout,
        'recurrent_dropout':
            self.recurrent_dropout,
        'implementation':
            self.implementation,
        'reset_after':
            self.reset_after
    }
    config.update(_config_for_enable_caching_device(self.cell))
    base_config = super(GRU, self).get_config()
    del base_config['cell']
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    # [FUNCTION DOC]
    # Class method rebuilding the layer from a configuration mapping. A stored implementation of
    # zero is upgraded to one (mutating the mapping) before constructing with all entries as
    # keywords. Returns the new layer.
    if 'implementation' in config and config['implementation'] == 0:
      config['implementation'] = 1
    return cls(**config)


class LSTMCell(DropoutRNNCellMixin, Layer):
  # [CLASS DOC]
  # Single-step long short-term memory cell. Uses input, forget, cell and output gates computed
  # from the input and previous hidden state, and keeps two states (hidden and carry). Holds the
  # configuration, a fused kernel and recurrent kernel (four gate blocks each) and an optional
  # bias that can start the forget-gate part at one. Supports split per-gate products
  # (implementation one) and fused products. State size is two entries of units; output size is
  # units. Used inside the RNN layer; PeepholeLSTMCell extends it.
  """Cell class for the LSTM layer.

  Args:
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use.
      Default: hyperbolic tangent (`tanh`).
      If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    recurrent_activation: Activation function to use
      for the recurrent step.
      Default: hard sigmoid (`hard_sigmoid`).
      If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix,
      used for the linear transformation of the inputs.
    recurrent_initializer: Initializer for the `recurrent_kernel`
      weights matrix,
      used for the linear transformation of the recurrent state.
    bias_initializer: Initializer for the bias vector.
    unit_forget_bias: Boolean.
      If True, add 1 to the bias of the forget gate at initialization.
      Setting it to true will also force `bias_initializer="zeros"`.
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
    inputs: A 2D tensor.
    states: List of state tensors corresponding to the previous timestep.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. Only relevant when `dropout` or
      `recurrent_dropout` is used.
  """

  def __init__(self,
               units,
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
    # KNOWN ISSUE: the error message demands a positive integer, but only negative values are
    # rejected; zero units is accepted.
    # KNOWN ISSUE: the implementation keyword is read from the keywords only after the base layer
    # constructor has already received them, so the removal comes too late to matter.
    # Constructor. Takes units, activation (default tanh), recurrent_activation (default hard
    # sigmoid), use_bias, initializer, regularizer and constraint specifications, unit_forget_bias
    # (default true), and dropout rates (clipped to zero to one). Keywords enable_caching_device
    # and implementation (default one) are honored; recurrent dropout with an implementation other
    # than one forces one with a debug log. Raises a value error for negative units. State size is
    # a two-element list of units (hidden and carry); output size is units.
    if units < 0:
      raise ValueError(f'Received an invalid value for units, expected '
                       f'a positive integer, got {units}.')
    # By default use cached variable under v2 mode, see b/143699808.
    if ops.executing_eagerly_outside_functions():
      self._enable_caching_device = kwargs.pop('enable_caching_device', True)
    else:
      self._enable_caching_device = kwargs.pop('enable_caching_device', False)
    super(LSTMCell, self).__init__(**kwargs)
    self.units = units
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
    implementation = kwargs.pop('implementation', 1)
    if self.recurrent_dropout != 0 and implementation != 1:
      logging.debug(RECURRENT_DROPOUT_WARNING_MSG)
      self.implementation = 1
    else:
      self.implementation = implementation
    self.state_size = [self.units, self.units]
    self.output_size = self.units

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the class docstring says unit_forget_bias also forces the bias initializer to
    # zeros; the code instead keeps the configured initializer for the non-forget gate blocks and
    # only fills the forget gate block with ones.
    # Creates the weights. Input_shape's last entry is the feature size. Adds a kernel (features
    # by four times units), a recurrent kernel (units by four times units) and, with use_bias, a
    # bias of four times units. With unit_forget_bias the bias initializer is a composite that
    # concatenates the configured initializer for the input gate block, ones for the forget gate
    # block, and the configured initializer for the remaining two blocks. Without bias the bias
    # attribute is nothing. Marks the cell built.
    default_caching_device = _caching_device(self)
    input_dim = input_shape[-1]
    self.kernel = self.add_weight(
        shape=(input_dim, self.units * 4),
        name='kernel',
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        caching_device=default_caching_device)
    self.recurrent_kernel = self.add_weight(
        shape=(self.units, self.units * 4),
        name='recurrent_kernel',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint,
        caching_device=default_caching_device)

    if self.use_bias:
      if self.unit_forget_bias:

        def bias_initializer(_, *args, **kwargs):
          # [FUNCTION DOC]
          # HELPER: composite initializer built inside the build method.
          # Ignores the requested shape and returns a vector made of the configured initializer's
          # values for one units-sized block, ones for the forget-gate block, then the configured
          # initializer's values for a two-units-sized block. Extra arguments are forwarded to
          # each initializer.
          return backend.concatenate([
              self.bias_initializer((self.units,), *args, **kwargs),
              initializers.get('ones')((self.units,), *args, **kwargs),
              self.bias_initializer((self.units * 2,), *args, **kwargs),
          ])
      else:
        bias_initializer = self.bias_initializer
      self.bias = self.add_weight(
          shape=(self.units * 4,),
          name='bias',
          initializer=bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          caching_device=default_caching_device)
    else:
      self.bias = None
    self.built = True

  def _compute_carry_and_output(self, x, h_tm1, c_tm1):
    # [FUNCTION DOC]
    # HELPER: gate arithmetic for the split-kernel implementation of the step computation.
    # Takes the four input projections (input, forget, cell, output), the four hidden-state
    # variants (possibly dropout-masked) and the previous carry. Adds each recurrent projection,
    # applies the gate activations and computes the new carry (forget gate times old carry plus
    # input gate times candidate). Returns the new carry and the output gate value.
    """Computes carry and output using split kernels."""
    x_i, x_f, x_c, x_o = x
    h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1
    i = self.recurrent_activation(
        x_i + backend.dot(h_tm1_i, self.recurrent_kernel[:, :self.units]))
    f = self.recurrent_activation(x_f + backend.dot(
        h_tm1_f, self.recurrent_kernel[:, self.units:self.units * 2]))
    c = f * c_tm1 + i * self.activation(x_c + backend.dot(
        h_tm1_c, self.recurrent_kernel[:, self.units * 2:self.units * 3]))
    o = self.recurrent_activation(
        x_o + backend.dot(h_tm1_o, self.recurrent_kernel[:, self.units * 3:]))
    return c, o

  def _compute_carry_and_output_fused(self, z, c_tm1):
    # [FUNCTION DOC]
    # HELPER: gate arithmetic for the fused-kernel implementation of the step computation.
    # Takes the four already-combined gate pre-activations and the previous carry, applies the
    # gate activations, and computes the new carry. Returns the new carry and the output gate.
    """Computes carry and output using fused kernels."""
    z0, z1, z2, z3 = z
    i = self.recurrent_activation(z0)
    f = self.recurrent_activation(z1)
    c = f * c_tm1 + i * self.activation(z2)
    o = self.recurrent_activation(z3)
    return c, o

  def call(self, inputs, states, training=None):
    # [FUNCTION DOC]
    # One LSTM step. Inputs is batch by features; states is a two-element sequence of previous
    # hidden state and previous carry; training says whether dropout is active. Applies input and
    # recurrent dropout masks (four each), then for implementation one splits the kernel and bias
    # into gate blocks and computes projections separately, otherwise uses one fused product for
    # the kernel and recurrent kernel and splits the result. Computes new carry and output gate and
    # then the new hidden state as output gate times the activated carry. Returns the hidden state
    # as output and a two-element list of new hidden state and new carry.
    h_tm1 = states[0]  # previous memory state
    c_tm1 = states[1]  # previous carry state

    dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
    rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
        h_tm1, training, count=4)

    if self.implementation == 1:
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
      k_i, k_f, k_c, k_o = array_ops.split(
          self.kernel, num_or_size_splits=4, axis=1)
      x_i = backend.dot(inputs_i, k_i)
      x_f = backend.dot(inputs_f, k_f)
      x_c = backend.dot(inputs_c, k_c)
      x_o = backend.dot(inputs_o, k_o)
      if self.use_bias:
        b_i, b_f, b_c, b_o = array_ops.split(
            self.bias, num_or_size_splits=4, axis=0)
        x_i = backend.bias_add(x_i, b_i)
        x_f = backend.bias_add(x_f, b_f)
        x_c = backend.bias_add(x_c, b_c)
        x_o = backend.bias_add(x_o, b_o)

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
      x = (x_i, x_f, x_c, x_o)
      h_tm1 = (h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o)
      c, o = self._compute_carry_and_output(x, h_tm1, c_tm1)
    else:
      if 0. < self.dropout < 1.:
        inputs = inputs * dp_mask[0]
      z = backend.dot(inputs, self.kernel)
      z += backend.dot(h_tm1, self.recurrent_kernel)
      if self.use_bias:
        z = backend.bias_add(z, self.bias)

      z = array_ops.split(z, num_or_size_splits=4, axis=1)
      c, o = self._compute_carry_and_output_fused(z, c_tm1)

    h = o * self.activation(c)
    return h, [h, c]

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the serializable configuration: all constructor settings (serialized), including
    # unit_forget_bias and implementation, plus the caching-device option when non-default,
    # merged with the base layer config.
    config = {
        'units':
            self.units,
        'activation':
            activations.serialize(self.activation),
        'recurrent_activation':
            activations.serialize(self.recurrent_activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'unit_forget_bias':
            self.unit_forget_bias,
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint),
        'dropout':
            self.dropout,
        'recurrent_dropout':
            self.recurrent_dropout,
        'implementation':
            self.implementation
    }
    config.update(_config_for_enable_caching_device(self))
    base_config = super(LSTMCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    # [FUNCTION DOC]
    # Returns the zero-filled initial states as a list of two tensors (hidden and carry), each of
    # batch by units. Takes an optional inputs tensor or explicit batch size and dtype; raises a
    # value error if these cannot be determined.
    return list(_generate_zero_filled_state_for_cell(
        self, inputs, batch_size, dtype))


class PeepholeLSTMCell(LSTMCell):
  # [CLASS DOC]
  # LSTM cell variant with peephole connections: the input, forget and output gates also see the
  # carry state through additional per-unit weight vectors. Subclasses the standard LSTM cell,
  # reusing its configuration and kernels, adds three peephole weight vectors at build time and
  # overrides the two gate-arithmetic helpers. Deprecated; emits a warning when created.
  """Equivalent to LSTMCell class but adds peephole connections.

  Peephole connections allow the gates to utilize the previous internal state as
  well as the previous hidden state (which is what LSTMCell is limited to).
  This allows PeepholeLSTMCell to better learn precise timings over LSTMCell.

  From [Gers et al., 2002](
    http://www.jmlr.org/papers/volume3/gers02a/gers02a.pdf):

  "We find that LSTM augmented by 'peephole connections' from its internal
  cells to its multiplicative gates can learn the fine distinction between
  sequences of spikes spaced either 50 or 49 time steps apart without the help
  of any short training exemplars."

  The peephole implementation is based on:

  [Sak et al., 2014](https://research.google.com/pubs/archive/43905.pdf)

  Example:

  ```python
  # Create 2 PeepholeLSTMCells
  peephole_lstm_cells = [PeepholeLSTMCell(size) for size in [128, 256]]
  # Create a layer composed sequentially of the peephole LSTM cells.
  layer = RNN(peephole_lstm_cells)
  input = keras.Input((timesteps, input_dim))
  output = layer(input)
  ```
  """

  def __init__(self,
               units,
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
    # Constructor. Takes the same settings as the standard LSTM cell (units, activations, bias
    # flag, initializers, unit_forget_bias, regularizers, constraints, dropout rates) plus
    # keywords. Emits a deprecation warning pointing to the add-on package replacement, then
    # delegates to the parent constructor, passing implementation (default one) taken from the
    # keywords.
    warnings.warn('`tf.keras.experimental.PeepholeLSTMCell` is deprecated '
                  'and will be removed in a future version. '
                  'Please use tensorflow_addons.rnn.PeepholeLSTMCell '
                  'instead.')
    super(PeepholeLSTMCell, self).__init__(
        units=units,
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
        implementation=kwargs.pop('implementation', 1),
        **kwargs)

  def build(self, input_shape):
    # [FUNCTION DOC]
    # Builds the parent's weights, then adds three vectors of length units (input gate, forget
    # gate and output gate peephole weights), each initialized with the kernel initializer and
    # without regularizer, constraint or caching device. Input_shape is the step input shape.
    super(PeepholeLSTMCell, self).build(input_shape)
    # The following are the weight matrices for the peephole connections. These
    # are multiplied with the previous internal state during the computation of
    # carry and output.
    self.input_gate_peephole_weights = self.add_weight(
        shape=(self.units,),
        name='input_gate_peephole_weights',
        initializer=self.kernel_initializer)
    self.forget_gate_peephole_weights = self.add_weight(
        shape=(self.units,),
        name='forget_gate_peephole_weights',
        initializer=self.kernel_initializer)
    self.output_gate_peephole_weights = self.add_weight(
        shape=(self.units,),
        name='output_gate_peephole_weights',
        initializer=self.kernel_initializer)

  def _compute_carry_and_output(self, x, h_tm1, c_tm1):
    # [FUNCTION DOC]
    # HELPER: peephole version of the split-kernel gate arithmetic.
    # Same inputs and outputs as the parent version, but the input and forget gates add their
    # peephole weights times the previous carry, and the output gate adds its peephole weights
    # times the newly computed carry. Returns the new carry and output gate value.
    x_i, x_f, x_c, x_o = x
    h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1
    i = self.recurrent_activation(
        x_i + backend.dot(h_tm1_i, self.recurrent_kernel[:, :self.units]) +
        self.input_gate_peephole_weights * c_tm1)
    f = self.recurrent_activation(x_f + backend.dot(
        h_tm1_f, self.recurrent_kernel[:, self.units:self.units * 2]) +
                                  self.forget_gate_peephole_weights * c_tm1)
    c = f * c_tm1 + i * self.activation(x_c + backend.dot(
        h_tm1_c, self.recurrent_kernel[:, self.units * 2:self.units * 3]))
    o = self.recurrent_activation(
        x_o + backend.dot(h_tm1_o, self.recurrent_kernel[:, self.units * 3:]) +
        self.output_gate_peephole_weights * c)
    return c, o

  def _compute_carry_and_output_fused(self, z, c_tm1):
    # [FUNCTION DOC]
    # HELPER: peephole version of the fused-kernel gate arithmetic.
    # Takes four combined gate pre-activations and the previous carry; the input and forget gates
    # add peephole weights times the previous carry, the output gate adds its peephole weights
    # times the new carry. Returns the new carry and output gate value.
    z0, z1, z2, z3 = z
    i = self.recurrent_activation(z0 +
                                  self.input_gate_peephole_weights * c_tm1)
    f = self.recurrent_activation(z1 +
                                  self.forget_gate_peephole_weights * c_tm1)
    c = f * c_tm1 + i * self.activation(z2)
    o = self.recurrent_activation(z3 + self.output_gate_peephole_weights * c)
    return c, o


class LSTM(RNN):
  # [CLASS DOC]
  # Layer version of the long short-term memory network: an RNN layer owning one LSTM cell. Takes
  # the cell settings plus sequence options, expects three dimensional inputs, exposes cell
  # settings as read-only properties and supports an activity regularizer. Not optimized for GPU.
  """Long Short-Term Memory layer - Hochreiter 1997.

   Note that this cell is not optimized for performance on GPU. Please use
  `tf.compat.v1.keras.layers.CuDNNLSTM` for better performance on GPU.

  Args:
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use.
      Default: hyperbolic tangent (`tanh`).
      If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    recurrent_activation: Activation function to use
      for the recurrent step.
      Default: hard sigmoid (`hard_sigmoid`).
      If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix,
      used for the linear transformation of the inputs..
    recurrent_initializer: Initializer for the `recurrent_kernel`
      weights matrix,
      used for the linear transformation of the recurrent state.
    bias_initializer: Initializer for the bias vector.
    unit_forget_bias: Boolean.
      If True, add 1 to the bias of the forget gate at initialization.
      Setting it to true will also force `bias_initializer="zeros"`.
      This is recommended in [Jozefowicz et al., 2015](
        http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf).
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix.
    recurrent_regularizer: Regularizer function applied to
      the `recurrent_kernel` weights matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to
      the output of the layer (its "activation").
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
    unroll: Boolean (default False).
      If True, the network will be unrolled,
      else a symbolic loop will be used.
      Unrolling can speed-up a RNN,
      although it tends to be more memory-intensive.
      Unrolling is only suitable for short sequences.
    time_major: The shape format of the `inputs` and `outputs` tensors.
      If True, the inputs and outputs will be in shape
      `(timesteps, batch, ...)`, whereas in the False case, it will be
      `(batch, timesteps, ...)`. Using `time_major = True` is a bit more
      efficient because it avoids transposes at the beginning and end of the
      RNN calculation. However, most TensorFlow data is batch-major, so by
      default this function accepts input and emits output in batch-major
      form.

  Call arguments:
    inputs: A 3D tensor.
    mask: Binary tensor of shape `(samples, timesteps)` indicating whether
      a given timestep should be masked. An individual `True` entry indicates
      that the corresponding timestep should be utilized, while a `False`
      entry indicates that the corresponding timestep should be ignored.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. This argument is passed to the cell
      when calling it. This is only relevant if `dropout` or
      `recurrent_dropout` is used.
    initial_state: List of initial state tensors to be passed to the first
      call of the cell.
  """

  def __init__(self,
               units,
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
               dropout=0.,
               recurrent_dropout=0.,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               unroll=False,
               **kwargs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: for implementation zero the warning says it now defaults to one, but the value
    # zero is still passed to the cell; the cell only treats exactly one as the split variant, so
    # zero actually runs the fused variant.
    # Constructor. Takes the LSTM cell settings (units, activations, bias flag, initializers,
    # unit_forget_bias, regularizers, constraints, dropout rates), an activity regularizer and the
    # sequence flags (return_sequences, return_state, go_backwards, stateful, unroll). The
    # keyword implementation (default one) is removed and passed to the cell, with a logged
    # warning when zero; enable_caching_device is forwarded to the cell; dtype and trainable
    # keywords are copied to the cell; other keywords go to the base RNN layer. Sets the activity
    # regularizer and fixes the input spec to three dimensions.
    implementation = kwargs.pop('implementation', 1)
    if implementation == 0:
      logging.warning('`implementation=0` has been deprecated, '
                      'and now defaults to `implementation=1`.'
                      'Please update your layer call.')
    if 'enable_caching_device' in kwargs:
      cell_kwargs = {'enable_caching_device':
                     kwargs.pop('enable_caching_device')}
    else:
      cell_kwargs = {}
    cell = LSTMCell(
        units,
        activation=activation,
        recurrent_activation=recurrent_activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        unit_forget_bias=unit_forget_bias,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=bias_regularizer,
        kernel_constraint=kernel_constraint,
        recurrent_constraint=recurrent_constraint,
        bias_constraint=bias_constraint,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        implementation=implementation,
        dtype=kwargs.get('dtype'),
        trainable=kwargs.get('trainable', True),
        **cell_kwargs)
    super(LSTM, self).__init__(
        cell,
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        unroll=unroll,
        **kwargs)
    self.activity_regularizer = regularizers.get(activity_regularizer)
    self.input_spec = [InputSpec(ndim=3)]

  def call(self, inputs, mask=None, training=None, initial_state=None):
    # [FUNCTION DOC]
    # Forward computation; takes inputs, an optional mask, a training flag and an optional initial
    # state list, and delegates to the base RNN computation (no constants accepted). Returns the
    # base version's output(s).
    return super(LSTM, self).call(
        inputs, mask=mask, training=training, initial_state=initial_state)

  @property
  def units(self):
    # [FUNCTION DOC]
    # HELPER: read-only pass-through to the wrapped cell.
    # Returns the number of units of the cell.
    return self.cell.units

  @property
  def activation(self):
    # [FUNCTION DOC]
    # HELPER: read-only pass-through to the wrapped cell.
    # Returns the activation function of the cell.
    return self.cell.activation

  @property
  def recurrent_activation(self):
    # [FUNCTION DOC]
    # HELPER: read-only pass-through to the wrapped cell.
    # Returns the gate (recurrent) activation function of the cell.
    return self.cell.recurrent_activation

  @property
  def use_bias(self):
    # [FUNCTION DOC]
    # HELPER: read-only pass-through to the wrapped cell.
    # Returns whether the cell uses a bias.
    return self.cell.use_bias

  @property
  def kernel_initializer(self):
    # [FUNCTION DOC]
    # HELPER: read-only pass-through to the wrapped cell.
    # Returns the initializer of the input kernel.
    return self.cell.kernel_initializer

  @property
  def recurrent_initializer(self):
    # [FUNCTION DOC]
    # HELPER: read-only pass-through to the wrapped cell.
    # Returns the initializer of the recurrent kernel.
    return self.cell.recurrent_initializer

  @property
  def bias_initializer(self):
    # [FUNCTION DOC]
    # HELPER: read-only pass-through to the wrapped cell.
    # Returns the initializer of the bias.
    return self.cell.bias_initializer

  @property
  def unit_forget_bias(self):
    # [FUNCTION DOC]
    # HELPER: read-only pass-through to the wrapped cell.
    # Returns whether the forget-gate bias block starts at one.
    return self.cell.unit_forget_bias

  @property
  def kernel_regularizer(self):
    # [FUNCTION DOC]
    # HELPER: read-only pass-through to the wrapped cell.
    # Returns the regularizer of the input kernel.
    return self.cell.kernel_regularizer

  @property
  def recurrent_regularizer(self):
    # [FUNCTION DOC]
    # HELPER: read-only pass-through to the wrapped cell.
    # Returns the regularizer of the recurrent kernel.
    return self.cell.recurrent_regularizer

  @property
  def bias_regularizer(self):
    # [FUNCTION DOC]
    # HELPER: read-only pass-through to the wrapped cell.
    # Returns the regularizer of the bias.
    return self.cell.bias_regularizer

  @property
  def kernel_constraint(self):
    # [FUNCTION DOC]
    # HELPER: read-only pass-through to the wrapped cell.
    # Returns the constraint of the input kernel.
    return self.cell.kernel_constraint

  @property
  def recurrent_constraint(self):
    # [FUNCTION DOC]
    # HELPER: read-only pass-through to the wrapped cell.
    # Returns the constraint of the recurrent kernel.
    return self.cell.recurrent_constraint

  @property
  def bias_constraint(self):
    # [FUNCTION DOC]
    # HELPER: read-only pass-through to the wrapped cell.
    # Returns the constraint of the bias.
    return self.cell.bias_constraint

  @property
  def dropout(self):
    # [FUNCTION DOC]
    # HELPER: read-only pass-through to the wrapped cell.
    # Returns the input dropout rate of the cell.
    return self.cell.dropout

  @property
  def recurrent_dropout(self):
    # [FUNCTION DOC]
    # HELPER: read-only pass-through to the wrapped cell.
    # Returns the recurrent dropout rate of the cell.
    return self.cell.recurrent_dropout

  @property
  def implementation(self):
    # [FUNCTION DOC]
    # HELPER: read-only pass-through to the wrapped cell.
    # Returns the implementation mode actually used by the cell.
    return self.cell.implementation

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the serializable configuration: all cell settings (serialized), activity
    # regularizer, unit_forget_bias, implementation and the caching-device option if non-default,
    # merged with the base RNN config minus its nested cell entry.
    config = {
        'units':
            self.units,
        'activation':
            activations.serialize(self.activation),
        'recurrent_activation':
            activations.serialize(self.recurrent_activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'unit_forget_bias':
            self.unit_forget_bias,
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint),
        'dropout':
            self.dropout,
        'recurrent_dropout':
            self.recurrent_dropout,
        'implementation':
            self.implementation
    }
    config.update(_config_for_enable_caching_device(self.cell))
    base_config = super(LSTM, self).get_config()
    del base_config['cell']
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    # [FUNCTION DOC]
    # Class method rebuilding the layer from a configuration mapping. A stored implementation of
    # zero is upgraded to one (mutating the mapping) before constructing with all entries as
    # keywords. Returns the new layer.
    if 'implementation' in config and config['implementation'] == 0:
      config['implementation'] = 1
    return cls(**config)


def _generate_dropout_mask(ones, rate, training=None, count=1):
  # [FUNCTION DOC]
  # HELPER: creates dropout masks for the dropout mixin of RNN cells.
  # Takes a tensor of ones with the target shape, the dropout rate, an optional training flag (a
  # boolean or tensor; when not training the masks are just the ones) and a count of masks.
  # Returns a single mask, or a list of count independently drawn masks when count exceeds one.
  def dropped_inputs():
    # [FUNCTION DOC]
    # HELPER: nested helper that draws one dropout mask by applying dropout to the ones tensor
    # with the captured rate. Takes nothing, returns the mask tensor.
    return backend.dropout(ones, rate)

  if count > 1:
    return [
        backend.in_train_phase(dropped_inputs, ones, training=training)
        for _ in range(count)
    ]
  return backend.in_train_phase(dropped_inputs, ones, training=training)


def _standardize_args(inputs, initial_state, constants, num_constants):
  # [FUNCTION DOC]
  # HELPER: argument normalization for the RNN layer's entry point.
  # KNOWN ISSUE: when the inputs are a list, the function asserts (raising an assertion error
  # rather than a descriptive value error) that initial_state and constants are both absent, and
  # the docstring's parameter name for the constants count is misspelled.
  # Takes inputs (tensor or list possibly packing initial states and constants), initial_state,
  # constants and the expected number of constants. When inputs is a list, the trailing
  # constants and the states after the first element are split out; a single remaining input is
  # unwrapped, several become a tuple. Then initial_state and constants are normalized to lists
  # (nothing stays nothing, a tuple becomes a list, a single item is wrapped). Returns the triple
  # of inputs, initial_state and constants.
  """Standardizes `__call__` to a single list of tensor inputs.

  When running a model loaded from a file, the input tensors
  `initial_state` and `constants` can be passed to `RNN.__call__()` as part
  of `inputs` instead of by the dedicated keyword arguments. This method
  makes sure the arguments are separated and that `initial_state` and
  `constants` are lists of tensors (or None).

  Args:
    inputs: Tensor or list/tuple of tensors. which may include constants
      and initial states. In that case `num_constant` must be specified.
    initial_state: Tensor or list of tensors or None, initial states.
    constants: Tensor or list of tensors or None, constant tensors.
    num_constants: Expected number of constants (if constants are passed as
      part of the `inputs` list.

  Returns:
    inputs: Single tensor or tuple of tensors.
    initial_state: List of tensors or None.
    constants: List of tensors or None.
  """
  if isinstance(inputs, list):
    # There are several situations here:
    # In the graph mode, __call__ will be only called once. The initial_state
    # and constants could be in inputs (from file loading).
    # In the eager mode, __call__ will be called twice, once during
    # rnn_layer(inputs=input_t, constants=c_t, ...), and second time will be
    # model.fit/train_on_batch/predict with real np data. In the second case,
    # the inputs will contain initial_state and constants as eager tensor.
    #
    # For either case, the real input is the first item in the list, which
    # could be a nested structure itself. Then followed by initial_states, which
    # could be a list of items, or list of list if the initial_state is complex
    # structure, and finally followed by constants which is a flat list.
    assert initial_state is None and constants is None
    if num_constants:
      constants = inputs[-num_constants:]
      inputs = inputs[:-num_constants]
    if len(inputs) > 1:
      initial_state = inputs[1:]
      inputs = inputs[:1]

    if len(inputs) > 1:
      inputs = tuple(inputs)
    else:
      inputs = inputs[0]

  def to_list_or_none(x):
    # [FUNCTION DOC]
    # HELPER: nested helper of the argument normalization.
    # Returns nothing for nothing, lists unchanged, a list copy of a tuple, and any other value
    # wrapped in a one-element list.
    if x is None or isinstance(x, list):
      return x
    if isinstance(x, tuple):
      return list(x)
    return [x]

  initial_state = to_list_or_none(initial_state)
  constants = to_list_or_none(constants)

  return inputs, initial_state, constants


def _is_multiple_state(state_size):
  # [FUNCTION DOC]
  # HELPER: shared by the RNN layer and the stacked cells.
  # Returns true when the given state size is a sequence-like object with a length (list or
  # tuple of sizes) and not a single shape object; false for an integer or a shape.
  """Check whether the state_size contains multiple states."""
  return (hasattr(state_size, '__len__') and
          not isinstance(state_size, tensor_shape.TensorShape))


def _generate_zero_filled_state_for_cell(cell, inputs, batch_size, dtype):
  # [FUNCTION DOC]
  # HELPER: default initial-state creation shared by the cells.
  # Takes a cell, an optional inputs tensor, a batch size and a dtype. If inputs is given, its
  # leading dimension and dtype override the batch size and dtype arguments. Returns zero-filled
  # states shaped from the cell's state size; raises a value error if batch size or dtype remain
  # unknown.
  if inputs is not None:
    batch_size = array_ops.shape(inputs)[0]
    dtype = inputs.dtype
  return _generate_zero_filled_state(batch_size, cell.state_size, dtype)


def _generate_zero_filled_state(batch_size_tensor, state_size, dtype):
  # [FUNCTION DOC]
  # HELPER: zero-state factory used by the layer and the cells.
  # Takes the batch size (scalar or tensor), a state size (integer, shape, or nested structure of
  # them) and a dtype. Raises a value error if the batch size or dtype is missing. Returns zero
  # tensors of batch by state shape, with the same nesting as the state size.
  """Generate a zero filled tensor with shape [batch_size, state_size]."""
  if batch_size_tensor is None or dtype is None:
    raise ValueError(
        'batch_size and dtype cannot be None while constructing initial state: '
        'batch_size={}, dtype={}'.format(batch_size_tensor, dtype))

  def create_zeros(unnested_state_size):
    # [FUNCTION DOC]
    # HELPER: nested helper of the zero-state factory.
    # Builds one zero tensor whose shape is the captured batch size followed by the given
    # unnested state size, using the captured dtype.
    flat_dims = tensor_shape.TensorShape(unnested_state_size).as_list()
    init_state_size = [batch_size_tensor] + flat_dims
    return array_ops.zeros(init_state_size, dtype=dtype)

  if nest.is_nested(state_size):
    return nest.map_structure(create_zeros, state_size)
  else:
    return create_zeros(state_size)


def _caching_device(rnn_cell):
  # [FUNCTION DOC]
  # HELPER: weight-creation helper for the built-in cells.
  # KNOWN ISSUE: the docstring lists only the argument, but the function returns either nothing or
  # a callable that maps an operation to its device, and it also logs warnings in two cases.
  # Decides whether variable reads should be cached on the reading device. Takes a cell. Returns
  # nothing (no caching) when running eagerly, when the cell has caching disabled, when the graph
  # is inside a while loop (warning logged), or when the compute and variable dtypes differ under
  # mixed precision (warning logged). Otherwise returns a callable that gives the device of an
  # operation.
  """Returns the caching device for the RNN variable.

  This is useful for distributed training, when variable is not located as same
  device as the training worker. By enabling the device cache, this allows
  worker to read the variable once and cache locally, rather than read it every
  time step from remote when it is needed.

  Note that this is assuming the variable that cell needs for each time step is
  having the same value in the forward path, and only gets updated in the
  backprop. It is true for all the default cells (SimpleRNN, GRU, LSTM). If the
  cell body relies on any variable that gets updated every time step, then
  caching device will cause it to read the stall value.

  Args:
    rnn_cell: the rnn cell instance.
  """
  if context.executing_eagerly():
    # caching_device is not supported in eager mode.
    return None
  if not getattr(rnn_cell, '_enable_caching_device', False):
    return None
  # Don't set a caching device when running in a loop, since it is possible that
  # train steps could be wrapped in a tf.while_loop. In that scenario caching
  # prevents forward computations in loop iterations from re-reading the
  # updated weights.
  if control_flow_util.IsInWhileLoop(ops.get_default_graph()):
    logging.warning(
        'Variable read device caching has been disabled because the '
        'RNN is in tf.while_loop loop context, which will cause '
        'reading stalled value in forward path. This could slow down '
        'the training due to duplicated variable reads. Please '
        'consider updating your code to remove tf.while_loop if possible.')
    return None
  if (rnn_cell._dtype_policy.compute_dtype !=
      rnn_cell._dtype_policy.variable_dtype):
    logging.warning(
        'Variable read device caching has been disabled since it '
        'doesn\'t work with the mixed precision API. This is '
        'likely to cause a slowdown for RNN training due to '
        'duplicated read of variable for each timestep, which '
        'will be significant in a multi remote worker setting. '
        'Please consider disabling mixed precision API if '
        'the performance has been affected.')
    return None
  # Cache the value on the device that access the variable.
  return lambda op: op.device


def _config_for_enable_caching_device(rnn_cell):
  # [FUNCTION DOC]
  # HELPER: serialization helper used by the get-config methods of cells and layers.
  # Takes a cell. Returns a mapping containing the caching-device option only when the cell's
  # value differs from the default (which is true when running eagerly outside functions);
  # otherwise returns an empty mapping.
  """Return the dict config for RNN cell wrt to enable_caching_device field.

  Since enable_caching_device is a internal implementation detail for speed up
  the RNN variable read when running on the multi remote worker setting, we
  don't want this config to be serialized constantly in the JSON. We will only
  serialize this field when a none default value is used to create the cell.
  Args:
    rnn_cell: the RNN cell for serialize.

  Returns:
    A dict which contains the JSON config for enable_caching_device value or
    empty dict if the enable_caching_device value is same as the default value.
  """
  default_enable_caching_device = ops.executing_eagerly_outside_functions()
  if rnn_cell._enable_caching_device != default_enable_caching_device:
    return {'enable_caching_device': rnn_cell._enable_caching_device}
  return {}
