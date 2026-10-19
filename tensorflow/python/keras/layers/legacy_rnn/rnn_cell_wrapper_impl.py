# [MODULE DOC]
# HELPER: shared implementation module for the RNN cell wrappers.
# Holds the version-independent behavior of the dropout, residual and device cell wrappers as
# "Base" mixin classes, plus private helpers to serialize and restore user-supplied functions in
# wrapper configs and to enumerate nested structures. The public wrapper classes in the legacy
# cell module (V1) and in the Keras wrapper module (V2) combine these mixins with their own
# version-specific wrapper base class that provides the wrapped cell attribute and the
# constructor chain. No import-time side effects beyond imports.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Module contains the implementation of RNN cell wrappers."""
import hashlib
import numbers
import sys
import types as python_types
import warnings

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.util import nest


class DropoutWrapperBase(object):
  # [CLASS DOC]
  # HELPER: mixin with the dropout wrapper behavior shared by the V1 and V2 public wrappers.
  # Must be combined with a wrapper base that supplies the wrapped cell attribute and a
  # constructor taking the cell. Stores keep probabilities for input, state and output (floats
  # when constant, tensors otherwise), a variational-recurrent flag, input size, seed, the state
  # filter function, and, in variational mode, pre-drawn noise tensors reused at every step.
  # Delegates size properties, build and zero state to the wrapped cell and applies dropout
  # around each wrapped cell step.
  """Operator adding dropout to inputs and outputs of the given cell."""

  def __init__(self,
               cell,
               input_keep_prob=1.0,
               output_keep_prob=1.0,
               state_keep_prob=1.0,
               variational_recurrent=False,
               input_size=None,
               dtype=None,
               seed=None,
               dropout_state_filter_visitor=None,
               **kwargs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says a projection to output size is added to the cell (nothing
    # like that happens), describes state_keep_prob as an output probability, and the range-error
    # message formats the probability as an integer, so a value such as 1.5 is shown as 1.
    # Constructor. Inputs: the cell to wrap; keep probabilities for input, output and state
    # (floats or scalar tensors, default one meaning no dropout); variational-recurrent flag; the
    # input size structure (needed when variational and input keep is below one or unknown); the
    # dtype (needed when variational); an optional seed; an optional callable choosing which
    # parts of the state get dropout (default skips the LSTM cell state and tensor arrays); extra
    # base-layer arguments. Constant probabilities are stored as floats, others as tensors.
    # Raises a type error if the filter is not callable, and a value error if a constant
    # probability is outside zero to one, if variational mode lacks dtype, or lacks input size.
    # In variational mode it creates random noise tensors for input, state and output once, using
    # seeds derived from the seed, so the same mask is reused at every step.
    """Create a cell with added input, state, and/or output dropout.

    If `variational_recurrent` is set to `True` (**NOT** the default behavior),
    then the same dropout mask is applied at every step, as described in:
    [A Theoretically Grounded Application of Dropout in Recurrent
    Neural Networks. Y. Gal, Z. Ghahramani](https://arxiv.org/abs/1512.05287).

    Otherwise a different dropout mask is applied at every time step.

    Note, by default (unless a custom `dropout_state_filter` is provided),
    the memory state (`c` component of any `LSTMStateTuple`) passing through
    a `DropoutWrapper` is never modified.  This behavior is described in the
    above article.

    Args:
      cell: an RNNCell, a projection to output_size is added to it.
      input_keep_prob: unit Tensor or float between 0 and 1, input keep
        probability; if it is constant and 1, no input dropout will be added.
      output_keep_prob: unit Tensor or float between 0 and 1, output keep
        probability; if it is constant and 1, no output dropout will be added.
      state_keep_prob: unit Tensor or float between 0 and 1, output keep
        probability; if it is constant and 1, no output dropout will be added.
        State dropout is performed on the outgoing states of the cell. **Note**
        the state components to which dropout is applied when `state_keep_prob`
        is in `(0, 1)` are also determined by the argument
        `dropout_state_filter_visitor` (e.g. by default dropout is never applied
        to the `c` component of an `LSTMStateTuple`).
      variational_recurrent: Python bool.  If `True`, then the same dropout
        pattern is applied across all time steps per run call. If this parameter
        is set, `input_size` **must** be provided.
      input_size: (optional) (possibly nested tuple of) `TensorShape` objects
        containing the depth(s) of the input tensors expected to be passed in to
        the `DropoutWrapper`.  Required and used **iff** `variational_recurrent
        = True` and `input_keep_prob < 1`.
      dtype: (optional) The `dtype` of the input, state, and output tensors.
        Required and used **iff** `variational_recurrent = True`.
      seed: (optional) integer, the randomness seed.
      dropout_state_filter_visitor: (optional), default: (see below).  Function
        that takes any hierarchical level of the state and returns a scalar or
        depth=1 structure of Python booleans describing which terms in the state
        should be dropped out.  In addition, if the function returns `True`,
        dropout is applied across this sublevel.  If the function returns
        `False`, dropout is not applied across this entire sublevel.
        Default behavior: perform dropout on all terms except the memory (`c`)
          state of `LSTMCellState` objects, and don't try to apply dropout to
        `TensorArray` objects: ```
        def dropout_state_filter_visitor(s):
          if isinstance(s, LSTMCellState): # Never perform dropout on the c
            state. return LSTMCellState(c=False, h=True)
          elif isinstance(s, TensorArray): return False return True ```
      **kwargs: dict of keyword arguments for base layer.

    Raises:
      TypeError: if `cell` is not an `RNNCell`, or `keep_state_fn` is provided
        but not `callable`.
      ValueError: if any of the keep_probs are not between 0 and 1.
    """
    super(DropoutWrapperBase, self).__init__(cell, dtype=dtype, **kwargs)

    if (dropout_state_filter_visitor is not None and
        not callable(dropout_state_filter_visitor)):
      raise TypeError("dropout_state_filter_visitor must be callable")
    self._dropout_state_filter = (
        dropout_state_filter_visitor or _default_dropout_state_filter_visitor)
    with ops.name_scope_v2("DropoutWrapperInit"):

      def tensor_and_const_value(v):
        # [FUNCTION DOC]
        # HELPER: nested utility of the constructor. Converts a probability to a tensor and tries
        # to extract its constant value. Returns a pair of the tensor and the constant (null if
        # not statically known).
        tensor_value = tensor_conversion.convert_to_tensor_v2_with_dispatch(v)
        const_value = tensor_util.constant_value(tensor_value)
        return (tensor_value, const_value)

      for prob, attr in [(input_keep_prob, "input_keep_prob"),
                         (state_keep_prob, "state_keep_prob"),
                         (output_keep_prob, "output_keep_prob")]:
        tensor_prob, const_prob = tensor_and_const_value(prob)
        if const_prob is not None:
          if const_prob < 0 or const_prob > 1:
            raise ValueError("Parameter %s must be between 0 and 1: %d" %
                             (attr, const_prob))
          setattr(self, "_%s" % attr, float(const_prob))
        else:
          setattr(self, "_%s" % attr, tensor_prob)

    # Set variational_recurrent, seed before running the code below
    self._variational_recurrent = variational_recurrent
    self._input_size = input_size
    self._seed = seed

    self._recurrent_input_noise = None
    self._recurrent_state_noise = None
    self._recurrent_output_noise = None

    if variational_recurrent:
      if dtype is None:
        raise ValueError(
            "When variational_recurrent=True, dtype must be provided")

      def convert_to_batch_shape(s):
        # [FUNCTION DOC]
        # HELPER: nested utility of the constructor. Takes a per-example size, returns a shape
        # tensor with a leading one (batch dimension) so one mask is shared by all batch rows.
        # Prepend a 1 for the batch dimension; for recurrent
        # variational dropout we use the same dropout mask for all
        # batch elements.
        return array_ops.concat(([1], tensor_shape.TensorShape(s).as_list()), 0)

      def batch_noise(s, inner_seed):
        # [FUNCTION DOC]
        # HELPER: nested utility of the constructor. Given a per-example size and a seed, draws
        # a uniform random tensor of dtype given to the constructor, shaped with a leading one.
        # Returns the noise tensor; creates a random op.
        shape = convert_to_batch_shape(s)
        return random_ops.random_uniform(shape, seed=inner_seed, dtype=dtype)

      if (not isinstance(self._input_keep_prob, numbers.Real) or
          self._input_keep_prob < 1.0):
        if input_size is None:
          raise ValueError(
              "When variational_recurrent=True and input_keep_prob < 1.0 or "
              "is unknown, input_size must be provided")
        self._recurrent_input_noise = _enumerated_map_structure_up_to(
            input_size,
            lambda i, s: batch_noise(s, inner_seed=self._gen_seed("input", i)),
            input_size)
      self._recurrent_state_noise = _enumerated_map_structure_up_to(
          cell.state_size,
          lambda i, s: batch_noise(s, inner_seed=self._gen_seed("state", i)),
          cell.state_size)
      self._recurrent_output_noise = _enumerated_map_structure_up_to(
          cell.output_size,
          lambda i, s: batch_noise(s, inner_seed=self._gen_seed("output", i)),
          cell.output_size)

  def _gen_seed(self, salt_prefix, index):
    # [FUNCTION DOC]
    # HELPER: derives a deterministic per-use random seed. Inputs: a text prefix (input, state or
    # output) and an integer index of the leaf. Returns nothing/null if the wrapper has no seed;
    # otherwise hashes the seed plus prefix and index and returns a non-negative 31-bit integer.
    if self._seed is None:
      return None
    salt = "%s_%d" % (salt_prefix, index)
    string = (str(self._seed) + salt).encode("utf-8")
    return int(hashlib.md5(string).hexdigest()[:8], 16) & 0x7FFFFFFF

  @property
  def wrapped_cell(self):
    # [FUNCTION DOC]
    # Property returning the wrapped cell.
    return self.cell

  @property
  def state_size(self):
    # [FUNCTION DOC]
    # Property returning the wrapped cell's state size.
    return self.cell.state_size

  @property
  def output_size(self):
    # [FUNCTION DOC]
    # Property returning the wrapped cell's output size.
    return self.cell.output_size

  def build(self, inputs_shape):
    # [FUNCTION DOC]
    # Builds the wrapped cell with the given input shape and marks the wrapper as built. Returns
    # nothing; creates the wrapped cell's variables.
    self.cell.build(inputs_shape)
    self.built = True

  def zero_state(self, batch_size, dtype):
    # [FUNCTION DOC]
    # Returns the wrapped cell's zero state for the given batch size and dtype, created inside a
    # name scope named after the wrapper class.
    with ops.name_scope_v2(type(self).__name__ + "ZeroState"):
      return self.cell.zero_state(batch_size, dtype)

  def _variational_recurrent_dropout_value(
      self, unused_index, value, noise, keep_prob):
    # [FUNCTION DOC]
    # HELPER: applies one dropout with a pre-drawn noise tensor (variational dropout). Inputs:
    # an ignored leaf index, the tensor to drop from, the uniform noise tensor, the keep
    # probability. Adds the keep probability to the noise, floors it to a binary mask, scales the
    # value by one over keep probability and multiplies by the mask, then restores the static
    # shape. Returns the result.
    """Performs dropout given the pre-calculated noise tensor."""
    # uniform [keep_prob, 1.0 + keep_prob)
    random_tensor = keep_prob + noise

    # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
    binary_tensor = math_ops.floor(random_tensor)
    ret = math_ops.divide(value, keep_prob) * binary_tensor
    ret.set_shape(value.get_shape())
    return ret

  def _dropout(self,
               values,
               salt_prefix,
               recurrent_noise,
               keep_prob,
               shallow_filtered_substructure=None):
    # [FUNCTION DOC]
    # HELPER: applies dropout to a nested structure of tensors. Inputs: the values; a salt prefix
    # naming which part (input, state, output) for seed derivation; the pre-drawn noise structure
    # (used only in variational mode); the keep probability; an optional structure of booleans
    # (same top levels as the values) marking where dropout applies, defaulting to the values
    # themselves so everything is dropped. In ordinary mode applies standard dropout with a
    # derived seed to each selected leaf; in variational mode reuses the noise. Leaves marked
    # false pass through unchanged. Returns a structure like the values.
    """Decides whether to perform standard dropout or recurrent dropout."""

    if shallow_filtered_substructure is None:
      # Put something so we traverse the entire structure; inside the
      # dropout function we check to see if leafs of this are bool or not.
      shallow_filtered_substructure = values

    if not self._variational_recurrent:

      def dropout(i, do_dropout, v):
        # [FUNCTION DOC]
        # HELPER: nested leaf function for ordinary dropout. Inputs: leaf index, a flag (or
        # non-boolean marker meaning yes) saying whether to drop, and the tensor. Returns the
        # tensor after standard dropout with rate one minus keep probability and a derived seed,
        # or unchanged if the flag is false.
        if not isinstance(do_dropout, bool) or do_dropout:
          return nn_ops.dropout_v2(
              v, rate=1. - keep_prob, seed=self._gen_seed(salt_prefix, i))
        else:
          return v

      return _enumerated_map_structure_up_to(
          shallow_filtered_substructure, dropout,
          *[shallow_filtered_substructure, values])
    else:

      def dropout(i, do_dropout, v, n):
        # [FUNCTION DOC]
        # HELPER: nested leaf function for variational dropout. Inputs: leaf index, a flag saying
        # whether to drop, the tensor and its pre-drawn noise. Returns the tensor masked with
        # that noise, or unchanged if the flag is false.
        if not isinstance(do_dropout, bool) or do_dropout:
          return self._variational_recurrent_dropout_value(i, v, n, keep_prob)
        else:
          return v

      return _enumerated_map_structure_up_to(
          shallow_filtered_substructure, dropout,
          *[shallow_filtered_substructure, values, recurrent_noise])

  def _call_wrapped_cell(self, inputs, state, cell_call_fn, **kwargs):
    # [FUNCTION DOC]
    # HELPER: wrapper step used by both wrapper versions. Applies input dropout (if the input keep
    # probability is a tensor or below one), calls the wrapped cell through the given function
    # with extra keyword arguments, then applies state dropout (only on parts selected by the
    # state filter) and output dropout under the same rule. Returns the pair of output and new
    # state.
    """Runs the wrapped cell and applies dropout.

    Args:
      inputs: A tensor with wrapped cell's input.
      state: A tensor or tuple of tensors with wrapped cell's state.
      cell_call_fn: Wrapped cell's method to use for step computation (cell's
        `__call__` or 'call' method).
      **kwargs: Additional arguments.

    Returns:
      A pair containing:

      - Output: A tensor with cell's output.
      - New state: A tensor or tuple of tensors with new wrapped cell's state.
    """

    def _should_dropout(p):
      # [FUNCTION DOC]
      # HELPER: nested predicate. Returns true if the keep probability is not a plain float
      # (so possibly a tensor, unknown at build time) or is below one.
      return (not isinstance(p, float)) or p < 1

    if _should_dropout(self._input_keep_prob):
      inputs = self._dropout(inputs, "input", self._recurrent_input_noise,
                             self._input_keep_prob)
    output, new_state = cell_call_fn(inputs, state, **kwargs)
    if _should_dropout(self._state_keep_prob):
      # Identify which subsets of the state to perform dropout on and
      # which ones to keep.
      shallow_filtered_substructure = nest.get_traverse_shallow_structure(
          self._dropout_state_filter, new_state)
      new_state = self._dropout(new_state, "state", self._recurrent_state_noise,
                                self._state_keep_prob,
                                shallow_filtered_substructure)
    if _should_dropout(self._output_keep_prob):
      output = self._dropout(output, "output", self._recurrent_output_noise,
                             self._output_keep_prob)
    return output, new_state

  def get_config(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: keep probabilities that were given as tensors are stored as is, so the config
    # may hold tensors that are not serializable.
    # Returns the parent config merged with the keep probabilities, variational flag, input size,
    # seed and, if a custom state filter is used, its serialized form (function data, type, and
    # module). May raise a value error if the custom filter is not callable.
    """Returns the config of the dropout wrapper."""
    config = {
        "input_keep_prob": self._input_keep_prob,
        "output_keep_prob": self._output_keep_prob,
        "state_keep_prob": self._state_keep_prob,
        "variational_recurrent": self._variational_recurrent,
        "input_size": self._input_size,
        "seed": self._seed,
    }
    if self._dropout_state_filter != _default_dropout_state_filter_visitor:
      function, function_type, function_module = _serialize_function_to_config(
          self._dropout_state_filter)
      config.update({"dropout_fn": function,
                     "dropout_fn_type": function_type,
                     "dropout_fn_module": function_module})
    base_config = super(DropoutWrapperBase, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    # [FUNCTION DOC]
    # Class method that restores a dropout wrapper from a config mapping. If the config holds a
    # serialized state filter, copies the config, rebuilds the function (using the optional
    # custom objects mapping), and passes it under the constructor's filter argument. Then
    # delegates to the next base class's restoration. Returns the new wrapper.
    if "dropout_fn" in config:
      config = config.copy()
      dropout_state_filter = _parse_config_to_function(
          config, custom_objects, "dropout_fn", "dropout_fn_type",
          "dropout_fn_module")
      config.pop("dropout_fn")
      config["dropout_state_filter_visitor"] = dropout_state_filter
    return super(DropoutWrapperBase, cls).from_config(
        config, custom_objects=custom_objects)


class ResidualWrapperBase(object):
  # [CLASS DOC]
  # HELPER: mixin with the residual wrapper behavior shared by the V1 and V2 public wrappers.
  # Stores an optional residual function; after each wrapped cell step it combines the cell
  # input with the cell output (by default element-wise addition after checking matching
  # structure and compatible shapes). Delegates size properties and zero state to the wrapped
  # cell. Needs a wrapper base that supplies the cell attribute.
  """RNNCell wrapper that ensures cell inputs are added to the outputs."""

  def __init__(self, cell, residual_fn=None, **kwargs):
    # [FUNCTION DOC]
    # Constructor. Inputs: the cell to wrap; an optional function mapping the raw input and raw
    # output to the final output (default adds them); extra base-layer arguments. Stores the
    # function. Returns nothing.
    """Constructs a `ResidualWrapper` for `cell`.

    Args:
      cell: An instance of `RNNCell`.
      residual_fn: (Optional) The function to map raw cell inputs and raw cell
        outputs to the actual cell outputs of the residual network.
        Defaults to calling nest.map_structure on (lambda i, o: i + o), inputs
          and outputs.
      **kwargs: dict of keyword arguments for base layer.
    """
    super(ResidualWrapperBase, self).__init__(cell, **kwargs)
    self._residual_fn = residual_fn

  @property
  def state_size(self):
    # [FUNCTION DOC]
    # Property returning the wrapped cell's state size.
    return self.cell.state_size

  @property
  def output_size(self):
    # [FUNCTION DOC]
    # Property returning the wrapped cell's output size.
    return self.cell.output_size

  def zero_state(self, batch_size, dtype):
    # [FUNCTION DOC]
    # Returns the wrapped cell's zero state for the given batch size and dtype, created inside a
    # name scope named after the wrapper class.
    with ops.name_scope_v2(type(self).__name__ + "ZeroState"):
      return self.cell.zero_state(batch_size, dtype)

  def _call_wrapped_cell(self, inputs, state, cell_call_fn, **kwargs):
    # [FUNCTION DOC]
    # HELPER: wrapper step used by both wrapper versions. Calls the wrapped cell through the given
    # function (extra keyword arguments are forwarded), then combines the original input with the
    # output using the residual function or the default. Returns the pair of combined output and
    # the unchanged new state. The default raises a type or value error if input and output
    # structures differ and an error if shapes are incompatible.
    """Run the cell and then apply the residual_fn on its inputs to its outputs.

    Args:
      inputs: cell inputs.
      state: cell state.
      cell_call_fn: Wrapped cell's method to use for step computation (cell's
        `__call__` or 'call' method).
      **kwargs: Additional arguments passed to the wrapped cell's `call`.

    Returns:
      Tuple of cell outputs and new state.

    Raises:
      TypeError: If cell inputs and outputs have different structure (type).
      ValueError: If cell inputs and outputs have different structure (value).
    """
    outputs, new_state = cell_call_fn(inputs, state, **kwargs)

    # Ensure shapes match
    def assert_shape_match(inp, out):
      # [FUNCTION DOC]
      # HELPER: nested check. Raises an error unless the static shapes of one input leaf and one
      # output leaf are compatible. Returns nothing.
      inp.get_shape().assert_is_compatible_with(out.get_shape())

    def default_residual_fn(inputs, outputs):
      # [FUNCTION DOC]
      # HELPER: nested default residual function. Checks that inputs and outputs have the same
      # nested structure and compatible leaf shapes, then returns the element-wise sum per leaf.
      nest.assert_same_structure(inputs, outputs)
      nest.map_structure(assert_shape_match, inputs, outputs)
      return nest.map_structure(lambda inp, out: inp + out, inputs, outputs)

    res_outputs = (self._residual_fn or default_residual_fn)(inputs, outputs)
    return (res_outputs, new_state)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the parent config, merged with the serialized residual function (data, type,
    # module) when a custom one was given. Raises a value error if the function is not callable.
    """Returns the config of the residual wrapper."""
    if self._residual_fn is not None:
      function, function_type, function_module = _serialize_function_to_config(
          self._residual_fn)
      config = {
          "residual_fn": function,
          "residual_fn_type": function_type,
          "residual_fn_module": function_module
      }
    else:
      config = {}
    base_config = super(ResidualWrapperBase, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    # [FUNCTION DOC]
    # Class method restoring a residual wrapper from a config mapping. If a serialized residual
    # function is present, copies the config, rebuilds the function (with optional custom
    # objects) and puts it back under its key; then delegates to the next base class. Returns
    # the new wrapper.
    if "residual_fn" in config:
      config = config.copy()
      residual_function = _parse_config_to_function(config, custom_objects,
                                                    "residual_fn",
                                                    "residual_fn_type",
                                                    "residual_fn_module")
      config["residual_fn"] = residual_function
    return super(ResidualWrapperBase, cls).from_config(
        config, custom_objects=custom_objects)


class DeviceWrapperBase(object):
  # [CLASS DOC]
  # HELPER: mixin with the device wrapper behavior shared by the V1 and V2 public wrappers.
  # Stores a device specification and runs the wrapped cell step and its zero-state creation
  # inside that device scope. Delegates size properties to the wrapped cell.
  """Operator that ensures an RNNCell runs on a particular device."""

  def __init__(self, cell, device, **kwargs):
    # [FUNCTION DOC]
    # Constructor. Inputs: the cell to wrap; a device string or device function; extra base-layer
    # arguments. Stores the device. Returns nothing.
    """Construct a `DeviceWrapper` for `cell` with device `device`.

    Ensures the wrapped `cell` is called with `tf.device(device)`.

    Args:
      cell: An instance of `RNNCell`.
      device: A device string or function, for passing to `tf.device`.
      **kwargs: dict of keyword arguments for base layer.
    """
    super(DeviceWrapperBase, self).__init__(cell, **kwargs)
    self._device = device

  @property
  def state_size(self):
    # [FUNCTION DOC]
    # Property returning the wrapped cell's state size.
    return self.cell.state_size

  @property
  def output_size(self):
    # [FUNCTION DOC]
    # Property returning the wrapped cell's output size.
    return self.cell.output_size

  def zero_state(self, batch_size, dtype):
    # [FUNCTION DOC]
    # Returns the wrapped cell's zero state for the given batch size and dtype, created inside a
    # name scope named after the wrapper class and under the configured device.
    with ops.name_scope_v2(type(self).__name__ + "ZeroState"):
      with ops.device(self._device):
        return self.cell.zero_state(batch_size, dtype)

  def _call_wrapped_cell(self, inputs, state, cell_call_fn, **kwargs):
    # [FUNCTION DOC]
    # HELPER: wrapper step used by both wrapper versions. Calls the wrapped cell through the
    # given function (extra keyword arguments forwarded) inside the configured device scope and
    # returns its pair of output and new state unchanged.
    """Run the cell on specified device."""
    with ops.device(self._device):
      return cell_call_fn(inputs, state, **kwargs)

  def get_config(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the device is stored as given, so a device function (allowed by the
    # constructor) ends up in the config and cannot be serialized.
    # Returns the parent config merged with the device specification.
    config = {"device": self._device}
    base_config = super(DeviceWrapperBase, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


def _serialize_function_to_config(function):
  # [FUNCTION DOC]
  # HELPER: serializes a user function stored in a wrapper config.
  # KNOWN ISSUE: the lambda type check matches every plain Python function (lambdas and named
  # functions share a type), so named functions are dumped as bytecode and labelled lambda; the
  # name-based "function" branch is only reached for other callables such as callable objects.
  # Input: a callable. Returns a triple: the serialized function (bytecode dump or name), the
  # type label (lambda or function) and the function's module name. Raises a value error if the
  # input is not callable.
  """Serialize the function for get_config()."""
  if isinstance(function, python_types.LambdaType):
    output = generic_utils.func_dump(function)
    output_type = "lambda"
    module = function.__module__
  elif callable(function):
    output = function.__name__
    output_type = "function"
    module = function.__module__
  else:
    raise ValueError("Unrecognized function type for input: {}".format(
        type(function)))

  return output, output_type, module


def _parse_config_to_function(config, custom_objects, func_attr_name,
                              func_type_attr_name, module_attr_name):
  # [FUNCTION DOC]
  # HELPER: inverse of the function serializer used by the wrappers' restoration methods.
  # KNOWN ISSUE: it updates this module's own global namespace (not a copy) with the target
  # module's contents and the custom objects, polluting module state for later calls; lambdas
  # are rebuilt from raw bytecode, which is unsafe for untrusted configs.
  # Inputs: the config mapping; optional custom objects mapping; and the key names under which
  # the function data, its type label and its module name are stored. Removes the module and
  # type keys from the config (mutates it), warns if the module is not loaded, then restores the
  # function either by name lookup (function type) or by loading bytecode (lambda type). Raises a
  # type error for an unknown type label. Returns the function.
  """Reconstruct the function from the config."""
  globs = globals()
  module = config.pop(module_attr_name, None)
  if module in sys.modules:
    globs.update(sys.modules[module].__dict__)
  elif module is not None:
    # Note: we don't know the name of the function if it's a lambda.
    warnings.warn("{} is not loaded, but a layer uses it. "
                  "It may cause errors.".format(module), UserWarning)
  if custom_objects:
    globs.update(custom_objects)
  function_type = config.pop(func_type_attr_name)
  if function_type == "function":
    # Simple lookup in custom objects
    function = generic_utils.deserialize_keras_object(
        config[func_attr_name],
        custom_objects=custom_objects,
        printable_module_name="function in wrapper")
  elif function_type == "lambda":
    # Unsafe deserialization from bytecode
    function = generic_utils.func_load(
        config[func_attr_name], globs=globs)
  else:
    raise TypeError("Unknown function type:", function_type)
  return function


def _default_dropout_state_filter_visitor(substate):
  # [FUNCTION DOC]
  # HELPER: default state filter for the dropout wrapper. Input: one level of the state. Returns
  # a pair marking the LSTM cell state as not dropped and hidden state as dropped when given an
  # LSTM state tuple, false for tensor arrays, and true otherwise. Imports the LSTM state class
  # lazily to avoid a circular import.
  from tensorflow.python.keras.layers.legacy_rnn.rnn_cell_impl import LSTMStateTuple  # pylint: disable=g-import-not-at-top
  if isinstance(substate, LSTMStateTuple):
    # Do not perform dropout on the memory state.
    return LSTMStateTuple(c=False, h=True)
  elif isinstance(substate, tensor_array_ops.TensorArray):
    return False
  return True


def _enumerated_map_structure_up_to(shallow_structure, map_fn, *args, **kwargs):
  # [FUNCTION DOC]
  # HELPER: structure mapper for the dropout wrapper. Like the nest utility mapping a function up
  # to a shallow structure, but the function also receives a running integer index (starting at
  # zero, in visiting order) as its first argument, which is used to derive per-leaf seeds.
  # Inputs: shallow structure, the mapping function, the structures to map over and extra keyword
  # arguments. Returns the mapped structure.
  ix = [0]

  def enumerated_fn(*inner_args, **inner_kwargs):
    # [FUNCTION DOC]
    # HELPER: nested adapter that calls the user function with the current counter value as the
    # first argument, increments the shared counter, and returns the user function's result.
    r = map_fn(ix[0], *inner_args, **inner_kwargs)
    ix[0] += 1
    return r

  return nest.map_structure_up_to(shallow_structure, enumerated_fn, *args,
                                  **kwargs)
