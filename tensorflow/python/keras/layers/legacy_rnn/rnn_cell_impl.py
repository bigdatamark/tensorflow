# [MODULE DOC]
# Legacy (TF1 style) RNN cell implementations exported under the v1 nn.rnn_cell namespace.
# Defines the abstract RNNCell layer and the proper-layer variant LayerRNNCell, the concrete
# cells (basic RNN, GRU, basic LSTM, full LSTM with peepholes and projection), the LSTM state
# named tuple, the V1 wrapper base class plus dropout, residual and device wrappers (whose
# behavior lives in the wrapper implementation module), the multi-layer stacking cell, and
# private helpers for shape concatenation, zero state creation, dtype checks and cell-likeness
# assertions. All cells are deprecated in favor of the Keras RNN cells. Import-time side effects:
# registers the v1 API export names; nothing else is executed.
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
# pylint: disable=g-classes-have-attributes
"""Module implementing RNN Cells.

This module provides a number of basic commonly used RNN cells, such as LSTM
(Long Short Term Memory) or GRU (Gated Recurrent Unit), and a number of
operators that allow adding dropouts, projections, or embeddings for inputs.
Constructing multi-layer cells is supported by the class `MultiRNNCell`, or by
calling the `rnn` ops several times.
"""
import collections
import warnings

from tensorflow.python.eager import context
from tensorflow.python.framework import config as tf_config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.layers.legacy_rnn import rnn_cell_wrapper_impl
from tensorflow.python.keras.legacy_tf_layers import base as base_layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

# This can be used with self.assertRaisesRegexp for assert_like_rnncell.
ASSERT_LIKE_RNNCELL_ERROR_REGEXP = "is not an RNNCell"


def _hasattr(obj, attr_name):
  # [FUNCTION DOC]
  # HELPER: attribute probe used by assert_like_rnncell and zero_state caching.
  # Returns true if reading the named attribute from the object succeeds, false if that read
  # raises an attribute error (so properties that raise it count as missing). Other exceptions
  # propagate. Note that the attribute is actually evaluated, so properties run their code.
  try:
    getattr(obj, attr_name)
  except AttributeError:
    return False
  else:
    return True


def assert_like_rnncell(cell_name, cell):
  # [FUNCTION DOC]
  # Duck-type validation that an object behaves like an RNN cell. Inputs: a display name for the
  # argument (used in the message) and the candidate cell. The cell must have output size and
  # state size attributes, either an initial-state method or a zero-state method, and be
  # callable. Returns nothing. Raises a type error listing every failed requirement; the message
  # always contains the text matched by the module's error-regexp constant. No side effects.
  """Raises a TypeError if cell is not like an RNNCell.

  NOTE: Do not rely on the error message (in particular in tests) which can be
  subject to change to increase readability. Use
  ASSERT_LIKE_RNNCELL_ERROR_REGEXP.

  Args:
    cell_name: A string to give a meaningful error referencing to the name of
      the functionargument.
    cell: The object which should behave like an RNNCell.

  Raises:
    TypeError: A human-friendly exception.
  """
  conditions = [
      _hasattr(cell, "output_size"),
      _hasattr(cell, "state_size"),
      _hasattr(cell, "get_initial_state") or _hasattr(cell, "zero_state"),
      callable(cell),
  ]
  errors = [
      "'output_size' property is missing", "'state_size' property is missing",
      "either 'zero_state' or 'get_initial_state' method is required",
      "is not callable"
  ]

  if not all(conditions):

    errors = [error for error, cond in zip(errors, conditions) if not cond]
    raise TypeError("The argument {!r} ({}) is not an RNNCell: {}.".format(
        cell_name, cell, ", ".join(errors)))


def _concat(prefix, suffix, static=False):
  # [FUNCTION DOC]
  # HELPER: builds a state or output shape by joining a prefix (usually batch size) and a suffix
  # (the per-example size) for zero-state creation.
  # Inputs: prefix and suffix, each an integer, shape object, or tensor (tensors must be scalar
  # or vector); static flag. With static false, returns an int32 tensor with the concatenation;
  # with static true, returns a list of integers (entries may be unknown) or nothing/null if the
  # rank is unknown. Raises a value error for tensors that are not scalar or vector, and, in the
  # non-static case, when either part is not fully defined (null). Side effect: may create
  # constant ops in the current graph.
  """Concat that enables int, Tensor, or TensorShape values.

  This function takes a size specification, which can be an integer, a
  TensorShape, or a Tensor, and converts it into a concatenated Tensor
  (if static = False) or a list of integers (if static = True).

  Args:
    prefix: The prefix; usually the batch size (and/or time step size).
      (TensorShape, int, or Tensor.)
    suffix: TensorShape, int, or Tensor.
    static: If `True`, return a python list with possibly unknown dimensions.
      Otherwise return a `Tensor`.

  Returns:
    shape: the concatenation of prefix and suffix.

  Raises:
    ValueError: if `suffix` is not a scalar or vector (or TensorShape).
    ValueError: if prefix or suffix was `None` and asked for dynamic
      Tensors out.
  """
  if isinstance(prefix, tensor.Tensor):
    p = prefix
    p_static = tensor_util.constant_value(prefix)
    if p.shape.ndims == 0:
      p = array_ops.expand_dims(p, 0)
    elif p.shape.ndims != 1:
      raise ValueError("prefix tensor must be either a scalar or vector, "
                       "but saw tensor: %s" % p)
  else:
    p = tensor_shape.TensorShape(prefix)
    p_static = p.as_list() if p.ndims is not None else None
    p = (
        constant_op.constant(p.as_list(), dtype=dtypes.int32)
        if p.is_fully_defined() else None)
  if isinstance(suffix, tensor.Tensor):
    s = suffix
    s_static = tensor_util.constant_value(suffix)
    if s.shape.ndims == 0:
      s = array_ops.expand_dims(s, 0)
    elif s.shape.ndims != 1:
      raise ValueError("suffix tensor must be either a scalar or vector, "
                       "but saw tensor: %s" % s)
  else:
    s = tensor_shape.TensorShape(suffix)
    s_static = s.as_list() if s.ndims is not None else None
    s = (
        constant_op.constant(s.as_list(), dtype=dtypes.int32)
        if s.is_fully_defined() else None)

  if static:
    shape = tensor_shape.TensorShape(p_static).concatenate(s_static)
    shape = shape.as_list() if shape.ndims is not None else None
  else:
    if p is None or s is None:
      raise ValueError("Provided a prefix or suffix of None: %s and %s" %
                       (prefix, suffix))
    shape = array_ops.concat((p, s), 0)
  return shape


def _zero_state_tensors(state_size, batch_size, dtype):
  # [FUNCTION DOC]
  # HELPER: creates the zero initial state for RNNCell.zero_state.
  # Inputs: a state size (integer, shape, or arbitrarily nested structure of them), a batch size
  # (integer or scalar tensor) and a dtype. Returns zero-filled tensors with the same nesting as
  # the state size, each shaped batch size followed by that entry's size. Creates zeros ops.
  """Create tensors of zeros based on state_size, batch_size, and dtype."""

  def get_state_shape(s):
    # [FUNCTION DOC]
    # HELPER: nested mapper for the zero-state builder. Takes one state size entry, builds a
    # zeros tensor of batch size plus that entry in the enclosing dtype, and, in graph mode only,
    # sets its static shape from the static concatenation. Returns the zeros tensor.
    """Combine s with batch_size to get a proper tensor shape."""
    c = _concat(batch_size, s)
    size = array_ops.zeros(c, dtype=dtype)
    if not context.executing_eagerly():
      c_static = _concat(batch_size, s, static=True)
      size.set_shape(c_static)
    return size

  return nest.map_structure(get_state_shape, state_size)


@tf_export(v1=["nn.rnn_cell.RNNCell"])
class RNNCell(base_layer.Layer):
  # [CLASS DOC]
  # KNOWN ISSUE: the docstring says the state structure uses each entry of the batch size; the
  # sentence should refer to the state size.
  # Abstract base for legacy RNN cells, built on the legacy TF layer class. Subclasses define the
  # state size and output size properties and a call method mapping an input and a state to an
  # output and a new state. Adds variable-scope based variable capturing (variables created via
  # the scope getter are tracked as layer weights), zero-state helpers with eager caching of the
  # last result, and a flag marking it as a TF (not Keras) cell. Exported in the v1 namespace.
  """Abstract object representing an RNN cell.

  Every `RNNCell` must have the properties below and implement `call` with
  the signature `(output, next_state) = call(input, state)`.  The optional
  third input argument, `scope`, is allowed for backwards compatibility
  purposes; but should be left off for new subclasses.

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

  def __init__(self, trainable=True, name=None, dtype=None, **kwargs):
    # [FUNCTION DOC]
    # Constructor. Inputs: trainable flag (default true), optional layer name, optional dtype and
    # extra keyword arguments passed to the parent layer constructor. Sets a private flag marking
    # this as a TF RNN cell (single-tensor states are not wrapped in a list, unlike Keras cells).
    # Returns nothing.
    super(RNNCell, self).__init__(
        trainable=trainable, name=name, dtype=dtype, **kwargs)
    # Attribute that indicates whether the cell is a TF RNN cell, due the slight
    # difference between TF and Keras RNN cell. Notably the state is not wrapped
    # in a list for TF cell where they are single tensor state, whereas keras
    # cell will wrap the state into a list, and call() will have to unwrap them.
    self._is_tf_rnn_cell = True

  def __call__(self, inputs, state, scope=None):
    # [FUNCTION DOC]
    # Runs one step of the cell. Inputs: a 2-D input tensor, the state (tensor or tuple), and an
    # optional variable scope. Opens the given scope (or a lazily created scope cached on the
    # object under a private attribute, derived from the current variable scope) with the
    # variable-capturing getter installed, then invokes the parent layer call. Returns the pair of
    # output and new state. Side effects: may store the cached scope on the instance, creates
    # variables via the layer machinery.
    """Run this RNN cell on inputs, starting from the given state.

    Args:
      inputs: `2-D` tensor with shape `[batch_size, input_size]`.
      state: if `self.state_size` is an integer, this should be a `2-D Tensor`
        with shape `[batch_size, self.state_size]`.  Otherwise, if
        `self.state_size` is a tuple of integers, this should be a tuple with
        shapes `[batch_size, s] for s in self.state_size`.
      scope: VariableScope for the created subgraph; defaults to class name.

    Returns:
      A pair containing:

      - Output: A `2-D` tensor with shape `[batch_size, self.output_size]`.
      - New state: Either a single `2-D` tensor, or a tuple of tensors matching
        the arity and shapes of `state`.
    """
    if scope is not None:
      with vs.variable_scope(
          scope, custom_getter=self._rnn_get_variable) as scope:
        return super(RNNCell, self).__call__(inputs, state, scope=scope)
    else:
      scope_attrname = "rnncell_scope"
      scope = getattr(self, scope_attrname, None)
      if scope is None:
        scope = vs.variable_scope(
            vs.get_variable_scope(), custom_getter=self._rnn_get_variable)
        setattr(self, scope_attrname, scope)
      with scope:
        return super(RNNCell, self).__call__(inputs, state)

  def _rnn_get_variable(self, getter, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: custom variable getter installed by the call method so variables created through
    # variable scopes are tracked as weights of the cell.
    # Calls the real getter with the given arguments, decides whether the variable is trainable
    # (the variable's flag when eager outside functions; otherwise membership, or that of its
    # first partition, in the graph trainable collection), and appends it to the trainable or
    # non-trainable weight list unless the same object is already there. Returns the variable.
    variable = getter(*args, **kwargs)
    if ops.executing_eagerly_outside_functions():
      trainable = variable.trainable
    else:
      trainable = (
          variable in tf_variables.trainable_variables() or
          (base_layer_utils.is_split_variable(variable) and
           list(variable)[0] in tf_variables.trainable_variables()))
    if trainable and all(variable is not v for v in self._trainable_weights):
      self._trainable_weights.append(variable)
    elif not trainable and all(
        variable is not v for v in self._non_trainable_weights):
      self._non_trainable_weights.append(variable)
    return variable

  @property
  def state_size(self):
    # [FUNCTION DOC]
    # Abstract property for the size(s) of the state(s) of the cell (integer, shape, or nested
    # tuple of them). This base version always raises a not-implemented error.
    """size(s) of state(s) used by this cell.

    It can be represented by an Integer, a TensorShape or a tuple of Integers
    or TensorShapes.
    """
    raise NotImplementedError("Abstract method")

  @property
  def output_size(self):
    # [FUNCTION DOC]
    # Abstract property for the output size of the cell (integer or shape). This base version
    # always raises a not-implemented error.
    """Integer or TensorShape: size of outputs produced by this cell."""
    raise NotImplementedError("Abstract method")

  def build(self, _):
    # [FUNCTION DOC]
    # No-op build hook that ignores the input shape. Its presence tells the parent layer that
    # variables may be added during call. Returns nothing and does not mark the layer as built.
    # This tells the parent Layer object that it's OK to call
    # self.add_variable() inside the call() method.
    pass

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    # [FUNCTION DOC]
    # Returns the initial (all zero) state, with Keras-style arguments. Inputs: an optional input
    # tensor whose first dimension gives the batch size and whose dtype gives the dtype; an
    # optional batch size; an optional dtype. If inputs are given they are converted to a tensor
    # and any given batch size or dtype must match them. Raises a value error on a batch size or
    # dtype mismatch, or if the batch size or dtype is still unknown at the end. Delegates to
    # zero state and returns its result.
    if inputs is not None:
      # Validate the given batch_size and dtype against inputs if provided.
      inputs = tensor_conversion.convert_to_tensor_v2_with_dispatch(
          inputs, name="inputs"
      )
      if batch_size is not None:
        if tensor_util.is_tf_type(batch_size):
          static_batch_size = tensor_util.constant_value(
              batch_size, partial=True)
        else:
          static_batch_size = batch_size
        if inputs.shape.dims[0].value != static_batch_size:
          raise ValueError(
              "batch size from input tensor is different from the "
              "input param. Input tensor batch: {}, batch_size: {}".format(
                  inputs.shape.dims[0].value, batch_size))

      if dtype is not None and inputs.dtype != dtype:
        raise ValueError(
            "dtype from input tensor is different from the "
            "input param. Input tensor dtype: {}, dtype: {}".format(
                inputs.dtype, dtype))

      batch_size = inputs.shape.dims[0].value or array_ops.shape(inputs)[0]
      dtype = inputs.dtype
    if batch_size is None or dtype is None:
      raise ValueError(
          "batch_size and dtype cannot be None while constructing initial "
          "state: batch_size={}, dtype={}".format(batch_size, dtype))
    return self.zero_state(batch_size, dtype)

  def zero_state(self, batch_size, dtype):
    # [FUNCTION DOC]
    # Builds zero-filled state tensors matching the cell's state size. Inputs: batch size
    # (integer, float or scalar tensor) and dtype. In eager mode, if the batch size, dtype and
    # state size equal those of the previous call, the cached result is returned. Otherwise the
    # tensors are created inside a name scope named after the class and, in eager mode, cached on
    # the instance in a private attribute. Returns a tensor or nested structure of tensors.
    """Return zero-filled state tensor(s).

    Args:
      batch_size: int, float, or unit Tensor representing the batch size.
      dtype: the data type to use for the state.

    Returns:
      If `state_size` is an int or TensorShape, then the return value is a
      `N-D` tensor of shape `[batch_size, state_size]` filled with zeros.

      If `state_size` is a nested list or tuple, then the return value is
      a nested list or tuple (of the same structure) of `2-D` tensors with
      the shapes `[batch_size, s]` for each s in `state_size`.
    """
    # Try to use the last cached zero_state. This is done to avoid recreating
    # zeros, especially when eager execution is enabled.
    state_size = self.state_size
    is_eager = context.executing_eagerly()
    if is_eager and _hasattr(self, "_last_zero_state"):
      (last_state_size, last_batch_size, last_dtype,
       last_output) = getattr(self, "_last_zero_state")
      if (last_batch_size == batch_size and last_dtype == dtype and
          last_state_size == state_size):
        return last_output
    with backend.name_scope(type(self).__name__ + "ZeroState"):
      output = _zero_state_tensors(state_size, batch_size, dtype)
    if is_eager:
      self._last_zero_state = (state_size, batch_size, dtype, output)
    return output

  # TODO(b/134773139): Remove when contrib RNN cells implement `get_config`
  def get_config(self):  # pylint: disable=useless-super-delegation
    # [FUNCTION DOC]
    # Returns the parent layer's configuration mapping unchanged; exists only so legacy cells
    # expose a configuration method. No arguments, no side effects.
    return super(RNNCell, self).get_config()

  @property
  def _use_input_spec_as_call_signature(self):
    # [FUNCTION DOC]
    # HELPER: private hook read by the layer machinery. Always returns false so no call
    # signature is generated from the input spec, since the state argument's shape is not stored.
    # We do not store the shape information for the state argument in the call
    # function for legacy RNN cells, so do not generate an input signature.
    return False


class LayerRNNCell(RNNCell):
  # [CLASS DOC]
  # RNN cell base whose instances behave like ordinary layers: variables are created in a build
  # method (called once) rather than through scope-based variable getters in call. Overrides the
  # call entry point to skip the variable-capturing scope logic of the RNN cell parent. Parent of
  # the concrete cells in this module.
  """Subclass of RNNCells that act like proper `tf.Layer` objects.

  For backwards compatibility purposes, most `RNNCell` instances allow their
  `call` methods to instantiate variables via `tf.compat.v1.get_variable`.  The
  underlying
  variable scope thus keeps track of any variables, and returning cached
  versions.  This is atypical of `tf.layer` objects, which separate this
  part of layer building into a `build` method that is only called once.

  Here we provide a subclass for `RNNCell` objects that act exactly as
  `Layer` objects do.  They must provide a `build` method and their
  `call` methods do not access Variables `tf.compat.v1.get_variable`.
  """

  def __call__(self, inputs, state, scope=None, *args, **kwargs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the scope is forwarded as a keyword together with extra positional arguments,
    # so any extra positional arguments would bind to the parent's positional parameters after
    # the state rather than after the scope; the signature order is misleading.
    # Runs one cell step as a plain layer call. Inputs: input tensor, state, optional scope and
    # extra positional and keyword arguments, all passed to the base layer call directly,
    # bypassing the RNN cell scope handling. Returns the pair of output and new state.
    """Run this RNN cell on inputs, starting from the given state.

    Args:
      inputs: `2-D` tensor with shape `[batch_size, input_size]`.
      state: if `self.state_size` is an integer, this should be a `2-D Tensor`
        with shape `[batch_size, self.state_size]`.  Otherwise, if
        `self.state_size` is a tuple of integers, this should be a tuple with
        shapes `[batch_size, s] for s in self.state_size`.
      scope: optional cell scope.
      *args: Additional positional arguments.
      **kwargs: Additional keyword arguments.

    Returns:
      A pair containing:

      - Output: A `2-D` tensor with shape `[batch_size, self.output_size]`.
      - New state: Either a single `2-D` tensor, or a tuple of tensors matching
        the arity and shapes of `state`.
    """
    # Bypass RNNCell's variable capturing semantics for LayerRNNCell.
    # Instead, it is up to subclasses to provide a proper build
    # method.  See the class docstring for more details.
    return base_layer.Layer.__call__(
        self, inputs, state, scope=scope, *args, **kwargs)


@tf_export(v1=["nn.rnn_cell.BasicRNNCell"])
class BasicRNNCell(LayerRNNCell):
  # [CLASS DOC]
  # Simple recurrent cell: new state equals output equals the activation of a linear map of the
  # concatenated input and previous state. Holds the unit count, the activation (default tanh),
  # and, after build, one kernel and one bias variable. Requires 2-D inputs. Deprecated: emits a
  # warning at construction recommending the Keras simple RNN cell.
  """The most basic RNN cell.

  Note that this cell is not optimized for performance. Please use
  `tf.contrib.cudnn_rnn.CudnnRNNTanh` for better performance on GPU.

  Args:
    num_units: int, The number of units in the RNN cell.
    activation: Nonlinearity to use.  Default: `tanh`. It could also be string
      that is within Keras activation function names.
    reuse: (optional) Python boolean describing whether to reuse variables in an
      existing scope.  If not `True`, and the existing scope already has the
      given variables, an error is raised.
    name: String, the name of the layer. Layers with the same name will share
      weights, but to avoid mistakes we require reuse=True in such cases.
    dtype: Default dtype of the layer (default of `None` means use the type of
      the first input). Required when `build` is called before `call`.
    **kwargs: Dict, keyword named properties for common layer attributes, like
      `trainable` etc when constructing the cell from configs of get_config().
  """

  def __init__(self,
               num_units,
               activation=None,
               reuse=None,
               name=None,
               dtype=None,
               **kwargs):
    # [FUNCTION DOC]
    # Constructor. Inputs: number of units (integer); optional activation (callable or Keras
    # activation name, default tanh when unset); optional reuse flag passed to the parent;
    # optional name and dtype; extra layer keyword arguments. Emits a deprecation warning, checks
    # the dtype is a supported floating type (raises a value error otherwise), logs a performance
    # warning when running eagerly with a GPU present, and sets the input spec to require rank 2.
    warnings.warn("`tf.nn.rnn_cell.BasicRNNCell` is deprecated and will be "
                  "removed in a future version. This class "
                  "is equivalent as `tf.keras.layers.SimpleRNNCell`, "
                  "and will be replaced by that in Tensorflow 2.0.")
    super(BasicRNNCell, self).__init__(
        _reuse=reuse, name=name, dtype=dtype, **kwargs)
    _check_supported_dtypes(self.dtype)
    if context.executing_eagerly() and tf_config.list_logical_devices("GPU"):
      logging.warning(
          "%s: Note that this cell is not optimized for performance. "
          "Please use tf.contrib.cudnn_rnn.CudnnRNNTanh for better "
          "performance on GPU.", self)

    # Inputs must be 2-dimensional.
    self.input_spec = input_spec.InputSpec(ndim=2)

    self._num_units = num_units
    if activation:
      self._activation = activations.get(activation)
    else:
      self._activation = math_ops.tanh

  @property
  def state_size(self):
    # [FUNCTION DOC]
    # Returns the state size, which equals the number of units.
    return self._num_units

  @property
  def output_size(self):
    # [FUNCTION DOC]
    # Returns the output size, which equals the number of units.
    return self._num_units

  @tf_utils.shape_type_conversion
  def build(self, inputs_shape):
    # [FUNCTION DOC]
    # Creates the kernel (input depth plus units by units) and a zero-initialized bias (units),
    # then marks the cell built. Input: the input shape; its last dimension must be known,
    # otherwise a value error is raised. Also re-checks the dtype is supported. Side effect:
    # adds two layer variables.
    if inputs_shape[-1] is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" %
                       str(inputs_shape))
    _check_supported_dtypes(self.dtype)

    input_depth = inputs_shape[-1]
    self._kernel = self.add_variable(
        _WEIGHTS_VARIABLE_NAME,
        shape=[input_depth + self._num_units, self._num_units])
    self._bias = self.add_variable(
        _BIAS_VARIABLE_NAME,
        shape=[self._num_units],
        initializer=init_ops.zeros_initializer(dtype=self.dtype))

    self.built = True

  def call(self, inputs, state):
    # [FUNCTION DOC]
    # One step: validates that inputs and state have supported float dtypes (value error if not),
    # multiplies the concatenation of input and state by the kernel, adds the bias, applies the
    # activation. Returns the pair (output, new state) where both are the same tensor.
    """Most basic RNN: output = new_state = act(W * input + U * state + B)."""
    _check_rnn_cell_input_dtypes([inputs, state])
    gate_inputs = math_ops.matmul(
        array_ops.concat([inputs, state], 1), self._kernel)
    gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)
    output = self._activation(gate_inputs)
    return output, output

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the parent configuration merged with the unit count, the serialized activation and
    # the reuse flag (cell values win on key clashes). No side effects.
    config = {
        "num_units": self._num_units,
        "activation": activations.serialize(self._activation),
        "reuse": self._reuse,
    }
    base_config = super(BasicRNNCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf_export(v1=["nn.rnn_cell.GRUCell"])
class GRUCell(LayerRNNCell):
  # [CLASS DOC]
  # Gated recurrent unit cell with reset and update gates plus a candidate state. Holds unit
  # count, activation (default tanh), kernel and bias initializers, and after build four
  # variables (gate kernel and bias, candidate kernel and bias). Requires 2-D inputs; new state
  # equals output. Deprecated: warns at construction in favor of the Keras GRU cell.
  """Gated Recurrent Unit cell.

  Note that this cell is not optimized for performance. Please use
  `tf.contrib.cudnn_rnn.CudnnGRU` for better performance on GPU, or
  `tf.contrib.rnn.GRUBlockCellV2` for better performance on CPU.

  Args:
    num_units: int, The number of units in the GRU cell.
    activation: Nonlinearity to use.  Default: `tanh`.
    reuse: (optional) Python boolean describing whether to reuse variables in an
      existing scope.  If not `True`, and the existing scope already has the
      given variables, an error is raised.
    kernel_initializer: (optional) The initializer to use for the weight and
      projection matrices.
    bias_initializer: (optional) The initializer to use for the bias.
    name: String, the name of the layer. Layers with the same name will share
      weights, but to avoid mistakes we require reuse=True in such cases.
    dtype: Default dtype of the layer (default of `None` means use the type of
      the first input). Required when `build` is called before `call`.
    **kwargs: Dict, keyword named properties for common layer attributes, like
      `trainable` etc when constructing the cell from configs of get_config().

      References:
    Learning Phrase Representations using RNN Encoder Decoder for Statistical
    Machine Translation:
      [Cho et al., 2014]
      (https://aclanthology.coli.uni-saarland.de/papers/D14-1179/d14-1179)
      ([pdf](http://emnlp2014.org/papers/pdf/EMNLP2014179.pdf))
  """

  def __init__(self,
               num_units,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None,
               name=None,
               dtype=None,
               **kwargs):
    # [FUNCTION DOC]
    # Constructor. Inputs: unit count; optional activation (callable or name, default tanh);
    # optional reuse flag; optional kernel and bias initializers (resolved through the Keras
    # initializer lookup; unset bias means ones for gates and zeros for candidate at build time);
    # optional name, dtype and extra layer keyword arguments. Emits a deprecation warning, raises
    # a value error for unsupported dtype, logs a GPU performance warning when eager with a GPU,
    # and sets the input spec to rank 2.
    warnings.warn("`tf.nn.rnn_cell.GRUCell` is deprecated and will be removed "
                  "in a future version. This class "
                  "is equivalent as `tf.keras.layers.GRUCell`, "
                  "and will be replaced by that in Tensorflow 2.0.")
    super(GRUCell, self).__init__(
        _reuse=reuse, name=name, dtype=dtype, **kwargs)
    _check_supported_dtypes(self.dtype)

    if context.executing_eagerly() and tf_config.list_logical_devices("GPU"):
      logging.warning(
          "%s: Note that this cell is not optimized for performance. "
          "Please use tf.contrib.cudnn_rnn.CudnnGRU for better "
          "performance on GPU.", self)
    # Inputs must be 2-dimensional.
    self.input_spec = input_spec.InputSpec(ndim=2)

    self._num_units = num_units
    if activation:
      self._activation = activations.get(activation)
    else:
      self._activation = math_ops.tanh
    self._kernel_initializer = initializers.get(kernel_initializer)
    self._bias_initializer = initializers.get(bias_initializer)

  @property
  def state_size(self):
    # [FUNCTION DOC]
    # Returns the state size, which equals the number of units.
    return self._num_units

  @property
  def output_size(self):
    # [FUNCTION DOC]
    # Returns the output size, which equals the number of units.
    return self._num_units

  @tf_utils.shape_type_conversion
  def build(self, inputs_shape):
    # [FUNCTION DOC]
    # Creates the four GRU variables: gate kernel (input depth plus units by twice units), gate
    # bias (twice units, default constant one), candidate kernel and candidate bias (default
    # zero), then marks the cell built. Input: input shape whose last dimension must be known,
    # else a value error is raised; also re-checks dtype support.
    if inputs_shape[-1] is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" %
                       str(inputs_shape))
    _check_supported_dtypes(self.dtype)
    input_depth = inputs_shape[-1]
    self._gate_kernel = self.add_variable(
        "gates/%s" % _WEIGHTS_VARIABLE_NAME,
        shape=[input_depth + self._num_units, 2 * self._num_units],
        initializer=self._kernel_initializer)
    self._gate_bias = self.add_variable(
        "gates/%s" % _BIAS_VARIABLE_NAME,
        shape=[2 * self._num_units],
        initializer=(self._bias_initializer
                     if self._bias_initializer is not None else
                     init_ops.constant_initializer(1.0, dtype=self.dtype)))
    self._candidate_kernel = self.add_variable(
        "candidate/%s" % _WEIGHTS_VARIABLE_NAME,
        shape=[input_depth + self._num_units, self._num_units],
        initializer=self._kernel_initializer)
    self._candidate_bias = self.add_variable(
        "candidate/%s" % _BIAS_VARIABLE_NAME,
        shape=[self._num_units],
        initializer=(self._bias_initializer
                     if self._bias_initializer is not None else
                     init_ops.zeros_initializer(dtype=self.dtype)))

    self.built = True

  def call(self, inputs, state):
    # [FUNCTION DOC]
    # One GRU step. Validates dtypes of inputs and state, computes sigmoid gates from the
    # concatenated input and state, splits them into reset and update parts, computes the
    # candidate from the input and the reset-scaled state through the activation, and blends
    # old state and candidate by the update gate. Returns the new state twice (output and state).
    """Gated recurrent unit (GRU) with nunits cells."""
    _check_rnn_cell_input_dtypes([inputs, state])

    gate_inputs = math_ops.matmul(
        array_ops.concat([inputs, state], 1), self._gate_kernel)
    gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)

    value = math_ops.sigmoid(gate_inputs)
    r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

    r_state = r * state

    candidate = math_ops.matmul(
        array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
    candidate = nn_ops.bias_add(candidate, self._candidate_bias)

    c = self._activation(candidate)
    new_h = u * state + (1 - u) * c
    return new_h, new_h

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the parent configuration merged with the unit count, serialized kernel and bias
    # initializers, serialized activation and the reuse flag (cell values win on key clashes).
    config = {
        "num_units": self._num_units,
        "kernel_initializer": initializers.serialize(self._kernel_initializer),
        "bias_initializer": initializers.serialize(self._bias_initializer),
        "activation": activations.serialize(self._activation),
        "reuse": self._reuse,
    }
    base_config = super(GRUCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


_LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h"))


@tf_export(v1=["nn.rnn_cell.LSTMStateTuple"])
class LSTMStateTuple(_LSTMStateTuple):
  # [CLASS DOC]
  # Immutable two-element named tuple (cell state, then hidden/output state) used as the state
  # and state size of LSTM cells when tuple states are enabled. Adds a dtype property and no
  # instance dictionary. Exported in the v1 namespace.
  """Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.

  Stores two elements: `(c, h)`, in that order. Where `c` is the hidden state
  and `h` is the output.

  Only used when `state_is_tuple=True`.
  """
  __slots__ = ()

  @property
  def dtype(self):
    # [FUNCTION DOC]
    # Property returning the dtype shared by the two state tensors. Raises a type error if the
    # cell state and hidden state dtypes differ.
    (c, h) = self
    if c.dtype != h.dtype:
      raise TypeError("Inconsistent internal state: %s vs %s" %
                      (str(c.dtype), str(h.dtype)))
    return c.dtype


@tf_export(v1=["nn.rnn_cell.BasicLSTMCell"])
class BasicLSTMCell(LayerRNNCell):
  # [CLASS DOC]
  # KNOWN ISSUE: the docstring sentence "The implementation is based on" is truncated; the
  # reference it should cite is missing.
  # Baseline LSTM cell without peepholes, clipping or projection. State is either a pair of cell
  # and hidden tensors or one concatenated tensor, depending on the tuple flag. Holds unit count,
  # forget bias, activation (default tanh) and, after build, one fused kernel and bias covering
  # all four gates. Requires 2-D inputs. Deprecated in favor of the full LSTM cell; warns at
  # construction.
  """DEPRECATED: Please use `tf.compat.v1.nn.rnn_cell.LSTMCell` instead.

  Basic LSTM recurrent network cell.

  The implementation is based on

  We add forget_bias (default: 1) to the biases of the forget gate in order to
  reduce the scale of forgetting in the beginning of the training.

  It does not allow cell clipping, a projection layer, and does not
  use peep-hole connections: it is the basic baseline.

  For advanced models, please use the full `tf.compat.v1.nn.rnn_cell.LSTMCell`
  that follows.

  Note that this cell is not optimized for performance. Please use
  `tf.contrib.cudnn_rnn.CudnnLSTM` for better performance on GPU, or
  `tf.contrib.rnn.LSTMBlockCell` and `tf.contrib.rnn.LSTMBlockFusedCell` for
  better performance on CPU.
  """

  def __init__(self,
               num_units,
               forget_bias=1.0,
               state_is_tuple=True,
               activation=None,
               reuse=None,
               name=None,
               dtype=None,
               **kwargs):
    # [FUNCTION DOC]
    # Constructor. Inputs: unit count; forget bias (default one, added to the forget gate);
    # tuple-state flag (default true); optional activation (callable or name, default tanh);
    # optional reuse flag, name, dtype and extra layer keyword arguments. Emits a deprecation
    # warning, raises a value error for unsupported dtype, logs warnings for concatenated states
    # and for eager execution with a GPU, and sets the input spec to rank 2.
    """Initialize the basic LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (see above). Must set
        to `0.0` manually when restoring from CudnnLSTM-trained checkpoints.
      state_is_tuple: If True, accepted and returned states are 2-tuples of the
        `c_state` and `m_state`.  If False, they are concatenated along the
        column axis.  The latter behavior will soon be deprecated.
      activation: Activation function of the inner states.  Default: `tanh`. It
        could also be string that is within Keras activation function names.
      reuse: (optional) Python boolean describing whether to reuse variables in
        an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
      name: String, the name of the layer. Layers with the same name will share
        weights, but to avoid mistakes we require reuse=True in such cases.
      dtype: Default dtype of the layer (default of `None` means use the type of
        the first input). Required when `build` is called before `call`.
      **kwargs: Dict, keyword named properties for common layer attributes, like
        `trainable` etc when constructing the cell from configs of get_config().
        When restoring from CudnnLSTM-trained checkpoints, must use
        `CudnnCompatibleLSTMCell` instead.
    """
    warnings.warn("`tf.nn.rnn_cell.BasicLSTMCell` is deprecated and will be "
                  "removed in a future version. This class "
                  "is equivalent as `tf.keras.layers.LSTMCell`, "
                  "and will be replaced by that in Tensorflow 2.0.")
    super(BasicLSTMCell, self).__init__(
        _reuse=reuse, name=name, dtype=dtype, **kwargs)
    _check_supported_dtypes(self.dtype)
    if not state_is_tuple:
      logging.warning(
          "%s: Using a concatenated state is slower and will soon be "
          "deprecated.  Use state_is_tuple=True.", self)
    if context.executing_eagerly() and tf_config.list_logical_devices("GPU"):
      logging.warning(
          "%s: Note that this cell is not optimized for performance. "
          "Please use tf.contrib.cudnn_rnn.CudnnLSTM for better "
          "performance on GPU.", self)

    # Inputs must be 2-dimensional.
    self.input_spec = input_spec.InputSpec(ndim=2)

    self._num_units = num_units
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    if activation:
      self._activation = activations.get(activation)
    else:
      self._activation = math_ops.tanh

  @property
  def state_size(self):
    # [FUNCTION DOC]
    # Returns the state size: a pair of two unit counts when tuple states are used, otherwise
    # twice the unit count.
    return (LSTMStateTuple(self._num_units, self._num_units)
            if self._state_is_tuple else 2 * self._num_units)

  @property
  def output_size(self):
    # [FUNCTION DOC]
    # Returns the output size, which equals the number of units.
    return self._num_units

  @tf_utils.shape_type_conversion
  def build(self, inputs_shape):
    # [FUNCTION DOC]
    # Creates one fused kernel (input depth plus units by four times units) and a zero-initialized
    # bias (four times units) for all gates, then marks the cell built. Input: input shape whose
    # last dimension must be known, else a value error is raised; re-checks dtype support.
    if inputs_shape[-1] is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" %
                       str(inputs_shape))
    _check_supported_dtypes(self.dtype)
    input_depth = inputs_shape[-1]
    h_depth = self._num_units
    self._kernel = self.add_variable(
        _WEIGHTS_VARIABLE_NAME,
        shape=[input_depth + h_depth, 4 * self._num_units])
    self._bias = self.add_variable(
        _BIAS_VARIABLE_NAME,
        shape=[4 * self._num_units],
        initializer=init_ops.zeros_initializer(dtype=self.dtype))

    self.built = True

  def call(self, inputs, state):
    # [FUNCTION DOC]
    # One LSTM step. Inputs: 2-D input tensor and the state (pair of cell and hidden tensors, or
    # one concatenated tensor split in half). Validates float dtypes (value error otherwise),
    # computes the four gate pre-activations from the input and previous hidden state, applies
    # the forget bias, and produces new cell state and new hidden state. Returns the new hidden
    # state and the new state (pair or concatenation, matching the tuple flag).
    """Long short-term memory cell (LSTM).

    Args:
      inputs: `2-D` tensor with shape `[batch_size, input_size]`.
      state: An `LSTMStateTuple` of state tensors, each shaped `[batch_size,
        num_units]`, if `state_is_tuple` has been set to `True`.  Otherwise, a
        `Tensor` shaped `[batch_size, 2 * num_units]`.

    Returns:
      A pair containing the new hidden state, and the new state (either a
        `LSTMStateTuple` or a concatenated state, depending on
        `state_is_tuple`).
    """
    _check_rnn_cell_input_dtypes([inputs, state])

    sigmoid = math_ops.sigmoid
    one = constant_op.constant(1, dtype=dtypes.int32)
    # Parameters of gates are concatenated into one multiply for efficiency.
    if self._state_is_tuple:
      c, h = state
    else:
      c, h = array_ops.split(value=state, num_or_size_splits=2, axis=one)

    gate_inputs = math_ops.matmul(
        array_ops.concat([inputs, h], 1), self._kernel)
    gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)

    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    i, j, f, o = array_ops.split(
        value=gate_inputs, num_or_size_splits=4, axis=one)

    forget_bias_tensor = constant_op.constant(self._forget_bias, dtype=f.dtype)
    # Note that using `add` and `multiply` instead of `+` and `*` gives a
    # performance improvement. So using those at the cost of readability.
    add = math_ops.add
    multiply = math_ops.multiply
    new_c = add(
        multiply(c, sigmoid(add(f, forget_bias_tensor))),
        multiply(sigmoid(i), self._activation(j)))
    new_h = multiply(self._activation(new_c), sigmoid(o))

    if self._state_is_tuple:
      new_state = LSTMStateTuple(new_c, new_h)
    else:
      new_state = array_ops.concat([new_c, new_h], 1)
    return new_h, new_state

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the parent configuration merged with the unit count, forget bias, tuple-state flag,
    # serialized activation and reuse flag (cell values win on key clashes).
    config = {
        "num_units": self._num_units,
        "forget_bias": self._forget_bias,
        "state_is_tuple": self._state_is_tuple,
        "activation": activations.serialize(self._activation),
        "reuse": self._reuse,
    }
    base_config = super(BasicLSTMCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf_export(v1=["nn.rnn_cell.LSTMCell"])
class LSTMCell(LayerRNNCell):
  # [CLASS DOC]
  # Full LSTM cell with optional peephole connections, cell-state clipping and an output
  # projection layer, plus optional variable partitioning via the deprecated shard counts. Holds
  # all those settings, the forget bias, activation and initializer, computes state and output
  # sizes at construction (projection size replaces the hidden size when projection is on), and
  # after build holds fused kernel, bias, optional three peephole vectors and optional projection
  # kernel. Requires 2-D inputs; deprecated in favor of the Keras LSTM cell.
  """Long short-term memory unit (LSTM) recurrent network cell.

  The default non-peephole implementation is based on (Gers et al., 1999).
  The peephole implementation is based on (Sak et al., 2014).

  The class uses optional peep-hole connections, optional cell clipping, and
  an optional projection layer.

  Note that this cell is not optimized for performance. Please use
  `tf.contrib.cudnn_rnn.CudnnLSTM` for better performance on GPU, or
  `tf.contrib.rnn.LSTMBlockCell` and `tf.contrib.rnn.LSTMBlockFusedCell` for
  better performance on CPU.
  References:
    Long short-term memory recurrent neural network architectures for large
    scale acoustic modeling:
      [Sak et al., 2014]
      (https://www.isca-speech.org/archive/interspeech_2014/i14_0338.html)
      ([pdf]
      (https://www.isca-speech.org/archive/archive_papers/interspeech_2014/i14_0338.pdf))
    Learning to forget:
      [Gers et al., 1999]
      (http://digital-library.theiet.org/content/conferences/10.1049/cp_19991218)
      ([pdf](https://arxiv.org/pdf/1409.2329.pdf))
    Long Short-Term Memory:
      [Hochreiter et al., 1997]
      (https://www.mitpressjournals.org/doi/abs/10.1162/neco.1997.9.8.1735)
      ([pdf](http://ml.jku.at/publications/older/3504.pdf))
  """

  def __init__(self,
               num_units,
               use_peepholes=False,
               cell_clip=None,
               initializer=None,
               num_proj=None,
               proj_clip=None,
               num_unit_shards=None,
               num_proj_shards=None,
               forget_bias=1.0,
               state_is_tuple=True,
               activation=None,
               reuse=None,
               name=None,
               dtype=None,
               **kwargs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the shard-deprecation warning text mentions a parameter named proj_unit_shards
    # while the real parameter is num_proj_shards, and the deprecation date it cites is long past.
    # Constructor. Inputs: unit count; peephole flag; optional cell clip value; optional
    # initializer (resolved through the Keras lookup); optional projection size and projection
    # clip; deprecated unit and projection shard counts; forget bias (default one); tuple-state
    # flag; optional activation (default tanh); optional reuse, name, dtype and extra layer
    # keyword arguments. Emits a deprecation warning, raises a value error for unsupported dtype,
    # logs warnings for concatenated states, shard counts and eager GPU use, sets the input spec
    # to rank 2, and derives the state size and output size.
    """Initialize the parameters for an LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell.
      use_peepholes: bool, set True to enable diagonal/peephole connections.
      cell_clip: (optional) A float value, if provided the cell state is clipped
        by this value prior to the cell output activation.
      initializer: (optional) The initializer to use for the weight and
        projection matrices.
      num_proj: (optional) int, The output dimensionality for the projection
        matrices.  If None, no projection is performed.
      proj_clip: (optional) A float value.  If `num_proj > 0` and `proj_clip` is
        provided, then the projected values are clipped elementwise to within
        `[-proj_clip, proj_clip]`.
      num_unit_shards: Deprecated, will be removed by Jan. 2017. Use a
        variable_scope partitioner instead.
      num_proj_shards: Deprecated, will be removed by Jan. 2017. Use a
        variable_scope partitioner instead.
      forget_bias: Biases of the forget gate are initialized by default to 1 in
        order to reduce the scale of forgetting at the beginning of the
        training. Must set it manually to `0.0` when restoring from CudnnLSTM
        trained checkpoints.
      state_is_tuple: If True, accepted and returned states are 2-tuples of the
        `c_state` and `m_state`.  If False, they are concatenated along the
        column axis.  This latter behavior will soon be deprecated.
      activation: Activation function of the inner states.  Default: `tanh`. It
        could also be string that is within Keras activation function names.
      reuse: (optional) Python boolean describing whether to reuse variables in
        an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
      name: String, the name of the layer. Layers with the same name will share
        weights, but to avoid mistakes we require reuse=True in such cases.
      dtype: Default dtype of the layer (default of `None` means use the type of
        the first input). Required when `build` is called before `call`.
      **kwargs: Dict, keyword named properties for common layer attributes, like
        `trainable` etc when constructing the cell from configs of get_config().
        When restoring from CudnnLSTM-trained checkpoints, use
        `CudnnCompatibleLSTMCell` instead.
    """
    warnings.warn("`tf.nn.rnn_cell.LSTMCell` is deprecated and will be "
                  "removed in a future version. This class "
                  "is equivalent as `tf.keras.layers.LSTMCell`, "
                  "and will be replaced by that in Tensorflow 2.0.")
    super(LSTMCell, self).__init__(
        _reuse=reuse, name=name, dtype=dtype, **kwargs)
    _check_supported_dtypes(self.dtype)
    if not state_is_tuple:
      logging.warning(
          "%s: Using a concatenated state is slower and will soon be "
          "deprecated.  Use state_is_tuple=True.", self)
    if num_unit_shards is not None or num_proj_shards is not None:
      logging.warning(
          "%s: The num_unit_shards and proj_unit_shards parameters are "
          "deprecated and will be removed in Jan 2017.  "
          "Use a variable scope with a partitioner instead.", self)
    if context.executing_eagerly() and tf_config.list_logical_devices("GPU"):
      logging.warning(
          "%s: Note that this cell is not optimized for performance. "
          "Please use tf.contrib.cudnn_rnn.CudnnLSTM for better "
          "performance on GPU.", self)

    # Inputs must be 2-dimensional.
    self.input_spec = input_spec.InputSpec(ndim=2)

    self._num_units = num_units
    self._use_peepholes = use_peepholes
    self._cell_clip = cell_clip
    self._initializer = initializers.get(initializer)
    self._num_proj = num_proj
    self._proj_clip = proj_clip
    self._num_unit_shards = num_unit_shards
    self._num_proj_shards = num_proj_shards
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    if activation:
      self._activation = activations.get(activation)
    else:
      self._activation = math_ops.tanh

    if num_proj:
      self._state_size = (
          LSTMStateTuple(num_units, num_proj) if state_is_tuple else num_units +
          num_proj)
      self._output_size = num_proj
    else:
      self._state_size = (
          LSTMStateTuple(num_units, num_units) if state_is_tuple else 2 *
          num_units)
      self._output_size = num_units

  @property
  def state_size(self):
    # [FUNCTION DOC]
    # Returns the state size computed at construction (pair of cell size and projection or hidden
    # size, or their sum when states are concatenated).
    return self._state_size

  @property
  def output_size(self):
    # [FUNCTION DOC]
    # Returns the output size computed at construction: the projection size if set, else units.
    return self._output_size

  @tf_utils.shape_type_conversion
  def build(self, inputs_shape):
    # [FUNCTION DOC]
    # Creates the fused kernel (optionally partitioned by the unit shard count), the zero bias,
    # the three diagonal peephole vectors when peepholes are on, and the projection kernel
    # (optionally partitioned) when a projection size is set; then marks the cell built. Input:
    # input shape whose last dimension must be known, else a value error is raised.
    if inputs_shape[-1] is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" %
                       str(inputs_shape))
    _check_supported_dtypes(self.dtype)
    input_depth = inputs_shape[-1]
    h_depth = self._num_units if self._num_proj is None else self._num_proj
    maybe_partitioner = (
        partitioned_variables.fixed_size_partitioner(self._num_unit_shards)
        if self._num_unit_shards is not None else None)
    self._kernel = self.add_variable(
        _WEIGHTS_VARIABLE_NAME,
        shape=[input_depth + h_depth, 4 * self._num_units],
        initializer=self._initializer,
        partitioner=maybe_partitioner)
    if self.dtype is None:
      initializer = init_ops.zeros_initializer
    else:
      initializer = init_ops.zeros_initializer(dtype=self.dtype)
    self._bias = self.add_variable(
        _BIAS_VARIABLE_NAME,
        shape=[4 * self._num_units],
        initializer=initializer)
    if self._use_peepholes:
      self._w_f_diag = self.add_variable(
          "w_f_diag", shape=[self._num_units], initializer=self._initializer)
      self._w_i_diag = self.add_variable(
          "w_i_diag", shape=[self._num_units], initializer=self._initializer)
      self._w_o_diag = self.add_variable(
          "w_o_diag", shape=[self._num_units], initializer=self._initializer)

    if self._num_proj is not None:
      maybe_proj_partitioner = (
          partitioned_variables.fixed_size_partitioner(self._num_proj_shards)
          if self._num_proj_shards is not None else None)
      self._proj_kernel = self.add_variable(
          "projection/%s" % _WEIGHTS_VARIABLE_NAME,
          shape=[self._num_units, self._num_proj],
          initializer=self._initializer,
          partitioner=maybe_proj_partitioner)

    self.built = True

  def call(self, inputs, state):
    # [FUNCTION DOC]
    # One LSTM step with optional peepholes, cell clipping and projection. Inputs: 2-D input
    # tensor and the state (pair of cell and projected-hidden tensors, or one concatenated tensor
    # that is sliced). Validates float dtypes; raises a value error if the static input size
    # cannot be inferred. Computes gates, updates the cell state (clipped if configured), computes
    # the output, applies the projection and its optional clip. Returns the output and the new
    # state (pair or concatenation, matching the tuple flag).
    """Run one step of LSTM.

    Args:
      inputs: input Tensor, must be 2-D, `[batch, input_size]`.
      state: if `state_is_tuple` is False, this must be a state Tensor, `2-D,
        [batch, state_size]`.  If `state_is_tuple` is True, this must be a tuple
        of state Tensors, both `2-D`, with column sizes `c_state` and `m_state`.

    Returns:
      A tuple containing:

      - A `2-D, [batch, output_dim]`, Tensor representing the output of the
        LSTM after reading `inputs` when previous state was `state`.
        Here output_dim is:
           num_proj if num_proj was set,
           num_units otherwise.
      - Tensor(s) representing the new state of LSTM after reading `inputs` when
        the previous state was `state`.  Same type and shape(s) as `state`.

    Raises:
      ValueError: If input size cannot be inferred from inputs via
        static shape inference.
    """
    _check_rnn_cell_input_dtypes([inputs, state])

    num_proj = self._num_units if self._num_proj is None else self._num_proj
    sigmoid = math_ops.sigmoid

    if self._state_is_tuple:
      (c_prev, m_prev) = state
    else:
      c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
      m_prev = array_ops.slice(state, [0, self._num_units], [-1, num_proj])

    input_size = inputs.get_shape().with_rank(2).dims[1].value
    if input_size is None:
      raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    lstm_matrix = math_ops.matmul(
        array_ops.concat([inputs, m_prev], 1), self._kernel)
    lstm_matrix = nn_ops.bias_add(lstm_matrix, self._bias)

    i, j, f, o = array_ops.split(
        value=lstm_matrix, num_or_size_splits=4, axis=1)
    # Diagonal connections
    if self._use_peepholes:
      c = (
          sigmoid(f + self._forget_bias + self._w_f_diag * c_prev) * c_prev +
          sigmoid(i + self._w_i_diag * c_prev) * self._activation(j))
    else:
      c = (
          sigmoid(f + self._forget_bias) * c_prev +
          sigmoid(i) * self._activation(j))

    if self._cell_clip is not None:
      # pylint: disable=invalid-unary-operand-type
      c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)
      # pylint: enable=invalid-unary-operand-type
    if self._use_peepholes:
      m = sigmoid(o + self._w_o_diag * c) * self._activation(c)
    else:
      m = sigmoid(o) * self._activation(c)

    if self._num_proj is not None:
      m = math_ops.matmul(m, self._proj_kernel)

      if self._proj_clip is not None:
        # pylint: disable=invalid-unary-operand-type
        m = clip_ops.clip_by_value(m, -self._proj_clip, self._proj_clip)
        # pylint: enable=invalid-unary-operand-type

    new_state = (
        LSTMStateTuple(c, m)
        if self._state_is_tuple else array_ops.concat([c, m], 1))
    return m, new_state

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the parent configuration merged with all constructor settings of this cell (units,
    # peepholes, clips, serialized initializer, projection, shard counts, forget bias, tuple flag,
    # serialized activation, reuse). Cell values win on key clashes.
    config = {
        "num_units": self._num_units,
        "use_peepholes": self._use_peepholes,
        "cell_clip": self._cell_clip,
        "initializer": initializers.serialize(self._initializer),
        "num_proj": self._num_proj,
        "proj_clip": self._proj_clip,
        "num_unit_shards": self._num_unit_shards,
        "num_proj_shards": self._num_proj_shards,
        "forget_bias": self._forget_bias,
        "state_is_tuple": self._state_is_tuple,
        "activation": activations.serialize(self._activation),
        "reuse": self._reuse,
    }
    base_config = super(LSTMCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class _RNNCellWrapperV1(RNNCell):
  # [CLASS DOC]
  # HELPER: internal base class for the public V1 cell wrappers (dropout, residual, device).
  # Legacy RNN cell that wraps another cell held in the cell attribute (also tracked as a
  # trackable dependency when possible). Subclasses implement the wrapping step helper; this
  # class routes calls to it with the wrapped cell's call entry point and provides config
  # serialization. Counterpart of the V2 wrapper base in the Keras wrapper module.
  """Base class for cells wrappers V1 compatibility.

  This class along with `_RNNCellWrapperV2` allows to define cells wrappers that
  are compatible with V1 and V2, and defines helper methods for this purpose.
  """

  def __init__(self, cell, *args, **kwargs):
    # [FUNCTION DOC]
    # Constructor. Inputs: the cell to wrap plus extra arguments forwarded to the parent
    # constructor. Validates that the cell looks like an RNN cell (raises a type error if not),
    # stores it, and registers it as a trackable dependency named cell when it is trackable.
    super(_RNNCellWrapperV1, self).__init__(*args, **kwargs)
    assert_like_rnncell("cell", cell)
    self.cell = cell
    if isinstance(cell, trackable.Trackable):
      self._track_trackable(self.cell, name="cell")

  def _call_wrapped_cell(self, inputs, state, cell_call_fn, **kwargs):
    # [FUNCTION DOC]
    # HELPER: abstract hook for subclasses; implements the wrapper-specific logic around one step
    # of the wrapped cell. Inputs: input tensor, state, the function to call the wrapped cell
    # with, and extra keyword arguments. Intended to return the pair of output and new state.
    # This base version always raises a not-implemented error.
    """Calls the wrapped cell and performs the wrapping logic.

    This method is called from the wrapper's `call` or `__call__` methods.

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
    raise NotImplementedError

  def __call__(self, inputs, state, scope=None):
    # [FUNCTION DOC]
    # Runs one wrapped step using the wrapped cell's own call entry point (so the cell builds
    # itself on first use). Inputs: input tensor, state, optional variable scope forwarded to the
    # wrapped cell. Returns the pair of output and new state from the wrapping helper.
    """Runs the RNN cell step computation.

    We assume that the wrapped RNNCell is being built within its `__call__`
    method. We directly use the wrapped cell's `__call__` in the overridden
    wrapper `__call__` method.

    This allows to use the wrapped cell and the non-wrapped cell equivalently
    when using `__call__`.

    Args:
      inputs: A tensor with wrapped cell's input.
      state: A tensor or tuple of tensors with wrapped cell's state.
      scope: VariableScope for the subgraph created in the wrapped cells'
        `__call__`.

    Returns:
      A pair containing:

      - Output: A tensor with cell's output.
      - New state: A tensor or tuple of tensors with new wrapped cell's state.
    """
    return self._call_wrapped_cell(
        inputs, state, cell_call_fn=self.cell.__call__, scope=scope)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the parent configuration merged with an entry for the wrapped cell holding its
    # class name and its own configuration. No side effects.
    config = {
        "cell": {
            "class_name": self.cell.__class__.__name__,
            "config": self.cell.get_config()
        },
    }
    base_config = super(_RNNCellWrapperV1, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: unlike the V2 wrapper, this never deserializes the stored cell configuration;
    # the custom objects argument is ignored, and a plain mapping in the cell entry always fails
    # the cell-likeness check, so only configs whose cell entry was replaced by a real cell
    # object can be restored.
    # Class method. Copies the config (input not mutated), removes the cell entry, checks it looks
    # like an RNN cell and constructs the wrapper from it and the remaining entries. Any type
    # error is converted into a value error telling the caller to put a cell instance in the
    # config. Returns the new wrapper.
    config = config.copy()
    cell = config.pop("cell")
    try:
      assert_like_rnncell("cell", cell)
      return cls(cell, **config)
    except TypeError:
      raise ValueError("RNNCellWrapper cannot reconstruct the wrapped cell. "
                       "Please overwrite the cell in the config with a RNNCell "
                       "instance.")


@tf_export(v1=["nn.rnn_cell.DropoutWrapper"])
class DropoutWrapper(rnn_cell_wrapper_impl.DropoutWrapperBase,
                     _RNNCellWrapperV1):
  # [CLASS DOC]
  # Public V1 wrapper (v1 nn.rnn_cell namespace) adding dropout to the input, state and output of
  # the wrapped cell. Behavior comes from the shared dropout base in the wrapper implementation
  # module; the V1 base supplies the legacy cell plumbing.
  """Operator adding dropout to inputs and outputs of the given cell."""

  def __init__(self, *args, **kwargs):  # pylint: disable=useless-super-delegation
    # [FUNCTION DOC]
    # Constructor that forwards all positional and keyword arguments (the cell and the dropout
    # settings) to the base classes. Returns nothing; its documentation text is copied from the
    # shared dropout base class.
    super(DropoutWrapper, self).__init__(*args, **kwargs)

  __init__.__doc__ = rnn_cell_wrapper_impl.DropoutWrapperBase.__init__.__doc__


@tf_export(v1=["nn.rnn_cell.ResidualWrapper"])
class ResidualWrapper(rnn_cell_wrapper_impl.ResidualWrapperBase,
                      _RNNCellWrapperV1):
  # [CLASS DOC]
  # Public V1 wrapper that adds the cell input to the cell output (residual connection). Behavior
  # comes from the shared residual base in the wrapper implementation module.
  """RNNCell wrapper that ensures cell inputs are added to the outputs."""

  def __init__(self, *args, **kwargs):  # pylint: disable=useless-super-delegation
    # [FUNCTION DOC]
    # Constructor that forwards all positional and keyword arguments (the cell and optional
    # residual function) to the base classes. Returns nothing; documentation text is copied from
    # the shared residual base class.
    super(ResidualWrapper, self).__init__(*args, **kwargs)

  __init__.__doc__ = rnn_cell_wrapper_impl.ResidualWrapperBase.__init__.__doc__


@tf_export(v1=["nn.rnn_cell.DeviceWrapper"])
class DeviceWrapper(rnn_cell_wrapper_impl.DeviceWrapperBase,
                    _RNNCellWrapperV1):
  # [CLASS DOC]
  # Public V1 wrapper that runs the wrapped cell under a given device scope. Behavior comes from
  # the shared device base in the wrapper implementation module. It has no docstring of its own.

  def __init__(self, *args, **kwargs):  # pylint: disable=useless-super-delegation
    # [FUNCTION DOC]
    # Constructor that forwards all positional and keyword arguments (the cell and the device
    # specification) to the base classes. Returns nothing; documentation text is copied from the
    # shared device base class.
    super(DeviceWrapper, self).__init__(*args, **kwargs)

  __init__.__doc__ = rnn_cell_wrapper_impl.DeviceWrapperBase.__init__.__doc__


@tf_export(v1=["nn.rnn_cell.MultiRNNCell"])
class MultiRNNCell(RNNCell):
  # [CLASS DOC]
  # Stacks several RNN cells so that the output of each is the input of the next. Holds the list
  # of cells (tracked as dependencies) and the tuple-state flag. The state is a tuple with one
  # entry per cell, or all states concatenated along columns when the flag is off. Aggregates
  # weights from the cells. Deprecated in favor of the Keras stacked cells; logs a warning.
  """RNN cell composed sequentially of multiple simple cells.

  Example:

  ```python
  num_units = [128, 64]
  cells = [BasicLSTMCell(num_units=n) for n in num_units]
  stacked_rnn_cell = MultiRNNCell(cells)
  ```
  """

  def __init__(self, cells, state_is_tuple=True):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring lists only a value error, but a type error is also raised when
    # the cells argument is not a list or tuple (nested structure).
    # Constructor. Inputs: a non-empty list or tuple of cells; tuple-state flag (default true).
    # Logs a deprecation warning, raises a value error for empty cells, a type error for a
    # non-nested argument, logs once if the same cell object appears twice (shared weights),
    # registers each trackable cell as a dependency named by its position, and, when the flag is
    # off, raises a value error if any cell has a nested (tuple) state size.
    """Create a RNN cell composed sequentially of a number of RNNCells.

    Args:
      cells: list of RNNCells that will be composed in this order.
      state_is_tuple: If True, accepted and returned states are n-tuples, where
        `n = len(cells)`.  If False, the states are all concatenated along the
        column axis.  This latter behavior will soon be deprecated.

    Raises:
      ValueError: if cells is empty (not allowed), or at least one of the cells
        returns a state tuple but the flag `state_is_tuple` is `False`.
    """
    logging.warning("`tf.nn.rnn_cell.MultiRNNCell` is deprecated. This class "
                    "is equivalent as `tf.keras.layers.StackedRNNCells`, "
                    "and will be replaced by that in Tensorflow 2.0.")
    super(MultiRNNCell, self).__init__()
    if not cells:
      raise ValueError("Must specify at least one cell for MultiRNNCell.")
    if not nest.is_nested(cells):
      raise TypeError("cells must be a list or tuple, but saw: %s." % cells)

    if len(set(id(cell) for cell in cells)) < len(cells):
      logging.log_first_n(
          logging.WARN, "At least two cells provided to MultiRNNCell "
          "are the same object and will share weights.", 1)

    self._cells = cells
    for cell_number, cell in enumerate(self._cells):
      # Add Trackable dependencies on these cells so their variables get
      # saved with this object when using object-based saving.
      if isinstance(cell, trackable.Trackable):
        # TODO(allenl): Track down non-Trackable callers.
        self._track_trackable(cell, name="cell-%d" % (cell_number,))
    self._state_is_tuple = state_is_tuple
    if not state_is_tuple:
      if any(nest.is_nested(c.state_size) for c in self._cells):
        raise ValueError("Some cells return tuples of states, but the flag "
                         "state_is_tuple is not set.  State sizes are: %s" %
                         str([c.state_size for c in self._cells]))

  @property
  def state_size(self):
    # [FUNCTION DOC]
    # Returns a tuple of the cells' state sizes when tuple states are used, otherwise the sum of
    # the cells' state sizes.
    if self._state_is_tuple:
      return tuple(cell.state_size for cell in self._cells)
    else:
      return sum(cell.state_size for cell in self._cells)

  @property
  def output_size(self):
    # [FUNCTION DOC]
    # Returns the output size of the last cell in the stack.
    return self._cells[-1].output_size

  def zero_state(self, batch_size, dtype):
    # [FUNCTION DOC]
    # Builds the zero state inside a name scope named after the class. With tuple states, returns
    # a tuple of each cell's own zero state for the given batch size and dtype; otherwise uses the
    # parent implementation on the summed state size. Does not use the parent's eager cache in
    # the tuple case.
    with backend.name_scope(type(self).__name__ + "ZeroState"):
      if self._state_is_tuple:
        return tuple(cell.zero_state(batch_size, dtype) for cell in self._cells)
      else:
        # We know here that state_size of each cell is not a tuple and
        # presumably does not contain TensorArrays or anything else fancy
        return super(MultiRNNCell, self).zero_state(batch_size, dtype)

  @property
  def trainable_weights(self):
    # [FUNCTION DOC]
    # Property returning the concatenated trainable weights of all cells that are layers, or an
    # empty list when this stacked cell itself is not trainable. Note that weights tracked on the
    # stacked cell itself are not included.
    if not self.trainable:
      return []
    weights = []
    for cell in self._cells:
      if isinstance(cell, base_layer.Layer):
        weights += cell.trainable_weights
    return weights

  @property
  def non_trainable_weights(self):
    # [FUNCTION DOC]
    # Property returning the concatenated non-trainable weights of all cells that are layers.
    # When this stacked cell is marked not trainable, the cells' trainable weights are also
    # reported here (placed first), since they are frozen.
    weights = []
    for cell in self._cells:
      if isinstance(cell, base_layer.Layer):
        weights += cell.non_trainable_weights
    if not self.trainable:
      trainable_weights = []
      for cell in self._cells:
        if isinstance(cell, base_layer.Layer):
          trainable_weights += cell.trainable_weights
      return trainable_weights + weights
    return weights

  def call(self, inputs, state):
    # [FUNCTION DOC]
    # Runs one step through all cells in order, each inside a variable scope named by its
    # position. Inputs: the input tensor and the state (tuple with one entry per cell, or one
    # concatenated tensor that is sliced by each cell's state size). Raises a value error if
    # tuple states are expected but the state is not nested. Returns the last cell's output and
    # the new states (tuple, or concatenated along columns).
    """Run this multi-layer cell on inputs, starting from state."""
    cur_state_pos = 0
    cur_inp = inputs
    new_states = []
    for i, cell in enumerate(self._cells):
      with vs.variable_scope("cell_%d" % i):
        if self._state_is_tuple:
          if not nest.is_nested(state):
            raise ValueError(
                "Expected state to be a tuple of length %d, but received: %s" %
                (len(self.state_size), state))
          cur_state = state[i]
        else:
          cur_state = array_ops.slice(state, [0, cur_state_pos],
                                      [-1, cell.state_size])
          cur_state_pos += cell.state_size
        cur_inp, new_state = cell(cur_inp, cur_state)
        new_states.append(new_state)

    new_states = (
        tuple(new_states) if self._state_is_tuple else array_ops.concat(
            new_states, 1))

    return cur_inp, new_states


def _check_rnn_cell_input_dtypes(inputs):
  # [FUNCTION DOC]
  # HELPER: dtype guard used by the cells' call methods. Input: a tensor or nested structure of
  # tensors (inputs and state). Flattens it and checks each dtype with the supported-dtype
  # check, so a value error is raised if any is not floating or complex. Returns nothing.
  """Check whether the input tensors are with supported dtypes.

  Default RNN cells only support floats and complex as its dtypes since the
  activation function (tanh and sigmoid) only allow those types. This function
  will throw a proper error message if the inputs is not in a supported type.

  Args:
    inputs: tensor or nested structure of tensors that are feed to RNN cell as
      input or state.

  Raises:
    ValueError: if any of the input tensor are not having dtypes of float or
      complex.
  """
  for t in nest.flatten(inputs):
    _check_supported_dtypes(t.dtype)


def _check_supported_dtypes(dtype):
  # [FUNCTION DOC]
  # HELPER: validates a cell dtype. Input: a dtype or dtype-like value, or nothing/null (which
  # is accepted silently). Converts it to a dtype and raises a value error unless it is floating
  # point or complex. Returns nothing.
  if dtype is None:
    return
  dtype = dtypes.as_dtype(dtype)
  if not (dtype.is_floating or dtype.is_complex):
    raise ValueError("RNN cell only supports floating point inputs, "
                     "but saw dtype: %s" % dtype)
