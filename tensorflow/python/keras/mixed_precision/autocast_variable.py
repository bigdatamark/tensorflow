# [MODULE DOC]
# Defines AutoCastVariable, a wrapper around a floating point variable that automatically casts
# its value to another dtype (float16 or bfloat16) inside an enable_auto_cast_variables context,
# plus the factory create_autocast_variable and the context manager itself. Used by Keras layers
# under mixed precision policies. Import-time side effects: creates a thread-local holder for the
# active autocast dtype and registers a tensor conversion function for AutoCastVariable.
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
"""Contains AutoCastVariable, a variable which automatically casts itself."""

import threading

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.distribute import distributed_training_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.types import core


# _autocast_dtype.dtype is the dtype AutoCastVariables should be cast to, or
# None if AutoCastVariables should not be cast.
_autocast_dtype = threading.local()


def numpy_text(tensor, is_repr=False):
  # [FUNCTION DOC]
  # HELPER: used by the variable representation methods to print values.
  # Builds a human readable string for the numpy value of an eager tensor. Inputs: tensor, an eager
  # tensor; is_repr, when true uses the detailed representation instead of the plain string form.
  # If the dtype cannot be converted to numpy the text is a placeholder meaning unprintable.
  # Multi-line text gets a leading newline. Output: the string. Accesses the tensor's numpy value,
  # so it only works for eager tensors.
  """Human readable representation of a tensor's numpy value."""
  if tensor.dtype.is_numpy_compatible:
    # pylint: disable=protected-access
    text = repr(tensor._numpy()) if is_repr else str(tensor._numpy())
    # pylint: enable=protected-access
  else:
    text = '<unprintable>'
  if '\n' in text:
    text = '\n' + text
  return text


class AutoCastVariable(variables.Variable, core.Tensor):
  # [CLASS DOC]
  # Wrapper around a floating point variable that behaves like the variable but casts its value to
  # a different dtype when read inside an enable_auto_cast_variables context. Lets Keras layers
  # create float32 variables and use them as float16 or bfloat16 in layer calls. Holds the wrapped
  # variable and an operation placeholder (a marker meaning delegate to the wrapped variable's
  # operation). Reading methods, arithmetic operators and conversion to tensor cast the value;
  # attributes and state methods are delegated to the wrapped variable. Assign and scatter methods
  # return a new wrapper (or the same one) depending on eager versus graph mode. Also registers
  # nothing itself; a tensor conversion function is registered at module level.
  """Variable that will cast itself to a different dtype in applicable contexts.

  This class wraps a floating-point `tf.Variable`. It emulates the variable
  interface and delegates to the wrapped variable, but it additionally will cast
  the wrapped variable under an `enable_auto_cast_variables(dtype)` context
  manager.

  For example:

  >>> v = tf.Variable(1.0, dtype=tf.float32)
  >>> v = AutoCastVariable(v)
  >>> tf.identity(v).dtype
  tf.float32
  >>> with enable_auto_cast_variables(tf.float16):
  ...   tf.identity(v).dtype
  tf.float16

  The purpose of this class is to allow Keras layers to create variables in
  float32, and automatically cast them to float16 or bfloat16 when the layer is
  called.
  """

  def __init__(self, variable):
    # [FUNCTION DOC]
    # Wraps an existing variable. Input: variable, a floating point variable (must be a Variable
    # instance). Raises a value error if it is not a variable or its dtype is not floating point.
    # Stores the wrapped variable and sets the operation marker to the delegate value.
    """Creates an AutoCastVariable instance.

    Args:
      variable: A floating-point resource variable to wrap.

    Raises:
      ValueError: If `variable` is not a floating-point resource variable
    """
    if not isinstance(variable, variables.Variable):
      raise ValueError('variable must be of type tf.ResourceVariable, but got: '
                       '%s' % variable)
    if not variable.dtype.is_floating:
      raise ValueError('variable must be a floating point variable but has '
                       'type: %s' % variable.dtype.name)
    self._variable = variable
    # 'delegate' means AutoCastVariable.op return self._variable.op, which will
    # raise an AttributeError in Eager (as intended). If set to any other value,
    # AutoCastVariable.op returns that value instead, which is used to set the
    # op attribute in AutoCastVariable.assign().
    self._op = 'delegate'

  def _should_cast(self):
    # [FUNCTION DOC]
    # HELPER: decides whether reads should cast.
    # Returns true when a thread-local autocast dtype is active and differs from the variable's own
    # dtype. No side effects.
    """Returns True if this variable should be casted when accessed."""
    autocast_dtype = getattr(_autocast_dtype, 'dtype', None)
    return autocast_dtype is not None and self.dtype != autocast_dtype

  @property
  def dtype(self):
    # [FUNCTION DOC]
    # Property giving the dtype of the wrapped variable, i.e. the dtype before any autocasting.
    """The dtype of the underlying variable, before any casts are done."""
    return self._variable.dtype

  @property
  def true_dtype(self):
    # [FUNCTION DOC]
    # Property giving the wrapped variable's real dtype; a deprecated alias of dtype.
    """Deprecated alias of `dtype`."""
    return self._variable.dtype

  @property
  def _cast_dtype(self):
    # [FUNCTION DOC]
    # HELPER: dtype that reads are cast to.
    # Property returning the active thread-local autocast dtype if set, otherwise the wrapped
    # variable's dtype.
    dtype = getattr(_autocast_dtype, 'dtype', None)
    return dtype or self._variable.dtype

  def value(self):
    # [FUNCTION DOC]
    # Returns the variable's value as a tensor, cast to the autocast dtype only when casting applies
    # (see the should-cast check); otherwise returns the raw value of the wrapped variable.
    val = self._variable.value()
    if not self._should_cast():
      return val
    return math_ops.cast(val, self._cast_dtype)

  def read_value(self):
    # [FUNCTION DOC]
    # Reads the wrapped variable and always casts the result to the current cast dtype (a no-op cast
    # when no autocast is active). Returns a tensor.
    val = self._variable.read_value()
    return math_ops.cast(val, self._cast_dtype)

  def sparse_read(self, indices, name=None):
    # [FUNCTION DOC]
    # Gathers rows of the wrapped variable at the given indices and casts the result to the current
    # cast dtype. Inputs: indices, tensor of indices; name, optional operation name. Returns a tensor.
    """Reads the value of this variable sparsely, using `gather`."""
    val = self._variable.sparse_read(indices, name=name)
    return math_ops.cast(val, self._cast_dtype)

  def gather_nd(self, indices, name=None):
    # [FUNCTION DOC]
    # Gathers slices of the wrapped variable by multi-dimensional indices and casts the result to the
    # current cast dtype. Inputs: indices, tensor of indices; name, optional operation name. Returns a
    # tensor.
    """Gather slices of the variable into a Tensor."""
    val = self._variable.gather_nd(indices, name=name)
    return math_ops.cast(val, self._cast_dtype)

  def __getattr__(self, name):
    # [FUNCTION DOC]
    # Fallback attribute lookup: any attribute not found on the wrapper is fetched from the wrapped
    # variable. Input: attribute name. Output: the attribute value; errors from the wrapped variable
    # propagate.
    return getattr(self._variable, name)

  def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
    # [FUNCTION DOC]
    # Tensor conversion hook registered for this class. Inputs: dtype, optional requested dtype; name,
    # optional name; as_ref, whether a reference was requested. Raises a value error if as_ref is true
    # (not reachable via public APIs). When no autocast applies, converts the wrapped variable
    # directly. Otherwise raises a value error if the requested dtype is incompatible with the cast
    # dtype, converts the wrapped variable at its own dtype and casts to the cast dtype. Returns a
    # tensor.
    """Converts this variable to a tensor."""
    if as_ref:
      # This ValueError should not occur in practice since it is impossible to
      # pass as_ref=True using public APIs.
      raise ValueError('Cannot convert AutoCastVariable to a tensor if '
                       'as_ref=True is passed to convert_to_tensor')
    if not self._should_cast():
      return tensor_conversion.convert_to_tensor_v2_with_dispatch(
          self._variable, dtype=dtype, name=name
      )
    if dtype is not None and not dtype.is_compatible_with(self._cast_dtype):
      raise ValueError(
          'Incompatible type conversion requested to type {!r} for '
          'AutoCastVariable which is casted to type {!r}'.format(
              dtype.name, self._cast_dtype.name))
    val = tensor_conversion.convert_to_tensor_v2_with_dispatch(
        self._variable, dtype=self._variable.dtype, name=name
    )
    return math_ops.cast(val, self._cast_dtype)

  def _should_act_as_resource_variable(self):
    # [FUNCTION DOC]
    # HELPER: marker method that exists only so the resource variable type check passes for this
    # wrapper. Does nothing and returns nothing.
    """Pass resource_variable_ops.is_resource_variable check."""
    pass

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns a string describing the variable: name, shape, dtype and the dtype it is cast to. In
    # eager mode (and not in graph mode) the current numpy value is included, which reads the variable.
    if context.executing_eagerly() and not self._in_graph_mode:
      repr_str = ("<AutoCastVariable '{v.name}' shape={v.shape} "
                  'dtype={v.dtype.name} dtype_to_cast_to={v._cast_dtype.name}, '
                  'numpy={np_repr}>')
      return repr_str.format(
          v=self, np_repr=numpy_text(self.read_value(), is_repr=True))
    else:
      repr_str = ("<AutoCastVariable '{v.name}' shape={v.shape} "
                  'dtype={v.dtype.name} dtype_to_cast_to={v._cast_dtype.name}>')
      return repr_str.format(v=self)

  # Method delegations: We delegate the following methods to self._variable.
  # Each of these methods simply calls the same method on self._variable. The
  # base Variable raises NotImplementedError for most of these, so we must
  # override them.
  #
  # We do not define the following methods from Variable for the following
  # reasons:
  #   * 'count_up_to': This method only applies to int variables, which cannot
  #     be wrapped with an AutoCastVariable.
  #   * 'ref': Instead we inherit the definition from Variable.
  #     If we defined and delegated to Variable, the ref of an AutoCastVariable
  #     would be the same as the ref of the underlying variable, which would be
  #     strange as they are different Python objects.

  def set_shape(self, shape):
    # [FUNCTION DOC]
    # KNOWN ISSUE: passes the wrapper itself as an extra first argument to the wrapped variable's
    # set_shape, so the call is made with the wrong number of arguments and likely fails.
    # Intended to delegate setting the static shape to the wrapped variable. Input: shape. Returns
    # whatever the delegated call returns.
    return self._variable.set_shape(self, shape)

  @property
  def trainable(self):
    # [FUNCTION DOC]
    # Property delegating to the wrapped variable's trainable flag.
    return self._variable.trainable

  @property
  def synchronization(self):
    # [FUNCTION DOC]
    # Property delegating to the wrapped variable's synchronization setting.
    return self._variable.synchronization

  @property
  def aggregation(self):
    # [FUNCTION DOC]
    # Property delegating to the wrapped variable's aggregation setting.
    return self._variable.aggregation

  def eval(self, session=None):
    # [FUNCTION DOC]
    # Delegates evaluation to the wrapped variable (graph mode). Input: optional session. Returns the
    # variable's value as a numpy value; the result is not cast.
    return self._variable.eval(session)

  def initialized_value(self):
    # [FUNCTION DOC]
    # Delegates to the wrapped variable and returns its initialized value tensor (not cast).
    return self._variable.initialized_value()

  @property
  def initial_value(self):
    # [FUNCTION DOC]
    # Property delegating to the wrapped variable's initial value.
    return self._variable.initial_value

  @property
  def constraint(self):
    # [FUNCTION DOC]
    # Property delegating to the wrapped variable's constraint (or nothing if none).
    return self._variable.constraint

  def _apply_assign_update(self,
                           update_fn,
                           value,
                           use_locking=None,
                           name=None,
                           read_value=True):
    # [FUNCTION DOC]
    # HELPER: shared implementation of assign, assign_add and assign_sub.
    # Inputs: update_fn, the wrapped variable's update method; value, the value or delta; use_locking,
    # optional locking flag; name, optional name; read_value, whether to return the updated variable.
    # When eager (outside functions) the update is called without reading; if read_value is true a new
    # wrapper around the same variable is returned with its operation marker set to the update op,
    # else the update op is returned. In graph mode the update is called with read_value and a result
    # that is a resource variable is wrapped in a new autocast variable; otherwise returned as is.
    # TODO(b/146181571): This logic can be simplified once
    # DistributedVariable.assign returns a DistributedVariable. Currently for
    # MirroredStrategy, it returns a Mirrored value.
    if ops.executing_eagerly_outside_functions():
      assign_op = update_fn(value, use_locking, name, False)
      if read_value:
        # We create a new AutoCastVariable with the same underlying tf.Variable.
        # The new AutoCastVariable is identical except the 'op' attribute is
        # defined. This matches the behavior of tf.Variable.assign.
        var = create_autocast_variable(self._variable)
        var._op = assign_op  # pylint:disable=protected-access
        return var
      return assign_op

    # Fallback to wrapping the returned variable in graph mode if possible
    assign_var = update_fn(value, use_locking, name, read_value)
    if read_value and resource_variable_ops.is_resource_variable(assign_var):
      return create_autocast_variable(assign_var)
    return assign_var

  def _apply_update(self, update_fn, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: shared implementation of the scatter methods.
    # Inputs: update_fn, the wrapped variable's method; any further positional and keyword arguments
    # are passed through. Calls it, then returns this wrapper when eager (outside functions); in graph
    # mode wraps a resource variable result in a new autocast variable, else returns the result.
    update_var = update_fn(*args, **kwargs)
    if ops.executing_eagerly_outside_functions():
      return self

    # Fallback to wrapping the returned variable in graph mode if possible
    if resource_variable_ops.is_resource_variable(update_var):
      return create_autocast_variable(update_var)
    return update_var

  def assign(self, value, use_locking=None, name=None, read_value=True):
    # [FUNCTION DOC]
    # Assigns a new value to the wrapped variable. Inputs: value; use_locking, optional; name,
    # optional; read_value, whether to return the updated variable (default true). Returns an
    # autocast variable or update op as described by the shared assign helper. Mutates the variable.
    return self._apply_assign_update(self._variable.assign, value, use_locking,
                                     name, read_value)

  def assign_add(self, delta, use_locking=None, name=None, read_value=True):
    # [FUNCTION DOC]
    # Adds delta to the wrapped variable. Inputs: delta; use_locking, optional; name, optional;
    # read_value, whether to return the updated variable (default true). Return as for assign.
    # Mutates the variable.
    return self._apply_assign_update(self._variable.assign_add, delta,
                                     use_locking, name, read_value)

  def assign_sub(self, delta, use_locking=None, name=None, read_value=True):
    # [FUNCTION DOC]
    # Subtracts delta from the wrapped variable. Inputs: delta; use_locking, optional; name,
    # optional; read_value, whether to return the updated variable (default true). Return as for
    # assign. Mutates the variable.
    return self._apply_assign_update(self._variable.assign_sub, delta,
                                     use_locking, name, read_value)

  def scatter_sub(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Subtracts sparse updates from the wrapped variable. Inputs: sparse_delta, sparse indexed
    # slices; use_locking (default false); name, optional. Mutates the variable; returns this
    # wrapper in eager mode or a wrapped result in graph mode.
    return self._apply_update(self._variable.scatter_sub, sparse_delta,
                              use_locking, name)

  def scatter_add(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Adds sparse updates to the wrapped variable. Inputs: sparse_delta, sparse indexed slices;
    # use_locking (default false); name, optional. Mutates the variable; returns this wrapper in
    # eager mode or a wrapped result in graph mode.
    return self._apply_update(self._variable.scatter_add, sparse_delta,
                              use_locking, name)

  def scatter_max(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Takes the element-wise maximum of the variable and sparse updates. Inputs: sparse_delta;
    # use_locking (default false); name, optional. Mutates the variable; return as for scatter_add.
    return self._apply_update(self._variable.scatter_max, sparse_delta,
                              use_locking, name)

  def scatter_min(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Takes the element-wise minimum of the variable and sparse updates. Inputs: sparse_delta;
    # use_locking (default false); name, optional. Mutates the variable; return as for scatter_add.
    return self._apply_update(self._variable.scatter_min, sparse_delta,
                              use_locking, name)

  def scatter_mul(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Multiplies the variable by sparse updates. Inputs: sparse_delta; use_locking (default false);
    # name, optional. Mutates the variable; return as for scatter_add.
    return self._apply_update(self._variable.scatter_mul, sparse_delta,
                              use_locking, name)

  def scatter_div(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Divides the variable by sparse updates. Inputs: sparse_delta; use_locking (default false);
    # name, optional. Mutates the variable; return as for scatter_add.
    return self._apply_update(self._variable.scatter_div, sparse_delta,
                              use_locking, name)

  def scatter_update(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Overwrites slices of the variable with sparse updates. Inputs: sparse_delta; use_locking
    # (default false); name, optional. Mutates the variable; return as for scatter_add.
    return self._apply_update(self._variable.scatter_update, sparse_delta,
                              use_locking, name)

  def batch_scatter_update(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Batched sparse overwrite of the variable. Inputs: sparse_delta; use_locking (default false);
    # name, optional. Mutates the variable; return as for scatter_add.
    return self._apply_update(self._variable.batch_scatter_update, sparse_delta,
                              use_locking, name)

  def scatter_nd_sub(self, indices, updates, name=None):
    # [FUNCTION DOC]
    # Subtracts updates at multi-dimensional indices. Inputs: indices, updates, name (optional).
    # Mutates the variable; return as for scatter_add.
    return self._apply_update(self._variable.scatter_nd_sub, indices, updates,
                              name)

  def scatter_nd_add(self, indices, updates, name=None):
    # [FUNCTION DOC]
    # Adds updates at multi-dimensional indices. Inputs: indices, updates, name (optional). Mutates
    # the variable; return as for scatter_add.
    return self._apply_update(self._variable.scatter_nd_add, indices, updates,
                              name)

  def scatter_nd_update(self, indices, updates, name=None):
    # [FUNCTION DOC]
    # Overwrites values at multi-dimensional indices. Inputs: indices, updates, name (optional).
    # Mutates the variable; return as for scatter_add.
    return self._apply_update(self._variable.scatter_nd_update, indices,
                              updates, name)

  def load(self, value, session=None):
    # [FUNCTION DOC]
    # Delegates to the wrapped variable to load a value into it (graph mode). Inputs: value; session,
    # optional. Mutates the variable.
    return self._variable.load(value, session)

  @property
  def name(self):
    # [FUNCTION DOC]
    # Property delegating to the wrapped variable's name.
    return self._variable.name

  @property
  def _shared_name(self):
    # [FUNCTION DOC]
    # HELPER: property delegating to the wrapped variable's private shared name.
    return self._variable._shared_name  # pylint:disable=protected-access

  @property
  def initializer(self):
    # [FUNCTION DOC]
    # Property delegating to the wrapped variable's initializer operation.
    return self._variable.initializer

  @property
  def device(self):
    # [FUNCTION DOC]
    # Property delegating to the wrapped variable's device string.
    return self._variable.device

  @property
  def op(self):
    # [FUNCTION DOC]
    # Property returning the wrapped variable's operation if the marker is the delegate value (which
    # raises an attribute error in eager mode); otherwise returns the stored operation, which is set
    # on wrappers returned by assign operations.
    if self._op == 'delegate':
      return self._variable.op
    return self._op

  def _as_graph_element(self):
    # [FUNCTION DOC]
    # HELPER: graph element hook. Returns the wrapped variable's graph element, or the stored
    # operation when that is nothing.
    graph_element = self._variable._as_graph_element()  # pylint:disable=protected-access
    if graph_element is None:
      return self._op
    return graph_element

  @property
  def graph(self):
    # [FUNCTION DOC]
    # Property delegating to the wrapped variable's graph.
    return self._variable.graph

  @property
  def shape(self):
    # [FUNCTION DOC]
    # Property delegating to the wrapped variable's static shape.
    return self._variable.shape

  def get_shape(self) -> tensor_shape.TensorShape:
    # [FUNCTION DOC]
    # Returns the static tensor shape of the wrapped variable.
    return self._variable.get_shape()

  def _gather_saveables_for_checkpoint(self):
    # [FUNCTION DOC]
    # HELPER: checkpoint hook. Delegates to the wrapped variable so checkpoints with wrappers are
    # identical to checkpoints with plain variables. Returns its saveable mapping.
    # By delegating this method to the wrapped variable, checkpoints with
    # AutoCastVariables are identical to checkpoints with normal variables.
    # Therefore models checkpointed with AutoCastVariables can be restored on
    # models with normal variables, and vice versa.
    return self._variable._gather_saveables_for_checkpoint()  # pylint:disable=protected-access

  def _export_to_saved_model_graph(self, object_map, tensor_map, options,
                                   **kwargs):
    # [FUNCTION DOC]
    # HELPER: SavedModel export hook. Delegates to the wrapped variable, then records the wrapper in
    # the object map as mapping to the same exported object as the wrapped variable. Inputs: object_map
    # (mutated), tensor_map, options, extra keyword arguments. Returns the resource list of the
    # wrapped variable.
    # By delegating this method to the wrapped variable, SavedModel with
    # AutoCastVariables are identical to SavedModel with normal variables.
    resource_list = self._variable._export_to_saved_model_graph(  # pylint:disable=protected-access
        object_map, tensor_map, options, **kwargs)
    object_map[self] = object_map[self._variable]
    return resource_list

  # TODO(reedwm): Maybe encode the fact the variable is an AutoCastVariable in
  # to_proto().
  def to_proto(self, export_scope=None):
    # [FUNCTION DOC]
    # Delegates serialization to the wrapped variable's protocol buffer form. Input: optional export
    # scope. Returns the proto.
    return self._variable.to_proto(export_scope)

  def from_proto(self, variable_def, import_scope=None):
    # [FUNCTION DOC]
    # Delegates to the wrapped variable's from-proto method. Inputs: variable_def, the proto;
    # import_scope, optional. Returns whatever the wrapped variable returns.
    return self._variable.from_proto(variable_def, import_scope)

  # Delegate the private attributes _handle_name and _initializer_op to
  # self._variable. SavedModel sets these attributes when loading a model. For
  # example, it sets _handle_name here:
  # https://github.com/tensorflow/tensorflow/blob/db26bd574fa95b5bdd53c08463dd19407cc0297e/tensorflow/python/keras/saving/saved_model/load.py#L211
  # We need to expose these attributes on AutoCastVariable as well for
  # SavedModel to work properly.
  # TODO(reedwm/kathywu): Find a better way to support SavedModel. Exposing
  # private attributes is hacky and difficult to maintain.
  @property
  def _handle_name(self):
    # [FUNCTION DOC]
    # HELPER: getter that exposes the wrapped variable's private handle name; needed so SavedModel
    # loading works with wrappers.
    return self._variable._handle_name  # pylint: disable=protected-access

  @_handle_name.setter
  def _handle_name(self, handle_name):
    # [FUNCTION DOC]
    # HELPER: setter that stores the handle name on the wrapped variable (used by SavedModel loading).
    self._variable._handle_name = handle_name  # pylint: disable=protected-access

  @property
  def _initializer_op(self):
    # [FUNCTION DOC]
    # HELPER: getter that exposes the wrapped variable's private initializer operation; needed for
    # SavedModel loading.
    return self._variable._initializer_op  # pylint: disable=protected-access

  @_initializer_op.setter
  def _initializer_op(self, initializer_op):
    # [FUNCTION DOC]
    # HELPER: setter that stores the initializer operation on the wrapped variable.
    self._variable._initializer_op = initializer_op  # pylint: disable=protected-access

  # Operator overloads:
  # Note we only overload operators that support floating-point types, as
  # non-float variables cannot be wrapped with an AutoCastVariable.
  # Also note: We call read_value() instead of value(), because value() causes
  # gradients not to work properly when TPUStrategy is used: b/143380936

  def __add__(self, o):
    # [FUNCTION DOC]
    # Addition operator: reads the (cast) value and adds the other operand. Returns a tensor.
    return self.read_value() + o

  def __radd__(self, o):
    # [FUNCTION DOC]
    # Reflected addition: other operand plus the cast value. Returns a tensor.
    return o + self.read_value()

  def __sub__(self, o):
    # [FUNCTION DOC]
    # Subtraction operator: cast value minus the other operand. Returns a tensor.
    return self.read_value() - o

  def __rsub__(self, o):
    # [FUNCTION DOC]
    # Reflected subtraction: other operand minus the cast value. Returns a tensor.
    return o - self.read_value()

  def __mul__(self, o):
    # [FUNCTION DOC]
    # Multiplication operator: cast value times the other operand. Returns a tensor.
    return self.read_value() * o

  def __rmul__(self, o):
    # [FUNCTION DOC]
    # Reflected multiplication: other operand times the cast value. Returns a tensor.
    return o * self.read_value()

  def __truediv__(self, o):
    # [FUNCTION DOC]
    # True division operator: cast value divided by the other operand. Returns a tensor.
    return self.read_value() / o

  def __rtruediv__(self, o):
    # [FUNCTION DOC]
    # Reflected true division: other operand divided by the cast value. Returns a tensor.
    return o / self.read_value()

  def __floordiv__(self, o):
    # [FUNCTION DOC]
    # Floor division operator: cast value floor-divided by the other operand. Returns a tensor.
    return self.read_value() // o

  def __rfloordiv__(self, o):
    # [FUNCTION DOC]
    # Reflected floor division: other operand floor-divided by the cast value. Returns a tensor.
    return o // self.read_value()

  def __mod__(self, o):
    # [FUNCTION DOC]
    # Modulo operator: cast value modulo the other operand. Returns a tensor.
    return self.read_value() % o

  def __rmod__(self, o):
    # [FUNCTION DOC]
    # Reflected modulo: other operand modulo the cast value. Returns a tensor.
    return o % self.read_value()

  def __lt__(self, o):
    # [FUNCTION DOC]
    # Less-than comparison of the cast value against the other operand. Returns a boolean tensor.
    return self.read_value() < o

  def __le__(self, o):
    # [FUNCTION DOC]
    # Less-or-equal comparison of the cast value against the other operand. Returns a boolean tensor.
    return self.read_value() <= o

  def __gt__(self, o):
    # [FUNCTION DOC]
    # Greater-than comparison of the cast value against the other operand. Returns a boolean tensor.
    return self.read_value() > o

  def __ge__(self, o):
    # [FUNCTION DOC]
    # Greater-or-equal comparison of the cast value against the other operand. Returns a boolean
    # tensor.
    return self.read_value() >= o

  def __getitem__(self, o):
    # [FUNCTION DOC]
    # Indexing or slicing of the cast value. Input: the index or slice. Returns a tensor.
    return self.read_value()[o]

  def __pow__(self, o, modulo=None):
    # [FUNCTION DOC]
    # Power operator: cast value raised to the other operand, with an optional modulo argument passed
    # to the built-in power function. Returns a tensor.
    return pow(self.read_value(), o, modulo)

  def __rpow__(self, o):
    # [FUNCTION DOC]
    # Reflected power: other operand raised to the cast value. Returns a tensor.
    return pow(o, self.read_value())

  def __neg__(self):
    # [FUNCTION DOC]
    # Negation of the cast value. Returns a tensor.
    return -self.read_value()  # pylint: disable=invalid-unary-operand-type

  def __abs__(self):
    # [FUNCTION DOC]
    # Absolute value of the cast value. Returns a tensor.
    return abs(self.read_value())

  def __div__(self, o):
    # [FUNCTION DOC]
    # Legacy (Python 2 style) division operator: delegates to the cast value's legacy division method;
    # returns the not-implemented marker if the tensor has no such method.
    try:
      return self.read_value().__div__(o)
    except AttributeError:
      # See https://docs.python.org/3/library/constants.html#NotImplemented
      return NotImplemented

  def __rdiv__(self, o):
    # [FUNCTION DOC]
    # Legacy reflected division: delegates to the cast value's legacy reflected division method;
    # returns the not-implemented marker if it does not exist.
    try:
      return self.read_value().__rdiv__(o)
    except AttributeError:
      # See https://docs.python.org/3/library/constants.html#NotImplemented
      return NotImplemented

  def __matmul__(self, o):
    # [FUNCTION DOC]
    # Matrix multiplication operator: delegates to the cast value; returns the not-implemented marker
    # if the tensor lacks the method.
    try:
      return self.read_value().__matmul__(o)
    except AttributeError:
      # See https://docs.python.org/3/library/constants.html#NotImplemented
      return NotImplemented

  def __rmatmul__(self, o):
    # [FUNCTION DOC]
    # Reflected matrix multiplication: delegates to the cast value; returns the not-implemented
    # marker if the tensor lacks the method.
    try:
      return self.read_value().__rmatmul__(o)
    except AttributeError:
      # See https://docs.python.org/3/library/constants.html#NotImplemented
      return NotImplemented

  # pylint: enable=multiple-statements


tensor_conversion_registry.register_tensor_conversion_function(
    AutoCastVariable, AutoCastVariable._dense_var_to_tensor)  # pylint:disable=protected-access


def create_autocast_variable(variable):
  # [FUNCTION DOC]
  # Factory that wraps a variable. Input: variable, a floating point resource variable. If it is not
  # a distributed variable, returns a plain wrapper. Otherwise dynamically defines a class
  # inheriting from both the wrapper class and the variable's class, so type checks against the
  # distributed class still pass, and returns an instance of it. Raises a value error (from the
  # constructor) for invalid variables.
  """Creates an AutoCastVariable that wraps another variable.

  This typically just returns `AutoCastVariable(variable)`. But, if the variable
  is a DistributedVariable or one of its subclasses, we instead dynamically
  create a class that subclasses from both AutoCastVariable and
  variable.__class__. This is so the returned variable will still pass
  `isinstance(variable, variable.__class__)`, which is required for
  DistributedVariables and its subclasses to work properly.

  Args:
    variable: A floating-point resource variable to wrap.

  Returns:
    An AutoCastVariable that wraps the variable.
  """
  if not distributed_training_utils.is_distributed_variable(variable):
    return AutoCastVariable(variable)

  class AutoCastDistributedVariable(AutoCastVariable, variable.__class__):
    # [CLASS DOC]
    # HELPER: locally defined class created per call of the factory. Combines the autocast wrapper
    # with the class of a distributed or aggregating variable so isinstance checks keep working.
    # Only overrides the string representation.
    """An AutoCastVariable that also subclasses from variable.__class__.

    variable.__class__ is either a DistributedVariable or an
    AggregatingVariable.
    """

    def __repr__(self):
      # [FUNCTION DOC]
      # Returns a string with dtype, the dtype cast to, and the wrapped inner variable.

      # pylint: disable=missing-format-attribute
      return ('<AutoCastDistributedVariable dtype={v.dtype.name} '
              'dtype_to_cast_to={v._cast_dtype.name} '
              'inner_variable={v._variable}>'
             ).format(v=self)
      # pylint: enable=missing-format-attribute

  return AutoCastDistributedVariable(variable)


class enable_auto_cast_variables(object):  # pylint:disable=invalid-name
  # [CLASS DOC]
  # Context manager that sets the thread-local dtype to which autocast variables are cast. Holds the
  # target dtype and the previous one (restored on exit). Non-floating dtypes disable casting.
  # Side effect: modifies thread-local module state while active; nests correctly.
  """Context manager which enables the autocasting of `AutoCastVariable`s.

  Under this context manager, `AutoCastVariable`s will be cast to `dtype` if
  `dtype` is floating-point. Otherwise, `AutoCastVariable`s will not be cast.
  """

  __slots__ = ['_dtype', '_prev_dtype']

  def __init__(self, dtype):
    # [FUNCTION DOC]
    # Creates the context manager. Input: dtype, the dtype to cast to; if it is not floating point
    # (or empty) casting is disabled by storing nothing. No side effects until entering.
    if dtype and not dtype.is_floating:
      dtype = None
    self._dtype = dtype

  def __enter__(self):
    # [FUNCTION DOC]
    # Saves the current thread-local autocast dtype and sets it to this manager's dtype. Returns
    # nothing.
    self._prev_dtype = getattr(_autocast_dtype, 'dtype', None)
    _autocast_dtype.dtype = self._dtype

  def __exit__(self, type_arg, value_arg, traceback_arg):
    # [FUNCTION DOC]
    # Restores the thread-local autocast dtype saved on entry. Inputs are the usual exception details,
    # ignored. Does not suppress exceptions (returns nothing).
    _autocast_dtype.dtype = self._prev_dtype
