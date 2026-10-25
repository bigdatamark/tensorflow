# [MODULE DOC]
# Legacy (TF1-style) variable scoping machinery for TensorFlow. This module implements the
# variable store that maps full variable names to variable objects, the get_variable family of
# functions that create-or-reuse variables by name, and the variable_scope context manager that
# controls name prefixes, reuse mode, default initializers/regularizers/partitioners, custom
# getters, device and dtype defaults. It also holds helpers for partitioned variables and for
# keeping per-graph (or per-eager-context) scope state in graph collections.
# Exposes: AUTO_REUSE, VariableScope, VariableStore pieces, get_variable, get_local_variable,
# variable_scope, variable_op_scope, no_regularizer, VariableSynchronization, VariableAggregation,
# and several tf_export-ed helpers. Import-time side effects: registers exported symbols through
# tf_export, attaches documentation to the AUTO_REUSE constant, defines a global collection key
# name for the variable store, and creates module-level default settings. Used heavily by
# tf.compat.v1 layers and legacy models; in TF2 behavior most of it is a thin compatibility layer.
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
"""A class to store named variables and a scope operator to manage sharing."""

import copy
import enum
import functools
import sys
import threading
import traceback

from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import resource_variables_toggle
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import core
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export


__all__ = [
    "AUTO_REUSE", "VariableScope", "get_variable_scope", "get_variable",
    "get_local_variable", "variable_scope", "variable_op_scope",
    "no_regularizer", "VariableSynchronization", "VariableAggregation"
]


class _PartitionInfo:
  # [CLASS DOC]
  # HELPER: internal value object passed to initializers when creating one shard of a
  # partitioned variable.
  # Immutable-ish record (uses slots) of the full combined shape of a partitioned variable and
  # the offset of one particular shard inside it. Initializers that accept a partition_info
  # argument use it to know which slice of the full tensor they must produce. Created by
  # _get_partitioned_variable once per shard.
  """Holds partition info used by initializer functions."""

  __slots__ = ["_full_shape", "_var_offset"]

  def __init__(self, full_shape, var_offset):
    # [FUNCTION DOC]
    # Validates and stores the full shape and the shard offset. full_shape and var_offset must
    # both be lists or tuples of integers of equal length, and every offset must satisfy
    # zero <= offset < corresponding full dimension. Raises a type error if either argument is
    # not a list/tuple and a value error if lengths differ or an offset is out of range.
    # Returns nothing; sets the two internal attributes.
    """Constructor.

    Args:
      full_shape: Tuple or list of `int` indicating the full combined shape of
        the partitioned variables.
      var_offset: Tuple or list of `int` specifying offset of this partition
        with respect to the full variable for each dimension.

    Raises:
      TypeError: If `full_shape` or `var_offset` is not a sequence.
      ValueError: If `full_shape` or `var_offset` differ in length. If
        `var_offset` exceeds `full_shape` in any dimension.
    """
    if not isinstance(full_shape, (list, tuple)):
      raise TypeError(
          "`full_shape` must be a sequence (like tuple or list) instead of " +
          type(full_shape).__name__)

    if not isinstance(var_offset, (list, tuple)):
      raise TypeError(
          "`var_offset` must be a sequence (like tuple or list) instead of " +
          type(var_offset).__name__)

    if len(var_offset) != len(full_shape):
      raise ValueError(
          "Expected equal length, but `var_offset` is of length {} while "
          "full_shape is of length {}.".format(
              len(var_offset), len(full_shape)))

    for offset, shape in zip(var_offset, full_shape):
      if offset < 0 or offset >= shape:
        raise ValueError(
            "Expected 0 <= offset < shape but found offset={}, shape={} for "
            "var_offset={}, full_shape={}".format(offset, shape, var_offset,
                                                  full_shape))

    self._full_shape = full_shape
    self._var_offset = var_offset

  @property
  def full_shape(self):
    # [FUNCTION DOC]
    # Read-only property returning the full combined shape (list or tuple of ints) of the
    # partitioned variable that this record describes. No side effects.
    return self._full_shape

  @property
  def var_offset(self):
    # [FUNCTION DOC]
    # Read-only property returning the per-dimension offset (list or tuple of ints) of this
    # shard within the full variable. No side effects.
    return self._var_offset

  def single_offset(self, shape):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says only a value error can be raised, but single_slice_dim
    # can also raise a type error when shape is not a list/tuple.
    # Given the shape of one shard, returns the offset of the shard along the single dimension
    # in which the variable is partitioned, or 0 when no dimension is actually partitioned
    # (shard shape equals full shape). Delegates validation to single_slice_dim, so it raises
    # whatever that method raises (bad type, wrong rank, shard exceeding full shape, or
    # partitioned in more than one dimension). No side effects.
    """Returns the offset when the variable is partitioned in at most one dim.

    Args:
      shape: Tuple or list of `int` indicating the shape of one specific
        variable partition.

    Returns:
      `int` representing the offset in the dimension along which the variable is
       partitioned. Returns 0 if the variable is not being partitioned.

    Raises:
      ValueError: Depending on self.single_slice_dim().
    """

    single_slice_dim = self.single_slice_dim(shape)
    # If this variable is not being partitioned at all, single_slice_dim() could
    # return None.
    if single_slice_dim is None:
      return 0
    return self.var_offset[single_slice_dim]

  def single_slice_dim(self, shape):
    # [FUNCTION DOC]
    # Finds the one dimension along which this shard differs from the full shape. shape is the
    # shard shape (list or tuple of ints, same rank as the full shape). Returns that dimension's
    # index, or the null value if shard shape equals full shape in every dimension. Raises a
    # type error if shape is not a list/tuple; a value error if its length differs from the full
    # shape, if offset plus shard size exceeds the full size in some dimension, or if more than
    # one dimension differs from the full shape. No side effects.
    """Returns the slice dim when the variable is partitioned only in one dim.

    Args:
      shape: Tuple or list of `int` indicating the shape of one specific
        variable partition.

    Returns:
      `int` representing the dimension that the variable is partitioned in, or
      `None` if the variable doesn't seem to be partitioned at all.

    Raises:
      TypeError: If `shape` is not a sequence.
      ValueError: If `shape` is not the same length as `self.full_shape`. If
        the variable is partitioned in more than one dimension.
    """
    if not isinstance(shape, (tuple, list)):
      raise TypeError(
          "`shape` must be a sequence (like tuple or list) instead of " +
          type(shape).__name__)

    if len(shape) != len(self.full_shape):
      raise ValueError(
          "Expected equal length, but received shape={} of length {} while "
          "self.full_shape={} is of length {}.".format(shape, len(shape),
                                                       self.full_shape,
                                                       len(self.full_shape)))

    for i in range(len(shape)):
      if self.var_offset[i] + shape[i] > self.full_shape[i]:
        raise ValueError(
            "With self.var_offset={}, a partition of shape={} would exceed "
            "self.full_shape={} in dimension {}.".format(
                self.var_offset, shape, self.full_shape, i))

    slice_dim = None
    for i in range(len(shape)):
      if shape[i] == self.full_shape[i]:
        continue
      if slice_dim is not None:
        raise ValueError(
            "Cannot use single_slice_dim() with shape={} and "
            "self.full_shape={} since slice dim could be either dimension {} "
            "or {}.".format(shape, self.full_shape, i, slice_dim))
      slice_dim = i

    return slice_dim


class _ReuseMode(enum.Enum):
  # [CLASS DOC]
  # HELPER: internal enumeration backing the public AUTO_REUSE constant.
  # Enumeration of non-boolean reuse modes for variable scopes. Currently has the single member
  # AUTO_REUSE, meaning "return the variable if it exists, otherwise create it". Boolean/null
  # reuse values are handled separately by the scope code.
  """Mode for variable access within a variable scope."""

  # Indicates that variables are to be fetched if they already exist or
  # otherwise created.
  AUTO_REUSE = 1

  # TODO(alive): For TensorFlow 2.0, Deprecate True/False/None API in favor of
  #              enum values.
  # REUSE_FALSE = 2
  # REUSE_TRUE = 3


# TODO(apassos) remove these forwarding symbols.
VariableSynchronization = variables.VariableSynchronization  # pylint: disable=invalid-name
VariableAggregation = variables.VariableAggregation  # pylint: disable=invalid-name

AUTO_REUSE = _ReuseMode.AUTO_REUSE
tf_export(v1=["AUTO_REUSE"]).export_constant(__name__, "AUTO_REUSE")
AUTO_REUSE.__doc__ = """
@compatibility(TF2)
`tf.compat.v1.AUTO_REUSE` is a legacy API that is a no-op when TF2 behaviors
are enabled.

If you rely on `get_variable` and auto-reuse, see the
[model mapping guide](https://www.tensorflow.org/guide/migrate/model_mapping)
for more info on how to migrate your code.

Note: when you use the `tf.compat.v1.keras.utils.track_tf1_style_variables`
API as described in the above guide, `get_variable` will always behave as if
`v1.AUTO_REUSE` is set. Without the decorator, reuse will be ignored and new
variables will always be created, regardless of if they have already been
created.
@end_compatibility

When passed in as the value for the `reuse` flag, `AUTO_REUSE` indicates that
get_variable() should create the requested variable if it doesn't exist or, if
it does exist, simply return it.
"""


def _needs_no_arguments(python_callable):
  # [FUNCTION DOC]
  # HELPER: used by the variable store to decide whether an initializer callable can be invoked
  # with no arguments.
  # Takes any callable (function, partial, or callable object). Inspects its argument spec,
  # discounts the implicit self argument for non-function callables, and returns true when the
  # number of declared arguments equals the number of defaults, i.e. every argument has a
  # default. Has no side effects; may propagate errors from the argument inspection helper if the
  # callable cannot be introspected.
  """Returns true if the callable needs no arguments to call."""
  # TODO(bfontain): Switch to inspect.signature when we are python 3 only.
  # signature = inspect.signature(python_callable)
  # return not [1 for param in signature.parameters.values()
  #             if param.default == param.empty]
  num_arguments = len(tf_inspect.getargspec(python_callable).args)
  if not tf_inspect.isfunction(python_callable) and not isinstance(
      python_callable, functools.partial):
    # getargspec includes self for function objects (which aren't
    # functools.partial). This has no default so we need to remove it.
    # It is not even an argument so its odd that getargspec returns this.
    # Note that this is fixed with inspect.signature in Python 3.
    num_arguments -= 1
  return num_arguments == len(
      tf_inspect.getargspec(python_callable).defaults or [])


class _VariableStore:
  # [CLASS DOC]
  # KNOWN ISSUE: the docstring says stored variables are initialized with the initializer passed
  # to the constructor and mentions an attribute named vars; the constructor takes no arguments
  # and the attributes are the private dictionaries _vars and _partitioned_vars.
  # Registry of named variables for one graph (or one eager variable store). Holds a mapping from
  # full variable name to variable object, a mapping from name to partitioned variable, and a
  # flag saying whether eager-mode variables are tracked. Provides get_variable, which creates or
  # reuses variables according to reuse mode, shape/dtype compatibility checks, partitioners and
  # custom getters. One instance is stored in a graph collection and fetched by variable scopes.
  """Variable store that carries a number of named Variables.

  New variable names and new variables can be created; all stored
  variables are initialized with the initializer passed to __init__.

  Attributes:
    vars: a dictionary with string names (same as passed in GetVar) as keys and
      the corresponding TensorFlow Variables as values.
  """

  __slots__ = ["_vars", "_partitioned_vars", "_store_eager_variables"]

  def __init__(self):
    # [FUNCTION DOC]
    # Creates an empty store: empty mapping of variables, empty mapping of partitioned variables,
    # and eager-variable storing turned off. Takes no arguments, returns nothing.
    """Create a variable store."""
    self._vars = {}  # A dictionary of the stored TensorFlow variables.
    self._partitioned_vars = {}  # A dict of the stored PartitionedVariables.
    self._store_eager_variables = False

  def get_variable(self,
                   name,
                   shape=None,
                   dtype=dtypes.float32,
                   initializer=None,
                   regularizer=None,
                   reuse=None,
                   trainable=None,
                   collections=None,
                   caching_device=None,
                   partitioner=None,
                   validate_shape=True,
                   use_resource=None,
                   custom_getter=None,
                   constraint=None,
                   synchronization=VariableSynchronization.AUTO,
                   aggregation=VariableAggregation.NONE):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says the default initializer comes from the constructor, but the
    # constructor takes none; the default is chosen later in the single-variable path. It also
    # says use_resource defaults to False, while in eager mode (or when lifted by an init scope
    # into eager context) the code forces it to true, and it says reuse is always forced to false
    # in eager mode, whereas the code actually sets it to auto-reuse when the store keeps eager
    # variables.
    # Main entry point of the store: returns the variable registered under name, or creates it.
    # Inputs mirror the public get_variable (name, shape, dtype, initializer, regularizer, reuse,
    # trainable, collections, caching_device, partitioner, validate_shape, use_resource,
    # custom_getter, constraint, synchronization, aggregation); all but name are optional.
    # Steps: validate that custom_getter is callable; in eager mode force use_resource and
    # reject reuse unless an eager variable store is active; strip reference dtypes; validate
    # synchronization/aggregation/trainable; then either call the custom getter (passing the
    # internal getter and, only if it accepts them, the constraint) or call the internal getter
    # directly. Returns a variable or a partitioned variable. Raises a value error for a
    # non-callable custom getter or for reuse/shape/partition conflicts, and a runtime error if
    # reuse is requested in eager mode without an eager variable store. Side effects: may create
    # and register variables in the store and add them to graph collections.
    """Gets an existing variable with these parameters or create a new one.

    If a variable with the given name is already stored, we return the stored
    variable. Otherwise, we create a new one.

    Set `reuse` to `True` when you only want to reuse existing Variables.
    Set `reuse` to `False` when you only want to create new Variables.
    Set `reuse` to None (the default) or tf.compat.v1.AUTO_REUSE when you want
    variables to be created if they don't exist or returned if they do.

    If initializer is `None` (the default), the default initializer passed in
    the constructor is used. If that one is `None` too, we use a new
    `glorot_uniform_initializer`. If initializer is a Tensor, we use
    it as a value and derive the shape from the initializer.

    If a partitioner is provided, a `PartitionedVariable` is returned.
    Accessing this object as a `Tensor` returns the shards concatenated along
    the partition axis.

    Some useful partitioners are available.  See, e.g.,
    `variable_axis_size_partitioner` and `min_max_variable_partitioner`.

    Args:
      name: The name of the new or existing variable.
      shape: Shape of the new or existing variable.
      dtype: Type of the new or existing variable (defaults to `DT_FLOAT`).
      initializer: Initializer for the variable.
      regularizer: A (Tensor -> Tensor or None) function; the result of applying
        it on a newly created variable will be added to the collection
        GraphKeys.REGULARIZATION_LOSSES and can be used for regularization.
      reuse: a Boolean, None, or tf.AUTO_REUSE. Controls reuse or creation of
        variables. When eager execution is enabled  this argument is always
        forced to be False.
      trainable: If `True` also add the variable to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`). `trainable`
        defaults to `True`, unless `synchronization` is set to `ON_READ`, in
        which case it defaults to `False`.
      collections: List of graph collections keys to add the `Variable` to.
        Defaults to `[GraphKeys.GLOBAL_VARIABLES]` (see `tf.Variable`).
      caching_device: Optional device string or function describing where the
        Variable should be cached for reading.  Defaults to the Variable's
        device.  If not `None`, caches on another device.  Typical use is to
        cache on the device where the Ops using the `Variable` reside, to
        deduplicate copying through `Switch` and other conditional statements.
      partitioner: Optional callable that accepts a fully defined `TensorShape`
        and dtype of the `Variable` to be created, and returns a list of
        partitions for each axis (currently only one axis can be partitioned).
      validate_shape: If False, allows the variable to be initialized with a
        value of unknown shape. If True, the default, the shape of initial_value
        must be known.
      use_resource: If False, creates a regular Variable. If True, creates
        instead an experimental ResourceVariable which has well-defined
        semantics. Defaults to False (will later change to True). When eager
        execution is enabled this argument is always forced to be true.
      custom_getter: Callable that takes as a first argument the true getter,
        and allows overwriting the internal get_variable method. The signature
        of `custom_getter` should match that of this method,
        but the most future-proof version will allow for changes: `def
          custom_getter(getter, *args, **kwargs)`.  Direct access to
        all `get_variable` parameters is also allowed: `def
          custom_getter(getter, name, *args, **kwargs)`.  A simple identity
        custom getter that simply creates variables with modified names is:
          ```python
        def custom_getter(getter, name, *args, **kwargs): return getter(name +
          '_suffix', *args, **kwargs) ```
      constraint: An optional projection function to be applied to the variable
        after being updated by an `Optimizer` (e.g. used to implement norm
        constraints or value constraints for layer weights). The function must
        take as input the unprojected Tensor representing the value of the
        variable and return the Tensor for the projected value (which must have
        the same shape). Constraints are not safe to use when doing asynchronous
        distributed training.
      synchronization: Indicates when a distributed a variable will be
        aggregated. Accepted values are constants defined in the class
        `tf.VariableSynchronization`. By default the synchronization is set to
        `AUTO` and the current `DistributionStrategy` chooses when to
        synchronize.
      aggregation: Indicates how a distributed variable will be aggregated.
        Accepted values are constants defined in the class
        `tf.VariableAggregation`.

    Returns:
      The created or existing `Variable` (or `PartitionedVariable`, if a
      partitioner was used).

    Raises:
      ValueError: when creating a new variable and shape is not declared,
        when reusing a variable and specifying a conflicting shape,
        or when violating reuse during variable creation.
      RuntimeError: when eager execution is enabled and not called from an
        EagerVariableStore.
    """
    if custom_getter is not None and not callable(custom_getter):
      raise ValueError("Passed a custom_getter which is not callable: %s" %
                       custom_getter)

    with ops.init_scope():
      if context.executing_eagerly():
        # Variable creation and initialization takes place in `init_scope`s;
        # as such, if an `init_scope` lifts us into the eager context, then we
        # need to use `ResourceVariable`s.
        use_resource = True

    # Note that it's fine to reuse eager variables whose initialization was
    # lifted from a function-building graph into the eager context (that's why
    # the following clause is not wrapped in an `init_scope`); lifted variables
    # are tracked by the graph's `VariableStore`.
    if context.executing_eagerly():
      if not self._store_eager_variables and reuse:
        raise RuntimeError(
            "When eager execution is enabled variable reuse is only supported"
            " when an EagerVariableStore is active. See the documentation on"
            " EagerVariableStore for example usage.")
      if self._store_eager_variables:
        reuse = AUTO_REUSE

    # If a *_ref type is passed in an error would be triggered further down the
    # stack. We prevent this using base_dtype to get a non-ref version of the
    # type, before doing anything else. When _ref types are removed in favor of
    # resources, this line can be removed.
    try:
      dtype = dtype.base_dtype
    except AttributeError:
      # .base_dtype not existing means that we will try and use the raw dtype
      # which was passed in - this might be a NumPy type which is valid.
      pass

    # This is the main logic of get_variable.  However, custom_getter
    # may override this logic.  So we save it as a callable and pass
    # it to custom_getter.
    # Note: the parameters of _true_getter, and their documentation, match
    # *exactly* item-for-item with the docstring of this method.
    def _true_getter(  # pylint: disable=missing-docstring
        name,
        shape=None,
        dtype=dtypes.float32,
        initializer=None,
        regularizer=None,
        reuse=None,
        trainable=None,
        collections=None,
        caching_device=None,
        partitioner=None,
        validate_shape=True,
        use_resource=None,
        constraint=None,
        synchronization=VariableSynchronization.AUTO,
        aggregation=VariableAggregation.NONE):
      # [FUNCTION DOC]
      # HELPER: nested default getter handed to custom getters by get_variable.
      # Core create-or-reuse logic. If a partitioner is given and the requested shape is not a
      # scalar, validates the partitioner is callable (value error otherwise) and delegates to the
      # partitioned-variable path with the name scope cleared. If reuse is exactly true, no
      # partitioner is given and a partitioned variable of that name exists, reuses it through
      # the partitioned path. Otherwise raises a value error when shards of the same name exist
      # without a partitioner, else delegates to the single-variable path. Arguments are the same
      # as get_variable minus custom_getter; returns the variable found or created.
      is_scalar = (
          shape is not None and isinstance(shape, collections_abc.Sequence) and
          not shape)
      # Partitioned variable case
      if partitioner is not None and not is_scalar:
        if not callable(partitioner):
          raise ValueError("Partitioner must be callable, but received: %s" %
                           partitioner)
        with ops.name_scope(None):
          return self._get_partitioned_variable(
              name=name,
              shape=shape,
              dtype=dtype,
              initializer=initializer,
              regularizer=regularizer,
              reuse=reuse,
              trainable=trainable,
              collections=collections,
              caching_device=caching_device,
              partitioner=partitioner,
              validate_shape=validate_shape,
              use_resource=use_resource,
              constraint=constraint,
              synchronization=synchronization,
              aggregation=aggregation)

      # Special case for partitioned variable to allow reuse without having to
      # specify partitioner.
      if (reuse is True and partitioner is None
          and name in self._partitioned_vars):
        return self._get_partitioned_variable(
            name=name,
            shape=shape,
            dtype=dtype,
            initializer=initializer,
            regularizer=regularizer,
            reuse=reuse,
            trainable=trainable,
            collections=collections,
            caching_device=caching_device,
            partitioner=None,
            validate_shape=validate_shape,
            use_resource=use_resource,
            constraint=constraint,
            synchronization=synchronization,
            aggregation=aggregation)

      # Single variable case
      if "%s/part_0" % name in self._vars:
        raise ValueError(
            "No partitioner was provided, but a partitioned version of the "
            "variable was found: %s/part_0. Perhaps a variable of the same "
            "name was already created with partitioning?" % name)

      return self._get_single_variable(
          name=name,
          shape=shape,
          dtype=dtype,
          initializer=initializer,
          regularizer=regularizer,
          reuse=reuse,
          trainable=trainable,
          collections=collections,
          caching_device=caching_device,
          validate_shape=validate_shape,
          use_resource=use_resource,
          constraint=constraint,
          synchronization=synchronization,
          aggregation=aggregation)

    synchronization, aggregation, trainable = (
        variables.validate_synchronization_aggregation_trainable(
            synchronization, aggregation, trainable, name))

    if custom_getter is not None:
      # Handle backwards compatibility with getter arguments that were added
      # to the API after users started writing custom getters.
      custom_getter_kwargs = {
          "getter": _true_getter,
          "name": name,
          "shape": shape,
          "dtype": dtype,
          "initializer": initializer,
          "regularizer": regularizer,
          "reuse": reuse,
          "trainable": trainable,
          "collections": collections,
          "caching_device": caching_device,
          "partitioner": partitioner,
          "validate_shape": validate_shape,
          "use_resource": use_resource,
          "synchronization": synchronization,
          "aggregation": aggregation,
      }
      # `fn_args` and `has_kwargs` can handle functions, `functools.partial`,
      # `lambda`.
      if ("constraint" in function_utils.fn_args(custom_getter) or
          function_utils.has_kwargs(custom_getter)):
        custom_getter_kwargs["constraint"] = constraint
      return custom_getter(**custom_getter_kwargs)
    else:
      return _true_getter(
          name,
          shape=shape,
          dtype=dtype,
          initializer=initializer,
          regularizer=regularizer,
          reuse=reuse,
          trainable=trainable,
          collections=collections,
          caching_device=caching_device,
          partitioner=partitioner,
          validate_shape=validate_shape,
          use_resource=use_resource,
          constraint=constraint,
          synchronization=synchronization,
          aggregation=aggregation)

  def _get_partitioned_variable(self,
                                name,
                                partitioner,
                                shape=None,
                                dtype=dtypes.float32,
                                initializer=None,
                                regularizer=None,
                                reuse=None,
                                trainable=None,
                                collections=None,
                                caching_device=None,
                                validate_shape=True,
                                use_resource=None,
                                constraint=None,
                                synchronization=VariableSynchronization.AUTO,
                                aggregation=VariableAggregation.NONE):
    # [FUNCTION DOC]
    # HELPER: partitioned-variable branch of get_variable.
    # Creates or reuses a sharded variable. Inputs: name and a callable partitioner (required),
    # plus the usual shape/dtype/initializer/regularizer/reuse/trainable/collections/
    # caching_device/validate_shape/use_resource/constraint/synchronization/aggregation options.
    # Rejects names already used by an unpartitioned variable. Computes the shard counts by
    # calling the partitioner on the shape, merging shape with a tensor initializer's shape. If a
    # partitioned variable of that name exists it checks reuse is not false and that shape, dtype
    # and partitions are compatible, then returns it. Otherwise requires reuse to be not true,
    # checks existing shard names for a conflicting partitioning, and creates one shard variable
    # per slice (each named with a part suffix, with a partition-info object and sliced
    # initializer), then wraps them in a partitioned variable stored in the store and returns it.
    # Raises value errors on any conflict. Side effects: registers shards in the store.
    """Gets or creates a sharded variable list with these parameters.

    The `partitioner` must be a callable that accepts a fully defined
    `TensorShape` and returns a sequence of integers (the `partitions`).
    These integers describe how to partition the given sharded `Variable`
    along the given dimension.  That is, `partitions[1] = 3` means split
    the `Variable` into 3 shards along dimension 1.  Currently, sharding along
    only one axis is supported.

    If the list of variables with the given name (prefix) is already stored,
    we return the stored variables. Otherwise, we create a new one.

    Set `reuse` to `True` when you only want to reuse existing Variables.
    Set `reuse` to `False` when you only want to create new Variables.
    Set `reuse` to None (the default) or tf.compat.v1.AUTO_REUSE when you want
    variables to be created if they don't exist or returned if they do.

    If initializer is `None` (the default), the default initializer passed in
    the constructor is used. If that one is `None` too, we use a new
    `glorot_uniform_initializer`. If initializer is a Tensor, we use
    it as a value and derive the shape from the initializer.

    If the initializer is a callable, then it will be called for each
    shard.  Otherwise the initializer should match the shape of the entire
    sharded Variable, and it will be sliced accordingly for each shard.

    Some useful partitioners are available.  See, e.g.,
    `variable_axis_size_partitioner` and `min_max_variable_partitioner`.

    Args:
      name: the name of the new or existing sharded variable.
      partitioner: Optional callable that accepts a fully defined `TensorShape`
        and `dtype` of the Variable to be created, and returns a list of
        partitions for each axis (currently only one axis can be partitioned).
      shape: shape of the new or existing sharded variable.
      dtype: type of the new or existing sharded variable (defaults to
        `DT_FLOAT`).
      initializer: initializer for the sharded variable.
      regularizer: a (Tensor -> Tensor or None) function; the result of applying
        it on a newly created variable will be added to the collection
        GraphKeys.REGULARIZATION_LOSSES and can be used for regularization.
      reuse: a Boolean, None, or tf.AUTO_REUSE. Controls reuse or creation of
        variables.
      trainable: If `True` also add the variable to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
      collections: List of graph collections keys to add the Variable to.
        Defaults to `[GraphKeys.GLOBAL_VARIABLES]` (see `tf.Variable`).
      caching_device: Optional device string or function describing where the
        Variable should be cached for reading.  Defaults to the Variable's
        device.  If not `None`, caches on another device.  Typical use is to
        cache on the device where the Ops using the Variable reside, to
        deduplicate copying through `Switch` and other conditional statements.
      validate_shape: If False, allows the variable to be initialized with a
        value of unknown shape. If True, the default, the shape of initial_value
        must be known.
      use_resource: If False, creates a regular Variable. If True, creates an
        experimental ResourceVariable which has well-defined semantics. Defaults
        to False (will later change to True).
      constraint: An optional projection function to be applied to the variable
        after being updated by an `Optimizer` (e.g. used to implement norm
        constraints or value constraints for layer weights). The function must
        take as input the unprojected Tensor representing the value of the
        variable and return the Tensor for the projected value (which must have
        the same shape). Constraints are not safe to use when doing asynchronous
        distributed training.
      synchronization: Indicates when a distributed a variable will be
        aggregated. Accepted values are constants defined in the class
        `tf.VariableSynchronization`. By default the synchronization is set to
        `AUTO` and the current `DistributionStrategy` chooses when to
        synchronize.
      aggregation: Indicates how a distributed variable will be aggregated.
        Accepted values are constants defined in the class
        `tf.VariableAggregation`.

    Returns:
      A `PartitionedVariable` object.

    Raises:
      ValueError: when creating a new variable and shape is not declared,
        when reusing a variable and specifying a conflicting shape,
        when violating reuse during variable creation, or if an existing
        sharded variable exists for the given name but with different sharding.
    """
    initializing_from_value = initializer is not None and isinstance(
        initializer, tensor.Tensor)
    if name in self._vars:
      raise ValueError(
          "A partitioner was provided, but an unpartitioned version of the "
          "variable was found: %s.  Perhaps a variable of the same name was "
          "already created without partitioning?" % name)

    shape = tensor_shape.as_shape(shape)
    if initializing_from_value:
      shape = shape.merge_with(initializer.get_shape())

    partitions = None
    if not reuse or partitioner:
      partitions = _call_partitioner(partitioner, shape, dtype)

    if name in self._partitioned_vars:
      if reuse is False:
        raise ValueError(
            "Partitioned variable with name %s already exists. Did you mean to "
            "set reuse=True or reuse=tf.AUTO_REUSE in VarScope?" % name)

      existing_var = self._partitioned_vars[name]
      if not shape.is_compatible_with(existing_var.get_shape()):
        raise ValueError(
            "Trying to reuse partitioned variable %s, but specified shape %s "
            "and found shape %s." % (name, shape, existing_var.get_shape()))
      if not dtype.is_compatible_with(existing_var.dtype):
        raise ValueError(
            "Trying to reuse partitioned variable %s, but specified dtype %s "
            "and found dtype %s." % (name, dtype.name, existing_var.dtype.name))

      # pylint: disable=protected-access
      if (partitions is not None and
          existing_var._get_partitions() != partitions):
        raise ValueError(
            "Trying to reuse partitioned variable %s, but specified partitions "
            "%s and found partitions %s." %
            (name, partitions, existing_var._get_partitions()))
      # pylint: enable=protected-access

      return existing_var

    if reuse is True:
      raise ValueError("PartitionedVariable %s does not exist, or was not "
                       "created with tf.get_variable(). Did you mean to set "
                       "reuse=False or reuse=tf.AUTO_REUSE in VarScope?" % name)

    slice_dim, num_slices = _get_slice_dim_and_num_slices(partitions)

    if "%s/part_0" % name in self._vars:
      if "%s/part_%d" % (name, num_slices - 1) not in self._vars:
        raise ValueError(
            "Partitioner returned a different partitioning than what was "
            "already found.  Partitioner returned %d shards, and shard "
            "%s/part_0 was found, but %s/part_%d was not." %
            (num_slices, name, name, num_slices - 1))
      if "%s/part_%d" % (name, num_slices) in self._vars:
        raise ValueError(
            "Partitioner returned a different partitioning than what was "
            "already found.  Partitioner returned %d shards, and shard "
            "%s/part_0 was found, but so was the extra shard %s/part_%d." %
            (num_slices, name, name, num_slices))

    vs = []
    for i, (var_offset, var_shape) in enumerate(
        _iter_slices(shape.as_list(), num_slices, slice_dim)):
      partition_info = _PartitionInfo(
          full_shape=shape.as_list(), var_offset=var_offset)
      var_full_name = "%s/part_%d" % (name, i)
      with ops.name_scope(
          var_full_name + "/PartitionedInitializer", skip_on_eager=False):
        # Create the tensor to initialize the variable with default value.
        if initializer is None:
          init, initializing_from_value = self._get_default_initializer(
              name=name, shape=shape, dtype=dtype)
          if initializing_from_value:
            init_shape = None
          else:
            init_shape = var_shape
        elif callable(initializer):
          init = initializer
          init_shape = var_shape
        elif isinstance(initializer, tensor.Tensor):
          init = array_ops.slice(initializer, var_offset, var_shape)
          # Use the dtype of the given tensor.
          dtype = init.dtype.base_dtype
          init_shape = None
        else:
          init = ops.convert_to_tensor(initializer, dtype=dtype)
          init = array_ops.slice(init, var_offset, var_shape)
          init_shape = None

      with ops.name_scope(None):
        var = self._get_single_variable(
            name=var_full_name,
            shape=init_shape,
            dtype=dtype,
            initializer=init,
            partition_info=partition_info,
            regularizer=regularizer,
            reuse=reuse,
            trainable=trainable,
            collections=collections,
            caching_device=caching_device,
            validate_shape=validate_shape,
            use_resource=use_resource,
            constraint=constraint,
            synchronization=synchronization,
            aggregation=aggregation,
        )

      # pylint: disable=protected-access
      var._set_save_slice_info(
          variables.Variable.SaveSliceInfo(name, shape.as_list(), var_offset,
                                           var_shape))
      vs.append(var)
      # pylint: enable=protected-access

    partitioned_var = variables.PartitionedVariable(
        name=name,
        shape=shape,
        dtype=dtype,
        variable_list=vs,
        partitions=partitions)
    if not context.executing_eagerly() or self._store_eager_variables:
      self._partitioned_vars[name] = partitioned_var
    return partitioned_var

  def _get_single_variable(self,
                           name,
                           shape=None,
                           dtype=dtypes.float32,
                           initializer=None,
                           regularizer=None,
                           partition_info=None,
                           reuse=None,
                           trainable=None,
                           collections=None,
                           caching_device=None,
                           validate_shape=True,
                           use_resource=None,
                           constraint=None,
                           synchronization=VariableSynchronization.AUTO,
                           aggregation=VariableAggregation.NONE):
    # [FUNCTION DOC]
    # HELPER: single-variable (or single-shard) branch of get_variable.
    # KNOWN ISSUE: the docstring's summary line is broken across two paragraphs by a formatting
    # accident, and it omits that the regularizer is called twice: once immediately to test
    # whether it returns a value, and again lazily to build the loss tensor.
    # Looks up name in the store. If it exists: raises a value error when reuse is false (with
    # a traceback of the original definition for non-resource variables), when the requested
    # shape or dtype is incompatible with the stored variable; otherwise returns it. If it does
    # not exist: raises a value error if reuse is true, if a constant initializer is combined
    # with an explicit shape, if no initializer and no shape are given, or if the initializer is
    # neither a usable callable nor combinable with a fully defined shape. Chooses a default
    # initializer when none is given, instantiates initializer classes, binds shape, dtype and
    # partition info into an initial-value callable inside an init scope, picks use_resource from
    # the global toggle when unspecified, and builds the variable through the v1 variable
    # constructor. Side effects: stores the variable in the store (graph mode, or eager with an
    # eager store), adds it to collections in eager-store mode, logs creation at verbose level,
    # and registers a lazily evaluated regularization loss in the regularization-losses collection.
    # Returns the variable.
    """Get or create a single Variable (e.g.

    a shard or entire variable).

    See the documentation of get_variable above (ignore partitioning components)
    for details.

    Args:
      name: see get_variable.
      shape: see get_variable.
      dtype: see get_variable.
      initializer: see get_variable.
      regularizer: see get_variable.
      partition_info: _PartitionInfo object.
      reuse: see get_variable.
      trainable: see get_variable.
      collections: see get_variable.
      caching_device: see get_variable.
      validate_shape: see get_variable.
      use_resource: see get_variable.
      constraint: see get_variable.
      synchronization: see get_variable.
      aggregation: see get_variable.

    Returns:
      A Variable.  See documentation of get_variable above.

    Raises:
      ValueError: See documentation of get_variable above.
    """
    # Set to true if initializer is a constant.
    initializing_from_value = False
    if initializer is not None and not callable(initializer):
      initializing_from_value = True
    if shape is not None and initializing_from_value:
      raise ValueError("If initializer is a constant, do not specify shape.")

    dtype = dtypes.as_dtype(dtype)
    if shape is not None:
      shape = tensor_shape.as_shape(shape)

    if name in self._vars:
      # Here we handle the case when returning an existing variable.
      if reuse is False:
        var = self._vars[name]
        err_msg = ("Variable %s already exists, disallowed."
                   " Did you mean to set reuse=True or "
                   "reuse=tf.AUTO_REUSE in VarScope?" % name)
        # ResourceVariables don't have an op associated with so no traceback
        if isinstance(var, resource_variable_ops.ResourceVariable):
          raise ValueError(err_msg)
        tb = var.op.traceback[::-1]
        # Throw away internal tf entries and only take a few lines. In some
        # cases the traceback can be longer (e.g. if someone uses factory
        # functions to create variables) so we take more than needed in the
        # default case.
        tb = [x for x in tb if "tensorflow/python" not in x[0]][:5]
        raise ValueError("%s Originally defined at:\n\n%s" %
                         (err_msg, "".join(traceback.format_list(tb))))
      found_var = self._vars[name]
      if shape is not None and not shape.is_compatible_with(
          found_var.get_shape()
      ):
        raise ValueError("Trying to share variable %s, but specified shape %s"
                         " and found shape %s." %
                         (name, shape, found_var.get_shape()))
      if not dtype.is_compatible_with(found_var.dtype):
        dtype_str = dtype.name
        found_type_str = found_var.dtype.name
        raise ValueError("Trying to share variable %s, but specified dtype %s"
                         " and found dtype %s." %
                         (name, dtype_str, found_type_str))
      return found_var

    # The code below handles only the case of creating a new variable.
    if reuse is True:
      raise ValueError("Variable %s does not exist, or was not created with "
                       "tf.get_variable(). Did you mean to set "
                       "reuse=tf.AUTO_REUSE in VarScope?" % name)

    # Create the tensor to initialize the variable with default value.
    if initializer is None:
      if shape is None:
        raise ValueError(
            f"Variable {name} did not get an initializer, so its `shape`"
            " argument must be specified."
        )
      initializer, initializing_from_value = self._get_default_initializer(
          name=name, shape=shape, dtype=dtype)
    # Enter an init scope when creating the initializer.
    with ops.init_scope():
      if initializing_from_value:
        init_val = initializer
        variable_dtype = None
      else:
        # Instantiate initializer if provided initializer is a type object.
        if tf_inspect.isclass(initializer):
          initializer = initializer()
        if shape is not None and shape.is_fully_defined():
          if "partition_info" in tf_inspect.getargspec(initializer).args:
            init_val = functools.partial(initializer,
                                         shape.as_list(),
                                         dtype=dtype,
                                         partition_info=partition_info)
          else:
            init_val = functools.partial(initializer,
                                         shape.as_list(), dtype=dtype)
          variable_dtype = dtype.base_dtype
        elif _needs_no_arguments(initializer):
          init_val = initializer
          variable_dtype = None
        else:
          raise ValueError("The initializer passed is not valid. It should "
                           "be a callable with no arguments and the "
                           "shape should not be provided or an instance of "
                           "`tf.keras.initializers.*' and `shape` should be "
                           "fully defined.")

    # Create the variable.
    if use_resource is None:
      # Set the default value if unspecified.
      use_resource = resource_variables_toggle.resource_variables_enabled()
    v = _variable_v1(
        initial_value=init_val,
        name=name,
        trainable=trainable,
        collections=collections,
        caching_device=caching_device,
        dtype=variable_dtype,
        validate_shape=validate_shape,
        constraint=constraint,
        use_resource=use_resource,
        synchronization=synchronization,
        aggregation=aggregation,
        shape=shape,
    )
    if context.executing_eagerly() and self._store_eager_variables:
      if collections:
        ops.add_to_collections(collections, v)
      else:
        ops.add_to_collection(ops.GraphKeys.GLOBAL_VARIABLES, v)
      if trainable:
        ops.add_to_collection(ops.GraphKeys.TRAINABLE_VARIABLES, v)

    if not context.executing_eagerly() or self._store_eager_variables:
      # In eager mode we do not want to keep default references to Variable
      # objects as this will prevent their memory from being released.
      self._vars[name] = v
    logging.vlog(1, "Created variable %s with shape %s and init %s", v.name,
                 format(shape), initializer)

    # Run the regularizer if requested and save the resulting loss.
    if regularizer:
      def make_regularizer_op():
        # [FUNCTION DOC]
        # HELPER: thunk used by the lazily evaluated regularization loss tensor.
        # Applies the regularizer to the new variable v, colocated with it and inside a name
        # scope named after the variable plus a regularizer suffix; returns the resulting loss
        # tensor (or the null value if the regularizer returns nothing). Each call builds fresh ops.
        with ops.colocate_with(v):
          with ops.name_scope(name + "/Regularizer/"):
            return regularizer(v)

      if regularizer(v) is not None:
        lazy_eval_tensor = _LazyEvalTensor(make_regularizer_op)
        ops.add_to_collection(ops.GraphKeys.REGULARIZATION_LOSSES,
                              lazy_eval_tensor)

    return v

  # Initialize variable when no initializer provided
  def _get_default_initializer(self, name, shape=None, dtype=dtypes.float32):
    # [FUNCTION DOC]
    # HELPER: picks the initializer used when the caller supplied none.
    # KNOWN ISSUE: an inline comment says floats get a uniform unit scaling initializer, but the
    # code returns the Glorot uniform initializer. Complex and other non-listed dtypes are not
    # supported despite a note questioning it.
    # Inputs: name (used only in the error message), shape (ignored), dtype. Floating types get
    # a Glorot uniform initializer; integer, unsigned, boolean and string types get a zeros
    # initializer. Returns a pair of the initializer and a flag that is always false here (the
    # initializer is not a constant value). Raises a value error for any other dtype.
    """Provide a default initializer and a corresponding value.

    Args:
      name: see get_variable.
      shape: see get_variable.
      dtype: see get_variable.

    Returns:
      initializer and initializing_from_value. See get_variable above.

    Raises:
      ValueError: When giving unsupported dtype.
    """
    del shape
    # If dtype is DT_FLOAT, provide a uniform unit scaling initializer
    if dtype.is_floating:
      initializer = init_ops.glorot_uniform_initializer()
      initializing_from_value = False
    # If dtype is DT_INT/DT_UINT, provide a default value `zero`
    # If dtype is DT_BOOL, provide a default value `FALSE`
    elif (dtype.is_integer or dtype.is_unsigned or dtype.is_bool or
          dtype == dtypes.string):
      initializer = init_ops.zeros_initializer()
      initializing_from_value = False
    # NOTES:Do we need to support for handling DT_STRING and DT_COMPLEX here?
    else:
      raise ValueError("An initializer for variable %s of %s is required" %
                       (name, dtype.base_dtype))

    return initializer, initializing_from_value


class _LazyEvalTensor(core.Tensor):
  # [CLASS DOC]
  # HELPER: internal tensor proxy used for regularization-loss collection entries.
  # KNOWN ISSUE: the docstring says the thunk is only evaluated when used, but the constructor
  # calls the thunk immediately to build a master tensor (used for dtype/shape/name queries);
  # the thunk is called again on every conversion to a real tensor.
  # Tensor-like wrapper holding a callable that builds a tensor. Metadata properties and some
  # methods (device, dtype, graph, name, op, shape, value_index, get_shape, string conversion) are
  # forwarded to the master tensor; arithmetic and other operator methods rebuild the tensor by
  # calling the thunk afresh so the value can be used in whatever graph/function is current.
  # Forwarding members are attached to the class after its definition. It is also registered with
  # the tensor conversion registry and the session fetch mechanism at import time.
  """A Tensor-like object that only evaluates its thunk when used."""

  def __init__(self, thunk):
    # [FUNCTION DOC]
    # Stores the thunk (callable with no arguments returning a tensor) and immediately calls it
    # once to produce the master tensor used for metadata. Returns nothing.
    """Initializes a _LazyEvalTensor object.

    Args:
      thunk: A callable. A thunk which computes the value of the tensor.
    """
    self._thunk = thunk
    self._master_tensor = thunk()

  def _as_tensor(self, dtype=None, name=None, as_ref=False):
    # [FUNCTION DOC]
    # Conversion hook: builds and returns a fresh tensor by calling the thunk. The name argument
    # is ignored. Uses assertions (not proper errors): as_ref must be false, and dtype must be
    # null or equal to the tensor's dtype, otherwise an assertion error is raised.
    del name
    assert not as_ref
    assert dtype in [None, self.dtype]

    return self._thunk()


def _make_master_property(name):
  # [FUNCTION DOC]
  # HELPER: factory used at import time to attach forwarding properties to _LazyEvalTensor.
  # Takes an attribute name and returns a read-only property that fetches that attribute from the
  # instance's master tensor.
  @property
  def prop(self):
    # [FUNCTION DOC]
    # HELPER: nested getter returned by the property factory.
    # Returns the named attribute of the master tensor held by this lazy tensor.
    return getattr(self._master_tensor, name)  # pylint: disable=protected-access
  return prop

_master_property_list = ("device", "dtype", "graph", "name", "op", "shape",
                         "value_index")
for _name in _master_property_list:
  setattr(_LazyEvalTensor, _name, _make_master_property(_name))


def _make_master_method(name):
  # [FUNCTION DOC]
  # HELPER: factory used at import time to attach methods to _LazyEvalTensor that forward to the
  # master tensor. Takes a method name and returns a method with that behavior.
  def method(self, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: nested forwarding method. Calls the named method on the master tensor with the same
    # positional and keyword arguments and returns its result.
    return getattr(self._master_tensor, name)(*args, **kwargs)  # pylint: disable=protected-access
  return method

_master_method_list = ("get_shape", "__str__", "shape_as_list")
for _name in _master_method_list:
  setattr(_LazyEvalTensor, _name, _make_master_method(_name))


def _make_op_method(name):
  # [FUNCTION DOC]
  # HELPER: factory used at import time to attach operator-style methods (arithmetic, comparison,
  # indexing, evaluation, numpy conversion) to _LazyEvalTensor. Takes a method name and returns
  # a method with that behavior.
  def method(self, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: nested operator method. Rebuilds a fresh tensor by running the thunk, then calls
    # the named method on it with the given arguments and returns the result.
    return getattr(self._as_tensor(), name)(*args, **kwargs)  # pylint: disable=protected-access
  return method

_op_list = ("__abs__", "__add__", "__and__", "__bool__", "__div__", "__eq__",
            "__floordiv__", "__ge__", "__getitem__", "__gt__", "__invert__",
            "__iter__", "__le__", "__len__", "__lt__", "__matmul__", "__mod__",
            "__mul__", "__ne__", "__neg__", "__nonzero__", "__or__", "__pow__",
            "__radd__", "__rand__", "__rdiv__", "__rfloordiv__", "__rmatmul__",
            "__rmod__", "__rmul__", "__ror__", "__rpow__", "__rsub__",
            "__rtruediv__", "__rxor__", "__sub__", "__truediv__", "__xor__",
            "eval", "numpy")
for _name in _op_list:
  setattr(_LazyEvalTensor, _name, _make_op_method(_name))


tensor_conversion_registry.register_tensor_conversion_function(
    _LazyEvalTensor,
    lambda val, dtype, name, as_ref: val._as_tensor(dtype, name, as_ref)  # pylint: disable=protected-access
    )

session.register_session_run_conversion_functions(
    _LazyEvalTensor,
    lambda fetch: ([fetch._master_tensor], lambda fetched_vals: fetched_vals[0])  # pylint: disable=protected-access
    )


# To stop regularization, use this regularizer
@tf_export(v1=["no_regularizer"])
def no_regularizer(_):
  # [FUNCTION DOC]
  # Regularizer that does nothing: ignores its single argument (a variable) and returns the null
  # value, so no regularization loss is produced. Pass it to override an inherited scope
  # regularizer. Exported as a v1 API.
  """Use this function to prevent regularization of variables."""
  return None


# TODO(alive): support caching devices and partitioned variables in Eager mode.
@tf_export(v1=["VariableScope"])
class VariableScope:
  # [CLASS DOC]
  # KNOWN ISSUE: the docstring says reuse is always forced to false in eager mode, but the
  # constructor sets it to auto-reuse and only get_variable forces it to false at call time.
  # Immutable-name bundle of default settings used by get_variable: name prefix, initializer,
  # regularizer, reuse mode, caching device, partitioner, custom getter, original name scope,
  # dtype, use_resource and constraint. Offers read-only properties, setters for the mutable
  # defaults, collection lookups restricted to the scope's name prefix, and get_variable and a
  # private partitioned-variable method that fill in missing arguments from these defaults before
  # delegating to a variable store. Exported as the v1 VariableScope API; instances are created by
  # the variable_scope machinery and held by the thread-local scope store.
  """Variable scope object to carry defaults to provide to `get_variable`.

  Many of the arguments we need for `get_variable` in a variable store are most
  easily handled with a context. This object is used for the defaults.

  Attributes:
    name: name of the current scope, used as prefix in get_variable.
    initializer: default initializer passed to get_variable.
    regularizer: default regularizer passed to get_variable.
    reuse: Boolean, None, or tf.compat.v1.AUTO_REUSE, setting the reuse in
      get_variable. When eager execution is enabled this argument is always
      forced to be False.
    caching_device: string, callable, or None: the caching device passed to
      get_variable.
    partitioner: callable or `None`: the partitioner passed to `get_variable`.
    custom_getter: default custom getter passed to get_variable.
    name_scope: The name passed to `tf.name_scope`.
    dtype: default type passed to get_variable (defaults to DT_FLOAT).
    use_resource: if False, create a normal Variable; if True create an
      experimental ResourceVariable with well-defined semantics. Defaults to
      False (will later change to True). When eager execution is enabled this
      argument is always forced to be True.
    constraint: An optional projection function to be applied to the variable
      after being updated by an `Optimizer` (e.g. used to implement norm
      constraints or value constraints for layer weights). The function must
      take as input the unprojected Tensor representing the value of the
      variable and return the Tensor for the projected value (which must have
      the same shape). Constraints are not safe to use when doing asynchronous
      distributed training.
  """

  def __init__(self,
               reuse,
               name="",
               initializer=None,
               regularizer=None,
               caching_device=None,
               partitioner=None,
               custom_getter=None,
               name_scope="",
               dtype=dtypes.float32,
               use_resource=None,
               constraint=None):
    # [FUNCTION DOC]
    # Stores the given defaults (reuse is required; all others optional: name prefix defaults to
    # empty, dtype defaults to 32-bit float, name_scope defaults to empty). In eager mode, raises
    # a not-implemented error if a caching device was given, and forces reuse to auto-reuse and
    # use_resource to true. Returns nothing.
    """Creates a new VariableScope with the given properties."""
    self._name = name
    self._initializer = initializer
    self._regularizer = regularizer
    self._reuse = reuse
    self._caching_device = caching_device
    self._partitioner = partitioner
    self._custom_getter = custom_getter
    self._name_scope = name_scope
    self._dtype = dtype
    self._use_resource = use_resource
    self._constraint = constraint
    if context.executing_eagerly():
      if self._caching_device is not None:
        raise NotImplementedError("Caching devices is not yet supported "
                                  "when eager execution is enabled.")
      self._reuse = AUTO_REUSE
      self._use_resource = True

  @property
  def name(self):
    # [FUNCTION DOC]
    # Read-only property: the full name (prefix) of this scope; empty string for the root scope.
    return self._name

  @property
  def original_name_scope(self):
    # [FUNCTION DOC]
    # Read-only property: the name-scope string that was active when this variable scope was
    # created (used to re-enter the same name scope later).
    return self._name_scope

  @property
  def reuse(self):
    # [FUNCTION DOC]
    # Read-only property: the reuse setting (true, false, null, or auto-reuse) of this scope.
    return self._reuse

  @property
  def initializer(self):
    # [FUNCTION DOC]
    # Read-only property: the default initializer for variables created in this scope, or null.
    return self._initializer

  @property
  def dtype(self):
    # [FUNCTION DOC]
    # Read-only property: the default data type for variables created in this scope.
    return self._dtype

  @property
  def use_resource(self):
    # [FUNCTION DOC]
    # Read-only property: whether resource variables are the default here (true, false or null
    # meaning use the global default).
    return self._use_resource

  @property
  def regularizer(self):
    # [FUNCTION DOC]
    # Read-only property: the default regularizer callable for this scope, or null.
    return self._regularizer

  @property
  def caching_device(self):
    # [FUNCTION DOC]
    # Read-only property: the default caching device (string or callable) for this scope, or null.
    return self._caching_device

  @property
  def partitioner(self):
    # [FUNCTION DOC]
    # Read-only property: the default partitioner callable for this scope, or null.
    return self._partitioner

  @property
  def custom_getter(self):
    # [FUNCTION DOC]
    # Read-only property: the default custom getter callable for this scope, or null.
    return self._custom_getter

  @property
  def constraint(self):
    # [FUNCTION DOC]
    # Read-only property: the default variable constraint function for this scope, or null.
    return self._constraint

  def reuse_variables(self):
    # [FUNCTION DOC]
    # Switches this scope to reuse mode (sets reuse to true) so later get_variable calls must
    # find existing variables. Mutates the scope; takes no arguments and returns nothing.
    """Reuse variables in this scope."""
    self._reuse = True

  def set_initializer(self, initializer):
    # [FUNCTION DOC]
    # Sets the default initializer (callable, constant value, or null) of this scope. Mutates
    # the scope; returns nothing.
    """Set initializer for this scope."""
    self._initializer = initializer

  def set_dtype(self, dtype):
    # [FUNCTION DOC]
    # Sets the default data type of this scope. Mutates the scope; no validation; returns nothing.
    """Set data type for this scope."""
    self._dtype = dtype

  def set_use_resource(self, use_resource):
    # [FUNCTION DOC]
    # Sets whether resource variables are used by default in this scope. Raises a value error if
    # eager execution is enabled and the new value is false (or otherwise falsy, including null).
    # Mutates the scope; returns nothing.
    """Sets whether to use ResourceVariables for this scope."""
    if context.executing_eagerly() and not use_resource:
      raise ValueError("When eager execution is enabled, "
                       "use_resource cannot be set to false.")
    self._use_resource = use_resource

  def set_regularizer(self, regularizer):
    # [FUNCTION DOC]
    # Sets the default regularizer (callable or null) of this scope. Mutates the scope; returns
    # nothing.
    """Set regularizer for this scope."""
    self._regularizer = regularizer

  def set_caching_device(self, caching_device):
    # [FUNCTION DOC]
    # Sets the default caching device (string or callable) of this scope. Raises a
    # not-implemented error whenever eager execution is enabled (even when clearing it). Mutates
    # the scope; returns nothing.
    """Set caching_device for this scope."""
    if context.executing_eagerly():
      raise NotImplementedError("Caching devices are not yet supported "
                                "when eager execution is enabled.")
    self._caching_device = caching_device

  def set_partitioner(self, partitioner):
    # [FUNCTION DOC]
    # Sets the default partitioner (callable or null) of this scope. Mutates the scope; returns
    # nothing.
    """Set partitioner for this scope."""
    self._partitioner = partitioner

  def set_custom_getter(self, custom_getter):
    # [FUNCTION DOC]
    # Sets the default custom getter (callable or null) of this scope. Mutates the scope; returns
    # nothing.
    """Set custom getter for this scope."""
    self._custom_getter = custom_getter

  def get_collection(self, name):
    # [FUNCTION DOC]
    # Returns the items of the named graph collection whose names start with this scope's name
    # followed by a slash (all items if the scope name is empty). name is the collection key.
    # Reads the default graph; no mutation.
    """Get this scope's variables."""
    scope = self._name + "/" if self._name else ""
    return ops.get_collection(name, scope)

  def trainable_variables(self):
    # [FUNCTION DOC]
    # Returns the list of trainable variables whose names fall under this scope's prefix, read
    # from the trainable-variables graph collection.
    """Get this scope's trainable variables."""
    return self.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)

  def global_variables(self):
    # [FUNCTION DOC]
    # Returns the list of global variables whose names fall under this scope's prefix, read from
    # the global-variables graph collection.
    """Get this scope's global variables."""
    return self.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)

  def local_variables(self):
    # [FUNCTION DOC]
    # Returns the list of local variables whose names fall under this scope's prefix, read from
    # the local-variables graph collection.
    """Get this scope's local variables."""
    return self.get_collection(ops.GraphKeys.LOCAL_VARIABLES)

  def get_variable(self,
                   var_store,
                   name,
                   shape=None,
                   dtype=None,
                   initializer=None,
                   regularizer=None,
                   reuse=None,
                   trainable=None,
                   collections=None,
                   caching_device=None,
                   partitioner=None,
                   validate_shape=True,
                   use_resource=None,
                   custom_getter=None,
                   constraint=None,
                   synchronization=VariableSynchronization.AUTO,
                   aggregation=VariableAggregation.NONE):
    # [FUNCTION DOC]
    # Scope-aware get_variable. Takes the variable store plus the usual get_variable arguments
    # (name required). Fills in missing regularizer, caching device, partitioner and custom
    # getter from the scope; in eager mode forces reuse to false and use_resource to true,
    # otherwise falls back to the scope's reuse and use_resource. Builds the full name by
    # prefixing the scope name, then, outside any name scope, checks that a non-callable
    # initializer's dtype matches an explicit dtype (value error if not), fills in initializer,
    # constraint and dtype from the scope, and delegates to the store's get_variable. Returns the
    # variable (or partitioned variable). Errors from the store propagate.
    """Gets an existing variable with this name or create a new one."""
    if regularizer is None:
      regularizer = self._regularizer
    if caching_device is None:
      caching_device = self._caching_device
    if partitioner is None:
      partitioner = self._partitioner
    if custom_getter is None:
      custom_getter = self._custom_getter
    if context.executing_eagerly():
      reuse = False
      use_resource = True
    else:
      if reuse is None:
        reuse = self._reuse
      if use_resource is None:
        use_resource = self._use_resource

    full_name = self.name + "/" + name if self.name else name
    # Variable names only depend on variable_scope (full_name here),
    # not name_scope, so we reset it below for the time of variable creation.
    with ops.name_scope(None, skip_on_eager=False):
      # Check that `initializer` dtype and `dtype` are consistent before
      # replacing them with defaults.
      if (dtype is not None and initializer is not None and
          not callable(initializer)):
        init_dtype = ops.convert_to_tensor(initializer).dtype.base_dtype
        if init_dtype != dtype:
          raise ValueError("Initializer type '%s' and explicit dtype '%s' "
                           "don't match." % (init_dtype, dtype))
      if initializer is None:
        initializer = self._initializer
      if constraint is None:
        constraint = self._constraint
      if dtype is None:
        dtype = self._dtype
      return var_store.get_variable(
          full_name,
          shape=shape,
          dtype=dtype,
          initializer=initializer,
          regularizer=regularizer,
          reuse=reuse,
          trainable=trainable,
          collections=collections,
          caching_device=caching_device,
          partitioner=partitioner,
          validate_shape=validate_shape,
          use_resource=use_resource,
          custom_getter=custom_getter,
          constraint=constraint,
          synchronization=synchronization,
          aggregation=aggregation)

  def _get_partitioned_variable(self,
                                var_store,
                                name,
                                shape=None,
                                dtype=None,
                                initializer=None,
                                regularizer=None,
                                trainable=None,
                                collections=None,
                                caching_device=None,
                                partitioner=None,
                                validate_shape=True,
                                use_resource=None,
                                constraint=None,
                                synchronization=VariableSynchronization.AUTO,
                                aggregation=VariableAggregation.NONE):
    # [FUNCTION DOC]
    # HELPER: scope-level support for create_partitioned_variables and partitioned get_variable.
    # KNOWN ISSUE: the docstring is copied from get_variable; this method only gets or creates a
    # partitioned variable, has no reuse argument (it always passes the scope's own reuse, even in
    # eager mode) and requires a partitioner.
    # Fills missing initializer, regularizer, constraint, caching device, partitioner, dtype and
    # use_resource from scope defaults. Raises a value error if the scope has a custom getter or
    # if no partitioner is available. Forms the full name from the scope name and name (either may
    # be empty), then calls the store's private partitioned-variable method with name scope reset.
    # Returns the partitioned variable.
    """Gets an existing variable with this name or create a new one."""
    if initializer is None:
      initializer = self._initializer
    if regularizer is None:
      regularizer = self._regularizer
    if constraint is None:
      constraint = self._constraint
    if caching_device is None:
      caching_device = self._caching_device
    if partitioner is None:
      partitioner = self._partitioner
    if dtype is None:
      dtype = self._dtype
    if use_resource is None:
      use_resource = self._use_resource

    if self._custom_getter is not None:
      raise ValueError(
          "Private access to _get_partitioned_variable is not allowed when "
          "a custom getter is set.  Current custom getter: %s.  "
          "It is likely that you're using create_partitioned_variables.  "
          "If so, consider instead using get_variable with a non-empty "
          "partitioner parameter instead." % self._custom_getter)

    if partitioner is None:
      raise ValueError("No partitioner was specified")

    # This allows the variable scope name to be used as the variable name if
    # this function is invoked with an empty name arg, for backward
    # compatibility with create_partitioned_variables().
    full_name_list = []
    if self.name:
      full_name_list.append(self.name)
    if name:
      full_name_list.append(name)
    full_name = "/".join(full_name_list)

    # Variable names only depend on variable_scope (full_name here),
    # not name_scope, so we reset it below for the time of variable creation.
    with ops.name_scope(None, skip_on_eager=False):
      # pylint: disable=protected-access
      return var_store._get_partitioned_variable(
          full_name,
          shape=shape,
          dtype=dtype,
          initializer=initializer,
          regularizer=regularizer,
          reuse=self.reuse,
          trainable=trainable,
          collections=collections,
          caching_device=caching_device,
          partitioner=partitioner,
          validate_shape=validate_shape,
          use_resource=use_resource,
          constraint=constraint,
          synchronization=synchronization,
          aggregation=aggregation)
      # pylint: enable=protected-access


_VARSTORE_KEY = ("__variable_store",)
_VARSCOPESTORE_KEY = ("__varscope",)


class _VariableScopeStore(threading.local):
  # [CLASS DOC]
  # HELPER: internal state holder for the variable_scope machinery.
  # Thread-local record of the current VariableScope and a mapping from scope names to how many
  # times each has been opened, used to generate unique default scope names and to detect scope
  # reopening. One instance lives in a graph collection (see get_variable_scope_store).
  """A thread local store for the current variable scope and scope counts."""

  def __init__(self):
    # [FUNCTION DOC]
    # Initializes per-thread state: the current scope is a fresh root VariableScope with reuse
    # false, and the scope-count mapping is empty. Returns nothing.
    super(_VariableScopeStore, self).__init__()
    self.current_scope = VariableScope(False)
    self.variable_scopes_count = {}

  def open_variable_scope(self, scope_name):
    # [FUNCTION DOC]
    # Increments the open count for scope_name (string), starting it at one if unseen. Mutates
    # the count mapping; returns nothing.
    if scope_name in self.variable_scopes_count:
      self.variable_scopes_count[scope_name] += 1
    else:
      self.variable_scopes_count[scope_name] = 1

  def close_variable_subscopes(self, scope_name):
    # [FUNCTION DOC]
    # Resets the open counts to zero for sub-scopes. If scope_name is null, resets every recorded
    # scope; otherwise resets only scopes whose names begin with scope_name followed by a slash
    # (the scope itself is not reset). Mutates the count mapping; returns nothing.
    if scope_name is None:
      for k in self.variable_scopes_count:
        self.variable_scopes_count[k] = 0
    else:
      startswith_check = scope_name + "/"
      startswith_len = len(startswith_check)
      for k in self.variable_scopes_count:
        if k[:startswith_len] == startswith_check:
          self.variable_scopes_count[k] = 0

  def variable_scope_count(self, scope_name):
    # [FUNCTION DOC]
    # Returns how many times scope_name has been opened (an integer, zero if never opened).
    return self.variable_scopes_count.get(scope_name, 0)


def get_variable_scope_store():
  # [FUNCTION DOC]
  # HELPER: accessor for the scope store used by the variable_scope machinery.
  # KNOWN ISSUE: the docstring says the store is per thread; it is looked up in a graph
  # collection (so it is per graph), and only its contents are thread-local.
  # Looks up the scope store in the default graph's collection; if absent, creates one and adds it
  # to the collection (side effect on the graph). Returns the store.
  """Returns the variable scope store for current thread."""
  scope_store = ops.get_collection(_VARSCOPESTORE_KEY)

  if not scope_store:
    scope_store = _VariableScopeStore()
    ops.add_to_collection(_VARSCOPESTORE_KEY, scope_store)
  else:
    scope_store = scope_store[0]

  return scope_store


@tf_export(v1=["get_variable_scope"])
def get_variable_scope():
  # [FUNCTION DOC]
  # KNOWN ISSUE: the example in the docstring is not valid (it assigns the result of opening a
  # name scope without a with statement); it only illustrates intent.
  # Returns the current VariableScope object, taken from the scope store of the default graph.
  # No arguments, no mutation of the scope. Exported as the v1 API.
  """Returns the current variable scope.

  @compatibility(TF2)
  Although it is a legacy `compat.v1` api,
  `tf.compat.v1.get_variable` is compatible with eager
  execution and `tf.function`

  However, to maintain variable-scope based variable reuse
  you will need to combine it with
  `tf.compat.v1.keras.utils.track_tf1_style_variables`. (Though
  it will behave as if reuse is always set to `tf.compat.v1.AUTO_REUSE`.)

  See the
  [migration guide](https://www.tensorflow.org/guide/migrate/model_mapping)
  for more info.

  The TF2 equivalent, if you are just trying to track
  variable name prefixes and not control `get_variable`-based variable reuse,
  would be to use `tf.name_scope` and capture the output of opening the
  scope (which represents the current name prefix).

  For example:
  ```python
  x = tf.name_scope('foo') as current_scope:
    ...
  ```
  @end_compatibility
  """
  return get_variable_scope_store().current_scope


def _get_default_variable_store():
  # [FUNCTION DOC]
  # HELPER: returns the variable store of the default graph, used by get_variable and
  # variable_scope. Reads the store collection; if empty, creates a new store, registers it in
  # the collection (side effect) and returns it.
  store = ops.get_collection(_VARSTORE_KEY)
  if store:
    return store[0]
  store = _VariableStore()
  ops.add_to_collection(_VARSTORE_KEY, store)
  return store


@tf_contextlib.contextmanager
def with_variable_store(store):
  # [FUNCTION DOC]
  # Context manager that temporarily makes the given variable store the default one for the
  # current graph: replaces the store collection contents with the store on entry and restores
  # the previous contents on exit, even if the body raises. Yields nothing.
  store_collection = ops.get_collection_ref(_VARSTORE_KEY)
  old = list(store_collection)
  store_collection[:] = [store]
  try:
    yield
  finally:
    store_collection[:] = old


class EagerVariableStore:
  # [CLASS DOC]
  # KNOWN ISSUE: the docstring example refers to the long-gone tfe module name and a possibly
  # outdated usage; the class itself is a thin wrapper.
  # Wrapper around a variable store that is switched into eager-variable-storing mode so legacy
  # functional layers can reuse and list variables while eager execution is enabled. Provides a
  # context manager to make it the default store, accessors for stored variables, and helpers to
  # copy or filter them.
  """Wrapper allowing functional layers to be used with eager execution.

  When eager execution is enabled Variables get deleted when they go out of
  scope, and are not stored in global collections by default. A lot of code
  (mostly the functional layers in tf.layers) assumes that variables are kept in
  a global list.

  EagerVariableStore can be used in conjunction with this code to make it
  eager-friendly. For example, to create a dense layer, use:

  ```
    container = tfe.EagerVariableStore()
    for input in dataset_iterator:
      with container.as_default():
        x = tf.compat.v1.layers.dense(input, name="l1")
    print(container.variables)  # Should print the variables used in the layer.
  ```
  """

  def __init__(self, store=None):
    # [FUNCTION DOC]
    # Wraps an existing store or creates a new one. If store is given, it must already hold eager
    # variables, otherwise a value error is raised. In both cases the wrapped store is marked as
    # storing eager variables (mutation). Returns nothing.
    if store is not None:
      if not store._store_eager_variables:  # pylint: disable=protected-access
        raise ValueError("Cannot construct EagerVariableStore from a "
                         "VariableStore object that does not hold eager "
                         "variables.")
      self._store = store
    else:
      self._store = _VariableStore()
    self._store._store_eager_variables = True  # pylint: disable=protected-access

  def as_default(self):
    # [FUNCTION DOC]
    # Returns a context manager that makes the wrapped store the default variable store of the
    # current graph for the duration of a with block (see the store-swapping context manager).
    return with_variable_store(self._store)

  def variables(self):
    # [FUNCTION DOC]
    # Returns a list of all variables held by the wrapped store, sorted by variable name.
    # Does not include partitioned-variable wrappers, only stored variables.
    return sorted(self._store._vars.values(), key=lambda x: x.name)  # pylint: disable=protected-access

  def trainable_variables(self):
    # [FUNCTION DOC]
    # Returns a list of the wrapped store's variables whose trainable flag is true, sorted by name.
    # pylint: disable=protected-access
    return sorted([x for x in self._store._vars.values() if x.trainable],
                  key=lambda x: x.name)
    # pylint: enable=protected-access

  def non_trainable_variables(self):
    # [FUNCTION DOC]
    # Returns a list of the wrapped store's variables whose trainable flag is false, sorted by
    # name.
    # pylint: disable=protected-access
    return sorted([x for x in self._store._vars.values() if not x.trainable],
                  key=lambda x: x.name)
    # pylint: enable=protected-access

  def copy(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: only plain stored variables are copied; partitioned variables registered in
    # the store are silently dropped, and copied variables are always resource variables with the
    # current value but no initializer, constraint or other settings of the originals.
    # Creates a new EagerVariableStore and, for every variable in this store, makes a new
    # resource variable initialized from the current value, with the old name stripped of its
    # output-index suffix and the same trainable flag, stored under the same key. Returns the new
    # store; this store is unchanged.
    """Copy this variable store and all of its contents.

    Variables contained in this store will be copied over to the new variable
    store, meaning that they can be modified without affecting the variables in
    this store.

    Returns:
      A new EagerVariableStore instance containing copied variables.
    """
    # pylint: disable=protected-access
    new_store = EagerVariableStore()
    for key, var in self._store._vars.items():
      # Strip device out of variable name.
      try:
        index = var.name.index(":")
      except ValueError:
        stripped_var_name = var.name
      else:
        stripped_var_name = var.name[:index]

      # Create new variable with same value, name, and "trainable" flag.
      new_var = resource_variable_ops.ResourceVariable(
          var.read_value(), name=stripped_var_name, trainable=var.trainable)
      new_store._store._vars[key] = new_var
    return new_store
    # pylint: enable=protected-access


# The argument list for get_variable must match arguments to get_local_variable.
# So, if you are updating the arguments, also update arguments to
# get_local_variable below.
@tf_export(v1=["get_variable"])
def get_variable(name,
                 shape=None,
                 dtype=None,
                 initializer=None,
                 regularizer=None,
                 trainable=None,
                 collections=None,
                 caching_device=None,
                 partitioner=None,
                 validate_shape=True,
                 use_resource=None,
                 custom_getter=None,
                 constraint=None,
                 synchronization=VariableSynchronization.AUTO,
                 aggregation=VariableAggregation.NONE):
  # [FUNCTION DOC]
  # Public v1 get_variable. Its documentation is attached after definition from a shared template.
  # Gets an existing variable by name inside the current variable scope or creates a new one.
  # name is required; shape, dtype, initializer, regularizer, trainable, collections,
  # caching_device, partitioner, validate_shape, use_resource, custom_getter, constraint,
  # synchronization and aggregation are optional and fall back to scope defaults. There is no
  # reuse argument; reuse comes from the current scope. Delegates to the current scope's
  # get_variable using the default graph's variable store. Returns a variable or partitioned
  # variable. Raises a value error on reuse violations, missing shape or dtype mismatch.
  return get_variable_scope().get_variable(
      _get_default_variable_store(),
      name,
      shape=shape,
      dtype=dtype,
      initializer=initializer,
      regularizer=regularizer,
      trainable=trainable,
      collections=collections,
      caching_device=caching_device,
      partitioner=partitioner,
      validate_shape=validate_shape,
      use_resource=use_resource,
      custom_getter=custom_getter,
      constraint=constraint,
      synchronization=synchronization,
      aggregation=aggregation)


get_variable_or_local_docstring = ("""%s

@compatibility(TF2)
Although it is a legacy `compat.v1` api,
`tf.compat.v1.get_variable` is mostly compatible with eager
execution and `tf.function` but only if you combine it with the
`tf.compat.v1.keras.utils.track_tf1_style_variables` decorator. (Though
it will behave as if reuse is always set to `AUTO_REUSE`.)

See the
[model migration guide](https://www.tensorflow.org/guide/migrate/model_mapping)
for more info.

If you do not combine it with
`tf.compat.v1.keras.utils.track_tf1_style_variables`, `get_variable` will create
a brand new variable every single time it is called and will never reuse
variables, regardless of variable names or `reuse` arguments.

The TF2 equivalent of this symbol would be `tf.Variable`, but note
that when using `tf.Variable` you must make sure you track your variables
(and regularizer arguments) either manually or via `tf.Module` or
`tf.keras.layers.Layer` mechanisms.

A section of the
[migration guide](https://www.tensorflow.org/guide/migrate/model_mapping#incremental_migration_to_native_tf2)
provides more details on incrementally migrating these usages to `tf.Variable`
as well.

Note: The `partitioner` arg is not compatible with TF2 behaviors even when
using `tf.compat.v1.keras.utils.track_tf1_style_variables`. It can be replaced
by using `ParameterServerStrategy` and its partitioners. See the
[multi-gpu migration guide](https://www.tensorflow.org/guide/migrate/multi_worker_cpu_gpu_training)
and the ParameterServerStrategy guides it references for more info.
@end_compatibility

%sThis function prefixes the name with the current variable scope
and performs reuse checks. See the
[Variable Scope How To](https://tensorflow.org/guide/variables)
for an extensive description of how reusing works. Here is a basic example:

```python
def foo():
  with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
    v = tf.get_variable("v", [1])
  return v

v1 = foo()  # Creates v.
v2 = foo()  # Gets the same, existing v.
assert v1 == v2
```

If initializer is `None` (the default), the default initializer passed in
the variable scope will be used. If that one is `None` too, a
`glorot_uniform_initializer` will be used. The initializer can also be
a Tensor, in which case the variable is initialized to this value and shape.

Similarly, if the regularizer is `None` (the default), the default regularizer
passed in the variable scope will be used (if that is `None` too,
then by default no regularization is performed).

If a partitioner is provided, a `PartitionedVariable` is returned.
Accessing this object as a `Tensor` returns the shards concatenated along
the partition axis.

Some useful partitioners are available.  See, e.g.,
`variable_axis_size_partitioner` and `min_max_variable_partitioner`.

Args:
  name: The name of the new or existing variable.
  shape: Shape of the new or existing variable.
  dtype: Type of the new or existing variable (defaults to `DT_FLOAT`).
  initializer: Initializer for the variable if one is created. Can either be
    an initializer object or a Tensor. If it's a Tensor, its shape must be known
    unless validate_shape is False.
  regularizer: A (Tensor -> Tensor or None) function; the result of
    applying it on a newly created variable will be added to the collection
    `tf.GraphKeys.REGULARIZATION_LOSSES` and can be used for regularization.
  %scollections: List of graph collections keys to add the Variable to.
    Defaults to `[%s]` (see `tf.Variable`).
  caching_device: Optional device string or function describing where the
    Variable should be cached for reading.  Defaults to the Variable's
    device.  If not `None`, caches on another device.  Typical use is to
    cache on the device where the Ops using the Variable reside, to
    deduplicate copying through `Switch` and other conditional statements.
  partitioner: Optional callable that accepts a fully defined `TensorShape`
    and `dtype` of the Variable to be created, and returns a list of
    partitions for each axis (currently only one axis can be partitioned).
  validate_shape: If False, allows the variable to be initialized with a
      value of unknown shape. If True, the default, the shape of initial_value
      must be known. For this to be used the initializer must be a Tensor and
      not an initializer object.
  use_resource: If False, creates a regular Variable. If true, creates an
    experimental ResourceVariable instead with well-defined semantics.
    Defaults to False (will later change to True). When eager execution is
    enabled this argument is always forced to be True.
  custom_getter: Callable that takes as a first argument the true getter, and
    allows overwriting the internal get_variable method.
    The signature of `custom_getter` should match that of this method,
    but the most future-proof version will allow for changes:
    `def custom_getter(getter, *args, **kwargs)`.  Direct access to
    all `get_variable` parameters is also allowed:
    `def custom_getter(getter, name, *args, **kwargs)`.  A simple identity
    custom getter that simply creates variables with modified names is:
    ```python
    def custom_getter(getter, name, *args, **kwargs):
      return getter(name + '_suffix', *args, **kwargs)
    ```
  constraint: An optional projection function to be applied to the variable
    after being updated by an `Optimizer` (e.g. used to implement norm
    constraints or value constraints for layer weights). The function must
    take as input the unprojected Tensor representing the value of the
    variable and return the Tensor for the projected value
    (which must have the same shape). Constraints are not safe to
    use when doing asynchronous distributed training.
  synchronization: Indicates when a distributed a variable will be
    aggregated. Accepted values are constants defined in the class
    `tf.VariableSynchronization`. By default the synchronization is set to
    `AUTO` and the current `DistributionStrategy` chooses
    when to synchronize.
  aggregation: Indicates how a distributed variable will be aggregated.
    Accepted values are constants defined in the class
    `tf.VariableAggregation`.

Returns:
  The created or existing `Variable` (or `PartitionedVariable`, if a
  partitioner was used).

Raises:
  ValueError: when creating a new variable and shape is not declared,
    when violating reuse during variable creation, or when `initializer` dtype
    and `dtype` don't match. Reuse is set inside `variable_scope`.
""")
get_variable.__doc__ = get_variable_or_local_docstring % (
    "Gets an existing variable with these parameters or create a new one.", "",
    "trainable: If `True` also add the variable to the graph collection\n"
    "    `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).\n  ",
    "GraphKeys.GLOBAL_VARIABLES")


# The argument list for get_local_variable must match arguments to get_variable.
# So, if you are updating the arguments, also update arguments to get_variable.
@tf_export(v1=["get_local_variable"])
def get_local_variable(  # pylint: disable=missing-docstring
    name,
    shape=None,
    dtype=None,
    initializer=None,
    regularizer=None,
    trainable=False,  # pylint: disable=unused-argument
    collections=None,
    caching_device=None,
    partitioner=None,
    validate_shape=True,
    use_resource=None,
    custom_getter=None,
    constraint=None,
    synchronization=VariableSynchronization.AUTO,
    aggregation=VariableAggregation.NONE):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the trainable argument is ignored (always forced to false), and when a
  # collections list is passed it is extended in place, mutating the caller's list.
  # Public v1 function like get_variable, but the variable is added to the local-variables
  # collection (appended to any given collections) and is never trainable. Same optional
  # arguments as get_variable. Delegates to get_variable and returns its result. Its docs are
  # attached after definition from a shared template.
  if collections:
    collections += [ops.GraphKeys.LOCAL_VARIABLES]
  else:
    collections = [ops.GraphKeys.LOCAL_VARIABLES]
  return get_variable(
      name,
      shape=shape,
      dtype=dtype,
      initializer=initializer,
      regularizer=regularizer,
      trainable=False,
      collections=collections,
      caching_device=caching_device,
      partitioner=partitioner,
      validate_shape=validate_shape,
      use_resource=use_resource,
      synchronization=synchronization,
      aggregation=aggregation,
      custom_getter=custom_getter,
      constraint=constraint)


get_local_variable.__doc__ = get_variable_or_local_docstring % (
    "Gets an existing *local* variable or creates a new one.",
    "Behavior is the same as in `get_variable`, except that variables are\n"
    "added to the `LOCAL_VARIABLES` collection and `trainable` is set to\n"
    "`False`.\n", "", "GraphKeys.LOCAL_VARIABLES")


def _get_partitioned_variable(name,
                              shape=None,
                              dtype=None,
                              initializer=None,
                              regularizer=None,
                              trainable=True,
                              collections=None,
                              caching_device=None,
                              partitioner=None,
                              validate_shape=True,
                              use_resource=None,
                              constraint=None,
                              synchronization=VariableSynchronization.AUTO,
                              aggregation=VariableAggregation.NONE):
  # [FUNCTION DOC]
  # HELPER: private module-level entry used by create_partitioned_variables and similar code.
  # KNOWN ISSUE: the docstring says the default initializer comes from the constructor and that
  # a tuple of shards and partitions is returned; the code returns the partitioned variable
  # object produced by the scope, and it takes its defaults from the current variable scope.
  # Fetches the current variable scope; raises a value error if it has a custom getter. Delegates
  # to the scope's private partitioned-variable method with the default graph's variable store,
  # passing all arguments (name required; trainable defaults to true here). Returns whatever that
  # returns.
  """Gets or creates a sharded variable list with these parameters.

  The `partitioner` must be a callable that accepts a fully defined
  `TensorShape` and returns a sequence of integers (the `partitions`).
  These integers describe how to partition the given sharded `Variable`
  along the given dimension.  That is, `partitions[1] = 3` means split
  the `Variable` into 3 shards along dimension 1.  Currently, sharding along
  only one axis is supported.

  If the list of variables with the given name (prefix) is already stored,
  we return the stored variables. Otherwise, we create a new one.

  If initializer is `None` (the default), the default initializer passed in
  the constructor is used. If that one is `None` too, we use a new
  `glorot_uniform_initializer`. If initializer is a Tensor, we use
  it as a value and derive the shape from the initializer.

  If the initializer is a callable, then it will be called for each
  shard.  Otherwise the initializer should match the shape of the entire
  sharded Variable, and it will be sliced accordingly for each shard.

  Some useful partitioners are available.  See, e.g.,
  `variable_axis_size_partitioner` and `min_max_variable_partitioner`.

  Args:
    name: The name of the new or existing variable.
    shape: Shape of the new or existing variable.
    dtype: Type of the new or existing variable (defaults to `DT_FLOAT`).
    initializer: Initializer for the variable if one is created.
    regularizer: A (Tensor -> Tensor or None) function; the result of applying
      it on a newly created variable will be added to the collection
      GraphKeys.REGULARIZATION_LOSSES and can be used for regularization.
    trainable: If `True` also add the variable to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    collections: List of graph collections keys to add the Variable to. Defaults
      to `[GraphKeys.GLOBAL_VARIABLES]` (see `tf.Variable`).
    caching_device: Optional device string or function describing where the
      Variable should be cached for reading.  Defaults to the Variable's device.
      If not `None`, caches on another device.  Typical use is to cache on the
      device where the Ops using the Variable reside, to deduplicate copying
      through `Switch` and other conditional statements.
    partitioner: Optional callable that accepts a fully defined `TensorShape`
      and `dtype` of the Variable to be created, and returns a list of
      partitions for each axis (currently only one axis can be partitioned).
    validate_shape: If False, allows the variable to be initialized with a value
      of unknown shape. If True, the default, the shape of initial_value must be
      known.
    use_resource: If False, creates a regular Variable. If True, creates an
      experimental ResourceVariable instead which has well-defined semantics.
      Defaults to False (will later change to True).
    constraint: An optional projection function to be applied to the variable
      after being updated by an `Optimizer` (e.g. used to implement norm
      constraints or value constraints for layer weights). The function must
      take as input the unprojected Tensor representing the value of the
      variable and return the Tensor for the projected value (which must have
      the same shape). Constraints are not safe to use when doing asynchronous
      distributed training.
    synchronization: Indicates when a distributed a variable will be aggregated.
      Accepted values are constants defined in the class
      `tf.VariableSynchronization`. By default the synchronization is set to
      `AUTO` and the current `DistributionStrategy` chooses when to synchronize.
    aggregation: Indicates how a distributed variable will be aggregated.
      Accepted values are constants defined in the class
      `tf.VariableAggregation`.

  Returns:
    A tuple `(shards, partitions)` where `shards` is the list of `Variable`
    shards and `partitions` is the output of the partitioner on the input
    shape.

  Raises:
    ValueError: when creating a new variable and shape is not declared,
      or when violating reuse during variable creation. Reuse is set inside
      `variable_scope`.
  """
  # pylint: disable=protected-access
  scope = get_variable_scope()
  if scope.custom_getter is not None:
    raise ValueError(
        "Private access to _get_partitioned_variable is not allowed when "
        "a custom getter is set.  Current custom getter: %s.  "
        "It is likely that you're using create_partitioned_variables.  "
        "If so, consider instead using get_variable with a non-empty "
        "partitioner parameter instead." % scope.custom_getter)
  return scope._get_partitioned_variable(
      _get_default_variable_store(),
      name,
      shape=shape,
      dtype=dtype,
      initializer=initializer,
      regularizer=regularizer,
      trainable=trainable,
      collections=collections,
      caching_device=caching_device,
      partitioner=partitioner,
      validate_shape=validate_shape,
      use_resource=use_resource,
      constraint=constraint,
      synchronization=synchronization,
      aggregation=aggregation)
  # pylint: enable=protected-access


# Named like a function for compatibility with the previous
# @tf_contextlib.contextmanager definition.
class _pure_variable_scope:  # pylint: disable=invalid-name
  # [CLASS DOC]
  # HELPER: implementation core of the variable_scope context manager (name scopes excluded).
  # Context manager that opens either a named child of the current variable scope or a previously
  # captured VariableScope object, installing a new current scope in the thread-local scope store
  # and restoring the previous one on exit. Does not create a name scope.
  """A context for the variable_scope, see `variable_scope` for docs."""

  def __init__(self,
               name_or_scope,
               reuse=None,
               initializer=None,
               regularizer=None,
               caching_device=None,
               partitioner=None,
               custom_getter=None,
               old_name_scope=None,
               dtype=dtypes.float32,
               use_resource=None,
               constraint=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says reuse may only be true, null or auto-reuse, but nothing is
    # validated here; also when re-entering an existing VariableScope object the constraint passed
    # in replaces the scope's own constraint (the old one is not inherited).
    # Records all the options and looks up the default variable store and scope store. When
    # name_or_scope is a VariableScope, immediately builds a copy of it (reuse overridden if a
    # true reuse was given) and applies any non-null overrides for initializer, regularizer,
    # caching device, partitioner, custom getter (wrapped to chain to the old getter), dtype and
    # use_resource; the copy is cached for use on entry. A string name is handled on entry
    # instead. Returns nothing. May raise errors from the setters (e.g. in eager mode).
    """Creates a context for the variable_scope, see `variable_scope` for docs.

    Note: this does not create a name scope.

    Args:
      name_or_scope: `string` or `VariableScope`: the scope to open.
      reuse: `True` or None, or tf.compat.v1.AUTO_REUSE; if `None`, we inherit
        the parent scope's reuse flag.
      initializer: default initializer for variables within this scope.
      regularizer: default regularizer for variables within this scope.
      caching_device: default caching device for variables within this scope.
      partitioner: default partitioner for variables within this scope.
      custom_getter: default custom getter for variables within this scope.
      old_name_scope: the original name scope when re-entering a variable scope.
      dtype: type of the variables within this scope (defaults to `DT_FLOAT`).
      use_resource: If False, variables in this scope will be regular Variables.
        If True, experimental ResourceVariables will be creates instead, with
        well-defined semantics. Defaults to False (will later change to True).
      constraint: An optional projection function to be applied to the variable
        after being updated by an `Optimizer` (e.g. used to implement norm
        constraints or value constraints for layer weights). The function must
        take as input the unprojected Tensor representing the value of the
        variable and return the Tensor for the projected value (which must have
        the same shape). Constraints are not safe to use when doing asynchronous
        distributed training.
    """
    self._name_or_scope = name_or_scope
    self._reuse = reuse
    self._initializer = initializer
    self._regularizer = regularizer
    self._caching_device = caching_device
    self._partitioner = partitioner
    self._custom_getter = custom_getter
    self._old_name_scope = old_name_scope
    self._dtype = dtype
    self._use_resource = use_resource
    self._constraint = constraint
    self._var_store = _get_default_variable_store()
    self._var_scope_store = get_variable_scope_store()
    self._last_variable_scope_object = None
    if isinstance(self._name_or_scope, VariableScope):
      self._new_name = self._name_or_scope.name
      name_scope = self._name_or_scope._name_scope  # pylint: disable=protected-access
      # Handler for the case when we jump to a shared scope.  We create a new
      #   VariableScope (self._var_scope_object) that contains a copy of the
      #   provided shared scope, possibly with changed reuse and initializer, if
      #   the user requested this.
      variable_scope_object = VariableScope(
          self._name_or_scope.reuse if not self._reuse else self._reuse,
          name=self._new_name,
          initializer=self._name_or_scope.initializer,
          regularizer=self._name_or_scope.regularizer,
          caching_device=self._name_or_scope.caching_device,
          partitioner=self._name_or_scope.partitioner,
          dtype=self._name_or_scope.dtype,
          custom_getter=self._name_or_scope.custom_getter,
          name_scope=name_scope,
          use_resource=self._name_or_scope.use_resource,
          constraint=self._constraint)
      if self._initializer is not None:
        variable_scope_object.set_initializer(self._initializer)
      if self._regularizer is not None:
        variable_scope_object.set_regularizer(self._regularizer)
      if self._caching_device is not None:
        variable_scope_object.set_caching_device(self._caching_device)
      if self._partitioner is not None:
        variable_scope_object.set_partitioner(self._partitioner)
      if self._custom_getter is not None:
        variable_scope_object.set_custom_getter(
            _maybe_wrap_custom_getter(self._custom_getter,
                                      self._name_or_scope.custom_getter))
      if self._dtype is not None:
        variable_scope_object.set_dtype(self._dtype)
      if self._use_resource is not None:
        variable_scope_object.set_use_resource(self._use_resource)
      self._cached_variable_scope_object = variable_scope_object

  def __enter__(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring lists reuse-validation and type errors, but this method does not
    # raise them itself; only setter errors (for example from eager-mode restrictions) can occur.
    # Opens the scope. Saves the current scope. For a captured VariableScope: bumps its open
    # count, snapshots the sub-scope counts and uses the cached copy. For a string name: builds the
    # full name by appending to the current scope's name, inherits reuse (true if either is true),
    # chooses the name scope string, creates a child VariableScope inheriting the parent settings,
    # applies overrides, and bumps the open count. Installs the new scope as current and returns it.
    """Begins the scope block.

    Returns:
      A VariableScope.
    Raises:
      ValueError: when trying to reuse within a create scope, or create within
        a reuse scope, or if reuse is not `None` or `True`.
      TypeError: when the types of some arguments are not appropriate.
    """
    self._old = self._var_scope_store.current_scope
    if isinstance(self._name_or_scope, VariableScope):
      self._var_scope_store.open_variable_scope(self._new_name)
      self._old_subscopes = copy.copy(
          self._var_scope_store.variable_scopes_count)
      variable_scope_object = self._cached_variable_scope_object
    else:
      # Handler for the case when we just prolong current variable scope.
      #   VariableScope with name extended by the provided one, and inherited
      #   reuse and initializer (except if the user provided values to set).
      self._new_name = (
          self._old.name + "/" +
          self._name_or_scope if self._old.name else self._name_or_scope)
      self._reuse = (self._reuse or
                     self._old.reuse)  # Re-using is inherited by sub-scopes.
      if self._old_name_scope is None:
        name_scope = self._name_or_scope
      else:
        name_scope = self._old_name_scope
      variable_scope_object = VariableScope(
          self._reuse,
          name=self._new_name,
          initializer=self._old.initializer,
          regularizer=self._old.regularizer,
          caching_device=self._old.caching_device,
          partitioner=self._old.partitioner,
          dtype=self._old.dtype,
          use_resource=self._old.use_resource,
          custom_getter=self._old.custom_getter,
          name_scope=name_scope,
          constraint=self._constraint)
      if self._initializer is not None:
        variable_scope_object.set_initializer(self._initializer)
      if self._regularizer is not None:
        variable_scope_object.set_regularizer(self._regularizer)
      if self._caching_device is not None:
        variable_scope_object.set_caching_device(self._caching_device)
      if self._partitioner is not None:
        variable_scope_object.set_partitioner(self._partitioner)
      if self._custom_getter is not None:
        variable_scope_object.set_custom_getter(
            _maybe_wrap_custom_getter(self._custom_getter,
                                      self._old.custom_getter))
      if self._dtype is not None:
        variable_scope_object.set_dtype(self._dtype)
      if self._use_resource is not None:
        variable_scope_object.set_use_resource(self._use_resource)
      self._var_scope_store.open_variable_scope(self._new_name)
    self._var_scope_store.current_scope = variable_scope_object
    self._last_variable_scope_object = variable_scope_object
    return variable_scope_object

  def __exit__(self, type_arg, value_arg, traceback_arg):
    # [FUNCTION DOC]
    # Closes the scope. Raises a runtime error if the current scope is not the one this context
    # opened (improper nesting). For a captured scope restores the saved sub-scope counts;
    # for a named scope resets counts of sub-scopes under the new name. Restores the previous
    # current scope. Arguments describe any in-flight exception and are ignored; returns nothing
    # (exceptions are not suppressed).
    if (self._var_scope_store.current_scope is
        not self._last_variable_scope_object):
      raise RuntimeError("Improper nesting of variable_scope.")
    # If jumping out from a non-prolonged scope, restore counts.
    if isinstance(self._name_or_scope, VariableScope):
      self._var_scope_store.variable_scopes_count = self._old_subscopes
    else:
      self._var_scope_store.close_variable_subscopes(self._new_name)
    self._var_scope_store.current_scope = self._old


def _maybe_wrap_custom_getter(custom_getter, old_getter):
  # [FUNCTION DOC]
  # HELPER: chains custom getters of nested variable scopes.
  # Given a new custom getter and the enclosing scope's getter (possibly null), returns the new
  # getter unchanged if there is no old one; otherwise returns a wrapper that calls the new getter
  # with a getter that first goes through the old getter and then the true getter.
  """Wrap a call to a custom_getter to use the old_getter internally."""
  if old_getter is None:
    return custom_getter

  # The new custom_getter should call the old one
  def wrapped_custom_getter(getter, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: nested composite getter. Calls the new custom getter, giving it a getter that is the
    # old getter bound to the true getter, forwarding all extra arguments; returns its result.
    # Call:
    #  custom_getter(
    #    lambda: old_getter(true_getter, ...), *args, **kwargs)
    # which means custom_getter will call old_getter, which
    # will call the true_getter, perform any intermediate
    # processing, and return the results to the current
    # getter, which will also perform additional processing.
    return custom_getter(functools.partial(old_getter, getter), *args, **kwargs)

  return wrapped_custom_getter


def _get_unique_variable_scope(prefix):
  # [FUNCTION DOC]
  # HELPER: generates default variable-scope names for variable_scope with a default name.
  # Given a string prefix, returns the prefix if no scope with that full name (current scope name
  # plus prefix) has been opened, otherwise the prefix with an underscore and the smallest
  # positive integer making the full name unused. Reads the scope store counts; no mutation.
  """Get a name with the given prefix unique in the current variable scope."""
  var_scope_store = get_variable_scope_store()
  current_scope = get_variable_scope()
  name = current_scope.name + "/" + prefix if current_scope.name else prefix
  if var_scope_store.variable_scope_count(name) == 0:
    return prefix
  idx = 1
  while var_scope_store.variable_scope_count(name + ("_%d" % idx)) > 0:
    idx += 1
  return prefix + ("_%d" % idx)


# Named like a function for backwards compatibility with the
# @tf_contextlib.contextmanager version, which was switched to a class to avoid
# some object creation overhead.
@tf_export(v1=["variable_scope"])  # pylint: disable=invalid-name
class variable_scope:
  # [CLASS DOC]
  # Public v1 context manager (named like a function for historical reasons) that opens a
  # variable scope and, in graph mode, the matching default graph and name scope. It accepts a
  # scope name or an existing VariableScope plus optional default settings (initializer,
  # regularizer, caching device, partitioner, custom getter, reuse, dtype, use_resource,
  # constraint) and an option to skip the auxiliary name scope. It wraps an internal pure scope
  # object and caches it so re-entering the same instance is cheap. Entering yields the active
  # VariableScope; exiting restores the previous scope, name scope and graph context.
  """A context manager for defining ops that creates variables (layers).

  @compatibility(TF2)
  Although it is a legacy `compat.v1` api,
  `tf.compat.v1.variable_scope` is mostly compatible with eager
  execution and `tf.function` as long as you combine it with the
  `tf.compat.v1.keras.utils.track_tf1_style_variables` decorator (though
  it will behave as if reuse is always set to `AUTO_REUSE`.)

  See the
  [model migration guide](
      https://www.tensorflow.org/guide/migrate/model_mapping)
  for more info on
  migrating code that relies on `variable_scope`-based variable reuse.

  When you use it with eager execution enabled but without
  `tf.compat.v1.keras.utils.track_tf1_style_variables`,
  `tf.compat.v1.variable_scope` will still be able to prefix the names
  of variables created within the scope but it will not enable variable reuse
  or error-raising checks around variable reuse (`get_variable` calls within
  it would always create new variables).

  Once you have switched away from `get_variable`-based variable reuse
  mechanisms, to switch to TF2 APIs you can just use
  `tf.name_scope` to prefix variable names.
  @end_compatibility

  This context manager validates that the (optional) `values` are from the same
  graph, ensures that graph is the default graph, and pushes a name scope and a
  variable scope.

  If `name_or_scope` is not None, it is used as is. If `name_or_scope` is None,
  then `default_name` is used.  In that case, if the same name has been
  previously used in the same scope, it will be made unique by appending `_N`
  to it.

  Variable scope allows you to create new variables and to share already created
  ones while providing checks to not create or share by accident. For details,
  see the [Variable Scope How To](https://tensorflow.org/guide/variables), here
  we present only a few basic examples.

  The Variable Scope works as expected when the Eager Execution is Disabled.

  ```python
  tf.compat.v1.disable_eager_execution()
  ```

  Simple example of how to create a new variable:

  ```python
  with tf.compat.v1.variable_scope("foo"):
      with tf.compat.v1.variable_scope("bar"):
          v = tf.compat.v1.get_variable("v", [1])
          assert v.name == "foo/bar/v:0"
  ```

  Simple example of how to reenter a premade variable scope safely:

  ```python
  with tf.compat.v1.variable_scope("foo") as vs:
    pass

  # Re-enter the variable scope.
  with tf.compat.v1.variable_scope(vs,
                         auxiliary_name_scope=False) as vs1:
    # Restore the original name_scope.
    with tf.name_scope(vs1.original_name_scope):
        v = tf.compat.v1.get_variable("v", [1])
        assert v.name == "foo/v:0"
        c = tf.constant([1], name="c")
        assert c.name == "foo/c:0"
  ```

  Keep in mind that the counters for `default_name` are discarded once the
  parent scope is exited. Therefore when the code re-enters the scope (for
  instance by saving it), all nested default_name counters will be restarted.

  For instance:

  ```python
  with tf.compat.v1.variable_scope("foo") as vs:
    with tf.compat.v1.variable_scope(None, default_name="bar"):
      v = tf.compat.v1.get_variable("a", [1])
      assert v.name == "foo/bar/a:0", v.name
    with tf.compat.v1.variable_scope(None, default_name="bar"):
      v = tf.compat.v1.get_variable("b", [1])
      assert v.name == "foo/bar_1/b:0"

  with tf.compat.v1.variable_scope(vs):
    with tf.compat.v1.variable_scope(None, default_name="bar"):
      v = tf.compat.v1.get_variable("c", [1])
      assert v.name == "foo/bar/c:0"   # Uses bar instead of bar_2!
  ```

  Basic example of sharing a variable AUTO_REUSE:

  ```python
  def foo():
    with tf.compat.v1.variable_scope("foo", reuse=tf.compat.v1.AUTO_REUSE):
      v = tf.compat.v1.get_variable("v", [1])
    return v

  v1 = foo()  # Creates v.
  v2 = foo()  # Gets the same, existing v.
  assert v1 == v2
  ```

  Basic example of sharing a variable with reuse=True:

  ```python
  with tf.compat.v1.variable_scope("foo"):
      v = tf.compat.v1.get_variable("v", [1])
  with tf.compat.v1.variable_scope("foo", reuse=True):
      v1 = tf.compat.v1.get_variable("v", [1])
  assert v1 == v
  ```

  Sharing a variable by capturing a scope and setting reuse:

  ```python
  with tf.compat.v1.variable_scope("foo") as scope:
      v = tf.compat.v1.get_variable("v", [1])
      scope.reuse_variables()
      v1 = tf.compat.v1.get_variable("v", [1])
  assert v1 == v
  ```

  To prevent accidental sharing of variables, we raise an exception when getting
  an existing variable in a non-reusing scope.

  ```python
  with tf.compat.v1.variable_scope("foo"):
      v = tf.compat.v1.get_variable("v", [1])
      v1 = tf.compat.v1.get_variable("v", [1])
      #  Raises ValueError("... v already exists ...").
  ```

  Similarly, we raise an exception when trying to get a variable that does not
  exist in reuse mode.

  ```python
  with tf.compat.v1.variable_scope("foo", reuse=True):
      v = tf.compat.v1.get_variable("v", [1])
      #  Raises ValueError("... v does not exists ...").
  ```

  Note that the `reuse` flag is inherited: if we open a reusing scope, then all
  its sub-scopes become reusing as well.

  A note about name scoping: Setting `reuse` does not impact the naming of other
  ops such as mult. See related discussion on
  [github#6189](https://github.com/tensorflow/tensorflow/issues/6189)

  Note that up to and including version 1.0, it was allowed (though explicitly
  discouraged) to pass False to the reuse argument, yielding undocumented
  behaviour slightly different from None. Starting at 1.1.0 passing None and
  False as reuse has exactly the same effect.

  A note about using variable scopes in multi-threaded environment: Variable
  scopes are thread local, so one thread will not see another thread's current
  scope. Also, when using `default_name`, unique scopes names are also generated
  only on a per thread basis. If the same name was used within a different
  thread, that doesn't prevent a new thread from creating the same scope.
  However, the underlying variable store is shared across threads (within the
  same graph). As such, if another thread tries to create a new variable with
  the same name as a variable created by a previous thread, it will fail unless
  reuse is True.

  Further, each thread starts with an empty variable scope. So if you wish to
  preserve name prefixes from a scope from the main thread, you should capture
  the main thread's scope and re-enter it in each thread. For e.g.

  ```
  main_thread_scope = variable_scope.get_variable_scope()

  # Thread's target function:
  def thread_target_fn(captured_scope):
    with variable_scope.variable_scope(captured_scope):
      # .... regular code for this thread


  thread = threading.Thread(target=thread_target_fn, args=(main_thread_scope,))
  ```
  """

  def __init__(self,
               name_or_scope,
               default_name=None,
               values=None,
               initializer=None,
               regularizer=None,
               caching_device=None,
               partitioner=None,
               custom_getter=None,
               reuse=None,
               dtype=None,
               use_resource=None,
               constraint=None,
               auxiliary_name_scope=True):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says the call returns a scope and raises a value error for
    # reusing within a create scope; the constructor returns nothing, and the actual reuse error
    # message ("must be True or False or None") omits the allowed auto-reuse value.
    # Stores all options. Raises a type error if both name_or_scope and default_name are null, or
    # if auxiliary_name_scope is not a boolean. A reuse of false is converted to null (inherit);
    # any reuse other than true, null or auto-reuse raises a value error. A null values list
    # becomes empty. In graph mode (eager disabled) determines the graph from the values via the
    # graph-from-inputs helper (which may raise if values come from different graphs).
    """Initialize the context manager.

    Args:
      name_or_scope: `string` or `VariableScope`: the scope to open.
      default_name: The default name to use if the `name_or_scope` argument is
        `None`, this name will be uniquified. If name_or_scope is provided it
        won't be used and therefore it is not required and can be None.
      values: The list of `Tensor` arguments that are passed to the op function.
      initializer: default initializer for variables within this scope.
      regularizer: default regularizer for variables within this scope.
      caching_device: default caching device for variables within this scope.
      partitioner: default partitioner for variables within this scope.
      custom_getter: default custom getter for variables within this scope.
      reuse: `True`, None, or tf.compat.v1.AUTO_REUSE; if `True`, we go into
        reuse mode for this scope as well as all sub-scopes; if
        tf.compat.v1.AUTO_REUSE, we create variables if they do not exist, and
        return them otherwise; if None, we inherit the parent scope's reuse
        flag. When eager execution is enabled, new variables are always created
        unless an EagerVariableStore or template is currently active.
      dtype: type of variables created in this scope (defaults to the type in
        the passed scope, or inherited from parent scope).
      use_resource: If False, all variables will be regular Variables. If True,
        experimental ResourceVariables with well-defined semantics will be used
        instead. Defaults to False (will later change to True). When eager
        execution is enabled this argument is always forced to be True.
      constraint: An optional projection function to be applied to the variable
        after being updated by an `Optimizer` (e.g. used to implement norm
        constraints or value constraints for layer weights). The function must
        take as input the unprojected Tensor representing the value of the
        variable and return the Tensor for the projected value (which must have
        the same shape). Constraints are not safe to use when doing asynchronous
        distributed training.
      auxiliary_name_scope: If `True`, we create an auxiliary name scope with
        the scope. If `False`, we don't create it. Note that the argument is not
        inherited, and it only takes effect for once when creating. You should
        only use it for re-entering a premade variable scope.

    Returns:
      A scope that can be captured and reused.

    Raises:
      ValueError: when trying to reuse within a create scope, or create within
        a reuse scope.
      TypeError: when the types of some arguments are not appropriate.
    """
    self._name_or_scope = name_or_scope
    self._default_name = default_name
    self._values = values
    self._initializer = initializer
    self._regularizer = regularizer
    self._caching_device = caching_device
    self._partitioner = partitioner
    self._custom_getter = custom_getter
    self._reuse = reuse
    self._dtype = dtype
    self._use_resource = use_resource
    self._constraint = constraint
    if self._default_name is None and self._name_or_scope is None:
      raise TypeError("If default_name is None then name_or_scope is required")
    if self._reuse is False:
      # We don't allow non-inheriting scopes, False = None here.
      self._reuse = None
    if not (self._reuse is True
            or self._reuse is None
            or self._reuse is AUTO_REUSE):
      raise ValueError("The reuse parameter must be True or False or None.")
    if self._values is None:
      self._values = []
    self._in_graph_mode = not context.executing_eagerly()
    if self._in_graph_mode:
      self._graph = ops._get_graph_from_inputs(self._values)  # pylint: disable=protected-access
    self._cached_pure_variable_scope = None
    self._current_name_scope = None
    if not isinstance(auxiliary_name_scope, bool):
      raise TypeError("The auxiliary_name_scope must be `True` or `False`, "
                      "while get {}".format(auxiliary_name_scope))
    self._auxiliary_name_scope = auxiliary_name_scope

  def __enter__(self):
    # [FUNCTION DOC]
    # Opens the scope. Records whether the default graph is building a function. In graph mode
    # and not building a function, enters the graph chosen at construction as default. If a
    # pure scope was cached from an earlier entry, re-enters the saved name scope (if any) and the
    # cached pure scope and returns the variable scope. Otherwise builds the scope through the
    # uncached path. If that fails the graph context is exited and the error re-raised.
    # Returns the active VariableScope.
    # If the default graph is building a function, then we should not replace it
    # with the cached graph.
    if ops.get_default_graph().building_function:
      self._building_function = True
    else:
      self._building_function = False
    if self._in_graph_mode and not self._building_function:
      self._graph_context_manager = self._graph.as_default()
      self._graph_context_manager.__enter__()
    if self._cached_pure_variable_scope is not None:
      # Fast path for re-entering variable_scopes. We've held on to the pure
      # variable scope from a previous successful __enter__, so we avoid some
      # overhead by re-using that object.
      if self._current_name_scope is not None:
        self._current_name_scope.__enter__()
      return self._cached_pure_variable_scope.__enter__()

    try:
      return self._enter_scope_uncached()
    except:
      if (self._in_graph_mode and not self._building_function and
          self._graph_context_manager is not None):
        self._graph_context_manager.__exit__(*sys.exc_info())
      raise

  def _enter_scope_uncached(self):
    # [FUNCTION DOC]
    # HELPER: first-time entry path of variable_scope.
    # Chooses the name scope to use: if the auxiliary name scope is disabled, re-enters the
    # current name scope; otherwise creates one lazily. If a name or scope was given: raises a
    # type error unless it is a string or VariableScope; opens the name scope (named by the string
    # or last path component of the scope's name) unless entering the root scope, then creates and
    # enters an internal pure variable scope (re-entering uses the scope's original name scope).
    # If no name was given: raises a value error when reuse is true, opens a name scope named by
    # the default name, makes the default name unique among sibling scopes, and enters a pure scope
    # with that name. On any failure the partly entered scopes are exited and the error re-raised.
    # Caches the entered objects only after success and returns the VariableScope.
    """Enters the context manager when there is no cached scope yet.

    Returns:
      The entered variable scope.

    Raises:
      TypeError: A wrong type is passed as `scope` at __init__().
      ValueError: `reuse` is incorrectly set at __init__().
    """
    if self._auxiliary_name_scope:
      # Create a new name scope later
      current_name_scope = None
    else:
      # Reenter the current name scope
      name_scope = ops.get_name_scope()
      if name_scope:
        # Hack to reenter
        name_scope += "/"
        current_name_scope = ops.name_scope(name_scope, skip_on_eager=False)
      else:
        # Root scope
        current_name_scope = ops.name_scope(name_scope, skip_on_eager=False)

    # IMPORTANT: Only assign to self._cached_pure_variable_scope and
    # self._current_name_scope after successful __enter__() calls.
    if self._name_or_scope is not None:
      if not isinstance(self._name_or_scope, (VariableScope, str)):
        raise TypeError("VariableScope: name_or_scope must be a string or "
                        "VariableScope.")
      if isinstance(self._name_or_scope, str):
        name_scope = self._name_or_scope
      else:
        name_scope = self._name_or_scope.name.split("/")[-1]
      if name_scope or current_name_scope:
        current_name_scope = current_name_scope or ops.name_scope(
            name_scope, skip_on_eager=False)
        try:
          current_name_scope_name = current_name_scope.__enter__()
        except:
          current_name_scope.__exit__(*sys.exc_info())
          raise
        self._current_name_scope = current_name_scope
        if isinstance(self._name_or_scope, str):
          old_name_scope = current_name_scope_name
        else:
          old_name_scope = self._name_or_scope.original_name_scope
        pure_variable_scope = _pure_variable_scope(
            self._name_or_scope,
            reuse=self._reuse,
            initializer=self._initializer,
            regularizer=self._regularizer,
            caching_device=self._caching_device,
            partitioner=self._partitioner,
            custom_getter=self._custom_getter,
            old_name_scope=old_name_scope,
            dtype=self._dtype,
            use_resource=self._use_resource,
            constraint=self._constraint)
        try:
          entered_pure_variable_scope = pure_variable_scope.__enter__()
        except:
          pure_variable_scope.__exit__(*sys.exc_info())
          raise
        self._cached_pure_variable_scope = pure_variable_scope
        return entered_pure_variable_scope
      else:
        self._current_name_scope = None
        # This can only happen if someone is entering the root variable scope.
        pure_variable_scope = _pure_variable_scope(
            self._name_or_scope,
            reuse=self._reuse,
            initializer=self._initializer,
            regularizer=self._regularizer,
            caching_device=self._caching_device,
            partitioner=self._partitioner,
            custom_getter=self._custom_getter,
            dtype=self._dtype,
            use_resource=self._use_resource,
            constraint=self._constraint)
        try:
          entered_pure_variable_scope = pure_variable_scope.__enter__()
        except:
          pure_variable_scope.__exit__(*sys.exc_info())
          raise
        self._cached_pure_variable_scope = pure_variable_scope
        return entered_pure_variable_scope

    else:  # Here name_or_scope is None. Using default name, but made unique.
      if self._reuse:
        raise ValueError("reuse=True cannot be used without a name_or_scope")
      current_name_scope = current_name_scope or ops.name_scope(
          self._default_name, skip_on_eager=False)
      try:
        current_name_scope_name = current_name_scope.__enter__()
      except:
        current_name_scope.__exit__(*sys.exc_info())
        raise
      self._current_name_scope = current_name_scope
      unique_default_name = _get_unique_variable_scope(self._default_name)
      pure_variable_scope = _pure_variable_scope(
          unique_default_name,
          initializer=self._initializer,
          regularizer=self._regularizer,
          caching_device=self._caching_device,
          partitioner=self._partitioner,
          custom_getter=self._custom_getter,
          old_name_scope=current_name_scope_name,
          dtype=self._dtype,
          use_resource=self._use_resource,
          constraint=self._constraint)
      try:
        entered_pure_variable_scope = pure_variable_scope.__enter__()
      except:
        pure_variable_scope.__exit__(*sys.exc_info())
        raise
      self._cached_pure_variable_scope = pure_variable_scope
      return entered_pure_variable_scope

  def __exit__(self, type_arg, value_arg, traceback_arg):
    # [FUNCTION DOC]
    # Closes the scope in reverse order of entry: the internal pure scope first, then the name
    # scope (if one was opened), then the graph context (graph mode, not building a function).
    # Each later step runs even if an earlier one raises. The in-flight exception details are
    # forwarded to each; returns nothing.
    try:
      self._cached_pure_variable_scope.__exit__(type_arg, value_arg,
                                                traceback_arg)
    finally:
      try:
        if self._current_name_scope:
          self._current_name_scope.__exit__(type_arg, value_arg,
                                            traceback_arg)
      finally:
        if self._in_graph_mode and not self._building_function:
          self._graph_context_manager.__exit__(type_arg, value_arg,
                                               traceback_arg)


# pylint: disable=g-doc-return-or-yield
@tf_export(v1=["variable_op_scope"])
@tf_contextlib.contextmanager
def variable_op_scope(values,
                      name_or_scope,
                      default_name=None,
                      initializer=None,
                      regularizer=None,
                      caching_device=None,
                      partitioner=None,
                      custom_getter=None,
                      reuse=None,
                      dtype=None,
                      use_resource=None,
                      constraint=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: deprecated; it is marked as deprecated only by a logged warning, and the argument
  # order (values first) differs from variable_scope.
  # Context manager wrapper around variable_scope with the values list given first. Logs a
  # deprecation warning, then opens variable_scope with the same options and yields the
  # resulting VariableScope. Exported as a v1 API.
  """Deprecated: context manager for defining an op that creates variables."""
  logging.warn("tf.variable_op_scope(values, name, default_name) is deprecated,"
               " use tf.variable_scope(name, default_name, values)")
  with variable_scope(
      name_or_scope,
      default_name=default_name,
      values=values,
      initializer=initializer,
      regularizer=regularizer,
      caching_device=caching_device,
      partitioner=partitioner,
      custom_getter=custom_getter,
      reuse=reuse,
      dtype=dtype,
      use_resource=use_resource,
      constraint=constraint) as scope:
    yield scope


def _call_partitioner(partitioner, shape, dtype):
  # [FUNCTION DOC]
  # HELPER: validates and invokes a user partitioner for partitioned variable creation.
  # KNOWN ISSUE: the docstring says the shape must have at least two dimensions and that exactly
  # one element is greater than one; the code only requires rank at least one and allows no
  # partitioned axis (all ones).
  # Inputs: partitioner callable (called with shape and dtype), a tensor shape and a dtype. Raises
  # a value error if the shape is not fully defined, has rank below one, the partitioner result is
  # not a sequence, its length differs from the rank, any entry is below one, or more than one
  # entry exceeds one. Returns the validated list of partition counts per axis.
  """Call partitioner validating its inputs/output.

  Args:
    partitioner: a function mapping `Tensor` shape and dtype to a list of
      partitions.
    shape: shape of the `Tensor` to partition, must have at least two
      dimensions.
    dtype: dtype of the elements in the `Tensor`.

  Returns:
    A list with elements >=1 and exactly one >1. The index of that
    element corresponds to the partitioning axis.
  """
  if not shape.is_fully_defined():
    raise ValueError("Shape of a new partitioned variable must be "
                     "fully defined, but instead was %s." % (shape,))
  if shape.ndims < 1:
    raise ValueError("A partitioned Variable must have rank at least 1, "
                     "shape: %s" % shape)

  slicing = partitioner(shape=shape, dtype=dtype)
  if not isinstance(slicing, collections_abc.Sequence):
    raise ValueError("Partitioner must return a sequence, but saw: %s" %
                     slicing)
  if len(slicing) != shape.ndims:
    raise ValueError(
        "Partitioner returned a partition list that does not match the "
        "Variable's rank: %s vs. %s" % (slicing, shape))
  if any(p < 1 for p in slicing):
    raise ValueError("Partitioner returned zero partitions for some axes: %s" %
                     slicing)
  if sum(p > 1 for p in slicing) > 1:
    raise ValueError("Can only slice a variable along one dimension: "
                     "shape: %s, partitioning: %s" % (shape, slicing))
  return slicing


# TODO(slebedev): could be inlined, but
# `_VariableStore._get_partitioned_variable` is too complex even
# without this logic.
def _get_slice_dim_and_num_slices(slicing):
  # [FUNCTION DOC]
  # HELPER: interprets validated partitioner output for the variable store.
  # Takes the list of partition counts per axis. Returns a pair: the index of the first axis with
  # more than one partition and its count; if none, returns axis zero with one slice.
  """Get slicing dimension and number of slices from the partitioner output."""
  for slice_dim, num_slices in enumerate(slicing):
    if num_slices > 1:
      break
  else:
    # Degenerate case: no partitioning applied.
    slice_dim = 0
    num_slices = 1
  return slice_dim, num_slices


def _iter_slices(full_shape, num_slices, slice_dim):
  # [FUNCTION DOC]
  # HELPER: generator that splits a shape into shards for partitioned variables.
  # Inputs: full_shape (list of ints), number of slices, and the dimension to split. Yields
  # pairs of (offset list, shard shape list), one per slice, in order. The first slices get one
  # extra element when the dimension does not divide evenly. Offsets advance cumulatively along
  # the split dimension. Does not modify the input.
  """Slices a given a shape along the specified dimension."""
  num_slices_with_excess = full_shape[slice_dim] % num_slices
  offset = [0] * len(full_shape)
  min_slice_len = full_shape[slice_dim] // num_slices
  for i in range(num_slices):
    shape = full_shape[:]
    shape[slice_dim] = min_slice_len + bool(i < num_slices_with_excess)
    yield offset[:], shape
    offset[slice_dim] += shape[slice_dim]


def _make_getter(captured_getter, captured_previous):
  # [FUNCTION DOC]
  # HELPER: closure factory that binds the current values of a getter and its predecessor so they
  # are not affected by later loop variable changes. Returns a callable taking keyword arguments
  # that calls the captured getter with the captured previous getter and those arguments.
  """Gets around capturing loop variables in python being broken."""
  return lambda **kwargs: captured_getter(captured_previous, **kwargs)


_variable_v1 = None


def set_variable_v1(variable_v1):
  # [FUNCTION DOC]
  # HELPER: late-binding hook that breaks a circular import. Stores the given VariableV1 class in
  # the module-level reference used when creating variables. Mutates module global state; returns
  # nothing. Must be called before any variable is created through the store, otherwise the
  # stored reference is null and creation fails.
  """Sets a reference to variable_v1.VariableV1."""
  global _variable_v1
  _variable_v1 = variable_v1


@tf_export(v1=["variable_creator_scope"])
@tf_contextlib.contextmanager
def variable_creator_scope_v1(variable_creator):
  # [FUNCTION DOC]
  # Context manager (v1 API) that registers a variable creator function on the default graph for
  # the duration of the with block. variable_creator takes the next creator plus keyword
  # arguments describing the variable and returns a variable. Yields nothing. Side effect: the
  # creator is pushed onto the graph's creator stack and removed on exit. Documentation differs
  # from the v2 version only in the description of trainable.
  """Scope which defines a variable creation function to be used by variable().

  variable_creator is expected to be a function with the following signature:

  ```
    def variable_creator(next_creator, **kwargs)
  ```

  The creator is supposed to eventually call the next_creator to create a
  variable if it does want to create a variable and not call Variable or
  ResourceVariable directly. This helps make creators composable. A creator may
  choose to create multiple variables, return already existing variables, or
  simply register that a variable was created and defer to the next creators in
  line. Creators can also modify the keyword arguments seen by the next
  creators.

  Custom getters in the variable scope will eventually resolve down to these
  custom creators when they do create variables.

  The valid keyword arguments in kwds are:

   * initial_value: A `Tensor`, or Python object convertible to a `Tensor`,
        which is the initial value for the Variable. The initial value must have
        a shape specified unless `validate_shape` is set to False. Can also be a
        callable with no argument that returns the initial value when called. In
        that case, `dtype` must be specified. (Note that initializer functions
        from init_ops.py must first be bound to a shape before being used here.)
   * trainable: If `True`, the default, also adds the variable to the graph
        collection `GraphKeys.TRAINABLE_VARIABLES`. This collection is used as
        the default list of variables to use by the `Optimizer` classes.
        `trainable` defaults to `True`, unless `synchronization` is
        set to `ON_READ`, in which case it defaults to `False`.
   * collections: List of graph collections keys. The new variable is added to
        these collections. Defaults to `[GraphKeys.GLOBAL_VARIABLES]`.
   * validate_shape: If `False`, allows the variable to be initialized with a
        value of unknown shape. If `True`, the default, the shape of
        `initial_value` must be known.
   * caching_device: Optional device string describing where the Variable
        should be cached for reading.  Defaults to the Variable's device.
        If not `None`, caches on another device.  Typical use is to cache
        on the device where the Ops using the Variable reside, to deduplicate
        copying through `Switch` and other conditional statements.
   * name: Optional name for the variable. Defaults to `'Variable'` and gets
        uniquified automatically.
   * dtype: If set, initial_value will be converted to the given type.
        If `None`, either the datatype will be kept (if `initial_value` is
        a Tensor), or `convert_to_tensor` will decide.
   * constraint: A constraint function to be applied to the variable after
        updates by some algorithms.
   * use_resource: if True, a ResourceVariable is always created.
   * synchronization: Indicates when a distributed a variable will be
        aggregated. Accepted values are constants defined in the class
        `tf.VariableSynchronization`. By default the synchronization is set to
        `AUTO` and the current `DistributionStrategy` chooses
        when to synchronize.
   * aggregation: Indicates how a distributed variable will be aggregated.
        Accepted values are constants defined in the class
        `tf.VariableAggregation`.

  This set may grow over time, so it's important the signature of creators is as
  mentioned above.

  Args:
    variable_creator: the passed creator

  Yields:
    A scope in which the creator is active
  """
  with ops.get_default_graph()._variable_creator_scope(variable_creator):  # pylint: disable=protected-access
    yield


# Note: only the docstrings differ between this and v1.
@tf_export("variable_creator_scope", v1=[])
@tf_contextlib.contextmanager
def variable_creator_scope(variable_creator):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring's list of keyword arguments is missing the bullet marker before
  # dtype and omits use_resource, which the v1 variant lists.
  # Context manager (v2 API) that registers a variable creator function on the default graph for
  # the duration of the with block. variable_creator takes the next creator plus keyword
  # arguments and returns a variable. Yields nothing. Side effect: pushes then removes the creator
  # on the graph's creator stack.
  """Scope which defines a variable creation function to be used by variable().

  variable_creator is expected to be a function with the following signature:

  ```
    def variable_creator(next_creator, **kwargs)
  ```

  The creator is supposed to eventually call the next_creator to create a
  variable if it does want to create a variable and not call Variable or
  ResourceVariable directly. This helps make creators composable. A creator may
  choose to create multiple variables, return already existing variables, or
  simply register that a variable was created and defer to the next creators in
  line. Creators can also modify the keyword arguments seen by the next
  creators.

  Custom getters in the variable scope will eventually resolve down to these
  custom creators when they do create variables.

  The valid keyword arguments in kwds are:

   * initial_value: A `Tensor`, or Python object convertible to a `Tensor`,
        which is the initial value for the Variable. The initial value must have
        a shape specified unless `validate_shape` is set to False. Can also be a
        callable with no argument that returns the initial value when called. In
        that case, `dtype` must be specified. (Note that initializer functions
        from init_ops.py must first be bound to a shape before being used here.)
   * trainable: If `True`, the default, GradientTapes automatically watch
        uses of this Variable.
   * validate_shape: If `False`, allows the variable to be initialized with a
        value of unknown shape. If `True`, the default, the shape of
        `initial_value` must be known.
   * caching_device: Optional device string describing where the Variable
        should be cached for reading.  Defaults to the Variable's device.
        If not `None`, caches on another device.  Typical use is to cache
        on the device where the Ops using the Variable reside, to deduplicate
        copying through `Switch` and other conditional statements.
   * name: Optional name for the variable. Defaults to `'Variable'` and gets
        uniquified automatically.
      dtype: If set, initial_value will be converted to the given type.
        If `None`, either the datatype will be kept (if `initial_value` is
        a Tensor), or `convert_to_tensor` will decide.
   * constraint: A constraint function to be applied to the variable after
        updates by some algorithms.
   * synchronization: Indicates when a distributed a variable will be
        aggregated. Accepted values are constants defined in the class
        `tf.VariableSynchronization`. By default the synchronization is set to
        `AUTO` and the current `DistributionStrategy` chooses
        when to synchronize.
   * aggregation: Indicates how a distributed variable will be aggregated.
        Accepted values are constants defined in the class
        `tf.VariableAggregation`.

  This set may grow over time, so it's important the signature of creators is as
  mentioned above.

  Args:
    variable_creator: the passed creator

  Yields:
    A scope in which the creator is active
  """
  with ops.get_default_graph()._variable_creator_scope(variable_creator):  # pylint: disable=protected-access
    yield
