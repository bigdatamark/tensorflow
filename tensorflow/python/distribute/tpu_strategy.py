# [MODULE DOC]
# TPU distribution strategies for TensorFlow. Defines the public TPU strategy classes: the
# version 2 strategy (exported as the distribute TPUStrategy), the deprecated experimental
# TPUStrategy, and the version 1 strategy, plus the shared extended implementation class that
# does the real work (device discovery from the TPU cluster resolver, variable creation as
# mirrored TPU variables, input distribution, reductions, updates, and running a user function
# replicated across TPU cores through the TPU replicate machinery). Also contains a replica
# context specialised for TPUs (with an all-gather built on cross-replica sum or all-to-all)
# and a few module-level helpers (run-function validation, partial application of distributed
# variables, batch variable initialization switch). Import time effects: registers the public
# API symbols via the export decorators and defines module constants; nothing is run on import.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""TPU Strategy."""

import atexit
import collections
import contextlib
import copy
import functools
import weakref

from absl import logging
import numpy as np

from tensorflow.python.autograph.core import ag_ctx as autograph_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.compiler.xla.experimental import xla_sharding
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import input_util
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import tpu_replicated_variable
from tensorflow.python.distribute import tpu_util
from tensorflow.python.distribute import tpu_values
from tensorflow.python.distribute import values
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver as tpu_cluster_resolver_lib
from tensorflow.python.distribute.v1 import input_lib as input_lib_v1
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import device_spec
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.saved_model import save_context
from tensorflow.python.tpu import device_assignment as device_assignment_lib  # pylint: disable=unused-import
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu import tpu_hardware_feature
from tensorflow.python.tpu import training_loop
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import tf_export
from tensorflow.python.util import tf_inspect

_XLA_OP_BY_OP_INPUTS_LIMIT = 200

_EXPERIMENTAL_TPU_BATCH_VARIABLE_INITIALIZATION = False


def enable_batch_variable_initialization():
  # [FUNCTION DOC]
  # HELPER: query used by the variable creation logic of the TPU extended strategy.
  # Returns true only when the module-level experimental batch-initialization flag is on, the
  # code runs eagerly, and no saved-model save context is active. Takes no arguments.
  # The flag is False by default, so this normally returns false.
  """Whether to batch variable initialization in tf.function."""
  return (
      _EXPERIMENTAL_TPU_BATCH_VARIABLE_INITIALIZATION
      and context.executing_eagerly()
      and not save_context.in_save_context()
  )


@contextlib.contextmanager
def maybe_init_scope():
  # [FUNCTION DOC]
  # HELPER: context manager used when evaluating variable initial values.
  # Takes no arguments. If execution is eager outside of functions it simply yields; otherwise
  # it enters the graph initialization scope (lifting ops out of the current function graph)
  # for the duration of the with block. Yields nothing.
  if ops.executing_eagerly_outside_functions():
    yield
  else:
    with ops.init_scope():
      yield


def validate_run_function(fn):
  # [FUNCTION DOC]
  # HELPER: argument check for the strategy run methods.
  # Input: fn, the user function given to run. When executing eagerly, fn must be a traced
  # function object, a concrete function, or a callable object whose call method is a traced
  # function; otherwise a not-implemented error is raised explaining that pure eager execution
  # is unsupported. When not eager (already inside a function) anything is accepted.
  # Returns nothing.
  """Validate the function passed into strategy.run."""

  # We allow three types of functions/objects passed into TPUStrategy
  # run in eager mode:
  #   1. a user annotated tf.function
  #   2. a ConcreteFunction, this is mostly what you get from loading a saved
  #      model.
  #   3. a callable object and the `__call__` method itself is a tf.function.
  #
  # Otherwise we return an error, because we don't support eagerly running
  # run in TPUStrategy.

  if (context.executing_eagerly()
      and not isinstance(fn, def_function.Function)
      and not isinstance(fn, function.ConcreteFunction)
      and not (
          callable(fn) and isinstance(fn.__call__, def_function.Function))
      ):
    raise NotImplementedError(
        "TPUStrategy.run(fn, ...) does not support pure eager "
        "execution. please make sure the function passed into "
        "`strategy.run` is a `tf.function` or "
        "`strategy.run` is called inside a `tf.function` if "
        "eager behavior is enabled.")


def _maybe_partial_apply_variables(fn, args, kwargs):
  # [FUNCTION DOC]
  # HELPER: glue used by the run methods so distributed variables are not converted to tensors.
  # Inputs: fn (the function to run), args (positional arguments), kwargs (keyword arguments).
  # Finds which arguments are distributed variables (or nests starting with one). Variables
  # passed by keyword are partially applied to fn; variables passed positionally are mapped to
  # parameter names by index and moved into the keyword set (a leading parameter named self is
  # skipped). Returns a triple of the possibly wrapped function, the remaining positional
  # arguments and the remaining keyword arguments; if no variables are found, returns the inputs
  # unchanged. Raises a value error when a variable is passed into a variadic positional
  # parameter, when variables are mixed with variadic positionals, when too many positional
  # arguments are given, or when variables are mixed with positional-only parameters.
  """Inspects arguments to partially apply any DistributedVariable.

  This avoids an automatic cast of the current variable value to tensor.

  Note that a variable may be captured implicitly with Python scope instead of
  passing it to run(), but supporting run() keeps behavior consistent
  with MirroredStrategy.

  Since positional arguments must be applied from left to right, this function
  does some tricky function inspection to move variable positional arguments
  into kwargs. As a result of this, we can't support passing Variables as *args,
  nor as args to functions which combine both explicit positional arguments and
  *args.

  Args:
    fn: The function to run, as passed to run().
    args: Positional arguments to fn, as passed to run().
    kwargs: Keyword arguments to fn, as passed to run().

  Returns:
    A tuple of the function (possibly wrapped), args, kwargs (both
    possibly filtered, with members of args possibly moved to kwargs).
    If no variables are found, this function is a noop.

  Raises:
    ValueError: If the function signature makes unsupported use of *args, or if
      too many arguments are passed.
  """

  def is_distributed_var(x):
    # [FUNCTION DOC]
    # HELPER: predicate for the enclosing function.
    # Flattens x as a nest and returns true if it is non-empty and its first element is a
    # distributed variable; returns a false or empty value otherwise.
    flat = nest.flatten(x)
    return flat and isinstance(flat[0], values.DistributedVariable)

  # We will split kwargs into two dicts, one of which will be applied now.
  var_kwargs = {}
  nonvar_kwargs = {}

  if kwargs:
    var_kwargs = {k: v for k, v in kwargs.items() if is_distributed_var(v)}
  if var_kwargs:
    nonvar_kwargs = {
        k: v for k, v in kwargs.items() if not is_distributed_var(v)
    }

  # Dump the argument names of `fn` to a list. This will include both positional
  # and keyword arguments, but since positional arguments come first we can
  # look up names of positional arguments by index.
  positional_args = []
  index_of_star_args = None
  for i, p in enumerate(tf_inspect.signature(fn).parameters.values()):
    # Class methods define "self" as first argument, but we don't pass "self".
    # Note that this is a heuristic, as a method can name its first argument
    # something else, and a function can define a first argument "self" as well.
    # In both of these cases, using a Variable will fail with an unfortunate
    # error about the number of arguments.
    # inspect.is_method() seems not to work here, possibly due to the use of
    # tf.function().
    if i == 0 and p.name == "self":
      continue

    if p.kind == tf_inspect.Parameter.POSITIONAL_OR_KEYWORD:
      positional_args.append(p.name)

    elif p.kind == tf_inspect.Parameter.VAR_POSITIONAL:
      # We'll raise an error later if a variable is passed to *args, since we
      # can neither pass it by name nor partially apply it. This case only
      # happens once at most.
      index_of_star_args = i

    elif p.kind == tf_inspect.Parameter.POSITIONAL_ONLY:
      # This is a rare Python feature, indicating a / in the arg list.
      if var_kwargs or any(is_distributed_var(a) for a in args):
        raise ValueError(
            "Mixing Variables and positional-only parameters not supported by "
            f"TPUStrategy. Received {len(var_kwargs)} DistributedVariables in "
            f"**kwargs and {sum(is_distributed_var(a) for a in args)} in *args,"
            " expected zero for both."
        )
      return fn, args, kwargs

  star_args = []
  have_seen_var_arg = False

  for i, a in enumerate(args):
    if is_distributed_var(a):
      if index_of_star_args is not None and i >= index_of_star_args:
        raise ValueError(
            "TPUStrategy.run() cannot handle Variables passed to *args. "
            "Either name the function argument, or capture the Variable "
            "implicitly.")
      if len(positional_args) <= i:
        raise ValueError(
            "Too many positional arguments passed to call to TPUStrategy.run()."
        )
      var_kwargs[positional_args[i]] = a
      have_seen_var_arg = True
    else:
      if index_of_star_args is not None and i >= index_of_star_args:
        if have_seen_var_arg:
          raise ValueError(
              "TPUStrategy.run() cannot handle both Variables and a mix of "
              "positional args and *args. Either remove the *args, or capture "
              "the Variable implicitly.")
        else:
          star_args.append(a)
          continue

      if len(positional_args) <= i:
        raise ValueError(
            "Too many positional arguments passed to call to TPUStrategy.run()."
        )
      nonvar_kwargs[positional_args[i]] = a

  if var_kwargs:
    return functools.partial(fn, **var_kwargs), star_args, nonvar_kwargs
  return fn, args, kwargs


@tf_export.tf_export("distribute.TPUStrategy", v1=[])
class TPUStrategyV2(distribute_lib.Strategy):
  # [CLASS DOC]
  # Public synchronous training strategy for TPUs and TPU pods (the distribute TPUStrategy).
  # Created from an optional TPU cluster resolver, an optional device assignment (for model
  # parallelism) and an SPMD partitioning flag. It delegates all work to a TPU extended object
  # built in the constructor, and adds the run entry point and logical-device annotation helpers
  # (assign, split, replicate tensors over logical devices). Does not support pure eager run.
  """Synchronous training on TPUs and TPU Pods.

  To construct a TPUStrategy object, you need to run the
  initialization code as below:

  >>> resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
  >>> tf.config.experimental_connect_to_cluster(resolver)
  >>> tf.tpu.experimental.initialize_tpu_system(resolver)
  >>> strategy = tf.distribute.TPUStrategy(resolver)

  While using distribution strategies, the variables created within the
  strategy's scope will be replicated across all the replicas and can be kept in
  sync using all-reduce algorithms.

  To run TF2 programs on TPUs, you can either use `.compile` and
  `.fit` APIs in `tf.keras` with TPUStrategy, or write your own customized
  training loop by calling `strategy.run` directly. Note that
  TPUStrategy doesn't support pure eager execution, so please make sure the
  function passed into `strategy.run` is a `tf.function` or
  `strategy.run` is called inside a `tf.function` if eager
  behavior is enabled. See more details in https://www.tensorflow.org/guide/tpu.

  `distribute_datasets_from_function` and
  `experimental_distribute_dataset` APIs can be used to distribute the dataset
  across the TPU workers when writing your own training loop. If you are using
  `fit` and `compile` methods available in `tf.keras.Model`, then Keras will
  handle the distribution for you.

  An example of writing customized training loop on TPUs:

  >>> with strategy.scope():
  ...   model = tf.keras.Sequential([
  ...     tf.keras.layers.Dense(2, input_shape=(5,)),
  ...   ])
  ...   optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

  >>> def dataset_fn(ctx):
  ...   x = np.random.random((2, 5)).astype(np.float32)
  ...   y = np.random.randint(2, size=(2, 1))
  ...   dataset = tf.data.Dataset.from_tensor_slices((x, y))
  ...   return dataset.repeat().batch(1, drop_remainder=True)
  >>> dist_dataset = strategy.distribute_datasets_from_function(
  ...     dataset_fn)
  >>> iterator = iter(dist_dataset)

  >>> @tf.function()
  ... def train_step(iterator):
  ...
  ...   def step_fn(inputs):
  ...     features, labels = inputs
  ...     with tf.GradientTape() as tape:
  ...       logits = model(features, training=True)
  ...       loss = tf.keras.losses.sparse_categorical_crossentropy(
  ...           labels, logits)
  ...
  ...     grads = tape.gradient(loss, model.trainable_variables)
  ...     optimizer.apply_gradients(zip(grads, model.trainable_variables))
  ...
  ...   strategy.run(step_fn, args=(next(iterator),))

  >>> train_step(iterator)

  For the advanced use cases like model parallelism, you can set
  `experimental_device_assignment` argument when creating TPUStrategy to specify
  number of replicas and number of logical devices. Below is an example to
  initialize TPU system with 2 logical devices and 1 replica.

  >>> resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
  >>> tf.config.experimental_connect_to_cluster(resolver)
  >>> topology = tf.tpu.experimental.initialize_tpu_system(resolver)
  >>> device_assignment = tf.tpu.experimental.DeviceAssignment.build(
  ...     topology,
  ...     computation_shape=[1, 1, 1, 2],
  ...     num_replicas=1)
  >>> strategy = tf.distribute.TPUStrategy(
  ...     resolver, experimental_device_assignment=device_assignment)

  Then you can run a `tf.add` operation only on logical device 0.

  >>> @tf.function()
  ... def step_fn(inputs):
  ...   features, _ = inputs
  ...   output = tf.add(features, features)
  ...
  ...   # Add operation will be executed on logical device 0.
  ...   output = strategy.experimental_assign_to_logical_device(output, 0)
  ...   return output
  >>> dist_dataset = strategy.distribute_datasets_from_function(
  ...     dataset_fn)
  >>> iterator = iter(dist_dataset)
  >>> strategy.run(step_fn, args=(next(iterator),))

  `experimental_spmd_xla_partitioning` enables the experimental XLA SPMD feature
  for model parallelism. This flag can reduce the compilation time and HBM
  requirements. When running in this mode, every input tensor must either be
  partitioned (via `strategy.experimental_split_to_logical_devices`) or fully
  replicated (via `strategy.experimental_replicate_to_logical_devices`) to all
  logical devices. And calling `strategy.experimental_assign_to_logical_device`
  will result in a ValueError in this mode.
  """

  def __init__(self,
               tpu_cluster_resolver=None,
               experimental_device_assignment=None,
               experimental_spmd_xla_partitioning=False):
    # [FUNCTION DOC]
    # Constructor. Inputs: tpu_cluster_resolver (optional resolver describing the TPU cluster;
    # when missing a resolver for the local worker is assumed), experimental_device_assignment
    # (optional placement of replicas on cores), experimental_spmd_xla_partitioning (boolean,
    # default false; enables XLA SPMD partitioning). Builds the TPU extended implementation and
    # passes it to the base strategy, then records the strategy kind, number of workers and
    # replicas per worker in monitoring gauges, and enables packed variables in eager mode.
    # Returns nothing.
    """Synchronous training in TPU donuts or Pods.

    Args:
      tpu_cluster_resolver: A
        `tf.distribute.cluster_resolver.TPUClusterResolver` instance, which
        provides information about the TPU cluster. If None, it will assume
        running on a local TPU worker.
      experimental_device_assignment: Optional
        `tf.tpu.experimental.DeviceAssignment` to specify the placement of
        replicas on the TPU cluster.
      experimental_spmd_xla_partitioning: If True, enable the SPMD (Single
        Program Multiple Data) mode in XLA compiler. This flag only affects the
        performance of XLA compilation and the HBM requirement of the compiled
        TPU program. Ceveat: if this flag is True, calling
        `tf.distribute.TPUStrategy.experimental_assign_to_logical_device` will
        result in a ValueError.
    """
    super().__init__(
        TPUExtended(
            self,
            tpu_cluster_resolver,
            device_assignment=experimental_device_assignment,
            use_spmd_for_xla_partitioning=experimental_spmd_xla_partitioning,
        )
    )
    distribute_lib.distribution_strategy_gauge.get_cell("V2").set("TPUStrategy")
    distribute_lib.distribution_strategy_replica_gauge.get_cell(
        "num_workers").set(self.extended.num_hosts)
    distribute_lib.distribution_strategy_replica_gauge.get_cell(
        "num_replicas_per_worker").set(self.extended.num_replicas_per_host)
    # Packed variable is used to reduce the overhead of function execution.
    # For a DistributedVariable, only one variable handle is captured into a
    # function graph. It's only supported in eager mode.
    self._enable_packed_variable_in_eager_mode = True

  def run(self, fn, args=(), kwargs=None, options=None):
    # [FUNCTION DOC]
    # Runs fn on every TPU replica. Inputs: fn (function whose output is a nest of tensors),
    # args (positional arguments, default empty), kwargs (keyword arguments, optional), options
    # (optional run options; defaults to fresh run options). Validates fn (raising a
    # not-implemented error for pure eager use), partially applies distributed variables, converts
    # fn with autograph, and executes it replicated through the extended tpu_run. Returns the
    # merged per-replica results with the structure of fn output.
    """Run the computation defined by `fn` on each TPU replica.

    Executes ops specified by `fn` on each replica. If `args` or `kwargs` have
    `tf.distribute.DistributedValues`, such as those produced by a
    `tf.distribute.DistributedDataset` from
    `tf.distribute.Strategy.experimental_distribute_dataset` or
    `tf.distribute.Strategy.distribute_datasets_from_function`,
    when `fn` is executed on a particular replica, it will be executed with the
    component of `tf.distribute.DistributedValues` that correspond to that
    replica.

    `fn` may call `tf.distribute.get_replica_context()` to access members such
    as `all_reduce`.

    All arguments in `args` or `kwargs` should either be nest of tensors or
    `tf.distribute.DistributedValues` containing tensors or composite tensors.

    Example usage:

    >>> resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
    >>> tf.config.experimental_connect_to_cluster(resolver)
    >>> tf.tpu.experimental.initialize_tpu_system(resolver)
    >>> strategy = tf.distribute.TPUStrategy(resolver)
    >>> @tf.function
    ... def run():
    ...   def value_fn(value_context):
    ...     return value_context.num_replicas_in_sync
    ...   distributed_values = (
    ...       strategy.experimental_distribute_values_from_function(value_fn))
    ...   def replica_fn(input):
    ...     return input * 2
    ...   return strategy.run(replica_fn, args=(distributed_values,))
    >>> result = run()

    Args:
      fn: The function to run. The output must be a `tf.nest` of `Tensor`s.
      args: (Optional) Positional arguments to `fn`.
      kwargs: (Optional) Keyword arguments to `fn`.
      options: (Optional) An instance of `tf.distribute.RunOptions` specifying
        the options to run `fn`.

    Returns:
      Merged return value of `fn` across replicas. The structure of the return
      value is the same as the return value from `fn`. Each element in the
      structure can either be `tf.distribute.DistributedValues`, `Tensor`
      objects, or `Tensor`s (for example, if running on a single replica).
    """
    validate_run_function(fn)

    fn, args, kwargs = _maybe_partial_apply_variables(fn, args, kwargs)

    # Note: the target function is converted to graph even when in Eager mode,
    # so autograph is on by default here.
    fn = autograph.tf_convert(fn, autograph_ctx.control_status_ctx())
    options = options or distribute_lib.RunOptions()
    return self.extended.tpu_run(fn, args, kwargs, options)

  @property
  def cluster_resolver(self):
    # [FUNCTION DOC]
    # Property. Returns the cluster resolver stored on the extended object (user supplied or the
    # default local TPU resolver).
    """Returns the cluster resolver associated with this strategy.

    `tf.distribute.TPUStrategy` provides the associated
    `tf.distribute.cluster_resolver.ClusterResolver`. If the user provides one
    in `__init__`, that instance is returned; if the user does not, a default
    `tf.distribute.cluster_resolver.TPUClusterResolver` is provided.
    """
    return self.extended._tpu_cluster_resolver  # pylint: disable=protected-access

  def experimental_assign_to_logical_device(self, tensor, logical_device_id):
    # [FUNCTION DOC]
    # Annotates a tensor so that its ops are placed on the given logical device (core) of each
    # replica. Inputs: tensor to annotate and logical_device_id (integer). Raises a value error if
    # SPMD partitioning is enabled, or if the id is negative or not below the number of logical
    # devices per replica. Returns the annotated tensor with the same value.
    """Adds annotation that `tensor` will be assigned to a logical device.

    This adds an annotation to `tensor` specifying that operations on
    `tensor` will be invoked on logical core device id `logical_device_id`.
    When model parallelism is used, the default behavior is that all ops
    are placed on zero-th logical device.

    ```python

    # Initializing TPU system with 2 logical devices and 4 replicas.
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
    tf.config.experimental_connect_to_cluster(resolver)
    topology = tf.tpu.experimental.initialize_tpu_system(resolver)
    device_assignment = tf.tpu.experimental.DeviceAssignment.build(
        topology,
        computation_shape=[1, 1, 1, 2],
        num_replicas=4)
    strategy = tf.distribute.TPUStrategy(
        resolver, experimental_device_assignment=device_assignment)
    iterator = iter(inputs)

    @tf.function()
    def step_fn(inputs):
      output = tf.add(inputs, inputs)

      # Add operation will be executed on logical device 0.
      output = strategy.experimental_assign_to_logical_device(output, 0)
      return output

    strategy.run(step_fn, args=(next(iterator),))
    ```

    Args:
      tensor: Input tensor to annotate.
      logical_device_id: Id of the logical core to which the tensor will be
        assigned.

    Raises:
      ValueError: The logical device id presented is not consistent with total
      number of partitions specified by the device assignment or the TPUStrategy
      is constructed with `experimental_spmd_xla_partitioning=True`.

    Returns:
      Annotated tensor with identical value as `tensor`.
    """
    if self.extended._use_spmd_for_xla_partitioning:  # pylint: disable=protected-access
      raise ValueError(
          "Cannot assign a tensor to a logical device in SPMD mode. To disable "
          "SPMD, Please construct the TPUStrategy with "
          "`experimental_spmd_xla_partitioning=False`")

    num_logical_devices_per_replica = self.extended._tpu_devices.shape[1]  # pylint: disable=protected-access
    if (logical_device_id < 0 or
        logical_device_id >= num_logical_devices_per_replica):
      raise ValueError("`logical_core_id` to assign must be lower then total "
                       "number of logical devices per replica. Received "
                       "logical device id {} but there are only total of {} "
                       "logical devices in replica.".format(
                           logical_device_id, num_logical_devices_per_replica))
    return xla_sharding.assign_device(
        tensor, logical_device_id, use_sharding_op=True)

  def experimental_split_to_logical_devices(self, tensor, partition_dimensions):
    # [FUNCTION DOC]
    # Annotates a tensor so it is split across logical devices. Inputs: tensor, and
    # partition_dimensions, a flat list of integers with one entry per tensor dimension whose
    # product equals the number of logical devices per replica. Raises a value error if the list
    # length differs from the tensor rank, if a statically known dimension is not divisible by
    # its split count, or if the product does not match the number of logical devices. Returns
    # the tensor annotated with a tiled sharding. Unknown dimensions skip the divisibility check.
    """Adds annotation that `tensor` will be split across logical devices.

    This adds an annotation to tensor `tensor` specifying that operations on
    `tensor` will be split among multiple logical devices. Tensor `tensor` will
    be split across dimensions specified by `partition_dimensions`.
    The dimensions of `tensor` must be divisible by corresponding value in
    `partition_dimensions`.

    For example, for system with 8 logical devices, if `tensor` is an image
    tensor with shape (batch_size, width, height, channel) and
    `partition_dimensions` is [1, 2, 4, 1], then `tensor` will be split
    2 in width dimension and 4 way in height dimension and the split
    tensor values will be fed into 8 logical devices.

    ```python
    # Initializing TPU system with 8 logical devices and 1 replica.
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
    tf.config.experimental_connect_to_cluster(resolver)
    topology = tf.tpu.experimental.initialize_tpu_system(resolver)
    device_assignment = tf.tpu.experimental.DeviceAssignment.build(
        topology,
        computation_shape=[1, 2, 2, 2],
        num_replicas=1)
    # Construct the TPUStrategy. Since we are going to split the image across
    # logical devices, here we set `experimental_spmd_xla_partitioning=True`
    # so that the partitioning can be compiled in SPMD mode, which usually
    # results in faster compilation and smaller HBM requirement if the size of
    # input and activation tensors are much bigger than that of the model
    # parameters. Note that this flag is suggested but not a hard requirement
    # for `experimental_split_to_logical_devices`.
    strategy = tf.distribute.TPUStrategy(
        resolver, experimental_device_assignment=device_assignment,
        experimental_spmd_xla_partitioning=True)

    iterator = iter(inputs)

    @tf.function()
    def step_fn(inputs):
      inputs = strategy.experimental_split_to_logical_devices(
        inputs, [1, 2, 4, 1])

      # model() function will be executed on 8 logical devices with `inputs`
      # split 2 * 4  ways.
      output = model(inputs)
      return output

    strategy.run(step_fn, args=(next(iterator),))
    ```
    Args:
      tensor: Input tensor to annotate.
      partition_dimensions: An unnested list of integers with the size equal to
        rank of `tensor` specifying how `tensor` will be partitioned. The
        product of all elements in `partition_dimensions` must be equal to the
        total number of logical devices per replica.

    Raises:
      ValueError: 1) If the size of partition_dimensions does not equal to rank
        of `tensor` or 2) if product of elements of `partition_dimensions` does
        not match the number of logical devices per replica defined by the
        implementing DistributionStrategy's device specification or
        3) if a known size of `tensor` is not divisible by corresponding
        value in `partition_dimensions`.

    Returns:
      Annotated tensor with identical value as `tensor`.
    """
    num_logical_devices_per_replica = self.extended._tpu_devices.shape[1]  # pylint: disable=protected-access
    num_partition_splits = np.prod(partition_dimensions)
    input_shape = tensor.shape
    tensor_rank = len(input_shape)

    if tensor_rank != len(partition_dimensions):
      raise ValueError("Length of `partition_dimensions` must equal to the "
                       "rank of `tensor.shape` ({}). Received "
                       "len(partition_dimensions)={}.".format(
                           tensor_rank, len(partition_dimensions)))

    for dim_index, dim_size in enumerate(input_shape):
      if dim_size is None:
        continue

      split_size = partition_dimensions[dim_index]
      if dim_size % split_size != 0:
        raise ValueError("Tensor shape at `partition_dimensions[{}]` must be "
                         "divisible by corresponding value specified "
                         "by `partition_dimensions` ({}). Received: {}.".format(
                             dim_index, split_size, dim_size))

    if num_partition_splits != num_logical_devices_per_replica:
      raise ValueError(
          "The product of `partition_dimensions` should be the same as the "
          "number of logical devices (={}). Received `partition_dimensions`={},"
          "and their product is {}.".format(num_logical_devices_per_replica,
                                            partition_dimensions,
                                            num_partition_splits))

    tile_assignment = np.arange(num_partition_splits).reshape(
        partition_dimensions)
    return xla_sharding.tile(tensor, tile_assignment, use_sharding_op=True)

  def experimental_replicate_to_logical_devices(self, tensor):
    # [FUNCTION DOC]
    # Annotates a tensor so it is fully replicated to all logical devices. Input: tensor.
    # Returns the tensor annotated with a replicated sharding (same value).
    """Adds annotation that `tensor` will be replicated to all logical devices.

    This adds an annotation to tensor `tensor` specifying that operations on
    `tensor` will be invoked on all logical devices.

    ```python
    # Initializing TPU system with 2 logical devices and 4 replicas.
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
    tf.config.experimental_connect_to_cluster(resolver)
    topology = tf.tpu.experimental.initialize_tpu_system(resolver)
    device_assignment = tf.tpu.experimental.DeviceAssignment.build(
        topology,
        computation_shape=[1, 1, 1, 2],
        num_replicas=4)
    strategy = tf.distribute.TPUStrategy(
        resolver, experimental_device_assignment=device_assignment)

    iterator = iter(inputs)

    @tf.function()
    def step_fn(inputs):
      images, labels = inputs
      images = strategy.experimental_split_to_logical_devices(
        inputs, [1, 2, 4, 1])

      # model() function will be executed on 8 logical devices with `inputs`
      # split 2 * 4  ways.
      output = model(inputs)

      # For loss calculation, all logical devices share the same logits
      # and labels.
      labels = strategy.experimental_replicate_to_logical_devices(labels)
      output = strategy.experimental_replicate_to_logical_devices(output)
      loss = loss_fn(labels, output)

      return loss

    strategy.run(step_fn, args=(next(iterator),))
    ```
    Args:
      tensor: Input tensor to annotate.

    Returns:
      Annotated tensor with identical value as `tensor`.
    """
    return xla_sharding.replicate(tensor, use_sharding_op=True)


@tf_export.tf_export("distribute.experimental.TPUStrategy", v1=[])
@deprecation.deprecated_endpoints("distribute.experimental.TPUStrategy")
class TPUStrategy(distribute_lib.Strategy):
  # [CLASS DOC]
  # Deprecated experimental TPU strategy (version 2 API) for TPUs and pods. Like the main TPU
  # strategy but without SPMD and logical-device annotation helpers. Construction logs a
  # deprecation warning. Delegates to the TPU extended implementation. Exported under the
  # experimental namespace with a deprecated-endpoint marker.
  """Synchronous training on TPUs and TPU Pods.

  To construct a TPUStrategy object, you need to run the
  initialization code as below:

  >>> resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
  >>> tf.config.experimental_connect_to_cluster(resolver)
  >>> tf.tpu.experimental.initialize_tpu_system(resolver)
  >>> strategy = tf.distribute.experimental.TPUStrategy(resolver)

  While using distribution strategies, the variables created within the
  strategy's scope will be replicated across all the replicas and can be kept in
  sync using all-reduce algorithms.

  To run TF2 programs on TPUs, you can either use `.compile` and
  `.fit` APIs in `tf.keras` with TPUStrategy, or write your own customized
  training loop by calling `strategy.run` directly. Note that
  TPUStrategy doesn't support pure eager execution, so please make sure the
  function passed into `strategy.run` is a `tf.function` or
  `strategy.run` is called inside a `tf.function` if eager
  behavior is enabled.
  """

  def __init__(self,
               tpu_cluster_resolver=None,
               device_assignment=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: tpu_cluster_resolver (optional resolver for the TPU cluster; when
    # missing a local worker resolver is assumed) and device_assignment (optional placement of
    # replicas on cores). Logs a deprecation warning, builds the TPU extended implementation,
    # records strategy kind, number of workers and replicas per worker in monitoring gauges, and
    # enables packed variables in eager mode. Returns nothing.
    """Synchronous training in TPU donuts or Pods.

    Args:
      tpu_cluster_resolver: A tf.distribute.cluster_resolver.TPUClusterResolver,
        which provides information about the TPU cluster.
      device_assignment: Optional `tf.tpu.experimental.DeviceAssignment` to
        specify the placement of replicas on the TPU cluster.
    """
    logging.warning(
        "`tf.distribute.experimental.TPUStrategy` is deprecated, please use "
        "the non-experimental symbol `tf.distribute.TPUStrategy` instead.")

    super().__init__(
        TPUExtended(
            self,
            tpu_cluster_resolver,
            device_assignment=device_assignment,
        )
    )
    distribute_lib.distribution_strategy_gauge.get_cell("V2").set("TPUStrategy")
    distribute_lib.distribution_strategy_replica_gauge.get_cell(
        "num_workers").set(self.extended.num_hosts)
    distribute_lib.distribution_strategy_replica_gauge.get_cell(
        "num_replicas_per_worker").set(self.extended.num_replicas_per_host)
    # Packed variable is used to reduce the overhead of function execution.
    # For a DistributedVariable, only one variable handle is captured into a
    # function graph. It's only supported in eager mode.
    self._enable_packed_variable_in_eager_mode = True

  # TODO(cjfj): Modify `_call_for_each_replica` in `TPUExtended` such that this
  # can use the default implementation.
  # This implementation runs a single step. It does not use infeed or outfeed.
  def run(self, fn, args=(), kwargs=None, options=None):
    # [FUNCTION DOC]
    # Same behavior as the main TPU strategy run: validates fn, partially applies distributed
    # variables, converts fn with autograph and executes it replicated via tpu_run, returning the
    # merged per-replica results. Inputs: fn, args, kwargs, options (all as in the base class;
    # options default to fresh run options). Raises a not-implemented error for pure eager use.
    """See base class."""
    validate_run_function(fn)

    fn, args, kwargs = _maybe_partial_apply_variables(fn, args, kwargs)

    # Note: the target function is converted to graph even when in Eager mode,
    # so autograph is on by default here.
    fn = autograph.tf_convert(fn, autograph_ctx.control_status_ctx())
    options = options or distribute_lib.RunOptions()
    return self.extended.tpu_run(fn, args, kwargs, options)

  @property
  def cluster_resolver(self):
    # [FUNCTION DOC]
    # Property. Returns the cluster resolver stored on the extended object (user supplied or the
    # default local TPU resolver).
    """Returns the cluster resolver associated with this strategy.

    `tf.distribute.experimental.TPUStrategy` provides the
    associated `tf.distribute.cluster_resolver.ClusterResolver`. If the user
    provides one in `__init__`, that instance is returned; if the user does
    not, a default
    `tf.distribute.cluster_resolver.TPUClusterResolver` is provided.
    """
    return self.extended._tpu_cluster_resolver  # pylint: disable=protected-access


@tf_export.tf_export(v1=["distribute.experimental.TPUStrategy"])
class TPUStrategyV1(distribute_lib.StrategyV1):
  # [CLASS DOC]
  # Version 1 (compat) TPU strategy used with estimators and the legacy API. Created with a
  # cluster resolver, a steps-per-run count and an optional device assignment; delegates to
  # the TPU extended implementation and exposes steps_per_run and run.
  """TPU distribution strategy implementation."""

  def __init__(self,
               tpu_cluster_resolver=None,
               steps_per_run=None,
               device_assignment=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: tpu_cluster_resolver (optional resolver for the TPU cluster),
    # steps_per_run (optional number of steps run on device per host call), device_assignment
    # (optional placement of replicas on cores). Builds the TPU extended implementation, records
    # strategy kind, number of workers and replicas per worker in monitoring gauges, and enables
    # packed variables in eager mode. Returns nothing.
    """Initializes the TPUStrategy object.

    Args:
      tpu_cluster_resolver: A tf.distribute.cluster_resolver.TPUClusterResolver,
          which provides information about the TPU cluster.
      steps_per_run: Number of steps to run on device before returning to the
          host. Note that this can have side-effects on performance, hooks,
          metrics, summaries etc.
          This parameter is only used when Distribution Strategy is used with
          Keras.
      device_assignment: Optional `tf.tpu.experimental.DeviceAssignment` to
          specify the placement of replicas on the TPU cluster. Currently only
          supports the usecase of using a single core within a TPU cluster.
    """
    super().__init__(TPUExtended(
        self, tpu_cluster_resolver, steps_per_run, device_assignment))
    distribute_lib.distribution_strategy_gauge.get_cell("V1").set("TPUStrategy")
    distribute_lib.distribution_strategy_replica_gauge.get_cell(
        "num_workers").set(self.extended.num_hosts)
    distribute_lib.distribution_strategy_replica_gauge.get_cell(
        "num_replicas_per_worker").set(self.extended.num_replicas_per_host)
    # Packed variable is used to reduce the overhead of function execution.
    # For a DistributedVariable, only one variable handle is captured into a
    # function graph. It's only supported in eager mode.
    self._enable_packed_variable_in_eager_mode = True

  @property
  def steps_per_run(self):
    # [FUNCTION DOC]
    # Property. Returns the number of steps run on device per host call, read from the extended
    # object. Deprecated alias of the extended steps_per_run attribute.
    """DEPRECATED: use .extended.steps_per_run instead."""
    return self._extended.steps_per_run

  # TODO(cjfj): Modify `_call_for_each_replica` in `TPUExtended` such that this
  # can use the default implementation.
  # This implementation runs a single step. It does not use infeed or outfeed.
  def run(self, fn, args=(), kwargs=None, options=None):
    # [FUNCTION DOC]
    # Runs fn on every replica with per-replica arguments split out. Inputs: fn, args, kwargs,
    # options (optional run options such as bucketizing dynamic shapes; default fresh options).
    # Validates fn, partially applies distributed variables, converts fn with autograph and runs
    # it through tpu_run. Returns merged per-replica results. Raises not-implemented error for
    # pure eager use.
    """Run `fn` on each replica, with the given arguments.

    Executes ops specified by `fn` on each replica. If `args` or `kwargs` have
    "per-replica" values, such as those produced by a "distributed `Dataset`",
    when `fn` is executed on a particular replica, it will be executed with the
    component of those "per-replica" values that correspond to that replica.

    `fn` may call `tf.distribute.get_replica_context()` to access members such
    as `all_reduce`.

    All arguments in `args` or `kwargs` should either be nest of tensors or
    per-replica objects containing tensors or composite tensors.

    Users can pass strategy specific options to `options` argument. An example
    to enable bucketizing dynamic shapes in `TPUStrategy.run`
    is:

    >>> resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
    >>> tf.config.experimental_connect_to_cluster(resolver)
    >>> tf.tpu.experimental.initialize_tpu_system(resolver)
    >>> strategy = tf.distribute.experimental.TPUStrategy(resolver)

    >>> options = tf.distribute.RunOptions(
    ...     experimental_bucketizing_dynamic_shape=True)

    >>> dataset = tf.data.Dataset.range(
    ...    strategy.num_replicas_in_sync, output_type=dtypes.float32).batch(
    ...        strategy.num_replicas_in_sync, drop_remainder=True)
    >>> input_iterator = iter(strategy.experimental_distribute_dataset(dataset))

    >>> @tf.function()
    ... def step_fn(inputs):
    ...  output = tf.reduce_sum(inputs)
    ...  return output

    >>> strategy.run(step_fn, args=(next(input_iterator),), options=options)

    Args:
      fn: The function to run. The output must be a `tf.nest` of `Tensor`s.
      args: (Optional) Positional arguments to `fn`.
      kwargs: (Optional) Keyword arguments to `fn`.
      options: (Optional) An instance of `tf.distribute.RunOptions` specifying
        the options to run `fn`.

    Returns:
      Merged return value of `fn` across replicas. The structure of the return
      value is the same as the return value from `fn`. Each element in the
      structure can either be "per-replica" `Tensor` objects or `Tensor`s
      (for example, if running on a single replica).
    """
    validate_run_function(fn)

    fn, args, kwargs = _maybe_partial_apply_variables(fn, args, kwargs)

    fn = autograph.tf_convert(fn, autograph_ctx.control_status_ctx())
    options = options or distribute_lib.RunOptions()
    return self.extended.tpu_run(fn, args, kwargs, options)


# TODO(josh11b): Switch to V2 when we no longer need to support tf.compat.v1.
class TPUExtended(distribute_lib.StrategyExtendedV1):
  # [CLASS DOC]
  # Implementation behind all TPU strategies. State: the cluster resolver and its system
  # metadata, an optional device assignment, a two-dimensional array of TPU device names indexed
  # by replica then logical device, the host device, ordered maps of input worker devices (TPU
  # and host variants), steps per run, flags (static shapes required, SPMD partitioning,
  # variable policy, custom device, data reorder), a stack of the current logical device, a weak
  # cache of compiled TPU functions keyed by user function, and a lazy variable tracker.
  # Provides input pipelines, variable creation, reductions, gathers, updates, broadcasts and the
  # tpu_run entry point. Registers an exit hook to wait for async eager execution.
  """Implementation of TPUStrategy."""

  def __init__(
      self,
      container_strategy,
      tpu_cluster_resolver=None,
      steps_per_run=None,
      device_assignment=None,
      use_spmd_for_xla_partitioning=False,
  ):
    # [FUNCTION DOC]
    # Constructor. Inputs: container_strategy (the owning public strategy), tpu_cluster_resolver
    # (optional; defaults to a resolver for the local TPU worker), steps_per_run (optional,
    # defaults to one), device_assignment (optional placement of replicas onto cores),
    # use_spmd_for_xla_partitioning (boolean, default false). Queries the system metadata, builds
    # the table of TPU device names (from the assignment when given, otherwise one logical device
    # per TPU), determines the host device, groups input worker devices by host, sets the flags
    # described in the class doc and detects whether TPU devices are custom devices. Side
    # effects: when executing eagerly registers an exit hook waiting for async executors.
    # Returns nothing.
    super().__init__(container_strategy)

    if tpu_cluster_resolver is None:
      tpu_cluster_resolver = tpu_cluster_resolver_lib.TPUClusterResolver("")

    if steps_per_run is None:
      # TODO(frankchn): Warn when we are being used by DS/Keras and this is
      # not specified.
      steps_per_run = 1

    # `self._tpu_function_cache` is a dict of `tf.function`s, thus if a
    # `tf.function` is passed into `strategy.run` in eager mode, the
    # `tf.function` won't get retraced.
    self._tpu_function_cache = weakref.WeakKeyDictionary()

    self._tpu_cluster_resolver = tpu_cluster_resolver
    self._tpu_metadata = self._tpu_cluster_resolver.get_tpu_system_metadata()
    self._device_assignment = device_assignment

    tpu_devices_flat = [
        d.name for d in self._tpu_metadata.devices if "device:TPU:" in d.name]

    # `self._tpu_devices` is a two-dimensional NumPy array of strings. It is
    # indexed using `[replica_id][logical_device_id]`.
    if device_assignment is None:
      self._tpu_devices = np.array(
          [[d] for d in tpu_devices_flat], dtype=object)
    else:
      job_name = device_spec.DeviceSpecV2.from_string(tpu_devices_flat[0]).job

      tpu_devices = []
      for replica_id in range(device_assignment.num_replicas):
        replica_devices = []

        for logical_core in range(device_assignment.num_cores_per_replica):
          replica_devices.append(
              device_util.canonicalize(
                  device_assignment.tpu_device(
                      replica=replica_id,
                      logical_core=logical_core,
                      job=job_name)))

        tpu_devices.append(replica_devices)
      self._tpu_devices = np.array(tpu_devices, dtype=object)
    self._host_device = device_util.get_host_for_device(self._tpu_devices[0][0])

    # Preload the data onto the TPUs. Currently we always preload onto logical
    # device 0 for each replica.
    # TODO(cjfj): Create `InputWorkers` lazily, allowing users to place the
    # input onto a different logical device?
    self._device_input_worker_devices = collections.OrderedDict()
    self._host_input_worker_devices = collections.OrderedDict()
    for tpu_device in self._tpu_devices[:, 0]:
      host_device = device_util.get_host_for_device(tpu_device)
      self._device_input_worker_devices.setdefault(host_device, [])
      self._device_input_worker_devices[host_device].append(tpu_device)
      self._host_input_worker_devices.setdefault(host_device, [])
      self._host_input_worker_devices[host_device].append(host_device)

    # TODO(sourabhbajaj): Remove this once performance of running one step
    # at a time is comparable to multiple steps.
    self.steps_per_run = steps_per_run
    self._require_static_shapes = True

    self.experimental_enable_get_next_as_optional = True

    self._logical_device_stack = [0]

    if context.executing_eagerly():
      # In async remote eager, we want to sync the executors before exiting the
      # program.
      atexit.register(context.async_wait)

    # Flag to turn on VariablePolicy. Var policy is deprecated because there is
    # another effort unifying DistributedVariables (see values_v2.py). SPMD XLA
    # partitioning is not implemented for var policies.
    # TODO(b/202048882): remove var policy from TPUStrategy.
    self._use_var_policy = not use_spmd_for_xla_partitioning

    # Flag to enable XLA SPMD partitioning.
    self._use_spmd_for_xla_partitioning = use_spmd_for_xla_partitioning

    self._using_custom_device = False
    devices = self._tpu_devices[:, self._logical_device_stack[-1]]
    for d in devices:
      if context.is_custom_device(d):
        self._using_custom_device = True
        break

    # This is a flag to enable data reorder which is used
    # to match IteratorGetNext's device with the TPUExecute device.
    self._enable_data_reorder = False

  def _place_input_on_local_cpu_devices(self):
    # [FUNCTION DOC]
    # HELPER: rebuilds the input worker device maps so that each TPU device is fed from the host
    # CPU whose index matches the TPU device index (instead of one CPU per host). Takes no
    # arguments, returns nothing, and replaces the two ordered maps of device and host input
    # worker devices on the instance. Intended for use when a device assignment is set.
    """Place input on local CPU devices.

    For example, if the tpu_devices are:
    '/job:worker/replica:0/task:0/device:TPU:0',
    '/job:worker/replica:0/task:1/device:TPU:0',
    '/job:worker/replica:0/task:1/device:TPU:1',
    '/job:worker/replica:0/task:0/device:TPU:1',


    the host_input_worker_devices will be:
    {
        '/job:worker/replica:0/task:0/device:CPU:0': [
            '/job:worker/replica:0/task:0/device:TPU:0',
        ],
        '/job:worker/replica:0/task:1/device:CPU:0', [
            '/job:worker/replica:0/task:1/device:TPU:0',
        ],
        '/job:worker/replica:0/task:1/device:CPU:1': [
            '/job:worker/replica:0/task:1/device:TPU:1',
        ],
        '/job:worker/replica:0/task:0/device:CPU:1': [
            '/job:worker/replica:0/task:0/device:TPU:1',
        ],
    }
    This will make sure that the input is placed on the corresponding host CPU
    device if the device assignment is set.
    """
    self._device_input_worker_devices = collections.OrderedDict()
    self._host_input_worker_devices = collections.OrderedDict()
    for tpu_device in self._tpu_devices[:, 0]:
      host_device = device_util.get_host_for_device(
          tpu_device,
          device_index=tf_device.DeviceSpec.from_string(
              tpu_device
          ).device_index,
      )
      self._device_input_worker_devices.setdefault(host_device, [])
      self._device_input_worker_devices[host_device].append(tpu_device)
      self._host_input_worker_devices.setdefault(host_device, [])
      self._host_input_worker_devices[host_device].append(host_device)

  def _get_replica_order(self):
    # [FUNCTION DOC]
    # HELPER: ordering of replicas for input iterators.
    # Returns nothing (a null value) when data reordering is disabled (the default). Otherwise
    # returns a list of replica indices obtained by sorting the TPU devices on job, replica,
    # device type, task and device index, so data can be placed next to its computation.
    """Get the replica order based on the tpu device order.

    For example, if the tpu_devices are:
    '/job:worker/replica:0/task:0/device:TPU:0',
    '/job:worker/replica:0/task:0/device:TPU:2',
    '/job:worker/replica:0/task:1/device:TPU:0',
    '/job:worker/replica:0/task:1/device:TPU:2',
    '/job:worker/replica:0/task:1/device:TPU:6',
    '/job:worker/replica:0/task:1/device:TPU:4',
    '/job:worker/replica:0/task:0/device:TPU:6',
    '/job:worker/replica:0/task:0/device:TPU:4',

    the returned replica order will be:
    [0, 1, 7, 6, 2, 3, 5, 4]

    This replica order will be used to reorder the data returned by the
    iterators,
    so that they can be placed on the same node as their computation graphs.

    Returns:
      A list containing the order ids of corresponding TPU devices.
    """
    if not self._enable_data_reorder:
      return None

    tpu_devices = self._tpu_devices[:, 0]

    devices_with_ids = []
    for i, tpu_device in enumerate(tpu_devices):
      spec = tf_device.DeviceSpec.from_string(tpu_device)
      devices_with_ids.append((
          (
              spec.job,
              spec.replica,
              spec.device_type,
              spec.task,
              spec.device_index,
          ),
          i,
      ))
    return [i for _, i in sorted(devices_with_ids)]

  def _validate_colocate_with_variable(self, colocate_with_variable):
    # [FUNCTION DOC]
    # Checks that the variable to colocate with belongs to this strategy by delegating to the
    # shared validation utility. Returns nothing; may raise from the utility.
    distribute_utils.validate_colocate(colocate_with_variable, self)

  def _make_dataset_iterator(self, dataset):
    # [FUNCTION DOC]
    # Version 1 input path. Builds input workers from the device input worker map and returns a
    # dataset iterator over the given dataset, using the container strategy and the number of
    # replicas in sync.
    """Make iterators for each of the TPU hosts."""
    input_workers = input_lib.InputWorkers(
        tuple(self._device_input_worker_devices.items()))
    return input_lib_v1.DatasetIterator(
        dataset,
        input_workers,
        self._container_strategy(),
        num_replicas_in_sync=self._num_replicas_in_sync)

  def _make_input_fn_iterator(
      self,
      input_fn,
      replication_mode=distribute_lib.InputReplicationMode.PER_WORKER):
    # [FUNCTION DOC]
    # KNOWN ISSUE: replication_mode is accepted but ignored.
    # Version 1 input path. Inputs: input_fn (function building a dataset given an input
    # context) and replication_mode (defaults to per worker, but is not used). Creates one input
    # context per input worker and returns an input-function iterator.
    input_contexts = []
    input_workers = input_lib.InputWorkers(
        tuple(self._device_input_worker_devices.items()))
    num_workers = input_workers.num_workers
    for i in range(num_workers):
      input_contexts.append(
          distribute_lib.InputContext(
              num_input_pipelines=num_workers,
              input_pipeline_id=i,
              num_replicas_in_sync=self._num_replicas_in_sync))
    return input_lib_v1.InputFunctionIterator(input_fn, input_workers,
                                              input_contexts,
                                              self._container_strategy())

  def _experimental_make_numpy_dataset(self, numpy_input, session):
    # [FUNCTION DOC]
    # Wraps numpy input into a one-host dataset placed on the host device, using the given
    # session. Returns that dataset.
    return numpy_dataset.one_host_numpy_dataset(
        numpy_input, numpy_dataset.SingleDevice(self._host_device),
        session)

  def _get_input_workers(self, options):
    # [FUNCTION DOC]
    # HELPER: selects input worker devices. Input: options (optional input options). When there
    # are no options or fetching to device is enabled, returns input workers over the TPU
    # devices grouped by host; otherwise returns input workers over the host devices.
    if not options or options.experimental_fetch_to_device:
      return input_lib.InputWorkers(
          tuple(self._device_input_worker_devices.items()))
    else:
      return input_lib.InputWorkers(
          tuple(self._host_input_worker_devices.items()))

  def _check_spec(self, element_spec):
    # [FUNCTION DOC]
    # HELPER: validation for datasets that prefetch to device.
    # KNOWN ISSUE: the error message says "with spec" but formats the Python type of the spec,
    # not the spec itself.
    # Input: element_spec, the dataset
    # element structure (a per-replica spec is unwrapped to its components). Raises a value
    # error naming the offending path if any component is a sparse or ragged tensor spec.
    # Returns nothing.
    if isinstance(element_spec, values.PerReplicaSpec):
      element_spec = element_spec._component_specs  # pylint: disable=protected-access
    specs = nest.flatten_with_joined_string_paths(element_spec)
    for path, spec in specs:
      if isinstance(spec, (sparse_tensor.SparseTensorSpec,
                           ragged_tensor.RaggedTensorSpec)):
        raise ValueError(
            "Found tensor {} with spec {}. TPUStrategy does not support "
            "distributed datasets with device prefetch when using sparse or "
            "ragged tensors. If you intend to use sparse or ragged tensors, "
            "please pass a tf.distribute.InputOptions object with "
            "experimental_fetch_to_device set to False to your dataset "
            "distribution function.".format(path, type(spec)))

  def _experimental_distribute_dataset(self, dataset, options):
    # [FUNCTION DOC]
    # Distributes a dataset over TPU hosts. Inputs: dataset and options (may be null). Raises a
    # not-implemented error if per-replica replication mode is requested. When fetching to device,
    # validates the element spec. Returns a distributed dataset built with the input workers,
    # container strategy, replicas in sync, options and replica order.
    if (options and options.experimental_replication_mode ==
        distribute_lib.InputReplicationMode.PER_REPLICA):
      raise NotImplementedError(
          "InputReplicationMode.PER_REPLICA "
          "is only supported in "
          "`experimental_distribute_datasets_from_function`."
      )
    if options is None or options.experimental_fetch_to_device:
      self._check_spec(dataset.element_spec)

    return input_util.get_distributed_dataset(
        dataset,
        self._get_input_workers(options),
        self._container_strategy(),
        num_replicas_in_sync=self._num_replicas_in_sync,
        options=options,
        replica_order=self._get_replica_order(),
    )

  def _distribute_datasets_from_function(self, dataset_fn, options):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the not-implemented error message says per-replica mode is only supported by
    # mirrored strategy, which is misleading for this TPU strategy.
    # Distributes datasets built by a function. Inputs: dataset_fn (builds a dataset given an
    # input context) and options (may be null). Raises a not-implemented error for per-replica
    # replication mode. Creates one input context per input worker, builds the distributed
    # datasets, validates the element spec if fetching to device, and returns the result.
    if (options and options.experimental_replication_mode ==
        distribute_lib.InputReplicationMode.PER_REPLICA):
      raise NotImplementedError(
          "InputReplicationMode.PER_REPLICA "
          "is only supported in "
          " `experimental_distribute_datasets_from_function` "
          "of tf.distribute.MirroredStrategy")
    input_workers = self._get_input_workers(options)
    input_contexts = []
    num_workers = input_workers.num_workers
    for i in range(num_workers):
      input_contexts.append(distribute_lib.InputContext(
          num_input_pipelines=num_workers,
          input_pipeline_id=i,
          num_replicas_in_sync=self._num_replicas_in_sync))

    distributed_dataset = input_util.get_distributed_datasets_from_function(
        dataset_fn,
        input_workers,
        input_contexts,
        self._container_strategy(),
        options=options,
        replica_order=self._get_replica_order(),
    )

    # We can only check after the dataset_fn is called.
    if options is None or options.experimental_fetch_to_device:
      self._check_spec(distributed_dataset.element_spec)
    return distributed_dataset

  def _experimental_distribute_values_from_function(self, value_fn):
    # [FUNCTION DOC]
    # Input: value_fn, called once per replica with a value context (replica id and number of
    # replicas). Collects the per-replica results and returns them regrouped as a per-replica
    # value (always wrapped).
    per_replica_values = []
    for replica_id in range(self._num_replicas_in_sync):
      per_replica_values.append(
          value_fn(distribute_lib.ValueContext(replica_id,
                                               self._num_replicas_in_sync)))
    return distribute_utils.regroup(per_replica_values, always_wrap=True)

  # TODO(priyag): Deal with OutOfRange errors once b/111349762 is fixed.
  # TODO(sourabhbajaj): Remove the initial_loop_values parameter when we have
  # a mechanism to infer the outputs of `fn`. Pending b/110550782.
  def _experimental_run_steps_on_iterator(
      self, fn, multi_worker_iterator, iterations, initial_loop_values=None):
    # [FUNCTION DOC]
    # Version 1 multi-step execution. Inputs: fn (step function taking a multi-step context and
    # inputs), multi_worker_iterator (supplies per-replica inputs), iterations (steps to run),
    # initial_loop_values (optional loop carried values, defaults to empty). Wraps fn so that
    # last-step outputs are forced after the step; builds a rewritten step that gets the next
    # inputs, selects each replica part and replicates the step across TPU cores; runs it once
    # when steps per run is one, otherwise in a TPU repeat loop, placed on the host device.
    # Groups outputs into the context run op and, when the outputs are a list, regroups the
    # flat tensor outputs by output instead of by device. Returns the multi-step context.
    # Side effects: temporarily stores the outer control flow context on the instance.
    # Wrap `fn` for repeat.
    if initial_loop_values is None:
      initial_loop_values = {}
    initial_loop_values = nest.flatten(initial_loop_values)
    ctx = input_lib.MultiStepContext()

    def run_fn(inputs):
      # [FUNCTION DOC]
      # HELPER: single step run on a TPU core. Calls the step function with the context and the
      # inputs; if the context has last-step outputs, returns identity copies of them made after
      # the step result, otherwise returns the step result.
      """Single step on the TPU device."""
      fn_result = fn(ctx, inputs)
      flat_last_step_outputs = nest.flatten(ctx.last_step_outputs)
      if flat_last_step_outputs:
        with ops.control_dependencies([fn_result]):
          return [array_ops.identity(f) for f in flat_last_step_outputs]
      else:
        return fn_result

    # We capture the control_flow_context at this point, before we run `fn`
    # inside a while_loop and TPU replicate context. This is useful in cases
    # where we might need to exit these contexts and get back to the outer
    # context to do some things, for e.g. create an op which should be
    # evaluated only once at the end of the loop on the host. One such usage
    # is in creating metrics' value op.
    self._outer_control_flow_context = (
        ops.get_default_graph()._get_control_flow_context())  # pylint: disable=protected-access

    def rewrite_fn(*args):
      # [FUNCTION DOC]
      # HELPER: the step function rewritten for TPU. Ignores its arguments, takes the next
      # per-replica inputs from the iterator, selects each replica share, replicates the step
      # across cores with the device assignment and SPMD option, and returns the replicated
      # outputs (flattened when they are a list of lists).
      """The rewritten step fn running on TPU."""
      del args

      per_replica_inputs = multi_worker_iterator.get_next()
      replicate_inputs = []
      for replica_id in range(self._num_replicas_in_sync):
        select_replica = lambda x: distribute_utils.select_replica(  # pylint: disable=g-long-lambda
            replica_id, x)   # pylint: disable=cell-var-from-loop
        replicate_inputs.append((nest.map_structure(
            select_replica, per_replica_inputs),))

      replicate_outputs = tpu.replicate(
          run_fn,
          replicate_inputs,
          device_assignment=self._device_assignment,
          xla_options=tpu.XLAOptions(use_spmd_for_xla_partitioning=self
                                     ._use_spmd_for_xla_partitioning))
      # If run_fn has tensor outputs, tpu.replicate returns a list of list. We
      # will flatten it in this case. If run_fn has no tensor outputs,
      # tpu.replicate returns a list of no_ops, we will keep the output as it
      # is.
      if isinstance(replicate_outputs[0], list):
        replicate_outputs = nest.flatten(replicate_outputs)

      return replicate_outputs

    # TODO(sourabhbajaj): The input to while loop should be based on the
    # output type of the step_fn
    assert isinstance(initial_loop_values, list)
    initial_loop_values = initial_loop_values * self._num_replicas_in_sync

    # Put the while loop op on TPU host 0.
    with ops.device(self._host_device):
      if self.steps_per_run == 1:
        replicate_outputs = rewrite_fn()
      else:
        replicate_outputs = training_loop.repeat(iterations, rewrite_fn,
                                                 initial_loop_values)

    del self._outer_control_flow_context
    ctx.run_op = control_flow_ops.group(replicate_outputs)

    if isinstance(replicate_outputs, list):
      # Filter out any ops from the outputs, typically this would be the case
      # when there were no tensor outputs.
      last_step_tensor_outputs = [
          x for x in replicate_outputs if not isinstance(x, ops.Operation)
      ]

      # Outputs are currently of the structure (flattened)
      # [output0_device0, output1_device0, output2_device0,
      #  output0_device1, output1_device1, output2_device1,
      #  ...]
      # Convert this to the following structure instead: (grouped by output)
      # [[output0_device0, output0_device1],
      #  [output1_device0, output1_device1],
      #  [output2_device0, output2_device1]]
      output_num = len(last_step_tensor_outputs) // self._num_replicas_in_sync
      last_step_tensor_outputs = [
          last_step_tensor_outputs[i::output_num] for i in range(output_num)
      ]
    else:
      # no tensors returned.
      last_step_tensor_outputs = []

    _set_last_step_outputs(ctx, last_step_tensor_outputs)
    return ctx

  def _call_for_each_replica(self, fn, args, kwargs):
    # [FUNCTION DOC]
    # Calls fn with the given positional and keyword arguments once, inside a TPU replica
    # context for the container strategy, and returns its result.
    # TODO(jhseu): Consider making it so call_for_each_replica implies that
    # we're in a tpu.rewrite(), and update TPUMirroredVariable accordingly.
    with _TPUReplicaContext(self._container_strategy()):
      return fn(*args, **kwargs)

  @contextlib.contextmanager
  def experimental_logical_device(self, logical_device_id):
    # [FUNCTION DOC]
    # Context manager placing variables and ops on a logical device. Input: logical_device_id
    # (integer). Raises a value error if it is not below the number of logical devices per
    # replica. Pushes the id on the logical device stack for the duration of the block; inside a
    # TPU context also scopes ops to that core. Pops the stack on exit. Yields nothing.
    """Places variables and ops on the specified logical device."""
    num_logical_devices_per_replica = self._tpu_devices.shape[1]
    if logical_device_id >= num_logical_devices_per_replica:
      raise ValueError(
          "`logical_device_id` not in range (was {}, but there are only {} "
          "logical devices per replica).".format(
              logical_device_id, num_logical_devices_per_replica))

    self._logical_device_stack.append(logical_device_id)
    try:
      if tpu_util.enclosing_tpu_context() is None:
        yield
      else:
        with ops.device(tpu.core(logical_device_id)):
          yield
    finally:
      self._logical_device_stack.pop()

  def _experimental_initialize_system(self):
    # [FUNCTION DOC]
    # HELPER: private hook for estimators. Initializes the TPU system using the cluster resolver.
    # Takes no arguments and returns nothing.
    """Experimental method added to be used by Estimator.

    This is a private method only to be used by Estimator. Other frameworks
    should directly be calling `tf.tpu.experimental.initialize_tpu_system`
    """
    tpu_cluster_resolver_lib.initialize_tpu_system(self._tpu_cluster_resolver)

  def _create_variable(self, next_creator, **kwargs):
    # [FUNCTION DOC]
    # Creates a mirrored TPU variable. Inputs: next_creator (the next variable creator in the
    # chain) and keyword arguments for the variable. Special keywords consumed: a flag to skip
    # the mirrored creator (just delegates to next_creator), a custom TPU variable creator
    # (called instead with next_creator), and colocate_with (a single device places creation
    # there; otherwise supplies the devices to use, default is the current logical device across
    # replicas). Chooses a creator: replicated variables per logical core when SPMD partitioning
    # is on with more than one core per replica, otherwise one variable per replica; the
    # uninitialized (lazy batched) variants are used when batch variable initialization is on and
    # no custom device is in use. Then builds the mirrored variable through the shared utility
    # with the TPU class and policy mappings. Returns the mirrored variable; in the batched mode
    # it is registered with the lazy variable tracker.
    """Create a TPUMirroredVariable. See `DistributionStrategy.scope`."""
    # TODO(bfontain): Replace all uses of skip_mirrored_creator with
    # a trivial custom_tpu_variable_creator.
    if kwargs.pop("skip_mirrored_creator", False):
      return next_creator(**kwargs)

    custom_tpu_variable_creator = kwargs.pop(
        "custom_tpu_variable_creator", None
    )
    if custom_tpu_variable_creator is not None:
      return custom_tpu_variable_creator(next_creator, **kwargs)

    colocate_with = kwargs.pop("colocate_with", None)
    if colocate_with is None:
      devices = self._tpu_devices[:, self._logical_device_stack[-1]]
    elif isinstance(colocate_with, numpy_dataset.SingleDevice):
      with ops.device(colocate_with.device):
        return next_creator(**kwargs)
    else:
      devices = colocate_with._devices  # pylint: disable=protected-access

    num_replicas, num_cores_per_replica = self._tpu_devices.shape

    def _create_mirrored_tpu_variables(**kwargs):
      # [FUNCTION DOC]
      # HELPER: creator of one variable per replica. Takes the variable keyword arguments and
      # returns the list of variables, one per device. The initial value is evaluated once (in
      # the init scope, calling it if callable) and reused; later replicas get distinct names
      # with a replica suffix. Variables are created with silent device placement policy.
      # Asserts none of the created variables is already a TPU mirrored variable.
      """Returns a list of `tf.Variable`s.

      The list contains `number_replicas` `tf.Variable`s and can be used to
      initialize a `TPUMirroredVariable`.

      Args:
        **kwargs: the keyword arguments for creating a variable
      """
      initial_value = None
      value_list = []
      for i, d in enumerate(devices):
        with ops.device(d):
          if i == 0:
            initial_value = kwargs["initial_value"]
            # Note: some v1 code expects variable initializer creation to happen
            # inside a init_scope.
            with maybe_init_scope():
              initial_value = initial_value() if callable(
                  initial_value) else initial_value

          if i > 0:
            # Give replicas meaningful distinct names:
            var0name = value_list[0].name.split(":")[0]
            # We append a / to variable names created on replicas with id > 0 to
            # ensure that we ignore the name scope and instead use the given
            # name as the absolute name of the variable.
            kwargs["name"] = "%s/replica_%d/" % (var0name, i)
          kwargs["initial_value"] = initial_value

          with context.device_policy(context.DEVICE_PLACEMENT_SILENT):
            v = next_creator(**kwargs)

          assert not isinstance(v, tpu_values.TPUMirroredVariable)
          value_list.append(v)
      return value_list

    def _create_mirrored_tpu_replicated_variables(**kwargs):
      # [FUNCTION DOC]
      # HELPER: creator for SPMD mode. Takes the variable keyword arguments and returns a list
      # with one replicated-variable object per replica, each holding one variable per logical
      # core, created on that core with the shared initial value (evaluated once in the init
      # scope). Replica variables are named with the base name plus a replica suffix.
      """Returns a list of `TPUReplicatedVariable`s.

      The list consists of `num_replicas` `TPUReplicatedVariable`s and can be
      used to initialize a `TPUMirroredVariable`. Each `TPUReplicatedVariable`
      contains a list of `tf.Variable`s which are replicated to
      `num_cores_per_replica` logical cores to enable XLA SPMD compilation.

      Args:
        **kwargs: the keyword arguments for creating a variable
      """
      initial_value = kwargs["initial_value"]
      # Note: some v1 code expects variable initializer creation to happen
      # inside a init_scope.
      with maybe_init_scope():
        initial_value = initial_value() if callable(
            initial_value) else initial_value

      mirrored_replicated_var_list = []

      for replica_id in range(num_replicas):
        replicated_var_list = []
        for logic_core_id in range(num_cores_per_replica):
          with ops.device(self._tpu_devices[replica_id][logic_core_id]):
            kwargs["initial_value"] = initial_value
            v = next_creator(**kwargs)
          replicated_var_list.append(v)
        replica_name = "{}/r:{}".format(kwargs["name"], replica_id)
        tpu_replicated_var = tpu_replicated_variable.TPUReplicatedVariable(
            variables=replicated_var_list, name=replica_name)

        mirrored_replicated_var_list.append(tpu_replicated_var)
      return mirrored_replicated_var_list

    # TODO(b/271767559): Consider either changing the innermost default_creator
    # to uninitialized_variable_creator or only swapping the next_creator with
    # uninitialized_variable_creator if the next_creator is the default_creator.

    def uninitialized_variable_creator(**kwargs):
      # [FUNCTION DOC]
      # HELPER: creates a TPU uninitialized variable from the keyword arguments and attaches the
      # lazy variable tracker as its lazy scope, after registering it with the tracker. Returns
      # the variable.
      uninitialized_variable = tpu_util.TPUUninitializedVariable(**kwargs)

      self.lazy_variable_tracker.add_uninitialized_var(
          uninitialized_variable
      )
      setattr(uninitialized_variable, "_lazy_scope", self.lazy_variable_tracker)
      return uninitialized_variable

    def _create_uninitialized_mirrored_tpu_variables(**kwargs):
      # [FUNCTION DOC]
      # HELPER: like the per-replica creator but for batched initialization. Takes the variable
      # keyword arguments. With no initial value, falls back to the eager per-replica creator.
      # Otherwise converts the initial value to a tensor once, fills in the dtype and shape from
      # it when missing, names replicas distinctly, and creates uninitialized variables under
      # silent device placement. Returns the list of per-replica variables.
      """Returns a list of `tf.Variable`s.

      The list contains `number_replicas` `tf.Variable`s and can be used to
      initialize a `TPUMirroredVariable`.

      Args:
        **kwargs: the keyword arguments for creating a variable
      """
      if kwargs.get("initial_value", None) is None:
        return _create_mirrored_tpu_variables(**kwargs)

      value_list = []
      initial_value = None

      for i, d in enumerate(devices):
        with ops.device(d):
          if i == 0:
            initial_value = kwargs.get("initial_value", None)

            with maybe_init_scope():
              if initial_value is not None:
                if callable(initial_value):
                  initial_value = initial_value()

                initial_value = ops.convert_to_tensor(
                    initial_value, dtype=kwargs.get("dtype", None)
                )

          if i > 0:
            # Give replicas meaningful distinct names:
            var0name = value_list[0].name.split(":")[0]
            # We append a / to variable names created on replicas with id > 0 to
            # ensure that we ignore the name scope and instead use the given
            # name as the absolute name of the variable.
            kwargs["name"] = "%s/replica_%d/" % (var0name, i)

          kwargs["initial_value"] = initial_value

          if kwargs.get("dtype", None) is None:
            kwargs["dtype"] = kwargs["initial_value"].dtype

          if kwargs.get("shape", None) is None:
            kwargs["shape"] = kwargs["initial_value"].shape

          with context.device_policy(context.DEVICE_PLACEMENT_SILENT):
            v = uninitialized_variable_creator(**kwargs)

          assert not isinstance(v, tpu_values.TPUMirroredVariable)
          value_list.append(v)
      return value_list

    def _create_uninitialized_mirrored_tpu_replicated_variables(**kwargs):
      # [FUNCTION DOC]
      # HELPER: batched-initialization creator for SPMD mode. Takes the variable keyword
      # arguments. With no initial value, falls back to the eager replicated creator. Otherwise
      # converts the initial value once, fills in dtype and shape, and creates uninitialized
      # variables on each logical core of each replica, wrapping each replica group in a
      # replicated-variable object with a replica-suffixed name. Returns the list of them.
      """Returns a list of `TPUReplicatedVariable`s.

      The list consists of `num_replicas` `TPUReplicatedVariable`s and can be
      used to initialize a `TPUMirroredVariable`. Each `TPUReplicatedVariable`
      contains a list of `tf.Variable`s which are replicated to
      `num_cores_per_replica` logical cores to enable XLA SPMD compilation.

      Args:
        **kwargs: the keyword arguments for creating a variable
      """
      dtype = kwargs.get("dtype", None)
      shape = kwargs.get("shape", None)
      initial_value = kwargs.get("initial_value", None)

      if initial_value is None:
        return _create_mirrored_tpu_replicated_variables(**kwargs)

      with maybe_init_scope():
        if initial_value is not None:
          if callable(initial_value):
            initial_value = initial_value()

          initial_value = ops.convert_to_tensor(
              initial_value, dtype=dtype
          )

          kwargs["initial_value"] = initial_value

          if dtype is None:
            kwargs["dtype"] = kwargs["initial_value"].dtype
          if shape is None:
            kwargs["shape"] = kwargs["initial_value"].shape

      mirrored_replicated_var_list = []

      for replica_id in range(num_replicas):
        replicated_var_list = []
        for logic_core_id in range(num_cores_per_replica):
          with ops.device(self._tpu_devices[replica_id][logic_core_id]):
            v = uninitialized_variable_creator(**kwargs)
          replicated_var_list.append(v)
        replica_name = "{}/r:{}".format(kwargs["name"], replica_id)
        tpu_replicated_var = tpu_replicated_variable.TPUReplicatedVariable(
            variables=replicated_var_list, name=replica_name
        )

        mirrored_replicated_var_list.append(tpu_replicated_var)
      return mirrored_replicated_var_list

    if not self._using_custom_device and enable_batch_variable_initialization():
      if self._use_spmd_for_xla_partitioning and num_cores_per_replica > 1:
        real_creator = _create_uninitialized_mirrored_tpu_replicated_variables
      else:
        real_creator = _create_uninitialized_mirrored_tpu_variables

      kwargs["experimental_batch_initialization"] = True

    else:
      if self._use_spmd_for_xla_partitioning and num_cores_per_replica > 1:
        real_creator = _create_mirrored_tpu_replicated_variables
      else:
        real_creator = _create_mirrored_tpu_variables

    mirrored_variable = distribute_utils.create_mirrored_variable(
        self._container_strategy(),
        real_creator,
        distribute_utils.TPU_VARIABLE_CLASS_MAPPING,
        distribute_utils.TPU_VARIABLE_POLICY_MAPPING,
        **kwargs,
    )

    if not self._using_custom_device and enable_batch_variable_initialization():
      setattr(mirrored_variable, "_lazy_scope", self.lazy_variable_tracker)

    return mirrored_variable

  @property
  def lazy_variable_tracker(self):
    # [FUNCTION DOC]
    # Property. Returns the lazy variable tracker, creating and caching it on first access.
    if not getattr(self, "_lazy_variable_tracker", None):
      self._lazy_variable_tracker = tpu_util.LazyVariableTracker()
    return self._lazy_variable_tracker

  def _resource_creator_scope(self):
    # [FUNCTION DOC]
    # Returns a resource creator scope for static hash tables. Tables are created once on each
    # input host device and wrapped as a per-worker resource. Other resource kinds are not yet
    # handled.

    def lookup_creator(next_creator, *args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: creator used by the scope. Creates the table with next_creator on every host
      # device and returns a per-worker resource mapping host device to table.
      host_to_table = collections.OrderedDict()
      for host_device in self._device_input_worker_devices.keys():
        with ops.device(host_device):
          host_to_table[host_device] = next_creator(*args, **kwargs)

      return values.PerWorkerResource(self._container_strategy(), host_to_table)

    # TODO(b/194362531): Define creator(s) for other resources.
    return ops.resource_creator_scope("StaticHashTable", lookup_creator)

  def _gather_to_implementation(self, value, destinations, axis, options):
    # [FUNCTION DOC]
    # Gathers a distributed value by concatenation along an axis. Inputs: value, destinations,
    # axis, options (unused). Non-distributed values are returned unchanged. Uses packed variable
    # components when available. Concatenates in chunks if the number of values exceeds the XLA
    # op-by-op input limit. Result is broadcast to the destinations and returned.
    if not isinstance(value, values.DistributedValues):
      return value

    value_list = list(value.values)
    # pylint: disable=protected-access
    if isinstance(
        value,
        values.DistributedVariable) and value._packed_variable is not None:
      value_list = list(
          value._packed_variable.on_device(d)
          for d in value._packed_variable.devices)
    # pylint: enable=protected-access

    # Currently XLA op by op mode has a limit for the number of inputs for a
    # single op, thus we break one `add_n` op into a group of `add_n` ops to
    # work around the constraint.
    if len(value.values) <= _XLA_OP_BY_OP_INPUTS_LIMIT:
      output = array_ops.concat(value_list, axis=axis)
    else:
      output = array_ops.concat(
          value_list[:_XLA_OP_BY_OP_INPUTS_LIMIT], axis=axis)
      for i in range(_XLA_OP_BY_OP_INPUTS_LIMIT, len(value_list),
                     _XLA_OP_BY_OP_INPUTS_LIMIT - 1):
        output = array_ops.concat(
            [output] + value_list[i:i + _XLA_OP_BY_OP_INPUTS_LIMIT - 1],
            axis=axis)

    output = self._broadcast_output(destinations, output)
    return output

  def _broadcast_output(self, destinations, output):
    # [FUNCTION DOC]
    # HELPER: places a reduced or gathered output at the destinations. If there is a single
    # destination device different from the host device, copies the output there via identity;
    # with several destinations uses a simple broadcast. Returns the output.
    devices = cross_device_ops_lib.get_devices_from(destinations)

    if len(devices) == 1:
      # If necessary, copy to requested destination.
      dest_canonical = device_util.canonicalize(devices[0])
      host_canonical = device_util.canonicalize(self._host_device)

      if dest_canonical != host_canonical:
        with ops.device(dest_canonical):
          output = array_ops.identity(output)
    else:
      output = cross_device_ops_lib.simple_broadcast(output, destinations)

    return output

  def _reduce_to(self, reduce_op, value, destinations, options):
    # [FUNCTION DOC]
    # Reduces a value across replicas. Inputs: reduce_op (sum or mean), value, destinations,
    # options (unused). Inside a TPU context, uses cross-replica sum (scaling for mean first) and
    # raises a not-implemented error for other reduce ops. Non-distributed values go to the
    # non-distributed reduction helper. Otherwise adds the per-replica values (in chunks beyond
    # the XLA input limit), divides by the count for mean, then broadcasts to destinations.
    if (isinstance(value, values.DistributedValues) or
        tensor_util.is_tf_type(value)
       ) and tpu_util.enclosing_tpu_context() is not None:
      if reduce_op == reduce_util.ReduceOp.MEAN:
        # TODO(jhseu):  Revisit once we support model-parallelism.
        # scalar_mul maintains the type of value: tensor or IndexedSlices.
        value = math_ops.scalar_mul((1./self._num_replicas_in_sync), value)
      elif reduce_op != reduce_util.ReduceOp.SUM:
        raise NotImplementedError(
            f"`reduce_op`={reduce_op} is not supported. Currently we only "
            "support ReduceOp.SUM and ReduceOp.MEAN in TPUStrategy.")
      return tpu_ops.cross_replica_sum(value)

    if not isinstance(value, values.DistributedValues):
      # This function handles reducing values that are not PerReplica or
      # Mirrored values. For example, the same value could be present on all
      # replicas in which case `value` would be a single value or value could
      # be 0.
      return cross_device_ops_lib.reduce_non_distributed_value(
          reduce_op, value, destinations, self._num_replicas_in_sync)

    value_list = value.values
    # pylint: disable=protected-access
    if isinstance(
        value,
        values.DistributedVariable) and value._packed_variable is not None:
      value_list = tuple(
          value._packed_variable.on_device(d)
          for d in value._packed_variable.devices)
    # pylint: enable=protected-access

    # Currently XLA op by op mode has a limit for the number of inputs for a
    # single op, thus we break one `add_n` op into a group of `add_n` ops to
    # work around the constraint.
    # TODO(cjfj): Detect when it is possible to use `cross_replica_sum`.
    if len(value.values) <= _XLA_OP_BY_OP_INPUTS_LIMIT:
      output = math_ops.add_n(value_list)
    else:
      output = array_ops.zeros_like(value_list[0], dtype=value_list[0].dtype)
      for i in range(0, len(value_list), _XLA_OP_BY_OP_INPUTS_LIMIT):
        output += math_ops.add_n(value_list[i:i + _XLA_OP_BY_OP_INPUTS_LIMIT])

    if reduce_op == reduce_util.ReduceOp.MEAN:
      output *= (1. / len(value_list))

    output = self._broadcast_output(destinations, output)
    return output

  def _update(self, var, fn, args, kwargs, group):
    # [FUNCTION DOC]
    # Applies an update function to a variable. Inputs: var (a TPU variable or resource
    # variable), fn, args, kwargs, group (whether to return grouped result). Asserts var type.
    # Inside a TPU context calls fn directly. In a non-eager function with a packed variable,
    # calls fn on the packed variable. Otherwise updates each replica component on its device
    # under update context and name scope, checking arguments are mirrored for synchronized
    # aggregated variables, and regroups the updates. Returns the update result.
    assert isinstance(var, tpu_values.TPUVariableMixin) or isinstance(
        var, resource_variable_ops.BaseResourceVariable)
    if tpu_util.enclosing_tpu_context() is not None:
      if group:
        return fn(var, *args, **kwargs)
      else:
        return (fn(var, *args, **kwargs),)

    # Inside `tf.function`, we don't expand PackedVariable in python as it will
    # be expanded later during function instantiation in the runtime.
    packed_var = var._packed_variable  # pylint: disable=protected-access
    if packed_var is not None and not context.executing_eagerly():
      if group:
        return fn(packed_var, *args, **kwargs)
      else:
        return (fn(packed_var, *args, **kwargs),)

    # Otherwise, we revert to MirroredStrategy behavior and update the variable
    # on each replica directly.
    updates = []
    values_and_devices = []
    if packed_var is not None:
      for device in packed_var.devices:
        values_and_devices.append((packed_var, device))
    else:
      for value in var.values:
        values_and_devices.append((value, value.device))

    if (var.synchronization != variables_lib.VariableSynchronization.ON_READ and
        var.aggregation != variables_lib.VariableAggregation.NONE):
      distribute_utils.assert_mirrored(args)
      distribute_utils.assert_mirrored(kwargs)
    for i, value_and_device in enumerate(values_and_devices):
      value = value_and_device[0]
      device = value_and_device[1]
      name = "update_%d" % i
      with ops.device(device), \
           distribute_lib.UpdateContext(i), \
           ops.name_scope(name):
        # If args and kwargs are not mirrored, the value is returned as is.
        updates.append(
            fn(value, *distribute_utils.select_replica(i, args),
               **distribute_utils.select_replica(i, kwargs)))
    return distribute_utils.update_regroup(self, updates, group)

  def read_var(self, var):
    # [FUNCTION DOC]
    # Reads the current value of a TPU or resource variable (asserts the type) and returns it.
    assert isinstance(var, tpu_values.TPUVariableMixin) or isinstance(
        var, resource_variable_ops.BaseResourceVariable)
    return var.read_value()

  def value_container(self, value):
    # [FUNCTION DOC]
    # Returns the container of a value; for TPU it is the value itself.
    return value

  def _broadcast_to(self, tensor, destinations):
    # [FUNCTION DOC]
    # Broadcasts a value to all replicas. Inputs: tensor and destinations (ignored). Python
    # float or integer values are returned as is. Inside a TPU context does an all-to-all and
    # returns the first replica result; outside returns the tensor unchanged.
    del destinations
    # This is both a fast path for Python constants, and a way to delay
    # converting Python values to a tensor until we know what type it
    # should be converted to. Otherwise we have trouble with:
    #   global_step.assign_add(1)
    # since the `1` gets broadcast as an int32 but global_step is int64.
    if isinstance(tensor, (float, int)):
      return tensor
    if tpu_util.enclosing_tpu_context() is not None:
      broadcast_tensor = [tensor for _ in range(self._num_replicas_in_sync)]
      result = tpu_ops.all_to_all(
          broadcast_tensor,
          concat_dimension=0,
          split_dimension=0,
          split_count=self._num_replicas_in_sync)

      # This uses the broadcasted value from the first replica because the only
      # caller of this is for ONLY_FIRST_REPLICA variables aggregation.
      return result[0]
    return tensor

  @property
  def num_hosts(self):
    # [FUNCTION DOC]
    # Property. Number of TPU hosts: from the system metadata, or the count of distinct host
    # devices over replicas when a device assignment is set.
    if self._device_assignment is None:
      return self._tpu_metadata.num_hosts

    return len(set([self._device_assignment.host_device(r)
                    for r in range(self._device_assignment.num_replicas)]))

  @property
  def num_replicas_per_host(self):
    # [FUNCTION DOC]
    # Property. Replicas per host: cores per host from the metadata, or with a device assignment
    # the smaller of the number of replicas and cores per host divided by cores per replica.
    if self._device_assignment is None:
      return self._tpu_metadata.num_of_cores_per_host

    # TODO(sourabhbajaj): Remove this method we use inputs and remove infeed
    # as the computation of num_replicas_per_host is not a constant
    # when using device_assignment. This is a temporary workaround to support
    # StatefulRNN as everything is 1 in that case.
    # This method needs to take host_id as input for correct computation.
    max_models_per_host = (self._tpu_metadata.num_of_cores_per_host //
                           self._device_assignment.num_cores_per_replica)
    return min(self._device_assignment.num_replicas, max_models_per_host)

  @property
  def _num_replicas_in_sync(self):
    # [FUNCTION DOC]
    # Property. Total replicas: the number of cores, or the device assignment replica count.
    if self._device_assignment is None:
      return self._tpu_metadata.num_cores
    return self._device_assignment.num_replicas

  @property
  def experimental_between_graph(self):
    # [FUNCTION DOC]
    # Property. Always false: TPU strategy is not between-graph replicated.
    return False

  @property
  def experimental_should_init(self):
    # [FUNCTION DOC]
    # Property. Always true: the strategy should run initialization.
    return True

  @property
  def should_checkpoint(self):
    # [FUNCTION DOC]
    # Property. Always true: this worker should write checkpoints.
    return True

  @property
  def should_save_summary(self):
    # [FUNCTION DOC]
    # Property. Always true: this worker should save summaries.
    return True

  @property
  def worker_devices(self):
    # [FUNCTION DOC]
    # Property. Tuple of TPU device names for the current logical device, one per replica.
    return tuple(self._tpu_devices[:, self._logical_device_stack[-1]])

  @property
  def parameter_devices(self):
    # [FUNCTION DOC]
    # Property. Same as the worker devices.
    return self.worker_devices

  @property
  def tpu_hardware_feature(self):
    # [FUNCTION DOC]
    # Property. Returns a hardware feature object built from the cluster resolver hardware
    # feature information.
    """Return the `tf.tpu.experimental.HardwareFeature` class."""
    return tpu_hardware_feature.HardwareFeature(
        self._tpu_cluster_resolver.tpu_hardware_feature)

  def non_slot_devices(self, var_list):
    # [FUNCTION DOC]
    # Returns the host device for non-slot variables. The variable list is ignored.
    return self._host_device

  def _update_non_slot(self, colocate_with, fn, args, kwargs, group):
    # [FUNCTION DOC]
    # Runs an update on the host device in an update context. Inputs: colocate_with (ignored),
    # fn, args, kwargs, group. Returns fn result if grouped; otherwise the result structure
    # mapped to local results.
    del colocate_with
    with ops.device(self._host_device), distribute_lib.UpdateContext(None):
      result = fn(*args, **kwargs)
      if group:
        return result
      else:
        return nest.map_structure(self._local_results, result)

  def _configure(self,
                 session_config=None,
                 cluster_spec=None,
                 task_type=None,
                 task_id=None):
    # [FUNCTION DOC]
    # Configures the strategy. Inputs: session_config (optional), cluster_spec, task_type and
    # task_id (all ignored). If a session config is given it is overwritten in place by the
    # updated copy. Returns nothing.
    del cluster_spec, task_type, task_id
    if session_config:
      session_config.CopyFrom(self._update_config_proto(session_config))

  def _update_config_proto(self, config_proto):
    # [FUNCTION DOC]
    # Returns a deep copy of the given session config with session state isolation turned on and
    # the cluster definition copied from the resolver cluster spec if there is one. Input is not
    # modified.
    updated_config = copy.deepcopy(config_proto)
    updated_config.isolate_session_state = True
    cluster_spec = self._tpu_cluster_resolver.cluster_spec()
    if cluster_spec:
      updated_config.cluster_def.CopyFrom(cluster_spec.as_cluster_def())
    return updated_config

  # TODO(priyag): Delete this once all strategies use global batch size.
  @property
  def _global_batch_size(self):
    # [FUNCTION DOC]
    # Property. Always true: dataset and numpy iterators use global batch size.
    """`make_dataset_iterator` and `make_numpy_iterator` use global batch size.

    `make_input_fn_iterator` assumes per-replica batching.

    Returns:
      Boolean.
    """
    return True

  def tpu_run(self, fn, args, kwargs, options=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: options defaults to null but the created function reads attributes of options
    # unconditionally, so calling without options would fail; callers always pass it.
    # Runs fn replicated on TPU. Inputs: fn, args, kwargs, options (run options).
    # Builds or fetches the TPU function and calls it with args and kwargs; returns its result.
    func = self._tpu_function_creator(fn, options)
    return func(args, kwargs)

  def _tpu_function_creator(self, fn, options):
    # [FUNCTION DOC]
    # Builds the function that replicates the user function across TPU cores. Inputs: fn and
    # options. When eager and a function for fn is cached, returns the cache entry. Otherwise
    # defines the inner function, wraps it as a traced function when eager and caches it by fn
    # (side effect), then returns it. Not eager: returns the plain inner function.
    if context.executing_eagerly() and fn in self._tpu_function_cache:
      return self._tpu_function_cache[fn]

    strategy = self._container_strategy()

    def tpu_function(args, kwargs):
      # [FUNCTION DOC]
      # Replicated computation. Inputs: args and kwargs (null means empty). Builds per-replica
      # inputs (replica id constant, selected args, selected kwargs), computes maximum shapes
      # when dynamic batch size is enabled (raising a value error for unknown rank inputs), picks
      # power-of-two padding when bucketizing, and replicates the wrapped user function with the
      # XLA options inside the strategy scope. Filters operations out of outputs, handles the
      # single or no tensor return case, repacks outputs per replica and returns them regrouped.
      """TF Function used to replicate the user computation."""
      logging.vlog(1,
                   "`TPUStrategy.run` is called with [args: %s] [kwargs: %s]",
                   args, kwargs)

      if kwargs is None:
        kwargs = {}

      # Used to re-structure flattened output tensors from `tpu.replicate()`
      # into a structured format.
      result = [[]]

      def replicated_fn(replica_id, replica_args, replica_kwargs):
        # [FUNCTION DOC]
        # Wrapper run on each core. Runs the user function inside a TPU replica context with the
        # given replica id, stores the result in an outer holder (so its structure can be
        # recovered) and returns it.
        """Wraps user function to provide replica ID and `Tensor` inputs."""
        with _TPUReplicaContext(strategy, replica_id_in_sync_group=replica_id):
          result[0] = fn(*replica_args, **replica_kwargs)
        return result[0]

      replicate_inputs = []  # By replica.
      for i in range(strategy.num_replicas_in_sync):
        replicate_inputs.append(
            [constant_op.constant(i, dtype=dtypes.int32),
             distribute_utils.select_replica(i, args),
             distribute_utils.select_replica(i, kwargs)])

      # Construct and pass `maximum_shapes` so that we could support dynamic
      # shapes using dynamic padder.
      if options.experimental_enable_dynamic_batch_size and replicate_inputs:
        maximum_shapes = []
        flattened_list = nest.flatten(replicate_inputs[0])
        for input_tensor in flattened_list:
          if tensor_util.is_tf_type(input_tensor):
            rank = input_tensor.shape.rank
          else:
            rank = np.ndim(input_tensor)
          if rank is None:
            raise ValueError(
                "input tensor {} to TPUStrategy.run() has unknown rank, "
                "which is not allowed".format(input_tensor))
          maximum_shape = tensor_shape.TensorShape([None] * rank)
          maximum_shapes.append(maximum_shape)
        maximum_shapes = nest.pack_sequence_as(replicate_inputs[0],
                                               maximum_shapes)
      else:
        maximum_shapes = None

      if options.experimental_bucketizing_dynamic_shape:
        padding_spec = tpu.PaddingSpec.POWER_OF_TWO
      else:
        padding_spec = None

      with strategy.scope():
        xla_options = options.experimental_xla_options or tpu.XLAOptions(
            use_spmd_for_xla_partitioning=self._use_spmd_for_xla_partitioning)
        replicate_outputs = tpu.replicate(
            replicated_fn,
            replicate_inputs,
            device_assignment=self._device_assignment,
            maximum_shapes=maximum_shapes,
            padding_spec=padding_spec,
            xla_options=xla_options)

      # Remove all no ops that may have been added during 'tpu.replicate()'
      filter_ops = lambda x: [o for o in x if not isinstance(o, ops.Operation)]
      if isinstance(result[0], list):
        result[0] = filter_ops(result[0])

      # Workaround for `tpu.replicate` behaviour when single `Tensor` returned.
      if result[0] is None or isinstance(result[0], ops.Operation):
        replicate_outputs = [None] * len(replicate_outputs)
      else:
        replicate_outputs = [
            nest.pack_sequence_as(result[0], filter_ops(nest.flatten(output)))
            for output in replicate_outputs
        ]
      return distribute_utils.regroup(replicate_outputs)

    if context.executing_eagerly():
      tpu_function = def_function.function(tpu_function)
      self._tpu_function_cache[fn] = tpu_function
    return tpu_function

  def _in_multi_worker_mode(self):
    # [FUNCTION DOC]
    # Always false: the whole TPU cluster is treated as a single worker by higher layers.
    """Whether this strategy indicates working in multi-worker settings."""
    # TPUStrategy has different distributed training structure that the whole
    # cluster should be treated as single worker from higher-level (e.g. Keras)
    # library's point of view.
    # TODO(rchao): Revisit this as we design a fault-tolerance solution for
    # TPUStrategy.
    return False

  def _get_local_replica_id(self, replica_id_in_sync_group):
    # [FUNCTION DOC]
    # Returns the given replica id unchanged since all replicas are local.
    return replica_id_in_sync_group


def _make_axis_nonnegative(axis, rank):
  # [FUNCTION DOC]
  # HELPER: for the all-gather. Converts an axis to non-negative by adding rank when negative.
  # An integer axis is handled directly; otherwise (tensor axis) uses an element-wise select.
  # Convert a potentially negative `axis` to a non-negative one.
  if isinstance(axis, int):
    if axis >= 0:
      return axis
    else:
      return axis + rank
  else:
    return array_ops.where_v2(
        math_ops.greater_equal(axis, 0),
        axis,
        axis + rank)


# List of Tensor dtypes supported by cross_replica_sum().
_DTYPES_SUPPORTED_BY_CROSS_REPLICA_SUM = (
    dtypes.bfloat16,
    dtypes.float16,
    dtypes.float32,
    dtypes.float64,
    dtypes.int32,
    dtypes.uint32,
)


class _TPUReplicaContext(distribute_lib.ReplicaContext):
  # [CLASS DOC]
  # HELPER: replica context used inside TPU computations. Reports TPU core devices, forwards
  # logical device scoping to the extended strategy and implements all-gather.
  """Replication Context class for TPU Strategy."""

  # TODO(sourabhbajaj): Call for each replica should be updating this.
  # TODO(b/118385803): Always properly initialize replica_id.
  def __init__(self, strategy, replica_id_in_sync_group=0):
    # [FUNCTION DOC]
    # Constructor. Inputs: strategy and replica_id_in_sync_group (default zero, may be a tensor).
    # Initializes the base replica context. Returns nothing.
    distribute_lib.ReplicaContext.__init__(
        self, strategy, replica_id_in_sync_group=replica_id_in_sync_group)

  @property
  def devices(self):
    # [FUNCTION DOC]
    # Property. Requires a replica context (raises otherwise). If the replica id is a constant,
    # returns a one-tuple with that replica worker device; for a non-constant id (inside the TPU
    # replicate) returns a one-tuple holding TPU core zero.
    distribute_lib.require_replica_context(self)
    ds = self._strategy
    replica_id = tensor_util.constant_value(self.replica_id_in_sync_group)

    if replica_id is None:  # Non-constant `Tensor` inside `tpu.replicate`.
      # TODO(cjfj): Return other devices when model parallelism is supported.
      return (tpu.core(0),)
    else:
      return (ds.extended.worker_devices[replica_id],)

  def experimental_logical_device(self, logical_device_id):
    # [FUNCTION DOC]
    # Delegates to the extended strategy context manager that places variables and ops on the
    # given logical device id, and returns it.
    """Places variables and ops on the specified logical device."""
    return self.strategy.extended.experimental_logical_device(logical_device_id)

  def _compute_all_gather_output_shape(self, value_shape, value_rank, axis):
    # [FUNCTION DOC]
    # HELPER: for all-gather.
    # KNOWN ISSUE: the dynamic-rank branch multiplies by the replica count taken from the imported
    # eager context module, which has no such attribute, instead of this object replica count,
    # so that branch would fail when reached.
    # Inputs: value_shape, value_rank, axis. With integer rank, returns
    # the shape list with the axis dimension multiplied by the number of replicas; otherwise
    # returns a dynamic shape tensor with that axis scaled.
    if isinstance(value_rank, int):
      output_shape = list(value_shape)
      output_shape[axis] *= self.num_replicas_in_sync
    else:
      output_shape = array_ops.where_v2(
          math_ops.equal(math_ops.range(value_rank), axis),
          value_shape * context.num_replicas_in_sync,
          value_shape)
    return output_shape

  def all_gather(self, value, axis, experimental_hints=None):
    # [FUNCTION DOC]
    # Gathers values from all replicas, concatenated along axis. Inputs: value (nest of
    # tensors), axis (integer, may be negative), experimental_hints (ignored). Raises a
    # not-implemented error for sparse indexed slices. For each tensor, builds the gather by
    # placing the local value in its replica slot (a mask by one-hot replica id) and summing
    # across replicas when the dtype supports cross-replica sum, otherwise tiling and using
    # all-to-all then reordering by replica id. Returns the nest of gathered tensors.
    del experimental_hints
    for v in nest.flatten(value):
      if isinstance(v, indexed_slices.IndexedSlices):
        raise NotImplementedError("all_gather does not support IndexedSlices")

    def _all_gather_tensor(value, axis):
      # [FUNCTION DOC]
      # HELPER: all-gather of one tensor. Computes static or dynamic shape and rank, makes the
      # axis non-negative, and builds the gathered tensor by one of the two strategies described
      # in the outer function. Returns the gathered tensor reshaped to the output shape.
      value = ops.convert_to_tensor(value)

      # Compute the shape and rank and rank of the input tensor. Use static
      # shapes when possible to help with shape inference in graph mode, but
      # fall back on dynamic shapes when necessary.
      if value.shape.rank is None:
        value_rank = array_ops.rank(value)
        value_shape = array_ops.shape(value)
      else:
        value_rank = value.shape.rank
        value_shape = value.shape.as_list()
        value_shape_tensor = array_ops.shape(value)
        for i in range(len(value_shape)):
          if value_shape[i] is None:
            value_shape[i] = value_shape_tensor[i]

      # In the code below, we will insert a new "replica" dimension immediately
      # *before* `axis`. To ensure that it's inserted before and not after, we
      # must make `axis` non-negative.
      axis = _make_axis_nonnegative(axis, value_rank)

      # Create a list or 1D int Tensor such as
      #     [1, 1, ..., 1, num_replicas_in_sync, 1, ..., 1],
      # which is equal to `num_replicas_in_sync` at index `axis`
      # and is equal to 1 everywhere else.
      if isinstance(value_rank, int):
        replica_broadcast_shape = [1] * (value_rank + 1)
        replica_broadcast_shape[axis] = self.num_replicas_in_sync
      else:
        replica_broadcast_shape = array_ops.where_v2(
            math_ops.equal(math_ops.range(value_rank+1), axis),
            self.num_replicas_in_sync,
            1)

      output_shape = self._compute_all_gather_output_shape(
          value_shape, value_rank, axis)

      if value.dtype in _DTYPES_SUPPORTED_BY_CROSS_REPLICA_SUM:
        # optimized all_gather implementation based on cross_replica_sum().
        replica_id_mask = array_ops.one_hot(
            self.replica_id_in_sync_group, self.num_replicas_in_sync)
        replica_id_mask = array_ops.reshape(
            replica_id_mask, replica_broadcast_shape)
        replica_id_mask = math_ops.cast(replica_id_mask, value.dtype)

        gathered_value = array_ops.expand_dims(value, axis) * replica_id_mask
        gathered_value = self.all_reduce(
            reduce_util.ReduceOp.SUM, gathered_value)
        return array_ops.reshape(gathered_value, output_shape)
      else:
        # value.dtype isn't supported by cross_replica_sum(), so we fall back
        # on a less efficient implementation based on all_to_all().

        # The underlying AllToAllOp first do a split of the input value and then
        # cross-replica communication and concatenation of the result. So we
        # concatenate the local tensor here first.
        inputs = array_ops.expand_dims(value, axis=axis)
        inputs = array_ops.tile(inputs, replica_broadcast_shape)
        unordered_output = tpu_ops.all_to_all(
            inputs,
            concat_dimension=axis,
            split_dimension=axis,
            split_count=self.num_replicas_in_sync)

        # Re-order since xla.replica_id and ReplicaContext.replica_id mismatch.
        # Start by computing a permutation -- a 1D Tensor which maps
        #     tensor[xla.replica_id] = ReplicaContext.replica_id
        concat_replica_id = array_ops.reshape(
            self.replica_id_in_sync_group, [1])
        concat_replica_id = array_ops.tile(
            concat_replica_id, [self.num_replicas_in_sync])
        xla_to_replica_context_id = tpu_ops.all_to_all(
            concat_replica_id,
            concat_dimension=0,
            split_dimension=0,
            split_count=self.num_replicas_in_sync)

        # Now invert the mapping to get
        #    tensor[ReplicaContext.replica_id] = xla.replica_id
        replica_context_to_xla_id = math_ops.argmax(
            array_ops.one_hot(xla_to_replica_context_id,
                              self.num_replicas_in_sync),
            axis=0)

        # Reorder the output elements so that they're sorted based on
        # ReplicaContext.replica_id instead of xla.replica_id.
        sorted_with_extra_dim = array_ops.gather(
            unordered_output, replica_context_to_xla_id, axis=axis)
        return array_ops.reshape(sorted_with_extra_dim, output_shape)

    ys = [_all_gather_tensor(t, axis=axis) for t in nest.flatten(value)]
    return nest.pack_sequence_as(value, ys)


def _set_last_step_outputs(ctx, last_step_tensor_outputs):
  # [FUNCTION DOC]
  # HELPER: for the multi-step run. Inputs: ctx (multi-step context) and last_step_tensor_outputs
  # (nest of flat tensors). Repacks them to the original dict structure, then for each named
  # output stores a per-replica value when there is no reduce op, or its first element
  # otherwise, and writes the result back to the context. Returns nothing.
  """Sets the last step outputs on the given context."""
  # Convert replicate_outputs to the original dict structure of
  # last_step_outputs.
  last_step_tensor_outputs_dict = nest.pack_sequence_as(
      ctx.last_step_outputs, last_step_tensor_outputs)

  for name, reduce_op in ctx._last_step_outputs_reduce_ops.items():  # pylint: disable=protected-access
    output = last_step_tensor_outputs_dict[name]
    # For outputs that aren't reduced, return a PerReplica of all values. Else
    # take the first value from the list as each value should be the same.
    if reduce_op is None:
      last_step_tensor_outputs_dict[name] = values.PerReplica(output)
    else:
      # TODO(priyag): Should this return the element or a list with 1 element
      last_step_tensor_outputs_dict[name] = output[0]
  ctx._set_last_step_outputs(last_step_tensor_outputs_dict)  # pylint: disable=protected-access
