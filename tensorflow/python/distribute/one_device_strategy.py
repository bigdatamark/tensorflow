# [MODULE DOC]
# Implements the one-device distribution strategy: a strategy that places variables and computation
# on a single named device (and prefetches input to it), mainly useful for testing code against the
# strategy API. Defines the public V2 strategy class, the V1 variant, the extended implementation
# class holding all behavior, and a replica context with a single replica. Most public methods on
# the strategy classes are thin pass-through wrappers that exist to carry documentation.
# Import-time side effects: registers the public strategy names for export (v1 and v2 variants).

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
"""A tf.distribute.Strategy for running on a single device."""

from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import input_util
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute.v1 import input_lib as input_lib_v1
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export


# TODO(josh11b): Do we wrap values in types to generate errors if you are
# doing something that won't work with other DistributionStrategy
# implementations?


@tf_export("distribute.OneDeviceStrategy", v1=[])
class OneDeviceStrategy(distribute_lib.Strategy):
  # [CLASS DOC]
  # Public strategy that runs everything on one device. Wraps an extended implementation object
  # bound to the device; variables created in its scope, replica function calls and prefetched input
  # all use that device. There is exactly one replica, so reductions are trivial. Created with a
  # device string. Records itself in the strategy usage gauge on creation. Most methods only
  # delegate to the base strategy class.
  """A distribution strategy for running on a single device.

  Using this strategy will place any variables created in its scope on the
  specified device. Input distributed through this strategy will be
  prefetched to the specified device. Moreover, any functions called via
  `strategy.run` will also be placed on the specified device
  as well.

  Typical usage of this strategy could be testing your code with the
  tf.distribute.Strategy API before switching to other strategies which
  actually distribute to multiple devices/machines.

  For example:
  ```
  strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")

  with strategy.scope():
    v = tf.Variable(1.0)
    print(v.device)  # /job:localhost/replica:0/task:0/device:GPU:0

  def step_fn(x):
    return x * 2

  result = 0
  for i in range(10):
    result += strategy.run(step_fn, args=(i,))
  print(result)  # 90
  ```
  """

  def __init__(self, device):
    # [FUNCTION DOC]
    # Constructor. Input: device string naming where variables and computation go (for example a
    # CPU or GPU device name). Builds the extended implementation for the device, calls the base
    # constructor with it, and sets the V2 usage gauge cell to the strategy name (global metric
    # side effect). Returns nothing.
    """Creates a `OneDeviceStrategy`.

    Args:
      device: Device string identifier for the device on which the variables
        should be placed. See class docs for more details on how the device is
        used. Examples: "/cpu:0", "/gpu:0", "/device:CPU:0", "/device:GPU:0"
    """
    super(OneDeviceStrategy, self).__init__(OneDeviceExtended(self, device))
    distribute_lib.distribution_strategy_gauge.get_cell("V2").set(
        "OneDeviceStrategy")

  def experimental_distribute_dataset(self, dataset, options=None):  # pylint: disable=useless-super-delegation
    # [FUNCTION DOC]
    # Pass-through to the base strategy implementation (kept for documentation). Inputs: a dataset
    # and optional input options. Returns a distributed dataset prefetched to the device. See the
    # extended implementation for the actual work.
    """Distributes a tf.data.Dataset instance provided via dataset.

    In this case, there is only one device, so this is only a thin wrapper
    around the input dataset. It will, however, prefetch the input data to the
    specified device. The returned distributed dataset can be iterated over
    similar to how regular datasets can.

    NOTE: Currently, the user cannot add any more transformations to a
    distributed dataset.

    Example:
    ```
    strategy = tf.distribute.OneDeviceStrategy()
    dataset = tf.data.Dataset.range(10).batch(2)
    dist_dataset = strategy.experimental_distribute_dataset(dataset)
    for x in dist_dataset:
      print(x)  # [0, 1], [2, 3],...
    ```
    Args:
      dataset: `tf.data.Dataset` to be prefetched to device.
      options: `tf.distribute.InputOptions` used to control options on how this
        dataset is distributed.
    Returns:
      A "distributed `Dataset`" that the caller can iterate over.
    """
    return super(OneDeviceStrategy, self).experimental_distribute_dataset(
        dataset, options)

  def distribute_datasets_from_function(
      self,
      dataset_fn,  # pylint: disable=useless-super-delegation
      options=None):
    # [FUNCTION DOC]
    # Pass-through to the base strategy implementation. Inputs: a function taking an input context
    # and returning a dataset, and optional input options. The function is called once (one worker,
    # one device). Returns a distributed dataset.
    """Distributes `tf.data.Dataset` instances created by calls to `dataset_fn`.

    `dataset_fn` will be called once for each worker in the strategy. In this
    case, we only have one worker and one device so `dataset_fn` is called
    once.

    The `dataset_fn` should take an `tf.distribute.InputContext` instance where
    information about batching and input replication can be accessed:

    ```
    def dataset_fn(input_context):
      batch_size = input_context.get_per_replica_batch_size(global_batch_size)
      d = tf.data.Dataset.from_tensors([[1.]]).repeat().batch(batch_size)
      return d.shard(
          input_context.num_input_pipelines, input_context.input_pipeline_id)

    inputs = strategy.distribute_datasets_from_function(dataset_fn)

    for batch in inputs:
      replica_results = strategy.run(replica_fn, args=(batch,))
    ```

    IMPORTANT: The `tf.data.Dataset` returned by `dataset_fn` should have a
    per-replica batch size, unlike `experimental_distribute_dataset`, which uses
    the global batch size.  This may be computed using
    `input_context.get_per_replica_batch_size`.

    Args:
      dataset_fn: A function taking a `tf.distribute.InputContext` instance and
        returning a `tf.data.Dataset`.
      options: `tf.distribute.InputOptions` used to control options on how this
        dataset is distributed.

    Returns:
      A "distributed `Dataset`", which the caller can iterate over like regular
      datasets.
    """
    return super(OneDeviceStrategy,
                 self).distribute_datasets_from_function(dataset_fn, options)

  def experimental_local_results(self, value):  # pylint: disable=useless-super-delegation
    # [FUNCTION DOC]
    # Pass-through to the base strategy implementation. Input: a value from running a function or a
    # variable. Returns a tuple of the local values; with one device, a one-element tuple holding
    # the value.
    """Returns the list of all local per-replica values contained in `value`.

    In `OneDeviceStrategy`, the `value` is always expected to be a single
    value, so the result is just the value in a tuple.

    Args:
      value: A value returned by `experimental_run()`, `run()`,
        `extended.call_for_each_replica()`, or a variable created in `scope`.

    Returns:
      A tuple of values contained in `value`. If `value` represents a single
      value, this returns `(value,).`
    """
    return super(OneDeviceStrategy, self).experimental_local_results(value)

  def run(self, fn, args=(), kwargs=None, options=None):  # pylint: disable=useless-super-delegation
    # [FUNCTION DOC]
    # Pass-through to the base strategy implementation. Inputs: function to run, optional positional
    # arguments (default empty), optional keyword arguments (default none) and optional run
    # options. Calls the function once inside the device scope; returns its result.
    """Run `fn` on each replica, with the given arguments.

    In `OneDeviceStrategy`, `fn` is simply called within a device scope for the
    given device, with the provided arguments.

    Args:
      fn: The function to run. The output must be a `tf.nest` of `Tensor`s.
      args: (Optional) Positional arguments to `fn`.
      kwargs: (Optional) Keyword arguments to `fn`.
      options: (Optional) An instance of `tf.distribute.RunOptions` specifying
        the options to run `fn`.

    Returns:
      Return value from running `fn`.
    """
    return super(OneDeviceStrategy, self).run(fn, args, kwargs, options)

  def reduce(self, reduce_op, value, axis):  # pylint: disable=useless-super-delegation
    # [FUNCTION DOC]
    # Pass-through to the base strategy implementation. Inputs: reduction operation kind, a
    # per-replica value and an axis (or null). With a single replica, a null axis returns the value
    # unchanged; otherwise the value is reduced along that axis. Returns a tensor.
    """Reduce `value` across replicas.

    In `OneDeviceStrategy`, there is only one replica, so if axis=None, value
    is simply returned. If axis is specified as something other than None,
    such as axis=0, value is reduced along that axis and returned.

    Example:
    ```
    t = tf.range(10)

    result = strategy.reduce(tf.distribute.ReduceOp.SUM, t, axis=None).numpy()
    # result: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    result = strategy.reduce(tf.distribute.ReduceOp.SUM, t, axis=0).numpy()
    # result: 45
    ```

    Args:
      reduce_op: A `tf.distribute.ReduceOp` value specifying how values should
        be combined.
      value: A "per replica" value, e.g. returned by `run` to
        be combined into a single tensor.
      axis: Specifies the dimension to reduce along within each
        replica's tensor. Should typically be set to the batch dimension, or
        `None` to only reduce across replicas (e.g. if the tensor has no batch
        dimension).

    Returns:
      A `Tensor`.
    """
    return super(OneDeviceStrategy, self).reduce(reduce_op, value, axis)

  def scope(self):  # pylint: disable=useless-super-delegation
    # [FUNCTION DOC]
    # Pass-through to the base strategy implementation. Returns a context manager that makes this
    # strategy current, so variables created inside are placed on the strategy's device.
    """Returns a context manager selecting this Strategy as current.

    Inside a `with strategy.scope():` code block, this thread
    will use a variable creator set by `strategy`, and will
    enter its "cross-replica context".

    In `OneDeviceStrategy`, all variables created inside `strategy.scope()`
    will be on `device` specified at strategy construction time.
    See example in the docs for this class.

    Returns:
      A context manager to use for creating variables with this strategy.
    """
    return super(OneDeviceStrategy, self).scope()


@tf_export(v1=["distribute.OneDeviceStrategy"])  # pylint: disable=empty-docstring
class OneDeviceStrategyV1(distribute_lib.StrategyV1):
  # [CLASS DOC]
  # Version-1 API flavor of the one-device strategy, built on the V1 strategy base class. Behavior
  # is identical (same extended implementation); only the usage gauge label differs. Its class and
  # constructor documentation are copied from the V2 class at class creation, with the example
  # adjusted to enable eager execution first.

  __doc__ = (OneDeviceStrategy.__doc__ or "").replace(
      "For example:\n  ```",
      "For example:\n  ```\n  tf.enable_eager_execution()",
  )

  def __init__(self, device):
    # [FUNCTION DOC]
    # Constructor. Input: device string. Builds the extended implementation for the device, calls
    # the V1 base constructor, and sets the V1 usage gauge cell to the strategy name (global metric
    # side effect). Its documentation is assigned from the V2 constructor after the definition.
    super(OneDeviceStrategyV1, self).__init__(OneDeviceExtended(self, device))
    distribute_lib.distribution_strategy_gauge.get_cell("V1").set(
        "OneDeviceStrategy")
  __init__.__doc__ = OneDeviceStrategy.__init__.__doc__


# TODO(josh11b): Switch to V2 after callers have been updated to only V2 APIs.
class OneDeviceExtended(distribute_lib.StrategyExtendedV1):
  # [CLASS DOC]
  # Implementation object behind the one-device strategies. Holds the resolved full device name and
  # the host device used for input pipelines. Implements variable creation on the device, input
  # distribution, running functions in a device scope (with a single-replica context), reductions
  # (identity), updates, and the reporting properties (one replica, one worker device, always
  # checkpoint, no between-graph mode, etc.). Always treats the only replica as the whole cluster.
  """Implementation of OneDeviceStrategy."""

  def __init__(self, container_strategy, device):
    # [FUNCTION DOC]
    # Constructor. Inputs: the owning strategy and a device string. Resolves the device to its full
    # name and derives the host device of that device for input placement. Returns nothing.
    super(OneDeviceExtended, self).__init__(container_strategy)
    self._device = device_util.resolve(device)
    self._input_device = device_util.get_host_for_device(self._device)

  def _input_workers_with_options(self, options=None):
    # [FUNCTION DOC]
    # HELPER: builds the input workers description. Input: optional input options. If there are no
    # options or fetching to the device is enabled, the single input worker (host) feeds the compute
    # device; otherwise the data stays on the host device. Returns an input workers object.
    if not options or options.experimental_fetch_to_device:
      return input_lib.InputWorkers([(self._input_device, (self._device,))])
    else:
      return input_lib.InputWorkers([(self._input_device,
                                      (self._input_device,))])

  @property
  def _input_workers(self):
    # [FUNCTION DOC]
    # HELPER: property giving the input workers for default options (fetch to the compute device).
    return self._input_workers_with_options()

  def _create_variable(self, next_creator, **kwargs):
    # [FUNCTION DOC]
    # Variable creator hook. Inputs: the next creator in the chain and keyword arguments for the
    # variable, which may include a colocation target (removed from the arguments). With no target,
    # creates the variable on the strategy's device; with a single-device marker, on that marker's
    # device; with anything else, colocated with that target. Returns the created variable.
    colocate_with = kwargs.pop("colocate_with", None)
    if colocate_with is None:
      with ops.device(self._device):
        return next_creator(**kwargs)
    elif isinstance(colocate_with, numpy_dataset.SingleDevice):
      with ops.device(colocate_with.device):
        return next_creator(**kwargs)
    else:
      with ops.colocate_with(colocate_with):
        return next_creator(**kwargs)

  def _validate_colocate_with_variable(self, colocate_with_variable):
    # [FUNCTION DOC]
    # Checks that a variable to colocate with was created under this strategy, using the shared
    # validation utility (which raises an error if not). Returns nothing.
    distribute_utils.validate_colocate(colocate_with_variable, self)

  def _make_dataset_iterator(self, dataset):
    # [FUNCTION DOC]
    # Builds a V1 dataset iterator over the input workers for the dataset, without splitting
    # batches across replicas. Input: a dataset. Returns the iterator.
    """Make iterator from dataset without splitting the batch."""
    # Note that split_batch_by argument is not passed because it is always 1 in
    # this strategy, and adding it adds unnecessary overhead to the dataset.
    return input_lib_v1.DatasetIterator(dataset, self._input_workers,
                                        self._container_strategy())

  def _make_input_fn_iterator(
      self,
      input_fn,
      replication_mode=distribute_lib.InputReplicationMode.PER_WORKER):
    # [FUNCTION DOC]
    # Builds a V1 input-function iterator. Inputs: the input function and a replication mode
    # (default per worker). KNOWN ISSUE: the replication mode is accepted but ignored; a single
    # input context is always used. Returns the iterator.
    return input_lib_v1.InputFunctionIterator(input_fn, self._input_workers,
                                              [distribute_lib.InputContext()],
                                              self._container_strategy())

  def _experimental_make_numpy_dataset(self, numpy_input, session):
    # [FUNCTION DOC]
    # Creates a dataset from NumPy data. Inputs: array or nested arrays and a session (graph mode).
    # Variables holding the data are placed on the input (host) device. Returns the dataset.
    return numpy_dataset.one_host_numpy_dataset(
        numpy_input, numpy_dataset.SingleDevice(self._input_device), session)

  def _broadcast_to(self, tensor, destinations):
    # [FUNCTION DOC]
    # Broadcast is trivial with one device: ignores destinations and returns the tensor unchanged.
    del destinations
    return tensor

  def _experimental_distribute_dataset(self, dataset, options):
    # [FUNCTION DOC]
    # Distributes a dataset. Inputs: dataset and input options (may be null). Raises a
    # not-implemented error if the options request per-replica replication mode. Otherwise returns
    # a distributed dataset built on the input workers for the options, with one input context.
    # KNOWN ISSUE: the error message mentions a function name with a plural "datasets" that differs
    # from the actual public method name.
    # Note that split_batch_by argument is not passed because it is always 1 in
    # this strategy, and adding it adds unnecessary overhead to the dataset.
    if (options and options.experimental_replication_mode ==
        distribute_lib.InputReplicationMode.PER_REPLICA):
      raise NotImplementedError(
          "InputReplicationMode.PER_REPLICA "
          "is only supported in  "
          "`experimental_distribute_datasets_from_function`."
      )
    return input_util.get_distributed_dataset(
        dataset,
        self._input_workers_with_options(options),
        self._container_strategy(),
        options=options)

  def _distribute_datasets_from_function(self, dataset_fn, options):
    # [FUNCTION DOC]
    # Distributes datasets made by a function. Inputs: dataset function and input options (may be
    # null). Raises a not-implemented error if the options request per-replica replication mode
    # (the message says it is only supported by the mirrored strategy). Otherwise returns a
    # distributed dataset built by calling the function with one input context.
    if (options and options.experimental_replication_mode ==
        distribute_lib.InputReplicationMode.PER_REPLICA):
      raise NotImplementedError(
          "InputReplicationMode.PER_REPLICA "
          "is only supported in "
          "`experimental_distribute_datasets_from_function` "
          "of tf.distribute.MirroredStrategy")
    return input_util.get_distributed_datasets_from_function(
        dataset_fn,
        self._input_workers_with_options(options),
        [distribute_lib.InputContext()],
        self._container_strategy(),
        options=options)

  def _experimental_distribute_values_from_function(self, value_fn):
    # [FUNCTION DOC]
    # Calls the value function once with a new value context and returns its result as is (a plain
    # value, not a per-replica value; see the to-do about changing this).
    # TODO(b/137795644): This should return a PerReplica value but other
    # methods like run in OneDeviceStrategy need to be modified
    # to do the same.
    return value_fn(distribute_lib.ValueContext())

  # TODO(priyag): Deal with OutOfRange errors  once b/111349762 is fixed.
  def _experimental_run_steps_on_iterator(self, fn, iterator, iterations,
                                          initial_loop_values=None):
    # [FUNCTION DOC]
    # Graph-mode multi-step loop. Inputs: step function taking a multi-step context and the next
    # iterator element, an iterator, the number of iterations, and optional initial loop values
    # (default empty). Builds a while loop running the step function the given number of times
    # (serially, no gradients, no memory swapping), with the loop counter and last-step outputs as
    # loop variables. Temporarily records the outer control flow context on the object while the
    # loop is built. Stores the grouped loop operation as the run op and the final last-step
    # outputs, repacked into their original structure, in the context. Returns the context.
    if initial_loop_values is None:
      initial_loop_values = {}
    initial_loop_values = nest.flatten(initial_loop_values)

    ctx = input_lib.MultiStepContext()
    def body(i, *args):
      # [FUNCTION DOC]
      # HELPER: while loop body. Inputs: loop counter and ignored extra loop values. Runs the step
      # function with the context and next iterator element, then, after its result is computed,
      # returns the incremented counter followed by the flattened last-step outputs.
      """A wrapper around `fn` to create the while loop body."""
      del args
      fn_result = fn(ctx, iterator.get_next())
      flat_last_step_outputs = nest.flatten(ctx.last_step_outputs)
      with ops.control_dependencies([fn_result]):
        return [i + 1] + flat_last_step_outputs

    # We capture the control_flow_context at this point, before we run `fn`
    # inside a while_loop. This is useful in cases where we might need to exit
    # these contexts and get back to the outer context to do some things, for
    # e.g. create an op which should be evaluated only once at the end of the
    # loop on the host. One such usage is in creating metrics' value op.
    self._outer_control_flow_context = (
        ops.get_default_graph()._get_control_flow_context())  # pylint: disable=protected-access

    # TODO(priyag): Use max_iterations instead of an explicit counter.
    cond = lambda i, *args: i < iterations
    i = constant_op.constant(0)
    loop_result = while_loop.while_loop(
        cond,
        body, [i] + initial_loop_values,
        name="",
        parallel_iterations=1,
        back_prop=False,
        swap_memory=False,
        return_same_structure=True)
    del self._outer_control_flow_context

    ctx.run_op = control_flow_ops.group(loop_result)

    # Convert the last_step_outputs from a list to the original dict structure
    # of last_step_outputs.
    last_step_tensor_outputs = loop_result[1:]
    last_step_tensor_outputs_dict = nest.pack_sequence_as(
        ctx.last_step_outputs, last_step_tensor_outputs)

    ctx._set_last_step_outputs(last_step_tensor_outputs_dict)  # pylint: disable=protected-access
    return ctx

  def _call_for_each_replica(self, fn, args, kwargs):
    # [FUNCTION DOC]
    # Runs the function once, on the strategy's device, inside a single-replica context.
    # Inputs: function, positional arguments sequence, keyword arguments mapping. Returns the
    # function's result.
    strategy = self._container_strategy()
    with ops.device(self._device), _OneDeviceReplicaContext(strategy):
      return fn(*args, **kwargs)

  def _reduce_to(self, reduce_op, value, destinations, options):
    # [FUNCTION DOC]
    # Reduction across replicas is trivial here: ignores operation, destinations and options and
    # returns the value unchanged.
    del reduce_op, destinations, options
    return value

  def _gather_to_implementation(self, value, destinations, axis, options):
    # [FUNCTION DOC]
    # Gathering across replicas is trivial here: ignores destinations, axis and options and returns
    # the value unchanged.
    del destinations, axis, options
    return value

  def _update(self, var, fn, args, kwargs, group):
    # [FUNCTION DOC]
    # Applies an update function to a variable. Inputs: the variable, function, positional
    # arguments, keyword arguments and a group flag. Delegates to the non-slot update with the
    # variable prepended to the arguments and as the colocation target. Returns the result (grouped
    # or as local results, depending on the flag).
    # The implementations of _update() and _update_non_slot() are identical
    # except _update() passes `var` as the first argument to `fn()`.
    return self._update_non_slot(var, fn, (var,) + tuple(args), kwargs, group)

  def _update_non_slot(self, colocate_with, fn, args, kwargs, group):
    # [FUNCTION DOC]
    # Runs an update function on the device inside an update context. Inputs: colocation target
    # (ignored), function, positional arguments, keyword arguments, group flag. If group is true
    # returns the function's result directly; otherwise maps the local-results conversion over the
    # result structure (each leaf becomes a one-element tuple).
    del colocate_with
    with ops.device(self._device), distribute_lib.UpdateContext(self._device):
      result = fn(*args, **kwargs)
      if group:
        return result
      else:
        return nest.map_structure(self._local_results, result)

  def read_var(self, replica_local_var):
    # [FUNCTION DOC]
    # Reads a replica-local variable: with one replica, returns an identity copy of its value.
    """Read the aggregate value of a replica-local variable."""
    return array_ops.identity(replica_local_var)

  def _local_results(self, value):
    # [FUNCTION DOC]
    # Returns the value wrapped in a one-element tuple (the only local result).
    return (value,)

  def value_container(self, value):
    # [FUNCTION DOC]
    # Returns the value itself; there are no mirrored containers in this strategy.
    return value

  def _in_multi_worker_mode(self):
    # [FUNCTION DOC]
    # Always false: this strategy never runs in a multi-worker setting.
    """Whether this strategy indicates working in multi-worker settings."""
    return False

  @property
  def _num_replicas_in_sync(self):
    # [FUNCTION DOC]
    # Property: the number of replicas, always one.
    return 1

  @property
  def worker_devices(self):
    # [FUNCTION DOC]
    # Property: tuple holding only the strategy's device.
    return (self._device,)

  @property
  def parameter_devices(self):
    # [FUNCTION DOC]
    # Property: tuple holding only the strategy's device (variables live there).
    return (self._device,)

  def non_slot_devices(self, var_list):
    # [FUNCTION DOC]
    # Devices for non-slot variables. Input: variable list, ignored. Returns a tuple holding only
    # the strategy's device.
    del var_list
    return (self._device,)

  @property
  def experimental_should_init(self):
    # [FUNCTION DOC]
    # Property: always true; this worker should initialize variables.
    return True

  @property
  def experimental_between_graph(self):
    # [FUNCTION DOC]
    # Property: always false; no between-graph replication.
    return False

  @property
  def should_checkpoint(self):
    # [FUNCTION DOC]
    # Property: always true; the single worker saves checkpoints.
    return True

  @property
  def should_save_summary(self):
    # [FUNCTION DOC]
    # Property: always true; the single worker writes summaries.
    return True

  # TODO(priyag): Delete this once all strategies use global batch size.
  @property
  def _global_batch_size(self):
    # [FUNCTION DOC]
    # Property: always true, meaning batch sizes are global (equal to per-replica here).
    """Global and per-replica batching are equivalent for OneDeviceStrategy."""
    return True

  @property
  def _support_per_replica_values(self):
    # [FUNCTION DOC]
    # Property: always false; per-replica values are not supported by this strategy.
    return False

  def _get_local_replica_id(self, replica_id_in_sync_group):
    # [FUNCTION DOC]
    # Maps a replica id within the sync group to a local id; with one replica it is the same value.
    return replica_id_in_sync_group


class _OneDeviceReplicaContext(distribute_lib.ReplicaContext):
  # [CLASS DOC]
  # HELPER: replica context used while running functions in the one-device strategy. It has a
  # single replica with id zero.
  """ReplicaContext for OneDeviceStrategy."""

  def __init__(self, strategy):
    # [FUNCTION DOC]
    # Constructor. Input: the owning strategy. Initializes the base context with replica id zero.
    distribute_lib.ReplicaContext.__init__(
        self, strategy, replica_id_in_sync_group=0)

  @property
  def devices(self):
    # [FUNCTION DOC]
    # Property: the worker devices of the strategy's extended implementation (just one device).
    return self._strategy.extended.worker_devices
