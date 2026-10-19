# [MODULE DOC]
# Implements the parameter server tf.distribute strategy (the V1, Estimator-only variant) and its
# extended class, which is also the base for central storage strategies. Variables are placed on
# parameter servers (round-robin through a device setter) or on a single local device, while
# operations run on workers or local compute devices. Exposes ParameterServerStrategyV1 (exported
# under the experimental distribute namespace of the V1 API) and ParameterServerStrategyExtended.
# Defines a module constant for the local CPU device. No import-time registration other than the
# export decorator.
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
"""Class implementing a multi-worker parameter server tf.distribute strategy."""

import copy


from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import input_util
from tensorflow.python.distribute import mirrored_run
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute import ps_values
from tensorflow.python.distribute import values
from tensorflow.python.distribute.cluster_resolver import cluster_resolver as cluster_resolver_lib
from tensorflow.python.distribute.cluster_resolver import tfconfig_cluster_resolver
from tensorflow.python.distribute.v1 import input_lib as input_lib_v1
from tensorflow.python.eager import context
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import device_setter
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export

_LOCAL_CPU = "/device:CPU:0"


@tf_export(v1=["distribute.experimental.ParameterServerStrategy"])  # pylint: disable=missing-docstring
class ParameterServerStrategyV1(distribute_lib.StrategyV1):
  # [CLASS DOC]
  # Public V1 strategy for asynchronous multi-worker parameter server training with the Estimator
  # API. Thin wrapper over the base V1 strategy that builds a ParameterServerStrategyExtended
  # from a cluster resolver and blocks eager-mode use of dataset distribution, run and scope by
  # raising a not-implemented error. Reports itself to the strategy usage gauge on creation.
  """An asynchronous multi-worker parameter server tf.distribute strategy.

  This strategy requires two roles: workers and parameter servers. Variables and
  updates to those variables will be assigned to parameter servers and other
  operations are assigned to workers.

  When each worker has more than one GPU, operations will be replicated on all
  GPUs. Even though operations may be replicated, variables are not and each
  worker shares a common view for which parameter server a variable is assigned
  to.

  By default it uses `TFConfigClusterResolver` to detect configurations for
  multi-worker training. This requires a 'TF_CONFIG' environment variable and
  the 'TF_CONFIG' must have a cluster spec.

  This class assumes each worker is running the same code independently, but
  parameter servers are running a standard server. This means that while each
  worker will synchronously compute a single gradient update across all GPUs,
  updates between workers proceed asynchronously. Operations that occur only on
  the first replica (such as incrementing the global step), will occur on the
  first replica *of every worker*.

  It is expected to call `call_for_each_replica(fn, ...)` for any
  operations which potentially can be replicated across replicas (i.e. multiple
  GPUs) even if there is only CPU or one GPU. When defining the `fn`, extra
  caution needs to be taken:

  1) It is generally not recommended to open a device scope under the strategy's
  scope. A device scope (i.e. calling `tf.device`) will be merged with or
  override the device for operations but will not change the device for
  variables.

  2) It is also not recommended to open a colocation scope (i.e. calling
  `tf.compat.v1.colocate_with`) under the strategy's scope. For colocating
  variables, use `strategy.extended.colocate_vars_with` instead. Colocation of
  ops will possibly create device assignment conflicts.

  Note: This strategy only works with the Estimator API. Pass an instance of
  this strategy to the `experimental_distribute` argument when you create the
  `RunConfig`. This instance of `RunConfig` should then be passed to the
  `Estimator` instance on which `train_and_evaluate` is called.

  For Example:
  ```
  strategy = tf.distribute.experimental.ParameterServerStrategy()
  run_config = tf.estimator.RunConfig(
      experimental_distribute.train_distribute=strategy)
  estimator = tf.estimator.Estimator(config=run_config)
  tf.estimator.train_and_evaluate(estimator,...)
  ```
  """

  def __init__(self, cluster_resolver=None):
    # [FUNCTION DOC]
    # Creates the strategy. cluster_resolver is an optional cluster resolver object; when null,
    # a resolver reading the cluster configuration environment variable is created. Builds the
    # extended strategy with it, initializes the base class, and records the strategy name in the
    # V1 usage gauge (side effect on global metrics). Returns nothing.
    """Initializes this strategy with an optional `cluster_resolver`.

    Args:
      cluster_resolver: Optional
        `tf.distribute.cluster_resolver.ClusterResolver` object. Defaults to a
        `tf.distribute.cluster_resolver.TFConfigClusterResolver`.
    """
    if cluster_resolver is None:
      cluster_resolver = tfconfig_cluster_resolver.TFConfigClusterResolver()
    super(ParameterServerStrategyV1, self).__init__(
        ParameterServerStrategyExtended(
            self, cluster_resolver=cluster_resolver))
    distribute_lib.distribution_strategy_gauge.get_cell("V1").set(
        "ParameterServerStrategy")

  def experimental_distribute_dataset(self, dataset, options=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the result of the base class call is discarded, so this returns nothing
    # instead of the distributed dataset.
    # Distributes a dataset. Raises a not-implemented error if options request the per-replica
    # input replication mode, or if executing eagerly; otherwise delegates to the base class.
    if (options and options.experimental_replication_mode ==
        distribute_lib.InputReplicationMode.PER_REPLICA):
      raise NotImplementedError(
          "InputReplicationMode.PER_REPLICA "
          "is only supported in "
          "`experimental_distribute_datasets_from_function`."
      )
    self._raise_pss_error_if_eager()
    super(ParameterServerStrategyV1,
          self).experimental_distribute_dataset(dataset=dataset,
                                                options=options)

  def distribute_datasets_from_function(self, dataset_fn, options=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the result of the base class call is discarded, so this returns nothing
    # instead of the distributed dataset.
    # Builds distributed datasets from a dataset-producing callback. Raises a not-implemented
    # error for the per-replica replication mode in options, or when executing eagerly; otherwise
    # delegates to the base class.
    if (options and options.experimental_replication_mode ==
        distribute_lib.InputReplicationMode.PER_REPLICA):
      raise NotImplementedError(
          "InputReplicationMode.PER_REPLICA "
          "is only supported in "
          "`experimental_distribute_datasets_from_function` "
          "of tf.distribute.MirroredStrategy")
    self._raise_pss_error_if_eager()
    super(ParameterServerStrategyV1, self).distribute_datasets_from_function(
        dataset_fn=dataset_fn, options=options)

  def run(self, fn, args=(), kwargs=None, options=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the result of the base class run is discarded, so this returns nothing.
    # Runs fn on the replicas with the given positional arguments, keyword arguments and options.
    # Raises a not-implemented error when executing eagerly; otherwise delegates to the base class.
    self._raise_pss_error_if_eager()
    super(ParameterServerStrategyV1, self).run(
        fn, args=args, kwargs=kwargs, options=options)

  def scope(self):
    # [FUNCTION DOC]
    # Returns the strategy scope context manager from the base class. Raises a not-implemented
    # error when executing eagerly.
    self._raise_pss_error_if_eager()
    return super(ParameterServerStrategyV1, self).scope()

  def _raise_pss_error_if_eager(self):
    # [FUNCTION DOC]
    # HELPER: guard used by the public methods of this strategy.
    # Raises a not-implemented error saying the strategy only works with the Estimator API when
    # executing eagerly; otherwise does nothing.
    if context.executing_eagerly():
      raise NotImplementedError(
          "`tf.compat.v1.distribute.experimental.ParameterServerStrategy` "
          "currently only works with the tf.Estimator API")


# TODO(josh11b): Switch to V2 when we no longer need to support tf.compat.v1.
class ParameterServerStrategyExtended(distribute_lib.StrategyExtendedV1):
  # [CLASS DOC]
  # Implementation of the parameter server strategy (also used for central storage). Holds the
  # worker device, compute devices (one per replica), variable device or device function,
  # parameter devices, chief flag, cluster spec, task type and id, the input host device and the
  # cross-device reduction helper (reduce to the local CPU). Supports a multi-worker mode (with
  # ps jobs) and a local single-machine mode. Handles variable creation (optionally wrapped in
  # aggregating variables), dataset distribution, broadcast/reduce/gather, and updates.
  """Implementation of ParameterServerStrategy and CentralStorageStrategy."""

  def __init__(self,
               container_strategy,
               cluster_resolver=None,
               compute_devices=None,
               parameter_device=None):
    # [FUNCTION DOC]
    # Creates the extended strategy. container_strategy is the owning strategy object;
    # cluster_resolver, compute_devices (list of device names) and parameter_device (device name
    # or device function) are optional. Initializes the base class, sets up devices through the
    # strategy initialization routine, and installs a cross-device reduction that reduces to the
    # local CPU (all-reduce is typically not needed). Returns nothing.
    super(ParameterServerStrategyExtended, self).__init__(container_strategy)
    self._initialize_strategy(
        cluster_resolver=cluster_resolver,
        compute_devices=compute_devices,
        parameter_device=parameter_device)

    # We typically don't need to do all-reduce in this strategy.
    self._cross_device_ops = (
        cross_device_ops_lib.ReductionToOneDevice(reduce_to_device=_LOCAL_CPU))

  def _initialize_strategy(self,
                           cluster_resolver=None,
                           compute_devices=None,
                           parameter_device=None):
    # [FUNCTION DOC]
    # HELPER: dispatches device setup for the constructor.
    # If a cluster resolver is given and its cluster spec is non-empty, sets up multi-worker mode
    # from the resolver; otherwise sets up local mode using the given compute devices, parameter
    # device and resolver (any may be null). Returns nothing.
    if cluster_resolver and cluster_resolver.cluster_spec():
      self._initialize_multi_worker(cluster_resolver)
    else:
      self._initialize_local(
          compute_devices, parameter_device, cluster_resolver=cluster_resolver)

  def _initialize_multi_worker(self, cluster_resolver):
    # [FUNCTION DOC]
    # HELPER: multi-worker device setup.
    # Determines the GPU count (from the context for the environment-based resolver, otherwise
    # from the resolver's accelerator counts) and remembers it. Requires task type and task id
    # from the resolver, else raises a value error; normalizes the cluster spec. Sets the worker
    # device from task type and id, the input host device, one canonical compute device per GPU
    # (or the worker device if none), a replica device setter placing variables on ps tasks
    # round-robin (raises a value error if there are no ps jobs), the list of ps parameter
    # devices, the default device, the chief flag, cluster spec, task type and task id. Logs the
    # configuration. Returns nothing.
    """Initialize devices for multiple workers.

    It creates variable devices and compute devices. Variables and operations
    will be assigned to them respectively. We have one compute device per
    replica. The variable device is a device function or device string. The
    default variable device assigns variables to parameter servers in a
    round-robin fashion.

    Args:
      cluster_resolver: a descendant of `ClusterResolver` object.

    Raises:
      ValueError: if the cluster doesn't have ps jobs.
    """
    # TODO(b/126786766): TFConfigClusterResolver returns wrong number of GPUs in
    # some cases.
    if isinstance(
        cluster_resolver, tfconfig_cluster_resolver.TFConfigClusterResolver):
      num_gpus = context.num_gpus()
    else:
      num_gpus = cluster_resolver.num_accelerators().get("GPU", 0)

    # Save the num_gpus_per_worker for configure method.
    self._num_gpus_per_worker = num_gpus

    cluster_spec = cluster_resolver.cluster_spec()
    task_type = cluster_resolver.task_type
    task_id = cluster_resolver.task_id
    if not task_type or task_id is None:
      raise ValueError("When `cluster_spec` is given, you must also specify "
                       "`task_type` and `task_id`")
    cluster_spec = multi_worker_util.normalize_cluster_spec(cluster_spec)
    assert cluster_spec.as_dict()

    self._worker_device = "/job:%s/task:%d" % (task_type, task_id)
    self._input_host_device = numpy_dataset.SingleDevice(self._worker_device)

    # Define compute devices which is a list of device strings and one for each
    # replica. When there are GPUs, replicate operations on these GPUs.
    # Otherwise, place operations on CPU.
    if num_gpus > 0:
      compute_devices = tuple(
          "%s/device:GPU:%d" % (self._worker_device, i)
          for i in range(num_gpus))
    else:
      compute_devices = (self._worker_device,)

    self._compute_devices = [
        device_util.canonicalize(d) for d in compute_devices]

    # In distributed mode, place variables on ps jobs in a round-robin fashion.
    # Note that devices returned from `replica_device_setter` are not
    # canonical and therefore we don't canonicalize all variable devices to
    # make them consistent.
    # TODO(yuefengz): support passing a strategy object to control variable
    # assignment.
    # TODO(yuefengz): merge the logic of replica_device_setter into this
    # class.
    num_ps_replicas = len(cluster_spec.as_dict().get("ps", []))
    if num_ps_replicas == 0:
      raise ValueError("The cluster spec needs to have `ps` jobs.")
    self._variable_device = device_setter.replica_device_setter(
        ps_tasks=num_ps_replicas,
        worker_device=self._worker_device,
        merge_devices=True,
        cluster=cluster_spec)

    # The `_parameter_devices` is needed for the `parameter_devices` property
    # and is a list of all variable devices. Here parameter devices are all
    # tasks of the "ps" job.
    self._parameter_devices = tuple(map("/job:ps/task:{}".format,
                                        range(num_ps_replicas)))

    # Add a default device so that ops without specified devices will not end up
    # on other workers.
    self._default_device = self._worker_device

    self._is_chief = multi_worker_util.is_chief(cluster_spec, task_type,
                                                task_id)
    self._cluster_spec = cluster_spec
    self._task_type = task_type
    self._task_id = task_id

    logging.info(
        "Multi-worker ParameterServerStrategy with "
        "cluster_spec = %r, task_type = %r, task_id = %r, "
        "num_ps_replicas = %r, is_chief = %r, compute_devices = %r, "
        "variable_device = %r", cluster_spec.as_dict(), task_type, task_id,
        num_ps_replicas, self._is_chief, self._compute_devices,
        self._variable_device)

  # TODO(yuefengz): get rid of cluster_resolver argument when contrib's
  # version no longer depends on this class.
  def _initialize_local(self,
                        compute_devices,
                        parameter_device,
                        cluster_resolver=None):
    # [FUNCTION DOC]
    # HELPER: local (single machine) device setup.
    # Sets the worker device to the local CPU and the input host device. When compute_devices is
    # null, determines the GPU count (from the context, or from the resolver if given), remembers
    # it, and derives local devices. Canonicalizes compute devices. When parameter_device is null,
    # uses the single compute device if there is exactly one, else the local CPU. Records variable
    # device, compute devices, parameter devices, marks this as chief, clears cluster spec, task
    # type and id, and logs. Returns nothing.
    """Initialize local devices for training."""
    self._worker_device = device_util.canonicalize("/device:CPU:0")
    self._input_host_device = numpy_dataset.SingleDevice(self._worker_device)

    if compute_devices is None:
      if not cluster_resolver:
        num_gpus = context.num_gpus()
      else:
        num_gpus = cluster_resolver.num_accelerators().get("GPU", 0)
      # Save the num_gpus_per_worker for configure method which is used by the
      # contrib version.
      self._num_gpus_per_worker = num_gpus

      compute_devices = device_util.local_devices_from_num_gpus(num_gpus)

    compute_devices = [device_util.canonicalize(d) for d in compute_devices]

    if parameter_device is None:
      # If there is only one GPU, put everything on that GPU. Otherwise, place
      # variables on CPU.
      if len(compute_devices) == 1:
        parameter_device = compute_devices[0]
      else:
        parameter_device = _LOCAL_CPU

    self._variable_device = parameter_device
    self._compute_devices = compute_devices
    self._parameter_devices = (parameter_device,)
    self._is_chief = True
    self._cluster_spec = None
    self._task_type = None
    self._task_id = None

    logging.info(
        "ParameterServerStrategy (CentralStorageStrategy if you are using a "
        "single machine) with compute_devices = %r, variable_device = %r",
        compute_devices, self._variable_device)

  def _input_workers_with_options(self, options=None):
    # [FUNCTION DOC]
    # Builds the input workers description: one worker (the worker device) with its compute
    # devices. If options are null or request fetching to device, input goes to the compute
    # devices; otherwise every entry is the worker device itself (host). Returns the input workers.
    if not options or options.experimental_fetch_to_device:
      return input_lib.InputWorkers(
          [(self._worker_device, self._compute_devices)])
    else:
      return input_lib.InputWorkers(
          [(self._worker_device,
            (self._worker_device,) * len(self._compute_devices))])

  @property
  def _input_workers(self):
    # [FUNCTION DOC]
    # Returns the input workers built with default (null) options.
    return self._input_workers_with_options()

  def _validate_colocate_with_variable(self, colocate_with_variable):
    # [FUNCTION DOC]
    # Validates that the given variable may be colocated with under this strategy, via the shared
    # validation utility; raises if invalid. Returns nothing.
    distribute_utils.validate_colocate(colocate_with_variable, self)

  def _experimental_distribute_dataset(self, dataset, options):
    # [FUNCTION DOC]
    # Returns a distributed dataset created from the given dataset using the input workers for
    # the options, the container strategy and the number of replicas in sync.
    return input_util.get_distributed_dataset(
        dataset,
        self._input_workers_with_options(options),
        self._container_strategy(),
        num_replicas_in_sync=self._num_replicas_in_sync,
        options=options)

  def _make_dataset_iterator(self, dataset):
    # [FUNCTION DOC]
    # Returns a V1 dataset iterator over the dataset spanning the input workers, using the
    # container strategy and the number of replicas in sync.
    return input_lib_v1.DatasetIterator(
        dataset,
        self._input_workers,
        self._container_strategy(),
        num_replicas_in_sync=self._num_replicas_in_sync)

  def _make_input_fn_iterator(
      self,
      input_fn,
      replication_mode=distribute_lib.InputReplicationMode.PER_WORKER):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the replication_mode argument is accepted but never used.
    # Creates a V1 input-function iterator. input_fn is a callback receiving an input context.
    # Builds one input context: in multi-worker mode, pipeline id and count come from this task's
    # position among workers; otherwise one pipeline with id zero. Returns the iterator.
    """Distributes the dataset to each local GPU."""
    if self._cluster_spec:
      input_pipeline_id = multi_worker_util.id_in_cluster(
          self._cluster_spec, self._task_type, self._task_id)
      num_input_pipelines = multi_worker_util.worker_count(
          self._cluster_spec, self._task_type)
    else:
      input_pipeline_id = 0
      num_input_pipelines = 1
    input_context = distribute_lib.InputContext(
        num_input_pipelines=num_input_pipelines,
        input_pipeline_id=input_pipeline_id,
        num_replicas_in_sync=self._num_replicas_in_sync)
    return input_lib_v1.InputFunctionIterator(input_fn, self._input_workers,
                                              [input_context],
                                              self._container_strategy())

  def _experimental_make_numpy_dataset(self, numpy_input, session):
    # [FUNCTION DOC]
    # Creates a dataset from in-memory numpy input on the input host device, using the given
    # session for initialization. Returns the dataset.
    return numpy_dataset.one_host_numpy_dataset(
        numpy_input, self._input_host_device, session)

  def _distribute_datasets_from_function(self, dataset_fn, options):
    # [FUNCTION DOC]
    # Creates distributed datasets from a dataset callback. Builds a single input context (pipeline
    # id and count from the cluster position in multi-worker mode, else one pipeline with id zero)
    # and delegates to the input utility with the input workers for the options. Returns the
    # distributed datasets.
    if self._cluster_spec:
      input_pipeline_id = multi_worker_util.id_in_cluster(
          self._cluster_spec, self._task_type, self._task_id)
      num_input_pipelines = multi_worker_util.worker_count(
          self._cluster_spec, self._task_type)
    else:
      input_pipeline_id = 0
      num_input_pipelines = 1

    input_context = distribute_lib.InputContext(
        num_input_pipelines=num_input_pipelines,
        input_pipeline_id=input_pipeline_id,
        num_replicas_in_sync=self._num_replicas_in_sync)

    return input_util.get_distributed_datasets_from_function(
        dataset_fn,
        self._input_workers_with_options(options), [input_context],
        self._container_strategy(),
        options=options)

  def _experimental_distribute_values_from_function(self, value_fn):
    # [FUNCTION DOC]
    # Calls value_fn once per replica with a value context (replica id, replica count) and
    # regroups the per-replica results into a distributed value, always wrapping. Returns it.
    per_replica_values = []
    for replica_id in range(self._num_replicas_in_sync):
      per_replica_values.append(
          value_fn(distribute_lib.ValueContext(replica_id,
                                               self._num_replicas_in_sync)))
    return distribute_utils.regroup(per_replica_values, always_wrap=True)

  def _broadcast_to(self, tensor, destinations):
    # [FUNCTION DOC]
    # Broadcasts a value to destinations. Python floats and integers are returned unchanged (fast
    # path that delays dtype choice). If destinations are not valid, defaults to all compute
    # devices. Otherwise delegates to the cross-device broadcast. Returns the broadcast value.
    # This is both a fast path for Python constants, and a way to delay
    # converting Python values to a tensor until we know what type it
    # should be converted to. Otherwise we have trouble with:
    #   global_step.assign_add(1)
    # since the `1` gets broadcast as an int32 but global_step is int64.
    if isinstance(tensor, (float, int)):
      return tensor
    if not cross_device_ops_lib.check_destinations(destinations):
      # TODO(josh11b): Use current logical device instead of 0 here.
      destinations = self._compute_devices
    return self._cross_device_ops.broadcast(tensor, destinations)

  def _allow_variable_partition(self):
    # [FUNCTION DOC]
    # Returns true when variable partitioning is allowed, i.e. when not executing eagerly.
    return not context.executing_eagerly()

  def _create_var_creator(self, next_creator, **kwargs):
    # [FUNCTION DOC]
    # HELPER: builds the variable creation callback used by variable creation.
    # KNOWN ISSUE: the error message concatenates the aggregation value (an enum, not text) and
    # may itself fail with a type error instead of raising the intended value error.
    # If there is more than one replica, reads the aggregation mode (default none) from keyword
    # arguments, validates it is none, sum, mean or only-first-replica (else raises a value
    # error), and returns a creator that wraps each new variable in an aggregating variable and
    # fixes up graph collections (removes the raw trainable variable, adds the wrapper; handles
    # the global step in eager mode). With a single replica returns next_creator unchanged.
    if self._num_replicas_in_sync > 1:
      aggregation = kwargs.pop("aggregation", vs.VariableAggregation.NONE)
      if aggregation not in (
          vs.VariableAggregation.NONE,
          vs.VariableAggregation.SUM,
          vs.VariableAggregation.MEAN,
          vs.VariableAggregation.ONLY_FIRST_REPLICA
      ):
        raise ValueError("Invalid variable aggregation mode: " + aggregation +
                         " for variable: " + kwargs["name"])

      def var_creator(**kwargs):
        # [FUNCTION DOC]
        # Nested creator: calls the next creator with collections cleared, wraps the result in an
        # aggregating variable, and registers the wrapper in the requested collections (graph mode:
        # replacing the trainable variable entry; eager: only the global step collection).
        # Returns the wrapper.
        """Create an AggregatingVariable and fix up collections."""
        # Record what collections this variable should be added to.
        collections = kwargs.pop("collections", None)
        if collections is None:
          collections = [ops.GraphKeys.GLOBAL_VARIABLES]
        kwargs["collections"] = []

        # Create and wrap the variable.
        v = next_creator(**kwargs)
        wrapped = ps_values.AggregatingVariable(self._container_strategy(), v,
                                                aggregation)

        # Add the wrapped variable to the requested collections.
        # The handling of eager mode and the global step matches
        # ResourceVariable._init_from_args().
        if not context.executing_eagerly():
          g = ops.get_default_graph()
          # If "trainable" is True, next_creator() will add the contained
          # variable to the TRAINABLE_VARIABLES collection, so we manually
          # remove it and replace with the wrapper. We can't set "trainable"
          # to False for next_creator() since that causes functions like
          # implicit_gradients to skip those variables.
          if kwargs.get("trainable", True):
            collections.append(ops.GraphKeys.TRAINABLE_VARIABLES)
            l = g.get_collection_ref(ops.GraphKeys.TRAINABLE_VARIABLES)
            if v in l:
              l.remove(v)
          g.add_to_collections(collections, wrapped)
        elif ops.GraphKeys.GLOBAL_STEP in collections:
          ops.add_to_collections(ops.GraphKeys.GLOBAL_STEP, wrapped)

        return wrapped
      return var_creator
    else:
      return next_creator

  # TODO(yuefengz): Not all ops in device_setter.STANDARD_PS_OPS will go through
  # this creator, such as "MutableHashTable".
  def _create_variable(self, next_creator, **kwargs):
    # [FUNCTION DOC]
    # Creates a variable using the creator built for the given next creator and keyword arguments.
    # If a colocation target is supplied: for a single-device placeholder, creates under that
    # device; otherwise clears the device and colocates with the target. With no target, creates
    # with colocation cleared under the strategy's variable device (parameter server placement).
    # Returns the created (possibly aggregating-wrapped) variable.
    var_creator = self._create_var_creator(next_creator, **kwargs)

    if "colocate_with" in kwargs:
      colocate_with = kwargs["colocate_with"]
      if isinstance(colocate_with, numpy_dataset.SingleDevice):
        with ops.device(colocate_with.device):
          return var_creator(**kwargs)
      with ops.device(None):
        with ops.colocate_with(colocate_with):
          return var_creator(**kwargs)

    with ops.colocate_with(None, ignore_existing=True):
      with ops.device(self._variable_device):
        return var_creator(**kwargs)

  def _call_for_each_replica(self, fn, args, kwargs):
    # [FUNCTION DOC]
    # Runs fn with the given positional and keyword arguments once per replica through the
    # mirrored-run machinery and returns the per-replica results.
    return mirrored_run.call_for_each_replica(self._container_strategy(), fn,
                                              args, kwargs)

  def _verify_destinations_not_different_worker(self, destinations):
    # [FUNCTION DOC]
    # HELPER: validation for reduce and gather operations.
    # Does nothing in local mode or when destinations is null. Otherwise raises a value error if
    # any destination device belongs to the same job type as this task but a different task.
    if not self._cluster_spec:
      return
    if destinations is None:
      return
    for d in cross_device_ops_lib.get_devices_from(destinations):
      d_spec = tf_device.DeviceSpec.from_string(d)
      if d_spec.job == self._task_type and d_spec.task != self._task_id:
        raise ValueError(
            "Cannot reduce to another worker: %r, current worker is %r" %
            (d, self._worker_device))

  def _gather_to_implementation(self, value, destinations, axis,
                                options):
    # [FUNCTION DOC]
    # Gathers value to destinations along axis with options. Verifies destinations are not on a
    # different worker (raises a value error otherwise). Non-distributed values are returned
    # unchanged; distributed values go through the cross-device gather.
    self._verify_destinations_not_different_worker(destinations)
    if not isinstance(value, values.DistributedValues):
      return value
    return self._cross_device_ops._gather(  # pylint: disable=protected-access
        value,
        destinations=destinations,
        axis=axis,
        options=options)

  def _reduce_to(self, reduce_op, value, destinations, options):
    # [FUNCTION DOC]
    # Reduces value to destinations using reduce_op. Verifies destinations are not on another
    # worker (raises a value error otherwise). Non-distributed values are reduced using the
    # replica count; distributed values via the cross-device reduction with the options.
    self._verify_destinations_not_different_worker(destinations)
    if not isinstance(value, values.DistributedValues):
      # pylint: disable=protected-access
      return cross_device_ops_lib.reduce_non_distributed_value(
          reduce_op, value, destinations, self._num_replicas_in_sync)
    return self._cross_device_ops.reduce(
        reduce_op, value, destinations=destinations, options=options)

  def _batch_reduce_to(self, reduce_op, value_destination_pairs, options):
    # [FUNCTION DOC]
    # Reduces a list of (value, destinations) pairs with reduce_op. Verifies every destination is
    # not on another worker (raises a value error), then uses the cross-device batch reduction
    # with the options. Returns the list of reduced values.
    for _, destinations in value_destination_pairs:
      self._verify_destinations_not_different_worker(destinations)
    return self._cross_device_ops.batch_reduce(reduce_op,
                                               value_destination_pairs, options)

  def _select_single_value(self, structured):
    # [FUNCTION DOC]
    # HELPER: used when passing arguments to a variable update.
    # Maps over a nested structure, replacing each mirrored or per-replica value with its primary
    # component and leaving other values unchanged. Returns the new structure.
    """Select any single value in `structured`."""

    def _select_fn(x):  # pylint: disable=g-missing-docstring
      # [FUNCTION DOC]
      # Returns the primary component of a mirrored or per-replica value, else the value itself.
      if isinstance(x, values.Mirrored) or isinstance(x, values.PerReplica):
        return x._primary  # pylint: disable=protected-access
      else:
        return x

    return nest.map_structure(_select_fn, structured)

  def _update(self, var, fn, args, kwargs, group):
    # [FUNCTION DOC]
    # Applies fn to a variable. An aggregating wrapper is unwrapped first; raises a value error if
    # var is not a resource variable. Runs fn colocated with var inside an update context for var's
    # device, passing single-value selections of args and kwargs. If group is true returns the
    # raw result, otherwise maps the structure to local results.
    if isinstance(var, ps_values.AggregatingVariable):
      var = var.get()
    if not resource_variable_ops.is_resource_variable(var):
      raise ValueError(
          "You can not update `var` %r. It must be a Variable." % var)
    with ops.colocate_with(var), distribute_lib.UpdateContext(var.device):
      result = fn(var, *self._select_single_value(args),
                  **self._select_single_value(kwargs))
      if group:
        return result
      else:
        return nest.map_structure(self._local_results, result)

  # TODO(yuefengz): does it need to call _select_single_value?
  def _update_non_slot(self, colocate_with, fn, args, kwargs, group):
    # [FUNCTION DOC]
    # Applies fn (with args and kwargs unchanged) for a non-slot variable under the device of the
    # colocation target within an update context. If group is true returns the raw result,
    # otherwise maps the structure to local results.
    with ops.device(
        colocate_with.device), distribute_lib.UpdateContext(colocate_with):
      result = fn(*args, **kwargs)
      if group:
        return result
      else:
        return nest.map_structure(self._local_results, result)

  def value_container(self, val):
    # [FUNCTION DOC]
    # Returns the aggregating wrapper variable for val if val has a live reference to one and is
    # not itself an aggregating variable; otherwise returns val.
    if (hasattr(val, "_aggregating_container") and
        not isinstance(val, ps_values.AggregatingVariable)):
      wrapper = val._aggregating_container()  # pylint: disable=protected-access
      if wrapper is not None:
        return wrapper
    return val

  def read_var(self, var):
    # [FUNCTION DOC]
    # Returns an identity tensor of the variable's current value (no distinction between normal
    # and replica-local variables).
    # No need to distinguish between normal variables and replica-local
    # variables.
    return array_ops.identity(var)

  def _configure(self,
                 session_config=None,
                 cluster_spec=None,
                 task_type=None,
                 task_id=None):
    # [FUNCTION DOC]
    # Reconfigures the strategy. If a cluster spec is given (dict, cluster definition or spec),
    # builds a simple resolver from it with the task type and id and the previously saved GPU
    # count per worker, and re-initializes multi-worker mode. If a session config is given, it is
    # overwritten in place with the updated config. Returns nothing.
    """Configures the strategy class with `cluster_spec`.

    The strategy object will be re-initialized if `cluster_spec` is passed to
    `configure` but was not passed when instantiating the strategy.

    Args:
      session_config: Session config object.
      cluster_spec: a dict, ClusterDef or ClusterSpec object specifying the
        cluster configurations.
      task_type: the current task type.
      task_id: the current task id.

    Raises:
      ValueError: if `cluster_spec` is given but `task_type` or `task_id` is
        not.
    """
    if cluster_spec:
      # Use the num_gpus_per_worker recorded in constructor since _configure
      # doesn't take num_gpus.
      cluster_resolver = cluster_resolver_lib.SimpleClusterResolver(
          cluster_spec=multi_worker_util.normalize_cluster_spec(cluster_spec),
          task_type=task_type,
          task_id=task_id,
          num_accelerators={"GPU": self._num_gpus_per_worker})
      self._initialize_multi_worker(cluster_resolver)

    if session_config:
      session_config.CopyFrom(self._update_config_proto(session_config))

  def _update_config_proto(self, config_proto):
    # [FUNCTION DOC]
    # Returns a deep copy of the session config adapted to this strategy. In local mode, enables
    # session state isolation. In multi-worker mode, disables isolation (asserting task type and id
    # are set), clears device filters, and for chief or worker tasks allows only this task and the
    # ps job, for evaluators only this task. The input is not modified.
    updated_config = copy.deepcopy(config_proto)
    if not self._cluster_spec:
      updated_config.isolate_session_state = True
      return updated_config

    updated_config.isolate_session_state = False

    assert self._task_type
    assert self._task_id is not None

    # The device filters prevent communication between workers.
    del updated_config.device_filters[:]
    if self._task_type in ["chief", "worker"]:
      updated_config.device_filters.extend(
          ["/job:%s/task:%d" % (self._task_type, self._task_id), "/job:ps"])
    elif self._task_type == "evaluator":
      updated_config.device_filters.append(
          "/job:%s/task:%d" % (self._task_type, self._task_id))
    return updated_config

  def _in_multi_worker_mode(self):
    # [FUNCTION DOC]
    # Returns true when a cluster spec is set (multi-worker mode), false otherwise.
    """Whether this strategy indicates working in multi-worker settings."""
    return self._cluster_spec is not None

  @property
  def _num_replicas_in_sync(self):
    # [FUNCTION DOC]
    # Returns the number of replicas, equal to the number of compute devices.
    return len(self._compute_devices)

  @property
  def worker_devices(self):
    # [FUNCTION DOC]
    # Returns the compute devices (one per replica).
    return self._compute_devices

  @property
  def worker_devices_by_replica(self):
    # [FUNCTION DOC]
    # Returns a list with one single-element list per replica holding that replica's device.
    return [[d] for d in self._compute_devices]

  @property
  def parameter_devices(self):
    # [FUNCTION DOC]
    # Returns the tuple of devices where variables live (ps tasks or the single local device).
    return self._parameter_devices

  def non_slot_devices(self, var_list):
    # [FUNCTION DOC]
    # Returns the variable with the lexicographically smallest name from var_list, used as the
    # placement for non-slot variables. Raises if the list is empty.
    return min(var_list, key=lambda x: x.name)

  @property
  def experimental_between_graph(self):
    # [FUNCTION DOC]
    # Always returns true: between-graph replication is used.
    # TODO(yuefengz): Should this return False in the local case?
    return True

  @property
  def experimental_should_init(self):
    # [FUNCTION DOC]
    # Returns whether this task should run variable initialization: true only for the chief.
    return self._is_chief

  @property
  def should_checkpoint(self):
    # [FUNCTION DOC]
    # Returns whether this task should write checkpoints: true only for the chief.
    return self._is_chief

  @property
  def should_save_summary(self):
    # [FUNCTION DOC]
    # Returns whether this task should write summaries: true only for the chief.
    return self._is_chief

  # TODO(priyag): Delete this once all strategies use global batch size.
  @property
  def _global_batch_size(self):
    # [FUNCTION DOC]
    # Always returns true: dataset and numpy iterators use the global batch size.
    """`make_dataset_iterator` and `make_numpy_iterator` use global batch size.

    `make_input_fn_iterator` assumes per-replica batching.

    Returns:
      Boolean.
    """
    return True

  def _get_local_replica_id(self, replica_id_in_sync_group):
    # [FUNCTION DOC]
    # Returns the given replica id in the sync group unchanged (local id equals group id here).
    return replica_id_in_sync_group

  def _get_replica_id_in_sync_group(self, replica_id):
    # [FUNCTION DOC]
    # Returns the given replica id unchanged (the id in the sync group equals the replica id).
    return replica_id
