# [MODULE DOC]
# Implementation of the multi-worker synchronous distribution strategy (public name: multi-worker
# mirrored strategy), built on collective operations. Defines the V2 strategy class, a deprecated
# experimental subclass kept for compatibility (with a metaclass so instance checks against it
# succeed for the main class), the V1 strategy class, and the extended class that holds the real
# implementation (cluster setup, device placement, variable creation, dataset distribution,
# reductions, health checks and session configuration). Public classes are registered with the
# API export decorator at import time and the experimental class takes the main class's name.
# Relies on cross-device ops, cluster resolvers and the mirrored strategy base classes.
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
"""Class CollectiveAllReduceStrategy implementing DistributionStrategy."""

import copy
import threading
import time
import weakref

from xla.tsl.protobuf import coordination_config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import cross_device_utils
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import input_util
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values
from tensorflow.python.distribute.cluster_resolver import cluster_resolver as cluster_resolver_lib
from tensorflow.python.distribute.cluster_resolver import tfconfig_cluster_resolver
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.distribute.v1 import input_lib as input_lib_v1
from tensorflow.python.eager import context
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import collective_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


# pylint: disable=line-too-long
@tf_export("distribute.MultiWorkerMirroredStrategy", v1=[])
class CollectiveAllReduceStrategy(distribute_lib.Strategy):
  # [CLASS DOC]
  # Public V2 strategy for synchronous training across multiple workers, each possibly with
  # several local devices. Variables and computation are replicated per device and reduced using
  # collectives. A thin wrapper: all logic lives in the extended implementation object created in
  # the constructor. Class attribute: starting number for collective keys (only changed in
  # tests). Records usage in distribution-strategy metrics gauges at construction.
  """A distribution strategy for synchronous training on multiple workers.

  This strategy implements synchronous distributed training across multiple
  workers, each with potentially multiple GPUs. Similar to
  `tf.distribute.MirroredStrategy`, it replicates all variables and computations
  to each local device. The difference is that it uses a distributed collective
  implementation (e.g. all-reduce), so that multiple workers can work together.

  You need to launch your program on each worker and configure
  `cluster_resolver` correctly. For example, if you are using
  `tf.distribute.cluster_resolver.TFConfigClusterResolver`, each worker needs to
  have its corresponding `task_type` and `task_id` set in the `TF_CONFIG`
  environment variable. An example TF_CONFIG on worker-0 of a two worker cluster
  is:

  ```
  TF_CONFIG = '{"cluster": {"worker": ["localhost:12345", "localhost:23456"]},
  "task": {"type": "worker", "index": 0} }'
  ```

  Your program runs on each worker as-is. Note that collectives require each
  worker to participate. All `tf.distribute` and non `tf.distribute` API may use
  collectives internally, e.g. checkpointing and saving since reading a
  `tf.Variable` with `tf.VariableSynchronization.ON_READ` all-reduces the value.
  Therefore it's recommended to run exactly the same program on each worker.
  Dispatching based on `task_type` or `task_id` of the worker is error-prone.

  `cluster_resolver.num_accelerators()` determines the number of GPUs the
  strategy uses. If it's zero, the strategy uses the CPU. All workers need to
  use the same number of devices, otherwise the behavior is undefined.

  This strategy is not intended for TPU. Use `tf.distribute.TPUStrategy`
  instead.

  After setting up TF_CONFIG, using this strategy is similar to using
  `tf.distribute.MirroredStrategy` and `tf.distribute.TPUStrategy`.

  ```
  strategy = tf.distribute.MultiWorkerMirroredStrategy()

  with strategy.scope():
    model = tf.keras.Sequential([
      tf.keras.layers.Dense(2, input_shape=(5,)),
    ])
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

  def dataset_fn(ctx):
    x = np.random.random((2, 5)).astype(np.float32)
    y = np.random.randint(2, size=(2, 1))
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    return dataset.repeat().batch(1, drop_remainder=True)
  dist_dataset = strategy.distribute_datasets_from_function(dataset_fn)

  model.compile()
  model.fit(dist_dataset)
  ```

  You can also write your own training loop:

  ```
  @tf.function
  def train_step(iterator):

    def step_fn(inputs):
      features, labels = inputs
      with tf.GradientTape() as tape:
        logits = model(features, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            labels, logits)

      grads = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))

    strategy.run(step_fn, args=(next(iterator),))

  for _ in range(NUM_STEP):
    train_step(iterator)
  ```

  See
  [Multi-worker training with
  Keras](https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras)
  for a detailed tutorial.

  __Saving__

  You need to save and checkpoint on all workers instead of just one. This is
  because variables whose synchronization=ON_READ triggers aggregation during
  saving. It's recommended to save to a different path on each worker to avoid
  race conditions. Each worker saves the same thing. See
  [Multi-worker training with
  Keras](https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras#model_saving_and_loading)
  tutorial for examples.

  __Known Issues__

  * `tf.distribute.cluster_resolver.TFConfigClusterResolver` does not return the
  correct number of accelerators. The strategy uses all available GPUs if
  `cluster_resolver` is `tf.distribute.cluster_resolver.TFConfigClusterResolver`
  or `None`.
  * In eager mode, the strategy needs to be created before calling any other
  Tensorflow API.
  """

  # pylint: enable=line-too-long

  # TODO(anjalisridhar): Update our guides with examples showing how we can use
  # the cluster_resolver argument.

  # The starting number for collective keys. This should only be set in tests.
  _collective_key_base = 0

  def __init__(self, cluster_resolver=None, communication_options=None):
    # [FUNCTION DOC]
    # Creates the strategy. Parameters: cluster_resolver (optional cluster resolver; nothing means
    # a resolver reading the cluster configuration environment variable) and communication_options
    # (optional options object; nothing means default options). Builds the extended
    # implementation (which may raise a value error for bad argument types) and passes it to the
    # base class. Side effects: sets metrics gauges recording the strategy name, number of
    # workers and number of replicas per worker. Returns nothing.
    """Creates the strategy.

    Args:
      cluster_resolver: optional
        `tf.distribute.cluster_resolver.ClusterResolver`. If `None`,
        `tf.distribute.cluster_resolver.TFConfigClusterResolver` is used.
      communication_options: optional
        `tf.distribute.experimental.CommunicationOptions`. This configures the
        default options for cross device communications. It can be overridden by
        options provided to the communication APIs like
        `tf.distribute.ReplicaContext.all_reduce`. See
        `tf.distribute.experimental.CommunicationOptions` for details.
    """
    if communication_options is None:
      communication_options = collective_util.Options()
    super(CollectiveAllReduceStrategy, self).__init__(
        CollectiveAllReduceExtended(
            self,
            cluster_resolver=cluster_resolver,
            communication_options=communication_options,
        )
    )

    distribute_lib.distribution_strategy_gauge.get_cell("V2").set(
        "MultiWorkerMirroredStrategy"
    )
    # pylint: disable=protected-access
    distribute_lib.distribution_strategy_replica_gauge.get_cell(
        "num_workers"
    ).set(self.extended._num_workers)
    distribute_lib.distribution_strategy_replica_gauge.get_cell(
        "num_replicas_per_worker"
    ).set(self.extended._num_devices_per_worker)

  @classmethod
  def _from_local_devices(cls, devices, communication_options=None):
    # [FUNCTION DOC]
    # HELPER: convenience constructor (mostly for tests) that builds a strategy limited to an
    # explicit list of local device names. Takes the device list and optional communication
    # options. Creates a default strategy, then re-initializes its extended object as a
    # single-worker setup on those devices. Returns the strategy.
    """A convenience method to create an object with a list of devices."""
    obj = cls(communication_options=communication_options)
    obj.extended._initialize_local(  # pylint: disable=protected-access
        tfconfig_cluster_resolver.TFConfigClusterResolver(), devices=devices
    )
    return obj

  @property
  def cluster_resolver(self):
    # [FUNCTION DOC]
    # Property returning the cluster resolver held by the extended implementation: the one
    # supplied at construction, or the default environment-based one.
    """Returns the cluster resolver associated with this strategy.

    As a multi-worker strategy, `tf.distribute.MultiWorkerMirroredStrategy`
    provides the associated `tf.distribute.cluster_resolver.ClusterResolver`. If
    the user provides one in `__init__`, that instance is returned; if the user
    does not, a default `TFConfigClusterResolver` is provided.
    """
    return self.extended._cluster_resolver  # pylint: disable=protected-access


class _CollectiveAllReduceStrategyExperimentalMeta(type):
  # [CLASS DOC]
  # HELPER: metaclass for the deprecated experimental strategy class. Overrides the instance
  # check so any instance of the main strategy class counts as an instance of the experimental
  # class (some libraries perform that check).

  @classmethod
  def __instancecheck__(cls, instance):
    # [FUNCTION DOC]
    # HELPER: instance-check hook for the experimental alias. Takes an object and returns true
    # when it is an instance of the main strategy class, false otherwise.
    # This is to make isinstance(tf.distribute.MultiWorkerMirroredStrategy(),
    # tf.distribute.experimental.MultiWorkerMirroredStrategy). Some libraries is
    # performing such check.
    return isinstance(instance, CollectiveAllReduceStrategy)


@tf_export("distribute.experimental.MultiWorkerMirroredStrategy", v1=[])
class _CollectiveAllReduceStrategyExperimental(
    CollectiveAllReduceStrategy,
    metaclass=_CollectiveAllReduceStrategyExperimentalMeta,
):
  # [CLASS DOC]
  # Deprecated experimental public name of the multi-worker strategy. Subclass of the main
  # strategy that takes a communication implementation hint instead of an options object and
  # reuses the main class documentation. Its name is overwritten after the class body so it
  # appears as the main class. Instance checks go through a custom metaclass.

  __doc__ = CollectiveAllReduceStrategy.__doc__

  @deprecation.deprecated(
      None, "use distribute.MultiWorkerMirroredStrategy instead"
  )
  def __init__(
      self,
      communication=collective_util.CommunicationImplementation.AUTO,
      cluster_resolver=None,
  ):
    # [FUNCTION DOC]
    # Deprecated constructor (logs a deprecation warning). Parameters: communication (preferred
    # communication implementation, default automatic) and cluster_resolver (optional, nothing
    # means the environment-based default). Wraps the implementation hint in an options object
    # and calls the main strategy constructor. Returns nothing.
    """Creates the strategy.

    Args:
      communication: optional
        `tf.distribute.experimental.CommunicationImplementation`. This is a hint
        on the preferred collective communication implementation. Possible
        values include `AUTO`, `RING`, and `NCCL`.
      cluster_resolver: optional
        `tf.distribute.cluster_resolver.ClusterResolver`. If `None`,
        `tf.distribute.cluster_resolver.TFConfigClusterResolver` is used.
    """
    communication_options = collective_util.Options(
        implementation=communication
    )
    super(_CollectiveAllReduceStrategyExperimental, self).__init__(
        cluster_resolver, communication_options
    )

  @classmethod
  def _from_local_devices(
      cls,
      devices,
      communication=collective_util.CommunicationImplementation.AUTO,
  ):
    # [FUNCTION DOC]
    # HELPER: test convenience constructor for the experimental class. Takes a list of local
    # device names and a communication implementation hint (default automatic). Builds the
    # strategy and re-initializes it as a single-worker setup on those devices. Returns it.
    """A convenience method to create an object with a list of devices."""
    obj = cls(communication)
    # pylint: disable=protected-access
    obj.extended._initialize_local(
        tfconfig_cluster_resolver.TFConfigClusterResolver(), devices=devices
    )
    return obj


_CollectiveAllReduceStrategyExperimental.__name__ = (
    CollectiveAllReduceStrategy.__name__
)


@tf_export(v1=["distribute.experimental.MultiWorkerMirroredStrategy"])  # pylint: disable=missing-docstring
class CollectiveAllReduceStrategyV1(distribute_lib.StrategyV1):
  # [CLASS DOC]
  # V1-API flavor of the multi-worker synchronous strategy, exported under the experimental name
  # for the first-generation API. Shares documentation with the V2 class and uses the same
  # extended implementation. Class attribute: starting number for collective keys (tests only).

  __doc__ = CollectiveAllReduceStrategy.__doc__

  # The starting number for collective keys. This should only be set in tests.
  _collective_key_base = 0

  def __init__(
      self,
      communication=collective_util.CommunicationImplementation.AUTO,
      cluster_resolver=None,
  ):
    # [FUNCTION DOC]
    # Creates the V1 strategy. Parameters: communication (preferred implementation hint, default
    # automatic) and cluster_resolver (optional; nothing means environment-based default).
    # Builds the extended implementation with an options object wrapping the hint and passes it
    # to the V1 base class. Side effects: sets metrics gauges for strategy name, number of
    # workers and GPUs per worker (zero when local devices are not GPUs). Returns nothing.
    """Initializes the object."""
    communication_options = collective_util.Options(
        implementation=communication
    )
    super(CollectiveAllReduceStrategyV1, self).__init__(
        CollectiveAllReduceExtended(
            self,
            cluster_resolver=cluster_resolver,
            communication_options=communication_options,
        )
    )
    distribute_lib.distribution_strategy_gauge.get_cell("V1").set(
        "MultiWorkerMirroredStrategy"
    )
    # pylint: disable=protected-access
    distribute_lib.distribution_strategy_replica_gauge.get_cell(
        "num_workers"
    ).set(self.extended._num_workers)
    distribute_lib.distribution_strategy_replica_gauge.get_cell(
        "num_gpu_per_worker"
    ).set(
        self.extended._num_devices_per_worker
        if self.extended._local_device_type == "GPU"
        else 0
    )


def _is_gpu_device(device):
  # [FUNCTION DOC]
  # HELPER: device-type test used when deciding whether merge calls are needed. Takes a device
  # name string; returns true if it parses to a device of type GPU.
  return tf_device.DeviceSpec.from_string(device).device_type == "GPU"


class CollectiveAllReduceExtended(mirrored_strategy.MirroredExtended):
  # [CLASS DOC]
  # Implementation object behind the multi-worker strategies, extending the mirrored strategy
  # implementation. Holds the cluster resolver, communication options, collective key base,
  # worker and device lists, cross-device ops, a cache of functions, and health-check settings
  # (class attributes: enable flag, interval, initial timeout, retry limit, per-check timeout).
  # Decides between single-worker and multi-worker setup, creates variables and datasets,
  # performs reductions and broadcasts through collectives, and configures sessions and the
  # coordination service.
  """Implementation of CollectiveAllReduceStrategy."""

  # Whether to perdically check the health of the cluster. If any worker is not
  # reachable, collectives are aborted and the user program should get a
  # tf.errors.UnavailableError. It's required to restart in order to recover.
  _enable_check_health = True
  # Check health interval in seconds.
  _check_health_interval = 30
  # Timeout in seconds for the first check health. The first check health needs
  # to wait for cluster, which may make a longer time.
  _check_health_initial_timeout = 0
  # Times to retry before considering the peer is down.
  _check_health_retry_limit = 3
  # Timeout in seconds the each check health.
  _check_health_timeout = 10

  def __init__(
      self,
      container_strategy,
      cluster_resolver,
      communication_options,
      devices=None,
  ):
    # [FUNCTION DOC]
    # Builds the implementation. Parameters: container_strategy (the owning strategy object),
    # cluster_resolver (cluster resolver or nothing for the environment-based default),
    # communication_options (must be an options object), devices (optional explicit list of local
    # devices; cannot be combined with a cluster resolver). Raises a value error if the options
    # are not an options object, if both a resolver and devices are given, or if the resolver is
    # not a cluster resolver. Initializes the base class, copies the collective key base from the
    # strategy, sets up single or multi-worker state, creates the function cache, enables
    # get-next-as-optional, and asserts the cross-device ops are the collective all-reduce kind.
    if not isinstance(communication_options, collective_util.Options):
      raise ValueError(
          "communication_options must be an instance of "
          "tf.distribute.experimental.CommunicationOptions"
      )
    if cluster_resolver and devices:
      raise ValueError(
          "cluster_resolver and devices cannot be set at the same time"
      )

    self._cluster_resolver = (
        cluster_resolver or tfconfig_cluster_resolver.TFConfigClusterResolver()
    )
    if not isinstance(
        self._cluster_resolver, cluster_resolver_lib.ClusterResolver
    ):
      raise ValueError(
          "cluster_resolver must be an instance of "
          "tf.distribute.cluster_resolver.ClusterResolver"
      )
    distribute_lib.StrategyExtendedV1.__init__(self, container_strategy)
    self._communication_options = communication_options
    self._collective_key_base = container_strategy._collective_key_base  # pylint: disable=protected-access
    self._initialize_strategy(self._cluster_resolver, devices=devices)
    self._cfer_fn_cache = weakref.WeakKeyDictionary()
    self.experimental_enable_get_next_as_optional = True
    assert isinstance(
        self._cross_device_ops, cross_device_ops_lib.CollectiveAllReduce
    )

  def _use_merge_call(self):
    # [FUNCTION DOC]
    # Decides whether cross-replica operations need a merge call. Returns false only when the
    # current graph (or a parent) is inside an XLA compilation context and every device is a GPU;
    # otherwise returns true.
    # We currently only disable merge_call when XLA is used to compile the `fn`
    # passed to `strategy.run` and all devices are GPU.
    return not control_flow_util.GraphOrParentsInXlaContext(
        ops.get_default_graph()
    ) or not all([_is_gpu_device(d) for d in self._devices])

  def _initialize_strategy(self, cluster_resolver, devices):
    # [FUNCTION DOC]
    # HELPER: dispatcher for setup. If explicit devices were given or the resolver's cluster
    # specification is empty, performs single-worker local initialization; otherwise multi-worker
    # initialization. Takes the resolver and devices; returns nothing.
    # If devices are provided or cluster_spec is not specified, initialize
    # single worker. Otherwise initialize multi workers.
    if devices or not cluster_resolver.cluster_spec().as_dict():
      self._initialize_local(cluster_resolver, devices=devices)
    else:
      self._initialize_multi_worker(cluster_resolver)

  def _initialize_local_devices(self, cluster_resolver, worker_device):
    # [FUNCTION DOC]
    # HELPER: works out the local devices of this worker. Takes a cluster resolver and a worker
    # device prefix string. For the environment-based resolver it counts GPUs through the eager
    # context (and assumes no TPUs); for other resolvers it reads the accelerator counts from the
    # resolver. Prefers GPUs, then TPUs, otherwise one CPU. Returns a pair: a tuple of full device
    # name strings (prefix plus device type and index) and the device type string.
    # TODO(b/126786766): TFConfigClusterResolver returns wrong number of GPUs in
    # some cases.
    if isinstance(
        cluster_resolver, tfconfig_cluster_resolver.TFConfigClusterResolver
    ):
      num_gpus = context.num_gpus()
      num_tpus = 0
    else:
      num_gpus = cluster_resolver.num_accelerators().get("GPU", 0)
      num_tpus = cluster_resolver.num_accelerators().get("TPU", 0)

    if num_gpus:
      local_device_type = "GPU"
      num_local_devices = num_gpus
    elif num_tpus:
      local_device_type = "TPU"
      num_local_devices = num_tpus
    else:
      local_device_type = "CPU"
      num_local_devices = 1
    local_devices = tuple(
        f"{worker_device}/device:{local_device_type}:{i}"
        for i in range(num_local_devices)
    )
    return local_devices, local_device_type

  def _initialize_local(self, cluster_resolver, devices=None):
    # [FUNCTION DOC]
    # Sets this object up as a single-worker (local) strategy. Takes a cluster resolver and an
    # optional list of device names (nothing means detect devices). Marks the worker as chief with
    # one worker; when eager outside functions, tries to configure collective ops with scoped
    # allocation (logs a warning if the runtime is already initialized). The device type is taken
    # from the first given device name or detected. Creates collective keys, cross-device ops for
    # the local devices and for per-host tensors, initializes the single-worker base state,
    # clears cluster fields, flags local/standalone-client mode, and records devices per worker,
    # device type and rpc layer. Warns if NCCL is requested without GPUs and logs the setup.
    # Returns nothing.
    """Initializes the object for local training."""
    self._is_chief = True
    self._num_workers = 1

    if ops.executing_eagerly_outside_functions():
      try:
        context.context().configure_collective_ops(
            scoped_allocator_enabled_ops=("CollectiveReduce",)
        )
      except RuntimeError:
        logging.warning(
            "Collective ops is not configured at program startup. "
            "Some performance features may not be enabled."
        )
      self._collective_ops_configured = True

    if devices:
      local_devices = devices
      if "GPU" in devices[0]:
        local_device_type = "GPU"
      elif "TPU" in devices[0]:
        local_device_type = "TPU"
      else:
        local_device_type = "CPU"
    else:
      local_devices, local_device_type = self._initialize_local_devices(
          cluster_resolver, worker_device=""
      )

    self._worker_device = device_util.canonicalize("/device:CPU:0")
    self._host_input_device = numpy_dataset.SingleDevice(self._worker_device)

    self._collective_keys = cross_device_utils.CollectiveKeys(
        group_key_start=1 + self._collective_key_base
    )
    self._cross_device_ops = cross_device_ops_lib.CollectiveAllReduce(
        devices=local_devices,
        group_size=len(local_devices),
        options=self._communication_options,
        collective_keys=self._collective_keys,
    )
    # CrossDeviceOps for per host tensors.
    self._host_cross_device_ops = cross_device_ops_lib.CollectiveAllReduce(
        devices=[self._worker_device],
        group_size=self._num_workers,
        options=self._communication_options,
        collective_keys=self._collective_keys,
    )
    super(CollectiveAllReduceExtended, self)._initialize_single_worker(
        local_devices
    )

    self._cluster_spec = None
    self._task_type = None
    self._task_id = None
    self._id_in_cluster = 0

    # This is a mark to tell whether we are running with standalone client or
    # independent worker. Right now with standalone client, strategy object is
    # created as local strategy and then turn into multi-worker strategy via
    # configure call.
    self._local_or_standalone_client_mode = True

    # Save the num_devices_per_worker and rpc_layer for configure method.
    self._num_devices_per_worker = len(local_devices)
    self._local_device_type = local_device_type
    self._rpc_layer = cluster_resolver.rpc_layer
    self._warn_nccl_no_gpu()

    logging.info(
        "Single-worker MultiWorkerMirroredStrategy with local_devices "
        "= %r, communication = %s",
        local_devices,
        self._communication_options.implementation,
    )

  def _initialize_multi_worker(self, cluster_resolver):
    # [FUNCTION DOC]
    # Sets this object up for multi-worker training from a resolver that has a cluster
    # specification. Raises a value error if the task type or task id is missing, or if the
    # cluster has no worker, chief or evaluator tasks. Records cluster spec, task, id in cluster,
    # worker count, chief flag and worker device. In eager mode (and not standalone-client mode)
    # configures collective ops with a leader and device filters, and, if no coordination service
    # exists and the task is a chief or worker, configures a standalone coordination service
    # over the chief and worker jobs. In eager independent-worker mode, starts the server once
    # with the session configuration (disabling the Python health check when a coordination
    # service type is set). Then detects local devices (initializing the TPU system for TPUs),
    # creates collective keys and cross-device ops sized for all workers, sets the default device
    # to this task, records device counts and rpc layer, warns about NCCL without GPUs, starts the
    # health-check thread if enabled and eager, and logs the configuration. Returns nothing.
    """Initializes the object for multi-worker training."""
    cluster_spec = multi_worker_util.normalize_cluster_spec(
        cluster_resolver.cluster_spec()
    )
    task_type = cluster_resolver.task_type
    task_id = cluster_resolver.task_id
    if task_type is None or task_id is None:
      raise ValueError(
          "When `cluster_spec` is given, you must also specify "
          "`task_type` and `task_id`."
      )
    self._cluster_spec = cluster_spec
    self._task_type = task_type
    self._task_id = task_id
    self._id_in_cluster = multi_worker_util.id_in_cluster(
        self._cluster_spec, self._task_type, self._task_id
    )

    self._num_workers = multi_worker_util.worker_count(cluster_spec, task_type)
    if not self._num_workers:
      raise ValueError(
          "No `worker`, `chief` or `evaluator` tasks can be found "
          "in `cluster_spec`."
      )

    self._is_chief = multi_worker_util.is_chief(
        cluster_spec, task_type, task_id
    )

    self._worker_device = "/job:%s/task:%d" % (task_type, task_id)
    self._host_input_device = numpy_dataset.SingleDevice(self._worker_device)

    if ops.executing_eagerly_outside_functions() and not getattr(
        self, "_local_or_standalone_client_mode", False
    ):
      context.context().configure_collective_ops(
          collective_leader=multi_worker_util.collective_leader(
              cluster_spec, task_type, task_id
          ),
          scoped_allocator_enabled_ops=("CollectiveReduce",),
          device_filters=("/job:%s/task:%d" % (task_type, task_id),),
      )
      self._collective_ops_configured = True
      if context.context().coordination_service is None:
        coordinated_jobs = ["chief", "worker"]
        if task_type in coordinated_jobs:
          coordinated_job_config = []
          for job in coordinated_jobs:
            if job in cluster_spec.jobs:
              coordinated_job_config.append(
                  coordination_config_pb2.CoordinatedJob(
                      name=job, num_tasks=cluster_spec.num_tasks(job)
                  )
              )
          context.context().configure_coordination_service(
              service_type="standalone",
              service_leader=multi_worker_util.coordination_leader(
                  cluster_spec
              ),
              coordinated_jobs=coordinated_job_config,
          )

    # Starting a std server in eager mode and in independent worker mode.
    if (
        context.executing_eagerly()
        and not getattr(self, "_std_server_started", False)
        and not getattr(self, "_local_or_standalone_client_mode", False)
    ):
      # Checking _local_or_standalone_client_mode as well because we should not
      # create the std server in standalone client mode.
      config_proto = copy.deepcopy(context.context().config)
      config_proto = self._update_config_proto(config_proto)

      # If coordination service is enabled, use its internal heartbeat to detect
      # peer failures instead of the Python-level health check.
      if config_proto.experimental.coordination_config.service_type:
        self._enable_check_health = False

      if hasattr(cluster_resolver, "port"):
        port = cluster_resolver.port
      else:
        port = 0
      server_def = tensorflow_server_pb2.ServerDef(
          cluster=cluster_spec.as_cluster_def(),
          default_session_config=config_proto,
          job_name=task_type,
          task_index=task_id,
          protocol=cluster_resolver.rpc_layer or "grpc",
          port=port,
      )
      context.context().enable_collective_ops(server_def)
      self._std_server_started = True
      # The `ensure_initialized` is needed before calling
      # `context.context().devices()`.
      context.context().ensure_initialized()
      logging.info(
          "Enabled multi-worker collective ops with available devices: %r",
          context.context().devices(),
      )

    # TODO(yuefengz): The `num_gpus` is only for this particular task. It
    # assumes all workers have the same number of GPUs. We should remove this
    # assumption by querying all tasks for their numbers of GPUs.
    # TODO(b/126786766): TFConfigClusterResolver returns wrong number of GPUs in
    # some cases.
    local_devices, local_device_type = self._initialize_local_devices(
        cluster_resolver, self._worker_device
    )
    if local_device_type == "TPU":
      tpu_cluster_resolver.initialize_tpu_system()

    self._collective_keys = cross_device_utils.CollectiveKeys(
        group_key_start=1 + self._collective_key_base
    )
    self._cross_device_ops = cross_device_ops_lib.CollectiveAllReduce(
        devices=local_devices,
        group_size=len(local_devices) * self._num_workers,
        options=self._communication_options,
        collective_keys=self._collective_keys,
    )
    # CrossDeviceOps for per host tensors.
    self._host_cross_device_ops = cross_device_ops_lib.CollectiveAllReduce(
        devices=[self._worker_device],
        group_size=self._num_workers,
        options=self._communication_options,
        collective_keys=self._collective_keys,
    )
    super(CollectiveAllReduceExtended, self)._initialize_single_worker(
        local_devices
    )

    # Add a default device so that ops without specified devices will not end up
    # on other workers.
    self._default_device = "/job:%s/task:%d" % (task_type, task_id)

    # Save the num_devices_per_worker and rpc_layer for configure method.
    self._num_devices_per_worker = len(local_devices)
    self._local_device_type = local_device_type
    self._rpc_layer = cluster_resolver.rpc_layer
    self._warn_nccl_no_gpu()

    if self._enable_check_health and context.executing_eagerly():
      self._start_check_health_thread()
    else:
      logging.info("Check health not enabled.")

    logging.info(
        "MultiWorkerMirroredStrategy with cluster_spec = %r, task_type = %r, "
        "task_id = %r, num_workers = %r, local_devices = %r, "
        "communication = %s",
        cluster_spec.as_dict(),
        task_type,
        task_id,
        self._num_workers,
        local_devices,
        self._communication_options.implementation,
    )

  def __del__(self):
    # [FUNCTION DOC]
    # Finalizer: stops the background health-check thread, if any. Returns nothing.
    self._stop_check_health_thread()

  def _input_workers_with_options(self, options=None):
    # [FUNCTION DOC]
    # Builds the input-workers description for this worker. Takes optional input options. If no
    # options are given or fetching to device is enabled, input is placed on the host with the
    # worker devices as targets; otherwise the targets are the host devices of the worker
    # devices. Returns an input-workers object with one host entry.
    host_device = device_util.get_host_for_device(self._worker_device)
    if not options or options.experimental_fetch_to_device:
      return input_lib.InputWorkers([(host_device, self.worker_devices)])
    else:
      return input_lib.InputWorkers([(
          host_device,
          [
              device_util.get_host_for_device(worker)
              for worker in self.worker_devices
          ],
      )])

  @property
  def _input_workers(self):
    # [FUNCTION DOC]
    # Property: the input-workers description built with default options.
    return self._input_workers_with_options()

  def _get_variable_creator_initial_value(
      self, replica_id, device, primary_var, **kwargs
  ):
    # [FUNCTION DOC]
    # HELPER: supplies the initial-value provider for a variable replica. Takes the replica id,
    # target device, the primary variable (must be nothing for replica zero) and keyword
    # arguments including the initial value. For the first replica on each worker, returns a
    # function that evaluates the initial value on the device and, with several workers, makes
    # all workers share the chief's value by broadcast (chief sends, others receive); a single
    # worker just returns the value. Asserts a device is given and the primary variable is absent.
    # For other replicas, defers to the base class implementation.
    if replica_id == 0:  # First replica on each worker.
      assert device is not None
      assert primary_var is None

      def initial_value_fn():  # pylint: disable=g-missing-docstring
        # [FUNCTION DOC]
        # HELPER: nested initial-value producer for the first replica. Takes no arguments.
        # Obtains group and instance keys for the device, evaluates the initial value (calling it
        # if callable, unwrapping checkpoint-restored values), converts it to a tensor with the
        # requested type, and with more than one worker broadcasts it from the chief (the send
        # is made a control dependency of the returned identity) or receives it on other workers.
        # Returns the resulting tensor.
        # Only the first device participates in the broadcast of initial values.
        group_key = self._collective_keys.get_group_key([device])
        group_size = self._num_workers
        collective_instance_key = self._collective_keys.get_instance_key(
            group_key, device
        )

        with ops.device(device):
          initial_value = kwargs["initial_value"]
          if callable(initial_value):
            initial_value = initial_value()
          if isinstance(initial_value, base.CheckpointInitialValue):
            initial_value = initial_value.wrapped_value
          assert not callable(initial_value)
          initial_value = ops.convert_to_tensor(
              initial_value, dtype=kwargs.get("dtype", None)
          )

          if self._num_workers > 1:
            if self._is_chief:
              bcast_send = collective_ops.broadcast_send(
                  initial_value,
                  initial_value.shape,
                  initial_value.dtype,
                  group_size,
                  group_key,
                  collective_instance_key,
              )
              with ops.control_dependencies([bcast_send]):
                return array_ops.identity(initial_value)
            else:
              return collective_ops.broadcast_recv(
                  initial_value.shape,
                  initial_value.dtype,
                  group_size,
                  group_key,
                  collective_instance_key,
              )
          return initial_value

      return initial_value_fn
    else:
      return super(
          CollectiveAllReduceExtended, self
      )._get_variable_creator_initial_value(
          replica_id=replica_id,
          device=device,
          primary_var=primary_var,
          **kwargs,
      )

  def _make_input_context(self):
    # [FUNCTION DOC]
    # HELPER: builds the input context for this worker: one input pipeline per worker, with this
    # worker's id in the cluster as the pipeline id and the total replicas in sync. Returns it.
    input_context = distribute_lib.InputContext(
        num_input_pipelines=self._num_workers,
        input_pipeline_id=self._id_in_cluster,
        num_replicas_in_sync=self._num_replicas_in_sync,
    )
    return input_context

  def _experimental_distribute_dataset(self, dataset, options):
    # [FUNCTION DOC]
    # Distributes a dataset across workers and local replicas. Takes the dataset and input
    # options (may be nothing). Raises a not-implemented error if the replication mode is
    # per-replica (only supported when distributing from a function). Returns a distributed
    # dataset sharded by this worker's input context.
    if (
        options
        and options.experimental_replication_mode
        == distribute_lib.InputReplicationMode.PER_REPLICA
    ):
      raise NotImplementedError(
          "InputReplicationMode.PER_REPLICA "
          "is only supported in "
          "`distribute_datasets_from_function` "
          "of tf.distribute.MirroredStrategy"
      )
    input_context = self._make_input_context()
    return input_util.get_distributed_dataset(
        dataset,
        self._input_workers_with_options(options),
        self._container_strategy(),
        num_replicas_in_sync=self._num_replicas_in_sync,
        input_context=input_context,
        options=options,
    )

  def _distribute_datasets_from_function(self, dataset_fn, options):
    # [FUNCTION DOC]
    # Creates a distributed dataset by calling a user dataset function with this worker's input
    # context. Takes the dataset function and input options (may be nothing). Raises a
    # not-implemented error for the per-replica replication mode. Returns the distributed
    # datasets object.
    # KNOWN ISSUE: the error message names the distribute-from-function API of the mirrored
    # strategy as the only supporter although this is a multi-worker strategy.
    if (
        options
        and options.experimental_replication_mode
        == distribute_lib.InputReplicationMode.PER_REPLICA
    ):
      raise NotImplementedError(
          "InputReplicationMode.PER_REPLICA "
          "is only supported in "
          "`distribute_datasets_from_function` "
          "of tf.distribute.MirroredStrategy"
      )
    input_context = self._make_input_context()
    return input_util.get_distributed_datasets_from_function(
        dataset_fn=dataset_fn,
        input_workers=self._input_workers_with_options(options),
        input_contexts=[input_context],
        strategy=self._container_strategy(),
        options=options,
    )

  def _experimental_distribute_values_from_function(self, value_fn):
    # [FUNCTION DOC]
    # Builds per-replica values by calling a user function once per local replica. Takes a
    # function receiving a value context (global replica id and total replicas in sync). The
    # global replica id is this worker's cluster index times local replica count plus the local
    # index. Returns the values regrouped into a per-replica structure (always wrapped).
    per_replica_values = []
    num_local_replicas = len(self.worker_devices)
    for local_replica_id in range(num_local_replicas):
      replica_id = self._id_in_cluster * num_local_replicas + local_replica_id
      value_context = distribute_lib.ValueContext(
          replica_id, self._num_replicas_in_sync
      )
      per_replica_values.append(value_fn(value_context))
    return distribute_utils.regroup(per_replica_values, always_wrap=True)

  def _make_dataset_iterator(self, dataset):
    # [FUNCTION DOC]
    # V1 API support: wraps a dataset in a distributed dataset iterator sharded by this worker's
    # input context. Takes the dataset; returns the iterator.
    """Distributes the dataset to each local GPU."""
    input_context = self._make_input_context()
    return input_lib_v1.DatasetIterator(
        dataset,
        self._input_workers,
        self._container_strategy(),
        num_replicas_in_sync=self._num_replicas_in_sync,
        input_context=input_context,
    )

  def _make_input_fn_iterator(
      self,
      input_fn,
      replication_mode=distribute_lib.InputReplicationMode.PER_WORKER,
  ):
    # [FUNCTION DOC]
    # V1 API support: builds an input-function iterator. Takes the input function and a
    # replication mode (default per-worker; the argument is not used). Calls the function with
    # this worker's input context and returns the iterator.
    """Distributes the input function to each local GPU."""
    input_context = self._make_input_context()
    return input_lib_v1.InputFunctionIterator(
        input_fn,
        self._input_workers,
        [input_context],
        self._container_strategy(),
    )

  def _configure(
      self, session_config=None, cluster_spec=None, task_type=None, task_id=None
  ):
    # [FUNCTION DOC]
    # Reconfigures the object, usually from a standalone client. Parameters: session config,
    # cluster spec (dictionary, cluster definition or cluster spec), task type and task id; all
    # optional. If a cluster spec is given, builds a simple resolver from it (reusing the current
    # local device counts and rpc layer), re-initializes as multi-worker, and asserts the
    # cross-device ops stay of the collective all-reduce kind. Then (see following code) updates
    # the session config. Raises a value error if the task type is not in the cluster spec.
    """Configures the object.

    Args:
      session_config: a `tf.compat.v1.ConfigProto`
      cluster_spec: a dict, ClusterDef or ClusterSpec object specifying the
        cluster configurations.
      task_type: the current task type, such as "worker".
      task_id: the current task id.

    Raises:
      ValueError: if `task_type` is not in the `cluster_spec`.
    """
    if cluster_spec:
      cluster_resolver = cluster_resolver_lib.SimpleClusterResolver(
          cluster_spec=multi_worker_util.normalize_cluster_spec(cluster_spec),
          task_type=task_type,
          task_id=task_id,
          num_accelerators={
              self._local_device_type: self._num_devices_per_worker
          },
          rpc_layer=self._rpc_layer,
      )
      self._initialize_multi_worker(cluster_resolver)
      assert isinstance(
          self._cross_device_ops, cross_device_ops_lib.CollectiveAllReduce
      )

    if session_config:
      session_config.CopyFrom(self._update_config_proto(session_config))

  def _update_config_proto(self, config_proto):
    # [FUNCTION DOC]
    # Returns an updated deep copy of a session configuration; the input is not mutated. Turns on
    # the scoped-allocator graph optimization for collective reduce ops only. In graph mode with
    # the NCCL implementation requested, enables the NCCL collective flag. If a cluster spec
    # exists, asserts the task type and id are set, sets the collective group leader and replaces
    # the device filters with one restricting communication to this task. Returns the new config.
    updated_config = copy.deepcopy(config_proto)
    # Enable the scoped allocator optimization for CollectiveOps.  This
    # optimization converts many small all-reduces into fewer larger
    # all-reduces.
    rewrite_options = updated_config.graph_options.rewrite_options
    rewrite_options.scoped_allocator_optimization = (
        rewriter_config_pb2.RewriterConfig.ON
    )
    # We turn on ScopedAllocator only for CollectiveReduce op, i.e. enable_op =
    # ["CollectiveReduce"].  Since we can't assign to a repeated proto field, we
    # clear and then append.
    del rewrite_options.scoped_allocator_opts.enable_op[:]
    rewrite_options.scoped_allocator_opts.enable_op.append("CollectiveReduce")

    if (
        not ops.executing_eagerly_outside_functions()
        and self._communication_options.implementation
        == collective_util.CommunicationImplementation.NCCL
    ):
      updated_config.experimental.collective_nccl = True

    if not self._cluster_spec:
      return updated_config

    assert self._task_type
    assert self._task_id is not None

    # Collective group leader is needed for collective ops to coordinate
    # workers.
    updated_config.experimental.collective_group_leader = (
        multi_worker_util.collective_leader(
            self._cluster_spec, self._task_type, self._task_id
        )
    )

    # The device filters prevent communication between workers.
    del updated_config.device_filters[:]
    updated_config.device_filters.append(
        "/job:%s/task:%d" % (self._task_type, self._task_id)
    )

    return updated_config

  def _get_cross_device_ops(self, value):
    # [FUNCTION DOC]
    # HELPER: picks the cross-device ops for a value. Counts the per-device components of a
    # distributed value (one for anything else). If the count equals the number of local worker
    # devices returns the compute-device ops (which may use NCCL), otherwise the per-host ops.
    # CollectiveAllReduce works on a predefined set of devices. In most cases
    # they should be the compute devices, but certain use cases may reduce host
    # tensors as well (e.g. early stopping). We infer the cross_device_ops to
    # use based on the number of devices, since inputs don't always have device
    # annotations. The compute devices one is preferred since we can potentially
    # leverage NCCL.
    if isinstance(value, values.DistributedValues):
      num_devices = len(value._values)  # pylint: disable=protected-access
    else:
      num_devices = 1
    if num_devices == len(self.worker_devices):
      return self._cross_device_ops
    else:
      return self._host_cross_device_ops

  def _gather_to_implementation(self, value, destinations, axis, options):
    # [FUNCTION DOC]
    # Gathers values across devices and workers. Takes the value, destinations, the axis to
    # concatenate along and communication options. Delegates to the chosen cross-device ops'
    # gather and returns its result.
    return self._get_cross_device_ops(value)._gather(  # pylint: disable=protected-access
        value, destinations=destinations, axis=axis, options=options
    )

  def _reduce_to(self, reduce_op, value, destinations, options):
    # [FUNCTION DOC]
    # Reduces a value to the destinations. Takes the reduce operation, value, destinations and
    # per-call options (merged over the strategy's default options). A mirrored value with a mean
    # reduction is returned unchanged; other mirrored values trip an assertion. A distributed
    # value with a single local device is unwrapped to its only component. A non-distributed
    # value with a single worker is reduced locally by repeating over the local replica count.
    # Everything else is reduced by the chosen cross-device ops. Returns the reduced value.
    if (
        isinstance(value, values.Mirrored)
        and reduce_op == reduce_util.ReduceOp.MEAN
    ):
      return value
    assert not isinstance(value, values.Mirrored)

    if (
        isinstance(value, values.DistributedValues)
        and len(self.worker_devices) == 1
    ):
      value = value.values[0]

    # When there are multiple workers, we need to reduce across workers using
    # collective ops.
    if (
        not isinstance(value, values.DistributedValues)
        and self._num_workers == 1
    ):
      # This function handles reducing values that are not PerReplica or
      # Mirrored values. For example, the same value could be present on all
      # replicas in which case `value` would be a single value or value could
      # be 0.
      return cross_device_ops_lib.reduce_non_distributed_value(
          reduce_op, value, destinations, len(self.worker_devices)
      )
    return self._get_cross_device_ops(value).reduce(
        reduce_op,
        value,
        destinations=destinations,
        options=self._communication_options.merge(options),
    )

  def _replica_ctx_all_reduce(self, reduce_op, value, options=None):
    # [FUNCTION DOC]
    # All-reduce from within a replica context. Takes the reduce operation, value and optional
    # options (default options if nothing). In eager mode defers to the base implementation (which
    # uses merge calls) because sequentially run replicas would deadlock on blocking collectives.
    # In graph mode asserts a replica context exists and launches the collective all-reduce
    # directly for the current replica, without a merge call. Returns the reduced value.
    # KNOWN ISSUE: the options passed on to the cross-device all-reduce are not merged with the
    # strategy's default communication options, unlike the reduce path.
    """Implements `StrategyExtendedV2._replica_ctx_all_reduce`."""
    # This implementation avoids using `merge_call` and just launches collective
    # ops in one replica.
    if options is None:
      options = collective_util.Options()

    if context.executing_eagerly():
      # In eager mode, falls back to the default implementation that uses
      # `merge_call`. Replica functions are running sequentially in eager mode,
      # and due to the blocking nature of collective ops, execution will hang if
      # collective ops are to be launched sequentially.
      return super()._replica_ctx_all_reduce(reduce_op, value, options)

    replica_context = distribute_lib.get_replica_context()
    assert replica_context, (
        "`StrategyExtended._replica_ctx_all_reduce` must be called in a "
        "replica context"
    )
    return self._cross_device_ops._all_reduce(  # pylint: disable=protected-access
        reduce_op,
        value,
        replica_context._replica_id,  # pylint: disable=protected-access
        options,
    )

  def _check_health(self):
    # [FUNCTION DOC]
    # HELPER: body of the background health-check thread. Loops until the stop event is set:
    # for every task in the cluster it asks the runtime to check the peer's health with the
    # per-check timeout. Unavailable, failed-precondition or deadline errors are retried up to the
    # retry limit; after that, or on any unexpected error, it logs the problem, aborts collective
    # ops with an unavailable (or internal) status and ends the thread. After a full pass it sleeps
    # for the check interval. Returns nothing.
    while True:
      if self._check_health_thread_should_stop.is_set():
        return
      for job in self._cluster_spec.jobs:
        for task_id in range(self._cluster_spec.num_tasks(job)):
          peer = "/job:{}/replica:0/task:{}".format(job, task_id)
          attempts = 0
          while True:
            attempts += 1
            try:
              context.context().check_collective_ops_peer_health(
                  peer, timeout_in_ms=self._check_health_timeout * 1000
              )
              # If check_collective_ops_peer_health doesn't raise an Exception,
              # the peer is healthy.
              break
            except (
                errors.UnavailableError,
                errors.FailedPreconditionError,
                errors.DeadlineExceededError,
            ) as e:
              # TODO(b/151232436): Always raise UnavailableError when a peer
              # fails. Now there could be many kinds of errors:
              # - Unavailable: when the peer is not reachable, e.g. it's down.
              # - FailedPrecondition: when the peer has restarted.
              if attempts < self._check_health_retry_limit:
                logging.warning(
                    "%s seems down, retrying %d/%d",
                    peer,
                    attempts,
                    self._check_health_retry_limit,
                )
                continue
              logging.error(
                  "Cluster check alive failed, %s is down, "
                  "aborting collectives: %s",
                  peer,
                  e,
              )
              context.context().abort_collective_ops(
                  errors.UNAVAILABLE,
                  "cluster check alive failed, {} is down".format(peer),
              )
              return
            except Exception as e:  # pylint: disable=broad-except
              logging.error("Unexpected exception in check alive: %s", e)
              context.context().abort_collective_ops(
                  errors.INTERNAL, "unexpected exception in check alive: %s" % e
              )
              return
      time.sleep(self._check_health_interval)

  def _start_check_health_thread(self):
    # [FUNCTION DOC]
    # HELPER: waits for the cluster and starts health checking. First runs a dummy sum reduction
    # over the per-host ops (ring implementation, initial timeout) as a barrier so all workers are
    # up, waiting for async work in async mode. Raises a runtime error if that times out. Then
    # creates the stop event and starts the daemon health-check thread. Returns nothing.
    # Use a dummy all-reduce as a barrier to wait for all workers to be up,
    # otherwise the check health may fail immediately.

    # Use array_ops.identity to create the dummy tensor so that we have a new
    # Tensor. If we use constant it may be a cached from on a /job:localhost
    # device, which will cause some code that relies on tensor.device to error.
    #
    # TODO(b/151232436): change to an explicit barrier if we have it.
    dummy_value = array_ops.identity([])
    logging.info(
        "Waiting for the cluster, timeout = %s",
        self._check_health_initial_timeout or "inf",
    )
    try:
      self._host_cross_device_ops.reduce(
          reduce_util.ReduceOp.SUM,
          dummy_value,
          dummy_value,
          options=collective_util.Options(
              timeout_seconds=self._check_health_initial_timeout,
              implementation=collective_util.CommunicationImplementation.RING,
          ),
      )
      if context.is_async():
        context.async_wait()
    except errors.DeadlineExceededError:
      raise RuntimeError(
          "Timeout waiting for the cluster, timeout is %d seconds"
          % self._check_health_initial_timeout
      )
    logging.info("Cluster is ready.")
    self._check_health_thread_should_stop = threading.Event()
    # Start the thread as daemon to avoid it blocking the program from exiting.
    # We try best to shutdown the thread but __del__ is not guaranteed to be
    # called when program exists.
    self._check_health_thread = threading.Thread(
        target=self._check_health, daemon=True
    )
    self._check_health_thread.start()

  def _stop_check_health_thread(self):
    # [FUNCTION DOC]
    # HELPER: if a health-check thread is running, signals it to stop, waits for it to finish and
    # clears the reference (so a later deep copy check still sees the attribute). Does nothing
    # otherwise. Returns nothing.
    if getattr(self, "_check_health_thread", None):
      logging.info("stopping check health thread")
      self._check_health_thread_should_stop.set()
      self._check_health_thread.join()
      self._check_health_thread = None
      logging.info("check health thread stopped")

  def _warn_nccl_no_gpu(self):
    # [FUNCTION DOC]
    # HELPER: logs a warning when the NCCL implementation is requested but the local device type
    # is not GPU. Returns nothing.
    if (
        self._communication_options.implementation
        == collective_util.CommunicationImplementation.NCCL
    ) and self._local_device_type != "GPU":
      logging.warning(
          "Enabled NCCL communication but no GPUs detected/specified."
      )

  def _in_multi_worker_mode(self):
    # [FUNCTION DOC]
    # Returns true when there is more than one worker.
    """Whether this strategy indicates working in multi-worker settings."""
    return self._num_workers > 1

  @property
  def experimental_between_graph(self):
    # [FUNCTION DOC]
    # Property: always true; each worker builds its own graph (between-graph replication).
    return True

  @property
  def experimental_should_init(self):
    # [FUNCTION DOC]
    # Property: always true; every worker should run variable initialization.
    return True

  @property
  def should_checkpoint(self):
    # [FUNCTION DOC]
    # Property: true only on the chief worker.
    return self._is_chief

  @property
  def should_save_summary(self):
    # [FUNCTION DOC]
    # Property: true only on the chief worker.
    return self._is_chief

  @property
  def _num_replicas_in_sync(self):
    # [FUNCTION DOC]
    # Property: total replicas across the cluster, i.e. local devices times number of workers.
    return len(self.worker_devices) * self._num_workers

  # TODO(priyag): Delete this once all strategies use global batch size.
  @property
  def _global_batch_size(self):
    # [FUNCTION DOC]
    # Property: always true, meaning datasets built by the dataset and numpy iterator helpers use
    # a global batch size (input-function iterators assume per-replica batching).
    """`make_dataset_iterator` and `make_numpy_iterator` use global batch size.

    `make_input_fn_iterator` assumes per-replica batching.

    Returns:
      Boolean.
    """
    return True

  def _get_replica_id_in_sync_group(self, replica_id):
    # [FUNCTION DOC]
    # Converts a local replica index to a global one: this worker's cluster index times the local
    # device count plus the local index. Returns an integer.
    return self._id_in_cluster * len(self.worker_devices) + replica_id

  def _get_local_replica_id(self, replica_id_in_sync_group):
    # [FUNCTION DOC]
    # Inverse of the global-replica-id conversion: subtracts this worker's offset (cluster index
    # times local device count) from a global replica index. Returns the local index.
    return replica_id_in_sync_group - self._id_in_cluster * len(
        self.worker_devices
    )

  def __deepcopy__(self, memo):
    # [FUNCTION DOC]
    # Custom deep copy. Takes the memo mapping of already-copied objects. Raises a value error if
    # the object has (ever had) a health-check thread attribute, i.e. in eager multi-worker mode.
    # Otherwise creates a new instance without running the constructor, registers it in the memo
    # and deep-copies every attribute into it. Returns the copy.
    # KNOWN ISSUE: the check tests only whether the attribute exists; after the thread is stopped
    # the attribute is set to nothing but still exists, so copying remains refused.
    # We check the check health thread instead of whether we are in eager mode
    # to limit the backward incompatibility.
    if hasattr(self, "_check_health_thread"):
      raise ValueError(
          "MultiWorkerMirroredStrategy cannot be deep copied in eager mode."
      )
    # Otherwise, do a regular deepcopy.
    cls = self.__class__
    result = cls.__new__(cls)
    memo[id(self)] = result
    for k, v in self.__dict__.items():
      setattr(result, k, copy.deepcopy(v, memo))
    return result
