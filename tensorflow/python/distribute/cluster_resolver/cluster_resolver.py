# [MODULE DOC]
# Defines the cluster resolver abstraction used by distribution strategies to discover cluster
# membership: the abstract base class, a simple resolver holding explicitly supplied values, and a
# union resolver merging several resolvers. Also provides two helper functions (master URL
# formatting and accelerator device listing). The classes are exported in the public
# distribute cluster resolver namespace through the export decorator (import-time registration).
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Cluster Resolvers are used for dynamic cluster IP/hostname resolution."""

import abc

import collections

import six

from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import ops
from tensorflow.python.training.server_lib import ClusterSpec
from tensorflow.python.util.tf_export import tf_export


def format_master_url(master, rpc_layer=None):
  # [FUNCTION DOC]
  # HELPER: formats a session master address, used by the resolvers when building master strings.
  # Inputs: master (address string) and optional rpc_layer (protocol name). Returns the address
  # prefixed with the protocol and a separator when a protocol is given, otherwise the address
  # unchanged.
  if rpc_layer:
    return '%s://%s' % (rpc_layer, master)
  else:
    return master


def get_accelerator_devices(master, config_proto):
  # [FUNCTION DOC]
  # HELPER: lists accelerator devices for the base resolver's accelerator counting.
  # Inputs: master (session master address) and config_proto (optional session configuration).
  # In eager mode, returns logical devices from the runtime configuration, excluding CPU and XLA
  # CPU types, wrapped as device attribute records with zero memory sizes. Otherwise, opens a
  # temporary session in a fresh graph against the master, and returns its listed devices.
  # Side effect: creates and closes a session in graph mode (may contact the master).
  """Returns accelerator devices given a master and a configuration."""
  if context.executing_eagerly():
    logical_devices = config.list_logical_devices()
    devices = []
    for d in logical_devices:
      if d.device_type == 'CPU' or d.device_type == 'XLA_CPU':  # Filter CPUs
        continue
      devices.append(session._DeviceAttributes(d.name, d.device_type, 0, 0))  # pylint: disable=protected-access
    return devices
  else:
    with ops.Graph().as_default():
      with session.Session(master, config=config_proto) as s:
        devices = s.list_devices()
    return devices


@tf_export('distribute.cluster_resolver.ClusterResolver')
@six.add_metaclass(abc.ABCMeta)
class ClusterResolver(object):
  # [CLASS DOC]
  # Abstract base class for cluster resolvers, which tell TensorFlow about the machines in a
  # distributed cluster. Subclasses must implement the cluster specification and master address
  # methods. The base class provides accelerator counting, an environment string (empty by
  # default), and task type, task id properties backed by private attributes that subclasses may
  # set. Uses an abstract-base-class metaclass and is exported publicly.
  """Abstract class for all implementations of ClusterResolvers.

  This defines the skeleton for all implementations of ClusterResolvers.
  ClusterResolvers are a way for TensorFlow to communicate with various cluster
  management systems (e.g. GCE, AWS, etc...) and gives TensorFlow necessary
  information to set up distributed training.

  By letting TensorFlow communicate with these systems, we will be able to
  automatically discover and resolve IP addresses for various TensorFlow
  workers. This will eventually allow us to automatically recover from
  underlying machine failures and scale TensorFlow worker clusters up and down.

  Note to Implementors of `tf.distribute.cluster_resolver.ClusterResolver`
  subclass: In addition to these abstract methods, when task_type, task_id, and
  rpc_layer attributes are applicable, you should also implement them either as
  properties with getters or setters, or directly set the attributes
  `self._task_type`, `self._task_id`, or `self._rpc_layer` so the base class'
  getters and setters are used. See
  `tf.distribute.cluster_resolver.SimpleClusterResolver.__init__` for an
  example.

  In general, multi-client tf.distribute strategies such as
  `tf.distribute.experimental.MultiWorkerMirroredStrategy` require task_type and
  task_id properties to be available in the `ClusterResolver` they are using. On
  the other hand, these concepts are not applicable in single-client strategies,
  such as `tf.distribute.experimental.TPUStrategy`, because the program is only
  expected to be run on one task, so there should not be a need to have code
  branches according to task type and task id.

  - task_type is the name of the server's current named job (e.g. 'worker',
     'ps' in a distributed parameterized training job).
  - task_id is the ordinal index of the server within the task type.
  - rpc_layer is the protocol used by TensorFlow to communicate with other
      TensorFlow servers in a distributed environment.
  """

  @abc.abstractmethod
  def cluster_spec(self):
    # [FUNCTION DOC]
    # Abstract: returns the current cluster specification object. Implementations should query the
    # underlying cluster manager on every call rather than caching. The base version raises a
    # not-implemented error.
    """Retrieve the current state of the cluster and return a `tf.train.ClusterSpec`.

    Returns:
      A `tf.train.ClusterSpec` representing the state of the cluster at the
      moment this function is called.

    Implementors of this function must take care in ensuring that the
    ClusterSpec returned is up-to-date at the time of calling this function.
    This usually means retrieving the information from the underlying cluster
    management system every time this function is invoked and reconstructing
    a cluster_spec, rather than attempting to cache anything.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def master(self, task_type=None, task_id=None, rpc_layer=None):
    # [FUNCTION DOC]
    # Abstract: returns the session master name or URL (only meaningful for TF 1.x). Optional
    # inputs: task type, task id, and rpc layer to target a particular task and protocol. The base
    # version raises a not-implemented error.
    """Retrieves the name or URL of the session master.

    Note: this is only useful for TensorFlow 1.x.

    Args:
      task_type: (Optional) The type of the TensorFlow task of the master.
      task_id: (Optional) The index of the TensorFlow task of the master.
      rpc_layer: (Optional) The RPC protocol for the given cluster.

    Returns:
      The name or URL of the session master.

    Implementors of this function must take care in ensuring that the master
    returned is up-to-date at the time to calling this function. This usually
    means retrieving the master every time this function is invoked.
    """
    raise NotImplementedError()

  def num_accelerators(self,
                       task_type=None,
                       task_id=None,
                       config_proto=None):
    # [FUNCTION DOC]
    # Returns a mapping from accelerator device type to the number of such devices per worker.
    # Optional inputs: task_type and task_id (when both are given only devices whose names belong
    # to that job and task are counted; the master for that task is used) and config_proto (session
    # configuration used when a session must be created). Uses the master address, lists the
    # accelerator devices and counts them by type.
    """Returns the number of accelerator cores per worker.

    This returns the number of accelerator cores (such as GPUs and TPUs)
    available per worker.

    Optionally, we allow callers to specify the task_type, and task_id, for
    if they want to target a specific TensorFlow task to query
    the number of accelerators. This is to support heterogenous environments,
    where the number of accelerators cores per host is different.

    Args:
      task_type: (Optional) The type of the TensorFlow task of the machine we
        want to query.
      task_id: (Optional) The index of the TensorFlow task of the machine we
        want to query.
      config_proto: (Optional) Configuration for starting a new session to
        query how many accelerator cores it has.

    Returns:
      A map of accelerator types to number of cores.
    """
    master = self.master(task_type, task_id)
    # TODO(b/126786766): in eager mode, we should check whether
    # `tf.config.experimental_connect_to_cluster` is called or not.
    devices = get_accelerator_devices(master, config_proto)
    mapping = collections.defaultdict(int)
    for device in devices:
      if task_type is not None and task_id is not None:
        job_path = '/job:%s' % task_type
        task_path = '/task:%s' % task_id
        if job_path not in device.name or task_path not in device.name:
          continue
      mapping[device.device_type] += 1
    return mapping

  @property
  def environment(self):
    # [FUNCTION DOC]
    # Property: returns the environment name. The base class returns an empty string, meaning not
    # a Google-internal environment; subclasses working in both settings should override it.
    """Returns the current environment which TensorFlow is running in.

    There are two possible return values, "google" (when TensorFlow is running
    in a Google-internal environment) or an empty string (when TensorFlow is
    running elsewhere).

    If you are implementing a ClusterResolver that works in both the Google
    environment and the open-source world (for instance, a TPU ClusterResolver
    or similar), you will have to return the appropriate string depending on the
    environment, which you will have to detect.

    Otherwise, if you are implementing a ClusterResolver that will only work
    in open-source TensorFlow, you do not need to implement this property.
    """
    return ''

  @property
  def task_type(self):
    # [FUNCTION DOC]
    # Property: returns the task type (such as chief, worker, parameter server or evaluator) of
    # this resolver, read from the private attribute if a subclass or setter set it, or nothing/null
    # value if unavailable or not applicable.
    """Returns the task type this `ClusterResolver` indicates.

    In TensorFlow distributed environment, each job may have an applicable
    task type. Valid task types in TensorFlow include
    'chief': a worker that is designated with more responsibility,
    'worker': a regular worker for training/evaluation,
    'ps': a parameter server, or
    'evaluator': an evaluator that evaluates the checkpoints for metrics.

    See [Multi-worker configuration](
    https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras#multi-worker_configuration)
    for more information about 'chief' and 'worker' task type, which are most
    commonly used.

    Having access to such information is useful when user needs to run specific
    code according to task types. For example,

    ```python
    cluster_spec = tf.train.ClusterSpec({
        "ps": ["localhost:2222", "localhost:2223"],
        "worker": ["localhost:2224", "localhost:2225", "localhost:2226"]
    })

    # SimpleClusterResolver is used here for illustration; other cluster
    # resolvers may be used for other source of task type/id.
    simple_resolver = SimpleClusterResolver(cluster_spec, task_type="worker",
                                            task_id=1)

    ...

    if cluster_resolver.task_type == 'worker':
      # Perform something that's only applicable on workers. This block
      # will run on this particular instance since we've specified this task to
      # be a worker in above cluster resolver.
    elif cluster_resolver.task_type == 'ps':
      # Perform something that's only applicable on parameter servers. This
      # block will not run on this particular instance.
    ```

    Returns `None` if such information is not available or is not applicable
    in the current distributed environment, such as training with
    `tf.distribute.experimental.TPUStrategy`.

    For more information, please see
    `tf.distribute.cluster_resolver.ClusterResolver`'s class doc.
    """
    return getattr(self, '_task_type', None)

  @property
  def task_id(self):
    # [FUNCTION DOC]
    # Property: returns the task index within its task type, read from the private attribute if
    # set, or nothing/null value if unavailable or not applicable.
    """Returns the task id this `ClusterResolver` indicates.

    In TensorFlow distributed environment, each job may have an applicable
    task id, which is the index of the instance within its task type. This is
    useful when user needs to run specific code according to task index. For
    example,

    ```python
    cluster_spec = tf.train.ClusterSpec({
        "ps": ["localhost:2222", "localhost:2223"],
        "worker": ["localhost:2224", "localhost:2225", "localhost:2226"]
    })

    # SimpleClusterResolver is used here for illustration; other cluster
    # resolvers may be used for other source of task type/id.
    simple_resolver = SimpleClusterResolver(cluster_spec, task_type="worker",
                                            task_id=0)

    ...

    if cluster_resolver.task_type == 'worker' and cluster_resolver.task_id == 0:
      # Perform something that's only applicable on 'worker' type, id 0. This
      # block will run on this particular instance since we've specified this
      # task to be a 'worker', id 0 in above cluster resolver.
    else:
      # Perform something that's only applicable on other ids. This block will
      # not run on this particular instance.
    ```

    Returns `None` if such information is not available or is not applicable
    in the current distributed environment, such as training with
    `tf.distribute.cluster_resolver.TPUClusterResolver`.

    For more information, please see
    `tf.distribute.cluster_resolver.ClusterResolver`'s class docstring.
    """
    return getattr(self, '_task_id', None)

  @task_type.setter
  def task_type(self, task_type):
    # [FUNCTION DOC]
    # Setter for the task type property; stores the given value in the private attribute.
    """Setter of `task_type` property. See `task_type` property doc."""
    self._task_type = task_type

  @task_id.setter
  def task_id(self, task_id):
    # [FUNCTION DOC]
    # Setter for the task id property; stores the given value in the private attribute.
    """Setter of `task_id` property. See `task_type` property doc."""
    self._task_id = task_id


@tf_export('distribute.cluster_resolver.SimpleClusterResolver')
class SimpleClusterResolver(ClusterResolver):
  # [CLASS DOC]
  # Cluster resolver that simply holds values passed to its constructor (cluster specification,
  # master, task type and id, environment, accelerator counts, rpc layer) and returns them.
  # No discovery is performed. Exported publicly.
  """Simple implementation of ClusterResolver that accepts all attributes.

  Please see the base class for documentation of arguments of its constructor.

  It is useful if you want to specify some or all attributes.

  Usage example with `tf.distribute.Strategy`:

    ```Python
    cluster = tf.train.ClusterSpec({"worker": ["worker0.example.com:2222",
                                               "worker1.example.com:2222"]})

    # On worker 0
    cluster_resolver = SimpleClusterResolver(cluster, task_type="worker",
                                             task_id=0,
                                             num_accelerators={"GPU": 8},
                                             rpc_layer="grpc")
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
        cluster_resolver=cluster_resolver)

    # On worker 1
    cluster_resolver = SimpleClusterResolver(cluster, task_type="worker",
                                             task_id=1,
                                             num_accelerators={"GPU": 8},
                                             rpc_layer="grpc")
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
        cluster_resolver=cluster_resolver)
    ```
  """

  def __init__(self, cluster_spec, master='', task_type=None, task_id=None,
               environment='', num_accelerators=None,
               rpc_layer=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: cluster_spec (required cluster specification object), master (string
    # address, default empty), task_type, task_id, num_accelerators (mapping of device type to
    # count) and rpc_layer (all optional, default nothing), and environment (string, default empty).
    # Stores the values. Raises a type error if cluster_spec is not a cluster specification or if
    # master is not a string.
    """Creates a SimpleClusterResolver from a ClusterSpec."""
    super(SimpleClusterResolver, self).__init__()

    self._task_type = task_type
    self._task_id = task_id
    self._environment = environment

    self._num_accelerators = num_accelerators
    self._rpc_layer = rpc_layer

    if not isinstance(cluster_spec, ClusterSpec):
      raise TypeError('cluster_spec must be a `tf.train.ClusterSpec`.')
    self._cluster_spec = cluster_spec

    if not isinstance(master, str):
      raise TypeError('master must be a string.')
    self._master = master

  def cluster_spec(self):
    # [FUNCTION DOC]
    # Returns the cluster specification given to the constructor.
    """Returns the ClusterSpec passed into the constructor."""
    return self._cluster_spec

  def master(self, task_type=None, task_id=None, rpc_layer=None):
    # [FUNCTION DOC]
    # Returns the master address. Optional inputs: task_type and task_id (when both given the
    # address of that task in the cluster specification is used, else the stored master) and
    # rpc_layer (overrides the stored one). The result is formatted with the protocol prefix when a
    # protocol is available.
    """Returns the master address to use when creating a session.

    Note: this is only useful for TensorFlow 1.x.

    Args:
      task_type: (Optional) The type of the TensorFlow task of the master.
      task_id: (Optional) The index of the TensorFlow task of the master.
      rpc_layer: (Optional) The RPC used by distributed TensorFlow.

    Returns:
      The name or URL of the session master.

    If a task_type and task_id is given, this will override the `master`
    string passed into the initialization function.
    """
    if task_type is not None and task_id is not None:
      master = self.cluster_spec().task_address(task_type, task_id)
    else:
      master = self._master

    return format_master_url(master, rpc_layer=rpc_layer or self._rpc_layer)

  @property
  def task_type(self):
    # [FUNCTION DOC]
    # Property: returns the stored task type.
    return self._task_type

  @property
  def task_id(self):
    # [FUNCTION DOC]
    # Property: returns the stored task id.
    return self._task_id

  @task_type.setter
  def task_type(self, task_type):
    # [FUNCTION DOC]
    # Setter: stores the task type.
    self._task_type = task_type

  @task_id.setter
  def task_id(self, task_id):
    # [FUNCTION DOC]
    # Setter: stores the task id.
    self._task_id = task_id

  @property
  def environment(self):
    # [FUNCTION DOC]
    # Property: returns the environment string given to the constructor.
    return self._environment

  def num_accelerators(self,
                       task_type=None,
                       task_id=None,
                       config_proto=None):
    # [FUNCTION DOC]
    # Returns the accelerator count mapping supplied to the constructor, or an empty mapping if
    # none was supplied. All three arguments (task type, task id, config proto) are ignored.
    """Returns the number of accelerator cores per worker.

    The SimpleClusterResolver does not do automatic detection of accelerators,
    and thus all arguments are unused and we simply return the value provided
    in the constructor.

    Args:
      task_type: Unused.
      task_id: Unused.
      config_proto: Unused.
    """
    # Unused
    del task_type, task_id, config_proto
    if self._num_accelerators is None:
      return {}
    return self._num_accelerators

  @property
  def rpc_layer(self):
    # [FUNCTION DOC]
    # Property: returns the stored rpc layer (protocol name, may be nothing/null).
    return self._rpc_layer

  @rpc_layer.setter
  def rpc_layer(self, rpc_layer):
    # [FUNCTION DOC]
    # Setter: stores the rpc layer.
    self._rpc_layer = rpc_layer


@tf_export('distribute.cluster_resolver.UnionResolver')
class UnionClusterResolver(ClusterResolver):
  # [CLASS DOC]
  # Cluster resolver that combines two or more other resolvers. It merges their cluster
  # specifications; other properties (task type, task id, rpc layer, environment, master, and
  # accelerator counts) come from the first resolver unless overridden through constructor
  # arguments. Holds the overrides and the tuple of resolvers. Exported publicly under the name
  # union resolver.
  """Performs a union on underlying ClusterResolvers.

  This class performs a union given two or more existing ClusterResolvers. It
  merges the underlying ClusterResolvers, and returns one unified ClusterSpec
  when cluster_spec is called. The details of the merge function is
  documented in the cluster_spec function.

  For additional ClusterResolver properties such as task type, task index,
  rpc layer, environment, etc..., we will return the value from the first
  ClusterResolver in the union.

  An example to combine two cluster resolvers:

    ```Python
    cluster_0 = tf.train.ClusterSpec({"worker": ["worker0.example.com:2222",
                                                 "worker1.example.com:2222"]})
    cluster_resolver_0 = SimpleClusterResolver(cluster, task_type="worker",
                                               task_id=0,
                                               rpc_layer="grpc")

    cluster_1 = tf.train.ClusterSpec({"ps": ["ps0.example.com:2222",
                                             "ps1.example.com:2222"]})
    cluster_resolver_1 = SimpleClusterResolver(cluster, task_type="ps",
                                               task_id=0,
                                               rpc_layer="grpc")

    # Its task type would be "worker".
    cluster_resolver = UnionClusterResolver(cluster_resolver_0,
                                            cluster_resolver_1)
    ```

  An example to override the number of GPUs in a TFConfigClusterResolver
  instance:

    ```Python
    tf_config = TFConfigClusterResolver()
    gpu_override = SimpleClusterResolver(tf_config.cluster_spec(),
                                         num_accelerators={"GPU": 1})
    cluster_resolver = UnionResolver(gpu_override, tf_config)
    ```
  """

  def __init__(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Constructor. Inputs: one or more resolver objects (positional), plus optional keyword
    # overrides rpc_layer, task_type and task_id (default nothing). Stores them. Raises a value
    # error if any other keyword is given or if no resolver is passed, and a type error if any
    # positional argument is not a cluster resolver.
    """Initializes a UnionClusterResolver with other ClusterResolvers.

    Args:
      *args: `ClusterResolver` objects to be unionized.
      **kwargs:
        rpc_layer - (Optional) Override value for the RPC layer used by
          TensorFlow.
        task_type - (Optional) Override value for the current task type.
        task_id - (Optional) Override value for the current task index.

    Raises:
      TypeError: If any argument is not a subclass of `ClusterResolvers`.
      ValueError: If there are no arguments passed.
    """
    super(UnionClusterResolver, self).__init__()

    self._rpc_layer = kwargs.pop('rpc_layer', None)
    self._task_type = kwargs.pop('task_type', None)
    self._task_id = kwargs.pop('task_id', None)

    if kwargs:
      raise ValueError('Unexpected kwargs provided {!r}'.format(kwargs))

    if not args:
      raise ValueError('At least one ClusterResolver is required.')

    for cluster_resolver in args:
      if not isinstance(cluster_resolver, ClusterResolver):
        raise TypeError('All arguments must be a sub-class of '
                        '`ClusterResolver.`')
    self._cluster_resolvers = args

  def cluster_spec(self):
    # [FUNCTION DOC]
    # Returns the merged cluster specification. First determines, for each job name, whether the
    # result is a list (all lists) or a mapping (if any resolver reports a mapping), then merges:
    # lists are concatenated in resolver order; mappings are combined (lists converted by
    # position index). Raises a key error when two resolvers provide the same task index for a job
    # in mapping mode. Queries each resolver's cluster specification twice.
    """Returns a union of all the ClusterSpecs from the ClusterResolvers.

    Returns:
      A ClusterSpec containing host information merged from all the underlying
      ClusterResolvers.

    Raises:
      KeyError: If there are conflicting keys detected when merging two or
      more dictionaries, this exception is raised.

    Note: If there are multiple ClusterResolvers exposing ClusterSpecs with the
    same job name, we will merge the list/dict of workers.

    If *all* underlying ClusterSpecs expose the set of workers as lists, we will
    concatenate the lists of workers, starting with the list of workers from
    the first ClusterResolver passed into the constructor.

    If *any* of the ClusterSpecs expose the set of workers as a dict, we will
    treat all the sets of workers as dicts (even if they are returned as lists)
    and will only merge them into a dict if there is no conflicting keys. If
    there is a conflicting key, we will raise a `KeyError`.
    """

    merged_cluster = {}

    # We figure out whether it is all lists for a particular job, or whether
    # there are dicts inside.
    for cluster_resolver in self._cluster_resolvers:
      cluster_spec = cluster_resolver.cluster_spec()
      cluster_dict = cluster_spec.as_dict()

      for job_name, tasks in cluster_dict.items():
        if job_name in merged_cluster:
          # If we see a dict, then we write a dict out regardless.
          if isinstance(tasks, dict):
            merged_cluster[job_name] = {}
        else:
          # We take whichever type is present.
          if isinstance(tasks, list):
            merged_cluster[job_name] = []
          else:
            merged_cluster[job_name] = {}

    # We then do the merge as appropriate in merged_cluster[job].
    for cluster_resolver in self._cluster_resolvers:
      cluster_spec = cluster_resolver.cluster_spec()
      cluster_dict = cluster_spec.as_dict()

      for job_name, tasks in cluster_dict.items():
        if isinstance(merged_cluster[job_name], list):
          # We all have lists, we can just concatenate and be done.
          merged_cluster[job_name].extend(tasks)
        else:
          if isinstance(tasks, list):
            # We convert to a dictionary if the type is a list.
            task_dict = dict(zip(range(0, len(tasks)), tasks))
          else:
            # We can simply make a copy (for update) and be done.
            task_dict = tasks.copy()

          # We detect if there are duplicates, and raise an error if so.
          task_keys = set(task_dict)
          merged_keys = set(merged_cluster[job_name].keys())
          intersected_keys = task_keys.intersection(merged_keys)
          if intersected_keys:
            raise KeyError('Duplicate keys detected when merging two '
                           'ClusterSpecs: %s' % repr(intersected_keys))

          # We do the merge after all the processing.
          merged_cluster[job_name].update(task_dict)

    return ClusterSpec(merged_cluster)

  def master(self, task_type=None, task_id=None, rpc_layer=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: when no task is specified, the first resolver's master is used with only the
    # rpc_layer argument; this union's own overriding rpc layer is not applied in that case.
    # Returns the master address. If both task_type and task_id are given, uses the merged cluster
    # specification's address for that task, formatted with the rpc layer argument or this union's
    # rpc layer. Otherwise returns the first resolver's master.
    """Returns the master address to use when creating a session.

    This usually returns the master from the first ClusterResolver passed in,
    but you can override this by specifying the task_type and task_id.

    Note: this is only useful for TensorFlow 1.x.

    Args:
      task_type: (Optional) The type of the TensorFlow task of the master.
      task_id: (Optional) The index of the TensorFlow task of the master.
      rpc_layer: (Optional) The RPC protocol for the given cluster.

    Returns:
      The name or URL of the session master.
    """
    if task_type is not None and task_id is not None:
      master = self.cluster_spec().task_address(task_type, task_id)
      return format_master_url(master, rpc_layer or self._rpc_layer)

    return self._cluster_resolvers[0].master(rpc_layer=rpc_layer)

  @property
  def task_type(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: uses a truthiness test, so an override that is an empty string is ignored (task
    # type is normally a non-empty string, so this rarely matters).
    # Property: returns the override task type if set, else the first resolver's task type.
    return self._task_type or self._cluster_resolvers[0].task_type

  @property
  def task_id(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: uses a truthiness test, so an override task id of zero is treated as unset and
    # the first resolver's task id is returned instead.
    # Property: returns the override task id if set, else the first resolver's task id.
    return self._task_id or self._cluster_resolvers[0].task_id

  @task_type.setter
  def task_type(self, task_type):
    # [FUNCTION DOC]
    # Setter: stores the task type override.
    self._task_type = task_type

  @task_id.setter
  def task_id(self, task_id):
    # [FUNCTION DOC]
    # Setter: stores the task id override.
    self._task_id = task_id

  @property
  def environment(self):
    # [FUNCTION DOC]
    # Property: returns the first resolver's environment.
    return self._cluster_resolvers[0].environment

  def num_accelerators(self,
                       task_type=None,
                       task_id=None,
                       config_proto=None):
    # [FUNCTION DOC]
    # Returns the first resolver's accelerator counts, forwarding the task type, task id and
    # config proto arguments (all optional).
    return self._cluster_resolvers[0].num_accelerators(
        task_type, task_id, config_proto)

  @property
  def rpc_layer(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: uses a truthiness test, so an empty-string override is treated as unset.
    # Property: returns the override rpc layer if set, else the first resolver's rpc layer.
    return self._rpc_layer or self._cluster_resolvers[0].rpc_layer

  @rpc_layer.setter
  def rpc_layer(self, rpc_layer):
    # [FUNCTION DOC]
    # Setter: stores the rpc layer override.
    self._rpc_layer = rpc_layer
