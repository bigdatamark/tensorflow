# [MODULE DOC]
# Python interface for creating in-process TensorFlow servers for distributed training.
# Exposes the Server class (public as the distribute server and the train server in the v1 API),
# the ClusterSpec class (describes jobs and tasks of a cluster as a mapping from job names to
# network addresses) and the ClusterDeviceFilters class (per-task device visibility filters).
# A private helper builds a server definition protocol buffer from several accepted input forms.
# Server wraps the native server created through the session C API bindings. Import-time side
# effects: only the public-API export registrations and the deprecation registration for the
# train-server endpoint.
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
"""A Python interface for creating TensorFlow servers."""

from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import device_filters_pb2
from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.framework import errors
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


def _make_server_def(server_or_cluster_def, job_name, task_index, protocol,
                     config):
  # [FUNCTION DOC]
  # HELPER: Used by Server's constructor to normalize its arguments into a server definition.
  # KNOWN ISSUE: Docstring says the argument may be a ClusterDef protocol buffer; the code only
  # special-cases a ServerDef and otherwise passes the argument to the ClusterSpec constructor
  # (which does accept a ClusterDef, a ClusterSpec or a dictionary, so this works indirectly).
  # Builds a server definition. Inputs: server_or_cluster_def (a server definition, or anything
  # a ClusterSpec can be built from), job_name, task_index, protocol and config (all may be
  # nothing/null meaning "not given"). If a server definition is given, it is copied and any
  # non-null overrides are applied (the session config is merged in). Otherwise a ClusterSpec is
  # built (a TypeError is raised if conversion fails); a missing job_name is inferred only when
  # the cluster has exactly one job, a missing task_index only when that job has exactly one
  # task, and a missing protocol defaults to grpc; a ValueError is raised when inference is
  # impossible. A non-null config is merged into the default session config. Returns the new
  # server definition; the input is never mutated.
  """Creates a `tf.train.ServerDef` protocol buffer.

  Args:
    server_or_cluster_def: A `tf.train.ServerDef` or `tf.train.ClusterDef`
      protocol buffer, or a `tf.train.ClusterSpec` object, describing the server
      to be defined and/or the cluster of which it is a member.
    job_name: (Optional.) Specifies the name of the job of which the server is a
      member. Defaults to the value in `server_or_cluster_def`, if specified.
    task_index: (Optional.) Specifies the task index of the server in its job.
      Defaults to the value in `server_or_cluster_def`, if specified. Otherwise
      defaults to 0 if the server's job has only one task.
    protocol: (Optional.) Specifies the protocol to be used by the server.
      Acceptable values include `"grpc", "grpc+verbs"`. Defaults to the value in
      `server_or_cluster_def`, if specified. Otherwise defaults to `"grpc"`.
    config: (Options.) A `tf.compat.v1.ConfigProto` that specifies default
      configuration options for all sessions that run on this server.

  Returns:
    A `tf.train.ServerDef`.

  Raises:
    TypeError: If the arguments do not have the appropriate type.
    ValueError: If an argument is not specified and cannot be inferred.
  """
  server_def = tensorflow_server_pb2.ServerDef()
  if isinstance(server_or_cluster_def, tensorflow_server_pb2.ServerDef):
    server_def.MergeFrom(server_or_cluster_def)
    if job_name is not None:
      server_def.job_name = job_name
    if task_index is not None:
      server_def.task_index = task_index
    if protocol is not None:
      server_def.protocol = protocol
    if config is not None:
      server_def.default_session_config.MergeFrom(config)
  else:
    try:
      cluster_spec = ClusterSpec(server_or_cluster_def)
    except TypeError:
      raise TypeError("Could not convert `server_or_cluster_def` to a "
                      "`tf.train.ServerDef` or `tf.train.ClusterSpec`.")
    if job_name is None:
      if len(cluster_spec.jobs) == 1:
        job_name = cluster_spec.jobs[0]
      else:
        raise ValueError("Must specify an explicit `job_name`.")
    if task_index is None:
      task_indices = cluster_spec.task_indices(job_name)
      if len(task_indices) == 1:
        task_index = task_indices[0]
      else:
        raise ValueError("Must specify an explicit `task_index`.")
    if protocol is None:
      protocol = "grpc"

    server_def = tensorflow_server_pb2.ServerDef(
        cluster=cluster_spec.as_cluster_def(),
        job_name=job_name,
        task_index=task_index,
        protocol=protocol)
    if config is not None:
      server_def.default_session_config.MergeFrom(config)
  return server_def


@tf_export("distribute.Server", v1=["distribute.Server", "train.Server"])
@deprecation.deprecated_endpoints("train.Server")
class Server:
  # [CLASS DOC]
  # An in-process TensorFlow server that can take part in distributed training as one task of a
  # named job in a cluster. Holds the server definition and the handle to the native server
  # object. Created from a server definition, a cluster definition or a ClusterSpec; started
  # immediately by default; sessions connect to it through its target string. Stopped (but not
  # freed) when the object is garbage collected. Exported publicly as the distribute server, with
  # the train server name deprecated.
  """An in-process TensorFlow server, for use in distributed training.

  A `tf.distribute.Server` instance encapsulates a set of devices and a
  `tf.compat.v1.Session` target that
  can participate in distributed training. A server belongs to a
  cluster (specified by a `tf.train.ClusterSpec`), and
  corresponds to a particular task in a named job. The server can
  communicate with any other server in the same cluster.
  """

  def __init__(self,
               server_or_cluster_def,
               job_name=None,
               task_index=None,
               protocol=None,
               config=None,
               start=True):
    # [FUNCTION DOC]
    # Constructor. Inputs: server_or_cluster_def (server definition, cluster definition,
    # ClusterSpec or dictionary), optional job_name, task_index, protocol and config overriding
    # or completing it, and start (boolean, default true). Builds the server definition, creates
    # the native server from its serialized form and, if start is true, starts it. Raises
    # TypeError or ValueError from definition building, or a TensorFlow op error from the native
    # layer. Side effect: may open network ports and allocate devices.
    """Creates a new server with the given definition.

    The `job_name`, `task_index`, and `protocol` arguments are optional, and
    override any information provided in `server_or_cluster_def`.

    Args:
      server_or_cluster_def: A `tf.train.ServerDef` or `tf.train.ClusterDef`
        protocol buffer, or a `tf.train.ClusterSpec` object, describing the
        server to be created and/or the cluster of which it is a member.
      job_name: (Optional.) Specifies the name of the job of which the server is
        a member. Defaults to the value in `server_or_cluster_def`, if
        specified.
      task_index: (Optional.) Specifies the task index of the server in its job.
        Defaults to the value in `server_or_cluster_def`, if specified.
        Otherwise defaults to 0 if the server's job has only one task.
      protocol: (Optional.) Specifies the protocol to be used by the server.
        Acceptable values include `"grpc", "grpc+verbs"`. Defaults to the value
        in `server_or_cluster_def`, if specified. Otherwise defaults to
        `"grpc"`.
      config: (Options.) A `tf.compat.v1.ConfigProto` that specifies default
        configuration options for all sessions that run on this server.
      start: (Optional.) Boolean, indicating whether to start the server after
        creating it. Defaults to `True`.

    Raises:
      tf.errors.OpError: Or one of its subclasses if an error occurs while
        creating the TensorFlow server.
    """
    self._server_def = _make_server_def(server_or_cluster_def, job_name,
                                        task_index, protocol, config)
    self._server = c_api.TF_NewServer(self._server_def.SerializeToString())
    if start:
      self.start()

  def __del__(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The native server is only stopped, never deleted (clean shutdown is not yet
    # implemented upstream), so the native object leaks on purpose.
    # Finalizer. Stops the native server and clears the handle. Errors are swallowed: an
    # unimplemented-stop error (or any error if the errors module is already gone at interpreter
    # shutdown) and attribute errors from already-collected modules are ignored. Returns nothing.
    # At shutdown, `errors` may have been garbage collected.
    if errors is not None:
      exception = errors.UnimplementedError
    else:
      exception = Exception
    try:
      c_api.TF_ServerStop(self._server)
      # Clean shutdown of servers is not yet implemented, so
      # we leak instead of calling c_api.TF_DeleteServer here.
      # See:
      # https://github.com/tensorflow/tensorflow/blob/0495317a6e9dd4cac577b9d5cf9525e62b571018/tensorflow/core/distributed_runtime/rpc/grpc_server_lib.h#L73
    except AttributeError:
      # At shutdown, `c_api` may have been garbage collected.
      pass
    except exception:
      pass
    self._server = None

  def start(self):
    # [FUNCTION DOC]
    # Starts the native server so it begins serving. No inputs, returns nothing; raises a
    # TensorFlow op error (or subclass) if the native start fails.
    """Starts this server.

    Raises:
      tf.errors.OpError: Or one of its subclasses if an error occurs while
        starting the TensorFlow server.
    """
    c_api.TF_ServerStart(self._server)

  def join(self):
    # [FUNCTION DOC]
    # Blocks the calling thread on the native server until it shuts down (currently forever).
    # No inputs, returns nothing; raises a TensorFlow op error (or subclass) on native failure.
    """Blocks until the server has shut down.

    This method currently blocks forever.

    Raises:
      tf.errors.OpError: Or one of its subclasses if an error occurs while
        joining the TensorFlow server.
    """
    c_api.TF_ServerJoin(self._server)

  @property
  def server_def(self):
    # [FUNCTION DOC]
    # Read-only property returning the server definition (the same object, not a copy) that this
    # server was built from.
    """Returns the `tf.train.ServerDef` for this server.

    Returns:
      A `tf.train.ServerDef` protocol buffer that describes the configuration
      of this server.
    """
    return self._server_def

  @property
  def target(self):
    # [FUNCTION DOC]
    # Read-only property returning the string a session should use as its target to connect to
    # this server, asked from the native server. No side effects.
    """Returns the target for a `tf.compat.v1.Session` to connect to this server.

    To create a
    `tf.compat.v1.Session` that
    connects to this server, use the following snippet:

    ```python
    server = tf.distribute.Server(...)
    with tf.compat.v1.Session(server.target):
      # ...
    ```

    Returns:
      A string containing a session target for this server.
    """
    return c_api.TF_ServerTarget(self._server)

  @staticmethod
  def create_local_server(config=None, start=True):
    # [FUNCTION DOC]
    # Static convenience factory. Inputs: optional session config and start flag (default true).
    # Creates a Server for a one-task cluster in a job named local on the local host, using the
    # grpc protocol and letting the operating system pick a free port. Returns the new Server;
    # errors come from the Server constructor.
    """Creates a new single-process cluster running on the local host.

    This method is a convenience wrapper for creating a
    `tf.distribute.Server` with a `tf.train.ServerDef` that specifies a
    single-process cluster containing a single task in a job called
    `"local"`.

    Args:
      config: (Options.) A `tf.compat.v1.ConfigProto` that specifies default
        configuration options for all sessions that run on this server.
      start: (Optional.) Boolean, indicating whether to start the server after
        creating it. Defaults to `True`.

    Returns:
      A local `tf.distribute.Server`.
    """
    # Specifying port 0 means that the OS will choose a free port for the
    # server.
    return Server({"localhost": ["localhost:0"]},
                  protocol="grpc",
                  config=config,
                  start=start)


@tf_export("train.ClusterSpec")
class ClusterSpec:
  # [CLASS DOC]
  # Immutable-style description of a cluster: a mapping from job names to tasks, where each job
  # maps integer task indices (possibly sparse) to network addresses. Holds an internal nested
  # mapping plus an equivalent cluster definition protocol buffer, both built at construction.
  # Can be created from a dictionary (lists or index-keyed dictionaries), a cluster definition or
  # another ClusterSpec. Supports truthiness (non-empty), equality against dictionaries or other
  # specs, and a readable representation. Exported publicly as the train cluster spec.
  """Represents a cluster as a set of "tasks", organized into "jobs".

  A `tf.train.ClusterSpec` represents the set of processes that
  participate in a distributed TensorFlow computation. Every
  `tf.distribute.Server` is constructed in a particular cluster.

  To create a cluster with two jobs and five tasks, you specify the
  mapping from job names to lists of network addresses (typically
  hostname-port pairs).

  ```python
  cluster = tf.train.ClusterSpec({"worker": ["worker0.example.com:2222",
                                             "worker1.example.com:2222",
                                             "worker2.example.com:2222"],
                                  "ps": ["ps0.example.com:2222",
                                         "ps1.example.com:2222"]})
  ```

  Each job may also be specified as a sparse mapping from task indices
  to network addresses. This enables a server to be configured without
  needing to know the identity of (for example) all other worker
  tasks:

  ```python
  cluster = tf.train.ClusterSpec({"worker": {1: "worker1.example.com:2222"},
                                  "ps": ["ps0.example.com:2222",
                                         "ps1.example.com:2222"]})
  ```
  """

  def __init__(self, cluster):
    # [FUNCTION DOC]
    # Constructor. Input: cluster, one of a dictionary (job name to list or tuple of addresses,
    # or to a dictionary of task index to address with indices converted to integers), a cluster
    # definition protocol buffer, or another ClusterSpec. Builds the internal mapping and the
    # cluster definition (dictionary input builds it; protobuf input is stored by reference,
    # not copied; ClusterSpec input is copied). Raises TypeError for a job whose tasks are
    # neither list nor dictionary, for names or addresses that are neither bytes nor text, or
    # for an unsupported cluster type.
    """Creates a `ClusterSpec`.

    Args:
      cluster: A dictionary mapping one or more job names to (i) a list of
        network addresses, or (ii) a dictionary mapping integer task indices to
        network addresses; or a `tf.train.ClusterDef` protocol buffer.

    Raises:
      TypeError: If `cluster` is not a dictionary mapping strings to lists
        of strings, and not a `tf.train.ClusterDef` protobuf.
    """
    if isinstance(cluster, dict):
      self._cluster_spec = {}
      for job_name, tasks in cluster.items():
        if isinstance(tasks, (list, tuple)):
          job_tasks = {i: task for i, task in enumerate(tasks)}
        elif isinstance(tasks, dict):
          job_tasks = {int(i): task for i, task in tasks.items()}
        else:
          raise TypeError("The tasks for job %r must be a list or a dictionary "
                          "from integers to strings." % job_name)
        self._cluster_spec[job_name] = job_tasks
      self._make_cluster_def()
    elif isinstance(cluster, cluster_pb2.ClusterDef):
      self._cluster_def = cluster
      self._cluster_spec = {}
      for job_def in self._cluster_def.job:
        self._cluster_spec[job_def.name] = {
            i: t for i, t in job_def.tasks.items()
        }
    elif isinstance(cluster, ClusterSpec):
      self._cluster_def = cluster_pb2.ClusterDef()
      self._cluster_def.MergeFrom(cluster.as_cluster_def())
      self._cluster_spec = {}
      for job_def in self._cluster_def.job:
        self._cluster_spec[job_def.name] = {
            i: t for i, t in job_def.tasks.items()
        }
    else:
      raise TypeError("`cluster` must be a dictionary mapping one or more "
                      "job names to lists of network addresses, or a "
                      "`ClusterDef` protocol buffer")

  def __bool__(self):
    # [FUNCTION DOC]
    # Truth value: true if the cluster has at least one job, false for an empty cluster.
    return bool(self._cluster_spec)

  # Python 2.x
  __nonzero__ = __bool__

  def __eq__(self, other):
    # [FUNCTION DOC]
    # KNOWN ISSUE: Compares the internal mapping directly with the other object, so equality
    # works with plain dictionaries in internal form (job to index-keyed dictionary, never
    # lists) and with other specs only through reflected comparison. Defining equality without a
    # hash also makes instances unhashable.
    # Returns whether the internal job-to-tasks mapping equals the other object.
    return self._cluster_spec == other

  def __ne__(self, other):
    # [FUNCTION DOC]
    # Inverse of equality: returns whether the internal job-to-tasks mapping differs from the
    # other object (same comparison caveats as the equality method).
    return self._cluster_spec != other

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns a string showing the class name and the as-dictionary form with job names sorted
    # alphabetically, so the output is deterministic.
    key_values = self.as_dict()
    string_items = [
        repr(k) + ": " + repr(key_values[k]) for k in sorted(key_values)
    ]
    return "ClusterSpec({" + ", ".join(string_items) + "})"

  def as_dict(self):
    # [FUNCTION DOC]
    # Returns a new dictionary from job name to tasks. A job with no tasks maps to an empty
    # dictionary; a job whose task indices are dense (zero up to count minus one) maps to a list
    # of addresses; otherwise it maps to a dictionary from task index to address. No side
    # effects.
    """Returns a dictionary from job names to their tasks.

    For each job, if the task index space is dense, the corresponding
    value will be a list of network addresses; otherwise it will be a
    dictionary mapping (sparse) task indices to the corresponding
    addresses.

    Returns:
      A dictionary mapping job names to lists or dictionaries
      describing the tasks in those jobs.
    """
    ret = {}
    for job in self.jobs:
      task_indices = self.task_indices(job)
      if len(task_indices) == 0:
        ret[job] = {}
        continue
      if max(task_indices) + 1 == len(task_indices):
        # Return a list because the task indices are dense. This
        # matches the behavior of `as_dict()` before support for
        # sparse jobs was added.
        ret[job] = self.job_tasks(job)
      else:
        ret[job] = {i: self.task_address(job, i) for i in task_indices}
    return ret

  def as_cluster_def(self):
    # [FUNCTION DOC]
    # Returns the cluster definition protocol buffer held by this spec (the stored object, not a
    # copy, so callers should not mutate it).
    """Returns a `tf.train.ClusterDef` protocol buffer based on this cluster."""
    return self._cluster_def

  @property
  def jobs(self):
    # [FUNCTION DOC]
    # Read-only property returning a new list of the job names in insertion order of the
    # internal mapping.
    """Returns a list of job names in this cluster.

    Returns:
      A list of strings, corresponding to the names of jobs in this cluster.
    """
    return list(self._cluster_spec.keys())

  def num_tasks(self, job_name):
    # [FUNCTION DOC]
    # Input: job_name (string). Returns the number of tasks defined in that job (for sparse jobs
    # this is the count of defined tasks, not the highest index plus one). Raises a ValueError
    # if the job is not in the cluster.
    """Returns the number of tasks defined in the given job.

    Args:
      job_name: The string name of a job in this cluster.

    Returns:
      The number of tasks defined in the given job.

    Raises:
      ValueError: If `job_name` does not name a job in this cluster.
    """
    try:
      job = self._cluster_spec[job_name]
    except KeyError:
      raise ValueError("No such job in cluster: %r" % job_name)
    return len(job)

  def task_indices(self, job_name):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The docstring's Raises section also mentions a missing task index, but this
    # method has no task index input; it only raises when the job is unknown.
    # Input: job_name (string). Returns a new sorted list of the task indices defined in that
    # job. Raises a ValueError if the job is not in the cluster.
    """Returns a list of valid task indices in the given job.

    Args:
      job_name: The string name of a job in this cluster.

    Returns:
      A list of valid task indices in the given job.

    Raises:
      ValueError: If `job_name` does not name a job in this cluster,
      or no task with index `task_index` is defined in that job.
    """
    try:
      job = self._cluster_spec[job_name]
    except KeyError:
      raise ValueError("No such job in cluster: %r" % job_name)
    return list(sorted(job.keys()))

  def task_address(self, job_name, task_index):
    # [FUNCTION DOC]
    # Inputs: job_name (string) and task_index (non-negative integer). Returns the network
    # address of that task. Raises a ValueError if the job is unknown or the job has no task with
    # that index.
    """Returns the address of the given task in the given job.

    Args:
      job_name: The string name of a job in this cluster.
      task_index: A non-negative integer.

    Returns:
      The address of the given task in the given job.

    Raises:
      ValueError: If `job_name` does not name a job in this cluster,
      or no task with index `task_index` is defined in that job.
    """
    try:
      job = self._cluster_spec[job_name]
    except KeyError:
      raise ValueError("No such job in cluster: %r" % job_name)
    try:
      return job[task_index]
    except KeyError:
      raise ValueError("No task with index %r in job %r" %
                       (task_index, job_name))

  def job_tasks(self, job_name):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The docstring's first line says it returns a mapping from task ID to address,
    # but the code returns a list indexed by task index (gaps filled with nothing/null).
    # KNOWN ISSUE: For a job that has no tasks, taking the maximum of an empty set of indices
    # raises a ValueError with an unrelated message.
    # Input: job_name (string). Returns a list whose length is the highest task index plus one,
    # holding each task's address at its index and null for undefined indices. Raises a
    # ValueError if the job is unknown.
    """Returns a mapping from task ID to address in the given job.

    NOTE: For backwards compatibility, this method returns a list. If
    the given job was defined with a sparse set of task indices, the
    length of this list may not reflect the number of tasks defined in
    this job. Use the `tf.train.ClusterSpec.num_tasks` method
    to find the number of tasks defined in a particular job.

    Args:
      job_name: The string name of a job in this cluster.

    Returns:
      A list of task addresses, where the index in the list
      corresponds to the task index of each task. The list may contain
      `None` if the job was defined with a sparse set of task indices.

    Raises:
      ValueError: If `job_name` does not name a job in this cluster.
    """
    try:
      job = self._cluster_spec[job_name]
    except KeyError:
      raise ValueError("No such job in cluster: %r" % job_name)
    ret = [None for _ in range(max(job.keys()) + 1)]
    for i, task in job.items():
      ret[i] = task
    return ret

  def _make_cluster_def(self):
    # [FUNCTION DOC]
    # HELPER: Used by the constructor when the cluster was given as a dictionary.
    # KNOWN ISSUE: The docstring mentions a cluster_spec argument; the method takes no inputs and
    # reads the internal mapping instead.
    # Builds a fresh cluster definition protocol buffer from the internal mapping and stores it.
    # Jobs and tasks are processed in sorted order so the serialized form is deterministic. Job
    # names and task addresses are converted to bytes; raises a TypeError if any is neither
    # bytes nor text. Sets the stored cluster definition as its side effect; returns nothing.
    """Creates a `tf.train.ClusterDef` based on the given `cluster_spec`.

    Raises:
      TypeError: If `cluster_spec` is not a dictionary mapping strings to lists
        of strings.
    """
    self._cluster_def = cluster_pb2.ClusterDef()

    # NOTE(mrry): Sort by job_name to produce deterministic protobufs.
    for job_name, tasks in sorted(self._cluster_spec.items()):
      try:
        job_name = compat.as_bytes(job_name)
      except TypeError:
        raise TypeError("Job name %r must be bytes or unicode" % job_name)

      job_def = self._cluster_def.job.add()
      job_def.name = job_name

      for i, task_address in sorted(tasks.items()):
        try:
          task_address = compat.as_bytes(task_address)
        except TypeError:
          raise TypeError("Task address %r must be bytes or unicode" %
                          task_address)
        job_def.tasks[i] = task_address


@tf_export("config.experimental.ClusterDeviceFilters")
class ClusterDeviceFilters:
  # [CLASS DOC]
  # Experimental collection of device filters for remote workers in a cluster. Holds a nested
  # mapping from job name to task index to a list of filter strings, plus a lazily built cluster
  # device filters protocol buffer that is invalidated whenever filters change. A worker with
  # filters set only sees devices matching one of them; tasks without filters see everything.
  # Exported publicly through the experimental config namespace.
  """Represent a collection of device filters for the remote workers in cluster.

  NOTE: this is an experimental API and subject to changes.

  Set device filters for selective jobs and tasks. For each remote worker, the
  device filters are a list of strings. When any filters are present, the remote
  worker will ignore all devices which do not match any of its filters. Each
  filter can be partially specified, e.g. "/job:ps", "/job:worker/replica:3",
  etc. Note that a device is always visible to the worker it is located on.

  For example, to set the device filters for a parameter server cluster:

  ```python
  cdf = tf.config.experimental.ClusterDeviceFilters()
  for i in range(num_workers):
    cdf.set_device_filters('worker', i, ['/job:ps'])
  for i in range(num_ps):
    cdf.set_device_filters('ps', i, ['/job:worker'])

  tf.config.experimental_connect_to_cluster(cluster_def,
                                            cluster_device_filters=cdf)
  ```

  The device filters can be partically specified. For remote tasks that do not
  have device filters specified, all devices will be visible to them.
  """

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor with no inputs. Starts with no filters and no cached protocol buffer.
    # `_device_filters` is a dict mapping job names to job device filters.
    # Job device filters further maps task IDs to task device filters.
    # Task device filters are a list of strings, each one is a device filter.
    self._device_filters = {}

    # Serialized protobuf for cluster device filters.
    self._cluster_device_filters = None

  def set_device_filters(self, job_name, task_index, device_filters):
    # [FUNCTION DOC]
    # KNOWN ISSUE: Input validation uses an assertion, so a non-string filter raises an
    # assertion error (and nothing at all when assertions are disabled) instead of a type error.
    # Inputs: job_name (string), task_index (integer) and device_filters (iterable of strings
    # such as partial device names). Stores a copy of the filters for that task, replacing any
    # earlier ones, and clears the cached protocol buffer. Returns nothing.
    """Set the device filters for given job name and task id."""
    assert all(isinstance(df, str) for df in device_filters)
    self._device_filters.setdefault(job_name, {})
    self._device_filters[job_name][task_index] = [df for df in device_filters]
    # Due to updates in data, invalidate the serialized proto cache.
    self._cluster_device_filters = None

  def _as_cluster_device_filters(self):
    # [FUNCTION DOC]
    # HELPER: Internal accessor used when connecting to a cluster with device filters.
    # KNOWN ISSUE: The docstring says a serialized protobuf is returned, but the code returns the
    # protocol buffer message object itself (not serialized bytes).
    # Returns the cached cluster device filters message, building it first if no cache exists
    # (or the cache is empty-valued). Side effect: may populate the cache.
    """Returns a serialized protobuf of cluster device filters."""
    if self._cluster_device_filters:
      return self._cluster_device_filters

    self._make_cluster_device_filters()
    return self._cluster_device_filters

  def _make_cluster_device_filters(self):
    # [FUNCTION DOC]
    # HELPER: Builds the cached message for the internal accessor.
    # Creates a fresh cluster device filters protocol buffer from the stored filters and caches
    # it. Jobs and tasks are visited in sorted order for deterministic output; job names and
    # filters are converted to bytes, raising a TypeError if a name or filter is neither bytes
    # nor text. Sets the cache as its side effect; returns nothing.
    """Creates `ClusterDeviceFilters` proto based on the `_device_filters`.

    Raises:
      TypeError: If `_device_filters` is not a dictionary mapping strings to
      a map of task indices and device filters.
    """
    self._cluster_device_filters = device_filters_pb2.ClusterDeviceFilters()

    # Sort by job_name to produce deterministic protobufs.
    for job_name, tasks in sorted(self._device_filters.items()):
      try:
        job_name = compat.as_bytes(job_name)
      except TypeError:
        raise TypeError("Job name %r must be bytes or unicode" % job_name)

      jdf = self._cluster_device_filters.jobs.add()
      jdf.name = job_name

      for i, task_device_filters in sorted(tasks.items()):
        for tdf in task_device_filters:
          try:
            tdf = compat.as_bytes(tdf)
          except TypeError:
            raise TypeError("Device filter %r must be bytes or unicode" % tdf)
          jdf.tasks[i].device_filters.append(tdf)
