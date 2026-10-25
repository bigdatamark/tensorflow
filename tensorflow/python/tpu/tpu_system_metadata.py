# [MODULE DOC]
# KNOWN ISSUE: the comment on the pinging timeout constant says ten minutes, but the value is
# five minutes in milliseconds.
# Discovers and describes the TPU system. Exposes the public TPU system metadata record type
# (core count, host count, cores per host, topology, device list), a private query that detects
# the metadata either from the eager context or by listing devices on a TensorFlow master with
# retries, a topology fetcher, a session-config helper with an operation timeout, and a function
# that infers the job name for TPU placement. Defines timeout, retry and default job name
# constants. Importing registers the public export of the metadata type; no other side effects.
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
# ===================================================================
"""TPU system metadata and associated tooling."""

import collections

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.distribute import device_util
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tpu
from tensorflow.python.util.tf_export import tf_export

_PINGING_MASTER_TIMEOUT_IN_MS = 5 * 60 * 1000  # 10 min
_RETRY_TIMES = 12 * 24  # 1 day
_INITIAL_TPU_SYSTEM_TIMEOUT_IN_MS = 300 * 1000  # 5 mins

_DEFAULT_JOB_NAME = 'tpu_worker'
_DEFAULT_COORDINATOR_JOB_NAME = 'coordinator'
_LOCAL_MASTERS = ('', 'local')


@tf_export('tpu.experimental.TPUSystemMetadata')
class TPUSystemMetadata(
    collections.namedtuple('TPUSystemMetadata', [
        'num_cores',
        'num_hosts',
        'num_of_cores_per_host',
        'topology',
        'devices',
    ])):
  # [CLASS DOC]
  # Immutable record (named tuple) describing a TPU system: total core count, host count, cores
  # per host, the physical topology object (may be null) and a tuple of device attributes.
  # Created by the system query function or directly by users; exported as a public API class.
  """Describes some metadata about the TPU system.

  Attributes:
    num_cores: interger. Total number of TPU cores in the TPU system.
    num_hosts: interger. Total number of hosts (TPU workers) in the TPU system.
    num_of_cores_per_host: interger. Number of TPU cores per host (TPU worker).
    topology: an instance of `tf.tpu.experimental.Topology`, which describes the
      physical topology of TPU system.
    devices: a tuple of strings, which describes all the TPU devices in the
      system.
  """

  def __new__(cls, num_cores, num_hosts, num_of_cores_per_host, topology,
              devices):
    # [FUNCTION DOC]
    # Constructor creating the record from the five fields in order: core count, host count,
    # cores per host, topology and devices. No validation. Returns the new instance.
    return super(TPUSystemMetadata,
                 cls).__new__(cls, num_cores, num_hosts, num_of_cores_per_host,
                              topology, devices)


def _query_tpu_system_metadata(master_address, cluster_def=None,
                               query_topology=False):
  # [FUNCTION DOC]
  # HELPER: used by TPU cluster/strategy setup to learn the shape of the TPU system.
  # Detects the TPU system. Inputs: master address (text), optional cluster definition, and a flag
  # asking whether to also fetch the topology (default false). In eager mode the devices come from
  # the logical device list; otherwise a session on the master lists devices, retrying on
  # deadline-exceeded errors up to a large retry limit (then raising a value error) and logging
  # warnings. Counts TPU devices per task, raises a runtime error if hosts have different core
  # counts, and, when the topology is requested, raises a runtime error if no TPU cores exist,
  # otherwise initializes the system to fetch the topology. Devices are sorted by job, replica,
  # task, type and index. Returns a TPUSystemMetadata record and logs a summary.
  """Automatically detects the TPU system metadata in the system."""
  tpu_core_count = 0
  devices = []
  device_dict = collections.defaultdict(list)

  if context.executing_eagerly():
    logical_devices = config.list_logical_devices()

    # We want the output type to match in both eager and session mode
    devices = [session_lib._DeviceAttributes(device_util.canonicalize(d.name),  # pylint: disable=protected-access
                                             d.device_type, 0, 0)
               for d in logical_devices]
  else:
    # TODO(b/120564445): Replace with standard library for retries.
    retry_count = 1
    while True:
      logging.info('Querying Tensorflow master (%s) for TPU system metadata.',
                   master_address)
      try:
        with ops.Graph().as_default():
          with session_lib.Session(
              master_address,
              config=get_session_config_with_timeout(
                  _PINGING_MASTER_TIMEOUT_IN_MS,
                  cluster_def)) as sess:
            devices = sess.list_devices()
            break
      except errors.DeadlineExceededError:
        msg = ('Failed to connect to the Tensorflow master. The TPU worker may '
               'not be ready (still scheduling) or the Tensorflow master '
               'address is incorrect: got (%s).' %
               (master_address))

        # TODO(xiejw): For local or grpc master we might not need retry logic
        # here.
        if retry_count <= _RETRY_TIMES:
          logging.warning('%s', msg)
          logging.warning('Retrying (%d/%d).', retry_count, _RETRY_TIMES)
          retry_count += 1
        else:
          raise ValueError(msg)

  for device in devices:
    spec = tf_device.DeviceSpec.from_string(device.name)
    if spec.device_type == 'TPU':
      device_dict[spec.task].append(spec.device_index)
      tpu_core_count += 1

  num_of_cores_per_host = 0
  if tpu_core_count:
    num_cores_per_host_set = set(
        [len(core_ids) for core_ids in device_dict.values()])
    if len(num_cores_per_host_set) != 1:
      raise RuntimeError(
          'TPU cores on each host is not same. This should not happen!. '
          'devices: {}'.format(devices))
    num_of_cores_per_host = num_cores_per_host_set.pop()

  topology = None
  if query_topology:
    if not tpu_core_count:
      raise RuntimeError(
          'Cannot find any TPU cores in the system (master address {}). '
          'This usually means the master address is incorrect or the '
          'TPU worker has some problems. Available devices: {}'.format(
              master_address, devices))

    topology = _obtain_topology(master_address, cluster_def)

  # We sort the metadata devices so that downstream users get a sorted list
  # for creating mirrored variables correctly.
  def _sort_key(device):
    # [FUNCTION DOC]
    # HELPER: sort key for the device list. Takes a device attributes object and returns a tuple
    # of job, replica, task, device type and device index parsed from its name.
    spec = tf_device.DeviceSpec.from_string(device.name)
    return (spec.job, spec.replica, spec.task, spec.device_type,
            spec.device_index)
  devices = tuple(sorted(devices, key=_sort_key))

  metadata = TPUSystemMetadata(
      num_cores=tpu_core_count,
      num_hosts=len(device_dict),
      num_of_cores_per_host=num_of_cores_per_host,
      topology=topology,
      devices=devices)

  if tpu_core_count:
    logging.info('Found TPU system:')
    logging.info('*** Num TPU Cores: %d', metadata.num_cores)
    logging.info('*** Num TPU Workers: %d', metadata.num_hosts)
    logging.info('*** Num TPU Cores Per Worker: %d',
                 metadata.num_of_cores_per_host)
    for device in metadata.devices:
      logging.info('*** Available Device: %s', device)
  else:
    logging.info('Failed to find TPU: %s', metadata)
  return metadata


def _obtain_topology(master_address, cluster_def):
  # [FUNCTION DOC]
  # HELPER: used by the system metadata query.
  # Fetches the serialized TPU topology by running the system initialization op in a new graph
  # and session against the master, with a five-minute operation timeout. Inputs: master address
  # and cluster definition. Returns the serialized topology; raises a value error if the
  # deadline is exceeded. Side effect: initializes the TPU system.
  """Obtains TPU fabric topology."""
  try:
    logging.info('Initializing TPU system (master: %s) to fetch topology '
                 'for model parallelism. This might take a while.',
                 master_address)
    with ops.Graph().as_default():
      session_config = get_session_config_with_timeout(
          _INITIAL_TPU_SYSTEM_TIMEOUT_IN_MS, cluster_def)
      with session_lib.Session(
          master_address, config=session_config) as sess:
        topology = sess.run(tpu.initialize_system())
        return topology
  except errors.DeadlineExceededError:
    raise ValueError(
        'Fail to initialize TPU system with master (%s). '
        'Please double check the TPU system is functional.' % (
            master_address))


def get_session_config_with_timeout(timeout_in_secs, cluster_def):
  # [FUNCTION DOC]
  # HELPER: builds session configuration for the query helpers.
  # KNOWN ISSUE: the docstring says it returns a session, but it returns a configuration
  # message; and the parameter is named seconds while the value is stored as milliseconds
  # (callers pass millisecond constants).
  # Inputs: a timeout number and a cluster definition. Returns a config message whose operation
  # timeout and cluster definition are set from them.
  """Returns a session given a timeout and a cluster configuration."""
  config_proto = config_pb2.ConfigProto(
      operation_timeout_in_ms=timeout_in_secs, cluster_def=cluster_def)
  return config_proto


def master_job(master, cluster_def):
  # [FUNCTION DOC]
  # Infers the job name for placing TPU computations. Inputs: master address text and a cluster
  # definition (may be null). Returns null for local masters (empty or "local"); the default
  # worker job name if the cluster definition has no jobs; the single job name if there is one;
  # with two jobs one being the coordinator, the other job's name. Raises a value error if the
  # default worker name is already used as a job, or if the name cannot be inferred.
  """Returns the canonical job name to use to place TPU computations on.

  Args:
    master: A `string` representing the TensorFlow master to use.
    cluster_def: A ClusterDef object describing the TPU cluster.

  Returns:
    A string containing the job name, or None if no job should be specified.

  Raises:
    ValueError: If the user needs to specify a tpu_job_name, because we are
      unable to infer the job name automatically, or if the user-specified job
      names are inappropriate.
  """
  # If the user specifies the tpu_job_name, use that.

  if master in _LOCAL_MASTERS:
    return None

  if (not cluster_def or not cluster_def.job):
    return _DEFAULT_JOB_NAME
  job_names = set(job.name for job in cluster_def.job)
  if _DEFAULT_JOB_NAME in job_names:
    # b/37868888 tracks allowing ClusterSpec propagation to reuse job names.
    raise ValueError('Currently, tpu_worker is not an allowed job name.')
  if len(job_names) == 1:
    return cluster_def.job[0].name
  if len(job_names) == 2:
    if _DEFAULT_COORDINATOR_JOB_NAME in job_names:
      job_names.remove(_DEFAULT_COORDINATOR_JOB_NAME)
      return job_names.pop()
    # TODO(b/67716447): Include more sophisticated heuristics.
  raise ValueError('Could not infer TPU job name.')
