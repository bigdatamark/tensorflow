# [MODULE DOC]
# Defines an experimental multi-worker mirrored strategy built on the low-level DTensor API.
# Exposes one strategy class plus private helpers that translate a cluster resolver into the
# DTensor environment variables, write them into the process environment, and build a global
# one-dimensional batch mesh across all clients. Relates to the DTensor strategy extended and
# DTensor utility modules. No import-time side effects; the environment is modified only when a
# strategy is constructed without a mesh.
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Implement a MultiMirroredStrategy based on the DTensor low level API.

This is an experiment to validate the viability of the DTensor API, and expose
any potential feature gaps between the current API and the need.
"""

import os

from tensorflow.dtensor.python import config as d_config
from tensorflow.dtensor.python import mesh_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute.cluster_resolver import tfconfig_cluster_resolver
from tensorflow.python.distribute.experimental import dtensor_strategy_extended
from tensorflow.python.distribute.experimental import dtensor_util


class MultiWorkerMirroredStrategy(distribute_lib.Strategy):
  # [CLASS DOC]
  # Experimental synchronous multi-worker data-parallel strategy built on DTensor. Subclasses the
  # generic strategy and delegates to a DTensor strategy extended object over a one-dimensional
  # global batch mesh. Holds the mesh (exposed as a property) and the cluster resolver used
  # (null if a mesh was supplied). Created by users with either a cluster resolver, a mesh, or
  # neither (then the environment-based cluster configuration is used).
  """A distribution strategy for synchronous training on multiple workers.

  This strategy implements synchronous distributed training across multiple
  workers, each with potentially multiple GPUs. Similar to
  `tf.distribute.MirroredStrategy`, it replicates all variables and computations
  to each local device. The difference is that it uses a distributed collective
  implementation (e.g. all-reduce), so that multiple workers can work together.
  """

  def __init__(self, cluster_resolver=None, communication_options=None, *,
               mesh=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: communication_options is accepted but never used (the docstring says it is
    # ignored). The stored cluster resolver attribute stays null when a mesh is supplied.
    # Builds the strategy. Parameters: cluster_resolver, optional cluster resolver object;
    # communication_options, ignored; mesh (keyword only), optional global one-dimensional
    # DTensor mesh. Validates that mesh and cluster resolver are not both given and that a mesh is
    # one-dimensional (value error otherwise). With no mesh: uses the environment-variable based
    # cluster resolver if none was given, derives the DTensor environment variables from it,
    # writes them into the process environment (global side effect), and builds the distributed
    # mesh. Then creates the DTensor strategy extended object, calls the parent constructor and
    # stores the mesh and cluster resolver.
    """Creates the strategy.

    Args:
      cluster_resolver: optional
        `tf.distribute.cluster_resolver.ClusterResolver`. In case neither `mesh`
        nor `cluster_resolver` are provided,
        `tf.distribute.cluster_resolver.TFConfigClusterResolver` is used.
      communication_options: currently ignore.
      mesh: optional Dtensor global mesh for the computation. Note that either
        `mesh` or the `cluster_resolver` should be provided. and not both.
    """
    self._validate_init_args(mesh, cluster_resolver)
    if not mesh:
      if not cluster_resolver:
        # Use the TFConfigClusterResolver as default
        cluster_resolver = tfconfig_cluster_resolver.TFConfigClusterResolver()
      dtensor_env_var = _parse_dtensor_env_var_from_cluster_resolver(
          cluster_resolver)
      _config_dtensor_env_var(dtensor_env_var)
      mesh = _build_distributed_mesh(dtensor_util.DEFAULT_BATCH_MESH_DIM_NAME)
    extended = dtensor_strategy_extended.DTensorStrategyExtended(
        container_strategy=self, mesh=mesh)
    super().__init__(extended)
    self._mesh = mesh
    self._cluster_resolver = cluster_resolver

  @classmethod
  def _validate_init_args(cls, mesh, cluster_resolver):
    # [FUNCTION DOC]
    # HELPER: argument validation for the strategy constructor.
    # Class method taking an optional mesh and optional cluster resolver. Raises a value error if
    # both are provided, or if a mesh is provided that is not one-dimensional. Returns nothing.
    if mesh and cluster_resolver:
      raise ValueError('Mesh and cluster_resolver can not be provided at the '
                       f'same time. Received mesh = {mesh}, cluster_resolver = '
                       f'{cluster_resolver}')
    if mesh and len(mesh.shape()) != 1:
      raise ValueError('The mesh for MultiWorkerMirroredStrategy must be 1D, '
                       f'received: {len(mesh.shape())}D')

  def reduce(self, reduce_op, value, axis):
    # [FUNCTION DOC]
    # Overrides the strategy reduce using the DTensor reduce helper. Parameters: reduce_op, the
    # reduction kind; value, the distributed value; axis, optional extra dimension to reduce
    # over. Returns the reduced result.
    return dtensor_util.dtensor_reduce(self, reduce_op, value, axis)

  @property
  def mesh(self):
    # [FUNCTION DOC]
    # Read-only property returning the DTensor mesh used by the strategy.
    """Returns the mesh used by the strategy."""
    return self._mesh


def _parse_dtensor_env_var_from_cluster_resolver(cluster_resolver):
  # [FUNCTION DOC]
  # HELPER: converts cluster information into DTensor environment settings for the strategy.
  # KNOWN ISSUE: if the resolver's task type is neither chief nor worker, the client id is never
  # assigned and the function fails with an unbound-variable error rather than a clear message.
  # The documented Raises section is also vague; the only explicit raise is a value error when
  # a job address in the cluster spec is null.
  # Input: a cluster resolver. Normalizes its cluster spec, collects the chief tasks followed
  # by the worker tasks as the DTensor jobs, and builds a mapping of strings to strings with
  # the comma-joined job list, the number of clients, this process's client id (zero for the
  # chief; the worker task id, shifted by one when a chief exists) and the job name set to worker.
  # Returns that mapping; does not modify the environment itself.
  """Parse the env vars for Dtensor based on the cluster resolver.

  In the multi-client setting, each of the DTensor jobs need to aware of each
  other, and the interface to setup those values are via the envvars. The
  value used by dtensor are different from the existing
  `MultiWorkerMirroredStrategy`. This function will parse the value from
  cluster resolver, and populate the corresponding value for DTensor jobs in the
  `os.environ`.

  Args:
    cluster_resolver: A `tf.distribute.cluster_resolver.ClusterResolver`
      instance.

  Returns:
    A dict of {Str:Str} which contains all the env vars needed by DTensor jobs.
    The value is for verification purpose.

  Raises:
    The value parsed from existing cluster spec is not valid.
  """
  result = {}

  # Retrieve the number of host, cluster config from the resolver.
  cluster_spec = multi_worker_util.normalize_cluster_spec(
      cluster_resolver.cluster_spec())
  # Export all the necessary envvars for dtensor
  # Get all the jobs from the cluster spec. Note that the in the normal
  # setting, it could be multiple worker devices without chief, and the
  # worker 0 will be the chief, or an explicit chief with multiple worker job.
  dtensor_jobs = []
  if 'chief' in cluster_spec.jobs:
    dtensor_jobs.extend(cluster_spec.job_tasks('chief'))
  if 'worker' in cluster_spec.jobs:
    dtensor_jobs.extend(cluster_spec.job_tasks('worker'))

  if None in dtensor_jobs:
    raise ValueError('Unexpected dtensor job address from cluster spec: '
                     f'{cluster_spec}')
  result['DTENSOR_JOBS'] = ','.join(dtensor_jobs)
  result['DTENSOR_NUM_CLIENTS'] = str(len(dtensor_jobs))

  if cluster_resolver.task_type == 'chief':
    dtensor_client_id = 0
  elif cluster_resolver.task_type == 'worker':
    dtensor_client_id = cluster_resolver.task_id
    if 'chief' in cluster_spec.jobs:
      dtensor_client_id += 1
  result['DTENSOR_CLIENT_ID'] = str(dtensor_client_id)
  result['DTENSOR_JOB_NAME'] = 'worker'

  return result


def _config_dtensor_env_var(dtensor_env_vars):
  # [FUNCTION DOC]
  # HELPER: applies the DTensor environment settings for the strategy constructor.
  # Takes a mapping of environment variable names to string values and writes each into the
  # process environment, overwriting existing values. Returns nothing.
  for k, v in dtensor_env_vars.items():
    os.environ[k] = v


def _build_distributed_mesh(batch_dim_name):
  # [FUNCTION DOC]
  # HELPER: builds the global mesh for the strategy constructor.
  # Takes the name for the single batch dimension. Detects the preferred device type, counts the
  # local devices and the number of clients, initializes the accelerator system once, and creates
  # a distributed one-dimensional mesh whose size is local devices times clients (assuming each
  # client has the same number of devices). Returns the mesh.
  device_type = d_config.preferred_device_type()
  local_devices = d_config.local_devices(device_type)
  number_clients = d_config.num_clients()
  dtensor_util.initialize_accelerator_system_once(device_type)
  # This assumes each client has same number of devices.
  mesh_dims = [(batch_dim_name, len(local_devices) * number_clients)]
  return mesh_util.create_distributed_mesh(
      mesh_dims, device_type=device_type)
