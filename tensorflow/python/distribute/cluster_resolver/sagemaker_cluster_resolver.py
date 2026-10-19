# [MODULE DOC]
# Cluster resolver for Amazon SageMaker training jobs. Builds a cluster specification from the
# environment variables set by SageMaker (the list of all hosts and the current host name):
# every host becomes a worker task with a common port, and the current host's position in the
# sorted host list is its task index. Only meant for multi-worker mirrored training.
# Defines module-level key constants, a master URL formatting function (a local copy of the one
# in the base resolver module) and private helpers that derive a configuration mapping from the
# environment. This resolver class is not exported through the public API.
# No import-time side effects other than defining constants.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Implementation of Cluster Resolvers for SageMaker Environment."""

import json
import os

from tensorflow.python.distribute.cluster_resolver.cluster_resolver import ClusterResolver
from tensorflow.python.training.server_lib import ClusterSpec

# List of envs
# https://github.com/aws/sagemaker-training-toolkit/blob/master/ENVIRONMENT_VARIABLES.md
# Only support Multi-Worker Mirrored Strategy

_SESSION_MASTER_KEY = 'session_master'
_RPC_LAYER_KEY = 'rpc_layer'
_TASK_KEY = 'task'
_CLUSTER_KEY = 'cluster'
_WORKER_KEY = 'worker'
_INDEX_KEY = 'index'
_TYPE_KEY = 'type'

_SM_CURRENT_HOST = 'SM_CURRENT_HOST'
_SM_HOSTS = 'SM_HOSTS'


def format_master_url(master, rpc_layer=None):
  # [FUNCTION DOC]
  # HELPER: builds the master address string for the resolver's master method.
  # KNOWN ISSUE: duplicates the identically named function in the base cluster resolver module
  # (which this module could import instead).
  # Inputs: master address text and an optional RPC layer name. If the layer is non-empty returns
  # the layer, a scheme separator and the address; otherwise returns the address unchanged.
  if rpc_layer:
    return '%s://%s' % (rpc_layer, master)
  else:
    return master


def _load_tf_config(port):
  # [FUNCTION DOC]
  # HELPER: converts SageMaker environment variables into a TensorFlow-style configuration
  # mapping for the resolver's properties and cluster specification.
  # KNOWN ISSUE: the check that the environment is a SageMaker one uses an assertion, which is
  # skipped when assertions are disabled (optimized mode), and it raises an assertion error rather
  # than a descriptive exception.
  # Input: port (text or number) appended to every host. Reads the hosts variable (a JSON list of
  # host names, possibly empty text meaning none) and the current host variable; hosts are sorted.
  # Returns an empty mapping if the current host is not in the list. Otherwise returns a mapping
  # with a cluster section (one worker job listing host and port strings) and a task section
  # (type worker, index equal to the current host's sorted position). Raises an assertion error if
  # either variable is missing; a malformed host list raises a JSON decoding error.
  # Create a tf_config from SM Variables
  assert all([x in os.environ for x in [_SM_CURRENT_HOST, _SM_HOSTS]
             ]), 'Not a SageMaker Environment'
  hosts = sorted(json.loads(
      os.environ[_SM_HOSTS])) if os.environ[_SM_HOSTS] != '' else []
  current_host = os.environ[_SM_CURRENT_HOST]

  if current_host not in hosts:
    return {}

  host_index = hosts.index(current_host)
  # Assign ports
  hosts = ['%s:%s' % (host, port) for host in hosts]

  tf_config = {
      _CLUSTER_KEY: {
          _WORKER_KEY: hosts
      },
      _TASK_KEY: {
          _TYPE_KEY: _WORKER_KEY,
          _INDEX_KEY: host_index
      }
  }
  return tf_config


def _get_value_in_tfconfig(key, port, default=None):
  # [FUNCTION DOC]
  # HELPER: looks up one top-level entry of the environment-derived configuration.
  # Inputs: key name, port, optional default (null). Rebuilds the configuration from the
  # environment on each call and returns the entry for the key, or the default if absent.
  tf_config = _load_tf_config(port)
  return tf_config[key] if key in tf_config else default


class SageMakerClusterResolver(ClusterResolver):
  # [CLASS DOC]
  # KNOWN ISSUE: the docstring's first line is one very long run-on sentence rather than a short
  # summary line.
  # Resolver that reads cluster information from SageMaker environment variables.
  # State: port (stored as text) plus optional overrides for task type, task index, RPC layer and
  # environment. Values not overridden are derived from the environment at access time (not
  # cached), so they can change if the environment changes. All hosts are workers.
  """Implementation of a ClusterResolver which reads the Sagemaker EnvVars. This is an implementation of cluster resolvers when running in a SageMaker environment to set information about the cluster.

  The cluster spec returned will be initialized from the SageMaker
  environment variables.
  Currently this Cluster Resolver only supports Multi-Worker Mirrored Strategy.
  It assumes all nodes in a SageMaker Cluster are workers.
  """

  def __init__(self,
               port=2223,
               task_type=None,
               task_id=None,
               rpc_layer=None,
               environment=None):
    # [FUNCTION DOC]
    # Constructor. Inputs, all optional: port (default 2223, converted to text), task type, task
    # index, RPC layer and environment name overrides (default null meaning derive from the
    # environment). Stores them; does not read the environment yet. Returns nothing.
    """Creates a new SageMakerClusterResolver.

    Args:
      port: (integer, optional) Override default port usage of 2223
      task_type: (String, optional) Overrides the task type.
      task_id: (Integer, optional) Overrides the task index.
      rpc_layer: (String, optional) Overrides the rpc layer TensorFlow uses.
      environment: (String, optional) Overrides the environment TensorFlow
        operates in.
    """
    self._task_type = task_type
    self._task_id = task_id
    self._rpc_layer = rpc_layer
    self._environment = environment
    self._port = str(port)

  @property
  def task_type(self):
    # [FUNCTION DOC]
    # Property getter. Returns the override as text if set; otherwise the task type derived from
    # the environment (always worker when the current host is listed), or null if unavailable.
    # May raise an assertion error if the SageMaker variables are missing.
    if self._task_type is None:
      task_info = _get_value_in_tfconfig(_TASK_KEY, self._port, {})
      return str(task_info['type']) if 'type' in task_info else None
    else:
      return str(self._task_type)

  @property
  def task_id(self):
    # [FUNCTION DOC]
    # Property getter. Returns the override as an integer if set; otherwise the current host's
    # sorted index derived from the environment, or null if the host is not listed. May raise an
    # assertion error if the SageMaker variables are missing.
    if self._task_id is None:
      task_info = _get_value_in_tfconfig(_TASK_KEY, self._port, {})
      return int(task_info['index']) if 'index' in task_info else None
    else:
      return int(self._task_id)

  @task_type.setter
  def task_type(self, task_type):
    # [FUNCTION DOC]
    # Property setter: stores the task type override. Returns nothing.
    self._task_type = task_type

  @task_id.setter
  def task_id(self, task_id):
    # [FUNCTION DOC]
    # Property setter: stores the task index override. Returns nothing.
    self._task_id = task_id

  @property
  def environment(self):
    # [FUNCTION DOC]
    # Property getter: returns the environment name given at construction (null by default).
    return self._environment

  @property
  def rpc_layer(self):
    # [FUNCTION DOC]
    # Property getter: returns the RPC layer override if set; otherwise looks for an RPC layer
    # entry in the environment-derived configuration, which is never populated, so it effectively
    # returns null.
    if self._rpc_layer is None:
      return _get_value_in_tfconfig(_RPC_LAYER_KEY, self._port)
    else:
      return self._rpc_layer

  @rpc_layer.setter
  def rpc_layer(self, rpc_layer):
    # [FUNCTION DOC]
    # Property setter: stores the RPC layer override. Returns nothing.
    self._rpc_layer = rpc_layer

  def num_accelerators(self, task_type=None, task_id=None, config_proto=None):
    # [FUNCTION DOC]
    # Counts accelerators per device type. Inputs, all optional: task type and task index
    # (default to this resolver's own values) and a session configuration. Delegates to the base
    # class implementation and returns its mapping from device type to count.
    task_type = self.task_type if task_type is None else task_type
    task_id = self.task_id if task_id is None else task_id
    return super(SageMakerClusterResolver,
                 self).num_accelerators(task_type, task_id, config_proto)

  def cluster_spec(self):
    # [FUNCTION DOC]
    # Builds the cluster specification from the SageMaker environment variables (read on each
    # call). Returns an empty specification if the current host is not in the host list;
    # otherwise one worker job with all hosts (sorted) and the configured port. Takes no
    # arguments. May raise an assertion error if the variables are missing.
    """Returns a ClusterSpec based on the SageMaker environment variables.

    Returns:
      A ClusterSpec with information from the SageMaker environment variables.
    """
    tf_config = _load_tf_config(self._port)
    if 'cluster' not in tf_config:
      return ClusterSpec({})
    return ClusterSpec(tf_config['cluster'])

  def master(self, task_type=None, task_id=None, rpc_layer=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says a runtime error is raised when task information is missing,
    # but the code raises nothing itself; an unknown or null task is passed to the address lookup,
    # which raises whatever error it raises. Also, the configuration never contains a session
    # master entry, so that first branch is effectively dead.
    # Returns the master address. Inputs, all optional: task type, task index, RPC layer
    # (null values fall back to the resolver's own values). Returns an explicit session master if
    # configured; returns an empty string when the cluster has no jobs or only one job with a
    # single task; otherwise returns the task's address formatted with the RPC layer prefix.
    """Returns the master address to use when creating a TensorFlow session.

    Note: this is only useful for TensorFlow 1.x.

    Args:
      task_type: (String, optional) Overrides and sets the task_type of the
        master.
      task_id: (Integer, optional) Overrides and sets the task id of the master.
      rpc_layer: (String, optional) Overrides and sets the protocol over which
        TensorFlow nodes communicate with each other.

    Returns:
      The address of the master.

    Raises:
      RuntimeError: If the task_type or task_id is not specified and the
        SageMaker environment variables does not contain a task section.
    """

    # If `session_master` is set, just use that.
    session_master = _get_value_in_tfconfig(_SESSION_MASTER_KEY, self._port)
    if session_master is not None:
      return session_master

    # Return an empty string if we are the only job in the ClusterSpec.
    cluster_spec = self.cluster_spec()
    if (not cluster_spec.jobs or
        (len(cluster_spec.jobs) == 1 and
         len(cluster_spec.job_tasks(cluster_spec.jobs[0])) == 1)):
      return ''

    # We try to auto-detect the task type and id, but uses the user-supplied one
    # where available
    task_type = task_type if task_type is not None else self.task_type
    task_id = task_id if task_id is not None else self.task_id
    rpc_layer = rpc_layer if rpc_layer is not None else self.rpc_layer

    return format_master_url(
        cluster_spec.task_address(task_type, task_id), rpc_layer)
