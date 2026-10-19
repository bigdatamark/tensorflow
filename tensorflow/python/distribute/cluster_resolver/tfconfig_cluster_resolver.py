# [MODULE DOC]
# Cluster resolver that discovers cluster information from the TF_CONFIG environment variable,
# which holds a JSON document with optional keys for the cluster layout, the task (type and index),
# a session master address and an rpc layer. Exposes the public resolver class
# (exported as the distribute cluster-resolver TFConfigClusterResolver) and a public helper that
# formats a master address with an optional protocol prefix. Private helpers read and parse the
# environment variable on every call, so changes to it are picked up dynamically.
# No import-time side effects other than the public API export registration.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Implementation of Cluster Resolvers for TF_CONFIG Environment Variables."""


import json
import os

from tensorflow.python.distribute.cluster_resolver.cluster_resolver import ClusterResolver
from tensorflow.python.training.server_lib import ClusterSpec
from tensorflow.python.util.tf_export import tf_export

_TF_CONFIG_ENV = 'TF_CONFIG'
_SESSION_MASTER_KEY = 'session_master'
_RPC_LAYER_KEY = 'rpc_layer'
_TASK_KEY = 'task'


def format_master_url(master, rpc_layer=None):
  # [FUNCTION DOC]
  # Builds a master address string. If rpc_layer is given and non-empty, returns the protocol,
  # a colon-slash-slash separator and the master address joined together; otherwise returns the
  # master address unchanged. Inputs: master (address string), rpc_layer (optional protocol name).
  # No side effects, no exceptions of its own.
  if rpc_layer:
    return '%s://%s' % (rpc_layer, master)
  else:
    return master


def _load_tf_config():
  # [FUNCTION DOC]
  # HELPER: reads the TF_CONFIG environment variable for the TFConfigClusterResolver.
  # Returns the parsed JSON content of the environment variable as a mapping; if the variable
  # is unset, parses an empty mapping. Raises a JSON decode error if the value is not valid JSON.
  # Takes no arguments and has no side effects.
  return json.loads(os.environ.get(_TF_CONFIG_ENV, '{}'))


def _get_value_in_tfconfig(key, default=None):
  # [FUNCTION DOC]
  # HELPER: looks up a single top-level entry of the TF_CONFIG mapping for the resolver.
  # Inputs: key (name of the entry), default (value returned when absent, defaults to null).
  # Returns the entry's value if present, else the default. Re-parses the environment variable
  # on each call; may raise a JSON decode error for malformed content.
  tf_config = _load_tf_config()
  return tf_config[key] if key in tf_config else default


@tf_export('distribute.cluster_resolver.TFConfigClusterResolver')
class TFConfigClusterResolver(ClusterResolver):
  # [CLASS DOC]
  # Concrete cluster resolver whose cluster spec, task type/index, rpc layer and session master
  # come from the TF_CONFIG environment variable, with optional per-instance overrides given at
  # construction or through property setters. Holds four private attributes: task type, task id,
  # rpc layer and environment (null means "fall back to TF_CONFIG"). Subclasses the abstract
  # ClusterResolver base class; exported as a public distribute cluster-resolver class.
  """Implementation of a ClusterResolver which reads the TF_CONFIG EnvVar.

  This is an implementation of cluster resolvers when using TF_CONFIG to set
  information about the cluster. The cluster spec returned will be
  initialized from the TF_CONFIG environment variable.

  An example to set TF_CONFIG is:

    ```Python
    os.environ['TF_CONFIG'] = json.dumps({
      'cluster': {
          'worker': ["localhost:12345", "localhost:23456"]
      },
      'task': {'type': 'worker', 'index': 0}
    })
    ```

  However, sometimes the container orchestration framework will set TF_CONFIG
  for you. In this case, you can just create an instance without passing in any
  arguments. You can find an example here to let Kuburnetes set TF_CONFIG for
  you: https://github.com/tensorflow/ecosystem/tree/master/kubernetes. Then you
  can use it with `tf.distribute.Strategy` as:

    ```Python
    # `TFConfigClusterResolver` is already the default one in the following
    # strategy.
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
        cluster_resolver=TFConfigClusterResolver())
    ```
  """

  def __init__(self,
               task_type=None,
               task_id=None,
               rpc_layer=None,
               environment=None):
    # [FUNCTION DOC]
    # Constructor. Stores optional overrides: task_type (string), task_id (integer), rpc_layer
    # (string) and environment (string). Any left as null defers to the TF_CONFIG variable
    # where applicable. Does not read the environment at construction time; returns nothing.
    """Creates a new TFConfigClusterResolver.

    Args:
      task_type: (String, optional) Overrides the task type specified in the
        TF_CONFIG environment variable.
      task_id: (Integer, optional) Overrides the task index specified in the
        TF_CONFIG environment variable.
      rpc_layer: (String, optional) Overrides the rpc layer TensorFlow uses.
      environment: (String, optional) Overrides the environment TensorFlow
        operates in.
    """
    self._task_type = task_type
    self._task_id = task_id
    self._rpc_layer = rpc_layer
    self._environment = environment

  @property
  def task_type(self):
    # [FUNCTION DOC]
    # Property getter for the task type. If no override was set, reads the task entry of TF_CONFIG
    # and returns its type as a string, or null if absent. Otherwise returns the override
    # converted to a string.
    if self._task_type is None:
      task_info = _get_value_in_tfconfig(_TASK_KEY, {})
      return str(task_info['type']) if 'type' in task_info else None
    else:
      return str(self._task_type)

  @property
  def task_id(self):
    # [FUNCTION DOC]
    # Property getter for the task index. If no override was set, reads the task entry of TF_CONFIG
    # and returns its index as an integer, or null if absent. Otherwise returns the override
    # converted to an integer (raises a value error if not convertible).
    if self._task_id is None:
      task_info = _get_value_in_tfconfig(_TASK_KEY, {})
      return int(task_info['index']) if 'index' in task_info else None
    else:
      return int(self._task_id)

  @task_type.setter
  def task_type(self, task_type):
    # [FUNCTION DOC]
    # Property setter: stores the given task type as an override of TF_CONFIG (stored as given,
    # converted to a string only when read). Returns nothing.
    self._task_type = task_type

  @task_id.setter
  def task_id(self, task_id):
    # [FUNCTION DOC]
    # Property setter: stores the given task index as an override of TF_CONFIG (converted to an
    # integer only when read). Returns nothing.
    self._task_id = task_id

  @property
  def environment(self):
    # [FUNCTION DOC]
    # Property getter returning the environment string given at construction (null if none).
    # Never consults TF_CONFIG and has no setter.
    return self._environment

  @property
  def rpc_layer(self):
    # [FUNCTION DOC]
    # Property getter for the rpc protocol name. Returns the override if one was set, otherwise
    # the rpc layer entry of TF_CONFIG (null if absent).
    if self._rpc_layer is None:
      return _get_value_in_tfconfig(_RPC_LAYER_KEY)
    else:
      return self._rpc_layer

  @rpc_layer.setter
  def rpc_layer(self, rpc_layer):
    # [FUNCTION DOC]
    # Property setter: stores the given rpc protocol name as an override of TF_CONFIG.
    # Returns nothing.
    self._rpc_layer = rpc_layer

  def num_accelerators(self,
                       task_type=None,
                       task_id=None,
                       config_proto=None):
    # [FUNCTION DOC]
    # Returns the number of accelerators on a task, as reported by the base class. Inputs:
    # task_type (string, optional), task_id (integer, optional), config_proto (optional session
    # configuration). Missing task type or id default to this resolver's own task type and id
    # (which may come from TF_CONFIG), then the base class implementation is called.
    task_type = self.task_type if task_type is None else task_type
    task_id = self.task_id if task_id is None else task_id
    return super(TFConfigClusterResolver, self).num_accelerators(
        task_type, task_id, config_proto)

  def cluster_spec(self):
    # [FUNCTION DOC]
    # Reads TF_CONFIG and returns a cluster specification built from its cluster entry, or an
    # empty cluster specification if that entry is missing. Takes no arguments. May raise a JSON
    # decode error if the variable is malformed.
    """Returns a ClusterSpec based on the TF_CONFIG environment variable.

    Returns:
      A ClusterSpec with information from the TF_CONFIG environment variable.
    """
    tf_config = _load_tf_config()
    if 'cluster' not in tf_config:
      return ClusterSpec({})
    return ClusterSpec(tf_config['cluster'])

  def master(self, task_type=None, task_id=None, rpc_layer=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says a runtime error is raised when no task section exists, but
    # this method raises nothing itself; failures come from the cluster specification's task
    # address lookup (typically a value error for an unknown job or task).
    # Returns the master address. Order of precedence: the session master entry of TF_CONFIG if
    # present; else an empty string when the cluster has no jobs or exactly one job with one
    # task; else the address of the task given by task_type/task_id (each optional, defaulting
    # to this resolver's values), prefixed with the rpc layer (optional, defaults to this
    # resolver's rpc layer) when one is available.
    """Returns the master address to use when creating a TensorFlow session.

    Note: this is only useful for TensorFlow 1.x.

    Args:
      task_type: (String, optional) Overrides and sets the task_type of the
        master.
      task_id: (Integer, optional) Overrides and sets the task id of the
        master.
      rpc_layer: (String, optional) Overrides and sets the protocol over which
        TensorFlow nodes communicate with each other.

    Returns:
      The address of the master.

    Raises:
      RuntimeError: If the task_type or task_id is not specified and the
        `TF_CONFIG` environment variable does not contain a task section.
    """

    # If `session_master` is set, just use that.
    session_master = _get_value_in_tfconfig(_SESSION_MASTER_KEY)
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

    return format_master_url(cluster_spec.task_address(task_type, task_id),
                             rpc_layer)
