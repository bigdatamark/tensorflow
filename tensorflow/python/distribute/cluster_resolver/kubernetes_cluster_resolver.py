# [MODULE DOC]
# Cluster resolver for Kubernetes. Exposes an enumeration describing where the running code lives
# relative to the cluster (inside a pod or outside) and a resolver class that lists pods matching
# label selectors via the Kubernetes API and builds a cluster specification from the pods' host IP
# addresses. Both are exported in the public distribute cluster-resolver API.
# The Kubernetes Python client is imported lazily inside methods, so there are no import-time
# side effects besides the API export registrations. Part of the cluster resolver package.
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
"""Implementation of Cluster Resolvers for Kubernetes."""

import enum

from tensorflow.python.distribute.cluster_resolver.cluster_resolver import ClusterResolver
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import format_master_url
from tensorflow.python.training import server_lib
from tensorflow.python.util.tf_export import tf_export


@tf_export('distribute.cluster_resolver.KubernetesExecutableLocation')
class ExecutableLocation(enum.Enum):
  # [CLASS DOC]
  # Enumeration with two members telling the resolver how to load Kubernetes configuration:
  # within-cluster (running in a pod, use in-cluster configuration) and off-cluster (use the local
  # kube configuration file). Used as the executable-location argument of the Kubernetes resolver.
  """Defines where the executable runs on.

  This is used to determine how to resolve the configuration
  to talk with the kube api server.

  `WITHIN_CLUSTER` means that the TensorFlow code you are running is running
  in a pod within the cluster itself.
  `OFF_CLUSTER` means any other enviroment outside the cluster.
  """

  WITHIN_CLUSTER = 0
  OFF_CLUSTER = 1


@tf_export('distribute.cluster_resolver.KubernetesClusterResolver')
class KubernetesClusterResolver(ClusterResolver):
  # [CLASS DOC]
  # Resolver that builds a cluster specification from Kubernetes pods.
  # State: a mapping from job names to lists of label selectors, the TensorFlow server port, an
  # optional injected Kubernetes client, and public task type, task index (both initially null)
  # and RPC layer attributes that callers may set directly. Pods are queried across all
  # namespaces on every cluster specification request; every selected pod must be running.
  """ClusterResolver for Kubernetes.

  This is an implementation of cluster resolvers for Kubernetes. When given the
  the Kubernetes namespace and label selector for pods, we will retrieve the
  pod IP addresses of all running pods matching the selector, and return a
  ClusterSpec based on that information.

  Note: it cannot retrieve `task_type`, `task_id` or `rpc_layer`. To use it
  with some distribution strategies like
  `tf.distribute.experimental.MultiWorkerMirroredStrategy`, you will need to
  specify `task_type` and `task_id` by setting these attributes.

  Usage example with tf.distribute.Strategy:

    ```Python
    # On worker 0
    cluster_resolver = KubernetesClusterResolver(
        {"worker": ["job-name=worker-cluster-a", "job-name=worker-cluster-b"]})
    cluster_resolver.task_type = "worker"
    cluster_resolver.task_id = 0
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
        cluster_resolver=cluster_resolver)

    # On worker 1
    cluster_resolver = KubernetesClusterResolver(
        {"worker": ["job-name=worker-cluster-a", "job-name=worker-cluster-b"]})
    cluster_resolver.task_type = "worker"
    cluster_resolver.task_id = 1
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
        cluster_resolver=cluster_resolver)
    ```
  """

  def __init__(
      self,
      job_to_label_mapping=None,
      tf_server_port=8470,
      rpc_layer='grpc',
      override_client=None,
      executable_location=ExecutableLocation.WITHIN_CLUSTER,
  ):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring lists a runtime error for a bad autoresolve option, but no such
    # parameter or error exists in the code.
    # KNOWN ISSUE: an invalid executable location is only rejected (value error) when the
    # Kubernetes client is installed and no override client is given; otherwise it is ignored.
    # Constructor. Inputs, all optional: mapping from job name to list of label selectors
    # (default one worker job selecting pods labeled job-name=tensorflow; an empty mapping also
    # gets the default), server port (default 8470), RPC layer (default grpc), an injected
    # Kubernetes client, and the executable location (default within cluster).
    # When no client is injected and the Kubernetes library is importable, loads the matching
    # configuration: kube config file for off-cluster, in-cluster config for within-cluster.
    # If the library is missing and no client is injected, raises an import error with install
    # instructions. Stores the settings and initializes task type and task index to null.
    # Side effects: loads global Kubernetes client configuration. Returns nothing.
    """Initializes a new KubernetesClusterResolver.

    This initializes a new Kubernetes ClusterResolver. The ClusterResolver
    will attempt to talk to the Kubernetes master to retrieve all the instances
    of pods matching a label selector.

    Args:
      job_to_label_mapping: A mapping of TensorFlow jobs to label selectors.
        This allows users to specify many TensorFlow jobs in one Cluster
        Resolver, and each job can have pods belong with different label
        selectors. For example, a sample mapping might be
        ```
        {'worker': ['job-name=worker-cluster-a', 'job-name=worker-cluster-b'],
         'ps': ['job-name=ps-1', 'job-name=ps-2']}
        ```
      tf_server_port: The port the TensorFlow server is listening on.
      rpc_layer: (Optional) The RPC layer TensorFlow should use to communicate
        between tasks in Kubernetes. Defaults to 'grpc'.
      override_client: The Kubernetes client (usually automatically retrieved
        using `from kubernetes import client as k8sclient`). If you pass this
        in, you are responsible for setting Kubernetes credentials manually and
        calling `k8sconfig.load_kube_config()` or
        `k8sconfig.load_incluster_config()` before using this ClusterResolver.
      executable_location: Parameter that specifies whether or not this
        TensorFlow code is running from within a K8S cluster or not.

    Raises:
      ImportError: If the Kubernetes Python client is not installed and no
        `override_client` is passed in.
      RuntimeError: If autoresolve_task is not a boolean or a callable.
      ValueError: If `executable_location` is not a valid value.
    """
    try:
      from kubernetes import config as k8sconfig  # pylint: disable=g-import-not-at-top

      if not override_client:
        if executable_location == ExecutableLocation.OFF_CLUSTER:
          k8sconfig.load_kube_config()
        elif executable_location == ExecutableLocation.WITHIN_CLUSTER:
          k8sconfig.load_incluster_config()
        else:
          raise ValueError('The executable location provided is invalid.')

    except ImportError:
      if not override_client:
        raise ImportError('The Kubernetes Python client must be installed '
                          'before using the Kubernetes Cluster Resolver. '
                          'To install the Kubernetes Python client, run '
                          '`pip install kubernetes` on your command line.')

    if not job_to_label_mapping:
      job_to_label_mapping = {'worker': ['job-name=tensorflow']}

    self._job_to_label_mapping = job_to_label_mapping
    self._tf_server_port = tf_server_port
    self._override_client = override_client

    self.task_type = None
    self.task_id = None
    self.rpc_layer = rpc_layer

  def master(self, task_type=None, task_id=None, rpc_layer=None):
    # [FUNCTION DOC]
    # Returns the master address of a task. Inputs, all optional: task type, task index, RPC
    # layer; null arguments fall back to the stored attributes (RPC layer falls back when null or
    # empty). If both task type and index are known, looks up the task address in a freshly built
    # cluster specification (queries Kubernetes) and formats it as a master URL with the RPC layer;
    # otherwise returns an empty string. Raises whatever the lookup raises for unknown tasks or
    # non-running pods.
    """Returns the master address to use when creating a session.

    You must have set the task_type and task_id object properties before
    calling this function, or pass in the `task_type` and `task_id`
    parameters when using this function. If you do both, the function parameters
    will override the object properties.

    Note: this is only useful for TensorFlow 1.x.

    Args:
      task_type: (Optional) The type of the TensorFlow task of the master.
      task_id: (Optional) The index of the TensorFlow task of the master.
      rpc_layer: (Optional) The RPC protocol for the given cluster.

    Returns:
      The name or URL of the session master.
    """
    task_type = task_type if task_type is not None else self.task_type
    task_id = task_id if task_id is not None else self.task_id

    if task_type is not None and task_id is not None:
      return format_master_url(
          self.cluster_spec().task_address(task_type, task_id),
          rpc_layer or self.rpc_layer)

    return ''

  def cluster_spec(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: when no client was injected, this always loads the local kube config file,
    # ignoring the executable location given at construction (which may be within-cluster).
    # Builds a cluster specification from Kubernetes. Uses the injected client or creates a core
    # API client after loading the kube config. For every job and each of its label selectors,
    # lists pods across all namespaces, sorts them by pod name, and records host IP plus the
    # server port for each. Addresses of a job are concatenated selector by selector. Raises a
    # runtime error naming the pod and its phase if any selected pod is not in the running phase.
    # Takes no arguments; returns the cluster specification.
    """Returns a ClusterSpec object based on the latest info from Kubernetes.

    We retrieve the information from the Kubernetes master every time this
    method is called.

    Returns:
      A ClusterSpec containing host information returned from Kubernetes.

    Raises:
      RuntimeError: If any of the pods returned by the master is not in the
        `Running` phase.
    """
    if self._override_client:
      client = self._override_client
    else:
      from kubernetes import config as k8sconfig  # pylint: disable=g-import-not-at-top
      from kubernetes import client as k8sclient  # pylint: disable=g-import-not-at-top

      k8sconfig.load_kube_config()
      client = k8sclient.CoreV1Api()

    cluster_map = {}

    for tf_job in self._job_to_label_mapping:
      all_pods = []
      for selector in self._job_to_label_mapping[tf_job]:
        ret = client.list_pod_for_all_namespaces(label_selector=selector)
        selected_pods = []

        # Sort the list by the name to make sure it doesn't change call to call.
        for pod in sorted(ret.items, key=lambda x: x.metadata.name):
          if pod.status.phase == 'Running':
            selected_pods.append(
                '%s:%s' % (pod.status.host_ip, self._tf_server_port))
          else:
            raise RuntimeError('Pod "%s" is not running; phase: "%s"' %
                               (pod.metadata.name, pod.status.phase))
        all_pods.extend(selected_pods)
      cluster_map[tf_job] = all_pods

    return server_lib.ClusterSpec(cluster_map)
