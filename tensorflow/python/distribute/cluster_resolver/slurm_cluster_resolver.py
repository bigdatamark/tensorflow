# [MODULE DOC]
# Cluster resolver for clusters managed by the Slurm workload manager. Reads Slurm environment
# variables (process rank, step task count, node list, tasks per node) to find which nodes host
# which tasks, assigns each task a port (base port plus its position on the node) and a block of
# GPUs, and builds a cluster specification with the requested jobs.
# Exposes public parsing helpers for Slurm host lists and tasks-per-node expressions, a function
# counting local GPUs (NVIDIA only), private helpers reading Slurm variables, and the resolver
# class (exported in the public distribute cluster-resolver API).
# Side effects: resolving the cluster can set the visible CUDA devices environment variable.
# Part of the cluster resolver package.
# Copyright 2018-2020 The TensorFlow Authors. All Rights Reserved.
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
"""Implementation of Cluster Resolvers for Slurm workload manager."""

import os
import re
import subprocess

from tensorflow.python.distribute.cluster_resolver.cluster_resolver import ClusterResolver
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import format_master_url
from tensorflow.python.training.server_lib import ClusterSpec
from tensorflow.python.util.tf_export import tf_export


def expand_hostlist(hostlist):
  # [FUNCTION DOC]
  # Expands a Slurm compact host list into individual host names. Input: text such as comma
  # separated names where a name may carry a bracketed list of numbers and numeric ranges
  # (zero padding of the range start is preserved). Returns a list of host names in order, with
  # no deduplication. Raises a value error naming the whole input if any part is malformed
  # (the inner failures, including assertion failures, are wrapped into this error).
  """Create a list of hosts out of a SLURM hostlist.

  The order of nodes is preserved and no deduplication is done
  Input: 'n[1-2],m5,o[3-4,6,7-9]')
  Output: ['n1', 'n2', 'm5', 'o3', 'o4', 'o6', 'o7', 'o8', 'o9']
  """

  def split_hostlist(hostlist):
    # [FUNCTION DOC]
    # HELPER: generator used by the host list expansion. Splits the input text at commas that are
    # not inside square brackets and yields each part in order. Uses assertions to reject nested
    # opening brackets and empty parts between commas (these surface as assertion errors, which
    # the caller wraps).
    """Split hostlist at commas outside of range expressions ('[3-5]')."""
    in_brackets = False
    cur_host = ''
    for c in hostlist:
      if in_brackets:
        assert c != '['
        if c == ']':
          in_brackets = False
      elif c == '[':
        in_brackets = True
      elif c == ',':
        assert cur_host != ''
        yield cur_host
        cur_host = ''
        continue
      cur_host += c
    if cur_host:
      yield cur_host

  def expand_range_expression(range_exp):
    # [FUNCTION DOC]
    # HELPER: generator used by the host list expansion. Input: text inside brackets, a comma
    # separated list of single numbers or low-high ranges. Yields each number as text, inclusive
    # of range ends, zero-padded to the width of the range start. Asserts that a range has exactly
    # two ends; non-numeric text raises a value error (wrapped by the caller).
    """Expand a range expression like '3-5' to values 3,4,5."""
    for part in range_exp.split(','):
      sub_range = part.split('-')
      if len(sub_range) == 1:
        sub_range = sub_range * 2
      else:
        assert len(sub_range) == 2
      num_digits = len(sub_range[0])
      for i in range(int(sub_range[0]), int(sub_range[1]) + 1):
        yield str(i).zfill(num_digits)

  hosts = []
  try:
    for part in split_hostlist(hostlist):
      # Match prefix (anything but a range expression) and range expression
      # Both are optional
      m = re.match(r'([^,[\]]*)(\[([^\]]+)\])?$', part)
      if m is None:
        raise ValueError('Invalid part: %s' % part)
      prefix = m.group(1) or ''
      if m.group(3) is None:
        hosts.append(prefix)
      else:
        hosts.extend(prefix + i for i in expand_range_expression(m.group(3)))
  except Exception as e:
    raise ValueError('Invalid hostlist format "%s": %s' % (hostlist, e))
  return hosts


def expand_tasks_per_node(tasks_per_node):
  # [FUNCTION DOC]
  # Expands Slurm's tasks-per-node text: comma separated counts, each optionally followed by a
  # repetition marker giving how many consecutive nodes share that count. Returns a list of
  # integers in order, aligned with the node list. Raises a value error naming the input if any
  # part does not match the expected format.
  """Expands the tasks per node expression from SLURM.

  The order is preserved so it can be matched to the hostlist
  Input: '3(x2),2,1'
  Output: [3, 3, 2, 1]
  """
  result = []
  try:
    for part in tasks_per_node.split(','):
      m = re.match(r'(\d+)(\(x(\d+)\))?$', part)
      assert m is not None
      num_tasks = int(m.group(1))
      num_repetitions = int(m.group(3) or 1)
      result.extend([num_tasks] * num_repetitions)
  except Exception as e:
    raise ValueError('Invalid tasks-per-node list format "%s": %s' %
                     (tasks_per_node, e))
  return result


def _get_slurm_var(name):
  # [FUNCTION DOC]
  # HELPER: reads Slurm environment variables for the resolver.
  # Input: variable name without the Slurm prefix. Returns the value of the prefixed environment
  # variable as text. Raises a runtime error saying the process may not be inside a Slurm step if
  # the variable is missing.
  """Gets the SLURM variable from the environment.

  Args:
    name: Name of the step variable

  Returns:
    SLURM_<name> from os.environ
  Raises:
    RuntimeError if variable is not found
  """
  name = 'SLURM_' + name
  try:
    return os.environ[name]
  except KeyError:
    raise RuntimeError('%s not found in environment. '
                       'Not running inside a SLURM step?' % name)


def _get_num_slurm_tasks():
  # [FUNCTION DOC]
  # HELPER: returns the number of tasks in the current Slurm job step as an integer, read from
  # the step task count variable. Raises a runtime error if the variable is missing.
  """Returns the number of SLURM tasks of the current job step.

  Returns:
    The number of tasks as an int
  """
  return int(_get_slurm_var('STEP_NUM_TASKS'))


def _get_num_nvidia_gpus():
  # [FUNCTION DOC]
  # HELPER: GPU counting for the public GPU count function.
  # KNOWN ISSUE: an empty visible-devices variable (meaning no GPUs) is counted as one GPU
  # because splitting empty text yields one empty item.
  # KNOWN ISSUE: the docstring and error text suggest a missing nvidia-smi is handled, but a
  # missing executable raises a file-not-found error, which is not caught; only a failing exit
  # status is converted to a runtime error.
  # Returns the number of entries in the visible CUDA devices variable if it is set; otherwise
  # runs the nvidia-smi listing command and counts lines starting with the GPU marker. Raises a
  # runtime error including the command output if the command exits with failure.
  """Gets the number of NVIDIA GPUs by using CUDA_VISIBLE_DEVICES and nvidia-smi.

  Returns:
    Number of GPUs available on the node
  Raises:
    RuntimeError if executing nvidia-smi failed
  """
  try:
    return len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
  except KeyError:
    pass  # Ignore and fallback to using nvidia-smi
  try:
    output = subprocess.check_output(['nvidia-smi', '--list-gpus'],
                                     encoding='utf-8')
    return sum(l.startswith('GPU ') for l in output.strip().split('\n'))
  except subprocess.CalledProcessError as e:
    raise RuntimeError('Could not get number of GPUs from nvidia-smi. '
                       'Maybe it is missing?\nOutput: %s' % e.output)


def get_num_gpus():
  # [FUNCTION DOC]
  # Returns the number of GPUs visible on the current node as an integer. Only NVIDIA GPUs are
  # supported; delegates to the private NVIDIA counting helper and may raise its errors.
  """Returns the number of GPUs visible on the current node.

  Currently only implemented for NVIDIA GPUs.
  """
  return _get_num_nvidia_gpus()


@tf_export('distribute.cluster_resolver.SlurmClusterResolver')
class SlurmClusterResolver(ClusterResolver):
  # [CLASS DOC]
  # Resolver for Slurm-managed clusters.
  # State: this process's rank, the requested jobs (mapping from job name to task count), the
  # base port, a mapping from host name to number of tasks on it, GPUs per node and per task, a
  # flag for automatically setting visible GPUs, public task type, task index (null until the
  # cluster specification is computed) and RPC layer, and the most recent GPU and per-job address
  # allocations (rebuilt on each cluster specification request).
  # Tasks are packed onto the nodes in sorted host order and then assigned to jobs in sorted job
  # name order, each job receiving a consecutive block of ranks.
  """ClusterResolver for system with Slurm workload manager.

  This is an implementation of ClusterResolver for Slurm clusters. This allows
  the specification of jobs and task counts, number of tasks per node, number
  of GPUs on each node and number of GPUs for each task. It retrieves system
  attributes by Slurm environment variables, resolves allocated computing node
  names, constructs a cluster and returns a ClusterResolver object which can be
  used for distributed TensorFlow.
  """

  def __init__(self,
               jobs=None,
               port_base=8888,
               gpus_per_node=None,
               gpus_per_task=None,
               tasks_per_node=None,
               auto_set_gpu=True,
               rpc_layer='grpc'):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says a cluster resolver object is returned, but a constructor
    # returns nothing; it initializes this instance.
    # KNOWN ISSUE: a computed per-task GPU count of zero (more tasks per node than GPUs) is not
    # rejected and later makes the cluster specification computation fail on a zero range step.
    # Constructor. Inputs, all optional: jobs (mapping from job name to task count; default one
    # worker job with as many tasks as the Slurm step has), base port (default 8888), GPUs per node
    # (default counted on this node), GPUs per task (default GPUs per node divided by the maximum
    # tasks on any node, rounded down), tasks per node (a number for all nodes, a mapping from host
    # name to count, or null to read from the Slurm environment), flag to set visible GPUs
    # automatically (default true) and RPC layer (default grpc).
    # Reads the process rank (required) and other missing values from Slurm environment variables;
    # may run an external GPU query. Raises a runtime error if the requested GPUs exceed what a
    # node has, if the job task counts do not sum to the number of assigned tasks, or if required
    # environment values are missing. Returns nothing.
    """Creates a new SlurmClusterResolver object.

    For any parameter not set it will query the environment for the value.
    It uses those parameters to check which nodes have processes reside on and
    resolves their hostnames.
    With the number tasks per node it offsets the port number for each process.
    With the number of GPUs per node and per task it allocates GPUs to tasks by
    setting environment variables.
    Using the resolver works best (and is easier) with homogeneous tasks but
    heterogeneous tasks (number of tasks varying per node) are also possible as
    long as the number of GPUs per task stays constant.

    Used environment variables:
      - SLURM_PROCID
      - (opt) SLURM_STEP_NUM_TASKS
      - (opt) SLURM_STEP_NODELIST
      - (opt) SLURM_STEP_TASKS_PER_NODE

    Args:
      jobs: Dictionary with job names as key and number of tasks in the job as
        value. Defaults to as many 'worker's as there are (Slurm) tasks.
      port_base: The first port number to start with for processes on a node.
      gpus_per_node: Number of GPUs available on each node. Defaults to the
        number of GPUs reported by nvidia-smi
      gpus_per_task: Number of GPUs to be used for each task. Default is to
        evenly distribute the gpus_per_node to tasks_per_node.
      tasks_per_node: Number of tasks running on each node. Can be an integer if
        the number of tasks per node is constant or a dictionary mapping
        hostnames to number of tasks on that node. If not set the Slurm
        environment is queried for the correct mapping.
      auto_set_gpu: Set the visible CUDA devices automatically while resolving
        the cluster by setting CUDA_VISIBLE_DEVICES environment variable.
        Defaults to True.
      rpc_layer: The protocol TensorFlow used to communicate between nodes.
        Defaults to 'grpc'.

    Returns:
      A ClusterResolver object which can be used with distributed TensorFlow.

    Raises:
      RuntimeError: If requested more GPUs per node than available or
        requested more tasks than assigned tasks or
        resolving missing values from the environment failed.
    """

    self._rank = self._resolve_own_rank()

    if jobs is None:
      jobs = {'worker': self._resolve_num_tasks()}

    self._jobs = jobs
    self._port_base = port_base

    if tasks_per_node is None:
      self._task_configuration = self._resolve_task_configuration()
    elif isinstance(tasks_per_node, dict):
      # User can pass in an explicit configuration as a dict
      self._task_configuration = tasks_per_node
    else:
      # User can pass a fixed number of tasks per node
      hostlist = self._resolve_hostlist()
      self._task_configuration = {
          host: int(tasks_per_node) for host in hostlist
      }

    max_tasks_per_node = max(self._task_configuration.values())
    num_tasks = sum(self._task_configuration.values())

    if gpus_per_node is None:
      gpus_per_node = get_num_gpus()
    if gpus_per_task is None:
      gpus_per_task = gpus_per_node // max_tasks_per_node
    self._gpus_per_node = gpus_per_node
    self._gpus_per_task = gpus_per_task

    self._auto_set_gpu = auto_set_gpu
    self.task_type = None
    self.task_id = None
    self.rpc_layer = rpc_layer

    self._gpu_allocation = []
    self._cluster_allocation = {}

    if max_tasks_per_node * self._gpus_per_task > self._gpus_per_node:
      raise RuntimeError('Requested more GPUs per node than available.')

    if sum(self._jobs.values()) != num_tasks:
      raise RuntimeError('Requested {} tasks but only {} were assigned.'.format(
          sum(self._jobs.values()), num_tasks))

  def _resolve_own_rank(self):
    # [FUNCTION DOC]
    # HELPER: constructor glue. Returns this process's rank (zero-based integer index among the
    # step's tasks) from the Slurm process id variable; raises a runtime error if it is missing.
    """Returns the rank of the current task in range [0, num_tasks)."""
    return int(_get_slurm_var('PROCID'))

  def _resolve_num_tasks(self):
    # [FUNCTION DOC]
    # HELPER: constructor glue. Returns the number of tasks in the current step as an integer
    # from the Slurm environment.
    """Returns the number of tasks for the current job step."""
    return _get_num_slurm_tasks()

  def _resolve_hostlist(self):
    # [FUNCTION DOC]
    # HELPER: constructor glue. Returns the list of host names for the current step by expanding
    # the Slurm step node list variable.
    """Returns a list of hostnames for nodes running the current job step."""
    return expand_hostlist(_get_slurm_var('STEP_NODELIST'))

  def _resolve_task_configuration(self):
    # [FUNCTION DOC]
    # HELPER: constructor glue. Reads the step node list and tasks-per-node variables and returns
    # a mapping from host name to number of tasks, pairing them in order (extra entries of the
    # longer list are silently dropped).
    """Creates a mapping of hostnames to the number of tasks allocated on it.

    Reads the SLURM environment to determine the nodes involved in the current
    job step and number of tasks running on each node.

    Returns a dictionary mapping each hostname to the number of tasks.
    """
    hostlist = self._resolve_hostlist()
    tasks_per_node = expand_tasks_per_node(
        _get_slurm_var('STEP_TASKS_PER_NODE'))
    return {
        host: num_tasks for (host, num_tasks) in zip(hostlist, tasks_per_node)
    }

  def cluster_spec(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring refers to "instance group info" (copy-paste from another
    # resolver); the information actually comes from the constructor arguments and Slurm.
    # KNOWN ISSUE: if a node has fewer GPU slots than tasks, the pairing of ports and GPU blocks
    # silently drops the extra tasks, so fewer addresses than configured may be produced and the
    # automatic GPU assignment may index past the end of the allocation list.
    # Builds the cluster specification. Takes no arguments. Visits hosts in sorted order; each task
    # on a host gets the host name with the base port plus its index on that host, and a block of
    # consecutive GPU numbers. Then walks the jobs in sorted name order giving each a consecutive
    # slice of the address list, and sets this resolver's task type and task index when its own
    # rank falls inside a job's slice. If automatic GPU setting is on, overwrites the visible CUDA
    # devices environment variable with this rank's GPU block. Replaces the stored allocations on
    # every call. Returns the cluster specification.
    """Returns a ClusterSpec object based on the latest instance group info.

    This returns a ClusterSpec object for use based on information from the
    specified initialization parameters and Slurm environment variables. The
    cluster specification is resolved each time this function is called. The
    resolver extract hostnames of nodes by scontrol and pack tasks in that
    order until a node a has number of tasks that is equal to specification.
    GPUs on nodes are allocated to tasks by specification through setting
    CUDA_VISIBLE_DEVICES environment variable.

    Returns:
      A ClusterSpec containing host information retrieved from Slurm's
        environment variables.
    """

    task_list = []
    self._gpu_allocation = []
    self._cluster_allocation = {}

    # Sort to make sure the order is the same for each run
    for host, num_tasks in sorted(self._task_configuration.items()):
      for port_offset, gpu_offset in zip(
          range(num_tasks), range(0, self._gpus_per_node, self._gpus_per_task)):

        host_addr = '%s:%d' % (host, self._port_base + port_offset)
        task_list.append(host_addr)
        gpu_id_list = []

        for gpu_id in range(gpu_offset, gpu_offset + self._gpus_per_task):
          gpu_id_list.append(str(gpu_id))

        self._gpu_allocation.append(','.join(gpu_id_list))

    cluster_rank_offset_start = 0
    cluster_rank_offset_end = 0

    # Sort to make sure the order is the same for each run
    for task_type, num_tasks in sorted(self._jobs.items()):
      cluster_rank_offset_end = cluster_rank_offset_start + num_tasks

      self._cluster_allocation[task_type] = (
          task_list[cluster_rank_offset_start:cluster_rank_offset_end])

      if cluster_rank_offset_start <= self._rank < cluster_rank_offset_end:
        self.task_type = task_type
        self.task_id = self._rank - cluster_rank_offset_start

      cluster_rank_offset_start = cluster_rank_offset_end

    if self._auto_set_gpu:
      os.environ['CUDA_VISIBLE_DEVICES'] = self._gpu_allocation[self._rank]

    return ClusterSpec(self._cluster_allocation)

  def get_task_info(self):
    # [FUNCTION DOC]
    # Returns a pair of the job name and task index of this process. Both are null until the
    # cluster specification has been computed at least once (or they are set manually). No side
    # effects.
    """Returns job name and task_id for the process which calls this.

    This returns the job name and task index for the process which calls this
    function according to its rank and cluster specification. The job name and
    task index are set after a cluster is constructed by cluster_spec otherwise
    defaults to None.

    Returns:
      A string specifying job name the process belongs to and an integer
        specifying the task index the process belongs to in that job.
    """
    return self.task_type, self.task_id

  def master(self, task_type=None, task_id=None, rpc_layer=None):
    # [FUNCTION DOC]
    # Returns the master address string. Inputs, all optional: task type, task index and RPC
    # layer; null values fall back to the stored ones (RPC layer falls back when null or empty).
    # If both a task type and index are available, computes the cluster specification (with its
    # side effects, including possibly changing the visible CUDA devices variable and this
    # resolver's task info) and returns the task's address formatted with the RPC layer prefix.
    # Otherwise returns an empty string.
    """Returns the master string for connecting to a TensorFlow master.

    Args:
      task_type: (Optional) Overrides the default auto-selected task type.
      task_id: (Optional) Overrides the default auto-selected task index.
      rpc_layer: (Optional) Overrides the default RPC protocol TensorFlow uses
        to communicate across nodes.

    Returns:
      A connection string for connecting to a TensorFlow master.
    """
    task_type = task_type if task_type is not None else self.task_type
    task_id = task_id if task_id is not None else self.task_id

    if task_type is not None and task_id is not None:
      return format_master_url(
          self.cluster_spec().task_address(task_type, task_id),
          rpc_layer or self.rpc_layer)

    return ''

  def num_accelerators(self,
                       task_type=None,
                       task_id=None,
                       config_proto=None):
    # [FUNCTION DOC]
    # Reports accelerators for a task. All three arguments (task type, task index, session
    # configuration) are ignored. Returns a mapping with a single GPU entry whose value is the
    # configured GPUs per task.
    # Unused, since this is set in __init__ manually.
    del task_type, task_id, config_proto
    return {'GPU': self._gpus_per_task}
