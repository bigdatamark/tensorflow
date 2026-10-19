# [MODULE DOC]
# Helper functions for working with device name strings in the distribute package: canonicalizing
# partial names into full names, resolving a name against the current device scope, finding the
# current device, deriving a host CPU device from a device, and listing local GPU names.
# Also defines two private fake classes that mimic an operation so the current graph device
# functions can be evaluated. No import-time side effects. Used widely by distribution strategies.

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
"""Device-related support functions."""


from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import ops


def canonicalize(d, default=None):
  # [FUNCTION DOC]
  # Turns a possibly partial device name into a fully specified one. Inputs: d, a device name
  # string or a logical device object; default, optional device name string whose fields
  # override the built-in defaults. Starts from replica 0, task 0, CPU, index 0; when executing
  # eagerly (outside functions) also fills in job, replica and task from the first logical CPU,
  # or uses the local host job name if that CPU has no job. Then merges the default, and finally d
  # so that d wins. Returns the canonical device string. Raises an assertion error if the device
  # type in d is not all upper case. Reads the global runtime context; no mutation.
  """Canonicalize device string.

  If d has missing components, the rest would be deduced from the `default`
  argument or from '/replica:0/task:0/device:CPU:0'. For example:
    If d = '/cpu:0', default='/job:worker/task:1', it returns
      '/job:worker/replica:0/task:1/device:CPU:0'.
    If d = '/cpu:0', default='/job:worker', it returns
      '/job:worker/replica:0/task:0/device:CPU:0'.
    If d = '/gpu:0', default=None, it returns
      '/replica:0/task:0/device:GPU:0'.

  Note: This uses "job:localhost" as the default if executing eagerly.

  Args:
    d: a device string or tf.config.LogicalDevice
    default: a string for default device if d doesn't have all components.

  Returns:
    a canonicalized device string.
  """
  if isinstance(d, context.LogicalDevice):
    d = tf_device.DeviceSpec.from_string(d.name)
  else:
    d = tf_device.DeviceSpec.from_string(d)

  assert d.device_type is None or d.device_type == d.device_type.upper(), (
      "Device type '%s' must be all-caps." % (d.device_type,))
  # Fill in missing device fields using defaults.
  result = tf_device.DeviceSpec(
      replica=0, task=0, device_type="CPU", device_index=0)
  if ops.executing_eagerly_outside_functions():
    # Try to deduce job, replica and task in case it's in a multi worker setup.
    # TODO(b/151452748): Using list_logical_devices is not always safe since it
    # may return remote devices as well, but we're already doing this elsewhere.
    host_cpu = tf_device.DeviceSpec.from_string(
        config.list_logical_devices("CPU")[0].name)
    if host_cpu.job:
      result = result.make_merged_spec(host_cpu)
    else:
      # The default job is localhost if eager execution is enabled
      result = result.replace(job="localhost")
  if default:
    # Overrides any defaults with values from the default device if given.
    result = result.make_merged_spec(
        tf_device.DeviceSpec.from_string(default))

  # Apply `d` last, so that it's values take precedence over the defaults.
  result = result.make_merged_spec(d)
  return result.to_string()


def canonicalize_without_job_and_task(d):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring examples mention a default argument, but this function accepts
  # only d and takes no default.
  # Canonicalizes d (device name string or logical device object) and then clears the job and
  # task fields and sets the replica to 0. Returns the resulting device string. Useful when names
  # made on a coordinator are used on workers.
  """Partially canonicalize device string.

  This returns device string from `d` without including job and task.
  This is most useful for parameter server strategy where the device strings are
  generated on the chief, but executed on workers.

   For example:
    If d = '/cpu:0', default='/job:worker/task:1', it returns
      '/replica:0/device:CPU:0'.
    If d = '/cpu:0', default='/job:worker', it returns
      '/replica:0/device:CPU:0'.
    If d = '/gpu:0', default=None, it returns
      '/replica:0/device:GPU:0'.

  Note: This uses "job:localhost" as the default if executing eagerly.

  Args:
    d: a device string or tf.config.LogicalDevice

  Returns:
    a partially canonicalized device string.
  """
  canonicalized_device = canonicalize(d)
  spec = tf_device.DeviceSpec.from_string(canonicalized_device)
  spec = spec.replace(job=None, task=None, replica=0)
  return spec.to_string()


def resolve(d):
  # [FUNCTION DOC]
  # Canonicalizes device d using the current device (from the active device scope) as the default.
  # Returns the full device string. See the canonicalize and current functions.
  """Canonicalize `d` with current device as default."""
  return canonicalize(d, default=current())


class _FakeNodeDef(object):
  # [CLASS DOC]
  # HELPER: stand-in graph node definition used by the fake operation in this module.
  # Holds only an operation type and a name, both empty strings; uses fixed slots.
  """A fake NodeDef for _FakeOperation."""

  __slots__ = ["op", "name"]

  def __init__(self):
    # [FUNCTION DOC]
    # Initializes the operation type and name to empty strings.
    self.op = ""
    self.name = ""


class _FakeOperation(object):
  # [CLASS DOC]
  # HELPER: minimal stand-in for a graph operation, used by the current-device lookup.
  # Has a device string, type, name and a fake node definition. Passed to the graph's device
  # functions so they can write the device they would assign.
  """A fake Operation object to pass to device functions."""

  def __init__(self):
    # [FUNCTION DOC]
    # Initializes empty device, type and name strings and a fresh fake node definition.
    self.device = ""
    self.type = ""
    self.name = ""
    self.node_def = _FakeNodeDef()

  def _set_device(self, device):
    # [FUNCTION DOC]
    # HELPER: callback used by graph device functions to assign a device to the fake operation.
    # Takes a device specification and stores its string form as the device. Returns nothing.
    self.device = ops._device_string(device)  # pylint: disable=protected-access

  def _set_device_from_string(self, device_str):
    # [FUNCTION DOC]
    # HELPER: callback used by graph device functions to assign a device given as a string.
    # Stores the string directly as the device. Returns nothing.
    self.device = device_str


def current():
  # [FUNCTION DOC]
  # Returns the current device name as a string, not canonicalized. When executing eagerly
  # outside functions, returns the device name from the runtime context. Otherwise applies the
  # default graph's device functions to a fake operation and returns the device it ends up with.
  # Takes no inputs; may return an empty string when no device scope is active in graph mode.
  """Return a string (not canonicalized) for the current device."""
  # TODO(josh11b): Work out how this function interacts with ops.colocate_with.
  if ops.executing_eagerly_outside_functions():
    d = context.context().device_name
  else:
    op = _FakeOperation()
    ops.get_default_graph()._apply_device_functions(op)  # pylint: disable=protected-access
    d = op.device
  return d


def get_host_for_device(device, device_index=0):
  # [FUNCTION DOC]
  # Computes the host CPU device in the same job, replica and task as the given device.
  # Inputs: device, a device name string; device_index, optional CPU index (default 0).
  # Returns the host device string.
  """Returns the corresponding host device for the given device."""
  spec = tf_device.DeviceSpec.from_string(device)
  return tf_device.DeviceSpec(
      job=spec.job,
      replica=spec.replica,
      task=spec.task,
      device_type="CPU",
      device_index=device_index,
  ).to_string()


def local_devices_from_num_gpus(num_gpus):
  # [FUNCTION DOC]
  # Input: num_gpus, the number of local GPUs. Returns a tuple of GPU device strings with indices
  # zero up to num_gpus minus one, or a single-element tuple holding the first CPU device when
  # num_gpus is zero.
  """Returns device strings for local GPUs or CPU."""
  return (tuple("/device:GPU:%d" % i for i in range(num_gpus)) or
          ("/device:CPU:0",))
