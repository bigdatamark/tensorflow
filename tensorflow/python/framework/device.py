# [MODULE DOC]
# Defines the device abstraction used by the framework. It picks the DeviceSpec implementation at
# import time: the version 2 (immutable) spec when TF2 behavior is enabled, otherwise the version 1
# (mutable) spec from the device_spec module, and exposes it under the single name DeviceSpec.
# Also provides helpers to validate a spec string, test whether an object is a spec, canonicalize
# a device name, and build cached merge callables (MergeDevice) used by device scopes to combine
# partial device specifications. Import-time side effects: reads the TF2 enabled flag and creates
# two module-level caches (merger objects and merged strings) that are never evicted.
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

"""Class to represent a device."""

from tensorflow.python import tf2
from tensorflow.python.framework import device_spec

if tf2.enabled():
  DeviceSpec = device_spec.DeviceSpecV2
else:
  DeviceSpec = device_spec.DeviceSpecV1


def check_valid(spec):
  # [FUNCTION DOC]
  # Validates a device specification string by parsing it with the active DeviceSpec class.
  # Input: spec, a device string. Returns nothing. Raises whatever the parser raises (normally a
  # value error) when the string is malformed. No side effects.
  """Check that a device spec is valid.

  Args:
    spec: a string.

  Raises:
    An exception if the spec is invalid.
  """
  # Construct a DeviceSpec.  It will assert a failure if spec is invalid.
  DeviceSpec.from_string(spec)


def is_device_spec(obj):
  # [FUNCTION DOC]
  # Returns true when obj is an instance of the version 2 device spec class (the common base of
  # both spec versions), false otherwise. Input: any object. No side effects.
  """Abstract away the fact that DeviceSpecV2 is the base class."""
  return isinstance(obj, device_spec.DeviceSpecV2)


def canonical_name(device):
  # [FUNCTION DOC]
  # Converts a device (nothing, a device spec object, or a device string) to its canonical string
  # form. Nothing yields the empty string; a spec is serialized directly; a string is parsed with
  # the active DeviceSpec class and re-serialized. Raises a parse error for an invalid string.
  """Returns a canonical name for the given `DeviceSpec` or device name."""
  if device is None:
    return ""
  if is_device_spec(device):
    return device.to_string()
  else:
    device = DeviceSpec.from_string(device)
    return device.to_string()


# Performance caches
_cached_mergers = {}
_string_merge_cache = {}


def merge_device(spec):
  # [FUNCTION DOC]
  # Returns a MergeDevice callable for the given partial device specification. If spec is already
  # a MergeDevice it is returned unchanged. Otherwise a module-level cache keyed by spec is
  # consulted, and a new MergeDevice is built and stored on a miss (no locking is used).
  # Input: spec, a device spec object or string. Raises a value error if the spec is invalid.
  # Side effect: grows the module-level merger cache; the spec must be hashable.
  """Returns a device function that merges devices specifications.

  This can be used to merge partial specifications of devices. The
  innermost setting for a device field takes precedence. For example:

    with tf.device(merge_device("/device:GPU:0"))
      # Nodes created here have device "/device:GPU:0"
      with tf.device(merge_device("/job:worker")):
        # Nodes created here have device "/job:worker/device:GPU:0"
        with tf.device(merge_device("/device:CPU:0")):
          # Nodes created here have device "/job:worker/device:CPU:0"
          with tf.device(merge_device("/job:ps")):
            # Nodes created here have device "/job:ps/device:CPU:0"

  Args:
    spec: A `DeviceSpec` or a device spec string (partially) describing the
      device that should be used for all nodes created in the scope of
      the returned device function's with block.

  Returns:
    A MergeDevice object with the above-described behavior.

  Raises:
    ValueError: if the spec was not valid.
  """

  if isinstance(spec, MergeDevice):
    return spec

  merger = _cached_mergers.get(spec)
  if merger:
    return merger
  merger = MergeDevice(spec)
  # No locking needed, since updates are stateless.
  _cached_mergers[spec] = merger
  return merger


class MergeDevice(object):
  # [CLASS DOC]
  # Callable wrapper around a single device specification that merges it into the device of a
  # node definition. Holds one attribute, the wrapped spec (slots are used to keep it small).
  # Created by merge_device and used by device scopes; also offers a string-level merge with a
  # shared cache for speed, and a property telling whether the wrapped spec is empty.
  """Wraps a device specification (DeviceSpec or str) with merge functionality.

  When called, this class will merge a node_def with its own spec. It also
  exposes a `shortcut_string_merge` method which can significantly improve
  performance of device placement.
  """

  __slots__ = ["_spec"]

  def __init__(self, spec):
    # [FUNCTION DOC]
    # Stores the spec as a DeviceSpec. A version 2 spec is kept as is; a version 1 (mutable) spec
    # is snapshotted by serializing and re-parsing it; any other input (a string) is parsed.
    # Raises a parse error for an invalid string. Returns nothing.
    if isinstance(spec, device_spec.DeviceSpecV2):
      self._spec = spec
    elif isinstance(spec, device_spec.DeviceSpecV1):
      # Capture a snapshot of spec.
      self._spec = spec.__class__.from_string(spec.to_string())
    else:
      self._spec = DeviceSpec.from_string(spec)

  def __call__(self, node_def):
    # [FUNCTION DOC]
    # Merges the wrapped spec with the device already set on node_def (empty if unset) and returns
    # the resulting DeviceSpec. Fields set in the current node device take precedence over those
    # in the wrapped spec, as defined by the spec's merged-spec operation. The node is not
    # modified. Input: node_def, an operation-like object with a device attribute.
    # In general a user may create a device function which takes into account
    # arbitrary properties of an op. (For instance dynamically placing ops based
    # on type.) So even though the standard DeviceSpec route only uses the
    # device attribute, we take an entire node_def to maintain a consistent
    # signature with general device functions.
    current_device = DeviceSpec.from_string(node_def.device or "")
    return self._spec.make_merged_spec(current_device)

  def shortcut_string_merge(self, node_def):
    # [FUNCTION DOC]
    # Returns the merged device as a string, using a module-level cache keyed by the wrapped spec
    # and the node's current device string so no DeviceSpec needs to be materialized on a hit.
    # On a miss it performs the normal merge, stores the string and returns it.
    # Input: node_def, an operation-like object with a device attribute. Side effect: updates the
    # shared string cache (not locked, since results are deterministic).
    """Merge a node def without materializing a full DeviceSpec object.

    Often a device merge is invoked in order to generate a string which can be
    passed into the c api. In such a case, we can cache the
      node_def.device  ->  merge_result_string

    map, and in most cases avoid:
      - Materializing a copy of self._spec (In the case of DeviceSpecV1)
      - Materializing a DeviceSpec for node_def.device
      - A DeviceSpec.merge_from invocation

    In practice the cache hit rate for this function is very high, because the
    number of invocations when iterating through the device stack is much
    larger than the number of devices.

    Args:
      node_def: An Operation (or Operation-like) to merge device constraints
        with self._spec

    Returns:
      A string containing the merged device specification.
    """
    device = node_def.device or ""

    merge_key = (self._spec, device)
    result = _string_merge_cache.get(merge_key)
    if result is None:
      # This update is not atomic, however because the merge is stateless
      # we don't need to lock when updating the cache.
      result = self.__call__(node_def).to_string()
      _string_merge_cache[merge_key] = result

    return result

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns the default object representation followed by the wrapped spec in string form.
    return "{} (spec: {})".format(
        super(MergeDevice, self).__repr__(), self._spec.to_string())

  @property
  def is_null_merge(self):
    # [FUNCTION DOC]
    # Property that is true when the wrapped spec serializes to an empty string, meaning merging
    # would change nothing. Takes no inputs and has no side effects.
    """Indicate whether the wrapped spec is empty.

    In the degenerate case where self._spec is an empty specification, a caller
    may wish to skip a merge step entirely. (However this class does not have
    enough information to make that determination.)

    Returns:
      A boolean indicating whether a device merge will be trivial.
    """
    return not bool(self._spec.to_string())
