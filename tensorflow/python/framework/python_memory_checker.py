# [MODULE DOC]
# HELPER: internal helper module behind the public memory checker wrapper (used by tests to
# detect leaks of Python objects).
# Implements Python object leak detection by taking snapshots of all objects tracked by the
# garbage collector (grouped by readable type name, storing object identities) and comparing
# snapshots. Exposes a private checker class that records snapshots, and can assert that no new
# objects appeared beyond a threshold or that no object type grew in every snapshot (possibly
# except one). Snapshot recording goes through a native helper so that native allocations made by
# the checker itself can be filtered from native leak checks. Not meant to be used directly.
# No import-time side effects beyond imports.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Python memory leak detection utility.

Please don't use this class directly.  Instead, use `MemoryChecker` wrapper.
"""

import collections
import copy
import gc

from tensorflow.python.framework import _python_memory_checker_helper
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace


def _get_typename(obj):
  # [FUNCTION DOC]
  # HELPER: used by the snapshot code to group objects by type.
  # Input: obj, any object. Returns the fully qualified type name as module name, a dot and the
  # type name, or only the type name if the type has no module. No side effects.
  """Return human readable pretty type name string."""
  objtype = type(obj)
  name = objtype.__name__
  module = getattr(objtype, '__module__', None)
  if module:
    return '{}.{}'.format(module, name)
  else:
    return name


def _create_python_object_snapshot():
  # [FUNCTION DOC]
  # HELPER: used by the memory checker to record the state of live objects.
  # Runs a full garbage collection, then enumerates all objects tracked by the collector and
  # builds a mapping from type name to the set of identities of objects of that type. No
  # inputs. Returns that mapping (a default-valued mapping, so missing type names give empty
  # sets). Side effect: triggers garbage collection.
  gc.collect()
  all_objects = gc.get_objects()
  result = collections.defaultdict(set)
  for obj in all_objects:
    result[_get_typename(obj)].add(id(obj))
  return result


def _snapshot_diff(old_snapshot, new_snapshot, exclude_ids):
  # [FUNCTION DOC]
  # HELPER: used by the checker's snapshot comparison method.
  # Computes per-type growth between two snapshots. Inputs: old_snapshot and new_snapshot,
  # mappings from type name to sets of object identities; exclude_ids, a set of identities (the
  # checker's own bookkeeping objects) ignored on both sides. For each type in the new snapshot
  # the result is the count of new objects minus the count of old ones, after exclusion. Types
  # present only in the old snapshot are ignored, and reading the old snapshot for a missing
  # type adds an empty entry to it (a default-valued mapping). Returns a counter containing only
  # strictly positive entries.
  result = collections.Counter()
  for new_name, new_ids in new_snapshot.items():
    old_ids = old_snapshot[new_name]
    result[new_name] = len(new_ids - exclude_ids) - len(old_ids - exclude_ids)

  # This removes zero or negative value entries.
  result += collections.Counter()
  return result


class _PythonMemoryChecker(object):
  # [CLASS DOC]
  # HELPER: private implementation behind the public memory checker wrapper.
  # Holds a list of snapshots of live Python objects (taken by record_snapshot) and offers
  # assertions over them: no new objects beyond a threshold between the first and last snapshot,
  # and no object type that grows in every consecutive snapshot pair (possibly except one). A
  # report method exists but is unimplemented. Most methods are wrapped with a trace decorator.
  """Python memory leak detection class."""

  def __init__(self):
    # [FUNCTION DOC]
    # Creates a checker with an empty list of snapshots and prepares a reusable callback that
    # appends a new snapshot to that list. The callback is created once and stored so that
    # creating it later does not itself show up as a new object in the measurement. No inputs;
    # returns nothing.
    self._snapshots = []
    # cache the function used by mark_stack_trace_and_call to avoid
    # contaminating the leak measurement.
    def _record_snapshot():
      # [FUNCTION DOC]
      # HELPER: inner callback stored by the constructor. Takes a fresh object snapshot and
      # appends it to the checker's snapshot list. No inputs; returns nothing.
      self._snapshots.append(_create_python_object_snapshot())

    self._record_snapshot = _record_snapshot

  # We do not enable trace_wrapper on this function to avoid contaminating
  # the snapshot.
  def record_snapshot(self):
    # [FUNCTION DOC]
    # Records a new snapshot of live Python objects by invoking the stored callback through the
    # native marking helper, so native allocations made while snapshotting are recognizable in
    # native stack traces and can be filtered out. Deliberately not trace-wrapped, to avoid
    # contaminating the snapshot. No inputs; returns nothing. Side effects: runs garbage
    # collection and appends to the snapshot list.
    # Function called using `mark_stack_trace_and_call` will have
    # "_python_memory_checker_helper" string in the C++ stack trace.  This will
    # be used to filter out C++ memory allocations caused by this function,
    # because we are not interested in detecting memory growth caused by memory
    # checker itself.
    _python_memory_checker_helper.mark_stack_trace_and_call(
        self._record_snapshot)

  @trace.trace_wrapper
  def report(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: unimplemented stub (marked with a TODO); it does nothing and reports nothing.
    # Intended to report memory findings. Takes no inputs and returns nothing.
    # TODO(kkb): Implement.
    pass

  @trace.trace_wrapper
  def assert_no_leak_if_all_possibly_except_one(self):
    # [FUNCTION DOC]
    # Leak assertion over the recorded snapshots. Computes the per-type growth between each
    # consecutive pair of snapshots, counts for each type in how many pairs it grew, and treats a
    # type as leaking if it grew in at least all but one of the pairs. Takes no inputs. Returns
    # nothing. Raises an assertion error listing the leaking type names if any exist. With fewer
    # than two snapshots there are no pairs and no type is counted, so nothing is raised.
    """Raises an exception if a leak is detected.

    This algorithm classifies a series of allocations as a leak if it's the same
    type at every snapshot, but possibly except one snapshot.
    """

    snapshot_diffs = []
    for i in range(0, len(self._snapshots) - 1):
      snapshot_diffs.append(self._snapshot_diff(i, i + 1))

    allocation_counter = collections.Counter()
    for diff in snapshot_diffs:
      for name, count in diff.items():
        if count > 0:
          allocation_counter[name] += 1

    leaking_object_names = {
        name for name, count in allocation_counter.items()
        if count >= len(snapshot_diffs) - 1
    }

    if leaking_object_names:
      object_list_to_print = '\n'.join(
          [' - ' + name for name in leaking_object_names])
      raise AssertionError(
          'These Python objects were allocated in every snapshot possibly '
          f'except one.\n\n{object_list_to_print}')

  @trace.trace_wrapper
  def assert_no_new_objects(self, threshold=None):
    # [FUNCTION DOC]
    # Asserts that the number of new objects per type between the first and the last snapshot
    # stays within an allowed threshold. Input: threshold, optional mapping from type name to the
    # allowed growth (default none, meaning an empty mapping, so no growth allowed). Computes
    # the growth, subtracts the threshold, and raises an assertion error listing the threshold
    # and the raw growth (most common first) if any type exceeds it. If instead growth was
    # below the threshold for some type, logs a warning with the same details. Raises an
    # index error if no snapshots were recorded. Returns nothing.
    """Assert no new Python objects."""

    if not threshold:
      threshold = {}

    count_diff = self._snapshot_diff(0, -1)
    original_count_diff = copy.deepcopy(count_diff)
    count_diff.subtract(collections.Counter(threshold))

    if max(count_diff.values() or [0]) > 0:
      raise AssertionError('New Python objects created exceeded the threshold.'
                           '\nPython object threshold:\n'
                           f'{threshold}\n\nNew Python objects:\n'
                           f'{original_count_diff.most_common()}')
    elif min(count_diff.values(), default=0) < 0:
      logging.warning('New Python objects created were less than the threshold.'
                      '\nPython object threshold:\n'
                      f'{threshold}\n\nNew Python objects:\n'
                      f'{original_count_diff.most_common()}')

  @trace.trace_wrapper
  def _snapshot_diff(self, old_index, new_index):
    # [FUNCTION DOC]
    # HELPER: used by the assertion methods.
    # Inputs: old_index and new_index, positions (negative allowed) into the snapshot list.
    # Returns the per-type growth counter between those two snapshots via the module-level
    # function, excluding the checker's own bookkeeping objects. Raises an index error for
    # invalid positions.
    return _snapshot_diff(self._snapshots[old_index],
                          self._snapshots[new_index],
                          self._get_internal_object_ids())

  @trace.trace_wrapper
  def _get_internal_object_ids(self):
    # [FUNCTION DOC]
    # HELPER: used when comparing snapshots, so the checker's own data is not counted as leaks.
    # Collects the identities of every stored snapshot mapping and of each per-type identity set
    # inside them. No inputs. Returns the set of identities.
    ids = set()
    for snapshot in self._snapshots:
      ids.add(id(snapshot))
      for v in snapshot.values():
        ids.add(id(v))
    return ids
