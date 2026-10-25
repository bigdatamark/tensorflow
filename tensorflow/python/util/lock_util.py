# [MODULE DOC]
# HELPER: threading utility module providing a group lock, which allows many threads of one
# group to share a resource while excluding threads of other groups. No import-time side effects.
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
"""Locking related utils."""

import threading


class GroupLock(object):
  # [CLASS DOC]
  # Lock that lets any number of threads of one group hold it simultaneously while blocking
  # threads of every other group (a generalised reader-writer lock). Instances are created with
  # a number of groups and used via the context manager returned by the group method or via
  # explicit acquire/release with a group number. State (restricted to fixed slots): a condition
  # variable guarding access, the group count, and a per-group list of current member counts.
  # There is no fairness: a steady stream of one group's members can starve other groups.
  """A lock to allow many members of a group to access a resource exclusively.

  This lock provides a way to allow access to a resource by multiple threads
  belonging to a logical group at the same time, while restricting access to
  threads from all other groups. You can think of this as an extension of a
  reader-writer lock, where you allow multiple writers at the same time. We
  made it generic to support multiple groups instead of just two - readers and
  writers.

  Simple usage example with two groups accessing the same resource:

  ```python
  lock = GroupLock(num_groups=2)

  # In a member of group 0:
  with lock.group(0):
    # do stuff, access the resource
    # ...

  # In a member of group 1:
  with lock.group(1):
    # do stuff, access the resource
    # ...
  ```

  Using as a context manager with `.group(group_id)` is the easiest way. You
  can also use the `acquire` and `release` method directly.
  """

  __slots__ = ["_ready", "_num_groups", "_group_member_counts"]

  def __init__(self, num_groups=2):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring has a Returns section claiming a group lock is returned, but a
    # constructor returns nothing; the object itself is the result.
    # Initializes the lock. Input: num_groups, a positive integer count of groups (default two).
    # Raises a value error if num_groups is below one. Creates the condition variable (wrapping
    # a plain lock), stores the group count and a zeroed per-group member counter list.
    """Initialize a group lock.

    Args:
      num_groups: The number of groups that will be accessing the resource under
        consideration. Should be a positive number.

    Returns:
      A group lock that can then be used to synchronize code.

    Raises:
      ValueError: If num_groups is less than 1.
    """
    if num_groups < 1:
      raise ValueError(
          "Argument `num_groups` must be a positive integer. "
          f"Received: num_groups={num_groups}")
    self._ready = threading.Condition(threading.Lock())
    self._num_groups = num_groups
    self._group_member_counts = [0] * self._num_groups

  def group(self, group_id):
    # [FUNCTION DOC]
    # Creates a context manager for holding the lock as a member of one group.
    # Input: group_id, integer from zero up to the group count minus one. Raises a value error
    # if out of range. Returns a context-manager helper object; the lock is only acquired when
    # the context is entered, and released on exit. The entered value is nothing.
    """Enter a context where the lock is with group `group_id`.

    Args:
      group_id: The group for which to acquire and release the lock.

    Returns:
      A context manager which will acquire the lock for `group_id`.
    """
    self._validate_group_id(group_id)
    return self._Context(self, group_id)

  def acquire(self, group_id):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the internal condition is acquired and released manually without a protected
    # section, so an exception (for example an interrupt) while waiting would leave the
    # internal lock held.
    # Blocks until no member of any other group holds the lock, then registers the caller as a
    # member of group_id by incrementing that group's count. Input: group_id (integer in range;
    # otherwise a value error is raised). Returns nothing. Side effect: changes member counts.
    """Acquire the group lock for a specific group `group_id`."""
    self._validate_group_id(group_id)

    self._ready.acquire()
    while self._another_group_active(group_id):
      self._ready.wait()
    self._group_member_counts[group_id] += 1
    self._ready.release()

  def release(self, group_id):
    # [FUNCTION DOC]
    # KNOWN ISSUE: there is no check that the caller actually holds the lock; releasing without
    # a matching acquire drives the count negative and corrupts the lock state.
    # Removes one member of group_id by decrementing its count; when the count reaches zero all
    # waiting threads are woken so other groups can proceed. Input: group_id (integer in range;
    # otherwise a value error). Returns nothing.
    """Release the group lock for a specific group `group_id`."""
    self._validate_group_id(group_id)

    self._ready.acquire()
    self._group_member_counts[group_id] -= 1
    if self._group_member_counts[group_id] == 0:
      self._ready.notify_all()
    self._ready.release()

  def _another_group_active(self, group_id):
    # [FUNCTION DOC]
    # HELPER: wait-condition check for acquire.
    # Input: group_id, the caller's group. Returns true if any other group currently has at
    # least one member holding the lock, otherwise false. No side effects.
    return any(
        c > 0 for g, c in enumerate(self._group_member_counts) if g != group_id)

  def _validate_group_id(self, group_id):
    # [FUNCTION DOC]
    # HELPER: argument check shared by group, acquire and release.
    # Input: group_id, expected integer. Raises a value error (message includes the group count
    # and the received value) if it is negative or not below the group count. Returns nothing.
    if group_id < 0 or group_id >= self._num_groups:
      raise ValueError(
          "Argument `group_id` should verify `0 <= group_id < num_groups` "
          f"(with `num_groups={self._num_groups}`). "
          f"Received: group_id={group_id}")

  class _Context(object):
    # [CLASS DOC]
    # HELPER: context manager returned by GroupLock.group. Holds the lock and the group number
    # (fixed slots); entering acquires the lock for the group and exiting releases it.
    """Context manager helper for `GroupLock`."""

    __slots__ = ["_lock", "_group_id"]

    def __init__(self, lock, group_id):
      # [FUNCTION DOC]
      # HELPER: stores the group lock and the group number (both validated by the caller).
      # Returns nothing; does not acquire anything.
      self._lock = lock
      self._group_id = group_id

    def __enter__(self):
      # [FUNCTION DOC]
      # HELPER: acquires the group lock for the stored group, blocking while another group
      # holds it. Returns nothing (so a with-as target would be the null value).
      self._lock.acquire(self._group_id)

    def __exit__(self, type_arg, value_arg, traceback_arg):
      # [FUNCTION DOC]
      # HELPER: releases the group lock for the stored group. The three exception arguments are
      # ignored, so exceptions raised inside the context are not suppressed. Returns nothing.
      del type_arg, value_arg, traceback_arg
      self._lock.release(self._group_id)
