# [MODULE DOC]
# HELPER: helper module for the automatic control dependency machinery.
# Utilities describing how operations use resource handles: a registry of op types known to only
# read their resources, the attribute names used for read-only resource input indices and for
# collective manager ids, and functions that classify an operation's resource inputs as reads or
# writes. Import-time side effect: creates the module-level set of read-only resource op types.
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
"""Utilities for AutomaticControlDependencies."""

from tensorflow.python.framework import dtypes
from tensorflow.python.util import object_identity

READ_ONLY_RESOURCE_INPUTS_ATTR = "_read_only_resource_inputs"
RESOURCE_READ_OPS = set()


COLLECTIVE_MANAGER_IDS = "_collective_manager_ids"


def register_read_only_resource_op(op_type):
  # [FUNCTION DOC]
  # Adds an op type name (a string) to the module-level set of ops that never update the
  # resources they touch. Returns nothing. Side effect: mutates global state shared with the
  # automatic control dependency logic.
  """Declares that `op_type` does not update its touched resource."""
  RESOURCE_READ_OPS.add(op_type)


def get_read_only_resource_input_indices_graph(func_graph):
  # [FUNCTION DOC]
  # Given a function graph, returns the ascending list of indices (into its inputs) of resource
  # inputs that are only read: every consumer op must treat that input as read-only. Non-resource
  # inputs are skipped. Per-op results are cached within the call. A resource with no consumers
  # counts as read-only. No side effects.
  """Returns sorted list of read-only resource indices in func_graph.inputs."""
  result = []
  # A cache to store the read only resource inputs of an Op.
  # Operation -> ObjectIdentitySet of resource handles.
  op_read_only_resource_inputs = {}
  for input_index, t in enumerate(func_graph.inputs):
    if t.dtype != dtypes.resource:
      continue
    read_only = True
    for op in t.consumers():
      if op in op_read_only_resource_inputs:
        if t not in op_read_only_resource_inputs[op]:
          read_only = False
          break
      else:
        indices = _get_read_only_resource_input_indices_op(op)
        op_read_only_resource_inputs[op] = object_identity.ObjectIdentitySet(
            [op.inputs[i] for i in indices])
        if t not in op_read_only_resource_inputs[op]:
          read_only = False
          break
    if read_only:
      result.append(input_index)
  return result


def _get_read_only_resource_input_indices_op(op):
  # [FUNCTION DOC]
  # HELPER: per-op worker for get_read_only_resource_input_indices_graph.
  # Returns the sorted list of positions in the op's inputs that hold read-only resource
  # handles. For registered read-only op types, every resource-typed input position is returned.
  # Otherwise the read-only indices attribute is consulted; if it is missing, returns an empty
  # list (all resources treated as written). Walks inputs in order matching resource-typed inputs
  # to the attribute's entries. No side effects.
  """Returns sorted list of read-only resource indices in op.inputs."""
  if op.type in RESOURCE_READ_OPS:
    return [i for i, t in enumerate(op.inputs) if t.dtype == dtypes.resource]

  try:
    read_only_input_indices = op.get_attr(READ_ONLY_RESOURCE_INPUTS_ATTR)
  except ValueError:
    # Attr was not set. Add all resource inputs to `writes` and return.
    return []

  read_only_index = 0
  result = []
  for i, t in enumerate(op.inputs):
    if read_only_index >= len(read_only_input_indices):
      break
    if op.inputs[i].dtype != dtypes.resource:
      continue
    if (read_only_index < len(read_only_input_indices) and
        i == read_only_input_indices[read_only_index]):
      result.append(i)
      read_only_index += 1

  return result


def get_read_write_resource_inputs(op):
  # [FUNCTION DOC]
  # Classifies the resource-typed inputs of an operation. Input is a graph operation. Returns a
  # pair of identity sets: read-only handles and read-write handles. Ops of registered read-only
  # types have all resource inputs as reads. Otherwise the read-only indices attribute (indices
  # into the inputs list, ascending) decides; if the attribute is absent, all resource inputs are
  # writes. Non-resource inputs are ignored. No side effects.
  """Returns a tuple of resource reads, writes in op.inputs.

  Args:
    op: Operation

  Returns:
    A 2-tuple of ObjectIdentitySets, the first entry containing read-only
    resource handles and the second containing read-write resource handles in
    `op.inputs`.
  """
  reads = object_identity.ObjectIdentitySet()
  writes = object_identity.ObjectIdentitySet()

  if op.type in RESOURCE_READ_OPS:
    # Add all resource inputs to `reads` and return.
    reads.update(t for t in op.inputs if t.dtype == dtypes.resource)
    return (reads, writes)

  try:
    read_only_input_indices = op.get_attr(READ_ONLY_RESOURCE_INPUTS_ATTR)
  except ValueError:
    # Attr was not set. Add all resource inputs to `writes` and return.
    writes.update(t for t in op.inputs if t.dtype == dtypes.resource)
    return (reads, writes)

  read_only_index = 0
  for i, t in enumerate(op.inputs):
    if op.inputs[i].dtype != dtypes.resource:
      continue
    if (read_only_index < len(read_only_input_indices) and
        i == read_only_input_indices[read_only_index]):
      reads.add(op.inputs[i])
      read_only_index += 1
    else:
      writes.add(op.inputs[i])
  return (reads, writes)


def _op_writes_to_resource(handle, op):
  # [FUNCTION DOC]
  # HELPER: internal predicate for resource access analysis (appears unused within this module).
  # Inputs: a resource handle that must be an input of the op, and the operation. Returns false
  # if the op type is registered as read-only, or if the handle's input position is listed in
  # the op's read-only resource input indices attribute; returns true if that attribute is
  # missing (conservative assumption) or the position is not listed. Raises a value error if the
  # handle is not an input of the op (when the op is not a registered read-only type).
  """Returns whether op writes to resource handle.

  Args:
    handle: Resource handle. Must be an input of `op`.
    op: Operation.

  Returns:
    Returns False if op is a read-only op registered using
    `register_read_only_resource_op` or if `handle` is an input at one of
    the indices in the `READ_ONLY_RESOURCE_INPUTS_ATTR` attr of the op, True
    otherwise.

  Raises:
    ValueError: if `handle` is not an input of `op`.
  """
  if op.type in RESOURCE_READ_OPS:
    return False
  input_index = _input_index(op, handle)
  try:
    read_only_input_indices = op.get_attr(READ_ONLY_RESOURCE_INPUTS_ATTR)
  except ValueError:
    # Attr was not set. Conservatively assume that the resource is written to.
    return True
  return input_index not in read_only_input_indices


def _input_index(op, handle):
  # [FUNCTION DOC]
  # HELPER: lookup used when deciding whether a given resource handle is written by an op.
  # KNOWN ISSUE: the docstring says the error is raised when the handle and its replicated input
  # are both missing, but the code only checks for the handle itself by identity.
  # Scans the inputs of the operation and returns the position of the first input that is the
  # very same object as the handle (identity comparison, not equality). Raises a value error
  # if the handle is not among the inputs.
  """Returns the index of `handle` in `op.inputs`.

  Args:
    op: Operation.
    handle: Resource handle.

  Returns:
    Index in `op.inputs` receiving the resource `handle`.

  Raises:
    ValueError: If handle and its replicated input are both not found in
    `op.inputs`.
  """
  for i, t in enumerate(op.inputs):
    if handle is t:
      return i
  raise ValueError(f"{handle!s} not in list of inputs for op: {op!r}")
