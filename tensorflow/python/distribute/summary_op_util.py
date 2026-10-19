# [MODULE DOC]
# HELPER: small helper module for summary ops running under a distribution strategy.
# Exposes a single function, skip_summary, which tells summary-writing code whether the current
# replica should skip recording so that only the first replica (replica id zero) writes
# summaries when several replicas run in one synchronized group. Depends on the replica-context
# lookup in the distribute library and on tensor constant-value extraction. No import-time side
# effects.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
#==============================================================================
"""Contains utility functions used by summary ops in distribution strategy."""


from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util


def skip_summary():
  # [FUNCTION DOC]
  # HELPER: lets summary ops decide whether to record under a distribution strategy.
  # KNOWN ISSUE: docstring promises True/False, but the final expression is a logical "and", so it
  # can return a falsy non-boolean (zero, or nothing/null when a tensor replica id is not a
  # compile-time constant) instead of False; a non-constant tensor id is thus never skipped.
  # Takes no arguments. Looks up the current replica context; if there is none (cross-replica or
  # no strategy) returns False. Otherwise reads the replica id within the sync group, resolving a
  # tensor id to its constant value when possible, and reports true when that id is greater than
  # zero, i.e. summaries are skipped on every replica except the first. No side effects, no
  # exceptions raised by this function itself.
  """Determines if summary should be skipped.

  If using multiple replicas in distributed strategy, skip summaries on all
  replicas except the first one (replica_id=0).

  Returns:
    True if the summary is skipped; False otherwise.
  """

  # TODO(priyag): Add a new optional argument that will provide multiple
  # alternatives to override default behavior. (e.g. run on last replica,
  # compute sum or mean across replicas).
  replica_context = distribute_lib.get_replica_context()
  if not replica_context:
    return False
  # TODO(b/118385803): when replica_id of _TPUReplicaContext is properly
  # initialized, remember to change here as well.
  replica_id = replica_context.replica_id_in_sync_group
  if isinstance(replica_id, tensor.Tensor):
    replica_id = tensor_util.constant_value(replica_id)
  return replica_id and replica_id > 0
