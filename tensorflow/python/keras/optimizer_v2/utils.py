# [MODULE DOC]
# HELPER: utility module for the legacy Keras optimizer framework.
# Provides gradient helpers used by the optimizers: all-reduce summation of gradients across
# replicas, removal of pairs with missing gradients (with a warning), factories that build
# gradient transformation functions for clipping by norm, global norm or value, and a check of
# whether the active distribution strategy can run without merge calls. No import-time side effects.
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
"""Optimizer utilities."""

from tensorflow.python.distribute import central_storage_strategy
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import reduce_util as ds_reduce_util
from tensorflow.python.ops import clip_ops
from tensorflow.python.platform import tf_logging as logging


def all_reduce_sum_gradients(grads_and_vars):
  # [FUNCTION DOC]
  # HELPER: gradient aggregation step for the optimizers in distributed training.
  # Input: an iterable of (gradient, variable) pairs, where gradients may be nothing. Pairs with
  # missing gradients are filtered out (which may raise a value error if all are missing), the
  # rest are summed across replicas (directly in replica context if the strategy supports it,
  # otherwise through a merge call), and the missing-gradient pairs are put back in original order
  # with a nothing gradient. Returns a list of (reduced gradient, variable) pairs. An assertion
  # fails if the number of reduced gradients does not match. Logs a warning via the filter helper.
  """Returns all-reduced gradients aggregated via summation.

  Args:
    grads_and_vars: List of (gradient, variable) pairs.

  Returns:
    List of (gradient, variable) pairs where gradients have been all-reduced.
  """
  grads_and_vars = list(grads_and_vars)
  filtered_grads_and_vars = filter_empty_gradients(grads_and_vars)
  if filtered_grads_and_vars:
    if strategy_supports_no_merge_call():
      grads = [pair[0] for pair in filtered_grads_and_vars]
      reduced = distribute_lib.get_strategy().extended._replica_ctx_all_reduce(  # pylint: disable=protected-access
          ds_reduce_util.ReduceOp.SUM, grads)
    else:
      # TODO(b/183257003): Remove this branch
      reduced = distribute_lib.get_replica_context().merge_call(
          _all_reduce_sum_fn, args=(filtered_grads_and_vars,))
  else:
    reduced = []
  # Copy 'reduced' but add None gradients back in
  reduced_with_nones = []
  reduced_pos = 0
  for g, v in grads_and_vars:
    if g is None:
      reduced_with_nones.append((None, v))
    else:
      reduced_with_nones.append((reduced[reduced_pos], v))
      reduced_pos += 1
  assert reduced_pos == len(reduced), "Failed to add all gradients"
  return reduced_with_nones


def filter_empty_gradients(grads_and_vars):
  # [FUNCTION DOC]
  # HELPER: removes pairs without gradients before applying updates.
  # Input: iterable of (gradient, variable) pairs. Returns a tuple of the pairs whose gradient is
  # not nothing (an empty input is returned as an empty tuple). Raises a value error naming the
  # variables if no pair has a gradient. Logs a warning listing variables that had no gradient.
  """Filter out `(grad, var)` pairs that have a gradient equal to `None`."""
  grads_and_vars = tuple(grads_and_vars)
  if not grads_and_vars:
    return grads_and_vars

  filtered = []
  vars_with_empty_grads = []
  for grad, var in grads_and_vars:
    if grad is None:
      vars_with_empty_grads.append(var)
    else:
      filtered.append((grad, var))
  filtered = tuple(filtered)

  if not filtered:
    raise ValueError("No gradients provided for any variable: %s." %
                     ([v.name for _, v in grads_and_vars],))
  if vars_with_empty_grads:
    logging.warning(
        ("Gradients do not exist for variables %s when minimizing the loss."),
        ([v.name for v in vars_with_empty_grads]))
  return filtered


def make_gradient_clipnorm_fn(clipnorm):
  # [FUNCTION DOC]
  # HELPER: factory for the per-gradient norm clipping transformation used by optimizers.
  # Input: clipnorm, a number or nothing. If nothing, returns an identity function on the pairs.
  # Otherwise returns a function that clips each gradient individually to the given norm and
  # returns the new list of (gradient, variable) pairs.
  """Creates a gradient transformation function for clipping by norm."""
  if clipnorm is None:
    return lambda grads_and_vars: grads_and_vars

  def gradient_clipnorm_fn(grads_and_vars):
    # [FUNCTION DOC]
    # HELPER: the transformation returned by the per-gradient clip-by-norm factory.
    # KNOWN ISSUE: the error message misspells the strategy name as "CenteralStorageStrategy".
    # Input: list of (gradient, variable) pairs. Raises a value error if the current distribution
    # strategy is central storage. Returns a new list with each gradient clipped to the norm.

    if isinstance(distribute_lib.get_strategy(),
                  (central_storage_strategy.CentralStorageStrategy,
                   central_storage_strategy.CentralStorageStrategyV1)):
      raise ValueError(
          "`clipnorm` is not supported with `CenteralStorageStrategy`")

    clipped_grads_and_vars = [
        (clip_ops.clip_by_norm(g, clipnorm), v) for g, v in grads_and_vars
    ]
    return clipped_grads_and_vars

  return gradient_clipnorm_fn


def make_global_gradient_clipnorm_fn(clipnorm):
  # [FUNCTION DOC]
  # HELPER: factory for the global-norm clipping transformation used by optimizers.
  # KNOWN ISSUE: the docstring is copied from the per-gradient clip-by-norm factory; this one
  # clips all gradients jointly by their global norm, not each by its own norm.
  # Input: clipnorm, a number or nothing. If nothing, returns an identity function on the pairs.
  # Otherwise returns a function that rescales all gradients together so their global norm is at
  # most clipnorm and returns the new list of (gradient, variable) pairs.
  """Creates a gradient transformation function for clipping by norm."""
  if clipnorm is None:
    return lambda grads_and_vars: grads_and_vars

  def gradient_clipnorm_fn(grads_and_vars):
    # [FUNCTION DOC]
    # HELPER: the transformation returned by the global-norm clipping factory.
    # KNOWN ISSUE: the error message misspells the strategy name as "CenteralStorageStrategy".
    # Input: list of (gradient, variable) pairs. Raises a value error if the current distribution
    # strategy is central storage. Splits the pairs into gradients and variables, clips the
    # gradients by their global norm, and returns a new list of pairs.

    if isinstance(distribute_lib.get_strategy(),
                  (central_storage_strategy.CentralStorageStrategy,
                   central_storage_strategy.CentralStorageStrategyV1)):
      raise ValueError(
          "`global_clipnorm` is not supported with `CenteralStorageStrategy`")

    grads, variables = zip(*grads_and_vars)
    clipped_grads, _ = clip_ops.clip_by_global_norm(grads, clipnorm)
    clipped_grads_and_vars = list(zip(clipped_grads, variables))
    return clipped_grads_and_vars

  return gradient_clipnorm_fn


def make_gradient_clipvalue_fn(clipvalue):
  # [FUNCTION DOC]
  # HELPER: factory for the clip-by-value gradient transformation used by optimizers.
  # Input: clipvalue, a number or nothing. If nothing, returns an identity function on the pairs.
  # Otherwise returns a function clipping each gradient elementwise to the range from minus
  # clipvalue to clipvalue.
  """Creates a gradient transformation function for clipping by value."""
  if clipvalue is None:
    return lambda grads_and_vars: grads_and_vars

  def gradient_clipvalue_fn(grads_and_vars):
    # [FUNCTION DOC]
    # HELPER: the transformation returned by the clip-by-value factory.
    # KNOWN ISSUE: the error message misspells the strategy name as "CenteralStorageStrategy".
    # Input: list of (gradient, variable) pairs. Raises a value error if the current distribution
    # strategy is central storage. Returns a new list with each gradient clipped by value.

    if isinstance(distribute_lib.get_strategy(),
                  (central_storage_strategy.CentralStorageStrategy,
                   central_storage_strategy.CentralStorageStrategyV1)):
      raise ValueError(
          "`clipvalue` is not supported with `CenteralStorageStrategy`")

    clipped_grads_and_vars = [(clip_ops.clip_by_value(g, -clipvalue,
                                                      clipvalue), v)
                              for g, v in grads_and_vars]
    return clipped_grads_and_vars

  return gradient_clipvalue_fn


def _all_reduce_sum_fn(distribution, grads_and_vars):
  # [FUNCTION DOC]
  # HELPER: merge-call callback for the all-reduce summation helper.
  # Inputs: the distribution strategy and a list of (gradient, variable) pairs. Returns the
  # result of a batched sum-reduction of the gradients to their destinations via the strategy.
  return distribution.extended.batch_reduce_to(ds_reduce_util.ReduceOp.SUM,
                                               grads_and_vars)


def strategy_supports_no_merge_call():
  # [FUNCTION DOC]
  # HELPER: tells the optimizer whether it can reduce gradients directly in replica context.
  # Takes no input. Returns true if there is no active distribution strategy, otherwise true when
  # the strategy's extended object does not require merge calls; false otherwise.
  """Returns if the current Strategy can operate in pure replica context."""
  if not distribute_lib.has_strategy():
    return True
  strategy = distribute_lib.get_strategy()
  return not strategy.extended._use_merge_call()  # pylint: disable=protected-access
