# [MODULE DOC]
# Defines CrossShardOptimizer (exported for the v1 API), an optimizer wrapper for TPU training
# that sums gradients across replicas (shards) before delegating to the wrapped optimizer, and
# scales the loss beforehand when averaging is requested. Depends on the TPU shard context and the
# TPU cross-replica-sum operation. No import-time side effects other than the export.
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
# =============================================================================

"""Optimizer that implements cross-shard gradient reduction for TPU."""


from tensorflow.python.framework import ops
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tpu_function
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.training import optimizer
from tensorflow.python.util.tf_export import tf_export


@tf_export(v1=["tpu.CrossShardOptimizer"])
class CrossShardOptimizer(optimizer.Optimizer):
  # [CLASS DOC]
  # Wraps an existing training optimizer so that gradients are summed over TPU replicas (optionally
  # within subgroups of replicas). In mean mode the loss is scaled down by the shard (or subgroup)
  # count in gradient computation so the sum acts as an average. Holds the wrapped optimizer, the
  # reduction mode and the optional group assignment. Slot access and variable listing are
  # forwarded to the wrapped optimizer. Only works with legacy training optimizers.
  """An optimizer that averages gradients across TPU shards."""

  def __init__(self,
               opt,
               reduction=losses.Reduction.MEAN,
               name="CrossShardOptimizer",
               group_assignment=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring lists only a value error, but a type error is also raised when
    # the wrapped optimizer is not a legacy training optimizer (for example a Keras optimizer).
    # Constructor. Inputs: opt, the optimizer to wrap; reduction, sum or mean (default mean);
    # name, op-name prefix (default "CrossShardOptimizer"); group_assignment, optional list of
    # lists of replica ids defining subgroups. Raises a value error for an unsupported reduction
    # and a type error for a non-legacy optimizer. Initialises the base optimizer with locking
    # disabled and stores the arguments.
    """Construct a new cross-shard optimizer.

    Args:
      opt: An existing `Optimizer` to encapsulate.
      reduction: The reduction to apply to the shard losses.
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to "CrossShardOptimizer".
      group_assignment: Optional 2d int32 lists with shape
        [num_groups, num_replicas_per_group] which describles how to apply
        optimizer to subgroups.

    Raises:
      ValueError: If reduction is not a valid cross-shard reduction.
    """
    accepted_reductions = (losses.Reduction.SUM, losses.Reduction.MEAN)
    if reduction not in accepted_reductions:
      raise ValueError(
          f"Argument `reduction` should be one of {accepted_reductions}. "
          f"Received: {reduction}")
    if not isinstance(opt, optimizer.Optimizer):
      raise TypeError(
          "CrossShardOptimizer only works with tf.training.Optimizer and not "
          f"Keras Optimizer. Received: {opt}. "
          "If you are using TPUStrategy, "
          "Keras Optimizer will sum gradients across replicas."
          "If you want to average your gradients, rescale your loss with: "
          "`loss /= global_batch_size`")

    super(CrossShardOptimizer, self).__init__(False, name)
    self._opt = opt
    self._reduction = reduction
    self._group_assignment = group_assignment

  def _verify_and_get_subgroup_size(self, group_assignment, num_shards):
    # [FUNCTION DOC]
    # HELPER: validates the replica group assignment for compute_gradients.
    # Inputs: group_assignment, list of lists of replica ids (or empty/null); num_shards, total
    # replica count. Returns null when no assignment is given, otherwise the common group size.
    # Raises a value error if it is not a list of lists, if the ids are not exactly the numbers
    # from zero to num_shards minus one, or if the groups differ in size.
    """Verify group_assignment and get the subgroup size".

    Args:
      group_assignment: list of group ids for applying the optimizer
        to subgroups.
      num_shards: The number of TPU shards.

    Returns:
      The size of one subgroup in group_assignment.

    Raises:
      ValueError: If group_assignment is invalid.
    """
    if not group_assignment:
      return None
    if not (isinstance(group_assignment, list) and
            all(isinstance(i, list) for i in group_assignment)):
      raise ValueError(
          f"Argument `group_assignment` must be a list of lists. "
          f"Received: {group_assignment}")

    replica_ids = set()
    for g in group_assignment:
      for i in g:
        replica_ids.add(i)

    if set(range(num_shards)) != replica_ids:
      raise ValueError(
          f"Argument `group_assignment` must be a permutation of "
          f"range({num_shards}). Received: {group_assignment}")

    subgroup_size_list = [len(group) for group in group_assignment]
    if all(subgroup_size_list[0] == size for size in subgroup_size_list):
      return subgroup_size_list[0]
    else:
      raise ValueError("The size of each subgroup in `group_assignment` must "
                       f"be equal. Received: {group_assignment}")

  def compute_gradients(self, loss, var_list=None, **kwargs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says a value error is raised when called outside a TPU shard
    # context, but the code only logs a warning and assumes a single shard.
    # Computes gradients with the wrapped optimizer. Inputs: loss, tensor to minimise; var_list,
    # optional variables (default: trainable variables); extra keyword arguments are forwarded.
    # Reads the shard count from the TPU context (1 if unset, with a warning). In mean mode with
    # more than one shard the loss is multiplied by one over the subgroup size (if a group
    # assignment is set) or one over the shard count. Returns the list of gradient/variable pairs.
    # Raises a value error if the group assignment is invalid.
    """Compute gradients of "loss" for the variables in "var_list".

    This simply wraps `compute_gradients()` from the real optimizer. The
    gradients will be aggregated in `apply_gradients()` so that user can
    modify the gradients like clipping with per replica global norm if needed.
    The global norm with aggregated gradients can be bad as one replica's huge
    gradients can hurt the gradients from other replicas.

    When the CrossShardOptimizer is constructed with
    `reduction == losses.Reduction.MEAN` (default), this function scales the
    loss by `1.0 / num_shards` before computing the gradients. Assuming the
    optimizer uses the default implementation of `compute_gradients()`, the
    gradients of the scaled loss are scaled by `1.0 / num_shards` compared to
    the gradients of the original loss. This scaling factor is important because
    `apply_gradients()` sums gradients across shards, rather than averaging
    them. However, the scaling factor must be taken into account when clipping
    the norm of the gradients or performing other postprocessing.

    Args:
      loss: A Tensor containing the value to minimize.
      var_list: Optional list or tuple of `tf.Variable` to update to minimize
        `loss`.  Defaults to the list of variables collected in the graph
        under the key `GraphKey.TRAINABLE_VARIABLES`.
      **kwargs: Keyword arguments for compute_gradients().

    Returns:
      A list of (gradient, variable) pairs.

    Raises:
      ValueError: If not within a tpu_shard_context or group_assignment is
        invalid.
    """
    num_shards = tpu_function.get_tpu_context().number_of_shards
    if num_shards is None:
      logging.warning(
          "CrossShardOptimizer should be used within a tpu_shard_context, but "
          "got unset number_of_shards. Assuming 1.")
      num_shards = 1

    subgroup_size = self._verify_and_get_subgroup_size(self._group_assignment,
                                                       num_shards)

    if num_shards > 1 and self._reduction == losses.Reduction.MEAN:
      if self._group_assignment:
        scale = 1.0 / subgroup_size
      else:
        scale = 1.0 / num_shards
      loss *= scale

    return self._opt.compute_gradients(loss, var_list=var_list, **kwargs)

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    # [FUNCTION DOC]
    # Sums each non-null gradient across replicas (colocated with the gradient, honouring the
    # group assignment) and passes the summed pairs to the wrapped optimizer. Null gradients pass
    # through unchanged. Inputs: grads_and_vars, list of gradient/variable pairs; global_step,
    # optional step variable to increment; name, optional name for the returned op. Returns the
    # operation that applies the update. Malformed pairs surface as unpacking errors.
    """Apply gradients to variables.

    Calls tpu_ops.cross_replica_sum() to sum gradient contributions across
    replicas, and then applies the real optimizer.

    Args:
      grads_and_vars: List of (gradient, variable) pairs as returned by
        compute_gradients().
      global_step: Optional Variable to increment by one after the
        variables have been updated.
      name: Optional name for the returned operation.  Default to the
        name passed to the Optimizer constructor.

    Returns:
      An `Operation` that applies the gradients. If `global_step` was not None,
      that operation also increments `global_step`.

    Raises:
      ValueError: If the grads_and_vars is malformed.
    """
    summed_grads_and_vars = []
    for (grad, var) in grads_and_vars:
      if grad is None:
        summed_grads_and_vars.append((grad, var))
      else:
        with ops.colocate_with(grad):
          summed_grads_and_vars.append((tpu_ops.cross_replica_sum(
              grad, self._group_assignment), var))
    return self._opt.apply_gradients(summed_grads_and_vars, global_step, name)

  def get_slot(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Forwards all arguments to the wrapped optimizer's slot lookup and returns the slot variable,
    # or null if it was never created.
    """Return a slot named "name" created for "var" by the Optimizer.

    This simply wraps the get_slot() from the actual optimizer.

    Args:
      *args: Arguments for get_slot().
      **kwargs: Keyword arguments for get_slot().

    Returns:
      The `Variable` for the slot if it was created, `None` otherwise.
    """
    return self._opt.get_slot(*args, **kwargs)

  def get_slot_names(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Forwards all arguments to the wrapped optimizer and returns its list of slot name strings.
    """Return a list of the names of slots created by the `Optimizer`.

    This simply wraps the get_slot_names() from the actual optimizer.

    Args:
      *args: Arguments for get_slot().
      **kwargs: Keyword arguments for get_slot().

    Returns:
      A list of strings.
    """
    return self._opt.get_slot_names(*args, **kwargs)

  def variables(self):
    # [FUNCTION DOC]
    # Returns the variables of the wrapped optimizer (this wrapper holds none of its own).
    """Forwarding the variables from the underlying optimizer."""
    return self._opt.variables()
