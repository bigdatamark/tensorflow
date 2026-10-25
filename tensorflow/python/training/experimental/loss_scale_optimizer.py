# [MODULE DOC]
# Defines the legacy (TF1 style) optimizer wrapper that applies loss scaling to another optimizer.
# The wrapper multiplies the loss by a loss scale before gradient computation and divides the
# resulting gradients by the same value, which helps avoid underflow in float16 training. The
# loss scale is described by a loss scale object from the sibling loss scale module (fixed,
# dynamic or a plain number converted to a fixed one).
# Exposes one public class, registered through the export decorator under the version 1 endpoints
# for mixed precision and for the experimental training namespace, with a deprecation notice on
# the experimental endpoint. No flags or global state are created at import time apart from the
# export and deprecation registrations.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Contains LossScale classes."""
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import smart_cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training.experimental import loss_scale as loss_scale_module
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


@deprecation.deprecated_endpoints(
    'train.experimental.MixedPrecisionLossScaleOptimizer')
@tf_export(v1=['mixed_precision.MixedPrecisionLossScaleOptimizer',
               'train.experimental.MixedPrecisionLossScaleOptimizer'])
class MixedPrecisionLossScaleOptimizer(optimizer.Optimizer):
  # [CLASS DOC]
  # Optimizer decorator that wraps another legacy optimizer and applies loss scaling to it.
  # State: the wrapped optimizer and a loss scale object; both are registered as tracked
  # dependencies named base_optimizer and loss_scale so they are saved with checkpoints.
  # Gradient computation scales the loss up and scales the gradients back down. With a dynamic
  # loss scale, applying gradients is conditional: the loss scale is updated and gradients are
  # applied only if all of them are finite. With a fixed loss scale the call is forwarded
  # directly. Created by passing an optimizer and a loss scale specification; the class is
  # exported as a version 1 API only.
  """An optimizer that applies loss scaling.

  Loss scaling is a process that multiplies the loss by a multiplier called the
  loss scale, and divides each gradient by the same multiplier. The pseudocode
  for this process is:

  ```
  loss = ...
  loss *= loss_scale
  grads = gradients(loss, vars)
  grads /= loss_scale
  ```

  Mathematically, loss scaling has no effect, but can help avoid numerical
  underflow in intermediate gradients when float16 tensors are used for mixed
  precision training. By multiplying the loss, each intermediate gradient will
  have the same multiplier applied.

  The loss scale can either be a fixed constant, chosen by the user, or be
  dynamically determined. Dynamically determining the loss scale is convenient
  as a loss scale does not have to be explicitly chosen. However it reduces
  performance.

  This optimizer wraps another optimizer and applies loss scaling to it via a
  `LossScale`. Loss scaling is applied whenever gradients are
  computed, such as through `minimize()`.
  """

  def __init__(self, opt, loss_scale):
    # [FUNCTION DOC]
    # Constructor. Inputs: opt, the optimizer to wrap (must be an instance of the legacy
    # optimizer base class); loss_scale, anything the loss scale module's getter accepts (a
    # number for a fixed scale, the string "dynamic", or a loss scale object).
    # Copies the locking flag and name from the wrapped optimizer into the base constructor,
    # converts loss_scale to a loss scale object and registers both as tracked dependencies.
    # Raises a value error if opt is not an optimizer or if the converted loss scale is null.
    # Returns nothing.
    if not isinstance(opt, optimizer.Optimizer):
      raise ValueError('"opt" must be an instance of Optimizer, but got: %s' %
                       type(opt))
    self._optimizer = opt

    use_locking = opt._use_locking  # pylint: disable=protected-access
    name = opt.get_name()
    super(MixedPrecisionLossScaleOptimizer, self).__init__(use_locking, name)

    self._loss_scale = loss_scale_module.get(loss_scale)
    if self._loss_scale is None:
      raise ValueError('loss_scale cannot be None')
    self._track_trackable(self._optimizer, 'base_optimizer')
    self._track_trackable(self._loss_scale, 'loss_scale')

  def _doing_dynamic_loss_scaling(self):
    # [FUNCTION DOC]
    # HELPER: decides which apply path apply_gradients takes.
    # Returns true when the loss scale object is a dynamic loss scale, false otherwise.
    # No side effects.
    """Check if `_loss_scale` dynamically manages the loss scale."""
    return isinstance(self._loss_scale, loss_scale_module.DynamicLossScale)

  def compute_gradients(self,
                        loss,
                        var_list=None,
                        gate_gradients=optimizer.Optimizer.GATE_OP,
                        aggregation_method=None,
                        colocate_gradients_with_ops=False,
                        grad_loss=None):
    # [FUNCTION DOC]
    # Computes gradients with loss scaling. The loss (a tensor or a no-argument callable
    # returning one) is multiplied by the current loss scale, the wrapped optimizer computes
    # gradients of the scaled loss for the given variable list, and each gradient is multiplied
    # by the reciprocal of the loss scale. Missing gradients stay null; sparse gradients keep
    # their sparse form.
    # Inputs: loss; var_list (optional variables); gate_gradients (gating mode, default gate per
    # op); aggregation_method; colocate_gradients_with_ops (default false); grad_loss (optional
    # initial gradient). All but loss are passed unchanged to the wrapped optimizer.
    # Returns a list of pairs of unscaled gradient and variable. Reads the loss scale value, so
    # it may create graph operations; exceptions come from the wrapped optimizer.
    """Compute gradients of `loss` for the variables in `var_list`.

    This adjusts the dynamic range of the gradient evaluation by scaling up
    the `loss` value. The gradient values are then scaled back down by the
    reciprocal of the loss scale. This is useful in reduced precision training
    where small gradient values would otherwise underflow the representable
    range.

    Args:
      loss: A Tensor containing the value to minimize or a callable taking no
        arguments which returns the value to minimize. When eager execution is
        enabled it must be a callable.
      var_list: Optional list or tuple of `tf.Variable` to update to minimize
        `loss`.  Defaults to the list of variables collected in the graph under
        the key `GraphKeys.TRAINABLE_VARIABLES`.
      gate_gradients: How to gate the computation of gradients.  Can be
        `GATE_NONE`, `GATE_OP`, or `GATE_GRAPH`.
      aggregation_method: Specifies the method used to combine gradient terms.
        Valid values are defined in the class `AggregationMethod`.
      colocate_gradients_with_ops: If True, try colocating gradients with the
        corresponding op.
      grad_loss: Optional. A `Tensor` holding the gradient computed for `loss`.

    Returns:
      A list of (gradient, variable) pairs. Variable is always present, but
      gradient can be `None`.
    """
    loss = self._scale_loss(loss)
    grads_and_vars = self._optimizer.compute_gradients(
        loss=loss,
        var_list=var_list,
        gate_gradients=gate_gradients,
        aggregation_method=aggregation_method,
        colocate_gradients_with_ops=colocate_gradients_with_ops,
        grad_loss=grad_loss)

    grads = [g for g, _ in grads_and_vars]
    variables = [v for _, v in grads_and_vars]
    unscaled_grads = self._unscale_grads(grads)
    return list(zip(unscaled_grads, variables))

  def _scale_loss(self, loss):
    # [FUNCTION DOC]
    # HELPER: loss scaling step of compute_gradients.
    # Input: loss, either a tensor or a no-argument callable returning a tensor. Reads the
    # current loss scale once when called. For a tensor, returns it multiplied by the loss scale
    # cast to the loss dtype. For a callable, returns a new callable that evaluates the original
    # and applies the same multiplication to its result.
    loss_scale = self._loss_scale()
    if callable(loss):
      def new_loss():
        # [FUNCTION DOC]
        # HELPER: wrapper callable returned by the loss scaling step for callable losses.
        # Takes no inputs; calls the original loss callable and returns its value multiplied by
        # the captured loss scale cast to the value dtype.
        loss_val = loss()
        return loss_val * math_ops.cast(loss_scale, loss_val.dtype)
      return new_loss
    else:
      return loss * math_ops.cast(loss_scale, loss.dtype)

  def _unscale_grads(self, grads):
    # [FUNCTION DOC]
    # HELPER: gradient unscaling step of compute_gradients.
    # Input: grads, a list of gradients (dense, sparse or null). Reads the current loss scale,
    # computes its reciprocal and returns a new list in which each non-null gradient is
    # multiplied by the reciprocal; null entries stay null.
    loss_scale = self._loss_scale()
    loss_scale_reciprocal = 1 / loss_scale
    return [
        None if g is None else self._scale_grad(g, loss_scale_reciprocal)
        for g in grads
    ]

  def _scale_grad(self, grad, loss_scale_reciprocal):
    # [FUNCTION DOC]
    # HELPER: multiplies a single gradient for the unscaling step.
    # Inputs: grad, a dense tensor or sparse indexed slices; loss_scale_reciprocal, the factor.
    # For sparse input only the values are multiplied and a new sparse object with the same
    # indices and dense shape is returned; for dense input the product tensor is returned.
    if isinstance(grad, indexed_slices.IndexedSlices):
      grad_vals = grad.values * loss_scale_reciprocal
      return indexed_slices.IndexedSlices(grad_vals, grad.indices,
                                          grad.dense_shape)
    return grad * loss_scale_reciprocal

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it raises a runtime error, but the code raises a value
    # error (when called in a cross replica context). The docstring also claims gradients are
    # conditionally applied always; this only happens for a dynamic loss scale, a fixed loss
    # scale forwards straight to the wrapped optimizer.
    # Applies gradients to variables. Inputs: grads_and_vars, iterable of pairs of gradient and
    # variable (as from compute_gradients); global_step, optional variable to increment;
    # name, optional operation name.
    # Fixed loss scale: delegates to the wrapped optimizer and returns its result. Dynamic loss
    # scale: converts the pairs to a tuple and uses a merge call into the cross replica
    # distributed apply helper, returning its grouped operation.
    # Raises a value error when invoked in a cross replica context.
    """Apply gradients to variables.

    This is the second part of `minimize()`. It returns an `Operation` that
    conditionally applies gradients if all gradient values are finite.
    Otherwise no update is performed (nor is `global_step` incremented).

    Args:
      grads_and_vars: List of (gradient, variable) pairs as returned by
        `compute_gradients()`.
      global_step: Optional `Variable` to increment by one after the variables
        have been updated.
      name: Optional name for the returned operation.  Default to the name
        passed to the `Optimizer` constructor.

    Returns:
      An `Operation` that conditionally applies the specified gradients. If
      `global_step` was not None, that operation also increments `global_step`.

    Raises:
      RuntimeError: If you should use `_distributed_apply()` instead.
    """
    if distribute_lib.in_cross_replica_context():
      raise ValueError('apply_gradients() must be called in a replica context.')

    if not self._doing_dynamic_loss_scaling():
      return self._optimizer.apply_gradients(grads_and_vars, global_step, name)

    replica_context = distribute_lib.get_replica_context()
    grads_and_vars = tuple(grads_and_vars)

    # TODO(nluehr) cleanup GraphKeys.TRAIN_OP
    return replica_context.merge_call(
        self._distributed_apply, args=(grads_and_vars, global_step, name))

  def _distributed_apply(self,
                         distribution,
                         grads_and_vars,
                         global_step=None,
                         name=None):
    # [FUNCTION DOC]
    # HELPER: cross replica implementation behind apply_gradients for dynamic loss scaling.
    # Inputs: distribution, the distribution strategy; grads_and_vars, pairs of aggregated
    # gradient and variable; global_step, optional variable to increment; name, optional
    # operation name (defaults to the optimizer name).
    # Asks the loss scale object to update itself from the gradients, obtaining an update
    # operation and a flag saying whether all gradients are finite. Gradients are applied only
    # when the flag is true (via a smart conditional), otherwise a no-op runs.
    # Returns one grouped operation containing the conditional apply and the loss scale update.
    # Side effects: updates the loss scale state.
    """A version of `apply_gradients` for cross replica context.

    When users are in a cross replica strategy, they must call this rather than
    `apply_gradients()`.

    Args:
      distribution: a `DistributionStrategy` object.
      grads_and_vars: List of (gradient, variable) pairs as returned by
        `compute_gradients()` and then aggregated across replicas.
      global_step: Optional (mirrored) `Variable` to increment by one after the
        variables have been updated.
      name: Optional name for the returned operation. Default to the name passed
        to the `Optimizer` constructor.

    Returns:
      An `Operation` that applies the specified gradients across all
      replicas. If `global_step` was not None, that operation also
      increments `global_step`
    """
    name = name if name is not None else self.get_name()
    grads = [g for g, _ in grads_and_vars]
    loss_scale_update_op, should_apply_grads = (self._loss_scale.update(grads))

    def apply_fn():
      # [FUNCTION DOC]
      # HELPER: branch taken when gradients are finite.
      # Takes no inputs; applies the gradients through the unconditional apply helper using the
      # enclosing distribution, pairs, global step and the name with a wrapped suffix, and
      # returns that operation.
      return self._apply_gradients(distribution, grads_and_vars, global_step,
                                   name + '-wrapped')

    maybe_apply_op = smart_cond.smart_cond(should_apply_grads, apply_fn,
                                           control_flow_ops.no_op)
    return control_flow_ops.group(
        maybe_apply_op, loss_scale_update_op, name=name)

  def _apply_gradients(self, distribution, grads_and_vars, global_step, name):
    # [FUNCTION DOC]
    # HELPER: unconditional apply used by the distributed apply helper.
    # Inputs: distribution strategy, pairs of gradient and variable, global step, name.
    # Calls the wrapped optimizer's apply method once per replica and returns the grouped
    # update operations. Side effects: updates variables.
    """Unconditionally apply gradients in cross replica context."""
    update_ops = distribution.extended.call_for_each_replica(
        self._optimizer.apply_gradients,
        args=(grads_and_vars, global_step, name))
    return distribution.group(update_ops)

  def _apply_sparse(self, grad, var):
    # [FUNCTION DOC]
    # HELPER: required optimizer hook that is intentionally unusable in this wrapper.
    # Inputs grad and var are ignored. Always raises a runtime error, because updates are
    # delegated to the wrapped optimizer.
    """This function should never be called."""
    raise RuntimeError('This function should never be called')

  def _apply_dense(self, grad, var):
    # [FUNCTION DOC]
    # HELPER: required optimizer hook that is intentionally unusable in this wrapper.
    # Inputs grad and var are ignored. Always raises a runtime error, because updates are
    # delegated to the wrapped optimizer.
    """This function should never be called."""
    raise RuntimeError('This function should never be called')

  def _resource_apply_sparse(self, grad, handle, indices):
    # [FUNCTION DOC]
    # HELPER: required optimizer hook that is intentionally unusable in this wrapper.
    # Inputs grad, handle and indices are ignored. Always raises a runtime error, because
    # updates are delegated to the wrapped optimizer.
    """This function should never be called."""
    raise RuntimeError('This function should never be called')

  def _resource_apply_dense(self, grad, handle):
    # [FUNCTION DOC]
    # HELPER: required optimizer hook that is intentionally unusable in this wrapper.
    # Inputs grad and handle are ignored. Always raises a runtime error, because updates are
    # delegated to the wrapped optimizer.
    """This function should never be called."""
    raise RuntimeError('This function should never be called')

  def variables(self):
    # [FUNCTION DOC]
    # Returns a list of the variables of this optimizer: the wrapped optimizer's variables
    # followed by the loss scale's weight variables (read from the loss scale object's private
    # weights mapping, so it relies on internals of the loss scale class).
    # Takes no inputs and has no side effects.
    """Returns the variables of the Optimizer."""
    return (self._optimizer.variables() +
            list(self._loss_scale._weights.values()))  # pylint: disable=protected-access
