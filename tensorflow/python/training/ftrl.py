# [MODULE DOC]
# Defines the FTRL-proximal optimizer (v1 graph-mode API, exported as the train Ftrl optimizer
# under the v1 namespace). It exposes one class that subclasses the base training Optimizer and
# applies updates through the generated FTRL training ops (dense, sparse, resource variants,
# with or without L2 shrinkage). The only import-time side effect is the export registration
# done by the export decorator. Not a helper module.
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
"""Ftrl-proximal for TensorFlow."""
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_training_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer
from tensorflow.python.util.tf_export import tf_export


@tf_export(v1=["train.FtrlOptimizer"])
class FtrlOptimizer(optimizer.Optimizer):
  # [CLASS DOC]
  # Optimizer implementing Follow-The-Regularized-Leader with proximal updates for large sparse
  # models, supporting L1, L2 and an optional L2 shrinkage penalty plus a beta term. Instances
  # are created with a learning rate and regularization hyper-parameters, then used through the
  # base optimizer minimize / apply-gradients flow. It keeps hyper-parameters, their tensor
  # versions (built in the prepare step) and per-variable "accum" and "linear" slots. Overrides
  # slot creation, preparation and the dense, sparse and resource apply methods. Class-level
  # side effect: exported in the v1 train namespace.
  """Optimizer that implements the FTRL algorithm.

  This version has support for both online L2 (McMahan et al., 2013) and
  shrinkage-type L2, which is the addition of an L2 penalty
  to the loss function.

  References:
    Ad-click prediction:
      [McMahan et al., 2013](https://dl.acm.org/citation.cfm?id=2488200)
      ([pdf](https://dl.acm.org/ft_gateway.cfm?id=2488200&ftid=1388399&dwn=1&CFID=32233078&CFTOKEN=d60fe57a294c056a-CB75C374-F915-E7A6-1573FBBC7BF7D526))
  """

  def __init__(self,
               learning_rate,
               learning_rate_power=-0.5,
               initial_accumulator_value=0.1,
               l1_regularization_strength=0.0,
               l2_regularization_strength=0.0,
               use_locking=False,
               name="Ftrl",
               accum_name=None,
               linear_name=None,
               l2_shrinkage_regularization_strength=0.0,
               beta=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says the learning rate power and regularization strengths
    # "must be" within ranges and the docstring Raises section is generic; the check for
    # learning_rate_power rejects only positive values, and no check at all is made on beta or
    # learning_rate.
    # Constructor. Inputs: learning_rate (float or scalar tensor), learning_rate_power (default
    # -0.5, must be zero or negative), initial_accumulator_value (default 0.1, must be
    # non-negative), l1_regularization_strength and l2_regularization_strength (default 0, must
    # be non-negative), use_locking (default false), name (default "Ftrl"), accum_name and
    # linear_name (optional slot name suffixes, falling back to the optimizer name), the
    # l2_shrinkage_regularization_strength (default 0, non-negative) and beta (optional, treated
    # as zero when null). Calls the base constructor, validates the arguments and stores them;
    # tensor versions are left null until the prepare step. Raises a value error when any
    # validated argument is out of range. Returns nothing.
    r"""Construct a new FTRL optimizer.

    Args:
      learning_rate: A float value or a constant float `Tensor`.
      learning_rate_power: A float value, must be less or equal to zero.
        Controls how the learning rate decreases during training. Use zero for
        a fixed learning rate. See section 3.1 in (McMahan et al., 2013).
      initial_accumulator_value: The starting value for accumulators.
        Only zero or positive values are allowed.
      l1_regularization_strength: A float value, must be greater than or
        equal to zero.
      l2_regularization_strength: A float value, must be greater than or
        equal to zero.
      use_locking: If `True` use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients.  Defaults to "Ftrl".
      accum_name: The suffix for the variable that keeps the gradient squared
        accumulator.  If not present, defaults to name.
      linear_name: The suffix for the variable that keeps the linear gradient
        accumulator.  If not present, defaults to name + "_1".
      l2_shrinkage_regularization_strength: A float value, must be greater than
        or equal to zero. This differs from L2 above in that the L2 above is a
        stabilization penalty, whereas this L2 shrinkage is a magnitude penalty.
        The FTRL formulation can be written as:
        w_{t+1} = argmin_w(\hat{g}_{1:t}w + L1*||w||_1 + L2*||w||_2^2), where
        \hat{g} = g + (2*L2_shrinkage*w), and g is the gradient of the loss
        function w.r.t. the weights w.
        Specifically, in the absence of L1 regularization, it is equivalent to
        the following update rule:
        w_{t+1} = w_t - lr_t / (beta + 2*L2*lr_t) * g_t -
                  2*L2_shrinkage*lr_t / (beta + 2*L2*lr_t) * w_t
        where lr_t is the learning rate at t.
        When input is sparse shrinkage will only happen on the active weights.
      beta: A float value; corresponds to the beta parameter in the paper.

    Raises:
      ValueError: If one of the arguments is invalid.

    References:
      Ad-click prediction:
        [McMahan et al., 2013](https://dl.acm.org/citation.cfm?id=2488200)
        ([pdf](https://dl.acm.org/ft_gateway.cfm?id=2488200&ftid=1388399&dwn=1&CFID=32233078&CFTOKEN=d60fe57a294c056a-CB75C374-F915-E7A6-1573FBBC7BF7D526))
    """
    super(FtrlOptimizer, self).__init__(use_locking, name)

    if initial_accumulator_value < 0.0:
      raise ValueError(
          "initial_accumulator_value %f needs to be positive or zero" %
          initial_accumulator_value)
    if learning_rate_power > 0.0:
      raise ValueError("learning_rate_power %f needs to be negative or zero" %
                       learning_rate_power)
    if l1_regularization_strength < 0.0:
      raise ValueError(
          "l1_regularization_strength %f needs to be positive or zero" %
          l1_regularization_strength)
    if l2_regularization_strength < 0.0:
      raise ValueError(
          "l2_regularization_strength %f needs to be positive or zero" %
          l2_regularization_strength)
    if l2_shrinkage_regularization_strength < 0.0:
      raise ValueError(
          "l2_shrinkage_regularization_strength %f needs to be positive"
          " or zero" % l2_shrinkage_regularization_strength)

    self._learning_rate = learning_rate
    self._learning_rate_power = learning_rate_power
    self._initial_accumulator_value = initial_accumulator_value
    self._l1_regularization_strength = l1_regularization_strength
    self._l2_regularization_strength = l2_regularization_strength
    self._beta = (0.0 if beta is None else beta)
    self._l2_shrinkage_regularization_strength = (
        l2_shrinkage_regularization_strength)
    self._learning_rate_tensor = None
    self._learning_rate_power_tensor = None
    self._l1_regularization_strength_tensor = None
    self._adjusted_l2_regularization_strength_tensor = None
    self._l2_shrinkage_regularization_strength_tensor = None
    self._accum_name = accum_name
    self._linear_name = linear_name

  def _create_slots(self, var_list):
    # [FUNCTION DOC]
    # Override of the base slot creation. For every variable in var_list (list of variables) it
    # creates an "accum" slot initialized to the initial accumulator value and a zero-filled
    # "linear" slot, named using the accum / linear name suffix or the optimizer name. Returns
    # nothing; side effect is creating slot variables.
    # Create the "accum" and "linear" slots.
    def _accum_initializer(shape, dtype=dtypes.float32, partition_info=None):
      # [FUNCTION DOC]
      # HELPER: initializer for the accumulator slot. Takes a shape, a dtype (default 32-bit
      # float) and an ignored partition info; returns a tensor of that shape filled with the
      # initial accumulator value.
      del partition_info
      return array_ops.ones(
          shape=shape, dtype=dtype) * self._initial_accumulator_value
    for v in var_list:
      self._get_or_make_slot_with_initializer(
          v, _accum_initializer, v.shape, v.dtype, "accum",
          self._accum_name or self._name)
      self._zeros_slot(v, "linear", self._linear_name or self._name)

  def _prepare(self):
    # [FUNCTION DOC]
    # Override of the base prepare step, called before applying gradients. Converts the
    # learning rate, L1 strength, L2 shrinkage strength, learning rate power and beta into
    # tensors, and builds an adjusted L2 strength equal to the L2 strength plus beta divided by
    # twice the learning rate (the learning rate is floored at a tiny positive value to avoid
    # division by zero) so the underlying ops need no beta input. Returns nothing; stores the
    # tensors on the instance.
    self._learning_rate_tensor = ops.convert_to_tensor(
        self._learning_rate, name="learning_rate")
    self._l1_regularization_strength_tensor = ops.convert_to_tensor(
        self._l1_regularization_strength, name="l1_regularization_strength")
    # L2 regularization strength with beta added in so that the underlying
    # TensorFlow ops do not need to include that parameter.
    self._adjusted_l2_regularization_strength_tensor = ops.convert_to_tensor(
        self._l2_regularization_strength + self._beta /
        (2. * math_ops.maximum(self._learning_rate, 1e-36)),
        name="adjusted_l2_regularization_strength")
    assert self._adjusted_l2_regularization_strength_tensor is not None
    self._beta_tensor = ops.convert_to_tensor(self._beta, name="beta")
    self._l2_shrinkage_regularization_strength_tensor = ops.convert_to_tensor(
        self._l2_shrinkage_regularization_strength,
        name="l2_shrinkage_regularization_strength")
    self._learning_rate_power_tensor = ops.convert_to_tensor(
        self._learning_rate_power, name="learning_rate_power")

  def _apply_dense(self, grad, var):
    # [FUNCTION DOC]
    # Applies a dense FTRL update to a reference variable. Inputs: grad (dense gradient tensor)
    # and var (variable). Fetches the accum and linear slots and runs the FTRL training op, or
    # its L2-shrinkage version when the shrinkage strength is positive, casting all
    # hyper-parameter tensors to the variable dtype. Returns the update operation. Uses
    # locking as configured.
    accum = self.get_slot(var, "accum")
    linear = self.get_slot(var, "linear")
    if self._l2_shrinkage_regularization_strength <= 0.0:
      return gen_training_ops.apply_ftrl(
          var,
          accum,
          linear,
          grad,
          math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
          math_ops.cast(self._l1_regularization_strength_tensor,
                        var.dtype.base_dtype),
          math_ops.cast(self._adjusted_l2_regularization_strength_tensor,
                        var.dtype.base_dtype),
          math_ops.cast(self._learning_rate_power_tensor, var.dtype.base_dtype),
          use_locking=self._use_locking)
    else:
      return gen_training_ops.apply_ftrl_v2(
          var,
          accum,
          linear,
          grad,
          math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
          math_ops.cast(self._l1_regularization_strength_tensor,
                        var.dtype.base_dtype),
          math_ops.cast(self._adjusted_l2_regularization_strength_tensor,
                        var.dtype.base_dtype),
          math_ops.cast(self._l2_shrinkage_regularization_strength_tensor,
                        var.dtype.base_dtype),
          math_ops.cast(self._learning_rate_power_tensor, var.dtype.base_dtype),
          use_locking=self._use_locking)

  def _resource_apply_dense(self, grad, var):
    # [FUNCTION DOC]
    # Applies a dense FTRL update to a resource variable. Inputs: grad (dense gradient) and var
    # (resource variable, whose handle and slot handles are passed). Chooses the plain or
    # L2-shrinkage resource op depending on the shrinkage strength, casting hyper-parameters to
    # the variable dtype. Returns the update operation.
    accum = self.get_slot(var, "accum")
    linear = self.get_slot(var, "linear")
    if self._l2_shrinkage_regularization_strength <= 0.0:
      return gen_training_ops.resource_apply_ftrl(
          var.handle,
          accum.handle,
          linear.handle,
          grad,
          math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
          math_ops.cast(self._l1_regularization_strength_tensor,
                        var.dtype.base_dtype),
          math_ops.cast(self._adjusted_l2_regularization_strength_tensor,
                        var.dtype.base_dtype),
          math_ops.cast(self._learning_rate_power_tensor, var.dtype.base_dtype),
          use_locking=self._use_locking)
    else:
      return gen_training_ops.resource_apply_ftrl_v2(
          var.handle,
          accum.handle,
          linear.handle,
          grad,
          math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
          math_ops.cast(self._l1_regularization_strength_tensor,
                        var.dtype.base_dtype),
          math_ops.cast(self._adjusted_l2_regularization_strength_tensor,
                        var.dtype.base_dtype),
          math_ops.cast(self._l2_shrinkage_regularization_strength_tensor,
                        var.dtype.base_dtype),
          math_ops.cast(self._learning_rate_power_tensor, var.dtype.base_dtype),
          use_locking=self._use_locking)

  def _apply_sparse(self, grad, var):
    # [FUNCTION DOC]
    # KNOWN ISSUE: in the shrinkage branch the shrinkage strength is cast to the gradient dtype
    # while all other hyper-parameters are cast to the variable dtype; harmless when the dtypes
    # match, inconsistent otherwise.
    # Applies a sparse FTRL update to a reference variable. Inputs: grad (indexed-slices
    # gradient with values and indices) and var (variable). Uses the sparse FTRL op, or its
    # L2-shrinkage version when the shrinkage strength is positive. Returns the update
    # operation.
    accum = self.get_slot(var, "accum")
    linear = self.get_slot(var, "linear")
    if self._l2_shrinkage_regularization_strength <= 0.0:
      return gen_training_ops.sparse_apply_ftrl(
          var,
          accum,
          linear,
          grad.values,
          grad.indices,
          math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
          math_ops.cast(self._l1_regularization_strength_tensor,
                        var.dtype.base_dtype),
          math_ops.cast(self._adjusted_l2_regularization_strength_tensor,
                        var.dtype.base_dtype),
          math_ops.cast(self._learning_rate_power_tensor, var.dtype.base_dtype),
          use_locking=self._use_locking)
    else:
      return gen_training_ops.sparse_apply_ftrl_v2(
          var,
          accum,
          linear,
          grad.values,
          grad.indices,
          math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
          math_ops.cast(self._l1_regularization_strength_tensor,
                        var.dtype.base_dtype),
          math_ops.cast(self._adjusted_l2_regularization_strength_tensor,
                        var.dtype.base_dtype),
          math_ops.cast(self._l2_shrinkage_regularization_strength_tensor,
                        grad.dtype.base_dtype),
          math_ops.cast(self._learning_rate_power_tensor, var.dtype.base_dtype),
          use_locking=self._use_locking)

  def _resource_apply_sparse(self, grad, var, indices):
    # [FUNCTION DOC]
    # Applies a sparse FTRL update to a resource variable. Inputs: grad (gradient values for
    # the touched rows), var (resource variable) and indices (row indices being updated).
    # Chooses the plain or L2-shrinkage resource sparse op by the shrinkage strength; here the
    # hyper-parameters are cast to the gradient dtype. Returns the update operation.
    accum = self.get_slot(var, "accum")
    linear = self.get_slot(var, "linear")
    if self._l2_shrinkage_regularization_strength <= 0.0:
      return gen_training_ops.resource_sparse_apply_ftrl(
          var.handle,
          accum.handle,
          linear.handle,
          grad,
          indices,
          math_ops.cast(self._learning_rate_tensor, grad.dtype),
          math_ops.cast(self._l1_regularization_strength_tensor, grad.dtype),
          math_ops.cast(self._adjusted_l2_regularization_strength_tensor,
                        grad.dtype),
          math_ops.cast(self._learning_rate_power_tensor, grad.dtype),
          use_locking=self._use_locking)
    else:
      return gen_training_ops.resource_sparse_apply_ftrl_v2(
          var.handle,
          accum.handle,
          linear.handle,
          grad,
          indices,
          math_ops.cast(self._learning_rate_tensor, grad.dtype),
          math_ops.cast(self._l1_regularization_strength_tensor, grad.dtype),
          math_ops.cast(self._adjusted_l2_regularization_strength_tensor,
                        grad.dtype),
          math_ops.cast(self._l2_shrinkage_regularization_strength_tensor,
                        grad.dtype),
          math_ops.cast(self._learning_rate_power_tensor, grad.dtype),
          use_locking=self._use_locking)
