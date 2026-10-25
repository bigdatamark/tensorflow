# [MODULE DOC]
# Defines the v1 Proximal Adagrad optimizer, a subclass of the base training optimizer. It keeps
# one accumulator slot per variable and delegates the actual update to the fused native training
# operations (dense, resource-dense, sparse, resource-sparse variants), which apply the adaptive
# step followed by an L1/L2 proximal shrinkage. Import-time effect: the class is exported under the
# v1 training namespace.
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

"""ProximalAdagrad for TensorFlow."""
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_training_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer
from tensorflow.python.util.tf_export import tf_export


@tf_export(v1=["train.ProximalAdagradOptimizer"])
class ProximalAdagradOptimizer(optimizer.Optimizer):
  # [CLASS DOC]
  # Optimizer implementing Adagrad with a proximal step for L1 and L2 regularization. State: the
  # learning rate, the initial accumulator value, the two regularization strengths, and tensor
  # versions of those hyperparameters created lazily in the prepare step. Each trainable variable
  # gets one slot named accumulator, initialized to the initial accumulator value. Instances are
  # constructed with hyperparameters and then used through the base class minimize or
  # apply-gradients flow, which calls the overridden slot-creation, prepare and apply hooks.
  # pylint: disable=line-too-long
  """Optimizer that implements the Proximal Adagrad algorithm.

  References:
    Adaptive Subgradient Methods for Online Learning and Stochastic Optimization:
      [Duchi et al., 2011](http://jmlr.org/papers/v12/duchi11a.html)
      ([pdf](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf))
    Efficient Learning using Forward-Backward Splitting:
      [Duchi et al., 2009](http://papers.nips.cc/paper/3793-efficient-learning-using-forward-backward-splitting)
      ([pdf](http://papers.nips.cc/paper/3793-efficient-learning-using-forward-backward-splitting.pdf))
  """

  def __init__(self, learning_rate, initial_accumulator_value=0.1,
               l1_regularization_strength=0.0, l2_regularization_strength=0.0,
               use_locking=False, name="ProximalAdagrad"):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says the name defaults to "Adagrad", but the default is
    # "ProximalAdagrad".
    # Constructor. Inputs: learning_rate (tensor or float, required); initial_accumulator_value
    # (float, default 0.1, must be strictly positive); l1_regularization_strength and
    # l2_regularization_strength (floats, default 0.0, documented as non-negative but not
    # validated); use_locking (bool, default false, passed to the base class); name (string
    # prefix for created operations). Raises a value error if the initial accumulator value is
    # zero or negative. Stores the hyperparameters and sets the derived tensor attributes to null
    # until the prepare step. Returns nothing.
    """Construct a new ProximalAdagrad optimizer.

    Args:
      learning_rate: A `Tensor` or a floating point value.  The learning rate.
      initial_accumulator_value: A floating point value.
        Starting value for the accumulators, must be positive.
      l1_regularization_strength: A float value, must be greater than or
        equal to zero.
      l2_regularization_strength: A float value, must be greater than or
        equal to zero.
      use_locking: If `True` use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients.  Defaults to "Adagrad".

    Raises:
      ValueError: If the `initial_accumulator_value` is invalid.
    """
    if initial_accumulator_value <= 0.0:
      raise ValueError("initial_accumulator_value must be positive: %s" %
                       initial_accumulator_value)
    super(ProximalAdagradOptimizer, self).__init__(use_locking, name)
    self._learning_rate = learning_rate
    self._initial_accumulator_value = initial_accumulator_value
    self._l1_regularization_strength = l1_regularization_strength
    self._l2_regularization_strength = l2_regularization_strength
    # Created in Initialize.
    self._l1_regularization_strength_tensor = None
    self._l2_regularization_strength_tensor = None
    self._learning_rate_tensor = None

  def _create_slots(self, var_list):
    # [FUNCTION DOC]
    # HELPER: slot-creation hook called by the base optimizer before applying gradients.
    # For each variable in var_list, creates a constant tensor with the variable's shape and base
    # dtype filled with the initial accumulator value (placed on the variable's device), and
    # registers it as the slot named accumulator. Returns nothing; side effect is slot creation.
    for v in var_list:
      with ops.colocate_with(v):
        val = constant_op.constant(self._initial_accumulator_value,
                                   shape=v.get_shape(),
                                   dtype=v.dtype.base_dtype)
      self._get_or_make_slot(v, val, "accumulator", self._name)

  def _prepare(self):
    # [FUNCTION DOC]
    # HELPER: prepare hook called by the base optimizer once per apply call.
    # Converts the learning rate and the L1 and L2 strengths into tensors (named learning_rate,
    # l1_regularization_strength, l2_regularization_strength) and stores them on the instance for
    # use by the apply methods. Returns nothing.
    self._learning_rate_tensor = ops.convert_to_tensor(self._learning_rate,
                                                       name="learning_rate")
    self._l1_regularization_strength_tensor = ops.convert_to_tensor(
        self._l1_regularization_strength,
        name="l1_regularization_strength")
    self._l2_regularization_strength_tensor = ops.convert_to_tensor(
        self._l2_regularization_strength,
        name="l2_regularization_strength")

  def _apply_dense(self, grad, var):
    # [FUNCTION DOC]
    # HELPER: dense update hook for reference-style variables.
    # Inputs: grad (dense gradient tensor) and var (the variable). Looks up the accumulator slot
    # and runs the native proximal Adagrad update in place using the prepared hyperparameter
    # tensors and the instance locking flag. Returns the update operation's output.
    acc = self.get_slot(var, "accumulator")
    return gen_training_ops.apply_proximal_adagrad(
        var, acc, self._learning_rate_tensor,
        self._l1_regularization_strength_tensor,
        self._l2_regularization_strength_tensor,
        grad, use_locking=self._use_locking)

  def _resource_apply_dense(self, grad, var):
    # [FUNCTION DOC]
    # HELPER: dense update hook for resource variables.
    # Same as the reference-variable dense update but passes the resource handles of the variable
    # and its accumulator slot. Input grad is a dense gradient tensor. Returns the update operation.
    acc = self.get_slot(var, "accumulator")
    return gen_training_ops.resource_apply_proximal_adagrad(
        var.handle, acc.handle, self._learning_rate_tensor,
        self._l1_regularization_strength_tensor,
        self._l2_regularization_strength_tensor,
        grad, use_locking=self._use_locking)

  def _apply_sparse(self, grad, var):
    # [FUNCTION DOC]
    # HELPER: sparse update hook for reference-style variables.
    # Input grad is a sparse (indexed slices) gradient; its values and indices are passed to the
    # native sparse proximal Adagrad update together with the accumulator slot and prepared
    # hyperparameters. Only the referenced rows are updated. Returns the update operation's output.
    acc = self.get_slot(var, "accumulator")
    return gen_training_ops.sparse_apply_proximal_adagrad(
        var, acc, self._learning_rate_tensor,
        self._l1_regularization_strength_tensor,
        self._l2_regularization_strength_tensor,
        grad.values, grad.indices,
        use_locking=self._use_locking)

  def _resource_apply_sparse(self, grad, var, indices):
    # [FUNCTION DOC]
    # HELPER: sparse update hook for resource variables.
    # Inputs: grad (gradient values for the selected rows), var (resource variable), indices (row
    # indices to update). The prepared hyperparameter tensors are cast to the gradient dtype before
    # calling the native resource sparse update with the variable and accumulator handles.
    # Returns the update operation.
    acc = self.get_slot(var, "accumulator")
    return gen_training_ops.resource_sparse_apply_proximal_adagrad(
        var.handle, acc.handle,
        math_ops.cast(self._learning_rate_tensor, grad.dtype),
        math_ops.cast(self._l1_regularization_strength_tensor, grad.dtype),
        math_ops.cast(self._l2_regularization_strength_tensor, grad.dtype),
        grad, indices,
        use_locking=self._use_locking)
