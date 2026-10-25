# [MODULE DOC]
# Defines the v1 proximal gradient descent optimizer, a subclass of the base training optimizer.
# It has no slots; each update is a plain gradient step followed by an L1/L2 proximal shrinkage,
# performed by fused native training operations (dense, resource-dense, sparse, resource-sparse).
# Import-time effect: the class is exported under the v1 training namespace.
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

"""ProximalGradientDescent for TensorFlow."""
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_training_ops
# pylint: disable=unused-import
from tensorflow.python.ops import math_ops
# pylint: enable=unused-import
from tensorflow.python.training import optimizer
from tensorflow.python.util.tf_export import tf_export


@tf_export(v1=["train.ProximalGradientDescentOptimizer"])
class ProximalGradientDescentOptimizer(optimizer.Optimizer):
  # [CLASS DOC]
  # Optimizer implementing gradient descent with a proximal step for L1 and L2 regularization.
  # State: the learning rate and the two regularization strengths, plus tensor versions of them
  # created in the prepare step. It creates no slot variables. Instances are built with the
  # hyperparameters and used through the base class minimize or apply-gradients flow, which calls
  # the overridden prepare and apply hooks.
  # pylint: disable=line-too-long
  """Optimizer that implements the proximal gradient descent algorithm.

  References:
    Efficient Learning using Forward-Backward Splitting:
      [Duchi et al., 2009](http://papers.nips.cc/paper/3793-efficient-learning-using-forward-backward-splitting)
      ([pdf](http://papers.nips.cc/paper/3793-efficient-learning-using-forward-backward-splitting.pdf))
  """

  def __init__(self, learning_rate, l1_regularization_strength=0.0,
               l2_regularization_strength=0.0, use_locking=False,
               name="ProximalGradientDescent"):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says the name defaults to "GradientDescent", but the default is
    # "ProximalGradientDescent".
    # Constructor. Inputs: learning_rate (tensor or float, required); l1_regularization_strength and
    # l2_regularization_strength (floats, default 0.0, documented as non-negative but not
    # validated); use_locking (bool, default false, passed to the base class); name (string prefix
    # for created operations). Stores the hyperparameters and sets the two regularization tensor
    # attributes to null until the prepare step. The learning rate tensor attribute is not
    # initialized here and exists only after preparation. Returns nothing; raises nothing.
    """Construct a new proximal gradient descent optimizer.

    Args:
      learning_rate: A Tensor or a floating point value.  The learning
        rate to use.
      l1_regularization_strength: A float value, must be greater than or
        equal to zero.
      l2_regularization_strength: A float value, must be greater than or
        equal to zero.
      use_locking: If True use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to "GradientDescent".
    """
    super(ProximalGradientDescentOptimizer, self).__init__(use_locking, name)
    self._learning_rate = learning_rate
    self._l1_regularization_strength = l1_regularization_strength
    self._l2_regularization_strength = l2_regularization_strength
    self._l1_regularization_strength_tensor = None
    self._l2_regularization_strength_tensor = None

  def _apply_dense(self, grad, var):
    # [FUNCTION DOC]
    # HELPER: dense update hook for reference-style variables.
    # Inputs: grad (dense gradient tensor), var (variable). Runs the native proximal gradient
    # descent update in place using the prepared tensors and locking flag. Returns the underlying
    # operation of the update (not the tensor).
    return gen_training_ops.apply_proximal_gradient_descent(
        var,
        self._learning_rate_tensor,
        self._l1_regularization_strength_tensor,
        self._l2_regularization_strength_tensor,
        grad,
        use_locking=self._use_locking).op

  def _resource_apply_dense(self, grad, var):
    # [FUNCTION DOC]
    # HELPER: dense update hook for resource variables.
    # Same update as the reference-variable version but uses the variable's resource handle.
    # Input grad is a dense gradient tensor. Returns the update operation.
    return gen_training_ops.resource_apply_proximal_gradient_descent(
        var.handle,
        self._learning_rate_tensor,
        self._l1_regularization_strength_tensor,
        self._l2_regularization_strength_tensor,
        grad,
        use_locking=self._use_locking)

  def _apply_sparse(self, grad, var):
    # [FUNCTION DOC]
    # HELPER: sparse update hook for reference-style variables.
    # Input grad is a sparse (indexed slices) gradient; its values and indices are passed to the
    # native sparse update, so only the referenced rows change. Returns the underlying operation.
    return gen_training_ops.sparse_apply_proximal_gradient_descent(
        var,
        self._learning_rate_tensor,
        self._l1_regularization_strength_tensor,
        self._l2_regularization_strength_tensor,
        grad.values,
        grad.indices,
        use_locking=self._use_locking).op

  def _resource_apply_sparse(self, grad, var, indices):
    # [FUNCTION DOC]
    # HELPER: sparse update hook for resource variables.
    # Inputs: grad (gradient values for selected rows), var (resource variable), indices (rows to
    # update). The prepared hyperparameter tensors are cast to the gradient dtype before the native
    # resource sparse update is called. Returns the update operation.
    return gen_training_ops.resource_sparse_apply_proximal_gradient_descent(
        var.handle,
        math_ops.cast(self._learning_rate_tensor, grad.dtype),
        math_ops.cast(self._l1_regularization_strength_tensor, grad.dtype),
        math_ops.cast(self._l2_regularization_strength_tensor, grad.dtype),
        grad,
        indices,
        use_locking=self._use_locking)

  def _prepare(self):
    # [FUNCTION DOC]
    # HELPER: prepare hook called by the base optimizer before applying gradients.
    # Converts the learning rate and the L1 and L2 strengths into named tensors and stores them on
    # the instance for the apply methods. Returns nothing.
    self._learning_rate_tensor = ops.convert_to_tensor(self._learning_rate,
                                                       name="learning_rate")
    self._l1_regularization_strength_tensor = ops.convert_to_tensor(
        self._l1_regularization_strength, name="l1_regularization_strength")
    self._l2_regularization_strength_tensor = ops.convert_to_tensor(
        self._l2_regularization_strength, name="l2_regularization_strength")
