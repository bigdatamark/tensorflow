# [MODULE DOC]
# Defines the plain gradient descent optimizer (v1 graph-mode API, exported in the v1 train
# namespace). Exposes one class subclassing the base training Optimizer that subtracts the
# learning rate times the gradient from variables, with dense, resource and sparse paths.
# Import-time side effect: export registration through the decorator. Not a helper module.
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

"""GradientDescent for TensorFlow."""
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_training_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import optimizer
from tensorflow.python.util.tf_export import tf_export


@tf_export(v1=["train.GradientDescentOptimizer"])
class GradientDescentOptimizer(optimizer.Optimizer):
  # [CLASS DOC]
  # Optimizer implementing stochastic gradient descent: each variable is decreased by the
  # learning rate times its gradient. Created with a learning rate (value, tensor or, in eager
  # mode, a no-argument callable), a locking flag and a name; used via the base optimizer
  # minimize / apply-gradients flow. Holds the raw learning rate and its tensor form built in
  # the prepare step. No slots are needed. Class-level side effect: exported in the v1 train
  # namespace.
  """Optimizer that implements the gradient descent algorithm.
  """

  def __init__(self, learning_rate, use_locking=False, name="GradientDescent"):
    # [FUNCTION DOC]
    # Constructor. Inputs: learning_rate (float, tensor, or callable in eager mode), use_locking
    # (default false; use locks for updates) and name (default "GradientDescent"). Calls the
    # base constructor and stores the learning rate, leaving its tensor form null until the
    # prepare step. Returns nothing; raises nothing itself.
    """Construct a new gradient descent optimizer.

    Args:
      learning_rate: A Tensor or a floating point value.  The learning
        rate to use.
      use_locking: If True use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to "GradientDescent".

    @compatibility(eager)
    When eager execution is enabled, `learning_rate` can be a callable that
    takes no arguments and returns the actual value to use. This can be useful
    for changing these values across different invocations of optimizer
    functions.
    @end_compatibility
    """
    super(GradientDescentOptimizer, self).__init__(use_locking, name)
    self._learning_rate = learning_rate
    self._learning_rate_tensor = None

  def _apply_dense(self, grad, var):
    # [FUNCTION DOC]
    # Applies a dense gradient descent step to a reference variable. Inputs: grad (dense
    # gradient) and var (variable). Runs the generated gradient descent op with the learning
    # rate tensor cast to the variable dtype. Returns the operation of the update.
    return gen_training_ops.apply_gradient_descent(
        var,
        math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
        grad,
        use_locking=self._use_locking).op

  def _resource_apply_dense(self, grad, handle):
    # [FUNCTION DOC]
    # Applies a dense gradient descent step to a resource variable. Inputs: grad (dense
    # gradient) and handle (the resource variable object, whose handle is used; despite the
    # name it is not a raw handle). Returns the update operation.
    return gen_training_ops.resource_apply_gradient_descent(
        handle.handle, math_ops.cast(self._learning_rate_tensor,
                                     grad.dtype.base_dtype),
        grad, use_locking=self._use_locking)

  def _resource_apply_sparse_duplicate_indices(self, grad, handle, indices):
    # [FUNCTION DOC]
    # Sparse update for a resource variable that tolerates repeated indices. Inputs: grad
    # (values for the touched rows), handle (resource variable object) and indices (row
    # indices, may repeat). Scatter-adds minus the gradient times the learning rate at those
    # rows, so duplicates accumulate. Returns the scatter operation.
    return resource_variable_ops.resource_scatter_add(
        handle.handle,
        indices,
        -grad * math_ops.cast(self._learning_rate_tensor,
                              grad.dtype.base_dtype))

  def _apply_sparse_duplicate_indices(self, grad, var):
    # [FUNCTION DOC]
    # Sparse update for a variable that tolerates repeated indices. Inputs: grad (indexed
    # slices with values, indices and dense shape) and var (variable). Builds indexed slices
    # of the gradient values scaled by the learning rate (cast to the variable dtype) and
    # subtracts them from the variable by scatter, honoring the locking flag. Returns the
    # scatter operation.
    delta = indexed_slices.IndexedSlices(
        grad.values *
        math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
        grad.indices, grad.dense_shape)
    return var.scatter_sub(delta, use_locking=self._use_locking)

  def _prepare(self):
    # [FUNCTION DOC]
    # Override of the base prepare step. Resolves the learning rate (calling it if callable)
    # and converts it to a tensor named learning_rate, stored for use by the apply methods.
    # Returns nothing.
    learning_rate = self._call_if_callable(self._learning_rate)
    self._learning_rate_tensor = ops.convert_to_tensor(
        learning_rate, name="learning_rate")
