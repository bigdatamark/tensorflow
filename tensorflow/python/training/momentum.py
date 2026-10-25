# [MODULE DOC]
# Defines the legacy (TF1 style) momentum optimizer, exported as the v1 train momentum optimizer.
# It builds on the base optimizer class and uses the generated training kernels for dense and
# sparse, reference-variable and resource-variable updates. Import-time side effect: the export
# decorator registers the class in the public v1 API.
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

"""Momentum for TensorFlow."""
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_training_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer
from tensorflow.python.util.tf_export import tf_export


@tf_export(v1=["train.MomentumOptimizer"])
class MomentumOptimizer(optimizer.Optimizer):
  # [CLASS DOC]
  # Optimizer implementing gradient descent with momentum, optionally Nesterov momentum. Holds the
  # learning rate and momentum (numbers, tensors or, in eager mode, callables returning them) and
  # the nesterov flag; creates one "momentum" accumulator slot per variable. Dense updates apply
  # to the whole accumulator; sparse updates touch only the used slices. Instances are used via
  # the inherited minimize / apply_gradients methods, which call the private apply hooks below.
  """Optimizer that implements the Momentum algorithm.

  Computes (if `use_nesterov = False`):

  ```
  accumulation = momentum * accumulation + gradient
  variable -= learning_rate * accumulation
  ```

  Note that in the dense version of this algorithm, `accumulation` is updated
  and applied regardless of a gradient's value, whereas the sparse version (when
  the gradient is an `IndexedSlices`, typically because of `tf.gather` or an
  embedding) only updates variable slices and corresponding `accumulation` terms
  when that part of the variable was used in the forward pass.

  @compatibility(TF2)
  tf.compat.v1.train.MomentumOptimizer is compatible with eager mode and
  `tf.function`.
  When eager execution is enabled, `learning_rate`,`momentum`, can each be a
  callable that takes no arguments and returns the actual value to use. This
  can be useful for changing these values across different invocations of
  optimizer functions.

  To switch to native TF2 style, please directly use
  [`tf.keras.optimizers.SGD`]
  (https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/SGD)
  with the `momentum` argument.

  #### Structural mapping to native TF2

  Before:

  ```python
  optimizer = tf.compat.v1.train.MomentumOptimizer(
    learning_rate=learning_rate,
    momentum=momentum,
    use_nesterov=use_nesterov)
  ```

  After:

  ```python
  optimizer = tf.keras.optimizers.SGD(
    learning_rate=learning_rate,
    momentum=momentum,
    nesterov=use_nesterov)
  ```

  #### How to map arguments
  | TF1 Arg Name       | TF2 Arg Name   | Note                             |
  | ------------------ | -------------  | -------------------------------  |
  | `learning_rate`    | `learning_rate`| Be careful of setting           |
  : : : learning_rate tensor value computed from the global step.          :
  : : : In TF1 this was usually meant to imply a dynamic learning rate and :
  : : : would recompute in each step. In TF2 (eager + function) it will    :
  : : : treat it as a scalar value that only gets computed once instead of :
  : : : a symbolic placeholder to be computed each time.                   :
  | `momentum`         | `momentum`     | -                                |
  | `use_locking`      | -              | Not applicable in TF2.           |
  | `use_nesterov`     | `nesterov`     | -                                |

  #### Before & after usage example
  Before:

  ```python
  x = tf.Variable([1,2,3], dtype=tf.float32)
  grad = tf.constant([0.1, 0.2, 0.3])
  optimizer = tf.compat.v1.train.MomentumOptimizer(
    learning_rate=0.001,
    momentum=0.9,
    use_nesterov=False)
  optimizer.apply_gradients(zip([grad], [x]))
  ```

  After:

  ```python
  x = tf.Variable([1,2,3], dtype=tf.float32)
  grad = tf.constant([0.1, 0.2, 0.3])
  optimizer = tf.keras.optimizers.SGD(
    learning_rate=0.001,
    momentum=0.9,
    nesterov=False)
  optimizer.apply_gradients(zip([grad], [x]))
  ```

  @end_compatibility

  """

  def __init__(self, learning_rate, momentum,
               use_locking=False, name="Momentum", use_nesterov=False):
    # [FUNCTION DOC]
    # Constructor. Inputs: learning_rate and momentum (tensor, number, or callable in eager mode,
    # required); use_locking (boolean, default false) for update locking; name (string, default
    # "Momentum"); use_nesterov (boolean, default false). Calls the base constructor with locking
    # and name, and stores the three values for later use. Returns nothing; no graph ops created.
    """Construct a new Momentum optimizer.

    Args:
      learning_rate: A `Tensor` or a floating point value.  The learning rate.
      momentum: A `Tensor` or a floating point value.  The momentum.
      use_locking: If `True` use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients.  Defaults to "Momentum".
      use_nesterov: If `True` use Nesterov Momentum.
        See (Sutskever et al., 2013).
        This implementation always computes gradients at the value of the
        variable(s) passed to the optimizer. Using Nesterov Momentum makes the
        variable(s) track the values called `theta_t + mu*v_t` in the paper.
        This implementation is an approximation of the original formula, valid
        for high values of momentum. It will compute the "adjusted gradient"
        in NAG by assuming that the new gradient will be estimated by the
        current average gradient plus the product of momentum and the change
        in the average gradient.

    References:
      On the importance of initialization and momentum in deep learning:
        [Sutskever et al., 2013]
        (http://proceedings.mlr.press/v28/sutskever13.html)
        ([pdf](http://proceedings.mlr.press/v28/sutskever13.pdf))


    """
    super(MomentumOptimizer, self).__init__(use_locking, name)
    self._learning_rate = learning_rate
    self._momentum = momentum
    self._use_nesterov = use_nesterov

  def _create_slots(self, var_list):
    # [FUNCTION DOC]
    # HELPER: slot creation hook called by the base optimizer when applying gradients.
    # Input: list of variables to be trained. For each, creates a zero-initialized slot named
    # "momentum" (scoped under the optimizer name). Returns nothing.
    for v in var_list:
      self._zeros_slot(v, "momentum", self._name)

  def _prepare(self):
    # [FUNCTION DOC]
    # HELPER: pre-update hook called by the base optimizer before applying gradients.
    # Resolves the learning rate and momentum (calling them if they are callables) and converts
    # each to a tensor named "learning_rate" and "momentum", stored as private attributes used by
    # the apply methods. Returns nothing; creates constant ops in the current graph.
    learning_rate = self._learning_rate
    if callable(learning_rate):
      learning_rate = learning_rate()
    self._learning_rate_tensor = ops.convert_to_tensor(learning_rate,
                                                       name="learning_rate")
    momentum = self._momentum
    if callable(momentum):
      momentum = momentum()
    self._momentum_tensor = ops.convert_to_tensor(momentum, name="momentum")

  def _apply_dense(self, grad, var):
    # [FUNCTION DOC]
    # HELPER: update hook for dense gradients on reference variables.
    # Inputs: grad (dense gradient tensor) and var (variable). Looks up the momentum slot, casts
    # learning rate and momentum to the variable's base dtype, and runs the fused apply-momentum
    # kernel with the locking and nesterov settings. Returns the operation performing the update.
    mom = self.get_slot(var, "momentum")
    return gen_training_ops.apply_momentum(
        var, mom,
        math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
        grad,
        math_ops.cast(self._momentum_tensor, var.dtype.base_dtype),
        use_locking=self._use_locking,
        use_nesterov=self._use_nesterov).op

  def _resource_apply_dense(self, grad, var):
    # [FUNCTION DOC]
    # HELPER: update hook for dense gradients on resource variables.
    # Inputs: grad (dense gradient tensor) and var (resource variable). Like the reference version
    # but passes the variable and slot handles and casts hyperparameters to the gradient dtype.
    # Returns the update operation (or nothing meaningful in eager mode).
    mom = self.get_slot(var, "momentum")
    return gen_training_ops.resource_apply_momentum(
        var.handle, mom.handle,
        math_ops.cast(self._learning_rate_tensor, grad.dtype.base_dtype),
        grad,
        math_ops.cast(self._momentum_tensor, grad.dtype.base_dtype),
        use_locking=self._use_locking,
        use_nesterov=self._use_nesterov)

  def _apply_sparse(self, grad, var):
    # [FUNCTION DOC]
    # HELPER: update hook for sparse gradients on reference variables.
    # Inputs: grad (indexed slices with values and indices) and var (variable). Runs the sparse
    # apply-momentum kernel so only the referenced rows of variable and accumulator are updated;
    # hyperparameters cast to the variable's base dtype. Returns the update operation.
    mom = self.get_slot(var, "momentum")
    return gen_training_ops.sparse_apply_momentum(
        var, mom,
        math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
        grad.values, grad.indices,
        math_ops.cast(self._momentum_tensor, var.dtype.base_dtype),
        use_locking=self._use_locking,
        use_nesterov=self._use_nesterov).op

  def _resource_apply_sparse(self, grad, var, indices):
    # [FUNCTION DOC]
    # HELPER: update hook for sparse gradients on resource variables.
    # Inputs: grad (values tensor), var (resource variable), indices (row indices to update).
    # Runs the resource sparse apply-momentum kernel using the variable and slot handles, with
    # hyperparameters cast to the gradient dtype. Returns the update operation.
    mom = self.get_slot(var, "momentum")
    return gen_training_ops.resource_sparse_apply_momentum(
        var.handle, mom.handle,
        math_ops.cast(self._learning_rate_tensor, grad.dtype),
        grad, indices,
        math_ops.cast(self._momentum_tensor, grad.dtype),
        use_locking=self._use_locking,
        use_nesterov=self._use_nesterov)
