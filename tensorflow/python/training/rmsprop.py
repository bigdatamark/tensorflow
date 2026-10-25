# [MODULE DOC]
# Defines the v1 RMSProp optimizer (exported under the train namespace). It keeps per-variable
# slots for a running average of squared gradients, momentum, and, in centered mode, a running
# average of gradients, and applies updates through native training kernels for dense, sparse,
# resource-variable dense and resource-variable sparse gradients. Builds on the base optimizer
# class. Import-time side effect: registers the export only.
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
"""One-line documentation for rmsprop module.

rmsprop algorithm [tieleman2012rmsprop]

A detailed description of rmsprop.

- maintain a moving (discounted) average of the square of gradients
- divide gradient by the root of this average

mean_square = decay * mean_square{t-1} + (1-decay) * gradient ** 2
mom = momentum * mom{t-1} + learning_rate * g_t / sqrt(mean_square + epsilon)
delta = - mom

This implementation of RMSProp uses plain momentum, not Nesterov momentum.

The centered version additionally maintains a moving (discounted) average of the
gradients, and uses that average to estimate the variance:

mean_grad = decay * mean_grad{t-1} + (1-decay) * gradient
mean_square = decay * mean_square{t-1} + (1-decay) * gradient ** 2
mom = momentum * mom{t-1} + learning_rate * g_t /
    sqrt(mean_square - mean_grad**2 + epsilon)
delta = - mom
"""

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_training_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer
from tensorflow.python.util.tf_export import tf_export


@tf_export(v1=["train.RMSPropOptimizer"])
class RMSPropOptimizer(optimizer.Optimizer):
  # [CLASS DOC]
  # Optimizer implementing RMSProp with plain (not Nesterov) momentum and an optional centered
  # variant. State: learning rate, decay, momentum and epsilon (each a number, tensor or callable
  # with no arguments), the centered flag, and tensor copies created when preparing an update.
  # Per variable it owns slots named rms (initialized to ones), momentum (zeros) and, if centered,
  # mg (zeros). Subclasses the base optimizer and overrides its slot creation, prepare and apply
  # hooks. Created directly by user code and used through the base minimize/apply methods.
  """Optimizer that implements the RMSProp algorithm (Tielemans et al.

  2012).

  References:
    Coursera slide 29:
    Hinton, 2012
    ([pdf](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf))

  @compatibility(TF2)
  tf.compat.v1.train.RMSPropOptimizer is compatible with eager mode and
  `tf.function`.
  When eager execution is enabled, `learning_rate`, `decay`, `momentum`,
  and `epsilon` can each be a callable that
  takes no arguments and returns the actual value to use. This can be useful
  for changing these values across different invocations of optimizer
  functions.

  To switch to native TF2 style, use [`tf.keras.optimizers.RMSprop`]
  (https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/RMSprop)
  instead. Please notice that due to the implementation differences,
  `tf.keras.optimizers.RMSprop` and
  `tf.compat.v1.train.RMSPropOptimizer` may have slight differences in
  floating point numerics even though the formula used for the variable
  updates still matches.

  #### Structural mapping to native TF2

  Before:

  ```python
  optimizer = tf.compat.v1.train.RMSPropOptimizer(
    learning_rate=learning_rate,
    decay=decay,
    momentum=momentum,
    epsilon=epsilon)
  ```

  After:

  ```python
  optimizer = tf.keras.optimizers.RMSprop(
    learning_rate=learning_rate,
    rho=decay,
    momentum=momentum,
    epsilon=epsilon)
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
  | `decay`            | `rho`          | -                                |
  | `momentum`         | `momentum`     | -                                |
  | `epsilon`          | `epsilon`      | Default value is 1e-10 in TF1,   |
  :                    :                : but 1e-07 in TF2.                :
  | `use_locking`      | -              | Not applicable in TF2.           |

  #### Before & after usage example
  Before:

  ```python
  x = tf.Variable([1,2,3], dtype=tf.float32)
  grad = tf.constant([0.1, 0.2, 0.3])
  optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=0.001)
  optimizer.apply_gradients(zip([grad], [x]))
  ```

  After:

  ```python
  x = tf.Variable([1,2,3], dtype=tf.float32)
  grad = tf.constant([0.1, 0.2, 0.3])
  optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
  optimizer.apply_gradients(zip([grad], [x]))
  ```

  @end_compatibility
  """

  def __init__(self,
               learning_rate,
               decay=0.9,
               momentum=0.0,
               epsilon=1e-10,
               use_locking=False,
               centered=False,
               name="RMSProp"):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring describes momentum as a scalar tensor and decay vaguely, but any
    # number, tensor or no-argument callable is accepted and stored unvalidated.
    # Constructor. Inputs: learning_rate (required number, tensor or callable); decay (default 0.9);
    # momentum (default 0); epsilon (default 1e-10); use_locking (default false) passed to the base
    # class along with name (default "RMSProp"); centered (default false) selects the centered
    # variant. Stores the hyper-parameters and sets the tensor placeholders (learning rate, decay,
    # momentum, epsilon) to nothing until the prepare step. Raises nothing itself.
    """Construct a new RMSProp optimizer.

    Note that in the dense implementation of this algorithm, variables and their
    corresponding accumulators (momentum, gradient moving average, square
    gradient moving average) will be updated even if the gradient is zero
    (i.e. accumulators will decay, momentum will be applied). The sparse
    implementation (used when the gradient is an `IndexedSlices` object,
    typically because of `tf.gather` or an embedding lookup in the forward pass)
    will not update variable slices or their accumulators unless those slices
    were used in the forward pass (nor is there an "eventual" correction to
    account for these omitted updates). This leads to more efficient updates for
    large embedding lookup tables (where most of the slices are not accessed in
    a particular graph execution), but differs from the published algorithm.

    Args:
      learning_rate: A Tensor or a floating point value.  The learning rate.
      decay: Discounting factor for the history/coming gradient
      momentum: A scalar tensor.
      epsilon: Small value to avoid zero denominator.
      use_locking: If True use locks for update operation.
      centered: If True, gradients are normalized by the estimated variance of
        the gradient; if False, by the uncentered second moment. Setting this to
        True may help with training, but is slightly more expensive in terms of
        computation and memory. Defaults to False.
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to "RMSProp".

    """
    super(RMSPropOptimizer, self).__init__(use_locking, name)
    self._learning_rate = learning_rate
    self._decay = decay
    self._momentum = momentum
    self._epsilon = epsilon
    self._centered = centered

    # Tensors for learning rate and momentum.  Created in _prepare.
    self._learning_rate_tensor = None
    self._decay_tensor = None
    self._momentum_tensor = None
    self._epsilon_tensor = None

  def _create_slots(self, var_list):
    # [FUNCTION DOC]
    # HELPER: slot-creation hook called by the base optimizer.
    # Input: list of variables. For each variable creates an rms slot initialized to ones (using an
    # initializer if the shape is fully known, otherwise a ones tensor shaped like the variable),
    # a zeros mg slot only when centered, and a zeros momentum slot. Returns nothing; adds slot
    # variables to the graph, named under the optimizer name.
    for v in var_list:
      if v.get_shape().is_fully_defined():
        init_rms = init_ops.ones_initializer(dtype=v.dtype.base_dtype)
      else:
        init_rms = array_ops.ones_like(v)
      self._get_or_make_slot_with_initializer(v, init_rms, v.get_shape(),
                                              v.dtype.base_dtype, "rms",
                                              self._name)
      if self._centered:
        self._zeros_slot(v, "mg", self._name)
      self._zeros_slot(v, "momentum", self._name)

  def _prepare(self):
    # [FUNCTION DOC]
    # HELPER: pre-update hook called by the base optimizer.
    # Evaluates any callable hyper-parameters (learning rate, decay, momentum, epsilon) and converts
    # each to a named tensor, storing them on the instance for the apply hooks. Returns nothing.
    lr = self._call_if_callable(self._learning_rate)
    decay = self._call_if_callable(self._decay)
    momentum = self._call_if_callable(self._momentum)
    epsilon = self._call_if_callable(self._epsilon)

    self._learning_rate_tensor = ops.convert_to_tensor(lr, name="learning_rate")
    self._decay_tensor = ops.convert_to_tensor(decay, name="decay")
    self._momentum_tensor = ops.convert_to_tensor(momentum, name="momentum")
    self._epsilon_tensor = ops.convert_to_tensor(epsilon, name="epsilon")

  def _apply_dense(self, grad, var):
    # [FUNCTION DOC]
    # HELPER: dense update hook of the base optimizer for reference variables.
    # Inputs: grad, dense gradient tensor; var, the variable. Fetches rms and momentum slots (and mg
    # when centered), casts the hyper-parameter tensors to the variable's base type and calls the
    # native (centered or plain) RMSProp apply kernel, honoring the locking flag. Returns the
    # resulting operation. Mutates the variable and slots when the operation runs.
    rms = self.get_slot(var, "rms")
    mom = self.get_slot(var, "momentum")
    if self._centered:
      mg = self.get_slot(var, "mg")
      return gen_training_ops.apply_centered_rms_prop(
          var,
          mg,
          rms,
          mom,
          math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
          math_ops.cast(self._decay_tensor, var.dtype.base_dtype),
          math_ops.cast(self._momentum_tensor, var.dtype.base_dtype),
          math_ops.cast(self._epsilon_tensor, var.dtype.base_dtype),
          grad,
          use_locking=self._use_locking).op
    else:
      return gen_training_ops.apply_rms_prop(
          var,
          rms,
          mom,
          math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
          math_ops.cast(self._decay_tensor, var.dtype.base_dtype),
          math_ops.cast(self._momentum_tensor, var.dtype.base_dtype),
          math_ops.cast(self._epsilon_tensor, var.dtype.base_dtype),
          grad,
          use_locking=self._use_locking).op

  def _resource_apply_dense(self, grad, var):
    # [FUNCTION DOC]
    # HELPER: dense update hook of the base optimizer for resource variables.
    # Same as the reference-variable dense update but passes resource handles of the variable and
    # slots and casts hyper-parameters to the gradient's base type. Returns the created operation
    # (not its output). Updates variable and slots in place when run.
    rms = self.get_slot(var, "rms")
    mom = self.get_slot(var, "momentum")
    if self._centered:
      mg = self.get_slot(var, "mg")
      return gen_training_ops.resource_apply_centered_rms_prop(
          var.handle,
          mg.handle,
          rms.handle,
          mom.handle,
          math_ops.cast(self._learning_rate_tensor, grad.dtype.base_dtype),
          math_ops.cast(self._decay_tensor, grad.dtype.base_dtype),
          math_ops.cast(self._momentum_tensor, grad.dtype.base_dtype),
          math_ops.cast(self._epsilon_tensor, grad.dtype.base_dtype),
          grad,
          use_locking=self._use_locking)
    else:
      return gen_training_ops.resource_apply_rms_prop(
          var.handle,
          rms.handle,
          mom.handle,
          math_ops.cast(self._learning_rate_tensor, grad.dtype.base_dtype),
          math_ops.cast(self._decay_tensor, grad.dtype.base_dtype),
          math_ops.cast(self._momentum_tensor, grad.dtype.base_dtype),
          math_ops.cast(self._epsilon_tensor, grad.dtype.base_dtype),
          grad,
          use_locking=self._use_locking)

  def _apply_sparse(self, grad, var):
    # [FUNCTION DOC]
    # HELPER: sparse update hook of the base optimizer for reference variables.
    # Inputs: grad, an indexed-slices gradient (values and indices); var, the variable. Uses the
    # rms, momentum and (when centered) mg slots and calls the native sparse RMSProp kernel,
    # updating only the referenced rows. Returns the resulting operation output. Hyper-parameters
    # are cast to the variable's base type.
    rms = self.get_slot(var, "rms")
    mom = self.get_slot(var, "momentum")
    if self._centered:
      mg = self.get_slot(var, "mg")
      return gen_training_ops.sparse_apply_centered_rms_prop(
          var,
          mg,
          rms,
          mom,
          math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
          math_ops.cast(self._decay_tensor, var.dtype.base_dtype),
          math_ops.cast(self._momentum_tensor, var.dtype.base_dtype),
          math_ops.cast(self._epsilon_tensor, var.dtype.base_dtype),
          grad.values,
          grad.indices,
          use_locking=self._use_locking)
    else:
      return gen_training_ops.sparse_apply_rms_prop(
          var,
          rms,
          mom,
          math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
          math_ops.cast(self._decay_tensor, var.dtype.base_dtype),
          math_ops.cast(self._momentum_tensor, var.dtype.base_dtype),
          math_ops.cast(self._epsilon_tensor, var.dtype.base_dtype),
          grad.values,
          grad.indices,
          use_locking=self._use_locking)

  def _resource_apply_sparse(self, grad, var, indices):
    # [FUNCTION DOC]
    # HELPER: sparse update hook of the base optimizer for resource variables.
    # Inputs: grad, gradient values; var, the resource variable; indices, rows to update. Uses
    # resource handles of variable and slots, casts hyper-parameters to the gradient type and calls
    # the native resource sparse RMSProp kernel (centered or plain). Returns the created operation.
    rms = self.get_slot(var, "rms")
    mom = self.get_slot(var, "momentum")
    if self._centered:
      mg = self.get_slot(var, "mg")
      return gen_training_ops.resource_sparse_apply_centered_rms_prop(
          var.handle,
          mg.handle,
          rms.handle,
          mom.handle,
          math_ops.cast(self._learning_rate_tensor, grad.dtype),
          math_ops.cast(self._decay_tensor, grad.dtype),
          math_ops.cast(self._momentum_tensor, grad.dtype),
          math_ops.cast(self._epsilon_tensor, grad.dtype),
          grad,
          indices,
          use_locking=self._use_locking)
    else:
      return gen_training_ops.resource_sparse_apply_rms_prop(
          var.handle,
          rms.handle,
          mom.handle,
          math_ops.cast(self._learning_rate_tensor, grad.dtype),
          math_ops.cast(self._decay_tensor, grad.dtype),
          math_ops.cast(self._momentum_tensor, grad.dtype),
          math_ops.cast(self._epsilon_tensor, grad.dtype),
          grad,
          indices,
          use_locking=self._use_locking)
