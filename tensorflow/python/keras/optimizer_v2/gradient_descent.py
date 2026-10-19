# [MODULE DOC]
# Defines the legacy Keras (v2 optimizer API) SGD optimizer: plain gradient descent with optional
# classical or Nesterov momentum. It builds on the generic optimizer base class in the
# optimizer_v2 package and uses fused training kernels for dense updates and for sparse updates
# with momentum, and a direct scatter-add for sparse updates without momentum. Import has no
# side effects beyond defining the class.
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
"""SGD optimizer implementation."""
# pylint: disable=g-classes-have-attributes

from tensorflow.python.framework import tensor
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gen_training_ops


class SGD(optimizer_v2.OptimizerV2):
  # [CLASS DOC]
  # Stochastic gradient descent optimizer with optional momentum and Nesterov momentum. Holds
  # hyperparameters learning rate, decay and momentum (managed by the base class), a flag saying
  # whether momentum is in use (which decides whether a momentum slot is created per variable)
  # and the nesterov flag. Created by user code and driven by the base class minimize /
  # apply-gradients flow.
  r"""Gradient descent (with momentum) optimizer.

  Update rule for parameter `w` with gradient `g` when `momentum` is 0:

  ```python
  w = w - learning_rate * g
  ```

  Update rule when `momentum` is larger than 0:

  ```python
  velocity = momentum * velocity - learning_rate * g
  w = w + velocity
  ```

  When `nesterov=True`, this rule becomes:

  ```python
  velocity = momentum * velocity - learning_rate * g
  w = w + momentum * velocity - learning_rate * g
  ```

  Args:
    learning_rate: A `Tensor`, floating point value, or a schedule that is a
      `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
      that takes no arguments and returns the actual value to use. The
      learning rate. Defaults to 0.01.
    momentum: float hyperparameter >= 0 that accelerates gradient descent
      in the relevant
      direction and dampens oscillations. Defaults to 0, i.e., vanilla gradient
      descent.
    nesterov: boolean. Whether to apply Nesterov momentum.
      Defaults to `False`.
    name: Optional name prefix for the operations created when applying
      gradients.  Defaults to `"SGD"`.
    **kwargs: Keyword arguments. Allowed to be one of
      `"clipnorm"` or `"clipvalue"`.
      `"clipnorm"` (float) clips gradients by norm; `"clipvalue"` (float) clips
      gradients by value.

  Usage:

  >>> opt = tf.keras.optimizers.SGD(learning_rate=0.1)
  >>> var = tf.Variable(1.0)
  >>> loss = lambda: (var ** 2)/2.0         # d(loss)/d(var1) = var1
  >>> step_count = opt.minimize(loss, [var]).numpy()
  >>> # Step is `- learning_rate * grad`
  >>> var.numpy()
  0.9

  >>> opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
  >>> var = tf.Variable(1.0)
  >>> val0 = var.value()
  >>> loss = lambda: (var ** 2)/2.0         # d(loss)/d(var1) = var1
  >>> # First step is `- learning_rate * grad`
  >>> step_count = opt.minimize(loss, [var]).numpy()
  >>> val1 = var.value()
  >>> (val0 - val1).numpy()
  0.1
  >>> # On later steps, step-size increases because of momentum
  >>> step_count = opt.minimize(loss, [var]).numpy()
  >>> val2 = var.value()
  >>> (val1 - val2).numpy()
  0.18

  Reference:
      - For `nesterov=True`, See [Sutskever et al., 2013](
        http://jmlr.org/proceedings/papers/v28/sutskever13.pdf).
  """

  _HAS_AGGREGATE_GRAD = True

  def __init__(self,
               learning_rate=0.01,
               momentum=0.0,
               nesterov=False,
               name="SGD",
               **kwargs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says momentum only needs to be >= 0, but a plain number
    # outside the range zero to one raises an error; tensor or callable momentum values are
    # not range checked at all.
    # Constructs the optimizer. Inputs: learning_rate (number, tensor, schedule or callable,
    # default 0.01; the legacy keyword "lr" in the extra keyword arguments takes precedence),
    # momentum (default 0.0), nesterov (use Nesterov momentum, default false), name (default
    # "SGD") and extra keyword arguments forwarded to the base class. Momentum is considered
    # enabled when it is a tensor, a callable or a positive number. Raises a value error when a
    # numeric momentum is below zero or above one. Registers hyperparameters, sets the
    # momentum-enabled flag and stores nesterov. Returns nothing.
    super(SGD, self).__init__(name, **kwargs)
    self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
    self._set_hyper("decay", self._initial_decay)

    self._momentum = False
    if isinstance(
        momentum, tensor.Tensor) or callable(momentum) or momentum > 0:
      self._momentum = True
    if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
      raise ValueError("`momentum` must be between [0, 1].")
    self._set_hyper("momentum", momentum)

    self.nesterov = nesterov

  def _create_slots(self, var_list):
    # [FUNCTION DOC]
    # HELPER: slot setup hook called by the optimizer base class. When momentum is enabled,
    # creates a momentum slot for every variable in var_list; otherwise does nothing. Returns
    # nothing; creates slot variables as a side effect.
    if self._momentum:
      for var in var_list:
        self.add_slot(var, "momentum")

  def _prepare_local(self, var_device, var_dtype, apply_state):
    # [FUNCTION DOC]
    # HELPER: per-device, per-dtype coefficient precomputation. Calls the base implementation,
    # then stores the momentum hyperparameter, converted to the variable dtype, in the
    # apply_state entry keyed by (device, dtype). Returns nothing; mutates apply_state.
    super(SGD, self)._prepare_local(var_device, var_dtype, apply_state)
    apply_state[(var_device, var_dtype)]["momentum"] = array_ops.identity(
        self._get_hyper("momentum", var_dtype))

  def _resource_apply_dense(self, grad, var, apply_state=None):
    # [FUNCTION DOC]
    # HELPER: dense update step called by the optimizer base class. Inputs: grad (dense
    # gradient), var (resource variable), apply_state (optional coefficient mapping; a fallback
    # is computed if missing). With momentum enabled runs the fused Keras momentum kernel using
    # the momentum slot and the nesterov flag; otherwise runs the fused gradient descent kernel.
    # Returns the update operation; modifies the variable (and momentum slot) in place.
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    if self._momentum:
      momentum_var = self.get_slot(var, "momentum")
      return gen_training_ops.ResourceApplyKerasMomentum(
          var=var.handle,
          accum=momentum_var.handle,
          lr=coefficients["lr_t"],
          grad=grad,
          momentum=coefficients["momentum"],
          use_locking=self._use_locking,
          use_nesterov=self.nesterov)
    else:
      return gen_training_ops.ResourceApplyGradientDescent(
          var=var.handle,
          alpha=coefficients["lr_t"],
          delta=grad,
          use_locking=self._use_locking)

  def _resource_apply_sparse_duplicate_indices(self, grad, var, indices,
                                               **kwargs):
    # [FUNCTION DOC]
    # HELPER: sparse update entry point that may receive repeated indices. Inputs: grad
    # (gradient row values), var (variable), indices (row indices, duplicates allowed) and
    # keyword arguments that may contain an apply_state mapping. With momentum enabled it defers
    # to the base class implementation (which merges duplicate indices and calls the sparse
    # apply method). Without momentum it directly scatter-adds the negative gradient scaled by
    # the learning rate into the variable, which handles duplicates naturally. Returns the
    # update operation.
    if self._momentum:
      return super(SGD, self)._resource_apply_sparse_duplicate_indices(
          grad, var, indices, **kwargs)
    else:
      var_device, var_dtype = var.device, var.dtype.base_dtype
      coefficients = (kwargs.get("apply_state", {}).get((var_device, var_dtype))
                      or self._fallback_apply_state(var_device, var_dtype))

      return gen_resource_variable_ops.ResourceScatterAdd(
          resource=var.handle,
          indices=indices,
          updates=-grad * coefficients["lr_t"])

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    # [FUNCTION DOC]
    # HELPER: sparse update step used only when momentum is enabled. Inputs: grad (gradient row
    # values), var (variable), indices (unique row indices), apply_state (optional coefficient
    # mapping). Runs the fused sparse Keras momentum kernel on the variable and its momentum
    # slot with the nesterov flag. Returns the update operation. Fails if no momentum slot
    # exists, i.e. when momentum is disabled.
    # This method is only needed for momentum optimization.
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    momentum_var = self.get_slot(var, "momentum")
    return gen_training_ops.ResourceSparseApplyKerasMomentum(
        var=var.handle,
        accum=momentum_var.handle,
        lr=coefficients["lr_t"],
        grad=grad,
        indices=indices,
        momentum=coefficients["momentum"],
        use_locking=self._use_locking,
        use_nesterov=self.nesterov)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a serializable mapping of the base configuration extended with the learning rate,
    # decay, momentum and nesterov flag. No side effects.
    config = super(SGD, self).get_config()
    config.update({
        "learning_rate": self._serialize_hyperparameter("learning_rate"),
        "decay": self._initial_decay,
        "momentum": self._serialize_hyperparameter("momentum"),
        "nesterov": self.nesterov,
    })
    return config
