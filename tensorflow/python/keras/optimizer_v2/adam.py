# [MODULE DOC]
# Defines two legacy Keras (v2 optimizer API) implementations of the Adam optimizer: Adam, which
# applies updates through fused training kernels, and NonFusedAdam, which expresses the same
# update with ordinary math operations compiled with XLA. Both build on the generic optimizer
# base class from the optimizer_v2 package and add first/second moment slots (plus an optional
# AMSGrad slot). Import has no side effects beyond defining the classes.
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
"""Adam optimizer implementation."""
# pylint: disable=g-classes-have-attributes

from tensorflow.python.eager import def_function
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.keras import backend_config
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_training_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops


class Adam(optimizer_v2.OptimizerV2):
  # [CLASS DOC]
  # Adam optimizer (adaptive first and second moment estimates) using fused native training
  # kernels for dense updates and explicit scatter operations for sparse gradients. Holds
  # hyperparameters learning rate, decay, beta_1 and beta_2 (managed by the base class), the
  # epsilon constant and an amsgrad flag. Creates per-variable slots for the first moment,
  # second moment and, with AMSGrad, the running maximum of the second moment. Created by user
  # code and used via the base class minimize / apply-gradients flow.
  r"""Optimizer that implements the Adam algorithm.

  Adam optimization is a stochastic gradient descent method that is based on
  adaptive estimation of first-order and second-order moments.

  According to
  [Kingma et al., 2014](http://arxiv.org/abs/1412.6980),
  the method is "*computationally
  efficient, has little memory requirement, invariant to diagonal rescaling of
  gradients, and is well suited for problems that are large in terms of
  data/parameters*".

  Args:
    learning_rate: A `Tensor`, floating point value, or a schedule that is a
      `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
      that takes no arguments and returns the actual value to use, The
      learning rate. Defaults to 0.001.
    beta_1: A float value or a constant float tensor, or a callable
      that takes no arguments and returns the actual value to use. The
      exponential decay rate for the 1st moment estimates. Defaults to 0.9.
    beta_2: A float value or a constant float tensor, or a callable
      that takes no arguments and returns the actual value to use, The
      exponential decay rate for the 2nd moment estimates. Defaults to 0.999.
    epsilon: A small constant for numerical stability. This epsilon is
      "epsilon hat" in the Kingma and Ba paper (in the formula just before
      Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to
      1e-7.
    amsgrad: Boolean. Whether to apply AMSGrad variant of this algorithm from
      the paper "On the Convergence of Adam and beyond". Defaults to `False`.
    name: Optional name for the operations created when applying gradients.
      Defaults to `"Adam"`.
    **kwargs: Keyword arguments. Allowed to be one of
      `"clipnorm"` or `"clipvalue"`.
      `"clipnorm"` (float) clips gradients by norm; `"clipvalue"` (float) clips
      gradients by value.

  Usage:

  >>> opt = tf.keras.optimizers.Adam(learning_rate=0.1)
  >>> var1 = tf.Variable(10.0)
  >>> loss = lambda: (var1 ** 2)/2.0       # d(loss)/d(var1) == var1
  >>> step_count = opt.minimize(loss, [var1]).numpy()
  >>> # The first step is `-learning_rate*sign(grad)`
  >>> var1.numpy()
  9.9

  Reference:
    - [Kingma et al., 2014](http://arxiv.org/abs/1412.6980)
    - [Reddi et al., 2018](
        https://openreview.net/pdf?id=ryQu7f-RZ) for `amsgrad`.

  Notes:

  The default value of 1e-7 for epsilon might not be a good default in
  general. For example, when training an Inception network on ImageNet a
  current good choice is 1.0 or 0.1. Note that since Adam uses the
  formulation just before Section 2.1 of the Kingma and Ba paper rather than
  the formulation in Algorithm 1, the "epsilon" referred to here is "epsilon
  hat" in the paper.

  The sparse implementation of this algorithm (used when the gradient is an
  IndexedSlices object, typically because of `tf.gather` or an embedding
  lookup in the forward pass) does apply momentum to variable slices even if
  they were not used in the forward pass (meaning they have a gradient equal
  to zero). Momentum decay (beta1) is also applied to the entire momentum
  accumulator. This means that the sparse behavior is equivalent to the dense
  behavior (in contrast to some momentum implementations which ignore momentum
  unless a variable slice was actually used).
  """

  _HAS_AGGREGATE_GRAD = True

  def __init__(self,
               learning_rate=0.001,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-7,
               amsgrad=False,
               name='Adam',
               **kwargs):
    # [FUNCTION DOC]
    # Constructs the optimizer. Inputs: learning_rate (number, tensor, schedule or callable,
    # default 0.001; an alternative legacy keyword "lr" in the extra keyword arguments takes
    # precedence), beta_1 and beta_2 (exponential decay rates for the moments, defaults 0.9
    # and 0.999), epsilon (numerical stability constant, default 1e-7; a false-like value such
    # as zero or nothing falls back to the backend default epsilon), amsgrad (whether to use
    # the AMSGrad variant), name (operation name prefix) and extra keyword arguments such as
    # clipnorm, clipvalue, decay forwarded to the base class. Registers the hyperparameters
    # and stores epsilon and amsgrad. Returns nothing; the base class may raise on unknown
    # keyword arguments.
    super(Adam, self).__init__(name, **kwargs)
    self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
    self._set_hyper('decay', self._initial_decay)
    self._set_hyper('beta_1', beta_1)
    self._set_hyper('beta_2', beta_2)
    self.epsilon = epsilon or backend_config.epsilon()
    self.amsgrad = amsgrad

  def _create_slots(self, var_list):
    # [FUNCTION DOC]
    # HELPER: slot setup hook called by the optimizer base class before applying gradients.
    # Creates, for every variable in var_list, a first moment slot, then a second moment slot
    # (separate loops to keep the slot ordering of the v1 optimizer), and when amsgrad is
    # enabled also a running-maximum second moment slot. Returns nothing; side effect is
    # creating slot variables.
    # Create slots for the first and second moments.
    # Separate for-loops to respect the ordering of slot variables from v1.
    for var in var_list:
      self.add_slot(var, 'm')
    for var in var_list:
      self.add_slot(var, 'v')
    if self.amsgrad:
      for var in var_list:
        self.add_slot(var, 'vhat')

  def _prepare_local(self, var_device, var_dtype, apply_state):
    # [FUNCTION DOC]
    # HELPER: per-device, per-dtype coefficient precomputation for the update step.
    # Calls the base implementation, then computes the 1-based step count, the beta values,
    # their powers at that step, the bias-corrected learning rate, epsilon as a tensor and the
    # one-minus-beta terms, and merges them into the entry of the apply_state mapping keyed by
    # (device, dtype). Returns nothing; mutates apply_state.
    super(Adam, self)._prepare_local(var_device, var_dtype, apply_state)

    local_step = math_ops.cast(self.iterations + 1, var_dtype)
    beta_1_t = array_ops.identity(self._get_hyper('beta_1', var_dtype))
    beta_2_t = array_ops.identity(self._get_hyper('beta_2', var_dtype))
    beta_1_power = math_ops.pow(beta_1_t, local_step)
    beta_2_power = math_ops.pow(beta_2_t, local_step)
    lr = (apply_state[(var_device, var_dtype)]['lr_t'] *
          (math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)))
    apply_state[(var_device, var_dtype)].update(
        dict(
            lr=lr,
            epsilon=tensor_conversion.convert_to_tensor_v2_with_dispatch(
                self.epsilon, var_dtype
            ),
            beta_1_t=beta_1_t,
            beta_1_power=beta_1_power,
            one_minus_beta_1_t=1 - beta_1_t,
            beta_2_t=beta_2_t,
            beta_2_power=beta_2_power,
            one_minus_beta_2_t=1 - beta_2_t,
        )
    )

  def set_weights(self, weights):
    # [FUNCTION DOC]
    # Restores optimizer state from a list of arrays. Input: weights, a list of values in the
    # order of the optimizer's weights. If the list has the length used by the Keras v1
    # optimizer (three slots per variable plus the iteration counter) it is truncated to the
    # current weight count, dropping the extra second-moment-maximum entries, then delegated
    # to the base class. Returns nothing; the base class raises a value error on count or
    # shape mismatch.
    params = self.weights
    # If the weights are generated by Keras V1 optimizer, it includes vhats
    # even without amsgrad, i.e, V1 optimizer has 3x + 1 variables, while V2
    # optimizer has 2x + 1 variables. Filter vhats out for compatibility.
    num_vars = int((len(params) - 1) / 2)
    if len(weights) == 3 * num_vars + 1:
      weights = weights[:len(params)]
    super(Adam, self).set_weights(weights)

  def _resource_apply_dense(self, grad, var, apply_state=None):
    # [FUNCTION DOC]
    # HELPER: dense update step called by the optimizer base class for each variable.
    # Inputs: grad (dense gradient tensor), var (resource variable to update), apply_state
    # (optional mapping from (device, dtype) to precomputed coefficients; if missing a
    # fallback is computed). Looks up the m and v slots and runs the fused Adam kernel, or the
    # fused AMSGrad kernel (which also uses the vhat slot) when amsgrad is set, honoring the
    # locking setting. Returns the resulting update operation; modifies variable and slots.
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    m = self.get_slot(var, 'm')
    v = self.get_slot(var, 'v')

    if not self.amsgrad:
      return gen_training_ops.ResourceApplyAdam(
          var=var.handle,
          m=m.handle,
          v=v.handle,
          beta1_power=coefficients['beta_1_power'],
          beta2_power=coefficients['beta_2_power'],
          lr=coefficients['lr_t'],
          beta1=coefficients['beta_1_t'],
          beta2=coefficients['beta_2_t'],
          epsilon=coefficients['epsilon'],
          grad=grad,
          use_locking=self._use_locking)
    else:
      vhat = self.get_slot(var, 'vhat')
      return gen_training_ops.ResourceApplyAdamWithAmsgrad(
          var=var.handle,
          m=m.handle,
          v=v.handle,
          vhat=vhat.handle,
          beta1_power=coefficients['beta_1_power'],
          beta2_power=coefficients['beta_2_power'],
          lr=coefficients['lr_t'],
          beta1=coefficients['beta_1_t'],
          beta2=coefficients['beta_2_t'],
          epsilon=coefficients['epsilon'],
          grad=grad,
          use_locking=self._use_locking)

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    # [FUNCTION DOC]
    # HELPER: sparse update step called by the optimizer base class for each variable.
    # Inputs: grad (values of the gradient rows), var (resource variable), indices (row
    # indices the gradient values belong to), apply_state (optional coefficient mapping).
    # Decays the whole first and second moment slots, then scatter-adds the scaled gradient
    # (and squared gradient) at the given indices, so momentum is applied to all rows as in
    # the dense case. Then subtracts the bias-corrected step from the variable, using the
    # running maximum second moment when amsgrad is set (updating that slot). Returns a
    # grouped operation covering the variable update and slot updates.
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    # m_t = beta1 * m + (1 - beta1) * g_t
    m = self.get_slot(var, 'm')
    m_scaled_g_values = grad * coefficients['one_minus_beta_1_t']
    m_t = state_ops.assign(m, m * coefficients['beta_1_t'],
                           use_locking=self._use_locking)
    with ops.control_dependencies([m_t]):
      m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)

    # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
    v = self.get_slot(var, 'v')
    v_scaled_g_values = (grad * grad) * coefficients['one_minus_beta_2_t']
    v_t = state_ops.assign(v, v * coefficients['beta_2_t'],
                           use_locking=self._use_locking)
    with ops.control_dependencies([v_t]):
      v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)

    if not self.amsgrad:
      v_sqrt = math_ops.sqrt(v_t)
      var_update = state_ops.assign_sub(
          var, coefficients['lr'] * m_t / (v_sqrt + coefficients['epsilon']),
          use_locking=self._use_locking)
      return control_flow_ops.group(*[var_update, m_t, v_t])
    else:
      v_hat = self.get_slot(var, 'vhat')
      v_hat_t = math_ops.maximum(v_hat, v_t)
      with ops.control_dependencies([v_hat_t]):
        v_hat_t = state_ops.assign(
            v_hat, v_hat_t, use_locking=self._use_locking)
      v_hat_sqrt = math_ops.sqrt(v_hat_t)
      var_update = state_ops.assign_sub(
          var,
          coefficients['lr'] * m_t / (v_hat_sqrt + coefficients['epsilon']),
          use_locking=self._use_locking)
      return control_flow_ops.group(*[var_update, m_t, v_t, v_hat_t])

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a serializable mapping of the base configuration extended with the current
    # learning rate, decay, beta_1, beta_2, epsilon and amsgrad flag. No side effects.
    config = super(Adam, self).get_config()
    config.update({
        'learning_rate': self._serialize_hyperparameter('learning_rate'),
        'decay': self._initial_decay,
        'beta_1': self._serialize_hyperparameter('beta_1'),
        'beta_2': self._serialize_hyperparameter('beta_2'),
        'epsilon': self.epsilon,
        'amsgrad': self.amsgrad,
    })
    return config


class NonFusedAdam(optimizer_v2.OptimizerV2):
  # [CLASS DOC]
  # Adam optimizer variant that writes the update with plain math and variable-assignment
  # operations, wrapping the per-variable apply functions in XLA-compiled functions, instead of
  # using fused training kernels. Same hyperparameters, slots (m, v, optional vhat), epsilon and
  # amsgrad state as the fused Adam class.
  # KNOWN ISSUE: the dense update subtracts epsilon from the square root of the second moment
  # instead of adding it, unlike the sparse path and the documented formula.
  r"""Optimizer that implements the Adam algorithm without fused kernels.

  Adam optimization is a stochastic gradient descent method that is based on
  adaptive estimation of first-order and second-order moments.
  According to the paper
  [Adam: A Method for Stochastic Optimization. Kingma et al.,
  2014](http://arxiv.org/abs/1412.6980), the method is "*computationally
  efficient, has little memory requirement, invariant to diagonal rescaling of
  gradients, and is well suited for problems that are large in terms of
  data/parameters*".

  For AMSGrad see [On The Convergence Of Adam And Beyond.
  Reddi et al., 5-8](https://openreview.net/pdf?id=ryQu7f-RZ).

  **If amsgrad = False**:

  initialize $m_0$ as 1st moment vector
  initialize $v_0$ as 2nd moment vector

  The update rule for $\theta$ with gradient $g$ uses an optimization
  described at the end of section 2 of the paper:

  $$lr_t = \mathrm{learning\_rate} *
    \sqrt{1 - \beta_2^t} / (1 - \beta_1^t)$$
  $$m_t = \beta_1 * m_{t-1} + (1 - \beta_1) * g$$
  $$v_t = \beta_2 * v_{t-1} + (1 - \beta_2) * g^2$$
  $$\theta_t = \theta_{t-1} - lr_t * m_t / (\sqrt{v_t} + \epsilon)$$

  **If amsgrad = True**:

  initialize $m_0$ as 1st moment vector
  initialize $v_0$ as 2nd moment vector
  initialize $\hat{v}_0$ as 2nd moment vector

  The update rule for $\theta$ with gradient $g$ uses an optimization
  described at the end of section 2 of the paper:

  $$lr_t = \mathrm{learning\_rate} *
    \sqrt{1 - \beta_2^t} / (1 - \beta_1^t)$$

  $$m_t = \beta_1 * m_{t-1} + (1 - \beta_1) * g$$
  $$v_t = \beta_2 * v_{t-1} + (1 - \beta_2) * g^2$$
  $$\hat{v}_t = \max(\hat{v}_{t-1}, v_t)$$
  $$\theta_t = \theta_{t-1} - lr_t * m_t / (\sqrt{\hat{v}_t} + \epsilon)$$

  The default value of 1e-7 for epsilon might not be a good default in
  general. For example, when training an Inception network on ImageNet a
  current good choice is 1.0 or 0.1. Note that since Adam uses the
  formulation just before Section 2.1 of the Kingma and Ba paper rather than
  the formulation in Algorithm 1, the "epsilon" referred to here is "epsilon
  hat" in the paper.

  The sparse implementation of this algorithm (used when the gradient is an
  IndexedSlices object, typically because of `tf.gather` or an embedding
  lookup in the forward pass) does apply momentum to variable slices even if
  they were not used in the forward pass (meaning they have a gradient equal
  to zero). Momentum decay (beta1) is also applied to the entire momentum
  accumulator. This means that the sparse behavior is equivalent to the dense
  behavior (in contrast to some momentum implementations which ignore momentum
  unless a variable slice was actually used).

  Usage:

  >>> opt = tf.keras.optimizers.Adam(learning_rate=0.1)
  >>> var1 = tf.Variable(10.0)
  >>> loss = lambda: (var1 ** 2)/2.0       # d(loss)/d(var1) == var1
  >>> step_count = opt.minimize(loss, [var1]).numpy()
  >>> # The first step is `-learning_rate*sign(grad)`
  >>> var1.numpy()
  9.9
  """

  _HAS_AGGREGATE_GRAD = True

  def __init__(self,
               learning_rate=0.001,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-7,
               amsgrad=False,
               name='Adam',
               **kwargs):
    # [FUNCTION DOC]
    # Constructs the non-fused optimizer. Inputs: learning_rate (number, tensor, schedule or
    # callable, default 0.001; the legacy keyword "lr" in the extra keyword arguments takes
    # precedence), beta_1 and beta_2 (moment decay rates, defaults 0.9 and 0.999), epsilon
    # (stability constant, default 1e-7; a false-like value falls back to the backend default),
    # amsgrad (use the AMSGrad variant), name (default "Adam") and extra keyword arguments
    # forwarded to the base class. Registers hyperparameters, stores epsilon and amsgrad.
    # Returns nothing.
    """Construct a new Adam optimizer.

    Args:
      learning_rate: A `Tensor`, floating point value, or a schedule that is a
        `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable that
        takes no arguments and returns the actual value to use, The learning
        rate. Defaults to 0.001.
      beta_1: A float value or a constant float tensor, or a callable that takes
        no arguments and returns the actual value to use. The exponential decay
        rate for the 1st moment estimates. Defaults to 0.9.
      beta_2: A float value or a constant float tensor, or a callable that takes
        no arguments and returns the actual value to use, The exponential decay
        rate for the 2nd moment estimates. Defaults to 0.999.
      epsilon: A small constant for numerical stability. This epsilon is
        "epsilon hat" in the Kingma and Ba paper (in the formula just before
        Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to
        1e-7.
      amsgrad: Boolean. Whether to apply AMSGrad variant of this algorithm from
        the paper "On the Convergence of Adam and beyond". Defaults to `False`.
      name: Optional name for the operations created when applying gradients.
        Defaults to "Adam".
      **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
        `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip
        gradients by value, `decay` is included for backward compatibility to
        allow time inverse decay of learning rate. `lr` is included for backward
        compatibility, recommended to use `learning_rate` instead.
    """

    super(NonFusedAdam, self).__init__(name, **kwargs)
    self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
    self._set_hyper('decay', self._initial_decay)
    self._set_hyper('beta_1', beta_1)
    self._set_hyper('beta_2', beta_2)
    self.epsilon = epsilon or backend_config.epsilon()
    self.amsgrad = amsgrad

  def _create_slots(self, var_list):
    # [FUNCTION DOC]
    # HELPER: slot setup hook called by the optimizer base class before applying gradients.
    # Creates first moment slots for all variables in var_list, then second moment slots, and
    # when amsgrad is enabled a running-maximum second moment slot. Returns nothing; side
    # effect is creating slot variables.
    # Create slots for the first and second moments.
    # Separate for-loops to respect the ordering of slot variables from v1.
    for var in var_list:
      self.add_slot(var, 'm')
    for var in var_list:
      self.add_slot(var, 'v')
    if self.amsgrad:
      for var in var_list:
        self.add_slot(var, 'vhat')

  def _prepare_local(self, var_device, var_dtype, apply_state):
    # [FUNCTION DOC]
    # HELPER: per-device, per-dtype coefficient precomputation for the update step.
    # Calls the base implementation, then computes the step count, beta values and powers, the
    # bias-corrected learning rate, epsilon tensor and one-minus-beta terms, merging them into
    # the apply_state entry keyed by (device, dtype). Returns nothing; mutates apply_state.
    super(NonFusedAdam, self)._prepare_local(var_device, var_dtype, apply_state)

    local_step = math_ops.cast(self.iterations + 1, var_dtype)
    beta_1_t = array_ops.identity(self._get_hyper('beta_1', var_dtype))
    beta_2_t = array_ops.identity(self._get_hyper('beta_2', var_dtype))
    beta_1_power = math_ops.pow(beta_1_t, local_step)
    beta_2_power = math_ops.pow(beta_2_t, local_step)
    lr = (
        apply_state[(var_device, var_dtype)]['lr_t'] *
        (math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)))
    apply_state[(var_device, var_dtype)].update(
        dict(
            lr=lr,
            epsilon=tensor_conversion.convert_to_tensor_v2_with_dispatch(
                self.epsilon, var_dtype
            ),
            beta_1_t=beta_1_t,
            beta_1_power=beta_1_power,
            one_minus_beta_1_t=1 - beta_1_t,
            beta_2_t=beta_2_t,
            beta_2_power=beta_2_power,
            one_minus_beta_2_t=1 - beta_2_t,
        )
    )

  def set_weights(self, weights):
    # [FUNCTION DOC]
    # Restores optimizer state from a list of arrays in weight order. A list with the Keras v1
    # length (three slots per variable plus the counter) is truncated to the current weight
    # count to drop the extra entries, then passed to the base class, which raises a value
    # error on mismatch. Returns nothing.
    params = self.weights
    # If the weights are generated by Keras V1 optimizer, it includes vhats
    # even without amsgrad, i.e, V1 optimizer has 3x + 1 variables, while V2
    # optimizer has 2x + 1 variables. Filter vhats out for compatibility.
    num_vars = int((len(params) - 1) / 2)
    if len(weights) == 3 * num_vars + 1:
      weights = weights[:len(params)]
    super(NonFusedAdam, self).set_weights(weights)

  @def_function.function(jit_compile=True)
  def _resource_apply_dense(self, grad, var, apply_state=None):
    # [FUNCTION DOC]
    # HELPER: dense update step (compiled with XLA) called by the optimizer base class.
    # KNOWN ISSUE: the final variable update divides by the square root of the second moment
    # MINUS epsilon, whereas the documented rule and the sparse path add epsilon; this can
    # divide by zero or a negative number. It also returns nothing, unlike the fused class.
    # Inputs: grad (dense gradient), var (variable), apply_state (optional coefficient
    # mapping). Updates the first and second moment slots by exponential moving averages, with
    # AMSGrad takes the running maximum of the second moment, and subtracts the bias-corrected
    # scaled step from the variable in place.
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype)) or
                    self._fallback_apply_state(var_device, var_dtype))

    m = self.get_slot(var, 'm')
    v = self.get_slot(var, 'v')

    alpha = (
        coefficients['lr_t'] * math_ops.sqrt(1 - coefficients['beta_2_power']) /
        (1 - coefficients['beta_1_power']))
    m.assign_add((grad - m) * (1 - coefficients['beta_1_t']))
    v.assign_add((math_ops.square(grad) - v) * (1 - coefficients['beta_2_t']))
    if self.amsgrad:
      vhat = self.get_slot(var, 'vhat')
      vhat.assign(math_ops.maximum(vhat, v))
      v = vhat
    var.assign_sub(
        (m * alpha) / (math_ops.sqrt(v) - coefficients['epsilon']))

  @def_function.function(jit_compile=True)
  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    # [FUNCTION DOC]
    # HELPER: sparse update step (compiled with XLA) called by the optimizer base class.
    # Inputs: grad (gradient row values), var (variable), indices (row indices of the
    # gradient), apply_state (optional coefficient mapping). Decays the full moment slots and
    # scatter-adds the scaled gradient and squared gradient at the indices, then subtracts the
    # bias-corrected step from the variable (using the running maximum second moment with
    # AMSGrad). Updates in place and returns nothing.
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype)) or
                    self._fallback_apply_state(var_device, var_dtype))

    # m_t = beta1 * m + (1 - beta1) * g_t
    m = self.get_slot(var, 'm')
    m_scaled_g_values = grad * coefficients['one_minus_beta_1_t']
    m.assign(m * coefficients['beta_1_t'])
    m.scatter_add(indexed_slices.IndexedSlices(m_scaled_g_values, indices))

    # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
    v = self.get_slot(var, 'v')
    v_scaled_g_values = (grad * grad) * coefficients['one_minus_beta_2_t']
    v.assign(v * coefficients['beta_2_t'])
    v.scatter_add(indexed_slices.IndexedSlices(v_scaled_g_values, indices))

    if not self.amsgrad:
      var.assign_sub(coefficients['lr'] * m /
                     (math_ops.sqrt(v) + coefficients['epsilon']))
    else:
      v_hat = self.get_slot(var, 'vhat')
      v_hat.assign(math_ops.maximum(v_hat, v))
      var.assign_sub(coefficients['lr'] * m /
                     (math_ops.sqrt(v_hat) + coefficients['epsilon']))

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a serializable mapping of the base configuration extended with the current
    # learning rate, decay, beta_1, beta_2, epsilon and amsgrad flag. No side effects.
    config = super(NonFusedAdam, self).get_config()
    config.update({
        'learning_rate': self._serialize_hyperparameter('learning_rate'),
        'decay': self._initial_decay,
        'beta_1': self._serialize_hyperparameter('beta_1'),
        'beta_2': self._serialize_hyperparameter('beta_2'),
        'epsilon': self.epsilon,
        'amsgrad': self.amsgrad,
    })
    return config
