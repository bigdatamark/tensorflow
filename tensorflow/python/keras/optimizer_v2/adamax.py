# [MODULE DOC]
# Defines the legacy Keras (v2 optimizer API) Adamax optimizer, a variant of Adam based on the
# infinity norm of the gradients. It builds on the generic optimizer base class in the
# optimizer_v2 package and uses fused training kernels for dense updates. Import has no side
# effects beyond defining the class.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Adamax optimizer implementation."""
# pylint: disable=g-classes-have-attributes

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.keras import backend_config
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_training_ops
from tensorflow.python.ops import math_ops


class Adamax(optimizer_v2.OptimizerV2):
  # [CLASS DOC]
  # Adamax optimizer: keeps a first moment estimate and an exponentially weighted infinity norm
  # per variable (two slots, m and v) and scales the step by the bias-corrected learning rate.
  # Dense gradients use a fused kernel; sparse gradients only update the rows present in the
  # gradient. Holds hyperparameters learning rate, decay, beta_1, beta_2 and an epsilon constant.
  # Created by user code and driven by the base class minimize / apply-gradients flow.
  """Optimizer that implements the Adamax algorithm.

  It is a variant of Adam based on the infinity norm.
  Default parameters follow those provided in the paper.
  Adamax is sometimes superior to adam, specially in models with embeddings.

  Initialization:

  ```python
  m = 0  # Initialize initial 1st moment vector
  v = 0  # Initialize the exponentially weighted infinity norm
  t = 0  # Initialize timestep
  ```

  The update rule for parameter `w` with gradient `g` is
  described at the end of section 7.1 of the paper:

  ```python
  t += 1
  m = beta1 * m + (1 - beta) * g
  v = max(beta2 * v, abs(g))
  current_lr = learning_rate / (1 - beta1 ** t)
  w = w - current_lr * m / (v + epsilon)
  ```

  Similarly to `Adam`, the epsilon is added for numerical stability
  (especially to get rid of division by zero when `v_t == 0`).

  In contrast to `Adam`, the sparse implementation of this algorithm
  (used when the gradient is an IndexedSlices object, typically because of
  `tf.gather` or an embedding lookup in the forward pass) only updates
  variable slices and corresponding `m_t`, `v_t` terms when that part of
  the variable was used in the forward pass. This means that the sparse
  behavior is contrast to the dense behavior (similar to some momentum
  implementations which ignore momentum unless a variable slice was actually
  used).

  Args:
    learning_rate: A `Tensor`, floating point value, or a schedule that is a
      `tf.keras.optimizers.schedules.LearningRateSchedule`. The learning rate.
    beta_1: A float value or a constant float tensor. The exponential decay
      rate for the 1st moment estimates.
    beta_2: A float value or a constant float tensor. The exponential decay
      rate for the exponentially weighted infinity norm.
    epsilon: A small constant for numerical stability.
    name: Optional name for the operations created when applying gradients.
      Defaults to `"Adamax"`.
    **kwargs: Keyword arguments. Allowed to be one of
      `"clipnorm"` or `"clipvalue"`.
      `"clipnorm"` (float) clips gradients by norm; `"clipvalue"` (float) clips
      gradients by value.

  Reference:
    - [Kingma et al., 2014](http://arxiv.org/abs/1412.6980)
  """

  _HAS_AGGREGATE_GRAD = True

  def __init__(self,
               learning_rate=0.001,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-7,
               name='Adamax',
               **kwargs):
    # [FUNCTION DOC]
    # Constructs the optimizer. Inputs: learning_rate (number, tensor or schedule, default
    # 0.001; the legacy keyword "lr" in the extra keyword arguments takes precedence), beta_1
    # (first moment decay, default 0.9), beta_2 (infinity norm decay, default 0.999), epsilon
    # (stability constant, default 1e-7; a false-like value falls back to the backend default),
    # name (default "Adamax") and extra keyword arguments such as clipnorm and clipvalue
    # forwarded to the base class. Registers hyperparameters and stores epsilon. Returns nothing.
    super(Adamax, self).__init__(name, **kwargs)
    self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
    self._set_hyper('decay', self._initial_decay)
    self._set_hyper('beta_1', beta_1)
    self._set_hyper('beta_2', beta_2)
    self.epsilon = epsilon or backend_config.epsilon()

  def _create_slots(self, var_list):
    # [FUNCTION DOC]
    # HELPER: slot setup hook called by the optimizer base class. Creates a first moment slot
    # for every variable in var_list, then a second moment (infinity norm) slot for each, in
    # separate loops to keep the v1 ordering. Returns nothing; creates slot variables.
    # Separate for-loops to respect the ordering of slot variables from v1.
    for var in var_list:
      self.add_slot(var, 'm')  # Create slots for the first moments.
    for var in var_list:
      self.add_slot(var, 'v')  # Create slots for the second moments.

  def _prepare_local(self, var_device, var_dtype, apply_state):
    # [FUNCTION DOC]
    # HELPER: per-device, per-dtype coefficient precomputation. Calls the base implementation,
    # then computes the 1-based step, beta values, beta_1 power, the negated bias-corrected
    # learning rate, epsilon tensor, one-minus-beta_1 and a zero integer constant (used as the
    # gather axis), merging them into the apply_state entry keyed by (device, dtype). Returns
    # nothing; mutates apply_state.
    super(Adamax, self)._prepare_local(var_device, var_dtype, apply_state)

    local_step = math_ops.cast(self.iterations + 1, var_dtype)
    beta_1_t = array_ops.identity(self._get_hyper('beta_1', var_dtype))
    beta_2_t = array_ops.identity(self._get_hyper('beta_2', var_dtype))
    beta_1_power = math_ops.pow(beta_1_t, local_step)
    lr_t = apply_state[(var_device, var_dtype)]['lr_t']

    apply_state[(var_device, var_dtype)].update(
        dict(
            neg_scaled_lr=-lr_t / (1 - beta_1_power),
            epsilon=tensor_conversion.convert_to_tensor_v2_with_dispatch(
                self.epsilon, var_dtype
            ),
            beta_1_t=beta_1_t,
            beta_1_power=beta_1_power,
            one_minus_beta_1_t=1 - beta_1_t,
            beta_2_t=beta_2_t,
            zero=array_ops.zeros((), dtype=dtypes.int64),
        )
    )

  def _resource_apply_dense(self, grad, var, apply_state=None):
    # [FUNCTION DOC]
    # HELPER: dense update step called by the optimizer base class. Inputs: grad (dense
    # gradient), var (resource variable), apply_state (optional coefficient mapping; a
    # fallback is computed if missing). Runs the fused Adamax kernel on the variable and its m
    # and v slots, honoring the locking setting. Returns the update operation.
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    m = self.get_slot(var, 'm')
    v = self.get_slot(var, 'v')
    return gen_training_ops.ResourceApplyAdaMax(
        var=var.handle,
        m=m.handle,
        v=v.handle,
        beta1_power=coefficients['beta_1_power'],
        lr=coefficients['lr_t'],
        beta1=coefficients['beta_1_t'],
        beta2=coefficients['beta_2_t'],
        epsilon=coefficients['epsilon'],
        grad=grad,
        use_locking=self._use_locking)

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    # [FUNCTION DOC]
    # HELPER: sparse update step called by the optimizer base class. Inputs: grad (gradient
    # row values), var (variable), indices (rows the values belong to), apply_state (optional
    # coefficient mapping). Gathers the affected rows of m and v, computes the new first moment
    # and the new infinity norm (maximum of decayed value and absolute gradient) for those rows
    # only, writes them back by scatter update, then scatter-adds the scaled step to the
    # variable. Unused rows are untouched. Returns a grouped operation of the three updates.
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    # m_t = beta1 * m + (1 - beta1) * g_t
    m = self.get_slot(var, 'm')
    m_slice = array_ops.gather(m, indices, axis=coefficients['zero'])
    m_t_slice = (m_slice * coefficients['beta_1_t'] +
                 grad * coefficients['one_minus_beta_1_t'])
    with ops.control_dependencies([m_t_slice]):
      m_t = self._resource_scatter_update(m, indices, m_t_slice)

    # u_t = max(beta2 * u, abs(g_t))
    v = self.get_slot(var, 'v')
    v_slice = array_ops.gather(v, indices, axis=coefficients['zero'])
    v_t_slice = math_ops.maximum(v_slice * coefficients['beta_2_t'],
                                 math_ops.abs(grad))
    with ops.control_dependencies([v_t_slice]):
      v_t = self._resource_scatter_update(v, indices, v_t_slice)
    # theta_t = theta - lr / (1 - beta1^t) * m_t / u_t
    var_slice = coefficients['neg_scaled_lr'] * (
        m_t_slice / (v_t_slice + coefficients['epsilon']))
    with ops.control_dependencies([var_slice]):
      var_update = self._resource_scatter_add(var, indices, var_slice)
    return control_flow_ops.group(*[var_update, m_t, v_t])

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a serializable mapping of the base configuration extended with the current
    # learning rate, decay, beta_1, beta_2 and epsilon. No side effects.
    config = super(Adamax, self).get_config()
    config.update({
        'learning_rate': self._serialize_hyperparameter('learning_rate'),
        'decay': self._initial_decay,
        'beta_1': self._serialize_hyperparameter('beta_1'),
        'beta_2': self._serialize_hyperparameter('beta_2'),
        'epsilon': self.epsilon,
    })
    return config
