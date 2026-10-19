# [MODULE DOC]
# Implements the Nadam optimizer (Adam with Nesterov momentum) on top of the base optimizer
# class of this package. Exposes a single class, Nadam. Has no import-time side effects beyond
# imports. Not a helper module.
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
"""Nadam optimizer implementation."""
# pylint: disable=g-classes-have-attributes

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.keras import backend_config
from tensorflow.python.keras.optimizer_v2 import learning_rate_schedule
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables as tf_variables


class Nadam(optimizer_v2.OptimizerV2):
  # [CLASS DOC]
  # Optimizer implementing Nadam: Adam moment estimates combined with Nesterov momentum using a
  # decaying momentum schedule. Holds hyperparameters (learning rate, decay, beta_1, beta_2),
  # epsilon, and a scalar non-trainable momentum-cache variable (the running product of the
  # momentum schedule), plus per-variable slots for first (m) and second (v) moments.
  # Subclasses the base optimizer and overrides slot creation, per-device coefficient
  # preparation, dense and sparse update rules and configuration export. Aggregated gradients
  # are supported.
  r"""Optimizer that implements the NAdam algorithm.
  Much like Adam is essentially RMSprop with momentum, Nadam is Adam with
  Nesterov momentum.

  Args:
    learning_rate: A Tensor or a floating point value.  The learning rate.
    beta_1: A float value or a constant float tensor. The exponential decay
      rate for the 1st moment estimates.
    beta_2: A float value or a constant float tensor. The exponential decay
      rate for the exponentially weighted infinity norm.
    epsilon: A small constant for numerical stability.
    name: Optional name for the operations created when applying gradients.
      Defaults to `"Nadam"`.
    **kwargs: Keyword arguments. Allowed to be one of
      `"clipnorm"` or `"clipvalue"`.
      `"clipnorm"` (float) clips gradients by norm; `"clipvalue"` (float) clips
      gradients by value.

  Usage Example:
    >>> opt = tf.keras.optimizers.Nadam(learning_rate=0.2)
    >>> var1 = tf.Variable(10.0)
    >>> loss = lambda: (var1 ** 2) / 2.0
    >>> step_count = opt.minimize(loss, [var1]).numpy()
    >>> "{:.1f}".format(var1.numpy())
    9.8

  Reference:
    - [Dozat, 2015](http://cs229.stanford.edu/proj2015/054_report.pdf).
  """

  _HAS_AGGREGATE_GRAD = True

  def __init__(self,
               learning_rate=0.001,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-7,
               name='Nadam',
               **kwargs):
    # [FUNCTION DOC]
    # Constructs the optimizer. Inputs: learning rate (default 0.001), beta_1 (0.9), beta_2
    # (0.999), epsilon (1e-7; a null/zero value falls back to the backend default epsilon), name
    # (default "Nadam") and extra keyword arguments (clipnorm, clipvalue, legacy lr alias, and
    # the legacy schedule decay argument which is mapped to the decay setting, default 0.004).
    # Raises a value error if the learning rate is a learning rate schedule object.
    # Side effects: registers hyperparameters with the base class and sets the momentum cache
    # to unset until slots are created.
    # Backwards compatibility with keras NAdam optimizer.
    kwargs['decay'] = kwargs.pop('schedule_decay', 0.004)
    learning_rate = kwargs.get('lr', learning_rate)
    if isinstance(learning_rate, learning_rate_schedule.LearningRateSchedule):
      raise ValueError('The Nadam optimizer does not support '
                       'tf.keras.optimizers.LearningRateSchedules as the '
                       'learning rate.')

    super(Nadam, self).__init__(name, **kwargs)
    self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
    self._set_hyper('decay', self._initial_decay)
    self._set_hyper('beta_1', beta_1)
    self._set_hyper('beta_2', beta_2)
    self.epsilon = epsilon or backend_config.epsilon()
    self._m_cache = None

  def _create_slots(self, var_list):
    # [FUNCTION DOC]
    # HELPER: slot setup hook called by the base optimizer before applying gradients.
    # Creates, once, the scalar non-trainable momentum cache weight (initialised to one, using
    # the dtype of the first variable, aggregated from the first replica only) and appends it to
    # the optimizer weights. Then creates a first-moment slot and a second-moment slot for every
    # variable in the given list (in two separate passes to keep legacy ordering).
    var_dtype = var_list[0].dtype.base_dtype
    if self._m_cache is None:
      self._m_cache = self.add_weight(
          'momentum_cache',
          shape=[],
          dtype=var_dtype,
          initializer='ones',
          trainable=False,
          aggregation=tf_variables.VariableAggregation.ONLY_FIRST_REPLICA)
      self._weights.append(self._m_cache)
    # Separate for-loops to respect the ordering of slot variables from v1.
    for var in var_list:
      # Create slots for the first moments.
      self.add_slot(var, 'm')
    for var in var_list:
      # Create slots for the second moments.
      self.add_slot(var, 'v')

  def _prepare_local(self, var_device, var_dtype, apply_state):
    # [FUNCTION DOC]
    # HELPER: computes per-device, per-dtype update coefficients for Nadam.
    # Reads learning rate, beta_1 and beta_2, computes the current and next momentum values from
    # a 0.96-based decay over the step count, updates the momentum cache (running product of the
    # momentum schedule) when the dtype matches the cache dtype, and stores a mapping of
    # coefficients (rate, negated rate, epsilon, betas and their complements, momentum terms,
    # schedule complements, bias-correction denominator) into the apply-state mapping under the
    # device and dtype key. Mutates the apply-state argument; returns nothing.
    # Side effect: assigns to the momentum cache variable.
    lr_t = array_ops.identity(self._get_hyper('learning_rate', var_dtype))
    beta_1_t = array_ops.identity(self._get_hyper('beta_1', var_dtype))
    beta_2_t = array_ops.identity(self._get_hyper('beta_2', var_dtype))
    local_step = math_ops.cast(self.iterations + 1, var_dtype)
    next_step = math_ops.cast(self.iterations + 2, var_dtype)

    decay_base = math_ops.cast(0.96, var_dtype)

    m_t = beta_1_t * (1. - 0.5 * (
        math_ops.pow(decay_base, self._initial_decay * local_step)))
    m_t_1 = beta_1_t * (1. - 0.5 * (
        math_ops.pow(decay_base, self._initial_decay * next_step)))

    m_schedule_new = math_ops.cast(self._m_cache_read, var_dtype) * m_t
    if var_dtype is self._m_cache.dtype:
      m_schedule_new = array_ops.identity(state_ops.assign(
          self._m_cache, m_schedule_new, use_locking=self._use_locking))
    m_schedule_next = m_schedule_new * m_t_1

    apply_state[(var_device, var_dtype)] = dict(
        lr_t=lr_t,
        neg_lr_t=-lr_t,  # pylint: disable=invalid-unary-operand-type
        epsilon=tensor_conversion.convert_to_tensor_v2_with_dispatch(
            self.epsilon, var_dtype
        ),
        beta_1_t=beta_1_t,
        beta_2_t=beta_2_t,
        m_t=m_t,
        m_t_1=m_t_1,
        one_minus_beta_1_t=1 - beta_1_t,
        one_minus_beta_2_t=1 - beta_2_t,
        one_minus_m_t=1.0 - m_t,
        one_minus_m_schedule_new=1.0 - m_schedule_new,
        one_minus_m_schedule_next=1.0 - m_schedule_next,
        v_t_prime_denominator=1.0 - math_ops.pow(beta_2_t, local_step),
    )

  def _prepare(self, var_list):
    # [FUNCTION DOC]
    # HELPER: pre-update hook. Snapshots the current momentum cache value into an attribute so
    # all variables in this step use the same value, then delegates to the base class and
    # returns its result. Input: the list of variables being updated.
    # Get the value of the momentum cache before starting to apply gradients.
    self._m_cache_read = array_ops.identity(self._m_cache)
    return super(Nadam, self)._prepare(var_list)

  def _resource_apply_dense(self, grad, var, apply_state=None):
    # [FUNCTION DOC]
    # Dense Nadam update for one resource variable. Inputs: gradient tensor, the variable, and an
    # optional mapping of precomputed coefficients per device and dtype (computed on demand if
    # missing). Updates the first and second moment slots, forms the bias-corrected Nesterov
    # step and assigns the new variable value. Returns the update operation.
    # Side effects: assigns to the m slot, v slot and the variable.
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    m = self.get_slot(var, 'm')
    v = self.get_slot(var, 'v')

    g_prime = grad / coefficients['one_minus_m_schedule_new']
    m_t = (coefficients['beta_1_t'] * m +
           coefficients['one_minus_beta_1_t'] * grad)
    m_t = state_ops.assign(m, m_t, use_locking=self._use_locking)
    m_t_prime = m_t / coefficients['one_minus_m_schedule_next']
    v_t = (coefficients['beta_2_t'] * v +
           coefficients['one_minus_beta_2_t'] * math_ops.square(grad))
    v_t = state_ops.assign(v, v_t, use_locking=self._use_locking)
    v_t_prime = v_t / coefficients['v_t_prime_denominator']
    m_t_bar = (coefficients['one_minus_m_t'] * g_prime +
               coefficients['m_t_1'] * m_t_prime)
    var_t = var - coefficients['lr_t'] * m_t_bar / (
        math_ops.sqrt(v_t_prime) + coefficients['epsilon'])
    return state_ops.assign(var, var_t, use_locking=self._use_locking).op

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    # [FUNCTION DOC]
    # Sparse Nadam update for one resource variable. Inputs: gradient values, the variable, the
    # row indices the gradient applies to, and an optional coefficient mapping (computed on
    # demand if missing). Decays the whole m and v slots, scatter-adds the scaled gradient terms
    # at the indices (ordered with control dependencies), gathers the touched slices, and
    # scatter-adds the Nesterov step into the variable. Returns a grouped operation of the
    # variable update, the step term and the second-moment update.
    # Side effects: modifies m, v and the variable in place.
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    m = self.get_slot(var, 'm')
    v = self.get_slot(var, 'v')

    g_prime = grad / coefficients['one_minus_m_schedule_new']

    # m_t = beta1 * m + (1 - beta1) * g_t
    m_scaled_g_values = grad * coefficients['one_minus_beta_1_t']
    m_t = state_ops.assign(m, m * coefficients['beta_1_t'],
                           use_locking=self._use_locking)

    with ops.control_dependencies([m_t]):
      m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)
      m_t_slice = array_ops.gather(m_t, indices)

    m_t_prime = m_t_slice / coefficients['one_minus_m_schedule_next']
    m_t_bar = (coefficients['one_minus_m_t'] * g_prime +
               coefficients['m_t_1'] * m_t_prime)

    # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
    v_scaled_g_values = (grad * grad) * coefficients['one_minus_beta_2_t']
    v_t = state_ops.assign(v, v * coefficients['beta_2_t'],
                           use_locking=self._use_locking)

    with ops.control_dependencies([v_t]):
      v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)
      v_t_slice = array_ops.gather(v_t, indices)

    v_t_prime = v_t_slice / coefficients['v_t_prime_denominator']
    v_prime_sqrt_plus_eps = math_ops.sqrt(v_t_prime) + coefficients['epsilon']

    var_update = self._resource_scatter_add(
        var, indices,
        coefficients['neg_lr_t'] * m_t_bar / v_prime_sqrt_plus_eps)
    return control_flow_ops.group(*[var_update, m_t_bar, v_t])

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a serializable configuration mapping: the base class configuration extended with
    # the current learning rate, decay, beta_1 and beta_2 values (serialized hyperparameters) and
    # epsilon. No side effects.
    config = super(Nadam, self).get_config()
    config.update({
        'learning_rate': self._serialize_hyperparameter('learning_rate'),
        'decay': self._initial_decay,
        'beta_1': self._serialize_hyperparameter('beta_1'),
        'beta_2': self._serialize_hyperparameter('beta_2'),
        'epsilon': self.epsilon,
    })
    return config
