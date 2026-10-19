# [MODULE DOC]
# Implements the Adagrad optimizer for the legacy Keras OptimizerV2 framework. Exposes the
# Adagrad class, which keeps one accumulator slot per variable initialized to a configurable
# value and applies updates through the fused training kernels for dense and sparse gradients.
# Part of the optimizer_v2 package; no import-time side effects.
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
"""Adagrad optimizer implementation."""
# pylint: disable=g-classes-have-attributes

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.keras import backend_config
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_training_ops
from tensorflow.python.ops import init_ops


class Adagrad(optimizer_v2.OptimizerV2):
  # [CLASS DOC]
  # Keras OptimizerV2 implementing Adagrad. Hyperparameters: learning rate and decay (tracked)
  # plus the initial accumulator value and epsilon (plain attributes). State: one accumulator
  # slot per trained variable, filled with the initial accumulator value. Overrides slot
  # creation, coefficient preparation, dense and sparse updates, weight setting, configuration
  # serialization and deserialization. Declares support for aggregated gradients.
  r"""Optimizer that implements the Adagrad algorithm.

  Adagrad is an optimizer with parameter-specific learning rates,
  which are adapted relative to how frequently a parameter gets
  updated during training. The more updates a parameter receives,
  the smaller the updates.

  Args:
    learning_rate: Initial value for the learning rate:
      either a floating point value,
      or a `tf.keras.optimizers.schedules.LearningRateSchedule` instance.
      Defaults to 0.001.
      Note that `Adagrad` tends to benefit from higher initial learning rate
      values compared to other optimizers.
      To match the exact form in the original paper, use 1.0.
    initial_accumulator_value: Floating point value.
      Starting value for the accumulators (per-parameter momentum values).
      Must be non-negative.
    epsilon: Small floating point value used to maintain numerical stability.
    name: Optional name prefix for the operations created when applying
      gradients.  Defaults to `"Adagrad"`.
    **kwargs: Keyword arguments. Allowed to be one of
      `"clipnorm"` or `"clipvalue"`.
      `"clipnorm"` (float) clips gradients by norm and represents
      the maximum L2 norm of each weight variable;
      `"clipvalue"` (float) clips gradient by value and represents the
      maximum absolute value of each weight variable.

  Reference:
    - [Duchi et al., 2011](
      http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf).
  """

  _HAS_AGGREGATE_GRAD = True

  def __init__(self,
               learning_rate=0.001,
               initial_accumulator_value=0.1,
               epsilon=1e-7,
               name='Adagrad',
               **kwargs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: Epsilon is defaulted twice: the explicit nothing check is redundant because
    # the later falsy check also replaces zero, so passing zero silently yields the default.
    # Constructor. Inputs: learning_rate (number or schedule, default 0.001; a legacy lr keyword
    # argument takes precedence), initial_accumulator_value (non-negative number, default 0.1),
    # epsilon (default 1e-7), name (default Adagrad) and keyword arguments (clipnorm, clipvalue
    # and other base options). Raises a value error if the initial accumulator value is
    # negative. Initializes the base optimizer, registers learning rate and decay
    # hyperparameters and stores the accumulator value and epsilon. Returns nothing.
    if initial_accumulator_value < 0.0:
      raise ValueError('initial_accumulator_value must be non-negative: %s' %
                       initial_accumulator_value)
    if epsilon is None:
      epsilon = backend_config.epsilon()
    super(Adagrad, self).__init__(name, **kwargs)
    self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
    self._set_hyper('decay', self._initial_decay)
    self._initial_accumulator_value = initial_accumulator_value
    self.epsilon = epsilon or backend_config.epsilon()

  def _create_slots(self, var_list):
    # [FUNCTION DOC]
    # HELPER: Slot factory called by the base optimizer. Input: list of variables. For each,
    # adds an accumulator slot initialized to the configured initial accumulator value in the
    # variable's base dtype. Returns nothing.
    for var in var_list:
      dtype = var.dtype.base_dtype
      init = init_ops.constant_initializer(
          self._initial_accumulator_value, dtype=dtype)
      self.add_slot(var, 'accumulator', init)

  def _prepare_local(self, var_device, var_dtype, apply_state):
    # [FUNCTION DOC]
    # HELPER: Precomputes per device and dtype coefficients.
    # KNOWN ISSUE: the negated learning rate and zero entries are not used by the update
    # methods in this file.
    # Inputs: var_device, var_dtype and
    # apply_state (mapping keyed by device and dtype, mutated). After the base class fills the
    # entry, adds epsilon as a tensor, the negated learning rate and an int64 zero scalar.
    # Returns nothing.
    super(Adagrad, self)._prepare_local(var_device, var_dtype, apply_state)
    apply_state[(var_device, var_dtype)].update(
        dict(
            epsilon=tensor_conversion.convert_to_tensor_v2_with_dispatch(
                self.epsilon, var_dtype
            ),
            neg_lr_t=-apply_state[(var_device, var_dtype)]['lr_t'],
            zero=array_ops.zeros((), dtype=dtypes.int64),
        )
    )

  def set_weights(self, weights):
    # [FUNCTION DOC]
    # Input: weights, a list of arrays. If the list is exactly one shorter than the optimizer's
    # current weights (a version 1 optimizer list without the iteration count at the head), a
    # zero iteration entry is prepended. Then delegates to the base set_weights, which assigns
    # the values and may raise a value error on a length or shape mismatch. Mutates variables.
    params = self.weights
    # Override set_weights for backward compatibility of Keras V1 optimizer
    # since it does not include iteration at head of the weight list. Set
    # iteration to 0.
    if len(params) == len(weights) + 1:
      weights = [np.array(0)] + weights
    super(Adagrad, self).set_weights(weights)

  @classmethod
  def from_config(cls, config, custom_objects=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The input configuration mapping is modified in place (defaults added, lr key
    # renamed) although the docstring does not mention it; custom_objects is accepted but unused.
    # Class method. Input: config mapping. Adds the initial accumulator value 0.1 if absent
    # (older configurations), renames a legacy lr entry to learning rate, and returns a new
    # instance built from the mapping.
    """Creates an optimizer from its config.

    This method is the reverse of `get_config`,
    capable of instantiating the same optimizer from the config
    dictionary.

    Args:
        config: A Python dictionary, typically the output of get_config.
        custom_objects: A Python dictionary mapping names to additional Python
          objects used to create this optimizer, such as a function used for a
          hyperparameter.

    Returns:
        An optimizer instance.
    """
    if 'initial_accumulator_value' not in config:
      config['initial_accumulator_value'] = 0.1
    if 'lr' in config:
      config['learning_rate'] = config.pop('lr')
    return cls(**config)

  def _resource_apply_dense(self, grad, var, apply_state=None):
    # [FUNCTION DOC]
    # Applies one Adagrad update for a dense gradient. Inputs: grad (tensor), var (resource
    # variable) and apply_state (optional precomputed coefficients; a fallback is computed if
    # missing). Runs the fused training kernel with the accumulator slot, learning rate and
    # epsilon, honoring the locking option. Returns the update operation; mutates the variable
    # and accumulator.
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    acc = self.get_slot(var, 'accumulator')
    return gen_training_ops.ResourceApplyAdagradV2(
        var=var.handle,
        accum=acc.handle,
        lr=coefficients['lr_t'],
        epsilon=coefficients['epsilon'],
        grad=grad,
        use_locking=self._use_locking)

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    # [FUNCTION DOC]
    # Applies one Adagrad update for a sparse gradient. Inputs: grad (values for the updated
    # rows), var, indices (rows of the variable that grad refers to) and apply_state (optional
    # coefficients). Uses the sparse fused kernel with the accumulator slot, learning rate and
    # epsilon. Returns the update operation; mutates the touched rows of variable and slot.
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    acc = self.get_slot(var, 'accumulator')
    return gen_training_ops.ResourceSparseApplyAdagradV2(
        var=var.handle,
        accum=acc.handle,
        lr=coefficients['lr_t'],
        epsilon=coefficients['epsilon'],
        grad=grad,
        indices=indices,
        use_locking=self._use_locking)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the base configuration extended with the serialized learning rate, decay, initial
    # accumulator value and epsilon, for re-creating the optimizer. No side effects.
    config = super(Adagrad, self).get_config()
    config.update({
        'learning_rate': self._serialize_hyperparameter('learning_rate'),
        'decay': self._initial_decay,
        'initial_accumulator_value': self._initial_accumulator_value,
        'epsilon': self.epsilon,
    })
    return config
