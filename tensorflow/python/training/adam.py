# [MODULE DOC]
# Defines the TF1 AdamOptimizer (exported under the v1 train namespace), an optimizer subclass
# implementing the Adam algorithm with first and second moment slots per variable and shared
# beta power accumulators that are updated once per step. Dense updates use generated fused ops;
# sparse updates are composed from assignment and scatter-add operations. Import-time side
# effect: registers the public export of the class.
#
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
"""Adam for TensorFlow."""
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_training_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import optimizer
from tensorflow.python.util.tf_export import tf_export


@tf_export(v1=["train.AdamOptimizer"])
class AdamOptimizer(optimizer.Optimizer):
  # [CLASS DOC]
  # Optimizer implementing Adam. Holds learning rate, beta1, beta2 and epsilon (numbers,
  # tensors or, in eager mode, callables) and their tensor versions created at prepare time.
  # Per variable it keeps zero-initialized moment slots m and v; across variables it keeps two
  # non-slot variables (beta1 power and beta2 power) created on the device of the first variable
  # by name. Overrides the dense and sparse apply hooks and the finish hook which advances the
  # power accumulators.
  """Optimizer that implements the Adam algorithm.

  References:
    Adam - A Method for Stochastic Optimization:
      [Kingma et al., 2015](https://arxiv.org/abs/1412.6980)
      ([pdf](https://arxiv.org/pdf/1412.6980.pdf))

  @compatibility(TF2)
  tf.compat.v1.train.AdamOptimizer is compatible with eager mode and
  `tf.function`.
  When eager execution is enabled, `learning_rate`, `beta1`, `beta2`, and
  `epsilon` can each be a callable that takes no arguments and returns the
  actual value to use. This can be useful for changing these values across
  different invocations of optimizer functions.

  To switch to native TF2 style, use [`tf.keras.optimizers.Adam`]
  (https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam)
  instead. Please notice that due to the implementation differences,
  `tf.keras.optimizers.Adam` and
  `tf.compat.v1.train.AdamOptimizer` may have slight differences in
  floating point numerics even though the formula used for the variable
  updates still matches.

  #### Structural Mapping to Native TF2

  Before:

  ```python
  optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
  ```

  After:

  ```python
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
  ```

  #### How to Map Arguments
  |TF1 Arg Name          |TF2 Arg Name |Note                  |
  |----------------------|-------------|----------------------|
  |learning_rate         |learning_rate|Be careful of setting learning_rate as a
  :                      :             : tensor value computed from the global
  :                      :             : step. In TF1 this was usually meant to
  :                      :             : imply a dynamic learning rate and would
  :                      :             : recompute in each step. In TF2 (eager +
  :                      :             : function) it will treat it as a scalar
  :                      :             : value that only gets computed once
  :                      :             : instead of a symbolic placeholder to be
  :                      :             : computed each time.                   :
  |beta1                 |beta_1        |                      |
  |beta2                 |beta_2        |                      |
  |epsilon               |epsilon      | Default value is 1e-08 in TF1, but
  :                      :             : 1e-07 in TF2.                     :
  |use_locking           |N/A          |Not applicable in TF2. |

  #### Before & After Usage Example
  Before:

  ```python
  x = tf.Variable([1,2,3], dtype=tf.float32)
  grad = tf.constant([0.1, 0.2, 0.3])
  optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
  optimizer.apply_gradients(zip([grad], [x]))
  ```

  After:

  ```python
  x = tf.Variable([1,2,3], dtype=tf.float32)
  grad = tf.constant([0.1, 0.2, 0.3])
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
  optimizer.apply_gradients(zip([grad], [x]))
  ```

  @end_compatibility
  """

  def __init__(
      self,
      learning_rate=0.001,
      beta1=0.9,
      beta2=0.999,
      epsilon=1e-8,
      use_locking=False,
      name="Adam",
      use_own_namescope_for_non_slot_vars=False,
  ):
    # [FUNCTION DOC]
    # Constructor. Inputs: learning_rate (default 0.001), beta1 (default 0.9) and beta2 (default
    # 0.999) as numbers or tensors, epsilon (default 1e-8), use_locking (default false), name
    # (default "Adam"), use_own_namescope_for_non_slot_vars (default false; passed to the base
    # class to place non-slot variables in a root name scope). Stores the hyperparameters;
    # tensor copies start empty and are filled by the prepare hook. Returns nothing.
    r"""Construct a new Adam optimizer.

    Initialization:

    $$m_0 := 0 \text{(Initialize initial 1st moment vector)}$$
    $$v_0 := 0 \text{(Initialize initial 2nd moment vector)}$$
    $$t := 0 \text{(Initialize timestep)}$$

    The update rule for `variable` with gradient `g` uses an optimization
    described at the end of section 2 of the paper:

    $$t := t + 1$$
    $$\text{lr}_t := \mathrm{learning_rate} *
      \sqrt{1 - \beta_2^t} / (1 - \beta_1^t)$$

    $$m_t := \beta_1 * m_{t-1} + (1 - \beta_1) * g$$
    $$v_t := \beta_2 * v_{t-1} + (1 - \beta_2) * g * g$$
    $$\text{variable} := \text{variable} -
      \text{lr}_t * m_t / (\sqrt{v_t} + \epsilon)$$

    The default value of 1e-8 for epsilon might not be a good default in
    general. For example, when training an Inception network on ImageNet a
    current good choice is 1.0 or 0.1. Note that since AdamOptimizer uses the
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

    Args:
      learning_rate: A Tensor or a floating point value.  The learning rate.
      beta1: A float value or a constant float tensor. The exponential decay
        rate for the 1st moment estimates.
      beta2: A float value or a constant float tensor. The exponential decay
        rate for the 2nd moment estimates.
      epsilon: A small constant for numerical stability. This epsilon is
        "epsilon hat" in the Kingma and Ba paper (in the formula just before
        Section 2.1), not the epsilon in Algorithm 1 of the paper.
      use_locking: If True use locks for update operations.
      name: Optional name for the operations created when applying gradients.
        Defaults to "Adam".
      use_own_namescope_for_non_slot_vars: If True, use a root namescope for
        non-slot variables.
    """

    super().__init__(use_locking, name, use_own_namescope_for_non_slot_vars)
    self._lr = learning_rate
    self._beta1 = beta1
    self._beta2 = beta2
    self._epsilon = epsilon

    # Tensor versions of the constructor arguments, created in _prepare().
    self._lr_t = None
    self._beta1_t = None
    self._beta2_t = None
    self._epsilon_t = None

  def _get_beta_accumulators(self):
    # [FUNCTION DOC]
    # HELPER: fetches the two shared accumulators. Takes no arguments. Inside the initialization
    # scope, looks up the non-slot variables named beta1 power and beta2 power for the current
    # graph (none in eager mode). Returns a pair of those variables.
    with ops.init_scope():
      if context.executing_eagerly():
        graph = None
      else:
        graph = ops.get_default_graph()
      return (self._get_non_slot_variable("beta1_power", graph=graph),
              self._get_non_slot_variable("beta2_power", graph=graph))

  def _create_slots(self, var_list):
    # [FUNCTION DOC]
    # HELPER: base-class hook creating optimizer state. Input: var_list, variables to update.
    # Creates the beta1 power and beta2 power non-slot variables (initialized to beta1 and beta2)
    # colocated with the variable having the smallest name, so the device is the same across
    # workers; then creates zero-filled slots m and v for every variable. Returns nothing.
    # Create the beta1 and beta2 accumulators on the same device as the first
    # variable. Sort the var_list to make sure this device is consistent across
    # workers (these need to go on the same PS, otherwise some updates are
    # silently ignored).
    first_var = min(var_list, key=lambda x: x.name)
    self._create_non_slot_variable(
        initial_value=self._beta1, name="beta1_power", colocate_with=first_var)
    self._create_non_slot_variable(
        initial_value=self._beta2, name="beta2_power", colocate_with=first_var)

    # Create slots for the first and second moments.
    for v in var_list:
      self._zeros_slot(v, "m", self._name)
      self._zeros_slot(v, "v", self._name)

  def _prepare(self):
    # [FUNCTION DOC]
    # HELPER: base-class hook run before applying gradients. Resolves learning rate, beta1,
    # beta2 and epsilon (calling any callables) and converts them to tensors stored on the
    # optimizer. Returns nothing.
    lr = self._call_if_callable(self._lr)
    beta1 = self._call_if_callable(self._beta1)
    beta2 = self._call_if_callable(self._beta2)
    epsilon = self._call_if_callable(self._epsilon)

    self._lr_t = ops.convert_to_tensor(lr, name="learning_rate")
    self._beta1_t = ops.convert_to_tensor(beta1, name="beta1")
    self._beta2_t = ops.convert_to_tensor(beta2, name="beta2")
    self._epsilon_t = ops.convert_to_tensor(epsilon, name="epsilon")

  def _apply_dense(self, grad, var):
    # [FUNCTION DOC]
    # Dense Adam update for a reference variable. Inputs: grad, dense gradient; var, the
    # variable. Uses its m and v slots, the beta power accumulators and all hyperparameters cast
    # to the variable's dtype, honoring locking. Returns the underlying operation of the fused
    # update (not the output tensor).
    m = self.get_slot(var, "m")
    v = self.get_slot(var, "v")
    beta1_power, beta2_power = self._get_beta_accumulators()
    return gen_training_ops.apply_adam(
        var,
        m,
        v,
        math_ops.cast(beta1_power, var.dtype.base_dtype),
        math_ops.cast(beta2_power, var.dtype.base_dtype),
        math_ops.cast(self._lr_t, var.dtype.base_dtype),
        math_ops.cast(self._beta1_t, var.dtype.base_dtype),
        math_ops.cast(self._beta2_t, var.dtype.base_dtype),
        math_ops.cast(self._epsilon_t, var.dtype.base_dtype),
        grad,
        use_locking=self._use_locking).op

  def _resource_apply_dense(self, grad, var):
    # [FUNCTION DOC]
    # Dense update for a resource variable: uses resource handles of the variable and slots and
    # casts accumulators and hyperparameters to the gradient's dtype. Inputs: grad, var.
    # Returns the update operation.
    m = self.get_slot(var, "m")
    v = self.get_slot(var, "v")
    beta1_power, beta2_power = self._get_beta_accumulators()
    return gen_training_ops.resource_apply_adam(
        var.handle,
        m.handle,
        v.handle,
        math_ops.cast(beta1_power, grad.dtype.base_dtype),
        math_ops.cast(beta2_power, grad.dtype.base_dtype),
        math_ops.cast(self._lr_t, grad.dtype.base_dtype),
        math_ops.cast(self._beta1_t, grad.dtype.base_dtype),
        math_ops.cast(self._beta2_t, grad.dtype.base_dtype),
        math_ops.cast(self._epsilon_t, grad.dtype.base_dtype),
        grad,
        use_locking=self._use_locking)

  def _apply_sparse_shared(self, grad, var, indices, scatter_add):
    # [FUNCTION DOC]
    # HELPER: common sparse update logic for reference and resource variables. Inputs: grad,
    # gradient values for the selected rows; var, the variable; indices, selected row indices;
    # scatter_add, a callable adding values at indices into a variable and returning the
    # updated value. Computes the bias-corrected learning rate from the beta powers, decays the
    # whole first moment by beta1 and scatter-adds the scaled gradient, does the same for the
    # second moment with beta2 and squared gradient, then subtracts the Adam step from the
    # variable. Ordering is enforced with control dependencies. Returns a grouped operation of
    # the variable, m and v updates.
    beta1_power, beta2_power = self._get_beta_accumulators()
    beta1_power = math_ops.cast(beta1_power, var.dtype.base_dtype)
    beta2_power = math_ops.cast(beta2_power, var.dtype.base_dtype)
    lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
    beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
    beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
    epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
    lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))
    # m_t = beta1 * m + (1 - beta1) * g_t
    m = self.get_slot(var, "m")
    m_scaled_g_values = grad * (1 - beta1_t)
    m_t = state_ops.assign(m, m * beta1_t, use_locking=self._use_locking)
    with ops.control_dependencies([m_t]):
      m_t = scatter_add(m, indices, m_scaled_g_values)
    # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
    v = self.get_slot(var, "v")
    v_scaled_g_values = (grad * grad) * (1 - beta2_t)
    v_t = state_ops.assign(v, v * beta2_t, use_locking=self._use_locking)
    with ops.control_dependencies([v_t]):
      v_t = scatter_add(v, indices, v_scaled_g_values)
    v_sqrt = math_ops.sqrt(v_t)
    var_update = state_ops.assign_sub(
        var, lr * m_t / (v_sqrt + epsilon_t), use_locking=self._use_locking)
    return control_flow_ops.group(*[var_update, m_t, v_t])

  def _apply_sparse(self, grad, var):
    # [FUNCTION DOC]
    # Sparse update for a reference variable. Inputs: grad, indexed-slices gradient; var, the
    # variable. Delegates to the shared sparse logic with the gradient values and indices and a
    # scatter-add that honors the locking setting. Returns that operation.
    return self._apply_sparse_shared(
        grad.values,
        var,
        grad.indices,
        lambda x, i, v: state_ops.scatter_add(  # pylint: disable=g-long-lambda
            x,
            i,
            v,
            use_locking=self._use_locking))

  def _resource_scatter_add(self, x, i, v):
    # [FUNCTION DOC]
    # HELPER: scatter-add for resource variables. Inputs: x, resource variable; i, indices;
    # v, values to add. Performs the resource scatter add and, after it completes (control
    # dependency), returns the variable's current value.
    with ops.control_dependencies(
        [resource_variable_ops.resource_scatter_add(x.handle, i, v)]):
      return x.value()

  def _resource_apply_sparse(self, grad, var, indices):
    # [FUNCTION DOC]
    # Sparse update for a resource variable. Inputs: grad, gradient values; var, resource
    # variable; indices, row indices. Delegates to the shared sparse logic using the resource
    # scatter-add helper. Returns that operation.
    return self._apply_sparse_shared(grad, var, indices,
                                     self._resource_scatter_add)

  def _finish(self, update_ops, name_scope):
    # [FUNCTION DOC]
    # Base-class hook run after all variable updates. Inputs: update_ops, list of update
    # operations for the variables; name_scope, name for the final grouped operation. After
    # all updates complete (control dependency), multiplies the beta1 power and beta2 power
    # accumulators by beta1 and beta2 (colocated, honoring locking). Returns one grouped
    # operation covering the updates and both accumulator updates.
    # Update the power accumulators.
    with ops.control_dependencies(update_ops):
      beta1_power, beta2_power = self._get_beta_accumulators()
      with ops.colocate_with(beta1_power):
        update_beta1 = beta1_power.assign(
            beta1_power * self._beta1_t, use_locking=self._use_locking)
        update_beta2 = beta2_power.assign(
            beta2_power * self._beta2_t, use_locking=self._use_locking)
    return control_flow_ops.group(
        *update_ops + [update_beta1, update_beta2], name=name_scope)
