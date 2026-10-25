# [MODULE DOC]
# Maintains moving averages (exponential decay) of variables, mainly for training/evaluation
# workflows where averaged weights give better results than the last trained values.
# Exposes: assign_moving_average (exported as an internal training symbol), which updates a
# variable toward a new value with optional zero-debiasing and is distribution-strategy aware;
# weighted_moving_average, which keeps two internal moving averages and returns their ratio;
# and the ExponentialMovingAverage class (exported as the public training class) which creates
# shadow variables for a list of variables and updates them on each apply call.
# Private helpers (_update, _zero_debias) route updates through the distribution strategy and
# compute the debiased update amount. Relies on slot_creator for shadow variables and on
# variable scopes for naming. No import-time side effects other than API export registration.
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
"""Maintain moving averages of parameters."""
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import reduce_util as ds_reduce_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.training import slot_creator
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls


@tf_export("__internal__.train.assign_moving_average", v1=[])
def assign_moving_average(variable, value, decay, zero_debias=True, name=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring example calls a top-level assign-moving-average symbol that is
  # not exported under that name (it is exported only as an internal training symbol).
  # Moves the given variable toward the given value by an exponential moving average step.
  # Inputs: variable (a variable), value (tensor of the same shape), decay (float or scalar
  # tensor, the average decay; internally converted to one minus decay and cast to the
  # variable's base dtype), zero_debias (boolean, default true; use debiased update that assumes
  # zero initialization), name (optional op name scope, default "AssignMovingAvg").
  # Behavior: opens a name scope. If called in a replica context, it merges across replicas,
  # reducing the value by mean to the variable, then applies the update in cross-replica
  # context; otherwise it updates directly using the cross-replica strategy. The update is
  # either the zero-debiased update or a plain subtract-assign of (variable - value) * decay.
  # Returns the update result (tensor/op that when evaluated performs the update).
  # Side effects: may create shadow variables (biased value and local step) when debiasing.
  # Raises nothing itself; errors from conversion or strategy calls propagate.
  """Compute the moving average of a variable.

  The moving average of 'variable' updated with 'value' is:
    variable * decay + value * (1 - decay)

  The returned Operation sets 'variable' to the newly computed moving average,
  by performing this subtraction:
     variable -= (1 - decay) * (variable - value)

  Since variables that are initialized to a `0` value will be `0` biased,
  `zero_debias` optionally enables scaling by the mathematically correct
  debiasing factor of
    1 - decay ** num_updates
  See Section 3 of (Kingma et al., 2015) for more details.

  The names of the debias shadow variables, by default, include both the scope
  they were created in and the scope of the variables they debias. They are also
  given a uniquifying-suffix.

  E.g.:

  ```
    with tf.compat.v1.variable_scope('scope1'):
      with tf.compat.v1.variable_scope('scope2'):
        var = tf.compat.v1.get_variable('foo')
        update_1 = tf.assign_moving_average(var, 0.0, 1.0)
        update_2 = tf.assign_moving_average(var, 0.0, 0.9)

    # var.name: 'scope1/scope2/foo'
    # shadow var names: 'scope1/scope2/scope1/scope2/foo/biased'
    #                   'scope1/scope2/scope1/scope2/foo/biased_1'
  ```

  Args:
    variable: A Variable.
    value: A tensor with the same shape as 'variable'.
    decay: A float `Tensor` or float value. The moving average decay.
    zero_debias: A python bool. If true, assume the variable is 0-initialized
      and unbias it, as in (Kingma et al., 2015). See docstring in
        `_zero_debias` for more details.
    name: Optional name of the returned operation.

  Returns:
    A tensor which if evaluated will compute and return the new moving average.

  References:
    Adam - A Method for Stochastic Optimization:
      [Kingma et al., 2015](https://arxiv.org/abs/1412.6980)
      ([pdf](https://arxiv.org/pdf/1412.6980.pdf))
  """
  with ops.name_scope(name, "AssignMovingAvg",
                      [variable, value, decay]) as scope:
    decay = ops.convert_to_tensor(1.0 - decay, name="decay")
    if decay.dtype != variable.dtype.base_dtype:
      decay = math_ops.cast(decay, variable.dtype.base_dtype)

    def update_fn(v, value):
      # [FUNCTION DOC]
      # HELPER: nested update callback for assign_moving_average.
      # Subtracts (v - value) * decay from variable v in place, using the enclosing name scope
      # as the op name, and returns the assign-subtract result. Uses the converted decay captured
      # from the enclosing function.
      return state_ops.assign_sub(v, (v - value) * decay, name=scope)

    def update(strategy, v, value):
      # [FUNCTION DOC]
      # HELPER: nested dispatcher for assign_moving_average.
      # Given a strategy, variable and (already reduced) value, returns the zero-debiased update
      # if zero_debias was requested, otherwise applies the plain update callback through the
      # strategy-aware update helper. Returns the resulting update.
      if zero_debias:
        return _zero_debias(strategy, v, value, decay)
      else:
        return _update(strategy, v, update_fn, args=(value,))

    replica_context = distribute_lib.get_replica_context()
    if replica_context:
      # In a replica context, we update variable using the mean of value across
      # replicas.
      def merge_fn(strategy, v, value):
        # [FUNCTION DOC]
        # HELPER: nested merge callback for assign_moving_average in a replica context.
        # Reduces the per-replica value to the variable's location using the mean reduction,
        # then performs the update in cross-replica context and returns its result.
        value = strategy.extended.reduce_to(ds_reduce_util.ReduceOp.MEAN, value,
                                            v)
        return update(strategy, v, value)

      return replica_context.merge_call(merge_fn, args=(variable, value))
    else:
      strategy = distribute_lib.get_cross_replica_context()
      return update(strategy, variable, value)


def weighted_moving_average(value,
                            decay,
                            weight,
                            truediv=True,
                            collections=None,
                            name=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says it returns an Operation, but it actually returns the
  # tensor produced by dividing the two moving averages (the weighted average value).
  # Computes a weighted moving average: the moving average of value times weight divided by
  # the moving average of weight. Inputs: value (numeric tensor), decay (float or scalar
  # tensor), weight (tensor broadcastable against value), truediv (boolean, default true:
  # floating point division, otherwise dtype-implied division), collections (list of graph
  # collection keys for the internal variables, default the global variables collection),
  # name (optional scope name, default "WeightedMovingAvg").
  # Side effects: creates two non-trainable zero-initialized variables inside a variable scope
  # (value-times-weight and weight) and updates them without debiasing. Returns the division
  # result tensor. Errors from variable creation or shape mismatch propagate.
  """Compute the weighted moving average of `value`.


  Conceptually, the weighted moving average is:
    `moving_average(value * weight) / moving_average(weight)`,
  where a moving average updates by the rule
    `new_value = decay * old_value + (1 - decay) * update`
  Internally, this Op keeps moving average variables of both `value * weight`
  and `weight`.

  Args:
    value: A numeric `Tensor`.
    decay: A float `Tensor` or float value. The moving average decay.
    weight:  `Tensor` that keeps the current value of a weight. Shape should be
      able to multiply `value`.
    truediv:  Boolean, if `True`, dividing by `moving_average(weight)` is
      floating point division.  If `False`, use division implied by dtypes.
    collections:  List of graph collections keys to add the internal variables
      `value * weight` and `weight` to. Defaults to
      `[GraphKeys.GLOBAL_VARIABLES]`.
    name: Optional name of the returned operation. Defaults to
      "WeightedMovingAvg".

  Returns:
    An Operation that updates and returns the weighted moving average.
  """
  # Unlike assign_moving_average, the weighted moving average doesn't modify
  # user-visible variables. It is the ratio of two internal variables, which are
  # moving averages of the updates.  Thus, the signature of this function is
  # quite different than assign_moving_average.
  if collections is None:
    collections = [ops.GraphKeys.GLOBAL_VARIABLES]
  with variable_scope.variable_scope(name, "WeightedMovingAvg",
                                     [value, weight, decay]) as scope:
    value_x_weight_var = variable_scope.get_variable(
        "value_x_weight",
        shape=value.get_shape(),
        dtype=value.dtype,
        initializer=init_ops.zeros_initializer(),
        trainable=False,
        collections=collections)
    weight_var = variable_scope.get_variable(
        "weight",
        shape=weight.get_shape(),
        dtype=weight.dtype,
        initializer=init_ops.zeros_initializer(),
        trainable=False,
        collections=collections)
    numerator = assign_moving_average(
        value_x_weight_var, value * weight, decay, zero_debias=False)
    denominator = assign_moving_average(
        weight_var, weight, decay, zero_debias=False)

    if truediv:
      return math_ops.truediv(numerator, denominator, name=scope.name)
    else:
      return math_ops.divide(numerator, denominator, name=scope.name)


def _update(strategy, var, update_fn, args):
  # [FUNCTION DOC]
  # HELPER: strategy-aware update dispatcher for the moving average functions.
  # Inputs: strategy (distribution strategy), var (variable to update), update_fn (callback
  # taking the variable followed by the extra arguments), args (tuple of extra arguments).
  # Must be called in cross-replica context, otherwise an assertion error is raised. If already
  # inside an update context (an update replica id exists) it calls the callback directly on the
  # variable; otherwise it asks the strategy to run the update on the variable. Returns the
  # callback/strategy update result.
  """Applies updates depending on the context."""
  assert distribute_lib.in_cross_replica_context(), (
      "_update can only be called in cross-replica context")
  if distribute_lib.get_update_replica_id() is not None:
    # Call update_fn on var to delegate the implementation. We expect `var` will
    # do the right thing in update context, e.g, if `var` is a MirroredVariable,
    # it should pick its component variable based on `update_replica_id` and
    # only update that.
    return update_fn(var, *args)
  else:
    return strategy.extended.update(var, update_fn, args)


def _zero_debias(strategy, unbiased_var, value, decay):
  # [FUNCTION DOC]
  # HELPER: zero-debiasing step used by assign_moving_average.
  # Inputs: strategy (distribution strategy), unbiased_var (the variable holding the debiased
  # average), value (most recent value tensor), decay (tensor holding one minus the EMA decay).
  # Creates, under a variable scope named after the variable, two non-trainable shadow variables
  # colocated with it: a biased average (same shape/dtype) and a scalar local step, both zero
  # initialized (created in the init scope); names get a numeric suffix to stay unique unless
  # the scope reuses variables. Then, via the strategy update helper, it updates the biased
  # average, increments the step, computes the bias factor as one minus (one minus decay) raised
  # to the step, and assigns the biased average divided by that factor to the unbiased variable.
  # Returns the update result. Side effects: variable creation and assignments.
  """Compute the delta required for a debiased Variable.

  All exponential moving averages initialized with Tensors are initialized to 0,
  and therefore are biased to 0. Variables initialized to 0 and used as EMAs are
  similarly biased. This function creates the debias updated amount according to
  a scale factor, as in (Kingma et al., 2015).

  To demonstrate the bias the results from 0-initialization, take an EMA that
  was initialized to `0` with decay `b`. After `t` timesteps of seeing the
  constant `c`, the variable have the following value:

  ```
    EMA = 0*b^(t) + c*(1 - b)*b^(t-1) + c*(1 - b)*b^(t-2) + ...
        = c*(1 - b^t)
  ```

  To have the true value `c`, we would divide by the scale factor `1 - b^t`.

  In order to perform debiasing, we use two shadow variables. One keeps track of
  the biased estimate, and the other keeps track of the number of updates that
  have occurred.

  Args:
    strategy: `Strategy` used to create and update variables.
    unbiased_var: A Variable representing the current value of the unbiased EMA.
    value: A Tensor representing the most recent value.
    decay: A Tensor representing `1-decay` for the EMA.

  Returns:
    The amount that the unbiased variable should be updated. Computing this
    tensor will also update the shadow variables appropriately.

  References:
    Adam - A Method for Stochastic Optimization:
      [Kingma et al., 2015](https://arxiv.org/abs/1412.6980)
      ([pdf](https://arxiv.org/pdf/1412.6980.pdf))

  """
  with variable_scope.variable_scope(
      unbiased_var.name[:-len(":0")], values=[unbiased_var, value, decay]):
    with ops.init_scope():
      biased_initializer = init_ops.zeros_initializer()
      local_step_initializer = init_ops.zeros_initializer()

    def _maybe_get_unique(name):
      # [FUNCTION DOC]
      # HELPER: picks a non-colliding shadow variable name for _zero_debias.
      # Returns the name unchanged if the current variable scope reuses variables or no variable
      # with that scoped name exists; otherwise appends an increasing numeric suffix until the
      # name is unused among the scope's global variables.
      """Get name for a unique variable, if not `reuse=True`."""
      if variable_scope.get_variable_scope().reuse:
        return name
      vs_vars = [
          x.op.name
          for x in variable_scope.get_variable_scope().global_variables()
      ]
      full_name = variable_scope.get_variable_scope().name + "/" + name
      if full_name not in vs_vars:
        return name
      idx = 1
      while full_name + ("_%d" % idx) in vs_vars:
        idx += 1
      return name + ("_%d" % idx)

    with strategy.extended.colocate_vars_with(unbiased_var):
      biased_var = variable_scope.get_variable(
          _maybe_get_unique("biased"),
          initializer=biased_initializer,
          shape=unbiased_var.get_shape(),
          dtype=unbiased_var.dtype,
          trainable=False)
      local_step = variable_scope.get_variable(
          _maybe_get_unique("local_step"),
          shape=[],
          dtype=unbiased_var.dtype,
          initializer=local_step_initializer,
          trainable=False)

  def update_fn(v, value, biased_var, local_step):
    # [FUNCTION DOC]
    # HELPER: nested update callback for _zero_debias.
    # Subtracts (biased - value) * decay from the biased variable, increments the local step by
    # one, computes the bias factor from the new step, and assigns the debiased result (biased
    # update divided by factor) to variable v under the current name scope. Returns the assign
    # result.
    update_biased = state_ops.assign_sub(biased_var,
                                         (biased_var - value) * decay)
    update_local_step = local_step.assign_add(1)

    # This function gets `1 - decay`, so use `1.0 - decay` in the exponent.
    bias_factor = 1 - math_ops.pow(1.0 - decay, update_local_step)
    return state_ops.assign(
        v, update_biased / bias_factor, name=ops.get_name_scope() + "/")

  return _update(
      strategy, unbiased_var, update_fn, args=(value, biased_var, local_step))


@tf_export("train.ExponentialMovingAverage")
class ExponentialMovingAverage:
  # [CLASS DOC]
  # Maintains exponential moving averages of a set of variables in non-trainable shadow
  # variables. State: decay (float, tensor or variable), optional num_updates (used to lower
  # the decay early in training), zero_debias flag, a name used as prefix for ops and shadow
  # variables, and a mapping from variable references to their shadow variables. Created with
  # a decay; apply creates shadow variables on first call and updates them on later calls;
  # average returns a shadow variable; average_name and variables_to_restore support legacy
  # TF1 saver workflows. Instances are not trackable by checkpoints. Exported as a public API.
  """Maintains moving averages of variables by employing an exponential decay.

  When training a model, it is often beneficial to maintain moving averages of
  the trained parameters.  Evaluations that use averaged parameters sometimes
  produce significantly better results than the final trained values.

  The `apply()` method adds shadow copies of trained variables the first time
  it is called, and maintains a moving average of the trained variables in
  their shadow copies at every additional invocation.
  It should generally be called immediately after creating the model weights,
  and then after each training step.

  The `average()` method gives access to the shadow variables.
  It allows you to use the moving averages in place of the last trained values
  for evaluations, by loading the moving averages into your model via
  `var.assign(ema.average(var))`.
  Additionally, although `ExponentialMovingAverage`
  objects are not directly trackable by checkpoints,
  `average()` returns the moving average variables for your model weights,
  which you can then checkpoint. (There is an example
  of this near the bottom of this docstring).
  So, `average()` is useful when
  building an evaluation model, or when restoring a model from a checkpoint
  file.

  The moving averages are computed using exponential decay.  You specify the
  decay value (as a scalar float value, `Tensor`, or `Variable`) when creating
  the `ExponentialMovingAverage` object.  The shadow variables are initialized
  with the same initial values as the trained variables.  When you run `apply`
  to update the moving averages, each shadow variable is updated with the
  formula:

    `shadow_variable -= (1 - decay) * (shadow_variable - variable)`

  This is mathematically equivalent to the classic formula below, but the use
  of an `assign_sub` op (the `"-="` in the formula) allows concurrent lockless
  updates to the variables:

    `shadow_variable = decay * shadow_variable + (1 - decay) * variable`

  Reasonable values for `decay` are close to 1.0, typically in the
  multiple-nines range: 0.999, 0.9999, etc.

  To have fine-grained control over the value of the decay parameter during
  training, pass a scalar `tf.Variable` as the `decay` value to the constructor,
  and update the variable as needed.

  Example usage when creating a training model:

  ```python
  # Create variables.
  var0 = tf.Variable(...)
  var1 = tf.Variable(...)
  # ... use the variables to build a training model...

  # Create an ExponentialMovingAverage object
  ema = tf.train.ExponentialMovingAverage(decay=0.9999)

  # The first `apply` creates the shadow variables that hold the moving averages
  ema.apply([var0, var1])

  # grab the moving averages for checkpointing purposes or to be able to
  # load the moving averages into the model weights
  averages = [ema.average(var0), ema.average(var1)]

  ...
  def train_step(...):
  ...
    # Apply the optimizer.
    opt.minimize(my_loss, [var0, var1])

    # Update the moving averages
    # of var0 and var1 with additional calls to `apply`
    ema.apply([var0, var1])

  ...train the model by running train_step multiple times...
  ```

  There are several ways to use the moving averages for evaluations:

  1. Assign the values of the shadow variables to your model variables with
     `Variable.assign(...)` before evaluating your
     model. You can use the `average()`
     method to get the shadow variable for a given variable. To continue
     training after using this approach, make sure to record the unaveraged
     weights and restore them before continuing to train. You can see the
     tensorflow-addons' MovingAverage optimizer's `swap_weights` method for
     one example of how to swap variables efficiently in distributed settings:
     https://github.com/tensorflow/addons/blob/v0.13.0/tensorflow_addons/optimizers/moving_average.py#L151
  2. Make sure to checkpoint out your moving average variables in your
     `tf.train.Checkpoint`. At evaluation time, create your shadow variables and
     use `tf.train.Checkpoint` to restore the moving averages into the shadow
     variables. Then, load the moving averages into the actual model weights via
     `var.assign(moving_avg)`.
  3. Checkpoint out your moving average variables in your `tf.train.Checkpoint`.
     For evaluation, restore your model weights directly from the moving
     averages instead of from the non-averaged weights.
     Caution: If you choose this approach, include only the object-graph paths
     to the averaged path in your checkpoint restore.
     If you point both the unaveraged and averaged paths in a checkpoint
     restore to the same variables, it is hard to reason about whether your
     model will restore the averaged or non-averaged variables.

  Example of saving out then restoring the shadow variable values:

  ```python
  # Create variables.
  var0 = tf.Variable(...)
  var1 = tf.Variable(...)
  # ... use the variables to build a training model...

  # Create an ExponentialMovingAverage object, create the shadow variables,
  # and grab the moving averages for checkpointing purposes.
  # (The ExponentialMovingAverage object itself is not checkpointable)
  ema = tf.train.ExponentialMovingAverage(decay=0.9999)
  ema.apply([var0, var1])
  avg_var0 = ema.average(var0)
  avg_var1 = ema.average(var1)

  # Create a Checkpoint that will manage the model weights and the averages,
  checkpoint = tf.train.Checkpoint(model_weights=[var0, var1],
                                   averaged_weights=[avg_var0, avg_var1])
  ... # Do training

  # Save out the checkpoint including the model weights and the moving averages
  checkpoint.save(...)
  ```

  Restore option: restore all averaged & non-averaged weights, then load
  moving averages into the model via `var.assign()`
  ```python
  # Create variables.
  var0 = tf.Variable(...)
  var1 = tf.Variable(...)
  # ... use the variables to build a training model...

  # Create an ExponentialMovingAverage object, create the shadow variables,
  # and grab the moving averages for checkpoint restore purposes.
  # (The ExponentialMovingAverage object itself is not checkpointable)
  ema = tf.train.ExponentialMovingAverage(decay=0.9999)
  ema.apply([var0, var1])
  avg_var0 = ema.average(var0)
  avg_var1 = ema.average(var1)

  # Create a Checkpoint that will manage the model weights and the averages,
  checkpoint = tf.train.Checkpoint(model_weights=[var0, var1],
                                   averaged_weights=[avg_var0, avg_var1])
  checkpoint.restore(...)
  var0.assign(avg_var0)
  var1.assign(avg_var1)
  # var0 and var1 now hold the moving average values
  ```

  Restore option: Directly restore the moving averages into the model weights.
  ```python
  # Create variables.
  var0 = tf.Variable(...)
  var1 = tf.Variable(...)
  # ... use the variables to build a training model...

  # Create a Checkpoint that will manage two objects with trackable state,
  checkpoint = tf.train.Checkpoint(averaged_weights=[var0, var1])
  checkpoint.restore(...)
  # var0 and var1 now hold the moving average values
  ```
  """

  def __init__(self,
               decay,
               num_updates=None,
               zero_debias=False,
               name="ExponentialMovingAverage"):
    # [FUNCTION DOC]
    # Constructor. Inputs: decay (scalar float, tensor or variable), num_updates (optional step
    # count that, if given, caps the decay at (1+n)/(10+n)), zero_debias (boolean, default
    # false; debias averages initialized from non-variable tensors), name (string prefix for
    # ops and shadow variables, default "ExponentialMovingAverage"). Only stores these values and
    # an empty mapping of shadow variables; creates nothing and raises nothing.
    """Creates a new ExponentialMovingAverage object.

    The `apply()` method has to be called to create shadow variables.
    Follow-on calls to the `apply()` method will update the moving averages
    in the shadow variables.
    (In TF 1.x graphs `apply()` will return an update op to update
    the moving averages which must be explicitly run).

    The optional `num_updates` parameter allows one to tweak the decay rate
    dynamically. It is typical to pass the count of training steps, usually
    kept in a variable that is incremented at each step, in which case the
    decay rate is lower at the start of training.  This makes moving averages
    move faster.  If passed, the actual decay rate used is:

      `min(decay, (1 + num_updates) / (10 + num_updates))`

    Args:
      decay: A scalar float value, `Tensor`, or `Variable`. The decay parameter.
      num_updates: Optional count of number of updates applied to variables.
      zero_debias: If `True`, zero debias moving-averages that are initialized
        with tensors. (Note: moving averages may not be initialized with
        non-variable tensors when eager execution is enabled).
      name: String. Optional prefix name to use for the name of ops added in
        `apply()`.
    """
    self._decay = decay
    self._num_updates = num_updates
    self._zero_debias = zero_debias
    self._name = name
    self._averages = {}

  @property
  def name(self):
    # [FUNCTION DOC]
    # Read-only property returning the name string given at construction.
    """The name of this ExponentialMovingAverage object."""
    return self._name

  def apply(self, var_list=None):
    # [FUNCTION DOC]
    # Creates shadow variables for any variable not yet tracked, then updates all averages.
    # Input: var_list, a list of variables (tensors only in legacy graph mode); defaults to the
    # trainable variables. Raises a type error for non-variable tensors under eager execution
    # and for unsupported dtypes. New shadow variables are created in the init scope on the same
    # device as the source variable (initialized from its value if initialized, else its initial
    # value); tensors get zero slots (optionally zero-debiased). Variables are added to the
    # moving-average variables collection; shadow mapping is updated. Decay is converted to
    # float32 and, if num_updates was given, capped by (1+n)/(10+n). Returns a grouped update
    # op for all variables under a name scope.
    """Maintains moving averages of variables.

    `var_list` must be a list of `Variable` objects.  This method
    creates shadow variables (holding the moving averages)
    for all elements of `var_list`, and
    updates the moving averages using the current `var_list` values. Shadow
    variables for `Variable` objects are initialized to the variable's initial
    value.

    Shadow variables are created with `trainable=False`. To access them you
    can use the EMA object's `average` method. Note that `EMA` objects are
    not trackable by checkpoints, so if you want to checkpoint or restore the
    moving variables you will need to manually grab the shadow
    variables via `average()` and assign them as `tf.Module` properties or
    directly pass them to your `tf.train.Checkpoint`.

    Note that `apply()` can be called multiple times. When eager execution is
    enabled each call to apply will update the variables once, so this needs to
    be called in a loop.

    In legacy TF 1.x graphs, this method returns an op that updates all
    shadow variables from the current value of their associated variables. In
    TF 1.x graphs without automatically control dependencies this op needs to be
    manually run.

    Args:
      var_list: A list of Variable objects. The variables
        must be of types bfloat16, float16, float32, or float64.
        (In legacy TF 1.x graphs these may be tensors, but this is unsupported
        when eager execution is enabled.)

    Returns:
      An Operation that updates the moving averages.

    Raises:
      TypeError: If the arguments are not an allowed type.
    """
    # TODO(touts): op_scope
    if var_list is None:
      var_list = variables.trainable_variables()
    for v in var_list:
      if (isinstance(v, tensor.Tensor)
          and ops.executing_eagerly_outside_functions()):
        raise TypeError(
            "tf.train.ExponentialMovingAverage does not support non-Variable"
            " tensors when eager execution is enabled.")
    zero_debias_true = set()  # set of vars to set `zero_debias=True`
    for var in var_list:
      if var.dtype.base_dtype not in [
          dtypes.bfloat16, dtypes.float16, dtypes.float32, dtypes.float64
      ]:
        raise TypeError("The variables must be half, float, or double: %s" %
                        var.name)

      if var.ref() not in self._averages:
        # For variables: to lower communication bandwidth across devices we keep
        # the moving averages on the same device as the variables. For other
        # tensors, we rely on the existing device allocation mechanism.
        with ops.init_scope():
          if isinstance(var, variables.Variable):
            with ops.device(var.device):
              initialized_value = cond.cond(
                  variable_v1.is_variable_initialized(var), var.read_value,
                  lambda: var.initial_value)  # pylint: disable=cell-var-from-loop
            avg = slot_creator.create_slot(
                var,
                initialized_value,
                self.name,
                colocate_with_primary=True,
                copy_xla_sharding=True)
            # NOTE(mrry): We only add `tf.Variable` objects to the
            # `MOVING_AVERAGE_VARIABLES` collection.
            ops.add_to_collection(ops.GraphKeys.MOVING_AVERAGE_VARIABLES, var)
          else:
            avg = slot_creator.create_zeros_slot(
                var,
                self.name,
                colocate_with_primary=(var.op.type in [
                    "Variable", "VariableV2", "VarHandleOp"
                ]),
                copy_xla_sharding=True)
            if self._zero_debias:
              zero_debias_true.add(avg.ref())
        self._averages[var.ref()] = avg

    with ops.name_scope(self.name) as scope:
      decay = ops.convert_to_tensor(
          self._decay, dtype=dtypes.float32, name="decay")
      if self._num_updates is not None:
        num_updates = math_ops.cast(
            self._num_updates, dtypes.float32, name="num_updates")
        decay = math_ops.minimum(decay,
                                 (1.0 + num_updates) / (10.0 + num_updates))
      updates = []
      for var in var_list:
        avg = self._averages[var.ref()]
        zero_debias = avg.ref() in zero_debias_true
        updates.append(assign_moving_average(avg, var, decay, zero_debias))
      return control_flow_ops.group(*updates, name=scope)

  def average(self, var):
    # [FUNCTION DOC]
    # Looks up the shadow variable for the given variable (by reference). Returns it, or the
    # null value if apply has not been called for that variable. No side effects.
    """Returns the `Variable` holding the average of `var`.

    Args:
      var: A `Variable` object.

    Returns:
      A `Variable` object or `None` if the moving average of `var`
      is not maintained.
    """
    return self._averages.get(var.ref(), None)

  @doc_controls.do_not_generate_docs
  def average_name(self, var):
    # [FUNCTION DOC]
    # Returns the name (without the output suffix) of the shadow variable for the given
    # variable, for use with legacy savers. If a shadow exists, its actual name is used;
    # otherwise a unique name is computed from the variable name and this object's name in the
    # default graph without marking it used. Hidden from generated docs. No side effects.
    """[Meant for TF1] Returns name of `Variable` holding the average for `var`.

    (Designed to work with legacy `tf.compat.v1.train.Saver`, it is sensitive to
    specific variable names and not recommended for TF2)

    The typical scenario for `ExponentialMovingAverage` is to compute moving
    averages of variables during training, and restore the variables from the
    computed moving averages during evaluations.

    To restore variables, you have to know the name of the shadow variables.
    That name and the original variable can then be passed to a `Saver()` object
    to restore the variable from the moving average value with:
      `saver = tf.compat.v1.train.Saver({ema.average_name(var): var})`

    `average_name()` can be called whether or not `apply()` has been called.

    Args:
      var: A `Variable` object.

    Returns:
      A string: The name of the variable that will be used or was used
      by the `ExponentialMovingAverage class` to hold the moving average of
      `var`.
    """
    if var.ref() in self._averages:
      return self._averages[var.ref()].name[:-len(":0")]
    return ops.get_default_graph().unique_name(
        var.name[:-len(":0")] + "/" + self.name, mark_as_used=False)

  @doc_controls.do_not_generate_docs
  def variables_to_restore(self, moving_avg_variables=None):
    # [FUNCTION DOC]
    # Builds a mapping from checkpoint names to variables for a legacy saver. Input:
    # moving_avg_variables, optional list of variables that should be restored from their
    # moving-average names; defaults to trainable variables plus the moving-average collection
    # (note: the default path extends the list returned by the trainable-variables call).
    # Duplicates are removed; each such variable maps from its average name. All other global
    # variables map from their own op names unless that name is already taken. Returns the
    # mapping. Hidden from generated docs; no other side effects.
    """[Designed for TF 1.x] Returns a map of names to `Variables` to restore.

    (Designed to work with legacy `tf.compat.v1.train.Saver`, sensitive to
    specific variable names and not recommended for TF2)

    If a variable has a moving average, use the moving average variable name as
    the restore name; otherwise, use the variable name.

    For example,

    ```python
      variables_to_restore = ema.variables_to_restore()
      saver = tf.compat.v1.train.Saver(variables_to_restore)
    ```

    Below is an example of such mapping:

    ```
      conv/batchnorm/gamma/ExponentialMovingAverage: conv/batchnorm/gamma,
      conv_4/conv2d_params/ExponentialMovingAverage: conv_4/conv2d_params,
      global_step: global_step
    ```

    Args:
      moving_avg_variables: a list of variables that require to use of the
        moving average variable name to be restored. If None, it will default to
        variables.moving_average_variables() + variables.trainable_variables()

    Returns:
      A map from restore_names to variables. The restore_name is either the
      original or the moving average version of the variable name, depending
      on whether the variable name is in the `moving_avg_variables`.
    """
    name_map = {}
    if moving_avg_variables is None:
      # Include trainable variables and variables which have been explicitly
      # added to the moving_average_variables collection.
      moving_avg_variables = variables.trainable_variables()
      moving_avg_variables += variables.moving_average_variables()
    # Remove duplicates
    moving_avg_variables = set(v.ref() for v in moving_avg_variables)
    # Collect all the variables with moving average,
    for v in moving_avg_variables:
      name_map[self.average_name(v.deref())] = v.deref()
    # Make sure we restore variables without moving averages as well.
    moving_avg_variable_names = set(
        v.deref().name for v in moving_avg_variables)
    for v in list(set(variables.global_variables())):
      if v.name not in moving_avg_variable_names and v.op.name not in name_map:
        name_map[v.op.name] = v
    return name_map
