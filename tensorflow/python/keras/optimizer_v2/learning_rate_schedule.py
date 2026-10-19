# [MODULE DOC]
# Learning rate schedules for the legacy Keras optimizers. Defines the abstract base class
# LearningRateSchedule plus concrete schedules (exponential, piecewise constant, polynomial,
# inverse time, cosine, cosine with restarts, linear cosine and noisy linear cosine decay). Each
# schedule is a callable object mapping the current training step to a learning rate tensor, and
# is configurable and serializable through its configuration mapping. The module also exposes
# serialize and deserialize helpers that convert schedules to and from configuration mappings
# using the Keras generic serialization utilities. Import has no side effects beyond definitions.
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
"""Various learning rate decay functions."""

import abc
import math

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_case
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.util import nest


class LearningRateSchedule(object):
  # [CLASS DOC]
  # KNOWN ISSUE: the abstract-method markers have no effect because the class does not use the
  # abstract base class metaclass, so subclasses lacking the overrides can still be instantiated;
  # the error only appears when the methods are called.
  # Base class of all learning rate schedules. A schedule is a callable taking the current step
  # and returning a learning rate; it should also provide a configuration mapping and can be
  # rebuilt from one. Subclasses override the call and get-config methods.
  """The learning rate schedule base class.

  You can use a learning rate schedule to modulate how the learning rate
  of your optimizer changes over time.

  Several built-in learning rate schedules are available, such as
  `tf.keras.optimizers.schedules.ExponentialDecay` or
  `tf.keras.optimizers.schedules.PiecewiseConstantDecay`:

  ```python
  lr_schedule = keras.optimizers.schedules.ExponentialDecay(
      initial_learning_rate=1e-2,
      decay_steps=10000,
      decay_rate=0.9)
  optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)
  ```

  A `LearningRateSchedule` instance can be passed in as the `learning_rate`
  argument of any optimizer.

  To implement your own schedule object, you should implement the `__call__`
  method, which takes a `step` argument (scalar integer tensor, the
  current training step count).
  Like for any other Keras object, you can also optionally
  make your object serializable by implementing the `get_config`
  and `from_config` methods.

  Example:

  ```python
  class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, initial_learning_rate):
      self.initial_learning_rate = initial_learning_rate

    def __call__(self, step):
       return self.initial_learning_rate / (step + 1)

  optimizer = tf.keras.optimizers.SGD(learning_rate=MyLRSchedule(0.1))
  ```
  """

  @abc.abstractmethod
  def __call__(self, step):
    # [FUNCTION DOC]
    # Abstract hook: subclasses compute the learning rate for the given step (scalar integer
    # tensor or number). This base version always raises a not-implemented error.
    raise NotImplementedError("Learning rate schedule must override __call__")

  @abc.abstractmethod
  def get_config(self):
    # [FUNCTION DOC]
    # Abstract hook: subclasses return a serializable mapping of their constructor arguments.
    # This base version always raises a not-implemented error.
    raise NotImplementedError("Learning rate schedule must override get_config")

  @classmethod
  def from_config(cls, config):
    # [FUNCTION DOC]
    # Class-level factory. Input: config, a mapping of constructor argument names to values (as
    # produced by the get-config method). Returns a new instance of the calling class built by
    # passing the mapping as keyword arguments; unknown keys make the constructor raise.
    """Instantiates a `LearningRateSchedule` from its config.

    Args:
        config: Output of `get_config()`.

    Returns:
        A `LearningRateSchedule` instance.
    """
    return cls(**config)


class ExponentialDecay(LearningRateSchedule):
  # [CLASS DOC]
  # Schedule that multiplies the initial learning rate by the decay rate raised to the power of
  # step divided by decay steps (rounded down when staircase is enabled). Stores the initial
  # rate, decay steps, decay rate, staircase flag and name. Used by passing an instance as an
  # optimizer learning rate.
  """A LearningRateSchedule that uses an exponential decay schedule.

  When training a model, it is often useful to lower the learning rate as
  the training progresses. This schedule applies an exponential decay function
  to an optimizer step, given a provided initial learning rate.

  The schedule a 1-arg callable that produces a decayed learning
  rate when passed the current optimizer step. This can be useful for changing
  the learning rate value across different invocations of optimizer functions.
  It is computed as:

  ```python
  def decayed_learning_rate(step):
    return initial_learning_rate * decay_rate ^ (step / decay_steps)
  ```

  If the argument `staircase` is `True`, then `step / decay_steps` is
  an integer division and the decayed learning rate follows a
  staircase function.

  You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
  as the learning rate.
  Example: When fitting a Keras model, decay every 100000 steps with a base
  of 0.96:

  ```python
  initial_learning_rate = 0.1
  lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
      initial_learning_rate,
      decay_steps=100000,
      decay_rate=0.96,
      staircase=True)

  model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  model.fit(data, labels, epochs=5)
  ```

  The learning rate schedule is also serializable and deserializable using
  `tf.keras.optimizers.schedules.serialize` and
  `tf.keras.optimizers.schedules.deserialize`.

  Returns:
    A 1-arg callable learning rate schedule that takes the current optimizer
    step and outputs the decayed learning rate, a scalar `Tensor` of the same
    type as `initial_learning_rate`.
  """

  def __init__(
      self,
      initial_learning_rate,
      decay_steps,
      decay_rate,
      staircase=False,
      name=None):
    # [FUNCTION DOC]
    # Stores the schedule parameters without validation. Inputs: initial_learning_rate (number
    # or float tensor), decay_steps (positive number or integer tensor), decay_rate (number or
    # float tensor), staircase (boolean, default false) and name (optional string). Returns
    # nothing.
    """Applies exponential decay to the learning rate.

    Args:
      initial_learning_rate: A scalar `float32` or `float64` `Tensor` or a
        Python number.  The initial learning rate.
      decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
        Must be positive.  See the decay computation above.
      decay_rate: A scalar `float32` or `float64` `Tensor` or a
        Python number.  The decay rate.
      staircase: Boolean.  If `True` decay the learning rate at discrete
        intervals
      name: String.  Optional name of the operation.  Defaults to
        'ExponentialDecay'.
    """
    super(ExponentialDecay, self).__init__()
    self.initial_learning_rate = initial_learning_rate
    self.decay_steps = decay_steps
    self.decay_rate = decay_rate
    self.staircase = staircase
    self.name = name

  def __call__(self, step):
    # [FUNCTION DOC]
    # Computes the decayed learning rate for the given step (number or tensor). Converts the
    # initial rate to a tensor, casts the other values to its dtype, divides step by decay
    # steps (rounding down when staircase is set) and returns initial rate times decay rate to
    # that power, as a scalar tensor of the initial rate's dtype. Creates operations in a name
    # scope; no other side effects.
    with ops.name_scope_v2(self.name or "ExponentialDecay") as name:
      initial_learning_rate = (
          tensor_conversion.convert_to_tensor_v2_with_dispatch(
              self.initial_learning_rate, name="initial_learning_rate"
          )
      )
      dtype = initial_learning_rate.dtype
      decay_steps = math_ops.cast(self.decay_steps, dtype)
      decay_rate = math_ops.cast(self.decay_rate, dtype)

      global_step_recomp = math_ops.cast(step, dtype)
      p = global_step_recomp / decay_steps
      if self.staircase:
        p = math_ops.floor(p)
      return math_ops.multiply(
          initial_learning_rate, math_ops.pow(decay_rate, p), name=name)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a mapping with the initial learning rate, decay steps, decay rate, staircase flag
    # and name, suitable for rebuilding the schedule.
    return {
        "initial_learning_rate": self.initial_learning_rate,
        "decay_steps": self.decay_steps,
        "decay_rate": self.decay_rate,
        "staircase": self.staircase,
        "name": self.name
    }


class PiecewiseConstantDecay(LearningRateSchedule):
  # [CLASS DOC]
  # Schedule returning one of several constant values depending on which interval, defined by
  # increasing step boundaries, the current step falls in. Stores boundaries, values and name.
  """A LearningRateSchedule that uses a piecewise constant decay schedule.

  The function returns a 1-arg callable to compute the piecewise constant
  when passed the current optimizer step. This can be useful for changing the
  learning rate value across different invocations of optimizer functions.

  Example: use a learning rate that's 1.0 for the first 100001 steps, 0.5
    for the next 10000 steps, and 0.1 for any additional steps.

  ```python
  step = tf.Variable(0, trainable=False)
  boundaries = [100000, 110000]
  values = [1.0, 0.5, 0.1]
  learning_rate_fn = keras.optimizers.schedules.PiecewiseConstantDecay(
      boundaries, values)

  # Later, whenever we perform an optimization step, we pass in the step.
  learning_rate = learning_rate_fn(step)
  ```

  You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
  as the learning rate. The learning rate schedule is also serializable and
  deserializable using `tf.keras.optimizers.schedules.serialize` and
  `tf.keras.optimizers.schedules.deserialize`.

  Returns:
    A 1-arg callable learning rate schedule that takes the current optimizer
    step and outputs the decayed learning rate, a scalar `Tensor` of the same
    type as the boundary tensors.

    The output of the 1-arg function that takes the `step`
    is `values[0]` when `step <= boundaries[0]`,
    `values[1]` when `step > boundaries[0]` and `step <= boundaries[1]`, ...,
    and values[-1] when `step > boundaries[-1]`.
  """

  def __init__(
      self,
      boundaries,
      values,
      name=None):
    # [FUNCTION DOC]
    # Inputs: boundaries (list of numbers or tensors, increasing, same type as the step), values
    # (list with exactly one more element than boundaries) and name (optional string). Raises a
    # value error when the lengths do not match. Strict increase of boundaries is not checked.
    # Stores the three values and returns nothing.
    """Piecewise constant from boundaries and interval values.

    Args:
      boundaries: A list of `Tensor`s or `int`s or `float`s with strictly
        increasing entries, and with all elements having the same type as the
        optimizer step.
      values: A list of `Tensor`s or `float`s or `int`s that specifies the
        values for the intervals defined by `boundaries`. It should have one
        more element than `boundaries`, and all elements should have the same
        type.
      name: A string. Optional name of the operation. Defaults to
        'PiecewiseConstant'.

    Raises:
      ValueError: if the number of elements in the lists do not match.
    """
    super(PiecewiseConstantDecay, self).__init__()

    if len(boundaries) != len(values) - 1:
      raise ValueError(
          "The length of boundaries should be 1 less than the length of values")

    self.boundaries = boundaries
    self.values = values
    self.name = name

  def __call__(self, step):
    # [FUNCTION DOC]
    # Returns the value for the interval containing step. Converts boundaries and values to
    # tensors, converts the step, casts boundaries to the step dtype when it differs, then builds
    # a mutually exclusive case selection: the first value when step is at most the first
    # boundary, the last value when step is above the last boundary, and middle values for the
    # half-open intervals between consecutive boundaries. Returns a tensor; no other side effects.
    # Note the lambdas of the interval predicates bind each value at creation time.
    with ops.name_scope_v2(self.name or "PiecewiseConstant"):
      boundaries = nest.map_structure(
          tensor_conversion.convert_to_tensor_v2_with_dispatch,
          nest.flatten(self.boundaries),
      )
      values = nest.map_structure(
          tensor_conversion.convert_to_tensor_v2_with_dispatch,
          nest.flatten(self.values),
      )
      x_recomp = tensor_conversion.convert_to_tensor_v2_with_dispatch(step)
      for i, b in enumerate(boundaries):
        if b.dtype.base_dtype != x_recomp.dtype.base_dtype:
          # We cast the boundaries to have the same type as the step
          b = math_ops.cast(b, x_recomp.dtype.base_dtype)
          boundaries[i] = b
      pred_fn_pairs = []
      pred_fn_pairs.append((x_recomp <= boundaries[0], lambda: values[0]))
      pred_fn_pairs.append((x_recomp > boundaries[-1], lambda: values[-1]))
      for low, high, v in zip(boundaries[:-1], boundaries[1:], values[1:-1]):
        # Need to bind v here; can do this with lambda v=v: ...
        pred = (x_recomp > low) & (x_recomp <= high)
        pred_fn_pairs.append((pred, lambda v=v: v))

      # The default isn't needed here because our conditions are mutually
      # exclusive and exhaustive, but tf.case requires it.
      default = lambda: values[0]
      return control_flow_case.case(pred_fn_pairs, default, exclusive=True)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a mapping with the boundaries, values and name, suitable for rebuilding the
    # schedule.
    return {
        "boundaries": self.boundaries,
        "values": self.values,
        "name": self.name
    }


class PolynomialDecay(LearningRateSchedule):
  # [CLASS DOC]
  # Schedule that moves from an initial learning rate to an end learning rate over a number of
  # steps following a polynomial of configurable power, optionally cycling (restarting with a
  # longer period) once the steps are exhausted. Stores initial rate, decay steps, end rate,
  # power, cycle flag and name.
  """A LearningRateSchedule that uses a polynomial decay schedule.

  It is commonly observed that a monotonically decreasing learning rate, whose
  degree of change is carefully chosen, results in a better performing model.
  This schedule applies a polynomial decay function to an optimizer step,
  given a provided `initial_learning_rate`, to reach an `end_learning_rate`
  in the given `decay_steps`.

  It requires a `step` value to compute the decayed learning rate. You
  can just pass a TensorFlow variable that you increment at each training
  step.

  The schedule is a 1-arg callable that produces a decayed learning rate
  when passed the current optimizer step. This can be useful for changing the
  learning rate value across different invocations of optimizer functions.
  It is computed as:

  ```python
  def decayed_learning_rate(step):
    step = min(step, decay_steps)
    return ((initial_learning_rate - end_learning_rate) *
            (1 - step / decay_steps) ^ (power)
           ) + end_learning_rate
  ```

  If `cycle` is True then a multiple of `decay_steps` is used, the first one
  that is bigger than `step`.

  ```python
  def decayed_learning_rate(step):
    decay_steps = decay_steps * ceil(step / decay_steps)
    return ((initial_learning_rate - end_learning_rate) *
            (1 - step / decay_steps) ^ (power)
           ) + end_learning_rate
  ```

  You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
  as the learning rate.
  Example: Fit a model while decaying from 0.1 to 0.01 in 10000 steps using
  sqrt (i.e. power=0.5):

  ```python
  ...
  starter_learning_rate = 0.1
  end_learning_rate = 0.01
  decay_steps = 10000
  learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
      starter_learning_rate,
      decay_steps,
      end_learning_rate,
      power=0.5)

  model.compile(optimizer=tf.keras.optimizers.SGD(
                    learning_rate=learning_rate_fn),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  model.fit(data, labels, epochs=5)
  ```

  The learning rate schedule is also serializable and deserializable using
  `tf.keras.optimizers.schedules.serialize` and
  `tf.keras.optimizers.schedules.deserialize`.

  Returns:
    A 1-arg callable learning rate schedule that takes the current optimizer
    step and outputs the decayed learning rate, a scalar `Tensor` of the same
    type as `initial_learning_rate`.
  """

  def __init__(
      self,
      initial_learning_rate,
      decay_steps,
      end_learning_rate=0.0001,
      power=1.0,
      cycle=False,
      name=None):
    # [FUNCTION DOC]
    # Stores the schedule parameters without validation. Inputs: initial_learning_rate (number
    # or float tensor), decay_steps (positive number or integer tensor), end_learning_rate
    # (default 0.0001), power (default 1.0, i.e. linear), cycle (boolean, default false) and
    # name (optional string). Returns nothing.
    """Applies a polynomial decay to the learning rate.

    Args:
      initial_learning_rate: A scalar `float32` or `float64` `Tensor` or a
        Python number.  The initial learning rate.
      decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
        Must be positive.  See the decay computation above.
      end_learning_rate: A scalar `float32` or `float64` `Tensor` or a
        Python number.  The minimal end learning rate.
      power: A scalar `float32` or `float64` `Tensor` or a
        Python number.  The power of the polynomial. Defaults to linear, 1.0.
      cycle: A boolean, whether or not it should cycle beyond decay_steps.
      name: String.  Optional name of the operation. Defaults to
        'PolynomialDecay'.
    """
    super(PolynomialDecay, self).__init__()

    self.initial_learning_rate = initial_learning_rate
    self.decay_steps = decay_steps
    self.end_learning_rate = end_learning_rate
    self.power = power
    self.cycle = cycle
    self.name = name

  def __call__(self, step):
    # [FUNCTION DOC]
    # Computes the learning rate for the given step. All values are cast to the dtype of the
    # initial rate. With cycle enabled the decay length is scaled to the first multiple of decay
    # steps at or above the step (a multiplier of one at step zero); otherwise the step is capped
    # at decay steps. Returns end rate plus (initial minus end) times (one minus step fraction)
    # raised to the power, as a scalar tensor. No side effects beyond creating operations.
    with ops.name_scope_v2(self.name or "PolynomialDecay") as name:
      initial_learning_rate = (
          tensor_conversion.convert_to_tensor_v2_with_dispatch(
              self.initial_learning_rate, name="initial_learning_rate"
          )
      )
      dtype = initial_learning_rate.dtype
      end_learning_rate = math_ops.cast(self.end_learning_rate, dtype)
      power = math_ops.cast(self.power, dtype)

      global_step_recomp = math_ops.cast(step, dtype)
      decay_steps_recomp = math_ops.cast(self.decay_steps, dtype)
      if self.cycle:
        # Find the first multiple of decay_steps that is bigger than
        # global_step. If global_step is zero set the multiplier to 1
        multiplier = array_ops.where_v2(
            math_ops.equal(global_step_recomp, 0), 1.0,
            math_ops.ceil(global_step_recomp / self.decay_steps))
        decay_steps_recomp = math_ops.multiply(decay_steps_recomp, multiplier)
      else:
        # Make sure that the global_step used is not bigger than decay_steps.
        global_step_recomp = math_ops.minimum(global_step_recomp,
                                              decay_steps_recomp)

      p = math_ops.divide(global_step_recomp, decay_steps_recomp)
      return math_ops.add(
          math_ops.multiply(initial_learning_rate - end_learning_rate,
                            math_ops.pow(1 - p, power)),
          end_learning_rate,
          name=name)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a mapping with the initial rate, decay steps, end rate, power, cycle flag and name,
    # suitable for rebuilding the schedule.
    return {
        "initial_learning_rate": self.initial_learning_rate,
        "decay_steps": self.decay_steps,
        "end_learning_rate": self.end_learning_rate,
        "power": self.power,
        "cycle": self.cycle,
        "name": self.name
    }


class InverseTimeDecay(LearningRateSchedule):
  # [CLASS DOC]
  # Schedule that divides the initial learning rate by one plus decay rate times step over decay
  # steps (rounded down when staircase is enabled). Stores initial rate, decay steps, decay rate,
  # staircase flag and name.
  """A LearningRateSchedule that uses an inverse time decay schedule.

  When training a model, it is often useful to lower the learning rate as
  the training progresses. This schedule applies the inverse decay function
  to an optimizer step, given a provided initial learning rate.
  It requires a `step` value to compute the decayed learning rate. You can
  just pass a TensorFlow variable that you increment at each training step.

  The schedule a 1-arg callable that produces a decayed learning
  rate when passed the current optimizer step. This can be useful for changing
  the learning rate value across different invocations of optimizer functions.
  It is computed as:

  ```python
  def decayed_learning_rate(step):
    return initial_learning_rate / (1 + decay_rate * step / decay_step)
  ```

  or, if `staircase` is `True`, as:

  ```python
  def decayed_learning_rate(step):
    return initial_learning_rate / (1 + decay_rate * floor(step / decay_step))
  ```

  You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
  as the learning rate.
  Example: Fit a Keras model when decaying 1/t with a rate of 0.5:

  ```python
  ...
  initial_learning_rate = 0.1
  decay_steps = 1.0
  decay_rate = 0.5
  learning_rate_fn = keras.optimizers.schedules.InverseTimeDecay(
    initial_learning_rate, decay_steps, decay_rate)

  model.compile(optimizer=tf.keras.optimizers.SGD(
                    learning_rate=learning_rate_fn),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  model.fit(data, labels, epochs=5)
  ```

  Returns:
    A 1-arg callable learning rate schedule that takes the current optimizer
    step and outputs the decayed learning rate, a scalar `Tensor` of the same
    type as `initial_learning_rate`.
  """

  def __init__(
      self,
      initial_learning_rate,
      decay_steps,
      decay_rate,
      staircase=False,
      name=None):
    # [FUNCTION DOC]
    # Stores the schedule parameters without validation. Inputs: initial_learning_rate (number or
    # float tensor), decay_steps (how often to apply decay), decay_rate (number), staircase
    # (boolean, default false) and name (optional string). Returns nothing.
    """Applies inverse time decay to the initial learning rate.

    Args:
      initial_learning_rate: A scalar `float32` or `float64` `Tensor` or a
        Python number.  The initial learning rate.
      decay_steps: How often to apply decay.
      decay_rate: A Python number.  The decay rate.
      staircase: Whether to apply decay in a discrete staircase, as opposed to
        continuous, fashion.
      name: String.  Optional name of the operation.  Defaults to
        'InverseTimeDecay'.
    """
    super(InverseTimeDecay, self).__init__()

    self.initial_learning_rate = initial_learning_rate
    self.decay_steps = decay_steps
    self.decay_rate = decay_rate
    self.staircase = staircase
    self.name = name

  def __call__(self, step):
    # [FUNCTION DOC]
    # Computes the learning rate for the given step: casts values to the initial rate's dtype,
    # divides step by decay steps (rounded down when staircase is set), and returns the initial
    # rate divided by one plus decay rate times that quotient, as a scalar tensor. No side
    # effects beyond creating operations.
    with ops.name_scope_v2(self.name or "InverseTimeDecay") as name:
      initial_learning_rate = (
          tensor_conversion.convert_to_tensor_v2_with_dispatch(
              self.initial_learning_rate, name="initial_learning_rate"
          )
      )
      dtype = initial_learning_rate.dtype
      decay_steps = math_ops.cast(self.decay_steps, dtype)
      decay_rate = math_ops.cast(self.decay_rate, dtype)

      global_step_recomp = math_ops.cast(step, dtype)
      p = global_step_recomp / decay_steps
      if self.staircase:
        p = math_ops.floor(p)
      const = math_ops.cast(constant_op.constant(1), dtype)
      denom = math_ops.add(const, math_ops.multiply(decay_rate, p))
      return math_ops.divide(initial_learning_rate, denom, name=name)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a mapping with the initial rate, decay steps, decay rate, staircase flag and name,
    # suitable for rebuilding the schedule.
    return {
        "initial_learning_rate": self.initial_learning_rate,
        "decay_steps": self.decay_steps,
        "decay_rate": self.decay_rate,
        "staircase": self.staircase,
        "name": self.name
    }


class CosineDecay(LearningRateSchedule):
  # [CLASS DOC]
  # Schedule that decays the learning rate along half a cosine wave from the initial rate down
  # to alpha times the initial rate over decay steps, staying at that floor afterwards. Stores
  # initial rate, decay steps, alpha and name.
  """A LearningRateSchedule that uses a cosine decay schedule.

  See [Loshchilov & Hutter, ICLR2016](https://arxiv.org/abs/1608.03983),
  SGDR: Stochastic Gradient Descent with Warm Restarts.

  When training a model, it is often useful to lower the learning rate as
  the training progresses. This schedule applies a cosine decay function
  to an optimizer step, given a provided initial learning rate.
  It requires a `step` value to compute the decayed learning rate. You can
  just pass a TensorFlow variable that you increment at each training step.

  The schedule a 1-arg callable that produces a decayed learning
  rate when passed the current optimizer step. This can be useful for changing
  the learning rate value across different invocations of optimizer functions.
  It is computed as:

  ```python
  def decayed_learning_rate(step):
    step = min(step, decay_steps)
    cosine_decay = 0.5 * (1 + cos(pi * step / decay_steps))
    decayed = (1 - alpha) * cosine_decay + alpha
    return initial_learning_rate * decayed
  ```

  Example usage:
  ```python
  decay_steps = 1000
  lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
      initial_learning_rate, decay_steps)
  ```

  You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
  as the learning rate. The learning rate schedule is also serializable and
  deserializable using `tf.keras.optimizers.schedules.serialize` and
  `tf.keras.optimizers.schedules.deserialize`.

  Returns:
    A 1-arg callable learning rate schedule that takes the current optimizer
    step and outputs the decayed learning rate, a scalar `Tensor` of the same
    type as `initial_learning_rate`.
  """

  def __init__(
      self,
      initial_learning_rate,
      decay_steps,
      alpha=0.0,
      name=None):
    # [FUNCTION DOC]
    # Stores the schedule parameters without validation. Inputs: initial_learning_rate (number
    # or float tensor), decay_steps (number of steps to decay over), alpha (minimum rate as a
    # fraction of the initial rate, default 0.0) and name (optional string). Returns nothing.
    """Applies cosine decay to the learning rate.

    Args:
      initial_learning_rate: A scalar `float32` or `float64` Tensor or a
        Python number. The initial learning rate.
      decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
        Number of steps to decay over.
      alpha: A scalar `float32` or `float64` Tensor or a Python number.
        Minimum learning rate value as a fraction of initial_learning_rate.
      name: String. Optional name of the operation.  Defaults to 'CosineDecay'.
    """
    super(CosineDecay, self).__init__()

    self.initial_learning_rate = initial_learning_rate
    self.decay_steps = decay_steps
    self.alpha = alpha
    self.name = name

  def __call__(self, step):
    # [FUNCTION DOC]
    # Computes the learning rate for the given step: caps the step at decay steps, computes the
    # completed fraction, a cosine factor of one half times one plus cosine of pi times that
    # fraction, mixes it with alpha, and returns the initial rate times the result as a scalar
    # tensor. No side effects beyond creating operations.
    with ops.name_scope_v2(self.name or "CosineDecay"):
      initial_learning_rate = (
          tensor_conversion.convert_to_tensor_v2_with_dispatch(
              self.initial_learning_rate, name="initial_learning_rate"
          )
      )
      dtype = initial_learning_rate.dtype
      decay_steps = math_ops.cast(self.decay_steps, dtype)

      global_step_recomp = math_ops.cast(step, dtype)
      global_step_recomp = math_ops.minimum(global_step_recomp, decay_steps)
      completed_fraction = global_step_recomp / decay_steps
      cosine_decayed = 0.5 * (1.0 + math_ops.cos(
          constant_op.constant(math.pi) * completed_fraction))

      decayed = (1 - self.alpha) * cosine_decayed + self.alpha
      return math_ops.multiply(initial_learning_rate, decayed)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a mapping with the initial rate, decay steps, alpha and name, suitable for
    # rebuilding the schedule.
    return {
        "initial_learning_rate": self.initial_learning_rate,
        "decay_steps": self.decay_steps,
        "alpha": self.alpha,
        "name": self.name
    }


class CosineDecayRestarts(LearningRateSchedule):
  # [CLASS DOC]
  # Cosine decay schedule with warm restarts: after the first period of first_decay_steps, each
  # new period is t_mul times longer and starts from m_mul times the previous peak rate. Stores
  # initial rate, first decay steps, the multipliers (privately), alpha and name.
  """A LearningRateSchedule that uses a cosine decay schedule with restarts.

  See [Loshchilov & Hutter, ICLR2016](https://arxiv.org/abs/1608.03983),
  SGDR: Stochastic Gradient Descent with Warm Restarts.

  When training a model, it is often useful to lower the learning rate as
  the training progresses. This schedule applies a cosine decay function with
  restarts to an optimizer step, given a provided initial learning rate.
  It requires a `step` value to compute the decayed learning rate. You can
  just pass a TensorFlow variable that you increment at each training step.

  The schedule a 1-arg callable that produces a decayed learning
  rate when passed the current optimizer step. This can be useful for changing
  the learning rate value across different invocations of optimizer functions.

  The learning rate multiplier first decays
  from 1 to `alpha` for `first_decay_steps` steps. Then, a warm
  restart is performed. Each new warm restart runs for `t_mul` times more
  steps and with `m_mul` times smaller initial learning rate.

  Example usage:
  ```python
  first_decay_steps = 1000
  lr_decayed_fn = (
    tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate,
        first_decay_steps))
  ```

  You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
  as the learning rate. The learning rate schedule is also serializable and
  deserializable using `tf.keras.optimizers.schedules.serialize` and
  `tf.keras.optimizers.schedules.deserialize`.

  Returns:
    A 1-arg callable learning rate schedule that takes the current optimizer
    step and outputs the decayed learning rate, a scalar `Tensor` of the same
    type as `initial_learning_rate`.
  """

  def __init__(
      self,
      initial_learning_rate,
      first_decay_steps,
      t_mul=2.0,
      m_mul=1.0,
      alpha=0.0,
      name=None):
    # [FUNCTION DOC]
    # Stores the schedule parameters without validation. Inputs: initial_learning_rate (number
    # or float tensor), first_decay_steps (length of the first period), t_mul (period growth
    # factor, default 2.0), m_mul (peak rate factor per restart, default 1.0), alpha (minimum
    # rate fraction, default 0.0) and name (optional string; the default operation name is
    # "SGDRDecay"). Returns nothing.
    """Applies cosine decay with restarts to the learning rate.

    Args:
      initial_learning_rate: A scalar `float32` or `float64` Tensor or a Python
        number. The initial learning rate.
      first_decay_steps: A scalar `int32` or `int64` `Tensor` or a Python
        number. Number of steps to decay over.
      t_mul: A scalar `float32` or `float64` `Tensor` or a Python number.
        Used to derive the number of iterations in the i-th period
      m_mul: A scalar `float32` or `float64` `Tensor` or a Python number.
        Used to derive the initial learning rate of the i-th period:
      alpha: A scalar `float32` or `float64` Tensor or a Python number.
        Minimum learning rate value as a fraction of the initial_learning_rate.
      name: String. Optional name of the operation.  Defaults to 'SGDRDecay'.
    """
    super(CosineDecayRestarts, self).__init__()

    self.initial_learning_rate = initial_learning_rate
    self.first_decay_steps = first_decay_steps
    self._t_mul = t_mul
    self._m_mul = m_mul
    self.alpha = alpha
    self.name = name

  def __call__(self, step):
    # [FUNCTION DOC]
    # Computes the learning rate for the given step. Divides step by first decay steps, then
    # determines the restart index and the fraction completed within the current period (using
    # plain floor when t_mul equals one, or a geometric-series formula otherwise, selected with
    # a conditional operation). Applies a cosine factor scaled by m_mul to the power of the
    # restart index, mixes with alpha, and returns the initial rate times the result as a scalar
    # tensor. No side effects beyond creating operations.
    with ops.name_scope_v2(self.name or "SGDRDecay") as name:
      initial_learning_rate = (
          tensor_conversion.convert_to_tensor_v2_with_dispatch(
              self.initial_learning_rate, name="initial_learning_rate"
          )
      )
      dtype = initial_learning_rate.dtype
      first_decay_steps = math_ops.cast(self.first_decay_steps, dtype)
      alpha = math_ops.cast(self.alpha, dtype)
      t_mul = math_ops.cast(self._t_mul, dtype)
      m_mul = math_ops.cast(self._m_mul, dtype)

      global_step_recomp = math_ops.cast(step, dtype)
      completed_fraction = global_step_recomp / first_decay_steps

      def compute_step(completed_fraction, geometric=False):
        # [FUNCTION DOC]
        # HELPER: branch body for the conditional in the enclosing call method. Inputs:
        # completed_fraction (tensor, step divided by first decay steps) and geometric (boolean).
        # In geometric mode (period growth factor not one) finds the restart index with a
        # logarithm and the sum of the geometric series of earlier periods, and rescales the
        # fraction into the current period; otherwise the index is the integer part and the
        # fraction is the remainder. Returns a pair: restart index and fraction within period.
        """Helper for `cond` operation."""
        if geometric:
          i_restart = math_ops.floor(
              math_ops.log(1.0 - completed_fraction * (1.0 - t_mul)) /
              math_ops.log(t_mul))

          sum_r = (1.0 - t_mul**i_restart) / (1.0 - t_mul)
          completed_fraction = (completed_fraction - sum_r) / t_mul**i_restart

        else:
          i_restart = math_ops.floor(completed_fraction)
          completed_fraction -= i_restart

        return i_restart, completed_fraction

      i_restart, completed_fraction = cond.cond(
          math_ops.equal(t_mul, 1.0),
          lambda: compute_step(completed_fraction, geometric=False),
          lambda: compute_step(completed_fraction, geometric=True))

      m_fac = m_mul**i_restart
      cosine_decayed = 0.5 * m_fac * (1.0 + math_ops.cos(
          constant_op.constant(math.pi) * completed_fraction))
      decayed = (1 - alpha) * cosine_decayed + alpha

      return math_ops.multiply(initial_learning_rate, decayed, name=name)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a mapping with the initial rate, first decay steps, t_mul, m_mul, alpha and name,
    # suitable for rebuilding the schedule.
    return {
        "initial_learning_rate": self.initial_learning_rate,
        "first_decay_steps": self.first_decay_steps,
        "t_mul": self._t_mul,
        "m_mul": self._m_mul,
        "alpha": self.alpha,
        "name": self.name
    }


# Note: this code is still used by V1 APIs.
class LinearCosineDecay(LearningRateSchedule):
  # [CLASS DOC]
  # Schedule combining a linear decay with a cosine oscillation (number of periods configurable),
  # offset by alpha and beta. Mainly kept for the v1 API. Stores initial rate, decay steps,
  # number of periods, alpha, beta and name.
  """A LearningRateSchedule that uses a linear cosine decay schedule.

  See [Bello et al., ICML2017] Neural Optimizer Search with RL.
  https://arxiv.org/abs/1709.07417

  For the idea of warm starts here controlled by `num_periods`,
  see [Loshchilov & Hutter, ICLR2016] SGDR: Stochastic Gradient Descent
  with Warm Restarts. https://arxiv.org/abs/1608.03983

  Note that linear cosine decay is more aggressive than cosine decay and
  larger initial learning rates can typically be used.

  When training a model, it is often recommended to lower the learning rate as
  the training progresses. This schedule applies a linear cosine decay
  function to an optimizer step, given a provided initial learning rate.
  It requires a `step` value to compute the decayed learning rate. You can
  just pass a TensorFlow variable that you increment at each training step.

  The schedule a 1-arg callable that produces a decayed learning
  rate when passed the current optimizer step. This can be useful for changing
  the learning rate value across different invocations of optimizer functions.
  It is computed as:

  ```python
  def decayed_learning_rate(step):
    step = min(step, decay_steps)
    linear_decay = (decay_steps - step) / decay_steps
    cosine_decay = 0.5 * (
        1 + cos(pi * 2 * num_periods * step / decay_steps))
    decayed = (alpha + linear_decay) * cosine_decay + beta
    return initial_learning_rate * decayed
  ```

  Example usage:
  ```python
  decay_steps = 1000
  lr_decayed_fn = (
    tf.keras.experimental.LinearCosineDecay(
      initial_learning_rate, decay_steps))
  ```

  You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
  as the learning rate. The learning rate schedule is also serializable and
  deserializable using `tf.keras.optimizers.schedules.serialize` and
  `tf.keras.optimizers.schedules.deserialize`.

  Returns:
    A 1-arg callable learning rate schedule that takes the current optimizer
    step and outputs the decayed learning rate, a scalar `Tensor` of the same
    type as `initial_learning_rate`.
  """

  def __init__(
      self,
      initial_learning_rate,
      decay_steps,
      num_periods=0.5,
      alpha=0.0,
      beta=0.001,
      name=None):
    # [FUNCTION DOC]
    # Stores the schedule parameters without validation. Inputs: initial_learning_rate (number
    # or float tensor), decay_steps (steps to decay over), num_periods (default 0.5), alpha
    # (default 0.0), beta (default 0.001) and name (optional string). Returns nothing.
    """Applies linear cosine decay to the learning rate.

    Args:
      initial_learning_rate: A scalar `float32` or `float64` Tensor or a Python
        number. The initial learning rate.
      decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
        Number of steps to decay over.
      num_periods: Number of periods in the cosine part of the decay.
        See computation above.
      alpha: See computation above.
      beta: See computation above.
      name: String.  Optional name of the operation.  Defaults to
        'LinearCosineDecay'.
    """
    super(LinearCosineDecay, self).__init__()

    self.initial_learning_rate = initial_learning_rate
    self.decay_steps = decay_steps
    self.num_periods = num_periods
    self.alpha = alpha
    self.beta = beta
    self.name = name

  def __call__(self, step):
    # [FUNCTION DOC]
    # Computes the learning rate for the given step: caps the step at decay steps, forms the
    # linear decay fraction and a cosine factor over the configured number of periods, combines
    # them as (alpha plus linear) times cosine plus beta, and returns the initial rate times the
    # result as a scalar tensor. No side effects beyond creating operations.
    with ops.name_scope_v2(self.name or "LinearCosineDecay") as name:
      initial_learning_rate = (
          tensor_conversion.convert_to_tensor_v2_with_dispatch(
              self.initial_learning_rate, name="initial_learning_rate"
          )
      )
      dtype = initial_learning_rate.dtype
      decay_steps = math_ops.cast(self.decay_steps, dtype)
      num_periods = math_ops.cast(self.num_periods, dtype)
      alpha = math_ops.cast(self.alpha, dtype)
      beta = math_ops.cast(self.beta, dtype)

      global_step_recomp = math_ops.cast(step, dtype)
      global_step_recomp = math_ops.minimum(global_step_recomp, decay_steps)
      linear_decayed = (decay_steps - global_step_recomp) / decay_steps
      completed_fraction = global_step_recomp / decay_steps
      fraction = 2.0 * num_periods * completed_fraction
      cosine_decayed = 0.5 * (
          1.0 + math_ops.cos(constant_op.constant(math.pi) * fraction))

      linear_cosine_decayed = (alpha + linear_decayed) * cosine_decayed + beta
      return math_ops.multiply(initial_learning_rate, linear_cosine_decayed,
                               name=name)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a mapping with the initial rate, decay steps, number of periods, alpha, beta and
    # name, suitable for rebuilding the schedule.
    return {
        "initial_learning_rate": self.initial_learning_rate,
        "decay_steps": self.decay_steps,
        "num_periods": self.num_periods,
        "alpha": self.alpha,
        "beta": self.beta,
        "name": self.name
    }


# Note: this code is still used by V1 APIs.
class NoisyLinearCosineDecay(LearningRateSchedule):
  # [CLASS DOC]
  # Like the linear cosine schedule, but adds zero-centered gaussian noise to the linear part
  # whose variance decays with the step. Mainly kept for the v1 API. Stores initial rate, decay
  # steps, initial variance, variance decay, number of periods, alpha, beta and name. Evaluating
  # it is random.
  # KNOWN ISSUE: the docstring formula shows an unbalanced parenthesis in the linear decay line.
  """A LearningRateSchedule that uses a noisy linear cosine decay schedule.

  See [Bello et al., ICML2017] Neural Optimizer Search with RL.
  https://arxiv.org/abs/1709.07417

  For the idea of warm starts here controlled by `num_periods`,
  see [Loshchilov & Hutter, ICLR2016] SGDR: Stochastic Gradient Descent
  with Warm Restarts. https://arxiv.org/abs/1608.03983

  Note that linear cosine decay is more aggressive than cosine decay and
  larger initial learning rates can typically be used.

  When training a model, it is often recommended to lower the learning rate as
  the training progresses. This schedule applies a noisy linear cosine decay
  function to an optimizer step, given a provided initial learning rate.
  It requires a `step` value to compute the decayed learning rate. You can
  just pass a TensorFlow variable that you increment at each training step.

  The schedule a 1-arg callable that produces a decayed learning
  rate when passed the current optimizer step. This can be useful for changing
  the learning rate value across different invocations of optimizer functions.
  It is computed as:

  ```python
  def decayed_learning_rate(step):
    step = min(step, decay_steps)
    linear_decay = (decay_steps - step) / decay_steps)
    cosine_decay = 0.5 * (
        1 + cos(pi * 2 * num_periods * step / decay_steps))
    decayed = (alpha + linear_decay + eps_t) * cosine_decay + beta
    return initial_learning_rate * decayed
  ```
  where eps_t is 0-centered gaussian noise with variance
  initial_variance / (1 + global_step) ** variance_decay

  Example usage:
  ```python
  decay_steps = 1000
  lr_decayed_fn = (
    tf.keras.experimental.NoisyLinearCosineDecay(
      initial_learning_rate, decay_steps))
  ```

  You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
  as the learning rate. The learning rate schedule is also serializable and
  deserializable using `tf.keras.optimizers.schedules.serialize` and
  `tf.keras.optimizers.schedules.deserialize`.

  Returns:
    A 1-arg callable learning rate schedule that takes the current optimizer
    step and outputs the decayed learning rate, a scalar `Tensor` of the same
    type as `initial_learning_rate`.
  """

  def __init__(
      self,
      initial_learning_rate,
      decay_steps,
      initial_variance=1.0,
      variance_decay=0.55,
      num_periods=0.5,
      alpha=0.0,
      beta=0.001,
      name=None):
    # [FUNCTION DOC]
    # Stores the schedule parameters without validation. Inputs: initial_learning_rate (number
    # or float tensor), decay_steps, initial_variance (default 1.0) and variance_decay (default
    # 0.55) for the noise, num_periods (default 0.5), alpha (default 0.0), beta (default 0.001)
    # and name (optional string). Returns nothing.
    """Applies noisy linear cosine decay to the learning rate.

    Args:
      initial_learning_rate: A scalar `float32` or `float64` Tensor or a Python
        number. The initial learning rate.
      decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
        Number of steps to decay over.
      initial_variance: initial variance for the noise. See computation above.
      variance_decay: decay for the noise's variance. See computation above.
      num_periods: Number of periods in the cosine part of the decay.
        See computation above.
      alpha: See computation above.
      beta: See computation above.
      name: String.  Optional name of the operation.  Defaults to
        'NoisyLinearCosineDecay'.
    """
    super(NoisyLinearCosineDecay, self).__init__()

    self.initial_learning_rate = initial_learning_rate
    self.decay_steps = decay_steps
    self.initial_variance = initial_variance
    self.variance_decay = variance_decay
    self.num_periods = num_periods
    self.alpha = alpha
    self.beta = beta
    self.name = name

  def __call__(self, step):
    # [FUNCTION DOC]
    # Computes the learning rate for the given step. Caps the step at decay steps, computes the
    # linear decay fraction, draws gaussian noise (shape of that fraction) whose standard
    # deviation comes from the initial variance divided by one plus step to the variance decay,
    # adds it to the linear part, applies the cosine factor, alpha and beta as in the plain
    # linear cosine schedule, and returns the initial rate times the result. Side effect: uses
    # the random number generator, so repeated calls give different values.
    with ops.name_scope_v2(self.name or "NoisyLinearCosineDecay") as name:
      initial_learning_rate = (
          tensor_conversion.convert_to_tensor_v2_with_dispatch(
              self.initial_learning_rate, name="initial_learning_rate"
          )
      )
      dtype = initial_learning_rate.dtype
      decay_steps = math_ops.cast(self.decay_steps, dtype)
      initial_variance = math_ops.cast(self.initial_variance, dtype)
      variance_decay = math_ops.cast(self.variance_decay, dtype)
      num_periods = math_ops.cast(self.num_periods, dtype)
      alpha = math_ops.cast(self.alpha, dtype)
      beta = math_ops.cast(self.beta, dtype)

      global_step_recomp = math_ops.cast(step, dtype)
      global_step_recomp = math_ops.minimum(global_step_recomp, decay_steps)
      linear_decayed = (decay_steps - global_step_recomp) / decay_steps
      variance = initial_variance / (
          math_ops.pow(1.0 + global_step_recomp, variance_decay))
      std = math_ops.sqrt(variance)
      noisy_linear_decayed = (
          linear_decayed + random_ops.random_normal(
              linear_decayed.shape, stddev=std))

      completed_fraction = global_step_recomp / decay_steps
      fraction = 2.0 * num_periods * completed_fraction
      cosine_decayed = 0.5 * (
          1.0 + math_ops.cos(constant_op.constant(math.pi) * fraction))
      noisy_linear_cosine_decayed = (
          (alpha + noisy_linear_decayed) * cosine_decayed + beta)

      return math_ops.multiply(
          initial_learning_rate, noisy_linear_cosine_decayed, name=name)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a mapping with the initial rate, decay steps, initial variance, variance decay,
    # number of periods, alpha, beta and name, suitable for rebuilding the schedule.
    return {
        "initial_learning_rate": self.initial_learning_rate,
        "decay_steps": self.decay_steps,
        "initial_variance": self.initial_variance,
        "variance_decay": self.variance_decay,
        "num_periods": self.num_periods,
        "alpha": self.alpha,
        "beta": self.beta,
        "name": self.name
    }


def serialize(learning_rate_schedule):
  # [FUNCTION DOC]
  # Converts a learning rate schedule object into a serializable mapping (class name plus its
  # configuration) using the Keras generic serialization utility. Input: the schedule object.
  # Returns the mapping.
  return generic_utils.serialize_keras_object(learning_rate_schedule)


def deserialize(config, custom_objects=None):
  # [FUNCTION DOC]
  # Rebuilds a learning rate schedule from a serialized mapping. Inputs: config (mapping with a
  # class name and configuration) and custom_objects (optional mapping from names to user
  # classes consulted first). Looks up classes among this module's names, so only the built-in
  # schedules and custom objects resolve; the generic utility raises a value error for unknown
  # names. Returns the new schedule object.
  return generic_utils.deserialize_keras_object(
      config,
      module_objects=globals(),
      custom_objects=custom_objects,
      printable_module_name="decay")
