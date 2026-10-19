# [MODULE DOC]
# Step function abstraction for the older distribution strategy API: a base step interface, a
# step with standard input handling (an input iterator created from a dataset function), and a
# step that implements a single-loss training step (compute gradients per replica, apply them
# with the optimizer, optionally for several iterations per step). Used by test and example
# helpers. No import-time side effects.
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
"""The step function abstraction represents a single training step."""

from tensorflow.python.eager import backprop
from tensorflow.python.training import optimizer as optimizer_lib


class Step(object):
  # [CLASS DOC]
  # Base interface for one step of a training algorithm. Holds the distribution strategy it
  # runs under; descendants implement the call method.
  """Interface for performing each step of a training algorithm."""

  def __init__(self, distribution):
    # [FUNCTION DOC]
    # Constructor. Input: distribution, the distribution strategy object; stored for later use.
    self._distribution = distribution

  @property
  def distribution(self):
    # [FUNCTION DOC]
    # Property returning the distribution strategy this step was created with.
    return self._distribution

  def initialize(self):
    # [FUNCTION DOC]
    # Returns the initialization operations for this step; the base version returns an empty list.
    return []

  def __call__(self):
    # [FUNCTION DOC]
    # Abstract: performs one step. Always raises a not-implemented error in the base class.
    """Perform one step of this training algorithm."""
    raise NotImplementedError("must be implemented in descendants")

  # TODO(priyag): Add an method to access initialization and finalize ops.


class StandardInputStep(Step):
  # [CLASS DOC]
  # Step that sets up input handling: creates an input iterator from the dataset function through
  # the distribution strategy and exposes its initializer. Base for steps that consume a dataset.
  """Step with a standard implementation of input handling.

  Args:
    dataset_fn: a function that returns a tf.data Dataset that produces the
      input for the model.
  """

  def __init__(self, dataset_fn, distribution):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the class docstring lists only dataset_fn as an argument but the constructor
    # also requires distribution.
    # Constructor. Inputs: dataset_fn (callable with no arguments returning a dataset) and
    # distribution (strategy). Stores the strategy via the base class and creates an input
    # iterator through the strategy from an input function that ignores its argument and calls
    # dataset_fn. Side effect: creates the iterator.
    super(StandardInputStep, self).__init__(distribution)
    self._iterator = distribution.make_input_fn_iterator(lambda _: dataset_fn())

  def initialize(self):
    # [FUNCTION DOC]
    # Returns the input iterator's initializer (an operation or list of them), unlike the base
    # version which returns an empty list.
    return self._iterator.initializer


class StandardSingleLossStep(StandardInputStep):
  # [CLASS DOC]
  # Training step for a feed-forward network with one loss. Holds the loss function, optimizer
  # and iterations per step; calling an instance runs the configured number of iterations on the
  # strategy and returns the resulting run operation. Inherits input iterator handling.
  """A step function that implements a training step for a feed forward network.

  An instance of this class is intended to be used as a callable:

  ```python
  ...
  step = step_fn.StandardSingleLossStep(
      dataset, loss_fn, optimizer, distribution)

  # Run a single training step on a given DistributionStrategy:
  step(distribution)
  ...
  ```

  Args:
    dataset_fn: a function that returns a tf.data Dataset that produces the
      input for the model.
    loss_fn: a function that takes a context and inputs as arguments. It returns
      the loss for those inputs. `context` is an instance of
      `values.MultiStepContext` that will be passed when `loss_fn` is run.
      `context` can be used to specify the outputs to be returned from
      `loss_fn`, among other things.
    optimizer: an optimizer that implements an update rule.
    distribution: a `DistributionStrategy` object.
  """

  def __init__(self, dataset_fn, loss_fn, optimizer, distribution,
               iterations_per_step=1):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the class docstring omits the iterations_per_step argument.
    # Constructor. Inputs: dataset_fn, loss_fn (takes a multi-step context and inputs, returns the
    # loss), optimizer, distribution and iterations_per_step (default 1). Sets up input handling
    # via the base class and stores the loss function, optimizer and iteration count.
    super(StandardSingleLossStep, self).__init__(dataset_fn, distribution)
    self._loss_fn = loss_fn
    self._optimizer = optimizer
    self._iterations_per_step = iterations_per_step

  def __call__(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the class docstring shows calling the step with the distribution as an
    # argument, but this method takes no arguments.
    # Runs the training step inside the strategy's scope: defines a per-iteration function, asks
    # the strategy to run it for the configured number of iterations over the input iterator, and
    # returns the run operation of the resulting context. Outputs and context are not returned.
    with self._distribution.scope():
      def step_fn(ctx, inputs):
        # [FUNCTION DOC]
        # Runs one iteration. Inputs: ctx (multi-step context) and inputs (batch). Computes
        # gradients of the loss function on each replica (filtering out missing gradients),
        # then applies them through the optimizer's distributed apply. Returns the update ops.
        """Function to run one iteration with one input."""
        gradients_fn = backprop.implicit_grad(self._loss_fn)
        gradients_fn = optimizer_lib.get_filtered_grad_fn(gradients_fn)

        grads_and_vars = self.distribution.extended.call_for_each_replica(
            gradients_fn, args=(ctx, inputs))
        # If threads use layers, then we need to run the first step
        # sequentially, so that layers.build() is not executed in parallel.
        # Otherwise, multiple sets of mirrored variables are going to be
        # created.
        return self._optimizer._distributed_apply(  # pylint: disable=protected-access
            self.distribution, grads_and_vars)

      # TODO(priyag): Return the outputs, context, etc as well.
      ctx = self.distribution.extended.experimental_run_steps_on_iterator(
          step_fn, self._iterator, self._iterations_per_step)
      return ctx.run_op
