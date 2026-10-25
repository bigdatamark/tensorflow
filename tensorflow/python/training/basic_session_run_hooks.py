# [MODULE DOC]
# Collection of commonly used SessionRunHook implementations for the legacy session-based
# training loop (exported to the v1 train namespace): step/second timers, tensor logging, stop at
# step, checkpoint saving (with listeners), step counting, NaN checking, summary saving, global
# step waiting, final-ops evaluation, feeding and profiling hooks. Also provides the helper that
# creates the steps-per-run variable used by Estimator-style multi-step loops. Import time effects
# are limited to registering the exported symbols and defining two module constants.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Some common SessionRunHook classes.

Note that the symbols that are exported to v1 tf.train namespace are also
exported to v2 in tf.estimator namespace. See
https://github.com/tensorflow/estimator/blob/master/tensorflow_estimator/python/estimator/hooks/basic_session_run_hooks.py
"""

import os
import time

import numpy as np

from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.python.client import timeline
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.training.session_run_hook import SessionRunArgs
from tensorflow.python.training.summary_io import SummaryWriterCache
from tensorflow.python.util.tf_export import tf_export

_HOOKS = "hooks"
_STEPS_PER_RUN_VAR = "steps_per_run"


class _HookTimer:
  # [CLASS DOC]
  # HELPER: abstract base for the timers that decide when hooks should fire.
  # Defines the interface (reset, should-trigger query, record-trigger, last triggered step).
  # Holds no state; subclasses implement the behavior.
  """Base timer for determining when Hooks should trigger.

  Should not be instantiated directly.
  """

  def __init__(self):
    # [FUNCTION DOC]
    # Does nothing; the base timer has no state.
    pass

  def reset(self):
    # [FUNCTION DOC]
    # Default reset: does nothing. Subclasses clear their trigger history here.
    """Resets the timer."""
    pass

  def should_trigger_for_step(self, step):
    # [FUNCTION DOC]
    # Abstract: subclasses return whether the hook should fire at the given step number.
    # Always raises a not-implemented error in the base class.
    """Return true if the timer should trigger for the specified step."""
    raise NotImplementedError

  def update_last_triggered_step(self, step):
    # [FUNCTION DOC]
    # Abstract: subclasses record that a trigger happened at the given step and return the elapsed
    # seconds and steps since the previous trigger (both null the first time). The base class
    # always raises a not-implemented error.
    """Update the last triggered time and step number.

    Args:
      step: The current step.

    Returns:
      A pair `(elapsed_time, elapsed_steps)`, where `elapsed_time` is the number
      of seconds between the current trigger and the last one (a float), and
      `elapsed_steps` is the number of steps between the current trigger and
      the last one. Both values will be set to `None` on the first trigger.
    """
    raise NotImplementedError

  def last_triggered_step(self):
    # [FUNCTION DOC]
    # Abstract: subclasses return the step of the last trigger or null if never triggered.
    # The base class always raises a not-implemented error.
    """Returns the last triggered time step or None if never triggered."""
    raise NotImplementedError


@tf_export(v1=["train.SecondOrStepTimer"])
class SecondOrStepTimer(_HookTimer):
  # [CLASS DOC]
  # Concrete timer that fires when either a time interval in seconds or a step interval has
  # passed since the last trigger (exactly one of the two is configured). Keeps the configured
  # intervals plus the step and wall-clock time of the last trigger. Used by logging, saving and
  # summary hooks.
  """Timer that triggers at most once every N seconds or once every N steps.

  This symbol is also exported to v2 in tf.estimator namespace. See
  https://github.com/tensorflow/estimator/blob/master/tensorflow_estimator/python/estimator/hooks/basic_session_run_hooks.py
  """

  def __init__(self, every_secs=None, every_steps=None):
    # [FUNCTION DOC]
    # Creates the timer. Inputs: every_secs (number of seconds between triggers, optional) and
    # every_steps (number of steps between triggers, optional). Exactly one must be given.
    # Clears trigger history. Raises a value error if neither or both are provided.
    self.reset()
    self._every_secs = every_secs
    self._every_steps = every_steps

    if self._every_secs is None and self._every_steps is None:
      raise ValueError("Either every_secs or every_steps should be provided.")
    if (self._every_secs is not None) and (self._every_steps is not None):
      raise ValueError("Can not provide both every_secs and every_steps.")

    super(SecondOrStepTimer, self).__init__()

  def reset(self):
    # [FUNCTION DOC]
    # Forgets the last triggered step and time, so the next query triggers. Returns nothing.
    self._last_triggered_step = None
    self._last_triggered_time = None

  def should_trigger_for_step(self, step):
    # [FUNCTION DOC]
    # Input: step, the current step number. Returns true if nothing has triggered yet; false if
    # the last trigger was at this same step; otherwise true when the configured seconds have
    # elapsed since the last trigger or the configured number of steps has been reached.
    # No side effects (the trigger is recorded separately by the update method).
    """Return true if the timer should trigger for the specified step.

    Args:
      step: Training step to trigger on.

    Returns:
      True if the difference between the current time and the time of the last
      trigger exceeds `every_secs`, or if the difference between the current
      step and the last triggered step exceeds `every_steps`. False otherwise.
    """
    if self._last_triggered_step is None:
      return True

    if self._last_triggered_step == step:
      return False

    if self._every_secs is not None:
      if time.time() >= self._last_triggered_time + self._every_secs:
        return True

    if self._every_steps is not None:
      if step >= self._last_triggered_step + self._every_steps:
        return True

    return False

  def update_last_triggered_step(self, step):
    # [FUNCTION DOC]
    # Records a trigger at the given step using the current wall-clock time. Returns a pair of
    # elapsed seconds and elapsed steps since the previous trigger, both null on the first
    # trigger. Side effect: stores the new last triggered time and step.
    current_time = time.time()
    if self._last_triggered_time is None:
      elapsed_secs = None
      elapsed_steps = None
    else:
      elapsed_secs = current_time - self._last_triggered_time
      elapsed_steps = step - self._last_triggered_step

    self._last_triggered_time = current_time
    self._last_triggered_step = step
    return (elapsed_secs, elapsed_steps)

  def last_triggered_step(self):
    # [FUNCTION DOC]
    # Returns the step number of the most recent trigger, or null if it never triggered.
    return self._last_triggered_step


class NeverTriggerTimer(_HookTimer):
  # [CLASS DOC]
  # Timer that never fires; used by hooks that only act at the end of a run. Stateless.
  """Timer that never triggers."""

  def should_trigger_for_step(self, step):
    # [FUNCTION DOC]
    # Ignores the step and always returns false.
    _ = step
    return False

  def update_last_triggered_step(self, step):
    # [FUNCTION DOC]
    # Ignores the step; returns a pair of two null values (no elapsed time or steps). No state.
    _ = step
    return (None, None)

  def last_triggered_step(self):
    # [FUNCTION DOC]
    # Always returns null since this timer never triggers.
    return None


@tf_export(v1=["train.LoggingTensorHook"])
class LoggingTensorHook(session_run_hook.SessionRunHook):
  # [CLASS DOC]
  # Session hook that logs the values of chosen tensors at INFO level, every N local steps, every
  # N seconds, and/or once at the end of the run. Holds the tag-to-tensor mapping, the display
  # order of the tags, an optional formatter, a trigger timer, the log-at-end flag, and per-run
  # state (iteration counter, resolved tensors and whether the current run triggers).
  """Prints the given tensors every N local steps, every N seconds, or at end.

  The tensors will be printed to the log, with `INFO` severity. If you are not
  seeing the logs, you might want to add the following line after your imports:

  ```python
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  ```

  Note that if `at_end` is True, `tensors` should not include any tensor
  whose evaluation produces a side effect such as consuming additional inputs.

  @compatibility(TF2)
  Please check this [notebook][notebook] on how to migrate the API to TF2.

  [notebook]:https://github.com/tensorflow/docs/blob/master/site/en/guide/migrate/logging_stop_hook.ipynb

  @end_compatibility

  """

  def __init__(self,
               tensors,
               every_n_iter=None,
               every_n_secs=None,
               at_end=False,
               formatter=None):
    # [FUNCTION DOC]
    # Configures the hook. Inputs: tensors (mapping from tag to tensor or tensor name, or a list of
    # tensors/names which then serve as their own tags), every_n_iter (step interval, optional),
    # every_n_secs (seconds interval, optional), at_end (also log at session end, default false),
    # formatter (optional callable turning the tag-to-value mapping into a string).
    # Raises a value error if neither at_end nor exactly one interval is given, or if every_n_iter
    # is not positive. Uses a never-triggering timer when only at_end is requested.
    # Note: tags from a mapping are sorted; list input keeps its order.
    """Initializes a `LoggingTensorHook`.

    Args:
      tensors: `dict` that maps string-valued tags to tensors/tensor names, or
        `iterable` of tensors/tensor names.
      every_n_iter: `int`, print the values of `tensors` once every N local
        steps taken on the current worker.
      every_n_secs: `int` or `float`, print the values of `tensors` once every N
        seconds. Exactly one of `every_n_iter` and `every_n_secs` should be
        provided.
      at_end: `bool` specifying whether to print the values of `tensors` at the
        end of the run.
      formatter: function, takes dict of `tag`->`Tensor` and returns a string.
        If `None` uses default printing all tensors.

    Raises:
      ValueError: if `every_n_iter` is non-positive.
    """
    only_log_at_end = (
        at_end and (every_n_iter is None) and (every_n_secs is None))
    if (not only_log_at_end and
        (every_n_iter is None) == (every_n_secs is None)):
      raise ValueError(
          "either at_end and/or exactly one of every_n_iter and every_n_secs "
          "must be provided.")
    if every_n_iter is not None and every_n_iter <= 0:
      raise ValueError("invalid every_n_iter=%s." % every_n_iter)
    if not isinstance(tensors, dict):
      self._tag_order = tensors
      tensors = {item: item for item in tensors}
    else:
      self._tag_order = sorted(tensors.keys())
    self._tensors = tensors
    self._formatter = formatter
    self._timer = (
        NeverTriggerTimer() if only_log_at_end else SecondOrStepTimer(
            every_secs=every_n_secs, every_steps=every_n_iter))
    self._log_at_end = at_end

  def begin(self):
    # [FUNCTION DOC]
    # Called before the graph is finalized: resets the timer and iteration counter and resolves
    # every configured tensor name into a graph element of the default graph.
    # Raises (via the lookup) if a name is not found in the graph.
    self._timer.reset()
    self._iter_count = 0
    # Convert names to tensors if given
    self._current_tensors = {
        tag: _as_graph_element(tensor)
        for (tag, tensor) in self._tensors.items()
    }

  def before_run(self, run_context):  # pylint: disable=unused-argument
    # [FUNCTION DOC]
    # Asks the timer whether to log on this run (remembered for after_run). If so returns session
    # run arguments requesting the tensors to be fetched; otherwise returns null. run_context unused.
    self._should_trigger = self._timer.should_trigger_for_step(self._iter_count)
    if self._should_trigger:
      return SessionRunArgs(self._current_tensors)
    else:
      return None

  def _log_tensors(self, tensor_values):
    # [FUNCTION DOC]
    # HELPER: formats and logs fetched tensor values for this hook.
    # Input: tensor_values, mapping from tag to fetched value. Records the trigger with the timer,
    # then logs either the formatter output or a comma separated "tag = value" list (with elapsed
    # seconds when known) at INFO level. Temporarily disables scientific notation in numpy printing
    # and restores the previous options afterwards (not restored if logging raises).
    original = np.get_printoptions()
    np.set_printoptions(suppress=True)
    elapsed_secs, _ = self._timer.update_last_triggered_step(self._iter_count)
    if self._formatter:
      logging.info(self._formatter(tensor_values))
    else:
      stats = []
      for tag in self._tag_order:
        stats.append("%s = %s" % (tag, tensor_values[tag]))
      if elapsed_secs is not None:
        logging.info("%s (%.3f sec)", ", ".join(stats), elapsed_secs)
      else:
        logging.info("%s", ", ".join(stats))
    np.set_printoptions(**original)

  def after_run(self, run_context, run_values):
    # [FUNCTION DOC]
    # If this run was selected for logging, logs the fetched values. Always increments the
    # iteration counter. run_context is unused.
    _ = run_context
    if self._should_trigger:
      self._log_tensors(run_values.results)

    self._iter_count += 1

  def end(self, session):
    # [FUNCTION DOC]
    # When logging at end was requested, evaluates all configured tensors once with the given
    # session and logs them. Otherwise does nothing.
    if self._log_at_end:
      values = session.run(self._current_tensors)
      self._log_tensors(values)


def get_or_create_steps_per_run_variable():
  # [FUNCTION DOC]
  # Returns the single non-trainable int32 scalar resource variable (initialized to one) that
  # controls loop iterations per device run. Looks it up in a graph collection of the default
  # graph; returns it if exactly one exists, raises a runtime error if more than one exists, and
  # otherwise creates it under the hooks scope (reusing if present) and adds it to the collection
  # and local variables. Side effect: may add a variable to the default graph.
  """Gets or creates the steps_per_run variable.

  In Estimator, the user provided computation, the model_fn, is wrapped
  inside a tf.while_loop for peak performance. The iterations of the loop are
  specified by this variable, which adjusts its value on the CPU after each
  device program execution and before the next execution.

  The purpose of using a variable, rather than a constant, is to allow
  Estimator adapt the device training iterations according to the final steps
  specified by users. For example, if the user sets the steps_per_run as
  4 and steps as 10 in Estimator.train(), the steps_per_run
  variable will have the following value before each training run.

      - 1-st execution: steps_per_run = 4
      - 2-nd execution: steps_per_run = 4
      - 3-rd execution: steps_per_run = 2

  As model_fn increases the global step once per train_op invocation, the global
  step is 10 after all executions, matching the steps=10 inputs passed in by
  users.

  Returns:
    A TF non-trainable resource variable.

  Raises:
    RuntimeError: If multi steps_per_run variables were found.
  """
  graph = ops.get_default_graph()
  collection_name = "{}_{}".format(_HOOKS, _STEPS_PER_RUN_VAR)
  steps_per_run_vars = graph.get_collection(collection_name)
  if len(steps_per_run_vars) == 1:
    return steps_per_run_vars[0]
  elif len(steps_per_run_vars) > 1:
    raise RuntimeError("Multiple steps_per_run_var in collection.")

  with variable_scope.variable_scope(_HOOKS, reuse=variable_scope.AUTO_REUSE):
    return variable_scope.get_variable(
        _STEPS_PER_RUN_VAR,
        initializer=init_ops.ones_initializer(),
        shape=[],
        dtype=dtypes.int32,
        trainable=False,
        collections=[collection_name, ops.GraphKeys.LOCAL_VARIABLES],
        use_resource=True)


class _MultiStepStopAtStepHook(session_run_hook.SessionRunHook):
  # [CLASS DOC]
  # HELPER: internal variant of the stop-at-step hook for loops that run several steps per
  # session call. Besides requesting a stop it updates the steps-per-run variable so the final
  # run executes only the remaining steps. Holds the step budget, the target last step, the
  # initial steps per run, the global step tensor and the steps-per-run variable.
  """Hook that requests stop at a specified step."""

  def __init__(self, num_steps=None, last_step=None, steps_per_run=1):
    # [FUNCTION DOC]
    # Inputs: num_steps (steps to run after start, optional), last_step (absolute final step,
    # optional) and steps_per_run (steps per session call, default 1). Exactly one of num_steps and
    # last_step is required. Raises a value error if both or neither are given, or if steps_per_run
    # is null or below one. Stores the values.
    """Initializes a `MultiStepStopAtStepHook`.

    This hook requests stop after either a number of steps have been
    executed or a last step has been reached. Only one of the two options can be
    specified.

    if `num_steps` is specified, it indicates the number of steps to execute
    after `begin()` is called. If instead `last_step` is specified, it
    indicates the last step we want to execute, as passed to the `after_run()`
    call.

    In Estimator, the user provided computation, the model_fn, is wrapped
    inside a tf.while_loop for peak performance. The steps_per_run variable
    determines the number of iterations of the loop before returning to the CPU.

    Args:
      num_steps: Number of steps to execute.
      last_step: Step after which to stop.
      steps_per_run: Number of steps executed per run call.

    Raises:
      ValueError: If one of the arguments is invalid.
    """
    if num_steps is None and last_step is None:
      raise ValueError("One of num_steps or last_step must be specified.")
    if num_steps is not None and last_step is not None:
      raise ValueError("Only one of num_steps or last_step can be specified.")
    if steps_per_run is None or steps_per_run < 1:
      raise ValueError("steps_per_run should be greater than 0")
    self._num_steps = num_steps
    self._last_step = last_step
    self._steps_per_run_initial_value = steps_per_run

  def begin(self):
    # [FUNCTION DOC]
    # Finds the global step tensor and the steps-per-run variable (creating the latter if needed).
    # Raises a runtime error if no global step exists in the graph.
    # KNOWN ISSUE: the error message names StopAtStepHook although this is the multi-step variant.
    self._global_step_tensor = training_util.get_global_step()
    if self._global_step_tensor is None:
      raise RuntimeError("Global step should be created to use StopAtStepHook.")
    self._steps_per_run_variable = get_or_create_steps_per_run_variable()

  def _update_steps_per_run_variable(self, global_step, session):
    # [FUNCTION DOC]
    # HELPER: sets the steps-per-run variable to the smaller of the steps remaining until the last
    # step and the configured steps per run. Inputs: current global step value and the session in
    # which the variable value is loaded (side effect on that session).
    steps = min(self._last_step - global_step,
                self._steps_per_run_initial_value)
    self._steps_per_run_variable.load(steps, session=session)

  def after_create_session(self, session, coord):
    # [FUNCTION DOC]
    # Reads the current global step; if only a step count was given, converts it into an absolute
    # last step. Then initializes the steps-per-run variable. coord is unused.
    global_step = session.run(self._global_step_tensor)
    if self._last_step is None:
      self._last_step = global_step + self._num_steps
    self._update_steps_per_run_variable(global_step, session)

  def after_run(self, run_context, run_values):
    # [FUNCTION DOC]
    # Reads the global step with a separate session call. Requests the run to stop once the last
    # step is reached, otherwise updates the steps-per-run variable for the next run.
    # Global step cannot be retrieved via SessionRunArgs and before_run due to
    # race condition in hook execution.
    global_step = run_context.session.run(self._global_step_tensor)
    if global_step >= self._last_step:
      run_context.request_stop()
    else:
      self._update_steps_per_run_variable(global_step, run_context.session)


@tf_export(v1=["train.StopAtStepHook"])
class StopAtStepHook(session_run_hook.SessionRunHook):
  # [CLASS DOC]
  # Session hook that asks the training loop to stop after a given number of steps or when a
  # given global step is reached. Holds the step budget or absolute last step and the global step
  # read tensor.
  """Hook that requests stop at a specified step.

  @compatibility(TF2)
  Please check this [notebook][notebook] on how to migrate the API to TF2.

  [notebook]:https://github.com/tensorflow/docs/blob/master/site/en/guide/migrate/logging_stop_hook.ipynb

  @end_compatibility
  """

  def __init__(self, num_steps=None, last_step=None):
    # [FUNCTION DOC]
    # Inputs: num_steps (steps to run after the session starts, optional) and last_step (absolute
    # global step at which to stop, optional). Exactly one is required; raises a value error if
    # both or neither are given. Stores the values.
    """Initializes a `StopAtStepHook`.

    This hook requests stop after either a number of steps have been
    executed or a last step has been reached. Only one of the two options can be
    specified.

    if `num_steps` is specified, it indicates the number of steps to execute
    after `begin()` is called. If instead `last_step` is specified, it
    indicates the last step we want to execute, as passed to the `after_run()`
    call.

    Args:
      num_steps: Number of steps to execute.
      last_step: Step after which to stop.

    Raises:
      ValueError: If one of the arguments is invalid.
    """
    if num_steps is None and last_step is None:
      raise ValueError("One of num_steps or last_step must be specified.")
    if num_steps is not None and last_step is not None:
      raise ValueError("Only one of num_steps or last_step can be specified.")
    self._num_steps = num_steps
    self._last_step = last_step

  def begin(self):
    # [FUNCTION DOC]
    # Gets or creates the global step read tensor. Raises a runtime error if there is no global
    # step. May add ops to the default graph.
    self._global_step_tensor = training_util._get_or_create_global_step_read()  # pylint: disable=protected-access
    if self._global_step_tensor is None:
      raise RuntimeError("Global step should be created to use StopAtStepHook.")

  def after_create_session(self, session, coord):
    # [FUNCTION DOC]
    # If only a step count was given, reads the current global step and sets the absolute last
    # step to current plus the count. coord is unused.
    if self._last_step is None:
      global_step = session.run(self._global_step_tensor)
      self._last_step = global_step + self._num_steps

  def before_run(self, run_context):  # pylint: disable=unused-argument
    # [FUNCTION DOC]
    # Requests the global step value to be fetched along with each run. run_context unused.
    return SessionRunArgs(self._global_step_tensor)

  def after_run(self, run_context, run_values):
    # [FUNCTION DOC]
    # Takes the global step fetched before the run, adds one (assuming the run incremented it),
    # and if that reaches the last step re-reads the real global step and requests stop when it
    # really has reached the last step.
    global_step = run_values.results + 1
    if global_step >= self._last_step:
      # Check latest global step to ensure that the targeted last step is
      # reached. global_step read tensor is the value of global step
      # before running the operation. We're not sure whether current session.run
      # incremented the global_step or not. Here we're checking it.

      step = run_context.session.run(self._global_step_tensor)
      if step >= self._last_step:
        run_context.request_stop()


@tf_export(v1=["train.CheckpointSaverListener"])
class CheckpointSaverListener:
  # [CLASS DOC]
  # Interface (with do-nothing defaults) for callbacks run by the checkpoint saver hook: at
  # session start, before and after each save, and at session end. A listener may stop training
  # by returning true from the after-save callback. Stateless by default.
  """Interface for listeners that take action before or after checkpoint save.

  `CheckpointSaverListener` triggers only in steps when `CheckpointSaverHook` is
  triggered, and provides callbacks at the following points:
   - before using the session
   - before each call to `Saver.save()`
   - after each call to `Saver.save()`
   - at the end of session

  To use a listener, implement a class and pass the listener to a
  `CheckpointSaverHook`, as in this example:

  ```python
  class ExampleCheckpointSaverListener(CheckpointSaverListener):
    def begin(self):
      # You can add ops to the graph here.
      print('Starting the session.')
      self.your_tensor = ...

    def before_save(self, session, global_step_value):
      print('About to write a checkpoint')

    def after_save(self, session, global_step_value):
      print('Done writing checkpoint.')
      if decided_to_stop_training():
        return True

    def end(self, session, global_step_value):
      print('Done with the session.')

  ...
  listener = ExampleCheckpointSaverListener()
  saver_hook = tf.estimator.CheckpointSaverHook(
      checkpoint_dir, listeners=[listener])
  with
  tf.compat.v1.train.MonitoredTrainingSession(chief_only_hooks=[saver_hook]):
    ...
  ```

  A `CheckpointSaverListener` may simply take some action after every
  checkpoint save. It is also possible for the listener to use its own schedule
  to act less frequently, e.g. based on global_step_value. In this case,
  implementors should implement the `end()` method to handle actions related to
  the last checkpoint save. But the listener should not act twice if
  `after_save()` already handled this last checkpoint save.

  A `CheckpointSaverListener` can request training to be stopped, by returning
  True in `after_save`. Please note that, in replicated distributed training
  setting, only `chief` should use this behavior. Otherwise each worker will do
  their own evaluation, which may be wasteful of resources.
  """

  def begin(self):
    # [FUNCTION DOC]
    # Called once before the session is used; may add ops to the graph. Default does nothing.
    pass

  def before_save(self, session, global_step_value):
    # [FUNCTION DOC]
    # Called before each checkpoint is written, with the session and current global step value.
    # Default does nothing.
    pass

  def after_save(self, session, global_step_value):
    # [FUNCTION DOC]
    # Called after each checkpoint is written, with the session and global step value. May return
    # true to request that training stops. Default does nothing and returns null.
    pass

  def end(self, session, global_step_value):
    # [FUNCTION DOC]
    # Called at the end of the session, with the session and final global step value. Default does
    # nothing.
    pass


@tf_export(v1=["train.CheckpointSaverHook"])
class CheckpointSaverHook(session_run_hook.SessionRunHook):
  # [CLASS DOC]
  # Session hook that periodically (every N steps or seconds) saves checkpoints, writes the graph
  # definition and meta graph, emits session-log summaries, and notifies registered listeners.
  # Holds the saver or scaffold, target paths, a timer, the listeners, the global step tensor, a
  # summary writer and the steps-per-run estimate.
  """Saves checkpoints every N steps or seconds."""

  def __init__(self,
               checkpoint_dir,
               save_secs=None,
               save_steps=None,
               saver=None,
               checkpoint_basename="model.ckpt",
               scaffold=None,
               listeners=None,
               save_graph_def=True):
    # [FUNCTION DOC]
    # Inputs: checkpoint_dir (directory for files), save_secs / save_steps (interval, exactly one
    # required by the timer), saver (optional saver), checkpoint_basename (file base name, default
    # model.ckpt), scaffold (optional, used to obtain the saver), listeners (optional list of
    # listeners), save_graph_def (write the graph and meta graphs, default true).
    # Raises a value error if both saver and scaffold are given, and (from the timer) if the
    # intervals are missing or both given. Logs a creation message.
    """Initializes a `CheckpointSaverHook`.

    Args:
      checkpoint_dir: `str`, base directory for the checkpoint files.
      save_secs: `int`, save every N secs.
      save_steps: `int`, save every N steps.
      saver: `Saver` object, used for saving.
      checkpoint_basename: `str`, base name for the checkpoint files.
      scaffold: `Scaffold`, use to get saver object.
      listeners: List of `CheckpointSaverListener` subclass instances. Used for
        callbacks that run immediately before or after this hook saves the
        checkpoint.
      save_graph_def: Whether to save the GraphDef and MetaGraphDef to
        `checkpoint_dir`. The GraphDef is saved after the session is created as
        `graph.pbtxt`. MetaGraphDefs are saved out for every checkpoint as
        `model.ckpt-*.meta`.

    Raises:
      ValueError: One of `save_steps` or `save_secs` should be set.
      ValueError: At most one of `saver` or `scaffold` should be set.
    """
    logging.info("Create CheckpointSaverHook.")
    if saver is not None and scaffold is not None:
      raise ValueError("You cannot provide both saver and scaffold.")
    self._saver = saver
    self._checkpoint_dir = checkpoint_dir
    self._save_path = os.path.join(checkpoint_dir, checkpoint_basename)
    self._scaffold = scaffold
    self._timer = SecondOrStepTimer(
        every_secs=save_secs, every_steps=save_steps)
    self._listeners = listeners or []
    # Set sufficiently high default that it never skips checking the actual
    # global step counter -- unless the user overrides it with the right value
    # for the steps_per_run.
    self._steps_per_run = 1000000
    self._save_graph_def = save_graph_def

  def _set_steps_per_run(self, steps_per_run):
    # [FUNCTION DOC]
    # HELPER: lets the surrounding training machinery tell the hook how many steps each run
    # executes. Stores the value; returns nothing.
    self._steps_per_run = steps_per_run

  def begin(self):
    # [FUNCTION DOC]
    # Gets the cached summary writer for the checkpoint directory, finds or creates the global
    # step read tensor and calls begin on all listeners. Raises a runtime error if there is no
    # global step.
    self._summary_writer = SummaryWriterCache.get(self._checkpoint_dir)
    self._global_step_tensor = training_util._get_or_create_global_step_read()  # pylint: disable=protected-access
    if self._global_step_tensor is None:
      raise RuntimeError(
          "Global step should be created to use CheckpointSaverHook.")
    for l in self._listeners:
      l.begin()

  def after_create_session(self, session, coord):
    # [FUNCTION DOC]
    # Called once the session exists. Reads the global step; optionally writes the graph
    # definition text file into the checkpoint directory; builds the meta graph (with the saver
    # definition if a saver is available) and adds the graph and meta graph to the summary writer;
    # then saves an initial checkpoint at the current step and records it with the timer.
    # coord is unused. Side effects: files written, summaries emitted, listeners invoked.
    global_step = session.run(self._global_step_tensor)
    if self._save_graph_def:
      # We do write graph and saver_def at the first call of before_run.
      # We cannot do this in begin, since we let other hooks to change graph and
      # add variables in begin. Graph is finalized after all begin calls.
      training_util.write_graph(
          ops.get_default_graph().as_graph_def(add_shapes=True),
          self._checkpoint_dir, "graph.pbtxt")
    saver_def = self._get_saver().saver_def if self._get_saver() else None
    graph = ops.get_default_graph()
    meta_graph_def = meta_graph.create_meta_graph_def(
        graph_def=graph.as_graph_def(add_shapes=True), saver_def=saver_def)
    self._summary_writer.add_graph(graph)
    self._summary_writer.add_meta_graph(meta_graph_def)
    # The checkpoint saved here is the state at step "global_step".
    self._save(session, global_step)
    self._timer.update_last_triggered_step(global_step)

  def before_run(self, run_context):  # pylint: disable=unused-argument
    # [FUNCTION DOC]
    # Requests the global step value to be fetched with every run. run_context unused.
    return SessionRunArgs(self._global_step_tensor)

  def after_run(self, run_context, run_values):
    # [FUNCTION DOC]
    # Uses the (possibly stale) fetched step plus steps per run to cheaply decide whether a save
    # may be due; if so reads the real global step, rechecks the timer, saves a checkpoint and
    # requests stop if any listener asked for it.
    stale_global_step = run_values.results
    if self._timer.should_trigger_for_step(stale_global_step +
                                           self._steps_per_run):
      # get the real value after train op.
      global_step = run_context.session.run(self._global_step_tensor)
      if self._timer.should_trigger_for_step(global_step):
        self._timer.update_last_triggered_step(global_step)
        if self._save(run_context.session, global_step):
          run_context.request_stop()

  def end(self, session):
    # [FUNCTION DOC]
    # Reads the final global step, saves a last checkpoint if that step was not already saved, and
    # calls the end callback of every listener with the step.
    last_step = session.run(self._global_step_tensor)
    if last_step != self._timer.last_triggered_step():
      self._save(session, last_step)
    for l in self._listeners:
      l.end(session, last_step)

  def _save(self, session, step):
    # [FUNCTION DOC]
    # HELPER: performs one checkpoint save for the hook.
    # Inputs: session and the global step value. Calls every listener's before-save callback, saves
    # through the saver (writing the meta graph only if configured), adds a checkpoint session log
    # to the summary writer, then calls every after-save callback. Returns true if any listener
    # requested training to stop. Logs progress. Errors from saver or listeners propagate.
    """Saves the latest checkpoint, returns should_stop."""
    logging.info("Calling checkpoint listeners before saving checkpoint %d...",
                 step)
    for l in self._listeners:
      l.before_save(session, step)

    logging.info("Saving checkpoints for %d into %s.", step, self._save_path)
    self._get_saver().save(session, self._save_path, global_step=step,
                           write_meta_graph=self._save_graph_def)
    self._summary_writer.add_session_log(
        SessionLog(
            status=SessionLog.CHECKPOINT, checkpoint_path=self._save_path),
        step)
    logging.info("Calling checkpoint listeners after saving checkpoint %d...",
                 step)
    should_stop = False
    for l in self._listeners:
      if l.after_save(session, step):
        logging.info(
            "A CheckpointSaverListener requested that training be stopped. "
            "listener: {}".format(l))
        should_stop = True
    return should_stop

  def _get_saver(self):
    # [FUNCTION DOC]
    # HELPER: resolves the saver to use. Returns the explicit saver if given, else the scaffold's
    # saver (which may be null), else the single saver found in the saver collection of the default
    # graph (cached for later). Raises a runtime error if that collection is empty or holds more
    # than one item.
    if self._saver is not None:
      return self._saver
    elif self._scaffold is not None:
      return self._scaffold.saver

    # Get saver from the SAVERS collection if present.
    collection_key = ops.GraphKeys.SAVERS
    savers = ops.get_collection(collection_key)
    if not savers:
      raise RuntimeError(
          "No items in collection {}. Please add a saver to the collection "
          "or provide a saver or scaffold.".format(collection_key))
    elif len(savers) > 1:
      raise RuntimeError(
          "More than one item in collection {}. "
          "Please indicate which one to use by passing it to the constructor."
          .format(collection_key))

    self._saver = savers[0]
    return savers[0]


@tf_export(v1=["train.StepCounterHook"])
class StepCounterHook(session_run_hook.SessionRunHook):
  # [CLASS DOC]
  # Session hook that measures training speed in global steps per second at a configurable
  # interval, logs it, and writes it as a scalar summary when a summary writer is available.
  # Holds a timer, optional writer or output directory, the last seen global step and the steps
  # per run estimate.
  """Hook that counts steps per second."""

  def __init__(self,
               every_n_steps=100,
               every_n_secs=None,
               output_dir=None,
               summary_writer=None):
    # [FUNCTION DOC]
    # Inputs: every_n_steps (step interval, default 100), every_n_secs (seconds interval),
    # output_dir (directory for a lazily created summary writer) and summary_writer (explicit
    # writer). Raises a value error unless exactly one of the two intervals is non-null.
    # KNOWN ISSUE: because every_n_steps defaults to 100, asking for a seconds interval alone also
    # requires explicitly setting every_n_steps to null, otherwise the value error is raised.

    if (every_n_steps is None) == (every_n_secs is None):
      raise ValueError(
          "exactly one of every_n_steps and every_n_secs should be provided.")
    self._timer = SecondOrStepTimer(
        every_steps=every_n_steps, every_secs=every_n_secs)

    self._summary_writer = summary_writer
    self._output_dir = output_dir
    self._last_global_step = None
    self._steps_per_run = 1

  def _set_steps_per_run(self, steps_per_run):
    # [FUNCTION DOC]
    # HELPER: lets the training machinery tell the hook how many steps each run executes.
    # Stores the value; returns nothing.
    self._steps_per_run = steps_per_run

  def begin(self):
    # [FUNCTION DOC]
    # Creates a cached summary writer from the output directory if no writer was given, finds or
    # creates the global step read tensor and builds the summary tag from the global step name.
    # Raises a runtime error if no global step exists.
    if self._summary_writer is None and self._output_dir:
      self._summary_writer = SummaryWriterCache.get(self._output_dir)
    self._global_step_tensor = training_util._get_or_create_global_step_read()  # pylint: disable=protected-access
    if self._global_step_tensor is None:
      raise RuntimeError(
          "Global step should be created to use StepCounterHook.")
    self._summary_tag = training_util.get_global_step().op.name + "/sec"

  def before_run(self, run_context):  # pylint: disable=unused-argument
    # [FUNCTION DOC]
    # Requests the global step value to be fetched with every run. run_context unused.
    return SessionRunArgs(self._global_step_tensor)

  def _log_and_record(self, elapsed_steps, elapsed_time, global_step):
    # [FUNCTION DOC]
    # HELPER: reports speed. Inputs: steps and seconds elapsed since the previous trigger and the
    # current global step. Computes steps per second, writes it as a scalar summary if a writer
    # exists, and logs it at INFO level. Division by zero elapsed time is not guarded.
    steps_per_sec = elapsed_steps / elapsed_time
    if self._summary_writer is not None:
      summary = Summary(value=[
          Summary.Value(tag=self._summary_tag, simple_value=steps_per_sec)
      ])
      self._summary_writer.add_summary(summary, global_step)
    logging.info("%s: %g", self._summary_tag, steps_per_sec)

  def after_run(self, run_context, run_values):
    # [FUNCTION DOC]
    # Uses the fetched (stale) step plus steps per run to check the timer; if due, reads the real
    # step, and on a confirmed trigger records it and logs/writes the speed (skipped on the very
    # first trigger when no interval exists). Also warns (at most five times) when the fetched
    # global step equals the previous one, i.e. the global step is not being incremented.
    # Remembers the fetched step. run_context is otherwise unused.
    _ = run_context

    stale_global_step = run_values.results
    if self._timer.should_trigger_for_step(stale_global_step +
                                           self._steps_per_run):
      # get the real value after train op.
      global_step = run_context.session.run(self._global_step_tensor)
      if self._timer.should_trigger_for_step(global_step):
        elapsed_time, elapsed_steps = self._timer.update_last_triggered_step(
            global_step)
        if elapsed_time is not None:
          self._log_and_record(elapsed_steps, elapsed_time, global_step)

    # Check whether the global step has been increased. Here, we do not use the
    # timer.last_triggered_step as the timer might record a different global
    # step value such that the comparison could be unreliable. For simplicity,
    # we just compare the stale_global_step with previously recorded version.
    if stale_global_step == self._last_global_step:
      # Here, we give a warning in the first 5 times if we have observed that
      # the global step has not been increased. For some Optimizers, the global
      # step is not increased each time by design. For example,
      # SyncReplicaOptimizer doesn't increase the global step in worker's main
      # train step.
      logging.log_first_n(
          logging.WARN,
          "It seems that global step (tf.train.get_global_step) has not "
          "been increased. Current value (could be stable): %s vs previous "
          "value: %s. You could increase the global step by passing "
          "tf.train.get_global_step() to Optimizer.apply_gradients or "
          "Optimizer.minimize.", 5, stale_global_step, self._last_global_step)

    self._last_global_step = stale_global_step


@tf_export(v1=["train.NanLossDuringTrainingError"])
class NanLossDuringTrainingError(RuntimeError):
  # [CLASS DOC]
  # Runtime error raised by the NaN-tensor hook when the monitored loss becomes NaN and the hook
  # is configured to fail. Carries a fixed message.

  def __str__(self):
    # [FUNCTION DOC]
    # Returns the fixed text saying that a NaN loss occurred during training.
    return "NaN loss during training."


@tf_export(v1=["train.NanTensorHook"])
class NanTensorHook(session_run_hook.SessionRunHook):
  # [CLASS DOC]
  # Session hook that fetches a loss tensor every run and reacts when it is NaN: either raises
  # a NaN-loss error or requests a clean stop. Holds the loss tensor and the fail flag.
  """Monitors the loss tensor and stops training if loss is NaN.

  Can either fail with exception or just stop training.
  """

  def __init__(self, loss_tensor, fail_on_nan_loss=True):
    # [FUNCTION DOC]
    # Inputs: loss_tensor (tensor to monitor) and fail_on_nan_loss (default true: raise on NaN;
    # false: just request stop). Stores both; returns nothing.
    """Initializes a `NanTensorHook`.

    Args:
      loss_tensor: `Tensor`, the loss tensor.
      fail_on_nan_loss: `bool`, whether to raise exception when loss is NaN.
    """
    self._loss_tensor = loss_tensor
    self._fail_on_nan_loss = fail_on_nan_loss

  def before_run(self, run_context):  # pylint: disable=unused-argument
    # [FUNCTION DOC]
    # Requests the loss tensor to be fetched with every run. run_context unused.
    return SessionRunArgs(self._loss_tensor)

  def after_run(self, run_context, run_values):
    # [FUNCTION DOC]
    # Checks the fetched loss for NaN. If NaN: logs an error and raises the NaN-loss error when
    # failing is enabled, otherwise logs a warning and asks the run context to stop.
    # Does nothing for a non-NaN loss.
    if np.isnan(run_values.results):
      failure_message = "Model diverged with loss = NaN."
      if self._fail_on_nan_loss:
        logging.error(failure_message)
        raise NanLossDuringTrainingError
      else:
        logging.warning(failure_message)
        # We don't raise an error but we request stop without an exception.
        run_context.request_stop()


@tf_export(v1=["train.SummarySaverHook"])
class SummarySaverHook(session_run_hook.SessionRunHook):
  # [CLASS DOC]
  # Session hook that runs summary ops every N steps or seconds and writes the results with a
  # summary writer, also emitting a session-start log on the first run and flushing at the end.
  # Holds the summary op or scaffold, writer or output directory, a timer and per-run state
  # (next expected step, whether this run requests summaries, global step tensor).
  """Saves summaries every N steps."""

  def __init__(self,
               save_steps=None,
               save_secs=None,
               output_dir=None,
               summary_writer=None,
               scaffold=None,
               summary_op=None):
    # [FUNCTION DOC]
    # Inputs: save_steps / save_secs (interval, validated by the timer), output_dir (used to create
    # a writer when none is supplied), summary_writer, scaffold (source of the summary op) and
    # summary_op (tensor or list of tensors). Raises a value error unless exactly one of scaffold
    # and summary_op is given.
    # KNOWN ISSUE: a TODO in the code notes that no error is raised when both output_dir and
    # summary_writer are missing; summaries are then silently dropped.
    """Initializes a `SummarySaverHook`.

    Args:
      save_steps: `int`, save summaries every N steps. Exactly one of
        `save_secs` and `save_steps` should be set.
      save_secs: `int`, save summaries every N seconds.
      output_dir: `string`, the directory to save the summaries to. Only used if
        no `summary_writer` is supplied.
      summary_writer: `SummaryWriter`. If `None` and an `output_dir` was passed,
        one will be created accordingly.
      scaffold: `Scaffold` to get summary_op if it's not provided.
      summary_op: `Tensor` of type `string` containing the serialized `Summary`
        protocol buffer or a list of `Tensor`. They are most likely an output by
        TF summary methods like `tf.compat.v1.summary.scalar` or
        `tf.compat.v1.summary.merge_all`. It can be passed in as one tensor; if
        more than one, they must be passed in as a list.

    Raises:
      ValueError: Exactly one of scaffold or summary_op should be set.
    """
    if ((scaffold is None and summary_op is None) or
        (scaffold is not None and summary_op is not None)):
      raise ValueError(
          "Exactly one of scaffold or summary_op must be provided.")
    self._summary_op = summary_op
    self._summary_writer = summary_writer
    self._output_dir = output_dir
    self._scaffold = scaffold
    self._timer = SecondOrStepTimer(
        every_secs=save_secs, every_steps=save_steps)
    # TODO(mdan): Throw an error if output_dir and summary_writer are None.

  def begin(self):
    # [FUNCTION DOC]
    # Creates a cached summary writer from the output directory if needed, clears the next-step
    # marker and finds or creates the global step read tensor. Raises a runtime error if there is
    # no global step.
    if self._summary_writer is None and self._output_dir:
      self._summary_writer = SummaryWriterCache.get(self._output_dir)
    self._next_step = None
    self._global_step_tensor = training_util._get_or_create_global_step_read()  # pylint: disable=protected-access
    if self._global_step_tensor is None:
      raise RuntimeError(
          "Global step should be created to use SummarySaverHook.")

  def before_run(self, run_context):  # pylint: disable=unused-argument
    # [FUNCTION DOC]
    # Decides whether this run should produce summaries (the first run, or when the timer fires for
    # the expected next step). Returns run arguments always fetching the global step and, when
    # summaries are due and a summary op exists, also the summary tensors. run_context unused.
    self._request_summary = (
        self._next_step is None or
        self._timer.should_trigger_for_step(self._next_step))
    requests = {"global_step": self._global_step_tensor}
    if self._request_summary:
      if self._get_summary_op() is not None:
        requests["summary"] = self._get_summary_op()

    return SessionRunArgs(requests)

  def after_run(self, run_context, run_values):
    # [FUNCTION DOC]
    # Does nothing without a summary writer. Otherwise determines the global step (reading the
    # real one on the first run or when summaries were requested, else stale plus one), writes a
    # session-start log on the first run, and when summaries were requested records the trigger and
    # writes each fetched summary. Updates the expected next step.
    _ = run_context
    if not self._summary_writer:
      return

    stale_global_step = run_values.results["global_step"]
    global_step = stale_global_step + 1
    if self._next_step is None or self._request_summary:
      global_step = run_context.session.run(self._global_step_tensor)

    if self._next_step is None:
      self._summary_writer.add_session_log(
          SessionLog(status=SessionLog.START), global_step)

    if self._request_summary:
      self._timer.update_last_triggered_step(global_step)
      if "summary" in run_values.results:
        for summary in run_values.results["summary"]:
          self._summary_writer.add_summary(summary, global_step)

    self._next_step = global_step + 1

  def end(self, session=None):
    # [FUNCTION DOC]
    # Flushes pending summaries of the writer, if any. The session argument is optional and unused.
    if self._summary_writer:
      self._summary_writer.flush()

  def _get_summary_op(self):
    # [FUNCTION DOC]
    # HELPER: resolves the summary tensors. Prefers the explicit summary op, else the scaffold's
    # summary op. Returns null if none is available, otherwise a list (a single tensor is wrapped).
    """Fetches the summary op either from self._summary_op or self._scaffold.

    Returns:
      Returns a list of summary `Tensor`.
    """
    summary_op = None
    if self._summary_op is not None:
      summary_op = self._summary_op
    elif self._scaffold.summary_op is not None:
      summary_op = self._scaffold.summary_op

    if summary_op is None:
      return None

    if not isinstance(summary_op, list):
      return [summary_op]
    return summary_op


@tf_export(v1=["train.GlobalStepWaiterHook"])
class GlobalStepWaiterHook(session_run_hook.SessionRunHook):
  # [CLASS DOC]
  # Session hook that blocks the first training run until the global step (advanced by other
  # workers) reaches a threshold, used to stagger worker start in distributed training. Holds the
  # threshold, a started flag and the global step tensor.
  """Delays execution until global step reaches `wait_until_step`.

  This hook delays execution until global step reaches to `wait_until_step`. It
  is used to gradually start workers in distributed settings. One example usage
  would be setting `wait_until_step=int(K*log(task_id+1))` assuming that
  task_id=0 is the chief.
  """

  def __init__(self, wait_until_step):
    # [FUNCTION DOC]
    # Input: wait_until_step, the global step (integer) to wait for. Stores it; returns nothing.
    """Initializes a `GlobalStepWaiterHook`.

    Args:
      wait_until_step: an `int` shows until which global step should we wait.
    """
    self._wait_until_step = wait_until_step

  def begin(self):
    # [FUNCTION DOC]
    # Marks the worker as not started and finds or creates the global step read tensor.
    # Raises a runtime error if there is no global step.
    # KNOWN ISSUE: the error message names a class called _GlobalStepWaiterHook, which does not
    # exist (the class is GlobalStepWaiterHook).
    self._worker_is_started = False
    self._global_step_tensor = training_util._get_or_create_global_step_read()  # pylint: disable=protected-access
    if self._global_step_tensor is None:
      raise RuntimeError(
          "Global step should be created to use _GlobalStepWaiterHook.")

  def before_run(self, run_context):
    # [FUNCTION DOC]
    # Returns null immediately if already started or the threshold is not positive. Otherwise
    # blocks, polling the global step through the run context's session every half second (sleeping
    # in between) and logging progress when the step advanced by over a thousand, until the step
    # reaches the threshold; then marks the worker started and returns null.
    if self._worker_is_started:
      return None

    if self._wait_until_step <= 0:
      self._worker_is_started = True
      return None

    logging.info("Waiting for global step %d before starting training.",
                 self._wait_until_step)
    last_logged_step = 0
    while True:
      current_step = run_context.session.run(self._global_step_tensor)
      if current_step >= self._wait_until_step:
        self._worker_is_started = True
        return None
      if current_step - last_logged_step > 1000:
        logging.info(
            "Waiting for global step %d before starting training. "
            "Current step is %d.", self._wait_until_step, current_step)
        last_logged_step = current_step
      time.sleep(0.5)


@tf_export(v1=["train.FinalOpsHook"])
class FinalOpsHook(session_run_hook.SessionRunHook):
  # [CLASS DOC]
  # Session hook that evaluates given tensors once when the session ends and keeps the results,
  # available through the final_ops_values property. Holds the ops, an optional feed mapping and
  # the computed values (null until the end).
  """A hook which evaluates `Tensors` at the end of a session."""

  def __init__(self, final_ops, final_ops_feed_dict=None):
    # [FUNCTION DOC]
    # Inputs: final_ops (a tensor, list of tensors or mapping from names to tensors) and
    # final_ops_feed_dict (optional feed mapping). Stores them and sets the result to null.
    """Initializes `FinalOpHook` with ops to run at the end of the session.

    Args:
      final_ops: A single `Tensor`, a list of `Tensors` or a dictionary of names
        to `Tensors`.
      final_ops_feed_dict: A feed dictionary to use when running
        `final_ops_dict`.
    """
    self._final_ops = final_ops
    self._final_ops_feed_dict = final_ops_feed_dict
    self._final_ops_values = None

  @property
  def final_ops_values(self):
    # [FUNCTION DOC]
    # Property returning the values computed at session end (same structure as the final ops), or
    # null if the session has not ended or no ops were given.
    return self._final_ops_values

  def end(self, session):
    # [FUNCTION DOC]
    # If final ops were given, runs them in the session with the feed mapping and stores the
    # values. If an out-of-range error or stop-iteration occurs (meaning the ops read more input),
    # logs an explanatory warning and re-raises it.
    if self._final_ops is not None:
      try:
        self._final_ops_values = session.run(
            self._final_ops, feed_dict=self._final_ops_feed_dict)
      except (errors.OutOfRangeError, StopIteration) as e:
        logging.warning(
            "An OutOfRangeError or StopIteration exception is raised by the "
            "code in FinalOpsHook. This typically means the Ops running by the "
            "FinalOpsHook have a dependency back to some input source, which "
            "should not happen. For example, for metrics in "
            "tf.estimator.Estimator, all metrics functions return two Ops: "
            "`value_op` and  `update_op`. Estimator.evaluate calls the "
            "`update_op` for each batch of the data in input source and, once "
            "it is exhausted, it call the `value_op` to get the metric values. "
            "The `value_op` here should have dependency back to variables "
            "reading only, rather than reading another batch from input. "
            "Otherwise, the `value_op`, executed by `FinalOpsHook`, triggers "
            "another data reading, which ends OutOfRangeError/StopIteration. "
            "Please fix that.")
        raise e


@tf_export(v1=["train.FeedFnHook"])
class FeedFnHook(session_run_hook.SessionRunHook):
  # [CLASS DOC]
  # Session hook that calls a user supplied function before each run and uses the returned
  # mapping as extra feed values. Holds only that function.
  """Runs `feed_fn` and sets the `feed_dict` accordingly."""

  def __init__(self, feed_fn):
    # [FUNCTION DOC]
    # Input: feed_fn, a callable without arguments returning a mapping from tensors to values.
    # Stores it as a public attribute.
    """Initializes a `FeedFnHook`.

    Args:
      feed_fn: function that takes no arguments and returns `dict` of `Tensor`
        to feed.
    """
    self.feed_fn = feed_fn

  def before_run(self, run_context):  # pylint: disable=unused-argument
    # [FUNCTION DOC]
    # Calls the feed function and returns run arguments with no fetches and that feed mapping.
    # run_context is unused.
    return session_run_hook.SessionRunArgs(
        fetches=None, feed_dict=self.feed_fn())


@tf_export(v1=["train.ProfilerHook"])
class ProfilerHook(session_run_hook.SessionRunHook):
  # [CLASS DOC]
  # Session hook that periodically runs a step with full tracing and saves the collected
  # execution statistics as a Chrome trace timeline file per profiled step, also adding the run
  # metadata to a summary writer. Holds the output file pattern, writer, display options, a
  # timer and per-run state.
  """Captures CPU/GPU profiling information every N steps or seconds.

  This produces files called "timeline-<step>.json", which are in Chrome
  Trace format.

  For more information see:
  https://github.com/catapult-project/catapult/blob/master/tracing/README.md
  """

  def __init__(self,
               save_steps=None,
               save_secs=None,
               output_dir="",
               show_dataflow=True,
               show_memory=False):
    # [FUNCTION DOC]
    # Inputs: save_steps / save_secs (interval, validated by the timer), output_dir (directory for
    # trace files, default current directory), show_dataflow (default true) and show_memory
    # (default false) trace options. Builds the output file pattern, obtains a cached summary
    # writer for the directory (side effect) and creates the timer.
    """Initializes a hook that takes periodic profiling snapshots.

    `options.run_metadata` argument of `tf.Session.Run` is used to collect
    metadata about execution. This hook sets the metadata and dumps it in Chrome
    Trace format.


    Args:
      save_steps: `int`, save profile traces every N steps. Exactly one of
        `save_secs` and `save_steps` should be set.
      save_secs: `int` or `float`, save profile traces every N seconds.
      output_dir: `string`, the directory to save the profile traces to.
        Defaults to the current directory.
      show_dataflow: `bool`, if True, add flow events to the trace connecting
        producers and consumers of tensors.
      show_memory: `bool`, if True, add object snapshot events to the trace
        showing the sizes and lifetimes of tensors.
    """
    self._output_file = os.path.join(output_dir, "timeline-{}.json")
    self._file_writer = SummaryWriterCache.get(output_dir)
    self._show_dataflow = show_dataflow
    self._show_memory = show_memory
    self._timer = SecondOrStepTimer(
        every_secs=save_secs, every_steps=save_steps)

  def begin(self):
    # [FUNCTION DOC]
    # Clears the next-step marker and finds or creates the global step read tensor.
    # Raises a runtime error if there is no global step.
    self._next_step = None
    self._global_step_tensor = training_util._get_or_create_global_step_read()  # pylint: disable=protected-access
    if self._global_step_tensor is None:
      raise RuntimeError("Global step should be created to use ProfilerHook.")

  def before_run(self, run_context):
    # [FUNCTION DOC]
    # Decides whether this run is profiled (not the first run, and the timer fires for the expected
    # next step). Returns run arguments fetching the global step, with full-trace run options when
    # profiling and none otherwise. run_context is unused.
    self._request_summary = (
        self._next_step is not None and
        self._timer.should_trigger_for_step(self._next_step))
    requests = {"global_step": self._global_step_tensor}
    opts = (
        config_pb2.RunOptions(trace_level=config_pb2.RunOptions.FULL_TRACE)
        if self._request_summary else None)

    return SessionRunArgs(requests, options=opts)

  def after_run(self, run_context, run_values):
    # [FUNCTION DOC]
    # On the first run, starts the timer so profiling waits a full interval. When this run was
    # profiled, reads the real global step, records the trigger, writes the timeline file and adds
    # the run metadata to the summary writer under a step-named tag. Updates the next expected step.
    stale_global_step = run_values.results["global_step"]
    if self._next_step is None:
      # Update the timer so that it does not activate until N steps or seconds
      # have passed.
      self._timer.update_last_triggered_step(stale_global_step)
    global_step = stale_global_step + 1
    if self._request_summary:
      global_step = run_context.session.run(self._global_step_tensor)
      self._timer.update_last_triggered_step(global_step)
      self._save(global_step, self._output_file.format(global_step),
                 run_values.run_metadata.step_stats)
      self._file_writer.add_run_metadata(run_values.run_metadata,
                                         "step_%d" % global_step)

    self._next_step = global_step + 1

  def _save(self, step, save_path, step_stats):
    # [FUNCTION DOC]
    # HELPER: writes one profile. Inputs: step number (for logging), destination file path and the
    # collected step statistics. Converts the statistics to Chrome trace JSON honoring the dataflow
    # and memory options and writes it to the file (created or overwritten). Returns nothing; file
    # errors propagate.
    logging.info("Saving timeline for %d into '%s'.", step, save_path)
    with gfile.Open(save_path, "w") as f:
      trace = timeline.Timeline(step_stats)
      f.write(
          trace.generate_chrome_trace_format(
              show_dataflow=self._show_dataflow, show_memory=self._show_memory))


def _as_graph_element(obj):
  # [FUNCTION DOC]
  # HELPER: resolves a tensor or tensor name to a graph element of the default graph, for the
  # tensor logging hook. A non-string object must belong to the current graph and is returned
  # as is, else a value error is raised. A string with a colon is looked up directly; a string
  # without one is looked up as output zero, and a value error is raised if the operation has a
  # second output (ambiguous name). Lookup failures from the graph propagate.
  """Retrieves Graph element."""
  graph = ops.get_default_graph()
  if not isinstance(obj, str):
    if not hasattr(obj, "graph") or obj.graph != graph:
      raise ValueError("Passed %s should have graph attribute that is equal "
                       "to current graph %s." % (obj, graph))
    return obj
  if ":" in obj:
    element = graph.as_graph_element(obj)
  else:
    element = graph.as_graph_element(obj + ":0")
    # Check that there is no :1 (e.g. it's single output).
    try:
      graph.as_graph_element(obj + ":1")
    except (KeyError, ValueError):
      pass
    else:
      raise ValueError("Name %s is ambiguous, "
                       "as this `Operation` has multiple outputs "
                       "(at least 2)." % obj)
  return element
