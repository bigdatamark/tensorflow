# [MODULE DOC]
# HELPER: internal helper module supporting the evaluation routines of the training package.
# Provides graph-mode (session based) building blocks for evaluating a model from a checkpoint:
# a function that gets or creates the local eval-step counter, a function that reads the counter
# after update operations ran, two session-run hooks that stop evaluation after a given number
# of evaluations, and a function that restores a checkpoint and runs evaluation operations.
# No registrations or flags are created at import time; it only imports framework, ops and
# training modules.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains functions for evaluation and summarization of metrics."""

import math
import time

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import monitored_session
from tensorflow.python.training import session_run_hook


def _get_or_create_eval_step():
  # [FUNCTION DOC]
  # HELPER: used by the evaluation routines to obtain the shared evaluation counter.
  # Looks up the collection of eval-step tensors in the default graph. If exactly one entry
  # exists it is returned. If none exists, a scalar 64-bit integer local, non-trainable variable
  # named eval_step, initialized to zero, is created and registered in both the local-variables
  # and eval-step collections, then returned. Takes no arguments. Raises a value error when more
  # than one tensor is already in the eval-step collection. Side effect: may add a variable to
  # the default graph and its collections.
  """Gets or creates the eval step `Tensor`.

  Returns:
    A `Tensor` representing a counter for the evaluation step.

  Raises:
    ValueError: If multiple `Tensors` have been added to the
      `tf.GraphKeys.EVAL_STEP` collection.
  """
  graph = ops.get_default_graph()
  eval_steps = graph.get_collection(ops.GraphKeys.EVAL_STEP)
  if len(eval_steps) == 1:
    return eval_steps[0]
  elif len(eval_steps) > 1:
    raise ValueError('Multiple tensors added to tf.GraphKeys.EVAL_STEP')
  else:
    counter = variable_scope.get_variable(
        'eval_step',
        shape=[],
        dtype=dtypes.int64,
        initializer=init_ops.zeros_initializer(),
        trainable=False,
        collections=[ops.GraphKeys.LOCAL_VARIABLES, ops.GraphKeys.EVAL_STEP])
    return counter


def _get_latest_eval_step_value(update_ops):
  # [FUNCTION DOC]
  # HELPER: used by the evaluation routines to read the eval step after updates have run.
  # Input update_ops is a list of tensors or a mapping from names to tensors; a mapping is
  # reduced to the list of its values. Returns an identity tensor of the current eval-step value
  # that has a control dependency on all update operations, so reading happens after they run.
  # Creates the eval-step variable if it does not exist yet (graph side effect). No exceptions
  # of its own, but the eval-step lookup may raise a value error if several steps are registered.
  """Gets the eval step `Tensor` value after running `update_ops`.

  Args:
    update_ops: A list of `Tensors` or a dictionary of names to `Tensors`, which
      are run before reading the eval step value.

  Returns:
    A `Tensor` representing the value for the evaluation step.
  """
  if isinstance(update_ops, dict):
    update_ops = list(update_ops.values())

  with ops.control_dependencies(update_ops):
    return array_ops.identity(_get_or_create_eval_step().read_value())


class _MultiStepStopAfterNEvalsHook(session_run_hook.SessionRunHook):
  # [CLASS DOC]
  # HELPER: internal hook for the evaluation routines when several evaluation steps are executed
  # per session run call.
  # Session-run hook that stops evaluation once the requested number of evaluations completed.
  # It keeps the requested count, the initial steps-per-run value, the tensor holding the
  # completed evaluation count (set later by the evaluation function) and the shared
  # steps-per-run variable. Before each run it fetches the completed count; after each run it
  # reloads the steps-per-run variable so the final run does not overshoot, logs progress and
  # requests a stop when enough evaluations are done. A count of None means run until inputs
  # are exhausted.
  """Run hook used by the evaluation routines to run the `eval_ops` N times."""

  def __init__(self, num_evals, steps_per_run=1):
    # [FUNCTION DOC]
    # Constructor. num_evals is the total number of evaluations to run, or nothing/null to run
    # until the input is exhausted. steps_per_run is how many steps one session run executes
    # (default 1). Stores both values and initializes the completed-evaluations tensor to null.
    # Returns nothing, raises nothing, no graph side effects.
    """Constructs the run hook.

    Args:
      num_evals: The number of evaluations to run for. if set to None, will
        iterate the dataset until all inputs are exhausted.
      steps_per_run: Number of steps executed per run call.
    """
    self._num_evals = num_evals
    self._evals_completed = None
    self._steps_per_run_initial_value = steps_per_run

  def _set_evals_completed_tensor(self, updated_eval_step):
    # [FUNCTION DOC]
    # HELPER: called by the evaluation function to wire the hook to the eval-step tensor.
    # Stores updated_eval_step, a tensor holding the number of completed evaluations, so that
    # the hook can fetch it on each run. Returns nothing.
    self._evals_completed = updated_eval_step

  def begin(self):
    # [FUNCTION DOC]
    # Hook callback invoked once before the session is created. Gets or creates the shared
    # steps-per-run variable in the default graph and keeps a reference to it. Takes no
    # arguments, returns nothing; side effect is possible variable creation in the graph.
    self._steps_per_run_variable = \
        basic_session_run_hooks.get_or_create_steps_per_run_variable()

  def after_create_session(self, session, coord):
    # [FUNCTION DOC]
    # Hook callback invoked after the session is created. Inputs: the new session and the
    # coordinator (unused). Loads the steps-per-run variable with the initial steps per run,
    # limited to the requested number of evaluations when that is set. Returns nothing; side
    # effect is writing the variable value inside the session.
    # Update number of steps to run in the first run call
    if self._num_evals is None:
      steps = self._steps_per_run_initial_value
    else:
      steps = min(self._steps_per_run_initial_value, self._num_evals)
    self._steps_per_run_variable.load(steps, session=session)

  def before_run(self, run_context):
    # [FUNCTION DOC]
    # Hook callback before every session run. The run context is unused. Returns run arguments
    # asking the session to also fetch the completed-evaluations tensor under a fixed key.
    # Requires that the tensor has been set beforehand, otherwise the fetch is null.
    return session_run_hook.SessionRunArgs(
        {'evals_completed': self._evals_completed})

  def after_run(self, run_context, run_values):
    # [FUNCTION DOC]
    # Hook callback after every session run. Reads the completed-evaluations count from the
    # fetched results, reloads the steps-per-run variable (initial value, or the remaining
    # evaluations if smaller, when a count is set), logs progress, and asks the run context to
    # stop once the completed count reaches the requested number. Returns nothing. Side effects:
    # writes a variable in the session, emits an info log line, may request stop.
    evals_completed = run_values.results['evals_completed']
    # Update number of steps to run in the next iteration
    if self._num_evals is None:
      steps = self._steps_per_run_initial_value
    else:
      steps = min(self._num_evals - evals_completed,
                  self._steps_per_run_initial_value)
    self._steps_per_run_variable.load(steps, session=run_context.session)

    if self._num_evals is None:
      logging.info('Evaluation [%d]', evals_completed)
    else:
      logging.info('Evaluation [%d/%d]', evals_completed, self._num_evals)
    if self._num_evals is not None and evals_completed >= self._num_evals:
      run_context.request_stop()


class _StopAfterNEvalsHook(session_run_hook.SessionRunHook):
  # [CLASS DOC]
  # HELPER: internal hook for the evaluation routines, one step per session run.
  # Session-run hook that requests a stop after a requested number of evaluations. Holds the
  # requested count (or null for unlimited), the tensor with the completed count (set by the
  # evaluation function), a progress-logging flag and a logging frequency (every evaluation for
  # fewer than 20 evaluations, otherwise roughly every tenth of the total).
  # KNOWN ISSUE: the class docstring is identical to the multi-step hook and does not mention
  # the logging frequency throttling or the log flag.
  """Run hook used by the evaluation routines to run the `eval_ops` N times."""

  def __init__(self, num_evals, log_progress=True):
    # [FUNCTION DOC]
    # Constructor. num_evals is the number of evaluations to run, or null to run until inputs
    # are exhausted. log_progress (default true) controls progress logging. Stores them,
    # initializes the completed-evaluations tensor to null and computes the log frequency: 1
    # when the count is unknown or below 20, otherwise the floor of a tenth of the count.
    # Returns nothing, raises nothing.
    """Constructs the run hook.

    Args:
      num_evals: The number of evaluations to run for. if set to None, will
        iterate the dataset until all inputs are exhausted.
      log_progress: Whether to log evaluation progress, defaults to True.
    """
    # The number of evals to run for.
    self._num_evals = num_evals
    self._evals_completed = None
    self._log_progress = log_progress
    # Reduce logging frequency if there are 20 or more evaluations.
    self._log_frequency = (1 if (num_evals is None or num_evals < 20) else
                           math.floor(num_evals / 10.))

  def _set_evals_completed_tensor(self, updated_eval_step):
    # [FUNCTION DOC]
    # HELPER: called by the evaluation function to wire the hook to the eval-step tensor.
    # Stores updated_eval_step, the tensor with the number of completed evaluations. Returns
    # nothing.
    self._evals_completed = updated_eval_step

  def before_run(self, run_context):
    # [FUNCTION DOC]
    # Hook callback before every session run. The run context is unused. Returns run arguments
    # asking the session to also fetch the completed-evaluations tensor under a fixed key.
    return session_run_hook.SessionRunArgs(
        {'evals_completed': self._evals_completed})

  def after_run(self, run_context, run_values):
    # [FUNCTION DOC]
    # Hook callback after every session run. Reads the completed count from the fetched
    # results. If progress logging is on, logs the count (and the total when known) either on
    # every run for unknown totals or only on multiples of the log frequency and at the final
    # evaluation. Requests the run context to stop when the completed count reaches the
    # requested number. Returns nothing; side effects are log output and the stop request.
    evals_completed = run_values.results['evals_completed']
    if self._log_progress:
      if self._num_evals is None:
        logging.info('Evaluation [%d]', evals_completed)
      else:
        if ((evals_completed % self._log_frequency) == 0 or
            (self._num_evals == evals_completed)):
          logging.info('Evaluation [%d/%d]', evals_completed, self._num_evals)
    if self._num_evals is not None and evals_completed >= self._num_evals:
      run_context.request_stop()


def _evaluate_once(checkpoint_path,
                   master='',
                   scaffold=None,
                   eval_ops=None,
                   feed_dict=None,
                   final_ops=None,
                   final_ops_feed_dict=None,
                   hooks=None,
                   config=None):
  # [FUNCTION DOC]
  # HELPER: shared implementation behind the public evaluate-once style routines.
  # KNOWN ISSUE: the docstring refers to contrib classes (StopAfterNEvalsHook, SummaryAtEndHook,
  # get_or_create_eval_step) and a BNS master address; the code actually uses the private hooks
  # in this module and plain session master strings.
  # Restores the model from checkpoint_path and runs evaluation. Inputs: checkpoint_path (path
  # of the checkpoint), master (session master address, default empty), scaffold (optional
  # scaffold for initialization), eval_ops (a tensor, list or mapping of names to tensors run
  # repeatedly), feed_dict (values fed to eval_ops runs), final_ops and final_ops_feed_dict
  # (fetched once at the end), hooks (extra session-run hooks), config (optional session
  # configuration). It gets or creates the eval step; when eval_ops is given it appends an
  # increment of the eval step (by steps-per-run if a multi-step hook is present, else by one)
  # to the ops (mutating a mapping argument by adding an update_eval_step entry), computes the
  # latest step value and hands it to any stop-after hooks. It then builds a chief session
  # creator restoring the checkpoint, appends a final-ops hook, and runs eval_ops in a
  # monitored session until it should stop. Returns the values fetched for final_ops, or null if
  # there are none. Side effects: graph ops, a local variable, info logs with timing.
  """Evaluates the model at the given checkpoint path.

  During a single evaluation, the `eval_ops` is run until the session is
  interrupted or requested to finish. This is typically requested via a
  `tf.contrib.training.StopAfterNEvalsHook` which results in `eval_ops` running
  the requested number of times.

  Optionally, a user can pass in `final_ops`, a single `Tensor`, a list of
  `Tensors` or a dictionary from names to `Tensors`. The `final_ops` is
  evaluated a single time after `eval_ops` has finished running and the fetched
  values of `final_ops` are returned. If `final_ops` is left as `None`, then
  `None` is returned.

  One may also consider using a `tf.contrib.training.SummaryAtEndHook` to record
  summaries after the `eval_ops` have run. If `eval_ops` is `None`, the
  summaries run immediately after the model checkpoint has been restored.

  Note that `evaluate_once` creates a local variable used to track the number of
  evaluations run via `tf.contrib.training.get_or_create_eval_step`.
  Consequently, if a custom local init op is provided via a `scaffold`, the
  caller should ensure that the local init op also initializes the eval step.

  Args:
    checkpoint_path: The path to a checkpoint to use for evaluation.
    master: The BNS address of the TensorFlow master.
    scaffold: An tf.compat.v1.train.Scaffold instance for initializing variables
      and restoring variables. Note that `scaffold.init_fn` is used by the
      function to restore the checkpoint. If you supply a custom init_fn, then
      it must also take care of restoring the model from its checkpoint.
    eval_ops: A single `Tensor`, a list of `Tensors` or a dictionary of names to
      `Tensors`, which is run until the session is requested to stop, commonly
      done by a `tf.contrib.training.StopAfterNEvalsHook`.
    feed_dict: The feed dictionary to use when executing the `eval_ops`.
    final_ops: A single `Tensor`, a list of `Tensors` or a dictionary of names
      to `Tensors`.
    final_ops_feed_dict: A feed dictionary to use when evaluating `final_ops`.
    hooks: List of `tf.estimator.SessionRunHook` callbacks which are run inside
      the evaluation loop.
    config: An instance of `tf.compat.v1.ConfigProto` that will be used to
      configure the `Session`. If left as `None`, the default will be used.

  Returns:
    The fetched values of `final_ops` or `None` if `final_ops` is `None`.
  """
  eval_step = _get_or_create_eval_step()

  # Prepare the run hooks.
  hooks = list(hooks or [])

  if eval_ops is not None:
    if any(isinstance(h, _MultiStepStopAfterNEvalsHook) for h in hooks):
      steps_per_run_variable = \
          basic_session_run_hooks.get_or_create_steps_per_run_variable()
      update_eval_step = state_ops.assign_add(
          eval_step,
          math_ops.cast(steps_per_run_variable, dtype=eval_step.dtype),
          use_locking=True)
    else:
      update_eval_step = state_ops.assign_add(eval_step, 1, use_locking=True)

    if isinstance(eval_ops, dict):
      eval_ops['update_eval_step'] = update_eval_step
    elif isinstance(eval_ops, (tuple, list)):
      eval_ops = list(eval_ops) + [update_eval_step]
    else:
      eval_ops = [eval_ops, update_eval_step]

    eval_step_value = _get_latest_eval_step_value(eval_ops)

    for h in hooks:
      if isinstance(h, (_StopAfterNEvalsHook, _MultiStepStopAfterNEvalsHook)):
        h._set_evals_completed_tensor(eval_step_value)  # pylint: disable=protected-access

  logging.info('Starting evaluation at ' +
               time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime()))
  start = time.time()
  # Prepare the session creator.
  session_creator = monitored_session.ChiefSessionCreator(
      scaffold=scaffold,
      checkpoint_filename_with_path=checkpoint_path,
      master=master,
      config=config)

  final_ops_hook = basic_session_run_hooks.FinalOpsHook(final_ops,
                                                        final_ops_feed_dict)
  hooks.append(final_ops_hook)

  with monitored_session.MonitoredSession(
      session_creator=session_creator, hooks=hooks) as session:
    if eval_ops is not None:
      while not session.should_stop():
        session.run(eval_ops, feed_dict)
  logging.info('Inference Time : {:0.5f}s'.format(time.time() - start))

  logging.info('Finished evaluation at ' +
               time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime()))
  return final_ops_hook.final_ops_values
