# [MODULE DOC]
# Provides a session-run hook for the graph-mode (TF1 style) training loop that saves checkpoints
# asynchronously: the checkpoint writing runs in a separate thread so the main training thread is
# blocked only briefly. Also writes the graph definition in a background thread, adds graph and
# meta-graph to the summary writer, notifies checkpoint listeners, and records checkpoint timing
# metrics (blocking write duration, asynchronous write duration, and training time saved) under an
# async-checkpoint-v1 label.
# Module state: a global timestamp of the end of the last write guarded by a lock, shared by all
# hook instances in the process (initialized when the first hook is constructed), and a metric
# label constant. Exposes the hook class and a private duration helper.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ======================================
"""Hook for asynchronous checkpointing.

This hook dispatches checkpoint writing operations in a separate thread to
allow execution to continue on the main thread.
"""

import os
import threading
import time
from typing import Any, List, Optional, Text

from tensorflow.core.util import event_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import monitored_session
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.training.summary_io import SummaryWriterCache


# Captures the timestamp of the first Saver object instantiation or end of a
# save operation. Can be accessed by multiple Saver instances.
_END_TIME_OF_LAST_WRITE = None
_END_TIME_OF_LAST_WRITE_LOCK = threading.Lock()

# API label for cell names used in TF1 async checkpoint metrics.
_ASYNC_CHECKPOINT_V1 = "async_checkpoint_v1"


def _get_duration_microseconds(start_time_seconds, end_time_seconds) -> int:
  # [FUNCTION DOC]
  # HELPER: converts a pair of timestamps into a metric value for the checkpoint hook.
  # Inputs: start and end times in seconds (numbers). Returns the difference in whole
  # microseconds as an integer, clamped to zero if the end is before the start. No side effects.
  """Returns the duration between start and end time in microseconds."""
  return max(int((end_time_seconds - start_time_seconds) * 1000000), 0)


class AsyncCheckpointSaverHook(basic_session_run_hooks.CheckpointSaverHook):
  # [CLASS DOC]
  # Session-run hook that saves checkpoints every N steps or seconds (using a step or time
  # timer), performing the save in a background thread. Subclasses the standard checkpoint saver
  # hook but overrides construction, begin, after_create_session, before_run, after_run, end and
  # the save and saver lookup logic (it does not call the parent constructor).
  # State: the saver or scaffold used to find the saver, the save thread and graph-writing
  # thread, checkpoint directory and full save path, the timer, the listeners list, steps per run,
  # the summary writer, the global step tensor, and the last step for which a checkpoint was
  # started. Used by passing an instance in the hooks of a monitored training session or an
  # estimator train call. Shares the module-level last-write timestamp across instances.
  """Saves checkpoints every N steps or seconds."""

  def __init__(self,
               checkpoint_dir: Text,
               save_secs: Optional[int] = None,
               save_steps: Optional[int] = None,
               saver: Optional[saver_lib.Saver] = None,
               checkpoint_basename: Text = "model.ckpt",
               scaffold: Optional[monitored_session.Scaffold] = None,
               listeners: Optional[List[
                   basic_session_run_hooks.CheckpointSaverListener]] = None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it initializes a plain checkpoint saver hook and lists a
    # ValueError when neither save_steps nor save_secs is set; this constructor does not check
    # that itself (the timer it creates does) and the only check it performs directly is that
    # saver and scaffold are not both given. It also does not call the parent constructor.
    # Constructor. Inputs: checkpoint_dir (directory string), save_secs and save_steps (optional
    # integers choosing the trigger interval), saver (optional saver object), checkpoint_basename
    # (file name prefix, default model.ckpt), scaffold (optional scaffold used to obtain the
    # saver), listeners (optional list of checkpoint listeners called around each save).
    # Builds the save path, logs it, raises a value error if both saver and scaffold are given,
    # creates a step-or-second timer, and initializes all state fields to empty. Also sets the
    # module-level last-write timestamp to the current time if it was never set (under a lock).
    # Returns nothing.
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

    Raises:
      ValueError: One of `save_steps` or `save_secs` should be set.
      ValueError: At most one of `saver` or `scaffold` should be set.
    """
    save_path = os.path.join(checkpoint_dir, checkpoint_basename)
    logging.info("Create AsyncCheckpointSaverHook saving to path\n%s",
                 save_path)
    if listeners:
      logging.info(" with %d listener(s).", len(listeners))
    if saver is not None and scaffold is not None:
      raise ValueError("You cannot provide both saver and scaffold.")
    self._saver = saver
    self._save_thread = None
    self._write_graph_thread = None
    self._checkpoint_dir = checkpoint_dir
    self._save_path = save_path
    self._scaffold = scaffold
    self._timer = basic_session_run_hooks.SecondOrStepTimer(
        every_secs=save_secs, every_steps=save_steps)
    self._listeners = listeners or []
    self._steps_per_run = 1
    self._summary_writer = None
    self._global_step_tensor = None

    self._last_checkpoint_step = None

    # Initialize the first timestamp for _END_TIME_OF_LAST_WRITE.
    global _END_TIME_OF_LAST_WRITE
    with _END_TIME_OF_LAST_WRITE_LOCK:
      if _END_TIME_OF_LAST_WRITE is None:
        _END_TIME_OF_LAST_WRITE = time.time()

  def _set_steps_per_run(self, steps_per_run):
    # [FUNCTION DOC]
    # HELPER: internal setter, called by the TPU estimator machinery. Stores the number of
    # training steps executed per session run (integer) in the hook. Returns nothing.
    self._steps_per_run = steps_per_run

  def begin(self):
    # [FUNCTION DOC]
    # Hook callback invoked before the session is created. Gets the cached summary writer for
    # the checkpoint directory, gets or creates the global step read tensor, and calls begin on
    # every listener. Raises a runtime error if the global step cannot be obtained. Returns
    # nothing. Side effects: may create the global step and the summary writer.
    self._summary_writer = SummaryWriterCache.get(self._checkpoint_dir)
    self._global_step_tensor = training_util._get_or_create_global_step_read()  # pylint: disable=protected-access
    if self._global_step_tensor is None:
      raise RuntimeError(
          "Global step should be created to use CheckpointSaverHook.")
    for l in self._listeners:
      l.begin()

  def after_create_session(self, session: session_lib.Session, coord: Any):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the in-code comment says the graph is written at the first call of before_run,
    # but it is actually written here, right after session creation.
    # Hook callback after the session is created. Inputs: session (the active session) and coord
    # (coordinator, unused). Reads the current global step, starts a background thread that
    # writes the graph definition (with shapes) as a text file in the checkpoint directory, adds
    # the graph and a meta-graph (including the saver definition if a saver is available) to the
    # summary writer, saves an initial checkpoint for the current step (asynchronously) and tells
    # the timer that this step was triggered. Raises a value error if the summary writer was not
    # initialized. Returns nothing.
    # Nested function _write_graph_fn takes the hook and writes the graph file; it is the target
    # of the background thread.
    global_step = session.run(self._global_step_tensor)

    # We do write graph and saver_def at the first call of before_run.
    # We cannot do this in begin, since we let other hooks to change graph and
    # add variables in begin. Graph is finalized after all begin calls.
    def _write_graph_fn(self):
      # [FUNCTION DOC]
      # HELPER: background thread body used by after_create_session. Input: self, the hook
      # (passed explicitly as the thread argument). Writes the default graph definition (with
      # shapes) to a graph text file in the hook's checkpoint directory. Returns nothing.
      training_util.write_graph(
          ops.get_default_graph().as_graph_def(add_shapes=True),
          self._checkpoint_dir, "graph.pbtxt")
    self._write_graph_thread = threading.Thread(target=_write_graph_fn,
                                                args=[self])
    self._write_graph_thread.start()

    saver_def = self._get_saver().saver_def if self._get_saver() else None
    graph = ops.get_default_graph()
    meta_graph_def = meta_graph.create_meta_graph_def(
        graph_def=graph.as_graph_def(add_shapes=True), saver_def=saver_def)
    if self._summary_writer is None:
      raise ValueError("Summary writer is not initialised")
    self._summary_writer.add_graph(graph)
    self._summary_writer.add_meta_graph(meta_graph_def)
    # The checkpoint saved here is the state at step "global_step".
    self._save(session, global_step)
    self._timer.update_last_triggered_step(global_step)

  def before_run(self, run_context: Any):  # pylint: disable=unused-argument
    # [FUNCTION DOC]
    # Hook callback before each session run. Input: run_context (unused). Returns a request to
    # additionally fetch the global step tensor so after_run can read it. No side effects.
    return session_run_hook.SessionRunArgs(self._global_step_tensor)

  def after_run(self, run_context: session_run_hook.SessionRunContext,
                run_values: Any):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the save method never returns a value, so the stop request below can never
    # happen; the return value of the save call is always nothing.
    # Hook callback after each session run. Inputs: run_context (gives the session and a stop
    # request method) and run_values (unused). Reads the global step with a separate session run;
    # if the timer says a checkpoint is due for that step, records the step as triggered, logs,
    # and starts a save. If the save call returned a true value it would request the loop to
    # stop. Returns nothing.
    global_step = run_context.session.run(self._global_step_tensor)
    if self._timer.should_trigger_for_step(global_step):
      self._timer.update_last_triggered_step(global_step)
      logging.info("Triggering checkpoint. %s", global_step)
      if self._save(run_context.session, global_step):
        run_context.request_stop()

  def end(self, session: session_lib.Session):
    # [FUNCTION DOC]
    # Hook callback when training ends. Input: session. Blocks until any pending save thread and
    # graph-writing thread finish (logging that it waits), reads the final global step, performs
    # a final synchronous save if the last started checkpoint was for a different step, and calls
    # end on every listener with the session and final step. Returns nothing.
    if self._save_thread:
      logging.info("Waiting for any pending checkpoints to finish.")
      self._save_thread.join()
    if self._write_graph_thread:
      logging.info("Waiting for any pending write_graph to finish.")
      self._write_graph_thread.join()

    last_step = session.run(self._global_step_tensor)

    if self._last_checkpoint_step != last_step:
      self._save(session, last_step, asynchronous=False)

    for l in self._listeners:
      l.end(session, last_step)

  def _save(self, session, step, asynchronous=True):
    # [FUNCTION DOC]
    # HELPER: core save routine of the hook.
    # KNOWN ISSUE: the docstring says it returns should_stop, but no path returns a value; it
    # always returns nothing.
    # Inputs: session, step (global step number), asynchronous (default true; false saves in the
    # calling thread). Defines a nested save function that logs, calls listener before-save
    # callbacks, saves with the saver, adds a checkpoint session log to the summary writer, calls
    # listener after-save callbacks, records the async write duration metric, records the time
    # saved metric based on the module-level last-write timestamp, then updates that timestamp,
    # and logs timings. The blocking time seen by the main thread is recorded as a metric.
    # Synchronous mode: marks the step as the last checkpoint step, runs the save inline, records
    # blocking time and returns. Asynchronous mode: if a previous save thread is still running
    # after a short join (0.1 seconds), skips this checkpoint (logs it); otherwise marks the step,
    # starts a new thread for the save and records blocking time. Raises a value error (inside the
    # save routine) if the summary writer is missing. Side effects: threads, checkpoint files,
    # summaries, metrics, and the global timestamp.
    """Saves the latest checkpoint, returns should_stop."""

    def _save_fn():
      # [FUNCTION DOC]
      # HELPER: the body executed (inline or in the save thread) by the enclosing save routine.
      # KNOWN ISSUE: the final assignment of the module-level last-write timestamp happens outside
      # the lock that protects the read just above it, so concurrent hooks could race.
      # Takes no arguments; uses the enclosing session and step. Performs the listener callbacks,
      # the checkpoint save, the summary log entry and the timing metrics described there.
      # Raises a value error if no summary writer is set. Returns nothing.
      """Run the saver process."""
      logging.info("Saving checkpoints for %d into %s.", step, self._save_path)

      start_time = time.time()
      for l in self._listeners:
        l.before_save(session, step)

      self._get_saver().save(session, self._save_path, global_step=step)
      if self._summary_writer is None:
        raise ValueError("Summary writer is not initialised")
      self._summary_writer.add_session_log(
          event_pb2.SessionLog(
              status=event_pb2.SessionLog.CHECKPOINT,
              checkpoint_path=self._save_path), step)

      for l in self._listeners:
        l.after_save(session, step)

      # Measure the async checkpoint write duration, i.e., non-blocking time.
      end_time = time.time()
      metrics.AddAsyncCheckpointWriteDuration(
          api_label=_ASYNC_CHECKPOINT_V1,
          microseconds=_get_duration_microseconds(start_time, end_time))

      # Measure the elapsed time since the last checkpoint.
      # Due to the nature of async checkpoint, here it actually captures the
      # duration between the start_time of the previous checkpoint and the start
      # time of this checkpoint. As a result, the duration of the final async
      # checkpoint is excluded, which is fine since it does not take much time.
      global _END_TIME_OF_LAST_WRITE
      with _END_TIME_OF_LAST_WRITE_LOCK:
        metrics.AddTrainingTimeSaved(
            api_label=_ASYNC_CHECKPOINT_V1,
            microseconds=_get_duration_microseconds(_END_TIME_OF_LAST_WRITE,
                                                    start_time))
      _END_TIME_OF_LAST_WRITE = start_time

      logging.info("Checkpoint actual writing time: (%.3f sec)",
                   end_time - start_time)
      logging.info("Checkpoint finished for %d into %s.", step, self._save_path)

    # Measure the checkpoint write duration that is blocking the main thread.
    blocking_start_time = time.time()
    def end_of_blocking_time():
      # [FUNCTION DOC]
      # HELPER: nested helper of the save routine. Takes no arguments. Takes the current time and
      # records the elapsed time since the save routine started as the checkpoint write duration
      # metric (the time the main thread was blocked). Returns nothing.
      blocking_end_time = time.time()
      metrics.AddCheckpointWriteDuration(
          api_label=_ASYNC_CHECKPOINT_V1,
          microseconds=_get_duration_microseconds(blocking_start_time,
                                                  blocking_end_time))

    if not asynchronous:
      self._last_checkpoint_step = step
      _save_fn()
      end_of_blocking_time()
      return

    if self._save_thread is not None:
      self._save_thread.join(timeout=0.1)
      if self._save_thread.is_alive():
        logging.info("Saver thread still in progress, skipping checkpoint.")
        end_of_blocking_time()
        return

    self._last_checkpoint_step = step
    self._save_thread = threading.Thread(target=_save_fn)
    self._save_thread.start()
    end_of_blocking_time()

  def _get_saver(self):
    # [FUNCTION DOC]
    # HELPER: finds the saver to use. Returns the saver given to the constructor if any;
    # otherwise the scaffold's saver if a scaffold was given; otherwise the single saver in the
    # graph's savers collection (cached in the hook). Raises a runtime error if that collection
    # is empty or holds more than one saver. Side effect: caches the found saver.
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
