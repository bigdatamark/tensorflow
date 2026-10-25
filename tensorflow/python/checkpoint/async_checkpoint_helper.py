# [MODULE DOC]
# Implements asynchronous checkpointing: variables are copied from accelerators to host CPU
# memory synchronously, and the actual write to storage happens on a background thread so
# training can continue. Exposes the AsyncCheckpointHelper class, which wraps a checkpoint
# implementation class. Also holds module-level state: the end time of the last async write
# (shared across instances and guarded by a lock), a metrics label, and the name of an attribute
# used to recognize TPU embedding objects. Reports duration metrics. Registers an exit hook per
# helper instance when instances are created. Not a pure helper module; used by the checkpoint
# class when async checkpointing is enabled.
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for saving/loading Trackable objects asynchronously."""

import atexit
import copy
import queue
import threading
import time
import weakref

from absl import logging

from tensorflow.python.checkpoint import checkpoint_context
from tensorflow.python.checkpoint import trackable_view
from tensorflow.python.distribute import device_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import executor
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.trackable import base
from tensorflow.python.util import object_identity

# Captures the timestamp of the first Checkpoint instantiation or end of a write
# operation. Can be accessed by multiple Checkpoint instances.
_END_TIME_OF_LAST_ASYNC_WRITE = None
_END_TIME_OF_LAST_ASYNC_WRITE_LOCK = threading.Lock()

# API label for cell names used in async checkpoint metrics.
_ASYNC_CHECKPOINT = "async_checkpoint"

# Name of TPUEmbedding attribute. This is a temporary workaround
# to identify TPUEmbedding while avoiding import cycles.
_TPU_EMBEDDING_ATTR = "_create_copy_for_async_checkpoint"


def _get_duration_microseconds(start_time_seconds, end_time_seconds):
  # [FUNCTION DOC]
  # HELPER: converts a pair of timestamps in seconds to a duration for metrics.
  # Returns the rounded difference in microseconds, or zero if the end precedes the start
  # (clock skew). No side effects.
  """Calculate the duration between start and end time.

  Args:
    start_time_seconds: The start time in seconds.
    end_time_seconds: The end time in seconds.

  Returns:
    The duration between the start and the end time. Return 0 if
    end_time_seconds < start_time_seconds.
  """
  if end_time_seconds < start_time_seconds:
    # Avoid returning negative value in case of clock skew.
    return 0
  return round((end_time_seconds - start_time_seconds) * 1000000)


def _get_all_trackables(root, exclude_set):
  # [FUNCTION DOC]
  # HELPER: gathers the trackables to copy for async checkpointing.
  # Takes the descendants of root (a trackable), removes the objects in exclude_set (an identity
  # set; items found are removed from it, so it is mutated and the scan stops early once it is
  # empty), and then selects those that need saving, meaning their own namespace or any class in
  # their inheritance chain (except the base trackable class) defines a serialize, gather-saveables
  # or copy-to-cpu hook. Returns a pair: the list of saveable trackables, and the list of all
  # remaining trackables.
  """Return the list of checkpointable trackables dependent on `root`.

  Args:
    root: The root trackable from where we get all its dependent trackables.
    exclude_set: An ObjectIdentitySet of Trackables to exclude before returning.
        Each element in `exclude_set` is a specific instance of a `Trackable`
        and appears precisely once in `TrackableView(root).descendants()`.

  Returns:
    saveable_trackables: All trackables that are saveable in `all_trackables`
        (see definition of "saveable" in `_trackable_needs_to_be_saved()`). A
        subset of `all_trackables`.
    all_trackables: All trackables returned by `TrackableView`'s `descendants()`
        after excluding `exclude_set`. A superset of `saveable_trackables`.
  """
  all_trackables = trackable_view.TrackableView(root=root).descendants()

  # Kick out the trackable we want to exclude.
  # The goal of writing such loop is to only scan the list once and stop
  # scanning as early as possible (unlike filtering with list comprehension).
  trackable_index = 0
  while trackable_index < len(all_trackables) and exclude_set:
    # While we have not excluded all items, or gone through all trackables.
    if all_trackables[trackable_index] in exclude_set:
      # If want to exclude this trackable, we pop it and do not update ptr
      exclude_set.discard(all_trackables[trackable_index])
      all_trackables.pop(trackable_index)
    else:
      # Otherwise update ptr
      trackable_index += 1

  # Kick out trackables that do not need to be saved (e.g. ListWrapper, etc.)
  # We define any trackable that does not implement `_serialize_to_tensor` or
  # `_gather_saveables` as "no need to be saved". If the trackable has one or
  # both of the methods defined, it should have `_copy_trackable_to_cpu`
  # defined; if not, we will raise warning in `_copy_to_cpu()`. In case of
  # special case, we also check whether a trackable (who has neither of the
  # other two methods defined) defines `_copy_trackable_to_cpu` only; we still
  # define such cases as "needs to be saved".
  def _trackable_needs_to_be_saved(obj):
    # [FUNCTION DOC]
    # HELPER: predicate used to filter trackables. Returns true if the object's own attributes
    # (when it has any) or any class in its inheritance chain other than the base trackable
    # class defines one of the three hooks: serialize-to-tensors, gather-saveables-for-checkpoint
    # or copy-trackable-to-cpu; otherwise false. Input obj: a trackable. No side effects.
    """Returns whether a trackable needs to be saved.

    Returns a bool to indicate whether obj's class has `_serialize_to_tensors`,
    `gather_saveables_for_checkpoint`, or `_copy_trackable_to_cpu` defined.

    Args:
      obj: A Trackable object.
    """
    if hasattr(obj, "__dict__"):
      # Data structure proxy wrappers don't have __dict__.
      if ("_serialize_to_tensors" in obj.__dict__
          or "_gather_saveables_for_checkpoint" in obj.__dict__
          or "_copy_trackable_to_cpu" in obj.__dict__):
        return True

    # Use MRO so that if a parent class has one of the three methods, we still
    # consider `t` as needed to be saved.
    for t in type(obj).mro():
      if t is base.Trackable:
        # Base class always has them implemented, but would raise error.
        continue
      elif ("_serialize_to_tensors" in t.__dict__
            or "_gather_saveables_for_checkpoint" in t.__dict__
            or "_copy_trackable_to_cpu" in t.__dict__):
        return True

    return False

  saveable_trackables = [x for x in all_trackables if
                         _trackable_needs_to_be_saved(x)]

  return saveable_trackables, all_trackables


class AsyncCheckpointHelper:
  # [CLASS DOC]
  # Wraps a checkpoint implementation class to provide asynchronous save. On save/write it waits
  # for the previous async save, copies variable values to host CPU (using copies kept in an
  # object map), then signals a background daemon thread through a one-slot queue to write the
  # CPU copies to disk. Errors from the thread are stored and re-raised on the next call. Also
  # offers read/restore (delegating to the underlying checkpoint after waiting for pending
  # saves), sync, and a save counter. Special handling exists for TPU embedding objects and
  # optimizer slot variables. Created with the checkpoint class, an optional root, and keyword
  # items; registers an exit hook to join the thread.
  """Helper class for async checkpoint."""

  def __init__(self, checkpointer_impl, root=None, **kwargs):
    # [FUNCTION DOC]
    # Constructor. checkpointer_impl: checkpoint class used to build the underlying checkpoint
    # (required). root: optional trackable or weak reference to one; added to the checkpoint
    # items under the root key and initialized. kwargs: other items to checkpoint. Raises an
    # attribute error if checkpointer_impl is null. Creates the underlying checkpoint, an empty
    # state (not yet initialized), a default device (current or CPU 0, canonicalized), a queue of
    # size one, registers an exit hook joining the async thread, and sets the module-wide last
    # write time to now if unset. Note: root handling occurs before the null check on
    # checkpointer_impl.
    """Initialize AsyncCheckpoint.

    Args:
      checkpointer_impl: The Checkpoint class to power the AsyncCheckpoint.
      root: The root object to checkpoint. `root` may be a trackable object or
        `WeakRef` of a trackable object.
      **kwargs: The keyword arguments representing the checkpointed variables.

    Raises:
      AttributeError: when checkpointer_impl is None.
    """
    # TODO(chienchunh): Make sure the processing for the root object is
    #   consistent when integrating with the public API, e.g., adding all kwarg
    #   items as the child of the root object.
    if root:
      trackable_root = root() if isinstance(root, weakref.ref) else root
      kwargs["root"] = trackable_root
      trackable_root._maybe_initialize_trackable()

    # The underlying Checkpoint instance and its items.
    if checkpointer_impl is None:
      raise AttributeError(
          "checkpointer_impl cannot be None for AsyncCheckpointHelper."
      )
    self._checkpointer_impl = checkpointer_impl
    self._checkpoint_items = kwargs
    self._checkpoint = None
    self.checkpointer()
    self._checkpoint_options = None

    # Indicate whether async checkpoint has finished traversing the variable
    # list and created the object map between the original and copied variables.
    self._initialized = False

    # The list of all nodes from the original checkpoint items.
    # TODO(chienchunh): Consider changing this to local variable.
    self._original_nodes = None
    # The mapping between the original and the copied resource variables.
    # The copied variables are used for the underlying checkpointing.
    self._object_map = None
    # A list of TPUEmbedding objects included in the checkpoint items.
    self._tpu_embedding_objects = None
    # A list of highest level `Trackable`s we will copy; does not contain
    # TPUEmbedding objects
    self._saveable_trackables = None

    self._default_device = device_util.current() or "CPU:0"
    self._default_device = device_util.canonicalize(self._default_device)

    self._save_file_prefix = None
    self._use_checkpoint_save = False
    self._async_save_thread = None
    # Concurrent queue that coordinates the events for writing/reading the
    # cpu-copied variables. A 'True' in the queue triggers the async thread to
    # perform saving; a 'False' breaks the while loop so that the async thread
    # exits; no other values will be added to the queue.
    # Maxsize is set to 1 only to ensure the exit procedure. We could have used
    # queue.join() in _join_async_save_thread(), but queue.join() does not have
    # a timeout argument. Hence we use queue.put(timeout=300), in case the last
    # checkpoint takes forever. To achieve that, maxsize needs to be 1.
    self._queue = queue.Queue(maxsize=1)

    # Register to join the async save thread upon exit.
    atexit.register(self._join_async_save_thread)

    self._async_error = None

    global _END_TIME_OF_LAST_ASYNC_WRITE
    with _END_TIME_OF_LAST_ASYNC_WRITE_LOCK:
      if _END_TIME_OF_LAST_ASYNC_WRITE is None:
        _END_TIME_OF_LAST_ASYNC_WRITE = time.time()

  @def_function.function
  def _copy_to_cpu(self):
    # [FUNCTION DOC]
    # HELPER: compiled function (decorated) that refreshes the host CPU copies. Asks each
    # saveable trackable to copy itself into the object map (a warning is logged and the item
    # skipped if it does not support copying), then retrieves variables of each TPU embedding
    # object. Returns nothing. Requires initialization to have populated the lists.
    """Copy the checkpointed variables from the accelerator to the host CPU.

    TODO(chienchunh): Get the concrete function before firstly called to avoid
                      hangining the accelerators idle during function tracing.
    """
    for t in self._saveable_trackables:
      try:
        t._copy_trackable_to_cpu(object_map=self._object_map)  # pylint: disable=protected-access
      except NotImplementedError as e:
        logging.warning("Trackable %s skipped due to: %s", t, e)

    for tpu_embedding in self._tpu_embedding_objects:
      tpu_embedding._retrieve_variables()  # pylint: disable=protected-access

  def checkpointer(self):
    # [FUNCTION DOC]
    # Returns the underlying checkpoint object, creating it on first use from the stored
    # checkpoint class and items.
    """Gets or creates the underlying Checkpoint instance."""
    if self._checkpoint is None:
      self._checkpoint = self._checkpointer_impl(**self._checkpoint_items)
    return self._checkpoint

  def _ensure_initialized(self):
    # [FUNCTION DOC]
    # HELPER: one-time setup run on the first save. Builds the object map and TPU embedding list,
    # collects saveable trackables of the checkpoint (excluding the checkpoint and its save
    # counter), handles TPU embeddings and optimizer slot variables (adding slot variables to
    # the saveable list), logs the save counter, hands the object map to the checkpoint's saver,
    # performs an initial copy to CPU (done by hand since the compiled copy function would hit
    # an out-of-scope error), starts the daemon async save thread, and marks the helper
    # initialized. Returns nothing.
    """Initialize the async checkpoint internal state."""
    # This map will be used to store the CPU copy of all checkpointable objects
    self._object_map = object_identity.ObjectIdentityDictionary()
    self._tpu_embedding_objects = []

    # Populate self._all_tracakbles, but exclude the checkpoint instance itself
    # and its save_counter, as they will be returned by `descendants()`.
    exclude_set = object_identity.ObjectIdentitySet()
    exclude_set.add(self.checkpointer())
    exclude_set.add(self.checkpointer().save_counter)
    self._saveable_trackables, all_trackables = _get_all_trackables(
        root=self.checkpointer(), exclude_set=exclude_set)

    # Handle special cases: TPU Embedding, and slot variables.
    # 1. TPUEmbedding: Different from other trackables, TPUEmbedding needs to
    # call `_retrieve_variables` to checkpoint, while populating a dummy copy to
    # the object map.
    # 2. Slot variables: they need to be handled differently as they cannot be
    # retrieved from `TrackableView.descendants()`.

    # Note: dir() is used rather than hasattr() here to avoid triggering
    # custom __getattr__ code, see b/152031870 for context.
    for t in all_trackables:
      # Special case 1: TPU Embedding, populate object_map here
      # Special case 1: Handle TPU Embedding by addnig a dummy instance to the
      # object map. Also add TPUEmbedding to separate list for special handling
      # with values copy.
      if hasattr(type(t), _TPU_EMBEDDING_ATTR):
        self._handle_tpu_embedding(t)
      # Special case 2: handle slot variables. The object_map is populated later
      # when the variable values are being copied to host CPU for the first
      # time.
      if "get_slot_names" in dir(t):
        slot_names = t.get_slot_names()
        for slot_name in slot_names:
          for original_variable in all_trackables:
            if not isinstance(original_variable, variables.Variable):
              continue
            try:
              # Usage of hasattr may result in KeyError
              original_slot_variable = t.get_slot(original_variable, slot_name)
            except (AttributeError, KeyError):
              continue
            if isinstance(original_slot_variable, base.Trackable):
              self._saveable_trackables.append(original_slot_variable)

    # Initiate the underlying Checkpoint instance's save_counter.
    save_counter = self.checkpointer().save_counter.numpy()
    logging.info("Initializing async checkpoint's save_counter: %d",
                 save_counter)

    # Pass the object map of the copied variables to the underlying Checkpoint.
    self.checkpointer()._saver._object_map = self._object_map  # pylint: disable=protected-access

    # We perform a `_copy_to_cpu()` to populate `self._object_map`,
    # initializing copies. We do not call `self._copy_to_cpu()` directly
    # because it is a tf function, which leads to access out of scope error.

    # TODO(charlieruan) Figure out a better work around to solve the access
    # out of scope error.
    for t in self._saveable_trackables:
      try:
        t._copy_trackable_to_cpu(object_map=self._object_map)  # pylint: disable=protected-access
      except NotImplementedError as e:
        logging.warning("Trackable %s skipped due to: %s", t, e)

    for tpu_embedding in self._tpu_embedding_objects:
      tpu_embedding._retrieve_variables()  # pylint: disable=protected-access

    # Initiate the async thread for checkpoint saving.
    self._async_save_thread = threading.Thread(
        target=self._async_save, daemon=True)
    self._async_save_thread.start()

    self._initialized = True

  def _check_async_thread_error(self):
    # [FUNCTION DOC]
    # HELPER: if the async thread stored an error, clears it, logs it, and raises it in the
    # caller. Otherwise does nothing.
    """Expose the most recent error from the async saving thread to the caller.
    """
    if self._async_error:
      e = self._async_error
      self._async_error = None
      logging.error("Propagating the most recent error from the async thread "
                    "before joining: %s", str(e))
      raise e

  def _join_async_save_thread(self):
    # [FUNCTION DOC]
    # HELPER: shutdown routine, also registered as an exit hook. Puts a stop signal on the queue
    # (waiting up to 300 seconds for the last save to finish), joins the thread if one exists,
    # logs an error on timeout (last checkpoint may be incomplete), and always finally re-raises
    # any stored async error. Returns nothing.
    """Join the async save thread.

    The steps for terminating the async save thread:
    1). Put will succeed when the last async save event is done. Putting a false
        triggers the async save thread's while loop to end. We use put instead
        of sync because sync does not have a timeout argument.
    2). Join the async save thread. (The thread may finish before joining.)
    """
    try:
      self._queue.put(False, timeout=300)  # Step-1.
      logging.info("Joining the async save thread.")
      if self._async_save_thread is not None:
        self._async_save_thread.join()  # Step-2.
    except queue.Full:
      logging.error("Timeout waiting for the async save thread; terminating the"
                    " thread instead. The last checkpoint may be incomeplete.")
    finally:
      self._check_async_thread_error()

  def _async_save(self):
    # [FUNCTION DOC]
    # HELPER: body of the background thread. Within a synchronous executor scope, loops reading
    # the queue until a false signal arrives. For each true signal it writes the checkpoint on
    # the default device within the async metrics context: using full save (with counter) if the
    # flag set by save is on, else the write-only path. Exceptions are stored for later
    # propagation; the queue task is always marked done. Afterwards it records write-duration and
    # training-time-saved metrics and updates the module-wide last-write timestamp under a lock.
    # Returns nothing when the loop ends.
    """The thread function for the async checkpoint save."""
    with context.executor_scope(
        executor.new_executor(
            enable_async=False, enable_streaming_enqueue=False)):
      # The main thread inserts: a True to the queue when the user calls save,
      # triggering async save; and a False when we exit the Checkpoint instance.
      while self._queue.get():
        logging.info("Starting async checkpoint save on the device: %s",
                     self._default_device)

        async_save_start_time = time.time()

        # Specify the ops placement on the worker if running with
        # coordinator-worker mode. This is required as launching a new thread
        # would clear the placement policy and make localhost the default
        # placement, while the main thread's default placement would be the
        # master worker's CPU:0.
        try:
          with ops.device(self._default_device):
            with checkpoint_context.async_metrics_context():
              if self._use_checkpoint_save:
                self.checkpointer().save(
                    self._save_file_prefix, self._checkpoint_options
                )
              else:
                self.checkpointer()._write(  # pylint: disable=protected-access
                    self._save_file_prefix,
                    options=self._checkpoint_options,
                )
        except Exception as e:   # # pylint: disable=broad-except
          self._async_error = e
        finally:
          self._queue.task_done()

        async_save_end_time = time.time()
        metrics.AddAsyncCheckpointWriteDuration(
            api_label=_ASYNC_CHECKPOINT,
            microseconds=_get_duration_microseconds(async_save_start_time,
                                                    async_save_end_time))

        # Measure the elapsed time since the last checkpoint.
        # Due to the nature of async checkpoint, here it actually captures the
        # duration between the start_time of the previous checkpoint and the
        # start time of this checkpoint. As a result, the duration of the final
        # async checkpoint is excluded, which is fine since it does not take
        # much time.
        global _END_TIME_OF_LAST_ASYNC_WRITE
        with _END_TIME_OF_LAST_ASYNC_WRITE_LOCK:
          metrics.AddTrainingTimeSaved(
              api_label=_ASYNC_CHECKPOINT,
              microseconds=_get_duration_microseconds(
                  _END_TIME_OF_LAST_ASYNC_WRITE, async_save_start_time))
          _END_TIME_OF_LAST_ASYNC_WRITE = async_save_start_time
    logging.info("Async save thread reached the end of the execution.")

  def _handle_tpu_embedding(self, tpu_embedding):
    # [FUNCTION DOC]
    # HELPER: registers a TPU embedding object. Raises an attribute error if the object's type
    # lacks the copy-for-async hook or it is not callable. Creates a dummy copy (using the
    # original's feature config, first table config as optimizer or nothing, and pipeline
    # setting), stores it in the object map so the save callback is not triggered, and appends
    # the original to the TPU embedding list if absent. Returns nothing.
    """Handle TPUEmbedding.

    This is the only place where we populate object map in the class of
    `AsyncCheckpointHelper`. For all other checkpointable trackables, we
    populate object map using the trackable's own `_copy_trackable_to_cpu()`.

    Args:
      tpu_embedding: TPUEmbedding object to be handled.

    Raises:
      AttributeError: if the input trackable is not TPUEmbedding type.
    """
    if not hasattr(type(tpu_embedding), _TPU_EMBEDDING_ATTR) or not callable(
        tpu_embedding._create_copy_for_async_checkpoint  # pylint: disable=protected-access
    ):
      raise AttributeError(
          "Expecting TPUEmbedding type; got %s" % type(tpu_embedding)
      )

    # Create a dummy TPUEmbedding object and add it to the object_map. This is
    # to prevent the TPUEmbedding's save_callback from being triggered because
    # the embedding values have already being retrieved by AsyncCheckpoint.
    # pylint: disable=protected-access
    new_embedding = tpu_embedding._create_copy_for_async_checkpoint(
        feature_config=tpu_embedding._feature_config,
        optimizer=tpu_embedding._table_config[0]
        if tpu_embedding._table_config
        else None,
        pipeline_execution_with_tensor_core=tpu_embedding._pipeline_execution_with_tensor_core,
    )
    self._object_map[tpu_embedding] = new_embedding
    # pylint: enable=protected-access

    if tpu_embedding not in self._tpu_embedding_objects:
      self._tpu_embedding_objects.append(tpu_embedding)

  @property
  def save_counter(self):
    # [FUNCTION DOC]
    # Property returning the underlying checkpoint's save counter variable.
    """An integer variable numbering the checkpoint events.

    This is maintained by the underlying tf.train.Checkpoint object employed by
    AsyncCheckpoint class. The number starts at 0 and gets incremented for each
    checkpoint event.

    Returns:
      The save counter variable.
    """
    return self.checkpointer().save_counter

  def write(self, save_path, options=None):
    # [FUNCTION DOC]
    # Public wrapper for the write path: saves without incrementing the save counter. Inputs
    # save_path (file prefix) and optional options. Returns the path given.
    """Save the checkpointed variables.

    Args:
      save_path: The file prefix of the checkpoint file.
      options: Optional CheckpointOption instance.

    Returns:
      The full path of the checkpoint file.
    """
    return self._write(save_path, options)

  def _write(self, save_path, options=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it returns the full checkpoint path, but it returns
    # save_path unchanged (no counter suffix is added).
    # Starts an asynchronous write. First initializes on first use, or else waits for the
    # previous save and refreshes CPU copies; re-raises any stored thread error; waits for async
    # copies to finish; records the prefix and sets the flag so the thread uses the write-only
    # path; copies options with async checkpointing turned off (avoids recursion); signals the
    # thread via the queue; records write-duration metrics. Returns save_path.
    """Save the checkpointed variables.

    This method has exactly the same logic as save(), except it does not
    increment the underlying save_counter, which is done by the caller, e.g.,
    CheckpointManager.

    Args:
      save_path: The file prefix of the checkpoint file.
      options: Optional CheckpointOption instance.

    Returns:
      The full path of the checkpoint file.
    """
    write_start_time = time.time()

    if not self._initialized:
      self._ensure_initialized()
    else:
      # First wait for async thread to finish the previous save, then copy the
      # variable values to the host CPU.
      self._queue.join()
      self._copy_to_cpu()

    # Surface the error from the async thread, if any.
    # This step should come after the sem acquisition step in the above, so that
    # it makes sure it waits until the previous async save finishes storing the
    # error.
    self._check_async_thread_error()

    # Trigger the async thread to checkpoint the cpu-copied variables.
    # Need to wait until the weight copying finishes before checkpoint save.
    context.async_wait()
    self._save_file_prefix = save_path
    self._use_checkpoint_save = False

    # Ensure that we do not request async checkpointing to the underlying
    # checkpointer as this could lead to an infinite loop.
    self._checkpoint_options = copy.copy(options) if options else None
    if self._checkpoint_options:
      self._checkpoint_options.experimental_enable_async_checkpoint = False

    self._queue.put(True)  # Trigger save in async thread

    write_end_time = time.time()
    metrics.AddCheckpointWriteDuration(
        api_label=_ASYNC_CHECKPOINT,
        microseconds=_get_duration_microseconds(write_start_time,
                                                write_end_time))

    return save_path

  def save(self, save_path, options=None):
    # [FUNCTION DOC]
    # Starts an asynchronous full save. Same flow as the write path (initialize or wait and
    # re-copy, surface thread errors, wait for copies, copy options with async disabled, signal
    # the thread) but sets the flag so the thread calls the checkpoint's save, which increments
    # the counter. Predicts the final path from the current save counter plus one, joined to
    # save_path with a dash, before triggering. Returns that predicted full path.
    """Save the checkpointed variables.

    Args:
      save_path: The file prefix of the checkpoint file.
      options: Optional CheckpointOption instance.

    Returns:
      The full path of the checkpoint file.
    """
    save_start_time = time.time()

    # If this is the first time that AsyncCheckpoint.save() is called,
    # initialize the internal states like `self._saveable_trackables`. We also
    # populate `self._object_map` (i.e. initializing the cpu-copied variables
    # and copy over the value for the first time) by essentially performing a
    # `self._copy_to_cpu()`, hence the if-else logic here.
    #
    # This is not performed in the initializer because some variables, e.g.,
    # slot variables of the optimizer, were not created until actually running
    # the train function, so we could only get the complete list of the
    # variables after some train steps were run.
    if not self._initialized:
      self._ensure_initialized()
    else:
      # First wait for async thread to finish the previous save, then copy the
      # variable values to the host CPU.
      self._queue.join()
      self._copy_to_cpu()

    # Surface the error from the async thread, if any.
    # This step should come after the sem acquisition step in the above, so that
    # it makes sure it waits until the previous async save finishes storing the
    # error.
    self._check_async_thread_error()

    # Retrieve the save counter from the underlying checkpoint object to
    # re-construct the full path of the checkpoint file.
    # This step has to happen before triggering the underlying checkpoint;
    # otherwise, the save_counter value may or may not have been updated.
    save_counter = self.checkpointer().save_counter.numpy() + 1
    full_path = "{}-{}".format(save_path, save_counter)

    # Trigger the async thread to checkpoint the cpu-copied variables.
    # Need to wait until the weight copying finishes before checkpoint save.
    context.async_wait()
    self._save_file_prefix = save_path
    self._use_checkpoint_save = True

    # Ensure that we do not request async checkpointing to the underlying
    # checkpointer as this could lead to an infinite loop.
    self._checkpoint_options = copy.copy(options) if options else None
    if self._checkpoint_options:
      self._checkpoint_options.experimental_enable_async_checkpoint = False

    self._queue.put(True)  # Trigger save in async thread

    save_end_time = time.time()
    metrics.AddCheckpointWriteDuration(
        api_label=_ASYNC_CHECKPOINT,
        microseconds=_get_duration_microseconds(save_start_time, save_end_time))

    return full_path

  def read(self, save_path, options=None):
    # [FUNCTION DOC]
    # Alias of restore, present to match the checkpoint interface. Returns the load status from
    # restore.
    """Restore the checkpointed variables.

    This method has exactly the same logic as restore(). This method is
    implemented only to fulfill the duty of subclassing tf.train.Checkpoint.

    Args:
      save_path: The full name of the checkpoint file to be restored.
      options: CheckpointOption instance.

    Returns:
      A load status object, which can be used to make assertions about the
      status of a checkpoint restoration. See tf.train.Checkpoint.restore()
      for more details.
    """
    return self.restore(save_path, options)

  def restore(self, save_path, options=None):
    # [FUNCTION DOC]
    # Restores from save_path. Stores a copy of options (or keeps the previous ones if none
    # given) with async checkpointing disabled, waits for ongoing saves to finish, then restores
    # through the underlying checkpoint. Returns its load status object.
    """Restore the checkpointed variables.

    Args:
      save_path: The full name of the checkpoint file to be restored.
      options: CheckpointOption instance.

    Returns:
      A load status object, which can be used to make assertions about the
      status of a checkpoint restoration. See tf.train.Checkpoint.restore()
      for more details.
    """
    # Ensure that we do not request async checkpointing to the underlying
    # checkpointer as this could lead to an infinite loop.
    self._checkpoint_options = (
        copy.copy(options) if options else self._checkpoint_options)
    if self._checkpoint_options:
      self._checkpoint_options.experimental_enable_async_checkpoint = False

    # Wait for any ongoing checkpoint event to finish.
    self._queue.join()
    # Restore values of the cpu-copied variables directly back to accelerators
    status = self.checkpointer().restore(save_path, self._checkpoint_options)

    return status

  def sync(self):
    # [FUNCTION DOC]
    # Blocks until all queued save work is done, then logs. Returns nothing.
    """Sync on any ongoing save or restore events."""
    self._queue.join()
    logging.info("Sync on ongoing save/restore.")
