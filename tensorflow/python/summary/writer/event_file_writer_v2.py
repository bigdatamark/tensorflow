# [MODULE DOC]
# Graph-based variant of the event file writer. Defines EventFileWriterV2, which writes Event
# protocol buffers through a shared v2 summary writer resource and graph operations executed in a
# session, instead of using a native writer and background thread. Presents the same interface as
# the thread-based writer (get log dir, reopen, add event, flush, close).
# No import-time side effects other than imports.
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
"""Writes events to disk in a logdir."""

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.platform import gfile


class EventFileWriterV2(object):
  # [CLASS DOC]
  # Event file writer backed by the v2 summary file writer resource. Holds the session, log
  # directory, a closed flag, a string placeholder for serialized events, and graph operations for
  # adding an event, initializing, flushing and closing the resource. Every stateful method runs
  # these operations in the session. Created with a session and log directory.
  """Writes `Event` protocol buffers to an event file via the graph.

  The `EventFileWriterV2` class is backed by the summary file writer in the v2
  summary API (currently in tf.contrib.summary), so it uses a shared summary
  writer resource and graph ops to write events.

  As with the original EventFileWriter, this class will asynchronously write
  Event protocol buffers to the backing file. The Event file is encoded using
  the tfrecord format, which is similar to RecordIO.
  """

  def __init__(self, session, logdir, max_queue=10, flush_secs=120,
               filename_suffix=''):
    # [FUNCTION DOC]
    # Constructor. Inputs: session (v1 session whose graph holds the writer ops), logdir (directory
    # path), max_queue (pending event capacity, default 10), flush_secs (seconds between flushes,
    # default 120, converted to milliseconds), filename_suffix (text appended to file names,
    # default empty). Side effects: creates the log directory, adds a file writer resource, a
    # placeholder, and import-event, init, flush and close operations to the session graph under a
    # name scope, and runs the init operation in the session. Marks the writer open. Returns
    # nothing.
    """Creates an `EventFileWriterV2` and an event file to write to.

    On construction, this calls `tf.contrib.summary.create_file_writer` within
    the graph from `session.graph` to look up a shared summary writer resource
    for `logdir` if one exists, and create one if not. Creating the summary
    writer resource in turn creates a new event file in `logdir` to be filled
    with `Event` protocol buffers passed to `add_event`. Graph ops to control
    this writer resource are added to `session.graph` during this init call;
    stateful methods on this class will call `session.run()` on these ops.

    Note that because the underlying resource is shared, it is possible that
    other parts of the code using the same session may interact independently
    with the resource, e.g. by flushing or even closing it. It is the caller's
    responsibility to avoid any undesirable sharing in this regard.

    The remaining arguments to the constructor (`flush_secs`, `max_queue`, and
    `filename_suffix`) control the construction of the shared writer resource
    if one is created. If an existing resource is reused, these arguments have
    no effect.  See `tf.contrib.summary.create_file_writer` for details.

    Args:
      session: A `tf.compat.v1.Session`. Session that will hold shared writer
        resource. The writer ops will be added to session.graph during this
        init call.
      logdir: A string. Directory where event file will be written.
      max_queue: Integer. Size of the queue for pending events and summaries.
      flush_secs: Number. How often, in seconds, to flush the
        pending events and summaries to disk.
      filename_suffix: A string. Every event file's name is suffixed with
        `filename_suffix`.
    """
    self._session = session
    self._logdir = logdir
    self._closed = False
    gfile.MakeDirs(self._logdir)

    with self._session.graph.as_default():
      with ops.name_scope('filewriter'):
        file_writer = summary_ops_v2.create_file_writer(
            logdir=self._logdir,
            max_queue=max_queue,
            flush_millis=flush_secs * 1000,
            filename_suffix=filename_suffix)
        with summary_ops_v2.always_record_summaries(), file_writer.as_default():
          self._event_placeholder = array_ops.placeholder_with_default(
              constant_op.constant('unused', dtypes.string),
              shape=[])
          self._add_event_op = summary_ops_v2.import_event(
              self._event_placeholder)
        self._init_op = file_writer.init()  # pylint: disable=assignment-from-no-return
        self._flush_op = file_writer.flush()  # pylint: disable=assignment-from-no-return
        self._close_op = file_writer.close()  # pylint: disable=assignment-from-no-return
      self._session.run(self._init_op)

  def get_logdir(self):
    # [FUNCTION DOC]
    # Returns the log directory given at construction. No side effects.
    """Returns the directory where event file will be written."""
    return self._logdir

  def reopen(self):
    # [FUNCTION DOC]
    # If closed, clears the closed flag and re-runs the init operation in the session so a new
    # events file is created. Does nothing if not closed. Returns nothing.
    """Reopens the EventFileWriter.

    Can be called after `close()` to add more events in the same directory.
    The events will go into a new events file.

    Does nothing if the EventFileWriter was not closed.
    """
    if self._closed:
      self._closed = False
      self._session.run(self._init_op)

  def add_event(self, event):
    # [FUNCTION DOC]
    # Input: event, an Event protocol buffer. Unless closed, serializes it and runs the import
    # event operation in the session, feeding the bytes through the placeholder. Silently ignored
    # when closed. Session errors propagate. Returns nothing.
    """Adds an event to the event file.

    Args:
      event: An `Event` protocol buffer.
    """
    if not self._closed:
      event_pb = event.SerializeToString()
      self._session.run(
          self._add_event_op, feed_dict={self._event_placeholder: event_pb})

  def flush(self):
    # [FUNCTION DOC]
    # Runs the flush operation in the session so pending events reach disk. Unlike the thread-based
    # writer, it does not check the closed flag, so calling it after close runs the operation on
    # the closed resource. Returns nothing.
    """Flushes the event file to disk.

    Call this method to make sure that all pending events have been written to
    disk.
    """
    self._session.run(self._flush_op)

  def close(self):
    # [FUNCTION DOC]
    # If not already closed, flushes, runs the close operation in the session and marks the writer
    # closed. Does nothing if already closed. Returns nothing.
    """Flushes the event file to disk and close the file.

    Call this method when you do not need the summary writer anymore.
    """
    if not self._closed:
      self.flush()
      self._session.run(self._close_op)
      self._closed = True
