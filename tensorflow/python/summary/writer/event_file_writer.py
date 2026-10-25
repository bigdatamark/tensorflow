# [MODULE DOC]
# Asynchronous event-file writer used by the v1 summary FileWriter. It defines EventFileWriter,
# which creates an events file in a log directory and writes Event protocol buffers to it from a
# background daemon thread fed through a bounded, closeable queue. It also defines the private
# writer thread class, the CloseableQueue (a stripped-down closeable fork of the standard library
# queue) and QueueClosedError raised when putting to a closed queue.
# Importing has no side effects beyond loading the native events writer extension.
# Consumed by the summary writer module of this package.
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

import collections
import os.path
import sys
import threading
import time

from tensorflow.python.client import _pywrap_events_writer
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat


class EventFileWriter:
  # [CLASS DOC]
  # Writes Event protocol buffers to a tfrecord events file inside a log directory. Holds the log
  # directory, queue size and flush interval, a native events writer, a closeable queue and a
  # background writer thread, plus sentinel objects used to request flush and close and an event
  # used to signal flush completion. Created with a log directory and used through add_event,
  # flush, close and reopen. Not thread-safe.
  """Writes `Event` protocol buffers to an event file.

  The `EventFileWriter` class creates an event file in the specified directory,
  and asynchronously writes Event protocol buffers to the file. The Event file
  is encoded using the tfrecord format, which is similar to RecordIO.

  This class is not thread-safe.
  """

  def __init__(self, logdir, max_queue=10, flush_secs=120,
               filename_suffix=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: logdir (directory path, converted to text), max_queue (maximum number
    # of pending events before adding blocks, default 10), flush_secs (seconds between automatic
    # flushes, default 120), filename_suffix (optional text appended to the event file name).
    # Side effects: creates the log directory if missing, creates the native events writer for a
    # file named with the events prefix, applies the suffix when given, and starts the background
    # writer thread via the initialization helper. Marks the writer as open. Returns nothing.
    """Creates a `EventFileWriter` and an event file to write to.

    On construction the summary writer creates a new event file in `logdir`.
    This event file will contain `Event` protocol buffers, which are written to
    disk via the add_event method.

    The other arguments to the constructor control the asynchronous writes to
    the event file:

    *  `flush_secs`: How often, in seconds, to flush the added summaries
       and events to disk.
    *  `max_queue`: Maximum number of summaries or events pending to be
       written to disk before one of the 'add' calls block.

    Args:
      logdir: A string. Directory where event file will be written.
      max_queue: Integer. Size of the queue for pending events and summaries.
      flush_secs: Number. How often, in seconds, to flush the
        pending events and summaries to disk.
      filename_suffix: A string. Every event file's name is suffixed with
        `filename_suffix`.
    """
    self._logdir = str(logdir)
    gfile.MakeDirs(self._logdir)
    self._max_queue = max_queue
    self._flush_secs = flush_secs
    self._flush_complete = threading.Event()
    self._flush_sentinel = object()
    self._close_sentinel = object()
    self._ev_writer = _pywrap_events_writer.EventsWriter(
        compat.as_bytes(os.path.join(self._logdir, "events")))
    if filename_suffix:
      self._ev_writer.InitWithSuffix(compat.as_bytes(filename_suffix))
    self._initialize()
    self._closed = False

  def _initialize(self):
    # [FUNCTION DOC]
    # HELPER: Sets up the queue and worker thread for the constructor and reopen.
    # Creates a fresh closeable queue of the configured size and a new writer thread bound to it,
    # the native writer, flush interval and sentinels, then starts the thread. Takes no inputs
    # and returns nothing.
    """Initializes or re-initializes the queue and writer thread.

    The EventsWriter itself does not need to be re-initialized explicitly,
    because it will auto-initialize itself if used after being closed.
    """
    self._event_queue = CloseableQueue(self._max_queue)
    self._worker = _EventLoggerThread(self._event_queue, self._ev_writer,
                                      self._flush_secs, self._flush_complete,
                                      self._flush_sentinel,
                                      self._close_sentinel)

    self._worker.start()

  def get_logdir(self):
    # [FUNCTION DOC]
    # Returns the log directory path (as text) in which the event file is written. No side effects.
    """Returns the directory where event file will be written."""
    return self._logdir

  def reopen(self):
    # [FUNCTION DOC]
    # If the writer was closed, restarts the queue and worker thread and marks it open again, so
    # new events go to a new events file in the same directory. Does nothing when not closed.
    # Takes no inputs and returns nothing.
    """Reopens the EventFileWriter.

    Can be called after `close()` to add more events in the same directory.
    The events will go into a new events file.

    Does nothing if the EventFileWriter was not closed.
    """
    if self._closed:
      self._initialize()
      self._closed = False

  def add_event(self, event):
    # [FUNCTION DOC]
    # Enqueues an Event protocol buffer for the background thread to write. Input: event. Silently
    # ignored if the writer is closed. May block while the queue is full. If the worker thread had
    # failed, the original exception is re-raised by the enqueue helper. Returns nothing.
    """Adds an event to the event file.

    Args:
      event: An `Event` protocol buffer.
    """
    if not self._closed:
      self._try_put(event)

  def _try_put(self, item):
    # [FUNCTION DOC]
    # HELPER: Enqueue primitive shared by add_event, flush and close.
    # Puts the item (an event or a sentinel) on the queue. If the queue is closed, closes the
    # writer internally and, when the worker thread recorded a failure, re-raises that exception
    # (without chaining the queue error). If no failure was recorded the error is swallowed.
    # Returns nothing.
    """Attempts to enqueue an item to the event queue.

    If the queue is closed, this will close the EventFileWriter and reraise the
    exception that caused the queue closure, if one exists.

    Args:
      item: the item to enqueue
    """
    try:
      self._event_queue.put(item)
    except QueueClosedError:
      self._internal_close()
      if self._worker.failure_exc_info:
        _, exception, _ = self._worker.failure_exc_info
        raise exception from None

  def flush(self):
    # [FUNCTION DOC]
    # Blocks until all queued events are written and flushed to disk. Does nothing if closed.
    # Clears the completion event, enqueues the flush sentinel, and waits for the worker to signal
    # completion. If the worker thread has failed, closes the writer and re-raises the worker's
    # original exception. Takes no inputs and returns nothing.
    """Flushes the event file to disk.

    Call this method to make sure that all pending events have been written to
    disk.
    """
    if not self._closed:
      # Request a flush operation by enqueuing a sentinel and then waiting for
      # the writer thread to mark the flush as complete.
      self._flush_complete.clear()
      self._try_put(self._flush_sentinel)
      self._flush_complete.wait()
      if self._worker.failure_exc_info:
        self._internal_close()
        _, exception, _ = self._worker.failure_exc_info
        raise exception

  def close(self):
    # [FUNCTION DOC]
    # Flushes pending events, enqueues the close sentinel so the worker exits, then joins the
    # worker and closes the native writer. Does nothing if already closed. May raise the worker's
    # failure exception from the flush. Takes no inputs and returns nothing.
    """Flushes the event file to disk and close the file.

    Call this method when you do not need the summary writer anymore.
    """
    if not self._closed:
      self.flush()
      self._try_put(self._close_sentinel)
      self._internal_close()

  def _internal_close(self):
    # [FUNCTION DOC]
    # HELPER: Shared teardown for close, flush and the enqueue helper.
    # Marks the writer closed, waits for the worker thread to finish, and closes the native events
    # writer. Does not flush or enqueue anything. Takes no inputs and returns nothing.
    self._closed = True
    self._worker.join()
    self._ev_writer.Close()


class _EventLoggerThread(threading.Thread):
  # [CLASS DOC]
  # HELPER: Background worker for EventFileWriter.
  # Daemon thread named for event logging that repeatedly takes items from the closeable queue and
  # writes events with the native writer, flushing periodically, on request (flush sentinel) and
  # exiting on the close sentinel. Records exception info when the loop fails so the owner can
  # re-raise it. Always unblocks pending flushes and closes the queue on exit.
  """Thread that logs events."""

  def __init__(self, queue, ev_writer, flush_secs, flush_complete,
               flush_sentinel, close_sentinel):
    # [FUNCTION DOC]
    # HELPER: Constructor of the writer thread.
    # Inputs: queue (closeable queue to read from), ev_writer (native events writer), flush_secs
    # (seconds between periodic flushes), flush_complete (threading event set after each requested
    # flush), flush_sentinel and close_sentinel (marker objects compared by identity). Initializes
    # the base thread with a fixed name, marks it as a daemon, schedules the first flush
    # immediately, and sets failure info to empty. Does not start the thread. Returns nothing.
    """Creates an _EventLoggerThread.

    Args:
      queue: A CloseableQueue from which to dequeue events. The queue will be
        closed just before the thread exits, whether due to `close_sentinel` or
        any exception raised in the writing loop.
      ev_writer: An event writer. Used to log brain events for
        the visualizer.
      flush_secs: How often, in seconds, to flush the
        pending file to disk.
      flush_complete: A threading.Event that will be set whenever a flush
        operation requested via `flush_sentinel` has been completed.
      flush_sentinel: A sentinel element in queue that tells this thread to
        flush the writer and mark the current flush operation complete.
      close_sentinel: A sentinel element in queue that tells this thread to
        terminate and close the queue.
    """
    threading.Thread.__init__(self, name="EventLoggerThread")
    self.daemon = True
    self._queue = queue
    self._ev_writer = ev_writer
    self._flush_secs = flush_secs
    # The first event will be flushed immediately.
    self._next_event_flush_time = 0
    self._flush_complete = flush_complete
    self._flush_sentinel = flush_sentinel
    self._close_sentinel = close_sentinel
    # Populated when writing logic raises an exception and kills the thread.
    self.failure_exc_info = ()

  def run(self):
    # [FUNCTION DOC]
    # Thread body. Loops forever taking items from the queue: the close sentinel ends the loop; the
    # flush sentinel flushes the writer and sets the completion event; any other item is written
    # as an event and the writer is flushed if the flush interval has elapsed. If any exception
    # occurs it is logged as an error, its exception info is stored in the failure attribute, and
    # it is re-raised. In all cases on exit it sets the completion event and closes the queue so
    # that blocked callers do not deadlock. Returns nothing.
    try:
      while True:
        event = self._queue.get()
        if event is self._close_sentinel:
          return
        elif event is self._flush_sentinel:
          self._ev_writer.Flush()
          self._flush_complete.set()
        else:
          self._ev_writer.WriteEvent(event)
          # Flush the event writer every so often.
          now = time.time()
          if now > self._next_event_flush_time:
            self._ev_writer.Flush()
            self._next_event_flush_time = now + self._flush_secs
    except Exception as e:
      logging.error("EventFileWriter writer thread error: %s", e)
      self.failure_exc_info = sys.exc_info()
      raise
    finally:
      # When exiting the thread, always complete any pending flush operation
      # (to unblock flush() calls) and close the queue (to unblock add_event()
      # calls, including those used by flush() and close()), which ensures that
      # code using EventFileWriter doesn't deadlock if this thread dies.
      self._flush_complete.set()
      self._queue.close()


class CloseableQueue:
  # [CLASS DOC]
  # Bounded first-in first-out queue shared between producer callers and the writer thread. Holds a
  # double-ended buffer, a maximum size, a closed flag, and one lock shared by two condition
  # variables (not empty, not full). Closing makes pending and future puts fail. Gets are
  # blocking and are not affected by closing.
  """Stripped-down fork of the standard library Queue that is closeable."""

  def __init__(self, maxsize=0):
    # [FUNCTION DOC]
    # Constructor. Input: maxsize, the integer capacity; zero or negative means unbounded
    # (default 0). Creates the empty buffer, the open state, the lock and the two conditions.
    # Returns nothing.
    """Create a queue object with a given maximum size.

    Args:
      maxsize: int size of queue. If <= 0, the queue size is infinite.
    """
    self._maxsize = maxsize
    self._queue = collections.deque()
    self._closed = False
    # Mutex must be held whenever queue is mutating; shared by conditions.
    self._mutex = threading.Lock()
    # Notify not_empty whenever an item is added to the queue; a
    # thread waiting to get is notified then.
    self._not_empty = threading.Condition(self._mutex)
    # Notify not_full whenever an item is removed from the queue;
    # a thread waiting to put is notified then.
    self._not_full = threading.Condition(self._mutex)

  def get(self):
    # [FUNCTION DOC]
    # Removes and returns the oldest item. Blocks while the queue is empty (even if the queue has
    # been closed, so it never raises on closure). Wakes one blocked producer after removal.
    """Remove and return an item from the queue.

    If the queue is empty, blocks until an item is available.

    Returns:
      an item from the queue
    """
    with self._not_empty:
      while not self._queue:
        self._not_empty.wait()
      item = self._queue.popleft()
      self._not_full.notify()
      return item

  def put(self, item):
    # [FUNCTION DOC]
    # Appends an item. Input: item, any object. Raises QueueClosedError immediately if the queue
    # is closed, or if it becomes closed while waiting for space in a full bounded queue. Blocks
    # while full. Wakes one waiting consumer after appending. Returns nothing.
    """Put an item into the queue.

    If the queue is closed, fails immediately.

    If the queue is full, blocks until space is available or until the queue
    is closed by a call to close(), at which point this call fails.

    Args:
      item: an item to add to the queue

    Raises:
      QueueClosedError: if insertion failed because the queue is closed
    """
    with self._not_full:
      if self._closed:
        raise QueueClosedError()
      if self._maxsize > 0:
        while len(self._queue) == self._maxsize:
          self._not_full.wait()
          if self._closed:
            raise QueueClosedError()
      self._queue.append(item)
      self._not_empty.notify()

  def close(self):
    # [FUNCTION DOC]
    # Marks the queue closed and wakes all producers blocked on a full queue so their puts fail.
    # Already queued items remain retrievable. Returns nothing.
    """Closes the queue, causing any pending or future `put()` calls to fail."""
    with self._not_full:
      self._closed = True
      self._not_full.notify_all()


class QueueClosedError(Exception):
  # [CLASS DOC]
  # Exception type raised by the closeable queue when an item is put into a closed queue. Carries
  # no extra state; caught by the event file writer to trigger shutdown handling.
  """Raised when CloseableQueue.put() fails because the queue is closed."""
