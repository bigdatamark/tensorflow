# [MODULE DOC]
# HELPER: backward-compatibility shim kept so old code importing summary writing names from the
# training package keeps working.
# Legacy module for reading and writing summary event files. It re-exports the summary iterator
# function, the cache of file writers (under the name SummaryWriterCache), and defines a
# deprecated SummaryWriter class that is only a renamed file writer. At import time it only
# performs imports and class definition; the deprecation decorator emits a warning when the
# constructor is later used. New code should use the summary package's file writer instead.
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
"""Reads Summaries from and writes Summaries to event files."""

# pylint: disable=unused-import
from tensorflow.python.summary.summary_iterator import summary_iterator
from tensorflow.python.summary.writer.writer import FileWriter as _FileWriter
from tensorflow.python.summary.writer.writer_cache import FileWriterCache as SummaryWriterCache
# pylint: enable=unused-import
from tensorflow.python.util.deprecation import deprecated


class SummaryWriter(_FileWriter):
  # [CLASS DOC]
  # HELPER: deprecated alias kept for backward compatibility.
  # Deprecated subclass of the summary file writer that adds no state or behavior of its own;
  # it only forwards construction arguments to the parent. Instances write summaries, graphs
  # and events asynchronously to an event file in a log directory, with all methods inherited
  # from the parent class. The deprecation decorator marks the constructor as deprecated since
  # late 2016.

  @deprecated("2016-11-30",
              "Please switch to tf.summary.FileWriter. The interface and "
              "behavior is the same; this is just a rename.")
  def __init__(self,
               logdir,
               graph=None,
               max_queue=10,
               flush_secs=120,
               graph_def=None):
    # [FUNCTION DOC]
    # HELPER: thin deprecated forwarding constructor for the legacy writer class.
    # Creates the writer and its event file by passing every argument unchanged to the parent
    # constructor. Inputs: logdir (directory path string where the event file is created),
    # graph (optional graph to record immediately), max_queue (pending event queue size,
    # default 10), flush_secs (seconds between flushes to disk, default 120), graph_def
    # (deprecated older way of passing a graph description, optional). Returns nothing.
    # Side effects: creates the directory and event file via the parent, starts its background
    # writing, and triggers a deprecation warning through the decorator.
    """Creates a `SummaryWriter` and an event file.

    This class is deprecated, and should be replaced with tf.summary.FileWriter.

    On construction the summary writer creates a new event file in `logdir`.
    This event file will contain `Event` protocol buffers constructed when you
    call one of the following functions: `add_summary()`, `add_session_log()`,
    `add_event()`, or `add_graph()`.

    If you pass a `Graph` to the constructor it is added to
    the event file. (This is equivalent to calling `add_graph()` later).

    TensorBoard will pick the graph from the file and display it graphically so
    you can interactively explore the graph you built. You will usually pass
    the graph from the session in which you launched it:

    ```python
    ...create a graph...
    # Launch the graph in a session.
    sess = tf.compat.v1.Session()
    # Create a summary writer, add the 'graph' to the event file.
    writer = tf.compat.v1.summary.FileWriter(<some-directory>, sess.graph)
    ```

    The other arguments to the constructor control the asynchronous writes to
    the event file:

    *  `flush_secs`: How often, in seconds, to flush the added summaries
       and events to disk.
    *  `max_queue`: Maximum number of summaries or events pending to be
       written to disk before one of the 'add' calls block.

    Args:
      logdir: A string. Directory where event file will be written.
      graph: A `Graph` object, such as `sess.graph`.
      max_queue: Integer. Size of the queue for pending events and summaries.
      flush_secs: Number. How often, in seconds, to flush the
        pending events and summaries to disk.
      graph_def: DEPRECATED: Use the `graph` argument instead.
    """
    super(SummaryWriter, self).__init__(logdir, graph, max_queue, flush_secs,
                                        graph_def)
