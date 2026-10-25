# [MODULE DOC]
# Version 1 summary writer API. Defines SummaryToEventTransformer, which converts summaries, session
# logs, graphs, meta graphs and run metadata into Event protocol buffers and hands them to an event
# writer, and FileWriter (exported publicly for the v1 summary API), which adds a real event writer:
# the thread-based one by default or the session-graph-based one when a session is supplied.
# Also writes plugin asset files next to graphs. Not usable under eager execution.
# Import-time side effects: registers FileWriter in the public API export table.
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
"""Provides an API for generating Event protocol buffers."""

import os.path
import time
import warnings

from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import summary_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import plugin_asset
from tensorflow.python.summary.writer.event_file_writer import EventFileWriter
from tensorflow.python.summary.writer.event_file_writer_v2 import EventFileWriterV2
from tensorflow.python.util.tf_export import tf_export

_PLUGINS_DIR = "plugins"


class SummaryToEventTransformer(object):
  # [CLASS DOC]
  # Base class implementing the summary writer endpoints (add summary, session log, graph, meta
  # graph, run metadata). Each endpoint builds an Event protocol buffer, stamps it with wall clock
  # time and optional step, and forwards it to the contained event writer. Holds the event writer,
  # a record of tags used for run metadata, and the set of summary tags already seen (used to
  # strip repeated metadata). Subclassed by FileWriter.
  """Abstractly implements the SummaryWriter API.

  This API basically implements a number of endpoints (add_summary,
  add_session_log, etc). The endpoints all generate an event protobuf, which is
  passed to the contained event_writer.
  """

  def __init__(self, event_writer, graph=None, graph_def=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: event_writer (object with add event and get log directory methods),
    # graph (optional graph, or possibly a graph definition passed positionally) and graph_def
    # (deprecated optional graph definition). Stores the writer and initializes the tag
    # bookkeeping. If either graph argument is given, adds the graph to the event stream and also
    # exports and adds a meta graph built from it. May raise errors from add graph (value error if
    # both are given, type error for unsupported types). Returns nothing.
    # The seen-tags set is created after the graph is added, which is harmless since add graph
    # does not use it.
    """Creates a `SummaryWriter` and an event file.

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


    Args:
      event_writer: An EventWriter. Implements add_event and get_logdir.
      graph: A `Graph` object, such as `sess.graph`.
      graph_def: DEPRECATED: Use the `graph` argument instead.
    """
    self.event_writer = event_writer
    # For storing used tags for session.run() outputs.
    self._session_run_tags = {}
    if graph is not None or graph_def is not None:
      # Calling it with both graph and graph_def for backward compatibility.
      self.add_graph(graph=graph, graph_def=graph_def)
      # Also export the meta_graph_def in this case.
      # graph may itself be a graph_def due to positional arguments
      maybe_graph_as_def = (graph.as_graph_def(add_shapes=True)
                            if isinstance(graph, ops.Graph) else graph)
      self.add_meta_graph(
          meta_graph.create_meta_graph_def(graph_def=graph_def or
                                           maybe_graph_as_def))

    # This set contains tags of Summary Values that have been encountered
    # already. The motivation here is that the SummaryWriter only keeps the
    # metadata property (which is a SummaryMetadata proto) of the first Summary
    # Value encountered for each tag. The SummaryWriter strips away the
    # SummaryMetadata for all subsequent Summary Values with tags seen
    # previously. This saves space.
    self._seen_summary_tags = set()

  def add_summary(self, summary, global_step=None):
    # [FUNCTION DOC]
    # Writes a summary event. Inputs: summary (Summary proto or its serialized bytes, which are
    # parsed) and global_step (optional number recorded as the event step). To save space, the
    # metadata of any value whose tag was already seen is removed; the first metadata-bearing
    # value for a tag keeps it and the tag is remembered. Note this mutates the passed summary.
    # Wraps the summary in an Event and forwards it. Returns nothing.
    """Adds a `Summary` protocol buffer to the event file.

    This method wraps the provided summary in an `Event` protocol buffer
    and adds it to the event file.

    You can pass the result of evaluating any summary op, using
    `tf.Session.run` or
    `tf.Tensor.eval`, to this
    function. Alternatively, you can pass a `tf.compat.v1.Summary` protocol
    buffer that you populate with your own data. The latter is
    commonly done to report evaluation results in event files.

    Args:
      summary: A `Summary` protocol buffer, optionally serialized as a string.
      global_step: Number. Optional global step value to record with the
        summary.
    """
    if isinstance(summary, bytes):
      summ = summary_pb2.Summary()
      summ.ParseFromString(summary)
      summary = summ

    # We strip metadata from values with tags that we have seen before in order
    # to save space - we just store the metadata on the first value with a
    # specific tag.
    for value in summary.value:
      if not value.metadata:
        continue

      if value.tag in self._seen_summary_tags:
        # This tag has been encountered before. Strip the metadata.
        value.ClearField("metadata")
        continue

      # We encounter a value with a tag we have not encountered previously. And
      # it has metadata. Remember to strip metadata from future values with this
      # tag string.
      self._seen_summary_tags.add(value.tag)

    event = event_pb2.Event(summary=summary)
    self._add_event(event, global_step)

  def add_session_log(self, session_log, global_step=None):
    # [FUNCTION DOC]
    # Wraps a SessionLog proto in an Event and forwards it with the optional global_step. Returns
    # nothing.
    """Adds a `SessionLog` protocol buffer to the event file.

    This method wraps the provided session in an `Event` protocol buffer
    and adds it to the event file.

    Args:
      session_log: A `SessionLog` protocol buffer.
      global_step: Number. Optional global step value to record with the
        summary.
    """
    event = event_pb2.Event(session_log=session_log)
    self._add_event(event, global_step)

  def _add_graph_def(self, graph_def, global_step=None):
    # [FUNCTION DOC]
    # HELPER: Final stage of add graph.
    # Serializes a graph definition proto to bytes, wraps it in an Event and forwards it with the
    # optional global_step. Returns nothing.
    graph_bytes = graph_def.SerializeToString()
    event = event_pb2.Event(graph_def=graph_bytes)
    self._add_event(event, global_step)

  def add_graph(self, graph, global_step=None, graph_def=None):
    # [FUNCTION DOC]
    # Writes a graph event. Inputs: graph (a Graph, or a deprecated graph definition), global_step
    # (optional number) and graph_def (deprecated; may hold a Graph or graph definition).
    # Raises a value error if both graph and graph_def are given, and a type error if neither is a
    # Graph or graph definition. A Graph found in graph_def triggers a warning and is used. For a
    # Graph, serializes it with shapes and also writes plugin asset files into the log directory.
    # A graph definition triggers a deprecation warning and is written as is. Returns nothing.
    """Adds a `Graph` to the event file.

    The graph described by the protocol buffer will be displayed by
    TensorBoard. Most users pass a graph in the constructor instead.

    Args:
      graph: A `Graph` object, such as `sess.graph`.
      global_step: Number. Optional global step counter to record with the
        graph.
      graph_def: DEPRECATED. Use the `graph` parameter instead.

    Raises:
      ValueError: If both graph and graph_def are passed to the method.
    """

    if graph is not None and graph_def is not None:
      raise ValueError("Please pass only graph, or graph_def (deprecated), "
                       "but not both.")

    if isinstance(graph, ops.Graph) or isinstance(graph_def, ops.Graph):
      # The user passed a `Graph`.

      # Check if the user passed it via the graph or the graph_def argument and
      # correct for that.
      if not isinstance(graph, ops.Graph):
        logging.warning("When passing a `Graph` object, please use the `graph`"
                        " named argument instead of `graph_def`.")
        graph = graph_def

      # Serialize the graph with additional info.
      true_graph_def = graph.as_graph_def(add_shapes=True)
      self._write_plugin_assets(graph)
    elif (isinstance(graph, graph_pb2.GraphDef) or
          isinstance(graph_def, graph_pb2.GraphDef)):
      # The user passed a `GraphDef`.
      logging.warning("Passing a `GraphDef` to the SummaryWriter is deprecated."
                      " Pass a `Graph` object instead, such as `sess.graph`.")

      # Check if the user passed it via the graph or the graph_def argument and
      # correct for that.
      if isinstance(graph, graph_pb2.GraphDef):
        true_graph_def = graph
      else:
        true_graph_def = graph_def

    else:
      # The user passed neither `Graph`, nor `GraphDef`.
      raise TypeError("The passed graph must be an instance of `Graph` "
                      "or the deprecated `GraphDef`")
    # Finally, add the graph_def to the summary writer.
    self._add_graph_def(true_graph_def, global_step)

  def _write_plugin_assets(self, graph):
    # [FUNCTION DOC]
    # HELPER: Persists plugin assets for add graph.
    # Input: graph, whose registered plugin asset containers are collected. For each container,
    # creates a plugins/<plugin name> subdirectory under the writer's log directory and writes each
    # named asset's content as a file there. Side effects: directory creation and file writes.
    # Returns nothing.
    plugin_assets = plugin_asset.get_all_plugin_assets(graph)
    logdir = self.event_writer.get_logdir()
    for asset_container in plugin_assets:
      plugin_name = asset_container.plugin_name
      plugin_dir = os.path.join(logdir, _PLUGINS_DIR, plugin_name)
      gfile.MakeDirs(plugin_dir)
      assets = asset_container.assets()
      for (asset_name, content) in assets.items():
        asset_path = os.path.join(plugin_dir, asset_name)
        with gfile.Open(asset_path, "w") as f:
          f.write(content)

  def add_meta_graph(self, meta_graph_def, global_step=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring's error description is garbled ("If both meta_graph_def is not");
    # the code simply raises a type error when the input is not a MetaGraphDef.
    # Serializes a MetaGraphDef proto, wraps it in an Event and forwards it with the optional
    # global_step. Raises a type error for any other input type. Returns nothing.
    """Adds a `MetaGraphDef` to the event file.

    The `MetaGraphDef` allows running the given graph via
    `saver.import_meta_graph()`.

    Args:
      meta_graph_def: A `MetaGraphDef` object, often as returned by
        `saver.export_meta_graph()`.
      global_step: Number. Optional global step counter to record with the
        graph.

    Raises:
      TypeError: If both `meta_graph_def` is not an instance of `MetaGraphDef`.
    """
    if not isinstance(meta_graph_def, meta_graph_pb2.MetaGraphDef):
      raise TypeError("meta_graph_def must be type MetaGraphDef, saw type: %s" %
                      type(meta_graph_def))
    meta_graph_bytes = meta_graph_def.SerializeToString()
    event = event_pb2.Event(meta_graph_def=meta_graph_bytes)
    self._add_event(event, global_step)

  def add_run_metadata(self, run_metadata, tag, global_step=None):
    # [FUNCTION DOC]
    # Writes a tagged run metadata event. Inputs: run_metadata (RunMetadata proto, stored in
    # serialized form), tag (name, must be unused so far on this writer) and global_step
    # (optional). Raises a value error if the tag was used before; otherwise records the tag,
    # builds the event and forwards it. Returns nothing.
    """Adds a metadata information for a single session.run() call.

    Args:
      run_metadata: A `RunMetadata` protobuf object.
      tag: The tag name for this metadata.
      global_step: Number. Optional global step counter to record with the
        StepStats.

    Raises:
      ValueError: If the provided tag was already used for this type of event.
    """
    if tag in self._session_run_tags:
      raise ValueError("The provided tag was already used for this event type")
    self._session_run_tags[tag] = True

    tagged_metadata = event_pb2.TaggedRunMetadata()
    tagged_metadata.tag = tag
    # Store the `RunMetadata` object as bytes in order to have postponed
    # (lazy) deserialization when used later.
    tagged_metadata.run_metadata = run_metadata.SerializeToString()
    event = event_pb2.Event(tagged_run_metadata=tagged_metadata)
    self._add_event(event, global_step)

  def _add_event(self, event, step):
    # [FUNCTION DOC]
    # HELPER: Common sink for all endpoints.
    # Sets the event's wall time to now, sets its step to the integer value of step when step is
    # not null, and passes it to the event writer. Mutates the event. Returns nothing.
    event.wall_time = time.time()
    if step is not None:
      event.step = int(step)
    self.event_writer.add_event(event)


@tf_export(v1=["summary.FileWriter"])
class FileWriter(SummaryToEventTransformer):
  # [CLASS DOC]
  # Public v1 summary writer that writes Summary protos and other events to an events file in a
  # log directory. Chooses the event writer backend at construction (thread-based, or
  # session-graph-based when a session is given), tracks a closed flag, warns when used after
  # close, and supports use as a context manager. Not thread-safe and not usable in eager mode.
  """Writes `Summary` protocol buffers to event files.

  The `FileWriter` class provides a mechanism to create an event file in a
  given directory and add summaries and events to it. The class updates the
  file contents asynchronously. This allows a training program to call methods
  to add data to the file directly from the training loop, without slowing down
  training.

  When constructed with a `tf.compat.v1.Session` parameter, a `FileWriter`
  instead forms a compatibility layer over new graph-based summaries
  to facilitate the use of new summary writing with
  pre-existing code that expects a `FileWriter` instance.

  This class is not thread-safe.

  @compatibility(TF2)
  This API is not compatible with eager execution or `tf.function`. To migrate
  to TF2, please use `tf.summary.create_file_writer` instead for summary
  management. To specify the summary step, you can manage the context with
  `tf.summary.SummaryWriter`, which is returned by
  `tf.summary.create_file_writer()`. Or, you can also use the `step` argument
  of summary functions such as `tf.summary.histogram`.
  See the usage example shown below.

  For a comprehensive `tf.summary` migration guide, please follow
  [Migrating tf.summary usage to
  TF 2.0](https://www.tensorflow.org/tensorboard/migrate#in_tf_1x).

  #### How to Map Arguments

  | TF1 Arg Name        | TF2 Arg Name    | Note                              |
  | :---------------- | :---------------- | :-------------------------------- |
  | `logdir`          | `logdir`          | -                                 |
  | `graph`           | Not supported     | -                                 |
  | `max_queue`       | `max_queue`       | -                                 |
  | `flush_secs`      | `flush_millis`    | The unit of time is changed       |
  :                     :                 : from seconds to milliseconds.     :
  | `graph_def`       | Not supported     | -                                 |
  | `filename_suffix` | `filename_suffix` | -                                 |
  | `name`            | `name`            | -                                 |

  #### TF1 & TF2 Usage Example

  TF1:

  ```python
  dist = tf.compat.v1.placeholder(tf.float32, [100])
  tf.compat.v1.summary.histogram(name="distribution", values=dist)
  writer = tf.compat.v1.summary.FileWriter("/tmp/tf1_summary_example")
  summaries = tf.compat.v1.summary.merge_all()

  sess = tf.compat.v1.Session()
  for step in range(100):
    mean_moving_normal = np.random.normal(loc=step, scale=1, size=[100])
    summ = sess.run(summaries, feed_dict={dist: mean_moving_normal})
    writer.add_summary(summ, global_step=step)
  ```

  TF2:

  ```python
  writer = tf.summary.create_file_writer("/tmp/tf2_summary_example")
  for step in range(100):
    mean_moving_normal = np.random.normal(loc=step, scale=1, size=[100])
    with writer.as_default(step=step):
      tf.summary.histogram(name='distribution', data=mean_moving_normal)
  ```

  @end_compatibility
  """

  def __init__(self,
               logdir,
               graph=None,
               max_queue=10,
               flush_secs=120,
               graph_def=None,
               filename_suffix=None,
               session=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: logdir (directory path), graph (optional graph), max_queue (pending
    # queue size, default 10), flush_secs (seconds between flushes, default 120), graph_def
    # (deprecated), filename_suffix (optional file name suffix) and session (optional v1 session).
    # Raises a runtime error if eager execution is enabled. With a session, builds the
    # session-graph-based writer (adding ops to the session graph and running init); otherwise the
    # thread-based writer, which creates the directory and file and starts a thread. Marks the
    # writer open and initializes the base class, which writes the graph if one was given.
    """Creates a `FileWriter`, optionally shared within the given session.

    Typically, constructing a file writer creates a new event file in `logdir`.
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

    The `session` argument to the constructor makes the returned `FileWriter` a
    compatibility layer over new graph-based summaries (`tf.summary`).
    Crucially, this means the underlying writer resource and events file will
    be shared with any other `FileWriter` using the same `session` and `logdir`.
    In either case, ops will be added to `session.graph` to control the
    underlying file writer resource.

    Args:
      logdir: A string. Directory where event file will be written.
      graph: A `Graph` object, such as `sess.graph`.
      max_queue: Integer. Size of the queue for pending events and summaries.
      flush_secs: Number. How often, in seconds, to flush the
        pending events and summaries to disk.
      graph_def: DEPRECATED: Use the `graph` argument instead.
      filename_suffix: A string. Every event file's name is suffixed with
        `suffix`.
      session: A `tf.compat.v1.Session` object. See details above.

    Raises:
      RuntimeError: If called with eager execution enabled.

    @compatibility(eager)
      `v1.summary.FileWriter` is not compatible with eager execution.
      To write TensorBoard summaries under eager execution,
      use `tf.summary.create_file_writer` or
      a `with v1.Graph().as_default():` context.
    @end_compatibility
    """
    if context.executing_eagerly():
      raise RuntimeError(
          "v1.summary.FileWriter is not compatible with eager execution. "
          "Use `tf.summary.create_file_writer`,"
          "or a `with v1.Graph().as_default():` context")
    if session is not None:
      event_writer = EventFileWriterV2(
          session, logdir, max_queue, flush_secs, filename_suffix)
    else:
      event_writer = EventFileWriter(logdir, max_queue, flush_secs,
                                     filename_suffix)

    self._closed = False
    super(FileWriter, self).__init__(event_writer, graph, graph_def)

  def __enter__(self):
    # [FUNCTION DOC]
    # Context manager entry. Returns the writer itself; no other effects.
    """Make usable with "with" statement."""
    return self

  def __exit__(self, unused_type, unused_value, unused_traceback):
    # [FUNCTION DOC]
    # Context manager exit. Ignores the exception information and closes the writer (flushing
    # pending events). Does not suppress exceptions. Returns nothing.
    """Make usable with "with" statement."""
    self.close()

  def get_logdir(self):
    # [FUNCTION DOC]
    # Returns the log directory reported by the underlying event writer. No side effects.
    """Returns the directory where event file will be written."""
    return self.event_writer.get_logdir()

  def _warn_if_event_writer_is_closed(self):
    # [FUNCTION DOC]
    # HELPER: Guard for event-producing methods.
    # Emits a user warning that a closed writer is being used (operations are no-ops until
    # reopened) when the closed flag is set. Returns nothing.
    if self._closed:
      warnings.warn("Attempting to use a closed FileWriter. "
                    "The operation will be a noop unless the FileWriter "
                    "is explicitly reopened.")

  def _add_event(self, event, step):
    # [FUNCTION DOC]
    # HELPER: Override of the base sink. Warns if the writer is closed, then delegates to the base
    # implementation (stamp time and step, forward to the event writer). Returns nothing.
    self._warn_if_event_writer_is_closed()
    super(FileWriter, self)._add_event(event, step)

  def add_event(self, event):
    # [FUNCTION DOC]
    # Passes a prebuilt Event proto directly to the event writer (without stamping time or step),
    # warning first if the writer is closed. Returns nothing.
    """Adds an event to the event file.

    Args:
      event: An `Event` protocol buffer.
    """
    self._warn_if_event_writer_is_closed()
    self.event_writer.add_event(event)

  def flush(self):
    # [FUNCTION DOC]
    # Warns if the writer is closed, then asks the event writer to flush pending events to disk.
    # With the session-based backend, flushing a closed writer may raise an error from the
    # session; the thread-based backend treats it as a no-op. Returns nothing.
    """Flushes the event file to disk.

    Call this method to make sure that all pending events have been written to
    disk.
    """
    # Flushing a closed EventFileWriterV2 raises an exception. It is,
    # however, a noop for EventFileWriter.
    self._warn_if_event_writer_is_closed()
    self.event_writer.flush()

  def close(self):
    # [FUNCTION DOC]
    # Closes the underlying event writer (which flushes first) and sets the closed flag. Returns
    # nothing.
    """Flushes the event file to disk and close the file.

    Call this method when you do not need the summary writer anymore.
    """
    self.event_writer.close()
    self._closed = True

  def reopen(self):
    # [FUNCTION DOC]
    # Reopens the underlying event writer (new events file in the same directory) and clears the
    # closed flag. Harmless if the writer was not closed. Returns nothing.
    """Reopens the EventFileWriter.

    Can be called after `close()` to add more events in the same directory.
    The events will go into a new events file.

    Does nothing if the EventFileWriter was not closed.
    """
    self.event_writer.reopen()
    self._closed = False
