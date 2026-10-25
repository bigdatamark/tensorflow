# [MODULE DOC]
# HELPER: Test helper module providing a fake summary writer for unit tests.
# Defines FakeSummaryWriter, an in-memory stand-in for the file writer that records summaries,
# graphs, meta graphs, session logs and run metadata instead of writing files, and can swap itself
# in for the real writer class in the writer and writer cache modules. No import-time side effects.
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
"""Fake summary writer for unit tests."""
from tensorflow.core.framework import summary_pb2
from tensorflow.python.framework import test_util
from tensorflow.python.summary.writer import writer
from tensorflow.python.summary.writer import writer_cache


# TODO(ptucker): Replace with mock framework.
class FakeSummaryWriter(object):
  # [CLASS DOC]
  # HELPER: Test double for the summary file writer.
  # Keeps a mapping from step to the list of summaries added, lists of added graphs, meta graphs
  # and session logs, and a mapping from tag to run metadata. Class-level state remembers the real
  # writer class replaced by install so uninstall can restore it. Provides an assertion helper for
  # tests and no-op flush, reopen and close.
  """Fake summary writer."""

  _replaced_summary_writer = None

  @classmethod
  def install(cls):
    # [FUNCTION DOC]
    # Class method. Replaces the real file writer class in both the writer module and the writer
    # cache module with this fake, remembering the original. Raises a value error if already
    # installed. Global side effect on those modules; returns nothing.
    if cls._replaced_summary_writer:
      raise ValueError('FakeSummaryWriter already installed.')
    cls._replaced_summary_writer = writer.FileWriter
    writer.FileWriter = FakeSummaryWriter
    writer_cache.FileWriter = FakeSummaryWriter

  @classmethod
  def uninstall(cls):
    # [FUNCTION DOC]
    # Class method. Restores the original file writer class in the writer and writer cache modules
    # and forgets it. Raises a value error if not installed. Returns nothing.
    if not cls._replaced_summary_writer:
      raise ValueError('FakeSummaryWriter not installed.')
    writer.FileWriter = cls._replaced_summary_writer
    writer_cache.FileWriter = cls._replaced_summary_writer
    cls._replaced_summary_writer = None

  def __init__(self, logdir, graph=None):
    # [FUNCTION DOC]
    # Constructor, signature compatible with the real writer's basic arguments. Inputs: logdir
    # (stored for later assertions) and graph (optional, stored). Initializes empty storage for
    # summaries, graphs, meta graphs, session logs and run metadata. No files are created.
    self._logdir = logdir
    self._graph = graph
    self._summaries = {}
    self._added_graphs = []
    self._added_meta_graphs = []
    self._added_session_logs = []
    self._added_run_metadata = {}

  @property
  def summaries(self):
    # [FUNCTION DOC]
    # Property returning the mapping from global step to the list of summary protos added at that
    # step. Returns the live internal mapping, not a copy.
    return self._summaries

  def assert_summaries(self,
                       test_case,
                       expected_logdir=None,
                       expected_graph=None,
                       expected_summaries=None,
                       expected_added_graphs=None,
                       expected_added_meta_graphs=None,
                       expected_session_logs=None):
    # [FUNCTION DOC]
    # Test assertion helper. Inputs: test_case (a test case providing assertion methods) and
    # optional expectations, each skipped when null: expected_logdir (must equal the stored log
    # directory), expected_graph (must be the identical graph object), expected_summaries (mapping
    # from step to mapping of tag to simple value; each step must be present, and the recorded
    # simple values for that step, ignoring the global steps per second tag, must equal it),
    # expected_added_graphs (must equal the list of added graphs), expected_added_meta_graphs (same
    # length, and each pair compared with the meta graph comparison utility) and
    # expected_session_logs (must equal the list of session logs). Failures surface as assertion
    # failures in the test case. Returns nothing.
    """Assert expected items have been added to summary writer."""
    if expected_logdir is not None:
      test_case.assertEqual(expected_logdir, self._logdir)
    if expected_graph is not None:
      test_case.assertTrue(expected_graph is self._graph)
    expected_summaries = expected_summaries or {}
    for step in expected_summaries:
      test_case.assertTrue(
          step in self._summaries,
          msg='Missing step %s from %s.' % (step, self._summaries.keys()))
      actual_simple_values = {}
      for step_summary in self._summaries[step]:
        for v in step_summary.value:
          # Ignore global_step/sec since it's written by Supervisor in a
          # separate thread, so it's non-deterministic how many get written.
          if 'global_step/sec' != v.tag:
            actual_simple_values[v.tag] = v.simple_value
      test_case.assertEqual(expected_summaries[step], actual_simple_values)
    if expected_added_graphs is not None:
      test_case.assertEqual(expected_added_graphs, self._added_graphs)
    if expected_added_meta_graphs is not None:
      test_case.assertEqual(len(expected_added_meta_graphs),
                            len(self._added_meta_graphs))
      for expected, actual in zip(expected_added_meta_graphs,
                                  self._added_meta_graphs):
        test_util.assert_meta_graph_protos_equal(test_case, expected, actual)
    if expected_session_logs is not None:
      test_case.assertEqual(expected_session_logs, self._added_session_logs)

  def add_summary(self, summ, current_global_step):
    # [FUNCTION DOC]
    # Records a summary under a step. Inputs: summ (a Summary proto or its serialized bytes, which
    # are parsed first) and current_global_step (key for grouping). Appends to the list for that
    # step, creating it if needed. Returns nothing.
    """Add summary."""
    if isinstance(summ, bytes):
      summary_proto = summary_pb2.Summary()
      summary_proto.ParseFromString(summ)
      summ = summary_proto
    if current_global_step in self._summaries:
      step_summaries = self._summaries[current_global_step]
    else:
      step_summaries = []
      self._summaries[current_global_step] = step_summaries
    step_summaries.append(summ)

  # NOTE: Ignore global_step since its value is non-deterministic.
  def add_graph(self, graph, global_step=None, graph_def=None):
    # [FUNCTION DOC]
    # Records a graph. Inputs: graph (stored), global_step (optional; only validated, not stored),
    # graph_def (must be null). Raises a value error for a negative global step or when graph_def
    # is given. Returns nothing.
    """Add graph."""
    if (global_step is not None) and (global_step < 0):
      raise ValueError('Invalid global_step %s.' % global_step)
    if graph_def is not None:
      raise ValueError('Unexpected graph_def %s.' % graph_def)
    self._added_graphs.append(graph)

  def add_meta_graph(self, meta_graph_def, global_step=None):
    # [FUNCTION DOC]
    # Records a meta graph definition in the list of added meta graphs. global_step is optional and
    # only validated: a negative value raises a value error. Returns nothing.
    """Add metagraph."""
    if (global_step is not None) and (global_step < 0):
      raise ValueError('Invalid global_step %s.' % global_step)
    self._added_meta_graphs.append(meta_graph_def)

  # NOTE: Ignore global_step since its value is non-deterministic.
  def add_session_log(self, session_log, global_step=None):
    # [FUNCTION DOC]
    # Records a session log in the list of session logs. global_step is accepted but ignored
    # (not validated or stored) because its value is non-deterministic. Returns nothing.
    # pylint: disable=unused-argument
    self._added_session_logs.append(session_log)

  def add_run_metadata(self, run_metadata, tag, global_step=None):
    # [FUNCTION DOC]
    # Stores run metadata under a tag, replacing any earlier value for the same tag. global_step is
    # optional and only validated: a negative value raises a value error. Returns nothing.
    if (global_step is not None) and (global_step < 0):
      raise ValueError('Invalid global_step %s.' % global_step)
    self._added_run_metadata[tag] = run_metadata

  def flush(self):
    # [FUNCTION DOC]
    # No-op; present for interface compatibility with the real writer.
    pass

  def reopen(self):
    # [FUNCTION DOC]
    # No-op; present for interface compatibility with the real writer.
    pass

  def close(self):
    # [FUNCTION DOC]
    # No-op; present for interface compatibility with the real writer.
    pass
