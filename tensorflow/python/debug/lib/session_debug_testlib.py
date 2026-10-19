# [MODULE DOC]
# Shared test library for the tfdbg debugger with graph-mode sessions. Provides a base test class
# (SessionDebugTestBase) whose concrete subclasses supply the debug URLs and dump directory
# (file system or gRPC streaming), a small test RNN cell, a session-config helper that disables
# graph rewrites, and a test class for concurrent debugged run calls. Tests cover dumping of
# tensors, while-loops, cond, RNN training, bad-value search, graph structure lookup, causality
# checks, numeric summaries, error handling and Python traceback lookup. Helper/test-library
# module: it defines no tests that run by itself (the base class is meant to be subclassed).
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
"""Tests for debugger functionalities in tf.Session."""
import collections
import functools
import glob
import os
import tempfile
import threading

import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.debug.lib import debug_utils
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
import tensorflow.python.ops.tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent


def no_rewrite_session_config():
  # [FUNCTION DOC]
  # HELPER: small shared test utility that builds session configurations for the tests below.
  # Takes no arguments. Returns a session configuration whose graph optimizer is set to disable
  # model pruning, arithmetic optimization and dependency optimization, so that the graph
  # executed is the one the test built (nodes are not removed or rewritten).
  rewriter_config = rewriter_config_pb2.RewriterConfig(
      disable_model_pruning=True,
      arithmetic_optimization=rewriter_config_pb2.RewriterConfig.OFF,
      dependency_optimization=rewriter_config_pb2.RewriterConfig.OFF)
  graph_options = config_pb2.GraphOptions(rewrite_options=rewriter_config)
  return config_pb2.ConfigProto(graph_options=graph_options)


class _RNNCellForTest(rnn_cell_impl.RNNCell):
  # [CLASS DOC]
  # HELPER: minimal recurrent cell used by the dynamic RNN training test.
  # Holds a single scalar trainable weight (initial value one, named "w"); the output is the
  # input multiplied by that weight and the state is passed through unchanged. The input/output
  # size and state size are fixed at construction.
  """RNN cell for testing."""

  def __init__(self, input_output_size, state_size):
    # [FUNCTION DOC]
    # Constructor. Stores the input/output size and state size (integers) and creates the scalar
    # float variable "w" initialized to one. Creates a variable in the default graph.
    self._input_output_size = input_output_size
    self._state_size = state_size
    self._w = variable_v1.VariableV1(1.0, dtype=dtypes.float32, name="w")

  @property
  def output_size(self):
    # [FUNCTION DOC]
    # Property: returns the output size given at construction.
    return self._input_output_size

  @property
  def state_size(self):
    # [FUNCTION DOC]
    # Property: returns the state size given at construction.
    return self._state_size

  def __call__(self, input_, state, scope=None):
    # [FUNCTION DOC]
    # Runs one time step. Parameters: input_ (tensor), state (tensor), scope (optional, unused).
    # Returns a pair: the input multiplied element-wise by the weight variable, and the unchanged
    # state.
    return (math_ops.multiply(self._w, input_), state)


@test_util.run_v1_only("b/120545219")
class SessionDebugTestBase(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Abstract base class for tfdbg tests that run with graph-mode sessions (version one only).
  # Subclasses must implement the debug URL and dump directory providers; the tests then
  # exercise tensor dumping and the resulting debug dump directory object. Class state holds the
  # expected partition graph count, expected device count and main device name (depending on
  # whether a GPU is available); instance state holds a temporary dump root directory created
  # per test and removed afterwards.
  """Base class for unit tests of tfdbg running with tf.Session."""

  @classmethod
  def setUpClass(cls):
    # [FUNCTION DOC]
    # Class-level setup. Detects whether a GPU is available and records on the class the
    # expected number of partition graphs and devices (two each with a GPU, one each without)
    # and the main device name (the GPU device or the local CPU device).
    if test.is_gpu_available():
      cls._expected_partition_graph_count = 2
      cls._expected_num_devices = 2
      gpu_name = test_util.gpu_device_name()
      cls._main_device = "/job:localhost/replica:0/task:0" + gpu_name
    else:
      cls._expected_partition_graph_count = 1
      cls._expected_num_devices = 1
      cls._main_device = "/job:localhost/replica:0/task:0/device:CPU:0"

  @classmethod
  def tearDownClass(cls):
    # [FUNCTION DOC]
    # Class-level cleanup; intentionally does nothing.
    pass

  def setUp(self):
    # [FUNCTION DOC]
    # Per-test setup: creates a fresh temporary directory and stores its path as the dump root.
    self._dump_root = tempfile.mkdtemp()

  def tearDown(self):
    # [FUNCTION DOC]
    # Per-test cleanup: resets the default graph and recursively deletes the dump root directory
    # if it still exists.
    ops.reset_default_graph()

    # Tear down temporary dump directory.
    if os.path.isdir(self._dump_root):
      file_io.delete_recursively(self._dump_root)

  def _debug_urls(self, run_number=None):
    # [FUNCTION DOC]
    # HELPER: abstract hook for subclasses. Should return the list of debug URLs to watch with;
    # run_number is an optional run index. The base implementation always raises a
    # not-implemented error.
    raise NotImplementedError(
        "_debug_urls() method is not implemented in the base test class.")

  def _debug_dump_dir(self, run_number=None):
    # [FUNCTION DOC]
    # HELPER: abstract hook for subclasses. Should return the directory where dumps of the given
    # (optional) run number are found. The base implementation always raises a not-implemented
    # error.
    raise NotImplementedError(
        "_debug_dump_dir() method is not implemented in the base test class.")

  def _debug_run_and_get_dump(self,
                              sess,
                              fetches,
                              feed_dict=None,
                              debug_ops="DebugIdentity",
                              tolerate_debug_op_creation_failures=False,
                              global_step=-1,
                              validate=True,
                              expected_partition_graph_count=None):
    # [FUNCTION DOC]
    # HELPER: shared by many tests to perform one debugged session run.
    # Builds run options that request partition graphs and watches every node of the session's
    # graph with the given debug ops and the subclass's debug URLs (optionally tolerating debug
    # op creation failures, and tagging the global step). Runs the fetches with the optional feed
    # mapping, optionally asserts the number of returned partition graphs, and loads a debug dump
    # object from the dump root with those partition graphs (validating against the graph unless
    # validate is false). Returns a pair: the run output and the dump object.
    """Run fetches with debugging and obtain DebugDumpDir.

    Args:
      sess: the tf.compat.v1.Session to be used.
      fetches: fetches of the Session.run().
      feed_dict: feed dict for the Session.run().
      debug_ops: name(s) of the debug ops to be used.
      tolerate_debug_op_creation_failures: whether to tolerate debug op
        creation failures.
      global_step: Optional global step.
      validate: whether to validate dumped tensors against graph.
      expected_partition_graph_count: optional count of partition graphs to
        assert on.

    Returns:
      1. Return values of the Session.run().
      2. The DebugDumpDir object from the debugged run().
    """

    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    debug_utils.watch_graph(
        run_options,
        sess.graph,
        debug_ops=debug_ops,
        debug_urls=self._debug_urls(),
        tolerate_debug_op_creation_failures=tolerate_debug_op_creation_failures,
        global_step=global_step)
    run_metadata = config_pb2.RunMetadata()
    run_output = sess.run(fetches,
                          feed_dict=feed_dict,
                          options=run_options,
                          run_metadata=run_metadata)

    if expected_partition_graph_count is not None:
      self.assertEqual(expected_partition_graph_count,
                       len(run_metadata.partition_graphs))
    return run_output, debug_data.DebugDumpDir(
        self._dump_root, partition_graphs=run_metadata.partition_graphs,
        validate=validate)

  def _generate_dump_from_simple_addition_graph(self):
    # [FUNCTION DOC]
    # HELPER: fixture generator shared by several tests.
    # In a session with graph rewrites disabled, builds two variables u (2x2) and v (2x1) and
    # their matrix product w, initializes the variables, watches the read ops of u and v with
    # file dump URLs under the dump root, runs w, asserts the expected partition graph count, and
    # loads the debug dump. Returns a named tuple with the initial values of u and v, the
    # variables and product tensor, their names, and the dump object.
    with session.Session(config=no_rewrite_session_config()) as sess:
      u_init_val = np.array([[5.0, 3.0], [-1.0, 0.0]])
      v_init_val = np.array([[2.0], [-1.0]])

      # Use node names with overlapping namespace (i.e., parent directory) to
      # test concurrent, non-racing directory creation.
      u_name = "u"
      v_name = "v"
      w_name = "w"

      u_init = constant_op.constant(u_init_val, shape=[2, 2])
      u = variable_v1.VariableV1(u_init, name=u_name)
      v_init = constant_op.constant(v_init_val, shape=[2, 1])
      v = variable_v1.VariableV1(v_init, name=v_name)

      w = math_ops.matmul(u, v, name=w_name)

      u.initializer.run()
      v.initializer.run()

      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      debug_urls = "file://%s" % self._dump_root

      # Add debug tensor watch for u.
      debug_utils.add_debug_tensor_watch(
          run_options, "%s/read" % u_name, 0, debug_urls=debug_urls)
      # Add debug tensor watch for v.
      debug_utils.add_debug_tensor_watch(
          run_options, "%s/read" % v_name, 0, debug_urls=debug_urls)

      run_metadata = config_pb2.RunMetadata()

      # Invoke Session.run().
      sess.run(w, options=run_options, run_metadata=run_metadata)

      self.assertEqual(self._expected_partition_graph_count,
                       len(run_metadata.partition_graphs))

      dump = debug_data.DebugDumpDir(
          self._dump_root, partition_graphs=run_metadata.partition_graphs)

    simple_add_results = collections.namedtuple("SimpleAddResults", [
        "u_init_val", "v_init_val", "u", "v", "w", "u_name", "v_name", "w_name",
        "dump"
    ])
    return simple_add_results(u_init_val, v_init_val, u, v, w, u_name, v_name,
                              w_name, dump)

  def testCopyNodesHaveCorrectDebugOpsAndURLsAttributeValues(self):
    # [FUNCTION DOC]
    # Test: copy nodes inserted by the debugger carry correct debug-ops specifications.
    # Setup: two variables multiplied; watches the first with a gated numeric summary and an
    # identity op, and the second with a numeric summary, using the subclass's debug URLs. Runs
    # and checks the product is 42. Then finds the copy nodes in the partition graphs and
    # asserts the debug-ops-spec attribute of each lists the expected entries: debug op name,
    # URL, and the gated flag (one for the gated op, zero otherwise).
    with session.Session() as sess:
      u = variable_v1.VariableV1(2.1, name="u")
      v = variable_v1.VariableV1(20.0, name="v")
      w = math_ops.multiply(u, v, name="w")

      sess.run(variables.global_variables_initializer())

      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      debug_urls = self._debug_urls()
      debug_utils.add_debug_tensor_watch(
          run_options,
          "u",
          0, ["DebugNumericSummary(gated_grpc=True)", "DebugIdentity"],
          debug_urls=debug_urls)
      debug_utils.add_debug_tensor_watch(
          run_options, "v", 0, ["DebugNumericSummary"], debug_urls=debug_urls)

      run_metadata = config_pb2.RunMetadata()
      r = sess.run(w, options=run_options, run_metadata=run_metadata)
      self.assertAllClose(42.0, r)

      u_copy_node_def = None
      v_copy_node_def = None
      for partition_graph in run_metadata.partition_graphs:
        for node_def in partition_graph.node:
          if debug_graphs.is_copy_node(node_def.name):
            if node_def.name == "__copy_u_0":
              u_copy_node_def = node_def
            elif node_def.name == "__copy_v_0":
              v_copy_node_def = node_def

      self.assertIsNotNone(u_copy_node_def)
      debug_ops_spec = u_copy_node_def.attr["debug_ops_spec"].list.s
      self.assertEqual(2, len(debug_ops_spec))
      self.assertEqual("DebugNumericSummary;%s;1" % debug_urls[0],
                       debug_ops_spec[0].decode("utf-8"))
      self.assertEqual("DebugIdentity;%s;0" % debug_urls[0],
                       debug_ops_spec[1].decode("utf-8"))

      self.assertIsNotNone(v_copy_node_def)
      debug_ops_spec = v_copy_node_def.attr["debug_ops_spec"].list.s
      self.assertEqual(1, len(debug_ops_spec))
      self.assertEqual("DebugNumericSummary;%s;0" % debug_urls[0],
                       debug_ops_spec[0].decode("utf-8"))

  def testConcurrentDumpingToPathsWithOverlappingParentDirsWorks(self):
    # [FUNCTION DOC]
    # Test: dumping tensors whose node names share parent directories works. Uses the simple
    # addition fixture. Asserts the partition graphs are loaded, the core metadata has the
    # default global step of minus one, non-negative run and executor indices, no inputs, the
    # product as the sole output and no targets; the dump holds two tensors whose values equal
    # the initial values, with non-negative relative timestamps and positive dump sizes.
    results = self._generate_dump_from_simple_addition_graph()
    self.assertTrue(results.dump.loaded_partition_graphs())

    # Since global_step is not explicitly specified, it should take its default
    # value: -1.
    self.assertEqual(-1, results.dump.core_metadata.global_step)
    self.assertGreaterEqual(results.dump.core_metadata.session_run_index, 0)
    self.assertGreaterEqual(results.dump.core_metadata.executor_step_index, 0)
    self.assertEqual([], results.dump.core_metadata.input_names)
    self.assertEqual([results.w.name], results.dump.core_metadata.output_names)
    self.assertEqual([], results.dump.core_metadata.target_nodes)

    # Verify the dumped tensor values for u and v.
    self.assertEqual(2, results.dump.size)

    self.assertAllClose([results.u_init_val],
                        results.dump.get_tensors("%s/read" % results.u_name, 0,
                                                 "DebugIdentity"))
    self.assertAllClose([results.v_init_val],
                        results.dump.get_tensors("%s/read" % results.v_name, 0,
                                                 "DebugIdentity"))

    self.assertGreaterEqual(
        results.dump.get_rel_timestamps("%s/read" % results.u_name, 0,
                                        "DebugIdentity")[0], 0)
    self.assertGreaterEqual(
        results.dump.get_rel_timestamps("%s/read" % results.v_name, 0,
                                        "DebugIdentity")[0], 0)

    self.assertGreater(
        results.dump.get_dump_sizes_bytes("%s/read" % results.u_name, 0,
                                          "DebugIdentity")[0], 0)
    self.assertGreater(
        results.dump.get_dump_sizes_bytes("%s/read" % results.v_name, 0,
                                          "DebugIdentity")[0], 0)

  def testGetOpTypeWorks(self):
    # [FUNCTION DOC]
    # Test: looking up node op types from a dump. Uses the simple addition fixture and asserts
    # the dump reports the variable's op type for u, and op types containing the expected type
    # for v and w; an unknown node name raises a value error saying no device has such a node.
    results = self._generate_dump_from_simple_addition_graph()

    self.assertEqual(results.u.op.type,
                     results.dump.node_op_type(results.u_name))
    self.assertIn(results.v.op.type, results.dump.node_op_type(results.v_name))
    self.assertIn(results.w.op.type, results.dump.node_op_type(results.w_name))

    with self.assertRaisesRegex(
        ValueError, r"None of the .* device\(s\) has a node named "
    ):
      results.dump.node_op_type("foo_bar")

  def testDumpStringTensorsWorks(self):
    # [FUNCTION DOC]
    # Test: string tensors can be dumped. Builds two string variables and an add (string
    # concatenation) op, watches the variables' reads, runs, and asserts a single partition
    # graph (string ops live on CPU). In the dump asserts both nodes are present, two tensors
    # were dumped with the expected string values, timestamps are non-negative and dump sizes
    # are positive.
    with session.Session(config=no_rewrite_session_config()) as sess:
      str1_init_val = np.array(b"abc")
      str2_init_val = np.array(b"def")

      str1_init = constant_op.constant(str1_init_val)
      str2_init = constant_op.constant(str2_init_val)

      str1_name = "str1"
      str2_name = "str2"
      str1 = variable_v1.VariableV1(str1_init, name=str1_name)
      str2 = variable_v1.VariableV1(str2_init, name=str2_name)
      # Concatenate str1 and str2
      str_concat = math_ops.add(str1, str2, name="str_concat")

      str1.initializer.run()
      str2.initializer.run()

      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      debug_urls = self._debug_urls()

      # Add debug tensor watch for u.
      debug_utils.add_debug_tensor_watch(
          run_options, "%s/read" % str1_name, 0, debug_urls=debug_urls)
      # Add debug tensor watch for v.
      debug_utils.add_debug_tensor_watch(
          run_options, "%s/read" % str2_name, 0, debug_urls=debug_urls)

      run_metadata = config_pb2.RunMetadata()
      sess.run(str_concat, options=run_options, run_metadata=run_metadata)

      # String ops are located on CPU.
      self.assertEqual(1, len(run_metadata.partition_graphs))

      dump = debug_data.DebugDumpDir(
          self._dump_root, partition_graphs=run_metadata.partition_graphs)

      self.assertIn(str1_name, dump.nodes())
      self.assertIn(str2_name, dump.nodes())

      self.assertEqual(2, dump.size)

      self.assertEqual([str1_init_val],
                       dump.get_tensors("%s/read" % str1_name, 0,
                                        "DebugIdentity"))
      self.assertEqual([str2_init_val],
                       dump.get_tensors("%s/read" % str2_name, 0,
                                        "DebugIdentity"))

      self.assertGreaterEqual(
          dump.get_rel_timestamps("%s/read" % str1_name, 0, "DebugIdentity")[0],
          0)
      self.assertGreaterEqual(
          dump.get_rel_timestamps("%s/read" % str2_name, 0, "DebugIdentity")[0],
          0)

      self.assertGreater(
          dump.get_dump_sizes_bytes("%s/read" % str1_name, 0,
                                    "DebugIdentity")[0], 0)
      self.assertGreater(
          dump.get_dump_sizes_bytes("%s/read" % str2_name, 0,
                                    "DebugIdentity")[0], 0)

  def testDumpUninitializedVariable(self):
    # [FUNCTION DOC]
    # Test: dumping variables during their initialization. Creates a numeric and a string
    # variable under a namespace, watches both, and runs the global initializer with the debug
    # options. Loads the dump and asserts two tensors were dumped, the expected partition graph
    # count, and that the dumped values reflect the uninitialized state versus initial values as
    # asserted in the rest of the test.
    op_namespace = "testDumpUninitializedVariable"
    with session.Session() as sess:
      u_init_val = np.array([[5.0, 3.0], [-1.0, 0.0]])
      s_init_val = b"str1"

      u_name = "%s/u" % op_namespace
      s_name = "%s/s" % op_namespace

      u_init = constant_op.constant(u_init_val, shape=[2, 2])
      u = variable_v1.VariableV1(u_init, name=u_name)
      s_init = constant_op.constant(s_init_val)
      s = variable_v1.VariableV1(s_init, name=s_name)

      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      debug_urls = self._debug_urls()

      # Add debug tensor watch for u.
      debug_utils.add_debug_tensor_watch(
          run_options, u_name, 0, debug_urls=debug_urls)
      debug_utils.add_debug_tensor_watch(
          run_options, s_name, 0, debug_urls=debug_urls)

      run_metadata = config_pb2.RunMetadata()

      # Initialize u and s.
      sess.run(variables.global_variables_initializer(),
               options=run_options,
               run_metadata=run_metadata)

      # Verify the dump file for the uninitialized value of u.
      dump = debug_data.DebugDumpDir(
          self._dump_root, partition_graphs=run_metadata.partition_graphs)

      self.assertEqual(2, dump.size)
      self.assertEqual(self._expected_partition_graph_count,
                       len(run_metadata.partition_graphs))

      # Verify that the variable is properly initialized by the run() call.
      u_vals = dump.get_tensors(u_name, 0, "DebugIdentity")
      s_vals = dump.get_tensors(s_name, 0, "DebugIdentity")
      self.assertEqual(1, len(u_vals))
      self.assertIsInstance(u_vals[0], debug_data.InconvertibleTensorProto)
      self.assertFalse(u_vals[0].initialized)
      self.assertEqual(1, len(s_vals))
      self.assertIsInstance(s_vals[0], debug_data.InconvertibleTensorProto)
      self.assertFalse(s_vals[0].initialized)

      # Call run() again, to check that u is initialized properly.
      self.assertAllClose(u_init_val, sess.run(u))
      self.assertEqual(s_init_val, sess.run(s))

  def testDebugWhileLoopGeneratesMultipleDumps(self):
    # [FUNCTION DOC]
    # Test: a while loop with watched nodes yields one dump per iteration. Setup: a variable u
    # (11.0) incremented each iteration by a variable v (2.0) inside a ten-iteration loop, graph
    # rewrites disabled; watches u, v's read, and the loop's identity and constant-one nodes.
    # Asserts: expected partition graph count; loop result of ten; final u equals the initial
    # value plus ten increments; dump directories for u and v exist; the dump holds
    # 1 + 1 + 10 + 10 tensors; values of u and v are the initial ones; the loop identity node has
    # ten dumps with values 0..9, non-decreasing relative timestamps and equal dump sizes; the
    # debug watch keys of the identity node are as expected, ten data records match its watch
    # key, and an unknown key gives an empty list.
    with session.Session(config=no_rewrite_session_config()) as sess:
      num_iter = 10

      # "u" is the Variable being updated in the loop.
      u_name = "testDumpToFileWhileLoop/u"
      u_namespace = u_name.split("/")[0]

      u_init_val = np.array(11.0)
      u_init = constant_op.constant(u_init_val)
      u = variable_v1.VariableV1(u_init, name=u_name)

      # "v" is the increment.
      v_name = "testDumpToFileWhileLoop/v"
      v_namespace = v_name.split("/")[0]

      v_init_val = np.array(2.0)
      v_init = constant_op.constant(v_init_val)
      v = variable_v1.VariableV1(v_init, name=v_name)

      u.initializer.run()
      v.initializer.run()

      i = constant_op.constant(0, name="testDumpToFileWhileLoop/i")

      def cond(i):
        # [FUNCTION DOC]
        # Loop condition: returns whether the counter i is less than the iteration count.
        return math_ops.less(i, num_iter)

      def body(i):
        # [FUNCTION DOC]
        # Loop body: adds the increment variable to u, increments the counter, and makes the
        # new counter depend on the update op so that u is updated every iteration. Returns a
        # list holding the new counter.
        new_u = state_ops.assign_add(u, v)
        new_i = math_ops.add(i, 1)
        op = control_flow_ops.group(new_u)
        new_i = control_flow_ops.with_dependencies([op], new_i)
        return [new_i]

      loop = while_loop.while_loop(cond, body, [i], parallel_iterations=10)

      # Create RunOptions for debug-watching tensors
      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      debug_urls = self._debug_urls()

      # Add debug tensor watch for u.
      debug_utils.add_debug_tensor_watch(
          run_options, u_name, 0, debug_urls=debug_urls)
      # Add debug tensor watch for v.
      debug_utils.add_debug_tensor_watch(
          run_options, "%s/read" % v_name, 0, debug_urls=debug_urls)
      # Add debug tensor watch for while/Identity.
      debug_utils.add_debug_tensor_watch(
          run_options, "while/Identity", 0, debug_urls=debug_urls)
      # Add debug tensor watch for while/Add/y.
      debug_utils.add_debug_tensor_watch(
          run_options, "while/Add/y", 0, debug_urls=debug_urls)

      run_metadata = config_pb2.RunMetadata()
      r = sess.run(loop, options=run_options, run_metadata=run_metadata)

      self.assertEqual(self._expected_partition_graph_count,
                       len(run_metadata.partition_graphs))

      self.assertEqual(num_iter, r)
      u_val_final = sess.run(u)
      self.assertAllClose(u_init_val + num_iter * v_init_val, u_val_final)

      # Verify dump files
      self.assertTrue(os.path.isdir(self._dump_root))

      u_glob_out = glob.glob(os.path.join(self._dump_root, "*", u_namespace))
      v_glob_out = glob.glob(os.path.join(
          self._dump_root, "*", v_namespace, "v"))
      self.assertTrue(os.path.isdir(u_glob_out[0]))
      self.assertTrue(os.path.isdir(v_glob_out[0]))

      dump = debug_data.DebugDumpDir(
          self._dump_root, partition_graphs=run_metadata.partition_graphs)

      # Expected dumped tensors: u, v/read, 10 iterations of while/Identity,
      # and 10 iterations of while/Add/y.
      self.assertEqual(1 + 1 + num_iter + num_iter, dump.size)

      # Verify tensor values.
      self.assertAllClose([u_init_val],
                          dump.get_tensors(u_name, 0, "DebugIdentity"))
      self.assertAllClose([v_init_val],
                          dump.get_tensors("%s/read" % v_name, 0,
                                           "DebugIdentity"))

      while_id_tensors = dump.get_tensors("while/Identity", 0, "DebugIdentity")
      self.assertEqual(10, len(while_id_tensors))
      for k in range(len(while_id_tensors)):
        self.assertAllClose(np.array(k), while_id_tensors[k])

      # Verify ascending timestamps from the while loops.
      while_id_rel_timestamps = dump.get_rel_timestamps("while/Identity", 0,
                                                        "DebugIdentity")
      while_id_dump_sizes_bytes = dump.get_dump_sizes_bytes("while/Identity", 0,
                                                            "DebugIdentity")
      self.assertEqual(10, len(while_id_rel_timestamps))
      prev_rel_time = 0
      prev_dump_size_bytes = while_id_dump_sizes_bytes[0]
      for rel_time, dump_size_bytes in zip(while_id_rel_timestamps,
                                           while_id_dump_sizes_bytes):
        self.assertGreaterEqual(rel_time, prev_rel_time)
        self.assertEqual(dump_size_bytes, prev_dump_size_bytes)
        prev_rel_time = rel_time
        prev_dump_size_bytes = dump_size_bytes

      # Test querying debug watch keys from node name.
      watch_keys = dump.debug_watch_keys("while/Identity")
      self.assertEqual(["while/Identity:0:DebugIdentity"], watch_keys)

      # Test querying debug datum instances from debug watch key.
      self.assertEqual(10, len(dump.watch_key_to_data(watch_keys[0])))
      self.assertEqual([], dump.watch_key_to_data("foo"))

  def testDebugWhileLoopWatchingWholeGraphWorks(self):
    # [FUNCTION DOC]
    # Test: watching an entire graph containing a while loop. Loop adds two to a counter that
    # starts at ten until it reaches at least sixteen; runs through the shared debug-run helper.
    # Asserts the result is sixteen, the loop-entry node dumped [10], and the next-iteration node
    # dumped 12, 14 and 16 in order.
    with session.Session() as sess:
      loop_body = lambda i: math_ops.add(i, 2)
      loop_cond = lambda i: math_ops.less(i, 16)

      i = constant_op.constant(10, name="i")
      loop = while_loop.while_loop(loop_cond, loop_body, [i])

      loop_result, dump = self._debug_run_and_get_dump(sess, loop)
      self.assertEqual(16, loop_result)

      self.assertEqual(
          [[10]], dump.get_tensors("while/Enter", 0, "DebugIdentity"))
      self.assertEqual(
          [[12], [14], [16]],
          dump.get_tensors("while/NextIteration", 0, "DebugIdentity"))

  def testDebugTrainingDynamicRNNWorks(self):
    # [FUNCTION DOC]
    # Test: debugging a training step of a dynamic RNN does not fail. Builds a dynamic RNN with
    # the test cell over random time-major inputs with random sequence lengths, a toy sum-of-
    # squares loss and a gradient descent training op. Watches the whole graph except nodes
    # matching the RNN while-loop and tensor-array name patterns (excluded to avoid
    # uninitialized-tensor memory sanitizer warnings with both file and gRPC URLs), runs the
    # training op with random inputs fed, and only asserts that the dump directory loads
    # without error.
    with session.Session() as sess:
      input_size = 3
      state_size = 2
      time_steps = 4
      batch_size = 2

      input_values = np.random.randn(time_steps, batch_size, input_size)
      sequence_length = np.random.randint(0, time_steps, size=batch_size)
      concat_inputs = array_ops.placeholder(
          dtypes.float32, shape=(time_steps, batch_size, input_size))

      outputs_dynamic, _ = rnn.dynamic_rnn(
          _RNNCellForTest(input_size, state_size),
          inputs=concat_inputs,
          sequence_length=sequence_length,
          time_major=True,
          dtype=dtypes.float32)
      toy_loss = math_ops.reduce_sum(outputs_dynamic * outputs_dynamic)
      train_op = gradient_descent.GradientDescentOptimizer(
          learning_rate=0.1).minimize(toy_loss, name="train_op")

      sess.run(variables.global_variables_initializer())

      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      debug_utils.watch_graph_with_denylists(
          run_options,
          sess.graph,
          node_name_regex_denylist="(.*rnn/while/.*|.*TensorArray.*)",
          debug_urls=self._debug_urls())
      # b/36870549: Nodes with these name patterns need to be excluded from
      # tfdbg in order to prevent MSAN warnings of uninitialized Tensors
      # under both file:// and grpc:// debug URL schemes.

      run_metadata = config_pb2.RunMetadata()
      sess.run(train_op, feed_dict={concat_inputs: input_values},
               options=run_options, run_metadata=run_metadata)

      debug_data.DebugDumpDir(
          self._dump_root, partition_graphs=run_metadata.partition_graphs)

  def testDebugCondWatchingWholeGraphWorks(self):
    # [FUNCTION DOC]
    # Test: watching an entire graph containing a conditional. Two variables (10 and 20), a
    # conditional choosing the first plus one if the first is greater, else the second plus one.
    # Asserts the result is 21 and the merge node of the conditional dumped [21.0].
    with session.Session() as sess:
      x = variable_v1.VariableV1(10.0, name="x")
      y = variable_v1.VariableV1(20.0, name="y")
      cond = tf_cond.cond(
          x > y, lambda: math_ops.add(x, 1), lambda: math_ops.add(y, 1))

      sess.run(variables.global_variables_initializer())

      cond_result, dump = self._debug_run_and_get_dump(sess, cond)
      self.assertEqual(21, cond_result)

      self.assertAllClose(
          [21.0], dump.get_tensors("cond/Merge", 0, "DebugIdentity"))

  def testFindNodesWithBadTensorValues(self):
    # [FUNCTION DOC]
    # Test: searching a dump for tensors with infinity or not-a-number values. Graph: w = u - v
    # gives [0, 3]; x = u / w gives [inf, 1.33]; y = w * x gives [nan, 4]; z = y * y. Runs with
    # whole-graph watching and the expected partition graph count. Uses a predicate that detects
    # NaN or infinity; asserts at least three offending data records including x, y and z, and
    # that restricting to the first match returns exactly one record.
    with session.Session() as sess:
      u_name = "testFindNodesWithBadTensorValues/u"
      v_name = "testFindNodesWithBadTensorValues/v"
      w_name = "testFindNodesWithBadTensorValues/w"
      x_name = "testFindNodesWithBadTensorValues/x"
      y_name = "testFindNodesWithBadTensorValues/y"
      z_name = "testFindNodesWithBadTensorValues/z"

      u_init = constant_op.constant([2.0, 4.0])
      u = variable_v1.VariableV1(u_init, name=u_name)
      v_init = constant_op.constant([2.0, 1.0])
      v = variable_v1.VariableV1(v_init, name=v_name)

      # Expected output: [0.0, 3.0]
      w = math_ops.subtract(u, v, name=w_name)

      # Expected output: [inf, 1.3333]
      x = math_ops.div(u, w, name=x_name)

      # Expected output: [nan, 4.0]
      y = math_ops.multiply(w, x, name=y_name)

      z = math_ops.multiply(y, y, name=z_name)

      u.initializer.run()
      v.initializer.run()

      _, dump = self._debug_run_and_get_dump(
          sess, z,
          expected_partition_graph_count=self._expected_partition_graph_count)

      def has_bad_value(_, tensor):
        # [FUNCTION DOC]
        # Predicate for the dump search: ignores the first argument (datum) and returns true if
        # the tensor value contains any not-a-number or infinite element.
        return np.any(np.isnan(tensor)) or np.any(np.isinf(tensor))

      # Find all "offending tensors".
      bad_data = dump.find(has_bad_value)

      # Verify that the nodes with bad values are caught through running find
      # on the debug dump.
      self.assertLessEqual(3, len(bad_data))
      node_names = [datum.node_name for datum in bad_data]
      self.assertIn(x_name, node_names)
      self.assertIn(y_name, node_names)
      self.assertIn(z_name, node_names)

      # Test first_n kwarg of find(): Find the first offending tensor.
      first_bad_datum = dump.find(has_bad_value, first_n=1)
      self.assertEqual(1, len(first_bad_datum))

  def testFindInfOrNanWithOpNameExclusion(self):
    # [FUNCTION DOC]
    # Test: the bad-value search can exclude nodes by name pattern. Same graph as the bad-value
    # search test (x has infinity, y and z have not-a-number). Searches with the library
    # infinity-or-NaN predicate while excluding node names ending in "/x"; asserts at least two
    # results that include y and z (x is excluded), and that limiting to the first match
    # returns exactly one record.
    with session.Session() as sess:
      u_name = "testFindInfOrNanWithOpNameExclusion/u"
      v_name = "testFindInfOrNanWithOpNameExclusion/v"
      w_name = "testFindInfOrNanWithOpNameExclusion/w"
      x_name = "testFindInfOrNanWithOpNameExclusion/x"
      y_name = "testFindInfOrNanWithOpNameExclusion/y"
      z_name = "testFindInfOrNanWithOpNameExclusion/z"

      u_init = constant_op.constant([2.0, 4.0])
      u = variable_v1.VariableV1(u_init, name=u_name)
      v_init = constant_op.constant([2.0, 1.0])
      v = variable_v1.VariableV1(v_init, name=v_name)

      # Expected output: [0.0, 3.0]
      w = math_ops.subtract(u, v, name=w_name)

      # Expected output: [inf, 1.3333]
      x = math_ops.div(u, w, name=x_name)

      # Expected output: [nan, 4.0]
      y = math_ops.multiply(w, x, name=y_name)

      z = math_ops.multiply(y, y, name=z_name)

      u.initializer.run()
      v.initializer.run()

      _, dump = self._debug_run_and_get_dump(
          sess, z,
          expected_partition_graph_count=self._expected_partition_graph_count)

      # Find all "offending tensors".
      bad_data = dump.find(debug_data.has_inf_or_nan,
                           exclude_node_names=".*/x$")

      # Verify that the nodes with bad values are caught through running find
      # on the debug dump.
      self.assertLessEqual(2, len(bad_data))
      # Assert that the node `x` should have been excluded.
      node_names = [datum.node_name for datum in bad_data]
      self.assertIn(y_name, node_names)
      self.assertIn(z_name, node_names)

      first_bad_datum = dump.find(
          debug_data.has_inf_or_nan, first_n=1, exclude_node_names=".*/x$")
      self.assertEqual(1, len(first_bad_datum))

  def _session_run_for_graph_structure_lookup(self):
    # [FUNCTION DOC]
    # HELPER: fixture shared by the graph-structure lookup tests. Builds a variable u (two
    # values), v = u + u and w = v + v with graph rewrites disabled, initializes u, and performs
    # a whole-graph debugged run asserting the expected partition graph count. Returns a tuple:
    # the names of u, v and w, and the dump object.
    with session.Session(config=no_rewrite_session_config()) as sess:
      u_name = "testDumpGraphStructureLookup/u"
      v_name = "testDumpGraphStructureLookup/v"
      w_name = "testDumpGraphStructureLookup/w"

      u_init = constant_op.constant([2.0, 4.0])
      u = variable_v1.VariableV1(u_init, name=u_name)
      v = math_ops.add(u, u, name=v_name)
      w = math_ops.add(v, v, name=w_name)

      u.initializer.run()

      _, dump = self._debug_run_and_get_dump(
          sess, w,
          expected_partition_graph_count=self._expected_partition_graph_count)

    return u_name, v_name, w_name, dump

  def testGraphStructureLookupGivesDevicesAndNodesInfo(self):
    # [FUNCTION DOC]
    # Test: device and node-existence queries on a dump. Asserts the number of devices equals the
    # expected count, the device of u equals the main device, querying the device of a
    # non-existent node raises a value error ("does not exist in partition graphs"), and node
    # existence is true for u and its read op but false for a made-up child name.
    u_name, _, _, dump = self._session_run_for_graph_structure_lookup()

    # Test num_devices().
    self.assertEqual(self._expected_num_devices, len(dump.devices()))

    # Test node_device().
    self.assertEqual(self._main_device, dump.node_device(u_name))

    with self.assertRaisesRegex(
        ValueError, "does not exist in partition graphs"
    ):
      dump.node_device(u_name + "foo")

    # Test node_exists().
    self.assertTrue(dump.node_exists(u_name))
    self.assertTrue(dump.node_exists(u_name + "/read"))
    self.assertFalse(dump.node_exists(u_name + "/read" + "/foo"))

  def testGraphStructureLookupGivesNodesAndAttributes(self):
    # [FUNCTION DOC]
    # Test: node list and attribute queries. Asserts u and its read op appear in the node list
    # (restricted to the GPU device when a GPU exists); the attributes of u give float32 dtype and
    # a one-dimensional shape of size two; querying attributes of an unknown node raises a value
    # error saying no device has such a node.
    u_name, _, _, dump = self._session_run_for_graph_structure_lookup()

    u_read_name = u_name + "/read"

    # Test node name list lookup of the DebugDumpDir object.
    if test_util.gpu_device_name():
      node_names = dump.nodes(
          device_name="/job:localhost/replica:0/task:0/device:GPU:0")
    else:
      node_names = dump.nodes()
    self.assertTrue(u_name in node_names)
    self.assertTrue(u_read_name in node_names)

    # Test querying node attributes.
    u_attr = dump.node_attributes(u_name)
    self.assertEqual(dtypes.float32, u_attr["dtype"].type)
    self.assertEqual(1, len(u_attr["shape"].shape.dim))
    self.assertEqual(2, u_attr["shape"].shape.dim[0].size)

    with self.assertRaisesRegex(
        ValueError, r"None of the .* device\(s\) has a node named "
    ):
      dump.node_attributes("foo")

  def testGraphStructureLookupGivesDebugWatchKeys(self):
    # [FUNCTION DOC]
    # Test: debug watch key queries. For u, v and w the watch keys are the single key of output
    # slot zero with the identity debug op; an unknown node gives an empty list. The data
    # records for u's key hold one record with the node name, slot zero, the identity debug op
    # and a non-negative timestamp; an unknown key gives an empty list.
    u_name, v_name, w_name, dump = (
        self._session_run_for_graph_structure_lookup())

    # Test querying the debug watch keys with node names.
    self.assertEqual(["%s:0:DebugIdentity" % u_name],
                     dump.debug_watch_keys(u_name))
    self.assertEqual(["%s:0:DebugIdentity" % v_name],
                     dump.debug_watch_keys(v_name))
    self.assertEqual(["%s:0:DebugIdentity" % w_name],
                     dump.debug_watch_keys(w_name))
    self.assertEqual([], dump.debug_watch_keys("foo"))

    # Test querying debug datum instances from debug watch.
    u_data = dump.watch_key_to_data(dump.debug_watch_keys(u_name)[0])
    self.assertEqual(1, len(u_data))
    self.assertEqual(u_name, u_data[0].node_name)
    self.assertEqual(0, u_data[0].output_slot)
    self.assertEqual("DebugIdentity", u_data[0].debug_op)
    self.assertGreaterEqual(u_data[0].timestamp, 0)
    self.assertEqual([], dump.watch_key_to_data("foo"))

  def testGraphStructureLookupGivesNodeInputsAndRecipients(self):
    # [FUNCTION DOC]
    # Test: input, recipient and transitive-input queries on the u, v, w graph. Asserts regular
    # inputs (the read op for u, v takes the read op twice, w takes v twice), no control inputs
    # or control recipients anywhere, recipients including the expected consumers with the
    # right multiplicities, invalid node names raising value errors, and transitive inputs
    # listing all upstream nodes (also raising for unknown names).
    u_name, v_name, w_name, dump = (
        self._session_run_for_graph_structure_lookup())

    u_read_name = u_name + "/read"

    # Test the inputs lookup of the DebugDumpDir object.
    self.assertEqual([], dump.node_inputs(u_name))
    self.assertEqual([u_name], dump.node_inputs(u_read_name))
    self.assertEqual([u_read_name] * 2, dump.node_inputs(v_name))
    self.assertEqual([v_name] * 2, dump.node_inputs(w_name))

    self.assertEqual([], dump.node_inputs(u_name, is_control=True))
    self.assertEqual([], dump.node_inputs(u_read_name, is_control=True))
    self.assertEqual([], dump.node_inputs(v_name, is_control=True))
    self.assertEqual([], dump.node_inputs(w_name, is_control=True))

    # Test the outputs recipient lookup of the DebugDumpDir object.
    self.assertTrue(u_read_name in dump.node_recipients(u_name))
    self.assertEqual(2, dump.node_recipients(u_read_name).count(v_name))
    self.assertEqual(2, dump.node_recipients(v_name).count(w_name))

    self.assertEqual([], dump.node_recipients(u_name, is_control=True))
    self.assertEqual([], dump.node_recipients(u_read_name, is_control=True))
    self.assertEqual([], dump.node_recipients(v_name, is_control=True))
    self.assertEqual([], dump.node_recipients(w_name, is_control=True))

    # Test errors raised on invalid node names.
    with self.assertRaisesRegex(
        ValueError, r"None of the .* device\(s\) has a node named "
    ):
      dump.node_inputs(u_name + "foo")
    with self.assertRaisesRegex(
        ValueError, r"None of the .* device\(s\) has a node named "
    ):
      dump.node_recipients(u_name + "foo")

    # Test transitive_inputs().
    self.assertEqual([], dump.transitive_inputs(u_name))
    self.assertEqual([u_name], dump.transitive_inputs(u_read_name))
    self.assertEqual(
        set([u_name, u_read_name]), set(dump.transitive_inputs(v_name)))
    self.assertEqual(
        set([u_name, u_read_name, v_name]), set(dump.transitive_inputs(w_name)))

    with self.assertRaisesRegex(
        ValueError, r"None of the .* device\(s\) has a node named "
    ):
      dump.transitive_inputs(u_name + "foo")

  def testGraphStructureLookupWithoutPartitionGraphsDoesNotErrorOut(self):
    # [FUNCTION DOC]
    # Test: a dump can be reloaded without supplying partition graphs. After a normal debugged
    # run, loads the dump directory again with validation off and no partition graphs, and
    # asserts that partition graphs were nevertheless loaded from the dump directory.
    _, _, _, dump = self._session_run_for_graph_structure_lookup()

    # Now load the dump again, without the partition graphs, so we can check
    # errors are not raised because the partition graphs are loaded from the
    # dump directory.
    dump = debug_data.DebugDumpDir(self._dump_root, validate=False)
    self.assertTrue(dump.loaded_partition_graphs())

  def testGraphPathFindingOnControlEdgesWorks(self):
    # [FUNCTION DOC]
    # Test: path finding across control edges. Graph: three variables, a = v1 + v2, and c =
    # v3 - v3 created under a control dependency on a. Asserts the path from v1 to c is v1, its
    # read, a, c, and that no path exists when control edges are excluded.
    with session.Session(config=no_rewrite_session_config()) as sess:
      v1 = variable_v1.VariableV1(1.0, name="v1")
      v2 = variable_v1.VariableV1(2.0, name="v2")
      v3 = variable_v1.VariableV1(3.0, name="v3")
      a = math_ops.add(v1, v2, name="a")
      with ops.control_dependencies([a]):
        c = math_ops.subtract(v3, v3, name="c")

      sess.run(variables.global_variables_initializer())
      _, dump = self._debug_run_and_get_dump(sess, c)

      self.assertEqual(["v1", "v1/read", "a", "c"],
                       dump.find_some_path("v1", "c"))
      self.assertIsNone(dump.find_some_path("v1", "c", include_control=False))

  def testGraphPathFindingReverseRefEdgeWorks(self):
    # [FUNCTION DOC]
    # Test: path finding through reference (reversed) edges. Graph: variable v, variable delta,
    # and an assign-add op inc_v. Asserts a path from delta to v exists (delta, its read, inc_v,
    # v) only when reversed reference edges are included, and none otherwise.
    with session.Session(config=no_rewrite_session_config()) as sess:
      v = variable_v1.VariableV1(10.0, name="v")
      delta = variable_v1.VariableV1(1.0, name="delta")
      inc_v = state_ops.assign_add(v, delta, name="inc_v")

      sess.run(variables.global_variables_initializer())
      _, dump = self._debug_run_and_get_dump(sess, inc_v)

      self.assertEqual(
          ["delta", "delta/read", "inc_v", "v"],
          dump.find_some_path("delta", "v", include_reversed_ref=True))
      self.assertIsNone(dump.find_some_path("delta", "v"))

  def testCausalityCheckOnDumpsDetectsWrongTemporalOrder(self):
    # [FUNCTION DOC]
    # Test: the dump loader detects causality violations. Runs the u, v, w graph with whole-graph
    # watching and checks loading works with and without partition graphs. Then renames the dump
    # files of v and w so that their timestamps are swapped (w's shifted just before v's) and
    # asserts loading with validation raises a value error mentioning violated causality, while
    # loading with validation off succeeds. Finally gives both files the same timestamp and
    # asserts loading succeeds again. Side effects: renames files in the dump directory.
    with session.Session(config=no_rewrite_session_config()) as sess:
      u_name = "testDumpCausalityCheck/u"
      v_name = "testDumpCausalityCheck/v"
      w_name = "testDumpCausalityCheck/w"

      u_init = constant_op.constant([2.0, 4.0])
      u = variable_v1.VariableV1(u_init, name=u_name)
      v = math_ops.add(u, u, name=v_name)
      w = math_ops.add(v, v, name=w_name)

      u.initializer.run()

      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      debug_utils.watch_graph(
          run_options,
          sess.graph,
          debug_ops=["DebugIdentity"],
          debug_urls=self._debug_urls())

      run_metadata = config_pb2.RunMetadata()
      sess.run(w, options=run_options, run_metadata=run_metadata)

      self.assertEqual(self._expected_partition_graph_count,
                       len(run_metadata.partition_graphs))

      # First, loading the original dump without supplying the
      # partition_graphs should not cause a LookupError, validation occurs
      # only with partition_graphs loaded.
      debug_data.DebugDumpDir(self._dump_root)

      # Now, loading the original dump with partition graphs supplied should
      # succeed. The validation should pass quietly.
      dump = debug_data.DebugDumpDir(
          self._dump_root, partition_graphs=run_metadata.partition_graphs)

      # Get the dump file names and compute their timestamps.
      self.assertEqual(
          1, len(dump.get_tensor_file_paths(v_name, 0, "DebugIdentity")))
      v_file_path = dump.get_tensor_file_paths(v_name, 0, "DebugIdentity")[0]

      self.assertEqual(
          1, len(dump.get_tensor_file_paths(w_name, 0, "DebugIdentity")))
      w_file_path = dump.get_tensor_file_paths(w_name, 0, "DebugIdentity")[0]

      v_timestamp = int(v_file_path[v_file_path.rindex("_") + 1:])
      w_timestamp = int(w_file_path[w_file_path.rindex("_") + 1:])

      # Swap and slightly shift the time stamps of the last two dumped tensors,
      # to simulate "causality violation", which can happen if the dump
      # directory contains incomplete data and/or mixes data from different
      # Session.run() calls.
      v_file_path_1 = v_file_path[:v_file_path.rindex(
          "_")] + "_%d" % w_timestamp
      w_file_path_1 = w_file_path[:w_file_path.rindex("_")] + "_%d" % (
          v_timestamp - 1)

      os.rename(v_file_path, v_file_path_1)
      os.rename(w_file_path, w_file_path_1)

      # Load the dump directory again. Now a ValueError is expected to be
      # raised due to the timestamp swap.
      with self.assertRaisesRegex(ValueError, "Causality violated"):
        dump = debug_data.DebugDumpDir(
            self._dump_root, partition_graphs=run_metadata.partition_graphs)

      # Loading the dump directory with kwarg "validate" set explicitly to
      # False should get rid of the error.
      dump = debug_data.DebugDumpDir(
          self._dump_root,
          partition_graphs=run_metadata.partition_graphs,
          validate=False)

      # Next, set the two times stamps to be the same, which should be fine.
      v_file_path_2 = v_file_path[:v_file_path.rindex(
          "_")] + "_%d" % w_timestamp
      w_file_path_2 = w_file_path[:w_file_path.rindex(
          "_")] + "_%d" % w_timestamp

      os.rename(v_file_path_1, v_file_path_2)
      os.rename(w_file_path_1, w_file_path_2)

      debug_data.DebugDumpDir(
          self._dump_root, partition_graphs=run_metadata.partition_graphs)

  def testWatchingOnlyOneOfTwoOutputSlotsDoesNotLeadToCausalityFailure(self):
    # [FUNCTION DOC]
    # Test: watching only one of a node's two output slots does not cause a validation failure.
    # Builds a unique-with-counts op over an integer variable (three outputs used partially),
    # adds derived ops, watches only slot zero of the unique op plus two downstream nodes, runs,
    # and loads the dump with validation enabled. Asserts loading succeeds and the dumped unique
    # values are one, three, seven.
    with session.Session() as sess:
      x_name = "oneOfTwoSlots/x"
      u_name = "oneOfTwoSlots/u"
      v_name = "oneOfTwoSlots/v"
      w_name = "oneOfTwoSlots/w"
      y_name = "oneOfTwoSlots/y"

      x = variable_v1.VariableV1([1, 3, 3, 7], dtype=dtypes.int32, name=x_name)
      sess.run(x.initializer)

      unique_x, indices, _ = array_ops.unique_with_counts(x, name=u_name)

      v = math_ops.add(unique_x, unique_x, name=v_name)
      w = math_ops.add(indices, indices, name=w_name)
      y = math_ops.add(w, w, name=y_name)

      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      # Watch only the first output slot of u, even though it has two output
      # slots.
      debug_utils.add_debug_tensor_watch(
          run_options, u_name, 0, debug_urls=self._debug_urls())
      debug_utils.add_debug_tensor_watch(
          run_options, w_name, 0, debug_urls=self._debug_urls())
      debug_utils.add_debug_tensor_watch(
          run_options, y_name, 0, debug_urls=self._debug_urls())

      run_metadata = config_pb2.RunMetadata()
      sess.run([v, y], options=run_options, run_metadata=run_metadata)

      dump = debug_data.DebugDumpDir(
          self._dump_root,
          partition_graphs=run_metadata.partition_graphs,
          validate=True)

      self.assertAllClose([1, 3, 7],
                          dump.get_tensors(u_name, 0, "DebugIdentity")[0])

  def testOutputSlotWithoutOutgoingEdgeCanBeWatched(self):
    # [FUNCTION DOC]
    # Test: an output tensor with no outgoing data edge can be watched. A constant u only runs
    # because of a control dependency from a no-op z. Runs z with whole-graph watching and
    # asserts exactly one tensor was dumped, for u, slot zero, identity debug op, with the
    # constant's value.
    """Test watching output slots not attached to any outgoing edges."""

    with session.Session(config=no_rewrite_session_config()) as sess:
      u_init_val = np.array([[5.0, 3.0], [-1.0, 0.0]])
      u = constant_op.constant(u_init_val, shape=[2, 2], name="u")

      # Create a control edge from a node with an output: From u to z.
      # Node u will get executed only because of the control edge. The output
      # tensor u:0 is not attached to any outgoing edge in the graph. This test
      # checks that the debugger can watch such a tensor.
      with ops.control_dependencies([u]):
        z = control_flow_ops.no_op(name="z")

      _, dump = self._debug_run_and_get_dump(sess, z)

      # Assert that the DebugIdentity watch on u works properly.
      self.assertEqual(1, len(dump.dumped_tensor_data))
      datum = dump.dumped_tensor_data[0]
      self.assertEqual("u", datum.node_name)
      self.assertEqual(0, datum.output_slot)
      self.assertEqual("DebugIdentity", datum.debug_op)
      self.assertAllClose([[5.0, 3.0], [-1.0, 0.0]], datum.get_tensor())

  def testWatchingVariableUpdateOpsSeesUpdatedValues(self):
    # [FUNCTION DOC]
    # Test: watching variable-update ops (which emit no edges) shows the updated values. Graph:
    # variables u (10) and v (20), w = u * v, one step of gradient descent with learning rate
    # 0.1. After a whole-graph debugged run of the training op asserts each update op was
    # dumped once with the updated values (8.0 for u, 19.0 for v), and that running the
    # variables afterwards returns the same updated values.
    """Watch output slots on Variable-updating ops, with no emitted edges."""

    with session.Session(config=no_rewrite_session_config()) as sess:
      u_init = constant_op.constant(10.0)
      u = variable_v1.VariableV1(u_init, name="gdo/u")
      v_init = constant_op.constant(20.0)
      v = variable_v1.VariableV1(v_init, name="gdo/v")

      w = math_ops.multiply(u, v, name="gdo/w")
      # gdo stands for GradientDescentOptimizer.

      train_op = gradient_descent.GradientDescentOptimizer(
          learning_rate=0.1).minimize(
              w, name="gdo/train")

      u.initializer.run()
      v.initializer.run()

      _, dump = self._debug_run_and_get_dump(sess, train_op)

      update_u_data = dump.watch_key_to_data(
          "gdo/train/update_gdo/u/ApplyGradientDescent:0:DebugIdentity")
      self.assertEqual(1, len(update_u_data))

      # Gradient descent on u: w = u * v, so dw / du = v.
      # Updated value of u should be:
      #   10.0 - learning_rate * v = 10.0 - 0.1 * 20.0 = 8.0
      self.assertAllClose(8.0, update_u_data[0].get_tensor())

      update_v_data = dump.watch_key_to_data(
          "gdo/train/update_gdo/v/ApplyGradientDescent:0:DebugIdentity")
      self.assertEqual(1, len(update_v_data))

      # Gradient descent on u: w = u * v, so dw / dv = u.
      # Updated value of u should be:
      #   20.0 - learning_rate * u = 20.0 - 0.1 * 10.0 = 19.0
      self.assertAllClose(19.0, update_v_data[0].get_tensor())

      # Verify that the Variables u and v are updated properly.
      self.assertAllClose(8.0, sess.run(u))
      self.assertAllClose(19.0, sess.run(v))

  def testAllowsWatchingUnconnectedOutputTensor(self):
    # [FUNCTION DOC]
    # Test: an output slot that emits no edges at all (not even control edges) can be watched.
    # Builds a unique op over a variable, uses only its slot zero in an add op, and verifies by
    # scanning the graph that slot zero has the add op as its only recipient and slot one has
    # none. After a whole-graph debugged run asserts the result is [2, 4, 7], and both slots
    # were dumped exactly once, with unique values for slot zero and index mapping for slot one.
    """Watch an output slot not emitting any edges.

    (Not even control edges from the node.)
    """

    with session.Session() as sess:
      x_init = constant_op.constant([2, 2, 3, 5, 5])
      x = variable_v1.VariableV1(x_init, name="unconnected/x")

      # The UniqueOp (tf.unique) has two output slots. Use only slot 0 in the
      # graph. Let the debugger watch the unused slot 1.
      unique_x, _ = array_ops.unique(x, name="unconnected/unique_x")
      y = math_ops.add(unique_x, [0, 1, 2], name="unconnected/y")

      x.initializer.run()

      # Verify that only slot 0 of unique_x has recipients, while slot 1 of the
      # same node does not have recipients.
      unique_x_slot_0_recipients = []
      unique_x_slot_1_recipients = []
      for op in sess.graph.get_operations():
        for inp in op.inputs:
          if inp.name == "unconnected/unique_x:0":
            unique_x_slot_0_recipients.append(op.name)
          elif inp.name == "unconnected/unique_x:1":
            unique_x_slot_1_recipients.append(op.name)

      self.assertEqual(["unconnected/y"], unique_x_slot_0_recipients)
      self.assertEqual([], unique_x_slot_1_recipients)

      y_result, dump = self._debug_run_and_get_dump(sess, y)
      self.assertAllClose([2, 4, 7], y_result)

      # Assert that the connected slot (slot 0) is dumped properly.
      unique_x_slot_0_dumps = dump.watch_key_to_data(
          "unconnected/unique_x:0:DebugIdentity")
      self.assertEqual(1, len(unique_x_slot_0_dumps))
      self.assertEqual("unconnected/unique_x",
                       unique_x_slot_0_dumps[0].node_name)
      self.assertEqual(0, unique_x_slot_0_dumps[0].output_slot)
      self.assertAllClose([2, 3, 5], unique_x_slot_0_dumps[0].get_tensor())

      # Assert that the unconnected slot (slot 1) is dumped properly.
      unique_x_slot_1_dumps = dump.watch_key_to_data(
          "unconnected/unique_x:1:DebugIdentity")
      self.assertEqual(1, len(unique_x_slot_1_dumps))
      self.assertEqual("unconnected/unique_x",
                       unique_x_slot_1_dumps[0].node_name)
      self.assertEqual(1, unique_x_slot_1_dumps[0].output_slot)
      self.assertAllClose([0, 0, 1, 2, 2],
                          unique_x_slot_1_dumps[0].get_tensor())

  def testSuccessiveDebuggingRunsIncreasesCounters(self):
    # [FUNCTION DOC]
    # Test: core metadata counters across successive debugged runs. Uses a placeholder feeding a
    # transpose and a squeeze. First run: global step one, executor step index zero, expected
    # input, output names and no targets. Second run with the same feed and fetch (dump
    # directory cleared between runs): session run index and executor step index each increase
    # by one. Third run with a different fetch: session run index increases by one but executor
    # step index is back to zero. Side effects: deletes the dump root between runs.
    """Test repeated Session.run() calls with debugger increments counters."""

    with session.Session() as sess:
      ph = array_ops.placeholder(dtypes.float32, name="successive/ph")
      x = array_ops.transpose(ph, name="mismatch/x")
      y = array_ops.squeeze(ph, name="mismatch/y")

      _, dump1 = self._debug_run_and_get_dump(
          sess, x, feed_dict={ph: np.array([[7.0, 8.0]])}, global_step=1)
      self.assertEqual(1, dump1.core_metadata.global_step)
      self.assertGreaterEqual(dump1.core_metadata.session_run_index, 0)
      self.assertEqual(0, dump1.core_metadata.executor_step_index)
      self.assertEqual([ph.name], dump1.core_metadata.input_names)
      self.assertEqual([x.name], dump1.core_metadata.output_names)
      self.assertEqual([], dump1.core_metadata.target_nodes)
      file_io.delete_recursively(self._dump_root)

      # Calling run() with the same feed, same output and same debug watch
      # options should increment both session_run_index and
      # executor_step_index.
      _, dump2 = self._debug_run_and_get_dump(
          sess, x, feed_dict={ph: np.array([[7.0, 8.0]])}, global_step=2)
      self.assertEqual(2, dump2.core_metadata.global_step)
      self.assertEqual(dump1.core_metadata.session_run_index + 1,
                       dump2.core_metadata.session_run_index)
      self.assertEqual(dump1.core_metadata.executor_step_index + 1,
                       dump2.core_metadata.executor_step_index)
      self.assertEqual([ph.name], dump2.core_metadata.input_names)
      self.assertEqual([x.name], dump2.core_metadata.output_names)
      self.assertEqual([], dump2.core_metadata.target_nodes)
      file_io.delete_recursively(self._dump_root)

      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      debug_utils.watch_graph(
          run_options, sess.graph, debug_urls=self._debug_urls(), global_step=3)

      # Calling run() with a different output should increment
      # session_run_index, but not executor_step_index.
      _, dump3 = self._debug_run_and_get_dump(
          sess, y, feed_dict={ph: np.array([[7.0, 8.0]])}, global_step=3)
      self.assertEqual(3, dump3.core_metadata.global_step)
      self.assertEqual(dump2.core_metadata.session_run_index + 1,
                       dump3.core_metadata.session_run_index)
      self.assertEqual(0, dump3.core_metadata.executor_step_index)
      self.assertEqual([ph.name], dump3.core_metadata.input_names)
      self.assertEqual([y.name], dump3.core_metadata.output_names)
      self.assertEqual([], dump3.core_metadata.target_nodes)

  def testDebuggingDuringOpError(self):
    # [FUNCTION DOC]
    # Test: tensors are still dumped when the run fails with an op error. Graph: placeholder,
    # transpose, constant 1x2 matrix, and a matrix multiplication whose shapes mismatch for the
    # fed value. Watches the whole graph and expects the run to raise an op error. Then loads the
    # dump without partition graphs and asserts the core metadata (non-negative indices, input
    # and output names, no targets), partition graphs loaded from the directory, and the dumps
    # of the constant and the transpose have the expected values.
    """Test the debug tensor dumping when error occurs in graph runtime."""

    with session.Session() as sess:
      ph = array_ops.placeholder(dtypes.float32, name="mismatch/ph")
      x = array_ops.transpose(ph, name="mismatch/x")
      m = constant_op.constant(
          np.array(
              [[1.0, 2.0]], dtype=np.float32), name="mismatch/m")
      y = math_ops.matmul(m, x, name="mismatch/y")

      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      debug_utils.watch_graph(
          run_options,
          sess.graph,
          debug_ops=["DebugIdentity"],
          debug_urls=self._debug_urls())

      with self.assertRaises(errors.OpError):
        sess.run(y,
                 options=run_options,
                 feed_dict={ph: np.array([[-3.0], [0.0]])})

      dump = debug_data.DebugDumpDir(self._dump_root)

      self.assertGreaterEqual(dump.core_metadata.session_run_index, 0)
      self.assertGreaterEqual(dump.core_metadata.executor_step_index, 0)
      self.assertEqual([ph.name], dump.core_metadata.input_names)
      self.assertEqual([y.name], dump.core_metadata.output_names)
      self.assertEqual([], dump.core_metadata.target_nodes)

      # Despite the fact that the run() call errored out and partition_graphs
      # are not available via run_metadata, the partition graphs should still
      # have been loaded from the dump directory.
      self.assertTrue(dump.loaded_partition_graphs())

      m_dumps = dump.watch_key_to_data("mismatch/m:0:DebugIdentity")
      self.assertEqual(1, len(m_dumps))
      self.assertAllClose(np.array([[1.0, 2.0]]), m_dumps[0].get_tensor())

      x_dumps = dump.watch_key_to_data("mismatch/x:0:DebugIdentity")
      self.assertEqual(1, len(x_dumps))
      self.assertAllClose(np.array([[-3.0, 0.0]]), x_dumps[0].get_tensor())

  def testDebugNumericSummaryOnInitializedTensorGivesCorrectResult(self):
    # [FUNCTION DOC]
    # Test: the numeric summary debug op on an initialized float tensor. A variable holds 18
    # values mixing NaN, infinities, zeros and positive/negative numbers; adds a zero variable.
    # After a debugged run with the numeric summary op, asserts the summary of the first
    # variable's read equals the expected fifteen numbers (initialized flag, element count,
    # counts of NaN, negative infinity, negative, zero, positive and positive infinity values,
    # min, max, mean, variance, dtype, number of dimensions and the dimension size).
    with session.Session(config=no_rewrite_session_config()) as sess:
      a = variable_v1.VariableV1([
          np.nan, np.nan, 0.0, 0.0, 0.0, -1.0, -3.0, 3.0, 7.0, -np.inf, -np.inf,
          np.inf, np.inf, np.inf, np.inf, np.inf, np.nan, np.nan
      ],
                                 dtype=np.float32,
                                 name="numeric_summary/a")
      b = variable_v1.VariableV1(
          [0.0] * 18, dtype=np.float32, name="numeric_summary/b")
      c = math_ops.add(a, b, name="numeric_summary/c")

      sess.run(variables.global_variables_initializer())

      _, dump = self._debug_run_and_get_dump(
          sess, c, debug_ops=["DebugNumericSummary"])
      self.assertTrue(dump.loaded_partition_graphs())

      self.assertAllClose([[
          1.0, 18.0, 4.0, 2.0, 2.0, 3.0, 2.0, 5.0, -3.0, 7.0, 0.85714286,
          8.97959184, 1.0, 1.0, 18.0
      ]], dump.get_tensors("numeric_summary/a/read", 0, "DebugNumericSummary"))

  def testDebugNumericSummaryOnUninitializedTensorGivesCorrectResult(self):
    # [FUNCTION DOC]
    # Test: the numeric summary debug op on an uninitialized variable. Runs the variable's
    # initializer while watching the variable. Asserts the summary has zeros for the first eight
    # entries, positive infinity as the minimum slot, negative infinity as the maximum slot,
    # NaN for mean and variance, and the trailing dtype, rank and size entries of one each.
    with session.Session() as sess:
      a = variable_v1.VariableV1([42],
                                 dtype=np.float32,
                                 name="numeric_summary_uninit/a")

      _, dump = self._debug_run_and_get_dump(
          sess, a.initializer, debug_ops=["DebugNumericSummary"])

      self.assertTrue(dump.loaded_partition_graphs())

      # DebugNumericSummary output should reflect the uninitialized state of
      # the watched tensor.
      numeric_summary = dump.get_tensors("numeric_summary_uninit/a", 0,
                                         "DebugNumericSummary")[0]
      self.assertAllClose([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          numeric_summary[0:8])
      # Check dtype (index 12), ndims (index 13) and dimension sizes (index
      # 14+).
      self.assertAllClose([1.0, 1.0, 1.0], numeric_summary[12:])
      self.assertTrue(np.isinf(numeric_summary[8]))
      self.assertGreater(numeric_summary[8], 0.0)
      self.assertTrue(np.isinf(numeric_summary[9]))
      self.assertLess(numeric_summary[9], 0.0)
      self.assertTrue(np.isnan(numeric_summary[10]))
      self.assertTrue(np.isnan(numeric_summary[11]))

  def testDebugNumericSummaryFailureIsToleratedWhenOrdered(self):
    # [FUNCTION DOC]
    # Test: failures to create numeric summary debug ops on string tensors can be tolerated.
    # Graph: string variables concatenated, converted to a number n and doubled to m. Watching
    # the whole graph with the numeric summary op raises a failed-precondition error by default;
    # with the tolerate-creation-failures option the run succeeds with result 264, and the
    # integer tensors n and m still have numeric summary watch keys.
    with session.Session() as sess:
      a = variable_v1.VariableV1("1", name="a")
      b = variable_v1.VariableV1("3", name="b")
      c = variable_v1.VariableV1("2", name="c")

      d = math_ops.add(a, b, name="d")
      e = math_ops.add(d, c, name="e")
      n = parsing_ops.string_to_number(e, name="n")
      m = math_ops.add(n, n, name="m")

      sess.run(variables.global_variables_initializer())

      # Using DebugNumericSummary on sess.run(m) with the default
      # tolerate_debug_op_creation_failures=False should error out due to the
      # presence of string-dtype Tensors in the graph.
      run_metadata = config_pb2.RunMetadata()
      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      debug_utils.watch_graph(
          run_options,
          sess.graph,
          debug_ops=["DebugNumericSummary"],
          debug_urls=self._debug_urls())
      with self.assertRaises(errors.FailedPreconditionError):
        sess.run(m, options=run_options, run_metadata=run_metadata)

      # Using tolerate_debug_op_creation_failures=True should get rid of the
      # error.
      m_result, dump = self._debug_run_and_get_dump(
          sess, m, debug_ops=["DebugNumericSummary"],
          tolerate_debug_op_creation_failures=True)
      self.assertEqual(264, m_result)

      # The integer-dtype Tensors in the graph should have been dumped
      # properly.
      self.assertIn("n:0:DebugNumericSummary", dump.debug_watch_keys("n"))
      self.assertIn("m:0:DebugNumericSummary", dump.debug_watch_keys("m"))

  def testDebugNumericSummaryInvalidAttributesStringAreCaught(self):
    # [FUNCTION DOC]
    # Test: invalid attributes in the numeric summary debug op specification are reported.
    # Builds a graph with a division by zero, then runs with three specifications: one unknown
    # attribute, two unknown attributes, and one unknown attribute mixed with a valid one. Each
    # run must raise a failed-precondition error saying how many attribute keys were not valid
    # for the debug node (and, when only one, naming the key).
    with session.Session(config=no_rewrite_session_config()) as sess:
      a = variable_v1.VariableV1(10.0, name="a")
      b = variable_v1.VariableV1(0.0, name="b")
      c = variable_v1.VariableV1(0.0, name="c")

      x = math_ops.divide(a, b, name="x")
      y = math_ops.multiply(x, c, name="y")

      sess.run(variables.global_variables_initializer())

      run_metadata = config_pb2.RunMetadata()
      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      debug_utils.watch_graph(
          run_options,
          sess.graph,
          debug_ops=["DebugNumericSummary(foo=1.0)"],
          debug_urls=self._debug_urls())
      with self.assertRaisesRegex(
          errors.FailedPreconditionError,
          r"1 attribute key\(s\) were not valid for debug node "
          r"__dbg_.:0_0_DebugNumericSummary: foo",
      ):
        sess.run(y, options=run_options, run_metadata=run_metadata)

      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      debug_utils.watch_graph(
          run_options,
          sess.graph,
          debug_ops=["DebugNumericSummary(foo=1.0; bar=false)"],
          debug_urls=self._debug_urls())
      with self.assertRaisesRegex(
          errors.FailedPreconditionError,
          r"2 attribute key\(s\) were not valid for debug node "
          r"__dbg_.:0_0_DebugNumericSummary:",
      ):
        sess.run(y, options=run_options, run_metadata=run_metadata)

      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      debug_utils.watch_graph(
          run_options,
          sess.graph,
          debug_ops=["DebugNumericSummary(foo=1.0; mute_if_healthy=true)"],
          debug_urls=self._debug_urls())
      with self.assertRaisesRegex(
          errors.FailedPreconditionError,
          r"1 attribute key\(s\) were not valid for debug node "
          r"__dbg_.:0_0_DebugNumericSummary: foo",
      ):
        sess.run(y, options=run_options, run_metadata=run_metadata)

  def testDebugNumericSummaryMuteOnHealthyMutesOnlyHealthyTensorDumps(self):
    # [FUNCTION DOC]
    # Test: the mute-if-healthy option suppresses dumps of healthy tensors only. Graph: a/b with
    # b zero (infinity) and times c (NaN). Runs with the option on and validation off (to avoid a
    # causality failure caused by muted nodes), asserts at least two dumps and the exact summary
    # values for the unhealthy nodes x and y. After clearing the dump root, runs again with the
    # default (not muted) setting and asserts at least eight dumps.
    with session.Session(config=no_rewrite_session_config()) as sess:
      a = variable_v1.VariableV1(10.0, name="a")
      b = variable_v1.VariableV1(0.0, name="b")
      c = variable_v1.VariableV1(0.0, name="c")

      x = math_ops.divide(a, b, name="x")
      y = math_ops.multiply(x, c, name="y")

      sess.run(variables.global_variables_initializer())

      # Here, validate=False is necessary to avoid causality check error.
      # TODO(cais): Maybe let DebugDumpDir constructor automatically ignore
      #   debug ops with mute_if_healthy=false attribute during validation.
      _, dump = self._debug_run_and_get_dump(
          sess, y, debug_ops=["DebugNumericSummary(mute_if_healthy=true)"],
          validate=False)

      self.assertLessEqual(2, dump.size)
      self.assertAllClose([[
          1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, np.inf, -np.inf, np.nan,
          np.nan, 1.0, 0.0
      ]], dump.get_tensors("x", 0, "DebugNumericSummary"))
      self.assertAllClose([[
          1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.inf, -np.inf, np.nan,
          np.nan, 1.0, 0.0
      ]], dump.get_tensors("y", 0, "DebugNumericSummary"))

      # Another run with the default mute_if_healthy (false) value should
      # dump all the tensors.
      file_io.delete_recursively(self._dump_root)
      _, dump = self._debug_run_and_get_dump(
          sess, y, debug_ops=["DebugNumericSummary()"])
      self.assertLessEqual(8, dump.size)

  def testDebugNumericSummaryMuteOnHealthyAndCustomBoundsWork(self):
    # [FUNCTION DOC]
    # Test: mute-if-healthy combined with a custom upper bound. Graph: x = a + b gives [20, 12],
    # y = x / b gives [2, 6]. With the upper bound at 11 only x counts as unhealthy, so the dump
    # holds exactly one tensor (x) with the expected summary numbers. Validation is turned off
    # to avoid a causality check failure.
    with session.Session() as sess:
      a = variable_v1.VariableV1([10.0, 10.0], name="a")
      b = variable_v1.VariableV1([10.0, 2.0], name="b")

      x = math_ops.add(a, b, name="x")  # [20.0, 12.0]
      y = math_ops.divide(x, b, name="y")  # [2.0, 6.0]

      sess.run(variables.global_variables_initializer())

      # Here, validate=False is necessary to avoid causality check error.
      # TODO(cais): Maybe let DebugDumpDir constructor automatically ignore
      #   debug ops with mute_if_healthy=false attribute during validation.
      _, dump = self._debug_run_and_get_dump(
          sess, y, debug_ops=[
              "DebugNumericSummary(mute_if_healthy=true; upper_bound=11.0)"],
          validate=False)

      self.assertEqual(1, dump.size)
      self.assertAllClose([[
          1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 12.0, 20.0, 16.0, 16.0, 1.0,
          1.0, 2.0]], dump.get_tensors("x", 0, "DebugNumericSummary"))

  def testDebugQueueOpsDoesNotoErrorOut(self):
    # [FUNCTION DOC]
    # Test: watching queue ops does not error out (the test name has a typo in "Noto"). Creates
    # a first-in-first-out queue of capacity three and an enqueue-many op, runs the enqueue
    # with whole-graph watching, and asserts partition graphs were loaded, the queue handle
    # tensor is dumped as an initialized but unconvertible tensor, and the enqueued component
    # values equal 101, 202, 303.
    with session.Session() as sess:
      q = data_flow_ops.FIFOQueue(3, "float", name="fifo_queue")
      q_init = q.enqueue_many(([101.0, 202.0, 303.0],), name="enqueue_many")

      _, dump = self._debug_run_and_get_dump(sess, q_init)
      self.assertTrue(dump.loaded_partition_graphs())

      fifo_queue_tensor = dump.get_tensors("fifo_queue", 0, "DebugIdentity")[0]
      self.assertIsInstance(fifo_queue_tensor,
                            debug_data.InconvertibleTensorProto)
      self.assertTrue(fifo_queue_tensor.initialized)
      self.assertAllClose(
          [101.0, 202.0, 303.0],
          dump.get_tensors("enqueue_many/component_0", 0, "DebugIdentity")[0])

  def testLookUpNodePythonTracebackWorks(self):
    # [FUNCTION DOC]
    # Test: looking up the Python creation traceback of nodes in a dump. After a debugged run,
    # asserts that lookup before setting the Python graph raises a lookup error; after setting
    # the graph from the session, an unknown node raises a key error; and lookup by node name and
    # by tensor name returns a non-empty tuple whose entries are tuples.
    with session.Session() as sess:
      u_init = constant_op.constant(10.0)
      u = variable_v1.VariableV1(u_init, name="traceback/u")
      v_init = constant_op.constant(20.0)
      v = variable_v1.VariableV1(v_init, name="traceback/v")

      w = math_ops.multiply(u, v, name="traceback/w")

      sess.run(variables.global_variables_initializer())
      _, dump = self._debug_run_and_get_dump(sess, w)

      # Prior to setting the Python graph, attempts to do traceback lookup
      # should lead to exceptions.
      with self.assertRaisesRegex(
          LookupError, "Python graph is not available for traceback lookup"
      ):
        dump.node_traceback("traceback/w")

      dump.set_python_graph(sess.graph)

      # After setting the Python graph, attempts to look up nonexistent nodes
      # should lead to exceptions.
      with self.assertRaisesRegex(
          KeyError, r"Cannot find node \"foo\" in Python graph"
      ):
        dump.node_traceback("foo")

      # Lookup should work with node name input.
      traceback = dump.node_traceback("traceback/w")
      self.assertIsInstance(traceback, tuple)
      self.assertGreater(len(traceback), 0)
      for trace in traceback:
        self.assertIsInstance(trace, tuple)

      # Lookup should also work with tensor name input.
      traceback = dump.node_traceback("traceback/w:0")
      self.assertIsInstance(traceback, tuple)
      self.assertGreater(len(traceback), 0)
      for trace in traceback:
        self.assertIsInstance(trace, tuple)


class DebugConcurrentRunCallsTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test base class for debugging several concurrent session run calls. Concrete subclasses must
  # provide the number of concurrent runs, per-run dump root directories and the per-run debug
  # URLs through the abstract hook below.
  """Test for debugging concurrent Session.run() calls."""

  def _get_concurrent_debug_urls(self):
    # [FUNCTION DOC]
    # HELPER: abstract hook for subclasses. Should return one debug URL list per concurrent run.
    # The base implementation always raises a not-implemented error.
    """Abstract method to generate debug URLs for concurrent debugged runs."""
    raise NotImplementedError(
        "_get_concurrent_debug_urls is not implemented in the base test class")

  def testDebugConcurrentVariableUpdates(self):
    # [FUNCTION DOC]
    # Test: debugging concurrent session runs that update one variable. Skipped when a GPU is
    # available. Uses subclass-provided attributes (number of concurrent runs, dump roots).
    # Setup: variable v (30.0) and one locked assign-add op per concurrent run, each adding one.
    # Action: starts one thread per run; each thread watches the whole graph with its own debug
    # URLs and runs its increment op one hundred times; then joins all threads. Asserts the final
    # variable value equals 30 plus one hundred per run. For each run's dump directory asserts
    # partition graphs loaded, one hundred dumps of v, and from the core metadata files (ordered
    # by wall time) that executor step index grows by exactly one at a time and session run
    # index strictly increases. Finally asserts the session run indices of all runs together
    # are unique.
    if test.is_gpu_available():
      self.skipTest("No testing concurrent runs on a single GPU.")

    with session.Session() as sess:
      v = variable_v1.VariableV1(30.0, name="v")
      constants = []
      for i in range(self._num_concurrent_runs):
        constants.append(constant_op.constant(1.0, name="c%d" % i))
      incs = [
          state_ops.assign_add(
              v, c, use_locking=True, name=("inc%d" % i))
          for (i, c) in enumerate(constants)
      ]
      sess.run(v.initializer)

      concurrent_debug_urls = self._get_concurrent_debug_urls()

      def inc_job(index):
        # [FUNCTION DOC]
        # Thread body: for the given run index, builds run options watching the whole graph with
        # that run's debug URLs, then runs that run's increment op one hundred times. Returns
        # nothing.
        run_options = config_pb2.RunOptions(output_partition_graphs=True)
        debug_utils.watch_graph(
            run_options, sess.graph, debug_urls=concurrent_debug_urls[index])
        for _ in range(100):
          sess.run(incs[index], options=run_options)

      inc_threads = []
      for index in range(self._num_concurrent_runs):
        inc_thread = threading.Thread(target=functools.partial(inc_job, index))
        inc_thread.start()
        inc_threads.append(inc_thread)
      for inc_thread in inc_threads:
        inc_thread.join()

      self.assertAllClose(30.0 + 1.0 * self._num_concurrent_runs * 100,
                          sess.run(v))

      all_session_run_indices = []
      for index in range(self._num_concurrent_runs):
        dump = debug_data.DebugDumpDir(self._dump_roots[index])
        self.assertTrue(dump.loaded_partition_graphs())

        v_data = dump.get_tensors("v", 0, "DebugIdentity")
        self.assertEqual(100, len(v_data))

        # Examine all the core metadata files
        core_metadata_files = glob.glob(
            os.path.join(self._dump_roots[index], "_tfdbg_core*"))

        timestamps = []
        session_run_indices = []
        executor_step_indices = []
        for core_metadata_file in core_metadata_files:
          with open(core_metadata_file, "rb") as f:
            event = event_pb2.Event()
            event.ParseFromString(f.read())
            core_metadata = (
                debug_data.extract_core_metadata_from_event_proto(event))
            timestamps.append(event.wall_time)
            session_run_indices.append(core_metadata.session_run_index)
            executor_step_indices.append(core_metadata.executor_step_index)

        all_session_run_indices.extend(session_run_indices)

        # Assert that executor_step_index increases by one at a time.
        executor_step_indices = zip(timestamps, executor_step_indices)
        executor_step_indices = sorted(
            executor_step_indices, key=lambda x: x[0])
        for i in range(len(executor_step_indices) - 1):
          self.assertEqual(
              executor_step_indices[i][1] + 1, executor_step_indices[i + 1][1]
          )

        # Assert that session_run_index increase monotonically.
        session_run_indices = zip(timestamps, session_run_indices)
        session_run_indices = sorted(session_run_indices, key=lambda x: x[0])
        for i in range(len(session_run_indices) - 1):
          self.assertGreater(session_run_indices[i + 1][1],
                             session_run_indices[i][1])

      # Assert that the session_run_indices from the concurrent run() calls are
      # all unique.
      self.assertEqual(len(all_session_run_indices),
                       len(set(all_session_run_indices)))


if __name__ == "__main__":
  googletest.main()
