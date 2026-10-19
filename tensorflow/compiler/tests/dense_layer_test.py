# [MODULE DOC]
# Tests that a dense (fully connected) layer is compiled by XLA into a single compile/run op pair,
# both in global auto-JIT mode and inside an explicit JIT scope (static and undefined input
# shapes). Provides small helpers to read profiling labels from run metadata. Defines a module-level
# alias for the experimental JIT scope. When run as the main program it sets the environment flag
# for lazy XLA compilation and disables eager execution, since the tests use sessions.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for DenseLayer JIT compilation on the CPU and GPU devices."""

import os

import numpy as np

from tensorflow.compiler.tests import test_utils
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.compiler.xla import jit
from tensorflow.python.framework import ops
from tensorflow.python.layers import layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

jit_scope = jit.experimental_jit_scope

def GetRunMetadataLabels(run_metadata):
  # [FUNCTION DOC]
  # HELPER: collects execution trace labels for the dense layer tests.
  # Takes a run-metadata message and returns a list of the timeline label of every node
  # statistic across all devices, in order. No side effects.
  """Returns all labels in run_metadata."""
  labels = []
  for dev_stats in run_metadata.step_stats.dev_stats:
    for node_stats in dev_stats.node_stats:
      labels.append(node_stats.timeline_label)
  return labels


def InLabels(labels, substr):
  # [FUNCTION DOC]
  # HELPER: substring search over trace labels for the dense layer tests.
  # Returns true if any label in the list contains the given substring, false otherwise.
  """Returns true iff one of the labels contains substr."""
  return any(substr in x for x in labels)


class DenseLayerTest(test.TestCase):
  # [CLASS DOC]
  # Test case verifying XLA JIT compilation of a dense layer on CPU/GPU, by inspecting trace labels
  # of a fully traced session run for compile and run ops.

  def countXlaOps(self, labels):
    # [FUNCTION DOC]
    # HELPER: counts compiled-cluster executions in a list of trace labels.
    # Counts labels containing the compile marker and those containing the run marker, asserts the
    # two counts are equal, and returns the run count.
    """Count how many XlaCompile/XlaRun labels are present."""
    xla_compile_count = sum("XlaCompile(" in x for x in labels)
    xla_run_count = sum("XlaRun(" in x for x in labels)
    self.assertEqual(xla_compile_count, xla_run_count)
    return xla_run_count


  def testDenseLayerAutoJit(self):
    # [FUNCTION DOC]
    # Side effect: prepends a flag enabling global CPU JIT to the XLA flags environment variable
    # (never restored). Enables global JIT level ON_1 in the session config, builds a dense layer
    # with 3 units over a placeholder of unknown batch and length, runs it with warmup under full
    # tracing, and asserts exactly one compile/run pair appears and no matrix-multiply label does.
    """Tests dense layer compilation in auto-jit mode.

    Dense layer should be compiled into a single XlaCompile/XlaRun op pair in
    auto-jit mode.
    """

    os.environ["TF_XLA_FLAGS"] = (
        "--tf_xla_cpu_global_jit " + os.environ.get("TF_XLA_FLAGS", ""))
    config = config_pb2.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = (
        config_pb2.OptimizerOptions.ON_1)

    with self.session(config=config) as sess:
      x = array_ops.placeholder(shape=[None, None, 3], dtype=np.float32)
      y = layers.dense(x, 3)

      self.evaluate(variables.global_variables_initializer())
      run_metadata = config_pb2.RunMetadata()
      test_utils.RunWithWarmup(
          sess,
          y, {x: np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])},
          run_metadata=run_metadata,
          options=config_pb2.RunOptions(
              trace_level=config_pb2.RunOptions.FULL_TRACE))

    labels = GetRunMetadataLabels(run_metadata)
    self.assertEqual(1, self.countXlaOps(labels))
    self.assertFalse(InLabels(labels, "MatMult"))

  def testDenseLayerJitScopeDefinedShape(self):
    # [FUNCTION DOC]
    # Builds a dense layer (3 units) inside an experimental JIT scope on a placeholder of fully
    # static shape 2x2x3, runs it with warmup under full tracing, and asserts exactly one
    # compile/run pair. Does not check for matrix-multiply labels (no list-difference op needed).
    """Tests that the dense layer node is properly compiled in jit scope.

    Dense layer with static shape input tensor should be compiled into a single
    XlaCompile/XlaRun op pair by XLA.
    """

    with self.session() as sess:
      x = array_ops.placeholder(shape=[2, 2, 3], dtype=np.float32)
      with jit_scope():
        y = layers.dense(x, 3)

      self.evaluate(variables.global_variables_initializer())
      run_metadata = config_pb2.RunMetadata()
      test_utils.RunWithWarmup(
          sess,
          y, {x: np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])},
          run_metadata=run_metadata,
          options=config_pb2.RunOptions(
              trace_level=config_pb2.RunOptions.FULL_TRACE))

    labels = GetRunMetadataLabels(run_metadata)
    self.assertEqual(1, self.countXlaOps(labels))
    # No need to check whether ListDiff is compiled or not because ListDiff op
    # is not used when input tensor shape is fully defined.

  def testDenseLayerJitScopeUndefinedShape(self):
    # [FUNCTION DOC]
    # Same as the defined-shape test but the placeholder has unknown batch and length dimensions.
    # Asserts one compile/run pair and that no matrix-multiply label appears in the trace.
    """Tests that the dense layer node is properly compiled in jit scope.
    """

    with self.session() as sess:
      x = array_ops.placeholder(shape=[None, None, 3], dtype=np.float32)
      with jit_scope():
        y = layers.dense(x, 3)

      self.evaluate(variables.global_variables_initializer())
      run_metadata = config_pb2.RunMetadata()
      test_utils.RunWithWarmup(
          sess,
          y, {x: np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])},
          run_metadata=run_metadata,
          options=config_pb2.RunOptions(
              trace_level=config_pb2.RunOptions.FULL_TRACE))

    labels = GetRunMetadataLabels(run_metadata)
    self.assertEqual(1, self.countXlaOps(labels))
    self.assertFalse(InLabels(labels, "MatMult"))


if __name__ == "__main__":
  os.environ["TF_XLA_FLAGS"] = ("--tf_xla_enable_lazy_compilation=true " +
                                os.environ.get("TF_XLA_FLAGS", ""))
  # This test is using Tensorflow sessions which are not compatible with eager
  # mode.
  ops.disable_eager_execution()
  test.main()
