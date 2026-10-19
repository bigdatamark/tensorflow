# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# [MODULE DOC]
# Integration tests for TF-TensorRT checking how the per-node max batch size annotation on graph
# nodes affects the max batch size attribute of the generated TRT engine ops. Defines a skipped
# base class plus a test that annotates nodes and a test with plain static shapes. Built on the
# shared TfTrtIntegrationTestBase; runs via the standard test main.
"""Testing the impact of graph node _tftrt_op_max_batch_size annotation on TRTEngineOp attributes."""

import unittest

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class MaxBatchSizesTestBase(trt_test.TfTrtIntegrationTestBase):
  # [CLASS DOC]
  # HELPER: shared base for the max-batch-size tests.
  # Defines four tensor shapes with different batch sizes, test parameters, restricts to static
  # engines and expects one engine per batch size with matching max batch size. Skipped itself.

  @classmethod
  def setUpClass(cls):
    # [FUNCTION DOC]
    # Skips the test class when it is the abstract base itself (raises a skip signal); otherwise
    # delegates to the parent class setup.
    if cls is MaxBatchSizesTestBase:
      raise unittest.SkipTest(
          'MaxBatchSizesTestBase defines base class for other tests.')
    super(MaxBatchSizesTestBase, cls).setUpClass()

  @property
  def tensor_shapes(self):
    # [FUNCTION DOC]
    # Property: the four tensor shapes used by the graph (batch sizes 1, 64, 32 and 16).
    return [[1, 512, 1, 1], [64, 2, 2, 2], [32, 4, 2, 2], [16, 8, 2, 2]]

  @property
  def max_batch_sizes(self):
    # [FUNCTION DOC]
    # Property: the first dimension (batch size) of each tensor shape, as a list.
    return [shape[0] for shape in self.tensor_shapes]

  def GetParams(self):
    # [FUNCTION DOC]
    # Builds the test parameters: float32 graph function, input shape equal to the first tensor
    # shape and output shape equal to the last tensor shape.
    """Gets the build parameters for the test."""
    return self.BuildParams(
        self.GraphFn,
        dtype=dtypes.float32,
        input_shapes=[self.tensor_shapes[0]],
        output_shapes=[self.tensor_shapes[-1]])

  def ShouldRunTest(self, run_params):
    # [FUNCTION DOC]
    # Runs the test only for static engines (not dynamic engines), because dynamic engines use the
    # batch size detected at run time. Returns a pair of a boolean and a reason string.
    # The maximum batch size for dynamic engines will be the actual batch size
    # detected at runtime. Therefore, we don't run the test with dynamic
    # engines.
    return (not run_params.dynamic_engine, 'test static engine only.')

  def GetMaxBatchSize(self, run_params):
    # [FUNCTION DOC]
    # Returns the converter max batch size: nothing for dynamic engines, else the smallest of the
    # batch sizes.
    """Returns the max_batch_size that the converter should use for tests."""
    if run_params.dynamic_engine:
      return None
    return min(self.max_batch_sizes)

  def ExpectedEnginesToBuild(self, run_params):
    # [FUNCTION DOC]
    # Returns the expected engine names: one engine per batch size, named by zero-padded sequence
    # number starting at zero.
    """Checks that the expected engine is built.

    Args:
      run_params: the run parameters.

    Returns:
      the expected engines to build.

    There shall be engines generated for each maximum batch size.
    """
    return [
        f'TRTEngineOp_{seq_id:03d}'
        for seq_id in range(len(self.max_batch_sizes))
    ]

  def ExpectedMaxBatchSizes(self, run_params):
    # [FUNCTION DOC]
    # Returns the expected max batch size attribute of each generated engine: the list of batch
    # sizes of the tensor shapes.
    """Checks that the expected maximum batch sizes for the generated engines.

    Args:
      run_params: the run parameters.

    Returns:
      the expected maximum batch sizes for the generated engines.

    There shall be engines generated for each maximum batch size.
    """
    return self.max_batch_sizes


class AnnotateMaxBatchSizesTest(MaxBatchSizesTestBase):
  # [CLASS DOC]
  # Test where successive stages of the graph are reshaped to different batch sizes and each stage
  # is annotated with its own max batch size; expects each engine to carry its stage's value.

  def GraphFn(self, inp):
    # [FUNCTION DOC]
    # Builds the graph: multiply, reshape, add, reshape, multiply, reshape, add-in-place, with the
    # add and multiply stages and final stage wrapped in an attribute scope setting the max batch
    # size annotation to the 2nd, 3rd and 4th batch sizes. Returns the identity output tensor.
    """Builds a tf.Graph for the test."""
    tensor = inp * 2.0
    tensor = array_ops.reshape(tensor, [-1] + self.tensor_shapes[1][1:])
    with ops.get_default_graph()._attr_scope({
        '_tftrt_op_max_batch_size':
            attr_value_pb2.AttrValue(i=self.max_batch_sizes[1])
    }):
      tensor = tensor + 3.0
    tensor = array_ops.reshape(tensor, [-1] + self.tensor_shapes[2][1:])
    with ops.get_default_graph()._attr_scope({
        '_tftrt_op_max_batch_size':
            attr_value_pb2.AttrValue(i=self.max_batch_sizes[2])
    }):
      tensor = tensor * 4.0
    tensor = array_ops.reshape(tensor, [-1] + self.tensor_shapes[3][1:])
    with ops.get_default_graph()._attr_scope({
        '_tftrt_op_max_batch_size':
            attr_value_pb2.AttrValue(i=self.max_batch_sizes[3])
    }):
      tensor += tensor + 5.0
    return array_ops.identity(tensor, name='output_0')


class StaticBatchSizeTest(MaxBatchSizesTestBase):
  # [CLASS DOC]
  # Same graph shape as the annotation test but with no annotations, reshaping to fully static
  # shapes; expects the max batch size of each engine to follow from the static shapes.

  def GraphFn(self, inp):
    # [FUNCTION DOC]
    # Builds the graph: multiply, reshape to static shape, add, reshape, multiply, reshape,
    # add-in-place, no annotations. Returns the identity output tensor.
    """Builds a tf.Graph for the test."""
    tensor = inp * 2.0
    tensor = array_ops.reshape(tensor, self.tensor_shapes[1])
    tensor = tensor + 3.0
    tensor = array_ops.reshape(tensor, self.tensor_shapes[2])
    tensor = tensor * 4.0
    tensor = array_ops.reshape(tensor, self.tensor_shapes[3])
    tensor += tensor + 5.0
    return array_ops.identity(tensor, name='output_0')


if __name__ == '__main__':
  test.main()
