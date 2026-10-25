# [MODULE DOC]
# Tests for the SavedModel tensor-info utilities exposed through the utils facade: building
# tensor-info messages for ops, dense, sparse and ragged tensors, and resolving them back to
# graph tensors, including other graphs, import scopes and error cases. Most tests run only
# under v1 graph mode. No import-time side effects.
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
"""Tests for SavedModel utils."""

from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.saved_model import utils


class UtilsTest(test.TestCase):
  # [CLASS DOC]
  # Test case for building and resolving tensor-info messages. No shared fixtures; each test
  # creates its own graph tensors.

  @test_util.run_v1_only(
      "b/120545219: `build_tensor_info` is only available in graph mode.")
  def testBuildTensorInfoOp(self):
    # [FUNCTION DOC]
    # Groups two constants into an operation and builds a tensor-info from it. Asserts the name
    # is the op name, the dtype is the invalid marker and the shape has no dimensions.
    x = constant_op.constant(1, name="x")
    y = constant_op.constant(2, name="y")
    z = control_flow_ops.group([x, y], name="op_z")
    z_op_info = utils.build_tensor_info_from_op(z)
    self.assertEqual("op_z", z_op_info.name)
    self.assertEqual(types_pb2.DT_INVALID, z_op_info.dtype)
    self.assertEqual(0, len(z_op_info.tensor_shape.dim))

  @test_util.run_v1_only(
      "b/120545219: `build_tensor_info` is only available in graph mode.")
  def testBuildTensorInfoDense(self):
    # [FUNCTION DOC]
    # Builds tensor info for a float placeholder of length one; asserts name "x:0", float dtype
    # and a single dimension of size one.
    x = array_ops.placeholder(dtypes.float32, 1, name="x")
    x_tensor_info = utils.build_tensor_info(x)
    self.assertEqual("x:0", x_tensor_info.name)
    self.assertEqual(types_pb2.DT_FLOAT, x_tensor_info.dtype)
    self.assertEqual(1, len(x_tensor_info.tensor_shape.dim))
    self.assertEqual(1, x_tensor_info.tensor_shape.dim[0].size)

  @test_util.run_v1_only(
      "b/120545219: `build_tensor_info` is only available in graph mode.")
  def testBuildTensorInfoSparse(self):
    # [FUNCTION DOC]
    # Builds tensor info for a sparse placeholder of shape 42 by 69; asserts the stored values,
    # indices and dense-shape tensor names, the float dtype and the two dimension sizes.
    x = array_ops.sparse_placeholder(dtypes.float32, [42, 69], name="x")
    x_tensor_info = utils.build_tensor_info(x)
    self.assertEqual(x.values.name,
                     x_tensor_info.coo_sparse.values_tensor_name)
    self.assertEqual(x.indices.name,
                     x_tensor_info.coo_sparse.indices_tensor_name)
    self.assertEqual(x.dense_shape.name,
                     x_tensor_info.coo_sparse.dense_shape_tensor_name)
    self.assertEqual(types_pb2.DT_FLOAT, x_tensor_info.dtype)
    self.assertEqual(2, len(x_tensor_info.tensor_shape.dim))
    self.assertEqual(42, x_tensor_info.tensor_shape.dim[0].size)
    self.assertEqual(69, x_tensor_info.tensor_shape.dim[1].size)

  @test_util.run_v1_only(
      "b/120545219: `build_tensor_info` is only available in graph mode.")
  def testBuildTensorInfoRagged(self):
    # [FUNCTION DOC]
    # Builds tensor info for a ragged constant; asserts the two composite components are the
    # values (int32) and row splits (int64) tensors by name and dtype, and that the stored type
    # specification decodes to the tensor's own type specification.
    x = ragged_factory_ops.constant([[1, 2], [3]])
    x_tensor_info = utils.build_tensor_info(x)
    # Check components
    self.assertEqual(x.values.name,
                     x_tensor_info.composite_tensor.components[0].name)
    self.assertEqual(types_pb2.DT_INT32,
                     x_tensor_info.composite_tensor.components[0].dtype)
    self.assertEqual(x.row_splits.name,
                     x_tensor_info.composite_tensor.components[1].name)
    self.assertEqual(types_pb2.DT_INT64,
                     x_tensor_info.composite_tensor.components[1].dtype)
    # Check type_spec.
    spec_proto = struct_pb2.StructuredValue(
        type_spec_value=x_tensor_info.composite_tensor.type_spec)
    spec = nested_structure_coder.decode_proto(spec_proto)
    self.assertEqual(spec, x._type_spec)

  def testBuildTensorInfoEager(self):
    # [FUNCTION DOC]
    # Under eager mode, asserts that building tensor info raises a runtime error whose message
    # says the function is not supported.
    x = constant_op.constant(1, name="x")
    with context.eager_mode(), self.assertRaisesRegex(
        RuntimeError, "`build_tensor_info` is not supported"):
      utils.build_tensor_info(x)

  @test_util.run_v1_only(
      "b/120545219: `build_tensor_info` is only available in graph mode.")
  def testGetTensorFromInfoDense(self):
    # [FUNCTION DOC]
    # Round trip for a dense placeholder: builds tensor info then resolves it; asserts a tensor
    # with the same name is returned.
    expected = array_ops.placeholder(dtypes.float32, 1, name="x")
    tensor_info = utils.build_tensor_info(expected)
    actual = utils.get_tensor_from_tensor_info(tensor_info)
    self.assertIsInstance(actual, tensor.Tensor)
    self.assertEqual(expected.name, actual.name)

  @test_util.run_v1_only(
      "b/120545219: `build_tensor_info` is only available in graph mode.")
  def testGetTensorFromInfoSparse(self):
    # [FUNCTION DOC]
    # Round trip for a sparse placeholder; asserts a sparse tensor is returned whose values,
    # indices and dense shape tensor names match the original.
    expected = array_ops.sparse_placeholder(dtypes.float32, name="x")
    tensor_info = utils.build_tensor_info(expected)
    actual = utils.get_tensor_from_tensor_info(tensor_info)
    self.assertIsInstance(actual, sparse_tensor.SparseTensor)
    self.assertEqual(expected.values.name, actual.values.name)
    self.assertEqual(expected.indices.name, actual.indices.name)
    self.assertEqual(expected.dense_shape.name, actual.dense_shape.name)

  @test_util.run_v1_only(
      "b/120545219: `build_tensor_info` is only available in graph mode.")
  def testGetTensorFromInfoRagged(self):
    # [FUNCTION DOC]
    # Round trip for a ragged constant; asserts a ragged tensor is returned whose values and row
    # splits names match the original.
    expected = ragged_factory_ops.constant([[1, 2], [3]], name="x")
    tensor_info = utils.build_tensor_info(expected)
    actual = utils.get_tensor_from_tensor_info(tensor_info)
    self.assertIsInstance(actual, ragged_tensor.RaggedTensor)
    self.assertEqual(expected.values.name, actual.values.name)
    self.assertEqual(expected.row_splits.name, actual.row_splits.name)

  def testGetTensorFromInfoInOtherGraph(self):
    # [FUNCTION DOC]
    # Builds tensor info in one graph while a second graph holds a differently named
    # placeholder. Resolving with the first graph passed explicitly must return a tensor that
    # belongs to that graph and has the original name.
    with ops.Graph().as_default() as expected_graph:
      expected = array_ops.placeholder(dtypes.float32, 1, name="right")
      tensor_info = utils.build_tensor_info(expected)
    with ops.Graph().as_default():  # Some other graph.
      array_ops.placeholder(dtypes.float32, 1, name="other")
    actual = utils.get_tensor_from_tensor_info(tensor_info,
                                               graph=expected_graph)
    self.assertIsInstance(actual, tensor.Tensor)
    self.assertIs(actual.graph, expected_graph)
    self.assertEqual(expected.name, actual.name)

  def testGetTensorFromInfoInScope(self):
    # [FUNCTION DOC]
    # Builds tensor info for a tensor named "bar/x:0", then in a new graph creates "foo/bar/x:0"
    # (as if imported into scope foo). Asserts that resolving with import scope "foo" finds the
    # prefixed tensor.
    # Build a TensorInfo with name "bar/x:0".
    with ops.Graph().as_default():
      with ops.name_scope("bar"):
        unscoped = array_ops.placeholder(dtypes.float32, 1, name="x")
        tensor_info = utils.build_tensor_info(unscoped)
        self.assertEqual("bar/x:0", tensor_info.name)
    # Build a graph with node "foo/bar/x:0", akin to importing into scope foo.
    with ops.Graph().as_default():
      with ops.name_scope("foo"):
        with ops.name_scope("bar"):
          expected = array_ops.placeholder(dtypes.float32, 1, name="x")
      self.assertEqual("foo/bar/x:0", expected.name)
      # Test that tensor is found by prepending the import scope.
      actual = utils.get_tensor_from_tensor_info(tensor_info,
                                                 import_scope="foo")
      self.assertEqual(expected.name, actual.name)

  @test_util.run_v1_only(
      "b/120545219: `build_tensor_info` is only available in graph mode.")
  def testGetTensorFromInfoRaisesErrors(self):
    # [FUNCTION DOC]
    # Asserts that resolving tensor info with a nonexistent tensor name raises a key error, and
    # that clearing the name field (leaving no encoding) raises a value error.
    expected = array_ops.placeholder(dtypes.float32, 1, name="x")
    tensor_info = utils.build_tensor_info(expected)
    tensor_info.name = "blah:0"  # Nonexistent name.
    with self.assertRaises(KeyError):
      utils.get_tensor_from_tensor_info(tensor_info)
    tensor_info.ClearField("name")  # Malformed (missing encoding).
    with self.assertRaises(ValueError):
      utils.get_tensor_from_tensor_info(tensor_info)

if __name__ == "__main__":
  test.main()
