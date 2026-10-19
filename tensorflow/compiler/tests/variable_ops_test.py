# [MODULE DOC]
# Tests for resource variable operations on an XLA test device: reading and writing variables,
# sparse reads with various index ranks, variable shape queries, invalid-shape errors, a gradient
# descent training step, aliased-write safety, the resource scatter family (add, sub, mul, div,
# min, max, update, nd-add, nd-update; tensor and scalar updates) and strided-slice assignment
# compared against numpy through a small checker class. Starts the standard test runner when
# run as a script.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for reading and writing variables."""
import re

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer


class VariableOpsTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # XLA test case for resource-variable operators. Each test builds variables or variable handles
  # inside the XLA test scope, runs updates in a session and checks values against expectations.
  # Most tests loop over all numeric dtypes supported by the device.
  """Test cases for resource variable operators."""

  def testWriteEmptyShape(self):
    # [FUNCTION DOC]
    # Scenario: for each numeric dtype, a variable of shape 3 by 0 is created, assigned from a
    # placeholder and read back with a control dependency on the assign. Expects the read value
    # to equal the empty zero array fed in.
    # Verifies that we can pass an uninitialized variable with an empty shape,
    # assign it a value, and successfully return it.
    for dtype in self.numeric_types:
      with self.session() as sess, self.test_scope():
        zeros = np.zeros([3, 0], dtype=dtype)
        v = resource_variable_ops.ResourceVariable(zeros)
        p = array_ops.placeholder(dtype)
        x = v.assign(p)
        with ops.control_dependencies([x]):
          y = v.read_value()
        self.assertAllClose(zeros, sess.run(y, {p: zeros}))

  def testOneWriteOneOutput(self):
    # [FUNCTION DOC]
    # Scenario (regression test): a computation with one variable update and one non-constant
    # output. For each dtype, a 2x2 variable is initialised, a placeholder of ones is added with
    # assign-add and the variable is read after the update. Expects the element-wise sum.
    # Regression test for a bug where computations with one non-constant
    # output and one variable update were mishandled.
    for dtype in self.numeric_types:
      init = np.array([[1, 2j], [3, 4]]).astype(dtype)
      with self.session() as sess, self.test_scope():
        v = resource_variable_ops.ResourceVariable(init)
        sess.run(variables.variables_initializer([v]))
        p = array_ops.placeholder(dtype)
        x = v.assign_add(p)
        with ops.control_dependencies([x]):
          y = v.read_value()
        self.assertAllClose(
            np.array([[2, 1 + 2j], [4, 5]]).astype(dtype),
            sess.run(y, {p: [[1, 1], [1, 1]]}))

  def testSparseRead0DIndices(self):
    # [FUNCTION DOC]
    # Scenario: sparse read of a 3x4 variable with a scalar index (2); expects the third row.
    for dtype in self.numeric_types:
      init = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8j, 9, 10,
                                                    11]]).astype(dtype)
      with self.session() as sess, self.test_scope():
        v = resource_variable_ops.ResourceVariable(init)
        sess.run(variables.variables_initializer([v]))
        x = v.sparse_read(2)
        self.assertAllClose(
            np.array([8j, 9, 10, 11]).astype(dtype), self.evaluate(x))

  def testSparseRead1DIndices(self):
    # [FUNCTION DOC]
    # Scenario: sparse read of a 3x4 variable with the index list 2,1; expects rows 2 and 1 in order.
    for dtype in self.numeric_types:
      init = np.array([[0, 1, 2, 3], [4, 5, 6j, 7], [8, 9, 10,
                                                     11]]).astype(dtype)
      with self.session() as sess, self.test_scope():
        v = resource_variable_ops.ResourceVariable(init)
        sess.run(variables.variables_initializer([v]))
        x = v.sparse_read([2, 1])
        self.assertAllClose(
            np.array([[8, 9, 10, 11], [4, 5, 6j, 7]]).astype(dtype),
            self.evaluate(x))

  def testSparseRead2DIndices(self):
    # [FUNCTION DOC]
    # Scenario: sparse read of a 3x4 variable with a 2x2 index matrix; expects a 2x2x4 result
    # gathering the indexed rows.
    for dtype in self.numeric_types:
      init = np.array([[0, 1, 2j, 3], [4, 5, 6, 7], [8, 9, 10,
                                                     11]]).astype(dtype)
      with self.session() as sess, self.test_scope():
        v = resource_variable_ops.ResourceVariable(init)
        sess.run(variables.variables_initializer([v]))
        x = v.sparse_read([[2, 1], [0, 2]])
        self.assertAllClose(
            np.array([[[8, 9, 10, 11], [4, 5, 6, 7]],
                      [[0, 1, 2j, 3], [8, 9, 10, 11]]]).astype(dtype),
            self.evaluate(x))

  def testSparseRead2DIndices3DTensor(self):
    # [FUNCTION DOC]
    # Scenario: sparse read of a 4x2x3 variable with a 2x2 index matrix; expects a 2x2x2x3 result
    # gathering the indexed slices.
    for dtype in self.numeric_types:
      init = np.array([[[0, 1, 2], [3, 4, 5]], [[10, 11, 12], [13, 14, 15]],
                       [[20, 21, 22], [23, 24j, 25]],
                       [[30, 31, 32], [33, 34, 35]]]).astype(dtype)
      with self.session() as sess, self.test_scope():
        v = resource_variable_ops.ResourceVariable(init)
        sess.run(variables.variables_initializer([v]))
        x = v.sparse_read([[2, 1], [3, 0]])
        self.assertAllClose(
            np.array(
                [[[[20, 21, 22], [23, 24j, 25]], [[10, 11, 12], [13, 14, 15]]],
                 [[[30, 31, 32], [33, 34, 35]], [[0, 1, 2], [3, 4, 5]]]
                ],).astype(dtype), self.evaluate(x))

  def testShape(self):
    # [FUNCTION DOC]
    # Scenario: queries the shape of a 2x3 variable through its handle with the default and with
    # a 64-bit integer output type. Expects 32-bit and 64-bit integer results, both equal to 2,3.
    for dtype in self.numeric_types:
      init = np.ones([2, 3]).astype(dtype)
      with self.session() as session, self.test_scope():
        v = resource_variable_ops.ResourceVariable(init)
        session.run(variables.variables_initializer([v]))
        h = v.handle
        s32, s64 = session.run([
            resource_variable_ops.variable_shape(h),
            resource_variable_ops.variable_shape(h, out_type=dtypes.int64)
        ])
        self.assertEqual(s32.dtype, np.int32)
        self.assertEqual(s64.dtype, np.int64)
        self.assertAllEqual(s32, [2, 3])
        self.assertAllEqual(s64, [2, 3])

  def testInvalidShape(self):
    # [FUNCTION DOC]
    # Scenario: assign-add and assign-sub of a scalar onto a 4-element vector variable inside the
    # XLA scope. Each is expected to raise an error whose message matches the phrase "shapes must
    # be equal" (case-insensitive).
    pattern = re.compile("shapes must be equal", re.IGNORECASE)
    # test invalid shape on assign_add in XLA
    with self.assertRaisesRegex(Exception, pattern):
      with self.session() as sess, self.test_scope():
        v = resource_variable_ops.ResourceVariable([0, 1, 2, 3])
        sess.run(variables.variables_initializer([v]))
        x = v.assign_add(1)
        sess.run(x)

    # test invalid shape raised on assign_sub in XLA
    with self.assertRaisesRegex(Exception, pattern):
      with self.session() as sess, self.test_scope():
        v = resource_variable_ops.ResourceVariable([0, 1, 2, 3])
        sess.run(variables.variables_initializer([v]))
        x = v.assign_sub(1)
        sess.run(x)

  def testReadWrite(self):
    # [FUNCTION DOC]
    # Scenario: a scalar resource variable initialised to 2 is read, assigned 47, read again,
    # increased by a complex constant, decreased by 3 and read again, chained with control
    # dependencies. Expects reads of 2, 47 and the final value 50 plus 2i cast to the dtype.
    """Tests initialization, reading, and writing a resource variable."""
    for dtype in self.numeric_types:
      with self.session() as session:
        with self.test_scope():
          with variable_scope.variable_scope("ascope", use_resource=True):
            x = variable_scope.get_variable(
                "x",
                shape=[],
                dtype=dtype,
                initializer=init_ops.constant_initializer(2))
            a = x.read_value()
            with ops.control_dependencies([a]):
              b = state_ops.assign(x, dtype(47))
            with ops.control_dependencies([b]):
              c = x.read_value()
            with ops.control_dependencies([c]):
              d = state_ops.assign_add(x, np.array(6 + 2j).astype(dtype))
            with ops.control_dependencies([d]):
              e = state_ops.assign_sub(x, dtype(3))
            with ops.control_dependencies([e]):
              f = x.read_value()

        session.run(variables.global_variables_initializer())
        v1, v2, v3 = session.run([a, c, f])
        self.assertAllClose(dtype(2), v1)
        self.assertAllClose(dtype(47), v2)
        self.assertAllClose(np.array(50 + 2j).astype(dtype), v3)

  def testTraining(self):
    # [FUNCTION DOC]
    # Scenario: one gradient-descent step (learning rate 0.1) for a linear model with a 4x2 weight
    # and 2-element bias, loss equal to the sum of outputs, input row 7,3,5,9. Expects weights
    # and bias reduced by 0.1 times the gradient, within relative tolerance 1e-4.
    """Tests a gradient descent step for a simple model."""
    with self.session() as session:
      with self.test_scope():
        with variable_scope.variable_scope("ascope", use_resource=True):
          w = variable_scope.get_variable(
              "w",
              shape=[4, 2],
              dtype=dtypes.float32,
              initializer=init_ops.constant_initializer(
                  np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)))
          b = variable_scope.get_variable(
              "b",
              shape=[2],
              dtype=dtypes.float32,
              initializer=init_ops.constant_initializer(
                  np.array([2, 3], dtype=np.float32)))

          x = array_ops.placeholder(dtypes.float32, shape=[1, 4])
          y = math_ops.matmul(x, w) + b
          loss = math_ops.reduce_sum(y)
          optimizer = GradientDescentOptimizer(0.1)
          train = optimizer.minimize(loss)

      session.run(variables.global_variables_initializer())
      session.run(train, {x: np.array([[7, 3, 5, 9]], dtype=np.float32)})
      vw, vb = session.run([w, b])
      self.assertAllClose(
          np.array(
              [[0.3, 1.3], [2.7, 3.7], [4.5, 5.5], [6.1, 7.1]],
              dtype=np.float32),
          vw,
          rtol=1e-4)
      self.assertAllClose(np.array([1.9, 2.9], dtype=np.float32), vb, rtol=1e-4)

  def testWriteOfAliasedTensor(self):
    # [FUNCTION DOC]
    # Scenario: the variable is read (keeping a reference to its old value) and then assigned the
    # value of a placeholder via an identity op, so the update cannot reuse the old buffer.
    # Expects the earlier read to still show the initial value and the assign result and the
    # identity output to show the new value.
    for dtype in self.numeric_types:
      init = np.array([[1, 2j], [3, 4]]).astype(dtype)
      update = np.array([[7, 1j], [2, 11]]).astype(dtype)
      with self.session() as sess, self.test_scope():
        v = resource_variable_ops.ResourceVariable(init)
        sess.run(variables.variables_initializer([v]))
        p = array_ops.placeholder(dtype)
        q = array_ops.identity(p)
        x = v.read_value()
        # Writes the value of 'p' to 'v', but keeps a reference to the original
        # value of 'v' so the variable update cannot reuse its buffer.
        with ops.control_dependencies([x]):
          y = v.assign(q)
        result = sess.run([x, y, q], {p: update})
        self.assertAllClose(init, result[0])
        self.assertAllClose(update, result[1])
        self.assertAllClose(update, result[2])

  def testScatterAdd(self):
    # [FUNCTION DOC]
    # Scenario: integer variable of shape 2x1 holding 1,7; scatter-add 2 at row 0. Expects 3,7.
    with self.session() as sess, self.test_scope():
      handle = resource_variable_ops.var_handle_op(
          dtype=dtypes.int32, shape=[2, 1])
      sess.run(
          resource_variable_ops.assign_variable_op(
              handle, constant_op.constant([[1], [7]], dtype=dtypes.int32)))
      sess.run(
          resource_variable_ops.resource_scatter_add(
              handle, [0], constant_op.constant([[2]], dtype=dtypes.int32)))
      read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
      self.assertAllEqual(self.evaluate(read), [[3], [7]])

  @test_util.disable_mlir_bridge("TODO: MLIR bridge does not yet"
                                 "support ResourceScatterSub")
  def testScatterSub(self):
    # [FUNCTION DOC]
    # Scenario: variable 4,1; scatter-subtract 2 at row 1. Expects 4,-1. Skipped with the MLIR bridge.
    with self.session() as sess, self.test_scope():
      handle = resource_variable_ops.var_handle_op(
          dtype=dtypes.int32, shape=[2, 1])
      sess.run(
          resource_variable_ops.assign_variable_op(
              handle, constant_op.constant([[4], [1]], dtype=dtypes.int32)))
      sess.run(
          resource_variable_ops.resource_scatter_sub(
              handle, [1], constant_op.constant([[2]], dtype=dtypes.int32)))
      read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
      self.assertAllEqual(self.evaluate(read), [[4], [-1]])

  @test_util.disable_mlir_bridge("TODO: MLIR bridge does not yet"
                                 "support ResourceScatterMul")
  def testScatterMul(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: asserts with plain equality on array values rather than the array-aware
    # equality helper; works only because the 1x1 result is truth-testable.
    # Scenario: 1x1 variable 1; scatter-multiply by 5 at row 0. Expects 5. Skipped with MLIR bridge.
    with self.session() as sess, self.test_scope():
      handle = resource_variable_ops.var_handle_op(
          dtype=dtypes.int32, shape=[1, 1])
      sess.run(
          resource_variable_ops.assign_variable_op(
              handle, constant_op.constant([[1]], dtype=dtypes.int32)))
      sess.run(
          resource_variable_ops.resource_scatter_mul(
              handle, [0], constant_op.constant([[5]], dtype=dtypes.int32)))
      read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
      self.assertEqual(self.evaluate(read), [[5]])

  @test_util.disable_mlir_bridge("TODO: MLIR bridge does not yet"
                                 "support ResourceScatterDiv")
  def testScatterDiv(self):
    # [FUNCTION DOC]
    # Scenario: 1x1 variable 6; scatter-divide by 3 at row 0. Expects 2. Skipped with MLIR bridge.
    with self.session() as sess, self.test_scope():
      handle = resource_variable_ops.var_handle_op(
          dtype=dtypes.int32, shape=[1, 1])
      sess.run(
          resource_variable_ops.assign_variable_op(
              handle, constant_op.constant([[6]], dtype=dtypes.int32)))
      sess.run(
          resource_variable_ops.resource_scatter_div(
              handle, [0], constant_op.constant([[3]], dtype=dtypes.int32)))
      read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
      self.assertAllEqual(self.evaluate(read), [[2]])

  @test_util.disable_mlir_bridge("TODO: MLIR bridge does not yet"
                                 "support ResourceScatterMin")
  def testScatterMin(self):
    # [FUNCTION DOC]
    # Scenario: 1x1 variable 6; scatter-minimum with 3 at row 0. Expects 3. Skipped with MLIR bridge.
    with self.session() as sess, self.test_scope():
      handle = resource_variable_ops.var_handle_op(
          dtype=dtypes.int32, shape=[1, 1])
      sess.run(
          resource_variable_ops.assign_variable_op(
              handle, constant_op.constant([[6]], dtype=dtypes.int32)))
      sess.run(
          resource_variable_ops.resource_scatter_min(
              handle, [0], constant_op.constant([[3]], dtype=dtypes.int32)))
      read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
      self.assertEqual(self.evaluate(read), [[3]])

  @test_util.disable_mlir_bridge("TODO: MLIR bridge does not yet"
                                 "support ResourceScatterMax")
  def testScatterMax(self):
    # [FUNCTION DOC]
    # Scenario: 1x1 variable 6; scatter-maximum with 3 at row 0. Expects 6. Skipped with MLIR bridge.
    with self.session() as sess, self.test_scope():
      handle = resource_variable_ops.var_handle_op(
          dtype=dtypes.int32, shape=[1, 1])
      sess.run(
          resource_variable_ops.assign_variable_op(
              handle, constant_op.constant([[6]], dtype=dtypes.int32)))
      sess.run(
          resource_variable_ops.resource_scatter_max(
              handle, [0], constant_op.constant([[3]], dtype=dtypes.int32)))
      read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
      self.assertEqual(self.evaluate(read), [[6]])

  def testScatterUpdate(self):
    # [FUNCTION DOC]
    # Scenario: 1x1 variable 6; scatter-update row 0 with 3. Expects 3.
    with self.session() as sess, self.test_scope():
      handle = resource_variable_ops.var_handle_op(
          dtype=dtypes.int32, shape=[1, 1])
      sess.run(
          resource_variable_ops.assign_variable_op(
              handle, constant_op.constant([[6]], dtype=dtypes.int32)))
      sess.run(
          resource_variable_ops.resource_scatter_update(
              handle, [0], constant_op.constant([[3]], dtype=dtypes.int32)))
      read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
      self.assertEqual(self.evaluate(read), [[3]])

  def testScatterScalarUpdate(self):
    # [FUNCTION DOC]
    # Scenario: 1x1 variable 6; scatter-update row 0 using a scalar (rank-0) update 3. Expects 3.
    with self.session() as sess, self.test_scope():
      handle = resource_variable_ops.var_handle_op(
          dtype=dtypes.int32, shape=[1, 1])
      sess.run(
          resource_variable_ops.assign_variable_op(
              handle, constant_op.constant([[6]], dtype=dtypes.int32)))
      sess.run(
          resource_variable_ops.resource_scatter_update(
              handle, [0], constant_op.constant(3, dtype=dtypes.int32)))
      read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
      self.assertEqual(self.evaluate(read), [[3]])

  def testScatterAddScalarUpdate(self):
    # [FUNCTION DOC]
    # Scenario: 1x1 variable 1; scatter-add of scalar 2 at row 0. Expects 3.
    with self.session() as sess, self.test_scope():
      handle = resource_variable_ops.var_handle_op(
          dtype=dtypes.int32, shape=[1, 1])
      sess.run(
          resource_variable_ops.assign_variable_op(
              handle, constant_op.constant([[1]], dtype=dtypes.int32)))
      sess.run(
          resource_variable_ops.resource_scatter_add(
              handle, [0], constant_op.constant(2, dtype=dtypes.int32)))
      read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
      self.assertEqual(self.evaluate(read), [[3]])

  @test_util.disable_mlir_bridge("TODO: MLIR bridge does not yet"
                                 "support ResourceScatterSub")
  def testScatterSubScalar(self):
    # [FUNCTION DOC]
    # Scenario: 1x1 variable 1; scatter-subtract scalar 2 at row 0. Expects -1. Skipped with MLIR bridge.
    with self.session() as sess, self.test_scope():
      handle = resource_variable_ops.var_handle_op(
          dtype=dtypes.int32, shape=[1, 1])
      sess.run(
          resource_variable_ops.assign_variable_op(
              handle, constant_op.constant([[1]], dtype=dtypes.int32)))
      sess.run(
          resource_variable_ops.resource_scatter_sub(
              handle, [0], constant_op.constant(2, dtype=dtypes.int32)))
      read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
      self.assertEqual(self.evaluate(read), [[-1]])

  @test_util.disable_mlir_bridge("TODO: MLIR bridge does not yet"
                                 "support ResourceScatterMul")
  def testScatterMulScalar(self):
    # [FUNCTION DOC]
    # Scenario: 1x1 variable 1; scatter-multiply by scalar 5 at row 0. Expects 5. Skipped with MLIR bridge.
    with self.session() as sess, self.test_scope():
      handle = resource_variable_ops.var_handle_op(
          dtype=dtypes.int32, shape=[1, 1])
      sess.run(
          resource_variable_ops.assign_variable_op(
              handle, constant_op.constant([[1]], dtype=dtypes.int32)))
      sess.run(
          resource_variable_ops.resource_scatter_mul(
              handle, [0], constant_op.constant(5, dtype=dtypes.int32)))
      read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
      self.assertEqual(self.evaluate(read), [[5]])

  @test_util.disable_mlir_bridge("TODO: MLIR bridge does not yet"
                                 "support ResourceScatterDiv")
  def testScatterDivScalar(self):
    # [FUNCTION DOC]
    # Scenario: 1x1 variable 6; scatter-divide by scalar 3 at row 0. Expects 2. Skipped with MLIR bridge.
    with self.session() as sess, self.test_scope():
      handle = resource_variable_ops.var_handle_op(
          dtype=dtypes.int32, shape=[1, 1])
      sess.run(
          resource_variable_ops.assign_variable_op(
              handle, constant_op.constant([[6]], dtype=dtypes.int32)))
      sess.run(
          resource_variable_ops.resource_scatter_div(
              handle, [0], constant_op.constant(3, dtype=dtypes.int32)))
      read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
      self.assertEqual(self.evaluate(read), [[2]])

  @test_util.disable_mlir_bridge("TODO: MLIR bridge does not yet"
                                 "support ResourceScatterMin")
  def testScatterMinScalar(self):
    # [FUNCTION DOC]
    # Scenario: 1x1 variable 6; scatter-minimum with scalar 3 at row 0. Expects 3. Skipped with MLIR bridge.
    with self.session() as sess, self.test_scope():
      handle = resource_variable_ops.var_handle_op(
          dtype=dtypes.int32, shape=[1, 1])
      sess.run(
          resource_variable_ops.assign_variable_op(
              handle, constant_op.constant([[6]], dtype=dtypes.int32)))
      sess.run(
          resource_variable_ops.resource_scatter_min(
              handle, [0], constant_op.constant(3, dtype=dtypes.int32)))
      read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
      self.assertEqual(self.evaluate(read), [[3]])

  @test_util.disable_mlir_bridge("TODO: MLIR bridge does not yet"
                                 "support ResourceScatterMax")
  def testScatterMaxScalar(self):
    # [FUNCTION DOC]
    # Scenario: 1x1 variable 6; scatter-maximum with scalar 3 at row 0. Expects 6. Skipped with MLIR bridge.
    with self.session() as sess, self.test_scope():
      handle = resource_variable_ops.var_handle_op(
          dtype=dtypes.int32, shape=[1, 1])
      sess.run(
          resource_variable_ops.assign_variable_op(
              handle, constant_op.constant([[6]], dtype=dtypes.int32)))
      sess.run(
          resource_variable_ops.resource_scatter_max(
              handle, [0], constant_op.constant(3, dtype=dtypes.int32)))
      read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
      self.assertEqual(self.evaluate(read), [[6]])

  @test_util.disable_mlir_bridge("TODO: MLIR bridge does not yet"
                                 "support ResourceScatterNdAdd")
  def testScatterNdAddOps(self):
    # [FUNCTION DOC]
    # Scenario: float vector variable of eight ones; nd-scatter-add of 9,10,11,12 at positions
    # 4,3,1,7. Expects ones with those positions increased. Skipped with the MLIR bridge.
    with self.session() as sess, self.test_scope():
      handle = resource_variable_ops.var_handle_op(
          dtype=dtypes.float32, shape=[8])
      sess.run(
          resource_variable_ops.assign_variable_op(
              handle, constant_op.constant([1] * 8, dtype=dtypes.float32)))
      indices = constant_op.constant([[4], [3], [1], [7]], dtype=dtypes.int32)
      updates = constant_op.constant([9, 10, 11, 12], dtype=dtypes.float32)
      expected = np.array([1, 12, 1, 11, 10, 1, 1, 13])
      sess.run(gen_state_ops.resource_scatter_nd_add(handle, indices, updates))
      read = resource_variable_ops.read_variable_op(
          handle, dtype=dtypes.float32)
      self.assertAllClose(expected, self.evaluate(read))

  @test_util.disable_mlir_bridge("TODO: MLIR bridge does not yet"
                                 "support ResourceScatterNdUpdateAdd")
  def testScatterNdUpdateAddOps(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: name and skip message mention add, but the test applies nd-scatter-update
    # (overwrite), not add.
    # Scenario: float vector variable of eight ones; nd-scatter-update with 9,10,11,12 at
    # positions 4,3,1,7. Expects those positions overwritten. Skipped with the MLIR bridge.
    with self.session() as sess, self.test_scope():
      handle = resource_variable_ops.var_handle_op(
          dtype=dtypes.float32, shape=[8])
      sess.run(
          resource_variable_ops.assign_variable_op(
              handle, constant_op.constant([1] * 8, dtype=dtypes.float32)))
      indices = constant_op.constant([[4], [3], [1], [7]], dtype=dtypes.int32)
      updates = constant_op.constant([9, 10, 11, 12], dtype=dtypes.float32)
      expected = np.array([1, 11, 1, 10, 9, 1, 1, 12])
      sess.run(
          gen_state_ops.resource_scatter_nd_update(handle, indices, updates))
      read = resource_variable_ops.read_variable_op(
          handle, dtype=dtypes.float32)
      self.assertAllClose(expected, self.evaluate(read))


class StridedSliceAssignChecker(object):
  # [CLASS DOC]
  # HELPER: test utility supporting index-assignment syntax. Holds the owning test case, the
  # numpy dtype, a numpy copy of the initial data and a mode flag that alternates between two
  # assignment styles. Each indexed assignment builds a fresh variable, performs the slice
  # assignment in TensorFlow and compares to the same assignment done in numpy.
  """Compares the results of a slice assignment using Tensorflow and numpy."""

  def __init__(self, test, x, dtype):
    # [FUNCTION DOC]
    # Constructor. Inputs: test, the test case providing session, scope and assertions; x, the
    # initial array-like data; dtype, numpy dtype to cast to. Stores them, makes the numpy copy
    # and picks a random starting mode (zero or one) using the global numpy random state.
    self.dtype = dtype
    self.test = test
    self.x_np = np.array(x).astype(dtype)
    # Randomly start on mode 0 or 1.
    self.which_mode = np.random.randint(2, size=1)[0]

  def __setitem__(self, index, value):
    # [FUNCTION DOC]
    # Index-assignment hook. Inputs: index, any slice/index expression; value, array-like data
    # cast to the dtype. Flips the mode, creates and initialises a variable from the stored data
    # in the XLA scope, then assigns value to the indexed part either through the variable-slice
    # assign method (mode zero) or the generic assign function (mode one). Computes the same
    # assignment on a numpy copy and asserts the TensorFlow result equals it. Returns nothing.
    self.which_mode = 1 - self.which_mode
    value = np.array(value).astype(self.dtype)

    with self.test.session() as sess, self.test.test_scope():
      x = constant_op.constant(self.x_np, dtype=self.dtype)
      var = resource_variable_ops.ResourceVariable(x)
      sess.run(variables.variables_initializer([var]))

      if self.which_mode == 0:
        val = sess.run(var[index].assign(value))
      else:
        assert self.which_mode == 1
        val = sess.run(state_ops.assign(var[index], value))
      valnp = np.copy(self.x_np)
      valnp[index] = np.array(value)
      self.test.assertAllEqual(val, valnp)


class SliceAssignTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # XLA test case for strided-slice assignment to resource variables, using the checker class
  # for numpy comparison, plus an uninitialised-variable error test.

  @test_util.disable_mlir_bridge("TODO: MLIR bridge does not yet"
                                 "support ResourceStridedSliceAssign")
  def testSliceAssign(self):
    # [FUNCTION DOC]
    # Scenario: for each numeric dtype, performs many slice assignments on a 2x3 matrix via the
    # checker: whole-array, 1x1 block, dimension-shrinking, new-axis, non-unit and negative
    # strides, and degenerate empty intervals; then on a scalar variable with empty index,
    # ellipsis and new axis. Scalar-number assignments are skipped for bfloat16 (known numpy
    # issue). Each assignment is verified inside the checker. Skipped with the MLIR bridge.
    for dtype in self.numeric_types:
      checker = StridedSliceAssignChecker(
          self, [[1, 2, 3], [4, 5, 6]], dtype=dtype)
      # No-op assignment
      checker[:] = [[10, 20, 30], [40, 50, 60]]
      # Checks trivial (1,1) shape tensor
      checker[1:2, 1:2] = [[66]]
      # shrink shape changes
      checker[1:2, 1] = [66]
      checker[1, 1:2] = [66]
      if dtype != dtypes.bfloat16.as_numpy_dtype:
        # TODO(b/68813416): valnp call above results in an ndarray and not a
        # number for bfloat16s.
        checker[1, 1] = 66
      # newaxis shape changes
      checker[:, None, :] = [[[10, 20, 30]], [[40, 50, 50]]]
      # shrink and newaxis
      checker[None, None, 0, 0:1] = [[[99]]]
      # Non unit strides
      checker[::1, 1::-1] = [[3, 33], [4, 44]]
      # degenerate interval
      checker[8:10, 0] = []
      checker[8:10, 8:10] = [[]]

      # Assign vector to scalar (rank-0) using newaxis
      checker2 = StridedSliceAssignChecker(self, 222, dtype=dtype)
      if dtype != dtypes.bfloat16.as_numpy_dtype:
        # TODO(b/68813416): valnp call above results in an ndarray and not a
        # number for bfloat16s.
        checker2[()] = 6  # no indices
        checker2[...] = 6  # ellipsis
      checker2[None] = [6]  # new axis

  @test_util.disable_mlir_bridge("TODO: MLIR bridge does not yet"
                                 "support uninitialized resource variable")
  def testUninitialized(self):
    # [FUNCTION DOC]
    # Scenario: slice-assign into a variable that was never initialised. Expects a
    # failed-precondition error mentioning "uninitialized". Skipped with the MLIR bridge.
    with self.assertRaisesRegex(errors.FailedPreconditionError,
                                "uninitialized"):
      with self.session() as sess, self.test_scope():
        v = resource_variable_ops.ResourceVariable([1, 2])
        sess.run(v[:].assign([1, 2]))


if __name__ == "__main__":
  googletest.main()
