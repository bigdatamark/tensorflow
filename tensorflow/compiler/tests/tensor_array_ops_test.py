# [MODULE DOC]
# Functional tests for TensorArray ops (write, read, stack, concat, unstack, split, scatter,
# gather, gradients, close, size, identity and error cases) when compiled with XLA, using the
# XLA compile helper inside the XLA test device scope. Defines a dtype converter helper and one
# test class that runs under the graph-mode (version 1) test setting with control flow v2 variants;
# several tests are disabled for control flow v2 with bug references. A large block of while-loop
# tests is commented out pending XLA while-loop support. No import-time side effects.
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
"""Functional tests for XLA TensorArray Ops."""

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.compiler.xla import xla
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


def _make_converter(dtype):
  # [FUNCTION DOC]
  # HELPER: factory for dtype conversion functions used by the tests.
  # Input: dtype, a TensorFlow data type. Returns a function that converts any array-like value to
  # a numpy array of the numpy type corresponding to that dtype.
  def _converter(x):
    # [FUNCTION DOC]
    # HELPER: converts the given array-like value to a numpy array of the captured dtype.
    return np.asarray(x).astype(dtype.as_numpy_dtype)
  return _converter


# This lets me define `fn` repeatedly to pass to xla.compile.
#
# pylint: disable=function-redefined
@test_util.run_v1_only("b/")  # Support TF2 list operations
@test_util.with_control_flow_v2
class TensorArrayTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # Test case for TensorArray operations compiled with XLA on the XLA test device. Decorated to run
  # only in graph mode (version 1) and under both control flow variants. Each test builds its graph
  # inside a function passed to the XLA compile helper, evaluates it, and checks values, shapes,
  # gradients or raised errors. Helper methods prefixed with an underscore hold shared bodies for
  # tests that loop over data types. Holds no instance state.

  @test_util.disable_control_flow_v2("Tries to evaluate flow")
  def testTensorArrayWriteRead(self):
    # [FUNCTION DOC]
    # Test: writes three 1 by 2 float matrices into a size-3 array, reads them back by index, and
    # also returns the flow scalar. Asserts the read values are as written and flow is a scalar.
    with self.session() as session, self.test_scope():

      def fn():
        # [FUNCTION DOC]
        # Graph body compiled with XLA: write three values, read all three, return the reads and
        # the final flow.
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32, tensor_array_name="foo", size=3)

        w0 = ta.write(0, [[4.0, 5.0]])
        w1 = w0.write(1, [[1.0, 3.0]])
        w2 = w1.write(2, [[7.0, -8.5]])

        r0 = w2.read(0)
        r1 = w2.read(1)
        r2 = w2.read(2)
        flow = w2.flow
        return [r0, r1, r2, flow]

      d0, d1, d2, flow_val = self.evaluate(xla.compile(fn))
      self.assertAllEqual([[4.0, 5.0]], d0)
      self.assertAllEqual([[1.0, 3.0]], d1)
      self.assertAllEqual([[7.0, -8.5]], d2)
      self.assertAllEqual([], flow_val.shape)

  def _testTensorArrayWritePack(self, tf_dtype):
    # [FUNCTION DOC]
    # HELPER: body of the write-and-stack test for one data type (tf_dtype). Writes three 1 by 2
    # matrices, stacks them and asserts the result is the 3 by 1 by 2 tensor of the written values.
    with self.session(), self.test_scope():
      convert = _make_converter(tf_dtype)

      def fn():
        # [FUNCTION DOC]
        # Graph body: write three values and return the stacked array.
        ta = tensor_array_ops.TensorArray(
            dtype=tf_dtype, tensor_array_name="foo", size=3)

        w0 = ta.write(0, convert([[4.0, 5.0]]))
        w1 = w0.write(1, convert([[6.0, 7.0]]))
        w2 = w1.write(2, convert([[8.0, 9.0]]))

        return w2.stack()

      self.assertAllEqual(
          convert([[[4.0, 5.0]], [[6.0, 7.0]], [[8.0, 9.0]]]),
          self.evaluate(xla.compile(fn)[0]))

  def testTensorArrayWritePack(self):
    # [FUNCTION DOC]
    # Test: runs the write-and-stack check for every numeric type supported by the XLA device.
    for dtype in self.numeric_tf_types:
      self._testTensorArrayWritePack(dtype)

  def testEmptyTensorArrayPack(self):
    # [FUNCTION DOC]
    # Test: writes three empty (zero by one) float arrays and stacks them; asserts the stacked
    # result has shape 3, 0, 1.
    with self.session(), self.test_scope():

      def fn():
        # [FUNCTION DOC]
        # Graph body: write three empty elements and return the stacked array.
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32, tensor_array_name="foo", size=3)

        empty_element = np.zeros((0, 1), dtype=np.float32)
        w0 = ta.write(0, empty_element)
        w1 = w0.write(1, empty_element)
        w2 = w1.write(2, empty_element)

        return w2.stack()

      self.assertAllEqual([3, 0, 1], self.evaluate(xla.compile(fn)[0]).shape)

  def _testTensorArrayWriteConcat(self, tf_dtype):
    # [FUNCTION DOC]
    # HELPER: body of the write-and-concat test for one data type (tf_dtype). Writes three 2 by 2
    # matrices, concatenates them along the first dimension and asserts the 6 by 2 result.
    with self.session(), self.test_scope():
      convert = _make_converter(tf_dtype)

      def fn():
        # [FUNCTION DOC]
        # Graph body: write three values and return the concatenation of the array.
        ta = tensor_array_ops.TensorArray(
            dtype=tf_dtype, tensor_array_name="foo", size=3)

        w0 = ta.write(0, convert([[4.0, 5.0], [104.0, 105.0]]))
        w1 = w0.write(1, convert([[6.0, 7.0], [106.0, 107.0]]))
        w2 = w1.write(2, convert([[8.0, 9.0], [124.0, 125.0]]))

        return w2.concat()

      self.assertAllEqual(
          convert([[4.0, 5.0], [104.0, 105.0], [6.0, 7.0], [106.0, 107.0],
                   [8.0, 9.0], [124.0, 125.0]]),
          self.evaluate(xla.compile(fn)[0]))

  @test_util.disable_control_flow_v2("b/122315751 (concat)")
  def testTensorArrayWriteConcat(self):
    # [FUNCTION DOC]
    # Test (disabled for control flow v2): runs the write-and-concat check for every numeric type.
    for dtype in self.numeric_tf_types:
      self._testTensorArrayWriteConcat(dtype)

  def _testTensorArrayUnpackRead(self, tf_dtype):
    # [FUNCTION DOC]
    # HELPER: body of the unstack-and-read test for one data type (tf_dtype). Three scenarios, each
    # compiled and evaluated: unstack a vector into scalars, unstack a 3 by 2 matrix into vectors,
    # and unstack an empty 3 by 0 matrix (must not fail). Asserts each read returns the expected
    # element.
    with self.session() as session, self.test_scope():
      convert = _make_converter(tf_dtype)

      def fn():
        # [FUNCTION DOC]
        # Graph body: unstack a vector of three values and read each element.
        ta = tensor_array_ops.TensorArray(
            dtype=tf_dtype, tensor_array_name="foo", size=3)

        # Unpack a vector into scalars
        w0 = ta.unstack(convert([1.0, 2.0, 3.0]))
        r0 = w0.read(0)
        r1 = w0.read(1)
        r2 = w0.read(2)

        return [r0, r1, r2]

      d0, d1, d2 = self.evaluate(xla.compile(fn))
      self.assertAllEqual(convert(1.0), d0)
      self.assertAllEqual(convert(2.0), d1)
      self.assertAllEqual(convert(3.0), d2)

      def fn():
        # [FUNCTION DOC]
        # Graph body: unstack a 3 by 2 matrix into three vectors and read each.
        ta = tensor_array_ops.TensorArray(
            dtype=tf_dtype, tensor_array_name="foo", size=3)

        # Unpack a matrix into vectors.
        w1 = ta.unstack(
            convert([[1.0, 1.03125], [2.0, 2.03125], [3.0, 3.03125]]))
        r0 = w1.read(0)
        r1 = w1.read(1)
        r2 = w1.read(2)
        return [r0, r1, r2]

      d0, d1, d2 = self.evaluate(xla.compile(fn))

      self.assertAllEqual(convert([1.0, 1.03125]), d0)
      self.assertAllEqual(convert([2.0, 2.03125]), d1)
      self.assertAllEqual(convert([3.0, 3.03125]), d2)

      def fn():
        # [FUNCTION DOC]
        # Graph body: create a fresh array and unstack an empty 3 by 0 matrix, reading each element.
        # Reset ta because we're going to change the shape, else shape
        # inference will throw an error.
        ta = tensor_array_ops.TensorArray(
            dtype=tf_dtype, tensor_array_name="foo", size=3)

        # Try unpacking an empty matrix, which should not cause an error.
        w2 = ta.unstack(convert([[], [], []]))
        r0 = w2.read(0)
        r1 = w2.read(1)
        r2 = w2.read(2)
        return [r0, r1, r2]

      d0, d1, d2 = self.evaluate(xla.compile(fn))
      self.assertAllEqual(convert([]), d0)
      self.assertAllEqual(convert([]), d1)
      self.assertAllEqual(convert([]), d2)

  def _testTensorArrayUnpackReadMaybeLegacy(self):
    # [FUNCTION DOC]
    # HELPER: runs the unstack-and-read check for every numeric type supported by the device.
    for dtype in self.numeric_tf_types:
      self._testTensorArrayUnpackRead(dtype)

  def testTensorArrayUnpackRead(self):
    # [FUNCTION DOC]
    # Test: delegates to the helper that runs the unstack-and-read check for all numeric types.
    self._testTensorArrayUnpackReadMaybeLegacy()

  def _testTensorArraySplitRead(self, tf_dtype):
    # [FUNCTION DOC]
    # HELPER: body of the split-and-read test for one data type (tf_dtype). Three scenarios: split
    # an empty vector with zero lengths, split a three-element vector into unit pieces, and split a
    # 3 by 2 matrix into single-row pieces. Asserts each read returns the expected piece.
    with self.session() as session, self.test_scope():
      convert = _make_converter(tf_dtype)

      def fn():
        # [FUNCTION DOC]
        # Graph body: split an empty vector into three empty pieces and read each.
        ta = tensor_array_ops.TensorArray(
            dtype=tf_dtype, tensor_array_name="foo", size=3)

        # Split an empty vector.
        lengths = constant_op.constant([0, 0, 0])
        w0 = ta.split(convert([]), lengths=lengths)
        r0 = w0.read(0)
        r1 = w0.read(1)
        r2 = w0.read(2)
        return [r0, r1, r2]

      d0, d1, d2 = self.evaluate(xla.compile(fn))

      self.assertAllEqual(convert([]), d0)
      self.assertAllEqual(convert([]), d1)
      self.assertAllEqual(convert([]), d2)

      def fn():
        # [FUNCTION DOC]
        # Graph body: split a three-element vector into three pieces of length one and read each.
        # Split a vector.
        ta = tensor_array_ops.TensorArray(
            dtype=tf_dtype, tensor_array_name="foo", size=3)
        lengths = constant_op.constant([1, 1, 1])
        w0 = ta.split(convert([1.0, 2.0, 3.0]), lengths=lengths)
        r0 = w0.read(0)
        r1 = w0.read(1)
        r2 = w0.read(2)
        return [r0, r1, r2]

      d0, d1, d2 = self.evaluate(xla.compile(fn))

      self.assertAllEqual(convert([1.0]), d0)
      self.assertAllEqual(convert([2.0]), d1)
      self.assertAllEqual(convert([3.0]), d2)

      def fn():
        # [FUNCTION DOC]
        # Graph body: split a 3 by 2 matrix into three single-row pieces and read each.
        # Split a matrix.
        ta = tensor_array_ops.TensorArray(
            dtype=tf_dtype, tensor_array_name="foo", size=3)
        lengths = constant_op.constant([1, 1, 1])
        w0 = ta.split(
            convert([[1.0, 101.0], [2.0, 121.0], [3.0, 127.0]]),
            lengths=lengths)
        r0 = w0.read(0)
        r1 = w0.read(1)
        r2 = w0.read(2)
        return [r0, r1, r2]

      d0, d1, d2 = self.evaluate(xla.compile(fn))
      self.assertAllEqual(convert([[1.0, 101.0]]), d0)
      self.assertAllEqual(convert([[2.0, 121.0]]), d1)
      self.assertAllEqual(convert([[3.0, 127.0]]), d2)

  @test_util.disable_control_flow_v2("b/122315872 (split)")
  def testTensorArraySplitRead(self):
    # [FUNCTION DOC]
    # Test (disabled for control flow v2): runs the split-and-read check for all numeric types.
    for dtype in self.numeric_tf_types:
      self._testTensorArraySplitRead(dtype)

  @test_util.disable_control_flow_v2("TensorArray.grad is not supported in v2")
  def testTensorGradArrayWriteRead(self):
    # [FUNCTION DOC]
    # Test (disabled for control flow v2): writes three values to an array and three other values
    # to its gradient array (named grad), reads both back and asserts all six values are as written.
    with self.session() as session, self.test_scope():

      def fn():
        # [FUNCTION DOC]
        # Graph body: write to the array and its gradient array, read all and return six reads.
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32, tensor_array_name="foo", size=3)

        w0 = ta.write(0, [[4.0]])
        w1 = w0.write(1, [[1.0]])
        w2 = w1.write(2, [[-3.0]])

        g_ta = w2.grad("grad")

        g_w0 = g_ta.write(0, [[5.0]])
        g_w1 = g_w0.write(1, [[2.0]])
        g_w2 = g_w1.write(2, [[-2.0]])

        r0 = w2.read(0)
        r1 = w2.read(1)
        r2 = w2.read(2)

        g_r0 = g_w2.read(0)
        g_r1 = g_w2.read(1)
        g_r2 = g_w2.read(2)

        return [r0, r1, r2, g_r0, g_r1, g_r2]

      d0, d1, d2, g_d0, g_d1, g_d2 = self.evaluate(xla.compile(fn))
      self.assertAllEqual([[4.0]], d0)
      self.assertAllEqual([[1.0]], d1)
      self.assertAllEqual([[-3.0]], d2)
      self.assertAllEqual([[5.0]], g_d0)
      self.assertAllEqual([[2.0]], g_d1)
      self.assertAllEqual([[-2.0]], g_d2)

  @test_util.disable_control_flow_v2("TensorArray.grad is not supported in v2")
  def testTensorGradArrayDynamicWriteRead(self):
    # [FUNCTION DOC]
    # Test (disabled for control flow v2): like the gradient-array write/read test, but the gradient
    # array is created before its contents are written, and the sizes of both arrays are also
    # fetched. Asserts values as written and both sizes equal 3.
    with self.session() as session, self.test_scope():

      def fn():
        # [FUNCTION DOC]
        # Graph body: write to array and gradient array, read all, and return reads plus both sizes.
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32, tensor_array_name="foo", size=3)

        w0 = ta.write(0, [[4.0]])
        w1 = w0.write(1, [[1.0]])
        w2 = w1.write(2, [[-3.0]])

        g_ta = w2.grad("grad")  # Get gradient array here so we know the shape

        s = w2.size()
        g_s = g_ta.size()

        g_w0 = g_ta.write(0, [[5.0]])
        g_w1 = g_w0.write(1, [[2.0]])
        g_w2 = g_w1.write(2, [[-2.0]])

        r0 = w2.read(0)
        r1 = w2.read(1)
        r2 = w2.read(2)

        g_r0 = g_w2.read(0)
        g_r1 = g_w2.read(1)
        g_r2 = g_w2.read(2)

        return [r0, r1, r2, g_r0, g_r1, g_r2, s, g_s]

      d0, d1, d2, g_d0, g_d1, g_d2, vs, g_vs = self.evaluate(xla.compile(fn))
      self.assertAllEqual([[4.0]], d0)
      self.assertAllEqual([[1.0]], d1)
      self.assertAllEqual([[-3.0]], d2)
      self.assertAllEqual([[5.0]], g_d0)
      self.assertAllEqual([[2.0]], g_d1)
      self.assertAllEqual([[-2.0]], g_d2)
      self.assertAllEqual(3, vs)
      self.assertAllEqual(3, g_vs)

  @test_util.disable_control_flow_v2("TensorArray.grad is not supported in v2")
  def testTensorGradAccessTwiceReceiveSameObject(self):
    # [FUNCTION DOC]
    # Test (disabled for control flow v2): requesting the gradient array twice under one name gives
    # handles that refer to the same underlying array. Writes through one handle and reads through
    # the other, asserting the value written is read back. The direct handle comparison is left out
    # (explained in a comment) because handles cannot be fetched or returned from XLA clusters.
    with self.session() as session, self.test_scope():
      ta_out = {}

      def fn():
        # [FUNCTION DOC]
        # Graph body: get the gradient array twice, record the handles in the enclosing mapping,
        # write via the first, read via the second (ordered by control dependency) and return the read.
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32,
            tensor_array_name="foo",
            size=3,
            element_shape=[1, 2])

        g_ta_0 = ta.grad("grad")
        g_ta_1 = ta.grad("grad")

        ta_out[0] = g_ta_0.handle
        ta_out[1] = g_ta_1.handle

        with ops.control_dependencies([g_ta_0.write(0, [[4.0, 5.0]]).flow]):
          # Write with one gradient handle, read with another copy of it
          r1_0 = g_ta_1.read(0)

        with ops.control_dependencies([g_ta_0.handle.op, g_ta_1.handle.op]):
          return [r1_0]

      [d_r1_0] = self.evaluate(xla.compile(fn))
      self.assertAllEqual([[4.0, 5.0]], d_r1_0)

      # Can't assert this because adding a side output like we have here fails
      # as follows:
      #
      # ValueError: Operation u'TensorArrayGrad/TensorArrayGradV3' has been
      # marked as not fetchable.
      #
      # On the other hand, legitimately returning the handle from the
      # xla.compile function fails because we don't support DT_RESOURCE outputs
      # from XLA clusters.
      #
      # self.assertAllEqual(ta_out[0], ta_out[1])

  @test_util.disable_control_flow_v2("b/124334470")
  def testTensorArrayWriteWrongIndexOrDataTypeFails(self):
    # [FUNCTION DOC]
    # Test (disabled for control flow v2): writing an int32 tensor at index minus 1 into a float32
    # array must fail. Expects a value error or invalid-argument error whose message mentions the
    # dtype mismatch. KNOWN ISSUE: despite its name it only checks the wrong data type; the
    # negative index is not what triggers the checked message.
    with self.session(), self.test_scope():

      def fn():
        # [FUNCTION DOC]
        # Graph body: create a float32 array and return the flow after writing an int32 value.
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32, tensor_array_name="foo", size=3)
        return ta.write(-1, constant_op.constant(7)).flow

      # Test writing the wrong datatype.
      # TODO(b/129870929): Remove InvalidArgumentError/second regexp after all
      # callers provide proper init dtype.
      with self.assertRaisesRegex(
          (ValueError, errors.InvalidArgumentError), r"("
          r"conversion requested dtype float32 for Tensor with dtype int32"
          r"|"
          r"TensorArray dtype is float but op has dtype int32"
          r")"):
        xla.compile(fn)[0].eval()

  @test_util.disable_control_flow_v2("b/124334096 verify dtype")
  def testTensorArrayReadWrongIndexOrDataTypeFails(self):
    # [FUNCTION DOC]
    # Test (disabled for control flow v2): only runs if the device supports at least two float
    # types. Case one: reading with a different dtype than the array's must raise an op error
    # mentioning the array dtype. Case two: write at index 0 and read at index 1 under XLA must
    # compile and evaluate without raising. KNOWN ISSUE: case two merely compiles and evaluates
    # the result with no assertion, though the name suggests a failure is expected.
    # Find two different floating point types, create an array of
    # the first type, but try to read the other type.
    if len(self.float_types) > 1:
      dtype1, dtype2 = list(self.float_types)[:2]
      with self.session(), self.test_scope():

        def fn():
          # [FUNCTION DOC]
          # Graph body: write to a dtype1 array, then read it back requesting dtype2 through the
          # raw read op; returns that read.
          ta = tensor_array_ops.TensorArray(
              dtype=dtype1, tensor_array_name="foo", size=3)

          w0 = ta.write(0, math_ops.cast([[4.0, 5.0]], dtype1))

          # Test reading wrong datatype.
          return gen_data_flow_ops.tensor_array_read_v3(
              handle=w0.handle, index=0, dtype=dtype2, flow_in=w0.flow)

        with self.assertRaisesOpError("TensorArray dtype is "):
          self.evaluate(xla.compile(fn))

        def fn():
          # [FUNCTION DOC]
          # Graph body: write at index 0, read index 1 under a control dependency, return 1.0.
          ta = tensor_array_ops.TensorArray(
              dtype=dtype1, tensor_array_name="foo", size=3)

          w0 = ta.write(0, math_ops.cast([[4.0, 5.0]], dtype1))

          # Test reading from a different index than the one we wrote to
          with ops.control_dependencies([w0.read(1)]):
            return 1.0

        xla.compile(fn)[0].eval()

  @test_util.disable_control_flow_v2("b/122315872 (split)")
  def testTensorArraySplitIncompatibleShapesFails(self):
    # [FUNCTION DOC]
    # Test (disabled for control flow v2): four invalid split calls must each raise an error: a
    # scalar lengths argument (value error about rank), lengths not matching the value (op error
    # about unequal lengths), a scalar value (op error about rank), and a lengths vector whose size
    # differs from the array size (op error about size mismatch).
    with self.session(), self.test_scope():

      def fn():
        # [FUNCTION DOC]
        # Graph body: split a vector using a scalar as lengths; returns the flow.
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32,
            tensor_array_name="foo",
            size=3,
            infer_shape=False)
        return ta.split([1.0, 2.0, 3.0], 1).flow

      with self.assertRaisesWithPredicateMatch(
          ValueError, r"Shape must be rank 1 but is rank 0"):
        xla.compile(fn)[0].eval()

      def fn():
        # [FUNCTION DOC]
        # Graph body: split a three-element vector with lengths one, two, three; returns the flow.
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32,
            tensor_array_name="foo",
            size=3,
            infer_shape=False)
        return ta.split([1.0, 2.0, 3.0], [1, 2, 3]).flow

      with self.assertRaisesOpError(
          r"lengths must be equal: 1 vs. 2"):
        xla.compile(fn)[0].eval()

      def fn():
        # [FUNCTION DOC]
        # Graph body: split a scalar value with lengths of one element; returns the flow.
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32,
            tensor_array_name="foo",
            size=3,
            infer_shape=False)
        return ta.split(1.0, [1]).flow

      with self.assertRaisesOpError(
          r"value must have rank >= 1"):
        xla.compile(fn)[0].eval()

      def fn():
        # [FUNCTION DOC]
        # Graph body: split a one-element vector with one length into a size-2 array; returns flow.
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32,
            tensor_array_name="foo",
            size=2,
            infer_shape=False)

        return ta.split([1.0], [1]).flow

      with self.assertRaisesOpError(
          r"TensorArray's size is not equal to the size of lengths "
          r"\(1 vs. 2\)"):
        xla.compile(fn)[0].eval()

  def _testTensorArrayWriteGradientAddMultipleAdds(self, dtype):
    # [FUNCTION DOC]
    # HELPER: body of the gradient-accumulation test for one data type (dtype). Writing three
    # values to the same index of the gradient array must accumulate them (3 + 4 + 5 gives 12).
    # A second graph writes values of different shapes to the same gradient index and must raise an
    # op error about mismatched sizes.
    with self.session(), self.test_scope():
      c = lambda x: np.asarray(x, dtype=dtype.as_numpy_dtype)

      def fn():
        # [FUNCTION DOC]
        # Graph body: write twice to index 2, get the gradient array, write three values to its
        # index 2 and read the accumulated sum.
        ta = tensor_array_ops.TensorArray(
            dtype=dtype, tensor_array_name="foo", size=3, infer_shape=False)

        w0 = ta.write(2, c(3.0))
        w1 = w0.write(2, c(4.0))

        ta_grad = w1.grad("grad")

        w0_grad = ta_grad.write(2, c(3.0))
        w1_grad = w0_grad.write(2, c(4.0))
        w2_grad = w1_grad.write(2, c(5.0))

        return w2_grad.read(2)

      # Assert that aggregation works correctly
      self.assertAllEqual(c(12.00), xla.compile(fn)[0])

      def fn():
        # [FUNCTION DOC]
        # Graph body: write to the gradient array at the same index with a scalar and then a
        # one-element vector (differing shapes); returns the flow so the error surfaces on run.
        ta = tensor_array_ops.TensorArray(
            dtype=dtype, tensor_array_name="foo", size=3, infer_shape=False)

        w0 = ta.write(2, c(3.0))
        w1 = w0.write(2, c(4.0))

        ta_grad = w1.grad("grad")
        # Using differing shapes causes an exception
        wb0_grad = ta_grad.write(1, c(1.0))
        wb1_grad = wb0_grad.write(1, c([1.0]))

        return wb1_grad.flow

      with self.assertRaisesOpError(
          r"Mismatched TensorArray sizes"):
        xla.compile(fn)[0].eval()

  @test_util.disable_control_flow_v2("TensorArray.grad is not supported in v2")
  def testTensorArrayWriteGradientAddMultipleAdds(self):
    # [FUNCTION DOC]
    # Test (disabled for control flow v2): runs the gradient-accumulation check for all numeric types.
    for dtype in self.numeric_tf_types:
      self._testTensorArrayWriteGradientAddMultipleAdds(dtype)

  def testMultiTensorArray(self):
    # [FUNCTION DOC]
    # Test: two independent arrays each store one scalar (4 and 5); reading both and adding gives 9.
    with self.session(), self.test_scope():

      def fn():
        # [FUNCTION DOC]
        # Graph body: create two size-1 arrays, write a scalar to each, read both and return the sum.
        h1 = tensor_array_ops.TensorArray(
            size=1, dtype=dtypes.float32, tensor_array_name="foo")
        w1 = h1.write(0, 4.0)
        r1 = w1.read(0)

        h2 = tensor_array_ops.TensorArray(
            size=1, dtype=dtypes.float32, tensor_array_name="bar")

        w2 = h2.write(0, 5.0)
        r2 = w2.read(0)
        return r1 + r2

      self.assertAllClose(9.0, self.evaluate(xla.compile(fn)[0]))

  def _testTensorArrayGradientWriteReadType(self, dtype):
    # [FUNCTION DOC]
    # HELPER: body of the gradient write/read test for one numpy-compatible dtype. Writes two
    # values, reads them (one read twice) and computes gradients of the reads with respect to the
    # inputs, individually and combined, with given upstream gradients. Asserts that gradients of
    # repeated reads of one element add up, and combined gradients equal the individual ones.
    with self.session() as session, self.test_scope():
      c = lambda x: np.array(x, dtype=dtype)

      def fn():
        # [FUNCTION DOC]
        # Graph body: write two values, read them, compute four gradient sets (single read, two
        # reads of the same element, other element, and all combined) and return them.
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.as_dtype(dtype),
            tensor_array_name="foo",
            size=3,
            infer_shape=False)

        value_0 = constant_op.constant(c([[4.0, 5.0]]))
        value_1 = constant_op.constant(c([[3.0, 3.5]]))

        w0 = ta.write(0, value_0)
        w1 = w0.write(1, value_1)
        r0 = w1.read(0)
        r1 = w1.read(1)
        r0_2 = w1.read(0)

        # Test individual components' gradients
        grad_just_r0 = gradients_impl.gradients(
            ys=[r0], xs=[value_0], grad_ys=[c([[2.0, 3.0]])])
        grad_r0_r0_2 = gradients_impl.gradients(
            ys=[r0, r0_2],
            xs=[value_0],
            grad_ys=[c([[2.0, 3.0]]), c([[1.0, -1.0]])])
        grad_just_r1 = gradients_impl.gradients(
            ys=[r1], xs=[value_1], grad_ys=[c([[-2.0, -4.0]])])
        # Test combined gradients
        grad = gradients_impl.gradients(
            ys=[r0, r0_2, r1],
            xs=[value_0, value_1],
            grad_ys=[c([[2.0, 3.0]]),
                     c([[1.0, -1.0]]),
                     c([[-2.0, -10.0]])])

        return [grad_just_r0, grad_r0_r0_2, grad_just_r1, grad]

      [grad_just_r0_vals, grad_r0_r0_2_vals, grad_just_r1_vals,
       grad_vals] = self.evaluate(xla.compile(fn))

      self.assertAllEqual(c([[2.0, 3.0]]), grad_just_r0_vals[0])

      self.assertAllEqual(c([[3.0, 2.0]]), grad_r0_r0_2_vals[0])

      self.assertAllEqual(c([[-2.0, -4.0]]), grad_just_r1_vals[0])

      self.assertEqual(len(grad_vals), 2)
      self.assertAllEqual(c([[3.0, 2.0]]), grad_vals[0])
      self.assertAllEqual(c([[-2.0, -10.0]]), grad_vals[1])

  def testTensorArrayGradientWriteRead(self):
    # [FUNCTION DOC]
    # Test: runs the gradient write/read check for every float type and every complex type
    # supported by the XLA device.
    for dtype in self.float_types:
      self._testTensorArrayGradientWriteReadType(dtype)
    for dtype in self.complex_types:
      self._testTensorArrayGradientWriteReadType(dtype)

  def _testTensorArrayGradientWritePackConcatAndRead(self):
    # [FUNCTION DOC]
    # HELPER: body of the combined stack/read/concat gradient test. Writes two vectors, then gets
    # gradients of the stack, a single read and the concatenation with respect to the two inputs.
    # Asserts gradients from the different consumers of the same element are summed correctly.
    with self.session() as sess, self.test_scope():

      def fn():
        # [FUNCTION DOC]
        # Graph body: write two vectors, stack, read element 0 and concatenate, then return the
        # gradients of those three results with respect to the inputs.
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32,
            tensor_array_name="foo",
            size=2,
            clear_after_read=False)

        value_0 = constant_op.constant([-1.0, 1.0])
        value_1 = constant_op.constant([-10.0, 10.0])

        w0 = ta.write(0, value_0)
        w1 = w0.write(1, value_1)
        p0 = w1.stack()
        r0 = w1.read(0)
        s0 = w1.concat()

        # Test gradient accumulation between read(0), pack(), and concat().
        with ops.control_dependencies([p0, r0, s0]):
          return gradients_impl.gradients(
              ys=[p0, r0, s0],
              xs=[value_0, value_1],
              grad_ys=[
                  [[2.0, 3.0], [4.0, 5.0]],  # stack gradient
                  [-0.5, 1.5],  # read(0) gradient
                  [20.0, 30.0, 40.0, 50.0],  # concat gradient
              ])

      grad_vals = self.evaluate(xla.compile(fn))  # 2 + 2 entries

      self.assertAllClose([2.0 - 0.5 + 20.0, 3.0 + 1.5 + 30.0], grad_vals[0])
      self.assertAllEqual([4.0 + 40.0, 5.0 + 50.0], grad_vals[1])

  @test_util.disable_control_flow_v2("b/122315751 (concat)")
  def testTensorArrayGradientWritePackConcatAndRead(self):
    # [FUNCTION DOC]
    # Test (disabled for control flow v2): delegates to the stack/read/concat gradient helper.
    self._testTensorArrayGradientWritePackConcatAndRead()

  def testTensorArrayReadTwice(self):
    # [FUNCTION DOC]
    # Test: with clear-after-read disabled, an element can be read twice (second read ordered after
    # the first); the first returned read equals the first row of the unstacked value.
    with self.session(), self.test_scope():

      def fn():
        # [FUNCTION DOC]
        # Graph body: unstack a 2 by 2 constant, read element 0 twice (second under a control
        # dependency on the first) and return both reads.
        value = constant_op.constant([[1.0, -1.0], [10.0, -10.0]])

        ta_readtwice = tensor_array_ops.TensorArray(
            dtype=dtypes.float32,
            tensor_array_name="foo",
            size=2,
            clear_after_read=False)
        w_readtwice = ta_readtwice.unstack(value)
        r0_readtwice = w_readtwice.read(0)
        with ops.control_dependencies([r0_readtwice]):
          r1_readtwice = w_readtwice.read(0)

        return [r0_readtwice, r1_readtwice]

      self.assertAllEqual([1.0, -1.0], self.evaluate(xla.compile(fn))[0])

  def _testTensorArrayGradientUnpackRead(self):
    # [FUNCTION DOC]
    # HELPER: body of the unstack/read gradient test. Gradients of two reads of element 0 and one
    # read of element 1 with respect to the unstacked value are computed; asserts that gradients
    # of the repeated reads of element 0 are summed and element 1's gradient is kept separately.
    with self.session() as session, self.test_scope():

      def fn():
        # [FUNCTION DOC]
        # Graph body: unstack a 2 by 2 constant, read elements and return gradients of the reads
        # with respect to the constant.
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32,
            tensor_array_name="foo",
            size=2,
            clear_after_read=False)

        value = constant_op.constant([[1.0, -1.0], [10.0, -10.0]])

        w = ta.unstack(value)
        r0 = w.read(0)
        r0_1 = w.read(0)
        r1 = w.read(1)

        # Test combined gradients + aggregation of read(0).
        return gradients_impl.gradients(
            ys=[r0, r0_1, r1],
            xs=[value],
            grad_ys=[[2.0, 3.0], [-1.5, 1.5], [4.0, 5.0]])

      grad_vals = self.evaluate(xla.compile(fn))

      self.assertEqual(len(grad_vals), 1)
      self.assertAllEqual([[2.0 - 1.5, 3.0 + 1.5], [4.0, 5.0]], grad_vals[0])

  def testTensorArrayGradientUnpackRead(self):
    # [FUNCTION DOC]
    # Test: delegates to the unstack/read gradient helper.
    self._testTensorArrayGradientUnpackRead()

  @test_util.disable_control_flow_v2("b/122315751(concat), b/122315872(split)")
  def testTensorArrayGradientSplitConcat(self):
    # [FUNCTION DOC]
    # Test (disabled for control flow v2): splits a 4 by 2 constant into two pieces of two rows,
    # concatenates them and computes the gradient of the result with respect to the constant.
    # Asserts the gradient equals the supplied upstream gradient.
    with self.session() as session, self.test_scope():

      def fn():
        # [FUNCTION DOC]
        # Graph body: split the constant, concatenate and return the gradient of the concatenation.
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32, tensor_array_name="foo", size=2)

        value = constant_op.constant([[1.0, -1.0], [10.0, -10.0],
                                      [100.0, -100.0], [1000.0, -1000.0]])

        w = ta.split(value, [2, 2])
        r = w.concat()

        # Test combined gradients
        return gradients_impl.gradients(
            ys=[r],
            xs=[value],
            grad_ys=[[[2.0, -2.0], [20.0, -20.0], [200.0, -200.0],
                      [2000.0, -2000.0]]])

      grad_vals = self.evaluate(xla.compile(fn))

      self.assertEqual(len(grad_vals), 1)
      self.assertAllEqual([[2.0, -2.0], [20.0, -20.0], [200.0, -200.0],
                           [2000.0, -2000.0]],
                          grad_vals[0])

  def testCloseTensorArray(self):
    # [FUNCTION DOC]
    # Test: closing an array inside a compiled function succeeds (no error on evaluation).
    with self.session() as session, self.test_scope():

      def fn():
        # [FUNCTION DOC]
        # Graph body: create an array, close it, and return 1.0 after the close via a control dependency.
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32, tensor_array_name="foo", size=3)
        with ops.control_dependencies([ta.close()]):
          return 1.0

      self.evaluate(xla.compile(fn)[0])

  def testSizeTensorArray(self):
    # [FUNCTION DOC]
    # Test: the size operation of a size-3 array returns 3.
    with self.session(), self.test_scope():

      def fn():
        # [FUNCTION DOC]
        # Graph body: create a size-3 array and return its size.
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32, tensor_array_name="foo", size=3)
        return ta.size()

      self.assertAllEqual(3, self.evaluate(xla.compile(fn))[0])

  def testWriteCloseTensorArray(self):
    # [FUNCTION DOC]
    # Test: writing two values and then closing the array in a compiled function runs without error.
    with self.session(), self.test_scope():

      def fn():
        # [FUNCTION DOC]
        # Graph body: write two values, close the array and return 1.0 after the close.
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32,
            tensor_array_name="foo",
            size=3,
            infer_shape=False)
        w0 = ta.write(0, [[4.0, 5.0]])
        w1 = w0.write(1, [[3.0, 1.0]])
        with ops.control_dependencies([w1.close()]):
          return 1.0

      self.evaluate(xla.compile(fn))

  # TODO(phawkins): implement while loops.
  # def _testWhileLoopWritePackGradients(self, dynamic_size, dtype):
  #   np_dtype = dtype.as_numpy_dtype
  #   with self.session() as session, self.test_scope():
  #     v0 = array_ops.identity(np.arange(3 * 5, dtype=np_dtype).reshape(3, 5))
  #     var = variables.Variable(np.arange(100, 105, dtype=np_dtype))
  #     state0 = array_ops.identity(np.array([1] * 5, dtype=np_dtype))
  #     ta = tensor_array_ops.TensorArray(
  #         dtype=dtype,
  #         tensor_array_name="foo",
  #         size=0 if dynamic_size else 3,
  #         dynamic_size=dynamic_size)
  #     time_0 = array_ops.identity(0)

  #     def body(time, ta_t, state):
  #       sliced = array_ops.slice(
  #           v0, begin=array_ops_stack.stack([time, 0]), size=[1, -1])
  #       sliced = array_ops.squeeze(sliced)
  #       out = sliced + var + state
  #       state += sliced
  #       ta_t = ta_t.write(time, out)
  #       return (time + 1, ta_t, state)

  #     (unused_0, h_final, unused_2) = control_flow_ops.while_loop(
  #         cond=lambda time, unused_1, unused_2: time < 3,
  #         body=body,
  #         loop_vars=(time_0, ta, state0),
  #         shape_invariants=(time_0.get_shape(), tensor_shape.unknown_shape(),
  #                           tensor_shape.unknown_shape()),
  #         parallel_iterations=3)
  #     vout = h_final.stack()

  #     grad_val = -np.arange(3 * 5, dtype=np_dtype).reshape(3, 5)
  #     v0_grad = gradients_impl.gradients([vout], [v0], [grad_val])[0]
  #     state0_grad = gradients_impl.gradients([vout], [state0], [grad_val])[0]
  #     var_grad = gradients_impl.gradients([vout], [var], [grad_val])[0]

  #     self.evaluate(variables.global_variables_initializer())
  #     state0_t, var_t, v0_t, vout_t, v0_grad_t, var_grad_t, state0_grad_t = (
  #         self.evaluate([state0, var, v0, vout, v0_grad, var_grad, state0_grad])
  #     )
  #     just_v0_grad_t, = self.evaluate([v0_grad])

  #     # state = [ state0 | state0 + v0[0] | state0 + v0[0] + v0[1] ]
  #     # vout = [ v0[0] + var + state[0] |
  #     #          v0[1] + var + state[1] |
  #     #          v0[2] + var + state[2] ]
  #     #      = [ v0[0] + var + state0 |
  #     #          v0[1] + var + state0 + v0[0] |
  #     #          v0[2] + var + state0 + v0[0] + v0[1] ]
  #     #
  #     # d(vout[0])/d(v0) = [1 | 0 | 0 ]
  #     # d(vout[1])/d(v0) = [1 | 1 | 0 ]
  #     # d(vout[2])/d(v0) = [1 | 1 | 1 ]
  #     # d(vout)/d(var) = [1 | 1 | 1]
  #     # d(vout)/d(state0) = [ 1 | 1 | 1 ]

  #     state_per_time = np.array(
  #         [state0_t, state0_t + v0_t[0, :],
  #         state0_t + v0_t[0, :] + v0_t[1, :]])

  #     # Compare forward prop
  #     self.assertAllClose(v0_t + var_t + state_per_time, vout_t)

  #     # Compare backward prop
  #     expected_v0_grad_t = np.array([
  #         grad_val[0, :] + grad_val[1, :] + grad_val[2, :],
  #         grad_val[1, :] + grad_val[2, :], grad_val[2, :]
  #     ])

  #     self.assertAllEqual(expected_v0_grad_t, v0_grad_t)
  #     self.assertAllEqual(expected_v0_grad_t, just_v0_grad_t)
  #     self.assertAllClose(grad_val.sum(axis=0), var_grad_t)
  #     self.assertAllClose(grad_val.sum(axis=0), state0_grad_t)

  # def testWhileLoopWritePackGradients(self):
  #   self._testWhileLoopWritePackGradients(
  #       dynamic_size=False, dtype=dtypes.float32)
  #   # TODO(ebrevdo): re-enable when While supports non-float32 gradients.
  #   # self._testWhileLoopWritePackGradients(
  #   #     dynamic_size=False, dtype=tf.int64)

  # def testWhileLoopDynamicWritePackGradients(self):
  #   self._testWhileLoopWritePackGradients(
  #       dynamic_size=True, dtype=dtypes.float32)

  # def testGradSerialTwoLoops(self):
  #   with self.session(), self.test_scope():
  #     num_steps = 100
  #     acc = tensor_array_ops.TensorArray(
  #         dtype=dtypes.float32,
  #         size=num_steps,
  #         clear_after_read=False,
  #         element_shape=tensor_shape.scalar())
  #     i = constant_op.constant(0, name="i")
  #     x = constant_op.constant(2.0, name="x")

  #     c = lambda i, acc: i < 5

  #     def b(i, acc):
  #       x1 = cond.cond(
  #           math_ops.equal(i, 0), lambda: x,
  #           lambda: math_ops.multiply(acc.read(i - 1), 2.0))
  #       return i + 1, acc.write(i, x1)

  #     i1, acc1 = control_flow_ops.while_loop(c, b, [i, acc])

  #     z = constant_op.constant(0.0)

  #     def fn(i, acc):
  #       return i + 1, acc.write(i, z)

  #     _, acc2 = control_flow_ops.while_loop(lambda i, acc: i < num_steps, fn,
  #                                           [i1, acc1])

  #     r = acc2.stack()
  #     grad = gradients_impl.gradients(r, [x])[0]
  #     self.assertAllClose(31.0, self.evaluate(grad))

  def testSumOfTwoReadVariablesWithoutRepeatGrad(self):
    # [FUNCTION DOC]
    # Test: two 3 by 5 matrices are written to a size-2 array and read once each; the sum of the
    # reads is differentiated with respect to each input with an upstream gradient of negative
    # values. Gradients computed individually and jointly must both equal the upstream gradient.
    with self.session() as session, self.test_scope():
      g0 = -(np.arange(3 * 5, dtype=np.float32).reshape(3, 5) + 1)

      def fn():
        # [FUNCTION DOC]
        # Graph body: write two matrices, read both, add, and return the gradients of the sum with
        # respect to each matrix.
        a = array_ops.identity(
            np.arange(3 * 5, dtype=np.float32).reshape(3, 5) + 1)
        b = array_ops.identity(
            np.arange(3 * 5, dtype=np.float32).reshape(3, 5) + 1 + 3 * 5)
        ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=2)
        ta = ta.write(0, a, name="write_a")
        ta = ta.write(1, b, name="write_b")
        c = (
            ta.read(0, name="read_a_0") +  # a + b
            ta.read(1, name="read_b_0"))
        grad_a = gradients_impl.gradients([c], [a], [g0])[0]  # d(a+b)/da = 1
        grad_b = gradients_impl.gradients([c], [b], [g0])[0]  # d(a+b)/db = 1

        return [grad_a, grad_b]

      grad_a, grad_b = xla.compile(fn)

      # Test gradients calculated individually
      grad_a_t, = self.evaluate([grad_a])
      self.assertAllEqual(grad_a_t, g0)

      grad_b_t, = self.evaluate([grad_b])
      self.assertAllEqual(grad_b_t, g0)

      # Test gradients calculated jointly.
      joint_grad_a_t, joint_grad_b_t = self.evaluate([grad_a, grad_b])
      self.assertAllEqual(joint_grad_a_t, g0)
      self.assertAllEqual(joint_grad_b_t, g0)

  def testWriteShape(self):
    # [FUNCTION DOC]
    # Test: static shape inference through the array. Writing and reading a vector keeps its static
    # shape; writing two vectors keeps each shape on read; writing a vector of a different length
    # to the same index must raise a value error.
    with self.session(), self.test_scope():

      def fn():
        # [FUNCTION DOC]
        # Graph body: write a two-element vector, read it back, return the original and the read.
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32, tensor_array_name="foo", size=3)
        c0 = constant_op.constant([4.0, 5.0])
        w0 = ta.write(0, c0)
        r0 = w0.read(0)

        return [c0, r0]

      c0, r0 = xla.compile(fn)

      self.assertAllEqual(c0.get_shape(), r0.get_shape())

      def fn():
        # [FUNCTION DOC]
        # Graph body: write an outer vector at index 0 and a new vector at index 1, read both and
        # return the first read, the new vector and the second read.
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32, tensor_array_name="foo", size=3)
        c1 = constant_op.constant([6.0, 7.0])
        w0 = ta.write(0, c0)
        w1 = w0.write(1, c1)
        r0 = w1.read(0)
        r1 = w1.read(1)

        return [r0, c1, r1]

      [r0, c1, r1] = xla.compile(fn)

      self.assertAllEqual(c0.get_shape(), r0.get_shape())
      self.assertAllEqual(c1.get_shape(), r1.get_shape())

      def fn():
        # [FUNCTION DOC]
        # Graph body: write a two-element vector at index 0, then write a three-element vector at
        # index 0 again (shape conflict); returns the flow.
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32, tensor_array_name="foo", size=3)
        w0 = ta.write(0, c0)
        c2 = constant_op.constant([4.0, 5.0, 6.0])
        return w0.write(0, c2).flow

      with self.assertRaises(ValueError):
        self.evaluate(xla.compile(fn))

  def _testGradientWhenNotAllComponentsRead(self):
    # [FUNCTION DOC]
    # HELPER: body of the partial-read gradient test. Unstacks a two-element vector, reads only
    # element 0, and takes the gradient with respect to the vector; asserts it is one for the read
    # element and zero for the unread one.
    with self.session() as session, self.test_scope():

      def fn():
        # [FUNCTION DOC]
        # Graph body: unstack a constant, read element 0 and return the gradient of that read.
        ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=2)
        x = constant_op.constant([2.0, 3.0])
        w = ta.unstack(x)
        r0 = w.read(0)
        # Calculate (dr0/dx0, dr0/dx1).  since r0 = x0, gradients are (1, 0).
        return gradients_impl.gradients(ys=[r0], xs=[x], grad_ys=[1.0])

      grad_r0_vals = self.evaluate(xla.compile(fn))[0]
      self.assertAllEqual(grad_r0_vals, [1.0, 0.0])

  def testGradientWhenNotAllComponentsRead(self):
    # [FUNCTION DOC]
    # Test: delegates to the partial-read gradient helper.
    self._testGradientWhenNotAllComponentsRead()

  def _testTensorArrayEvalEmpty(self):
    # [FUNCTION DOC]
    # HELPER: body of the empty-array test. Stacking a size-0 array with shape inference disabled
    # and no writes must raise an invalid-argument error saying the array is uninitialized.
    with self.session(), self.test_scope():

      def fn():
        # [FUNCTION DOC]
        # Graph body: create a size-0 array without shape inference and return its stack.
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32, size=0, infer_shape=False)
        return ta.stack()

      with self.assertRaisesWithPredicateMatch(
          errors.InvalidArgumentError, "Uninitialized TensorArray passed to "
          "TensorArrayStack/TensorArrayGatherV3"):
        xla.compile(fn)[0].eval()

  @test_util.disable_control_flow_v2("b/124335246")
  def testTensorArrayEvalEmpty(self):
    # [FUNCTION DOC]
    # Test (disabled for control flow v2): delegates to the empty-array error helper.
    self._testTensorArrayEvalEmpty()

  def _testTensorArrayEvalEmptyWithDefault(self):
    # [FUNCTION DOC]
    # HELPER: body of the empty-array-with-shape-inference test. A size-0 array unstacked from a
    # zero by 3 by 5 tensor reports size 0 and stacks to shape 0, 3, 5. When control flow v2 is
    # not enabled, the concatenation additionally must have shape 0, 5.
    with self.session(), self.test_scope():

      def fn():
        # [FUNCTION DOC]
        # Graph body: create a size-0 array, get its size, unstack an empty tensor and return the
        # size and the stack.
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32, size=0, infer_shape=True)
        size = ta.size()
        ta = ta.unstack(array_ops.zeros([0, 3, 5]))
        return [size, ta.stack()]

      [size, stack] = self.evaluate(xla.compile(fn))
      self.assertEqual(0, size)
      self.assertAllEqual([0, 3, 5], stack.shape)
      # Concatenating zero tensors along their first dimension gives a
      # first dimension of zero
      if not control_flow_util.ENABLE_CONTROL_FLOW_V2:

        def fn():
          # [FUNCTION DOC]
          # Graph body: unstack an empty tensor into a size-0 array and return its concatenation.
          ta = tensor_array_ops.TensorArray(
              dtype=dtypes.float32, size=0, infer_shape=True)
          ta = ta.unstack(array_ops.zeros([0, 3, 5]))
          return ta.concat()

        # TODO(b/122315751): Enable this.
        self.assertAllEqual([0, 5], self.evaluate(xla.compile(fn))[0].shape)

  def testTensorArrayEvalEmptyWithDefault(self):
    # [FUNCTION DOC]
    # Test: delegates to the empty-array-with-default-shape helper.
    self._testTensorArrayEvalEmptyWithDefault()

  def _testTensorArrayScatterRead(self, tf_dtype):
    # [FUNCTION DOC]
    # HELPER: body of the scatter-and-read test for one data type (tf_dtype). Scatters two rows to
    # indices 1 and 8 of a size-10 array, reads them with indices fed through placeholders and
    # asserts the rows read back are as scattered.
    with self.session() as session, self.test_scope():
      convert = _make_converter(tf_dtype)
      id0 = array_ops.placeholder(dtypes.int32)
      id1 = array_ops.placeholder(dtypes.int32)

      def fn():
        # [FUNCTION DOC]
        # Graph body: scatter two rows into the array and read at the two fed indices.
        ta = tensor_array_ops.TensorArray(
            dtype=tf_dtype, tensor_array_name="foo", size=10)

        indices = constant_op.constant([1, 8])
        value = constant_op.constant(convert([[1.0, 5.0], [10.0, 20.0]]))

        w = ta.scatter(indices, value)
        r0 = w.read(id0)
        r1 = w.read(id1)

        return [r0, r1]

      # Test aggregation of read
      read_vals = session.run(xla.compile(fn), feed_dict={id0: 1, id1: 8})
      self.assertAllEqual(convert([1.0, 5.0]), read_vals[0])
      self.assertAllEqual(convert([10.0, 20.0]), read_vals[1])

  @test_util.disable_control_flow_v2("b/122315734 (scatter)")
  def testTensorArrayScatterRead(self):
    # [FUNCTION DOC]
    # Test (disabled for control flow v2): runs the scatter-and-read check for all numeric types and
    # also for boolean.
    for dtype in self.numeric_tf_types:
      self._testTensorArrayScatterRead(dtype)
    self._testTensorArrayScatterRead(dtypes.bool)

  @test_util.disable_control_flow_v2("b/122315734 (scatter)")
  def testTensorArrayScatterReadAndGradients(self):
    # [FUNCTION DOC]
    # Test (disabled for control flow v2): scatters two rows to indices 1 and 8 of a size-10 array,
    # reads them at fed indices and takes gradients of the reads with respect to the scattered
    # value. Asserts the read values and that the gradient equals the upstream gradients.
    with self.session() as session, self.test_scope():
      id0 = array_ops.placeholder(dtypes.int32)
      id1 = array_ops.placeholder(dtypes.int32)

      def fn():
        # [FUNCTION DOC]
        # Graph body: scatter, read at two fed indices, compute gradients; returns reads and grads.
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32, tensor_array_name="foo", size=10)

        indices = constant_op.constant([1, 8])
        value = constant_op.constant([[1.0, -1.0], [10.0, -10.0]])

        w = ta.scatter(indices, value)
        r0 = w.read(id0)
        r1 = w.read(id1)

        # Test combined gradients + aggregation of read(0).
        grad = gradients_impl.gradients(
            ys=[r0, r1], xs=[value], grad_ys=[[2.0, 3.0], [4.0, 5.0]])
        return [[r0, r1], grad]

      read_vals, grad_vals = session.run(
          xla.compile(fn), feed_dict={
              id0: 1,
              id1: 8
          })

      self.assertEqual(len(read_vals), 2)
      self.assertEqual(len(grad_vals), 1)
      self.assertAllEqual([1.0, -1.0], read_vals[0])
      self.assertAllEqual([10.0, -10.0], read_vals[1])
      self.assertAllEqual([[2.0, 3.0], [4.0, 5.0]], grad_vals[0])

  @test_util.disable_control_flow_v2("b/122315378 (gather)")
  def testTensorArrayWriteGatherAndGradients(self):
    # [FUNCTION DOC]
    # Test (disabled for control flow v2): unstacks a 10 by 2 value, gathers rows 1 and 8, and
    # takes the gradient of the gathered result with respect to the value. Asserts the gathered
    # rows and that the gradient is zero except at rows 1 and 8, which get the upstream gradients.
    with self.session() as session, self.test_scope():

      def fn():
        # [FUNCTION DOC]
        # Graph body: unstack, gather two indices, compute the gradient; returns gather and grads.
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32, tensor_array_name="foo", size=10)

        values = constant_op.constant([[1.0 * x, -1.0 * x] for x in range(10)])
        indices = constant_op.constant([1, 8])

        w = ta.unstack(values)
        g = w.gather(indices)

        # Test combined gradients + aggregation of read(0).
        grad = gradients_impl.gradients(
            ys=[g], xs=[values], grad_ys=[[[2.0, 3.0], [4.0, 5.0]]])
        return [[g], grad]

      g_vals, grad_vals = self.evaluate(xla.compile(fn))

      # Gradients for 8 of the 10 unread components are zero.
      expected_grad = np.zeros((10, 2))
      expected_grad[1] = [2.0, 3.0]
      expected_grad[8] = [4.0, 5.0]

      self.assertEqual(len(g_vals), 1)
      self.assertEqual(len(grad_vals), 1)
      self.assertAllEqual([[1.0, -1.0], [8.0, -8.0]], g_vals[0])
      self.assertAllEqual(expected_grad, grad_vals[0])

  def testTensorArrayIdentity(self):
    # [FUNCTION DOC]
    # Test: the identity op on arrays returns a usable array with preserved dtype and contents, and
    # respects control dependencies. Two resource variables are incremented before each identity;
    # asserts both are 1.0, array dtypes are float32 and int32, reads return 0 and 1 and sizes are
    # 2 and 4.
    with self.session() as session, self.test_scope():
      tensor_arrays = {}

      v0 = resource_variable_ops.ResourceVariable(0.0)
      v1 = resource_variable_ops.ResourceVariable(0.0)

      def fn():
        # [FUNCTION DOC]
        # Graph body: create a float and an int array, write one element each, apply identity under
        # control dependencies that increment the variables, record the arrays in the enclosing
        # mapping and return reads, sizes and the variables.
        ta0 = tensor_array_ops.TensorArray(
            dtype=dtypes.float32, size=2, infer_shape=False)
        ta1 = tensor_array_ops.TensorArray(
            dtype=dtypes.int32, size=4, infer_shape=True)

        ta0 = ta0.write(0, 0.)
        ta1 = ta1.write(0, 1)

        with ops.control_dependencies([v0.assign_add(1.0)]):
          ta0 = ta0.identity()

        with ops.control_dependencies([v1.assign_add(1.0)]):
          ta1 = ta1.identity()

        read0 = ta0.read(0)
        read1 = ta1.read(0)

        size0 = ta0.size()
        size1 = ta1.size()

        tensor_arrays[0] = ta0
        tensor_arrays[1] = ta1

        return [read0, read1, size0, size1, v0, v1]

      self.evaluate(variables.global_variables_initializer())

      read0_v, read1_v, size0_v, size1_v, v0, v1 = self.evaluate(
          xla.compile(fn))

      # Tests correct properties on new TensorArrays.
      self.assertEqual(dtypes.float32, tensor_arrays[0].dtype)
      self.assertEqual(dtypes.int32, tensor_arrays[1].dtype)

      # Tests that the control dependencies was added and executed.
      self.assertEqual(1.0, v0)
      self.assertEqual(1.0, v1)

      # Tests correct TensorArray.
      self.assertEqual(read0_v, 0)
      self.assertEqual(read1_v, 1)
      self.assertEqual(size0_v, 2)
      self.assertEqual(size1_v, 4)

if __name__ == "__main__":
  test.main()
