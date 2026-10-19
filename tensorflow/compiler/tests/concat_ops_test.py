# [MODULE DOC]
# Functional tests for the XLA implementations of the concat, concat-offset and pack (stack) ops
# on the XLA test device. Covers forward results, gradients, tuple inputs, scalar rejection and a
# large-number-of-inputs stress case. Defines three test classes and runs googletest main when
# executed as a script.
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
"""Functional tests for XLA Concat Op."""

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python import pywrap_sanitizers
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest


class ConcatTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # XLA test case for the concat op: basic stacking, dtypes, axis types, random shapes,
  # gradients and error handling.

  def testHStack(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name suggests horizontal stacking, but the test concatenates along axis 0
    # (vertical stacking).
    # Concatenates two random 4x4 float32 placeholders along axis 0 inside the test scope and
    # asserts the result shape equals the static shape and that the top and bottom halves equal
    # the first and second inputs. Returns nothing.
    with self.session():
      p1 = array_ops.placeholder(dtypes.float32, shape=[4, 4])
      p2 = array_ops.placeholder(dtypes.float32, shape=[4, 4])
      with self.test_scope():
        c = array_ops.concat([p1, p2], 0)
      params = {
          p1: np.random.rand(4, 4).astype("f"),
          p2: np.random.rand(4, 4).astype("f")
      }
      result = c.eval(feed_dict=params)

    self.assertEqual(result.shape, c.get_shape())
    self.assertAllEqual(result[:4, :], params[p1])
    self.assertAllEqual(result[4:, :], params[p2])

  def testVStack(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name suggests vertical stacking, but the test concatenates along axis 1
    # (horizontal stacking).
    # Concatenates two random 4x4 float32 placeholders along axis 1 and asserts the result shape
    # matches the static shape and that the left and right halves equal the inputs. Returns
    # nothing.
    with self.session():
      p1 = array_ops.placeholder(dtypes.float32, shape=[4, 4])
      p2 = array_ops.placeholder(dtypes.float32, shape=[4, 4])
      with self.test_scope():
        c = array_ops.concat([p1, p2], 1)
      params = {
          p1: np.random.rand(4, 4).astype("f"),
          p2: np.random.rand(4, 4).astype("f")
      }
      result = c.eval(feed_dict=params)

    self.assertEqual(result.shape, c.get_shape())
    self.assertAllEqual(result[:, :4], params[p1])
    self.assertAllEqual(result[:, 4:], params[p2])

  def testInt32(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: random values in the unit interval are cast to int32, so all data is zero and
    # the test only weakly checks the values.
    # Concatenates two 2x3 int32 constants along axis 0 and asserts the first two rows equal the
    # first input and the remaining rows equal the second. Returns nothing.
    with self.session():
      p1 = np.random.rand(2, 3).astype("i")
      p2 = np.random.rand(2, 3).astype("i")
      x1 = constant_op.constant(p1)
      x2 = constant_op.constant(p2)
      with self.test_scope():
        c = array_ops.concat([x1, x2], 0)
      result = self.evaluate(c)
    self.assertAllEqual(result[:2, :], p1)
    self.assertAllEqual(result[2:, :], p2)

  def testAxisInt64(self):
    # [FUNCTION DOC]
    # Same as the int32 test, but the concat axis is supplied as a 64-bit integer constant (zero).
    # Asserts the concatenated result's two row blocks equal the two inputs. The input data is
    # all zeros because random unit-interval values are cast to integers. Returns nothing.
    with self.session():
      p1 = np.random.rand(2, 3).astype("i")
      p2 = np.random.rand(2, 3).astype("i")
      x1 = constant_op.constant(p1)
      x2 = constant_op.constant(p2)
      axis = constant_op.constant(0, dtype=dtypes.int64)
      with self.test_scope():
        c = array_ops.concat([x1, x2], axis)
      result = self.evaluate(c)
    self.assertAllEqual(result[:2, :], p1)
    self.assertAllEqual(result[2:, :], p2)

  def _testRandom(self, dtype):
    # [FUNCTION DOC]
    # HELPER: shared body of the randomized concat test, parameterized by dtype.
    # Input: dtype is the tensor dtype to concatenate. Picks a random rank-5 shape, a random
    # number (2 to 9) of tensors and a random concat dimension, giving each tensor a random size
    # along that dimension. Placeholders are fed as float32 when the dtype is bfloat16 and cast to
    # bfloat16 before concat and back after. Asserts the result shape and that each input's slice
    # appears at the right offset (exact equality, or tolerance 0.01 when casting was used).
    # Uses global numpy randomness; returns nothing.
    # Random dims of rank 5
    shape = np.random.randint(1, 5, size=5)
    # Random number of tensors, but always > 1.
    num_tensors = np.random.randint(2, 10)
    # Random dim to concat on
    concat_dim = np.random.randint(5)
    params = {}
    if dtype == dtypes.bfloat16:
      dtype_feed = dtypes.float32
    else:
      dtype_feed = dtype
    with self.session():
      p = []
      for i in np.arange(num_tensors):
        input_shape = shape
        input_shape[concat_dim] = np.random.randint(1, 5)
        placeholder = array_ops.placeholder(dtype_feed, shape=input_shape)
        p.append(placeholder)

        t = dtype_feed.as_numpy_dtype
        params[placeholder] = np.random.rand(*input_shape).astype(t)

      if dtype != dtype_feed:
        concat_inputs = [math_ops.cast(p_i, dtype) for p_i in p]
      else:
        concat_inputs = p
      with self.test_scope():
        c = array_ops.concat(concat_inputs, concat_dim)
        if dtype != dtype_feed:
          c = math_ops.cast(c, dtype_feed)
      result = c.eval(feed_dict=params)

    self.assertEqual(result.shape, c.get_shape())
    cur_offset = 0

    for i in np.arange(num_tensors):
      # The index into the result is the ':' along all dimensions
      # except the concat_dim. slice(0, size) is used for ':', and
      # a list of slices is used to index into result.
      ind = [slice(0, params[p[i]].shape[j]) for j in np.arange(5)]
      ind[concat_dim] = slice(cur_offset,
                              cur_offset + params[p[i]].shape[concat_dim])
      cur_offset += params[p[i]].shape[concat_dim]
      if dtype == dtype_feed:
        self.assertAllEqual(result[tuple(ind)], params[p[i]])
      else:
        self.assertAllClose(result[tuple(ind)], params[p[i]], 0.01)

  def testRandom(self):
    # [FUNCTION DOC]
    # Runs the randomized concat helper for float32 and then int32. Returns nothing.
    self._testRandom(dtypes.float32)
    self._testRandom(dtypes.int32)

  def _testGradientsSimple(self):
    # [FUNCTION DOC]
    # HELPER: body of the gradient test for concat along axis 1.
    # Builds three random float32 constants of shapes 10 by 1, 2 and 6 by 2, concatenates them
    # along axis 1 (giving 10 by 9 by 2), computes gradients of the output with a random upstream
    # gradient, concatenates the input gradients along axis 1 and asserts they equal the upstream
    # gradient exactly. Returns nothing.
    with self.session():
      inp = []
      inp_tensors = []
      with self.test_scope():
        for x in [1, 2, 6]:
          shape = [10, x, 2]
          t = np.random.rand(*shape).astype("f")
          inp.append(t)
          inp_tensors.append(
              constant_op.constant(
                  [float(y) for y in t.flatten()],
                  shape=shape,
                  dtype=dtypes.float32))
        c = array_ops.concat(inp_tensors, 1)
        output_shape = [10, 9, 2]
        grad_inp = np.random.rand(*output_shape).astype("f")
        grad_tensor = constant_op.constant(
            [float(x) for x in grad_inp.flatten()], shape=output_shape)
        grad = gradients_impl.gradients([c], inp_tensors, [grad_tensor])
        concated_grad = array_ops.concat(grad, 1)
      result = self.evaluate(concated_grad)
    self.assertAllEqual(result, grad_inp)

  def testGradientsSimpleAll(self):
    # [FUNCTION DOC]
    # Test entry point that runs the simple (axis 1) concat gradient helper. Returns nothing.
    self._testGradientsSimple()

  def _testGradientsFirstDim(self):
    # [FUNCTION DOC]
    # HELPER: body of the gradient test for concat along the first axis.
    # Same scheme as the simple gradient helper, but tensors have shapes 1, 2 and 6 by 10 by 2 and
    # are concatenated along axis 0. Asserts the re-concatenated input gradients equal the random
    # upstream gradient. Returns nothing.
    with self.session():
      inp = []
      inp_tensors = []
      with self.test_scope():
        for x in [1, 2, 6]:
          shape = [x, 10, 2]
          t = np.random.rand(*shape).astype("f")
          inp.append(t)
          inp_tensors.append(
              constant_op.constant(
                  [float(y) for y in t.flatten()],
                  shape=shape,
                  dtype=dtypes.float32))
        c = array_ops.concat(inp_tensors, 0)
        output_shape = [9, 10, 2]
        grad_inp = np.random.rand(*output_shape).astype("f")
        grad_tensor = constant_op.constant(
            [float(x) for x in grad_inp.flatten()], shape=output_shape)
        grad = gradients_impl.gradients([c], inp_tensors, [grad_tensor])
        concated_grad = array_ops.concat(grad, 0)
        result = self.evaluate(concated_grad)

    self.assertAllEqual(result, grad_inp)

  def testGradientsFirstDimAll(self):
    # [FUNCTION DOC]
    # Test entry point that runs the first-axis concat gradient helper. Returns nothing.
    self._testGradientsFirstDim()

  def _testGradientsLastDim(self):
    # [FUNCTION DOC]
    # HELPER: body of the gradient test for concat along the last axis.
    # Tensors have shapes 10 by 2 by 1, 2 and 6, concatenated along axis 2. Asserts the
    # re-concatenated input gradients equal the random upstream gradient. Returns nothing.
    with self.session():
      inp = []
      inp_tensors = []
      with self.test_scope():
        for x in [1, 2, 6]:
          shape = [10, 2, x]
          t = np.random.rand(*shape).astype("f")
          inp.append(t)
          inp_tensors.append(
              constant_op.constant(
                  [float(y) for y in t.flatten()],
                  shape=shape,
                  dtype=dtypes.float32))
        c = array_ops.concat(inp_tensors, 2)
        output_shape = [10, 2, 9]
        grad_inp = np.random.rand(*output_shape).astype("f")
        grad_tensor = constant_op.constant(
            [float(x) for x in grad_inp.flatten()], shape=output_shape)
        grad = gradients_impl.gradients([c], inp_tensors, [grad_tensor])
        concated_grad = array_ops.concat(grad, 2)
        result = self.evaluate(concated_grad)

    self.assertAllEqual(result, grad_inp)

  def testGradientsLastDimAll(self):
    # [FUNCTION DOC]
    # Test entry point that runs the last-axis concat gradient helper. Returns nothing.
    self._testGradientsLastDim()

  def _RunAndVerifyGradientsRandom(self):
    # [FUNCTION DOC]
    # HELPER: randomized gradient check for concat.
    # Picks a random rank-5 shape, a random number (1 to 9) of tensors, a random concat dimension
    # and a random size along it per tensor; concatenates float32 constants, backpropagates a
    # random upstream gradient, re-concatenates the input gradients and asserts they equal the
    # upstream gradient. Returns nothing; uses global numpy randomness.
    # Random dims of rank 5
    input_shape = np.random.randint(1, 5, size=5)
    # Random number of tensors
    num_tensors = np.random.randint(1, 10)
    # Random dim to concat on
    concat_dim = np.random.randint(5)
    concat_dim_sizes = np.random.randint(1, 5, size=num_tensors)
    with self.session():
      inp = []
      inp_tensors = []
      with self.test_scope():
        for x in concat_dim_sizes:
          shape = input_shape
          shape[concat_dim] = x
          t = np.random.rand(*shape).astype("f")
          inp.append(t)
          inp_tensors.append(
              constant_op.constant(
                  [float(y) for y in t.flatten()],
                  shape=shape,
                  dtype=dtypes.float32))
        c = array_ops.concat(inp_tensors, concat_dim)
        output_shape = input_shape
        output_shape[concat_dim] = concat_dim_sizes.sum()
        grad_inp = np.random.rand(*output_shape).astype("f")
        grad_tensor = constant_op.constant(
            [float(x) for x in grad_inp.flatten()], shape=output_shape)
        grad = gradients_impl.gradients([c], inp_tensors, [grad_tensor])
        concated_grad = array_ops.concat(grad, concat_dim)
        result = self.evaluate(concated_grad)

    self.assertAllEqual(result, grad_inp)

  def testGradientsRandom(self):
    # [FUNCTION DOC]
    # Repeats the randomized concat gradient check five times. Returns nothing.
    for _ in range(5):
      self._RunAndVerifyGradientsRandom()

  # Re-enable once zero-element Retvals are handled correctly.
  def DISABLED_testZeroSize(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: disabled (name prefix prevents the runner from collecting it) until zero-element
    # return values are handled correctly, so it never runs.
    # Intended scenario: with a fixed random seed, concatenates pairs of random arrays that may
    # have zero-length dimensions, for several prefix and suffix shapes, and asserts the result
    # matches the reference concatenation and that the gradients re-concatenate to the upstream
    # gradient. Returns nothing.
    # Verify that concat doesn't crash and burn for zero size inputs
    np.random.seed(7)
    with self.session():
      with self.test_scope():
        for shape0 in (), (2,):
          axis = len(shape0)
          for shape1 in (), (3,):
            for n0 in 0, 1, 2:
              for n1 in 0, 1, 2:
                x0 = np.random.randn(*(shape0 + (n0,) + shape1))
                x1 = np.random.randn(*(shape0 + (n1,) + shape1))
                correct = np.concatenate([x0, x1], axis=axis)
                # TODO(irving): Make tf.concat handle map, then drop list().
                xs = list(map(constant_op.constant, [x0, x1]))
                c = array_ops.concat(xs, axis)
                self.assertAllEqual(c, correct)
                # Check gradients
                dc = np.random.randn(*c.get_shape().as_list())
                dxs = self.evaluate(gradients_impl.gradients(c, xs, dc))
                self.assertAllEqual(dc, np.concatenate(dxs, axis=axis))

  def testConcatTuple(self):
    # [FUNCTION DOC]
    # Concatenates two random 4x4 float32 arrays along axis 0 once passed as a list and once as a
    # tuple, and asserts both results are equal. Returns nothing.
    c1 = np.random.rand(4, 4).astype(np.float32)
    c2 = np.random.rand(4, 4).astype(np.float32)
    with self.session():
      with self.test_scope():
        concat_list_t = array_ops.concat([c1, c2], 0)
        concat_tuple_t = array_ops.concat((c1, c2), 0)
      self.assertAllEqual(concat_list_t, self.evaluate(concat_tuple_t))

  def testConcatNoScalars(self):
    # [FUNCTION DOC]
    # Asserts that concatenating three scalar constants with a placeholder axis raises a value
    # error stating that scalars cannot be concatenated and that stack should be used. Returns
    # nothing.
    with self.session():
      with self.test_scope():
        scalar = constant_op.constant(7)
        dim = array_ops.placeholder(dtypes.int32)
        with self.assertRaisesRegex(
            ValueError, r"Can't concatenate scalars \(use tf\.stack instead\)"):
          array_ops.concat([scalar, scalar, scalar], dim)

  # The purpose of this is to ensure that XLA on GPU will not run out of memory
  # with too many arguments.
  def testConcatLargeNumberOfTensors(self):
    # [FUNCTION DOC]
    # Stress test: concatenates 1001 random 7 by 13 float32 placeholders along axis 0 and then
    # axis 1 in the test scope, checking the result shape and that every input appears at the
    # correct offset. Skipped on CPU devices and under address, thread, memory or undefined
    # behavior sanitizers because of timeouts. Returns nothing.
    if "CPU" in self.device:
      self.skipTest("This test can time out on CPU, so we will just allow "
                    "other backends to catch this specific error.")
    if (pywrap_sanitizers.is_asan_enabled() or
        pywrap_sanitizers.is_tsan_enabled() or
        pywrap_sanitizers.is_msan_enabled() or
        pywrap_sanitizers.is_ubsan_enabled()):
      self.skipTest("This test can time out on *SAN.")
    with self.session():
      with self.test_scope():
        for concat_dim in range(2):
          params = {}
          p = []
          shape = np.array([7, 13])
          num_tensors = 1001
          for i in np.arange(num_tensors):
            input_shape = shape
            placeholder = array_ops.placeholder(
                dtypes.float32, shape=input_shape)
            p.append(placeholder)
            params[placeholder] = np.random.rand(*input_shape).astype(
                np.float32)

          concat_inputs = p
          c = array_ops.concat(concat_inputs, concat_dim)
          result = c.eval(feed_dict=params)

          self.assertEqual(result.shape, c.get_shape())
          cur_offset = 0

          for i in np.arange(num_tensors):
            # The index into the result is the ':' along all dimensions
            # except the concat_dim. slice(0, size) is used for ':', and
            # a list of slices is used to index into result.
            index = [slice(0, params[p[i]].shape[j]) for j in np.arange(2)]
            index[concat_dim] = slice(
                cur_offset, cur_offset + params[p[i]].shape[concat_dim])
            cur_offset += params[p[i]].shape[concat_dim]
            self.assertAllEqual(result[tuple(index)], params[p[i]])


class ConcatOffsetTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # XLA test case for the concat-offset op.

  def testBasic(self):
    # [FUNCTION DOC]
    # Computes concat offsets along dimension 1 for shapes (2,3,5), (2,7,5) and (2,20,5) and
    # asserts the offsets are zero except the dimension-1 entries 0, 3 and 10. Returns nothing.
    with self.session():
      with self.test_scope():
        cdim = constant_op.constant(1, dtypes.int32)
        s0 = constant_op.constant([2, 3, 5], dtypes.int32)
        s1 = constant_op.constant([2, 7, 5], dtypes.int32)
        s2 = constant_op.constant([2, 20, 5], dtypes.int32)
        off = gen_array_ops.concat_offset(cdim, [s0, s1, s2])
        ans = self.evaluate(off)
        self.assertAllEqual(ans, [[0, 0, 0], [0, 3, 0], [0, 10, 0]])


class PackTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # XLA test case for the pack (stack) op.

  def testBasic(self):
    # [FUNCTION DOC]
    # Stacks three int32 vectors of length three and asserts the 3x3 result holds them as rows.
    # Returns nothing.
    with self.session():
      with self.test_scope():
        s0 = constant_op.constant([2, 3, 5], dtypes.int32)
        s1 = constant_op.constant([2, 7, 5], dtypes.int32)
        s2 = constant_op.constant([2, 20, 5], dtypes.int32)
        packed = array_ops_stack.stack([s0, s1, s2])
        ans = self.evaluate(packed)
        self.assertAllEqual(ans, [[2, 3, 5], [2, 7, 5], [2, 20, 5]])

  def testScalars(self):
    # [FUNCTION DOC]
    # Stacks three int32 scalars (2, 3, 5) and asserts the result is the vector of those values.
    # Returns nothing.
    with self.session():
      with self.test_scope():
        s0 = constant_op.constant(2, dtypes.int32)
        s1 = constant_op.constant(3, dtypes.int32)
        s2 = constant_op.constant(5, dtypes.int32)
        packed = array_ops_stack.stack([s0, s1, s2])
        ans = self.evaluate(packed)
        self.assertAllEqual(ans, [2, 3, 5])

  def testEmpty(self):
    # [FUNCTION DOC]
    # Stacks three int32 tensors of shape one by zero and asserts the result has shape three by
    # one by zero (compared with nested empty lists). Returns nothing.
    with self.session():
      with self.test_scope():
        s0 = constant_op.constant([[]], dtypes.int32)
        s1 = constant_op.constant([[]], dtypes.int32)
        s2 = constant_op.constant([[]], dtypes.int32)
        packed = array_ops_stack.stack([s0, s1, s2])
        ans = self.evaluate(packed)
        self.assertAllEqual(ans, [[[]], [[]], [[]]])


if __name__ == "__main__":
  googletest.main()
