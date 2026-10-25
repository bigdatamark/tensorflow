# [MODULE DOC]
# Tests for the nth-element op, which selects the n-th smallest (or, with the reverse flag,
# n-th largest) value along the last dimension. Covers small hand-computed examples, large
# random inputs of several shapes and dtypes compared with numpy sorting, enumeration of
# every valid n, error handling for invalid rank and invalid n, and gradients in eager and
# graph modes. No import-time side effects beyond importing nn gradient registrations.
# Running as a script starts the standard test runner.
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

import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


class NthElementTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the nth-element op. Uses a shared helper that evaluates the op on CPU and
  # compares shape and values; individual tests cover examples, large and enumerated inputs,
  # invalid arguments (static and with placeholders) and gradients.

  def _validateNthElement(self, inputs, dtype, n, reverse, expected_values):
    # [FUNCTION DOC]
    # HELPER: shared assertion helper for nth-element tests.
    # Converts inputs (nested list or array) to a tensor of the given dtype, applies the op
    # with the integer n and boolean reverse flag, evaluates it in a CPU-only cached session,
    # and asserts the result has the shape of and is close to expected_values. Returns nothing.
    np_expected_values = np.array(expected_values)
    with self.cached_session(use_gpu=False) as sess:
      inputs_op = ops.convert_to_tensor(inputs, dtype=dtype)
      values_op = nn_ops.nth_element(inputs_op, n, reverse=reverse)
      values = self.evaluate(values_op)

      self.assertShapeEqual(np_expected_values, values_op)
      self.assertAllClose(np_expected_values, values)

  def testExample1(self):
    # [FUNCTION DOC]
    # One dimensional float32 input of five values: expects the second smallest (2.2) and, with
    # reverse, the second largest (4.4).
    inputs = [2.2, 4.4, 1.1, 5.5, 3.3]
    self._validateNthElement(inputs, dtypes.float32, 1, False, 2.2)
    self._validateNthElement(inputs, dtypes.float32, 1, True, 4.4)

  def testExample2(self):
    # [FUNCTION DOC]
    # Two by three float64 input with n equal to 2: expects the largest per row (4.4, 6.6) and
    # with reverse the smallest per row (1.1, 3.3). The op works along the last dimension.
    inputs = [[2.2, 4.4, 1.1], [5.5, 3.3, 6.6]]
    self._validateNthElement(inputs, dtypes.float64, 2, False, [4.4, 6.6])
    self._validateNthElement(inputs, dtypes.float64, 2, True, [1.1, 3.3])

  def testExample3(self):
    # [FUNCTION DOC]
    # Three dimensional integer input with n equal to 0, in int32 and int64: expects the
    # minimum per innermost row without reverse and the maximum per row with reverse.
    inputs = [[[2, 4, 1], [5, -3, 6]],
              [[7, 9, -8], [9, 0, 4]]]
    self._validateNthElement(inputs, dtypes.int32, 0, False,
                             [[1, -3], [-8, 0]])
    self._validateNthElement(inputs, dtypes.int64, 0, True,
                             [[4, 6], [9, 9]])

  def _testFloatLargeInput(self, input_shape):
    # [FUNCTION DOC]
    # HELPER: for testLargeInput. Generates random floats of input_shape, picks a random n
    # below the last dimension size, builds expected values by sorting along the last axis
    # (ascending for float32, descending for float64 with reverse), and validates both.
    # Uses numpy global randomness. Returns nothing.
    inputs = np.random.random_sample(input_shape)
    n = np.random.randint(input_shape[-1])
    sort_inputs = np.sort(inputs)
    expected_values = sort_inputs[..., n]
    self._validateNthElement(
        inputs, dtypes.float32, n, False, expected_values)
    expected_values = sort_inputs[..., ::-1][..., n]
    self._validateNthElement(
        inputs, dtypes.float64, n, True, expected_values)

  def _testIntLargeInput(self, input_shape):
    # [FUNCTION DOC]
    # HELPER: for testLargeInput. Same as the float variant but with random integers in
    # roughly minus a thousand to a thousand, validated as int32 (ascending) and int64
    # (reverse). Uses numpy global randomness. Returns nothing.
    inputs = np.random.randint(-1e3, 1e3, input_shape)
    n = np.random.randint(input_shape[-1])
    sort_inputs = np.sort(inputs)
    expected_values = sort_inputs[..., n]
    self._validateNthElement(
        inputs, dtypes.int32, n, False, expected_values)
    expected_values = sort_inputs[..., ::-1][..., n]
    self._validateNthElement(
        inputs, dtypes.int64, n, True, expected_values)

  def _testLargeInput(self, input_shape):
    # [FUNCTION DOC]
    # HELPER: for testLargeInput. Runs both the float and integer random-input checks for the
    # given shape.
    self._testFloatLargeInput(input_shape)
    self._testIntLargeInput(input_shape)

  def testLargeInput(self):
    # [FUNCTION DOC]
    # Runs the random large-input checks for shapes from a single element up to
    # 50 x 10 x 10 x 100 and 50 x 10000, expecting results to equal numpy sort based values.
    self._testLargeInput([1])
    self._testLargeInput([10])
    self._testLargeInput([5, 10])
    self._testLargeInput([50, 100])
    self._testLargeInput([50, 10000])
    self._testLargeInput([50, 10, 100])
    self._testLargeInput([50, 10, 10, 100])

  def _testEnumerateN(self, input_shape):
    # [FUNCTION DOC]
    # HELPER: for testEnumerateN. Generates random floats of input_shape and, for every n from
    # zero to the last dimension size minus one, validates the op in float32 (ascending) and
    # float64 (reverse) against numpy sorted values. Returns nothing.
    inputs = np.random.random_sample(input_shape)
    sort_inputs = np.sort(inputs)
    for n in range(input_shape[-1]):
      expected_values = sort_inputs[..., n]
      self._validateNthElement(
          inputs, dtypes.float32, n, False, expected_values)
      expected_values = sort_inputs[..., ::-1][..., n]
      self._validateNthElement(
          inputs, dtypes.float64, n, True, expected_values)

  def testEnumerateN(self):
    # [FUNCTION DOC]
    # Runs the exhaustive-n check on shapes of rank one to four with small dimensions,
    # expecting every n to give the right order statistic.
    self._testEnumerateN([1])
    self._testEnumerateN([10])
    self._testEnumerateN([10, 10])
    self._testEnumerateN([10, 10, 10])
    self._testEnumerateN([10, 10, 10, 10])

  def testInvalidInput(self):
    # [FUNCTION DOC]
    # Scenario: scalar (rank zero) input. Expects a value error or invalid-argument error
    # mentioning that rank at least one is required, both for a constant scalar and for a
    # placeholder fed a scalar at run time (error from the op execution) in a fresh graph.
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "at least rank 1 but is rank 0"):
      nn_ops.nth_element(5, 0)

    # Test with placeholders
    with ops.Graph().as_default():
      with self.session(use_gpu=False):
        v = array_ops.placeholder(dtype=dtypes.int32)
        with self.assertRaisesOpError("at least rank 1 but is rank 0"):
          nn_ops.nth_element(v, 0).eval(feed_dict={v: 5})

  def testInvalidN(self):
    # [FUNCTION DOC]
    # Scenario: bad n values. Negative n and non-scalar n (a one element list) must raise a
    # value error or invalid-argument error with messages about non-negative and scalar
    # requirements. Also checks a placeholder n fed minus one fails at run time.
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "non-negative but is -1"):
      nn_ops.nth_element([5], -1)
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "scalar but has rank 1"):
      nn_ops.nth_element([5, 6, 3], [1])

    # Test with placeholders
    with ops.Graph().as_default():
      with self.session(use_gpu=False):
        n = array_ops.placeholder(dtypes.int32)
        values = nn_ops.nth_element([5], n)
        with self.assertRaisesOpError("non-negative but is -1"):
          values.eval(feed_dict={n: -1})

  def testNTooLarge(self):
    # [FUNCTION DOC]
    # Scenario: two by two input with n equal to 2 (not less than the last dimension size).
    # Expects an error saying the last dimension must be greater than n, both statically and
    # when n is a placeholder fed 2 at run time.
    inputs = [[0.1, 0.2], [0.3, 0.4]]
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "must have last dimension > n = 2"):
      nn_ops.nth_element(inputs, 2)

    # Test with placeholders
    with ops.Graph().as_default():
      with self.session(use_gpu=False):
        n = array_ops.placeholder(dtypes.int32)
        values = nn_ops.nth_element(inputs, n)
        with self.assertRaisesOpError("must have last dimension > n = 2"):
          values.eval(feed_dict={n: 2})

  def testGradients(self):
    # [FUNCTION DOC]
    # Scenario: three rows of five values (including ties) with n equal to 3 and upstream
    # gradients of -1, 2, 5. Expected input gradient splits the gradient evenly among tied
    # selected values (-0.5 each in row one) and spreads over all entries for the constant row.
    # Checked with a gradient tape when running eagerly, and always with graph-mode gradients
    # in a new graph on CPU.
    x = [
        [2., -1., 1000., 3., 1000.],
        [1., 5., 2., 4., 3.],
        [2., 2., 2., 2., 2.],
    ]
    grad_ys = [[-1., 2., 5.]]
    result = [
        [0, 0, -0.5, 0, -0.5],
        [0, 0, 0, 2, 0],
        [1, 1, 1, 1, 1],
    ]
    if context.executing_eagerly():
      inputs = ops.convert_to_tensor(x)
      with backprop.GradientTape() as tape:
        tape.watch(inputs)
        values = nn_ops.nth_element(inputs, 3)
      grad = tape.gradient(values, inputs, ops.convert_to_tensor(grad_ys))
      self.assertAllClose(grad[0], result)

    # Test with tf.gradients
    with ops.Graph().as_default():
      with self.session(use_gpu=False) as sess:
        inputs = array_ops.placeholder(dtypes.float32, shape=[3, 5])
        values = nn_ops.nth_element(inputs, 3)
        grad = sess.run(
            gradients_impl.gradients(values, inputs, grad_ys=grad_ys),
            feed_dict={inputs: x})
    self.assertAllClose(grad[0], result)



if __name__ == "__main__":
  test.main()
