# [MODULE DOC]
# Tests for version 1 of the multiplex custom op example. Exercises the multiplex wrapper
# (elementwise select between two tensors based on a boolean condition) in both graph and eager
# modes: success for several types and ranks, plus type, size and shape mismatch errors. Importing
# the op module loads a compiled shared library. Run directly, it starts the TensorFlow test runner.
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for multiplex_1."""

import numpy as np
import tensorflow as tf

from tensorflow.examples.custom_ops_doc.multiplex_1 import multiplex_1_op
# This pylint disable is only needed for internal google users
from tensorflow.python.framework import errors_impl  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.with_eager_op_as_function
class MultiplexOpRank1Test(tf.test.TestCase):
  # [CLASS DOC]
  # Test case for the multiplex wrapper of version 1. Each test builds small constant tensors,
  # runs the op in graph and eager modes (the class is decorated to also run ops as functions),
  # and compares with expected values or error messages. No shared fixtures.

  @test_util.run_in_graph_and_eager_modes
  def test_multiplex_int(self):
    # [FUNCTION DOC]
    # Selects between two integer vectors of length five with an alternating condition and asserts
    # the result equals the reference computed with numpy. Returns nothing.
    a = tf.constant([1, 2, 3, 4, 5])
    b = tf.constant([10, 20, 30, 40, 50])
    cond = tf.constant([True, False, True, False, True], dtype=bool)
    expect = np.where(self.evaluate(cond), self.evaluate(a), self.evaluate(b))
    # expected result is [1, 20, 3, 40, 5]
    result = multiplex_1_op.multiplex(cond, a, b)
    self.assertAllEqual(result, expect)

  @test_util.run_in_graph_and_eager_modes
  def test_multiplex_float(self):
    # [FUNCTION DOC]
    # Same as the integer test but with float vectors; asserts the result equals the numpy
    # reference. Returns nothing.
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])
    b = tf.constant([10.0, 20.0, 30.0, 40.0, 50.0])
    cond = tf.constant([True, False, True, False, True], dtype=bool)
    # expected result is [1.0, 20.0, 3.0, 40.0, 5.0]
    expect = np.where(self.evaluate(cond), self.evaluate(a), self.evaluate(b))
    result = multiplex_1_op.multiplex(cond, a, b)
    self.assertAllEqual(result, expect)

  @test_util.run_in_graph_and_eager_modes
  def test_multiplex_bad_types(self):
    # [FUNCTION DOC]
    # Passes a float tensor for a and an integer tensor for b and asserts evaluation raises an
    # invalid-argument error (eager mode) or a type error (graph mode) whose message reports the
    # type mismatch for the op. Returns nothing.
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])  # float
    b = tf.constant([10, 20, 30, 40, 50])  # int32
    cond = tf.constant([True, False, True, False, True], dtype=bool)
    with self.assertRaisesRegex(
        (errors_impl.InvalidArgumentError, TypeError),
        # Eager mode raises InvalidArgumentError with the following message
        r'(cannot compute Examples1>MultiplexDense as input #2\(zero-based\) '
        r'was expected to be a float tensor but is a int32 tensor '
        r'\[Op:Examples1>MultiplexDense\]'
        r')|('
        # Graph mode raises TypeError with the following message
        r"Input 'b_values' of 'Examples1>MultiplexDense' Op has type int32 that "
        r"does not match type float32 of argument 'a_values'.)"):
      self.evaluate(multiplex_1_op.multiplex(cond, a, b))

  @test_util.run_in_graph_and_eager_modes
  def test_multiplex_bad_size(self):
    # [FUNCTION DOC]
    # Passes vectors of length five and two and asserts evaluation raises an invalid-argument
    # error (eager) or value error (graph) reporting that the two shapes must be equal.
    # Returns nothing.
    a = tf.constant([1, 2, 3, 4, 5])  # longer than b
    b = tf.constant([10, 20])  # shorter than a
    cond = tf.constant([True, False, True, False, True], dtype=bool)
    with self.assertRaisesRegex(
        (errors_impl.InvalidArgumentError, ValueError),
        # Eager mode raises InvalidArgumentError with the following message
        r'(?s)(a_values and b_values must have the same shape. '
        r'a_values shape: \[5\] b_values shape: \[2\].* '
        r'\[Op:Examples1>MultiplexDense\]'
        r')|('
        # Graph mode raises ValueError with the following message
        r'Dimension 0 in both shapes must be equal, but are 5 and 2\. '
        r'Shapes are \[5\] and \[2\]\.)'):
      self.evaluate(multiplex_1_op.multiplex(cond, a, b))

  @test_util.run_in_graph_and_eager_modes
  def test_multiplex_2d(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the inline comment about the expected result lists a 2 by 2 value, but the
    # inputs are 2 by 3; the assertion itself uses the numpy reference so it is unaffected.
    # Selects between two 2 by 3 matrices with a 2 by 3 condition and asserts the result equals
    # the numpy reference. Returns nothing.
    a = tf.constant([[1, 2, 3], [4, 5, 6]])
    b = tf.constant([[10, 20, 30], [40, 50, 60]])
    cond = tf.constant([[True, False, True], [False, True, False]], dtype=bool)
    expect = np.where(self.evaluate(cond), self.evaluate(a), self.evaluate(b))
    # expected result is [[1, 20], [3, 40]]
    result = multiplex_1_op.multiplex(cond, a, b)
    self.assertAllEqual(result, expect)

  @test_util.run_in_graph_and_eager_modes
  def test_multiplex_bad_shape(self):
    # [FUNCTION DOC]
    # Passes a 2 by 3 and a 3 by 2 tensor and asserts evaluation raises an invalid-argument error
    # (eager) or value error (graph) reporting that the shapes must match. Returns nothing.
    a = tf.constant([[1, 2, 3], [4, 5, 6]])  # shape (2,3)
    b = tf.constant([[10, 20], [30, 40], [50, 60]])  # shape (3,2)
    cond = tf.constant([[True, False, True], [False, True, False]], dtype=bool)
    with self.assertRaisesRegex(
        (errors_impl.InvalidArgumentError, ValueError),
        # Eager mode raises InvalidArgumentError with the following message
        r'(a_values and b_values must have the same shape.'
        r' a_values shape: \[2,3\] b_values shape: \[3,2\]'
        r')|('
        # Graph mode raises ValueError with the following message
        r'Dimension 0 in both shapes must be equal, '
        r'but are 2 and 3\. Shapes are \[2,3\] and \[3,2\])\.'):
      self.evaluate(multiplex_1_op.multiplex(cond, a, b))


if __name__ == '__main__':
  tf.test.main()
