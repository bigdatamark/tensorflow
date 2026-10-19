# [MODULE DOC]
# Unit tests for the numerics ops module: the verify-all-finite op and the graph-wide
# add-check-numerics-ops utility (a graph-mode-only API). It defines two test classes and
# runs them through the standard test main entry point when executed as a script. No
# import-time side effects beyond imports. Not a helper module.
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
"""Tests for tensorflow.ops.numerics."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import numerics
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import test


class VerifyTensorAllFiniteTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the verify-tensor-all-finite op: it must pass finite tensors through
  # unchanged and raise an op error carrying the user message for NaN or infinite values.
  # Holds no shared fixtures; each test builds its own random float input.

  def testVerifyTensorAllFiniteSucceeds(self):
    # [FUNCTION DOC]
    # Builds a 5 by 4 random finite float32 tensor, runs the verification op with a message
    # (preferring a GPU device when available) and asserts the evaluated result is numerically
    # close to the original input. Returns nothing; fails on assertion mismatch.
    x_shape = [5, 4]
    x = np.random.random_sample(x_shape).astype(np.float32)
    with test_util.use_gpu():
      t = constant_op.constant(x, shape=x_shape, dtype=dtypes.float32)
      t_verified = numerics.verify_tensor_all_finite(t,
                                                     "Input is not a number.")
      self.assertAllClose(x, self.evaluate(t_verified))

  def testVerifyTensorAllFiniteFails(self):
    # [FUNCTION DOC]
    # Puts a NaN and then an infinity into the first row of a random 5 by 4 float32 input and,
    # for each, runs the verification op with a custom message. Asserts that evaluation raises
    # an op error whose text matches that message. Returns nothing.
    x_shape = [5, 4]
    x = np.random.random_sample(x_shape).astype(np.float32)
    my_msg = "Input is not a number."

    # Test NaN.
    x[0] = np.nan
    with test_util.use_gpu():
      with self.assertRaisesOpError(my_msg):
        t = constant_op.constant(x, shape=x_shape, dtype=dtypes.float32)
        t_verified = numerics.verify_tensor_all_finite(t, my_msg)
        self.evaluate(t_verified)

    # Test Inf.
    x[0] = np.inf
    with test_util.use_gpu():
      with self.assertRaisesOpError(my_msg):
        t = constant_op.constant(x, shape=x_shape, dtype=dtypes.float32)
        t_verified = numerics.verify_tensor_all_finite(t, my_msg)
        self.evaluate(t_verified)


@test_util.run_v1_only("add_check_numerics_op() is meant to be a v1-only API")
class NumericsTest(test.TestCase):
  # [CLASS DOC]
  # Graph-mode-only test case (skipped outside v1 behavior) for the add-check-numerics-ops
  # utility, which attaches numeric checks to all floating point tensors in the default graph.
  # Covers detection of infinity, NaN, both, pass-through of the check op, and the error raised
  # when the graph contains control flow. Each test uses its own fresh graph or the default one.

  def testInf(self):
    # [FUNCTION DOC]
    # In a fresh graph divides one by zero, adds the check-numerics ops, makes the result depend
    # on the check, and asserts that evaluating raises an op error mentioning infinity.
    with self.session(graph=ops.Graph()):
      t1 = constant_op.constant(1.0)
      t2 = constant_op.constant(0.0)
      a = math_ops.div(t1, t2)
      check = numerics.add_check_numerics_ops()
      a = control_flow_ops.with_dependencies([check], a)
      with self.assertRaisesOpError("Inf"):
        self.evaluate(a)

  def testNaN(self):
    # [FUNCTION DOC]
    # In a fresh graph divides zero by zero, adds the check-numerics ops, makes the result depend
    # on the check, and asserts that evaluating raises an op error mentioning NaN.
    with self.session(graph=ops.Graph()):
      t1 = constant_op.constant(0.0)
      t2 = constant_op.constant(0.0)
      a = math_ops.div(t1, t2)
      check = numerics.add_check_numerics_ops()
      a = control_flow_ops.with_dependencies([check], a)
      with self.assertRaisesOpError("NaN"):
        self.evaluate(a)

  def testBoth(self):
    # [FUNCTION DOC]
    # In a fresh graph divides the vector one, zero by zero, zero (yielding an infinity and a
    # NaN), adds the check-numerics ops and asserts that evaluation raises an op error whose
    # message reports both infinity and NaN.
    with self.session(graph=ops.Graph()):
      t1 = constant_op.constant([1.0, 0.0])
      t2 = constant_op.constant([0.0, 0.0])
      a = math_ops.div(t1, t2)
      check = numerics.add_check_numerics_ops()
      a = control_flow_ops.with_dependencies([check], a)
      with self.assertRaisesOpError("Inf and NaN"):
        self.evaluate(a)

  def testPassThrough(self):
    # [FUNCTION DOC]
    # Applies the plain check-numerics op with a message to a finite 2 by 3 constant in a fresh
    # graph. Asserts the evaluated value equals the input and the static shape is preserved.
    with self.session(graph=ops.Graph()):
      t1 = constant_op.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
      checked = array_ops.check_numerics(t1, message="pass through test")
      value = self.evaluate(checked)
      self.assertAllEqual(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), value)
      self.assertEqual([2, 3], checked.get_shape())

  def testControlFlowCond(self):
    # [FUNCTION DOC]
    # Builds a conditional on a boolean placeholder in the default graph, then asserts that
    # adding check-numerics ops raises a value error stating the utility is incompatible with
    # control flow such as conditionals and while loops.
    predicate = array_ops.placeholder(dtypes.bool, shape=[])
    _ = cond.cond(predicate,
                  lambda: constant_op.constant([37.]),
                  lambda: constant_op.constant([42.]))
    with self.assertRaisesRegex(
        ValueError, r"`tf\.add_check_numerics_ops\(\) is not compatible with "
        r"TensorFlow control flow operations such as `tf\.cond\(\)` "
        r"or `tf.while_loop\(\)`\."):
      numerics.add_check_numerics_ops()

  def testControlFlowWhile(self):
    # [FUNCTION DOC]
    # Builds a while loop driven by a boolean placeholder in the default graph, then asserts
    # that adding check-numerics ops raises a value error stating the utility is incompatible
    # with control flow such as conditionals and while loops.
    predicate = array_ops.placeholder(dtypes.bool, shape=[])
    _ = while_loop.while_loop(lambda _: predicate,
                              lambda _: constant_op.constant([37.]),
                              [constant_op.constant([42.])])
    with self.assertRaisesRegex(
        ValueError, r"`tf\.add_check_numerics_ops\(\) is not compatible with "
        r"TensorFlow control flow operations such as `tf\.cond\(\)` "
        r"or `tf.while_loop\(\)`\."):
      numerics.add_check_numerics_ops()


if __name__ == "__main__":
  test.main()
