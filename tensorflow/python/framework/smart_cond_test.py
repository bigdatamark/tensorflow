# [MODULE DOC]
# Unit tests for the smart_cond module: smart_cond, smart_case and smart_constant_value.
# Verifies static branch selection for Python bool and constant predicates, dynamic fallback to
# the graph conditional for placeholder predicates, argument validation, and constant extraction.
# Run as a standard test script; no import-time side effects beyond imports.
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

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest


def raise_exception():
  # [FUNCTION DOC]
  # HELPER: Test helper used as a branch callable that must never be executed.
  # Takes no inputs and always raises a runtime error saying it was not expected to be called,
  # so a test fails if a branch that should be statically skipped is invoked.
  raise RuntimeError("did not expect to be called")


class SmartCondTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for smart_cond covering constant true/false predicates, unknown (placeholder)
  # predicates, statically evaluable predicates and missing callable arguments. Each test builds
  # its own graph and session.

  def testTrue(self):
    # [FUNCTION DOC]
    # Scenario: predicate is the Python value true. Calls smart_cond with a multiply branch for
    # each outcome using two constants. Expects the true branch result (2 times 16 equals 32).
    with ops.Graph().as_default():
      with session.Session():
        x = constant_op.constant(2)
        y = constant_op.constant(5)
        z = smart_cond.smart_cond(True, lambda: math_ops.multiply(x, 16),
                                  lambda: math_ops.multiply(y, 5))
        self.assertEqual(self.evaluate(z), 32)

  def testFalse(self):
    # [FUNCTION DOC]
    # Scenario: predicate is the Python value false. Expects the false branch result
    # (3 times 3 equals 9) to be returned after evaluating in a session.
    with ops.Graph().as_default():
      with session.Session():
        x = constant_op.constant(4)
        y = constant_op.constant(3)
        z = smart_cond.smart_cond(False, lambda: math_ops.multiply(x, 16),
                                  lambda: math_ops.multiply(y, 3))
        self.assertEqual(self.evaluate(z), 9)

  def testUnknown(self):
    # [FUNCTION DOC]
    # Scenario: predicate depends on an integer placeholder (value unknown at build time), so a
    # dynamic conditional is built. Feeding a positive value must yield 1; a negative value 2.
    with ops.Graph().as_default():
      with session.Session():
        x = array_ops.placeholder(dtype=dtypes.int32)
        y = smart_cond.smart_cond(x > 0, lambda: constant_op.constant(1),
                                  lambda: constant_op.constant(2))
        self.assertEqual(y.eval(feed_dict={x: 1}), 1)
        self.assertEqual(y.eval(feed_dict={x: -1}), 2)

  def testEval(self):
    # [FUNCTION DOC]
    # Scenario: the predicate (product of two constants greater than zero) is computable at
    # graph construction time, so the false branch (which raises if called) must never run.
    # Asserts that evaluating the result with a feed gives 1.
    with ops.Graph().as_default():
      with session.Session():
        x = constant_op.constant(1)
        y = constant_op.constant(2)
        # x * y > 0 can be evaluated at graph construction time, so the false
        # branch shouldn't be evaluated at all.
        z = smart_cond.smart_cond(x * y > 0, lambda: constant_op.constant(1),
                                  raise_exception)
        self.assertEqual(z.eval(feed_dict={x: 1}), 1)

  def testPlaceholderWithDefault(self):
    # [FUNCTION DOC]
    # Scenario: predicate derives from a placeholder-with-default (default 1). Without feeding,
    # the result must be 1; feeding the placeholder with minus one must give 2, showing the
    # predicate is treated as dynamic rather than folded to its default.
    with ops.Graph().as_default():
      with session.Session():
        x = array_ops.placeholder_with_default(1, shape=())
        y = smart_cond.smart_cond(x > 0, lambda: constant_op.constant(1),
                                  lambda: constant_op.constant(2))
        self.assertEqual(self.evaluate(y), 1)
        self.assertEqual(y.eval(feed_dict={x: -1}), 2)

  def testMissingArg1(self):
    # [FUNCTION DOC]
    # Scenario: smart_cond called with a true predicate but no true callable. Asserts that a
    # type error is raised.
    with ops.Graph().as_default():
      with session.Session():
        x = constant_op.constant(1)
        with self.assertRaises(TypeError):
          smart_cond.smart_cond(True, false_fn=lambda: x)

  def testMissingArg2(self):
    # [FUNCTION DOC]
    # Scenario: smart_cond called with a true predicate and a true callable but no false
    # callable. Asserts that a type error is raised even though the false branch is not needed.
    with ops.Graph().as_default():
      with session.Session():
        x = constant_op.constant(1)
        with self.assertRaises(TypeError):
          smart_cond.smart_cond(True, lambda: x)


class SmartCaseTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for smart_case with constant, false and mixed constant/dynamic predicates, in both
  # exclusive and non-exclusive modes. Tests run in legacy graph mode.

  @test_util.run_deprecated_v1
  def testTrue(self):
    # [FUNCTION DOC]
    # Scenario: first predicate is the constant true and a later dynamic predicate and the
    # default both raise if used. Builds exclusive and non-exclusive cases and expects both to
    # evaluate to 1 without feeding anything, since the case is resolved statically.
    x = array_ops.placeholder(dtype=dtypes.int32, shape=[])
    conditions = [(True, lambda: constant_op.constant(1)),
                  (x == 0, raise_exception)]
    y = smart_cond.smart_case(conditions, default=raise_exception,
                              exclusive=False)
    z = smart_cond.smart_case(conditions, default=raise_exception,
                              exclusive=True)
    with session.Session() as sess:
      # No feed_dict necessary
      self.assertEqual(self.evaluate(y), 1)
      self.assertEqual(self.evaluate(z), 1)

  @test_util.run_deprecated_v1
  def testFalse(self):
    # [FUNCTION DOC]
    # Scenario: the only predicate is the constant false (its branch raises if called), so the
    # default branch returning 1 is selected. Checks exclusive and non-exclusive results equal 1.
    conditions = [(False, raise_exception)]
    y = smart_cond.smart_case(conditions,
                              default=lambda: constant_op.constant(1),
                              exclusive=False)
    z = smart_cond.smart_case(conditions,
                              default=lambda: constant_op.constant(1),
                              exclusive=True)
    with session.Session() as sess:
      self.assertEqual(self.evaluate(y), 1)
      self.assertEqual(self.evaluate(z), 1)

  @test_util.run_deprecated_v1
  def testMix(self):
    # [FUNCTION DOC]
    # Scenario: a mixture of one dynamic predicate (placeholder greater than one), a constant
    # evaluable false predicate, a literal false and a literal true, with a raising default.
    # Expects 1 when the placeholder is 2 and 3 when it is 0, via a session run with feeds.
    x = array_ops.placeholder(dtype=dtypes.int32, shape=[])
    y = constant_op.constant(10)
    conditions = [(x > 1, lambda: constant_op.constant(1)),
                  (y < 1, raise_exception),
                  (False, raise_exception),
                  (True, lambda: constant_op.constant(3))]
    z = smart_cond.smart_case(conditions, default=raise_exception)
    with session.Session() as sess:
      self.assertEqual(sess.run(z, feed_dict={x: 2}), 1)
      self.assertEqual(sess.run(z, feed_dict={x: 0}), 3)


class SmartConstantValueTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for smart_constant_value; currently a single regression test for constant evaluation
  # of a graph conditional output.

  # TODO(skyewm): this is essentially a regression test for
  # TF_TryEvaluateConstant, and is not really a valid smart_constant_value test
  # (smart_constant_value is only supposed to return bools). Move the
  # TF_TryEvaluateConstant call to tensor_util.constant_value and make this a
  # constant_value test instead.
  def testCond(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: per the TODO above, this is really a regression test for the constant
    # evaluation fallback, not a valid test of smart_constant_value (which should only return
    # booleans).
    # Scenario: builds a graph conditional on a placeholder-with-default true predicate and
    # asserts smart_constant_value of its output is nothing (null), i.e. not statically known.
    with ops.Graph().as_default():
      pred = array_ops.placeholder_with_default(True, shape=())
      x = cond.cond(pred,
                    lambda: constant_op.constant(1),
                    lambda: constant_op.constant(2))
      self.assertIsNone(smart_cond.smart_constant_value(x))


if __name__ == "__main__":
  googletest.main()
