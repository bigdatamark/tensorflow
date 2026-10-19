# [MODULE DOC]
# Unit tests for the assert_stmt operator in the exceptions operators module, covering tensor
# conditions (passing, failing, multiple printed values) and Python conditions (message
# callable not evaluated on success, evaluated on failure). No import-time side effects.
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
"""Tests for exceptions module."""

from tensorflow.python.autograph.operators import exceptions
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


class ExceptionsTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the assert operator; no shared fixtures.

  def test_assert_tf_untriggered(self):
    # [FUNCTION DOC]
    # Stages an assert with a true tensor condition and evaluates it in a session; asserts that
    # evaluation succeeds without error.
    with self.cached_session() as sess:
      t = exceptions.assert_stmt(
          constant_op.constant(True), lambda: constant_op.constant('ignored'))
      self.evaluate(t)

  @test_util.run_deprecated_v1
  def test_assert_tf_triggered(self):
    # [FUNCTION DOC]
    # Runs in graph (v1) mode. Stages an assert with a false tensor condition and a message
    # tensor; asserts evaluating it raises an invalid-argument error containing the message.
    with self.cached_session() as sess:
      t = exceptions.assert_stmt(
          constant_op.constant(False),
          lambda: constant_op.constant('test message'))

      with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                  'test message'):
        self.evaluate(t)

  @test_util.run_deprecated_v1
  def test_assert_tf_multiple_printed_values(self):
    # [FUNCTION DOC]
    # Runs in graph (v1) mode. Stages a failing assert whose message callable returns a list of
    # two tensors; asserts the invalid-argument error text contains both messages in order.
    two_tensors = [
        constant_op.constant('test message'),
        constant_op.constant('another message')
    ]
    with self.cached_session() as sess:
      t = exceptions.assert_stmt(
          constant_op.constant(False), lambda: two_tensors)

      with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                  'test message.*another message'):
        self.evaluate(t)

  def test_assert_python_untriggered(self):
    # [FUNCTION DOC]
    # Runs a Python assert with a true condition and asserts the message callable (which records
    # a side effect) was never called.
    side_effect_trace = []

    def expression_with_side_effects():
      # [FUNCTION DOC]
      # HELPER: message callable that records a side effect in the enclosing list, then returns
      # a fixed message string.
      side_effect_trace.append(object())
      return 'test message'

    exceptions.assert_stmt(True, expression_with_side_effects)

    self.assertListEqual(side_effect_trace, [])

  def test_assert_python_triggered(self):
    # [FUNCTION DOC]
    # Skips (returns early) when not in debug mode. Runs a Python assert with a false condition
    # and asserts an assertion error with the message is raised and that the message callable
    # was called exactly once (recorded side effect).
    if not __debug__:
      # Python assertions only be tested when in debug mode.
      return

    side_effect_trace = []
    tracer = object()

    def expression_with_side_effects():
      # [FUNCTION DOC]
      # HELPER: message callable that records a side effect in the enclosing list, then returns
      # a fixed message string.
      side_effect_trace.append(tracer)
      return 'test message'

    with self.assertRaisesRegex(AssertionError, 'test message'):
      exceptions.assert_stmt(False, expression_with_side_effects)
    self.assertListEqual(side_effect_trace, [tracer])


if __name__ == '__main__':
  test.main()
