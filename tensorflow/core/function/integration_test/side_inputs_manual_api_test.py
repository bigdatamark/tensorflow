# [MODULE DOC]
# Skipped (feature not implemented) tests for the manual side-input API of tf.function, where
# a function explicitly captures external values through the experimental capture call, either
# by reference or by value. Covers direct, nested, local and global capture, error cases and
# mutation of captured values. Run as a standalone test via the unittest main entry point.
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for manual API to support side inputs in tf.function."""

import unittest
from absl.testing import parameterized

import tensorflow as tf


# TODO(panzf): Enable tests after the side inputs manual API is supported
class SideInputsTest(parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case for the manual side-input capture API of tf.function. Every test is
  # currently skipped because the feature is not implemented. No shared fixtures.

  @unittest.skip("Feature not implemented")
  @parameterized.parameters(
      (1, tf.constant, 2, tf.constant),
      (1.0, tf.constant, 2.0, tf.constant),
      (1, int, 2, int),
      (1.0, float, 2.0, float),
      (1, int, 2, tf.constant),
      (1, tf.constant, 2, int))
  def test_direct_capture(self, val_before, type_before, val_after, type_after):
    # [FUNCTION DOC]
    # Skipped test. A global value, built as a tensor or plain number, is captured via the manual
    # capture call and incremented; the value is then rebound to a new value of another kind.
    # Expects the traced function to match the eager function both before and after rebinding.
    def f():
      # [FUNCTION DOC]
      # Test body: returns the manually captured global plus one.
      return tf.func.experimental.capture(lambda: x) + 1

    tf_f = tf.function(f)
    x = type_before(val_before)
    self.assertEqual(f(), tf_f())
    x = type_after(val_after)
    self.assertEqual(f(), tf_f())

  @unittest.skip("Feature not implemented")
  def test_direct_capture_mutation(self):
    # [FUNCTION DOC]
    # Skipped test. A captured global list gets a new element appended after the first call.
    # Expects the traced function to see the mutation and match the eager result both times.
    def f():
      # [FUNCTION DOC]
      # Test body: captures the global list and returns its last element plus a constant zero.
      cglob = tf.func.experimental.capture(lambda: glob)
      return cglob[-1] + tf.constant(0)

    tf_f = tf.function(f)
    glob = [tf.constant(1), tf.constant(2)]
    self.assertEqual(f(), tf_f())
    glob.append(tf.constant(3))
    self.assertEqual(f(), tf_f())

  @unittest.skip("Feature not implemented")
  @parameterized.parameters(
      tf.constant,
      int)
  def test_dict_capture_mutation_with_tensor_and_non_tensor(self, capture_type):
    # [FUNCTION DOC]
    # Skipped test, parameterized over tensor or integer values. A captured global mapping holds
    # integer, tensor and parameterized-type entries; the "val" entry is replaced after the first
    # call. Expects traced and eager results to match before and after the mutation.
    def f():
      # [FUNCTION DOC]
      # Test body: captures the global mapping and returns its "val" entry.
      cd = tf.func.experimental.capture(lambda: d)
      return cd["val"]

    tf_f = tf.function(f)
    d = {"int": 1, "tensor": tf.constant(2), "val": capture_type(3)}
    self.assertEqual(f(), tf_f())
    d["val"] = capture_type(4)
    self.assertEqual(f(), tf_f())

  @unittest.skip("Feature not implemented")
  @parameterized.parameters(tf.constant, int)
  def test_capture_with_duplicate_usage(self, capture_type):
    # [FUNCTION DOC]
    # Skipped test, parameterized over tensor or integer. A function uses the same captured
    # value twice. Expects results to match eager and the function's capture container to hold
    # exactly one entry (the value is captured only once).
    def f():
      # [FUNCTION DOC]
      # Test body: captures the global once and adds it to itself.
      cx = tf.func.experimental.capture(lambda: x)
      return cx + cx  # should capture x just once.

    tf_f = tf.function(f)

    x = capture_type(1)  # pylint: disable=unused-variable
    self.assertEqual(f(), tf_f())
    self.assertLen(tf_f._variable_creation_config._captures_container, 1)

  @unittest.skip("Feature not implemented")
  def test_local_capture(self):
    # [FUNCTION DOC]
    # Skipped test. An inner function captures a variable local to the outer function while a
    # global of the same name also exists. Expects the traced result to equal the local value
    # (zero, though the test hardcodes the expected value of 100 for the first comparison) and
    # later calls to match eager after the global changes.
    # KNOWN ISSUE: the first expected value is hardcoded to 100 while the local value is 0, so
    # the first assertion would not hold if the test were enabled.
    def f():
      # [FUNCTION DOC]
      # Test body: defines a local constant zero and returns the result of an inner function.
      x = tf.constant(0)
      def g():
        # [FUNCTION DOC]
        # Test body: captures the enclosing local value through the manual capture call.
        return tf.func.experimental.capture(lambda: x)
      return g()

    tf_f = tf.function(f)
    x = tf.constant(100)  # pylint: disable=unused-variable
    # a = f()
    a = 100
    b = tf_f()
    self.assertEqual(a, b)
    x = tf.constant(200)
    self.assertEqual(f(), tf_f())

  @unittest.skip("Feature not implemented")
  @parameterized.parameters(
      tf.constant,
      int)
  def test_capture_by_nested_function(self, capture_type):
    # [FUNCTION DOC]
    # Skipped test, parameterized over tensor or integer. A nested inner function captures a
    # global. Expects traced results to match eager before and after the global is rebound.
    def f():
      # [FUNCTION DOC]
      # Test body: returns the result of calling a nested function.
      def g():
        # [FUNCTION DOC]
        # Test body: manually captures the global and returns it.
        cx = tf.func.experimental.capture(lambda: x)
        return cx
      return g()

    tf_f = tf.function(f)
    x = capture_type(1)  # pylint: disable=unused-variable
    self.assertEqual(f(), tf_f())
    x = capture_type(2)
    self.assertEqual(f(), tf_f())

  @unittest.skip("Feature not implemented")
  @parameterized.parameters(tf.constant, int)
  def test_outer_capture_with_function_call(self, capture_type):
    # [FUNCTION DOC]
    # Skipped test, parameterized over tensor or integer. A module-level style function that
    # captures a global is called from the traced function. Expects matching results with eager
    # before and after the global is rebound.
    def g():
      # [FUNCTION DOC]
      # Test body: manually captures the global and returns it.
      cx = tf.func.experimental.capture(lambda: x)
      return cx

    def f():
      # [FUNCTION DOC]
      # Test body: calls the capturing function and returns its result.
      return g()
    tf_f = tf.function(f)

    x = capture_type(1)  # pylint: disable=unused-variable
    self.assertEqual(f(), tf_f())
    x = capture_type(2)
    self.assertEqual(f(), tf_f())

  @unittest.skip("Feature not implemented")
  @parameterized.parameters(tf.constant, int)
  def test_outer_capture_with_nested_function_call(self, capture_type):
    # [FUNCTION DOC]
    # Skipped test, parameterized over tensor or integer. A local value is captured by a
    # function created inside a factory function, which is called from the traced function.
    # Expects matching eager and traced results before and after the value is rebound.
    x = capture_type(1)  # pylint: disable=unused-variable
    def g_factory():
      # [FUNCTION DOC]
      # Test body: defines and immediately calls an inner function that captures the value.
      def g():
        # [FUNCTION DOC]
        # Test body: manually captures the value and returns it.
        cx = tf.func.experimental.capture(lambda: x)
        return cx
      return g()

    def f():
      # [FUNCTION DOC]
      # Test body: obtains the factory function and calls it.
      h = g_factory
      return h()
    tf_f = tf.function(f)

    self.assertEqual(f(), tf_f())
    x = capture_type(2)
    self.assertEqual(f(), tf_f())

  @unittest.skip("Feature not implemented")
  @parameterized.parameters(tf.constant, int)
  def test_capture_within_function_argument(self, capture_type):
    # [FUNCTION DOC]
    # Skipped test, parameterized over tensor or integer. A capturing function is passed as an
    # argument to the traced function, which calls it. Expects traced and eager results to match
    # before and after the global is rebound.
    def g():
      # [FUNCTION DOC]
      # Test body: manually captures the global and returns it.
      cx = tf.func.experimental.capture(lambda: x)
      return cx

    def f(h):
      # [FUNCTION DOC]
      # Test body: calls the function passed in and returns its result.
      return h()
    tf_f = tf.function(f)

    x = capture_type(1)  # pylint: disable=unused-variable
    self.assertEqual(f(g), tf_f(g))
    x = capture_type(2)
    self.assertEqual(f(g), tf_f(g))

  @unittest.skip("Feature not implemented")
  def test_inner_nested_tf_function_raise_error(self):
    # [FUNCTION DOC]
    # Skipped test. An inner nested compiled function uses the manual capture call. Expects
    # calling the outer function to raise a not-implemented error whose message mentions manual
    # side input usage for inner nested functions.
    @tf.function
    def tf_f():
      # [FUNCTION DOC]
      # Test body: outer compiled function that calls the inner compiled function.

      @tf.function
      def tf_g():
        # [FUNCTION DOC]
        # Test body: inner compiled function that manually captures a global and returns it.
        cx = tf.func.experimental.capture(lambda: x)
        return cx

      return tf_g()

    x = tf.constant(0)  # pylint: disable=unused-variable
    with self.assertRaisesRegex(
        NotImplementedError, "Manual side input usage for inner nested"):
      tf_f()

  @unittest.skip("Feature not implemented")
  @parameterized.parameters(
      tf.constant,
      int)
  def test_outer_nested_tf_function_with_global_capture(self, capture_type):
    # [FUNCTION DOC]
    # Skipped test, parameterized over tensor or integer. The outer compiled function manually
    # captures a global and passes it as an argument to an inner compiled function. Expects the
    # result to equal the current global value (0, then 1) after the global is rebound.
    @tf.function
    def tf_f():
      # [FUNCTION DOC]
      # Test body: captures the global and passes it to the inner compiled function.

      @tf.function
      def tf_g(x):
        # [FUNCTION DOC]
        # Test body: identity function returning its argument.
        return x

      cx = tf.func.experimental.capture(lambda: x)
      return tf_g(cx)

    x = capture_type(0)  # pylint: disable=unused-variable
    self.assertEqual(tf_f(), tf.constant(0))
    x = capture_type(1)
    self.assertEqual(tf_f(), tf.constant(1))

  @unittest.skip("Feature not implemented")
  def test_non_callable_function_raise_error(self):
    # [FUNCTION DOC]
    # Skipped test. The manual capture call is given a non-callable value (a number, then a
    # tensor) instead of a callable. Expects calling the traced function to raise a type error
    # in both cases.
    def f():
      # [FUNCTION DOC]
      # Test body: passes a non-callable global to the capture call and adds one.
      return tf.func.experimental.capture(x) + 1

    tf_f = tf.function(f)
    x = 1
    with self.assertRaises(TypeError):
      _ = tf_f()
    x = tf.constant(1)
    with self.assertRaises(TypeError):
      _ = tf_f()

  @unittest.skip("Feature not implemented")
  @parameterized.parameters(
      (1, tf.constant, 2, tf.constant),
      (1, int, 2, int))
  def test_call_by_value(self, val_before, type_before, val_after, type_after):
    # [FUNCTION DOC]
    # Skipped test. The value is captured by value (not by reference). Expects the traced
    # function to keep returning the originally captured value even after the global is rebound.
    def f():
      # [FUNCTION DOC]
      # Test body: captures the global by value and returns it.
      return tf.func.experimental.capture(lambda: x, by_ref=False)

    tf_f = tf.function(f)
    x = type_before(val_before)
    self.assertEqual(tf_f(), val_before)
    x = type_after(val_after)
    self.assertEqual(tf_f(), val_before)


if __name__ == "__main__":
  unittest.main()
