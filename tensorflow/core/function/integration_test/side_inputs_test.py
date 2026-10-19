# [MODULE DOC]
# Skipped (feature not implemented) tests for implicit side inputs in tf.function, i.e. functions
# that read global or enclosing-scope values without an explicit capture call. Checks that the
# traced function tracks changes to those values, captures duplicates once, and handles nested
# functions. Run as a standalone test via the unittest main entry point.
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
"""Tests for side inputs in tf.function."""

import unittest
from absl.testing import parameterized

import tensorflow as tf


# TODO(panzf): Enable tests after side inputs are supported
class SideInputsTest(parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case for implicit side inputs of tf.function. All tests are skipped
  # because the feature is not implemented. No shared fixtures.

  @parameterized.parameters(
      (1, tf.constant, 2, tf.constant),
      (1.0, tf.constant, 2.0, tf.constant),
      (1, int, 2, int),
      (1.0, float, 2.0, float),
      (1, int, 2, tf.constant),
      (1, tf.constant, 2, int))
  @unittest.skip("Feature not implemented")
  def test_direct_capture(self, val_before, type_before, val_after, type_after):
    # [FUNCTION DOC]
    # Skipped test. A function reads a global (tensor or plain number) and adds one; the global
    # is then rebound to a value of another kind. Expects traced results to match eager both
    # before and after rebinding.
    def f():
      # [FUNCTION DOC]
      # Test body: returns the global plus one.
      return x + 1

    tf_f = tf.function(f)
    x = type_before(val_before)
    self.assertEqual(f(), tf_f())
    x = type_after(val_after)
    self.assertEqual(f(), tf_f())

  @unittest.skip("Feature not implemented")
  def test_direct_capture_mutation(self):
    # [FUNCTION DOC]
    # Skipped test. A global list gets a new tensor appended after the first call. Expects the
    # traced function to reflect the mutation and match eager both times.
    def f():
      # [FUNCTION DOC]
      # Test body: returns the last element of the global list plus a constant zero.
      return glob[-1] + tf.constant(0)

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
    # Skipped test, parameterized over tensor or integer. A global mapping with mixed entries has
    # its "val" entry replaced after the first call. Expects traced and eager results to match
    # before and after the mutation.
    def f():
      # [FUNCTION DOC]
      # Test body: returns the "val" entry of the global mapping.
      return d["val"]

    tf_f = tf.function(f)
    d = {"int": 1, "tensor": tf.constant(2), "val": capture_type(3)}
    self.assertEqual(f(), tf_f())
    d["val"] = capture_type(4)
    self.assertEqual(f(), tf_f())

  @unittest.skip("Feature not implemented")
  @parameterized.parameters(tf.constant, int)
  def test_capture_with_duplicate_usage(self, capture_type):
    # [FUNCTION DOC]
    # Skipped test, parameterized over tensor or integer. A function uses the same global twice.
    # Expects results to match eager and the concrete function graph to have exactly one input
    # (captured once), both before and after the global is rebound.
    def f():
      # [FUNCTION DOC]
      # Test body: adds the global to itself.
      return x + x  # should capture x just once.

    tf_f = tf.function(f)

    x = capture_type(1)
    self.assertEqual(f(), tf_f())
    self.assertLen(tf_f.get_concrete_function().graph.inputs, 1)

    x = capture_type(2)
    self.assertEqual(f(), tf_f())
    self.assertLen(tf_f.get_concrete_function().graph.inputs, 1)

  @unittest.skip("Feature not implemented")
  def test_local_capture(self):
    # [FUNCTION DOC]
    # Skipped test. An inner function reads a variable local to the outer function while a
    # global of the same name is rebound between calls. Expects traced results to match eager,
    # i.e. the local value wins.
    def f():
      # [FUNCTION DOC]
      # Test body: defines a local constant zero and returns the result of an inner function.
      x = tf.constant(0)
      def g():
        # [FUNCTION DOC]
        # Test body: returns the enclosing local value.
        return x
      return g()

    tf_f = tf.function(f)
    x = tf.constant(100)  # pylint: disable=unused-variable
    self.assertEqual(f(), tf_f())
    x = tf.constant(200)
    self.assertEqual(f(), tf_f())

  @parameterized.parameters(
      tf.constant,
      int)
  @unittest.skip("Feature not implemented")
  def test_capture_by_nested_function(self, capture_type):
    # [FUNCTION DOC]
    # Skipped test, parameterized over tensor or integer. A nested inner function reads a global.
    # Expects traced results to match eager before and after the global is rebound.
    def f():
      # [FUNCTION DOC]
      # Test body: returns the result of calling a nested function.
      def g():
        # [FUNCTION DOC]
        # Test body: returns the global value.
        return x
      return g()

    tf_f = tf.function(f)

    x = capture_type(1)
    self.assertEqual(f(), tf_f())
    x = capture_type(2)
    self.assertEqual(f(), tf_f())

  @parameterized.parameters(tf.constant, int)
  @unittest.skip("Feature not implemented")
  def test_outer_capture_with_function_call(self, capture_type):
    # [FUNCTION DOC]
    # Skipped test, parameterized over tensor or integer. A function reading a global is called
    # from a compiled function, which is then wrapped again. Expects traced and eager results to
    # match before and after the global is rebound.
    def g():
      # [FUNCTION DOC]
      # Test body: returns the global value.
      return x

    @tf.function
    def f():
      # [FUNCTION DOC]
      # Test body: calls the global-reading function and returns its result.
      return g()
    tf_f = tf.function(f)

    x = capture_type(1)
    self.assertEqual(f(), tf_f())
    x = capture_type(2)
    self.assertEqual(f(), tf_f())

  @parameterized.parameters(tf.constant, int)
  @unittest.skip("Feature not implemented")
  def test_outer_capture_with_nested_function_call(self, capture_type):
    # [FUNCTION DOC]
    # Skipped test, parameterized over tensor or integer. A factory function defines and calls
    # an inner function reading a global; the traced function calls the factory. Expects traced
    # and eager results to match before and after the global is rebound.
    # KNOWN ISSUE: the factory already calls the inner function and returns its result, yet the
    # traced function then calls that result as if it were a function; this would fail if enabled.
    def g_factory():
      # [FUNCTION DOC]
      # Test body: defines an inner function and returns the result of calling it.
      def g():
        # [FUNCTION DOC]
        # Test body: returns the global value.
        return x
      return g()

    def f():
      # [FUNCTION DOC]
      # Test body: calls the factory and then calls whatever it returned.
      h = g_factory()
      return h()
    tf_f = tf.function(f)

    x = capture_type(1)
    self.assertEqual(f(), tf_f())
    x = capture_type(2)
    self.assertEqual(f(), tf_f())

  @parameterized.parameters(tf.constant, int)
  @unittest.skip("Feature not implemented")
  def test_capture_within_function_argument(self, capture_type):
    # [FUNCTION DOC]
    # Skipped test, parameterized over tensor or integer. A global-reading function is passed as
    # an argument to the traced function, which calls it. Expects traced and eager results to
    # match before and after the global is rebound.
    def g():
      # [FUNCTION DOC]
      # Test body: returns the global value.
      return x

    def f(h):
      # [FUNCTION DOC]
      # Test body: calls the function passed in and returns its result.
      return h()
    tf_f = tf.function(f)

    x = capture_type(1)
    self.assertEqual(f(g), tf_f(g))
    x = capture_type(2)
    self.assertEqual(f(g), tf_f(g))

  @parameterized.parameters(
      tf.constant,
      int)
  @unittest.skip("Feature not implemented")
  def test_nested_tf_function_with_capture(self, capture_type):
    # [FUNCTION DOC]
    # Skipped test, parameterized over tensor or integer. An inner compiled function reads a
    # global from within an outer compiled function. Expects the first result to be zero and
    # (as written) the result after rebinding to still be zero, and the outer concrete
    # function graph to report one capture.
    # KNOWN ISSUE: the test checks length of the graph capture method rather than a collection of
    # captures, which looks wrong; it is also skipped.
    @tf.function
    def tf_f():
      # [FUNCTION DOC]
      # Test body: outer compiled function returning the result of the inner one.
      @tf.function
      def tf_g():
        # [FUNCTION DOC]
        # Test body: inner compiled function returning the global value.
        return x
      return tf_g()

    x = capture_type(0)
    self.assertEqual(tf_f(), tf.constant(0))
    x = capture_type(1)
    self.assertEqual(tf_f(), tf.constant(0))
    # Test the outer function doesn't have any captures
    self.assertLen(tf_f.get_concrete_function().graph.capture, 1)


if __name__ == "__main__":
  unittest.main()
