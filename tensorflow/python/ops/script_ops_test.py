# [MODULE DOC]
# Unit tests for script_ops: covers the decorator forms and argument validation of the
# py_function and numpy_function wrappers, stateless versus stateful numpy functions inside
# tf.function, variable arguments to py_function, and rejection of an invalid token by the raw op.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for script operations."""
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gen_script_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops.script_ops import numpy_function
from tensorflow.python.platform import test


class ArgsAndDecoratorsTest(test.TestCase):
  # [CLASS DOC]
  # Tests for using numpy_function and eager_py_func as decorators and for the argument
  # validation that rejects misuse (missing output type, wrongly supplied inputs). No shared
  # fixtures.

  @test_util.run_in_graph_and_eager_modes
  def test_decorator(self):
    # [FUNCTION DOC]
    # Scenario: decorate an addition function with the numpy function decorator (output type int32),
    # call it with 1 and 2. Expects the result to equal the int32 constant 3 and the function to
    # have run exactly once. Runs in both graph and eager modes.
    count = 0

    @script_ops.numpy_function(Tout=dtypes.int32)
    def plus(a, b):
      # [FUNCTION DOC]
      # Test helper: increments the enclosing call counter and returns the sum of its arguments.
      nonlocal count
      count += 1
      return a + b

    actual_result = plus(1, 2)
    expect_result = constant_op.constant(3, dtypes.int32)
    self.assertAllEqual(actual_result, expect_result)
    self.assertEqual(count, 1)

  @test_util.run_in_graph_and_eager_modes
  def test_inline_decorator(self):
    # [FUNCTION DOC]
    # Scenario: apply the eager py_function decorator (given only the output type) to an existing
    # addition function by calling it, then call the wrapped function with 1 and 2. Expects the
    # int32 constant 3 and exactly one execution of the function. Graph and eager modes.
    count = 0

    def plus(a, b):
      # [FUNCTION DOC]
      # Test helper: increments the enclosing call counter and returns the sum of its arguments.
      nonlocal count
      count += 1
      return a + b

    py_plus = script_ops.eager_py_func(Tout=dtypes.int32)(plus)

    actual_result = py_plus(1, 2)
    expect_result = constant_op.constant(3, dtypes.int32)
    self.assertAllEqual(actual_result, expect_result)
    self.assertEqual(count, 1)

  def test_bad_args(self):
    # [FUNCTION DOC]
    # Scenario: misuse the eager py_function decorator in several ways. Expects a type error
    # matching: missing output type (bare decorator); output type passed positionally instead of by
    # name; inputs given while decorating; inputs given with no function; and a function given
    # without inputs (both direct call and decorator form, which then reports missing inputs).
    def minus(a, b):
      # [FUNCTION DOC]
      # Test helper: returns the difference of its two arguments; used as a valid function.
      return a - b

    # No Tout
    with self.assertRaisesRegex(TypeError, r"Missing.*Tout"):

      @script_ops.eager_py_func
      def plus1(a, b):
        # [FUNCTION DOC]
        # Test helper: never successfully decorated (decoration raises); returns the sum.
        return a + b

    # Forgot to set arg-name
    with self.assertRaisesRegex(TypeError, r"Missing.*Tout"):

      @script_ops.eager_py_func(dtypes.int32)
      def plus2(a, b):
        # [FUNCTION DOC]
        # Test helper: never successfully decorated (decoration raises); returns the sum.
        return a + b

    # inputs in the decoration doesn't make sense
    with self.assertRaisesRegex(TypeError, r"Don't.*inp.*decorator"):

      @script_ops.eager_py_func(inp=[], Tout=dtypes.int32)
      def plus3(a, b):
        # [FUNCTION DOC]
        # Test helper: never successfully decorated (decoration raises); returns the sum.
        return a + b

    # Forgot to set the function is still a type-error.
    with self.assertRaisesRegex(TypeError, r"Don't.*inp.*decorator"):
      script_ops.eager_py_func(inp=[], Tout=dtypes.int32)

    # Still an error to call without inp
    with self.assertRaisesRegex(TypeError, r"Missing.*inp"):
      script_ops.eager_py_func(minus, Tout=dtypes.int32)

    # Why would you do this?
    with self.assertRaisesRegex(TypeError, r"Missing.*inp"):

      @script_ops.eager_py_func(func=minus, Tout=dtypes.int32)
      def plus4(a, b):
        # [FUNCTION DOC]
        # Test helper: never successfully decorated (decoration raises); returns the sum.
        return a + b

  def test_decorator_pass_through_extra_args(self):
    # [FUNCTION DOC]
    # Scenario: build a decorator from the internal argument checker using a fake op builder, an
    # output type and an extra keyword option. Decorate an addition function, check the option
    # has not been seen yet, call the function with 1 and 2. Expects result 3 and the fake builder
    # to have received the extra option value.
    got_extra = None

    def dummy_script_op(func, inp, extra=None, **kwargs):
      # [FUNCTION DOC]
      # Test helper: fake op builder. Records the extra option in the enclosing variable, ignores
      # other keywords, and returns the function applied to the inputs.
      del kwargs
      nonlocal got_extra
      got_extra = extra
      return func(*inp)

    decorator = script_ops._check_args_and_maybe_make_decorator(
        dummy_script_op, "dummy", Tout=dtypes.int32, extra="extra"
    )

    @decorator
    def plus(a, b):
      # [FUNCTION DOC]
      # Test helper: returns the sum of its arguments.
      return a + b

    self.assertIsNone(got_extra)
    self.assertEqual(plus(1, 2), 3)
    self.assertEqual(got_extra, "extra")


class NumpyFunctionTest(test.TestCase):
  # [CLASS DOC]
  # Tests for numpy_function: plain python arguments, and stateless versus stateful behavior when
  # called inside nested tf.function graphs. No shared fixtures.

  @test_util.run_in_graph_and_eager_modes
  def test_numpy_arguments(self):
    # [FUNCTION DOC]
    # Scenario: call numpy_function directly on an addition function with plain python numbers 1
    # and 2 and int32 output. Expects the int32 constant 3. Graph and eager modes.

    def plus(a, b):
      # [FUNCTION DOC]
      # Test helper: returns the sum of its arguments.
      return a + b

    actual_result = script_ops.numpy_function(plus, [1, 2], dtypes.int32)
    expect_result = constant_op.constant(3, dtypes.int32)
    self.assertAllEqual(actual_result, expect_result)

  def test_stateless(self):
    # [FUNCTION DOC]
    # Scenario: a counting addition function wrapped as a stateless numpy function inside a
    # tf.function, which is called twice with identical inputs from an outer tf.function. Runs it
    # with constants 1 and 2. Expects the call count to be 1 or 2, since the runtime may
    # deduplicate identical stateless calls.
    call_count = 0

    def plus(a, b):
      # [FUNCTION DOC]
      # Test helper: increments the enclosing call counter and returns the sum.
      nonlocal call_count
      call_count += 1
      return a + b

    @def_function.function
    def numpy_func_stateless(a, b):
      # [FUNCTION DOC]
      # Test helper graph function: wraps the addition function as a stateless numpy function with
      # int32 output and returns its result.
      return numpy_function(plus, [a, b], dtypes.int32, stateful=False)

    @def_function.function
    def func_stateless(a, b):
      # [FUNCTION DOC]
      # Test helper graph function: calls the stateless wrapper twice with the same inputs and
      # returns the sum of both results.
      sum1 = numpy_func_stateless(a, b)
      sum2 = numpy_func_stateless(a, b)
      return sum1 + sum2

    self.evaluate(func_stateless(
        constant_op.constant(1),
        constant_op.constant(2),
    ))

    self.assertIn(call_count, (1, 2))  # as stateless, func may be deduplicated

  def test_stateful(self):
    # [FUNCTION DOC]
    # Scenario: same as the stateless test but the numpy function is stateful. Calls the outer
    # tf.function with constants 1 and 2. Expects the counting function to have run exactly twice,
    # because stateful calls must not be deduplicated.
    call_count = 0

    def plus(a, b):
      # [FUNCTION DOC]
      # Test helper: increments the enclosing call counter and returns the sum.
      nonlocal call_count
      call_count += 1
      return a + b

    @def_function.function
    def numpy_func_stateful(a, b):
      # [FUNCTION DOC]
      # Test helper graph function: wraps the addition function as a stateful numpy function with
      # int32 output and returns its result.
      return numpy_function(plus, [a, b], dtypes.int32, stateful=True)

    @def_function.function
    def func_stateful(a, b):
      # [FUNCTION DOC]
      # Test helper graph function: calls the stateful wrapper twice with the same inputs and
      # returns the sum of both results.
      sum1 = numpy_func_stateful(a, b)
      sum2 = numpy_func_stateful(a, b)
      return sum1 + sum2

    self.evaluate(func_stateful(
        constant_op.constant(1),
        constant_op.constant(2),
    ))

    self.assertEqual(call_count,
                     2)  # as stateful, func is guaranteed to execute twice


class PyFunctionTest(test.TestCase):
  # [CLASS DOC]
  # Tests for eager_py_func (public py_function): variable inputs and invalid token handling in
  # the raw py_func op. No shared fixtures.
  @test_util.run_in_graph_and_eager_modes
  def test_variable_arguments(self):
    # [FUNCTION DOC]
    # Scenario: create and initialize a resource variable holding 1, then call py_function on an
    # addition function with the variable and the number 2 as inputs, int32 output. Expects the
    # int32 constant 3 (variables are converted to tensors). Graph and eager modes.

    def plus(a, b):
      # [FUNCTION DOC]
      # Test helper: returns the sum of its arguments.
      return a + b

    v1 = resource_variable_ops.ResourceVariable(1)
    self.evaluate(v1.initializer)

    actual_result = script_ops.eager_py_func(plus, [v1, 2], dtypes.int32)
    expect_result = constant_op.constant(3, dtypes.int32)
    self.assertAllEqual(actual_result, expect_result)

  @test_util.run_in_graph_and_eager_modes
  def test_fail_on_non_utf8_token(self):
    # [FUNCTION DOC]
    # Scenario: run the raw py_func op with an int32 vector input and a token that is not valid
    # utf-8 (a single byte 0xb0). Expects either an internal error or a unicode decode error,
    # rather than a crash. Graph and eager modes.
    value = constant_op.constant(value=[1, 2])
    token = b"\xb0"
    data_type = [dtypes.int32]
    with self.assertRaises((errors.InternalError, UnicodeDecodeError)):
      self.evaluate(
          gen_script_ops.py_func(input=[value], token=token, Tout=data_type))


if __name__ == "__main__":
  test.main()
