# -*- coding: utf-8 -*-
# [MODULE DOC]
# Test module for the script_ops py_func and eager_py_func operations, which let a graph or eager
# program call back into arbitrary Python code. It defines a shared base test class with an
# exception-translation checker, tests for the graph-style py_func (data types, strings, numpy
# conversion, statefulness, gradients, parallel execution, error mapping, reference lifetime), tests
# that are shared by both flavours (cleanup of the global registry of Python functions), and tests
# for eager_py_func (single and multiple outputs, gradients via tape and graph, device placement,
# ragged tensor arguments and results, bad arguments). It also defines two tiny module-level helper
# functions. No flags or registrations happen at import time; the tests run through the standard
# test main entry at the bottom.
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
"""Tests for py_func op."""

import gc
import queue
import re

import numpy as np

from tensorflow.python.client import session as session_lib
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import batch_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test


def np_func(x, y):
  # [FUNCTION DOC]
  # HELPER: small numpy-based callable used as the Python body in several py_func tests.
  # Takes two array-like inputs x and y and returns the element-wise hyperbolic sine of x plus the
  # hyperbolic cosine of y, as a numpy value. Pure function, no side effects, no exceptions of its own.
  return np.sinh(x) + np.cosh(y)


def matmul(x, y):
  # [FUNCTION DOC]
  # HELPER: small callable used as the Python body for eager_py_func tests.
  # Takes two tensors x and y and returns their matrix product computed with the TensorFlow matmul op.
  # Raises whatever matmul raises on incompatible shapes or dtypes; no side effects.
  return math_ops.matmul(x, y)


class PyFuncTestBase(test.TestCase):
  # [CLASS DOC]
  # HELPER: shared base class for the py_func test classes in this module.
  # Derives from the standard TensorFlow test case and contributes one reusable verification method
  # that checks how Python exceptions raised inside a registered function are translated into
  # TensorFlow error types and whether the Python stack trace is preserved in the message. It has no
  # tests of its own and holds no state.

  def verifyExceptionHandling(self, py_exp, tf_exp, eager=False):
    # [FUNCTION DOC]
    # HELPER: shared assertion helper for the exception-handling tests of py_func and eager_py_func.
    # Inputs: py_exp is a Python exception class to raise from inside the callback; tf_exp is the
    # TensorFlow error class expected to surface; eager (default false) selects eager_py_func instead of
    # py_func. Builds a callback that raises py_exp with the message "blah" through two nested calls and
    # builds a predicate that searches the error text for the message, the names of both nested
    # functions in a stack trace, and the message again, in that order. In eager flavour while executing
    # eagerly, the op is created inside an expectation-of-error context and the method returns right away.
    # Otherwise the op is created (graph mode) and evaluated inside the expectation context. Fails the
    # test if no matching error of type tf_exp is raised. Returns nothing.

    def inner_exception():
      # [FUNCTION DOC]
      # HELPER: innermost callback used by the exception-handling check.
      # Takes no arguments and always raises the exception class chosen by the enclosing method, with the
      # message "blah". Exists so the stack trace contains a distinct inner frame name.
      raise py_exp("blah")  # pylint: disable=not-callable

    def raise_exception():
      # [FUNCTION DOC]
      # HELPER: outer callback registered with the py_func op by the exception-handling check.
      # Takes no arguments, calls the inner exception raiser and thereby propagates its exception; it
      # returns nothing normally. Provides the outer frame name expected in the stack trace.
      inner_exception()

    expected_regexp = r": blah.*"               # Error at the top
    expected_regexp += r"in raise_exception.*"  # Stacktrace outer
    expected_regexp += r"in inner_exception.*"  # Stacktrace inner
    expected_regexp += r": blah"                # Stacktrace of raise
    def expected_error_check(exception):
      # [FUNCTION DOC]
      # HELPER: predicate passed to the assertion for the expected-error check.
      # Takes the exception that was caught and returns a truthy match result if its text (with dots
      # matching newlines) contains the expected ordered pattern: message, outer frame, inner frame,
      # message; otherwise returns nothing/false. No side effects.
      return re.search(expected_regexp, str(exception), re.DOTALL)

    if eager:
      if context.executing_eagerly():
        with self.assertRaisesWithPredicateMatch(tf_exp, expected_error_check):
          f = script_ops.eager_py_func(raise_exception, [], [])
        return
      else:
        f = script_ops.eager_py_func(raise_exception, [], [])
    else:
      f = script_ops.py_func(raise_exception, [], [])

    with self.assertRaisesWithPredicateMatch(tf_exp, expected_error_check):
      self.evaluate(f)


class PyFuncTest(PyFuncTestBase):
  # [CLASS DOC]
  # Tests that exercise only the graph-style py_func operation (not eager_py_func). Inherits the
  # exception-handling helper from the base class. Each test builds py_func ops on constants or
  # placeholders, evaluates them (usually inside a cached session), and compares with expected values.
  # Many tests are restricted to graph-mode (v1) execution by decorators.
  """Encapsulates tests for py_func only."""

  def testRealDataTypes(self):
    # [FUNCTION DOC]
    # Scenario: py_func with a Python addition callback over each real numeric dtype (float16/32/64,
    # uint8/16, int8/16/32/64). For each dtype, creates two constants (1 and 2) of that dtype, wraps the
    # callback that adds them, evaluates, and asserts the result equals 3. No exceptions expected.
    def sum_func(x, y):
      # [FUNCTION DOC]
      # HELPER: callback for the real-dtype test; returns the sum of its two arguments.
      return x + y
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64,
                  dtypes.uint8, dtypes.int8, dtypes.uint16, dtypes.int16,
                  dtypes.int32, dtypes.int64]:
      with self.cached_session():
        x = constant_op.constant(1, dtype=dtype)
        y = constant_op.constant(2, dtype=dtype)
        z = self.evaluate(script_ops.py_func(sum_func, [x, y], dtype))
        self.assertEqual(z, 3)

  def testComplexDataTypes(self):
    # [FUNCTION DOC]
    # Scenario: py_func with a subtraction callback over complex64 and complex128. Creates constants
    # 1+1j and 2-2j of the dtype, evaluates the py_func of the subtraction callback, and asserts the
    # result equals -1+3j.
    def sub_func(x, y):
      # [FUNCTION DOC]
      # HELPER: callback for the complex-dtype test; returns the first argument minus the second.
      return x - y
    for dtype in [dtypes.complex64, dtypes.complex128]:
      with self.cached_session():
        x = constant_op.constant(1 + 1j, dtype=dtype)
        y = constant_op.constant(2 - 2j, dtype=dtype)
        z = self.evaluate(script_ops.py_func(sub_func, [x, y], dtype))
        self.assertEqual(z, -1 + 3j)

  def testBoolDataTypes(self):
    # [FUNCTION DOC]
    # Scenario: py_func with boolean inputs. Creates constants True and False, uses a callback that
    # returns the logical-and of the two, evaluates, and asserts the result is false.
    def and_func(x, y):
      # [FUNCTION DOC]
      # HELPER: callback for the bool test; returns the logical-and (short-circuit) of its two arguments.
      return x and y
    dtype = dtypes.bool
    with self.cached_session():
      x = constant_op.constant(True, dtype=dtype)
      y = constant_op.constant(False, dtype=dtype)
      z = self.evaluate(script_ops.py_func(and_func, [x, y], dtype))
      self.assertEqual(z, False)

  def testSingleType(self):
    # [FUNCTION DOC]
    # Scenario: py_func with a single output type (float32) and the numpy helper np_func. Two scalar
    # float32 constants (1 and 2) are passed; asserts the evaluated result equals the helper applied to
    # the same Python numbers, cast to float32.
    with self.cached_session():
      x = constant_op.constant(1.0, dtypes.float32)
      y = constant_op.constant(2.0, dtypes.float32)
      z = self.evaluate(script_ops.py_func(np_func, [x, y], dtypes.float32))
      self.assertEqual(z, np_func(1.0, 2.0).astype(np.float32))

  def testScalar(self):
    # [FUNCTION DOC]
    # Scenario: eager_py_func (despite the surrounding py_func test class) with scalar float32 inputs and
    # a list of one output dtype. Evaluates it in a cached session and asserts the first output equals
    # the numpy helper result cast to float32.
    with self.cached_session():
      x = constant_op.constant(1.0, dtypes.float32)
      y = constant_op.constant(2.0, dtypes.float32)
      z = self.evaluate(
          script_ops.eager_py_func(np_func, [x, y], [dtypes.float32]))
      self.assertEqual(z[0], np_func(1.0, 2.0).astype(np.float32))

  @test_util.run_v1_only("b/120545219")
  def testArray(self):
    # [FUNCTION DOC]
    # Scenario: py_func with float64 vectors using the numpy helper and a one-element output list. Only
    # runs in graph-mode v1 (decorator). Asserts the first returned array equals the helper applied
    # to the same Python lists, cast to float64.
    with self.cached_session():
      x = constant_op.constant([1.0, 2.0], dtypes.float64)
      y = constant_op.constant([2.0, 3.0], dtypes.float64)
      z = self.evaluate(script_ops.py_func(np_func, [x, y], [dtypes.float64]))
      self.assertAllEqual(z[0],
                          np_func([1.0, 2.0], [2.0, 3.0]).astype(np.float64))

  def testComplexType(self):
    # [FUNCTION DOC]
    # Scenario: py_func with complex64 scalars through the numpy helper; asserts the result is close to
    # the helper applied to the same complex Python numbers.
    with self.cached_session():
      x = constant_op.constant(1 + 2j, dtypes.complex64)
      y = constant_op.constant(3 + 4j, dtypes.complex64)
      z = self.evaluate(script_ops.py_func(np_func, [x, y], dtypes.complex64))
      self.assertAllClose(z, np_func(1 + 2j, 3 + 4j))

  def testRFFT(self):
    # [FUNCTION DOC]
    # Scenario: py_func with a callback that computes a real FFT of a float32 vector and returns complex64.
    # Asserts the evaluated output is close to numpy's real FFT of the same list.
    with self.cached_session():
      x = constant_op.constant([1., 2., 3., 4.], dtypes.float32)

      def rfft(x):
        # [FUNCTION DOC]
        # HELPER: callback for the real-FFT test; takes an array and returns its real-input FFT as a
        # complex64 numpy array.
        return np.fft.rfft(x).astype(np.complex64)

      y = self.evaluate(script_ops.py_func(rfft, [x], dtypes.complex64))
      self.assertAllClose(y, np.fft.rfft([1., 2., 3., 4.]))

  def testPythonLiteral(self):
    # [FUNCTION DOC]
    # Scenario: the callback returns a plain Python float rather than an array. A float64 constant 0 is
    # passed; the callback returns 1.0 when the value equals zero, else 0.0. Asserts the result is close
    # to 1.0, proving Python literals are converted to the declared output dtype.
    with self.cached_session():

      def literal(x):
        # [FUNCTION DOC]
        # HELPER: callback for the Python-literal test; converts its argument to a float and returns
        # 1.0 if it is zero, otherwise 0.0 (a Python float).
        return 1.0 if float(x) == 0.0 else 0.0

      x = constant_op.constant(0.0, dtypes.float64)
      y = self.evaluate(script_ops.py_func(literal, [x], dtypes.float64))
      self.assertAllClose(y, 1.0)

  def testList(self):
    # [FUNCTION DOC]
    # Scenario: the callback returns a Python list of two values and two float64 output dtypes are
    # declared. Asserts the evaluated outputs are close to zero and one for input zero.
    with self.cached_session():

      def list_func(x):
        # [FUNCTION DOC]
        # HELPER: callback for the list test; returns a list holding the argument and the argument plus one.
        return [x, x + 1]

      x = constant_op.constant(0.0, dtypes.float64)
      y = self.evaluate(
          script_ops.py_func(list_func, [x], [dtypes.float64] * 2))
      self.assertAllClose(y, [0.0, 1.0])

  def testTuple(self):
    # [FUNCTION DOC]
    # Scenario: the callback returns a Python tuple. Part one passes the input and output types as
    # lists; part two passes them as tuples. Both evaluations must give zero and one for input zero.
    # returns a tuple
    with self.cached_session():

      def tuple_func(x):
        # [FUNCTION DOC]
        # HELPER: callback for the tuple test; returns a tuple holding the argument and the argument plus one.
        return x, x + 1

      x = constant_op.constant(0.0, dtypes.float64)
      y = self.evaluate(
          script_ops.py_func(tuple_func, [x], [dtypes.float64] * 2))
      self.assertAllClose(y, [0.0, 1.0])

    # returns a tuple, Tout and inp a tuple
    with self.cached_session():
      x = constant_op.constant(0.0, dtypes.float64)
      y = self.evaluate(
          script_ops.py_func(tuple_func, (x,),
                             (dtypes.float64, dtypes.float64)))
      self.assertAllClose(y, [0.0, 1.0])

  @test_util.run_v1_only("b/120545219")
  def testStrings(self):
    # [FUNCTION DOC]
    # Scenario: string tensors passed to and returned from py_func. Graph-mode only. One py_func returns
    # a fixed-length numpy byte-string array, a second concatenates its two inputs. Asserts the final
    # result is the byte strings "hello there" and "hi there".

    def read_fixed_length_numpy_strings():
      # [FUNCTION DOC]
      # HELPER: callback returning a numpy array containing the single byte string " there".
      return np.array([b" there"])

    def read_and_return_strings(x, y):
      # [FUNCTION DOC]
      # HELPER: callback that returns the element-wise sum (concatenation for strings) of its two inputs.
      return x + y

    with self.cached_session():
      x = constant_op.constant([b"hello", b"hi"], dtypes.string)
      y = self.evaluate(
          script_ops.py_func(read_fixed_length_numpy_strings, [],
                             dtypes.string))
      z = self.evaluate(
          script_ops.py_func(read_and_return_strings, [x, y], dtypes.string))
      self.assertAllEqual(z, [b"hello there", b"hi there"])

  @test_util.run_v1_only("b/120545219")
  def testStringsAreConvertedToBytes(self):
    # [FUNCTION DOC]
    # Scenario: like the string test, but the callback returns text (unicode) strings, which py_func must
    # convert to bytes. Graph-mode only. Asserts the result is the byte strings "hello there" and
    # "hi there".

    def read_fixed_length_numpy_strings():
      # [FUNCTION DOC]
      # HELPER: callback returning a numpy array containing the single unicode string " there".
      return np.array([" there"])

    def read_and_return_strings(x, y):
      # [FUNCTION DOC]
      # HELPER: callback that returns the sum (concatenation) of its two inputs.
      return x + y

    with self.cached_session():
      x = constant_op.constant(["hello", "hi"], dtypes.string)
      y = self.evaluate(
          script_ops.py_func(read_fixed_length_numpy_strings, [],
                             dtypes.string))
      z = self.evaluate(
          script_ops.py_func(read_and_return_strings, [x, y], dtypes.string))
      self.assertAllEqual(z, [b"hello there", b"hi there"])

  @test_util.run_v1_only("b/120545219")
  def testObjectArraysAreConvertedToBytes(self):
    # [FUNCTION DOC]
    # Scenario: callback returns an object-dtype numpy array mixing a byte string and a unicode string.
    # Graph-mode only. Both are converted to byte strings; asserts the concatenated result list is
    # "hello there" and "hi ya" as bytes.

    def read_object_array():
      # [FUNCTION DOC]
      # HELPER: callback returning an object-dtype numpy array with one byte string and one unicode string.
      return np.array([b" there", u" ya"], dtype=np.object_)

    def read_and_return_strings(x, y):
      # [FUNCTION DOC]
      # HELPER: callback that returns the sum (concatenation) of its two inputs.
      return x + y

    with self.cached_session():
      x = constant_op.constant(["hello", "hi"], dtypes.string)
      y, = script_ops.py_func(read_object_array, [],
                              [dtypes.string])
      z, = script_ops.py_func(read_and_return_strings, [x, y], [dtypes.string])
      self.assertListEqual(list(self.evaluate(z)), [b"hello there", b"hi ya"])

  @test_util.run_v1_only("b/120545219")
  def testStringPadding(self):
    # [FUNCTION DOC]
    # Scenario: py_func callback returns a nested list of byte strings of different lengths, with a
    # string output type. Graph-mode only. Asserts the result equals the original list of byte strings.
    correct = [b"this", b"is", b"a", b"test"]
    with self.cached_session():
      s, = script_ops.py_func(lambda: [correct], [], [dtypes.string])
      self.assertAllEqual(s, correct)

  @test_util.run_v1_only("b/120545219")
  def testStringPaddingAreConvertedToBytes(self):
    # [FUNCTION DOC]
    # Scenario: same as the string padding test but the callback returns text strings; they must be
    # converted to bytes. Graph-mode only. Asserts the output equals the byte-string list.
    inp = ["this", "is", "a", "test"]
    correct = [b"this", b"is", b"a", b"test"]
    with self.cached_session():
      s, = script_ops.py_func(lambda: [inp], [], [dtypes.string])
      self.assertAllEqual(s, correct)

  @test_util.run_v1_only("b/120545219")
  def testNulTerminatedStrings(self):
    # [FUNCTION DOC]
    # Scenario: callback returns a fixed-width numpy string array whose entries contain trailing NUL
    # characters. Graph-mode only. Asserts the output strings have the NULs stripped.
    inp = np.array(["this\0", "is\0\0", "a\0", "test\0\0"], dtype=np.str_)
    correct = [b"this", b"is", b"a", b"test"]
    with self.cached_session():
      s, = script_ops.py_func(lambda: [inp], [], [dtypes.string])
      self.assertAllEqual(s, correct)

  @test_util.run_v1_only("b/120545219")
  def testLarge(self):
    # [FUNCTION DOC]
    # Scenario: stress test with large data. Graph-mode only. Creates a one-million-element zeros float32
    # tensor and two py_funcs (add one, multiply by two) and runs both ops a hundred times in a session.
    # There is no assertion; the test passes if nothing fails or crashes.
    with self.cached_session() as sess:
      x = array_ops.zeros([1000000], dtype=np.float32)
      y = script_ops.py_func(lambda x: x + 1, [x], [dtypes.float32])
      z = script_ops.py_func(lambda x: x * 2, [x], [dtypes.float32])
      for _ in range(100):
        sess.run([y[0].op, z[0].op])

  def testNoInput(self):
    # [FUNCTION DOC]
    # Scenario: py_func with an empty input list; the callback returns the constant 42.0 as float64.
    # Asserts the evaluated value is close to 42.0.
    with self.cached_session():
      x = self.evaluate(script_ops.py_func(lambda: 42.0, [], dtypes.float64))
      self.assertAllClose(x, 42.0)

  @test_util.run_v1_only("b/120545219")
  def testAlias(self):
    # [FUNCTION DOC]
    # Scenario: aliasing safety. Graph-mode only. A callback returns a numpy array that is held outside;
    # the py_func result is added to a constant and the op is run. Asserts the original numpy array is
    # unchanged (still one and two), so the op does not modify returned memory in place.
    with self.cached_session():
      np_array = np.array([1.0, 2.0], dtype=np.float32)
      tf_array = script_ops.py_func(lambda: np_array, [], [dtypes.float32])
      value = tf_array + constant_op.constant([2.0, 3.0], dtype=dtypes.float32)
      value.op.run()
      self.assertAllEqual(np_array, [1.0, 2.0])

  @test_util.run_v1_only("b/120545219")
  def testReturnUnicodeString(self):
    # [FUNCTION DOC]
    # Scenario: callback returns a non-ASCII unicode string (Chinese text) with a string output type.
    # Graph-mode only. Asserts the evaluated result equals the UTF-8 encoded bytes of the text.
    with self.cached_session():
      correct = u"你好 世界"

      def unicode_string():
        # [FUNCTION DOC]
        # HELPER: callback returning the non-ASCII string from the enclosing test.
        return correct

      z, = script_ops.py_func(unicode_string, [], [dtypes.string])
      self.assertEqual(self.evaluate(z), correct.encode("utf8"))

  @test_util.run_v1_only("b/120545219")
  def testBadNumpyReturnType(self):
    # [FUNCTION DOC]
    # Scenario: callback returns a structured numpy array (field based dtype), which is unsupported.
    # Graph-mode only. Evaluating the op must raise an internal error whose message mentions an
    # unsupported numpy data type.
    with self.cached_session():

      def bad():
        # [FUNCTION DOC]
        # HELPER: callback returning an empty structured numpy array with a named float32 field.
        # Structured numpy arrays aren't supported.
        return np.array([], dtype=[("foo", np.float32)])

      y, = script_ops.py_func(bad, [], [dtypes.float32])

      with self.assertRaisesRegex(errors.InternalError,
                                  "Unsupported numpy data type"):
        self.evaluate(y)

  @test_util.run_v1_only("b/120545219")
  def testBadReturnType(self):
    # [FUNCTION DOC]
    # Scenario: callback returns a dictionary, which is not a supported return type. Graph-mode only.
    # Evaluating must raise an internal error mentioning an unsupported object type.
    with self.cached_session():

      def bad():
        # [FUNCTION DOC]
        # HELPER: callback returning a dictionary mapping a name to a dtype object.
        # Non-string python objects aren't supported.
        return {"foo": dtypes.float32}

      z, = script_ops.py_func(bad, [], [dtypes.int64])

      with self.assertRaisesRegex(errors.InternalError,
                                  "Unsupported object type"):
        self.evaluate(z)

  @test_util.run_v1_only("b/120545219")
  def testReturnInput(self):
    # [FUNCTION DOC]
    # Scenario: the callback returns an element of its input (a numpy view aliasing a tensor's memory).
    # Graph-mode only. A placeholder is fed zero in a one element list; the result plus zero is
    # evaluated via feed. Asserts the result is 0.0, checking aliased buffers are handled correctly.
    with self.cached_session():

      def ident(x):
        # [FUNCTION DOC]
        # HELPER: callback returning the first element of its argument.
        return x[0]

      p = array_ops.placeholder(dtypes.float32)

      # Create a numpy array aliasing a tensor and a tensor aliasing this array
      z, = script_ops.py_func(ident, [p], [dtypes.float32])
      z += 0.0  # Makes sure we release the tensor aliasing the numpy array x[0]
      # above instead of using its memory as the return value of
      # session.run
      self.assertEqual(0.0, z.eval(feed_dict={p: [0.0]}))

  def testStateful(self):
    # [FUNCTION DOC]
    # Scenario: statefulness (default). Inside a fresh session (not the cached one so optimization is
    # enabled), py_func wraps a callback taking the next value of a Python iterator. Evaluating three
    # times must return 0, 1, 2, showing the op is re-executed each time.
    # Not using self.cached_session(), which disables optimization.
    with session_lib.Session():
      producer = iter(range(3))
      x, = script_ops.py_func(lambda: next(producer), [], [dtypes.int64])
      self.assertEqual(self.evaluate(x), 0)
      self.assertEqual(self.evaluate(x), 1)
      self.assertEqual(self.evaluate(x), 2)

  @test_util.enable_tf_xla_constant_folding("b/134376434")
  def testStateless(self):
    # [FUNCTION DOC]
    # Scenario: stateful flag set to false. Same iterator callback; the op is treated as pure and
    # constant folded, so three evaluations must all return 0. Decorator enables XLA constant folding.
    # Not using self.cached_session(), which disables optimization.
    with session_lib.Session():
      producer = iter(range(3))
      x, = script_ops.py_func(
          lambda: next(producer), [], [dtypes.int64], stateful=False)
      self.assertEqual(self.evaluate(x), 0)
      self.assertEqual(self.evaluate(x), 0)
      self.assertEqual(self.evaluate(x), 0)

  @test_util.run_v1_only("b/120545219")
  def testGradientFunction(self):
    # [FUNCTION DOC]
    # Scenario: gradient registration for py_func ops. Graph-mode only. Creates stateful and stateless
    # py_funcs taking a constant input and asserts that looking up the gradient function of either op
    # gives nothing (null). Comment notes an input is needed, otherwise lookup is null anyway.
    # Input to tf.compat.v1.py_func is necessary,
    # otherwise get_gradient_function() returns None per default.
    a = constant_op.constant(0)
    x, = script_ops.py_func(lambda a: 0, [a], [dtypes.int64])
    y, = script_ops.py_func(lambda a: 0, [a], [dtypes.int64], stateful=False)
    self.assertEqual(None, ops.get_gradient_function(x.op))
    self.assertEqual(None, ops.get_gradient_function(y.op))

  @test_util.run_v1_only("b/120545219")
  def testCOrder(self):
    # [FUNCTION DOC]
    # Scenario: memory layout. Graph-mode only. Callback returns a Fortran-ordered numpy matrix; asserts
    # the evaluated tensor equals the original nested list, so layout does not corrupt data.
    with self.cached_session():
      val = [[1, 2], [3, 4]]
      x, = script_ops.py_func(lambda: np.array(val, order="F"), [],
                              [dtypes.int64])
      self.assertAllEqual(val, self.evaluate(x))

  @test_util.run_v1_only("b/120545219")
  def testParallel(self):
    # [FUNCTION DOC]
    # Scenario: py_func callbacks can run concurrently. Graph-mode only. One callback puts a value in a
    # queue of size one and waits for completion; another gets from it and marks done. Both ops are run
    # in one session call; the test would deadlock if they were not executed in parallel. No explicit
    # asserts.
    # Tests that tf.compat.v1.py_func's can run in parallel if they release
    # the GIL.
    with self.cached_session() as session:
      q = queue.Queue(1)

      def blocking_put():
        # [FUNCTION DOC]
        # HELPER: producer callback for the parallel test; puts 42 in the shared queue, blocks until the
        # consumer calls task done, and returns 42.
        q.put(42)
        q.join()  # Wait for task_done().
        return 42

      def blocking_get():
        # [FUNCTION DOC]
        # HELPER: consumer callback for the parallel test; blocks until a value is available, marks the task
        # done, and returns the value.
        v = q.get(block=True)  # Wait for put().
        q.task_done()
        return v

      x, = script_ops.py_func(blocking_put, [], [dtypes.int64])
      y, = script_ops.py_func(blocking_get, [], [dtypes.int64])

      # This will result in a deadlock if the py_func's don't run in parallel.
      session.run([x, y])

  def testNoReturnValueStateful(self):
    # [FUNCTION DOC]
    # Scenario: stateful py_func with no outputs. A small class keeps a numpy counter and exposes an
    # increment op that wraps a py_func with empty output list and stateful true. Evaluates it with
    # increment two on a fresh counter of one; asserts the result of evaluation is null and the counter
    # state became three.

    class State:
      # [CLASS DOC]
      # HELPER: local test fixture holding a numpy integer counter, initial value one, which is mutated by
      # a py_func callback. Exposes increment (build the op), the private mutator, and a read-only value
      # property.

      def __init__(self):
        # [FUNCTION DOC]
        # HELPER: constructor; stores a one-element int64 numpy array with value 1 as the counter.
        self._value = np.array([1], np.int64)

      def _increment(self, diff):
        # [FUNCTION DOC]
        # HELPER: callback invoked by py_func; adds the diff argument to the stored counter in place. Returns
        # nothing.
        self._value += diff

      def increment(self, diff):
        # [FUNCTION DOC]
        # Takes diff (an int64 tensor) and returns a stateful py_func op with no outputs that calls the
        # private mutator with it when run. The result is the op, not a value.
        return script_ops.py_func(self._increment, [diff], [], stateful=True)

      @property
      def value(self):
        # [FUNCTION DOC]
        # Read-only property returning the current numpy counter array.
        return self._value

    with self.cached_session():
      s = State()
      op = s.increment(constant_op.constant(2, dtypes.int64))
      ret = self.evaluate(op)
      self.assertIsNone(ret)
      self.assertAllEqual([3], s.value)

  @test_util.run_v1_only("b/120545219")
  def testNoReturnValueStateless(self):
    # [FUNCTION DOC]
    # Scenario: stateless (stateful false) py_func with an empty output list and an int64 constant input.
    # Graph-mode only. Asserts the evaluation result is an empty list.

    def do_nothing(unused_x):
      # [FUNCTION DOC]
      # HELPER: callback that ignores its argument and returns nothing.
      pass

    f = script_ops.py_func(
        do_nothing, [constant_op.constant(3, dtypes.int64)], [], stateful=False)
    with self.cached_session():
      self.assertEqual(self.evaluate(f), [])

  @test_util.run_v1_only("b/120545219")
  def testExceptionHandling(self):
    # [FUNCTION DOC]
    # Scenario: translation of Python exception types to TensorFlow error types for py_func, using the
    # base class helper. Graph-mode only. Expected: value error and type error become invalid-argument,
    # stop-iteration becomes out-of-range, memory error becomes resource-exhausted, not-implemented
    # becomes unimplemented, and a custom unrelated exception type becomes unknown.
    with self.cached_session():
      self.verifyExceptionHandling(ValueError, errors.InvalidArgumentError)
      self.verifyExceptionHandling(TypeError, errors.InvalidArgumentError)
      self.verifyExceptionHandling(StopIteration, errors.OutOfRangeError)
      self.verifyExceptionHandling(MemoryError, errors.ResourceExhaustedError)
      self.verifyExceptionHandling(NotImplementedError,
                                   errors.UnimplementedError)

      class WeirdError(Exception):
        # [CLASS DOC]
        # HELPER: local custom exception type with no behavior, defined to verify that unrecognized
        # exceptions are mapped to the unknown error type.
        pass

      self.verifyExceptionHandling(WeirdError, errors.UnknownError)

  def testFunctionReferencesAreKept(self):
    # [FUNCTION DOC]
    # Scenario: lifetime of callbacks. In a new graph, a constant is created and a batch-decorated
    # function (batch size one, timeout ten, queue 100000) wraps a py_func with a lambda. The test calls
    # it, runs garbage collection, and then evaluates the result. Passes if no error occurs, showing the
    # py_func registry keeps the lambda alive after the function body exits.
    g = ops.Graph()
    with g.as_default():
      c = constant_op.constant([1.], dtypes.float32)
      @batch_ops.batch_function(1, 10, 100000)
      def fn(x):
        # [FUNCTION DOC]
        # HELPER: batch-decorated function used by the test; returns a py_func op over a lambda that returns
        # its input as float32. The only reference to the lambda lives in the py_func registry.
        # Upon exiting this function, the py_func holds the sole reference
        # to this lambda, without which it would be garbage collected.
        return script_ops.py_func(lambda x: x, [x], [dtypes.float32])
      result = fn(c)
      gc.collect()
      self.evaluate(result)


class PyFuncAndEagerPyFuncTest(PyFuncTestBase):
  # [CLASS DOC]
  # Test class for behavior shared by both py_func and eager_py_func, namely cleanup of entries in the
  # module-level registry of Python functions when graphs go away. Inherits the base class.
  """Encapsulates tests shared between py_func and eager_py_func."""

  def verifyPyFuncsNoIncrease(self, make_graph):
    # [FUNCTION DOC]
    # HELPER: shared verification routine. Takes make_graph, a callback that builds one graph with
    # py_func ops. Resets the default graph, collects garbage twice, records the size of the registry of
    # Python functions, calls the callback one thousand times, resets and collects again, and asserts
    # the registry size equals the starting size (no leak). Returns nothing.
    ops.reset_default_graph()
    gc.collect()
    gc.collect()
    initial_size = script_ops._py_funcs.size()

    for _ in range(1000):
      make_graph()

    ops.reset_default_graph()
    gc.collect()
    gc.collect()
    self.assertEqual(initial_size, script_ops._py_funcs.size())

  def testCleanup(self):
    # [FUNCTION DOC]
    # Scenario: no leak of registered functions. Defines a graph builder that creates a graph with
    # py_func and eager_py_func ops, including ones whose lambda references a tensor which references the
    # graph (a reference cycle, issue 18292). Passes it to the leak verifier.

    def make_graph():
      # [FUNCTION DOC]
      # HELPER: graph builder for the cleanup test; creates a new graph with a float32 constant and four
      # py_func or eager_py_func ops, two of which capture the constant in the lambda.
      g = ops.Graph()
      with g.as_default():
        c = constant_op.constant([1.], dtypes.float32)
        _ = script_ops.py_func(lambda x: x + 1, [c], [dtypes.float32])
        _ = script_ops.eager_py_func(lambda x: x + 1, [c], [dtypes.float32])
        # These ops have a reference to 'c' which has a reference to the
        # graph.
        # Checks if the functions are being deleted though the graph is
        # referenced from them (see #18292).
        script_ops.py_func(
            lambda x: x + c.shape[0], [c], [dtypes.float32])
        script_ops.eager_py_func(
            lambda x: x + c.shape[0], [c], [dtypes.float32])

    self.verifyPyFuncsNoIncrease(make_graph)

  def testCleanupInTfFunction(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test is disabled; its first statement skips it (bug b/144098211), so the code
    # below never runs.
    # Scenario (intended): same as the cleanup test but the graph builder wraps the op creation in a
    # function decorated with the tf function wrapper and calls it. Would verify no leaks via the
    # verifier.

    self.skipTest("b/144098211")

    def make_graph():
      # [FUNCTION DOC]
      # HELPER: graph builder (never run due to skip); builds a graph and defines and calls a traced
      # function that creates py_func and eager_py_func ops, some capturing the constant.
      g = ops.Graph()
      with g.as_default():
        @def_function.function
        def fn():
          # [FUNCTION DOC]
          # HELPER: traced function body used by the skipped test; creates a float32 constant and four
          # py_func or eager_py_func ops, two capturing the constant. Returns nothing.
          c = constant_op.constant([1.], dtypes.float32)
          _ = script_ops.py_func(lambda x: x + 1, [c], [dtypes.float32])
          _ = script_ops.eager_py_func(lambda x: x + 1, [c], [dtypes.float32])
          # These ops have a reference to 'c' which has a reference to the
          # graph.
          # Checks if the functions are being deleted though the graph is
          # referenced from them (see #18292).
          script_ops.py_func(
              lambda x: x + c.shape[0], [c], [dtypes.float32])
          script_ops.eager_py_func(
              lambda x: x + c.shape[0], [c], [dtypes.float32])
        fn()

    self.verifyPyFuncsNoIncrease(make_graph)


class EagerPyFuncTest(PyFuncTestBase):
  # [CLASS DOC]
  # Tests covering only eager_py_func. Inherits the exception helper from the base class. Covers
  # single and multiple outputs, gradients (tape and graph), device placement, tf function wrapping,
  # exceptions, tape cache bookkeeping, and ragged tensor inputs and outputs. Many tests run in both
  # graph and eager modes via decorator.
  """Encapsulates tests for eager_py_func only."""

  @test_util.run_in_graph_and_eager_modes
  def testEagerSingleOutputInt32(self):
    # [FUNCTION DOC]
    # Scenario: eager_py_func with the matrix multiplication helper on int32 3x3 and 3x1 ones, with a
    # single int32 output type. Runs in graph and eager modes. Asserts the result is a column of threes.
    a = array_ops.ones((3, 3), dtype=dtypes.int32)
    x = array_ops.ones((3, 1), dtype=dtypes.int32)
    output = script_ops.eager_py_func(matmul, inp=[a, x], Tout=dtypes.int32)
    ret = self.evaluate(output)
    self.assertAllEqual(ret, [[3], [3], [3]])

  @test_util.run_in_graph_and_eager_modes
  def testRenamedDeviceInTestClusterCorrectlyIdentifiedAsLocalhost(self):
    # [FUNCTION DOC]
    # Scenario: eager_py_func running on a renamed worker device of a local in-process cluster. Graph
    # and eager modes by decorator; skipped when executing eagerly (b/126565353). Creates a one-worker
    # cluster and a session on it, places float32 ones and the op on worker task zero's CPU, runs the
    # session and asserts the result is a column of threes. Side effect: starts local servers.
    if context.executing_eagerly():
      self.skipTest("b/126565353: We don't test eager's remote execution.")

    workers, _ = test_util.create_local_cluster(num_workers=1, num_ps=0)
    worker = workers[0]
    session = session_lib.Session(worker.target)
    with ops.device("/job:worker/task:0/cpu:0"):
      a = array_ops.ones((3, 3), dtype=dtypes.float32)
      x = array_ops.ones((3, 1), dtype=dtypes.float32)
      output = script_ops.eager_py_func(matmul, inp=[a, x], Tout=dtypes.float32)
    ret = session.run(output)
    self.assertAllClose(ret, [[3.0], [3.0], [3.0]])

  @test_util.run_in_graph_and_eager_modes
  def testEagerSingleOutputFloat32(self):
    # [FUNCTION DOC]
    # Scenario: eager_py_func with the matmul helper on float32 ones under a device context allowing
    # GPU. Runs in graph and eager modes. Asserts the result is close to a column of threes.
    with test_util.device(use_gpu=True):
      a = array_ops.ones((3, 3), dtype=dtypes.float32)
      x = array_ops.ones((3, 1), dtype=dtypes.float32)
      output = script_ops.eager_py_func(matmul, inp=[a, x], Tout=dtypes.float32)
      ret = self.evaluate(output)
      self.assertAllClose(ret, [[3.0], [3.0], [3.0]])

  @test_util.run_in_graph_and_eager_modes
  def testEagerArrayOutput(self):
    # [FUNCTION DOC]
    # Scenario: callback returns a list with one matmul result and Tout is a list with one dtype. Under a
    # device context allowing GPU. Asserts the output is a list holding the 3x1 column of threes.
    with test_util.device(use_gpu=True):
      a = array_ops.ones((3, 3), dtype=dtypes.float32)
      x = array_ops.ones((3, 1), dtype=dtypes.float32)
      output = script_ops.eager_py_func(
          lambda a, x: [matmul(a, x)], inp=[a, x], Tout=[dtypes.float32])
      ret = self.evaluate(output)
      self.assertAllEqual(ret, [[[3.0], [3.0], [3.0]]])

  @test_util.run_in_graph_and_eager_modes
  def testEagerReturnNone(self):
    # [FUNCTION DOC]
    # Scenario: callback returns nothing with an empty input list and empty output list. Evaluates. In
    # eager mode asserts the result has length zero, in graph mode asserts it is null.
    with test_util.device(use_gpu=True):
      def no_return_value():
        # [FUNCTION DOC]
        # HELPER: callback that takes no arguments and returns nothing.
        return

      output = script_ops.eager_py_func(no_return_value, inp=[], Tout=[])
      ret = self.evaluate(output)
      if context.executing_eagerly():
        self.assertEqual(len(ret), 0)
      else:
        self.assertIsNone(ret)

  @test_util.run_in_graph_and_eager_modes
  @test_util.disable_tfrt("b/180469928")
  def testEagerPyFuncInDefun(self):
    # [FUNCTION DOC]
    # Scenario: eager_py_func inside a traced function. Graph and eager modes; disabled for the TFRT
    # runtime. A traced function containing the op is called and the result asserted to be a column of
    # threes.
    with test_util.device(use_gpu=True):
      def wrapper():
        # [FUNCTION DOC]
        # HELPER: function body wrapped by the tracing decorator; builds float32 ones 3x3 and 3x1 and returns
        # the eager_py_func result of the matmul helper.
        a = array_ops.ones((3, 3), dtype=dtypes.float32)
        x = array_ops.ones((3, 1), dtype=dtypes.float32)
        return script_ops.eager_py_func(matmul, inp=[a, x], Tout=dtypes.float32)

      wrapped = def_function.function(wrapper)
      ret = self.evaluate(wrapped())
      self.assertAllEqual(ret, [[3.0], [3.0], [3.0]])

  @test_util.run_in_graph_and_eager_modes
  @test_util.run_v1_only("b/120545219")
  def testEagerExceptionHandling(self):
    # [FUNCTION DOC]
    # Scenario: exception translation for eager_py_func, using the base helper with the eager flag. Runs
    # in both modes but decorated for graph-mode v1 only effectively. Expected mappings as in the
    # py_func test: value and type errors to invalid-argument, stop-iteration to out-of-range, memory
    # error to resource-exhausted, not-implemented to unimplemented, a custom type to unknown.
    with test_util.device(use_gpu=True):
      self.verifyExceptionHandling(
          ValueError, errors.InvalidArgumentError, eager=True)
      self.verifyExceptionHandling(
          TypeError, errors.InvalidArgumentError, eager=True)
      self.verifyExceptionHandling(
          StopIteration, errors.OutOfRangeError, eager=True)
      self.verifyExceptionHandling(
          MemoryError, errors.ResourceExhaustedError, eager=True)
      self.verifyExceptionHandling(
          NotImplementedError, errors.UnimplementedError, eager=True)

      class WeirdError(Exception):
        # [CLASS DOC]
        # HELPER: local custom exception type without behavior, to check unrecognized exceptions map to the
        # unknown error.
        pass

      self.verifyExceptionHandling(WeirdError, errors.UnknownError, eager=True)

  @test_util.run_in_graph_and_eager_modes
  @test_util.run_v1_only("b/120545219")
  def testEagerReturningVariableRaisesError(self):
    # [FUNCTION DOC]
    # Scenario: callback returns a resource variable, which is not allowed. Expect an unknown error
    # whose message says an attempt was made to return a variable. The op creation and evaluation are
    # both inside the expectation block.
    def return_variable():
      # [FUNCTION DOC]
      # HELPER: callback returning a new float32 resource variable initialized to zero.
      return resource_variable_ops.ResourceVariable(0.0)

    with self.assertRaisesRegex(errors.UnknownError,
                                "Attempting to return a variable"):
      output = script_ops.eager_py_func(
          return_variable, inp=[], Tout=dtypes.float32)
      self.evaluate(output)

  @test_util.run_in_graph_and_eager_modes
  def testTapeCache(self):
    # [FUNCTION DOC]
    # Scenario: regression for b/198962664 (issue 51839), the gradient tape cache. Records the size of
    # the module-level tape cache; creates an op without an active tape and asserts size unchanged; then
    # under a gradient tape, creates the op, and asserts the cache grew by one when eager (unchanged in
    # graph mode); computes the gradient, evaluates it, and asserts the cache returned to its original
    # size.
    # Testing for b/198962664 (gh:#51839)
    old_cache_size = len(script_ops.tape_cache)

    def f(x):
      # [FUNCTION DOC]
      # HELPER: callback returning the square of its argument.
      return x**2

    x = constant_op.constant(3.0)

    y = script_ops.eager_py_func(f, inp=[x], Tout=dtypes.float32)

    # No cache if there is no active tape
    self.assertEqual(len(script_ops.tape_cache), old_cache_size)

    with backprop.GradientTape() as tape:
      tape.watch(x)
      y = script_ops.eager_py_func(f, inp=[x], Tout=dtypes.float32)
      # A new cache entry is created when running eagerly.
      if context.executing_eagerly():
        self.assertEqual(len(script_ops.tape_cache), old_cache_size + 1)
      else:
        self.assertEqual(len(script_ops.tape_cache), old_cache_size)
    dy_dx = tape.gradient(y, x)
    # Force a evaluation.
    self.evaluate(dy_dx)
    # Cache entry consumed after gradient calculation.
    self.assertEqual(len(script_ops.tape_cache), old_cache_size)

  @test_util.run_in_graph_and_eager_modes
  def testEagerGradientTape(self):
    # [FUNCTION DOC]
    # Scenario: gradients of eager_py_func under a tape, both modes. For float input three, callback
    # squares it; asserts the gradient is close to six. Repeats with complex input 3+3i and a complex128
    # output and asserts the gradient is close to the conjugate result, six minus six i.

    def f(x):
      # [FUNCTION DOC]
      # HELPER: callback returning the square of its argument.
      return x**2

    x = constant_op.constant(3.0)
    with backprop.GradientTape() as tape:
      tape.watch(x)
      y = script_ops.eager_py_func(f, inp=[x], Tout=dtypes.float32)
    dy_dx = tape.gradient(y, x)
    self.assertAllClose(self.evaluate(dy_dx), 6.0)

    # Test complex values
    x = constant_op.constant(3.0 + 3.0j)
    with backprop.GradientTape() as tape:
      tape.watch(x)
      y = script_ops.eager_py_func(f, inp=[x], Tout=dtypes.complex128)
    dy_dx = tape.gradient(y, x)
    # Gradient of complex will be the conj
    self.assertAllClose(self.evaluate(dy_dx), 6.0 - 6.0j)

  @test_util.run_v1_only("b/120545219")
  def testEagerGradientGraph(self):
    # [FUNCTION DOC]
    # Scenario: graph-style gradient of eager_py_func, graph-mode only. Squares a constant three and
    # asserts the gradient from the graph gradient function equals six.

    def f(x):
      # [FUNCTION DOC]
      # HELPER: callback returning the square of its argument.
      return x**2

    x = constant_op.constant(3.0)
    y = script_ops.eager_py_func(f, inp=[x], Tout=dtypes.float32)
    dy_dx = gradients_impl.gradients(y, x)[0]
    self.assertEqual(self.evaluate(dy_dx), 6.0)

  @test_util.run_v1_only("b/120545219")
  def testEagerGradientGraphTwoOutputs(self):
    # [FUNCTION DOC]
    # Scenario: gradient through an eager_py_func with two outputs (product and quotient). Graph-mode
    # only. Inputs three and two; gradient of the sum of both outputs with respect to the first input
    # is asserted to be 2.5 (two plus a half).

    def f(x, y):
      # [FUNCTION DOC]
      # HELPER: callback returning the tuple of the product and the quotient of its two arguments.
      return x * y, x / y

    x = constant_op.constant(3.0)
    y = constant_op.constant(2.0)
    fa, fb = script_ops.eager_py_func(f, inp=[x, y],
                                      Tout=[dtypes.float32, dtypes.float32])
    dy_dx = gradients_impl.gradients(fa + fb, x)[0]
    self.assertEqual(self.evaluate(dy_dx), 2.5)

  @test_util.run_in_graph_and_eager_modes
  def testEagerGradientTapeMultipleArgs(self):
    # [FUNCTION DOC]
    # Scenario: tape gradients with respect to two inputs, both modes. Callback computes sum of squares
    # for three and four; asserts the gradients are six and eight.

    def f(x, y):
      # [FUNCTION DOC]
      # HELPER: callback returning the sum of the squares of its two arguments.
      return x**2 + y**2

    x = constant_op.constant(3.0)
    y = constant_op.constant(4.0)
    with backprop.GradientTape() as tape:
      tape.watch(x)
      tape.watch(y)
      z = script_ops.eager_py_func(f, inp=[x, y], Tout=dtypes.float32)

    dz_dx, dz_dy = tape.gradient(z, [x, y])
    self.assertEqual(self.evaluate(dz_dx), 6.0)
    self.assertEqual(self.evaluate(dz_dy), 8.0)

  @test_util.run_v1_only("b/120545219")
  def testEagerGradientGraphMultipleArgs(self):
    # [FUNCTION DOC]
    # Scenario: graph gradients with respect to two inputs. Graph-mode only. Same function and values as
    # the tape test; asserts gradients six and eight.

    def f(x, y):
      # [FUNCTION DOC]
      # HELPER: callback returning the sum of the squares of its two arguments.
      return x**2 + y**2

    x = constant_op.constant(3.0)
    y = constant_op.constant(4.0)
    z = script_ops.eager_py_func(f, inp=[x, y], Tout=dtypes.float32)

    dz_dx, dz_dy = gradients_impl.gradients(z, [x, y])
    self.assertEqual(self.evaluate(dz_dx), 6.0)
    self.assertEqual(self.evaluate(dz_dy), 8.0)

  @test_util.run_v1_only("b/120545219")
  def testEagerGradientGraphLogHuber(self):
    # [FUNCTION DOC]
    # Scenario: gradient of a function with Python control flow. Graph-mode only. Placeholders x and m
    # feed a Huber-style callback; the gradient with respect to x is evaluated with x one and m two,
    # taking the first branch. Asserts value 1.0 and gradient 2.0.

    def log_huber(x, m):
      # [FUNCTION DOC]
      # HELPER: callback implementing a log-Huber loss with a Python branch; returns x squared if the
      # absolute value of x is at most m, otherwise m squared times one minus two log m plus log of x
      # squared.
      if math_ops.abs(x) <= m:
        return x**2
      else:
        return m**2 * (1 - 2 * math_ops.log(m) + math_ops.log(x**2))

    x = array_ops.placeholder(dtypes.float32)
    m = array_ops.placeholder(dtypes.float32)

    y = script_ops.eager_py_func(
        func=log_huber, inp=[x, m], Tout=dtypes.float32)
    dy_dx = gradients_impl.gradients(y, x)[0]

    with self.cached_session() as sess:
      # Takes the first branch of log_huber.
      y, dy_dx = sess.run([y, dy_dx], feed_dict={x: 1.0, m: 2.0})
      self.assertEqual(y, 1.0)
      self.assertEqual(dy_dx, 2.0)

  @test_util.run_v1_only("b/120545219")
  def testEagerRespectsDevicePlacementOfOp(self):
    # [FUNCTION DOC]
    # Scenario: device placement is respected. Graph-mode only. Under an explicit CPU device context, two
    # chained eager_py_funcs (square, then double) are built over a placeholder; run with three; asserts
    # the result is 18.

    def f(x):
      # [FUNCTION DOC]
      # HELPER: callback returning the square of its argument via the math op.
      return math_ops.square(x)

    def g(x):
      # [FUNCTION DOC]
      # HELPER: callback returning the argument added to itself via the math op.
      return math_ops.add(x, x)

    with ops.device("/CPU:0"):
      # Explicitly ask for the py_funcs to execute on CPU, even if
      # a GPU is available.
      x = array_ops.placeholder(dtypes.float32)
      y = script_ops.eager_py_func(func=f, inp=[x], Tout=dtypes.float32)
      z = script_ops.eager_py_func(func=g, inp=[y], Tout=dtypes.float32)

    with self.session() as sess:
      output = sess.run(z, feed_dict={x: 3.0})
      self.assertEqual(output, 18.0)

  @test_util.run_in_graph_and_eager_modes
  def testEagerPyFuncOnGPUWithStrings(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name mentions GPU but the test never requests or checks GPU placement; it only runs
    # with default placement.
    # Scenario: string tensors through eager_py_func in both modes. The callback returns the string form
    # of its input's dtype; asserts the result equals the UTF-8 bytes of the textual dtype representation
    # for strings.

    def fn(a):
      # [FUNCTION DOC]
      # HELPER: callback that returns the textual representation of its argument's dtype.
      return str(a.dtype)

    x = constant_op.constant("x", dtype=dtypes.string)
    output = script_ops.eager_py_func(fn, inp=[x], Tout=dtypes.string)
    self.assertEqual(self.evaluate(output), "<dtype: 'string'>".encode("utf8"))

  @test_util.run_in_graph_and_eager_modes
  def testEagerPyFuncNotACallable(self):
    # [FUNCTION DOC]
    # Scenario: passing a non-callable (a tensor) as the function. Both modes. Expects a value error whose
    # message mentions callable.
    x = constant_op.constant("x", dtype=dtypes.string)

    with self.assertRaisesRegex(ValueError, "callable"):
      _ = script_ops.eager_py_func(x, inp=[x], Tout=dtypes.string)

  def testUnsupportedToutType(self):
    # [FUNCTION DOC]
    # Scenario: invalid output type. A dictionary is passed in the output type list. Expects a type error
    # saying it cannot be converted to a TensorFlow DType.
    with self.assertRaisesRegex(
        TypeError, "Cannot convert .* to a TensorFlow DType."):
      script_ops.eager_py_func(lambda x: x, [1], [{}])

  def testRaggedTensorArg(self):
    # [FUNCTION DOC]
    # Scenario: a ragged tensor input. Passes a three-row ragged constant to eager_py_func with a
    # reduction callback and int32 output; asserts the result equals 21.
    x = ragged_factory_ops.constant([[1, 2, 3], [4], [5, 6]])
    y, = script_ops.eager_py_func(math_ops.reduce_sum, [x], [dtypes.int32])
    self.assertAllEqual(y, 21)

  def testRaggedTensorReturn(self):
    # [FUNCTION DOC]
    # Scenario: ragged tensor result. Callback builds a ragged tensor from values and row lengths; the
    # output type is a ragged tensor spec with two unknown dimensions. Asserts the result is a ragged
    # tensor equal to the nested list of rows three, one, two long.

    def fn(v, l):
      # [FUNCTION DOC]
      # HELPER: callback building a ragged tensor from flat values v and row lengths l.
      return ragged_tensor.RaggedTensor.from_row_lengths(v, l)

    values = [1, 2, 3, 4, 5, 6]
    lengths = constant_op.constant([3, 1, 2], dtypes.int64)
    out_signature = [ragged_tensor.RaggedTensorSpec([None, None], dtypes.int32)]
    y, = script_ops.eager_py_func(fn, [values, lengths], out_signature)
    self.assertIsInstance(y, ragged_tensor.RaggedTensor)
    self.assertAllEqual(y, [[1, 2, 3], [4], [5, 6]])

  def testRaggedTensorBroadcast(self):
    # [FUNCTION DOC]
    # Scenario: broadcasting needs shape info. A ragged tensor is added to the result of eager_py_func
    # that doubles it, with the output declared by a spec derived from the value. Evaluated eagerly and
    # inside a traced function; both results must equal the expected nested list.
    # Check that eager_py_func preserves output shape information, which is
    # required by broadcasting.
    def fn(x):
      # [FUNCTION DOC]
      # HELPER: callback returning twice its argument.
      return 2 * x

    def foo(x):
      # [FUNCTION DOC]
      # HELPER: adds a ragged tensor to the result of eager_py_func applied to it; builds the output spec
      # from the input value and returns the sum.
      spec = ragged_tensor.RaggedTensorSpec.from_value(x)
      res = script_ops.eager_py_func(fn, [x], spec)
      return x + res

    x = ragged_factory_ops.constant([[1.0, 2.0], [3.0]])
    expected_result = [[3.0, 6.0], [9.0]]
    result1 = foo(x)
    self.assertAllEqual(result1, expected_result)
    result2 = def_function.function(foo)(x)
    self.assertAllEqual(result2, expected_result)

  def testRaggedExpectedListGotList(self):
    # [FUNCTION DOC]
    # Scenario: Tout is a list with one ragged spec and the callback returns a one-element list; result
    # must equal the input.
    x = ragged_factory_ops.constant([[1, 2, 3], [4], [5, 6]])
    x_spec = type_spec.type_spec_from_value(x)
    y, = script_ops.eager_py_func(lambda v: [v], [x], [x_spec])
    self.assertAllEqual(y, x)

  def testRaggedExpectedListGotTuple(self):
    # [FUNCTION DOC]
    # Scenario: Tout is a list with one ragged spec and the callback returns a one-element tuple; result
    # must equal the input.
    x = ragged_factory_ops.constant([[1, 2, 3], [4], [5, 6]])
    x_spec = type_spec.type_spec_from_value(x)
    y, = script_ops.eager_py_func(lambda v: (v,), [x], [x_spec])
    self.assertAllEqual(y, x)

  def testRaggedExpectedListGotSingleValue(self):
    # [FUNCTION DOC]
    # Scenario: Tout is a list with one ragged spec and the callback returns the bare ragged value;
    # result must equal the input.
    x = ragged_factory_ops.constant([[1, 2, 3], [4], [5, 6]])
    x_spec = type_spec.type_spec_from_value(x)
    y, = script_ops.eager_py_func(lambda v: v, [x], [x_spec])
    self.assertAllEqual(y, x)

  def testRaggedNoReturnValue(self):
    # [FUNCTION DOC]
    # Scenario: callback returning null with a ragged input and empty Tout. Asserts an empty result when
    # eager, otherwise a null result.
    x = ragged_factory_ops.constant([[1, 2, 3], [4], [5, 6]])
    result = self.evaluate(script_ops.eager_py_func(lambda v: None, [x], []))
    if context.executing_eagerly():
      self.assertEqual(result, [])
    else:
      self.assertIsNone(result)

  def testRaggedBadReturnTypeExpectedTensorReturnedRagged(self):
    # [FUNCTION DOC]
    # Scenario: the spec expects a tensor but the callback returns a ragged tensor. Expects a value error
    # or invalid-argument error saying the function returned a value not matching the declared output
    # type. Op creation and evaluation both inside the expectation.
    rt = ragged_factory_ops.constant([[1, 2], [3, 4, 5]])
    with self.assertRaisesRegex(
        (ValueError, errors.InvalidArgumentError),
        "py_function: func=.* returned .* which did not match Tout=.*"):
      result = script_ops.eager_py_func(lambda x: x + 3, [rt], [dtypes.int32])
      self.evaluate(result)

  def testRaggedBadReturnTypeExpectedRaggedReturnedTensor(self):
    # [FUNCTION DOC]
    # Scenario: the spec expects a ragged tensor but the callback returns a plain dense tensor. Expects
    # a value error or invalid-argument error saying the returned value did not match the output type.
    with self.assertRaisesRegex(
        (ValueError, errors.InvalidArgumentError),
        "py_function: func=.* returned .* which did not match Tout=.*"):
      result = script_ops.eager_py_func(
          func=lambda x: x,
          inp=[constant_op.constant([[1, 2, 3]])],
          Tout=[ragged_tensor.RaggedTensorSpec([None, None], dtypes.int32)])
      self.evaluate(result)


if __name__ == "__main__":
  test.main()
