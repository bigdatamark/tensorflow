# [MODULE DOC]
# Unit tests for the runtime assertion and checking operations in the check ops module
# (assertions on equality, ordering, closeness, sign, rank, shape, dtype, scalar-ness, plus
# monotonicity predicates and the ensure-shape op).
# Each test class covers one public check function (or family): it builds small constant or
# placeholder tensors, wraps the assertion in a control dependency, evaluates, and verifies that
# the assertion either passes silently or raises the expected error type (invalid-argument,
# value, or type error) with a message matching a pattern.
# Many tests run in both graph and eager mode, where static (graph-construction time) failures
# and dynamic (run time) failures surface with different exception types; some are marked as
# deprecated-v1 graph-only tests that use placeholders and sessions.
# Also contains a small benchmark class that measures the overhead of the ensure-shape op.
# Import-time side effects: none beyond imports; running the file as a script starts the
# TensorFlow test runner.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.check_ops."""

import time

import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test


# pylint:disable=g-error-prone-assert-raises
class AssertV2Asserts(test.TestCase):
  # [CLASS DOC]
  # Test case covering the version-2 (eager-oriented) assertion functions as a group.
  # It has no state of its own; the single test iterates over a table of assertion functions.

  def test_passes_when_it_should(self):
    # [FUNCTION DOC]
    # Test: every v2 assertion function passes on good arguments and fails on bad ones.
    # Setup: inside eager mode, creates scalar and vector int/float constants, then builds a table
    # of cases; each case is the assertion function, a tuple of passing arguments, a tuple of
    # failing arguments and optionally the expected exception type (default: invalid-argument error).
    # Action: for each case, calls the function on the passing arguments (must not raise), then
    # defines and calls a compiled function that invokes it on the failing arguments with the
    # message "fail", and deletes that function.
    # Asserts: the call raises the case's expected error type with a message matching "fail".
    # Covers equal, less, near, greater, negative, positive, less-equal, none-equal, non-negative,
    # non-positive, greater-equal, type, integer, scalar, rank, rank-in and rank-at-least.
    # Side effects: prints progress lines to standard output.
    # This is a v2 test and need to run eagerly
    with context.eager_mode():
      c1 = constant_op.constant(-1, name="minus_one", dtype=dtypes.int32)
      c2 = constant_op.constant(2, name="two", dtype=dtypes.int32)
      c3 = constant_op.constant([3., 3.], name="three", dtype=dtypes.float32)
      c4 = constant_op.constant([3., 3.5], name="three_and_a_half",
                                dtype=dtypes.float32)
      scalar = c1
      non_scalar = c3
      integer = c1
      non_integer = c3
      positive = c2
      negative = c1
      cases = [
          (check_ops.assert_equal_v2, (c1, c1), (c1, c2)),
          (check_ops.assert_less_v2, (c1, c2), (c1, c1)),
          (check_ops.assert_near_v2, (c3, c3), (c3, c4)),
          (check_ops.assert_greater_v2, (c2, c1), (c1, c1)),
          (check_ops.assert_negative_v2, (negative,), (positive,)),
          (check_ops.assert_positive_v2, (positive,), (negative,)),
          (check_ops.assert_less_equal_v2, (c1, c1), (c2, c1)),
          (check_ops.assert_none_equal_v2, (c1, c2), (c3, c4)),
          (check_ops.assert_non_negative_v2, (positive,), (negative,)),
          (check_ops.assert_non_positive_v2, (negative,), (positive,)),
          (check_ops.assert_greater_equal_v2, (c1, c1), (c1, c2)),
          (check_ops.assert_type_v2, (c1, dtypes.int32), (c1, dtypes.float32),
           TypeError),
          (check_ops.assert_integer_v2, (integer,), (non_integer,),
           TypeError),
          (check_ops.assert_scalar_v2, (scalar,), (non_scalar,),
           ValueError),
          (check_ops.assert_rank_v2, (c1, 0), (c3, 2), ValueError),
          (check_ops.assert_rank_in_v2, (c1, [0, 1]), (c1, [1, 2]),
           ValueError),
          (check_ops.assert_rank_at_least_v2, (non_scalar, 1), (scalar, 1),
           ValueError),
      ]

      for case in cases:
        fn = case[0]
        passing_args = case[1]
        failing_args = case[2]
        error = errors.InvalidArgumentError if len(case) < 4 else case[3]

        print("Testing %s passing properly." % fn)

        fn(*passing_args)

        print("Testing %s failing properly." % fn)

        @def_function.function
        def failing_fn():
          # [FUNCTION DOC]
          # HELPER: Nested helper for the table-driven test above; wraps one failing assertion call.
          # Compiled (graph function) wrapper that invokes the current case's assertion function on the
          # failing arguments with the message "fail". Takes no arguments, returns nothing, and raises
          # whatever the assertion raises. It closes over the loop variables of the enclosing test and is
          # called immediately, so the late-binding closure is safe.
          fn(*failing_args, message="fail")  # pylint: disable=cell-var-from-loop

        with self.assertRaisesRegex(error, "fail"):
          failing_fn()

        del failing_fn


class AssertProperIterableTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the proper-iterable assertion, which rejects single tensors, sparse tensors,
  # arrays and strings (things that are iterable but not a proper collection of tensors) and
  # non-iterables, while accepting lists and generator-like collections. No shared fixtures.

  @test_util.run_in_graph_and_eager_modes
  def test_single_tensor_raises(self):
    # [FUNCTION DOC]
    # Test: passing one dense tensor instead of a collection is rejected.
    # Runs in graph and eager mode. Expects a type error whose message mentions "proper".
    tensor = constant_op.constant(1)
    with self.assertRaisesRegex(TypeError, "proper"):
      check_ops.assert_proper_iterable(tensor)

  @test_util.run_in_graph_and_eager_modes
  def test_single_sparse_tensor_raises(self):
    # [FUNCTION DOC]
    # Test: passing a single sparse tensor (built from indices, values, dense shape) is rejected.
    # Runs in graph and eager mode. Expects a type error whose message mentions "proper".
    ten = sparse_tensor.SparseTensor(
        indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])
    with self.assertRaisesRegex(TypeError, "proper"):
      check_ops.assert_proper_iterable(ten)

  @test_util.run_in_graph_and_eager_modes
  def test_single_ndarray_raises(self):
    # [FUNCTION DOC]
    # Test: passing a single numpy array is rejected as not a proper iterable.
    # Runs in graph and eager mode. Expects a type error whose message mentions "proper".
    array = np.array([1, 2, 3])
    with self.assertRaisesRegex(TypeError, "proper"):
      check_ops.assert_proper_iterable(array)

  @test_util.run_in_graph_and_eager_modes
  def test_single_string_raises(self):
    # [FUNCTION DOC]
    # Test: passing a plain string is rejected as not a proper iterable.
    # Runs in graph and eager mode. Expects a type error whose message mentions "proper".
    mystr = "hello"
    with self.assertRaisesRegex(TypeError, "proper"):
      check_ops.assert_proper_iterable(mystr)

  @test_util.run_in_graph_and_eager_modes
  def test_non_iterable_object_raises(self):
    # [FUNCTION DOC]
    # Test: passing an integer (not iterable at all) is rejected.
    # Runs in graph and eager mode. Expects a type error whose message contains "to be iterable".
    non_iterable = 1234
    with self.assertRaisesRegex(TypeError, "to be iterable"):
      check_ops.assert_proper_iterable(non_iterable)

  @test_util.run_in_graph_and_eager_modes
  def test_list_does_not_raise(self):
    # [FUNCTION DOC]
    # Test: a list of two tensors is accepted.
    # Runs in graph and eager mode. Calls the assertion on the list; nothing is asserted beyond
    # the absence of an exception.
    list_of_stuff = [
        constant_op.constant([11, 22]), constant_op.constant([1, 2])
    ]
    check_ops.assert_proper_iterable(list_of_stuff)

  @test_util.run_in_graph_and_eager_modes
  def test_generator_does_not_raise(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name and variable say "generator", but the code builds a tuple of two
    # tensors, not a generator, so no generator input is actually exercised.
    # Test: a collection of two tensors is accepted by the proper-iterable assertion.
    # Runs in graph and eager mode; asserts only that no exception is raised.
    generator_of_stuff = (constant_op.constant([11, 22]), constant_op.constant(
        [1, 2]))
    check_ops.assert_proper_iterable(generator_of_stuff)


class AssertEqualTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the equality assertion: passing cases (equal, broadcastable, empty, identical
  # inputs), failing cases (static and dynamic, with message formatting and summarize limits),
  # and the eager-mode behavior of returning nothing. No shared fixtures.

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_equal(self):
    # [FUNCTION DOC]
    # Test: asserting a tensor equal to itself passes.
    # Setup: a two-element integer constant. Action: runs the assertion as a control dependency of
    # an identity op and evaluates it. Asserts: evaluation succeeds without error (graph and eager).
    small = constant_op.constant([1, 2], name="small")
    with ops.control_dependencies([check_ops.assert_equal(small, small)]):
      out = array_ops.identity(small)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_scalar_comparison(self):
    # [FUNCTION DOC]
    # Test: comparing the boolean constants true and false fails with a custom message.
    # Runs in graph and eager mode. Expects an invalid-argument error whose text matches "fail".
    const_true = constant_op.constant(True, name="true")
    const_false = constant_op.constant(False, name="false")
    with self.assertRaisesRegex(errors.InvalidArgumentError, "fail"):
      check_ops.assert_equal(const_true, const_false, message="fail")

  def test_returns_none_with_eager(self):
    # [FUNCTION DOC]
    # Test: in eager mode the equality assertion returns nothing (null value) when it passes.
    # Setup: eager context, equal tensors. Asserts the returned value is nothing.
    with context.eager_mode():
      small = constant_op.constant([1, 2], name="small")
      x = check_ops.assert_equal(small, small)
      assert x is None

  @test_util.run_in_graph_and_eager_modes
  @test_util.run_deprecated_v1
  def test_raises_when_greater(self):
    # [FUNCTION DOC]
    # Test: static check - a larger constant compared with a smaller constant fails.
    # Both constants are known at graph-construction time, so the error is raised immediately by the
    # assertion call itself. Expects an invalid-argument error matching "fail". Marked as
    # deprecated-v1 and run in graph and eager modes.
    # Static check
    static_small = constant_op.constant([1, 2], name="small")
    static_big = constant_op.constant([3, 4], name="big")
    with self.assertRaisesRegex(errors.InvalidArgumentError, "fail"):
      check_ops.assert_equal(static_big, static_small, message="fail")

  @test_util.run_deprecated_v1
  def test_raises_when_greater_dynamic(self):
    # [FUNCTION DOC]
    # Test: dynamic check - unequal values fed through placeholders fail at run time.
    # Setup (graph mode, in a cached session): two int placeholders, assertion on (big, small) with
    # message "fail" as a control dependency of an identity. Feeds big=[3,4], small=[1,2] and
    # expects an op error matching "fail" followed by names "big" and "small".
    with self.cached_session():
      small = array_ops.placeholder(dtypes.int32, name="small")
      big = array_ops.placeholder(dtypes.int32, name="big")
      with ops.control_dependencies(
          [check_ops.assert_equal(big, small, message="fail")]):
        out = array_ops.identity(small)
      with self.assertRaisesOpError("fail.*big.*small"):
        out.eval(feed_dict={small: [1, 2], big: [3, 4]})

  def test_error_message_eager(self):
    # [FUNCTION DOC]
    # Test: the detailed error text produced in eager mode, for three summarize settings.
    # Setup: 3x2 integer tensors that differ in several positions. Defines three expected message
    # patterns (summarize 10 shows 6 elements and 3 differing indices, the default shows 3 elements,
    # summarize 2 shows 2 differing indices and 2 elements).
    # Action: calls the equality assertion with the message "big does not equal small" and each
    # setting. Asserts an invalid-argument error whose text matches the corresponding pattern,
    # including the differing indices and x/y values.
    expected_error_msg_full = r"""big does not equal small
Condition x == y did not hold.
Indices of first 3 different values:
\[\[0 0\]
 \[1 1\]
 \[2 0\]\]
Corresponding x values:
\[2 3 6\]
Corresponding y values:
\[20 30 60\]
First 6 elements of x:
\[2 2 3 3 6 6\]
First 6 elements of y:
\[20  2  3 30 60  6\]"""
    expected_error_msg_default = r"""big does not equal small
Condition x == y did not hold.
Indices of first 3 different values:
\[\[0 0\]
 \[1 1\]
 \[2 0\]\]
Corresponding x values:
\[2 3 6\]
Corresponding y values:
\[20 30 60\]
First 3 elements of x:
\[2 2 3\]
First 3 elements of y:
\[20  2  3\]"""
    expected_error_msg_short = r"""big does not equal small
Condition x == y did not hold.
Indices of first 2 different values:
\[\[0 0\]
 \[1 1\]\]
Corresponding x values:
\[2 3\]
Corresponding y values:
\[20 30\]
First 2 elements of x:
\[2 2\]
First 2 elements of y:
\[20  2\]"""
    with context.eager_mode():
      big = constant_op.constant([[2, 2], [3, 3], [6, 6]])
      small = constant_op.constant([[20, 2], [3, 30], [60, 6]])
      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  expected_error_msg_full):
        check_ops.assert_equal(big, small, message="big does not equal small",
                               summarize=10)
      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  expected_error_msg_default):
        check_ops.assert_equal(big, small, message="big does not equal small")
      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  expected_error_msg_short):
        check_ops.assert_equal(big, small, message="big does not equal small",
                               summarize=2)

  @test_util.run_in_graph_and_eager_modes
  @test_util.run_deprecated_v1
  def test_raises_when_less(self):
    # [FUNCTION DOC]
    # Test: static check - a smaller constant compared with a larger one fails.
    # Expects an invalid-argument error matching "fail" at the assertion call. Deprecated-v1 mode,
    # run in graph and eager modes.
    # Static check
    static_small = constant_op.constant([3, 1], name="small")
    static_big = constant_op.constant([4, 2], name="big")
    with self.assertRaisesRegex(errors.InvalidArgumentError, "fail"):
      check_ops.assert_equal(static_big, static_small, message="fail")

  @test_util.run_deprecated_v1
  def test_raises_when_less_dynamic(self):
    # [FUNCTION DOC]
    # Test: dynamic check with placeholders for unequal values (small=[3,1], big=[4,2]).
    # Graph mode in a cached session; expects an op error whose message matches "small" then "big"
    # when the identity op is evaluated.
    with self.cached_session():
      small = array_ops.placeholder(dtypes.int32, name="small")
      big = array_ops.placeholder(dtypes.int32, name="big")
      with ops.control_dependencies([check_ops.assert_equal(small, big)]):
        out = array_ops.identity(small)
      with self.assertRaisesOpError("small.*big"):
        out.eval(feed_dict={small: [3, 1], big: [4, 2]})

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_equal_and_broadcastable_shapes(self):
    # [FUNCTION DOC]
    # Test: equal values with broadcastable shapes pass.
    # Compares a 2x2 tensor with a length-2 vector holding the same row; the assertion is attached
    # as a control dependency and the output evaluated without error (graph and eager).
    small = constant_op.constant([[1, 2], [1, 2]], name="small")
    small_2 = constant_op.constant([1, 2], name="small_2")
    with ops.control_dependencies([check_ops.assert_equal(small, small_2)]):
      out = array_ops.identity(small)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_raises_when_equal_but_non_broadcastable_shapes(self):
    # [FUNCTION DOC]
    # Test: shapes of length 3 and 2 cannot be broadcast, so the assertion fails.
    # Because eager mode checks shapes in the kernel and graph mode while building the op, accepts
    # either an invalid-argument or a value error via the incompatible-shapes assertion helper.
    small = constant_op.constant([1, 1, 1], name="small")
    small_2 = constant_op.constant([1, 1], name="small_2")
    # The exception in eager and non-eager mode is different because
    # eager mode relies on shape check done as part of the C++ op, while
    # graph mode does shape checks when creating the `Operation` instance.
    with self.assertRaisesIncompatibleShapesError(
        (errors.InvalidArgumentError, ValueError)):
      with ops.control_dependencies([check_ops.assert_equal(small, small_2)]):
        out = array_ops.identity(small)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_raises_when_not_equal_and_broadcastable_shapes(self):
    # [FUNCTION DOC]
    # Test: comparing a boolean vector (true, false) with the scalar false fails after broadcast.
    # Expects an invalid-argument error matching "fail" (graph and eager).
    cond = constant_op.constant([True, False], name="small")
    with self.assertRaisesRegex(errors.InvalidArgumentError, "fail"):
      check_ops.assert_equal(cond, False, message="fail")

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_both_empty(self):
    # [FUNCTION DOC]
    # Test: two empty tensors are trivially equal; the assertion passes and the output evaluates.
    larry = constant_op.constant([])
    curly = constant_op.constant([])
    with ops.control_dependencies([check_ops.assert_equal(larry, curly)]):
      out = array_ops.identity(larry)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_noop_when_both_identical(self):
    # [FUNCTION DOC]
    # Test: asserting a tensor equal to the very same tensor object is optimized away.
    # In eager mode the result must be nothing; in graph mode the returned op must be of type
    # "NoOp".
    larry = constant_op.constant([])
    check_op = check_ops.assert_equal(larry, larry)
    if context.executing_eagerly():
      self.assertIs(check_op, None)
    else:
      self.assertEqual(check_op.type, "NoOp")


class AssertNoneEqualTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the none-equal assertion (every element pair must differ): passing, failing,
  # broadcast, empty, eager-return, static-check and message-format tests. No shared fixtures.

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_not_equal(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: both constants are given the op name "small" (the second one, holding larger
    # values, is stored in a variable called big), so the name is misleading and duplicated.
    # Test: tensors with all-different elements pass the none-equal assertion.
    # Evaluates an identity op guarded by the assertion; no error expected (graph and eager).
    small = constant_op.constant([1, 2], name="small")
    big = constant_op.constant([10, 20], name="small")
    with ops.control_dependencies(
        [check_ops.assert_none_equal(big, small)]):
      out = array_ops.identity(small)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  @test_util.run_deprecated_v1
  def test_raises_when_equal(self):
    # [FUNCTION DOC]
    # Test: asserting a tensor none-equal to itself fails.
    # Expects an op error containing "x != y did not hold" when the guarded identity is evaluated.
    # Deprecated-v1 mode, run in graph and eager modes.
    small = constant_op.constant([3, 1], name="small")
    with self.assertRaisesOpError("x != y did not hold"):
      with ops.control_dependencies(
          [check_ops.assert_none_equal(small, small)]):
        out = array_ops.identity(small)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_not_equal_and_broadcastable_shapes(self):
    # [FUNCTION DOC]
    # Test: a length-2 tensor and a length-1 tensor with different values pass after broadcasting.
    # Evaluates the guarded identity op without error.
    small = constant_op.constant([1, 2], name="small")
    big = constant_op.constant([3], name="big")
    with ops.control_dependencies(
        [check_ops.assert_none_equal(small, big)]):
      out = array_ops.identity(small)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_raises_when_not_equal_but_non_broadcastable_shapes(self):
    # [FUNCTION DOC]
    # Test: shapes of length 3 and 2 are not broadcastable, so the assertion fails.
    # Eager and graph mode raise different error types (kernel vs. op construction), so either a
    # value error or an invalid-argument error is accepted through the incompatible-shapes helper.
    small = constant_op.constant([1, 1, 1], name="small")
    big = constant_op.constant([10, 10], name="big")
    # The exception in eager and non-eager mode is different because
    # eager mode relies on shape check done as part of the C++ op, while
    # graph mode does shape checks when creating the `Operation` instance.
    with self.assertRaisesIncompatibleShapesError(
        (ValueError, errors.InvalidArgumentError)):
      with ops.control_dependencies(
          [check_ops.assert_none_equal(small, big)]):
        out = array_ops.identity(small)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_both_empty(self):
    # [FUNCTION DOC]
    # Test: two empty tensors satisfy the none-equal assertion (no element pairs to compare).
    larry = constant_op.constant([])
    curly = constant_op.constant([])
    with ops.control_dependencies(
        [check_ops.assert_none_equal(larry, curly)]):
      out = array_ops.identity(larry)
    self.evaluate(out)

  def test_returns_none_with_eager(self):
    # [FUNCTION DOC]
    # Test: in eager mode a passing none-equal assertion returns nothing (null value).
    with context.eager_mode():
      t1 = constant_op.constant([1, 2])
      t2 = constant_op.constant([3, 4])
      x = check_ops.assert_none_equal(t1, t2)
      assert x is None

  def test_static_check_in_graph_mode(self):
    # [FUNCTION DOC]
    # Test: in graph mode, two equal Python integers are detected statically.
    # Calls the assertion on 1 and 1 with a custom message and expects an invalid-argument error
    # matching "Custom error message" immediately, without running a session.
    with ops.Graph().as_default():
      with self.assertRaisesRegex(  # pylint:disable=g-error-prone-assert-raises
          errors.InvalidArgumentError, "Custom error message"):
        check_ops.assert_none_equal(1, 1, message="Custom error message")

  def test_error_message_eager(self):
    # [FUNCTION DOC]
    # Test: the printed data in the eager error message follows the summarize argument.
    # Setup: a 2x3 float tensor holding 0..5 compared with itself (so every element fails).
    # Expected patterns list all six values for summarize 10 and -1, three values for the default
    # and two values for summarize 2. Asserts an invalid-argument error matching each pattern.
    # Note that the following three strings are regexes
    expected_error_msg_full = r"""\[ *0\. +1\. +2\. +3\. +4\. +5\.\]"""
    expected_error_msg_default = r"""\[ *0\. +1\. +2\.\]"""
    expected_error_msg_short = r"""\[ *0\. +1\.\]"""
    with context.eager_mode():
      t = constant_op.constant(
          np.array(range(6)), shape=[2, 3], dtype=np.float32)
      with self.assertRaisesRegex(  # pylint:disable=g-error-prone-assert-raises
          errors.InvalidArgumentError, expected_error_msg_full):
        check_ops.assert_none_equal(
            t, t, message="This is the error message.", summarize=10)
      with self.assertRaisesRegex(  # pylint:disable=g-error-prone-assert-raises
          errors.InvalidArgumentError, expected_error_msg_full):
        check_ops.assert_none_equal(
            t, t, message="This is the error message.", summarize=-1)
      with self.assertRaisesRegex(  # pylint:disable=g-error-prone-assert-raises
          errors.InvalidArgumentError, expected_error_msg_default):
        check_ops.assert_none_equal(t, t, message="This is the error message.")
      with self.assertRaisesRegex(  # pylint:disable=g-error-prone-assert-raises
          errors.InvalidArgumentError, expected_error_msg_short):
        check_ops.assert_none_equal(
            t, t, message="This is the error message.", summarize=2)


class AssertAllCloseTest(test.TestCase):
  # [CLASS DOC]
  # KNOWN ISSUE: the class is named for "all close" but every test exercises the near assertion
  # (assert_near), not an all-close function.
  # Test case for the near assertion: tolerance defaults for 32 and 64-bit floats, custom
  # relative/absolute tolerances, empty tensors, violations, eager return value and complex input.

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_equal(self):
    # [FUNCTION DOC]
    # Test: identical scalar floats pass the near assertion; evaluates the guarded identity op.
    x = constant_op.constant(1., name="x")
    y = constant_op.constant(1., name="y")
    with ops.control_dependencies(
        [check_ops.assert_near(x, y, message="failure message")]):
      out = array_ops.identity(x)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_close_enough_32_bit_due_to_default_rtol(self):
    # [FUNCTION DOC]
    # Test: float32 values 1 and 1 plus twice machine epsilon pass with absolute tolerance forced to
    # zero, so only the default relative tolerance (ten times epsilon) lets them match.
    eps = np.finfo(np.float32).eps
    # Default rtol/atol is 10*eps
    x = constant_op.constant(1., name="x")
    y = constant_op.constant(1. + 2 * eps, name="y", dtype=np.float32)
    with ops.control_dependencies(
        [check_ops.assert_near(x, y, atol=0., message="failure message")]):
      out = array_ops.identity(x)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_close_enough_32_bit_due_to_default_atol(self):
    # [FUNCTION DOC]
    # Test: float32 values 0 and twice machine epsilon pass with relative tolerance forced to zero,
    # so only the default absolute tolerance (ten times epsilon) lets them match.
    eps = np.finfo(np.float32).eps
    # Default rtol/atol is 10*eps
    x = constant_op.constant(0., name="x")
    y = constant_op.constant(0. + 2 * eps, name="y", dtype=np.float32)
    with ops.control_dependencies(
        [check_ops.assert_near(x, y, rtol=0., message="failure message")]):
      out = array_ops.identity(x)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_close_enough_64_bit_due_to_default_rtol(self):
    # [FUNCTION DOC]
    # Test: float64 version of the default relative tolerance check; 1 and 1 plus twice epsilon with
    # absolute tolerance zero pass.
    eps = np.finfo(np.float64).eps
    # Default rtol/atol is 10*eps
    x = constant_op.constant(1., name="x", dtype=np.float64)
    y = constant_op.constant(1. + 2 * eps, name="y", dtype=np.float64)
    with ops.control_dependencies(
        [check_ops.assert_near(x, y, atol=0., message="failure message")]):
      out = array_ops.identity(x)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_close_enough_64_bit_due_to_default_atol(self):
    # [FUNCTION DOC]
    # Test: float64 version of the default absolute tolerance check; 0 and twice epsilon with
    # relative tolerance zero pass.
    eps = np.finfo(np.float64).eps
    # Default rtol/atol is 10*eps
    x = constant_op.constant(0., name="x", dtype=np.float64)
    y = constant_op.constant(0. + 2 * eps, name="y", dtype=np.float64)
    with ops.control_dependencies(
        [check_ops.assert_near(x, y, rtol=0., message="failure message")]):
      out = array_ops.identity(x)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_close_enough_due_to_custom_rtol(self):
    # [FUNCTION DOC]
    # Test: 1.0 and 1.1 pass when absolute tolerance is 0 and relative tolerance is 0.5.
    x = constant_op.constant(1., name="x")
    y = constant_op.constant(1.1, name="y")
    with ops.control_dependencies(
        [check_ops.assert_near(x, y, atol=0., rtol=0.5,
                               message="failure message")]):
      out = array_ops.identity(x)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_close_enough_due_to_custom_atol(self):
    # [FUNCTION DOC]
    # Test: 0.0 and 0.1 pass when absolute tolerance is 0.5 and relative tolerance is 0.
    x = constant_op.constant(0., name="x")
    y = constant_op.constant(0.1, name="y", dtype=np.float32)
    with ops.control_dependencies(
        [check_ops.assert_near(x, y, atol=0.5, rtol=0.,
                               message="failure message")]):
      out = array_ops.identity(x)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_both_empty(self):
    # [FUNCTION DOC]
    # Test: two empty tensors pass the near assertion with default tolerances.
    larry = constant_op.constant([])
    curly = constant_op.constant([])
    with ops.control_dependencies([check_ops.assert_near(larry, curly)]):
      out = array_ops.identity(larry)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_raises_when_atol_violated(self):
    # [FUNCTION DOC]
    # Test: values 10.0 and 10.2 with absolute tolerance 0.1 fail.
    # Expects an op error containing "x and y not equal to tolerance" on evaluation.
    x = constant_op.constant(10., name="x")
    y = constant_op.constant(10.2, name="y")
    with self.assertRaisesOpError(  # pylint:disable=g-error-prone-assert-raises
        "x and y not equal to tolerance"):
      with ops.control_dependencies(
          [check_ops.assert_near(x, y, atol=0.1,
                                 message="failure message")]):
        out = array_ops.identity(x)
        self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_raises_when_default_rtol_violated(self):
    # [FUNCTION DOC]
    # Test: values 0.1 and 0.0 with default tolerances fail.
    # Expects an op error containing "x and y not equal to tolerance" on evaluation.
    x = constant_op.constant(0.1, name="x")
    y = constant_op.constant(0.0, name="y")
    with self.assertRaisesOpError(  # pylint:disable=g-error-prone-assert-raises
        "x and y not equal to tolerance"):
      with ops.control_dependencies(
          [check_ops.assert_near(x, y, message="failure message")]):
        out = array_ops.identity(x)
        self.evaluate(out)

  def test_returns_none_with_eager(self):
    # [FUNCTION DOC]
    # Test: in eager mode a passing near assertion returns nothing (null value).
    with context.eager_mode():
      t1 = constant_op.constant([1., 2.])
      t2 = constant_op.constant([1., 2.])
      x = check_ops.assert_near(t1, t2)
      assert x is None

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_complex(self):
    # [FUNCTION DOC]
    # Test: complex constants (1+0.1i and 1.1+0.1i) pass with absolute tolerance 0 and relative
    # tolerance 0.5; evaluation succeeds in graph and eager modes.
    x = constant_op.constant(1. + 0.1j, name="x")
    y = constant_op.constant(1.1 + 0.1j, name="y")
    with ops.control_dependencies([
        check_ops.assert_near(
            x, y, atol=0., rtol=0.5, message="failure message")
    ]):
      out = array_ops.identity(x)
      self.evaluate(out)


class AssertLessTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the strict less-than assertion: equal and greater inputs fail, smaller inputs
  # pass, broadcasting, empty tensors, eager return value and static detection in graph mode.

  @test_util.run_in_graph_and_eager_modes
  @test_util.run_deprecated_v1
  def test_raises_when_equal(self):
    # [FUNCTION DOC]
    # Test: comparing a tensor with itself fails because less-than is strict.
    # Expects an op error matching "failure message" followed by "x < y did not hold".
    # Deprecated-v1 mode, run in graph and eager modes.
    small = constant_op.constant([1, 2], name="small")
    with self.assertRaisesOpError(  # pylint:disable=g-error-prone-assert-raises
        "failure message.*\n*.* x < y did not hold"):
      with ops.control_dependencies(
          [check_ops.assert_less(
              small, small, message="failure message")]):
        out = array_ops.identity(small)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  @test_util.run_deprecated_v1
  def test_raises_when_greater(self):
    # [FUNCTION DOC]
    # Test: asserting big < small fails; expects an op error containing "x < y did not hold".
    small = constant_op.constant([1, 2], name="small")
    big = constant_op.constant([3, 4], name="big")
    with self.assertRaisesOpError(  # pylint:disable=g-error-prone-assert-raises
        "x < y did not hold"):
      with ops.control_dependencies([check_ops.assert_less(big, small)]):
        out = array_ops.identity(small)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_less(self):
    # [FUNCTION DOC]
    # Test: [3,1] < [4,2] passes; guarded identity evaluates without error.
    small = constant_op.constant([3, 1], name="small")
    big = constant_op.constant([4, 2], name="big")
    with ops.control_dependencies([check_ops.assert_less(small, big)]):
      out = array_ops.identity(small)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_less_and_broadcastable_shapes(self):
    # [FUNCTION DOC]
    # Test: a length-1 tensor [1] is less than a length-2 tensor [3,2] after broadcasting; passes.
    small = constant_op.constant([1], name="small")
    big = constant_op.constant([3, 2], name="big")
    with ops.control_dependencies([check_ops.assert_less(small, big)]):
      out = array_ops.identity(small)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_raises_when_less_but_non_broadcastable_shapes(self):
    # [FUNCTION DOC]
    # Test: shapes of length 3 and 2 cannot broadcast, so the assertion fails even though values are
    # smaller. Either a value error or an invalid-argument error is accepted (graph vs. eager).
    small = constant_op.constant([1, 1, 1], name="small")
    big = constant_op.constant([3, 2], name="big")
    # The exception in eager and non-eager mode is different because
    # eager mode relies on shape check done as part of the C++ op, while
    # graph mode does shape checks when creating the `Operation` instance.
    with self.assertRaisesIncompatibleShapesError(
        (ValueError, errors.InvalidArgumentError)):
      with ops.control_dependencies([check_ops.assert_less(small, big)]):
        out = array_ops.identity(small)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_both_empty(self):
    # [FUNCTION DOC]
    # Test: two empty tensors pass the less-than assertion.
    larry = constant_op.constant([])
    curly = constant_op.constant([])
    with ops.control_dependencies([check_ops.assert_less(larry, curly)]):
      out = array_ops.identity(larry)
    self.evaluate(out)

  def test_returns_none_with_eager(self):
    # [FUNCTION DOC]
    # Test: in eager mode a passing less-than assertion returns nothing (null value).
    with context.eager_mode():
      t1 = constant_op.constant([1, 2])
      t2 = constant_op.constant([3, 4])
      x = check_ops.assert_less(t1, t2)
      assert x is None

  def test_static_check_in_graph_mode(self):
    # [FUNCTION DOC]
    # Test: in graph mode, 1 < 1 on Python integers is decided statically.
    # Expects an invalid-argument error matching "Custom error message" at the call.
    with ops.Graph().as_default():
      with self.assertRaisesRegex(  # pylint:disable=g-error-prone-assert-raises
          errors.InvalidArgumentError, "Custom error message"):
        check_ops.assert_less(1, 1, message="Custom error message")


class AssertLessEqualTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the less-than-or-equal assertion: equal and smaller pass, greater fails,
  # broadcasting, empty tensors and static detection in graph mode.

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_equal(self):
    # [FUNCTION DOC]
    # Test: a tensor is less-or-equal to itself; guarded identity evaluates without error.
    small = constant_op.constant([1, 2], name="small")
    with ops.control_dependencies(
        [check_ops.assert_less_equal(small, small)]):
      out = array_ops.identity(small)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  @test_util.run_deprecated_v1
  def test_raises_when_greater(self):
    # [FUNCTION DOC]
    # Test: asserting big <= small fails with message "fail"; expects an op error matching "fail".
    # Deprecated-v1 mode, run in graph and eager modes.
    small = constant_op.constant([1, 2], name="small")
    big = constant_op.constant([3, 4], name="big")
    with self.assertRaisesOpError(  # pylint:disable=g-error-prone-assert-raises
        "fail"):
      with ops.control_dependencies(
          [check_ops.assert_less_equal(
              big, small, message="fail")]):
        out = array_ops.identity(small)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_less_equal(self):
    # [FUNCTION DOC]
    # Test: [1,2] <= [3,2] passes (one smaller, one equal element).
    small = constant_op.constant([1, 2], name="small")
    big = constant_op.constant([3, 2], name="big")
    with ops.control_dependencies([check_ops.assert_less_equal(small, big)]):
      out = array_ops.identity(small)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_less_equal_and_broadcastable_shapes(self):
    # [FUNCTION DOC]
    # Test: length-1 tensor [1] <= length-2 tensor [3,1] passes after broadcasting.
    small = constant_op.constant([1], name="small")
    big = constant_op.constant([3, 1], name="big")
    with ops.control_dependencies([check_ops.assert_less_equal(small, big)]):
      out = array_ops.identity(small)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_raises_when_less_equal_but_non_broadcastable_shapes(self):
    # [FUNCTION DOC]
    # Test: shapes of length 2 and 3 cannot broadcast, so the assertion fails.
    # Expects an invalid-argument or value error whose message is either "Incompatible shapes: [2]
    # vs. [3]" or "Dimensions must be equal, but are 2 and 3" (graph vs. eager wording).
    small = constant_op.constant([3, 1], name="small")
    big = constant_op.constant([1, 1, 1], name="big")
    # The exception in eager and non-eager mode is different because
    # eager mode relies on shape check done as part of the C++ op, while
    # graph mode does shape checks when creating the `Operation` instance.
    with self.assertRaisesRegex(  # pylint:disable=g-error-prone-assert-raises
        (errors.InvalidArgumentError, ValueError),
        (r"Incompatible shapes: \[2\] vs. \[3\]|"
         r"Dimensions must be equal, but are 2 and 3")):
      with ops.control_dependencies(
          [check_ops.assert_less_equal(small, big)]):
        out = array_ops.identity(small)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_both_empty(self):
    # [FUNCTION DOC]
    # Test: two empty tensors pass the less-or-equal assertion.
    larry = constant_op.constant([])
    curly = constant_op.constant([])
    with ops.control_dependencies(
        [check_ops.assert_less_equal(larry, curly)]):
      out = array_ops.identity(larry)
    self.evaluate(out)

  def test_static_check_in_graph_mode(self):
    # [FUNCTION DOC]
    # Test: in graph mode, 1 <= 0 on Python integers is decided statically.
    # Expects an invalid-argument error matching "Custom error message" at the call.
    with ops.Graph().as_default():
      with self.assertRaisesRegex(  # pylint:disable=g-error-prone-assert-raises
          errors.InvalidArgumentError, "Custom error message"):
        check_ops.assert_less_equal(1, 0, message="Custom error message")


class AssertGreaterTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the strict greater-than assertion: equal and smaller inputs fail, larger inputs
  # pass, broadcasting, empty tensors and static detection in graph mode.

  @test_util.run_in_graph_and_eager_modes
  @test_util.run_deprecated_v1
  def test_raises_when_equal(self):
    # [FUNCTION DOC]
    # Test: a tensor is not strictly greater than itself; expects an op error matching "fail".
    # Deprecated-v1 mode, run in graph and eager modes.
    small = constant_op.constant([1, 2], name="small")
    with self.assertRaisesOpError(  # pylint:disable=g-error-prone-assert-raises
        "fail"):
      with ops.control_dependencies(
          [check_ops.assert_greater(
              small, small, message="fail")]):
        out = array_ops.identity(small)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  @test_util.run_deprecated_v1
  def test_raises_when_less(self):
    # [FUNCTION DOC]
    # Test: asserting small > big fails; expects an op error containing "x > y did not hold".
    small = constant_op.constant([1, 2], name="small")
    big = constant_op.constant([3, 4], name="big")
    with self.assertRaisesOpError(  # pylint:disable=g-error-prone-assert-raises
        "x > y did not hold"):
      with ops.control_dependencies([check_ops.assert_greater(small, big)]):
        out = array_ops.identity(big)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_greater(self):
    # [FUNCTION DOC]
    # Test: [4,2] > [3,1] passes; guarded identity evaluates without error.
    small = constant_op.constant([3, 1], name="small")
    big = constant_op.constant([4, 2], name="big")
    with ops.control_dependencies([check_ops.assert_greater(big, small)]):
      out = array_ops.identity(small)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_greater_and_broadcastable_shapes(self):
    # [FUNCTION DOC]
    # Test: length-2 tensor [3,2] > length-1 tensor [1] passes after broadcasting.
    small = constant_op.constant([1], name="small")
    big = constant_op.constant([3, 2], name="big")
    with ops.control_dependencies([check_ops.assert_greater(big, small)]):
      out = array_ops.identity(small)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_raises_when_greater_but_non_broadcastable_shapes(self):
    # [FUNCTION DOC]
    # Test: shapes of length 2 and 3 cannot broadcast, so the assertion fails.
    # Expects an invalid-argument or value error whose message is either "Incompatible shapes: [2]
    # vs. [3]" or "Dimensions must be equal, but are 2 and 3".
    small = constant_op.constant([1, 1, 1], name="small")
    big = constant_op.constant([3, 2], name="big")
    # The exception in eager and non-eager mode is different because
    # eager mode relies on shape check done as part of the C++ op, while
    # graph mode does shape checks when creating the `Operation` instance.
    with self.assertRaisesRegex(  # pylint:disable=g-error-prone-assert-raises
        (errors.InvalidArgumentError, ValueError),
        (r"Incompatible shapes: \[2\] vs. \[3\]|"
         r"Dimensions must be equal, but are 2 and 3")):
      with ops.control_dependencies([check_ops.assert_greater(big, small)]):
        out = array_ops.identity(small)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_both_empty(self):
    # [FUNCTION DOC]
    # Test: two empty tensors pass the greater-than assertion.
    larry = constant_op.constant([])
    curly = constant_op.constant([])
    with ops.control_dependencies([check_ops.assert_greater(larry, curly)]):
      out = array_ops.identity(larry)
    self.evaluate(out)

  def test_static_check_in_graph_mode(self):
    # [FUNCTION DOC]
    # Test: in graph mode, 0 > 1 on Python integers is decided statically.
    # Expects an invalid-argument error matching "Custom error message" at the call.
    with ops.Graph().as_default():
      with self.assertRaisesRegex(  # pylint:disable=g-error-prone-assert-raises
          errors.InvalidArgumentError, "Custom error message"):
        check_ops.assert_greater(0, 1, message="Custom error message")


class AssertGreaterEqualTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the greater-than-or-equal assertion: equal and larger pass, smaller fails,
  # broadcasting, empty tensors and static detection in graph mode.

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_equal(self):
    # [FUNCTION DOC]
    # Test: a tensor is greater-or-equal to itself; guarded identity evaluates without error.
    small = constant_op.constant([1, 2], name="small")
    with ops.control_dependencies(
        [check_ops.assert_greater_equal(small, small)]):
      out = array_ops.identity(small)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  @test_util.run_deprecated_v1
  def test_raises_when_less(self):
    # [FUNCTION DOC]
    # Test: asserting small >= big fails with message "fail"; expects an op error matching "fail".
    # Deprecated-v1 mode, run in graph and eager modes.
    small = constant_op.constant([1, 2], name="small")
    big = constant_op.constant([3, 4], name="big")
    with self.assertRaisesOpError(  # pylint:disable=g-error-prone-assert-raises
        "fail"):
      with ops.control_dependencies(
          [check_ops.assert_greater_equal(
              small, big, message="fail")]):
        out = array_ops.identity(small)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_greater_equal(self):
    # [FUNCTION DOC]
    # Test: [3,2] >= [1,2] passes (one larger, one equal element).
    small = constant_op.constant([1, 2], name="small")
    big = constant_op.constant([3, 2], name="big")
    with ops.control_dependencies(
        [check_ops.assert_greater_equal(big, small)]):
      out = array_ops.identity(small)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_greater_equal_and_broadcastable_shapes(self):
    # [FUNCTION DOC]
    # Test: length-2 tensor [3,1] >= length-1 tensor [1] passes after broadcasting.
    small = constant_op.constant([1], name="small")
    big = constant_op.constant([3, 1], name="big")
    with ops.control_dependencies(
        [check_ops.assert_greater_equal(big, small)]):
      out = array_ops.identity(small)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_raises_when_less_equal_but_non_broadcastable_shapes(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name says "less_equal" but the test exercises the greater-equal assertion,
    # and the op names are swapped (the length-3 tensor is named "big", the length-2 one "small").
    # Test: shapes of length 3 and 2 cannot broadcast, so the assertion fails.
    # Expects an invalid-argument or value error with message "Incompatible shapes: [2] vs. [3]" or
    # "Dimensions must be equal, but are 2 and 3".
    small = constant_op.constant([1, 1, 1], name="big")
    big = constant_op.constant([3, 1], name="small")
    # The exception in eager and non-eager mode is different because
    # eager mode relies on shape check done as part of the C++ op, while
    # graph mode does shape checks when creating the `Operation` instance.
    with self.assertRaisesRegex(  # pylint:disable=g-error-prone-assert-raises
        (errors.InvalidArgumentError, ValueError),
        (r"Incompatible shapes: \[2\] vs. \[3\]|"
         r"Dimensions must be equal, but are 2 and 3")):
      with ops.control_dependencies(
          [check_ops.assert_greater_equal(big, small)]):
        out = array_ops.identity(small)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_both_empty(self):
    # [FUNCTION DOC]
    # Test: two empty tensors pass the greater-or-equal assertion.
    larry = constant_op.constant([])
    curly = constant_op.constant([])
    with ops.control_dependencies(
        [check_ops.assert_greater_equal(larry, curly)]):
      out = array_ops.identity(larry)
    self.evaluate(out)

  def test_static_check_in_graph_mode(self):
    # [FUNCTION DOC]
    # Test: in graph mode, 0 >= 1 on Python integers is decided statically.
    # Expects an invalid-argument error matching "Custom error message" at the call.
    with ops.Graph().as_default():
      with self.assertRaisesRegex(  # pylint:disable=g-error-prone-assert-raises
          errors.InvalidArgumentError, "Custom error message"):
        check_ops.assert_greater_equal(0, 1, message="Custom error message")


class AssertNegativeTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the strictly-negative assertion: negative values pass; positive and zero fail;
  # empty tensors pass vacuously; Python scalars are checked statically in graph mode.

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_negative(self):
    # [FUNCTION DOC]
    # Test: [-1,-2] passes the negative assertion; guarded identity evaluates without error.
    frank = constant_op.constant([-1, -2], name="frank")
    with ops.control_dependencies([check_ops.assert_negative(frank)]):
      out = array_ops.identity(frank)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  @test_util.run_deprecated_v1
  def test_raises_when_positive(self):
    # [FUNCTION DOC]
    # Test: [1,2] with message "fail" fails; expects an op error matching "fail".
    # Deprecated-v1 mode, run in graph and eager modes.
    doug = constant_op.constant([1, 2], name="doug")
    with self.assertRaisesOpError(  # pylint:disable=g-error-prone-assert-raises
        "fail"):
      with ops.control_dependencies(
          [check_ops.assert_negative(
              doug, message="fail")]):
        out = array_ops.identity(doug)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  @test_util.run_deprecated_v1
  def test_raises_when_zero(self):
    # [FUNCTION DOC]
    # Test: zero is not strictly negative; expects an op error containing "x < 0 did not hold".
    claire = constant_op.constant([0], name="claire")
    with self.assertRaisesOpError(  # pylint:disable=g-error-prone-assert-raises
        "x < 0 did not hold"):
      with ops.control_dependencies([check_ops.assert_negative(claire)]):
        out = array_ops.identity(claire)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_empty_tensor_doesnt_raise(self):
    # [FUNCTION DOC]
    # Test: an empty tensor passes because the "all elements negative" condition is vacuously true.
    # A tensor is negative when it satisfies:
    #   For every element x_i in x, x_i < 0
    # and an empty tensor has no elements, so this is trivially satisfied.
    # This is standard set theory.
    empty = constant_op.constant([], name="empty")
    with ops.control_dependencies([check_ops.assert_negative(empty)]):
      out = array_ops.identity(empty)
    self.evaluate(out)

  def test_static_check_in_graph_mode(self):
    # [FUNCTION DOC]
    # Test: in graph mode the Python integer 1 is statically rejected with an invalid-argument
    # error matching "Custom error message".
    with ops.Graph().as_default():
      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  "Custom error message"):
        check_ops.assert_negative(1, message="Custom error message")


# pylint:disable=g-error-prone-assert-raises
class AssertPositiveTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the strictly-positive assertion: positive values pass; negative and zero fail;
  # empty tensors pass vacuously; Python scalars are checked statically in graph mode.

  @test_util.run_in_graph_and_eager_modes
  @test_util.run_deprecated_v1
  def test_raises_when_negative(self):
    # [FUNCTION DOC]
    # Test: [-1,-2] with message "fail" fails; expects an op error matching "fail".
    # Deprecated-v1 mode, run in graph and eager modes.
    freddie = constant_op.constant([-1, -2], name="freddie")
    with self.assertRaisesOpError("fail"):
      with ops.control_dependencies(
          [check_ops.assert_positive(
              freddie, message="fail")]):
        out = array_ops.identity(freddie)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_positive(self):
    # [FUNCTION DOC]
    # Test: [1,2] passes the positive assertion; guarded identity evaluates without error.
    remmy = constant_op.constant([1, 2], name="remmy")
    with ops.control_dependencies([check_ops.assert_positive(remmy)]):
      out = array_ops.identity(remmy)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  @test_util.run_deprecated_v1
  def test_raises_when_zero(self):
    # [FUNCTION DOC]
    # Test: zero is not strictly positive; expects an op error containing "x > 0 did not hold".
    meechum = constant_op.constant([0], name="meechum")
    with self.assertRaisesOpError("x > 0 did not hold"):
      with ops.control_dependencies([check_ops.assert_positive(meechum)]):
        out = array_ops.identity(meechum)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_empty_tensor_doesnt_raise(self):
    # [FUNCTION DOC]
    # Test: an empty tensor passes because the "all elements positive" condition is vacuously true.
    # A tensor is positive when it satisfies:
    #   For every element x_i in x, x_i > 0
    # and an empty tensor has no elements, so this is trivially satisfied.
    # This is standard set theory.
    empty = constant_op.constant([], name="empty")
    with ops.control_dependencies([check_ops.assert_positive(empty)]):
      out = array_ops.identity(empty)
    self.evaluate(out)

  def test_static_check_in_graph_mode(self):
    # [FUNCTION DOC]
    # Test: in graph mode the Python integer -1 is statically rejected with an invalid-argument
    # error matching "Custom error message".
    with ops.Graph().as_default():
      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  "Custom error message"):
        check_ops.assert_positive(-1, message="Custom error message")


class EnsureShapeTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the ensure-shape op, which asserts and merges a tensor's static shape and
  # checks it again at run time. Static tests inspect inferred shapes; dynamic tests feed values
  # through placeholders in a session; also checks gradient pass-through. All tests are
  # deprecated-v1 graph-mode tests.

  # Static shape inference
  @test_util.run_deprecated_v1
  def testStaticShape(self):
    # [FUNCTION DOC]
    # Test: ensure-shape on a placeholder of unknown shape yields static shape (3,3,3).
    placeholder = array_ops.placeholder(dtypes.int32)
    ensure_shape_op = check_ops.ensure_shape(placeholder, (3, 3, 3))
    self.assertEqual(ensure_shape_op.get_shape(), (3, 3, 3))

  @test_util.run_deprecated_v1
  def testStaticShape_MergesShapes(self):
    # [FUNCTION DOC]
    # Test: a partially known input shape (unknown, unknown, 3) merged with (5,4,unknown) yields
    # static shape (5,4,3).
    placeholder = array_ops.placeholder(dtypes.int32, shape=(None, None, 3))
    ensure_shape_op = check_ops.ensure_shape(placeholder, (5, 4, None))
    self.assertEqual(ensure_shape_op.get_shape(), (5, 4, 3))

  @test_util.run_deprecated_v1
  def testStaticShape_RaisesErrorWhenRankIncompatible(self):
    # [FUNCTION DOC]
    # Test: requiring a rank-2 shape for a rank-3 placeholder raises a value error at graph build.
    placeholder = array_ops.placeholder(dtypes.int32, shape=(None, None, 3))
    with self.assertRaises(ValueError):
      check_ops.ensure_shape(placeholder, (2, 3))

  @test_util.run_deprecated_v1
  def testStaticShape_RaisesErrorWhenDimIncompatible(self):
    # [FUNCTION DOC]
    # Test: requiring (2,2,4) when the last known dimension is 3 raises a value error at graph build.
    placeholder = array_ops.placeholder(dtypes.int32, shape=(None, None, 3))
    with self.assertRaises(ValueError):
      check_ops.ensure_shape(placeholder, (2, 2, 4))

  @test_util.run_deprecated_v1
  def testStaticShape_CanSetUnknownShape(self):
    # [FUNCTION DOC]
    # Test: requiring a null (fully unknown) shape on a derived tensor keeps the static shape
    # unknown; asserts the resulting shape equals the null value.
    placeholder = array_ops.placeholder(dtypes.int32)
    derived = placeholder / 3
    ensure_shape_op = check_ops.ensure_shape(derived, None)
    self.assertEqual(ensure_shape_op.get_shape(), None)

  # Dynamic shape check
  @test_util.run_deprecated_v1
  @test_util.disable_xla(
      "b/123337890")  # Dynamic shapes not supported now with XLA
  def testEnsuresDynamicShape_RaisesError(self):
    # [FUNCTION DOC]
    # Test: a fed value of shape (2,1) does not match the required (3,3,3) at run time.
    # Skipped under XLA (dynamic shapes unsupported, bug reference in the decorator). Expects an
    # invalid-argument error saying the shape of tensor "MyDivide" [2,1] is not compatible with
    # expected shape [3,3,3].
    placeholder = array_ops.placeholder(dtypes.int32)
    derived = math_ops.divide(placeholder, 3, name="MyDivide")
    derived = check_ops.ensure_shape(derived, (3, 3, 3))
    feed_val = [[1], [2]]
    with self.cached_session() as sess:
      with self.assertRaisesWithPredicateMatch(
          errors.InvalidArgumentError,
          r"Shape of tensor MyDivide \[2,1\] is not compatible with "
          r"expected shape \[3,3,3\]."):
        sess.run(derived, feed_dict={placeholder: feed_val})

  @test_util.run_deprecated_v1
  @test_util.disable_xla(
      "b/123337890")  # Dynamic shapes not supported now with XLA
  def testEnsuresDynamicShape_RaisesErrorDimUnknown(self):
    # [FUNCTION DOC]
    # Test: requirement (unknown, unknown, 3) with a fed value of shape (2,1) fails at run time.
    # Skipped under XLA. Expects an invalid-argument error mentioning expected shape [?,?,3].
    placeholder = array_ops.placeholder(dtypes.int32)
    derived = placeholder / 3
    derived = check_ops.ensure_shape(derived, (None, None, 3))
    feed_val = [[1], [2]]
    with self.cached_session() as sess:
      with self.assertRaisesWithPredicateMatch(
          errors.InvalidArgumentError,
          r"Shape of tensor [A-Za-z_]* \[2,1\] is not compatible with "
          r"expected shape \[\?,\?,3\]."):
        sess.run(derived, feed_dict={placeholder: feed_val})

  @test_util.run_deprecated_v1
  def testEnsuresDynamicShape(self):
    # [FUNCTION DOC]
    # Test: a fed value of shape (2,1) satisfies required shape (2,1); the session run succeeds.
    placeholder = array_ops.placeholder(dtypes.int32)
    derived = placeholder / 3
    derived = check_ops.ensure_shape(derived, (2, 1))
    feed_val = [[1], [2]]
    with self.cached_session() as sess:
      sess.run(derived, feed_dict={placeholder: feed_val})

  @test_util.run_deprecated_v1
  def testEnsuresDynamicShape_WithUnknownDims(self):
    # [FUNCTION DOC]
    # Test: required shape (unknown, unknown) accepts a fed value of shape (2,1); run succeeds.
    placeholder = array_ops.placeholder(dtypes.int32)
    derived = placeholder / 3
    derived = check_ops.ensure_shape(derived, (None, None))
    feed_val = [[1], [2]]
    with self.cached_session() as sess:
      sess.run(derived, feed_dict={placeholder: feed_val})

  @test_util.run_deprecated_v1
  def testGradient(self):
    # [FUNCTION DOC]
    # Test: the gradient flows through ensure-shape unchanged.
    # Computes the gradient of the op output with respect to a float placeholder, feeds
    # [[4.0],[-1.0]] and asserts the gradient values equal all ones of the same shape.
    placeholder = array_ops.placeholder(dtypes.float32)
    derived = check_ops.ensure_shape(placeholder, (None, None))
    gradient = gradients.gradients(derived, placeholder)

    feed_val = [[4.0], [-1.0]]
    with self.cached_session() as sess:
      gradient_values, = sess.run(gradient, feed_dict={placeholder: feed_val})

    expected = [[1.0], [1.0]]
    self.assertAllEqual(gradient_values, expected)


class EnsureShapeBenchmark(test.Benchmark):
  # [CLASS DOC]
  # Benchmark class measuring the overhead of the ensure-shape op versus identity ops.
  # Methods named with the benchmark prefix are discovered by the benchmark runner; each reports a
  # median per-run wall time in microseconds. Graph optimizations are disabled so that the ops
  # are not folded away.

  def _grappler_all_off_config(self):
    # [FUNCTION DOC]
    # HELPER: Builds the session configuration used by this benchmark's run helper.
    # Returns a session configuration proto with the optimizer level lowered, model pruning
    # disabled, and constant folding, layout, arithmetic and dependency optimizations turned off,
    # so measured ops are executed as written. Takes no arguments and has no side effects.
    config = config_pb2.ConfigProto()
    off = rewriter_config_pb2.RewriterConfig.OFF
    config.graph_options.optimizer_options.opt_level = -1
    config.graph_options.rewrite_options.disable_model_pruning = 1
    config.graph_options.rewrite_options.constant_folding = off
    config.graph_options.rewrite_options.layout_optimizer = off
    config.graph_options.rewrite_options.arithmetic_optimization = off
    config.graph_options.rewrite_options.dependency_optimization = off
    return config

  def _run(self, op, feed_dict=None, num_iters=5000, name=None, **kwargs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the local variables are called "mean_time"/"mean_us", but the value is the
    # median of the per-iteration durations (and a commented-out line shows an older mean).
    # HELPER: Timing harness shared by all benchmark methods in this class.
    # Inputs: op (the tensor or op to run), feed_dict (optional mapping from placeholders to
    # values, default none), num_iters (number of timed runs, default 5000), name (benchmark
    # name reported), extra keyword arguments (forwarded as extra report fields).
    # Creates a session with the all-optimizations-off configuration, runs the op 5 times to warm
    # up, then times each of num_iters runs with the wall clock, and takes the median in
    # microseconds. Side effect: reports the benchmark (name, wall time, extras) to the framework.
    # Returns nothing.
    config = self._grappler_all_off_config()
    with session.Session(config=config) as sess:
      deltas = []
      # Warm up the session
      for _ in range(5):
        sess.run(op, feed_dict=feed_dict)
      for _ in range(num_iters):
        start = time.time()
        sess.run(op, feed_dict=feed_dict)
        end = time.time()
        deltas.append(end - start)
      mean_time = np.median(deltas)
      mean_us = mean_time * 1e6
      # mean_us = (end - start) * 1e6 / num_iters
      self.report_benchmark(
          name=name,
          wall_time=mean_us,
          extras=kwargs,
      )

  def benchmark_const_op(self):
    # [FUNCTION DOC]
    # Benchmark: baseline cost of running a single identity op over a random-normal tensor of
    # shape (3,3,100); the per-call overhead of running a session is expected to dominate.
    # Reports under the name "SingleConstOp".
    # In this case, we expect that the overhead of a `session.run` call
    # far outweighs the time taken to execute the op...
    shape = (3, 3, 100)
    input_op = random_ops.random_normal(shape)
    self._run(array_ops.identity(input_op), name="SingleConstOp")

  def benchmark_single_ensure_op(self):
    # [FUNCTION DOC]
    # Benchmark: cost of one ensure-shape op on a random-normal tensor of shape (3,3,100); the
    # session call overhead is expected to dominate. Reports under "SingleEnsureShapeOp".
    # In this case, we expect that the overhead of a `session.run` call
    # far outweighs the time taken to execute the op...
    shape = (3, 3, 100)
    input_op = random_ops.random_normal(shape)
    ensure_shape_op = check_ops.ensure_shape(input_op, shape)
    self._run(ensure_shape_op, name="SingleEnsureShapeOp")

  def _apply_n_times(self, op, target, n=1000):
    # [FUNCTION DOC]
    # HELPER: Builds long op chains for the n-ops benchmarks.
    # Inputs: op (a callable mapping a tensor to a tensor), target (the starting tensor), n (number
    # of repetitions, default 1000). Applies op to the running value n times and returns the final
    # tensor. Only graph construction; no session is run.
    for _ in range(n):
      target = op(target)
    return target

  def benchmark_n_ops(self):
    # [FUNCTION DOC]
    # Benchmark: a chain of 1000 identity ops on a random-normal vector of length 1000.
    # Reports under "NIdentityOps_1000"; serves as the baseline for the ensure-shape chain.
    shape = (1000,)
    input_op = random_ops.random_normal(shape)
    n_ops = self._apply_n_times(array_ops.identity, input_op)
    self._run(n_ops, name="NIdentityOps_1000")

  def benchmark_n_ensure_ops(self):
    # [FUNCTION DOC]
    # Benchmark: a chain of 1000 repetitions of (identity followed by ensure-shape with the vector's
    # shape) on a random-normal vector of length 1000. Reports under "NEnsureShapeAndIdentityOps_1000".
    shape = (1000,)
    input_op = random_ops.random_normal(shape)
    n_ensure_ops = self._apply_n_times(
        lambda x: check_ops.ensure_shape(array_ops.identity(x), shape),
        input_op)
    self._run(n_ensure_ops, name="NEnsureShapeAndIdentityOps_1000")


class AssertRankTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the exact-rank assertion. For each scenario there is a static-rank variant
  # (constant tensor; errors are value errors at graph construction) and a dynamic-rank variant
  # (placeholder; errors are op errors at run time, deprecated-v1 graph mode). Also tests that the
  # desired rank must be an integer scalar.

  @test_util.run_in_graph_and_eager_modes
  def test_rank_zero_tensor_raises_if_rank_too_small_static_rank(self):
    # [FUNCTION DOC]
    # Test: scalar constant with desired rank 1 and message "fail" raises a value error matching
    # "fail" and "must have rank 1".
    tensor = constant_op.constant(1, name="my_tensor")
    desired_rank = 1
    with self.assertRaisesRegex(ValueError, "fail.*must have rank 1"):
      with ops.control_dependencies(
          [check_ops.assert_rank(
              tensor, desired_rank, message="fail")]):
        self.evaluate(array_ops.identity(tensor))

  @test_util.run_deprecated_v1
  def test_rank_zero_tensor_raises_if_rank_too_small_dynamic_rank(self):
    # [FUNCTION DOC]
    # Test: float placeholder fed a scalar with desired rank 1 raises an op error matching "fail",
    # the tensor name "my_tensor" and "rank" when evaluated.
    with self.cached_session():
      tensor = array_ops.placeholder(dtypes.float32, name="my_tensor")
      desired_rank = 1
      with ops.control_dependencies(
          [check_ops.assert_rank(
              tensor, desired_rank, message="fail")]):
        with self.assertRaisesOpError("fail.*my_tensor.*rank"):
          array_ops.identity(tensor).eval(feed_dict={tensor: 0})

  @test_util.run_in_graph_and_eager_modes
  def test_rank_zero_tensor_doesnt_raise_if_rank_just_right_static_rank(self):
    # [FUNCTION DOC]
    # Test: scalar constant with desired rank 0 passes; guarded identity evaluates without error.
    tensor = constant_op.constant(1, name="my_tensor")
    desired_rank = 0
    with ops.control_dependencies(
        [check_ops.assert_rank(tensor, desired_rank)]):
      self.evaluate(array_ops.identity(tensor))

  @test_util.run_deprecated_v1
  def test_rank_zero_tensor_doesnt_raise_if_rank_just_right_dynamic_rank(self):
    # [FUNCTION DOC]
    # Test: placeholder fed a scalar with desired rank 0 passes at run time.
    with self.cached_session():
      tensor = array_ops.placeholder(dtypes.float32, name="my_tensor")
      desired_rank = 0
      with ops.control_dependencies(
          [check_ops.assert_rank(tensor, desired_rank)]):
        array_ops.identity(tensor).eval(feed_dict={tensor: 0})

  @test_util.run_in_graph_and_eager_modes
  def test_rank_one_tensor_raises_if_rank_too_large_static_rank(self):
    # [FUNCTION DOC]
    # Test: vector constant with desired rank 0 raises a value error mentioning "rank".
    tensor = constant_op.constant([1, 2], name="my_tensor")
    desired_rank = 0
    with self.assertRaisesRegex(ValueError, "rank"):
      with ops.control_dependencies(
          [check_ops.assert_rank(tensor, desired_rank)]):
        self.evaluate(array_ops.identity(tensor))

  @test_util.run_deprecated_v1
  def test_rank_one_tensor_raises_if_rank_too_large_dynamic_rank(self):
    # [FUNCTION DOC]
    # Test: placeholder fed a vector with desired rank 0 raises an op error matching "my_tensor"
    # and "rank" when evaluated.
    with self.cached_session():
      tensor = array_ops.placeholder(dtypes.float32, name="my_tensor")
      desired_rank = 0
      with ops.control_dependencies(
          [check_ops.assert_rank(tensor, desired_rank)]):
        with self.assertRaisesOpError("my_tensor.*rank"):
          array_ops.identity(tensor).eval(feed_dict={tensor: [1, 2]})

  @test_util.run_in_graph_and_eager_modes
  def test_rank_one_tensor_doesnt_raise_if_rank_just_right_static_rank(self):
    # [FUNCTION DOC]
    # Test: vector constant with desired rank 1 passes.
    tensor = constant_op.constant([1, 2], name="my_tensor")
    desired_rank = 1
    with ops.control_dependencies(
        [check_ops.assert_rank(tensor, desired_rank)]):
      self.evaluate(array_ops.identity(tensor))

  @test_util.run_deprecated_v1
  def test_rank_one_tensor_doesnt_raise_if_rank_just_right_dynamic_rank(self):
    # [FUNCTION DOC]
    # Test: placeholder fed a vector with desired rank 1 passes at run time.
    with self.cached_session():
      tensor = array_ops.placeholder(dtypes.float32, name="my_tensor")
      desired_rank = 1
      with ops.control_dependencies(
          [check_ops.assert_rank(tensor, desired_rank)]):
        array_ops.identity(tensor).eval(feed_dict={tensor: [1, 2]})

  @test_util.run_in_graph_and_eager_modes
  def test_rank_one_tensor_raises_if_rank_too_small_static_rank(self):
    # [FUNCTION DOC]
    # Test: vector constant with desired rank 2 raises a value error mentioning "rank".
    tensor = constant_op.constant([1, 2], name="my_tensor")
    desired_rank = 2
    with self.assertRaisesRegex(ValueError, "rank"):
      with ops.control_dependencies(
          [check_ops.assert_rank(tensor, desired_rank)]):
        self.evaluate(array_ops.identity(tensor))

  @test_util.run_deprecated_v1
  def test_rank_one_tensor_raises_if_rank_too_small_dynamic_rank(self):
    # [FUNCTION DOC]
    # Test: placeholder fed a vector with desired rank 2 raises an op error matching "my_tensor"
    # and "rank" when evaluated.
    with self.cached_session():
      tensor = array_ops.placeholder(dtypes.float32, name="my_tensor")
      desired_rank = 2
      with ops.control_dependencies(
          [check_ops.assert_rank(tensor, desired_rank)]):
        with self.assertRaisesOpError("my_tensor.*rank"):
          array_ops.identity(tensor).eval(feed_dict={tensor: [1, 2]})

  @test_util.run_in_graph_and_eager_modes
  def test_raises_if_rank_is_not_scalar_static(self):
    # [FUNCTION DOC]
    # Test: a desired rank given as an empty numpy array (not a scalar) raises a value error
    # "Rank must be a scalar" at the call (graph and eager).
    tensor = constant_op.constant([1, 2], name="my_tensor")
    with self.assertRaisesRegex(ValueError, "Rank must be a scalar"):
      check_ops.assert_rank(tensor, np.array([], dtype=np.int32))

  @test_util.run_deprecated_v1
  def test_raises_if_rank_is_not_scalar_dynamic(self):
    # [FUNCTION DOC]
    # Test: desired rank supplied through an int placeholder fed a two-element list raises an op
    # error "Rank must be a scalar" when evaluated (deprecated-v1 graph mode).
    with self.cached_session():
      tensor = constant_op.constant(
          [1, 2], dtype=dtypes.float32, name="my_tensor")
      rank_tensor = array_ops.placeholder(dtypes.int32, name="rank_tensor")
      with self.assertRaisesOpError("Rank must be a scalar"):
        with ops.control_dependencies(
            [check_ops.assert_rank(tensor, rank_tensor)]):
          array_ops.identity(tensor).eval(feed_dict={rank_tensor: [1, 2]})

  @test_util.run_in_graph_and_eager_modes
  def test_raises_if_rank_is_not_integer_static(self):
    # [FUNCTION DOC]
    # Test: a floating-point desired rank (0.5) raises a type error saying it must be of type int32
    # (graph and eager).
    tensor = constant_op.constant([1, 2], name="my_tensor")
    with self.assertRaisesRegex(TypeError, "must be of type tf.int32"):
      check_ops.assert_rank(tensor, .5)

  @test_util.run_deprecated_v1
  def test_raises_if_rank_is_not_integer_dynamic(self):
    # [FUNCTION DOC]
    # Test: a float32 placeholder used as the desired rank raises a type error saying it must be of
    # type int32 (deprecated-v1 graph mode; raised when the assertion is built).
    with self.cached_session():
      tensor = constant_op.constant(
          [1, 2], dtype=dtypes.float32, name="my_tensor")
      rank_tensor = array_ops.placeholder(dtypes.float32, name="rank_tensor")
      with self.assertRaisesRegex(TypeError, "must be of type tf.int32"):
        with ops.control_dependencies(
            [check_ops.assert_rank(tensor, rank_tensor)]):
          array_ops.identity(tensor).eval(feed_dict={rank_tensor: .5})


class AssertRankInTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the rank-in-set assertion (rank must be one of several allowed values), with
  # static and dynamic variants like the exact-rank tests, plus scalar and integer-type validation
  # of the allowed ranks.

  @test_util.run_in_graph_and_eager_modes
  def test_rank_zero_tensor_raises_if_rank_mismatch_static_rank(self):
    # [FUNCTION DOC]
    # Test: scalar constant with allowed ranks (1,2) and message "fail" raises a value error
    # matching "fail", "must have rank", "in", "1" and "2".
    tensor_rank0 = constant_op.constant(42, name="my_tensor")
    with self.assertRaisesRegex(ValueError, "fail.*must have rank.*in.*1.*2"):
      with ops.control_dependencies([
          check_ops.assert_rank_in(tensor_rank0, (1, 2), message="fail")]):
        self.evaluate(array_ops.identity(tensor_rank0))

  @test_util.run_deprecated_v1
  def test_rank_zero_tensor_raises_if_rank_mismatch_dynamic_rank(self):
    # [FUNCTION DOC]
    # Test: placeholder fed a scalar with allowed ranks (1,2) raises an op error matching "fail",
    # "my_tensor" and "rank" on evaluation.
    with self.cached_session():
      tensor_rank0 = array_ops.placeholder(dtypes.float32, name="my_tensor")
      with ops.control_dependencies([
          check_ops.assert_rank_in(tensor_rank0, (1, 2), message="fail")]):
        with self.assertRaisesOpError("fail.*my_tensor.*rank"):
          array_ops.identity(tensor_rank0).eval(feed_dict={tensor_rank0: 42.0})

  @test_util.run_in_graph_and_eager_modes
  def test_rank_zero_tensor_doesnt_raise_if_rank_matches_static_rank(self):
    # [FUNCTION DOC]
    # Test: scalar constant passes when rank 0 appears anywhere in the allowed set (three orderings).
    tensor_rank0 = constant_op.constant(42, name="my_tensor")
    for desired_ranks in ((0, 1, 2), (1, 0, 2), (1, 2, 0)):
      with ops.control_dependencies([
          check_ops.assert_rank_in(tensor_rank0, desired_ranks)]):
        self.evaluate(array_ops.identity(tensor_rank0))

  @test_util.run_deprecated_v1
  def test_rank_zero_tensor_doesnt_raise_if_rank_matches_dynamic_rank(self):
    # [FUNCTION DOC]
    # Test: placeholder fed a scalar passes for three orderings of allowed ranks that contain 0.
    with self.cached_session():
      tensor_rank0 = array_ops.placeholder(dtypes.float32, name="my_tensor")
      for desired_ranks in ((0, 1, 2), (1, 0, 2), (1, 2, 0)):
        with ops.control_dependencies([
            check_ops.assert_rank_in(tensor_rank0, desired_ranks)]):
          array_ops.identity(tensor_rank0).eval(feed_dict={tensor_rank0: 42.0})

  @test_util.run_in_graph_and_eager_modes
  def test_rank_one_tensor_doesnt_raise_if_rank_matches_static_rank(self):
    # [FUNCTION DOC]
    # Test: vector constant passes when rank 1 appears anywhere in the allowed set (three orderings).
    tensor_rank1 = constant_op.constant([42, 43], name="my_tensor")
    for desired_ranks in ((0, 1, 2), (1, 0, 2), (1, 2, 0)):
      with ops.control_dependencies([
          check_ops.assert_rank_in(tensor_rank1, desired_ranks)]):
        self.evaluate(array_ops.identity(tensor_rank1))

  @test_util.run_deprecated_v1
  def test_rank_one_tensor_doesnt_raise_if_rank_matches_dynamic_rank(self):
    # [FUNCTION DOC]
    # Test: placeholder fed a vector passes for three orderings of allowed ranks that contain 1.
    with self.cached_session():
      tensor_rank1 = array_ops.placeholder(dtypes.float32, name="my_tensor")
      for desired_ranks in ((0, 1, 2), (1, 0, 2), (1, 2, 0)):
        with ops.control_dependencies([
            check_ops.assert_rank_in(tensor_rank1, desired_ranks)]):
          array_ops.identity(tensor_rank1).eval(feed_dict={
              tensor_rank1: (42.0, 43.0)
          })

  @test_util.run_in_graph_and_eager_modes
  def test_rank_one_tensor_raises_if_rank_mismatches_static_rank(self):
    # [FUNCTION DOC]
    # Test: vector constant with allowed ranks (0,2) raises a value error mentioning "rank".
    tensor_rank1 = constant_op.constant((42, 43), name="my_tensor")
    with self.assertRaisesRegex(ValueError, "rank"):
      with ops.control_dependencies([
          check_ops.assert_rank_in(tensor_rank1, (0, 2))]):
        self.evaluate(array_ops.identity(tensor_rank1))

  @test_util.run_deprecated_v1
  def test_rank_one_tensor_raises_if_rank_mismatches_dynamic_rank(self):
    # [FUNCTION DOC]
    # Test: placeholder fed a vector with allowed ranks (0,2) raises an op error matching
    # "my_tensor" and "rank" on evaluation.
    with self.cached_session():
      tensor_rank1 = array_ops.placeholder(dtypes.float32, name="my_tensor")
      with ops.control_dependencies([
          check_ops.assert_rank_in(tensor_rank1, (0, 2))]):
        with self.assertRaisesOpError("my_tensor.*rank"):
          array_ops.identity(tensor_rank1).eval(feed_dict={
              tensor_rank1: (42.0, 43.0)
          })

  @test_util.run_in_graph_and_eager_modes
  def test_raises_if_rank_is_not_scalar_static(self):
    # [FUNCTION DOC]
    # Test: allowed ranks given as numpy arrays where one is a two-element vector raise a value
    # error "Rank must be a scalar" at the call (graph and eager).
    tensor = constant_op.constant((42, 43), name="my_tensor")
    desired_ranks = (
        np.array(1, dtype=np.int32),
        np.array((2, 1), dtype=np.int32))
    with self.assertRaisesRegex(ValueError, "Rank must be a scalar"):
      check_ops.assert_rank_in(tensor, desired_ranks)

  @test_util.run_deprecated_v1
  def test_raises_if_rank_is_not_scalar_dynamic(self):
    # [FUNCTION DOC]
    # Test: allowed ranks supplied via two int placeholders, one fed a two-element list, raise an
    # op error "Rank must be a scalar" on evaluation (deprecated-v1 graph mode).
    with self.cached_session():
      tensor = constant_op.constant(
          (42, 43), dtype=dtypes.float32, name="my_tensor")
      desired_ranks = (
          array_ops.placeholder(dtypes.int32, name="rank0_tensor"),
          array_ops.placeholder(dtypes.int32, name="rank1_tensor"))
      with self.assertRaisesOpError("Rank must be a scalar"):
        with ops.control_dependencies(
            (check_ops.assert_rank_in(tensor, desired_ranks),)):
          array_ops.identity(tensor).eval(feed_dict={
              desired_ranks[0]: 1,
              desired_ranks[1]: [2, 1],
          })

  @test_util.run_in_graph_and_eager_modes
  def test_raises_if_rank_is_not_integer_static(self):
    # [FUNCTION DOC]
    # Test: an allowed-rank set containing 1 and 0.5 raises a type error saying it must be of type
    # int32 (graph and eager).
    tensor = constant_op.constant((42, 43), name="my_tensor")
    with self.assertRaisesRegex(TypeError, "must be of type tf.int32"):
      check_ops.assert_rank_in(tensor, (1, .5,))

  @test_util.run_deprecated_v1
  def test_raises_if_rank_is_not_integer_dynamic(self):
    # [FUNCTION DOC]
    # Test: an allowed-rank set containing a float32 placeholder raises a type error saying it must
    # be of type int32 (deprecated-v1 graph mode).
    with self.cached_session():
      tensor = constant_op.constant(
          (42, 43), dtype=dtypes.float32, name="my_tensor")
      rank_tensor = array_ops.placeholder(dtypes.float32, name="rank_tensor")
      with self.assertRaisesRegex(TypeError, "must be of type tf.int32"):
        with ops.control_dependencies(
            [check_ops.assert_rank_in(tensor, (1, rank_tensor))]):
          array_ops.identity(tensor).eval(feed_dict={rank_tensor: .5})


class AssertRankAtLeastTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the minimum-rank assertion (rank >= desired), with static (constant, value
  # errors) and dynamic (placeholder, op errors, deprecated-v1) variants for tensors of rank 0
  # and 1 against desired ranks 0, 1 and 2.

  @test_util.run_in_graph_and_eager_modes
  def test_rank_zero_tensor_raises_if_rank_too_small_static_rank(self):
    # [FUNCTION DOC]
    # Test: scalar constant with minimum rank 1 raises a value error matching "rank at least 1".
    tensor = constant_op.constant(1, name="my_tensor")
    desired_rank = 1
    with self.assertRaisesRegex(ValueError, "rank at least 1"):
      with ops.control_dependencies(
          [check_ops.assert_rank_at_least(tensor, desired_rank)]):
        self.evaluate(array_ops.identity(tensor))

  @test_util.run_deprecated_v1
  def test_rank_zero_tensor_raises_if_rank_too_small_dynamic_rank(self):
    # [FUNCTION DOC]
    # Test: placeholder fed a scalar with minimum rank 1 raises an op error matching "my_tensor"
    # and "rank" on evaluation.
    with self.cached_session():
      tensor = array_ops.placeholder(dtypes.float32, name="my_tensor")
      desired_rank = 1
      with ops.control_dependencies(
          [check_ops.assert_rank_at_least(tensor, desired_rank)]):
        with self.assertRaisesOpError("my_tensor.*rank"):
          array_ops.identity(tensor).eval(feed_dict={tensor: 0})

  @test_util.run_in_graph_and_eager_modes
  def test_rank_zero_tensor_doesnt_raise_if_rank_just_right_static_rank(self):
    # [FUNCTION DOC]
    # Test: scalar constant with minimum rank 0 passes.
    tensor = constant_op.constant(1, name="my_tensor")
    desired_rank = 0
    with ops.control_dependencies(
        [check_ops.assert_rank_at_least(tensor, desired_rank)]):
      self.evaluate(array_ops.identity(tensor))

  @test_util.run_deprecated_v1
  def test_rank_zero_tensor_doesnt_raise_if_rank_just_right_dynamic_rank(self):
    # [FUNCTION DOC]
    # Test: placeholder fed a scalar with minimum rank 0 passes at run time.
    with self.cached_session():
      tensor = array_ops.placeholder(dtypes.float32, name="my_tensor")
      desired_rank = 0
      with ops.control_dependencies(
          [check_ops.assert_rank_at_least(tensor, desired_rank)]):
        array_ops.identity(tensor).eval(feed_dict={tensor: 0})

  @test_util.run_in_graph_and_eager_modes
  def test_rank_one_ten_doesnt_raise_raise_if_rank_too_large_static_rank(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the method name has a duplicated "raise_raise" and the abbreviation "ten".
    # Test: vector constant with minimum rank 0 passes (its rank is larger than required).
    tensor = constant_op.constant([1, 2], name="my_tensor")
    desired_rank = 0
    with ops.control_dependencies(
        [check_ops.assert_rank_at_least(tensor, desired_rank)]):
      self.evaluate(array_ops.identity(tensor))

  @test_util.run_deprecated_v1
  def test_rank_one_ten_doesnt_raise_if_rank_too_large_dynamic_rank(self):
    # [FUNCTION DOC]
    # Test: placeholder fed a vector with minimum rank 0 passes at run time (rank exceeds minimum).
    with self.cached_session():
      tensor = array_ops.placeholder(dtypes.float32, name="my_tensor")
      desired_rank = 0
      with ops.control_dependencies(
          [check_ops.assert_rank_at_least(tensor, desired_rank)]):
        array_ops.identity(tensor).eval(feed_dict={tensor: [1, 2]})

  @test_util.run_in_graph_and_eager_modes
  def test_rank_one_tensor_doesnt_raise_if_rank_just_right_static_rank(self):
    # [FUNCTION DOC]
    # Test: vector constant with minimum rank 1 passes.
    tensor = constant_op.constant([1, 2], name="my_tensor")
    desired_rank = 1
    with ops.control_dependencies(
        [check_ops.assert_rank_at_least(tensor, desired_rank)]):
      self.evaluate(array_ops.identity(tensor))

  @test_util.run_deprecated_v1
  def test_rank_one_tensor_doesnt_raise_if_rank_just_right_dynamic_rank(self):
    # [FUNCTION DOC]
    # Test: placeholder fed a vector with minimum rank 1 passes at run time.
    with self.cached_session():
      tensor = array_ops.placeholder(dtypes.float32, name="my_tensor")
      desired_rank = 1
      with ops.control_dependencies(
          [check_ops.assert_rank_at_least(tensor, desired_rank)]):
        array_ops.identity(tensor).eval(feed_dict={tensor: [1, 2]})

  @test_util.run_in_graph_and_eager_modes
  def test_rank_one_tensor_raises_if_rank_too_small_static_rank(self):
    # [FUNCTION DOC]
    # Test: vector constant with minimum rank 2 raises a value error matching "rank at least 2".
    tensor = constant_op.constant([1, 2], name="my_tensor")
    desired_rank = 2
    with self.assertRaisesRegex(ValueError, "rank at least 2"):
      with ops.control_dependencies(
          [check_ops.assert_rank_at_least(tensor, desired_rank)]):
        self.evaluate(array_ops.identity(tensor))

  @test_util.run_deprecated_v1
  def test_rank_one_tensor_raises_if_rank_too_small_dynamic_rank(self):
    # [FUNCTION DOC]
    # Test: placeholder fed a vector with minimum rank 2 raises an op error matching "my_tensor"
    # and "rank" on evaluation.
    with self.cached_session():
      tensor = array_ops.placeholder(dtypes.float32, name="my_tensor")
      desired_rank = 2
      with ops.control_dependencies(
          [check_ops.assert_rank_at_least(tensor, desired_rank)]):
        with self.assertRaisesOpError("my_tensor.*rank"):
          array_ops.identity(tensor).eval(feed_dict={tensor: [1, 2]})


class AssertNonNegativeTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the non-negative (>= 0) assertion: negative fails, zero and positive pass, empty
  # tensors pass vacuously, Python scalars are checked statically in graph mode.

  @test_util.run_in_graph_and_eager_modes
  @test_util.run_deprecated_v1
  def test_raises_when_negative(self):
    # [FUNCTION DOC]
    # Test: [-1,-2] fails; expects an op error containing "x >= 0 did not hold".
    # Deprecated-v1 mode, run in graph and eager modes.
    zoe = constant_op.constant([-1, -2], name="zoe")
    with self.assertRaisesOpError("x >= 0 did not hold"):
      with ops.control_dependencies([check_ops.assert_non_negative(zoe)]):
        out = array_ops.identity(zoe)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_zero_and_positive(self):
    # [FUNCTION DOC]
    # Test: [0,2] passes; guarded identity evaluates without error.
    lucas = constant_op.constant([0, 2], name="lucas")
    with ops.control_dependencies([check_ops.assert_non_negative(lucas)]):
      out = array_ops.identity(lucas)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_empty_tensor_doesnt_raise(self):
    # [FUNCTION DOC]
    # Test: an empty tensor passes because the "all elements non-negative" condition is vacuously
    # true.
    # A tensor is non-negative when it satisfies:
    #   For every element x_i in x, x_i >= 0
    # and an empty tensor has no elements, so this is trivially satisfied.
    # This is standard set theory.
    empty = constant_op.constant([], name="empty")
    with ops.control_dependencies([check_ops.assert_non_negative(empty)]):
      out = array_ops.identity(empty)
    self.evaluate(out)

  def test_static_check_in_graph_mode(self):
    # [FUNCTION DOC]
    # Test: in graph mode the Python integer -1 is statically rejected with an invalid-argument
    # error matching "Custom error message".
    with ops.Graph().as_default():
      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  "Custom error message"):
        check_ops.assert_non_negative(-1, message="Custom error message")


class AssertNonPositiveTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the non-positive (<= 0) assertion: positive fails, zero and negative pass, empty
  # tensors pass vacuously, Python scalars are checked statically in graph mode.

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_zero_and_negative(self):
    # [FUNCTION DOC]
    # Test: [0,-2] passes; guarded identity evaluates without error.
    tom = constant_op.constant([0, -2], name="tom")
    with ops.control_dependencies([check_ops.assert_non_positive(tom)]):
      out = array_ops.identity(tom)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  @test_util.run_deprecated_v1
  def test_raises_when_positive(self):
    # [FUNCTION DOC]
    # Test: [0,2] fails; expects an op error containing "x <= 0 did not hold".
    # Deprecated-v1 mode, run in graph and eager modes.
    rachel = constant_op.constant([0, 2], name="rachel")
    with self.assertRaisesOpError("x <= 0 did not hold"):
      with ops.control_dependencies([check_ops.assert_non_positive(rachel)]):
        out = array_ops.identity(rachel)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_empty_tensor_doesnt_raise(self):
    # [FUNCTION DOC]
    # Test: an empty tensor passes because the "all elements non-positive" condition is vacuously
    # true.
    # A tensor is non-positive when it satisfies:
    #   For every element x_i in x, x_i <= 0
    # and an empty tensor has no elements, so this is trivially satisfied.
    # This is standard set theory.
    empty = constant_op.constant([], name="empty")
    with ops.control_dependencies([check_ops.assert_non_positive(empty)]):
      out = array_ops.identity(empty)
    self.evaluate(out)

  def test_static_check_in_graph_mode(self):
    # [FUNCTION DOC]
    # Test: in graph mode the Python integer 1 is statically rejected with an invalid-argument
    # error matching "Custom error message".
    with ops.Graph().as_default():
      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  "Custom error message"):
        check_ops.assert_non_positive(1, message="Custom error message")


class AssertIntegerTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the integer-dtype assertion: integer tensors pass, float tensors raise a type
  # error.

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_integer(self):
    # [FUNCTION DOC]
    # Test: an int32 constant passes; guarded identity evaluates without error (graph and eager).
    integers = constant_op.constant([1, 2], name="integers")
    with ops.control_dependencies([check_ops.assert_integer(integers)]):
      out = array_ops.identity(integers)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_raises_when_float(self):
    # [FUNCTION DOC]
    # Test: a float32 constant raises a type error matching "Expected" and "integer".
    floats = constant_op.constant([1.0, 2.0], name="floats")
    with self.assertRaisesRegex(TypeError, "Expected.*integer"):
      check_ops.assert_integer(floats)


class AssertTypeTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the dtype assertion on dense, sparse and ragged tensors: matching dtypes pass,
  # mismatches raise a type error, and a non-dtype expected type is rejected.

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_correct_type(self):
    # [FUNCTION DOC]
    # Test: an int64 constant checked against int64 passes; guarded identity evaluates.
    integers = constant_op.constant([1, 2], dtype=dtypes.int64)
    with ops.control_dependencies([
        check_ops.assert_type(integers, dtypes.int64)]):
      out = array_ops.identity(integers)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_sparsetensor_doesnt_raise_when_correct_type(self):
    # [FUNCTION DOC]
    # Test: a float32 sparse tensor checked against float32 passes; rebuilds a sparse tensor with
    # an identity on its values under the control dependency and evaluates it.
    sparse_float = sparse_tensor.SparseTensor(
        constant_op.constant([[111], [232]], dtypes.int64),
        constant_op.constant([23.4, -43.2], dtypes.float32),
        constant_op.constant([500], dtypes.int64))

    with ops.control_dependencies(
        [check_ops.assert_type(sparse_float, dtypes.float32)]):
      out = sparse_tensor.SparseTensor(sparse_float.indices,
                                       array_ops.identity(sparse_float.values),
                                       sparse_float.dense_shape)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_raggedtensor_doesnt_raise_when_correct_type(self):
    # [FUNCTION DOC]
    # Test: a float ragged tensor checked against float32 passes; the identity of it equals the
    # original (asserted).
    x = ragged_factory_ops.constant([[1., 2.], [3.]])
    with ops.control_dependencies(
        [check_ops.assert_type(x, dtypes.float32)]):
      y = array_ops.identity(x)
    self.assertAllEqual(x, y)

  @test_util.run_in_graph_and_eager_modes
  def test_raises_when_wrong_type(self):
    # [FUNCTION DOC]
    # Test: a float16 tensor checked against float32 raises a type error "must be of type
    # tf.float32; got tf.float16".
    floats = constant_op.constant([1.0, 2.0], dtype=dtypes.float16)
    with self.assertRaisesRegex(TypeError, "must be of type tf.float32; "
                                "got tf.float16"):
      check_ops.assert_type(floats, dtypes.float32)

  @test_util.run_in_graph_and_eager_modes
  def test_sparsetensor_raises_when_wrong_type(self):
    # [FUNCTION DOC]
    # Test: a float16 sparse tensor checked against float32 raises a type error matching "must be
    # of type" and "float32".
    sparse_float16 = sparse_tensor.SparseTensor(
        constant_op.constant([[111], [232]], dtypes.int64),
        constant_op.constant([23.4, -43.2], dtypes.float16),
        constant_op.constant([500], dtypes.int64))
    with self.assertRaisesRegex(TypeError, "must be of type.*float32"):
      check_ops.assert_type(sparse_float16, dtypes.float32)

  @test_util.run_in_graph_and_eager_modes
  def test_raggedtensor_raises_when_wrong_type(self):
    # [FUNCTION DOC]
    # Test: an integer ragged tensor checked against float32 raises a type error matching "must be
    # of type" and "float32".
    x = ragged_factory_ops.constant([[1, 2], [3]])
    with self.assertRaisesRegex(TypeError, "must be of type.*float32"):
      check_ops.assert_type(x, dtypes.float32)

  def test_raise_when_tf_type_is_not_dtype(self):
    # [FUNCTION DOC]
    # Test (regression for a public issue report): passing a tuple containing a dtype instead of a
    # dtype itself as the expected type raises a type error matching "Cannot convert ... to a
    # TensorFlow DType".
    # Test case for GitHub issue:
    # https://github.com/tensorflow/tensorflow/issues/45975
    value = constant_op.constant(0.0)
    with self.assertRaisesRegex(TypeError,
                                 "Cannot convert.*to a TensorFlow DType"):
      check_ops.assert_type(value, (dtypes.float32,))


class AssertShapesTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the shapes assertion, which checks a list of (tensor, shape specification) pairs
  # for consistency. Specifications may use symbols (shared across tensors), explicit sizes,
  # unknown markers (null, ".") and a leading variable-length marker (ellipsis or "*"), given as
  # tuples, lists, strings or arrays. Static failures raise value errors at graph construction;
  # dynamic failures raise invalid-argument errors at run time. Has two shared helper methods.

  @test_util.run_in_graph_and_eager_modes
  def test_raise_static_shape_mismatch(self):
    # [FUNCTION DOC]
    # Test: symbol "N" is bound by a 3x2 tensor to size 3 but a 2x3 tensor uses it for dimension 0
    # (size 2); a static value error "Specified by tensor ... dimension 0 ... must have size 3 ...
    # Received size 2" is expected, via the static error helper.
    x = array_ops.ones([3, 2], name="x")
    y = array_ops.ones([2, 3], name="y")
    shapes = [
        (x, ("N", "Q")),
        (y, ("N", "D")),
    ]
    regex = (r"Specified by tensor .* dimension 0.  "
             r"Tensor .* dimension 0 must have size 3.  "
             r"Received size 2")
    self.raises_static_error(shapes=shapes, regex=regex)

  def test_raise_dynamic_shape_mismatch(self):
    # [FUNCTION DOC]
    # Test: same symbol conflict as the static case but with placeholders of partly unknown shape,
    # fed 3x2 and 2x3 arrays in a session. Expects an invalid-argument error whose bracketed text
    # says the specification came from tensor x and tensor y dimension 0 must have size 3.
    # Uses the dynamic error helper; graph mode only.
    with ops.Graph().as_default():
      x = array_ops.placeholder(dtypes.float32, [None, 2], name="x")
      y = array_ops.placeholder(dtypes.float32, [None, 3], name="y")
      shapes = [
          (x, ("N", "Q")),
          (y, ("N", "D")),
      ]
      regex = (r"\[Specified by tensor x.* dimension 0\] "
               r"\[Tensor y.* dimension\] \[0\] \[must have size\] \[3\]")
      feed_dict = {x: np.ones([3, 2]), y: np.ones([2, 3])}
      self.raises_dynamic_error(shapes=shapes, regex=regex, feed_dict=feed_dict)

  @test_util.run_in_graph_and_eager_modes
  def test_raise_static_shape_explicit_mismatch(self):
    # [FUNCTION DOC]
    # Test: explicit size 3 for dimension 0 of a 2x3 tensor produces a static value error "Specified
    # explicitly ... must have size 3 ... Received size 2".
    x = array_ops.ones([3, 2], name="x")
    y = array_ops.ones([2, 3], name="y")
    shapes = [
        (x, (3, "Q")),
        (y, (3, "D")),
    ]
    regex = (r"Specified explicitly.  "
             r"Tensor .* dimension 0 must have size 3.  "
             r"Received size 2")
    self.raises_static_error(shapes=shapes, regex=regex)

  @test_util.run_in_graph_and_eager_modes
  def test_rank_zero_rank_one_size_one_equivalence(self):
    # [FUNCTION DOC]
    # Test: a rank-1 size-1 tensor and a scalar are interchangeable; both match an empty
    # specification and both match the specification (1,). No error expected.
    rank_one_size_one = array_ops.ones([1], name="rank_one_size_one")
    rank_zero = array_ops.constant(5, name="rank_zero")
    check_ops.assert_shapes([
        (rank_one_size_one, ()),
        (rank_zero, ()),
    ])
    check_ops.assert_shapes([
        (rank_one_size_one, (1,)),
        (rank_zero, (1,)),
    ])

  @test_util.run_in_graph_and_eager_modes
  def test_raise_static_rank_1_size_not_1_mismatch_scalar(self):
    # [FUNCTION DOC]
    # Test: a two-element vector against an empty (scalar) specification fails statically with
    # "Specified explicitly ... dimension 0 must have size 1 ... Received size 2".
    x = array_ops.constant([2, 2], name="x")
    shapes = [
        (x, ()),
    ]
    regex = (r"Specified explicitly.  "
             r"Tensor .* dimension 0 must have size 1.  "
             r"Received size 2")
    self.raises_static_error(shapes=shapes, regex=regex)

  @test_util.run_in_graph_and_eager_modes
  def test_raise_static_scalar_mismatch_rank_1_size_not_1(self):
    # [FUNCTION DOC]
    # Test: a scalar against specification (2,) fails statically with "Specified explicitly ...
    # dimension 0 must have size 2 ... Received size 1".
    x = array_ops.constant(2, name="x")
    shapes = [
        (x, (2,)),
    ]
    regex = (r"Specified explicitly.  "
             r"Tensor .* dimension 0 must have size 2.  "
             r"Received size 1")
    self.raises_static_error(shapes=shapes, regex=regex)

  @test_util.run_in_graph_and_eager_modes
  def test_scalar_implies_size_one(self):
    # [FUNCTION DOC]
    # Test: a scalar is treated as size 1 for symbol "a", which then conflicts with a 2x2 tensor
    # using "a" for dimension 0; expects a static error "Specified by tensor ... must have size 1
    # ... Received size 2".
    scalar = array_ops.constant(5, name="rank_zero")
    x = array_ops.ones([2, 2], name="x")
    shapes = [(scalar, ("a",)), (x, ("a", 2))]
    regex = (r"Specified by tensor .* dimension 0.  "
             r"Tensor .* dimension 0 must have size 1.  "
             r"Received size 2")
    self.raises_static_error(shapes=shapes, regex=regex)

  @test_util.run_in_graph_and_eager_modes
  def test_raise_not_iterable(self):
    # [FUNCTION DOC]
    # Test: a specification that is a plain integer instead of an iterable fails statically with a
    # message saying the specified shape must be an iterable (having an iterate or get-item
    # attribute) and showing the received specification 2.
    x = array_ops.constant([1, 2], name="x")
    shapes = [(x, 2)]
    regex = (r"Tensor .*.  "
             r"Specified shape must be an iterable.  "
             r"An iterable has the attribute `__iter__` or `__getitem__`.  "
             r"Received specified shape: 2")
    self.raises_static_error(shapes=shapes, regex=regex)

  def test_raise_dynamic_shape_explicit_mismatch(self):
    # [FUNCTION DOC]
    # Test: explicit size 3 for dimension 0 of a fed 2x3 value (with placeholders of partly unknown
    # shape) fails at run time with a bracketed "Specified explicitly ... Tensor y ... must have
    # size 3" invalid-argument error. Uses the dynamic error helper; graph mode only.
    with ops.Graph().as_default():
      x = array_ops.placeholder(dtypes.float32, [None, 2], name="xa")
      y = array_ops.placeholder(dtypes.float32, [None, 3], name="y")
      shapes = [
          (x, (3, "Q")),
          (y, (3, "D")),
      ]
      regex = (r"\[Specified explicitly\] "
               r"\[Tensor y.* dimension\] \[0\] \[must have size\] \[3\]")
      feed_dict = {x: np.ones([3, 2]), y: np.ones([2, 3])}
      self.raises_dynamic_error(shapes=shapes, regex=regex, feed_dict=feed_dict)

  @test_util.run_in_graph_and_eager_modes
  def test_no_op_when_specified_as_unknown(self):
    # [FUNCTION DOC]
    # Test: a null specification for a tensor means "unknown shape"; the assertion is a no-op and
    # the guarded identity evaluates without error.
    x = array_ops.constant([1, 1], name="x")
    assertion = check_ops.assert_shapes([(x, None)])
    with ops.control_dependencies([assertion]):
      out = array_ops.identity(x)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_raises_static_incorrect_rank(self):
    # [FUNCTION DOC]
    # Test: specifications of the wrong rank (several spellings of rank-2 and rank-3 specs, including
    # symbols, sizes and null entries) are rejected statically for tensors of rank 1, 2 and 0 with
    # "must have rank N. Received rank M". Uses a nested helper that loops over specifications.
    rank_two_shapes = [
        (1, 1),
        (1, 3),
        ("a", "b"),
        (None, None),
    ]
    rank_three_shapes = [
        (1, 1, 1),
        ("a", "b", "c"),
        (None, None, None),
        (1, "b", None),
    ]

    def raises_static_rank_error(shapes, x, correct_rank, actual_rank):
      # [FUNCTION DOC]
      # HELPER: Nested helper for the static incorrect-rank test.
      # Inputs: shapes (list of specifications), x (tensor), correct_rank and actual_rank (integers
      # used to build the expected message). For each specification, asserts via the static error
      # helper that the assertion raises a value error "Tensor ... must have rank <correct>. Received
      # rank <actual>". Returns nothing.
      for shape in shapes:
        regex = (r"Tensor .* must have rank %d.  Received rank %d" %
                 (correct_rank, actual_rank))
        self.raises_static_error(shapes=[(x, shape)], regex=regex)

    raises_static_rank_error(
        rank_two_shapes, array_ops.ones([1]), correct_rank=2, actual_rank=1)
    raises_static_rank_error(
        rank_three_shapes,
        array_ops.ones([1, 1]),
        correct_rank=3,
        actual_rank=2)
    raises_static_rank_error(
        rank_three_shapes, array_ops.constant(1), correct_rank=3, actual_rank=0)

  def test_raises_dynamic_incorrect_rank(self):
    # [FUNCTION DOC]
    # Test: a placeholder of unknown shape fed a scalar fails rank-2 specifications at run time.
    # Loops over four specifications and expects an invalid-argument error matching "Tensor ... must
    # have rank] [2]" through the dynamic error helper (graph mode only).
    x_value = 5
    rank_two_shapes = [(1, 1), (1, 3), ("a", "b"), (None, None)]
    with ops.Graph().as_default():
      x = array_ops.placeholder(dtypes.float32, None)

      for shape in rank_two_shapes:
        regex = r"Tensor .* must have rank\] \[2\]"
        self.raises_dynamic_error(
            shapes=[(x, shape)], regex=regex, feed_dict={x: x_value})

  @test_util.run_in_graph_and_eager_modes
  def test_correctly_matching(self):
    # [FUNCTION DOC]
    # Test: consistent symbolic and explicit specifications pass.
    # Builds a scalar and tensors of shapes (1,2), (3), (1,2,3), (3,1,2), (2,3,1) and checks them
    # with shared symbols a, b, c (including a scalar used as "a"), then with explicit sizes
    # in place of some symbols (and an empty specification for the scalar). Both assertions are
    # evaluated without error.
    u = array_ops.constant(1, name="u")
    v = array_ops.ones([1, 2], name="v")
    w = array_ops.ones([3], name="w")
    x = array_ops.ones([1, 2, 3], name="x")
    y = array_ops.ones([3, 1, 2], name="y")
    z = array_ops.ones([2, 3, 1], name="z")
    assertion = check_ops.assert_shapes([
        (x, ("a", "b", "c")),
        (y, ("c", "a", "b")),
        (z, ("b", "c", "a")),
        (v, ("a", "b")),
        (w, ("c",)),
        (u, "a")
    ])
    with ops.control_dependencies([assertion]):
      out = array_ops.identity(x)
    self.evaluate(out)
    assertion = check_ops.assert_shapes([
        (x, (1, "b", "c")),
        (y, ("c", "a", 2)),
        (z, ("b", 3, "a")),
        (v, ("a", 2)),
        (w, (3,)),
        (u, ())
    ])
    with ops.control_dependencies([assertion]):
      out = array_ops.identity(x)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_variable_length_symbols(self):
    # [FUNCTION DOC]
    # Test: multi-character symbol names ("num_observations", "input_dim", "output_dim") are
    # supported; two tensors sharing the leading size 4 pass.
    x = array_ops.ones([4, 1], name="x")
    y = array_ops.ones([4, 2], name="y")
    assertion = check_ops.assert_shapes([
        (x, ("num_observations", "input_dim")),
        (y, ("num_observations", "output_dim")),
    ])
    with ops.control_dependencies([assertion]):
      out = array_ops.identity(x)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_raise_implicit_mismatch_using_iterable_alternatives(self):
    # [FUNCTION DOC]
    # Test: the symbol mismatch (size 2 vs. size 1 for "A") is detected regardless of how the
    # specification is written: tuples, strings, lists, numpy arrays, or a mix.
    # Each style expects a static error "Specified by tensor ... dimension 0 ... must have size 2 ...
    # Received size 1".
    x = array_ops.ones([2, 2], name="x")
    y = array_ops.ones([1, 3], name="y")
    styles = [[
        (x, ("A", "B")),
        (y, ("A", "C")),
    ], [
        (x, "AB"),
        (y, "AC")
    ], [
        (x, ["A", "B"]),
        (y, ["A", "C"]),
    ], [
        (x, np.array(["A", "B"])),
        (y, np.array(["A", "C"]))
    ], [
        (x, ("A", "B")),
        (y, "AC")
    ]]
    for shapes in styles:
      self.raises_static_error(
          shapes=shapes,
          regex=(r"Specified by tensor .* dimension 0.  "
                 "Tensor .* dimension 0 must have size 2.  "
                 "Received size 1"))

  @test_util.run_in_graph_and_eager_modes
  def test_raise_explicit_mismatch_using_iterable_alternatives(self):
    # [FUNCTION DOC]
    # Test: explicit-size mismatch (2 expected, 1 received) is detected for tuples, digit strings,
    # lists, numpy arrays, or a mix; expects "Specified explicitly ... must have size 2 ... Received
    # size 1" statically.
    x = array_ops.ones([2, 2], name="x")
    y = array_ops.ones([1, 3], name="y")
    styles = [[
        (x, (2, 2)),
        (y, (2, 3)),
    ], [
        (x, "22"),
        (y, "23")
    ], [
        (x, [2, 2]),
        (y, [2, 3]),
    ], [
        (x, np.array([2, 2])),
        (y, np.array([2, 3]))
    ], [
        (x, (2, 2)),
        (y, "23")
    ]]
    for shapes in styles:
      self.raises_static_error(
          shapes=shapes,
          regex=(r"Specified explicitly.  "
                 "Tensor .* dimension 0 must have size 2.  "
                 "Received size 1"))

  @test_util.run_in_graph_and_eager_modes
  def test_dim_size_specified_as_unknown(self):
    # [FUNCTION DOC]
    # Test: unknown dimensions can be written as null, a dot character, or a dot in a string
    # specification; three assertions on 1x2x3 and 2x1 tensors all pass.
    x = array_ops.ones([1, 2, 3], name="x")
    y = array_ops.ones([2, 1], name="y")
    a1 = check_ops.assert_shapes([
        (x, (None, 2, None)),
        (y, (None, 1)),
    ])
    a2 = check_ops.assert_shapes([
        (x, (".", 2, ".")),
        (y, (".", 1)),
    ])
    a3 = check_ops.assert_shapes([
        (x, ".2."),
        (y, ".1"),
    ])
    with ops.control_dependencies([a1, a2, a3]):
      out = array_ops.identity(x)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_raise_static_shape_explicit_mismatch_innermost_dims(self):
    # [FUNCTION DOC]
    # Test: a leading variable-length marker (ellipsis or star) matches only innermost dimensions;
    # a 2x3 tensor against explicit (3, "D") fails with "Specified explicitly ... dimension -2 must
    # have size 3 ... Received size 2". Both tuple and string spellings are checked.
    x = array_ops.ones([3, 2], name="x")
    y = array_ops.ones([2, 3], name="y")
    s1 = [
        (x, (3, "Q")),
        (y, (Ellipsis, 3, "D")),
    ]
    s2 = [
        (x, "3Q"),
        (y, "*3D"),
    ]
    regex = (r"Specified explicitly.  "
             r"Tensor .* dimension -2 must have size 3.  "
             r"Received size 2")
    self.raises_static_error(shapes=s1, regex=regex)
    self.raises_static_error(shapes=s2, regex=regex)

  @test_util.run_in_graph_and_eager_modes
  def test_correctly_matching_innermost_dims(self):
    # [FUNCTION DOC]
    # Test: with a leading variable-length marker, innermost dimensions N and Q/D are matched across
    # tensors of shapes 1x2x3x2 and 2x3x3; both tuple and string spellings pass.
    x = array_ops.ones([1, 2, 3, 2], name="x")
    y = array_ops.ones([2, 3, 3], name="y")
    a1 = check_ops.assert_shapes([
        (x, (Ellipsis, "N", "Q")),
        (y, (Ellipsis, "N", "D")),
    ])
    a2 = check_ops.assert_shapes([
        (x, "*NQ"),
        (y, "*ND"),
    ])
    with ops.control_dependencies([a1, a2]):
      out = array_ops.identity(x)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_raise_variable_num_outer_dims_prefix_misuse(self):
    # [FUNCTION DOC]
    # Test: the variable-length marker is only allowed as the first entry; placing it in the middle
    # (tuple or string) gives a static error saying the symbol for a variable number of unspecified
    # dimensions is only allowed as the first entry.
    x = array_ops.ones([1, 2], name="x")
    s1 = [
        (x, ("N", Ellipsis, "Q")),
    ]
    s2 = [
        (x, "N*Q"),
    ]
    regex = (r"Tensor .* specified shape index .*.  "
             r"Symbol `...` or `\*` for a variable number of "
             r"unspecified dimensions is only allowed as the first entry")
    self.raises_static_error(shapes=s1, regex=regex)
    self.raises_static_error(shapes=s2, regex=regex)

  @test_util.run_in_graph_and_eager_modes
  def test_empty_shapes_dict_no_op(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name says "dict" but an empty list is passed; no mapping is involved.
    # Test: asserting an empty collection of shape pairs is a no-op; guarded identity of 0 evaluates.
    assertion = check_ops.assert_shapes([])
    with ops.control_dependencies([assertion]):
      out = array_ops.identity(0)
    self.evaluate(out)

  def raises_static_error(self, shapes, regex):
    # [FUNCTION DOC]
    # HELPER: Shared assertion helper for the shapes tests (not itself a test).
    # Inputs: shapes (list of tensor/specification pairs) and regex (pattern). Calls the shapes
    # assertion and checks that it raises a value error matching the pattern. Returns nothing.
    with self.assertRaisesRegex(ValueError, regex):
      check_ops.assert_shapes(shapes)

  def raises_dynamic_error(self, shapes, regex, feed_dict):
    # [FUNCTION DOC]
    # HELPER: Shared assertion helper for the shapes tests (not itself a test).
    # Inputs: shapes (list of pairs), regex (pattern), feed_dict (mapping from placeholders to
    # values). Opens a session, builds the shapes assertion as a control dependency of an identity of
    # zero, runs it with the feed, and checks that an invalid-argument error matching the pattern is
    # raised. Returns nothing.
    with self.session() as sess:
      with self.assertRaisesRegex(errors.InvalidArgumentError, regex):
        assertion = check_ops.assert_shapes(shapes)
        with ops.control_dependencies([assertion]):
          out = array_ops.identity(0)
        sess.run(out, feed_dict=feed_dict)


class AssertShapesSparseTensorTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the shapes assertion applied to sparse tensors (using their dense shape): scalar,
  # fully and partially specified, symbolic, wrong rank, multiple assertions, and mixing with
  # dense tensors. Failures are value errors raised when building the assertion.

  @test_util.run_in_graph_and_eager_modes
  def test_assert_shapes_sparse_tensor_scalar_target_success(self):
    # [FUNCTION DOC]
    # Test: a scalar-shaped sparse tensor matches an empty shape specification; evaluates fine.
    sparse_float = sparse_tensor.SparseTensor(
        constant_op.constant([[]], dtypes.int64),
        constant_op.constant([42], dtypes.float32),
        constant_op.constant([], dtypes.int64))
    assertion = check_ops.assert_shapes([(sparse_float, [])])
    with ops.control_dependencies([assertion]):
      out = array_ops.identity(sparse_float)
    self.evaluate(out)

  def test_assert_shapes_sparse_tensor_nonscalar_target_fail(self):
    # [FUNCTION DOC]
    # Test: a scalar-shaped sparse tensor against a two-dimension specification raises a value error
    # "must have rank 2 ... Received rank 0". Graph-only (not wrapped for both modes).
    sparse_float = sparse_tensor.SparseTensor(
        constant_op.constant([[]], dtypes.int64),
        constant_op.constant([42], dtypes.float32),
        constant_op.constant([], dtypes.int64))
    with self.assertRaisesRegex(ValueError,
                                 r"must have rank 2.*Received rank 0"):
      assertion = check_ops.assert_shapes([(sparse_float, [None, None])])
      with ops.control_dependencies([assertion]):
        out = array_ops.identity(sparse_float)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_assert_shapes_sparse_tensor_fully_specified_target_success(self):
    # [FUNCTION DOC]
    # Test: a sparse tensor with dense shape [500] matches the specification [500]; passes.
    sparse_float = sparse_tensor.SparseTensor(
        constant_op.constant([[111], [232]], dtypes.int64),
        constant_op.constant([23.4, -43.2], dtypes.float32),
        constant_op.constant([500], dtypes.int64))
    assertion = check_ops.assert_shapes([(sparse_float, [500])])
    with ops.control_dependencies([assertion]):
      out = array_ops.identity(sparse_float)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_assert_shapes_sparse_tensor_fully_specified_target_fail(self):
    # [FUNCTION DOC]
    # Test: dense shape [500] against specification [499] raises a value error "dimension 0 must
    # have size 499".
    sparse_float = sparse_tensor.SparseTensor(
        constant_op.constant([[111], [232]], dtypes.int64),
        constant_op.constant([23.4, -43.2], dtypes.float32),
        constant_op.constant([500], dtypes.int64))
    with self.assertRaisesRegex(ValueError, r"dimension 0 must have size 499"):
      assertion = check_ops.assert_shapes([(sparse_float, [499])])
      with ops.control_dependencies([assertion]):
        out = array_ops.identity(sparse_float)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_assert_shapes_sparse_tensor_partially_specified_target_success(self):
    # [FUNCTION DOC]
    # Test: a sparse tensor of dense shape [30,40] matches the partial specification [unknown, 40].
    sparse_int = sparse_tensor.SparseTensor(
        constant_op.constant([[5, 6], [7, 8]], dtypes.int64),
        constant_op.constant([23, -43], dtypes.int32),
        constant_op.constant([30, 40], dtypes.int64))
    assertion = check_ops.assert_shapes([(sparse_int, [None, 40])])
    with ops.control_dependencies([assertion]):
      out = array_ops.identity(sparse_int)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_assert_shapes_sparse_tensor_symbolic_match_success(self):
    # [FUNCTION DOC]
    # Test: a sparse tensor of dense shape [30,30,40] matches symbolic specification N, N, D.
    sparse_int = sparse_tensor.SparseTensor(
        constant_op.constant([[5, 6, 7], [8, 9, 10]], dtypes.int64),
        constant_op.constant([23, -43], dtypes.int32),
        constant_op.constant([30, 30, 40], dtypes.int64))
    assertion = check_ops.assert_shapes([(sparse_int, ["N", "N", "D"])])
    with ops.control_dependencies([assertion]):
      out = array_ops.identity(sparse_int)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_assert_shapes_sparse_tensor_partially_specified_target_fail(self):
    # [FUNCTION DOC]
    # Test: dense shape [30,40] against [unknown, 41] raises a value error "dimension 1 must have
    # size 41".
    sparse_int = sparse_tensor.SparseTensor(
        constant_op.constant([[5, 6], [7, 8]], dtypes.int64),
        constant_op.constant([23, -43], dtypes.int32),
        constant_op.constant([30, 40], dtypes.int64))
    with self.assertRaisesRegex(ValueError, r"dimension 1 must have size 41"):
      assertion = check_ops.assert_shapes([(sparse_int, [None, 41])])
      with ops.control_dependencies([assertion]):
        out = array_ops.identity(sparse_int)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_assert_shapes_sparse_tensor_wrong_rank_fail(self):
    # [FUNCTION DOC]
    # Test: a rank-2 sparse tensor against a rank-3 specification raises a value error "must have
    # rank 3 ... Received rank 2".
    sparse_int = sparse_tensor.SparseTensor(
        constant_op.constant([[5, 6], [7, 8]], dtypes.int64),
        constant_op.constant([23, -43], dtypes.int32),
        constant_op.constant([30, 40], dtypes.int64))
    with self.assertRaisesRegex(ValueError,
                                 r"must have rank 3\..* Received rank 2"):
      assertion = check_ops.assert_shapes([(sparse_int, [None, None, 40])])
      with ops.control_dependencies([assertion]):
        out = array_ops.identity(sparse_int)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_assert_shapes_sparse_tensor_wrong_symbolic_match_fail(self):
    # [FUNCTION DOC]
    # Test: dense shape [30,40] against symbolic specification D, D raises a value error "dimension
    # 1 must have size 30".
    sparse_int = sparse_tensor.SparseTensor(
        constant_op.constant([[5, 6], [7, 8]], dtypes.int64),
        constant_op.constant([23, -43], dtypes.int32),
        constant_op.constant([30, 40], dtypes.int64))
    with self.assertRaisesRegex(ValueError, r"dimension 1 must have size 30"):
      assertion = check_ops.assert_shapes([(sparse_int, ["D", "D"])])
      with ops.control_dependencies([assertion]):
        out = array_ops.identity(sparse_int)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_assert_shapes_sparse_tensor_multiple_assertions_success(self):
    # [FUNCTION DOC]
    # Test: a scalar sparse tensor (spec empty) and a 30x30 sparse tensor (spec N, N) are checked in
    # one call and pass.
    sparse_scalar = sparse_tensor.SparseTensor(
        constant_op.constant([[]], dtypes.int64),
        constant_op.constant([42], dtypes.float32),
        constant_op.constant([], dtypes.int64))
    sparse_2d = sparse_tensor.SparseTensor(
        constant_op.constant([[5, 6], [7, 8]], dtypes.int64),
        constant_op.constant([23, -43], dtypes.int32),
        constant_op.constant([30, 30], dtypes.int64))
    assertion = check_ops.assert_shapes([(sparse_scalar, []),
                                         (sparse_2d, ["N", "N"])])
    with ops.control_dependencies([assertion]):
      out = array_ops.identity(sparse_2d)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_assert_shapes_sparse_tensor_multiple_assertions_fail(self):
    # [FUNCTION DOC]
    # Test: same pair as the success test but the second sparse tensor is 30x40; expects a value
    # error "dimension 1 must have size 30".
    sparse_scalar = sparse_tensor.SparseTensor(
        constant_op.constant([[]], dtypes.int64),
        constant_op.constant([42], dtypes.float32),
        constant_op.constant([], dtypes.int64))
    sparse_2d = sparse_tensor.SparseTensor(
        constant_op.constant([[5, 6], [7, 8]], dtypes.int64),
        constant_op.constant([23, -43], dtypes.int32),
        constant_op.constant([30, 40], dtypes.int64))
    with self.assertRaisesRegex(ValueError, r"dimension 1 must have size 30"):
      assertion = check_ops.assert_shapes([(sparse_scalar, []),
                                           (sparse_2d, ["N", "N"])])
      with ops.control_dependencies([assertion]):
        out = array_ops.identity(sparse_2d)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_assert_shapes_sparse_tensor_mixed_dense_and_sparse_success(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the variable called dense_scalar is really a one-element vector, not a scalar; it
    # matches the empty specification only through the size-one equivalence of rank 0 and rank 1.
    # Test: a dense tensor (spec empty) and a 30x30 sparse tensor (spec N, N) pass together.
    dense_scalar = constant_op.constant([42], dtypes.float32)
    sparse_2d = sparse_tensor.SparseTensor(
        constant_op.constant([[5, 6], [7, 8]], dtypes.int64),
        constant_op.constant([23, -43], dtypes.int32),
        constant_op.constant([30, 30], dtypes.int64))
    assertion = check_ops.assert_shapes([(dense_scalar, []),
                                         (sparse_2d, ["N", "N"])])
    with ops.control_dependencies([assertion]):
      out = array_ops.identity(sparse_2d)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_assert_shapes_sparse_tensor_mixed_dense_and_sparse_fail(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the variable called dense_scalar is really a one-element vector, not a scalar.
    # Test: same mix as the success case but the sparse tensor is 30x40; expects a value error
    # "dimension 1 must have size 30".
    dense_scalar = constant_op.constant([42], dtypes.float32)
    sparse_2d = sparse_tensor.SparseTensor(
        constant_op.constant([[5, 6], [7, 8]], dtypes.int64),
        constant_op.constant([23, -43], dtypes.int32),
        constant_op.constant([30, 40], dtypes.int64))
    with self.assertRaisesRegex(ValueError, r"dimension 1 must have size 30"):
      assertion = check_ops.assert_shapes([(dense_scalar, []),
                                           (sparse_2d, ["N", "N"])])
      with ops.control_dependencies([assertion]):
        out = array_ops.identity(sparse_2d)
      self.evaluate(out)


class IsStrictlyIncreasingTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the strictly-increasing predicate, which evaluates to a boolean over the
  # flattened tensor: constant, decreasing, increasing, rank-two, single-element and empty inputs.

  @test_util.run_in_graph_and_eager_modes
  def test_constant_tensor_is_not_strictly_increasing(self):
    # [FUNCTION DOC]
    # Test: [1,1,1] is not strictly increasing; the evaluated result must be false.
    self.assertFalse(self.evaluate(check_ops.is_strictly_increasing([1, 1, 1])))

  @test_util.run_in_graph_and_eager_modes
  def test_decreasing_tensor_is_not_strictly_increasing(self):
    # [FUNCTION DOC]
    # Test: [1,0,-1] is not strictly increasing; the evaluated result must be false.
    self.assertFalse(self.evaluate(
        check_ops.is_strictly_increasing([1, 0, -1])))

  @test_util.run_in_graph_and_eager_modes
  def test_2d_decreasing_tensor_is_not_strictly_increasing(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name says "decreasing" but the input [[1,3],[2,4]] is not decreasing; in
    # row-major order it goes 1,3,2,4 (non-monotonic), which is why it is not increasing.
    # Test: that rank-two input yields false.
    self.assertFalse(
        self.evaluate(check_ops.is_strictly_increasing([[1, 3], [2, 4]])))

  @test_util.run_in_graph_and_eager_modes
  def test_increasing_tensor_is_increasing(self):
    # [FUNCTION DOC]
    # Test: [1,2,3] is strictly increasing; the evaluated result must be true.
    self.assertTrue(self.evaluate(check_ops.is_strictly_increasing([1, 2, 3])))

  @test_util.run_in_graph_and_eager_modes
  def test_increasing_rank_two_tensor(self):
    # [FUNCTION DOC]
    # Test: [[-1,2],[3,4]] is strictly increasing in row-major order; result must be true.
    self.assertTrue(
        self.evaluate(check_ops.is_strictly_increasing([[-1, 2], [3, 4]])))

  @test_util.run_in_graph_and_eager_modes
  def test_tensor_with_one_element_is_strictly_increasing(self):
    # [FUNCTION DOC]
    # Test: a single-element tensor is trivially strictly increasing; result must be true.
    self.assertTrue(self.evaluate(check_ops.is_strictly_increasing([1])))

  @test_util.run_in_graph_and_eager_modes
  def test_empty_tensor_is_strictly_increasing(self):
    # [FUNCTION DOC]
    # Test: an empty tensor is trivially strictly increasing; result must be true.
    self.assertTrue(self.evaluate(check_ops.is_strictly_increasing([])))


class IsNonDecreasingTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the non-decreasing predicate, which evaluates to a boolean over the flattened
  # tensor: constant, decreasing, increasing, rank-two, single-element and empty inputs.

  @test_util.run_in_graph_and_eager_modes
  def test_constant_tensor_is_non_decreasing(self):
    # [FUNCTION DOC]
    # Test: [1,1,1] is non-decreasing; the evaluated result must be true.
    self.assertTrue(self.evaluate(check_ops.is_non_decreasing([1, 1, 1])))

  @test_util.run_in_graph_and_eager_modes
  def test_decreasing_tensor_is_not_non_decreasing(self):
    # [FUNCTION DOC]
    # Test: [3,2,1] is not non-decreasing; the evaluated result must be false.
    self.assertFalse(self.evaluate(check_ops.is_non_decreasing([3, 2, 1])))

  @test_util.run_in_graph_and_eager_modes
  def test_2d_decreasing_tensor_is_not_non_decreasing(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name says "decreasing" but the input [[1,3],[2,4]] is not decreasing; in
    # row-major order it goes 1,3,2,4 (non-monotonic), which is why it is not non-decreasing.
    # Test: that rank-two input yields false.
    self.assertFalse(self.evaluate(
        check_ops.is_non_decreasing([[1, 3], [2, 4]])))

  @test_util.run_in_graph_and_eager_modes
  def test_increasing_rank_one_tensor_is_non_decreasing(self):
    # [FUNCTION DOC]
    # Test: [1,2,3] is non-decreasing; the evaluated result must be true.
    self.assertTrue(self.evaluate(check_ops.is_non_decreasing([1, 2, 3])))

  @test_util.run_in_graph_and_eager_modes
  def test_increasing_rank_two_tensor(self):
    # [FUNCTION DOC]
    # Test: [[-1,2],[3,3]] is non-decreasing (equal neighbours allowed); result must be true.
    self.assertTrue(self.evaluate(
        check_ops.is_non_decreasing([[-1, 2], [3, 3]])))

  @test_util.run_in_graph_and_eager_modes
  def test_tensor_with_one_element_is_non_decreasing(self):
    # [FUNCTION DOC]
    # Test: a single-element tensor is trivially non-decreasing; result must be true.
    self.assertTrue(self.evaluate(check_ops.is_non_decreasing([1])))

  @test_util.run_in_graph_and_eager_modes
  def test_empty_tensor_is_non_decreasing(self):
    # [FUNCTION DOC]
    # Test: an empty tensor is trivially non-decreasing; result must be true.
    self.assertTrue(self.evaluate(check_ops.is_non_decreasing([])))


class FloatDTypeTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the same-float-dtype helper, which returns the single common floating dtype
  # of a list of tensors and an optional expected dtype (defaulting to float32 when nothing
  # constrains it) and raises a value error on conflicts or non-float dtypes.

  @test_util.run_in_graph_and_eager_modes
  def test_assert_same_float_dtype(self):
    # [FUNCTION DOC]
    # Test: dtype resolution for many input combinations.
    # Asserts the result is float32 for null/empty/null-element lists with or without an explicit
    # float32; float32 for a float32 constant or sparse tensor with float32; and a value error for
    # mismatches with int32 or float64, for a float tensor mixed with an int tensor, and for an
    # int-only list. A list mixing a float32 dense and float32 sparse tensor yields float32.
    self.assertIs(dtypes.float32,
                  check_ops.assert_same_float_dtype(None, None))
    self.assertIs(dtypes.float32, check_ops.assert_same_float_dtype([], None))
    self.assertIs(dtypes.float32,
                  check_ops.assert_same_float_dtype([], dtypes.float32))
    self.assertIs(dtypes.float32,
                  check_ops.assert_same_float_dtype(None, dtypes.float32))
    self.assertIs(dtypes.float32,
                  check_ops.assert_same_float_dtype([None, None], None))
    self.assertIs(
        dtypes.float32,
        check_ops.assert_same_float_dtype([None, None], dtypes.float32))

    const_float = constant_op.constant(3.0, dtype=dtypes.float32)
    self.assertIs(
        dtypes.float32,
        check_ops.assert_same_float_dtype([const_float], dtypes.float32))
    self.assertRaises(ValueError, check_ops.assert_same_float_dtype,
                      [const_float], dtypes.int32)

    sparse_float = sparse_tensor.SparseTensor(
        constant_op.constant([[111], [232]], dtypes.int64),
        constant_op.constant([23.4, -43.2], dtypes.float32),
        constant_op.constant([500], dtypes.int64))
    self.assertIs(dtypes.float32,
                  check_ops.assert_same_float_dtype([sparse_float],
                                                    dtypes.float32))
    self.assertRaises(ValueError, check_ops.assert_same_float_dtype,
                      [sparse_float], dtypes.int32)
    self.assertRaises(ValueError, check_ops.assert_same_float_dtype,
                      [const_float, None, sparse_float], dtypes.float64)

    self.assertIs(dtypes.float32,
                  check_ops.assert_same_float_dtype(
                      [const_float, sparse_float]))
    self.assertIs(dtypes.float32,
                  check_ops.assert_same_float_dtype(
                      [const_float, sparse_float], dtypes.float32))

    const_int = constant_op.constant(3, dtype=dtypes.int32)
    self.assertRaises(ValueError, check_ops.assert_same_float_dtype,
                      [sparse_float, const_int])
    self.assertRaises(ValueError, check_ops.assert_same_float_dtype,
                      [sparse_float, const_int], dtypes.int32)
    self.assertRaises(ValueError, check_ops.assert_same_float_dtype,
                      [sparse_float, const_int], dtypes.float32)
    self.assertRaises(ValueError, check_ops.assert_same_float_dtype,
                      [const_int])


class AssertScalarTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the scalar assertion: scalar tensors and Python scalars (numbers, strings) pass;
  # a vector raises a value error.

  @test_util.run_in_graph_and_eager_modes
  def test_assert_scalar(self):
    # [FUNCTION DOC]
    # Test: the scalar assertion accepts an integer constant, a string constant, a Python integer
    # and a Python string without error, and raises a value error matching "Expected scalar" for a
    # two-element vector constant. Runs in graph and eager modes.
    check_ops.assert_scalar(constant_op.constant(3))
    check_ops.assert_scalar(constant_op.constant("foo"))
    check_ops.assert_scalar(3)
    check_ops.assert_scalar("foo")
    with self.assertRaisesRegex(ValueError, "Expected scalar"):
      check_ops.assert_scalar(constant_op.constant([3, 4]))


if __name__ == "__main__":
  test.main()
