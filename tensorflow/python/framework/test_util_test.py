# [MODULE DOC]
# Test module for the framework test utilities (the test_util module): the base test case class
# and its assertion helpers (proto equality, approximate array equality, range and set checks,
# shape checks, graph-definition comparison, checked threads), the decorators that run tests in
# graph and/or eager modes, skip helpers, leak-detection decorators, run-functions-eagerly and
# device synchronization utilities. It also defines a small data class used to test evaluating
# custom nested structures. Executed as a script it starts the standard test runner.
# No import-time side effects beyond imports.
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
"""Tests for tensorflow.ops.test_util."""

import collections
import copy
import dataclasses
import random
import sys
import threading
import time
import unittest
import weakref

from absl.testing import parameterized
import numpy as np

from google.protobuf import text_format
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python import pywrap_sanitizers
from tensorflow.python.compat import compat
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import combinations
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor
from tensorflow.python.framework import test_ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import googletest
from tensorflow.python.util.protobuf import compare_test_pb2


@dataclasses.dataclass
class MaskedTensor:
  # [CLASS DOC]
  # HELPER: small test fixture type used to check that evaluation handles custom nested types.
  # Data class with two fields: a boolean mask and a tensor value. It implements the flatten and
  # unflatten protocol (mask is metadata, value is the only component) so that TensorFlow can
  # traverse and rebuild it, and defines equality as equal mask and equal value.
  mask: bool
  value: tensor.Tensor

  def __tf_flatten__(self):
    # [FUNCTION DOC]
    # HELPER: part of the flatten protocol for the MaskedTensor fixture.
    # Returns a pair: metadata holding the mask (as a one-element tuple) and components holding
    # the value tensor (as a one-element tuple). No side effects.
    metadata = (self.mask,)
    components = (self.value,)
    return metadata, components

  @classmethod
  def __tf_unflatten__(cls, metadata, components):
    # [FUNCTION DOC]
    # HELPER: class-level inverse of the flatten protocol for the MaskedTensor fixture.
    # Inputs: metadata (tuple whose first item is the mask) and components (tuple whose first
    # item is the value). Returns a new MaskedTensor built from them. No side effects.
    mask = metadata[0]
    value = components[0]
    return MaskedTensor(mask=mask, value=value)

  def __eq__(self, other):
    # [FUNCTION DOC]
    # HELPER: equality for the MaskedTensor fixture.
    # Input: another MaskedTensor-like object with mask and value fields. Returns true when the
    # masks are equal and the values compare equal (note the value comparison is whatever the
    # value type's equality returns, so for tensors it may be a tensor rather than a plain
    # boolean). Raises an attribute error if the other object lacks those fields.
    return self.mask == other.mask and self.value == other.value


class TestUtilTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Main test case for the test utilities: exercises the base class assertions, decorators for
  # running in graph and eager modes, evaluation of nested structures, forward-compatibility
  # horizons and related helpers. Tests are independent and create their own tensors/graphs.

  def test_assert_ops_in_graph(self):
    # [FUNCTION DOC]
    # In a fresh graph creates a constant named hello. Expects the ops-in-graph assertion to
    # pass for the correct name and type pair, and to raise a value error both for a missing op
    # name and for an existing name with the wrong op type.
    with ops.Graph().as_default():
      constant_op.constant(["hello", "taffy"], name="hello")
      test_util.assert_ops_in_graph({"hello": "Const"}, ops.get_default_graph())

      self.assertRaises(ValueError, test_util.assert_ops_in_graph,
                        {"bye": "Const"}, ops.get_default_graph())

      self.assertRaises(ValueError, test_util.assert_ops_in_graph,
                        {"hello": "Variable"}, ops.get_default_graph())

  @test_util.run_deprecated_v1
  def test_session_functions(self):
    # [FUNCTION DOC]
    # Graph-mode (deprecated v1) test of session helpers. Verifies that the cached session
    # returns the same session as test_session, that requesting a cached session with a
    # different graph or with forced GPU raises a value error, that the cached session stays
    # open after its with-scope ends, and that a plain session object is distinct from the
    # cached one and is closed when its with-scope ends.
    with self.test_session() as sess:
      sess_ref = weakref.ref(sess)
      with self.cached_session(graph=None, config=None) as sess2:
        # We make sure that sess2 is sess.
        assert sess2 is sess
        # We make sure we raise an exception if we use cached_session with
        # different values.
        with self.assertRaises(ValueError):
          with self.cached_session(graph=ops.Graph()) as sess2:
            pass
        with self.assertRaises(ValueError):
          with self.cached_session(force_gpu=True) as sess2:
            pass
    # We make sure that test_session will cache the session even after the
    # with scope.
    assert not sess_ref()._closed
    with self.session() as unique_sess:
      unique_sess_ref = weakref.ref(unique_sess)
      with self.session() as sess2:
        assert sess2 is not unique_sess
    # We make sure the session is closed when we leave the with statement.
    assert unique_sess_ref()._closed

  def test_assert_equal_graph_def(self):
    # [FUNCTION DOC]
    # Builds an empty graph definition and two definitions with the same two constants created
    # in opposite order. Expects their text forms to differ, the graph-definition equality
    # assertion to treat the two as equal (order independent), and to fail with an unexpected
    # node message when compared against the empty definition.
    with ops.Graph().as_default() as g:
      def_empty = g.as_graph_def()
      constant_op.constant(5, name="five")
      constant_op.constant(7, name="seven")
      def_57 = g.as_graph_def()
    with ops.Graph().as_default() as g:
      constant_op.constant(7, name="seven")
      constant_op.constant(5, name="five")
      def_75 = g.as_graph_def()
    # Comparing strings is order dependent
    self.assertNotEqual(str(def_57), str(def_75))
    # assert_equal_graph_def doesn't care about order
    test_util.assert_equal_graph_def(def_57, def_75)
    # Compare two unequal graphs
    with self.assertRaisesRegex(AssertionError,
                                r"^Found unexpected node '{{node seven}}"):
      test_util.assert_equal_graph_def(def_57, def_empty)

  def test_assert_equal_graph_def_hash_table(self):
    # [FUNCTION DOC]
    # Builds two graph definitions each containing a static hash table with a unique generated
    # shared name. Expects the equality assertion to fail mentioning the hash table name when
    # shared names are compared, and to pass when the option to ignore hash table shared names is
    # set (that option modifies the definitions in place).
    def get_graph_def():
      # [FUNCTION DOC]
      # HELPER: builds the graph definition compared by the hash table test.
      # Creates, in a new graph, constants for lookup input, keys, values and default, a static
      # hash table from them, and a lookup of the input; returns the graph definition.
      with ops.Graph().as_default() as g:
        x = constant_op.constant([2, 9], name="x")
        keys = constant_op.constant([1, 2], name="keys")
        values = constant_op.constant([3, 4], name="values")
        default = constant_op.constant(-1, name="default")
        table = lookup_ops.StaticHashTable(
            lookup_ops.KeyValueTensorInitializer(keys, values), default)
        _ = table.lookup(x)
      return g.as_graph_def()
    def_1 = get_graph_def()
    def_2 = get_graph_def()
    # The unique shared_name of each table makes the graph unequal.
    with self.assertRaisesRegex(AssertionError, "hash_table_"):
      test_util.assert_equal_graph_def(def_1, def_2,
                                       hash_table_shared_name=False)
    # That can be ignored. (NOTE: modifies GraphDefs in-place.)
    test_util.assert_equal_graph_def(def_1, def_2,
                                     hash_table_shared_name=True)

  def testIsGoogleCudaEnabled(self):
    # [FUNCTION DOC]
    # Smoke test: calls the Google CUDA build query and prints whether it is enabled. Asserts
    # nothing; only ensures the native wrapper function exists and is callable.
    # The test doesn't assert anything. It ensures the py wrapper
    # function is generated correctly.
    if test_util.IsGoogleCudaEnabled():
      print("GoogleCuda is enabled")
    else:
      print("GoogleCuda is disabled")

  def testIsMklEnabled(self):
    # [FUNCTION DOC]
    # Smoke test: calls the MKL build query and prints whether MKL is enabled. Asserts nothing;
    # only ensures the native wrapper function exists and is callable.
    # This test doesn't assert anything.
    # It ensures the py wrapper function is generated correctly.
    if test_util.IsMklEnabled():
      print("MKL is enabled")
    else:
      print("MKL is disabled")

  @test_util.disable_asan("Skip test if ASAN is enabled.")
  def testDisableAsan(self):
    # [FUNCTION DOC]
    # Decorated to be skipped when the address sanitizer is enabled; when it runs, asserts the
    # sanitizer query reports that ASAN is not enabled.
    self.assertFalse(pywrap_sanitizers.is_asan_enabled())

  @test_util.disable_msan("Skip test if MSAN is enabled.")
  def testDisableMsan(self):
    # [FUNCTION DOC]
    # Decorated to be skipped when the memory sanitizer is enabled; when it runs, asserts the
    # sanitizer query reports that MSAN is not enabled.
    self.assertFalse(pywrap_sanitizers.is_msan_enabled())

  @test_util.disable_tsan("Skip test if TSAN is enabled.")
  def testDisableTsan(self):
    # [FUNCTION DOC]
    # Decorated to be skipped when the thread sanitizer is enabled; when it runs, asserts the
    # sanitizer query reports that TSAN is not enabled.
    self.assertFalse(pywrap_sanitizers.is_tsan_enabled())

  @test_util.disable_ubsan("Skip test if UBSAN is enabled.")
  def testDisableUbsan(self):
    # [FUNCTION DOC]
    # Decorated to be skipped when the undefined-behavior sanitizer is enabled; when it runs,
    # asserts the sanitizer query reports that UBSAN is not enabled.
    self.assertFalse(pywrap_sanitizers.is_ubsan_enabled())

  @test_util.run_in_graph_and_eager_modes
  def testAssertProtoEqualsStr(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. Parses a text-format graph definition and expects the proto
    # equality assertion to accept the text form against the parsed message, and a message
    # against itself.

    graph_str = "node { name: 'w1' op: 'params' }"
    graph_def = graph_pb2.GraphDef()
    text_format.Parse(graph_str, graph_def)

    # test string based comparison
    self.assertProtoEquals(graph_str, graph_def)

    # test original comparison
    self.assertProtoEquals(graph_def, graph_def)

  @test_util.run_in_graph_and_eager_modes
  def testAssertProtoEqualsAny(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. Builds an outer meta graph definition that packs an inner
    # one into a generic Any field. Expects proto equality to accept the text form and the
    # message itself, and the failure message produced when comparing against an empty string to
    # include the inner proto's content.
    # Test assertProtoEquals with a protobuf.Any field.
    meta_graph_def_str = """
    meta_info_def {
      meta_graph_version: "outer"
      any_info {
        [type.googleapis.com/tensorflow.MetaGraphDef] {
          meta_info_def {
            meta_graph_version: "inner"
          }
        }
      }
    }
    """
    meta_graph_def_outer = meta_graph_pb2.MetaGraphDef()
    meta_graph_def_outer.meta_info_def.meta_graph_version = "outer"
    meta_graph_def_inner = meta_graph_pb2.MetaGraphDef()
    meta_graph_def_inner.meta_info_def.meta_graph_version = "inner"
    meta_graph_def_outer.meta_info_def.any_info.Pack(meta_graph_def_inner)
    self.assertProtoEquals(meta_graph_def_str, meta_graph_def_outer)
    self.assertProtoEquals(meta_graph_def_outer, meta_graph_def_outer)

    # Check if the assertion failure message contains the content of
    # the inner proto.
    with self.assertRaisesRegex(AssertionError, r'meta_graph_version: "inner"'):
      self.assertProtoEquals("", meta_graph_def_outer)

  def test_float_relative_tolerance(self):
    # [FUNCTION DOC]
    # Compares two float messages that differ in the sixth significant digit. Expects the proto
    # equality assertion to fail when a very tight relative tolerance is supplied.
    pb1 = compare_test_pb2.Floats(float_=65061.0420, double_=164107.8938)
    pb2 = compare_test_pb2.Floats(float_=65061.0322, double_=164107.9087)
    self.assertRaises(
        AssertionError,
        self.assertProtoEquals,
        pb1,
        pb2,
        relative_tolerance=1e-7,
    )

  def test_float_relative_tolerance_nan(self):
    # [FUNCTION DOC]
    # Two messages holding not-a-number values must compare equal under a relative tolerance,
    # while a not-a-number against the number two must fail the equality assertion.
    pb1 = compare_test_pb2.Floats(float_=float("nan"))
    pb2 = compare_test_pb2.Floats(float_=float("nan"))
    self.assertProtoEquals(pb1, pb2, relative_tolerance=1e-7)
    pb2 = compare_test_pb2.Floats(float_=2)
    self.assertRaises(
        AssertionError,
        self.assertProtoEquals,
        pb1,
        pb2,
        relative_tolerance=1e-7,
    )

  def test_float_relative_tolerance_inf(self):
    # [FUNCTION DOC]
    # Two messages holding positive infinity must compare equal under a relative tolerance,
    # while infinity against the number one must fail the equality assertion.
    pb1 = compare_test_pb2.Floats(float_=float("inf"))
    pb2 = compare_test_pb2.Floats(float_=float("inf"))
    self.assertProtoEquals(pb1, pb2, relative_tolerance=1e-5)
    pb1 = compare_test_pb2.Floats(float_=1)
    self.assertRaises(
        AssertionError,
        self.assertProtoEquals,
        pb1,
        pb2,
        relative_tolerance=1e-7,
    )

  def test_float_relative_tolerance_denormal(self):
    # [FUNCTION DOC]
    # Two messages holding the same denormal (subnormal) float value must compare equal under a
    # relative tolerance, verifying that tiny values do not break the relative comparison.
    pb1 = compare_test_pb2.Floats(
        float_=sys.float_info.min * sys.float_info.epsilon
    )
    pb2 = compare_test_pb2.Floats(
        float_=sys.float_info.min * sys.float_info.epsilon
    )
    self.assertProtoEquals(pb1, pb2, relative_tolerance=1e-5)

  def test_repeated_float_relative_tolerance(self):
    # [FUNCTION DOC]
    # Messages with repeated float fields whose last element differs slightly. Expects failure
    # with a very tight relative tolerance and success with a looser one.
    pb1 = compare_test_pb2.RepeatedFloats(
        float_=(x for x in [1.0, 2.0, 65061.042])
    )
    pb2 = compare_test_pb2.RepeatedFloats(
        float_=(x for x in [1.0, 2.0, 65061.0322])
    )
    self.assertRaises(
        AssertionError,
        self.assertProtoEquals,
        pb1,
        pb2,
        relative_tolerance=1e-7,
    )
    self.assertProtoEquals(pb1, pb2, relative_tolerance=1e-5)

  def test_nested_float_relative_tolerance(self):
    # [FUNCTION DOC]
    # Messages holding a nested message with a float that differs slightly. Expects failure with
    # a very tight relative tolerance and success with a looser one.
    pb1 = compare_test_pb2.NestedFloats()
    pb2 = compare_test_pb2.NestedFloats()
    pb1.floats.float_ = 65061.0420
    pb2.floats.float_ = 65061.0322
    self.assertRaises(
        AssertionError,
        self.assertProtoEquals,
        pb1,
        pb2,
        relative_tolerance=1e-7,
    )
    self.assertProtoEquals(pb1, pb2, relative_tolerance=1e-5)

  def test_map_float_relative_tolerance(self):
    # [FUNCTION DOC]
    # Messages holding a map from integers to messages with slightly different float and double
    # values. Expects failure with a very tight relative tolerance and success with a looser one.
    pb1 = compare_test_pb2.MapFloats()
    pb2 = compare_test_pb2.MapFloats()
    pb1.int_to_floats[1].float_ = 65061.0420
    pb2.int_to_floats[1].float_ = 65061.0322
    pb1.int_to_floats[1].double_ = 164107.8938
    pb2.int_to_floats[1].double_ = 164107.9087
    self.assertRaises(
        AssertionError,
        self.assertProtoEquals,
        pb1,
        pb2,
        relative_tolerance=1e-7,
    )
    self.assertProtoEquals(pb1, pb2, relative_tolerance=1e-5)

  @test_util.run_in_graph_and_eager_modes
  def testNDArrayNear(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. Expects the private array-nearness helper to return true for
    # identical two-by-three arrays and false for arrays differing by a large amount, with a tiny
    # tolerance.
    a1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    a2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    a3 = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    self.assertTrue(self._NDArrayNear(a1, a2, 1e-5))
    self.assertFalse(self._NDArrayNear(a1, a3, 1e-5))

  @test_util.run_in_graph_and_eager_modes
  def testCheckedThreadSucceeds(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. Starts a checked thread whose target sets an event, joins it
    # without error, and expects the event to have been set (and to be unset beforehand).

    def noop(ev):
      # [FUNCTION DOC]
      # HELPER: thread target for the success test. Input: an event object; sets it and returns
      # nothing.
      ev.set()

    event_arg = threading.Event()

    self.assertFalse(event_arg.is_set())
    t = self.checkedThread(target=noop, args=(event_arg,))
    t.start()
    t.join()
    self.assertTrue(event_arg.is_set())

  @test_util.run_in_graph_and_eager_modes
  def testCheckedThreadFails(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. Starts a checked thread whose target divides by zero and
    # expects joining it to raise the test failure exception, with a message containing the
    # integer division by zero text.

    def err_func():
      # [FUNCTION DOC]
      # HELPER: thread target that always fails. Takes no input and raises a division-by-zero
      # error (it never returns normally).
      return 1 // 0

    t = self.checkedThread(target=err_func)
    t.start()
    with self.assertRaises(self.failureException) as fe:
      t.join()
    self.assertTrue("integer division or modulo by zero" in str(fe.exception))

  @test_util.run_in_graph_and_eager_modes
  def testCheckedThreadWithWrongAssertionFails(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. Starts a checked thread whose target asserts a false
    # condition and expects joining it to raise the test failure exception whose message says
    # that false is not true.
    x = 37

    def err_func():
      # [FUNCTION DOC]
      # HELPER: thread target that asserts that an enclosing number is below ten; since it is
      # thirty-seven the assertion fails. Takes no input, returns nothing.
      self.assertTrue(x < 10)

    t = self.checkedThread(target=err_func)
    t.start()
    with self.assertRaises(self.failureException) as fe:
      t.join()
    self.assertTrue("False is not true" in str(fe.exception))

  @test_util.run_in_graph_and_eager_modes
  def testMultipleThreadsWithOneFailure(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. Starts ten checked threads where only the one given index
    # seven fails its assertion. Expects joining that thread to raise the test failure exception
    # and joining all the others to succeed.

    def err_func(i):
      # [FUNCTION DOC]
      # HELPER: thread target. Input: integer index; asserts it differs from seven, so only the
      # thread with index seven fails. Returns nothing.
      self.assertTrue(i != 7)

    threads = [
        self.checkedThread(
            target=err_func, args=(i,)) for i in range(10)
    ]
    for t in threads:
      t.start()
    for i, t in enumerate(threads):
      if i == 7:
        with self.assertRaises(self.failureException):
          t.join()
      else:
        t.join()

  def _WeMustGoDeeper(self, msg):
    # [FUNCTION DOC]
    # HELPER: shared by the op-error assertion test.
    # Input: msg, text expected in the error. Inside the op-error assertion context, in a new
    # graph, builds an original op and an op derived from it, then raises an unauthenticated error
    # carrying the derived op and the message "true_err". Fails with an assertion error if the
    # error text (including op names) does not match msg.
    with self.assertRaisesOpError(msg):
      with ops.Graph().as_default():
        node_def = ops._NodeDef("IntOutput", "name")
        node_def_orig = ops._NodeDef("IntOutput", "orig")
        op_orig = ops.Operation.from_node_def(
            node_def_orig, ops.get_default_graph()
        )
        op = ops.Operation.from_node_def(
            node_def, ops.get_default_graph(), original_op=op_orig
        )
        raise errors.UnauthenticatedError(node_def, op, "true_err")

  @test_util.run_in_graph_and_eager_modes
  def testAssertRaisesOpErrorDoesNotPassMessageDueToLeakedStack(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. Expects the op-error assertion to fail for an unrelated
    # message, and to succeed for the error message, the op name and the original op name.
    with self.assertRaises(AssertionError):
      self._WeMustGoDeeper("this_is_not_the_error_you_are_looking_for")

    self._WeMustGoDeeper("true_err")
    self._WeMustGoDeeper("name")
    self._WeMustGoDeeper("orig")

  @parameterized.named_parameters(
      dict(testcase_name="tensors", ragged_tensors=False),
      dict(testcase_name="ragged_tensors", ragged_tensors=True))
  @test_util.run_in_graph_and_eager_modes
  def testAllCloseTensors(self, ragged_tensors: bool):
    # [FUNCTION DOC]
    # Parameterized over plain tensors and ragged tensors; runs in graph and eager modes.
    # Input: ragged_tensors flag selecting conversion of the data to ragged tensors. Expects the
    # approximate-equality assertion to accept tensor versus tensor, tensor versus raw list,
    # dictionaries of them, and lists mixing tensors and raw data.
    a_raw_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    a = constant_op.constant(a_raw_data)
    b = math_ops.add(1, constant_op.constant([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
    if ragged_tensors:
      a = ragged_tensor.RaggedTensor.from_tensor(a)
      b = ragged_tensor.RaggedTensor.from_tensor(b)

    self.assertAllClose(a, b)
    self.assertAllClose(a, a_raw_data)

    a_dict = {"key": a}
    b_dict = {"key": b}
    self.assertAllClose(a_dict, b_dict)

    x_list = [a, b]
    y_list = [a_raw_data, b]
    self.assertAllClose(x_list, y_list)

  @test_util.run_in_graph_and_eager_modes
  def testAllCloseScalars(self):
    # [FUNCTION DOC]
    # Expects scalars differing by a tiny amount to be close, and scalars differing by a larger
    # amount to fail with a not-equal-to-tolerance message.
    self.assertAllClose(7, 7 + 1e-8)
    with self.assertRaisesRegex(AssertionError, r"Not equal to tolerance"):
      self.assertAllClose(7, 7 + 1e-5)

  @test_util.run_in_graph_and_eager_modes
  def testAllCloseList(self):
    # [FUNCTION DOC]
    # Expects comparing two different one-element lists to fail with a message about values not
    # being close.
    with self.assertRaisesRegex(AssertionError, r"not close dif"):
      self.assertAllClose([0], [1])

  @test_util.run_in_graph_and_eager_modes
  def testAllCloseDictToNonDict(self):
    # [FUNCTION DOC]
    # Expects comparing a dictionary with a non-dictionary (in either order) to raise a value
    # error saying a dictionary cannot be compared to a non-dictionary.
    with self.assertRaisesRegex(ValueError, r"Can't compare dict to non-dict"):
      self.assertAllClose(1, {"a": 1})
    with self.assertRaisesRegex(ValueError, r"Can't compare dict to non-dict"):
      self.assertAllClose({"a": 1}, 1)

  @test_util.run_in_graph_and_eager_modes
  def testAllCloseNamedtuples(self):
    # [FUNCTION DOC]
    # Expects a dictionary of values to be approximately equal to a named tuple with the same
    # field names and values, and a named tuple to equal an identical named tuple.
    a = 7
    b = (2., 3.)
    c = np.ones((3, 2, 4)) * 7.
    expected = {"a": a, "b": b, "c": c}
    my_named_tuple = collections.namedtuple("MyNamedTuple", ["a", "b", "c"])

    # Identity.
    self.assertAllClose(expected, my_named_tuple(a=a, b=b, c=c))
    self.assertAllClose(
        my_named_tuple(a=a, b=b, c=c), my_named_tuple(a=a, b=b, c=c))

  @test_util.run_in_graph_and_eager_modes
  def testAllCloseDicts(self):
    # [FUNCTION DOC]
    # Checks dictionary comparison: identical and copied dictionaries match; removing any key
    # fails with a mismatched keys message; changing a scalar, lengthening a tuple, or perturbing
    # one array element fails with a not-equal-to-tolerance or shape mismatch message.
    a = 7
    b = (2., 3.)
    c = np.ones((3, 2, 4)) * 7.
    expected = {"a": a, "b": b, "c": c}

    # Identity.
    self.assertAllClose(expected, expected)
    self.assertAllClose(expected, dict(expected))

    # With each item removed.
    for k in expected:
      actual = dict(expected)
      del actual[k]
      with self.assertRaisesRegex(AssertionError, r"mismatched keys"):
        self.assertAllClose(expected, actual)

    # With each item changed.
    with self.assertRaisesRegex(AssertionError, r"Not equal to tolerance"):
      self.assertAllClose(expected, {"a": a + 1e-5, "b": b, "c": c})
    with self.assertRaisesRegex(AssertionError, r"Shape mismatch"):
      self.assertAllClose(expected, {"a": a, "b": b + (4.,), "c": c})
    c_copy = np.array(c)
    c_copy[1, 1, 1] += 1e-5
    with self.assertRaisesRegex(AssertionError, r"Not equal to tolerance"):
      self.assertAllClose(expected, {"a": a, "b": b, "c": c_copy})

  @test_util.run_in_graph_and_eager_modes
  def testAllCloseListOfNamedtuples(self):
    # [FUNCTION DOC]
    # Expects a list of named tuples of arrays to be approximately equal to a list of plain tuples
    # of nested lists with the same values.
    my_named_tuple = collections.namedtuple("MyNamedTuple", ["x", "y"])
    l1 = [
        my_named_tuple(x=np.array([[2.3, 2.5]]), y=np.array([[0.97, 0.96]])),
        my_named_tuple(x=np.array([[3.3, 3.5]]), y=np.array([[0.98, 0.99]]))
    ]
    l2 = [
        ([[2.3, 2.5]], [[0.97, 0.96]]),
        ([[3.3, 3.5]], [[0.98, 0.99]]),
    ]
    self.assertAllClose(l1, l2)

  @test_util.run_in_graph_and_eager_modes
  def testAllCloseNestedStructure(self):
    # [FUNCTION DOC]
    # Expects a deeply nested structure to equal itself and a deep copy of itself, and after one
    # deeply nested value is altered, expects the failure message to include the path to that
    # value.
    a = {"x": np.ones((3, 2, 4)) * 7, "y": (2, [{"nested": {"m": 3, "n": 4}}])}
    self.assertAllClose(a, a)

    b = copy.deepcopy(a)
    self.assertAllClose(a, b)

    # Test mismatched values
    b["y"][1][0]["nested"]["n"] = 4.2
    with self.assertRaisesRegex(AssertionError,
                                r"\[y\]\[1\]\[0\]\[nested\]\[n\]"):
      self.assertAllClose(a, b)

  @test_util.run_in_graph_and_eager_modes
  def testAssertDictEqual(self):
    # [FUNCTION DOC]
    # Expects the dictionary equality assertion to accept a dictionary equal to itself and one
    # where an array entry is replaced by an equivalent constant tensor.
    a = 7
    b = (2., 3.)
    c = np.ones((3, 2, 4)) * 7.
    d = "testing123"
    expected = {"a": a, "b": b, "c": c, "d": d}
    actual = {"a": a, "b": b, "c": constant_op.constant(c), "d": d}

    self.assertDictEqual(expected, expected)
    self.assertDictEqual(expected, actual)

  @test_util.run_in_graph_and_eager_modes
  def testArrayNear(self):
    # [FUNCTION DOC]
    # Checks the array-nearness assertion: lists of different lengths raise an assertion error, a
    # flat list versus a nested list raises a type error, and equal lists pass.
    a = [1, 2]
    b = [1, 2, 5]
    with self.assertRaises(AssertionError):
      self.assertArrayNear(a, b, 0.001)
    a = [1, 2]
    b = [[1, 2], [3, 4]]
    with self.assertRaises(TypeError):
      self.assertArrayNear(a, b, 0.001)
    a = [1, 2]
    b = [1, 2]
    self.assertArrayNear(a, b, 0.001)

  @test_util.skip_if(True)  # b/117665998
  def testForceGPU(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test is permanently skipped by an unconditional skip decorator (bug
    # reference in the comment above it), so its body never runs; it is effectively dead code.
    # Intended scenario: forcing a GPU session and running an assert op that has no GPU kernel
    # should raise an invalid-argument error.
    with self.assertRaises(errors.InvalidArgumentError):
      with self.test_session(force_gpu=True):
        # this relies on us not having a GPU implementation for assert, which
        # seems sensible
        x = constant_op.constant(True)
        y = [15]
        control_flow_assert.Assert(x, y).run()

  @test_util.run_in_graph_and_eager_modes
  def testAssertAllCloseAccordingToType(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. Checks that tolerances are chosen by dtype: plain integers
    # use the default tolerances; float64 arrays and tensors use them as given; float32 inputs use
    # the float-specific tolerances; float16 inputs use the half-specific tolerances. For each
    # type a pass case and (for arrays) a fail case with a larger difference are verified.
    # test plain int
    self.assertAllCloseAccordingToType(1, 1, rtol=1e-8, atol=1e-8)

    # test float64
    self.assertAllCloseAccordingToType(
        np.asarray([1e-8], dtype=np.float64),
        np.asarray([2e-8], dtype=np.float64),
        rtol=1e-8, atol=1e-8
    )

    self.assertAllCloseAccordingToType(
        constant_op.constant([1e-8], dtype=dtypes.float64),
        constant_op.constant([2e-8], dtype=dtypes.float64),
        rtol=1e-8,
        atol=1e-8)

    with (self.assertRaises(AssertionError)):
      self.assertAllCloseAccordingToType(
          np.asarray([1e-7], dtype=np.float64),
          np.asarray([2e-7], dtype=np.float64),
          rtol=1e-8, atol=1e-8
      )

    # test float32
    self.assertAllCloseAccordingToType(
        np.asarray([1e-7], dtype=np.float32),
        np.asarray([2e-7], dtype=np.float32),
        rtol=1e-8, atol=1e-8,
        float_rtol=1e-7, float_atol=1e-7
    )

    self.assertAllCloseAccordingToType(
        constant_op.constant([1e-7], dtype=dtypes.float32),
        constant_op.constant([2e-7], dtype=dtypes.float32),
        rtol=1e-8,
        atol=1e-8,
        float_rtol=1e-7,
        float_atol=1e-7)

    with (self.assertRaises(AssertionError)):
      self.assertAllCloseAccordingToType(
          np.asarray([1e-6], dtype=np.float32),
          np.asarray([2e-6], dtype=np.float32),
          rtol=1e-8, atol=1e-8,
          float_rtol=1e-7, float_atol=1e-7
      )

    # test float16
    self.assertAllCloseAccordingToType(
        np.asarray([1e-4], dtype=np.float16),
        np.asarray([2e-4], dtype=np.float16),
        rtol=1e-8, atol=1e-8,
        float_rtol=1e-7, float_atol=1e-7,
        half_rtol=1e-4, half_atol=1e-4
    )

    self.assertAllCloseAccordingToType(
        constant_op.constant([1e-4], dtype=dtypes.float16),
        constant_op.constant([2e-4], dtype=dtypes.float16),
        rtol=1e-8,
        atol=1e-8,
        float_rtol=1e-7,
        float_atol=1e-7,
        half_rtol=1e-4,
        half_atol=1e-4)

    with (self.assertRaises(AssertionError)):
      self.assertAllCloseAccordingToType(
          np.asarray([1e-3], dtype=np.float16),
          np.asarray([2e-3], dtype=np.float16),
          rtol=1e-8, atol=1e-8,
          float_rtol=1e-7, float_atol=1e-7,
          half_rtol=1e-4, half_atol=1e-4
      )

  @test_util.run_in_graph_and_eager_modes
  def testAssertAllEqual(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. Creates a variable, a constant and their sum, initializes
    # variables, and expects exact-equality assertions to pass against lists of the right values
    # and to fail with a message containing the left-hand values when they differ.
    i = variables.Variable([100] * 3, dtype=dtypes.int32, name="i")
    j = constant_op.constant([20] * 3, dtype=dtypes.int32, name="j")
    k = math_ops.add(i, j, name="k")

    self.evaluate(variables.global_variables_initializer())
    self.assertAllEqual([100] * 3, i)
    self.assertAllEqual([120] * 3, k)
    self.assertAllEqual([20] * 3, j)

    with self.assertRaisesRegex(AssertionError, r"not equal lhs"):
      self.assertAllEqual([0] * 3, k)

  @test_util.run_in_graph_and_eager_modes
  def testAssertNotAllEqual(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. Expects the not-all-equal assertion to pass when values
    # differ (including with broadcasting), and to fail with a message saying the two values are
    # equal at all elements when they match, with any extra message appended when supplied.
    i = variables.Variable([100], dtype=dtypes.int32, name="i")
    j = constant_op.constant([20], dtype=dtypes.int32, name="j")
    k = math_ops.add(i, j, name="k")

    self.evaluate(variables.global_variables_initializer())
    self.assertNotAllEqual([100] * 3, i)
    self.assertNotAllEqual([120] * 3, k)
    self.assertNotAllEqual([20] * 3, j)

    with self.assertRaisesRegex(
        AssertionError, r"two values are equal at all elements. $"):
      self.assertNotAllEqual([120], k)

    with self.assertRaisesRegex(
        AssertionError, r"two values are equal at all elements.*extra message"):
      self.assertNotAllEqual([120], k, msg="extra message")

  @test_util.run_in_graph_and_eager_modes
  def testAssertNotAllClose(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. Expects the not-all-close assertion to pass for clearly
    # different arrays or tensors and to raise an assertion error for values that are close.
    # Test with arrays
    self.assertNotAllClose([0.1], [0.2])
    with self.assertRaises(AssertionError):
      self.assertNotAllClose([-1.0, 2.0], [-1.0, 2.0])

    # Test with tensors
    x = constant_op.constant([1.0, 1.0], name="x")
    y = math_ops.add(x, x)

    self.assertAllClose([2.0, 2.0], y)
    self.assertNotAllClose([0.9, 1.0], x)

    with self.assertRaises(AssertionError):
      self.assertNotAllClose([1.0, 1.0], x)

  @test_util.run_in_graph_and_eager_modes
  def testAssertNotAllCloseRTol(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. Expects that when a relative tolerance of twenty percent is
    # given, values within that tolerance make the not-all-close assertion fail (arrays and
    # tensors).
    # Test with arrays
    with self.assertRaises(AssertionError):
      self.assertNotAllClose([1.1, 2.1], [1.0, 2.0], rtol=0.2)

    # Test with tensors
    x = constant_op.constant([1.0, 1.0], name="x")
    y = math_ops.add(x, x)

    self.assertAllClose([2.0, 2.0], y)

    with self.assertRaises(AssertionError):
      self.assertNotAllClose([0.9, 1.0], x, rtol=0.2)

  @test_util.run_in_graph_and_eager_modes
  def testAssertNotAllCloseATol(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. Expects that when an absolute tolerance of 0.2 is given,
    # values within that tolerance make the not-all-close assertion fail (arrays and tensors).
    # Test with arrays
    with self.assertRaises(AssertionError):
      self.assertNotAllClose([1.1, 2.1], [1.0, 2.0], atol=0.2)

    # Test with tensors
    x = constant_op.constant([1.0, 1.0], name="x")
    y = math_ops.add(x, x)

    self.assertAllClose([2.0, 2.0], y)

    with self.assertRaises(AssertionError):
      self.assertNotAllClose([0.9, 1.0], x, atol=0.2)

  @test_util.run_in_graph_and_eager_modes
  def testAssertAllGreaterLess(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. Checks strict greater-than and less-than assertions on a
    # tensor: they pass with bounds below/above all elements and raise an assertion error when
    # any element violates the bound.
    x = constant_op.constant([100.0, 110.0, 120.0], dtype=dtypes.float32)
    y = constant_op.constant([10.0] * 3, dtype=dtypes.float32)
    z = math_ops.add(x, y)

    self.assertAllClose([110.0, 120.0, 130.0], z)

    self.assertAllGreater(x, 95.0)
    self.assertAllLess(x, 125.0)

    with self.assertRaises(AssertionError):
      self.assertAllGreater(x, 105.0)
    with self.assertRaises(AssertionError):
      self.assertAllGreater(x, 125.0)

    with self.assertRaises(AssertionError):
      self.assertAllLess(x, 115.0)
    with self.assertRaises(AssertionError):
      self.assertAllLess(x, 95.0)

  @test_util.run_in_graph_and_eager_modes
  def testAssertAllGreaterLessEqual(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. Checks greater-or-equal and less-or-equal assertions on a
    # tensor: they pass with suitable bounds and raise an assertion error when any element
    # violates the bound.
    x = constant_op.constant([100.0, 110.0, 120.0], dtype=dtypes.float32)
    y = constant_op.constant([10.0] * 3, dtype=dtypes.float32)
    z = math_ops.add(x, y)

    self.assertAllEqual([110.0, 120.0, 130.0], z)

    self.assertAllGreaterEqual(x, 95.0)
    self.assertAllLessEqual(x, 125.0)

    with self.assertRaises(AssertionError):
      self.assertAllGreaterEqual(x, 105.0)
    with self.assertRaises(AssertionError):
      self.assertAllGreaterEqual(x, 125.0)

    with self.assertRaises(AssertionError):
      self.assertAllLessEqual(x, 115.0)
    with self.assertRaises(AssertionError):
      self.assertAllLessEqual(x, 95.0)

  def testAssertAllInRangeWithNonNumericValuesFails(self):
    # [FUNCTION DOC]
    # Expects the in-range assertion to raise an assertion error for string, complex and boolean
    # tensors, since range checks only apply to real numeric values.
    s1 = constant_op.constant("Hello, ", name="s1")
    c = constant_op.constant([1 + 2j, -3 + 5j], name="c")
    b = constant_op.constant([False, True], name="b")

    with self.assertRaises(AssertionError):
      self.assertAllInRange(s1, 0.0, 1.0)
    with self.assertRaises(AssertionError):
      self.assertAllInRange(c, 0.0, 1.0)
    with self.assertRaises(AssertionError):
      self.assertAllInRange(b, 0, 1)

  @test_util.run_in_graph_and_eager_modes
  def testAssertAllInRange(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. Values equal to the bounds pass by default, and fail when the
    # lower bound, the upper bound, or both are marked open (exclusive).
    x = constant_op.constant([10.0, 15.0], name="x")
    self.assertAllInRange(x, 10, 15)

    with self.assertRaises(AssertionError):
      self.assertAllInRange(x, 10, 15, open_lower_bound=True)
    with self.assertRaises(AssertionError):
      self.assertAllInRange(x, 10, 15, open_upper_bound=True)
    with self.assertRaises(AssertionError):
      self.assertAllInRange(
          x, 10, 15, open_lower_bound=True, open_upper_bound=True)

  @test_util.run_in_graph_and_eager_modes
  def testAssertAllInRangeScalar(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. For scalar tensors: a value inside the range passes, a
    # not-a-number fails, a value on an open bound fails, and an out-of-range value fails.
    x = constant_op.constant(10.0, name="x")
    nan = constant_op.constant(np.nan, name="nan")
    self.assertAllInRange(x, 5, 15)
    with self.assertRaises(AssertionError):
      self.assertAllInRange(nan, 5, 15)

    with self.assertRaises(AssertionError):
      self.assertAllInRange(x, 10, 15, open_lower_bound=True)
    with self.assertRaises(AssertionError):
      self.assertAllInRange(x, 1, 2)

  @test_util.run_in_graph_and_eager_modes
  def testAssertAllInRangeErrorMessageEllipses(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. With a large array whose values exceed the range, expects an
    # assertion error (exercising the failure message path that elides many violations).
    x_init = np.array([[10.0, 15.0]] * 12)
    x = constant_op.constant(x_init, name="x")
    with self.assertRaises(AssertionError):
      self.assertAllInRange(x, 5, 10)

  @test_util.run_in_graph_and_eager_modes
  def testAssertAllInRangeDetectsNaNs(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. A tensor containing not-a-number and infinite values must
    # fail the in-range assertion.
    x = constant_op.constant(
        [[np.nan, 0.0], [np.nan, np.inf], [np.inf, np.nan]], name="x")
    with self.assertRaises(AssertionError):
      self.assertAllInRange(x, 0.0, 2.0)

  @test_util.run_in_graph_and_eager_modes
  def testAssertAllInRangeWithInfinities(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. An infinite upper bound accepts an infinite value, but the
    # assertion fails when that upper bound is marked open (exclusive).
    x = constant_op.constant([10.0, np.inf], name="x")
    self.assertAllInRange(x, 10, np.inf)
    with self.assertRaises(AssertionError):
      self.assertAllInRange(x, 10, np.inf, open_upper_bound=True)

  @test_util.run_in_graph_and_eager_modes
  def testAssertAllInSet(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. Boolean and integer tensors pass the in-set assertion when
    # the allowed values are given as a list, tuple or set, and fail when an element is missing
    # from the allowed collection.
    b = constant_op.constant([True, False], name="b")
    x = constant_op.constant([13, 37], name="x")

    self.assertAllInSet(b, [False, True])
    self.assertAllInSet(b, (False, True))
    self.assertAllInSet(b, {False, True})
    self.assertAllInSet(x, [0, 13, 37, 42])
    self.assertAllInSet(x, (0, 13, 37, 42))
    self.assertAllInSet(x, {0, 13, 37, 42})

    with self.assertRaises(AssertionError):
      self.assertAllInSet(b, [False])
    with self.assertRaises(AssertionError):
      self.assertAllInSet(x, (42,))

  @test_util.run_in_graph_and_eager_modes
  def testAssertShapeEqualSameInputTypes(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. The shape-equality assertion passes for two arrays of the
    # same shape and for two tensors of the same shape, and fails for differing shapes.
    # Test with arrays
    array_a = np.random.rand(3, 1)
    array_b = np.random.rand(3, 1)
    array_c = np.random.rand(4, 2)

    self.assertShapeEqual(array_a, array_b)
    with self.assertRaises(AssertionError):
      self.assertShapeEqual(array_a, array_c)

    # Test with tensors
    tensor_x = random_ops.random_uniform((5, 2, 1))
    tensor_y = random_ops.random_uniform((5, 2, 1))
    tensor_z = random_ops.random_uniform((2, 4))

    self.assertShapeEqual(tensor_x, tensor_y)
    with self.assertRaises(AssertionError):
      self.assertShapeEqual(tensor_x, tensor_z)

  @test_util.run_in_graph_and_eager_modes
  def testAssertShapeEqualMixedInputTypes(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. The shape-equality assertion works with an array and a
    # tensor in either order, for multi-dimensional and one-element shapes, and fails when
    # shapes differ.

    # Test mixed multi-dimensional inputs
    array_input = np.random.rand(4, 3, 2)
    tensor_input = random_ops.random_uniform((4, 3, 2))
    tensor_input_2 = random_ops.random_uniform((10, 5))

    self.assertShapeEqual(array_input, tensor_input)
    self.assertShapeEqual(tensor_input, array_input)
    with self.assertRaises(AssertionError):
      self.assertShapeEqual(array_input, tensor_input_2)

    # Test with scalar inputs
    array_input = np.random.rand(1)
    tensor_input = random_ops.random_uniform((1,))
    tensor_input_2 = random_ops.random_uniform((3, 1))

    self.assertShapeEqual(array_input, tensor_input)
    self.assertShapeEqual(tensor_input, array_input)
    with self.assertRaises(AssertionError):
      self.assertShapeEqual(array_input, tensor_input_2)

  def testAssertShapeEqualDynamicShapes(self):
    # [FUNCTION DOC]
    # Uses a unique op whose output length is data dependent. In eager mode the shape is known so
    # the shape-equality assertion passes against an array of matching length. In graph mode the
    # shape is unknown when traced but not evaluated, so the assertion must fail.

    array_a = np.random.rand(4)
    values = [1, 1, 2, 3, 4, 4]

    # Dynamic shape should be resolved in eager execution.
    with context.eager_mode():
      tensor_b = array_ops.unique(values)[0]
      self.assertShapeEqual(array_a, tensor_b)

    # Shape comparison should fail when a graph is traced but not evaluated.
    with context.graph_mode():
      tensor_c = array_ops.unique(values)[0]
      with self.assertRaises(AssertionError):
        self.assertShapeEqual(array_a, tensor_c)

  def testRandomSeed(self):
    # [FUNCTION DOC]
    # In eager mode calls the test set-up twice, drawing a Python random integer, a NumPy random
    # value and a TensorFlow normal sample after each. Expects both sets of draws to be equal,
    # showing that set-up reseeds all three random sources. Side effect: reruns set-up.
    # Call setUp again for WithCApi case (since it makes a new default graph
    # after setup).
    # TODO(skyewm): remove this when C API is permanently enabled.
    with context.eager_mode():
      self.setUp()
      a = random.randint(1, 1000)
      a_np_rand = np.random.rand(1)
      a_rand = random_ops.random_normal([1])
      # ensure that randomness in multiple testCases is deterministic.
      self.setUp()
      b = random.randint(1, 1000)
      b_np_rand = np.random.rand(1)
      b_rand = random_ops.random_normal([1])
      self.assertEqual(a, b)
      self.assertEqual(a_np_rand, b_np_rand)
      self.assertAllEqual(a_rand, b_rand)

  def testIndexedSlices(self):
    # [FUNCTION DOC]
    # In eager mode evaluates an indexed-slices object built from two constants, checking that
    # evaluate accepts that structure without raising. No explicit assertion on the result.
    with context.eager_mode():
      self.evaluate(
          indexed_slices.IndexedSlices(
              constant_op.constant(1.0), constant_op.constant(0.0)))

  @test_util.run_in_graph_and_eager_modes
  def test_callable_evaluate(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. In eager mode, passes a callable that creates a resource
    # variable and adds one to it to the evaluate helper, expecting the result to equal two.
    def model():
      # [FUNCTION DOC]
      # HELPER: callable passed to evaluate. Takes no input; creates a resource variable named
      # same_name with initial value one and returns that variable plus one.
      return resource_variable_ops.ResourceVariable(
          name="same_name",
          initial_value=1) + 1
    with context.eager_mode():
      self.assertEqual(2, self.evaluate(model))

  @test_util.run_in_graph_and_eager_modes
  def test_nested_tensors_evaluate(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. Evaluates a nested dictionary of constant tensors and
    # expects a dictionary of the same shape holding plain Python/NumPy values.
    expected = {"a": 1, "b": 2, "nested": {"d": 3, "e": 4}}
    nested = {"a": constant_op.constant(1),
              "b": constant_op.constant(2),
              "nested": {"d": constant_op.constant(3),
                         "e": constant_op.constant(4)}}

    self.assertEqual(expected, self.evaluate(nested))

  @test_util.run_in_graph_and_eager_modes
  def test_custom_dataclass_evaluate(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. Evaluates masked-tensor fixtures: the result keeps the mask
    # and holds the evaluated value; two evaluations of equal inputs compare equal and one with a
    # different value compares not equal.
    mt = MaskedTensor(mask=True, value=constant_op.constant([1]))
    mt_val = self.evaluate(mt)
    self.assertEqual(mt_val.mask, True)
    self.assertAllEqual(mt_val.value, [1])

    mt2 = MaskedTensor(mask=True, value=constant_op.constant([1]))
    mt2_val = self.evaluate(mt2)
    self.assertEqual(mt_val, mt2_val)

    mt3 = MaskedTensor(mask=True, value=constant_op.constant([2]))
    mt3_val = self.evaluate(mt3)
    self.assertNotEqual(mt_val, mt3_val)

  def test_run_in_graph_and_eager_modes(self):
    # [FUNCTION DOC]
    # Applies the graph-and-eager decorator both directly (without parentheses) and as a call
    # with parentheses to a recording function, runs each, and expects four records in total:
    # each variant executed once in graph mode and once in eager mode.
    l = []
    def inc(self, with_brackets):
      # [FUNCTION DOC]
      # HELPER: test body recorded by the decorator test. Inputs: the test instance (ignored) and
      # a flag saying whether the decorator was used with parentheses. Appends a pair of that
      # flag as a label and the current execution mode to the enclosing list; returns nothing.
      del self  # self argument is required by run_in_graph_and_eager_modes.
      mode = "eager" if context.executing_eagerly() else "graph"
      with_brackets = "with_brackets" if with_brackets else "without_brackets"
      l.append((with_brackets, mode))

    f = test_util.run_in_graph_and_eager_modes(inc)
    f(self, with_brackets=False)
    f = test_util.run_in_graph_and_eager_modes()(inc)  # pylint: disable=assignment-from-no-return
    f(self, with_brackets=True)

    self.assertEqual(len(l), 4)
    self.assertEqual(set(l), {
        ("with_brackets", "graph"),
        ("with_brackets", "eager"),
        ("without_brackets", "graph"),
        ("without_brackets", "eager"),
    })

  def test_get_node_def_from_graph(self):
    # [FUNCTION DOC]
    # Builds a graph definition with a single node named foo. Expects the lookup helper to return
    # that very node for the name foo and nothing for an unknown name.
    graph_def = graph_pb2.GraphDef()
    node_foo = graph_def.node.add()
    node_foo.name = "foo"
    self.assertIs(test_util.get_node_def_from_graph("foo", graph_def), node_foo)
    self.assertIsNone(test_util.get_node_def_from_graph("bar", graph_def))

  def test_run_in_eager_and_graph_modes_test_class(self):
    # [FUNCTION DOC]
    # Expects applying the graph-and-eager decorator to a class (not a method) to raise a value
    # error saying only test methods are supported.
    msg = "`run_in_graph_and_eager_modes` only supports test methods.*"
    with self.assertRaisesRegex(ValueError, msg):

      @test_util.run_in_graph_and_eager_modes()
      class Foo(object):
        # [CLASS DOC]
        # HELPER: throwaway empty class used only as an invalid decoration target; the decorator
        # is expected to reject it with a value error, and it is deleted right after.
        pass
      del Foo  # Make pylint unused happy.

  def test_run_in_eager_and_graph_modes_skip_graph_runs_eager(self):
    # [FUNCTION DOC]
    # Decorates a body that skips itself in graph mode and records the mode otherwise. Expects
    # the graph run to be skipped without failing and only the eager run to be recorded.
    modes = []
    def _test(self):
      # [FUNCTION DOC]
      # HELPER: test body. Input: the test instance. Skips the test when not executing eagerly;
      # otherwise appends the eager mode name to the enclosing list. Returns nothing.
      if not context.executing_eagerly():
        self.skipTest("Skipping in graph mode")
      modes.append("eager" if context.executing_eagerly() else "graph")
    test_util.run_in_graph_and_eager_modes(_test)(self)
    self.assertEqual(modes, ["eager"])

  def test_run_in_eager_and_graph_modes_skip_eager_runs_graph(self):
    # [FUNCTION DOC]
    # Decorates a body that skips itself in eager mode and records the mode otherwise. Expects
    # the eager run to be skipped without failing and only the graph run to be recorded.
    modes = []
    def _test(self):
      # [FUNCTION DOC]
      # HELPER: test body. Input: the test instance. Skips the test when executing eagerly;
      # otherwise appends the graph mode name to the enclosing list. Returns nothing.
      if context.executing_eagerly():
        self.skipTest("Skipping in eager mode")
      modes.append("eager" if context.executing_eagerly() else "graph")
    test_util.run_in_graph_and_eager_modes(_test)(self)
    self.assertEqual(modes, ["graph"])

  def test_run_in_graph_and_eager_modes_setup_in_same_mode(self):
    # [FUNCTION DOC]
    # Defines an example test case whose set-up and body record the current execution mode, runs
    # set-up and then the decorated body, and expects the body to run first in graph mode and
    # then in eager mode, with set-up re-run in eager mode right before the eager run so that
    # set-up and body share the same mode.
    modes = []
    mode_name = lambda: "eager" if context.executing_eagerly() else "graph"

    class ExampleTest(test_util.TensorFlowTestCase):
      # [CLASS DOC]
      # HELPER: example test case that records, in the enclosing list, the execution mode seen
      # during set-up and during its single decorated test body.

      def runTest(self):
        # [FUNCTION DOC]
        # HELPER: required entry point so the case can be instantiated without a method name;
        # does nothing.
        pass

      def setUp(self):
        # [FUNCTION DOC]
        # HELPER: records the current execution mode with a set-up prefix in the enclosing list.
        modes.append("setup_" + mode_name())

      @test_util.run_in_graph_and_eager_modes
      def testBody(self):
        # [FUNCTION DOC]
        # HELPER: records the current execution mode with a run prefix in the enclosing list.
        modes.append("run_" + mode_name())

    e = ExampleTest()
    e.setUp()
    e.testBody()

    self.assertEqual(modes[1:2], ["run_graph"])
    self.assertEqual(modes[2:], ["setup_eager", "run_eager"])

  @parameterized.named_parameters(dict(testcase_name="argument",
                                       arg=True))
  @test_util.run_in_graph_and_eager_modes
  def test_run_in_graph_and_eager_works_with_parameterized_keyword(self, arg):
    # [FUNCTION DOC]
    # Checks the graph-and-eager decorator combined with named parameterization using a keyword
    # argument. Input: arg, expected to be true; the test asserts that.
    self.assertEqual(arg, True)

  @combinations.generate(combinations.combine(arg=True))
  @test_util.run_in_graph_and_eager_modes
  def test_run_in_graph_and_eager_works_with_combinations(self, arg):
    # [FUNCTION DOC]
    # Checks the graph-and-eager decorator combined with the combinations generator. Input: arg,
    # expected to be true; the test asserts that.
    self.assertEqual(arg, True)

  def test_build_as_function_and_v1_graph(self):
    # [FUNCTION DOC]
    # Defines an inner test case with one test decorated to be built both as a v1 graph and as a
    # function. Runs the generated graph variant and function variant and expects the body to have
    # executed exactly once outside a function and exactly once inside a function.

    class GraphModeAndFunctionTest(parameterized.TestCase):
      # [CLASS DOC]
      # HELPER: inner test case holding two flags recording whether the body ran in plain graph
      # mode and inside a function. Its decorated test updates the flags; the outer test checks them.

      def __init__(inner_self):  # pylint: disable=no-self-argument
        # [FUNCTION DOC]
        # HELPER: constructor taking no test name; initializes both flags to false.
        super(GraphModeAndFunctionTest, inner_self).__init__()
        inner_self.graph_mode_tested = False
        inner_self.inside_function_tested = False

      def runTest(self):
        # [FUNCTION DOC]
        # HELPER: required entry point; does nothing.
        del self

      @test_util.build_as_function_and_v1_graph
      def test_modes(inner_self):  # pylint: disable=no-self-argument
        # [FUNCTION DOC]
        # HELPER: body generated in two variants. If inside a function, asserts the function flag
        # was not yet set and sets it; otherwise does the same with the graph-mode flag.
        if ops.inside_function():
          self.assertFalse(inner_self.inside_function_tested)
          inner_self.inside_function_tested = True
        else:
          self.assertFalse(inner_self.graph_mode_tested)
          inner_self.graph_mode_tested = True

    test_object = GraphModeAndFunctionTest()
    test_object.test_modes_v1_graph()
    test_object.test_modes_function()
    self.assertTrue(test_object.graph_mode_tested)
    self.assertTrue(test_object.inside_function_tested)

  @test_util.run_in_graph_and_eager_modes
  def test_consistent_random_seed_in_assert_all_equal(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. Sets a global seed and creates a seeded random shuffle, then
    # expects exact equality of the tensor with itself to pass, which would fail if the two sides
    # were evaluated separately with different random results.
    random_seed.set_seed(1066)
    index = random_ops.random_shuffle([0, 1, 2, 3, 4], seed=2021)
    # This failed when `a` and `b` were evaluated in separate sessions.
    self.assertAllEqual(index, index)

  def test_with_forward_compatibility_horizons(self):
    # [FUNCTION DOC]
    # Applies the forward-compatibility horizons decorator with no horizon and a horizon past a
    # 2050 cutoff to a test that calls a function checking forward compatibility. Expects both the
    # present-behavior and future-behavior code paths to have been taken.

    tested_codepaths = set()
    def some_function_with_forward_compat_behavior():
      # [FUNCTION DOC]
      # HELPER: records "future" in the enclosing set if forward compatibility holds for the
      # date January 1 2050, otherwise records "present". Takes no input, returns nothing.
      if compat.forward_compatible(2050, 1, 1):
        tested_codepaths.add("future")
      else:
        tested_codepaths.add("present")

    @test_util.with_forward_compatibility_horizons(None, [2051, 1, 1])
    def some_test(self):
      # [FUNCTION DOC]
      # HELPER: decorated body. Input: test instance (unused, may be nothing). Calls the
      # forward-compat function once per horizon. Returns nothing.
      del self  # unused
      some_function_with_forward_compat_behavior()

    some_test(None)
    self.assertEqual(tested_codepaths, set(["present", "future"]))


class SkipTestTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the skip-if-error context manager. Set-up and tear-down both run a shared check
  # of the manager's behavior, so every test also verifies it around the test body.

  def _verify_test_in_set_up_or_tear_down(self):
    # [FUNCTION DOC]
    # HELPER: shared check called from set-up and tear-down. Expects a matching value error (one
    # of a list of messages) to turn into a skip, and a non-matching message to propagate as the
    # original value error rather than skip; raises a runtime error if a skip happens wrongly.
    with self.assertRaises(unittest.SkipTest):
      with test_util.skip_if_error(self, ValueError,
                                   ["foo bar", "test message"]):
        raise ValueError("test message")
    try:
      with self.assertRaisesRegex(ValueError, "foo bar"):
        with test_util.skip_if_error(self, ValueError, "test message"):
          raise ValueError("foo bar")
    except unittest.SkipTest:
      raise RuntimeError("Test is not supposed to skip.")

  def setUp(self):
    # [FUNCTION DOC]
    # Runs the parent set-up and then the shared skip-if-error check.
    super(SkipTestTest, self).setUp()
    self._verify_test_in_set_up_or_tear_down()

  def tearDown(self):
    # [FUNCTION DOC]
    # Runs the parent tear-down and then the shared skip-if-error check.
    super(SkipTestTest, self).tearDown()
    self._verify_test_in_set_up_or_tear_down()

  def test_skip_if_error_should_skip(self):
    # [FUNCTION DOC]
    # Raising a value error whose text contains the expected message inside the skip-if-error
    # manager must raise a skip exception.
    with self.assertRaises(unittest.SkipTest):
      with test_util.skip_if_error(self, ValueError, "test message"):
        raise ValueError("test message")

  def test_skip_if_error_should_skip_with_list(self):
    # [FUNCTION DOC]
    # Same as the basic skip case but the expected messages are given as a list; a value error
    # matching one of them must raise a skip exception.
    with self.assertRaises(unittest.SkipTest):
      with test_util.skip_if_error(self, ValueError,
                                   ["foo bar", "test message"]):
        raise ValueError("test message")

  def test_skip_if_error_should_skip_without_expected_message(self):
    # [FUNCTION DOC]
    # When no expected message is given, any value error with any text inside the manager must
    # raise a skip exception.
    with self.assertRaises(unittest.SkipTest):
      with test_util.skip_if_error(self, ValueError):
        raise ValueError("test message")

  def test_skip_if_error_should_skip_without_error_message(self):
    # [FUNCTION DOC]
    # With no expected message, a value error that has no message text must still raise a skip
    # exception.
    with self.assertRaises(unittest.SkipTest):
      with test_util.skip_if_error(self, ValueError):
        raise ValueError()

  def test_skip_if_error_should_raise_message_mismatch(self):
    # [FUNCTION DOC]
    # A value error whose text does not contain the expected message must propagate as the
    # original value error and not become a skip; a raised skip is converted into a runtime error.
    try:
      with self.assertRaisesRegex(ValueError, "foo bar"):
        with test_util.skip_if_error(self, ValueError, "test message"):
          raise ValueError("foo bar")
    except unittest.SkipTest:
      raise RuntimeError("Test is not supposed to skip.")

  def test_skip_if_error_should_raise_no_message(self):
    # [FUNCTION DOC]
    # A value error with no text, when a specific message is expected, must propagate as the
    # original value error and not become a skip; a raised skip is converted into a runtime error.
    try:
      with self.assertRaisesRegex(ValueError, ""):
        with test_util.skip_if_error(self, ValueError, "test message"):
          raise ValueError()
    except unittest.SkipTest:
      raise RuntimeError("Test is not supposed to skip.")


# Its own test case to reproduce variable sharing issues which only pop up when
# setUp() is overridden and super() is not called.
class GraphAndEagerNoVariableSharing(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case reproducing variable-sharing problems that only appear when set-up is overridden
  # without calling the parent set-up, checking that graph-and-eager runs still do not share
  # variables.

  def setUp(self):
    # [FUNCTION DOC]
    # Deliberately overrides set-up as a no-op without calling the parent implementation, to
    # reproduce the scenario under test.
    pass  # Intentionally does not call TensorFlowTestCase's super()

  @test_util.run_in_graph_and_eager_modes
  def test_no_variable_sharing(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. Creates a non-trainable resource variable with a fixed name
    # and expects this to work in both runs, meaning the variable store is not shared across runs.
    variable_scope.get_variable(
        name="step_size",
        initializer=np.array(1e-5, np.float32),
        use_resource=True,
        trainable=False)


class GarbageCollectionTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the leak-detection decorators: one that fails if new tensors stay alive after a
  # test, and one that fails if new Python objects accumulate while running eagerly.


  @test_util.run_in_graph_and_eager_modes
  def test_no_leaked_tensor_decorator(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. Defines a helper class with one method that keeps a tensor
    # alive on the instance and one that does not, both wrapped by the no-new-tensors decorator.
    # Expects the leaking method to fail with a tensors-not-deallocated assertion error and the
    # other to pass.

    class LeakedTensorTest(object):
      # [CLASS DOC]
      # HELPER: plain helper class with a leaking and a non-leaking decorated method.

      def __init__(inner_self):  # pylint: disable=no-self-argument
        # [FUNCTION DOC]
        # HELPER: constructor; borrows the outer test's equality assertion so the decorator can
        # call it on this instance.
        inner_self.assertEqual = self.assertEqual  # pylint: disable=invalid-name

      @test_util.assert_no_new_tensors
      def test_has_leak(self):
        # [FUNCTION DOC]
        # HELPER: stores a newly created constant tensor on the instance, causing a leak.
        self.a = constant_op.constant([3.], name="leak")

      @test_util.assert_no_new_tensors
      def test_has_no_leak(self):
        # [FUNCTION DOC]
        # HELPER: creates a constant tensor without keeping a reference, so nothing leaks.
        constant_op.constant([3.], name="no-leak")

    with self.assertRaisesRegex(AssertionError, "Tensors not deallocated"):
      LeakedTensorTest().test_has_leak()

    LeakedTensorTest().test_has_no_leak()

  def test_no_new_objects_decorator(self):
    # [FUNCTION DOC]
    # Defines an inner test case with a method that accumulates new Python objects (marked as an
    # expected failure) and one that does not, both wrapped by the no-new-Python-objects eager
    # decorator. Runs each and expects both runs to be reported successful (the first because its
    # failure was expected).

    class LeakedObjectTest(unittest.TestCase):
      # [CLASS DOC]
      # HELPER: inner test case with a list that its leaking test appends to.

      def __init__(self, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: constructor forwarding all arguments to the parent and creating an empty list
        # used to accumulate objects.
        super(LeakedObjectTest, self).__init__(*args, **kwargs)
        self.accumulation = []

      @unittest.expectedFailure
      @test_util.assert_no_new_pyobjects_executing_eagerly()
      def test_has_leak(self):
        # [FUNCTION DOC]
        # HELPER: appends a new list to the accumulating list, creating persistent new objects, so
        # the decorator should fail (the failure is expected).
        self.accumulation.append([1.])

      @test_util.assert_no_new_pyobjects_executing_eagerly()
      def test_has_no_leak(self):
        # [FUNCTION DOC]
        # HELPER: assigns a list to a single attribute (replaced, not accumulated), so no net new
        # objects are expected to persist.
        self.not_accumulating = [1.]

    self.assertTrue(LeakedObjectTest("test_has_leak").run().wasSuccessful())
    self.assertTrue(LeakedObjectTest("test_has_no_leak").run().wasSuccessful())


class RunFunctionsEagerlyInV2Test(test_util.TensorFlowTestCase,
                                  parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the context manager that forces tf.function-wrapped functions to run eagerly
  # (or not), parameterized over both settings.
  @parameterized.named_parameters(
      [("_RunEagerly", True), ("_RunGraph", False)])
  def test_run_functions_eagerly(self, run_eagerly):  # pylint: disable=g-wrong-blank-lines
    # [FUNCTION DOC]
    # KNOWN ISSUE: the assertions pass a generator expression to the true-assertion; a generator
    # object is always truthy and its loop variable is never evaluated, so these checks can never
    # fail and verify nothing about the types of the collected values.
    # Input: run_eagerly, whether to force eager execution. Defines a compiled function that adds
    # two five times and records each intermediate value in a list, runs it inside the
    # run-functions-eagerly context, and (intended) checks the recorded values are eager tensors
    # when run eagerly and graph tensors otherwise.
    results = []

    @def_function.function
    def add_two(x):
      # [FUNCTION DOC]
      # HELPER: compiled function. Input: a number or tensor x. Five times adds two to x and appends
      # the running value to the enclosing results list (a side effect visible only when run
      # eagerly or during tracing); returns the final value.
      for _ in range(5):
        x += 2
        results.append(x)
      return x

    with test_util.run_functions_eagerly(run_eagerly):
      add_two(constant_op.constant(2.))
      if context.executing_eagerly():
        if run_eagerly:
          self.assertTrue(isinstance(t, ops.EagerTensor) for t in results)
        else:
          self.assertTrue(isinstance(t, tensor.Tensor) for t in results)
      else:
        self.assertTrue(isinstance(t, tensor.Tensor) for t in results)


class SyncDevicesTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the device synchronization helper using a sleep op to observe asynchronous versus
  # synchronous execution on CPU and GPU, and its error in graph mode. Tear-down restores
  # synchronous execution.

  def tearDown(self):
    # [FUNCTION DOC]
    # Runs the parent tear-down and then turns synchronous execution back on, undoing any
    # asynchronous setting made by a test (global runtime configuration side effect).
    super().tearDown()
    config.set_synchronous_execution(True)

  def test_sync_device_cpu(self):
    # [FUNCTION DOC]
    # In eager mode on the CPU: with asynchronous execution a one-second sleep op returns in under
    # a second and the sync call then waits until over a second has passed; with synchronous
    # execution the sleep op takes at least a second and a following sync call returns promptly.
    with context.eager_mode(), ops.device("/CPU:0"):
      config.set_synchronous_execution(False)
      start = time.time()
      test_ops.sleep_op(sleep_seconds=1)
      self.assertLess(time.time() - start, 1.0)
      test_util.sync_devices()
      self.assertGreater(time.time() - start, 1.0)

      config.set_synchronous_execution(True)
      start = time.time()
      test_ops.sleep_op(sleep_seconds=1)
      self.assertGreaterEqual(time.time() - start, 1.0)
      start = time.time()
      test_util.sync_devices()
      self.assertLess(time.time() - start, 1.0)

  def test_sync_device_gpu(self):
    # [FUNCTION DOC]
    # Skipped unless a GPU with compute capability 7.0 is available (needed by the sleep op). In
    # eager mode on the GPU with asynchronous execution, the sleep op returns quickly and the sync
    # call waits past a second. Then with synchronous execution the sleep op is still expected to
    # return in under a second (GPU kernel launch stays asynchronous) and the sync call then waits
    # at least a second.
    if not test_util.is_gpu_available(min_cuda_compute_capability=(7, 0)):
      # sleep_op requires compute capability 7.0
      self.skipTest("Requires GPU with compute capability 7.0")

    with context.eager_mode(), ops.device("/GPU:0"):
      config.set_synchronous_execution(False)
      start = time.time()
      test_ops.sleep_op(sleep_seconds=1)
      self.assertLess(time.time() - start, 1.0)
      test_util.sync_devices()
      self.assertGreater(time.time() - start, 1.0)

      config.set_synchronous_execution(True)
      start = time.time()
      test_ops.sleep_op(sleep_seconds=1)
      self.assertLess(time.time() - start, 1.0)
      start = time.time()
      test_util.sync_devices()
      self.assertGreaterEqual(time.time() - start, 1.0)

  def test_sync_devices_graph_mode_error(self):
    # [FUNCTION DOC]
    # Calling the device sync helper in graph mode must raise a runtime error saying it must only
    # be called in eager mode.
    with context.graph_mode():
      with self.assertRaisesRegex(
          RuntimeError, r"sync_devices\(\) must only be called in Eager mode"
      ):
        test_util.sync_devices()


if __name__ == "__main__":
  googletest.main()
