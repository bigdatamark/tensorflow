# [MODULE DOC]
# Unit tests for the dispatch module in the util package. Covers the legacy fallback dispatch
# decorator (overriding generated C++-backed ops and Python ops, signature compatibility rules,
# error cases), global dispatchers (using a tracer object that records ops applied to it),
# interaction with the deprecation decorator, and the type-based dispatch system (registering
# targets by signature, lists, unions, optionals, annotations and positional signatures, error
# reporting, name-scope handling, iterable parameters, registry queries) plus the unary and
# binary elementwise dispatch decorators and docstring API-list updates.
# Defines fake ops, a fake composite tensor class, a tracer class and two extension types as
# shared fixtures. Import-time side effects: exports two test ops under test names, overloads
# the tracer's operators. Runs the tests under the standard test main when executed directly.
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
"""Tests for operator dispatch."""

import collections
import typing
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import extension_type
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import bitwise_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.linalg import linear_operator_diag
from tensorflow.python.ops.proto_ops import decode_proto
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging
from tensorflow.python.types import core as core_tf_types
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import get_canonical_name_for_symbol
from tensorflow.python.util.tf_export import tf_export


class CustomTensor(object):
  # [CLASS DOC]
  # HELPER: shared fixture for the fallback dispatch tests.
  # Minimal stand-in for a composite tensor holding a converted tensor and a numeric score.
  # Used as the dispatch type in fallback-dispatch tests.
  """A fake composite tensor class, for testing type-based dispatching."""

  def __init__(self, tensor, score):
    # [FUNCTION DOC]
    # Constructor. Converts the given tensor-like value to a tensor and stores it, and stores the
    # score (a number). Returns nothing.
    self.tensor = ops.convert_to_tensor(tensor)
    self.score = score


@tf_export("test_op")
@dispatch.add_dispatch_support
def test_op(x, y, z):
  # [FUNCTION DOC]
  # HELPER: fake dispatch-enabled Python op used as an override target by the tests.
  # Inputs: three numeric tensors or values. Returns the first plus twice the second plus three
  # times the third. No side effects.
  """A fake op for testing dispatch of Python ops."""
  return x + (2 * y) + (3 * z)


@tf_export("test_op_with_optional")
@dispatch.add_dispatch_support
def test_op_with_optional(x, y, z, optional=None):
  # [FUNCTION DOC]
  # HELPER: fake dispatch-enabled Python op with an extra optional parameter, used to test
  # signature compatibility. Inputs: three numeric values and an optional value that is ignored.
  # Returns the first plus twice the second plus three times the third.
  """A fake op for testing dispatch of Python ops."""
  del optional
  return x + (2 * y) + (3 * z)


@tf_export("test_op_with_kwonly")
@dispatch.add_dispatch_support
def test_op_with_kwonly(*, x, y, z, optional=None):
  # [FUNCTION DOC]
  # HELPER: fake dispatch-enabled Python op whose parameters are keyword-only, used to test
  # signature compatibility. Inputs: x, y, z numeric values and an ignored optional value, all
  # passed by keyword. Returns the first plus twice the second plus three times the third.
  """A fake op for testing dispatch of Python ops."""
  del optional
  return x + (2 * y) + (3 * z)


class TensorTracer(object):
  # [CLASS DOC]
  # HELPER: shared fixture for the global dispatcher tests.
  # Symbolic tensor-like object recording an op name plus the arguments it was called with, so
  # the global dispatcher can build a printable trace of applied ops. State: name, args, kwargs,
  # plus a fixed shape and float32 dtype. Tensor operators are overloaded onto it at import time.
  """An object used to trace TensorFlow graphs.

  This is an example class that is used to test global op dispatchers.  The
  global op dispatcher for TensorTracers is defined below.
  """

  def __init__(self, name, args=None, kwargs=None):
    # [FUNCTION DOC]
    # Constructor. Stores the op name (string) and optional arguments (sequence) and keyword
    # arguments (mapping) that produced this trace, and sets a fixed 4 by 4 shape (taken from a
    # tensor of ones created here) and float32 data type. Returns nothing.
    self.name = name
    self.args = args
    self.kwargs = kwargs
    self.shape = array_ops.ones(shape=(4, 4)).shape
    self.dtype = dtypes.float32

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns the plain name when there are no recorded arguments; otherwise a call-like string
    # with the name, the string forms of the positional arguments, then the keyword arguments as
    # name equals value pairs sorted alphabetically, all comma separated.
    if self.args is None and self.kwargs is None:
      return self.name
    else:
      args = [str(x) for x in self.args]
      args += sorted(
          ["{}={}".format(name, x) for (name, x) in self.kwargs.items()])
      return "{}({})".format(self.name, ", ".join(args))

  @property
  def is_tensor_like(self):
    # [FUNCTION DOC]
    # Property marking the object as tensor-like. Always returns true.
    return True

  @classmethod
  def _overload_all_operators(cls):  # pylint: disable=invalid-name
    # [FUNCTION DOC]
    # HELPER: one-time setup for the tracer fixture.
    # Class method. Overloads every operator the tensor class lists as overloadable onto this
    # class. Takes no inputs, returns nothing; mutates the class.
    """Register overloads for all operators."""
    for operator in tensor_lib.Tensor.OVERLOADABLE_OPERATORS:
      cls._overload_operator(operator)

  @classmethod
  def _overload_operator(cls, operator):  # pylint: disable=invalid-name
    # [FUNCTION DOC]
    # HELPER: used by the all-operators overload step.
    # Class method. Input: operator, the name of an operator method on the tensor class. Looks up
    # that method, unwraps it to the underlying plain function if needed, and sets it on this
    # class under the same name. Returns nothing; mutates the class.
    """Overload an operator with the same overloading as `tensor_lib.Tensor`."""
    tensor_oper = getattr(tensor_lib.Tensor, operator)

    # Compatibility with Python 2:
    # Python 2 unbound methods have type checks for the first arg,
    # so we need to extract the underlying function
    tensor_oper = getattr(tensor_oper, "__func__", tensor_oper)
    setattr(cls, operator, tensor_oper)


TensorTracer._overload_all_operators()  # pylint: disable=protected-access


class TensorTracerOpDispatcher(dispatch.GlobalOpDispatcher):
  # [CLASS DOC]
  # HELPER: shared fixture for the global dispatcher tests.
  # Global dispatcher that handles any op call where at least one argument contains a tracer,
  # returning a new tracer named after the op's canonical public name that records the call.
  """Global op dispatcher for TensorTracer."""

  def _flatten_with_slice_flattening(self, x):
    # [FUNCTION DOC]
    # HELPER: lets the tracer check see inside slices.
    # Input: any nested structure. Returns a flat list of its leaves, where each slice object is
    # replaced by its start, stop and step values.
    flat = []
    for val in nest.flatten(x):
      if isinstance(val, slice):
        flat.extend((val.start, val.stop, val.step))
      else:
        flat.append(val)
    return flat

  def handle(self, op, args, kwargs):
    # [FUNCTION DOC]
    # Global handler. Inputs: op, positional arguments, keyword arguments. If no argument (or
    # keyword value) contains a tracer, returns the not-supported sentinel. Otherwise returns a
    # new tracer named with the op's canonical public name and recording the arguments.
    # Dispatcher only applies if at least one arg is a TensorTracer.
    if not (any(self.is_tensor_tracer_arg(x) for x in args) or
            any(self.is_tensor_tracer_arg(x) for x in kwargs.values())):
      return self.NOT_SUPPORTED

    symbol_name = get_canonical_name_for_symbol(op)
    return TensorTracer(symbol_name, args, kwargs)

  def is_tensor_tracer_arg(self, value):
    # [FUNCTION DOC]
    # HELPER: predicate for the global handler. Input: any value or nested structure. Returns true
    # if any leaf (with slices expanded) is a tracer object.
    return any(
        isinstance(x, TensorTracer)
        for x in self._flatten_with_slice_flattening(value))


@test_util.run_all_in_graph_and_eager_modes
class DispatchTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for the fallback (legacy) dispatch system: per-type dispatch decorator on generated
  # and Python ops, signature compatibility errors, interaction with the deprecation decorator,
  # and global dispatchers using the tracer fixture. Run in both graph and eager modes. Tests
  # restore any modified dispatcher lists afterwards.

  def testAddDispatchForTypes_With_CppOp(self):
    # [FUNCTION DOC]
    # Scenario: override the generated arctangent-of-two-arguments op for the fake composite
    # type so it behaves like addition. Asserts the dispatcher list grew by one, that fake-type
    # inputs yield summed tensors and averaged scores, that ordinary inputs keep normal behavior,
    # and that unsupported argument types still raise a type or value error. Restores the list.
    original_handlers = gen_math_ops.atan2._tf_fallback_dispatchers[:]

    # Override the behavior of gen_math_ops.atan2 and make it look like add.
    @dispatch.dispatch_for_types(gen_math_ops.atan2, CustomTensor)
    def custom_atan2(y, x, name=None):  # pylint: disable=unused-variable
      # [FUNCTION DOC]
      # Override used by the test: adds the two tensors of the fake type and returns a new fake
      # object whose score is the average of the input scores. The name is passed to the add.
      return CustomTensor(
          gen_math_ops.add(y.tensor, x.tensor, name), (x.score + y.score) / 2.0)

    self.assertEqual(
        len(math_ops.atan2._tf_fallback_dispatchers),
        len(original_handlers) + 1)

    # Test that we see the overridden behavior when using CustomTensors.
    x = CustomTensor([1., 2., 3.], 2.0)
    y = CustomTensor([7., 8., 2.], 0.0)
    x_plus_y = gen_math_ops.atan2(y, x)
    self.assertAllEqual(self.evaluate(x_plus_y.tensor), [8, 10, 5])
    self.assertNear(x_plus_y.score, 1.0, 0.001)

    # Test that we still get the right behavior when using normal Tensors.
    a = [1., 2., 3.]
    b = [7., 8., 2.]
    a_plus_b = gen_math_ops.atan2(a, b)
    self.assertAllClose(a_plus_b, [0.14189707, 0.24497867, 0.98279375])

    # Test that we still get a TypeError or ValueError if we pass some
    # type that's not supported by any dispatcher.
    with self.assertRaises((TypeError, ValueError)):
      gen_math_ops.atan2(a, None)

    # Clean up
    gen_math_ops.atan2._tf_fallback_dispatchers = original_handlers

  def testAddDispatchForTypes_With_PythonOp(self):
    # [FUNCTION DOC]
    # Scenario: register an override of the fake Python op for the fake composite type. Asserts
    # the decorator returns the override unchanged, and that calling the op with fake objects
    # returns the combined tensor (15, 21, 13) and the averaged score (0.4). Restores the list.
    original_handlers = test_op._tf_fallback_dispatchers[:]

    def override_for_test_op(x, y, z):  # pylint: disable=unused-variable
      # [FUNCTION DOC]
      # Override: applies the fake op to the tensors of three fake objects and returns a new fake
      # object with the averaged score.
      return CustomTensor(
          test_op(x.tensor, y.tensor, z.tensor),
          (x.score + y.score + z.score) / 3.0)

    override = dispatch.dispatch_for_types(test_op, CustomTensor)(
        override_for_test_op
    )

    self.assertIs(override, override_for_test_op)

    x = CustomTensor([1, 2, 3], 0.2)
    y = CustomTensor([7, 8, 2], 0.4)
    z = CustomTensor([0, 1, 2], 0.6)

    result = test_op(x, y, z)
    self.assertAllEqual(self.evaluate(result.tensor), [15, 21, 13])
    self.assertNear(result.score, 0.4, 0.001)

    # Clean up
    test_op._tf_fallback_dispatchers = original_handlers

  def testDispatchForTypes_MissingArgs(self):
    # [FUNCTION DOC]
    # Scenario: the override lacks the op's optional parameter. Asserts registration works (the
    # override is returned unchanged) and a call without the optional argument dispatches to the
    # override, giving the expected combined tensor and averaged score. Restores the list.
    original_handlers = test_op_with_optional._tf_fallback_dispatchers[:]

    def override_for_test_op(x, y, z):  # pylint: disable=unused-variable
      # [FUNCTION DOC]
      # Override without the optional parameter: combines the tensors of three fake objects with
      # the fake op and returns a fake object with the averaged score.
      return CustomTensor(
          test_op(x.tensor, y.tensor, z.tensor),
          (x.score + y.score + z.score) / 3.0,
      )

    override = dispatch.dispatch_for_types(test_op_with_optional, CustomTensor)(
        override_for_test_op
    )

    self.assertIs(override, override_for_test_op)

    x = CustomTensor([1, 2, 3], 0.2)
    y = CustomTensor([7, 8, 2], 0.4)
    z = CustomTensor([0, 1, 2], 0.6)

    result = test_op_with_optional(x, y, z)
    self.assertAllEqual(self.evaluate(result.tensor), [15, 21, 13])
    self.assertNear(result.score, 0.4, 0.001)

    # Clean up
    test_op_with_optional._tf_fallback_dispatchers = original_handlers

  def testDispatchForTypes_ProvidingMissingArgs(self):
    # [FUNCTION DOC]
    # Scenario: the override lacks the op's optional parameter but the caller passes a non-default
    # value for it. Asserts an assertion error saying the dispatched op was called with an
    # unsupported non-default argument. Restores the list.
    original_handlers = test_op_with_optional._tf_fallback_dispatchers[:]

    @dispatch.dispatch_for_types(test_op_with_optional, CustomTensor)
    def override_for_test_op(x, y, z):  # pylint: disable=unused-variable
      # [FUNCTION DOC]
      # Override without the optional parameter; never successfully reached in this test because
      # the wrapper rejects the extra argument first.
      return CustomTensor(
          test_op(x.tensor, y.tensor, z.tensor),
          (x.score + y.score + z.score) / 3.0,
      )

    x = CustomTensor([1, 2, 3], 0.2)
    y = CustomTensor([7, 8, 2], 0.4)
    z = CustomTensor([0, 1, 2], 0.6)

    with self.assertRaisesRegex(
        AssertionError,
        "Dispatched op is called with argument `optional` set to a non-default"
        " value, which is not supported by the decorated function",
    ):
      test_op_with_optional(x, y, z, optional=3)

    # Clean up
    test_op_with_optional._tf_fallback_dispatchers = original_handlers

  def testDispatchForTypes_NewArgs(self):
    # [FUNCTION DOC]
    # Scenario: the override has an extra optional parameter that the op lacks. Asserts calls
    # still dispatch and return the expected combined tensor and averaged score. Restores the
    # list.
    original_handlers = test_op_with_optional._tf_fallback_dispatchers[:]

    @dispatch.dispatch_for_types(test_op_with_optional, CustomTensor)
    def override_for_test_op(x, y, z, u=None):  # pylint: disable=unused-variable
      # [FUNCTION DOC]
      # Override with an extra ignored optional parameter: combines the tensors of three fake
      # objects and returns a fake object with the averaged score.
      del u
      return CustomTensor(
          test_op(x.tensor, y.tensor, z.tensor),
          (x.score + y.score + z.score) / 3.0,
      )

    x = CustomTensor([1, 2, 3], 0.2)
    y = CustomTensor([7, 8, 2], 0.4)
    z = CustomTensor([0, 1, 2], 0.6)

    result = test_op_with_optional(x, y, z)
    self.assertAllEqual(self.evaluate(result.tensor), [15, 21, 13])
    self.assertNear(result.score, 0.4, 0.001)

    # Clean up
    test_op_with_optional._tf_fallback_dispatchers = original_handlers

  def testDispatchForTypes_SignatureMismatchOrder(self):
    # [FUNCTION DOC]
    # Scenario: the override's required parameters are in a different order than the op's.
    # Asserts registration raises an assertion error saying the non-default arguments must be
    # identical.
    with self.assertRaisesRegex(
        AssertionError,
        "The decorated function's non-default arguments must be identical to"
        " that of the overridden op.",
    ):

      @dispatch.dispatch_for_types(test_op, CustomTensor)
      def override_for_test_op(x, z, y):  # pylint: disable=unused-variable
        # [FUNCTION DOC]
        # Override with swapped parameter order; registration is expected to be rejected, so it is
        # never called.
        return CustomTensor(
            test_op(x.tensor, y.tensor, z.tensor),
            (x.score + y.score + z.score) / 3.0,
        )

  def testDispatchForTypes_MissingKwOnly(self):
    # [FUNCTION DOC]
    # Scenario: the op takes keyword-only parameters but the override uses plain positional ones.
    # Asserts registration raises an assertion error saying the non-default arguments must be
    # identical.
    with self.assertRaisesRegex(
        AssertionError,
        "The decorated function's non-default arguments must be identical to"
        " that of the overridden op.",
    ):

      @dispatch.dispatch_for_types(test_op_with_kwonly, CustomTensor)
      def override_for_test_op(x, z, y):  # pylint: disable=unused-variable
        # [FUNCTION DOC]
        # Override with mismatched parameters; registration is expected to be rejected, so it is
        # never called.
        return CustomTensor(
            test_op(x.tensor, y.tensor, z.tensor),
            (x.score + y.score + z.score) / 3.0,
        )

  def testDispatchForTypes_SignatureMismatchNames(self):
    # [FUNCTION DOC]
    # Scenario: the override's required parameters have different names than the op's. Asserts
    # registration raises an assertion error saying the non-default arguments must be identical.
    with self.assertRaisesRegex(
        AssertionError,
        "The decorated function's non-default arguments must be identical to"
        " that of the overridden op.",
    ):
      @dispatch.dispatch_for_types(test_op, CustomTensor)
      def override_for_test_op(a, b, c):  # pylint: disable=unused-variable
        # [FUNCTION DOC]
        # Override with differently named parameters; registration is expected to be rejected, so
        # it is never called.
        return CustomTensor(
            test_op(a.tensor, b.tensor, c.tensor),
            (a.score + b.score + c.score) / 3.0)

  def testDispatchForTypes_OpDoesNotSupportDispatch(self):
    # [FUNCTION DOC]
    # Scenario: try to register an override on a plain function that has no dispatch support.
    # Asserts an assertion error saying dispatching is not enabled.

    def some_op(x, y):
      # [FUNCTION DOC]
      # Plain function without dispatch support; returns the sum of its two inputs.
      return x + y

    with self.assertRaisesRegex(AssertionError, "Dispatching not enabled for"):

      @dispatch.dispatch_for_types(some_op, CustomTensor)
      def override_for_some_op(x, y):  # pylint: disable=unused-variable
        # [FUNCTION DOC]
        # Override that would pick whichever fake object has a positive score; never called
        # because registration is rejected.
        return x if x.score > 0 else y

  @test.mock.patch.object(tf_logging, "warning", autospec=True)
  def testInteractionWithDeprecationWarning(self, mock_warning):
    # [FUNCTION DOC]
    # Scenario: a dispatch-enabled function is also wrapped by the deprecation decorator. Calls
    # it once with a mocked warning logger (injected by the patch decorator) and asserts the
    # logged message names the function and says it is deprecated and will be removed.

    @deprecation.deprecated(date=None, instructions="Instructions")
    @dispatch.add_dispatch_support
    def some_op(x):
      # [FUNCTION DOC]
      # Identity function used as the deprecated, dispatch-enabled op; returns its input.
      return x

    some_op(5)

    message = mock_warning.call_args[0][0] % mock_warning.call_args[0][1:]
    self.assertRegex(
        message, r".*some_op \(from __main__\) is deprecated and will be "
        "removed in a future version.*")

  def testGlobalDispatcher(self):
    # [FUNCTION DOC]
    # Scenario: register the tracer global dispatcher, then apply reduce-sum, add and abs ops to
    # tracer objects. Asserts the printed trace shows the nested canonical op names and
    # arguments, and that a protobuf-decoding op also produces a trace. Restores the global list.
    original_global_dispatchers = dispatch._GLOBAL_DISPATCHERS
    try:
      TensorTracerOpDispatcher().register()

      x = TensorTracer("x")
      y = TensorTracer("y")
      trace = math_ops.reduce_sum(math_ops.add(math_ops.abs(x), y), axis=3)
      self.assertEqual(
          str(trace), "math.reduce_sum(math.add(math.abs(x), y), axis=3)")

      proto_val = TensorTracer("proto")
      trace = decode_proto(proto_val, "message_type", ["field"], ["float32"])
      self.assertIn("io.decode_proto(bytes=proto,", str(trace))

    finally:
      # Clean up.
      dispatch._GLOBAL_DISPATCHERS = original_global_dispatchers

  def testGlobalDispatcherConvertToTensor(self):
    # [FUNCTION DOC]
    # Scenario: with the tracer global dispatcher registered, convert a tracer to a tensor and
    # wrap it in abs and add ops. Asserts the trace string shows the conversion call nested
    # inside the other ops. Restores the global list.
    original_global_dispatchers = dispatch._GLOBAL_DISPATCHERS
    try:
      TensorTracerOpDispatcher().register()

      x = TensorTracer("x")
      y = TensorTracer("y")
      trace = math_ops.add(
          math_ops.abs(tensor_conversion.convert_to_tensor_v2_with_dispatch(x)),
          y,
      )
      self.assertEqual(
          str(trace), "math.add(math.abs(convert_to_tensor(x)), y)")

    finally:
      # Clean up.
      dispatch._GLOBAL_DISPATCHERS = original_global_dispatchers

  def testGlobalDispatcherGetItem(self):
    # [FUNCTION DOC]
    # Scenario: index and slice tracers (and a real tensor indexed by a tracer) with the tracer
    # global dispatcher registered. Asserts the traces show the item-access op with the expected
    # index or slice arguments. Restores the global list.
    original_global_dispatchers = dispatch._GLOBAL_DISPATCHERS
    try:
      TensorTracerOpDispatcher().register()

      x = TensorTracer("x")
      trace = x[0]
      self.assertEqual(str(trace), "__operators__.getitem(x, 0)")

      x = TensorTracer("x")
      y = TensorTracer("y")
      trace = x[y]
      self.assertEqual(str(trace), "__operators__.getitem(x, y)")

      x = TensorTracer("x")
      y = TensorTracer("y")
      trace = x[:y]  # pylint: disable=invalid-slice-index
      self.assertEqual(
          str(trace), "__operators__.getitem(x, slice(None, y, None))")

      x = array_ops.ones(shape=(3, 3))
      y = TensorTracer("y")
      trace = x[y]
      self.assertEqual(str(trace), "__operators__.getitem(%s, y)" % x)

      trace = x[:y]  # pylint: disable=invalid-slice-index
      self.assertEqual(
          str(trace), "__operators__.getitem(%s, slice(None, y, None))" % x)

    finally:
      # Clean up.
      dispatch._GLOBAL_DISPATCHERS = original_global_dispatchers

  def testGlobalDispatcherLinearOperators(self):
    # [FUNCTION DOC]
    # Scenario: build a diagonal linear operator from a tracer and call eigenvalues, add-to-tensor
    # and non-singularity assertion. Asserts the traces contain the expected conversion, set-diag,
    # add, diagonal-part and less-than-assertion calls with their messages, showing the operator
    # code path goes through dispatchable ops. Restores the global list.
    original_global_dispatchers = dispatch._GLOBAL_DISPATCHERS
    try:
      TensorTracerOpDispatcher().register()

      x = TensorTracer("x")

      # To grab the eigenvalues the diag operator just calls convert_to_tensor
      # (twice) in this case.
      trace = linear_operator_diag.LinearOperatorDiag(x).eigvals()
      self.assertEqual(
          str(trace),
          "convert_to_tensor(convert_to_tensor(x, dtype=None, dtype_hint=None, "
          "name=diag))")

      # The diagonal tensor addition gets traced even though the linear_operator
      # API only uses dispatchable ops instead of directly exposing dispatching.
      trace = linear_operator_diag.LinearOperatorDiag(x).add_to_tensor(x)
      self.assertIn(
          "linalg.set_diag(convert_to_tensor(x, name=x), __operators__.add("
          "convert_to_tensor(x, dtype=None, dtype_hint=None, name=diag), "
          "linalg.diag_part(convert_to_tensor(x, name=x)), "
          "name=", str(trace))

      # The dispatch-supporting ops the non-singular check calls out to
      # get traced.
      trace = linear_operator_diag.LinearOperatorDiag(x).assert_non_singular()
      self.assertIn("debugging.assert_less", str(trace))
      self.assertIn(
          "message=Singular operator:  Diagonal contained zero values.",
          str(trace))

    finally:
      # Clean up.
      dispatch._GLOBAL_DISPATCHERS = original_global_dispatchers


class MaskedTensor(extension_type.ExtensionType):
  # [CLASS DOC]
  # HELPER: shared fixture for the type-based dispatch tests.
  # Extension type pairing a tensor of values with a mask tensor; the main dispatch type used as
  # an annotation and as test data. Fields: values, mask.
  """Simple ExtensionType for testing v2 dispatch."""
  values: tensor_lib.Tensor
  mask: tensor_lib.Tensor


class SillyTensor(extension_type.ExtensionType):
  # [CLASS DOC]
  # HELPER: shared fixture for the type-based dispatch tests.
  # Second extension type (a tensor plus a float silliness level), used to check that lookups by
  # class only return signatures mentioning the requested class. Fields: value, how_silly.
  """Simple ExtensionType for testing v2 dispatch."""
  value: tensor_lib.Tensor
  how_silly: float


@test_util.run_all_in_graph_and_eager_modes
class DispatchV2Test(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for the type-based (v2) dispatch system: registering dispatch targets for the add,
  # concat and other APIs by explicit signatures, annotations, lists, unions and optionals;
  # elementwise unary and binary handlers; error reporting; registry queries; and docstring
  # updates. Run in both graph and eager modes. Most tests unregister their targets in a cleanup
  # step so the global dispatch tables are left unchanged.

  def testDispatchForOneSignature(self):
    # [FUNCTION DOC]
    # Scenario: register a target for addition when both arguments are masked tensors. Asserts the
    # result's values are the sum and its mask is the logical and of the masks. Unregisters.

    @dispatch.dispatch_for_api(math_ops.add, {
        "x": MaskedTensor,
        "y": MaskedTensor
    })
    def masked_add(x, y, name=None):
      # [FUNCTION DOC]
      # Dispatch target: inside a name scope, adds the values and ands the masks of two masked
      # tensors, returning a new masked tensor.
      with ops.name_scope(name):
        return MaskedTensor(x.values + y.values, x.mask & y.mask)

    try:
      x = MaskedTensor([1, 2, 3, 4, 5], [1, 0, 1, 1, 1])
      y = MaskedTensor([1, 1, 1, 1, 1], [1, 1, 0, 1, 0])
      z = math_ops.add(x, y)
      self.assertAllEqual(z.values, x.values + y.values)
      self.assertAllEqual(z.mask, x.mask & y.mask)

    finally:
      # Clean up dispatch table.
      dispatch.unregister_dispatch_for(masked_add)

  def testDispatchSignatureWithUnspecifiedParameter(self):
    # [FUNCTION DOC]
    # Scenario: the signature constrains only the first argument. Asserts the target is invoked
    # whenever x is a masked tensor, whatever the type of y (tensor, list, number, nothing), with
    # the expected summed values. Unregisters.

    @dispatch.dispatch_for_api(math_ops.add, {"x": MaskedTensor})
    def masked_add(x, y):
      # [FUNCTION DOC]
      # Dispatch target: returns x when y is nothing; otherwise adds y (using its values and mask
      # if it is a masked tensor, else y itself with an all-true mask) to x.
      if y is None:
        return x
      y_values = y.values if isinstance(y, MaskedTensor) else y
      y_mask = y.mask if isinstance(y, MaskedTensor) else True
      return MaskedTensor(x.values + y_values, x.mask & y_mask)

    try:
      a = MaskedTensor([1, 2, 3, 4, 5], [1, 0, 1, 1, 1])
      b = constant_op.constant([10, 20, 30, 40, 50])
      c = [10, 20, 30, 40, 50]
      d = 50
      e = None
      # As long as `x` is a MaskedTensor, the dispatcher will be called
      # (regardless of the type for `y`):
      self.assertAllEqual(math_ops.add(a, b).values, [11, 22, 33, 44, 55])
      self.assertAllEqual(math_ops.add(a, c).values, [11, 22, 33, 44, 55])
      self.assertAllEqual(math_ops.add(a, d).values, [51, 52, 53, 54, 55])
      self.assertAllEqual(math_ops.add(a, e).values, [1, 2, 3, 4, 5])

    finally:
      # Clean up dispatch table.
      dispatch.unregister_dispatch_for(masked_add)

  def testDispatchForMultipleSignatures(self):
    # [FUNCTION DOC]
    # Scenario: register one target with two signatures (x masked, or y masked). Calls add with a
    # masked x and an ordinary tensor y; asserts values are summed and the mask is x's mask.
    # Unregisters.

    @dispatch.dispatch_for_api(math_ops.add, {"x": MaskedTensor},
                               {"y": MaskedTensor})
    def masked_add(x, y, name=None):
      # [FUNCTION DOC]
      # Dispatch target: unwraps either argument that is masked (an ordinary argument gets an
      # all-true mask), then adds the values and ands the masks into a new masked tensor.
      with ops.name_scope(name):
        x_values = x.values if isinstance(x, MaskedTensor) else x
        x_mask = x.mask if isinstance(x, MaskedTensor) else True
        y_values = y.values if isinstance(y, MaskedTensor) else y
        y_mask = y.mask if isinstance(y, MaskedTensor) else True
        return MaskedTensor(x_values + y_values, x_mask & y_mask)

    try:
      x = MaskedTensor([1, 2, 3, 4, 5], [1, 0, 1, 1, 1])
      y = constant_op.constant([10, 20, 30, 40, 50])
      z = math_ops.add(x, y)
      self.assertAllEqual(z.values, x.values + y)
      self.assertAllEqual(z.mask, x.mask)

    finally:
      # Clean up dispatch table.
      dispatch.unregister_dispatch_for(masked_add)

  def testDispatchForList(self):
    # [FUNCTION DOC]
    # Scenario: register a concat target for a list of masked tensors. Asserts concatenating two
    # masked tensors along axis zero concatenates both values and masks. Unregisters.

    @dispatch.dispatch_for_api(array_ops.concat,
                               {"values": typing.List[MaskedTensor]})
    def masked_concat(values, axis, name=None):
      # [FUNCTION DOC]
      # Dispatch target: inside a name scope, concatenates the values and the masks of the list
      # along the given axis into a new masked tensor.
      with ops.name_scope(name):
        return MaskedTensor(
            array_ops.concat([v.values for v in values], axis),
            array_ops.concat([v.mask for v in values], axis))

    try:
      x = MaskedTensor([1, 2, 3, 4, 5], [1, 0, 1, 1, 1])
      y = MaskedTensor([1, 1, 1], [1, 1, 0])
      z = array_ops.concat([x, y], axis=0)
      self.assertAllEqual(z.values, array_ops.concat([x.values, y.values], 0))
      self.assertAllEqual(z.mask, array_ops.concat([x.mask, y.mask], 0))

    finally:
      # Clean up dispatch table.
      dispatch.unregister_dispatch_for(masked_concat)

  def testDispatchForUnion(self):
    # [FUNCTION DOC]
    # Scenario: signature uses a union of masked tensor or plain tensor for both arguments. Adds a
    # masked tensor and a plain tensor; asserts the sum of values and the original mask (the
    # dispatch triggers because one argument is a composite type). Unregisters.
    MaybeMasked = typing.Union[MaskedTensor, tensor_lib.Tensor]

    @dispatch.dispatch_for_api(math_ops.add, {
        "x": MaybeMasked,
        "y": MaybeMasked
    })
    def masked_add(x, y, name=None):
      # [FUNCTION DOC]
      # Dispatch target: unwraps masked arguments (plain ones get an all-true mask), adds values,
      # ands masks, returns a masked tensor.
      with ops.name_scope(name):
        x_values = x.values if isinstance(x, MaskedTensor) else x
        x_mask = x.mask if isinstance(x, MaskedTensor) else True
        y_values = y.values if isinstance(y, MaskedTensor) else y
        y_mask = y.mask if isinstance(y, MaskedTensor) else True
        return MaskedTensor(x_values + y_values, x_mask & y_mask)

    try:
      x = MaskedTensor([1, 2, 3, 4, 5], [1, 0, 1, 1, 1])
      y = constant_op.constant([10, 20, 30, 40, 50])
      z = math_ops.add(x, y)
      self.assertAllEqual(z.values, x.values + y)
      self.assertAllEqual(z.mask, x.mask)

    finally:
      # Clean up dispatch table.
      dispatch.unregister_dispatch_for(masked_add)

  def testDispatchForTensorLike(self):
    # [FUNCTION DOC]
    # Scenario: the signature is read from annotations that union the masked type with the
    # tensor-like type. Adds a masked tensor to a list, an array, a constant tensor and a
    # variable (initializing variables in graph mode); asserts values are summed and the mask is
    # kept each time. Unregisters.
    MaskedOrTensorLike = typing.Union[MaskedTensor, core_tf_types.TensorLike]

    @dispatch.dispatch_for_api(math_ops.add)
    def masked_add(x: MaskedOrTensorLike, y: MaskedOrTensorLike, name=None):
      # [FUNCTION DOC]
      # Dispatch target: unwraps masked arguments (others get an all-true mask), adds values, ands
      # masks, returns a masked tensor.
      with ops.name_scope(name):
        x_values = x.values if isinstance(x, MaskedTensor) else x
        x_mask = x.mask if isinstance(x, MaskedTensor) else True
        y_values = y.values if isinstance(y, MaskedTensor) else y
        y_mask = y.mask if isinstance(y, MaskedTensor) else True
        return MaskedTensor(x_values + y_values, x_mask & y_mask)

    try:
      x = MaskedTensor([1, 2, 3, 4, 5], [1, 0, 1, 1, 1])
      y1 = [10, 20, 30, 40, 50]
      y2 = np.array([10, 20, 30, 40, 50])
      y3 = constant_op.constant([10, 20, 30, 40, 50])
      y4 = variables.Variable([5, 4, 3, 2, 1])
      if not context.executing_eagerly():
        self.evaluate(variables.global_variables_initializer())
      for y in [y1, y2, y3, y4]:
        z = math_ops.add(x, y)
        self.assertAllEqual(z.values, x.values + y)
        self.assertAllEqual(z.mask, x.mask)

    finally:
      # Clean up dispatch table.
      dispatch.unregister_dispatch_for(masked_add)

  def testDispatchForOptional(self):
    # [FUNCTION DOC]
    # Scenario: signature for the conditional-select API with optional masked arguments. Asserts
    # that calling with only the condition, and with all three masked arguments, both reach the
    # target and return its stub string. Unregisters.
    # Note: typing.Optional[X] == typing.Union[X, NoneType].

    @dispatch.dispatch_for_api(
        array_ops.where_v2, {
            "condition": MaskedTensor,
            "x": typing.Optional[MaskedTensor],
            "y": typing.Optional[MaskedTensor]
        })
    def masked_where(condition, x=None, y=None, name=None):
      # [FUNCTION DOC]
      # Dispatch target stub: ignores all inputs and returns a fixed stub string.
      del condition, x, y, name
      return "stub"

    try:
      x = MaskedTensor([True, False, True, True, True], [1, 0, 1, 1, 1])
      self.assertEqual(array_ops.where_v2(x), "stub")
      self.assertEqual(array_ops.where_v2(x, x, x), "stub")

    finally:
      # Clean up dispatch table.
      dispatch.unregister_dispatch_for(masked_where)

  def testDispatchForSignatureFromAnnotations(self):
    # [FUNCTION DOC]
    # Scenario: no explicit signature; it is derived from the target's type annotations. Asserts
    # adding two masked tensors uses the target (summed values, anded masks). Unregisters.

    @dispatch.dispatch_for_api(math_ops.add)
    def masked_add(x: MaskedTensor, y: MaskedTensor, name=None):
      # [FUNCTION DOC]
      # Dispatch target: inside a name scope adds the values and ands the masks of two masked
      # tensors into a new masked tensor.
      with ops.name_scope(name):
        return MaskedTensor(x.values + y.values, x.mask & y.mask)

    try:
      x = MaskedTensor([1, 2, 3, 4, 5], [1, 0, 1, 1, 1])
      y = MaskedTensor([1, 1, 1, 1, 1], [1, 1, 0, 1, 0])
      z = math_ops.add(x, y)
      self.assertAllEqual(z.values, x.values + y.values)
      self.assertAllEqual(z.mask, x.mask & y.mask)

    finally:
      # Clean up dispatch table.
      dispatch.unregister_dispatch_for(masked_add)

  def testDispatchForPositionalSignature(self):
    # [FUNCTION DOC]
    # Scenario: the signature keys are parameter positions rather than names. Asserts adding two
    # masked tensors uses the target (summed values, anded masks). Unregisters.

    @dispatch.dispatch_for_api(math_ops.add, {0: MaskedTensor, 1: MaskedTensor})
    def masked_add(x, y, name=None):
      # [FUNCTION DOC]
      # Dispatch target: inside a name scope adds the values and ands the masks of two masked
      # tensors into a new masked tensor.
      with ops.name_scope(name):
        return MaskedTensor(x.values + y.values, x.mask & y.mask)

    try:
      x = MaskedTensor([1, 2, 3, 4, 5], [1, 0, 1, 1, 1])
      y = MaskedTensor([1, 1, 1, 1, 1], [1, 1, 0, 1, 0])
      z = math_ops.add(x, y)
      self.assertAllEqual(z.values, x.values + y.values)
      self.assertAllEqual(z.mask, x.mask & y.mask)

    finally:
      # Clean up dispatch table.
      dispatch.unregister_dispatch_for(masked_add)

  def testDispatchWithVarargs(self):
    # [FUNCTION DOC]
    # Scenario: the target accepts only variadic positional and keyword arguments (always allowed
    # by the signature check). Calls add with positional masked arguments; the target asserts it
    # receives both positionally with no keywords, and the call returns its stub string.
    # Unregisters.

    @dispatch.dispatch_for_api(math_ops.add, {
        "x": MaskedTensor,
        "y": MaskedTensor
    })
    def masked_add(*args, **kwargs):
      # [FUNCTION DOC]
      # Dispatch target: asserts the two positional arguments' values equal those of the enclosing
      # test's masked tensors and that no keyword arguments were passed, then returns a stub.
      self.assertAllEqual(args[0].values, x.values)
      self.assertAllEqual(args[1].values, y.values)
      self.assertEmpty(kwargs)
      return "stub"

    try:
      x = MaskedTensor([1, 2, 3, 4, 5], [1, 0, 1, 1, 1])
      y = MaskedTensor([1, 1, 1, 1, 1], [1, 1, 0, 1, 0])
      self.assertEqual(math_ops.add(x, y), "stub")

    finally:
      # Clean up dispatch table.
      dispatch.unregister_dispatch_for(masked_add)

  def testDispatchWithKwargs(self):
    # [FUNCTION DOC]
    # Scenario: like the variadic test but the API is called with keyword arguments. The target
    # asserts both arrive as keywords with no positional arguments, and the call returns its stub
    # string. Unregisters.

    @dispatch.dispatch_for_api(math_ops.add, {
        "x": MaskedTensor,
        "y": MaskedTensor
    })
    def masked_add(*args, **kwargs):
      # [FUNCTION DOC]
      # Dispatch target: asserts the keyword arguments' values equal those of the enclosing test's
      # masked tensors and that no positional arguments were passed, then returns a stub.
      self.assertAllEqual(kwargs["x"].values, x.values)
      self.assertAllEqual(kwargs["y"].values, y.values)
      self.assertEmpty(args)
      return "stub"

    try:
      x = MaskedTensor([1, 2, 3, 4, 5], [1, 0, 1, 1, 1])
      y = MaskedTensor([1, 1, 1, 1, 1], [1, 1, 0, 1, 0])
      self.assertEqual(math_ops.add(x=x, y=y), "stub")

    finally:
      # Clean up dispatch table.
      dispatch.unregister_dispatch_for(masked_add)

  def testDispatchErrorForBadAPI(self):
    # [FUNCTION DOC]
    # Scenario: try to register a target for a plain function lacking dispatch support. Asserts a
    # value error saying the API does not support dispatch.

    def api_without_dispatch_support(x):
      # [FUNCTION DOC]
      # Plain function with no dispatch support; returns its input plus one.
      return x + 1

    with self.assertRaisesRegex(ValueError, ".* does not support dispatch."):

      @dispatch.dispatch_for_api(api_without_dispatch_support,
                                 {"x": MaskedTensor})
      def my_version(x):  # pylint: disable=unused-variable
        # [FUNCTION DOC]
        # Target that is never registered because the API is rejected; discards its input.
        del x

  def testDispatchErrorForNoSignature(self):
    # [FUNCTION DOC]
    # Scenario: register a target with neither explicit signatures nor annotations. Asserts a value
    # error saying the decorator must be given at least one signature.
    with self.assertRaisesRegex(ValueError,
                                "must be called with at least one signature"):

      @dispatch.dispatch_for_api(math_ops.add)
      def my_add(x, y, name=None):  # pylint: disable=unused-variable
        # [FUNCTION DOC]
        # Unannotated target that is never registered; discards its inputs.
        del x, y, name

  def testDispatchErrorSignatureMismatchParamName(self):
    # [FUNCTION DOC]
    # Scenario: the target renames the second parameter. Asserts a value error stating the target's
    # signature does not match the API's signature.
    with self.assertRaisesRegex(
        ValueError, r"Dispatch function's signature \(x, why, name=None\) does "
        r"not match API's signature \(x, y, name=None\)."):

      @dispatch.dispatch_for_api(math_ops.add, {"x": MaskedTensor})
      def my_add(x, why, name=None):  # pylint: disable=unused-variable
        # [FUNCTION DOC]
        # Target with a mismatching parameter name; rejected at registration; discards inputs.
        del x, why, name

  def testDispatchErrorSignatureMismatchExtraParam(self):
    # [FUNCTION DOC]
    # Scenario: the target has an extra parameter beyond the API's. Asserts a value error stating
    # the target's signature does not match the API's signature.
    with self.assertRaisesRegex(
        ValueError, r"Dispatch function's signature \(x, y, name=None, extra_"
        r"arg=None\) does not match API's signature \(x, y, name=None\)."):

      @dispatch.dispatch_for_api(math_ops.add, {"x": MaskedTensor})
      def my_add(x, y, name=None, extra_arg=None):  # pylint: disable=unused-variable
        # [FUNCTION DOC]
        # Target with an extra parameter; rejected at registration; discards its inputs.
        del x, y, name, extra_arg

  def testDispatchErrorForUnsupportedTypeAnnotation(self):
    # [FUNCTION DOC]
    # Scenario: the signature uses a fixed-length tuple annotation, which dispatch does not
    # support. Asserts a value error saying the type annotation is not currently supported.
    with self.assertRaisesRegex(
        ValueError,
        "Type annotation .* is not currently supported by dispatch."):

      @dispatch.dispatch_for_api(math_ops.add,
                                 {"x": typing.Tuple[MaskedTensor]})
      def my_add(x, y, name=None):  # pylint: disable=unused-variable
        # [FUNCTION DOC]
        # Target never registered because the annotation is rejected; discards its inputs.
        del x, y, name

  def testDispatchErrorForUnknownParameter(self):
    # [FUNCTION DOC]
    # Scenario: the signature names a parameter the API does not have. Asserts a value error
    # saying the signature includes an annotation for an unknown parameter.
    with self.assertRaisesRegex(
        ValueError, "signature includes annotation for unknown parameter 'z'."):

      @dispatch.dispatch_for_api(math_ops.add, {"z": MaskedTensor})
      def my_add(x, y, name=None):  # pylint: disable=unused-variable
        # [FUNCTION DOC]
        # Target never registered because the parameter is unknown; discards its inputs.
        del x, y, name

  def testDispatchErrorUnsupportedKeywordOnlyAnnotation(self):
    # [FUNCTION DOC]
    # Scenario: annotate a keyword-only parameter of a dispatch-enabled function. Asserts a value
    # error saying dispatch only supports annotations for positional parameters.

    @dispatch.add_dispatch_support
    def foo(x, *, y):
      # [FUNCTION DOC]
      # Dispatch-enabled test function with a keyword-only parameter; returns the sum of inputs.
      return x + y

    with self.assertRaisesRegex(
        ValueError, "Dispatch currently only supports type "
        "annotations for positional parameters"):

      @dispatch.dispatch_for_api(foo, {"y": MaskedTensor})
      def masked_foo(x, *, y):  # pylint: disable=unused-variable
        # [FUNCTION DOC]
        # Target never registered because the keyword-only annotation is rejected; discards inputs.
        del x, y

  def testDispatchErrorBadSignatureType(self):
    # [FUNCTION DOC]
    # Scenario: pass a signature that is a list, and another keyed by the null value, instead of
    # a mapping of parameter names or positions. Asserts a type error saying signatures must be
    # dictionaries mapping parameter names to type annotations, both times.
    with self.assertRaisesRegex(
        TypeError, "signatures must be dictionaries mapping parameter "
        "names to type annotations"):

      @dispatch.dispatch_for_api(math_ops.add, [MaskedTensor])
      def my_add(x, y, name=None):  # pylint: disable=unused-variable
        # [FUNCTION DOC]
        # Target never registered because the signature is malformed; discards its inputs.
        del x, y, name

    with self.assertRaisesRegex(
        TypeError, "signatures must be dictionaries mapping parameter "
        "names to type annotations"):

      @dispatch.dispatch_for_api(math_ops.multiply, {None: MaskedTensor})
      def my_multiply(x, y, name=None):  # pylint: disable=unused-variable
        # [FUNCTION DOC]
        # Target never registered because the signature key is invalid; discards its inputs.
        del x, y, name

  def testDispatchErrorNotCallable(self):
    # [FUNCTION DOC]
    # Scenario: apply the decorator returned for the absolute-value API to a string. Asserts a
    # type error saying the dispatch target is expected to be callable.
    with self.assertRaisesRegex(TypeError,
                                "Expected dispatch_target to be callable"):
      dispatch.dispatch_for_api(math_ops.abs, {0: MaskedTensor})("not_callable")

  def testRegisterDispatchableType(self):
    # [FUNCTION DOC]
    # Scenario: register a plain named tuple class as a dispatchable type, then register an add
    # target keyed on it. Asserts adding two such tuples (holding constant tensors) returns a
    # tuple with summed fields (11 and 23). Unregisters the target.
    Car = collections.namedtuple("Car", ["size", "speed"])
    dispatch.register_dispatchable_type(Car)

    @dispatch.dispatch_for_api(math_ops.add, {"x": Car, "y": Car})
    def add_car(x, y, name=None):
      # [FUNCTION DOC]
      # Dispatch target: inside a name scope returns a tuple whose fields are the sums of the
      # corresponding size and speed fields.
      with ops.name_scope(name):
        return Car(x.size + y.size, x.speed + y.speed)

    try:
      x = Car(constant_op.constant(1), constant_op.constant(3))
      y = Car(constant_op.constant(10), constant_op.constant(20))
      z = math_ops.add(x, y)
      self.assertAllEqual(z.size, 11)
      self.assertAllEqual(z.speed, 23)

    finally:
      # Clean up dispatch table.
      dispatch.unregister_dispatch_for(add_car)

  def testTypeCheckersAreCached(self):
    # [FUNCTION DOC]
    # Scenario: build a type checker for the same class twice. Asserts both calls return the very
    # same cached checker object.
    checker1 = dispatch.make_type_checker(int)
    checker2 = dispatch.make_type_checker(int)
    self.assertIs(checker1, checker2)

  def testDispatchTargetWithNoNameArgument(self):
    # [FUNCTION DOC]
    # Scenario: the target omits the API's name parameter, so dispatch wraps it. Calls add with the
    # name given by keyword, by position, and not at all. In graph mode asserts result tensor names
    # start with the given name scope (or the default op names). Unregisters.

    @dispatch.dispatch_for_api(math_ops.add, {
        "x": MaskedTensor,
        "y": MaskedTensor
    })
    def masked_add(x, y):
      # [FUNCTION DOC]
      # Dispatch target without a name parameter: adds values and ands masks of two masked
      # tensors into a new masked tensor.
      return MaskedTensor(x.values + y.values, x.mask & y.mask)

    try:
      x = MaskedTensor([1, 2, 3, 4, 5], [1, 0, 1, 1, 1])
      y = MaskedTensor([1, 1, 1, 1, 1], [1, 1, 0, 1, 0])

      # pass name w/ keyword arg
      a = math_ops.add(x, y, name="MyAdd")
      if not context.executing_eagerly():  # names not defined in eager mode.
        self.assertRegex(a.values.name, r"^MyAdd/add.*")
        self.assertRegex(a.mask.name, r"^MyAdd/and.*")

      # pass name w/ positional arg
      b = math_ops.add(x, y, "B")
      if not context.executing_eagerly():  # names not defined in eager mode.
        self.assertRegex(b.values.name, r"^B/add.*")
        self.assertRegex(b.mask.name, r"^B/and.*")

      # default name value
      c = math_ops.add(x, y)
      if not context.executing_eagerly():  # names not defined in eager mode.
        self.assertRegex(c.values.name, r"^add.*")
        self.assertRegex(c.mask.name, r"^and.*")

    finally:
      # Clean up dispatch table.
      dispatch.unregister_dispatch_for(masked_add)

  def testDispatchApiWithNoNameArg(self):
    # [FUNCTION DOC]
    # Scenario: the equality API has no name parameter. Registers a target with matching
    # parameters (accepted, then unregistered), then one with an extra name parameter. Asserts the
    # second registration raises a value error about mismatched signatures.
    # Note: The "tensor_equals" API has no "name" argument.
    signature = {"self": MaskedTensor, "other": MaskedTensor}

    @dispatch.dispatch_for_api(math_ops.tensor_equals, signature)
    def masked_tensor_equals(self, other):
      # [FUNCTION DOC]
      # Target stub with the same parameters as the API; discards its inputs and returns nothing.
      del self, other

    dispatch.unregister_dispatch_for(masked_tensor_equals)  # clean up.

    with self.assertRaisesRegex(
        ValueError, r"Dispatch function's signature \(self, other, name=None\) "
        r"does not match API's signature \(self, other\)\."):

      @dispatch.dispatch_for_api(math_ops.tensor_equals, signature)
      def masked_tensor_equals_2(self, other, name=None):
        # [FUNCTION DOC]
        # Target with an extra name parameter; rejected at registration; discards its inputs.
        del self, other, name

      del masked_tensor_equals_2  # avoid pylint unused variable warning.

  def testDispatchWithIterableParams(self):
    # [FUNCTION DOC]
    # Scenario: the sum-of-many API accepts one-shot iterables. Registers a target for a list of
    # masked tensors and calls the API with a generator of five masked tensors. Asserts the result
    # has the summed values (10) and the mask all-true. Unregisters.
    # The add_n API supports having `inputs` be an iterable (and not just
    # a sequence).
    @dispatch.dispatch_for_api(math_ops.add_n,
                               {"inputs": typing.List[MaskedTensor]})
    def masked_add_n(inputs):
      # [FUNCTION DOC]
      # Dispatch target: stacks the masks and reduces them with logical and along the first axis,
      # sums the values of all inputs, and returns a new masked tensor.
      masks = array_ops_stack.stack([x.mask for x in inputs])
      return MaskedTensor(
          math_ops.add_n([x.values for x in inputs]),
          math_ops.reduce_all(masks, axis=0))

    try:
      generator = (MaskedTensor([i], [True]) for i in range(5))
      y = math_ops.add_n(generator)
      self.assertAllEqual(y.values, [0 + 1 + 2 + 3 + 4])
      self.assertAllEqual(y.mask, [True])

    finally:
      # Clean up dispatch table.
      dispatch.unregister_dispatch_for(masked_add_n)

  def testBadIterableParametersError(self):
    # [FUNCTION DOC]
    # Scenario: pass a plain string instead of a list or tuple as the iterable-parameters option of
    # the dispatch-support decorator. Asserts a type error saying it should be a list or tuple of
    # strings.
    fn = lambda x: [t + 1 for t in x]
    with self.assertRaisesRegex(
        TypeError, "iterable_parameters should be a list or tuple of string"):
      dispatch.add_dispatch_support(iterable_parameters="x")(fn)

  def testUnregisterDispatchTargetBadTargetError(self):
    # [FUNCTION DOC]
    # Scenario: unregister a function that was never registered. Asserts a value error saying the
    # function was not registered.
    fn = lambda x: x + 1
    with self.assertRaisesRegex(ValueError, "Function .* was not registered"):
      dispatch.unregister_dispatch_for(fn)

  def testAddDuplicateApiDisptacherError(self):
    # [FUNCTION DOC]
    # Scenario: attach a type-based dispatcher to a function, then attach one again. Asserts the
    # second attempt raises a value error saying it already has one.
    some_op = lambda x: x
    some_op = dispatch.add_type_based_api_dispatcher(some_op)
    with self.assertRaisesRegex(
        ValueError, ".* already has a type-based API dispatcher."):
      some_op = dispatch.add_type_based_api_dispatcher(some_op)

  def testGetApisWithTypeBasedDispatch(self):
    # [FUNCTION DOC]
    # Scenario: list the APIs that support type-based dispatch. Asserts add and concat are among
    # them.
    dispatch_apis = dispatch.apis_with_type_based_dispatch()
    self.assertIn(math_ops.add, dispatch_apis)
    self.assertIn(array_ops.concat, dispatch_apis)

  def testTypeBasedDispatchTargetsFor(self):
    # [FUNCTION DOC]
    # Scenario: register targets for masked and silly extension types on add, concat and abs.
    # Queries the signatures that mention the masked type and asserts the result maps only add and
    # concat to their masked signatures (nothing from the silly type). Unregisters all targets.
    MaskedTensorList = typing.List[
        typing.Union[MaskedTensor, tensor_lib.Tensor]]
    try:

      @dispatch.dispatch_for_api(math_ops.add)
      def masked_add(x: MaskedTensor, y: MaskedTensor):
        # [FUNCTION DOC]
        # Annotated target stub for masked addition; discards its inputs.
        del x, y

      @dispatch.dispatch_for_api(array_ops.concat)
      def masked_concat(values: MaskedTensorList, axis):
        # [FUNCTION DOC]
        # Annotated target stub for concatenating a list of masked or plain tensors; discards its
        # inputs.
        del values, axis

      @dispatch.dispatch_for_api(math_ops.add)
      def silly_add(x: SillyTensor, y: SillyTensor):
        # [FUNCTION DOC]
        # Annotated target stub for silly-type addition; discards its inputs.
        del x, y

      @dispatch.dispatch_for_api(math_ops.abs)
      def silly_abs(x: SillyTensor):
        # [FUNCTION DOC]
        # Annotated target stub for silly-type absolute value; discards its input.
        del x

      # Note: `expected` does not contain keys or values from SillyTensor.
      targets = dispatch.type_based_dispatch_signatures_for(MaskedTensor)
      expected = {math_ops.add: [{"x": MaskedTensor, "y": MaskedTensor}],
                  array_ops.concat: [{"values": MaskedTensorList}]}
      self.assertEqual(targets, expected)

    finally:
      # Clean up dispatch table.
      dispatch.unregister_dispatch_for(masked_add)
      dispatch.unregister_dispatch_for(masked_concat)
      dispatch.unregister_dispatch_for(silly_add)
      dispatch.unregister_dispatch_for(silly_abs)

  def testDispatchForUnaryElementwiseAPIs(self):
    # [FUNCTION DOC]
    # Scenario: register one handler for all unary elementwise APIs on masked tensors. Calls abs,
    # sign, negative, invert and ones-like with mixed positional and keyword arguments and names.
    # Asserts the values match the expected elementwise results, every mask is unchanged, and (in
    # graph mode) names are scoped as given. Unregisters the handler.

    @dispatch.dispatch_for_unary_elementwise_apis(MaskedTensor)
    def unary_elementwise_api_handler(api_func, x):
      # [FUNCTION DOC]
      # Handler: applies the elementwise function to the values and keeps the mask, returning a
      # new masked tensor.
      return MaskedTensor(api_func(x.values), x.mask)

    try:
      x = MaskedTensor([1, -2, -3], [True, True, False])
      # Test calls with positional & keyword argument (& combinations)
      abs_x = math_ops.abs(x)
      sign_x = math_ops.sign(x=x)
      neg_x = math_ops.negative(x, "neg_x")
      invert_x = bitwise_ops.invert(x, name="invert_x")
      ones_like_x = array_ops.ones_like(x, name="ones_like_x")
      ones_like_x_float = array_ops.ones_like(
          x, dtypes.float32, name="ones_like_x_float")
      self.assertAllEqual(abs_x.values, [1, 2, 3])
      self.assertAllEqual(sign_x.values, [1, -1, -1])
      self.assertAllEqual(neg_x.values, [-1, 2, 3])
      self.assertAllEqual(invert_x.values, [-2, 1, 2])
      self.assertAllEqual(ones_like_x.values, [1, 1, 1])
      self.assertAllEqual(ones_like_x_float.values, [1., 1., 1.])
      for result in [
          abs_x, sign_x, neg_x, invert_x, ones_like_x, ones_like_x_float
      ]:
        self.assertAllEqual(result.mask, [True, True, False])
      if not context.executing_eagerly():  # names not defined in eager mode.
        self.assertRegex(neg_x.values.name, r"^neg_x/Neg:.*")
        self.assertRegex(invert_x.values.name, r"^invert_x/.*")
        self.assertRegex(ones_like_x.values.name, r"^ones_like_x/.*")
        self.assertRegex(ones_like_x_float.values.name,
                         r"^ones_like_x_float/.*")

    finally:
      dispatch.unregister_dispatch_for(unary_elementwise_api_handler)

  def testDispatchForBinaryElementwiseAPIs(self):
    # [FUNCTION DOC]
    # Scenario: register one handler for all binary elementwise APIs on pairs of masked tensors.
    # Calls multiply, add, subtract and minimum with positional and keyword arguments and names.
    # Asserts the values match the expected results, masks are the logical and of the inputs, and
    # (in graph mode) result names are scoped as given. Unregisters the handler.

    @dispatch.dispatch_for_binary_elementwise_apis(MaskedTensor, MaskedTensor)
    def binary_elementwise_api_handler(api_func, x, y):
      # [FUNCTION DOC]
      # Handler: applies the elementwise function to the two values and ands the masks, returning a
      # new masked tensor.
      return MaskedTensor(api_func(x.values, y.values), x.mask & y.mask)

    try:
      x = MaskedTensor([1, -2, -3], [True, True, False])
      y = MaskedTensor([10, 20, 30], [True, False, True])
      # Test calls with positional & keyword arguments (& combinations)
      x_times_y = math_ops.multiply(x, y)
      x_plus_y = math_ops.add(x, y=y)
      x_minus_y = math_ops.subtract(x=x, y=y)
      min_x_y = math_ops.minimum(x, y, "min_x_y")
      y_times_x = math_ops.multiply(y, x, name="y_times_x")
      y_plus_x = math_ops.add(y, y=x, name="y_plus_x")
      y_minus_x = math_ops.subtract(x=y, y=x, name="y_minus_x")
      self.assertAllEqual(x_times_y.values, [10, -40, -90])
      self.assertAllEqual(x_plus_y.values, [11, 18, 27])
      self.assertAllEqual(x_minus_y.values, [-9, -22, -33])
      self.assertAllEqual(min_x_y.values, [1, -2, -3])
      self.assertAllEqual(y_times_x.values, [10, -40, -90])
      self.assertAllEqual(y_plus_x.values, [11, 18, 27])
      self.assertAllEqual(y_minus_x.values, [9, 22, 33])
      for result in [
          x_times_y, x_plus_y, x_minus_y, min_x_y, y_times_x, y_plus_x,
          y_minus_x
      ]:
        self.assertAllEqual(result.mask, [True, False, False])
      if not context.executing_eagerly():  # names not defined in eager mode.
        self.assertRegex(min_x_y.values.name, r"^min_x_y/Minimum:.*")
        self.assertRegex(min_x_y.mask.name, r"^min_x_y/and:.*")
        self.assertRegex(y_times_x.values.name, r"^y_times_x/.*")
        self.assertRegex(y_plus_x.values.name, r"^y_plus_x/.*")
        self.assertRegex(y_minus_x.values.name, r"^y_minus_x/.*")

    finally:
      dispatch.unregister_dispatch_for(binary_elementwise_api_handler)

  def testDuplicateDispatchForUnaryElementwiseAPIsError(self):
    # [FUNCTION DOC]
    # Scenario: register a second unary elementwise handler for the same type. Asserts a value
    # error saying a unary handler has already been registered. Unregisters the first handler.

    @dispatch.dispatch_for_unary_elementwise_apis(MaskedTensor)
    def handler(api_func, x):
      # [FUNCTION DOC]
      # First handler: applies the function to the values and keeps the mask.
      return MaskedTensor(api_func(x.values), x.mask)

    try:
      with self.assertRaisesRegex(
          ValueError, r"A unary elementwise dispatch handler \(.*\) has "
          "already been registered for .*"):

        @dispatch.dispatch_for_unary_elementwise_apis(MaskedTensor)
        def another_handler(api_func, x):
          # [FUNCTION DOC]
          # Duplicate handler (inverts the mask); registration is rejected so it is never called.
          return MaskedTensor(api_func(x.values), ~x.mask)

        del another_handler

    finally:
      dispatch.unregister_dispatch_for(handler)

  def testDuplicateDispatchForBinaryElementwiseAPIsError(self):
    # [FUNCTION DOC]
    # Scenario: register a second binary elementwise handler for the same type pair. Asserts a
    # value error saying a binary handler has already been registered. Unregisters the first.

    @dispatch.dispatch_for_binary_elementwise_apis(MaskedTensor, MaskedTensor)
    def handler(api_func, x, y):
      # [FUNCTION DOC]
      # First handler: applies the function to both values and ands the masks.
      return MaskedTensor(api_func(x.values, y.values), x.mask & y.mask)

    try:
      with self.assertRaisesRegex(
          ValueError, r"A binary elementwise dispatch handler \(.*\) has "
          "already been registered for .*"):

        @dispatch.dispatch_for_binary_elementwise_apis(MaskedTensor,
                                                       MaskedTensor)
        def another_handler(api_func, x, y):
          # [FUNCTION DOC]
          # Duplicate handler (keeps only the first mask); registration is rejected so it is never
          # called.
          return MaskedTensor(api_func(x.values, y.values), x.mask)

        del another_handler

    finally:
      dispatch.unregister_dispatch_for(handler)

  def testRegisterUnaryElementwiseApiAfterHandler(self):
    # [FUNCTION DOC]
    # Scenario: a unary handler exists first, then a new dispatch-enabled op is registered as a
    # unary elementwise API. Asserts calling the new op on a masked tensor uses the handler
    # (doubled values, unchanged mask). Unregisters the handler.
    # Test that it's ok to call register_unary_elementwise_api after
    # dispatch_for_unary_elementwise_apis.

    @dispatch.dispatch_for_unary_elementwise_apis(MaskedTensor)
    def handler(api_func, x):
      # [FUNCTION DOC]
      # Handler: applies the function to the values and keeps the mask.
      return MaskedTensor(api_func(x.values), x.mask)

    try:

      @dispatch.register_unary_elementwise_api
      @dispatch.add_dispatch_support
      def some_op(x):
        # [FUNCTION DOC]
        # Test unary op: returns its input doubled.
        return x * 2

      x = MaskedTensor([1, 2, 3], [True, False, True])
      y = some_op(x)
      self.assertAllEqual(y.values, [2, 4, 6])
      self.assertAllEqual(y.mask, [True, False, True])

    finally:
      dispatch.unregister_dispatch_for(handler)

  def testRegisterBinaryElementwiseApiAfterHandler(self):
    # [FUNCTION DOC]
    # Scenario: a binary handler exists first, then a new dispatch-enabled op is registered as a
    # binary elementwise API. Asserts calling the new op on two masked tensors uses the handler
    # (values 12, 24, 36; mask true, false, false). Unregisters the handler.
    # Test that it's ok to call register_binary_elementwise_api after
    # dispatch_for_binary_elementwise_apis.

    @dispatch.dispatch_for_binary_elementwise_apis(MaskedTensor, MaskedTensor)
    def handler(api_func, x, y):
      # [FUNCTION DOC]
      # Handler: applies the function to both values and ands the masks.
      return MaskedTensor(api_func(x.values, y.values), x.mask & y.mask)

    try:

      @dispatch.register_binary_elementwise_api
      @dispatch.add_dispatch_support
      def some_op(x, y):
        # [FUNCTION DOC]
        # Test binary op: returns twice the first input plus the second.
        return x * 2 + y

      x = MaskedTensor([1, 2, 3], [True, False, True])
      y = MaskedTensor([10, 20, 30], [True, True, False])
      z = some_op(x, y)
      self.assertAllEqual(z.values, [12, 24, 36])
      self.assertAllEqual(z.mask, [True, False, False])

    finally:
      dispatch.unregister_dispatch_for(handler)

  def testElementwiseApiLists(self):
    # [FUNCTION DOC]
    # Scenario: query the registered unary and binary elementwise API collections. Asserts abs and
    # cos are unary, and add and multiply are binary.
    self.assertIn(math_ops.abs, dispatch.unary_elementwise_apis())
    self.assertIn(math_ops.cos, dispatch.unary_elementwise_apis())
    self.assertIn(math_ops.add, dispatch.binary_elementwise_apis())
    self.assertIn(math_ops.multiply, dispatch.binary_elementwise_apis())

  def testUpdateDocstringsWithAPILists(self):
    # [FUNCTION DOC]
    # Scenario: run the docstring updater, then check the decorators' docstrings. Asserts the
    # generic dispatch decorator lists concat and add, the unary decorator lists abs and cos, and
    # the binary decorator lists add and multiply, each with parameter names. Mutates the
    # decorators' docstrings (idempotent once placeholders are replaced).
    dispatch.update_docstrings_with_api_lists()
    self.assertRegex(
        dispatch.dispatch_for_api.__doc__,
        r"(?s)The TensorFlow APIs that may be overridden "
        r"by `@dispatch_for_api` are:.*"
        r"\* `tf\.concat\(values, axis, name\)`.*"
        r"\* `tf\.math\.add\(x, y, name\)`.*")
    self.assertRegex(
        dispatch.dispatch_for_unary_elementwise_apis.__doc__,
        r"(?s)The unary elementwise APIs are:.*"
        r"\* `tf\.math\.abs\(x, name\)`.*"
        r"\* `tf\.math\.cos\(x, name\)`.*")
    self.assertRegex(
        dispatch.dispatch_for_binary_elementwise_apis.__doc__,
        r"(?s)The binary elementwise APIs are:.*"
        r"\* `tf\.math\.add\(x, y, name\)`.*"
        r"\* `tf\.math\.multiply\(x, y, name\)`.*")


if __name__ == "__main__":
  googletest.main()

