# [MODULE DOC]
# Unit tests for the core graph-construction module tensorflow.python.framework.ops.
# Covers tensors and their shapes, bitwise operators, indexed slices and their type spec, operation
# construction from node definitions, creating operations in a graph (including from native
# operations and inside cond/while control-flow contexts), name scopes and unique naming, device
# scopes and device functions, graph state isolation between threads, collections, control
# dependencies, colocation, graph-level utilities, gradient helpers, tensor/op conversion
# registration, and related behaviors.
# Test module only; exposes no API. It defines test case classes plus one small shared helper that
# creates an operation and returns its outputs. The module is run through the standard test runner
# entry point at the bottom of the file.
# Import-time side effects: marks the float-output test op as not differentiable, registers gradient
# functions for two test copy ops (including an override name), registers a statistics function for
# a test op type, and registers a tensor conversion function for a pretend user tuple class.
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
"""Tests for tensorflow.python.framework.ops."""

import gc
import os
import threading
import weakref

from absl.testing import parameterized
import numpy as np

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import full_type_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as eager_function
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import function
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import versions
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import resources
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
import tensorflow.python.ops.gradients  # pylint: disable=unused-import
from tensorflow.python.platform import googletest
from tensorflow.python.util import compat


class ResourceTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for resource handles and the shared-resource registry (stub resource ops).
  # Each test builds a graph-mode session through the cached session fixture and uses test-only
  # stub resource ops to create and report on resources.

  @test_util.run_deprecated_v1
  def testBuildGraph(self):
    # [FUNCTION DOC]
    # Graph-mode test (deprecated v1 behavior). Creates a stub resource handle with a container and
    # shared name, and runs the resource-create op on it. Asserts nothing explicitly; passes if the
    # graph builds and the op runs without raising.
    with self.cached_session():
      pt = test_ops.stub_resource_handle_op(container="a", shared_name="b")
      test_ops.resource_create_op(pt).run()

  @test_util.run_deprecated_v1
  def testInitialize(self):
    # [FUNCTION DOC]
    # Graph-mode test. Registers a stub resource (handle, create op, initialized-check op) with the
    # resource registry. Asserts that exactly one resource is reported uninitialized, then runs
    # resource initialization and asserts that none are reported uninitialized afterwards.
    with self.cached_session():
      handle = test_ops.stub_resource_handle_op(container="a", shared_name="b")
      resources.register_resource(
          handle=handle,
          create_op=test_ops.resource_create_op(handle),
          is_initialized_op=test_ops.resource_initialized_op(handle))
      self.assertEqual(
          len(
              resources.report_uninitialized_resources(
                  resources.shared_resources()).eval()), 1)
      resources.initialize_resources(resources.shared_resources()).run()
      self.assertEqual(
          len(
              resources.report_uninitialized_resources(
                  resources.shared_resources()).eval()), 0)


class TensorAndShapeTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for tensor and tensor-shape behavior: static shape inference through arithmetic,
  # numpy-like attribute handling, iteration and implicit-bool errors in graph mode, tensor
  # reference objects (hashable wrappers), and the bitwise operators on integer and boolean tensors.

  def testShape(self):
    # [FUNCTION DOC]
    # Creates an operation with one float output from a node definition and checks that its output
    # tensor initially has unknown shape, then that setting the shape to a known three-dimensional
    # shape is reflected by the tensor's reported shape.
    op = ops.Operation.from_node_def(
        ops._NodeDef("FloatOutput", "myop"), ops.Graph(), [], [dtypes.float32]
    )
    t = op.outputs[0]
    self.assertEqual(tensor_shape.unknown_shape(), t.get_shape())
    t.set_shape([1, 2, 3])
    self.assertEqual([1, 2, 3], t.get_shape())

  def testNdim(self):
    # [FUNCTION DOC]
    # Checks that the number-of-dimensions attribute of a tensor is correct inside a traced function.
    # Builds a function (traced by the function decorator) that asserts the rank of its argument is
    # two, then calls it with a three-by-four tensor of zeros. Nothing is returned.

    @def_function.function
    def f(a):
      # [FUNCTION DOC]
      # HELPER: Traced body used by the number-of-dimensions test.
      # Nested function traced by the function decorator. Takes a tensor argument, asserts that its
      # number of dimensions equals two (failing the enclosing test otherwise), and returns the integer
      # zero.
      self.assertEqual(a.ndim, 2)
      return 0

    x = array_ops.zeros((3, 4))
    f(x)

  def testIterable(self):
    # [FUNCTION DOC]
    # Eager-only test (skipped when not executing eagerly). Builds an operation output tensor and
    # asserts that trying to iterate over it raises a type error whose message says the tensor
    # cannot be iterated.
    if not context.executing_eagerly():
      self.skipTest("Eager-mode test")
    op = ops.Operation.from_node_def(
        ops._NodeDef("FloatOutput", "myop"), ops.Graph(), [], [dtypes.float32]
    )
    t = op.outputs[0]
    with self.assertRaisesRegex(TypeError, "Cannot iterate"):
      iter(t)

  def testIterableGraph(self):
    # [FUNCTION DOC]
    # Graph-mode-only test (skipped when executing eagerly). Asserts that iterating a symbolic tensor
    # raises a type error with a message about iteration being disallowed in graph mode. Repeats under
    # an enabled autograph status (message mentions unsupported autograph feature) and under a
    # disabled autograph status (message says the code may not be visible to autograph).
    if context.executing_eagerly():
      self.skipTest("Graph-mode test")

    op = ops.Operation.from_node_def(
        ops._NodeDef("FloatOutput", "myop"), ops.Graph(), [], [dtypes.float32]
    )
    t = op.outputs[0]
    with self.assertRaisesRegex(
        TypeError, "Iterating.*not allowed.*Graph mode"):
      next(iter(t))
    with self.assertRaisesRegex(
        TypeError, "Iterating.*AutoGraph.*unsupported feature"):
      with ag_ctx.ControlStatusCtx(ag_ctx.Status.ENABLED):
        next(iter(t))
    with self.assertRaisesRegex(
        TypeError, "Iterating.*AutoGraph.*not be visible"):
      with ag_ctx.ControlStatusCtx(ag_ctx.Status.DISABLED):
        next(iter(t))

  def testImplicitBool(self):
    # [FUNCTION DOC]
    # Asserts that using a symbolic boolean tensor as a Python truth value raises a type error whose
    # message says implicit bool use is not allowed in graph mode. Repeats under an enabled autograph
    # control status (message about an unsupported autograph feature) and a disabled one (message
    # that the code may not be visible to autograph).
    op = ops.Operation.from_node_def(
        ops._NodeDef("FloatOutput", "myop"), ops.Graph(), [], [dtypes.bool]
    )
    t = op.outputs[0]
    with self.assertRaisesRegex(
        TypeError, "Using.*as a.*bool.*not allowed.*Graph mode"):
      bool(t)
    with self.assertRaisesRegex(
        TypeError, "Using.*as a.*bool.*AutoGraph.*unsupported feature"):
      with ag_ctx.ControlStatusCtx(ag_ctx.Status.ENABLED):
        bool(t)
    with self.assertRaisesRegex(
        TypeError, "Using.*as a.*bool.*AutoGraph.*not be visible"):
      with ag_ctx.ControlStatusCtx(ag_ctx.Status.DISABLED):
        bool(t)

  def testAddShape(self):
    # [FUNCTION DOC]
    # Adds a two-by-three tensor of zeros to a one-by-three tensor of ones and asserts the sum's
    # static shape is two by three (broadcasting shape inference).
    with self.cached_session():
      a = array_ops.zeros([2, 3])
      b = array_ops.ones([1, 3])
      c = a + b
      self.assertEqual([2, 3], c.shape)

  @test_util.run_deprecated_v1
  def testUnknownDim(self):
    # [FUNCTION DOC]
    # Graph-mode test. Adds two placeholders of shape two by unknown by three and asserts that the
    # sum's static shape as a list is two, unknown, three.
    with self.cached_session():
      a = array_ops.placeholder(dtype=dtypes.float32, shape=[2, None, 3])
      b = array_ops.placeholder(dtype=dtypes.float32, shape=[2, None, 3])
      c = a + b
      self.assertEqual([2, None, 3], c.shape.as_list())

  @test_util.run_deprecated_v1
  def testUnknownShape(self):
    # [FUNCTION DOC]
    # Graph-mode test. Adds a placeholder of completely unknown shape to a one-by-three tensor and
    # asserts the result has unknown shape.
    with self.cached_session():
      a = array_ops.placeholder(dtype=dtypes.float32, shape=None)
      b = array_ops.ones([1, 3])
      c = a + b
      self.assertEqual(tensor_shape.unknown_shape(), c.shape)

  @test_util.run_deprecated_v1
  def testScalarShape(self):
    # [FUNCTION DOC]
    # Graph-mode test. Adds a scalar-shaped placeholder to a scalar tensor and asserts the result's
    # static shape is the empty (scalar) shape.
    with self.cached_session():
      a = array_ops.placeholder(dtype=dtypes.float32, shape=[])
      b = array_ops.ones([])
      c = a + b
      self.assertEqual(tensor_shape.TensorShape([]), c.shape)

  @test_util.run_deprecated_v1
  def testShapeFunctionError(self):
    # [FUNCTION DOC]
    # Graph-mode test. Adds tensors of shapes one-two-three and four-five-six and asserts a value
    # error is raised whose message says dimensions must be equal (2 versus 5) for the add op and
    # includes both input shapes.
    with self.cached_session():
      a = array_ops.ones([1, 2, 3])
      b = array_ops.ones([4, 5, 6])
      with self.assertRaisesRegex(
          ValueError, r"Dimensions must be equal, but are 2 and 5 for .*add"
          r".*Add(V2)?.* with input shapes: \[1,2,3\], \[4,5,6\]."):
        _ = a + b

  def testNumpyArray(self):
    # [FUNCTION DOC]
    # Asserts that converting a symbolic graph tensor to a numpy array raises a not-implemented error
    # naming the tensor, and that taking its length raises a type error saying the length is not well
    # defined. Then confirms that eager tensors still behave like numpy arrays: array conversion
    # equals an array of ones and the length is three.
    with ops.Graph().as_default():
      x = array_ops.ones((3, 4), name="test_ones")

    with self.assertRaisesRegex(NotImplementedError,
                                r"Cannot convert a symbolic.+test_ones"):
      np.array(x)

    with self.assertRaisesRegex(TypeError, "not well defined.+test_ones"):
      len(x)

    # EagerTensors should still behave as numpy arrays.
    with context.eager_mode():
      x = array_ops.ones((3, 4))

    self.assertAllEqual(x, np.ones((3, 4)))
    self.assertAllEqual(np.array(x), np.ones((3, 4)))
    self.assertLen(x, 3)

  def testConstructor(self):
    # [FUNCTION DOC]
    # Asserts that accessing numpy-only attribute names (transpose, reshape, clip, size, tolist, data
    # and similar) on a tensor raises an attribute error with a hint about looking for numpy-related
    # methods, and that calling a nonexistent method raises an attribute error saying the object has
    # no such attribute.
    a = array_ops.ones([])
    for name in ["T", "astype", "ravel", "transpose", "reshape", "clip", "size",
                 "tolist", "data"]:
      with self.assertRaisesRegex(
          AttributeError, r"If you are looking for numpy-related methods"):
        getattr(a, name)
    with self.assertRaisesRegex(
        AttributeError, r"object has no attribute"):
      a.foo_bar()

  def testRef(self):
    # [FUNCTION DOC]
    # Checks equality semantics of tensor and variable reference wrappers: a reference equals another
    # reference to the same object (including aliases), and references to different tensors or
    # variables, even with equal values, are not equal.
    x1 = constant_op.constant(3)
    x2 = x1
    y = constant_op.constant(3)
    z = constant_op.constant([6, 10])
    w = variables.Variable(5)

    self.assertEqual(x1.ref(), x1.ref())
    self.assertEqual(x2.ref(), x2.ref())
    self.assertEqual(x1.ref(), x2.ref())
    self.assertEqual(y.ref(), y.ref())
    self.assertEqual(z.ref(), z.ref())
    self.assertEqual(w.ref(), w.ref())

    self.assertNotEqual(x1.ref(), y.ref())
    self.assertNotEqual(x1.ref(), z.ref())
    self.assertNotEqual(x1.ref(), w.ref())
    self.assertNotEqual(y.ref(), z.ref())
    self.assertNotEqual(y.ref(), w.ref())
    self.assertNotEqual(z.ref(), w.ref())

  def testRefDeref(self):
    # [FUNCTION DOC]
    # Checks that dereferencing a tensor or variable reference returns the very same object it came
    # from (identity), and that references to distinct objects dereference to distinct objects.
    x1 = constant_op.constant(3)
    x2 = x1
    y = constant_op.constant(3)
    z = constant_op.constant([6, 10])
    w = variables.Variable(5)

    self.assertIs(x1, x1.ref().deref())
    self.assertIs(x2, x2.ref().deref())
    self.assertIs(x1, x2.ref().deref())
    self.assertIs(x2, x1.ref().deref())
    self.assertIs(y, y.ref().deref())
    self.assertIs(z, z.ref().deref())

    self.assertIsNot(x1, y.ref().deref())
    self.assertIsNot(x1, z.ref().deref())
    self.assertIsNot(x1, w.ref().deref())
    self.assertIsNot(y, z.ref().deref())
    self.assertIsNot(y, w.ref().deref())
    self.assertIsNot(z, w.ref().deref())

  def testRefInSet(self):
    # [FUNCTION DOC]
    # Checks that tensor and variable references are hashable and usable in a set: five insertions of
    # which two are aliases of the same tensor yield a set of four, and membership holds for each.
    x1 = constant_op.constant(3)
    x2 = x1
    y = constant_op.constant(3)
    z = constant_op.constant([6, 10])
    w = variables.Variable(5)

    self.assertEqual(x1.ref(), x2.ref())

    tensor_set = {
        x1.ref(),
        x2.ref(),
        y.ref(),
        z.ref(),
        w.ref(),
    }

    self.assertLen(tensor_set, 4)
    self.assertIn(x1.ref(), tensor_set)
    self.assertIn(x2.ref(), tensor_set)
    self.assertIn(y.ref(), tensor_set)
    self.assertIn(z.ref(), tensor_set)
    self.assertIn(w.ref(), tensor_set)

  def testRefInDict(self):
    # [FUNCTION DOC]
    # Checks that references work as dictionary keys: four distinct keys give four entries; writing
    # through an alias reference overwrites the existing entry without growing the dictionary, and
    # all lookups return the expected values.
    x1 = constant_op.constant(3)
    x2 = x1
    y = constant_op.constant(3)
    z = constant_op.constant([6, 10])
    w = variables.Variable(5)

    self.assertEqual(x1.ref(), x2.ref())

    tensor_dict = {
        x1.ref(): "x1",
        y.ref(): "y",
        z.ref(): "z",
        w.ref(): "w",
    }

    self.assertLen(tensor_dict, 4)

    # Overwriting x1
    tensor_dict[x2.ref()] = "x2"
    self.assertLen(tensor_dict, 4)

    self.assertEqual(tensor_dict[x1.ref()], "x2")
    self.assertEqual(tensor_dict[x2.ref()], "x2")
    self.assertEqual(tensor_dict[y.ref()], "y")
    self.assertEqual(tensor_dict[z.ref()], "z")
    self.assertEqual(tensor_dict[w.ref()], "w")

  def testTensorRefStrong(self):
    # [FUNCTION DOC]
    # Checks that a tensor reference holds a strong reference: after deleting the original tensor
    # name, dereferencing the stored reference still returns a non-null object.
    x = constant_op.constant(1.)
    x_ref = x.ref()
    del x
    self.assertIsNotNone(x_ref.deref())

  def testVariableRefStrong(self):
    # [FUNCTION DOC]
    # Same as the tensor-reference strong-reference test but for a variable: after deleting the
    # variable name, dereferencing its reference still returns a non-null object.
    x = variables.Variable(1.)
    x_ref = x.ref()
    del x
    self.assertIsNotNone(x_ref.deref())

  @test_util.run_in_graph_and_eager_modes
  def testBitwiseAndNumeric(self):
    # [FUNCTION DOC]
    # Runs in both graph and eager modes. Checks bitwise-and on integer tensors gives the elementwise
    # result (zero, one, one for the sample inputs).
    x = constant_op.constant([0, 1, 3])
    y = constant_op.constant([1, 1, 1])

    z = x & y

    self.assertAllEqual(z, [0, 1, 1])

  @test_util.run_in_graph_and_eager_modes
  def testBitwiseAndBool(self):
    # [FUNCTION DOC]
    # Runs in both graph and eager modes. Checks bitwise-and on boolean tensors behaves as logical
    # and elementwise.
    x = constant_op.constant([False, False, True, True])
    y = constant_op.constant([False, True, False, True])

    z = x & y

    self.assertAllEqual(z, [False, False, False, True])

  @test_util.run_in_graph_and_eager_modes
  def testBitwiseAndErrors(self):
    # [FUNCTION DOC]
    # Runs in both graph and eager modes. Asserts that bitwise-and between an integer tensor and a
    # boolean tensor, a string tensor, or two string tensors raises an error: an invalid-argument
    # error when executing eagerly and a type error in graph mode.
    x_int = constant_op.constant(0)
    x_bool = constant_op.constant(True)

    if context.executing_eagerly():  # :(
      expected_errtype = errors.InvalidArgumentError
    else:
      expected_errtype = TypeError

    with self.assertRaises(expected_errtype):
      _ = x_int & x_bool
    with self.assertRaises(expected_errtype):
      _ = x_int & constant_op.constant("a")

    with self.assertRaises(expected_errtype):
      _ = x_bool & x_int
    with self.assertRaises(expected_errtype):
      _ = x_bool & constant_op.constant("a")

    with self.assertRaises(expected_errtype):
      _ = constant_op.constant("a") & constant_op.constant("b")

  @test_util.run_in_graph_and_eager_modes
  def testBitwiseOrNumeric(self):
    # [FUNCTION DOC]
    # Runs in both graph and eager modes. Checks bitwise-or on integer tensors gives the elementwise
    # result (one, one, three for the sample inputs).
    x = constant_op.constant([0, 1, 2])
    y = constant_op.constant([1, 1, 1])

    z = x | y

    self.assertAllEqual(z, [1, 1, 3])

  @test_util.run_in_graph_and_eager_modes
  def testBitwiseOrBool(self):
    # [FUNCTION DOC]
    # Runs in both graph and eager modes. Checks bitwise-or on boolean tensors behaves as logical or
    # elementwise.
    x = constant_op.constant([False, False, True, True])
    y = constant_op.constant([False, True, False, True])

    z = x | y

    self.assertAllEqual(z, [False, True, True, True])

  @test_util.run_in_graph_and_eager_modes
  def testBitwiseOrErrors(self):
    # [FUNCTION DOC]
    # Runs in both graph and eager modes. Asserts that bitwise-or with mismatched or string operands
    # (int with bool, int with string, bool with int, bool with string, string with string) raises
    # an invalid-argument error in eager mode and a type error in graph mode.
    x_int = constant_op.constant(0)
    x_bool = constant_op.constant(True)

    if context.executing_eagerly():  # :(
      expected_errtype = errors.InvalidArgumentError
    else:
      expected_errtype = TypeError

    with self.assertRaises(expected_errtype):
      _ = x_int | x_bool
    with self.assertRaises(expected_errtype):
      _ = x_int | constant_op.constant("a")

    with self.assertRaises(expected_errtype):
      _ = x_bool | x_int
    with self.assertRaises(expected_errtype):
      _ = x_bool | constant_op.constant("a")

    with self.assertRaises(expected_errtype):
      _ = constant_op.constant("a") | constant_op.constant("b")

  @test_util.run_in_graph_and_eager_modes
  def testBitwiseXorNumeric(self):
    # [FUNCTION DOC]
    # Runs in both graph and eager modes. Checks bitwise-xor on integer tensors gives the elementwise
    # result (one, zero, two for the sample inputs).
    x = constant_op.constant([0, 1, 3])
    y = constant_op.constant([1, 1, 1])

    z = x ^ y

    self.assertAllEqual(z, [1, 0, 2])

  @test_util.run_in_graph_and_eager_modes
  def testBitwiseXorBool(self):
    # [FUNCTION DOC]
    # Runs in both graph and eager modes. Checks bitwise-xor on boolean tensors behaves as logical
    # exclusive or elementwise.
    x = constant_op.constant([False, False, True, True])
    y = constant_op.constant([False, True, False, True])

    z = x ^ y

    self.assertAllEqual(z, [False, True, True, False])

  @test_util.run_in_graph_and_eager_modes
  def testBitwiseXorErrors(self):
    # [FUNCTION DOC]
    # Runs in both graph and eager modes. Asserts that bitwise-xor with mismatched or string operands
    # raises an invalid-argument error in eager mode and a type error in graph mode.
    x_int = constant_op.constant(0)
    x_bool = constant_op.constant(True)

    if context.executing_eagerly():  # :(
      expected_errtype = errors.InvalidArgumentError
    else:
      expected_errtype = TypeError

    with self.assertRaises(expected_errtype):
      _ = x_int ^ x_bool
    with self.assertRaises(expected_errtype):
      _ = x_int ^ constant_op.constant("a")

    with self.assertRaises(expected_errtype):
      _ = x_bool ^ x_int
    with self.assertRaises(expected_errtype):
      _ = x_bool ^ constant_op.constant("a")

    with self.assertRaises(expected_errtype):
      _ = constant_op.constant("a") ^ constant_op.constant("b")

  @test_util.run_in_graph_and_eager_modes
  def testBitwiseNotNumeric(self):
    # [FUNCTION DOC]
    # Runs in both graph and eager modes. Checks bitwise-not (inversion) on integer tensors, including
    # the minimum 32-bit integer which maps to the maximum, giving negative one, max, and negative two.
    x = constant_op.constant([0, dtypes.int32.min, 1])

    # pylint: disable=invalid-unary-operand-type
    y = ~x

    self.assertAllEqual(y, [-1, dtypes.int32.max, -2])

  @test_util.run_in_graph_and_eager_modes
  def testBitwiseNotBool(self):
    # [FUNCTION DOC]
    # Runs in both graph and eager modes. Checks bitwise-not on a boolean tensor flips each element
    # (logical negation).
    x = constant_op.constant([False, True])

    # pylint: disable=invalid-unary-operand-type
    y = ~x

    self.assertAllEqual(y, [True, False])

  @test_util.run_in_graph_and_eager_modes
  def testBitwiseNotErrors(self):
    # [FUNCTION DOC]
    # Runs in both graph and eager modes. Asserts that bitwise-not of a string tensor raises an
    # invalid-argument error in eager mode and a type error in graph mode.
    if context.executing_eagerly():  # :(
      expected_errtype = errors.InvalidArgumentError
    else:
      expected_errtype = TypeError

    # pylint: disable=invalid-unary-operand-type
    with self.assertRaises(expected_errtype):
      _ = ~constant_op.constant("a")


@test_util.run_all_in_graph_and_eager_modes
class IndexedSlicesTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for the indexed-slices value type (sparse gradient representation). The whole class
  # is wrapped to run each test in both graph and eager modes.

  def testToTensor(self):
    # [FUNCTION DOC]
    # Checks conversion of indexed slices to a dense tensor. Without a dense shape, conversion raises
    # a value error and the shape is unknown. With dense shape three by two, conversion succeeds, the
    # tensor shape matches, and the evaluated contents place the rows at indices zero and two with
    # zeros elsewhere.
    values = constant_op.constant([2, 3, 5, 7], shape=[2, 2])
    indices = constant_op.constant([0, 2])
    x = indexed_slices.IndexedSlices(values, indices)
    with self.assertRaises(ValueError):
      tensor = ops.convert_to_tensor(x, name="tensor")
    self.assertEqual(tensor_shape.TensorShape(None), x.shape)

    dense_shape = constant_op.constant([3, 2])
    y = indexed_slices.IndexedSlices(values, indices, dense_shape)
    tensor = ops.convert_to_tensor(y, name="tensor")
    self.assertAllEqual(tensor.shape, y.shape)
    self.assertAllEqual(self.evaluate(tensor), [[2, 3], [0, 0], [5, 7]])

  @test_util.run_gpu_only
  def testEagerCopy(self):
    # [FUNCTION DOC]
    # Runs only when a GPU is available. In eager mode, gathers twice from a variable and combines
    # the results with an einsum under a gradient tape, then asserts the gradient (possibly indexed
    # slices) has values of shape four by one. Guards against a copy problem with indexed-slices
    # gradients on GPU.
    with context.eager_mode():
      var = variables.Variable([[0.0], [0.0], [0.0], [0.0]], name="tensor")
      with backprop.GradientTape() as tape:
        a = array_ops.gather(array_ops.gather(var, [0, 1]), [0, 1])
        b = array_ops.gather(array_ops.gather(var, [2, 3]), [0, 1])
        r = special_math_ops.einsum("ij,ij->i", a, b)
      g = tape.gradient(r, [var])[0]
      values = g.values if isinstance(g, indexed_slices.IndexedSlices) else g
      self.assertAllEqual(values.get_shape(), [4, 1])

  def testNegation(self):
    # [FUNCTION DOC]
    # Checks that negating indexed slices negates the values and leaves the indices unchanged.
    values = constant_op.constant([2, 3, 5, 7], shape=[2, 2])
    indices = constant_op.constant([0, 2])
    x = -indexed_slices.IndexedSlices(values, indices)
    self.assertAllEqual(x.values, [[-2, -3], [-5, -7]])
    self.assertAllEqual(x.indices, [0, 2])

  def testScalarMul(self):
    # [FUNCTION DOC]
    # Checks that multiplying indexed slices by a scalar with the scalar-multiply op scales the
    # values and leaves the indices unchanged.
    values = constant_op.constant([2, 3, 5, 7], shape=[2, 2])
    indices = constant_op.constant([0, 2])
    x = math_ops.scalar_mul(-2, indexed_slices.IndexedSlices(values, indices))
    self.assertAllEqual(x.values, [[-4, -6], [-10, -14]])
    self.assertAllEqual(x.indices, [0, 2])


@test_util.run_all_in_graph_and_eager_modes
class IndexedSlicesSpecTest(test_util.TensorFlowTestCase,
                            parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the type specification of indexed slices (shape, dtypes of values, indices and
  # dense shape): construction, serialization, component specs and component conversion. Runs in
  # both graph and eager modes and also uses parameterized test cases.

  def assertAllTensorsEqual(self, list1, list2):
    # [FUNCTION DOC]
    # HELPER: Comparison helper used by the indexed-slices spec tests.
    # Asserts that two lists of tensors have equal length and that each pair of corresponding tensors
    # holds equal values. Takes the two lists; returns nothing; fails the test on any mismatch.
    self.assertLen(list1, len(list2))
    for (t1, t2) in zip(list1, list2):
      self.assertAllEqual(t1, t2)

  def testConstruction(self):
    # [FUNCTION DOC]
    # Checks the indexed-slices spec constructor. The default spec has unknown rank, float32 values,
    # int64 indices, no dense-shape dtype and an indices shape of one unknown dimension. A spec built
    # with explicit shape, dtypes and indices shape stores each of them.
    spec1 = indexed_slices.IndexedSlicesSpec()
    self.assertIsNone(spec1._shape.rank)
    self.assertEqual(spec1._values_dtype, dtypes.float32)
    self.assertEqual(spec1._indices_dtype, dtypes.int64)
    self.assertIsNone(spec1._dense_shape_dtype)
    self.assertEqual(spec1._indices_shape.as_list(), [None])

    spec2 = indexed_slices.IndexedSlicesSpec([None, None], dtypes.string,
                                             dtypes.int32, dtypes.int64, [10])
    self.assertEqual(spec2._shape.as_list(), [None, None])
    self.assertEqual(spec2._values_dtype, dtypes.string)
    self.assertEqual(spec2._indices_dtype, dtypes.int32)
    self.assertEqual(spec2._dense_shape_dtype, dtypes.int64)
    self.assertEqual(spec2._indices_shape.as_list(), [10])

  def testValueType(self):
    # [FUNCTION DOC]
    # Checks that the value type reported by a default indexed-slices spec is the indexed-slices class.
    spec1 = indexed_slices.IndexedSlicesSpec()
    self.assertEqual(spec1.value_type, indexed_slices.IndexedSlices)

  @parameterized.parameters([
      (indexed_slices.IndexedSlicesSpec(),
       (tensor_shape.TensorShape(None), dtypes.float32, dtypes.int64, None,
        tensor_shape.TensorShape([None]))),
      (indexed_slices.IndexedSlicesSpec(shape=[5, None, None]),
       (tensor_shape.TensorShape([5, None, None]), dtypes.float32,
        dtypes.int64, None, tensor_shape.TensorShape([None]))),
      (indexed_slices.IndexedSlicesSpec(
          dtype=dtypes.int32, dense_shape_dtype=dtypes.int64),
       (tensor_shape.TensorShape(None), dtypes.int32, dtypes.int64,
        dtypes.int64, tensor_shape.TensorShape([None]))),
      (indexed_slices.IndexedSlicesSpec(indices_shape=[100]),
       (tensor_shape.TensorShape(None), dtypes.float32, dtypes.int64, None,
        tensor_shape.TensorShape([100]))),
  ])  # pyformat: disable
  def testSerialize(self, spec, expected):
    # [FUNCTION DOC]
    # Parameterized test. For each (spec, expected tuple) pair asserts that the spec's serialization
    # has the same textual representation as the expected tuple (shape, values dtype, indices dtype,
    # dense-shape dtype, indices shape). Representation is compared because shape equality is
    # unconventional.
    serialization = spec._serialize()
    # TensorShape has an unconventional definition of equality, so we can't use
    # assertEqual directly here.  But repr() is deterministic and lossless for
    # the expected values, so we can use that instead.
    self.assertEqual(repr(serialization), repr(expected))

  @parameterized.parameters([
      (indexed_slices.IndexedSlicesSpec(dtype=dtypes.string), (
          tensor_lib.TensorSpec(None, dtypes.string),
          tensor_lib.TensorSpec([None], dtypes.int64),
      )),
      (indexed_slices.IndexedSlicesSpec(
          dtype=dtypes.string, dense_shape_dtype=dtypes.int32), (
              tensor_lib.TensorSpec(None, dtypes.string),
              tensor_lib.TensorSpec([None], dtypes.int64),
              tensor_lib.TensorSpec([None], dtypes.int32),
          )),
      (indexed_slices.IndexedSlicesSpec(
          shape=[5, 10, 15], dense_shape_dtype=dtypes.int32), (
              tensor_lib.TensorSpec([None, 10, 15], dtypes.float32),
              tensor_lib.TensorSpec([None], dtypes.int64),
              tensor_lib.TensorSpec([3], dtypes.int32),
          )),
      (indexed_slices.IndexedSlicesSpec(
          shape=[5, 10, 15], dense_shape_dtype=dtypes.int32,
          indices_shape=[20]), (
              tensor_lib.TensorSpec([20, 10, 15], dtypes.float32),
              tensor_lib.TensorSpec([20], dtypes.int64),
              tensor_lib.TensorSpec([3], dtypes.int32),
          )),
  ])
  def testComponentSpecs(self, spec, expected):
    # [FUNCTION DOC]
    # Parameterized test. For each (spec, expected tuple of tensor specs) pair asserts the spec's
    # component specs match, covering string values, a dense-shape component, a partly known shape
    # whose first dimension becomes unknown, and a custom indices shape.
    self.assertEqual(spec._component_specs, expected)

  @parameterized.parameters([
      {
          "spec": indexed_slices.IndexedSlicesSpec(),
          "values": [3.0, 5.0],
          "indices": [5, 10]
      },
      {
          "spec":
              indexed_slices.IndexedSlicesSpec(dense_shape_dtype=dtypes.int32),
          "values": [3.0, 5.0],
          "indices": [5, 10],
          "dense_shape": [100]
      },
  ])
  def testToFromComponents(self, spec, indices, values, dense_shape=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the indexed-slices constructor takes values first and indices second, but this test
    # passes its parameter named indices first, so the names are swapped relative to the constructor;
    # the checks are self-consistent so the test still passes.
    # Parameterized test. Inputs: spec, indices, values and optional dense shape (default none).
    # Builds indexed slices, converts them to components and asserts the components equal the inputs
    # (with dense shape only when provided), then rebuilds from components and asserts indices,
    # values and dense shape (or its absence) round-trip.
    x = indexed_slices.IndexedSlices(indices, values, dense_shape)
    actual_components = spec._to_components(x)
    if dense_shape is None:
      self.assertAllTensorsEqual(actual_components, [indices, values])
    else:
      self.assertAllTensorsEqual(actual_components,
                                 [indices, values, dense_shape])
    st_reconstructed = spec._from_components(actual_components)
    self.assertAllEqual(x.indices, st_reconstructed.indices)
    self.assertAllEqual(x.values, st_reconstructed.values)
    if dense_shape is None:
      self.assertIsNone(st_reconstructed.dense_shape)
    else:
      self.assertAllEqual(x.dense_shape, st_reconstructed.dense_shape)

  @test_util.run_v1_only("IndexedSlicesValue is deprecated in v2")
  def testFromNumpyComponents(self):
    # [FUNCTION DOC]
    # Runs only in v1 mode (the value class is deprecated in v2). Builds components from numpy arrays
    # and asserts that creating from components yields the indexed-slices value type with matching
    # indices, values and dense shape, both with and without a dense-shape dtype (dense shape is null
    # in the second case).
    indices = np.array([3, 8])
    values = np.array([1.0, 9.0])
    dense_shape = np.array([100])

    spec1 = indexed_slices.IndexedSlicesSpec(dense_shape_dtype=dtypes.int32)
    st1 = spec1._from_components((values, indices, dense_shape))
    self.assertIsInstance(st1, indexed_slices.IndexedSlicesValue)
    self.assertAllEqual(st1.indices, indices)
    self.assertAllEqual(st1.values, values)
    self.assertAllEqual(st1.dense_shape, dense_shape)

    spec2 = indexed_slices.IndexedSlicesSpec()
    st2 = spec2._from_components((values, indices))
    self.assertIsInstance(st2, indexed_slices.IndexedSlicesValue)
    self.assertAllEqual(st2.indices, indices)
    self.assertAllEqual(st2.values, values)
    self.assertIsNone(st2.dense_shape)


class NodeDefConstructorTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for building a node definition protocol buffer by the private helper.

  def testNoArgs(self):
    # [FUNCTION DOC]
    # Builds a node definition with op type and name and asserts it equals the expected text-format
    # protocol buffer with only the op and name fields.
    nodedef = ops._NodeDef("None", "bar")
    self.assertProtoEquals("op: 'None' name: 'bar'", nodedef)


def _apply_op(g, *args, **kwargs):
  # [FUNCTION DOC]
  # HELPER: Shared helper for the operation-creation tests (ApplyOpTest and others).
  # Creates an operation in graph g with create_op, passing through all further positional and
  # keyword arguments. Returns the single output tensor if the op has exactly one output, otherwise
  # the list of output tensors. Raises whatever create_op raises.
  op = g.create_op(*args, **kwargs)
  if len(op.outputs) == 1:
    return op.outputs[0]
  else:
    return op.outputs


class OperationTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for the Operation class: inputs, outputs, device, attributes, conversion to tensors,
  # control inputs, input updating and error handling. Several tests are graph-mode only.

  def testTraceback(self):
    # [FUNCTION DOC]
    # Creates an operation from a node definition and asserts that the second-to-last entry of its
    # recorded creation traceback mentions this test's name, i.e. the operation remembers where it
    # was created.
    g = ops.Graph()
    op1 = ops.Operation.from_node_def(
        ops._NodeDef("None", "op1"), g, [], [dtypes.float32_ref, dtypes.float32]
    )
    self.assertIn("testTraceback", op1.traceback[-2])

  @test_util.run_deprecated_v1
  def testNoInputs(self):
    # [FUNCTION DOC]
    # Graph-mode test. Uses an op with no inputs and two outputs (float and string). Asserts the op
    # has two values and zero inputs, the right name, output dtypes, op references, output indices,
    # no consumers and node-def input names (name and name with colon one), and that the node def
    # matches the expected text.
    op = test_ops.float_output_string_output(name="myop").a.op
    self.assertEqual(2, len(op.values()))
    self.assertEqual(0, len(op.inputs))
    self.assertEqual("myop", op.name)

    float_t, label_str_t = op.values()
    self.assertEqual(dtypes.float32, float_t.dtype)
    self.assertEqual(op, float_t.op)
    self.assertEqual(0, float_t.value_index)
    self.assertEqual(0, len(float_t.consumers()))
    self.assertEqual("myop", float_t._as_node_def_input())

    self.assertEqual(dtypes.string, label_str_t.dtype)
    self.assertEqual(op, label_str_t.op)
    self.assertEqual(1, label_str_t.value_index)
    self.assertEqual(0, len(label_str_t.consumers()))
    self.assertEqual("myop:1", label_str_t._as_node_def_input())

    self.assertProtoEquals("op:'FloatOutputStringOutput' name:'myop'",
                           op.node_def)

  @test_util.run_deprecated_v1
  def testNoOutputs(self):
    # [FUNCTION DOC]
    # Graph-mode test. Checks an op with an input but no outputs: the producer's output has one
    # consumer (the new op), the consumer has zero values and one input that is the producer's output,
    # and both node definitions match the expected text including the input reference.
    op1 = test_ops.float_output(name="myop1").op
    float_t, = op1.values()
    op2 = test_ops.float_input(float_t, name="myop2")
    self.assertEqual(0, len(op2.values()))
    self.assertEqual(1, len(op2.inputs))
    self.assertIs(float_t, op2.inputs[0])

    self.assertEqual(1, len(float_t.consumers()))
    self.assertEqual(op2, float_t.consumers()[0])

    self.assertProtoEquals("op:'FloatOutput' name:'myop1'", op1.node_def)
    self.assertProtoEquals("op:'FloatInput' name:'myop2' input:'myop1'",
                           op2.node_def)

  @test_util.run_deprecated_v1
  def testInputsAndOutputs(self):
    # [FUNCTION DOC]
    # Graph-mode test. Builds a producer with one output, a producer with two outputs, and a consumer
    # that consumes the second output twice. Asserts consumer counts per tensor (one, zero, two) and
    # the consumer node def lists inputs in order including the duplicated input.
    op1 = test_ops.float_output(name="myop1").op
    self.assertEqual(1, len(op1.values()))
    float1_t, = op1.values()

    op2 = test_ops.float_output_string_output(name="myop2").a.op
    self.assertEqual(2, len(op2.values()))
    float2_t, label2_str_t = op2.values()

    # Note that we consume label2_str_t twice here.
    op3 = test_ops.foo2(float1_t, label2_str_t, label2_str_t, name="myop3").d.op
    self.assertEqual(2, len(op3.values()))

    self.assertEqual(1, len(float1_t.consumers()))
    self.assertEqual(op3, float1_t.consumers()[0])

    self.assertEqual(0, len(float2_t.consumers()))

    self.assertEqual(2, len(label2_str_t.consumers()))
    self.assertEqual(op3, label2_str_t.consumers()[0])
    self.assertEqual(op3, label2_str_t.consumers()[1])

    self.assertProtoEquals("""
    op:'Foo2' name:'myop3'
    input:'myop1' input:'myop2:1' input:'myop2:1'
    """, op3.node_def)

  def testDeviceObject(self):
    # [FUNCTION DOC]
    # Checks setting a device on an operation, first with a device string then with a device
    # specification object, and asserts the node definition's device field shows the normalized
    # device name in both cases.
    op = ops.Operation.from_node_def(
        ops._NodeDef("None", "myop"), ops.Graph(), [], []
    )
    op._set_device("/job:goo/device:GPU:0")
    self.assertProtoEquals(
        "op:'None' name:'myop' device:'/job:goo/device:GPU:0' ", op.node_def)
    op = ops.Operation.from_node_def(
        ops._NodeDef("None", "op2"), ops.Graph(), [], []
    )
    op._set_device(
        pydev.DeviceSpec(
            job="muu", device_type="CPU", device_index=0))
    self.assertProtoEquals(
        "op:'None' name:'op2' device:'/job:muu/device:CPU:0'", op.node_def)

  def testReferenceInput(self):
    # [FUNCTION DOC]
    # Checks operations with reference-typed inputs. A first op has a reference-typed output; a
    # second op consumes it only when input types are explicitly given as reference and plain float.
    # Asserts node defs and input lists, and that a third op consuming the same tensors without
    # explicit types is also accepted with the same input names.
    g = ops.Graph()
    op1 = ops.Operation.from_node_def(
        ops._NodeDef("RefOutputFloatOutput", "op1"),
        g,
        [],
        [dtypes.float32_ref, dtypes.float32],
    )
    self.assertProtoEquals("op:'RefOutputFloatOutput' name:'op1'", op1.node_def)
    self.assertEqual([], list(op1.inputs))
    ref_t, nonref_t = op1.values()
    # NOTE(mrry): Must specify input_types to preserve ref-typed input.
    op2 = ops.Operation.from_node_def(
        ops._NodeDef("RefInputFloatInput", "op2"),
        g,
        [ref_t, nonref_t],
        [],
        input_types=[dtypes.float32_ref, dtypes.float32],
    )
    self.assertProtoEquals(
        "op:'RefInputFloatInput' name:'op2' input:'op1' input:'op1:1'",
        op2.node_def)
    self.assertEqual([ref_t, nonref_t], list(op2.inputs))
    op3 = ops.Operation.from_node_def(
        ops._NodeDef("TwoFloatInputs", "op3"), g, [ref_t, nonref_t], []
    )
    self.assertProtoEquals(
        "op:'TwoFloatInputs' name:'op3' input:'op1' input:'op1:1'",
        op3.node_def)

  def testInvalidNames(self):
    # [FUNCTION DOC]
    # Asserts that creating operations whose names are empty, start with an underscore, a dash or a
    # slash, or contain a colon index raises a value error each time.
    g = ops.Graph()
    with self.assertRaises(ValueError):
      ops.Operation.from_node_def(ops._NodeDef("op", ""), g)
    with self.assertRaises(ValueError):
      ops.Operation.from_node_def(ops._NodeDef("op", "_invalid"), g)
    with self.assertRaises(ValueError):
      ops.Operation.from_node_def(ops._NodeDef("op", "-invalid"), g)
    with self.assertRaises(ValueError):
      ops.Operation.from_node_def(ops._NodeDef("op", "/invalid"), g)
    with self.assertRaises(ValueError):
      ops.Operation.from_node_def(ops._NodeDef("op", "invalid:0"), g)

  @test_util.run_deprecated_v1
  def testNoShapeFunction(self):
    # [FUNCTION DOC]
    # Graph-mode test. Creates an op that has no registered shape function and asserts its reported
    # shape is completely unknown.
    op = test_ops.a()
    self.assertEqual(tensor_shape.unknown_shape(), op.get_shape())

  @test_util.run_in_graph_and_eager_modes
  def testConvertToTensorNestedArray(self):
    # [FUNCTION DOC]
    # Runs in both graph and eager modes. Converts a nested list of single-element lists to a tensor
    # and asserts the shape is four by one and values match.
    values = [[2], [3], [5], [7]]
    tensor = ops.convert_to_tensor(values)
    self.assertAllEqual((4, 1), tensor.get_shape().as_list())
    self.assertAllEqual(values, self.evaluate(tensor))

  def testShapeTuple(self):
    # [FUNCTION DOC]
    # Checks that a constant's private shape-tuple accessor returns the empty tuple for a scalar.
    with self.cached_session():
      c = constant_op.constant(1)
      self.assertEqual(c._shape_tuple(), ())  # pylint: disable=protected-access

  def testConvertToTensorEager(self):
    # [FUNCTION DOC]
    # Eager mode. Asserts that a constant and the result of converting it, and the result of
    # converting a Python integer, are all eager tensors.
    with context.eager_mode():
      t = constant_op.constant(1)
      self.assertTrue(isinstance(t, ops.EagerTensor))
      converted = ops.convert_to_tensor(t)
      self.assertTrue(isinstance(converted, ops.EagerTensor))
      converted = ops.convert_to_tensor(1)
      self.assertTrue(isinstance(converted, ops.EagerTensor))

  @test_util.run_in_graph_and_eager_modes
  def testConvertToTensorNestedTuple(self):
    # [FUNCTION DOC]
    # Runs in both graph and eager modes. Converts a nested tuple of one-element tuples to a tensor
    # and asserts shape four by one and equal values.
    values = ((2,), (3,), (5,), (7,))
    tensor = ops.convert_to_tensor(values)
    self.assertAllEqual((4, 1), tensor.get_shape().as_list())
    self.assertAllEqual(values, self.evaluate(ops.convert_to_tensor(values)))

  @test_util.run_in_graph_and_eager_modes
  def testConvertToTensorNestedTensors(self):
    # [FUNCTION DOC]
    # Runs in both graph and eager modes. Converts nested lists of constants (rows of tensors and
    # also lists of scalar tensors) and asserts shape four by one and equal values.
    values = ((2,), (3,), (5,), (7,))
    tensor = ops.convert_to_tensor(
        [constant_op.constant(row) for row in values])
    self.assertAllEqual((4, 1), tensor.get_shape().as_list())
    self.assertAllEqual(values, self.evaluate(tensor))
    tensor = ops.convert_to_tensor(
        [[constant_op.constant(v) for v in row] for row in values])
    self.assertAllEqual((4, 1), tensor.get_shape().as_list())
    self.assertAllEqual(values, self.evaluate(tensor))

  @test_util.run_in_graph_and_eager_modes
  def testConvertToTensorNestedMix(self):
    # [FUNCTION DOC]
    # Runs in both graph and eager modes. Converts a mixed nest of list, tuple, scalar tensor and
    # vector tensor and asserts shape four by one and the expected values.
    values = ([2], (3,), [constant_op.constant(5)], constant_op.constant([7]))
    tensor = ops.convert_to_tensor(values)
    self.assertAllEqual((4, 1), tensor.get_shape().as_list())
    self.assertAllEqual(((2,), (3,), (5,), (7,)), self.evaluate(tensor))

  @test_util.run_in_graph_and_eager_modes
  def testConvertToTensorPreferred(self):
    # [FUNCTION DOC]
    # Runs in both graph and eager modes. Checks conversion with a preferred dtype: integers convert
    # to float32 when preferred; an empty list converts to the preferred int64; a float value with an
    # int64 preference falls back to float32 instead of failing.
    values = [2, 3, 5, 7]
    tensor = ops.convert_to_tensor(values, preferred_dtype=dtypes.float32)
    self.assertEqual(dtypes.float32, tensor.dtype)

    # Convert empty tensor to anything.
    values = []
    tensor = ops.convert_to_tensor(values, preferred_dtype=dtypes.int64)
    self.assertEqual(dtypes.int64, tensor.dtype)

    # The preferred dtype is a type error and will convert to
    # float32 instead.
    values = [1.23]
    tensor = ops.convert_to_tensor(values, preferred_dtype=dtypes.int64)
    self.assertEqual(dtypes.float32, tensor.dtype)

  @test_util.run_in_graph_and_eager_modes
  def testConvertToInvalidTensorType(self):
    # [FUNCTION DOC]
    # Runs in both graph and eager modes. Asserts that forcing an incompatible dtype (float value with
    # int64) when converting raises a type error.
    with self.assertRaises(TypeError):
      # Forcing an invalid dtype should fail with a type error.
      values = [1.23]
      ops.convert_to_tensor(values, dtype=dtypes.int64)

  @test_util.run_in_graph_and_eager_modes
  def testConvertToLongLongTensorType(self):
    # [FUNCTION DOC]
    # Runs in both graph and eager modes. Converts a numpy integer scalar derived from a shape tuple
    # (a numpy long-long type) to int64 and asserts the resulting dtype is int64.
    tensor = ops.convert_to_tensor(
        # Get a numpy array of dtype NPY_LONGLONG
        np.prod(constant_op.constant([1])._shape_tuple()),
        dtype=dtypes.int64)
    self.assertEqual(dtypes.int64, tensor.dtype)

  @test_util.run_in_graph_and_eager_modes
  def testConvertToTensorFromValidTensor(self):
    # [FUNCTION DOC]
    # Runs in both graph and eager modes. Converting an int64 tensor with a compatible requested dtype
    # returns an equal tensor (the same instance).
    tensor = constant_op.constant(413, dtype=dtypes.int64)
    converted = ops.convert_to_tensor(tensor, dtype=dtypes.int64)
    # If dtype is compatible, the returned tensor should be the same instance.
    self.assertEqual(tensor, converted)

  @test_util.run_in_graph_and_eager_modes
  def testConvertToTensorFromInvalidTensor(self):
    # [FUNCTION DOC]
    # Runs in both graph and eager modes. Asserts that converting a float32 tensor while requesting
    # int32 raises a value error.
    tensor = constant_op.constant(42.0, dtype=dtypes.float32)
    with self.assertRaises(ValueError):
      ops.convert_to_tensor(tensor, dtype=dtypes.int32)

  @test_util.run_in_graph_and_eager_modes
  def testConvertToTensorProtocol(self):
    # [FUNCTION DOC]
    # Runs in both graph and eager modes. Defines a local class implementing the tensor-conversion
    # protocol (returns a constant of one, two, three with the given dtype and name), converts an
    # instance requesting int32, and asserts the dtype and values.
    class TensorCompatible:
      # [CLASS DOC]
      # HELPER: Local class used by the tensor-protocol conversion test.
      # Plain class that implements the tensor-conversion protocol method so that the conversion
      # function can turn instances into tensors.

      def __tf_tensor__(self, dtype=None, name=None):
        # [FUNCTION DOC]
        # HELPER: Conversion protocol hook used by the conversion test.
        # Takes an optional dtype and optional name and returns a constant tensor with values one, two,
        # three created with those arguments.
        return constant_op.constant((1, 2, 3), dtype=dtype, name=name)

    tc = TensorCompatible()

    tensor = ops.convert_to_tensor(tc, dtype=dtypes.int32)
    self.assertEqual(tensor.dtype, dtypes.int32)
    self.assertAllEqual((1, 2, 3), self.evaluate(tensor))

  @test_util.run_deprecated_v1
  def testNoConvert(self):
    # [FUNCTION DOC]
    # Graph-mode test. Asserts that converting an operation (not a tensor) with the tensor conversion
    # function raises a type error saying the operation cannot be converted to a tensor.
    # Operation cannot be converted to Tensor.
    op = gen_control_flow_ops.no_op()
    with self.assertRaisesRegex(TypeError,
                                "can't convert Operation '.+' to Tensor"):
      ops.convert_to_tensor(op)

  def testStr(self):
    # [FUNCTION DOC]
    # Asserts that the string form of an operation equals the string form of its node definition.
    node_def = ops._NodeDef("None", "op1")
    op = ops.Operation.from_node_def(
        node_def, ops.Graph(), [], [dtypes.float32]
    )
    self.assertEqual(str(node_def), str(op))

  def testRepr(self):
    # [FUNCTION DOC]
    # Asserts the repr of an operation named op1 of type None is the angle-bracket form showing the
    # name and type.
    op = ops.Operation.from_node_def(
        ops._NodeDef("None", "op1"), ops.Graph(), [], [dtypes.float32]
    )
    self.assertEqual("<tf.Operation 'op1' type=None>", repr(op))

  @test_util.run_deprecated_v1
  def testGetAttr(self):
    # [FUNCTION DOC]
    # Graph-mode test. Reads every attribute type from an op with default attributes (string, string
    # list, int, int list, float, float list, bool, bool list, shape, shape list, tensor, tensor list,
    # type, type list) and asserts the defaults; checks type attributes are dtype objects. Defines a
    # function and checks the function attribute returns a name-attribute list with the function name.
    # Asserts that reading a missing attribute raises a value error saying the op has no such attr.
    op = test_ops.default_attrs()
    self.assertEqual(op.get_attr("string_val"), b"abc")
    self.assertEqual(op.get_attr("string_list_val"), [b"abc", b""])
    self.assertEqual(op.get_attr("int_val"), 123)
    self.assertEqual(op.get_attr("int_list_val"), [1, 2, 3])
    self.assertEqual(op.get_attr("float_val"), 10.0)
    self.assertEqual(op.get_attr("float_list_val"), [10.0])
    self.assertEqual(op.get_attr("bool_val"), True)
    self.assertEqual(op.get_attr("bool_list_val"), [True, False])
    self.assertEqual(op.get_attr("shape_val"),
                     tensor_shape.as_shape([2, 1]).as_proto())
    self.assertEqual(op.get_attr("shape_list_val"),
                     [tensor_shape.as_shape([]).as_proto(),
                      tensor_shape.as_shape([1]).as_proto()])
    self.assertEqual(op.get_attr("tensor_val"),
                     tensor_util.make_tensor_proto(1, dtypes.int32))
    self.assertEqual(op.get_attr("tensor_list_val"),
                     [tensor_util.make_tensor_proto(1, dtypes.int32)])

    type_val = op.get_attr("type_val")
    # First check that type_val is a DType, because the assertEqual will work
    # no matter what since DType overrides __eq__
    self.assertIsInstance(type_val, dtypes.DType)
    self.assertEqual(type_val, dtypes.int32)

    type_list_val = op.get_attr("type_list_val")
    self.assertTrue(all(isinstance(x, dtypes.DType) for x in type_list_val))
    self.assertEqual(type_list_val, [dtypes.int32, dtypes.float32])

    @function.Defun(dtypes.float32, func_name="MyFunc")
    def func(x):
      # [FUNCTION DOC]
      # HELPER: Function used by the attribute test.
      # Decorated with the legacy function-definition decorator (name MyFunc, one float32 argument);
      # returns its argument unchanged. Serves as the value of a function-typed attribute.
      return x

    op = test_ops.func_attr(func)
    self.assertEqual(op.get_attr("f"),
                     attr_value_pb2.NameAttrList(name="MyFunc"))

    # Try fetching missing attr
    with self.assertRaisesRegex(
        ValueError, "Operation 'FuncAttr' has no attr named 'FakeAttr'."):
      op.get_attr("FakeAttr")

  # TODO(b/65162920): remove this test when users who are directly mutating the
  # node_def have been updated to proper usage.
  @test_util.run_deprecated_v1
  def testSetAttr(self):
    # [FUNCTION DOC]
    # Graph-mode test. Sets an integer attribute on an operation through the private setter and
    # asserts reading it back returns the value two. A to-do comment notes this test should go away
    # once callers stop mutating the node definition directly.
    op = test_ops.int_attr().op
    op._set_attr("foo", attr_value_pb2.AttrValue(i=2))
    # TODO(skyewm): add node_def check
    self.assertEqual(op.get_attr("foo"), 2)

  @test_util.run_v2_only
  def testSetFullType(self):
    # [FUNCTION DOC]
    # Runs in v2 only. Inside a traced function, creates a dataset variant tensor, sets an
    # experimental full type (product) on its op, and asserts the op's node definition now reports
    # that type id. The traced function is called once at the end.
    @def_function.function
    def test_fn():
      # [FUNCTION DOC]
      # HELPER: Traced function body for the full-type test.
      # Takes no arguments, returns nothing; creates a dataset of range three, sets the full type
      # product on its op, and asserts the node definition shows the product type.
      ds = dataset_ops.Dataset.range(3)._variant_tensor

      ds.op.experimental_set_type(
          full_type_pb2.FullTypeDef(type_id=full_type_pb2.TFT_PRODUCT))

      self.assertEqual(ds.op.node_def.experimental_type.type_id,
                       full_type_pb2.TFT_PRODUCT)

    test_fn()

  # TODO(nolivia): test all error cases
  def testAddControlInput(self):
    # [FUNCTION DOC]
    # Checks adding control inputs: adding the same control input twice yields only one, adding a list
    # with duplicates adds only the new unique ones in order, and the source op records the dependent
    # op in its control outputs.
    with ops.Graph().as_default():
      x = constant_op.constant(1).op
      y = constant_op.constant(2).op
      z = constant_op.constant(3).op
    z._add_control_input(x)  # pylint: disable=protected-access
    self.assertEqual(z.control_inputs, [x])
    z._add_control_input(x)  # pylint: disable=protected-access
    self.assertEqual(z.control_inputs, [x])
    z._add_control_inputs([x, y, y])  # pylint: disable=protected-access
    self.assertEqual(z.control_inputs, [x, y])
    self.assertEqual(x._control_outputs, [z])

  @test_util.run_deprecated_v1
  def testRemoveAllControlInputs(self):
    # [FUNCTION DOC]
    # Graph-mode test. Builds ops with and without control dependencies, asserts their control
    # inputs, then removes all control inputs from each and asserts they are empty while the data
    # inputs of the sum op remain unchanged.
    a = constant_op.constant(1)
    with ops.control_dependencies([a]):
      b = constant_op.constant(2)
    c = constant_op.constant(3)
    d = constant_op.constant(4)
    e = constant_op.constant(5)
    with ops.control_dependencies([a, c]):
      f = d + e

    self.assertEqual(a.op.control_inputs, [])
    self.assertEqual(b.op.control_inputs, [a.op])
    self.assertEqual(f.op.control_inputs, [a.op, c.op])

    a.op._remove_all_control_inputs()  # pylint: disable=protected-access
    self.assertEqual(a.op.control_inputs, [])

    b.op._remove_all_control_inputs()  # pylint: disable=protected-access
    self.assertEqual(b.op.control_inputs, [])

    f.op._remove_all_control_inputs()  # pylint: disable=protected-access
    self.assertEqual(f.op.control_inputs, [])
    self.assertEqual(list(f.op.inputs), [d, e])

  @test_util.run_deprecated_v1
  def testControlInputCycle(self):
    # [FUNCTION DOC]
    # Graph-mode test. Creates a control-dependency cycle between two ops through the private add
    # function, then evaluating one of them is expected to raise an invalid-argument error saying the
    # graph contains a cycle with two nodes.
    graph = ops.Graph()
    with graph.as_default():
      z = constant_op.constant(0)
      x = constant_op.constant(1)
      y = constant_op.constant(2)
      y.op._add_control_input(z.op)  # pylint: disable=protected-access
      y.op._add_control_input(x.op)  # pylint: disable=protected-access
      x.op._add_control_input(y.op)  # pylint: disable=protected-access
    with self.session(graph=graph) as sess:
      with self.assertRaisesRegex(
          errors.InvalidArgumentError,
          "Graph is invalid, contains a cycle with 2 nodes"):
        self.evaluate(x)

  def testUpdateInput(self):
    # [FUNCTION DOC]
    # Checks replacing an op input by index through the private update function: after each update the
    # inputs, consumer lists and the evaluated result are asserted (sum becomes four, then three).
    # Updating index one to the same value it already has changes nothing.
    g = ops.Graph()
    with g.as_default():
      x = constant_op.constant(1)
      y = constant_op.constant(2)
      z = x + y

    z.op._update_input(0, y)  # pylint: disable=protected-access
    self.assertEqual(list(z.op.inputs), [y, y])
    self.assertEqual(x.consumers(), [])
    self.assertEqual(y.consumers(), [z.op, z.op])
    with session.Session(graph=g) as sess:
      self.assertEqual(self.evaluate(z), 4)

    z.op._update_input(0, x)  # pylint: disable=protected-access
    self.assertEqual(list(z.op.inputs), [x, y])
    self.assertEqual(x.consumers(), [z.op])
    self.assertEqual(y.consumers(), [z.op])
    with session.Session(graph=g) as sess:
      self.assertEqual(self.evaluate(z), 3)

    z.op._update_input(1, y)  # pylint: disable=protected-access
    self.assertEqual(list(z.op.inputs), [x, y])
    self.assertEqual(x.consumers(), [z.op])
    self.assertEqual(y.consumers(), [z.op])
    with session.Session(graph=g) as sess:
      self.assertEqual(self.evaluate(z), 3)

  def testUpdateInputGraphError(self):
    # [FUNCTION DOC]
    # Asserts that updating an input of an op in one graph with a tensor from another graph raises a
    # value error saying they must come from the same graph.
    g_0 = ops.Graph()
    g_1 = ops.Graph()
    with g_0.as_default():
      x = constant_op.constant(1)
    with g_1.as_default():
      y = constant_op.constant(2)
      z = y * 2
      with self.assertRaisesRegex(ValueError, "must be from the same graph"):
        z.op._update_input(0, x)  # pylint: disable=protected-access

  def testUpdateInputTypeError(self):
    # [FUNCTION DOC]
    # Updates an int add op's input to a string tensor, then evaluating raises an invalid-argument
    # error saying input zero was passed a string incompatible with expected int32.
    g = ops.Graph()
    with g.as_default():
      w = constant_op.constant(0)
      x = constant_op.constant("")
      y = constant_op.constant(1)
      z = y + w
      z.op._update_input(0, x)  # pylint: disable=protected-access
    with session.Session(graph=g) as sess:
      with self.assertRaisesRegex(
          errors.InvalidArgumentError,
          "Input 0 of node add was passed string from Const_1:0 incompatible "
          "with expected int32"):
        self.evaluate(z)

  def testUpdateInputShapeError(self):
    # [FUNCTION DOC]
    # Asserts that updating an op's input with a tensor of an incompatible static shape raises an
    # invalid-argument error saying the edge cannot be updated due to incompatible shapes.
    g = ops.Graph()
    with g.as_default():
      w = constant_op.constant(2, shape=[3, 1])
      x = constant_op.constant(0, shape=[3, 1])
      y = constant_op.constant(1, shape=[2, 2])
      z = w + x
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        r"Cannot update edge, incompatible shapes: \[2,2\] and \[3,1\]"):
      z.op._update_input(0, y)  # pylint: disable=protected-access

  def testUpdateInputOutOfRange(self):
    # [FUNCTION DOC]
    # Asserts that updating an input index greater than the number of inputs raises an out-of-range
    # error describing the index and total inputs.
    g = ops.Graph()
    with g.as_default():
      x = constant_op.constant(1)
    with self.assertRaisesRegex(
        errors.OutOfRangeError,
        r"Cannot update edge. Input index \[1\] is greater than the number of "
        r"total inputs \[0\]."):
      x.op._update_input(1, x)  # pylint: disable=protected-access

  @test_util.enable_control_flow_v2
  @test_util.run_v1_only("b/120545219")
  def testAddWhileInput(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the recursive call to the traced inner function is indented inside that inner
    # function, and the inner function is never called from the test body, so none of the assertions
    # ever run; the test effectively does nothing.
    # Runs only in v1 with control flow v2 enabled. Checks adding extra inputs to a while op.
    # Intended behavior: in the inner function build a while loop, give it a control input, clear its
    # output shapes, extend the type list, add two while inputs, assert adding beyond the type list
    # raises an out-of-range error, and assert two more inputs.

    @def_function.function
    def test():
      # [FUNCTION DOC]
      # HELPER: Traced function body for the add-while-input test.
      # KNOWN ISSUE: it ends by calling itself, which would recurse forever if it were ever called; it
      # is never called.
      # Builds a while loop, checks the op type is stateless while, adds a control input, widens the
      # type attribute, adds extra while inputs, and asserts input counts.
      output = while_loop.while_loop(lambda x: x < 3, lambda x: x + 1, [1])
      while_op = output.op
      self.assertEqual(while_op.type, "StatelessWhile")
      orig_num_inputs = len(while_op.inputs)

      # Make sure we can handle the while op having a control input.
      while_op._add_control_input(constant_op.constant(0).op)

      new_input1 = constant_op.constant(1.0)
      new_input2 = constant_op.constant(True)

      # Clear output shapes to bypass shape checking.
      while_op._set_shape_list_attr("output_shapes", [])
      while_op._set_type_list_attr("T", [t.dtype for t in while_op.inputs] +
                                   [new_input1.dtype, new_input2.dtype])

      while_op._add_while_inputs([new_input1, new_input2])
      # Can't add an edge beyond what's specified by "T"
      with self.assertRaises(errors.OutOfRangeError):
        while_op._add_while_inputs([new_input2])
      self.assertLen(while_op.inputs, orig_num_inputs + 2)  # pylint: disable=g-deprecated-assert

      test()

  @test_util.run_deprecated_v1
  def testOpDef(self):
    # [FUNCTION DOC]
    # Graph-mode test. Checks op definition metadata: a constant's op def name is the constant op with
    # zero inputs and one output; an add op's def name matches add or add-v2 with two inputs and one
    # output.
    x = constant_op.constant(0)
    y = constant_op.constant(1)
    z = x + y

    self.assertEqual(x.op.op_def.name, "Const")
    self.assertLen(x.op.op_def.input_arg, 0)
    self.assertLen(x.op.op_def.output_arg, 1)

    self.assertRegex(z.op.op_def.name, "Add(V2)?")
    self.assertLen(z.op.op_def.input_arg, 2)
    self.assertLen(z.op.op_def.output_arg, 1)

  def testInputFromDifferentGraphError(self):
    # [FUNCTION DOC]
    # Asserts that multiplying tensors from two different graphs raises a value error saying they
    # must be from the same graph.
    g_0 = ops.Graph()
    g_1 = ops.Graph()
    with g_0.as_default():
      x = constant_op.constant(1)
    with g_1.as_default():
      y = constant_op.constant(2)
      with self.assertRaisesRegex(ValueError, "must be from the same graph"):
        y * x  # pylint: disable=pointless-statement

  def testInputsAreImmutable(self):
    # [FUNCTION DOC]
    # Asserts that the inputs sequence of an operation is immutable: appending to it raises an
    # attribute error saying a tuple has no append.
    g = ops.Graph()
    with g.as_default():
      x = test_ops.int_output()
      op = test_ops.int_input_int_output(x, name="myop").op
    with self.assertRaisesRegex(AttributeError,
                                "'tuple' object has no attribute 'append'"):
      op.inputs.append(None)


class CreateOpTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for creating operations directly with the graph's create-op method.

  def testNodeDefArgs(self):
    # [FUNCTION DOC]
    # Creates ops through create_op with and without a device scope, with several inputs and output
    # types. Asserts devices (none, GPU zero, none) and the node definitions of each, including the
    # input name references.
    g = ops.Graph()
    op1 = g.create_op("FloatOutput", [], [dtypes.float32], None, name="myop1")
    with g.device("/device:GPU:0"):
      op2 = g.create_op(
          "FloatOutputStringOutput", [], [dtypes.float32, dtypes.string], None,
          name="myop2")
    op3 = g.create_op(
        "Foo3",
        [list(op1.values())[0], list(op2.values())[1], list(op2.values())[0]],
        [dtypes.float32, dtypes.int32],
        None,
        name="myop3")
    self.assertDeviceEqual(None, op1.device)
    self.assertDeviceEqual("/device:GPU:0", op2.device)
    self.assertDeviceEqual(None, op3.device)
    self.assertProtoEquals("name:'myop1' op:'FloatOutput'", op1.node_def)
    self.assertProtoEquals(
        "name:'myop2' op:'FloatOutputStringOutput' device:'/device:GPU:0'",
        op2.node_def)
    self.assertProtoEquals(
        "name:'myop3' input:'myop1' input:'myop2:1' input:'myop2' op:'Foo3'",
        op3.node_def)

  def testReferenceInput(self):
    # [FUNCTION DOC]
    # Checks reference-typed inputs with create_op: an op with a reference output, then an op that
    # consumes it with explicit input types, then another op with default types. Asserts all node
    # definitions.
    g = ops.Graph()
    op1 = g.create_op(
        "RefOutputFloatOutput", [], [dtypes.float32_ref, dtypes.float32],
        name="op1")
    self.assertProtoEquals("op:'RefOutputFloatOutput' name:'op1'", op1.node_def)
    ref_t, nonref_t = op1.values()
    # NOTE(mrry): Must specify input_types to preserve ref-typed input.
    op2 = g.create_op(
        "RefInputFloatInput", [ref_t, nonref_t], [],
        input_types=[dtypes.float32_ref, dtypes.float32],
        name="op2")
    self.assertProtoEquals(
        "op:'RefInputFloatInput' name:'op2' input:'op1' input:'op1:1'",
        op2.node_def)
    op3 = g.create_op("TwoFloatInputs", [ref_t, nonref_t], [], name="op3")
    self.assertProtoEquals(
        "op:'TwoFloatInputs' name:'op3' input:'op1' input:'op1:1'",
        op3.node_def)

  def testFinalized(self):
    # [FUNCTION DOC]
    # Asserts that creating an op in a finalized graph raises a runtime error, and that after unsafely
    # unfinalizing the graph, op creation works again.
    g = ops.Graph()
    g.finalize()
    with self.assertRaises(RuntimeError):
      g.create_op("FloatOutput", [], [dtypes.float32], None, name="myop1")

    # Test unfinalize.
    g._unsafe_unfinalize()
    g.create_op("FloatOutput", [], [dtypes.float32], None, name="myop1")


# NOTE(skyewm): these cases test the private Graph._create_op_from_tf_operation
# method. Arguably we should only test the public APIs that depend on this
# method. However, this logic is complex and tricky, and it can be difficult to
# ascertain if we have adequate coverage (e.g. a graph may run successfully if
# the control flow context isn't set properly, but a more complicated use case
# that might not be obvious to test will fail). Thus we instead explicitly test
# the low-level behavior.
class CreateOpFromTFOperationTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for the private graph method that wraps an already created native operation into a
  # Python operation. Covers naming, shapes, control flow context and control dependencies.

  @test_util.run_deprecated_v1
  def testBasic(self):
    # [FUNCTION DOC]
    # Graph-mode test. Creates a native op with one input through the private helper, then wraps it.
    # Asserts name, type, one output of unknown shape, inputs, no control inputs, owning graph,
    # consumer registration, non-null traceback that mentions the test name, and lookup by name for
    # both the operation and its output tensor.
    g = ops.Graph()
    with g.as_default():
      x = test_ops.int_output()
      c_op = ops._create_c_op(
          g, ops._NodeDef("IntInputIntOutput", "myop"), [x], [])
      op = g._create_op_from_tf_operation(c_op)

    self.assertEqual(op.name, "myop")
    self.assertEqual(op.type, "IntInputIntOutput")
    self.assertLen(op.outputs, 1)
    self.assertEqual(op.outputs[0].shape, tensor_shape.unknown_shape())
    self.assertEqual(list(op.inputs), [x])
    self.assertEqual(op.control_inputs, [])
    self.assertEqual(op.graph, g)
    self.assertEqual(x.consumers(), [op])
    self.assertIsNotNone(op.traceback)
    self.assertIn("testBasic", op.traceback[-1])
    self.assertEqual(g.get_operation_by_name("myop"), op)
    self.assertEqual(g.get_tensor_by_name("myop:0"), op.outputs[0])

  def testShape(self):
    # [FUNCTION DOC]
    # Wraps a native identity op over a two-by-three constant and asserts name, type, a single output
    # and that the output shape is inferred as two by three.
    g = ops.Graph()
    with g.as_default():
      x = constant_op.constant([[1, 2, 3], [4, 5, 6]])
      c_op = ops._create_c_op(g, ops._NodeDef("Identity", "myop"), [x], [])
      op = g._create_op_from_tf_operation(c_op)

    self.assertEqual(op.name, "myop")
    self.assertEqual(op.type, "Identity")
    self.assertLen(op.outputs, 1)
    self.assertEqual(op.outputs[0].shape, tensor_shape.TensorShape([2, 3]))

  def testUniqueName(self):
    # [FUNCTION DOC]
    # Creates two native ops named myop and myop_1, wraps them, then creates two more through the
    # normal API with the same names. Asserts the new names are made unique (myop_2 and myop_1_1)
    # while the wrapped ones keep their names.
    g = ops.Graph()
    with g.as_default():
      c_op = ops._create_c_op(g, ops._NodeDef("IntOutput", "myop"), [], [])
      c_op2 = ops._create_c_op(g, ops._NodeDef("IntOutput", "myop_1"), [], [])
      op = g._create_op_from_tf_operation(c_op)
      op2 = g._create_op_from_tf_operation(c_op2)

      # Create ops with same names as op1 and op2. We expect the new names to be
      # uniquified.
      op3 = test_ops.int_output(name="myop").op
      op4 = test_ops.int_output(name="myop_1").op

    self.assertEqual(op.name, "myop")
    self.assertEqual(op2.name, "myop_1")
    self.assertEqual(op3.name, "myop_2")
    self.assertEqual(op4.name, "myop_1_1")

  @test_util.run_v1_only("b/120545219")
  def testCond(self):
    # [FUNCTION DOC]
    # Runs in v1 only. Inside a conditional branch function, creates a native op and asks the graph to
    # add the new operations, asserting exactly one was added. Afterwards asserts the op is named in
    # the conditional scope, its input is a switch op fed by the original tensor, and its control flow
    # context is the conditional context.
    g = ops.Graph()
    with g.as_default():
      x = test_ops.int_output()

      def true_fn():
        # [FUNCTION DOC]
        # HELPER: Branch function for the conditional test.
        # Creates a native op with the conditional-scope name in the default graph, asserts the graph
        # reports exactly one new operation, and returns the captured outer tensor.
        ops._create_c_op(ops.get_default_graph(),
                         ops._NodeDef("IntInput", "cond/myop"), [x], [])
        new_ops = g._add_new_tf_operations()
        self.assertLen(new_ops, 1)
        return x

      cond.cond(x < 10, true_fn, lambda: x)

    op = g.get_operation_by_name("cond/myop")
    self.assertIsNotNone(op)
    self.assertEqual(op.name, "cond/myop")
    self.assertEqual(op.type, "IntInput")
    self.assertEqual(op.outputs, [])
    op_input = op.inputs[0].op
    self.assertEqual(op_input.type, "Switch")
    self.assertEqual(op_input.inputs[0], x)
    self.assertEqual(op.graph, g)
    # pylint: disable=protected-access
    self.assertIsNotNone(op._get_control_flow_context())
    self.assertEqual(op._get_control_flow_context().name,
                     "cond/cond_text")
    # pylint: enable=protected-access

  @test_util.run_v1_only("b/120545219")
  def testWhileLoop(self):
    # [FUNCTION DOC]
    # Runs in v1 only. Same idea as the conditional test but within a while loop body: asserts the new
    # op sits under the loop name, its input is an enter op fed by the outer tensor, and its control
    # flow context is the while context.
    g = ops.Graph()
    with g.as_default():
      x = test_ops.int_output()

      def body(i):
        # [FUNCTION DOC]
        # HELPER: Loop body for the while-loop test.
        # Takes the loop variable, creates a native op under the loop name scope, asserts one new
        # operation was added to the graph, and returns the loop variable unchanged.
        ops._create_c_op(ops.get_default_graph(),
                         ops._NodeDef("IntInput", "myloop/myop"), [x], [])
        new_ops = g._add_new_tf_operations()
        self.assertLen(new_ops, 1)
        return i

      while_loop.while_loop(lambda i: i < 10, body, [0], name="myloop")

    op = g.get_operation_by_name("myloop/myop")
    self.assertIsNotNone(op)
    self.assertEqual(op.name, "myloop/myop")
    self.assertEqual(op.type, "IntInput")
    self.assertEqual(op.outputs, [])
    op_input = op.inputs[0].op
    self.assertEqual(op_input.type, "Enter")
    self.assertEqual(list(op_input.inputs), [x])
    self.assertEqual(op.graph, g)
    # pylint: disable=protected-access
    self.assertIsNotNone(op._get_control_flow_context())
    self.assertEqual(op._get_control_flow_context().name,
                     "myloop/while_context")
    # pylint: enable=protected-access

  @test_util.run_v1_only("b/120545219")
  def testWhileLoopWithInternalControlDep(self):
    # [FUNCTION DOC]
    # Runs in v1 only. Creates a native op inside a while body after creating an internal constant,
    # adds the new ops under a control dependency on that constant, and asserts the new op keeps that
    # internal control dependency.
    g = ops.Graph()
    with g.as_default():
      x = test_ops.int_output()

      def body(i):
        # [FUNCTION DOC]
        # HELPER: Loop body for the internal-control-dependency test.
        # Takes the loop variable; creates a constant, creates a native op, then under a control
        # dependency on the constant asserts one new operation was added; returns the loop variable.
        c = constant_op.constant(1.0, name="c")
        ops._create_c_op(ops.get_default_graph(),
                         ops._NodeDef("IntInput", "myloop/myop"), [x], [])
        with ops.control_dependencies([c]):
          new_ops = g._add_new_tf_operations()
          self.assertLen(new_ops, 1)
        return i

      while_loop.while_loop(lambda i: i < 10, body, [0], name="myloop")

    op = g.get_operation_by_name("myloop/myop")
    self.assertIsNotNone(op)
    c = g.get_operation_by_name("myloop/c")
    self.assertIsNotNone(c)
    # Internal control dep is preserved
    self.assertEqual(op.control_inputs, [c])

  @test_util.run_v1_only("b/120545219")
  def testWhileLoopWithExternalControlDep(self):
    # [FUNCTION DOC]
    # Runs in v1 only. Adds a native op in a while body under a control dependency on an external
    # constant created outside the loop. Asserts the external dependency is replaced by an internal
    # control dependency (the control input is not the outside op) that has a control flow context.
    g = ops.Graph()
    with g.as_default():
      x = test_ops.int_output()
      c = constant_op.constant(1.0)

      def body(i):
        # [FUNCTION DOC]
        # HELPER: Loop body for the external-control-dependency test.
        # Takes the loop variable; creates a native op, then under a control dependency on the outer
        # constant asserts one new operation was added; returns the loop variable.
        ops._create_c_op(ops.get_default_graph(),
                         ops._NodeDef("IntInput", "myloop/myop"), [x], [])
        with ops.control_dependencies([c]):
          new_ops = g._add_new_tf_operations()
          self.assertLen(new_ops, 1)
        return i

      while_loop.while_loop(lambda i: i < 10, body, [0], name="myloop")

    op = g.get_operation_by_name("myloop/myop")
    self.assertIsNotNone(op)
    # External control dep is removed and replaced with internal control dep
    self.assertNotEqual(op.control_inputs[0], c.op)
    self.assertIsNotNone(op.control_inputs[0]._get_control_flow_context())


class ApplyOpTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for applying operations through the shared helper that returns tensors rather than ops.

  def testNodeDefArgs(self):
    # [FUNCTION DOC]
    # Uses the helper to create ops with and without a device scope and with multiple outputs.
    # Asserts the return types (tensor or list of tensors), node-def input names for outputs (name,
    # name with colon one), and the node definitions including the device.
    g = ops.Graph()
    t1 = _apply_op(g, "FloatOutput", [], [dtypes.float32], name="myop1")
    with g.device("/device:GPU:0"):
      t2 = _apply_op(
          g, "TwoIntOutputs", [], [dtypes.int32, dtypes.int32], name="myop2")
    t3 = _apply_op(
        g,
        "Foo1", [t1, t2[1], t2[0]], [dtypes.float32, dtypes.int32],
        name="myop3")
    self.assertTrue(isinstance(t1, tensor_lib.Tensor))
    self.assertTrue(isinstance(t2, list))
    self.assertTrue(isinstance(t3, list))
    self.assertTrue(isinstance(t3[0], tensor_lib.Tensor))
    self.assertEqual("myop1", t1._as_node_def_input())
    self.assertEqual("myop2", t2[0]._as_node_def_input())
    self.assertEqual("myop2:1", t2[1]._as_node_def_input())
    self.assertEqual("myop3", t3[0]._as_node_def_input())
    # Validate that we got the right ops as well
    self.assertProtoEquals("name:'myop1' op:'FloatOutput'", t1.op.node_def)
    self.assertProtoEquals(
        "name:'myop2' op:'TwoIntOutputs' device:'/device:GPU:0'",
        t2[0].op.node_def)
    self.assertProtoEquals(
        "name:'myop3' input:'myop1' input:'myop2:1' input:'myop2' op:'Foo1'",
        t3[0].op.node_def)

  def testReferenceInput(self):
    # [FUNCTION DOC]
    # Uses the helper for reference-typed inputs: a reference-output op, then ops consuming its outputs
    # with explicit input types and with defaults, and asserts the node definitions.
    g = ops.Graph()
    ref_t, nonref_t = _apply_op(
        g, "RefOutputFloatOutput", [], [dtypes.float32_ref, dtypes.float32],
        name="op1")
    self.assertProtoEquals("op:'RefOutputFloatOutput' name:'op1'",
                           ref_t.op.node_def)
    # NOTE(mrry): Must specify input_types to preserve ref-typed input.
    out_2 = _apply_op(
        g,
        "RefInputFloatInputIntOutput", [ref_t, nonref_t], [dtypes.int32],
        input_types=[dtypes.float32_ref, dtypes.float32],
        name="op2")
    self.assertProtoEquals(
        "op:'RefInputFloatInputIntOutput' name:'op2' input:'op1' input:'op1:1'",
        out_2.op.node_def)
    out_3 = _apply_op(
        g, "TwoFloatInputsIntOutput", [ref_t, nonref_t], [dtypes.int32],
        name="op3")
    self.assertProtoEquals(
        "op:'TwoFloatInputsIntOutput' name:'op3' input:'op1' input:'op1:1'",
        out_3.op.node_def)


class NameStackTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for unique name generation in a graph and for name scopes.

  def testBasics(self):
    # [FUNCTION DOC]
    # Exhaustively checks the graph's unique-name method with and without marking names used,
    # including suffix chains, nested name scopes, scopes entered more than once, clearing the scope
    # with null, and names reused after scopes close. Asserts every returned name.
    g = ops.Graph()
    self.assertEqual("foo", g.unique_name("foo", mark_as_used=False))
    self.assertEqual("foo", g.unique_name("foo", mark_as_used=False))
    self.assertEqual("foo", g.unique_name("foo"))
    self.assertEqual("foo_1", g.unique_name("foo", mark_as_used=False))
    self.assertEqual("foo_1", g.unique_name("foo"))
    self.assertEqual("foo_2", g.unique_name("foo", mark_as_used=False))
    self.assertEqual("foo_2", g.unique_name("foo"))
    self.assertEqual("foo_1_1", g.unique_name("foo_1", mark_as_used=False))
    self.assertEqual("foo_1_1", g.unique_name("foo_1"))
    self.assertEqual("foo_1_2", g.unique_name("foo_1", mark_as_used=False))
    self.assertEqual("foo_1_2", g.unique_name("foo_1"))
    self.assertEqual("foo_1_2_1", g.unique_name("foo_1_2", mark_as_used=False))
    self.assertEqual("foo_1_2_1", g.unique_name("foo_1_2"))
    with g.name_scope("bar"):
      self.assertEqual("bar/foo", g.unique_name("foo", mark_as_used=False))
      self.assertEqual("bar/foo", g.unique_name("foo"))
      self.assertEqual("bar/foo_1", g.unique_name("foo", mark_as_used=False))
      self.assertEqual("bar/foo_1", g.unique_name("foo"))
      with g.name_scope(None):
        self.assertEqual("foo_3", g.unique_name("foo", mark_as_used=False))
        self.assertEqual("foo_3", g.unique_name("foo"))
      with g.name_scope("baz"):
        self.assertEqual(
            "bar/baz/foo", g.unique_name(
                "foo", mark_as_used=False))
        self.assertEqual("bar/baz/foo", g.unique_name("foo"))
        self.assertEqual(
            "bar/baz/foo_1", g.unique_name(
                "foo", mark_as_used=False))
        self.assertEqual("bar/baz/foo_1", g.unique_name("foo"))
      with g.name_scope("baz"):
        self.assertEqual(
            "bar/baz_1/foo", g.unique_name(
                "foo", mark_as_used=False))
        self.assertEqual("bar/baz_1/foo", g.unique_name("foo"))
        self.assertEqual(
            "bar/baz_1/foo_1", g.unique_name(
                "foo", mark_as_used=False))
        self.assertEqual("bar/baz_1/foo_1", g.unique_name("foo"))
    with g.name_scope("quux"):
      self.assertEqual("quux/foo", g.unique_name("foo", mark_as_used=False))
      self.assertEqual("quux/foo", g.unique_name("foo"))
    with g.name_scope("bar"):
      with g.name_scope("baz"):
        self.assertEqual(
            "bar_1/baz/foo", g.unique_name(
                "foo", mark_as_used=False))
        self.assertEqual("bar_1/baz/foo", g.unique_name("foo"))
    self.assertEqual("foo_4", g.unique_name("foo", mark_as_used=False))
    self.assertEqual("foo_4", g.unique_name("foo"))
    self.assertEqual("bar_2", g.unique_name("bar", mark_as_used=False))
    self.assertEqual("bar_2", g.unique_name("bar"))

  def testBackslashAndDashRegex(self):
    # [FUNCTION DOC]
    # Regression check for a GitHub issue: name scopes whose names contain backslashes and dashes must
    # be accepted at top level and when nested in either order. Passes if no error is raised.
    # GitHub issue 39019, all should pass
    g = ops.Graph()
    with g.name_scope("n_CatCntc-campaign\\c_campaign"):
      pass
    with g.name_scope("foo"):
      with g.name_scope("n_CatCntc-campaign\\c_campaign"):
        pass
    with g.name_scope("n_CatCntc-campaign\\c_campaign"):
      with g.name_scope("foo"):
        pass

  @test_util.run_deprecated_v1
  def testNameAndVariableScope(self):
    # [FUNCTION DOC]
    # Graph-mode test. Nests a name scope, variable scope and name scope, and asserts the resulting
    # scope strings and unique names include the combined prefix (l0 slash l1 slash l1, and so on).
    with self.cached_session() as sess:
      with sess.graph.name_scope("l0"):
        with variable_scope.variable_scope("l1"):
          with sess.graph.name_scope("l1") as scope:
            self.assertEqual("l0/l1/l1/", scope)
            self.assertEqual(
                "l0/l1/l1/foo",
                sess.graph.unique_name(
                    "foo", mark_as_used=False))
            self.assertEqual("l0/l1/l1/foo", sess.graph.unique_name("foo"))
          with sess.graph.name_scope("l2") as scope:
            self.assertEqual("l0/l1/l2/", scope)
            self.assertEqual(
                "l0/l1/l2/foo",
                sess.graph.unique_name(
                    "foo", mark_as_used=False))
            self.assertEqual("l0/l1/l2/foo", sess.graph.unique_name("foo"))

  def testOutOfOrderUniqueName(self):
    # [FUNCTION DOC]
    # Asserts unique names handle suffixes requested out of order: requesting foo_2 first, then foo
    # gives foo then foo_1 then foo_3, skipping the used name.
    g = ops.Graph()
    self.assertEqual("foo_2", g.unique_name("foo_2"))
    self.assertEqual("foo", g.unique_name("foo"))
    self.assertEqual("foo_1", g.unique_name("foo"))
    self.assertEqual("foo_3", g.unique_name("foo"))

  def testUniqueNameCaseInsensitivity(self):
    # [FUNCTION DOC]
    # Asserts unique names ignore case differences: foo then Foo gives Foo_1, and the same applies to
    # scope names (bar versus Bar).
    g = ops.Graph()
    self.assertEqual("foo", g.unique_name("foo"))
    self.assertEqual("Foo_1", g.unique_name("Foo"))
    with g.name_scope("bar"):
      self.assertEqual("bar/foo", g.unique_name("foo"))
    with g.name_scope("Bar"):
      self.assertEqual("Bar_1/foo", g.unique_name("foo"))

  def testInvalidNameRaisesError(self):
    # [FUNCTION DOC]
    # Asserts name-scope validation: empty names, names with a trailing slash and a leading
    # underscore inside a slash-terminated scope are allowed; a name containing a colon, or a bare
    # leading underscore name, raises a value error.
    g = ops.Graph()
    with g.name_scope(""):  # Should not raise
      pass
    with g.name_scope("foo/"):  # Should not raise
      with g.name_scope("_bar"):  # Should not raise
        pass
    with self.assertRaises(ValueError):
      with g.name_scope("foo:0"):
        pass
    with self.assertRaises(ValueError):
      with g.name_scope("_bar"):
        pass

  def testEmptyScopeEdgeCases(self):
    # [FUNCTION DOC]
    # Asserts edge cases for empty scopes: initially the name scope is empty; entering a scope with an
    # empty string or null yields an empty scope string and name scope; nested empty or null inside a
    # named scope resets to empty.
    g = ops.Graph()
    self.assertEqual("", g.get_name_scope())
    with g.name_scope("") as scope:
      self.assertEqual("", scope)
      self.assertEqual("", g.get_name_scope())
    with g.name_scope(None) as scope:
      self.assertEqual("", scope)
      self.assertEqual("", g.get_name_scope())
    with g.name_scope("foo") as scope:
      self.assertEqual("foo/", scope)
      self.assertEqual("foo", g.get_name_scope())
      with g.name_scope("") as scope:
        self.assertEqual("", scope)
        self.assertEqual("", g.get_name_scope())
      with g.name_scope(None) as scope:
        self.assertEqual("", scope)
        self.assertEqual("", g.get_name_scope())


class NameTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for automatic naming of operations created in a graph and name scopes.

  def testGenerateName(self):
    # [FUNCTION DOC]
    # Creates ops without names and asserts default names from the op type, with suffix numbers for
    # repeats (name, name_1), output tensor names with colon indices, and explicitly given names
    # used as-is.
    g = ops.Graph()
    op0 = g.create_op("TwoFloatOutputs", [], [dtypes.float32, dtypes.float32])
    self.assertEqual("TwoFloatOutputs", op0.name)
    self.assertEqual("TwoFloatOutputs:0", op0.outputs[0].name)
    self.assertEqual("TwoFloatOutputs:1", op0.outputs[1].name)

    op1 = g.create_op("FloatOutput", [], [dtypes.float32])
    self.assertEqual("FloatOutput", op1.name)
    self.assertEqual("FloatOutput:0", op1.outputs[0].name)

    op2 = g.create_op("FloatOutput", [], [dtypes.float32])
    self.assertEqual("FloatOutput_1", op2.name)
    self.assertEqual("FloatOutput_1:0", op2.outputs[0].name)

    op3 = g.create_op("FloatOutput", [], [dtypes.float32], name="my_op")
    self.assertEqual("my_op", op3.name)
    self.assertEqual("my_op:0", op3.outputs[0].name)

  def testNameScope(self):
    # [FUNCTION DOC]
    # Checks name scopes: nested scope strings with trailing slashes, null scope clearing the prefix,
    # empty-string scope, prefix on created op names, repeated scope names getting numeric suffixes,
    # reuse of a scope string from a with-as clause not pushing a new level, and trailing slash names
    # being used as is.
    g = ops.Graph()

    with g.name_scope("foo") as foo:
      self.assertEqual("foo/", foo)
      with g.name_scope("foo2") as foo2:
        self.assertEqual("foo/foo2/", foo2)
      with g.name_scope(None) as empty1:
        self.assertEqual("", empty1)
        with g.name_scope("foo3") as foo3:
          self.assertEqual("foo3/", foo3)
      with g.name_scope("") as empty2:
        self.assertEqual("", empty2)

    self.assertEqual("FloatOutput",
                     g.create_op("FloatOutput", [], [dtypes.float32]).name)
    with g.name_scope("bar") as scope:
      self.assertEqual("bar/FloatOutput",
                       g.create_op("FloatOutput", [], [dtypes.float32]).name)
      self.assertEqual("bar/FloatOutput_1",
                       g.create_op("FloatOutput", [], [dtypes.float32]).name)
      # If you use the value from "with .. as", that values is used as-is.
      self.assertEqual(
          "bar", g.create_op(
              "FloatOutput", [], [dtypes.float32], name=scope).name)
    with g.name_scope("baz") as scope:
      with g.name_scope("quux"):
        self.assertEqual("baz/quux/FloatOutput",
                         g.create_op("FloatOutput", [], [dtypes.float32]).name)
      # If you use the value from the enclosing "with .. as", nothing is pushed.
      with g.name_scope(scope):
        self.assertEqual("baz/FloatOutput",
                         g.create_op("FloatOutput", [], [dtypes.float32]).name)
        self.assertEqual(
            "baz", g.create_op(
                "FloatOutput", [], [dtypes.float32], name=scope).name)
        self.assertEqual(
            "trailing",
            g.create_op(
                "FloatOutput", [], [dtypes.float32], name="trailing/").name)
    with g.name_scope("bar"):
      self.assertEqual("bar_1/FloatOutput",
                       g.create_op("FloatOutput", [], [dtypes.float32]).name)
    with g.name_scope("bar/"):
      self.assertEqual("bar/FloatOutput_2",
                       g.create_op("FloatOutput", [], [dtypes.float32]).name)


class DeviceTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for device scopes on a graph: how the device specification of created operations
  # is recorded in the graph definition, nesting rules, merge functions, wildcards and error cases.

  def testNoDevice(self):
    # [FUNCTION DOC]
    # Creates an op without any device scope; asserts its device is unset and that the graph
    # definition contains the node with no device field.
    g = ops.Graph()
    op = g.create_op("FloatOutput", [], [dtypes.float32])
    self.assertDeviceEqual(None, op.device)
    gd = g.as_graph_def()
    self.assertProtoEqualsVersion("""
      node { name: "FloatOutput" op: "FloatOutput" }
    """, gd)

  def testEagerBackingDevice(self):
    # [FUNCTION DOC]
    # Eager mode. Inside a CPU device scope, creates a constant and asserts both its device and its
    # backing device mention the CPU zero device.
    with context.eager_mode():
      with ops.device("/device:CPU:0"):
        t = constant_op.constant(1.0)
        self.assertRegex(t.device, "/device:CPU:0")
        self.assertRegex(t.backing_device, "/device:CPU:0")

  def testDevicePartialString(self):
    # [FUNCTION DOC]
    # Creates an op inside a partial device string scope (job worker, replica two) and asserts the
    # graph definition records that exact partial device string.
    g = ops.Graph()
    with g.device("/job:worker/replica:2"):
      g.create_op("FloatOutput", [], [dtypes.float32])
    gd = g.as_graph_def()
    self.assertProtoEqualsVersion("""
      node { name: "FloatOutput" op: "FloatOutput"
             device: "/job:worker/replica:2" }
    """, gd)

  def testDeviceFull(self):
    # [FUNCTION DOC]
    # Creates an op inside a scope given as a full device specification object (job, replica, task,
    # CPU type, index three) and asserts the graph definition records the fully expanded device name.
    g = ops.Graph()
    with g.device(
        pydev.DeviceSpec(
            job="worker", replica=2, task=0, device_type="CPU",
            device_index=3)):
      g.create_op("FloatOutput", [], [dtypes.float32])
    gd = g.as_graph_def()
    self.assertProtoEqualsVersion("""
      node { name: "FloatOutput" op: "FloatOutput"
             device: "/job:worker/replica:2/task:0/device:CPU:3" }
    """, gd)

  def testNesting(self):
    # [FUNCTION DOC]
    # Nests a device scope for a different replica and task inside an outer scope, creating ops in the
    # outer, inner and outer scopes again. Asserts each node records the innermost active scope at its
    # creation time.
    g = ops.Graph()
    with g.device("/job:worker/replica:2"):
      g.create_op("FloatOutput", [], [dtypes.float32])
      with g.device("/job:worker/replica:3/task:0"):
        g.create_op("FloatOutput", [], [dtypes.float32])
      g.create_op("FloatOutput", [], [dtypes.float32])
    gd = g.as_graph_def()
    self.assertProtoEqualsVersion("""
      node { name: "FloatOutput" op: "FloatOutput"
             device: "/job:worker/replica:2" }
      node { name: "FloatOutput_1" op: "FloatOutput"
             device: "/job:worker/replica:3/task:0" }
      node { name: "FloatOutput_2" op: "FloatOutput"
             device: "/job:worker/replica:2" }
    """, gd)

  def testNestingString(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the body is identical to the preceding nesting test, which also uses device strings,
    # so the name suggests a distinction (strings versus specification objects) that does not exist.
    # Same scenario as the nesting test (outer scope, inner scope, outer scope) using device strings.
    # Asserts the same three node devices.
    g = ops.Graph()
    with g.device("/job:worker/replica:2"):
      g.create_op("FloatOutput", [], [dtypes.float32])
      with g.device("/job:worker/replica:3/task:0"):
        g.create_op("FloatOutput", [], [dtypes.float32])
      g.create_op("FloatOutput", [], [dtypes.float32])
    gd = g.as_graph_def()
    self.assertProtoEqualsVersion("""
      node { name: "FloatOutput" op: "FloatOutput"
             device: "/job:worker/replica:2" }
      node { name: "FloatOutput_1" op: "FloatOutput"
             device: "/job:worker/replica:3/task:0" }
      node { name: "FloatOutput_2" op: "FloatOutput"
             device: "/job:worker/replica:2" }
    """, gd)

  def testNestingOverrideGpuCpu(self):
    # [FUNCTION DOC]
    # Nests a GPU device scope inside a CPU device scope within the same job and replica. Asserts the
    # inner op uses the GPU device and ops outside it use the CPU device.
    g = ops.Graph()
    with g.device("/job:worker/replica:2/device:CPU:1"):
      g.create_op("FloatOutput", [], [dtypes.float32])
      with g.device("/job:worker/replica:2/device:GPU:2"):
        g.create_op("FloatOutput", [], [dtypes.float32])
      g.create_op("FloatOutput", [], [dtypes.float32])
    gd = g.as_graph_def()
    self.assertProtoEqualsVersion("""
      node { name: "FloatOutput" op: "FloatOutput"
             device: "/job:worker/replica:2/device:CPU:1"  }
      node { name: "FloatOutput_1" op: "FloatOutput"
             device: "/job:worker/replica:2/device:GPU:2" }
      node { name: "FloatOutput_2" op: "FloatOutput"
             device: "/job:worker/replica:2/device:CPU:1" }
    """, gd)

  def testNestingWithMergeDeviceFunction(self):
    # [FUNCTION DOC]
    # Nests scopes created from merge-device functions (GPU zero, job worker, CPU zero, job ps, then
    # a null merge). Asserts each node's device merges components from enclosing scopes: later
    # components override earlier ones of the same kind, and a null merge keeps the enclosing device.
    g = ops.Graph()

    with g.device(pydev.merge_device("/device:GPU:0")):
      g.create_op("FloatOutput", [], [dtypes.float32])
      with g.device(pydev.merge_device("/job:worker")):
        g.create_op("FloatOutput", [], [dtypes.float32])
        with g.device(pydev.merge_device("/device:CPU:0")):
          g.create_op("FloatOutput", [], [dtypes.float32])
          with g.device(pydev.merge_device("/job:ps")):
            g.create_op("FloatOutput", [], [dtypes.float32])
            with g.device(pydev.merge_device(None)):
              g.create_op("FloatOutput", [], [dtypes.float32])

    gd = g.as_graph_def()
    self.assertProtoEqualsVersion("""
      node { name: "FloatOutput" op: "FloatOutput"
             device: "/device:GPU:0" }
      node { name: "FloatOutput_1" op: "FloatOutput"
             device: "/job:worker/device:GPU:0" }
      node { name: "FloatOutput_2" op: "FloatOutput"
             device: "/job:worker/device:CPU:0" }
      node { name: "FloatOutput_3" op: "FloatOutput"
             device: "/job:ps/device:CPU:0" }
      node { name: "FloatOutput_4" op: "FloatOutput"
             device: "/job:ps/device:CPU:0" }
    """, gd)

  def testNestingWithDeviceStrings(self):
    # [FUNCTION DOC]
    # Same nesting as the merge-function test but using plain device strings; the innermost scope is
    # an empty string. Asserts the same merged device names, showing an empty string leaves the
    # enclosing device unchanged.
    g = ops.Graph()

    with g.device("/device:GPU:0"):
      g.create_op("FloatOutput", [], [dtypes.float32])
      with g.device("/job:worker"):
        g.create_op("FloatOutput", [], [dtypes.float32])
        with g.device("/device:CPU:0"):
          g.create_op("FloatOutput", [], [dtypes.float32])
          with g.device("/job:ps"):
            g.create_op("FloatOutput", [], [dtypes.float32])
            with g.device(""):
              g.create_op("FloatOutput", [], [dtypes.float32])

    gd = g.as_graph_def()
    self.assertProtoEqualsVersion("""
      node { name: "FloatOutput" op: "FloatOutput"
             device: "/device:GPU:0" }
      node { name: "FloatOutput_1" op: "FloatOutput"
             device: "/job:worker/device:GPU:0" }
      node { name: "FloatOutput_2" op: "FloatOutput"
             device: "/job:worker/device:CPU:0" }
      node { name: "FloatOutput_3" op: "FloatOutput"
             device: "/job:ps/device:CPU:0" }
      node { name: "FloatOutput_4" op: "FloatOutput"
             device: "/job:ps/device:CPU:0" }
    """, gd)

  def testNestingWithDeviceStringWildcard(self):
    # [FUNCTION DOC]
    # Checks wildcard device indices in nested scopes: a GPU wildcard inside GPU seven keeps seven,
    # while a CPU wildcard followed by explicit CPU five gives the wildcard for the outer op and five
    # for the inner op. Asserts the four node devices.
    g = ops.Graph()

    with g.device("/device:GPU:7"):
      g.create_op("FloatOutput", [], [dtypes.float32])
      with g.device("/device:GPU:*"):
        g.create_op("FloatOutput", [], [dtypes.float32])

    with g.device("/device:CPU:*"):
      g.create_op("FloatOutput", [], [dtypes.float32])
      with g.device("/device:CPU:5"):
        g.create_op("FloatOutput", [], [dtypes.float32])

    gd = g.as_graph_def()
    self.assertProtoEqualsVersion("""
      node { name: "FloatOutput" op: "FloatOutput"
             device: "/device:GPU:7" }
      node { name: "FloatOutput_1" op: "FloatOutput"
             device: "/device:GPU:7" }
      node { name: "FloatOutput_2" op: "FloatOutput"
             device: "/device:CPU:*" }
      node { name: "FloatOutput_3" op: "FloatOutput"
             device: "/device:CPU:5" }
    """, gd)

  def testNestingErrorGraph(self):
    # [FUNCTION DOC]
    # Enters a graph device scope manually, enters another scope inside, then leaves the outer scope
    # out of order and asserts a runtime error is raised.
    g = ops.Graph()
    scope = g.device("/device:GPU:8")
    scope.__enter__()
    with g.device("/device:GPU:9"):
      with self.assertRaises(RuntimeError):
        scope.__exit__(None, None, None)

  def testNestingErrorEager(self):
    # [FUNCTION DOC]
    # Eager-mode version of the out-of-order scope exit: enters a CPU device scope manually, enters a
    # null device scope, then exits the outer scope and asserts a runtime error.
    with context.eager_mode():
      scope = ops.device("/device:CPU:0")
      scope.__enter__()
      with ops.device(None):
        with self.assertRaises(RuntimeError):
          scope.__exit__(None, None, None)

  def testNoneClearsDefault(self):
    # [FUNCTION DOC]
    # Asserts a null device scope clears the inherited default device: the op created inside it has no
    # device, while ops outside keep the outer device.
    g = ops.Graph()
    with g.device("/job:worker/replica:2/device:CPU:1"):
      g.create_op("FloatOutput", [], [dtypes.float32])
      with g.device(None):
        g.create_op("FloatOutput", [], [dtypes.float32])
      g.create_op("FloatOutput", [], [dtypes.float32])
    gd = g.as_graph_def()
    self.assertProtoEqualsVersion("""
      node { name: "FloatOutput" op: "FloatOutput"
             device: "/job:worker/replica:2/device:CPU:1" }
      node { name: "FloatOutput_1" op: "FloatOutput" }
      node { name: "FloatOutput_2" op: "FloatOutput"
             device: "/job:worker/replica:2/device:CPU:1" }
    """, gd)

  def testNoneIgnoresOuterDeviceFunction(self):
    # [FUNCTION DOC]
    # Asserts a null device scope also ignores an outer device function (a function returning a fixed
    # device): ops inside the null scope have no device, ops outside use the function's device.
    g = ops.Graph()
    with g.device(lambda op: "/job:worker/replica:2/device:CPU:1"):
      g.create_op("FloatOutput", [], [dtypes.float32])
      with g.device(None):
        g.create_op("FloatOutput", [], [dtypes.float32])
      g.create_op("FloatOutput", [], [dtypes.float32])
    gd = g.as_graph_def()
    self.assertProtoEqualsVersion("""
      node { name: "FloatOutput" op: "FloatOutput"
             device: "/job:worker/replica:2/device:CPU:1" }
      node { name: "FloatOutput_1" op: "FloatOutput" }
      node { name: "FloatOutput_2" op: "FloatOutput"
             device: "/job:worker/replica:2/device:CPU:1" }
    """, gd)

  def _overwritingDeviceFunction(self, unused_op):
    # [FUNCTION DOC]
    # HELPER: Device function used by the overwriting-behavior test.
    # Takes the operation and ignores it, always returning the job-overwrite device string, so any
    # operation under this function has its device unconditionally replaced. The existing comment
    # notes that such functions are not recommended.
    # This device function unconditionally overwrites the device of ops.
    #
    # NOTE(mrry): Writing device functions like this is not
    # recommended. Instead, in most cases you should use
    # `pydev.merge_device("/job:ps")` or simply `"/job:ps"` as the
    # argument to `tf.device()` and the device component will be merged in.
    return "/job:overwrite"

  def testOverwritingBehavior(self):
    # [FUNCTION DOC]
    # Checks how an unconditional device function overrides inner device scopes: ops under inner
    # string or merge scopes still get the overwrite device, while ops under a null scope that is
    # nested before the inner scope get the inner device. Asserts all five node devices.
    g = ops.Graph()
    with g.device(self._overwritingDeviceFunction):
      g.create_op("FloatOutput", [], [dtypes.float32])
      with g.device("/job:ps"):  # Will be overwritten.
        g.create_op("FloatOutput", [], [dtypes.float32])
      with g.device(pydev.merge_device("/job:ps")):  # Will be overwritten.
        g.create_op("FloatOutput", [], [dtypes.float32])
      with g.device(None):  # Disables overwriting device function
        with g.device("/job:ps"):
          g.create_op("FloatOutput", [], [dtypes.float32])
      with g.device(None):  # Disables overwriting device function
        with g.device(pydev.merge_device("/job:ps")):
          g.create_op("FloatOutput", [], [dtypes.float32])
    gd = g.as_graph_def()
    self.assertProtoEqualsVersion("""
      node { name: "FloatOutput" op: "FloatOutput"
             device: "/job:overwrite" }
      node { name: "FloatOutput_1" op: "FloatOutput"
             device: "/job:overwrite" }
      node { name: "FloatOutput_2" op: "FloatOutput"
             device: "/job:overwrite" }
      node { name: "FloatOutput_3" op: "FloatOutput"
             device: "/job:ps" }
      node { name: "FloatOutput_4" op: "FloatOutput"
             device: "/job:ps" }
    """, gd)


class MultithreadedGraphStateTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case verifying that a graph's device, colocation, control-dependency and name stacks are
  # thread-local when the graph is switched to thread-local state. Uses helper thread classes that
  # mutate the graph, signal the main thread, wait, then create an op.

  class TestThread(threading.Thread):
    # [CLASS DOC]
    # HELPER: Base thread class for the multithreaded graph-state tests.
    # Holds a graph, a replica id and two events: one the thread sets after mutating the graph's
    # state stack, and one it waits on before adding an op. The run method is abstract and raises a
    # not-implemented error; subclasses supply the behavior.

    def __init__(self, graph, replica_id):
      # [FUNCTION DOC]
      # HELPER: Constructor of the base test thread.
      # Inputs: graph and replica id. Calls the base thread constructor, stores both, and creates the
      # has-mutated and should-continue events. Returns nothing.
      super(MultithreadedGraphStateTest.TestThread, self).__init__()
      self._graph = graph
      self._replica_id = replica_id
      # This thread sets this event when it mutated the graph.  The caller can
      # wait for that.
      self.has_mutated_graph = threading.Event()
      # This thread waits for when it should continue.  The caller can set this
      # event.
      self.should_continue = threading.Event()

    def run(self):
      # [FUNCTION DOC]
      # HELPER: Abstract run method of the base test thread.
      # Always raises a not-implemented error saying it must be implemented in descendants.
      # Mutate a graph's stack, then set `has_mutated_graph`, then wait for
      # `should_continue`, then add an op to the graph affected by the graph's
      # stack.
      raise NotImplementedError("must be implemented in descendants")

  def testDeviceFunctionStack(self):
    # [FUNCTION DOC]
    # Starts three threads, each entering a device scope for its own replica, signalling the main
    # thread, waiting, then creating a named op. The graph is switched to thread-local mode first.
    # Asserts each op is placed on its own thread's replica despite interleaving.

    class DeviceSettingThread(self.TestThread):
      # [CLASS DOC]
      # HELPER: Thread class for the device-function-stack test.
      # Subclass of the base test thread whose run method enters a device scope for its replica,
      # signals mutation, waits to continue, then creates an op named after its replica id.

      def run(self):
        # [FUNCTION DOC]
        # HELPER: Thread body: enters the device scope for this thread's replica, sets the mutated event,
        # waits for and clears the continue event, then creates a float output op named with the replica
        # id in the enclosing test's graph. Returns nothing.
        with g.device("/job:worker/replica:{}".format(self._replica_id)):
          self.has_mutated_graph.set()
          self.should_continue.wait()
          self.should_continue.clear()
          g.create_op(
              "FloatOutput", [], [dtypes.float32],
              name="FloatOutput_{}".format(self._replica_id))

    g = ops.Graph()
    # If `switch_to_thread` isn't called, then device placement of the ops
    # below is not deterministic.
    g.switch_to_thread_local()
    threads = [DeviceSettingThread(g, i) for i in range(3)]
    for t in threads:
      t.start()
      t.has_mutated_graph.wait()
      t.has_mutated_graph.clear()
    for t in threads:
      t.should_continue.set()
      t.join()

    gd = g.as_graph_def()
    self.assertProtoEqualsVersion("""
      node { name: "FloatOutput_0" op: "FloatOutput"
             device: "/job:worker/replica:0" }
      node { name: "FloatOutput_1" op: "FloatOutput"
             device: "/job:worker/replica:1" }
      node { name: "FloatOutput_2" op: "FloatOutput"
             device: "/job:worker/replica:2" }
    """, gd)

  def testColocateWith(self):
    # [FUNCTION DOC]
    # Same threading scenario for colocation: three ops are created, each on a different replica;
    # three threads each colocate with a different one, then create an op. Asserts each new op gets
    # both its colocated op's device and a class attribute naming the colocation target.

    class ColocatingThread(self.TestThread):
      # [CLASS DOC]
      # HELPER: Thread class for the colocation test, holding the op to colocate with.

      def __init__(self, graph, replica_id, op_to_colocate_with):
        # [FUNCTION DOC]
        # HELPER: Constructor. Inputs: graph, replica id and the operation to colocate with. Calls the
        # base constructor and stores the operation. Returns nothing.
        super(ColocatingThread, self).__init__(graph, replica_id)
        self._op_to_colocate_with = op_to_colocate_with

      def run(self):
        # [FUNCTION DOC]
        # HELPER: Thread body: enters colocation with its stored op, signals the main thread, waits to
        # continue, then creates a float output op named with the replica id. Returns nothing.
        with g.colocate_with(self._op_to_colocate_with):
          self.has_mutated_graph.set()
          self.should_continue.wait()
          self.should_continue.clear()
          g.create_op(
              "FloatOutput", [], [dtypes.float32],
              name="FloatOutput_{}".format(self._replica_id))

    g = ops.Graph()
    ops_to_colocate_with = []
    for i in range(3):
      with g.device("/job:worker/replica:{}".format(i)):
        ops_to_colocate_with.append(
            g.create_op(
                "FloatOutput", [], [dtypes.float32],
                name="ColocateWithMe_{}".format(i)))

    # If `switch_to_thread` isn't called, then `device` and `attr` values for
    # the ops below are not deterministic.
    g.switch_to_thread_local()
    threads = [
        ColocatingThread(g, i, ops_to_colocate_with[i]) for i in range(3)
    ]
    for t in threads:
      t.start()
      t.has_mutated_graph.wait()
      t.has_mutated_graph.clear()
    for t in threads:
      t.should_continue.set()
      t.join()

    gd = g.as_graph_def()
    self.assertProtoEqualsVersion("""
      node { name: "ColocateWithMe_0" op: "FloatOutput"
             device: "/job:worker/replica:0" }
      node { name: "ColocateWithMe_1" op: "FloatOutput"
             device: "/job:worker/replica:1" }
      node { name: "ColocateWithMe_2" op: "FloatOutput"
             device: "/job:worker/replica:2" }
      node { name: "FloatOutput_0" op: "FloatOutput"
             device: "/job:worker/replica:0"
             attr { key: "_class"
               value { list {
                 s: "loc:@ColocateWithMe_0"}}}}
      node { name: "FloatOutput_1" op: "FloatOutput"
             device: "/job:worker/replica:1"
             attr { key: "_class"
               value { list {
                 s: "loc:@ColocateWithMe_1"}}}}
      node { name: "FloatOutput_2" op: "FloatOutput"
             device: "/job:worker/replica:2"
             attr { key: "_class"
               value { list {
                 s: "loc:@ColocateWithMe_2"}}}}
    """, gd)

  def testControlDependencies(self):
    # [FUNCTION DOC]
    # Same threading scenario for control dependencies: three independent ops are created, then each
    # thread enters a control dependency scope on a different one and creates an op. Asserts each new
    # op depends only on its own thread's dependency op and the dependency ops carry the manual
    # control-dependency marker attribute.

    class DependingThread(self.TestThread):
      # [CLASS DOC]
      # HELPER: Thread class for the control-dependencies test, holding the dependency op.

      def __init__(self, graph, replica_id, dependency_op):
        # [FUNCTION DOC]
        # HELPER: Constructor. Inputs: graph, replica id and the dependency operation. Calls the base
        # constructor and stores the operation. Returns nothing.
        super(DependingThread, self).__init__(graph, replica_id)
        self._dependency_op = dependency_op

      def run(self):
        # [FUNCTION DOC]
        # HELPER: Thread body: enters a control dependency scope on its stored op, signals, waits to
        # continue, then creates a float output op named with the replica id. Returns nothing.
        with g.control_dependencies([self._dependency_op]):
          self.has_mutated_graph.set()
          self.should_continue.wait()
          self.should_continue.clear()
          g.create_op(
              "FloatOutput", [], [dtypes.float32],
              name="FloatOutput_{}".format(self._replica_id))

    g = ops.Graph()
    dependency_ops = []
    for i in range(3):
      dependency_ops.append(
          g.create_op(
              "FloatOutput", [], [dtypes.float32],
              name="ColocateWithMe_{}".format(i)))

    # If `switch_to_thread` isn't called, then `input` values for the ops below
    # are not deterministic.
    g.switch_to_thread_local()
    threads = [DependingThread(g, i, dependency_ops[i]) for i in range(3)]
    for t in threads:
      t.start()
      t.has_mutated_graph.wait()
      t.has_mutated_graph.clear()
    for t in threads:
      t.should_continue.set()
      t.join()

    gd = g.as_graph_def()
    self.assertProtoEqualsVersion(
        """
      node { name: "ColocateWithMe_0" op: "FloatOutput"
             attr { key: "_has_manual_control_dependencies"
                    value { b: true } } }
      node { name: "ColocateWithMe_1" op: "FloatOutput"
             attr { key: "_has_manual_control_dependencies"
                    value { b: true } } }
      node { name: "ColocateWithMe_2" op: "FloatOutput"
             attr { key: "_has_manual_control_dependencies"
                    value { b: true } } }
      node { name: "FloatOutput_0" op: "FloatOutput"
             input: "^ColocateWithMe_0" }
      node { name: "FloatOutput_1" op: "FloatOutput"
             input: "^ColocateWithMe_1" }
      node { name: "FloatOutput_2" op: "FloatOutput"
             input: "^ColocateWithMe_2" }
    """, gd)

  def testNameStack(self):
    # [FUNCTION DOC]
    # Checks that the name scope stack is per-thread even without switching the graph to thread-local
    # mode: three threads each enter a scope named foo and create two ops. Asserts the scope names are
    # uniquified per thread (foo, foo_1, foo_2) and the second op in each thread gets suffix one.

    class NameSettingThread(self.TestThread):
      # [CLASS DOC]
      # HELPER: Thread class for the name-stack test; its result attribute holds the pair of created ops.

      def run(self):
        # [FUNCTION DOC]
        # HELPER: Thread body: enters a name scope called foo, creates an op, signals, waits to continue,
        # creates a second op, and stores both ops in a result attribute. Returns nothing.
        with g.name_scope("foo"):
          op1 = g.create_op("FloatOutput", [], [dtypes.float32])
          self.has_mutated_graph.set()
          self.should_continue.wait()
          self.should_continue.clear()
          op2 = g.create_op("FloatOutput", [], [dtypes.float32])
          self.result = (op1, op2)

    g = ops.Graph()
    threads = [NameSettingThread(g, i) for i in range(3)]
    for t in threads:
      t.start()
      t.has_mutated_graph.wait()
      t.has_mutated_graph.clear()

    for t in threads:
      t.should_continue.set()
      t.join()

    suffixes = ["", "_1", "_2"]
    for t, s in zip(threads, suffixes):
      self.assertEqual("foo" + s + "/FloatOutput", t.result[0].name)
      self.assertEqual("foo" + s + "/FloatOutput_1", t.result[1].name)


class ObjectWithName(object):
  # [CLASS DOC]
  # HELPER: Simple object with a name property, used by the collection tests to exercise filtering
  # of collection items by a name pattern. Holds a single name given at construction.

  def __init__(self, name):
    # [FUNCTION DOC]
    # HELPER: Constructor. Input: name. Stores it. Returns nothing.
    self._name = name

  @property
  def name(self):
    # [FUNCTION DOC]
    # HELPER: Read-only property returning the stored name string.
    return self._name


class CollectionTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for graph collections: listing keys, adding items, reading with name filters,
  # snapshot versus reference semantics, adding to multiple collections and the default graph.

  def test_get_collections(self):
    # [FUNCTION DOC]
    # Asserts the collection key list is empty for a new graph, then contains each key once after
    # adding items (key, then other), and that the all-keys method agrees.
    g = ops.Graph()
    self.assertSequenceEqual(g.collections, [])
    g.add_to_collection("key", 12)
    g.add_to_collection("key", 15)
    self.assertSequenceEqual(g.collections, ["key"])
    g.add_to_collection("other", "foo")
    self.assertSequenceEqual(sorted(g.collections), ["key", "other"])
    self.assertSequenceEqual(
        sorted(g.get_all_collection_keys()), ["key", "other"])

  def test_add_to_collection(self):
    # [FUNCTION DOC]
    # Adds items to several collections and asserts reads: items in insertion order, empty list for
    # unknown collections, filtering by a name prefix or pattern returns only items whose names match
    # (non-named items are dropped), the plain getter returns a copy that does not see later additions,
    # the reference getter returns the live list (same object every time, appends visible), and a
    # reference to an empty collection is a modifiable list that becomes the stored collection.
    g = ops.Graph()
    g.add_to_collection("key", 12)
    g.add_to_collection("other", "foo")
    g.add_to_collection("key", 34)

    # Note that only blank1 is returned.
    g.add_to_collection("blah", 27)
    blank1 = ObjectWithName("prefix/foo")
    g.add_to_collection("blah", blank1)
    blank2 = ObjectWithName("junk/foo")
    g.add_to_collection("blah", blank2)

    self.assertEqual([12, 34], g.get_collection("key"))
    self.assertEqual([], g.get_collection("nothing"))
    self.assertEqual([27, blank1, blank2], g.get_collection("blah"))
    self.assertEqual([blank1], g.get_collection("blah", "prefix"))
    self.assertEqual([blank1], g.get_collection("blah", ".*x"))

    # Make sure that get_collection() returns a first-level
    # copy of the collection, while get_collection_ref() returns
    # the original list.
    other_collection_snapshot = g.get_collection("other")
    other_collection_ref = g.get_collection_ref("other")
    self.assertEqual(["foo"], other_collection_snapshot)
    self.assertEqual(["foo"], other_collection_ref)
    g.add_to_collection("other", "bar")
    self.assertEqual(["foo"], other_collection_snapshot)
    self.assertEqual(["foo", "bar"], other_collection_ref)
    self.assertEqual(["foo", "bar"], g.get_collection("other"))
    self.assertTrue(other_collection_ref is g.get_collection_ref("other"))

    # Verify that getting an empty collection ref returns a modifiable list.
    empty_coll_ref = g.get_collection_ref("empty")
    self.assertEqual([], empty_coll_ref)
    empty_coll = g.get_collection("empty")
    self.assertEqual([], empty_coll)
    self.assertFalse(empty_coll is empty_coll_ref)
    empty_coll_ref2 = g.get_collection_ref("empty")
    self.assertTrue(empty_coll_ref2 is empty_coll_ref)
    # Add to the collection.
    empty_coll_ref.append("something")
    self.assertEqual(["something"], empty_coll_ref)
    self.assertEqual(["something"], empty_coll_ref2)
    self.assertEqual([], empty_coll)
    self.assertEqual(["something"], g.get_collection("empty"))
    empty_coll_ref3 = g.get_collection_ref("empty")
    self.assertTrue(empty_coll_ref3 is empty_coll_ref)

  def test_add_to_collections_uniquify(self):
    # [FUNCTION DOC]
    # Adds a key to a list of collection names with a repeated name and asserts the item is stored only
    # once in that collection.
    g = ops.Graph()
    g.add_to_collections([1, 2, 1], "key")
    # Make sure "key" is not added twice
    self.assertEqual(["key"], g.get_collection(1))

  def test_add_to_collections_from_list(self):
    # [FUNCTION DOC]
    # Adds an item to collections named by a list and asserts both collections contain it.
    g = ops.Graph()
    g.add_to_collections(["abc", "123"], "key")
    self.assertEqual(["key"], g.get_collection("abc"))
    self.assertEqual(["key"], g.get_collection("123"))

  def test_add_to_collections_from_tuple(self):
    # [FUNCTION DOC]
    # Adds an item to collections named by a tuple and asserts both collections contain it.
    g = ops.Graph()
    g.add_to_collections(("abc", "123"), "key")
    self.assertEqual(["key"], g.get_collection("abc"))
    self.assertEqual(["key"], g.get_collection("123"))

  def test_add_to_collections_from_generator(self):
    # [FUNCTION DOC]
    # Adds an item to collections named by a generator of names (any iterable) and asserts both
    # collections contain it.
    g = ops.Graph()

    def generator():
      # [FUNCTION DOC]
      # HELPER: Generator yielding the two collection names used by the generator test. Yields two
      # strings, abc and 123, and returns nothing.
      yield "abc"
      yield "123"

    g.add_to_collections(generator(), "key")
    self.assertEqual(["key"], g.get_collection("abc"))
    self.assertEqual(["key"], g.get_collection("123"))

  def test_add_to_collections_from_set(self):
    # [FUNCTION DOC]
    # Adds an item to collections named by a set and asserts both collections contain it.
    g = ops.Graph()
    g.add_to_collections(set(["abc", "123"]), "key")
    self.assertEqual(["key"], g.get_collection("abc"))
    self.assertEqual(["key"], g.get_collection("123"))

  def test_add_to_collections_from_string(self):
    # [FUNCTION DOC]
    # Adds an item to a collection named by a single string and asserts it is stored in that one
    # collection (the string is not split into characters).
    g = ops.Graph()
    g.add_to_collections("abc", "key")
    self.assertEqual(["key"], g.get_collection("abc"))

  def test_default_graph(self):
    # [FUNCTION DOC]
    # Uses the module-level add and get functions for the default graph and asserts the collection
    # keeps items in insertion order.
    with ops.Graph().as_default():
      ops.add_to_collection("key", 90)
      ops.add_to_collection("key", 100)
      # Collections are ordered.
      self.assertEqual([90, 100], ops.get_collection("key"))


ops.NotDifferentiable("FloatOutput")


@ops.RegisterGradient("CopyOp")
def _CopyGrad(op, x_grad):  # pylint: disable=invalid-name
  # [FUNCTION DOC]
  # HELPER: Gradient function registered for the test copy op.
  # Inputs: the operation and the incoming gradient. Ignores the operation and returns the incoming
  # gradient unchanged. Registered as a side effect at import time; used by the gradient
  # registration tests.
  _ = op
  return x_grad


@ops.RegisterGradient("copy_override")
def _CopyOverrideGrad(op, x_grad):  # pylint: disable=invalid-name
  # [FUNCTION DOC]
  # HELPER: Gradient function registered under the override name for the copy op.
  # Inputs: the operation and the incoming gradient. Ignores the operation and returns the incoming
  # gradient unchanged. Registered at import time; used by the gradient override test.
  _ = op
  return x_grad


class RegistrationTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for the gradient registry: lookup of registered gradient functions and per-graph
  # gradient overrides.

  @test_util.run_deprecated_v1
  def testRegisterGradients(self):
    # [FUNCTION DOC]
    # Graph-mode test. Creates a copy op and asserts the registry returns the gradient function that
    # was registered for that op type.
    x = test_ops.float_output()
    y = test_ops.copy_op(x)
    fn = ops.get_gradient_function(y.op)
    self.assertEqual(_CopyGrad, fn)

  def testOverrideGradients(self):
    # [FUNCTION DOC]
    # Within a graph's gradient override map that maps the copy op type to the override name, creates a
    # copy op and asserts the registry returns the overriding gradient function.
    g = ops.Graph()
    with g.as_default():
      x = test_ops.float_output()
      with g.gradient_override_map({"CopyOp": "copy_override"}):
        y = test_ops.copy_op(x)
      fn = ops.get_gradient_function(y.op)
      self.assertEqual(_CopyOverrideGrad, fn)

  def testNonExistentOverride(self):
    # [FUNCTION DOC]
    # Maps the copy op type to an override name that is not registered, creates the op, and asserts
    # looking up its gradient function raises a lookup error mentioning the unknown name.
    g = ops.Graph()
    with g.as_default():
      x = test_ops.float_output()
      with g.gradient_override_map({"CopyOp": "unknown_override"}):
        y = test_ops.copy_op(x)
      with self.assertRaisesRegex(LookupError, "unknown_override"):
        ops.get_gradient_function(y.op)


class ComparisonTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for tensor comparison and membership behavior.

  def testMembershipAllowed(self):
    # [FUNCTION DOC]
    # Creates two distinct tensors and asserts list membership checks work (a tensor is in a list
    # containing it and not in a list containing only the other), i.e. membership is allowed and uses
    # identity-based semantics.
    g = ops.Graph()
    t1 = _apply_op(g, "FloatOutput", [], [dtypes.float32], name="myop1")
    t2 = _apply_op(g, "FloatOutput", [], [dtypes.float32], name="myop2")
    self.assertTrue(isinstance(t1, tensor_lib.Tensor))
    self.assertTrue(isinstance(t2, tensor_lib.Tensor))
    self.assertTrue(t1 in [t1])
    self.assertTrue(t1 not in [t2])


class ControlDependenciesTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for control dependency scopes: dependency propagation, nesting, clearing, conversion
  # of dependency objects, de-duplication and the manual-dependency marker attribute.

  @test_util.run_deprecated_v1
  def testBasic(self):
    # [FUNCTION DOC]
    # Graph-mode test. Within a control dependency on one constant, creates a constant, an identity of
    # another constant, and an identity of the first new constant. Asserts the first two get the
    # control input and the third does not because it is already dominated by a data dependency.
    g = ops.Graph()
    with g.as_default():
      # Creating unregistered ops with _apply_op() doesn't work with the C API
      # TODO(skyewm): address this more consistently. Possible solutions are
      # to use registered ops in all tests, create a way to register ops in
      # Python tests, or conditionally disable the op registration check in
      # the C API.
      a = constant_op.constant(1.0)
      b = constant_op.constant(1.0)
      with g.control_dependencies([a]):
        c = constant_op.constant(1.0)
        d = array_ops.identity(b)
        e = array_ops.identity(c)

    self.assertEqual(c.op.control_inputs, [a.op])
    self.assertEqual(d.op.control_inputs, [a.op])
    # e should be dominated by c.
    self.assertEqual(e.op.control_inputs, [])

  @test_util.run_in_graph_and_eager_modes
  def testEager(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. Passes a callable that counts its calls as a dependency. In eager
    # mode the callable is invoked once on scope entry; in graph mode a constant created by calling it
    # becomes a control input of the new op (inputs asserted in order) and the callable was called once.
    def future():
      # [FUNCTION DOC]
      # HELPER: Dependency callable for the eager/graph dependency test.
      # Increments a call counter attribute stored on itself and returns a constant of value two.
      future.calls += 1
      return constant_op.constant(2.0)
    future.calls = 0

    if context.executing_eagerly():
      a = constant_op.constant(1.0)
      b = future
      with ops.control_dependencies([a, b]):
        c = constant_op.constant(3.0)
      self.assertEqual(future.calls, 1)
    else:
      g = ops.Graph()
      with g.as_default():
        a = constant_op.constant(1.0)
        b = future()
        with g.control_dependencies([a, b]):
          c = constant_op.constant(3.0)
      self.assertEqual(c.op.control_inputs, [a.op, b.op])
      self.assertEqual(future.calls, 1)

  def testBasicWithConversion(self):
    # [FUNCTION DOC]
    # Defines an object with a graph-element conversion method returning a tensor, passes it as a
    # control dependency, and asserts the created op depends on the converted tensor's op.
    g = ops.Graph()
    a = _apply_op(g, "FloatOutput", [], [dtypes.float32])

    class ConvertibleObj(object):
      # [CLASS DOC]
      # HELPER: Local class convertible to a graph element, used to test dependency conversion.

      def _as_graph_element(self):
        # [FUNCTION DOC]
        # HELPER: Returns the enclosing test's tensor, allowing the object to be converted to a graph
        # element when used as a control dependency.
        return a

    with g.control_dependencies([ConvertibleObj()]):
      c = _apply_op(g, "FloatOutput", [], [dtypes.float32])

    self.assertEqual(c.op.control_inputs, [a.op])

  def testNested(self):
    # [FUNCTION DOC]
    # Asserts nested control dependency scopes accumulate: four deps in one scope equal four
    # single-dependency scopes nested, giving the same control inputs on the new op.
    g = ops.Graph()
    a_1 = _apply_op(g, "FloatOutput", [], [dtypes.float32])
    a_2 = _apply_op(g, "FloatOutput", [], [dtypes.float32])
    a_3 = _apply_op(g, "FloatOutput", [], [dtypes.float32])
    a_4 = _apply_op(g, "FloatOutput", [], [dtypes.float32])

    with g.control_dependencies([a_1, a_2, a_3, a_4]):
      b_1 = _apply_op(g, "FloatOutput", [], [dtypes.float32])

    with g.control_dependencies([a_1]):
      with g.control_dependencies([a_2]):
        with g.control_dependencies([a_3]):
          with g.control_dependencies([a_4]):
            b_2 = _apply_op(g, "FloatOutput", [], [dtypes.float32])

    self.assertItemsEqual([a_1.op, a_2.op, a_3.op, a_4.op],
                          b_1.op.control_inputs)
    self.assertItemsEqual(b_1.op.control_inputs, b_2.op.control_inputs)

  def testClear(self):
    # [FUNCTION DOC]
    # Checks clearing of control dependencies with a null scope inside nested scopes: ops created at
    # each level have the expected dependency sets (the innermost two, one, none, the outer pair, the
    # outer one, and none again after a second clear).
    g = ops.Graph()
    a_1 = _apply_op(g, "FloatOutput", [], [dtypes.float32])
    a_2 = _apply_op(g, "FloatOutput", [], [dtypes.float32])
    a_3 = _apply_op(g, "FloatOutput", [], [dtypes.float32])
    a_4 = _apply_op(g, "FloatOutput", [], [dtypes.float32])

    with g.control_dependencies([a_1]):
      with g.control_dependencies([a_2]):
        with g.control_dependencies(None):
          with g.control_dependencies([a_3]):
            with g.control_dependencies([a_4]):
              # deps [a_3, a_4]
              b_3_4 = _apply_op(g, "FloatOutput", [], [dtypes.float32])
            # deps = [a_3]
            b_3 = _apply_op(g, "FloatOutput", [], [dtypes.float32])
          # deps back to None
          b_none = _apply_op(g, "FloatOutput", [], [dtypes.float32])
        # deps back to [a_1, a_2]
        b_1_2 = _apply_op(g, "FloatOutput", [], [dtypes.float32])
      # deps back to [a_1]
      b_1 = _apply_op(g, "FloatOutput", [], [dtypes.float32])
      with g.control_dependencies(None):
        # deps are None again
        b_none2 = _apply_op(g, "FloatOutput", [], [dtypes.float32])

    self.assertItemsEqual([a_3.op, a_4.op], b_3_4.op.control_inputs)
    self.assertItemsEqual([a_3.op], b_3.op.control_inputs)
    self.assertItemsEqual([], b_none.op.control_inputs)
    self.assertItemsEqual([a_1.op, a_2.op], b_1_2.op.control_inputs)
    self.assertItemsEqual([a_1.op], b_1.op.control_inputs)
    self.assertItemsEqual([], b_none2.op.control_inputs)

  def testComplex(self):
    # [FUNCTION DOC]
    # Builds a four-level nested control-dependency structure with multiply ops (per-scope b, c, d, e
    # nodes, as described in a comment in the code). Asserts the control inputs: ops depending on
    # outer-scope tensors accumulate dependencies from scopes, ops already reachable through data
    # inputs do not get redundant control inputs, and chained ops depend only on their own scope.
    g = ops.Graph()

    # Usage pattern:
    # * Nodes a_i are constants defined at the outermost scope, and are used
    #   as control inputs for the ith nested scope.
    # * Nodes b_i are defined as Mul(a_3, a_4) at each scope.
    # * Nodes c_i are defined as Mul(a_1, b_1) at each scope.
    # * Nodes d_i are defined as Mul(b_i, c_i) at each scope.
    # * Nodes e_i are defined as Mul(e_i-1, e_i-1) at each scope i > 1.

    a_1 = _apply_op(g, "FloatOutput", [], [dtypes.float32])
    a_2 = _apply_op(g, "FloatOutput", [], [dtypes.float32])
    a_3 = _apply_op(g, "FloatOutput", [], [dtypes.float32])
    a_4 = _apply_op(g, "FloatOutput", [], [dtypes.float32])

    with g.control_dependencies([a_1]):
      b_1 = _apply_op(g, "TwoFloatInputsFloatOutput", [a_3, a_4],
                      [dtypes.float32])
      c_1 = _apply_op(g, "TwoFloatInputsFloatOutput", [a_1, b_1],
                      [dtypes.float32])
      d_1 = _apply_op(g, "TwoFloatInputsFloatOutput", [b_1, c_1],
                      [dtypes.float32])
      e_1 = _apply_op(g, "FloatOutput", [], [dtypes.float32])
      with g.control_dependencies([a_2]):
        b_2 = _apply_op(g, "TwoFloatInputsFloatOutput", [a_3, a_4],
                        [dtypes.float32])
        c_2 = _apply_op(g, "TwoFloatInputsFloatOutput", [a_1, b_1],
                        [dtypes.float32])
        d_2 = _apply_op(g, "TwoFloatInputsFloatOutput", [b_2, c_2],
                        [dtypes.float32])
        e_2 = _apply_op(g, "TwoFloatInputsFloatOutput", [e_1, e_1],
                        [dtypes.float32])
        with g.control_dependencies([a_3]):
          b_3 = _apply_op(g, "TwoFloatInputsFloatOutput", [a_3, a_4],
                          [dtypes.float32])
          c_3 = _apply_op(g, "TwoFloatInputsFloatOutput", [a_1, b_1],
                          [dtypes.float32])
          d_3 = _apply_op(g, "TwoFloatInputsFloatOutput", [b_3, c_3],
                          [dtypes.float32])
          e_3 = _apply_op(g, "TwoFloatInputsFloatOutput", [e_2, e_2],
                          [dtypes.float32])
          with g.control_dependencies([a_4]):
            b_4 = _apply_op(g, "TwoFloatInputsFloatOutput", [a_3, a_4],
                            [dtypes.float32])
            c_4 = _apply_op(g, "TwoFloatInputsFloatOutput", [a_1, b_1],
                            [dtypes.float32])
            d_4 = _apply_op(g, "TwoFloatInputsFloatOutput", [b_4, c_4],
                            [dtypes.float32])
            e_4 = _apply_op(g, "TwoFloatInputsFloatOutput", [e_3, e_3],
                            [dtypes.float32])

    self.assertItemsEqual([a_1.op], b_1.op.control_inputs)
    self.assertItemsEqual([a_1.op, a_2.op], b_2.op.control_inputs)
    self.assertItemsEqual([a_1.op, a_2.op], b_3.op.control_inputs)
    self.assertItemsEqual([a_1.op, a_2.op], b_4.op.control_inputs)

    self.assertItemsEqual([], c_1.op.control_inputs)
    self.assertItemsEqual([a_2.op], c_2.op.control_inputs)
    self.assertItemsEqual([a_2.op, a_3.op], c_3.op.control_inputs)
    self.assertItemsEqual([a_2.op, a_3.op, a_4.op], c_4.op.control_inputs)

    self.assertItemsEqual([], d_1.op.control_inputs)
    self.assertItemsEqual([], d_2.op.control_inputs)
    self.assertItemsEqual([], d_3.op.control_inputs)
    self.assertItemsEqual([], d_4.op.control_inputs)

    self.assertItemsEqual([a_1.op], e_1.op.control_inputs)
    self.assertItemsEqual([a_2.op], e_2.op.control_inputs)
    self.assertItemsEqual([a_3.op], e_3.op.control_inputs)
    self.assertItemsEqual([a_4.op], e_4.op.control_inputs)

  def testRepeatedDependency(self):
    # [FUNCTION DOC]
    # Depends on two outputs of the same operation in nested scopes and asserts the new ops each have
    # a single control input, the producing op, with no duplicates.
    g = ops.Graph()
    a = g.create_op("TwoFloatOutputs", [], [dtypes.float32, dtypes.float32])
    a_0, a_1 = a.outputs
    with g.control_dependencies([a_0]):
      b = _apply_op(g, "FloatOutput", [], [dtypes.float32])
      with g.control_dependencies([a_1]):
        c = _apply_op(g, "FloatOutput", [], [dtypes.float32])

    self.assertEqual(b.op.control_inputs, [a])
    self.assertEqual(c.op.control_inputs, [a])

  def testNoControlDependencyWithDataDependency(self):
    # [FUNCTION DOC]
    # Asserts that when the new op already has a data dependency on the dependency tensor (an identity
    # of it), no extra control input is added.
    g = ops.Graph()
    a = _apply_op(g, "FloatOutput", [], [dtypes.float32])
    with g.control_dependencies([a]):
      b = _apply_op(g, "Identity", [a], [dtypes.float32])

    self.assertEqual(b.op.control_inputs, [])

  def testMonitoringAttributeAddedWhenUsingManualControlDep(self):
    # [FUNCTION DOC]
    # Checks the monitoring attribute that records use of the manual control-dependency API. After
    # using manual dependencies, the control input is recorded, the new op does not carry the attribute
    # (reading it raises a value error), but the dependency source op does carry it with a true value.
    # Also holds when the manual dependency was pruned because of a data dependency.
    g = ops.Graph()
    a = _apply_op(g, "FloatOutput", [], [dtypes.float32])
    b = _apply_op(g, "FloatOutput", [], [dtypes.float32])
    with g.control_dependencies([a]):
      c = _apply_op(g, "Identity", [b], [dtypes.float32])

    with g.control_dependencies([b]):
      d = _apply_op(g, "Identity", [b], [dtypes.float32])

    # Validate that the monitoring attribute is set to track usage of the
    # `control_dependencies(...)` API.
    self.assertEqual(c.op.control_inputs, [a.op])
    with self.assertRaises(ValueError):
      c.op.get_attr("_has_manual_control_dependencies")
    self.assertEqual(a.op.get_attr("_has_manual_control_dependencies"), True)

    # Validate that the monitoring attribute is set to track usage of the
    # `control_dependencies(...)` API even when the manual control deps actually
    # happened to be pruned at runtime.
    self.assertEqual(d.op.control_inputs, [])
    with self.assertRaises(ValueError):
      d.op.get_attr("_has_manual_control_dependencies")
    self.assertEqual(b.op.get_attr("_has_manual_control_dependencies"), True)


class OpScopeTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for the name-scope helper functions of the ops module (the older style that also
  # accepts a default name and a list of values) and the re-entrant v2 scope.

  @test_util.run_in_graph_and_eager_modes
  def testNames(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. Exercises name scopes with skip-on-eager disabled: nesting,
    # null and empty names, names ending in a slash being used as-is, double slashes, a bare slash, and
    # names with embedded double slashes. Asserts the scope strings returned.
    with ops.name_scope("foo", skip_on_eager=False) as foo:
      self.assertEqual("foo/", foo)
      with ops.name_scope("foo2", skip_on_eager=False) as foo2:
        self.assertEqual("foo/foo2/", foo2)
      with ops.name_scope(None, skip_on_eager=False) as empty1:
        self.assertEqual("", empty1)
        with ops.name_scope("foo3", skip_on_eager=False) as foo3:
          self.assertEqual("foo3/", foo3)
      with ops.name_scope("", skip_on_eager=False) as empty2:
        self.assertEqual("", empty2)
    with ops.name_scope("foo/", skip_on_eager=False) as outer_foo:
      self.assertEqual("foo/", outer_foo)
      with ops.name_scope("", skip_on_eager=False) as empty3:
        self.assertEqual("", empty3)
      with ops.name_scope("foo4", skip_on_eager=False) as foo4:
        self.assertEqual("foo/foo4/", foo4)
      with ops.name_scope("foo5//", skip_on_eager=False) as foo5:
        self.assertEqual("foo5//", foo5)
        with ops.name_scope("foo6", skip_on_eager=False) as foo6:
          self.assertEqual("foo5//foo6/", foo6)
      with ops.name_scope("/", skip_on_eager=False) as foo7:
        self.assertEqual("/", foo7)
      with ops.name_scope("//", skip_on_eager=False) as foo8:
        self.assertEqual("//", foo8)
      with ops.name_scope("a//b/c", skip_on_eager=False) as foo9:
        self.assertEqual("foo/a//b/c/", foo9)
    with ops.name_scope("a//b/c", skip_on_eager=False) as foo10:
      self.assertEqual("a//b/c/", foo10)

  @test_util.run_in_graph_and_eager_modes
  def testEagerDefaultScopeName(self):
    # [FUNCTION DOC]
    # Runs in both modes. Checks that a scope with no name and a default name uses the default, and
    # that nested default names compose as default slash default2.
    with ops.name_scope(None, "default", skip_on_eager=False) as scope:
      self.assertEqual(scope, "default/")
      with ops.name_scope(None, "default2", skip_on_eager=False) as scope2:
        self.assertEqual(scope2, "default/default2/")

  @test_util.run_in_graph_and_eager_modes
  def testNameScopeV2IsReEntrant(self):
    # [FUNCTION DOC]
    # Runs in both modes. Checks the v2 name scope object can be re-entered: reusing the same scope
    # object nests it again (foo then foo slash foo), and combined with another scope composes names.
    foo = ops.name_scope_v2("foo")
    bar = ops.name_scope_v2("bar")
    with foo as scope_name:
      self.assertEqual("foo/", scope_name)
      with foo as scope_name:
        self.assertEqual("foo/foo/", scope_name)
      with bar as scope_name:
        self.assertEqual("foo/bar/", scope_name)
        with foo as scope_name:
          self.assertEqual("foo/bar/foo/", scope_name)
    with bar as scope_name:
      self.assertEqual("bar/", scope_name)

  @test_util.run_deprecated_v1
  def testNoScopeName(self):
    # [FUNCTION DOC]
    # Graph-mode test. Asserts that opening a name scope with a null name and no default name, given
    # only values, raises a value error (both keyword and positional forms).
    g0 = ops.Graph()
    values = [
        g0.create_op("A", [], [dtypes.float32]),
        g0.create_op("B", [], [dtypes.float32])
    ]
    with self.assertRaises(ValueError):
      with ops.name_scope(None, values=values):
        pass
    with self.assertRaises(ValueError):
      with ops.name_scope(None, None, values):
        pass

  @test_util.run_deprecated_v1
  def testEmptyScopeName(self):
    # [FUNCTION DOC]
    # Graph-mode test. Asserts an empty scope name yields an empty scope string and keeps the same
    # default graph, including when a default name is given.
    g0 = ops.Graph()
    a = g0.create_op("A", [], [dtypes.float32])
    b = g0.create_op("B", [], [dtypes.float32])
    with ops.name_scope("", values=[a, b]) as scope:
      self.assertEqual("", scope)
      self.assertEqual(g0, ops.get_default_graph())
    with ops.name_scope("", "my_default_scope", [a, b]) as scope:
      self.assertEqual("", scope)
      self.assertEqual(g0, ops.get_default_graph())

  @test_util.run_deprecated_v1
  def testDefaultScopeName(self):
    # [FUNCTION DOC]
    # Graph-mode test. Asserts a given scope name wins over the default name; with a null name the
    # default name is used; passing a list of values as the second argument (wrong position) raises a
    # type error. Default graph stays the same.
    g0 = ops.Graph()
    a = g0.create_op("A", [], [dtypes.float32])
    b = g0.create_op("B", [], [dtypes.float32])
    scope_name = "my_scope"
    default_scope_name = "my_default_scope"
    with ops.name_scope(scope_name, default_scope_name, [a, b]) as scope:
      self.assertEqual("%s/" % scope_name, scope)
      self.assertEqual(g0, ops.get_default_graph())
    with ops.name_scope(None, default_scope_name, [a, b]) as scope:
      self.assertEqual("%s/" % default_scope_name, scope)
      self.assertEqual(g0, ops.get_default_graph())
    with self.assertRaises(TypeError):
      with ops.name_scope(scope_name, [a, b]):
        pass

  def _testGraphElements(self, graph_elements):
    # [FUNCTION DOC]
    # HELPER: Shared check for the scope tests on tensors, sparse tensors and variables.
    # Input: a list of graph elements. Opens a named scope with those values and asserts the scope
    # string and that the default graph is the elements' graph; then asserts that adding an element
    # from a different graph raises a value error.
    scope_name = "my_scope"
    with ops.name_scope(scope_name, values=graph_elements) as scope:
      self.assertEqual("%s/" % scope_name, scope)
      self.assertEqual(graph_elements[0].graph, ops.get_default_graph())
    g1 = ops.Graph()
    a = g1.create_op("A", [], [dtypes.float32])
    with self.assertRaises(ValueError):
      with ops.name_scope(scope_name, values=graph_elements + [a]):
        pass

  @test_util.run_in_graph_and_eager_modes
  def testGetCurrentNameScope(self):
    # [FUNCTION DOC]
    # Runs in both modes. Asserts the current name scope string is empty at first and tracks nested v2
    # scopes (aaa, then aaa slash bbb) and returns to the outer values after leaving each scope.
    self.assertEqual(ops.get_current_name_scope(), "")
    with ops.name_scope_v2("aaa"):
      self.assertEqual(ops.get_current_name_scope(), "aaa")
      with ops.name_scope_v2("bbb"):
        self.assertEqual(ops.get_current_name_scope(), "aaa/bbb")
      self.assertEqual(ops.get_current_name_scope(), "aaa")
    self.assertEqual(ops.get_current_name_scope(), "")

  @test_util.run_deprecated_v1
  def testTensor(self):
    # [FUNCTION DOC]
    # Graph-mode test. Creates two ops and runs the shared graph-elements check on their outputs.
    g0 = ops.Graph()
    a = g0.create_op("A", [], [dtypes.float32])
    b = g0.create_op("B", [], [dtypes.float32])
    self._testGraphElements([a, b])

  @test_util.run_deprecated_v1
  def testSparseTensor(self):
    # [FUNCTION DOC]
    # Graph-mode test. Creates two ops and a sparse tensor (built from op outputs of integer, float and
    # integer type) and runs the shared graph-elements check on the list of op, sparse tensor and op.
    g0 = ops.Graph()
    a = g0.create_op("A", [], [dtypes.float32])
    b = g0.create_op("B", [], [dtypes.float32])
    sparse = sparse_tensor.SparseTensor(
        _apply_op(g0, "Int64Output", [], [dtypes.int64]),
        _apply_op(g0, "FloatOutput", [], [dtypes.float32]),
        _apply_op(g0, "Int64Output", [], [dtypes.int64]))
    self._testGraphElements([a, sparse, b])

  @test_util.run_deprecated_v1
  def testVariable(self):
    # [FUNCTION DOC]
    # Graph-mode test. Creates a variable in a graph and two ops, then runs the shared graph-elements
    # check on the list of op, variable and op.
    g0 = ops.Graph()
    with g0.as_default():
      variable = variables.Variable([1.0])
    a = g0.create_op("A", [], [dtypes.float32])
    b = g0.create_op("B", [], [dtypes.float32])
    self._testGraphElements([a, variable, b])


class InitScopeTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for the initialization scope, which lifts operations out of function-building graphs
  # into the outermost non-function context (or eager context), clearing control dependencies and
  # preserving devices and name scopes where documented.

  def testClearsControlDependencies(self):
    # [FUNCTION DOC]
    # Nests control dependency scopes in a graph, enters the init scope, and nests more scopes inside.
    # Asserts the init scope clears inherited control dependencies (inner ops depend only on scopes
    # opened inside it) and that the outer dependencies are restored on leaving the init scope.
    g = ops.Graph()
    a_1 = _apply_op(g, "FloatOutput", [], [dtypes.float32])
    a_2 = _apply_op(g, "FloatOutput", [], [dtypes.float32])
    a_3 = _apply_op(g, "FloatOutput", [], [dtypes.float32])
    a_4 = _apply_op(g, "FloatOutput", [], [dtypes.float32])

    with g.as_default():
      with g.control_dependencies([a_1]):
        with g.control_dependencies([a_2]):
          with ops.init_scope():
            with g.control_dependencies([a_3]):
              with g.control_dependencies([a_4]):
                # deps [a_3, a_4]
                b_3_4 = _apply_op(g, "FloatOutput", [], [dtypes.float32])
              # deps = [a_3]
              b_3 = _apply_op(g, "FloatOutput", [], [dtypes.float32])
            # deps back to None
            b_none = _apply_op(g, "FloatOutput", [], [dtypes.float32])
          # deps back to [a_1, a_2]
          b_1_2 = _apply_op(g, "FloatOutput", [], [dtypes.float32])
        # deps back to [a_1]
        b_1 = _apply_op(g, "FloatOutput", [], [dtypes.float32])
        with ops.init_scope():
          # deps are None again
          b_none2 = _apply_op(g, "FloatOutput", [], [dtypes.float32])

    self.assertItemsEqual([a_3.op, a_4.op], b_3_4.op.control_inputs)
    self.assertItemsEqual([a_3.op], b_3.op.control_inputs)
    self.assertItemsEqual([], b_none.op.control_inputs)
    self.assertItemsEqual([a_1.op, a_2.op], b_1_2.op.control_inputs)
    self.assertItemsEqual([a_1.op], b_1.op.control_inputs)
    self.assertItemsEqual([], b_none2.op.control_inputs)

  def testLiftsOpsFromFunctions(self):
    # [FUNCTION DOC]
    # Builds three graphs where two are marked as building functions, nests them as default, and
    # creates a constant under the init scope. Asserts the op lands in the outermost graph only and the
    # two function graphs stay empty.
    g0 = ops.Graph()
    g1 = ops.Graph()
    g1._building_function = True  # pylint: disable=protected-access
    g2 = ops.Graph()
    g2._building_function = True  # pylint: disable=protected-access

    with g0.as_default():
      with g1.as_default():
        with g2.as_default():
          with ops.init_scope():
            _ = constant_op.constant(1.0)

    self.assertLen(g2.get_operations(), 0)
    self.assertLen(g1.get_operations(), 0)
    self.assertLen(g0.get_operations(), 1)

  def testPreservesDevices(self):
    # [FUNCTION DOC]
    # Checks that devices set in the function-building graph are preserved by the init scope while
    # devices from the outer graph apply once the function graph's own scopes are left: constants have
    # the GPU device inside it, no device in a bare function graph, and the CPU device from the outer
    # graph when created under the init scope there.
    g0 = ops.Graph()
    with g0.as_default(), ops.device("CPU:0"):
      g1 = ops.Graph()
      g1._building_function = True  # pylint: disable=protected-access
      with g1.as_default():
        with ops.device("GPU:0"):
          with ops.init_scope():
            # init_scope should preserve device set under `g1`.
            on_gpu = constant_op.constant(1.0)
            self.assertEqual(on_gpu.device, "/device:GPU:0")
          still_on_gpu = constant_op.constant(1.0)
          self.assertEqual(still_on_gpu.device, "/device:GPU:0")
        blank = constant_op.constant(1.0)
        self.assertEqual(blank.device, "")
        with ops.init_scope():
          now_on_cpu = constant_op.constant(1.0)
          self.assertEqual(now_on_cpu.device, "/device:CPU:0")
      on_cpu = constant_op.constant(1.0)
      self.assertEqual(on_cpu.device, "/device:CPU:0")

  def testComposes(self):
    # [FUNCTION DOC]
    # Checks that nested init scopes compose over several graphs, including one graph not marked as
    # building a function. Asserts which graph is the default during the scopes and the final operation
    # counts per graph (the non-function graph gets one op, outer graph two, function graphs none).
    g0 = ops.Graph()
    g1 = ops.Graph()
    g1._building_function = True  # pylint: disable=protected-access
    g2 = ops.Graph()
    g2._building_function = True  # pylint: disable=protected-access
    g3 = ops.Graph()
    g3._building_function = False  # pylint: disable=protected-access

    with g0.as_default():
      with g1.as_default():
        with ops.init_scope():
          # This op should be lifted into g0.
          _ = constant_op.constant(1.0)
          self.assertIs(g0, ops.get_default_graph())
          self.assertLen(g2.get_operations(), 0)
          self.assertLen(g1.get_operations(), 0)
          self.assertLen(g0.get_operations(), 1)
        with g2.as_default():
          with ops.init_scope():
            # This op should be lifted into g0.
            _ = constant_op.constant(1.0)
            self.assertIs(g0, ops.get_default_graph())
            with g3.as_default():
              with ops.init_scope():
                # This op should be lifted into g3, because g3 is not building a
                # function.
                _ = constant_op.constant(1.0)
                self.assertIs(g3, ops.get_default_graph())

    self.assertLen(g3.get_operations(), 1)
    self.assertLen(g2.get_operations(), 0)
    self.assertLen(g1.get_operations(), 0)
    self.assertLen(g0.get_operations(), 2)

  def testEscapesToEagerContext(self):
    # [FUNCTION DOC]
    # Within eager mode and then graph mode with a function-building graph, asserts that the init scope
    # escapes to eager execution, and that on exit the graph is the default again and eager execution
    # is off.
    g = ops.Graph()
    g._building_function = True  # pylint: disable=protected-access
    with context.eager_mode():
      with context.graph_mode():
        with g.as_default():
          with ops.init_scope():
            # Because g is building a function, init_scope should
            # escape out to the eager context.
            self.assertTrue(context.executing_eagerly())
          # g should be reinstated as the default graph, and the
          # graph context should be re-entered.
          self.assertIs(g, ops.get_default_graph())
          self.assertFalse(context.executing_eagerly())

  def testStaysInEagerWhenOnlyEagerContextActive(self):
    # [FUNCTION DOC]
    # Asserts that using the init scope while only the eager context is active stays in eager mode.
    with context.eager_mode():
      with ops.init_scope():
        self.assertTrue(context.eager_mode())
      self.assertTrue(context.eager_mode())

  @test_util.run_v1_only("b/120545219")
  def testFallsBackToGlobalGraphWhenAllGraphsAreBuildingFunctions(self):
    # [FUNCTION DOC]
    # Runs in v1 only. In graph mode, resets the default graph, remembers the global graph, and enters
    # a function-building graph. Asserts the init scope pushes onto the default graph stack and ops
    # created there land in the global graph, and that the stack is restored (and empty after leaving).
    # Uses private stack state.
    with context.graph_mode():
      ops.reset_default_graph()
      # This doesn't push anything onto the graph stack, but it does
      # set the stack's global graph.
      global_graph = ops.get_default_graph()
      fn_graph = ops.Graph()

      # pylint: disable=protected-access
      fn_graph._building_function = True
      self.assertLen(ops._default_graph_stack.stack, 0)
      with fn_graph.as_default():
        self.assertLen(ops._default_graph_stack.stack, 1)
        with ops.init_scope():
          self.assertGreater(len(ops._default_graph_stack.stack), 1)
          dummy = constant_op.constant(1.0)
        self.assertLen(ops._default_graph_stack.stack, 1)
      # Note that the global graph is _not_ on the graph stack.
      self.assertLen(ops._default_graph_stack.stack, 0)
      # Ensure that `dummy` was added to the global graph.
      self.assertEqual(global_graph, dummy.graph)
      # pylint: enable=protected-access

  def testInstallsDefaultGraphWhenGraphStackIsEmptyInGraphMode(self):
    # [FUNCTION DOC]
    # Graph mode with an empty default graph stack: asserts entering the init scope installs a default
    # graph on the stack and removes it on exit.
    with context.graph_mode():
      # pylint: disable=protected-access
      self.assertLen(ops._default_graph_stack.stack, 0)
      with ops.init_scope():
        self.assertGreater(len(ops._default_graph_stack.stack), 0)
      self.assertLen(ops._default_graph_stack.stack, 0)
      # pylint: enable=protected-access

  def testPreservesNameScopeInGraphConstruction(self):
    # [FUNCTION DOC]
    # Asserts the current name scope (inner) is preserved by the init scope when building a function
    # graph, and that the outer scope is empty after leaving.
    with ops.Graph().as_default():
      function_graph = ops.Graph()
      with function_graph.as_default():
        with ops.name_scope("inner", skip_on_eager=False), ops.init_scope():
          self.assertEqual(ops.get_name_scope(), "inner")
      self.assertEqual(ops.get_name_scope(), "")

  def testEnteringGraphFromEagerIsSticky(self):
    # [FUNCTION DOC]
    # Eager mode. Asserts that entering a graph and then the init scope leaves execution in graph mode
    # (not eager) with that graph as default, i.e. entering a graph from eager mode is sticky.
    with context.eager_mode():
      g = ops.Graph()
      with g.as_default():
        with ops.init_scope():
          self.assertFalse(context.executing_eagerly())
          self.assertEqual(g, ops.get_default_graph())

  def testMixGraphEager(self):
    # [FUNCTION DOC]
    # Mixing eager and graph tensors. In eager mode creates a constant, then in a new graph asserts
    # that using the eager tensor in an add raises a runtime error about capturing an eager tensor;
    # afterwards using a tensor created in the graph outside it raises a type error.
    with context.eager_mode():
      c = constant_op.constant(1.0)
      with ops.Graph().as_default():
        with self.assertRaisesRegex(RuntimeError,
                                    "Attempting to capture an EagerTensor"):
          math_ops.add(c, c)
        c2 = constant_op.constant(2.0)
      with self.assertRaises(TypeError):
        math_ops.add(c2, c2)

  def testPreservesNameScopeInEagerExecution(self):
    # [FUNCTION DOC]
    # Checks name scope handling of the init scope in eager mode and inside a traced function: inside
    # an eager scope the context scope name is inner with a trailing slash, otherwise the name scope is
    # inner. Runs the function directly and compiled and asserts the scope is empty afterwards.
    with context.eager_mode():
      def foo():
        # [FUNCTION DOC]
        # HELPER: Function exercised directly and as a traced function by the name-scope test.
        # Opens a name scope named inner and then an init scope; when executing eagerly asserts the context
        # scope name is inner with a trailing slash, otherwise asserts the name scope is inner. Returns
        # nothing.
        with ops.name_scope("inner", skip_on_eager=False), ops.init_scope():
          if context.executing_eagerly():
            # A trailing slash is always appended when eager execution is
            # enabled.
            self.assertEqual(context.context().scope_name, "inner/")
          else:
            self.assertEqual(ops.get_name_scope(), "inner")

      foo()
      self.assertEqual(ops.get_name_scope(), "")
      foo_compiled = def_function.function(foo)
      foo_compiled()
      self.assertEqual(ops.get_name_scope(), "")

  def testExecutingEagerlyOutsideFunctions(self):
    # [FUNCTION DOC]
    # Asserts the function that reports whether eager execution is enabled outside functions: false in
    # graph mode (and when evaluated in a session), true in eager mode, and false inside a new default
    # graph even while eager mode is enabled.

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # HELPER: Traced function body that returns the result of the check for executing eagerly outside
      # functions. Takes no arguments.
      return ops.executing_eagerly_outside_functions()

    with context.graph_mode():
      self.assertFalse(ops.executing_eagerly_outside_functions())
      with session.Session():
        # Need self.evaluate for these as the return type of functions is
        # tensors.
        self.assertFalse(self.evaluate(f()))

    with context.eager_mode():
      self.assertTrue(ops.executing_eagerly_outside_functions())
      self.assertTrue(f())

      with ops.Graph().as_default():
        self.assertFalse(ops.executing_eagerly_outside_functions())
        with session.Session():
          self.assertFalse(self.evaluate(f()))


class GraphTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for the Graph class: default graph management, context manager behavior, feeding and
  # fetching restrictions, element conversion, garbage collection and recovery after errors.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture setup run before each test; resets the default graph to a fresh one. Takes no arguments,
    # returns nothing, modifies global default graph state. Note: does not call the base class setup.
    ops.reset_default_graph()

  def _AssertDefault(self, expected):
    # [FUNCTION DOC]
    # HELPER: Asserts the default graph is the expected graph (identity). Input: expected graph.
    self.assertIs(expected, ops.get_default_graph())

  def testResetDefaultGraphNesting(self):
    # [FUNCTION DOC]
    # Asserts that resetting the default graph while a graph context is active raises an assertion
    # error.
    g0 = ops.Graph()
    with self.assertRaises(AssertionError):
      with g0.as_default():
        ops.reset_default_graph()

  def testGraphContextManagerCancelsEager(self):
    # [FUNCTION DOC]
    # Asserts that entering a graph as default while in eager mode turns eager execution off inside it.
    with context.eager_mode():
      with ops.Graph().as_default():
        self.assertFalse(context.executing_eagerly())

  def testGraphContextManager(self):
    # [FUNCTION DOC]
    # Asserts that entering the graph context manager returns the same graph object.
    g0 = ops.Graph()
    with g0.as_default() as g1:
      self.assertIs(g0, g1)

  def testDefaultGraph(self):
    # [FUNCTION DOC]
    # Checks default graph tracking: before any graph is made the default graph is a global one and
    # the has-default flag is false; creating a graph or context manager does not change it; inside the
    # with-block the graph is default and has-default is true; nested graphs stack and unwind correctly.
    orig = ops.get_default_graph()
    self.assertFalse(ops.has_default_graph())
    self._AssertDefault(orig)
    g0 = ops.Graph()
    self.assertFalse(ops.has_default_graph())
    self._AssertDefault(orig)
    context_manager_0 = g0.as_default()
    self.assertFalse(ops.has_default_graph())
    self._AssertDefault(orig)
    with context_manager_0 as g0:
      self._AssertDefault(g0)
      with ops.Graph().as_default() as g1:
        self.assertTrue(ops.has_default_graph())
        self._AssertDefault(g1)
      self._AssertDefault(g0)
    self._AssertDefault(orig)
    self.assertFalse(ops.has_default_graph())

  def testPreventFeeding(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the constant is created in the default graph while the feedability calls are made on
    # a separate new graph, so the test exercises the graph with a tensor from another graph.
    # Asserts a tensor can be fed by default, and cannot be fed after feeding is prevented for it.
    g = ops.Graph()
    a = constant_op.constant(2.0)
    self.assertTrue(g.is_feedable(a))
    g.prevent_feeding(a)
    self.assertFalse(g.is_feedable(a))

  @test_util.run_deprecated_v1
  def testPreventFetching(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the constant is created in the default graph while the fetchability calls are made
    # on a separate new graph, so the tensor belongs to a different graph than the one queried.
    # Graph-mode test. Asserts a tensor is fetchable by default and not fetchable after fetching is
    # prevented for its op.
    g = ops.Graph()
    a = constant_op.constant(2.0)
    self.assertTrue(g.is_fetchable(a))
    g.prevent_fetching(a.op)
    self.assertFalse(g.is_fetchable(a))

  def testAsGraphElementConversions(self):
    # [FUNCTION DOC]
    # Defines a convertible object whose conversion method returns a tensor name and a plain object
    # without it. Asserts the graph converts the first to the named tensor and raises a type error
    # for the second.

    class ConvertibleObj(object):
      # [CLASS DOC]
      # HELPER: Local class with a graph-element conversion method returning the output name string.

      def _as_graph_element(self):
        # [FUNCTION DOC]
        # HELPER: Returns the name string of the float output tensor so the graph can look it up.
        return "FloatOutput:0"

    class NonConvertibleObj(object):
      # [CLASS DOC]
      # HELPER: Local class with no conversion method, used to check type error on conversion.

      pass

    g = ops.Graph()
    a = _apply_op(g, "FloatOutput", [], [dtypes.float32])
    self.assertEqual(a, g.as_graph_element(ConvertibleObj()))
    with self.assertRaises(TypeError):
      g.as_graph_element(NonConvertibleObj())

  # Regression test against creating custom __del__ functions in classes
  # involved in cyclic references, e.g. Graph and Operation. (Python won't gc
  # cycles that require calling a __del__ method, because the __del__ method can
  # theoretically increase the object's refcount to "save" it from gc, and any
  # already-deleted objects in the cycle would have be to restored.)
  def testGarbageCollected(self):
    # [FUNCTION DOC]
    # Regression test against custom destructors in graph and operation classes (cyclic references
    # that the collector could not free). Builds a graph, session and ops, evaluates, deletes all
    # references, forces garbage collection and asserts a weak reference to the graph is cleared.
    # Create a graph we can delete and a weak reference to monitor if it's gc'd
    g = ops.Graph()
    g_ref = weakref.ref(g)
    # Create some ops
    with g.as_default():
      a = constant_op.constant(2.0)
      b = constant_op.constant(3.0)
      c = math_ops.add(a, b)
    # Create a session we can delete
    with session.Session(graph=g) as sess:
      self.evaluate(c)
    # Delete all references and trigger gc
    del g
    del a
    del b
    del c
    del sess
    gc.collect()
    self.assertIsNone(g_ref())

  def testRunnableAfterInvalidShape(self):
    # [FUNCTION DOC]
    # Asserts that after an op with incompatible shapes raises a value error, the default graph is
    # still usable: a constant can be created and evaluated in a session.
    with ops.Graph().as_default():
      with self.assertRaises(ValueError):
        math_ops.add([1, 2], [1, 2, 3])
      a = constant_op.constant(1)
      with session.Session() as sess:
        self.evaluate(a)

  def testRunnableAfterInvalidShapeWithKernelLabelMap(self):
    # [FUNCTION DOC]
    # Same as the previous test but with the kernel label map active during the failing op creation
    # (a kernel-label-required op). Asserts the value error and that the graph remains runnable.
    g = ops.Graph()
    with g.as_default():
      with g._kernel_label_map({"KernelLabelRequired": "overload_1"}):
        with self.assertRaises(ValueError):
          test_ops.kernel_label_required(1)
      a = constant_op.constant(1)
      with session.Session() as sess:
        self.evaluate(a)


class AttrScopeTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for graph attribute scopes that add attributes to ops created inside them.

  def _get_test_attrs(self):
    # [FUNCTION DOC]
    # HELPER: Creates a no-op and reads attributes named A and B (as text), returning a pair with
    # null for any attribute that is missing. Takes no arguments.
    x = gen_control_flow_ops.no_op()
    try:
      a = compat.as_text(x.get_attr("_A"))
    except ValueError:
      a = None
    try:
      b = compat.as_text(x.get_attr("_B"))
    except ValueError:
      b = None
    return (a, b)

  @test_util.run_deprecated_v1
  def testNoLabel(self):
    # [FUNCTION DOC]
    # Graph-mode test. Asserts that with no attribute scope active neither attribute is set.
    with self.cached_session():
      self.assertAllEqual((None, None), self._get_test_attrs())

  @test_util.run_deprecated_v1
  def testLabelMap(self):
    # [FUNCTION DOC]
    # Graph-mode test. Nests attribute scopes: setting A, then clearing A and setting B, then
    # overriding A, and checks the pair at each level and after leaving each scope (including null
    # values restoring and the outer values returning).
    with self.cached_session() as sess:
      a1 = self._get_test_attrs()
      with sess.graph._attr_scope({
          "_A": attr_value_pb2.AttrValue(s=compat.as_bytes("foo"))
      }):
        a2 = self._get_test_attrs()
        with sess.graph._attr_scope({
            "_A": None,
            "_B": attr_value_pb2.AttrValue(s=compat.as_bytes("bar"))
        }):
          a3 = self._get_test_attrs()
          with sess.graph._attr_scope({
              "_A": attr_value_pb2.AttrValue(s=compat.as_bytes("baz"))
          }):
            a4 = self._get_test_attrs()
          a5 = self._get_test_attrs()
        a6 = self._get_test_attrs()
      a7 = self._get_test_attrs()

      self.assertAllEqual((None, None), a1)
      self.assertAllEqual(("foo", None), a2)
      self.assertAllEqual((None, "bar"), a3)
      self.assertAllEqual(("baz", "bar"), a4)
      self.assertAllEqual((None, "bar"), a5)
      self.assertAllEqual(("foo", None), a6)
      self.assertAllEqual((None, None), a7)


class KernelLabelTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for kernel label maps, which choose among kernel implementations by label.

  @test_util.run_deprecated_v1
  def testNoLabel(self):
    # [FUNCTION DOC]
    # Graph-mode test. Evaluates the kernel-label op without a label map and asserts the default label
    # string result.
    with self.cached_session():
      self.assertAllEqual(b"My label is: default",
                          test_ops.kernel_label().eval())

  @test_util.run_deprecated_v1
  def testLabelMap(self):
    # [FUNCTION DOC]
    # Graph-mode test. Nests label maps (overload one, overload two, empty string) and asserts which
    # label each op got when evaluated: default, overload one, overload two, default (empty resets),
    # overload one again, and default after all scopes close.
    with self.cached_session() as sess:
      default_1 = test_ops.kernel_label()
      # pylint: disable=protected-access
      with sess.graph._kernel_label_map({"KernelLabel": "overload_1"}):
        overload_1_1 = test_ops.kernel_label()
        with sess.graph._kernel_label_map({"KernelLabel": "overload_2"}):
          overload_2 = test_ops.kernel_label()
          with sess.graph._kernel_label_map({"KernelLabel": ""}):
            default_2 = test_ops.kernel_label()
        overload_1_2 = test_ops.kernel_label()
      # pylint: enable=protected-access
      default_3 = test_ops.kernel_label()

      self.assertAllEqual(b"My label is: default", self.evaluate(default_1))
      self.assertAllEqual(b"My label is: default", self.evaluate(default_2))
      self.assertAllEqual(b"My label is: default", self.evaluate(default_3))
      self.assertAllEqual(b"My label is: overload_1",
                          self.evaluate(overload_1_1))
      self.assertAllEqual(b"My label is: overload_1",
                          self.evaluate(overload_1_2))
      self.assertAllEqual(b"My label is: overload_2", self.evaluate(overload_2))


class AsGraphDefTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for exporting a graph to a graph definition: producer version and shape attributes.

  def testGraphDefVersion(self):
    # [FUNCTION DOC]
    # Asserts the graph definition producer version is passed through to kernels: evaluates a test op
    # that returns the graph's version and compares.
    """Test that the graphdef version is plumbed through to kernels."""
    with ops.Graph().as_default() as g:
      version = g.graph_def_versions.producer
      with self.session(graph=g):
        v = test_ops.graph_def_version().eval()
        self.assertEqual(version, v)

  def testAddShapes(self):
    # [FUNCTION DOC]
    # Sets shapes (unknown, scalar, one unknown dimension, fully known, partly known) on five outputs
    # of one op and creates a constant, then asserts the exported graph definition with add-shapes
    # includes output-shape attributes encoding each.
    with ops.Graph().as_default() as g:
      t1, t2, t3, t4, t5 = _apply_op(g, "FiveFloatOutputs", [],
                                     [dtypes.float32] * 5)
      t1.set_shape(None)
      t2.set_shape([])
      t3.set_shape([None])
      t4.set_shape([43, 37])
      t5.set_shape([43, None])

      b = constant_op.constant(1.0)  # pylint: disable=unused-variable

      gd = g.as_graph_def(add_shapes=True)
      self.assertProtoEqualsVersion("""
      node { name: "FiveFloatOutputs" op: "FiveFloatOutputs"
        attr {
          key: "_output_shapes"
          value {
            list {
              shape { unknown_rank: true }
              shape { }
              shape { dim { size: -1 } }
              shape { dim { size: 43 } dim { size: 37 } }
              shape { dim { size: 43 } dim { size: -1 } }
            }
          }
        }
      }
    node { name: "Const" op: "Const"
      attr {
        key: "_output_shapes"
        value {
          list {
            shape { }
          }
        }
      }
      attr {
        key: "dtype"
        value { type: DT_FLOAT }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_FLOAT
            tensor_shape { }
         float_val: 1.0  } } } }
      """, gd)


@ops.RegisterStatistics("a", "flops")
def _calc_a_forward_flops(unused_graph, unused_node):
  # [FUNCTION DOC]
  # HELPER: Statistics function registered for node type a and statistic flops.
  # Inputs: graph and node, both unused. Returns an op statistics object with statistic name flops and
  # value twenty. Registered as an import-time side effect for the statistics tests.
  return ops.OpStats("flops", 20)


class StatisticsTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for the per-node statistics registry and the statistics accumulation object.

  def testRegisteredNode(self):
    # [FUNCTION DOC]
    # Looks up the flops statistic for a node whose type has a registered function and asserts the value
    # is twenty; looks up an unregistered statistic name and asserts the value is null.
    graph = ops.Graph()
    node = ops._NodeDef("a", "an_a")
    flops = ops.get_stats_for_node_def(graph, node, "flops")
    self.assertEqual(20, flops.value)
    missing_stat = ops.get_stats_for_node_def(graph, node, "missing_stat")
    self.assertEqual(None, missing_stat.value)

  def testUnregisteredNode(self):
    # [FUNCTION DOC]
    # Looks up a statistic for a node type with no registrations and asserts the value is null.
    graph = ops.Graph()
    node = ops._NodeDef("b", "a_b")
    weight_params = ops.get_stats_for_node_def(graph, node, "weight_params")
    self.assertEqual(None, weight_params.value)

  def testAccumulateStatistics(self):
    # [FUNCTION DOC]
    # Creates an empty statistics object (null value), adds one with value three and asserts the total
    # value is three, showing addition from null.
    flops_total = ops.OpStats("flops")
    self.assertEqual(None, flops_total.value)
    second_flops = ops.OpStats("flops", 3)
    flops_total += second_flops
    self.assertEqual(3, flops_total.value)


class DeviceStackTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for device assignment metadata (the record of which device statements applied to an op,
  # with object, file name and line number).

  @test_util.run_deprecated_v1
  def testBasicDeviceAssignmentMetadata(self):
    # [FUNCTION DOC]
    # Graph-mode test. Creates constants outside any scope, in nested string device scopes and under a
    # device function. Asserts the number of recorded assignments per op, the recorded device strings
    # and that the file name is this test file. For the device function case asserts its description
    # matches a pattern with the function name, file and a line number (not a fixed number).

    def device_func(unused_op):
      # [FUNCTION DOC]
      # HELPER: Device function used by the metadata test. Takes the unused operation and returns the
      # wildcard cpu device string.
      return "/cpu:*"

    const_zero = constant_op.constant([0.0], name="zero")
    with ops.device("/cpu"):
      const_one = constant_op.constant([1.0], name="one")
      with ops.device("/cpu:0"):
        const_two = constant_op.constant([2.0], name="two")
    with ops.device(device_func):
      const_three = constant_op.constant(3.0, name="three")

    self.assertEqual(0, len(const_zero.op._device_assignments))

    one_list = const_one.op._device_assignments
    self.assertEqual(1, len(one_list))
    self.assertEqual("/cpu", one_list[0].obj)
    self.assertEqual("ops_test.py", os.path.basename(one_list[0].filename))

    two_list = const_two.op._device_assignments
    self.assertEqual(2, len(two_list))
    devices = [t.obj for t in two_list]
    self.assertEqual(set(["/cpu", "/cpu:0"]), set(devices))

    three_list = const_three.op._device_assignments
    self.assertEqual(1, len(three_list))
    func_description = three_list[0].obj
    expected_regex = r"device_func<.*ops_test.py, [0-9]+"
    self.assertRegex(func_description, expected_regex)

  @test_util.run_deprecated_v1
  def testDeviceAssignmentMetadataForGraphDeviceAndTfDeviceFunctions(self):
    # [FUNCTION DOC]
    # Graph-mode test. Makes one constant under the global device scope and one under the default
    # graph's own device scope, then asserts both record this file name and that the two recorded
    # line numbers differ by exactly two lines (the distance between the two statements).

    with ops.device("/cpu"):
      const_one = constant_op.constant([1.0], name="one")
    with ops.get_default_graph().device("/cpu"):
      const_two = constant_op.constant([2.0], name="two")

    one_metadata = const_one.op._device_assignments[0]
    two_metadata = const_two.op._device_assignments[0]

    # Verify both types of device assignment return the right stack info.
    self.assertRegex("ops_test.py", os.path.basename(one_metadata.filename))
    self.assertEqual(one_metadata.filename, two_metadata.filename)
    self.assertEqual(one_metadata.lineno + 2, two_metadata.lineno)


class ColocationGroupTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for colocation groups: colocation constraints, their metadata, interaction with
  # devices, and nested or reset colocation.

  @test_util.run_deprecated_v1
  def testBasic(self):
    # [FUNCTION DOC]
    # Graph-mode test. Creates an op, then another under a colocation scope on it, then a third outside.
    # Asserts both the first and second report the first's colocation group, and the third has no class
    # attribute (reading it raises a value error).
    a = constant_op.constant([2.0], name="a")
    with ops.colocate_with(a.op):
      b = constant_op.constant(3.0)
    c = constant_op.constant(4.0)
    self.assertEqual([b"loc:@a"], a.op.colocation_groups())
    self.assertEqual([b"loc:@a"], b.op.colocation_groups())
    with self.assertRaises(ValueError):
      c.op.get_attr("_class")

  @test_util.run_deprecated_v1
  def testBasicColocationMetadata(self):
    # [FUNCTION DOC]
    # Graph-mode test. Asserts the colocation metadata of an op created under colocation records the
    # target name, a null object and this test file's name as the file containing the statement.
    const_two = constant_op.constant([2.0], name="two")
    with ops.colocate_with(const_two.op):
      const_three = constant_op.constant(3.0, name="three")
    locations_dict = const_three.op._colocation_dict
    self.assertIn("two", locations_dict)
    metadata = locations_dict["two"]
    self.assertIsNone(metadata.obj)
    # Check that this test's filename is recorded as the file containing the
    # colocation statement.
    self.assertEqual("ops_test.py", os.path.basename(metadata.filename))

  @test_util.run_deprecated_v1
  def testColocationDeviceInteraction(self):
    # [FUNCTION DOC]
    # Graph-mode test. Asserts colocation overrides an enclosing device scope: the new op is on the same
    # device as the op it is colocated with even though a different device scope encloses it.
    with ops.device("/cpu:0"):
      with ops.device("/device:GPU:0"):
        a = constant_op.constant([2.0], name="a")
      with ops.colocate_with(a.op):
        # 'b' is created in the scope of /cpu:0, but it is
        # colocated with 'a', which is on '/device:GPU:0'.  colocate_with
        # overrides devices because it is a stronger constraint.
        b = constant_op.constant(3.0)
    self.assertEqual([b"loc:@a"], b.op.colocation_groups())
    self.assertEqual(a.op.device, b.op.device)

  @test_util.run_deprecated_v1
  def testColocationCanonicalization(self):
    # [FUNCTION DOC]
    # Graph-mode test. Checks device names are canonicalized when colocating: an op placed via a device
    # function, then an op colocated with it inside an equivalent device string scope, end up with the
    # same device.
    with ops.device("/device:GPU:0"):
      _ = constant_op.constant(2.0)
    with ops.device(lambda op: "/device:GPU:0"):
      b = constant_op.constant(3.0)
    with ops.get_default_graph().colocate_with(b):
      with ops.device("/device:GPU:0"):
        c = constant_op.constant(4.0)

    # A's device will be /device:GPU:0
    # B's device will be /device:GPU:0
    # C's device will be /device:GPU:0 because it
    # inherits B's device name, after canonicalizing the names.
    self.assertEqual(b.op.device, c.op.device)

  @test_util.run_deprecated_v1
  def testLocationOverrides(self):
    # [FUNCTION DOC]
    # Graph-mode test. Asserts colocation inside a device scope is recorded in the graph, the colocated
    # op keeps the device, later ops in the same scope keep the device and ops after the scope get the
    # outer device (the device function stack is restored).
    with ops.device("/cpu:0"):
      with ops.device("/device:GPU:0"):
        a = constant_op.constant([2.0], name="a")
        # Note that this colocation is "redundant", since we are
        # within the scope of "/device:GPU:0".  However, we would like to
        # preserve in the GraphDef that these two ops should be
        # colocated in a portable way.
        with ops.colocate_with(a.op):
          b = constant_op.constant(3.0)
        c = constant_op.constant(4.0)
      d = constant_op.constant(5.0)

    self.assertEqual([b"loc:@a"], b.op.colocation_groups())
    self.assertEqual("/device:GPU:0", a.op.device)
    self.assertEqual(a.op.device, b.op.device)

    # Test that device function stack is restored.
    self.assertEqual("/device:GPU:0", c.op.device)
    self.assertEqual("/device:CPU:0", d.op.device)

  @test_util.run_deprecated_v1
  def testNestedColocateWith(self):
    # [FUNCTION DOC]
    # Graph-mode test. Nests colocation on an op and then on an op that is itself colocated with the
    # first; asserts both new ops report the original op's group.
    a = constant_op.constant([2.0], name="a")
    with ops.colocate_with(a.op):
      b = constant_op.constant(3.0)
      with ops.colocate_with(b.op):
        c = constant_op.constant(4.0)
    self.assertEqual([b"loc:@a"], b.op.colocation_groups())
    self.assertEqual([b"loc:@a"], c.op.colocation_groups())

  @test_util.run_deprecated_v1
  def testMultiColocationGroups(self):
    # [FUNCTION DOC]
    # Graph-mode test. Nests colocation on two different ops; asserts the new op belongs to both groups.
    a = constant_op.constant([2.0], name="a")
    b = constant_op.constant(3.0, name="b")
    with ops.colocate_with(a.op):
      with ops.colocate_with(b.op):
        c = constant_op.constant(4.0)
    self.assertEqual(set([b"loc:@a", b"loc:@b"]), set(c.op.colocation_groups()))

  @test_util.run_deprecated_v1
  def testColocationIgnoreStack(self):
    # [FUNCTION DOC]
    # Graph-mode test. Asserts that colocation with the option to ignore existing groups drops the
    # outer group so the new op belongs only to the inner target's group.
    a = constant_op.constant([2.0], name="a")
    b = constant_op.constant(3.0, name="b")
    with ops.colocate_with(a.op):
      with ops.colocate_with(b.op, ignore_existing=True):
        c = constant_op.constant(4.0)
    self.assertEqual(set([b"loc:@b"]), set(c.op.colocation_groups()))

  @test_util.run_deprecated_v1
  def testColocateWithReset(self):
    # [FUNCTION DOC]
    # Graph-mode test. Asserts that colocating with nothing while ignoring existing groups resets
    # colocation: an op created there reports a group of itself, while the outer op keeps the original.
    a = constant_op.constant([2.0], name="a")
    with ops.colocate_with(a.op):
      b = constant_op.constant(3.0, name="b")
      with ops.colocate_with(None, ignore_existing=True):
        c = constant_op.constant(4.0, name="c")
    self.assertEqual([b"loc:@a"], b.op.colocation_groups())
    self.assertEqual([b"loc:@c"], c.op.colocation_groups())

  @test_util.run_deprecated_v1
  def testColocateWithInitialNoneThenNested(self):
    # [FUNCTION DOC]
    # Graph-mode test. Resets colocation to nothing then nests colocation on an op created in the reset
    # scope; asserts both ops report the inner target's group.
    a = constant_op.constant([2.0], name="a")
    with ops.colocate_with(a.op):
      with ops.colocate_with(None, ignore_existing=True):
        b = constant_op.constant(3.0, name="b")
        with ops.colocate_with(b.op):
          c = constant_op.constant(4.0, name="c")
    self.assertEqual([b"loc:@b"], b.op.colocation_groups())
    self.assertEqual([b"loc:@b"], c.op.colocation_groups())

  @test_util.run_deprecated_v1
  def testColocateVariables(self):
    # [FUNCTION DOC]
    # Graph-mode test. Colocates a new variable with an existing variable and asserts the new variable's
    # op reports the first's group.
    a = variables.Variable([2.0], name="a")
    with ops.colocate_with(a.op):
      b = variables.Variable([3.0], name="b")
    self.assertEqual([b"loc:@a"], b.op.colocation_groups())

  @test_util.run_deprecated_v1
  def testColocateResourceVariablesInFunction(self):
    # [FUNCTION DOC]
    # Graph-mode test. Creates a resource variable on CPU, then inside a traced function colocates a
    # tensor creation with it and asserts the created op's device is the CPU device. Calls the function.
    with ops.device("/device:CPU:0"):
      a = resource_variable_ops.ResourceVariable(1.0)

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # HELPER: Traced function body. Colocates with the outer resource variable, creates an op named
      # output and asserts its device is the CPU device. Returns nothing.
      with ops.colocate_with(a):
        b = array_ops.ones([], name="output")
        self.assertEqual("/device:CPU:0", b.op.device)
    f()

  def testColocateWithVariableInFunction(self):
    # [FUNCTION DOC]
    # Asserts that colocating inside a traced function with a variable created outside works and that
    # the resulting function graph definition can be wrapped as a callable function from a graph
    # definition with the output name. Passes if nothing raises.
    v = variables.Variable(1.)

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # HELPER: Traced function body. Colocates with the outer variable and returns a tensor of ones
      # named output.
      with ops.colocate_with(v):
        return array_ops.ones([], name="output")

    f()
    graph_def = f.get_concrete_function().graph.as_graph_def()
    wrap_function.function_from_graph_def(graph_def, [], ["output"])


class DeadlineTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for per-run deadlines (timeouts) passed to a session run.

  def testNoDeadlineSet(self):
    # [FUNCTION DOC]
    # Asserts that running an op that reads the deadline with run options that set no timeout raises an
    # invalid-argument error.
    with ops.Graph().as_default() as g:
      get_deadline = test_ops.get_deadline()
      with self.session(graph=g) as sess:
        run_options = config_pb2.RunOptions()
        with self.assertRaises(errors.InvalidArgumentError):
          sess.run(get_deadline, options=run_options)

  def testDeadlineSetTimesOut(self):
    # [FUNCTION DOC]
    # Asserts that running an op that sleeps ten seconds with a three second timeout in the run options
    # raises a deadline-exceeded error. Takes about three seconds of wall time.
    with ops.Graph().as_default() as g:
      sleep_op = test_ops.sleep_op(10)
      with self.session(graph=g) as sess:
        run_options = config_pb2.RunOptions(timeout_in_ms=3_000)
        with self.assertRaises(errors.DeadlineExceededError):
          sess.run(sleep_op, options=run_options)


class DeprecatedTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for ops marked deprecated at a graph definition version.

  def testSuccess(self):
    # [FUNCTION DOC]
    # Sets the graph's producer version to seven (before removal) and asserts the deprecated op can be
    # created and run without error.
    with ops.Graph().as_default() as g:
      test_util.set_producer_version(g, 7)
      old = test_ops.old()
      with self.session(graph=g):
        old.run()

  def _error(self):
    # [FUNCTION DOC]
    # HELPER: Builds the expected error regular expression for the deprecated op at the current graph
    # definition version: the op is not available in this version, was removed in version eight, with
    # the stated reason. Takes no arguments; returns a pattern string.
    return ((r"Op Old is not available in GraphDef version %d\. "
             r"It has been removed in version 8\. For reasons\.") %
            versions.GRAPH_DEF_VERSION)

  def testGraphConstructionFail(self):
    # [FUNCTION DOC]
    # Asserts that creating the deprecated op at the current graph definition version raises a
    # not-implemented error matching the expected message.
    with ops.Graph().as_default():
      with self.assertRaisesRegex(NotImplementedError, self._error()):
        test_ops.old()


class NameScopeTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for name scope string helpers and the graph's current name scope query.

  def testStripAndPrependScope(self):
    # [FUNCTION DOC]
    # Checks stripping and prepending name scopes on strings: names with the same prefix (including
    # extra slashes, a leading caret and a location prefix) lose the prefix; names with a different or
    # partial prefix keep it; then prepending the new scope gives the expected names.
    strs = [
        "hidden1/hidden1/weights",  # Same prefix. Should strip.
        "hidden1///hidden1/weights",  # Extra "/". Should strip.
        "^hidden1/hidden1/weights",  # Same prefix. Should strip.
        "loc:@hidden1/hidden1/weights",  # Same prefix. Should strip.
        "hhidden1/hidden1/weights",  # Different prefix. Should keep.
        "hidden1"
    ]  # Not a prefix. Should keep.
    expected_striped = [
        "hidden1/weights", "hidden1/weights", "^hidden1/weights",
        "loc:@hidden1/weights", "hhidden1/hidden1/weights", "hidden1"
    ]
    expected_prepended = [
        "hidden2/hidden1/weights", "hidden2/hidden1/weights",
        "^hidden2/hidden1/weights", "loc:@hidden2/hidden1/weights",
        "hidden2/hhidden1/hidden1/weights", "hidden2/hidden1"
    ]
    name_scope_to_strip = "hidden1"
    name_scope_to_add = "hidden2"
    for es, ep, s in zip(expected_striped, expected_prepended, strs):
      striped = ops.strip_name_scope(s, name_scope_to_strip)
      self.assertEqual(es, striped)
      self.assertEqual(ep, ops.prepend_name_scope(striped, name_scope_to_add))

  def testGetNameScope(self):
    # [FUNCTION DOC]
    # Nests three name scopes and asserts the graph's name scope at each level and that it is empty
    # after leaving them all.
    with ops.Graph().as_default() as g:
      with ops.name_scope("scope1"):
        with ops.name_scope("scope2"):
          with ops.name_scope("scope3"):
            self.assertEqual("scope1/scope2/scope3", g.get_name_scope())
          self.assertEqual("scope1/scope2", g.get_name_scope())
        self.assertEqual("scope1", g.get_name_scope())
      self.assertEqual("", g.get_name_scope())

  def testTwoGraphs(self):
    # [FUNCTION DOC]
    # Asserts that opening a name scope named with a lone underscore inside nested default graphs
    # raises a value error saying it is not a valid scope name (allowing "root" wording variants).

    def f():
      # [FUNCTION DOC]
      # HELPER: Function passed to the raises check. Creates two graphs, nests them and opens an invalid
      # name scope. Takes no arguments; the expected error is raised inside it.
      g1 = ops.Graph()
      g2 = ops.Graph()
      with g1.as_default():
        with g2.as_default():
          with ops.name_scope("_"):
            pass

    self.assertRaisesRegex(ValueError,
                           "'_' is not a valid (?:root )?scope name", f)


class EnableEagerExecutionTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for validation of arguments of the eager execution enabling function.

  @test_util.run_v1_only("b/120545219")
  def testBadArgumentsToEnableEagerExecution(self):
    # [FUNCTION DOC]
    # Runs in v1 only. Asserts enabling eager execution with a wrong config argument raises a type error
    # (config must be a configuration protocol buffer), a wrong device policy raises a value error, and a
    # wrong execution mode raises a value error.
    with self.assertRaisesRegex(TypeError, "config must be a tf.ConfigProto"):
      ops.enable_eager_execution(context.DEVICE_PLACEMENT_SILENT)
    with self.assertRaisesRegex(ValueError, "device_policy must be one of"):
      c = config_pb2.ConfigProto()
      ops.enable_eager_execution(c, c)
    with self.assertRaisesRegex(ValueError, "execution_mode must be one of"):
      c = config_pb2.ConfigProto()
      ops.enable_eager_execution(c, execution_mode=c)


class _TupleTensor(composite_tensor.CompositeTensor):
  # [CLASS DOC]
  # HELPER: Composite tensor that mimics a tuple of tensors, used to test registering custom
  # conversion to composite tensors.
  # KNOWN ISSUE: the type specification property passes a generator to the spec rather than a
  # concrete sequence, so the spec holds a single-use iterator.
  # Stores a tuple of tensors converted from the given components, and supports indexing, length and
  # iteration over them.
  """`Tensor`-like `tuple`-like for custom `Tensor` conversion masquerading."""

  def __init__(self, components):
    # [FUNCTION DOC]
    # HELPER: Constructor. Input: an iterable of components. Calls the base constructor and stores a
    # tuple with every component converted to a tensor.
    super(_TupleTensor, self).__init__()
    self._components = tuple(ops.convert_to_tensor(c) for c in components)

  @property
  def _type_spec(self):
    # [FUNCTION DOC]
    # HELPER: Property returning the type spec of this composite tensor, built from the type specs of
    # each stored component tensor.
    # KNOWN ISSUE: the specs are passed as a generator expression, which can be consumed only once, so
    # reading the specs of the returned spec a second time would yield nothing.
    # Returns a new tuple-tensor spec object on every access.
    return _TupleTensorSpec(type_spec.from_value(c) for c in self._components)

  def __getitem__(self, key):
    # [FUNCTION DOC]
    # HELPER: Returns the stored component at the given key (index or slice).
    return self._components[key]

  def __len__(self):
    # [FUNCTION DOC]
    # HELPER: Returns the number of stored components.
    return len(self._components)

  def __iter__(self):
    # [FUNCTION DOC]
    # HELPER: Returns an iterator over the stored component tensors.
    return iter(self._components)


class _TupleTensorSpec(type_spec.TypeSpec):
  # [CLASS DOC]
  # HELPER: Type specification class for the tuple-like composite tensor above. Holds the component
  # specs; its value type is the tuple tensor class and its component specs are the stored specs.
  # Provides conversion to and from components and a serialization.

  def __init__(self, specs):
    # [FUNCTION DOC]
    # HELPER: Constructor. Input: the component specs. Stores them. Does not call the base constructor.
    self._specs = specs

  value_type = property(lambda self: _TupleTensor)
  _component_specs = property(lambda self: self._specs)

  def _to_components(self, value):
    # [FUNCTION DOC]
    # HELPER: Takes a tuple tensor value and returns its stored components.
    return value._components

  def _from_components(self, components):
    # [FUNCTION DOC]
    # HELPER: Takes the components and builds a new tuple tensor from them.
    # KNOWN ISSUE: it unpacks the components into separate positional arguments, but the tuple tensor
    # constructor accepts a single iterable argument, so any value with other than one component would
    # fail with a type error; it works only if exactly one component is given.
    # Returns the new tuple tensor.
    return _TupleTensor(*components)

  def _serialize(self):
    # [FUNCTION DOC]
    # HELPER: Serialization of the spec: returns a one-element tuple holding the component specs.
    return (self._specs,)


class _MyTuple(object):
  # [CLASS DOC]
  # HELPER: Pretend user-side tuple class that the conversion test registers a converter for.
  # KNOWN ISSUE: its docstring refers to a test class name that does not match the actual test class
  # (it omits the word Custom) and has unbalanced quote markup.
  # Stores a tuple of components; supports indexing, length and iteration.
  """Pretend user-side class for `ConvertToCompositeTensorTest ."""

  def __init__(self, components):
    # [FUNCTION DOC]
    # HELPER: Constructor. Input: an iterable of components, stored as a tuple.
    super(_MyTuple, self).__init__()
    self._components = tuple(components)

  def __getitem__(self, key):
    # [FUNCTION DOC]
    # HELPER: Returns the component at the given key (index or slice).
    return self._components[key]

  def __len__(self):
    # [FUNCTION DOC]
    # HELPER: Returns the number of components.
    return len(self._components)

  def __iter__(self):
    # [FUNCTION DOC]
    # HELPER: Returns an iterator over the components.
    return iter(self._components)


tensor_conversion_registry.register_tensor_conversion_function(
    _MyTuple, conversion_func=lambda x, *_, **__: _TupleTensor(x))


class CustomConvertToCompositeTensorTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for registering a conversion function that turns a user class into a composite tensor.

  @test_util.disable_tfrt("TODO(kkb): This makes Kokoro tests fail.")
  def testCompositeTensorConversion(self):
    # [FUNCTION DOC]
    # Converts an instance of the user tuple class (nested numbers and lists) with the or-composite
    # conversion function. Asserts the result is not an ordinary tensor but the tuple composite tensor
    # type, has the same length, and each component is a regular tensor whose constant value equals the
    # corresponding input. Disabled on one runtime where it is known to fail.
    """Tests that a user can register a CompositeTensor converter."""
    x = _MyTuple((1, [2., 3.], [[4, 5], [6, 7]]))
    y = ops.convert_to_tensor_or_composite(x)
    self.assertFalse(tensor_util.is_tf_type(y))
    self.assertIsInstance(y, _TupleTensor)
    self.assertLen(y, len(x))
    for x_, y_ in zip(x, y):
      self.assertIsInstance(y_, tensor_lib.Tensor)
      self.assertTrue(tensor_util.is_tf_type(y_))
      self.assertAllEqual(x_, tensor_util.constant_value(y_))


@test_util.disable_tfrt("Packing EagerTensors is not supported yet.")
class PackEagerTensorTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for packing eager tensors that live on different devices into one packed tensor. The
  # class is disabled on the runtime that does not support packing.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture setup. Calls the base setup, resets the eager context, and configures two virtual logical
    # CPU devices on the first physical CPU. Modifies global context and device configuration.
    super(PackEagerTensorTest, self).setUp()
    context._reset_context()
    cpus = config.list_physical_devices("CPU")
    # Set 2 virtual CPUs
    config.set_logical_device_configuration(cpus[0], [
        context.LogicalDeviceConfiguration(),
        context.LogicalDeviceConfiguration(),
    ])

  def testPack(self):
    # [FUNCTION DOC]
    # Eager test. Creates resource variables and constants on two virtual CPUs, packs the handles of
    # two variables, and asserts the packed tensor is flagged packed, keeps dtype, shape and handle data
    # of the first, reports a composite device and backing device, and that reading its value raises an
    # invalid-argument error. Asserts value errors for packing tensors with different dtypes, different
    # shapes, and different handle data.
    with context.eager_mode():
      with ops.device("CPU:0"):
        var0 = resource_variable_ops.ResourceVariable(1.0)
        c0 = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
      with ops.device("CPU:1"):
        var1 = resource_variable_ops.ResourceVariable(2.0)
        var2 = resource_variable_ops.ResourceVariable([3.0])
        c1 = constant_op.constant([9.0])

      packed_var0 = ops.pack_eager_tensors([var0.handle, var1.handle])
      self.assertTrue(packed_var0.is_packed)
      self.assertEqual(packed_var0.dtype, var0.handle.dtype)
      self.assertEqual(packed_var0.shape, var0.handle.shape)
      self.assertEqual(packed_var0._handle_data, var0.handle._handle_data)
      self.assertIn("COMPOSITE:0", packed_var0.device)
      self.assertIn("COMPOSITE:0", packed_var0.backing_device)
      with self.assertRaises(errors.InvalidArgumentError):
        packed_var0.numpy()

      # Different dtypes
      with self.assertRaises(ValueError):
        ops.pack_eager_tensors([var0.handle, c1])

      # Different shapes
      with self.assertRaises(ValueError):
        ops.pack_eager_tensors([c0, c1])

      # Different handle data
      with self.assertRaises(ValueError):
        ops.pack_eager_tensors([var0.handle, var2.handle])


class GraphDefInputShapesTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for input-shape attributes on function definitions when a function is added to a
  # graph and the graph is exported with shapes.

  def setUpInputShapes(self, pre_add_input_shapes):
    # [FUNCTION DOC]
    # HELPER: Builds a traced function with a partly known input shape, calls it once, and optionally
    # pre-attaches an input-shapes attribute to the concrete function. Input: a flag saying whether to
    # add the attribute before adding the function to a new graph. Adds the function to a fresh graph,
    # exports it with shapes, asserts exactly one function in the library, and returns the input-shapes
    # attribute of that function definition.

    test_tensor_shape = [None, 1, 1, 1]

    @def_function.function(input_signature=[
        tensor_lib.TensorSpec(shape=test_tensor_shape, dtype=dtypes.float32)
    ])
    def f(x):
      # [FUNCTION DOC]
      # HELPER: Traced function with an input signature of a float tensor of shape unknown, one, one, one;
      # returns an identity of its input named output.
      return array_ops.identity(x, name="output")

    x = array_ops.ones([2, 1, 1, 1], dtype=dtypes.float32)
    f(x)

    tensor_shape_proto = tensor_shape_pb2.TensorShapeProto(dim=[
        tensor_shape_pb2.TensorShapeProto.Dim(size=-1 if d is None else d)
        for d in test_tensor_shape
    ])
    list_proto = attr_value_pb2.AttrValue.ListValue(shape=[tensor_shape_proto])
    concrete_function = f.get_concrete_function()
    if pre_add_input_shapes:
      attr_value = attr_value_pb2.AttrValue(list=list_proto)
      concrete_function = eager_function.ConcreteFunction.from_func_graph(
          concrete_function.graph,
          concrete_function.function_type,
          attrs={"_input_shapes": attr_value},
      )

    test_graph = ops.Graph()
    with test_graph.as_default():
      concrete_function.add_to_graph(g=test_graph)
    graph_def = test_graph.as_graph_def(add_shapes=True)
    self.assertLen(graph_def.library.function, 1)
    function_def = graph_def.library.function[0]
    input_shapes = function_def.attr["_input_shapes"]
    return input_shapes

  def testGraphDefInputShapes(self):
    # [FUNCTION DOC]
    # Asserts that the input-shapes attribute is the same whether it was added manually before adding the
    # function to the graph or computed automatically when exporting the graph definition.
    pre_added_input_shapes = self.setUpInputShapes(pre_add_input_shapes=True)
    post_added_input_shapes = self.setUpInputShapes(pre_add_input_shapes=False)
    self.assertProtoEquals(pre_added_input_shapes, post_added_input_shapes)


class TensorTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for conversion of tensors to numpy arrays.

  def testToArrayEagerMode(self):
    # [FUNCTION DOC]
    # Eager mode. Asserts converting a tensor to a numpy array honors the requested dtype (float32) and
    # otherwise uses the tensor's own dtype (int64).

    with context.eager_mode():
      a = np.array(constant_op.constant(32), dtype=np.float32)
      b = np.array(constant_op.constant(32, dtype=dtypes.int64))

      self.assertEqual(a.dtype, np.dtype(np.float32))
      self.assertEqual(b.dtype, np.dtype(np.int64))

  def testToArrayFunctionMode(self):
    # [FUNCTION DOC]
    # Asserts that converting a symbolic tensor to a numpy array inside two traced functions (one with a
    # requested dtype, one without) raises a not-implemented error about symbolic tensors when called.

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # HELPER: Traced function body converting a constant to a numpy array with integer dtype; raises
      # while tracing.
      # Raises during trace compilation.
      return np.array(constant_op.constant(32), dtype=np.int32)

    @def_function.function
    def g():
      # [FUNCTION DOC]
      # HELPER: Traced function body converting a constant to a numpy array; raises while tracing.
      # Raises during trace compilation.
      return np.array(constant_op.constant(32))

    with self.assertRaisesRegex(NotImplementedError,
                                "Cannot convert a symbolic tf.Tensor"):
      f()

    with self.assertRaisesRegex(NotImplementedError,
                                "Cannot convert a symbolic tf.Tensor"):
      g()


if __name__ == "__main__":
  googletest.main()
