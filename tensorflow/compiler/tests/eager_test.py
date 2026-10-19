# [MODULE DOC]
# Test module for eager execution on XLA-backed devices (CPU/GPU/TPU test devices).
# It defines four XLATestCase-derived test classes: EagerTest (ops, gradients, variables, shape
# ops, Adam optimizer run eagerly under the XLA test scope), EagerFunctionTest (compiled
# functions, resource variables, control flow, nested functions and gradient tapes),
# ExcessivePaddingTest (tensors that TPU would flatten) and MultiDeviceTest (two TPU cores).
# Also holds a small helper that counts TPU devices. When run as a script it enables eager
# execution with device placement logging and starts the test runner. No import-time side effects.
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
"""Test cases for eager execution using XLA."""

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.layers import convolutional
from tensorflow.python.layers import pooling
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops import cond
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import googletest
from tensorflow.python.training import adam


class EagerTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # Eager-mode tests that run ops, gradient tapes, variables, shape queries and optimizers
  # inside the XLA test scope (the device under test). No shared fixtures beyond the base class.

  def testBasic(self):
    # [FUNCTION DOC]
    # Multiplies two scalar integer constants eagerly on the test device and asserts the
    # product equals fifteen.
    with self.test_scope():
      three = constant_op.constant(3)
      five = constant_op.constant(5)
      product = three * five
      self.assertAllEqual(15, product)

  def testGradientTape(self):
    # [FUNCTION DOC]
    # Records a persistent gradient tape over the expression x plus y plus x times y with
    # x of one and y of ten, then asserts the partial derivatives are eleven (with respect to x)
    # and two (with respect to y).
    with self.test_scope():

      x = constant_op.constant(1.0)
      y = constant_op.constant(10.0)
      with backprop.GradientTape(persistent=True) as tape:
        tape.watch(x)
        tape.watch(y)
        a = x + y + x * y
      da_dx = tape.gradient(a, x)
      da_dy = tape.gradient(a, y)

    self.assertEqual(11.0, da_dx.numpy())
    self.assertEqual(2.0, da_dy.numpy())

  def testExecuteListOutputLen0(self):
    # [FUNCTION DOC]
    # Unstacks an empty float tensor along axis zero and asserts the result is an empty list
    # (an op with a list output of length zero).
    with self.test_scope():
      empty = constant_op.constant([], dtype=dtypes.float32)
      result = array_ops_stack.unstack(empty, 0)
      self.assertTrue(isinstance(result, list))
      self.assertEqual(0, len(result))

  def testExecuteListOutputLen1(self):
    # [FUNCTION DOC]
    # Splits a two-by-three tensor into one piece along axis one and asserts the result is a
    # list of length one whose only element equals the original tensor.
    with self.test_scope():
      split_dim = constant_op.constant(1)
      value = constant_op.constant([[0., 1., 2.], [3., 4., 5.]])
      result = array_ops.split(value, 1, axis=split_dim)
      self.assertTrue(isinstance(result, list))
      self.assertEqual(1, len(result))
      self.assertAllEqual([[0, 1, 2], [3, 4, 5]], result[0])

  def testExecuteListOutputLen3(self):
    # [FUNCTION DOC]
    # Splits a two-by-three tensor into three pieces along axis one and asserts the result is a
    # list of three columns, each of shape two by one, with the expected values.
    with self.test_scope():
      split_dim = constant_op.constant(1)
      value = constant_op.constant([[0., 1., 2.], [3., 4., 5.]])
      result = array_ops.split(value, 3, axis=split_dim)
      self.assertTrue(isinstance(result, list))
      self.assertEqual(3, len(result))
      self.assertAllEqual([[0], [3]], result[0])
      self.assertAllEqual([[1], [4]], result[1])
      self.assertAllEqual([[2], [5]], result[2])

  def testBasicGraph(self):
    # [FUNCTION DOC]
    # Runs the scalar multiplication of three and five first eagerly, then again inside a
    # graph-mode session (evaluated through the session), asserting fifteen both times.
    # Run some ops eagerly
    with self.test_scope():
      three = constant_op.constant(3)
      five = constant_op.constant(5)
      product = three * five
      self.assertAllEqual(15, product)

    # Run some ops graphly
    with context.graph_mode(), self.session():
      with self.test_scope():
        three = constant_op.constant(3)
        five = constant_op.constant(5)
        product = three * five
        self.assertAllEqual(15, self.evaluate(product))

  def testDegenerateSlices(self):
    # [FUNCTION DOC]
    # Slices a three-by-two-by-three float tensor with degenerate ranges (forward interval with
    # negative stride, reverse interval with positive stride, empty intervals in every
    # dimension) and asserts the results match the same slicing done on the numpy array.
    with self.test_scope():
      npt = np.arange(1, 19, dtype=np.float32).reshape(3, 2, 3)
      t = constant_op.constant(npt)
      # degenerate by offering a forward interval with a negative stride
      self.assertAllEqual(npt[0:-1:-1, :, :], t[0:-1:-1, :, :])
      # degenerate with a reverse interval with a positive stride
      self.assertAllEqual(npt[-1:0, :, :], t[-1:0, :, :])
      # empty interval in every dimension
      self.assertAllEqual(npt[-1:0, 2:2, 2:3:-1], t[-1:0, 2:2, 2:3:-1])

  def testIdentity(self):
    # [FUNCTION DOC]
    # Applies the identity op to the integer two on the test device and asserts it is unchanged.
    with self.test_scope():
      self.assertAllEqual(2, array_ops.identity(2))

  def testRandomOps(self):
    # [FUNCTION DOC]
    # Generates a two-by-two uniform random float tensor and asserts its two rows are not
    # entirely equal (a probabilistic check that the generator is not constant).
    with self.test_scope():
      tensor = gen_random_ops.random_uniform((2, 2), dtypes.float32)
      row0 = tensor[0].numpy()
      row1 = tensor[1].numpy()
      # It should be very unlikely to rng to generate two equal rows.
      self.assertFalse((row0 == row1).all())

  def testIdentityOnVariable(self):
    # [FUNCTION DOC]
    # Creates a boolean resource variable, applies identity to it and asserts the resulting
    # value is true.
    with self.test_scope():
      v = resource_variable_ops.ResourceVariable(True)
      i = array_ops.identity(v)
    self.assertAllEqual(True, i.numpy())

  def testAssignAddVariable(self):
    # [FUNCTION DOC]
    # Creates a float resource variable of one, adds two in place, and asserts the value is three.
    with self.test_scope():
      v = resource_variable_ops.ResourceVariable(1.0)
      v.assign_add(2.0)
    self.assertEqual(3.0, v.numpy())

  def testReadAssignRead(self):
    # [FUNCTION DOC]
    # Reads a variable, adds two in place, reads again, and asserts the first read still holds
    # the old value (one) while the second holds the new value (three).
    with self.test_scope():
      v = resource_variable_ops.ResourceVariable(1.0)
      val1 = v.read_value()
      v.assign_add(2.0)
      val2 = v.read_value()
    self.assertEqual(1.0, val1.numpy())
    self.assertEqual(3.0, val2.numpy())

  def testGradient(self):
    # [FUNCTION DOC]
    # Builds the gradient function of an identity function and asserts that, given an upstream
    # gradient of two at input one, the returned gradient is two.
    def f(x):
      # [FUNCTION DOC]
      # HELPER: function under differentiation in testGradient.
      # Returns its input unchanged.
      return x

    with self.test_scope():
      grad_fn = backprop.gradients_function(f)
      self.assertAllEqual(2., grad_fn(1., dy=2.)[0])

  def testVariableGradient(self):
    # [FUNCTION DOC]
    # Computes the implicit gradient of the square of a variable (initial one) and asserts the
    # gradient of the variable equals two.
    with self.test_scope():
      v0 = resource_variable_ops.ResourceVariable(1.0)

      def f():
        # [FUNCTION DOC]
        # HELPER: function under differentiation in testVariableGradient.
        # Returns the square of the captured variable v0.
        x = v0 * v0
        return x

      grads = backprop.implicit_grad(f)()
    self.assertEqual(2., grads[0][0].numpy())

  def testMultipleVariableReads(self):
    # [FUNCTION DOC]
    # Memory regression test: creates a roughly 128 MiB variable and reads its value one hundred
    # times. If reads copied the underlying buffer this would exhaust memory; nothing is asserted
    # beyond completing without failure.
    # This test makes sure consecutive variable reads don't copy
    # the underlying memory.
    with self.test_scope():
      # Create 128MiB variables
      var = resource_variable_ops.ResourceVariable(
          array_ops.ones([32, 1024, 1024]))

      # Read the same variable 100 times. If the underlying tensor
      # is not copied, this is a trivial operation. If it is copied,
      # this will eat over 13GB and OOM.
      values = []
      for _ in range(100):
        values.append(var.value())

  # The shape, shape_n, size, and rank are tested here because their
  # execution kernels (as opposed to compilation only tf2xla kernels)
  # are distincts from tf2xla kernels.

  def testShape(self):
    # [FUNCTION DOC]
    # Checks the shape op on scalars and tensors of rank one to three, both for directly
    # constructed constants and for tensors produced by an op running on the device, asserting
    # the returned shapes are correct.
    def const(value):
      # [FUNCTION DOC]
      # HELPER: shape-of-constant helper for testShape.
      # Returns, as a numpy array, the shape of a constant tensor built from the given value.
      return array_ops.shape(
          constant_op.constant(value)).numpy()

    def ones(value):
      # [FUNCTION DOC]
      # HELPER: shape-of-ones helper for testShape.
      # Returns, as a numpy array, the shape of a tensor of ones whose shape is the given value.
      return array_ops.shape(
          array_ops.ones(value)).numpy()

    with self.test_scope():
      # Shapes of directly constructed tensors
      self.assertAllEqual([], const(3))
      self.assertAllEqual([3], const([1.0, 2.0, 3.0]))
      self.assertAllEqual([2, 2], const([[1.0, 2.0], [3.0, 4.0]]))
      self.assertAllEqual([2, 1, 2], const([[[1.0, 2.0]], [[3.0, 4.0]]]))

      # Shapes of tensors created by op running on device
      # We make this distinction because directly constructed tensors
      # are treated differently in a few places that can influence shape:
      #  - they always have on_host_tensor
      #  - they and their shapes can be cached
      #  - they end up on device via a copy, instead of as program output
      self.assertAllEqual([], ones([]))
      self.assertAllEqual([3], ones([3]))
      self.assertAllEqual([2, 2], ones([2, 2]))
      self.assertAllEqual([2, 1, 2], ones([2, 1, 2]))

  def testShapeN(self):
    # [FUNCTION DOC]
    # Applies the shape_n op to three constants (ranks zero, one, two), then to three tensors
    # of ones built on the device, asserting both times the shapes are empty, [3] and [2, 2].
    with self.test_scope():
      # Shapes of directly constructed tensors
      shapes = array_ops.shape_n([
          constant_op.constant(1.0),
          constant_op.constant([1.0, 2.0, 3.0]),
          constant_op.constant([[1.0, 2.0], [3.0, 4.0]])])
      self.assertAllEqual(
          [[], [3], [2, 2]],
          [x.numpy().tolist() for x in shapes])

      # Shapes of tensors created by op running on device
      shapes = array_ops.shape_n([
          array_ops.ones([]),
          array_ops.ones([3]),
          array_ops.ones([2, 2])])
      self.assertAllEqual(
          [[], [3], [2, 2]],
          [x.numpy().tolist() for x in shapes])

  def testSize(self):
    # [FUNCTION DOC]
    # Asserts the size op returns one, three and four for a scalar, a three-element vector and
    # a two-by-two matrix of constants.
    with self.test_scope():
      self.assertEqual(
          1, array_ops.size(constant_op.constant(1.0)).numpy())
      self.assertEqual(
          3, array_ops.size(constant_op.constant([1.0, 2.0, 3.0])).numpy())
      self.assertEqual(
          4, array_ops.size(
              constant_op.constant([[1.0, 2.0], [3.0, 4.0]])).numpy())

  def testRank(self):
    # [FUNCTION DOC]
    # Asserts the rank op returns zero, one and two for a scalar, a vector and a matrix of
    # constants.
    with self.test_scope():
      self.assertEqual(
          0, array_ops.rank(constant_op.constant(1.0)).numpy())
      self.assertEqual(
          1, array_ops.rank(constant_op.constant([1.0, 2.0, 3.0])).numpy())
      self.assertEqual(
          2, array_ops.rank(
              constant_op.constant([[1.0, 2.0], [3.0, 4.0]])).numpy())

  def testAdam(self):
    # [FUNCTION DOC]
    # Minimizes the square of a variable starting at ten with one Adam step of learning rate
    # 0.1 (gradient taken with a tape and applied with the optimizer) and asserts the variable
    # is about 9.9 to three decimal places.
    with self.test_scope():
      optimizer = adam.AdamOptimizer(0.1)
      x = resource_variable_ops.ResourceVariable(10.0)
      with backprop.GradientTape() as tape:
        y = x * x
      dy_dx = tape.gradient(y, x)
      optimizer.apply_gradients([(dy_dx, x)])
      self.assertAlmostEqual(9.9, x.numpy(), places=3)

  def testAdamSparse(self):
    # [FUNCTION DOC]
    # Creates a three-by-two embedding variable on the CPU, looks up row one on the test device
    # and sums it, takes the sparse (indexed slices) gradient, applies it with Adam and then
    # adds ones to the variable. Asserts the gradient is an indexed-slices object and the final
    # embedding is two for rows zero and two and 1.9 for row one. Optimizer updates and the
    # assign-add run on the CPU because they are placed with the resource.
    with ops.device('/cpu:0'):
      # Create 2-D embedding for 3 objects on CPU because sparse/sliced updates
      # are not implemented on TPU.
      embedding_matrix = resource_variable_ops.ResourceVariable(
          array_ops.ones([3, 2]))

    with self.test_scope():
      with backprop.GradientTape() as tape:
        embedding = embedding_ops.embedding_lookup(embedding_matrix, [1])
        y = math_ops.reduce_sum(embedding)
      dy_dx = tape.gradient(y, embedding_matrix)
      self.assertIsInstance(dy_dx, indexed_slices.IndexedSlices)
      optimizer = adam.AdamOptimizer(0.1)
      # The gradient application operations will run on CPU because optimizer
      # updates are always collocated with the variable.
      optimizer.apply_gradients([(dy_dx, embedding_matrix)])

      # This assign_add will run on CPU because when an input to an
      # operation is a resource, this operation is placed on the resource's
      # device by the eager runtime.
      embedding_matrix.assign_add(array_ops.ones([3, 2]))

    self.assertAllClose([[2.0, 2.0],
                         [1.9, 1.9],
                         [2.0, 2.0]], embedding_matrix.numpy())


class EagerFunctionTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # Tests that wrap Python functions with the function decorator/wrapper and run them under the
  # XLA test scope: arguments of various kinds, resource variables, gradient tapes, nesting,
  # control flow and device placement.

  def testBasic(self):
    # [FUNCTION DOC]
    # Wraps the matrix multiply op as a function, multiplies a two-by-two matrix by itself with
    # the first operand transposed and asserts the flattened result is 10, 14, 14, 20.
    with self.test_scope():
      matmul = def_function.function(math_ops.matmul)
      t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
      sq = matmul(t, t, transpose_a=True)
      self.assertAllEqual(sq.numpy().reshape(-1), [10, 14, 14, 20])

  def testConv(self):
    # [FUNCTION DOC]
    # Builds a one-filter two-by-two valid convolution (ones kernel, zero bias, relu) followed by
    # two-by-two max pooling, wraps the model as a function, feeds a one-by-four-by-four-by-one
    # tensor of ones and asserts the output is a single value of four. Skipped on GPU devices
    # because the random normal kernel is slow and denylisted there.
    if 'GPU' in self.device:
      # TODO(b/32333178)
      self.skipTest('Current implementation of RandomStandardNormal kernel '
                    'is very slow on GPU, and has been denylisted.')
    with self.test_scope():
      data_format = 'channels_last'
      conv = convolutional.Conv2D(
          filters=1, kernel_size=2, padding='VALID',
          data_format=data_format, activation=nn_ops.relu,
          kernel_initializer=init_ops.ones_initializer(),
          bias_initializer=init_ops.zeros_initializer())
      pool = pooling.MaxPooling2D(2, 2, data_format=data_format)

      def model(x):
        # [FUNCTION DOC]
        # HELPER: model body for testConv.
        # Applies the convolution layer then the pooling layer to the input and returns the result.
        x = conv(x)
        return pool(x)
      model = def_function.function(model)

      x = array_ops.ones([1, 4, 4, 1])
      y = model(x)
      self.assertAllEqual(y.numpy(), [[[[4.]]]])

  def testReadVariable(self):
    # [FUNCTION DOC]
    # Reads a variable inside a wrapped function and asserts the returned value is one.
    with self.test_scope():
      v = resource_variable_ops.ResourceVariable(1.0)

      @def_function.function
      def f():
        # [FUNCTION DOC]
        # HELPER: function body for testReadVariable.
        # Returns the current value of the captured variable v.
        return v.read_value()

      var = f()
      self.assertEqual(1.0, var.numpy())

  def testResourceVariableNoInlineReadWrite(self):
    # [FUNCTION DOC]
    # Defines two no-inline functions: g adds its argument to variable w then returns v plus the
    # argument times w; f calls g with one through five and sums. Asserts f returns 145 and the
    # final w is 15, checking both reads and writes of captured resources across calls.
    with self.test_scope():
      v = resource_variable_ops.ResourceVariable(1.0)
      w = resource_variable_ops.ResourceVariable(0.0)

      @def_function.function(experimental_attributes={'_noinline': True})
      def g(x):
        # [FUNCTION DOC]
        # HELPER: inner no-inline function for this test.
        # Adds x to the captured variable w, then returns v plus x times the updated w.
        w.assign(w.read_value() + x)
        return v.read_value() + x * w.read_value()

      @def_function.function(experimental_attributes={'_noinline': True})
      def f():
        # [FUNCTION DOC]
        # HELPER: outer no-inline function for this test.
        # Returns the sum of g applied to one, two, three, four and five.
        return g(1.0) + g(2.0) + g(3.0) + g(4.0) + g(5.0)

      # 1 + 1*1 + 1 + 2*3 + 1 + 3*6 + 1 + 4*10 + 1 + 5*15
      self.assertEqual(145.0, f().numpy())
      self.assertEqual(15.0, w.read_value().numpy())

  def testResourceVariableNoInlineReadOnly(self):
    # [FUNCTION DOC]
    # Defines a no-inline function g that only reads a variable (ten) and a no-inline function f
    # that sums five calls of g; asserts the result is fifty.
    with self.test_scope():
      v = resource_variable_ops.ResourceVariable(10.0)

      @def_function.function(experimental_attributes={'_noinline': True})
      def g():
        # [FUNCTION DOC]
        # HELPER: inner read-only no-inline function for this test.
        # Returns the value of the captured variable v.
        return v.read_value()

      @def_function.function(experimental_attributes={'_noinline': True})
      def f():
        # [FUNCTION DOC]
        # HELPER: outer no-inline function for this test.
        # Returns the sum of five calls of g.
        return g() + g() + g() + g() + g()

      self.assertEqual(50.0, f().numpy())

  def testResourceVariableNoInlineWriteOnly(self):
    # [FUNCTION DOC]
    # Defines a no-inline function g that only assigns its argument to a variable and a no-inline
    # function f that calls g with one through five; after calling f asserts the variable is five.
    with self.test_scope():
      v = resource_variable_ops.ResourceVariable(0.0)

      @def_function.function(experimental_attributes={'_noinline': True})
      def g(x):
        # [FUNCTION DOC]
        # HELPER: inner write-only no-inline function for this test.
        # Assigns x to the captured variable v and returns nothing.
        v.assign(x)

      @def_function.function(experimental_attributes={'_noinline': True})
      def f():
        # [FUNCTION DOC]
        # HELPER: outer no-inline function for this test.
        # Calls g with one, two, three, four and five in order; returns nothing.
        g(1.0)
        g(2.0)
        g(3.0)
        g(4.0)
        g(5.0)

      f()
      self.assertEqual(5.0, v.read_value().numpy())

  def testUpdateVariable(self):
    # [FUNCTION DOC]
    # Passes a variable into a wrapped function that adds one in place and returns it; asserts the
    # returned variable's value is two.
    with self.test_scope():
      v = resource_variable_ops.ResourceVariable(1.0)

      def f(v):
        # [FUNCTION DOC]
        # HELPER: function body for testUpdateVariable.
        # Adds one in place to the given variable and returns the variable.
        v.assign_add(1.0)
        return v

      f = def_function.function(f)

      var = f(v)
      self.assertEqual(2.0, var.numpy())

  def testReturnResourceHandle(self):
    # [FUNCTION DOC]
    # Wraps a function that returns a variable's resource handle, calls it, then reads the
    # variable through the handle and asserts the value matches the original variable.
    with self.test_scope():
      v = resource_variable_ops.ResourceVariable([[1.0, 2.0], [3.0, 4.0]])

      def f(v):
        # [FUNCTION DOC]
        # HELPER: function body for testReturnResourceHandle.
        # Returns the resource handle of the given variable.
        return v.handle

      f = def_function.function(f)
      handle = f(v)
      self.assertAllEqual(v.numpy(),
                          resource_variable_ops.read_variable_op(
                              handle, dtypes.float32).numpy())

  def testReturnMultipleResourceHandles(self):
    # [FUNCTION DOC]
    # Wraps a function that returns a handle of its argument variable, three times that variable,
    # a handle of a captured second variable and the sum of both variables. Asserts that the
    # handles read back the original values and the tensor outputs are 3.75 and 3.25.
    with self.test_scope():
      v1 = resource_variable_ops.ResourceVariable(1.25)
      v2 = resource_variable_ops.ResourceVariable(2.0)

      def f(v):
        # [FUNCTION DOC]
        # HELPER: function body for testReturnMultipleResourceHandles.
        # Returns four values: the handle of v, three times v, the handle of captured v2, and
        # v plus v2.
        return v.handle, 3.0 * v, v2.handle, v + v2

      f = def_function.function(f)
      v1_handle, v1_times_3, v2_handle, variable_sum = f(v1)
      self.assertAllEqual(v1.numpy(),
                          resource_variable_ops.read_variable_op(
                              v1_handle, dtypes.float32).numpy())
      self.assertEqual(3.75, v1_times_3.numpy())
      self.assertAllEqual(v2.numpy(),
                          resource_variable_ops.read_variable_op(
                              v2_handle, dtypes.float32).numpy())
      self.assertEqual(3.25, variable_sum.numpy())

  def testAllArgumentKinds(self):
    # [FUNCTION DOC]
    # Calls a wrapped function mixing compile-time constants, regular tensors and resource
    # variables in scrambled argument order, and asserts the four outputs (sum, cast plus
    # variable, slice of a variable, product) have the expected values.
    """Test a complex function that takes different argument kinds.

    tf2xla machinery that translates, compiles, and runs defuns
    classifies arguments into: compile-time constants, regular tensors,
    and resources. This test creates a function with a mix of all these
    kinds. Moreover, the order of function arguments is intentionally mixed up.

    This also tests the case when the same argument is a compile-time constant
    as well as used in an operation that normally expects its inputs to be
    in device memory - addition in this case.
    """
    with self.test_scope():
      def foo(c1, r1, v1, c2, v2, r2):
        # [FUNCTION DOC]
        # HELPER: function under test in testAllArgumentKinds.
        # Takes constants c1 and c2, tensors r1 and r2 and variables v1 and v2. Returns four
        # values: c1 plus r1; c2 cast to float plus v2; a slice of v1 with begin c1 and size c2;
        # and r2 times v2.
        # c1 and c2 are compile-time constants
        # r1 and r2 are regular tensors
        # v1 and v2 are resource variables
        a = c1 + r1
        b = math_ops.cast(c2, dtypes.float32) + v2
        c = array_ops.slice(v1, c1, c2)
        d = r2 * v2
        return a, b, c, d

      foo = def_function.function(foo)

      c1 = [0, 0]
      c2 = array_ops.ones([2], dtype=dtypes.int32)

      r1 = array_ops.ones([2])
      r2 = [[2., 2.], [3., 3.]]

      v1 = resource_variable_ops.ResourceVariable([[1., 2.], [3., 4.]])
      v2 = resource_variable_ops.ResourceVariable([[10., 20.], [30., 40.]])

      a, b, c, d = foo(c1, r1, v1, c2, v2, r2)

      self.assertAllEqual([1, 1], a.numpy())
      self.assertAllEqual([[11., 21.], [31., 41.]], b.numpy())
      self.assertAllEqual([[1.]], c.numpy())
      self.assertAllEqual([[20., 40.], [90., 120.]], d.numpy())

  def testDefunInGradientTape(self):
    # [FUNCTION DOC]
    # Calls a wrapped function that multiplies a captured variable (five) by itself and an input
    # (three) inside a gradient tape; asserts the output is 75 and the gradient with respect to
    # the variable is 30.
    with self.test_scope():
      v0 = resource_variable_ops.ResourceVariable(5.0)

      @def_function.function
      def f(x):
        # [FUNCTION DOC]
        # HELPER: function body for testDefunInGradientTape.
        # Returns the captured variable v0 squared times x.
        x = v0 * v0 * x
        return x

      x = constant_op.constant(3.0)
      with backprop.GradientTape() as tape:
        y = f(x)
      dy = tape.gradient(y, v0)

    self.assertEqual(75, y.numpy())
    self.assertEqual(30, dy.numpy())

  def testGradientTapeInDefun(self):
    # [FUNCTION DOC]
    # Uses a gradient tape inside a wrapped function to differentiate a captured variable times
    # one with respect to the variable; asserts the gradient returned from the function is one.
    with self.test_scope():
      v0 = resource_variable_ops.ResourceVariable(5.0)

      @def_function.function
      def f():
        # [FUNCTION DOC]
        # HELPER: function body for testGradientTapeInDefun.
        # Records v0 times one on a tape and returns the gradient with respect to v0.
        x = constant_op.constant(1.0)
        with backprop.GradientTape() as tape:
          y = v0 * x
        dy = tape.gradient(y, v0)
        return dy

      dy = f()
      self.assertEqual(1.0, dy.numpy())

  def testSliceInDefun(self):
    # [FUNCTION DOC]
    # Slices a two-by-three-by-four tensor of ones inside a wrapped function using a strided first
    # dimension, a dynamic start in the second dimension and an ellipsis, under a gradient tape
    # watching both inputs. Asserts the output is ones of shape one by two by four and the gradient
    # with respect to the input has shape two by three by four.
    with self.test_scope():

      @def_function.function
      def f(x, y):
        # [FUNCTION DOC]
        # HELPER: function body for testSliceInDefun.
        # Returns every second element of x along the first dimension, starting at index y in
        # the second dimension, keeping all remaining dimensions.
        return x[0::2, y:, ...]

      x = array_ops.ones([2, 3, 4], dtype=dtypes.float32)
      y = array_ops.ones([], dtype=dtypes.int32)
      with backprop.GradientTape() as tape:
        tape.watch(x)
        tape.watch(y)
        z = f(x, y)
      dz = tape.gradient(z, x)

      self.assertAllEqual(np.ones([1, 2, 4]), z.numpy())
      self.assertAllEqual((2, 3, 4), dz.shape.as_list())

  def testNestedDefun(self):
    # [FUNCTION DOC]
    # Defines a wrapped function that doubles its input and a wrapped function that calls it and
    # adds one; applies it to [2, 3, 4] and asserts the result is [5, 7, 9].
    with self.test_scope():

      @def_function.function
      def times_two(x):
        # [FUNCTION DOC]
        # HELPER: inner function for testNestedDefun.
        # Returns twice the input.
        return 2. * x

      @def_function.function
      def two_x_plus_1(x):
        # [FUNCTION DOC]
        # HELPER: outer function for testNestedDefun.
        # Returns twice the input plus one by calling times_two.
        return times_two(x) + 1.

      x = constant_op.constant([2., 3., 4.])
      y = two_x_plus_1(x)
      self.assertAllEqual([5., 7., 9.], y.numpy())

  def testNestedDefunWithVariable(self):
    # [FUNCTION DOC]
    # Nests a wrapped function g (multiplies by a captured variable of five) inside f, which
    # first multiplies its input by the same variable. With input three, asserts the result is 75.
    with self.test_scope():
      v0 = resource_variable_ops.ResourceVariable(5.0)

      @def_function.function
      def g(x):
        # [FUNCTION DOC]
        # HELPER: inner function for this test.
        # Returns the input multiplied by the captured variable v0.
        x = v0 * x
        return x

      @def_function.function
      def f(x):
        # [FUNCTION DOC]
        # HELPER: outer function for this test.
        # Multiplies the input by v0 and passes the product to g; returns g's result.
        x = g(v0 * x)
        return x

      x = constant_op.constant(3.0)
      y = f(x)

    self.assertEqual(75.0, y.numpy())

  def testNestedDefunInGradientTape(self):
    # [FUNCTION DOC]
    # Same nested functions as the variable test, called under a gradient tape. With a variable of
    # five and input three, asserts the output is 75 and the gradient with respect to the
    # variable is 30.
    with self.test_scope():
      v0 = resource_variable_ops.ResourceVariable(5.0)

      @def_function.function
      def g(x):
        # [FUNCTION DOC]
        # HELPER: inner function for this test.
        # Returns the input multiplied by the captured variable v0.
        x = v0 * x
        return x

      @def_function.function
      def f(x):
        # [FUNCTION DOC]
        # HELPER: outer function for this test.
        # Multiplies the input by v0 and passes the product to g; returns g's result.
        x = g(v0 * x)
        return x

      x = constant_op.constant(3.0)
      with backprop.GradientTape() as tape:
        y = f(x)
      dy = tape.gradient(y, v0)

    self.assertEqual(75, y.numpy())
    self.assertEqual(30, dy.numpy())

  def testNestedDefunInGradientTapeDifferentVars(self):
    # [FUNCTION DOC]
    # Nested wrapped functions where the inner one uses a second variable (three) and the outer
    # one the first (five), called with input three under a persistent tape. Asserts the output
    # is 45 and the gradients with respect to the first and second variables are 9 and 15.
    with self.test_scope():
      v0 = resource_variable_ops.ResourceVariable(5.0)
      v1 = resource_variable_ops.ResourceVariable(3.0)

      @def_function.function
      def g(x):
        # [FUNCTION DOC]
        # HELPER: inner function for this test.
        # Returns the input multiplied by the captured variable v1.
        x = v1 * x
        return x

      @def_function.function
      def f(x):
        # [FUNCTION DOC]
        # HELPER: outer function for this test.
        # Multiplies the input by v0 and passes the product to g; returns g's result.
        x = g(v0 * x)
        return x

      x = constant_op.constant(3.0)
      with backprop.GradientTape(persistent=True) as tape:
        y = f(x)
      dy_v0 = tape.gradient(y, v0)
      dy_v1 = tape.gradient(y, v1)

    self.assertEqual(45, y.numpy())
    self.assertEqual(9, dy_v0.numpy())
    self.assertEqual(15, dy_v1.numpy())

  def testWhileInDefun(self):
    # [FUNCTION DOC]
    # Runs a while loop (increment by one while less than thirteen) inside a wrapped function,
    # starting from three, and asserts the result is thirteen.
    with self.test_scope():
      @def_function.function
      def f(start):
        # [FUNCTION DOC]
        # HELPER: function body for testWhileInDefun.
        # Loops from the start value, adding one each iteration while the value is below
        # thirteen; returns the final value.
        c = lambda x: math_ops.less(x, 13.0)
        b = lambda x: math_ops.add(x, 1.0)
        return while_loop.while_loop(c, b, [start])

      y = f(constant_op.constant(3.0))
    self.assertEqual(13.0, y.numpy())

  def testAutoGraphWhileInDefun(self):
    # [FUNCTION DOC]
    # Same as the while-loop test but written as a native Python loop that the function
    # conversion turns into a graph loop; starting at three, asserts the result is thirteen.
    with self.test_scope():
      @def_function.function
      def f(start):
        # [FUNCTION DOC]
        # HELPER: function body for testAutoGraphWhileInDefun.
        # Adds one to the start value while it is below thirteen; returns the final value.
        x = start
        while x < 13.0:
          x += 1.0
        return x

      y = f(constant_op.constant(3.0))
    self.assertEqual(13.0, y.numpy())

  def testCondInDefun(self):
    # [FUNCTION DOC]
    # Runs a conditional op inside a wrapped function that adds one when the predicate is true and
    # subtracts one otherwise. With value ten, asserts results eleven (true) and nine (false).
    with self.test_scope():
      @def_function.function
      def f(pred, value):
        # [FUNCTION DOC]
        # HELPER: function body for testCondInDefun.
        # Returns value plus one if pred is true, otherwise value minus one, using a graph
        # conditional.
        fn1 = lambda: math_ops.add(value, 1.0)
        fn2 = lambda: math_ops.subtract(value, 1.0)
        return cond.cond(pred, fn1, fn2)

      plus_one = f(constant_op.constant(True), constant_op.constant(10.0))
      minus_one = f(constant_op.constant(False), constant_op.constant(10.0))
    self.assertEqual(11.0, plus_one.numpy())
    self.assertEqual(9.0, minus_one.numpy())

  def testAutoGraphCondInDefun(self):
    # [FUNCTION DOC]
    # Same as the conditional test but using a native Python if statement converted by the
    # function conversion; asserts eleven for a true predicate and nine for false with value ten.
    with self.test_scope():
      @def_function.function
      def f(pred, value):
        # [FUNCTION DOC]
        # HELPER: function body for testAutoGraphCondInDefun.
        # Returns value plus one if pred is true, otherwise value minus one.
        if pred:
          return value + 1.0
        else:
          return value - 1.0

      plus_one = f(constant_op.constant(True), constant_op.constant(10.0))
      minus_one = f(constant_op.constant(False), constant_op.constant(10.0))
    self.assertEqual(11.0, plus_one.numpy())
    self.assertEqual(9.0, minus_one.numpy())

  def testScanInDefun(self):
    # [FUNCTION DOC]
    # Runs a scan (running product) over [1..6] with initializer two inside a wrapped function and
    # asserts the result is [2, 4, 12, 48, 240, 1440].
    with self.test_scope():
      elems = constant_op.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='data')
      v = constant_op.constant(2.0, name='v')

      @def_function.function
      def f(y):
        # [FUNCTION DOC]
        # HELPER: function body for testScanInDefun.
        # Scans the input sequence with multiplication starting from the captured initializer v;
        # returns the sequence of running products.
        # pylint: disable=unnecessary-lambda
        return functional_ops.scan(
            lambda a, x: math_ops.multiply(a, x), y, initializer=v)
        # pylint: enable=unnecessary-lambda

      r = f(elems)
      self.assertAllEqual([2., 4., 12., 48., 240., 1440.], self.evaluate(r))

  def testFeedDeviceMemoryToOpExpectingHostMemory(self):
    # [FUNCTION DOC]
    # Creates an int64 tensor on the test device and feeds it to a wrapped function (run outside
    # the test scope) that fills a shape given by that tensor, an input normally expected in
    # host memory. Fills with value three for shape [4]; asserts the result is four threes.
    @def_function.function
    def f(dims, value):
      # [FUNCTION DOC]
      # HELPER: function body for this test.
      # Returns a tensor with the shape given by dims filled with value.
      return array_ops.fill(dims, value)

    with self.test_scope():
      x = constant_op.constant([4], dtype=dtypes.int64)

    y = f(x, 3)
    self.assertAllEqual([3, 3, 3, 3], y)

  def testRequestNotToCompile(self):
    # [FUNCTION DOC]
    # Wraps one function twice: wholly compiled (default) and with compilation disabled
    # (op-by-op). The function returns its input and twice the input computed on the CPU device.
    # Asserts values are [0, 2] and [0, 4] in both modes; when running eagerly, asserts that
    # the compiled version places both outputs on the test device while the op-by-op version
    # honors the CPU request for the second output.
    with self.test_scope():
      def f(x):
        # [FUNCTION DOC]
        # HELPER: function body for testRequestNotToCompile.
        # Computes twice x under an explicit CPU device request and returns x and that product.
        with ops.device('device:CPU:0'):
          y = 2.0 * x
        return x, y

      wholly_compiled_f = def_function.function(f)
      op_by_op_f = def_function.function(f, jit_compile=False)

      x = array_ops.identity([0.0, 2.0], name='data')

      # When function is wholly compiled, all outputs will be on the
      # device on which it is run.
      r_x, r_y = wholly_compiled_f(x)
      self.assertAllEqual([0.0, 2.0], r_x)
      self.assertAllEqual([0.0, 4.0], r_y)
      if context.executing_eagerly():
        # backing_device is only available for eager tensors.
        self.assertRegex(r_x.backing_device, self.device)
        self.assertRegex(r_y.backing_device, self.device)

      # When function is executed op-by-op, requested devices will be
      # respected.
      r_x, r_y = op_by_op_f(x)
      self.assertAllEqual([0.0, 2.0], r_x)
      self.assertAllEqual([0.0, 4.0], r_y)
      if context.executing_eagerly():
        # backing_device is only available for eager tensors.
        self.assertRegex(r_x.backing_device, self.device)
        self.assertRegex(r_y.backing_device, 'device:CPU:0')


class ExcessivePaddingTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # Tests that tensors which TPU would flatten to one dimension (to avoid excessive padding)
  # still behave correctly in eager mode and as function inputs and outputs. Transparent on
  # other backends.
  """Test that eager execution works with TPU flattened tensors.

  Tensors that would normally be excessively padded when written
  to TPU memory are reshaped to 1-D flat tensors.

  This test case verifies that such tensors work with eager execution.

  The flattening currently only happens on TPU, but tests should work
  fine with all backends as flattening is transparent.
  """

  def testFromConstant(self):
    # [FUNCTION DOC]
    # Builds a constant of shape one hundred by two by one, reduces it with a sum over axis one and
    # asserts every result is twelve.
    with self.test_scope():
      # Create constant of shape [100, 2, 1]. This tensor would be
      # excessively padded on TPU.
      tensor = constant_op.constant(100 * [[[10.0], [2.0]]])
      # Use reduce_sum since it requires correctly working with
      # a particular dimension.
      reduced = math_ops.reduce_sum(tensor, axis=1)
      self.assertAllEqual(100 * [[12.0]], reduced)

  def testFromOperation(self):
    # [FUNCTION DOC]
    # Builds a tensor of ones of shape three by one hundred by two by two with an op, sums over
    # axes zero, two and three and asserts one hundred values of twelve.
    with self.test_scope():
      tensor = array_ops.ones([3, 100, 2, 2])
      reduced = math_ops.reduce_sum(tensor, axis=[0, 2, 3])
      self.assertAllEqual(100 * [12.0], reduced)

  def testAsFunctionInput(self):
    # [FUNCTION DOC]
    # Passes a constant of shape one hundred by one by two to a wrapped function that sums over
    # axis two; asserts the result is one hundred rows of twelve.
    with self.test_scope():

      @def_function.function
      def f(x):
        # [FUNCTION DOC]
        # HELPER: function body for testAsFunctionInput.
        # Returns the sum of the input over axis two.
        return math_ops.reduce_sum(x, axis=2)

      tensor = constant_op.constant(100 * [[[10.0, 2.0]]])
      reduced = f(tensor)
      self.assertAllEqual(100 * [[12.0]], reduced)

  def testAsFunctionOutput(self):
    # [FUNCTION DOC]
    # Calls a wrapped function that multiplies its input by a large constant (shape one hundred
    # by one by two); with input three, sums the output over axis two and asserts one hundred
    # rows of thirty-six.
    with self.test_scope():

      @def_function.function
      def f(x):
        # [FUNCTION DOC]
        # HELPER: function body for testAsFunctionOutput.
        # Returns the input multiplied by a constant of shape one hundred by one by two.
        return x * constant_op.constant(100 * [[[10.0, 2.0]]])

      y = f(3)
      reduced = math_ops.reduce_sum(y, axis=2)
      self.assertAllEqual(100 * [[36.0]], reduced)


def multiple_tpus():
  # [FUNCTION DOC]
  # HELPER: guard for the multi-device test.
  # Queries the eager context for its device names and returns true if more than one is a TPU
  # device, otherwise false. Takes no arguments.
  devices = context.context().devices()
  return len([d for d in devices if 'device:TPU:' in d]) > 1


class MultiDeviceTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # Tests running computations on more than one TPU core. Skipped when fewer than two TPUs exist.
  """Test running TPU computation on more than one core."""

  def testBasic(self):
    # [FUNCTION DOC]
    # Skips unless multiple TPUs exist. Computes two times five on the first TPU and two times
    # three on the second TPU, asserting ten and six, then adds both on the host and asserts
    # sixteen (checking cross-device copies).
    if not multiple_tpus():
      self.skipTest('MultiDeviceTest requires multiple TPU devices.')

    # Compute 10 on TPU core 0
    with ops.device('device:TPU:0'):
      two = constant_op.constant(2)
      five = constant_op.constant(5)
      ten = two * five
      self.assertAllEqual(10, ten)

    # Compute 6 on TPU core 1
    with ops.device('device:TPU:1'):
      two = constant_op.constant(2)
      three = constant_op.constant(3)
      six = two * three
      self.assertAllEqual(6, six)

    # Copy 10 and 6 to CPU and sum them
    self.assertAllEqual(16, ten + six)


if __name__ == '__main__':
  ops.enable_eager_execution(
      config=config_pb2.ConfigProto(log_device_placement=True))
  googletest.main()
