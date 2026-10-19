# [MODULE DOC]
# Tests for eager execution on XLA-backed devices (CPU/GPU/TPU depending on the test target).
# Defines four test classes: basic eager ops, eager execution of compiled functions, tensors that
# would be heavily padded on TPU, and multi-TPU placement. No import-time side effects; when run
# as a script it enables eager execution with device placement logging and starts the test runner.
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
  # Tests eager-mode ops (constants, gradients, variables, shapes, optimizers) placed on the XLA
  # test device via the test scope. Inherits from the XLA test base class.

  def testBasic(self):
    # [FUNCTION DOC]
    # Multiplies two constants eagerly on the test device and asserts the product is 15.
    with self.test_scope():
      three = constant_op.constant(3)
      five = constant_op.constant(5)
      product = three * five
      self.assertAllEqual(15, product)

  def testGradientTape(self):
    # [FUNCTION DOC]
    # Uses a persistent gradient tape to differentiate a sum-plus-product of two watched constants
    # (1.0 and 10.0); asserts the partial derivatives are 11.0 and 2.0.
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
    # Unstacks an empty tensor along axis 0 and asserts the result is an empty list.
    with self.test_scope():
      empty = constant_op.constant([], dtype=dtypes.float32)
      result = array_ops_stack.unstack(empty, 0)
      self.assertTrue(isinstance(result, list))
      self.assertEqual(0, len(result))

  def testExecuteListOutputLen1(self):
    # [FUNCTION DOC]
    # Splits a 2x3 tensor into one piece along axis 1; asserts a list of length one equal to input.
    with self.test_scope():
      split_dim = constant_op.constant(1)
      value = constant_op.constant([[0., 1., 2.], [3., 4., 5.]])
      result = array_ops.split(value, 1, axis=split_dim)
      self.assertTrue(isinstance(result, list))
      self.assertEqual(1, len(result))
      self.assertAllEqual([[0, 1, 2], [3, 4, 5]], result[0])

  def testExecuteListOutputLen3(self):
    # [FUNCTION DOC]
    # Splits a 2x3 tensor into three pieces along axis 1; asserts three columns of two elements.
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
    # Runs a multiplication eagerly and then again in graph mode inside a session; both must give 15.
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
    # Slices a 3x2x3 tensor with degenerate (empty, reversed or negative-stride) ranges and asserts
    # results match the equivalent numpy slices.
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
    # Asserts the identity op returns its scalar input unchanged.
    with self.test_scope():
      self.assertAllEqual(2, array_ops.identity(2))

  def testRandomOps(self):
    # [FUNCTION DOC]
    # Draws a 2x2 uniform random tensor and asserts its two rows are not all equal (a sanity check
    # that the generator produces varying values).
    with self.test_scope():
      tensor = gen_random_ops.random_uniform((2, 2), dtypes.float32)
      row0 = tensor[0].numpy()
      row1 = tensor[1].numpy()
      # It should be very unlikely to rng to generate two equal rows.
      self.assertFalse((row0 == row1).all())

  def testIdentityOnVariable(self):
    # [FUNCTION DOC]
    # Applies identity to a boolean resource variable and asserts the value read back is True.
    with self.test_scope():
      v = resource_variable_ops.ResourceVariable(True)
      i = array_ops.identity(v)
    self.assertAllEqual(True, i.numpy())

  def testAssignAddVariable(self):
    # [FUNCTION DOC]
    # Creates a variable of 1.0, adds 2.0 in place, and asserts it holds 3.0.
    with self.test_scope():
      v = resource_variable_ops.ResourceVariable(1.0)
      v.assign_add(2.0)
    self.assertEqual(3.0, v.numpy())

  def testReadAssignRead(self):
    # [FUNCTION DOC]
    # Reads a variable, increments it, reads again; asserts the first read stays 1.0 (not aliased to
    # the update) and the second is 3.0.
    with self.test_scope():
      v = resource_variable_ops.ResourceVariable(1.0)
      val1 = v.read_value()
      v.assign_add(2.0)
      val2 = v.read_value()
    self.assertEqual(1.0, val1.numpy())
    self.assertEqual(3.0, val2.numpy())

  def testGradient(self):
    # [FUNCTION DOC]
    # Computes the gradient function of an identity function with an upstream gradient of 2.0 and
    # asserts the gradient is 2.0.
    def f(x):
      # [FUNCTION DOC]
      # HELPER: identity function used as the differentiated function in testGradient.
      # Returns its input unchanged.
      return x

    with self.test_scope():
      grad_fn = backprop.gradients_function(f)
      self.assertAllEqual(2., grad_fn(1., dy=2.)[0])

  def testVariableGradient(self):
    # [FUNCTION DOC]
    # Computes the implicit gradient of the square of a variable (1.0) and asserts it equals 2.0.
    with self.test_scope():
      v0 = resource_variable_ops.ResourceVariable(1.0)

      def f():
        # [FUNCTION DOC]
        # HELPER: function differentiated in testVariableGradient; returns the variable squared.
        x = v0 * v0
        return x

      grads = backprop.implicit_grad(f)()
    self.assertEqual(2., grads[0][0].numpy())

  def testMultipleVariableReads(self):
    # [FUNCTION DOC]
    # Creates a large (about 128 MiB) variable and reads it 100 times. Only checks no out-of-memory
    # occurs, which would indicate each read copies the underlying memory. No assertions.
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
    # Verifies the shape op, for both directly constructed constants and device-produced tensors of
    # ranks 0 to 3, returns the expected shapes.
    def const(value):
      # [FUNCTION DOC]
      # HELPER: returns the shape (as numpy) of a constant built from the given value.
      return array_ops.shape(
          constant_op.constant(value)).numpy()

    def ones(value):
      # [FUNCTION DOC]
      # HELPER: returns the shape (as numpy) of a ones tensor created with the given shape.
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
    # Verifies shape_n over several tensors, both constants and device-computed, returns the
    # expected shapes (scalar, vector of 3, 2x2).
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
    # Asserts the size op returns 1, 3 and 4 for a scalar, a 3-vector and a 2x2 matrix.
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
    # Asserts the rank op returns 0, 1 and 2 for a scalar, a 3-vector and a 2x2 matrix.
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
    # Minimizes the square of a variable (10.0) with one step of Adam (rate 0.1) and asserts the
    # result is about 9.9 to three decimal places.
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
    # Applies Adam to a sparse (indexed-slices) gradient from an embedding lookup; the embedding
    # variable lives on CPU because sparse updates are unsupported on TPU. Then adds ones to the
    # variable and asserts the updated values (row 1 is 1.9 after the step, others 2.0).
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
  # Tests eager execution of compiled functions (function-decorated callables) on the XLA test
  # device: variables, gradient tapes, nesting, control flow and device placement.

  def testBasic(self):
    # [FUNCTION DOC]
    # Wraps matmul in a function, calls it with transpose of the first argument, and asserts the
    # product of [[1,2],[3,4]] with itself transposed.
    with self.test_scope():
      matmul = def_function.function(math_ops.matmul)
      t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
      sq = matmul(t, t, transpose_a=True)
      self.assertAllEqual(sq.numpy().reshape(-1), [10, 14, 14, 20])

  def testConv(self):
    # [FUNCTION DOC]
    # Skipped on GPU (the random normal kernel is slow there). Builds a convolution plus max pooling
    # model in a function and asserts a 4x4 ones input yields a single value 4.
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
        # HELPER: applies the convolution then the pooling layer to the input.
        x = conv(x)
        return pool(x)
      model = def_function.function(model)

      x = array_ops.ones([1, 4, 4, 1])
      y = model(x)
      self.assertAllEqual(y.numpy(), [[[[4.]]]])

  def testReadVariable(self):
    # [FUNCTION DOC]
    # Reads a captured variable inside a function and asserts the value 1.0.
    with self.test_scope():
      v = resource_variable_ops.ResourceVariable(1.0)

      @def_function.function
      def f():
        # [FUNCTION DOC]
        # HELPER: function body that returns the current value of the captured variable.
        return v.read_value()

      var = f()
      self.assertEqual(1.0, var.numpy())

  def testResourceVariableNoInlineReadWrite(self):
    # [FUNCTION DOC]
    # Calls non-inlined functions that both read and write captured variables several times; asserts
    # the combined result 145 and final written variable 15.
    with self.test_scope():
      v = resource_variable_ops.ResourceVariable(1.0)
      w = resource_variable_ops.ResourceVariable(0.0)

      @def_function.function(experimental_attributes={'_noinline': True})
      def g(x):
        # [FUNCTION DOC]
        # HELPER: adds x to a variable w and returns the other variable plus x times w.
        w.assign(w.read_value() + x)
        return v.read_value() + x * w.read_value()

      @def_function.function(experimental_attributes={'_noinline': True})
      def f():
        # [FUNCTION DOC]
        # HELPER: sums the results of five calls to the inner function with arguments 1 to 5.
        return g(1.0) + g(2.0) + g(3.0) + g(4.0) + g(5.0)

      # 1 + 1*1 + 1 + 2*3 + 1 + 3*6 + 1 + 4*10 + 1 + 5*15
      self.assertEqual(145.0, f().numpy())
      self.assertEqual(15.0, w.read_value().numpy())

  def testResourceVariableNoInlineReadOnly(self):
    # [FUNCTION DOC]
    # Non-inlined functions that only read a variable (10.0); calling five times sums to 50.
    with self.test_scope():
      v = resource_variable_ops.ResourceVariable(10.0)

      @def_function.function(experimental_attributes={'_noinline': True})
      def g():
        # [FUNCTION DOC]
        # HELPER: returns the value of the captured variable.
        return v.read_value()

      @def_function.function(experimental_attributes={'_noinline': True})
      def f():
        # [FUNCTION DOC]
        # HELPER: sums five calls to the read-only inner function.
        return g() + g() + g() + g() + g()

      self.assertEqual(50.0, f().numpy())

  def testResourceVariableNoInlineWriteOnly(self):
    # [FUNCTION DOC]
    # Non-inlined functions that only write a variable; after writing 1 through 5, asserts it is 5.
    with self.test_scope():
      v = resource_variable_ops.ResourceVariable(0.0)

      @def_function.function(experimental_attributes={'_noinline': True})
      def g(x):
        # [FUNCTION DOC]
        # HELPER: assigns x to the captured variable; returns nothing.
        v.assign(x)

      @def_function.function(experimental_attributes={'_noinline': True})
      def f():
        # [FUNCTION DOC]
        # HELPER: calls the writer function with 1.0 through 5.0; returns nothing.
        g(1.0)
        g(2.0)
        g(3.0)
        g(4.0)
        g(5.0)

      f()
      self.assertEqual(5.0, v.read_value().numpy())

  def testUpdateVariable(self):
    # [FUNCTION DOC]
    # A function takes a variable argument, increments it, and returns it; asserts the result is 2.0.
    with self.test_scope():
      v = resource_variable_ops.ResourceVariable(1.0)

      def f(v):
        # [FUNCTION DOC]
        # HELPER: adds 1.0 to the given variable in place and returns the variable.
        v.assign_add(1.0)
        return v

      f = def_function.function(f)

      var = f(v)
      self.assertEqual(2.0, var.numpy())

  def testReturnResourceHandle(self):
    # [FUNCTION DOC]
    # A function returns a variable's resource handle; asserts reading the handle gives the
    # variable's value.
    with self.test_scope():
      v = resource_variable_ops.ResourceVariable([[1.0, 2.0], [3.0, 4.0]])

      def f(v):
        # [FUNCTION DOC]
        # HELPER: returns the resource handle of the given variable.
        return v.handle

      f = def_function.function(f)
      handle = f(v)
      self.assertAllEqual(v.numpy(),
                          resource_variable_ops.read_variable_op(
                              handle, dtypes.float32).numpy())

  def testReturnMultipleResourceHandles(self):
    # [FUNCTION DOC]
    # A function returns two handles plus computed values (3 times a variable, sum of variables);
    # asserts the handles read back correctly and the values are 3.75 and 3.25.
    with self.test_scope():
      v1 = resource_variable_ops.ResourceVariable(1.25)
      v2 = resource_variable_ops.ResourceVariable(2.0)

      def f(v):
        # [FUNCTION DOC]
        # HELPER: returns a 4-tuple: the argument's handle, three times the argument, the captured
        # variable's handle, and the sum of the argument and the captured variable.
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
    # Calls a function mixing compile-time constants, regular tensors and resource variables in
    # shuffled argument order, including a constant also used in addition; asserts all four outputs.
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
        # HELPER: computes a sum of constant and tensor, a cast constant plus variable, a slice of
        # a variable, and a product of tensor and variable; returns the four results.
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
    # Differentiates a function (variable squared times input) with respect to the variable under
    # a tape outside the function; asserts value 75 and gradient 30.
    with self.test_scope():
      v0 = resource_variable_ops.ResourceVariable(5.0)

      @def_function.function
      def f(x):
        # [FUNCTION DOC]
        # HELPER: returns the captured variable squared times x.
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
    # Computes a gradient with a tape created inside a function; asserts it equals 1.0.
    with self.test_scope():
      v0 = resource_variable_ops.ResourceVariable(5.0)

      @def_function.function
      def f():
        # [FUNCTION DOC]
        # HELPER: builds a tape inside the function and returns the gradient of variable times a
        # constant with respect to the variable.
        x = constant_op.constant(1.0)
        with backprop.GradientTape() as tape:
          y = v0 * x
        dy = tape.gradient(y, v0)
        return dy

      dy = f()
      self.assertEqual(1.0, dy.numpy())

  def testSliceInDefun(self):
    # [FUNCTION DOC]
    # Strided slicing with a tensor index inside a function, with gradient w.r.t. the input;
    # asserts the slice is ones of shape 1x2x4 and the gradient has shape 2x3x4.
    with self.test_scope():

      @def_function.function
      def f(x, y):
        # [FUNCTION DOC]
        # HELPER: returns the slice of x with every second row on axis 0 and starting at y on axis 1.
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
    # Calls a function that calls another; asserts 2x+1 on [2,3,4] gives [5,7,9].
    with self.test_scope():

      @def_function.function
      def times_two(x):
        # [FUNCTION DOC]
        # HELPER: returns twice the input.
        return 2. * x

      @def_function.function
      def two_x_plus_1(x):
        # [FUNCTION DOC]
        # HELPER: returns the doubling function applied to x, plus one.
        return times_two(x) + 1.

      x = constant_op.constant([2., 3., 4.])
      y = two_x_plus_1(x)
      self.assertAllEqual([5., 7., 9.], y.numpy())

  def testNestedDefunWithVariable(self):
    # [FUNCTION DOC]
    # Nested functions that both use a captured variable (5.0); asserts the result 75 for input 3.
    with self.test_scope():
      v0 = resource_variable_ops.ResourceVariable(5.0)

      @def_function.function
      def g(x):
        # [FUNCTION DOC]
        # HELPER: inner function; multiplies x by the captured variable.
        x = v0 * x
        return x

      @def_function.function
      def f(x):
        # [FUNCTION DOC]
        # HELPER: outer function; multiplies x by the variable and then calls the inner function.
        x = g(v0 * x)
        return x

      x = constant_op.constant(3.0)
      y = f(x)

    self.assertEqual(75.0, y.numpy())

  def testNestedDefunInGradientTape(self):
    # [FUNCTION DOC]
    # Same nested-function setup, differentiated under a tape; asserts value 75 and gradient 30.
    with self.test_scope():
      v0 = resource_variable_ops.ResourceVariable(5.0)

      @def_function.function
      def g(x):
        # [FUNCTION DOC]
        # HELPER: inner function; multiplies x by the captured variable.
        x = v0 * x
        return x

      @def_function.function
      def f(x):
        # [FUNCTION DOC]
        # HELPER: outer function; multiplies x by the variable and calls the inner function.
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
    # Nested functions using two different variables under a persistent tape; asserts value 45 and
    # gradients 9 and 15 for the two variables.
    with self.test_scope():
      v0 = resource_variable_ops.ResourceVariable(5.0)
      v1 = resource_variable_ops.ResourceVariable(3.0)

      @def_function.function
      def g(x):
        # [FUNCTION DOC]
        # HELPER: inner function; multiplies x by the second variable.
        x = v1 * x
        return x

      @def_function.function
      def f(x):
        # [FUNCTION DOC]
        # HELPER: outer function; multiplies x by the first variable and calls the inner function.
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
    # Runs an explicit while loop inside a function incrementing from 3.0 until 13.0; asserts 13.0.
    with self.test_scope():
      @def_function.function
      def f(start):
        # [FUNCTION DOC]
        # HELPER: loops adding 1.0 while the value is below 13.0; returns the loop result.
        c = lambda x: math_ops.less(x, 13.0)
        b = lambda x: math_ops.add(x, 1.0)
        return while_loop.while_loop(c, b, [start])

      y = f(constant_op.constant(3.0))
    self.assertEqual(13.0, y.numpy())

  def testAutoGraphWhileInDefun(self):
    # [FUNCTION DOC]
    # Same as the while test but with an ordinary Python-style loop converted by autograph.
    with self.test_scope():
      @def_function.function
      def f(start):
        # [FUNCTION DOC]
        # HELPER: increments the start value by 1.0 while it is below 13.0 and returns it.
        x = start
        while x < 13.0:
          x += 1.0
        return x

      y = f(constant_op.constant(3.0))
    self.assertEqual(13.0, y.numpy())

  def testCondInDefun(self):
    # [FUNCTION DOC]
    # Conditional inside a function using the cond op; asserts adding or subtracting 1 from 10.
    # gives 11 and 9 depending on the predicate.
    with self.test_scope():
      @def_function.function
      def f(pred, value):
        # [FUNCTION DOC]
        # HELPER: returns value plus 1.0 if pred is true, otherwise value minus 1.0.
        fn1 = lambda: math_ops.add(value, 1.0)
        fn2 = lambda: math_ops.subtract(value, 1.0)
        return cond.cond(pred, fn1, fn2)

      plus_one = f(constant_op.constant(True), constant_op.constant(10.0))
      minus_one = f(constant_op.constant(False), constant_op.constant(10.0))
    self.assertEqual(11.0, plus_one.numpy())
    self.assertEqual(9.0, minus_one.numpy())

  def testAutoGraphCondInDefun(self):
    # [FUNCTION DOC]
    # Same as the cond test but using an ordinary conditional converted by autograph.
    with self.test_scope():
      @def_function.function
      def f(pred, value):
        # [FUNCTION DOC]
        # HELPER: returns value plus 1.0 if pred is true, otherwise value minus 1.0.
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
    # Runs a cumulative scan (multiplication with initial value 2.0) over 1..6 inside a function;
    # asserts the results [2, 4, 12, 48, 240, 1440].
    with self.test_scope():
      elems = constant_op.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='data')
      v = constant_op.constant(2.0, name='v')

      @def_function.function
      def f(y):
        # [FUNCTION DOC]
        # HELPER: scans the multiply function over y with the captured initializer.
        # pylint: disable=unnecessary-lambda
        return functional_ops.scan(
            lambda a, x: math_ops.multiply(a, x), y, initializer=v)
        # pylint: enable=unnecessary-lambda

      r = f(elems)
      self.assertAllEqual([2., 4., 12., 48., 240., 1440.], self.evaluate(r))

  def testFeedDeviceMemoryToOpExpectingHostMemory(self):
    # [FUNCTION DOC]
    # Passes a device-resident int64 tensor as the dims argument of a fill op (which expects host
    # memory); asserts the result is four 3s.
    @def_function.function
    def f(dims, value):
      # [FUNCTION DOC]
      # HELPER: returns a tensor of shape dims filled with value.
      return array_ops.fill(dims, value)

    with self.test_scope():
      x = constant_op.constant([4], dtype=dtypes.int64)

    y = f(x, 3)
    self.assertAllEqual([3, 3, 3, 3], y)

  def testRequestNotToCompile(self):
    # [FUNCTION DOC]
    # Compares a fully compiled function with one marked not to compile (op-by-op). The function
    # computes a product on CPU; in the compiled case both outputs sit on the test device, in the
    # op-by-op case the second output sits on CPU. Device checks happen only in eager mode.
    with self.test_scope():
      def f(x):
        # [FUNCTION DOC]
        # HELPER: returns x and twice x, the latter computed under an explicit CPU device scope.
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
  # Tests eager execution with tensors that TPU would flatten to avoid excessive padding; the
  # behavior must be transparent on every backend.
  """Test that eager execution works with TPU flattened tensors.

  Tensors that would normally be excessively padded when written
  to TPU memory are reshaped to 1-D flat tensors.

  This test case verifies that such tensors work with eager execution.

  The flattening currently only happens on TPU, but tests should work
  fine with all backends as flattening is transparent.
  """

  def testFromConstant(self):
    # [FUNCTION DOC]
    # Reduces a constant of shape 100x2x1 along axis 1; asserts 100 rows of 12.0.
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
    # Reduces a ones tensor of shape 3x100x2x2 over axes 0, 2 and 3; asserts 100 values of 12.0.
    with self.test_scope():
      tensor = array_ops.ones([3, 100, 2, 2])
      reduced = math_ops.reduce_sum(tensor, axis=[0, 2, 3])
      self.assertAllEqual(100 * [12.0], reduced)

  def testAsFunctionInput(self):
    # [FUNCTION DOC]
    # Passes a padded-prone tensor into a function that sums axis 2; asserts 100 values of 12.0.
    with self.test_scope():

      @def_function.function
      def f(x):
        # [FUNCTION DOC]
        # HELPER: sums x along axis 2.
        return math_ops.reduce_sum(x, axis=2)

      tensor = constant_op.constant(100 * [[[10.0, 2.0]]])
      reduced = f(tensor)
      self.assertAllEqual(100 * [[12.0]], reduced)

  def testAsFunctionOutput(self):
    # [FUNCTION DOC]
    # A function returns a large constant scaled by 3; sums over axis 2 outside and asserts 36.0.
    with self.test_scope():

      @def_function.function
      def f(x):
        # [FUNCTION DOC]
        # HELPER: multiplies x by a constant of shape 100x1x2.
        return x * constant_op.constant(100 * [[[10.0, 2.0]]])

      y = f(3)
      reduced = math_ops.reduce_sum(y, axis=2)
      self.assertAllEqual(100 * [[36.0]], reduced)


def multiple_tpus():
  # [FUNCTION DOC]
  # HELPER: tells whether more than one TPU device is available to the eager context.
  # Returns true if the context lists more than one device whose name contains the TPU marker.
  devices = context.context().devices()
  return len([d for d in devices if 'device:TPU:' in d]) > 1


class MultiDeviceTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # Tests eager execution spread across more than one TPU core.
  """Test running TPU computation on more than one core."""

  def testBasic(self):
    # [FUNCTION DOC]
    # Skipped unless multiple TPUs exist. Computes 10 on TPU 0 and 6 on TPU 1, then adds them
    # (copying across devices) and asserts 16.
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
