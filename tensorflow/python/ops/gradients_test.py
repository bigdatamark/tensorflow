# [MODULE DOC]
# Unit tests for the gradient-construction code in the gradients and gradients_impl modules and
# related custom-gradient utilities. Test classes cover gradients (device colocation, aggregation,
# stop gradients, unconnected gradients), function gradients, Hessians, indexed slices, custom
# gradients, recompute-gradient and grad pass-through. Running the file starts the test runner.
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
"""Tests for tensorflow.ops.gradients."""

import sys
import warnings

from absl.testing import parameterized
import numpy as np
from tensorflow.python.client import session
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function as framework_function
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import test_ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework.constant_op import constant
from tensorflow.python.layers import core as core_layers
from tensorflow.python.ops import array_grad  # pylint: disable=unused-import
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_grad  # pylint: disable=unused-import
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import data_flow_grad  # pylint: disable=unused-import
from tensorflow.python.ops import data_flow_ops  # pylint: disable=unused-import
from tensorflow.python.ops import functional_ops  # pylint: disable=unused-import
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import gradients
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import gradients_util
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import math_grad  # pylint: disable=unused-import
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_grad  # pylint: disable=unused-import
from tensorflow.python.ops import ref_variable
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_grad  # pylint: disable=unused-import
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import unconnected_gradients
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.nn_ops import bias_add
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import googletest
from tensorflow.python.util import nest


class GradientsTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Main test case for the graph-mode gradients function: checks generated gradient op types,
  # device colocation, aggregation methods, stop-gradients handling, unconnected-gradient options,
  # string outputs, indexed slices, variables, and identity-n optimization. Each test builds its
  # own graph; no shared fixtures. Supports parameterized tests.

  def testGradients(self):
    # [FUNCTION DOC]
    # Builds a matmul followed by bias add and takes the gradient of the result with respect to the
    # weight constant. Asserts the gradient op is a matmul whose original forward op is the forward
    # matmul, with the first operand transposed and the second not.
    with ops.Graph().as_default():
      inp = constant(1.0, shape=[32, 100], name="in")
      w = constant(1.0, shape=[100, 10], name="w")
      b = constant(1.0, shape=[10], name="b")
      xw = math_ops.matmul(inp, w, name="xw")
      h = bias_add(xw, b, name="h")
      w_grad = gradients.gradients(h, w)[0]
    self.assertEqual("MatMul", w_grad.op.type)
    self.assertEqual(w_grad.op._original_op, xw.op)
    self.assertTrue(w_grad.op.get_attr("transpose_a"))
    self.assertFalse(w_grad.op.get_attr("transpose_b"))

  def testUnusedOutput(self):
    # [FUNCTION DOC]
    # Splits a matmul result in two and reduces only the second piece, so one split output is
    # unused. Asserts the gradient with respect to the left operand can still be built and its op
    # is a matmul.
    with ops.Graph().as_default():
      w = constant(1.0, shape=[2, 2])
      x = constant(1.0, shape=[2, 2])
      wx = math_ops.matmul(w, x)
      split_wx = array_ops.split(value=wx, num_or_size_splits=2, axis=0)
      c = math_ops.reduce_sum(split_wx[1])
      gw = gradients.gradients(c, [w])[0]
    self.assertEqual("MatMul", gw.op.type)

  def testColocateGradients(self):
    # [FUNCTION DOC]
    # Places a matmul on a GPU device and requests gradients with colocation enabled. Asserts the
    # gradient op has the same colocation groups as the forward matmul.
    with ops.Graph().as_default() as g:
      w = constant(1.0, shape=[1, 1])
      x = constant(1.0, shape=[1, 2])
      with g.device("/device:GPU:0"):
        wx = math_ops.matmul(w, x)
      gw = gradients.gradients(wx, [w], colocate_gradients_with_ops=True)[0]
    self.assertEqual(gw.op.colocation_groups(), wx.op.colocation_groups())

  def testColocateGradientsWithAggregation(self):
    # [FUNCTION DOC]
    # A weight on one GPU feeds two matmuls summed on another GPU, so its gradient terms must be
    # aggregated. Asserts that with colocation enabled the gradient shares the forward matmul's
    # colocation groups, and that with colocation disabled it does not.
    with ops.Graph().as_default() as g:
      with g.device("/device:GPU:1"):
        w = constant(1.0, shape=[1, 1])
      x = constant(1.0, shape=[1, 2])
      y = constant(1.0, shape=[1, 2])
      wx = math_ops.matmul(w, x)
      wy = math_ops.matmul(w, y)
      with g.device("/device:GPU:0"):
        z = wx + wy

      gw1 = gradients.gradients(z, [w], colocate_gradients_with_ops=True)[0]
      self.assertEqual(gw1.op.colocation_groups(), wx.op.colocation_groups())

      gw2 = gradients.gradients(z, [w], colocate_gradients_with_ops=False)[0]
      self.assertNotEqual(wx.op.colocation_groups(), gw2.op.colocation_groups())

  def testColocateGradientsWithAggregationInMultipleDevices(self):
    # [FUNCTION DOC]
    # Like the aggregation test, but the two consuming matmuls sit on different tasks. Asserts that
    # with colocation enabled the aggregated gradient is colocated with the weight itself, and
    # that with colocation disabled it is not.
    with ops.Graph().as_default() as g:
      with g.device("/device:GPU:1"):
        w = constant(1.0, shape=[1, 1])
      x = constant(1.0, shape=[1, 2])
      y = constant(1.0, shape=[1, 2])
      with g.device("/task:1"):
        wx = math_ops.matmul(w, x)
      with g.device("/task:2"):
        wy = math_ops.matmul(w, y)
      with g.device("/device:GPU:0"):
        z = wx + wy

      gw1 = gradients.gradients(z, [w], colocate_gradients_with_ops=True)[0]
      self.assertEqual(gw1.op.colocation_groups(), w.op.colocation_groups())

      gw2 = gradients.gradients(z, [w], colocate_gradients_with_ops=False)[0]
      self.assertNotEqual(w.op.colocation_groups(), gw2.op.colocation_groups())

  def testColocateGradientsWithGateGradients(self):
    # [FUNCTION DOC]
    # Skipped when no GPU exists. Builds ops across CPU and GPU, requests gradients with both
    # colocation and gating enabled, then evaluates the gradient to confirm the device placer
    # raises no complaint.
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")
    with ops.Graph().as_default() as g:
      with g.device("/device:CPU:0"):
        x = constant(1.0, shape=[1, 1])
        y = constant(1.0, shape=[1, 1])
        s = x + y
      with g.device("/device:GPU:0"):
        z = math_ops.reduce_sum(s)

      gz_x = gradients.gradients(z, [x], colocate_gradients_with_ops=True,
                                 gate_gradients=True)[0]

      # Make sure the placer doesn't complain.
      self.evaluate(gz_x)

  def testBoundaryStop(self):
    # [FUNCTION DOC]
    # Takes the gradient of a downstream value with respect to an intermediate identity tensor and
    # asserts the gradient code does not differentiate through that tensor, so every returned
    # gradient is present.
    # Test that we don't differentiate 'x'. The gradient function for 'x' is
    # set explicitly to None so we will get an exception if the gradient code
    # tries to differentiate 'x'.
    with ops.Graph().as_default():
      c = constant(1.0)
      x = array_ops.identity(c)
      y = x + 1.0
      z = y + 1
      grads = gradients.gradients(z, [x])
      self.assertTrue(all(x is not None for x in grads))

  @test_util.run_v1_only("b/120545219")
  def testBoundaryContinue(self):
    # [FUNCTION DOC]
    # Graph-mode only. With x feeding y feeding z, differentiates z with respect to both x and y,
    # asserting both gradients exist and the gradient for x evaluates to six.
    # Test that we differentiate both 'x' and 'y' correctly when x is a
    # predecessor of y.
    with self.cached_session():
      x = constant(1.0)
      y = x * 2.0
      z = y * 3.0
      grads = gradients.gradients(z, [x, y])
      self.assertTrue(all(x is not None for x in grads))
      self.assertEqual(6.0, grads[0].eval())

  @test_util.run_v1_only("b/120545219")
  def testAggregationMethodAccumulateN(self):
    # [FUNCTION DOC]
    # Graph-mode only. Sums ten copies of a tensor and takes gradients using the experimental
    # accumulate-n aggregation method. Asserts the gradients are present and equal twenty (input)
    # and ten (intermediate).
    with self.cached_session():
      x = constant(1.0)
      y = x * 2.0
      z = y + y + y + y + y + y + y + y + y + y
      grads = gradients.gradients(
          z, [x, y],
          aggregation_method=gradients.AggregationMethod.
          EXPERIMENTAL_ACCUMULATE_N)
      self.assertTrue(all(x is not None for x in grads))
      self.assertEqual(20.0, grads[0].eval())
      self.assertEqual(10.0, grads[1].eval())

  @test_util.run_v1_only("b/120545219")
  def testAggregationMethodAddN(self):
    # [FUNCTION DOC]
    # Graph-mode only. Same scenario as the accumulate-n test but with the add-n aggregation
    # method; asserts gradients equal twenty and ten.
    with self.cached_session():
      x = constant(1.0)
      y = x * 2.0
      z = y + y + y + y + y + y + y + y + y + y
      grads = gradients.gradients(
          z, [x, y], aggregation_method=gradients.AggregationMethod.ADD_N)
      self.assertTrue(all(x is not None for x in grads))
      self.assertEqual(20.0, grads[0].eval())
      self.assertEqual(10.0, grads[1].eval())

  @test_util.run_v1_only("b/120545219")
  def testAggregationMethodTree(self):
    # [FUNCTION DOC]
    # Graph-mode only. Same scenario as the accumulate-n test but with the experimental tree
    # aggregation method; asserts gradients equal twenty and ten.
    with self.cached_session():
      x = constant(1.0)
      y = x * 2.0
      z = y + y + y + y + y + y + y + y + y + y
      grads = gradients.gradients(
          z, [x, y],
          aggregation_method=gradients.AggregationMethod.EXPERIMENTAL_TREE)
      self.assertTrue(all(x is not None for x in grads))
      self.assertEqual(20.0, grads[0].eval())
      self.assertEqual(10.0, grads[1].eval())

  def testNoGradientForStringOutputs(self):
    # [FUNCTION DOC]
    # Registers a gradient function for a test op that has a float and a string output, then takes
    # gradients through it. The registered function asserts it receives a float32 gradient and a
    # falsy string gradient. Asserts the final gradients are tensors, both when differentiating
    # the first consumer and a later consumer.
    with ops.Graph().as_default():

      def _TestOpGrad(_, float_grad, string_grad):
        # [FUNCTION DOC]
        # HELPER: gradient function registered for the test string-output op in this test.
        # Asserts the float gradient is float32 and the string-output gradient is falsy, then
        # returns the float gradient unchanged.
        """Gradient function for TestStringOutput."""
        self.assertEqual(float_grad.dtype, dtypes.float32)
        self.assertFalse(string_grad)
        return float_grad

      ops.RegisterGradient("TestStringOutput")(_TestOpGrad)

      c = constant(1.0)
      x, _ = test_ops.test_string_output(c)
      z = x * 2.0
      w = z * 3.0
      grads = gradients.gradients(z, [c])
      self.assertIsInstance(grads[0], tensor.Tensor)
      grads = gradients.gradients(w, [c])
      self.assertIsInstance(grads[0], tensor.Tensor)

  def testNoGradientForStringOutputsWithOpNamespace(self):
    # [FUNCTION DOC]
    # Same as the string-output test but for an op registered under a namespace, so the gradient
    # is registered under the namespaced op name. Asserts gradients are tensors.
    with ops.Graph().as_default():

      def _TestOpGrad(_, float_grad, string_grad):
        # [FUNCTION DOC]
        # HELPER: gradient function registered for the namespaced test string-output op.
        # Asserts the float gradient is float32 and the string gradient is falsy; returns the
        # float gradient.
        """Gradient function for TestStringOutput."""
        self.assertEqual(float_grad.dtype, dtypes.float32)
        self.assertFalse(string_grad)
        return float_grad

      ops.RegisterGradient("Namespace>TestStringOutput")(_TestOpGrad)

      c = constant(1.0)
      x, _ = test_ops.namespace_test_string_output(c)
      z = x * 2.0
      w = z * 3.0
      grads = gradients.gradients(z, [c])
      self.assertIsInstance(grads[0], tensor.Tensor)
      grads = gradients.gradients(w, [c])
      self.assertIsInstance(grads[0], tensor.Tensor)

  def testSingletonIndexedSlices(self):
    # [FUNCTION DOC]
    # Passes an indexed-slices value as the initial gradient of an identity op and runs the graph
    # in a session with fed values. Asserts the resulting gradient equals the supplied initial
    # gradient, since identity's gradient is the identity map.
    with ops.Graph().as_default():
      x = array_ops.placeholder(dtypes.float32)
      y = array_ops.identity(x)
      dy = indexed_slices.IndexedSlices(
          array_ops.placeholder(dtypes.float32),
          array_ops.placeholder(dtypes.int32))
      dx, = gradients.gradients(y, x, grad_ys=dy)
      # The IndexedSlices gradient of tf.identity is the identity map.
      with self.cached_session() as sess:
        vdx, vdy = sess.run(
            [dx, dy], feed_dict={x: [1.0], dy.indices: [0], dy.values: [2.0]})
      self.assertEqual(vdx, vdy)

  @test_util.run_v1_only("b/120545219")
  def testNonDifferentiableSwitchInWhileLoop(self):
    # [FUNCTION DOC]
    # Graph-mode only. Runs a while loop whose body casts a float placeholder to an integer, so
    # the value flows through a non-differentiable path into a tensor array. Asserts the gradient
    # of the final read with respect to the placeholder is null.
    with ops.Graph().as_default():
      v = array_ops.placeholder(dtypes.float32, [])

      def _Step(i, a, ta):
        # [FUNCTION DOC]
        # HELPER: while loop body for this test. Adds the integer cast of the placeholder to the
        # accumulator, writes it to the tensor array at the counter, and returns the incremented
        # counter, accumulator and array.
        a += math_ops.cast(v, dtypes.int32)
        return (i + 1, a, ta.write(i, a))

      n = 4
      i, _, ta = while_loop.while_loop(
          lambda i, *_: i < n, _Step,
          [0, 0, tensor_array_ops.TensorArray(dtypes.int32, size=n)])
      target = ta.read(i - 1)
      grad, = gradients.gradients(target, v)
      self.assertIsNone(grad)

  def testVariableReadValueGradient(self):
    # [FUNCTION DOC]
    # Creates a variable and takes the gradient of its read value with respect to the variable;
    # asserts a result is returned (not null).
    with ops.Graph().as_default():
      init = constant_op.constant(100.0)
      var = variables.Variable(init)
      gradient = gradients.gradients(var.read_value(), var)
      self.assertIsNotNone(gradient)

  @parameterized.parameters(dtypes.float32, dtypes.float64)
  def testVariableDefaultGrad(self, dtype):
    # [FUNCTION DOC]
    # Parameterized over float32 and float64. Differentiates an unrelated constant with respect to
    # a variable of the given dtype with unconnected gradients set to zero. Asserts the zero
    # gradient is present and has the variable's dtype.
    with ops.Graph().as_default():
      init = constant_op.constant(100.0, dtype=dtype)
      var = variables.Variable(init)
      dummy_const = constant_op.constant(0.0)
      gradient = gradients.gradients(
          dummy_const,
          var,
          unconnected_gradients=unconnected_gradients.UnconnectedGradients.ZERO
      )[0]
      self.assertEqual(gradient.dtype, dtype)
      self.assertIsNotNone(gradient)

  def testVariableAsGraphElementGradient(self):
    # [FUNCTION DOC]
    # Converts a variable to a graph element and takes its gradient with respect to the variable;
    # asserts a result is returned.
    with ops.Graph().as_default() as graph:
      init = constant_op.constant(100.0)
      var = variables.Variable(init)
      gradient = gradients.gradients(graph.as_graph_element(var), var)
      self.assertIsNotNone(gradient)

  @test_util.run_v1_only("b/120545219")
  def testVariableRefGradient(self):
    # [FUNCTION DOC]
    # Graph-mode only. Takes the gradient of a reference variable's ref tensor with respect to the
    # variable and asserts a result is returned.
    with ops.Graph().as_default():
      init = constant_op.constant(100.0)
      var = variable_v1.VariableV1(init)
      gradient = gradients.gradients(var._ref(), var)
      self.assertIsNotNone(gradient)

  @test_util.run_v1_only("b/120545219")
  def testDependentYs(self):
    # [FUNCTION DOC]
    # Graph-mode only. Uses ys where one depends on another (x squared, then repeatedly squared)
    # and checks the gradient with respect to x is 17502 whether ys is a list, a sum, or identity
    # copies of the values.
    with self.cached_session():
      x = constant_op.constant(3.0)
      y = math_ops.square(x)
      y1 = math_ops.square(y)
      y2 = math_ops.square(y1)
      g = gradients.gradients([y, y2], x)
      self.assertAllClose(17502.0, g[0])
      g = gradients.gradients(y + y2, x)
      self.assertAllClose(17502.0, g[0])
      z = array_ops.identity(y)
      z2 = array_ops.identity(y2)
      g = gradients.gradients([z, z2], x)
      self.assertAllClose(17502.0, g[0])

  @test_util.run_v1_only("b/120545219")
  def testPartialDerivatives(self):
    # [FUNCTION DOC]
    # Graph-mode only. For z equal to x plus two x, asserts total gradients with respect to x and
    # y are three and one, and that passing both as stop-gradients yields partial derivatives of
    # one and one.
    with self.cached_session():
      x = constant_op.constant(1.)
      y = 2 * x
      z = x + y
      totalg = gradients.gradients(z, [x, y])
      self.assertEqual([3.0, 1.0], [g.eval() for g in totalg])
      partialg = gradients.gradients(z, [x, y], stop_gradients=[x, y])
      self.assertEqual([1.0, 1.0], [g.eval() for g in partialg])

  @test_util.run_v1_only("b/120545219")
  def testStopGradients(self):
    # [FUNCTION DOC]
    # Graph-mode only, randomized. Builds a four-node chain of random matrix expressions (a to d)
    # twice per subset of nodes: once with explicit stop-gradient ops on the subset and once with
    # the stop-gradients argument. For every subset of variables it computes both gradients
    # (null replaced by zeros), evaluates everything in one session run, and asserts each pair
    # of results is numerically close.
    def _MakeGraph(rng, stop_gradients=()):
      # [FUNCTION DOC]
      # HELPER: builds the four dependent random-matrix tensors a, b, c, d for this test, wrapping
      # those whose names appear in stop_gradients in a stop-gradient op. rng is the random state;
      # returns a mapping from names to tensors.
      def _FunctionOf(xs, k=3):
        # [FUNCTION DOC]
        # HELPER: returns a tensor that is the sum of random k-by-k matrices multiplied by each
        # input in xs, plus a random k-by-k matrix. Uses the enclosing rng.
        return ops.convert_to_tensor(
            sum(math_ops.matmul(rng.rand(k, k), x) for x in xs)
            + rng.rand(k, k))

      a = _FunctionOf([])
      if "a" in stop_gradients: a = array_ops.stop_gradient(a)
      b = _FunctionOf([a])
      if "b" in stop_gradients: b = array_ops.stop_gradient(b)
      c = _FunctionOf([a, b])
      if "c" in stop_gradients: c = array_ops.stop_gradient(c)
      d = _FunctionOf([b, c])
      if "d" in stop_gradients: d = array_ops.stop_gradient(d)
      return dict(a=a, b=b, c=c, d=d)

    def _Gradients(ys, xs, **kwargs):
      # [FUNCTION DOC]
      # HELPER: wrapper around the gradients function that forwards extra keyword arguments and
      # replaces any null gradient with a zero tensor shaped like its x. Returns the list.
      dydxs = gradients.gradients(ys, xs, **kwargs)
      dydxs = [0. * x if dydx is None else dydx
               for x, dydx in zip(xs, dydxs)]
      return dydxs

    seed = np.random.randint(1000)
    cases = []
    subsets = [""] + "a b c d ab ac ad bc bd cd abc abd acd bcd abcd".split()
    graph = _MakeGraph(np.random.RandomState(seed))
    for constants in subsets:
      graph_with_stops = _MakeGraph(np.random.RandomState(seed), constants)
      for variables_ in subsets:
        # compute the gradient when stopped using tf.stop_gradients
        grad1 = _Gradients([graph_with_stops["d"]],
                           [graph_with_stops[v] for v in variables_])
        # compute the gradient when stopped using the stop_gradients kwarg
        grad2 = _Gradients([graph["d"]],
                           [graph[v] for v in variables_],
                           stop_gradients=[graph[v] for v in constants])
        cases.append(dict(grad1=grad1, grad2=grad2,
                          constants=constants, variables=variables_))

    # evaluate all tensors in one call to session.run for speed
    with self.cached_session() as sess:
      results = sess.run([(case["grad1"], case["grad2"]) for case in cases])

    for (npgrad1, npgrad2), case in zip(results, cases):
      for a, b in zip(npgrad1, npgrad2):
        np.testing.assert_allclose(a, b)

  def testUnconnectedGradientsNoneUnconnectedGradients(self):
    # [FUNCTION DOC]
    # Differentiates an unrelated constant with respect to another with the none option for
    # unconnected gradients; asserts the result is null.
    with ops.Graph().as_default():
      x = constant(1.0, shape=[2, 2])
      y = constant(3.0, shape=[3, 1])
      grad = gradients.gradients(
          [y], [x], unconnected_gradients="none")
    self.assertIsNone(grad[0])

  def testUnconnectedGradientsZerosUnconnectedGradients(self):
    # [FUNCTION DOC]
    # Same unconnected setup with the zero option; asserts the evaluated gradient is a two-by-two
    # matrix of zeros.
    with ops.Graph().as_default():
      x = constant(1.0, shape=[2, 2])
      y = constant(3.0, shape=[3, 1])
      grads = gradients.gradients(
          [y], [x], unconnected_gradients="zero")

      self.assertAllEqual([[0.0, 0.0], [0.0, 0.0]], self.evaluate(grads)[0])

  def testUnconnectedGradientsZeroConnectedGradients(self):
    # [FUNCTION DOC]
    # With a connected pair (y is three times x) and the zero option, asserts the real gradient of
    # three is returned rather than zeros.
    with ops.Graph().as_default():
      x = constant(1.0)
      y = x * 3.0
      grad = gradients.gradients(
          [y], [x], unconnected_gradients="zero")

      self.assertEqual(3.0, self.evaluate(grad)[0])

  def testUnknownUnconnectedGradientsValueGiven(self):
    # [FUNCTION DOC]
    # Passes an invalid string for the unconnected-gradients option and asserts a value error whose
    # message says the value is unknown.
    with ops.Graph().as_default():
      x = constant(1.0)
      y = constant(1.0)
      with self.assertRaisesRegex(
          ValueError, "Unknown value for unconnected_gradients: 'nonsense'"):
        gradients.gradients([y], [x], unconnected_gradients="nonsense")

  @parameterized.parameters(unconnected_gradients.UnconnectedGradients.ZERO,
                            unconnected_gradients.UnconnectedGradients.NONE)
  def testUnconnectedOpWithMultipleOutputs(self, unconnected_gradients_val):
    # [FUNCTION DOC]
    # Parameterized over the zero and none options. An identity-n op has two outputs but only the
    # first is used downstream; differentiating with respect to the unused second output must give
    # a non-null (zero) gradient for zero and a null result for none.
    with ops.Graph().as_default():
      #  a    b
      #  |    |
      # IdentityN
      #  |    |
      #  c    d
      #  |
      # Identity
      #  |
      #  e
      a = constant_op.constant(1.0)
      b = constant_op.constant(1.0)
      c, d = array_ops.identity_n([a, b])
      e = array_ops.identity(c)
      # The aggregated grads for the IdentityN node would look like
      # [Tensor, None]. We expect this None to be converted to zeros.
      output = gradients.gradients(
          e, d, unconnected_gradients=unconnected_gradients_val)
      if (unconnected_gradients_val ==
          unconnected_gradients.UnconnectedGradients.ZERO):
        self.assertIsNotNone(output[0])
      else:
        self.assertIsNone(output[0])

  def testOptimizeIdentityN(self):
    # [FUNCTION DOC]
    # Compares the normal gradients function with the internal helper using the identity-n
    # optimization. For cross pairs of identity-n outputs and inputs the optimized helper returns
    # null while the normal function returns a gradient; for matching pairs both return a gradient.
    with ops.Graph().as_default():
      #  a    b
      #  |    |
      # IdentityN
      #  |    |
      #  c    d
      a = constant_op.constant(1.0)
      b = constant_op.constant(1.0)
      c, d = array_ops.identity_n([a, b])

      def OptimizedGradients(y, x):
        # [FUNCTION DOC]
        # HELPER: calls the internal gradients helper with the identity-n optimization turned on
        # for ys y and xs x, and returns its result.
        return gradients_util._GradientsHelper(
            ys=y,
            xs=x,
            optimize_identity_n=True,
        )

      self.assertIsNone(OptimizedGradients(d, a)[0])
      self.assertIsNone(OptimizedGradients(c, b)[0])

      self.assertIsNotNone(gradients.gradients(d, a)[0])
      self.assertIsNotNone(gradients.gradients(c, b)[0])

      for grad_impl in [gradients.gradients, OptimizedGradients]:
        self.assertIsNotNone(grad_impl(c, a)[0])
        self.assertIsNotNone(grad_impl(d, b)[0])

  @parameterized.parameters(unconnected_gradients.UnconnectedGradients.ZERO,
                            unconnected_gradients.UnconnectedGradients.NONE)
  def testUnconnectedOpWithMultipleOutputsStopGradient(
      self, unconnected_gradients_val):
    # [FUNCTION DOC]
    # Parameterized over zero and none. One output of an identity-n op passes through a stop
    # gradient before being summed with the other. Differentiating with respect to the stopped
    # output must yield a non-null result for zero and null for none.
    with ops.Graph().as_default():
      #  a    b
      #  |    |
      # IdentityN
      #  |    |
      #  c    d
      #  |    |
      #  SG   |
      #  |    |
      #   \  /
      #    +
      #    e
      a = constant_op.constant(1.0)
      b = constant_op.constant(1.0)
      c, d = array_ops.identity_n([a, b])
      e = array_ops.stop_gradient(c) + d
      # The aggregated grads for the IdentityN node would look like
      # [None, Tensor]. We expect this None to be converted to zeros.
      output = gradients.gradients(
          e, c, unconnected_gradients=unconnected_gradients_val)
      if (unconnected_gradients_val ==
          unconnected_gradients.UnconnectedGradients.ZERO):
        self.assertIsNotNone(output[0])
      else:
        self.assertIsNone(output[0])


class FunctionGradientsTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests gradients through legacy graph functions (Defun) and through function-decorated code:
  # basic and composed function gradients, custom gradient functions, registered python gradient
  # functions, capturing outer tensors and variables, and nesting. Provides class-level helper
  # functions that define a square-plus-bias function and its perturbed gradient implementations.

  @classmethod
  def XSquarePlusB(cls, x, b):
    # [FUNCTION DOC]
    # HELPER: forward function wrapped by the function-definition decorator in these tests.
    # Returns x squared plus b, elementwise.
    return x * x + b

  @classmethod
  def XSquarePlusBGradient(cls, x, b, g):
    # [FUNCTION DOC]
    # HELPER: explicit gradient function for the square-plus-bias function. Doubles the incoming
    # gradient g (to prove it was used) and returns a pair: g times two times x for the first
    # input, and g for the second input.
    # Perturb gradients (multiply by 2), so we can test that this was called.
    g *= 2.0
    return g * 2.0 * x, g

  @classmethod
  def _PythonGradient(cls, op, grad):
    # [FUNCTION DOC]
    # HELPER: python gradient function registered for the square-plus-bias function. Triples the
    # incoming gradient (to prove it was used) and returns a pair: the tripled gradient times the
    # op's first input times two, and the tripled gradient.
    # Perturb gradients (multiply by 3), so we can test that this was called.
    grad *= 3.0
    return grad * op.inputs[0] * 2.0, grad

  @classmethod
  def _GetFunc(cls, **kwargs):
    # [FUNCTION DOC]
    # HELPER: wraps the square-plus-bias function as a two-float32-argument graph function,
    # forwarding keyword options such as a gradient function or python gradient function. Returns
    # the wrapped function.
    return framework_function.Defun(dtypes.float32, dtypes.float32, **
                                    kwargs)(cls.XSquarePlusB)

  def _GetFuncGradients(self, f, x_value, b_value):
    # [FUNCTION DOC]
    # HELPER: creates constants from x_value and b_value, calls function f on them, takes the
    # gradients of the result with respect to both, evaluates them and returns the evaluated list.
    x = constant_op.constant(x_value, name="x")
    b = constant_op.constant(b_value, name="b")

    y = f(x, b)
    grads = gradients.gradients(y, [x, b])

    return self.evaluate(grads)

  def testFunctionGradientsBasic(self):
    # [FUNCTION DOC]
    # Gradients of the plain function at x=2, b=1: asserts they are 4 and 1.
    g = ops.Graph()
    with g.as_default():
      f = self._GetFunc()
      # Get gradients (should add SymbolicGradient node for function).
      grads = self._GetFuncGradients(f, [2.0], [1.0])
      self.assertAllEqual([4.0], grads[0])
      self.assertAllEqual([1.0], grads[1])

  def testFunctionGradientsComposition(self):
    # [FUNCTION DOC]
    # Composes the function with itself and asserts the gradients with respect to the innermost
    # input and first bias evaluate to 40 and 10.
    with ops.Graph().as_default():
      f = self._GetFunc()
      x = constant_op.constant([2.0], name="x")
      b1 = constant_op.constant([1.0], name="b1")
      b2 = constant_op.constant([1.0], name="b2")

      y = f(f(x, b1), b2)
      # Build gradient graph (should add SymbolicGradient node for function).
      grads = gradients.gradients(y, [x, b1])

      self.assertAllEqual([40.0], self.evaluate(grads)[0])
      self.assertAllEqual([10.0], self.evaluate(grads)[1])

  def testFunctionGradientsWithGradFunc(self):
    # [FUNCTION DOC]
    # Attaches an explicit gradient function that doubles gradients; asserts the gradients are
    # twice the plain values (8 and 2).
    g = ops.Graph()
    with g.as_default():
      grad_func = framework_function.Defun(dtypes.float32, dtypes.float32,
                                           dtypes.float32)(
                                               self.XSquarePlusBGradient)
      f = self._GetFunc(grad_func=grad_func)
      # Get gradients (should add SymbolicGradient node for function, which
      # uses the grad_func above, which multiplies all gradients by 2).
      grads = self._GetFuncGradients(f, [2.0], [1.0])
      self.assertAllEqual([4.0 * 2], grads[0])
      self.assertAllEqual([1.0 * 2], grads[1])

  def testFunctionGradientWithRegistration(self):
    # [FUNCTION DOC]
    # Attaches a python gradient function that triples gradients; asserts gradients are 12 and 3.
    g = ops.Graph()
    with g.as_default():
      f = self._GetFunc(python_grad_func=self._PythonGradient)
      # Get gradients, using the python gradient function. It multiplies the
      # gradients by 3.
      grads = self._GetFuncGradients(f, [2.0], [1.0])
      self.assertAllEqual([4.0 * 3], grads[0])
      self.assertAllEqual([1.0 * 3], grads[1])

  def testFunctionGradientWithGradFuncAndRegistration(self):
    # [FUNCTION DOC]
    # Supplies both an explicit gradient function and a python gradient function and asserts a
    # value error saying the gradient was defined twice.
    g = ops.Graph()
    with g.as_default():
      grad_func = framework_function.Defun(dtypes.float32, dtypes.float32,
                                           dtypes.float32)(
                                               self.XSquarePlusBGradient)
      with self.assertRaisesRegex(ValueError, "Gradient defined twice"):
        f = self._GetFunc(
            grad_func=grad_func, python_grad_func=self._PythonGradient)
        f.add_to_graph(ops.Graph())

  def testGradientWrtCaptured(self):
    # [FUNCTION DOC]
    # Inside a decorated function, differentiates a value with respect to an outer graph tensor
    # that the function captured; asserts the result evaluates to 2.
    with ops.Graph().as_default():
      x = constant_op.constant(1.0, name="x")

      @def_function.function
      def Foo():
        # [FUNCTION DOC]
        # HELPER: function body for this test: multiplies the captured tensor by two and returns
        # the gradient of that product with respect to the captured tensor.
        y = math_ops.multiply(x, 2.0, name="y")
        g = gradients_impl.gradients(y, x)
        return g[0]

      f = Foo()

      self.assertEqual(self.evaluate(f), 2.0)

  def testGradientOfCaptured(self):
    # [FUNCTION DOC]
    # Builds the forward product in the outer graph and takes its gradient inside a legacy graph
    # function that captures both tensors; asserts the result is 2.
    with ops.Graph().as_default():
      x = constant_op.constant(1.0, name="x")
      y = math_ops.multiply(x, 2.0, name="y")

      @framework_function.Defun()
      def Foo():
        # [FUNCTION DOC]
        # HELPER: function body for this test: returns the gradient of the captured product with
        # respect to the captured input.
        g = gradients_impl.gradients(y, x)
        return g[0]

      f = Foo()

      self.assertEqual(self.evaluate(f), 2.0)

  def testCapturedResourceVariable(self):
    # [FUNCTION DOC]
    # A decorated function multiplies a captured resource variable by two and differentiates with
    # respect to the variable. After initializing variables, asserts the gradient is 2.
    with ops.Graph().as_default():
      var = resource_variable_ops.ResourceVariable(1.0, name="var")

      @def_function.function
      def Foo():
        # [FUNCTION DOC]
        # HELPER: function body for this test: returns the gradient of two times the captured
        # variable with respect to the variable.
        y = math_ops.multiply(var, 2.0, name="y")
        g = gradients_impl.gradients(y, var)
        return g[0]

      f = Foo()

      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(self.evaluate(f), 2.0)

  def testCapturedNested(self):
    # [FUNCTION DOC]
    # Two nested decorated functions capture outer tensors; the inner one differentiates several
    # identity copies (some captured from the outer function, some from the graph) plus x1 with
    # respect to x1 and x2. Asserts the two gradients are 4 and 2 as summed in the comments.
    with ops.Graph().as_default():
      x1 = constant_op.constant(1.0, name="x1")
      x2 = constant_op.constant(2.0, name="x2")
      x3 = math_ops.multiply(x1, x2, name="x3")

      @def_function.function
      def Outer():
        # [FUNCTION DOC]
        # HELPER: outer function body: copies x1 and returns the result of calling the inner
        # function, which computes the gradients.
        outer1 = array_ops.identity(x1, name="outer1")

        @def_function.function
        def Inner():
          # [FUNCTION DOC]
          # HELPER: inner function body: copies the outer copy, x2 and x3, and returns gradients
          # of those copies together with x1 with respect to x1 and x2.
          inner1 = array_ops.identity(outer1, name="inner1")
          inner2 = array_ops.identity(x2, name="inner2")
          inner3 = array_ops.identity(x3, name="inner3")
          return gradients_impl.gradients([inner1, inner2, inner3, x1],
                                          [x1, x2])

        return Inner()

      x1_grad, x2_grad = Outer()

      # 1.0 + None + 2.0 + 1.0 = 4.0
      self.assertEqual(self.evaluate(x1_grad), 4.0)
      # None + 1.0 + 1.0 + None = 2.0
      self.assertEqual(self.evaluate(x2_grad), 2.0)

  def testCapturedFromFunction(self):
    # [FUNCTION DOC]
    # An inner decorated function differentiates a value with respect to a tensor created in the
    # outer decorated function (so captured from it). Asserts the gradient is 3.
    with ops.Graph().as_default():
      x = constant_op.constant(1.0, name="x")

      @def_function.function
      def Outer():
        # [FUNCTION DOC]
        # HELPER: outer function body: computes two times x and returns the inner function's
        # result.
        y = math_ops.multiply(x, 2.0, name="y")

        @def_function.function
        def Inner():
          # [FUNCTION DOC]
          # HELPER: inner function body: multiplies the captured y by three and returns the
          # gradient of that product with respect to y.
          z = math_ops.multiply(y, 3.0, name="z")
          g = gradients_impl.gradients(z, y)
          return g[0]

        return Inner()

      z_grad = Outer()

      self.assertEqual(self.evaluate(z_grad), 3.0)

  def testCapturedEagerTensors(self):
    # [FUNCTION DOC]
    # In eager mode, a decorated function captures an eager tensor that is unrelated to the
    # differentiated path; the gradient code must ignore it (regression for a known bug).
    # Asserts the gradient of the product with respect to x is 4.
    # Test that we can handle captured eager tensors unrelated to the gradient
    # computation (i.e. we need to ignore them).
    # TODO(skyewm): make it an error if you try to take the gradient wrt a
    # captured EagerTensor
    with context.eager_mode():
      c = constant_op.constant(2.0, name="c")

      @def_function.function
      def Foo():
        # [FUNCTION DOC]
        # HELPER: function body for this test: builds products involving the captured eager
        # tensor and returns the gradient of the final product with respect to x.
        x = constant_op.constant(10.0, name="x")
        y = math_ops.multiply(x, c, name="y")
        # Regression test for b/122564611.
        z = math_ops.multiply(c, y, name="z")
        g = gradients_impl.gradients(z, x)
        return g[0]

      self.assertEqual(Foo().numpy(), 4.0)


class StopGradientTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests that the stop-gradient op blocks gradient flow.

  def testStopGradient(self):
    # [FUNCTION DOC]
    # Passes a constant through stop-gradient and asserts the gradient of the output with respect
    # to the input is null.
    with ops.Graph().as_default():
      inp = constant(1.0, shape=[100, 32], name="in")
      out = array_ops.stop_gradient(inp)
      igrad = gradients.gradients(out, inp)[0]
    assert igrad is None


class PreventGradientTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests the prevent-gradient op, which makes any attempt to backpropagate through it an error.

  def testPreventGradient(self):
    # [FUNCTION DOC]
    # Applies prevent-gradient to a constant and asserts that requesting its gradient raises a
    # lookup error mentioning that the gradient was explicitly disabled.
    with ops.Graph().as_default():
      inp = constant(1.0, shape=[100, 32], name="in")
      out = array_ops.prevent_gradient(inp)
      with self.assertRaisesRegex(LookupError, "explicitly disabled"):
        _ = gradients.gradients(out, inp)


class HessianVectorProductTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests the private Hessian-vector product helper against an explicitly computed Hessian.

  @test_util.run_v1_only("b/120545219")
  def testHessianVectorProduct(self):
    # [FUNCTION DOC]
    # Graph-mode only. For the quadratic form x transpose A x with random float32 data, the Hessian
    # is A plus A transpose. On CPU and GPU sessions, computes the helper's product with a random
    # vector and asserts it is close to the explicit matrix-vector product.
    # Manually compute the Hessian explicitly for a low-dimensional problem
    # and check that HessianVectorProduct matches multiplication by the
    # explicit Hessian.
    # Specifically, the Hessian of f(x) = x^T A x is
    # H = A + A^T.
    # We expect HessianVectorProduct(f(x), x, v) to be H v.
    m = 4
    rng = np.random.RandomState([1, 2, 3])
    mat_value = rng.randn(m, m).astype("float32")
    v_value = rng.randn(m, 1).astype("float32")
    x_value = rng.randn(m, 1).astype("float32")
    hess_value = mat_value + mat_value.T
    hess_v_value = np.dot(hess_value, v_value)
    for use_gpu in [False, True]:
      with self.cached_session(use_gpu=use_gpu):
        mat = constant_op.constant(mat_value)
        v = constant_op.constant(v_value)
        x = constant_op.constant(x_value)
        mat_x = math_ops.matmul(mat, x, name="Ax")
        x_mat_x = math_ops.matmul(array_ops.transpose(x), mat_x, name="xAx")
        hess_v = gradients_impl._hessian_vector_product(x_mat_x, [x], [v])[0]
        hess_v_actual = self.evaluate(hess_v)
      self.assertAllClose(hess_v_value, hess_v_actual)


class HessianTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests the public hessians function on one-dimensional, multi-tensor and matrix inputs, plus
  # invalid input shapes. All tests are graph-mode only.

  @test_util.run_v1_only("b/120545219")
  def testHessian1D(self):
    # [FUNCTION DOC]
    # Hessian of the quadratic form of a random vector and matrix; asserts it matches A plus
    # A transpose.
    # Manually compute the Hessian explicitly for a low-dimensional problem
    # and check that `hessian` matches. Specifically, the Hessian of
    # f(x) = x^T A x is H = A + A^T.
    m = 4
    rng = np.random.RandomState([1, 2, 3])
    mat_value = rng.randn(m, m).astype("float32")
    x_value = rng.randn(m).astype("float32")
    hess_value = mat_value + mat_value.T
    with self.session():
      mat = constant_op.constant(mat_value)
      x = constant_op.constant(x_value)
      x_mat_x = math_ops.reduce_sum(x[:, None] * mat * x[None, :])
      hess = gradients.hessians(x_mat_x, x)[0]
      hess_actual = self.evaluate(hess)
    self.assertAllClose(hess_value, hess_actual)

  @test_util.run_v1_only("b/120545219")
  def testHessian1D_multi(self):
    # [FUNCTION DOC]
    # Computes Hessians of a list of three quadratic forms with respect to a list of three
    # vectors in one call and asserts each equals its matrix plus transpose.
    # Test the computation of the hessian with respect to multiple tensors
    m = 4
    n = 3
    rng = np.random.RandomState([1, 2, 3])
    mat_values = [rng.randn(m, m).astype("float32") for _ in range(n)]
    x_values = [rng.randn(m).astype("float32") for _ in range(n)]
    hess_values = [mat_value + mat_value.T for mat_value in mat_values]
    with self.session():
      mats = [constant_op.constant(mat_value) for mat_value in mat_values]
      xs = [constant_op.constant(x_value) for x_value in x_values]
      xs_mats_xs = [
          math_ops.reduce_sum(x[:, None] * mat * x[None, :])
          for x, mat in zip(xs, mats)
      ]
      hessians = gradients.hessians(xs_mats_xs, xs)
      hessians_actual = [hess.eval() for hess in hessians]
    for hess_value, hess_actual in zip(hess_values, hessians_actual):
      self.assertAllClose(hess_value, hess_actual)

  @test_util.run_v1_only("b/120545219")
  def testHessianInvalidDimension(self):
    # [FUNCTION DOC]
    # For a placeholder of a two-dimensional shape and one of unknown shape, asserts that asking
    # for the Hessian of the placeholder with respect to itself raises a value error.
    for shape in [(10, 10), None]:
      with self.cached_session():
        x = array_ops.placeholder(dtypes.float32, shape)
        # Expect a ValueError because the dimensions are wrong
        with self.assertRaises(ValueError):
          gradients.hessians(x, x)

  @test_util.run_v1_only("b/120545219")
  def testHessian2D_square_matrix(self):
    # [FUNCTION DOC]
    # Hessian of half the sum of x transpose x for a square random matrix. Asserts the result has
    # shape m by m by m by m and, reshaped to a square matrix, equals the block identity pattern.
    # Manually compute the Hessian explicitly for a low-dimensional problem
    # and check that `hessian` matches. Specifically, the Hessian of
    # f(x) = 1/2 * x^T * x is H = constant (block identity matrix)
    m = 3
    rng = np.random.RandomState([1, 2, 3])
    x_value = rng.randn(m, m).astype("float32")
    with self.session():
      x = constant_op.constant(x_value)
      x_square = math_ops.reduce_sum(
          math_ops.matmul(array_ops.transpose(x), x) * 0.5
      )
      hess = gradients.hessians(x_square, x)[0]
      hess_actual = self.evaluate(hess)
    hess_value = np.bmat([
        [elem*np.ones((m, m)) for elem in vec]
        for vec in np.eye(m)
    ]).astype("float32")
    self.assertAllEqual((m, m, m, m), hess_actual.shape)
    self.assertAllClose(hess_value, hess_actual.reshape((m * m, m * m)))

  @test_util.run_v1_only("b/120545219")
  def testHessian2D_non_square_matrix(self):
    # [FUNCTION DOC]
    # Same as the square case with a three-by-four matrix; asserts shape (3, 4, 3, 4) and values
    # equal to a block matrix of ones scaled by the identity.
    m = 3
    n = 4
    rng = np.random.RandomState([1, 2, 3])
    x_value = rng.randn(m, n).astype("float32")
    with self.session():
      x = constant_op.constant(x_value)
      x_square = math_ops.reduce_sum(
          math_ops.matmul(array_ops.transpose(x), x) * 0.5
      )
      hess = gradients.hessians(x_square, x)[0]
      hess_actual = self.evaluate(hess)
    hess_value = np.bmat([
        [elem*np.ones((n, n)) for elem in vec]
        for vec in np.eye(m)
    ]).astype("float32")
    self.assertAllEqual((m, n, m, n), hess_actual.shape)
    self.assertAllClose(hess_value, hess_actual.reshape((m * n, m * n)))


class IndexedSlicesToTensorTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests implicit conversion of indexed-slices values to dense tensors (in multiplication and
  # stacking), int64 indices, and the memory warnings emitted for large or unknown dense shapes.
  # All tests are graph-mode only.

  @test_util.run_v1_only("b/120545219")
  def testIndexedSlicesToTensor(self):
    # [FUNCTION DOC]
    # Converts a random dense tensor to indexed slices, multiplies it by one (forcing conversion
    # back to dense) and asserts the dense shape and the values match the original.
    with self.cached_session():
      np_val = np.random.rand(4, 4, 4, 4).astype(np.float32)
      c = constant_op.constant(np_val)
      c_sparse = math_ops._as_indexed_slices(c)
      self.assertAllEqual(np_val.shape, c_sparse.dense_shape)
      c_dense = math_ops.multiply(c_sparse, 1.0)
      self.assertAllClose(np_val, self.evaluate(c_dense))

  @test_util.run_v1_only("b/120545219")
  def testIndexedSlicesToTensorList(self):
    # [FUNCTION DOC]
    # Stacks a list of three dense tensors and a list of their indexed-slices equivalents, and
    # asserts the two stacked results are numerically equal.
    with self.cached_session():
      numpy_list = []
      dense_list = []
      sparse_list = []
      for _ in range(3):
        np_val = np.random.rand(4, 4, 4, 4).astype(np.float32)
        c = constant_op.constant(np_val)
        c_sparse = math_ops._as_indexed_slices(c)
        numpy_list.append(np_val)
        dense_list.append(c)
        sparse_list.append(c_sparse)
      packed_dense = array_ops_stack.stack(dense_list)
      packed_sparse = array_ops_stack.stack(sparse_list)
      self.assertAllClose(packed_dense, self.evaluate(packed_sparse))

  @test_util.run_v1_only("b/120545219")
  def testInt64Indices(self):
    # [FUNCTION DOC]
    # Same as the basic conversion test but with indices cast to 64-bit integers; asserts shape
    # and values are preserved after conversion to dense.
    with self.cached_session():
      np_val = np.random.rand(4, 4, 4, 4).astype(np.float32)
      c = constant_op.constant(np_val)
      c_sparse = math_ops._as_indexed_slices(c)
      c_sparse = indexed_slices.IndexedSlices(
          c_sparse.values,
          math_ops.cast(c_sparse.indices, dtypes.int64), c_sparse.dense_shape)
      self.assertAllEqual(np_val.shape, c_sparse.dense_shape)
      c_dense = math_ops.multiply(c_sparse, 1.0)
      self.assertAllClose(np_val, self.evaluate(c_dense))

  @test_util.run_v1_only("b/120545219")
  def testWarnings(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test skips itself whenever the Python version is 3.5 or newer (a TODO ties
    # this to a protobuf issue), so on every modern interpreter it never runs and its assertions are dead.
    # Intended scenario: converting indexed slices to dense with a small dense shape gives no
    # warning, while a large shape (100 million elements) or unknown shape emits exactly one
    # warning about large memory use, whose message text is checked.
    # TODO(gunan) Reenable after this issue is fixed:
    # https://github.com/google/protobuf/issues/2812
    if sys.version_info >= (3, 5):
      self.skipTest("Skipped test for Python 3.5+")

    # Smaller than the threshold: no warning.
    c_sparse = indexed_slices.IndexedSlices(
        array_ops.placeholder(dtypes.float32),
        array_ops.placeholder(dtypes.int32), constant([4, 4, 4, 4]))
    with warnings.catch_warnings(record=True) as w:
      math_ops.multiply(c_sparse, 1.0)
    self.assertEqual(0, len(w))

    # Greater than or equal to the threshold: warning.
    c_sparse = indexed_slices.IndexedSlices(
        array_ops.placeholder(dtypes.float32),
        array_ops.placeholder(dtypes.int32), constant([100, 100, 100, 100]))
    # "always" filter prevents the warning from being suppressed if it was
    # already triggered in a different test.
    warnings.simplefilter("always")
    with warnings.catch_warnings(record=True) as w:
      math_ops.multiply(c_sparse, 1.0)
    self.assertEqual(1, len(w))
    self.assertIn(
        "with 100000000 elements. This may consume a large amount of memory.",
        str(w[0].message))

    # Unknown dense shape: warning.
    c_sparse = indexed_slices.IndexedSlices(
        array_ops.placeholder(dtypes.float32),
        array_ops.placeholder(dtypes.int32),
        array_ops.placeholder(dtypes.int32))
    with warnings.catch_warnings(record=True) as w:
      math_ops.multiply(c_sparse, 1.0)
    self.assertEqual(1, len(w))
    self.assertIn(
        "of unknown shape. This may consume a large amount of memory.",
        str(w[0].message))


class OnlyRealGradientsTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests that gradients of complex tensors need explicit initial gradients.

  @test_util.run_v1_only("b/120545219")
  def testRealOnly(self):
    # [FUNCTION DOC]
    # Graph-mode only. Squares a complex64 constant and requests its gradient without grad_ys;
    # asserts a type error saying complex gradients require grad_ys of the matching dtype.
    x = constant_op.constant(7+3j, dtype=dtypes.complex64)
    y = math_ops.square(x)
    with self.assertRaisesRegex(
        TypeError, r"Gradients of complex tensors .* must set grad_ys "
        r"\(y\.dtype = complex64\)"):
      gradients.gradients(y, x)


class ResourceCondTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests gradients of resource variables used inside a conditional.

  @test_util.run_v1_only("b/120545219")
  def testBasic(self):
    # [FUNCTION DOC]
    # Graph-mode only. A conditional on a placeholder (default true) adds a resource variable to
    # ones in one branch and returns ones in the other. Asserts the gradient with respect to the
    # variable is not null.
    gamma = resource_variable_ops.ResourceVariable(
        np.random.random((3,)),
        dtype="float32", name="gamma")

    inputs = array_ops.ones(shape=(3,), dtype="float32")

    def TestFn():
      # [FUNCTION DOC]
      # HELPER: true-branch function: returns the inputs plus the gamma variable.
      output = inputs + gamma
      return output

    training = array_ops.placeholder_with_default(True, shape=())
    output = cond.cond(
        training, TestFn, lambda: inputs)

    loss = output

    grads = gradients.gradients(
        loss, [gamma])
    self.assertNotIn(None, grads)


class GetDependentVariablesTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests the private custom-gradient utility that finds trainable variables a computation depends
  # on between given inputs and outputs, plus lookup of a variable by name.

  def testNoVariables(self):
    # [FUNCTION DOC]
    # A computation with no variables: asserts the dependent-variable list is empty.
    with ops.Graph().as_default():
      func = lambda x: array_ops.identity(x) + 5.0
      input_t = constant_op.constant(2.0)
      result_t = func(input_t)
      dependent_vars = custom_gradient._get_dependent_variables(
          [input_t], [result_t])

      # There are no variables.
      self.assertEqual(dependent_vars, [])

  def testVariablesOutside(self):
    # [FUNCTION DOC]
    # The computation closes over a variable defined outside it; asserts exactly that variable is
    # found.
    with ops.Graph().as_default():
      init = constant_op.constant(100.0)
      var = variables.Variable(init)

      # The variable is closed over. It should be found.
      func = lambda x: array_ops.identity(x) + 5.0 + var

      input_t = constant_op.constant(2.0)
      result_t = func(input_t)
      dependent_vars = custom_gradient._get_dependent_variables(
          [input_t], [result_t])
      self.assertEqual(dependent_vars, [var])

  def testVariableSamePrefix(self):
    # [FUNCTION DOC]
    # Two variables whose names share a prefix are both used; asserts both are found (no
    # prefix-matching confusion).
    with ops.Graph().as_default():
      var_name = "my_variable"
      v_z = variable_scope.get_variable(var_name, shape=())
      v_o = variable_scope.get_variable(var_name + "_ones", shape=())

      # The variable is closed over. It should be found.
      func = lambda x: array_ops.identity(x) + 5.0 + v_z + v_o

      input_t = constant_op.constant(2.0)
      result_t = func(input_t)
      dependent_vars = custom_gradient._get_dependent_variables(
          [input_t], [result_t])
      self.assertEqual(set(dependent_vars), set([v_o, v_z]))

  def testVariablesOutsideButDSeparated(self):
    # [FUNCTION DOC]
    # The variable influences the function input but lies before it; since the input cuts the
    # dependency, asserts no variables are reported.
    with ops.Graph().as_default():
      init = constant_op.constant(100.0)
      var = variables.Variable(init)

      # The variable is d-separated by the inputs. It should not be found.
      input_t = array_ops.identity(var) * 5.0

      func = lambda x: array_ops.identity(x) + 5.0
      result_t = func(input_t)
      dependent_vars = custom_gradient._get_dependent_variables(
          [input_t], [result_t])
      self.assertEqual(dependent_vars, [])

  def testVariablesOutsideAndNonDifferentiable(self):
    # [FUNCTION DOC]
    # The function depends on a variable only through a non-differentiable ones-like op; asserts
    # the variable is not reported.
    with ops.Graph().as_default():
      init = constant_op.constant(100.0, shape=(5,))
      var = variables.Variable(init, shape=(5,))

      def _Func(x):
        # [FUNCTION DOC]
        # HELPER: test computation: adds five and a ones tensor shaped like the variable to the
        # identity of x.
        # non-differentiable dependency on var.
        # the variable should not be found.
        y = array_ops.ones_like(var)
        return array_ops.identity(x) + 5.0 + y

      input_t = constant_op.constant(2.0)
      result_t = _Func(input_t)
      dependent_vars = custom_gradient._get_dependent_variables(
          [input_t], [result_t])
      self.assertEqual(dependent_vars, [])

  def testGetVariableByName(self):
    # [FUNCTION DOC]
    # In graph mode, makes a variable named with a replica suffix, rewires its internal tensor
    # (reference or handle) to an identity named "a", then asserts looking up a variable by the
    # name "a" returns the variable with the original name.
    with context.graph_mode():
      init = constant_op.constant(100.0)
      var = variable_v1.VariableV1(init, name="a/replica_1")
      if isinstance(var, ref_variable.RefVariable):
        var._variable = array_ops.identity(var, name="a")
      else:
        var._handle = array_ops.identity(var, name="a")
      var2 = custom_gradient.get_variable_by_name("a")
      self.assertEqual(var2.name, var.name)

  def testVariablesOutsideAndNonTrainable(self):
    # [FUNCTION DOC]
    # Uses one trainable and one non-trainable variable in the computation; asserts only the
    # trainable one is reported.
    with ops.Graph().as_default():
      init = constant_op.constant(100.0, shape=(5,))

      # Both variables are used in the function but only the trainable one
      # should be found.
      var_trainable = variables.Variable(init, shape=(5,))
      var_nontrainable = variables.Variable(init, shape=(5,), trainable=False)

      def _Func(x):
        # [FUNCTION DOC]
        # HELPER: test computation: ignores its argument and returns the sum of the trainable
        # and non-trainable variables.
        del x
        return var_trainable + var_nontrainable

      input_t = constant_op.constant(2.0)
      result_t = _Func(input_t)
      dependent_vars = custom_gradient._get_dependent_variables(
          [input_t], [result_t])
      self.assertEqual(dependent_vars, [var_trainable])

  def testVariablesOutsideAndCustomGradient(self):
    # [FUNCTION DOC]
    # The computation depends on a variable only through an op that is non-differentiable but has
    # a custom gradient defined; asserts the variable is reported.
    with ops.Graph().as_default():
      init = constant_op.constant(100.0, shape=(5,))
      var = variables.Variable(init, shape=(5,))

      @custom_gradient.custom_gradient
      def _MyOnesLike(x):
        # [FUNCTION DOC]
        # HELPER: ones-like function decorated with a custom gradient. Returns a ones tensor shaped
        # like x and a gradient function.
        """Dummy version of ones_like which defines a gradient."""

        output = array_ops.ones_like(x)

        def _Grad(dy):
          # [FUNCTION DOC]
          # HELPER: custom gradient for the dummy ones-like function; returns the upstream
          # gradient dy through an identity op.
          return array_ops.identity(dy)

        return output, _Grad

      def _Func(x):
        # [FUNCTION DOC]
        # HELPER: test computation: adds five and the dummy ones-like of the variable to the
        # identity of x.
        # non-differentiable operation with custom gradient.
        # The variable should be found.
        y = _MyOnesLike(var)
        return array_ops.identity(x) + 5.0 + y

      input_t = constant_op.constant(2.0)
      result_t = _Func(input_t)
      dependent_vars = custom_gradient._get_dependent_variables(
          [input_t], [result_t])
      self.assertEqual(dependent_vars, [var])


class CustomGradientTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests the custom-gradient decorator: simple and multi-input functions, methods on classes,
  # error propagation, variables (resource and non-resource), ragged tensors, quantized dtypes,
  # captured tensors, keyword-only variable arguments, numpy inputs, nested custom gradients and
  # structured inputs and outputs. Supports parameterized tests.

  def testCustomGradientTrivial(self):
    # [FUNCTION DOC]
    # Applies an identity function with a custom gradient (triple the upstream gradient) twice in
    # sequence. Asserts the total gradient evaluates to nine.

    @custom_gradient.custom_gradient
    def MyIdentity(x):
      # [FUNCTION DOC]
      # HELPER: custom-gradient identity; returns x and the gradient function below.

      def Grad(dy):
        # [FUNCTION DOC]
        # HELPER: gradient of the identity: returns a one-item list with three times dy.
        return [3 * dy]

      return x, Grad

    with ops.Graph().as_default():
      x = constant(3.)
      y = MyIdentity(MyIdentity(x))
      dy = gradients.gradients(y, x)[0]
      with session.Session():
        self.assertEqual(9., self.evaluate(dy))

  def testCustomGradient(self):
    # [FUNCTION DOC]
    # A two-input multiply function whose custom gradient returns the swapped partials (the
    # ordering is deliberately switched). With inputs 3 and 5, asserts the gradients evaluate to
    # 3 and 5.

    @custom_gradient.custom_gradient
    def MyMultiply(x1, x2):
      # [FUNCTION DOC]
      # HELPER: custom-gradient multiply; returns the product and the gradient function.
      result = x1 * x2

      def Grad(dy):
        # [FUNCTION DOC]
        # HELPER: returns the list of dy times x1 and dy times x2 (swapped relative to calculus).
        # Switched the ordering here.
        return [dy * x1, dy * x2]

      return result, Grad

    with ops.Graph().as_default():
      x1 = constant(3.)
      x2 = constant(5.)
      y = MyMultiply(x1, x2)
      dy = gradients.gradients(y, [x1, x2])

      self.assertAllEqual([3., 5.], self.evaluate(dy))

  def testCustomGradientClass(self):
    # [FUNCTION DOC]
    # Uses the custom-gradient decorator on a method of a local class and asserts gradients of
    # the product with respect to its inputs evaluate to 3 and 5 (swapped partials).

    class Model:
      # [CLASS DOC]
      # Local helper class with a single method decorated for custom gradients.

      @custom_gradient.custom_gradient
      def Multiply(self, x1, x2):
        # [FUNCTION DOC]
        # HELPER: method returning the product of x1 and x2 and a gradient function that returns
        # a pair of dy times x1 and dy times x2.
        result = x1 * x2
        grad = lambda dy: (dy * x1, dy * x2)
        return result, grad

    with ops.Graph().as_default():
      x1 = constant(3.)
      x2 = constant(5.)
      m = Model()
      y = m.Multiply(x1, x2)
      dy = gradients.gradients(y, [x1, x2])
      self.assertAllEqual([3., 5.], self.evaluate(dy))

  def testCustomGradientErrors(self):
    # [FUNCTION DOC]
    # The custom gradient function raises a runtime error; asserts that exception propagates out
    # of the gradients call.

    @custom_gradient.custom_gradient
    def F(x):
      # [FUNCTION DOC]
      # HELPER: identity with a gradient function that always fails.

      def Grad(_):
        # [FUNCTION DOC]
        # HELPER: gradient function that unconditionally raises a runtime error.
        raise RuntimeError("x")

      return x, Grad

    with ops.Graph().as_default():
      x = constant(1.0)
      y = F(x)
      with self.assertRaises(RuntimeError):
        gradients.gradients(y, x)

  def testCustomGradientWithVariables(self):
    # [FUNCTION DOC]
    # A custom-gradient function applies a dense layer; its gradient function receives the list of
    # variables used and returns ones for the kernel gradient. Asserts both gradients are non-null
    # and that the kernel gradient sums to 12.

    @custom_gradient.custom_gradient
    def F(x):
      # [FUNCTION DOC]
      # HELPER: applies a bias-free dense layer of width three and returns the output with a
      # gradient function.
      out = core_layers.dense(x, 3, use_bias=False)

      def Grad(out_grad, variables=None):  # pylint: disable=redefined-outer-name
        # [FUNCTION DOC]
        # HELPER: gradient function; asserts exactly one variable was passed, computes the input
        # gradient through the layer, and returns it with a four-by-three tensor of ones as the
        # kernel gradient.
        self.assertEqual(1, len(variables))  # pylint: disable=g-generic-assert
        grads = gradients.gradients(out, [x, variables[0]], grad_ys=out_grad)
        return grads[0], [array_ops.ones((4, 3))]

      return out, Grad

    with ops.Graph().as_default():
      x = array_ops.ones((2, 4))
      with variable_scope.variable_scope("f", use_resource=True) as vs:
        y = F(x)
        all_vars = vs.global_variables()
        assert len(all_vars) == 1
      grads = gradients.gradients(y, [x, all_vars[0]])
      for g in grads:
        self.assertIsNotNone(g)

      self.evaluate(variables.global_variables_initializer())
      dw = self.evaluate(math_ops.reduce_sum(grads[1]))
      self.assertEqual(12., dw)

  @parameterized.named_parameters([
      dict(
          testcase_name="Eager",
          decorator=lambda f: f),
      dict(
          testcase_name="Function",
          decorator=def_function.function),
  ])
  def testCustomGradientRaggedTensor(self, decorator):
    # [FUNCTION DOC]
    # Parameterized over running eagerly or wrapped in a function. In eager mode, applies a custom
    # gradient square function to a ragged tensor and asserts both the output and the gradient are
    # ragged tensors with the expected values (squares, and threes).
    with context.eager_mode():
      @custom_gradient.custom_gradient
      def F(x):
        # [FUNCTION DOC]
        # HELPER: squares x; returns the result and a gradient function.
        out = x * x

        def Grad(*grad):
          # [FUNCTION DOC]
          # HELPER: returns three times the first upstream gradient.
          return 3 * grad[0]

        return out, Grad

      rt = ragged_factory_ops.constant([[1., 2.], [3.]])
      with backprop.GradientTape() as tape:
        tape.watch(rt.values)
        out = decorator(F)(rt)
        result = tape.gradient(out, rt)

      self.assertIsInstance(out, ragged_tensor.RaggedTensor)
      self.assertAllEqual(out, [[1., 4.], [9.]])
      self.assertIsInstance(result, ragged_tensor.RaggedTensor)
      self.assertAllEqual(result, [[3., 3.], [3.]])

  @parameterized.named_parameters([
      dict(
          testcase_name="Eager",
          decorator=lambda f: f),
      dict(
          testcase_name="Function",
          decorator=def_function.function),
  ])
  def testCustomGradientMultipleRaggedTensors(self, decorator):
    # [FUNCTION DOC]
    # Parameterized over eager or function execution. A custom-gradient function takes two ragged
    # tensors and returns a pair (square of the first, double of the second). Asserts the outputs
    # and the gradients (threes and fours) are ragged tensors with the expected values.
    with context.eager_mode():
      @custom_gradient.custom_gradient
      def F(x, y):
        # [FUNCTION DOC]
        # HELPER: returns the pair of x squared and two times y with a gradient function.
        out = (x * x, 2 * y)

        def Grad(*grad):
          # [FUNCTION DOC]
          # HELPER: returns three times the first upstream gradient and four times the second.
          return (3 * grad[0], 4 * grad[1])

        return out, Grad

      rt1 = ragged_factory_ops.constant([[1., 2.], [3.]])
      rt2 = ragged_factory_ops.constant([[4.], [5., 6.]])
      with backprop.GradientTape() as tape:
        tape.watch((rt1, rt2))
        out1, out2 = decorator(F)(rt1, rt2)
        grad1, grad2 = tape.gradient((out1, out2), (rt1, rt2))

      self.assertIsInstance(out1, ragged_tensor.RaggedTensor)
      self.assertAllEqual(out1, [[1., 4.], [9.]])
      self.assertIsInstance(out2, ragged_tensor.RaggedTensor)
      self.assertAllEqual(out2, [[8.], [10., 12.]])
      self.assertIsInstance(grad1, ragged_tensor.RaggedTensor)
      self.assertAllEqual(grad1, [[3., 3.], [3.]])
      self.assertIsInstance(grad2, ragged_tensor.RaggedTensor)
      self.assertAllEqual(grad2, [[4.], [4., 4.]])

  @test_util.enable_quantized_dtypes_training
  def testCustomGradientQuantizedDtypeTraining(self):
    # [FUNCTION DOC]
    # Skipped on macOS (known failure, tracked by a bug). In eager mode with quantized-dtype
    # training enabled, applies an identity custom gradient to a qint8 tensor under a gradient tape.
    # Asserts the output equals the input, the gradient is all ones, and the gradient dtype is
    # qint8.
    # TODO(b/309175067): Remove below skipTest() when fixed.
    if sys.platform == "darwin":
      self.skipTest("This test fails in TF MacOS nightly and continuous builds")
    with context.eager_mode():
      @custom_gradient.custom_gradient
      def F(x):
        # [FUNCTION DOC]
        # HELPER: identity returning x and a gradient function.
        out = x

        def Grad(*grad):
          # [FUNCTION DOC]
          # HELPER: passes the upstream gradients through unchanged.
          return grad

        return out, Grad

      x = constant_op.constant([[1, 2], [3, 4]], dtype=dtypes.qint8)
      with backprop.GradientTape() as tape:
        tape.watch(x)
        out = F(x)
        result = tape.gradient(out, x)

      self.assertAllEqual(out, [[1, 2], [3, 4]])
      self.assertAllEqual(result, [[1, 1], [1, 1]])
      self.assertEqual(result.dtype, dtypes.qint8)

  def testCustomGradientWithCapture(self):
    # [FUNCTION DOC]
    # Inside a legacy graph function, a custom-gradient multiply uses a tensor captured from the
    # outer graph and the function argument. The function returns the gradient with respect to its
    # argument; asserts it evaluates to 5 (the gradient function returns dy times x2 for the second input).
    with ops.Graph().as_default():
      x = constant(3.)

      @framework_function.Defun(dtypes.float32)
      def F(y):
        # [FUNCTION DOC]
        # HELPER: graph function body: multiplies the captured x by y using a custom-gradient
        # multiply and returns the gradients of the product with respect to y.

        @custom_gradient.custom_gradient
        def MyMultiply(x1, x2):
          # [FUNCTION DOC]
          # HELPER: custom-gradient multiply returning the product and a gradient function.
          result = x1 * x2

          def Grad(dy):
            # [FUNCTION DOC]
            # HELPER: returns dy times x1 and dy times x2 (swapped ordering).
            # Switched the ordering here.
            return [dy * x1, dy * x2]

          return result, Grad

        res = MyMultiply(x, y)
        return gradients.gradients(res, [y])

      y = constant(5.)
      dy = F(y)
      self.assertAllEqual(5., self.evaluate(dy))

  def testCustomGradientWithVariablesNoFalsePositives(self):
    # [FUNCTION DOC]
    # Builds several dense layers outside a custom-gradient function and one inside it, in a
    # variable scope. Verifies only the variable used inside the decorated function is passed to
    # its gradient function (which asserts exactly one), that all four variables get gradients,
    # and that the last variable's gradient sums to nine.

    @custom_gradient.custom_gradient
    def F(x):
      # [FUNCTION DOC]
      # HELPER: applies a bias-free dense layer of width three; returns the output and gradient
      # function.
      out = core_layers.dense(x, 3, use_bias=False)

      def Grad(out_grad, variables=None):  # pylint: disable=redefined-outer-name
        # [FUNCTION DOC]
        # HELPER: asserts one variable was passed; returns the input gradient and a three-by-three
        # tensor of ones as the variable gradient.
        self.assertEqual(1, len(variables))  # pylint: disable=g-generic-assert
        grads = gradients.gradients(out, [x, variables[0]], grad_ys=out_grad)
        return grads[0], [array_ops.ones((3, 3))]

      return out, Grad

    with ops.Graph().as_default():
      with variable_scope.variable_scope("f", use_resource=True) as vs:
        a = array_ops.ones((2, 4))

        # Variables in these layers shouldn't be picked up by the decorator.
        b = core_layers.dense(a, 3, use_bias=False)
        c = core_layers.dense(b, 3, use_bias=False)
        x = core_layers.dense(b, 3, use_bias=False) + c

        # Only the variables used in F.
        y = F(x)

        all_vars = vs.global_variables()
        assert len(all_vars) == 4
      grads = gradients.gradients(y, [x] + all_vars)
      _, var_grads = grads[0], grads[1:]
      for g in grads:
        self.assertIsNotNone(g)

      self.evaluate(variables.global_variables_initializer())
      dw = self.evaluate(math_ops.reduce_sum(var_grads[-1]))
      self.assertEqual(9., dw)

  def testCustomGradientWithVariablesEager(self):
    # [FUNCTION DOC]
    # Eager variant: a custom-gradient function applies a dense layer object under a gradient tape;
    # its gradient function returns ones for input and kernel. Asserts the input gradient sums to 6
    # and the kernel gradient sums to 8.
    with context.eager_mode():
      layer = core_layers.Dense(4, use_bias=False)

      @custom_gradient.custom_gradient
      def F(x):
        # [FUNCTION DOC]
        # HELPER: applies the layer to x; returns output and gradient function.
        out = layer(x)

        def Grad(out_grad, variables=None):  # pylint: disable=redefined-outer-name
          # [FUNCTION DOC]
          # HELPER: asserts one variable was passed; returns ones shaped (3, 2) for the input and
          # ones shaped (2, 4) for the kernel; ignores the upstream gradient.
          del out_grad
          self.assertEqual(1, len(variables))  # pylint: disable=g-generic-assert
          return (array_ops.ones((3, 2)),
                  [array_ops.ones((2, 4))])

        return out, Grad

      x = array_ops.ones((3, 2)) + 2.
      with backprop.GradientTape() as tape:
        tape.watch(x)
        y = F(x)
      w, = layer.variables
      dx, dw = tape.gradient(y, [x, w])
      self.assertEqual(6., math_ops.reduce_sum(dx).numpy())
      self.assertEqual(8., math_ops.reduce_sum(dw).numpy())

  @test_util.run_v1_only("b/120545219")
  def testCustomGradientErrorsWithNonResourceVariables(self):
    # [FUNCTION DOC]
    # Graph-mode only. Calls custom-gradient functions that create variables with and without
    # resource semantics. Inside an outer scope that enables resource variables, a function whose
    # inner scope forces non-resource variables must raise a type error saying they must be
    # resource variables; the reverse combination must succeed.

    def F(x, use_resource=False):
      # [FUNCTION DOC]
      # HELPER: creates a dense layer in a scope with the requested resource setting; returns the
      # output and a gradient function (which asserts one variable and returns ones).
      with variable_scope.variable_scope("f", use_resource=use_resource):
        out = core_layers.dense(x, 4, use_bias=False)

      def Grad(out_grad, variables=None):  # pylint: disable=redefined-outer-name
        # [FUNCTION DOC]
        # HELPER: ignores out_grad; asserts one variable was passed; returns ones for the input
        # and kernel gradients.
        del out_grad
        self.assertEqual(1, len(variables))  # pylint: disable=g-generic-assert
        return (array_ops.ones((3, 2)), [array_ops.ones((2, 4))])

      return out, Grad

    @custom_gradient.custom_gradient
    def FResource(x):
      # [FUNCTION DOC]
      # HELPER: custom-gradient wrapper around F using resource variables.
      return F(x, use_resource=True)

    @custom_gradient.custom_gradient
    def FNonResource(x):
      # [FUNCTION DOC]
      # HELPER: custom-gradient wrapper around F using non-resource variables.
      return F(x, use_resource=False)

    x = array_ops.ones((3, 2)) + 2.

    # Wrapping scope has use_resource=True but inner scope sets to False. Fails.
    with variable_scope.variable_scope("vs1", use_resource=True):
      with self.assertRaisesWithPredicateMatch(TypeError,
                                               "must be `ResourceVariable`s"):
        FNonResource(x)

    # Wrapping scope has use_resource=False but inner scope sets to True.
    # Passes.
    with variable_scope.variable_scope("vs2", use_resource=False):
      FResource(x)

  @parameterized.parameters(True, False)
  def testCustomGradientVariablesKwonlyArgs(self, anonymous_varargs):
    # [FUNCTION DOC]
    # Parameterized on whether the gradient function uses anonymous or named variadic positional
    # arguments. In eager mode, a custom-gradient function multiplies its input by a captured
    # variable; its gradient function receives the variables as a keyword-only argument. Asserts
    # the captured variable is passed and the gradient with respect to x equals 12.
    with context.eager_mode():
      x_captured = variables.Variable(3.)  # Used by FuncMult
      @custom_gradient.custom_gradient
      def FuncMult(x):
        # [FUNCTION DOC]
        # HELPER: returns x times the captured variable and a gradient function chosen by the
        # parameterized flag.
        def ActualGrad(dy, variables):  # pylint: disable=redefined-outer-name
          # [FUNCTION DOC]
          # HELPER: shared gradient logic; asserts one variable which is the captured one, then
          # returns four times captured times dy for x and a list with five times x times dy as
          # the variable gradient.
          self.assertLen(variables, 1)
          self.assertIs(variables[0], x_captured)
          x_captured_grad = 5. * x * dy
          return (4. * x_captured * dy, [x_captured_grad])
        # Define the returned GradMult, using varargs; "variables" is kwonlyarg
        if anonymous_varargs:
          def GradMult(dy, *, variables=None):  # pylint: disable=redefined-outer-name
            # [FUNCTION DOC]
            # HELPER: gradient function with a single upstream gradient and keyword-only
            # variables; delegates to the shared gradient logic.
            return ActualGrad(dy, variables)
        else:
          def GradMult(*dys, variables=None):  # pylint: disable=redefined-outer-name
            # [FUNCTION DOC]
            # HELPER: gradient function taking variadic upstream gradients and keyword-only
            # variables; delegates to the shared logic with the first gradient.
            return ActualGrad(dys[0], variables)

        return x * x_captured, GradMult

      x = variables.Variable(6.)
      with backprop.GradientTape(persistent=True) as g:
        y = FuncMult(x)
      self.assertAllEqual(g.gradient(y, x), 4. * 3.)

  def testWithNumpyInputs(self):
    # [FUNCTION DOC]
    # Smoke test in eager mode: calls a custom-gradient function with a numpy array input and
    # expects no error; nothing is asserted beyond that.
    with context.eager_mode():

      @custom_gradient.custom_gradient
      def F(x):
        # [FUNCTION DOC]
        # HELPER: identity returning x and a gradient function.
        out = x

        def Grad(_):
          # [FUNCTION DOC]
          # HELPER: gradient function returning a pair of null values.
          return (None, None)

        return out, Grad

      x = np.ones((3, 2), dtype=np.float32)
      # Smoke test to ensure numpy inputs are accepted
      F(x)

  @test_util.run_v1_only("b/120545219")
  def testRVGradientsDynamicCond(self):
    # [FUNCTION DOC]
    # Graph-mode only. A conditional picks two times or three times a resource variable based on a
    # placeholder defaulting to true. Asserts the gradient is 2 by default and 3 when the
    # condition is fed false.
    with self.cached_session():
      alpha = resource_variable_ops.ResourceVariable(
          np.random.random((1,)),
          dtype="float32")

      conditional = array_ops.placeholder_with_default(True, shape=())
      output = cond.cond(
          conditional, lambda: alpha * 2, lambda: alpha * 3)

      g, = gradients_impl.gradients(output, alpha)
      self.evaluate(variables.global_variables_initializer())
      self.assertAllEqual(g, [2.0])
      self.assertAllEqual(g.eval(feed_dict={conditional: False}), [3.0])

  def testRecursiveCustomGradient(self):
    # [FUNCTION DOC]
    # Nests one custom-gradient function (a dense layer) inside another that calls it. Each
    # gradient function asserts one variable was passed and returns ones for the kernel. Asserts
    # all gradients are non-null and the kernel gradient sums to 12.
    @custom_gradient.custom_gradient
    def F(x):
      # [FUNCTION DOC]
      # HELPER: inner custom-gradient function applying a bias-free dense layer of width three.
      out = core_layers.dense(x, 3, use_bias=False)

      def Grad(out_grad, variables=None):  # pylint: disable=redefined-outer-name
        # [FUNCTION DOC]
        # HELPER: asserts one variable; returns the input gradient and a four-by-three tensor of
        # ones for the kernel.
        self.assertEqual(1, len(variables))  # pylint: disable=g-generic-assert
        grads = gradients.gradients(out, [x, variables[0]], grad_ys=out_grad)
        return grads[0], [array_ops.ones((4, 3))]

      return out, Grad

    @custom_gradient.custom_gradient
    def DoubleF(x):
      # [FUNCTION DOC]
      # HELPER: outer custom-gradient function that calls the inner one.
      out = F(x)

      def Grad(out_grad, variables=None):  # pylint: disable=redefined-outer-name
        # [FUNCTION DOC]
        # HELPER: outer gradient function; asserts one variable; returns the input gradient and
        # ones for the kernel gradient.
        self.assertEqual(1, len(variables))  # pylint: disable=g-generic-assert
        grads = gradients.gradients(out, [x, variables[0]], grad_ys=out_grad)
        return grads[0], [array_ops.ones((4, 3))]

      return out, Grad
    with ops.Graph().as_default():
      x = array_ops.ones((2, 4))
      with variable_scope.variable_scope("f", use_resource=True) as vs:
        y = DoubleF(x)
        all_vars = vs.global_variables()
        assert len(all_vars) == 1
      grads = gradients.gradients(y, [x, all_vars[0]])
      for g in grads:
        self.assertIsNotNone(g)

      self.evaluate(variables.global_variables_initializer())
      dw = self.evaluate(math_ops.reduce_sum(grads[1]))
      self.assertEqual(12., dw)

  @parameterized.named_parameters(
      [(("_%s_%s" % (x_struct, y_struct)).replace(" ", "").replace("None", ""),  # pylint: disable=g-complex-comprehension
        x_struct, y_struct)
       for y_struct in [[None, ()], (None, (), [], (None, ((), None)))]
       for x_struct in [(None, ()), (((), ()), [None, None], [], (None, ()))]
      ])
  @test_util.run_in_graph_and_eager_modes
  def testCustomGradientStructuralInputOutput(self, x_struct, y_struct):
    # [FUNCTION DOC]
    # Parameterized over combinations of nested input and output structures (with empty and null
    # entries), in graph and eager modes. A custom-gradient function checks it sees the input
    # structure and returns zeros in the output structure; its gradient function checks the
    # gradient count and returns zeros for inputs. Asserts the structures of the computed output
    # and input gradients equal the expected ones.
    """Tests that custom_gradient can handle structured inputs/outputs."""
    def Zeros(x):
      # [FUNCTION DOC]
      # HELPER: returns a structure with the same nesting as x where every leaf is a float32
      # scalar zero.
      return nest.map_structure(lambda _: array_ops.zeros([], "float32"), x)
    def GetStruct(x):
      # [FUNCTION DOC]
      # HELPER: returns a structure with the nesting of x where every leaf is replaced with null.
      return nest.map_structure(lambda _: None, x)

    def MakeVjp(f, *x):
      # [FUNCTION DOC]
      # HELPER: runs f on x under a persistent gradient tape and returns the output together with
      # a function mapping output gradients to input gradients.
      with backprop.GradientTape(persistent=True) as tape:
        tape.watch(nest.flatten(x))
        y = f(*x)
      def Vjp(dy):
        # [FUNCTION DOC]
        # HELPER: returns the tape gradient of y with respect to x seeded with dy.
        return tape.gradient(y, x, output_gradients=dy)
      return y, Vjp

    @custom_gradient.custom_gradient
    def F(*x):
      # [FUNCTION DOC]
      # HELPER: custom-gradient function; asserts the input structure equals x_struct and returns
      # zeros shaped like y_struct with a gradient function.
      self.assertEqual(x_struct, GetStruct(x))
      def Vjp(*dy):
        # [FUNCTION DOC]
        # HELPER: gradient function; asserts the number of flattened upstream gradients equals the
        # number of leaves in y_struct; returns a flat list of zeros for the inputs.
        self.assertEqual(len(nest.flatten(y_struct)),
                         len(nest.flatten(dy)))
        return nest.flatten(Zeros(x_struct))
      return Zeros(y_struct), Vjp

    x, dy = Zeros([x_struct, y_struct])
    y, vjp = MakeVjp(F, *x)
    dx = vjp(dy)
    self.assertEqual(x_struct, GetStruct(dx))
    self.assertEqual(y_struct, GetStruct(y))


class TensorListGradientsTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests gradients where outputs or initial gradients are tensor lists.

  def testDefaultGradYs(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the variable meant to hold a separate gradient list is created empty and then
    # overwritten by pushing onto the forward list rather than the empty gradient list, so the
    # initial gradient list is not independent of the forward list.
    # Pushes a value onto a tensor list and takes the gradient of the list with respect to the
    # value using the list as the initial gradient; asserts the result evaluates to 5.
    with ops.Graph().as_default():
      tl = list_ops.empty_tensor_list(
          element_dtype=dtypes.float32,
          element_shape=ops.convert_to_tensor([], dtype=dtypes.int32))
      a = constant(1.0)
      tl = list_ops.tensor_list_push_back(tl, a)

      grad_tl = list_ops.empty_tensor_list(
          element_dtype=dtypes.float32,
          element_shape=ops.convert_to_tensor([], dtype=dtypes.int32))
      grad_tl = list_ops.tensor_list_push_back(tl, constant(5.0))

      grad = gradients.gradients(tl, a, grad_ys=grad_tl)[0]

      self.assertEqual(self.evaluate(grad), 5.)


class VariablesGradientTest(test_util.TensorFlowTestCase,
                            parameterized.TestCase):
  # [CLASS DOC]
  # Tests the recompute-gradient wrapper: that wrapped functions give the same gradients as
  # unwrapped ones with respect to inputs and variables, handle zero-size and mixed-dtype inputs,
  # variable scopes, repeated arguments and higher-order derivatives. Provides helper methods for
  # computing gradients and numerically checking them. Supports parameterized tests.

  def _TestFnVariablesGradient(self, inputs, test_fn, vars_to_grad):
    # [FUNCTION DOC]
    # HELPER: computes gradients of a function with and without recompute-gradient wrapping.
    # Inputs: inputs (first argument to the function), test_fn (function of inputs and variables),
    # vars_to_grad (tensor or variable to differentiate with respect to, also the second function
    # argument). Under a persistent tape, runs both versions, and returns a pair: the gradients
    # of the wrapped version and of the plain version.
    """Returns gradients of `test_model` with respect to `vars_to_grad`."""

    test_fn_re = custom_gradient.recompute_grad(test_fn)

    with backprop.GradientTape(persistent=True) as tape:
      tape.watch(vars_to_grad)
      out_re = test_fn_re(inputs, vars_to_grad)
      out = test_fn(inputs, vars_to_grad)

    grads_re = tape.gradient(out_re, vars_to_grad)
    grads = tape.gradient(out, vars_to_grad)

    return grads_re, grads

  def _grad(self, f, argnums=0):
    # [FUNCTION DOC]
    # HELPER: returns a function that computes the gradient of f. Inputs: f (callable) and argnums
    # (index of the argument to differentiate, default first). The returned function calls f under
    # a gradient tape and returns the gradient with respect to that argument, using zeros for
    # unconnected gradients.
    """Return a function which computes the gradient of `f`."""

    def F(*params):
      # [FUNCTION DOC]
      # HELPER: returned gradient function: watches all params, evaluates f, and returns the
      # gradient with respect to the chosen parameter (zeros if unconnected).
      with backprop.GradientTape() as tape:
        tape.watch(params)
        outputs = f(*params)
      return tape.gradient(
          outputs,
          params[argnums],
          unconnected_gradients=unconnected_gradients.UnconnectedGradients.ZERO)

    return F

  def _test_gradients(self, f, inputs, order, delta=1e-3, rtol=1e-2, atol=1e-6):
    # [FUNCTION DOC]
    # HELPER: numerically checks backward Jacobians for gradients of orders one through order.
    # Inputs: f (function), inputs (list of input tensors), order (positive integer), delta
    # (finite-difference step), rtol and atol (comparison tolerances). Raises a value error if
    # order is less than one. Recursively checks the gradient function for lower orders, then
    # compares the symbolic and numeric Jacobians of f, asserting they are close.
    """Tests backward jacobians of `f`'s [0, `order`)-order gradients."""
    if order < 1:
      raise ValueError(
          "`order` should be a positive integer, got '{}'.".format(order))
    if order > 1:
      self._test_gradients(
          f=self._grad(f),
          inputs=inputs,
          order=order - 1,
          delta=delta,
          rtol=rtol,
          atol=atol)
    sym_jac_back, num_jac = gradient_checker_v2.compute_gradient(
        f, inputs, delta=delta)
    self.assertAllClose(num_jac, sym_jac_back, rtol=rtol, atol=atol)

  def testRecomputeGradWrapped(self):
    # [FUNCTION DOC]
    # Asserts the function returned by the recompute-gradient wrapper exposes the original
    # function as its wrapped attribute.

    def f(x):  # pylint: disable=invalid-name
      # [FUNCTION DOC]
      # HELPER: simple function returning two times x.
      return 2 * x

    g = custom_gradient.recompute_grad(f)
    self.assertIs(g.__wrapped__, f)

  def testRecomputeGradZeroSizeInput(self):
    # [FUNCTION DOC]
    # With an empty input, compares gradients of a doubling function with and without recompute
    # wrapping, both as a plain function and as a traced function with an unknown-shape signature;
    # asserts each pair is close.

    def F(x):
      # [FUNCTION DOC]
      # HELPER: returns two times x.
      return 2 * x

    x = array_ops.constant(())
    grads_re = self._grad(custom_gradient.recompute_grad(F))(x)
    grads = self._grad(F)(x)
    self.assertAllClose(grads_re, grads)

    f_graph = def_function.function(
        F, input_signature=[tensor.TensorSpec(None)])
    grads_re = self._grad(custom_gradient.recompute_grad(f_graph))(x)
    grads = self._grad(f_graph)(x)
    self.assertAllClose(grads_re, grads)

  def testRecomputeGradDifferentDtypesInputs(self):
    # [FUNCTION DOC]
    # Uses a function with an integer and a float input; compares gradients with and without
    # recompute wrapping, both plain and traced with a typed input signature; asserts they are
    # close.

    def F(x1, x2):
      # [FUNCTION DOC]
      # HELPER: returns a pair of two times each input.
      return 2 * x1, 2 * x2

    x1 = array_ops.constant(1, dtype=dtypes.int32)
    x2 = array_ops.constant(1., dtype=dtypes.float32)
    grads_re = self._grad(custom_gradient.recompute_grad(F))(x1, x2)
    grads = self._grad(F)(x1, x2)
    self.assertAllClose(grads_re, grads)

    f_graph = def_function.function(
        F,
        input_signature=[
            tensor.TensorSpec(None, dtype=dtypes.int32),
            tensor.TensorSpec(None, dtype=dtypes.float32),
        ])
    grads_re = self._grad(custom_gradient.recompute_grad(f_graph))(x1, x2)
    grads = self._grad(f_graph)(x1, x2)
    self.assertAllClose(grads_re, grads)

  @test_util.run_v2_only
  def testCustomGradientRecomputeGradHigherOrder(self):
    # [FUNCTION DOC]
    # Version-2 only. Numerically checks gradients up to third order of a recompute-wrapped
    # function (product of squared tanh values) at a one-element input.

    @custom_gradient.recompute_grad
    def F(x):
      # [FUNCTION DOC]
      # HELPER: returns the reduced product of tanh of x, squared.
      return math_ops.reduce_prod(math_ops.tanh(x)**2)

    self._test_gradients(F, [constant_op.constant([1.])], order=3)

  @test_util.run_in_graph_and_eager_modes
  def testFnRecompute(self):
    # [FUNCTION DOC]
    # In graph and eager modes, compares wrapped and unwrapped gradients for a multiply function
    # and a function returning a sequence, differentiating with respect to a tensor input and to a
    # resource variable. Asserts all corresponding gradients are close.
    """Checks that recompute_grad works grads of function args."""

    def TestFn(inputs, input_vars):
      # [FUNCTION DOC]
      # HELPER: returns inputs times input_vars.
      return inputs * input_vars

    def TestFnSeq(inputs, input_vars):
      # [FUNCTION DOC]
      # HELPER: returns a pair: the product and twice the product.
      return (inputs * input_vars, inputs * input_vars * 2.0)

    with variable_scope.variable_scope("test", use_resource=True):
      test_var = variable_scope.get_variable(
          name="test_var",
          shape=10,
          trainable=True,
      )
      self.evaluate(test_var.assign(np.ones([10])))
      test_input = constant(np.ones((10, 10), dtype=np.float32))

      grads_re, grads = self._TestFnVariablesGradient(test_input, TestFn,
                                                      test_input)

      grads_re = self.evaluate(grads_re)
      grads = self.evaluate(grads)
      for g, g_re in zip(grads, grads_re):
        self.assertAllClose(g, g_re)

      grads_re, grads = self._TestFnVariablesGradient(test_input, TestFn,
                                                      test_var)
      grads_re = self.evaluate(grads_re)
      grads = self.evaluate(grads)
      for g, g_re in zip(grads, grads_re):
        self.assertAllClose(g, g_re)

      # Regression test for wrapping sequence outputting functions.
      grads_re, grads = self._TestFnVariablesGradient(test_input, TestFnSeq,
                                                      test_input)
      grads_re = self.evaluate(grads_re)
      grads = self.evaluate(grads)
      for g, g_re in zip(grads, grads_re):
        self.assertAllClose(g, g_re)

      grads_re, grads = self._TestFnVariablesGradient(test_input, TestFnSeq,
                                                      test_var)
      grads_re = self.evaluate(grads_re)
      grads = self.evaluate(grads)
      for g, g_re in zip(grads, grads_re):
        self.assertAllClose(g, g_re)

  @parameterized.parameters(set((True, context.executing_eagerly())))
  def testFnRecomputeWithScopeGradient(self, use_tape):
    # [FUNCTION DOC]
    # Parameterized on whether to use a gradient tape. Wraps a function that creates a variable
    # inside a nested variable scope, runs wrapped and unwrapped versions under an output scope
    # that allows reuse, then compares gradients with respect to all trainable variables; asserts
    # they are close.
    """Checks that recompute_grad works with var scope and GradientTape."""

    def TestFn(input_t):
      # [FUNCTION DOC]
      # HELPER: inside an inner variable scope, gets a trainable variable of size ten and returns
      # the input times it.
      with variable_scope.variable_scope("inner_scope"):
        test_var = variable_scope.get_variable(
            name="test_var",
            shape=10,
            trainable=True,
        )
        return input_t * test_var

    test_input_t = constant(np.zeros((10, 10), dtype=np.float32))

    with variable_scope.variable_scope(
        "output_scope", reuse=variable_scope.AUTO_REUSE, use_resource=True):
      test_fn_re = custom_gradient.recompute_grad(TestFn)

      with test_util.AbstractGradientTape(
          use_tape=use_tape, persistent=True) as tape:
        out_re = test_fn_re(test_input_t)
        out = TestFn(test_input_t)

    self.evaluate(variables.global_variables_initializer())
    grads_re = tape.gradient(out_re, variables.trainable_variables())
    grads = tape.gradient(out, variables.trainable_variables())

    grads_re = self.evaluate(grads_re)
    grads = self.evaluate(grads)
    for g, g_re in zip(grads, grads_re):
      self.assertAllClose(g, g_re)

  @test_util.run_in_graph_and_eager_modes
  def testFnRecomputeSameTensor(self):
    # [FUNCTION DOC]
    # Regression test: calls a recompute-wrapped function with the same tensor for both arguments,
    # for a multiplying function and one that ignores its second argument; asserts wrapped and
    # unwrapped gradients are close.
    """Check recompute_grad when wrapped f called as f(x, x) - b/147369366."""

    def TestFnMul(x, y):
      # [FUNCTION DOC]
      # HELPER: returns x times y.
      return x * y

    def TestFnSingleVar(x, y):
      # [FUNCTION DOC]
      # HELPER: returns x and ignores y.
      # pylint: disable=unused-argument
      return x

    with variable_scope.variable_scope("test", use_resource=True):
      x = array_ops.ones((10))

      grads_re, grads = self._TestFnVariablesGradient(x, TestFnMul,
                                                      x)
      grads_re = self.evaluate(grads_re)
      grads = self.evaluate(grads)
      for g, g_re in zip(grads, grads_re):
        self.assertAllClose(g, g_re)

      grads_re, grads = self._TestFnVariablesGradient(x, TestFnSingleVar,
                                                      x)
      grads_re = self.evaluate(grads_re)
      grads = self.evaluate(grads)
      for g, g_re in zip(grads, grads_re):
        self.assertAllClose(g, g_re)


class GradPassThroughTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests the grad-pass-through wrapper, which forwards gradients through a non-differentiable
  # op, in graph (version 1) and eager (version 2) forms.

  @test_util.run_v1_only("b/120545219")
  def test_gradients_v1(self):
    # [FUNCTION DOC]
    # Graph-mode only. Shows an assign op has a null gradient, that wrapping it with grad
    # pass-through forwards gradients so the gradient of the square of z is six, and that a
    # variable involved in the wrapped op gets a null gradient.
    x = variable_scope.get_variable(
        name="x", shape=(), initializer=init_ops.constant_initializer(1.0),
        use_resource=True)
    z = variable_scope.get_variable(
        name="z", shape=(), initializer=init_ops.constant_initializer(3.0),
        use_resource=True)

    # Verify that assign op is not differentiable
    y = state_ops.assign(x, z**2)
    grads = gradients.gradients(y, z)
    self.assertIsNone(grads[0])

    # Verify that when the (non differentiable) assign op is wrapped with
    # grad_pass_through, gradients are correctly forwarded to the inputs.
    # Form an input as quadratic function of variable z and check that the
    # gradient of output wrt to z is correct.
    y = custom_gradient.grad_pass_through(
        lambda v: state_ops.assign(x, v))(z**2)
    grads = gradients.gradients(y, z)

    with self.cached_session():
      self.evaluate(variables.global_variables_initializer())
      self.assertAllClose(grads[0], 6.0)

    # Verify that variables involved in the wrapped op do not receive gradients.
    y = custom_gradient.grad_pass_through(lambda v: x * v)(z)
    grads = gradients.gradients(y, x)
    self.assertIsNone(grads[0])

  @test_util.run_v2_only
  def test_gradients_v2(self):
    # [FUNCTION DOC]
    # Version-2 (eager) counterpart using gradient tapes: an assign has a null gradient; wrapped
    # with grad pass-through, the gradient with respect to z is six; a variable inside the wrapped
    # op gets a null gradient.
    x = variables.Variable(1.0, name="x")
    z = variables.Variable(3.0, name="z")

    # Verify that assign op is not differentiable
    with backprop.GradientTape() as tape:
      y = x.assign(z**2)
    grads = tape.gradient(y, z)
    self.assertIsNone(grads)

    # Verify that when the (non differentiable) assign op is wrapped with
    # grad_pass_through, gradients are correctly forwarded to the inputs.
    # Form an input as quadratic function of variable z and check that the
    # gradient of output wrt to z is correct.
    with backprop.GradientTape() as tape:
      y = custom_gradient.grad_pass_through(x.assign)(z**2)
    grads = tape.gradient(y, z)
    self.assertAllClose(grads, 6.0)

    # Verify that variables involved in the wrapped op do not receive gradients.
    with backprop.GradientTape() as tape:
      y = custom_gradient.grad_pass_through(lambda v: x * v)(z)
    grads = tape.gradient(y, x)
    self.assertIsNone(grads)


if __name__ == "__main__":
  googletest.main()
