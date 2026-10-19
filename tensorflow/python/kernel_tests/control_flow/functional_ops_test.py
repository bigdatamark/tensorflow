# [MODULE DOC]
# Tests for the functional ops module (fold left/right, scan, remote calls, legacy If/While/For/Gradient
# ops and the Case op). Contains a module-level scoped reducer helper and two test case classes; many tests
# are graph-mode only and some require GPUs or local clusters. Importing registers the tensor array
# gradient functions as a side effect. Run through the standard test main. Not a helper module.
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
"""Tests for tensorflow.kernels.functional_ops."""

import numpy as np

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import cancellation
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function as eager_def_function
from tensorflow.python.eager import executor
from tensorflow.python.framework import config as framework_config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import collective_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
import tensorflow.python.ops.tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


# pylint: disable=invalid-name
def simple_scoped_fn(a, x):
  # [FUNCTION DOC]
  # HELPER: small reducer shared by the scoped fold and scan tests.
  # Takes an accumulator a and an element x (integer tensors). Inside a variable scope named "body" it
  # gets or creates an integer scalar variable "two" initialized to 2 and returns (a + x) multiplied by
  # that variable. Side effect: creates or reuses a variable in the current variable scope, so callers
  # can check that exactly one trainable variable named root/body/two:0 exists.
  """Simple function: (a, x) -> 2(x+a), but with "2" as a variable in scope."""
  with variable_scope.variable_scope("body"):
    # Dummy variable, just to check that scoping works as intended.
    two = variable_scope.get_variable(
        "two", [],
        dtype=dtypes.int32,
        initializer=init_ops.constant_initializer(2))
    return math_ops.multiply(math_ops.add(a, x), two)


@test_util.with_control_flow_v2
class FunctionalOpsTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the functional ops module: fold left, fold right, scan, remote function calls, the
  # legacy If, While, For and Gradient function-call ops, and the collective-in-remote-call setup.
  # Decorated to also run with control flow version 2. Many tests are graph-mode only (deprecated
  # version 1 or version 1 only). No shared fixtures other than the module-level scoped reducer helper.

  @test_util.run_in_graph_and_eager_modes
  def testFoldl_Simple(self):
    # [FUNCTION DOC]
    # Test (graph and eager) of a left fold over the six numbers one to six where each step adds the
    # element and doubles. Expects 208 without an initializer, and 880 when the initializer is 10.
    elems = constant_op.constant([1, 2, 3, 4, 5, 6], name="data")

    r = functional_ops.foldl(
        lambda a, x: math_ops.multiply(math_ops.add(a, x), 2),
        elems)
    self.assertAllEqual(208, self.evaluate(r))

    r = functional_ops.foldl(
        lambda a, x: math_ops.multiply(math_ops.add(a, x), 2),
        elems,
        initializer=10)
    self.assertAllEqual(880, self.evaluate(r))

  @test_util.run_in_graph_and_eager_modes
  def testFoldl_SingleInputMultiOutput(self):
    # [FUNCTION DOC]
    # Test (graph and eager) of a left fold that sums a float vector elements into a two-element
    # accumulator initialized to one and minus one. Expects the final accumulator to be 22 and 20.
    elems = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    initializer = np.array([1, -1.0])
    r = functional_ops.foldl(lambda a, x: a + x, elems, initializer)
    r_value = self.evaluate(r)

    self.assertAllEqual(22, r_value[0])
    self.assertAllEqual(20, r_value[1])

  @test_util.run_in_graph_and_eager_modes
  def testFoldl_MultiInputSingleOutput(self):
    # [FUNCTION DOC]
    # Test (graph and eager) of a left fold over a pair of inputs (a vector and its negation) with a
    # scalar float accumulator starting at one. Each step adds both components, so the total is
    # unchanged; asserts the result is 1.
    elems = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    initializer = np.array(1.0)
    r = functional_ops.foldl(lambda a, x: a + x[0] + x[1], (elems, -elems),
                             initializer)
    self.assertAllEqual(1, self.evaluate(r))

  @test_util.run_in_graph_and_eager_modes
  def testFoldl_MultiInputDifferentDimsSingleOutput(self):
    # [FUNCTION DOC]
    # Test (graph and eager) of a left fold over two inputs with different trailing dimensions: a two by
    # three matrix and a length-two vector (minus one, one). Accumulator starts at zeros of length three
    # and adds the row times the scalar each step. Expects the vector one, two, three.
    elems = np.array([[1.0, 1.0, 1.0], [2.0, 3.0, 4.0]])
    other_elems = np.array([-1.0, 1.0])
    initializer = np.array([0.0, 0.0, 0.0])
    r = functional_ops.foldl(lambda a, x: a + x[0] * x[1],
                             (elems, other_elems), initializer)
    self.assertAllEqual([1.0, 2.0, 3.0], self.evaluate(r))

  @test_util.run_deprecated_v1
  def testFoldl_Scoped(self):
    # [FUNCTION DOC]
    # Test (deprecated version 1 graph mode) that a left fold works with a reducer that creates a
    # variable in a variable scope. Under scope "root", folds the scoped reducer, checks there is exactly
    # one trainable variable named root/body/two:0, initializes variables and expects 208. Then enables
    # variable reuse, folds again with initializer 10 and expects 880 while still having one variable.
    with self.cached_session() as sess:
      with variable_scope.variable_scope("root") as varscope:
        elems = constant_op.constant([1, 2, 3, 4, 5, 6], name="data")

        r = functional_ops.foldl(simple_scoped_fn, elems)
        # Check that we have the one variable we asked for here.
        self.assertEqual(len(variables.trainable_variables()), 1)
        self.assertEqual(variables.trainable_variables()[0].name,
                         "root/body/two:0")
        sess.run([variables.global_variables_initializer()])
        self.assertAllEqual(208, self.evaluate(r))

        # Now let's reuse our single variable.
        varscope.reuse_variables()
        r = functional_ops.foldl(simple_scoped_fn, elems, initializer=10)
        self.assertEqual(len(variables.trainable_variables()), 1)
        self.assertAllEqual(880, self.evaluate(r))

  @test_util.run_in_graph_and_eager_modes
  def testFoldr_Simple(self):
    # [FUNCTION DOC]
    # Test (graph and eager) of a right fold over one to six with the add-then-double reducer.
    # Expects 450 without an initializer and 1282 with initializer 10.
    elems = constant_op.constant([1, 2, 3, 4, 5, 6], name="data")

    r = functional_ops.foldr(
        lambda a, x: math_ops.multiply(math_ops.add(a, x), 2),
        elems)
    self.assertAllEqual(450, self.evaluate(r))

    r = functional_ops.foldr(
        lambda a, x: math_ops.multiply(math_ops.add(a, x), 2),
        elems,
        initializer=10)
    self.assertAllEqual(1282, self.evaluate(r))

  @test_util.run_in_graph_and_eager_modes
  def testFoldr_SingleInputMultiOutput(self):
    # [FUNCTION DOC]
    # Test (graph and eager) of a right fold that sums a float vector into a two-element accumulator
    # initialized to one and minus one. Expects final values 22 and 20.
    elems = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    initializer = np.array([1, -1.0])
    r = functional_ops.foldr(lambda a, x: a + x, elems, initializer)
    r_value = self.evaluate(r)

    self.assertAllEqual(22, r_value[0])
    self.assertAllEqual(20, r_value[1])

  @test_util.run_in_graph_and_eager_modes
  def testFoldr_MultiInputSingleOutput(self):
    # [FUNCTION DOC]
    # Test (graph and eager) of a right fold over a vector and its negation with scalar accumulator
    # starting at one; the contributions cancel so the result must be 1.
    elems = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    initializer = np.array(1.0)
    r = functional_ops.foldr(lambda a, x: a + x[0] + x[1], (elems, -elems),
                             initializer)
    self.assertAllEqual(1, self.evaluate(r))

  @test_util.run_deprecated_v1
  def testFoldr_Scoped(self):
    # [FUNCTION DOC]
    # Test (deprecated version 1 graph mode) of a right fold with the variable-creating scoped reducer:
    # checks one trainable variable root/body/two:0 exists, expects 450, then reuses the variable with
    # initializer 10 and expects 1282 with still a single trainable variable.
    with self.cached_session() as sess:
      with variable_scope.variable_scope("root") as varscope:
        elems = constant_op.constant([1, 2, 3, 4, 5, 6], name="data")

        r = functional_ops.foldr(simple_scoped_fn, elems)
        # Check that we have the one variable we asked for here.
        self.assertEqual(len(variables.trainable_variables()), 1)
        self.assertEqual(variables.trainable_variables()[0].name,
                         "root/body/two:0")
        sess.run([variables.global_variables_initializer()])
        self.assertAllEqual(450, self.evaluate(r))

        # Now let's reuse our single variable.
        varscope.reuse_variables()
        r = functional_ops.foldr(simple_scoped_fn, elems, initializer=10)
        self.assertEqual(len(variables.trainable_variables()), 1)
        self.assertAllEqual(1282, self.evaluate(r))

  # pylint: disable=unnecessary-lambda
  @test_util.run_deprecated_v1
  def testFold_Grad(self):
    # [FUNCTION DOC]
    # Test (deprecated version 1 graph mode) of gradients through folds. Builds a left fold and then a
    # right fold that multiply floats one to six starting from initializer 2, takes the gradient of the
    # result with respect to the initializer and expects 720 in each case.
    with self.cached_session():
      elems = constant_op.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name="data")
      v = constant_op.constant(2.0, name="v")
      r = functional_ops.foldl(
          lambda a, x: math_ops.multiply(a, x), elems, initializer=v)
      r = gradients_impl.gradients(r, v)[0]
      self.assertAllEqual(720.0, self.evaluate(r))

      r = functional_ops.foldr(
          lambda a, x: math_ops.multiply(a, x), elems, initializer=v)
      r = gradients_impl.gradients(r, v)[0]
      self.assertAllEqual(720.0, self.evaluate(r))
  # pylint: enable=unnecessary-lambda

  @test_util.run_in_graph_and_eager_modes
  def testScan_Simple(self):
    # [FUNCTION DOC]
    # Test (graph and eager) of a cumulative product scan over one to six. Without an initializer expects
    # one, two, six, 24, 120, 720; with initializer 2 expects 2, 4, 12, 48, 240, 1440.
    elems = constant_op.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name="data")
    v = constant_op.constant(2.0, name="v")

    # pylint: disable=unnecessary-lambda
    r = functional_ops.scan(lambda a, x: math_ops.multiply(a, x), elems)
    self.assertAllEqual([1., 2., 6., 24., 120., 720.], self.evaluate(r))

    r = functional_ops.scan(
        lambda a, x: math_ops.multiply(a, x), elems, initializer=v)
    self.assertAllEqual([2., 4., 12., 48., 240., 1440.], self.evaluate(r))
    # pylint: enable=unnecessary-lambda

  @test_util.run_in_graph_and_eager_modes
  def testScan_Reverse(self):
    # [FUNCTION DOC]
    # Test (graph and eager) of the cumulative product scan run in reverse order. Without an initializer
    # expects 720, 720, 360, 120, 30, 6; with initializer 2 expects 1440, 1440, 720, 240, 60, 12.
    elems = constant_op.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name="data")
    v = constant_op.constant(2.0, name="v")

    # pylint: disable=unnecessary-lambda
    r = functional_ops.scan(lambda a, x: math_ops.multiply(a, x), elems,
                            reverse=True)
    self.assertAllEqual([720., 720., 360., 120., 30., 6.], self.evaluate(r))
    r = functional_ops.scan(
        lambda a, x: math_ops.multiply(a, x), elems, initializer=v,
        reverse=True)
    self.assertAllEqual([1440., 1440., 720., 240., 60., 12.],
                        self.evaluate(r))
    # pylint: enable=unnecessary-lambda

  @test_util.run_in_graph_and_eager_modes
  def testScan_SingleInputMultiOutput(self):
    # [FUNCTION DOC]
    # Test (graph and eager) of a scan with one input and a pair of accumulators (initial one and minus
    # one) that are multiplied by the element with alternating sign. Expects the factorial sequence for the
    # first output and the same sequence with alternating signs for the second.
    elems = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    initializer = (np.array(1.0), np.array(-1.0))
    r = functional_ops.scan(lambda a, x: (a[0] * x, -a[1] * x), elems,
                            initializer)
    r_value = self.evaluate(r)

    self.assertAllEqual([1.0, 2.0, 6.0, 24.0, 120.0, 720.0], r_value[0])
    self.assertAllEqual([1.0, -2.0, 6.0, -24.0, 120.0, -720.0], r_value[1])

  @test_util.run_in_graph_and_eager_modes
  def testScan_MultiInputSingleOutput(self):
    # [FUNCTION DOC]
    # Test (graph and eager) of a scan over a pair of inputs where each step multiplies the accumulator by
    # the sum of the two components, which is always one. Expects a vector of six ones.
    elems = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    initializer = np.array(1.0)
    # Multiply a * 1 each time
    r = functional_ops.scan(lambda a, x: a * (x[0] + x[1]),
                            (elems + 1, -elems), initializer)
    self.assertAllEqual([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], self.evaluate(r))

  @test_util.run_in_graph_and_eager_modes
  def testScan_MultiInputSameTypeOutput(self):
    # [FUNCTION DOC]
    # Test (graph and eager) of a scan with no initializer over a pair (vector, negated vector), where the
    # output is the running sum of each component. Expects the cumulative sums of each.
    elems = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    r = functional_ops.scan(lambda a, x: (a[0] + x[0], a[1] + x[1]),
                            (elems, -elems))
    r_value = self.evaluate(r)
    self.assertAllEqual(np.cumsum(elems), r_value[0])
    self.assertAllEqual(np.cumsum(-elems), r_value[1])

  @test_util.run_in_graph_and_eager_modes
  def testScan_MultiOutputMismatchedInitializer(self):
    # [FUNCTION DOC]
    # Test (graph and eager) of error handling: scan whose function returns a pair while the initializer
    # is a single scalar. Expects a value error complaining that the two structures do not have the same
    # nested structure.
    elems = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    initializer = np.array(1.0)
    # Multiply a * 1 each time
    with self.assertRaisesRegex(
        ValueError, "two structures don't have the same nested structure"):
      functional_ops.scan(lambda a, x: (a, -a), elems, initializer)

  @test_util.run_deprecated_v1
  def testScan_Scoped(self):
    # [FUNCTION DOC]
    # Test (deprecated version 1 graph mode) of a scan with the variable-creating scoped reducer. Checks
    # exactly one trainable variable root/body/two:0, expects 1, 6, 18, 44, 98, 208; after variable reuse
    # with initializer 2 expects 6, 16, 38, 84, 178, 368 and still one variable.
    with self.cached_session() as sess:
      with variable_scope.variable_scope("root") as varscope:
        elems = constant_op.constant([1, 2, 3, 4, 5, 6], name="data")

        r = functional_ops.scan(simple_scoped_fn, elems)
        # Check that we have the one variable we asked for here.
        self.assertEqual(len(variables.trainable_variables()), 1)
        self.assertEqual(variables.trainable_variables()[0].name,
                         "root/body/two:0")
        sess.run([variables.global_variables_initializer()])
        results = np.array([1, 6, 18, 44, 98, 208])
        self.assertAllEqual(results, self.evaluate(r))

        # Now let's reuse our single variable.
        varscope.reuse_variables()
        r = functional_ops.scan(simple_scoped_fn, elems, initializer=2)
        self.assertEqual(len(variables.trainable_variables()), 1)
        results = np.array([6, 16, 38, 84, 178, 368])
        self.assertAllEqual(results, self.evaluate(r))

  @test_util.run_in_graph_and_eager_modes
  def testScanFoldl_Nested(self):
    # [FUNCTION DOC]
    # Test (graph and eager) of a left fold nested inside a scan. The scan runs over four floats and for
    # each step folds over two halves, multiplying by the half and the outer element. Expects
    # approximately one, one, 2.25, 9 as explained by the comments in the test.
    elems = constant_op.constant([1.0, 2.0, 3.0, 4.0], name="data")
    inner_elems = constant_op.constant([0.5, 0.5], name="data")

    def r_inner(a, x):
      # [FUNCTION DOC]
      # HELPER: scan function for testScanFoldl_Nested. Takes accumulator a and outer element x and returns
      # a left fold over the captured inner elements with initializer a where each step multiplies the
      # running value by the inner element and by x.
      return functional_ops.foldl(
          lambda b, y: b * y * x, inner_elems, initializer=a)

    r = functional_ops.scan(r_inner, elems)

    # t == 0 (returns 1)
    # t == 1, a == 1, x == 2 (returns 1)
    #   t_0 == 0, b == a == 1, y == 0.5, returns b * y * x = 1
    #   t_1 == 1, b == 1,      y == 0.5, returns b * y * x = 1
    # t == 2, a == 1, x == 3 (returns 1.5*1.5 == 2.25)
    #   t_0 == 0, b == a == 1, y == 0.5, returns b * y * x = 1.5
    #   t_1 == 1, b == 1.5,    y == 0.5, returns b * y * x = 1.5*1.5
    # t == 3, a == 2.25, x == 4 (returns 9)
    #   t_0 == 0, b == a == 2.25, y == 0.5, returns b * y * x = 4.5
    #   t_1 == 1, b == 4.5,       y == 0.5, returns b * y * x = 9
    self.assertAllClose([1., 1., 2.25, 9.], self.evaluate(r))

  @test_util.run_deprecated_v1
  def testScan_Control(self):
    # [FUNCTION DOC]
    # Test (deprecated version 1 graph mode) that scan builds correctly under a control dependency on a
    # boolean placeholder. Runs a scan multiplying running values over a placeholder vector fed one, three,
    # three, feeding the boolean true, and expects one, three, nine.
    with self.cached_session() as sess:
      s = array_ops.placeholder(dtypes.float32, shape=[None])
      b = array_ops.placeholder(dtypes.bool)

      with ops.control_dependencies([b]):
        c = functional_ops.scan(lambda a, x: x * a, s)
      self.assertAllClose(
          np.array([1.0, 3.0, 9.0]), sess.run(c, {s: [1, 3, 3],
                                                  b: True}))

  @test_util.run_deprecated_v1
  def testScan_Grad(self):
    # [FUNCTION DOC]
    # Test (deprecated version 1 graph mode) of the gradient of a product scan with respect to its
    # initializer 2 over one to six; the gradient of the scan output (all elements) is expected to be 873.
    with self.cached_session():
      elems = constant_op.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name="data")
      v = constant_op.constant(2.0, name="v")

      # pylint: disable=unnecessary-lambda
      r = functional_ops.scan(
          lambda a, x: math_ops.multiply(a, x), elems, initializer=v)
      # pylint: enable=unnecessary-lambda
      r = gradients_impl.gradients(r, v)[0]
      self.assertAllEqual(873.0, self.evaluate(r))

  @test_util.run_deprecated_v1
  def testScanGradientWithPartStopGradient(self):
    # [FUNCTION DOC]
    # Test (deprecated version 1 graph mode) that gradients can be computed through a scan with two
    # variable-valued outputs when one output goes through a stop-gradient. Builds the scan over five
    # zeros with a pair initializer, sums the first output and the stopped second output, differentiates
    # with respect to both variables, and just evaluates the gradients after initialization. No value is
    # asserted; it passes if nothing raises.
    a = variables.Variable(0.0, name="a")
    b = variables.Variable(0.0, name="b")
    elems = array_ops.zeros(5)
    l0, l1 = functional_ops.scan(
        lambda elem_, input_: (a, b), elems, initializer=(0., 0.))
    loss = l0 + array_ops.stop_gradient(l1)
    grad = gradients_impl.gradients(ys=[loss], xs=[a, b])
    with self.test_session():
      self.evaluate(variables.global_variables_initializer())
      self.evaluate(grad)

  @test_util.run_in_graph_and_eager_modes
  def testFoldShape(self):
    # [FUNCTION DOC]
    # Test (graph and eager) that the static shape of a left fold result matches the evaluated shape. Folds
    # over a two by three matrix with a function returning the current row and a zeros initializer of
    # length three; asserts the static shape equals the actual shape.
    x = constant_op.constant([[1, 2, 3], [4, 5, 6]])

    def fn(_, current_input):
      # [FUNCTION DOC]
      # HELPER: fold function for testFoldShape. Ignores the accumulator and returns the current input.
      return current_input

    initializer = constant_op.constant([0, 0, 0])
    y = functional_ops.foldl(fn, x, initializer=initializer)
    self.assertAllEqual(y.get_shape(), self.evaluate(y).shape)

  @test_util.run_in_graph_and_eager_modes
  def testScanShape(self):
    # [FUNCTION DOC]
    # Test (graph and eager) that the static shape of a scan result equals the evaluated shape. Scans over
    # a two by three matrix with a function returning the current row, with a zeros initializer.
    x = constant_op.constant([[1, 2, 3], [4, 5, 6]])

    def fn(_, current_input):
      # [FUNCTION DOC]
      # HELPER: scan function for testScanShape. Ignores the accumulator and returns the current input.
      return current_input

    initializer = constant_op.constant([0, 0, 0])
    y = functional_ops.scan(fn, x, initializer=initializer)
    self.assertAllEqual(y.get_shape(), self.evaluate(y).shape)

  # TODO(akshayka): this test fails in eager: the iterable is of length 0 so
  # so the body of the while loop never executes
  @test_util.run_deprecated_v1
  def testScanEmptyTensor(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the TODO above notes this test fails in eager mode (the body never runs for an empty
    # input), so it is restricted to deprecated version 1 graph mode.
    # Test of scanning over an empty range with an initializer of shape two by four. Expects the static
    # result shape to be zero by two by four and equal to the evaluated shape.
    with self.cached_session():
      x = functional_ops.scan(
          lambda x, _: x, math_ops.range(0), initializer=array_ops.ones([2, 4]))
      self.assertAllEqual([0, 2, 4], x.get_shape())
      self.assertAllEqual(x.get_shape(), self.evaluate(x).shape)

  @test_util.run_deprecated_v1
  def testScanUnknownShape(self):
    # [FUNCTION DOC]
    # Test (deprecated version 1 graph mode) that scanning a placeholder of unknown shape with an unknown
    # shaped initializer yields a result with fully unknown static shape (no dimensions information).
    x = array_ops.placeholder(dtypes.float32)
    initializer = array_ops.placeholder(dtypes.float32)

    def fn(_, current_input):
      # [FUNCTION DOC]
      # HELPER: scan function for testScanUnknownShape. Ignores the accumulator and returns the input.
      return current_input

    y = functional_ops.scan(fn, x, initializer=initializer)
    self.assertIs(None, y.get_shape().dims)

  @test_util.run_deprecated_v1
  def testScanVaryingShape(self):
    # [FUNCTION DOC]
    # Test (deprecated version 1 graph mode) that scanning works when shapes vary. Scans a placeholder
    # with unknown first dimension and its transpose (with shape inference disabled), computes gradients
    # of both with respect to their inputs, and runs all four with a fed two-element row as a smoke test.
    # No values are asserted.
    with self.cached_session() as sess:
      x = array_ops.placeholder(dtype=dtypes.float32, shape=[None, 2])
      x_t = array_ops.transpose(x)
      # scan over dimension 0 (with shape None)
      result = functional_ops.scan(lambda a, x: a + x, x)
      # scanned over transposed dimension 0 (with shape 2)
      result_t = functional_ops.scan(lambda a, x: a + x, x_t, infer_shape=False)
      # ensure gradients can be calculated
      result_grad = gradients_impl.gradients(result, [x])[0]
      result_t_grad = gradients_impl.gradients(result_t, [x_t])[0]

      # smoke test to ensure they all evaluate
      sess.run([result, result_t, result_grad, result_t_grad],
               feed_dict={x: [[1.0, 2.0]]})

  @test_util.run_deprecated_v1
  def testRemoteFunction(self):
    # [FUNCTION DOC]
    # Test (deprecated version 1 graph mode) of a remote call across devices in a local cluster. Starts a
    # one-worker one-parameter-server cluster with two CPU devices per worker, defines a multiply
    # function, places two variables (2 and 3) on the parameter server and calls the function remotely on
    # the second worker CPU. Expects the result to be a list containing 6.
    worker_config = config_pb2.ConfigProto()
    worker_config.device_count["CPU"] = 2
    worker, _ = test_util.create_local_cluster(
        1, 1, worker_config=worker_config)

    @function.Defun(dtypes.int32, dtypes.int32)
    def _remote_fn(a, b):
      # [FUNCTION DOC]
      # HELPER: legacy defined function used as the remote call target. Takes integers a and b and returns
      # their product.
      return math_ops.multiply(a, b)

    with ops.device("/job:ps/task:0"):
      a = variables.Variable(2, dtype=dtypes.int32)
      b = variables.Variable(3, dtype=dtypes.int32)

    with ops.device("/job:worker/replica:0/task:0/cpu:0"):
      remote_op = functional_ops.remote_call(
          args=[a, b],
          Tout=[dtypes.int32],
          f=_remote_fn,
          target="/job:worker/replica:0/task:0/cpu:1")

    with session.Session(worker[0].target) as sess:
      self.evaluate(variables.global_variables_initializer())
      mul = self.evaluate(remote_op)
      self.assertEqual(mul, [6])

  @test_util.run_deprecated_v1
  def testRemoteFunctionDirectSession(self):
    # [FUNCTION DOC]
    # Test (deprecated version 1 graph mode) of remote call between two CPU devices on localhost within a
    # direct session configured with two CPUs. Variables 2 and 3 on cpu 0 are multiplied by the remote
    # function targeted at cpu 1; expects a list containing 6.
    worker_config = config_pb2.ConfigProto()
    worker_config.device_count["CPU"] = 2

    @function.Defun(dtypes.int32, dtypes.int32)
    def _remote_fn(a, b):
      # [FUNCTION DOC]
      # HELPER: legacy defined function used as the remote call target. Takes integers a and b and returns
      # their product.
      return math_ops.multiply(a, b)

    with ops.device("/job:localhost/replica:0/task:0/cpu:0"):
      a = variables.Variable(2, dtype=dtypes.int32)
      b = variables.Variable(3, dtype=dtypes.int32)

    with ops.device("/job:localhost/replica:0/task:0/cpu:0"):
      remote_op = functional_ops.remote_call(
          args=[a, b],
          Tout=[dtypes.int32],
          f=_remote_fn,
          target="/job:localhost/replica:0/task:0/cpu:1")

    with self.test_session(config=worker_config) as sess:
      self.evaluate(variables.global_variables_initializer())
      mul = self.evaluate(remote_op)
      self.assertEqual(mul, [6])

  @test_util.run_deprecated_v1
  def testRemoteFunctionSameDeviceDirectSession(self):
    # [FUNCTION DOC]
    # Test (deprecated version 1 graph mode) of a remote call whose target is the same device (cpu 0) as
    # the caller in a cached session. Variables 2 and 3 are multiplied; expects a list containing 6.

    @function.Defun(dtypes.int32, dtypes.int32)
    def _remote_fn(a, b):
      # [FUNCTION DOC]
      # HELPER: legacy defined function used as the remote call target. Takes integers a and b and returns
      # their product.
      return math_ops.multiply(a, b)

    with ops.device("/cpu:0"):
      a = variables.Variable(2, dtype=dtypes.int32)
      b = variables.Variable(3, dtype=dtypes.int32)

    with ops.device("/cpu:0"):
      remote_op = functional_ops.remote_call(
          args=[a, b], Tout=[dtypes.int32], f=_remote_fn, target="/cpu:0")

    with self.cached_session() as sess:
      self.evaluate(variables.global_variables_initializer())
      mul = self.evaluate(remote_op)
      self.assertEqual(mul, [6])

  @test_util.run_deprecated_v1
  def testRemoteFunctionCPUGPU(self):
    # [FUNCTION DOC]
    # Test (deprecated version 1 graph mode) of a remote call from a CPU to a GPU on the same host.
    # Skipped when no GPU is available. Multiplies floats 2 and 3 on the GPU and adds 3.0 to the result;
    # expects 9.0. Note the unused session variable in the final block.
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    @function.Defun(dtypes.float32, dtypes.float32)
    def _remote_fn(a, b):
      # [FUNCTION DOC]
      # HELPER: legacy defined function used as the remote call target. Takes floats a and b and returns
      # their product.
      return math_ops.multiply(a, b)

    with ops.device("/job:localhost/replica:0/task:0/cpu:0"):
      a = variables.Variable(2, dtype=dtypes.float32)
      b = variables.Variable(3, dtype=dtypes.float32)

    with ops.device("/job:localhost/replica:0/task:0/cpu:0"):
      remote_op = functional_ops.remote_call(
          args=[a, b],
          Tout=[dtypes.float32],
          f=_remote_fn,
          target="/job:localhost/replica:0/task:0/device:GPU:0")[0] + 3.0

    with self.cached_session() as sess:
      self.evaluate(variables.global_variables_initializer())
      mul = self.evaluate(remote_op)
      self.assertEqual(mul, 9.0)

  @test_util.run_deprecated_v1
  def testRemoteFunctionGPUCPU(self):
    # [FUNCTION DOC]
    # Test (deprecated version 1 graph mode) of a remote call from a GPU to the CPU. Skipped when no GPU
    # is available. Variables 2 and 3 live on the GPU, product computed targeting cpu 0 and 3.0 added;
    # expects 9.0.
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    @function.Defun(dtypes.float32, dtypes.float32)
    def _remote_fn(a, b):
      # [FUNCTION DOC]
      # HELPER: legacy defined function used as the remote call target. Takes floats a and b and returns
      # their product.
      return math_ops.multiply(a, b)

    with ops.device("/job:localhost/replica:0/task:0/device:GPU:0"):
      a = variables.Variable(2, dtype=dtypes.float32)
      b = variables.Variable(3, dtype=dtypes.float32)

    with ops.device("/job:localhost/replica:0/task:0/device:GPU:0"):
      remote_op = functional_ops.remote_call(
          args=[a, b],
          Tout=[dtypes.float32],
          f=_remote_fn,
          target="/job:localhost/replica:0/task:0/cpu:0")[0] + 3.0

    with self.cached_session() as sess:
      self.evaluate(variables.global_variables_initializer())
      mul = self.evaluate(remote_op)
      self.assertEqual(mul, 9.0)

  @test_util.run_deprecated_v1
  def testRemoteFunctionGPUCPUStrings(self):
    # [FUNCTION DOC]
    # Test (deprecated version 1 graph mode) that a string tensor can be passed from GPU placement to a
    # remote function on the CPU. Skipped when no GPU is available. The function is an identity; expects
    # the single result to be the byte string "a".
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    @function.Defun(dtypes.string)
    def _remote_fn(inp):
      # [FUNCTION DOC]
      # HELPER: legacy defined function used as the remote call target; takes a string tensor and returns it
      # unchanged via an identity op.
      return array_ops.identity(inp)

    a = array_ops.constant("a")

    with ops.device("/gpu:0"):
      remote_op = functional_ops.remote_call(
          args=[a], Tout=[dtypes.string], f=_remote_fn, target="/cpu:0")

    with self.cached_session() as sess:
      ret = self.evaluate(remote_op)
      self.assertAllEqual(ret, [b"a"])

  @test_util.run_deprecated_v1
  def testRemoteFunctionCrossProcess(self):
    # [FUNCTION DOC]
    # Test (deprecated version 1 graph mode) of a remote call across processes. Creates a local cluster
    # with two workers and one parameter server, places variables 2 and 3 on the parameter server, runs
    # the multiply on worker task 1 invoked from worker task 0, adds 3.0 and expects 9.
    workers, _ = test_util.create_local_cluster(2, 1)

    @function.Defun(dtypes.float32, dtypes.float32)
    def _remote_fn(a, b):
      # [FUNCTION DOC]
      # HELPER: legacy defined function used as the remote call target. Takes floats a and b and returns
      # their product.
      return math_ops.multiply(a, b)

    with ops.device("/job:ps/task:0"):
      a = variables.Variable(2, dtype=dtypes.float32)
      b = variables.Variable(3, dtype=dtypes.float32)

    with ops.device("/job:worker/replica:0/task:0/cpu:0"):
      remote_op = functional_ops.remote_call(
          args=[a, b],
          Tout=[dtypes.float32],
          f=_remote_fn,
          target="/job:worker/replica:0/task:1/cpu:0")[0] + 3.0

    with session.Session(workers[0].target) as sess:
      self.evaluate(variables.global_variables_initializer())
      mul = self.evaluate(remote_op)
      self.assertEqual(mul, 9)

  @test_util.run_v2_only
  def testRemoteFunctionCancellation(self):
    # [FUNCTION DOC]
    # Test (version 2 only) that a remote call blocked in a collective can be cancelled. Resets the eager
    # context and splits the physical CPU into two logical devices. Runs, on an async executor with a
    # cancellation manager, a compiled function that remote-calls a function performing two all-reduce
    # collectives (group size 2); the second never completes. A separate local all-reduce completes the
    # first instance, then the cancellation is started, and waiting on the executor must raise a
    # cancelled error. Side effects: resets global eager context and reconfigures logical devices.
    context._reset_context()
    logical_devices = []
    logical_devices.append(context.LogicalDeviceConfiguration())
    logical_devices.append(context.LogicalDeviceConfiguration())
    framework_config.set_logical_device_configuration(
        framework_config.list_physical_devices("CPU")[0], logical_devices)

    @function.Defun(dtypes.float32)
    def _remote_fn(v):
      # [FUNCTION DOC]
      # HELPER: legacy defined function used as the remote call target in the cancellation test. Takes a
      # float tensor, runs an all-reduce with group size 2 as an anchor, then under a control dependency on it
      # runs a second all-reduce with a different instance key and returns that result. The second collective
      # is expected never to finish.
      # We run two collectives here to make sure we cancel in the middle of the
      # RemoteCall. The second one should never finish.
      anchor = collective_ops.all_reduce_v2(
          v, group_size=2, group_key=1, instance_key=1)
      with ops.control_dependencies([anchor]):
        return collective_ops.all_reduce_v2(
            v, group_size=2, group_key=1, instance_key=2)

    @eager_def_function.function
    def run():
      # [FUNCTION DOC]
      # HELPER: compiled function for testRemoteFunctionCancellation. On cpu 0 it remote-calls the
      # collective function targeted at cpu 1 with a constant one-element vector plus the captured inputs and
      # returns the first result.
      with ops.device("/cpu:0"):
        return functional_ops.remote_call(
            args=[constant_op.constant([1.])] + _remote_fn.captured_inputs,
            Tout=[dtypes.float32],
            f=_remote_fn,
            target="/cpu:1")[0]

    async_executor = executor.new_executor(enable_async=True)
    cancel_mgr = cancellation.CancellationManager()
    with context.executor_scope(async_executor):
      # This should never finish.
      cancel_mgr.get_cancelable_function(run.get_concrete_function())()
    with ops.device("/cpu:0"):
      collective_ops.all_reduce_v2([1.],
                                   group_size=2,
                                   group_key=1,
                                   instance_key=1)
    cancel_mgr.start_cancel()
    with self.assertRaises(errors.CancelledError):
      async_executor.wait()

  @test_util.run_deprecated_v1
  def testIf(self):
    # [FUNCTION DOC]
    # Test (deprecated version 1 graph mode) of the legacy If op. Defines two functions: doubling and
    # triple-plus-one. With a float placeholder x, If chooses doubling when x is greater than zero. Expects
    # 18 for 9, minus 23 for minus 8 and 1 for 0 (the else branch is used for zero).

    @function.Defun(dtypes.float32)
    def Twice(x):
      # [FUNCTION DOC]
      # HELPER: then-branch for testIf. Returns the float input multiplied by 2.
      return x * 2

    @function.Defun(dtypes.float32)
    def Thrice(x):
      # [FUNCTION DOC]
      # HELPER: else-branch for testIf. Returns the float input multiplied by 3 plus 1.
      return x * 3 + 1

    with self.test_session(use_gpu=False) as sess:

      x = array_ops.placeholder(dtypes.float32)
      ret = functional_ops.If(math_ops.greater(x, 0), [x], Twice, Thrice)[0]

      self.assertAllEqual(sess.run(ret, feed_dict={x: 9.}), 18.)
      self.assertAllEqual(sess.run(ret, feed_dict={x: -8.}), -23.)
      self.assertAllEqual(sess.run(ret, feed_dict={x: 0.}), 1.)

  def testWhile(self):
    # [FUNCTION DOC]
    # Test of the legacy While op in a fresh graph for both GPU settings. Loop condition is n greater
    # than zero; the body decrements n and adds n to a running sum. Running with n=20 expects 210 and
    # n=100 expects 5050.

    for use_gpu in (True, False):
      with ops.Graph().as_default() as g:

        @function.Defun(*[dtypes.float32] * 2)
        def Cond(n, unused_x):
          # [FUNCTION DOC]
          # HELPER: loop condition for the While tests. Takes the float counter and an unused value and returns
          # whether the counter is greater than zero.
          return n > 0

        @function.Defun(*[dtypes.float32] * 2)
        def Body(n, x):
          # [FUNCTION DOC]
          # HELPER: loop body for the While tests. Takes the counter n and running sum x and returns n minus one
          # and x plus n.
          return n - 1, x + n

        def Run(sess, n):
          # [FUNCTION DOC]
          # HELPER: runs the While loop in the given session. Takes a session and the starting n, runs the loop
          # with an initial sum of zero and returns the final sum (second loop output).
          return sess.run(functional_ops.While([n, 0.], Cond, Body))[1]

        with self.session(graph=g, use_gpu=use_gpu) as sess:
          self.assertAllEqual(Run(sess, 20.), 210.)
          self.assertAllEqual(Run(sess, 100.), 5050.)

  def testToBool(self):
    # [FUNCTION DOC]
    # Test of the to-bool op truthiness rules. Scalars are false only when zero or false; strings are false
    # only when empty; tensors with rank above zero are true exactly when they have at least one element
    # (so empty nested lists are false and a list containing zero or false is true).
    # For 0D tensors, the truthiness depends on whether the value is "zero".
    self.assertAllEqual(gen_functional_ops.to_bool(0), False)
    self.assertAllEqual(gen_functional_ops.to_bool(1), True)
    self.assertAllEqual(gen_functional_ops.to_bool(42), True)
    self.assertAllEqual(gen_functional_ops.to_bool(0.), False)
    self.assertAllEqual(gen_functional_ops.to_bool(1.), True)
    self.assertAllEqual(gen_functional_ops.to_bool(42.), True)
    self.assertAllEqual(gen_functional_ops.to_bool(False), False)
    self.assertAllEqual(gen_functional_ops.to_bool(True), True)
    # For strings, "zero" is the empty string.
    self.assertAllEqual(gen_functional_ops.to_bool(""), False)
    self.assertAllEqual(gen_functional_ops.to_bool("a"), True)

    # For >0D tensors, the truthiness only depends on whether there are
    # elements or not.
    self.assertAllEqual(gen_functional_ops.to_bool([]), False)
    self.assertAllEqual(gen_functional_ops.to_bool([[]]), False)
    self.assertAllEqual(gen_functional_ops.to_bool([[[]]]), False)
    self.assertAllEqual(gen_functional_ops.to_bool([0]), True)
    self.assertAllEqual(gen_functional_ops.to_bool([1]), True)
    self.assertAllEqual(gen_functional_ops.to_bool([[0]]), True)
    self.assertAllEqual(gen_functional_ops.to_bool([False]), True)
    self.assertAllEqual(gen_functional_ops.to_bool([True]), True)

  # Like above, but using int32 in order to ensure that int32 tensors don't get
  # copied to the GPU during the application of the while.
  def testWhileInt32(self):
    # [FUNCTION DOC]
    # Test of the legacy While op with 32-bit integer state, to ensure integer tensors are not copied to the
    # GPU while applying the loop. Condition n greater than zero, body decrements and accumulates. Expects
    # 210 for 20 and 5050 for 100 in a GPU-enabled session.
    with ops.Graph().as_default() as g:

      @function.Defun(*[dtypes.int32] * 2)
      def Cond(n, unused_x):
        # [FUNCTION DOC]
        # HELPER: integer loop condition for testWhileInt32. Returns whether the counter is greater than zero.
        return n > 0

      @function.Defun(*[dtypes.int32] * 2)
      def Body(n, x):
        # [FUNCTION DOC]
        # HELPER: integer loop body for testWhileInt32. Returns n minus one and x plus n.
        return n - 1, x + n

      def Run(sess, n):
        # [FUNCTION DOC]
        # HELPER: runs the integer While loop in the given session from starting value n with initial sum
        # zero and returns the final sum.
        return sess.run(functional_ops.While([n, 0], Cond, Body))[1]

      with self.session(graph=g, use_gpu=True) as sess:
        self.assertAllEqual(Run(sess, 20), 210)
        self.assertAllEqual(Run(sess, 100), 5050)

  @test_util.run_deprecated_v1
  def testWhileLowering(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: in the inner Run helper the return statement is inside the loop over the GPU setting, so
    # only the first setting (GPU enabled) is ever executed and the CPU-only iteration is dead code.
    # Test (deprecated version 1 graph mode) that a While op marked for lowering to switch and merge
    # gives the same results whether the output is fetched by tensor or by name "my_while:1". Expects 210
    # for n=20 and 5050 for n=100 in both fetch styles.

    def Run(n, fetch_by_name):
      # [FUNCTION DOC]
      # HELPER: builds the lowered While graph and evaluates it. Takes the start value n and a boolean
      # fetch_by_name. Creates a While op named my_while, sets the lowering attribute on its op, and returns
      # the evaluated second output, fetched by name when requested. Returns on the first loop iteration.
      for use_gpu in (True, False):
        with ops.Graph().as_default() as g:

          @function.Defun(*[dtypes.float32] * 2)
          def Cond(n, unused_x):
            # [FUNCTION DOC]
            # HELPER: loop condition; returns whether the counter is greater than zero.
            return n > 0

          @function.Defun(*[dtypes.float32] * 2)
          def Body(n, x):
            # [FUNCTION DOC]
            # HELPER: loop body; returns n minus one and x plus n.
            return n - 1, x + n

          # outputs: [0, n*(n+1)/2]
          outputs = functional_ops.While([n, 0.], Cond, Body, name="my_while")

          # `outputs` is the list of output tensors of the While op. We
          # arbitrarily choose the 0th tensor to get the While op and set the
          # lowering attribute on it.
          outputs[0].op._set_attr("_lower_using_switch_merge",
                                  attr_value_pb2.AttrValue(b=True))
          if not fetch_by_name:
            fetch = outputs[1]
          else:
            fetch = "my_while:1"
        with self.session(graph=g, use_gpu=use_gpu) as sess:
          return self.evaluate(fetch)

    self.assertAllEqual(Run(20., False), 210.)
    self.assertAllEqual(Run(20., True), 210.)
    self.assertAllEqual(Run(100., False), 5050.)
    self.assertAllEqual(Run(100., True), 5050.)

  @test_util.run_v1_only("b/120545219")
  @test_util.disable_xla("b/123337890")  # Different error message
  def testWhileError(self):
    # [FUNCTION DOC]
    # Test (version 1 only, XLA disabled because of a different error message) of While argument
    # validation. Uses a condition returning two values and a body returning three. Evaluating the former
    # must raise an invalid-argument error about expecting a single scalar but getting two tensors; the
    # latter an invalid-argument error that the loop body returned three arguments while two expected.
    for use_gpu in (True, False):
      with ops.Graph().as_default() as g:

        @function.Defun(*[dtypes.float32] * 2)
        def Cond(n, unused_x):
          # [FUNCTION DOC]
          # HELPER: valid loop condition; returns whether n is greater than zero.
          return n > 0

        @function.Defun(*[dtypes.float32] * 2)
        def CondReturnsTooManyArgs(n, x):
          # [FUNCTION DOC]
          # HELPER: deliberately invalid loop condition for testWhileError; returns a pair (boolean and x)
          # instead of a single scalar.
          return n > 0, x

        @function.Defun(*[dtypes.float32] * 2)
        def Body(n, x):
          # [FUNCTION DOC]
          # HELPER: valid loop body; returns n minus one and x plus n.
          return n - 1, x + n

        @function.Defun(*[dtypes.float32] * 2)
        def BodyReturnsTooManyArgs(n, x):
          # [FUNCTION DOC]
          # HELPER: deliberately invalid loop body for testWhileError; returns three values instead of two.
          return n - 1, x + n, x

        with self.session(graph=g, use_gpu=use_gpu):
          with self.assertRaisesRegex(
              errors.InvalidArgumentError,
              "Expected a single scalar.*got 2 tensors."):
            functional_ops.While([5., 0.], CondReturnsTooManyArgs,
                                 Body)[0].eval()
          with self.assertRaisesRegex(
              errors.InvalidArgumentError,
              "While loop body returned 3 arguments. Expected: 2"):
            functional_ops.While([5., 0.], Cond,
                                 BodyReturnsTooManyArgs)[0].eval()

  def testWhileInMultipleSubgraphs(self):
    # [FUNCTION DOC]
    # Test of the legacy While op when different subgraphs are run in the same session. Fetches the loop
    # result with a fed start value (210 for 20, 5050 for 100) and then together with an unrelated
    # constant, checking the result is the same.

    for use_gpu in (True, False):
      with ops.Graph().as_default() as g:

        @function.Defun(*[dtypes.float32] * 2)
        def Cond(n, x):  # pylint: disable=unused-argument
          # [FUNCTION DOC]
          # HELPER: loop condition; returns whether n is greater than zero (the second value is unused).
          return n > 0

        @function.Defun(*[dtypes.float32] * 2)
        def Body(n, x):
          # [FUNCTION DOC]
          # HELPER: loop body; returns n minus one and x plus n.
          return n - 1, x + n

        with self.session(graph=g, use_gpu=use_gpu) as sess:
          n = array_ops.placeholder(dtypes.float32)
          _, result = functional_ops.While([n, 0.], Cond, Body)
          c = constant_op.constant(37.)

          self.assertAllEqual(210., sess.run(result, feed_dict={n: 20.}))
          self.assertAllEqual(5050., sess.run(result, feed_dict={n: 100.}))
          # Test that the result is the same when we run a different subgraph.
          self.assertAllEqual(5050.,
                              sess.run([result, c], feed_dict={n: 100.})[0])

  # pylint: disable=cell-var-from-loop
  def testWhileCapturedInputs(self):
    # [FUNCTION DOC]
    # Test of While loops whose body captures an outer variable. Builds unary and binary-state loops that
    # add the captured variable, with condition n less than ten, starting from one. Expects the unary
    # result 10 and 54 and the binary 10, 54 and 9. Also expects a value error when the condition function
    # itself captures the variable.
    for use_gpu in (True, False):
      with ops.Graph().as_default() as g:
        v = variables.Variable(1.0)

        def TestCond(n, *args):
          # [FUNCTION DOC]
          # HELPER: loop condition used by testWhileCapturedInputs. Ignores extra state and returns whether n is
          # less than ten.
          del args
          return n < 10

        @function.Defun(*[dtypes.float32] * 2)
        def TestUnary(n, x):
          # [FUNCTION DOC]
          # HELPER: body with two loop values; returns n plus one and x plus n plus the captured variable.
          return math_ops.add(n, 1), x + n + v

        @function.Defun(*[dtypes.float32] * 3)
        def TestBinary(n, x, x2):
          # [FUNCTION DOC]
          # HELPER: body with three loop values; returns n plus one, x plus n plus the captured variable and x2
          # plus the captured variable.
          return math_ops.add(n, 1), x + n + v, x2 + v

        with self.session(graph=g, use_gpu=use_gpu) as sess:
          result_unary = functional_ops.While(
              [1.0, 0.],
              function.Defun(*[dtypes.float32] * 2)(TestCond), TestUnary)
          result_binary = functional_ops.While(
              [1.0, 0., 0.],
              function.Defun(*[dtypes.float32] * 3)(TestCond), TestBinary)
          self.evaluate(variables.global_variables_initializer())
          assert len(result_unary) == 2
          self.assertEqual([10.0, 54.0], self.evaluate(result_unary))
          assert len(result_binary) == 3
          self.assertEqual([10.0, 54.0, 9.0], self.evaluate(result_binary))

          def TestCondCapture(n, *args):
            # [FUNCTION DOC]
            # HELPER: invalid condition for testWhileCapturedInputs that captures the outer variable; casts n to
            # float, adds the variable and returns whether the sum is less than ten. Using it must raise a value
            # error.
            del args
            return math_ops.cast(n, dtypes.float32) + v < 10

          with self.assertRaises(ValueError):
            _ = functional_ops.While(
                [1],
                function.Defun(dtypes.int32)(TestCondCapture),
                function.Defun(dtypes.int32, dtypes.float32)(TestUnary))

  # pylint: enable=cell-var-from-loop

  def _tfSum(self, use_gpu, rewrite_with_while):
    # [FUNCTION DOC]
    # HELPER: shared driver for testFor and testForWithWhile. Takes use_gpu and rewrite_with_while flags.
    # In a fresh graph and session builds two For loops with a body adding the loop index to a float
    # accumulator: one ascending from one to twenty (sum 210) and one descending from 100 to one (sum 5050),
    # and asserts those results.
    with ops.Graph().as_default() as g:
      with self.session(graph=g, use_gpu=use_gpu) as sess:

        @function.Defun(dtypes.int32, dtypes.float32)
        def Body(n, x):
          # [FUNCTION DOC]
          # HELPER: loop body for _tfSum; adds the integer index cast to float to the float accumulator.
          return x + math_ops.cast(n, dtypes.float32)

        xs = [
            # 1 + 2  + ... + 20
            functional_ops.For(
                1, 21, 1, [0.], Body, rewrite_with_while=rewrite_with_while)[0],
            # 100 + 99 + ... + 1
            functional_ops.For(
                100, 0, -1, [0.], Body, rewrite_with_while=rewrite_with_while)
            [0],
        ]
        xvals = self.evaluate(xs)
      self.assertAllEqual(210, xvals[0])
      self.assertAllEqual(5050, xvals[1])

  def testFor(self):
    # [FUNCTION DOC]
    # Test of the legacy For op without rewriting to a while loop, run for both GPU settings through the
    # summation helper.
    for use_gpu in (True, False):
      self._tfSum(use_gpu, False)

  def testForWithWhile(self):
    # [FUNCTION DOC]
    # Test of the legacy For op rewritten to a while loop, run for both GPU settings through the summation
    # helper.
    for use_gpu in (True, False):
      self._tfSum(use_gpu, True)

  def testForWithWhileNaming(self):
    # [FUNCTION DOC]
    # Test that rewriting For to a while loop keeps names of generated functions derived from the body
    # function name. Defines a body with an explicit function name TestBody, builds the For with rewrite
    # enabled and expects the graph function library to contain TestBody, TestBody_Cond and TestBody_Body.
    g = ops.Graph()
    with g.as_default():

      @function.Defun(dtypes.int32, dtypes.float32, func_name="TestBody")
      def TestBody(n, x):
        # [FUNCTION DOC]
        # HELPER: body function for testForWithWhileNaming; adds the index cast to float to the accumulator.
        return x + math_ops.cast(n, dtypes.float32)

      _ = functional_ops.For(
          1, 21, 1, [0.], TestBody, rewrite_with_while=True)[0]

    names = []
    for func in g.as_graph_def().library.function:
      names.append(func.signature.name)
    self.assertTrue("TestBody" in names)
    self.assertTrue("TestBody_Cond" in names)
    self.assertTrue("TestBody_Body" in names)

  @test_util.run_deprecated_v1
  def testForCapturedInputs(self):
    # [FUNCTION DOC]
    # Test (deprecated version 1 graph mode) of For loops whose bodies capture a variable, both with and
    # without rewriting to a while loop. Builds nullary, unary and binary state variants from one to ten.
    # The nullary version returns nothing and is run by fetching the op by name; the unary expects 54 and
    # the binary 54 and 9.
    v = variables.Variable(1.0)

    @function.Defun(dtypes.int32)
    def TestNullary(n):
      # [FUNCTION DOC]
      # HELPER: nullary-state body; computes the captured variable plus the index but returns nothing.
      v + math_ops.cast(n, dtypes.float32)  # pylint: disable=expression-not-assigned

    @function.Defun(dtypes.int32, dtypes.float32)
    def TestUnary(n, x):
      # [FUNCTION DOC]
      # HELPER: unary-state body; returns x plus the index plus the captured variable.
      return x + math_ops.cast(n, dtypes.float32) + v

    @function.Defun(dtypes.int32, dtypes.float32, dtypes.float32)
    def TestBinary(n, x, x2):
      # [FUNCTION DOC]
      # HELPER: binary-state body; returns x plus index plus the captured variable, and x2 plus the variable.
      return x + math_ops.cast(n, dtypes.float32) + v, x2 + v

    for rewrite_with_while in (True, False):
      use_gpu = not rewrite_with_while
      with self.test_session(use_gpu=use_gpu) as sess:
        result_nullary = functional_ops.For(
            1, 10, 1, [], TestNullary,
            rewrite_with_while=rewrite_with_while)
        result_unary = functional_ops.For(
            1, 10, 1, [0.], TestUnary,
            rewrite_with_while=rewrite_with_while)
        result_binary = functional_ops.For(
            1, 10, 1, [0., 0.], TestBinary,
            rewrite_with_while=rewrite_with_while)
        self.evaluate(variables.global_variables_initializer())
        assert not result_nullary
        # The nullary variant doesn't return anything so we can't easily run it.
        # As a total hack, fetch the operation by name and run it.
        sess.run(ops.get_default_graph().get_operation_by_name(
            "While" if rewrite_with_while else "For"))
        assert len(result_unary) == 1
        self.assertEqual([54.0], self.evaluate(result_unary))
        assert len(result_binary) == 2
        self.assertEqual([54.0, 9.0], self.evaluate(result_binary))

  def _tfMLP(self, xval, wsval, bsval, rewrite_with_while):
    # [FUNCTION DOC]
    # HELPER: runs a multi-layer perceptron with the For op for testForMLP. Takes input, stacked weights and
    # biases, and a rewrite flag (GPU is used only when not rewriting). Applies a layer per iteration and
    # returns the evaluated final activation.
    # On GPU, don't rewrite using a while loop.
    use_gpu = not rewrite_with_while
    with self.test_session(use_gpu=use_gpu):

      @function.Defun(dtypes.int32, *[dtypes.float64] * 3)
      def MLP(i, a, ws, bs):
        # [FUNCTION DOC]
        # HELPER: loop body for _tfMLP. Takes the layer index, activation a, weights ws and biases bs, applies
        # a tanh of matmul plus bias for that layer and returns the new activation with weights and biases.
        a = math_ops.tanh(math_ops.matmul(a, ws[i, :]) + bs[i, :])
        return a, ws, bs

      ret = functional_ops.For(
          0,
          wsval.shape[0],
          1, [xval, wsval, bsval],
          MLP,
          rewrite_with_while=rewrite_with_while)[0]

      return self.evaluate(ret)

  def _npMLP(self, xval, wsval, bsval):
    # [FUNCTION DOC]
    # HELPER: reference implementation of the same multi-layer perceptron in numpy; loops over layers
    # applying tanh of dot product plus bias to the input and returns the result.
    for i in range(wsval.shape[0]):
      xval = np.tanh(np.dot(xval, wsval[i, :]) + bsval[i, :])
    return xval

  def _testForMLP(self, rewrite_with_while):
    # [FUNCTION DOC]
    # HELPER: shared body of testForMLP and testForMLPWhile. Creates random input (2 by 3), five layers of
    # 3 by 3 weights and 3-element biases, computes numpy and For results and asserts they are close.
    # We construct a 5-layer Multi-Layer Perceptron network here.
    # Each layer have the same number of hidden unites (3), and the
    # activation function is tanh().  We feed the input (xval) with
    # batch size 2.
    xval = np.random.normal(size=(2, 3))
    wsval = np.random.normal(size=(5, 3, 3))
    bsval = np.random.normal(size=(5, 3))
    np_ans = self._npMLP(xval, wsval, bsval)
    tf_for_ans = self._tfMLP(xval, wsval, bsval, rewrite_with_while)
    self.assertAllClose(np_ans, tf_for_ans)

  @test_util.run_deprecated_v1
  def testForMLP(self):
    # [FUNCTION DOC]
    # Test (deprecated version 1 graph mode) of the multi-layer perceptron via the For op without rewrite.
    self._testForMLP(False)

  @test_util.run_deprecated_v1
  @test_util.disable_xla(
      "Test uses strided slice without compile time constant values")
  def testForMLPWhile(self):
    # [FUNCTION DOC]
    # Test (deprecated version 1 graph mode, skipped under XLA because strided slice needs compile-time
    # constants) of the multi-layer perceptron via the For op rewritten to a while loop.
    self._testForMLP(True)

  @test_util.run_v1_only("b/120545219")
  def testForError(self):
    # [FUNCTION DOC]
    # Test (version 1 only) of For argument validation. Expects invalid-argument errors for a non-scalar
    # start ("must be a scalar"), a negative delta with an increasing range ("Invalid start/limit/delta")
    # and a body returning two values where one is expected.

    @function.Defun(dtypes.int32, dtypes.float32)
    def Foo(i, v):
      # [FUNCTION DOC]
      # HELPER: valid body for testForError; returns the index cast to float plus the value.
      return math_ops.cast(i, dtypes.float32) + v

    @function.Defun(dtypes.int32, dtypes.float32)
    def ReturnsTooManyArgs(unused_i, v):
      # [FUNCTION DOC]
      # HELPER: deliberately invalid body for testForError that returns two values (v, v) where one is
      # expected.
      return v, v

    with self.test_session():
      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  "must be a scalar"):
        functional_ops.For([0], 10, 1, [0.0], Foo)[0].eval()
      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  "Invalid start/limit/delta"):
        functional_ops.For(0, 10, -1, [0.0], Foo)[0].eval()
      with self.assertRaisesRegex(
          errors.InvalidArgumentError,
          "For loop body returned 2 arguments. Expected: 1"):
        functional_ops.For(0, 10, 1, [0.0], ReturnsTooManyArgs)[0].eval()

  @test_util.run_deprecated_v1
  def testGradient(self):
    # [FUNCTION DOC]
    # Test (deprecated version 1 graph mode) of the Gradient function-call op. Defines a cubic polynomial
    # and a function computing its derivative via the Gradient op. Expects values 8 and 4 at zero and 17
    # and 16 at one.

    @function.Defun(dtypes.float32)
    def Poly(x):
      # [FUNCTION DOC]
      # HELPER: polynomial 2x cubed plus 3x squared plus 4x plus 8 on a float.
      # y = 2x^3+3x^2+4x+8
      return 2 * x * x * x + 3 * x * x + 4 * x + 8

    @function.Defun(dtypes.float32)
    def Grad(x):
      # [FUNCTION DOC]
      # HELPER: returns the derivative of the polynomial at x via the Gradient op (6x squared plus 6x plus 4).
      # dy/dx = dy/dy * dy/dx = 1.0 * (6x^2+6x+4)
      return functional_ops.Gradient([x, 1.0], Poly)[0]

    with self.test_session(use_gpu=False) as sess:
      a = constant_op.constant(0.)
      avals = [Poly(a), Grad(a)]
      b = constant_op.constant(1.)
      bvals = [Poly(b), Grad(b)]
      self.assertAllEqual(self.evaluate(avals), [8., 4.])
      self.assertAllEqual(self.evaluate(bvals), [17., 16.])

  @test_util.run_v2_only
  def testCollective(self):
    # [FUNCTION DOC]
    # Test (version 2 only) that a collective can run inside a remote-called function. Resets the eager
    # context and creates two logical CPU devices. A compiled function remote-calls a function doing an
    # all-reduce of group size one on cpu 1; expects the result list containing a list with 1.
    # Side effects: resets global context and device configuration.
    context._reset_context()
    logical_devices = []
    logical_devices.append(context.LogicalDeviceConfiguration())
    logical_devices.append(context.LogicalDeviceConfiguration())
    framework_config.set_logical_device_configuration(
        framework_config.list_physical_devices("CPU")[0], logical_devices)

    @function.Defun(dtypes.float32)
    def collective_fn(t):
      # [FUNCTION DOC]
      # HELPER: remote target; runs an all-reduce with group size one on the float tensor and returns it.
      # Run a dummy collective of group size 1 to test the setup.
      return collective_ops.all_reduce_v2(
          t, group_size=1, group_key=1, instance_key=1)

    @eager_def_function.function
    def run():
      # [FUNCTION DOC]
      # HELPER: compiled function that remote-calls the collective function on cpu 1 with a constant one
      # element vector plus captured inputs and returns the results.
      with ops.device("/cpu:0"):
        return functional_ops.remote_call(
            args=[constant_op.constant([1.])] + collective_fn.captured_inputs,
            Tout=[dtypes.float32],
            f=collective_fn,
            target="/cpu:1")

    self.assertAllEqual(run(), [[1.]])


@test_util.run_all_in_graph_and_eager_modes
@test_util.with_control_flow_v2
class FunctionalOpsCaseTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the Case op of the functional ops generated module. Decorated to run every test in
  # graph and eager modes and with control flow version 2.

  def testCase(self):
    # [FUNCTION DOC]
    # Test of the Case op with three branches (times two, three, four). Wraps each in a concrete function
    # and runs with branch indices 0, 1, 2, minus one and 6. Expects 2, 3, 4, then 4 for both
    # out-of-range indices (the last branch is the default).
    @eager_def_function.function
    def two(x):
      # [FUNCTION DOC]
      # HELPER: compiled function returning x times 2.
      return x * 2

    @eager_def_function.function
    def three(x):
      # [FUNCTION DOC]
      # HELPER: compiled function returning x times 3.
      return x * 3

    @eager_def_function.function
    def four(x):
      # [FUNCTION DOC]
      # HELPER: compiled function returning x times 4.
      return x * 4

    def f(branch, x):
      # [FUNCTION DOC]
      # HELPER: runs the Case op for testCase.
      # KNOWN ISSUE: the comprehension variable also named f shadows this helper within the expression.
      # Takes a branch index and tensor x, builds concrete functions of the three
      # branches using a zeros template, runs the Case op and returns the identity of its first output.
      tmpl = array_ops.zeros_like(x)
      return array_ops.identity(gen_functional_ops.case(
          branch, input=[x], Tout=[dtypes.float32],
          branches=[f.get_concrete_function(tmpl)
                    for f in (two, three, four)])[0])
    one = array_ops.ones([])
    self.assertAllEqual(np.float32(2), self.evaluate(f(0, one)))
    self.assertAllEqual(np.float32(3), self.evaluate(f(1, one)))
    self.assertAllEqual(np.float32(4), self.evaluate(f(2, one)))
    self.assertAllEqual(np.float32(4), self.evaluate(f(-1, one)))  # <0 default
    self.assertAllEqual(np.float32(4), self.evaluate(f(6, one)))  # >=N default

  @test_util.run_deprecated_v1
  @test_util.disable_xla("Don't lower for XLA")
  def testSkipEagerCaseLoweringPreservesNameForFetch(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name implies skipping eager but the test only has the deprecated version 1 marker
    # and the class decorator also asks for eager runs.
    # Test (XLA disabled) that a Case op marked for lowering keeps its name for fetching. For each GPU
    # setting it runs branches of three defined functions returning an integer and a scaled float, fetching
    # the second output by tensor and by name "my_case:1". Expects 2, 3 and 4 times the input, with
    # out-of-range indices (7 and minus one) using the last branch.
    for use_gpu in (True, False):
      def Run(branch, x, fetch_by_name, use_gpu=use_gpu):
        # [FUNCTION DOC]
        # HELPER: builds the Case graph and runs it. Takes branch index, x, fetch_by_name and use_gpu. Builds
        # a Case op named my_case, sets the lowering attribute, adds an identity-n, and returns the evaluated
        # second output (by name or by tensor).
        with ops.Graph().as_default() as g:
          @function.Defun(dtypes.float32)
          def two(x):
            # [FUNCTION DOC]
            # HELPER: defined function returning minus one and x times 2.
            return -1, x * 2

          @function.Defun(dtypes.float32)
          def three(x):
            # [FUNCTION DOC]
            # HELPER: defined function returning zero and x times 3.
            return 0, x * 3

          @function.Defun(dtypes.float32)
          def four(x):
            # [FUNCTION DOC]
            # HELPER: defined function returning one and x times 4.
            return 1, x * 4

          outputs = gen_functional_ops.case(branch, input=[x],
                                            Tout=[dtypes.int32, dtypes.float32],
                                            branches=[two, three, four],
                                            name="my_case")

          # `outputs` is the list of output tensors of the Case op. We
          # arbitrarily choose the 0th tensor to get the Case op and set the
          # lowering attribute on it.
          outputs[0].op._set_attr("_lower_using_switch_merge",
                                  attr_value_pb2.AttrValue(b=True))
          outputs = array_ops.identity_n(outputs)
        with self.session(graph=g, use_gpu=use_gpu) as sess:
          return sess.run("my_case:1" if fetch_by_name else outputs[1])

      self.assertAllEqual(2 * 1., Run(0, 1., False))
      self.assertAllEqual(2 * 1., Run(0, 1., True))
      self.assertAllEqual(3 * 7., Run(1, 7., False))
      self.assertAllEqual(3 * 7., Run(1, 7., True))
      self.assertAllEqual(4 * -3., Run(2, -3., False))
      self.assertAllEqual(4 * -3., Run(2, -3., True))
      self.assertAllEqual(4 * -4., Run(7, -4., False))  # >= N default
      self.assertAllEqual(4 * -4., Run(7, -4., True))  # >= N default
      self.assertAllEqual(4 * -5., Run(-1, -5., False))  # <0 default
      self.assertAllEqual(4 * -5., Run(-1, -5., True))  # <0 default

  @test_util.disable_xla("Don't lower for XLA")
  def testCaseLowering(self):
    # [FUNCTION DOC]
    # Test (XLA disabled) that a lowered Case op in a compiled function yields right results on the CPU
    # or GPU. With branches times 2, 3 and 4, expects 2, 3, 4 for indices 0, 1, 2 and 4 times the input
    # for out-of-range indices 7 and minus one.
    for use_gpu in (True, False):
      @eager_def_function.function
      def Run(branch, x):
        # [FUNCTION DOC]
        # HELPER: compiled function building a Case op over three defined functions with the lowering
        # attribute set on it and returns the second output after an identity-n. Takes branch index and x.
        @function.Defun(dtypes.float32)
        def two(x):
          # [FUNCTION DOC]
          # HELPER: defined function returning minus one and x times 2.
          return -1, x * 2

        @function.Defun(dtypes.float32)
        def three(x):
          # [FUNCTION DOC]
          # HELPER: defined function returning zero and x times 3.
          return 0, x * 3

        @function.Defun(dtypes.float32)
        def four(x):
          # [FUNCTION DOC]
          # HELPER: defined function returning one and x times 4.
          return 1, x * 4

        outputs = gen_functional_ops.case(branch, input=[x],
                                          Tout=[dtypes.int32, dtypes.float32],
                                          branches=[two, three, four])

        # `outputs` is the list of output tensors of the Case op. We
        # arbitrarily choose the 0th tensor to get the Case op and set the
        # lowering attribute on it.
        outputs[0].op._set_attr("_lower_using_switch_merge",
                                attr_value_pb2.AttrValue(b=True))
        outputs = array_ops.identity_n(outputs)
        return outputs[1]

      with ops.device(test.gpu_device_name() if use_gpu else "CPU:0"):
        self.assertAllEqual(2 * 1., self.evaluate(Run(0, 1.)))
        self.assertAllEqual(3 * 7., self.evaluate(Run(1, 7.)))
        self.assertAllEqual(4 * -3., self.evaluate(Run(2, -3.)))
        self.assertAllEqual(4 * -4., self.evaluate(Run(7, -4.)))  # >=N default
        self.assertAllEqual(4 * -5., self.evaluate(Run(-1, -5.)))  # <0 default

if __name__ == "__main__":
  test.main()

# pylint: enable=invalid-name
