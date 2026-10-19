# [MODULE DOC]
# Test module for the legacy defined-function machinery in the framework package. Contains test
# cases covering function declaration and calling, naming, gradients, captured values, control
# flow, stateful ops, optimizer configurations, reconstruction from function definitions and
# libraries, an unrolled LSTM comparison, variable hoisting, templates and device placement.
# Defines a generator of optimizer configurations used across tests. Run as a script via the test
# main entry point; no import-time side effects other than imports.
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
# =============================================================================
"""Tests for functions."""

import re
import time

from absl.testing import parameterized
import numpy as np

from tensorflow.core.framework import function_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import function
from tensorflow.python.framework import graph_to_function_def
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.framework.errors import InvalidArgumentError
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import gen_logging_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import template
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


def _OptimizerOptions():
  # [FUNCTION DOC]
  # HELPER: generator of session configurations shared by the tests in this file.
  # Generator taking no inputs. It iterates over every combination of three switches (common
  # subexpression elimination, function inlining, constant folding), yielding one session configuration
  # message per combination (eight in total). Each configuration uses optimization level zero with
  # those switches set, and additionally turns the matching rewriter options (arithmetic optimization,
  # function optimization, constant folding) on or off to agree with the switches. Raises nothing and
  # has no side effects beyond building new configuration messages.
  for cse in [False, True]:
    for inline in [False, True]:
      for cfold in [False, True]:
        cfg = config_pb2.ConfigProto(
            graph_options=config_pb2.GraphOptions(
                optimizer_options=config_pb2.OptimizerOptions(
                    opt_level=config_pb2.OptimizerOptions.L0,
                    do_common_subexpression_elimination=cse,
                    do_function_inlining=inline,
                    do_constant_folding=cfold)))
        if cse:
          cfg.graph_options.rewrite_options.arithmetic_optimization = (
              rewriter_config_pb2.RewriterConfig.ON)
        else:
          cfg.graph_options.rewrite_options.arithmetic_optimization = (
              rewriter_config_pb2.RewriterConfig.OFF)
        if inline:
          cfg.graph_options.rewrite_options.function_optimization = (
              rewriter_config_pb2.RewriterConfig.ON)
        else:
          cfg.graph_options.rewrite_options.function_optimization = (
              rewriter_config_pb2.RewriterConfig.OFF)
        if cfold:
          cfg.graph_options.rewrite_options.constant_folding = (
              rewriter_config_pb2.RewriterConfig.ON)
        else:
          cfg.graph_options.rewrite_options.constant_folding = (
              rewriter_config_pb2.RewriterConfig.OFF)
        yield cfg


class FunctionTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the legacy function-definition decorator: defining functions, calling them,
  # naming, outputs, gradients (symbolic and custom), capturing outside tensors, control flow, errors,
  # statefulness and device handling. The docstring says the methods are mixed into two test cases
  # with and without C API support; in this file the class is a plain test case with no shared
  # fixtures, each test builds its own graph.
  """Test methods for verifying Function support.

  These test methods are used as mix-ins in two test cases: with
  and without C API support.
  """

  def testIdentity(self):
    # [FUNCTION DOC]
    # Scenario: an identity function declared with a float input and a custom name is called on a
    # constant list inside a fresh graph. Expects the call operation to carry the custom name and the
    # evaluated result to equal the input value.

    @function.Defun(dtypes.float32, func_name="MyIdentity")
    def MyIdentityFunc(a):
      # [FUNCTION DOC]
      # HELPER: function body under test; returns its single input unchanged.
      return a

    with ops.Graph().as_default():
      call = MyIdentityFunc([18.0])
      self.assertEqual("MyIdentity", call.op.name)
      with session.Session() as sess:
        self.assertAllEqual([18.0], self.evaluate(call))

  @test_util.run_v1_only("b/120545219")
  def testIdentityImplicitDeref(self):
    # [FUNCTION DOC]
    # Scenario (graph-mode only, skipped under the second API version): a reference variable is passed
    # to an identity function as a raw reference tensor. Expects the call op name to be the custom name
    # and, for every optimizer configuration, that after initializing the variable the result equals the
    # variable's value.

    @function.Defun(dtypes.float32, func_name="MyIdentity")
    def MyIdentityFunc(a):
      # [FUNCTION DOC]
      # HELPER: function body under test; returns its single input unchanged.
      return a

    with ops.Graph().as_default():
      var = variable_v1.VariableV1([18.0])
      call = MyIdentityFunc(var._ref())  # pylint: disable=protected-access
      self.assertEqual("MyIdentity", call.op.name)
      for cfg in _OptimizerOptions():
        with session.Session(config=cfg) as sess:
          self.evaluate(var.initializer)
          self.assertAllEqual([18.0], self.evaluate(call))

  def testIdentityOutputName(self):
    # [FUNCTION DOC]
    # Scenario: an identity function is declared with an explicit output name list of one entry. Expects
    # the call op name to equal the function name and the evaluated value to equal the input.

    @function.Defun(
        dtypes.float32, func_name="MyIdentity", out_names=["my_result_name"])
    def MyIdentityFunc(a):
      # [FUNCTION DOC]
      # HELPER: function body under test; returns its single input unchanged.
      return a

    with ops.Graph().as_default():
      call = MyIdentityFunc([18.0])
      self.assertEqual("MyIdentity", call.op.name)
      with session.Session() as sess:
        self.assertAllEqual([18.0], self.evaluate(call))

  def testTooManyOutputNames(self):
    # [FUNCTION DOC]
    # Scenario: a function with one output is declared with two output names. Calling it must raise an
    # invalid-argument error whose message states that output names must be empty or equal in size to
    # the outputs, reporting sizes two and one.

    @function.Defun(
        dtypes.float32,
        func_name="MyIdentity",
        out_names=["my_result1", "my_result2"])
    def MyIdentityFunc(a):
      # [FUNCTION DOC]
      # HELPER: function body under test; returns its single input unchanged.
      return a

    with ops.Graph().as_default():
      with self.assertRaisesRegex(
          errors_impl.InvalidArgumentError,
          (r"output names must be either empty or equal in size to outputs. "
           "output names size = 2 outputs size = 1")):
        MyIdentityFunc([18.0])

  def testDefineFunction2Args(self):
    # [FUNCTION DOC]
    # Scenario: a two-argument function computing the first plus twice the second is called with one
    # element lists. Expects the call op to have the declared name and the result to be five.

    @function.Defun(dtypes.float32, dtypes.float32, func_name="APlus2B")
    def APlus2B(a, b):
      # [FUNCTION DOC]
      # HELPER: function body under test; returns first argument plus two times the second.
      return a + b * 2

    with ops.Graph().as_default():
      call = APlus2B([1.0], [2.0])
      self.assertEqual("APlus2B", call.op.name)
      with session.Session() as sess:
        self.assertAllEqual([5.0], self.evaluate(call))

  def testFunctionWithNoOutput(self):
    # [FUNCTION DOC]
    # Scenario: a function whose body builds some ops but returns nothing is called inside a graph.
    # Expects no exception to be raised; nothing else is asserted.

    @function.Defun(dtypes.float32, dtypes.float32)
    def APlus2B(a, b):
      # [FUNCTION DOC]
      # HELPER: function body under test; computes a temporary value and prints it (only to avoid an
      # unused-variable lint warning) and returns nothing.
      c = a + b * 2  # Create some ops to have nodes in the body
      print(c)  # Using 'print' to make lint happy

    with ops.Graph().as_default():
      # Call function. There should be no exceptions.
      APlus2B([1.0], [2.0])

  def testDefineFunction2ArgsOutputName(self):
    # [FUNCTION DOC]
    # Scenario: the two-argument function is declared with one explicit output name. Expects the
    # function to report no stateful ops, the call op to carry the function name and the result to be
    # five.

    @function.Defun(
        dtypes.float32,
        dtypes.float32,
        func_name="APlus2B",
        out_names=["my_result_name"])
    def APlus2B(a, b):
      # [FUNCTION DOC]
      # HELPER: function body under test; returns first argument plus two times the second.
      return a + b * 2

    # APlus2B is stateless.
    self.assertEqual([], APlus2B.stateful_ops)
    with ops.Graph().as_default():
      call = APlus2B([1.0], [2.0])
      self.assertEqual("APlus2B", call.op.name)
      with session.Session() as sess:
        self.assertAllEqual([5.0], self.evaluate(call))

  def testDefineFunctionDuplicateOutputs(self):
    # [FUNCTION DOC]
    # Scenario: a function returns the same tensor twice. After calling it in a graph, the function
    # signature in the graph's function library must list two outputs whose names are different even
    # though the same tensor feeds both.

    @function.Defun(dtypes.float32, func_name="Duplicate")
    def Duplicate(a):
      # [FUNCTION DOC]
      # HELPER: function body under test; adds one to its input and returns that single tensor twice.
      b = a + 1.0
      return b, b

    g = ops.Graph()
    with g.as_default():
      Duplicate([3.0])
      func_sig = g.as_graph_def().library.function[0].signature
      # The names given to both outputs should be different
      # even though the same tensor is emitted to both.
      out_names = [a.name for a in func_sig.output_arg]
      self.assertEqual(2, len(out_names))
      self.assertNotEqual(out_names[0], out_names[1])

  def testGradientFunc(self):
    # [FUNCTION DOC]
    # Scenario: a function computing x squared plus one and an explicit gradient function built from the
    # symbolic gradient op referencing the first function by name. Evaluates both at two with upstream
    # gradient 0.1 and expects five and 0.4 respectively.

    @function.Defun(dtypes.float32, func_name="XSquarePlusOneFn")
    def XSquarePlusOne(x):
      # [FUNCTION DOC]
      # HELPER: forward function under test; returns x times x plus one.
      return x * x + 1.0

    @function.Defun(dtypes.float32, dtypes.float32)
    def XSquarePlusOneGrad(x, dy):
      # [FUNCTION DOC]
      # HELPER: hand-written gradient function; takes x and the upstream gradient and returns the result of
      # the symbolic gradient op applied to the forward function by name.
      dx = functional_ops.symbolic_gradient(
          input=[x, dy], Tout=[dtypes.float32], f="XSquarePlusOneFn", name="dx")
      return dx

    g = ops.Graph()
    with g.as_default():
      call_f = XSquarePlusOne([2.0])
      call_g = XSquarePlusOneGrad([2.0], [0.1])

      with session.Session() as sess:
        self.assertAllClose([5.0], self.evaluate(call_f))
        self.assertAllClose([0.4], self.evaluate(call_g))

  def testTanhSymGrad(self):
    # [FUNCTION DOC]
    # Scenario: a function that sums the hyperbolic tangent of its input is differentiated with the
    # standard gradient builder. With function inlining enabled at optimization level one, running the
    # gradient on four sample values must match one minus the square of the tangent of each input.

    @function.Defun(dtypes.float32)
    def Forward(x):
      # [FUNCTION DOC]
      # HELPER: function body under test; returns the sum of tanh of its input.
      return math_ops.reduce_sum(math_ops.tanh(x))

    g = ops.Graph()
    with g.as_default():
      x = array_ops.placeholder(dtypes.float32)
      y = Forward(x)
      dx = gradients_impl.gradients([y], [x])

    inp = np.array([-1, 1, 2, -2], dtype=np.float32)
    feed = {x: inp}
    cfg = config_pb2.ConfigProto(
        graph_options=config_pb2.GraphOptions(
            optimizer_options=config_pb2.OptimizerOptions(
                opt_level=config_pb2.OptimizerOptions.L1,
                do_function_inlining=True)))
    with session.Session(graph=g, config=cfg) as sess:
      out, = sess.run(dx, feed)
    self.assertAllClose(1 - np.square(np.tanh(inp)), out)

  def testCustomGradient(self):
    # [FUNCTION DOC]
    # Scenario: a softmax cross-entropy style function is given a custom gradient function that
    # deliberately returns the exponential of the true gradient. Gradients are computed with respect to
    # the logits and run under every optimizer configuration with random data; the result must match
    # the exponential of the probabilities minus the labels within a relative tolerance. Logs each
    # configuration.
    dtype = dtypes.float32

    @function.Defun(dtype, dtype, dtype)
    def XentLossGrad(logits, labels, dloss):
      # [FUNCTION DOC]
      # HELPER: custom gradient under test; returns the exponential of the logit gradient (to distinguish
      # it from the correct gradient) and zeros for the labels gradient. The third argument is the
      # upstream loss gradient.
      dlogits = array_ops.reshape(dloss, [-1, 1]) * (
          nn_ops.softmax(logits) - labels)
      dlabels = array_ops.zeros_like(labels)
      # Takes exp(dlogits) to differentiate it from the "correct" gradient.
      return math_ops.exp(dlogits), dlabels

    @function.Defun(dtype, dtype, grad_func=XentLossGrad)
    def XentLoss(logits, labels):
      # [FUNCTION DOC]
      # HELPER: forward function under test; sums labels times log-softmax-like values across axis one.
      # Uses the custom gradient function declared above.
      return math_ops.reduce_sum(labels * math_ops.log(nn_ops.softmax(logits)),
                                 1)

    g = ops.Graph()
    with g.as_default():
      logits = array_ops.placeholder(dtype)
      labels = array_ops.placeholder(dtype)
      loss = XentLoss(logits, labels)
      dlogits = gradients_impl.gradients([loss], [logits])

    x = np.random.uniform(-10., 10., size=(4, 9)).astype(np.float32)
    prob = np.exp(x) / np.sum(np.exp(x), 1, keepdims=1)
    y = np.random.uniform(-10., 10., size=(4, 9)).astype(np.float32)
    for cfg in _OptimizerOptions():
      tf_logging.info("cfg = %s", cfg)
      with session.Session(graph=g, config=cfg) as sess:
        out, = sess.run(dlogits, {logits: x, labels: y})
      self.assertAllClose(out, np.exp(prob - y), rtol=1e-5)

  @test_util.disable_xla("b/124286351")  # No error is raised
  def testCustomGradientError(self):
    # [FUNCTION DOC]
    # Scenario (disabled under the XLA compiler because no error is raised there): a function with two
    # outputs is given a gradient function that returns only a single result in the wrong arity.
    # Running the gradient must raise an invalid-argument error saying the symbolic gradient expects one
    # result but received two.
    dtype = dtypes.float32

    @function.Defun(dtype, dtype, dtype)
    def Grad(x, dy, dz):
      # [FUNCTION DOC]
      # HELPER: deliberately wrong gradient function; returns the first input and the sum of the other two
      # instead of the expected single result.
      # Should have returned 1 result.
      return x, dy + dz

    @function.Defun(dtype, grad_func=Grad)
    def Forward(x):
      # [FUNCTION DOC]
      # HELPER: forward function under test; returns its input twice.
      return x, x

    g = ops.Graph()
    with g.as_default():
      inp = array_ops.placeholder(dtype)
      out = math_ops.add_n(Forward(inp))
      dinp = gradients_impl.gradients(out, [inp])

    x = np.random.uniform(-10., 10., size=(4, 9)).astype(np.float32)
    with session.Session(graph=g) as sess:
      with self.assertRaisesRegex(
          errors_impl.InvalidArgumentError,
          "SymGrad expects to return 1.*but get 2.*instead"):
        _ = sess.run(dinp, {inp: x})

  def testSymGradShape(self):
    # [FUNCTION DOC]
    # Scenario: the symbolic gradient op is built for an undefined function name with placeholders of
    # different shapes. Expects the static shapes of the two returned gradients to equal the shapes of
    # the corresponding inputs.
    g = ops.Graph()
    with g.as_default():
      x = array_ops.placeholder(dtypes.float32, [25, 4])
      y = array_ops.placeholder(dtypes.float32, [200, 100])
      dz = array_ops.placeholder(dtypes.float32, [1])
      # We assume Foo is a function of (x, y) -> (z) Then, Foo's
      # gradient function is (x, y, dz) -> (dx, dy).  dx's shape
      # should be the same as x's; and dy's shape should be the same
      # as y's.
      dx, dy = functional_ops.symbolic_gradient(
          input=[x, y, dz], Tout=[dtypes.float32] * 2, f="Foo")
      self.assertEqual(x.get_shape(), dx.get_shape())
      self.assertEqual(y.get_shape(), dy.get_shape())

  @test_util.run_deprecated_v1
  def testSymGradAttr(self):
    # [FUNCTION DOC]
    # Scenario (graph-mode only): a function declared as not inlinable is instantiated for float
    # inputs; its definition must carry the no-inline attribute set to true. It is then called on a
    # constant and differentiated with inlining, constant folding and elimination enabled; the value
    # must be six and the gradient two.

    @function.Defun(noinline=True)
    def Foo(x):
      # [FUNCTION DOC]
      # HELPER: function body under test; returns its input doubled.
      return x * 2

    self.assertTrue(
        Foo.instantiate([dtypes.float32]).definition.attr["_noinline"].b)

    g = ops.Graph()
    with g.as_default():
      x = constant_op.constant(3.0)
      y = Foo(x)
      dx, = gradients_impl.gradients(y, [x])

    cfg = config_pb2.ConfigProto(
        graph_options=config_pb2.GraphOptions(
            optimizer_options=config_pb2.OptimizerOptions(
                opt_level=config_pb2.OptimizerOptions.L0,
                do_common_subexpression_elimination=True,
                do_function_inlining=True,
                do_constant_folding=True)))

    with self.session(graph=g, config=cfg):
      self.assertAllClose(y, 6.)
      self.assertAllClose(dx, 2.)

  def _testZNoDepOnY(self, use_const_grad_ys):
    # [FUNCTION DOC]
    # HELPER: shared body for the two tests that follow.
    # KNOWN ISSUE: a comment inside is truncated mid-word ("z doe") so the intent of the comment is lost.
    # Builds a function of two arguments that ignores the second, calls it on constants one and two and
    # differentiates the output with respect to both inputs, optionally passing explicit constant
    # upstream gradients. Parameter use_const_grad_ys selects that variant (true) or the default (false).
    # Asserts the gradient with respect to the first input is two and with respect to the unused second
    # input is zero.

    @function.Defun(dtypes.float32, dtypes.float32)
    def Foo(x, y):  # pylint: disable=unused-argument
      # [FUNCTION DOC]
      # HELPER: function body under test; returns the first argument doubled and ignores the second.
      return x * 2

    with ops.Graph().as_default():
      # z = Foo(x, y). z doe
      x = constant_op.constant(1.0)
      y = constant_op.constant(2.0)
      z = Foo(x, y)
      if use_const_grad_ys:
        dx, dy = gradients_impl.gradients([z], [x, y], grad_ys=[1.0])
      else:
        dx, dy = gradients_impl.gradients([z], [x, y])
      with session.Session() as sess:
        dx_val, dy_val = self.evaluate([dx, dy])
        self.assertEqual([2.0], dx_val)
        self.assertEqual([0.0], dy_val)

  def testZNoDepOnY(self):
    # [FUNCTION DOC]
    # Scenario: the output does not depend on the second input and the gradient is taken without
    # explicit upstream gradients; delegates to the shared helper with the constant-gradient option off.
    self._testZNoDepOnY(False)

  def testZNoDepOnYConstGradYs(self):
    # [FUNCTION DOC]
    # Scenario: same as the previous test but with explicit constant upstream gradients, which exercises
    # constant folding of those gradients; delegates to the shared helper with the option on.
    # Tests for constant folding of grad_ys
    self._testZNoDepOnY(True)

  def testDefineFunctionNoArgs(self):
    # [FUNCTION DOC]
    # Scenario: a function with no arguments returning the constant forty-two is called in a graph.
    # Expects the call op to carry the function name and the result to equal forty-two.

    @function.Defun(func_name="AConstant")
    def AConstant():
      # [FUNCTION DOC]
      # HELPER: function body under test; returns a constant one element tensor holding forty-two.
      return constant_op.constant([42])

    with ops.Graph().as_default():

      call = AConstant()
      self.assertEqual("AConstant", call.op.name)
      with session.Session() as sess:
        self.assertAllEqual([42], self.evaluate(call))

  def testDefineFunctionNames(self):
    # [FUNCTION DOC]
    # Scenario: the same function is called four times: plainly twice, once with an explicit name and
    # once with an explicit name inside a name scope. Expects op names to be the function name, the
    # function name with a numeric suffix, the explicit name, and the scope-prefixed explicit name.

    @function.Defun(dtypes.float32, func_name="Foo")
    def Foo(a):
      # [FUNCTION DOC]
      # HELPER: function body under test; returns its input plus one.
      return a + 1

    with ops.Graph().as_default():
      call1 = Foo([1.0])
      self.assertEqual("Foo", call1.op.name)
      call2 = Foo([1.0])
      self.assertEqual("Foo_1", call2.op.name)
      # pylint: disable=unexpected-keyword-arg
      call3 = Foo([1.0], name="mine")
      self.assertEqual("mine", call3.op.name)
      with ops.name_scope("my"):
        call4 = Foo([1.0], name="precious")
        self.assertEqual("my/precious", call4.op.name)

  def testNoOp(self):
    # [FUNCTION DOC]
    # Scenario: a function prints a message, then has a no-op depend on the print and returns twice its
    # input under those control dependencies. Runs on a cached session without GPU (the function is not
    # partitioned so a print cannot run on GPU) and expects the result six.

    @function.Defun(dtypes.float32)
    def Foo(x):
      # [FUNCTION DOC]
      # HELPER: function body under test; prints its input, creates a no-op under a control dependency
      # on the print, and returns the input doubled under a control dependency on the no-op.
      y = logging_ops.Print(x, [], "Hello")
      with ops.control_dependencies([y]):
        z = gen_control_flow_ops.no_op()
      with ops.control_dependencies([z]):
        return x * 2

    # @function.Defun creates a non-partitioned function.  If we place this on
    # the GPU then the inner `Print` op cannot be run.
    with ops.Graph().as_default(), self.cached_session(use_gpu=False):
      z = Foo(constant_op.constant(3.0))
      self.assertAllEqual(z, 6.0)

  def testAssertOp(self):
    # [FUNCTION DOC]
    # Scenario: a function contains an assertion that the input is positive. Expects the function to
    # list the assertion as a stateful op, to evaluate to six for input three, and to raise an
    # invalid-argument error mentioning the failed assertion and the value minus three for input minus
    # three.

    @function.Defun(dtypes.float32)
    def Foo(x):
      # [FUNCTION DOC]
      # HELPER: function body under test; asserts the input is greater than zero and returns the input
      # doubled under a control dependency on that assertion.
      check = gen_logging_ops._assert(math_ops.greater(x, 0), [x])
      with ops.control_dependencies([check]):
        return x * 2

    # Foo contains a stateful op (Assert).
    self.assertEqual([("Assert", "Assert")], Foo.stateful_ops)
    g = ops.Graph()
    with g.as_default(), self.cached_session():
      self.assertAllEqual(Foo(constant_op.constant(3.0)), 6.0)
      with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                  "assertion failed.*-3"):
        self.assertAllEqual(Foo(constant_op.constant(-3.0)), 6.0)

  @test_util.run_deprecated_v1
  def testAssertWrapper(self):
    # [FUNCTION DOC]
    # Scenario (graph-mode only): a function uses the public assert wrapper to require the input to be
    # at most ten. Input one must evaluate to one; input one hundred must raise an invalid-argument
    # error containing the word assertion.

    @function.Defun(dtypes.float32)
    def MyFn(x):
      # [FUNCTION DOC]
      # HELPER: function body under test; asserts the input is at most ten and returns an identity of the
      # input under that control dependency.
      with ops.control_dependencies(
          [control_flow_assert.Assert(math_ops.less_equal(x, 10.0), [x])]):
        return array_ops.identity(x)

    with self.cached_session():
      self.assertEqual(1.0, MyFn(1.0).eval())
      with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                  "assertion"):
        _ = MyFn(100.0).eval()

  @test_util.run_deprecated_v1
  def testWhileLoopCallsFunc(self):
    # [FUNCTION DOC]
    # Scenario (graph-mode only): a while loop whose body calls a function that multiplies by a value
    # computed from an integer constant cast to float (potentially placed on the GPU). Starting from one
    # and looping while below one hundred thousand, expects the final value to be 131072.
    with self.session():

      @function.Defun(dtypes.float32)
      def Times2(x):
        # [FUNCTION DOC]
        # HELPER: function body under test; casts an integer constant two to float and returns the input
        # times that value.
        constant_two = constant_op.constant(2, dtypes.int32)
        two_on_gpu = math_ops.cast(constant_two, dtypes.float32)
        return x * two_on_gpu

      def Body(x):
        # [FUNCTION DOC]
        # HELPER: while-loop body; calls the doubling function, sets the static shape of the result to a
        # scalar and returns it.
        x2 = Times2(x)
        x2.set_shape([])
        return x2

      loop = while_loop.while_loop(lambda x: x < 1e5, Body, [1.0])

      ans = self.evaluate(loop)
      self.assertAllClose(ans, 131072.)

  @test_util.run_deprecated_v1
  def testControlFlowStrictness(self):
    # [FUNCTION DOC]
    # Scenario (graph-mode only): a function containing an assertion that always fails and has no data
    # dependency on its input is used in the untaken branch of a conditional and as the body of a
    # while loop, on the CPU. Inlining is enabled and dependency optimization disabled. Expects no
    # failure when the failing branch is not taken (results four and three) and an invalid-argument
    # error mentioning the assertion when it is taken, for both the conditional and the loop.
    """Inlined functions must not execute in a untaken control flow branch."""

    @function.Defun(dtypes.int32)
    def AssertFail(x):
      # [FUNCTION DOC]
      # HELPER: function body under test; creates an always-failing assertion and returns an identity of
      # the input under a control dependency on it.
      # Assertion that always fails and does not have a data dependency on `x`.
      assert_false = control_flow_assert.Assert(False, [42])
      with ops.control_dependencies([assert_false]):
        return array_ops.identity(x)

    with ops.device("CPU"):
      pred = array_ops.placeholder(dtypes.bool)
      x = array_ops.placeholder(dtypes.int32)
      cond = tf_cond.cond(pred, lambda: x + 1, lambda: AssertFail(x))
      # pylint: disable=unnecessary-lambda
      loop = while_loop.while_loop(lambda y: pred, lambda y: AssertFail(y), [x])
      # pylint: enable=unnecessary-lambda

    rewriter_config = rewriter_config_pb2.RewriterConfig(
        dependency_optimization=rewriter_config_pb2.RewriterConfig.OFF)
    # Enables inlining.
    config = config_pb2.ConfigProto(
        graph_options=config_pb2.GraphOptions(
            optimizer_options=config_pb2.OptimizerOptions(
                opt_level=config_pb2.OptimizerOptions.L0,
                do_common_subexpression_elimination=True,
                do_function_inlining=True,
                do_constant_folding=True),
            rewrite_options=rewriter_config))

    with session.Session(config=config) as sess:
      # Since the 'False' branch is not taken, the assertion should not fire.
      self.assertEqual(4, sess.run(cond, {pred: True, x: 3}))

      # The assertion should still fire if the False branch is taken.
      with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                  "assertion"):
        sess.run(cond, {pred: False, x: 3})

      # Similarly for loops.
      self.assertEqual(3, sess.run(loop, {pred: False, x: 3}))
      with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                  "assertion"):
        sess.run(loop, {pred: True, x: 3})

  @test_util.run_deprecated_v1
  def testVar(self):
    # [FUNCTION DOC]
    # Scenario (graph-mode only): a function squaring its argument and adding one is called on a
    # variable initialized to ten. After initializing global variables, expects 101.

    @function.Defun(dtypes.float32)
    def Foo(x):
      # [FUNCTION DOC]
      # HELPER: function body under test; returns input squared plus one.
      return x * x + 1

    g = ops.Graph()
    with g.as_default():
      v = variables.Variable(constant_op.constant(10.0))
      z = Foo(v)

    with self.session(graph=g):
      self.evaluate(variables.global_variables_initializer())
      self.assertAllEqual(z, 101.)

  @test_util.run_deprecated_v1
  def testResourceVarAsImplicitInput(self):
    # [FUNCTION DOC]
    # Scenario (graph-mode only): a function with no arguments implicitly captures a resource variable
    # on CPU, makes an identity of it and checks inside the body that its dtype and shape match the
    # variable's, returning it together with its shape. After initializing the variable, expects the
    # returned value and shape to equal the variable's value and shape.
    g = ops.Graph()
    with g.as_default(), ops.device("cpu:0"):
      expected_type = dtypes.float32
      expected_shape = tensor_shape.TensorShape((4, 4))
      v = variable_scope.get_variable(
          "var", expected_shape, expected_type, use_resource=True)

      @function.Defun()
      def Foo():
        # [FUNCTION DOC]
        # HELPER: function body under test; reads the captured resource variable through an identity,
        # asserts dtype and shape of the captured tensor, and returns it plus its shape.
        captured = array_ops.identity(v)
        self.assertEqual(expected_type, captured.dtype)
        self.assertEqual(expected_shape, captured.shape)
        return captured, array_ops.shape(captured)

      expected_val = v.value()
      actual_val, actual_shape = Foo()

    with self.session(graph=g):
      v.initializer.run()
      self.assertAllEqual(expected_val, self.evaluate(actual_val))
      self.assertAllEqual(expected_shape, self.evaluate(actual_shape))

  def testDefineErrors(self):
    # [FUNCTION DOC]
    # Scenario: defining invalid functions must raise a value error when the definition is built.
    # Cases: returning two null values (cannot return nothing), a default argument value and arbitrary
    # keyword arguments (both unsupported), and a mismatch between the number of declared input types
    # and the number of function arguments (too few and too many types, message about input types).
    with ops.Graph().as_default():
      with self.assertRaisesRegex(ValueError, "can not return None"):

        @function.Defun()
        def TwoNone():
          # [FUNCTION DOC]
          # HELPER: invalid function under test; returns two null values.
          return None, None

        _ = TwoNone.definition

      with self.assertRaisesRegex(ValueError, "are not supported"):

        @function.Defun()
        def DefaultArg(unused_a=12):
          # [FUNCTION DOC]
          # HELPER: invalid function under test; has an argument with a default value.
          return constant_op.constant([1])

        _ = DefaultArg.definition
      with self.assertRaisesRegex(ValueError, "are not supported"):

        @function.Defun()
        def KwArgs(**unused_kwargs):
          # [FUNCTION DOC]
          # HELPER: invalid function under test; accepts arbitrary keyword arguments.
          return constant_op.constant([1])

        _ = KwArgs.definition
      with self.assertRaisesRegex(ValueError, "tf.function input types"):

        @function.Defun(dtypes.float32)
        def PlusMinusV2(a, b):
          # [FUNCTION DOC]
          # HELPER: invalid function under test; two arguments but only one declared input type; returns sum
          # and difference.
          return a + b, b - a

        _ = PlusMinusV2.definition
      with self.assertRaisesRegex(ValueError, "tf.function input types"):

        @function.Defun(dtypes.float32, dtypes.float32, dtypes.float32)
        def PlusMinusV3(a, b):
          # [FUNCTION DOC]
          # HELPER: invalid function under test; two arguments but three declared input types; returns sum
          # and difference.
          return a + b, b - a

        _ = PlusMinusV3.definition

  def testCallErrors(self):
    # [FUNCTION DOC]
    # Scenario: calling defined functions with the wrong number of positional arguments must raise a
    # value error stating the expected count (zero, one, two), valid calls must succeed, passing an
    # explicit op name is accepted, and passing an unknown keyword argument such as a device must raise
    # a value error about unknown keyword arguments.

    @function.Defun()
    def Const():
      # [FUNCTION DOC]
      # HELPER: function under test with no arguments; returns the constant one.
      return constant_op.constant(1)

    @function.Defun(dtypes.int32)
    def PlusOne(a):
      # [FUNCTION DOC]
      # HELPER: function under test with one integer argument; returns it plus one.
      return a + 1

    @function.Defun(dtypes.int32, dtypes.int32)
    def PlusMinus(a, b):
      # [FUNCTION DOC]
      # HELPER: function under test with two integer arguments; returns their sum and the second minus the
      # first.
      return a + b, b - a

    with ops.Graph().as_default():

      _ = Const()
      # pylint: disable=too-many-function-args
      # pylint: disable=unexpected-keyword-arg
      # pylint: disable=no-value-for-parameter
      with self.assertRaisesRegex(ValueError, "Expected 0"):
        _ = Const(1)
      with self.assertRaisesRegex(ValueError, "Expected 0"):
        _ = Const(1, 2)

      with self.assertRaisesRegex(ValueError, "Expected 1"):
        _ = PlusOne()
      _ = PlusOne(1)
      with self.assertRaisesRegex(ValueError, "Expected 1"):
        _ = PlusOne(1, 2)

      with self.assertRaisesRegex(ValueError, "Expected 2"):
        _ = PlusMinus()
      with self.assertRaisesRegex(ValueError, "Expected 2"):
        _ = PlusMinus(1)
      _ = PlusMinus(1, 2)

      _ = PlusOne(1, name="p1")
      with self.assertRaisesRegex(ValueError, "Unknown keyword arguments"):
        _ = PlusOne(1, device="/device:GPU:0")

  def testFunctionDecorator(self):
    # [FUNCTION DOC]
    # Scenario: the decorator yields an object of the internal defined-function type with the given name.
    # Calls it twice, the second time with a custom op name chained on the first result. Expects the
    # custom name on the second call op and values one and zero.

    @function.Defun(dtypes.float32, func_name="Minus1")
    def Minus1(b):
      # [FUNCTION DOC]
      # HELPER: function body under test; returns its input minus one.
      return b - 1.0

    with ops.Graph().as_default():
      call1 = Minus1([2.])
      self.assertTrue(isinstance(Minus1, function._DefinedFunction))
      self.assertEqual(Minus1.name, "Minus1")
      # pylint: disable=unexpected-keyword-arg
      call2 = Minus1(call1, name="next")
      # pylint: enable=unexpected-keyword-arg
      self.assertEqual("next", call2.op.name)
      with session.Session() as sess:
        self.assertAllEqual([1], self.evaluate(call1))
        self.assertAllEqual([0], self.evaluate(call2))

  def testNestedFunction(self):
    # [FUNCTION DOC]
    # Scenario: one defined function calls another defined at the outer scope. Cube of three minus two
    # must equal twenty-five.

    @function.Defun(dtypes.float32)
    def Cube(x):
      # [FUNCTION DOC]
      # HELPER: function body under test; returns the input cubed.
      return x * x * x

    @function.Defun(dtypes.float32, dtypes.float32)
    def CubeXPlusY(x, y):
      # [FUNCTION DOC]
      # HELPER: function body under test; calls the cube function on the first argument and adds the
      # second.
      return Cube(x) + y

    with ops.Graph().as_default():
      z = CubeXPlusY(3.0, -2.0)
      with self.cached_session():
        self.assertAllEqual(z, 25.0)

  def testNestedDefinedFunction(self):
    # [FUNCTION DOC]
    # Scenario: like the previous test but the cube function is defined inside the body of the outer
    # function. The result must be twenty-five.

    @function.Defun(dtypes.float32, dtypes.float32)
    def CubeXPlusY(x, y):
      # [FUNCTION DOC]
      # HELPER: outer function body under test; defines a cube function inside itself, calls it on the
      # first argument and adds the second.

      @function.Defun(dtypes.float32)
      def Cube(x):
        # [FUNCTION DOC]
        # HELPER: inner function body under test; returns the input cubed.
        return x * x * x

      return Cube(x) + y

    with ops.Graph().as_default():
      z = CubeXPlusY(3.0, -2.0)
      with self.cached_session():
        self.assertAllEqual(z, 25.0)

  def testUnusedFunction(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the flag recorded by the two unused functions is a local variable inside each
    # function body, so the outer flag can never change and the assertion that it stays false would
    # pass even if a function body were run. Only the library-size assertion truly checks anything.
    # Scenario: two functions are defined but never called (one inside a graph). Expects they were not
    # invoked and that the graph's function library is empty.
    invoked = False
    # pylint: disable=unused-variable
    @function.Defun()
    def Unused():
      # [FUNCTION DOC]
      # HELPER: never-called function; sets a local flag and returns a constant forty-two.
      invoked = True
      return constant_op.constant(42.)

    self.assertFalse(invoked)
    g = ops.Graph()
    with g.as_default():

      @function.Defun()
      def Unused2():
        # [FUNCTION DOC]
        # HELPER: never-called function defined inside a graph; sets a local flag and returns a constant
        # seven.
        invoked = True
        return constant_op.constant(7.)

      constant_op.constant(3.)
    # pylint: enable=unused-variable
    self.assertFalse(invoked)
    gdef = g.as_graph_def()
    self.assertEqual(0, len(gdef.library.function))

  @test_util.run_deprecated_v1
  def testReduction(self):
    # [FUNCTION DOC]
    # Scenario (graph-mode only): batch normalization computed along rows as a plain graph versus the
    # same computation wrapped in a function. On random data of shape three by seven, expects both the
    # outputs and the gradients to match.
    g = ops.Graph()

    # BN0 is computing batch normed matrix along rows.
    def BN0(x):
      # [FUNCTION DOC]
      # HELPER: plain batch-norm computation; subtracts the column mean, divides by the reciprocal square
      # root of the biased variance plus a small constant, and returns the normalized tensor.
      mean = math_ops.reduce_mean(x, [0])
      var = math_ops.reduce_mean(math_ops.square(x - mean))  # biased var
      rstd = math_ops.rsqrt(var + 1e-8)
      return (x - mean) * rstd

    # Wraps BatchNorm in a tf function.
    @function.Defun(dtypes.float32)
    def BN1(x):
      # [FUNCTION DOC]
      # HELPER: function wrapper that simply calls the plain batch-norm computation.
      return BN0(x)

    with g.as_default():
      x = array_ops.placeholder(dtypes.float32)
      y0 = BN0(x)  # A plain graph
      y1 = BN1(x)  # A tf function
      dx0, = gradients_impl.gradients([y0], [x])
      dx1, = gradients_impl.gradients([y1], [x])

    # Both should produce the same result and gradient.
    with self.session(graph=g) as sess:
      vals = sess.run([y0, y1, dx0, dx1], {x: np.random.uniform(size=(3, 7))})
      self.assertAllClose(vals[0], vals[1])
      self.assertAllClose(vals[2], vals[3])

  @test_util.run_deprecated_v1
  def testCapture(self):
    # [FUNCTION DOC]
    # Scenario (graph-mode only): functions capture variables defined outside. A function captures a
    # matrix variable and contains a nested function capturing a bias variable; another no-argument
    # function returns the captured matrix. After initialization, expects the first result twelve and
    # the second one.
    g = ops.Graph()
    with g.as_default():
      w = variables.Variable(constant_op.constant([[1.0]]))
      b = variables.Variable(constant_op.constant([2.0]))

      # Foo() captures w and b.
      @function.Defun(dtypes.float32)
      def Foo(x):
        # [FUNCTION DOC]
        # HELPER: function body under test; multiplies the captured matrix with its input and passes the
        # result to a nested function that adds the captured bias.

        # Plus() captures b.
        @function.Defun(dtypes.float32)
        def Plus(y):
          # [FUNCTION DOC]
          # HELPER: nested function under test; returns its input plus the captured bias variable.
          return y + b

        return Plus(math_ops.matmul(w, x))

      y = Foo(constant_op.constant([[10.]]))

      @function.Defun()
      def Bar():
        # [FUNCTION DOC]
        # HELPER: function body under test; returns the captured matrix variable.
        return w

      z = Bar()

    with self.session(graph=g):
      self.evaluate(variables.global_variables_initializer())
      self.assertAllEqual(y, [[12.0]])
      self.assertAllEqual(z, [[1.0]])

  def testCaptureControls(self):
    # [FUNCTION DOC]
    # Scenario: a function adds a control dependency on an outer tensor created by a print op. Capturing
    # control dependencies is not supported, so calling it must raise a value error that the tensor is
    # not an element of this graph.
    g = ops.Graph()
    with g.as_default():
      x = constant_op.constant([10.0])
      x = logging_ops.Print(x, [x], "outer")

      @function.Defun(dtypes.float32)
      def Foo(y):
        # [FUNCTION DOC]
        # HELPER: function body under test; under a control dependency on the outer print output, prints and
        # returns its input.
        with ops.control_dependencies([x]):
          y = logging_ops.Print(y, [y], "inner")
        return y

      with self.assertRaisesRegex(ValueError, "not an element of this graph."):
        # NOTE: We still do not support capturing control deps.
        _ = Foo(x)

  @test_util.run_deprecated_v1
  def testCaptureInWhileLoop(self):
    # [FUNCTION DOC]
    # Scenario (graph-mode only): a function with no arguments runs a while loop that adds a captured
    # outer constant one while the counter is below ten. Expects the result ten.
    g = ops.Graph()
    with g.as_default():
      x = constant_op.constant(1)

      @function.Defun()
      def Foo():
        # [FUNCTION DOC]
        # HELPER: function body under test; runs a loop from zero adding the captured constant until ten.
        return while_loop.while_loop(lambda i: i < 10, lambda i: i + x, [0])

      y = Foo()

    with self.session(graph=g) as sess:
      self.assertEqual(self.evaluate(y), 10)

  @test_util.run_deprecated_v1
  def testCaptureInCond(self):
    # [FUNCTION DOC]
    # Scenario (graph-mode only): a function with a boolean input returns a captured outer value or that
    # value plus one depending on a conditional. Called with true and false; expects one and two.
    g = ops.Graph()
    with g.as_default():
      x = constant_op.constant(1)

      @function.Defun(dtypes.bool)
      def Foo(pred):
        # [FUNCTION DOC]
        # HELPER: function body under test; conditional returning the captured value or the value plus one.
        return tf_cond.cond(pred, lambda: x, lambda: x + 1)

      y = Foo(True)
      z = Foo(False)

    with self.session(graph=g) as sess:
      self.assertEqual(self.evaluate(y), 1)
      self.assertEqual(self.evaluate(z), 2)

  @test_util.run_deprecated_v1
  def testSignatureHash(self):
    # [FUNCTION DOC]
    # Scenario (graph-mode only): two functions each define an inner function with the same body and
    # name but different signatures (one versus three arguments). They must be treated as different
    # functions, and both outer calls must evaluate to twenty.
    # Foo.Inner and Bar.Inner have identical function body but have
    # different signatures. They should be treated as two different functions.

    @function.Defun()
    def Foo(x):
      # [FUNCTION DOC]
      # HELPER: outer function body under test; defines a one-argument inner function and calls it.

      @function.Defun()
      def Inner(x):
        # [FUNCTION DOC]
        # HELPER: inner function with one argument; returns it plus ten.
        return x + 10.

      return Inner(x)

    @function.Defun()
    def Bar(x):
      # [FUNCTION DOC]
      # HELPER: outer function body under test; defines a three-argument inner function and calls it with
      # extra constants.

      @function.Defun()
      def Inner(x, unused_y, unused_z):
        # [FUNCTION DOC]
        # HELPER: inner function with three arguments (two unused); returns the first plus ten.
        return x + 10.

      return Inner(x, 2., 3.)

    g = ops.Graph()
    with g.as_default():
      x = constant_op.constant(10.0)
      y = Foo(x)
      z = Bar(x)

    with self.session(graph=g) as sess:
      v0, v1 = self.evaluate([y, z])
      self.assertAllEqual(v0, 20.)
      self.assertAllEqual(v1, 20.)

  def testShapeFunction(self):
    # [FUNCTION DOC]
    # Scenario: functions declared with custom shape functions. One keeps the input shape; the other
    # prepends a dimension of one. Expects the static result shapes of a vector of length two and of an
    # input of shape one by two by three (stacked) to be shapes two and one by one by two by three
    # respectively.

    @function.Defun(
        dtypes.float32, shape_func=lambda op: [op.inputs[0].get_shape()])
    def Foo(x):
      # [FUNCTION DOC]
      # HELPER: function body under test; returns its input plus one.
      return x + 1.0

    @function.Defun(
        shape_func=lambda op: [[1] + op.inputs[0].get_shape().as_list()])
    def Bar(x):
      # [FUNCTION DOC]
      # HELPER: function body under test; stacks its input into a new leading axis.
      return array_ops_stack.stack([x])

    g = ops.Graph()
    with g.as_default():
      x = Foo([1.0, 2.0])
      self.assertEqual(x.get_shape().as_list(), [2])
      y = Bar(array_ops.zeros([1, 2, 3]))
      self.assertAllEqual(y.get_shape().as_list(), [1, 1, 2, 3])

  @test_util.run_deprecated_v1
  def testVariableReuse(self):
    # [FUNCTION DOC]
    # Scenario (graph-mode only): a function body builds two linear layers where the second reuses the
    # variable of the first. Expects exactly one global variable named with the linear scope prefix and
    # that running with random input of shape thirty-two by one hundred yields that same shape.

    def LinearWithReuse(input_tensor, reuse=None):
      # [FUNCTION DOC]
      # HELPER: linear layer builder; creates or reuses a square weight variable in a scope named linear
      # according to the optional reuse flag and returns the input multiplied by it.
      size = input_tensor.shape.dims[1]
      with variable_scope.variable_scope("linear", reuse=reuse):
        w = variable_scope.get_variable(
            "w", shape=[size, size], dtype=input_tensor.dtype)
      return math_ops.matmul(input_tensor, w)

    @function.Defun(dtypes.float32)
    def Foo(inputs):
      # [FUNCTION DOC]
      # HELPER: function body under test; reshapes input to thirty-two by one hundred and applies the
      # linear layer twice, the second time reusing variables.
      inputs = array_ops.reshape(inputs, [32, 100])
      hidden = LinearWithReuse(inputs)
      return LinearWithReuse(hidden, reuse=True)

    input_op = array_ops.placeholder(shape=[32, 100], dtype=dtypes.float32)
    output_op = Foo(input_op)

    global_vars = variables.global_variables()
    self.assertEqual(len(global_vars), 1)
    self.assertEqual(global_vars[0].name, "linear/w:0")

    with session.Session() as sess:
      self.evaluate(variables.global_variables_initializer())
      output_val = sess.run(
          output_op, feed_dict={input_op: np.random.rand(32, 100)})
      self.assertEqual(output_val.shape, (32, 100))

  @test_util.run_deprecated_v1
  def testFunctionCallInDifferentVariableScopes(self):
    # [FUNCTION DOC]
    # Scenario (graph-mode only): a function creates a variable of ones via a variable getter. It is
    # called under two different variable scopes. Expects only the variable from the first scope to
    # exist, and both results equal the input plus one.

    @function.Defun(dtypes.float32)
    def Foo(inputs):
      # [FUNCTION DOC]
      # HELPER: function body under test; gets a variable of ten ones and returns the input plus it.
      var = variable_scope.get_variable(
          "var",
          shape=[10],
          dtype=dtypes.float32,
          initializer=init_ops.ones_initializer())
      return inputs + var

    input_op = array_ops.placeholder(shape=[10], dtype=dtypes.float32)
    with variable_scope.variable_scope("vs1"):
      out1_op = Foo(input_op)

    with variable_scope.variable_scope("vs2"):
      out2_op = Foo(input_op)

    global_vars = variables.global_variables()
    self.assertEqual(len(global_vars), 1)
    self.assertEqual(global_vars[0].name, "vs1/var:0")

    with session.Session() as sess:
      self.evaluate(variables.global_variables_initializer())
      out1, out2 = sess.run(
          [out1_op, out2_op], feed_dict={input_op: np.linspace(1, 10, 10)})
      self.assertAllEqual(out1, np.linspace(2, 11, 10))
      self.assertAllEqual(out2, np.linspace(2, 11, 10))

  def testTwoInputsSameOp(self):
    # [FUNCTION DOC]
    # Scenario: builds a graph with a singular value decomposition producing three outputs and converts
    # the operations (skipping the placeholder) to a function definition with the three outputs as
    # inputs. Expects the signature to list three input arguments (one per tensor even though they come
    # from the same op).
    g = ops.Graph()
    with g.as_default():
      m = array_ops.placeholder(dtypes.float32)
      s, u, v = linalg_ops.svd(m)
      ss = math_ops.reduce_sum(s)
      uu = math_ops.reduce_sum(u)
      vv = math_ops.reduce_sum(v)
      result = ss + uu + vv
    f = graph_to_function_def.graph_to_function_def(
        g,
        g.get_operations()[1:],  # skip the placeholder
        [s, u, v],
        [result])
    self.assertEqual(len(f.signature.input_arg), 3)

  def testGradientWithIntegerFunctionArgument(self):
    # [FUNCTION DOC]
    # Scenario: a function takes an integer index and a float tensor and returns the indexed element.
    # Differentiating with respect to the float input must work despite the integer argument; with a
    # zero vector input of length two the gradient must be one and zero.

    @function.Defun(dtypes.int32, dtypes.float32)
    def Foo(t, x):
      # [FUNCTION DOC]
      # HELPER: function body under test; indexes the second argument with the first.
      return x[t]

    g = ops.Graph()
    with g.as_default():
      inp = array_ops.placeholder(dtypes.float32)
      t = constant_op.constant(0, dtypes.int32)
      out = Foo(t, inp)
      dinp, = gradients_impl.gradients(out, [inp])

    x = np.zeros((2,)).astype(np.float32)
    with session.Session(graph=g) as sess:
      self.assertAllClose(
          np.array([1.0, 0.0]).astype(np.float32), sess.run(dinp, {inp: x}))

  @test_util.run_deprecated_v1
  def testFunctionMarkedStateful(self):
    # [FUNCTION DOC]
    # Scenario (graph-mode only): two functions are manually marked stateful by setting the stateful
    # flag on their private signatures (a workaround since no public way existed). Calls them and
    # expects values four and one hundred, individually and fetched together.

    @function.Defun(dtypes.int32, dtypes.float32)
    def Foo(t, x):
      # [FUNCTION DOC]
      # HELPER: function body under test; indexes the second argument with the first.
      return x[t]

    @function.Defun(dtypes.int64)
    def Bar(x):
      # [FUNCTION DOC]
      # HELPER: function body under test; returns its 64-bit integer input.
      return x

    # NOTE(mrry): All functions are currently considered stateless by the
    # runtime, so we simulate a "stateful" function.
    # TODO(b/70565970): Remove this hack when we are able to build stateful
    # functions using the API.
    # pylint: disable=protected-access
    Foo._signature.is_stateful = True
    Bar._signature.is_stateful = True
    # pylint: enable=protected-access

    result_1 = Foo(3, [1.0, 2.0, 3.0, 4.0])
    result_2 = Bar(constant_op.constant(100, dtype=dtypes.int64))

    with session.Session() as sess:
      self.assertEqual(4.0, self.evaluate(result_1))
      self.assertEqual(100, self.evaluate(result_2))
      self.assertEqual((4.0, 100), sess.run((result_1, result_2)))

  @test_util.run_deprecated_v1
  def testStatefulFunction(self):
    # [FUNCTION DOC]
    # Scenario (graph-mode only): the stateful flag must propagate from ops to the function and to
    # functions that call it. Functions with a stateless op, a random op and calls to each are defined;
    # only those involving the random op must be flagged stateful. Two calls of the stateful function
    # must give different values, and values must change across repeated runs, under every optimizer
    # configuration.

    @function.Defun()
    def FunctionWithStatelessOp():
      # [FUNCTION DOC]
      # HELPER: function body under test; returns a constant.
      return constant_op.constant(42.0)

    @function.Defun()
    def FunctionWithStatefulOp():
      # [FUNCTION DOC]
      # HELPER: function body under test; returns a hundred random integers below ten (a stateful op).
      return random_ops.random_uniform([100], maxval=10, dtype=dtypes.int32)

    @function.Defun()
    def FunctionWithStatelessFunctionCall():
      # [FUNCTION DOC]
      # HELPER: function body under test; calls the stateless function.
      return FunctionWithStatelessOp()

    @function.Defun()
    def FunctionWithStatefulFunctionCall():
      # [FUNCTION DOC]
      # HELPER: function body under test; calls the stateful function.
      return FunctionWithStatefulOp()

    # Test that the `is_stateful` bit is propagated.
    self.assertFalse(FunctionWithStatelessOp.definition.signature.is_stateful)
    self.assertTrue(FunctionWithStatefulOp.definition.signature.is_stateful)
    self.assertFalse(
        FunctionWithStatelessFunctionCall.definition.signature.is_stateful)
    self.assertTrue(
        FunctionWithStatefulFunctionCall.definition.signature.is_stateful)

    # Ensure that two invocations of the same random-number-generating
    # function produce different results.
    result1 = FunctionWithStatefulFunctionCall()
    result2 = FunctionWithStatefulFunctionCall()

    # Statefulness affects how the function is treated by the various
    # optimization passes, so run the test in each optimizer
    # configuration.
    for config in _OptimizerOptions():
      with session.Session(config=config) as sess:
        val1, val2 = sess.run((result1, result2))
        self.assertFalse(all(val1 == val2))
        val3, val4 = sess.run((result1, result2))
        self.assertFalse(all(val3 == val1))
        self.assertFalse(all(val4 == val2))

  @test_util.run_v1_only("currently failing on v2")
  def testStatefulFunctionWithAllowlisting(self):
    # [FUNCTION DOC]
    # Scenario (graph-mode only; skipped under the second API version): a function that captures a
    # random tensor by value must fail with a value error "Cannot capture a stateful node". When the
    # random op is allowlisted in the declaration, the capture succeeds and each element of the result
    # must be at least three.
    t = random_ops.random_uniform([100], maxval=10, dtype=dtypes.int32)

    @function.Defun(capture_by_value=True)
    def StatefulFn():
      # [FUNCTION DOC]
      # HELPER: function body under test; adds three to the captured random tensor, captured by value.
      return t + constant_op.constant(3, dtype=dtypes.int32)

    # First time we try to capture a stateful RandomUniform op.
    with self.assertRaisesRegex(ValueError, "Cannot capture a stateful node"):
      res = StatefulFn()

    # This time we allowlist this op, so that its recreated.
    @function.Defun(capture_by_value=True, allowlisted_stateful_ops=set([t.op]))
    def StatefulFn2():
      # [FUNCTION DOC]
      # HELPER: function body under test; same as the previous function but its stateful op is on the
      # allowlist.
      return t + constant_op.constant(3, dtype=dtypes.int32)

    res = StatefulFn2()
    with session.Session() as sess:
      r = sess.run(res)
      for i in r:
        self.assertGreaterEqual(i, 3)

  @test_util.run_deprecated_v1
  def testSameFunctionOnTwoDevices(self):
    # [FUNCTION DOC]
    # Scenario (graph-mode only): the same function is called on two different CPU devices. For each
    # optimizer configuration with two CPU devices, expects results 42 and 44 individually and
    # together. Mutates the device count of each generated configuration.

    @function.Defun(dtypes.float32)
    def AddOne(x):
      # [FUNCTION DOC]
      # HELPER: function body under test; returns its input plus one.
      return x + 1.0

    with ops.device("/cpu:0"):
      f_0 = AddOne(41.0)

    with ops.device("/cpu:1"):
      f_1 = AddOne(43.0)

    for config in _OptimizerOptions():
      config.device_count["CPU"] = 2
      with session.Session(config=config) as sess:
        self.assertEqual(42.0, self.evaluate(f_0))
        self.assertEqual(44.0, self.evaluate(f_1))
        self.assertEqual((42.0, 44.0), sess.run((f_0, f_1)))

  @test_util.run_deprecated_v1
  def testGuaranteedConstsAreCaptured(self):
    # [FUNCTION DOC]
    # Scenario (graph-mode only): captured tensors that are (or derive through identities from) a
    # guaranteed-constant op must be captured as guaranteed constants while sums and placeholders are
    # not. Inside the function body the extra args are inspected: first three are consumed by a
    # guarantee-const op, last two are not. Then the function is run on a CPU-only session.
    var = variables.Variable(1.0)
    const = array_ops.guarantee_const(var)
    also_const = array_ops.identity(const)
    still_const = array_ops.identity(also_const)
    not_const = still_const + var
    also_not_const = array_ops.placeholder(dtypes.float32)

    @function.Defun()
    def CapturesGuaranteedConst():
      # [FUNCTION DOC]
      # HELPER: function body under test; sums five captured tensors, asserts which captured extra
      # arguments are consumed by guaranteed-const ops, and returns the sum.
      output = const + also_const + still_const + not_const + also_not_const
      first, second, third, fourth, fifth = function.get_extra_args()
      self.assertEqual("GuaranteeConst", first.consumers()[0].node_def.op)
      self.assertEqual("GuaranteeConst", second.consumers()[0].node_def.op)
      self.assertEqual("GuaranteeConst", third.consumers()[0].node_def.op)
      self.assertNotEqual("GuaranteeConst", fourth.consumers()[0].node_def.op)
      self.assertNotEqual("GuaranteeConst", fifth.consumers()[0].node_def.op)
      return output

    with self.session(use_gpu=False) as sess:
      self.evaluate(var.initializer)
      _ = sess.run(CapturesGuaranteedConst(), {also_not_const: 1.0})

  @test_util.run_deprecated_v1
  def testSameFunctionDifferentGrads(self):
    # [FUNCTION DOC]
    # Scenario (graph-mode only): three same-named functions each doubling the input are given a default
    # gradient, a defined gradient function, and a Python gradient function. At the value one hundred
    # the gradients must be two, one hundred one and fifty.

    def PartOne(x):
      # [FUNCTION DOC]
      # HELPER: builds a doubling function with default gradient and calls it.

      # Default grad is dx = dy * 2
      @function.Defun(dtypes.float32)
      def Foo(x):
        # [FUNCTION DOC]
        # HELPER: function body; returns input doubled; default gradient.
        return x * 2

      return Foo(x)

    def PartTwo(x):
      # [FUNCTION DOC]
      # HELPER: builds a doubling function whose gradient is a defined function and calls it.

      @function.Defun(dtypes.float32, dtypes.float32)
      def Bar(x, dy):
        # [FUNCTION DOC]
        # HELPER: gradient function; returns input plus upstream gradient (deliberately not the real gradient).
        return x + dy  # crazy backprop

      @function.Defun(dtypes.float32, grad_func=Bar)
      def Foo(x):
        # [FUNCTION DOC]
        # HELPER: function body; returns input doubled, with the defined gradient function attached.
        return x * 2

      return Foo(x)

    def PartThree(x):
      # [FUNCTION DOC]
      # HELPER: builds a doubling function whose gradient is a Python gradient function and calls it.

      def Bar(op, dy):
        # [FUNCTION DOC]
        # HELPER: Python gradient function; returns the op's first input times the upstream gradient over two.
        return op.inputs[0] * dy / 2  # crazy backprop

      @function.Defun(dtypes.float32, python_grad_func=Bar)
      def Foo(x):
        # [FUNCTION DOC]
        # HELPER: function body; returns input doubled, with the Python gradient function attached.
        return x * 2

      return Foo(x)

    g = ops.Graph()
    with g.as_default():
      x = constant_op.constant(100.)
      x0 = x
      y0 = PartOne(x0)
      dx0, = gradients_impl.gradients(ys=[y0], xs=[x0])
      x1 = x
      y1 = PartTwo(x1)
      dx1, = gradients_impl.gradients(ys=[y1], xs=[x1])
      x2 = x
      y2 = PartThree(x2)
      dx2, = gradients_impl.gradients(ys=[y2], xs=[x2])

    with self.session(graph=g) as sess:
      v0, v1, v2 = self.evaluate([dx0, dx1, dx2])

    self.assertAllEqual(v0, 2.)
    self.assertAllEqual(v1, 101.)
    self.assertAllEqual(v2, 50.)


class FunctionsFromProtos(test.TestCase):
  # [CLASS DOC]
  # Test case checking that function objects can be reconstructed from function definition messages
  # and function libraries: basic round trips, gradient functions, captured inputs, nested functions,
  # errors for malformed libraries, and custom attributes. Provides two helper methods for comparing
  # function objects.

  def stripInternalFunctionDefAnnotations(self, f_def):
    # [FUNCTION DOC]
    # HELPER: normalizer used when comparing function definitions in this test class.
    # Takes a function definition message and returns a copy with the internal construction-context
    # attribute removed (if present). The input is not modified.
    result = function_pb2.FunctionDef()
    result.CopyFrom(f_def)
    result.attr.pop("_construction_context", None)
    return result

  def expectFunctionsEqual(self, func, grad_func=None, new_func=None):
    # [FUNCTION DOC]
    # HELPER: assertion helper used by several tests in this class.
    # Asserts that a function equals its reconstruction. Inputs: func, the original defined function;
    # grad_func, optional gradient function passed when reconstructing; new_func, optional
    # pre-reconstructed function. If new_func is absent it is made by serializing and parsing the
    # definition of func. Asserts equal names, equal definitions (after stripping internal annotations),
    # equal gradient function names, declared input types and captured inputs. Returns nothing; raises a
    # test failure on mismatch.
    if new_func is None:
      # Make a copy of func.definition to avoid any bugs masked by using the
      # same object
      serialized_fdef = func.definition.SerializeToString()
      # Serialize and then deserialize `func` to create `new_func`
      fdef = function_pb2.FunctionDef.FromString(serialized_fdef)
      new_func = function._from_definition(fdef, grad_func=grad_func)
    self.assertEqual(func.name, new_func.name)
    self.assertEqual(
        self.stripInternalFunctionDefAnnotations(func.definition),
        self.stripInternalFunctionDefAnnotations(new_func.definition))
    self.assertEqual(func.grad_func_name, new_func.grad_func_name)
    self.assertEqual(func.declared_input_types, new_func.declared_input_types)
    self.assertEqual(func.captured_inputs, new_func.captured_inputs)

  @test_util.run_deprecated_v1
  def testBasic(self):
    # [FUNCTION DOC]
    # Scenario (graph-mode only): a function adding two floats is rebuilt from its definition and
    # compared with the original using the equality helper.

    @function.Defun(dtypes.float32, dtypes.float32)
    def Foo(x, y):
      # [FUNCTION DOC]
      # HELPER: function under test; returns the sum of its two arguments.
      return x + y

    self.expectFunctionsEqual(Foo)

  def testGradFunc(self):
    # [FUNCTION DOC]
    # Scenario: a function with a gradient function is rebuilt from its definition, passing the
    # gradient function, and compared with the original.

    @function.Defun(dtypes.float32, dtypes.float32)
    def G(x, dy):
      # [FUNCTION DOC]
      # HELPER: gradient function; returns input times upstream gradient.
      return x * dy

    @function.Defun(dtypes.float32, grad_func=G)
    def F(x):
      # [FUNCTION DOC]
      # HELPER: forward function; returns exp of input minus exp of negative input; uses the gradient above.
      return math_ops.exp(x) - math_ops.exp(-x)

    self.expectFunctionsEqual(F, grad_func=G)

  def testCapturedInputs(self):
    # [FUNCTION DOC]
    # Scenario: a function captures an outer 64-bit integer constant. The rebuilt function must have the
    # same name, definition and gradient name, but the captured input becomes an ordinary declared
    # input (declared types gain the integer type) and there are no captured inputs anymore.
    c = constant_op.constant(10, dtypes.int64)

    @function.Defun(dtypes.int64)
    def Foo(x):
      # [FUNCTION DOC]
      # HELPER: function under test; returns input plus the captured constant.
      return x + c

    new_func = function._from_definition(Foo.definition)

    self.assertEqual(Foo.name, new_func.name)
    self.assertEqual(
        self.stripInternalFunctionDefAnnotations(Foo.definition),
        self.stripInternalFunctionDefAnnotations(new_func.definition))
    self.assertEqual(Foo.grad_func_name, new_func.grad_func_name)

    # Captured inputs are added as regular inputs to the function definition
    self.assertEqual(new_func.declared_input_types,
                     Foo.declared_input_types + (dtypes.int64,))
    self.assertEqual(len(new_func.captured_inputs), 0)

  def testNestedFunctions(self):
    # [FUNCTION DOC]
    # Scenario: an outer function calls an inner function twice. Rebuilding the outer function from its
    # definition must give an equal function.

    @function.Defun(dtypes.float32)
    def Outer(x):
      # [FUNCTION DOC]
      # HELPER: outer function under test; defines an inner function and applies it twice.

      @function.Defun(dtypes.float32)
      def Inner(y):
        # [FUNCTION DOC]
        # HELPER: inner function; returns input plus one.
        return y + 1

      return Inner(Inner(x))

    self.expectFunctionsEqual(Outer)

  def testFromLibrary(self):
    # [FUNCTION DOC]
    # Scenario: six functions (two gradient functions, two forward functions sharing one gradient, one
    # with the other gradient, one with no gradient) are instantiated in a graph and its function library
    # is turned into function objects. Each original must match the corresponding rebuilt function.
    # Define some functions with different gradient functions. Note that many of
    # the below functions are identical since function bodies don't matter for
    # this test.

    @function.Defun(dtypes.float32, dtypes.float32)
    def G1(x, dy):
      # [FUNCTION DOC]
      # HELPER: gradient function one; returns input times upstream gradient.
      return x * dy

    @function.Defun(dtypes.float32, dtypes.float32)
    def G2(x, dy):
      # [FUNCTION DOC]
      # HELPER: gradient function two; identical body to the first.
      return x * dy

    # F1 and F2 have the same gradient function
    @function.Defun(dtypes.float32, grad_func=G1)
    def F1(x):
      # [FUNCTION DOC]
      # HELPER: forward function with the first gradient function.
      return math_ops.exp(x) - math_ops.exp(-x)

    @function.Defun(dtypes.float32, grad_func=G1)
    def F2(x):
      # [FUNCTION DOC]
      # HELPER: forward function sharing the first gradient function.
      return math_ops.exp(x) - math_ops.exp(-x)

    # F3 has a different gradient function
    @function.Defun(dtypes.float32, grad_func=G2)
    def F3(x):
      # [FUNCTION DOC]
      # HELPER: forward function with the second gradient function.
      return math_ops.exp(x) - math_ops.exp(-x)

    # F4 has no gradient function
    @function.Defun(dtypes.float32)
    def F4(x):
      # [FUNCTION DOC]
      # HELPER: forward function with no gradient function.
      return math_ops.exp(x) - math_ops.exp(-x)

    # Instantiate all functions
    g = ops.Graph()
    with g.as_default():
      c = constant_op.constant(1.0, dtypes.float32)
      f1 = F1(c)
      f2 = F2(c)
      f3 = F3(c)
      f4 = F4(c)
      gradients_impl.gradients([f1, f2, f3, f4], c)

    library = g.as_graph_def().library
    new_funcs = function.from_library(library)

    def CheckNewFunc(func):
      # [FUNCTION DOC]
      # HELPER: verification helper; finds exactly one rebuilt function whose name matches the given
      # original and asserts equality using the class comparison helper.
      new_func = [f for f in new_funcs if f.name == func.name]
      self.assertEqual(len(new_func), 1)
      self.expectFunctionsEqual(func, new_func=new_func[0])

    CheckNewFunc(G1)
    CheckNewFunc(G2)
    CheckNewFunc(F1)
    CheckNewFunc(F2)
    CheckNewFunc(F3)
    CheckNewFunc(F4)

  def testFromLibraryEmptyLib(self):
    # [FUNCTION DOC]
    # Scenario: building functions from an empty function library must give an empty list.
    library = function_pb2.FunctionDefLibrary()
    self.assertEqual(len(function.from_library(library)), 0)

  def testFromLibraryMissingFuncDef(self):
    # [FUNCTION DOC]
    # Scenario: a library holds a gradient entry linking two functions but is missing one of the two
    # function definitions, first the gradient and then the forward function. Each case must raise a
    # value error saying the library is missing a function definition with the matching generated name.

    @function.Defun(dtypes.float32, dtypes.float32)
    def G1(x, dy):
      # [FUNCTION DOC]
      # HELPER: gradient function; returns input times upstream gradient.
      return x * dy

    @function.Defun(dtypes.float32)
    def F1(x):
      # [FUNCTION DOC]
      # HELPER: forward function; exp of input minus exp of negative input.
      return math_ops.exp(x) - math_ops.exp(-x)

    gradient = function_pb2.GradientDef()
    gradient.function_name = F1.name
    gradient.gradient_func = G1.name

    # Create invalid function def that is missing G1 function def
    library = function_pb2.FunctionDefLibrary()
    library.gradient.extend([gradient])
    library.function.extend([F1.definition])

    with self.assertRaisesRegex(
        ValueError,
        "FunctionDefLibrary missing 'G1_[0-9a-zA-Z]{8,11}' FunctionDef"):
      function.from_library(library)

    # Create invalid function def that is missing F1 function def
    library = function_pb2.FunctionDefLibrary()
    library.gradient.extend([gradient])
    library.function.extend([G1.definition])

    with self.assertRaisesRegex(
        ValueError,
        "FunctionDefLibrary missing 'F1_[0-9a-zA-Z]{8,11}' FunctionDef"):
      function.from_library(library)

  def testFromLibraryCyclicGradFuncs(self):
    # [FUNCTION DOC]
    # Scenario: a library where function one has function two as its gradient and vice versa. Building
    # functions from it must raise a value error that the library contains cyclic gradient functions.

    @function.Defun(dtypes.float32)
    def F1(x):
      # [FUNCTION DOC]
      # HELPER: forward function with a hyperbolic-sine-like body.
      return math_ops.exp(x) - math_ops.exp(-x)

    @function.Defun(dtypes.float32)
    def F2(x):
      # [FUNCTION DOC]
      # HELPER: forward function with the same body as the first.
      return math_ops.exp(x) - math_ops.exp(-x)

    # Create invalid function def library where F1 has gradient function F2 and
    # F2 has gradient function F1
    library = function_pb2.FunctionDefLibrary()
    library.function.extend([F1.definition, F2.definition])

    gradient1 = function_pb2.GradientDef()
    gradient1.function_name = F1.name
    gradient1.gradient_func = F2.name

    gradient2 = function_pb2.GradientDef()
    gradient2.function_name = F2.name
    gradient2.gradient_func = F1.name

    library.gradient.extend([gradient1, gradient2])

    with self.assertRaisesRegex(
        ValueError, "FunctionDefLibrary contains cyclic gradient functions!"):
      function.from_library(library)

  def testExperimentalAttrs(self):
    # [FUNCTION DOC]
    # Scenario: functions declared with an experimental tag of string, integer, float and boolean
    # values. Expects each definition's attribute map to contain the tag, stored in the matching typed
    # field (bytes, integer, float, bool) with the same value.

    @function.Defun(dtypes.int32, experimental_tag="tag_value")
    def FunctionWithStrAttr(i):
      # [FUNCTION DOC]
      # HELPER: function under test; identity of its integer input with a string tag.
      return array_ops.identity(i)

    @function.Defun(dtypes.int32, experimental_tag=123)
    def FunctionWithIntAttr(i):
      # [FUNCTION DOC]
      # HELPER: function under test; identity of its integer input with an integer tag.
      return array_ops.identity(i)

    @function.Defun(dtypes.int32, experimental_tag=123.0)
    def FunctionWithFloatAttr(i):
      # [FUNCTION DOC]
      # HELPER: function under test; identity of its integer input with a float tag.
      return array_ops.identity(i)

    @function.Defun(dtypes.int32, experimental_tag=True)
    def FunctionWithBoolAttr(i):
      # [FUNCTION DOC]
      # HELPER: function under test; identity of its integer input with a boolean tag.
      return array_ops.identity(i)

    self.assertTrue("experimental_tag" in FunctionWithStrAttr.definition.attr)
    self.assertEqual(FunctionWithStrAttr.definition.attr["experimental_tag"].s,
                     b"tag_value")
    self.assertTrue("experimental_tag" in FunctionWithIntAttr.definition.attr)
    self.assertEqual(FunctionWithIntAttr.definition.attr["experimental_tag"].i,
                     123)
    self.assertTrue("experimental_tag" in FunctionWithFloatAttr.definition.attr)
    self.assertEqual(
        FunctionWithFloatAttr.definition.attr["experimental_tag"].f, 123.0)
    self.assertTrue("experimental_tag" in FunctionWithBoolAttr.definition.attr)
    self.assertEqual(FunctionWithBoolAttr.definition.attr["experimental_tag"].b,
                     True)

  def testImplementsReferenceAttrs(self):
    # [FUNCTION DOC]
    # Scenario: a function declared with the implements and reference attributes. Expects the definition
    # to contain both attributes holding the given strings.

    @function.Defun(
        dtypes.int32, _implements="org.google.lstm", _reference="arxiv.org")
    def FunctionWithStrAttr(i):
      # [FUNCTION DOC]
      # HELPER: function under test; identity of its integer input.
      return array_ops.identity(i)

    self.assertIn("_implements", FunctionWithStrAttr.definition.attr)
    self.assertEqual(FunctionWithStrAttr.definition.attr["_implements"].s,
                     b"org.google.lstm")
    self.assertIn("_reference", FunctionWithStrAttr.definition.attr)
    self.assertEqual(FunctionWithStrAttr.definition.attr["_reference"].s,
                     b"arxiv.org")


class FunctionOverloadTest(test.TestCase):
  # [CLASS DOC]
  # Test case checking that one defined function can be instantiated for several input dtypes
  # (overloading), including gradients and doc string propagation.

  @test_util.run_deprecated_v1
  def testBasic(self):
    # [FUNCTION DOC]
    # Scenario (graph-mode only): a hyperbolic-sine function with no declared types is called with 32-bit
    # and 64-bit floats at 0.25; both results must match the numpy value.

    @function.Defun()
    def Sinh(x):
      # [FUNCTION DOC]
      # HELPER: function under test; one half times exp of input minus exp of negative input.
      return 1 / 2. * (math_ops.exp(x) - math_ops.exp(-x))

    g = ops.Graph()
    with g.as_default():
      x = Sinh(constant_op.constant(0.25, dtypes.float32))
      y = Sinh(constant_op.constant(0.25, dtypes.float64))

    with self.session(graph=g):
      self.assertAllClose(x, np.sinh(0.25))
      self.assertAllClose(y, np.sinh(0.25))

  def testGradient(self):
    # [FUNCTION DOC]
    # Scenario: a function with a gradient function with an explicit name is applied to 32-bit and 64-bit
    # floats; the gradient at 0.25 must be 0.25 for both, since the gradient function returns input times
    # the upstream gradient.

    @function.Defun(func_name="Spec")
    def G(x, dy):
      # [FUNCTION DOC]
      # HELPER: gradient function (named Spec); returns input times upstream gradient.
      return x * dy

    @function.Defun(grad_func=G)
    def F(x):
      # [FUNCTION DOC]
      # HELPER: forward function with the gradient function attached.
      return math_ops.exp(x) - math_ops.exp(-x)

    for dtype in [dtypes.float32, dtypes.float64]:
      g = ops.Graph()
      with g.as_default():
        x = constant_op.constant(0.25, dtype)
        y = F(x)
        dx, = gradients_impl.gradients(y, x)

        with self.session(graph=g):
          self.assertAllClose(dx, 0.25)

  def testDocString(self):
    # [FUNCTION DOC]
    # Scenario: the doc string of a function body must become the description of the signature in the
    # graph's function library.

    @function.Defun()
    def Foo(x):
      # [FUNCTION DOC]
      # HELPER: function under test; has a one-line doc string and returns input plus one.
      """Successor of x."""
      return x + 1

    g = ops.Graph()
    with g.as_default():
      _ = Foo(1)

    self.assertEqual(g.as_graph_def().library.function[0].signature.description,
                     "Successor of x.")


class FunctionCaptureByValueTest(test.TestCase):
  # [CLASS DOC]
  # Test case for functions declared to capture outer tensors by value (copying them into the
  # function) rather than as extra inputs.

  @test_util.run_deprecated_v1
  def testCaptureByValue(self):
    # [FUNCTION DOC]
    # Scenario (graph-mode only): an outer function and a nested function capture constants by value.
    # Expects the nested and outer functions to have zero captured inputs and the result to be twelve.
    g = ops.Graph()
    with g.as_default():
      w = constant_op.constant([[1.0]])
      b = constant_op.constant([2.0])

      # Foo() captures w and b.
      @function.Defun(dtypes.float32, capture_by_value=True)
      def Foo(x):
        # [FUNCTION DOC]
        # HELPER: outer function body under test; asserts nested function has no captured inputs and applies
        # it to the matrix product.

        # Plus() captures b.
        @function.Defun(dtypes.float32, capture_by_value=True)
        def Plus(y):
          # [FUNCTION DOC]
          # HELPER: nested function body under test; returns input plus the captured bias.
          return y + b

        self.assertEqual(0, len(Plus.captured_inputs))

        return Plus(math_ops.matmul(w, x))

      y = Foo(constant_op.constant([[10.]]))

    self.assertEqual(0, len(Foo.captured_inputs))

    with self.session(graph=g):
      self.assertAllEqual(y, [[12.0]])


@test_util.run_all_without_tensor_float_32(
    "Calls matmul in custom LSTM function")
class UnrollLSTMTest(test.TestCase):
  # [CLASS DOC]
  # Test case comparing an unrolled LSTM built in several ways (plain graph, cell as function, whole
  # loop as function, ten steps per function) for equal forward results and gradients under all
  # optimizer configurations. Class constants set batch size (sixteen), LSTM width (thirty-two) and
  # unroll count (twenty). Decorated to run without reduced-precision matrix multiplication.
  BATCH_SIZE = 16
  LSTM_DIMS = 32
  NUM_UNROLL = 20

  def _Weights(self):
    # [FUNCTION DOC]
    # HELPER: fixture producing a fixed-seed random weight matrix of shape two times width by four times
    # width, uniform in minus one to one.
    dims = self.LSTM_DIMS
    return random_ops.random_uniform([2 * dims, 4 * dims], -1, 1, seed=123456)

  def _Input(self):
    # [FUNCTION DOC]
    # HELPER: fixture producing a fixed-seed random input tensor shaped unroll count by batch size by width.
    return random_ops.random_uniform(
        [self.NUM_UNROLL, self.BATCH_SIZE, self.LSTM_DIMS], seed=654321)

  # Helper to construct a LSTM cell graph.
  @classmethod
  def LSTMCell(cls, x, mprev, cprev, weights):
    # [FUNCTION DOC]
    # HELPER: builds one LSTM cell step; class-level method (not instance).
    # Takes input, previous output, previous cell state and weights. Concatenates input and previous
    # output, multiplies by weights, splits into four gates, computes new cell state from forget and input
    # gates (clipped to minus fifty to fifty) and new output from the output gate. Returns the new output
    # and cell state.
    xm = array_ops.concat([x, mprev], 1)
    i_i, i_g, f_g, o_g = array_ops.split(
        value=math_ops.matmul(xm, weights), num_or_size_splits=4, axis=1)
    new_c = math_ops.sigmoid(f_g) * cprev + math_ops.sigmoid(
        i_g) * math_ops.tanh(i_i)
    new_c = math_ops.maximum(math_ops.minimum(new_c, 50.0), -50.0)
    new_m = math_ops.sigmoid(o_g) * math_ops.tanh(new_c)
    return new_m, new_c

  def _BuildForward(self, weights, inp, mode="cell"):
    # [FUNCTION DOC]
    # HELPER: builds the forward LSTM graph in one of four modes for the tests below.
    # KNOWN ISSUE: for a mode other than the four known names (complete, cell, loop, loop10) it falls
    # through and returns nothing instead of raising an error.
    # Inputs: weights, input tensor and mode string (default "cell"). The complete mode loops in plain
    # graph code; cell wraps the cell as a function; loop wraps the whole loop as a function; loop10
    # wraps groups of ten steps in a function and the loop in another. Returns the final output tensor.

    def Loop(cell, w, i):
      # [FUNCTION DOC]
      # HELPER: unrolls the recurrence; given a cell callable, weights and a time-major input, splits the
      # input along time, starts from zero output and state and applies the cell once per step. Returns the
      # last output.
      x = array_ops_stack.unstack(i, self.NUM_UNROLL)
      m = array_ops.zeros_like(x[0])
      c = array_ops.zeros_like(x[0])
      for i in range(self.NUM_UNROLL):
        m, c = cell(x[i], m, c, w)
      return m

    cell = UnrollLSTMTest.LSTMCell
    if mode == "complete":
      # Constructs the complete graph in python.
      return Loop(cell, weights, inp)

    cell = function.Defun(dtypes.float32, dtypes.float32, dtypes.float32,
                          dtypes.float32)(
                              cell)
    if mode == "cell":
      # Just represent the LSTM as a function.
      return Loop(cell, weights, inp)

    if mode == "loop":
      # Wraps the whole loop as a function.
      @function.Defun(dtypes.float32, dtypes.float32)
      def LSTMLoop(w, i):
        # [FUNCTION DOC]
        # HELPER: function body wrapping the whole unrolled loop.
        return Loop(cell, w, i)

      return LSTMLoop(weights, inp)

    if mode == "loop10":
      # Wraps 10 lstm steps into one function, and the whole loop
      # into another calling the formers.

      # Groups 10 steps at a time.
      @function.Defun(dtypes.float32, dtypes.float32, dtypes.float32,
                      *([dtypes.float32] * 10))
      def Loop10(w, m, c, *args):
        # [FUNCTION DOC]
        # HELPER: function body that applies the cell to each of ten step inputs in order, starting from
        # given output and state, returning final output and state.
        for x in args:
          m, c = cell(x, m, c, w)
        return m, c

      @function.Defun(dtypes.float32, dtypes.float32)
      def LSTMLoop10(weights, inp):
        # [FUNCTION DOC]
        # HELPER: function body that unstacks the input, asserts the unroll count is divisible by ten and
        # applies the ten-step function repeatedly. Returns the final output.
        x = array_ops_stack.unstack(inp, self.NUM_UNROLL)
        m = array_ops.zeros_like(x[0])
        c = array_ops.zeros_like(x[0])
        assert self.NUM_UNROLL % 10 == 0
        for i in range(0, self.NUM_UNROLL, 10):
          m, c = Loop10(weights, m, c, *x[i:i + 10])
        return m

      return LSTMLoop10(weights, inp)

  def testUnrollLSTM(self):
    # [FUNCTION DOC]
    # Scenario: runs the forward pass in all four modes; results from the three function-based modes
    # must match the plain graph within relative tolerance, under every optimizer configuration.
    # Run one step of the unrolled lstm graph.
    def RunForward(mode, cfg=None):
      # [FUNCTION DOC]
      # HELPER: builds a graph in the given mode (plus optional session configuration), logs the time and
      # graph sizes, evaluates in a session and returns the numeric result.
      tf_logging.info("mode = %s", mode)
      g = ops.Graph()
      start = time.time()
      with g.as_default():
        weights = self._Weights()
        inp = self._Input()
        m = self._BuildForward(weights, inp, mode)
      gdef = g.as_graph_def()
      finish = time.time()
      tf_logging.info("time: %f txt size: %d gdef bin size: %d", finish - start,
                      len(str(gdef)), len(gdef.SerializeToString()))
      with g.as_default(), session.Session(config=cfg) as sess:
        return self.evaluate(m)

    mv0 = RunForward("complete")
    for cfg in _OptimizerOptions():
      tf_logging.info("cfg = %s", cfg)
      mv1 = RunForward("cell", cfg)
      mv2 = RunForward("loop", cfg)
      mv3 = RunForward("loop10", cfg)
      self.assertAllClose(mv0, mv1, rtol=1e-4)
      self.assertAllClose(mv0, mv2, rtol=1e-4)
      self.assertAllClose(mv0, mv3, rtol=1e-4)

  def testUnrollLSTMGrad(self):
    # [FUNCTION DOC]
    # Scenario: like the previous test, but compares gradients of the sum of squares of the output with
    # respect to the weights across the four modes with relative and absolute tolerance.
    # Run one step of the unrolled lstm graph.
    def RunForwardBackward(mode, cfg=None):
      # [FUNCTION DOC]
      # HELPER: builds the graph in the given mode, computes the weights gradient of a squared-sum loss,
      # logs time and sizes, evaluates it in a session (optional configuration) and returns the result.
      tf_logging.info("mode = %s", mode)
      g = ops.Graph()
      start = time.time()
      with g.as_default():
        weights = self._Weights()
        inp = self._Input()
        m = self._BuildForward(weights, inp, mode)
        loss = math_ops.reduce_sum(math_ops.square(m))
        dw = gradients_impl.gradients([loss], [weights])
      gdef = g.as_graph_def()
      finish = time.time()
      tf_logging.info("time: %f txt size: %d gdef bin size: %d", finish - start,
                      len(str(gdef)), len(gdef.SerializeToString()))
      with g.as_default(), session.Session(config=cfg) as sess:
        return self.evaluate(dw)

    d0 = RunForwardBackward("complete")
    for cfg in _OptimizerOptions():
      tf_logging.info("cfg = %s", cfg)
      d1 = RunForwardBackward("cell", cfg)
      d2 = RunForwardBackward("loop", cfg)
      d3 = RunForwardBackward("loop10", cfg)
      self.assertAllClose(d0, d1, rtol=1e-4, atol=1e-4)
      self.assertAllClose(d0, d2, rtol=1e-4, atol=1e-4)
      self.assertAllClose(d0, d3, rtol=1e-4, atol=1e-4)


class FunctionInlineControlTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case checking that the declared no-inline option controls whether function
  # calls remain in the executed graph.

  @parameterized.parameters((True), (False))
  @test_util.disable_xla("XLA changes the names, breaking graph analysis")
  def testFoo(self, noinline):
    # [FUNCTION DOC]
    # Scenario: parameterized over the no-inline flag (true or false); disabled under XLA. A cell
    # function (flagged accordingly) is applied ten times inside another function and differentiated.
    # Runs with inlining enabled and full tracing and expects values about 255.971 and gradient sum about
    # 13.0408. Unless built with ROCm, the cell definition's no-inline attribute must equal the flag.
    # Finally, the trace metadata must mention the cell function call exactly when the flag is true.
    dtype = dtypes.float32
    cfg = config_pb2.ConfigProto(
        graph_options=config_pb2.GraphOptions(
            optimizer_options=config_pb2.OptimizerOptions(
                opt_level=config_pb2.OptimizerOptions.L0,
                do_common_subexpression_elimination=True,
                do_function_inlining=True,
                do_constant_folding=True)))
    cell_func_call_pattern = re.compile(r"Cell[^/]*\(")
    @function.Defun(dtype, noinline=noinline)
    def Cell(v):
      # [FUNCTION DOC]
      # HELPER: function body under test; hyperbolic tangent of the sum of input and its transpose, then
      # reduce-sum over axis one keeping the dimension.
      # If v is a vector [n, 1], x is a big square matrix.
      x = math_ops.tanh(v + array_ops.transpose(v, [1, 0]))
      return math_ops.reduce_sum(x, 1, keepdims=True)

    @function.Defun(dtype)
    def Forward(x):
      # [FUNCTION DOC]
      # HELPER: function body under test; applies the cell ten times then sums everything.
      for _ in range(10):
        # pylint: disable=cell-var-from-loop
        x = Cell(x)
      return math_ops.reduce_sum(x, [0, 1])

    # Disabling this check on the ROCm platform, because it fails
    # The failure might not be ROCm specific(see commit message for details)
    if not test.is_built_with_rocm():
      self.assertEqual(noinline, Cell.definition.attr["_noinline"].b)

    g = ops.Graph()
    with g.as_default():
      x = array_ops.placeholder(dtype)
      y = Forward(x)
      dx, = gradients_impl.gradients([y], [x])

    np.random.seed(321)
    inp = np.random.uniform(-1, 1, [16, 1]).astype(np.float32)
    run_metadata = config_pb2.RunMetadata()
    with session.Session(graph=g, config=cfg) as sess:
      ans = sess.run(
          [y, dx], {x: inp},
          run_metadata=run_metadata,
          options=config_pb2.RunOptions(
              trace_level=config_pb2.RunOptions.FULL_TRACE))
      self.assertAllClose(ans[0], 255.971, rtol=1e-3)
      self.assertAllClose(np.sum(ans[1]), 13.0408, rtol=1e-3)

    def MetadataHasCell(run_metadata):
      # [FUNCTION DOC]
      # HELPER: returns true when any node in the run metadata has a timeline label that matches a call to
      # the cell function; otherwise false.
      for dev_stats in run_metadata.step_stats.dev_stats:
        for node_stats in dev_stats.node_stats:
          if cell_func_call_pattern.search(node_stats.timeline_label):
            return True
      return False

    self.assertEqual(MetadataHasCell(run_metadata), noinline)


class ModuleFunctionTest(test.TestCase):
  # [CLASS DOC]
  # Test case for functions that call other defined functions.

  @test_util.run_deprecated_v1
  def testBasic(self):
    # [FUNCTION DOC]
    # Scenario (graph-mode only): a linear-plus-relu function and a second function calling it twice
    # are evaluated on one-by-one matrices of values zero through four; the results must be one and five.

    @function.Defun(*[dtypes.float32] * 3)
    def LinearWithCApi(w, b, x):
      # [FUNCTION DOC]
      # HELPER: function body under test; relu of the matrix product plus bias.
      return nn_ops.relu(math_ops.matmul(x, w) + b)

    @function.Defun(*[dtypes.float32] * 5)
    def Linear2WithCApi(w1, b1, w2, b2, x):
      # [FUNCTION DOC]
      # HELPER: function body under test; applies the linear function twice, first with the first weight
      # pair and then with the second.
      return LinearWithCApi(w2, b2, LinearWithCApi(w1, b1, x))

    with ops.Graph().as_default():
      a, b, c, d, e = [
          constant_op.constant([[_]], dtype=dtypes.float32) for _ in range(5)
      ]
      y = LinearWithCApi(a, b, c)
      z = Linear2WithCApi(a, b, c, d, e)
      with session.Session() as sess:
        self.assertAllEqual([[1]], self.evaluate(y))
        self.assertAllEqual([[5]], self.evaluate(z))


class VariableHoistingTest(test.TestCase):
  # [CLASS DOC]
  # Test case checking that variables created inside functions are hoisted to the outer graph and
  # exposed as extra function arguments usable for gradients.

  def _testSimpleModel(self, use_forward_func, use_resource=False):
    # [FUNCTION DOC]
    # HELPER: shared scenario for the tests below.
    # Inputs: use_forward_func (true calls the model through a function, false calls it directly) and
    # use_resource (optional, default false; chooses resource variables). Builds a model under a
    # variable scope, computes a loss and gradients inside another function using the hoisted variables'
    # extra arguments, and expects two hoisted variables with names derived from the scope, correct
    # shapes and the numeric sums and losses stated in the assertions.

    def _Model(x):
      # [FUNCTION DOC]
      # HELPER: model body; creates a weight matrix and a bias variable and returns the sigmoid of the
      # matrix product plus the bias.
      # KNOWN ISSUE: a trailing comma after the bias variable creation makes the bias a one-element tuple
      # rather than the variable itself; it still broadcasts in the addition, hiding the problem.
      w = variable_scope.get_variable(
          "w", (64, 64),
          initializer=init_ops.random_uniform_initializer(seed=312),
          use_resource=use_resource)
      b = variable_scope.get_variable(
          "b", (64),
          initializer=init_ops.zeros_initializer(),
          use_resource=use_resource),
      return math_ops.sigmoid(math_ops.matmul(x, w) + b)

    @function.Defun()
    def Model(x):
      # [FUNCTION DOC]
      # HELPER: function wrapper calling the model body.
      return _Model(x)

    cvars = []

    @function.Defun()
    def Grad(x, y0):
      # [FUNCTION DOC]
      # HELPER: function body that computes the model output (through the function or directly), a mean
      # cross-entropy-style loss against the second input, checks the hoisted extra arguments' shapes,
      # takes gradients with respect to them, records the extra variables in an outer list and returns
      # loss and both gradients.
      if use_forward_func:
        y = Model(x)
      else:
        y = _Model(x)
      loss = math_ops.reduce_mean(
          math_ops.reduce_sum(y0 * math_ops.log(y), 1), 0)
      arg_w, arg_b = function.get_extra_args()
      self.assertEqual(arg_w.get_shape(), tensor_shape.TensorShape([64, 64]))
      self.assertEqual(arg_b.get_shape(), tensor_shape.TensorShape([64]))
      dw, db = gradients_impl.gradients(loss, [arg_w, arg_b])
      cvars.extend(function.get_extra_vars())
      return loss, dw, db

    g = ops.Graph()
    with g.as_default():
      x = random_ops.random_normal([64, 64], seed=100)
      y0 = random_ops.random_normal([64, 64], seed=200)
      with variable_scope.variable_scope("Foo"):
        loss, dw, db = Grad(x, y0)

    self.assertEqual(2, len(cvars))
    w, b = cvars[:2]
    self.assertEqual("Foo/w", w.op.name)
    self.assertEqual("Foo/b", b.op.name)

    with self.session(graph=g) as sess:
      self.evaluate(variables.global_variables_initializer())
      w, b, x, y0, loss, dw, db = self.evaluate([w, b, x, y0, loss, dw, db])

    self.assertAllEqual(w.shape, (64, 64))
    self.assertAllClose(np.sum(w), 2050.44)
    self.assertAllEqual(b.shape, (64,))
    self.assertAllClose(np.sum(b), 0.0)
    self.assertAllClose(loss, -2.27, rtol=1e-2)
    self.assertAllEqual(dw.shape, (64, 64))
    self.assertAllClose(np.sum(dw), -1.04, rtol=1e-2)
    self.assertAllEqual(db.shape, (64,))
    self.assertAllClose(np.sum(db), 0.509, rtol=1e-2)

  @test_util.run_deprecated_v1
  def testBasic(self):
    # [FUNCTION DOC]
    # Scenario (graph-mode only): runs the shared scenario with reference variables, with and without the
    # forward function.
    self._testSimpleModel(False)
    self._testSimpleModel(True)

  @test_util.run_deprecated_v1
  def testBasicResource(self):
    # [FUNCTION DOC]
    # Scenario (graph-mode only): runs the shared scenario with resource variables, with and without the
    # forward function.
    self._testSimpleModel(False, use_resource=True)
    self._testSimpleModel(True, use_resource=True)


class TemplateTest(test.TestCase):
  # [CLASS DOC]
  # Test case checking that variable sharing via make_template works across a defined function and
  # direct calls, in either order and with resource or reference variables. Only runs in graph mode.

  @test_util.run_v1_only("make_template not supported in TF2")
  def testBasic(self):
    # [FUNCTION DOC]
    # Scenario (graph-mode only): resource variables, direct call first, function second.
    self.assertTemplateVariableSharing(use_resource=True, defun_first=False)

  @test_util.run_v1_only("make_template not supported in TF2")
  def testBasicRef(self):
    # [FUNCTION DOC]
    # Scenario (graph-mode only): reference variables, direct call first.
    self.assertTemplateVariableSharing(use_resource=False, defun_first=False)

  @test_util.run_v1_only("make_template not supported in TF2")
  def testBasicDefunFirst(self):
    # [FUNCTION DOC]
    # Scenario (graph-mode only): resource variables, function call first.
    self.assertTemplateVariableSharing(use_resource=True, defun_first=True)

  @test_util.run_v1_only("make_template not supported in TF2")
  def testBasicRefDefunFirst(self):
    # [FUNCTION DOC]
    # Scenario (graph-mode only): reference variables, function call first.
    self.assertTemplateVariableSharing(use_resource=False, defun_first=True)

  def assertTemplateVariableSharing(self, use_resource, defun_first):
    # [FUNCTION DOC]
    # HELPER: shared assertion routine for the tests above.
    # Inputs: use_resource (resource versus reference variables) and defun_first (call through function
    # before direct call). Builds a template model creating weight and bias, calls the model directly and
    # through a function in the chosen order, and asserts that the weight and bias created by the second
    # call are the same objects as those from the first.
    parameters = []

    def MakeModel(x):
      # [FUNCTION DOC]
      # HELPER: model body for the template; creates weight and bias variables, records them in a list and
      # returns the sigmoid of the matrix product plus bias.
      w = variable_scope.get_variable(
          "w", (64, 64),
          initializer=init_ops.random_uniform_initializer(seed=312),
          use_resource=use_resource)
      b = variable_scope.get_variable(
          "b", (64),
          initializer=init_ops.zeros_initializer(),
          use_resource=use_resource)
      parameters.extend((w, b))
      return math_ops.sigmoid(math_ops.matmul(x, w) + b)

    model = template.make_template("f", MakeModel, create_scope_now_=True)

    @function.Defun()
    def ModelDefun(x):
      # [FUNCTION DOC]
      # HELPER: function body; calls the template model.
      return model(x)

    x = array_ops.placeholder(dtypes.float32)
    if defun_first:
      ModelDefun(x)
      model(x)
    else:
      model(x)
      ModelDefun(x)
    w1, b1, w2, b2 = parameters  # pylint: disable=unbalanced-tuple-unpacking
    self.assertIs(w1, w2)
    self.assertIs(b1, b2)


class DevicePlacementTest(test.TestCase):
  # [CLASS DOC]
  # Test case checking which device is recorded on nodes inside function definitions.

  def testNoDeviceGraph(self):
    # [FUNCTION DOC]
    # Scenario: a function defined and called without a device scope. Expects exactly one function in the
    # graph library and an empty device string on every node.
    with ops.Graph().as_default():

      @function.Defun(*[dtypes.float32] * 2)
      def Matmul(a, b):
        # [FUNCTION DOC]
        # HELPER: function body under test; matrix product of its two arguments.
        return math_ops.matmul(a, b)

      Matmul(1., 2.)

      gdef = ops.get_default_graph().as_graph_def()
      self.assertAllEqual(len(gdef.library.function), 1)
      fdef = gdef.library.function[0]

      for node in fdef.node_def:
        self.assertAllEqual(node.device, "")

  def testNestedDevices(self):
    # [FUNCTION DOC]
    # Scenario: a matrix-product function defined under one CPU device scope and a divide function defined
    # inside a nested second CPU scope. Expects nodes of the first to record the first CPU and nodes of
    # the second the second CPU.
    with ops.Graph().as_default(), ops.device("CPU:0"):

      @function.Defun(*[dtypes.float32] * 2)
      def Matmul(a, b):
        # [FUNCTION DOC]
        # HELPER: function body under test; matrix product.
        return math_ops.matmul(a, b)

      with ops.device("CPU:1"):

        @function.Defun(*[dtypes.float32] * 2)
        def Divide(a, b):
          # [FUNCTION DOC]
          # HELPER: function body under test; divides first argument by the second.
          return math_ops.divide(a, b)

        Divide(Matmul(1., 2.), 3.)

      gdef = ops.get_default_graph().as_graph_def()
      matmul_fdef = [
          f for f in gdef.library.function if "Matmul" in f.signature.name
      ]
      divide_fdef = [
          f for f in gdef.library.function if "Divide" in f.signature.name
      ]
      self.assertAllEqual(len(matmul_fdef), 1)
      self.assertAllEqual(len(divide_fdef), 1)
      for node in matmul_fdef[0].node_def:
        self.assertAllEqual(node.device, "/device:CPU:0")
      for node in divide_fdef[0].node_def:
        self.assertAllEqual(node.device, "/device:CPU:1")

  def _testNestedDeviceWithSameFunction(self, func_name):
    # [FUNCTION DOC]
    # HELPER: shared scenario for the two tests below.
    # Takes func_name (a name or nothing). Defines a matrix-product function with that name inside a
    # wrapper, calls it under CPU zero and again under CPU one, then collects the devices of nodes from
    # the first two library functions and asserts the sorted device list is the two CPU devices. May
    # raise from graph building if two functions with the same name conflict.

    def MatmulWrap(a, b):
      # [FUNCTION DOC]
      # HELPER: defines the matrix-product function with the given name and calls it.

      @function.Defun(
          func_name=func_name, *[dtypes.int32] * 2)
      def Matmul(a, b):
        # [FUNCTION DOC]
        # HELPER: function body; matrix product of two integer arguments.
        return math_ops.matmul(a, b)

      return Matmul(a, b)

    with ops.Graph().as_default(), ops.device("CPU:0"):
      c = MatmulWrap(1, 2)

      with ops.device("CPU:1"):
        MatmulWrap(c, 3)

      gdef = ops.get_default_graph().as_graph_def()

      devices = []
      for node in gdef.library.function[0].node_def:
        devices.append(node.device)
      for node in gdef.library.function[1].node_def:
        devices.append(node.device)

      self.assertAllEqual(sorted(devices), ["/device:CPU:0", "/device:CPU:1"])

  def testFunctionWithName(self):
    # [FUNCTION DOC]
    # Scenario: the shared scenario with a fixed explicit function name; defining two functions with the
    # same name but different devices must fail with an invalid-argument error whose message says
    # a different function with that name already exists.
    with self.assertRaises(InvalidArgumentError) as cm:
      self._testNestedDeviceWithSameFunction("MatmulTest")
    self.assertEqual(
        cm.exception.message,
        "Cannot add function \'MatmulTest\' because a different "
        "function with the same name already exists.")

  def testFunctionWithoutName(self):
    # [FUNCTION DOC]
    # Scenario: the shared scenario with no explicit name; automatically generated unique names let both
    # functions coexist, with devices CPU zero and CPU one.
    self._testNestedDeviceWithSameFunction(None)


if __name__ == "__main__":
  test.main()
