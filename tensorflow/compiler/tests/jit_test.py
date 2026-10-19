# [MODULE DOC]
# Tests for JIT (XLA) compilation of TensorFlow graphs on CPU/GPU devices using classic sessions.
# Checks that compiled functions and explicitly scoped regions produce correct results and that
# XLA compile/run kernels actually appear in the run-metadata trace; also covers loops, cond,
# nested functions, gradients, lazy compilation and megamorphic cluster detection.
# Defines small helper functions for session configuration and trace inspection. Importing has
# no side effects beyond defining names; when run directly it enables lazy compilation through
# the XLA flags environment variable, disables eager execution and runs the tests.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for JIT compilation on the CPU and GPU devices."""

import os

import numpy as np

from tensorflow.compiler.tests import test_utils
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.compiler.xla import jit
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import test


jit_scope = jit.experimental_jit_scope

# Disable rewrites to make sure we don't end up having to update this test
# whenever we implement new ones.
def NoRewriteSessionConfig():
  # [FUNCTION DOC]
  # HELPER: builds a session configuration that turns off model pruning and the arithmetic,
  # dependency and function graph rewrites, so the tests do not depend on new optimizer passes.
  # Takes no arguments and returns a configuration message object.
  rewriter_config = rewriter_config_pb2.RewriterConfig(
      disable_model_pruning=True,
      arithmetic_optimization=rewriter_config_pb2.RewriterConfig.OFF,
      dependency_optimization=rewriter_config_pb2.RewriterConfig.OFF,
      function_optimization=rewriter_config_pb2.RewriterConfig.OFF)
  graph_options = config_pb2.GraphOptions(rewrite_options=rewriter_config)
  return config_pb2.ConfigProto(graph_options=graph_options)


def CompiledKernel(fn, *inputs, **kwargs):
  # [FUNCTION DOC]
  # HELPER: wraps a Python function as an XLA-compiled graph function and calls it.
  # Inputs: fn, a callable building the computation; positional inputs, the tensors passed in;
  # optional keyword name (function name) and noinline (whether to forbid inlining), both
  # default to nothing. Returns the output tensor(s) of the call.
  """Execute 'fn' as a compiled XLA kernel, with 'inputs'."""
  name = kwargs.pop("name", None)
  noinline = kwargs.pop("noinline", None)

  @function.Defun(func_name=name, noinline=noinline, compiled=True)
  def Compiled(*args):
    # [FUNCTION DOC]
    # HELPER: the body of the compiled function; forwards its arguments to the user callable
    # and returns its result.
    return fn(*args)

  return Compiled(*inputs)


def RunMetadataLabels(run_metadata):
  # [FUNCTION DOC]
  # HELPER: collects the timeline label of every node execution recorded in a run-metadata
  # object across all devices. Returns a list of strings.
  """Returns all labels in run_metadata."""
  labels = []
  for dev_stats in run_metadata.step_stats.dev_stats:
    for node_stats in dev_stats.node_stats:
      labels.append(node_stats.timeline_label)
  return labels


def InLabels(labels, substr):
  # [FUNCTION DOC]
  # HELPER: returns true if at least one label in the list contains the given substring.
  """Returns true iff one of the labels contains substr."""
  return any(substr in x for x in labels)


def MetadataHasXlaRunOp(run_metadata):
  # [FUNCTION DOC]
  # HELPER: returns true if the trace in the run-metadata contains an executed XLA run kernel
  # (a label containing the XLA run marker). Relies on trace labels, which the code itself
  # calls hacky.
  """Returns true if there are XlaRun kernels in run_metadata's timeline."""

  # TODO(phawkins): find a less hacky way to test whether a kernel ran.
  return InLabels(RunMetadataLabels(run_metadata), "_XlaRun")


class JitLaunchTest(test.TestCase):
  # [CLASS DOC]
  # Tests that functions compiled with the compiled flag are launched through XLA and give the
  # same results as direct execution. Uses a shared comparison helper.

  # Evaluates 'fn' on 'args' both directly and as a compiled XLA kernel.
  # Verifies that the outputs match and that XLA was invoked. 'fn' must take
  # the same number of tensors as arguments that are in 'args', and must return
  # a tuple of output tensors.
  #
  # If 'require_kernel_launch' is True, then we verify that an XlaCompile/XlaRun
  # node actually ran. However, it is sometimes possible for XlaCompile/XlaRun
  # ops to be constant-folded away, so the check is optional.
  def _compare(self,
               fn,
               args,
               require_kernel_launch=True,
               name=None,
               noinline=None):
    # [FUNCTION DOC]
    # HELPER: runs a function both directly and as a compiled XLA kernel and compares.
    # Inputs: fn, callable taking as many tensors as there are args and returning a tuple of
    # outputs; args, list of NumPy arrays fed through placeholders; require_kernel_launch
    # (default true) to also check an XLA run kernel executed (it may be constant-folded away
    # otherwise); name and noinline are passed to the compiled wrapper.
    # Uses a session without graph rewrites and a full trace. Prints the results and asserts
    # closeness (loose tolerances). When require_kernel_launch is false, only the compiled
    # execution runs and no comparison is made. Returns nothing.
    with session_lib.Session(config=NoRewriteSessionConfig()) as sess:
      placeholders = []
      feeds = {}
      for arg in args:
        placeholder = array_ops.placeholder(
            dtypes.as_dtype(arg.dtype), list(arg.shape))
        placeholders.append(placeholder)
        feeds[placeholder] = arg

      compiled_op = CompiledKernel(
          fn, *placeholders, name=name, noinline=noinline)
      direct_op = fn(*placeholders)

      run_metadata = config_pb2.RunMetadata()
      compiled = test_utils.RunWithWarmup(
          sess, compiled_op, feeds,
          config_pb2.RunOptions(trace_level=config_pb2.RunOptions.FULL_TRACE),
          run_metadata)
      print("Compiled Result {}".format(compiled))

      if require_kernel_launch:
        self.assertTrue(MetadataHasXlaRunOp(run_metadata))

        direct = sess.run(direct_op, feeds)
        print("Direct Result {}".format(direct))

        if (isinstance(compiled, (tuple, list)) and
            (isinstance(direct, (tuple, list)))):
          for (x, y) in zip(compiled, direct):
            self.assertAllClose(x, y, rtol=1e-1)
        else:
          self.assertAllClose(compiled, direct, rtol=1e-2)

  def testNoOutputs(self):
    # [FUNCTION DOC]
    # Checks that calling a compiled function with no outputs does not crash.
    with session_lib.Session() as sess:

      # Check that calling the result as a compiled kernel doesn't crash.
      @function.Defun(compiled=True)
      def KernelWithNoOutputs():
        # [FUNCTION DOC]
        # HELPER: compiled kernel with no outputs; creates an unused constant and returns nothing.
        a = constant_op.constant(100)  # pylint: disable=unused-variable

      call = KernelWithNoOutputs()  # pylint: disable=assignment-from-no-return
      test_utils.RunWithWarmup(sess, call, {})

  def testAliasing(self):
    # [FUNCTION DOC]
    # Regression test: compiled functions returning the same buffer twice (aliased outputs) must
    # work, both with and without inlining forbidden.
    """Regression test for compiled functions that return an aliased buffer.

       XLA returns aliased buffers if outputs are identical. Tests that
       we handle that case.
    """

    def AddOnceReturnTwice(x):
      # [FUNCTION DOC]
      # HELPER: adds the input to itself and returns that same result twice.
      y = math_ops.add(x, x)
      return y, y

    # Exercises compiling a function (say, Foo) which calls another function
    # (say, Bar) which is not inlined. When the compiler compiles Foo, it needs
    # to symbolically execute Bar correctly regardless of whether Bar is inlined
    # or not.

    # Tests compiled=True and noinline=True.
    self._compare(
        AddOnceReturnTwice, [np.array([[[0.5, -1.0]]], dtype=np.float32)],
        name="AddOnceReturnTwice_inline",
        noinline=True)

    # Tests compiled=True and noinline=False.
    self._compare(
        AddOnceReturnTwice, [np.array([[[0.5, -1.0]]], dtype=np.float32)],
        name="AddOnceReturnTwice_noinline",
        noinline=False)

  def testOneConstOutput(self):
    # [FUNCTION DOC]
    # Compiled function whose only output is a constant; kernel launch is not required.
    """Test consisting of a single constant return value."""

    def OneConstOutput():
      # [FUNCTION DOC]
      # HELPER: returns a constant integer vector.
      return constant_op.constant([-3, 44, 99])

    self._compare(OneConstOutput, [], require_kernel_launch=False)

  def testConstZeroElementOutput(self):
    # [FUNCTION DOC]
    # Compiled function returning a constant empty (zero-element) tensor.
    """Test consisting of a constant zero element return value."""

    def ConstZeroElementOutput():
      # [FUNCTION DOC]
      # HELPER: returns an empty tensor of shape 7 by 0 filled with a float.
      return array_ops.fill([7, 0], 3.0)

    self._compare(ConstZeroElementOutput, [], require_kernel_launch=False)

  def testSomeConstOutputs(self):
    # [FUNCTION DOC]
    # Compiled function returning a mix of constant and non-constant outputs.
    """Test kernels that return a mixture of const and non-const outputs."""

    def SomeConstOutputs(x):
      # [FUNCTION DOC]
      # HELPER: returns a constant vector, the identity of the input, and a scalar constant.
      return constant_op.constant(
          [-2, 7]), array_ops.identity(x), constant_op.constant(3.5)

    self._compare(
        SomeConstOutputs, [np.array(
            [[1, 2, 3], [4, 5, 6]], dtype=np.float32)])

  def testInt32Input(self):
    # [FUNCTION DOC]
    # Compiled function adding an int32 input to itself (int32 lives in host memory on GPUs).
    """Test an int32-typed input.

       On a GPU, int32 tensors will be placed in host memory.
    """

    def AddToSelf(x):
      # [FUNCTION DOC]
      # HELPER: returns the input added to itself.
      return math_ops.add(x, x)

    self._compare(AddToSelf, [np.array([7, 1, 3], dtype=np.int32)])

  def testMandatoryConstantInput(self):
    # [FUNCTION DOC]
    # Tests an op (fill) with a mandatory compile-time constant shape input fed at run time.
    """Tests an operator that has a mandatory-constant shape input."""

    def FillWithFloat(x):
      # [FUNCTION DOC]
      # HELPER: fills a tensor of the given shape with the value 9.5.
      return array_ops.fill(x, 9.5)

    self._compare(FillWithFloat, [np.array([3, 2], dtype=np.int32)])

  def testMnistForwardFunc(self):
    # [FUNCTION DOC]
    # Compiles the MNIST beginner inference function (matrix multiply, bias add, softmax) with
    # random float32 weights, bias and batch of 16 flattened images, and compares to direct run.
    """Compute inference function from MNIST beginners tutorial."""
    batch_size = 16
    image_size = 28 * 28
    num_classes = 10

    # Define a TensorFlow function to compute the forward pass.
    def MnistForward(w, b, x):
      # [FUNCTION DOC]
      # HELPER: returns softmax of the product of x and w plus b.
      return nn_ops.softmax(math_ops.matmul(x, w) + b)

    w = np.random.random_sample((image_size, num_classes)).astype(np.float32)
    b = np.random.random_sample((num_classes)).astype(np.float32)
    x = np.random.random_sample((batch_size, image_size)).astype(np.float32)
    self._compare(MnistForward, [w, b, x])

  def testExplicitMarking(self):
    # [FUNCTION DOC]
    # Marks a matmul and a square op with the experimental JIT scope, leaving an add between
    # them unmarked, runs in a fresh graph and asserts an XLA run kernel executed and the
    # numeric output matches a NumPy computation.
    """Test explicit marking of operators to compile."""
    batch_size = 16
    image_size = 28 * 28
    num_classes = 10

    with ops.Graph().as_default():
      x = array_ops.placeholder(dtypes.float32)
      w = array_ops.placeholder(dtypes.float32)
      b = array_ops.placeholder(dtypes.float32)
      with jit_scope():
        y1 = math_ops.matmul(x, w)
      y2 = math_ops.add(y1, b)
      with jit_scope():
        y = math_ops.square(y2)

      dw = np.random.random_sample((image_size, num_classes)).astype(np.float32)
      db = np.random.random_sample((num_classes)).astype(np.float32)
      dx = np.random.random_sample((batch_size, image_size)).astype(np.float32)
      with session_lib.Session() as sess:
        run_metadata = config_pb2.RunMetadata()
        output = test_utils.RunWithWarmup(
            sess,
            y, {
                x: dx,
                w: dw,
                b: db
            },
            run_metadata=run_metadata,
            options=config_pb2.RunOptions(
                trace_level=config_pb2.RunOptions.FULL_TRACE))

        # TODO(phawkins): really we would like to test that there were exactly
        # two kernel launches. However, we have no reliable way to determine
        # that.
        self.assertTrue(MetadataHasXlaRunOp(run_metadata))

        expected = np.square(np.dot(dx, dw) + db)
        self.assertAllClose(expected, output, rtol=1e-1)


class XlaCompilationTest(test.TestCase):
  # [CLASS DOC]
  # Tests automatic compilation of JIT-scoped regions: constants, ignored arguments, loops,
  # conditionals, nested functions and gradients.
  """Tests for auto-compilation on CPU/GPU devices."""

  def testReshape(self):
    # [FUNCTION DOC]
    # Reshape inside a JIT scope with the shape fed at run time; asserts an XLA kernel ran and
    # the result is the expected 2x3 array.
    """Tests an operator with compile-time constant and non-constant inputs."""

    with self.session(config=NoRewriteSessionConfig()) as sess:
      x = array_ops.placeholder(dtypes.float32)
      y = array_ops.placeholder(dtypes.int32)
      with jit_scope():
        # Reshape's first argument is non-constant in the JIT, but its second
        # (shape) argument will be treated as a compile-time constant for
        # each JIT compilation.
        # We do not use a tf.const() argument since we want to ensure the
        # shape is still a run-time argument to the JIT, and not
        # statically known as part of the JIT compilation's input graph.
        z = array_ops.reshape(x, y)
      run_metadata = config_pb2.RunMetadata()
      out = test_utils.RunWithWarmup(
          sess,
          z, {
              x: np.array([1, 2, 3, 4, 5, 6], np.float32),
              y: [-1, 3]
          },
          run_metadata=run_metadata,
          options=config_pb2.RunOptions(
              trace_level=config_pb2.RunOptions.FULL_TRACE))
      self.assertTrue(MetadataHasXlaRunOp(run_metadata))
      self.assertAllClose(np.array([[1, 2, 3], [4, 5, 6]], np.float32), out)

  def testIgnoredArguments(self):
    # [FUNCTION DOC]
    # Cluster in which one input is not used by the output but is pulled in through control
    # dependencies; asserts compiled execution runs and result is 28.
    """Tests that JIT computations can ignore formal parameters."""

    with self.session(config=NoRewriteSessionConfig()) as sess:
      x = array_ops.placeholder(dtypes.int32)
      y = array_ops.placeholder(dtypes.int32)
      with jit_scope():
        z = math_ops.add(x, x)
        w = math_ops.add(y, y)
        # Pulls 'w' into the same compilation via control dependencies.
        with ops.control_dependencies([w]):
          n = control_flow_ops.no_op()
        with ops.control_dependencies([n]):
          t = math_ops.add(z, z)

      run_metadata = config_pb2.RunMetadata()
      out = test_utils.RunWithWarmup(
          sess,
          t, {
              x: np.int32(7),
              y: np.int32(404)
          },
          run_metadata=run_metadata,
          options=config_pb2.RunOptions(
              trace_level=config_pb2.RunOptions.FULL_TRACE))
      self.assertTrue(MetadataHasXlaRunOp(run_metadata))
      self.assertAllClose(28, out)

  def testLoops(self):
    # [FUNCTION DOC]
    # While loop inside a JIT scope; asserts XLA ran and the result is about 95.
    """Tests that compilation accepts computations containing loops."""

    with self.session(config=NoRewriteSessionConfig()) as session:
      x = array_ops.placeholder(dtypes.float32)
      with jit_scope():
        c = lambda i, _: math_ops.less(i, 5)
        b = lambda i, x: (i + 1, x * 2.0 + 1.0)
        _, y = while_loop.while_loop(c, b, (constant_op.constant(0), x))

      run_metadata = config_pb2.RunMetadata()
      result = session.run(y, {x: np.float32(2)},
                           run_metadata=run_metadata,
                           options=config_pb2.RunOptions(
                               trace_level=config_pb2.RunOptions.FULL_TRACE))
      self.assertTrue(MetadataHasXlaRunOp(run_metadata))
      self.assertAllClose(result, np.float32(95), rtol=1e-1)

  def testCond(self):
    # [FUNCTION DOC]
    # Conditional inside a JIT scope; clustering the producer with the consumer would deadlock.
    # Asserts XLA ran and the result is about 6.
    """Tests that compilation handles switch operators."""

    with self.session(config=NoRewriteSessionConfig()) as session:
      x = array_ops.placeholder(dtypes.float32)
      y = array_ops.placeholder(dtypes.float32)
      c = array_ops.placeholder(dtypes.bool)
      with jit_scope():
        z = x + 1.0
        w = cond.cond(c, lambda: z, lambda: y)
        t = math_ops.add(z, w)

      # If JIT compilation chooses to cluster z and t, then execution will
      # deadlock.

      run_metadata = config_pb2.RunMetadata()
      result = test_utils.RunWithWarmup(
          session,
          t, {
              x: np.float32(2),
              y: np.float32(4),
              c: True
          },
          run_metadata=run_metadata,
          options=config_pb2.RunOptions(
              trace_level=config_pb2.RunOptions.FULL_TRACE))
      self.assertTrue(MetadataHasXlaRunOp(run_metadata))
      self.assertAllClose(result, np.float32(6), rtol=1e-1)

  def testNestedFunction(self):
    # [FUNCTION DOC]
    # Builds nested compiled functions (an entry function calling a compiled function that
    # calls another) and asserts output values 20 and 2100 for inputs 2 and 10.
    g = ops.Graph()
    with g.as_default():

      @function.Defun(compiled=True)
      def Bar(x, y):
        # [FUNCTION DOC]
        # HELPER: compiled function returning the first argument plus twice the second.
        return x + 2 * y

      @function.Defun(compiled=True)
      def Foo(x):
        # [FUNCTION DOC]
        # HELPER: compiled function calling the inner function with the square and cube of x.
        return Bar(x * x, x * x * x)

      @function.Defun()
      def Entry(x):
        # [FUNCTION DOC]
        # HELPER: uncompiled entry function that calls the compiled outer function.
        return Foo(x)

      inp = array_ops.placeholder(dtypes.float32)
      out = Entry(inp)

    with self.session(
        config=NoRewriteSessionConfig(), graph=g, use_gpu=True) as sess:
      run_metadata = config_pb2.RunMetadata()
      val = sess.run(out,
                     feed_dict={inp: [2., 10.]},
                     run_metadata=run_metadata,
                     options=config_pb2.RunOptions(
                         trace_level=config_pb2.RunOptions.FULL_TRACE))
      self.assertAllClose(val, [20., 2100.])

  def testLoopDeadlock(self):
    # [FUNCTION DOC]
    # Regression test for deadlocks in graphs with loops inside a JIT scope; asserts the result
    # is about 63.
    """Regression test for bug that caused deadlocks in graphs with loops."""

    with self.session(config=NoRewriteSessionConfig()) as session:
      x = array_ops.placeholder(dtypes.float32)
      with jit_scope():
        y = x + 1.0
        c = lambda i, _x, _y: math_ops.less(i, 5)
        b = lambda i, x, _y: (i + 1, x * 2.0 + 1.0, x - 3.0)
        _, _, w = while_loop.while_loop(c, b, (constant_op.constant(0), y, x))
        u = w + y
      result = session.run(u, {x: np.float32(2)})
      self.assertAllClose(result, np.float32(63), rtol=1e-1)

  def testGradient(self):
    # [FUNCTION DOC]
    # Checks the backprop of a log function: without compilation the trace shows reciprocal and
    # multiplication ops and no XLA kernels; with compilation it shows XLA compile and run
    # kernels and none of those plain ops.
    """Tests that the backprop function is properly compiled."""

    def _Run(compiled):
      # [FUNCTION DOC]
      # HELPER: builds the log function (compiled or not per the flag), takes its gradient at
      # 100 with function inlining enabled, asserts the value is 0.01 and returns the list of
      # trace labels.

      @function.Defun(compiled=compiled)
      def Forward(x):
        # [FUNCTION DOC]
        # HELPER: forward function computing the natural log.
        return math_ops.log(x)

      g = ops.Graph()
      with g.as_default():
        x = array_ops.placeholder(dtypes.float32)
        y = Forward(x)
        dx, = gradients_impl.gradients(y, [x], 1.0)

      cfg = NoRewriteSessionConfig()
      cfg.graph_options.optimizer_options.opt_level = (
          config_pb2.OptimizerOptions.L1)
      cfg.graph_options.optimizer_options.do_function_inlining = True
      with session_lib.Session(graph=g, config=cfg) as sess:
        run_metadata = config_pb2.RunMetadata()
        dx_val = test_utils.RunWithWarmup(
            sess,
            dx,
            feed_dict={x: 100.},
            run_metadata=run_metadata,
            options=config_pb2.RunOptions(
                trace_level=config_pb2.RunOptions.FULL_TRACE))
      self.assertAllClose(dx_val, 0.01)
      return RunMetadataLabels(run_metadata)

    # SymGrad[f=log(x)](x, dy) = 1/x * dy
    #
    # Note: we don't need to compute log(x) for dx due to graph pruning.

    # Do not compile the backprop. We should see one Reciprocal and one Mul.
    labels = _Run(compiled=False)
    self.assertFalse(InLabels(labels, "Log"))
    self.assertTrue(InLabels(labels, "Reciprocal"))
    self.assertTrue(InLabels(labels, "Mul"))
    self.assertFalse(InLabels(labels, "XlaCompile"))
    self.assertFalse(InLabels(labels, "XlaRun"))

    # Compile the backprop. One XlaCompile/XlaRun pair.
    labels = _Run(compiled=True)
    self.assertFalse(InLabels(labels, "Log"))
    self.assertFalse(InLabels(labels, "Reciprocal"))
    self.assertFalse(InLabels(labels, "Mul"))
    self.assertTrue(InLabels(labels, "XlaCompile"))
    self.assertTrue(InLabels(labels, "XlaRun"))


class ElementWiseFusionTest(test.TestCase):
  # [CLASS DOC]
  # Intended to test element-wise fusion under a global JIT level.
  # KNOWN ISSUE: contains only a helper method and no test methods, so nothing is tested.

  # Runs a simple test with the input jit_level and fusion_only flag.
  def simpleTest(self, arg0, arg1, global_jit_level):
    # [FUNCTION DOC]
    # HELPER: builds a graph of element-wise ops separated by a matmul under the given global
    # JIT level, runs it with two float32 2x2 inputs and counts XLA compile and run kernels.
    # Asserts the counts are equal and returns the output and the run count.
    # KNOWN ISSUE: never called by any test in this file; the name does not start with test.
    config = config_pb2.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = global_jit_level

    with session_lib.Session(config=config) as sess:
      a1 = array_ops.placeholder(dtypes.float32, [2, 2], name="a1")
      a2 = array_ops.placeholder(dtypes.float32, [2, 2], name="a2")
      # Two element-wise ops. We need at least two ops since single
      # element clusters are not passed to XLA in fusion_only mode.
      a3 = a1 * a2
      a4 = a3 + a1
      # A matmul to break XLA clustering.
      a5 = math_ops.matmul(a4, a1)
      # Two more element-wise ops.
      a6 = a5 - a4
      a7 = a6 + a2

      run_metadata = config_pb2.RunMetadata()
      output = test_utils.RunWithWarmup(
          sess,
          a7, {
              a1: arg0,
              a2: arg1
          },
          run_metadata=run_metadata,
          options=config_pb2.RunOptions(
              trace_level=config_pb2.RunOptions.FULL_TRACE))

      labels = RunMetadataLabels(run_metadata)

      xla_compile_count = sum("XlaCompile(" in x for x in labels)
      xla_run_count = sum("XlaRun(" in x for x in labels)
      self.assertEqual(xla_compile_count, xla_run_count)

      return output, xla_run_count


class LazyCompilationTest(test.TestCase):
  # [CLASS DOC]
  # Tests lazy compilation (compile on second sighting of a shape) and megamorphic detection.

  def testLazyCompilation(self):
    # [FUNCTION DOC]
    # With a compiled log function: the first run is compiled eagerly; a new shape compiles
    # only the second time it is seen (first sighting shows a compile marker but no run);
    # asserts these trace patterns.

    @function.Defun(compiled=True)
    def CompiledFunction(x):
      # [FUNCTION DOC]
      # HELPER: compiled function computing log.
      return math_ops.log(x)

    with session_lib.Session(config=NoRewriteSessionConfig()) as sess:
      x = array_ops.placeholder(dtypes.float32)
      y = CompiledFunction(x)

      # The very first run of the cluster is always compiled (non-lazily).
      run_metadata_for_first_run = config_pb2.RunMetadata()
      sess.run(
          y,
          feed_dict={x: [2., 10., 19., 77., 100.]},
          run_metadata=run_metadata_for_first_run,
          options=config_pb2.RunOptions(
              trace_level=config_pb2.RunOptions.FULL_TRACE))
      self.assertTrue(
          InLabels(
              RunMetadataLabels(run_metadata_for_first_run), "_XlaCompile"))
      self.assertTrue(
          InLabels(RunMetadataLabels(run_metadata_for_first_run), "_XlaRun"))

      run_metadata_before_warmup = config_pb2.RunMetadata()
      sess.run(
          y,
          feed_dict={x: [2., 10.]},
          run_metadata=run_metadata_before_warmup,
          options=config_pb2.RunOptions(
              trace_level=config_pb2.RunOptions.FULL_TRACE))
      self.assertTrue(
          InLabels(
              RunMetadataLabels(run_metadata_before_warmup), "_XlaCompile"))
      self.assertFalse(
          InLabels(RunMetadataLabels(run_metadata_before_warmup), "_XlaRun"))

      # We compile when we see the same shape a second time.

      run_metadata_after_warmup = config_pb2.RunMetadata()
      sess.run(
          y,
          feed_dict={x: [2., 10.]},
          run_metadata=run_metadata_after_warmup,
          options=config_pb2.RunOptions(
              trace_level=config_pb2.RunOptions.FULL_TRACE))
      self.assertTrue(
          InLabels(RunMetadataLabels(run_metadata_after_warmup), "_XlaCompile"))
      self.assertTrue(
          InLabels(RunMetadataLabels(run_metadata_after_warmup), "_XlaRun"))

      run_metadata_for_new_shape = config_pb2.RunMetadata()
      sess.run(
          y,
          feed_dict={x: [2., 10., 12.]},
          run_metadata=run_metadata_for_new_shape,
          options=config_pb2.RunOptions(
              trace_level=config_pb2.RunOptions.FULL_TRACE))
      self.assertTrue(
          InLabels(
              RunMetadataLabels(run_metadata_for_new_shape), "_XlaCompile"))
      self.assertFalse(
          InLabels(RunMetadataLabels(run_metadata_for_new_shape), "_XlaRun"))

  def testIsMegamorphic(self):
    # [FUNCTION DOC]
    # Runs many shapes only a few times each so the cluster becomes megamorphic, then asserts a
    # new shape never gets run (only the compile marker appears).

    @function.Defun(compiled=True)
    def CompiledFunction(x):
      # [FUNCTION DOC]
      # HELPER: compiled function computing log.
      return math_ops.log(x)

    with session_lib.Session(config=NoRewriteSessionConfig()) as sess:
      x = array_ops.placeholder(dtypes.float32)
      y = CompiledFunction(x)

      # Make the cluster go megamorphic by running it with lots of shape
      # signatures where the cluster is executed with each signature only a few
      # times.  Then check that we don't compile the cluster ever again.

      for shape in range(10, 50):
        for _ in range(0, 49):
          sess.run(y, feed_dict={x: [0.] * shape})

      for _ in range(0, 50):
        run_metadata = config_pb2.RunMetadata()
        sess.run(
            y,
            feed_dict={x: [0.] * 60},
            run_metadata=run_metadata,
            options=config_pb2.RunOptions(
                trace_level=config_pb2.RunOptions.FULL_TRACE))
        self.assertTrue(
            InLabels(RunMetadataLabels(run_metadata), "_XlaCompile"))
        self.assertFalse(InLabels(RunMetadataLabels(run_metadata), "_XlaRun"))

  def testIsNotMegamorphic(self):
    # [FUNCTION DOC]
    # Runs many shapes many times each so the cluster is not megamorphic, then asserts a new
    # shape is compiled and run.

    @function.Defun(compiled=True)
    def CompiledFunction(x):
      # [FUNCTION DOC]
      # HELPER: compiled function computing log.
      return math_ops.log(x)

    with session_lib.Session(config=NoRewriteSessionConfig()) as sess:
      x = array_ops.placeholder(dtypes.float32)
      y = CompiledFunction(x)

      # Run the cluster with lots of shape signatures, but in a way that it
      # isn't megamorphic (i.e. each shape signature sees a lot of executions).
      # Then check that the cluster has not been marked as megamorphic.

      for shape in range(10, 50):
        for _ in range(0, 1000):
          sess.run(y, feed_dict={x: [0.] * shape})

      for _ in range(0, 10):
        sess.run(y, feed_dict={x: [0.] * 60})

      run_metadata = config_pb2.RunMetadata()
      sess.run(
          y,
          feed_dict={x: [0.] * 60},
          run_metadata=run_metadata,
          options=config_pb2.RunOptions(
              trace_level=config_pb2.RunOptions.FULL_TRACE))
      self.assertTrue(InLabels(RunMetadataLabels(run_metadata), "_XlaCompile"))
      self.assertTrue(InLabels(RunMetadataLabels(run_metadata), "_XlaRun"))


if __name__ == "__main__":
  os.environ["TF_XLA_FLAGS"] = ("--tf_xla_enable_lazy_compilation=true " +
                                os.environ.get("TF_XLA_FLAGS", ""))
  # This test is using Tensorflow sessions which are not compatible with eager
  # mode.
  ops.disable_eager_execution()
  test.main()
