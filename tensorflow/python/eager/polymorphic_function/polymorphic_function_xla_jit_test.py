# [MODULE DOC]
# Test module for the tracing function decorator with XLA just-in-time compilation requested. It
# defines one XLA test case class that runs on whichever XLA test device the test base selects and
# checks compilation, nesting, gradients, control flow, variable updates, error messages, compiler
# intermediate representation retrieval, constant propagation and usage counters.
# Importing it has no side effects; running it as a script enables eager execution and starts the
# test runner.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

import gc
import re

from tensorflow.compiler.tests import xla_test
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import collective_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import test
from tensorflow.python.util import nest


@test_util.with_eager_op_as_function
class FunctionTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # Test case for functions traced with XLA compilation, built on the XLA test base class (which
  # supplies the device name used to place ops). The class is wrapped so that eager ops run as
  # functions. Contains one shared helper that compares two ways of fetching compiler output, and many
  # independent scenario tests.

  def _compareTwoMethodsCompilerIROutput(self, f, args, kwargs):
    # [FUNCTION DOC]
    # HELPER: shared check used by the compiler-output tests of this class.
    # Inputs: a traced function, a sequence of positional arguments and a mapping of keyword arguments.
    # If any argument is not a tensor the test is skipped. Otherwise it fetches the compiler output
    # text once using the tensor arguments and once using tensor specifications derived from them, and
    # fails the test with both texts in the message if they differ. Returns nothing.
    """Assert the two different methods (tensor_spec inputs or tensor inputs) experimental_get_compiler give same HLO text."""
    flat_args = list(args) + list(kwargs.values())
    if not all([isinstance(x, tensor.Tensor) for x in flat_args]):
      self.skipTest('It only support args and kwargs are all tf.Tensor types.')

    args_spec = nest.map_structure(tensor.TensorSpec.from_tensor, args)
    kwargs_spec = nest.map_structure(tensor.TensorSpec.from_tensor, kwargs)

    hlo_1 = f.experimental_get_compiler_ir(*args, **kwargs)()
    hlo_2 = f.experimental_get_compiler_ir(*args_spec, **kwargs_spec)()

    if hlo_1 != hlo_2:
      self.fail(
          'The tensor_spec way experimental_get_compiler_ir give diff result to'
          f' normal experimental_get_compiler_ir. \nhlo_1:\n{hlo_1}'
          f'\nhlo_2:\n{hlo_2}\n'
      )

  def testAutoclusteringWithTfFunction(self):
    # [FUNCTION DOC]
    # Scenario: an uncompiled outer function calls an inner function marked for compilation, all on the
    # test device, while optimized graphs are collected. Skipped on TPU. Asserts that the optimized
    # graph contains the XLA run node when XLA is enabled for tests and does not contain it otherwise.
    if 'tpu' in self.device.lower():
      self.skipTest('Autoclustering does not run on TPU')

    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=False)
      def outer(a, b, c):
        # [FUNCTION DOC]
        # Uncompiled outer function: multiplies its first argument by the inner function's result for the
        # last two arguments and adds the last argument.
        return a * inner(b, c) + c

      @polymorphic_function.function(jit_compile=True)
      def inner(b, c):
        # [FUNCTION DOC]
        # Compiled inner function: returns the first argument plus the product of the two arguments.
        return b + c * b

      i1 = constant_op.constant([1.0, 2.0, 3.0, 4.0, 5.0])
      i2 = constant_op.constant([1.0, 2.0, 3.0, 4.0, 5.0])
      i3 = constant_op.constant([1.0, 2.0, 3.0, 4.0, 5.0])

      with context.collect_graphs(optimized=True) as graphs:
        outer(i1, i2, i3)

      if test_util.is_xla_enabled():
        self.assertIn('_XlaRun', [n.op for n in graphs[0].node])
      else:
        self.assertNotIn('_XlaRun', [n.op for n in graphs[0].node])

  def testBasic(self):
    # [FUNCTION DOC]
    # Scenario: the same addition function is wrapped once without and once with compilation, and both
    # are called with an integer tensor and a Python number. Asserts both give the expected elementwise
    # sums.
    with ops.device('device:{}:0'.format(self.device)):

      def fn(x, a):
        # [FUNCTION DOC]
        # Adds its second argument to the first; used to build both the plain and compiled versions.
        return x + a

      func = polymorphic_function.function(fn, jit_compile=False)
      xla_func = polymorphic_function.function(fn, jit_compile=True)

      inputs = constant_op.constant([1, 2, 2, 3, 3])
      self.assertAllClose([2, 3, 3, 4, 4], func(inputs, 1))
      self.assertAllClose([2, 3, 3, 4, 4], xla_func(inputs, 1))

  def testBasicInt32(self):
    # [FUNCTION DOC]
    # Scenario: a compiled addition function is called with a 32-bit integer tensor and a Python
    # number; asserts the expected elementwise sum, covering integer data under compilation.
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def fn(x, a):
        # [FUNCTION DOC]
        # Compiled function returning the sum of its two arguments.
        return x + a

      inputs = constant_op.constant([1, 2, 2, 3, 3], dtype=dtypes.int32)
      self.assertAllClose([2, 3, 3, 4, 4], fn(inputs, 1))

  def testDerivative(self):
    # [FUNCTION DOC]
    # Scenario: a compiled linear function is differentiated with a gradient tape; asserts the gradient
    # is two everywhere. It then inspects the forward and backward functions built for the derivative
    # and asserts that the must-compile attribute was propagated to both.
    with ops.device('device:{}:0'.format(self.device)):

      def fn(x, a):
        # [FUNCTION DOC]
        # Returns twice the first argument plus the second; the function under differentiation.
        return 2 * x + a

      xla_func = polymorphic_function.function(fn, jit_compile=True)

      with backprop.GradientTape() as tape:
        inputs = constant_op.constant([1., 2., 2., 3., 3.])
        tape.watch(inputs)
        outputs = xla_func(inputs, 1)

      self.assertAllClose([2, 2, 2, 2, 2], tape.gradient(outputs, inputs))

      # pylint: disable=protected-access
      (forward, backward) = xla_func.get_concrete_function(
          inputs, 1)._delayed_rewrite_functions.forward_backward()

      # Check that the must-compile attribute gets correctly propagated to the
      # created derivatives.
      self.assertTrue(backward.function_def.attr['_XlaMustCompile'])
      self.assertTrue(forward.cached_definition.attr['_XlaMustCompile'])

  # Calling function with jit_compile=True from
  # jit_compile=False should compile the inner func.
  def testNestedCall(self):
    # [FUNCTION DOC]
    # Scenario: an uncompiled function calls a compiled function. Skipped on TPU. Asserts that the
    # call returns the expected elementwise sum, i.e. the inner function is compiled and works when
    # called from an uncompiled one.
    if 'tpu' in self.device.lower():
      self.skipTest('b/162800687: Inner function runs on host')

    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def fn(x, a):
        # [FUNCTION DOC]
        # Compiled inner function returning the sum of its arguments.
        return x + a

      @polymorphic_function.function(jit_compile=False)
      def fn2(x, a):
        # [FUNCTION DOC]
        # Uncompiled outer function that simply calls the compiled inner function.
        return fn(x, a)

      inputs = constant_op.constant([1, 2, 2, 3, 3])
      self.assertAllClose([2, 3, 3, 4, 4], fn2(inputs, 1))

  def testNestedCallUnsupportedOps(self):
    # [FUNCTION DOC]
    # Scenario: an uncompiled function calls a compiled function that uses string formatting and
    # string length, which XLA cannot compile. Skipped on TPU. Asserts that calling it raises an
    # invalid-argument error mentioning unsupported operations.
    if 'tpu' in self.device.lower():
      self.skipTest('Outside compilation will extract string_length to CPU')

    with ops.device('device:{}:0'.format(self.device)):

      def fn(x):
        # [FUNCTION DOC]
        # Returns the length of the string produced by formatting its argument; contains ops unsupported by
        # XLA.
        return string_ops.string_length(
            string_ops.string_format('{}', x))

      xla_func = polymorphic_function.function(fn, jit_compile=True)

      def fn2(x):
        # [FUNCTION DOC]
        # Uncompiled wrapper that calls the compiled function.
        return xla_func(x)

      func = polymorphic_function.function(fn2, jit_compile=False)
      inputs = constant_op.constant([1, 2, 2, 3, 3])
      with self.assertRaisesRegex(
          errors.InvalidArgumentError, 'unsupported operations'
      ):
        func(inputs)

  def testUnsupportedOps(self):
    # [FUNCTION DOC]
    # Scenario: a compiled function containing string-formatting ops is called with float values.
    # Asserts that an invalid-argument error mentioning unsupported operations is raised.
    with ops.device('device:{}:0'.format(self.device)):

      def fn(x):
        # [FUNCTION DOC]
        # Returns the length of the string produced by formatting its argument; unsupported by XLA.
        return string_ops.string_length(
            string_ops.string_format('{}', x))

      xla_func = polymorphic_function.function(fn, jit_compile=True)

      with self.assertRaisesRegex(
          errors.InvalidArgumentError, 'unsupported operations'
      ):
        xla_func(constant_op.constant([3.1, 3.2]))

  def testCollectiveReduceChannelId(self):
    # [FUNCTION DOC]
    # Scenario: a compiled function performs two collective all-reduce operations with the same group
    # settings on two inputs and adds them. The compiler text is fetched and searched for channel
    # identifiers; asserts exactly two are found and that they differ. Also runs the shared comparison
    # of the two compiler-output retrieval methods.
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def fn(x, y):
        # [FUNCTION DOC]
        # Performs one all-reduce on each argument (group of two) and returns the sum of the results.
        t0 = collective_ops.all_reduce_v2(
            t=x, group_size=2, group_key=1, instance_key=1)
        t1 = collective_ops.all_reduce_v2(
            t=y, group_size=2, group_key=1, instance_key=1)
        return t0 + t1

      inputs = constant_op.constant([1.0, 2.0, 3.0])
      # Make sure 2 different channel ids are assigned to the 2 all-reduce
      # instructions generated by XLA.
      hlo_str = fn.experimental_get_compiler_ir(inputs, inputs)()
      matches = re.findall('channel_id=([0-9]*),', hlo_str)
      self.assertLen(matches, 2)
      self.assertNotEqual(matches[0], matches[1])
      self._compareTwoMethodsCompilerIROutput(fn, [inputs, inputs], {})

  def testCollectiveReduceReplicaGroups(self):
    # [FUNCTION DOC]
    # Scenario: a compiled function performs a single collective all-reduce. Asserts the compiler text
    # contains replica group information, then runs the shared compiler-output comparison.
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def fn(x):
        # [FUNCTION DOC]
        # Performs one all-reduce on its argument (group of two) and returns the result.
        t0 = collective_ops.all_reduce_v2(
            t=x, group_size=2, group_key=1, instance_key=1)
        return t0

      inputs = constant_op.constant([1.0, 2.0, 3.0])
      # Make sure replica groups are assigned
      hlo_str = fn.experimental_get_compiler_ir(inputs)()
      self.assertIn('replica_groups={{', hlo_str)
      self._compareTwoMethodsCompilerIROutput(fn, [inputs], {})

  def testCollectiveReduceGroupAssignment(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: a comment in the body talks about two different channel ids for two all-reduce
    # instructions, but the test has one all-reduce and asserts replica groups instead (copy-paste
    # comment).
    # Scenario: a compiled function obtains group size and key from a group assignment and runs one
    # all-reduce with them. Asserts the compiler text contains the expected replica group for a single
    # device, then runs the shared compiler-output comparison.
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def fn(x):
        # [FUNCTION DOC]
        # Computes group size and key from a group assignment with one device, then all-reduces the
        # argument using them; returns the result.
        group_size, group_key = collective_ops.assign_group_v2(
            group_assignment=[[0]], device_index=0, base_key=1000)
        t0 = collective_ops.all_reduce_v2(
            t=x, group_size=group_size, group_key=group_key, instance_key=1)
        return t0

      inputs = constant_op.constant([1.0, 2.0, 3.0])
      # Make sure 2 different channel ids are assigned to the 2 all-reduce
      # instructions generated by XLA.
      hlo_str = fn.experimental_get_compiler_ir(inputs)()
      self.assertIn('replica_groups={{0}}', hlo_str)
      self._compareTwoMethodsCompilerIROutput(fn, [inputs], {})

  @test_util.disable_mlir_bridge('TODO(b/155782411): MLIR bridge does not'
                                 'support stack traces')
  def testPythonLocationInMetadata(self):
    # [FUNCTION DOC]
    # Scenario (skipped for the MLIR bridge): a compiled addition function is called; asserts the
    # compiler text contains the Python function's name (source location metadata), then runs the
    # shared compiler-output comparison.
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def add_fn(x, y):
        # [FUNCTION DOC]
        # Compiled function returning the sum of its two arguments; its name should appear in metadata.
        return x + y

      inputs = constant_op.constant([1, 2, 2, 3, 3])
      self.assertIn(
          'add_fn', add_fn.experimental_get_compiler_ir(inputs, inputs)()
      )
      self._compareTwoMethodsCompilerIROutput(add_fn, [inputs, inputs], {})

  @test_util.disable_mlir_bridge('TODO(b/155782411): MLIR bridge does not'
                                 'support stack traces')
  def testPythonLocationNestedInMetadata(self):
    # [FUNCTION DOC]
    # Scenario (skipped for the MLIR bridge): a compiled function calls another compiled function.
    # Asserts the compiler text for the outer one contains the outer function's name, then runs the
    # shared compiler-output comparison.
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def add_f(x, y):
        # [FUNCTION DOC]
        # Compiled inner function returning the sum of its arguments.
        return x + y

      @polymorphic_function.function(jit_compile=True)
      def add_g(x, y):
        # [FUNCTION DOC]
        # Compiled outer function that calls the inner compiled function.
        return add_f(x, y)

      inputs = constant_op.constant([1, 2, 2, 3, 3])
      self.assertIn(
          'add_g', add_g.experimental_get_compiler_ir(inputs, inputs)()
      )
      self._compareTwoMethodsCompilerIROutput(add_g, [inputs, inputs], {})

  def testPythonStackTrace(self):
    # [FUNCTION DOC]
    # Scenario: a compiled function with unsupported string ops is called. Asserts the raised
    # invalid-argument error message mentions the failing function's name, showing the Python stack
    # information reaches the error.
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def failure_fn(x):
        # [FUNCTION DOC]
        # Compiled function with string-formatting and string-length ops that XLA cannot compile.
        return string_ops.string_length(string_ops.string_format('{}', x))

      inputs = constant_op.constant([1, 2, 2, 3, 3])
      with self.assertRaisesRegex(errors.InvalidArgumentError, 'failure_fn'):
        failure_fn(inputs)

  def testPythonStackTraceUncompiledWithinCompiled(self):
    # [FUNCTION DOC]
    # Scenario: an uncompiled failing function is called from a compiled outer function. Asserts the
    # invalid-argument error message mentions the outer function's name.
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function
      def failure_fn(x):
        # [FUNCTION DOC]
        # Uncompiled function containing string ops that fail under compilation.
        return string_ops.string_length(string_ops.string_format('{}', x))

      @polymorphic_function.function(jit_compile=True)
      def outer(x):
        # [FUNCTION DOC]
        # Compiled outer function that calls the failing uncompiled function.
        return failure_fn(x)

      inputs = constant_op.constant([1, 2, 2, 3, 3])
      with self.assertRaisesRegex(errors.InvalidArgumentError, 'outer'):
        outer(inputs)

  @test_util.disable_mlir_bridge('TODO(b/155782411): MLIR bridge does not'
                                 'support stack traces')
  def testPythonStackTraceCompiledWithinUncompiled(self):
    # [FUNCTION DOC]
    # Scenario (skipped for the MLIR bridge): a compiled failing function is called from an uncompiled
    # outer function. Asserts the invalid-argument error message mentions the inner failing function's
    # name.
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def failure_fn(x):
        # [FUNCTION DOC]
        # Compiled function with string ops that XLA cannot compile.
        return string_ops.string_length(string_ops.string_format('{}', x))

      @polymorphic_function.function
      def outer(x):
        # [FUNCTION DOC]
        # Uncompiled outer function that calls the failing compiled function.
        return failure_fn(x)

      inputs = constant_op.constant([1, 2, 2, 3, 3])
      with self.assertRaisesRegex(errors.InvalidArgumentError, 'failure_fn'):
        outer(inputs)

  @test_util.disable_mlir_bridge('TODO(b/155782411): MLIR bridge does not'
                                 'support stack traces')
  def testPythonStackTraceCompiledWithinCompiled(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name says compiled-within-compiled, but the outer function is declared without
    # compilation, so it duplicates the compiled-within-uncompiled test.
    # Scenario (skipped for the MLIR bridge): a compiled failing function is called from an outer
    # function. Asserts the error message mentions the inner function's name.
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def failure_fn(x):
        # [FUNCTION DOC]
        # Compiled function with string ops that XLA cannot compile.
        return string_ops.string_length(string_ops.string_format('{}', x))

      @polymorphic_function.function
      def outer(x):
        # [FUNCTION DOC]
        # Outer function (declared without compilation) that calls the failing compiled function.
        return failure_fn(x)

      inputs = constant_op.constant([1, 2, 2, 3, 3])
      with self.assertRaisesRegex(errors.InvalidArgumentError, 'failure_fn'):
        outer(inputs)

  def testFunctionGradient(self):
    # [FUNCTION DOC]
    # Scenario: a function multiplying a variable (value two) by its input is wrapped without and with
    # compilation. For each wrapper, the gradient with respect to the variable is taken. Asserts the
    # output is six and the gradient is three in both cases.
    with ops.device('device:{}:0'.format(self.device)):
      v = resource_variable_ops.ResourceVariable(2.0)

      def fn(x):
        # [FUNCTION DOC]
        # Returns the variable multiplied by the input.
        return v * x

      func = polymorphic_function.function(fn, jit_compile=False)
      xla_func = polymorphic_function.function(fn, jit_compile=True)

      def run_and_check(test_func):
        # [FUNCTION DOC]
        # Runs the given wrapped function on a constant three under a gradient tape, takes the gradient of
        # the output with respect to the variable, and asserts the output is six and the gradient three.
        # Input: a callable. Returns nothing.
        x = constant_op.constant(3.0)
        with backprop.GradientTape() as tape:
          y = test_func(x)
        dy = tape.gradient(y, v)

        self.assertAllClose(6.0, y)
        self.assertAllClose(3.0, dy)

      run_and_check(func)
      run_and_check(xla_func)

  @test_util.disable_mlir_bridge('TODO(b/162521846): MLIR bridge fails'
                                 ' msan, function library not found')
  def testControlFlow(self):
    # [FUNCTION DOC]
    # Scenario (skipped for the MLIR bridge): a compiled function with a loop and a conditional is
    # differentiated inside another compiled function. Checks the XLA context is propagated by tracing
    # and calling the garbage-collected concrete function, then asserts the loop result (forty) and the
    # function-and-gradient results (forty and twenty-eight), both directly and through concrete
    # functions.

    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def f(x):
        # [FUNCTION DOC]
        # Compiled function that first asserts the graph is within an XLA context, converts its input to
        # a tensor and runs a bounded while loop whose body picks between two updates depending on the
        # counter. Returns the loop's accumulated value.
        assert control_flow_util.GraphOrParentsInXlaContext(
            ops.get_default_graph())
        x = ops.convert_to_tensor(x)

        def body(i, a):
          # [FUNCTION DOC]
          # Loop body: increments the counter and updates the accumulator using a conditional that adds the
          # square of the input once the counter exceeds two, otherwise adds three. Returns both values.
          return i + 1, cond.cond(i > 2, lambda: a + (x**2),
                                  lambda: a + 3)

        return while_loop.while_loop(
            lambda i, *_: i < 10,
            body, (constant_op.constant(0), constant_op.constant(3.)),
            maximum_iterations=10)[1]

      @polymorphic_function.function(jit_compile=True)
      def g(x):
        # [FUNCTION DOC]
        # Compiled function that computes the first function under a gradient tape and returns the output
        # together with the gradient with respect to the input.
        x = ops.convert_to_tensor(x)
        with backprop.GradientTape() as tape:
          tape.watch(x)
          y = f(x)
        return y, tape.gradient(y, x)

      # Test that XLA context gets correctly propagated.
      g._get_concrete_function_garbage_collected(2.0)(2.0)

      self.assertAllClose(40.0, f(2.0))
      self.assertAllClose([40.0, 28.0], g(2.0))
      self.assertAllClose(40.0, f.get_concrete_function(2.0)(2.0))
      self.assertAllClose([40.0, 28.0], g.get_concrete_function(2.0)(2.0))

  def testWhileLoopWithUnmodifiedCarriedShape(self):
    # [FUNCTION DOC]
    # Scenario: a compiled function with an unknown-length vector signature runs a while loop that
    # carries the dynamic shape as loop state and draws random values of that shape. Asserts the result
    # for a zero vector of length seven is everywhere positive, i.e. shape constness propagates.
    with ops.device('device:{}:0'.format(self.device)):
      signature = [tensor.TensorSpec(shape=[None], dtype=dtypes.float32)]

      # We define a signature that specifies unknown vector shape, then test
      # that tf.shape constness gets properly propagated into the while_loop
      # even when carried as part of the loop state.
      @polymorphic_function.function(
          input_signature=signature, jit_compile=True)
      def g(x):
        # [FUNCTION DOC]
        # Compiled function: runs a three-iteration while loop carrying the input and its dynamic shape,
        # adding squared random noise each round; returns the first loop variable.
        return while_loop.while_loop_v2(
            lambda *_: True,
            lambda y, shp: (y + random_ops.random_normal(shp)**2, shp),
            (x, array_ops.shape(x)),
            maximum_iterations=3)[0]

      self.assertAllGreater(g(array_ops.zeros([7])), 0.)

  def testNestedWhileLoopWithUnmodifiedCarriedShape(self):
    # [FUNCTION DOC]
    # Scenario: like the previous loop test, but with nested while loops that carry the dynamic shape,
    # including an identity op between the loops. Asserts the output for a zero vector of length seven
    # is everywhere positive.
    with ops.device('device:{}:0'.format(self.device)):
      signature = [tensor.TensorSpec(shape=[None], dtype=dtypes.float32)]

      @polymorphic_function.function(
          input_signature=signature, jit_compile=True)
      def g(x):
        # [FUNCTION DOC]
        # Compiled function that runs an outer while loop (four iterations) carrying the input and its
        # dynamic shape; returns the first loop variable.

        def inner(z, shp):
          # [FUNCTION DOC]
          # Inner loop body: adds squared random noise of the carried shape; returns the value and the shape.
          return z + random_ops.random_normal(shp)**2, shp

        def outer(y, shp):
          # [FUNCTION DOC]
          # Outer loop body: runs the inner loop for three iterations, passes both values through an
          # identity op, then runs the inner loop for five iterations and returns the final loop variables.
          y, shp = while_loop.while_loop_v2(
              lambda *_: True, inner, (y, shp), maximum_iterations=3)
          y, shp = array_ops.identity_n([y, shp])
          return while_loop.while_loop_v2(
              lambda *_: True, inner, (y, shp), maximum_iterations=5)

        shp = array_ops.shape(x, name='x_shp')
        return while_loop.while_loop_v2(
            lambda *_: True, outer, (x, shp), maximum_iterations=4)[0]

      self.assertAllGreater(g(array_ops.zeros([7])), 0.)

  def testNestedWhileLoopWithUnmodifiedCarriedShapeSlice(self):
    # [FUNCTION DOC]
    # Scenario: nested while loops carrying a dynamic two-dimensional shape, followed by a second loop
    # that uses a slice of the shape taken after the first loop. Asserts the result for a zero matrix
    # of seven by thirteen is everywhere positive.
    with ops.device('device:{}:0'.format(self.device)):
      signature = [
          tensor.TensorSpec(shape=[None, None], dtype=dtypes.float32)
      ]

      @polymorphic_function.function(
          input_signature=signature, jit_compile=True)
      def g(x):
        # [FUNCTION DOC]
        # Compiled function: runs the outer loop on the input and its shape, then runs it again on zeros
        # shaped like the first row with a sliced shape, and returns the sum of both results.

        def inner(z, shp):
          # [FUNCTION DOC]
          # Inner loop body: adds squared random noise of the carried shape; returns the value and shape.
          return z + random_ops.random_normal(shp)**2, shp

        def outer(y, shp):
          # [FUNCTION DOC]
          # Outer loop body: runs the inner loop three iterations then four iterations, returning the final
          # loop variables.
          y, shp = while_loop.while_loop_v2(
              lambda *_: True, inner, (y, shp), maximum_iterations=3)
          return while_loop.while_loop_v2(
              lambda *_: True, inner, (y, shp), maximum_iterations=4)

        shp = array_ops.shape(x, name='x_shp')
        x = while_loop.while_loop_v2(
            lambda *_: True, outer, (x, shp), maximum_iterations=5)[0]

        shp2 = array_ops.shape(x, name='x_shp_after')[1:]
        w = while_loop.while_loop_v2(
            lambda *_: True,
            outer, (array_ops.zeros_like(x[0]), shp2),
            maximum_iterations=6)[0]
        return x + w

      self.assertAllGreater(g(array_ops.zeros([7, 13])), 0.)

  def testMethodCompilation(self):
    # [FUNCTION DOC]
    # Scenario: a compiled method of a local class is called on an instance with a tensor and a number.
    # Asserts the expected elementwise sum.

    with ops.device('device:{}:0'.format(self.device)):

      class C(object):
        # [CLASS DOC]
        # Local helper class holding a single method that is wrapped for compilation.

        @polymorphic_function.function(jit_compile=True)
        def f1(self, x, a):
          # [FUNCTION DOC]
          # Compiled method: returns the sum of its two arguments (self is unused).
          return x + a

      inputs = constant_op.constant([1, 2, 2, 3, 3])
      c = C()
      self.assertAllClose([2, 3, 3, 4, 4], c.f1(inputs, 1))

  def testMethodCompilationUnsupportedFunc(self):
    # [FUNCTION DOC]
    # Scenario: a compiled method of a local class contains unsupported string ops. Asserts calling it
    # raises an invalid-argument error mentioning unsupported operations.
    with ops.device('device:{}:0'.format(self.device)):

      class C(object):
        # [CLASS DOC]
        # Local helper class holding a method with unsupported ops, wrapped for compilation.

        @polymorphic_function.function(jit_compile=True)
        def f1(self, x):
          # [FUNCTION DOC]
          # Compiled method returning the length of the string formed by formatting its argument; XLA cannot
          # compile this.
          return string_ops.string_length(
              string_ops.string_format('{}', x))

      inputs = constant_op.constant([1, 2, 2, 3, 3])
      c = C()
      with self.assertRaisesRegex(
          errors.InvalidArgumentError, 'unsupported operations'
      ):
        c.f1(inputs)

  def testMustBeConstantPropagation(self):
    # [FUNCTION DOC]
    # Scenario: a compiled function needing a compile-time constant argument (a transpose permutation)
    # gets it from another compiled function's output, called from an uncompiled function. Skipped on
    # TPU. Passes if the call completes without error; no value is asserted.
    if 'tpu' in self.device.lower():
      self.skipTest('b/162799319: Cannot resolve constant on TPU')

    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def f():
        # [FUNCTION DOC]
        # Compiled function returning a constant integer vector used as the permutation.
        return constant_op.constant([0, 2, 1], dtype=dtypes.int32)

      @polymorphic_function.function(jit_compile=True)
      def g(a, b):
        # [FUNCTION DOC]
        # Compiled function transposing the first argument using the second as permutation.
        return array_ops.transpose(a, b)

      @polymorphic_function.function
      def z():
        # [FUNCTION DOC]
        # Uncompiled function that calls the transposing function with a ones tensor and the permutation
        # produced by the constant function.
        return g(array_ops.ones([3, 4, 3], dtype=dtypes.float32), f())

      z()

  def testArgMinMax(self):
    # [FUNCTION DOC]
    # Scenario: compiled argmax and argmin functions are applied to a vector of ones, in both explicit
    # float32 and default dtype. Asserts the result is index zero in every case.
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def argmax(x):
        # [FUNCTION DOC]
        # Compiled function returning the index of the maximum value of its input.
        return math_ops.argmax(x)

      @polymorphic_function.function(jit_compile=True)
      def argmin(x):
        # [FUNCTION DOC]
        # Compiled function returning the index of the minimum value of its input.
        return math_ops.argmin(x)

      self.assertAllClose(0, argmax(array_ops.ones([10], dtype=dtypes.float32)))
      self.assertAllClose(0, argmax(array_ops.ones([10])))
      self.assertAllClose(0, argmin(array_ops.ones([10], dtype=dtypes.float32)))
      self.assertAllClose(0, argmin(array_ops.ones([10])))

  @test_util.disable_mlir_bridge('TensorArray support not implemented')
  def testErrorMessagePassingTensorArray(self):
    # [FUNCTION DOC]
    # Scenario (skipped for the MLIR bridge): a compiled function writes into and reads from a tensor
    # array, and is called under a gradient tape with the gradient requested. Asserts an unimplemented
    # error mentioning a list crossing the XLA/TF boundary.
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def f(x):
        # [FUNCTION DOC]
        # Compiled function that stores twice its input in a one-element tensor array, reads it back and
        # returns it.
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32, size=1, element_shape=[])
        ta = ta.write(0, 2 * x)
        y = ta.read(0)
        return y

      x = constant_op.constant(3.14)
      with backprop.GradientTape() as tape:
        tape.watch(x)
        with self.assertRaisesRegex(errors.UnimplementedError,
                                    'TensorList crossing the XLA/TF boundary'):
          y = f(x)
          tape.gradient(y, x)

  @test_util.disable_mlir_bridge('TODO(b/162281863): MLIR bridge errors out'
                                 ' lowering TensorListConcatV2')
  def testTensorListConcatV2(self):
    # [FUNCTION DOC]
    # Scenario (skipped for the MLIR bridge): a function writes two scaled copies of a vector to a
    # tensor array and concatenates them. Asserts the plain result equals the expected six values and
    # that the compiled version matches the plain one.
    with ops.device('device:{}:0'.format(self.device)):

      def f(x):
        # [FUNCTION DOC]
        # Writes two and three times the input into a two-element tensor array and returns the
        # concatenation.
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32, size=2, element_shape=[3])
        ta = ta.write(0, 2 * x)
        ta = ta.write(1, 3 * x)
        return ta.concat()

      compiled_f = polymorphic_function.function(jit_compile=True)(f)

      inputs = constant_op.constant([3.14, 2.68, 7.69])

      self.assertAllClose([6.28, 5.36, 15.38, 9.42, 8.04, 23.07], f(inputs))

      self.assertAllClose(compiled_f(inputs), f(inputs))

  @test_util.disable_mlir_bridge('TODO(b/162281863): MLIR bridge errors out'
                                 ' lowering TensorListConcatV2')
  def testTensorListConcatV2Multidim(self):
    # [FUNCTION DOC]
    # Scenario (skipped for the MLIR bridge): same as the previous concat test but with a matrix input
    # and element shape three by two. Asserts the compiled and plain results match.
    with ops.device('device:{}:0'.format(self.device)):

      def f(x):
        # [FUNCTION DOC]
        # Writes two and three times the input into a tensor array of matrices and returns the
        # concatenation.
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32, size=2, element_shape=[3, 2])
        ta = ta.write(0, 2 * x)
        ta = ta.write(1, 3 * x)
        return ta.concat()

      compiled_f = polymorphic_function.function(jit_compile=True)(f)

      inputs = constant_op.constant([[3.14, 21.1], [2.68, 22.2], [7.69, 23.3]])
      self.assertAllClose(f(inputs), compiled_f(inputs))

  @test_util.disable_mlir_bridge('TODO(b/162281863): MLIR bridge errors out'
                                 ' lowering TensorListConcatV2')
  def testTensorListConcatV2Scalars(self):
    # [FUNCTION DOC]
    # Scenario (skipped for the MLIR bridge): concatenation of tensor array elements of length one.
    # Asserts the compiled and plain results match.
    with ops.device('device:{}:0'.format(self.device)):

      def f(x):
        # [FUNCTION DOC]
        # Writes two and three times the input into a tensor array of single-element vectors and returns
        # the concatenation.
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32, size=2, element_shape=[1])
        ta = ta.write(0, 2 * x)
        ta = ta.write(1, 3 * x)
        return ta.concat()

      compiled_f = polymorphic_function.function(jit_compile=True)(f)
      inputs = constant_op.constant([3.14])
      self.assertAllClose(f(inputs), compiled_f(inputs))

  @test_util.disable_mlir_bridge('TODO(b/162281863): MLIR bridge errors out'
                                 ' lowering TensorListConcatV2')
  def testTensorListConcatGrad(self):
    # [FUNCTION DOC]
    # Scenario (skipped for the MLIR bridge): the gradient of a tensor-array concatenation is taken
    # under a gradient tape. Asserts the plain gradient is five for every element and that the compiled
    # version matches.
    with ops.device('device:{}:0'.format(self.device)):

      def f(x):
        # [FUNCTION DOC]
        # Writes two and three times the input into a tensor array and returns the concatenation.
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32, size=2, element_shape=[3])
        ta = ta.write(0, 2 * x)
        ta = ta.write(1, 3 * x)
        return ta.concat()

      def g():
        # [FUNCTION DOC]
        # Builds a constant vector, applies the concatenating function under a gradient tape and returns
        # the gradient with respect to the vector.
        x = constant_op.constant([3.14, 2.68, 7.69])
        with backprop.GradientTape() as tape:
          tape.watch(x)
          y = f(x)
          return tape.gradient(y, x)

      compiled_g = polymorphic_function.function(jit_compile=True)(g)

      self.assertAllClose([5.0, 5.0, 5.0], g())
      self.assertAllClose(compiled_g(), g())

  @test_util.disable_mlir_bridge('TODO(b/162281863): MLIR bridge errors out'
                                 ' lowering TensorListConcatV2')
  def testTensorListConcatGradNestedCompile(self):
    # [FUNCTION DOC]
    # Scenario (skipped for the MLIR bridge): a compiled function containing the tensor-array
    # concatenation is differentiated inside another compiled function. Asserts the gradient is five
    # for every element.
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def f(x):
        # [FUNCTION DOC]
        # Compiled function that writes two and three times the input into a tensor array and returns the
        # concatenation.
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32, size=2, element_shape=[3])
        ta = ta.write(0, 2 * x)
        ta = ta.write(1, 3 * x)
        return ta.concat()

      @polymorphic_function.function(jit_compile=True)
      def g():
        # [FUNCTION DOC]
        # Compiled function that differentiates the first function with respect to a constant vector and
        # returns the gradient.
        x = constant_op.constant([3.14, 2.68, 7.69])
        with backprop.GradientTape() as tape:
          tape.watch(x)
          y = f(x)
          out = tape.gradient(y, x)
        return out

      self.assertAllClose([5.0, 5.0, 5.0], g())

  def testCumsum(self):
    # [FUNCTION DOC]
    # Scenario: a compiled cumulative-sum function is called with a 64-bit float vector. Skipped on
    # TPU. Asserts the expected running sums.
    if 'tpu' in self.device.lower():
      self.skipTest('b/162771302: 64bit rewrite of cumsum not supported')

    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def f(x):
        # [FUNCTION DOC]
        # Compiled function returning the cumulative sum of its input.
        return math_ops.cumsum(x)

      f64_input = constant_op.constant([1.1, 2.2, 3.3], dtype=dtypes.float64)
      self.assertAllClose([1.1, 3.3, 6.6], f(f64_input))

  def testNoExcessiveRetracing(self):
    # [FUNCTION DOC]
    # Scenario: an outer function calling a compiled inner function is wrapped and called twice with
    # the same input. Asserts the inner function was traced only once, i.e. no excessive retracing.
    with ops.device('device:{}:0'.format(self.device)):
      inner_retracings = 0

      @polymorphic_function.function(jit_compile=True)
      def inner(a, b):
        # [FUNCTION DOC]
        # Compiled function that increments an outer trace counter (a side effect that happens only while
        # tracing) and returns a times b plus a.
        nonlocal inner_retracings
        inner_retracings += 1
        return a * b + a

      def outer(a, b):
        # [FUNCTION DOC]
        # Plain function that calls the compiled inner function.
        return inner(a, b)

      func_input = random_ops.random_normal([10, 10])
      for _ in range(2):
        polymorphic_function.function(outer)(func_input, func_input)

      self.assertEqual(inner_retracings, 1)

  def testUpdateVariable(self):
    # [FUNCTION DOC]
    # Scenario: a compiled function assigns new values to a variable. After calling it, asserts the
    # variable holds the assigned values.
    with ops.device('device:{}:0'.format(self.device)):
      v = variables.Variable([0.0, 0.0])

      @polymorphic_function.function(jit_compile=True)
      def f():
        # [FUNCTION DOC]
        # Compiled function with no inputs that assigns a fixed vector to the variable.
        v.assign([3.1, 2.3])

      f()
      self.assertAllClose(v, [3.1, 2.3])

  @test_util.disable_mlir_bridge('MLIR does not support resource update for'
                                 ' signature with compile-time constant.')
  def testUniqueDifferentSizes(self):
    # [FUNCTION DOC]
    # Scenario (skipped for the MLIR bridge, and unless running on GPU): a compiled function applies
    # the unique op to two inputs and adds the results. It works when the unique results have equal
    # size and asserts an internal error mentioning different size when they differ.
    if not 'gpu' in self.device.lower():
      self.skipTest('Currently works only on GPU')

    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def f(x, y):
        # [FUNCTION DOC]
        # Compiled function that adds the unique values of the two inputs.
        return array_ops.unique(x).y + array_ops.unique(y).y

      f(constant_op.constant([3.1, 3.2]), constant_op.constant([3.3, 3.2]))

      with self.assertRaisesRegex(errors.InternalError, 'different size'):
        f(
            constant_op.constant([3.1, 3.2]),
            constant_op.constant([3.1, 3.2, 3.3]))

  def testUniqueCompilability(self):
    # [FUNCTION DOC]
    # Scenario: a compiled function applying the unique op is called with repeated values. Asserts the
    # result contains the distinct values.
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def f(x):
        # [FUNCTION DOC]
        # Compiled function returning the unique values of its input.
        return array_ops.unique(x).y

      self.assertAllClose(f(constant_op.constant([3.1, 3.2, 3.2])), [3.1, 3.2])

  def testUpdateVariableMemoryUsage(self):
    # [FUNCTION DOC]
    # Scenario: a compiled function adds a product to a variable. On GPU, device memory use is
    # measured before and after the call (with garbage collection); on other devices both are zero.
    # Asserts memory usage is unchanged, i.e. the update does not leak or copy memory.
    with ops.device('device:{}:0'.format(self.device)):

      on_gpu = 'gpu' in self.device.lower()
      v = variables.Variable([3.1, 3.2])

      @polymorphic_function.function(jit_compile=True)
      def update_var(a, b):
        # [FUNCTION DOC]
        # Compiled function that adds the product of its inputs to the variable in place; returns nothing.
        v.assign_add(a * b)

      arg1 = random_ops.random_normal([2])
      arg2 = random_ops.random_normal([2])

      gc.collect()
      initial_usage = context.context().get_memory_info(
          v.device)['current'] if on_gpu else 0
      update_var(arg1, arg2)
      gc.collect()
      final_usage = context.context().get_memory_info(
          v.device)['current'] if on_gpu else 0
      self.assertEqual(initial_usage, final_usage)

  @test_util.disable_mlir_bridge('TODO(b/162381930): MLIR bridge renames '
                                 ' functions')
  def testUpdateVariableInClass(self):
    # [FUNCTION DOC]
    # Scenario (skipped for the MLIR bridge): a compiled method of a local class lazily creates a
    # variable on the instance and adds to it, and is invoked from an uncompiled function. Asserts the
    # variable ends at the expected value.
    with ops.device('device:{}:0'.format(self.device)):

      class C(object):
        # [CLASS DOC]
        # Local helper class whose compiled method creates and updates an instance variable.

        @polymorphic_function.function(jit_compile=True)
        def update_var(self, a, b):
          # [FUNCTION DOC]
          # Compiled method: creates the variable on the instance on first use, then adds the product of
          # the two arguments to it.
          if not hasattr(self, 'v'):
            self.v = variables.Variable(3.1)
          self.v.assign_add(a * b)

      c = C()

      @polymorphic_function.function
      def outer():
        # [FUNCTION DOC]
        # Uncompiled function calling the compiled method with two constants.
        c.update_var(constant_op.constant(0.7), constant_op.constant(0.6))

      outer()
      self.assertAllClose(c.v, 3.52)

  def testUpdateVariableMultipleOutputs(self):
    # [FUNCTION DOC]
    # Scenario: a compiled function updates a variable and also returns a value depending on the
    # updated variable. Asserts the variable and the returned value are as expected.
    with ops.device('device:{}:0'.format(self.device)):
      v = variables.Variable(3.1)

      @polymorphic_function.function(jit_compile=True)
      def update_var(a, b):
        # [FUNCTION DOC]
        # Adds the product of the arguments to the variable and returns the product plus the variable.
        v.assign_add(a * b)
        return a * b + v

      out = update_var(constant_op.constant(0.7), constant_op.constant(0.6))
      self.assertAllClose(v, 3.52)
      self.assertAllClose(out, 3.94)

  def testReturnIdentity(self):
    # [FUNCTION DOC]
    # Scenario: a compiled function simply returns its two inputs. On GPU, memory is measured before
    # and after the call; asserts usage is unchanged (no extra copies).
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def f(a, b):
        # [FUNCTION DOC]
        # Compiled function returning its two arguments unchanged as a pair.
        return (a, b)

      a = random_ops.random_normal([10, 10])
      b = random_ops.random_normal([10, 10])

      on_gpu = 'gpu' in self.device.lower()
      gc.collect()
      initial_usage = context.context().get_memory_info(
          b.backing_device)['current'] if on_gpu else 0

      f(a, b)

      gc.collect()
      final_usage = context.context().get_memory_info(
          b.backing_device)['current'] if on_gpu else 0
      self.assertEqual(initial_usage, final_usage)

  def testGetCompilerIrConstants(self):
    # [FUNCTION DOC]
    # Scenario: fetches the optimized compiler text for a compiled transpose whose permutation is a
    # constant. Skipped on TPU. Asserts the text contains the expected layout for the transpose.
    if 'tpu' in self.device.lower():
      self.skipTest('TPU generates different HLO')

    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def f(a, b):
        # [FUNCTION DOC]
        # Compiled function transposing its first argument by the second.
        return array_ops.transpose(a, b)

      a = array_ops.ones([3, 4, 3], dtype=dtypes.float32)
      b = constant_op.constant([0, 2, 1], dtype=dtypes.int32)

      self.assertIn('{2,1,0}',
                    f.experimental_get_compiler_ir(a, b)(stage='optimized_hlo'))

  @test_util.disable_mlir_bridge('TODO(b/168732524): MLIR bridge does not '
                                 ' optimize single-element tuples to scalars')
  def testGetCompilerIrResourceVars(self):
    # [FUNCTION DOC]
    # Scenario (skipped for the MLIR bridge): fetches optimized compiler text for a compiled function
    # updating a variable. Asserts it contains an input/output alias entry for the resource variable.
    with ops.device('device:{}:0'.format(self.device)):

      v = variables.Variable([3.1, 3.2])

      @polymorphic_function.function(jit_compile=True)
      def f(a, b):
        # [FUNCTION DOC]
        # Compiled function that adds the product of its inputs to the variable.
        v.assign_add(a * b)

      a = random_ops.random_normal([2])
      b = random_ops.random_normal([2])

      self.assertIn('input_output_alias={ {}: (2, {}, may-alias) }',
                    f.experimental_get_compiler_ir(a, b)('optimized_hlo'))

  def testGetCompilerIrNotCompiled(self):
    # [FUNCTION DOC]
    # Scenario: requesting compiler output from a function that is not marked for compilation.
    # Asserts a value error mentioning the missing compile marking.
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function
      def f(x):
        # [FUNCTION DOC]
        # Uncompiled function returning its input plus one.
        return x + 1

      a = random_ops.random_normal([10, 10])
      with self.assertRaisesRegex(ValueError,
                                  'marked with \'jit_compile'):
        f.experimental_get_compiler_ir(a)()

  def testGetCompilerIrNested(self):
    # [FUNCTION DOC]
    # Scenario: an uncompiled function tries to request compiler output of a compiled function while
    # being traced. Asserts a type error is raised when it is called.
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def fn(x, a):
        # [FUNCTION DOC]
        # Compiled function returning the sum of its arguments.
        return x + a

      @polymorphic_function.function(jit_compile=False)
      def fn2(x, a):
        # [FUNCTION DOC]
        # Uncompiled function that requests compiler output for the compiled function and then calls it.
        fn.experimental_get_compiler_ir(x, a)()
        return fn(x, a)

      inputs = constant_op.constant([1, 2, 2, 3, 3])
      with self.assertRaises(TypeError):
        fn2(inputs, 1)

  def testGetCompilerIrKwargs(self):
    # [FUNCTION DOC]
    # Scenario: compiler text is requested using keyword arguments in swapped order for a compiled
    # function that multiplies by a variable. Asserts the text contains a multiply instruction.
    with ops.device('device:{}:0'.format(self.device)):

      v = variables.Variable([0.1, 0.1])

      @polymorphic_function.function(jit_compile=True)
      def f(a, b):
        # [FUNCTION DOC]
        # Compiled function returning the sum of its arguments multiplied by the variable.
        return (a + b) * v

      a = constant_op.constant([1.1, 1.1])
      b = constant_op.constant([2.2, 2.2])

      self.assertIn('multiply',
                    f.experimental_get_compiler_ir(b=a, a=b)(stage='hlo'))

  def testGetCompilerIrDot(self):
    # [FUNCTION DOC]
    # Scenario: compiler text in the dot-graph stage is requested for a compiled addition. Asserts it
    # contains a label, then runs the shared comparison of the two retrieval methods.
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def f(a, b):
        # [FUNCTION DOC]
        # Compiled function returning the sum of its arguments.
        return a + b

      a = constant_op.constant([1.1, 1.1])
      b = constant_op.constant([2.2, 2.2])

      self.assertIn(
          'label',
          f.experimental_get_compiler_ir(a, b)(stage='optimized_hlo_dot'))
      self._compareTwoMethodsCompilerIROutput(f, [a, b], {})

  def testGetCompilerIrNoDevicePlacement(self):
    # [FUNCTION DOC]
    # Scenario: same as the dot-graph test but with no explicit device placement. Skipped unless on a
    # GPU. Asserts the text contains a label, then runs the shared comparison.
    if 'gpu' not in self.device.lower():
      self.skipTest('Testing get_compiler_ir on GPUs without placement')

    @polymorphic_function.function(jit_compile=True)
    def f(a, b):
      # [FUNCTION DOC]
      # Compiled function returning the sum of its arguments.
      return a + b

    a = constant_op.constant([1.1, 1.1])
    b = constant_op.constant([2.2, 2.2])

    self.assertIn(
        'label',
        f.experimental_get_compiler_ir(a, b)(stage='optimized_hlo_dot'))
    self._compareTwoMethodsCompilerIROutput(f, [a, b], {})

  def testGetCompilerIrNonTensors(self):
    # [FUNCTION DOC]
    # Scenario: compiler text is requested with a list of two tensors as one argument. Asserts the
    # text contains a tuple, then runs the shared comparison (which is skipped for non-tensor arguments).
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def f(l):
        # [FUNCTION DOC]
        # Compiled function returning the sum of the first two elements of a list argument.
        return l[0] + l[1]

      l = [constant_op.constant(1.1), constant_op.constant(2.2)]

      self.assertIn('tuple',
                    f.experimental_get_compiler_ir(l)())
      self._compareTwoMethodsCompilerIROutput(f, [l], {})

  def testGetCompilerIrSerialized(self):
    # [FUNCTION DOC]
    # Scenario: serialized compiler output is requested for both the unoptimized and optimized stages,
    # with an explicit device name. Asserts each result is a byte string, then runs the shared
    # comparison.
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def fn(x):
        # [FUNCTION DOC]
        # Compiled function returning the difference of its input with itself.
        return x - x

      inputs = constant_op.constant([1, 2, 2, 3, 3])
      for stage in ('hlo_serialized', 'optimized_hlo_serialized'):
        hlo = fn.experimental_get_compiler_ir(inputs)(
            stage=stage, device_name=f'/device:{self.device}:0')
        self.assertIsInstance(hlo, bytes)
      self._compareTwoMethodsCompilerIROutput(fn, [inputs], {})

  def testDotOptimizedHlo(self):
    # [FUNCTION DOC]
    # Scenario: optimized compiler text for a compiled matrix multiplication is fetched. Asserts it
    # matches a pattern containing dot or convolution, with custom-call additionally allowed when MKL
    # is enabled.
    with ops.device('device:{}:0'.format(self.device)):

      a = random_ops.random_normal([100, 100])
      b = random_ops.random_normal([100, 100])

      @polymorphic_function.function(jit_compile=True)
      def f(a, b):
        # [FUNCTION DOC]
        # Compiled function returning the matrix product of its arguments.
        return math_ops.matmul(a, b)

      if not test_util.IsMklEnabled():
        self.assertRegex(
            f.experimental_get_compiler_ir(a, b)('optimized_hlo'),
            '(dot)|(convolution)',
        )
      else:
        self.assertRegex(
            f.experimental_get_compiler_ir(a, b)('optimized_hlo'),
            '(dot)|(convolution)|(custom-call)',
        )

  def testConstantOnWrongDevice(self):
    # [FUNCTION DOC]
    # Scenario: a compiled reshape uses a shape computed from random values on the device. Fetches
    # compiler text and asserts it contains a tuple.
    with ops.device('device:{}:0'.format(self.device)):

      s = random_ops.random_uniform([2], 1, 10, dtypes.int32)
      l = random_ops.random_normal([s[0] * s[1]])

      @polymorphic_function.function(jit_compile=True)
      def f(l):
        # [FUNCTION DOC]
        # Compiled function reshaping its input to the outer shape tensor.
        return array_ops.reshape(l, s)

      self.assertIn('tuple',
                    f.experimental_get_compiler_ir(l)())

  @test_util.disable_mlir_bridge('TODO(b/172845417): MLIR bridge does not '
                                 'support getting constants out of resources')
  def testGetConstantOutOfResourceVariable(self):
    # [FUNCTION DOC]
    # Scenario (skipped for the MLIR bridge): a compiled reshape takes its target shape from the
    # values of two resource variables. Asserts the output shape is fifty by two, since the values are
    # known at compile time.
    with ops.device('device:{}:0'.format(self.device)):

      # Use floats to force device placement.
      a = variables.Variable(50.0)
      b = variables.Variable(2.0)

      @polymorphic_function.function(jit_compile=True)
      def f(x):
        # [FUNCTION DOC]
        # Compiled function reshaping its input to a shape given by the two variables cast to integers.
        return array_ops.reshape(
            x, [math_ops.cast(a, dtypes.int32),
                math_ops.cast(b, dtypes.int32)])

      # OK since the value is known at compile time.
      out = f(random_ops.random_normal([10, 10]))
      self.assertEqual(out.shape[0], 50)
      self.assertEqual(out.shape[1], 2)

  @test_util.disable_mlir_bridge('TODO(b/172845417): MLIR bridge does not '
                                 'support getting constants out of resources')
  def testGetConstantOutOfResourceVariableAfterWrite(self):
    # [FUNCTION DOC]
    # Scenario (skipped for the MLIR bridge): the function assigns new values to the two variables
    # before using them as the reshape target. Asserts an invalid-argument error mentioning concrete
    # values at compile time.
    with ops.device('device:{}:0'.format(self.device)):

      # Use floats to force device placement.
      a = variables.Variable(50.0)
      b = variables.Variable(2.0)

      @polymorphic_function.function(jit_compile=True)
      def f(x, val1, val2):
        # [FUNCTION DOC]
        # Compiled function that assigns the two value arguments to the variables, then reshapes the input
        # using the variables as the shape.
        a.assign(math_ops.cast(val1, dtypes.float32))
        b.assign(math_ops.cast(val2, dtypes.float32))
        return array_ops.reshape(
            x, [math_ops.cast(a, dtypes.int32),
                math_ops.cast(b, dtypes.int32)])

      val1 = constant_op.constant(2)
      val2 = constant_op.constant(50)

      # Returns an error, since the value known at compile time was overridden.
      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  'concrete values at compile time'):
        f(random_ops.random_normal([10, 10]), val1, val2)

  @test_util.disable_mlir_bridge('TODO(b/172845417): MLIR bridge does not '
                                 'support getting constants out of resources')
  def testGetConstantOutOfResourceVariableBeforeWrite(self):
    # [FUNCTION DOC]
    # Scenario (skipped for the MLIR bridge): the reshape uses the variables first and the assignments
    # happen afterwards. Asserts the call succeeds and the shape is fifty by two.
    with ops.device('device:{}:0'.format(self.device)):

      # Use floats to force device placement.
      a = variables.Variable(50.0)
      b = variables.Variable(2.0)

      @polymorphic_function.function(jit_compile=True)
      def f(x, val1, val2):
        # [FUNCTION DOC]
        # Compiled function that reshapes using the variables, then assigns the new values, and returns the
        # reshaped tensor.
        out = array_ops.reshape(
            x, [math_ops.cast(a, dtypes.int32),
                math_ops.cast(b, dtypes.int32)])
        a.assign(math_ops.cast(val1, dtypes.float32))
        b.assign(math_ops.cast(val2, dtypes.float32))
        return out

      val1 = constant_op.constant(2)
      val2 = constant_op.constant(50)

      # OK since the write happens after the reshape.
      out = f(random_ops.random_normal([10, 10]), val1, val2)
      self.assertEqual(out.shape[0], 50)
      self.assertEqual(out.shape[1], 2)

  def testTfAssert(self):
    # [FUNCTION DOC]
    # Scenario: a compiled function contains an assertion that holds. Asserts nothing explicitly;
    # passes if calling it with the value one raises no error.
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def f(x):
        # [FUNCTION DOC]
        # Compiled function that asserts its input equals one; returns nothing.
        control_flow_assert.Assert(x == 1, ['Wrong value'])

      f(constant_op.constant(1))

  def testTensorArrayErrorMessage(self):
    # [FUNCTION DOC]
    # Scenario: a compiled function uses a dynamically sized tensor array and concatenates it, which
    # fails. Asserts an invalid-argument error mentioning the function's name.
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def failure_fn():
        # [FUNCTION DOC]
        # Compiled function with no inputs that builds a dynamic-size tensor array with unknown element
        # shape and returns its concatenation.
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32,
            size=2,
            dynamic_size=True,
            element_shape=(None,),
        )
        return ta.concat()

      with self.assertRaisesRegex(errors.InvalidArgumentError, 'failure_fn'):
        failure_fn()

  def testCounter(self):
    # [FUNCTION DOC]
    # Scenario: reads the global function-tracing counters (one for uncompiled, one for compiled
    # functions) and checks that they increase when new functions are created and used: the plain
    # function raises the uncompiled count by one, a second call with a different dtype does not, a
    # compiled function raises the compiled count by one, and functions defining nested plain and
    # compiled inner functions raise the counts further. Asserts the exact counter values relative to
    # the starting values at each step.
    cell_nojit = polymorphic_function._tf_function_counter.get_cell('0')
    cell_jit = polymorphic_function._tf_function_counter.get_cell('1')
    orig_nojit = cell_nojit.value()
    orig_jit = cell_jit.value()

    with ops.device('device:{}:0'.format(self.device)):
      @polymorphic_function.function
      def f(a):
        # [FUNCTION DOC]
        # Uncompiled function returning twice its input.
        return a + a
      f(constant_op.constant(1))
      self.assertEqual(cell_nojit.value(), orig_nojit + 1)
      self.assertEqual(cell_jit.value(), orig_jit)
      f(constant_op.constant(1.))  # Calling again does not increment
      self.assertEqual(cell_nojit.value(), orig_nojit + 1)

      @polymorphic_function.function(jit_compile=True)
      def f1(a):
        # [FUNCTION DOC]
        # Compiled function returning twice its input.
        return a + a
      f1(constant_op.constant(1))
      self.assertEqual(cell_nojit.value(), orig_nojit + 1)
      self.assertEqual(cell_jit.value(), orig_jit + 1)

      @polymorphic_function.function
      def f2(a):
        # [FUNCTION DOC]
        # Uncompiled outer function that defines a plain and a compiled inner function and returns the sum
        # of their results.
        @polymorphic_function.function
        def g(a):
          # [FUNCTION DOC]
          # Uncompiled inner function returning twice its input.
          return a + a
        @polymorphic_function.function(jit_compile=True)
        def h(a):
          # [FUNCTION DOC]
          # Compiled inner function returning twice its input.
          return a + a
        return g(a) + h(a)
      f2(constant_op.constant(1))
      self.assertEqual(cell_nojit.value(), orig_nojit + 2)
      self.assertEqual(cell_jit.value(), orig_jit + 2)

      @polymorphic_function.function(jit_compile=True)
      def f3(a):
        # [FUNCTION DOC]
        # Compiled outer function that defines a plain and a compiled inner function and returns the sum
        # of their results.
        @polymorphic_function.function
        def g(a):
          # [FUNCTION DOC]
          # Uncompiled inner function returning twice its input.
          return a + a
        @polymorphic_function.function(jit_compile=True)
        def h(a):
          # [FUNCTION DOC]
          # Compiled inner function returning twice its input.
          return a + a
        return g(a) + h(a)
      f3(constant_op.constant(1))
      self.assertEqual(cell_nojit.value(), orig_nojit + 2)
      self.assertEqual(cell_jit.value(), orig_jit + 3)

  @test_util.disable_mlir_bridge('TODO(b/162272821): MLIR bridge returns '
                                 ' wrong status type')
  def testResourceWrongDevice(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: uses the deprecated option name for compilation instead of the current one used in
    # the rest of the file.
    # Scenario (skipped for the MLIR bridge, and unless a GPU is used): a variable lives on the CPU but
    # a function marked for compilation (using the older option name) updates it on the GPU. Asserts an
    # invalid-argument error saying the resource cannot be accessed.
    if 'gpu' not in self.device.lower():
      self.skipTest('Need a GPU to have non-trivial device placement')

    with ops.device('device:CPU:0'):
      v = variables.Variable([3.1, 3.2])

    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(experimental_compile=True)
      def update_var(a):
        # [FUNCTION DOC]
        # Compiled function adding its argument to the CPU-resident variable; returns nothing.
        v.assign_add(a)

      arg = random_ops.random_normal([2])
      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  'Trying to access resource .*'):
        update_var(arg)

  def testMustBeConstantInsideCondition(self):
    # [FUNCTION DOC]
    # Scenario: a compiled function reshapes differently in the two branches of a Python conditional,
    # one using a constant shape and one using a shape argument. Called with random input and a
    # constant shape; passes if no error is raised, nothing is asserted.
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def f(x, d):
        # [FUNCTION DOC]
        # Compiled function: if all elements of the input exceed random noise it reshapes twice the input
        # to a fixed shape, otherwise it reshapes three times the input to the given shape argument.
        if math_ops.reduce_all(
            math_ops.greater(x, random_ops.random_normal([10, 10]))):
          return array_ops.reshape(x * 2, constant_op.constant([100]))
        else:
          return array_ops.reshape(x * 3, d)

      f(random_ops.random_normal([10, 10]), constant_op.constant([100]))

  def testConditionalGradientTapeMathRegression(self):
    # [FUNCTION DOC]
    # Scenario: regression test for a compiled function with autograph disabled and a conditional
    # inside an active gradient tape. Asserts the result for a one-by-one matrix of two is one half
    # (the reciprocal branch is taken).
    with ops.device('device:{}:0'.format(self.device)):
      with backprop.GradientTape():

        @polymorphic_function.function(jit_compile=True, autograph=False)
        def f(x):
          # [FUNCTION DOC]
          # Compiled function (autograph disabled) that returns the reciprocal of its input when all its
          # elements exceed one, otherwise the input itself.
          return cond.cond(
              math_ops.reduce_all(x > 1), lambda: 1. / x, lambda: x)

        v = variables.Variable([[2.]])
        self.assertAllClose(f(v), constant_op.constant([[0.5]]))

  @test_util.disable_mlir_bridge('TODO(b/190444466): MLIR bridge seems to '
                                 'ignore resource assignments')
  def testErrMsgAssignWrongShape(self):
    # [FUNCTION DOC]
    # Scenario (skipped for the MLIR bridge): a compiled function assigns a wrongly shaped value to a
    # variable. Asserts the invalid-argument error says the shape cannot change after initialization,
    # and that a second call also fails with an error mentioning the function's name.
    with ops.device('device:{}:0'.format(self.device)):

      v = variables.Variable([3.1, 3.2])

      @polymorphic_function.function(jit_compile=True)
      def failure_fn(samples):
        # [FUNCTION DOC]
        # Compiled function that assigns a zeros tensor with the given shape to the variable.
        v.assign(array_ops.zeros(samples))

      with self.assertRaisesRegex(
          errors.InvalidArgumentError,
          'Shape .* cannot be changed after initialization'):
        failure_fn(constant_op.constant(6))

      with self.assertRaisesRegex(errors.InvalidArgumentError, 'failure_fn'):
        failure_fn(constant_op.constant(6))

  def testTfSummaryErrMsg(self):
    # [FUNCTION DOC]
    # Scenario: a compiled function writes a summary scalar using a file writer on the temp directory.
    # Skipped unless on a GPU. Asserts an invalid-argument error about accessing a resource.
    if 'gpu' not in self.device.lower():
      self.skipTest('Only runs on GPU')

    with ops.device('device:{}:0'.format(self.device)):
      writer = summary_ops_v2.create_file_writer(self.get_temp_dir())

      @polymorphic_function.function(jit_compile=True)
      def my_func_temp():
        # [FUNCTION DOC]
        # Compiled function with no inputs that writes a scalar summary through the writer's default scope.
        with writer.as_default():
          summary_ops_v2.scalar('my_metric', 0.5, step=10)

      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  'Trying to access resource .*'):
        my_func_temp()

  def testSinglePassArgmax(self):
    # [FUNCTION DOC]
    # Scenario: the unoptimized compiler text of a compiled argmax is fetched. Asserts the word reduce
    # appears exactly once (a single pass), then runs the shared comparison.
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def f(x):
        # [FUNCTION DOC]
        # Compiled function returning the index of the maximum of its input.
        return math_ops.argmax(x)

      inputs = array_ops.ones([10], dtype=dtypes.float32)
      hlo = f.experimental_get_compiler_ir(inputs)(stage='hlo')

      # Test that reduction occurs only once.
      self.assertEqual(hlo.count('reduce'), 1)
      self._compareTwoMethodsCompilerIROutput(f, [inputs], {})

  def testExperimentalGetCompilerIRBasic(self):
    # [FUNCTION DOC]
    # Scenario: runs the shared comparison of compiler-output retrieval methods for a compiled sine
    # function with a tensor input.
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def inner_tf_func(x):
        # [FUNCTION DOC]
        # Compiled function returning the sine of its input.
        return math_ops.sin(x)

      x = constant_op.constant([2.0, 3.0])
      self._compareTwoMethodsCompilerIROutput(inner_tf_func, [x], {})

  def testExperimentalGetCompilerIRAutograph(self):
    # [FUNCTION DOC]
    # Scenario: runs the shared comparison for a compiled function using autograph, whose Python
    # conditional on the first element is converted to a graph conditional.
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True, autograph=True)
      def f(x, y):
        # [FUNCTION DOC]
        # Compiled function returning the first element of the second argument if the first element of
        # the first argument exceeds one, else the second element of the second argument.
        if x[0] > 1:
          return y[0]
        else:
          return y[1]

      x, y = constant_op.constant([2, 3]), constant_op.constant([2, 3])
      self._compareTwoMethodsCompilerIROutput(f, [x, y], {})


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
