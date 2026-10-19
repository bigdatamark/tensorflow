# [MODULE DOC]
# GPU-only tests for "light outside compilation": TensorFlow ops that XLA cannot compile are
# called back from compiled code through a generic GPU callback custom call. The tests use
# special test ops (static, dynamic-shape, multi-output, must-be-constant input) from a test op
# library, compile functions with XLA, inspect the generated HLO text with a FileCheck wrapper,
# and compare numeric results. Everything is skipped when no GPU is available. Running the file
# as a program enables eager execution and starts the standard test runner.
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow.compiler.mlir.python.mlir_wrapper import filecheck_wrapper as fw
from tensorflow.compiler.tf2xla import test_ops_for_light_outside_compilation
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test


class LightOutsideCompilationTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for light outside compilation on GPU. Each test builds an XLA-compiled function
  # that calls a test op, then checks the HLO text via pattern matching and/or the numeric
  # output. All tests are skipped when no GPU is present (see setUp).

  def setUp(self):
    # [FUNCTION DOC]
    # Runs the parent setup, then skips the current test with a message if no GPU is available.
    super().setUp()
    if not test_util.is_gpu_available():
      self.skipTest('Light outside compilation only works for GPUs now')

  def assertFilecheck(self, actual, expected):
    # [FUNCTION DOC]
    # HELPER: shared assertion used by the tests in this class to match HLO text.
    # Inputs: actual is the text to check (usually HLO); expected is a FileCheck-style pattern
    # text with check directives. Returns nothing; fails the test, printing both the actual
    # text and the expected pattern, when the FileCheck wrapper reports no match.
    """Assert that FileCheck runs successfully."""
    if not fw.check(actual, expected):
      self.fail(f'Got output:\n{actual}\nExpected:\n{expected}')

  def test_static_tf_op(self):
    # [FUNCTION DOC]
    # Scenario: an XLA-compiled function applies the static-shape test op to a 2x2 random
    # float tensor on the first GPU. Expected: the HLO contains a custom call to the generic
    # GPU callback with matching 2x2 float shapes and mentioning the static test op, and
    # the compiled function returns the input unchanged.
    """Test operations with static shapes."""

    @def_function.function(jit_compile=True)
    def compiled_f(x):
      # [FUNCTION DOC]
      # Compiled body: applies the static test op to x and returns its result.
      return test_ops_for_light_outside_compilation.test_static_tf(x)

    with context.device('/gpu:0'):
      z = random_ops.random_normal([2, 2])

      self.assertFilecheck(
          compiled_f.experimental_get_compiler_ir(z)('hlo'), r"""
          CHECK: f32[2,2]{1,0} custom-call(f32[2,2]{1,0} [[v:.*]]), custom_call_target="GenericTfCallbackGPU"
          CHECK: TestStaticTf
          """)

      self.assertAllClose(compiled_f(z), z)

  def test_run_op_multiple_times(self):
    # [FUNCTION DOC]
    # Scenario: a compiled function with no arguments creates a random 2x2 tensor, passes it
    # through the static test op and returns both. It is called twice on the GPU.
    # Expected: in each call the op output equals its input, showing the callback works
    # repeatedly with different output buffers.
    """Call the op multiple times with different output buffers."""

    @def_function.function(jit_compile=True)
    def compiled_f():
      # [FUNCTION DOC]
      # Compiled body: makes a random 2x2 tensor, applies the static test op, returns the
      # pair of input and output.
      x = random_ops.random_normal([2, 2])
      y = test_ops_for_light_outside_compilation.test_static_tf(x)
      return (x, y)

    with context.device('/gpu:0'):
      x, y = compiled_f()
      self.assertAllClose(x, y)

      x1, y1 = compiled_f()
      self.assertAllClose(x1, y1)

  def test_unranked_output_error(self):
    # [FUNCTION DOC]
    # Scenario: a compiled function returns the output of a test op whose output has unknown
    # rank. Action: request the compiler IR on the GPU. Expected: a value error whose message
    # says output 0 has unknown rank.
    """Test that we error out for unranked dynamic shape."""

    @def_function.function(jit_compile=True)
    def compiled_f():
      # [FUNCTION DOC]
      # Compiled body: returns the result of the unranked-dynamic test op (no inputs).
      return test_ops_for_light_outside_compilation.dynamic_unranked()

    with context.device('/gpu:0'):

      with self.assertRaisesRegex(ValueError, 'Output 0 has unknown rank'):
        compiled_f.experimental_get_compiler_ir()()

  def test_dynamic_output_multidim(self):
    # [FUNCTION DOC]
    # Scenario: a compiled function takes a shape list (rank 5, each size 3,4,5,4,3) and calls
    # the multi-dimensional dynamic test op. Expected: the HLO (without metadata) has a custom
    # call whose output is a rank-5 float tensor with every dimension bounded by 20, with the
    # op name present, and the actual output equals a tensor of ones of the requested shape.
    """Test that we correctly handle multi-dimensional dynamic output."""

    @def_function.function(jit_compile=True)
    def compiled_f(shape):
      # [FUNCTION DOC]
      # Compiled body: calls the multi-dimensional dynamic test op with the given shape.
      return test_ops_for_light_outside_compilation.dynamic_multidim(shape)

    with context.device('/gpu:0'):

      # Rank is hardcoded to 5.
      shape = [3, 4, 5, 4, 3]
      hlo = compiled_f.experimental_get_compiler_ir(shape)('hlo_no_metadata')
      out = compiled_f(shape)

      self.assertFilecheck(
          hlo, r"""
          CHECK: f32[<=20,<=20,<=20,<=20,<=20]{4,3,2,1,0} custom-call(), custom_call_target="GenericTfCallbackGPU"
          CHECK: DynamicMultidim
          """)
      self.assertAllClose(out, array_ops.ones(shape))

  def test_dynamic_output_tf_op(self):
    # [FUNCTION DOC]
    # Scenario: a compiled function applies the dynamic-output test op (maximum size 5) to a
    # random 10-element vector. Expected: the HLO shows a custom call with bounded output size
    # of at most 5 and the op name; the real output is the first two elements of the input
    # and has length 2.
    # KNOWN ISSUE: none in the code itself; the expected length 2 is specific to the test op.
    """Test that dynamic output is sliced properly to the size known at runtime."""

    @def_function.function(jit_compile=True)
    def compiled_f(x):
      # [FUNCTION DOC]
      # Compiled body: calls the dynamic-output test op on x with a maximum output size of 5.
      return test_ops_for_light_outside_compilation.test_dynamic_tf(
          x, max_size=5)

    with context.device('/gpu:0'):
      z = random_ops.random_normal([10])
      out = compiled_f(z)
      hlo = compiled_f.experimental_get_compiler_ir(z)('hlo_no_metadata')

      self.assertFilecheck(
          hlo, r"""
          CHECK: f32[<=5]{0} custom-call(f32[10]{0} [[v:.*]]), custom_call_target="GenericTfCallbackGPU"
          CHECK: TestDynamicTf
          """)
      self.assertAllClose(out, z[:2])
      self.assertEqual(len(out), 2)

  def test_dynamic_input(self):
    # [FUNCTION DOC]
    # Scenario: the input is made dynamically shaped by taking the unique values of a random
    # vector, then fed to the dynamic test op. Expected: requesting the compiler IR raises a
    # value error saying input dynamic dimensions are not supported.
    # KNOWN ISSUE: the docstring says a bad status is returned at runtime, but the test
    # expects the error at compile/IR-request time.
    """Test dynamic input => returns bad status at runtime."""

    @def_function.function(jit_compile=True)
    def compiled_f(x):
      # [FUNCTION DOC]
      # Compiled body: reduces x to its unique values (dynamic size) then applies the dynamic
      # test op with maximum size 5.
      x = array_ops.unique(x).y
      return test_ops_for_light_outside_compilation.test_dynamic_tf(
          x, max_size=5)

    with context.device('/gpu:0'):
      z = random_ops.random_normal([10])

      with self.assertRaisesRegex(ValueError,
                                  'Input dynamic dimensions are not supported'):
        compiled_f.experimental_get_compiler_ir(z)()

  def test_multi_output_tf_op(self):
    # [FUNCTION DOC]
    # Scenario: a compiled function applies the multiple-output static test op to a random 2x2
    # tensor. Expected: the HLO has the generic GPU callback custom call and the op name, and
    # both outputs of the compiled function equal the input.
    """Test light outside compilation for mulitple outputs."""

    @def_function.function(jit_compile=True)
    def compiled_f(x):
      # [FUNCTION DOC]
      # Compiled body: applies the multiple-output static test op to x and returns its outputs.
      return test_ops_for_light_outside_compilation.test_static_multiple_output_tf(
          x)

    with context.device('/gpu:0'):
      z = random_ops.random_normal([2, 2])
      hlo = compiled_f.experimental_get_compiler_ir(z)('hlo_no_metadata')

      self.assertFilecheck(
          hlo, r"""
          CHECK: custom_call_target="GenericTfCallbackGPU"
          CHECK: TestStaticMultipleOutputTf
          """)
      self.assertAllClose(compiled_f(z)[0], z)
      self.assertAllClose(compiled_f(z)[1], z)

  def test_must_be_constant_tf_op(self):
    # [FUNCTION DOC]
    # Scenario: a compiled function passes a random 10-element vector and the number 5 (as a
    # compile-time constant argument) to the test op whose second input must be constant.
    # Expected: the HLO has the callback custom call on the vector and the op name, and the
    # compiled result equals each input element plus 5.
    """Test operations with must-be-constant input."""

    @def_function.function(jit_compile=True)
    def compiled_f(x, y):
      # [FUNCTION DOC]
      # Compiled body: calls the must-be-constant test op with x and the constant y.
      return test_ops_for_light_outside_compilation.test_tf_must_be_constant(
          x, constant_to_add=y)

    with context.device('/gpu:0'):

      z = random_ops.random_normal([10])
      hlo = compiled_f.experimental_get_compiler_ir(z, 5)('hlo_no_metadata')

      self.assertFilecheck(
          hlo, r"""
          CHECK: custom-call(f32[10]{0} [[v:.*]]), custom_call_target="GenericTfCallbackGPU"
          CHECK: TestTfMustBeConstant
          """)

      expected_output = [j + 5 for j in z]
      self.assertAllClose(compiled_f(z, 5), expected_output)

  def testTighterProvidedBounds(self):
    # [FUNCTION DOC]
    # Scenario: the dynamic test op variant with an explicit bound (maximum size 5) is applied
    # to a 10-element vector. Expected: the HLO custom call has output shape of 5 elements,
    # i.e. the provided bound is used instead of the looser bound shape inference would give.
    """Dynamic bounds are tighter than those deduced by shape inference."""

    @def_function.function(jit_compile=True)
    def compiled_f(x):
      # [FUNCTION DOC]
      # Compiled body: calls the bounded dynamic test op on x with maximum size 5.
      return test_ops_for_light_outside_compilation.test_dynamic_tf_with_bound(
          x, max_size=5)

    with context.device('/gpu:0'):
      z = random_ops.random_normal([10])
      hlo = compiled_f.experimental_get_compiler_ir(z)()
      self.assertFilecheck(
          hlo, r"""
          CHECK: f32[5]{0} custom-call(f32[10]{0} [[v:.*]]), custom_call_target="GenericTfCallbackGPU"
          """)

  def testFixedLayout(self):
    # [FUNCTION DOC]
    # Scenario: a convolution (random filters, stride 2 along width, same padding, channels-last)
    # feeds the static test op. Action: get the HLO for a random 1x3x4x3 input. Expected: the
    # custom call carries an operand layout constraint with the 1x3x2x2 float operand in
    # row-major layout order.
    """Test correct operand layout is fixed by the lowering."""

    @def_function.function(jit_compile=True)
    def compiled_f(conv_input):
      # [FUNCTION DOC]
      # Compiled body: builds random 2x3x3x2 filters, convolves the input, and passes the
      # convolution result to the static test op.
      filters = random_ops.random_uniform([2, 3, 3, 2])
      conv = nn_ops.conv2d(
          conv_input,
          filters,
          strides=[1, 1, 2, 1],
          dilations=[1, 1, 1, 1],
          padding='SAME',
          data_format='NHWC')
      return test_ops_for_light_outside_compilation.test_static_tf(conv)

    with context.device('/gpu:0'):
      hlo = compiled_f.experimental_get_compiler_ir(
          random_ops.random_uniform([1, 3, 4, 3]))()
      self.assertFilecheck(
          hlo, r"""
          CHECK: operand_layout_constraints={f32[1,3,2,2]{3,2,1,0}}
          """)


if __name__ == '__main__':
  ops.enable_eager_execution()
  googletest.main()
