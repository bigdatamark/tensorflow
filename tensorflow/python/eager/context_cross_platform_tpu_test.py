# [MODULE DOC]
# Test module for cross-platform compiler IR retrieval from an XLA-compiled function, run on
# a TPU host (the "TPU" test target) while requesting the GPU platform. It checks that
# unoptimized HLO stages can be produced for GPU, and that optimized HLO stages are rejected
# with a value error because they are not supported for a different device. When run as a
# script it enables eager execution and runs the test main. No import-time side effects.

# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

from absl.testing import parameterized

from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.platform import test


class ContextCrossPlatformTpuTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case covering the compiler IR query of XLA-compiled functions when the requested
  # target platform (GPU) differs from the host the test runs on (TPU). Combines the framework
  # test case with parameterized test support; holds no fixtures.

  @parameterized.named_parameters(
      [(f'_{stage}', stage) for stage in ['hlo', 'hlo_serialized']]
  )
  def testGetCompilerIrOnGpuPlatform(self, stage):
    # [FUNCTION DOC]
    # Parameterized test run for the "hlo" and "hlo_serialized" stages.
    # Scenario: defines a jit-compiled function that doubles a float scalar, calls it with the
    # constant one, then requests the compiler IR for the given stage targeting the GPU
    # platform by name. Asserts the returned IR is non-empty.
    @def_function.function(jit_compile=True)
    def test_func(x):
      # [FUNCTION DOC]
      # Jit-compiled traced function: returns its input multiplied by two.
      return 2.0 * x

    a = constant_op.constant(1.0)
    result = test_func.experimental_get_compiler_ir(a)(
        stage=stage, platform_name='GPU'
    )
    self.assertNotEmpty(result)

  @parameterized.named_parameters([
      (f'_{stage}', stage)
      for stage in [
          'optimized_hlo',
          'optimized_hlo_serialized',
          'optimized_hlo_proto_serialized',
          'optimized_hlo_dot',
      ]
  ])
  def testGetCompilerIrOnGpuPlatformOptimizedHlo(self, stage):
    # [FUNCTION DOC]
    # Parameterized test run for four optimized stages: optimized HLO, its serialized form, its
    # serialized proto form and its dot graph form.
    # Scenario: same jit-compiled doubling function and scalar input as the unoptimized test,
    # targeting the GPU platform from a TPU host. Asserts that requesting the IR raises a value
    # error whose message says the requested stage is not supported on this device.
    @def_function.function(jit_compile=True)
    def test_func(x):
      # [FUNCTION DOC]
      # Jit-compiled traced function: returns its input multiplied by two.
      return 2.0 * x

    a = constant_op.constant(1.0)
    with self.assertRaisesRegex(
        ValueError,
        'GetCompilerIr with requested stage is not supported on this device',
    ):
      _ = test_func.experimental_get_compiler_ir(a)(
          stage=stage, platform_name='GPU'
      )


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
