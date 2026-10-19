# [MODULE DOC]
# Test module for cross-platform compiler IR retrieval from a function compiled with XLA.
# It checks that a jit-compiled function, traced on a host that is not a TPU machine (the
# "GPU" test target), can still report compiler IR for the TPU platform. Defines one test
# case class and, when run as a script, enables eager execution and runs the test main.
# No import-time side effects beyond imports.

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
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class ContextCrossPlatformGpuTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case covering the compiler IR query of XLA-compiled functions when the requested
  # target platform differs from the machine the test runs on (here the target is TPU).
  # Combines the framework test case with parameterized test support; holds no fixtures.

  @parameterized.named_parameters(
      [(f'_{stage}', stage) for stage in ['hlo', 'hlo_serialized']]
  )
  def testGetCompilerIrOnTpuPlatform(self, stage):
    # [FUNCTION DOC]
    # Parameterized test run once for the "hlo" stage and once for the "hlo_serialized" stage.
    # Scenario: defines a jit-compiled function that doubles its input, creates a one-million
    # element float tensor of ones, then asks the function for its compiler IR at the given
    # stage targeting the TPU platform by name. Asserts the returned IR is non-empty.
    @def_function.function(jit_compile=True)
    def test_func(x):
      # [FUNCTION DOC]
      # Jit-compiled traced function: returns its input multiplied by two.
      return 2 * x

    a = array_ops.ones((1000, 1000))  # 4 * 1000 * 1000 in bytes
    result = test_func.experimental_get_compiler_ir(a)(
        stage=stage, platform_name='TPU'
    )
    self.assertNotEmpty(result)


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
