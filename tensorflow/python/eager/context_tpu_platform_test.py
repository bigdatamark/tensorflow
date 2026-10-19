# [MODULE DOC]
# Test module for compiler IR retrieval of an XLA-compiled function that captures a variable
# placed on a TPU device. Requires a TPU platform to run. When run as a script it enables
# eager execution and runs the test main. No import-time side effects.

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
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class ContextTpuPlatformTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case covering the compiler IR query of XLA-compiled functions on a TPU platform,
  # including functions that read a variable living on a TPU device. Combines the framework
  # test case with parameterized test support; holds no fixtures.

  @parameterized.named_parameters(
      [(f'_{stage}', stage) for stage in ['hlo', 'hlo_serialized']]
  )
  def testGetCompilerIrWithVariable(self, stage):
    # [FUNCTION DOC]
    # Parameterized test run for the "hlo" and "hlo_serialized" stages.
    # Scenario: creates a float variable with initial value one under the first TPU device
    # scope, defines a jit-compiled function that multiplies its input by that variable, and
    # requests the compiler IR for the given stage targeting the TPU platform, passing a float
    # scalar constant. Asserts the returned IR is non-empty.
    with ops.device('TPU:0'):
      v = variables.Variable(1.0)

    @def_function.function(jit_compile=True)
    def test_func(x):
      # [FUNCTION DOC]
      # Jit-compiled traced function: returns its input multiplied by the captured TPU variable.
      return x * v

    a = constant_op.constant(1.0)
    result = test_func.experimental_get_compiler_ir(a)(
        stage=stage, platform_name='TPU'
    )
    self.assertNotEmpty(result)


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
