# [MODULE DOC]
# Test module for the native XLA ops binding (pywrap_xla_ops). Checks that the lists of kernel op
# names compilable by XLA for GPU and CPU are non-empty and contain selected common ops exactly
# once. Runs through the googletest main. No import-time side effects.

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
from tensorflow.python.platform import googletest  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.util import pywrap_xla_ops


class XlaOpsetUtilsTest(googletest.TestCase):
  # [CLASS DOC]
  # Tests for the XLA opset helper functions exposed by the native binding. No shared fixtures.

  def testGetGpuCompilableKernelNames(self):
    # [FUNCTION DOC]
    # Scenario: asks the binding for GPU kernel names XLA can compile. Asserts the list is not
    # empty and that the names for max, min and matrix multiplication each appear exactly once.
    """Tests retrieving compilable op names for GPU."""
    op_names = pywrap_xla_ops.get_gpu_kernel_names()
    self.assertGreater(op_names.__len__(), 0)
    self.assertEqual(op_names.count('Max'), 1)
    self.assertEqual(op_names.count('Min'), 1)
    self.assertEqual(op_names.count('MatMul'), 1)

  def testGetCpuCompilableKernelNames(self):
    # [FUNCTION DOC]
    # Scenario: asks the binding for CPU kernel names XLA can compile. Asserts the list is not
    # empty and that the names for max, min and matrix multiplication each appear exactly once.
    """Tests retrieving compilable op names for CPU."""
    op_names = pywrap_xla_ops.get_cpu_kernel_names()
    self.assertGreater(op_names.__len__(), 0)
    self.assertEqual(op_names.count('Max'), 1)
    self.assertEqual(op_names.count('Min'), 1)
    self.assertEqual(op_names.count('MatMul'), 1)


if __name__ == '__main__':
  googletest.main()
