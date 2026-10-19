# [MODULE DOC]
# Tests for the kernel query functions in the kernels module: checks that at least one kernel is
# registered overall, and that the kernels for the test op KernelLabel can be found.
# Defines two test classes and a standard test entry point; no import-time side effects.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for querying registered kernels."""
from tensorflow.python.framework import kernels
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


class GetAllRegisteredKernelsTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests the function that lists all registered kernels. No shared fixtures.

  def testFindsAtLeastOneKernel(self):
    # [FUNCTION DOC]
    # Fetches the list of all registered kernels and asserts it contains at least one entry.
    kernel_list = kernels.get_all_registered_kernels()
    self.assertGreater(len(kernel_list.kernel), 0)


class GetRegisteredKernelsForOp(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests the function that lists registered kernels for one named op. No shared fixtures.

  def testFindsAtLeastOneKernel(self):
    # [FUNCTION DOC]
    # Queries the kernels for the op named KernelLabel (a test op), asserts at least one kernel is
    # returned, and asserts the first returned kernel's op name is KernelLabel.
    kernel_list = kernels.get_registered_kernels_for_op("KernelLabel")
    self.assertGreater(len(kernel_list.kernel), 0)
    self.assertEqual(kernel_list.kernel[0].op, "KernelLabel")


if __name__ == "__main__":
  googletest.main()
