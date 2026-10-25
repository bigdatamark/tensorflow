# [MODULE DOC]
# Unit test verifying that the generated build information mapping agrees with the runtime
# queries of how TensorFlow was built (CUDA, ROCm, TensorRT) and that its keys keep a stable
# order. Depends on a generated build-info module that exists only in built packages.

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
"""Test for the generated build_info script."""

import platform

from tensorflow.python.platform import build_info
from tensorflow.python.platform import test


class BuildInfoTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the generated build information mapping; no shared fixtures.

  def testBuildInfo(self):
    # [FUNCTION DOC]
    # Asserts that the ROCm and CUDA flags in the build information mapping equal the values
    # returned by the test module's built-with-ROCm and built-with-CUDA queries. On non-Windows
    # systems it also imports the TensorRT utility extension and asserts the TensorRT flag equals
    # whether TensorRT is enabled. Fails with an assertion error on mismatch, or an import error
    # if the extension is missing.
    self.assertEqual(build_info.build_info['is_rocm_build'],
                     test.is_built_with_rocm())
    self.assertEqual(build_info.build_info['is_cuda_build'],
                     test.is_built_with_cuda())

    # TODO(b/173044576): make the test work for Windows.
    if platform.system() != 'Windows':
      # pylint: disable=g-import-not-at-top
      from tensorflow.compiler.tf2tensorrt._pywrap_py_utils import is_tensorrt_enabled
      self.assertEqual(build_info.build_info['is_tensorrt_build'],
                       is_tensorrt_enabled())

  def testDeterministicOrder(self):
    # [FUNCTION DOC]
    # Asserts that the keys of the build information mapping contain, as a subsequence in order,
    # the CUDA, ROCm and TensorRT build flags (other platform-specific keys may be interleaved).
    # The dict may contain other keys depending on the platform, but the ones
    # it always contains should be in order.
    self.assertContainsSubsequence(
        build_info.build_info.keys(),
        ('is_cuda_build', 'is_rocm_build', 'is_tensorrt_build'))


if __name__ == '__main__':
  test.main()
