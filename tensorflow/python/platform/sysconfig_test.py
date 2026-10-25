# [MODULE DOC]
# Tests for the system configuration module: the build information mapping is a dictionary and
# agrees with the ROCm/CUDA queries, and the compile flags include an include directory, the C++
# ABI macro, the Eigen alignment macro and a language standard option. Runs googletest main when
# executed as a script.

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

import re

from tensorflow.python.platform import googletest
from tensorflow.python.platform import sysconfig as sysconfig_lib
from tensorflow.python.platform import test


class SysconfigTest(googletest.TestCase):
  # [CLASS DOC]
  # Test case for the system configuration functions; no shared fixtures.

  def test_get_build_info_works(self):
    # [FUNCTION DOC]
    # Fetches the build information and asserts it is a dictionary.
    build_info = sysconfig_lib.get_build_info()
    self.assertIsInstance(build_info, dict)

  def test_rocm_cuda_info_matches(self):
    # [FUNCTION DOC]
    # Asserts the ROCm and CUDA build flags in the build information equal the values of the test
    # module's built-with-ROCm and built-with-CUDA queries.
    build_info = sysconfig_lib.get_build_info()
    self.assertEqual(build_info["is_rocm_build"], test.is_built_with_rocm())
    self.assertEqual(build_info["is_cuda_build"], test.is_built_with_cuda())

  def test_compile_flags(self):
    # [FUNCTION DOC]
    # Gets the compile flags and asserts that some flag matches each of four patterns: an include
    # directory ending in include, the C++ ABI macro, the Eigen alignment macro, and a language
    # standard option. Matching is by regular expression anchored at the start of each flag.
    # Must contain an include directory, and define _GLIBCXX_USE_CXX11_ABI,
    # EIGEN_MAX_ALIGN_BYTES
    compile_flags = sysconfig_lib.get_compile_flags()

    def list_contains(items, regex_str):
      # [FUNCTION DOC]
      # HELPER: assertion helper for the compile flag test. Compiles the pattern text and returns
      # true if any item in the list matches it at its beginning, otherwise false.
      regex = re.compile(regex_str)
      return any(regex.match(item) for item in items)
    self.assertTrue(list_contains(compile_flags, ".*/include"))
    self.assertTrue(list_contains(compile_flags, ".*_GLIBCXX_USE_CXX11_ABI.*"))
    self.assertTrue(list_contains(compile_flags, ".*EIGEN_MAX_ALIGN_BYTES.*"))
    self.assertTrue(list_contains(compile_flags, ".*std.*"))


if __name__ == "__main__":
  googletest.main()
