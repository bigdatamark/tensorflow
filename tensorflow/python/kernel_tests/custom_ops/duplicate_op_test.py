# [MODULE DOC]
# Test module for loading a custom op library that registers a duplicate of an existing op
# (the add op) and checking the built-in op still works. One test class; script mode runs test main.
# The shared library is expected as a data file next to the test; no import-time side effects.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for custom user ops."""
import os

from tensorflow.python.framework import load_library
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test


class DuplicateOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering a custom op library that duplicates an existing op registration.
  # No shared fixtures beyond the cached session used inside the test.

  @test_util.run_deprecated_v1
  def testBasic(self):
    # [FUNCTION DOC]
    # Loads the duplicate-op shared library from the test data directory (return value ignored),
    # then in a cached session evaluates adding one and forty-one and asserts the result is
    # forty-two, showing the built-in add op still works. Legacy graph mode only.
    library_filename = os.path.join(resource_loader.get_data_files_path(),
                                    'duplicate_op.so')
    load_library.load_op_library(library_filename)

    with self.cached_session():
      self.assertEqual(math_ops.add(1, 41).eval(), 42)


if __name__ == '__main__':
  test.main()
