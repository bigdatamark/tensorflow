# [MODULE DOC]
# Test module checking that loading a deliberately invalid custom op library fails with an
# invalid-argument error. One test class; script mode runs the TensorFlow test main entry point.
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

from tensorflow.python.framework import errors
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test


class InvalidOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering loading of a custom op library that is deliberately invalid.
  # No shared fixtures.

  def testBasic(self):
    # [FUNCTION DOC]
    # Builds the path of the invalid-op shared library in the test data directory and asserts that
    # loading it raises an invalid-argument error. Nothing is returned.
    library_filename = os.path.join(resource_loader.get_data_files_path(),
                                    'invalid_op.so')
    with self.assertRaises(errors.InvalidArgumentError):
      load_library.load_op_library(library_filename)


if __name__ == '__main__':
  test.main()
