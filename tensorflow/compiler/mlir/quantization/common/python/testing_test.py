# [MODULE DOC]
# Unit tests for the quantization common testing utilities module: the parameter combination
# generator and the directory size and size ratio helpers. Run as a main program through the
# TensorFlow test runner.
# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow.compiler.mlir.quantization.common.python import testing
from tensorflow.python.platform import test


class TestingTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the parameter combination generator. No shared fixtures.

  def test_parameter_combinations(self):
    # [FUNCTION DOC]
    # Scenario: one mapping with two candidate shapes (3 by 3, and 3 by unknown) and two bias
    # flags (true, false). Action: generate combinations. Expected: exactly four results, one for
    # each pairing of shape and bias flag.
    """Tests that parameter_combinations returns correct combinations."""
    test_parameters = [{
        'shapes': [
            [3, 3],
            [3, None],
        ],
        'has_bias': [True, False],
    }]
    combinations = testing.parameter_combinations(test_parameters)

    self.assertLen(combinations, 4)
    self.assertIn({'shapes': [3, 3], 'has_bias': True}, combinations)
    self.assertIn({'shapes': [3, 3], 'has_bias': False}, combinations)
    self.assertIn({'shapes': [3, None], 'has_bias': True}, combinations)
    self.assertIn({'shapes': [3, None], 'has_bias': False}, combinations)


class FileSizeTestCase(test.TestCase):
  # [CLASS DOC]
  # Test case covering the size helpers. Shared fixture: two temporary directories, one with a
  # single 4-byte file and one with three files of 4, 2 and 2 bytes (8 bytes total).

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture setup. Calls the parent setup, creates the two temporary directories with their
    # files and stores their full paths as attributes path_a (4 bytes) and path_b (8 bytes).
    super().setUp()

    self.path_a = self.create_tempdir('dir_a').full_path
    self.create_tempfile(file_path='dir_a/w.txt', content='abcd')

    self.path_b = self.create_tempdir('dir_b').full_path
    self.create_tempfile(file_path='dir_b/x.txt', content='1234')
    self.create_tempfile(file_path='dir_b/y.txt', content='56')
    self.create_tempfile(file_path='dir_b/z.txt', content='78')

  def test_get_dir_size(self):
    # [FUNCTION DOC]
    # Scenario: the two fixture directories. Expected: the size helper returns 4 for the first
    # directory and 8 for the second.
    self.assertEqual(testing.get_dir_size(self.path_a), 4)
    self.assertEqual(testing.get_dir_size(self.path_b), 8)

  def test_get_size_ratio(self):
    # [FUNCTION DOC]
    # Scenario: the two fixture directories. Expected: the ratio of the first to the second is
    # 0.5 and the reverse ratio is 2.0.
    self.assertEqual(testing.get_size_ratio(self.path_a, self.path_b), 0.5)
    self.assertEqual(testing.get_size_ratio(self.path_b, self.path_a), 2.0)


if __name__ == '__main__':
  test.main()
