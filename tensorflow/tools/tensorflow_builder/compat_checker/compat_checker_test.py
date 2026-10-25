# [MODULE DOC]
# Unit tests for the ConfigCompatChecker class of the compat_checker module. Defines three sample
# user configurations (all in range, all out of range, and missing or empty entries) that are
# checked against the test configuration file in the compat_checker directory.
# No import-time side effects besides building those constants.
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
#
# ==============================================================================
"""Tests for version compatibility checker for TensorFlow Builder."""

import os

import unittest
from tensorflow.tools.tensorflow_builder.compat_checker import compat_checker

PATH_TO_DIR = "tensorflow/tools/tensorflow_builder/compat_checker"

USER_CONFIG_IN_RANGE = {
    "apple": ["1.0"],
    "banana": ["3"],
    "kiwi": ["2.0"],
    "watermelon": ["2.0.0"],
    "orange": ["4.1"],
    "cherry": ["1.5"],
    "cranberry": ["1.0"],
    "raspberry": ["3.0"],
    "tangerine": ["2.0.0"],
    "jackfruit": ["1.0"],
    "grapefruit": ["2.0"],
    "apricot": ["wind", "flower"],
    "grape": ["7.1"],
    "blueberry": ["3.0"]
}
USER_CONFIG_NOT_IN_RANGE = {
    "apple": ["4.0"],
    "banana": ["5"],
    "kiwi": ["3.5"],
    "watermelon": ["5.0"],
    "orange": ["3.5"],
    "cherry": ["2.0"],
    "raspberry": ["-1"],
    "cranberry": ["4.5"],
    "tangerine": ["0"],
    "jackfruit": ["5.0"],
    "grapefruit": ["2.5"],
    "apricot": ["hello", "world"],
    "blueberry": ["11.0"],
    "grape": ["7.0"],
    "cantaloupe": ["11.0"]
}
USER_CONFIG_MISSING = {
    "avocado": ["3.0"],
    "apple": [],
    "banana": ""
}


class CompatCheckerTest(unittest.TestCase):
  # [CLASS DOC]
  # Test case covering ConfigCompatChecker. The shared fixture is the path of the test ini
  # configuration file, built in setUp. Each test builds a checker from a sample user
  # configuration and asserts on the result of the compatibility check.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture run before each test: calls the parent setUp and stores the path of the test
    # configuration ini file (inside the compat_checker directory) on the instance. Returns nothing.
    """Set up test."""
    super(CompatCheckerTest, self).setUp()
    self.test_file = os.path.join(PATH_TO_DIR, "test_config.ini")

  def testWithUserConfigInRange(self):
    # [FUNCTION DOC]
    # Scenario: every configuration value in the "in range" sample is supported by the ini file.
    # Builds a checker, runs the compatibility check and asserts it reports success, that no
    # error messages were recorded, and that the number of recorded successes equals the number
    # of configuration keys.
    """Test a set of configs that are supported.

    Testing with the following combination should always return `success`:
      [1] A set of configurations that are supported and/or compatible.
      [2] `.ini` config file with proper formatting.
    """
    # Initialize compatibility checker.
    self.compat_checker = compat_checker.ConfigCompatChecker(
        USER_CONFIG_IN_RANGE, self.test_file)
    # Compatibility check should succeed.
    self.assertTrue(self.compat_checker.check_compatibility())
    # Make sure no warning or error messages are recorded.
    self.assertFalse(len(self.compat_checker.error_msg))
    # Make sure total # of successes match total # of configs.
    cnt = len(list(USER_CONFIG_IN_RANGE.keys()))
    self.assertEqual(len(self.compat_checker.successes), cnt)

  def testWithUserConfigNotInRange(self):
    # [FUNCTION DOC]
    # Scenario: every value in the "not in range" sample is unsupported. Runs the check and
    # asserts it reports failure, that the failure list is non-empty, and that its length equals
    # the number of configuration keys (one failure per key).
    """Test a set of configs that are NOT supported.

    Testing with the following combination should always return `failure`:
      [1] A set of configurations that are NOT supported and/or compatible.
      [2] `.ini` config file with proper formatting.
    """
    self.compat_checker = compat_checker.ConfigCompatChecker(
        USER_CONFIG_NOT_IN_RANGE, self.test_file)
    # Compatibility check should fail.
    self.assertFalse(self.compat_checker.check_compatibility())
    # Check error and warning messages.
    err_msg_list = self.compat_checker.failures
    self.assertTrue(len(err_msg_list))
    # Make sure total # of failures match total # of configs.
    cnt = len(list(USER_CONFIG_NOT_IN_RANGE.keys()))
    self.assertEqual(len(err_msg_list), cnt)

  def testWithUserConfigMissing(self):
    # [FUNCTION DOC]
    # Scenario: the sample configuration has an unknown key and keys with empty values (empty
    # list, empty string), i.e. missing specification. Asserts the compatibility check fails.
    """Test a set of configs that are empty or missing specification."""
    self.compat_checker = compat_checker.ConfigCompatChecker(
        USER_CONFIG_MISSING, self.test_file)
    # With missing specification in config file, the check should
    # always fail.
    self.assertFalse(self.compat_checker.check_compatibility())


if __name__ == "__main__":
  unittest.main()
