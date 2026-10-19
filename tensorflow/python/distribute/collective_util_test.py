# [MODULE DOC]
# Unit tests for the collective options utilities in the sibling collective utility module:
# construction of the options object through the exported API and through the deprecated hints
# class. Not a helper module; no import-time side effects.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Test for utilities for collectives."""

from tensorflow.python.distribute import collective_util
from tensorflow.python.eager import test


class OptionsTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the collective options class and its deprecated hints alias. No shared fixtures.

  def testCreateOptionsViaExportedAPI(self):
    # [FUNCTION DOC]
    # Creates options through the exported constructor with a pack size of one byte and asserts
    # the result is an instance of the options class with that pack size. Also asserts that a
    # negative pack size raises a value error.
    options = collective_util._OptionsExported(bytes_per_pack=1)
    self.assertIsInstance(options, collective_util.Options)
    self.assertEqual(options.bytes_per_pack, 1)
    with self.assertRaises(ValueError):
      collective_util._OptionsExported(bytes_per_pack=-1)

  def testCreateOptionsViaHints(self):
    # [FUNCTION DOC]
    # Creates the deprecated hints object with a pack size of 50 and a timeout of one second,
    # capturing log output. Asserts a deprecation message was logged, the result is an options
    # instance, and the pack size and timeout were carried over.
    with self.assertLogs() as cm:
      options = collective_util.Hints(50, 1)
    self.assertTrue(any("is deprecated" in msg for msg in cm.output))
    self.assertIsInstance(options, collective_util.Options)
    self.assertEqual(options.bytes_per_pack, 50)
    self.assertEqual(options.timeout_seconds, 1)


if __name__ == "__main__":
  test.main()
