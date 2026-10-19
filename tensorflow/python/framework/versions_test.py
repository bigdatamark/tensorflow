# [MODULE DOC]
# Test module for the version constants exposed by the framework versions module. Contains one
# test case class checking that version strings, graph definition versions and git/compiler
# versions have the expected types and plausible values. Runs via the standard test main entry.
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
"""Tests for exposed tensorflow versions."""

from tensorflow.python.framework import versions
from tensorflow.python.platform import test


class VersionTest(test.TestCase):
  # [CLASS DOC]
  # Test case verifying the public version constants of the versions module. No shared fixtures.

  def testVersion(self):
    # [FUNCTION DOC]
    # Checks that the version string and its alias are both strings and match the release pattern
    # (major.minor.patch with optional suffixes, or the head development form).
    self.assertEqual(type(versions.__version__), str)
    self.assertEqual(type(versions.VERSION), str)
    # This pattern will need to grow as we include alpha, builds, etc.
    self.assertRegex(
        versions.__version__, r'^\d+\.\d+\.(\d+(\-\w+)?(\+\w+)?|head)$'
    )
    self.assertRegex(
        versions.VERSION, r'^\d+\.\d+\.(\d+(\-\w+)?(\+\w+)?|head)$'
    )

  def testGraphDefVersion(self):
    # [FUNCTION DOC]
    # Checks that the graph definition version and its minimum consumer and producer versions are
    # integers, that both minimums are non-negative, and that the minimum producer does not
    # exceed the current version.
    version = versions.GRAPH_DEF_VERSION
    min_consumer = versions.GRAPH_DEF_VERSION_MIN_CONSUMER
    min_producer = versions.GRAPH_DEF_VERSION_MIN_PRODUCER
    for v in version, min_consumer, min_producer:
      self.assertEqual(type(v), int)
    self.assertLessEqual(0, min_consumer)
    self.assertLessEqual(0, min_producer)
    self.assertLessEqual(min_producer, version)

  def testGitAndCompilerVersion(self):
    # [FUNCTION DOC]
    # Checks that the git version and compiler version constants, in both dunder and upper-case
    # forms, are strings. Content is not inspected.
    self.assertEqual(type(versions.__git_version__), str)
    self.assertEqual(type(versions.__compiler_version__), str)
    self.assertEqual(type(versions.GIT_VERSION), str)
    self.assertEqual(type(versions.COMPILER_VERSION), str)


if __name__ == '__main__':
  test.main()
