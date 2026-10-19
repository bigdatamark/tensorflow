# [MODULE DOC]
# Tests for the private MatchingFilesDataset (glob-style file matching as a dataset): missing
# and empty directories, simple patterns, suffix/middle wildcards, nested directories, and
# iterator checkpointing. Test module only; no import-time side effects.
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
"""Tests for the private `MatchingFilesDataset`."""
import os
import shutil
import tempfile

from absl.testing import parameterized

from tensorflow.python.data.experimental.ops import matching_files
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.platform import test
from tensorflow.python.util import compat


class MatchingFilesDatasetTest(test_base.DatasetTestBase,
                               parameterized.TestCase):
  # [CLASS DOC]
  # Test case for MatchingFilesDataset. Each test works in a fresh temporary directory
  # (self.tmp_dir) created in setUp and removed in tearDown.

  def setUp(self):
    # [FUNCTION DOC]
    # Calls the parent setUp and creates a temporary directory stored as tmp_dir.
    super(MatchingFilesDatasetTest, self).setUp()
    self.tmp_dir = tempfile.mkdtemp()

  def tearDown(self):
    # [FUNCTION DOC]
    # Recursively removes the temporary directory (errors ignored), then calls the parent
    # tearDown.
    shutil.rmtree(self.tmp_dir, ignore_errors=True)
    super(MatchingFilesDatasetTest, self).tearDown()

  def _touchTempFiles(self, filenames):
    # [FUNCTION DOC]
    # HELPER: creates empty files with the given names inside the temporary directory.
    for filename in filenames:
      open(os.path.join(self.tmp_dir, filename), 'a').close()

  @combinations.generate(test_base.default_test_combinations())
  def testNonExistingDirectory(self):
    # [FUNCTION DOC]
    # Points the dataset at a wildcard pattern under a directory that does not exist and
    # asserts iteration fails with a not-found error.
    """Test the MatchingFiles dataset with a non-existing directory."""

    self.tmp_dir = os.path.join(self.tmp_dir, 'nonexistingdir')
    dataset = matching_files.MatchingFilesDataset(
        os.path.join(self.tmp_dir, '*'))
    self.assertDatasetProduces(
        dataset, expected_error=(errors.NotFoundError, ''))

  @combinations.generate(test_base.default_test_combinations())
  def testEmptyDirectory(self):
    # [FUNCTION DOC]
    # Matches a wildcard in an empty existing directory and asserts a not-found error.
    """Test the MatchingFiles dataset with an empty directory."""

    dataset = matching_files.MatchingFilesDataset(
        os.path.join(self.tmp_dir, '*'))
    self.assertDatasetProduces(
        dataset, expected_error=(errors.NotFoundError, ''))

  @combinations.generate(test_base.default_test_combinations())
  def testSimpleDirectory(self):
    # [FUNCTION DOC]
    # Creates files a, b, c and matches everything; asserts the dataset yields their full
    # paths as byte strings (order ignored).
    """Test the MatchingFiles dataset with a simple directory."""

    filenames = ['a', 'b', 'c']
    self._touchTempFiles(filenames)

    dataset = matching_files.MatchingFilesDataset(
        os.path.join(self.tmp_dir, '*'))
    self.assertDatasetProduces(
        dataset,
        expected_output=[
            compat.as_bytes(os.path.join(self.tmp_dir, filename))
            for filename in filenames
        ],
        assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testFileSuffixes(self):
    # [FUNCTION DOC]
    # Creates files with .txt, .py and .pyc suffixes, matches the .py pattern, and asserts
    # only the two .py files are produced (order ignored).
    """Test the MatchingFiles dataset using the suffixes of filename."""

    filenames = ['a.txt', 'b.py', 'c.py', 'd.pyc']
    self._touchTempFiles(filenames)

    dataset = matching_files.MatchingFilesDataset(
        os.path.join(self.tmp_dir, '*.py'))
    self.assertDatasetProduces(
        dataset,
        expected_output=[
            compat.as_bytes(os.path.join(self.tmp_dir, filename))
            for filename in filenames[1:-1]
        ],
        assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testFileMiddles(self):
    # [FUNCTION DOC]
    # Creates four files and matches a pattern starting with "b" containing ".py"; asserts the
    # two matching files (bb.py and bbc.pyc) are produced (order ignored).
    """Test the MatchingFiles dataset using the middles of filename."""

    filenames = ['aa.txt', 'bb.py', 'bbc.pyc', 'cc.pyc']
    self._touchTempFiles(filenames)

    dataset = matching_files.MatchingFilesDataset(
        os.path.join(self.tmp_dir, 'b*.py*'))
    self.assertDatasetProduces(
        dataset,
        expected_output=[
            compat.as_bytes(os.path.join(self.tmp_dir, filename))
            for filename in filenames[1:3]
        ],
        assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testNestedDirectories(self):
    # [FUNCTION DOC]
    # Builds a tree of 8 top-level directories each nested up to depth 4 with various files,
    # matches .txt and .log files at the deepest level using two multi-level wildcard
    # patterns, iterates the dataset to exhaustion, and asserts the produced paths equal the
    # created .txt/.log files regardless of order.
    """Test the MatchingFiles dataset with nested directories."""

    filenames = []
    width = 8
    depth = 4
    for i in range(width):
      for j in range(depth):
        new_base = os.path.join(self.tmp_dir, str(i),
                                *[str(dir_name) for dir_name in range(j)])
        os.makedirs(new_base)
        child_files = ['a.py', 'b.pyc'] if j < depth - 1 else ['c.txt', 'd.log']
        for f in child_files:
          filename = os.path.join(new_base, f)
          filenames.append(filename)
          open(filename, 'w').close()

    patterns = [
        os.path.join(self.tmp_dir, os.path.join(*['**' for _ in range(depth)]),
                     suffix) for suffix in ['*.txt', '*.log']
    ]

    dataset = matching_files.MatchingFilesDataset(patterns)
    next_element = self.getNext(dataset)
    expected_filenames = [
        compat.as_bytes(filename)
        for filename in filenames
        if filename.endswith('.txt') or filename.endswith('.log')
    ]
    actual_filenames = []
    while True:
      try:
        actual_filenames.append(compat.as_bytes(self.evaluate(next_element())))
      except errors.OutOfRangeError:
        break

    self.assertCountEqual(expected_filenames, actual_filenames)


class MatchingFilesDatasetCheckpointTest(
    checkpoint_test_base.CheckpointTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Checkpoint (save/restore of iterator state) test for MatchingFilesDataset, using the
  # shared checkpoint test base and its verification function.

  def _build_iterator_graph(self, test_patterns):
    # [FUNCTION DOC]
    # HELPER: returns a MatchingFilesDataset for the given pattern(s); used as the dataset
    # factory for the checkpoint verification.
    return matching_files.MatchingFilesDataset(test_patterns)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         checkpoint_test_base.default_test_combinations()))
  def test(self, verify_fn):
    # [FUNCTION DOC]
    # Takes a verification callback from the checkpoint combinations. Creates a temporary tree
    # of 16 top-level directories nested to depth 8, builds two multi-level wildcard patterns
    # for .txt and .log files, and runs the callback expecting 32 outputs (width times the
    # number of patterns). Finally removes the temporary tree.
    tmp_dir = tempfile.mkdtemp()
    width = 16
    depth = 8
    for i in range(width):
      for j in range(depth):
        new_base = os.path.join(tmp_dir, str(i),
                                *[str(dir_name) for dir_name in range(j)])
        if not os.path.exists(new_base):
          os.makedirs(new_base)
        child_files = ['a.py', 'b.pyc'] if j < depth - 1 else ['c.txt', 'd.log']
        for f in child_files:
          filename = os.path.join(new_base, f)
          open(filename, 'w').close()

    patterns = [
        os.path.join(tmp_dir, os.path.join(*['**'
                                             for _ in range(depth)]), suffix)
        for suffix in ['*.txt', '*.log']
    ]

    num_outputs = width * len(patterns)
    verify_fn(self, lambda: self._build_iterator_graph(patterns), num_outputs)

    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
  test.main()
