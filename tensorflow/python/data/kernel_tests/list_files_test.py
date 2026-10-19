# [MODULE DOC]
# Test module for the dataset list-files source in the data kernel tests. It creates temporary
# directories with empty files and checks matching of single and multiple glob patterns, shuffled
# and unshuffled ordering, seeded repeatability, error on empty matches, and (for the subclasses)
# global shuffling and its checkpointing behavior. Defines three test classes and runs the test
# main entry point when executed as a script. No import-time side effects.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for `tf.data.Dataset.list_files()`."""

from os import path
import shutil
import tempfile
from typing import Callable, Optional

from absl.testing import parameterized

from tensorflow.python.data.experimental.ops import global_shuffle_op
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.ops import test_mode
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.platform import test
from tensorflow.python.util import compat


class ListFilesTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the dataset list-files source. Each test works in a fresh temporary directory
  # (held in an attribute named tmp_dir) that is created in setUp and removed in tearDown, and
  # populated with empty files by a helper. It is also the base class of the global-shuffle test
  # classes below, which therefore re-run all of these tests.

  def setUp(self):
    # [FUNCTION DOC]
    # Calls the parent setUp and creates a new temporary directory, stored as tmp_dir.
    super(ListFilesTest, self).setUp()
    self.tmp_dir = tempfile.mkdtemp()

  def tearDown(self):
    # [FUNCTION DOC]
    # Recursively removes the temporary directory (ignoring errors), then calls the parent
    # tearDown.
    shutil.rmtree(self.tmp_dir, ignore_errors=True)
    super(ListFilesTest, self).tearDown()

  def _touchTempFiles(self, filenames):
    # [FUNCTION DOC]
    # HELPER: creates the fixture files used by the list-files tests.
    # Takes a list of file names and creates an empty file with each name inside the temporary
    # directory (opening in append mode so existing files are untouched). Returns nothing.
    for filename in filenames:
      open(path.join(self.tmp_dir, filename), 'a').close()

  @combinations.generate(test_base.default_test_combinations())
  def testEmptyDirectory(self):
    # [FUNCTION DOC]
    # Scenario: the temporary directory is empty and the pattern matches everything in it.
    # Expects an invalid-argument error containing "No files matched" when the dataset is built
    # and its iterator is created; initialization is required so the check is not lost when a
    # one-shot iterator captures the dataset (see the linked bug note in the code).
    with self.assertRaisesWithPredicateMatch(errors.InvalidArgumentError,
                                             'No files matched'):
      dataset = dataset_ops.Dataset.list_files(path.join(self.tmp_dir, '*'))
      # We need requires_initialization=True so that getNext uses
      # make_initializable_iterator instead of make_one_shot_iterator.
      # make_one_shot_iterator has an issue where it fails to capture control
      # dependencies when capturing the dataset, so it loses the assertion that
      # list_files matches at least one file.
      # TODO(b/140837601): Make this work with make_one_shot_iterator.
      self.getNext(dataset, requires_initialization=True)

  @combinations.generate(test_base.default_test_combinations())
  def testSimpleDirectory(self):
    # [FUNCTION DOC]
    # Scenario: three files in the directory, pattern matching all. Expects the dataset to
    # produce the full paths of all three as byte strings, in any order.
    filenames = ['a', 'b', 'c']
    self._touchTempFiles(filenames)

    dataset = dataset_ops.Dataset.list_files(path.join(self.tmp_dir, '*'))
    self.assertDatasetProduces(
        dataset,
        expected_output=[
            compat.as_bytes(path.join(self.tmp_dir, filename))
            for filename in filenames
        ],
        assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testSimpleDirectoryNotShuffled(self):
    # [FUNCTION DOC]
    # Scenario: three files created in unsorted order, listed with shuffling disabled. Expects
    # the full paths as byte strings in sorted (lexicographic) order.
    filenames = ['b', 'c', 'a']
    self._touchTempFiles(filenames)

    dataset = dataset_ops.Dataset.list_files(
        path.join(self.tmp_dir, '*'), shuffle=False)
    self.assertDatasetProduces(
        dataset,
        expected_output=[
            compat.as_bytes(path.join(self.tmp_dir, filename))
            for filename in sorted(filenames)
        ])

  def testFixedSeedResultsInRepeatableOrder(self):
    # [FUNCTION DOC]
    # Scenario: three files listed with shuffling on and a fixed seed, iterated three separate
    # times with fresh iterators until the end of data. Expects every run to contain the same
    # set of paths as the expected list, and all three runs to have identical order.
    # Note: unlike most tests here it has no combinations decorator.
    filenames = ['a', 'b', 'c']
    self._touchTempFiles(filenames)

    def dataset_fn():
      # [FUNCTION DOC]
      # Builds a list-files dataset over all files in the temporary directory, shuffled, with
      # seed 37. Takes no arguments and returns the dataset.
      return dataset_ops.Dataset.list_files(
          path.join(self.tmp_dir, '*'), shuffle=True, seed=37)

    expected_filenames = [
        compat.as_bytes(path.join(self.tmp_dir, filename))
        for filename in filenames
    ]

    all_actual_filenames = []
    for _ in range(3):
      actual_filenames = []
      next_element = self.getNext(dataset_fn(), requires_initialization=True)
      try:
        while True:
          actual_filenames.append(self.evaluate(next_element()))
      except errors.OutOfRangeError:
        pass
      all_actual_filenames.append(actual_filenames)

    # Each run should produce the same set of filenames, which may be
    # different from the order of `expected_filenames`.
    self.assertCountEqual(expected_filenames, all_actual_filenames[0])
    # However, the different runs should produce filenames in the same order
    # as each other.
    self.assertEqual(all_actual_filenames[0], all_actual_filenames[1])
    self.assertEqual(all_actual_filenames[0], all_actual_filenames[2])

  @combinations.generate(test_base.default_test_combinations())
  def tesEmptyDirectoryInitializer(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name is misspelled (missing "t" in "test"), so the test runner never
    # discovers or runs this method; it is effectively dead code.
    # Intended scenario: empty directory with a match-all pattern. Expects iteration (with
    # initialization required) to fail with an invalid-argument error containing
    # "No files matched pattern".

    def dataset_fn():
      # [FUNCTION DOC]
      # Builds a list-files dataset over everything in the temporary directory and returns it.
      return dataset_ops.Dataset.list_files(path.join(self.tmp_dir, '*'))

    self.assertDatasetProduces(
        dataset_fn(),
        expected_error=(errors.InvalidArgumentError,
                        'No files matched pattern'),
        requires_initialization=True)

  @combinations.generate(test_base.default_test_combinations())
  def testSimpleDirectoryInitializer(self):
    # [FUNCTION DOC]
    # Scenario: three files, match-all pattern. Expects the dataset to produce the three full
    # paths as byte strings in any order. Same checks as the plain simple-directory test.
    filenames = ['a', 'b', 'c']
    self._touchTempFiles(filenames)

    dataset = dataset_ops.Dataset.list_files(path.join(self.tmp_dir, '*'))
    self.assertDatasetProduces(
        dataset,
        expected_output=[
            compat.as_bytes(path.join(self.tmp_dir, filename))
            for filename in filenames
        ],
        assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testFileSuffixes(self):
    # [FUNCTION DOC]
    # Scenario: files with text, python and compiled-python extensions, pattern matching only
    # names ending in the python extension. Expects only the two python files, in any order.
    filenames = ['a.txt', 'b.py', 'c.py', 'd.pyc']
    self._touchTempFiles(filenames)

    dataset = dataset_ops.Dataset.list_files(path.join(self.tmp_dir, '*.py'))
    self.assertDatasetProduces(
        dataset,
        expected_output=[
            compat.as_bytes(path.join(self.tmp_dir, filename))
            for filename in filenames[1:-1]
        ],
        assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testFileMiddles(self):
    # [FUNCTION DOC]
    # Scenario: pattern with a wildcard after the python extension, so it matches both the
    # python and compiled-python files but not the text file. Expects those two, in any order.
    filenames = ['a.txt', 'b.py', 'c.pyc']
    self._touchTempFiles(filenames)

    dataset = dataset_ops.Dataset.list_files(path.join(self.tmp_dir, '*.py*'))
    self.assertDatasetProduces(
        dataset,
        expected_output=[
            compat.as_bytes(path.join(self.tmp_dir, filename))
            for filename in filenames[1:]
        ],
        assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testNoShuffle(self):
    # [FUNCTION DOC]
    # Scenario: three files listed without shuffling and repeated twice. Reads all six elements
    # one by one, then expects end of data. Asserts the collected paths are the expected ones
    # (each twice) and that the second half equals the first half, i.e. the order is identical
    # on each repetition. Relies on the file list being captured at iterator initialization.
    filenames = ['a', 'b', 'c']
    self._touchTempFiles(filenames)

    # Repeat the list twice and ensure that the order is the same each time.
    # NOTE(mrry): This depends on an implementation detail of `list_files()`,
    # which is that the list of files is captured when the iterator is
    # initialized. Otherwise, or if e.g. the iterator were initialized more than
    # once, it's possible that the non-determinism of `tf.matching_files()`
    # would cause this test to fail. However, it serves as a useful confirmation
    # that the `shuffle=False` argument is working as intended.
    # TODO(b/73959787): Provide some ordering guarantees so that this test is
    # more meaningful.
    dataset = dataset_ops.Dataset.list_files(
        path.join(self.tmp_dir, '*'), shuffle=False).repeat(2)
    next_element = self.getNext(dataset)

    expected_filenames = []
    actual_filenames = []
    for filename in filenames * 2:
      expected_filenames.append(
          compat.as_bytes(path.join(self.tmp_dir, filename)))
      actual_filenames.append(compat.as_bytes(self.evaluate(next_element())))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element())
    self.assertCountEqual(expected_filenames, actual_filenames)
    self.assertEqual(actual_filenames[:len(filenames)],
                     actual_filenames[len(filenames):])

  @combinations.generate(test_base.default_test_combinations())
  def testMultiplePatternsAsList(self):
    # [FUNCTION DOC]
    # Scenario: two patterns (python and text extensions) built as a Python list of full paths.
    # Expects files matching either pattern (three of the four), in any order.
    filenames = ['a.txt', 'b.py', 'c.py', 'd.pyc']
    self._touchTempFiles(filenames)

    patterns = [path.join(self.tmp_dir, pat) for pat in ['*.py', '*.txt']]
    dataset = dataset_ops.Dataset.list_files(patterns)
    self.assertDatasetProduces(
        dataset,
        expected_output=[
            compat.as_bytes(path.join(self.tmp_dir, filename))
            for filename in filenames[:-1]
        ],
        assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testMultiplePatternsAsTensor(self):
    # [FUNCTION DOC]
    # Scenario: same two patterns passed inline as a list argument (converted to a string
    # tensor by the dataset). Expects the three matching files in any order.
    filenames = ['a.txt', 'b.py', 'c.py', 'd.pyc']
    self._touchTempFiles(filenames)

    dataset = dataset_ops.Dataset.list_files(
        [path.join(self.tmp_dir, pat) for pat in ['*.py', '*.txt']])
    self.assertDatasetProduces(
        dataset,
        expected_output=[
            compat.as_bytes(path.join(self.tmp_dir, filename))
            for filename in filenames[:-1]
        ],
        assert_items_equal=True)


class ListFilesGlobalShuffleTest(ListFilesTest, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for applying the global shuffle transformation to a list-files dataset. Inherits
  # all list-files tests (and their fixtures) from the base test class and adds two more.

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              repetitions=[1, 2],
              seed=[None, 42],
              reshuffle_each_iteration=[True, False])))
  def test(
      self,
      repetitions: int,
      seed: Optional[int],
      reshuffle_each_iteration: bool):
    # [FUNCTION DOC]
    # Parameterized over repetitions (1 or 2), seed (none or 42) and reshuffle-each-iteration
    # (true or false). Creates eleven files, lists them unshuffled, prefetches, optionally
    # repeats, then applies global shuffle with the given seed and reshuffle flag. Expects the
    # output to contain exactly the expected paths (times repetitions) but in a different order
    # than unshuffled, and its length to equal the dataset cardinality.
    filenames = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']
    self._touchTempFiles(filenames)
    dataset = dataset_ops.Dataset.list_files(path.join(self.tmp_dir, '*'),
                                             shuffle=False)
    dataset = dataset.prefetch(buffer_size=dataset_ops.AUTOTUNE)
    if repetitions > 1:
      dataset = dataset.repeat(repetitions)
    dataset = global_shuffle_op._global_shuffle(
        dataset, seed=seed, reshuffle_each_iteration=reshuffle_each_iteration)

    expected = [
        compat.as_bytes(path.join(self.tmp_dir, filename))
        for filename in filenames
    ] * repetitions
    dataset_output = self.getDatasetOutput(
        dataset, requires_initialization=True)
    self.assertCountEqual(dataset_output, expected)
    self.assertNotEqual(dataset_output, expected)
    self.assertLen(dataset_output, self.evaluate(dataset.cardinality()))

  @combinations.generate(test_base.default_test_combinations())
  def testShuffleNotSupported(self):
    # [FUNCTION DOC]
    # Scenario: a list-files dataset that is already shuffled (the default) cannot be globally
    # shuffled. Expects a failed-precondition error when global shuffle is applied and the
    # output is read.
    filenames = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']
    self._touchTempFiles(filenames)
    dataset = dataset_ops.Dataset.list_files(
        path.join(self.tmp_dir, '*'), shuffle=True)
    with self.assertRaises(errors.FailedPreconditionError):
      dataset = global_shuffle_op._global_shuffle(dataset)
      self.getDatasetOutput(dataset, requires_initialization=True)


class ListFilesGlobalShuffleCheckpointTest(
    ListFilesTest,
    checkpoint_test_base.CheckpointTestBase,
    parameterized.TestCase):
  # [CLASS DOC]
  # Test case verifying checkpointing of a globally shuffled list-files dataset. Inherits the
  # list-files tests and fixtures plus the checkpoint test base, which supplies the verification
  # function used by the parameterized test.

  def setUp(self):
    # [FUNCTION DOC]
    # Calls the parent setUp and then switches off the data test mode (side effect on global
    # state) so the default warm-start option, unsupported with global shuffling, is bypassed.
    super().setUp()
    # Bypasses the default value for `warm_start`, which is not supported for
    # global shuffling:
    # https://github.com/tensorflow/tensorflow/blob/29561af231863afb3b6b89e3aa8a6a550c2b7bb0/tensorflow/python/data/ops/options.py#L633
    test_mode.toggle_test_mode(False)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(
              repetitions=[1, 2],
              reshuffle_each_iteration=[True, False],
              symbolic_checkpoint=[True, False])))
  def test(
      self,
      verify_fn: Callable[..., None],
      repetitions: int,
      reshuffle_each_iteration: bool,
      symbolic_checkpoint: bool):
    # [FUNCTION DOC]
    # Parameterized over a checkpoint verification function, repetitions, reshuffle flag and
    # whether symbolic checkpoints are used. Creates eleven files, then calls the verification
    # function with a dataset builder and the expected number of outputs (files times
    # repetitions); items are compared order-insensitively only when reshuffling each iteration.
    filenames = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']
    self._touchTempFiles(filenames)

    def _build_dataset() -> dataset_ops.Dataset:
      # [FUNCTION DOC]
      # HELPER: dataset builder passed to the checkpoint verification function.
      # Lists the files unshuffled, prefetches, optionally repeats, applies global shuffle with
      # seed 42 and the reshuffle flag, sets the symbolic-checkpoint option, and returns the
      # dataset with those options. Takes no arguments.
      dataset = dataset_ops.Dataset.list_files(path.join(self.tmp_dir, '*'),
                                               shuffle=False)
      dataset = dataset.prefetch(buffer_size=dataset_ops.AUTOTUNE)
      if repetitions > 1:
        dataset = dataset.repeat(repetitions)
      dataset = global_shuffle_op._global_shuffle(
          dataset, seed=42, reshuffle_each_iteration=reshuffle_each_iteration)
      options = options_lib.Options()
      options.experimental_symbolic_checkpoint = symbolic_checkpoint
      return dataset.with_options(options)

    verify_fn(
        self,
        _build_dataset,
        num_outputs=len(filenames) * repetitions,
        assert_items_equal=reshuffle_each_iteration)


if __name__ == '__main__':
  test.main()
