# [MODULE DOC]
# Tests for the counter dataset factory of the tf.data API. Contains a functional test class that
# checks the element dtype, shape and produced values for different start/step settings, and a
# checkpoint test class that verifies save/restore of iterator state for a counter dataset zipped
# with a range dataset. Uses the shared test base classes and combinations; no import-time side
# effects. Running the file as a script starts the TensorFlow test runner.
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
"""Tests for `tf.data.Dataset.counter`."""
from absl.testing import parameterized

from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import test


class CounterTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Functional tests for the counter dataset factory. Has no shared fixtures beyond the dataset
  # test base class helpers (getNext, evaluate).

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(start=3, step=4, expected_output=[[3, 7, 11]]) +
          combinations.combine(start=0, step=-1, expected_output=[[0, -1, -2]]))
  )
  def testCounter(self, start, step, expected_output):
    # [FUNCTION DOC]
    # Parameterized scenario: builds a counter dataset from the given start and step (two cases:
    # start 3 step 4, and start 0 step -1). Asserts that the legacy output shape is scalar (empty
    # list), that the output type is 64-bit integer, and that the first elements read from the
    # iterator equal the expected list of values. Inputs are the start, the step and a list of
    # expected outputs. Returns nothing.
    dataset = dataset_ops.Dataset.counter(start, step)
    self.assertEqual(
        [], dataset_ops.get_legacy_output_shapes(dataset).as_list())
    self.assertEqual(dtypes.int64, dataset_ops.get_legacy_output_types(dataset))
    get_next = self.getNext(dataset)
    for expected in expected_output:
      self.assertEqual(expected, self.evaluate(get_next()))


class CounterCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                            parameterized.TestCase):
  # [CLASS DOC]
  # Checkpoint tests for the counter dataset. Builds a counter dataset zipped with a range dataset
  # and relies on the checkpoint test base class to save, restore and verify iterator state.

  def _build_counter_dataset(self, start, step, num_outputs, options=None):
    # [FUNCTION DOC]
    # HELPER: builds the dataset under test for the checkpoint test in this class.
    # Creates a counter dataset (start, step) and a range dataset of num_outputs elements and zips
    # them together, so the zipped dataset has num_outputs pairs. If options (optional, default
    # none) is given, applies it to the dataset. Returns the resulting dataset.
    counter_dataset = dataset_ops.Dataset.counter(start, step)
    range_dataset = dataset_ops.Dataset.range(num_outputs)
    dataset = dataset_ops.Dataset.zip((counter_dataset, range_dataset))
    if options:
      dataset = dataset.with_options(options)
    return dataset

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(symbolic_checkpoint=[False, True])))
  def test(self, verify_fn, symbolic_checkpoint):
    # [FUNCTION DOC]
    # Checkpoint scenario: for both symbolic-checkpoint settings, builds options with the
    # symbolic checkpoint flag set accordingly and calls the verification callback supplied by the
    # checkpoint combinations. The callback builds the zipped counter/range dataset (start 2,
    # step 10, 10 outputs) via a factory function and checks save/restore yields all 10 outputs.
    # Returns nothing.
    num_outputs = 10
    options = options_lib.Options()
    options.experimental_symbolic_checkpoint = symbolic_checkpoint
    verify_fn(
        self, lambda: self._build_counter_dataset(
            start=2, step=10, num_outputs=num_outputs, options=options),
        num_outputs)


if __name__ == "__main__":
  test.main()
