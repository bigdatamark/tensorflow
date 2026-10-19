# [MODULE DOC]
# Tests for the enumerate transformation of the tf.data API. Contains a functional test that
# checks element types, shapes and values of an enumerated dataset with a tensor start offset,
# and a checkpoint test class verifying iterator save/restore. Uses the shared dataset test base
# classes; no import-time side effects. Running as a script starts the TensorFlow test runner.
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
"""Tests for `tf.data.Dataset.enumerate()`."""
from absl.testing import parameterized

from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import test


class EnumerateTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Functional tests for the enumerate dataset transformation. No shared fixtures.

  @combinations.generate(test_base.default_test_combinations())
  def testEnumerate(self):
    # [FUNCTION DOC]
    # Scenario: builds a dataset from three parallel component lists (strings, integers, floats)
    # sliced into two elements, then enumerates it with a start offset of 20 given as a 64-bit
    # integer constant tensor. Asserts the index component has 64-bit integer type and scalar
    # shape, that all three data components have scalar shape, and that the dataset produces
    # (20, (a, 1, 37.0)) and (21, (b, 2, 38.0)) with strings as bytes. Returns nothing.
    components = (["a", "b"], [1, 2], [37.0, 38])
    start = constant_op.constant(20, dtype=dtypes.int64)

    dataset = dataset_ops.Dataset.from_tensor_slices(components).enumerate(
        start)

    self.assertEqual(dtypes.int64,
                     dataset_ops.get_legacy_output_types(dataset)[0])
    dataset_output_shapes = dataset_ops.get_legacy_output_shapes(dataset)
    self.assertEqual((), dataset_output_shapes[0])
    self.assertEqual([tensor_shape.TensorShape([])] * 3,
                     [shape for shape in dataset_output_shapes[1]])

    self.assertDatasetProduces(dataset, [(20, (b"a", 1, 37.0)),
                                         (21, (b"b", 2, 38.0))])


class EnumerateCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                              parameterized.TestCase):
  # [CLASS DOC]
  # Checkpoint tests for the enumerate transformation, using the checkpoint test base class to
  # save, restore and verify iterator state.

  def _build_enumerate_dataset(self, start, stop, options=None):
    # [FUNCTION DOC]
    # HELPER: builds the dataset under test for the checkpoint test in this class.
    # Creates a range dataset from start up to (excluding) stop and enumerates it. If options
    # (optional, default none) is given, applies it to the dataset. Returns the dataset.
    dataset = dataset_ops.Dataset.range(start, stop).enumerate()
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
    # Checkpoint scenario: for both symbolic-checkpoint settings, sets that option and calls the
    # verification callback with a factory for the enumerated range dataset (start 2, stop 10).
    # The callback checks save/restore of iterator state across the 8 expected elements.
    # Returns nothing.
    start = 2
    stop = 10
    options = options_lib.Options()
    options.experimental_symbolic_checkpoint = symbolic_checkpoint
    verify_fn(
        self, lambda: self._build_enumerate_dataset(
            start=start, stop=stop, options=options), stop - start)


if __name__ == "__main__":
  test.main()
