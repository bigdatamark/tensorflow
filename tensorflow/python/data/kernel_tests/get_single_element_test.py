# [MODULE DOC]
# Tests for the dataset method that extracts the single element of a dataset. Covers plain and
# sparse elements, error cases for empty or multi-element datasets, use on nested datasets inside
# a transformation, side effects of mapped functions, and the optional operation name.
# No import-time side effects; the file runs the test main entry point when executed directly.
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
"""Tests for the experimental input pipeline ops."""
from absl.testing import parameterized

from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import def_function
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class GetSingleElementTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case covering getting the single element out of a dataset. Uses the shared dataset test
  # base class (evaluation and assertion helpers) and parameterized test generation across the
  # default graph/eager combinations.

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              skip=[0, 5, 10], take=[1], error=[None], error_msg=[None]) +
          combinations.combine(
              skip=[100],
              take=[1],
              error=[errors.InvalidArgumentError],
              error_msg=["Dataset was empty."]) + combinations.combine(
                  skip=[0],
                  take=[2],
                  error=[errors.InvalidArgumentError],
                  error_msg=["Dataset had more than one element."])))
  def testBasic(self, skip, take, error=None, error_msg=None):
    # [FUNCTION DOC]
    # Builds a range dataset of 100, skips `skip` items, maps each to a pair of its square and a
    # one-element sparse tensor, takes `take` items, then extracts the single element.
    # Parameters: skip (items skipped), take (items kept), error (expected error type or nothing),
    # error_msg (regular expression the error message must match, or nothing).
    # If no error is expected, asserts the dense value is skip squared and the sparse indices,
    # values and dense shape all describe the skipped count. Otherwise asserts the expected error
    # is raised (empty dataset when skipping past the end, or more than one element when taking 2).
    # Returns nothing.

    def make_sparse(x):
      # [FUNCTION DOC]
      # HELPER: builds the sparse element used by the basic test.
      # Takes a scalar tensor and returns a sparse tensor with one value at index zero-by-zero,
      # whose value is x and whose dense shape is one-dimensional of size one.
      x_1d = array_ops.reshape(x, [1])
      x_2d = array_ops.reshape(x, [1, 1])
      return sparse_tensor.SparseTensor(x_2d, x_1d, x_1d)

    dataset = dataset_ops.Dataset.range(100).skip(skip).map(
        lambda x: (x * x, make_sparse(x))).take(take)
    if error is None:
      dense_val, sparse_val = self.evaluate(dataset.get_single_element())
      self.assertEqual(skip * skip, dense_val)
      self.assertAllEqual([[skip]], sparse_val.indices)
      self.assertAllEqual([skip], sparse_val.values)
      self.assertAllEqual([skip], sparse_val.dense_shape)
    else:
      with self.assertRaisesRegex(error, error_msg):
        self.evaluate(dataset.get_single_element())

  @combinations.generate(test_base.default_test_combinations())
  def testWindow(self):
    # [FUNCTION DOC]
    # Verifies the single-element extraction works on the nested datasets produced by windowing.
    # Windows of size 2 over a range of 10 are each batched by 2 and reduced to their single
    # element inside a flat-map; expects the five batches [0,1] through [8,9].
    """Test that `get_single_element()` can consume a nested dataset."""

    def flat_map_func(ds):
      # [FUNCTION DOC]
      # HELPER: transformation function for the window test.
      # Receives a window dataset, batches it by 2, extracts the single batch and wraps it in a
      # one-element dataset which is returned.
      batched = ds.batch(2)
      element = batched.get_single_element()
      return dataset_ops.Dataset.from_tensors(element)

    dataset = dataset_ops.Dataset.range(10).window(2).flat_map(flat_map_func)
    self.assertDatasetProduces(dataset,
                               [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])

  @combinations.generate(test_base.default_test_combinations())
  def testSideEffect(self):
    # [FUNCTION DOC]
    # Verifies that a stateful mapped function runs when the single element is extracted inside a
    # compiled function whose output does not depend on it. A counter variable starts at zero; the
    # compiled function returns a constant string; asserts the string is returned and the counter
    # is exactly one afterwards.
    counter_var = variables.Variable(0)

    def increment_fn(x):
      # [FUNCTION DOC]
      # HELPER: stateful map function; adds one to the counter variable and returns its input.
      counter_var.assign_add(1)
      return x

    def dataset_fn():
      # [FUNCTION DOC]
      # HELPER: returns a one-element range dataset mapped through the incrementing function.
      return dataset_ops.Dataset.range(1).map(increment_fn)

    @def_function.function
    def fn():
      # [FUNCTION DOC]
      # HELPER: compiled function that extracts the single element, discards it and returns a
      # constant string, so the side effect must be kept by automatic control dependencies.
      _ = dataset_fn().get_single_element()
      return "hello"

    self.evaluate(counter_var.initializer)
    self.assertEqual(self.evaluate(fn()), b"hello")
    self.assertEqual(self.evaluate(counter_var), 1)

  @combinations.generate(test_base.default_test_combinations())
  def testAutomaticControlDependencies(self):
    # [FUNCTION DOC]
    # Verifies that two single-element extractions with stateful map functions execute in program
    # order inside a compiled function. The counter starts at 1, is incremented then doubled, so
    # the final value must be 4 (the reverse order would give 3). Also asserts the returned string.
    counter_var = variables.Variable(1)

    def increment_fn(x):
      # [FUNCTION DOC]
      # HELPER: stateful map function; adds one to the counter and returns its input.
      counter_var.assign(counter_var + 1)
      return x

    def multiply_fn(x):
      # [FUNCTION DOC]
      # HELPER: stateful map function; doubles the counter and returns its input.
      counter_var.assign(counter_var * 2)
      return x

    def dataset1_fn():
      # [FUNCTION DOC]
      # HELPER: one-element dataset mapped through the incrementing function.
      return dataset_ops.Dataset.range(1).map(increment_fn)

    def dataset2_fn():
      # [FUNCTION DOC]
      # HELPER: one-element dataset mapped through the doubling function.
      return dataset_ops.Dataset.range(1).map(multiply_fn)

    @def_function.function
    def fn():
      # [FUNCTION DOC]
      # HELPER: compiled function extracting the single element of each dataset in order,
      # discarding the results, and returning a constant string.
      _ = dataset1_fn().get_single_element()
      _ = dataset2_fn().get_single_element()
      return "hello"

    self.evaluate(counter_var.initializer)
    self.assertEqual(self.evaluate(fn()), b"hello")
    self.assertEqual(self.evaluate(counter_var), 4)

  @combinations.generate(test_base.default_test_combinations())
  def testName(self):
    # [FUNCTION DOC]
    # Verifies the single-element extraction accepts an operation name. Builds a dataset from the
    # scalar 42 and asserts that extracting with a name returns 42.
    dataset = dataset_ops.Dataset.from_tensors(42)
    self.assertEqual(
        self.evaluate(dataset.get_single_element(name="get_single_element")),
        42)


if __name__ == "__main__":
  test.main()
