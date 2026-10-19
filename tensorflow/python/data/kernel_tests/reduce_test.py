# [MODULE DOC]
# Tests for the dataset reduce action (folding a dataset into a single state value with a user
# function). Covers sums, tuple/nested/sparse states, placeholders in graph mode, side effects and
# automatic control dependencies inside traced functions, GPU placement, cancellation, invalid
# reduce functions, dataset options and the name argument. Test classes only.
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
"""Tests for `tf.data.Dataset.reduce()`."""
import time

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import def_function
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class ReduceTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the reduce action of datasets under the default graph/eager combinations (plus a few
  # graph-only cases).

  @combinations.generate(test_base.default_test_combinations())
  def testSum(self):
    # [FUNCTION DOC]
    # For lengths zero to nine, reduces a range starting at one with an addition function and an
    # initial state of zero. Asserts the result equals the arithmetic-series sum.
    for i in range(10):
      ds = dataset_ops.Dataset.range(1, i + 1)
      result = ds.reduce(np.int64(0), lambda x, y: x + y)
      self.assertEqual(((i + 1) * i) // 2, self.evaluate(result))

  @combinations.generate(test_base.default_test_combinations())
  def testSumTuple(self):
    # [FUNCTION DOC]
    # Zips a range with itself so elements are pairs and reduces with a function adding both
    # components to the state. Asserts the result is twice the arithmetic-series sum.

    def reduce_fn(state, value):
      # [FUNCTION DOC]
      # HELPER: reduce function for the tuple-sum test. Unpacks the pair value and returns the
      # state plus both components.
      v1, v2 = value
      return state + v1 + v2

    for i in range(10):
      ds = dataset_ops.Dataset.range(1, i + 1)
      ds = dataset_ops.Dataset.zip((ds, ds))
      result = ds.reduce(constant_op.constant(0, dtype=dtypes.int64), reduce_fn)
      self.assertEqual(((i + 1) * i), self.evaluate(result))

  @combinations.generate(test_base.default_test_combinations())
  def testSumAndCount(self):
    # [FUNCTION DOC]
    # Reduces ranges with a pair state of running sum and count. Asserts the final sum equals the
    # arithmetic-series sum and the count equals the number of elements.

    def reduce_fn(state, value):
      # [FUNCTION DOC]
      # HELPER: reduce function for the sum-and-count test. Takes a (sum, count) state and a
      # value; returns the sum plus the value and the count plus one.
      s, c = state
      return s + value, c + 1

    for i in range(10):
      ds = dataset_ops.Dataset.range(1, i + 1)
      result = ds.reduce((constant_op.constant(0, dtype=dtypes.int64),
                          constant_op.constant(0, dtype=dtypes.int64)),
                         reduce_fn)
      s, c = self.evaluate(result)
      self.assertEqual(((i + 1) * i) // 2, s)
      self.assertEqual(i, c)

  @combinations.generate(combinations.combine(tf_api_version=1, mode="graph"))
  def testSquareUsingPlaceholder(self):
    # [FUNCTION DOC]
    # Graph-mode only. The reduce function adds a fed placeholder value to the state for each
    # element; for each i, feeding i into a range of length i must yield i squared.
    delta = array_ops.placeholder(dtype=dtypes.int64)

    def reduce_fn(state, _):
      # [FUNCTION DOC]
      # HELPER: reduce function ignoring the element; returns the state plus the placeholder.
      return state + delta

    for i in range(10):
      ds = dataset_ops.Dataset.range(1, i + 1)
      result = ds.reduce(np.int64(0), reduce_fn)
      with self.cached_session() as sess:
        square = sess.run(result, feed_dict={delta: i})
        self.assertEqual(i * i, square)

  @combinations.generate(test_base.default_test_combinations())
  def testSparse(self):
    # [FUNCTION DOC]
    # Reduces a single-element dataset of sparse tensors with a function returning the element,
    # starting from a zero sparse state. Asserts the result equals the sparse value for element i.

    def reduce_fn(_, value):
      # [FUNCTION DOC]
      # HELPER: reduce function that ignores the state and returns the new value.
      return value

    def make_sparse_fn(i):
      # [FUNCTION DOC]
      # HELPER: builds a one-by-one sparse tensor value with a single entry at the origin whose
      # value is i; returns it.
      return sparse_tensor.SparseTensorValue(
          indices=np.array([[0, 0]]),
          values=(i * np.array([1])),
          dense_shape=np.array([1, 1]))

    for i in range(10):
      ds = dataset_ops.Dataset.from_tensors(make_sparse_fn(i+1))
      result = ds.reduce(make_sparse_fn(0), reduce_fn)
      self.assertValuesEqual(make_sparse_fn(i + 1), self.evaluate(result))

  @combinations.generate(test_base.default_test_combinations())
  def testNested(self):
    # [FUNCTION DOC]
    # Reduces a dataset of mappings with a dense and a sparse component, accumulating the dense
    # part and keeping the latest sparse part. Asserts the dense sum and last sparse value.

    def reduce_fn(state, value):
      # [FUNCTION DOC]
      # HELPER: adds the dense component of value into the state, replaces the state's sparse
      # component with the value's, and returns the (mutated) state mapping.
      state["dense"] += value["dense"]
      state["sparse"] = value["sparse"]
      return state

    def make_sparse_fn(i):
      # [FUNCTION DOC]
      # HELPER: builds a one-by-one sparse tensor value with a single entry whose value is i.
      return sparse_tensor.SparseTensorValue(
          indices=np.array([[0, 0]]),
          values=(i * np.array([1])),
          dense_shape=np.array([1, 1]))

    def map_fn(i):
      # [FUNCTION DOC]
      # HELPER: maps an integer to a mapping with the dense component (the integer as a 64-bit
      # integer) and a sparse component built from it.
      return {"dense": math_ops.cast(i, dtype=dtypes.int64),
              "sparse": make_sparse_fn(math_ops.cast(i, dtype=dtypes.int64))}

    for i in range(10):
      ds = dataset_ops.Dataset.range(1, i + 1).map(map_fn)
      result = ds.reduce(map_fn(0), reduce_fn)
      result = self.evaluate(result)
      self.assertEqual(((i + 1) * i) // 2, result["dense"])
      self.assertValuesEqual(make_sparse_fn(i), result["sparse"])

  @combinations.generate(test_base.default_test_combinations())
  def testDatasetSideEffect(self):
    # [FUNCTION DOC]
    # The side effect (incrementing a counter variable) lives in the dataset's map function. Inside
    # a traced function a reduce over that dataset must still execute it: the counter ends at ten.
    counter_var = variables.Variable(0)

    def increment_fn(x):
      # [FUNCTION DOC]
      # HELPER: map function that increments the counter variable and returns its input.
      counter_var.assign_add(1)
      return x

    def dataset_fn():
      # [FUNCTION DOC]
      # HELPER: returns a range of ten mapped through the incrementing function.
      return dataset_ops.Dataset.range(10).map(increment_fn)

    def reduce_fn(state, value):
      # [FUNCTION DOC]
      # HELPER: plain addition reduce function.
      return state + value

    @def_function.function
    def fn():
      # [FUNCTION DOC]
      # HELPER: traced function that reduces the dataset, discards the result and returns a
      # constant string, so only control dependencies keep the reduce alive.
      _ = dataset_fn().reduce(np.int64(0), reduce_fn)
      return "hello"

    self.evaluate(counter_var.initializer)
    self.assertEqual(self.evaluate(fn()), b"hello")
    self.assertEqual(self.evaluate(counter_var), 10)

  @combinations.generate(test_base.default_test_combinations())
  def testSideEffect(self):
    # [FUNCTION DOC]
    # The side effect (incrementing a counter variable) lives in the reduce function. Inside a
    # traced function that discards the reduce result it must still run: the counter ends at ten.
    counter_var = variables.Variable(0)

    def dataset_fn():
      # [FUNCTION DOC]
      # HELPER: returns a range of ten.
      return dataset_ops.Dataset.range(10)

    def reduce_fn(state, value):
      # [FUNCTION DOC]
      # HELPER: increments the counter variable and returns the state plus the value.
      counter_var.assign_add(1)
      return state + value

    @def_function.function
    def fn():
      # [FUNCTION DOC]
      # HELPER: traced function that runs the reduce, discards the result and returns a constant
      # string.
      _ = dataset_fn().reduce(np.int64(0), reduce_fn)
      return "hello"

    self.evaluate(counter_var.initializer)
    self.assertEqual(self.evaluate(fn()), b"hello")
    self.assertEqual(self.evaluate(counter_var), 10)

  @combinations.generate(test_base.default_test_combinations())
  def testAutomaticControlDependencies(self):
    # [FUNCTION DOC]
    # Two reduces over a one-element dataset in a traced function, the first adding one to a
    # counter and the second doubling it. Asserts automatic control dependencies order them as
    # written: starting from one the counter ends at four.
    counter_var = variables.Variable(1)

    def dataset_fn():
      # [FUNCTION DOC]
      # HELPER: returns a range of one element.
      return dataset_ops.Dataset.range(1)

    def reduce1_fn(state, value):
      # [FUNCTION DOC]
      # HELPER: increments the counter by one and returns the state plus the value.
      counter_var.assign(counter_var + 1)
      return state + value

    def reduce2_fn(state, value):
      # [FUNCTION DOC]
      # HELPER: doubles the counter and returns the state plus the value.
      counter_var.assign(counter_var * 2)
      return state + value

    @def_function.function
    def fn():
      # [FUNCTION DOC]
      # HELPER: traced function running both reduces in order and returning a constant string.
      _ = dataset_fn().reduce(np.int64(0), reduce1_fn)
      _ = dataset_fn().reduce(np.int64(0), reduce2_fn)
      return "hello"

    self.evaluate(counter_var.initializer)
    self.assertEqual(self.evaluate(fn()), b"hello")
    self.assertEqual(self.evaluate(counter_var), 4)

  @combinations.generate(test_base.default_test_combinations())
  def testNestedAutomaticControlDependencies(self):
    # [FUNCTION DOC]
    # A traced function iterates a dataset whose map function increments a counter variable and
    # returns the counter. Asserts the counter reads ten, i.e. the iteration side effects ran
    # before the return value was read.
    counter_var = variables.Variable(0)

    def map_fn(x):
      # [FUNCTION DOC]
      # HELPER: increments the counter and returns its input.
      counter_var.assign_add(1)
      return x

    def dataset_fn():
      # [FUNCTION DOC]
      # HELPER: returns a range of ten mapped through the incrementing function.
      return dataset_ops.Dataset.range(10).map(map_fn)

    @def_function.function
    def fn():
      # [FUNCTION DOC]
      # HELPER: traced function that loops over the dataset doing nothing and returns the
      # counter variable.
      for _ in dataset_fn():
        pass
      return counter_var

    self.evaluate(counter_var.initializer)
    self.assertEqual(self.evaluate(fn()), 10)

  @combinations.generate(test_base.default_test_combinations())
  def testStateOnGPU(self):
    # [FUNCTION DOC]
    # Skipped when no GPU is available. Reduces ranges with an addition function placed on the
    # first GPU and a constant initial state; asserts the arithmetic-series sums.
    if not test_util.is_gpu_available():
      self.skipTest("No GPUs available.")

    state = constant_op.constant(0, dtype=dtypes.int64)

    def reduce_fn(state, value):
      # [FUNCTION DOC]
      # HELPER: returns the state plus the value, computed under a GPU device scope.
      with ops.device("/gpu:0"):
        return state + value

    for i in range(10):
      ds = dataset_ops.Dataset.range(1, i + 1)
      result = ds.reduce(state, reduce_fn)
      self.assertEqual(((i + 1) * i) // 2, self.evaluate(result))

  @combinations.generate(combinations.combine(tf_api_version=1, mode="graph"))
  def testCancellation(self):
    # [FUNCTION DOC]
    # Graph-mode only. Reduces an infinite dataset in a background thread, waits briefly, closes
    # the session and joins the thread. Asserts the reduce operation was cancelled rather than
    # completing.
    ds = dataset_ops.Dataset.from_tensors(1).repeat()
    result = ds.reduce(0, lambda x, y: x + y)
    with self.cached_session() as sess:
      # The `result` op is guaranteed to not complete before cancelled because
      # the dataset that is being reduced is infinite.
      thread = self.checkedThread(self.assert_op_cancelled, args=(result,))
      thread.start()
      time.sleep(0.2)
      sess.close()
      thread.join()

  @combinations.generate(test_base.default_test_combinations())
  def testInvalidFunction(self):
    # [FUNCTION DOC]
    # Reduces a range with a function returning an empty tuple instead of a state matching the
    # initial one. Asserts an invalid-argument error is raised on evaluation.
    ds = dataset_ops.Dataset.range(5)
    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(ds.reduce(0, lambda _, __: ()))

  @combinations.generate(test_base.default_test_combinations())
  def testOptions(self):
    # [FUNCTION DOC]
    # Asserts, via a testing transformation, that map followed by batch is fused into a combined
    # map-and-batch op before reduce is applied, then evaluates a reduce over it.
    dataset = dataset_ops.Dataset.range(5)
    dataset = dataset.apply(testing.assert_next(["MapAndBatch"]))
    dataset = dataset.map(lambda x: x * 2).batch(5)
    self.evaluate(dataset.reduce(0, lambda state, value: state))

  @combinations.generate(test_base.default_test_combinations())
  def testName(self):
    # [FUNCTION DOC]
    # Reduces a single-element dataset with a custom operation name and a function returning the
    # new value; asserts the result is forty-two.
    dataset = dataset_ops.Dataset.from_tensors(42)
    self.assertEqual(
        self.evaluate(
            dataset.reduce(0, lambda state, value: value, name="reduce")), 42)


if __name__ == "__main__":
  test.main()
