# [MODULE DOC]
# Tests for the flat-map dataset transformation. Covers basic and nested flat-maps, a shared
# iterator resource used from two sessions on a local server, dictionary, sparse, tensor array and
# ragged elements, operation names, cardinality computation at several effort levels, error
# context for failing user functions, symbolic checkpoint size, iterator checkpointing (including
# error cases for captured variables and stateful ops, and skipping), and global shuffling (the
# two global-shuffle test classes are unconditionally skipped pending a kernel fix).
# No import-time side effects other than class decoration; runs the test main when executed.
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
"""Tests for `tf.data.Dataset.flat_map()`."""
import random
from typing import Callable, Optional
import unittest

from absl.testing import parameterized
import numpy as np

from tensorflow.core.framework import dataset_options_pb2
from tensorflow.python.client import session
from tensorflow.python.data.experimental.ops import global_shuffle_op
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import function
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.ragged import ragged_conversion_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib


class FlatMapTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Functional tests for the flat-map transformation (map each element to a dataset and
  # concatenate the results). Uses the shared dataset test base class.

  # pylint: disable=g-long-lambda
  @combinations.generate(test_base.default_test_combinations())
  def testFlatMapDataset(self):
    # [FUNCTION DOC]
    # Flat-maps a list of repeat counts, each element x becoming x copies of a one-item list.
    # Asserts the output is each value repeated as many times as its value (zero gives nothing).
    repeats = [1, 2, 3, 4, 5, 0, 1]
    components = np.array(repeats, dtype=np.int64)
    dataset = dataset_ops.Dataset.from_tensor_slices(components).flat_map(
        lambda x: dataset_ops.Dataset.from_tensors([x]).repeat(x)
    )
    expected_output = []
    for i in repeats:
      expected_output.extend([[i]] * i)
    self.assertDatasetProduces(dataset, expected_output=expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testNestedFlatMapDataset(self):
    # [FUNCTION DOC]
    # Flat-maps rows of a matrix into slices, which are themselves flat-mapped so that each value y
    # becomes y copies of itself. Asserts the output lists every value repeated by its own value.
    repeats = [[1, 2], [3, 4], [5, 0], [1, 7]]
    components = np.array(repeats, dtype=np.int64)
    dataset = dataset_ops.Dataset.from_tensor_slices(components).flat_map(
        lambda x: dataset_ops.Dataset.from_tensor_slices(x).flat_map(
            lambda y: dataset_ops.Dataset.from_tensors(y).repeat(y)
        )
    )
    expected_output = []
    for row in repeats:
      for i in row:
        expected_output.extend([i] * i)
    self.assertDatasetProduces(dataset, expected_output=expected_output)

  @combinations.generate(test_base.graph_only_combinations())
  def testSharedResourceNestedFlatMapDataset(self):
    # [FUNCTION DOC]
    # Graph-only test. Builds the nested flat-map pipeline with an initializable iterator that has
    # a shared name, starts a local server and opens two sessions on it. Three times, it
    # initializes the iterator through a randomly chosen session and then reads each expected
    # element through randomly chosen sessions, asserting each value matches the nested flat-map
    # order. Finally asserts that one more read raises an out-of-range error.
    # Side effects: starts an in-process local server and creates two sessions.
    repeats = [[1, 2], [3, 4], [5, 0], [1, 7]]
    components = np.array(repeats, dtype=np.int64)
    iterator = dataset_ops.make_initializable_iterator(
        dataset_ops.Dataset.from_tensor_slices(components).flat_map(
            lambda x: dataset_ops.Dataset.from_tensor_slices(x).flat_map(
                lambda y: dataset_ops.Dataset.from_tensors(y).repeat(y)
            )
        ),
        shared_name="shared_flat_map_iterator",
    )
    init_op = iterator.initializer
    get_next = iterator.get_next()

    # Create two concurrent sessions that share the same iterator
    # resource on the same server, and verify that a random
    # interleaving of `Session.run(get_next)` calls on the two
    # sessions yields the expected result.
    server = server_lib.Server.create_local_server()
    with session.Session(server.target) as sess1:
      with session.Session(server.target) as sess2:
        for _ in range(3):
          sess = random.choice([sess1, sess2])
          sess.run(init_op)
          for row in repeats:
            for i in row:
              for _ in range(i):
                sess = random.choice([sess1, sess2])
                self.assertEqual(i, sess.run(get_next))

        with self.assertRaises(errors.OutOfRangeError):
          sess = random.choice([sess1, sess2])
          sess.run(get_next)

  @combinations.generate(test_base.default_test_combinations())
  def testMapDict(self):
    # [FUNCTION DOC]
    # Maps a range of ten to dictionaries with a doubled value and a squared count, then flat-maps
    # each into the doubled value repeated by the squared count. Asserts every element in order and
    # that the iterator then raises an out-of-range error.
    dataset = (
        dataset_ops.Dataset.range(10)
        .map(lambda x: {"foo": x * 2, "bar": x**2})
        .flat_map(
            lambda d: dataset_ops.Dataset.from_tensors(d["foo"]).repeat(
                d["bar"]
            )
        )
    )
    get_next = self.getNext(dataset)
    for i in range(10):
      for _ in range(i**2):
        self.assertEqual(i * 2, self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testSparse(self):
    # [FUNCTION DOC]
    # Maps a range of ten to two-by-two diagonal sparse tensors (values i and minus i) and
    # flat-maps each by densifying and slicing into rows. Asserts the output rows are [i, 0] and
    # [0, minus i] for each i.
    def _map_fn(i):
      # [FUNCTION DOC]
      # HELPER: turns an integer i into a two-by-two sparse value with entries i and minus i on the
      # diagonal.
      return sparse_tensor.SparseTensorValue(
          indices=[[0, 0], [1, 1]], values=(i * [1, -1]), dense_shape=[2, 2]
      )

    def _flat_map_fn(x):
      # [FUNCTION DOC]
      # HELPER: converts a sparse tensor to a dense matrix and returns a dataset of its rows.
      return dataset_ops.Dataset.from_tensor_slices(
          sparse_ops.sparse_to_dense(x.indices, x.dense_shape, x.values)
      )

    dataset = dataset_ops.Dataset.range(10).map(_map_fn).flat_map(_flat_map_fn)
    expected_output = []
    for i in range(10):
      for j in range(2):
        expected_output.append([i, 0] if j % 2 == 0 else [0, -i])
    self.assertDatasetProduces(dataset, expected_output=expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testTensorArray(self):
    # [FUNCTION DOC]
    # Maps a range of ten to tensor arrays holding the integers below i and flat-maps each into a
    # dataset of its stacked values. Asserts the output is the concatenation of the ranges below
    # each i, and that the flat-map function receives a tensor array.
    def _map_fn(i):
      # [FUNCTION DOC]
      # HELPER: casts i to 32-bit integer and returns a tensor array of that size filled with the
      # integers from zero up to i.
      i = math_ops.cast(i, dtypes.int32)
      return tensor_array_ops.TensorArray(
          dtype=dtypes.int32, element_shape=(), size=i
      ).unstack(math_ops.range(i))

    def _flat_map_fn(x):
      # [FUNCTION DOC]
      # HELPER: asserts the input is a tensor array, stacks it and returns a dataset of its items.
      self.assertIsInstance(x, tensor_array_ops.TensorArray)
      return dataset_ops.Dataset.from_tensor_slices(x.stack())

    dataset = dataset_ops.Dataset.range(10).map(_map_fn).flat_map(_flat_map_fn)

    expected_output = []
    for i in range(10):
      for j in range(i):
        expected_output.append(j)

    self.assertDatasetProduces(dataset, expected_output=expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testRagged(self):
    # [FUNCTION DOC]
    # Maps a range of ten to ragged tensors with rows [i] and [minus i] and flat-maps each by
    # converting to a dense tensor and slicing it. Asserts the output is [i] then [minus i] for
    # each i.
    def _map_fn(i):
      # [FUNCTION DOC]
      # HELPER: builds a ragged tensor from the column matrix with entries i and minus i.
      return ragged_tensor.RaggedTensor.from_tensor(i * [[1], [-1]])

    def _flat_map_fn(x):
      # [FUNCTION DOC]
      # HELPER: converts the ragged tensor to dense and returns a dataset of its rows.
      return dataset_ops.Dataset.from_tensor_slices(
          ragged_conversion_ops.to_tensor(x)
      )

    dataset = dataset_ops.Dataset.range(10).map(_map_fn).flat_map(_flat_map_fn)
    expected_output = []
    for i in range(10):
      expected_output.append([i])
      expected_output.append([-i])
    self.assertDatasetProduces(dataset, expected_output=expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testName(self):
    # [FUNCTION DOC]
    # Verifies flat-map accepts an operation name: flat-maps a single-element dataset holding 42
    # through a function returning a one-element dataset, with a name, and asserts the output is 42.
    def fn(x):
      # [FUNCTION DOC]
      # HELPER: returns a one-element dataset containing its input.
      return dataset_ops.Dataset.from_tensors(x)

    dataset = dataset_ops.Dataset.from_tensors(42).flat_map(fn, name="flat_map")
    self.assertDatasetProduces(dataset, [42])

  @combinations.generate(test_base.default_test_combinations())
  def testCardinality(self):
    # [FUNCTION DOC]
    # Flat-maps three rows of three numbers into their slices and computes the cardinality with
    # the moderate compute level. Asserts it is nine.
    dataset = dataset_ops.Dataset.from_tensor_slices(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    dataset = dataset.flat_map(dataset_ops.Dataset.from_tensor_slices)
    options = dataset_options_pb2.CardinalityOptions(
        compute_level="CARDINALITY_COMPUTE_MODERATE")
    cardinality = dataset_ops.gen_dataset_ops.dataset_cardinality(
        dataset._variant_tensor, options.SerializeToString())
    self.assertEqual(self.evaluate(cardinality), 9)

  @combinations.generate(test_base.default_test_combinations())
  def testInfiniteCardinality(self):
    # [FUNCTION DOC]
    # Flat-maps each of three elements to an infinitely repeated dataset and computes the
    # cardinality at the moderate level. Asserts it is the infinite marker.
    dataset = dataset_ops.Dataset.from_tensor_slices(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    dataset = dataset.flat_map(lambda _: dataset_ops.Dataset.range(1).repeat())
    options = dataset_options_pb2.CardinalityOptions(
        compute_level="CARDINALITY_COMPUTE_MODERATE")
    cardinality = dataset_ops.gen_dataset_ops.dataset_cardinality(
        dataset._variant_tensor, options.SerializeToString())
    self.assertEqual(self.evaluate(cardinality), dataset_ops.INFINITE)

  @combinations.generate(test_base.default_test_combinations())
  def testUnknownCardinality(self):
    # [FUNCTION DOC]
    # Flat-maps each of three elements to a filtered range (odd numbers below ten), whose size is
    # not statically known, and computes the cardinality at the moderate level. Asserts it is the
    # unknown marker.
    dataset = dataset_ops.Dataset.from_tensor_slices(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    dataset = dataset.flat_map(
        lambda _: dataset_ops.Dataset.range(10).filter(lambda x: x % 2 == 1))
    options = dataset_options_pb2.CardinalityOptions(
        compute_level="CARDINALITY_COMPUTE_MODERATE")
    cardinality = dataset_ops.gen_dataset_ops.dataset_cardinality(
        dataset._variant_tensor, options.SerializeToString())
    self.assertEqual(self.evaluate(cardinality), dataset_ops.UNKNOWN)

  @combinations.generate(test_base.default_test_combinations())
  def testEmptyCardinality(self):
    # [FUNCTION DOC]
    # Flat-maps an empty range with the range constructor and computes the cardinality at the
    # moderate level. Asserts it is zero.
    dataset = dataset_ops.Dataset.range(0)
    dataset = dataset.flat_map(dataset_ops.Dataset.range)
    options = dataset_options_pb2.CardinalityOptions(
        compute_level="CARDINALITY_COMPUTE_MODERATE")
    cardinality = dataset_ops.gen_dataset_ops.dataset_cardinality(
        dataset._variant_tensor, options.SerializeToString())
    self.assertEqual(self.evaluate(cardinality), 0)

  @combinations.generate(test_base.default_test_combinations())
  def testCardinalityLowEffort(self):
    # [FUNCTION DOC]
    # Same dataset as the basic cardinality test, but computed at the low compute level. Asserts
    # the result is the unknown marker, since low effort does not look inside the flat-map.
    dataset = dataset_ops.Dataset.from_tensor_slices(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    dataset = dataset.flat_map(dataset_ops.Dataset.from_tensor_slices)
    options = dataset_options_pb2.CardinalityOptions(
        compute_level="CARDINALITY_COMPUTE_LOW")
    cardinality = dataset_ops.gen_dataset_ops.dataset_cardinality(
        dataset._variant_tensor, options.SerializeToString())
    self.assertEqual(self.evaluate(cardinality), dataset_ops.UNKNOWN)

  @combinations.generate(test_base.default_test_combinations())
  def testMapFuncFailWithErrorContext(self):
    # [FUNCTION DOC]
    # Flat-maps a single element through a function that divides by zero. Asserts reading the first
    # element raises an invalid-argument error whose message mentions the user-defined function
    # passed to the transformation and the iterator path.

    def fn(x):
      # [FUNCTION DOC]
      # HELPER: integer-divides its input by zero, which fails at run time, then wraps the result in
      # a one-element dataset.
      return dataset_ops.Dataset.from_tensors(x // 0)

    dataset = dataset_ops.Dataset.from_tensors(42).flat_map(fn, name="flat_map")
    get_next = self.getNext(dataset)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        r".*Error in user-defined function passed to .* transformation with "
        r"iterator: Iterator::Root::.*"):
      self.evaluate(get_next())

  @combinations.generate(test_base.v2_eager_only_combinations())
  def testSymbolicCheckpointSize(self):
    # [FUNCTION DOC]
    # Eager (version 2) only. Builds a pipeline whose inputs are large random vectors (over one
    # megabyte) and whose flat-map produces many large examples, enables the symbolic checkpoint
    # option, advances a numpy iterator by 30 elements, saves it, and asserts the serialized
    # checkpoint is under 10,000 bytes, i.e. no elements are stored in the checkpoint.
    examples_per_flat_map = 100
    example_len = 10_000

    def flat_map_fn(_):
      # [FUNCTION DOC]
      # HELPER: ignores its input and returns a dataset sliced from 100 deterministic random
      # vectors of length 10,000 (fixed seed).
      data = []
      for _ in range(examples_per_flat_map):
        data.append(
            stateless_random_ops.stateless_random_uniform(
                [example_len], seed=(42, 42)
            )
        )
      return dataset_ops.Dataset.from_tensor_slices(data)

    ds = dataset_ops.Dataset.range(10)
    # Inputs to flat_map are >1MB
    ds = ds.map(
        lambda x: stateless_random_ops.stateless_random_uniform(
            [1_000_000], seed=(42, 42)
        )
    )
    ds = ds.flat_map(flat_map_fn)

    options = options_lib.Options()
    options.experimental_symbolic_checkpoint = True
    ds = ds.with_options(options)

    it = ds.as_numpy_iterator()
    for _ in range(30):
      next(it)

    ckpt = it._save()
    # Make sure the checkpoint is smaller than the element sizes, i.e. no
    # elements are being stored in the checkpoint.
    self.assertLess(len(ckpt.numpy()), 10_000)


class FlatMapCheckpointTest(
    checkpoint_test_base.CheckpointTestBase, parameterized.TestCase
):
  # [CLASS DOC]
  # Checkpoint tests for the flat-map iterator. Inherits the checkpoint test base class, which
  # supplies the verification callback and the error-on-save check.

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(symbolic_checkpoint=[False, True]),
      )
  )
  def test(self, verify_fn, symbolic_checkpoint):
    # [FUNCTION DOC]
    # Parameters: verify_fn (verification callback), symbolic_checkpoint (whether the symbolic
    # checkpoint option is enabled). Builds a flat-map of five ranges of five consecutive numbers
    # (25 elements total) and runs the verification expecting 25 outputs.
    # Complicated way of saying range(start, start+25).
    def build_ds(start):
      # [FUNCTION DOC]
      # HELPER: builds the flat-mapped dataset starting at the given number, applying the
      # symbolic checkpoint option, and returns it.
      def map_fn(x):
        # [FUNCTION DOC]
        # HELPER: returns a range dataset of five consecutive numbers starting at x.
        return dataset_ops.Dataset.range(x, x + 5)

      dataset = dataset_ops.Dataset.range(start, start + 5 * 5, 5)
      dataset = dataset.flat_map(map_fn)
      options = options_lib.Options()
      options.experimental_symbolic_checkpoint = symbolic_checkpoint
      return dataset.with_options(options)

    verify_fn(self, lambda: build_ds(0), num_outputs=25)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
      )
  )
  def testNested(self, verify_fn):
    # [FUNCTION DOC]
    # Checkpoint test. Parameter verify_fn is the verification callback. Builds a dataset whose
    # single element is a dataset of 42 numbers, flattened by an identity flat-map, and verifies
    # checkpointing with 42 outputs.
    def build_ds():
      # [FUNCTION DOC]
      # HELPER: builds the nested-dataset pipeline under test and returns it.
      inner_ds = dataset_ops.Dataset.from_tensor_slices(range(42))
      ds = dataset_ops.Dataset.from_tensors(inner_ds)
      return ds.flat_map(lambda x: x)

    verify_fn(self, build_ds, num_outputs=42)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
      )
  )
  def testMapThenFlatMap(self, verify_fn):
    # [FUNCTION DOC]
    # Checkpoint test. Parameter verify_fn is the verification callback. Flat-maps a range of five
    # into mapped ranges of 100 (each value multiplied by ten) and verifies checkpointing with 500
    # outputs.
    def build_ds():
      # [FUNCTION DOC]
      # HELPER: builds the range-of-five dataset flat-mapped through the function below.
      def flat_map_fn(_):
        # [FUNCTION DOC]
        # HELPER: ignores its input and returns a range of 100 mapped through the function below.
        def map_fn(y):
          # [FUNCTION DOC]
          # HELPER: casts y to 32-bit integer and multiplies it by ten.
          return 10 * math_ops.cast(y, dtypes.int32)

        return dataset_ops.Dataset.range(100).map(map_fn)

      return dataset_ops.Dataset.range(5).flat_map(flat_map_fn)

    verify_fn(self, build_ds, num_outputs=500)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
      )
  )
  def testCaptureDefunInMapFn(self, verify_fn):
    # [FUNCTION DOC]
    # Checkpoint test. Parameter verify_fn is the verification callback. Flat-maps a range of 100
    # through a function that calls a graph function defined inside it, and verifies
    # checkpointing with 100 outputs.
    def build_ds():
      # [FUNCTION DOC]
      # HELPER: builds the range-of-100 dataset flat-mapped through the function below.
      def map_fn(x):
        # [FUNCTION DOC]
        # HELPER: returns a one-element dataset holding the result of the inner graph function.
        @function.Defun(dtypes.int64)
        def defun_fn(x):
          # [FUNCTION DOC]
          # HELPER: graph function that adds the constant 1000 to its 64-bit input cast to 32-bit.
          return constant_op.constant(1000) + math_ops.cast(x, dtypes.int32)

        return dataset_ops.Dataset.from_tensor_slices([defun_fn(x)])

      return dataset_ops.Dataset.range(100).flat_map(map_fn)

    verify_fn(self, build_ds, num_outputs=100)

  @combinations.generate(test_base.default_test_combinations())
  def testDisallowVariableCapture(self):
    # [FUNCTION DOC]
    # Builds a flat-map pipeline whose function captures a resource variable and asserts that
    # saving an iterator checkpoint fails with a failed-precondition error.
    def build_ds():
      # [FUNCTION DOC]
      # HELPER: creates a scalar resource variable and returns a range-of-five dataset flat-mapped
      # through a function that captures it.
      test_var = variable_scope.get_variable(
          name="test_var", shape=(), use_resource=True
      )
      return dataset_ops.Dataset.range(5).flat_map(
          lambda _: dataset_ops.Dataset.from_tensor_slices([test_var])
      )

    self.verify_error_on_save(build_ds, 5, errors.FailedPreconditionError)

  @combinations.generate(test_base.default_test_combinations())
  def testDisallowCapturingStatefulOps(self):
    # [FUNCTION DOC]
    # Builds a flat-map pipeline whose inner map uses a stateful random operation and asserts that
    # saving an iterator checkpoint fails with a failed-precondition error.
    def build_ds():
      # [FUNCTION DOC]
      # HELPER: builds the range-of-five dataset flat-mapped through the function below.
      def flat_map_fn(_):
        # [FUNCTION DOC]
        # HELPER: ignores its input and returns a range of 100 mapped through the function below.
        def map_fn(x):
          # [FUNCTION DOC]
          # HELPER: multiplies a random uniform integer in zero to ten (stateful) by x cast to
          # 32-bit integer.
          return random_ops.random_uniform(
              (), 0, 10, dtype=dtypes.int32
          ) * math_ops.cast(x, dtypes.int32)

        return dataset_ops.Dataset.range(100).map(map_fn)

      return dataset_ops.Dataset.range(5).flat_map(flat_map_fn)

    self.verify_error_on_save(build_ds, 500, errors.FailedPreconditionError)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
      )
  )
  def testSparse(self, verify_fn):
    # [FUNCTION DOC]
    # Checkpoint test. Parameter verify_fn is the verification callback. Maps a range of ten to
    # sparse diagonal matrices, flat-maps them into dense rows and verifies checkpointing with 20
    # outputs.
    def _map_fn(i):
      # [FUNCTION DOC]
      # HELPER: turns an integer i into a two-by-two sparse value with entries i and minus i on the
      # diagonal.
      return sparse_tensor.SparseTensorValue(
          indices=[[0, 0], [1, 1]], values=(i * [1, -1]), dense_shape=[2, 2]
      )

    def _flat_map_fn(x):
      # [FUNCTION DOC]
      # HELPER: converts a sparse tensor to a dense matrix and returns a dataset of its rows.
      return dataset_ops.Dataset.from_tensor_slices(
          sparse_ops.sparse_to_dense(x.indices, x.dense_shape, x.values)
      )

    def _build_ds():
      # [FUNCTION DOC]
      # HELPER: builds the range-of-ten, mapped then flat-mapped dataset under test.
      return dataset_ops.Dataset.range(10).map(_map_fn).flat_map(_flat_map_fn)

    verify_fn(self, _build_ds, num_outputs=20)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(symbolic_checkpoint=[True],
                               num_skips=[3, 4]),
      )
  )
  def testWithSkip(self, verify_fn, symbolic_checkpoint, num_skips):
    # [FUNCTION DOC]
    # Checkpoint test for a flat-map followed by a skip. Parameters: verify_fn (verification
    # callback), symbolic_checkpoint (symbolic checkpoint option value, always true here),
    # num_skips (number of leading elements skipped, three or four). Verifies checkpointing of a
    # dataset of 12 elements (three inner datasets of four) minus the skipped ones.
    """Test `.flat_map().skip()` checkpointing behavior.

    `SkipInternal` and `GetNextInternal` are separate functions
    but with slightly different implementations.
    Therefore, we should test this op's behavior when used with `.skip()`.

    Args:
      verify_fn: Verify the correctness of this dataset's checkpointing.
      symbolic_checkpoint: Whether symbolic checkpointing is turned on.
      num_skips: `.skip(num_skips)`
    """

    def build_dataset():
      # [FUNCTION DOC]
      # HELPER: flat-maps the indices zero to two through the function below, skips num_skips
      # elements, applies the symbolic checkpoint option and returns the dataset.
      def my_map(x):
        # [FUNCTION DOC]
        # HELPER: selects an inner dataset of four consecutive numbers by index: zero to three for
        # index 0, four to seven for index 1, otherwise eight to eleven.
        if x == 0:
          return dataset_ops.Dataset.from_tensor_slices([0, 1, 2, 3])
        elif x == 1:
          return dataset_ops.Dataset.from_tensor_slices([4, 5, 6, 7])
        else:
          return dataset_ops.Dataset.from_tensor_slices([8, 9, 10, 11])

      indices = dataset_ops.Dataset.from_tensor_slices([0, 1, 2])
      dataset = indices.flat_map(my_map)
      # Skip some elements
      dataset = dataset.skip(num_skips)

      options = options_lib.Options()
      options.experimental_symbolic_checkpoint = symbolic_checkpoint
      return dataset.with_options(options)

    verify_fn(self, build_dataset, num_outputs=3 * 4 - num_skips)


@unittest.skip(
    "TODO: b/355241367 - `flat_map_dataset_op.cc` still needs to be fixed."
    " Please use concatenate dataset op plus global shuffling instead."
)
class FlatMapGlobalShuffleTest(
    test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests global shuffling of a flat-map dataset. The whole class is skipped unconditionally
  # because the flat-map kernel still needs a fix (concatenation plus global shuffle is advised).

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
    # Skipped with its class. Parameters: repetitions (repeat count), seed (shuffle seed or
    # nothing), reshuffle_each_iteration (whether order changes per epoch). Builds a prefetched
    # flat-map over three rows, optionally repeats it, prefetches, globally shuffles, and asserts
    # the output has the same items as the repeated 1 to 9 range, differs in order, and its length
    # equals the dataset cardinality.
    dataset = dataset_ops.Dataset.from_tensor_slices(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    dataset = dataset.prefetch(buffer_size=dataset_ops.AUTOTUNE)
    dataset = dataset.flat_map(dataset_ops.Dataset.from_tensor_slices)
    if repetitions > 1:
      dataset = dataset.repeat(repetitions)
    dataset = dataset.prefetch(buffer_size=dataset_ops.AUTOTUNE)
    dataset = global_shuffle_op._global_shuffle(
        dataset, seed=seed, reshuffle_each_iteration=reshuffle_each_iteration)

    expected = list(range(1, 10)) * repetitions
    dataset_output = self.getDatasetOutput(
        dataset, requires_initialization=True)
    self.assertCountEqual(dataset_output, expected)
    self.assertNotEqual(dataset_output, expected)
    self.assertLen(dataset_output, self.evaluate(dataset.cardinality()))

  @combinations.generate(test_base.default_test_combinations())
  def testInputCardinalityTooLarge(self):
    # [FUNCTION DOC]
    # Skipped with its class. Flat-maps a dataset of 101 rows and applies the global shuffle with a
    # fixed seed; asserts reading the output raises a failed-precondition error saying the input
    # cardinality of the flat-map is too large to support global shuffling.
    dataset = dataset_ops.Dataset.from_tensor_slices([[i] for i in range(101)])
    dataset = dataset.flat_map(dataset_ops.Dataset.from_tensor_slices)
    with self.assertRaisesRegex(
        errors.FailedPreconditionError,
        "The cardinality of the input to FlatMapDataset is too large to support"
        " global shuffling",
    ):
      dataset = global_shuffle_op._global_shuffle(dataset, seed=42)
      self.getDatasetOutput(dataset, requires_initialization=True)


@unittest.skip(
    "TODO: b/355241367 - `flat_map_dataset_op.cc` still needs to be fixed."
    " Please use concatenate dataset op plus global shuffling instead."
)
class FlatMapGlobalShuffleCheckpointTest(
    checkpoint_test_base.CheckpointTestBase, parameterized.TestCase
):
  # [CLASS DOC]
  # Checkpoint tests for a globally shuffled flat-map dataset. Skipped unconditionally for the
  # same reason as the global shuffle test class.

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
    # Skipped with its class. Parameters: verify_fn (verification callback), repetitions (repeat
    # count), reshuffle_each_iteration (reshuffle per epoch), symbolic_checkpoint (value of the
    # symbolic checkpoint option). Runs the verification on the shuffled flat-map dataset
    # expecting nine times repetitions outputs, ignoring order when reshuffling each iteration.

    def _build_dataset() -> dataset_ops.Dataset:
      # [FUNCTION DOC]
      # HELPER: builds a prefetched flat-map over three rows, optionally repeated, prefetched,
      # globally shuffled with a fixed seed of 42 and given the symbolic checkpoint option.
      dataset = dataset_ops.Dataset.from_tensor_slices(
          [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
      dataset = dataset.prefetch(buffer_size=dataset_ops.AUTOTUNE)
      dataset = dataset.flat_map(dataset_ops.Dataset.from_tensor_slices)
      if repetitions > 1:
        dataset = dataset.repeat(repetitions)
      dataset = dataset.prefetch(buffer_size=dataset_ops.AUTOTUNE)
      dataset = global_shuffle_op._global_shuffle(
          dataset, seed=42, reshuffle_each_iteration=reshuffle_each_iteration)
      options = options_lib.Options()
      options.experimental_symbolic_checkpoint = symbolic_checkpoint
      return dataset.with_options(options)

    verify_fn(
        self,
        _build_dataset,
        num_outputs=9 * repetitions,
        assert_items_equal=reshuffle_each_iteration)


if __name__ == "__main__":
  test.main()
