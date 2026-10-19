# [MODULE DOC]
# Tests for the experimental map_and_batch transformation (fused map then batch): parallelism
# options, partial batches, parallel get-next, sparse inputs, error handling, dtype coverage,
# short-circuited map functions, control flow, large-batch checkpointing, and iterator
# checkpoint/restore. Test module only; no import-time side effects.
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
"""Tests for `tf.data.experimental.map_and_batch()`."""
import math

from absl.testing import parameterized
import numpy as np

from tensorflow.python import pywrap_sanitizers
from tensorflow.python.checkpoint import checkpoint as trackable_utils
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.data.experimental.ops import batching
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class MapAndBatchTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Behavioral tests for map_and_batch using the common dataset test base (getNext,
  # assertDatasetProduces and structured-dataset helpers). No shared fixtures.

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              num_parallel_calls=[None, 1, 2], num_parallel_batches=None) +
          combinations.combine(
              num_parallel_calls=None, num_parallel_batches=10)))
  def testMapAndBatch(self, num_parallel_calls, num_parallel_batches):
    # [FUNCTION DOC]
    # Parameterized over parallel calls (none, 1, 2) or parallel batches (10). Maps a squaring
    # function over three components of 7 elements, repeated, and batched. Checks the static
    # output shapes, full batches when batch size divides the total (14 of 28 repeats), a
    # smaller final batch when it does not (batch 8, 14 repeats), an empty dataset producing
    # nothing, and that batch size zero raises an invalid-argument error.
    """Test a dataset that maps a TF function across its input elements."""
    # The pipeline is TensorSliceDataset ->
    # RepeatDataset(count) -> MapAndBatchDataset(square_3, batch_size).
    components = (np.arange(7),
                  np.array([[1, 2, 3]]) * np.arange(7)[:, np.newaxis],
                  np.array(37.0) * np.arange(7))

    def _map_fn(x, y, z):
      # [FUNCTION DOC]
      # HELPER: squares each of the three components.
      return math_ops.square(x), math_ops.square(y), math_ops.square(z)

    def dataset_fn(batch_size, count):
      # [FUNCTION DOC]
      # HELPER: builds the pipeline from the components, repeated count times, followed by
      # map_and_batch with the enclosing test's parallelism settings; returns the dataset.
      dataset = dataset_ops.Dataset.from_tensor_slices(components).repeat(
          count).apply(
              batching.map_and_batch(
                  map_func=_map_fn,
                  batch_size=batch_size,
                  num_parallel_calls=num_parallel_calls,
                  num_parallel_batches=num_parallel_batches))
      return dataset

    # Batch of a finite input, where the batch_size divides the
    # total number of elements.
    dataset = dataset_fn(14, 28)
    get_next = self.getNext(dataset)
    self.assertEqual(
        [[None] + list(c.shape[1:]) for c in components],
        [shape.as_list()
         for shape in dataset_ops.get_legacy_output_shapes(dataset)])
    num_batches = (28 * 7) // 14
    for i in range(num_batches):
      result = self.evaluate(get_next())
      for component, result_component in zip(components, result):
        for j in range(14):
          self.assertAllEqual(component[(i * 14 + j) % 7]**2,
                              result_component[j])
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

    # Batch of a finite input, where the batch_size does not
    # divide the total number of elements.
    get_next = self.getNext(dataset_fn(8, 14))

    # We expect (num_batches - 1) full-sized batches.
    num_batches = int(math.ceil((14 * 7) / 8))
    for i in range(num_batches - 1):
      result = self.evaluate(get_next())
      for component, result_component in zip(components, result):
        for j in range(8):
          self.assertAllEqual(component[(i * 8 + j) % 7]**2,
                              result_component[j])

    result = self.evaluate(get_next())
    for component, result_component in zip(components, result):
      for j in range((14 * 7) % 8):
        self.assertAllEqual(component[((num_batches - 1) * 8 + j) % 7]**2,
                            result_component[j])
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

    # Batch of an empty input should fail straight away.
    self.assertDatasetProduces(dataset_fn(8, 0), expected_output=[])

    # Empty batch should be an initialization time error.
    with self.assertRaises(errors.InvalidArgumentError):
      self.assertDatasetProduces(dataset_fn(0, 14), expected_output=[])

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(drop_remainder=[True, False])))
  def testMapAndBatchPartialBatch(self, drop_remainder):
    # [FUNCTION DOC]
    # Maps squaring (reshaped to length one) over range(10) with batch size 4, with and without
    # dropping the remainder. Asserts the static shape (known batch dimension only when
    # dropping) and the produced batches, including the final partial batch when kept.
    dataset = (
        dataset_ops.Dataset.range(10).apply(
            batching.map_and_batch(
                lambda x: array_ops.reshape(x * x, [1]),
                batch_size=4,
                drop_remainder=drop_remainder)))

    if drop_remainder:
      self.assertEqual(
          [4, 1], dataset_ops.get_legacy_output_shapes(dataset).as_list())
    else:
      self.assertEqual(
          [None, 1], dataset_ops.get_legacy_output_shapes(dataset).as_list())
    expected_output = [[[0], [1], [4], [9]], [[16], [25], [36], [49]]]
    if not drop_remainder:
      expected_output.append([[64], [81]])
    self.assertDatasetProduces(dataset, expected_output=expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testMapAndBatchYieldsPartialBatch(self):
    # [FUNCTION DOC]
    # With default settings (no drop remainder), batch size 4 over range(10) gives an unknown
    # batch dimension and a final batch of two elements; asserts shape and output.
    dataset = (
        dataset_ops.Dataset.range(10).apply(
            batching.map_and_batch(lambda x: array_ops.reshape(x * x, [1]), 4)))

    self.assertEqual(
        [None, 1], dataset_ops.get_legacy_output_shapes(dataset).as_list())
    expected_output = [[[0], [1], [4], [9]], [[16], [25], [36], [49]],
                       [[64], [81]]]
    self.assertDatasetProduces(dataset, expected_output=expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testMapAndBatchParallelGetNext(self):
    # [FUNCTION DOC]
    # Creates 100 concurrent get-next calls on an iterator over 50000 elements batched by 100
    # (eager: the internal next function; graph: a one-shot iterator). Over five rounds,
    # evaluates all 100 together, sorts the batches and asserts they are consecutive ranges
    # of 100; then asserts the next evaluation raises an out-of-range error.
    dataset = dataset_ops.Dataset.range(50000).apply(
        batching.map_and_batch(lambda x: x, batch_size=100))

    if context.executing_eagerly():
      iterator = iter(dataset)
      get_next = iterator._next_internal  # pylint: disable=protected-access
    else:
      iterator = dataset_ops.make_one_shot_iterator(dataset)
      get_next = iterator.get_next

    elements = []
    for _ in range(100):
      elements.append(get_next)

    for i in range(5):
      got = self.evaluate([element() for element in elements])
      got.sort(key=lambda x: x[0])
      expected = []
      for j in range(100):
        expected.append(range(i * 10000 + j * 100, i * 10000 + (j + 1) * 100))
      self.assertAllEqual(got, expected)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate([element() for element in elements])

  @combinations.generate(test_base.default_test_combinations())
  def testMapAndBatchParallelGetNextDropRemainder(self):
    # [FUNCTION DOC]
    # Like the parallel get-next test but with 49999 elements and the remainder dropped: four
    # rounds of 100 concurrent batches of 100 are checked, then an out-of-range error is
    # expected.
    dataset = dataset_ops.Dataset.range(49999).apply(
        batching.map_and_batch(
            lambda x: x, batch_size=100, drop_remainder=True))

    if context.executing_eagerly():
      iterator = iter(dataset)
      get_next = iterator._next_internal  # pylint: disable=protected-access
    else:
      iterator = dataset_ops.make_one_shot_iterator(dataset)
      get_next = iterator.get_next

    elements = []
    for _ in range(100):
      elements.append(get_next)

    for i in range(4):
      got = self.evaluate([element() for element in elements])
      got.sort(key=lambda x: x[0])
      expected = []
      for j in range(100):
        expected.append(range(i * 10000 + j * 100, i * 10000 + (j + 1) * 100))
      self.assertAllEqual(got, expected)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate([element() for element in elements])

  @combinations.generate(test_base.default_test_combinations())
  def testMapAndBatchSparse(self):
    # [FUNCTION DOC]
    # Maps each integer of range(10) to a one-element sparse tensor and batches by 5; asserts
    # two sparse batches with the expected indices, values and dense shape.

    def _sparse(i):
      # [FUNCTION DOC]
      # HELPER: returns a sparse tensor value of dense shape [1] holding the integer at
      # index zero.
      return sparse_tensor.SparseTensorValue(
          indices=[[0]], values=(i * [1]), dense_shape=[1])

    dataset = dataset_ops.Dataset.range(10).apply(
        batching.map_and_batch(_sparse, 5))

    self.assertDatasetProduces(
        dataset,
        expected_output=[
            sparse_tensor.SparseTensorValue(
                indices=[[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]],
                values=[i * 5, i * 5 + 1, i * 5 + 2, i * 5 + 3, i * 5 + 4],
                dense_shape=[5, 1]) for i in range(2)
        ])

  @combinations.generate(test_base.default_test_combinations())
  def testMapAndBatchFails(self):
    # [FUNCTION DOC]
    # A dataset whose single element is a numeric check on a division by zero fails; asserts
    # fetching the first batch raises an invalid-argument error containing the check message.
    # KNOWN ISSUE: the docstring is copy-pasted from another test and does not describe this
    # test, which checks error propagation.
    """Test a dataset that maps a TF function across its input elements."""

    with self.assertRaisesRegex(errors.InvalidArgumentError, "oops"):
      dataset = dataset_ops.Dataset.from_tensors(
          array_ops.check_numerics(
              constant_op.constant(1.0) / constant_op.constant(0.0), "oops"))
      dataset = dataset.apply(batching.map_and_batch(lambda x: x, 14))
      get_next = self.getNext(dataset, requires_initialization=True)
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testMapAndBatchShapeMismatch(self):
    # [FUNCTION DOC]
    # Builds a dataset from a generator whose last element has a different shape than the
    # first three, batched by 4; asserts iteration fails with an invalid-argument error saying
    # the number of elements does not match.
    # KNOWN ISSUE: the docstring is copy-pasted and does not describe this test.
    """Test a dataset that maps a TF function across its input elements."""

    def generator():
      # [FUNCTION DOC]
      # HELPER: generator yielding [1], [2], [3] and then a differently shaped [[4, 5, 6]].
      yield [1]
      yield [2]
      yield [3]
      yield [[4, 5, 6]]

    dataset = dataset_ops.Dataset.from_generator(
        generator, output_types=dtypes.int32)
    batch_size = 4
    dataset = dataset.apply(batching.map_and_batch(lambda x: x, batch_size))
    self.assertDatasetProduces(
        dataset,
        expected_error=(errors.InvalidArgumentError,
                        "number of elements does not match"))

  @combinations.generate(test_base.default_test_combinations())
  def testMapAndBatchImplicitDispose(self):
    # [FUNCTION DOC]
    # Builds a large pipeline (1000 elements repeated 1000 times, squared, batched by 100,
    # prefetched) and consumes only three batches, verifying the unfinished pipeline is
    # cleaned up without error.
    # Tests whether a map and batch dataset will be cleaned up correctly when
    # the pipeline does not run it until exhaustion.
    # The pipeline is TensorSliceDataset -> RepeatDataset(1000) ->
    # MapAndBatchDataset(f=square_3, batch_size=100).
    components = (np.arange(1000),
                  np.array([[1, 2, 3]]) * np.arange(1000)[:, np.newaxis],
                  np.array(37.0) * np.arange(1000))

    def _map_fn(x, y, z):
      # [FUNCTION DOC]
      # HELPER: squares each of the three components.
      return math_ops.square(x), math_ops.square(y), math_ops.square(z)

    dataset = dataset_ops.Dataset.from_tensor_slices(components).repeat(
        1000).apply(batching.map_and_batch(_map_fn, batch_size=100))
    dataset = dataset.prefetch(5)
    get_next = self.getNext(dataset)
    for _ in range(3):
      self.evaluate(get_next())

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(threshold=[0, 5, 10, 90, 95, 99]))
  )
  def testMapAndBatchMapError(self, threshold):
    # [FUNCTION DOC]
    # Parameterized over a threshold. A python function raises a stop-iteration signal for
    # inputs at or above the threshold. Over range(100) in batches of 10, asserts the batches
    # below the threshold are correct, later batches raise invalid-argument errors, and the
    # iterator finally raises out-of-range.

    def raising_py_fn(i):
      # [FUNCTION DOC]
      # HELPER: returns i unchanged, or raises a stop-iteration exception when i is at or above
      # the enclosing threshold.
      if i >= threshold:
        raise StopIteration()
      else:
        return i

    dataset = dataset_ops.Dataset.range(100).apply(
        batching.map_and_batch(
            lambda x: script_ops.py_func(raising_py_fn, [x], dtypes.int64),
            batch_size=10))

    get_next = self.getNext(dataset)
    for i in range(threshold // 10):
      self.assertAllEqual([i * 10 + j for j in range(10)],
                          self.evaluate(get_next()))
    for i in range(threshold // 10, 10):
      with self.assertRaises(errors.InvalidArgumentError):
        self.evaluate(get_next())
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(element=False, dtype=dtypes.bool) +
          combinations.combine(
              element=-42,
              dtype=[dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64]) +
          combinations.combine(element=42, dtype=[dtypes.uint8, dtypes.uint16])
          + combinations.combine(
              element=42.0,
              dtype=[dtypes.float16, dtypes.float32, dtypes.float64]) +
          combinations.combine(element=b"hello", dtype=[dtypes.string])))
  def testMapAndBatchTypes(self, element, dtype):
    # [FUNCTION DOC]
    # Parameterized over an element value and dtype (bool, ints, uints, floats, string). Builds
    # a generator dataset repeated 100 times with an identity map_and_batch of 10; asserts
    # each of 10 batches contains ten copies of the element.

    def gen():
      # [FUNCTION DOC]
      # HELPER: generator yielding the single parameterized element.
      yield element

    dataset = dataset_ops.Dataset.from_generator(gen, dtype).repeat(100).apply(
        batching.map_and_batch(lambda x: x, batch_size=10))

    get_next = self.getNext(dataset)
    for _ in range(10):
      self.assertAllEqual([element for _ in range(10)],
                          self.evaluate(get_next()))

  @combinations.generate(test_base.default_test_combinations())
  def testShortCircuitIdentity(self):
    # [FUNCTION DOC]
    # Identity map function (short-circuited by the optimizer) over a structured dataset
    # batched by 10; asserts the first batch equals the expected structured element.
    map_fn = lambda x: x
    dataset = self.structuredDataset(None).repeat().apply(
        batching.map_and_batch(map_fn, batch_size=10))
    get_next = self.getNext(dataset)
    expected = map_fn(self.evaluate(self.structuredElement(None, shape=[10])))
    self.assertAllEqual(expected, self.evaluate(get_next()))

  @combinations.generate(test_base.default_test_combinations())
  def testShortCircuitReplicate(self):
    # [FUNCTION DOC]
    # Map function returning two copies of its input; asserts the first batch of 10 equals the
    # expected replicated structure.
    map_fn = lambda x: (x, x)
    dataset = self.structuredDataset(None).repeat().apply(
        batching.map_and_batch(map_fn, batch_size=10))
    get_next = self.getNext(dataset)
    expected = map_fn(self.evaluate(self.structuredElement(None, shape=[10])))
    self.assertAllEqual(expected, self.evaluate(get_next()))

  @combinations.generate(test_base.default_test_combinations())
  def testShortCircuitSwap(self):
    # [FUNCTION DOC]
    # Map function swapping the two components of a pair; asserts the first batch of 10 equals
    # the expected swapped structure.
    map_fn = lambda x, y: (y, x)
    dataset = self.structuredDataset(
        (None,
         None)).repeat().apply(batching.map_and_batch(map_fn, batch_size=10))
    get_next = self.getNext(dataset)
    expected = map_fn(
        *self.evaluate(self.structuredElement((None, None), shape=[10])))
    self.assertAllEqual(expected, self.evaluate(get_next()))

  @combinations.generate(test_base.default_test_combinations())
  def testShortCircuitProject(self):
    # [FUNCTION DOC]
    # Map function keeping only the first component of a pair; asserts the first batch of 10
    # equals the expected projection.
    map_fn = lambda x, y: x
    dataset = self.structuredDataset(
        (None,
         None)).repeat().apply(batching.map_and_batch(map_fn, batch_size=10))
    get_next = self.getNext(dataset)
    expected = map_fn(
        *self.evaluate(self.structuredElement((None, None), shape=[10])))
    self.assertAllEqual(expected, self.evaluate(get_next()))

  @combinations.generate(test_base.default_test_combinations())
  def testShortCircuitCapturedInput(self):
    # [FUNCTION DOC]
    # Map function ignoring its input and returning a captured variable (value 42); after
    # initializing variables, asserts the first batch is ten copies of 42.
    captured_t = variables.Variable(42)
    dataset = self.structuredDataset(None).repeat().apply(
        batching.map_and_batch(lambda x: captured_t, batch_size=10))
    self.evaluate(variables.global_variables_initializer())
    get_next = self.getNext(dataset, requires_initialization=True)
    self.assertAllEqual([42] * 10, self.evaluate(get_next()))

  @combinations.generate(test_base.default_test_combinations())
  def testMapAndBatchControlFlow(self):
    # [FUNCTION DOC]
    # Maps a conditional function (add one below 50, otherwise square) over range(100) in
    # batches of 10; asserts the first five batches are incremented values and the last five
    # are squares, then out-of-range at the end.

    def map_fn(x):
      # [FUNCTION DOC]
      # HELPER: temporarily forces control-flow v2 on, builds a conditional (x plus one if x is
      # below 50, else x squared), restores the previous global setting, and returns the
      # result. Mutates and restores a module-level flag.
      previous_control_flow_v2_value = control_flow_util.ENABLE_CONTROL_FLOW_V2
      control_flow_util.ENABLE_CONTROL_FLOW_V2 = True
      return_value = cond.cond(x < 50, lambda: x + 1, lambda: x * x)
      control_flow_util.ENABLE_CONTROL_FLOW_V2 = previous_control_flow_v2_value
      return return_value

    dataset = dataset_ops.Dataset.range(100).apply(
        batching.map_and_batch(map_fn, batch_size=10))
    get_next = self.getNext(dataset)
    for i in range(10):
      if i < 5:
        self.assertAllEqual([i * 10 + j + 1 for j in range(10)],
                            self.evaluate(get_next()))
      else:
        self.assertAllEqual(
            [((i * 10) + j) * ((i * 10) + j) for j in range(10)],
            self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(test_base.eager_only_combinations())
  def testCheckpointLargeBatches(self):
    # [FUNCTION DOC]
    # Eager only. Skipped when thread sanitizer is enabled (out-of-memory). Builds an endless
    # dataset of 256 MB tensors mapped in parallel and batched by 2, requests one element so the
    # buffer fills, then saves a checkpoint of the iterator via a checkpoint manager in the
    # temp dir. Passes if saving does not fail.
    # KNOWN ISSUE: comment says batches of 512M while the code builds 2 x 256 MB tensors
    # (that is 512 MB total); consistent only if sizes are counted per batch.
    if pywrap_sanitizers.is_tsan_enabled():
      self.skipTest("Creating a large buffer causes OOM when using tsan.")
    # Batches of size 512M
    dataset = dataset_ops.Dataset.from_tensors(
        array_ops.ones((64, 1024, 1024), dtype=dtypes.float32)).repeat()
    dataset = dataset.map(lambda x: x+1, num_parallel_calls=5)
    dataset = dataset.batch(2)
    iterator = iter(dataset)
    next(iterator)  # request an element to fill the buffer
    ckpt = trackable_utils.Checkpoint(iterator=iterator)
    manager = checkpoint_management.CheckpointManager(
        ckpt, self.get_temp_dir(), max_to_keep=1)
    manager.save()


class MapAndBatchCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                                parameterized.TestCase):
  # [CLASS DOC]
  # Checkpoint (save/restore of iterator state) tests for map_and_batch using the shared
  # checkpoint test base and its verification callback.

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(
              drop_remainder=[True, False], symbolic_checkpoint=[True, False])))
  def test(self, verify_fn, drop_remainder, symbolic_checkpoint):
    # [FUNCTION DOC]
    # Parameterized over the verification callback, drop-remainder and symbolic-checkpoint
    # flags. Computes the expected number of batches (range 11 sharded in 3, repeated twice,
    # batch size 5, 7 parallel calls) and runs the callback with a dataset factory.
    range_size = 11
    num_shards = 3
    num_repeats = 2
    batch_size = 5
    num_parallel_calls = 7
    total_outputs = (range_size // num_shards) * num_repeats
    if drop_remainder:
      num_outputs = total_outputs // batch_size
    else:
      num_outputs = int(math.ceil(total_outputs / batch_size))

    def build_ds(range_start, drop_remainder=False, symbolic_checkpoint=False):
      # [FUNCTION DOC]
      # HELPER: builds the sharded, repeated, map_and_batch dataset starting at range_start,
      # with options setting symbolic checkpointing; returns the dataset.

      def _map_fn(x):
        # [FUNCTION DOC]
        # HELPER: squares its input.
        return math_ops.square(x)

      dataset = dataset_ops.Dataset.range(
          range_start, range_start + range_size)
      dataset = dataset.shard(num_shards=num_shards, index=0)
      dataset = dataset.repeat(num_repeats)
      dataset = dataset.apply(
          batching.map_and_batch(
              map_func=_map_fn,
              batch_size=batch_size,
              num_parallel_calls=num_parallel_calls,
              drop_remainder=drop_remainder))
      options = options_lib.Options()
      options.experimental_symbolic_checkpoint = symbolic_checkpoint
      return dataset.with_options(options)

    verify_fn(
        self, lambda: build_ds(
            10,
            drop_remainder=drop_remainder,
            symbolic_checkpoint=symbolic_checkpoint), num_outputs)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(drop_remainder=[True, False])))
  def testNumParallelBatches(self, verify_fn, drop_remainder):
    # [FUNCTION DOC]
    # Same as the main checkpoint test but uses the parallel-batches option (2) instead of
    # parallel calls; runs the verification callback with the expected batch count.
    range_size = 11
    num_shards = 3
    num_repeats = 2
    batch_size = 5
    num_parallel_batches = 2
    total_outputs = (range_size // num_shards) * num_repeats
    if drop_remainder:
      num_outputs = total_outputs // batch_size
    else:
      num_outputs = int(math.ceil(total_outputs / batch_size))

    def build_ds(range_start, drop_remainder):
      # [FUNCTION DOC]
      # HELPER: builds the sharded, repeated range dataset with map_and_batch using parallel
      # batches; returns it.

      def _map_fn(x):
        # [FUNCTION DOC]
        # HELPER: squares its input.
        return math_ops.square(x)

      return dataset_ops.Dataset.range(
          range_start, range_start + range_size).shard(
              num_shards=num_shards, index=0).repeat(num_repeats).apply(
                  batching.map_and_batch(
                      map_func=_map_fn,
                      batch_size=batch_size,
                      num_parallel_batches=num_parallel_batches,
                      drop_remainder=drop_remainder))

    verify_fn(self, lambda: build_ds(10, drop_remainder=drop_remainder),
              num_outputs)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         checkpoint_test_base.default_test_combinations()))
  def testSparse(self, verify_fn):
    # [FUNCTION DOC]
    # Runs the checkpoint verification callback on a dataset mapping integers to sparse tensors
    # and batching by 5, expecting two outputs.

    def build_dataset():
      # [FUNCTION DOC]
      # HELPER: builds range(10) with sparse map_and_batch of 5 and returns it.

      def map_fn(i):
        # [FUNCTION DOC]
        # HELPER: returns a one-element sparse tensor value holding the integer.
        return sparse_tensor.SparseTensorValue(
            indices=[[0]], values=(i * [1]), dense_shape=[1])

      return dataset_ops.Dataset.range(10).apply(
          batching.map_and_batch(map_fn, 5))

    verify_fn(self, build_dataset, num_outputs=2)


if __name__ == "__main__":
  test.main()
