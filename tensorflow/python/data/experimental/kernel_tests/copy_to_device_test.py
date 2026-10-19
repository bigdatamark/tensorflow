# [MODULE DOC]
# Kernel tests for the experimental copy-to-device dataset transformation. Contains one test
# class that moves datasets between CPU and GPU devices (with and without prefetch,
# re-initialization, mapping and sparse structures, strings, int32, GPU map) and checks
# structure, dtype, shape and values. GPU tests skip themselves when no GPU is available.
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
"""Tests for `tf.data.experimental.copy_to_device()`."""
from absl.testing import parameterized

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.data.experimental.ops import prefetching_ops
from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.util import structure
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.util import compat as util_compat


# TODO(b/117581999): add eager coverage when supported.
class CopyToDeviceTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the copy-to-device transformation. Most tests are graph-mode only (eager
  # coverage is a pending to-do) and use a two-CPU session configuration; GPU tests are skipped
  # when no GPU exists.

  @combinations.generate(test_base.graph_only_combinations())
  def testCopyToDevice(self):
    # [FUNCTION DOC]
    # Test: copies a range dataset of 10 to the second CPU device, checks the structure is
    # compatible, dtype is int64 and shape scalar, and that all 10 values arrive in order
    # followed by an out-of-range error.
    host_dataset = dataset_ops.Dataset.range(10)
    device_dataset = host_dataset.apply(
        prefetching_ops.copy_to_device("/cpu:1"))

    with ops.device("/cpu:1"):
      iterator = dataset_ops.make_one_shot_iterator(device_dataset)
      next_element = iterator.get_next()

    self.assertTrue(
        structure.are_compatible(
            dataset_ops.get_structure(host_dataset),
            dataset_ops.get_structure(device_dataset)))

    self.assertEqual(dtypes.int64, next_element.dtype)
    self.assertEqual([], next_element.shape)

    worker_config = config_pb2.ConfigProto(device_count={"CPU": 2})
    with self.test_session(config=worker_config):
      for i in range(10):
        self.assertEqual(i, self.evaluate(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_element)

  @combinations.generate(test_base.graph_only_combinations())
  def testCopyToDeviceHostOptimizations(self):
    # [FUNCTION DOC]
    # Test: the host dataset asserts its next transformation is a fused map-and-batch, maps
    # squaring and batches 10, then is copied to the second CPU. Checks the batch of squares is
    # produced and then out-of-range, showing host-side optimizations still apply.
    host_dataset = dataset_ops.Dataset.range(10)
    host_dataset = host_dataset.apply(testing.assert_next(["MapAndBatch"]))
    host_dataset = host_dataset.map(lambda x: x*x).batch(10)
    device_dataset = host_dataset.apply(
        prefetching_ops.copy_to_device("/cpu:1"))

    with ops.device("/cpu:1"):
      iterator = dataset_ops.make_one_shot_iterator(device_dataset)
      next_element = iterator.get_next()

    worker_config = config_pb2.ConfigProto(device_count={"CPU": 2})
    with self.test_session(config=worker_config):
      self.assertAllEqual([x*x for x in range(10)], self.evaluate(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_element)

  @combinations.generate(test_base.graph_only_combinations())
  def testCopyToDeviceInt32(self):
    # [FUNCTION DOC]
    # Test: copies a single int32 vector element to the second CPU; checks structure
    # compatibility, int32 dtype, shape of four, the values, then out-of-range.
    host_dataset = dataset_ops.Dataset.from_tensors([0, 1, 2, 3])
    device_dataset = host_dataset.apply(
        prefetching_ops.copy_to_device("/cpu:1"))

    with ops.device("/cpu:1"):
      iterator = dataset_ops.make_one_shot_iterator(device_dataset)
      next_element = iterator.get_next()

    self.assertTrue(
        structure.are_compatible(
            dataset_ops.get_structure(host_dataset),
            dataset_ops.get_structure(device_dataset)))

    self.assertEqual(dtypes.int32, next_element.dtype)
    self.assertEqual((4,), next_element.shape)

    worker_config = config_pb2.ConfigProto(device_count={"CPU": 2})
    with self.test_session(config=worker_config):
      self.assertAllEqual([0, 1, 2, 3], self.evaluate(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_element)

  @combinations.generate(test_base.graph_only_combinations())
  def testCopyToSameDevice(self):
    # [FUNCTION DOC]
    # Test: copying a range dataset to the same device it lives on (first CPU) still works;
    # checks compatibility, dtype, shape and the ten values followed by out-of-range.
    host_dataset = dataset_ops.Dataset.range(10)
    device_dataset = host_dataset.apply(
        prefetching_ops.copy_to_device("/cpu:0"))

    with ops.device("/cpu:0"):
      iterator = dataset_ops.make_one_shot_iterator(device_dataset)
      next_element = iterator.get_next()

    self.assertTrue(
        structure.are_compatible(
            dataset_ops.get_structure(host_dataset),
            dataset_ops.get_structure(device_dataset)))

    self.assertEqual(dtypes.int64, next_element.dtype)
    self.assertEqual([], next_element.shape)

    worker_config = config_pb2.ConfigProto(device_count={"CPU": 2})
    with self.test_session(config=worker_config):
      for i in range(10):
        self.assertEqual(i, self.evaluate(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_element)

  @combinations.generate(test_base.graph_only_combinations())
  def testCopyToDeviceWithPrefetch(self):
    # [FUNCTION DOC]
    # Test: same as the basic copy test but with a prefetch of one element after the copy to the
    # second CPU; checks structure, dtype, shape, values and out-of-range.
    host_dataset = dataset_ops.Dataset.range(10)
    device_dataset = host_dataset.apply(
        prefetching_ops.copy_to_device("/cpu:1")).prefetch(1)

    with ops.device("/cpu:1"):
      iterator = dataset_ops.make_one_shot_iterator(device_dataset)
      next_element = iterator.get_next()

    self.assertTrue(
        structure.are_compatible(
            dataset_ops.get_structure(host_dataset),
            dataset_ops.get_structure(device_dataset)))

    self.assertEqual(dtypes.int64, next_element.dtype)
    self.assertEqual([], next_element.shape)

    worker_config = config_pb2.ConfigProto(device_count={"CPU": 2})
    with self.test_session(config=worker_config):
      for i in range(10):
        self.assertEqual(i, self.evaluate(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_element)

  @combinations.generate(test_base.graph_only_combinations())
  def testCopyDictToDevice(self):
    # [FUNCTION DOC]
    # Test: copies a dataset of single-key mappings to the second CPU; checks the structure,
    # int64 dtype and scalar shape of the entry, the ten mapping values, then out-of-range.
    host_dataset = dataset_ops.Dataset.range(10).map(lambda x: {"a": x})
    device_dataset = host_dataset.apply(
        prefetching_ops.copy_to_device("/cpu:1"))

    with ops.device("/cpu:1"):
      iterator = dataset_ops.make_one_shot_iterator(device_dataset)
      next_element = iterator.get_next()

    self.assertTrue(
        structure.are_compatible(
            dataset_ops.get_structure(host_dataset),
            dataset_ops.get_structure(device_dataset)))

    self.assertEqual(dtypes.int64, next_element["a"].dtype)
    self.assertEqual([], next_element["a"].shape)

    worker_config = config_pb2.ConfigProto(device_count={"CPU": 2})
    with self.test_session(config=worker_config):
      for i in range(10):
        self.assertEqual({"a": i}, self.evaluate(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_element)

  @combinations.generate(test_base.graph_only_combinations())
  def testCopyDictToDeviceWithPrefetch(self):
    # [FUNCTION DOC]
    # Test: like the mapping copy test but with a prefetch of one element after the copy.
    host_dataset = dataset_ops.Dataset.range(10).map(lambda x: {"a": x})
    device_dataset = host_dataset.apply(
        prefetching_ops.copy_to_device("/cpu:1")).prefetch(1)

    with ops.device("/cpu:1"):
      iterator = dataset_ops.make_one_shot_iterator(device_dataset)
      next_element = iterator.get_next()

    self.assertTrue(
        structure.are_compatible(
            dataset_ops.get_structure(host_dataset),
            dataset_ops.get_structure(device_dataset)))

    self.assertEqual(dtypes.int64, next_element["a"].dtype)
    self.assertEqual([], next_element["a"].shape)

    worker_config = config_pb2.ConfigProto(device_count={"CPU": 2})
    with self.test_session(config=worker_config):
      for i in range(10):
        self.assertEqual({"a": i}, self.evaluate(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_element)

  @combinations.generate(test_base.graph_only_combinations())
  def testCopySparseTensorsToDevice(self):
    # [FUNCTION DOC]
    # Test: maps each of ten integers to a 2x2 sparse value with one entry, copies the dataset
    # to the second CPU, and checks each element's values, indices and dense shape, then
    # out-of-range.

    def make_tensor(i):
      # [FUNCTION DOC]
      # HELPER: builds a sparse tensor value with one entry at (0, 0) holding the given integer
      # and dense shape 2x2.
      return sparse_tensor.SparseTensorValue(
          indices=[[0, 0]], values=(i * [1]), dense_shape=[2, 2])

    host_dataset = dataset_ops.Dataset.range(10).map(make_tensor)

    device_dataset = host_dataset.apply(
        prefetching_ops.copy_to_device("/cpu:1"))

    with ops.device("/cpu:1"):
      iterator = dataset_ops.make_one_shot_iterator(device_dataset)
      next_element = iterator.get_next()

    self.assertTrue(
        structure.are_compatible(
            dataset_ops.get_structure(host_dataset),
            dataset_ops.get_structure(device_dataset)))

    self.assertEqual(dtypes.int64, next_element.dtype)

    worker_config = config_pb2.ConfigProto(device_count={"CPU": 2})
    with self.test_session(config=worker_config):
      for i in range(10):
        actual = self.evaluate(next_element)
        self.assertAllEqual([i], actual.values)
        self.assertAllEqual([[0, 0]], actual.indices)
        self.assertAllEqual([2, 2], actual.dense_shape)
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_element)

  @combinations.generate(test_base.graph_only_combinations())
  def testCopySparseTensorsToDeviceWithPrefetch(self):
    # [FUNCTION DOC]
    # Test: like the sparse copy test but with a prefetch of one element after the copy.

    def make_tensor(i):
      # [FUNCTION DOC]
      # HELPER: builds a sparse tensor value with one entry at (0, 0) holding the given integer
      # and dense shape 2x2.
      return sparse_tensor.SparseTensorValue(
          indices=[[0, 0]], values=(i * [1]), dense_shape=[2, 2])

    host_dataset = dataset_ops.Dataset.range(10).map(make_tensor)

    device_dataset = host_dataset.apply(
        prefetching_ops.copy_to_device("/cpu:1")).prefetch(1)

    with ops.device("/cpu:1"):
      iterator = dataset_ops.make_one_shot_iterator(device_dataset)
      next_element = iterator.get_next()

    self.assertTrue(
        structure.are_compatible(
            dataset_ops.get_structure(host_dataset),
            dataset_ops.get_structure(device_dataset)))

    self.assertEqual(dtypes.int64, next_element.dtype)

    worker_config = config_pb2.ConfigProto(device_count={"CPU": 2})
    with self.test_session(config=worker_config):
      for i in range(10):
        actual = self.evaluate(next_element)
        self.assertAllEqual([i], actual.values)
        self.assertAllEqual([[0, 0]], actual.indices)
        self.assertAllEqual([2, 2], actual.dense_shape)
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_element)

  @combinations.generate(test_base.default_test_combinations())
  def testCopyToDeviceGpu(self):
    # [FUNCTION DOC]
    # Test: skipped without a GPU. Copies a range dataset to the GPU and asserts it produces
    # the values zero through nine.
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    host_dataset = dataset_ops.Dataset.range(10)
    device_dataset = host_dataset.apply(
        prefetching_ops.copy_to_device("/gpu:0"))

    with ops.device("/gpu:0"):
      self.assertDatasetProduces(device_dataset, list(range(10)))

  @combinations.generate(test_base.default_test_combinations())
  def testCopyToDeviceGpuWithPrefetch(self):
    # [FUNCTION DOC]
    # Test: skipped without a GPU. Like the GPU copy test but with a prefetch of one element.
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    host_dataset = dataset_ops.Dataset.range(10)
    device_dataset = host_dataset.apply(
        prefetching_ops.copy_to_device("/gpu:0")).prefetch(1)

    with ops.device("/gpu:0"):
      self.assertDatasetProduces(device_dataset, list(range(10)))

  @combinations.generate(test_base.graph_only_combinations())
  def testCopyToDeviceGpuWithMap(self):
    # [FUNCTION DOC]
    # Test: skipped without a GPU. A generator dataset of (int, float, string) triples is copied
    # to the GPU, mapped on the GPU by squaring the first two components (autotune disabled),
    # and consumed with soft placement off. Asserts squared int, squared float and the
    # unchanged string bytes for ten elements, then out-of-range.
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    def generator():
      # [FUNCTION DOC]
      # HELPER: generator yielding ten triples of an integer, its float value and its string form.
      for i in range(10):
        yield i, float(i), str(i)

    host_dataset = dataset_ops.Dataset.from_generator(
        generator, output_types=(dtypes.int32, dtypes.float32, dtypes.string))
    device_dataset = host_dataset.apply(
        prefetching_ops.copy_to_device("/gpu:0"))

    def gpu_map_func(x, y, z):
      # [FUNCTION DOC]
      # HELPER: map function run on the GPU; returns the square of the first two components and
      # the third (string) component unchanged.
      return math_ops.square(x), math_ops.square(y), z

    device_dataset = device_dataset.apply(
        prefetching_ops.map_on_gpu(gpu_map_func))
    options = options_lib.Options()
    options.autotune.enabled = False
    device_dataset = device_dataset.with_options(options)

    with ops.device("/gpu:0"):
      iterator = dataset_ops.make_initializable_iterator(device_dataset)
      next_element = iterator.get_next()

    with self.cached_session(
        config=config_pb2.ConfigProto(allow_soft_placement=False)):
      self.evaluate(iterator.initializer)
      for i in range(10):
        x, y, z = self.evaluate(next_element)
        self.assertEqual(i**2, x)
        self.assertEqual(float(i**2), y)
        self.assertEqual(util_compat.as_bytes(str(i)), z)
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_element)

  @combinations.generate(test_base.graph_only_combinations())
  def testCopyToDeviceGpuInt32(self):
    # [FUNCTION DOC]
    # Test: skipped without a GPU. Copies an int32 vector element to the GPU and reads it with
    # an initializable iterator; asserts the values and then out-of-range.
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    host_dataset = dataset_ops.Dataset.from_tensors([0, 1, 2, 3])
    device_dataset = host_dataset.apply(
        prefetching_ops.copy_to_device("/gpu:0"))

    with ops.device("/gpu:0"):
      iterator = dataset_ops.make_initializable_iterator(device_dataset)
      next_element = iterator.get_next()

    with self.cached_session(
        config=config_pb2.ConfigProto(allow_soft_placement=False)):
      self.evaluate(iterator.initializer)
      self.assertAllEqual([0, 1, 2, 3], self.evaluate(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_element)

  @combinations.generate(test_base.graph_only_combinations())
  def testCopyToDeviceGpuInt32AndPrefetch(self):
    # [FUNCTION DOC]
    # Test: skipped without a GPU. Same as the GPU int32 test with a prefetch of one element.
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    host_dataset = dataset_ops.Dataset.from_tensors([0, 1, 2, 3])
    device_dataset = host_dataset.apply(
        prefetching_ops.copy_to_device("/gpu:0")).prefetch(1)

    with ops.device("/gpu:0"):
      iterator = dataset_ops.make_initializable_iterator(device_dataset)
      next_element = iterator.get_next()

    with self.cached_session(
        config=config_pb2.ConfigProto(allow_soft_placement=False)):
      self.evaluate(iterator.initializer)
      self.assertAllEqual([0, 1, 2, 3], self.evaluate(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_element)

  @combinations.generate(test_base.graph_only_combinations())
  def testCopyToDeviceGpuStrings(self):
    # [FUNCTION DOC]
    # Test: skipped without a GPU. Copies a vector of three strings to the GPU; asserts the
    # byte-string values come back and then out-of-range.
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    host_dataset = dataset_ops.Dataset.from_tensors(["a", "b", "c"])
    device_dataset = host_dataset.apply(
        prefetching_ops.copy_to_device("/gpu:0"))

    with ops.device("/gpu:0"):
      iterator = dataset_ops.make_initializable_iterator(device_dataset)
      next_element = iterator.get_next()

    with self.cached_session(
        config=config_pb2.ConfigProto(allow_soft_placement=False)):
      self.evaluate(iterator.initializer)
      self.assertAllEqual([b"a", b"b", b"c"], self.evaluate(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_element)

  @combinations.generate(test_base.graph_only_combinations())
  def testCopyToDeviceGpuStringsAndPrefetch(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: despite the name, the dataset is not given a prefetch; the body is identical
    # to the plain GPU strings test.
    # Test: skipped without a GPU. Copies a vector of three strings to the GPU; asserts the
    # byte-string values and then out-of-range.
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    host_dataset = dataset_ops.Dataset.from_tensors(["a", "b", "c"])
    device_dataset = host_dataset.apply(
        prefetching_ops.copy_to_device("/gpu:0"))

    with ops.device("/gpu:0"):
      iterator = dataset_ops.make_initializable_iterator(device_dataset)
      next_element = iterator.get_next()

    with self.cached_session(
        config=config_pb2.ConfigProto(allow_soft_placement=False)):
      self.evaluate(iterator.initializer)
      self.assertAllEqual([b"a", b"b", b"c"], self.evaluate(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_element)

  @combinations.generate(test_base.graph_only_combinations())
  def testCopyToDevicePingPongCPUGPU(self):
    # [FUNCTION DOC]
    # Test: skipped without a GPU. Copies a range dataset from CPU to GPU and back to CPU, then
    # asserts the ten values arrive in order followed by out-of-range.
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    host_dataset = dataset_ops.Dataset.range(10)
    device_dataset = host_dataset.apply(
        prefetching_ops.copy_to_device("/gpu:0", source_device="/cpu:0"))
    back_to_cpu_dataset = device_dataset.apply(
        prefetching_ops.copy_to_device("/cpu:0", source_device="/gpu:0"))

    with ops.device("/cpu:0"):
      iterator = dataset_ops.make_initializable_iterator(back_to_cpu_dataset)
      next_element = iterator.get_next()

    with self.cached_session(
        config=config_pb2.ConfigProto(allow_soft_placement=False)):
      self.evaluate(iterator.initializer)
      for i in range(10):
        self.assertEqual(i, self.evaluate(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_element)

  @combinations.generate(test_base.graph_only_combinations())
  def testCopyToDeviceWithReInit(self):
    # [FUNCTION DOC]
    # Test: copies a range dataset to the second CPU with an initializable iterator, reads five
    # values, re-initializes, then reads all ten values and out-of-range.
    host_dataset = dataset_ops.Dataset.range(10)
    device_dataset = host_dataset.apply(
        prefetching_ops.copy_to_device("/cpu:1"))

    with ops.device("/cpu:1"):
      iterator = dataset_ops.make_initializable_iterator(device_dataset)
      next_element = iterator.get_next()

    self.assertTrue(
        structure.are_compatible(
            dataset_ops.get_structure(host_dataset),
            dataset_ops.get_structure(device_dataset)))

    self.assertEqual(dtypes.int64, next_element.dtype)
    self.assertEqual([], next_element.shape)

    worker_config = config_pb2.ConfigProto(device_count={"CPU": 2})
    with self.test_session(config=worker_config):
      self.evaluate(iterator.initializer)
      for i in range(5):
        self.assertEqual(i, self.evaluate(next_element))
      self.evaluate(iterator.initializer)
      for i in range(10):
        self.assertEqual(i, self.evaluate(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_element)

  @combinations.generate(test_base.graph_only_combinations())
  def testCopyToDeviceWithReInitAndPrefetch(self):
    # [FUNCTION DOC]
    # Test: same as the re-initialization test with a prefetch of one element after the copy.
    host_dataset = dataset_ops.Dataset.range(10)
    device_dataset = host_dataset.apply(
        prefetching_ops.copy_to_device("/cpu:1")).prefetch(1)

    with ops.device("/cpu:1"):
      iterator = dataset_ops.make_initializable_iterator(device_dataset)
      next_element = iterator.get_next()

    self.assertTrue(
        structure.are_compatible(
            dataset_ops.get_structure(host_dataset),
            dataset_ops.get_structure(device_dataset)))

    self.assertEqual(dtypes.int64, next_element.dtype)
    self.assertEqual([], next_element.shape)

    worker_config = config_pb2.ConfigProto(device_count={"CPU": 2})
    with self.test_session(config=worker_config):
      self.evaluate(iterator.initializer)
      for i in range(5):
        self.assertEqual(i, self.evaluate(next_element))
      self.evaluate(iterator.initializer)
      for i in range(10):
        self.assertEqual(i, self.evaluate(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_element)

  @combinations.generate(test_base.graph_only_combinations())
  def testCopyToDeviceGpuWithReInit(self):
    # [FUNCTION DOC]
    # Test: skipped without a GPU. Copy to GPU, read five values, re-initialize the iterator,
    # then read all ten values and out-of-range.
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    host_dataset = dataset_ops.Dataset.range(10)
    device_dataset = host_dataset.apply(
        prefetching_ops.copy_to_device("/gpu:0"))

    with ops.device("/gpu:0"):
      iterator = dataset_ops.make_initializable_iterator(device_dataset)
      next_element = iterator.get_next()

    with self.cached_session(
        config=config_pb2.ConfigProto(allow_soft_placement=False)):
      self.evaluate(iterator.initializer)
      for i in range(5):
        self.assertEqual(i, self.evaluate(next_element))
      self.evaluate(iterator.initializer)
      for i in range(10):
        self.assertEqual(i, self.evaluate(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_element)

  @combinations.generate(test_base.graph_only_combinations())
  def testCopyToDeviceGpuWithReInitAndPrefetch(self):
    # [FUNCTION DOC]
    # Test: skipped without a GPU. Same as the GPU re-initialization test with a prefetch of one.
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    host_dataset = dataset_ops.Dataset.range(10)
    device_dataset = host_dataset.apply(
        prefetching_ops.copy_to_device("/gpu:0")).prefetch(1)

    with ops.device("/gpu:0"):
      iterator = dataset_ops.make_initializable_iterator(device_dataset)
      next_element = iterator.get_next()

    with self.cached_session(
        config=config_pb2.ConfigProto(allow_soft_placement=False)):
      self.evaluate(iterator.initializer)
      for i in range(5):
        self.assertEqual(i, self.evaluate(next_element))
      self.evaluate(iterator.initializer)
      for i in range(10):
        self.assertEqual(i, self.evaluate(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_element)

  @combinations.generate(test_base.graph_only_combinations())
  def testIteratorGetNextAsOptionalOnGPU(self):
    # [FUNCTION DOC]
    # Test: skipped without a GPU. Reads a three-element GPU dataset through get-next-as-optional.
    # Before initialization, evaluating the optional's has-value or value fails with a
    # failed-precondition error. After initialization each of three elements reports a value
    # equal to its index; after exhaustion has-value is false and getting the value raises an
    # invalid-argument error (checked twice).
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    host_dataset = dataset_ops.Dataset.range(3)
    device_dataset = host_dataset.apply(
        prefetching_ops.copy_to_device("/gpu:0"))
    with ops.device("/gpu:0"):
      iterator = dataset_ops.make_initializable_iterator(device_dataset)
      next_elem = iterator_ops.get_next_as_optional(iterator)
      elem_has_value_t = next_elem.has_value()
      elem_value_t = next_elem.get_value()

    with self.cached_session(
        config=config_pb2.ConfigProto(allow_soft_placement=False)):
      # Before initializing the iterator, evaluating the optional fails with
      # a FailedPreconditionError.
      with self.assertRaises(errors.FailedPreconditionError):
        self.evaluate(elem_has_value_t)
      with self.assertRaises(errors.FailedPreconditionError):
        self.evaluate(elem_value_t)

      # For each element of the dataset, assert that the optional evaluates to
      # the expected value.
      self.evaluate(iterator.initializer)
      for i in range(3):
        elem_has_value, elem_value = self.evaluate(
            [elem_has_value_t, elem_value_t])
        self.assertTrue(elem_has_value)
        self.assertEqual(i, elem_value)

      # After exhausting the iterator, `next_elem.has_value()` will evaluate to
      # false, and attempting to get the value will fail.
      for _ in range(2):
        self.assertFalse(self.evaluate(elem_has_value_t))
        with self.assertRaises(errors.InvalidArgumentError):
          self.evaluate(elem_value_t)


if __name__ == "__main__":
  test.main()
