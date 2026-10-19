# [MODULE DOC]
# Tests for the experimental prefetch-to-device dataset transformation, which copies dataset
# elements to a target device (second CPU or GPU) ahead of consumption. Covers plain, dictionary
# and sparse elements, same-device prefetch, iterator re-initialization and device placement
# of the dataset variant. GPU tests skip themselves when no GPU is available. Most tests are
# graph-mode only. No import-time side effects; runs the test main when executed directly.
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
"""Tests for `tf.data.experimental.prefetch_to_device()`."""
from absl.testing import parameterized

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.data.experimental.ops import prefetching_ops
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import structure
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


# TODO(b/117581999): add eager coverage when supported.
class PrefetchToDeviceTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for prefetching dataset elements to a device. Stateless; each test builds its own
  # dataset and (for CPU tests) a session configured with two virtual CPU devices.

  @combinations.generate(test_base.graph_only_combinations())
  def testPrefetchToDevice(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): a range of ten prefetched to the second CPU device. Checks the
    # device dataset structure matches the host's, the element is a scalar 64-bit integer, and
    # that values 0 to 9 are produced in order followed by an out-of-range error.
    host_dataset = dataset_ops.Dataset.range(10)
    device_dataset = host_dataset.apply(
        prefetching_ops.prefetch_to_device("/cpu:1"))

    with ops.device("/cpu:1"):
      iterator = dataset_ops.make_one_shot_iterator(device_dataset)
      next_element = iterator.get_next()

    self.assertTrue(structure.are_compatible(
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
  def testPrefetchToSameDevice(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): same as the basic test but the prefetch target is the host's own
    # first CPU device given by its full name, while the iterator lives on the second CPU.
    # Expects identical structure, dtype, scalar shape and ordered values then end of data.
    host_dataset = dataset_ops.Dataset.range(10)
    device_dataset = host_dataset.apply(
        prefetching_ops.prefetch_to_device(
            "/job:localhost/replica:0/task:0/device:CPU:0"))

    with ops.device("/cpu:1"):
      iterator = dataset_ops.make_one_shot_iterator(device_dataset)
      next_element = iterator.get_next()

    self.assertTrue(structure.are_compatible(
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
  def testPrefetchDictToDevice(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): elements are single-entry dictionaries; prefetched to the second
    # CPU. Expects matching structure, a scalar 64-bit entry, dictionaries with ascending values
    # 0 to 9 and then an out-of-range error.
    host_dataset = dataset_ops.Dataset.range(10).map(lambda x: {"a": x})
    device_dataset = host_dataset.apply(
        prefetching_ops.prefetch_to_device("/cpu:1"))

    with ops.device("/cpu:1"):
      iterator = dataset_ops.make_one_shot_iterator(device_dataset)
      next_element = iterator.get_next()

    self.assertTrue(structure.are_compatible(
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
  def testPrefetchSparseTensorsToDevice(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): elements are 2x2 sparse tensor values with one entry holding the
    # counter; prefetched to the second CPU. Expects matching structure, 64-bit dtype, and for
    # each element the expected values, indices and dense shape, then an out-of-range error.
    def make_tensor(i):
      # [FUNCTION DOC]
      # Builds a sparse tensor value with a single entry at row 0, column 0 equal to i in a
      # 2x2 dense shape.
      return sparse_tensor.SparseTensorValue(
          indices=[[0, 0]], values=(i*[1]), dense_shape=[2, 2])
    host_dataset = dataset_ops.Dataset.range(10).map(make_tensor)

    device_dataset = host_dataset.apply(
        prefetching_ops.prefetch_to_device("/cpu:1"))

    with ops.device("/cpu:1"):
      iterator = dataset_ops.make_one_shot_iterator(device_dataset)
      next_element = iterator.get_next()

    self.assertTrue(structure.are_compatible(
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
  def testPrefetchToDeviceGpu(self):
    # [FUNCTION DOC]
    # Scenario: skipped when no GPU is available. Prefetches a range of ten to the first GPU
    # and asserts the dataset produces 0 to 9.
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    host_dataset = dataset_ops.Dataset.range(10)
    device_dataset = host_dataset.apply(
        prefetching_ops.prefetch_to_device("/gpu:0"))

    self.assertDatasetProduces(device_dataset, list(range(10)))

  @combinations.generate(test_base.default_test_combinations())
  def testPrefetchToDeviceCorrectPlacement(self):
    # [FUNCTION DOC]
    # Scenario: skipped when no GPU is available. Prefetches a range to the first GPU and
    # asserts the dataset's variant tensor device name mentions the GPU.

    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.apply(prefetching_ops.prefetch_to_device("/gpu:0"))

    self.assertIn("gpu:0", dataset._variant_tensor.device.lower())

  @combinations.generate(test_base.graph_only_combinations())
  def testPrefetchToDeviceWithReInit(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): initializable iterator over a dataset prefetched to the second
    # CPU. Initializes, reads five elements, re-initializes, then reads all ten in order and
    # expects out-of-range at the end; also checks structure, dtype and scalar shape.
    host_dataset = dataset_ops.Dataset.range(10)
    device_dataset = host_dataset.apply(
        prefetching_ops.prefetch_to_device("/cpu:1"))

    with ops.device("/cpu:1"):
      iterator = dataset_ops.make_initializable_iterator(device_dataset)
      next_element = iterator.get_next()

    self.assertTrue(structure.are_compatible(
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
  def testPrefetchToDeviceGpuWithReInit(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): skipped when no GPU is available. Same re-initialization pattern
    # as the CPU variant but prefetching to the first GPU, in a session with soft placement
    # disabled. Expects five elements, re-init, ten elements, then out-of-range.
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    host_dataset = dataset_ops.Dataset.range(10)
    device_dataset = host_dataset.apply(
        prefetching_ops.prefetch_to_device("/gpu:0"))

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

  @combinations.generate(test_base.eager_only_combinations())
  def testPrefetchToDevicePlacement(self):
    # [FUNCTION DOC]
    # Scenario (eager mode): skipped when no GPU is available. Asserts the prefetched dataset's
    # variant tensor lives on the fully qualified first GPU device name.
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    host_dataset = dataset_ops.Dataset.range(10)
    device_dataset = host_dataset.apply(
        prefetching_ops.prefetch_to_device("/gpu:0"))

    self.assertEqual(device_dataset._variant_tensor.device,
                     "/job:localhost/replica:0/task:0/device:GPU:0")


if __name__ == "__main__":
  test.main()
