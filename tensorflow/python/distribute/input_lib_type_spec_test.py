# [MODULE DOC]
# Tests for the type specifications (TypeSpec) of distributed iterators, distributed datasets and
# datasets created from a function. Covers structure checks, component round trips,
# function-tracing behavior, input signatures built from element specs, and most-specific
# compatible type computation, across several distribution strategies (mirrored, TPU, central
# storage, multi-worker) and with dense, ragged and sparse inputs. Defines two small private
# helpers (structure check and text file creation) and runs the distribute test main entry point
# when executed directly. No import-time side effects beyond imports.

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
"""Tests for the input_lib library which tests iterator type specs."""

import os

from absl.testing import parameterized
import numpy as np

from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import test_util
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.distribute import values
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_tensor as ragged_tensor_lib
from tensorflow.python.util import nest


class DistributedIteratorTest(test.TestCase,
                              parameterized.TestCase):
  # [CLASS DOC]
  # Tests the type spec of distributed iterators (built from a distributed dataset) under
  # mirrored and TPU strategies, with and without the optional get-next behavior. Parameterized
  # through the distribute combinations framework; has no shared fixtures.

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          input_type=["dataset"],
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
          ],
          enable_get_next_as_optional=[True, False]))
  def testTypeSpec(self, input_type, distribution,
                   enable_get_next_as_optional):
    # [FUNCTION DOC]
    # Scenario: a range dataset batched by two is distributed and an iterator is created inside
    # the strategy scope. Skipped when TF2 behavior is not enabled. Checks that the iterator and
    # its type spec share the same structure, that the spec's input workers equal the iterator's,
    # and that the spec's element spec holds two int64 tensor specs of unknown batch dimension
    # (one per replica). Parameters: input type (unused), strategy, and the get-next-as-optional
    # flag set on the strategy extended object (a side effect on the strategy).
    if not tf2.enabled():
      self.skipTest("DistributedIterator has CompositeTensor support in "
                    "TF 2 only.")
    dataset = dataset_ops.DatasetV2.range(10).batch(2)

    distribution.extended.experimental_enable_get_next_as_optional = (
        enable_get_next_as_optional)

    dist_dataset = distribution.experimental_distribute_dataset(dataset)
    with distribution.scope():
      iterator = iter(dist_dataset)
      _check_type_spec_structure(iterator)

    spec = iterator._type_spec
    self.assertEqual(spec._input_workers, iterator._input_workers)
    self.assertEqual(spec._element_spec._value_specs,
                     (tensor_spec.TensorSpec(shape=(None,), dtype=dtypes.int64,
                                             name=None),
                      tensor_spec.TensorSpec(shape=(None,), dtype=dtypes.int64,
                                             name=None)))

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          input_type=["dataset"],
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
          ],
          enable_get_next_as_optional=[True, False]))
  def testTypeSpecRoundTrip(self, input_type,
                            distribution, enable_get_next_as_optional):
    # [FUNCTION DOC]
    # Scenario: distribute a batched range dataset, create an iterator, convert it to components
    # through its type spec and rebuild an iterator from them. Skipped when TF2 behavior is off.
    # Asserts the rebuilt iterator has equal input workers and equal underlying per-device
    # iterators. Sets the get-next-as-optional flag on the strategy.
    if not tf2.enabled():
      self.skipTest("DistributedIterator CompositeTensor support is only "
                    "present in TF 2.0 only.")

    dataset = dataset_ops.DatasetV2.range(10).batch(2)

    distribution.extended.experimental_enable_get_next_as_optional = (
        enable_get_next_as_optional)

    dist_dataset = distribution.experimental_distribute_dataset(dataset)
    with distribution.scope():
      iterator = iter(dist_dataset)
      _check_type_spec_structure(iterator)

    spec = iterator._type_spec

    tensor_list = spec._to_components(iterator)
    re_iterator = spec._from_components(tensor_list)

    self.assertEqual(iterator._input_workers, re_iterator._input_workers)
    self.assertAllEqual(iterator._iterators, re_iterator._iterators)

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          input_type=["dataset"],
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
          ],
          enable_get_next_as_optional=[True, False],
          drop_remainder=[True, False],
          tf_api_version=2,
      ))
  def testDoesNotTriggerFunctionTracing(self, input_type, distribution,
                                        enable_get_next_as_optional,
                                        drop_remainder):
    # [FUNCTION DOC]
    # Scenario: a compiled function consumes five elements from a distributed iterator; it is
    # called with a freshly created iterator three times. Asserts the function body is traced only
    # once (trace counter stays at one, so iterator type specs are stable) and that it returns
    # five each time. Parameterized over the get-next-as-optional flag and drop-remainder.
    trace_count = [0]

    @def_function.function
    def f(iterator):
      # [FUNCTION DOC]
      # Nested compiled function: increments the outer trace counter each time it is traced,
      # takes five elements from the given iterator, and returns the count of elements taken.
      trace_count[0] += 1
      counter = np.int64(0)
      for _ in range(5):
        next(iterator)
        counter += 1
      return counter

    dataset = dataset_ops.DatasetV2.range(10).batch(
        2, drop_remainder=drop_remainder)

    distribution.extended.experimental_enable_get_next_as_optional = (
        enable_get_next_as_optional)

    dist_dataset = distribution.experimental_distribute_dataset(dataset)
    with distribution.scope():
      for _ in range(3):
        iterator = iter(dist_dataset)
        _check_type_spec_structure(iterator)
        counter = f(iterator)

        self.assertEqual(trace_count[0], 1)
        self.assertEqual(counter, 5)


class InputTypeSpecTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests input signatures and type spec compatibility for distributed datasets and iterators
  # across many strategies, including datasets created from a function with input options.

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          distribution=[
              strategy_combinations.one_device_strategy,
              strategy_combinations.mirrored_strategy_with_one_cpu,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
              strategy_combinations.central_storage_strategy_with_gpu_and_cpu,
              strategy_combinations.multi_worker_mirrored_2x1_cpu,
              strategy_combinations.multi_worker_mirrored_2x1_gpu,
              strategy_combinations.multi_worker_mirrored_2x2_gpu,
          ],
          tf_api_version=2,
          enable_get_next_as_optional=[True, False],
          drop_remainder=[True, False],
      ))
  def testInputSignatureForPerReplicaValues(self, distribution,
                                            enable_get_next_as_optional,
                                            drop_remainder):
    # [FUNCTION DOC]
    # Scenario: a dataset of nine float rows batched by four (optionally dropping the remainder)
    # is distributed. Checks the iterator type spec structure, that the dataset's element spec and
    # the iterator's element spec have the same structure and equal flattened contents, then
    # defines a compiled function with that element spec as input signature that runs an identity
    # on the strategy, and calls it for every element of the distributed dataset (must not fail
    # on signature mismatch).
    distribution.extended.experimental_enable_get_next_as_optional = (
        enable_get_next_as_optional)
    ds = dataset_ops.DatasetV2.from_tensor_slices(
        np.ones([9, 12]).astype(np.float32)).batch(
            4, drop_remainder=drop_remainder)
    ds = distribution.experimental_distribute_dataset(ds)
    _check_type_spec_structure(iter(ds))
    element_spec = ds.element_spec
    iter_element_spec = iter(ds).element_spec
    nest.assert_same_structure(element_spec, iter_element_spec)
    self.assertAllEqual(
        nest.flatten(element_spec), nest.flatten(iter_element_spec))

    @def_function.function(input_signature=[element_spec])
    def process_inputs(inputs):
      # [FUNCTION DOC]
      # Nested compiled function with a fixed input signature: runs an identity function on every
      # replica of the strategy with the given per-replica inputs. Returns nothing.
      distribution.run(lambda inputs: inputs, args=(inputs,))

    for x in ds:
      process_inputs(x)

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          distribution=[
              strategy_combinations.one_device_strategy,
              strategy_combinations.mirrored_strategy_with_one_cpu,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
              strategy_combinations.central_storage_strategy_with_two_gpus,
          ],
      ))
  def testInputSignatureForNestedPerReplicaValues(self, distribution):
    # [FUNCTION DOC]
    # Scenario: a dataset of tuples of two float arrays (ten rows each), batched by two, is
    # distributed. A compiled function whose input signature is the distributed dataset's element
    # spec runs an identity on the strategy and is called on every element. Expects no signature
    # mismatch errors for nested per-replica values.
    a = np.ones((10, 2)) * 5
    b = np.ones((10, 3)) * 6
    dataset = dataset_ops.DatasetV2.from_tensor_slices((a, b)).batch(2)

    dist_dataset = distribution.experimental_distribute_dataset(dataset)

    @def_function.function(input_signature=[dist_dataset.element_spec])
    def process_inputs(inputs):
      # [FUNCTION DOC]
      # Nested compiled function with a fixed input signature: runs an identity function on every
      # replica with the given nested per-replica inputs. Returns nothing.
      distribution.run(lambda inputs: inputs, args=(inputs,))

    for x in dist_dataset:
      process_inputs(x)

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          input_type=["dataset"],
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
          ],
          enable_get_next_as_optional=[True, False]))
  def testMostSpecificCompatibleType(self, input_type, distribution,
                                     enable_get_next_as_optional):
    # [FUNCTION DOC]
    # Scenario: two datasets with different element shapes (one with unknown inner dimension, one
    # with a known inner dimension of two) are distributed and iterated. Skipped when TF2 behavior
    # is off. Asserts the two iterator type specs differ, and that the most specific compatible
    # type of either spec with the other is the more general (first) spec, in both directions.
    if not tf2.enabled():
      self.skipTest("DistributedIterator has CompositeTensor support in "
                    "TF 2 only.")
    distribution.extended.experimental_enable_get_next_as_optional = (
        enable_get_next_as_optional)

    ds1 = dataset_ops.DatasetV2.range(10).batch(2).batch(5)
    ds2 = dataset_ops.DatasetV2.from_tensors(
        array_ops.zeros([5, 2], dtypes.int64))
    dist_ds1 = distribution.experimental_distribute_dataset(ds1)
    dist_ds2 = distribution.experimental_distribute_dataset(ds2)

    with distribution.scope():
      iter1 = iter(dist_ds1)
      iter2 = iter(dist_ds2)

    spec1 = iter1._type_spec  # Wrapped TensorSpec has shape [None, None]
    spec2 = iter2._type_spec  # Wrapped TensorSpec has shape [None, 2]

    self.assertNotEqual(spec1, spec2)
    self.assertEqual(spec1, spec1.most_specific_compatible_type(spec2))
    self.assertEqual(spec1, spec2.most_specific_compatible_type(spec1))

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          tf_api_version=2,
          distribution=[
              strategy_combinations.mirrored_strategy_with_two_gpus,
              strategy_combinations.mirrored_strategy_with_cpu_1_and_2,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          ],
          enable_get_next_as_optional=[True, False],
          experimental_place_dataset_on_device=[True, False],
          experimental_fetch_to_device=[True, False],
      ))
  def testFromFunctionInputSignatureForPerReplicaValuesWithOptions(
      self, distribution, enable_get_next_as_optional,
      experimental_place_dataset_on_device, experimental_fetch_to_device):
    # [FUNCTION DOC]
    # Scenario: two temporary text files (five and nine lines) are read by a dataset function that
    # shards by input pipeline and batches by the per-replica batch size of four. Distributed
    # through the from-function API with per-replica replication mode and input options (place on
    # device, fetch to device). Skipped when both options are true (not allowed in per-replica
    # mode). Checks iterator spec structure, a components round trip (equal input workers and
    # iterators), equal element-spec structure between dataset and iterator, and that a compiled
    # function with the element spec as input signature accepts every element.
    if experimental_place_dataset_on_device and experimental_fetch_to_device:
      self.skipTest("Setting experimental_place_dataset_on_device and "
                    "experimental_fetch_to_device to `True` is not "
                    "allowed when using "
                    "distribute_lib.InputReplicationMode.PER_REPLICA.")

    fname1 = os.path.join(self.get_temp_dir(), "1.txt")
    _create_text_file(fname1, 5)
    fname2 = os.path.join(self.get_temp_dir(), "2.txt")
    _create_text_file(fname2, 9)

    def dataset_fn(input_context):
      # [FUNCTION DOC]
      # Dataset function: builds a dataset of the two file names, shards it by the input context's
      # pipeline count and index, reads lines as text, converts them to numbers and batches by the
      # per-replica batch size for four. Parameter: the input context. Returns the dataset.
      dataset = dataset_ops.DatasetV2.from_tensor_slices([fname1, fname2])
      dataset = dataset.shard(input_context.num_input_pipelines,
                              input_context.input_pipeline_id)
      return readers.TextLineDatasetV2(dataset).map(
          string_ops.string_to_number).batch(
              input_context.get_per_replica_batch_size(4))

    options = distribute_lib.InputOptions(
        experimental_place_dataset_on_device=(
            experimental_place_dataset_on_device),
        experimental_fetch_to_device=experimental_fetch_to_device,
        experimental_replication_mode=(
            distribute_lib.InputReplicationMode.PER_REPLICA))

    distribution.extended.experimental_enable_get_next_as_optional = (
        enable_get_next_as_optional)
    ds = distribution.experimental_distribute_datasets_from_function(
        dataset_fn, options)

    iterator = iter(ds)
    _check_type_spec_structure(iterator)
    spec = iterator._type_spec
    tensor_list = spec._to_components(iterator)
    re_iterator = spec._from_components(tensor_list)

    _check_type_spec_structure(iter(ds))
    element_spec = ds.element_spec
    iter_element_spec = iter(ds).element_spec
    nest.assert_same_structure(element_spec, iter_element_spec)
    self.assertAllEqual(
        nest.flatten(element_spec), nest.flatten(iter_element_spec))
    self.assertEqual(iterator._input_workers, re_iterator._input_workers)
    self.assertAllEqual(iterator._iterators, re_iterator._iterators)

    @def_function.function(input_signature=[element_spec])
    def process_inputs(inputs):
      # [FUNCTION DOC]
      # Nested compiled function with a fixed input signature: runs an identity function on every
      # replica with the given per-replica inputs. Returns nothing.
      distribution.run(lambda inputs: inputs, args=(inputs,))

    for x in ds:
      process_inputs(x)


class DistributedDatasetTypeSpecTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests the type spec of distributed datasets (not iterators): element spec, compiled-function
  # return, ragged/sparse inputs and component round trips, using mirrored strategies.

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          tf_api_version=2,
          distribution=[
              strategy_combinations.mirrored_strategy_with_two_gpus,
              strategy_combinations.mirrored_strategy_with_cpu_1_and_2,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          ],
          enable_get_next_as_optional=[True, False]))
  def testTypeSpecBase(self, distribution, enable_get_next_as_optional):
    # [FUNCTION DOC]
    # Scenario: a range dataset batched by two is distributed. Asserts the dataset's type spec has
    # the same input workers as the dataset and an element spec of two int64 tensor specs with an
    # unknown batch dimension (one per replica).

    def create_dataset():
      # [FUNCTION DOC]
      # Builds and returns a range dataset of ten elements batched by two.
      dataset = dataset_ops.DatasetV2.range(10).batch(2)
      return dataset

    distribution.extended.experimental_enable_get_next_as_optional = (
        enable_get_next_as_optional)

    dist_dataset = distribution.experimental_distribute_dataset(
        create_dataset())

    spec = dist_dataset._type_spec
    self.assertEqual(spec._input_workers, dist_dataset._input_workers)
    self.assertEqual(
        spec._element_spec._value_specs,
        (tensor_spec.TensorSpec(shape=(None,), dtype=dtypes.int64, name=None),
         tensor_spec.TensorSpec(shape=(None,), dtype=dtypes.int64, name=None)))

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          tf_api_version=2,
          distribution=[
              strategy_combinations.mirrored_strategy_with_cpu_1_and_2,
          ],
          enable_get_next_as_optional=[True, False]))
  def testTypeSpecReturnedFromTFFunction(self, distribution,
                                         enable_get_next_as_optional):
    # [FUNCTION DOC]
    # Scenario: a compiled function creates a distributed dataset and returns it; the spec and
    # the data read from it are then checked (element spec, and that the values equal the expected
    # range). Currently the test is unconditionally skipped at its start because of a known
    # device-copy failure for dataset variants, so everything after the skip call is dead code.
    # KNOWN ISSUE: the test body after the unconditional skip never runs.
    # TODO(ishark): This is observed when tensor is copied from one device to
    # other and since DatasetVariantWrapper does not have a copy
    # function. Some Context: b/146981184
    # Try to renable with non-canonicalized input workers, which
    # helped in PS Strategy for similar error.
    self.skipTest("Failures observed in Ubuntu presubmit: No unary variant  "
                  "device copy function found for direction: 1 and Variant "
                  "type_index:tensorflow::data::(anonymous namespace)::"
                  "DatasetVariantWrapper")

    @def_function.function
    def create_dist_dataset():
      # [FUNCTION DOC]
      # Nested compiled function: builds a range dataset batched by two and returns it distributed
      # through the strategy (unreachable because the test is skipped earlier).
      dataset = dataset_ops.DatasetV2.range(10).batch(2)
      return distribution.experimental_distribute_dataset(dataset)

    distribution.extended.experimental_enable_get_next_as_optional = (
        enable_get_next_as_optional)

    dist_dataset = create_dist_dataset()

    spec = dist_dataset._type_spec
    self.assertEqual(spec._input_workers, dist_dataset._input_workers)
    self.assertEqual(
        spec._element_spec._value_specs,
        (tensor_spec.TensorSpec(shape=(None,), dtype=dtypes.int64, name=None),
         tensor_spec.TensorSpec(shape=(None,), dtype=dtypes.int64, name=None)))

    # Read distributed data to confirm values are correct.
    iterator = iter(dist_dataset)
    data = []
    for it in iterator:
      data.append(distribution.experimental_local_results(it))
    self.assertAllEqual(
        nest.flatten(data),
        list(dataset_ops.DatasetV2.range(10).batch(1).as_numpy_iterator()))

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          tf_api_version=2,
          distribution=[
              strategy_combinations.mirrored_strategy_with_two_gpus,
              strategy_combinations.mirrored_strategy_with_cpu_1_and_2,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          ],
          enable_get_next_as_optional=[True, False]))
  def testTypeSpecRaggedTensor(self, distribution, enable_get_next_as_optional):
    # [FUNCTION DOC]
    # Scenario: a dataset of mappings holding a dense, a ragged and a sparse version of the same
    # twenty-row ragged data (row lengths cycle from zero to three) is sharded by the input
    # context and batched by the per-replica batch size of eight (twenty is not divisible, to
    # exercise partial batches), then distributed. Asserts the dataset spec's input workers and
    # that its element spec maps each key to a per-replica spec of two copies of the matching
    # sparse, dense (unknown rows, three columns) or ragged float spec.
    ctx = distribute_lib.InputContext()
    batch_size = ctx.get_per_replica_batch_size(8)
    # Use 20 which isn't divisible by 8 to test partial batch behavior.
    row_lengths = np.mod(np.arange(20), 4).astype(np.int64)
    ragged_tensor = ragged_tensor_lib.RaggedTensor.from_row_lengths(
        np.repeat(np.arange(20, dtype=np.float32), row_lengths), row_lengths)
    dataset = dataset_ops.DatasetV2.from_tensor_slices({
        "dense": ragged_tensor.to_tensor(),
        "ragged": ragged_tensor,
        "sparse": ragged_tensor.to_sparse(),
    })
    dataset = dataset.shard(ctx.num_input_pipelines, ctx.input_pipeline_id)
    dataset = dataset.batch(batch_size)

    distribution.extended.experimental_enable_get_next_as_optional = (
        enable_get_next_as_optional)
    dist_dataset = distribution.experimental_distribute_dataset(dataset)
    spec = dist_dataset._type_spec
    self.assertEqual(spec._input_workers, dist_dataset._input_workers)
    self.assertEqual(
        spec._element_spec, {
            "sparse":
                values.PerReplicaSpec(
                    sparse_tensor.SparseTensorSpec(
                        tensor_shape.TensorShape([None, 3]), dtypes.float32),
                    sparse_tensor.SparseTensorSpec(
                        tensor_shape.TensorShape([None, 3]), dtypes.float32)),
            "dense":
                values.PerReplicaSpec(
                    tensor_spec.TensorSpec(
                        shape=(None, 3), dtype=dtypes.float32, name=None),
                    tensor_spec.TensorSpec(
                        shape=(None, 3), dtype=dtypes.float32, name=None)),
            "ragged":
                values.PerReplicaSpec(
                    ragged_tensor_lib.RaggedTensorSpec(
                        tensor_shape.TensorShape([None, None]), dtypes.float32,
                        1, dtypes.int64),
                    ragged_tensor_lib.RaggedTensorSpec(
                        tensor_shape.TensorShape([None, None]), dtypes.float32,
                        1, dtypes.int64))
        })

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          tf_api_version=2,
          distribution=[
              strategy_combinations.mirrored_strategy_with_two_gpus,
              strategy_combinations.mirrored_strategy_with_cpu_1_and_2,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          ],
          enable_get_next_as_optional=[True, False],
          experimental_place_dataset_on_device=[True, False],
          experimental_fetch_to_device=[True, False]))
  def testTypeSpecComponents(self, distribution, enable_get_next_as_optional,
                             experimental_place_dataset_on_device,
                             experimental_fetch_to_device):
    # [FUNCTION DOC]
    # Scenario: a batched range dataset is distributed with input options (place on device, fetch
    # to device). Asserts the type spec's input workers and element spec, then converts the
    # distributed dataset to components and back, and asserts the rebuilt dataset has equal input
    # workers, cloned datasets, element spec, get-next-as-optional setting and options.
    dataset = dataset_ops.DatasetV2.range(10).batch(2)
    distribution.extended.experimental_enable_get_next_as_optional = (
        enable_get_next_as_optional)

    options = distribute_lib.InputOptions(
        experimental_place_dataset_on_device=
        experimental_place_dataset_on_device,
        experimental_fetch_to_device=experimental_fetch_to_device)

    dist_dataset = distribution.experimental_distribute_dataset(
        dataset, options)

    spec = dist_dataset._type_spec
    self.assertEqual(spec._input_workers, dist_dataset._input_workers)
    self.assertEqual(
        spec._element_spec._value_specs,
        (tensor_spec.TensorSpec(shape=(None,), dtype=dtypes.int64, name=None),
         tensor_spec.TensorSpec(shape=(None,), dtype=dtypes.int64, name=None)))
    components = spec._to_components(dist_dataset)
    re_dist_dataset = spec._from_components(components)

    self.assertEqual(dist_dataset._input_workers,
                     re_dist_dataset._input_workers)
    self.assertAllEqual(dist_dataset._cloned_datasets,
                        re_dist_dataset._cloned_datasets)
    self.assertEqual(dist_dataset._element_spec, re_dist_dataset._element_spec)
    self.assertEqual(dist_dataset._enable_get_next_as_optional,
                     re_dist_dataset._enable_get_next_as_optional)
    self.assertEqual(dist_dataset._options, re_dist_dataset._options)


class DistributedDatasetsFromFunctionSpecTest(test.TestCase,
                                              parameterized.TestCase):
  # [CLASS DOC]
  # Tests the type spec of distributed datasets created from a dataset function, including the
  # components round trip and use of the element spec as a compiled-function input signature.

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          tf_api_version=2,
          distribution=[
              strategy_combinations.mirrored_strategy_with_two_gpus,
              strategy_combinations.mirrored_strategy_with_cpu_1_and_2,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          ],
          enable_get_next_as_optional=[True, False],
          experimental_place_dataset_on_device=[True, False],
          experimental_fetch_to_device=[True, False],
      ))
  def testDistributedDatasetsFromFunctionSpec(
      self, distribution, enable_get_next_as_optional,
      experimental_place_dataset_on_device, experimental_fetch_to_device):
    # [FUNCTION DOC]
    # Scenario: same two-text-file dataset function and per-replica input options as the iterator
    # test, but the distributed dataset itself is converted to components and rebuilt through its
    # type spec. Skipped when both place-on-device and fetch-to-device are true. Asserts the
    # rebuilt dataset's element spec has the same structure and flattened contents as the
    # iterator's element spec, that input workers and element spec equal the originals, and that a
    # compiled function using the element spec as input signature accepts every element.

    if experimental_place_dataset_on_device and experimental_fetch_to_device:
      self.skipTest("Setting experimental_place_dataset_on_device and "
                    "experimental_fetch_to_device to `True` is not "
                    "allowed when using "
                    "distribute_lib.InputReplicationMode.PER_REPLICA.")

    fname1 = os.path.join(self.get_temp_dir(), "1.txt")
    _create_text_file(fname1, 5)
    fname2 = os.path.join(self.get_temp_dir(), "2.txt")
    _create_text_file(fname2, 9)

    def dataset_fn(input_context):
      # [FUNCTION DOC]
      # Dataset function: shards the two file names by the input context, reads lines as text,
      # converts to numbers and batches by the per-replica batch size for four. Returns the
      # dataset.
      dataset = dataset_ops.DatasetV2.from_tensor_slices([fname1, fname2])
      dataset = dataset.shard(input_context.num_input_pipelines,
                              input_context.input_pipeline_id)
      return readers.TextLineDatasetV2(dataset).map(
          string_ops.string_to_number).batch(
              input_context.get_per_replica_batch_size(4))

    options = distribute_lib.InputOptions(
        experimental_place_dataset_on_device=
        experimental_place_dataset_on_device,
        experimental_fetch_to_device=experimental_fetch_to_device,
        experimental_replication_mode=(
            distribute_lib.InputReplicationMode.PER_REPLICA))

    distribution.extended.experimental_enable_get_next_as_optional = (
        enable_get_next_as_optional)
    ds = distribution.experimental_distribute_datasets_from_function(
        dataset_fn, options)

    spec = ds._type_spec
    components = spec._to_components(ds)
    re_ds = spec._from_components(components)

    element_spec = re_ds.element_spec
    iter_element_spec = iter(ds).element_spec
    nest.assert_same_structure(element_spec, iter_element_spec)
    self.assertAllEqual(
        nest.flatten(element_spec), nest.flatten(iter_element_spec))
    self.assertEqual(ds._input_workers, re_ds._input_workers)
    self.assertEqual(ds._element_spec, re_ds._element_spec)

    @def_function.function(input_signature=[element_spec])
    def process_inputs(inputs):
      # [FUNCTION DOC]
      # Nested compiled function with a fixed input signature: runs an identity function on every
      # replica with the given per-replica inputs. Returns nothing.
      distribution.run(lambda inputs: inputs, args=(inputs,))

    for x in ds:
      process_inputs(x)


class RaggedTensorDistributedIteratorTest(test.TestCase,
                                          parameterized.TestCase):
  # [CLASS DOC]
  # Tests the type spec of distributed iterators over datasets that contain dense, ragged and
  # sparse components, under mirrored, central-storage, TPU and multi-worker strategies.

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          tf_api_version=2,
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.central_storage_strategy_with_gpu_and_cpu,
              strategy_combinations.multi_worker_mirrored_2x2_gpu,
          ],
          enable_get_next_as_optional=[True, False]))
  def testTypeSpec(self, distribution, enable_get_next_as_optional):
    # [FUNCTION DOC]
    # Scenario: a dataset of mappings with dense, ragged and sparse versions of twenty rows of
    # ragged data is sharded and batched by the per-replica size of eight, distributed, and an
    # iterator is created in the strategy scope. Checks spec structure, equal input workers, and
    # that the element spec maps each key to a per-replica spec of two matching sparse, dense or
    # ragged float specs.
    ctx = distribute_lib.InputContext()
    batch_size = ctx.get_per_replica_batch_size(8)
    # Use 20 which isn't divisible by 8 to test partial batch behavior.
    row_lengths = np.mod(np.arange(20), 4).astype(np.int64)
    ragged_tensor = ragged_tensor_lib.RaggedTensor.from_row_lengths(
        np.repeat(np.arange(20, dtype=np.float32), row_lengths), row_lengths)
    dataset = dataset_ops.DatasetV2.from_tensor_slices({
        "dense": ragged_tensor.to_tensor(),
        "ragged": ragged_tensor,
        "sparse": ragged_tensor.to_sparse(),
    })
    dataset = dataset.shard(ctx.num_input_pipelines, ctx.input_pipeline_id)
    dataset = dataset.batch(batch_size)

    distribution.extended.experimental_enable_get_next_as_optional = (
        enable_get_next_as_optional)

    dist_dataset = distribution.experimental_distribute_dataset(dataset)
    with distribution.scope():
      iterator = iter(dist_dataset)
      _check_type_spec_structure(iterator)

    spec = iterator._type_spec
    self.assertEqual(spec._input_workers, iterator._input_workers)
    self.assertEqual(
        spec._element_spec, {
            "sparse":
                values.PerReplicaSpec(
                    sparse_tensor.SparseTensorSpec(
                        tensor_shape.TensorShape([None, 3]), dtypes.float32),
                    sparse_tensor.SparseTensorSpec(
                        tensor_shape.TensorShape([None, 3]), dtypes.float32)),
            "dense":
                values.PerReplicaSpec(
                    tensor_spec.TensorSpec(
                        shape=(None, 3), dtype=dtypes.float32, name=None),
                    tensor_spec.TensorSpec(
                        shape=(None, 3), dtype=dtypes.float32, name=None)),
            "ragged":
                values.PerReplicaSpec(
                    ragged_tensor_lib.RaggedTensorSpec(
                        tensor_shape.TensorShape([None, None]), dtypes.float32,
                        1, dtypes.int64),
                    ragged_tensor_lib.RaggedTensorSpec(
                        tensor_shape.TensorShape([None, None]), dtypes.float32,
                        1, dtypes.int64))
        })

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          tf_api_version=2,
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
              strategy_combinations.central_storage_strategy_with_gpu_and_cpu,
              strategy_combinations.multi_worker_mirrored_2x2_gpu,
          ],

          enable_get_next_as_optional=[True, False]))
  def testTypeSpecRoundTrip(self, distribution, enable_get_next_as_optional):
    # [FUNCTION DOC]
    # Scenario: same ragged/sparse/dense dataset as the type spec test. For TPU strategies, input
    # options disable fetch-to-device (device prefetch is unsupported with sparse or ragged
    # tensors); otherwise no options are used. The iterator is converted to components and
    # rebuilt; asserts equal input workers and equal per-device iterators.
    ctx = distribute_lib.InputContext()
    batch_size = ctx.get_per_replica_batch_size(8)
    # Use 20 which isn't divisible by 8 to test partial batch behavior.
    row_lengths = np.mod(np.arange(20), 4).astype(np.int64)
    ragged_tensor = ragged_tensor_lib.RaggedTensor.from_row_lengths(
        np.repeat(np.arange(20, dtype=np.float32), row_lengths), row_lengths)
    dataset = dataset_ops.DatasetV2.from_tensor_slices({
        "dense": ragged_tensor.to_tensor(),
        "ragged": ragged_tensor,
        "sparse": ragged_tensor.to_sparse(),
    })
    dataset = dataset.shard(ctx.num_input_pipelines, ctx.input_pipeline_id)
    dataset = dataset.batch(batch_size)

    distribution.extended.experimental_enable_get_next_as_optional = (
        enable_get_next_as_optional)

    if isinstance(distribution,
                  (tpu_strategy.TPUStrategyV2, tpu_strategy.TPUStrategy)):
      # TPUStrategy does not support distributed datasets with device prefetch
      # when using sparse or ragged tensors.
      options = distribute_lib.InputOptions(experimental_fetch_to_device=False)
    else:
      options = None

    dist_dataset = distribution.experimental_distribute_dataset(
        dataset, options)
    with distribution.scope():
      iterator = iter(dist_dataset)
      _check_type_spec_structure(iterator)

    spec = iterator._type_spec

    tensor_list = spec._to_components(iterator)
    re_iterator = spec._from_components(tensor_list)

    self.assertEqual(iterator._input_workers, re_iterator._input_workers)
    self.assertAllEqual(iterator._iterators, re_iterator._iterators)

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          tf_api_version=2,
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
          ],
          enable_get_next_as_optional=[True, False]))
  def testDoesNotTriggerFunctionTracing(self, distribution,
                                        enable_get_next_as_optional):
    # [FUNCTION DOC]
    # Scenario: a compiled function takes five elements from a distributed iterator over a fifty
    # row ragged/sparse/dense dataset (fetch-to-device disabled on TPU strategies). It is called
    # with a new iterator three times. Asserts it is traced only once and returns five each time.
    trace_count = [0]

    @def_function.function
    def f(iterator):
      # [FUNCTION DOC]
      # Nested compiled function: increments the outer trace counter on each trace, takes five
      # elements from the iterator and returns the number taken.
      trace_count[0] += 1
      counter = np.int64(0)
      for _ in range(5):
        next(iterator)
        counter += 1
      return counter

    ctx = distribute_lib.InputContext()
    batch_size = ctx.get_per_replica_batch_size(8)
    # Use 20 which isn't divisible by 8 to test partial batch behavior.
    row_lengths = np.mod(np.arange(50), 4).astype(np.int64)
    ragged_tensor = ragged_tensor_lib.RaggedTensor.from_row_lengths(
        np.repeat(np.arange(50, dtype=np.float32), row_lengths), row_lengths)
    dataset = dataset_ops.DatasetV2.from_tensor_slices({
        "dense": ragged_tensor.to_tensor(),
        "ragged": ragged_tensor,
        "sparse": ragged_tensor.to_sparse(),
    })
    dataset = dataset.shard(ctx.num_input_pipelines, ctx.input_pipeline_id)
    dataset = dataset.batch(batch_size)

    distribution.extended.experimental_enable_get_next_as_optional = (
        enable_get_next_as_optional)

    if isinstance(distribution,
                  (tpu_strategy.TPUStrategyV2, tpu_strategy.TPUStrategy)):
      # TPUStrategy does not support distributed datasets with device prefetch
      # when using sparse or ragged tensors.
      options = distribute_lib.InputOptions(experimental_fetch_to_device=False)
    else:
      options = None

    dist_dataset = distribution.experimental_distribute_dataset(
        dataset, options)
    with distribution.scope():
      for _ in range(3):
        iterator = iter(dist_dataset)
        _check_type_spec_structure(iterator)
        counter = f(iterator)

        self.assertEqual(trace_count[0], 1)
        self.assertEqual(counter, 5)


def _check_type_spec_structure(x):
  # [FUNCTION DOC]
  # HELPER: structure assertion used by the tests in this module.
  # If the argument is a composite tensor, asserts that it has the same nested structure as its
  # type spec (expanding composites). Otherwise does nothing. Returns nothing.
  """Verifies that `x` has the same structure as its `TypeSpec`."""
  if isinstance(x, composite_tensor.CompositeTensor):
    nest.assert_same_structure(x, x._type_spec, expand_composites=True)


def _create_text_file(fname, num_lines):
  # [FUNCTION DOC]
  # HELPER: creates input files for the from-function dataset tests.
  # Writes a text file at the given path containing the integers from zero up to (excluding) the
  # given line count, one per line. Overwrites an existing file. Returns nothing.
  with open(fname, "w") as f:
    for i in range(num_lines):
      f.write("%d\n" % i)


if __name__ == "__main__":
  test_util.main()
