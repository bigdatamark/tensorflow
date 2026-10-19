# [MODULE DOC]
# Kernel tests for the dataset options object (tf.data.Options). Covers default options, how
# options merge when applied repeatedly or when combining several input datasets, default
# sub-option objects, immutability of options read from a dataset, conversion to and from the
# serialized protocol buffer form, backward-compatible aliases (experimental threading and
# deterministic), persistence of options through functions and graph round trips, and the
# effect of options on optimization. Uses the shared dataset test base; no import-time side
# effects other than the test entry point.
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
"""Tests for `tf.data.Options`."""

from absl.testing import parameterized

from tensorflow.core.framework import dataset_options_pb2
from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import combinations
from tensorflow.python.platform import test


class OptionsTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for dataset options: merging, defaults, proto conversion, compatibility aliases
  # and persistence through graphs and functions. Provides one shared helper for reading the
  # effective options of a dataset in either execution mode.

  def _get_options(self, dataset):
    # [FUNCTION DOC]
    # HELPER: reads the effective options of a dataset for the tests. In eager mode it returns
    # the options directly; in graph mode it evaluates the dataset's options tensor and decodes
    # it into an options object. Parameter dataset is the dataset to inspect; returns the options.
    if context.executing_eagerly():
      return dataset.options()
    return dataset_ops.Dataset._options_tensor_to_options(
        self.evaluate(dataset._options()))

  @combinations.generate(test_base.default_test_combinations())
  def testOptionsDefault(self):
    # [FUNCTION DOC]
    # Scenario: a dataset with no options set. Expected: its options equal a freshly created
    # default options object.
    ds = dataset_ops.Dataset.range(0)
    self.assertEqual(options_lib.Options(), ds.options())

  @combinations.generate(test_base.default_test_combinations())
  def testOptionsOnce(self):
    # [FUNCTION DOC]
    # Scenario: options applied once, followed by a later transformation (cache). Expected: the
    # dataset's options still equal the applied options.
    options = options_lib.Options()
    ds = dataset_ops.Dataset.range(0).with_options(options).cache()
    self.assertEqual(options, ds.options())

  @combinations.generate(test_base.default_test_combinations())
  def testOptionsTwiceSame(self):
    # [FUNCTION DOC]
    # Scenario: the same options object (autotune enabled) is applied twice. Expected: the
    # resulting options equal that object.
    options = options_lib.Options()
    options.autotune.enabled = True
    ds = dataset_ops.Dataset.range(0).with_options(options).with_options(
        options)
    self.assertEqual(options, self._get_options(ds))

  @combinations.generate(test_base.default_test_combinations())
  def testOptionsTwiceDifferentOptions(self):
    # [FUNCTION DOC]
    # Scenario: two options objects setting different fields (autotune enabled; deterministic
    # off) applied in sequence. Expected: the result merges both: autotune is enabled and
    # deterministic is exactly false.
    options1 = options_lib.Options()
    options1.autotune.enabled = True
    options2 = options_lib.Options()
    options2.deterministic = False
    ds = dataset_ops.Dataset.range(0)
    ds = ds.with_options(options1)
    ds = ds.with_options(options2)
    options = self._get_options(ds)
    self.assertTrue(options.autotune.enabled)
    # Explicitly check that flag is False since assertFalse allows None
    self.assertIs(options.deterministic, False)

  @combinations.generate(test_base.default_test_combinations())
  def testOptionsTwiceSameOption(self):
    # [FUNCTION DOC]
    # Scenario: two options objects set the same field (autotune enabled) to false then true.
    # Expected: the later setting wins, so autotune is enabled.
    options1 = options_lib.Options()
    options1.autotune.enabled = False
    options2 = options_lib.Options()
    options2.autotune.enabled = True
    ds = dataset_ops.Dataset.range(0)
    ds = ds.with_options(options1)
    ds = ds.with_options(options2)
    self.assertTrue(self._get_options(ds).autotune.enabled)

  @combinations.generate(test_base.default_test_combinations())
  def testOptionsTwiceSameOptionWithMap(self):
    # [FUNCTION DOC]
    # Scenario: options with a framework-type list set before and after a map transformation.
    # Expected: the dataset still produces the mapped values one to five, and the framework-type
    # lists from both options are concatenated (length two).
    options1 = options_lib.Options()
    options1.framework_type = ["seqio"]
    options2 = options_lib.Options()
    options2.framework_type = ["tfgrain"]
    ds = dataset_ops.Dataset.range(5)
    ds = ds.with_options(options1)
    ds = ds.map(lambda x: x + 1)
    ds = ds.with_options(options2)
    self.assertDatasetProduces(ds, [1, 2, 3, 4, 5])
    self.assertLen(self._get_options(ds).framework_type, 2)

  @combinations.generate(test_base.default_test_combinations())
  def testOptionsMergeOptionsFromMultipleInputs(self):
    # [FUNCTION DOC]
    # Scenario: two datasets with different options are zipped. Expected: the zipped dataset's
    # options contain both settings (autotune enabled and deterministic true).
    options1 = options_lib.Options()
    options1.autotune.enabled = True
    options2 = options_lib.Options()
    options2.deterministic = True
    ds1 = dataset_ops.Dataset.range(0).with_options(options1)
    ds2 = dataset_ops.Dataset.range(0).with_options(options2)
    ds = dataset_ops.Dataset.zip((ds1, ds2))
    options = self._get_options(ds)
    self.assertTrue(options.autotune.enabled)
    self.assertTrue(options.deterministic)

  @combinations.generate(test_base.default_test_combinations())
  def testOptionsHaveDefaults(self):
    # [FUNCTION DOC]
    # Scenario: two fresh options objects. Expected: their optimization and threading
    # sub-options are distinct instances (not shared) yet each equals a default instance of its
    # type.
    options1 = options_lib.Options()
    options2 = options_lib.Options()
    self.assertIsNot(options1.experimental_optimization,
                     options2.experimental_optimization)
    self.assertIsNot(options1.threading, options2.threading)
    self.assertEqual(options1.experimental_optimization,
                     options_lib.OptimizationOptions())
    self.assertEqual(options1.threading, options_lib.ThreadingOptions())

  @combinations.generate(test_base.default_test_combinations())
  def testMutatingOptionsRaiseValueError(self):
    # [FUNCTION DOC]
    # Scenario: options obtained from a dataset that had options merged across a map are then
    # modified. Expected: assigning a field raises a value error because they are immutable.
    ds = dataset_ops.Dataset.range(0)
    options1 = options_lib.Options()
    options1.experimental_slack = True
    options2 = options_lib.Options()
    options2.autotune.enabled = True
    ds = ds.with_options(options1)
    ds = ds.map(lambda x: 2 * x)
    ds = ds.with_options(options2)
    dataset_options = ds.options()
    with self.assertRaises(ValueError):
      dataset_options.deterministic = True

  @combinations.generate(test_base.eager_only_combinations())
  def testNestedDataset(self):
    # [FUNCTION DOC]
    # Eager-only scenario: a single-element dataset is concatenated with itself ninety-nine
    # times, creating a deeply nested dataset graph. Expected: it produces the value zero one
    # hundred times (the options machinery handles deep nesting).
    ds = dataset_ops.Dataset.from_tensors(0)
    result = ds

    for _ in range(99):
      result = result.concatenate(ds)
    self.assertDatasetProduces(result, [0]*100)

  @combinations.generate(test_base.default_test_combinations())
  def testOptionsProtoRoundTrip(self):
    # [FUNCTION DOC]
    # Scenario: an options object with many fields set across autotune, distribution,
    # optimization, threading and top-level settings is converted to its protocol buffer form
    # and loaded into a fresh options object. Expected: the framework-type list and the whole
    # options object are equal after the round trip.
    options = options_lib.Options()
    options.autotune.enabled = True
    options.autotune.cpu_budget = 10
    options.autotune.ram_budget = 20
    options.deterministic = True
    options.experimental_external_state_policy = (
        options_lib.ExternalStatePolicy.FAIL)
    options.experimental_distribute.auto_shard_policy = (
        options_lib.AutoShardPolicy.DATA)
    options.experimental_distribute.num_devices = 1000
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.filter_fusion = True
    options.experimental_optimization.filter_parallelization = True
    options.experimental_optimization.inject_prefetch = False
    options.experimental_optimization.map_and_batch_fusion = True
    options.experimental_optimization.map_and_filter_fusion = True
    options.experimental_optimization.map_fusion = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.noop_elimination = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.shuffle_and_repeat_fusion = True
    options.experimental_optimization.seq_interleave_prefetch = True
    options.experimental_warm_start = True
    options.experimental_slack = True
    options.dataset_name = "test_name"
    options.framework_type = ["TFDS", "TfGrain"]
    options.threading.max_intra_op_parallelism = 30
    options.threading.private_threadpool_size = 40
    pb = options._to_proto()
    result = options_lib.Options()
    result._from_proto(pb)
    self.assertEqual(options.framework_type, result.framework_type)
    self.assertEqual(options, result)

  @combinations.generate(test_base.default_test_combinations())
  def testOptionsProtoDefaultValuesRoundTrip(self):
    # [FUNCTION DOC]
    # Scenario: default options are converted to a protocol buffer and loaded into a new options
    # object. Expected: the new object equals the original.
    options = options_lib.Options()
    pb = options._to_proto()
    result = options_lib.Options()
    result._from_proto(pb)
    self.assertEqual(options, result)

  @combinations.generate(test_base.default_test_combinations())
  def testProtoOptionsDefaultValuesRoundTrip(self):
    # [FUNCTION DOC]
    # Scenario: an empty options protocol buffer is loaded into an options object and converted
    # back. Expected: the result equals a buffer with empty autotune, distribute, optimization,
    # service and threading sub-messages and warm start enabled (the defaults filled in).
    pb = dataset_options_pb2.Options()
    options = options_lib.Options()
    options._from_proto(pb)
    result = options._to_proto()
    expected_pb = dataset_options_pb2.Options()
    expected_pb.autotune_options.CopyFrom(dataset_options_pb2.AutotuneOptions())
    expected_pb.distribute_options.CopyFrom(
        dataset_options_pb2.DistributeOptions())
    expected_pb.optimization_options.CopyFrom(
        dataset_options_pb2.OptimizationOptions())
    expected_pb.warm_start = True
    expected_pb.service_options.CopyFrom(
        dataset_options_pb2.ServiceOptions())
    expected_pb.threading_options.CopyFrom(
        dataset_options_pb2.ThreadingOptions())
    self.assertProtoEquals(expected_pb, result)

  @combinations.generate(test_base.default_test_combinations())
  def testThreadingOptionsBackwardCompatibility(self):
    # [FUNCTION DOC]
    # Scenario: threading settings written through the current name and read through the older
    # experimental name, and vice versa. Expected: both names refer to the same underlying
    # values (twenty and eighty).
    opts = options_lib.Options()
    opts.threading.max_intra_op_parallelism = 20
    self.assertEqual(opts.experimental_threading.max_intra_op_parallelism, 20)
    opts.experimental_threading.private_threadpool_size = 80
    self.assertEqual(opts.threading.private_threadpool_size, 80)

  @combinations.generate(test_base.default_test_combinations())
  def testExperimentalThreadingOptionsOverride(self):
    # [FUNCTION DOC]
    # Scenario: the same threading field is set through both the current and the experimental
    # alias (twenty, then forty), then round-tripped through the protocol buffer. Expected: the
    # two names are the same object initially and agree after the round trip.
    options = options_lib.Options()
    self.assertEqual(options.threading, options.experimental_threading)
    options.threading.max_intra_op_parallelism = 20
    options.experimental_threading.max_intra_op_parallelism = 40
    pb = options._to_proto()
    result = options_lib.Options()
    result._from_proto(pb)
    self.assertEqual(result.experimental_threading.max_intra_op_parallelism,
                     result.threading.max_intra_op_parallelism)

  @combinations.generate(test_base.default_test_combinations())
  def testExperimentalDeterministicOverride(self):
    # [FUNCTION DOC]
    # Scenario: the deterministic option is set via its experimental alias, round-tripped
    # through the protocol buffer, then reset to true through the alias. Expected: the alias and
    # the main field always agree (false, then true).
    options = options_lib.Options()
    self.assertEqual(options.deterministic, options.experimental_deterministic)
    options.experimental_deterministic = False
    pb = options._to_proto()
    result = options_lib.Options()
    result._from_proto(pb)
    self.assertFalse(result.deterministic)
    self.assertEqual(result.deterministic, result.experimental_deterministic)
    result.experimental_deterministic = True
    self.assertTrue(result.deterministic)
    self.assertEqual(result.deterministic, result.experimental_deterministic)

  @combinations.generate(test_base.default_test_combinations())
  def testPersistenceOptionsSetOutsideFunction(self):
    # [FUNCTION DOC]
    # Scenario: options (slack enabled) are applied to a dataset outside a traced function that
    # only maps over it. Expected: the options, decoded from the returned dataset's options
    # tensor, still have slack enabled.

    @def_function.function
    def fn(dataset):
      # [FUNCTION DOC]
      # HELPER: traced function for the persistence test; multiplies each element of the given
      # dataset by ten and returns the new dataset.
      dataset = dataset.map(lambda x: 10 * x)
      return dataset

    dataset = dataset_ops.Dataset.range(5)
    options = options_lib.Options()
    options.experimental_slack = True
    dataset = dataset.with_options(options)
    dataset = fn(dataset)
    result = dataset_ops.Dataset._options_tensor_to_options(
        self.evaluate(dataset._options()))
    self.assertTrue(result.experimental_slack)

  @combinations.generate(test_base.default_test_combinations())
  def testPersistenceOptionsSetInsideFunction(self):
    # [FUNCTION DOC]
    # Scenario: options (slack enabled) are applied inside a traced function to the dataset it
    # receives, followed by a map. Expected: the options decoded from the returned dataset's
    # options tensor have slack enabled.

    @def_function.function
    def fn(dataset):
      # [FUNCTION DOC]
      # HELPER: traced function for the persistence test; applies options with slack enabled to
      # the given dataset, maps each element times ten and returns the dataset.
      options = options_lib.Options()
      options.experimental_slack = True
      dataset = dataset.with_options(options)
      dataset = dataset.map(lambda x: 10 * x)
      return dataset

    dataset = dataset_ops.Dataset.range(5)
    dataset = fn(dataset)
    result = dataset_ops.Dataset._options_tensor_to_options(
        self.evaluate(dataset._options()))
    self.assertTrue(result.experimental_slack)

  @combinations.generate(test_base.default_test_combinations())
  def testOptionsPersistenceGraphRoundTrip(self):
    # [FUNCTION DOC]
    # Scenario: a dataset with slack enabled and default optimizations disabled is serialized to
    # a graph and rebuilt. Expected: the rebuilt dataset's options keep slack enabled and
    # default optimizations exactly false.
    dataset = dataset_ops.Dataset.range(5)
    options = options_lib.Options()
    options.experimental_slack = True
    options.experimental_optimization.apply_default_optimizations = False
    dataset = dataset.with_options(options)
    dataset = self.graphRoundTrip(dataset)
    result = self._get_options(dataset)
    self.assertTrue(result.experimental_slack)
    # Explicitly check that flag is False since assertFalse allows None
    self.assertIs(
        result.experimental_optimization.apply_default_optimizations, False)

  @combinations.generate(combinations.times(
      test_base.default_test_combinations(),
      combinations.combine(map_parallelization=[True, False])))
  def testOptionsGraphRoundTripOptimization(self, map_parallelization):
    # [FUNCTION DOC]
    # Scenario: a dataset with the map-parallelization option set to the parameter (true or
    # false) is graph-round-tripped, then a map is added after asserting what the next
    # transformation is. Expected: a parallel map when enabled, a plain map when disabled, and
    # the output is the squares zero to twenty-five.
    dataset = dataset_ops.Dataset.range(6)
    options = options_lib.Options()
    options.experimental_optimization.map_parallelization = (
        map_parallelization)
    dataset = dataset.with_options(options)
    dataset = self.graphRoundTrip(dataset)
    expected = "ParallelMap" if map_parallelization else "Map"
    dataset = dataset.apply(testing.assert_next([expected]))
    dataset = dataset.map(lambda x: x*x)
    self.assertDatasetProduces(dataset, expected_output=[0, 1, 4, 9, 16, 25])

  @combinations.generate(test_base.default_test_combinations())
  def testName(self):
    # [FUNCTION DOC]
    # Scenario: applying options with an explicit name to a single-element dataset. Expected: the
    # dataset still produces the element 42.
    dataset = dataset_ops.Dataset.from_tensors(42)
    options = options_lib.Options()
    dataset = dataset.with_options(options, name="options")
    self.assertDatasetProduces(dataset, [42])


if __name__ == "__main__":
  test.main()
