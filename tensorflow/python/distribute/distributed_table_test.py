# [MODULE DOC]
# Multi-process tests for distributed lookup tables under the parameter server strategy (version
# two) with a cluster coordinator. A shared in-process cluster of two workers and three parameter
# servers is created once for the test class. The tests cover creating static hash tables (from a
# text file or from key/value tensors) inside and outside the strategy scope, copying a
# distributed table, using tables inside dataset functions, map functions, step functions and
# nested while loops, capturing tables in saved-model-style modules, and saving/loading models
# with and without a strategy. The main entry point enables version-two behaviour and runs the
# multi-process test runner.
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for distributed_table."""

import copy
import os

from absl.testing import parameterized

from tensorflow.python.compat import v2_compat
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute import ps_values
from tensorflow.python.distribute.coordinator import cluster_coordinator as coordinator_lib
from tensorflow.python.distribute.coordinator import coordinator_context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.saved_model import load as tf_load
from tensorflow.python.saved_model import save as tf_save


source_combination = combinations.combine(source=["textfile", "keyvaluetensor"])


class DistributedTableTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests distributed hash tables with the parameter server strategy and cluster coordinator.
  # Shared fixture: a class-level in-process cluster (two workers, three parameter servers, gRPC)
  # and its cluster resolver, created in the class setup and stopped in the class teardown.
  # Most tests are parameterized over the table initializer source: a text file or key/value
  # tensors. Also defines a nested module class used for capture and save/load tests.

  @classmethod
  def setUpClass(cls):
    # [FUNCTION DOC]
    # Class-level setup: creates a multi-process cluster with two workers and three parameter
    # servers using gRPC, and stores the cluster and its resolver on the class. Side effect:
    # spawns processes.
    super(DistributedTableTest, cls).setUpClass()
    cls.cluster = multi_worker_test_base.create_multi_process_cluster(
        num_workers=2, num_ps=3, rpc_layer="grpc")
    cls.cluster_resolver = cls.cluster.cluster_resolver

  @classmethod
  def tearDownClass(cls):
    # [FUNCTION DOC]
    # Class-level teardown: runs the parent teardown and then stops the shared cluster.
    super(DistributedTableTest, cls).tearDownClass()
    cls.cluster.stop()

  def make_initializer(self, init_source, vals):
    # [FUNCTION DOC]
    # HELPER: builds a table initializer for the tests.
    # Parameters: init_source, either the text-file name or the key-value-tensor name; vals, a
    # list of integer values. For the text file source, writes the values one per line into a
    # file in the temporary directory and returns a text-file initializer keyed by line number
    # with whole-line values (64-bit integers). For the key-value source returns an initializer
    # mapping indices zero.. to the values. Raises a value error for any other source name.
    if init_source == "textfile":
      file = os.path.join(self.get_temp_dir(), "text_file_initializer")
      with open(file, "w") as f:
        f.write("\n".join(str(v) for v in vals) + "\n")
      return lookup_ops.TextFileInitializer(
          filename=file,
          key_dtype=dtypes.int64,
          key_index=lookup_ops.TextFileIndex.LINE_NUMBER,
          value_dtype=dtypes.int64,
          value_index=lookup_ops.TextFileIndex.WHOLE_LINE)
    elif init_source == "keyvaluetensor":
      keys_tensor = constant_op.constant(
          list(range(len(vals))), dtype=dtypes.int64)
      vals_tensor = constant_op.constant(vals, dtype=dtypes.int64)
      return lookup_ops.KeyValueTensorInitializer(keys_tensor, vals_tensor)
    else:
      raise ValueError("Unrecognized init_source: " + init_source)

  def createStaticHashTable(self,
                            init_source=None,
                            vals=None,
                            default_value=None,
                            initializer=None):
    # [FUNCTION DOC]
    # HELPER: creates a static hash table. Parameters (all optional): init_source and vals, used
    # to build an initializer when none is given; default_value, the value returned for missing
    # keys; initializer, a ready-made initializer that takes precedence. Returns the table.
    # Inside a strategy scope this yields a distributed table.
    if not initializer:
      initializer = self.make_initializer(init_source, vals)
    return lookup_ops.StaticHashTable(
        initializer=initializer, default_value=default_value)

  def makeDatasetFromTensorWithoutUsingResource(self, input_context, tensor):
    # [FUNCTION DOC]
    # HELPER: builds a dataset for use inside a dataset function. Takes the input context and a
    # tensor. Repeats the tensor forever, batches it with the per-replica share of a global batch
    # of 24 (dropping remainders), shards by the context's pipeline count and id, prefetches two
    # batches, and returns the dataset.
    """Returns a dataset made from `tensor`. To be called in a dataset_fn."""
    global_batch_size = 24
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)
    dataset = dataset_ops.DatasetV2.from_tensors(tensor).repeat().batch(
        batch_size, drop_remainder=True)
    dataset = dataset.shard(input_context.num_input_pipelines,
                            input_context.input_pipeline_id)
    dataset = dataset.prefetch(2)  # This prefetches 2 batches per device.
    return dataset

  @combinations.generate(source_combination)
  def testCreateDistributedTableInScope(self, source):
    # [FUNCTION DOC]
    # Scenario: with a parameter server strategy and coordinator, create a static hash table in
    # scope. Asserts it is a distributed table of size three and that a lookup on the
    # coordinator returns the stored values for present keys and the default (-2) for a missing
    # key.
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver)

    coordinator_lib.ClusterCoordinator(strategy=strategy)

    with strategy.scope():
      lookuptable = self.createStaticHashTable(
          init_source=source, vals=[0, 1, 2], default_value=-2)

    self.assertIsInstance(lookuptable, ps_values.DistributedTable)
    self.assertEqual(self.evaluate(lookuptable.size()), 3)

    # Lookup on the coordinator.
    output = lookuptable.lookup(
        constant_op.constant([0, 1, -1], dtype=dtypes.int64))
    self.assertAllEqual([0, 1, -2], output)
    self.assertEqual(lookuptable.size(), 3)

  @combinations.generate(source_combination)
  def testCopyDistributedTable(self, source):
    # [FUNCTION DOC]
    # Scenario: shallow-copy a distributed table created in scope. Asserts the copy has the same
    # attribute mapping as the original (no new coordinator or distributed tables are created).
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver)

    coordinator_lib.ClusterCoordinator(strategy=strategy)

    with strategy.scope():
      lookuptable = self.createStaticHashTable(
          init_source=source, vals=[0, 1, 2], default_value=-2)

    new_table = copy.copy(lookuptable)
    # No new coordinator instance or distributed tables are created.
    self.assertDictEqual(lookuptable.__dict__, new_table.__dict__)

  @combinations.generate(source_combination)
  def testCreateLookupInDatasetFnUnderScope(self, source):
    # [FUNCTION DOC]
    # Scenario: inside the strategy scope, a dataset function creates its own hash table and
    # looks up an out-of-range key. Asserts the table created inside the dataset function is not
    # a distributed table. The dataset is distributed per worker, ten steps are scheduled, and
    # each fetched sum equals minus forty-eight (default value minus two times batch size 24).
    # TODO(wxinyi): Warn the user of the inefficiency of this workflow (i.e.
    # creating `StaticHashTable` inside a `@tf.function`-wrapped `dataset_fn` to
    # be distributed with `distribute_datasets_from_function` and
    # `create_per_worker_dataset`.
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver)

    coordinator = coordinator_lib.ClusterCoordinator(strategy=strategy)

    with strategy.scope():

      def dataset_fn(input_context):
        # [FUNCTION DOC]
        # Dataset function: creates a hash table, asserts it is not a distributed table, looks
        # up an out-of-range key and returns a dataset built from that lookup result.
        some_out_of_range_tensor = constant_op.constant(10, dtype=dtypes.int64)
        lookuptable = self.createStaticHashTable(
            init_source=source, vals=[0, 1, 2], default_value=-2)

        self.assertNotIsInstance(lookuptable, ps_values.DistributedTable)

        generation_tensor = lookuptable.lookup(some_out_of_range_tensor)
        dataset = self.makeDatasetFromTensorWithoutUsingResource(
            input_context, generation_tensor)
        return dataset

      @def_function.function
      def per_worker_dataset_fn():
        # [FUNCTION DOC]
        # Compiled function returning the strategy-distributed dataset made from the dataset
        # function.
        return strategy.distribute_datasets_from_function(dataset_fn)

      per_worker_dataset = coordinator.create_per_worker_dataset(
          per_worker_dataset_fn)
      per_worker_iterator = iter(per_worker_dataset)

      @def_function.function
      def worker_fn(iterator):
        # [FUNCTION DOC]
        # Compiled worker step: takes the next batch from the iterator and returns its sum.
        return math_ops.reduce_sum(next(iterator))

      result = []
      for _ in range(10):
        result.append(
            coordinator.schedule(worker_fn, args=(per_worker_iterator,)))

      for r in result:
        returned_input = r.fetch()
        self.assertAllClose(-48, returned_input)

  @combinations.generate(source_combination)
  def testAccessingResourceHandleInDatasetFnWithoutMap(self, source):
    # [FUNCTION DOC]
    # Scenario: a table created in scope is looked up directly (no map) inside a dataset
    # function. Asserts the captured table is a distributed table, and that ten scheduled steps
    # each return minus forty-eight.

    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver)

    coordinator = coordinator_lib.ClusterCoordinator(strategy=strategy)

    with strategy.scope():
      lookuptable = self.createStaticHashTable(
          init_source=source, vals=[0, 1, 2], default_value=-2)

    def dataset_fn(input_context):
      # [FUNCTION DOC]
      # Dataset function: asserts the outer table is a distributed table, looks up an
      # out-of-range key in it and returns a dataset built from the result.
      some_out_of_range_tensor = constant_op.constant(10, dtype=dtypes.int64)

      self.assertIsInstance(lookuptable, ps_values.DistributedTable)

      generation_tensor = lookuptable.lookup(some_out_of_range_tensor)
      dataset = self.makeDatasetFromTensorWithoutUsingResource(
          input_context, generation_tensor)
      return dataset

    @def_function.function
    def per_worker_dataset_fn():
      # [FUNCTION DOC]
      # Compiled function returning the strategy-distributed dataset made from the dataset function.
      return strategy.distribute_datasets_from_function(dataset_fn)

    per_worker_dataset = coordinator.create_per_worker_dataset(
        per_worker_dataset_fn)
    per_worker_iterator = iter(per_worker_dataset)

    @def_function.function
    def worker_fn(iterator):
      # [FUNCTION DOC]
      # Compiled worker step: takes the next batch from the iterator and returns its sum.
      return math_ops.reduce_sum(next(iterator))

    result = []
    for _ in range(10):
      result.append(
          coordinator.schedule(worker_fn, args=(per_worker_iterator,)))

    for r in result:
      returned_input = r.fetch()
      self.assertAllClose(-48, returned_input)

  @combinations.generate(
      combinations.combine(
          source=["textfile", "keyvaluetensor"],
          create_datasets_under_scope=[True, False],
          using_dataset_instance_not_function=[True, False],
          create_per_worker_dataset_takes_instance=[True, False]))
  def testCreateTableUnderScopeCombo(self, source,
                                     create_datasets_under_scope,
                                     using_dataset_instance_not_function,
                                     create_per_worker_dataset_takes_instance):
    # [FUNCTION DOC]
    # Scenario (combinations of table source, whether datasets are created under scope, whether a
    # dataset instance or a dataset function is distributed, and whether the per-worker dataset
    # creation takes an instance or a function): create the table in scope, build a dataset that
    # maps lookups over repeated batches, distribute it, schedule ten steps and assert each step
    # sum is minus twenty-four.
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver)

    coordinator = coordinator_lib.ClusterCoordinator(strategy=strategy)

    with strategy.scope():
      lookup_table = self.createStaticHashTable(
          init_source=source, vals=[0, 1, 2], default_value=-2)

    if using_dataset_instance_not_function:

      def per_worker_dataset_fn():
        # [FUNCTION DOC]
        # Builds a repeated, batched dataset, maps the table lookup over it and returns it
        # distributed with the instance-based API.
        dataset = dataset_ops.DatasetV2.from_tensors(
            constant_op.constant([0, 1, 3], dtype=dtypes.int64))
        dataset = dataset.repeat().batch(24, drop_remainder=True).prefetch(2)
        dataset = dataset.map(lookup_table.lookup)

        return strategy.experimental_distribute_dataset(dataset)

    else:

      def per_worker_dataset_fn():
        # [FUNCTION DOC]
        # Defines a dataset function and returns it distributed with the function-based API.
        def dataset_fn(input_context):
          # [FUNCTION DOC]
          # Dataset function: batches and shards repeated data per the input context, prefetches
          # two batches, maps the table lookup and returns the dataset.
          batch_size = input_context.get_per_replica_batch_size(24)
          dataset = dataset_ops.DatasetV2.from_tensors(
              constant_op.constant([0, 1, 3], dtype=dtypes.int64))
          dataset = dataset.repeat().batch(batch_size, drop_remainder=True)
          dataset = dataset.shard(input_context.num_input_pipelines,
                                  input_context.input_pipeline_id)
          dataset = dataset.prefetch(2)  # This prefetches 2 batches per device.
          dataset = dataset.map(lookup_table.lookup)
          return dataset
        return strategy.distribute_datasets_from_function(dataset_fn)

    if create_datasets_under_scope:
      with strategy.scope():
        if create_per_worker_dataset_takes_instance:
          per_worker_dataset = coordinator.create_per_worker_dataset(
              per_worker_dataset_fn())
        else:
          per_worker_dataset = coordinator.create_per_worker_dataset(
              per_worker_dataset_fn)
        per_worker_iterator = iter(per_worker_dataset)

    else:
      if create_per_worker_dataset_takes_instance:
        per_worker_dataset = coordinator.create_per_worker_dataset(
            per_worker_dataset_fn())
      else:
        per_worker_dataset = coordinator.create_per_worker_dataset(
            per_worker_dataset_fn)
      per_worker_iterator = iter(per_worker_dataset)

    @def_function.function
    def worker_fn(iterator):
      # [FUNCTION DOC]
      # Compiled worker step: takes the next batch from the iterator and returns its sum.
      return math_ops.reduce_sum(next(iterator))

    result = []
    for _ in range(10):
      result.append(
          coordinator.schedule(worker_fn, args=(per_worker_iterator,)))

    for r in result:
      returned_input = r.fetch()
      self.assertAllClose(-24, returned_input)

  @combinations.generate(
      combinations.combine(
          source=["textfile", "keyvaluetensor"],
          create_datasets_under_scope=[True, False],
          using_dataset_instance_not_function=[True, False],
          create_per_worker_dataset_takes_instance=[True, False]))
  def testCreateTableInDatasetCombo(self, source, create_datasets_under_scope,
                                    using_dataset_instance_not_function,
                                    create_per_worker_dataset_takes_instance):
    # [FUNCTION DOC]
    # Scenario (same combinations as the under-scope variant): the table is created inside the
    # dataset building code rather than outside. The combination of a dataset instance with a
    # per-worker creation function is skipped (known serialization failure). When datasets are
    # created under scope, asserts the table kind: a distributed table when created while
    # building a distributed dataset instance, and a plain static hash table (not distributed)
    # when created inside a dataset function. Then schedules ten steps and asserts each sum is
    # minus twenty-four.

    if using_dataset_instance_not_function and (
        not create_per_worker_dataset_takes_instance):
      # This is the case that uses the `experimental_distribute_dataset` API to
      # distribute dataset (instead of the `distribute_datasets_from_function`
      # API), and passes `create_per_worker_dataset` a function that returns
      # the distributed dataset (instead of passing it the distributed dataset
      # directly).
      # TODO(b/201775366): evaluate whether we need to handle this case
      self.skipTest("Failed to serialize the input pipeline graph")

    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver)
    coordinator = coordinator_lib.ClusterCoordinator(strategy=strategy)

    if using_dataset_instance_not_function:

      def per_worker_dataset_fn():
        # [FUNCTION DOC]
        # Creates the table, asserts it is a distributed table when datasets are created under
        # scope, builds a repeated batched dataset mapped with the lookup and returns it
        # distributed with the instance-based API.
        # If this line is being called under strategy.scope(), it becomes a
        # DistributedTable. Interestingly, after
        # `experimental_distribute_dataset` serializes the dataset on chief and
        # deserializes it on workers, `lookup_table` becomes a
        # RestoredDistributedTable instead of a DistributedTable. And when it’s
        # `resource_handle` is being accessed on the worker, it does not detect
        # a DispatchContext, so it returns the restored resource handle,
        # which is also the one on the local worker. The LookupTableFindV2 ops
        # is on the local worker, too.
        lookup_table = self.createStaticHashTable(
            init_source=source, vals=[0, 1, 2], default_value=-2)

        if create_datasets_under_scope:
          self.assertIsInstance(lookup_table, ps_values.DistributedTable)

        dataset = dataset_ops.DatasetV2.from_tensors(
            constant_op.constant([0, 1, 3], dtype=dtypes.int64))
        dataset = dataset.repeat().batch(24, drop_remainder=True).prefetch(2)
        dataset = dataset.map(lookup_table.lookup)

        return strategy.experimental_distribute_dataset(dataset)

    else:

      def per_worker_dataset_fn():
        # [FUNCTION DOC]
        # Defines a dataset function that creates its own table and returns it distributed with
        # the function-based API.

        def dataset_fn(input_context):
          # [FUNCTION DOC]
          # Dataset function: creates a table (a plain static hash table, not a distributed
          # one, when datasets are created under scope), batches, shards and prefetches repeated
          # data, maps the lookup over it and returns the dataset.
          # When we're wrapping the initialization of a StaticHashTable inside a
          # `dataset_fn` to be distributed with
          # `distribute_datasets_from_function`, no matter it's called under
          # strategy.scope() or not, this call creates a StaticHashTable on
          # chief instead of a DistributedTable on chief and workers.
          # And correspondingly, LookupTableFindV2 ops is on chief and there are
          # send-recv communication for the lookup.
          lookup_table = self.createStaticHashTable(
              init_source=source, vals=[0, 1, 2], default_value=-2)
          if create_datasets_under_scope:
            self.assertIsInstance(lookup_table, lookup_ops.StaticHashTable)
            self.assertNotIsInstance(lookup_table, ps_values.DistributedTable)

          batch_size = input_context.get_per_replica_batch_size(24)
          dataset = dataset_ops.DatasetV2.from_tensors(
              constant_op.constant([0, 1, 3], dtype=dtypes.int64))
          dataset = dataset.repeat().batch(batch_size, drop_remainder=True)
          dataset = dataset.shard(input_context.num_input_pipelines,
                                  input_context.input_pipeline_id)
          dataset = dataset.prefetch(2)  # This prefetches 2 batches per device.
          dataset = dataset.map(lookup_table.lookup)
          return dataset

        return strategy.distribute_datasets_from_function(dataset_fn)

    if create_datasets_under_scope:
      with strategy.scope():
        if create_per_worker_dataset_takes_instance:
          per_worker_dataset = coordinator.create_per_worker_dataset(
              per_worker_dataset_fn())
        else:
          per_worker_dataset = coordinator.create_per_worker_dataset(
              per_worker_dataset_fn)
        per_worker_iterator = iter(per_worker_dataset)

    else:
      if create_per_worker_dataset_takes_instance:
        per_worker_dataset = coordinator.create_per_worker_dataset(
            per_worker_dataset_fn())
      else:
        per_worker_dataset = coordinator.create_per_worker_dataset(
            per_worker_dataset_fn)
      per_worker_iterator = iter(per_worker_dataset)

    @def_function.function
    def worker_fn(iterator):
      # [FUNCTION DOC]
      # Compiled worker step: takes the next batch from the iterator and returns its sum.
      return math_ops.reduce_sum(next(iterator))

    result = []
    for _ in range(10):
      result.append(
          coordinator.schedule(worker_fn, args=(per_worker_iterator,)))

    for r in result:
      returned_input = r.fetch()
      self.assertAllClose(-24, returned_input)

  @combinations.generate(source_combination)
  def testAccessingTableInStepFunction(self, source):
    # [FUNCTION DOC]
    # Scenario: a table created in scope is used inside a replica function run by the strategy
    # within a worker step function. Ten steps are scheduled and joined; each fetched sum must
    # equal minus twenty-four.
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver)
    coordinator = coordinator_lib.ClusterCoordinator(strategy=strategy)

    with strategy.scope():
      lookup_table = self.createStaticHashTable(
          init_source=source, vals=[0, 1, 2], default_value=-2)

    dataset = (
        dataset_ops.DatasetV2.from_tensors(
            constant_op.constant([0, 1, 3], dtype=dtypes.int64)).repeat().batch(
                24, drop_remainder=True).prefetch(2))
    dataset = dataset.map(lookup_table.lookup)

    distributed_dataset = strategy.experimental_distribute_dataset(dataset)
    distributed_dataset = coordinator.create_per_worker_dataset(
        distributed_dataset)

    @def_function.function
    def worker_fn(iterator):
      # [FUNCTION DOC]
      # Compiled worker step: runs the replica function through the strategy on the next batch
      # and returns the results.

      def replica_fn(inputs):
        # [FUNCTION DOC]
        # Replica function: looks up the input keys in the table and returns the sum.
        return math_ops.reduce_sum(lookup_table.lookup(inputs))

      all_results = strategy.run(replica_fn, args=(next(iterator),))
      return all_results

    steps_per_epoch = 10
    distributed_iterator = iter(distributed_dataset)
    result = []
    for _ in range(steps_per_epoch):

      result.append(
          coordinator.schedule(worker_fn, args=(distributed_iterator,)))

    coordinator.join()

    for r in result:
      returned_input = r.fetch()
      self.assertAllClose(-24, returned_input)

  @combinations.generate(source_combination)
  def testAccessingResourceHandleInDatasetFnWithMapFnDefinedOutside(
      self, source):
    # [FUNCTION DOC]
    # Scenario: a table created in scope is used by a map function defined outside the dataset
    # function and applied to a dataset inside it. Ten scheduled steps must each return minus
    # forty-eight.
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver)

    coordinator = coordinator_lib.ClusterCoordinator(strategy=strategy)

    with strategy.scope():
      lookuptable = self.createStaticHashTable(
          init_source=source, vals=[0, 1, 2], default_value=-2)

    def map_fn(vals):
      # [FUNCTION DOC]
      # Map function: looks up the given keys in the outer table and returns the values.
      return lookuptable.lookup(vals)

    def dataset_fn(input_context):
      # [FUNCTION DOC]
      # Dataset function: builds a dataset from a constant of three keys and maps the outer
      # lookup function over it.
      generation_tensor = constant_op.constant([0, 1, 3], dtype=dtypes.int64)
      dataset = self.makeDatasetFromTensorWithoutUsingResource(
          input_context, generation_tensor)
      dataset = dataset.map(map_fn)
      return dataset

    @def_function.function
    def per_worker_dataset_fn():
      # [FUNCTION DOC]
      # Compiled function returning the strategy-distributed dataset made from the dataset function.
      return strategy.distribute_datasets_from_function(dataset_fn)

    per_worker_dataset = coordinator.create_per_worker_dataset(
        per_worker_dataset_fn)
    per_worker_iterator = iter(per_worker_dataset)

    @def_function.function
    def worker_fn(iterator):
      # [FUNCTION DOC]
      # Compiled worker step: takes the next batch from the iterator and returns its sum.
      return math_ops.reduce_sum(next(iterator))

    result = []
    for _ in range(10):
      # batch_size == 24 and each input is [0, 1, -2]
      result.append(
          coordinator.schedule(worker_fn, args=(per_worker_iterator,)))

    for r in result:
      returned_input = r.fetch()
      self.assertAllClose(-24, returned_input)

  class Model(module.Module):
    # [CLASS DOC]
    # HELPER: small module holding a static hash table and a compiled lookup method, used by
    # the capture and save/load tests. The table is built in the constructor from a text file
    # initializer (when the source is the text file) or a key-value tensor initializer.

    def __init__(self, init_source, filepath):
      # [FUNCTION DOC]
      # Builds the table. Parameters: init_source, the initializer source name (the text-file
      # name selects a file initializer, anything else a key-value tensor initializer);
      # filepath, where the text file is written. Values are zero, one, two; the default value is
      # minus two. Side effect: writes the file in the text-file case. Stores the initializer and
      # the table on the instance.
      vals = [0, 1, 2]
      if init_source == "textfile":

        with open(filepath, "w") as f:
          f.write("\n".join(str(v) for v in vals) + "\n")

        self.initializer = lookup_ops.TextFileInitializer(
            filepath, dtypes.int64, lookup_ops.TextFileIndex.LINE_NUMBER,
            dtypes.int64, lookup_ops.TextFileIndex.WHOLE_LINE)
      else:
        keys_tensor = constant_op.constant(
            list(range(len(vals))), dtype=dtypes.int64)
        vals_tensor = constant_op.constant(vals, dtype=dtypes.int64)
        self.initializer = lookup_ops.KeyValueTensorInitializer(
            keys_tensor, vals_tensor)

      self.table = lookup_ops.StaticHashTable(
          self.initializer, default_value=-2)

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec(None, dtypes.int64)])
    def use_table(self, x):
      # [FUNCTION DOC]
      # Compiled lookup method (signature: a 64-bit integer tensor of any shape). Returns the
      # table values for the given keys.
      return self.table.lookup(x)

  def verifyWorkerLocalInstance(self, coordinator, model):
    # [FUNCTION DOC]
    # HELPER: assertion helper. For each worker of the coordinator's cluster, enters that
    # worker's dispatch context, gets the model lookup method's captured resource inputs and
    # asserts there is at least one and each lives on that worker's CPU device.
    # assert capturing a worker-local resource on each worker
    for worker in coordinator._cluster.workers:
      with coordinator_context.with_dispatch_context(worker):
        captures = model.use_table.get_concrete_function().captured_inputs
        resource_capture = [t for t in captures if t.dtype == dtypes.resource]
        self.assertNotEmpty(resource_capture)
        for capture in resource_capture:
          self.assertEqual(
              capture.device,
              device_util.canonicalize("/CPU:0", default=worker.device_name))

  @combinations.generate(source_combination)
  def testInModelAndCapture(self, source):
    # [FUNCTION DOC]
    # Scenario: build the model without a strategy and check its lookup function has two
    # external captures including the table handle and no deferred captures. Then build it in
    # strategy scope and check one external capture, the table handle no longer captured
    # directly, and non-empty deferred captures. Finally verifies worker-local resource captures.

    file_path = os.path.join(self.get_temp_dir(), "text_file_initializer")

    model = self.Model(source, file_path)
    func_captures = model.use_table.get_concrete_function(
    ).graph.external_captures
    self.assertLen(func_captures, 2)
    self.assertTrue(
        any(model.table.resource_handle is t for t in func_captures))
    deferred_captures = model.use_table.get_concrete_function(
    ).graph.deferred_external_captures
    self.assertEmpty(deferred_captures)

    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver)
    coordinator = coordinator_lib.ClusterCoordinator(strategy)
    with strategy.scope():
      distributed_model = self.Model("value", file_path)
    func_captures = distributed_model.use_table.get_concrete_function(
    ).graph.external_captures
    # One less external_capture, since the table handle becomes a closure in the
    # deferred_external_capture
    self.assertLen(func_captures, 1)
    self.assertFalse(
        any(model.table.resource_handle is t for t in func_captures))
    deferred_captures = distributed_model.use_table.get_concrete_function(
    ).graph.deferred_external_captures
    self.assertNotEmpty(deferred_captures)

    self.verifyWorkerLocalInstance(coordinator, distributed_model)

  @combinations.generate(source_combination)
  def testLookupInNestedTFWhileLoop(self, source):
    # [FUNCTION DOC]
    # Scenario: a model built in scope is looked up inside loops nested in a replica function
    # (ten iterations) which runs inside a step function (ten iterations). Ten steps are
    # scheduled; each result must equal minus two thousand four hundred.
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver)
    coordinator = coordinator_lib.ClusterCoordinator(strategy=strategy)

    file_path = os.path.join(self.get_temp_dir(), "text_file_initializer")
    with strategy.scope():
      model = self.Model(source, file_path)

    @def_function.function
    def replica_fn(batch_data):
      # [FUNCTION DOC]
      # Compiled replica function: ten times adds the sum of the model lookup of the batch to a
      # zero accumulator and returns it.
      replica_result = array_ops.zeros(shape=(), dtype=dtypes.int64)
      for _ in math_ops.range(10):
        replica_result += math_ops.reduce_sum(model.use_table(batch_data))
      return replica_result

    @def_function.function
    def step_fn(iterator):
      # [FUNCTION DOC]
      # Compiled step: ten times runs the replica function through the strategy on the next
      # batch and accumulates the result; returns the total.

      step_result = array_ops.zeros(shape=(), dtype=dtypes.int64)
      for _ in math_ops.range(10):
        step_result += strategy.run(replica_fn, args=(next(iterator),))

      return step_result

    dataset = (
        dataset_ops.DatasetV2.from_tensors(
            constant_op.constant([0, 1, 3], dtype=dtypes.int64)).repeat().batch(
                24, drop_remainder=True).prefetch(2))
    distributed_dataset = coordinator.create_per_worker_dataset(
        strategy.experimental_distribute_dataset(dataset))

    results = []
    for _ in range(10):
      results.append(
          coordinator.schedule(step_fn, args=(iter(distributed_dataset),)))

    coordinator.join()

    for r in results:
      self.assertAllClose(-2400, r.fetch())

  @combinations.generate(source_combination)
  def testDistributeTableSaveAndServe(self, source):
    # [FUNCTION DOC]
    # Scenario: build a model in strategy scope, save it, and load it without a strategy.
    # Asserts the loaded lookup function has two external captures and no deferred captures, and
    # that looking up keys zero, one and three returns zero, one and minus two.
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver)
    file_path = os.path.join(self.get_temp_dir(), "text_file_initializer")
    with strategy.scope():
      model = self.Model(source, file_path)

    model_dir = self.get_temp_dir()
    tf_save.save(model, model_dir)

    loaded_without_strategy = tf_load.load(model_dir)
    loaded_func_captures_without_strategy = (
        loaded_without_strategy.use_table.get_concrete_function().graph
        .external_captures)
    loaded_func_deferred_captures_without_strategy = (
        loaded_without_strategy.use_table.get_concrete_function().graph
        .deferred_external_captures)
    self.assertLen(loaded_func_captures_without_strategy, 2)
    self.assertEmpty(loaded_func_deferred_captures_without_strategy)

    self.assertAllEqual(
        loaded_without_strategy.use_table(
            constant_op.constant([0, 1, 3], dtype=dtypes.int64)), [0, 1, -2])

  @combinations.generate(source_combination)
  def testDistributeTableSaveAndLoadUnderStrategy(self, source):
    # [FUNCTION DOC]
    # Scenario: build and save a model in strategy scope, then load it in strategy scope.
    # Asserts one external capture, non-empty deferred captures, that the loaded table is a
    # distributed table, exactly one captured resource input, and worker-local resource captures.
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver)
    coordinator = coordinator_lib.ClusterCoordinator(strategy)
    file_path = os.path.join(self.get_temp_dir(), "text_file_initializer")
    with strategy.scope():
      model = self.Model(source, file_path)
    model_dir = self.get_temp_dir()
    tf_save.save(model, model_dir)

    with strategy.scope():
      loaded = tf_load.load(model_dir)

    loaded_func_captures = (
        loaded.use_table.get_concrete_function().graph.external_captures)
    loaded_func_deferred_captures = (
        loaded.use_table.get_concrete_function().graph
        .deferred_external_captures)
    # Compared with loading without strategy, there is one less
    # external_capture, since the captured table handle has been swapped to a
    # closure in the deferred_external_capture
    self.assertLen(loaded_func_captures, 1)
    self.assertNotEmpty(loaded_func_deferred_captures)

    self.assertIsInstance(loaded.table, ps_values.DistributedTable)

    self.assertLen([
        t for t in loaded.use_table.get_concrete_function().captured_inputs
        if t.dtype == dtypes.resource
    ], 1)

    self.verifyWorkerLocalInstance(coordinator, loaded)


if __name__ == "__main__":
  v2_compat.enable_v2_behavior()
  multi_process_runner.test_main()
