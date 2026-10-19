# [MODULE DOC]
# Unit tests for the sharded variable module: the fixed, minimum-size and maximum-size
# partitioners, and the ShardedVariable container (assign and scatter ops, sparse reads,
# conversion to tensor, checkpoint save and restore across different shard counts, SavedModel
# export, use as a function input, flattening, module tracking, embedding lookups, slicing,
# operator overloads, numpy conversion) plus save/load tests under a parameter server strategy.
# Creates one shared test cluster description at import time (no workers or servers started
# until the save/load tests set up their cluster resolver).
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for ShardedVariable."""

import os

from absl.testing import parameterized
import numpy as np
from tensorflow.python.checkpoint import checkpoint as util
from tensorflow.python.client import session as session_lib
from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute import sharded_variable
from tensorflow.python.distribute.cluster_resolver import cluster_resolver as cluster_resolver_lib
from tensorflow.python.distribute.test_util import get_cluster_def
from tensorflow.python.distribute.test_util import TestClusterParams
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.trackable import autotrackable
from tensorflow.python.training import server_lib
from tensorflow.python.util import nest

# We create one cluster to share between tests. The cluster should be large
# enough to accommodate all the tests. Adjust the following constants as needed
# but be aware of resource limitations in OSS tests.
test_cluster_params = TestClusterParams(None, 2, 3)


def _load_and_run(
    model_dir,
    inputs,
    signature_key=signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY):
  # [FUNCTION DOC]
  # HELPER: test utility for the SavedModel export test. Inputs: model_dir (directory of a saved
  # model), inputs (mapping from signature input names to values) and signature_key (defaults to
  # the default serving signature). Loads the model into a fresh graph and session, feeds the
  # inputs to the signature's input tensors and fetches all signature outputs. Returns a mapping
  # from output names to computed values.
  """Load a SavedModel into a TF 1.x-style graph and run `signature_key`."""
  graph = ops.Graph()
  with graph.as_default(), session_lib.Session() as session:
    meta_graph_def = loader.load(session, [tag_constants.SERVING], model_dir)
    signature = meta_graph_def.signature_def[signature_key]
    feed_dict = {}
    for arg_name in inputs.keys():
      input_tensor = session.graph.get_tensor_by_name(
          signature.inputs[arg_name].name)
      feed_dict[input_tensor] = inputs[arg_name]
    output_dict = {}
    for output_name, output_tensor_info in signature.outputs.items():
      output_dict[output_name] = session.graph.get_tensor_by_name(
          output_tensor_info.name)
    return session.run(output_dict, feed_dict=feed_dict)


class PartitionerTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the fixed-shards, min-size and max-size partitioners; no shared fixtures.

  def test_fixed_shards_partitioner(self):
    # [FUNCTION DOC]
    # Fixed partitioner with two shards on a ten by three shape: expects partitions of two and
    # one.
    partitioner = sharded_variable.FixedShardsPartitioner(num_shards=2)
    got = partitioner(tensor_shape.TensorShape([10, 3]), dtypes.float32)
    self.assertAllEqual(got, [2, 1])

  def test_min_size_partitioner(self):
    # [FUNCTION DOC]
    # Min-size partitioner on a six by one float shape with four-byte minimum shards: with a
    # maximum of two shards expects two partitions; with a maximum of ten expects six.
    partitioner = sharded_variable.MinSizePartitioner(
        min_shard_bytes=4, max_shards=2)
    got = partitioner(tensor_shape.TensorShape([6, 1]), dtypes.float32)
    self.assertAllEqual(got, [2, 1])

    partitioner = sharded_variable.MinSizePartitioner(
        min_shard_bytes=4, max_shards=10)
    got = partitioner(tensor_shape.TensorShape([6, 1]), dtypes.float32)
    self.assertAllEqual(got, [6, 1])

  def test_max_size_partitioner(self):
    # [FUNCTION DOC]
    # Max-size partitioner on a six by one float shape: four-byte maximum shards gives six
    # partitions; adding a cap of two shards gives two; a 1024-byte maximum gives one.
    partitioner = sharded_variable.MaxSizePartitioner(max_shard_bytes=4)
    got = partitioner(tensor_shape.TensorShape([6, 1]), dtypes.float32)
    self.assertAllEqual(got, [6, 1])

    partitioner = sharded_variable.MaxSizePartitioner(
        max_shard_bytes=4, max_shards=2)
    got = partitioner(tensor_shape.TensorShape([6, 1]), dtypes.float32)
    self.assertAllEqual(got, [2, 1])

    partitioner = sharded_variable.MaxSizePartitioner(max_shard_bytes=1024)
    got = partitioner(tensor_shape.TensorShape([6, 1]), dtypes.float32)
    self.assertAllEqual(got, [1, 1])


class ShardedVariableTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the ShardedVariable container: construction, assign and scatter ops, reads,
  # checkpointing, SavedModel export, function inputs, embedding lookups, slicing and operators.
  # Parameterized for the scatter operation tests; no shared fixtures.

  def test_sharded_variable_simple(self):
    # [FUNCTION DOC]
    # Builds a sharded variable from two one-element variables named s and checks the shard
    # list, combined shape of two, dtype and name.
    v0 = variables_lib.Variable([0])
    v1 = variables_lib.Variable([1])
    s = sharded_variable.ShardedVariable([v0, v1], name='s')
    self.assertEqual(s.variables[0], v0)
    self.assertEqual(s.variables[1], v1)
    self.assertEqual(s.shape.as_list(), [2])
    self.assertEqual(s.dtype, v0.dtype)
    self.assertEqual(s.name, 's')

  def test_assign(self):
    # [FUNCTION DOC]
    # Assigns a four-row value to a three-shard variable (shards of 1, 2, 1 rows); expects each
    # shard to receive its rows and the call to return the sharded variable itself.
    v0 = variables_lib.Variable([[0, 0]])
    v1 = variables_lib.Variable([[1, 1], [2, 2]])
    v2 = variables_lib.Variable([[3, 3]])
    s = sharded_variable.ShardedVariable([v0, v1, v2])
    ret = s.assign([[4, 4], [5, 5], [6, 6], [7, 7]])
    self.assertAllEqual(self.evaluate(s.variables[0]), [[4, 4]])
    self.assertAllEqual(self.evaluate(s.variables[1]), [[5, 5], [6, 6]])
    self.assertAllEqual(self.evaluate(s.variables[2]), [[7, 7]])
    self.assertIs(ret, s)

  def test_assign_add(self):
    # [FUNCTION DOC]
    # Add-assigns a four-row delta to a three-shard variable; expects each shard to hold its
    # original rows plus the matching delta rows, and the sharded variable to be returned.
    v0 = variables_lib.Variable([[0, 0]])
    v1 = variables_lib.Variable([[1, 1], [2, 2]])
    v2 = variables_lib.Variable([[3, 3]])
    s = sharded_variable.ShardedVariable([v0, v1, v2])
    ret = s.assign_add([[1, 1], [1, 1], [2, 2], [2, 2]])
    self.assertAllEqual(self.evaluate(s.variables[0]), [[1, 1]])
    self.assertAllEqual(self.evaluate(s.variables[1]), [[2, 2], [4, 4]])
    self.assertAllEqual(self.evaluate(s.variables[2]), [[5, 5]])
    self.assertIs(ret, s)

  def test_assign_sub(self):
    # [FUNCTION DOC]
    # Subtract-assigns a four-row delta to a three-shard variable; expects each shard to hold its
    # original rows minus the matching delta rows (all-zero or small results), and the sharded
    # variable to be returned.
    v0 = variables_lib.Variable([[0, 0]])
    v1 = variables_lib.Variable([[1, 1], [2, 2]])
    v2 = variables_lib.Variable([[3, 3]])
    s = sharded_variable.ShardedVariable([v0, v1, v2])
    ret = s.assign_sub([[0, 0], [1, 1], [1, 1], [3, 3]])
    self.assertAllEqual(self.evaluate(s.variables[0]), [[0, 0]])
    self.assertAllEqual(self.evaluate(s.variables[1]), [[0, 0], [1, 1]])
    self.assertAllEqual(self.evaluate(s.variables[2]), [[0, 0]])
    self.assertIs(ret, s)

  def test_scatter_add_uneven_partition(self):
    # [FUNCTION DOC]
    # Scatter-adds six rows to a plain 32-row variable and to a sharded variable with shards of
    # 11, 11 and 10 rows (the "div" layout); expects equal contents. Repeats inside a compiled
    # function and expects equality again.
    v = variables_lib.Variable(array_ops.zeros((32, 1)))
    sparse_delta = indexed_slices.IndexedSlices(
        values=constant_op.constant([[0.], [1.], [2.], [3.], [4.], [5.]]),
        indices=constant_op.constant([0, 10, 11, 12, 30, 31]))

    v0 = variables_lib.Variable(array_ops.zeros((11, 1)))
    v1 = variables_lib.Variable(array_ops.zeros((11, 1)))
    v2 = variables_lib.Variable(array_ops.zeros((10, 1)))
    sv = sharded_variable.ShardedVariable([v0, v1, v2])

    v.scatter_add(sparse_delta)
    sv.scatter_add(sparse_delta)
    self.assertAllEqual(v, ops.convert_to_tensor(sv))

    @def_function.function
    def func():
      # [FUNCTION DOC]
      # Scatter-adds the delta to both the plain and the sharded variable.
      v.scatter_add(sparse_delta)
      sv.scatter_add(sparse_delta)

    func()
    self.assertAllEqual(v, ops.convert_to_tensor(sv))

  @parameterized.parameters('scatter_add', 'scatter_div', 'scatter_max',
                            'scatter_min', 'scatter_mul', 'scatter_sub',
                            'scatter_update')
  def test_scatter_ops_even_partition(self, op):
    # [FUNCTION DOC]
    # Parameterized over add, div, max, min, mul, sub and update. Input: op, the name of the
    # scatter operation. Applies it with named ops to a plain 30-row variable and to a sharded
    # variable of three ten-row shards using the same indexed slices (non-zero values so divide
    # works); expects equal contents, then repeats inside a compiled function.
    v = variables_lib.Variable(array_ops.zeros((30, 1)))
    # Make sure values does not contain 0 due to testing `scatter_div`!
    sparse_delta = indexed_slices.IndexedSlices(
        values=constant_op.constant([[1.], [2.], [3.], [4.], [5.]]),
        indices=constant_op.constant([0, 10, 12, 21, 22]))

    v0 = variables_lib.Variable(array_ops.zeros((10, 1)))
    v1 = variables_lib.Variable(array_ops.zeros((10, 1)))
    v2 = variables_lib.Variable(array_ops.zeros((10, 1)))
    sv = sharded_variable.ShardedVariable([v0, v1, v2])

    getattr(v, op)(sparse_delta, name='scatter_v')
    getattr(sv, op)(sparse_delta, name='scatter_sv')
    self.assertAllEqual(v, ops.convert_to_tensor(sv))

    @def_function.function
    def func():
      # [FUNCTION DOC]
      # Applies the chosen scatter operation to both the plain and the sharded variable.
      getattr(v, op)(sparse_delta, name='scatter_v')
      getattr(sv, op)(sparse_delta, name='scatter_sv')

    func()
    self.assertAllEqual(v, ops.convert_to_tensor(sv))

  def test_batch_scatter_update(self):
    # [FUNCTION DOC]
    # Batch scatter-updates six consecutive rows on a plain 32-row variable and on an 11/11/10
    # shard layout; expects equal contents, then repeats inside a compiled function.
    v = variables_lib.Variable(array_ops.zeros((32, 1)))
    sparse_delta = indexed_slices.IndexedSlices(
        values=constant_op.constant([[0.], [1.], [2.], [3.], [4.], [5.]]),
        indices=constant_op.constant([10, 11, 12, 13, 14, 15]))

    v0 = variables_lib.Variable(array_ops.zeros((11, 1)))
    v1 = variables_lib.Variable(array_ops.zeros((11, 1)))
    v2 = variables_lib.Variable(array_ops.zeros((10, 1)))
    sv = sharded_variable.ShardedVariable([v0, v1, v2])

    v.batch_scatter_update(sparse_delta)
    sv.batch_scatter_update(sparse_delta)
    self.assertAllEqual(v, ops.convert_to_tensor(sv))

    @def_function.function
    def func():
      # [FUNCTION DOC]
      # Batch scatter-updates both the plain and the sharded variable.
      v.batch_scatter_update(sparse_delta)
      sv.batch_scatter_update(sparse_delta)

    func()
    self.assertAllEqual(v, ops.convert_to_tensor(sv))

  def test_sparse_read(self):
    # [FUNCTION DOC]
    # Sparse-reads five sorted indices from a plain 30-row variable and from a three-shard
    # variable; expects equal results, both eagerly and inside a compiled function.
    v = variables_lib.Variable(array_ops.zeros((30, 1)))
    indices = constant_op.constant([0, 10, 12, 21, 22])

    v0 = variables_lib.Variable(array_ops.zeros((10, 1)))
    v1 = variables_lib.Variable(array_ops.zeros((10, 1)))
    v2 = variables_lib.Variable(array_ops.zeros((10, 1)))
    sv = sharded_variable.ShardedVariable([v0, v1, v2])

    self.assertAllEqual(v.sparse_read(indices), sv.sparse_read(indices))

    @def_function.function
    def func():
      # [FUNCTION DOC]
      # Returns the sparse reads of the plain and the sharded variable as a pair.
      return v.sparse_read(indices), sv.sparse_read(indices)

    got, expect = func()
    self.assertAllEqual(got, expect)

  def test_control_dep_on_assign(self):
    # [FUNCTION DOC]
    # Inside a compiled function, uses the result of assigning to a sharded variable as a control
    # dependency, both directly and wrapped in a group op; expects tracing and running to succeed
    # (no assertions beyond not raising).
    v0 = variables_lib.Variable([[0, 0]])
    v1 = variables_lib.Variable([[1, 1], [2, 2]])
    v2 = variables_lib.Variable([[3, 3]])
    s = sharded_variable.ShardedVariable([v0, v1, v2])

    @def_function.function
    def func():
      # [FUNCTION DOC]
      # Assigns to the sharded variable and uses the result as a control dependency (directly and
      # via a group op) for two constant tensors, returned as a pair.
      ret = s.assign([[4, 4], [5, 5], [6, 6], [7, 7]])
      with ops.control_dependencies([ret]):
        a = array_ops.ones((1, 1))
      with ops.control_dependencies([control_flow_ops.group(ret)]):
        b = array_ops.ones((1, 1))
      return a, b

    func()

  def test_convert_to_tensor(self):
    # [FUNCTION DOC]
    # Converts a three-shard variable to a tensor; expects the shards concatenated in order.
    v0 = variables_lib.Variable([[0, 0]])
    v1 = variables_lib.Variable([[1, 1], [2, 2]])
    v2 = variables_lib.Variable([[3, 3]])
    s = sharded_variable.ShardedVariable([v0, v1, v2])
    t = ops.convert_to_tensor(s)
    self.assertAllEqual(t, [[0, 0], [1, 1], [2, 2], [3, 3]])

  def test_save_restore(self):
    # [FUNCTION DOC]
    # Writes a checkpoint of a four-shard variable, changes the first shard, restores, and
    # expects the first shard to return to its original value.
    fname = os.path.join(self.get_temp_dir(), 'checkpoint')
    variables = [
        variables_lib.Variable([0]),
        variables_lib.Variable([1]),
        variables_lib.Variable([2]),
        variables_lib.Variable([3])
    ]
    s = sharded_variable.ShardedVariable(variables, name='s')

    cp = util.Checkpoint(s=s)
    self.assertEqual(self.evaluate(cp.s.variables[0]), [0])
    cp.write(fname)

    self.evaluate(cp.s.variables[0].assign([4]))
    self.assertEqual(self.evaluate(cp.s.variables[0]), [4])

    cp.restore(fname)
    # Tests that the original weights are restored.
    self.assertEqual(self.evaluate(cp.s.variables[0]), [0])

  def test_save_restore_different_partitions(self):
    # [FUNCTION DOC]
    # Saves a four-shard variable and restores it into a one-shard variable (expects all four
    # values in order), then assigns new values, saves again and restores into the four-shard
    # variable, expecting each shard to receive its slice.
    fname = os.path.join(self.get_temp_dir(), 'checkpoint')
    variables = [
        variables_lib.Variable([0]),
        variables_lib.Variable([1]),
        variables_lib.Variable([2]),
        variables_lib.Variable([3])
    ]
    s = sharded_variable.ShardedVariable(variables, name='s')

    cp = util.Checkpoint(s=s)
    cp.write(fname)

    variables2 = [variables_lib.Variable([0, 0, 0, 0])]
    s2 = sharded_variable.ShardedVariable(variables2, name='s')

    # Restore from 4 partitions into 1.
    cp2 = util.Checkpoint(s=s2)
    cp2.restore(fname)
    self.assertAllEqual(self.evaluate(cp2.s.variables[0]), [0, 1, 2, 3])

    self.evaluate(cp2.s.variables[0].assign([5, 10, 15, 20]))
    cp2.write(fname)

    # Restore 1 partition into 4.
    cp.restore(fname)
    self.assertEqual(self.evaluate(cp.s.variables[0]), [5])
    self.assertEqual(self.evaluate(cp.s.variables[1]), [10])
    self.assertEqual(self.evaluate(cp.s.variables[2]), [15])
    self.assertEqual(self.evaluate(cp.s.variables[3]), [20])

  def test_save_restore_4_to_2_partitions(self):
    # [FUNCTION DOC]
    # Saves a four-shard variable and restores into a two-shard variable; expects two shards
    # holding the first two and last two values respectively.
    fname = os.path.join(self.get_temp_dir(), 'checkpoint')
    variables = [
        variables_lib.Variable([0]),
        variables_lib.Variable([1]),
        variables_lib.Variable([2]),
        variables_lib.Variable([3])
    ]
    s = sharded_variable.ShardedVariable(variables, name='s')
    cp = util.Checkpoint(s=s)
    cp.write(fname)

    variables2 = [
        variables_lib.Variable([0, 0]),
        variables_lib.Variable([0, 0])
    ]
    s2 = sharded_variable.ShardedVariable(variables2, name='s')
    cp2 = util.Checkpoint(s=s2)
    cp2.restore(fname)
    # Assert that weights from the 4 partitions were loaded here.
    self.assertLen(cp2.s.variables, 2)
    self.assertAllEqual(self.evaluate(cp2.s.variables[0]), [0, 1])
    self.assertAllEqual(self.evaluate(cp2.s.variables[1]), [2, 3])

  def test_delayed_restore(self):
    # [FUNCTION DOC]
    # Saves a model holding a four-shard variable, restores into an empty model first, and only
    # then attaches a new four-shard variable; expects the deferred restore to fill every shard
    # with the saved values.
    fname = os.path.join(self.get_temp_dir(), 'checkpoint')
    model = autotrackable.AutoTrackable()
    variables = [
        variables_lib.Variable([0]),
        variables_lib.Variable([1]),
        variables_lib.Variable([2]),
        variables_lib.Variable([3])
    ]
    model.s = sharded_variable.ShardedVariable(variables)
    cp = util.Checkpoint(model=model)
    cp.write(fname)

    model2 = autotrackable.AutoTrackable()
    cp2 = util.Checkpoint(model=model2)
    cp2.restore(fname)
    variables2 = [
        variables_lib.Variable([0]),
        variables_lib.Variable([0]),
        variables_lib.Variable([0]),
        variables_lib.Variable([0])
    ]
    model2.s = sharded_variable.ShardedVariable(variables2)
    self.assertAllEqual(self.evaluate(model2.s.variables[0]), [0])
    self.assertAllEqual(self.evaluate(model2.s.variables[1]), [1])
    self.assertAllEqual(self.evaluate(model2.s.variables[2]), [2])
    self.assertAllEqual(self.evaluate(model2.s.variables[3]), [3])

  def test_delayed_restore_4_to_2_partitions(self):
    # [FUNCTION DOC]
    # Same as the delayed restore test but the variable attached after restoring has two shards;
    # expects the four saved values to be redistributed two per shard.
    fname = os.path.join(self.get_temp_dir(), 'checkpoint')
    model = autotrackable.AutoTrackable()
    variables = [
        variables_lib.Variable([0]),
        variables_lib.Variable([1]),
        variables_lib.Variable([2]),
        variables_lib.Variable([3])
    ]
    model.s = sharded_variable.ShardedVariable(variables)
    cp = util.Checkpoint(model=model)
    cp.write(fname)

    model2 = autotrackable.AutoTrackable()
    cp2 = util.Checkpoint(model=model2)
    cp2.restore(fname)
    variables2 = [
        variables_lib.Variable([0, 0]),
        variables_lib.Variable([0, 0])
    ]
    model2.s = sharded_variable.ShardedVariable(variables2)
    self.assertAllEqual(self.evaluate(model2.s.variables[0]), [0, 1])
    self.assertAllEqual(self.evaluate(model2.s.variables[1]), [2, 3])

  def test_save_graph_def(self):
    # [FUNCTION DOC]
    # Builds a trackable root with a two-shard variable, a training function doing an embedding
    # lookup over the shard list and a serving function with an input signature that looks up in
    # the first shard. Traces and runs the training function, exports the model with the serving
    # function, loads it back in a legacy graph and session and runs it; expects the same values
    # both times, and that training still works afterwards.
    root = autotrackable.AutoTrackable()
    v1 = variables_lib.Variable([3.])
    v2 = variables_lib.Variable([2.])
    root.v = sharded_variable.ShardedVariable([v1, v2])
    root.train = def_function.function(
        lambda x: embedding_ops.embedding_lookup_v2(root.v.variables, x))
    # TODO(b/144057383): Remove the necessity of root.serve once saving context
    # is made to tf.function cache.
    root.serve = def_function.function(
        lambda x: embedding_ops.embedding_lookup_v2(root.v.variables[0], x),
        input_signature=[tensor_spec.TensorSpec([2], dtypes.int32, name='x')])

    # Trace and use root.train
    self.assertAllEqual([3., 2.], root.train([0, 1]).numpy())

    save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
    save.save(root, save_dir, root.serve)
    self.assertAllEqual([3., 2.],
                        _load_and_run(save_dir, {'x': [0, 1]})['output_0'])

    # Continue using root.train for training
    self.assertAllEqual([3., 2.], root.train([0, 1]).numpy())

  def test_validation_errors(self):
    # [FUNCTION DOC]
    # Checks constructor validation: nothing, a list with a non-variable, and an empty list raise
    # a type error; mixed dtypes and mismatched trailing shapes raise value errors; a shard that
    # already carries save slice info raises a value error.
    with self.assertRaisesRegex(TypeError, 'should be a non-empty list of'):
      sharded_variable.ShardedVariable(None)

    with self.assertRaisesRegex(TypeError, 'should be a non-empty list of'):
      sharded_variable.ShardedVariable(
          [variables_lib.Variable([0]), 'not-a-variable'])

    with self.assertRaisesRegex(TypeError, 'should be a non-empty list of'):
      sharded_variable.ShardedVariable([])

    with self.assertRaisesRegex(ValueError, 'must have the same dtype'):
      sharded_variable.ShardedVariable([
          variables_lib.Variable([0], dtype='int64'),
          variables_lib.Variable([1], dtype='int32')
      ])

    with self.assertRaisesRegex(ValueError, 'the same shapes except'):
      sharded_variable.ShardedVariable([
          variables_lib.Variable(array_ops.ones((5, 10))),
          variables_lib.Variable(array_ops.ones((5, 20)))
      ])

    with self.assertRaisesRegex(ValueError, '`SaveSliceInfo` should not'):
      v = variables_lib.Variable([0])
      v._set_save_slice_info(
          variables_lib.Variable.SaveSliceInfo(
              full_name='s', full_shape=[2], var_offset=[0], var_shape=[1]))
      sharded_variable.ShardedVariable([v])

  def test_as_function_input(self):
    # [FUNCTION DOC]
    # Passes two different two-shard variables to the same compiled function that assigns zeros;
    # expects both to be zeroed and the function to be traced only once (same spec reused).
    variables1 = [
        variables_lib.Variable([1]),
        variables_lib.Variable([1]),
    ]
    s = sharded_variable.ShardedVariable(variables1)
    variables2 = [
        variables_lib.Variable([2]),
        variables_lib.Variable([2]),
    ]
    s2 = sharded_variable.ShardedVariable(variables2)

    trace_count = [0]

    @def_function.function
    def func(sharded_var):
      # [FUNCTION DOC]
      # Increments the trace counter (counting traces, not calls) and assigns zeros to the given
      # sharded variable.
      trace_count[0] = trace_count[0] + 1
      sharded_var.assign([0, 0])

    func(s)
    self.assertAllEqual(ops.convert_to_tensor(s), [0, 0])
    self.assertEqual(trace_count[0], 1)
    func(s2)
    self.assertAllEqual(ops.convert_to_tensor(s2), [0, 0])
    self.assertEqual(trace_count[0], 1)

  def test_flatten(self):
    # [FUNCTION DOC]
    # Flattens a sharded variable with nest: expects the variable itself by default, and the
    # list of its shard variables when composites are expanded.
    variables = [
        variables_lib.Variable([0]),
        variables_lib.Variable([1]),
    ]
    s = sharded_variable.ShardedVariable(variables)

    got = nest.flatten(s)
    self.assertIs(s, got[0])

    got = nest.flatten(s, expand_composites=True)
    expected = nest.flatten(variables, expand_composites=True)
    self.assertEqual(got, expected)

  def test_tf_module(self):
    # [FUNCTION DOC]
    # Defines a module holding a sharded variable attribute. Expects the module's variables to be
    # the two shards (equal to its trainable variables) and its single tracked child to be the
    # sharded variable.

    class Model(module.Module):
      # [CLASS DOC]
      # Small module fixture owning one two-shard sharded variable attribute.

      def __init__(self):
        # [FUNCTION DOC]
        # Constructor: creates two one-element variables and stores them as a sharded variable
        # attribute.
        super().__init__()
        variables = [
            variables_lib.Variable([0]),
            variables_lib.Variable([1]),
        ]
        self.w = sharded_variable.ShardedVariable(variables)

    model = Model()

    self.assertLen(model.variables, 2)
    self.assertEqual(model.variables[0], [0])
    self.assertEqual(model.variables[1], [1])
    self.assertAllEqual(model.variables, model.trainable_variables)

    self.assertLen(model._trackable_children(), 1)
    self.assertIs(model._trackable_children().popitem()[1], model.w)

  def test_embedding_lookup(self):
    # [FUNCTION DOC]
    # Builds a sharded variable of three shards and compiled functions doing a dense lookup, a
    # sparse lookup and a safe sparse lookup with weights. Inspects each traced graph and expects
    # exactly one resource gather per shard (no accidental conversion to a single tensor), then
    # checks the numeric results of each lookup.
    v = [
        variables_lib.Variable([[1., 2.], [3., 4.]]),
        variables_lib.Variable([[5., 6.], [7., 8.]]),
        variables_lib.Variable([[9., 10.]])
    ]
    sv = sharded_variable.ShardedVariable(v)

    @def_function.function
    def lookup():
      # [FUNCTION DOC]
      # Dense embedding lookup of ids zero, three and four in the sharded variable.
      ids = constant_op.constant([0, 3, 4])
      return embedding_ops.embedding_lookup_v2(sv, ids)

    @def_function.function
    def sparse_lookup():
      # [FUNCTION DOC]
      # Sparse embedding lookup using a small sparse id tensor and no weights.
      sp_ids = sparse_tensor.SparseTensor(
          indices=[[0, 0], [0, 1], [1, 0], [2, 2]],
          values=[0, 3, 4, 1],
          dense_shape=[3, 3])
      return embedding_ops.embedding_lookup_sparse_v2(sv, sp_ids, None)

    @def_function.function
    def safe_sparse_lookup():
      # [FUNCTION DOC]
      # Safe sparse embedding lookup with sparse ids (including an invalid negative id) and sparse
      # weights.
      sp_ids = sparse_tensor.SparseTensor(
          indices=[[0, 0], [0, 1], [1, 0], [2, 2]],
          values=[0, -1, 4, 1],
          dense_shape=[3, 3])
      sp_weights = sparse_tensor.SparseTensor(
          indices=[[0, 0], [0, 1], [1, 0], [2, 2]],
          values=[1., 1., -1., 1.],
          dense_shape=[3, 3])
      return embedding_ops.safe_embedding_lookup_sparse_v2(
          sv, sp_ids, sp_weights)

    for func in [lookup, sparse_lookup, safe_sparse_lookup]:
      num_gather_ops = 0
      for op in func.get_concrete_function().graph.get_operations():
        if op.type == 'ResourceGather':
          num_gather_ops += 1
      self.assertEqual(
          num_gather_ops, len(v), 'Number of ResourceGather op '
          f'({num_gather_ops}) does not match expected ({len(v)}), possibly '
          'due to ShardedVariable accidentally being converted to tensor in '
          'embedding_lookup ops.')

    self.assertAllEqual(lookup(), [[1., 2.], [7., 8.], [9., 10.]])
    self.assertAllClose(sparse_lookup(), [[4., 5.], [9., 10.], [3., 4.]])
    self.assertAllClose(safe_sparse_lookup(), [[1., 2.], [0., 0.], [3., 4.]])

  def test_slicing(self):
    # [FUNCTION DOC]
    # Slices an eight-row variable split into shards of 3, 3 and 2 rows. Checks single indices,
    # positive-step slices, negative-step slices, empty slices, slicing other dimensions, integer
    # indexing (including negative and out-of-bounds, which must raise an index error), ellipsis,
    # new axis, a boolean mask, tensor indices (must raise a type error), and slicing inside a
    # compiled function. Each is compared with the expected rows.
    data = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14],
            [15, 16]]
    v = [
        variables_lib.Variable(data[:3]),
        variables_lib.Variable(data[3:6]),
        variables_lib.Variable(data[6:])
    ]
    sv = sharded_variable.ShardedVariable(v)
    empty = v[0][0:0]

    # Test cases: all individual indices
    for ix in range(len(data)):
      self.assertAllEqual(sv[ix].numpy(), data[ix])

    # Test cases: positive step
    self.assertAllEqual(sv[:], array_ops.concat(v, axis=0))
    self.assertAllEqual(sv[:2], [[1, 2], [3, 4]])
    self.assertAllEqual(sv[-8:2], [[1, 2], [3, 4]])
    self.assertAllEqual(sv[-10:2], [[1, 2], [3, 4]])
    self.assertAllEqual(sv[5:], [[11, 12], [13, 14], [15, 16]])
    self.assertAllEqual(sv[5:-1], [[11, 12], [13, 14]])
    self.assertAllEqual(sv[::3], [[1, 2], [7, 8], [13, 14]])
    self.assertAllEqual(sv[::5], [[1, 2], [11, 12]])
    self.assertAllEqual(sv[1::6], [[3, 4], [15, 16]])
    self.assertAllEqual(sv[1:5:6], [[3, 4]])
    self.assertAllEqual(sv[1::7], [[3, 4]])
    self.assertAllEqual(sv[2:7], [[5, 6], [7, 8], [9, 10], [11, 12], [13, 14]])
    self.assertAllEqual(sv[2:7:2], [[5, 6], [9, 10], [13, 14]])
    self.assertAllEqual(sv[2:7:3], [[5, 6], [11, 12]])

    # Test cases: negative step
    self.assertAllEqual(
        sv[::-1], array_ops.reverse(array_ops.concat(v, axis=0), axis=[0]))
    self.assertAllEqual(sv[2::-1], [[5, 6], [3, 4], [1, 2]])
    self.assertAllEqual(sv[2:-8:-1], [[5, 6], [3, 4]])
    self.assertAllEqual(sv[2:-10:-1], [[5, 6], [3, 4], [1, 2]])
    self.assertAllEqual(sv[4::-1], [[9, 10], [7, 8], [5, 6], [3, 4], [1, 2]])
    self.assertAllEqual(sv[-1:-3:-1], [[15, 16], [13, 14]])
    self.assertAllEqual(sv[::-5], [[15, 16], [5, 6]])
    self.assertAllEqual(sv[6::-6], [[13, 14], [1, 2]])
    self.assertAllEqual(sv[6:5:-6], [[13, 14]])
    self.assertAllEqual(sv[6::-7], [[13, 14]])
    self.assertAllEqual(sv[7:1:-1],
                        [[15, 16], [13, 14], [11, 12], [9, 10], [7, 8], [5, 6]])
    self.assertAllEqual(sv[7:1:-2], [[15, 16], [11, 12], [7, 8]])
    self.assertAllEqual(sv[7:1:-4], [[15, 16], [7, 8]])

    # Test cases: empty slice
    self.assertAllEqual(sv[0:0], empty)
    self.assertAllEqual(sv[5:3], empty)
    self.assertAllEqual(sv[3:5:-1], empty)
    self.assertAllEqual(sv[-1:0], empty)
    self.assertAllEqual(sv[2:-1:-1], empty)

    # Test cases: slicing other dimensions
    self.assertAllEqual(sv[:, 0], [1, 3, 5, 7, 9, 11, 13, 15])
    self.assertAllEqual(sv[:, 0:1], [[1], [3], [5], [7], [9], [11], [13], [15]])

    # Test cases: normal indexing
    self.assertAllEqual(sv[2], [5, 6])
    self.assertAllEqual(sv[6], [13, 14])
    self.assertAllEqual(sv[2, 1], 6)
    self.assertAllEqual(sv[-2], [13, 14])
    with self.assertRaisesRegex(IndexError, 'out of bounds'):
      _ = sv[100]
    with self.assertRaisesRegex(IndexError, 'out of bounds'):
      _ = sv[-100]

    # Test cases: Ellipsis
    self.assertAllEqual(sv[...], array_ops.concat(v, axis=0))
    self.assertAllEqual(sv[..., 0], [1, 3, 5, 7, 9, 11, 13, 15])
    self.assertAllEqual(sv[0:1, ...], [[1, 2]])

    # Test cases: newaxis
    self.assertAllEqual(
        sv[array_ops.newaxis, ...],
        array_ops.expand_dims_v2(array_ops.concat(v, axis=0), axis=0))

    # Test cases: boolean masks
    self.assertAllEqual(sv[ops.convert_to_tensor(sv) > 10],
                        [11, 12, 13, 14, 15, 16])

    # Test cases: tensor input
    with self.assertRaisesRegex(TypeError, 'not allowed'):
      _ = sv[constant_op.constant(1)::]
    with self.assertRaisesRegex(TypeError, 'not allowed'):
      _ = sv[:constant_op.constant(1):]
    with self.assertRaisesRegex(TypeError, 'not allowed'):
      _ = sv[constant_op.constant(1)]

    # Test cases: inside tf.function
    @def_function.function
    def func():
      # [FUNCTION DOC]
      # Returns the first column of the sharded variable via slicing, inside a compiled function.
      a = sv[:, 0]
      return a

    self.assertAllEqual(func(), [1, 3, 5, 7, 9, 11, 13, 15])

  def test_operator_overload(self):
    # [FUNCTION DOC]
    # Compares and adds two equal-valued two-shard variables; expects element-wise equality
    # results and a sum of doubled values.
    v1 = [
        variables_lib.Variable([1.]),
        variables_lib.Variable([2.]),
    ]
    sv1 = sharded_variable.ShardedVariable(v1)

    v2 = [
        variables_lib.Variable([1.]),
        variables_lib.Variable([2.]),
    ]
    sv2 = sharded_variable.ShardedVariable(v2)

    equal = sv1 == sv2
    self.assertAllEqual(equal, [True, True])
    self.assertAllEqual(sv1 + sv2, [2.0, 4.0])

  def test_shards_have_container_set(self):
    # [FUNCTION DOC]
    # Expects every shard to carry a weak reference attribute pointing back to the owning sharded
    # variable.
    v1 = [
        variables_lib.Variable([1.]),
        variables_lib.Variable([2.]),
    ]
    sv1 = sharded_variable.ShardedVariable(v1)
    for v in sv1.variables:
      self.assertTrue(hasattr(v, '_sharded_container'))
      self.assertIs(v._sharded_container(), sv1)

  def test_numpy(self):
    # [FUNCTION DOC]
    # Converts a two-shard variable to an array; expects an array instance with the combined
    # values.
    v1 = [
        variables_lib.Variable([1.]),
        variables_lib.Variable([2.]),
    ]
    sv1 = sharded_variable.ShardedVariable(v1)
    sv1_np = sv1.numpy()
    self.assertIsInstance(sv1_np, np.ndarray)
    self.assertAllEqual(sv1_np, np.array([1., 2.]))


class ShardedVariableSaveLoadTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests saving and loading models under parameter server strategies with differing numbers of
  # shards. Fixture: a cluster resolver built from the shared test cluster description (two
  # workers, three parameter servers), created per test and disconnected on teardown.

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture: runs base setup, builds a cluster definition for two workers and three
    # parameter servers, and stores a simple cluster resolver for it.
    super().setUp()
    cluster_def = get_cluster_def(test_cluster_params, num_workers=2, num_ps=3)
    self.cluster_resolver = cluster_resolver_lib.SimpleClusterResolver(
        server_lib.ClusterSpec(cluster_def))

  def tearDown(self):
    # [FUNCTION DOC]
    # Test fixture: runs base teardown and resets the eager context so the test disconnects from
    # the cluster.
    super().tearDown()
    # Reset context to disconnect from the cluster.
    context._reset_context()

  def _create_strategy(self, num_shards):
    # [FUNCTION DOC]
    # HELPER: builds a distribution strategy for the save/load tests. Input: num_shards, integer.
    # If greater than one, returns a parameter server strategy using the fixture's cluster
    # resolver and a fixed-shards partitioner with that many shards; otherwise returns the
    # default strategy.
    if num_shards > 1:
      strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
          self.cluster_resolver,
          variable_partitioner=sharded_variable.FixedShardsPartitioner(
              num_shards))
    else:
      strategy = distribute_lib._get_default_strategy()
    return strategy

  @combinations.generate(
      combinations.combine(
          shard_config=[[2, 2], [2, 3], [3, 2], [2, 1], [1, 1]],
      ))
  def testSaveAndLoadSingleVariable(self, shard_config):
    # [FUNCTION DOC]
    # Parameterized over pairs of shard counts (before save, after load). Input: shard_config, a
    # two-element list. Creates a six-element variable under a strategy with the first shard
    # count, saves it, loads it under a strategy with the second count, and expects six values
    # loaded and equal to the original (shards concatenated when more than one).
    """Test saving and loading ShardedVariable with different numbers of shards.

    Loading tf.Variables into multiple Shards is not yet supported

    Args:
      shard_config: The number of shards to use before and after loading. For
        example, [2, 1] means to create and save the variable with 2 shards and
        load it into 1 shard (i.e., a regular tf.Variable).
    """
    strategy = self._create_strategy(shard_config[0])

    with strategy.scope():
      var = variables_lib.Variable([1., 2., 3., 4., 5., 6.])

    # Save variable
    model_dir = self.get_temp_dir()
    save.save(var, model_dir)

    strategy2 = self._create_strategy(shard_config[1])
    with strategy2.scope():
      # Load variable
      loaded = load.load(model_dir)

    # Assert all values loaded, values are same
    if shard_config[1] > 1:
      loaded = array_ops.concat(loaded.variables, axis=0)
    self.assertLen(loaded.numpy(), 6)

    if shard_config[0] > 1:
      var = array_ops.concat(var.variables, axis=0)
    self.assertAllClose(var.numpy(), loaded.numpy())

  def testSaveAndLoadModuleUnderStrategy(self):
    # [FUNCTION DOC]
    # Saves a small dense module created under a two-shard strategy, loads it under a three-shard
    # strategy and expects calling it to raise a value error about running a loaded non-Keras
    # model; then loads without any strategy and expects the same output as the original.

    class Dense(module.Module):
      # [CLASS DOC]
      # Fixture module with a six by six kernel and a bias of six random values, whose call
      # method is a compiled function computing kernel times input plus bias.

      def __init__(self):
        # [FUNCTION DOC]
        # Constructor: creates the random kernel and bias variables.
        self.kernel = variables_lib.Variable(
            random_ops.random_uniform((6, 6)), name='kernel')
        self.bias = variables_lib.Variable(
            random_ops.random_uniform((6,)), name='bias')

      @def_function.function
      def __call__(self, x):
        # [FUNCTION DOC]
        # Computes the matrix product of kernel and x, adds the bias and returns the result.
        out = math_ops.matmul(self.kernel, x)
        out = out + self.bias
        return out

    x = constant_op.constant(
        math_ops.range(6, dtype=dtypes.float32), shape=[6, 1])

    strategy = self._create_strategy(2)
    with strategy.scope():
      layer = Dense()
      expect = layer(x)

    model_dir = self.get_temp_dir()
    save.save(layer, model_dir)

    strategy2 = self._create_strategy(3)
    with strategy2.scope():
      loaded_layer = load.load(model_dir)
      # Should fail with informative error
      with self.assertRaisesRegex(ValueError, 'run a loaded non-Keras'):
        got = loaded_layer(x)

    # Loading without a strategy should work, because the tf.function is traced
    # with a single variable as input
    loaded_layer = load.load(model_dir)
    got = loaded_layer(x)
    self.assertAllClose(got, expect)


if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  test.main()
