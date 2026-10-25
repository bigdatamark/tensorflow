# [MODULE DOC]
# Functional tests for the embedding lookup ops: dense embedding lookup (sharded and unsharded,
# mod and div partition strategies, partitioned variables, max-norm clipping, ragged ids and
# gradients), sparse embedding lookup with combiners (sum, mean, square-root-normalized), safe
# sparse lookup, fingerprint/hash style id handling and the scatter add/subtract state ops.
# It also defines module-level helpers that build sharded embedding parameters and compute the
# expected lookup results with numpy so the tests can compare against the real ops.
# Importing it has no side effects beyond imports; running as a script starts the test runner.
# This is a test module, not a helper module.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Functional tests for ops used with embeddings."""

import itertools
import math

from absl.testing import parameterized
import numpy as np

from tensorflow.python.compat import compat as forward_compat
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging
from tensorflow.python.util import compat


def _AsLong(array):
  # [FUNCTION DOC]
  # HELPER: converts numpy integer shape or index lists into plain Python integers for the tests.
  # Input: any iterable of integer-like values. Returns a new list with each element converted to
  # a built-in integer. No side effects.
  """Casts arrays elements to long type. Used to convert from numpy tf."""
  return [int(x) for x in array]


class ScatterAddSubTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the scatter add and scatter subtract state ops on variables. A shared helper
  # builds random parameters and updates, applies the op and compares with a numpy computation;
  # the test methods vary shapes, repeated indices and mismatched shapes.

  def _TestCase(self, shape, indices, scatter_op=state_ops.scatter_add):
    # [FUNCTION DOC]
    # HELPER: shared body of the scatter add/subtract tests.
    # KNOWN ISSUE: it calls the base-class setUp from inside the helper (every call), which is
    # unusual and not mentioned in the docstring; the docstring also says the indices update the
    # last dimension of the parameters, while they actually index the first dimension.
    # Inputs: shape (list of ints for the parameter array), indices (list of ints indexing the first
    # dimension, may repeat), scatter_op (scatter add by default, or scatter subtract). Creates a
    # random float variable and random update values, initializes the variable, runs the op in a
    # CPU-only session, then applies the same updates in numpy and asserts exact equality.
    # Returns nothing; fails the test on mismatch.
    """Run a random test case with the given shape and indices.

    Args:
      shape: Shape of the parameters array.
      indices: One-dimensional array of ints, the indices of the last dimension
               of the parameters to update.
      scatter_op: ScatterAdd or ScatterSub.
    """
    super(ScatterAddSubTest, self).setUp()
    with self.cached_session(use_gpu=False):
      # Create a random parameter array of given shape
      p_init = np.random.rand(*shape).astype("f")
      # Create the shape of the update array. All dimensions except the last
      # match the parameter array, the last dimension equals the # of indices.
      vals_shape = [len(indices)] + shape[1:]
      vals_init = np.random.rand(*vals_shape).astype("f")
      v_i = [float(x) for x in vals_init.ravel()]
      p = variables.Variable(p_init)
      vals = constant_op.constant(v_i, shape=vals_shape, name="vals")
      ind = constant_op.constant(indices, dtype=dtypes.int32)
      p2 = scatter_op(p, ind, vals, name="updated_p")
      # p = init
      self.evaluate(variables.global_variables_initializer())
      # p += vals
      result = self.evaluate(p2)
    # Compute the expected 'p' using numpy operations.
    for i, ind in enumerate(indices):
      if scatter_op == state_ops.scatter_add:
        p_init.reshape(shape[0], -1)[ind, :] += (vals_init.reshape(
            vals_shape[0], -1)[i, :])
      else:
        p_init.reshape(shape[0], -1)[ind, :] -= (vals_init.reshape(
            vals_shape[0], -1)[i, :])
    self.assertTrue(all((p_init == result).ravel()))

  @test_util.run_deprecated_v1
  def testNoRepetitions(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): scatter add with distinct indices on parameters of rank 2, 3 and 4.
    # Expected: the variable equals the numpy-computed sum of parameters and updates.
    self._TestCase([2, 2], [1])
    self._TestCase([4, 4, 4], [2, 0])
    self._TestCase([43, 20, 10, 10], [42, 5, 6, 1, 3, 5, 7, 9])

  @test_util.run_deprecated_v1
  def testWithRepetitions(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): scatter add where indices repeat (including all indices equal), so
    # updates for one row must accumulate. Expected: result equals the numpy accumulation.
    self._TestCase([2, 2], [1, 1])
    self._TestCase([5, 3, 9, 5], [2, 0, 4, 1, 3, 1, 4, 0, 4, 3])
    self._TestCase([32, 4, 4], [31] * 8)

  @test_util.run_deprecated_v1
  def testRandom(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): five rounds of scatter add with a random rank-4 parameter shape and
    # twice as many random indices as rows (so repeats occur). Expected: matches numpy each time.
    # Random shapes of rank 4, random indices
    for _ in range(5):
      shape = np.random.randint(1, 20, size=4)
      indices = np.random.randint(shape[0], size=2 * shape[0])
      self._TestCase(_AsLong(list(shape)), list(indices))

  @test_util.run_deprecated_v1
  def testSubRandom(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): same as the random add test but with scatter subtract. Five rounds of
    # random rank-4 shapes and random, possibly repeated indices. Expected: matches numpy.
    # Random shapes of rank 4, random indices
    for _ in range(5):
      shape = np.random.randint(1, 20, size=4)
      indices = np.random.randint(shape[0], size=2 * shape[0])
      self._TestCase(_AsLong(list(shape)), list(indices), state_ops.scatter_sub)

  @test_util.run_deprecated_v1
  def testWrongShape(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): scatter add on a 1024 by 64 by 64 variable, first with a values
    # placeholder whose leading size differs from the number of indices, then with values whose
    # trailing dimension differs from the variable. Expected: a value error in both cases.
    # Indices and values mismatch.
    var = variables.Variable(
        array_ops.zeros(shape=[1024, 64, 64], dtype=dtypes.float32))
    indices = array_ops.placeholder(dtypes.int32, shape=[32])
    values = array_ops.placeholder(dtypes.float32, shape=[33, 64, 64])
    with self.assertRaises(ValueError):
      state_ops.scatter_add(var, indices, values)

    # Var and values mismatch.
    values = array_ops.placeholder(dtypes.float32, shape=[32, 64, 63])
    with self.assertRaises(ValueError):
      state_ops.scatter_add(var, indices, values)


def _PName(param_id):
  # [FUNCTION DOC]
  # HELPER: naming convention for embedding shard parameters in these tests.
  # Input: integer shard id. Returns the string consisting of the letter p followed by the id.
  return "p" + str(param_id)


def _EmbeddingParams(num_shards,
                     vocab_size,
                     dtype=dtypes.float32,
                     shape=None,
                     use_shapeless_placeholder=False):
  # [FUNCTION DOC]
  # HELPER: builds sharded embedding parameters and matching random values for the lookup tests.
  # Inputs: num_shards (number of partitions), vocab_size (total rows), dtype (float32 default),
  # shape (per-row embedding shape, default a single dimension of ten), use_shapeless_placeholder
  # (if true, shards are placeholders of unknown shape instead of constants). Rows are divided as
  # evenly as possible with the excess going to the first shards. Random values in [1, 2) are
  # generated per shard (single or double precision by dtype). Returns a tuple: list of shard
  # tensors, mapping from tensor name to numpy values, and a feed mapping from tensor name to the
  # same values. Side effects: adds ops to the default graph and consumes numpy random numbers.
  p = []
  params = {}
  feed_dict = {}
  if not shape:
    shape = [10]
  for i in range(num_shards):
    shard_shape = [vocab_size // num_shards] + shape
    if i < vocab_size % num_shards:  # Excess goes evenly on the first shards
      shard_shape[0] += 1

    param_name = _PName(i)

    if use_shapeless_placeholder:
      param = array_ops.placeholder(dtype, shape=None, name=param_name)
    else:
      param = constant_op.constant(
          1.0, shape=shard_shape, dtype=dtype, name=param_name)
    p.append(param)
    np_type = "f" if dtype == dtypes.float32 else "d"
    val = (np.random.rand(*shard_shape).astype(np_type)) + 1
    params[param_name + ":0"] = val
    feed_dict[param.name] = val
  return p, params, feed_dict


def _EmbeddingParamsAsPartitionedVariable(num_shards,
                                          vocab_size,
                                          dtype=dtypes.float32,
                                          shape=None,
                                          use_resource=False):
  # [FUNCTION DOC]
  # HELPER: builds a partitioned variable initialized with the same values as the plain shards.
  # Inputs: num_shards, vocab_size, dtype (default float32), shape (per-row shape, default ten),
  # use_resource (whether to use resource variables). Calls the shard builder, then creates a
  # variable named p through the variable scope, partitioned into at most num_shards slices,
  # initialized with the concatenated shard values. Returns a tuple: shard tensors, the partitioned
  # variable, mapping of names to values and feed mapping. Side effects: creates variables in the
  # current variable scope (fails if one named p already exists in it).
  p, params, feed_dict = _EmbeddingParams(
      num_shards, vocab_size, dtype=dtype, shape=shape)
  shape = shape or [10]
  partitioned_variable = variable_scope.get_variable(
      "p",
      shape=[vocab_size] + shape,
      initializer=array_ops.concat([params[p_i.name] for p_i in p], 0),
      partitioner=partitioned_variables.min_max_variable_partitioner(
          max_partitions=num_shards, min_slice_size=1),
      use_resource=use_resource)
  return p, partitioned_variable, params, feed_dict


def _EmbeddingResult(params,
                     id_vals,
                     num_shards,
                     vocab_size,
                     partition_strategy="mod",
                     weight_vals=None):
  # [FUNCTION DOC]
  # HELPER: numpy reference that computes the expected result of an embedding lookup.
  # Inputs: params (mapping from shard tensor name to numpy values), id_vals (ids per batch entry:
  # scalars or sequences), num_shards, vocab_size, partition_strategy (mod or div; anything else
  # fails an assertion), weight_vals (optional per-id weights, default all ones). For each batch
  # entry, finds each id's row in the right shard, multiplies it by its weight and sums over the
  # entry. Returns three single-precision arrays: summed weighted embeddings, summed weights and
  # summed squared weights per entry. No side effects.
  if weight_vals is None:
    weight_vals = np.copy(id_vals)
    weight_vals.fill(1)
  values = []
  weights = []
  weights_squared = []
  for ids, wts in zip(id_vals, weight_vals):
    value_aggregation = None
    weight_aggregation = None
    squared_weight_aggregation = None
    if isinstance(ids, compat.integral_types):
      ids = [ids]
      wts = [wts]
    for i, weight_value in zip(ids, wts):
      if partition_strategy == "mod":
        val = np.copy(params[_PName(i % num_shards) + ":0"][
            i // num_shards, :]) * weight_value
      elif partition_strategy == "div":
        ids_per_partition, extras = divmod(vocab_size, num_shards)
        threshold = extras * (ids_per_partition + 1)
        if i < threshold:
          partition = i // (ids_per_partition + 1)
          offset = i % (ids_per_partition + 1)
        else:
          partition = extras + (i - threshold) // ids_per_partition
          offset = (i - threshold) % ids_per_partition
        val = np.copy(
            params[_PName(partition) + ":0"][offset, :]) * weight_value
      else:
        assert False
      if value_aggregation is None:
        assert weight_aggregation is None
        assert squared_weight_aggregation is None
        value_aggregation = val
        weight_aggregation = weight_value
        squared_weight_aggregation = weight_value * weight_value
      else:
        assert weight_aggregation is not None
        assert squared_weight_aggregation is not None
        value_aggregation += val
        weight_aggregation += weight_value
        squared_weight_aggregation += weight_value * weight_value
    values.append(value_aggregation)
    weights.append(weight_aggregation)
    weights_squared.append(squared_weight_aggregation)
  values = np.array(values).astype(np.float32)
  weights = np.array(weights).astype(np.float32)
  weights_squared = np.array(weights_squared).astype(np.float32)
  return values, weights, weights_squared


class EmbeddingLookupTest(test.TestCase):
  # [CLASS DOC]
  # Test case for dense embedding lookup: sharded parameters with mod and div partitioning, int32
  # and int64 ids, partitioned and resource variables, unknown parameter shapes, max-norm clipping,
  # higher-rank ids, gradient checks, graph construction and ragged ids. Uses the module helpers
  # to build parameters and numpy expected values.

  # This test looks up [0, 0] in a parameter matrix sharded 2 ways. Since
  # both the ids are in the first shard, one of the resulting lookup
  # vector is going to be empty. The subsequent DivOp fails because of that.
  # TODO(keveman): Disabling the test until the underlying problem is fixed.
  @test_util.run_deprecated_v1
  def testSimpleSharded(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the comment above says the test is disabled until a division problem is fixed,
    # but it is enabled (only the graph-mode decorator is present).
    # Scenario: two shards, vocabulary of four, lookup of ids zero and zero (both in the first
    # shard). Evaluates the lookup with a feed of shard values and compares with the numpy
    # reference. Expected: identical values and matching shape. Prints the ids shape.
    with self.cached_session():
      num_shards = 2
      vocab_size = 4
      p, params, feed_dict = _EmbeddingParams(num_shards, vocab_size)

      id_vals = np.array([0, 0])
      ids = constant_op.constant(list(id_vals), dtype=dtypes.int32)
      print("Construct ids", ids.get_shape())
      embedding = embedding_ops.embedding_lookup(p, ids)

      tf_result = embedding.eval(feed_dict=feed_dict)
    np_result, _, _ = _EmbeddingResult(params, id_vals, num_shards, vocab_size)
    self.assertAllEqual(np_result, tf_result)
    self.assertShapeEqual(np_result, embedding)

  @test_util.run_deprecated_v1
  def testMaxNorm(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): one embedding row of value two, lookup of id zero with max norm one.
    # Expected: the returned row is clipped to value one.
    with self.cached_session():
      embeddings = constant_op.constant([[2.0]])

      ids = constant_op.constant([0], dtype=dtypes.int32)
      embedding = embedding_ops.embedding_lookup(
          [embeddings], ids, max_norm=1.0)

      self.assertAllEqual(embedding, [[1.0]])

  @test_util.run_deprecated_v1
  def testMaxNormNontrivial(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): two-dimensional embedding rows looked up with max norm two. Expected:
    # each row is rescaled to norm two, i.e. equals two times the row divided by its own norm.
    with self.cached_session():
      embeddings = constant_op.constant([[2.0, 4.0], [3.0, 1.0]])

      ids = constant_op.constant([0, 1], dtype=dtypes.int32)
      embedding = embedding_ops.embedding_lookup(
          [embeddings], ids, max_norm=2.0)

      norms = math_ops.sqrt(
          math_ops.reduce_sum(embeddings * embeddings, axis=1))
      normalized = embeddings / array_ops_stack.stack([norms, norms], axis=1)
      self.assertAllClose(embedding, 2 * self.evaluate(normalized))

  @test_util.run_deprecated_v1
  def testSimpleShardedPartitionedVariable(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): two shards, vocabulary four, parameters stored in a partitioned
    # variable. Looks up ids zero and zero. Asserts that the variable components equal the original
    # shard values and that the lookup matches the numpy reference in value and shape.
    # The session variable is unused.
    with self.cached_session() as sess:
      num_shards = 2
      vocab_size = 4
      p, p_variable, params, feed_dict = _EmbeddingParamsAsPartitionedVariable(
          num_shards, vocab_size)

      id_vals = np.array([0, 0])
      ids = constant_op.constant(list(id_vals), dtype=dtypes.int32)
      print("Construct ids", ids.get_shape())
      embedding = embedding_ops.embedding_lookup(p_variable, ids)
      self.evaluate(variables.global_variables_initializer())
      params_values = [params[p_i.name] for p_i in p]
      # Test that the PartitionedVariable components equal the list in p
      p_var_val = self.evaluate(list(p_variable))
      # Actual test
      tf_result = embedding.eval(feed_dict=feed_dict)
    np_result, _, _ = _EmbeddingResult(params, id_vals, num_shards, vocab_size)
    self.assertAllEqual(params_values, p_var_val)
    self.assertAllEqual(np_result, tf_result)
    self.assertShapeEqual(np_result, embedding)

  @test_util.run_deprecated_v1
  def testSimpleShardedPartitionedResourceVariable(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): like the partitioned variable test but with resource variables.
    # Also prints the graph definition. Asserts the components equal the original values and the
    # lookup matches the numpy reference in value and shape.
    with self.cached_session() as sess:
      num_shards = 2
      vocab_size = 4
      p, p_variable, params, _ = _EmbeddingParamsAsPartitionedVariable(
          num_shards, vocab_size, use_resource=True)

      id_vals = np.array([0, 0])
      ids = constant_op.constant(list(id_vals), dtype=dtypes.int32)
      print("Construct ids", ids.get_shape())
      embedding = embedding_ops.embedding_lookup(p_variable, ids)
      self.evaluate(variables.global_variables_initializer())
      params_values = [params[p_i.name] for p_i in p]
      # Test that the PartitionedVariable components equal the list in p
      p_var_val = self.evaluate(list(p_variable))
      # Actual test
      print(ops.get_default_graph().as_graph_def())
      tf_result = self.evaluate(embedding)
    np_result, _, _ = _EmbeddingResult(params, id_vals, num_shards, vocab_size)
    self.assertAllEqual(params_values, p_var_val)
    self.assertAllEqual(np_result, tf_result)
    self.assertShapeEqual(np_result, embedding)

  @test_util.run_deprecated_v1
  def testShardedModPartitioningInt32Ids(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): five shards, vocabulary thirteen, thirty random int32 ids (so repeats),
    # default mod partitioning. Expected: lookup equals the numpy reference and has the same shape.
    with self.cached_session():
      num_shards = 5
      vocab_size = 13
      # Embedding dimensions is 10. The vocab_size x 10 embedding
      # parameters are spread in num_shards matrices, so the first
      # 3 shards are 3 x 10 and the last 2 shards are 2 x 10.
      p, params, feed_dict = _EmbeddingParams(num_shards, vocab_size)

      num_vals = 30
      # Fetch num_vals embeddings for random word ids. Since
      # num_vals > vocab_size, this ought to have repetitions, so
      # will test that aspect.
      id_vals = np.random.randint(vocab_size, size=num_vals)
      ids = constant_op.constant(list(id_vals), dtype=dtypes.int32)

      embedding = embedding_ops.embedding_lookup(p, ids)
      tf_result = embedding.eval(feed_dict=feed_dict)
    np_result, _, _ = _EmbeddingResult(params, id_vals, num_shards, vocab_size)
    self.assertAllEqual(np_result, tf_result)
    self.assertShapeEqual(np_result, embedding)

  @test_util.run_deprecated_v1
  def testShardedModPartitioningInt64Ids(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): same as the int32 mod partitioning test but ids are int64.
    # Expected: lookup equals the numpy reference and has the same shape.
    with self.cached_session():
      num_shards = 5
      vocab_size = 13
      # Embedding dimensions is 10. The vocab_size x 10 embedding
      # parameters are spread in num_shards matrices, so the first
      # 3 shards are 3 x 10 and the last 2 shards are 2 x 10.
      p, params, feed_dict = _EmbeddingParams(num_shards, vocab_size)

      num_vals = 30
      # Fetch num_vals embeddings for random word ids. Since
      # num_vals > vocab_size, this ought to have repetitions, so
      # will test that aspect.
      id_vals = np.random.randint(vocab_size, size=num_vals)
      ids = constant_op.constant(list(id_vals), dtype=dtypes.int64)

      embedding = embedding_ops.embedding_lookup(p, ids)
      tf_result = embedding.eval(feed_dict=feed_dict)
    np_result, _, _ = _EmbeddingResult(params, id_vals, num_shards, vocab_size)
    self.assertAllEqual(np_result, tf_result)
    self.assertShapeEqual(np_result, embedding)

  @test_util.run_deprecated_v1
  def testShardedDivPartitioningInt32Ids(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): five shards, vocabulary thirteen, thirty random int32 ids, div
    # partitioning. Expected: lookup equals the numpy reference using div strategy, same shape.
    with self.cached_session():
      num_shards = 5
      vocab_size = 13
      # Embedding dimensions is 10. The vocab_size x 10 embedding
      # parameters are spread in num_shards matrices, so the first
      # 3 shards are 3 x 10 and the last 2 shards are 2 x 10.
      p, params, feed_dict = _EmbeddingParams(num_shards, vocab_size)

      num_vals = 30
      # Fetch num_vals embeddings for random word ids. Since
      # num_vals > vocab_size, this ought to have repetitions, so
      # will test that aspect.
      id_vals = np.random.randint(vocab_size, size=num_vals)
      ids = constant_op.constant(list(id_vals), dtype=dtypes.int32)

      embedding = embedding_ops.embedding_lookup(
          p, ids, partition_strategy="div")
      tf_result = embedding.eval(feed_dict=feed_dict)
    np_result, _, _ = _EmbeddingResult(
        params, id_vals, num_shards, vocab_size, partition_strategy="div")
    self.assertAllEqual(np_result, tf_result)
    self.assertShapeEqual(np_result, embedding)

  @test_util.run_deprecated_v1
  def testShardedDivPartitioningInt32IdsPartitionedVariable(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): div partitioning with int32 ids where the parameters are a partitioned
    # variable (five slices, vocabulary thirteen). Initializes variables, looks up thirty random ids.
    # Expected: result equals the numpy reference and has the same shape.
    with self.cached_session():
      num_shards = 5
      vocab_size = 13
      # Embedding dimensions is 10. The vocab_size x 10 embedding
      # parameters are spread in num_shards matrices, so the first
      # 3 shards are 3 x 10 and the last 2 shards are 2 x 10.
      _, p_variable, params, feed_dict = _EmbeddingParamsAsPartitionedVariable(
          num_shards, vocab_size)

      num_vals = 30
      # Fetch num_vals embeddings for random word ids. Since
      # num_vals > vocab_size, this ought to have repetitions, so
      # will test that aspect.
      id_vals = np.random.randint(vocab_size, size=num_vals)
      ids = constant_op.constant(list(id_vals), dtype=dtypes.int32)
      self.evaluate(variables.global_variables_initializer())
      embedding = embedding_ops.embedding_lookup(
          p_variable, ids, partition_strategy="div")
      tf_result = embedding.eval(feed_dict=feed_dict)
    np_result, _, _ = _EmbeddingResult(
        params, id_vals, num_shards, vocab_size, partition_strategy="div")
    self.assertAllEqual(np_result, tf_result)
    self.assertShapeEqual(np_result, embedding)

  @test_util.run_deprecated_v1
  def testShardedDivPartitioningInt64Ids(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): div partitioning with int64 ids, five shards, vocabulary thirteen.
    # Expected: lookup equals the numpy reference using div strategy, and has the same shape.
    with self.cached_session():
      num_shards = 5
      vocab_size = 13
      # Embedding dimensions is 10. The vocab_size x 10 embedding
      # parameters are spread in num_shards matrices, so the first
      # 3 shards are 3 x 10 and the last 2 shards are 2 x 10.
      p, params, feed_dict = _EmbeddingParams(num_shards, vocab_size)

      num_vals = 30
      # Fetch num_vals embeddings for random word ids. Since
      # num_vals > vocab_size, this ought to have repetitions, so
      # will test that aspect.
      id_vals = np.random.randint(vocab_size, size=num_vals)
      ids = constant_op.constant(list(id_vals), dtype=dtypes.int64)

      embedding = embedding_ops.embedding_lookup(
          p, ids, partition_strategy="div")
      tf_result = embedding.eval(feed_dict=feed_dict)
    np_result, _, _ = _EmbeddingResult(
        params, id_vals, num_shards, vocab_size, partition_strategy="div")
    self.assertAllEqual(np_result, tf_result)
    self.assertShapeEqual(np_result, embedding)

  @test_util.run_deprecated_v1
  def testShardedDivPartitioningUnknownParamShape(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): div partitioning where the shard parameters are placeholders with no
    # static shape, fed at run time; int64 ids, five shards, vocabulary thirteen. Expected: lookup
    # equals the numpy reference (no static shape check is made).
    with self.cached_session():
      num_shards = 5
      vocab_size = 13
      # Embedding dimensions is 10. The vocab_size x 10 embedding
      # parameters are spread in num_shards matrices, so the first
      # 3 shards are 3 x 10 and the last 2 shards are 2 x 10.

      # We clear parameter shapes, to test when shape is not statically known.
      p, params, feed_dict = _EmbeddingParams(
          num_shards, vocab_size, use_shapeless_placeholder=True)

      num_vals = 30
      # Fetch num_vals embeddings for random word ids. Since
      # num_vals > vocab_size, this ought to have repetitions, so
      # will test that aspect.
      id_vals = np.random.randint(vocab_size, size=num_vals)
      ids = constant_op.constant(list(id_vals), dtype=dtypes.int64)

      embedding = embedding_ops.embedding_lookup(
          p, ids, partition_strategy="div")
      tf_result = embedding.eval(feed_dict=feed_dict)
    np_result, _, _ = _EmbeddingResult(
        params, id_vals, num_shards, vocab_size, partition_strategy="div")
    self.assertAllEqual(np_result, tf_result)

  @test_util.run_deprecated_v1
  def testGradientsEmbeddingLookup(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): numeric gradient check of the lookup with respect to the shard
    # parameters, for ids of shape ten and two by five, with one and three shards, vocabulary nine,
    # embedding width two. Expected: gradient error below 1e-4 in every case.
    vocab_size = 9
    num_ids = 10
    id_vals = list(np.random.randint(vocab_size, size=num_ids))
    tf_logging.vlog(1, id_vals)
    for ids_shape in [(10,), (2, 5)]:
      for num_shards in [1, 3]:
        with self.cached_session():
          ids = constant_op.constant(
              id_vals, shape=ids_shape, dtype=dtypes.int32)
          x, params, _ = _EmbeddingParams(num_shards, vocab_size, shape=[2])
          y = embedding_ops.embedding_lookup(x, ids)
          y_shape = ids_shape + tuple(params[_PName(0) + ":0"].shape[1:])
          x_name = [_PName(i) for i in range(num_shards)]
          x_init_value = [params[x_n + ":0"] for x_n in x_name]
          x_shape = [i.shape for i in x_init_value]
          err = gradient_checker.compute_gradient_error(
              x, x_shape, y, y_shape, x_init_value=x_init_value)
        self.assertLess(err, 1e-4)

  @test_util.run_deprecated_v1
  def testGradientsEmbeddingLookupWithComputedParams(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): gradient check where the looked-up parameters are the squares of the
    # shard tensors, which forces conversion of sparse gradient slices to dense tensors. One and
    # three shards, five ids. Expected: gradient error below 1e-3.
    vocab_size = 9
    num_ids = 5
    id_vals = list(np.random.randint(vocab_size, size=num_ids))
    tf_logging.vlog(1, id_vals)
    for num_shards in [1, 3]:
      with self.cached_session():
        ids = constant_op.constant(id_vals, dtype=dtypes.int32)
        x, params, _ = _EmbeddingParams(num_shards, vocab_size, shape=[2])
        # This will force a conversion from IndexedSlices to Tensor.
        x_squared = [math_ops.square(elem) for elem in x]
        y = embedding_ops.embedding_lookup(x_squared, ids)
        y_shape = [num_ids] + list(params[_PName(0) + ":0"].shape[1:])
        x_name = [_PName(i) for i in range(num_shards)]
        x_init_value = [params[x_n + ":0"] for x_n in x_name]
        x_shape = [i.shape for i in x_init_value]
        err = gradient_checker.compute_gradient_error(
            x, x_shape, y, y_shape, x_init_value=x_init_value)
      self.assertLess(err, 1e-3)

  def testConstructionNonSharded(self):
    # [FUNCTION DOC]
    # Scenario: in a fresh graph, builds a lookup on a single 100 by 100 variable passed as a one
    # element list. Expected: graph construction succeeds; nothing is run or asserted.
    with ops.Graph().as_default():
      p = variables.Variable(
          array_ops.zeros(shape=[100, 100], dtype=dtypes.float32))
      ids = constant_op.constant([0, 1, 1, 7], dtype=dtypes.int32)
      embedding_ops.embedding_lookup([p], ids)

  def testConstructionSharded(self):
    # [FUNCTION DOC]
    # Scenario: in a fresh graph, builds a lookup over a list of two 100 by 100 variables.
    # Expected: graph construction succeeds; nothing is run or asserted.
    with ops.Graph().as_default():
      p = []
      for _ in range(2):
        p += [
            variables.Variable(
                array_ops.zeros(shape=[100, 100], dtype=dtypes.float32))
        ]
        ids = constant_op.constant([0, 1, 1, 17], dtype=dtypes.int32)
      embedding_ops.embedding_lookup(p, ids)

  @test_util.run_deprecated_v1
  def testHigherRank(self):
    # [FUNCTION DOC]
    # Scenario (graph mode, fixed random seed): parameters of rank one and two, id tensors of shape
    # three by two and four by three. The unsharded lookup must equal a plain gather, and sharded
    # versions (one to three shards built by strided gathers) must equal the unsharded result.
    np.random.seed(8)
    with self.cached_session():
      for params_shape in (12,), (6, 3):
        params = np.random.randn(*params_shape)
        for ids_shape in (3, 2), (4, 3):
          ids = np.random.randint(
              params.shape[0], size=np.prod(ids_shape)).reshape(ids_shape)
          # Compare nonsharded to gather
          simple = embedding_ops.embedding_lookup(params, ids)
          self.assertAllEqual(simple, array_ops.gather(params, ids))
          # Run a few random sharded versions
          for procs in 1, 2, 3:
            stride = procs * math_ops.range(params.shape[0] // procs)
            split_params = [
                array_ops.gather(params, stride + p) for p in range(procs)
            ]
            sharded = embedding_ops.embedding_lookup(split_params, ids)
            self.assertAllEqual(simple, sharded)

  @test_util.run_deprecated_v1
  def testHigherRankMaxNorm(self):
    # [FUNCTION DOC]
    # Scenario (graph mode, fixed seed): parameters of rank one to three filled with two, ids of
    # rank zero to three, max norm one. The unsharded result must be close to a gather of the
    # normalized parameters, and sharded results (one to three shards) must equal it exactly.
    # A tolerance is used for the first comparison because square-root implementations differ.
    np.random.seed(8)
    with self.cached_session():
      for params_shape in (12,), (6, 3), (6, 2, 3):
        # Test embedding rank 0, 1, 2.
        # Note: the first dimension must be a common multiple of procs below.
        params = 2 * np.ones(params_shape)
        params_norm = params / np.sqrt(
            np.sum(
                params * params, tuple(range(params.ndim)[1:]), keepdims=True))
        for ids_shape in (), (3), (4, 3), (2, 3, 4):
          ids = np.random.randint(
              params.shape[0], size=np.prod(ids_shape,
                                            dtype=np.int64)).reshape(ids_shape)
          # Compare nonsharded to gather
          simple = embedding_ops.embedding_lookup(params, ids, max_norm=1.0)
          # assertAllClose is used here as different implementations of sqrt may
          # be used to compute each of the values being compared.  For example,
          # on AVX512 builds the embedding operation makes use of Eigen's fast
          # vectorized square root algorithm for doubles.  These different
          # implementations of sqrt are not guaranteed to produce exactly the
          # same results. Therefore, an exact comparison cannot be made.
          self.assertAllClose(simple, array_ops.gather(params_norm, ids))
          # Run a few different sharded versions.
          for procs in 1, 2, 3:
            stride = procs * math_ops.range(params.shape[0] // procs)
            split_params = [
                array_ops.gather(params, stride + p) for p in range(procs)
            ]
            sharded = embedding_ops.embedding_lookup(
                split_params, ids, max_norm=1.0)
            self.assertAllEqual(simple, sharded)

  @test_util.run_deprecated_v1
  def testTransform(self):
    # [FUNCTION DOC]
    # Scenario (graph mode, fixed seed): tests the private lookup-and-transform helper with max
    # norm two and a transform that replaces each embedding by its norm (reducing rank by one),
    # for ids of rank zero to three, sharded and unsharded parameters. Expected: the unsharded
    # result is close to a gather of the per-row norms of the clipped parameters, and the sharded
    # results are close to the unsharded one.
    # This tests all combinations of:
    #   - ids rank 0, 1, >1
    #   - params sharded/unsharded
    # It always applies max_norm.
    np.random.seed(8)
    l2_norm = 2.
    with self.cached_session():
      # Param values are in [l2_norm, l2_norm+1) so it will always clip.
      params = np.random.rand(6, 3) + l2_norm
      params_norm = l2_norm * params / np.sqrt(
          np.sum(params * params, axis=1, keepdims=True))
      # Compute the norm of each embedding. This will change the embedding
      # rank to 0.
      params_norm = np.linalg.norm(params_norm, axis=1)
      transform = lambda x: linalg_ops.norm(x, axis=1)
      for ids_shape in (), (3), (4, 3), (2, 3, 4):
        # Test ids rank 0, 1, 2, 3.
        ids = np.random.randint(
            params.shape[0], size=np.prod(ids_shape,
                                          dtype=np.int64)).reshape(ids_shape)
        # Compare nonsharded to gather.
        simple = embedding_ops._embedding_lookup_and_transform(
            params, ids, max_norm=l2_norm, transform_fn=transform)
        self.assertAllClose(simple, array_ops.gather(params_norm, ids))
        # Run a few different sharded versions.
        for procs in 1, 2, 3:
          stride = procs * math_ops.range(params.shape[0] // procs)
          split_params = [
              array_ops.gather(params, stride + p) for p in range(procs)
          ]
          sharded = embedding_ops._embedding_lookup_and_transform(
              split_params, ids, max_norm=l2_norm, transform_fn=transform)
          # assertAllClose is used here as different implementations of sqrt may
          # be used to compute each of the values being compared.  For example,
          # on AVX512 builds the embedding operation makes use of Eigen's fast
          # vectorized square root algorithm for doubles.  These different
          # implementations of sqrt are not guaranteed to produce exactly the
          # same results. Therefore, an exact comparison cannot be made.
          self.assertAllClose(simple, sharded)

  def testRaggedMaxNorm(self):
    # [FUNCTION DOC]
    # Scenario: lookup with ragged ids (rows of length two and one) into a single embedding row of
    # value two with max norm one. Expected: a ragged-shaped result with every entry clipped to one.
    embeddings = constant_op.constant([[2.0]])
    ids = ragged_factory_ops.constant([[0, 0], [0]], dtype=dtypes.int32)
    embedding = embedding_ops.embedding_lookup([embeddings], ids, max_norm=1.0)
    self.assertAllEqual(embedding, [[[1.0], [1.0]], [[1.0]]])


# TODO(philipphack): Consider moving this test to
# tensorflow/python/ops/embedding_ops_test.py
class EmbeddingLookupSparseTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case for sparse embedding lookup with combiners and weights. Helper methods
  # generate random sparse (or ragged) ids and weights and group flat values per batch entry; test
  # methods compare the op with the numpy reference across shards, combiners, dtypes and options.

  def _RandomIdsAndWeights(self, batch_size, vocab_size, ragged=False):
    # [FUNCTION DOC]
    # HELPER: generates random sparse ids and weights for the sparse lookup tests.
    # Inputs: batch_size, vocab_size, ragged (if true, returns ragged tensors instead of sparse).
    # Each batch entry gets between one and five random ids (uniform below vocab_size) with random
    # weights in [1, 2). Returns a tuple: ids container, weights container (sparse int32 ids and
    # float32 weights, both with dense shape batch size by six, or ragged equivalents), flat numpy
    # ids, flat numpy weights, and the count of values per batch entry. Consumes numpy randomness.
    max_val_per_entry = 6
    vals_per_batch_entry = np.random.randint(
        1, max_val_per_entry, size=batch_size)
    num_vals = np.sum(vals_per_batch_entry)

    ids = np.random.randint(vocab_size, size=num_vals)
    weights = 1 + np.random.rand(num_vals)

    indices = []
    for batch_entry, num_val in enumerate(vals_per_batch_entry):
      for val_index in range(num_val):
        indices.append([batch_entry, val_index])

    shape = [batch_size, max_val_per_entry]

    sp_ids = sparse_tensor.SparseTensor(
        constant_op.constant(indices, dtypes.int64),
        constant_op.constant(ids, dtypes.int32),
        constant_op.constant(shape, dtypes.int64))
    sp_weights = sparse_tensor.SparseTensor(
        constant_op.constant(indices, dtypes.int64),
        constant_op.constant(weights, dtypes.float32),
        constant_op.constant(shape, dtypes.int64))

    if ragged:
      sp_ids = ragged_tensor.RaggedTensor.from_sparse(sp_ids)
      sp_weights = ragged_tensor.RaggedTensor.from_sparse(sp_weights)

    return sp_ids, sp_weights, ids, weights, vals_per_batch_entry

  def _GroupByBatchEntry(self, vals, vals_per_batch_entry):
    # [FUNCTION DOC]
    # HELPER: splits a flat sequence of values into per-batch-entry lists.
    # Inputs: vals (flat sequence), vals_per_batch_entry (sequence of counts). Returns a list of
    # lists where the i-th list holds the next count-many values in order. No side effects.
    grouped_vals = []
    index = 0
    for num_val in vals_per_batch_entry:
      grouped_vals.append(list(vals[index:(index + num_val)]))
      index += num_val
    return grouped_vals

  @parameterized.parameters(
      itertools.product(
          [1, 5],
          ["sum", "mean", "sqrtn"],
          [dtypes.float16, dtypes.bfloat16, dtypes.float32, dtypes.float64],
          [True, False],
          [True, False],
          [True, False],
      )
  )
  @test_util.run_deprecated_v1
  def testEmbeddingLookupSparse(
      self,
      num_shards,
      combiner,
      dtype,
      ignore_weights,
      ragged,
      allow_fast_lookup,
  ):
    # [FUNCTION DOC]
    # Scenario (graph mode, parameterized over shard count, combiner, dtype, ignore-weights flag,
    # ragged flag and fast-lookup flag): random sparse or ragged ids and weights, vocabulary
    # thirteen, batch ten, embedding shape two by five. Runs the sparse lookup with the combiner,
    # with or without weights. Asserts the static result shape (unknown batch plus embedding
    # shape) and dtype, and that values are close to the numpy reference (weighted sum, divided
    # by weight sum for mean or by root of squared weight sum for square-root normalization),
    # with looser tolerances for half-precision types.
    vocab_size = 13
    batch_size = 10
    param_shape = [2, 5]
    expected_lookup_result_shape = [None] + param_shape

    sp_ids, sp_weights, ids, weights, vals_per_batch_entry = (
        self._RandomIdsAndWeights(batch_size, vocab_size, ragged)
    )

    grouped_ids = self._GroupByBatchEntry(ids, vals_per_batch_entry)
    grouped_weights = self._GroupByBatchEntry(weights, vals_per_batch_entry)
    grouped_ignored_weights = self._GroupByBatchEntry(
        np.ones(np.sum(vals_per_batch_entry)), vals_per_batch_entry
    )

    with self.cached_session():
      p, params, feed_dict = _EmbeddingParams(
          num_shards, vocab_size, shape=param_shape, dtype=dtype
      )
      embedding_sum = embedding_ops.embedding_lookup_sparse(
          p,
          sp_ids,
          None if ignore_weights else sp_weights,
          combiner=combiner,
          allow_fast_lookup=allow_fast_lookup,
      )

      self.assertEqual(
          embedding_sum.get_shape().as_list(), expected_lookup_result_shape
      )
      self.assertEqual(embedding_sum.dtype, dtype)

      tf_embedding_sum = embedding_sum.eval(feed_dict=feed_dict)

      np_embedding_sum, np_weight_sum, np_weight_sq_sum = _EmbeddingResult(
          params,
          grouped_ids,
          num_shards,
          vocab_size,
          weight_vals=grouped_ignored_weights
          if ignore_weights
          else grouped_weights,
      )
      if combiner == "mean":
        np_embedding_sum /= np.reshape(np_weight_sum, (batch_size, 1, 1))
      if combiner == "sqrtn":
        np_embedding_sum /= np.reshape(
            np.sqrt(np_weight_sq_sum), (batch_size, 1, 1)
        )

      rtol = 1e-6
      if dtype == dtypes.bfloat16:
        rtol = 1e-2
      elif dtype == dtypes.float16:
        rtol = 1e-3
      atol = rtol
      self.assertAllClose(np_embedding_sum, tf_embedding_sum, rtol, atol)

  @parameterized.parameters(
      itertools.product(["sum", "mean", "sqrtn"], [True, False], [True, False])
  )
  def testMissingInSparseIds(self, combiner, ragged, allow_fast_lookup):
    # [FUNCTION DOC]
    # Scenario (parameterized over combiner, ragged flag, fast-lookup flag; regression test for a
    # GitHub issue): a four-row batch where rows zero and two have no ids, parameters are a
    # four by five matrix of ones, weights are one. Expected: empty rows give zero vectors and
    # non-empty rows give vectors of ones for every combiner.
    # Github issue, 36359
    with self.test_session():
      x = array_ops.ones((4, 5))
      indices = [[1, 0], [3, 0]]
      sp_ids = sparse_tensor.SparseTensor(
          constant_op.constant(indices, dtypes.int64),
          constant_op.constant([0, 2], dtypes.int32),
          constant_op.constant([4, 1], dtypes.int64),
      )
      sp_weights = sparse_tensor.SparseTensor(
          constant_op.constant(indices, dtypes.int64),
          constant_op.constant([1, 1], dtypes.float32),
          constant_op.constant([4, 1], dtypes.int64),
      )

      if ragged:
        sp_ids = ragged_tensor.RaggedTensor.from_sparse(sp_ids)
        sp_weights = ragged_tensor.RaggedTensor.from_sparse(sp_weights)

      embedding_sum = embedding_ops.embedding_lookup_sparse(
          x,
          sp_ids,
          sp_weights,
          combiner=combiner,
          allow_fast_lookup=allow_fast_lookup,
      )

      tf_embedding_sum = ops.convert_to_tensor(embedding_sum)
      self.assertAllClose(tf_embedding_sum[0], np.zeros(5))
      self.assertAllClose(tf_embedding_sum[1], np.ones(5))
      self.assertAllClose(tf_embedding_sum[2], np.zeros(5))
      self.assertAllClose(tf_embedding_sum[3], np.ones(5))

  @parameterized.parameters(
      itertools.product(
          [1, 3],
          ["sum", "mean", "sqrtn"],
          [dtypes.float32, dtypes.float64],
          [True, False],
          [True, False],
          [True, False],
      )
  )
  @test_util.run_deprecated_v1
  def testGradientsEmbeddingLookupSparse(
      self,
      num_shards,
      combiner,
      dtype,
      ignore_weights,
      ragged,
      allow_fast_lookup,
  ):
    # [FUNCTION DOC]
    # Scenario (graph mode, parameterized over shard count, combiner, float32 or float64,
    # ignore-weights, ragged and fast-lookup flags): numeric gradient check of the sparse lookup
    # with respect to the shard parameters, vocabulary twelve, batch four, embedding shape two by
    # three. Expected: error below 1e-5 for double precision and below 2e-3 for single precision.
    vocab_size = 12
    batch_size = 4
    param_shape = [2, 3]

    sp_ids, sp_weights, _, _, _ = self._RandomIdsAndWeights(
        batch_size, vocab_size, ragged
    )
    with self.cached_session():
      x, params, _ = _EmbeddingParams(
          num_shards, vocab_size, shape=param_shape, dtype=dtype
      )

      y = embedding_ops.embedding_lookup_sparse(
          x,
          sp_ids,
          None if ignore_weights else sp_weights,
          combiner=combiner,
          allow_fast_lookup=allow_fast_lookup,
      )
      x_name = [_PName(i) for i in range(num_shards)]
      x_init_value = [params[x_n + ":0"] for x_n in x_name]
      x_shape = [i.shape for i in x_init_value]
      y_shape = [batch_size] + list(params[_PName(0) + ":0"].shape[1:])
      err = gradient_checker.compute_gradient_error(
          x, x_shape, y, y_shape, x_init_value=x_init_value
      )
    self.assertLess(err, 1e-5 if dtype == dtypes.float64 else 2e-3)

  @parameterized.parameters(itertools.product([True, False], [True, False]))
  @test_util.run_deprecated_v1
  def testIncompatibleShapes(self, ragged, allow_fast_lookup):
    # [FUNCTION DOC]
    # Scenario (graph mode, parameterized over ragged and fast-lookup flags): ids with dense shape
    # two by two and three entries, but weights with dense shape one by two and two entries, so the
    # shapes disagree; mean combiner. Expected: the sparse lookup raises a value error.
    with self.cached_session():
      x, _, _ = _EmbeddingParams(1, 10, dtype=dtypes.float32)
      indices = [[0, 0], [0, 1], [1, 0]]
      indices_weights = [[0, 0], [0, 1]]
      sp_ids = sparse_tensor.SparseTensor(
          constant_op.constant(indices, dtypes.int64),
          constant_op.constant([0, 1, 2], dtypes.int32),
          constant_op.constant([2, 2], dtypes.int64),
      )
      sp_weights = sparse_tensor.SparseTensor(
          constant_op.constant(indices_weights, dtypes.int64),
          constant_op.constant([12.0, 5.0], dtypes.float32),
          constant_op.constant([1, 2], dtypes.int64),
      )
      if ragged:
        sp_ids = ragged_tensor.RaggedTensor.from_sparse(sp_ids)
        sp_weights = ragged_tensor.RaggedTensor.from_sparse(sp_weights)

      with self.assertRaises(ValueError):
        embedding_ops.embedding_lookup_sparse(
            x,
            sp_ids,
            sp_weights,
            combiner="mean",
            allow_fast_lookup=allow_fast_lookup,
        )

  @test_util.run_deprecated_v1
  def test_incompatible_types(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): sparse ids combined with ragged weights (mixed container types).
    # Expected: the sparse lookup raises a type error.
    with self.cached_session():
      x = array_ops.ones((4, 5))
      indices = [[1, 0], [3, 0]]
      sp_ids = sparse_tensor.SparseTensor(
          constant_op.constant(indices, dtypes.int64),
          constant_op.constant([0, 2], dtypes.int32),
          constant_op.constant([4, 1], dtypes.int64),
      )
      sp_weights = sparse_tensor.SparseTensor(
          constant_op.constant(indices, dtypes.int64),
          constant_op.constant([1, 1], dtypes.float32),
          constant_op.constant([4, 1], dtypes.int64),
      )
      sp_weights = ragged_tensor.RaggedTensor.from_sparse(sp_weights)

      self.assertRaises(
          TypeError,
          embedding_ops.embedding_lookup_sparse,
          x,
          sp_ids,
          sp_weights,
      )

  def _SortByKey(self, keys, vals):
    # [FUNCTION DOC]
    # HELPER: puts two parallel tensors into a canonical order for gradient comparisons.
    # Inputs: keys (1-D tensor of ids) and vals (tensor whose first dimension matches keys).
    # Returns a pair of tensors: keys sorted ascending and vals reordered by the same permutation.
    perm = sort_ops.argsort(keys)
    return array_ops.gather(keys, perm), array_ops.gather(vals, perm)

  def _ExpectedSparseGradient(
      self, nnz, param_shape, np_type, sp_ids, sp_weights, combiner
  ):
    # [FUNCTION DOC]
    # HELPER: computes the expected sparse gradient of a sparse embedding lookup for the
    # resource-variable gradient test.
    # KNOWN ISSUE: the docstring is split by a blank line after its first sentence and is cut off
    # mid-sentence, so it reads as two fragments; the behavior it describes is correct.
    # Inputs: nnz (number of ids), param_shape (embedding shape as list), np_type (numpy type code
    # for values), sp_ids (sparse ids), sp_weights (sparse weights or nothing for unit weights),
    # combiner (sum, mean or sqrtn). Starts from ones per id, scales by the inverse of the per-row
    # weight sums (or their square root form for square-root normalization) for non-sum combiners,
    # multiplies by the weights if given, sorts by id and merges duplicate ids by summing.
    # Returns the unique sorted ids and the merged value rows as tensors.
    """Returns the expected indices and values corresponding to the (sparse)

    gradient of a sparse embedding lookup.
    """
    expected_values = np.ones([nnz] + param_shape, dtype=np_type)
    segment_ids = sp_ids.indices[:, 0]
    ignore_weights = sp_weights is None
    weights = (
        array_ops.ones(nnz, dtype=dtypes.float32)
        if ignore_weights
        else sp_weights.values
    )
    if combiner == "sqrtn":
      weights = weights**2
    segment_weights = math_ops.segment_sum(weights, segment_ids)
    if combiner != "sum":
      grad_scale = 1.0 / array_ops.gather(segment_weights, segment_ids)
      if combiner == "sqrtn":
        grad_scale = math_ops.sqrt(grad_scale)
      expected_values *= grad_scale[:, None]
    if not ignore_weights:
      expected_values *= sp_weights.values[:, None]
    expected_indices = sp_ids.values
    # Sort and deduplicate the indices in the expected sparse tensor.
    expected_indices, expected_values = self._SortByKey(
        expected_indices, expected_values
    )
    expected_indices, unique_mapping = array_ops.unique(expected_indices)
    expected_values = math_ops.segment_sum(expected_values, unique_mapping)
    return expected_indices, expected_values

  def testResourceVariableGradientEmbeddingLookupSparse(self):
    # [FUNCTION DOC]
    # Scenario: for each combiner, float32 or float64, and with or without weights, a resource
    # variable of shape 128 by 16 is looked up with random sparse ids (batch 32) inside a gradient
    # tape under a pinned forward-compatibility date. The gradient with respect to the variable
    # must be a sparse-slices object with the full dense shape, and after sorting by id its indices
    # and values must equal those from the expected-gradient helper.
    """Explicitly checks the gradient of a sparse embedding lookup with

    ResourceVariable input.
    """
    vocab_size = 128
    batch_size = 32
    param_shape = [16]
    sp_ids, sp_weights, _, _, _ = self._RandomIdsAndWeights(
        batch_size, vocab_size
    )

    for combiner, dtype, ignore_weights in itertools.product(
        ["sum", "mean", "sqrtn"],
        [dtypes.float32, dtypes.float64],
        [True, False],
    ):
      with self.test_session(), forward_compat.forward_compatibility_horizon(
          2023, 9, 26
      ):
        x_shape = [vocab_size] + param_shape
        np_type = "f" if dtype == dtypes.float32 else "d"
        x = np.random.uniform(size=x_shape).astype(np_type) + 1
        x = resource_variable_ops.ResourceVariable(x)
        self.evaluate(variables.global_variables_initializer())

        def forward(x_):
          # [FUNCTION DOC]
          # HELPER: forward pass used under the gradient tape. Input: the variable to look up.
          # Returns the sparse embedding lookup of the enclosing loop's ids, optional weights and
          # combiner. No other side effects.
          y_ = embedding_ops.embedding_lookup_sparse(
              x_,
              sp_ids,
              None if ignore_weights else sp_weights,
              combiner=combiner,
          )
          return y_

        with gradients.GradientTape() as g:
          y = forward(x)
        dx = g.gradient(y, x)
        self.assertAllEqual(dx.dense_shape, x_shape)

        actual_indices, actual_values = dx.indices, dx.values
        # The sort order of the output is not guaranteed, so we must sort it
        # into a consistent order before comparing.
        actual_indices, actual_values = self._SortByKey(
            actual_indices, actual_values
        )

        nnz = sp_ids.values.get_shape()[0]
        expected_indices, expected_values = self._ExpectedSparseGradient(
            nnz,
            param_shape,
            np_type,
            sp_ids,
            None if ignore_weights else sp_weights,
            combiner,
        )

        self.assertAllEqual(actual_indices, expected_indices)
        self.assertAllClose(actual_values, expected_values)


class SafeEmbeddingLookupSparseTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case for the safe sparse embedding lookup, which prunes invalid ids and
  # non-positive weights and returns a zero or default vector for rows left empty. Helper methods
  # create random partitioned weights and canned 2-D and 3-D sparse ids and weights covering
  # valid, invalid, missing, single and non-positive-weight cases.

  def _random_weights(self, vocab_size=4, embed_dim=4, num_shards=1):
    # [FUNCTION DOC]
    # HELPER: creates and evaluates random embedding weights for the safe lookup tests.
    # Inputs: vocab_size, embed_dim and num_shards (all positive, shards at most vocab_size).
    # Creates a partitioned variable named embedding_weights with truncated-normal initialization,
    # runs each slice's initializer, and returns a list of numpy arrays, one per shard. Raises an
    # assertion error for invalid sizes. Side effects: variable creation in the current scope, so
    # it can be called only once per scope.
    assert vocab_size > 0
    assert embed_dim > 0
    assert num_shards > 0
    assert num_shards <= vocab_size

    initializer = init_ops.truncated_normal_initializer(
        mean=0.0, stddev=1.0 / math.sqrt(vocab_size), dtype=dtypes.float32)
    embedding_weights = list(variable_scope.get_variable(
        name="embedding_weights",
        shape=[vocab_size, embed_dim],
        partitioner=partitioned_variables.fixed_size_partitioner(num_shards),
        initializer=initializer))
    for w in embedding_weights:
      self.evaluate(w.initializer)
    embedding_weights = [self.evaluate(w) for w in embedding_weights]
    return embedding_weights

  def _ids_and_weights_2d(self, ragged):
    # [FUNCTION DOC]
    # HELPER: canned two-dimensional ids and weights for the safe lookup tests.
    # Input: ragged (if true, returns ragged tensors rather than sparse ones). Builds a five by four
    # batch whose rows cover: several valid ids plus one invalid id, only invalid ids, no ids, a
    # single id, and ids with zero or negative weight. Returns the ids and weights containers.
    # Each row demonstrates a test case:
    #   Row 0: multiple valid ids, 1 invalid id, weighted mean
    #   Row 1: all ids are invalid (leaving no valid ids after pruning)
    #   Row 2: no ids to begin with
    #   Row 3: single id
    #   Row 4: all ids have <=0 weight
    indices = [[0, 0], [0, 1], [0, 2], [1, 0], [3, 0], [4, 0], [4, 1]]
    ids = [0, 1, -1, -1, 2, 0, 1]
    weights = [1.0, 2.0, 1.0, 1.0, 3.0, 0.0, -0.5]
    shape = [5, 4]

    sparse_ids = sparse_tensor.SparseTensor(
        constant_op.constant(indices, dtypes.int64),
        constant_op.constant(ids, dtypes.int64),
        constant_op.constant(shape, dtypes.int64))

    sparse_weights = sparse_tensor.SparseTensor(
        constant_op.constant(indices, dtypes.int64),
        constant_op.constant(weights, dtypes.float32),
        constant_op.constant(shape, dtypes.int64))

    if ragged:
      sparse_ids = ragged_tensor.RaggedTensor.from_sparse(sparse_ids)
      sparse_weights = ragged_tensor.RaggedTensor.from_sparse(sparse_weights)

    return sparse_ids, sparse_weights

  def _ids_and_weights_3d(self):
    # [FUNCTION DOC]
    # HELPER: canned three-dimensional (two by three by four) sparse ids and weights covering the
    # same cases as the two-dimensional helper (valid with an invalid id, all invalid, empty,
    # single, non-positive weights). Takes no arguments; returns sparse ids and sparse weights.
    # Each (2-D) index demonstrates a test case:
    #   Index 0, 0: multiple valid ids, 1 invalid id, weighted mean
    #   Index 0, 1: all ids are invalid (leaving no valid ids after pruning)
    #   Index 0, 2: no ids to begin with
    #   Index 1, 0: single id
    #   Index 1, 1: all ids have <=0 weight
    #   Index 1, 2: no ids to begin with
    indices = [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [1, 0, 0], [1, 1, 0],
               [1, 1, 1]]
    ids = [0, 1, -1, -1, 2, 0, 1]
    weights = [1.0, 2.0, 1.0, 1.0, 3.0, 0.0, -0.5]
    shape = [2, 3, 4]

    sparse_ids = sparse_tensor.SparseTensor(
        constant_op.constant(indices, dtypes.int64),
        constant_op.constant(ids, dtypes.int64),
        constant_op.constant(shape, dtypes.int64))

    sparse_weights = sparse_tensor.SparseTensor(
        constant_op.constant(indices, dtypes.int64),
        constant_op.constant(weights, dtypes.float32),
        constant_op.constant(shape, dtypes.int64))

    return sparse_ids, sparse_weights

  @parameterized.parameters(itertools.product([True, False], [True, False]))
  @test_util.run_deprecated_v1
  def test_safe_embedding_lookup_sparse_return_zero_vector(
      self, ragged, allow_fast_lookup
  ):
    # [FUNCTION DOC]
    # Scenario (graph mode, parameterized over ragged and fast-lookup flags): safe lookup with the
    # default id left unset on the canned five-row batch with weights. Expected: row zero is the
    # weighted mean of the first two embeddings, rows with no valid ids or only non-positive
    # weights are zero vectors, and the single-id row is that embedding.
    with self.cached_session():
      embedding_weights = self._random_weights()
      sparse_ids, sparse_weights = self._ids_and_weights_2d(ragged)

      embedding_lookup_result = embedding_ops.safe_embedding_lookup_sparse_v2(
          embedding_weights,
          sparse_ids,
          sparse_weights,
          allow_fast_lookup=allow_fast_lookup,
      )

      self.assertAllClose(
          embedding_lookup_result,
          [
              (1.0 * embedding_weights[0][0] + 2.0 * embedding_weights[0][1])
              / 3.0,
              [0] * 4,
              [0] * 4,
              embedding_weights[0][2],
              [0] * 4,
          ],
      )

  @parameterized.parameters(itertools.product([True, False], [True, False]))
  @test_util.run_deprecated_v1
  def test_safe_embedding_lookup_sparse_return_special_vector(
      self, ragged, allow_fast_lookup
  ):
    # [FUNCTION DOC]
    # Scenario (graph mode, parameterized over ragged and fast-lookup flags): same batch as the
    # zero-vector test but with default id three. Expected: rows that would be empty return the
    # embedding of id three instead of zeros; the other rows are unchanged.
    with self.cached_session():
      embedding_weights = self._random_weights()
      sparse_ids, sparse_weights = self._ids_and_weights_2d(ragged)

      embedding_lookup_result = embedding_ops.safe_embedding_lookup_sparse_v2(
          embedding_weights,
          sparse_ids,
          sparse_weights,
          default_id=3,
          allow_fast_lookup=allow_fast_lookup,
      )

      self.assertAllClose(
          embedding_lookup_result,
          [
              (1.0 * embedding_weights[0][0] + 2.0 * embedding_weights[0][1])
              / 3.0,
              embedding_weights[0][3],
              embedding_weights[0][3],
              embedding_weights[0][2],
              embedding_weights[0][3],
          ],
      )

  @parameterized.parameters(itertools.product([True, False], [True, False]))
  @test_util.run_deprecated_v1
  def test_safe_embedding_lookup_sparse_no_weights(
      self, ragged, allow_fast_lookup
  ):
    # [FUNCTION DOC]
    # Scenario (graph mode, parameterized over ragged and fast-lookup flags): safe lookup with no
    # weights, so all valid ids count equally. Expected: row zero and the last row are the plain
    # mean of embeddings zero and one, the single-id row is its embedding, empty rows are zeros.
    with self.cached_session():
      embedding_weights = self._random_weights()
      sparse_ids, _ = self._ids_and_weights_2d(ragged)

      embedding_lookup_result = embedding_ops.safe_embedding_lookup_sparse_v2(
          embedding_weights,
          sparse_ids,
          None,
          allow_fast_lookup=allow_fast_lookup,
      )

      self.assertAllClose(
          embedding_lookup_result,
          [
              (embedding_weights[0][0] + embedding_weights[0][1]) / 2.0,
              [0] * 4,
              [0] * 4,
              embedding_weights[0][2],
              (embedding_weights[0][0] + embedding_weights[0][1]) / 2.0,
          ],
      )

  @parameterized.parameters(itertools.product([True, False], [True, False]))
  @test_util.run_deprecated_v1
  def test_safe_embedding_lookup_sparse_partitioned(
      self, ragged, allow_fast_lookup
  ):
    # [FUNCTION DOC]
    # Scenario (graph mode, parameterized over ragged and fast-lookup flags): weights split across
    # three shards, no id weights. The shards are concatenated for the numpy reference. Expected:
    # same results as the unpartitioned no-weights test (means, single embedding, zero rows).
    with self.cached_session():
      embedding_weights = self._random_weights(num_shards=3)
      sparse_ids, _ = self._ids_and_weights_2d(ragged)

      embedding_lookup_result = embedding_ops.safe_embedding_lookup_sparse_v2(
          embedding_weights,
          sparse_ids,
          None,
          allow_fast_lookup=allow_fast_lookup,
      )

      embedding_weights_list = list(itertools.chain(*embedding_weights))
      self.assertAllClose(
          embedding_lookup_result,
          [
              (embedding_weights_list[0] + embedding_weights_list[1]) / 2.0,
              [0] * 4,
              [0] * 4,
              embedding_weights_list[2],
              (embedding_weights_list[0] + embedding_weights_list[1]) / 2.0,
          ],
      )

  @parameterized.parameters(itertools.product([True, False], [True, False]))
  @test_util.run_deprecated_v1
  def test_safe_embedding_lookup_sparse_partitioned_inconsistent_weights(
      self, ragged, allow_fast_lookup
  ):
    # [FUNCTION DOC]
    # Scenario (graph mode, parameterized over ragged and fast-lookup flags): three shards where
    # one shard is converted to double precision while the others stay single precision.
    # Expected: the first-generation safe lookup raises a type error for the mixed numpy arrays,
    # and after converting all shards to double constants it raises a value error when weights
    # are given. The fast-lookup flag is only passed to the second call, and the ragged flag
    # does not affect the sparse ids and weights it uses (it is only used to build them).
    with self.cached_session():
      embedding_weights = self._random_weights(num_shards=3)
      sparse_ids, sparse_weights = self._ids_and_weights_2d(ragged)

      embedding_weights[1] = embedding_weights[1].astype(np.float64)
      self.assertRaises(TypeError, embedding_ops.safe_embedding_lookup_sparse,
                        embedding_weights, sparse_ids)
      embedding_weights_constant = [
          constant_op.constant(w, dtype=dtypes.float64)
          for w in embedding_weights
      ]
      self.assertRaises(
          ValueError,
          embedding_ops.safe_embedding_lookup_sparse,
          embedding_weights_constant,
          sparse_ids,
          sparse_weights,
          allow_fast_lookup=allow_fast_lookup,
      )

  @test_util.run_deprecated_v1
  def test_safe_embedding_lookup_sparse_3d_return_zero_vector(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): safe lookup on the canned 3-D sparse ids with weights and no default
    # id. Expected: output of shape two by three by embedding width; positions with valid ids hold
    # the weighted mean or single embedding, others hold zero vectors.
    with self.cached_session():
      embedding_weights = self._random_weights()
      sparse_ids, sparse_weights = self._ids_and_weights_3d()

      embedding_lookup_result = (
          embedding_ops.safe_embedding_lookup_sparse_v2(embedding_weights,
                                                        sparse_ids,
                                                        sparse_weights))

      self.assertAllClose(embedding_lookup_result, [[
          (1.0 * embedding_weights[0][0] + 2.0 * embedding_weights[0][1]) / 3.0,
          [0] * 4, [0] * 4
      ], [embedding_weights[0][2], [0] * 4, [0] * 4]])

  @test_util.run_deprecated_v1
  def test_safe_embedding_lookup_sparse_3d_return_special_vector(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): safe lookup on the canned 3-D ids with weights and default id three.
    # Expected: positions without usable ids return the embedding of id three; the rest as usual.
    with self.cached_session():
      embedding_weights = self._random_weights()
      sparse_ids, sparse_weights = self._ids_and_weights_3d()

      embedding_lookup_result = (
          embedding_ops.safe_embedding_lookup_sparse_v2(
              embedding_weights, sparse_ids, sparse_weights, default_id=3))

      self.assertAllClose(
          embedding_lookup_result,
          [[(1.0 * embedding_weights[0][0] + 2.0 * embedding_weights[0][1]) /
            3.0, embedding_weights[0][3], embedding_weights[0][3]], [
                embedding_weights[0][2], embedding_weights[0][3],
                embedding_weights[0][3]
            ]])

  @test_util.run_deprecated_v1
  def test_safe_embedding_lookup_sparse_3d_no_weights(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): safe lookup on the canned 3-D ids with no weights. Expected: plain
    # means over valid ids per position, single embeddings for one-id positions, zeros otherwise.
    with self.cached_session():
      embedding_weights = self._random_weights()
      sparse_ids, _ = self._ids_and_weights_3d()

      embedding_lookup_result = (
          embedding_ops.safe_embedding_lookup_sparse_v2(embedding_weights,
                                                        sparse_ids, None))

      self.assertAllClose(embedding_lookup_result, [[(
          embedding_weights[0][0] + embedding_weights[0][1]) / 2.0, [0] * 4, [
              0
          ] * 4], [
              embedding_weights[0][2],
              (embedding_weights[0][0] + embedding_weights[0][1]) / 2.0, [0] * 4
          ]])

  @test_util.run_deprecated_v1
  def test_safe_embedding_lookup_sparse_3d_partitioned(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): 3-D ids without weights over weights split into three shards, with the
    # shards concatenated for the numpy reference. Expected: same as the unpartitioned 3-D
    # no-weights result.
    with self.cached_session():
      embedding_weights = self._random_weights(num_shards=3)
      sparse_ids, _ = self._ids_and_weights_3d()

      embedding_lookup_result = (
          embedding_ops.safe_embedding_lookup_sparse_v2(embedding_weights,
                                                        sparse_ids, None))

      embedding_weights = list(itertools.chain(*embedding_weights))
      self.assertAllClose(embedding_lookup_result, [[
          (embedding_weights[0] + embedding_weights[1]) / 2.0, [0] * 4, [0] * 4
      ], [
          embedding_weights[2],
          (embedding_weights[0] + embedding_weights[1]) / 2.0, [0] * 4
      ]])

  @test_util.run_deprecated_v1
  def test_safe_embedding_lookup_sparse_3d_partitioned_inconsistent_weights(
      self):
    # [FUNCTION DOC]
    # Scenario (graph mode): 3-D ids over three shards with one shard converted to double
    # precision. Expected: the first-generation safe lookup raises a type error for the mixed
    # arrays; after converting all shards to double constants and passing weights it raises a
    # value error.
    with self.cached_session():
      embedding_weights = self._random_weights(num_shards=3)
      sparse_ids, sparse_weights = self._ids_and_weights_3d()

      embedding_weights[1] = embedding_weights[1].astype(np.float64)
      self.assertRaises(TypeError, embedding_ops.safe_embedding_lookup_sparse,
                        embedding_weights, sparse_ids)
      embedding_weights = [
          constant_op.constant(w, dtype=dtypes.float64)
          for w in embedding_weights
      ]
      self.assertRaises(ValueError, embedding_ops.safe_embedding_lookup_sparse,
                        embedding_weights, sparse_ids, sparse_weights)


class DynamicStitchOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the dynamic stitch op, which merges several value tensors into one output by
  # index lists (later inputs win for repeated indices). Covers integer inputs on CPU and default
  # devices, a gradient-argument shaped case and merge order.

  @test_util.run_deprecated_v1
  def testCint32Cpu(self):
    # [FUNCTION DOC]
    # Scenario (graph mode, CPU-only session): stitch two index lists with a repeated index two.
    # Expected: output of four elements where the second input overrides the first at index two.
    with self.session(use_gpu=False):
      indices = [
          ops.convert_to_tensor([0, 1, 2]),
          ops.convert_to_tensor([2, 3])
      ]
      values = [
          ops.convert_to_tensor([12, 23, 34]),
          ops.convert_to_tensor([1, 2])
      ]
      self.assertAllEqual(
          data_flow_ops.dynamic_stitch(indices, values), [12, 23, 1, 2])

  @test_util.run_deprecated_v1
  def testCint32Gpu(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: despite the name, nothing forces a GPU; it uses the default session placement and
    # runs the same scenario as the CPU variant.
    # Scenario (graph mode): stitch two index lists with a repeated index two. Expected: output
    # of four elements where the second input overrides the first at index two.
    with self.session():
      indices = [
          ops.convert_to_tensor([0, 1, 2]),
          ops.convert_to_tensor([2, 3])
      ]
      values = [
          ops.convert_to_tensor([12, 23, 34]),
          ops.convert_to_tensor([1, 2])
      ]
      self.assertAllEqual(
          data_flow_ops.dynamic_stitch(indices, values), [12, 23, 1, 2])

  @test_util.run_deprecated_v1
  def testInt32Cpu(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: identical in content to the Cint32 CPU test (the same inputs and expectation).
    # Scenario (graph mode, CPU-only session): stitch two index lists with a repeated index two.
    # Expected: output of four elements where the second input overrides the first at index two.
    with self.session(use_gpu=False):
      indices = [
          ops.convert_to_tensor([0, 1, 2]),
          ops.convert_to_tensor([2, 3])
      ]
      values = [
          ops.convert_to_tensor([12, 23, 34]),
          ops.convert_to_tensor([1, 2])
      ]
      self.assertAllEqual(
          data_flow_ops.dynamic_stitch(indices, values), [12, 23, 1, 2])

  @test_util.run_deprecated_v1
  def testInt32Gpu(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: despite the name, nothing forces a GPU; it uses the default session placement and
    # duplicates the other int32 scenarios.
    # Scenario (graph mode): stitch two index lists with a repeated index two. Expected: output
    # of four elements where the second input overrides the first at index two.
    with self.session():
      indices = [
          ops.convert_to_tensor([0, 1, 2]),
          ops.convert_to_tensor([2, 3])
      ]
      values = [
          ops.convert_to_tensor([12, 23, 34]),
          ops.convert_to_tensor([1, 2])
      ]
      self.assertAllEqual(
          data_flow_ops.dynamic_stitch(indices, values), [12, 23, 1, 2])

  @test_util.run_deprecated_v1
  def testSumGradArgs(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name suggests a gradient-argument test, but no gradient is computed; it only
    # checks the forward stitch result.
    # Scenario (graph mode, CPU-only session): index lists of four and two entries overlapping at
    # indices two and three. Expected: output [2, 3, 1, 1], later inputs overriding earlier ones.
    with self.session(use_gpu=False):
      indices = [
          ops.convert_to_tensor([0, 1, 2, 3]),
          ops.convert_to_tensor([2, 3])
      ]
      values = [
          ops.convert_to_tensor([2, 3, 5, 7]),
          ops.convert_to_tensor([1, 1])
      ]
      self.assertAllEqual(
          data_flow_ops.dynamic_stitch(indices, values), [2, 3, 1, 1])

  # We expect that the values are merged in order.
  @test_util.run_deprecated_v1
  def testStitchOrder(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): ten identical index tensors covering zero to ninety-nine, each paired
    # with different random values. Expected: the stitched output equals the last values tensor,
    # showing that inputs are merged in order and the last wins.
    with self.cached_session():
      indices = []
      np_values = []
      values = []
      for _ in range(10):
        indices.extend([ops.convert_to_tensor(np.arange(100).astype(np.int32))])
        np_values.extend([np.random.uniform(size=100)])
        values.extend([ops.convert_to_tensor(np_values[-1])])
      stitched = data_flow_ops.dynamic_stitch(indices, values)
    self.assertAllEqual(np_values[-1], stitched)


class ParallelDynamicStitchOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the parallel dynamic stitch op, which merges value tensors by index lists that
  # are expected not to overlap. Checks integer inputs on CPU-only sessions.

  @test_util.run_deprecated_v1
  def testCint32Cpu(self):
    # [FUNCTION DOC]
    # Scenario (graph mode, CPU-only session): disjoint index lists of four and three entries.
    # Expected: a seven-element output with every value placed at its index.
    with self.session(use_gpu=False):
      indices = [
          ops.convert_to_tensor([0, 1, 4, 6]),
          ops.convert_to_tensor([2, 3, 5])
      ]
      values = [
          ops.convert_to_tensor([12, 23, 34, 45]),
          ops.convert_to_tensor([1, 2, 3])
      ]
      self.assertAllEqual(
          data_flow_ops.parallel_dynamic_stitch(indices, values),
          [12, 23, 1, 2, 34, 3, 45])

  @test_util.run_deprecated_v1
  def testInt32Cpu(self):
    # [FUNCTION DOC]
    # Scenario (graph mode, CPU-only session): disjoint index lists of five and three entries.
    # Expected: an eight-element output with every value placed at its index.
    with self.session(use_gpu=False):
      indices = [
          ops.convert_to_tensor([0, 1, 5, 6, 7]),
          ops.convert_to_tensor([2, 4, 3])
      ]
      values = [
          ops.convert_to_tensor([12, 23, 34, 45, 56]),
          ops.convert_to_tensor([1, 3, 2])
      ]
      self.assertAllEqual(
          data_flow_ops.parallel_dynamic_stitch(indices, values),
          [12, 23, 1, 2, 3, 34, 45, 56])

  @test_util.run_deprecated_v1
  def testSimple(self):
    # [FUNCTION DOC]
    # Scenario (graph mode, CPU-only session): two index lists of two entries covering zero to
    # three with values two, three, one, one. Expected: output [2, 3, 1, 1].
    with self.session(use_gpu=False):
      indices = [ops.convert_to_tensor([0, 1]), ops.convert_to_tensor([2, 3])]
      values = [ops.convert_to_tensor([2, 3]), ops.convert_to_tensor([1, 1])]
      self.assertAllEqual(
          data_flow_ops.parallel_dynamic_stitch(indices, values), [2, 3, 1, 1])


if __name__ == "__main__":
  test.main()
