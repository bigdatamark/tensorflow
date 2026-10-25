# [MODULE DOC]
# Unit tests for the ragged map function that applies a callback to each row of a ragged
# tensor. Covers many output shape/ragged-rank combinations, structured (mapping) inputs and
# outputs, zipping and batch gathering, ragged-rank mismatch errors, sparse-derived input and
# nested maps with an explicit output signature. Runs the test main when executed as a script.
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
"""Tests for ragged_map_ops.map_fn."""
from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import map_fn as map_fn_lib
from tensorflow.python.ops import math_ops as mo
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops.ragged import ragged_map_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import googletest


def _stack_mean_and_sum(x):
  # [FUNCTION DOC]
  # HELPER: callback used by the parameterized map test. Input: x, a numeric tensor (one row).
  # Output: a two-element tensor holding the mean then the sum of x.
  return array_ops_stack.stack([mo.reduce_mean(x), mo.reduce_sum(x)])


@test_util.run_all_in_graph_and_eager_modes
class RaggedMapOpTest(test_util.TensorFlowTestCase,
                      parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the ragged map function, run in graph and eager modes. Mixes one large
  # parameterized test (callback, input, expected output and dtype per case) with scenario tests.

  @parameterized.parameters([
      # The following test sets map over a RaggedTensor and apply a
      # transformation that returns with shape:
      # [d1, (d2)] -> [d1]
      dict(
          fn=mo.reduce_mean,
          elems=[[1, 2, 3], [4, 5], [6, 7]],
          elems_dtype=dtypes.int32,
          expected_output=[2, 4, 6],
          result_dtype=dtypes.int32,
      ),
      dict(
          fn=string_ops.reduce_join,
          elems=[['foo', 'bar', 'baz'], ['a'], ['b', 'c']],
          expected_output=[b'foobarbaz', b'a', b'bc'],
          elems_dtype=dtypes.string,
          result_dtype=dtypes.string,
      ),
      # [d1, (d2)] -> [d1, 2]
      dict(
          fn=_stack_mean_and_sum,
          elems=[[1, 2, 3], [4, 5], [6, 7]],
          expected_output=[[2, 6], [4.5, 9], [6.5, 13]],
          elems_dtype=dtypes.float32,
          result_dtype=dtypes.float32,
          expected_ragged_rank=0,
      ),
      # [d1, (d2)] -> [d1, (d2)]
      dict(
          fn=lambda x: x + np.int64(1),
          elems=[[1, 2, 3], [4, 5], [6, 7]],
          expected_output=[[2, 3, 4], [5, 6], [7, 8]],
          elems_dtype=dtypes.int64,
          result_dtype=ragged_tensor.RaggedTensorType(
              dtype=dtypes.int64, ragged_rank=1),
      ),
      # [d1, (d2), d3] -> [d1, (d2), d3]
      dict(
          fn=lambda x: x + np.int64(1),
          elems=[[[1, 2], [3, 4]], [], [[5, 6], [7, 8], [9, 0]]],
          elems_ragged_rank=1,
          expected_ragged_rank=1,
          result_dtype=ragged_tensor.RaggedTensorType(
              dtype=dtypes.int64, ragged_rank=1),
          expected_output=[[[2, 3], [4, 5]], [], [[6, 7], [8, 9], [10, 1]]],
      ),
      # [d1, (d2)] -> [d1, (d2), (d3)]
      dict(
          fn=lambda x: ragged_tensor.RaggedTensor.from_row_starts(x, [0]),
          elems=[[1, 2, 3], [4, 5], [6, 7]],
          expected_output=[[[1, 2, 3]], [[4, 5]], [[6, 7]]],
          result_dtype=ragged_tensor.RaggedTensorType(
              dtype=dtypes.int64, ragged_rank=2),
      ),
      # [d1, (d2), (d3)] -> [d1, (d2), (d3)]
      dict(
          fn=lambda x: ragged_functional_ops.map_flat_values(mo.add, x, 1),
          elems=[[[1, 2, 3]], [[4, 5], [6, 7]]],
          expected_output=[[[2, 3, 4]], [[5, 6], [7, 8]]],
          result_dtype=ragged_tensor.RaggedTensorType(
              dtype=dtypes.int64, ragged_rank=2),
      ),
      # [d1, (d2), (d3)] -> [d1, (d2)]
      dict(
          fn=lambda x: ragged_math_ops.reduce_sum(x, axis=1),
          elems=[[[1, 2, 3]], [[4, 5], [6, 7]]],
          expected_output=[[6], [9, 13]],
          result_dtype=ragged_tensor.RaggedTensorType(
              dtype=dtypes.int64, ragged_rank=1),
      ),
      # [d1, (d2), (d3)] -> [d1, (d3)]
      dict(
          fn=lambda x: ragged_math_ops.reduce_sum(x, axis=0),
          elems=[[[1, 2, 3]], [[4, 5], [6, 7]]],
          expected_output=[[1, 2, 3], [10, 12]],
          result_dtype=ragged_tensor.RaggedTensorType(
              dtype=dtypes.int64, ragged_rank=1),
      ),
      # [d1, (d2), (d3)] -> [d1]
      dict(
          fn=ragged_math_ops.reduce_sum,
          elems=[[[1, 2, 3]], [[4, 5], [6, 7]]],
          expected_output=[6, 22],
          result_dtype=dtypes.int64,
      ),
      # [d1] -> [d1, (d2)]
      dict(
          fn=mo.range,
          elems=[4, 0, 2],
          expected_output=[[0, 1, 2, 3], [], [0, 1]],
          result_dtype=ragged_tensor.RaggedTensorType(
              dtype=dtypes.int64, ragged_rank=1),
      ),
      # [d1] -> [d1, (d2), (d3)]
      dict(
          fn=lambda x: ragged_math_ops.range(mo.range(x)),
          elems=[5, 0, 3],
          expected_output=[[[], [0], [0, 1], [0, 1, 2], [0, 1, 2, 3]], [],
                           [[], [0], [0, 1]]],
          result_dtype=ragged_tensor.RaggedTensorType(
              dtype=dtypes.int64, ragged_rank=2),
      ),
      # [d1, (d2), (d3), (d4a), (d5)] ->  [d1, (d2), (d3), (d4b), (d5)]
      dict(
          fn=lambda x: x + np.int64(1),
          elems=[[[[[1, 2, 3]], [[4], [5]]]], [[[[6, 7]]], [[[8], []]]]],
          expected_output=[[[[[2, 3, 4]], [[5], [6]]]], [[[[7, 8]]], [[[9],
                                                                       []]]]],
          result_dtype=ragged_tensor.RaggedTensorType(
              dtype=dtypes.int64, ragged_rank=4),
      ),
      # [d1] -> [d1, (d2), (d3)]
      dict(
          fn=ragged_math_ops.range,
          elems=np.array([1, 2, 3], np.int64),
          expected_output=[[[0]], [[0, 1]], [[0, 1, 2]]],
          result_dtype=ragged_tensor.RaggedTensorType(
              dtype=dtypes.int64, ragged_rank=2)),
      # [0] -> [0, (d2), (d3)]  (github issue #36232)
      dict(
          fn=ragged_math_ops.range,
          elems=np.zeros([0], np.int64),
          expected_output=[],
          expected_ragged_rank=2,
          result_dtype=ragged_tensor.RaggedTensorType(
              dtype=dtypes.int64, ragged_rank=2)),
  ])

  def testRaggedMap(
      self,
      fn,
      elems,
      expected_output,
      expected_ragged_rank=None,
      result_ragged_rank=None,
      elems_ragged_rank=None,
      elems_dtype=dtypes.int64,
      result_dtype=None,
      infer_shape=True,
  ):
    # [FUNCTION DOC]
    # KNOWN ISSUE: parameter result_ragged_rank is accepted but never used.
    # Parameterized over callbacks mapping rows to dense, ragged or deeper ragged outputs
    # (including an empty input). Builds the ragged input from nested lists with the given dtype
    # and ragged rank, maps the callback with the given result dtype and shape-inference flag,
    # and asserts the output equals the expected ragged constant (with the expected ragged rank).
    elems = ragged_factory_ops.constant(elems, elems_dtype, elems_ragged_rank)
    output = ragged_map_ops.map_fn(
        fn=fn, elems=elems, dtype=result_dtype, infer_shape=infer_shape)

    expected_rt = ragged_factory_ops.constant(
        expected_output, ragged_rank=expected_ragged_rank)
    self.assertAllEqual(expected_rt, output)

  def testRaggedMapOnStructure(self):
    # [FUNCTION DOC]
    # Maps over a mapping of two ragged tensors (one ten times the other) with a callback that
    # sums everything in both entries per row; expects per-row totals 66, 44 and 198.
    batman = ragged_factory_ops.constant([[1, 2, 3], [4], [5, 6, 7]])
    # [[10, 20, 30], [40], [50, 60, 70]]
    robin = ragged_functional_ops.map_flat_values(mo.multiply, batman, 10)

    features = {'batman': batman, 'robin': robin}

    def _reduce_sum_from_all(f):
      # [FUNCTION DOC]
      # HELPER: callback. Input: f, mapping with the two row slices. Output: scalar sum of all
      # values in both entries.
      return mo.reduce_sum(f['batman']) + mo.reduce_sum(f['robin'])

    output = ragged_map_ops.map_fn(
        fn=_reduce_sum_from_all,
        elems=features,
        dtype=dtypes.int32,
    )

    self.assertAllEqual(output, [66, 44, 198])

  # Test mapping over a dict of RTs can produce a dict of RTs.
  def testRaggedMapOnStructure_RaggedOutputs(self):
    # [FUNCTION DOC]
    # Maps over a mapping of two ragged tensors with a callback returning a mapping of
    # incremented ragged rows, declaring a ragged output type per entry and disabling shape
    # inference; asserts both output entries equal the incremented inputs.
    batman = ragged_factory_ops.constant([[1, 2, 3], [4], [5, 6, 7]])
    # [[10, 20, 30], [40], [50, 60, 70]]
    robin = ragged_functional_ops.map_flat_values(mo.multiply, batman, 10)

    features = {'batman': batman, 'robin': robin}

    def _increment(f):
      # [FUNCTION DOC]
      # HELPER: callback. Input: f, mapping with two row slices. Output: mapping with each
      # entry incremented by one.
      return {
          'batman': f['batman'] + 1,
          'robin': f['robin'] + 1,
      }

    output = ragged_map_ops.map_fn(
        fn=_increment,
        elems=features,
        infer_shape=False,
        dtype={
            'batman':
                ragged_tensor.RaggedTensorType(
                    dtype=dtypes.int32, ragged_rank=1),
            'robin':
                ragged_tensor.RaggedTensorType(
                    dtype=dtypes.int32, ragged_rank=1)
        },
    )

    self.assertAllEqual(output['batman'], [[2, 3, 4], [5], [6, 7, 8]])
    self.assertAllEqual(output['robin'], [[11, 21, 31], [41], [51, 61, 71]])

  def testZip(self):
    # [FUNCTION DOC]
    # Maps over a pair of (row index column, ragged row) and pairs each element with its row
    # index by tiling the index and stacking; asserts the ragged result of index/value pairs.
    x = ragged_factory_ops.constant(
        [[10, 20], [30, 40], [50, 60], [70], [80, 90, 100]], dtypes.int64)
    y = array_ops.expand_dims(mo.range(x.nrows(out_type=dtypes.int64)), axis=1)

    def _zip(foo):
      # [FUNCTION DOC]
      # HELPER: callback. Input: foo, a pair (row index vector, row values). Output: matrix
      # pairing the repeated index with each value.
      y_val, x_val = foo
      bar = array_ops.tile(y_val, array_ops.shape(x_val))
      return array_ops_stack.stack([bar, x_val], axis=1)

    output = ragged_map_ops.map_fn(
        _zip, (y, x),
        dtype=ragged_tensor.RaggedTensorType(dtype=dtypes.int64, ragged_rank=1),
        infer_shape=False)

    self.assertAllEqual(
        output, [[[0, 10], [0, 20]], [[1, 30], [1, 40]], [[2, 50], [2, 60]],
                 [[3, 70]], [[4, 80], [4, 90], [4, 100]]])

  def testBatchGather(self):
    # [FUNCTION DOC]
    # Maps over a pair of ragged tensors (tokens, indices) gathering the indexed tokens per row;
    # asserts the ragged string result.
    tokens = ragged_factory_ops.constant([['hello', '.', 'there'], ['merhaba'],
                                          ['bonjour', '.', 'ca va', '?']])
    indices = ragged_factory_ops.constant([[0, 2], [0], [0, 2]])

    def gather(x):
      # [FUNCTION DOC]
      # HELPER: callback. Input: x, pair (row tokens, row indices). Output: tokens at indices.
      tokens_val, indices_val = x
      return array_ops.gather(tokens_val, indices_val)

    data = tokens, indices
    out = ragged_map_ops.map_fn(
        gather,
        data,
        dtype=ragged_tensor.RaggedTensorType(
            dtype=dtypes.string, ragged_rank=1),
        infer_shape=False)

    self.assertAllEqual(
        out, [[b'hello', b'there'], [b'merhaba'], [b'bonjour', b'ca va']])

  def testMismatchRaggedRank(self):
    # [FUNCTION DOC]
    # Maps a column-sum callback but declares an output ragged rank of 23; expects a value error
    # saying the function was expected to return one type but returned another.
    elems = ragged_factory_ops.constant([[[1, 2, 3]], [[4, 5], [6, 7]]])
    fn = lambda x: ragged_math_ops.reduce_sum(x, axis=0)
    with self.assertRaisesRegex(
        ValueError, r'(?s)Expected `fn` to return.*But it returned.*'):
      _ = ragged_map_ops.map_fn(
          fn,
          elems,
          dtype=ragged_tensor.RaggedTensorType(
              dtype=dtypes.int64, ragged_rank=23))

  def testMismatchRaggedRank2(self):
    # [FUNCTION DOC]
    # Maps a callback wrapping rows in a ragged tensor but declares output ragged rank ten;
    # expects the same expected-versus-returned value error.
    elems = ragged_factory_ops.constant([[1, 2, 3], [4, 5], [6, 7]])
    fn = lambda x: ragged_tensor.RaggedTensor.from_row_starts(x, [0])
    with self.assertRaisesRegex(
        ValueError, r'(?s)Expected `fn` to return.*But it returned.*'):
      _ = ragged_map_ops.map_fn(
          fn,
          elems,
          dtype=ragged_tensor.RaggedTensorType(
              dtype=dtypes.int64, ragged_rank=10))

  def testMapOnSparseTensor(self):
    # [FUNCTION DOC]
    # Converts a small sparse tensor to a ragged tensor and maps the identity over it; asserts
    # the rows come back unchanged.
    s = sparse_tensor.SparseTensor(
        indices=[[0, 0], [0, 1], [1, 0], [1, 1]],
        values=[0, 5, 0, 4],
        dense_shape=[2, 2],
    )
    t2 = ragged_tensor.RaggedTensor.from_sparse(s)
    id_t2 = ragged_map_ops.map_fn(
        lambda x: x, t2,
    )
    self.assertAllEqual(id_t2, [[0, 5], [0, 4]])

  def testRaggedMapWithIncorrectFnOutputSignature(self):
    # [FUNCTION DOC]
    # Runs a nested map without an output signature where inner results have unequal lengths;
    # evaluating it must raise an invalid-argument error about incompatible flat-value shapes.
    x = ragged_factory_ops.constant([[1, 2, 3, 4], [1]])
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                'All flat_values must have compatible shapes'):
      y = map_fn_lib.map_fn(lambda r: map_fn_lib.map_fn(lambda y: r, r), x)
      self.evaluate(y)

  def testNestedRaggedMapWithFnOutputSignature(self):
    # [FUNCTION DOC]
    # Runs the same nested map but with explicit ragged output specs for inner and outer maps;
    # asserts each element is replicated per inner element, giving the expected nested result.
    ragged1d = ragged_tensor.RaggedTensorSpec([None], dtypes.int32)
    ragged2d = ragged_tensor.RaggedTensorSpec([None, None], dtypes.int32)

    x = ragged_factory_ops.constant([[1, 2, 3, 4], [1]])
    # pylint: disable=g-long-lambda
    y = map_fn_lib.map_fn(
        lambda r: map_fn_lib.map_fn(
            lambda y: r, r, fn_output_signature=ragged1d),
        x,
        fn_output_signature=ragged2d)
    expected = [[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]], [[1]]]
    self.assertAllEqual(y, expected)


if __name__ == '__main__':
  googletest.main()
