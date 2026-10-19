# [MODULE DOC]
# Graph-mode (v1 only) tests for the experimental map_defun op, which applies a traced function
# to each slice along the first dimension of its inputs. Covers output shapes and types, shape
# inference, error propagation, cancellation, captured inputs, and variant/string tensors.
# Test module only; no import-time side effects.
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
"""Tests for MapDefunOp."""
import time

from absl.testing import parameterized

from tensorflow.python.client import session
from tensorflow.python.data.experimental.ops import map_defun
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.eager import def_function
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import test


# TODO(b/123903858): Add eager and V2 test coverage
def _test_combinations():
  # [FUNCTION DOC]
  # HELPER: returns the test combinations used by every test here: TF API version 1 in graph
  # mode only.
  return combinations.combine(tf_api_version=[1], mode=["graph"])


class MapDefunTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for map_defun. Each test defines a small traced function with a fixed input
  # signature and maps it over a constant or placeholder input in graph mode.

  @combinations.generate(_test_combinations())
  def testNoIntraOpLimit(self):
    # [FUNCTION DOC]
    # Maps a function computing twice the input plus three over a 3x2 integer constant with
    # intra-op parallelism limit zero (no limit); asserts the result equals the expected
    # elementwise computation.

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec([2], dtypes.int32)])
    def simple_fn(x):
      # [FUNCTION DOC]
      # HELPER: traced function returning twice the input plus three.
      return x * 2 + 3

    nums = [[1, 2], [3, 4], [5, 6]]
    elems = constant_op.constant(nums, dtype=dtypes.int32, name="data")
    r = map_defun.map_defun(
        simple_fn, [elems], [dtypes.int32], [(2,)],
        max_intra_op_parallelism=0)[0]
    expected = elems * 2 + 3
    self.assertAllEqual(self.evaluate(r), self.evaluate(expected))

  @combinations.generate(_test_combinations())
  def testMapDefunSimple(self):
    # [FUNCTION DOC]
    # Maps a function computing twice the input plus three over a 3x2 integer constant with
    # default settings; asserts the result equals the expected elementwise computation.

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec([2], dtypes.int32)])
    def simple_fn(x):
      # [FUNCTION DOC]
      # HELPER: traced function returning twice the input plus three.
      return x * 2 + 3

    nums = [[1, 2], [3, 4], [5, 6]]
    elems = constant_op.constant(nums, dtype=dtypes.int32, name="data")
    r = map_defun.map_defun(simple_fn, [elems], [dtypes.int32], [(2,)])[0]
    expected = elems * 2 + 3
    self.assertAllEqual(self.evaluate(r), self.evaluate(expected))

  @combinations.generate(_test_combinations())
  def testMapDefunMismatchedTypes(self):
    # [FUNCTION DOC]
    # The mapped function casts its input to float64 but the declared output type is int32;
    # asserts evaluating the result raises an invalid-argument error.

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec([], dtypes.int32)])
    def fn(x):
      # [FUNCTION DOC]
      # HELPER: traced function casting its input to 64-bit float.
      return math_ops.cast(x, dtypes.float64)

    nums = [1, 2, 3, 4, 5, 6]
    elems = constant_op.constant(nums, dtype=dtypes.int32, name="data")
    r = map_defun.map_defun(fn, [elems], [dtypes.int32], [()])[0]
    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(r)

  @combinations.generate(_test_combinations())
  def testMapDefunReduceDim(self):
    # [FUNCTION DOC]
    # The mapped function takes the first element of each length-2 row (output rank lower than
    # input rank); asserts the result is the first column [1, 3, 5].
    # Tests where the output has a different rank from the input

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec([2], dtypes.int32)])
    def fn(x):
      # [FUNCTION DOC]
      # HELPER: traced function gathering element zero of its input vector.
      return array_ops.gather(x, 0)

    nums = [[1, 2], [3, 4], [5, 6]]
    elems = constant_op.constant(nums, dtype=dtypes.int32, name="data")
    r = map_defun.map_defun(fn, [elems], [dtypes.int32], [()])[0]
    expected = constant_op.constant([1, 3, 5])
    self.assertAllEqual(self.evaluate(r), self.evaluate(expected))

  @combinations.generate(_test_combinations())
  def testMapDefunMultipleOutputs(self):
    # [FUNCTION DOC]
    # The mapped function returns two outputs of different types (the input, and a float64
    # twice-plus-three of it); asserts both mapped results match the expected values.

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec([2], dtypes.int32)])
    def fn(x):
      # [FUNCTION DOC]
      # HELPER: traced function returning the input and a float64 twice-plus-three of it.
      return (x, math_ops.cast(x * 2 + 3, dtypes.float64))

    nums = [[1, 2], [3, 4], [5, 6]]
    elems = constant_op.constant(nums, dtype=dtypes.int32, name="data")
    r = map_defun.map_defun(fn, [elems], [dtypes.int32, dtypes.float64], [(2,),
                                                                          (2,)])
    expected = [elems, elems * 2 + 3]
    self.assertAllEqual(self.evaluate(r), self.evaluate(expected))

  @combinations.generate(_test_combinations())
  def testMapDefunShapeInference(self):
    # [FUNCTION DOC]
    # Maps an identity function over a 3x2 constant; asserts the result's static shape is
    # (3, 2).

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec([2], dtypes.int32)])
    def fn(x):
      # [FUNCTION DOC]
      # HELPER: traced identity function.
      return x

    nums = [[1, 2], [3, 4], [5, 6]]
    elems = constant_op.constant(nums, dtype=dtypes.int32, name="data")
    result = map_defun.map_defun(fn, [elems], [dtypes.int32], [(2,)])[0]
    self.assertEqual(result.get_shape(), (3, 2))

  @combinations.generate(_test_combinations())
  def testMapDefunPartialShapeInference(self):
    # [FUNCTION DOC]
    # Maps an identity function over a placeholder with unknown leading dimension and row
    # length 2; asserts the static output shape is unknown-by-2.

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec([2], dtypes.int32)])
    def fn(x):
      # [FUNCTION DOC]
      # HELPER: traced identity function.
      return x

    elems = array_ops.placeholder(dtypes.int64, (None, 2))
    result = map_defun.map_defun(fn, [elems], [dtypes.int32], [(2,)])
    self.assertEqual(result[0].get_shape().as_list(), [None, 2])

  @combinations.generate(_test_combinations())
  def testMapDefunRaisesErrorOnRuntimeShapeMismatch(self):
    # [FUNCTION DOC]
    # Maps a two-input function over two placeholders fed with different leading dimensions
    # (5 and 3); asserts running raises an invalid-argument error stating all inputs must have
    # the same dimension 0.

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(None, dtypes.int32),
        tensor_spec.TensorSpec(None, dtypes.int32)
    ])
    def fn(x, y):
      # [FUNCTION DOC]
      # HELPER: traced function returning both of its inputs.
      return x, y

    elems1 = array_ops.placeholder(dtypes.int32)
    elems2 = array_ops.placeholder(dtypes.int32)
    result = map_defun.map_defun(fn, [elems1, elems2],
                                 [dtypes.int32, dtypes.int32], [(), ()])
    with self.cached_session() as sess:
      with self.assertRaisesWithPredicateMatch(
          errors.InvalidArgumentError,
          "All inputs must have the same dimension 0."):
        sess.run(result, feed_dict={elems1: [1, 2, 3, 4, 5], elems2: [1, 2, 3]})

  @combinations.generate(_test_combinations())
  def testMapDefunRaisesDefunError(self):
    # [FUNCTION DOC]
    # The mapped function asserts its input equals zero; over input [0, 0, 0, 37, 0] the
    # evaluation must raise an invalid-argument error.

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec([], dtypes.int32)])
    def fn(x):
      # [FUNCTION DOC]
      # HELPER: traced function that asserts its input is zero and returns it.
      with ops.control_dependencies([check_ops.assert_equal(x, 0)]):
        return array_ops.identity(x)

    elems = constant_op.constant([0, 0, 0, 37, 0])
    result = map_defun.map_defun(fn, [elems], [dtypes.int32], [()])
    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(result)

  @combinations.generate(_test_combinations())
  def testMapDefunCancelledCorrectly(self):
    # [FUNCTION DOC]
    # The mapped function gathers an out-of-range index from a length-5 vector, over 100 rows;
    # asserts evaluation raises an invalid-argument error matching the out-of-range index
    # message (the other mapped calls are cancelled).

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec([5], dtypes.int64)])
    def defun(x):
      # [FUNCTION DOC]
      # HELPER: traced function that gathers index 10 from a length-5 vector (always fails).
      # x has leading dimension 5, this will raise an error
      return array_ops.gather(x, 10)

    c = array_ops.tile(
        array_ops.expand_dims(
            constant_op.constant([1, 2, 3, 4, 5], dtype=dtypes.int64), 0),
        [100, 1])
    map_defun_op = map_defun.map_defun(defun, [c], [dtypes.int64], [()])[0]
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                r"indices = 10 is not in \[0, 5\)"):
      self.evaluate(map_defun_op)

  @combinations.generate(_test_combinations())
  def testMapDefunWithUnspecifiedOutputShape(self):
    # [FUNCTION DOC]
    # Maps a function with three outputs while declaring unknown, partially unknown and fully
    # known output shapes; asserts each mapped output equals the expected values.

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec([2], dtypes.int32)])
    def simple_fn(x):
      # [FUNCTION DOC]
      # HELPER: traced function returning r, r plus one and r plus two, where r is twice the
      # input plus three.
      res = x * 2 + 3
      return (res, res + 1, res + 2)

    nums = [[1, 2], [3, 4], [5, 6]]
    elems = constant_op.constant(nums, dtype=dtypes.int32, name="data")
    r = map_defun.map_defun(simple_fn, [elems],
                            [dtypes.int32, dtypes.int32, dtypes.int32],
                            [None, (None,), (2,)])
    expected = elems * 2 + 3
    self.assertAllEqual(self.evaluate(r[0]), self.evaluate(expected))
    self.assertAllEqual(self.evaluate(r[1]), self.evaluate(expected + 1))
    self.assertAllEqual(self.evaluate(r[2]), self.evaluate(expected + 2))

  @combinations.generate(_test_combinations())
  def testMapDefunWithDifferentOutputShapeEachRun(self):
    # [FUNCTION DOC]
    # Maps a function over a placeholder of unknown shape with unknown declared output shape;
    # runs the session twice with differently shaped feeds and asserts each result.

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec(None, dtypes.int32)])
    def simple_fn(x):
      # [FUNCTION DOC]
      # HELPER: traced function returning twice the input plus three.
      return x * 2 + 3

    elems = array_ops.placeholder(dtypes.int32, name="data")
    r = map_defun.map_defun(simple_fn, [elems], [dtypes.int32], [None])[0]
    with session.Session() as sess:
      self.assertAllEqual(sess.run(r, feed_dict={elems: [0]}), [3])
      self.assertAllEqual(
          sess.run(r, feed_dict={elems: [[0], [1]]}), [[3], [5]])

  @combinations.generate(_test_combinations())
  def testMapDefunWithWrongOutputShape(self):
    # [FUNCTION DOC]
    # Declares output shape (1,) while the function actually yields shape (2,); asserts
    # evaluation raises an invalid-argument error.

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec([2], dtypes.int32)])
    def simple_fn(x):
      # [FUNCTION DOC]
      # HELPER: traced function returning twice the input plus three.
      return x * 2 + 3

    nums = [[1, 2], [3, 4], [5, 6]]
    elems = constant_op.constant(nums, dtype=dtypes.int32, name="data")
    r = map_defun.map_defun(simple_fn, [elems], [dtypes.int32], [(1,)])[0]
    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(r)

  @combinations.generate(_test_combinations())
  def testMapDefunWithInvalidInput(self):
    # [FUNCTION DOC]
    # A scalar input has no dimension to map over: with a known shape, graph construction
    # raises a value error; with a placeholder fed a scalar, running raises an invalid-argument
    # error.

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec(None, dtypes.int32)])
    def simple_fn(x):
      # [FUNCTION DOC]
      # HELPER: traced function returning twice its input.
      return x * 2

    c = constant_op.constant(2)
    with self.assertRaises(ValueError):
      # Fails at graph construction time for inputs with known shapes.
      r = map_defun.map_defun(simple_fn, [c], [dtypes.int32], [None])[0]
    p = array_ops.placeholder(dtypes.int32)
    r = map_defun.map_defun(simple_fn, [p], [dtypes.int32], [None])[0]
    with session.Session() as sess:
      with self.assertRaises(errors.InvalidArgumentError):
        sess.run(r, feed_dict={p: 0})

  @combinations.generate(_test_combinations())
  def testMapDefunWithParentCancellation(self):
    # [FUNCTION DOC]
    # Starts a map whose function blocks dequeuing from an empty queue, runs a helper thread
    # that asserts the op is cancelled, then closes the session after a short sleep and joins
    # the thread. Verifies session cancellation propagates into the map op.
    # Checks that a cancellation of the parent graph is threaded through to
    # MapDefunOp correctly.
    @def_function.function(
        input_signature=[tensor_spec.TensorSpec([], dtypes.int32)])
    def simple_fn(x):
      # [FUNCTION DOC]
      # HELPER: traced function ignoring its input and blocking on dequeuing five items from an
      # empty queue.
      del x
      queue = data_flow_ops.FIFOQueue(10, dtypes.int32, ())
      # Blocking
      return queue.dequeue_many(5)

    c = constant_op.constant([1, 2, 3, 4, 5])
    map_defun_op = map_defun.map_defun(simple_fn, [c], [dtypes.int32], [()])[0]

    with self.cached_session() as sess:
      thread = self.checkedThread(
          self.assert_op_cancelled, args=(map_defun_op,))
      thread.start()
      time.sleep(0.2)
      sess.close()
      thread.join()

  @combinations.generate(_test_combinations())
  def testMapDefunWithCapturedInputs(self):
    # [FUNCTION DOC]
    # The mapped function adds a captured outer constant (2) to each element of [1, 2, 3, 4];
    # asserts the result equals the input plus the constant.
    c = constant_op.constant(2)

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec([], dtypes.int32)])
    def fn(x):
      # [FUNCTION DOC]
      # HELPER: traced function adding a captured constant to its input.
      return x + c

    x = constant_op.constant([1, 2, 3, 4])
    map_defun_op = map_defun.map_defun(fn, [x], [dtypes.int32], [()])[0]
    expected = x + c
    self.assertAllEqual(self.evaluate(expected), self.evaluate(map_defun_op))

  @combinations.generate(_test_combinations())
  def testMapDefunWithVariantTensor(self):
    # [FUNCTION DOC]
    # Serializes a sparse tensor to a variant, stacks two copies, maps an identity function
    # over them, deserializes the result, and asserts the sparse value has batch dimension 2
    # with the expected indices, values and dense shape.

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec([], dtypes.variant)])
    def fn(x):
      # [FUNCTION DOC]
      # HELPER: traced identity function.
      return x

    st = sparse_tensor.SparseTensor(
        indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])

    serialized = sparse_ops.serialize_sparse_v2(st, out_type=dtypes.variant)
    serialized = array_ops_stack.stack([serialized, serialized])
    map_defun_op = map_defun.map_defun(fn, [serialized], [dtypes.variant],
                                       [None])[0]
    deserialized = sparse_ops.deserialize_sparse(map_defun_op, dtypes.int32)
    expected = sparse_tensor.SparseTensorValue(
        indices=[[0, 0, 0], [0, 1, 2], [1, 0, 0], [1, 1, 2]],
        values=[1, 2, 1, 2],
        dense_shape=[2, 3, 4])
    actual = self.evaluate(deserialized)
    self.assertValuesEqual(expected, actual)

  @combinations.generate(_test_combinations())
  def testMapDefunWithVariantTensorAsCaptured(self):
    # [FUNCTION DOC]
    # Like the variant test, but the serialized sparse variant is captured by the mapped
    # function (which ignores its integer input and returns it) over two mapped elements;
    # asserts the deserialized sparse value equals the expected batch of two.

    st = sparse_tensor.SparseTensor(
        indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])
    serialized = sparse_ops.serialize_sparse_v2(st, out_type=dtypes.variant)

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec([], dtypes.int32)])
    def fn(x):
      # [FUNCTION DOC]
      # HELPER: traced function ignoring its input and returning a captured variant tensor.
      del x
      return serialized

    x = constant_op.constant([0, 0])
    map_defun_op = map_defun.map_defun(fn, [x], [dtypes.variant], [None])[0]
    deserialized = sparse_ops.deserialize_sparse(map_defun_op, dtypes.int32)
    expected = sparse_tensor.SparseTensorValue(
        indices=[[0, 0, 0], [0, 1, 2], [1, 0, 0], [1, 1, 2]],
        values=[1, 2, 1, 2],
        dense_shape=[2, 3, 4])
    actual = self.evaluate(deserialized)
    self.assertValuesEqual(expected, actual)

  @combinations.generate(_test_combinations())
  def testMapDefunWithStrTensor(self):
    # [FUNCTION DOC]
    # Same as the variant test but the sparse tensor is serialized to strings; maps an
    # identity function, deserializes, and asserts the expected sparse batch of two.

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec([], dtypes.string)])
    def fn(x):
      # [FUNCTION DOC]
      # HELPER: traced identity function.
      return x

    st = sparse_tensor.SparseTensor(
        indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])

    serialized = sparse_ops.serialize_sparse_v2(st, out_type=dtypes.string)
    serialized = array_ops_stack.stack([serialized, serialized])
    map_defun_op = map_defun.map_defun(fn, [serialized], [dtypes.string],
                                       [None])[0]
    deserialized = sparse_ops.deserialize_sparse(map_defun_op, dtypes.int32)
    expected = sparse_tensor.SparseTensorValue(
        indices=[[0, 0, 0], [0, 1, 2], [1, 0, 0], [1, 1, 2]],
        values=[1, 2, 1, 2],
        dense_shape=[2, 3, 4])
    actual = self.evaluate(deserialized)
    self.assertValuesEqual(expected, actual)


if __name__ == "__main__":
  test.main()
