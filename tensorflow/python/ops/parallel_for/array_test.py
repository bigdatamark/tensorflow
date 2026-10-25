# [MODULE DOC]
# Tests that the pfor vectorizer produces the same results as running the loop body
# iteration by iteration for array-manipulation kernels (gather, reshape, slicing, padding,
# splitting, diagonals, space/batch conversion and similar). Contains one test class whose
# methods each build a loop body around one op or family of ops and compare vectorized against
# sequential execution through the shared PFor test case helper. Runs in graph and eager modes.
# Import-time side effects: none beyond imports; running the file as a script launches the
# TensorFlow test runner.
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
"""Tests for vectorization of array kernels."""

from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.ops.parallel_for import control_flow_ops as pfor_control_flow_ops
from tensorflow.python.ops.parallel_for.test_util import PForTestCase
from tensorflow.python.platform import test


@test_util.with_eager_op_as_function
@test_util.run_all_in_graph_and_eager_modes
class ArrayTest(PForTestCase):
  # [CLASS DOC]
  # Test case covering pfor conversion of array ops. Derives from the shared PFor test base
  # class (which supplies the vectorized-versus-sequential comparison helper); decorated to
  # run every test in both graph and eager modes and with eager ops executed as functions.
  # Holds no state; each test creates its own random inputs.

  def test_gather(self):
    # [FUNCTION DOC]
    # Scenario: random 4-d tensor, also fed through a placeholder with unknown shape. The loop
    # body gathers with many index forms (scalar, int64, loop-dependent, lists, nested lists),
    # on axes first, third and last, plus batch-dimension variants including negative ones.
    # Expects vectorized outputs to equal sequential ones over 3 iterations.
    x = random_ops.random_uniform([3, 3, 3, 3])
    x2 = array_ops.placeholder_with_default(x, shape=None)  # Has dynamic shape.

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: Loop body for test_gather; returns a list of gather results for the i-th
      # slice and the full inputs under the index and axis variants.
      outputs = []
      x_i = array_ops.gather(x, i)
      for y in [x, x2, x_i]:
        for axis in [0, 2, -1]:
          outputs.append(array_ops.gather(y, 2, axis=axis))
          outputs.append(
              array_ops.gather(y, math_ops.cast(2, dtypes.int64), axis=axis))
          outputs.append(
              array_ops.gather(y, 2, axis=math_ops.cast(axis, dtypes.int64)))
          outputs.append(
              array_ops.gather(y, math_ops.cast(i, dtypes.int64), axis=axis))
          outputs.append(array_ops.gather(y, [i], axis=axis))
          outputs.append(array_ops.gather(y, [i, 2], axis=axis))
          outputs.append(array_ops.gather(y, [[2, i], [i, 1]], axis=axis))

        outputs.append(array_ops.gather(y, [0, 1, 2], axis=1, batch_dims=1))
        outputs.append(array_ops.gather(y, [i, 1, 2], axis=2, batch_dims=1))
        outputs.append(array_ops.gather(y, [[2, i], [i, 1], [2, 1]],
                                        axis=-1, batch_dims=1))
        outputs.append(
            array_ops.gather(y, [[0, 1, 2]] * 3, axis=2, batch_dims=2))
        outputs.append(array_ops.gather(y, [0, 1, 2], axis=1, batch_dims=-1))
        outputs.append(
            array_ops.gather(y, [[0, 1, 2]] * 3, axis=2, batch_dims=-2))

      return outputs

    self._test_loop_fn(loop_fn, 3)

  def test_gather_nd(self):
    # [FUNCTION DOC]
    # Scenario: random 3x3x3 tensor; the loop body applies n-d gather with a constant index,
    # a loop-dependent index and a batched loop-dependent index list. Expects vectorized
    # results to match sequential ones over 3 iterations.
    x = random_ops.random_uniform([3, 3, 3])

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: Loop body for test_gather_nd; returns the three n-d gather results.
      outputs = []
      x_i = array_ops.gather(x, i)
      outputs.append(array_ops.gather_nd(x_i, [0], batch_dims=0))
      outputs.append(array_ops.gather_nd(x_i, [i], batch_dims=0))
      outputs.append(array_ops.gather_nd(x_i, [[i], [i], [i]], batch_dims=1))
      return outputs

    self._test_loop_fn(loop_fn, 3)

  @test_util.run_v2_only
  def test_gather_pfor_grad(self):
    # [FUNCTION DOC]
    # Version-2-only test. Scenario: a zero tensor of shape one by two is watched by a gradient
    # tape; vectorized_map over a range of two gathers from it along the last axis using the
    # range element as index. Expects the gradient with respect to the tensor to be a one by
    # two matrix of ones.
    x = array_ops.zeros([1, 2])
    with backprop.GradientTape() as tape:
      tape.watch(x)
      r = pfor_control_flow_ops.vectorized_map(
          lambda t: array_ops.gather(x, t, axis=-1), math_ops.range(2))
    self.assertAllClose([[1., 1.]], tape.gradient(r, x))

  def test_shape(self):
    # [FUNCTION DOC]
    # Scenario: shape op on the i-th slice of a random tensor, with default and int64 output
    # type. Expects vectorized and sequential results to agree over 3 iterations.
    x = random_ops.random_uniform([3, 2, 3])

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: Loop body for test_shape; returns the shape of the slice in both output types.
      x_i = array_ops.gather(x, i)
      return array_ops.shape(x_i), array_ops.shape(x_i, out_type=dtypes.int64)

    self._test_loop_fn(loop_fn, 3)

  def test_size(self):
    # [FUNCTION DOC]
    # Scenario: element-count op on the i-th slice of a random tensor, with default and int64
    # output type. Expects vectorized and sequential results to agree over 3 iterations.
    x = random_ops.random_uniform([3, 2, 3])

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: Loop body for test_size; returns the size of the slice in both output types.
      x_i = array_ops.gather(x, i)
      return array_ops.size(x_i), array_ops.size(x_i, out_type=dtypes.int64)

    self._test_loop_fn(loop_fn, 3)

  def test_rank(self):
    # [FUNCTION DOC]
    # Scenario: rank op on the i-th slice of a random tensor. Expects vectorized and
    # sequential results to agree over 3 iterations.
    x = random_ops.random_uniform([3, 2, 3])

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: Loop body for test_rank; returns the rank of the i-th slice.
      x_i = array_ops.gather(x, i)
      return array_ops.rank(x_i)

    self._test_loop_fn(loop_fn, 3)

  def test_shape_n(self):
    # [FUNCTION DOC]
    # Scenario: two random tensors; the loop body asks for the shapes of a mix of loop-variant
    # slices and loop-invariant tensors in one call, with default and int64 output types.
    # Expects vectorized and sequential results to agree over 3 iterations.
    x = random_ops.random_uniform([3, 2, 3])
    y = random_ops.random_uniform([3])

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: Loop body for test_shape_n; returns the two lists of shapes.
      x_i = array_ops.gather(x, i)
      y_i = array_ops.gather(y, i)
      return array_ops.shape_n([x_i, x, y,
                                y_i]), array_ops.shape_n([x_i, x, y, y_i],
                                                         out_type=dtypes.int64)

    self._test_loop_fn(loop_fn, 3)

  def test_reshape(self):
    # [FUNCTION DOC]
    # Scenario: reshape of each slice of a random tensor to a flat vector and to a shape with
    # an inferred dimension. Expects vectorized and sequential results to agree (3 iterations).
    x = random_ops.random_uniform([3, 2, 3])

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: Loop body for test_reshape; returns the two reshaped versions of the slice.
      x1 = array_ops.gather(x, i)
      return array_ops.reshape(x1, [-1]), array_ops.reshape(x1, [1, 3, 1, -1])

    self._test_loop_fn(loop_fn, 3)

  def test_fill(self):
    # [FUNCTION DOC]
    # Scenario: fill a fixed two by three shape with the loop index as value. Expects
    # vectorized and sequential results to agree over 3 iterations.

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: Loop body for test_fill; returns a two by three tensor filled with i.
      return array_ops.fill((2, 3), i)

    self._test_loop_fn(loop_fn, 3)

  def test_broadcast_to(self):
    # [FUNCTION DOC]
    # Scenario: broadcast each slice of a random tensor (with a unit dimension) to two target
    # shapes, one adding a leading dimension and one expanding a unit dimension. Expects
    # vectorized and sequential results to agree over 3 iterations.
    x = random_ops.random_uniform([3, 2, 1, 3])

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: Loop body for test_broadcast_to; returns both broadcast results.
      x1 = array_ops.gather(x, i)
      return (array_ops.broadcast_to(x1, [2, 2, 3]),
              array_ops.broadcast_to(x1, [1, 2, 1, 3]))

    self._test_loop_fn(loop_fn, 3)

  def test_expand_dims(self):
    # [FUNCTION DOC]
    # Scenario: insert a new axis into each slice at the last, middle and an int64-specified
    # position. Expects vectorized and sequential results to agree over 3 iterations.
    x = random_ops.random_uniform([3, 2, 3])

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: Loop body for test_expand_dims; returns a list of the three expanded tensors.
      x1 = array_ops.gather(x, i)
      return [
          array_ops.expand_dims(x1, axis=-1),
          array_ops.expand_dims(x1, axis=1),
          array_ops.expand_dims(
              x1, axis=constant_op.constant(1, dtype=dtypes.int64))
      ]

    self._test_loop_fn(loop_fn, 3)

  def test_one_hot(self):
    # [FUNCTION DOC]
    # Scenario: random integer indices in zero to three; one-hot encode each slice with custom
    # on/off values and with the default values along a different axis. Expects vectorized and
    # sequential results to agree over 3 iterations.
    indices = random_ops.random_uniform([3, 2, 3],
                                        minval=0,
                                        maxval=4,
                                        dtype=dtypes.int32)

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: Loop body for test_one_hot; returns the two encodings of the i-th slice.
      indices_i = array_ops.gather(indices, i)
      return (array_ops.one_hot(indices_i, depth=4, on_value=2., off_value=-2.),
              array_ops.one_hot(indices_i, depth=4, axis=1))

    self._test_loop_fn(loop_fn, 3)

  def test_searchsorted(self):
    # [FUNCTION DOC]
    # Scenario: sorted inputs built by a cumulative sum of random values, plus loop-invariant
    # query values. The loop body runs a left-side search (int32 output) and a right-side
    # search (int64 output), exercising the lower-bound and upper-bound kernels. Expects
    # vectorized and sequential results to agree over 3 iterations.
    sorted_inputs = math_ops.cumsum(
        random_ops.random_uniform([3, 2, 4]), axis=-1)
    values = random_ops.random_uniform([2, 3], minval=-1, maxval=4.5)

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: Loop body for test_searchsorted; returns the two search results as a list.
      inputs_i = array_ops.gather(sorted_inputs, i)
      return [
          array_ops.searchsorted(
              inputs_i, values, out_type=dtypes.int32,
              side="left"),  # creates LowerBound op.
          array_ops.searchsorted(
              inputs_i, values, out_type=dtypes.int64, side="right")
      ]  # creates UpperBound op.

    self._test_loop_fn(loop_fn, 3)

  def test_slice(self):
    # [FUNCTION DOC]
    # Scenario: slice a fixed window out of each slice of a random tensor (constant begin and
    # size). Expects vectorized and sequential results to agree over 3 iterations.
    x = random_ops.random_uniform([3, 2, 3])

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: Loop body for test_slice; returns the fixed window of the i-th slice.
      x1 = array_ops.gather(x, i)
      return array_ops.slice(x1, begin=(0, 1), size=(2, 1))

    self._test_loop_fn(loop_fn, 3)

  def test_slice_loop_variant_begin(self):
    # [FUNCTION DOC]
    # Scenario: slice whose begin offsets depend on the loop index. Expects vectorized and
    # sequential results to agree over 3 iterations.
    x = random_ops.random_uniform([3, 2, 5, 3])

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: Loop body for test_slice_loop_variant_begin; slices with begin computed from i.
      x1 = array_ops.gather(x, i)
      return array_ops.slice(x1, begin=(0, 2 - i, i), size=(-1, 2, 1))

    self._test_loop_fn(loop_fn, 3)

  def test_tile(self):
    # [FUNCTION DOC]
    # Scenario: tile each slice of a random tensor with fixed multiples. Expects vectorized
    # and sequential results to agree over 3 iterations.
    x = random_ops.random_uniform([3, 2, 3])

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: Loop body for test_tile; returns the tiled i-th slice.
      x1 = array_ops.gather(x, i)
      return array_ops.tile(x1, [2, 1])

    self._test_loop_fn(loop_fn, 3)

  def test_tile_loop_dependent(self):
    # [FUNCTION DOC]
    # Scenario: tile whose multiples depend on the loop index, run through pfor with the
    # while-loop fallback disabled. Expects a value error whose message says the multiples are
    # expected to be loop invariant.
    x = random_ops.random_uniform([3, 2, 3])

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: Loop body for test_tile_loop_dependent; tiles with a loop-dependent multiple.
      x1 = array_ops.gather(x, i)
      return array_ops.tile(x1, [i, 1])

    with self.assertRaisesRegex(ValueError, "expected to be loop invariant"):
      pfor_control_flow_ops.pfor(loop_fn, 2, fallback_to_while_loop=False)

  def test_pack(self):
    # [FUNCTION DOC]
    # Scenario: stack a loop-variant slice with a loop-invariant tensor along the last axis.
    # Expects vectorized and sequential results to agree; note only a single iteration is run.
    x = random_ops.random_uniform([3, 2, 3])
    y = random_ops.random_uniform([2, 3])

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: Loop body for test_pack; returns the stacked pair.
      x1 = array_ops.gather(x, i)
      return array_ops_stack.stack([x1, y], axis=-1)

    self._test_loop_fn(loop_fn, 1)

  def test_unpack(self):
    # [FUNCTION DOC]
    # Scenario: unstack each slice of a random tensor along the last axis (four pieces) and
    # along the second axis (three pieces). Expects vectorized and sequential results to agree
    # over 3 iterations.
    x = random_ops.random_uniform([3, 2, 3, 4])

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: Loop body for test_unpack; returns both lists of unstacked tensors.
      x_i = array_ops.gather(x, i)
      return array_ops_stack.unstack(
          x_i, 4, axis=-1), array_ops_stack.unstack(
              x_i, 3, axis=1)

    self._test_loop_fn(loop_fn, 3)

  def test_pad(self):
    # [FUNCTION DOC]
    # Scenario: constant-mode padding of each slice with a loop-invariant padding matrix.
    # Expects vectorized and sequential results to agree over 3 iterations.
    x = random_ops.random_uniform([3, 2, 3])
    padding = constant_op.constant([[1, 2], [3, 4]])

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: Loop body for test_pad; returns the padded i-th slice.
      x1 = array_ops.gather(x, i)
      return array_ops.pad(x1, padding, mode="CONSTANT")

    self._test_loop_fn(loop_fn, 3)

  def test_pad_v2(self):
    # [FUNCTION DOC]
    # Scenario: same as test_pad but using the second-version padding op. Expects vectorized
    # and sequential results to agree over 3 iterations.
    x = random_ops.random_uniform([3, 2, 3])
    padding = constant_op.constant([[1, 2], [3, 4]])

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: Loop body for test_pad_v2; returns the padded i-th slice.
      x1 = array_ops.gather(x, i)
      return array_ops.pad_v2(x1, padding, mode="CONSTANT")

    self._test_loop_fn(loop_fn, 3)

  def test_split(self):
    # [FUNCTION DOC]
    # Scenario: split each slice into equal pieces (two along the first axis, three along the
    # last). Expects vectorized and sequential results to agree over 3 iterations.
    x = random_ops.random_uniform([3, 2, 3])

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: Loop body for test_split; returns both lists of pieces.
      x1 = array_ops.gather(x, i)
      return array_ops.split(x1, 2, axis=0), array_ops.split(x1, 3, axis=-1)

    self._test_loop_fn(loop_fn, 3)

  def test_split_v(self):
    # [FUNCTION DOC]
    # Scenario: split each slice into pieces of explicit unequal sizes (the variable-size split
    # op), along the first axis and the last axis. Expects vectorized and sequential results
    # to agree over 3 iterations.
    x = random_ops.random_uniform([3, 6, 3])

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: Loop body for test_split_v; returns both split results.
      x1 = array_ops.gather(x, i)
      return (array_ops.split(x1, [2, 1, 3],
                              axis=0), array_ops.split(x1, [3], axis=-1))

    self._test_loop_fn(loop_fn, 3)

  def test_squeeze(self):
    # [FUNCTION DOC]
    # Scenario: random tensor with unit dimensions; squeeze each slice along the first axis,
    # the last axis, and all unit axes. Expects vectorized and sequential results to agree
    # over 3 iterations (of the five slices available).
    x = random_ops.random_uniform([5, 1, 2, 1])

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: Loop body for test_squeeze; returns the three squeezed variants.
      x1 = array_ops.gather(x, i)
      return (array_ops.squeeze(x1, axis=0), array_ops.squeeze(x1, axis=-1),
              array_ops.squeeze(x1))

    self._test_loop_fn(loop_fn, 3)

  def test_reverse(self):
    # [FUNCTION DOC]
    # Scenario: reverse each slice along the first axis, along the last axis (negative index)
    # and along two axes at once. Expects vectorized and sequential results to agree over 3
    # iterations.
    x = random_ops.random_uniform([3, 4, 2, 3])

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: Loop body for test_reverse; returns the three reversed variants.
      x1 = array_ops.gather(x, i)
      return (array_ops.reverse(x1, axis=[0]),
              array_ops.reverse(x1, axis=[-1]),
              array_ops.reverse(x1, axis=[1, -1]))

    self._test_loop_fn(loop_fn, 3)

  def test_transpose(self):
    # [FUNCTION DOC]
    # Scenario: reverse the axis order of each rank-3 slice with an explicit permutation.
    # Expects vectorized and sequential results to agree over 3 iterations.
    x = random_ops.random_uniform([3, 2, 3, 4])

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: Loop body for test_transpose; returns the transposed i-th slice.
      x1 = array_ops.gather(x, i)
      return array_ops.transpose(x1, [2, 1, 0])

    self._test_loop_fn(loop_fn, 3)

  def test_top_k(self):
    # [FUNCTION DOC]
    # Scenario: top-two selection along the last axis of each slice (values and indices).
    # Expects vectorized and sequential results to agree over 3 iterations.
    x = random_ops.random_uniform([3, 2, 3, 4])

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: Loop body for test_top_k; returns the top-k result pair for the i-th slice.
      x1 = array_ops.gather(x, i)
      return nn.top_k(x1, k=2)

    self._test_loop_fn(loop_fn, 3)

  def test_conjugate_transpose(self):
    # [FUNCTION DOC]
    # Scenario: complex tensor built from two random real tensors; conjugate transpose of each
    # slice with an explicit permutation. Expects vectorized and sequential results to agree
    # over 3 iterations.
    x = math_ops.complex(
        random_ops.random_uniform([3, 2, 3, 4]),
        random_ops.random_uniform([3, 2, 3, 4]))

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: Loop body for test_conjugate_transpose; returns the conjugate-transposed slice.
      x_i = array_ops.gather(x, i)
      return array_ops.conjugate_transpose(x_i, [2, 1, 0])

    self._test_loop_fn(loop_fn, 3)

  def test_zeros_like(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the trailing comma in the loop body makes the zeros value a one-element
    # tuple rather than a tensor, so the "z + x1" output relies on implicit tuple-to-tensor
    # conversion and the test is not testing the plain zeros-like result as the name implies.
    # Scenario: zeros-like of each slice, returned alone and added to the slice. Expects
    # vectorized and sequential results to agree over 3 iterations.
    x = random_ops.random_uniform([3, 2, 3])

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: Loop body for test_zeros_like; returns the (tuple-wrapped) zeros and the sum.
      x1 = array_ops.gather(x, i)
      z = array_ops.zeros_like(x1),
      return z, z + x1

    self._test_loop_fn(loop_fn, 3)

  def test_ones_like(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: same trailing-comma problem as test_zeros_like; the ones value is a
    # one-element tuple, not a tensor.
    # Scenario: ones-like of each slice, returned alone and added to the slice. Expects
    # vectorized and sequential results to agree over 3 iterations.
    x = random_ops.random_uniform([3, 2, 3])

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: Loop body for test_ones_like; returns the (tuple-wrapped) ones and the sum.
      x1 = array_ops.gather(x, i)
      z = array_ops.ones_like(x1),
      return z, z + x1

    self._test_loop_fn(loop_fn, 3)

  def test_concat_v2(self):
    # [FUNCTION DOC]
    # Scenario: concatenate two copies of a loop-variant slice with a loop-invariant tensor
    # along the first axis, the last axis, and an int64-specified axis. Expects vectorized and
    # sequential results to agree over 3 iterations.
    x = random_ops.random_uniform([3, 2, 3])
    y = random_ops.random_uniform([2, 3])

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: Loop body for test_concat_v2; returns a list of the three concatenations.
      x1 = array_ops.gather(x, i)
      return [
          array_ops.concat([x1, x1, y], axis=0),
          array_ops.concat([x1, x1, y], axis=-1),
          array_ops.concat([x1, x1, y],
                           axis=constant_op.constant(0, dtype=dtypes.int64))
      ]

    self._test_loop_fn(loop_fn, 3)

  def test_unary_cwise_ops(self):
    # [FUNCTION DOC]
    # Scenario: for the identity and stop-gradient ops, a persistent gradient tape watches a
    # random matrix; the loop body applies the op to a row, adds the row, computes an l2 loss
    # and its gradient with respect to the row. Outputs the op applied to the whole matrix,
    # the sum, and the gradient. Expects vectorized and sequential results to agree over 3
    # iterations for each op.
    for op in [array_ops.identity, array_ops.stop_gradient]:
      with backprop.GradientTape(persistent=True) as g:
        x = random_ops.random_uniform([3, 5])
        g.watch(x)

      # pylint: disable=cell-var-from-loop
      def loop_fn(i):
        # [FUNCTION DOC]
        # HELPER: Loop body for test_unary_cwise_ops, closing over the current op and tape.
        # Returns the op on the full matrix, the op on the row plus the row, and the gradient
        # of the l2 loss of that sum with respect to the row.
        with g:
          x1 = array_ops.gather(x, i)
          y = op(x1) + x1
          loss = nn.l2_loss(y)
        return op(x), y, g.gradient(loss, x1)

      # pylint: enable=cell-var-from-loop

      self._test_loop_fn(loop_fn, 3)

  def test_identity_n(self):
    # [FUNCTION DOC]
    # Scenario: multi-input identity op applied to a loop-invariant matrix and one of its rows.
    # Expects vectorized and sequential results to agree over 3 iterations.
    x = random_ops.random_uniform([3, 4])

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: Loop body for test_identity_n; returns the identity-n outputs as a list.
      return array_ops.identity_n([x, array_ops.gather(x, i)])

    self._test_loop_fn(loop_fn, 3)

  def test_matrix_band_part(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the comparison call sits after the for loop rather than inside it, so only
    # the last band setting (one lower, one upper) is actually tested; the first two settings
    # are defined but never run.
    # Scenario: keep a band of each matrix slice of a random tensor for three lower/upper band
    # settings. Expects vectorized and sequential results to agree over 3 iterations.
    x = random_ops.random_uniform([3, 4, 2, 2])

    for num_lower, num_upper in ((0, -1), (-1, 0), (1, 1)):
      # pylint: disable=cell-var-from-loop
      def loop_fn(i):
        # [FUNCTION DOC]
        # HELPER: Loop body for test_matrix_band_part, closing over the current band limits.
        # Returns the banded i-th slice.
        return array_ops.matrix_band_part(
            array_ops.gather(x, i), num_lower=num_lower, num_upper=num_upper)

      # pylint: enable=cell-var-from-loop

    self._test_loop_fn(loop_fn, 3)

  def test_matrix_diag(self):
    # [FUNCTION DOC]
    # Scenario: build matrices from a band of diagonals (offset range zero to one) with
    # explicit row and column counts and right-left alignment. Expects vectorized and
    # sequential results to agree over 3 iterations.
    x = random_ops.random_uniform([3, 2, 4])

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: Loop body for test_matrix_diag; builds a matrix from the i-th diagonal band.
      diagonal = array_ops.gather(x, i)
      return array_ops.matrix_diag(
          diagonal, k=(0, 1), num_rows=4, num_cols=5, align="RIGHT_LEFT")

    self._test_loop_fn(loop_fn, 3)

  def test_matrix_diag_part(self):
    # [FUNCTION DOC]
    # Scenario: extract a band of diagonals (offsets minus two to zero) with a padding value of
    # three and right-left alignment from each matrix. Expects vectorized and sequential
    # results to agree over 3 iterations.
    x = random_ops.random_uniform([3, 4, 6])

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: Loop body for test_matrix_diag_part; returns the diagonal band of the matrix.
      input = array_ops.gather(x, i)  # pylint: disable=redefined-builtin
      return array_ops.matrix_diag_part(
          input, k=(-2, 0), padding_value=3, align="RIGHT_LEFT")

    self._test_loop_fn(loop_fn, 3)

  def test_diag(self):
    # [FUNCTION DOC]
    # Scenario: for a random matrix and a random rank-3 tensor, build a diagonal tensor from
    # each slice. Expects vectorized and sequential results to agree over 3 iterations. (The
    # second pylint comment says disable where enable was probably intended; harmless.)
    for x in (random_ops.random_uniform([3, 4]),
              random_ops.random_uniform([3, 4, 2])):
      # pylint: disable=cell-var-from-loop
      def loop_fn(i):
        # [FUNCTION DOC]
        # HELPER: Loop body for test_diag, closing over the current input; returns the
        # diagonal tensor built from the i-th slice.
        inp = array_ops.gather(x, i)
        return array_ops.diag(inp)

      # pylint: disable=cell-var-from-loop
      self._test_loop_fn(loop_fn, 3)

  def test_diag_part(self):
    # [FUNCTION DOC]
    # Scenario: for a random rank-3 tensor and a random rank-5 tensor, extract the diagonal
    # part of each slice. Expects vectorized and sequential results to agree over 3
    # iterations.
    for x in (random_ops.random_uniform([3, 2, 2]),
              random_ops.random_uniform([3, 4, 2, 4, 2])):
      # pylint: disable=cell-var-from-loop
      def loop_fn(i):
        # [FUNCTION DOC]
        # HELPER: Loop body for test_diag_part, closing over the current input; returns the
        # diagonal part of the i-th slice.
        inp = array_ops.gather(x, i)  # pylint: disable=redefined-builtin
        return array_ops.diag_part(inp)

      # pylint: disable=cell-var-from-loop
      self._test_loop_fn(loop_fn, 3)

  def test_matrix_set_diag(self):
    # [FUNCTION DOC]
    # Scenario: batches of matrices, diagonals and diagonal bands. The loop body overwrites
    # diagonals with loop-variant and loop-invariant combinations of matrix and diagonal, and
    # with a band (offsets minus one to one) under both alignments. Expects vectorized and
    # sequential results to agree over 3 iterations.
    matrices = random_ops.random_uniform([3, 4, 4])
    diags = random_ops.random_uniform([3, 4])
    bands = random_ops.random_uniform([3, 3, 4])

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: Loop body for test_matrix_set_diag; returns a list of the set-diagonal results
      # for every variant/invariant combination and alignment.
      matrix_i = array_ops.gather(matrices, i)
      diag_i = array_ops.gather(diags, i)
      results = [
          array_ops.matrix_set_diag(matrix_i, diag_i),
          array_ops.matrix_set_diag(matrices[0, ...], diag_i),
          array_ops.matrix_set_diag(matrix_i, diags[0, ...]),
      ]

      k = (-1, 1)
      band_i = array_ops.gather(bands, i)
      for align in ["RIGHT_LEFT", "LEFT_RIGHT"]:
        results.extend([
            array_ops.matrix_set_diag(matrix_i, band_i, k=k, align=align),
            array_ops.matrix_set_diag(
                matrices[0, ...], band_i, k=k, align=align),
            array_ops.matrix_set_diag(
                matrix_i, bands[0, ...], k=k, align=align)
        ])
      return results

    self._test_loop_fn(loop_fn, 3)

  def test_strided_slice(self):
    # [FUNCTION DOC]
    # Scenario: a persistent tape watches a random rank-7 tensor; the loop body applies a
    # complex index expression (ranges, strides, ellipsis, new axis, fixed index) to a slice,
    # takes an l2 loss and returns the result together with the gradient with respect to the
    # slice. Expects vectorized and sequential results to agree over 3 iterations.
    with backprop.GradientTape(persistent=True) as g:
      x = random_ops.random_uniform([3, 3, 4, 4, 2, 2, 2])
      g.watch(x)

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: Loop body for test_strided_slice; returns the strided-slice output and the
      # gradient of its l2 loss with respect to the slice.
      with g:
        x_i = array_ops.gather(x, i)
        y = x_i[:2, ::2, 1::3, ..., array_ops.newaxis, 1]
        loss = nn.l2_loss(y)
      return y, g.gradient(loss, x_i)

    self._test_loop_fn(loop_fn, 3)

  def test_strided_slice_loop_variant(self):
    # [FUNCTION DOC]
    # Scenario: strided slice whose bounds depend on the loop index. With the while-loop
    # fallback enabled, expects vectorized results to match sequential ones (the conversion
    # failure is handled). With the fallback disabled, expects a value error saying the
    # bounds are expected to be loop invariant.
    x = random_ops.random_uniform([3, 3, 4, 4, 2, 2, 2])

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: Loop body for test_strided_slice_loop_variant; slices from i to i plus one.
      x_i = array_ops.gather(x, i)
      return x_i[i:i+1, ...]

    # Test the fallback to while loop for a ConversionNotImplementedError is
    # handled.
    self._test_loop_fn(loop_fn, 3, fallback_to_while_loop=True)
    # Without fallback, ValueError is thrown.
    with self.assertRaisesRegex(ValueError, "expected to be loop invariant"):
      self._test_loop_fn(loop_fn, 3, fallback_to_while_loop=False)

  def test_depth_to_space(self):
    # [FUNCTION DOC]
    # Scenario: depth-to-space rearrangement with block size two in channels-last layout on
    # each slice of a random tensor. Expects vectorized and sequential results to agree over 2
    # iterations.
    x = random_ops.random_uniform([2, 3, 2, 2, 12])

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: Loop body for test_depth_to_space; returns the rearranged i-th slice.
      x1 = array_ops.gather(x, i)
      return array_ops.depth_to_space(x1, 2, data_format="NHWC")

    self._test_loop_fn(loop_fn, 2)

  def test_space_to_depth(self):
    # [FUNCTION DOC]
    # Scenario: space-to-depth rearrangement with block size two in channels-last layout on
    # each slice of a random tensor. Expects vectorized and sequential results to agree over 2
    # iterations.
    x = random_ops.random_uniform([2, 3, 12, 12, 3])

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: Loop body for test_space_to_depth; returns the rearranged i-th slice.
      x1 = array_ops.gather(x, i)
      return array_ops.space_to_depth(x1, 2, data_format="NHWC")

    self._test_loop_fn(loop_fn, 2)

  def test_batch_to_space_nd(self):
    # [FUNCTION DOC]
    # Scenario: n-d batch-to-space with block shape two by three and crops, applied to each
    # slice of a random tensor. Expects vectorized and sequential results to agree over 7
    # iterations.
    x = random_ops.random_uniform([7, 5 * 2 * 3, 2, 2, 3, 2])
    block_shapes = [2, 3]
    crops = [[1, 2], [1, 0]]

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: Loop body for test_batch_to_space_nd; returns the converted i-th slice.
      x1 = array_ops.gather(x, i)
      return array_ops.batch_to_space_nd(x1, block_shapes, crops)

    self._test_loop_fn(loop_fn, 7)

  def test_space_to_batch_nd(self):
    # [FUNCTION DOC]
    # Scenario: n-d space-to-batch with block shape two by three and paddings, applied to each
    # slice of a random tensor. Expects vectorized and sequential results to agree over 7
    # iterations.
    x = random_ops.random_uniform([7, 5, 2 * 2 - 3, 2 * 3 - 1, 3, 2])
    block_shapes = [2, 3]
    paddings = [[1, 2], [1, 0]]

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: Loop body for test_space_to_batch_nd; returns the converted i-th slice.
      x1 = array_ops.gather(x, i)
      return array_ops.space_to_batch_nd(x1, block_shapes, paddings)

    self._test_loop_fn(loop_fn, 7)

  def test_check_numerics(self):
    # [FUNCTION DOC]
    # Scenario: numeric-validity check op (with a message) applied to each slice of a random
    # finite tensor. Expects vectorized and sequential results to agree over 2 iterations.
    x = random_ops.random_uniform([2, 3, 4])

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: Loop body for test_check_numerics; returns the checked i-th slice.
      x_i = array_ops.gather(x, i)
      return array_ops.check_numerics(x_i, "test_message")

    self._test_loop_fn(loop_fn, 2)


if __name__ == "__main__":
  test.main()
