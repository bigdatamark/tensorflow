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
# [MODULE DOC]
# Test module for converting ragged tensors to dense tensors (the to-tensor method). Provides two
# small module-level helpers (placeholder wrapping and rebuilding a ragged tensor from value row
# ids), a large test class with parameterized cases (default values, explicit shapes, cropping and
# padding, broadcasting defaults, errors), gradient tests, and a benchmark class. Runs in graph
# and eager modes; has a main entry point that runs the test framework.
"""Tests for ragged.to_tensor."""

import random

from absl.testing import parameterized
import numpy as np

from tensorflow.python.client import session
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import googletest
from tensorflow.python.util import nest


def make_placeholder(t):
  # [FUNCTION DOC]
  # HELPER: shared test utility used to defeat static shape inference in the tests of this file.
  # Wraps the given value (tensor or convertible value) in a placeholder-with-default of unknown
  # shape and returns that tensor. No other side effects.
  return array_ops.placeholder_with_default(t, None)


def rebuild_ragged_tensor_with_value_rowids(rt, feed_dict=None, sess=None):
  # [FUNCTION DOC]
  # HELPER: test utility that forces the value-row-ids code path of ragged to-tensor conversion.
  # Inputs: rt, a ragged tensor (or a plain tensor at the recursion base); feed_dict, optional
  # mapping that is filled with placeholder-to-value entries; sess, a session used to evaluate
  # tensors, required when feed_dict is given. Recursively rebuilds the ragged tensor from its
  # values, value row ids and row count; when feed_dict is given each row-id, row-count and leaf
  # tensor is replaced by a placeholder and its evaluated value stored in feed_dict (a mutation).
  # Returns the rebuilt ragged tensor, or the (possibly placeholder) tensor for non-ragged input.
  """Returns a copy of `rt`, built using `from_value_rowids`.

  This ensures that RaggedTensor._cached_value_rowids is populated, which
  triggers a different code-path for converting ragged tensors to tensors.

  If `feed_dict` and `sess` are specified, then build the new `RaggedTensor`
  using placeholder tensors, and populate a feed dictionary that can be used
  to feed the placeholders.

  Args:
    rt: The RaggedTensor to copy.
    feed_dict: If specified, then build the new `RaggedTensor` using
      placeholders, and populate this dict with entries to feed those
      placeholders.
    sess: A session used to evaluate tensors; required if feed_dict is
      specified.

  Returns:
    A copy of `rt`, built using `from_value_rowids`.
  """
  if isinstance(rt, ragged_tensor.RaggedTensor):
    values = rebuild_ragged_tensor_with_value_rowids(rt.values, feed_dict, sess)
    rowids = rt.value_rowids()
    nrows = rt.nrows()
    if feed_dict is not None:
      rowids_ph = make_placeholder(rowids)
      nrows_ph = make_placeholder(nrows)
      feed_dict[rowids_ph] = sess.run(rowids)
      feed_dict[nrows_ph] = sess.run(nrows)
      rowids, nrows = rowids_ph, nrows_ph
    return ragged_tensor.RaggedTensor.from_value_rowids(values, rowids, nrows)
  else:
    if feed_dict is not None:
      rt_ph = make_placeholder(rt)
      feed_dict[rt_ph] = sess.run(rt)
      rt = rt_ph
    return rt


@test_util.run_all_in_graph_and_eager_modes
class RaggedTensorToTensorOpTest(test_util.TensorFlowTestCase,
                                 parameterized.TestCase):
  # [CLASS DOC]
  # Main test case for ragged to-tensor conversion, run in both graph and eager modes. Covers
  # default values, explicit shapes (list, tuple, shape object, tensor), error messages,
  # broadcasting of defaults, gradients and large/random inputs. No shared fixtures.

  def testDocStringExamples(self):
    # [FUNCTION DOC]
    # Converts a ragged tensor with rows (9,8,7), empty, (6,5), (4) to dense and asserts the
    # result is zero padded to a four by three matrix.
    """Example from ragged_to_tensor.__doc__."""
    rt = ragged_factory_ops.constant([[9, 8, 7], [], [6, 5], [4]])
    dt = rt.to_tensor()
    self.assertAllEqual(dt, [[9, 8, 7], [0, 0, 0], [6, 5, 0], [4, 0, 0]])

  @parameterized.named_parameters(
      # Simple 2D ragged tensors (with one ragged dimension)
      {
          'testcase_name': 'shape_2xN',
          'rt_input': [[0, 1, 2], [], [3]],
          'expected': [[0, 1, 2], [0, 0, 0], [3, 0, 0]]
      },
      {
          'testcase_name': 'shape_2xN_default_0D',
          'rt_input': [[0, 1, 2], [], [3]],
          'default': 5,
          'expected': [[0, 1, 2], [5, 5, 5], [3, 5, 5]]
      },
      {
          'testcase_name': 'empty_first_row',
          'rt_input': [[], [], [3, 4], []],
          'expected': [[0, 0], [0, 0], [3, 4], [0, 0]]
      },
      {
          'testcase_name': 'empty_last_row',
          'rt_input': [[0, 1, 2], [], [3], []],
          'expected': [[0, 1, 2], [0, 0, 0], [3, 0, 0], [0, 0, 0]]
      },
      {
          'testcase_name': 'shape_4xN',
          'rt_input': [[1, 2, 3], [], [4], [5, 6]],
          'expected': [[1, 2, 3], [0, 0, 0], [4, 0, 0], [5, 6, 0]]
      },
      {
          'testcase_name': 'shape_4xN_default_0D',
          'rt_input': [[1, 2, 3], [], [4], [5, 6]],
          'default': 9,
          'expected': [[1, 2, 3], [9, 9, 9], [4, 9, 9], [5, 6, 9]]
      },
      {
          'testcase_name': 'shape_2xN_already_dense',
          'rt_input': [[6, 7, 8], [9, 10, 11]],
          'expected': [[6, 7, 8], [9, 10, 11]],
      },
      {
          'testcase_name': 'shape_2xN_string_already_dense',
          'rt_input': [[b'a', b'b', b'c'],
                       [b'd', b'e', b'antidisestablishmentarianism']],
          'ragged_rank': 1,
          'expected': [[b'a', b'b', b'c'],
                       [b'd', b'e', b'antidisestablishmentarianism']],
      },
      # 3D ragged tensors with two ragged dimensions
      {
          'testcase_name': 'shape_4xNxM',
          'rt_input': [[[1, 2], [], [3, 4]], [], [[5]], [[6, 7], [8]]],
          'expected': [
              [[1, 2], [0, 0], [3, 4]],  #
              [[0, 0], [0, 0], [0, 0]],  #
              [[5, 0], [0, 0], [0, 0]],  #
              [[6, 7], [8, 0], [0, 0]],  #
          ]
      },
      {
          'testcase_name': 'shape_4xNxM_default_0D',
          'rt_input': [[[1, 2], [], [3, 4]], [], [[5]], [[6, 7], [8]]],
          'default': 9,
          'expected': [
              [[1, 2], [9, 9], [3, 4]],  #
              [[9, 9], [9, 9], [9, 9]],  #
              [[5, 9], [9, 9], [9, 9]],  #
              [[6, 7], [8, 9], [9, 9]],  #
          ]
      },
      {
          'testcase_name': 'shape_1xNx1_default_0D',
          'rt_input': [[[1], [2], [3]]],
          'ragged_rank': 1,
          'default': 0,
          'expected': [[[1], [2], [3]]],
      },
      {
          'testcase_name': 'shape_2xNx2_already_dense',
          'rt_input': [[[6, 7], [8, 9], [10, 11]],
                       [[12, 13], [14, 15], [16, 17]]],
          'ragged_rank': 1,
          'expected': [[[6, 7], [8, 9], [10, 11]],
                       [[12, 13], [14, 15], [16, 17]]],
      },
      {
          'testcase_name': 'shape_2xNx2_already_dense_default_1D',
          'rt_input': [[[6, 7], [8, 9], [10, 11]],
                       [[12, 13], [14, 15], [16, 17]]],
          'ragged_rank': 1,
          'default': [31, 32],
          'expected': [[[6, 7], [8, 9], [10, 11]],
                       [[12, 13], [14, 15], [16, 17]]],
      },
      {
          'testcase_name': 'shape_2xNx2_string_already_dense',
          'rt_input': [[[b'a', b'b'], [b'c', b'd'], [b'e', b'f']],
                       [[b'g', b'jalapeno'], [b'kangaroo', b'llama'],
                        [b'manzana', b'nectar']]],
          'ragged_rank': 1,
          'expected': [[[b'a', b'b'], [b'c', b'd'], [b'e', b'f']],
                       [[b'g', b'jalapeno'], [b'kangaroo', b'llama'],
                        [b'manzana', b'nectar']]],
      },
      # 3D ragged tensors with one ragged dimension
      {
          'testcase_name': 'shape_4xNx1_default_1D',
          'rt_input': [[[1], [2], [3]], [], [[4]], [[5], [6]]],
          'ragged_rank': 1,
          'default': [9],
          'expected': [[[1], [2], [3]],
                       [[9], [9], [9]],
                       [[4], [9], [9]],
                       [[5], [6], [9]]]
      },
      {
          'testcase_name': 'shape_2xNx2_default_0D',
          'rt_input': [[[6, 7], [8, 9], [10, 11]],
                       [[12, 13], [14, 15]]],
          'ragged_rank': 1,
          'default': 2,
          'expected': [[[6, 7], [8, 9], [10, 11]],
                       [[12, 13], [14, 15], [2, 2]]],
      },
      {
          'testcase_name': 'shape_2xNx2_default_1D',
          'rt_input': [[[6, 7], [8, 9], [10, 11]],
                       [[12, 13], [14, 15]]],
          'ragged_rank': 1,
          'default': [2, 3],
          'expected': [[[6, 7], [8, 9], [10, 11]],
                       [[12, 13], [14, 15], [2, 3]]],
      },
      # 4D ragged tensors with 3 ragged dimensions
      {
          'testcase_name': 'shape_1xNxMxK_default_0D',
          'rt_input': [[[[1], [2]], [], [[3]]]],
          'default': 9,
          'expected': [[[[1], [2]], [[9], [9]], [[3], [9]]]],
      },
      # Broadcast default
      {
          'testcase_name': 'shape_2xNx2x2_default_2x1',
          'rt_input': [[[[1, 2], [3, 4]]], []],
          'ragged_rank': 1,
          'default': [[5], [6]],
          'expected': [[[[1, 2], [3, 4]]],
                       [[[5, 5], [6, 6]]]],
      },
      {
          'testcase_name': 'shape_2xNx2x2_default_1x2',
          'rt_input': [[[[1, 2], [3, 4]]], []],
          'ragged_rank': 1,
          'default': [[5, 6]],
          'expected': [[[[1, 2], [3, 4]]],
                       [[[5, 6], [5, 6]]]],
      },
      # Explicit shape
      {
          'testcase_name': 'shape_4xN_with_crop',
          'rt_input': [[0, 1, 2, 3], [], [4], []],
          'shape': [2, 3],
          'expected': [[0, 1, 2], [0, 0, 0]],
      },
      {
          'testcase_name': 'shape_2xN_with_pad',
          'rt_input': [[1, 2], [3]],
          'shape': [3, 3],
          'expected': [[1, 2, 0], [3, 0, 0], [0, 0, 0]],
      },
      {
          'testcase_name': 'shape_4xN_with_crop_and_pad',
          'rt_input': [[0, 1, 2, 3], [], [4], []],
          'shape': [2, 8],
          'expected': [[0, 1, 2, 3, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0]],
      },
      {
          'testcase_name': 'shape_4xN_with_tuple_shape',
          'rt_input': [[0, 1, 2, 3], [], [4], []],
          'shape': (2, 3),
          'expected': [[0, 1, 2], [0, 0, 0]],
      },
      {
          'testcase_name': 'shape_4xN_with_tensorshape_shape',
          'rt_input': [[0, 1, 2, 3], [], [4], []],
          'shape': tensor_shape.TensorShape([2, 3]),
          'expected': [[0, 1, 2], [0, 0, 0]],
      },
      {
          'testcase_name': 'shape_4xN_with_partial_shape',
          'rt_input': [[0, 1, 2, 3], [], [4], []],
          'shape': tensor_shape.TensorShape([2, None]),
          'expected': [[0, 1, 2, 3], [0, 0, 0, 0]],
      },
      # Empty tensors
      {
          'testcase_name': 'shape_0xN',
          'rt_input': [],
          'ragged_rank': 1,
          'expected': [],
          'expected_shape': [0, 0],
      },
      {
          'testcase_name': 'shape_0xNxM',
          'rt_input': [],
          'ragged_rank': 2,
          'expected': [],
          'expected_shape': [0, 0, 0],
      },
      # {
      #     'testcase_name': 'shape_0xNx2',
      #     'rt_input': [],
      #     'ragged_rank': 1,
      #     'inner_shape': [2],
      #     'expected': [],
      #     'expected_shape': [0, 0, 2],
      # },
      {
          'testcase_name': 'shape_2xN_empty',
          'rt_input': [[], []],
          'expected': [[], []],
          'expected_shape': [2, 0],
      },
  )  # pyformat: disable
  def testRaggedTensorToTensor(self,
                               rt_input,
                               expected,
                               ragged_rank=None,
                               inner_shape=None,
                               default=None,
                               shape=None,
                               expected_shape=None):
    # [FUNCTION DOC]
    # Parameterized test. Inputs: rt_input (nested list), expected (nested list result),
    # ragged_rank and inner_shape (optional construction settings), default (optional padding
    # value), shape (optional explicit output shape), expected_shape (optional shape used to
    # build the expected array for empty results). Builds the ragged tensor two ways (plain and
    # rebuilt from value row ids) and, for each, with and without placeholders hiding static
    # shapes. Converts to dense with the default and shape and asserts the result is a tensor with
    # the same dtype, a shape compatible with the requested or ragged shape, and expected values.
    # KNOWN ISSUE: when the placeholder variant is used, the default is replaced by a placeholder
    # and stays replaced for later loop iterations; harmless but it mutates the loop variable.
    rt1 = ragged_factory_ops.constant(
        rt_input, ragged_rank=ragged_rank, inner_shape=inner_shape)
    rt2 = rebuild_ragged_tensor_with_value_rowids(rt1)
    for rt in [rt1, rt2]:
      for use_placeholder in [False, True]:
        if use_placeholder:
          if default is not None:
            default = make_placeholder(default)
          rt = nest.map_structure(make_placeholder, rt, expand_composites=True)
        dt = rt.to_tensor(default_value=default, shape=shape)
        self.assertIsInstance(dt, tensor_lib.Tensor)
        self.assertEqual(rt.dtype, dt.dtype)
        if shape is not None:
          self.assertTrue(dt.shape.is_compatible_with(shape))
        else:
          self.assertTrue(dt.shape.is_compatible_with(rt.shape))
        if expected_shape is not None:
          expected = np.ndarray(expected_shape, buffer=np.array(expected))
        self.assertAllEqual(dt, expected)

  @parameterized.parameters([
      {
          'rt_input': [[1, 2, 3]],
          'default': 'a',
          'error_type': TypeError,
          'error': r'Expected int32|Cannot convert',
      },
      {
          'rt_input': [[1, 2, 3]],
          'default': [0],
          'error': r'default_value\.shape=.* and '
                   r'rt_input\.flat_values\.shape=.* are incompatible: '
                   r'default_value\.rank = 1  must be less than '
                   r'rt_input\.flat_values\.rank = 1'
      },
      {
          'rt_input': [[[1, 2], [3, 4]], [[5, 6]]],
          'ragged_rank': 1,
          'default': [7, 8, 9],
          'error': r'default_value\.shape.* and '
                   r'rt_input\.flat_values\.shape.* are incompatible: '
                   r'default_value\.shape\[-1\] = 3 but '
                   r'rt_input\.flat_values\.shape\[-1\] = 2'
      },
      {
          'rt_input': [[1, 2, 3]],
          'shape': [3, 3, 3],
          'error': r'rt_input\.shape and shape=\[.,.,.\] are incompatible: '
                   r'rt_input\.rank = 2 but shape\.rank = 3'
      },
      {
          'rt_input': [[[1, 2, 3]]],
          'ragged_rank': 1,
          'shape': [1, 1, 4],
          'error': r'rt_input\.shape and shape=\[1,1,4\] are incompatible: '
                   r'rt_input\.shape\[2\] = 3 but shape\[2\] = 4'
      },
  ])
  def testError(self,
                rt_input,
                error,
                error_type=(ValueError, errors.InvalidArgumentError),
                default=None,
                ragged_rank=None,
                shape=None):
    # [FUNCTION DOC]
    # Parameterized error test. Inputs: rt_input (nested list), error (regular expression for the
    # message), error_type (exception type or tuple, default value or invalid-argument error),
    # default (optional padding value), ragged_rank and shape (optional). Asserts that converting
    # to dense with the bad default or shape raises the expected error, both for a constant ragged
    # tensor and for one built from placeholders. Cases: wrong dtype default, default of too high
    # rank, mismatched inner dimension, shape of wrong rank, shape smaller-incompatible inner dim.

    rt = ragged_factory_ops.constant(rt_input, ragged_rank=ragged_rank)
    with self.assertRaisesRegex(error_type, error):
      self.evaluate(rt.to_tensor(default_value=default, shape=shape))
    rt_placeholder = nest.map_structure(
        make_placeholder, rt, expand_composites=True)
    with self.assertRaisesRegex(error_type, error):
      self.evaluate(
          rt_placeholder.to_tensor(default_value=default, shape=shape))

  def test_shape_limit_shape_is_tensor(self):
    # [FUNCTION DOC]
    # Passes the output shape as an int64 tensor (two by three) to crop a ragged tensor; asserts
    # the cropped values and a fully defined static shape of two by three.
    input_data = ragged_factory_ops.constant([[0, 1, 2, 3], [], [4], []])
    actual = input_data.to_tensor(
        shape=constant_op.constant([2, 3], dtype=dtypes.int64))
    self.assertAllEqual(actual, [[0, 1, 2], [0, 0, 0]])
    self.assertEqual(actual.shape.as_list(), [2, 3])

  def test_shape_limit_shape_is_tensor_unknown_rank(self):
    # [FUNCTION DOC]
    # Passes a scalar tensor of minus one as the shape (meaning unknown rank, no limit); asserts
    # the output is padded to the natural four by four shape and statically compatible with it.
    input_data = ragged_factory_ops.constant([[0, 1, 2, 3], [], [4], []])
    actual = input_data.to_tensor(
        shape=constant_op.constant(-1, dtype=dtypes.int64))
    self.assertAllEqual(
        actual, [[0, 1, 2, 3], [0, 0, 0, 0], [4, 0, 0, 0], [0, 0, 0, 0]])
    self.assertTrue(actual.shape.is_compatible_with([4, 4]))

  def test_shape_limit_shape_is_tensor_unknown_dim(self):
    # [FUNCTION DOC]
    # Passes a shape tensor of two and minus one (second dimension unconstrained); asserts two
    # rows padded to the longest row length of four and static compatibility with two by unknown.
    input_data = ragged_factory_ops.constant([[0, 1, 2, 3], [], [4], []])
    actual = input_data.to_tensor(
        shape=constant_op.constant([2, -1], dtype=dtypes.int64))
    self.assertAllEqual(actual, [[0, 1, 2, 3], [0, 0, 0, 0]])
    self.assertTrue(actual.shape.is_compatible_with([2, None]))

  def test_shape_limit_shape_is_tensor_int32(self):
    # [FUNCTION DOC]
    # Same as the int64 shape-tensor crop test but the shape tensor is 32-bit; asserts the cropped
    # values and static shape of two by three.
    input_data = ragged_factory_ops.constant([[0, 1, 2, 3], [], [4], []])
    actual = input_data.to_tensor(
        shape=constant_op.constant([2, 3], dtype=dtypes.int32))
    self.assertAllEqual(actual, [[0, 1, 2], [0, 0, 0]])
    self.assertEqual(actual.shape.as_list(), [2, 3])

  def test_shape_expand_first_dim(self):
    # [FUNCTION DOC]
    # Requests a four by four output for a three-row ragged tensor; asserts the extra row and
    # columns are zero padded and the static shape is four by four.
    input_data = ragged_factory_ops.constant([[0, 1, 2], [], [3]])
    actual = input_data.to_tensor(shape=[4, 4])
    self.assertAllEqual(
        actual, [[0, 1, 2, 0], [0, 0, 0, 0], [3, 0, 0, 0], [0, 0, 0, 0]])
    self.assertEqual(actual.shape.as_list(), [4, 4])

  def test_value_transposed(self):
    # [FUNCTION DOC]
    # Builds a ragged tensor from a transposed (non-contiguous) values tensor with explicit row
    # ids and row count and validation on, and asserts it compares equal to the expected nested
    # list, checking that transposed data is handled.
    # Check that transposed data is not an issue.
    my_value = array_ops.transpose(
        constant_op.constant([[0, 1, 2, 3], [4, 5, 6, 7]]))
    input_data = RaggedTensor.from_value_rowids(
        values=my_value,
        value_rowids=constant_op.constant([0, 1, 2, 3], dtype=dtypes.int64),
        nrows=constant_op.constant(4, dtype=dtypes.int64),
        validate=True)
    self.assertAllEqual(input_data, [[[0, 4]], [[1, 5]], [[2, 6]], [[3, 7]]])

  def test_broadcast_default(self):
    # [FUNCTION DOC]
    # Converts a ragged tensor with a two by two dense inner shape using a placeholder default of
    # shape two by one; asserts the default is broadcast across the inner dimension to fill the
    # empty row.
    # The dense dimension here is 2 x 2
    input_data = ragged_factory_ops.constant([[[[1, 2], [3, 4]]], []],
                                             ragged_rank=1)
    # This placeholder has a 2 x 1 dimension.
    default_value = make_placeholder([[5], [6]])
    actual = input_data.to_tensor(default_value=default_value)
    expected = [[[[1, 2], [3, 4]]], [[[5, 5], [6, 6]]]]
    self.assertAllEqual(actual, expected)

  def test_broadcast_default_no_placeholder(self):
    # [FUNCTION DOC]
    # Same scenario as the placeholder broadcast test but the two by one default is a constant
    # tensor with statically known shape; asserts the same broadcast result.
    input_data = ragged_factory_ops.constant([[[[1, 2], [3, 4]]], []],
                                             ragged_rank=1)
    # default_value has a 2 x 1 dimension.
    default_value = constant_op.constant([[5], [6]], shape=None)
    actual = input_data.to_tensor(default_value=default_value)
    expected = [[[[1, 2], [3, 4]]], [[[5, 5], [6, 6]]]]
    self.assertAllEqual(actual, expected)

  def test_shape_expand_second_dim(self):
    # [FUNCTION DOC]
    # Requests shape three by four for a four-row ragged tensor; asserts the result is cropped to
    # three rows and padded with zeros to four columns.
    input_data = ragged_factory_ops.constant([[0, 1, 2], [], [3], []])
    actual = input_data.to_tensor(shape=[3, 4])
    self.assertAllEqual(actual, [[0, 1, 2, 0], [0, 0, 0, 0], [3, 0, 0, 0]])

  @parameterized.parameters(
      ([2, 3, 4], None, [2, 3, 4]),
      ([2, 3, 4], [None, None, None], [2, 3, 4]),
      ([2, 3, 4], [None, 3, None], [2, 3, 4]),
      ([2, 3, 4], [None, 3, 4], [2, 3, 4]),
      ([2, 3, 4], [2, 3, 4], [2, 3, 4]),
      )
  def test_preserve_shape_roundtrip(
      self, input_shape, to_tensor_shape, expected_shape):
    # [FUNCTION DOC]
    # Parameterized test. Inputs: input_shape (shape of a dense zeros tensor), to_tensor_shape
    # (shape argument for conversion back, possibly null or with unknown entries), expected_shape.
    # Converts the dense tensor to a ragged tensor of ragged rank two and back to dense, and
    # asserts that the original, ragged and recovered static shapes all equal the expected shape.
    tensor = array_ops.zeros(input_shape)
    ragged_from_tensor = RaggedTensor.from_tensor(tensor, ragged_rank=2)
    recovered_tensor = ragged_from_tensor.to_tensor(shape=to_tensor_shape)
    self.assertAllEqual(tensor.shape.as_list(), expected_shape)
    self.assertAllEqual(ragged_from_tensor.shape.as_list(), expected_shape)
    self.assertAllEqual(recovered_tensor.shape.as_list(), expected_shape)

  def test_empty_tensor_with_shape(self):
    # [FUNCTION DOC]
    # Builds a ragged tensor with two rows and no values and converts it with default three and
    # shape two by three; asserts the result is entirely filled with the default.
    input_data = RaggedTensor.from_value_rowids(
        values=constant_op.constant([], dtype=dtypes.int64),
        value_rowids=constant_op.constant([], dtype=dtypes.int64),
        nrows=constant_op.constant(2, dtype=dtypes.int64),
        validate=True)
    actual = input_data.to_tensor(default_value=3, shape=[2, 3])
    self.assertAllEqual(actual, [[3, 3, 3], [3, 3, 3]])

  # pylint: disable=bad-whitespace
  @parameterized.named_parameters([
      dict(
          testcase_name = '2d_default_shape',
          shape         = None,
          rt_value      = [[1, 2, 3], [4], [5, 6]],
          rt_grad       = [[9, 8, 7], [6], [3, 2]],
          default_value = 0,
          default_grad  = sum([5, 4, 1]),
          output_value  = [[1, 2, 3], [4, 0, 0], [5, 6, 0]],
          output_grad   = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]),
      dict(
          testcase_name = '2d_pad',
          shape         = [4, 4],
          rt_value      = [[1, 2, 3], [4], [5, 6]],
          rt_grad       = [[9, 8, 7], [5], [1, 0]],
          default_value = 0,
          default_grad  = sum([6, 4, 3, 2, 1, 2, 3, 4, 5, 6]),
          output_value  = [
              [1, 2, 3, 0], [4, 0, 0, 0], [5, 6, 0, 0], [0, 0, 0, 0]],
          output_grad   = [
              [9, 8, 7, 6], [5, 4, 3, 2], [1, 0, 1, 2], [3, 4, 5, 6]]),
      dict(
          testcase_name = '2d_pad_and_crop',
          shape         = [5, 3],
          rt_value      = [[1, 2, 3], [4], [5, 6, 7, 8, 9], [8]],
          rt_grad       = [[9, 8, 7], [6], [3, 2, 1, 0, 0], [2]],
          default_value = 0,
          default_grad  = sum([5, 4, 3, 4, 5, 6, 7]),
          output_value  = [
              [1, 2, 3], [4, 0, 0], [5, 6, 7], [8, 0, 0], [0, 0, 0]],
          output_grad   = [
              [9, 8, 7], [6, 5, 4], [3, 2, 1], [2, 3, 4], [5, 6, 7]]),
      dict(
          testcase_name = '3d_rrank_2',
          shape         = [2, 2, 2],
          rt_value      = [[[9, 8, 7], [6]], [[5, 4]]],
          rt_grad       = [[[1, 2, 0], [3]], [[5, 6]]],
          default_value = 3,
          default_grad  = sum([4, 7, 8]),
          output_value  = [[[9, 8], [6, 3]], [[5, 4], [3, 3]]],
          output_grad   = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
      dict(
          testcase_name = '3d_rrank_1_with_0d_default',
          ragged_rank   = 1,
          shape         = [2, 2, 2],
          rt_value      = [[[9, 8], [7, 6]], [[5, 4]]],
          rt_grad       = [[[1, 2], [3, 4]], [[5, 6]]],
          default_value = 3,
          default_grad  = sum([7, 8]),
          output_value  = [[[9, 8], [7, 6]], [[5, 4], [3, 3]]],
          output_grad   = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
      dict(
          testcase_name = '3d_rrank_1_with_1d_default',
          ragged_rank   = 1,
          shape         = [2, 2, 2],
          rt_value      = [[[9, 8], [7, 6]], [[5, 4]]],
          rt_grad       = [[[1, 2], [3, 4]], [[5, 6]]],
          default_value = [3, 2],
          default_grad  = [7, 8],
          output_value  = [[[9, 8], [7, 6]], [[5, 4], [3, 2]]],
          output_grad   = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
      dict(
          testcase_name = '3d_rrank_1_with_1d_broadcast_default',
          ragged_rank   = 1,
          shape         = [2, 2, 2],
          rt_value      = [[[9, 8], [7, 6]], [[5, 4]]],
          rt_grad       = [[[1, 2], [3, 4]], [[5, 6]]],
          default_value = [3],
          default_grad  = [7 + 8],
          output_value  = [[[9, 8], [7, 6]], [[5, 4], [3, 3]]],
          output_grad   = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
      dict(
          testcase_name = '4d_rrank_1_with_2d_default',
          ragged_rank   = 1,
          shape         = [3, 3, 2, 1],
          rt_value      = [[[[9], [8]], [[7], [6]]], [[[5], [4]]]],
          rt_grad       = [[[[1], [2]], [[3], [4]]], [[[7], [8]]]],
          default_value = [[3], [2]],
          default_grad  = [[5 + 9 + 2 + 4 + 6 + 8], [6 + 1 + 3 + 5 + 7 + 9]],
          output_value  = [[[[9], [8]], [[7], [6]], [[3], [2]]],
                           [[[5], [4]], [[3], [2]], [[3], [2]]],
                           [[[3], [2]], [[3], [2]], [[3], [2]]]],
          output_grad   = [[[[1], [2]], [[3], [4]], [[5], [6]]],
                           [[[7], [8]], [[9], [1]], [[2], [3]]],
                           [[[4], [5]], [[6], [7]], [[8], [9]]]]),
      dict(
          testcase_name = '4d_rrank_1_with_with_0d_default',
          ragged_rank   = 1,
          shape         = [3, 3, 2, 1],
          rt_value      = [[[[9], [8]], [[7], [6]]], [[[5], [4]]]],
          rt_grad       = [[[[1], [2]], [[3], [4]]], [[[7], [8]]]],
          default_value = 3,
          default_grad  = 5 + 9 + 2 + 4 + 6 + 8 + 6 + 1 + 3 + 5 + 7 + 9,
          output_value  = [[[[9], [8]], [[7], [6]], [[3], [3]]],
                           [[[5], [4]], [[3], [3]], [[3], [3]]],
                           [[[3], [3]], [[3], [3]], [[3], [3]]]],
          output_grad   = [[[[1], [2]], [[3], [4]], [[5], [6]]],
                           [[[7], [8]], [[9], [1]], [[2], [3]]],
                           [[[4], [5]], [[6], [7]], [[8], [9]]]]),
      dict(
          testcase_name = 'zero_size',
          shape         = [0, 0],
          rt_value      = [[9, 8], [7, 6, 5], [4]],
          rt_grad       = [[0, 0], [0, 0, 0], [0]],
          default_value = 3,
          default_grad  = 0,
          output_value  = [],
          output_grad   = [])
  ])  # pyformat: disable
  def test_gradient(self,
                    shape,
                    rt_value,
                    rt_grad,
                    default_value,
                    default_grad,
                    output_value,
                    output_grad,
                    ragged_rank=None):
    # [FUNCTION DOC]
    # Parameterized gradient test. Inputs: shape (output shape), rt_value (ragged input values),
    # rt_grad (expected gradient for the ragged input), default_value (padding value),
    # default_grad (expected gradient for the default), output_value (expected dense output),
    # output_grad (upstream gradient fed in), ragged_rank (optional). Builds float32 constants,
    # then for both row-splits and value-row-ids partitioned versions runs the gradient helper:
    # directly in eager mode, or in graph mode for each level of static shape information
    # (known, unknown dimensions, unknown rank). Asserts output, input gradient and default
    # gradient match the expectations.
    # KNOWN ISSUE: in the graph-mode loop the ragged input is re-wrapped on every iteration, so
    # later shape-info variants wrap an already wrapped value instead of the original.
    """Tests that ragged_to_dense generates the right gradient.

    Args:
      shape: The `shape` arg for `ragged_to_dense`.
      rt_value: The `rt_input` arg for `ragged_to_dense`.
      rt_grad: The expected gradient for `rt_value`.  Corresponds 1:1 with
        `rt_value`.
      default_value: The `default_value` arg for `ragged_to_dense`.
      default_grad: The expected gradient for `default_value`.  Corresponds 1:1
        with `default_value`.
      output_value: The expected output of `ragged_to_dense`.
      output_grad: The gradient for the output (used to generate the gradients
        `rt_grad` and `default_grad`).  Corresponds 1:1 with `output_value`.
      ragged_rank: Ragged rank for `rt_value`.
    """
    rt_value = ragged_factory_ops.constant(
        rt_value, dtype=dtypes.float32, ragged_rank=ragged_rank)
    rt_grad = ragged_factory_ops.constant(
        rt_grad, dtype=dtypes.float32, ragged_rank=ragged_rank)
    default_value = constant_op.constant(default_value, dtype=dtypes.float32)
    default_grad = constant_op.constant(default_grad, dtype=dtypes.float32)
    output_value = constant_op.constant(
        output_value, dtype=dtypes.float32, shape=shape)
    output_grad = constant_op.constant(
        output_grad, dtype=dtypes.float32, shape=shape)
    shape = tensor_shape.as_shape(shape)

    # There are different code paths for ragged_to_dense, depending on whether
    # the RaggedTensor was created from row_splits or value_rowids.  Make sure
    # that we test both.
    for partition_type in ['row_splits', 'value_rowids']:
      rt_val = self.rt_with_partition_type(rt_value, partition_type)
      if context.executing_eagerly():
        self._test_gradient_helper(rt_val, default_value, shape, output_grad,
                                   output_value, rt_grad, default_grad)
      else:
        # There are different code paths when computing the gradient for
        # default_value, depending on whether shape info is statically
        # available; make sure that we test all code paths.
        for shape_info in ['known', 'unknown_dims', 'unknown_rank']:
          rt_val = self.wrap_in_placeholder(rt_val, shape_info)
          default_val = self.wrap_in_placeholder(default_value, shape_info)
          shape_val = self.wrap_in_placeholder(shape, shape_info)
          self._test_gradient_helper(rt_val, default_val, shape_val,
                                     output_grad, output_value, rt_grad,
                                     default_grad)

  def _test_gradient_helper(self, rt_val, default_val, shape_val, output_grad,
                            expected_output_val, expected_rt_grad,
                            expected_default_grad):
    # [FUNCTION DOC]
    # HELPER: shared by the gradient test. Inputs: rt_val (ragged tensor), default_val (padding
    # value), shape_val (output shape), output_grad (upstream gradient), and the expected output,
    # ragged-input gradient and default gradient. Converts to dense and computes gradients with a
    # gradient tape in eager mode, or with the graph gradient function otherwise. Asserts the
    # output is close to expected, the input gradient is a ragged tensor close to expected, and
    # the default gradient is close to expected. Returns nothing.
    if context.executing_eagerly():
      with backprop.GradientTape() as tape:
        tape.watch([rt_val, default_val])
        out = rt_val.to_tensor(default_val, shape=shape_val)
        actual_rt_grad, actual_default_grad = tape.gradient(
            out, (rt_val, default_val), output_gradients=output_grad)
    else:
      out = rt_val.to_tensor(default_val, shape=shape_val)
      actual_rt_grad, actual_default_grad = gradients_impl.gradients(
          ys=out, xs=(rt_val, default_val), grad_ys=output_grad)

    self.assertAllClose(out, expected_output_val)
    self.assertIsInstance(actual_rt_grad, RaggedTensor)
    self.assertAllClose(actual_rt_grad, expected_rt_grad)
    self.assertAllClose(actual_default_grad, expected_default_grad)

  def rt_with_partition_type(self, rt, partition_type):
    # [FUNCTION DOC]
    # HELPER: for the gradient test. Inputs: rt, a tensor or ragged tensor; partition_type, either
    # row splits or value row ids. Returns plain tensors and row-splits input unchanged; for
    # value row ids, recursively rebuilds the ragged tensor from values, row ids and row count.
    # Raises an assertion error for any other partition type.
    if isinstance(rt, tensor_lib.Tensor):
      return rt
    if partition_type == 'row_splits':
      return rt
    if partition_type == 'value_rowids':
      return ragged_tensor.RaggedTensor.from_value_rowids(
          self.rt_with_partition_type(rt.values, partition_type),
          rt.value_rowids(), rt.nrows())
    raise AssertionError('Unexpected partition_type %r' % partition_type)

  def wrap_in_placeholder(self, arg, shape_info):
    # [FUNCTION DOC]
    # HELPER: for the gradient test, limits static shape information. Inputs: arg (tensor, ragged
    # tensor or shape object) and shape_info (known, unknown dims, unknown rank). Known returns
    # arg unchanged; ragged input has only its flat values wrapped; a shape object is first
    # turned into a vector (a shape of unknown rank is returned as is). Otherwise returns a
    # placeholder-with-default with unknown rank or with all dimensions unknown. Raises an
    # assertion error for an unrecognized shape_info.
    """Wraps `arg` in a placeholder to limit static shape info.

    Args:
      arg: The value to wrap.  A Tensor, RaggedTensor, or TensorShape.
      shape_info: One of ['known', 'unknown_dims', 'unknown_rank'].

    Returns:
      * If shape_info is 'known': returns `arg`.
      * If shape_info is 'unknown_dims': returns a placeholder wrapping `arg`
        where the dimension sizes are unknown.  If `arg` is a TensorShape,
        then convert it to a vector first.  If `arg` is a RaggedTensor, then
        wrap the flat_values.
      * If shape_info is 'unknown_rank': returns a placeholder wrapping `arg`
        where the rank is unknown.  If `arg` is a TensorShape, then convert it
        to a vector first.  If `arg` is a RaggedTensor, then wrap the
        flat_values.
    """
    if shape_info == 'known':
      return arg
    if isinstance(arg, ragged_tensor.RaggedTensor):
      return arg.with_flat_values(
          self.wrap_in_placeholder(arg.flat_values, shape_info))
    if isinstance(arg, tensor_shape.TensorShape):
      if arg.ndims is None:
        return arg
      arg = constant_op.constant(arg.as_list())
    if shape_info == 'unknown_rank':
      return array_ops.placeholder_with_default(arg, None)
    if shape_info == 'unknown_dims':
      return array_ops.placeholder_with_default(arg, [None] * arg.shape.rank)
    raise AssertionError('Unexpected shape_info %r' % shape_info)

  def test_shape_is_list_including_tensor_element(self):
    # [FUNCTION DOC]
    # Passes a shape list mixing a Python integer and a scalar tensor (two by two); asserts the
    # ragged tensor is cropped and padded to the matching two by two result.
    rt = ragged_factory_ops.constant([[1, 2, 3], [4], [5, 6]])
    result = rt.to_tensor(shape=[2, constant_op.constant(2)])
    self.assertAllEqual(result, [[1, 2], [4, 0]])


class RaggedToDenseBenchmark(googletest.Benchmark):
  # [CLASS DOC]
  # Benchmark for ragged to-tensor conversion over a table of configurations (shape, ragged rank,
  # dtype, fill ratio, default shape, iterations). Each configuration is benchmarked for inputs
  # partitioned by row splits and by value row ids, reporting memory usage and throughput.

  # Configurations to test.  See `run_benchmark` for config param docs.
  CONFIGS = [
      {'shape': [10, 10]},
      {'shape': [10, 1000]},
      {'shape': [1000, 10]},
      {'shape': [1000, 10], 'fill': [1, 0.95]},  # Mostly full.
      {'shape': [1000, 10], 'fill': [1, 0.05]},  # Mostly empty.
      {'shape': [1000, 10], 'dtype': dtypes.string},
      {'shape': [1000, 10], 'dtype': dtypes.int64},
      {'shape': [100, 100]},
      {'shape': [50, 50, 32]},
      {'shape': [100, 100, 100], 'min_iters': 100},
      {'shape': [1000, 1000], 'min_iters': 100},
      {'shape': [10, 10, 10, 10, 10]},
      {'shape': [10, 10, 10, 10, 10], 'ragged_rank': 1},
      {'shape': [10, 10, 10, 10, 10], 'ragged_rank': 2},
      {'shape': [50, 50, 32], 'ragged_rank': 1, 'default_shape': [32]},
      {'shape': [200, 50, 32], 'ragged_rank': 1, 'default_shape': [32]}
  ]  # pyformat: disable

  def run_benchmark(self,
                    shape=(100, 100),
                    ragged_rank=None,
                    dtype=dtypes.float32,
                    fill=None,
                    default_shape=(),
                    output_shape=None,
                    min_iters=1000):
    # [FUNCTION DOC]
    # Runs one benchmark configuration. Inputs: shape (bounding box), ragged_rank (default shape
    # length minus one), dtype (default 32-bit float), fill (per-dimension fill ratio, default
    # 0.8), default_shape (shape of padding value, default scalar), output_shape (accepted but
    # not used), min_iters (minimum iterations, default 1000). Generates random nested data,
    # builds placeholders for the row-splits and value-row-ids forms inside a session, and runs
    # the op benchmark for each with memory tracking. Returns nothing; reports through the
    # benchmark framework.
    # KNOWN ISSUE: the documented output_shape parameter is never used, so padding or cropping
    # to an output shape is not actually benchmarked.
    """Run a benchmark with the specified configuration parameters.

    Args:
      shape: Bounding box for the input ragged tensor.
      ragged_rank: Ragged rank for the input ragged tensor.  Defaults to
        `len(shape)-1`.
      dtype: Data type for the input ragged tensor.
      fill: How full each dimension should be (0-1).  Corresponds 1:1 with
        `shape`.  Defaults to 0.8 for each dimension.
      default_shape: Shape for the default (padding) value.
      output_shape: Output shape -- ragged tensor will be padded or cropped to
        this shape.
      min_iters: Minimum iterations for benchmark.
    """
    if ragged_rank is None:
      ragged_rank = len(shape) - 1
    if fill is None:
      fill = [0.8 for _ in shape]

    # Build the inputs for the op.
    rt_input = self._generateRaggedTensor(shape, ragged_rank, dtype, fill)
    default_value = constant_op.constant(
        self._generateRaggedTensor(default_shape, 0, dtype), dtype=dtype)

    mbs = np.prod(shape) / (2**20)
    with session.Session(config=benchmark.benchmark_config()) as sess:
      extras = {
          'shape': shape,
          'ragged_rank': ragged_rank,
          'dtype': dtype,
          'fill': fill,
          'default_shape': default_shape
      }
      rt = ragged_factory_ops.constant(rt_input, dtype, ragged_rank=ragged_rank)

      # Inputs for with_splits:
      splits_rt_placeholder = ragged_factory_ops.placeholder(
          dtype, ragged_rank, shape[ragged_rank + 1:])
      splits_feed_dict = {splits_rt_placeholder: sess.run(rt)}

      # Inputs for with_rowids:
      rowids_feed_dict = {}
      rowids_rt_placeholder = rebuild_ragged_tensor_with_value_rowids(
          rt, rowids_feed_dict, sess)

      # Common arguments for benchmarks:
      run_op_benchmark_kwargs = dict(
          sess=sess,
          store_memory_usage=True,
          min_iters=min_iters,
          burn_iters=max(5, min_iters // 10),
          mbs=mbs,
          extras=extras)

      ragged_to_tensor_with_splits = splits_rt_placeholder.to_tensor(
          default_value=default_value)
      self.run_op_benchmark(
          op_or_tensor=ragged_to_tensor_with_splits.op,
          name='ragged_to_tensor_with_splits',
          feed_dict=splits_feed_dict,
          **run_op_benchmark_kwargs)

      ragged_to_tensor_with_rowids = rowids_rt_placeholder.to_tensor(
          default_value=default_value)
      self.run_op_benchmark(
          op_or_tensor=ragged_to_tensor_with_rowids.op,
          name='ragged_to_tensor_with_rowids',
          feed_dict=rowids_feed_dict,
          **run_op_benchmark_kwargs)

  def _generateRaggedTensor(self, shape, ragged_rank, dtype, fill=None, axis=0):
    # [FUNCTION DOC]
    # HELPER: builds random benchmark input. Recursively generates a nested list for the given
    # shape. At the leaf level (axis equals shape length) returns a random number, converted to
    # a string for string dtype or scaled to an integer for integer dtypes. For the outermost
    # axis and axes beyond the ragged rank the slice size is the full dimension; for ragged axes
    # it is drawn from a geometric distribution with the fill value for that axis. Uses global
    # random state. Returns the nested list (or a single leaf value).
    if axis == len(shape):
      value = random.random()
      if dtype == dtypes.string:
        value = str(value)
      if dtype.is_integer:
        value = int(value * 1000)
      return value
    if axis == 0 or axis > ragged_rank:
      slice_size = shape[axis]
    else:
      slice_size = (np.random.geometric(fill[axis], shape[axis]) == 1).sum()
    return [
        self._generateRaggedTensor(shape, ragged_rank, dtype, fill, axis + 1)
        for _ in range(slice_size)
    ]

  def benchmark_ragged_to_dense(self):
    # [FUNCTION DOC]
    # Benchmark entry point. Seeds the Python random generator with a fixed value for
    # reproducibility, then runs the benchmark for every configuration in the class table.
    random.seed(5)
    for config in self.CONFIGS:
      self.run_benchmark(**config)


if __name__ == '__main__':
  googletest.main()
