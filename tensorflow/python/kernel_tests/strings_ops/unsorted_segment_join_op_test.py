# [MODULE DOC]
# Tests for the string unsorted-segment-join op, which joins string elements that share a segment
# id using a separator. Covers numpy and tensor inputs, id/count dtypes, multi-dimensional ids,
# separators and error cases. Runs in graph and eager modes. No import-time side effects.

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
"""Tests for unsorted_segment_join_op."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test


class UnicodeTestCase(test.TestCase):
  # [CLASS DOC]
  # HELPER: shared base class for the segment-join tests.
  # Test case adding a unicode-normalizing equality assertion so byte strings and text strings
  # compare equal. Holds no state.
  """Test case with Python3-compatible string comparator."""

  def assertAllEqualUnicode(self, truth, actual):
    # [FUNCTION DOC]
    # HELPER: comparison helper for string results in the tests.
    # Converts both arguments (truth and actual, array-likes of strings) to unicode arrays and
    # asserts they are element-wise equal. Returns nothing; fails the test on mismatch.
    self.assertAllEqual(
        np.array(truth).astype('U'),
        np.array(actual).astype('U'))


@test_util.run_all_in_graph_and_eager_modes
class UnsortedSegmentJoinOpTest(UnicodeTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the unsorted segment join string op, run in both graph and eager modes. Uses the
  # unicode comparison helper and parameterized cases for dimensions and separators.

  def test_basic_np_array(self):
    # [FUNCTION DOC]
    # Joins rows of a nested list of strings by segment ids 1,0,1 into two segments with a colon
    # separator and asserts the shape and contents (segment 1 joins rows 0 and 2).
    inputs = [['Y', 'q', 'c'], ['Y', '6', '6'], ['p', 'G', 'a']]
    segment_ids = [1, 0, 1]
    num_segments = 2
    separator = ':'
    output_array = [['Y', '6', '6'], ['Y:p', 'q:G', 'c:a']]

    res = self.evaluate(
        string_ops.unsorted_segment_join(
            inputs=inputs,
            segment_ids=segment_ids,
            num_segments=num_segments,
            separator=separator))
    self.assertAllEqual(res.shape, np.array(output_array).shape)
    self.assertAllEqualUnicode(res, output_array)

  def test_segment_id_and_input_empty(self):
    # [FUNCTION DOC]
    # Empty input and empty segment ids with three segments: expects three empty strings.
    inputs = np.array([], dtype=np.bytes_)
    segment_ids = np.array([], dtype=np.int32)
    num_segments = 3
    separator = ':'
    output_array = ['', '', '']
    res = self.evaluate(
        string_ops.unsorted_segment_join(
            inputs=inputs,
            segment_ids=segment_ids,
            num_segments=num_segments,
            separator=separator))
    self.assertAllEqual(res.shape, np.array(output_array).shape)
    self.assertAllEqualUnicode(res, output_array)

  def test_type_check(self):
    # [FUNCTION DOC]
    # Verifies that 32-bit and 64-bit integer numpy arrays work for both segment ids and the
    # number of segments, producing the same joined output each time.
    inputs = [['Y', 'q', 'c'], ['Y', '6', '6'], ['p', 'G', 'a']]
    segment_ids = np.array([1, 0, 1], dtype=np.int32)
    num_segments = np.array(2, dtype=np.int32)
    separator = ':'
    output_array = [['Y', '6', '6'], ['Y:p', 'q:G', 'c:a']]

    res = self.evaluate(
        string_ops.unsorted_segment_join(
            inputs=inputs,
            segment_ids=segment_ids,
            num_segments=num_segments,
            separator=separator))
    self.assertAllEqual(res.shape, np.array(output_array).shape)
    self.assertAllEqualUnicode(res, output_array)

    segment_ids = np.array([1, 0, 1], dtype=np.int64)
    num_segments = np.array(2, dtype=np.int64)
    res = self.evaluate(
        string_ops.unsorted_segment_join(
            inputs=inputs,
            segment_ids=segment_ids,
            num_segments=num_segments,
            separator=separator))
    self.assertAllEqual(res.shape, np.array(output_array).shape)
    self.assertAllEqualUnicode(res, output_array)

  def test_basic_tensor(self):
    # [FUNCTION DOC]
    # Same join as the basic test but with tensors as inputs and segment ids; asserts the result
    # and its static shape equal the expected tensor.
    inputs = constant_op.constant([['Y', 'q', 'c'], ['Y', '6', '6'],
                                   ['p', 'G', 'a']])
    segment_ids = constant_op.constant([1, 0, 1])
    num_segments = 2
    separator = ':'
    output_array = constant_op.constant([['Y', '6', '6'], ['Y:p', 'q:G',
                                                           'c:a']])

    res = self.evaluate(
        string_ops.unsorted_segment_join(
            inputs=inputs,
            segment_ids=segment_ids,
            num_segments=num_segments,
            separator=separator))
    self.assertAllEqual(res, output_array)
    self.assertAllEqual(res.shape, output_array.get_shape())

  def test_multiple_segment_join(self):
    # [FUNCTION DOC]
    # Applies the join twice: first by rows with a colon, then joins the two resulting rows with an
    # empty separator; asserts contents and shapes after each step.
    inputs = [['Y', 'q', 'c'], ['Y', '6', '6'], ['p', 'G', 'a']]
    segment_ids_1 = [1, 0, 1]
    num_segments_1 = 2
    separator_1 = ':'
    output_array_1 = [['Y', '6', '6'], ['Y:p', 'q:G', 'c:a']]

    res = self.evaluate(
        string_ops.unsorted_segment_join(
            inputs=inputs,
            segment_ids=segment_ids_1,
            num_segments=num_segments_1,
            separator=separator_1))
    self.assertAllEqualUnicode(res, output_array_1)
    self.assertAllEqual(res.shape, np.array(output_array_1).shape)

    segment_ids_2 = [1, 1]
    num_segments_2 = 2
    separator_2 = ''
    output_array_2 = [['', '', ''], ['YY:p', '6q:G', '6c:a']]

    res = self.evaluate(
        string_ops.unsorted_segment_join(
            inputs=res,
            segment_ids=segment_ids_2,
            num_segments=num_segments_2,
            separator=separator_2))
    self.assertAllEqualUnicode(res, output_array_2)
    self.assertAllEqual(res.shape, np.array(output_array_2).shape)

  @parameterized.parameters([
      {
          'inputs': [[[['q'], ['s']], [['f'], ['F']], [['h'], ['0']]],
                     [[['E'], ['j']], [['2'], ['k']], [['N'], ['d']]],
                     [[['G'], ['M']], [['1'], ['S']], [['N'], ['7']]],
                     [[['8'], ['W']], [['W'], ['G']], [['j'], ['d']]]],
          'segment_ids': [1, 1, 0, 2],
          'num_segments':
              3,
          'separator':
              ':',
          'output_array': [[[['G'], ['M']], [['1'], ['S']], [['N'], ['7']]],
                           [[['q:E'], ['s:j']], [['f:2'], ['F:k']],
                            [['h:N'], ['0:d']]],
                           [[['8'], ['W']], [['W'], ['G']], [['j'], ['d']]]],
      },
      {
          'inputs': [[['Q', 'b'], ['c', 'p']], [['i', '9'], ['n', 'b']],
                     [['T', 'h'], ['g', 'z']]],
          'segment_ids': [[0, 1], [1, 0], [1, 0]],
          'num_segments': 2,
          'separator': ':',
          'output_array': [['Q:n:g', 'b:b:z'], ['c:i:T', 'p:9:h']]
      },
      {
          'inputs': [[['Q', 'b'], ['b', 'p']], [['i', '9'], ['n', 'b']],
                     [['T', 'h'], ['g', 'z']]],
          'segment_ids': [[[2, 1], [0, 0]], [[2, 0], [2, 2]], [[0, 2], [1, 0]]],
          'num_segments': 3,
          'separator': ':',
          'output_array': ['b:p:9:T:z', 'b:g', 'Q:i:n:b:h']
      },
      {
          'inputs': [[['z'], ['h']], [['c'], ['z']], [['V'], ['T']]],
          'segment_ids': [0, 1, 1],
          'num_segments': 3,
          'separator': ':',
          'output_array': [[['z'], ['h']], [['c:V'], ['z:T']], [[''], ['']]]
      },
  ])
  def test_multiple_cases_with_different_dims(self, inputs, segment_ids,
                                              num_segments, separator,
                                              output_array):
    # [FUNCTION DOC]
    # Parameterized test over inputs of varying rank and segment id shapes (including
    # multi-dimensional ids and empty segments). Runs the join and asserts values and shape match
    # the expected output_array.
    res = self.evaluate(
        string_ops.unsorted_segment_join(
            inputs=inputs,
            segment_ids=segment_ids,
            num_segments=num_segments,
            separator=separator))
    self.assertAllEqualUnicode(res, output_array)
    self.assertAllEqual(res.shape, np.array(output_array).shape)

  @parameterized.parameters([
      {
          'separator': '',
          'output_array': ['thisisatest']
      },
      {
          'separator': ':',
          'output_array': ['this:is:a:test']
      },
      {
          'separator': 'UNK',
          'output_array': ['thisUNKisUNKaUNKtest']
      },
  ])
  def testSeparator(self, separator, output_array):
    # [FUNCTION DOC]
    # Parameterized over separators (empty, colon, multi-character). Joins four words into one
    # segment and asserts the single joined string and shape.
    inputs = ['this', 'is', 'a', 'test']
    segment_ids = [0, 0, 0, 0]
    num_segments = 1
    res = self.evaluate(
        string_ops.unsorted_segment_join(
            inputs=inputs,
            segment_ids=segment_ids,
            num_segments=num_segments,
            separator=separator))
    self.assertAllEqual(res.shape, np.array(output_array).shape)
    self.assertAllEqualUnicode(res, output_array)

  def test_fail_segment_id_exceeds_segment_nums(self):
    # [FUNCTION DOC]
    # A segment id larger than allowed by the segment count must raise an invalid-argument error.
    inputs = [['Y', 'q', 'c'], ['Y', '6', '6'], ['p', 'G', 'a']]
    segment_ids = [1, 0, 1]
    num_segments = 1
    separator = ':'

    with self.assertRaises(errors_impl.InvalidArgumentError):
      self.evaluate(
          string_ops.unsorted_segment_join(
              inputs=inputs,
              segment_ids=segment_ids,
              num_segments=num_segments,
              separator=separator))

  def test_fail_segment_id_dim_does_not_match(self):
    # [FUNCTION DOC]
    # Segment ids longer than the first input dimension: graph mode raises a value error at
    # construction, eager mode raises an invalid-argument error.
    inputs = [['Y', 'q', 'c'], ['Y', '6', '6'], ['p', 'G', 'a']]
    segment_ids = [1, 0, 1, 1]
    num_segments = 2
    separator = ':'

    if not context.executing_eagerly():
      with self.assertRaises(ValueError):
        self.evaluate(
            string_ops.unsorted_segment_join(
                inputs=inputs,
                segment_ids=segment_ids,
                num_segments=num_segments,
                separator=separator))
    else:
      with self.assertRaises(errors_impl.InvalidArgumentError):
        self.evaluate(
            string_ops.unsorted_segment_join(
                inputs=inputs,
                segment_ids=segment_ids,
                num_segments=num_segments,
                separator=separator))

  def test_fail_segment_id_empty_input_non_empty(self):
    # [FUNCTION DOC]
    # Empty segment ids with non-empty inputs must raise a value error or invalid-argument error.
    inputs = [['Y', 'q', 'c'], ['Y', '6', '6'], ['p', 'G', 'a']]
    segment_ids = np.array([], dtype=np.int32)
    num_segments = 2
    separator = ':'
    with self.assertRaises((ValueError, errors_impl.InvalidArgumentError)):
      self.evaluate(
          string_ops.unsorted_segment_join(
              inputs=inputs,
              segment_ids=segment_ids,
              num_segments=num_segments,
              separator=separator))

  def test_empty_input(self):
    # [FUNCTION DOC]
    # Empty input with non-empty segment ids must raise a value error or invalid-argument error.
    inputs = np.array([], dtype=np.bytes_)
    segment_ids = [1, 0, 1]
    num_segments = 2
    separator = ':'
    with self.assertRaises((ValueError, errors_impl.InvalidArgumentError)):
      self.evaluate(
          string_ops.unsorted_segment_join(
              inputs=inputs,
              segment_ids=segment_ids,
              num_segments=num_segments,
              separator=separator))

  def test_fail_negative_segment_id(self):
    # [FUNCTION DOC]
    # Negative segment ids must raise an invalid-argument error.
    inputs = [['Y', 'q', 'c'], ['Y', '6', '6'], ['p', 'G', 'a']]
    segment_ids = [-1, 0, -1]
    num_segments = 1
    separator = ':'
    with self.assertRaises(errors_impl.InvalidArgumentError):
      self.evaluate(
          string_ops.unsorted_segment_join(
              inputs=inputs,
              segment_ids=segment_ids,
              num_segments=num_segments,
              separator=separator))


if __name__ == '__main__':
  test.main()
