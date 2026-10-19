# [MODULE DOC]
# Test module for the TPU sharding utility ops that split a tensor into a grid of pieces and
# concatenate pieces back (with optional padding), in both tensor-input and resource-variable
# forms, under the XLA test harness. Module-level builder functions create the graphs for the
# tensor and resource variants; three parameterized test classes cover split validation and
# behavior, concat validation and behavior, and split-then-concat round trips over ranks one
# to eight. No import-time side effects beyond imports; as a script it starts the test runner.
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for sharding util ops (XlaSplitND, XlaConcatND)."""

from typing import Any, List, Optional

from absl.testing import parameterized
import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.client.session import Session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework.ops import control_dependencies
from tensorflow.python.framework.tensor import Tensor
from tensorflow.python.ops import gen_tpu_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


def create_tensor_split_graph(
    sess: Session,
    input_value: Any,
    input_dtype: Any,
    num_outputs: int,
    num_splits: List[int],
    paddings: Optional[List[int]] = None) -> List[Tensor]:
  # [FUNCTION DOC]
  # HELPER: Graph builder for the split tests, tensor-input variant (selected through the
  # parameterized tests).
  # Inputs: sess is unused (deleted); input_value is nested list or array data; input_dtype the
  # element type; num_outputs the expected number of pieces; num_splits list of split counts per
  # dimension; paddings optional list of per-dimension padding amounts (default none).
  # Builds a constant tensor and applies the split op. Returns the list of piece tensors.
  del sess

  const_input_op = constant_op.constant(input_value, dtype=input_dtype)
  return gen_tpu_ops.xla_split_nd(
      const_input_op, num_outputs, num_splits, paddings=paddings)


def create_resource_split_graph(
    sess: Session,
    input_value: Any,
    input_dtype: Any,
    num_outputs: int,
    num_splits: List[int],
    paddings: Optional[List[int]] = None) -> List[Tensor]:
  # [FUNCTION DOC]
  # HELPER: Graph builder for the split tests, resource-variable variant.
  # Inputs: sess is the session used to initialize the variable; input_value data; input_dtype
  # element type; num_outputs expected number of pieces; num_splits list of split counts per
  # dimension; paddings optional per-dimension padding amounts (default none).
  # Creates a resource variable with the data, initializes it in the session (side effect), and
  # applies the split op that reads directly from the variable handle. Returns the list of
  # piece tensors.
  variable = resource_variable_ops.ResourceVariable(
      initial_value=input_value, dtype=input_dtype)
  sess.run(variables.variables_initializer([variable]))
  return gen_tpu_ops.read_variable_xla_split_nd(
      variable.handle, input_dtype, num_outputs, num_splits, paddings=paddings)


class XlaSplitNDOpTest(xla_test.XLATestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized XLA test case for the N-dimensional split op, run against both the tensor and
  # resource-variable graph builders. Covers argument validation errors and split behavior with
  # and without padding. Holds no state of its own.

  @parameterized.named_parameters(('Tensor', create_tensor_split_graph),
                                  ('Resource', create_resource_split_graph))
  def testSplitDimensionZero(self, graph_fn):
    # [FUNCTION DOC]
    # Parameter graph_fn is a split graph builder. For each numeric type, requests a split count
    # of zero in the last dimension of a one-by-one-by-one input and expects an op error saying
    # the entry at index two must be positive but is zero. Returns nothing.
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        split = graph_fn(
            sess,
            input_value=[[[0]]],
            input_dtype=dtype,
            num_outputs=1,
            num_splits=[1, 1, 0])
        with self.assertRaisesOpError('index 2 must be positive, but got 0'):
          sess.run(split)

  @parameterized.named_parameters(('Tensor', create_tensor_split_graph),
                                  ('Resource', create_resource_split_graph))
  def testSplitDimensionNegative(self, graph_fn):
    # [FUNCTION DOC]
    # Parameter graph_fn is a split graph builder. For each numeric type, requests a negative
    # split count in the middle dimension and expects an op error saying the entry at index one
    # must be positive but is minus one. Returns nothing.
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        split = graph_fn(
            sess,
            input_value=[[[0]]],
            input_dtype=dtype,
            num_outputs=1,
            num_splits=[1, -1, 1])
        with self.assertRaisesOpError('index 1 must be positive, but got -1'):
          sess.run(split)

  @parameterized.named_parameters(('Tensor', create_tensor_split_graph),
                                  ('Resource', create_resource_split_graph))
  def testNumOutputsMismatch(self, graph_fn):
    # [FUNCTION DOC]
    # Parameter graph_fn is a split graph builder. For each numeric type, splits a two-element
    # vector in two but declares only one output, expecting an op error that the output count
    # must match the number of slices (two). Returns nothing.
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        split = graph_fn(
            sess,
            input_value=[0, 1],
            input_dtype=dtype,
            num_outputs=1,
            num_splits=[2])
        with self.assertRaisesOpError('\'N\' must match number of slices 2'):
          sess.run(split)

  @parameterized.named_parameters(('Tensor', create_tensor_split_graph),
                                  ('Resource', create_resource_split_graph))
  def testPaddingsLengthMismatch(self, graph_fn):
    # [FUNCTION DOC]
    # Parameter graph_fn is a split graph builder. For each numeric type, splits a two-by-two
    # input with a padding list of length one instead of two, expecting an op error that the
    # padding length should be two but is one. Returns nothing.
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        split = graph_fn(
            sess,
            input_value=[[0, 1], [2, 3]],
            input_dtype=dtype,
            num_outputs=4,
            num_splits=[2, 2],
            paddings=[0])
        with self.assertRaisesOpError('length 2, but got 1'):
          sess.run(split)

  @parameterized.named_parameters(('Tensor', create_tensor_split_graph),
                                  ('Resource', create_resource_split_graph))
  def testPaddingsNegative(self, graph_fn):
    # [FUNCTION DOC]
    # Parameter graph_fn is a split graph builder. For each numeric type, splits a two-by-two
    # input with a negative padding at index one, expecting an op error that paddings must be
    # non-negative. Returns nothing.
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        split = graph_fn(
            sess,
            input_value=[[0, 1], [2, 3]],
            input_dtype=dtype,
            num_outputs=4,
            num_splits=[2, 2],
            paddings=[0, -1])
        with self.assertRaisesOpError('non-negative, but got -1 at index 1'):
          sess.run(split)

  @parameterized.named_parameters(('Tensor', create_tensor_split_graph),
                                  ('Resource', create_resource_split_graph))
  def testInputRankSplitMismatch(self, graph_fn):
    # [FUNCTION DOC]
    # Parameter graph_fn is a split graph builder. For each numeric type, gives a rank-two input
    # but a split list of length three, expecting an op error that the split list length does not
    # match the input rank. Returns nothing.
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        split = graph_fn(
            sess,
            input_value=[[0, 1], [2, 3]],
            input_dtype=dtype,
            num_outputs=8,
            num_splits=[2, 2, 2])
        with self.assertRaisesOpError(
            '\'num_splits\' length 3, but got rank 2'):
          sess.run(split)

  @parameterized.named_parameters(('Tensor', create_tensor_split_graph),
                                  ('Resource', create_resource_split_graph))
  def testDimNotEvenlySplit(self, graph_fn):
    # [FUNCTION DOC]
    # Parameter graph_fn is a split graph builder. For each numeric type, splits a four-row input
    # into three parts along the first dimension, expecting an op error that the dimension size
    # is not divisible by the split count. Returns nothing.
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        split = graph_fn(
            sess,
            input_value=[[0, 1], [2, 3], [4, 5], [6, 7]],
            input_dtype=dtype,
            num_outputs=6,
            num_splits=[3, 2])
        with self.assertRaisesOpError('divisible by \'num_splits\' 3'):
          sess.run(split)

  @parameterized.named_parameters(('Tensor', create_tensor_split_graph),
                                  ('Resource', create_resource_split_graph))
  def testDimWithPaddingNotEvenlySplit(self, graph_fn):
    # [FUNCTION DOC]
    # Parameter graph_fn is a split graph builder. For each numeric type, splits a four-row input
    # into two parts along the first dimension with padding of zero and one; the padded size of
    # five is not divisible by two, expecting an op error about divisibility. Returns nothing.
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        split = graph_fn(
            sess,
            input_value=[[0, 1], [2, 3], [4, 5], [6, 7]],
            input_dtype=dtype,
            num_outputs=4,
            num_splits=[2, 2],
            paddings=[0, 1])
        with self.assertRaisesOpError('divisible by \'num_splits\' 2'):
          sess.run(split)

  @parameterized.named_parameters(('Tensor', create_tensor_split_graph),
                                  ('Resource', create_resource_split_graph))
  def testNoSplits(self, graph_fn):
    # [FUNCTION DOC]
    # Parameter graph_fn is a split graph builder. For each numeric type, uses split counts of
    # one in every dimension on a two-by-two-by-two input and expects exactly one output equal to
    # the input. Returns nothing.
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        split = graph_fn(
            sess,
            input_value=[[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
            input_dtype=dtype,
            num_outputs=1,
            num_splits=[1, 1, 1])
        results = sess.run(split)
      self.assertLen(results, 1)
      self.assertAllClose(results[0], [[[0, 1], [2, 3]], [[4, 5], [6, 7]]])

  @parameterized.named_parameters(('Tensor', create_tensor_split_graph),
                                  ('Resource', create_resource_split_graph))
  def testNoSplitsWithPadding(self, graph_fn):
    # [FUNCTION DOC]
    # Parameter graph_fn is a split graph builder. For each numeric type, uses split counts of
    # one on a two-by-one-by-one input with paddings of zero, one, one and expects one output
    # of shape two by two by two in which the original values are kept and the rest is zero.
    # Returns nothing.
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        split = graph_fn(
            sess,
            input_value=[[[0]], [[1]]],
            input_dtype=dtype,
            num_outputs=1,
            num_splits=[1, 1, 1],
            paddings=[0, 1, 1])
        results = sess.run(split)
      self.assertLen(results, 1)
      self.assertAllClose(results[0], [[[0, 0], [0, 0]], [[1, 0], [0, 0]]])

  @parameterized.named_parameters(('Tensor', create_tensor_split_graph),
                                  ('Resource', create_resource_split_graph))
  def testSplitNoPadding(self, graph_fn):
    # [FUNCTION DOC]
    # Parameter graph_fn is a split graph builder. For each numeric type, splits a four-by-four
    # matrix into a two-by-two grid and expects four two-by-two quadrants in row-major order
    # (top-left, top-right, bottom-left, bottom-right). Returns nothing.
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        split = graph_fn(
            sess,
            input_value=[
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
                [12, 13, 14, 15],
            ],
            input_dtype=dtype,
            num_outputs=4,
            num_splits=[2, 2])
        results = sess.run(split)
      self.assertLen(results, 4)
      self.assertAllClose(results[0], [[0, 1], [4, 5]])
      self.assertAllClose(results[1], [[2, 3], [6, 7]])
      self.assertAllClose(results[2], [[8, 9], [12, 13]])
      self.assertAllClose(results[3], [[10, 11], [14, 15]])

  @parameterized.named_parameters(('Tensor', create_tensor_split_graph),
                                  ('Resource', create_resource_split_graph))
  def testSplitPartialPadding(self, graph_fn):
    # [FUNCTION DOC]
    # Parameter graph_fn is a split graph builder. For each numeric type, splits a three-by-three
    # matrix into a two-by-two grid with padding of one in each dimension, so it is treated as
    # four-by-four with zero fill, and expects four quadrants where the padded cells are zero.
    # Returns nothing.
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        split = graph_fn(
            sess,
            input_value=[
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
            ],
            input_dtype=dtype,
            num_outputs=4,
            num_splits=[2, 2],
            paddings=[1, 1])
        results = sess.run(split)
      self.assertLen(results, 4)
      self.assertAllClose(results[0], [[0, 1], [3, 4]])
      self.assertAllClose(results[1], [[2, 0], [5, 0]])
      self.assertAllClose(results[2], [[6, 7], [0, 0]])
      self.assertAllClose(results[3], [[8, 0], [0, 0]])

  @parameterized.named_parameters(('Tensor', create_tensor_split_graph),
                                  ('Resource', create_resource_split_graph))
  def testSplitCompletePadding(self, graph_fn):
    # [FUNCTION DOC]
    # Parameter graph_fn is a split graph builder. For each numeric type, splits a two-by-one
    # matrix into a two-by-two grid with paddings of two and three, so most pieces are entirely
    # padding. Expects four two-by-two pieces: the first holds the data in its first column and
    # zeros elsewhere, the other three are all zero. Returns nothing.
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        split = graph_fn(
            sess,
            input_value=[[0], [1]],
            input_dtype=dtype,
            num_outputs=4,
            num_splits=[2, 2],
            paddings=[2, 3])
        results = sess.run(split)
      self.assertLen(results, 4)
      self.assertAllClose(results[0], [[0, 0], [1, 0]])
      self.assertAllClose(results[1], [[0, 0], [0, 0]])
      self.assertAllClose(results[2], [[0, 0], [0, 0]])
      self.assertAllClose(results[3], [[0, 0], [0, 0]])

  @parameterized.named_parameters(
      ('1Tensor', create_tensor_split_graph, 1),
      ('2Tensor', create_tensor_split_graph, 2),
      ('3Tensor', create_tensor_split_graph, 3),
      ('4Tensor', create_tensor_split_graph, 4),
      ('5Tensor', create_tensor_split_graph, 5),
      ('6Tensor', create_tensor_split_graph, 6),
      ('7Tensor', create_tensor_split_graph, 7),
      ('8Tensor', create_tensor_split_graph, 8),
      ('1Resource', create_resource_split_graph, 1),
      ('2Resource', create_resource_split_graph, 2),
      ('3Resource', create_resource_split_graph, 3),
      ('4Resource', create_resource_split_graph, 4),
      ('5Resource', create_resource_split_graph, 5),
      ('6Resource', create_resource_split_graph, 6),
      ('7Resource', create_resource_split_graph, 7),
      ('8Resource', create_resource_split_graph, 8),
  )
  def testRanked(self, graph_fn, rank):
    # [FUNCTION DOC]
    # Parameterized by a split graph builder and a rank from one to eight (each for the tensor
    # and resource forms). For each numeric type, splits a rank-dimensional input of shape two in
    # every dimension into two pieces per dimension, giving two to the power of rank pieces of
    # one element each. Expects that count of outputs and that piece number i holds the value i.
    # Returns nothing.
    num_splits = [2] * rank
    num_outputs = 2 << (rank - 1)
    input_value = np.reshape(np.arange(np.prod(num_splits)), num_splits)
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        split = graph_fn(
            sess,
            input_value=input_value,
            input_dtype=dtype,
            num_outputs=num_outputs,
            num_splits=num_splits)
        results = sess.run(split)
      self.assertLen(results, num_outputs)
      for i, result in enumerate(results):
        expected_output = np.reshape(i, [1] * rank).astype(dtype)
        self.assertAllClose(result, expected_output)


def create_tensor_concat_graph(
    sess: Session,
    input_values: List[Any],
    input_dtype: Any,
    num_concats: List[int],
    paddings: Optional[List[int]] = None,
    output_shape: Optional[List[int]] = None) -> Tensor:
  # [FUNCTION DOC]
  # HELPER: Graph builder for the concat tests, tensor-input variant.
  # Inputs: sess and output_shape are unused (deleted); input_values is a list of piece data;
  # input_dtype element type; num_concats list of piece counts per dimension; paddings optional
  # per-dimension amounts to trim from the result (default none). Builds a constant tensor per
  # piece and applies the concat op. Returns the concatenated tensor.
  del sess
  del output_shape

  const_input_ops = [
      constant_op.constant(i, dtype=input_dtype) for i in input_values
  ]
  return gen_tpu_ops.xla_concat_nd(const_input_ops, num_concats, paddings)


def create_resource_concat_graph(
    sess: Session,
    input_values: List[Any],
    input_dtype: Any,
    num_concats: List[int],
    paddings: Optional[List[int]] = None,
    output_shape: Optional[List[int]] = None) -> Tensor:
  # [FUNCTION DOC]
  # HELPER: Graph builder for the concat tests, resource-variable variant.
  # Inputs: sess session used to initialize the variable; input_values list of piece data;
  # input_dtype element type; num_concats piece counts per dimension; paddings optional trim
  # amounts (default none); output_shape optional shape of the destination variable (default a
  # scalar shape). Creates a zero-filled variable of that shape, initializes it in the session
  # (side effect), assigns the concatenation into it, and returns a tensor that reads the
  # variable after the assignment (through a control dependency).
  variable_shape = [] if output_shape is None else output_shape
  variable = resource_variable_ops.ResourceVariable(
      initial_value=np.zeros(variable_shape, dtype=input_dtype),
      dtype=input_dtype)
  sess.run(variables.variables_initializer([variable]))
  const_input_ops = [
      constant_op.constant(i, dtype=input_dtype) for i in input_values
  ]
  concat = gen_tpu_ops.assign_variable_xla_concat_nd(variable.handle,
                                                     const_input_ops,
                                                     num_concats, paddings)
  with control_dependencies([concat]):
    return variable.read_value()


class XlaConcatNDOpTest(xla_test.XLATestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized XLA test case for the N-dimensional concat op, run against both the tensor and
  # resource-variable graph builders. Covers argument validation errors and concatenation with
  # and without padding. Holds no state of its own.

  @parameterized.named_parameters(('Tensor', create_tensor_concat_graph),
                                  ('Resource', create_resource_concat_graph))
  def testConcatDimensionZero(self, graph_fn):
    # [FUNCTION DOC]
    # Parameter graph_fn is a concat graph builder. For each numeric type, requests a concat
    # count of zero in the last dimension and expects an op error saying the entry at index two
    # must be positive but is zero. Returns nothing.
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        concat = graph_fn(
            sess,
            input_values=[[[[0]]]],
            input_dtype=dtype,
            num_concats=[1, 1, 0])
        with self.assertRaisesOpError('index 2 must be positive, but got 0'):
          sess.run(concat)

  @parameterized.named_parameters(('Tensor', create_tensor_concat_graph),
                                  ('Resource', create_resource_concat_graph))
  def testConcatDimensionNegative(self, graph_fn):
    # [FUNCTION DOC]
    # Parameter graph_fn is a concat graph builder. For each numeric type, requests a negative
    # concat count in the middle dimension and expects an op error saying the entry at index one
    # must be positive but is minus one. Returns nothing.
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        concat = graph_fn(
            sess,
            input_values=[[[[0]]]],
            input_dtype=dtype,
            num_concats=[1, -1, 1])
        with self.assertRaisesOpError('index 1 must be positive, but got -1'):
          sess.run(concat)

  @parameterized.named_parameters(('Tensor', create_tensor_concat_graph),
                                  ('Resource', create_resource_concat_graph))
  def testNumInputsMismatch(self, graph_fn):
    # [FUNCTION DOC]
    # Parameter graph_fn is a concat graph builder. For each numeric type, supplies one input
    # piece but asks for two concatenations, expecting an op error that the input count must
    # match the number of slices (two). Returns nothing.
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        concat = graph_fn(
            sess, input_values=[[0, 1]], input_dtype=dtype, num_concats=[2])
        with self.assertRaisesOpError('\'N\' must match number of slices 2'):
          sess.run(concat)

  @parameterized.named_parameters(('Tensor', create_tensor_concat_graph),
                                  ('Resource', create_resource_concat_graph))
  def testPaddingsLengthMismatch(self, graph_fn):
    # [FUNCTION DOC]
    # Parameter graph_fn is a concat graph builder. For each numeric type, concatenates a
    # two-by-two piece with a padding list of length one instead of two, expecting an op error
    # that the padding length should be two but is one. Returns nothing.
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        concat = graph_fn(
            sess,
            input_values=[[[0, 1], [2, 3]]],
            input_dtype=dtype,
            num_concats=[1, 1],
            paddings=[0])
        with self.assertRaisesOpError('length 2, but got 1'):
          sess.run(concat)

  @parameterized.named_parameters(('Tensor', create_tensor_concat_graph),
                                  ('Resource', create_resource_concat_graph))
  def testPaddingsNegative(self, graph_fn):
    # [FUNCTION DOC]
    # Parameter graph_fn is a concat graph builder. For each numeric type, concatenates a
    # two-by-two piece with a negative padding at index one, expecting an op error that paddings
    # must be non-negative. Returns nothing.
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        concat = graph_fn(
            sess,
            input_values=[[[0, 1], [2, 3]]],
            input_dtype=dtype,
            num_concats=[1, 1],
            paddings=[0, -1])
        with self.assertRaisesOpError('non-negative, but got -1 at index 1'):
          sess.run(concat)

  @parameterized.named_parameters(('Tensor', create_tensor_concat_graph),
                                  ('Resource', create_resource_concat_graph))
  def testInputRankConcatMismatch(self, graph_fn):
    # [FUNCTION DOC]
    # Parameter graph_fn is a concat graph builder. For each numeric type, supplies a rank-one
    # piece but a concat list of length two, expecting an op error that the list length does not
    # match the input rank. Returns nothing.
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        concat = graph_fn(
            sess, input_values=[[0]], input_dtype=dtype, num_concats=[1, 1])
        with self.assertRaisesOpError(
            '\'num_concats\' length 2, but got rank 1'):
          sess.run(concat)

  @parameterized.named_parameters(('Tensor', create_tensor_concat_graph),
                                  ('Resource', create_resource_concat_graph))
  def testDifferentShapedInputs(self, graph_fn):
    # [FUNCTION DOC]
    # Parameter graph_fn is a concat graph builder. For each numeric type, supplies two pieces of
    # different lengths (one and two) and expects an op error that all inputs must share the
    # expected shape, naming the mismatch at index one. Returns nothing.
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        concat = graph_fn(
            sess,
            input_values=[[0], [1, 2]],
            input_dtype=dtype,
            num_concats=[2])
        with self.assertRaisesOpError(
            r'same expected shape \[1\], but got \[2\] at index 1'):
          sess.run(concat)

  @parameterized.named_parameters(('Tensor', create_tensor_concat_graph),
                                  ('Resource', create_resource_concat_graph))
  def testPaddingExceedsOutputDimSize(self, graph_fn):
    # [FUNCTION DOC]
    # Parameter graph_fn is a concat graph builder. For each numeric type, concatenates a single
    # one-element piece with padding of two, which exceeds the output dimension size, expecting
    # an op error about padding exceeding the expected output dimension. Returns nothing.
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        concat = graph_fn(
            sess,
            input_values=[[0]],
            input_dtype=dtype,
            num_concats=[1],
            paddings=[2])
        with self.assertRaisesOpError(
            'exceed expected output shape dimension 1 at index 0, but got 2'):
          sess.run(concat)

  @parameterized.named_parameters(('Tensor', create_tensor_concat_graph),
                                  ('Resource', create_resource_concat_graph))
  def testNoConcats(self, graph_fn):
    # [FUNCTION DOC]
    # Parameter graph_fn is a concat graph builder. For each numeric type, uses concat counts of
    # one in every dimension with a single two-by-two-by-two piece and expects the result to
    # equal that piece. Returns nothing.
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        concat = graph_fn(
            sess,
            input_values=[[[[0, 1], [2, 3]], [[4, 5], [6, 7]]]],
            input_dtype=dtype,
            num_concats=[1, 1, 1],
            output_shape=[2, 2, 2])
        result = sess.run(concat)
      self.assertAllClose(result, [[[0, 1], [2, 3]], [[4, 5], [6, 7]]])

  @parameterized.named_parameters(('Tensor', create_tensor_concat_graph),
                                  ('Resource', create_resource_concat_graph))
  def testNoConcatsWithPadding(self, graph_fn):
    # [FUNCTION DOC]
    # Parameter graph_fn is a concat graph builder. For each numeric type, uses concat counts of
    # one with a single two-by-two-by-two piece and padding of one in each dimension, so the
    # result is trimmed to shape one by one by one; expects the first element, zero.
    # Returns nothing.
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        concat = graph_fn(
            sess,
            input_values=[[[[0, 1], [2, 3]], [[4, 5], [6, 7]]]],
            input_dtype=dtype,
            num_concats=[1, 1, 1],
            output_shape=[1, 1, 1],
            paddings=[1, 1, 1])
        result = sess.run(concat)
      self.assertAllClose(result, [[[0]]])

  @parameterized.named_parameters(('Tensor', create_tensor_concat_graph),
                                  ('Resource', create_resource_concat_graph))
  def testConcatNoPadding(self, graph_fn):
    # [FUNCTION DOC]
    # Parameter graph_fn is a concat graph builder. For each numeric type, concatenates four
    # two-by-two pieces in a two-by-two grid (row-major order) into a four-by-four matrix and
    # expects the tiled values. Returns nothing.
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        concat = graph_fn(
            sess,
            input_values=[
                [[0, 1], [2, 3]],
                [[4, 5], [6, 7]],
                [[8, 9], [10, 11]],
                [[12, 13], [14, 15]],
            ],
            input_dtype=dtype,
            num_concats=[2, 2],
            output_shape=[4, 4])
        result = sess.run(concat)
      self.assertAllClose(
          result,
          [[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]])

  @parameterized.named_parameters(('Tensor', create_tensor_concat_graph),
                                  ('Resource', create_resource_concat_graph))
  def testConcatPartialPadding(self, graph_fn):
    # [FUNCTION DOC]
    # Parameter graph_fn is a concat graph builder. For each numeric type, concatenates four
    # two-by-two pieces in a two-by-two grid with padding of one in each dimension, so the
    # four-by-four result is trimmed to three-by-three. Expects the top-left three-by-three
    # corner of the tiled values. Returns nothing.
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        concat = graph_fn(
            sess,
            input_values=[
                [[0, 1], [2, 3]],
                [[4, 5], [6, 7]],
                [[8, 9], [10, 11]],
                [[12, 13], [14, 15]],
            ],
            input_dtype=dtype,
            num_concats=[2, 2],
            output_shape=[3, 3],
            paddings=[1, 1])
        result = sess.run(concat)
      self.assertAllClose(result, [[0, 1, 4], [2, 3, 6], [8, 9, 12]])

  @parameterized.named_parameters(('Tensor', create_tensor_concat_graph),
                                  ('Resource', create_resource_concat_graph))
  def testConcatCompletePadding(self, graph_fn):
    # [FUNCTION DOC]
    # Parameter graph_fn is a concat graph builder. For each numeric type, concatenates four
    # two-by-two pieces in a two-by-two grid with padding of two in each dimension, trimming the
    # result to two-by-two; expects just the first piece. Returns nothing.
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        concat = graph_fn(
            sess,
            input_values=[
                [[0, 1], [2, 3]],
                [[4, 5], [6, 7]],
                [[8, 9], [10, 11]],
                [[12, 13], [14, 15]],
            ],
            input_dtype=dtype,
            num_concats=[2, 2],
            output_shape=[2, 2],
            paddings=[2, 2])
        result = sess.run(concat)
      self.assertAllClose(result, [[0, 1], [2, 3]])

  @parameterized.named_parameters(
      ('1Tensor', create_tensor_concat_graph, 1),
      ('2Tensor', create_tensor_concat_graph, 2),
      ('3Tensor', create_tensor_concat_graph, 3),
      ('4Tensor', create_tensor_concat_graph, 4),
      ('5Tensor', create_tensor_concat_graph, 5),
      ('6Tensor', create_tensor_concat_graph, 6),
      ('7Tensor', create_tensor_concat_graph, 7),
      ('8Tensor', create_tensor_concat_graph, 8),
      ('1Resource', create_resource_concat_graph, 1),
      ('2Resource', create_resource_concat_graph, 2),
      ('3Resource', create_resource_concat_graph, 3),
      ('4Resource', create_resource_concat_graph, 4),
      ('5Resource', create_resource_concat_graph, 5),
      ('6Resource', create_resource_concat_graph, 6),
      ('7Resource', create_resource_concat_graph, 7),
      ('8Resource', create_resource_concat_graph, 8),
  )
  def testRanked(self, graph_fn, rank):
    # [FUNCTION DOC]
    # Parameterized by a concat graph builder and a rank from one to eight (each for the tensor
    # and resource forms). For each numeric type, concatenates two to the power of rank pieces,
    # each a single element numbered in order, in a grid of two per dimension, and expects the
    # result to be the numbers in row-major order with shape two in every dimension.
    # Returns nothing.
    num_concats = [2] * rank
    num_inputs = 2 << (rank - 1)
    input_values = [np.reshape(i, [1] * rank) for i in range(num_inputs)]
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        concat = graph_fn(
            sess,
            input_values=input_values,
            input_dtype=dtype,
            num_concats=num_concats,
            output_shape=num_concats)
        result = sess.run(concat)
      expected_output = np.arange(0,
                                  num_inputs).reshape(num_concats).astype(dtype)
      self.assertAllClose(result, expected_output)


def create_tensor_roundtrip_graph(
    sess: Session,
    value: Any,
    dtype: Any,
    num_partitions: List[int],
    paddings: Optional[List[int]] = None) -> Tensor:
  # [FUNCTION DOC]
  # HELPER: Graph builder for the round-trip tests, tensor-input variant.
  # Inputs: sess unused (deleted); value data array; dtype element type; num_partitions list of
  # split counts per dimension; paddings optional per-dimension padding (default none).
  # Splits a constant tensor with the split op, concatenates the pieces back with the concat op
  # using the same partitions and padding, and returns a tensor of element-wise equality between
  # the original and the round-tripped result.
  del sess

  const_input_op = constant_op.constant(value, dtype=dtype)
  split = gen_tpu_ops.xla_split_nd(
      const_input_op,
      np.prod(num_partitions),
      num_partitions,
      paddings=paddings)
  concat = gen_tpu_ops.xla_concat_nd(split, num_partitions, paddings)
  return math_ops.equal(const_input_op, concat)


def create_resource_roundtrip_graph(
    sess: Session,
    value: Any,
    dtype: Any,
    num_partitions: List[int],
    paddings: Optional[List[int]] = None) -> Tensor:
  # [FUNCTION DOC]
  # HELPER: Graph builder for the round-trip tests, resource-variable variant.
  # Inputs: sess session used to initialize the variable; value data array; dtype element type;
  # num_partitions split counts per dimension; paddings optional per-dimension padding.
  # Creates and initializes a resource variable (side effect), splits it directly from the
  # handle, assigns the concatenation of the pieces back into the same variable, and returns a
  # tensor of element-wise equality between the variable afterwards and the original value.
  variable = resource_variable_ops.ResourceVariable(
      initial_value=value, dtype=dtype)
  sess.run(variables.variables_initializer([variable]))
  split = gen_tpu_ops.read_variable_xla_split_nd(
      variable.handle,
      dtype,
      np.prod(num_partitions),
      num_partitions,
      paddings=paddings)
  concat = gen_tpu_ops.assign_variable_xla_concat_nd(variable.handle, split,
                                                     num_partitions, paddings)
  with control_dependencies([concat]):
    return math_ops.equal(variable.read_value(),
                          constant_op.constant(value, dtype=dtype))


class XlaSplitConcatNDTest(xla_test.XLATestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized XLA test case checking that splitting a tensor and concatenating the pieces
  # (with the same partitioning and padding) reproduces the input, for ranks one to eight and for
  # both tensor and resource-variable forms. Holds no state of its own.

  @parameterized.named_parameters(
      ('1Tensor', create_tensor_roundtrip_graph, 1),
      ('2Tensor', create_tensor_roundtrip_graph, 2),
      ('3Tensor', create_tensor_roundtrip_graph, 3),
      ('4Tensor', create_tensor_roundtrip_graph, 4),
      ('5Tensor', create_tensor_roundtrip_graph, 5),
      ('6Tensor', create_tensor_roundtrip_graph, 6),
      ('7Tensor', create_tensor_roundtrip_graph, 7),
      ('8Tensor', create_tensor_roundtrip_graph, 8),
      ('1Resource', create_resource_roundtrip_graph, 1),
      ('2Resource', create_resource_roundtrip_graph, 2),
      ('3Resource', create_resource_roundtrip_graph, 3),
      ('4Resource', create_resource_roundtrip_graph, 4),
      ('5Resource', create_resource_roundtrip_graph, 5),
      ('6Resource', create_resource_roundtrip_graph, 6),
      ('7Resource', create_resource_roundtrip_graph, 7),
      ('8Resource', create_resource_roundtrip_graph, 8),
  )
  def testNoPadding(self, graph_fn, rank):
    # [FUNCTION DOC]
    # Parameterized by a round-trip graph builder and a rank from one to eight. For each numeric
    # type, splits a tensor of shape four in every dimension into two parts per dimension and
    # concatenates it back; expects the equality result to be true everywhere. Returns nothing.
    num_partitions = [2] * rank
    shape = [4] * rank
    value = np.arange(0, np.prod(shape)).reshape(shape)
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        validate = graph_fn(sess, value, dtype, num_partitions)
        result = sess.run(validate)
      self.assertAllEqual(result, np.broadcast_to(True, shape))

  @parameterized.named_parameters(
      ('1Tensor', create_tensor_roundtrip_graph, 1),
      ('2Tensor', create_tensor_roundtrip_graph, 2),
      ('3Tensor', create_tensor_roundtrip_graph, 3),
      ('4Tensor', create_tensor_roundtrip_graph, 4),
      ('5Tensor', create_tensor_roundtrip_graph, 5),
      ('6Tensor', create_tensor_roundtrip_graph, 6),
      ('7Tensor', create_tensor_roundtrip_graph, 7),
      ('8Tensor', create_tensor_roundtrip_graph, 8),
      ('1Resource', create_resource_roundtrip_graph, 1),
      ('2Resource', create_resource_roundtrip_graph, 2),
      ('3Resource', create_resource_roundtrip_graph, 3),
      ('4Resource', create_resource_roundtrip_graph, 4),
      ('5Resource', create_resource_roundtrip_graph, 5),
      ('6Resource', create_resource_roundtrip_graph, 6),
      ('7Resource', create_resource_roundtrip_graph, 7),
      ('8Resource', create_resource_roundtrip_graph, 8),
  )
  def testPartialPadding(self, graph_fn, rank):
    # [FUNCTION DOC]
    # Parameterized by a round-trip graph builder and a rank from one to eight. Same as the
    # no-padding round trip but with padding of two in every dimension; expects the equality
    # result to be true everywhere. Returns nothing.
    num_partitions = [2] * rank
    shape = [4] * rank
    value = np.arange(0, np.prod(shape)).reshape(shape)
    paddings = [2] * rank
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        validate = graph_fn(sess, value, dtype, num_partitions, paddings)
        result = sess.run(validate)
      self.assertAllEqual(result, np.broadcast_to(True, shape))

  @parameterized.named_parameters(
      ('1Tensor', create_tensor_roundtrip_graph, 1),
      ('2Tensor', create_tensor_roundtrip_graph, 2),
      ('3Tensor', create_tensor_roundtrip_graph, 3),
      ('4Tensor', create_tensor_roundtrip_graph, 4),
      ('5Tensor', create_tensor_roundtrip_graph, 5),
      ('6Tensor', create_tensor_roundtrip_graph, 6),
      ('7Tensor', create_tensor_roundtrip_graph, 7),
      ('8Tensor', create_tensor_roundtrip_graph, 8),
      ('1Resource', create_resource_roundtrip_graph, 1),
      ('2Resource', create_resource_roundtrip_graph, 2),
      ('3Resource', create_resource_roundtrip_graph, 3),
      ('4Resource', create_resource_roundtrip_graph, 4),
      ('5Resource', create_resource_roundtrip_graph, 5),
      ('6Resource', create_resource_roundtrip_graph, 6),
      ('7Resource', create_resource_roundtrip_graph, 7),
      ('8Resource', create_resource_roundtrip_graph, 8),
  )
  def testCompletePadding(self, graph_fn, rank):
    # [FUNCTION DOC]
    # Parameterized by a round-trip graph builder and a rank from one to eight. Same as the
    # no-padding round trip but with padding of four in every dimension (a whole extra tile of
    # padding); expects the equality result to be true everywhere. Returns nothing.
    num_partitions = [2] * rank
    shape = [4] * rank
    value = np.arange(0, np.prod(shape)).reshape(shape)
    paddings = [4] * rank
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        validate = graph_fn(sess, value, dtype, num_partitions, paddings)
        result = sess.run(validate)
      self.assertAllEqual(result, np.broadcast_to(True, shape))


if __name__ == '__main__':
  test.main()
