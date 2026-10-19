# [MODULE DOC]
# Functional tests for the tensor split operation (even splits and splits with explicit or
# partially unknown size lists). Covers static and runtime shape inference and error cases,
# special cases such as a size of minus one, empty tensors, identity splits, very many outputs,
# randomized rank-5 splits over many dtypes (including complex and low-precision floats),
# gradients, and splitting very large tensors.
# Import-time side effect: builds the module-level tuple of test dtypes, adding the 4-bit integer
# type unless XLA is enabled.
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
"""Functional tests for Split Op."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

_TEST_DTYPES = (
    dtypes.int8,
    dtypes.float32,
    dtypes.float64,
    dtypes.complex64,
    dtypes.complex128,
    dtypes.bfloat16,
    dtypes.float8_e5m2,
    dtypes.float8_e4m3fn,
)

if not test_util.is_xla_enabled():
  # TODO(b/183567451): Support int4 in XLA
  _TEST_DTYPES += (dtypes.int4,)


class SplitOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the split operation. Uses the module-level tuple of dtypes for most checks and
  # helper methods (private names) that generate random data, compare with NumPy splitting, and
  # verify gradients. Holds no state.

  def _makeData(self, shape, dtype):
    # [FUNCTION DOC]
    # HELPER: test data generator. Inputs: shape (sequence of dimension sizes) and dtype (a
    # TensorFlow dtype). Returns a NumPy array of random uniform values converted to the dtype's
    # NumPy type; for complex dtypes the imaginary part is set to the negated real part (so
    # values are nonzero complex numbers).
    data = np.random.rand(*shape).astype(dtype.as_numpy_dtype)
    if dtype.is_complex:
      data -= 1j * data
    return data

  @test_util.run_deprecated_v1
  def testShapeInference(self):
    # [FUNCTION DOC]
    # Test (graph mode): (1) a size list that does not sum to the known dimension raises a value
    # error at construction; (2) with an unknown-shape input the same split fails at run time
    # with an invalid-argument error; (3) a scalar tensor as the splits argument raises a value
    # error for axes 0 and minus 2; (4) with a size list computed from a constant, non-split
    # dimensions of the outputs keep their static size; (5) a split with a partially unknown
    # input shape can be run with fed data. Returns nothing.
    model_input = array_ops.placeholder(dtypes.float32, shape=(1, 10))

    # check that we fail during static shape inference if sizes are known
    with self.assertRaises(ValueError):
      # pylint: disable=expression-not-assigned
      array_ops.split(model_input, [4], axis=1)[0]
      # pylint: enable=expression-not-assigned

    model_input = array_ops.placeholder(dtypes.float32)
    inp = np.zeros((1, 10))
    # check that we still fail at runtime if the shapes were unknown
    with self.cached_session() as sess:
      with self.assertRaises(errors_impl.InvalidArgumentError):
        sess.run(array_ops.split(model_input, [4]), {model_input: inp})

    # scalar Tensors are not permitted as num_splits
    for axis in [0, -2]:
      with self.cached_session() as sess:
        with self.assertRaises(ValueError):
          # pylint: disable=expression-not-assigned
          sess.run(
              array_ops.split(
                  array_ops.ones([4, 4]),
                  num_or_size_splits=constant_op.constant(2),
                  axis=axis))
          # pylint: enable=expression-not-assigned

    # test that none split dimensions remain, even if we don't know how
    # the split_dim will be split, but we do know the axis
    result = array_ops.split(
        array_ops.ones([5, 2]), array_ops.constant([2, 1, 2]) * 1, axis=0)

    self.assertEqual(result[0].shape[1], 2)
    self.assertEqual(result[1].shape[1], 2)
    self.assertEqual(result[2].shape[1], 2)

    model_input2 = array_ops.placeholder(dtypes.float32, shape=[None, 2])
    result = array_ops.split(model_input2, [2, 2], axis=0)[0]

    with self.cached_session() as sess:
      sess.run(result, feed_dict={model_input2: np.ones([4, 2])})

  @test_util.run_deprecated_v1
  def testFailWithoutExplicitNum(self):
    # [FUNCTION DOC]
    # Test (graph mode): splitting with sizes held in a placeholder of unknown length and no
    # explicit output count must raise a value error whose message says the num argument cannot
    # be inferred from the shape. Returns nothing.
    size_splits = array_ops.placeholder(dtype=dtypes.int32, shape=[None])

    value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    with self.session() as sess:
      with self.assertRaises(ValueError) as context:
        sess.run(array_ops.split(value, size_splits), {size_splits: [2, 2, 6]})
      self.assertIn("Cannot infer argument `num` from shape",
                    str(context.exception))

  @test_util.run_in_graph_and_eager_modes
  def testExplicitNum(self):
    # [FUNCTION DOC]
    # Test (graph and eager modes): giving three sizes but asking for four outputs raises an
    # invalid-argument or value error (the type differs by mode); asking for three outputs
    # returns pieces equal to the matching sublists of the input. Returns nothing.
    size_splits = array_ops.constant([2, 2, 6], dtype=dtypes.int32)
    value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Eager and Graph modes raise different exceptions
    with self.assertRaises((errors_impl.InvalidArgumentError, ValueError)):
      array_ops.split(value, size_splits, num=4)

    r = self.evaluate(array_ops.split(value, size_splits, num=3))
    self.assertAllEqual(r[0], value[0:2])
    self.assertAllEqual(r[1], value[2:4])
    self.assertAllEqual(r[2], value[4:])

  @test_util.run_in_graph_and_eager_modes
  def testListOfScalarTensors(self):
    # [FUNCTION DOC]
    # Test (graph and eager modes): splits an 11x11 random matrix with sizes given as a list of
    # two scalar tensors (5 and 6) on a GPU-preferred device; asserts the two parts equal the
    # first five rows and the remaining rows. Returns nothing.
    a = math_ops.cast(5, dtypes.int32)
    b = math_ops.cast(6, dtypes.int32)

    value = np.random.rand(11, 11)

    with test_util.device(use_gpu=True):
      result = self.evaluate(array_ops.split(value, [a, b]))

    self.assertAllEqual(result[0], value[0:5, :])
    self.assertAllEqual(result[1], value[5:, :])

  def _RunAndVerifyVariable(self, dtype, large_num_splits=False):
    # [FUNCTION DOC]
    # HELPER: randomized check of splitting with explicit sizes. Inputs: dtype, and a flag
    # (default false) choosing a larger number of splits (16 to 24) versus a smaller one (2 to 7).
    # Builds a random rank-5 array whose size along a random (possibly negative) split axis is the
    # sum of random sizes, splits it with the size list on a GPU-preferred device, and asserts each
    # piece equals the corresponding NumPy slice. Returns nothing.
    # Random dims of rank 5
    shape = np.random.randint(1, 5, size=5)
    split_dim = np.random.randint(-5, 5)
    if large_num_splits:
      num_split = np.random.randint(16, 25)
    else:
      num_split = np.random.randint(2, 8)
    size_splits = np.random.randint(2, 8, num_split, dtype=np.int32)
    shape[split_dim] = np.sum(size_splits)
    inp = self._makeData(shape, dtype)
    with test_util.device(use_gpu=True):
      result = self.evaluate(array_ops.split(inp, size_splits, split_dim))
    slices = [slice(0, x) for x in shape]
    offset = 0
    for i in range(num_split):
      slices[split_dim] = slice(offset, offset + size_splits[i])
      offset += size_splits[i]
      self.assertAllEqual(result[i], inp[tuple(slices)])

  def _testSpecialCasesVariable(self):
    # [FUNCTION DOC]
    # HELPER: checks that splitting a 4x4 matrix into a single piece of size 4 returns the whole
    # input, and that sizes of minus one and three infer the first piece as one row. Returns
    # nothing.
    inp = np.random.rand(4, 4).astype("f")

    with test_util.device(use_gpu=True):
      result = self.evaluate(array_ops.split(inp, [4], 0))
      self.assertAllEqual(result[0], inp)

      result = self.evaluate(array_ops.split(inp, [-1, 3], 0))
      self.assertAllEqual(result[0], inp[0:1, :])
      self.assertAllEqual(result[1], inp[1:4, :])

  def _testHugeNumberOfTensorsVariable(self, dtype):
    # [FUNCTION DOC]
    # HELPER: splits a 3-row matrix along its columns into 1000 pieces of random width (1 or 2)
    # and asserts each piece equals the matching NumPy slice. Input: dtype of the data. Returns
    # nothing.
    num_split = 1000
    size_splits = np.random.randint(1, 3, num_split, dtype=np.int32)
    shape = [3, np.sum(size_splits)]
    split_dim = 1
    inp = self._makeData(shape, dtype)
    with test_util.device(use_gpu=True):
      result = self.evaluate(array_ops.split(inp, size_splits, split_dim))
    slices = [slice(0, x) for x in shape]
    offset = 0
    for i in range(num_split):
      slices[split_dim] = slice(offset, offset + size_splits[i])
      offset += size_splits[i]
      self.assertAllEqual(result[i], inp[tuple(slices)])

  @test_util.run_in_graph_and_eager_modes
  def testSpecialCasesVariable(self):
    # [FUNCTION DOC]
    # Test (graph and eager modes): runs the special-case helper, then the huge-number-of-outputs
    # helper for every test dtype. Returns nothing.
    self._testSpecialCasesVariable()
    for dtype in _TEST_DTYPES:
      self._testHugeNumberOfTensorsVariable(dtype)

  @test_util.run_in_graph_and_eager_modes
  def testDegenerateVariable(self):
    # [FUNCTION DOC]
    # Test (graph and eager modes): splits a 4x4 matrix into two pieces where one has size 4 and
    # the other is inferred (minus one) as zero, along both axes and in both orders; asserts the
    # empty piece and the full piece match the NumPy slices. Returns nothing.
    inp = np.random.rand(4, 4).astype("f")
    with test_util.device(use_gpu=True):
      result = self.evaluate(array_ops.split(inp, [-1, 4], 0))
      self.assertAllEqual(result[0], inp[0:0, :])
      self.assertAllEqual(result[1], inp[0:4, :])

      result = self.evaluate(array_ops.split(inp, [4, -1], 0))
      self.assertAllEqual(result[0], inp[0:4, :])
      self.assertAllEqual(result[1], inp[4:4, :])

      result = self.evaluate(array_ops.split(inp, [-1, 4], 1))
      self.assertAllEqual(result[0], inp[:, 0:0])
      self.assertAllEqual(result[1], inp[:, 0:4])

      result = self.evaluate(array_ops.split(inp, [4, -1], 1))
      self.assertAllEqual(result[0], inp[:, 0:4])
      self.assertAllEqual(result[1], inp[:, 4:4])

  def _testGradientsSimpleVariable(self, dtype):
    # [FUNCTION DOC]
    # HELPER: gradient check for splitting a 4x4 tensor into column pieces of width 1 and 3 with an
    # explicit size list. Feeds random upstream gradients for both pieces and asserts the input
    # gradient equals them concatenated. Input: dtype. Returns nothing.
    inp = self._makeData((4, 4), dtype)
    with test_util.device(use_gpu=True):
      inp_tensor = ops.convert_to_tensor(inp)
      s = array_ops.split(inp_tensor, [1, 3], 1)
      inp_grads = [
          self._makeData((4, 1), dtype), self._makeData((4, 3), dtype)
      ]
      grad_tensors = [constant_op.constant(x) for x in inp_grads]
      grad = gradients_impl.gradients(s, [inp_tensor], grad_tensors)[-1]
      result = self.evaluate(grad)

    self.assertAllEqual(result[:, 0:1], inp_grads[0])
    self.assertAllEqual(result[:, 1:4], inp_grads[1])

  @test_util.run_deprecated_v1
  def testOutputShape(self):
    # [FUNCTION DOC]
    # Test (graph mode): splitting a placeholder of shape unknown by 12 into pieces of widths 3,
    # 7 and 2 along axis 1 and along axis minus 1; asserts each output static shape has an
    # unknown first dimension and the corresponding width. Returns nothing.
    for axis in [1, -1]:
      with self.cached_session():
        tensor = array_ops.placeholder(dtypes.float32, shape=[None, 12])
        size_splits = [3, 7, 2]
        outputs = array_ops.split(tensor, size_splits, axis)
        for i, output in enumerate(outputs):
          self.assertEqual(output.get_shape().as_list(), [None, size_splits[i]])

  def _compare(self, x, dim, num):
    # [FUNCTION DOC]
    # HELPER: compares an even split with NumPy. Inputs: x (array), dim (axis), num (number of
    # equal pieces). Splits on a GPU-preferred device, asserts the NumPy and TensorFlow piece
    # counts and the evaluated outputs match, and that static shapes match NumPy's shapes.
    # Returns nothing.
    np_ans = np.split(x, num, dim)
    with test_util.device(use_gpu=True):
      tf_ans = array_ops.split(value=x, num_or_size_splits=num, axis=dim)
      out = self.evaluate(tf_ans)
    self.assertEqual(num, len(np_ans))
    self.assertEqual(num, len(tf_ans))
    self.assertEqual(num, len(out))
    for i in range(num):
      self.assertAllEqual(np_ans[i], out[i])
      self.assertShapeEqual(np_ans[i], tf_ans[i])

  @test_util.run_in_graph_and_eager_modes
  def testSplitRows(self):
    # [FUNCTION DOC]
    # Test (graph and eager modes): for each test dtype, splits a 4x4 matrix into four row pieces
    # and compares with NumPy. Returns nothing.
    for dtype in _TEST_DTYPES:
      inp = self._makeData((4, 4), dtype)
      self._compare(inp, 0, 4)

  @test_util.run_in_graph_and_eager_modes
  def testSplitCols(self):
    # [FUNCTION DOC]
    # Test (graph and eager modes): for each test dtype, splits a 4x4 matrix into four column
    # pieces and compares with NumPy. Returns nothing.
    for dtype in _TEST_DTYPES:
      inp = self._makeData((4, 4), dtype)
      self._compare(inp, 1, 4)

  def _testEmpty(self, x, dim, num, expected_shape):
    # [FUNCTION DOC]
    # HELPER: splitting an empty array. Inputs: x (empty array), dim (axis), num (pieces),
    # expected_shape (shape of each piece). Splits on a GPU-preferred device and asserts the input
    # is empty, there are num outputs, and each output and its static shape equal expected_shape.
    # Returns nothing. (NumPy is not used for comparison because its split of an empty array
    # returns a rank-0 result.)
    with test_util.device(use_gpu=True):
      tf_ans = array_ops.split(value=x, num_or_size_splits=num, axis=dim)
      out = self.evaluate(tf_ans)
    self.assertEqual(x.size, 0)
    self.assertEqual(len(out), num)
    for i in range(num):
      self.assertEqual(out[i].shape, expected_shape)
      self.assertEqual(expected_shape, tf_ans[i].get_shape())

  @test_util.run_in_graph_and_eager_modes
  def testEmpty(self):
    # [FUNCTION DOC]
    # Test (graph and eager modes): for each test dtype, splits an 8x0x21 empty array along
    # different axes into several piece counts, checking piece shapes via the empty helper.
    # Returns nothing.
    # Note: np.split returns a rank-0 empty ndarray
    # if the input ndarray is empty.
    for dtype in _TEST_DTYPES:
      inp = self._makeData((8, 0, 21), dtype)
      self._testEmpty(inp, 0, 2, (4, 0, 21))
      self._testEmpty(inp, 0, 4, (2, 0, 21))
      self._testEmpty(inp, 1, 4, (8, 0, 21))
      self._testEmpty(inp, 2, 3, (8, 0, 7))
      self._testEmpty(inp, 2, 7, (8, 0, 3))

  @test_util.run_in_graph_and_eager_modes
  def testIdentity(self):
    # [FUNCTION DOC]
    # Test (graph and eager modes): for each test dtype, splitting a 2x2x2 array into one piece
    # along each axis returns the array unchanged. Returns nothing.
    for dtype in _TEST_DTYPES:
      inp = self._makeData((2, 2, 2), dtype)
      self._compare(inp, 0, 1)
      self._compare(inp, 1, 1)
      self._compare(inp, 2, 1)

  @test_util.run_in_graph_and_eager_modes
  def testSplitDim0(self):
    # [FUNCTION DOC]
    # Test (graph and eager modes): for each test dtype, splits three differently shaped arrays
    # into three pieces along axis 0 and compares with NumPy. Returns nothing.
    for dtype in _TEST_DTYPES:
      self._compare(self._makeData((6, 10, 18), dtype), 0, 3)
      self._compare(self._makeData((6, 7, 18), dtype), 0, 3)
      self._compare(self._makeData((6, 7, 9), dtype), 0, 3)

  def _RunAndVerify(self, dtype, large_num_splits=False):
    # [FUNCTION DOC]
    # HELPER: randomized check of even splitting. Inputs: dtype, and a flag (default false)
    # choosing a larger piece count (9 to 14) versus a smaller one (2 to 7). Builds a random rank-5
    # array (dimensions may be zero) whose size along a random (possibly negative) axis is a
    # multiple of the piece count, splits it on a GPU-preferred device, and asserts each piece
    # equals the matching NumPy slice. Returns nothing.
    # Random dims of rank 5
    shape = np.random.randint(0, 5, size=5)
    split_dim = np.random.randint(-5, 5)
    if large_num_splits:
      num_split = np.random.randint(9, 15)
    else:
      num_split = np.random.randint(2, 8)
    shape[split_dim] = np.random.randint(2, 5) * num_split
    inp = self._makeData(shape, dtype)
    with test_util.device(use_gpu=True):
      result = self.evaluate(
          array_ops.split(
              value=inp, num_or_size_splits=num_split, axis=split_dim))
    slices = [slice(0, x) for x in shape]
    offset = 0
    length = shape[split_dim] // num_split
    for i in range(num_split):
      slices[split_dim] = slice(offset, offset + length)
      offset += length
      self.assertAllEqual(result[i], inp[tuple(slices)])

  @test_util.run_in_graph_and_eager_modes
  def testRandom(self):
    # [FUNCTION DOC]
    # Test (graph and eager modes): for each test dtype, five times runs the randomized even-split
    # and explicit-size-split checks with both small and large piece counts. Returns nothing.
    for dtype in _TEST_DTYPES:
      for _ in range(5):
        self._RunAndVerify(dtype)
        self._RunAndVerify(dtype, large_num_splits=True)
        self._RunAndVerifyVariable(dtype)
        self._RunAndVerifyVariable(dtype, large_num_splits=True)

  def _testGradientsSimple(self, dtype):
    # [FUNCTION DOC]
    # HELPER: gradient check for splitting a 4x4 tensor into four equal column pieces. Feeds
    # random upstream gradients (one per piece) and asserts the input gradient columns equal them.
    # Input: dtype. Returns nothing.
    inp = self._makeData((4, 4), dtype)
    with self.cached_session():
      inp_tensor = ops.convert_to_tensor(inp)
      s = array_ops.split(value=inp_tensor, num_or_size_splits=4, axis=1)
      inp_grads = [self._makeData((4, 1), dtype)for _ in range(4)]
      grad_tensors = [constant_op.constant(x) for x in inp_grads]
      grad = gradients_impl.gradients(s, [inp_tensor], grad_tensors)[0]
      result = self.evaluate(grad)
    for i in range(4):
      self.assertAllEqual(result[:, i:i + 1], inp_grads[i])

  @test_util.run_deprecated_v1
  def testGradientsAll(self):
    # [FUNCTION DOC]
    # Test (graph mode): runs both gradient helpers for every test dtype that is not an integer
    # type and not one of the two 8-bit float types. Returns nothing.
    for dtype in _TEST_DTYPES:
      if not dtype.is_integer and dtype not in [
          dtypes.float8_e5m2,
          dtypes.float8_e4m3fn,
      ]:
        self._testGradientsSimple(dtype)
        self._testGradientsSimpleVariable(dtype)

  @test_util.run_deprecated_v1
  def testShapeFunctionEdgeCases(self):
    # [FUNCTION DOC]
    # Test (graph mode): invalid arguments raise value errors at construction: axis beyond the
    # rank, axis below minus rank, and a piece count that does not evenly divide the axis size
    # (message about evenly dividing). With an unknown axis, every output has two unknown
    # dimensions; with unknown input shape and unknown axis, outputs have unknown rank.
    # Returns nothing.
    # split_dim greater than rank of input.
    with self.assertRaises(ValueError):
      array_ops.split(value=[[0, 1], [2, 3]], num_or_size_splits=4, axis=2)

    # split dim less than -(rank of input)
    with self.assertRaises(ValueError):
      array_ops.split(value=[[0, 1], [2, 3]], num_or_size_splits=4, axis=-3)

    # num_split does not evenly divide the size in split_dim.
    with self.assertRaisesRegex(ValueError, "should evenly divide"):
      array_ops.split(value=[0, 1, 2, 3], num_or_size_splits=3, axis=0)

    # Unknown split_dim.
    splits = array_ops.split(
        value=[[0, 1, 2, 3]],
        num_or_size_splits=4,
        axis=array_ops.placeholder(dtypes.int32))
    for s in splits:
      self.assertEqual([None, None], s.get_shape().as_list())

    # Unknown split_dim and input shape.
    splits = array_ops.split(
        value=array_ops.placeholder(dtypes.float32),
        num_or_size_splits=4,
        axis=array_ops.placeholder(dtypes.int32))
    for s in splits:
      self.assertEqual(None, s.get_shape().ndims)

  @test_util.run_deprecated_v1
  def testVariableShapeFunction(self):
    # [FUNCTION DOC]
    # Test (graph mode): a size list whose explicit sizes exceed the axis length raises a value
    # error; a size list with a minus-one entry has its static shape inferred correctly (sizes 2
    # and 1 for a length-3 input). Returns nothing.
    # size_splits too big
    with self.assertRaises(ValueError):
      array_ops.split([0, 1], [3, -1], axis=0)

    # Correct inference of variable dimension
    s0, s1 = array_ops.split([0, 1, 2], [2, -1], axis=0)
    assert s0.shape.as_list() == [2]
    assert s1.shape.as_list() == [1]

  @test_util.run_deprecated_v1
  def testNonexistentDimTensor(self):
    # [FUNCTION DOC]
    # Test (graph mode): with a placeholder axis and placeholder sizes of unknown length, building
    # the split raises a value error saying it cannot infer the output count. With sizes of known
    # length three, building succeeds, but running with an empty array fed as the axis raises an
    # invalid-argument error saying the axis must have exactly one element. Returns nothing.
    x = array_ops.placeholder(dtypes.int32)
    values = np.zeros([5, 30])
    splits = array_ops.placeholder(dtypes.int32)
    with self.assertRaisesRegex(ValueError, "Cannot infer"):
      y = array_ops.split(values, splits, axis=x)

    splits = array_ops.placeholder(dtypes.int32, [3])
    y = array_ops.split(values, splits, axis=x)
    with self.session() as sess:
      with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                  "must have exactly one element"):
        sess.run(y, {x: np.array([], dtype=np.int32), splits: [4, 11, 15]})

  @test_util.run_in_graph_and_eager_modes
  def testNegativeSizes(self):
    # [FUNCTION DOC]
    # Test (graph and eager modes): a size list of minus one and minus two raises a value or
    # invalid-argument error saying the split size at index 1 must be at least the allowed
    # minimum, got minus two (only minus one is allowed as an inferred size). Returns nothing.
    x = constant_op.constant([1, 2, 3], dtypes.float32)
    # A size of -1 signifies to determine size based on sum of other splits.
    with self.assertRaisesRegex((ValueError, errors_impl.InvalidArgumentError),
                                "Split size at index 1 must be >= .*. Got: -2"):
      splits = [-1, -2]
      self.evaluate(array_ops.split(x, splits, axis=0))

  @test_util.run_in_graph_and_eager_modes
  def testBadSplitSizes(self):
    # [FUNCTION DOC]
    # Test (graph and eager modes): splitting a length-2 vector with sizes one and two raises a
    # value or invalid-argument error saying the determined shape must match the input or the
    # axis cannot be split. Returns nothing.
    x = constant_op.constant([1, 2], dtypes.float32)
    with self.assertRaisesRegex((ValueError, errors_impl.InvalidArgumentError),
                                "Determined shape must either match input"
                                "|can't split axis"):
      splits = [1, 2]
      self.evaluate(array_ops.split(x, splits, axis=0))

  @test_util.run_in_graph_and_eager_modes
  def testSplitVBigTensors(self):
    # [FUNCTION DOC]
    # Test (graph and eager modes): splits a 1x64x32768 float ramp tensor along axis 1 into 64
    # pieces of size 1 (explicit size list) and asserts each piece equals the matching NumPy
    # slice. Returns nothing.
    input_shape = [1, 64, 32768]
    x = np.linspace(
        start=1,
        stop=np.prod(input_shape),
        num=np.prod(input_shape),
        dtype=np.float32).reshape(input_shape)
    split_axis = 1
    size_splits = [1] * input_shape[split_axis]

    y = array_ops.split(x, num_or_size_splits=size_splits, axis=split_axis)

    for i in range(input_shape[split_axis]):
      result = y[i]
      expected = x[:, i:i + 1, :]
      self.assertAllEqual(result, expected)

  @test_util.run_in_graph_and_eager_modes
  def testSplitVBigTensorsWithIrregularSplits(self):
    # [FUNCTION DOC]
    # Test (graph and eager modes): splits the same large 1x64x32768 ramp tensor along axis 1 with
    # irregular sizes 32, 16, 8, 4, 2, 1, 1 and asserts each piece equals the matching NumPy
    # slice. Returns nothing.
    input_shape = [1, 64, 32768]
    x = np.linspace(start=1,
                    stop=np.prod(input_shape),
                    num=np.prod(input_shape),
                    dtype=np.float32).reshape(input_shape)
    split_axis = 1
    size_splits = [32, 16, 8, 4, 2, 1, 1]

    y = array_ops.split(x, num_or_size_splits=size_splits, axis=split_axis)

    start = 0
    for i in range(len(size_splits)):
      result = y[i]
      split_size = size_splits[i]
      expected = x[:, start:start+split_size, :]
      start += split_size
      self.assertAllEqual(result, expected)


if __name__ == "__main__":
  test.main()
