# [MODULE DOC]
# Tests for the reshape op. One test case checks reshaping of many element types, unspecified
# (inferred) dimensions, zero-size tensors, gradients, error messages for impossible reshapes,
# static shape inference with unknown or partially known shapes, shape specified as dimension
# objects or shape objects, shapes beyond 32-bit range, and rejection of too many elements.
# No import-time side effects besides running the test main when executed as a script.
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
"""Tests for tensorflow.ops.reshape_op."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.platform import test


@test_util.with_eager_op_as_function
class ReshapeTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the reshape op, decorated so that eager ops also run as compiled functions.
  # Private helpers reshape a numpy array via the op using both default and 64-bit integer shape
  # tensors and compare against numpy; public methods supply dtype and shape scenarios.

  def _testReshape(self, x, y, use_gpu=False):
    # [FUNCTION DOC]
    # HELPER: reshapes numpy array x to the target shape list y in a cached session (GPU
    # placement if use_gpu, default false). Asserts the static shape equals the evaluated shape
    # and matches numpy's reshape, then repeats with the shape given as a 64-bit integer tensor.
    with self.cached_session(use_gpu=use_gpu):
      np_ans = x.reshape(y)
      tf_ans = array_ops.reshape(x, y)
      out = self.evaluate(tf_ans)
      self.assertEqual(tf_ans.get_shape(), out.shape)
      self.assertShapeEqual(np_ans, tf_ans)

      # Repeat with an int64 shape tensor.
      y64 = constant_op.constant(y, dtype=dtypes.int64)
      tf_ans = array_ops.reshape(x, y64)
      out = self.evaluate(tf_ans)
      self.assertEqual(tf_ans.get_shape(), out.shape)
      self.assertShapeEqual(np_ans, tf_ans)

  def _testZeroDimReshape(self, x, shape, expected, use_gpu=False):
    # [FUNCTION DOC]
    # HELPER: reshapes x to shape (possibly containing an inferred dimension) in a cached session
    # and asserts the evaluated shape equals expected, then repeats with a 64-bit integer shape
    # tensor. use_gpu (default false) selects GPU placement.
    with self.cached_session(use_gpu=use_gpu):
      y = array_ops.reshape(x, shape)
      out = self.evaluate(y)
      self.assertEqual(expected, out.shape)

      # Repeat with an int64 shape tensor.
      shape64 = constant_op.constant(shape, dtype=dtypes.int64)
      y = array_ops.reshape(x, shape64)
      out = self.evaluate(y)
      self.assertEqual(expected, out.shape)

  def _testBothReshape(self, x, y):
    # [FUNCTION DOC]
    # HELPER: runs the reshape check for array x and target shape y on CPU and then on GPU.
    self._testReshape(x, y, False)
    self._testReshape(x, y, True)

  def testBoolBasic(self):
    # [FUNCTION DOC]
    # Reshapes a one by six boolean array to two by three and checks the result.
    x = np.arange(1., 7.).reshape([1, 6]) > 3
    self._testBothReshape(x, [2, 3])

  def testFloatBasic(self):
    # [FUNCTION DOC]
    # Reshapes a one by six float32 array to two by three and checks the result.
    x = np.arange(1., 7.).reshape([1, 6]).astype(np.float32)
    self._testBothReshape(x, [2, 3])

  def testFloat16Basic(self):
    # [FUNCTION DOC]
    # Reshapes a one by six float16 array to two by three and checks the result.
    x = np.arange(1., 7.).reshape([1, 6]).astype(np.float16)
    self._testBothReshape(x, [2, 3])

  def testBfloat16Basic(self):
    # [FUNCTION DOC]
    # Reshapes a one by six bfloat16 array to two by three and checks the result.
    x = np.arange(1., 7.).reshape([1, 6]).astype(dtypes.bfloat16.as_numpy_dtype)
    self._testBothReshape(x, [2, 3])

  def testDoubleBasic(self):
    # [FUNCTION DOC]
    # Reshapes a one by six float64 array to two by three and checks the result.
    x = np.arange(1., 7.).reshape([1, 6]).astype(np.float64)
    self._testBothReshape(x, [2, 3])

  def testInt32Basic(self):
    # [FUNCTION DOC]
    # Reshapes a one by six int32 array to two by three and checks the result.
    x = np.arange(1., 7.).reshape([1, 6]).astype(np.int32)
    self._testBothReshape(x, [2, 3])

  def testComplex64Basic(self):
    # [FUNCTION DOC]
    # Reshapes a one by six complex64 array to two by three and checks the result.
    x = np.arange(1., 7.).reshape([1, 6]).astype(np.complex64)
    self._testBothReshape(x, [2, 3])

  def testComplex128Basic(self):
    # [FUNCTION DOC]
    # Reshapes a one by six complex128 array to two by three and checks the result.
    x = np.arange(1., 7.).reshape([1, 6]).astype(np.complex128)
    self._testBothReshape(x, [2, 3])

  def testFloatReshapeThreeDimensions(self):
    # [FUNCTION DOC]
    # Reshapes a one by 27 float32 array to a three by three by three cube and checks the result.
    x = np.arange(1., 28.).reshape([1, 27]).astype(np.float32)
    self._testBothReshape(x, [3, 3, 3])

  def testFloatUnspecifiedDimOnly(self):
    # [FUNCTION DOC]
    # Reshapes a six-element vector using a shape whose only entry is the inferred marker (minus
    # one) and checks the result keeps all six elements.
    x = np.arange(1., 7.).reshape([6]).astype(np.float32)
    self._testBothReshape(x, [-1])

  def testFloatUnspecifiedDimBegin(self):
    # [FUNCTION DOC]
    # Reshapes a six-element vector to a shape with an inferred first dimension and second
    # dimension two; checks the inferred result.
    x = np.arange(1., 7.).reshape([6]).astype(np.float32)
    self._testBothReshape(x, [-1, 2])

  def testFloatUnspecifiedDimEnd(self):
    # [FUNCTION DOC]
    # Reshapes a six-element vector to a shape with first dimension three and an inferred last
    # dimension; checks the inferred result.
    x = np.arange(1., 7.).reshape([6]).astype(np.float32)
    self._testBothReshape(x, [3, -1])

  def testZeroDimBasic(self):
    # [FUNCTION DOC]
    # Reshapes a zero by six (empty) float32 array to zero by two by three; checks the result.
    x = np.zeros([0, 6]).astype(np.float32)
    self._testBothReshape(x, [0, 2, 3])

  def testZeroDimReshapeR1(self):
    # [FUNCTION DOC]
    # Reshapes an empty zero by six array to a rank-1 shape with an inferred dimension; checks
    # the result (an empty vector).
    x = np.zeros([0, 6]).astype(np.float32)
    self._testBothReshape(x, [-1])

  def testZeroDimReshapeR3(self):
    # [FUNCTION DOC]
    # Reshapes an empty zero by six array to a rank-3 shape with an inferred first dimension and
    # trailing two and three; checks the result.
    x = np.zeros([0, 6]).astype(np.float32)
    self._testBothReshape(x, [-1, 2, 3])

  # TODO(vrv): Add tests for failure conditions once python test_util
  # reports errors.

  def testFloatReshapeGradThreeDimensions(self):
    # [FUNCTION DOC]
    # Gradient test: reshapes a two by three by four float32 constant to one by eight by three
    # and uses the version-2 gradient checker to compare theoretical and numeric Jacobians,
    # asserting the maximum error is below 1e-3.
    x = np.arange(1., 25.).reshape([2, 3, 4]).astype(np.float32)
    input_tensor = constant_op.constant(x)

    def reshape(x):
      # [FUNCTION DOC]
      # HELPER: closure applying the reshape op to one-by-eight-by-three, used as the function
      # under test for the gradient checker. Takes a tensor, returns the reshaped tensor.
      return array_ops.reshape(x, [1, 8, 3])

    with self.cached_session():
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(reshape, [input_tensor]))
      self.assertLess(err, 1e-3)

  def testFloatEmpty(self):
    # [FUNCTION DOC]
    # Reshapes an all-zero-size four-dimensional float32 array to several shapes containing zeros
    # (and one with an inferred dimension next to a five) and checks each result.
    x = np.empty((0, 0, 0, 0), dtype=np.float32)
    self._testBothReshape(x, [1, 2, 3, 0])
    self._testBothReshape(x, [1, 0, 0, 4])
    self._testBothReshape(x, [0, 0, 0, 0])
    self._testBothReshape(x, [1, 2, 0])
    self._testBothReshape(x, [0, 0, 0])
    self._testBothReshape(x, [1, -1, 5])

  def testZeroDimWithUnspecifiedDim(self):
    # [FUNCTION DOC]
    # On GPU and CPU, reshapes an empty zero by six array to a shape with zero, an inferred
    # dimension and three, asserting the inferred dimension resolves to two.
    for use_gpu in (True, False):
      self._testZeroDimReshape(x=np.zeros([0, 6]).astype(np.float32),
                               shape=[0, -1, 3],
                               expected=(0, 2, 3),
                               use_gpu=use_gpu)

  @test_util.run_deprecated_v1
  def testErrors(self):
    # [FUNCTION DOC]
    # Graph-mode-only test. Asserts a value error saying the element count must be evenly
    # divisible by 17 when inferring a dimension for a 23 by 29 by 31 tensor, and a value error
    # about being unable to reshape a 4096-element tensor into 4095 elements.
    y = constant_op.constant(0.0, shape=[23, 29, 31])
    with self.assertRaisesRegex(ValueError, "must be evenly divisible by 17"):
      array_ops.reshape(y, [17, -1])

    z = constant_op.constant(0.0, shape=[32, 128])
    with self.assertRaisesRegex(ValueError,
                                "Cannot reshape a tensor with 4096 elements"):
      array_ops.reshape(z, [4095])

  def testPartialShapes(self):
    # [FUNCTION DOC]
    # In a graph, checks static shape inference with an unknown-shape placeholder input: partial
    # constant new shape, fully unknown new shape (unknown rank), new shape of known length only,
    # new shape assembled by stacking, concatenating or taking the shape of another placeholder.
    # Asserts the inferred shape lists, with unknown entries where information is missing.

    # Testing unknown shapes in graph building.
    with ops.Graph().as_default():
      x = array_ops.placeholder(dtypes.float32)

      # Unknown input shape, partial new shape.
      y = array_ops.reshape(x, [1, 1, -1, 1])
      self.assertEqual([1, 1, None, 1], y.get_shape().as_list())

      # Unknown input shape, unknown new shape.
      y = array_ops.reshape(x, array_ops.placeholder(dtypes.int32))
      self.assertEqual(None, y.get_shape().ndims)

      # Unknown input shape, known rank for new shape.
      y = array_ops.reshape(x, array_ops.placeholder(dtypes.int32, shape=(3,)))
      self.assertEqual([None, None, None], y.get_shape().as_list())

      # Unknown input shape, partial new shape using `tf.stack()`.
      y = array_ops.reshape(x, [array_ops.placeholder(dtypes.int32), 37])
      self.assertEqual([None, 37], y.get_shape().as_list())

      # Unknown input shape, partial new shape using `tf.concat()`.
      y = array_ops.reshape(
          x,
          array_ops.concat(
              [array_ops.placeholder(
                  dtypes.int32, shape=(2,)), [37, 42]], 0))
      self.assertEqual([None, None, 37, 42], y.get_shape().as_list())

      # Unknown input shape, partial new shape using `tf.shape()`.
      y = array_ops.reshape(
          x,
          array_ops.shape(
              array_ops.placeholder(
                  dtypes.float32, shape=[None, 37, None])))
      self.assertEqual([None, 37, None], y.get_shape().as_list())

  def testTensorShape(self):
    # [FUNCTION DOC]
    # Reshapes a one by 100 zeros tensor to one hundred by one, giving the target shape first as
    # a list of dimension objects and then as a shape object; asserts the static shape each time.
    x = array_ops.zeros([1, 100])
    y = array_ops.reshape(
        x, [tensor_shape.Dimension(100),
            tensor_shape.Dimension(1)])
    self.assertEqual([100, 1], y.get_shape().as_list())
    y = array_ops.reshape(x, tensor_shape.TensorShape([100, 1]))
    self.assertEqual([100, 1], y.get_shape().as_list())

  def testInt64Shape(self):
    # [FUNCTION DOC]
    # On CPU, builds a 50000 by 50000 boolean tensor and reshapes it to a vector of 2.5 billion
    # elements and to one by 2.5 billion, asserting static shapes are correct beyond the 32-bit
    # integer range. Only shape inference is checked; the result is not evaluated.
    with ops.device("/device:CPU:0"):
      x = array_ops.zeros([50000, 50000], dtype=dtypes.bool)
      # Provide dimension larger than int32
      y = array_ops.reshape(x, [50000**2])
      self.assertEqual([50000**2], y.get_shape().as_list())
      # Even if first dimension is within int32, ensure we correctly go to int64
      y = array_ops.reshape(x, [1, 50000**2])
      self.assertEqual([1, 50000**2], y.get_shape().as_list())

  @test_util.run_v2_only
  def testTooLargeShape(self):
    # [FUNCTION DOC]
    # Version-2-only test. Reshapes a single-element list to a five-dimensional shape whose
    # product is enormous and asserts evaluation raises an invalid-argument error mentioning too
    # many elements.
    with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                "too many elements"):
      x = array_ops.reshape([1], np.array([21943, 45817, 30516, 61760, 38987]))
      self.evaluate(x)


if __name__ == "__main__":
  test.main()
