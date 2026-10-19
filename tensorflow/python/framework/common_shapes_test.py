# [MODULE DOC]
# Unit tests for the broadcast shape helpers in the common shapes module. Checks that computing
# the broadcast of two tensor shapes (fully known, partially unknown, or entirely unknown rank)
# gives the expected shape, is symmetric in its arguments, agrees with numpy for fully known
# shapes, and that incompatible shapes are reported as not broadcast-compatible and raise a
# value error. Has no import-time side effects; runs the test main when executed directly.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for common shapes."""
import numpy as np

from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


class CommonShapesTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case covering broadcast shape computation and broadcast-compatibility checks for
  # tensor shapes. Holds no shared fixtures; private assertion helpers encapsulate the
  # symmetric-argument and numpy-agreement checks used by the individual tests.

  # Asserts that we get the same result with numpy (for known shapes), and that
  # the order of arguments does not matter (i.e., broadcasting is reflexive).
  def _assert_incompatible_broadcast(self, shape1, shape2):
    # [FUNCTION DOC]
    # HELPER: assertion helper for the broadcast tests in this test class.
    # Asserts that two tensor shapes cannot be broadcast together, in both argument orders.
    # Inputs: shape1 and shape2, tensor shape objects. When both have known rank, also checks
    # that numpy's broadcast raises a value error for the equivalent zero-filled arrays.
    # Then asserts the compatibility predicate returns false and that computing the broadcast
    # shape raises a value error, for both orderings. Returns nothing; fails the test otherwise.
    if shape1.dims is not None and shape2.dims is not None:
      zeros1 = np.zeros(shape1.as_list())
      zeros2 = np.zeros(shape2.as_list())
      with self.assertRaises(ValueError):
        np.broadcast(zeros1, zeros2)
      with self.assertRaises(ValueError):
        np.broadcast(zeros2, zeros1)
    self.assertFalse(common_shapes.is_broadcast_compatible(shape1, shape2))
    self.assertFalse(common_shapes.is_broadcast_compatible(shape2, shape1))
    with self.assertRaises(ValueError):
      common_shapes.broadcast_shape(shape1, shape2)
    with self.assertRaises(ValueError):
      common_shapes.broadcast_shape(shape2, shape1)

  # Asserts that we get the same result with numpy (for known shapes), and that
  # the order of arguments does not matter (i.e., broadcasting is reflexive).
  def _assert_broadcast(self, expected, shape1, shape2):
    # [FUNCTION DOC]
    # HELPER: assertion helper for the broadcast tests in this test class.
    # Asserts that broadcasting shape1 with shape2 yields the expected shape, regardless of
    # argument order. Inputs: expected, the expected resulting tensor shape; shape1 and shape2,
    # the tensor shapes to broadcast. If both have known rank, the result is also compared with
    # numpy's broadcast of zero-filled arrays and the library result in both orders; otherwise
    # only the library result is compared in both orders. Returns nothing.
    if shape1.dims is not None and shape2.dims is not None:
      expected_np = expected.as_list()
      zeros1 = np.zeros(shape1.as_list())
      zeros2 = np.zeros(shape2.as_list())
      self.assertAllEqual(expected_np, np.broadcast(zeros1, zeros2).shape)
      self.assertAllEqual(expected_np, np.broadcast(zeros2, zeros1).shape)
      self.assertEqual(
          expected, common_shapes.broadcast_shape(shape1, shape2))
      self.assertEqual(
          expected, common_shapes.broadcast_shape(shape2, shape1))
    else:
      self.assertEqual(expected, common_shapes.broadcast_shape(shape1, shape2))
      self.assertEqual(expected, common_shapes.broadcast_shape(shape2, shape1))

  def testBroadcast_one_dimension(self):
    # [FUNCTION DOC]
    # Test of broadcasting with rank-one shapes. Builds two different length-one-dimension
    # shapes, an unknown-rank shape, a scalar shape and a single-element shape. Asserts that
    # identical shapes broadcast to themselves, that the scalar and single-element shapes act
    # as identity, that anything with an unknown shape gives unknown, that scalar with
    # single-element gives single-element, and that two different non-one lengths are incompatible.
    s1 = tensor_shape.TensorShape([5])
    s2 = tensor_shape.TensorShape([7])

    unknown = tensor_shape.unknown_shape()
    scalar = tensor_shape.TensorShape([])
    expanded_scalar = tensor_shape.TensorShape([1])

    # Tensors with same shape should have the same broadcast result.
    for shape in (s1, s2, unknown, scalar, expanded_scalar):
      self._assert_broadcast(expected=shape, shape1=shape, shape2=shape)

    # [] and [1] act like identity.
    self._assert_broadcast(expected=s1, shape1=s1, shape2=scalar)
    self._assert_broadcast(expected=s2, shape1=s2, shape2=scalar)
    self._assert_broadcast(expected=s1, shape1=s1, shape2=expanded_scalar)
    self._assert_broadcast(expected=s2, shape1=s2, shape2=expanded_scalar)

    self._assert_broadcast(expected=unknown, shape1=s1, shape2=unknown)
    self._assert_broadcast(expected=unknown, shape1=s2, shape2=unknown)

    self._assert_broadcast(
        expected=expanded_scalar, shape1=scalar, shape2=expanded_scalar)

    self._assert_incompatible_broadcast(shape1=s1, shape2=s2)

  def testBroadcast_many_dimensions(self):
    # [FUNCTION DOC]
    # Test of broadcasting with fully known shapes of rank zero, one and two (scalar, one,
    # four, one-by-four, four-by-one, three-by-four, four-by-three). Asserts that equal shapes
    # broadcast to themselves, scalar and single-element shapes act as identity, unknown rank
    # gives unknown, several mixed pairs give the expected expanded shape, and that mismatched
    # pairs (for example four with four-by-three, or three-by-four with four-by-three) are
    # incompatible.
    unknown = tensor_shape.unknown_shape()
    shape_0 = tensor_shape.TensorShape([])
    shape_1 = tensor_shape.TensorShape([1])
    shape_4 = tensor_shape.TensorShape([4])
    shape_1x4 = tensor_shape.TensorShape([1, 4])
    shape_4x1 = tensor_shape.TensorShape([4, 1])
    shape_3x4 = tensor_shape.TensorShape([3, 4])
    shape_4x3 = tensor_shape.TensorShape([4, 3])

    # Tensors with same shape should have the same broadcast result.
    for shape in (
        shape_0, shape_1, shape_4, shape_1x4, shape_4x1, shape_3x4, shape_4x3):
      self._assert_broadcast(expected=shape, shape1=shape, shape2=shape)

    # [] and [1] act like identity.
    for identity in (shape_0, shape_1):
      for shape in (shape_4, shape_1x4, shape_4x1, shape_3x4, shape_4x3):
        self._assert_broadcast(expected=shape, shape1=identity, shape2=shape)

    # Unknown in, unknown out.
    for shape in (shape_4, shape_1x4, shape_4x1, shape_3x4, shape_4x3):
      self._assert_broadcast(expected=unknown, shape1=shape, shape2=unknown)

    self._assert_broadcast(expected=shape_1x4, shape1=shape_4, shape2=shape_1x4)
    shape_4x4 = tensor_shape.TensorShape([4, 4])
    self._assert_broadcast(expected=shape_4x4, shape1=shape_4, shape2=shape_4x1)
    self._assert_broadcast(expected=shape_3x4, shape1=shape_4, shape2=shape_3x4)
    self._assert_incompatible_broadcast(shape1=shape_4, shape2=shape_4x3)
    self._assert_broadcast(
        expected=shape_4x4, shape1=shape_1x4, shape2=shape_4x1)
    self._assert_broadcast(
        expected=shape_3x4, shape1=shape_1x4, shape2=shape_3x4)
    self._assert_incompatible_broadcast(shape1=shape_1x4, shape2=shape_4x3)
    self._assert_incompatible_broadcast(shape1=shape_4x1, shape2=shape_3x4)
    self._assert_broadcast(
        expected=shape_4x3, shape1=shape_4x1, shape2=shape_4x3)
    self._assert_incompatible_broadcast(shape1=shape_3x4, shape2=shape_4x3)

  # Asserts that the order of arguments does not matter (i.e., broadcasting is
  # reflexive).
  def _assert_broadcast_with_unknown_dims(self, expected, shape1, shape2):
    # [FUNCTION DOC]
    # HELPER: assertion helper for the unknown-dimension broadcast test.
    # Compares dimension values rather than shape equality, since unknown dimensions do not
    # compare equal. Inputs: expected, the expected tensor shape; shape1 and shape2, shapes to
    # broadcast. Asserts the broadcast result has the same rank and per-dimension values in
    # both argument orders (both unknown rank, or equal length with matching values), and that
    # it matches the expected shape the same way. Returns nothing.
    actual_dims = common_shapes.broadcast_shape(shape1, shape2).dims
    reflexive_actual_dims = common_shapes.broadcast_shape(shape2, shape1).dims

    if actual_dims is None:
      self.assertIsNone(reflexive_actual_dims)
    elif reflexive_actual_dims is None:
      self.assertIsNone(actual_dims)
    else:
      self.assertEqual(len(actual_dims), len(reflexive_actual_dims))
      for actual_dim, reflexive_actual_dim in zip(
          actual_dims, reflexive_actual_dims):
        self.assertEqual(actual_dim.value, reflexive_actual_dim.value)

    expected_dims = expected.dims
    if expected_dims is None:
      self.assertIsNone(actual_dims)
    elif actual_dims is None:
      self.assertIsNone(expected_dims)
    else:
      self.assertEqual(len(expected_dims), len(actual_dims))
      for expected_dim, actual_dim in zip(expected_dims, actual_dims):
        self.assertEqual(expected_dim.value, actual_dim.value)

  def testBroadcast_unknown_dims(self):
    # [FUNCTION DOC]
    # Test of broadcasting with shapes that have unknown dimension sizes (for example a single
    # unknown dimension, or one-by-unknown, unknown-by-one, four-by-unknown, unknown-by-four).
    # Asserts via the unknown-dimension helper that equal shapes broadcast to themselves, scalar
    # and single-element shapes act as identity, unknown rank gives unknown, and mixed pairs
    # produce the expected partially known shapes (for example four-by-unknown with
    # unknown-by-four gives four-by-four).
    unknown = tensor_shape.unknown_shape()
    shape_0 = tensor_shape.TensorShape([])
    shape_1 = tensor_shape.TensorShape([1])
    # pylint: disable=invalid-name
    shape_U = tensor_shape.TensorShape([None])
    shape_1xU = tensor_shape.TensorShape([1, None])
    shape_Ux1 = tensor_shape.TensorShape([None, 1])
    shape_4xU = tensor_shape.TensorShape([4, None])
    shape_Ux4 = tensor_shape.TensorShape([None, 4])
    # pylint: enable=invalid-name

    # Tensors with same shape should have the same broadcast result.
    for shape in (shape_U, shape_1xU, shape_Ux1, shape_4xU, shape_Ux4):
      self._assert_broadcast_with_unknown_dims(
          expected=shape, shape1=shape, shape2=shape)

    # [] and [1] act like identity.
    for identity in (shape_0, shape_1):
      for shape in (shape_U, shape_1xU, shape_Ux1, shape_4xU, shape_Ux4):
        self._assert_broadcast_with_unknown_dims(
            expected=shape, shape1=identity, shape2=shape)

    # Unknown in, unknown out.
    for shape in (shape_U, shape_1xU, shape_Ux1, shape_4xU, shape_Ux4):
      self._assert_broadcast_with_unknown_dims(
          expected=unknown, shape1=shape, shape2=unknown)

    self._assert_broadcast_with_unknown_dims(
        expected=shape_1xU, shape1=shape_U, shape2=shape_1xU)
    shape_UxU = tensor_shape.TensorShape([None, None])  # pylint: disable=invalid-name
    self._assert_broadcast_with_unknown_dims(
        expected=shape_UxU, shape1=shape_U, shape2=shape_Ux1)
    self._assert_broadcast_with_unknown_dims(
        expected=shape_4xU, shape1=shape_U, shape2=shape_4xU)
    self._assert_broadcast_with_unknown_dims(
        expected=shape_Ux4, shape1=shape_U, shape2=shape_Ux4)
    self._assert_broadcast_with_unknown_dims(
        expected=shape_UxU, shape1=shape_1xU, shape2=shape_Ux1)
    self._assert_broadcast_with_unknown_dims(
        expected=shape_4xU, shape1=shape_1xU, shape2=shape_4xU)
    self._assert_broadcast_with_unknown_dims(
        expected=shape_Ux4, shape1=shape_1xU, shape2=shape_Ux4)
    self._assert_broadcast_with_unknown_dims(
        expected=shape_4xU, shape1=shape_Ux1, shape2=shape_4xU)
    self._assert_broadcast_with_unknown_dims(
        expected=shape_Ux4, shape1=shape_Ux1, shape2=shape_Ux4)
    shape_4x4 = tensor_shape.TensorShape([4, 4])
    self._assert_broadcast_with_unknown_dims(
        expected=shape_4x4, shape1=shape_4xU, shape2=shape_Ux4)


if __name__ == "__main__":
  googletest.main()
