# [MODULE DOC]
# Tests for the shape creation helpers in the xla_shape module (shapes derived from numpy arrays
# and tuples). Checks element types, dimensions and minor-to-major layouts for 1-D, 2-D and 3-D
# arrays in row-major and column-major order, and for tuples and nested tuples.
# Runs under absltest; no import-time side effects.
# Copyright 2023 The OpenXLA Authors.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ======================================
"""Tests for third_party.tensorflow.compiler.xla.python_api.xla_shape."""

from absl.testing import absltest
import numpy as np
from xla import xla_data_pb2
from xla.python_api import xla_shape


def NumpyArrayF32(*args, **kwargs):
  # [FUNCTION DOC]
  # HELPER: Test utility for building the 32-bit float numpy arrays used by the shape tests.
  # Forwards all arguments to numpy array creation with the element type forced to 32-bit float
  # and returns the new array.
  """Convenience wrapper to create Numpy arrays with a np.float32 dtype."""
  return np.array(*args, dtype=np.float32, **kwargs)


class CreateShapeFromNumpyTest(absltest.TestCase):
  # [CLASS DOC]
  # Test case for building shapes from numpy data. Provides assertion helpers for shapes, layouts
  # and nested tuple shapes; each test creates a shape and checks them. No shared fixtures.

  def assertShape(self, shape, expected_dimensions, expected_element_type):
    # [FUNCTION DOC]
    # HELPER: Custom assertion shared by the tests in this class.
    # Asserts the shape wrapper's element type and dimensions equal the expected values. Fails
    # the test on mismatch; returns nothing.
    self.assertEqual(shape.element_type(), expected_element_type)
    self.assertEqual(shape.dimensions(), expected_dimensions)

  def assertLayout(self, layout, expected_minor_to_major):
    # [FUNCTION DOC]
    # HELPER: Custom assertion shared by the tests in this class.
    # Asserts the layout message's minor-to-major ordering equals the expected list. Fails the
    # test on mismatch; returns nothing.
    self.assertEqual(layout.minor_to_major, expected_minor_to_major)

  def assertTupleShape(self, shape, expected):
    # [FUNCTION DOC]
    # HELPER: Custom assertion shared by the tuple tests in this class.
    # Asserts the shape is a tuple, then walks the wrapper sub-shapes, the sub-messages and the
    # expected specification together: checks each wrapper's element type equals its message's,
    # recurses into nested tuples, and otherwise checks dimensions and element type against the
    # expected pair.
    # KNOWN ISSUE: The parallel walk stops at the shortest sequence, so a wrong number of
    # sub-shapes is not detected.
    self.assertEqual(shape.element_type(), xla_data_pb2.TUPLE)
    for sub_shape, sub_message, sub_expected in zip(
        shape.tuple_shapes(),
        shape.message.tuple_shapes,
        expected):
      self.assertEqual(sub_shape.element_type(), sub_message.element_type)
      if sub_shape.is_tuple():
        self.assertTupleShape(sub_shape, sub_expected)
      else:
        expected_dimensions, expected_element_types = sub_expected
        self.assertShape(
            sub_shape, expected_dimensions, expected_element_types)

  def testCreateShapeFromNumpy1D(self):
    # [FUNCTION DOC]
    # Creates a shape from a two-element float32 vector; expects dimensions of two, float32
    # element type and layout of zero.
    shape = xla_shape.CreateShapeFromNumpy(NumpyArrayF32([1.1, 2.2]))
    self.assertShape(shape, [2], xla_data_pb2.F32)
    self.assertLayout(shape.layout(), [0])

  def testCreateShapeFromNumpy2DRowMajor(self):
    # [FUNCTION DOC]
    # Creates a shape from an explicitly row-major two-by-three float32 matrix; expects layout
    # one then zero.
    shape = xla_shape.CreateShapeFromNumpy(
        NumpyArrayF32(
            [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]], order='C'))
    self.assertShape(shape, [2, 3], xla_data_pb2.F32)
    self.assertLayout(shape.layout(), [1, 0])

  def testCreateShapeFromNumpy2DColumnMajor(self):
    # [FUNCTION DOC]
    # Creates a shape from a column-major two-by-three float32 matrix; expects layout zero then
    # one.
    shape = xla_shape.CreateShapeFromNumpy(
        NumpyArrayF32(
            [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]], order='F'))
    self.assertShape(shape, [2, 3], xla_data_pb2.F32)
    self.assertLayout(shape.layout(), [0, 1])

  def testCreateShapeFromNumpy2DDefaultIsRowMajor(self):
    # [FUNCTION DOC]
    # Creates a shape from a two-by-three float32 matrix with default memory order and expects
    # the row-major layout (one then zero), since numpy defaults to row-major.
    # The default layout in Numpy is C (row major)
    shape = xla_shape.CreateShapeFromNumpy(
        NumpyArrayF32([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]))
    self.assertShape(shape, [2, 3], xla_data_pb2.F32)
    self.assertLayout(shape.layout(), [1, 0])

  def testCreateShapeFromNumpy3DRowMajor(self):
    # [FUNCTION DOC]
    # Creates a shape from a row-major two-by-three-by-one float32 array; expects layout two,
    # one, zero.
    shape = xla_shape.CreateShapeFromNumpy(
        NumpyArrayF32(
            [[[1.1], [2.2], [3.3]], [[4.4], [5.5], [6.6]]], order='C'))
    self.assertShape(shape, [2, 3, 1], xla_data_pb2.F32)
    self.assertLayout(shape.layout(), [2, 1, 0])

  def testCreateShapeFromNumpy3DColumnMajor(self):
    # [FUNCTION DOC]
    # Creates a shape from a column-major two-by-three-by-one float32 array; expects layout zero,
    # one, two.
    shape = xla_shape.CreateShapeFromNumpy(
        NumpyArrayF32(
            [[[1.1], [2.2], [3.3]], [[4.4], [5.5], [6.6]]], order='F'))
    self.assertShape(shape, [2, 3, 1], xla_data_pb2.F32)
    self.assertLayout(shape.layout(), [0, 1, 2])

  def testCreateShapeFromTupleOfNumpy3D(self):
    # [FUNCTION DOC]
    # Creates a shape from a tuple of three identical 3-D float32 arrays and expects a tuple shape
    # whose three components each have the array's dimensions and float32 type.
    inner_array = NumpyArrayF32(
        [[[1.1], [2.2], [3.3]], [[4.4], [5.5], [6.6]]])
    inner_spec = ([2, 3, 1], xla_data_pb2.F32)

    shape = xla_shape.CreateShapeFromNumpy(
        (inner_array, inner_array, inner_array))
    self.assertTupleShape(
        shape,
        (inner_spec, inner_spec, inner_spec))

  def testCreateShapeFromNestedTupleOfNumpy3D(self):
    # [FUNCTION DOC]
    # Creates a shape from a tuple containing an array, an inner tuple of two arrays and another
    # array; expects the nested tuple structure with matching dimensions and float32 types.
    inner_array = NumpyArrayF32(
        [[[1.1], [2.2], [3.3]], [[4.4], [5.5], [6.6]]])
    inner_spec = ([2, 3, 1], xla_data_pb2.F32)

    shape = xla_shape.CreateShapeFromNumpy(
        (inner_array, (inner_array, inner_array), inner_array))
    self.assertTupleShape(
        shape,
        (inner_spec, (inner_spec, inner_spec), inner_spec))


if __name__ == '__main__':
  absltest.main()
