# [MODULE DOC]
# Tests for building XLA literals from Python through the xla extension. For each of a set of
# numeric, boolean and complex numpy dtypes (via a class-level parameterization), the tests
# create a literal with a given array shape and layout, obtain a numpy view of it, copy random
# data in, and check that the literal's contents match the source data. Defines one helper that
# generates random arrays. Run directly it starts the absltest runner.
# Copyright 2024 The OpenXLA Authors.
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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from xla.python import xla_extension


def create_random_array(shape, dtype: np.dtype) -> np.ndarray:
  # [FUNCTION DOC]
  # HELPER: generates random input data for the literal construction tests below.
  # KNOWN ISSUE: for complex dtypes the real part is cast to the requested dtype but the added
  # imaginary part is a 64-bit float product, so the result is promoted to the widest complex type
  # (complex128 even when complex64 was requested); the returned array does not have the requested
  # dtype, so the complex64 case effectively tests complex128.
  # Inputs: shape, a sequence of dimension sizes; dtype, a numpy dtype. Returns an array of
  # uniform random values in [0, 1) with that shape; for complex dtypes the imaginary part is also
  # random. For integer dtypes the values are truncated to zero, and for the bool type to true
  # for nonzero values (almost always true). Uses the global numpy random state; raises nothing
  # itself.
  if np.issubdtype(dtype, np.complexfloating):
    return np.random.rand(*shape).astype(dtype) + 1j * np.random.rand(*shape)
  else:
    return np.random.rand(*shape).astype(dtype)


@parameterized.parameters(
    np.float16,
    np.float32,
    np.float64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.bool,
    np.complex64,
    np.complex128,
)
class ConstructLiteralTest(parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case, instantiated once per dtype listed in the class decorator (float16,
  # float32, float64, signed and unsigned integers of all widths, bool, complex64 and complex128).
  # Each test method receives the dtype as an extra argument. Covers constructing an XLA literal
  # from an array shape (with optional minor-to-major layout) and viewing/writing it with numpy.

  def test_create_literal_from_ndarray_rank_1(self, dtype):
    # [FUNCTION DOC]
    # Test: rank-1 literal. Generates random data of length 10 for the dtype, builds an array
    # shape with the default layout, creates a literal, takes a numpy view of the literal and
    # copies the data into it. Asserts that a fresh numpy copy of the literal equals the input,
    # proving the view wrote to the literal's own storage. Parameter dtype comes from the
    # class-level parameterization; returns nothing.
    input_array = create_random_array([10], dtype)
    shape = xla_extension.Shape.array_shape(
        input_array.dtype, input_array.shape
    )
    literal = xla_extension.Literal(shape)
    # use `np.asarray` to ensure that the array is a view into the literal.
    array = np.asarray(literal)
    np.copyto(array, input_array)
    # Intentionally check against `np.array(literal)` instead of `array`
    # to ensure that the underlying literal is actually updated and not some
    # rebinding to a new object. (This also exersises the copy functionality)
    np.testing.assert_array_equal(np.array(literal), input_array)

  def test_create_literal_from_ndarray_rank_2(self, dtype):
    # [FUNCTION DOC]
    # Test: rank-2 literal with an explicit layout. Random data of shape 20 by 5, shape with
    # layout order 1,0 (row-major), literal created, data copied through a numpy view, and
    # asserts that a numpy copy of the literal equals the input. dtype is the parameterized dtype.
    input_array = create_random_array([20, 5], dtype)
    shape = xla_extension.Shape.array_shape(
        input_array.dtype, input_array.shape, [1, 0]
    )
    literal = xla_extension.Literal(shape)
    array = np.asarray(literal)
    np.copyto(array, input_array)
    np.testing.assert_array_equal(np.array(literal), input_array)

  def test_create_literal_from_ndarray_rank_2_reverse_layout(self, dtype):
    # [FUNCTION DOC]
    # Test: rank-2 literal with the reversed layout order 0,1 (column-major). Random data of shape
    # 25 by 4, literal created from that shape and layout, data copied via a numpy view, and
    # asserts a numpy copy of the literal equals the input. dtype is the parameterized dtype.
    input_array = create_random_array([25, 4], dtype)
    shape = xla_extension.Shape.array_shape(
        input_array.dtype, input_array.shape, [0, 1]
    )
    literal = xla_extension.Literal(shape)
    array = np.asarray(literal)
    np.copyto(array, input_array)
    np.testing.assert_array_equal(np.array(literal), input_array)


if __name__ == "__main__":
  absltest.main()
