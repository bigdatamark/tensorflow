# [MODULE DOC]
# Functional tests for the tensor_util module: converting Python values and numpy arrays to tensor
# protocol buffers (make_tensor_proto) and back (MakeNdarray), shape comparison against a proto,
# the is-tensor-type predicate, constant value extraction from graph tensors (constant_value) and
# partial shape extraction (constant_value_as_shape), the maybe-set-static-shape helper and
# conversion of shapes to tensors. Contains five test classes. Many tests branch on host byte
# order so the expected raw tensor content matches. No import-time side effects beyond imports;
# run through the standard test main entry point.
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
"""Functional tests for tensor_util."""

import contextlib
import sys

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import shape_util
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class TensorUtilTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for tensor_util proto/array conversion, run in both graph and eager modes by the class
  # decorator and also parameterized by dtype for several integer tests. Each test builds a proto
  # from a Python or numpy value, compares it to expected text-format proto, then converts back to
  # an array and checks dtype and values. No shared fixtures.

  def testFloat(self):
    # [FUNCTION DOC]
    # Converts the Python float ten to a proto with no dtype given. Expects a float32 scalar proto
    # with empty shape and a single float value, and that converting back yields a float32 array
    # close to the original.
    value = 10.0
    t = tensor_util.make_tensor_proto(value)
    self.assertProtoEquals("""
      dtype: DT_FLOAT
      tensor_shape {}
      float_val: %.1f
      """ % value, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(np.float32, a.dtype)
    self.assertAllClose(np.array(value, dtype=np.float32), a)

  def testFloatN(self):
    # [FUNCTION DOC]
    # Converts a list of three floats. Expects a float32 proto of shape three whose raw tensor
    # content is the packed bytes (byte order chosen from the host endianness), and that the round
    # trip array is float32 with the same values.
    t = tensor_util.make_tensor_proto([10.0, 20.0, 30.0])
    if sys.byteorder == "big":
      self.assertProtoEquals(r"""
        dtype: DT_FLOAT
        tensor_shape { dim { size: 3 } }
        tensor_content: "A \000\000A\240\000\000A\360\000\000"
        """, t)
    else:
      self.assertProtoEquals(r"""
        dtype: DT_FLOAT
        tensor_shape { dim { size: 3 } }
        tensor_content: "\000\000 A\000\000\240A\000\000\360A"
        """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(np.float32, a.dtype)
    self.assertAllClose(np.array([10.0, 20.0, 30.0], dtype=np.float32), a)

  def testFloatTyped(self):
    # [FUNCTION DOC]
    # Same as the list-of-floats case but with an explicit float32 dtype. Expects identical packed
    # content (chosen by host endianness) and a float32 array on the round trip.
    t = tensor_util.make_tensor_proto([10.0, 20.0, 30.0], dtype=dtypes.float32)
    if sys.byteorder == "big":
      self.assertProtoEquals(r"""
        dtype: DT_FLOAT
        tensor_shape { dim { size: 3 } }
        tensor_content: "A \000\000A\240\000\000A\360\000\000"
        """, t)
    else:
      self.assertProtoEquals(r"""
        dtype: DT_FLOAT
        tensor_shape { dim { size: 3 } }
        tensor_content: "\000\000 A\000\000\240A\000\000\360A"
        """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(np.float32, a.dtype)
    self.assertAllClose(np.array([10.0, 20.0, 30.0], dtype=np.float32), a)

  def testFloatTypeCoerce(self):
    # [FUNCTION DOC]
    # Converts a list of integers while requesting float32. Expects the integers to be coerced, so
    # the proto has float32 packed content and the round trip array is float32 with the values.
    t = tensor_util.make_tensor_proto([10, 20, 30], dtype=dtypes.float32)
    if sys.byteorder == "big":
      self.assertProtoEquals(r"""
        dtype: DT_FLOAT
        tensor_shape { dim { size: 3 } }
        tensor_content: "A \000\000A\240\000\000A\360\000\000"
        """, t)
    else:
      self.assertProtoEquals(r"""
        dtype: DT_FLOAT
        tensor_shape { dim { size: 3 } }
        tensor_content: "\000\000 A\000\000\240A\000\000\360A"
        """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(np.float32, a.dtype)
    self.assertAllClose(np.array([10.0, 20.0, 30.0], dtype=np.float32), a)

  def testFloatTypeCoerceNdarray(self):
    # [FUNCTION DOC]
    # Converts an integer numpy array while requesting float32. Expects coercion to float32 packed
    # content and a float32 array with the same values after the round trip.
    arr = np.asarray([10, 20, 30], dtype="int")
    t = tensor_util.make_tensor_proto(arr, dtype=dtypes.float32)
    if sys.byteorder == "big":
      self.assertProtoEquals(r"""
        dtype: DT_FLOAT
        tensor_shape { dim { size: 3 } }
        tensor_content: "A \000\000A\240\000\000A\360\000\000"
        """, t)
    else:
      self.assertProtoEquals(r"""
        dtype: DT_FLOAT
        tensor_shape { dim { size: 3 } }
        tensor_content: "\000\000 A\000\000\240A\000\000\360A"
        """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(np.float32, a.dtype)
    self.assertAllClose(np.array([10.0, 20.0, 30.0], dtype=np.float32), a)

  def testFloatSizes(self):
    # [FUNCTION DOC]
    # Converts three floats with an explicit shape of one by three. Expects the proto shape to have
    # two dimensions (one and three) with packed float content and the array to be one row of three.
    t = tensor_util.make_tensor_proto([10.0, 20.0, 30.0], shape=[1, 3])
    if sys.byteorder == "big":
      self.assertProtoEquals(r"""
        dtype: DT_FLOAT
        tensor_shape { dim { size: 1 } dim { size: 3 } }
        tensor_content: "A \000\000A\240\000\000A\360\000\000"
        """, t)
    else:
      self.assertProtoEquals(r"""
        dtype: DT_FLOAT
        tensor_shape { dim { size: 1 } dim { size: 3 } }
        tensor_content: "\000\000 A\000\000\240A\000\000\360A"
        """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(np.float32, a.dtype)
    self.assertAllClose(np.array([[10.0, 20.0, 30.0]], dtype=np.float32), a)

  def testFloatSizes2(self):
    # [FUNCTION DOC]
    # Converts three floats with an explicit shape of three by one. Expects a two-dimensional proto
    # shape of three and one, packed content, and a column-shaped float32 array on the round trip.
    t = tensor_util.make_tensor_proto([10.0, 20.0, 30.0], shape=[3, 1])
    if sys.byteorder == "big":
      self.assertProtoEquals(r"""
        dtype: DT_FLOAT
        tensor_shape { dim { size: 3 } dim { size: 1 } }
        tensor_content: "A \000\000A\240\000\000A\360\000\000"
        """, t)
    else:
      self.assertProtoEquals(r"""
        dtype: DT_FLOAT
        tensor_shape { dim { size: 3 } dim { size: 1 } }
        tensor_content: "\000\000 A\000\000\240A\000\000\360A"
        """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(np.float32, a.dtype)
    self.assertAllClose(np.array([[10.0], [20.0], [30.0]], dtype=np.float32), a)

  def testFloatSizesLessValues(self):
    # [FUNCTION DOC]
    # Converts a single float with a target shape of one by three. Expects the proto to keep the
    # target shape and store one float value (to be repeated implicitly). It does not convert back
    # to an array because there are too few values.
    t = tensor_util.make_tensor_proto(10.0, shape=[1, 3])
    self.assertProtoEquals("""
      dtype: DT_FLOAT
      tensor_shape { dim { size: 1 } dim { size: 3 } }
      float_val: 10.0
      """, t)
    # No conversion to Ndarray for this one: not enough values.

  def testFloatNpArrayFloat64(self):
    # [FUNCTION DOC]
    # Converts a one by three float64 numpy array. Expects a double-typed proto with shape one by
    # three and packed eight-byte content (host-endianness dependent), and a float64 array of the
    # same values when converted back.
    t = tensor_util.make_tensor_proto(
        np.array([[10.0, 20.0, 30.0]], dtype=np.float64))
    if sys.byteorder == "big":
      self.assertProtoEquals(r"""
        dtype: DT_DOUBLE
        tensor_shape { dim { size: 1 } dim { size: 3 } }
        tensor_content: "@$\000\000\000\000\000\000@4\000\000\000\000\000\000@>\000\000\000\000\000\000"
        """, t)
    else:
      self.assertProtoEquals(r"""
        dtype: DT_DOUBLE
        tensor_shape { dim { size: 1 } dim { size: 3 } }
        tensor_content: "\000\000\000\000\000\000$@\000\000\000\000\000\0004@\000\000\000\000\000\000>@"
        """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(np.float64, a.dtype)
    self.assertAllClose(
        np.array([[10.0, 20.0, 30.0]], dtype=np.float64),
        tensor_util.MakeNdarray(t))

  def testFloatTypesWithImplicitRepeat(self):
    # [FUNCTION DOC]
    # For both float32 and float64, converts a one-element list with a three by four target shape.
    # Expects the single value to be repeated to fill the whole array when converted back.
    for dtype, nptype in [(dtypes.float32, np.float32),
                          (dtypes.float64, np.float64)]:
      t = tensor_util.make_tensor_proto([10.0], shape=[3, 4], dtype=dtype)
      a = tensor_util.MakeNdarray(t)
      self.assertAllClose(
          np.array(
              [[10.0, 10.0, 10.0, 10.0],
               [10.0, 10.0, 10.0, 10.0],
               [10.0, 10.0, 10.0, 10.0]],
              dtype=nptype),
          a)

  def testFloatMutateArray(self):
    # [FUNCTION DOC]
    # Converts floats to a proto, converts back to an array, then modifies the first element of the
    # array. Expects the array to reflect the change while the original proto content stays
    # unchanged, showing the returned array does not alias the proto data (a copy is returned).
    t = tensor_util.make_tensor_proto([10.0, 20.0, 30.0], dtype=dtypes.float32)
    a = tensor_util.MakeNdarray(t)
    a[0] = 5.0
    self.assertEqual(np.float32, a.dtype)
    self.assertAllClose(np.array([5.0, 20.0, 30.0], dtype=np.float32), a)
    if sys.byteorder == "big":
      self.assertProtoEquals(r"""
        dtype: DT_FLOAT
        tensor_shape { dim { size: 3 } }
        tensor_content: "A \000\000A\240\000\000A\360\000\000"
        """, t)
    else:
      self.assertProtoEquals(r"""
        dtype: DT_FLOAT
        tensor_shape { dim { size: 3 } }
        tensor_content: "\000\000 A\000\000\240A\000\000\360A"
        """, t)

  def testHalf(self):
    # [FUNCTION DOC]
    # Converts a float16 numpy array of two values. Expects a half-precision proto with packed
    # two-byte content (host-endianness dependent) and a float16 array when converted back.
    t = tensor_util.make_tensor_proto(np.array([10.0, 20.0], dtype=np.float16))
    if sys.byteorder == "big":
      self.assertProtoEquals(
          """
        dtype: DT_HALF
        tensor_shape { dim { size: 2 } }
        tensor_content: "I\000M\000"
        """,
          t,
      )
    else:
      self.assertProtoEquals(
          """
        dtype: DT_HALF
        tensor_shape { dim { size: 2 } }
        tensor_content: "\000I\000M"
        """,
          t,
      )

    a = tensor_util.MakeNdarray(t)
    self.assertEqual(np.float16, a.dtype)
    self.assertAllClose(np.array([10.0, 20.0], dtype=np.float16), a)

  def testBfloat16(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the expected content strings use escaped hexadecimal text that is passed into a
    # text-format comparison, which is unusual and hard to audit compared with the other tests.
    # Converts a bfloat16 numpy array of two values. Expects a bfloat16 proto of shape two with the
    # expected packed content for the host byte order and a round trip array of the same type.
    test_type = dtypes.bfloat16.as_numpy_dtype
    t = tensor_util.make_tensor_proto(np.array([10.0, 20.0], dtype=test_type))
    if sys.byteorder == "big":
      self.assertProtoEquals(
          """
        dtype: DT_BFLOAT16
        tensor_shape {
          dim {
            size: 2
          }
      }
      tensor_content: "\x41\x20\x41\x5C\x32\x34\x30"
      """,
          t,
      )
    else:
      self.assertProtoEquals(
          """
        dtype: DT_BFLOAT16
        tensor_shape {
          dim {
            size: 2
          }
        }
      tensor_content: "\x20\x41\x5C\x32\x34\x30\x41"
      """,
          t,
      )

    a = tensor_util.MakeNdarray(t)
    self.assertEqual(test_type, a.dtype)
    self.assertAllClose(np.array([10.0, 20.0], dtype=test_type), a)

  def testFloat8e5m2(self):
    # [FUNCTION DOC]
    # Converts a float8 (five exponent bits, two mantissa bits) array of two values. Expects the
    # proto to hold one raw byte per element matching the documented bit patterns and the round
    # trip array to have the same dtype and values.
    test_type = dtypes.float8_e5m2.as_numpy_dtype
    t = tensor_util.make_tensor_proto(np.array([10.0, 20.0], dtype=test_type))
    # 10.0: "I" = 73 = 10010 01: 2^(18 - 15) * (1 + 1/4)
    # 20.0: "M" = 77 = 10011 01: 2^(19 - 15) * (1 + 1/4)
    self.assertProtoEquals(
        """
      dtype: DT_FLOAT8_E5M2
      tensor_shape {
        dim {
          size: 2
        }
      }
      tensor_content: "IM"
      """, t)

    a = tensor_util.MakeNdarray(t)
    self.assertEqual(test_type, a.dtype)
    self.assertAllClose(np.array([10.0, 20.0], dtype=test_type), a)

  def testFloat8e4m3fn(self):
    # [FUNCTION DOC]
    # Converts a float8 (four exponent bits, three mantissa bits, finite only) array of two values.
    # Expects one raw byte per element in the proto. Does not convert back to an array, unlike the
    # similar tests for other eight-bit float types.
    test_type = dtypes.float8_e4m3fn.as_numpy_dtype
    t = tensor_util.make_tensor_proto(np.array([10.0, 20.0], dtype=test_type))
    # 10.0: "R" = 82 = 1010 010: 2^(10 - 7) * (1 + 1/4)
    # 20.0: "Z" = 90 = 1011 010: 2^(11 - 7) * (1 + 1/4)
    self.assertProtoEquals(
        """
      dtype: DT_FLOAT8_E4M3FN
      tensor_shape {
        dim {
          size: 2
        }
      }
      tensor_content: "RZ"
      """, t)

  def testFloat8e4m3fnuz(self):
    # [FUNCTION DOC]
    # Converts a float8 e4m3 variant with no negative zero or infinity (fnuz) array of two values.
    # Expects the proto to hold the shifted raw bytes (bias differs from the plain variant). Does
    # not convert back to an array.
    test_type = dtypes.float8_e4m3fnuz.as_numpy_dtype
    t = tensor_util.make_tensor_proto(np.array([10.0, 20.0], dtype=test_type))
    # 10.0: "Z" = 90 = 1010 010: 2^(10 - 7) * (1 + 1/4) + 8
    # 20.0: "b" = 98 = 1011 010: 2^(11 - 7) * (1 + 1/4) + 8
    self.assertProtoEquals(
        """
      dtype: DT_FLOAT8_E4M3FNUZ
      tensor_shape {
        dim {
          size: 2
        }
      }
      tensor_content: "Zb"
      """,
        t,
    )

  def testFloat8e4m3b11fnuz(self):
    # [FUNCTION DOC]
    # Converts a float8 e4m3 variant with exponent bias eleven (b11fnuz) array of two values.
    # Expects the proto to hold the matching raw bytes for ten and twenty. Does not convert back.
    test_type = dtypes.float8_e4m3b11fnuz.as_numpy_dtype
    t = tensor_util.make_tensor_proto(np.array([10.0, 20.0], dtype=test_type))
    # 10.0: "r" = 114 = 1010 010: 2^(10 - 7) * (1 + 1/4) + 36
    # 20.0: "z" = 126 = 1011 010: 2^(11 - 7) * (1 + 1/4) + 36
    self.assertProtoEquals(
        """
      dtype: DT_FLOAT8_E4M3B11FNUZ
      tensor_shape {
        dim {
          size: 2
        }
      }
      tensor_content: "rz"
      """,
        t,
    )

  def testFloat8e5m2fnuz(self):
    # [FUNCTION DOC]
    # Converts a float8 e5m2 fnuz variant array of two values. Expects the proto to hold the
    # matching raw bytes for ten and twenty. Does not convert back to an array.
    test_type = dtypes.float8_e5m2fnuz.as_numpy_dtype
    t = tensor_util.make_tensor_proto(np.array([10.0, 20.0], dtype=test_type))
    # 10.0: "M" = 77 = 1010 010: 2^(10 - 7) * (1 + 1/4) - 3
    # 20.0: "Q" = 87 = 1011 010: 2^(11 - 7) * (1 + 1/4) - 3
    self.assertProtoEquals(
        """
      dtype: DT_FLOAT8_E5M2FNUZ
      tensor_shape {
        dim {
          size: 2
        }
      }
      tensor_content: "MQ"
      """,
        t,
    )

  def testFloat4e2m1fn(self):
    # [FUNCTION DOC]
    # Converts a four-bit float (two exponent bits, one mantissa bit) array holding six and one
    # half. Expects the proto content to be one byte per element with the encoded values, and the
    # round trip array to have the same dtype and values.
    test_type = dtypes.float4_e2m1fn.as_numpy_dtype
    t = tensor_util.make_tensor_proto(np.array([6, 0.5], dtype=test_type))
    # 0x7 = 011 1 = 2^(3-1) x (1+0.5) = 6
    # 0x1 = 000 1 = 2^(0) x 0.5 = 0.5
    expected_bytes = r"\x07\x01"
    self.assertProtoEquals(
        f"""
      dtype: DT_FLOAT4_E2M1FN
      tensor_shape {{
        dim {{
          size: 2
        }}
      }}
      tensor_content: "{expected_bytes}"
      """,
        t,
    )

    a = tensor_util.MakeNdarray(t)
    self.assertEqual(test_type, a.dtype)
    self.assertAllClose(np.array([6, 0.5], dtype=test_type), a)

  def testInt(self):
    # [FUNCTION DOC]
    # Converts the Python integer ten. Expects an int32 scalar proto with one integer value and a
    # round trip int32 array equal to the original.
    t = tensor_util.make_tensor_proto(10)
    self.assertProtoEquals(
        """
      dtype: DT_INT32
      tensor_shape {}
      int_val: 10
      """,
        t,
    )
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(np.int32, a.dtype)
    self.assertAllClose(np.array(10, dtype=np.int32), a)

  def testInt4(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: contains a stray empty comment line before the assertion and omits the array
    # round trip check done by most other tests.
    # Converts a signed four-bit integer array spanning minus eight to seven. Expects an int4 proto
    # of shape five that stores each element as an integer value entry. No round trip is checked.
    test_type = dtypes.int4.as_numpy_dtype
    t = tensor_util.make_tensor_proto(
        np.array(
            [-8, -1, 0, 1, 7],
            dtype=test_type,
        )
    )
    #
    self.assertProtoEquals(
        """
      dtype: DT_INT4
      tensor_shape {
        dim {
          size: 5
        }
      }
      int_val: -8
      int_val: -1
      int_val: 0
      int_val: 1
      int_val: 7
      """,
        t,
    )

  def testUInt4(self):
    # [FUNCTION DOC]
    # Converts an unsigned four-bit integer array from zero to fifteen. Expects a uint4 proto of
    # shape five that stores each element as an integer value entry. No round trip is checked.
    test_type = dtypes.uint4.as_numpy_dtype
    t = tensor_util.make_tensor_proto(
        np.array(
            [0, 1, 7, 8, 15],
            dtype=test_type,
        )
    )
    self.assertProtoEquals(
        """
      dtype: DT_UINT4
      tensor_shape {
        dim {
          size: 5
        }
      }
      int_val: 0
      int_val: 1
      int_val: 7
      int_val: 8
      int_val: 15
      """,
        t,
    )

  def testInt2(self):
    # [FUNCTION DOC]
    # Converts a signed two-bit integer array covering minus two to one. Expects an int2 proto of
    # shape four with each element as an integer value entry. No round trip is checked.
    test_type = dtypes.int2.as_numpy_dtype
    t = tensor_util.make_tensor_proto(
        np.array(
            [-2, -1, 0, 1],
            dtype=test_type,
        )
    )
    self.assertProtoEquals(
        """
      dtype: DT_INT2
      tensor_shape {
        dim {
          size: 4
        }
      }
      int_val: -2
      int_val: -1
      int_val: 0
      int_val: 1
      """,
        t,
    )

  def testUInt2(self):
    # [FUNCTION DOC]
    # Converts an unsigned two-bit integer array covering zero to three. Expects a uint2 proto of
    # shape four with each element as an integer value entry. No round trip is checked.
    test_type = dtypes.uint2.as_numpy_dtype
    t = tensor_util.make_tensor_proto(
        np.array(
            [0, 1, 2, 3],
            dtype=test_type,
        )
    )
    self.assertProtoEquals(
        """
      dtype: DT_UINT2
      tensor_shape {
        dim {
          size: 4
        }
      }
      int_val: 0
      int_val: 1
      int_val: 2
      int_val: 3
      """,
        t,
    )

  def testLargeInt(self):
    # [FUNCTION DOC]
    # Converts the largest signed 64-bit integer. Expects an int64 scalar proto with the value in
    # the 64-bit value field and a round trip int64 array equal to it.
    value = np.iinfo(np.int64).max
    t = tensor_util.make_tensor_proto(value)
    self.assertProtoEquals("""
      dtype: DT_INT64
      tensor_shape {}
      int64_val: %d
      """ % value, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(np.int64, a.dtype)
    self.assertAllClose(np.array(value, dtype=np.int64), a)

  def testLargeNegativeInt(self):
    # [FUNCTION DOC]
    # Converts the smallest signed 64-bit integer plus one (the true minimum is avoided because its
    # absolute value overflows). Expects an int64 scalar proto and a matching int64 round trip.
    # We don't use the min np.int64 value here
    # because it breaks np.abs().
    #
    # np.iinfo(np.int64).min = -9223372036854775808
    # np.iinfo(np.int64).max = 9223372036854775807
    # np.abs(-9223372036854775808) = -9223372036854775808
    value = np.iinfo(np.int64).min + 1
    t = tensor_util.make_tensor_proto(value)
    self.assertProtoEquals("""
      dtype: DT_INT64
      tensor_shape {}
      int64_val: %d
      """ % value, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(np.int64, a.dtype)
    self.assertAllClose(np.array(value, dtype=np.int64), a)

  def testIntNDefaultType(self):
    # [FUNCTION DOC]
    # Converts four Python integers with shape two by two and no dtype. Expects the default int32
    # type, packed content (host-endianness dependent) and a two by two int32 array on return.
    t = tensor_util.make_tensor_proto([10, 20, 30, 40], shape=[2, 2])
    if sys.byteorder == "big":
      self.assertProtoEquals(r"""
        dtype: DT_INT32
        tensor_shape { dim { size: 2 } dim { size: 2 } }
        tensor_content: "\000\000\000\n\000\000\000\024\000\000\000\036\000\000\000("
        """, t)
    else:
      self.assertProtoEquals(r"""
        dtype: DT_INT32
        tensor_shape { dim { size: 2 } dim { size: 2 } }
        tensor_content: "\n\000\000\000\024\000\000\000\036\000\000\000(\000\000\000"
        """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(np.int32, a.dtype)
    self.assertAllClose(np.array([[10, 20], [30, 40]], dtype=np.int32), a)

  @parameterized.named_parameters(
      ("_int8", dtypes.int8, np.int8), ("_int16", dtypes.int16, np.int16),
      ("_int32", dtypes.int32, np.int32), ("_int64", dtypes.int64, np.int64),
      ("_uint8", dtypes.uint8, np.uint8), ("_uint16", dtypes.uint16, np.uint16),
      ("_uint32", dtypes.uint32, np.uint32),
      ("_uint64", dtypes.uint64, np.uint64))
  def testIntTypes(self, dtype, nptype):
    # [FUNCTION DOC]
    # Parameterized over eight signed and unsigned integer widths. Input dtype is the framework
    # dtype and nptype the matching numpy type. Converts a list and then a numpy array of three
    # values, expecting the proto dtype, shape of three, and round trip array dtype and values to
    # match.
    # Test with array.
    t = tensor_util.make_tensor_proto([10, 20, 30], dtype=dtype)
    self.assertEqual(dtype, t.dtype)
    self.assertProtoEquals("dim { size: 3 }", t.tensor_shape)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(nptype, a.dtype)
    self.assertAllClose(np.array([10, 20, 30], dtype=nptype), a)
    # Test with ndarray.
    t = tensor_util.make_tensor_proto(np.array([10, 20, 30], dtype=nptype))
    self.assertEqual(dtype, t.dtype)
    self.assertProtoEquals("dim { size: 3 }", t.tensor_shape)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(nptype, a.dtype)
    self.assertAllClose(np.array([10, 20, 30], dtype=nptype), a)

  @parameterized.named_parameters(
      ("_int8", dtypes.int8, np.int8), ("_int16", dtypes.int16, np.int16),
      ("_int32", dtypes.int32, np.int32), ("_int64", dtypes.int64, np.int64),
      ("_uint8", dtypes.uint8, np.uint8), ("_uint16", dtypes.uint16, np.uint16),
      ("_uint32", dtypes.uint32, np.uint32),
      ("_uint64", dtypes.uint64, np.uint64))
  def testIntTypesWithImplicitRepeat(self, dtype, nptype):
    # [FUNCTION DOC]
    # Parameterized over eight integer types. Converts three values with a three by four target
    # shape and expects the last value to repeat to fill the remaining entries in row-major order
    # when converted back to an array of the matching numpy type.
    self.assertAllEqual(
        np.array([[10, 11, 12, 12], [12, 12, 12, 12], [12, 12, 12, 12]],
                 dtype=nptype),
        tensor_util.MakeNdarray(
            tensor_util.make_tensor_proto([10, 11, 12],
                                          shape=[3, 4],
                                          dtype=dtype)))

  def testIntMixedWithDimension(self):
    # [FUNCTION DOC]
    # Regression test for a reported issue: a list mixing integers with a shape dimension object
    # and an explicit int32 dtype converts successfully. Expects int32 dtype and values ten,
    # twenty, thirty after the round trip.
    # Github issue: 11974
    dtype = dtypes.int32
    nptype = np.int32
    t = tensor_util.make_tensor_proto(
        [10, tensor_shape.Dimension(20), 30], dtype=dtype)
    self.assertEqual(dtype, t.dtype)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(nptype, a.dtype)
    self.assertAllClose(np.array([10, 20, 30], dtype=nptype), a)

  @parameterized.named_parameters(
      ("_int64", dtypes.int64, np.int64, "DT_INT64", "int64_val"),
      ("_uint64", dtypes.uint64, np.uint64, "DT_UINT64", "uint64_val"))
  def testLong(self, dtype, nptype, proto_dtype, proto_value_name):
    # [FUNCTION DOC]
    # Parameterized over int64 and uint64. Converts the integer ten with the given dtype and
    # expects a scalar proto with the given proto dtype name and value field name, and a round trip
    # array of the numpy type.
    t = tensor_util.make_tensor_proto(10, dtype=dtype)
    self.assertProtoEquals(
        """
      dtype: %s
      tensor_shape {}
      %s: 10
    """ % (proto_dtype, proto_value_name), t)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(nptype, a.dtype)
    self.assertAllClose(np.array(10, dtype=nptype), a)

  @parameterized.named_parameters(
      ("_int64", dtypes.int64, np.int64, "DT_INT64"),
      ("_uint64", dtypes.uint64, np.uint64, "DT_UINT64"))
  def testLongN(self, dtype, nptype, proto_dtype):
    # [FUNCTION DOC]
    # Parameterized over int64 and uint64. Converts three integers with shape one by three and
    # expects packed eight-byte content (host-endianness dependent) with the given proto dtype, and
    # a matching numpy array on return.
    t = tensor_util.make_tensor_proto([10, 20, 30], shape=[1, 3], dtype=dtype)
    if sys.byteorder == "big":
      # pylint: disable=line-too-long
      self.assertProtoEquals(
          r"""
        dtype: %s
        tensor_shape { dim { size: 1 } dim { size: 3 } }
        tensor_content: "\000\000\000\000\000\000\000\n\000\000\000\000\000\000\000\024\000\000\000\000\000\000\000\036"
        """ % proto_dtype, t)
      # pylint: enable=line-too-long
    else:
      # pylint: disable=line-too-long
      self.assertProtoEquals(
          r"""
        dtype: %s
        tensor_shape { dim { size: 1 } dim { size: 3 } }
        tensor_content: "\n\000\000\000\000\000\000\000\024\000\000\000\000\000\000\000\036\000\000\000\000\000\000\000"
        """ % proto_dtype, t)
      # pylint: enable=line-too-long
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(nptype, a.dtype)
    self.assertAllClose(np.array([[10, 20, 30]], dtype=nptype), a)

  @parameterized.named_parameters(("_int64", np.int64, "DT_INT64"),
                                  ("_uint64", np.uint64, "DT_UINT64"))
  def testLongNpArray(self, nptype, proto_dtype):
    # [FUNCTION DOC]
    # Parameterized over int64 and uint64. Converts a numpy array of three values and expects
    # packed eight-byte content (host-endianness dependent), the given proto dtype and a matching
    # array after the round trip.
    t = tensor_util.make_tensor_proto(np.array([10, 20, 30], dtype=nptype))
    if sys.byteorder == "big":
      # pylint: disable=line-too-long
      self.assertProtoEquals(
          r"""
        dtype: %s
        tensor_shape { dim { size: 3 } }
        tensor_content: "\000\000\000\000\000\000\000\n\000\000\000\000\000\000\000\024\000\000\000\000\000\000\000\036"
        """ % proto_dtype, t)
      # pylint: enable=line-too-long
    else:
      # pylint: disable=line-too-long
      self.assertProtoEquals(
          r"""
        dtype: %s
        tensor_shape { dim { size: 3 } }
        tensor_content: "\n\000\000\000\000\000\000\000\024\000\000\000\000\000\000\000\036\000\000\000\000\000\000\000"
        """ % proto_dtype, t)
      # pylint: enable=line-too-long
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(nptype, a.dtype)
    self.assertAllClose(np.array([10, 20, 30], dtype=nptype), a)

  def testQuantizedTypes(self):
    # [FUNCTION DOC]
    # Converts three single-element tuples as each quantized type in turn (qint32, quint8, qint8,
    # quint16, qint16). For each, expects the proto to carry the quantized dtype and packed content
    # (byte order matters for the multi-byte types), and the round trip array to have the matching
    # numpy type and values.
    # Test with array.
    data = [(21,), (22,), (23,)]

    t = tensor_util.make_tensor_proto(data, dtype=dtypes.qint32)
    if sys.byteorder == "big":
      self.assertProtoEquals(r"""
        dtype: DT_QINT32
        tensor_shape { dim { size: 3 } }
        tensor_content: "\000\000\000\025\000\000\000\026\000\000\000\027"
        """, t)
    else:
      self.assertProtoEquals(r"""
        dtype: DT_QINT32
        tensor_shape { dim { size: 3 } }
        tensor_content: "\025\000\000\000\026\000\000\000\027\000\000\000"
        """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(dtypes.qint32.as_numpy_dtype, a.dtype)
    self.assertAllEqual(np.array(data, dtype=a.dtype), a)

    t = tensor_util.make_tensor_proto(data, dtype=dtypes.quint8)
    self.assertProtoEquals(r"""
      dtype: DT_QUINT8
      tensor_shape { dim { size: 3 } }
      tensor_content: "\025\026\027"
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(dtypes.quint8.as_numpy_dtype, a.dtype)
    self.assertAllEqual(np.array(data, dtype=a.dtype), a)

    t = tensor_util.make_tensor_proto(data, dtype=dtypes.qint8)
    self.assertProtoEquals(r"""
      dtype: DT_QINT8
      tensor_shape { dim { size: 3 } }
      tensor_content: "\025\026\027"
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(dtypes.qint8.as_numpy_dtype, a.dtype)
    self.assertAllEqual(np.array(data, dtype=a.dtype), a)

    t = tensor_util.make_tensor_proto(data, dtype=dtypes.quint16)
    if sys.byteorder == "big":
      self.assertProtoEquals(r"""
        dtype: DT_QUINT16
        tensor_shape { dim { size: 3 } }
        tensor_content: "\000\025\000\026\000\027"
        """, t)
    else:
      self.assertProtoEquals(r"""
        dtype: DT_QUINT16
        tensor_shape { dim { size: 3 } }
        tensor_content: "\025\000\026\000\027\000"
        """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(dtypes.quint16.as_numpy_dtype, a.dtype)
    self.assertAllEqual(np.array(data, dtype=a.dtype), a)

    t = tensor_util.make_tensor_proto(data, dtype=dtypes.qint16)
    if sys.byteorder == "big":
      self.assertProtoEquals(r"""
        dtype: DT_QINT16
        tensor_shape { dim { size: 3 } }
        tensor_content: "\000\025\000\026\000\027"
        """, t)
    else:
      self.assertProtoEquals(r"""
        dtype: DT_QINT16
        tensor_shape { dim { size: 3 } }
        tensor_content: "\025\000\026\000\027\000"
        """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(dtypes.qint16.as_numpy_dtype, a.dtype)
    self.assertAllEqual(np.array(data, dtype=a.dtype), a)

  def testString(self):
    # [FUNCTION DOC]
    # Converts the Python string foo. Expects a string scalar proto with one string value and a
    # round trip object-typed array holding the byte string.
    t = tensor_util.make_tensor_proto("foo")
    self.assertProtoEquals("""
      dtype: DT_STRING
      tensor_shape {}
      string_val: "foo"
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(np.object_, a.dtype)
    self.assertEqual([b"foo"], a)

  def testStringWithImplicitRepeat(self):
    # [FUNCTION DOC]
    # Converts a two-string list with a three by four target shape. Expects the last string to
    # repeat so the round trip array of byte strings fills the shape.
    t = tensor_util.make_tensor_proto(["f", "g"], shape=[3, 4])
    a = tensor_util.MakeNdarray(t)
    self.assertAllEqual(
        np.array([[b"f", b"g", b"g", b"g"], [b"g", b"g", b"g", b"g"],
                  [b"g", b"g", b"g", b"g"]],
                 dtype=np.object_), a)

  def testStringN(self):
    # [FUNCTION DOC]
    # Converts three byte strings with shape one by three. Expects a string proto with three string
    # values and a one by three object array on return.
    t = tensor_util.make_tensor_proto([b"foo", b"bar", b"baz"], shape=[1, 3])
    self.assertProtoEquals("""
      dtype: DT_STRING
      tensor_shape { dim { size: 1 } dim { size: 3 } }
      string_val: "foo"
      string_val: "bar"
      string_val: "baz"
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(np.object_, a.dtype)
    self.assertAllEqual(np.array([[b"foo", b"bar", b"baz"]]), a)

  def testStringNpArray(self):
    # [FUNCTION DOC]
    # Converts a two by two numpy array of byte strings. Expects a string proto of shape two by two
    # with four values in row-major order, and an object array equal to the input on return.
    t = tensor_util.make_tensor_proto(
        np.array([[b"a", b"ab"], [b"abc", b"abcd"]]))
    self.assertProtoEquals("""
      dtype: DT_STRING
      tensor_shape { dim { size: 2 } dim { size: 2 } }
      string_val: "a"
      string_val: "ab"
      string_val: "abc"
      string_val: "abcd"
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(np.object_, a.dtype)
    self.assertAllEqual(np.array([[b"a", b"ab"], [b"abc", b"abcd"]]), a)

  def testArrayMethod(self):
    # [FUNCTION DOC]
    # Defines a local wrapper class that exposes the array conversion method returning three byte
    # strings, converts an instance with shape one by three, and expects a string proto and array
    # as if the numpy array had been passed directly.

    class Wrapper(object):
      # [CLASS DOC]
      # HELPER: local test fixture for testArrayMethod.
      # Object that is not an array but implements the array conversion protocol method so that
      # tensor conversion treats it as array-like.

      def __array__(self, dtype=None):
        # [FUNCTION DOC]
        # HELPER: array conversion protocol method of the local wrapper.
        # Ignores the requested dtype and returns a numpy array of three byte strings. No side effects.
        del dtype
        return np.array([b"foo", b"bar", b"baz"])

    t = tensor_util.make_tensor_proto(Wrapper(), shape=[1, 3])
    self.assertProtoEquals("""
      dtype: DT_STRING
      tensor_shape { dim { size: 1 } dim { size: 3 } }
      string_val: "foo"
      string_val: "bar"
      string_val: "baz"
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(np.object_, a.dtype)
    self.assertAllEqual(np.array([[b"foo", b"bar", b"baz"]]), a)

  def testArrayInterface(self):
    # [FUNCTION DOC]
    # Defines a local wrapper class exposing the array interface mapping of an internal numpy
    # array, converts an instance with shape one by three and expects the same string proto and
    # round trip array as for a plain numpy array.

    class Wrapper(object):
      # [CLASS DOC]
      # HELPER: local test fixture for testArrayInterface.
      # Holds a numpy array of three byte strings and exposes its array interface description so the
      # object is recognised as array-like.

      def __init__(self):
        # [FUNCTION DOC]
        # HELPER: constructor of the local wrapper. Creates the internal numpy array of three byte
        # strings and stores it. Takes no inputs and returns nothing.
        self.a = np.array([b"foo", b"bar", b"baz"])

      @property
      def __array_interface__(self):
        # [FUNCTION DOC]
        # HELPER: property of the local wrapper. Returns the array interface description of the stored
        # numpy array so the wrapper can be converted without copying through the array protocol.
        return self.a.__array_interface__

    t = tensor_util.make_tensor_proto(Wrapper(), shape=[1, 3])
    self.assertProtoEquals("""
      dtype: DT_STRING
      tensor_shape { dim { size: 1 } dim { size: 3 } }
      string_val: "foo"
      string_val: "bar"
      string_val: "baz"
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(np.object_, a.dtype)
    self.assertAllEqual(np.array([[b"foo", b"bar", b"baz"]]), a)

  def testStringTuple(self):
    # [FUNCTION DOC]
    # Converts a tuple of four byte strings. Expects a string proto of shape four with the values
    # in order and an object array of the same strings on return.
    t = tensor_util.make_tensor_proto((b"a", b"ab", b"abc", b"abcd"))
    self.assertProtoEquals("""
      dtype: DT_STRING
      tensor_shape { dim { size: 4 } }
      string_val: "a"
      string_val: "ab"
      string_val: "abc"
      string_val: "abcd"
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(np.object_, a.dtype)
    self.assertAllEqual(np.array((b"a", b"ab", b"abc", b"abcd")), a)

  def testStringNestedTuple(self):
    # [FUNCTION DOC]
    # Converts a nested tuple of byte strings (two by two). Expects a string proto of shape two by
    # two with four values in row-major order and a matching object array on return.
    t = tensor_util.make_tensor_proto(((b"a", b"ab"), (b"abc", b"abcd")))
    self.assertProtoEquals("""
      dtype: DT_STRING
      tensor_shape { dim { size: 2 } dim { size: 2 } }
      string_val: "a"
      string_val: "ab"
      string_val: "abc"
      string_val: "abcd"
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(np.object_, a.dtype)
    self.assertAllEqual(np.array(((b"a", b"ab"), (b"abc", b"abcd"))), a)

  def testComplex64(self):
    # [FUNCTION DOC]
    # Converts a complex scalar with complex64 dtype. Expects a proto with real and imaginary parts
    # stored as two single-precision values and a complex64 array on return.
    t = tensor_util.make_tensor_proto((1 + 2j), dtype=dtypes.complex64)
    self.assertProtoEquals("""
      dtype: DT_COMPLEX64
      tensor_shape {}
      scomplex_val: 1
      scomplex_val: 2
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(np.complex64, a.dtype)
    self.assertAllEqual(np.array(1 + 2j), a)

  def testComplex128(self):
    # [FUNCTION DOC]
    # Converts a complex scalar with complex128 dtype. Expects a proto with real and imaginary
    # parts stored as two double-precision values and a complex128 array on return.
    t = tensor_util.make_tensor_proto((1 + 2j), dtype=dtypes.complex128)
    self.assertProtoEquals("""
      dtype: DT_COMPLEX128
      tensor_shape {}
      dcomplex_val: 1
      dcomplex_val: 2
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(np.complex128, a.dtype)
    self.assertAllEqual(np.array(1 + 2j), a)

  def testComplexWithImplicitRepeat(self):
    # [FUNCTION DOC]
    # For complex64 and complex128, converts one complex value with a three by four target shape.
    # Expects the value to repeat to fill the array on conversion back.
    for dtype, np_dtype in [(dtypes.complex64, np.complex64),
                            (dtypes.complex128, np.complex128)]:
      t = tensor_util.make_tensor_proto((1 + 1j), shape=[3, 4], dtype=dtype)
      a = tensor_util.MakeNdarray(t)
      self.assertAllClose(
          np.array(
              [[(1 + 1j), (1 + 1j), (1 + 1j), (1 + 1j)],
               [(1 + 1j), (1 + 1j), (1 + 1j), (1 + 1j)],
               [(1 + 1j), (1 + 1j), (1 + 1j), (1 + 1j)]],
              dtype=np_dtype),
          a)

  def testComplex64N(self):
    # [FUNCTION DOC]
    # Converts three complex numbers with shape one by three as complex64. Expects interleaved real
    # and imaginary single values in the proto and a complex64 one by three array on return.
    t = tensor_util.make_tensor_proto(
        [(1 + 2j), (3 + 4j), (5 + 6j)], shape=[1, 3], dtype=dtypes.complex64)
    self.assertProtoEquals("""
      dtype: DT_COMPLEX64
      tensor_shape { dim { size: 1 } dim { size: 3 } }
      scomplex_val: 1
      scomplex_val: 2
      scomplex_val: 3
      scomplex_val: 4
      scomplex_val: 5
      scomplex_val: 6
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(np.complex64, a.dtype)
    self.assertAllEqual(np.array([[(1 + 2j), (3 + 4j), (5 + 6j)]]), a)

  def testComplex128N(self):
    # [FUNCTION DOC]
    # Converts three complex numbers with shape one by three as complex128. Expects interleaved
    # real and imaginary double values in the proto and a complex128 array on return.
    t = tensor_util.make_tensor_proto(
        [(1 + 2j), (3 + 4j), (5 + 6j)], shape=[1, 3], dtype=dtypes.complex128)
    self.assertProtoEquals("""
      dtype: DT_COMPLEX128
      tensor_shape { dim { size: 1 } dim { size: 3 } }
      dcomplex_val: 1
      dcomplex_val: 2
      dcomplex_val: 3
      dcomplex_val: 4
      dcomplex_val: 5
      dcomplex_val: 6
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(np.complex128, a.dtype)
    self.assertAllEqual(np.array([[(1 + 2j), (3 + 4j), (5 + 6j)]]), a)

  def testComplex64NpArray(self):
    # [FUNCTION DOC]
    # Converts a two by two complex numpy array requesting complex64. Expects eight interleaved
    # real and imaginary values in the proto and an equal complex64 array on return.
    t = tensor_util.make_tensor_proto(
        np.array([[(1 + 2j), (3 + 4j)], [(5 + 6j), (7 + 8j)]]),
        dtype=dtypes.complex64)
    # scomplex_val are real_0, imag_0, real_1, imag_1, ...
    self.assertProtoEquals("""
      dtype: DT_COMPLEX64
      tensor_shape { dim { size: 2 } dim { size: 2 } }
      scomplex_val: 1
      scomplex_val: 2
      scomplex_val: 3
      scomplex_val: 4
      scomplex_val: 5
      scomplex_val: 6
      scomplex_val: 7
      scomplex_val: 8
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(np.complex64, a.dtype)
    self.assertAllEqual(
        np.array([[(1 + 2j), (3 + 4j)], [(5 + 6j), (7 + 8j)]]), a)

  def testComplex128NpArray(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the inline comment about the value layout refers to the single-precision field
    # name although this test checks the double-precision field.
    # Converts a two by two complex numpy array requesting complex128. Expects eight interleaved
    # values in the proto and an equal complex128 array on return.
    t = tensor_util.make_tensor_proto(
        np.array([[(1 + 2j), (3 + 4j)], [(5 + 6j), (7 + 8j)]]),
        dtype=dtypes.complex128)
    # scomplex_val are real_0, imag_0, real_1, imag_1, ...
    self.assertProtoEquals("""
      dtype: DT_COMPLEX128
      tensor_shape { dim { size: 2 } dim { size: 2 } }
      dcomplex_val: 1
      dcomplex_val: 2
      dcomplex_val: 3
      dcomplex_val: 4
      dcomplex_val: 5
      dcomplex_val: 6
      dcomplex_val: 7
      dcomplex_val: 8
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(np.complex128, a.dtype)
    self.assertAllEqual(
        np.array([[(1 + 2j), (3 + 4j)], [(5 + 6j), (7 + 8j)]]), a)

  def testNestedNumpyArrayWithoutDType(self):
    # [FUNCTION DOC]
    # Converts a list that mixes floats with a zero-dimensional numpy array and no dtype. Expects
    # float32 inference and the values ten, twenty, thirty on return.
    t = tensor_util.make_tensor_proto([10.0, 20.0, np.array(30.0)])
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(np.float32, a.dtype)
    self.assertAllClose(np.array([10.0, 20.0, 30.0], dtype=np.float32), a)

  def testNestedNumpyArrayWithDType(self):
    # [FUNCTION DOC]
    # Same as the previous test but with an explicit float32 dtype. Expects a float32 array with
    # values ten, twenty, thirty on return.
    t = tensor_util.make_tensor_proto([10.0, 20.0, np.array(30.0)],
                                      dtype=dtypes.float32)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(np.float32, a.dtype)
    self.assertAllClose(np.array([10.0, 20.0, 30.0], dtype=np.float32), a)

  def testUnsupportedDTypes(self):
    # [FUNCTION DOC]
    # Checks error cases: a non-dtype second argument, and quantized dtype with scalar or list
    # input all raise a type error. Also checks that a list containing a shape dimension object
    # with no dtype raises a type error whose message says conversion of elements failed.
    with self.assertRaises(TypeError):
      tensor_util.make_tensor_proto(np.array([1]), 0)
    with self.assertRaises(TypeError):
      tensor_util.make_tensor_proto(3, dtype=dtypes.qint8)
    with self.assertRaises(TypeError):
      tensor_util.make_tensor_proto([3], dtype=dtypes.qint8)

    # Validate the helpful error message when trying to convert an
    # unconvertible list as strings.
    with self.assertRaisesRegex(TypeError, "Failed to convert elements"):
      tensor_util.make_tensor_proto([tensor_shape.Dimension(1)])

  def testTensorShapeVerification(self):
    # [FUNCTION DOC]
    # Checks shape verification: converting a two by one array with a matching shape and the verify
    # flag succeeds, whereas a mismatched one by two shape with verification raises a type error.
    array = np.array([[1], [2]])
    correct_shape = (2, 1)
    incorrect_shape = (1, 2)
    tensor_util.make_tensor_proto(array, shape=correct_shape, verify_shape=True)
    with self.assertRaises(TypeError):
      tensor_util.make_tensor_proto(
          array, shape=incorrect_shape, verify_shape=True)

  def testShapeTooLarge(self):
    # [FUNCTION DOC]
    # Checks that converting two values with a target shape that holds only one element raises a
    # value error.
    with self.assertRaises(ValueError):
      tensor_util.make_tensor_proto(np.array([1, 2]), shape=[1])

  def testLowRankSupported(self):
    # [FUNCTION DOC]
    # Converts a zero-dimensional numpy integer array. Expects an int64 scalar proto with empty
    # shape and the value seven.
    t = tensor_util.make_tensor_proto(np.array(7))
    self.assertProtoEquals("""
      dtype: DT_INT64
      tensor_shape {}
      int64_val: 7
      """, t)

  def testShapeEquals(self):
    # [FUNCTION DOC]
    # Builds a two by two proto and checks the shape-equality helper against shapes given as list,
    # tuple and shape proto (true), and against several different shapes such as different rank or
    # sizes, empty and extra dimension (all false).
    t = tensor_util.make_tensor_proto([10, 20, 30, 40], shape=[2, 2])
    self.assertTrue(tensor_util.ShapeEquals(t, [2, 2]))
    self.assertTrue(tensor_util.ShapeEquals(t, (2, 2)))
    self.assertTrue(
        tensor_util.ShapeEquals(t, tensor_shape.as_shape([2, 2]).as_proto()))
    self.assertFalse(tensor_util.ShapeEquals(t, [5, 3]))
    self.assertFalse(tensor_util.ShapeEquals(t, [1, 4]))
    self.assertFalse(tensor_util.ShapeEquals(t, [4]))
    self.assertFalse(tensor_util.ShapeEquals(t, [2]))
    self.assertFalse(tensor_util.ShapeEquals(t, []))
    self.assertFalse(tensor_util.ShapeEquals(t, [2, 2, 1]))


@test_util.run_all_in_graph_and_eager_modes
class IsTensorTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the predicate that decides whether a value is a framework tensor-like type, run in
  # graph and eager modes. Covers dense, ragged, sparse, indexed-slices and variable inputs and
  # their evaluated plain-value counterparts.

  def testConstantTensor(self):
    # [FUNCTION DOC]
    # Creates a numpy array and a constant tensor from it. Expects the predicate to be false for the
    # numpy array and true for the tensor.
    # KNOWN ISSUE: the array is random floats cast to int32, so it is always all zeros; the
    # randomness is meaningless.
    np_val = np.random.rand(3).astype(np.int32)
    tf_val = constant_op.constant(np_val)
    self.assertFalse(tensor_util.is_tf_type(np_val))
    self.assertTrue(tensor_util.is_tf_type(tf_val))

  def testRaggedTensor(self):
    # [FUNCTION DOC]
    # Builds a ragged tensor and its evaluated value. Expects the predicate to be true for the
    # ragged tensor and false for the evaluated value.
    rt = ragged_factory_ops.constant([[1, 2], [3]])
    rt_value = self.evaluate(rt)
    self.assertTrue(tensor_util.is_tf_type(rt))
    self.assertFalse(tensor_util.is_tf_type(rt_value))

  def testSparseTensor(self):
    # [FUNCTION DOC]
    # Builds a sparse tensor and its evaluated value. Expects true for the sparse tensor and false
    # for the evaluated sparse value.
    st = sparse_tensor.SparseTensor([[1, 2]], [3], [10, 10])
    st_value = self.evaluate(st)
    self.assertTrue(tensor_util.is_tf_type(st))
    self.assertFalse(tensor_util.is_tf_type(st_value))

  def testIndexedSlices(self):
    # [FUNCTION DOC]
    # Builds an indexed-slices object from constants and an indexed-slices value from numpy
    # arrays. Expects true for the former and false for the latter.
    x = indexed_slices.IndexedSlices(
        constant_op.constant([1, 2, 3]), constant_op.constant([10, 20, 30]))
    x_value = indexed_slices.IndexedSlicesValue(
        np.array([1, 2, 3]), np.array([10, 20, 30]), np.array([100]))
    self.assertTrue(tensor_util.is_tf_type(x))
    self.assertFalse(tensor_util.is_tf_type(x_value))

  def testVariable(self):
    # [FUNCTION DOC]
    # Creates a variable from a list of integers. Expects the predicate to be true for the variable.
    v = variables.Variable([1, 2, 3])
    self.assertTrue(tensor_util.is_tf_type(v))


class ConstantValueTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the constant-value extraction helper, which statically evaluates a tensor to a numpy
  # value when possible and otherwise returns nothing (or a partial result when requested). Covers
  # constants, shape, fill, size, rank, cast, concat, pack, unpack, split, comparison, identity,
  # stop-gradient, literals and variables. Not decorated for both modes; graph-only cases build
  # their own graph.

  def testConstant(self):
    # [FUNCTION DOC]
    # For a random float array of shape three by four by seven and for an empty-dimension array of
    # shape three by zero by seven, wraps each in a constant tensor and expects the extracted
    # constant value to be close to the original.
    np_val = np.random.rand(3, 4, 7).astype(np.float32)
    tf_val = constant_op.constant(np_val)
    self.assertAllClose(np_val, tensor_util.constant_value(tf_val))

    np_val = np.random.rand(3, 0, 7).astype(np.float32)
    tf_val = constant_op.constant(np_val)
    self.assertAllClose(np_val, tensor_util.constant_value(tf_val))

  def testUnknown(self):
    # [FUNCTION DOC]
    # Inside a fresh graph, creates a variable op whose value is not statically known. Expects the
    # extraction to return nothing.
    with ops.Graph().as_default():
      tf_val = gen_state_ops.variable(
          shape=[3, 4, 7],
          dtype=dtypes.float32,
          name="tf_val",
          container="",
          shared_name="")
      self.assertIs(None, tensor_util.constant_value(tf_val))

  def testShape(self):
    # [FUNCTION DOC]
    # Takes the shape of a constant of shape one, two, three. Expects the extracted value to equal
    # the array one, two, three with dtype int32.
    np_val = np.array([1, 2, 3], dtype=np.int32)
    tf_val = array_ops.shape(constant_op.constant(0.0, shape=[1, 2, 3]))
    c_val = tensor_util.constant_value(tf_val)
    self.assertAllEqual(np_val, c_val)
    self.assertEqual(np.int32, c_val.dtype)

  def testFill(self):
    # [FUNCTION DOC]
    # Fills a vector of length three with a constant minus one. Expects the extracted value to be
    # the float32 array of minus ones.
    np_val = np.array([-1, -1, -1], dtype=np.float32)
    tf_val = array_ops.fill([3], constant_op.constant(-1.0))
    c_val = tensor_util.constant_value(tf_val)
    self.assertAllEqual(np_val, c_val)
    self.assertEqual(np.float32, c_val.dtype)

  def testSize(self):
    # [FUNCTION DOC]
    # Takes the size of a constant of shape one, two, three. Expects the extracted value to be six.
    tf_val = array_ops.size(constant_op.constant(0.0, shape=[1, 2, 3]))
    c_val = tensor_util.constant_value(tf_val)
    self.assertEqual(6, c_val)

  def testSizeOfScalar(self):
    # [FUNCTION DOC]
    # Takes the size of a scalar constant. Expects the extracted value to be one and its type to be
    # either an array or a 32-bit integer scalar.
    tf_val = array_ops.size(constant_op.constant(0.0))
    c_val = tensor_util.constant_value(tf_val)
    self.assertEqual(1, c_val)
    self.assertIn(type(c_val), [np.ndarray, np.int32])

  def testRank(self):
    # [FUNCTION DOC]
    # Takes the rank of a constant of shape one, two, three, both through the public rank function
    # and through the internal version with optimization disabled. Expects value three, an empty
    # shape, and array or 32-bit integer type each time.
    # KNOWN ISSUE: the final assertion compares a one-element list to a scalar value, which only
    # passes through numpy equality and contradicts the empty-shape assertion made just before.
    tf_val = array_ops.rank(constant_op.constant(0.0, shape=[1, 2, 3]))
    c_val = tensor_util.constant_value(tf_val)

    self.assertIn(type(c_val), [np.ndarray, np.int32])
    self.assertEqual((), c_val.shape)
    self.assertEqual(3, c_val)

    # Repeat test using array_ops.rank_internal to avoid the optimization that
    # happens in the rank function.
    tf_val = array_ops.rank_internal(
        constant_op.constant(
            0.0, shape=[1, 2, 3]), optimize=False)
    c_val = tensor_util.constant_value(tf_val)

    self.assertIn(type(c_val), [np.ndarray, np.int32])
    self.assertEqual((), c_val.shape)
    self.assertEqual(3, c_val)
    self.assertEqual([3], c_val)

  def testCast(self):
    # [FUNCTION DOC]
    # Casts a float32 constant to float64, for a non-empty array and for an array with a zero-sized
    # dimension. Expects the extracted value to be close to the float64 version of the input.
    np_val = np.random.rand(3, 4, 7).astype(np.float32)
    tf_val = math_ops.cast(constant_op.constant(np_val), dtypes.float64)
    c_val = tensor_util.constant_value(tf_val)
    self.assertAllClose(np_val.astype(np.float64), c_val)

    np_val = np.random.rand(3, 0, 7).astype(np.float32)
    tf_val = math_ops.cast(constant_op.constant(np_val), dtypes.float64)
    c_val = tensor_util.constant_value(tf_val)
    self.assertAllClose(np_val.astype(np.float64), c_val)

  def testConcat(self):
    # [FUNCTION DOC]
    # Concatenates three constant slices along axis zero and expects the extracted value to equal
    # the original array. Then, in a graph, checks that concatenation with a placeholder axis, and
    # with a placeholder among the inputs, yields nothing.
    np_val = np.random.rand(3, 4, 7).astype(np.float32)
    tf_val = array_ops.concat(
        [np_val[0:1, :, :], np_val[1:2, :, :], np_val[2:3, :, :]], 0)
    c_val = tensor_util.constant_value(tf_val)
    self.assertAllClose(np_val, c_val)

    # This test needs a placeholder which means we need to construct a graph.
    with ops.Graph().as_default():
      tf_val = array_ops.concat(
          [np_val[0, :, :], np_val[1, :, :], np_val[2, :, :]],
          array_ops.placeholder(dtypes.int32))
      c_val = tensor_util.constant_value(tf_val)
      self.assertIs(None, c_val)

      tf_val = array_ops.concat([
          np_val[0, :, :],
          array_ops.placeholder(dtypes.float32), np_val[2, :, :]
      ], 1)
      c_val = tensor_util.constant_value(tf_val)
      self.assertIs(None, c_val)

  def testPack_Axis0(self):
    # [FUNCTION DOC]
    # Stacks three constant matrices along axis zero and expects the extracted value to equal the
    # stacked array. In a graph, stacking with a placeholder as one input must give nothing.
    inputs = [np.random.rand(4, 7) for _ in range(3)]
    np_val = np.array(inputs)
    tf_val = array_ops_stack.stack(inputs)
    c_val = tensor_util.constant_value(tf_val)
    self.assertAllClose(np_val, c_val)

    # This test needs a placeholder which means we need to construct a graph.
    with ops.Graph().as_default():
      tf_val = array_ops_stack.stack(
          [inputs[0],
           array_ops.placeholder(dtypes.float32), inputs[2]])
      c_val = tensor_util.constant_value(tf_val)
      self.assertIs(None, c_val)

  def testPack_Axis1(self):
    # [FUNCTION DOC]
    # In a graph, stacks three constant matrices along axis one and expects nothing to be returned,
    # then does the same with a placeholder among the inputs and expects nothing again.
    # Even the fully constant case yields nothing, so stacking along axis one is not supported by
    # the extraction.
    # This test needs a placeholder which means we need to construct a graph.
    with ops.Graph().as_default():
      inputs = [np.random.rand(4, 7) for _ in range(3)]
      tf_val = array_ops_stack.stack(inputs, axis=1)
      c_val = tensor_util.constant_value(tf_val)
      self.assertIsNone(c_val)

      tf_val = array_ops_stack.stack(
          [inputs[0],
           array_ops.placeholder(dtypes.float32), inputs[2]], axis=1)
      c_val = tensor_util.constant_value(tf_val)
      self.assertIs(None, c_val)

  def testPack_Partial_Axis0(self):
    # [FUNCTION DOC]
    # In a graph, stacks a constant matrix with a placeholder along axis zero and asks for a
    # partial result. Expects the first element to be the constant and the second to be nothing.
    input_ = np.random.rand(4, 7)
    # This test needs a placeholder which means we need to construct a graph.
    with ops.Graph().as_default():
      tf_val = array_ops_stack.stack(
          [input_, array_ops.placeholder(dtypes.float32)])
      c_val = tensor_util.constant_value(tf_val, partial=True)
      self.assertAllClose(input_, c_val[0])
      self.assertIsNone(c_val[1])

  def testPack_Partial_Axis1(self):
    # [FUNCTION DOC]
    # In a graph, stacks a constant and a placeholder along axis one and asks for a partial result.
    # Expects nothing overall, as partial results are only supported along axis zero.
    input_ = np.random.rand(4, 7)
    # This test needs a placeholder which means we need to construct a graph.
    with ops.Graph().as_default():
      tf_val = array_ops_stack.stack(
          [input_, array_ops.placeholder(dtypes.float32)], axis=1)
      c_val = tensor_util.constant_value(tf_val, partial=True)
      self.assertIsNone(c_val)

  def testUnpack_Axis0(self):
    # [FUNCTION DOC]
    # Unstacks a constant array of shape three by four by seven along axis zero. Expects the
    # extracted value of each piece to match the corresponding slice.
    inputs = np.random.rand(3, 4, 7)
    tf_vals = array_ops_stack.unstack(inputs)
    c_vals = [tensor_util.constant_value(x) for x in tf_vals]
    self.assertAllClose(inputs, c_vals)

  def testUnpack_Partial_Axis0(self):
    # [FUNCTION DOC]
    # In a graph, stacks a constant and a placeholder, unstacks the result and asks for partial
    # values. Expects the first piece to match the constant and the second to be nothing.
    input_ = np.random.rand(4, 7)
    # This test needs a placeholder which means we need to construct a graph.
    with ops.Graph().as_default():
      packed = array_ops_stack.stack(
          [input_, array_ops.placeholder(dtypes.float32)])
      tf_vals = array_ops_stack.unstack(packed)
      c_vals = [tensor_util.constant_value(x, partial=True) for x in tf_vals]
      self.assertAllClose(input_, c_vals[0])
      self.assertIsNone(c_vals[1])

  def testSplit_Axis0(self):
    # [FUNCTION DOC]
    # Splits a constant array of six rows into three parts along axis zero. Expects the extracted
    # values to match the equivalent numpy split.
    inputs = np.random.rand(6, 5, 7)
    tf_vals = array_ops.split(inputs, 3)
    c_vals = [tensor_util.constant_value(x) for x in tf_vals]
    self.assertAllClose(np.split(inputs, 3), c_vals)

  def testSplit_Partial_Axis0(self):
    # [FUNCTION DOC]
    # In a graph, stacks a constant and a fixed-shape placeholder, splits into two and asks for
    # partial values. Expects the first part to hold the constant and the second to be nothing.
    input_ = np.random.rand(4, 7)
    # This test needs a placeholder which means we need to construct a graph.
    with ops.Graph().as_default():
      placeholder = array_ops.placeholder(dtypes.float32, shape=(4, 7))
      # it'd be better to use concat here, but concat doesn't support partial
      packed = array_ops_stack.stack([input_, placeholder])
      tf_vals = array_ops.split(packed, 2)
      c_vals = [tensor_util.constant_value(x, partial=True) for x in tf_vals]
      self.assertAllClose(input_, c_vals[0][0])
      self.assertIsNone(c_vals[1][0])

  def testEqual(self):
    # [FUNCTION DOC]
    # Checks the extraction for elementwise equality: scalar equal and unequal inputs give true and
    # false, and a row against a column with broadcasting gives the expected two by two boolean
    # array.
    # Scalar inputs.
    tf_val = math_ops.equal(constant_op.constant(1), constant_op.constant(1))
    self.assertEqual(tensor_util.constant_value(tf_val), True)

    tf_val = math_ops.equal(constant_op.constant(1), constant_op.constant(0))
    self.assertEqual(tensor_util.constant_value(tf_val), False)

    # Shaped inputs with broadcast semantics.
    tf_val = math_ops.equal(constant_op.constant([[0, 1]]),
                            constant_op.constant([[0], [1]]))
    c_val = tensor_util.constant_value(tf_val)
    self.assertAllEqual(c_val, [[True, False], [False, True]])

  def testNotEqual(self):
    # [FUNCTION DOC]
    # Checks the extraction for elementwise inequality: scalar equal and unequal inputs give false
    # and true, and broadcasting a row against a column gives the complement boolean array.
    # Scalar inputs.
    tf_val = math_ops.not_equal(constant_op.constant(1),
                                constant_op.constant(1))
    self.assertEqual(tensor_util.constant_value(tf_val), False)

    tf_val = math_ops.not_equal(constant_op.constant(1),
                                constant_op.constant(0))
    self.assertEqual(tensor_util.constant_value(tf_val), True)

    # Shaped inputs with broadcast semantics.
    tf_val = math_ops.not_equal(constant_op.constant([[0, 1]]),
                                constant_op.constant([[0], [1]]))
    c_val = tensor_util.constant_value(tf_val)
    self.assertAllEqual(c_val, [[False, True], [True, False]])

  def testStopGradient(self):
    # [FUNCTION DOC]
    # Applies the stop-gradient operation to a numpy array and expects the extracted value to be
    # equal to the array (the operation is transparent to extraction).
    input_ = np.random.rand(4, 7)
    tf_val = array_ops.stop_gradient(input_)
    c_val = tensor_util.constant_value(tf_val)
    self.assertAllEqual(input_, c_val)

  def testIdentity(self):
    # [FUNCTION DOC]
    # Applies the identity operation to a numpy array and expects the extracted value to equal the
    # array.
    input_ = np.random.rand(4, 7)
    tf_val = array_ops.identity(input_)
    c_val = tensor_util.constant_value(tf_val)
    self.assertAllEqual(input_, c_val)

  def testLiteral(self):
    # [FUNCTION DOC]
    # Passes a plain string. Expects the extraction to return the very same object unchanged.
    x = "hi"
    self.assertIs(x, tensor_util.constant_value(x))

  def testNumpyNdarray(self):
    # [FUNCTION DOC]
    # Passes a numpy array. Expects the extraction to return the very same array object.
    np_val = np.random.rand(3, 4, 7).astype(np.float32)
    self.assertIs(np_val, tensor_util.constant_value(np_val))

  def testVariable(self):
    # [FUNCTION DOC]
    # Creates a variable. Expects the extraction to return nothing because variable values are not
    # constants.
    var = variables.Variable(1.0, name="variable_node")
    self.assertIsNone(tensor_util.constant_value(var))

  def testVariableV1(self):
    # [FUNCTION DOC]
    # Creates a legacy (version one) variable. Expects the extraction to return nothing.
    var = variable_v1.VariableV1(1.0, name="variable_node")
    self.assertIsNone(tensor_util.constant_value(var))


class ConstantValueAsShapeTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the helper that interprets a constant tensor as a partially known tensor shape,
  # treating minus one entries as unknown dimensions. Most tests run in both graph and eager
  # modes; those needing placeholders build their own graph.

  @test_util.run_in_graph_and_eager_modes
  def testConstant(self):
    # [FUNCTION DOC]
    # Converts a constant vector of integers, and an empty int32 constant, to shapes. Expects them
    # to equal the shape built from the same values (the empty one giving the scalar shape).
    np_val = np.random.rand(3).astype(np.int32)
    tf_val = constant_op.constant(np_val)
    self.assertEqual(
        tensor_shape.TensorShape(np_val),
        tensor_util.constant_value_as_shape(tf_val))

    tf_val = constant_op.constant([], dtype=dtypes.int32)
    self.assertEqual(
        tensor_shape.TensorShape([]),
        tensor_util.constant_value_as_shape(tf_val))

  @test_util.run_in_graph_and_eager_modes
  def testCast(self):
    # [FUNCTION DOC]
    # Casts the shape of a constant of shape one, two, three to int64 and converts it. Expects the
    # shape one, two, three.
    tf_val = math_ops.cast(
        array_ops.shape(constant_op.constant(0.0, shape=[1, 2, 3])),
        dtypes.int64)
    c_val = tensor_util.constant_value_as_shape(tf_val)
    self.assertEqual(tensor_shape.TensorShape([1, 2, 3]), c_val)

  @test_util.run_in_graph_and_eager_modes
  def testCastWithUnknown(self):
    # [FUNCTION DOC]
    # Casts a constant of minus one, one, minus one to int64 and converts it. Expects a shape whose
    # list form has unknown first and last dimensions and a middle dimension of one.
    tf_val = math_ops.cast(constant_op.constant([-1, 1, -1]), dtypes.int64)
    c_val = tensor_util.constant_value_as_shape(tf_val)
    self.assertEqual([None, 1, None], c_val.as_list())

  @test_util.run_in_graph_and_eager_modes
  def testShape(self):
    # [FUNCTION DOC]
    # Takes the shape tensor of a constant of shape one, two, three and converts it. Expects the
    # shape one, two, three.
    tf_val = array_ops.shape(constant_op.constant(0.0, shape=[1, 2, 3]))
    c_val = tensor_util.constant_value_as_shape(tf_val)
    self.assertEqual(tensor_shape.TensorShape([1, 2, 3]), c_val)

  @test_util.run_in_graph_and_eager_modes
  def testMinusOneBecomesNone(self):
    # [FUNCTION DOC]
    # Converts a constant vector with minus one entries to a shape. Expects minus one to be treated
    # as unknown, so the list form is unknown, one, unknown.
    tf_val = constant_op.constant([-1, 1, -1], shape=[3])
    c_val = tensor_util.constant_value_as_shape(tf_val)
    self.assertEqual([None, 1, None], c_val.as_list())

  def testPack(self):
    # [FUNCTION DOC]
    # In a graph, stacks a constant sixteen, the number thirty-seven and an integer placeholder,
    # then converts to a shape. Expects the list sixteen, thirty-seven, unknown.
    # This test needs a placeholder which means we need to construct a graph.
    with ops.Graph().as_default():
      tf_val = array_ops_stack.stack(
          [constant_op.constant(16), 37,
           array_ops.placeholder(dtypes.int32)])
      c_val = tensor_util.constant_value_as_shape(tf_val)
      self.assertEqual([16, 37, None], c_val.as_list())

  def testConcat(self):
    # [FUNCTION DOC]
    # In a graph, concatenates a literal pair with placeholders of known length and converts the
    # result. Expects unknown entries exactly where placeholder elements are (two unknowns for a
    # length two placeholder, one unknown between known values for a length one placeholder).
    # This test needs a placeholder which means we need to construct a graph.
    with ops.Graph().as_default():
      tf_val = array_ops.concat(
          [[16, 37], array_ops.placeholder(dtypes.int32, shape=(2,))], 0)
      c_val = tensor_util.constant_value_as_shape(tf_val)
      self.assertEqual([16, 37, None, None], c_val.as_list())

      tf_val = array_ops.concat(
          [[16, 37],
           array_ops.placeholder(dtypes.int32, shape=(1,)), [48]], 0)
      c_val = tensor_util.constant_value_as_shape(tf_val)
      self.assertEqual([16, 37, None, 48], c_val.as_list())

  def testSlice(self):
    # [FUNCTION DOC]
    # Exercises shape extraction through strided slicing. Covers a placeholder slice (all unknown),
    # simple and strided slices of a constant, and in a graph many slices of a concatenation that
    # has one unknown element, using positive and negative begin, end and stride. Expects the
    # listed known or unknown values for each. Also expects that an ellipsis slice gives all
    # unknown entries, a slice with a tensor bound gives a completely unknown shape, and a higher
    # rank slice raises a value error.
    # This test needs a placeholder which means we need to construct a graph.
    with ops.Graph().as_default():
      tf_val = array_ops.placeholder(dtypes.int32, shape=(4,))[0:2]
      c_val = tensor_util.constant_value_as_shape(tf_val)
      self.assertEqual([None, None], c_val.as_list())

    # begin:end
    tf_val = constant_op.constant([10, 20, 30])[1:3]
    c_val = tensor_util.constant_value_as_shape(tf_val)
    self.assertEqual([20, 30], c_val.as_list())

    # begin:end:stride
    tf_val = array_ops.strided_slice(
        constant_op.constant([10, 20, 30]), [1], [3], strides=[2])
    c_val = tensor_util.constant_value_as_shape(tf_val)
    self.assertEqual([20], c_val.as_list())

    # [1, 2, 16, 37, None, 48]
    # This test needs a placeholder which means we need to construct a graph.
    with ops.Graph().as_default():
      tf_val_orig = array_ops.concat(
          [[1, 2, 16, 37],
           array_ops.placeholder(dtypes.int32, shape=(1,)), [48]], 0)

      # begin: no end
      tf_val = tf_val_orig[2:]
      c_val = tensor_util.constant_value_as_shape(tf_val)
      self.assertEqual([16, 37, None, 48], c_val.as_list())

      # begin::negative slice
      tf_val = tf_val_orig[2::-1]
      c_val = tensor_util.constant_value_as_shape(tf_val)
      self.assertEqual([16, 2, 1], c_val.as_list())

      # :end:negative slice
      tf_val = tf_val_orig[:1:-2]
      c_val = tensor_util.constant_value_as_shape(tf_val)
      self.assertEqual([48, 37], c_val.as_list())

      # begin:end:negative slice
      tf_val = tf_val_orig[3:1:-1]
      c_val = tensor_util.constant_value_as_shape(tf_val)
      self.assertEqual([37, 16], c_val.as_list())

      # begin:negative end:slice
      tf_val = tf_val_orig[1:-3:1]
      c_val = tensor_util.constant_value_as_shape(tf_val)
      self.assertEqual([2, 16], c_val.as_list())

      # negative begin::slice
      tf_val = tf_val_orig[-3::1]
      c_val = tensor_util.constant_value_as_shape(tf_val)
      self.assertEqual([37, None, 48], c_val.as_list())

      # negative begin::negative slice
      tf_val = tf_val_orig[-3::-1]
      c_val = tensor_util.constant_value_as_shape(tf_val)
      self.assertEqual([37, 16, 2, 1], c_val.as_list())

      # negative begin:negative end:negative slice
      tf_val = tf_val_orig[-3:-5:-1]
      c_val = tensor_util.constant_value_as_shape(tf_val)
      self.assertEqual([37, 16], c_val.as_list())

      # Do not support shape inference for additional arguments
      tf_val = constant_op.constant([10, 20, 30])[...]
      c_val = tensor_util.constant_value_as_shape(tf_val)
      self.assertEqual([None, None, None], c_val.as_list())

      # Do not support shape inference for tensor slices.
      tf_val = constant_op.constant(
          [10, 20, 30])[array_ops.placeholder(dtypes.int32, shape=()):]
      c_val = tensor_util.constant_value_as_shape(tf_val)
      self.assertEqual(tensor_shape.unknown_shape(), c_val)

      # Do not support shape inference for higher rank
      with self.assertRaises(ValueError):
        tf_val = constant_op.constant([[10], [20], [30]])[:, 0:]
        c_val = tensor_util.constant_value_as_shape(tf_val)


class MaybeSetStaticShapeTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the helper that propagates static shape information while building graphs. Compares
  # the operations produced with and without the propagation enabled and relies on a context
  # manager that toggles a module level flag.

  @contextlib.contextmanager
  def disableSetStaticShape(self):
    # [FUNCTION DOC]
    # HELPER: context manager for the tests in this class.
    # Saves the module level enable flag of the shape utility module, turns it off for the duration
    # of the with block, and restores the saved value afterwards even if the block raises. Yields
    # nothing. Side effect: temporarily mutates global state in the shape utility module.
    flag_old = shape_util._ENABLE_MAYBE_SET_STATIC_SHAPE
    shape_util._ENABLE_MAYBE_SET_STATIC_SHAPE = False
    try:
      yield
    finally:
      shape_util._ENABLE_MAYBE_SET_STATIC_SHAPE = flag_old

  def testMaybeSetStaticShape(self):
    # [FUNCTION DOC]
    # In a graph, builds a function graph that creates ten zeros and reshapes them using a constant
    # shape tensor, once with propagation disabled and once enabled. Expects both graphs to contain
    # the same multiset of operation types, meaning propagation does not add or remove operations.
    shape = constant_op.constant([2, 5], dtype=dtypes.int32)

    def reshape():
      # [FUNCTION DOC]
      # HELPER: graph-building body for the test. Creates a vector of ten zeros and returns it
      # reshaped with the enclosing constant shape tensor.
      v = array_ops.zeros([10])
      return array_ops.reshape(v, shape)
    # This test needs a placeholder which means we need to construct a graph.
    with ops.Graph().as_default():
      with self.disableSetStaticShape():
        graph_without_shape_propagation = func_graph.func_graph_from_py_func(
            "without_shape_propagation", reshape, [], {})
      graph_with_shape_propagation = func_graph.func_graph_from_py_func(
          "with_shape_propagation", reshape, [], {})
      self.assertCountEqual(
          [op.type for op in graph_without_shape_propagation.get_operations()],
          [op.type for op in graph_with_shape_propagation.get_operations()])

  def testMaybeSetStaticShapeScalarShape(self):
    # [FUNCTION DOC]
    # Builds function graphs from a body that reshapes a float placeholder to a flat vector, with
    # propagation disabled and enabled. Expects both graphs to contain the same multiset of
    # operation types.

    def reshape():
      # [FUNCTION DOC]
      # HELPER: graph-building body for the test. Creates a float32 placeholder, reshapes it to a flat
      # vector and returns the result.
      v = array_ops.placeholder(dtypes.float32)
      t = array_ops.reshape(v, [-1])
      return t

    with self.disableSetStaticShape():
      graph_without_shape_propagation = func_graph.func_graph_from_py_func(
          "without_shape_propagation", reshape, [], {})
    graph_with_shape_propagation = func_graph.func_graph_from_py_func(
        "with_shape_propagation", reshape, [], {})
    self.assertCountEqual(
        [op.type for op in graph_without_shape_propagation.get_operations()],
        [op.type for op in graph_with_shape_propagation.get_operations()])


class ShapeTensorTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for converting a tensor shape to a tensor, derived from the framework base test case.

  @test_util.run_in_graph_and_eager_modes
  def testConversion(self):
    # [FUNCTION DOC]
    # Builds a fully known shape from a plain integer and a dimension object, converts it with the
    # shape-to-tensor helper, and expects the tensor to equal the values one and two.
    """Make sure fully known TensorShape objects convert to Tensors."""
    shape = tensor_shape.TensorShape([1, tensor_shape.Dimension(2)])
    shape_tensor = shape_util.shape_tensor(shape)
    self.assertAllEqual((1, 2), shape_tensor)


if __name__ == "__main__":
  test.main()
