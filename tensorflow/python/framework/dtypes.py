# [MODULE DOC]
# Defines the DType class, which represents the element type of tensors, and the canonical
# module level instances for every supported type (floating point, integer, complex, string,
# boolean, quantized, low-bit and 8-bit float, resource, variant, and their reference variants).
# Also provides lookup tables that map between TensorFlow type enumeration values, type names,
# Python types and NumPy types, the numeric range table, and the function that converts nearly
# any type-like value into the interned instance. Import-time side effects: registers the DType
# class as serializable with the trace type machinery, attaches documentation to each constant,
# exports the constants to the public API namespace, and builds the global lookup tables (an
# assertion checks that the tables have no key collisions).
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
"""Library of dtypes (Tensor element types)."""
import abc
import builtins
import dataclasses
from typing import Type, Sequence, Optional

import ml_dtypes
import numpy as np

from tensorflow.core.framework import types_pb2
# We need to import pywrap_tensorflow prior to the bfloat wrapper to avoid
# protobuf errors where a file is defined twice on MacOS.
# pylint: disable=invalid-import-order,g-bad-import-order
from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
from tensorflow.python.framework import _dtypes
from tensorflow.python.framework import cpp_shape_inference_pb2
from tensorflow.python.types import doc_typealias
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.types import trace
from tensorflow.core.function import trace_type
from tensorflow.tools.docs import doc_controls


class DTypeMeta(type(_dtypes.DType), abc.ABCMeta):
  # [CLASS DOC]
  # HELPER: metaclass glue so the DType class can combine the native base type's metaclass with
  # the abstract base class metaclass.
  # Empty metaclass that merges the metaclass of the native DType base with the abstract base
  # class metaclass, needed because DType inherits from both kinds of parents.
  pass


@dataclasses.dataclass(frozen=True)
class HandleData:
  # [CLASS DOC]
  # Immutable (frozen) record holding extra information carried by resource and variant dtypes:
  # an optional shape inference result for the handle and an optional integer alias identifier.
  # Both fields default to nothing. Used as the optional handle data of DType.
  """Holds resource/variant tensor specific data."""
  shape_inference: Optional[
      cpp_shape_inference_pb2.CppShapeInferenceResult.HandleData
  ] = None
  alias_id: Optional[int] = None


@tf_export("dtypes.DType", "DType")
class DType(
    _dtypes.DType,
    trace.TraceType,
    trace_type.Serializable,
    metaclass=DTypeMeta):
  # [CLASS DOC]
  # Python-level tensor element type. Extends the native DType base with trace type behavior and
  # serialization support. Each instance wraps a type enumeration value and may carry optional
  # handle data (only meaningful for resource and variant types). Instances for the standard types
  # are interned in module tables; use the conversion function to obtain them. Provides
  # properties for reference/base types, numpy type, min/max values, compatibility checks,
  # equality by enumeration value, hashing inherited from the native base, pickling via name,
  # and protocol buffer conversion. Uses a restricted set of slots holding the handle data.
  """Represents the type of the elements in a `Tensor`.

  `DType`'s are used to specify the output data type for operations which
  require it, or to inspect the data type of existing `Tensor`'s.

  Examples:

  >>> tf.constant(1, dtype=tf.int64)
  <tf.Tensor: shape=(), dtype=int64, numpy=1>
  >>> tf.constant(1.0).dtype
  tf.float32

  See `tf.dtypes` for a complete list of `DType`'s defined.
  """
  __slots__ = ["_handle_data"]

  def __init__(self, type_enum, handle_data=None):
    # [FUNCTION DOC]
    # Constructor. Takes type_enum, the integer type enumeration value of the type, and optional
    # handle_data (nothing by default) which must be a handle data record if given. Initializes
    # the native base and stores the handle data. Raises a type error if handle_data is not
    # nothing and not a handle data record.
    super().__init__(type_enum)

    # Resource and Variant dtypes have additional handle data information that
    # is necessary for manipulating those Tensors.
    if handle_data is not None and not isinstance(handle_data, HandleData):
      raise TypeError("handle_data must be of the type HandleData")

    self._handle_data = handle_data

  @property
  def _is_ref_dtype(self):
    # [FUNCTION DOC]
    # HELPER: internal predicate for reference-type handling in the TF1 variable machinery.
    # Property returning true when this type is a reference type, determined by the enumeration
    # value being greater than 100. No side effects.
    """Returns `True` if this `DType` represents a reference type."""
    return self._type_enum > 100

  @property
  def _as_ref(self):
    # [FUNCTION DOC]
    # HELPER: internal accessor for reference-type handling.
    # Property returning the reference variant of this type: itself if already a reference type,
    # otherwise the interned instance whose enumeration value is 100 higher. Raises a key error
    # if no reference variant exists in the intern table.
    """Returns a reference `DType` based on this `DType`."""
    if self._is_ref_dtype:
      return self
    else:
      return _INTERN_TABLE[self._type_enum + 100]

  @property
  def base_dtype(self):
    # [FUNCTION DOC]
    # Property returning the non-reference type: the interned instance with enumeration value 100
    # lower when this is a reference type, otherwise this instance. Exists for TF1 compatibility.
    """Returns a non-reference `DType` based on this `DType` (for TF1).

    Programs written for TensorFlow 2.x do not need this attribute.
    It exists only for compatibility with TensorFlow 1.x, which used
    reference `DType`s in the implementation of `tf.compat.v1.Variable`.
    In TensorFlow 2.x, `tf.Variable` is implemented without reference types.
    """
    if self._is_ref_dtype:
      return _INTERN_TABLE[self._type_enum - 100]
    else:
      return self

  @property
  def real_dtype(self):
    # [FUNCTION DOC]
    # Property returning the type of the real part: 32-bit float for single precision complex,
    # 64-bit float for double precision complex (judged on the base type), otherwise the type
    # itself.
    """Returns the `DType` corresponding to this `DType`'s real part."""
    base = self.base_dtype
    if base == complex64:
      return float32
    elif base == complex128:
      return float64
    else:
      return self

  @property
  def as_numpy_dtype(self):
    # [FUNCTION DOC]
    # Property returning the NumPy scalar type matching this type, by looking up the enumeration
    # value in the module's table. Raises a key error for types without a NumPy equivalent.
    """Returns a Python `type` object based on this `DType`."""
    return _TF_TO_NP[self._type_enum]

  @property
  def min(self):
    # [FUNCTION DOC]
    # Property returning the smallest representable value of this numeric type. Raises a type
    # error for quantized, boolean, string and complex types, and also when neither the floating
    # point nor the integer limits query of the ml_dtypes library succeeds.
    # Uses broad exception handling when probing the limits, so unrelated errors are masked as
    # the same type error.
    """Returns the minimum representable value in this data type.

    Raises:
      TypeError: if this is a non-numeric, unordered, or quantized type.

    """
    if (self.is_quantized or
        self.base_dtype in (bool, string, complex64, complex128)):
      raise TypeError(f"Cannot find minimum value of {self} with "
                      f"{'quantized type' if self.is_quantized else 'type'} "
                      f"{self.base_dtype}.")

    # There is no simple way to get the min value of a dtype, we have to check
    # float and int types separately.
    try:
      return ml_dtypes.finfo(self.as_numpy_dtype).min
    except:  # bare except as possible raises by finfo not documented
      try:
        return ml_dtypes.iinfo(self.as_numpy_dtype).min
      except:
        raise TypeError(f"Cannot find minimum value of {self}.")

  @property
  def max(self):
    # [FUNCTION DOC]
    # Property returning the largest representable value of this numeric type. Raises a type
    # error for quantized, boolean, string and complex types, and also when neither the floating
    # point nor the integer limits query of the ml_dtypes library succeeds.
    # Uses broad exception handling when probing the limits, so unrelated errors are masked as
    # the same type error.
    """Returns the maximum representable value in this data type.

    Raises:
      TypeError: if this is a non-numeric, unordered, or quantized type.

    """
    if (self.is_quantized or
        self.base_dtype in (bool, string, complex64, complex128)):
      raise TypeError(f"Cannot find maximum value of {self} with "
                      f"{'quantized type' if self.is_quantized else 'type'} "
                      f"{self.base_dtype}.")

    # there is no simple way to get the max value of a dtype, we have to check
    # float and int types separately
    try:
      return ml_dtypes.finfo(self.as_numpy_dtype).max
    except:  # bare except as possible raises by finfo not documented
      try:
        return ml_dtypes.iinfo(self.as_numpy_dtype).max
      except:
        raise TypeError(f"Cannot find maximum value of {self}.")

  @property
  def limits(self, clip_negative=True):
    # [FUNCTION DOC]
    # KNOWN ISSUE: this is declared as a property, so it is always accessed without arguments and
    # clip_negative can never be changed from its default of true; the docstring describing the
    # argument is garbled. Because of the default, the lower limit is always forced to zero, even
    # for signed types, and floating point types report zero to one.
    # Property returning a (min, max) pair of intensity limits from the module's range table for
    # this type's NumPy type. Raises a value error if the type has no entry in the table.
    """Return intensity limits, i.e.

    (min, max) tuple, of the dtype.
    Args:
      clip_negative : bool, optional If True, clip the negative range (i.e.
        return 0 for min intensity) even if the image dtype allows negative
        values. Returns
      min, max : tuple Lower and upper intensity limits.
    """
    if self.as_numpy_dtype in dtype_range:
      min, max = dtype_range[self.as_numpy_dtype]  # pylint: disable=redefined-builtin
    else:
      raise ValueError(str(self) + " does not have defined limits.")

    if clip_negative:
      min = 0  # pylint: disable=redefined-builtin
    return min, max

  def is_compatible_with(self, other):
    # [FUNCTION DOC]
    # Returns true if a value of the other type would be converted to this type. Input other is
    # anything convertible to a type; it is converted first (type error if not convertible).
    # Compatible means the enumeration equals the other's enumeration or its base type's.
    """Returns True if the `other` DType will be converted to this DType (TF1).

    Programs written for TensorFlow 2.x do not need this function.
    Instead, they can do equality comparison on `DType` objects directly:
    `tf.as_dtype(this) == tf.as_dtype(other)`.

    This function exists only for compatibility with TensorFlow 1.x, where it
    additionally allows conversion from a reference type (used by
    `tf.compat.v1.Variable`) to its base type.

    Args:
      other: A `DType` (or object that may be converted to a `DType`).

    Returns:
      True if a Tensor of the `other` `DType` will be implicitly converted to
      this `DType`.
    """
    other = as_dtype(other)
    return self._type_enum in (other.as_datatype_enum,
                               other.base_dtype.as_datatype_enum)

  def is_subtype_of(self, other: trace.TraceType) -> bool:
    # [FUNCTION DOC]
    # Trace type protocol implementation: a type is a subtype of another only when they are equal.
    # Input other is a trace type; returns a boolean.
    """See tf.types.experimental.TraceType base class."""
    return self == other

  def most_specific_common_supertype(
      self, types: Sequence[trace.TraceType]) -> Optional["DType"]:
    # [FUNCTION DOC]
    # Trace type protocol implementation: given a sequence of trace types, returns this type if
    # all of them equal it, otherwise nothing (no common supertype exists).
    """See tf.types.experimental.TraceType base class."""
    return self if all(self == other for other in types) else None

  @doc_controls.do_not_doc_inheritable
  def placeholder_value(self, placeholder_context):
    # [FUNCTION DOC]
    # Trace type protocol: delegates to the parent implementation to build a placeholder value
    # from the given placeholder context. Hidden from generated documentation as inherited API.
    """See tf.types.experimental.TraceType base class."""
    return super().placeholder_value(placeholder_context)

  @doc_controls.do_not_doc_inheritable
  def from_tensors(self, tensors):
    # [FUNCTION DOC]
    # Trace type protocol: delegates to the parent implementation to rebuild a value from the
    # given tensors. Hidden from generated documentation as inherited API.
    """See tf.types.experimental.TraceType base class."""
    return super().from_tensors(tensors)

  @doc_controls.do_not_doc_inheritable
  def to_tensors(self, value):
    # [FUNCTION DOC]
    # Trace type protocol: delegates to the parent implementation to decompose the given value
    # into tensors. Hidden from generated documentation as inherited API.
    """See tf.types.experimental.TraceType base class."""
    return super().to_tensors(value)

  @doc_controls.do_not_doc_inheritable
  def flatten(self):
    # [FUNCTION DOC]
    # Trace type protocol: delegates to the parent implementation to return the flattened list of
    # component types. Hidden from generated documentation as inherited API.
    """See tf.types.experimental.TraceType base class."""
    return super().flatten()

  @doc_controls.do_not_doc_inheritable
  def cast(self, value, cast_context):
    # [FUNCTION DOC]
    # Trace type protocol: delegates to the parent implementation to cast the given value using
    # the given cast context. Hidden from generated documentation as inherited API.
    """See tf.types.experimental.TraceType base class."""
    return super().cast(value, cast_context)

  @classmethod
  def experimental_type_proto(cls) -> Type[types_pb2.SerializedDType]:
    # [FUNCTION DOC]
    # Class method for the serialization protocol: returns the protocol buffer message class used
    # to serialize types, namely the serialized dtype message. Takes no inputs besides the class.
    """Returns the type of proto associated with DType serialization."""
    return types_pb2.SerializedDType

  @classmethod
  def experimental_from_proto(cls, proto: types_pb2.SerializedDType) -> "DType":
    # [FUNCTION DOC]
    # Class method for deserialization: builds a new type from the datatype enumeration stored in
    # the given serialized dtype message. Note it creates a fresh instance (no handle data) and
    # does not return the interned instance.
    """Returns a Dtype instance based on the serialized proto."""
    return DType(proto.datatype)

  def experimental_as_proto(self) -> types_pb2.SerializedDType:
    # [FUNCTION DOC]
    # Serializes this type into a serialized dtype message carrying its enumeration value. Handle
    # data is not included.
    """Returns a proto representation of the Dtype instance."""
    return types_pb2.SerializedDType(datatype=self._type_enum)

  def __eq__(self, other):
    # [FUNCTION DOC]
    # Equality: false if other is nothing. If other is not a DType instance of exactly this class,
    # tries to convert it to a type and returns false when conversion raises a type error.
    # Otherwise compares enumeration values, so reference and non-reference types differ and
    # handle data is ignored.
    """Returns True iff this DType refers to the same type as `other`."""
    if other is None:
      return False

    if type(other) != DType:  # pylint: disable=unidiomatic-typecheck
      try:
        other = as_dtype(other)
      except TypeError:
        return False

    return self._type_enum == other._type_enum  # pylint: disable=protected-access

  def __ne__(self, other):
    # [FUNCTION DOC]
    # Inequality: the negation of the equality comparison with the other value.
    """Returns True iff self != other."""
    return not self.__eq__(other)

  # "If a class that overrides __eq__() needs to retain the implementation
  #  of __hash__() from a parent class, the interpreter must be told this
  #  explicitly by setting __hash__ = <ParentClass>.__hash__."
  # TODO(slebedev): Remove once __eq__ and __ne__ are implemented in C++.
  __hash__ = _dtypes.DType.__hash__

  def __reduce__(self):
    # [FUNCTION DOC]
    # Pickling support: returns the conversion function and a one-item argument tuple holding the
    # type's name, so unpickling yields the interned instance. Handle data is lost.
    return as_dtype, (self.name,)

trace_type.register_serializable(DType)

# Define data type range of numpy dtype
dtype_range = {
    np.bool_: (False, True),
    np.uint8: (0, 255),
    np.uint16: (0, 65535),
    np.int8: (-128, 127),
    np.int16: (-32768, 32767),
    np.int64: (-(2**63), 2**63 - 1),
    np.uint64: (0, 2**64 - 1),
    np.int32: (-(2**31), 2**31 - 1),
    np.uint32: (0, 2**32 - 1),
    np.float32: (-1, 1),
    np.float64: (-1, 1),
}

# Define standard wrappers for the types_pb2.DataType enum.
resource = DType(types_pb2.DT_RESOURCE)
doc_typealias.document(
    obj=resource, doc="Handle to a mutable, dynamically allocated resource."
)
tf_export("dtypes.resource", "resource").export_constant(__name__, "resource")

variant = DType(types_pb2.DT_VARIANT)
doc_typealias.document(
    obj=variant, doc="Data of arbitrary type (known at runtime)."
)
tf_export("dtypes.variant", "variant").export_constant(__name__, "variant")

uint8 = DType(types_pb2.DT_UINT8)
doc_typealias.document(obj=uint8, doc="Unsigned 8-bit (byte) integer.")
tf_export("dtypes.uint8", "uint8").export_constant(__name__, "uint8")

uint16 = DType(types_pb2.DT_UINT16)
doc_typealias.document(obj=uint16, doc="Unsigned 16-bit (word) integer.")
tf_export("dtypes.uint16", "uint16").export_constant(__name__, "uint16")

uint32 = DType(types_pb2.DT_UINT32)
doc_typealias.document(obj=uint32, doc="Unsigned 32-bit (dword) integer.")
tf_export("dtypes.uint32", "uint32").export_constant(__name__, "uint32")

uint64 = DType(types_pb2.DT_UINT64)
doc_typealias.document(obj=uint64, doc="Unsigned 64-bit (qword) integer.")
tf_export("dtypes.uint64", "uint64").export_constant(__name__, "uint64")

int8 = DType(types_pb2.DT_INT8)
doc_typealias.document(obj=int8, doc="Signed 8-bit integer.")
tf_export("dtypes.int8", "int8").export_constant(__name__, "int8")

int16 = DType(types_pb2.DT_INT16)
doc_typealias.document(obj=int16, doc="Signed 16-bit integer.")
tf_export("dtypes.int16", "int16").export_constant(__name__, "int16")

int32 = DType(types_pb2.DT_INT32)
doc_typealias.document(obj=int32, doc="Signed 32-bit integer.")
tf_export("dtypes.int32", "int32").export_constant(__name__, "int32")

int64 = DType(types_pb2.DT_INT64)
doc_typealias.document(obj=int64, doc="Signed 64-bit integer.")
tf_export("dtypes.int64", "int64").export_constant(__name__, "int64")

float16 = DType(types_pb2.DT_HALF)
half = float16
doc_typealias.document(
    obj=float16, doc="16-bit (half precision) floating-point."
)
tf_export("dtypes.float16", "float16").export_constant(__name__, "float16")
tf_export("dtypes.half", "half").export_constant(__name__, "half")

float32 = DType(types_pb2.DT_FLOAT)
doc_typealias.document(
    obj=float32, doc="32-bit (single precision) floating-point."
)
tf_export("dtypes.float32", "float32").export_constant(__name__, "float32")

float64 = DType(types_pb2.DT_DOUBLE)
doc_typealias.document(
    obj=float64, doc="64-bit (double precision) floating-point."
)
tf_export("dtypes.float64", "float64").export_constant(__name__, "float64")
double = float64
tf_export("dtypes.double", "double").export_constant(__name__, "double")

complex64 = DType(types_pb2.DT_COMPLEX64)
doc_typealias.document(obj=complex64, doc="64-bit complex.")
tf_export("dtypes.complex64", "complex64").export_constant(
    __name__, "complex64"
)

complex128 = DType(types_pb2.DT_COMPLEX128)
doc_typealias.document(obj=complex128, doc="128-bit complex.")
tf_export("dtypes.complex128", "complex128").export_constant(
    __name__, "complex128"
)

string = DType(types_pb2.DT_STRING)
doc_typealias.document(
    obj=string, doc="Variable-length string, represented as byte array."
)
tf_export("dtypes.string", "string").export_constant(__name__, "string")

bool = DType(types_pb2.DT_BOOL)  # pylint: disable=redefined-builtin
doc_typealias.document(obj=bool, doc="Boolean.")
tf_export("dtypes.bool", "bool").export_constant(__name__, "bool")

qint8 = DType(types_pb2.DT_QINT8)
doc_typealias.document(obj=qint8, doc="Signed quantized 8-bit integer.")
tf_export("dtypes.qint8", "qint8").export_constant(__name__, "qint8")

qint16 = DType(types_pb2.DT_QINT16)
doc_typealias.document(obj=qint16, doc="Signed quantized 16-bit integer.")
tf_export("dtypes.qint16", "qint16").export_constant(__name__, "qint16")

qint32 = DType(types_pb2.DT_QINT32)
doc_typealias.document(obj=qint32, doc="signed quantized 32-bit integer.")
tf_export("dtypes.qint32", "qint32").export_constant(__name__, "qint32")

quint8 = DType(types_pb2.DT_QUINT8)
doc_typealias.document(obj=quint8, doc="Unsigned quantized 8-bit integer.")
tf_export("dtypes.quint8", "quint8").export_constant(__name__, "quint8")

quint16 = DType(types_pb2.DT_QUINT16)
doc_typealias.document(obj=quint16, doc="Unsigned quantized 16-bit integer.")
tf_export("dtypes.quint16", "quint16").export_constant(__name__, "quint16")

bfloat16 = DType(types_pb2.DT_BFLOAT16)
doc_typealias.document(
    obj=bfloat16, doc="16-bit bfloat (brain floating point)."
)
tf_export("dtypes.bfloat16", "bfloat16").export_constant(__name__, "bfloat16")

float8_e5m2 = DType(types_pb2.DT_FLOAT8_E5M2)
doc_typealias.document(
    obj=float8_e5m2, doc="8-bit float with 5 exponent bits and 2 mantissa bits."
)
tf_export(
    "dtypes.experimental.float8_e5m2", "experimental.float8_e5m2"
).export_constant(__name__, "float8_e5m2")

float8_e4m3fn = DType(types_pb2.DT_FLOAT8_E4M3FN)
doc_typealias.document(
    obj=float8_e4m3fn,
    doc=(
        "8-bit float with 4 exponent bits and 3 mantissa bits, with extended"
        " finite range.  This type has no representation for inf, and only two"
        " NaN values: 0xFF for negative NaN, and 0x7F for positive NaN."
    ),
)
tf_export(
    "dtypes.experimental.float8_e4m3fn", "experimental.float8_e4m3fn"
).export_constant(__name__, "float8_e4m3fn")

float8_e4m3fnuz = DType(types_pb2.DT_FLOAT8_E4M3FNUZ)
doc_typealias.document(
    obj=float8_e4m3fnuz,
    doc=(
        "8-bit float with 4 exponent bits and 3 mantissa bits, with extended"
        " finite range.  This type has no representation for inf, and only one"
        " NaN value: 0x80."
    ),
)
tf_export(
    "dtypes.experimental.float8_e4m3fnuz", "experimental.float8_e4m3fnuz"
).export_constant(__name__, "float8_e4m3fnuz")

float8_e4m3b11fnuz = DType(types_pb2.DT_FLOAT8_E4M3B11FNUZ)
doc_typealias.document(
    obj=float8_e4m3b11fnuz,
    doc=(
        "8-bit float with 4 exponent bits and 3 mantissa bits, with extended "
        "finite range and 11 bits of bias.  This type has no representation "
        "for inf, and only one NaN value: 0x80."
    ),
)
tf_export(
    "dtypes.experimental.float8_e4m3b11fnuz", "experimental.float8_e4m3b11fnuz"
).export_constant(__name__, "float8_e4m3b11fnuz")

float8_e5m2fnuz = DType(types_pb2.DT_FLOAT8_E5M2FNUZ)
doc_typealias.document(
    obj=float8_e5m2fnuz,
    doc=(
        "8-bit float with 5 exponent bits and 2 mantissa bits, with extended "
        "finite range.  This type has no representation for inf, and only one "
        "NaN value: 0x80."
    ),
)
tf_export(
    "dtypes.experimental.float8_e5m2fnuz", "experimental.float8_e5m2fnuz"
).export_constant(__name__, "float8_e5m2fnuz")

float4_e2m1fn = DType(types_pb2.DT_FLOAT4_E2M1FN)
doc_typealias.document(
    obj=float4_e2m1fn,
    doc=(
        "4-bit float with 2 exponent bits and 1 mantissa bits, with extended"
        " finite range.  This type has no representation for both inf and NaN."
    ),
)
tf_export(
    "dtypes.experimental.float4_e2m1fn", "experimental.float4_e2m1fn"
).export_constant(__name__, "float4_e2m1fn")


int4 = DType(types_pb2.DT_INT4)
doc_typealias.document(obj=int4, doc="Signed 4-bit integer.")
tf_export("dtypes.experimental.int4", "experimental.int4").export_constant(
    __name__, "int4"
)

uint4 = DType(types_pb2.DT_UINT4)
doc_typealias.document(obj=uint4, doc="Unsigned 4-bit integer.")
tf_export("dtypes.experimental.uint4", "experimental.uint4").export_constant(
    __name__, "uint4"
)

int2 = DType(types_pb2.DT_INT2)
doc_typealias.document(obj=int2, doc="Signed 2-bit integer.")
tf_export("dtypes.experimental.int2", "experimental.int2").export_constant(
    __name__, "int2"
)

uint2 = DType(types_pb2.DT_UINT2)
doc_typealias.document(obj=uint2, doc="Unsigned 2-bit integer.")
tf_export("dtypes.experimental.uint2", "experimental.uint2").export_constant(
    __name__, "uint2"
)

resource_ref = DType(types_pb2.DT_RESOURCE_REF)
variant_ref = DType(types_pb2.DT_VARIANT_REF)
float16_ref = DType(types_pb2.DT_HALF_REF)
half_ref = float16_ref
float32_ref = DType(types_pb2.DT_FLOAT_REF)
float64_ref = DType(types_pb2.DT_DOUBLE_REF)
double_ref = float64_ref
int32_ref = DType(types_pb2.DT_INT32_REF)
uint32_ref = DType(types_pb2.DT_UINT32_REF)
uint8_ref = DType(types_pb2.DT_UINT8_REF)
uint16_ref = DType(types_pb2.DT_UINT16_REF)
int16_ref = DType(types_pb2.DT_INT16_REF)
int8_ref = DType(types_pb2.DT_INT8_REF)
string_ref = DType(types_pb2.DT_STRING_REF)
complex64_ref = DType(types_pb2.DT_COMPLEX64_REF)
complex128_ref = DType(types_pb2.DT_COMPLEX128_REF)
int64_ref = DType(types_pb2.DT_INT64_REF)
uint64_ref = DType(types_pb2.DT_UINT64_REF)
bool_ref = DType(types_pb2.DT_BOOL_REF)
qint8_ref = DType(types_pb2.DT_QINT8_REF)
quint8_ref = DType(types_pb2.DT_QUINT8_REF)
qint16_ref = DType(types_pb2.DT_QINT16_REF)
quint16_ref = DType(types_pb2.DT_QUINT16_REF)
qint32_ref = DType(types_pb2.DT_QINT32_REF)
bfloat16_ref = DType(types_pb2.DT_BFLOAT16_REF)
float8_e5m2_ref = DType(types_pb2.DT_FLOAT8_E5M2_REF)
float8_e4m3fn_ref = DType(types_pb2.DT_FLOAT8_E4M3FN_REF)
float8_e4m3fnuz_ref = DType(types_pb2.DT_FLOAT8_E4M3FNUZ_REF)
float8_e4m3b11fnuz_ref = DType(types_pb2.DT_FLOAT8_E4M3B11FNUZ_REF)
float8_e5m2fnuz_ref = DType(types_pb2.DT_FLOAT8_E5M2FNUZ_REF)
float4_e2m1fn_ref = DType(types_pb2.DT_FLOAT4_E2M1FN_REF)
int4_ref = DType(types_pb2.DT_INT4_REF)
uint4_ref = DType(types_pb2.DT_UINT4_REF)
int2_ref = DType(types_pb2.DT_INT2_REF)
uint2_ref = DType(types_pb2.DT_UINT2_REF)

# Maintain an intern table so that we don't have to create a large
# number of small objects.
_INTERN_TABLE = {
    types_pb2.DT_HALF: float16,
    types_pb2.DT_FLOAT: float32,
    types_pb2.DT_DOUBLE: float64,
    types_pb2.DT_INT32: int32,
    types_pb2.DT_UINT8: uint8,
    types_pb2.DT_UINT16: uint16,
    types_pb2.DT_UINT32: uint32,
    types_pb2.DT_UINT64: uint64,
    types_pb2.DT_INT16: int16,
    types_pb2.DT_INT8: int8,
    types_pb2.DT_STRING: string,
    types_pb2.DT_COMPLEX64: complex64,
    types_pb2.DT_COMPLEX128: complex128,
    types_pb2.DT_INT64: int64,
    types_pb2.DT_BOOL: bool,
    types_pb2.DT_QINT8: qint8,
    types_pb2.DT_QUINT8: quint8,
    types_pb2.DT_QINT16: qint16,
    types_pb2.DT_QUINT16: quint16,
    types_pb2.DT_QINT32: qint32,
    types_pb2.DT_BFLOAT16: bfloat16,
    types_pb2.DT_FLOAT8_E5M2: float8_e5m2,
    types_pb2.DT_FLOAT8_E4M3FN: float8_e4m3fn,
    types_pb2.DT_FLOAT8_E4M3FNUZ: float8_e4m3fnuz,
    types_pb2.DT_FLOAT8_E4M3B11FNUZ: float8_e4m3b11fnuz,
    types_pb2.DT_FLOAT8_E5M2FNUZ: float8_e5m2fnuz,
    types_pb2.DT_FLOAT4_E2M1FN: float4_e2m1fn,
    types_pb2.DT_INT4: int4,
    types_pb2.DT_UINT4: uint4,
    types_pb2.DT_INT2: int2,
    types_pb2.DT_UINT2: uint2,
    types_pb2.DT_RESOURCE: resource,
    types_pb2.DT_VARIANT: variant,
    types_pb2.DT_HALF_REF: float16_ref,
    types_pb2.DT_FLOAT_REF: float32_ref,
    types_pb2.DT_DOUBLE_REF: float64_ref,
    types_pb2.DT_INT32_REF: int32_ref,
    types_pb2.DT_UINT32_REF: uint32_ref,
    types_pb2.DT_UINT8_REF: uint8_ref,
    types_pb2.DT_UINT16_REF: uint16_ref,
    types_pb2.DT_INT16_REF: int16_ref,
    types_pb2.DT_INT8_REF: int8_ref,
    types_pb2.DT_STRING_REF: string_ref,
    types_pb2.DT_COMPLEX64_REF: complex64_ref,
    types_pb2.DT_COMPLEX128_REF: complex128_ref,
    types_pb2.DT_INT64_REF: int64_ref,
    types_pb2.DT_UINT64_REF: uint64_ref,
    types_pb2.DT_BOOL_REF: bool_ref,
    types_pb2.DT_QINT8_REF: qint8_ref,
    types_pb2.DT_QUINT8_REF: quint8_ref,
    types_pb2.DT_QINT16_REF: qint16_ref,
    types_pb2.DT_QUINT16_REF: quint16_ref,
    types_pb2.DT_QINT32_REF: qint32_ref,
    types_pb2.DT_BFLOAT16_REF: bfloat16_ref,
    types_pb2.DT_FLOAT8_E5M2_REF: float8_e5m2_ref,
    types_pb2.DT_FLOAT8_E4M3FN_REF: float8_e4m3fn_ref,
    types_pb2.DT_FLOAT8_E4M3FNUZ_REF: float8_e4m3fnuz_ref,
    types_pb2.DT_FLOAT8_E4M3B11FNUZ_REF: float8_e4m3b11fnuz_ref,
    types_pb2.DT_FLOAT8_E5M2FNUZ_REF: float8_e5m2fnuz_ref,
    types_pb2.DT_FLOAT4_E2M1FN_REF: float4_e2m1fn_ref,
    types_pb2.DT_INT4_REF: int4_ref,
    types_pb2.DT_UINT4_REF: uint4_ref,
    types_pb2.DT_INT2_REF: int2_ref,
    types_pb2.DT_UINT2_REF: uint2_ref,
    types_pb2.DT_RESOURCE_REF: resource_ref,
    types_pb2.DT_VARIANT_REF: variant_ref,
}

# Standard mappings between types_pb2.DataType values and string names.
_TYPE_TO_STRING = {
    types_pb2.DT_HALF: "float16",
    types_pb2.DT_FLOAT: "float32",
    types_pb2.DT_DOUBLE: "float64",
    types_pb2.DT_INT32: "int32",
    types_pb2.DT_UINT8: "uint8",
    types_pb2.DT_UINT16: "uint16",
    types_pb2.DT_UINT32: "uint32",
    types_pb2.DT_UINT64: "uint64",
    types_pb2.DT_INT16: "int16",
    types_pb2.DT_INT8: "int8",
    types_pb2.DT_STRING: "string",
    types_pb2.DT_COMPLEX64: "complex64",
    types_pb2.DT_COMPLEX128: "complex128",
    types_pb2.DT_INT64: "int64",
    types_pb2.DT_BOOL: "bool",
    types_pb2.DT_QINT8: "qint8",
    types_pb2.DT_QUINT8: "quint8",
    types_pb2.DT_QINT16: "qint16",
    types_pb2.DT_QUINT16: "quint16",
    types_pb2.DT_QINT32: "qint32",
    types_pb2.DT_BFLOAT16: "bfloat16",
    types_pb2.DT_FLOAT8_E5M2: "float8_e5m2",
    types_pb2.DT_FLOAT8_E4M3FN: "float8_e4m3fn",
    types_pb2.DT_FLOAT8_E4M3FNUZ: "float8_e4m3fnuz",
    types_pb2.DT_FLOAT8_E4M3B11FNUZ: "float8_e4m3b11fnuz",
    types_pb2.DT_FLOAT8_E5M2FNUZ: "float8_e5m2fnuz",
    types_pb2.DT_FLOAT4_E2M1FN: "float4_e2m1fn",
    types_pb2.DT_INT4: "int4",
    types_pb2.DT_UINT4: "uint4",
    types_pb2.DT_INT2: "int2",
    types_pb2.DT_UINT2: "uint2",
    types_pb2.DT_RESOURCE: "resource",
    types_pb2.DT_VARIANT: "variant",
    types_pb2.DT_HALF_REF: "float16_ref",
    types_pb2.DT_FLOAT_REF: "float32_ref",
    types_pb2.DT_DOUBLE_REF: "float64_ref",
    types_pb2.DT_INT32_REF: "int32_ref",
    types_pb2.DT_UINT32_REF: "uint32_ref",
    types_pb2.DT_UINT8_REF: "uint8_ref",
    types_pb2.DT_UINT16_REF: "uint16_ref",
    types_pb2.DT_INT16_REF: "int16_ref",
    types_pb2.DT_INT8_REF: "int8_ref",
    types_pb2.DT_STRING_REF: "string_ref",
    types_pb2.DT_COMPLEX64_REF: "complex64_ref",
    types_pb2.DT_COMPLEX128_REF: "complex128_ref",
    types_pb2.DT_INT64_REF: "int64_ref",
    types_pb2.DT_UINT64_REF: "uint64_ref",
    types_pb2.DT_BOOL_REF: "bool_ref",
    types_pb2.DT_QINT8_REF: "qint8_ref",
    types_pb2.DT_QUINT8_REF: "quint8_ref",
    types_pb2.DT_QINT16_REF: "qint16_ref",
    types_pb2.DT_QUINT16_REF: "quint16_ref",
    types_pb2.DT_QINT32_REF: "qint32_ref",
    types_pb2.DT_BFLOAT16_REF: "bfloat16_ref",
    types_pb2.DT_FLOAT8_E5M2_REF: "float8_e5m2_ref",
    types_pb2.DT_FLOAT8_E4M3FN_REF: "float8_e4m3fn_ref",
    types_pb2.DT_FLOAT8_E4M3FNUZ_REF: "float8_e4m3fnuz_ref",
    types_pb2.DT_FLOAT8_E4M3B11FNUZ_REF: "float8_e4m3b11fnuz_ref",
    types_pb2.DT_FLOAT8_E5M2FNUZ_REF: "float8_e5m2fnuz_ref",
    types_pb2.DT_FLOAT4_E2M1FN_REF: "float4_e2m1fn_ref",
    types_pb2.DT_INT4_REF: "int4_ref",
    types_pb2.DT_UINT4_REF: "uint4_ref",
    types_pb2.DT_INT2_REF: "int2_ref",
    types_pb2.DT_UINT2_REF: "uint2_ref",
    types_pb2.DT_RESOURCE_REF: "resource_ref",
    types_pb2.DT_VARIANT_REF: "variant_ref",
}
_STRING_TO_TF = {
    value: _INTERN_TABLE[key] for key, value in _TYPE_TO_STRING.items()
}
# Add non-canonical aliases.
_STRING_TO_TF["half"] = float16
_STRING_TO_TF["half_ref"] = float16_ref
_STRING_TO_TF["float"] = float32
_STRING_TO_TF["float_ref"] = float32_ref
_STRING_TO_TF["double"] = float64
_STRING_TO_TF["double_ref"] = float64_ref

# Numpy representation for quantized dtypes.
#
# These are magic strings that are used in the swig wrapper to identify
# quantized types.
# TODO(mrry,keveman): Investigate Numpy type registration to replace this
# hard-coding of names.
_np_qint8 = np.dtype([("qint8", np.int8)])
_np_quint8 = np.dtype([("quint8", np.uint8)])
_np_qint16 = np.dtype([("qint16", np.int16)])
_np_quint16 = np.dtype([("quint16", np.uint16)])
_np_qint32 = np.dtype([("qint32", np.int32)])

# _np_bfloat16, _np_float8*, _np_(u)int4 are defined by module imports.
_np_bfloat16 = ml_dtypes.bfloat16
_np_float8_e4m3fn = ml_dtypes.float8_e4m3fn
_np_float8_e4m3fnuz = ml_dtypes.float8_e4m3fnuz
_np_float8_e4m3b11fnuz = ml_dtypes.float8_e4m3b11fnuz
_np_float8_e5m2 = ml_dtypes.float8_e5m2
_np_float8_e5m2fnuz = ml_dtypes.float8_e5m2fnuz
_np_float4_e2m1fn = ml_dtypes.float4_e2m1fn
_np_int4 = ml_dtypes.int4
_np_uint4 = ml_dtypes.uint4
_np_int2 = ml_dtypes.int2
_np_uint2 = ml_dtypes.uint2

# Custom struct dtype for directly-fed ResourceHandles of supported type(s).
np_resource = np.dtype([("resource", np.ubyte)])

# Standard mappings between types_pb2.DataType values and numpy.dtypes.
_NP_TO_TF = {
    np.float16: float16,
    np.float32: float32,
    np.float64: float64,
    np.int32: int32,
    np.int64: int64,
    np.uint8: uint8,
    np.uint16: uint16,
    np.uint32: uint32,
    np.uint64: uint64,
    np.int16: int16,
    np.int8: int8,
    np.complex64: complex64,
    np.complex128: complex128,
    np.object_: string,
    np.bytes_: string,
    np.str_: string,
    np.bool_: bool,
    _np_qint8: qint8,
    _np_quint8: quint8,
    _np_qint16: qint16,
    _np_quint16: quint16,
    _np_qint32: qint32,
    _np_bfloat16: bfloat16,
    _np_float8_e5m2: float8_e5m2,
    _np_float8_e4m3fn: float8_e4m3fn,
    _np_float8_e4m3fnuz: float8_e4m3fnuz,
    _np_float8_e4m3b11fnuz: float8_e4m3b11fnuz,
    _np_float8_e5m2fnuz: float8_e5m2fnuz,
    _np_float4_e2m1fn: float4_e2m1fn,
    _np_int4: int4,
    _np_uint4: uint4,
    _np_int2: int2,
    _np_uint2: uint2,
}

# Map (some) NumPy platform dtypes to TF ones using their fixed-width
# synonyms. Note that platform dtypes are not always simples aliases,
# i.e. reference equality is not guaranteed. See e.g. numpy/numpy#9799.
for pdt in [
    np.intc,
    np.uintc,
    np.int_,
    np.uint,
    np.longlong,
    np.ulonglong,
]:
  if pdt not in _NP_TO_TF:
    _NP_TO_TF[pdt] = next(
        _NP_TO_TF[dt] for dt in _NP_TO_TF if dt == pdt().dtype)  # pylint: disable=no-value-for-parameter

if hasattr(np.dtypes, "StringDType"):
  _NP_TO_TF[np.dtypes.StringDType()] = string

TF_VALUE_DTYPES = set(_NP_TO_TF.values())

_TF_TO_NP = {
    types_pb2.DT_HALF: np.float16,
    types_pb2.DT_FLOAT: np.float32,
    types_pb2.DT_DOUBLE: np.float64,
    types_pb2.DT_INT32: np.int32,
    types_pb2.DT_UINT8: np.uint8,
    types_pb2.DT_UINT16: np.uint16,
    types_pb2.DT_UINT32: np.uint32,
    types_pb2.DT_UINT64: np.uint64,
    types_pb2.DT_INT16: np.int16,
    types_pb2.DT_INT8: np.int8,
    # NOTE(touts): For strings we use object as it supports variable length
    # strings.
    types_pb2.DT_STRING: object,
    types_pb2.DT_COMPLEX64: np.complex64,
    types_pb2.DT_COMPLEX128: np.complex128,
    types_pb2.DT_INT64: np.int64,
    types_pb2.DT_BOOL: np.bool_,
    types_pb2.DT_QINT8: _np_qint8,
    types_pb2.DT_QUINT8: _np_quint8,
    types_pb2.DT_QINT16: _np_qint16,
    types_pb2.DT_QUINT16: _np_quint16,
    types_pb2.DT_QINT32: _np_qint32,
    types_pb2.DT_BFLOAT16: _np_bfloat16,
    types_pb2.DT_FLOAT8_E5M2: _np_float8_e5m2,
    types_pb2.DT_FLOAT8_E4M3FN: _np_float8_e4m3fn,
    types_pb2.DT_FLOAT8_E4M3FNUZ: _np_float8_e4m3fnuz,
    types_pb2.DT_FLOAT8_E4M3B11FNUZ: _np_float8_e4m3b11fnuz,
    types_pb2.DT_FLOAT8_E5M2FNUZ: _np_float8_e5m2fnuz,
    types_pb2.DT_FLOAT4_E2M1FN: _np_float4_e2m1fn,
    types_pb2.DT_INT4: _np_int4,
    types_pb2.DT_UINT4: _np_uint4,
    types_pb2.DT_INT2: _np_int2,
    types_pb2.DT_UINT2: _np_uint2,
    # Ref types
    types_pb2.DT_HALF_REF: np.float16,
    types_pb2.DT_FLOAT_REF: np.float32,
    types_pb2.DT_DOUBLE_REF: np.float64,
    types_pb2.DT_INT32_REF: np.int32,
    types_pb2.DT_UINT32_REF: np.uint32,
    types_pb2.DT_UINT8_REF: np.uint8,
    types_pb2.DT_UINT16_REF: np.uint16,
    types_pb2.DT_INT16_REF: np.int16,
    types_pb2.DT_INT8_REF: np.int8,
    types_pb2.DT_STRING_REF: np.object_,
    types_pb2.DT_COMPLEX64_REF: np.complex64,
    types_pb2.DT_COMPLEX128_REF: np.complex128,
    types_pb2.DT_INT64_REF: np.int64,
    types_pb2.DT_UINT64_REF: np.uint64,
    types_pb2.DT_BOOL_REF: np.bool_,
    types_pb2.DT_QINT8_REF: _np_qint8,
    types_pb2.DT_QUINT8_REF: _np_quint8,
    types_pb2.DT_QINT16_REF: _np_qint16,
    types_pb2.DT_QUINT16_REF: _np_quint16,
    types_pb2.DT_QINT32_REF: _np_qint32,
    types_pb2.DT_BFLOAT16_REF: _np_bfloat16,
    types_pb2.DT_FLOAT8_E5M2_REF: _np_float8_e5m2,
    types_pb2.DT_FLOAT8_E4M3FN_REF: _np_float8_e4m3fn,
    types_pb2.DT_FLOAT8_E4M3FNUZ_REF: _np_float8_e4m3fnuz,
    types_pb2.DT_FLOAT8_E4M3B11FNUZ_REF: _np_float8_e4m3b11fnuz,
    types_pb2.DT_FLOAT8_E5M2FNUZ_REF: _np_float8_e5m2fnuz,
    types_pb2.DT_FLOAT4_E2M1FN_REF: _np_float4_e2m1fn,
    types_pb2.DT_INT4_REF: _np_int4,
    types_pb2.DT_UINT4_REF: _np_uint4,
    types_pb2.DT_INT2_REF: _np_int2,
    types_pb2.DT_UINT2_REF: _np_uint2,
}

_QUANTIZED_DTYPES_NO_REF = frozenset([qint8, quint8, qint16, quint16, qint32])
_QUANTIZED_DTYPES_REF = frozenset(
    [qint8_ref, quint8_ref, qint16_ref, quint16_ref, qint32_ref])
QUANTIZED_DTYPES = _QUANTIZED_DTYPES_REF.union(_QUANTIZED_DTYPES_NO_REF)
tf_export(
    "dtypes.QUANTIZED_DTYPES",
    v1=["dtypes.QUANTIZED_DTYPES",
        "QUANTIZED_DTYPES"]).export_constant(__name__, "QUANTIZED_DTYPES")

_PYTHON_TO_TF = {
    builtins.float: float32,
    builtins.bool: bool,
    builtins.object: string
}

_ANY_TO_TF = {}
_ANY_TO_TF.update(_INTERN_TABLE)
_ANY_TO_TF.update(_STRING_TO_TF)
_ANY_TO_TF.update(_PYTHON_TO_TF)
_ANY_TO_TF.update(_NP_TO_TF)

# Ensure no collisions.
assert len(_ANY_TO_TF) == sum(
    len(d) for d in [_INTERN_TABLE, _STRING_TO_TF, _PYTHON_TO_TF, _NP_TO_TF])


@tf_export("dtypes.as_dtype", "as_dtype")
def as_dtype(type_value):
  # [FUNCTION DOC]
  # Converts a type-like value into the interned DType. Input type_value may be a DType, a type
  # enumeration integer, a type name string, a Python or NumPy type, a NumPy dtype, or an object
  # with a dtype attribute. Lookup order: a DType without handle data maps to its interned
  # instance (one with handle data is returned unchanged); a NumPy dtype is tried through its
  # scalar type; then the combined lookup table (unhashable values are skipped); then the dtype
  # attribute; then any native DType base instance. Returns the DType. Raises a type error when
  # no conversion applies. No side effects.
  """Converts the given `type_value` to a `tf.DType`.

  Inputs can be existing `tf.DType` objects, a [`DataType`
  enum](https://www.tensorflow.org/code/tensorflow/core/framework/types.proto),
  a string type name, or a
  [`numpy.dtype`](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html).

  Examples:
  >>> tf.as_dtype(2)  # Enum value for float64.
  tf.float64

  >>> tf.as_dtype('float')
  tf.float32

  >>> tf.as_dtype(np.int32)
  tf.int32

  Note: `DType` values are interned (i.e. a single instance of each dtype is
  stored in a map). When passed a new `DType` object, `as_dtype` always returns
  the interned value.

  Args:
    type_value: A value that can be converted to a `tf.DType` object.

  Returns:
    A `DType` corresponding to `type_value`.

  Raises:
    TypeError: If `type_value` cannot be converted to a `DType`.
  """
  if isinstance(type_value, DType):
    if type_value._handle_data is None:  # pylint:disable=protected-access
      return _INTERN_TABLE[type_value.as_datatype_enum]
    else:
      return type_value

  if isinstance(type_value, np.dtype):
    try:
      return _NP_TO_TF[type_value.type]
    except KeyError:
      pass

  try:
    return _ANY_TO_TF[type_value]
  except (KeyError, TypeError):
    # TypeError indicates that type_value is not hashable.
    pass

  if hasattr(type_value, "dtype"):
    try:
      return _NP_TO_TF[np.dtype(type_value.dtype).type]
    except (KeyError, TypeError):
      pass

  if isinstance(type_value, _dtypes.DType):
    return _INTERN_TABLE[type_value.as_datatype_enum]

  raise TypeError(f"Cannot convert the argument `type_value`: {type_value!r} "
                  "to a TensorFlow DType.")
