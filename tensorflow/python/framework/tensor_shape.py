# [MODULE DOC]
# HELPER: shape-inference support module used across the framework (tensors, ops, functions).
# Defines the TensorShape class (static shape of a tensor, possibly partially or fully unknown)
# and the legacy TF1 Dimension class, plus conversion helpers (as_dimension, as_shape,
# unknown_shape, dimension_value, dimension_at_index) and the functions that toggle V1 or V2
# indexing/iteration behavior of TensorShape.
# Import-time side effects: creates the API-usage gauge metric, registers TensorShape as a
# serializable trace type, and registers a codec for TensorShape with the nested structure coder.
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
"""Helper classes for tensor shape inference."""
import functools
import operator
from typing import Optional, Sequence, Type, Union

from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.function import trace_type
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python import tf2
from tensorflow.python.eager import monitoring
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import trace
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls

_TENSORSHAPE_V2_OVERRIDE = None

_api_usage_gauge = monitoring.BoolGauge(
    "/tensorflow/api/v2_tensorshape",
    "Whether tensor_shape.enable_v2_tensorshape() is called.")


@tf_export(v1=["enable_v2_tensorshape"])
def enable_v2_tensorshape():
  # [FUNCTION DOC]
  # Switches the process-wide TensorShape behavior to V2 semantics (v1-only exported API).
  # Takes no inputs and returns nothing. Side effects: sets the module-level override flag to true so
  # indexing and iterating a TensorShape yields plain integers or null values instead of Dimension
  # objects, writes a verbose-level log message, and sets the API-usage gauge metric to true.
  """In TensorFlow 2.0, iterating over a TensorShape instance returns values.

  This enables the new behavior.

  Concretely, `tensor_shape[i]` returned a Dimension instance in V1, but
  it V2 it returns either an integer, or None.

  Examples:

  ```
  #######################
  # If you had this in V1:
  value = tensor_shape[i].value

  # Do this in V2 instead:
  value = tensor_shape[i]

  #######################
  # If you had this in V1:
  for dim in tensor_shape:
    value = dim.value
    print(value)

  # Do this in V2 instead:
  for value in tensor_shape:
    print(value)

  #######################
  # If you had this in V1:
  dim = tensor_shape[i]
  dim.assert_is_compatible_with(other_shape)  # or using any other shape method

  # Do this in V2 instead:
  if tensor_shape.rank is None:
    dim = Dimension(None)
  else:
    dim = tensor_shape.dims[i]
  dim.assert_is_compatible_with(other_shape)  # or using any other shape method

  # The V2 suggestion above is more explicit, which will save you from
  # the following trap (present in V1):
  # you might do in-place modifications to `dim` and expect them to be reflected
  # in `tensor_shape[i]`, but they would not be.
  ```
  """
  global _TENSORSHAPE_V2_OVERRIDE  # pylint: disable=invalid-name
  _TENSORSHAPE_V2_OVERRIDE = True
  logging.vlog(1, "Enabling v2 tensorshape")
  _api_usage_gauge.get_cell().set(True)


@tf_export(v1=["disable_v2_tensorshape"])
def disable_v2_tensorshape():
  # [FUNCTION DOC]
  # Switches the process-wide TensorShape behavior back to V1 semantics (v1-only exported API).
  # Takes no inputs and returns nothing. Side effects: sets the module-level override flag to false
  # (forcing V1 even if TF2 behavior is enabled), logs a verbose message and sets the API-usage
  # gauge metric to false.
  """Disables the V2 TensorShape behavior and reverts to V1 behavior.

  See docstring for `enable_v2_tensorshape` for details about the new behavior.
  """
  global _TENSORSHAPE_V2_OVERRIDE  # pylint: disable=invalid-name
  _TENSORSHAPE_V2_OVERRIDE = False
  logging.vlog(1, "Disabling v2 tensorshape")
  _api_usage_gauge.get_cell().set(False)


@tf_export(
    "compat.dimension_value", v1=["dimension_value", "compat.dimension_value"]
)
def dimension_value(
    dimension: Union["Dimension", int, None]
) -> Union[int, None]:
  # [FUNCTION DOC]
  # Compatibility bridge that extracts a plain value from a dimension-like input.
  # Input: a Dimension object, an integer, or a null value. Output: the Dimension's value (an integer
  # or null) when given a Dimension, otherwise the input is returned unchanged. No errors, no side
  # effects.
  """Compatibility utility required to allow for both V1 and V2 behavior in TF.

  Until the release of TF 2.0, we need the legacy behavior of `TensorShape` to
  coexist with the new behavior. This utility is a bridge between the two.

  When accessing the value of a TensorShape dimension,
  use this utility, like this:

  ```
  # If you had this in your V1 code:
  value = tensor_shape[i].value

  # Use `dimension_value` as direct replacement compatible with both V1 & V2:
  value = dimension_value(tensor_shape[i])

  # This would be the V2 equivalent:
  value = tensor_shape[i]  # Warning: this will return the dim value in V2!
  ```

  Args:
    dimension: Either a `Dimension` instance, an integer, or None.

  Returns:
    A plain value, i.e. an integer or None.
  """
  if isinstance(dimension, Dimension):
    return dimension.value
  return dimension


@tf_export(
    "compat.dimension_at_index",
    v1=["dimension_at_index", "compat.dimension_at_index"])
def dimension_at_index(shape, index) -> "Dimension":
  # [FUNCTION DOC]
  # Compatibility bridge that returns the Dimension object at a given position of a TensorShape.
  # Inputs: shape (must be a TensorShape, enforced with an assertion, so other types raise an
  # assertion error) and index (integer or slice used on the dimension list). Output: an unknown
  # Dimension when the shape has unknown rank, otherwise the Dimension at that index of the shape's
  # dimension list. An out-of-range index raises an index error. No side effects.
  """Compatibility utility required to allow for both V1 and V2 behavior in TF.

  Until the release of TF 2.0, we need the legacy behavior of `TensorShape` to
  coexist with the new behavior. This utility is a bridge between the two.

  If you want to retrieve the Dimension instance corresponding to a certain
  index in a TensorShape instance, use this utility, like this:

  ```
  # If you had this in your V1 code:
  dim = tensor_shape[i]

  # Use `dimension_at_index` as direct replacement compatible with both V1 & V2:
  dim = dimension_at_index(tensor_shape, i)

  # Another possibility would be this, but WARNING: it only works if the
  # tensor_shape instance has a defined rank.
  dim = tensor_shape.dims[i]  # `dims` may be None if the rank is undefined!

  # In native V2 code, we recommend instead being more explicit:
  if tensor_shape.rank is None:
    dim = Dimension(None)
  else:
    dim = tensor_shape.dims[i]

  # Being more explicit will save you from the following trap (present in V1):
  # you might do in-place modifications to `dim` and expect them to be reflected
  # in `tensor_shape[i]`, but they would not be (as the Dimension object was
  # instantiated on the fly.
  ```

  Args:
    shape: A TensorShape instance.
    index: An integer index.

  Returns:
    A dimension object.
  """
  assert isinstance(shape, TensorShape)
  if shape.rank is None:
    return Dimension(None)
  else:
    return shape.dims[index]


@tf_export(v1=["Dimension"])
class Dimension(object):
  # [CLASS DOC]
  # Legacy (TF1) wrapper for the size of one tensor dimension; the size is either a non-negative
  # integer or unknown (null). Holds a single private value slot (restricted slots, no instance
  # dict). Instances are created from integers, null, other Dimensions or integer-like objects, and
  # support arithmetic, comparison and compatibility/merging operations in which an unknown operand
  # generally yields an unknown (or null) result. Exported only in the v1 API; TF2 uses plain
  # integers inside TensorShape instead.
  """Represents the value of one dimension in a TensorShape.

  @compatibility(TF2)
  In TF2, members of a `TensorShape` object are integers. The `Dimension` class
  is not part of TF2's data model.

  Please refer to the [TensorShape section of the migration guide]
  (https://www.tensorflow.org/guide/migrate/index#tensorshape) on common code
  patterns adapting Dimension objects to a TF2 syntax.
  @end_compatibility
  """

  __slots__ = ["_value"]

  def __init__(self, value):
    # [FUNCTION DOC]
    # Constructor. Input: value, which may be a non-negative integer, null (unknown size), another
    # Dimension (its value is copied), or any object providing an integer-conversion method.
    # Raises a value error for negative values and a type error when the object is not integer-like.
    # Side effect: stores the normalized value (integer or null) in the instance.
    """Creates a new Dimension with the given value."""
    if isinstance(value, int):  # Most common case.
      if value < 0:
        raise ValueError("Dimension %d must be >= 0" % value)
      self._value = value
    elif value is None:
      self._value = None
    elif isinstance(value, Dimension):
      self._value = value._value
    else:
      try:
        # int(...) compensates for the int/long dichotomy on Python 2.X.
        # TODO(b/143206389): Remove once we fully migrate to 3.X.
        self._value = int(value.__index__())
      except AttributeError:
        raise TypeError(
            "Dimension value must be integer or None or have "
            "an __index__ method, got value '{0!r}' with type '{1!r}'".format(
                value, type(value))) from None
      if self._value < 0:
        raise ValueError("Dimension %d must be >= 0" % self._value)

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns a debug string of the form Dimension(value), where value is shown with its own
    # representation (the integer or the null value). No side effects.
    return "Dimension(%s)" % repr(self._value)

  def __str__(self):
    # [FUNCTION DOC]
    # Returns the human-readable text of the dimension: a question mark when unknown, otherwise the
    # decimal integer. No side effects.
    value = self._value
    return "?" if value is None else str(value)

  def __eq__(self, other):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it returns true when the values match, but when either side is
    # unknown it returns a null value (neither true nor false), so equality is three-valued.
    # Equality comparison. Input: other, converted to a Dimension; if conversion fails with a type or
    # value error the not-implemented marker is returned so Python can try the reflected operation.
    # Returns null if either value is unknown, otherwise whether the two integers are equal.
    """Returns true if `other` has the same known value as this Dimension."""
    try:
      other = as_dimension(other)
    except (TypeError, ValueError):
      return NotImplemented
    if self._value is None or other.value is None:
      return None
    return self._value == other.value

  def __ne__(self, other):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it returns true when values differ, but if either dimension is
    # unknown it returns a null value instead of a boolean.
    # Inequality comparison. Input: other, converted to a Dimension; on a type or value error the
    # not-implemented marker is returned. Returns null if either value is unknown, otherwise whether
    # the integers differ.
    """Returns true if `other` has a different known value from `self`."""
    try:
      other = as_dimension(other)
    except (TypeError, ValueError):
      return NotImplemented
    if self._value is None or other.value is None:
      return None
    return self._value != other.value

  def __bool__(self):
    # [FUNCTION DOC]
    # Truth-value test. Returns the boolean truth of the stored value: false for zero and for unknown
    # (null), true for any positive size. No side effects.
    """Equivalent to `bool(self.value)`."""
    return bool(self._value)

  def __int__(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: for an unknown dimension this returns null, which makes the integer-conversion
    # protocol fail with a type error rather than yielding an integer.
    # Integer conversion: returns the stored value.
    return self._value

  # This is needed for Windows.
  # See https://github.com/tensorflow/tensorflow/pull/9780
  def __long__(self):
    # [FUNCTION DOC]
    # Legacy long-integer conversion kept for Windows/Python 2 compatibility; returns the stored value
    # (null for an unknown dimension, which will fail in the caller's conversion).
    return self._value

  def __index__(self):
    # [FUNCTION DOC]
    # Index-protocol conversion so a Dimension can be used where an integer index is needed (for
    # example in ranges). Returns the stored value; for an unknown dimension returns null, which makes
    # the caller raise a type error.
    # Allow use in Python 3 range
    return self._value

  @property
  def value(self):
    # [FUNCTION DOC]
    # Read-only property giving the dimension size as an integer, or null when unknown.
    """The value of this dimension, or None if it is unknown."""
    return self._value

  # TODO(b/225058047): Reconsider semantics.
  def is_compatible_with(self, other):
    # [FUNCTION DOC]
    # Checks whether this dimension and another can denote the same size.
    # Input: other, anything accepted by the dimension conversion helper (may raise a type or value
    # error if not convertible). Returns true when either is unknown or both known values are equal,
    # otherwise false. No side effects.
    """Returns true if `other` is compatible with this Dimension.

    Two known Dimensions are compatible if they have the same value.
    An unknown Dimension is compatible with all other Dimensions.

    Args:
      other: Another Dimension.

    Returns:
      True if this Dimension and `other` are compatible.
    """
    other = as_dimension(other)
    return (self._value is None or other.value is None or
            self._value == other.value)

  def assert_is_compatible_with(self, other):
    # [FUNCTION DOC]
    # Assertion form of the compatibility check. Input: other, converted to a Dimension. Returns
    # nothing; raises a value error stating both dimensions when they are not compatible.
    """Raises an exception if `other` is not compatible with this Dimension.

    Args:
      other: Another Dimension.

    Raises:
      ValueError: If `self` and `other` are not compatible (see
        is_compatible_with).
    """
    if not self.is_compatible_with(other):
      raise ValueError("Dimensions %s and %s are not compatible" %
                       (self, other))

  def merge_with(self, other):
    # [FUNCTION DOC]
    # Combines the information of two dimensions. Input: other, converted to a Dimension.
    # Returns a new Dimension: the known value if either side is known, unknown if both are unknown.
    # Raises a value error (via the compatibility assertion) when both are known and differ.
    """Returns a Dimension that combines the information in `self` and `other`.

    Dimensions are combined as follows:

    ```python
    tf.compat.v1.Dimension(n)   .merge_with(tf.compat.v1.Dimension(n))     ==
    tf.compat.v1.Dimension(n)
    tf.compat.v1.Dimension(n)   .merge_with(tf.compat.v1.Dimension(None))  ==
    tf.compat.v1.Dimension(n)
    tf.compat.v1.Dimension(None).merge_with(tf.compat.v1.Dimension(n))     ==
    tf.compat.v1.Dimension(n)
    # equivalent to tf.compat.v1.Dimension(None)
    tf.compat.v1.Dimension(None).merge_with(tf.compat.v1.Dimension(None))

    # raises ValueError for n != m
    tf.compat.v1.Dimension(n)   .merge_with(tf.compat.v1.Dimension(m))
    ```

    Args:
      other: Another Dimension.

    Returns:
      A Dimension containing the combined information of `self` and
      `other`.

    Raises:
      ValueError: If `self` and `other` are not compatible (see
        is_compatible_with).
    """
    other = as_dimension(other)
    self.assert_is_compatible_with(other)
    if self._value is None:
      return Dimension(other.value)
    else:
      return Dimension(self._value)

  def __add__(self, other):
    # [FUNCTION DOC]
    # Addition. Input: other, converted to a Dimension; if conversion fails with a type or value error
    # the not-implemented marker is returned. Returns a new Dimension holding the sum, or an unknown
    # Dimension if either operand is unknown.
    """Returns the sum of `self` and `other`.

    Dimensions are summed as follows:

    ```python
    tf.compat.v1.Dimension(m)    + tf.compat.v1.Dimension(n)     ==
    tf.compat.v1.Dimension(m + n)
    tf.compat.v1.Dimension(m)    + tf.compat.v1.Dimension(None)  # equiv. to
    tf.compat.v1.Dimension(None)
    tf.compat.v1.Dimension(None) + tf.compat.v1.Dimension(n)     # equiv. to
    tf.compat.v1.Dimension(None)
    tf.compat.v1.Dimension(None) + tf.compat.v1.Dimension(None)  # equiv. to
    tf.compat.v1.Dimension(None)
    ```

    Args:
      other: Another Dimension, or a value accepted by `as_dimension`.

    Returns:
      A Dimension whose value is the sum of `self` and `other`.
    """
    try:
      other = as_dimension(other)
    except (TypeError, ValueError):
      return NotImplemented
    if self._value is None or other.value is None:
      return Dimension(None)
    else:
      return Dimension(self._value + other.value)

  def __radd__(self, other):
    # [FUNCTION DOC]
    # Reflected addition (other plus self); simply delegates to addition and returns a Dimension (or the
    # not-implemented marker if other cannot be converted).
    """Returns the sum of `other` and `self`.

    Args:
      other: Another Dimension, or a value accepted by `as_dimension`.

    Returns:
      A Dimension whose value is the sum of `self` and `other`.
    """
    return self + other

  def __sub__(self, other):
    # [FUNCTION DOC]
    # Subtraction of other from self. Input: other, converted to a Dimension (conversion failure gives
    # the not-implemented marker). Returns a new Dimension with the difference, unknown if either side
    # is unknown. Note a negative known result raises a value error from the Dimension constructor.
    """Returns the subtraction of `other` from `self`.

    Dimensions are subtracted as follows:

    ```python
    tf.compat.v1.Dimension(m)    - tf.compat.v1.Dimension(n)     ==
    tf.compat.v1.Dimension(m - n)
    tf.compat.v1.Dimension(m)    - tf.compat.v1.Dimension(None)  # equiv. to
    tf.compat.v1.Dimension(None)
    tf.compat.v1.Dimension(None) - tf.compat.v1.Dimension(n)     # equiv. to
    tf.compat.v1.Dimension(None)
    tf.compat.v1.Dimension(None) - tf.compat.v1.Dimension(None)  # equiv. to
    tf.compat.v1.Dimension(None)
    ```

    Args:
      other: Another Dimension, or a value accepted by `as_dimension`.

    Returns:
      A Dimension whose value is the subtraction of `other` from `self`.
    """
    try:
      other = as_dimension(other)
    except (TypeError, ValueError):
      return NotImplemented
    if self._value is None or other.value is None:
      return Dimension(None)
    else:
      return Dimension(self._value - other.value)

  def __rsub__(self, other):
    # [FUNCTION DOC]
    # Reflected subtraction (other minus self). Input: other, converted to a Dimension; unlike the
    # forward form, conversion errors propagate as type or value errors. Returns an unknown Dimension
    # if either operand is unknown, otherwise a Dimension with the difference (negative results raise
    # a value error).
    """Returns the subtraction of `self` from `other`.

    Args:
      other: Another Dimension, or a value accepted by `as_dimension`.

    Returns:
      A Dimension whose value is the subtraction of `self` from `other`.
    """
    other = as_dimension(other)
    if self._value is None or other.value is None:
      return Dimension(None)
    else:
      return Dimension(other.value - self._value)

  def __mul__(self, other):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says dimensions are "summed" but this method multiplies them.
    # Multiplication. Input: other, converted to a Dimension (conversion failure gives the
    # not-implemented marker). Returns a new Dimension with the product, or unknown if either side is
    # unknown.
    """Returns the product of `self` and `other`.

    Dimensions are summed as follows:

    ```python
    tf.compat.v1.Dimension(m)    * tf.compat.v1.Dimension(n)     ==
    tf.compat.v1.Dimension(m * n)
    tf.compat.v1.Dimension(m)    * tf.compat.v1.Dimension(None)  # equiv. to
    tf.compat.v1.Dimension(None)
    tf.compat.v1.Dimension(None) * tf.compat.v1.Dimension(n)     # equiv. to
    tf.compat.v1.Dimension(None)
    tf.compat.v1.Dimension(None) * tf.compat.v1.Dimension(None)  # equiv. to
    tf.compat.v1.Dimension(None)
    ```

    Args:
      other: Another Dimension, or a value accepted by `as_dimension`.

    Returns:
      A Dimension whose value is the product of `self` and `other`.
    """
    try:
      other = as_dimension(other)
    except (TypeError, ValueError):
      return NotImplemented

    if self._value is None or other.value is None:
      return Dimension(None)
    else:
      return Dimension(self._value * other.value)

  def __rmul__(self, other):
    # [FUNCTION DOC]
    # Reflected multiplication; delegates to multiplication and returns a Dimension (or the
    # not-implemented marker).
    """Returns the product of `self` and `other`.

    Args:
      other: Another Dimension, or a value accepted by `as_dimension`.

    Returns:
      A Dimension whose value is the product of `self` and `other`.
    """
    return self * other

  def __floordiv__(self, other):
    # [FUNCTION DOC]
    # Floor division of self by other. Input: other, converted to a Dimension (conversion failure gives
    # the not-implemented marker). Returns a Dimension with the rounded-down quotient, or unknown if
    # either side is unknown. Dividing by a known zero raises a division-by-zero error.
    """Returns the quotient of `self` and `other` rounded down.

    Dimensions are divided as follows:

    ```python
    tf.compat.v1.Dimension(m)    // tf.compat.v1.Dimension(n)     ==
    tf.compat.v1.Dimension(m // n)
    tf.compat.v1.Dimension(m)    // tf.compat.v1.Dimension(None)  # equiv. to
    tf.compat.v1.Dimension(None)
    tf.compat.v1.Dimension(None) // tf.compat.v1.Dimension(n)     # equiv. to
    tf.compat.v1.Dimension(None)
    tf.compat.v1.Dimension(None) // tf.compat.v1.Dimension(None)  # equiv. to
    tf.compat.v1.Dimension(None)
    ```

    Args:
      other: Another Dimension, or a value accepted by `as_dimension`.

    Returns:
      A `Dimension` whose value is the integer quotient of `self` and `other`.
    """
    try:
      other = as_dimension(other)
    except (TypeError, ValueError):
      return NotImplemented
    if self._value is None or other.value is None:
      return Dimension(None)
    else:
      return Dimension(self._value // other.value)

  def __rfloordiv__(self, other):
    # [FUNCTION DOC]
    # Reflected floor division (other divided by self). Input: other, converted to a Dimension
    # (conversion errors propagate). Returns an unknown Dimension when either is unknown, otherwise the
    # rounded-down quotient; a known zero divisor raises a division-by-zero error.
    """Returns the quotient of `other` and `self` rounded down.

    Args:
      other: Another Dimension, or a value accepted by `as_dimension`.

    Returns:
      A `Dimension` whose value is the integer quotient of `self` and `other`.
    """
    other = as_dimension(other)
    if self._value is None or other.value is None:
      return Dimension(None)
    else:
      return Dimension(other.value // self._value)

  def __div__(self, other):
    # [FUNCTION DOC]
    # Deprecated legacy division operator; delegates to floor division and returns its result.
    # Input: other, another Dimension or convertible value.
    """DEPRECATED: Use `__floordiv__` via `x // y` instead.

    This function exists only for backwards compatibility purposes; new code
    should use `__floordiv__` via the syntax `x // y`.  Using `x // y`
    communicates clearly that the result rounds down, and is forward compatible
    to Python 3.

    Args:
      other: Another `Dimension`.

    Returns:
      A `Dimension` whose value is the integer quotient of `self` and `other`.
    """
    return self // other

  def __rdiv__(self, other):
    # [FUNCTION DOC]
    # Reflected true-division guard. Always raises a type error whose message names the left operand
    # type and asks the caller to use floor division. Input: other (only used in the message).
    """Use `__floordiv__` via `x // y` instead.

    This function exists only to have a better error message. Instead of:
    `TypeError: unsupported operand type(s) for /: 'int' and 'Dimension'`,
    this function will explicitly call for usage of `//` instead.

    Args:
      other: Another `Dimension`.

    Raises:
      TypeError.
    """
    raise TypeError("unsupported operand type(s) for /: '{}' and 'Dimension', "
                    "please use // instead".format(type(other).__name__))

  def __truediv__(self, other):
    # [FUNCTION DOC]
    # True-division guard. Always raises a type error whose message names the right operand type and
    # asks the caller to use floor division. Input: other (only used in the message).
    """Use `__floordiv__` via `x // y` instead.

    This function exists only to have a better error message. Instead of:
    `TypeError: unsupported operand type(s) for /: 'Dimension' and 'int'`,
    this function will explicitly call for usage of `//` instead.

    Args:
      other: Another `Dimension`.

    Raises:
      TypeError.
    """
    raise TypeError("unsupported operand type(s) for /: 'Dimension' and '{}', "
                    "please use // instead".format(type(other).__name__))

  def __rtruediv__(self, other):
    # [FUNCTION DOC]
    # Reflected true-division guard. Always raises a type error saying that dividing a value of the
    # other's type by a Dimension is unsupported and to use floor division instead.
    """Use `__floordiv__` via `x // y` instead.

    This function exists only to have a better error message. Instead of:
    `TypeError: unsupported operand type(s) for /: 'int' and 'Dimension'`,
    this function will explicitly call for usage of `//` instead.

    Args:
      other: Another `Dimension`.

    Raises:
      TypeError.
    """
    raise TypeError("unsupported operand type(s) for /: '{}' and 'Dimension', "
                    "please use // instead".format(type(other).__name__))

  def __mod__(self, other):
    # [FUNCTION DOC]
    # Modulo of self by other. Input: other, converted to a Dimension (conversion errors propagate).
    # Returns an unknown Dimension if either side is unknown, otherwise a Dimension with the
    # remainder; a known zero divisor raises a division-by-zero error.
    """Returns `self` modulo `other`.

    Dimension modulo are computed as follows:

    ```python
    tf.compat.v1.Dimension(m)    % tf.compat.v1.Dimension(n)     ==
    tf.compat.v1.Dimension(m % n)
    tf.compat.v1.Dimension(m)    % tf.compat.v1.Dimension(None)  # equiv. to
    tf.compat.v1.Dimension(None)
    tf.compat.v1.Dimension(None) % tf.compat.v1.Dimension(n)     # equiv. to
    tf.compat.v1.Dimension(None)
    tf.compat.v1.Dimension(None) % tf.compat.v1.Dimension(None)  # equiv. to
    tf.compat.v1.Dimension(None)
    ```

    Args:
      other: Another Dimension, or a value accepted by `as_dimension`.

    Returns:
      A Dimension whose value is `self` modulo `other`.
    """
    other = as_dimension(other)
    if self._value is None or other.value is None:
      return Dimension(None)
    else:
      return Dimension(self._value % other.value)

  def __rmod__(self, other):
    # [FUNCTION DOC]
    # Reflected modulo (other modulo self). Input: other, converted to a Dimension. Converts other and
    # delegates to the modulo of that Dimension by self; returns a Dimension.
    """Returns `other` modulo `self`.

    Args:
      other: Another Dimension, or a value accepted by `as_dimension`.

    Returns:
      A Dimension whose value is `other` modulo `self`.
    """
    other = as_dimension(other)
    return other % self

  def __lt__(self, other):
    # [FUNCTION DOC]
    # Less-than comparison. Input: other, converted to a Dimension (conversion errors propagate).
    # Returns null when either value is unknown, otherwise a boolean.
    """Returns True if `self` is known to be less than `other`.

    Dimensions are compared as follows:

    ```python
    (tf.compat.v1.Dimension(m)    < tf.compat.v1.Dimension(n))    == (m < n)
    (tf.compat.v1.Dimension(m)    < tf.compat.v1.Dimension(None)) == None
    (tf.compat.v1.Dimension(None) < tf.compat.v1.Dimension(n))    == None
    (tf.compat.v1.Dimension(None) < tf.compat.v1.Dimension(None)) == None
    ```

    Args:
      other: Another Dimension.

    Returns:
      The value of `self.value < other.value` if both are known, otherwise
      None.
    """
    other = as_dimension(other)
    if self._value is None or other.value is None:
      return None
    else:
      return self._value < other.value

  def __le__(self, other):
    # [FUNCTION DOC]
    # Less-than-or-equal comparison. Input: other, converted to a Dimension (errors propagate).
    # Returns null when either value is unknown, otherwise a boolean.
    """Returns True if `self` is known to be less than or equal to `other`.

    Dimensions are compared as follows:

    ```python
    (tf.compat.v1.Dimension(m)    <= tf.compat.v1.Dimension(n))    == (m <= n)
    (tf.compat.v1.Dimension(m)    <= tf.compat.v1.Dimension(None)) == None
    (tf.compat.v1.Dimension(None) <= tf.compat.v1.Dimension(n))    == None
    (tf.compat.v1.Dimension(None) <= tf.compat.v1.Dimension(None)) == None
    ```

    Args:
      other: Another Dimension.

    Returns:
      The value of `self.value <= other.value` if both are known, otherwise
      None.
    """
    other = as_dimension(other)
    if self._value is None or other.value is None:
      return None
    else:
      return self._value <= other.value

  def __gt__(self, other):
    # [FUNCTION DOC]
    # Greater-than comparison. Input: other, converted to a Dimension (errors propagate).
    # Returns null when either value is unknown, otherwise a boolean.
    """Returns True if `self` is known to be greater than `other`.

    Dimensions are compared as follows:

    ```python
    (tf.compat.v1.Dimension(m)    > tf.compat.v1.Dimension(n))    == (m > n)
    (tf.compat.v1.Dimension(m)    > tf.compat.v1.Dimension(None)) == None
    (tf.compat.v1.Dimension(None) > tf.compat.v1.Dimension(n))    == None
    (tf.compat.v1.Dimension(None) > tf.compat.v1.Dimension(None)) == None
    ```

    Args:
      other: Another Dimension.

    Returns:
      The value of `self.value > other.value` if both are known, otherwise
      None.
    """
    other = as_dimension(other)
    if self._value is None or other.value is None:
      return None
    else:
      return self._value > other.value

  def __ge__(self, other):
    # [FUNCTION DOC]
    # Greater-than-or-equal comparison. Input: other, converted to a Dimension (errors propagate).
    # Returns null when either value is unknown, otherwise a boolean.
    """Returns True if `self` is known to be greater than or equal to `other`.

    Dimensions are compared as follows:

    ```python
    (tf.compat.v1.Dimension(m)    >= tf.compat.v1.Dimension(n))    == (m >= n)
    (tf.compat.v1.Dimension(m)    >= tf.compat.v1.Dimension(None)) == None
    (tf.compat.v1.Dimension(None) >= tf.compat.v1.Dimension(n))    == None
    (tf.compat.v1.Dimension(None) >= tf.compat.v1.Dimension(None)) == None
    ```

    Args:
      other: Another Dimension.

    Returns:
      The value of `self.value >= other.value` if both are known, otherwise
      None.
    """
    other = as_dimension(other)
    if self._value is None or other.value is None:
      return None
    else:
      return self._value >= other.value

  def __reduce__(self):
    # [FUNCTION DOC]
    # Pickling support. Returns the pair (class, argument tuple holding the value) so that the
    # Dimension can be recreated by calling the constructor with its stored value.
    return Dimension, (self._value,)


def as_dimension(value):
  # [FUNCTION DOC]
  # Converts a value to a Dimension. A Dimension is returned unchanged (same object); null becomes
  # an unknown Dimension; an integer or integer-like object becomes a new Dimension. Errors from the
  # Dimension constructor (negative value, non-integer type) propagate.
  """Converts the given value to a Dimension.

  A Dimension input will be returned unmodified.
  An input of `None` will be converted to an unknown Dimension.
  An integer input will be converted to a Dimension with that value.

  Args:
    value: The value to be converted.

  Returns:
    A Dimension corresponding to the given value.
  """
  if isinstance(value, Dimension):
    return value
  else:
    return Dimension(value)


@tf_export("TensorShape")
class TensorShape(trace.TraceType, trace_type.Serializable):
  # [CLASS DOC]
  # Static shape of a Tensor: a list of dimension sizes (integers or unknown/null) or a completely
  # unknown shape (unknown rank). Stores a private tuple of sizes, or null for unknown rank (restricted
  # slots). Can be built from a list/tuple, a TensorShapeProto, another TensorShape, an iterable or a
  # single integer. Implements the trace-type interface (subtyping, supertype computation) and the
  # serializable interface, and offers merging, compatibility, rank assertions, concatenation and
  # conversion to list/proto. Whether indexing and iteration yield Dimension objects (V1) or plain
  # integers (V2) depends on the module override or TF2 mode. Registered for serialization at import.
  """Represents the shape of a `Tensor`.

  >>> t = tf.constant([[1,2,3],[4,5,6]])
  >>> t.shape
  TensorShape([2, 3])

  `TensorShape` is the *static* shape representation of a Tensor.
  During eager execution a Tensor always has a fully specified shape but
  when tracing a `tf.function` it may be one of the following:

  * *Fully-known shape:* has a known number of dimensions and a known size
    for each dimension. e.g. `TensorShape([16, 256])`
  * *Partially-known shape:* has a known number of dimensions, and an unknown
    size for one or more dimension. e.g. `TensorShape([None, 256])`
  * *Unknown shape:* has an unknown number of dimensions, and an unknown
    size in all dimensions. e.g. `TensorShape(None)`

  During function tracing `t.shape` will return a `TensorShape` object
  representing the shape of Tensor as it is known during tracing.
  This static representation will be partially defined in cases where the
  exact shape depends on the values within the tensors. To get the
  *dynamic* representation, please use `tf.shape(t)`
  which will return Tensor representing the fully defined shape of `t`.
  This way, you can express logic that manipulates the shapes of tensors by
  building other tensors that depend on the dynamic shape of `t`.

  Note: `tf.RaggedTensor.shape` also returns a `tf.TensorShape`,
  the lengths of any ragged dimensions are unknown (`None`).

  For example, this function prints the `TensorShape' (`t.shape`), when you
  trace the function, and returns a tensor `tf.shape(t)` for given input `t`:

  >>> @tf.function
  ... def get_dynamic_shape(t):
  ...   print("tracing...")
  ...   print(f"static shape is {t.shape}")
  ...   return tf.shape(t)

  Just calling the function traces it with a fully-specified static shape:

  >>> result = get_dynamic_shape(tf.constant([[1, 1, 1], [0, 0, 0]]))
  tracing...
  static shape is (2, 3)
  >>> result.numpy()
  array([2, 3], dtype=int32)

  But `tf.function` can also trace the function with a partially specified
  (or even unspecified) shape:

  >>> cf1 = get_dynamic_shape.get_concrete_function(tf.TensorSpec(
  ...                                               shape=[None, 2]))
  tracing...
  static shape is (None, 2)
  >>> cf1(tf.constant([[1., 0],[1, 0],[1, 0]])).numpy()
  array([3, 2], dtype=int32)

  >>> cf2 = get_dynamic_shape.get_concrete_function(tf.TensorSpec(shape=None))
  tracing...
  static shape is <unknown>
  >>> cf2(tf.constant([[[[[1., 0]]]]])).numpy()
  array([1, 1, 1, 1, 2], dtype=int32)

  If a tensor is produced by an operation of type `"Foo"`, its shape
  may be inferred if there is a registered shape function for
  `"Foo"`. See [Shape
  functions](https://www.tensorflow.org/guide/create_op#shape_functions_in_c)
  for details of shape functions and how to register them. Alternatively,
  you may set the shape explicitly using `tf.Tensor.ensure_shape`.
  """
  __slots__ = ["_dims"]

  def __init__(self, dims):
    # [FUNCTION DOC]
    # Constructor. Input: dims, one of: a list or tuple of dimension-like items, null (unknown rank), a
    # TensorShapeProto (unknown_rank flag gives unknown rank; size -1 means unknown dimension),
    # another TensorShape (its dimensions are shared), an iterable of dimension-like items, or a single
    # integer-like value treated as a one-dimensional shape.
    # Raises a type error when an element of an iterable cannot be converted (message names the
    # offending item), and value/type errors from dimension conversion (e.g. negative sizes).
    # Side effect: stores the dimensions as a tuple of integers or null values.
    """Creates a new TensorShape with the given dimensions.

    Args:
      dims: A list of Dimensions, or None if the shape is unspecified.

    Raises:
      TypeError: If dims cannot be converted to a list of dimensions.
    """
    if isinstance(dims, (tuple, list)):  # Most common case.
      self._dims = tuple(as_dimension(d).value for d in dims)
    elif dims is None:
      self._dims = None
    elif isinstance(dims, tensor_shape_pb2.TensorShapeProto):
      if dims.unknown_rank:
        self._dims = None
      else:
        self._dims = tuple(
            # Protos store variable-size dimensions as -1
            dim.size if dim.size != -1 else None
            for dim in dims.dim
            )
    elif isinstance(dims, TensorShape):
      self._dims = dims._dims
    else:
      try:
        dims_iter = iter(dims)
      except TypeError:
        # Treat as a singleton dimension
        self._dims = (as_dimension(dims).value,)
      else:
        self._dims = []
        for d in dims_iter:
          try:
            self._dims.append(as_dimension(d).value)
          except TypeError as e:
            raise TypeError(
                "Failed to convert '{0!r}' to a shape: '{1!r}'"
                "could not be converted to a dimension. A shape should "
                "either be single dimension (e.g. 10), or an iterable of "
                "dimensions (e.g. [1, 10, None]).".format(dims, d)) from e
        self._dims = tuple(self._dims)

  @property
  def _v2_behavior(self):
    # [FUNCTION DOC]
    # HELPER: internal switch used by repr, str, iteration and indexing to choose V1 or V2 output.
    # Read-only property. Returns the module-level override if one was set by the enable/disable
    # functions, otherwise whether TF2 behavior is currently enabled.
    if _TENSORSHAPE_V2_OVERRIDE is None:
      return tf2.enabled()
    return _TENSORSHAPE_V2_OVERRIDE

  def __repr__(self):
    # [FUNCTION DOC]
    # Debug representation. In V2 mode shows the list of integers/null values (or null for unknown
    # rank); in V1 mode shows the list of Dimension objects (or null when rank is unknown).
    if self._v2_behavior:
      if self._dims is not None:
        return f"TensorShape({list(self._dims)})"
      else:
        return "TensorShape(None)"
    else:
      return f"TensorShape({self.dims})"

  def __str__(self):
    # [FUNCTION DOC]
    # Human-readable text: the word unknown in angle brackets for unknown rank; a one-element shape is
    # shown as a parenthesized value followed by a comma; otherwise a parenthesized comma-separated
    # list, with unknown sizes shown as a question mark (V1 Dimensions) or the null word (V2 values).
    if self.rank is None:
      return "<unknown>"
    elif self.rank == 1:
      if self._v2_behavior:
        return "(%s,)" % self._dims[0]
      else:
        return "(%s,)" % self.dims[0]
    else:
      if self._v2_behavior:
        return "(%s)" % ", ".join(str(d) for d in self._dims)
      else:
        return "(%s)" % ", ".join(str(d) for d in self.dims)

  @property
  def rank(self):
    # [FUNCTION DOC]
    # Read-only property: the number of dimensions, or null if the rank is unknown.
    """Returns the rank of this shape, or None if it is unspecified."""
    if self._dims is not None:
      return len(self._dims)
    return None

  @property
  def dims(self):
    # [FUNCTION DOC]
    # Deprecated read-only property. Returns a new list of Dimension objects (one per dimension), or
    # null for unknown rank. Modifying the returned objects does not change the shape.
    """Deprecated.  Returns list of dimensions for this shape.

    Suggest `TensorShape.as_list` instead.

    Returns:
      A list containing `tf.compat.v1.Dimension`s, or None if the shape is
      unspecified.
    """
    if self._dims is None:
      return None
    return [as_dimension(d) for d in self._dims]

  @property
  def ndims(self):
    # [FUNCTION DOC]
    # Deprecated read-only alias of the rank property: number of dimensions or null if unknown.
    """Deprecated accessor for `rank`."""
    return self.rank

  def __len__(self):
    # [FUNCTION DOC]
    # Length protocol. Returns the rank; raises a value error if the rank is unknown.
    """Returns the rank of this shape, or raises ValueError if unspecified."""
    if self._dims is None:
      raise ValueError("Cannot take the length of shape with unknown rank.")
    return len(self._dims)

  def __bool__(self):
    # [FUNCTION DOC]
    # Truth-value test. Returns true unless the rank is unknown. Note a known scalar shape (rank 0) is
    # therefore still true.
    """Returns True if this shape contains non-zero information."""
    return self._dims is not None

  # Python 3 wants __bool__, Python 2.7 wants __nonzero__
  __nonzero__ = __bool__

  def __iter__(self):
    # [FUNCTION DOC]
    # Iteration protocol. Raises a value error when the rank is unknown. Otherwise returns an iterator
    # over plain integers/null values in V2 mode, or over Dimension objects in V1 mode.
    """Returns `self.dims` if the rank is known, otherwise raises ValueError."""
    if self._dims is None:
      raise ValueError("Cannot iterate over a shape with unknown rank.")
    else:
      if self._v2_behavior:
        return iter(d for d in self._dims)
      else:
        return iter(d for d in self.dims)

  def __getitem__(self, key):
    # [FUNCTION DOC]
    # Indexing. Input: key, an integer or a slice. For a known rank: a slice gives a new TensorShape of
    # the selected sizes; an integer gives the size (V2) or a Dimension (V1); out-of-range integers
    # raise an index error. For an unknown rank: an integer gives null (V2) or an unknown Dimension
    # (V1); a slice with a step raises a value error; a slice without an end, or with a negative start
    # or end, gives a fully unknown shape; otherwise an unknown-dimension shape of length end minus start.
    """Returns the value of a dimension or a shape, depending on the key.

    Args:
      key: If `key` is an integer, returns the dimension at that index;
        otherwise if `key` is a slice, returns a TensorShape whose dimensions
        are those selected by the slice from `self`.

    Returns:
      An integer if `key` is an integer, or a `TensorShape` if `key` is a
      slice.

    Raises:
      ValueError: If `key` is a slice and `self` is completely unknown and
        the step is set.
    """
    if self._dims is not None:
      if isinstance(key, slice):
        return TensorShape(self._dims[key])
      else:
        if self._v2_behavior:
          return self._dims[key]
        else:
          return self.dims[key]
    else:
      if isinstance(key, slice):
        start = key.start if key.start is not None else 0
        stop = key.stop

        if key.step is not None:
          # TODO(mrry): Handle these maybe.
          raise ValueError("Steps are not yet handled")
        if stop is None:
          # NOTE(mrry): This implies that TensorShape(None) is compatible with
          # TensorShape(None)[1:], which is obviously not true. It would be
          # possible to track the number of dimensions symbolically,
          # and perhaps we should do that.
          return unknown_shape()
        elif start < 0 or stop < 0:
          # TODO(mrry): Handle this better, as it will be useful for handling
          # suffixes of otherwise unknown shapes.
          return unknown_shape()
        else:
          return unknown_shape(rank=stop - start)
      else:
        if self._v2_behavior:
          return None
        else:
          return Dimension(None)

  def num_elements(self):
    # [FUNCTION DOC]
    # Returns the product of all dimension sizes, or null if the shape is not fully defined (including
    # unknown rank). A scalar shape gives one. No side effects.
    """Returns the total number of elements, or none for incomplete shapes."""
    if self.is_fully_defined():
      return functools.reduce(operator.mul, self.as_list(), 1)
    else:
      return None

  def merge_with(self, other):
    # [FUNCTION DOC]
    # Combines the information of two shapes element-wise. Input: other, anything convertible to a
    # TensorShape. If one side has unknown rank the other is returned as is (possibly the same
    # object). Otherwise ranks must match and each pair of dimensions is merged. Raises a value error
    # naming both shapes when the ranks or any known dimensions conflict.
    """Returns a `TensorShape` combining the information in `self` and `other`.

    The dimensions in `self` and `other` are merged element-wise,
    according to the rules below:

    ```python
    Dimension(n).merge_with(Dimension(None)) == Dimension(n)
    Dimension(None).merge_with(Dimension(n)) == Dimension(n)
    Dimension(None).merge_with(Dimension(None)) == Dimension(None)
    # raises ValueError for n != m
    Dimension(n).merge_with(Dimension(m))
    ```
    >> ts = tf.TensorShape([1,2])
    >> ot1 = tf.TensorShape([1,2])
    >> ts.merge_with(ot).as_list()
    [1,2]

    >> ot2 = tf.TensorShape([1,None])
    >> ts.merge_with(ot2).as_list()
    [1,2]

    >> ot3 = tf.TensorShape([None, None])
    >> ot3.merge_with(ot2).as_list()
    [1, None]

    Args:
      other: Another `TensorShape`.

    Returns:
      A `TensorShape` containing the combined information of `self` and
      `other`.

    Raises:
      ValueError: If `self` and `other` are not compatible.
    """
    other = as_shape(other)
    if self.dims is None:
      return other
    if other.dims is None:
      return self
    else:
      try:
        self.assert_same_rank(other)
        new_dims = [
            dim.merge_with(other_dim)
            for dim, other_dim in zip(self.dims, other.dims)
        ]
        return TensorShape(new_dims)
      except ValueError:
        raise ValueError("Shapes %s and %s are not compatible" % (self, other))

  def __add__(self, other):
    # [FUNCTION DOC]
    # Addition operator; same as concatenation. Input: other, a TensorShape or convertible value.
    # Returns the concatenated TensorShape.
    return self.concatenate(other)

  def __radd__(self, other):
    # [FUNCTION DOC]
    # Reflected addition. Converts other to a TensorShape if needed (errors from conversion
    # propagate), then returns other concatenated with self.
    if not isinstance(other, TensorShape):
      other = TensorShape(other)
    return other.concatenate(self)

  def concatenate(self, other):
    # [FUNCTION DOC]
    # Appends the dimensions of other after those of self. Input: other, anything convertible to a
    # TensorShape. Returns a new TensorShape; if either shape has unknown rank the result is a fully
    # unknown shape, discarding the other's information.
    """Returns the concatenation of the dimension in `self` and `other`.

    *N.B.* If either `self` or `other` is completely unknown,
    concatenation will discard information about the other shape. In
    future, we might support concatenation that preserves this
    information for use with slicing.

    Args:
      other: Another `TensorShape`.

    Returns:
      A `TensorShape` whose dimensions are the concatenation of the
      dimensions in `self` and `other`.
    """
    # TODO(mrry): Handle the case where we concatenate a known shape with a
    # completely unknown shape, so that we can use the partial information.
    other = as_shape(other)
    if self.dims is None or other.dims is None:
      return unknown_shape()
    else:
      return TensorShape(self.dims + other.dims)

  def assert_same_rank(self, other):
    # [FUNCTION DOC]
    # Asserts the two shapes can have equal rank. Input: other, convertible to a TensorShape. Returns
    # nothing; raises a value error naming both shapes only when both ranks are known and differ.
    """Raises an exception if `self` and `other` do not have compatible ranks.

    Args:
      other: Another `TensorShape`.

    Raises:
      ValueError: If `self` and `other` do not represent shapes with the
        same rank.
    """
    other = as_shape(other)
    if self.rank is not None and other.rank is not None:
      if self.rank != other.rank:
        raise ValueError("Shapes %s and %s must have the same rank" %
                         (self, other))

  def assert_has_rank(self, rank):
    # [FUNCTION DOC]
    # Asserts that this shape can have the given rank. Input: rank, an integer. Returns nothing;
    # raises a value error if the rank is known and differs from the given rank. Unknown rank passes.
    """Raises an exception if `self` is not compatible with the given `rank`.

    Args:
      rank: An integer.

    Raises:
      ValueError: If `self` does not represent a shape with the given `rank`.
    """
    if self.rank not in (None, rank):
      raise ValueError("Shape %s must have rank %d" % (self, rank))

  def with_rank(self, rank):
    # [FUNCTION DOC]
    # Returns this shape specialized to a given rank. Input: rank, an integer. Merges self with an
    # unknown shape of that rank, so an unknown shape becomes one with that many unknown dimensions.
    # Raises a value error stating the required rank if the ranks conflict.
    """Returns a shape based on `self` with the given rank.

    This method promotes a completely unknown shape to one with a
    known rank.

    Args:
      rank: An integer.

    Returns:
      A shape that is at least as specific as `self` with the given rank.

    Raises:
      ValueError: If `self` does not represent a shape with the given `rank`.
    """
    try:
      return self.merge_with(unknown_shape(rank=rank))
    except ValueError:
      raise ValueError("Shape %s must have rank %d" % (self, rank))

  def with_rank_at_least(self, rank):
    # [FUNCTION DOC]
    # Checks the rank is at least a minimum. Input: rank, an integer. Returns self unchanged when the
    # rank is unknown or large enough; raises a value error otherwise.
    """Returns a shape based on `self` with at least the given rank.

    Args:
      rank: An integer.

    Returns:
      A shape that is at least as specific as `self` with at least the given
      rank.

    Raises:
      ValueError: If `self` does not represent a shape with at least the given
        `rank`.
    """
    if self.rank is not None and self.rank < rank:
      raise ValueError("Shape %s must have rank at least %d" % (self, rank))
    else:
      return self

  def with_rank_at_most(self, rank):
    # [FUNCTION DOC]
    # Checks the rank is at most a maximum. Input: rank, an integer. Returns self unchanged when the
    # rank is unknown or small enough; raises a value error otherwise.
    """Returns a shape based on `self` with at most the given rank.

    Args:
      rank: An integer.

    Returns:
      A shape that is at least as specific as `self` with at most the given
      rank.

    Raises:
      ValueError: If `self` does not represent a shape with at most the given
        `rank`.
    """
    if self.rank is not None and self.rank > rank:
      raise ValueError("Shape %s must have rank at most %d" % (self, rank))
    else:
      return self

  def is_subtype_of(self, other: trace.TraceType) -> bool:
    # [FUNCTION DOC]
    # Trace-type subtyping test. Input: other, a trace type. Returns false if other is not a
    # TensorShape; true if other has unknown rank; false if ranks differ; otherwise true when each
    # dimension of other is unknown or equal to the matching dimension of self. If self has unknown
    # rank while other has a known rank the rank comparison fails, so the result is false.
    """Returns True iff `self` is subtype of `other`.

    Shape A is a subtype of shape B if shape B can successfully represent it:

    * A `TensorShape` of any rank is a subtype of `TensorShape(None)`.

    *  TensorShapes of equal ranks are covariant, i.e.
      `TensorShape([A1, A2, ..])` is a subtype of
      `TensorShape([B1, B2, ..])` iff An is a subtype of Bn.

      An is subtype of Bn iff An == Bn or Bn is None.

    * TensorShapes of different defined ranks have no subtyping relation.

    The subtyping relation is reflexive and transitive, but not symmetric.

    Some examples:
    * `TensorShape([32, 784])` is a subtype of `TensorShape(None)`, and
      `TensorShape([4, 4])` is also a subtype of `TensorShape(None)` but
      `TensorShape([32, 784])` and `TensorShape([4, 4])` are not subtypes of
      each other.

    * All two-dimensional shapes are subtypes of `TensorShape([None, None])`,
      such as `TensorShape([32, 784])`. There is no subtype relationship with,
      for example, `TensorShape([None])` or `TensorShape([None, None, None])`.

    * `TensorShape([32, None])` is also a subtype of `TensorShape([None, None])`
      and `TensorShape(None)`. It is not a subtype of, for example,
      `TensorShape([32])`, `TensorShape([32, None, 1])`,
      `TensorShape([64, None])` or `TensorShape([None, 32])`.

    * `TensorShape([32, 784])` is a subtype of itself, and also
      `TensorShape([32, None])`, `TensorShape([None, 784])`,
      `TensorShape([None, None])` and `TensorShape(None)`.
      It has no subtype relation with, for example, `TensorShape([32, 1, 784])`
      or `TensorShape([None])`.

    Args:
      other: Another `TensorShape`.

    Returns:
      True iff `self` is subtype of `other`.

    """
    if not isinstance(other, TensorShape):
      return False

    # All Tensors are subtypes of a Tensor with no shape.
    if other.rank is None:
      return True

    # Tensor with a defined shape can only be subtype of another with a defined
    # shape if they have the same number of dimensions.
    if self.rank != other.rank:
      return False

    # A Tensor is a subtype if each corresponding dimension is a subtype.
    return all(o is None or s == o for s, o in zip(self._dims, other._dims))  # pylint: disable=protected-access

  def most_specific_common_supertype(
      self, others: Sequence[trace.TraceType]) -> Optional["TensorShape"]:
    # [FUNCTION DOC]
    # Computes the most specific shape that is a supertype of self and all given shapes.
    # Input: others, a sequence of trace types. Returns null if any is not a TensorShape; an unknown
    # shape if self has unknown rank, any other has unknown rank or ranks differ; otherwise a shape that
    # keeps each dimension only if it is identical in all shapes and uses unknown elsewhere.
    """Returns the most specific supertype `TensorShape` of self and others.

    * `TensorShape([None, 1])` is the most specific `TensorShape` supertyping
      both `TensorShape([2, 1])` and `TensorShape([5, 1])`. Note that
      `TensorShape(None)` is also a supertype but it is not "most specific".

    * `TensorShape([1, 2, 3])` is the most specific `TensorShape` supertyping
      both `TensorShape([1, 2, 3])` and `TensorShape([1, 2, 3]`). There are
      other less specific TensorShapes that supertype above mentioned
      TensorShapes, e.g. `TensorShape([1, 2, None])`, `TensorShape(None)`.

     * `TensorShape([None, None])` is the most specific `TensorShape`
       supertyping both `TensorShape([2, None])` and `TensorShape([None, 3])`.
       As always, `TensorShape(None)` is also a supertype but not the most
       specific one.

     * `TensorShape(None`) is the only `TensorShape` supertyping both
       `TensorShape([1, 2, 3])` and `TensorShape([1, 2])`. In general, any two
       shapes that have different ranks will only have `TensorShape(None)`
       as a common supertype.

     * `TensorShape(None)` is the only `TensorShape` supertyping both
       `TensorShape([1, 2, 3])` and `TensorShape(None)`. In general, the common
       supertype of any shape with `TensorShape(None)` is `TensorShape(None)`.

    Args:
      others: Sequence of `TensorShape`.

    Returns:
      A `TensorShape` which is the most specific supertype shape of `self`
      and `others`. None if it does not exist.
    """
    if any(not isinstance(other, TensorShape) for other in others):
      return None

    # A Rankless TensorShape is already a global supertype so we return another
    # instance of it.
    if self.rank is None:
      return unknown_shape()

    # A Rankless TensorShape is the most specific supertype for shapes whose
    # ranks do not match.
    if any(other.dims is None or self.rank != other.rank for other in others):
      return unknown_shape()

    # Retain the integer dimension if it is the same across all others, else
    # use an undefined dimension.
    dims = [
        dim if all(dim == other._dims[i]
                   for other in others) else None
        for i, dim in enumerate(self._dims)
    ]
    return TensorShape(dims)

  @doc_controls.do_not_doc_inheritable
  def placeholder_value(self, placeholder_context):
    # [FUNCTION DOC]
    # Forwards to the trace-type base class implementation (marked not-for-inherited-docs).
    # Input: placeholder_context, the context object for building placeholders. Returns whatever the
    # base class returns.
    """See tf.types.experimental.TraceType base class."""
    return super().placeholder_value(placeholder_context)

  @doc_controls.do_not_doc_inheritable
  def from_tensors(self, tensors):
    # [FUNCTION DOC]
    # Forwards to the trace-type base class implementation (marked not-for-inherited-docs).
    # Input: tensors, the flat tensors to rebuild a value from. Returns the base class result.
    """See tf.types.experimental.TraceType base class."""
    return super().from_tensors(tensors)

  @doc_controls.do_not_doc_inheritable
  def to_tensors(self, value):
    # [FUNCTION DOC]
    # Forwards to the trace-type base class implementation (marked not-for-inherited-docs).
    # Input: value, the object to split into tensors. Returns the base class result.
    """See tf.types.experimental.TraceType base class."""
    return super().to_tensors(value)

  @doc_controls.do_not_doc_inheritable
  def flatten(self):
    # [FUNCTION DOC]
    # Forwards to the trace-type base class implementation (marked not-for-inherited-docs).
    # Takes no inputs and returns the base class result (a flattened list of components).
    """See tf.types.experimental.TraceType base class."""
    return super().flatten()

  @doc_controls.do_not_doc_inheritable
  def cast(self, value, cast_context):
    # [FUNCTION DOC]
    # Forwards to the trace-type base class implementation (marked not-for-inherited-docs).
    # Inputs: value, the object to cast, and cast_context, the casting context. Returns the base class
    # result.
    """See tf.types.experimental.TraceType base class."""
    return super().cast(value, cast_context)

  @classmethod
  def experimental_type_proto(cls) -> Type[tensor_shape_pb2.TensorShapeProto]:
    # [FUNCTION DOC]
    # Class method for the serializable interface. Takes no inputs and returns the proto message type
    # (TensorShapeProto) used to serialize a TensorShape.
    """Returns the type of proto associated with TensorShape serialization."""
    return tensor_shape_pb2.TensorShapeProto

  @classmethod
  def experimental_from_proto(
      cls, proto: tensor_shape_pb2.TensorShapeProto) -> "TensorShape":
    # [FUNCTION DOC]
    # Class method for the serializable interface. Input: proto, a TensorShapeProto. Returns a new
    # TensorShape built from it (unknown rank flag or per-dimension sizes, -1 meaning unknown).
    """Returns a TensorShape instance based on the serialized proto."""
    return TensorShape(proto)

  def experimental_as_proto(self) -> tensor_shape_pb2.TensorShapeProto:
    # [FUNCTION DOC]
    # Serialization hook. Takes no inputs; returns the TensorShapeProto form of this shape by
    # delegating to the proto conversion method.
    """Returns a proto representation of the TensorShape instance."""
    return self.as_proto()

  # TODO(b/216206374): Consider deprecation at TraceType release.
  def is_compatible_with(self, other):
    # [FUNCTION DOC]
    # Compatibility test. Input: other, convertible to a TensorShape (conversion errors propagate).
    # Returns true when some fully defined shape could match both: always true if either rank is
    # unknown, false if ranks differ or any pair of known dimensions differs, otherwise true.
    # Reflexive and symmetric but not transitive.
    """Returns True iff `self` is compatible with `other`.

    Two possibly-partially-defined shapes are compatible if there
    exists a fully-defined shape that both shapes can represent. Thus,
    compatibility allows the shape inference code to reason about
    partially-defined shapes. For example:

    * TensorShape(None) is compatible with all shapes.

    * TensorShape([None, None]) is compatible with all two-dimensional
      shapes, such as TensorShape([32, 784]), and also TensorShape(None). It is
      not compatible with, for example, TensorShape([None]) or
      TensorShape([None, None, None]).

    * TensorShape([32, None]) is compatible with all two-dimensional shapes
      with size 32 in the 0th dimension, and also TensorShape([None, None])
      and TensorShape(None). It is not compatible with, for example,
      TensorShape([32]), TensorShape([32, None, 1]) or TensorShape([64, None]).

    * TensorShape([32, 784]) is compatible with itself, and also
      TensorShape([32, None]), TensorShape([None, 784]), TensorShape([None,
      None]) and TensorShape(None). It is not compatible with, for example,
      TensorShape([32, 1, 784]) or TensorShape([None]).

    The compatibility relation is reflexive and symmetric, but not
    transitive. For example, TensorShape([32, 784]) is compatible with
    TensorShape(None), and TensorShape(None) is compatible with
    TensorShape([4, 4]), but TensorShape([32, 784]) is not compatible with
    TensorShape([4, 4]).

    Args:
      other: Another TensorShape.

    Returns:
      True iff `self` is compatible with `other`.

    """
    other = as_shape(other)
    if self._dims is not None and other._dims is not None:  # pylint: disable=protected-access
      if self.rank != other.rank:
        return False
      for x_dim, y_dim in zip(self._dims, other._dims):  # pylint: disable=protected-access
        # Inline TensorShape.dims logic for performance in tight loops.
        if x_dim is not None and y_dim is not None and x_dim != y_dim:
          return False
    return True

  def assert_is_compatible_with(self, other):
    # [FUNCTION DOC]
    # Assertion form of the compatibility test. Input: other, convertible to a TensorShape. Returns
    # nothing; raises a value error naming both shapes if they are not compatible.
    """Raises exception if `self` and `other` do not represent the same shape.

    This method can be used to assert that there exists a shape that both
    `self` and `other` represent.

    Args:
      other: Another TensorShape.

    Raises:
      ValueError: If `self` and `other` do not represent the same shape.
    """
    if not self.is_compatible_with(other):
      raise ValueError("Shapes %s and %s are incompatible" % (self, other))

  def most_specific_compatible_shape(self, other) -> "TensorShape":
    # [FUNCTION DOC]
    # Finds the most specific shape compatible with both. Input: other, convertible to a TensorShape.
    # Returns an unknown shape if either rank is unknown or the ranks differ; otherwise a shape that
    # keeps a dimension only when both are known and equal and uses unknown elsewhere.
    """Returns the most specific TensorShape compatible with `self` and `other`.

    * TensorShape([None, 1]) is the most specific TensorShape compatible with
      both TensorShape([2, 1]) and TensorShape([5, 1]). Note that
      TensorShape(None) is also compatible with above mentioned TensorShapes.

    * TensorShape([1, 2, 3]) is the most specific TensorShape compatible with
      both TensorShape([1, 2, 3]) and TensorShape([1, 2, 3]). There are more
      less specific TensorShapes compatible with above mentioned TensorShapes,
      e.g. TensorShape([1, 2, None]), TensorShape(None).

    Args:
      other: Another `TensorShape`.

    Returns:
      A `TensorShape` which is the most specific compatible shape of `self`
      and `other`.
    """

    other = as_shape(other)
    if self.dims is None or other.dims is None or self.rank != other.rank:
      return unknown_shape()

    dims = [
        d1 if d1 is not None and d2 is not None and d1 == d2 else None
        for d1, d2 in zip(self.dims, other.dims)
    ]
    return TensorShape(dims)

  def is_fully_defined(self):
    # [FUNCTION DOC]
    # Returns true only when the rank is known and every dimension has a known size.
    """Returns True iff `self` is fully defined in every dimension."""
    return (self._dims is not None and
            all(dim is not None for dim in self._dims))

  def assert_is_fully_defined(self):
    # [FUNCTION DOC]
    # Asserts the shape is fully defined. Returns nothing; raises a value error naming the shape when
    # the rank is unknown or any dimension is unknown.
    """Raises an exception if `self` is not fully defined in every dimension.

    Raises:
      ValueError: If `self` does not have a known value for every dimension.
    """
    if not self.is_fully_defined():
      raise ValueError("Shape %s is not fully defined" % self)

  def as_list(self):
    # [FUNCTION DOC]
    # Returns a new list of integers/null values, one per dimension. Raises a value error when the
    # rank is unknown.
    """Returns a list of integers or `None` for each dimension.

    Returns:
      A list of integers or `None` for each dimension.

    Raises:
      ValueError: If `self` is an unknown shape with an unknown rank.
    """
    if self._dims is None:
      raise ValueError("as_list() is not defined on an unknown TensorShape.")
    return list(self._dims)

  def as_proto(self):
    # [FUNCTION DOC]
    # Converts the shape to a TensorShapeProto: an unknown-rank proto for unknown rank, otherwise a
    # proto with one dimension entry per dimension, using -1 for unknown sizes.
    """Returns this shape as a `TensorShapeProto`."""
    if self._dims is None:
      return tensor_shape_pb2.TensorShapeProto(unknown_rank=True)
    else:
      return tensor_shape_pb2.TensorShapeProto(dim=[
          tensor_shape_pb2.TensorShapeProto.Dim(
              size=-1 if d is None else d) for d in self._dims
      ])

  def __eq__(self, other):
    # [FUNCTION DOC]
    # Equality operator. Input: other, convertible to a TensorShape; if conversion raises a type error
    # the not-implemented marker is returned. Returns true when the dimension tuples are identical;
    # two unknown-rank shapes are equal, and unknown dimensions compare equal to each other (unlike
    # Dimension equality).
    """Returns True if `self` is equivalent to `other`.

    It first tries to convert `other` to `TensorShape`. `TypeError` is thrown
    when the conversion fails. Otherwise, it compares each element in the
    TensorShape dimensions.

    * Two *Fully known* shapes, return True iff each element is equal.
    >>> t_a = tf.TensorShape([1,2])
    >>> a = [1, 2]
    >>> t_b = tf.TensorShape([1,2])
    >>> t_c = tf.TensorShape([1,2,3])
    >>> t_a.__eq__(a)
    True
    >>> t_a.__eq__(t_b)
    True
    >>> t_a.__eq__(t_c)
    False

    * Two *Partially-known* shapes, return True iff each element is equal.
    >>> p_a = tf.TensorShape([1,None])
    >>> p_b = tf.TensorShape([1,None])
    >>> p_c = tf.TensorShape([2,None])
    >>> p_a.__eq__(p_b)
    True
    >>> t_a.__eq__(p_a)
    False
    >>> p_a.__eq__(p_c)
    False

    * Two *Unknown shape*, return True.
    >>> unk_a = tf.TensorShape(None)
    >>> unk_b = tf.TensorShape(None)
    >>> unk_a.__eq__(unk_b)
    True
    >>> unk_a.__eq__(t_a)
    False

    Args:
      other: A `TensorShape` or type that can be converted to `TensorShape`.

    Returns:
      True if the dimensions are all equal.

    Raises:
      TypeError if `other` can not be converted to `TensorShape`.
    """

    try:
      other = as_shape(other)
    except TypeError:
      return NotImplemented

    return self._dims == other._dims

  def __hash__(self):
    # [FUNCTION DOC]
    # Hash support. Returns the hash of the stored dimension tuple (or of null for unknown rank), so
    # shapes that compare equal hash equally.
    return hash(self._dims)

  def __reduce__(self):
    # [FUNCTION DOC]
    # Pickling support. Returns the pair (class, argument tuple holding the Dimension list) so the shape
    # can be recreated by the constructor; note the list is null for unknown rank.
    return TensorShape, (self.dims,)

  def __concat__(self, other):
    # [FUNCTION DOC]
    # Concatenation protocol hook; delegates to the concatenate method. Input: other, a TensorShape or
    # convertible value. Returns the concatenated shape.
    return self.concatenate(other)

trace_type.register_serializable(TensorShape)


class _TensorShapeCodec:
  # [CLASS DOC]
  # HELPER: serialization glue for the nested-structure coder (saved model / function signatures).
  # Codec that converts TensorShape objects to and from StructuredValue protos. Stateless; one
  # instance is registered with the nested structure coder at import time.
  """Codec for `TensorShape`."""

  def can_encode(self, pyobj):
    # [FUNCTION DOC]
    # HELPER: part of the structure codec protocol.
    # Input: pyobj, any object. Returns true if it is a TensorShape, false otherwise.
    return isinstance(pyobj, TensorShape)

  def do_encode(self, tensor_shape_value, encode_fn):
    # [FUNCTION DOC]
    # HELPER: part of the structure codec protocol.
    # Inputs: tensor_shape_value, the TensorShape to encode; encode_fn, the recursive encoder callback
    # (unused, discarded). Returns a new StructuredValue proto whose tensor-shape field holds the
    # shape's proto form.
    del encode_fn
    encoded_tensor_shape = struct_pb2.StructuredValue()
    encoded_tensor_shape.tensor_shape_value.CopyFrom(
        tensor_shape_value.as_proto())
    return encoded_tensor_shape

  def can_decode(self, value):
    # [FUNCTION DOC]
    # HELPER: part of the structure codec protocol.
    # Input: value, a StructuredValue proto. Returns true if its tensor-shape field is set.
    return value.HasField("tensor_shape_value")

  def do_decode(self, value, decode_fn):
    # [FUNCTION DOC]
    # HELPER: part of the structure codec protocol.
    # Inputs: value, a StructuredValue proto with a tensor-shape field; decode_fn, the recursive decoder
    # callback (unused, discarded). Returns a TensorShape built from the stored proto.
    del decode_fn
    return TensorShape(value.tensor_shape_value)


nested_structure_coder.register_codec(_TensorShapeCodec())


def as_shape(shape) -> "TensorShape":
  # [FUNCTION DOC]
  # Converts an object to a TensorShape. A TensorShape is returned as is (same object); anything else
  # is passed to the TensorShape constructor, whose type/value errors propagate.
  """Converts the given object to a TensorShape."""
  if isinstance(shape, TensorShape):
    return shape
  else:
    return TensorShape(shape)


def unknown_shape(rank=None, **kwargs) -> "TensorShape":
  # [FUNCTION DOC]
  # Builds a fully unknown TensorShape, optionally with a known rank. Inputs: rank, optional integer
  # (when given, the shape has that many unknown dimensions); keyword arguments, where only the legacy
  # name for rank is honored (used when rank is not given). Returns the new shape. Raises a type error
  # if any other keyword argument remains. Note: if rank is given and the legacy name also appears,
  # the legacy keyword is left in place and triggers the type error.
  """Returns an unknown TensorShape, optionally with a known rank.

  Args:
    rank: (Optional) If specified, the number of dimensions in the shape.
    **kwargs: For backwards compatibility.

  Returns:
    An unknown TensorShape.

  Raises:
    TypeError: In case of invalid arguments.
  """
  if rank is None and "ndims" in kwargs:
    rank = kwargs.pop("ndims")
  if kwargs:
    raise TypeError("Unknown argument: %s" % kwargs)
  if rank is None:
    return TensorShape(None)
  else:
    return TensorShape([Dimension(None)] * rank)
