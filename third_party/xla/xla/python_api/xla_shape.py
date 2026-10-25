# [MODULE DOC]
# Python wrapper around the XLA shape protocol buffer, plus factory functions that derive shapes
# from numpy arrays, nested tuples of arrays, or a dtype with a dimension sequence.
# Exposes the shape wrapper class (with element type, dimensions, tuple sub-shapes and layout
# accessors) and the creation helpers. Used by the xla_literal module. Uses the type tables in the
# types module to map numpy dtypes to XLA primitive types. No import-time side effects.
# Copyright 2018 The OpenXLA Authors.
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
"""XLA Shape utilities."""

import numpy as _np  # Avoids becoming a part of public Tensorflow API.

from xla import xla_data_pb2
from xla.python_api import types_


class Shape(object):
  # [CLASS DOC]
  # Convenience wrapper around a shape protocol buffer message. Holds the message as a public
  # attribute and, for tuple shapes, a Python tuple of the component Shape wrappers. Instances are
  # built directly with an element type, dimensions and optional layout, or through the factory
  # functions in this module. Accessor methods are plain methods, not properties.
  """Wraps a xla_data_pb2.ShapeProto message with a convenient Python type.

  Provides direct access to the underlying xla_data_pb2.ShapeProto message in
  the
  message attribute, along with accessor wrappers to the message's fields.
  Avoid direct access to .message unless interacting directly with protobuf APIs
  like CopyFrom. In other words, prefer hauling the shape around in a Shape, and
  only access .message when strictly required by the protobuf API.
  """

  def __init__(self, element_type, dimensions, layout=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: element_type (XLA primitive type code); dimensions (sequence of integer
    # sizes, or for the tuple type a sequence of Shape wrappers); layout (optional minor-to-major
    # ordering, ignored for tuples; when omitted a row-major ordering, i.e. descending dimension
    # indices, is used). Creates a fresh shape message and fills it: tuples copy each component's
    # message into the tuple list and remember the component wrappers; other types store the
    # dimensions and layout. Raises a value error if the tuple type is given anything other than
    # Shape wrappers. No external side effects.
    """Creates a new XLA Shape.

    Args:
      element_type: element type from xla_data_pb2.
      dimensions: sequence of dimensions sizes (integers), or sequence
        of Shapes in the case of a tuple, i.e. when element_type is
        TUPLE.
      layout: optional minor_to_major sequence for layout. If not given, the
        default major-to-minor layout is used.

    Raises:
      ValueError: if element_type is TUPLE but dimensions are not Shape objects.
    """
    self.message = xla_data_pb2.ShapeProto()
    self.message.element_type = element_type
    if element_type == xla_data_pb2.TUPLE:
      if not all(isinstance(subshape, Shape) for subshape in dimensions):
        raise ValueError(
            'XLA tuple requires sequence of Shape objects as dimensions')
      self._tuple_shapes = tuple(dimensions)
      for component_shape in self._tuple_shapes:
        component_message = self.message.tuple_shapes.add()
        component_message.CopyFrom(component_shape.message)
    else:
      self.message.dimensions.extend(dimensions)
      if layout is None:
        layout = list(reversed(range(len(dimensions))))
      self.message.layout.minor_to_major.extend(layout)

  def element_type(self):
    # [FUNCTION DOC]
    # Returns the XLA primitive element type code stored in the underlying message.
    return self.message.element_type

  def is_tuple(self):
    # [FUNCTION DOC]
    # Returns true when the element type is the tuple type, false otherwise.
    return self.element_type() == xla_data_pb2.TUPLE

  def dimensions(self):
    # [FUNCTION DOC]
    # Returns the repeated field of dimension sizes of a non-tuple shape. Raises a value error
    # for tuple shapes, which have no dimensions (use the tuple sub-shapes accessor instead).
    if self.is_tuple():
      raise ValueError('Tuple shape has no dimensions. Try tuple_shapes()?')
    return self.message.dimensions

  def tuple_shapes(self):
    # [FUNCTION DOC]
    # Returns the Python tuple of component Shape wrappers recorded at construction. Raises a
    # value error when called on a non-tuple shape.
    """If this is a tuple, returns its sequence of constituent Shape objects.

    Returns:
      Tuple sub-shapes.

    Raises:
      ValueError: if this is not a tuple.
    """
    if not self.is_tuple():
      raise ValueError('tuple_shapes() called on a non-tuple shape')
    return self._tuple_shapes

  def layout(self):
    # [FUNCTION DOC]
    # Returns the layout message of the underlying shape (empty for tuple shapes). The ordering
    # list inside it is the minor-to-major dimension order.
    return self.message.layout

  @staticmethod
  def from_pyval(pyval):
    # [FUNCTION DOC]
    # Static factory: builds a Shape from a numpy array or nested tuple of arrays by delegating to
    # the module-level numpy shape creator. Raises a key error for unsupported dtypes.
    return CreateShapeFromNumpy(pyval)


def _CreateShapeFromNumpy(ndarray):  # pylint: disable=invalid-name
  # [FUNCTION DOC]
  # HELPER: Single-array worker for the public numpy shape creator.
  # Maps the array's dtype to an XLA primitive type via the type table (raises a key error if the
  # dtype is unsupported), takes the array's shape as dimensions, and picks the layout from memory
  # order: ascending indices for Fortran (column-major) arrays, descending for others (row-major).
  # Returns a new Shape wrapper.
  """Create a Shape from a given Numpy array.

  Args:
    ndarray: Numpy array.

  Returns:
    A Shape object.
  """
  element_type = types_.MAP_DTYPE_TO_RECORD[str(ndarray.dtype)].primitive_type
  dimensions = ndarray.shape

  # Set the shape's layout based on the ordering of ndarray.
  # Numpy arrays come in two orders: Fortran (column-major) and C (row-major).
  if _np.isfortran(ndarray):
    # Column-major layout. This corresponds to a "dimension order is
    # minor-to-major" layout in XLA.
    layout = range(ndarray.ndim)
  else:
    # Row-major layout. This corresponds to a "dimension order is
    # major-to-minor" layout int XLA.
    layout = list(reversed(range(ndarray.ndim)))

  return Shape(element_type, dimensions, layout)


def CreateShapeFromNumpy(value):  # pylint: disable=invalid-name
  # [FUNCTION DOC]
  # Creates a Shape from a numpy array or a possibly nested tuple of arrays. A tuple becomes a
  # tuple Shape whose components are created recursively; anything else goes to the private
  # single-array helper. Returns the new Shape; raises a key error for unsupported dtypes.
  """Create a Shape from a Numpy array or a nested tuple structure thereof.

  Args:
    value: Numpy array or (possibly nested) tuple structure that bottoms out in
      Numpy arrays.

  Returns:
    A Shape object.
  """
  if isinstance(value, tuple):
    return Shape(
        xla_data_pb2.TUPLE,
        [CreateShapeFromNumpy(component) for component in value])
  else:
    return _CreateShapeFromNumpy(value)


def CreateShapeFromDtypeAndTuple(dtype, shape_tuple):  # pylint: disable=invalid-name
  # [FUNCTION DOC]
  # Creates a Shape from a numpy dtype and a sequence of non-negative dimension sizes, using the
  # default row-major layout. The dtype is looked up by its string form in the type table; raises
  # a key error if unsupported. Returns the new Shape wrapper. The sizes are not validated.
  """Create a shape from a Numpy dtype and a sequence of nonnegative integers.

  Args:
    dtype: a numpy dtype, e.g. np.dtype('int32').
    shape_tuple: a sequence of nonnegative integers.

  Returns:
    A Shape object.
  """
  element_type = types_.MAP_DTYPE_TO_RECORD[str(dtype)].primitive_type
  return Shape(element_type, shape_tuple)
