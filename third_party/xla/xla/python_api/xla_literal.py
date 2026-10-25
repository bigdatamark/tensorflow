# [MODULE DOC]
# Conversion utilities between XLA literal protocol buffers and numpy arrays (including tuples).
# Public entry points: one that converts a literal into a numpy array or nested tuple of arrays,
# and one that converts a numpy array or nested tuple of arrays into a literal. A private helper
# handles the single-array case. Relies on the type tables in the types module and the shape
# helpers in the shape module. No import-time side effects.
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
"""XLA LiteralProto utilities."""

import numpy as _np  # Avoids becoming a part of public Tensorflow API.

from xla import xla_data_pb2
from xla.python_api import types_
from xla.python_api import xla_shape


def ConvertLiteralToNumpyArray(literal):
  # [FUNCTION DOC]
  # Converts a literal protocol buffer into numpy data. Input: a literal message with a shape.
  # Output: for tuple shapes, a Python tuple built recursively from the sub-literals; for scalars
  # (no dimensions), a zero-dimensional numpy array holding the first stored element; for arrays,
  # a numpy array of the shape's dimensions and the element dtype, reshaped from the linearized
  # element field using Fortran order when the layout's minor-to-major is ascending and C order
  # when it is descending. Raises a KeyError if the element type is missing from the type table
  # and NotImplementedError for any other layout. No side effects.
  """Converts a XLA literal to a Numpy array."""
  element_type = literal.shape.element_type
  if element_type == xla_data_pb2.TUPLE:
    return tuple(
        ConvertLiteralToNumpyArray(subliteral)
        for subliteral in literal.tuple_literals)

  type_record = types_.MAP_XLA_TYPE_TO_RECORD[element_type]
  if not literal.shape.dimensions:
    return _np.array(
        getattr(literal, type_record.literal_field_name)[0],
        type_record.numpy_dtype)
  else:
    # Infer the proper Numpy order from the LiteralProto's layout. The repeated
    # field representing the array's content in the Literal is linearized.
    # Reading is done in two steps:
    #
    # 1. Read the array as 1D from the LiteralProto repeated field.
    # 2. Reshape the array to its proper shape, using the right order depending
    #    on the LiteralProto's layout.
    layout_order = literal.shape.layout.minor_to_major
    numpy_shape = tuple(literal.shape.dimensions)
    if layout_order == list(range(len(literal.shape.dimensions))):
      numpy_reshaper = lambda arr: arr.reshape(numpy_shape, order='F')
    elif layout_order == list(range(len(literal.shape.dimensions) - 1, -1, -1)):
      numpy_reshaper = lambda arr: arr.reshape(numpy_shape, order='C')
    else:
      raise NotImplementedError('Unsupported layout: {0}'.format(layout_order))
    ndarray = _np.asarray(
        getattr(literal, type_record.literal_field_name),
        dtype=type_record.numpy_dtype)
    return numpy_reshaper(ndarray)


def _ConvertNumpyArrayToLiteral(ndarray):
  # [FUNCTION DOC]
  # HELPER: Single-array worker for the public numpy-to-literal conversion function.
  # Arrays are flattened with the "keep memory order" option, so the element order written
  # matches the layout the shape helper derives (column-major inputs stay column-major).
  # Boolean arrays are instead written element by element in iteration order.
  # Input: one numpy array. Output: a new literal whose shape comes from the shape helper and whose
  # type-specific repeated field holds the elements (scalars appended as one converted item, bool
  # arrays element by element, other arrays extended with the flattened data). Raises a KeyError
  # for unsupported dtypes.
  """Converts a Numpy array to a XLA literal."""
  type_record = types_.MAP_DTYPE_TO_RECORD[str(ndarray.dtype)]
  literal = xla_data_pb2.LiteralProto()
  literal.shape.CopyFrom(xla_shape.CreateShapeFromNumpy(ndarray).message)

  if ndarray.ndim == 0:
    getattr(literal, type_record.literal_field_name).append(
        ndarray.astype(type_record.literal_field_type).item())
  else:
    # Ndarrays with boolean dtypes need special type conversion with protobufs
    if ndarray.dtype in {_np.bool_, _np.dtype('bool')}:
      for element in _np.nditer(ndarray):
        getattr(literal, type_record.literal_field_name).append(
            type_record.literal_field_type(element))
    else:
      ndarray_flat = ndarray.ravel(order='A')
      getattr(literal, type_record.literal_field_name).extend(ndarray_flat)
  return literal


def ConvertNumpyArrayToLiteral(value):
  # [FUNCTION DOC]
  # Converts a numpy array, or a (possibly nested) tuple of numpy arrays, into a literal message.
  # For a tuple, builds a tuple-shaped literal and fills its tuple elements recursively with the
  # converted components; otherwise delegates to the private single-array converter. Raises a
  # KeyError for unsupported dtypes. Returns a new literal; inputs are not modified.
  """Converts a Numpy array or a nested tuple thereof to an XLA literal."""
  if isinstance(value, tuple):
    literal = xla_data_pb2.LiteralProto()
    literal.shape.CopyFrom(xla_shape.CreateShapeFromNumpy(value).message)
    for component in value:
      component_literal = literal.tuple_literals.add()
      component_literal.CopyFrom(ConvertNumpyArrayToLiteral(component))
    return literal
  else:
    return _ConvertNumpyArrayToLiteral(value)
