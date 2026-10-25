# [MODULE DOC]
# Public XLA client module for Python. It re-exports everything from the jax-provided jaxlib XLA
# client (wildcard import, plus the private native module, the primitive type enum and the shape
# class), and adds: the builder and op classes (XlaBuilder, XlaOp) and the ops and profiler
# submodules from sibling modules; a table mapping XLA primitive types to numpy dtypes and the
# reverse table keyed by dtype string, with a lookup helper; lightweight pure-Python stand-ins for
# several XLA protocol buffers (precision config, result accuracy, op metadata, padding config,
# dot / convolution / gather / scatter dimension numbers, replica group) that are plain objects
# with fixed attribute slots; and helpers that build those objects from compact specifications
# (padding configs, dot and convolution dimension numbers, replica groups, shapes from numpy
# values, source-location metadata, and padding values from a padding type). Also re-binds
# a few names (FftType, ShapeIndex, ResultAccuracyMode) from the ops module.
# Import-time side effects: imports jax, numpy and ml_dtypes, and builds the module-level tables.
# Not a helper module; it is a user-facing API surface that depends on jax's jaxlib.
# Copyright 2017 The OpenXLA Authors.
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
"""An XLA client in Python."""

# pylint: disable=unused-import

import enum
import inspect
import os
from typing import Sequence

from jax.jaxlib.xla_client import *  # pylint: disable=wildcard-import
from jax.jaxlib.xla_client import _xla
from jax.jaxlib.xla_client import PrimitiveType
from jax.jaxlib.xla_client import Shape

import ml_dtypes
import numpy as np


from . import _ops as ops
from . import _profiler as profiler

from ._xla_builder import XlaBuilder
from ._xla_builder import XlaOp


XLA_ELEMENT_TYPE_TO_DTYPE = {
    PrimitiveType.PRED: np.dtype('bool'),
    PrimitiveType.S4: np.dtype(ml_dtypes.int4),
    PrimitiveType.S8: np.dtype('int8'),
    PrimitiveType.S16: np.dtype('int16'),
    PrimitiveType.S32: np.dtype('int32'),
    PrimitiveType.S64: np.dtype('int64'),
    PrimitiveType.U4: np.dtype(ml_dtypes.uint4),
    PrimitiveType.U8: np.dtype('uint8'),
    PrimitiveType.U16: np.dtype('uint16'),
    PrimitiveType.U32: np.dtype('uint32'),
    PrimitiveType.U64: np.dtype('uint64'),
    PrimitiveType.F4E2M1FN: np.dtype(ml_dtypes.float4_e2m1fn),
    PrimitiveType.F8E3M4: np.dtype(ml_dtypes.float8_e3m4),
    PrimitiveType.F8E4M3: np.dtype(ml_dtypes.float8_e4m3),
    PrimitiveType.F8E4M3FN: np.dtype(ml_dtypes.float8_e4m3fn),
    PrimitiveType.F8E4M3B11FNUZ: np.dtype(ml_dtypes.float8_e4m3b11fnuz),
    PrimitiveType.F8E4M3FNUZ: np.dtype(ml_dtypes.float8_e4m3fnuz),
    PrimitiveType.F8E5M2: np.dtype(ml_dtypes.float8_e5m2),
    PrimitiveType.F8E5M2FNUZ: np.dtype(ml_dtypes.float8_e5m2fnuz),
    PrimitiveType.F8E8M0FNU: np.dtype(ml_dtypes.float8_e8m0fnu),
    PrimitiveType.BF16: np.dtype(ml_dtypes.bfloat16),
    PrimitiveType.F16: np.dtype('float16'),
    PrimitiveType.F32: np.dtype('float32'),
    PrimitiveType.F64: np.dtype('float64'),
    PrimitiveType.C64: np.dtype('complex64'),
    PrimitiveType.C128: np.dtype('complex128'),
    PrimitiveType.TUPLE: np.dtype(np.object_),
    PrimitiveType.TOKEN: np.dtype(np.object_),
}

# Note the conversion on the key. Numpy has a known issue wherein dtype hashing
# doesn't work as expected (https://github.com/numpy/numpy/issues/7242). Thus,
# when keying by dtype in this dict, we use the string form of dtypes.
DTYPE_TO_XLA_ELEMENT_TYPE = {
    str(dt): et for et, dt in XLA_ELEMENT_TYPE_TO_DTYPE.items()
}


def dtype_to_etype(dtype):
  # [FUNCTION DOC]
  # Looks up the XLA primitive element type for a given dtype.
  # Input: dtype, anything numpy accepts as a dtype (name, type object or dtype). It is converted
  # to a numpy dtype and then to its string form to index the module's reverse table.
  # Returns the matching primitive type enum value. Raises a key error if the dtype has no entry
  # in the table. No side effects.
  """Convenience function for reading DTYPE_TO_XLA_ELEMENT_TYPE."""
  return DTYPE_TO_XLA_ELEMENT_TYPE[str(np.dtype(dtype))]


class PrecisionConfig:
  # [CLASS DOC]
  # Plain Python stand-in for the XLA precision config protocol buffer. Uses fixed attribute slots
  # (no per-instance dictionary). Holds a single attribute: the list of per-operand precision
  # values. The nested alias Precision exposes the native precision enum for convenience.
  # Instances are created with no arguments and then filled in by callers.
  """Python representation of a xla.PrecisionConfig protobuf."""

  __slots__ = ('operand_precision',)

  Precision = ops.PrecisionConfig_Precision  # pylint: disable=invalid-name

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor with no parameters; creates an empty list of operand precisions. Returns nothing.
    self.operand_precision = []


FftType = ops.FftType
ShapeIndex = ops.ShapeIndex
ResultAccuracyMode = ops.ResultAccuracy_Mode


class ResultAccuracy:
  # [CLASS DOC]
  # Plain Python stand-in for the XLA result accuracy protocol buffer, with fixed attribute slots:
  # mode, absolute tolerance, relative tolerance and units-in-the-last-place tolerance. Created
  # with no arguments and then modified by the caller.
  """Python representation of a xla.ResultAccuracy protobuf."""

  __slots__ = ('mode', 'atol', 'rtol', 'ulps')

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor with no parameters; sets the mode to the default accuracy mode from the ops
    # module, absolute and relative tolerances to zero as floats, and ulps to integer zero.
    self.mode = ops.ResultAccuracy_Mode.DEFAULT
    self.atol = 0.0
    self.rtol = 0.0
    self.ulps = 0


class OpMetadata:
  # [CLASS DOC]
  # Plain Python stand-in for the XLA op metadata protocol buffer, with fixed attribute slots:
  # op type, op name, source file, source start and end line and source start and end column.
  # Used to attach source-location information to ops, e.g. as built by the helper that
  # inspects the call stack.
  """Python representation of a xla.OpMetadata protobuf."""

  __slots__ = ('op_type', 'op_name', 'source_file', 'source_line',
               'source_end_line', 'source_column', 'source_end_column')

  def __init__(self, op_type='', op_name='', source_file='', source_line=0,
               source_end_line=0, source_column=0, source_end_column=0):
    # [FUNCTION DOC]
    # Constructor storing the given metadata fields unchanged as attributes. All parameters are
    # optional: op type, op name and source file default to empty strings; source line, end line,
    # column and end column default to zero. No validation is performed; returns nothing.
    self.op_type = op_type
    self.op_name = op_name
    self.source_file = source_file
    self.source_line = source_line
    self.source_end_line = source_end_line
    self.source_column = source_column
    self.source_end_column = source_end_column


def current_source_info_metadata(op_type=None, op_name=None, skip_frames=1):
  # [FUNCTION DOC]
  # HELPER: builds source-location metadata for ops from the caller's position in the stack.
  # Inputs: op_type and op_name, optional values stored in the result (default to the null value,
  # not empty strings); skip_frames, how many stack levels to go up from this function's own frame
  # (default 1, meaning the immediate caller of this function, since index 0 is this function).
  # Returns an OpMetadata whose source file is the base name of that frame's file and whose
  # source line, end line, column and end column come from the frame position information when
  # the interpreter provides it; otherwise only the line number is filled in.
  # Raises an index error if the stack is shallower than skip_frames. Inspects the call stack,
  # which is comparatively slow.
  """Helper for use in source mapping that returns an OpMetadata object."""
  frame = inspect.stack()[skip_frames]
  filename = os.path.basename(frame.filename)
  if hasattr(frame, 'positions'):
    lineno, end_lineno, column, end_column = frame.positions
    return OpMetadata(op_type=op_type, op_name=op_name, source_file=filename,
                      source_line=lineno, source_end_line=end_lineno,
                      source_column=column, source_end_column=end_column)
  else:
    return OpMetadata(op_type=op_type, op_name=op_name, source_file=filename,
                      source_line=frame.lineno)


def shape_from_pyval(pyval, layout: Sequence[int] | None = None):
  # [FUNCTION DOC]
  # Computes the XLA shape that describes a value made of numpy arrays, possibly nested in tuples.
  # Inputs: pyval, a numpy array (anything with a dtype and a shape) or a tuple of such values
  # (nested tuples allowed); layout, optional sequence of ints giving the minor-to-major layout
  # applied to array shapes (default none, meaning the default layout).
  # Returns a tuple shape for a tuple value (built from the element shapes) or an array shape
  # otherwise. Raises a not-implemented error if a layout is given and a tuple is encountered.
  # No side effects.
  """Returns a Shape that describes a tuple-tree of Numpy arrays."""

  def convert(pyval):
    # [FUNCTION DOC]
    # HELPER: recursive worker of shape_from_pyval. Takes a value (array or tuple of values) and
    # returns its shape: a tuple shape from converting each element for tuples, otherwise an array
    # shape with the enclosing call's layout. Raises a not-implemented error when a layout was
    # given and the value is a tuple. Uses the layout of the enclosing function.
    if isinstance(pyval, tuple):
      if layout is not None:
        raise NotImplementedError(
            'shape_from_pyval does not support layouts for tuple shapes'
        )
      return Shape.tuple_shape(tuple(convert(elt) for elt in pyval))
    else:
      return Shape.array_shape(pyval.dtype, np.shape(pyval), layout)

  return convert(pyval)


class PaddingType(enum.Enum):
  # [CLASS DOC]
  # Enumeration of the two standard convolution/window padding modes: VALID (no padding, value 1)
  # and SAME (pad so output size is the input size divided by the stride rounded up, value 2).
  # Consumed by the padding computation helper in this module.
  VALID = 1
  SAME = 2


def window_padding_type_to_pad_values(
    padding_type, lhs_dims, rhs_dims, window_strides
):
  # [FUNCTION DOC]
  # Converts a padding mode into explicit padding amounts for a windowed operation such as a
  # convolution or pooling.
  # Inputs: padding_type, either a PaddingType value or a case-insensitive string "VALID"/"SAME";
  # lhs_dims, sequence of input (operand) dimension sizes; rhs_dims, sequence of filter (window)
  # dimension sizes; window_strides, sequence of strides. All sequences are per spatial dimension.
  # Returns a list with one (low, high) pair of ints per entry of window_strides: for VALID all
  # pairs are (0, 0); for SAME the total padding per dimension is chosen so the output size is
  # the ceiling of input size over stride (never negative), split with the smaller half first.
  # Raises a type error if padding_type is neither a string nor a PaddingType, and a value error
  # for an unknown string or an unexpected value. No side effects.
  """Maps PaddingType or string to pad values (list of pairs of ints)."""
  if not isinstance(padding_type, (str, PaddingType)):
    msg = 'padding_type must be str or PaddingType, got {}.'
    raise TypeError(msg.format(type(padding_type)))

  if isinstance(padding_type, str):
    if padding_type.upper() == 'VALID':
      padding_type = PaddingType.VALID
    elif padding_type.upper() == 'SAME':
      padding_type = PaddingType.SAME
    else:
      msg = 'Unknown padding type string: expected "VALID" or "SAME", got {}.'
      raise ValueError(msg.format(padding_type))

  if padding_type == PaddingType.VALID:
    return [(0, 0)] * len(window_strides)
  elif padding_type == PaddingType.SAME:
    out_shape = np.ceil(np.true_divide(lhs_dims, window_strides)).astype(int)
    pad_sizes = [
        max((out_size - 1) * stride + filter_size - in_size, 0)
        for out_size, stride, filter_size, in_size in zip(
            out_shape, window_strides, rhs_dims, lhs_dims
        )
    ]
    return [(pad_size // 2, pad_size - pad_size // 2) for pad_size in pad_sizes]
  else:
    msg = 'Unexpected PaddingType value: {}'
    raise ValueError(msg.format(padding_type))


class PaddingConfigDimension:
  # [CLASS DOC]
  # Plain Python stand-in for the XLA padding config dimension protocol buffer, with fixed
  # attribute slots: low edge padding, high edge padding and interior padding (all integers, with
  # class-level annotations). Created without arguments, then filled in.
  """Python representation of a xla.PaddingConfigDimension protobuf."""

  __slots__ = ('edge_padding_low', 'edge_padding_high', 'interior_padding')

  edge_padding_low: int
  edge_padding_high: int
  interior_padding: int

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor with no parameters; sets low edge, high edge and interior padding to zero.
    self.edge_padding_low = 0
    self.edge_padding_high = 0
    self.interior_padding = 0


class PaddingConfig:
  # [CLASS DOC]
  # Plain Python stand-in for the XLA padding config protocol buffer, with a single fixed
  # attribute slot: the list of per-dimension padding descriptions (PaddingConfigDimension
  # objects). Normally created by the helper that builds it from integer triples.
  """Python representation of a xla.PaddingConfig protobuf."""

  __slots__ = ('dimensions',)

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor with no parameters; creates an empty list of dimensions. Returns nothing.
    self.dimensions = []


def make_padding_config(
    padding_config: PaddingConfig | Sequence[tuple[int, int, int]],
) -> PaddingConfig:
  # [FUNCTION DOC]
  # Builds a padding config from a compact description.
  # Input: padding_config, either an existing PaddingConfig (returned as is, not copied) or a
  # sequence of triples of integers (edge padding low, edge padding high, interior padding), one
  # per dimension.
  # Returns a PaddingConfig with one dimension entry per triple (in order). Raises a value or type
  # error from unpacking if an item is not a triple of three values. The input is not modified.
  """Create PaddingConfig proto from list of triples of integers.

  Args:
    padding_config: either a PaddingConfig or a list of integer triples
      (edge_padding_low, edge_padding_high, interior_padding) representing the
      configuration of the padding operation.

  Returns:
    A `PaddingConfig` object.
  """
  if not isinstance(padding_config, PaddingConfig):
    triples = padding_config
    padding_config = PaddingConfig()
    for lo, hi, interior in triples:
      dimension = PaddingConfigDimension()
      dimension.edge_padding_low = lo
      dimension.edge_padding_high = hi
      dimension.interior_padding = interior
      padding_config.dimensions.append(dimension)
  return padding_config


class DotDimensionNumbers:
  # [CLASS DOC]
  # Plain Python stand-in for the XLA dot dimension numbers protocol buffer, with fixed attribute
  # slots holding four lists of dimension indices: left and right contracting dimensions and left
  # and right batch dimensions. Created without arguments and filled in by the caller or by the
  # helper that builds it from a nested tuple.
  """Python representation of a xla.DotDimensionNumbers protobuf."""

  __slots__ = (
      'lhs_contracting_dimensions',
      'rhs_contracting_dimensions',
      'lhs_batch_dimensions',
      'rhs_batch_dimensions',
  )

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor with no parameters; sets all four dimension lists (left/right contracting and
    # left/right batch) to empty lists. Returns nothing.
    self.lhs_contracting_dimensions = []
    self.rhs_contracting_dimensions = []
    self.lhs_batch_dimensions = []
    self.rhs_batch_dimensions = []


def make_dot_dimension_numbers(
    dimension_numbers: (
        DotDimensionNumbers
        | tuple[tuple[list[int], list[int]], tuple[list[int], list[int]]]
    ),
) -> DotDimensionNumbers:
  # [FUNCTION DOC]
  # Builds dot dimension numbers from a specification.
  # Input: dimension_numbers, either a nested tuple or list of the form ((left contracting,
  # right contracting), (left batch, right batch)) where each element is a list of ints, or an
  # already-built DotDimensionNumbers (any other kind of value is returned unchanged).
  # Returns a new DotDimensionNumbers filled from the nested pairs when given a list or tuple,
  # otherwise the input object itself. Raises a value error if the sequence cannot be unpacked
  # into that shape. No other side effects.
  """Builds a DotDimensionNumbers object from a specification.

  Args:
    dimension_numbers: either a `DotDimensionNumbers` or a nested tuple
      `((lhs_contract, rhs_contract), (lhs_batch, rhs_batch))` of lists of
      integers representing the dimensions to treat as contracting dimensions
      and batch dimensions on each input operand.

  Returns:
    A `DotDimensionNumbers` object.
  """
  if isinstance(dimension_numbers, (list, tuple)):
    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
    dot_dims_proto = DotDimensionNumbers()
    dot_dims_proto.lhs_contracting_dimensions.extend(lhs_contract)
    dot_dims_proto.rhs_contracting_dimensions.extend(rhs_contract)
    dot_dims_proto.lhs_batch_dimensions.extend(lhs_batch)
    dot_dims_proto.rhs_batch_dimensions.extend(rhs_batch)
    return dot_dims_proto
  else:
    return dimension_numbers


class ConvolutionDimensionNumbers:
  # [CLASS DOC]
  # Plain Python stand-in for the XLA convolution dimension numbers protocol buffer, with fixed
  # attribute slots: input batch, input feature and input spatial dimensions; kernel input
  # feature, kernel output feature and kernel spatial dimensions; output batch, output feature
  # and output spatial dimensions. Scalars are integers; the spatial entries are lists of ints.
  """Python representation of a xla.ConvolutionDimensionNumbers protobuf."""

  __slots__ = (
      'input_batch_dimension',
      'input_feature_dimension',
      'input_spatial_dimensions',
      'kernel_input_feature_dimension',
      'kernel_output_feature_dimension',
      'kernel_spatial_dimensions',
      'output_batch_dimension',
      'output_feature_dimension',
      'output_spatial_dimensions',
  )

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor with no parameters; sets all scalar dimension indices to zero and all three
    # spatial dimension lists to empty lists. Returns nothing.
    self.input_batch_dimension = 0
    self.input_feature_dimension = 0
    self.input_spatial_dimensions = []
    self.kernel_input_feature_dimension = 0
    self.kernel_output_feature_dimension = 0
    self.kernel_spatial_dimensions = []
    self.output_batch_dimension = 0
    self.output_feature_dimension = 0
    self.output_spatial_dimensions = []


def make_convolution_dimension_numbers(
    dimension_numbers: (
        None | ConvolutionDimensionNumbers | tuple[str, str, str]
    ),
    num_spatial_dimensions: int,
) -> ConvolutionDimensionNumbers:
  # [FUNCTION DOC]
  # Builds convolution dimension numbers from an optional specification.
  # Inputs: dimension_numbers, one of: the null value (use the default layout: batch first,
  # feature second, spatial dimensions after, kernel output feature first and kernel input
  # feature second); a tuple of three strings (input spec, kernel spec, output spec) with one
  # character per dimension where N is batch, C is feature, I and O are kernel input and output
  # feature, and any other distinct characters label spatial dimensions; or an existing
  # ConvolutionDimensionNumbers (returned as is). num_spatial_dimensions, integer count used
  # only for the default case.
  # Returns a ConvolutionDimensionNumbers. For the string form, spatial dimension indices of the
  # kernel are listed in kernel order, and the input and output spatial dimensions are sorted by
  # the position of their label in the kernel spec. Raises a value error if a required character
  # (N, C, I or O) is missing from a spec string. Strings of other tuple sizes raise a value
  # error from unpacking. No other side effects.
  """Builds a ConvolutionDimensionNumbers object from a specification.

  Args:
    dimension_numbers: optional, either a ConvolutionDimensionNumbers object or
      a tuple (lhs_spec, rhs_spec, out_spec). Each element is a string of length
      N+2 identifying by position: (1) batch dimensions in lhs, rhs, and the
      output with the character 'N', (2) feature dimensions in lhs and the
      output with the character 'C', (3) input and output feature dimensions in
      rhs with the characters 'I' and 'O' respectively, and (4) spatial
      dimension correspondences between lhs, rhs, and the output using any
      distinct characters. For example, to indicate dimension numbers consistent
      with the Conv operation with two spatial dimensions, one could use
      ('NCHW', 'OIHW', 'NCHW'). As another example, to indicate dimension
      numbers consistent with the TensorFlow Conv2D operation, one could use
      ('NHWC', 'HWIO', 'NHWC'). When using the latter form of convolution
      dimension specification, window strides are associated with spatial
      dimension character labels according to the order in which the labels
      appear in the rhs_spec string, so that window_strides[0] is matched with
      the dimension corresponding to the first character appearing in rhs_spec
      that is not 'I' or 'O'. By default, use the same dimension numbering as
      Conv and ConvWithGeneralPadding.
    num_spatial_dimensions: the number of spatial dimensions.

  Returns:
    A `ConvolutionDimensionNumbers` object.
  """
  if dimension_numbers is None:
    nd = num_spatial_dimensions
    dimension_numbers = ConvolutionDimensionNumbers()
    dimension_numbers.input_batch_dimension = 0
    dimension_numbers.input_feature_dimension = 1
    dimension_numbers.output_batch_dimension = 0
    dimension_numbers.output_feature_dimension = 1
    dimension_numbers.kernel_output_feature_dimension = 0
    dimension_numbers.kernel_input_feature_dimension = 1
    dimension_numbers.input_spatial_dimensions.extend(range(2, 2 + nd))
    dimension_numbers.kernel_spatial_dimensions.extend(range(2, 2 + nd))
    dimension_numbers.output_spatial_dimensions.extend(range(2, 2 + nd))
  elif isinstance(dimension_numbers, tuple):
    lhs_spec, rhs_spec, out_spec = dimension_numbers
    dimension_numbers = ConvolutionDimensionNumbers()

    dimension_numbers.input_batch_dimension = lhs_spec.index('N')
    dimension_numbers.input_feature_dimension = lhs_spec.index('C')
    dimension_numbers.output_batch_dimension = out_spec.index('N')
    dimension_numbers.output_feature_dimension = out_spec.index('C')
    dimension_numbers.kernel_output_feature_dimension = rhs_spec.index('O')
    dimension_numbers.kernel_input_feature_dimension = rhs_spec.index('I')

    dimension_numbers.kernel_spatial_dimensions.extend(
        i for i, c in enumerate(rhs_spec) if c not in {'I', 'O'}
    )
    dimension_numbers.input_spatial_dimensions.extend(
        sorted(
            (i for i, c in enumerate(lhs_spec) if c not in {'N', 'C'}),
            key=lambda i: rhs_spec.index(lhs_spec[i]),
        )
    )
    dimension_numbers.output_spatial_dimensions.extend(
        sorted(
            (i for i, c in enumerate(out_spec) if c not in {'N', 'C'}),
            key=lambda i: rhs_spec.index(out_spec[i]),
        )
    )
  return dimension_numbers


class GatherDimensionNumbers:
  # [CLASS DOC]
  # Plain Python stand-in for the XLA gather dimension numbers protocol buffer, with fixed
  # attribute slots: offset dims, collapsed slice dims, start index map (lists of ints) and
  # index vector dim (an int). It lacks the operand batching and start-indices batching fields
  # that newer versions of the native structure have.
  """Python representation of a xla.GatherDimensionNumbers protobuf."""

  __slots__ = (
      'offset_dims',
      'collapsed_slice_dims',
      'start_index_map',
      'index_vector_dim',
  )

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor with no parameters; sets the three lists to empty and the index vector
    # dimension to zero. Returns nothing.
    self.offset_dims = []
    self.collapsed_slice_dims = []
    self.start_index_map = []
    self.index_vector_dim = 0


class ScatterDimensionNumbers:
  # [CLASS DOC]
  # Plain Python stand-in for the XLA scatter dimension numbers protocol buffer, with fixed
  # attribute slots: update window dims, inserted window dims, scatter dims to operand dims (lists
  # of ints) and index vector dim (an int). It lacks the input batching and scatter indices
  # batching fields that newer versions of the native structure have.
  """Python representation of a xla.ScatterDimensionNumbers protobuf."""

  __slots__ = (
      'update_window_dims',
      'inserted_window_dims',
      'scatter_dims_to_operand_dims',
      'index_vector_dim',
  )

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor with no parameters; sets the three lists to empty and the index vector
    # dimension to zero. Returns nothing.
    self.update_window_dims = []
    self.inserted_window_dims = []
    self.scatter_dims_to_operand_dims = []
    self.index_vector_dim = 0


class ReplicaGroup:
  # [CLASS DOC]
  # Plain Python stand-in for the XLA replica group protocol buffer, with one fixed attribute
  # slot: the list of replica ids that form the group.
  """Python representation of a xla.ReplicaGroup protobuf."""

  __slots__ = ('replica_ids',)

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor with no parameters; creates an empty list of replica ids. Returns nothing.
    self.replica_ids = []


def _make_replica_group_proto(replica_group):
  # [FUNCTION DOC]
  # HELPER: converts one group of replica ids into a ReplicaGroup object for make_replica_groups.
  # Input: replica_group, any iterable of integer replica ids. Returns a new ReplicaGroup whose
  # ids are the items of the iterable in order. No other side effects.
  replica_group_proto = ReplicaGroup()
  replica_group_proto.replica_ids.extend(replica_group)
  return replica_group_proto


def make_replica_groups(replica_groups):
  # [FUNCTION DOC]
  # KNOWN ISSUE: has no docstring and its name does not say that the null value is mapped to an
  # empty list, which is a special value in the XLA API meaning no explicit replica groups.
  # Converts a collection of replica groups into a list of ReplicaGroup objects.
  # Input: replica_groups, either the null value or an iterable of iterables of integer replica
  # ids. Returns an empty list for the null value, otherwise a list with one ReplicaGroup per
  # inner group, in order. The outer iterable is consumed once (copied to a list). Raises a type
  # error if an element is not iterable. No other side effects.
  if replica_groups is None:
    replica_groups_protos = []  # special value for XLA API
  else:
    replica_groups = list(replica_groups)
    replica_groups_protos = [
        _make_replica_group_proto(group) for group in replica_groups
    ]
  return replica_groups_protos
