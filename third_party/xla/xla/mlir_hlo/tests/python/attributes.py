# Copyright 2021 The OpenXLA Authors.
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
"""Test for Python APIs accessing MHLO attributes."""

# pylint: disable=wildcard-import,undefined-variable,missing-function-docstring

from mlir import ir
from mlir.dialects import mhlo


def run(f):
  # [FUNCTION DOC]
  # HELPER: decorator used by the tests in this file to execute each test inside an MLIR context.
  # Takes a test function f with no arguments. Creates a fresh MLIR context, registers the mhlo
  # dialect in it, calls f while the context is active, then returns f unchanged. Because it is
  # used as a decorator, each test runs once at import time. Exceptions (assertion failures)
  # from f propagate to the caller; the context is released on exit.
  with ir.Context() as context:
    mhlo.register_mhlo_dialect(context)
    f()
  return f


@run
def test_channel_handle():
  # [FUNCTION DOC]
  # Test: builds a channel handle attribute from a handle value and a channel type and asserts it
  # is not null and that the handle and channel type accessors return the given values.
  # Takes no arguments, returns nothing; an assertion error signals a failure. Runs at import
  # time through the run decorator inside a context with the mhlo dialect registered.
  attr = mhlo.ChannelHandle.get(handle=1, type=2)
  assert attr is not None
  assert attr.handle == 1
  assert attr.channel_type == 2


@run
def test_comparison_direction_attr():
  # [FUNCTION DOC]
  # Test: builds a comparison direction attribute from the name of a direction, asserts it is not
  # null, that its text form is the mhlo comparison direction syntax and that the value accessor
  # returns the name.
  # Takes no arguments, returns nothing; an assertion error signals a failure. Runs at import
  # time through the run decorator inside a context with the mhlo dialect registered.
  attr = mhlo.ComparisonDirectionAttr.get("EQ")
  assert attr is not None
  assert str(attr) == "#mhlo<comparison_direction EQ>"
  assert attr.value == "EQ"


@run
def test_comparison_type_attr():
  # [FUNCTION DOC]
  # Test: builds a comparison type attribute from a type name, asserts it is not null, that its
  # text form uses the mhlo comparison type syntax and that the value accessor returns the name.
  # Takes no arguments, returns nothing; an assertion error signals a failure. Runs at import
  # time through the run decorator inside a context with the mhlo dialect registered.
  attr = mhlo.ComparisonTypeAttr.get("FLOAT")
  assert attr is not None
  assert str(attr) == "#mhlo<comparison_type FLOAT>"
  assert attr.value == "FLOAT"


@run
def test_conv_dimension_numbers():
  # [FUNCTION DOC]
  # Test: builds convolution dimension numbers from batch, feature and spatial dimension indices
  # for input, kernel and output; asserts it is not null, that its text form is the compact layout
  # notation, and that every accessor returns the supplied index or list.
  # Takes no arguments, returns nothing; an assertion error signals a failure. Runs at import
  # time through the run decorator inside a context with the mhlo dialect registered.
  attr = mhlo.ConvDimensionNumbers.get(
      input_batch_dimension=0,
      input_feature_dimension=1,
      input_spatial_dimensions=[2, 3, 4],
      kernel_input_feature_dimension=0,
      kernel_output_feature_dimension=1,
      kernel_spatial_dimensions=[2, 3],
      output_batch_dimension=0,
      output_feature_dimension=1,
      output_spatial_dimensions=[2, 3],
  )
  assert str(attr) == "#mhlo.conv<[b, f, 0, 1, 2]x[i, o, 0, 1]->[b, f, 0, 1]>"
  assert attr is not None
  assert attr.input_batch_dimension == 0
  assert attr.input_feature_dimension == 1
  assert attr.input_spatial_dimensions == [2, 3, 4]
  assert attr.kernel_input_feature_dimension == 0
  assert attr.kernel_output_feature_dimension == 1
  assert attr.kernel_spatial_dimensions == [2, 3]
  assert attr.output_batch_dimension == 0
  assert attr.output_feature_dimension == 1
  assert attr.output_spatial_dimensions == [2, 3]


@run
def test_dequantize_mode_attr():
  # [FUNCTION DOC]
  # Test: builds a dequantize mode attribute from a mode name, asserts it is not null, that its
  # text form uses the mhlo dequantize mode syntax and that the value accessor returns the name.
  # Takes no arguments, returns nothing; an assertion error signals a failure. Runs at import
  # time through the run decorator inside a context with the mhlo dialect registered.
  attr = mhlo.DequantizeModeAttr.get("MIN_COMBINED")
  assert attr is not None
  assert str(attr) == "#mhlo<dequantize_mode MIN_COMBINED>"
  assert attr.value == "MIN_COMBINED"


@run
def test_dot_dimension_numbers():
  # [FUNCTION DOC]
  # Test: builds dot dimension numbers from lists of batching and contracting dimensions for the
  # left and right operands; asserts it is not null, that the text form lists all four lists in
  # order, and that each accessor returns the supplied list.
  # Takes no arguments, returns nothing; an assertion error signals a failure. Runs at import
  # time through the run decorator inside a context with the mhlo dialect registered.
  attr = mhlo.DotDimensionNumbers.get(
      lhs_batching_dimensions=[0, 1],
      rhs_batching_dimensions=[2, 3],
      lhs_contracting_dimensions=[4, 5],
      rhs_contracting_dimensions=[6, 7],
  )
  assert attr is not None
  assert str(attr) == ("#mhlo.dot<lhs_batching_dimensions = [0, 1], "
                       "rhs_batching_dimensions = [2, 3], "
                       "lhs_contracting_dimensions = [4, 5], "
                       "rhs_contracting_dimensions = [6, 7]>")
  assert attr.lhs_batching_dimensions == [0, 1]
  assert attr.rhs_batching_dimensions == [2, 3]
  assert attr.lhs_contracting_dimensions == [4, 5]
  assert attr.rhs_contracting_dimensions == [6, 7]


@run
def test_fft_type_attr():
  # [FUNCTION DOC]
  # Test: builds an FFT type attribute from a type name, asserts it is not null, that its text
  # form uses the mhlo fft type syntax and that the value accessor returns the name.
  # Takes no arguments, returns nothing; an assertion error signals a failure. Runs at import
  # time through the run decorator inside a context with the mhlo dialect registered.
  attr = mhlo.FftTypeAttr.get("FFT")
  assert attr is not None
  assert str(attr) == "#mhlo<fft_type FFT>"
  assert attr.value == "FFT"


@run
def test_fusion_kind_attr():
  # [FUNCTION DOC]
  # Test: builds a fusion kind attribute from a kind name, asserts it is not null, that its text
  # form uses the mhlo fusion kind syntax and that the value accessor returns the name.
  # Takes no arguments, returns nothing; an assertion error signals a failure. Runs at import
  # time through the run decorator inside a context with the mhlo dialect registered.
  attr = mhlo.FusionKindAttr.get("kLoop")
  assert attr is not None
  assert str(attr) == "#mhlo<fusion_kind kLoop>"
  assert attr.value == "kLoop"


@run
def test_gather_dimension_numbers():
  # [FUNCTION DOC]
  # Test: builds gather dimension numbers from offset, collapsed, batching, index map and index
  # vector dimensions; asserts it is not null and that the text form lists offset dims, collapsed
  # slice dims, start index map and index vector dim. Accessors for the first, second, fifth and
  # sixth inputs are checked; the two (empty) batching lists are not checked through accessors or
  # the text form.
  # Takes no arguments, returns nothing; an assertion error signals a failure. Runs at import
  # time through the run decorator inside a context with the mhlo dialect registered.
  attr = mhlo.GatherDimensionNumbers.get(
      offset_dims=[1, 2],
      collapsed_slice_dims=[3, 4, 5],
      operand_batching_dims=[],
      start_indices_batching_dims=[],
      start_index_map=[6],
      index_vector_dim=7,
  )
  assert attr is not None
  assert (
      str(attr)
      == "#mhlo.gather<offset_dims = [1, 2], "
      "collapsed_slice_dims = [3, 4, 5], "
      "start_index_map = [6], "
      "index_vector_dim = 7>"
  )
  assert attr.offset_dims == [1, 2]
  assert attr.collapsed_slice_dims == [3, 4, 5]
  assert attr.start_index_map == [6]
  assert attr.index_vector_dim == 7


@run
def test_output_operand_alias():
  # [FUNCTION DOC]
  # Test: builds an output operand alias attribute from output tuple indices, an operand index and
  # operand tuple indices; asserts it is not null, that its text form lists the three fields and
  # that each accessor returns the supplied value.
  # Takes no arguments, returns nothing; an assertion error signals a failure. Runs at import
  # time through the run decorator inside a context with the mhlo dialect registered.
  attr = mhlo.OutputOperandAlias.get(
      output_tuple_indices=[0], operand_index=0, operand_tuple_indices=[1]
  )
  assert attr is not None
  assert str(attr) == ("#mhlo.output_operand_alias<output_tuple_indices = [0], "
                       "operand_index = 0, "
                       "operand_tuple_indices = [1]>")
  assert attr.output_tuple_indices == [0]
  assert attr.operand_index == 0
  assert attr.operand_tuple_indices == [1]


@run
def test_precision_attr():
  # [FUNCTION DOC]
  # Test: builds a precision attribute from a precision name, asserts it is not null, that its
  # text form uses the mhlo precision syntax and that the value accessor returns the name.
  # Takes no arguments, returns nothing; an assertion error signals a failure. Runs at import
  # time through the run decorator inside a context with the mhlo dialect registered.
  attr = mhlo.PrecisionAttr.get("DEFAULT")
  assert attr is not None
  assert str(attr) == "#mhlo<precision DEFAULT>"
  assert attr.value == "DEFAULT"


@run
def test_rng_algorithm_attr():
  # [FUNCTION DOC]
  # Test: builds an RNG algorithm attribute from an algorithm name, asserts it is not null, that
  # its text form uses the mhlo rng algorithm syntax and that the value accessor returns the name.
  # Takes no arguments, returns nothing; an assertion error signals a failure. Runs at import
  # time through the run decorator inside a context with the mhlo dialect registered.
  attr = mhlo.RngAlgorithmAttr.get("DEFAULT")
  assert attr is not None
  assert str(attr) == "#mhlo.rng_algorithm<DEFAULT>"
  assert attr.value == "DEFAULT"


@run
def test_rng_distribution_attr():
  # [FUNCTION DOC]
  # Test: builds an RNG distribution attribute from a distribution name, asserts it is not null,
  # that its text form uses the mhlo rng distribution syntax and that the value accessor returns
  # the name.
  # Takes no arguments, returns nothing; an assertion error signals a failure. Runs at import
  # time through the run decorator inside a context with the mhlo dialect registered.
  attr = mhlo.RngDistributionAttr.get("UNIFORM")
  assert attr is not None
  assert str(attr) == "#mhlo.rng_distribution<UNIFORM>"
  assert attr.value == "UNIFORM"


@run
def test_scatter_dimension_numbers():
  # [FUNCTION DOC]
  # Test: builds scatter dimension numbers from update window, inserted window, batching,
  # scatter-to-operand mapping and index vector dimensions; asserts it is not null, that the text
  # form lists the window dims, scatter dims to operand dims and index vector dim, and that the
  # accessors return the supplied values (the batching lists are not checked).
  # Takes no arguments, returns nothing; an assertion error signals a failure. Runs at import
  # time through the run decorator inside a context with the mhlo dialect registered.
  attr = mhlo.ScatterDimensionNumbers.get(
      update_window_dims=[1, 2, 3],
      inserted_window_dims=[4, 5],
      input_batching_dims=[],
      scatter_indices_batching_dims=[],
      scattered_dims_to_operand_dims=[6, 7],
      index_vector_dim=8,
  )
  assert attr is not None
  assert (
      str(attr)
      == "#mhlo.scatter<update_window_dims = [1, 2, 3], "
      "inserted_window_dims = [4, 5], "
      "scatter_dims_to_operand_dims = [6, 7], "
      "index_vector_dim = 8>"
  )
  assert attr.update_window_dims == [1, 2, 3]
  assert attr.inserted_window_dims == [4, 5]
  assert attr.scattered_dims_to_operand_dims == [6, 7]
  assert attr.index_vector_dim == 8


@run
def test_transpose_attr():
  # [FUNCTION DOC]
  # Test: builds a transpose attribute from a name, asserts it is not null, that its text form
  # uses the mhlo transpose syntax and that the value accessor returns the name.
  # Takes no arguments, returns nothing; an assertion error signals a failure. Runs at import
  # time through the run decorator inside a context with the mhlo dialect registered.
  attr = mhlo.TransposeAttr.get("TRANSPOSE")
  assert attr is not None
  assert str(attr) == "#mhlo<transpose TRANSPOSE>"
  assert attr.value == "TRANSPOSE"


@run
def test_type_extensions():
  # [FUNCTION DOC]
  # Test: builds type extensions with a bounds list containing the value 128 and the dynamic-size
  # marker obtained from the shaped type class, asserts it is not null and that the bounds
  # accessor returns the same list.
  # Takes no arguments, returns nothing; an assertion error signals a failure. Runs at import
  # time through the run decorator inside a context with the mhlo dialect registered.
  dyn_size = ir.ShapedType.get_dynamic_size()
  attr = mhlo.TypeExtensions.get(bounds=[128, dyn_size])
  assert attr is not None
  assert attr.bounds == [128, dyn_size]

