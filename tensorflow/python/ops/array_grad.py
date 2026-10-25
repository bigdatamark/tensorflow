# [MODULE DOC]
# Gradient functions for the array manipulation operations defined in array_ops.py (and the
# generated array/resource ops): packing, concatenation, slicing, splitting, diagonal ops,
# gather/scatter, reshape/transpose, padding, space/depth/batch rearrangements, patch extraction,
# quantize-and-dequantize and broadcast. Each gradient takes the forward operation plus the
# incoming gradient(s) and returns one gradient (or nothing/null value) per op input.
# Importing the module has the side effect of registering these functions in the global gradient
# registry (by op type name) and marking several ops (shape queries, constants, one-hot, etc.) as
# not differentiable. It exposes no public API of its own; it is used by the autodiff machinery.
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
"""Gradients for operators defined in array_ops.py."""

from tensorflow.compiler.tf2xla.ops import gen_xla_ops
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices as indexed_slices_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops


@ops.RegisterGradient("Pack")
def _PackGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of the Pack (stack) op. Takes the forward op (its attributes give the number of
  # packed inputs and the stacking axis) and the gradient of the packed output. Returns the
  # gradient unstacked along that axis, one tensor per original input.
  """Gradient for pack op."""
  return array_ops_stack.unstack(
      grad, num=op.get_attr("N"), axis=op.get_attr("axis")
  )


@ops.RegisterGradient("Unpack")
def _UnpackGrad(op: ops.Operation, *grads):
  # [FUNCTION DOC]
  # Gradient of the Unpack (unstack) op. Takes the forward op and one gradient per output, and
  # returns a single tensor made by stacking those gradients along the op's axis attribute.
  """Gradient for unpack op."""
  return array_ops_stack.stack(grads, axis=op.get_attr("axis"))


def _ConcatGradHelper(
    op: ops.Operation, grad, start_value_index, end_value_index, dim_index
):
  # [FUNCTION DOC]
  # HELPER: shared implementation behind the Concat and ConcatV2 gradient functions.
  # KNOWN ISSUE: in the degenerate case of exactly two op inputs (one value plus the axis) the
  # code does list concatenation with the gradient, which only works if grad is already a list;
  # a plain tensor or sparse-slices gradient would fail or misbehave there, despite the docstring
  # saying it accepts those types. Also the docstring's description of start/end indices is
  # loose: the end index may be -1 (meaning exclude the last input) for ConcatV2.
  # Splits the gradient of a concatenation back into per-input gradients. Inputs: the forward
  # op; grad, a dense tensor or sparse indexed-slices gradient of the output; the slice of op
  # inputs holding the concatenated values (start and end index) and the position of the axis
  # input (dim_index). Returns a list with one gradient per value input plus a null entry for the
  # axis input (placed first or last depending on dim_index). For dense gradients in eager mode
  # it uses a native shape-slice helper and splits; in graph mode it uses either a split by sizes
  # (when more than 16 inputs) or concat-offset plus slices. For sparse gradients it requires a
  # statically known axis (and static rank if the axis is negative) and builds sparse slices per
  # input, slicing values for axis > 0 or selecting index ranges for axis 0. Raises a value error
  # when the axis is not static for sparse gradients, and a type error for other gradient types.
  # Side effects: may duplicate a constant axis into the current control-flow context for XLA.
  """Gradient for concat op.

  Args:
    op: An operation.
    grad: `Tensor` or `IndexedSlices` representing the gradients with respect to
      each output of the op.
    start_value_index: An integer index of the first value in the op.inputs.
    end_value_index: An integer index of the last value in the op.inputs.
    dim_index: An integer index of concat_dim or axis parameter in op.inputs.

  Returns:
    Tensors representing the partial gradients with respect to each input
    of the op.

  Raises:
    ValueError: if concat_dim/axis is not statically known.
  """

  def _CreateDenseMaskAndBegin(sizes, concat_dim):
    # [FUNCTION DOC]
    # HELPER: used by the sparse-gradient branch of the concat gradient helper.
    # Builds two integer vectors whose length is the rank of the first input: a mask that is one
    # at the concat dimension and zero elsewhere, and an all-zero begin vector. Inputs: list of
    # shape tensors and the non-negative concat dimension. Returns the (mask, begin) pair.
    """Create variables for iteratively slicing a dense gradients tensor."""
    # Since shape is 1-D, shape_of_shape = [rank-of-inputs]
    shape_of_shape = array_ops.shape(sizes[0])
    # Make a vector of length equal to the input's dimensions,
    # with 0's everywhere and 1 in the concat dim position.
    # Note: Can't use sparse_to_dense since it isn't GPU-capable (for now)
    mask = array_ops.concat(
        [
            array_ops.zeros(
                array_ops.expand_dims(concat_dim, 0), dtype=dtypes.int32
            ),
            [1],
            array_ops.zeros(
                shape_of_shape - concat_dim - 1, dtype=dtypes.int32
            ),
        ],
        0,
    )
    begin = array_ops.zeros(shape_of_shape, dtype=dtypes.int32)
    return mask, begin

  def _ExtractInputShapes(inputs):
    # [FUNCTION DOC]
    # HELPER: used by the dense-gradient branch of the concat gradient helper.
    # Returns the shapes of the given input tensors. In eager mode, or when any shape is not a
    # compile-time constant, it returns the result of a single multi-input shape op; otherwise it
    # returns the list of constant per-input shape tensors. No side effects beyond creating ops.
    """Extract the shapes of a set of input tensors."""
    if context.executing_eagerly():
      return array_ops.shape_n(inputs)
    sizes = []
    fully_known = True
    for x in inputs:
      input_shape = array_ops.shape(x)
      if (
          not isinstance(input_shape, tensor.Tensor)
          or input_shape.op.type != "Const"
      ):
        fully_known = False
        break
      sizes.append(input_shape)

    if fully_known:
      return sizes
    else:
      return array_ops.shape_n(inputs)

  # Degenerate concatenation, just return grad.
  if len(op.inputs) == 2:
    return grad + [None] if end_value_index <= dim_index else [None] + grad

  concat_dim = op.inputs[dim_index]
  input_values = op.inputs[start_value_index:end_value_index]

  out_grads = []
  if isinstance(grad, tensor.Tensor):
    if context.executing_eagerly() or isinstance(concat_dim, ops.EagerTensor):
      # Using mod here for convenience since concat_dim is already verified
      # in concat implementation to be within the allowed [-rank, rank) range.
      non_neg_concat_dim = (
          concat_dim._numpy().item(0) % input_values[0]._rank()  # pylint: disable=protected-access
      )  # pylint: disable=protected-access
      # All inputs are guaranteed to be EagerTensors in eager mode
      sizes = pywrap_tfe.TFE_Py_TensorShapeSlice(
          input_values, non_neg_concat_dim
      )
      out_grads = array_ops.split(grad, sizes, non_neg_concat_dim)
    else:
      if constant_op.is_constant(concat_dim):
        # If concat_dim is a constant defined in a different context,
        # then we duplicate it in the current context to avoid passing it
        # through an Enter node.
        # This is a small optimization in general, but it is required when
        # compiling with XLA, as XLA needs the concat input to be folded into a
        # constant.
        grad_context = control_flow_util.GetOutputContext(grad.op)
        dim_context = control_flow_util.GetOutputContext(concat_dim.op)
        if dim_context != grad_context:
          value = tensor_util.constant_value(concat_dim)
          concat_dim = constant_op.constant(value=value, dtype=concat_dim.dtype)

      # Using mod here for convenience since concat_dim is already verified
      # in concat implementation to be within the allowed [-rank, rank) range.
      non_neg_concat_dim = concat_dim % array_ops.rank(input_values[0])

      # Get the inputs' tensor shapes
      sizes = _ExtractInputShapes(input_values)
      # The magic number of 16 was found through benchmarking a range of sizes
      # on CPUs and a Maxwell TitanX.  A speedup was seen in a large majority of
      # cases when switching implementations at N=16, but it is possible that
      # there will be a small number of performance regressions.
      if len(sizes) > 16:
        # extract the size of each input along the concat dimension
        sizes = array_ops.squeeze(
            array_ops.slice(
                array_ops_stack.stack(sizes, axis=1),
                [non_neg_concat_dim, 0],
                [1, -1],
            )
        )
        out_grads = array_ops.split(grad, sizes, non_neg_concat_dim)
      else:
        offset = gen_array_ops.concat_offset(non_neg_concat_dim, sizes)
        for begin, size in zip(offset, sizes):
          out_grads.append(array_ops.slice(grad, begin, size))
  elif isinstance(grad, indexed_slices_lib.IndexedSlices):
    # Using mod here for convenience since concat_dim is already verified
    # in concat implementation to be within the allowed [-rank, rank) range.
    non_neg_concat_dim = concat_dim % array_ops.rank(input_values[0])
    concat_dim_static = tensor_util.constant_value(concat_dim)
    if concat_dim_static is None:
      raise ValueError(
          "Can only compute IndexedSlices gradient with "
          "statically-known concat_dim"
      )
    if concat_dim_static < 0:
      rank = tensor_util.constant_value(array_ops.rank(input_values[0]))
      if rank is None:
        raise ValueError(
            "Can only compute IndexedSlices gradient with "
            "negative concat_dim when first value rank is "
            "statically-known."
        )
      concat_dim_static %= rank
    # Get the inputs' tensor shapes
    sizes = [array_ops.shape(x) for x in input_values]
    if concat_dim_static > 0:
      # IndexedSlices, non_neg_concat_dim > 0. Each input gets IndexedSlices
      # gradients with all the indices, but with grad.values sliced accordingly.
      # This is like the Tensor case, except shape(grad.values)[0] is not equal
      # to shape(sizes[i])[0], since only a subset of the dim-0 values are
      # stored.
      mask, begin = _CreateDenseMaskAndBegin(sizes, non_neg_concat_dim)
      for size in sizes:
        new_values = array_ops.slice(
            grad.values,
            begin,
            array_ops.concat([[-1], array_ops.slice(size, [1], [-1])], 0),
        )
        out_grads.append(
            indexed_slices_lib.IndexedSlices(new_values, grad.indices, size)
        )
        # Lint complains begin = begin + ...
        begin = math_ops.add(begin, size * mask)
    else:
      # IndexedSlices, concat_dim == 0. Each input gets IndexedSlices gradients
      # only for the relevant indices.
      start = constant_op.constant(0, dtype=grad.indices.dtype)
      for size in sizes:
        size_concat_dim = array_ops.gather(size, non_neg_concat_dim)
        if size_concat_dim.dtype != grad.indices.dtype:
          size_concat_dim = math_ops.cast(
              size_concat_dim, dtype=grad.indices.dtype
          )
        end = start + size_concat_dim
        # Compute the 1-D Tensor of indices relevant for this input.
        indices_to_select = array_ops.squeeze(
            array_ops.where(
                math_ops.logical_and(grad.indices >= start, grad.indices < end)
            ),
            axis=[1],
        )
        new_indices = array_ops.gather(grad.indices, indices_to_select) - start
        new_values = array_ops.gather(grad.values, indices_to_select)
        out_grads.append(
            indexed_slices_lib.IndexedSlices(new_values, new_indices, size)
        )
        start = end
  else:
    raise TypeError("Expected Tensor or IndexedSlices, got %s" % type(grad))

  return (
      out_grads + [None] if end_value_index <= dim_index else [None] + out_grads
  )


@ops.RegisterGradient("Concat")
def _ConcatGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of the legacy Concat op, where the axis is the first input and the values follow.
  # Delegates to the shared concat gradient helper with values starting at input 1 and the axis
  # at input 0. Returns the per-input gradients (null for the axis).
  return _ConcatGradHelper(
      op, grad, start_value_index=1, end_value_index=len(op.inputs), dim_index=0
  )


@ops.RegisterGradient("ConcatV2")
def _ConcatGradV2(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of the ConcatV2 op, where the axis is the last input. Delegates to the shared concat
  # gradient helper with values spanning all inputs except the last and the axis at the last
  # position. Returns the per-input gradients (null for the axis).
  return _ConcatGradHelper(
      op, grad, start_value_index=0, end_value_index=-1, dim_index=-1
  )


ops.NotDifferentiable("ConcatOffset")


@ops.RegisterGradient("Slice")
def _SliceGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of the Slice op. Pads the incoming gradient with zeros before (by the begin vector)
  # and after (input shape minus slice size minus begin) each dimension to recover the input
  # shape. When the default graph is in an XLA context it instead writes the gradient into a
  # zeros tensor with a dynamic update slice. Returns the input gradient plus null for begin and
  # size inputs.
  """Gradient for Slice op."""
  # Create an Nx2 padding where the first column represents how many
  # zeros are to be prepended for each dimension, and the second
  # column indicates how many zeros are appended.
  #
  # The number of zeros to append is the shape of the input
  # elementwise-subtracted by both the begin vector and sizes vector.
  #
  # Some more reshaping is needed to assemble this tensor with the
  # right dimensions.
  input_vec = op.inputs[0]
  begin_vec = op.inputs[1]
  input_rank = array_ops.rank(input_vec)
  index_dtype = begin_vec.dtype
  slice_size = array_ops.shape(op.outputs[0], out_type=index_dtype)
  if control_flow_util.GraphOrParentsInXlaContext(ops.get_default_graph()):
    return (
        gen_xla_ops.xla_dynamic_update_slice(
            array_ops.zeros_like(input_vec), grad, begin_vec
        ),
        None,
        None,
    )

  shape = array_ops_stack.stack([input_rank, 1])
  before_pad = array_ops.reshape(begin_vec, shape)
  after_pad = array_ops.reshape(
      array_ops.shape(input_vec, out_type=index_dtype) - slice_size - begin_vec,
      shape,
  )
  paddings = array_ops.concat([before_pad, after_pad], 1)
  return array_ops.pad(grad, paddings), None, None


@ops.RegisterGradient("StridedSlice")
def _StridedSliceGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of the StridedSlice op. Uses the strided-slice-grad op with the input shape, begin,
  # end and strides (replaced by static constant values when they can be determined, all in the
  # begin tensor's dtype) and the five mask attributes copied from the forward op. Returns the
  # input gradient followed by null for begin, end and strides.
  """Gradient for StridedSlice op."""
  begin = op.inputs[1]
  end = op.inputs[2]
  strides = op.inputs[3]
  # StridedSliceGrad requires `x`, `begin`, `end` and `strides` to be of the
  # same dtype so we build a shape of the same type as other args.
  # Note that the choice of `begin` for specifying `out_type` is arbitrary.
  # We could choose any of {begin|end|strides}.dtype since they are required to
  # be the same.
  x = array_ops.shape(op.inputs[0], out_type=begin.dtype)

  x_static = tensor_util.constant_value(x)
  x = x_static if x_static is not None else x
  begin_static = tensor_util.constant_value(begin)
  begin = begin_static if begin_static is not None else begin
  end_static = tensor_util.constant_value(end)
  end = end_static if end_static is not None else end
  strides_static = tensor_util.constant_value(strides)
  strides = strides_static if strides_static is not None else strides

  return (
      array_ops.strided_slice_grad(
          x,
          begin,
          end,
          strides,
          grad,
          begin_mask=op.get_attr("begin_mask"),
          end_mask=op.get_attr("end_mask"),
          ellipsis_mask=op.get_attr("ellipsis_mask"),
          new_axis_mask=op.get_attr("new_axis_mask"),
          shrink_axis_mask=op.get_attr("shrink_axis_mask"),
      ),
      None,
      None,
      None,
  )


@ops.RegisterGradient("StridedSliceGrad")
def _StridedSliceGradGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of the StridedSliceGrad op (second-order gradient). Applies a strided slice of the
  # incoming gradient using the forward op's begin, end, strides and mask attributes. Returns
  # null for the first four inputs (shape, begin, end, strides) and the sliced tensor for the
  # last input.
  """Gradient for StridedSliceGrad op."""
  begin = op.inputs[1]
  end = op.inputs[2]
  strides = op.inputs[3]

  return (
      None,
      None,
      None,
      None,
      array_ops.strided_slice(
          grad,
          begin,
          end,
          strides,
          begin_mask=op.get_attr("begin_mask"),
          end_mask=op.get_attr("end_mask"),
          ellipsis_mask=op.get_attr("ellipsis_mask"),
          new_axis_mask=op.get_attr("new_axis_mask"),
          shrink_axis_mask=op.get_attr("shrink_axis_mask"),
      ),
  )


@ops.RegisterGradient("TensorStridedSliceUpdate")
def _TensorStridedSliceUpdateGrad(op: ops.Operation, grad):  # pylint:disable=missing-function-docstring
  # [FUNCTION DOC]
  # Gradient of the TensorStridedSliceUpdate op. The gradient for the updated value is the
  # strided slice of the incoming gradient, sum-reduced over broadcast axes and reshaped to the
  # value's shape (since the value may have been broadcast). The gradient for the base tensor is
  # the incoming gradient with the sliced region overwritten by zeros. Returns the base gradient,
  # null for begin, end, strides, and the value gradient.
  begin = op.inputs[1]
  end = op.inputs[2]
  strides = op.inputs[3]
  begin_mask = op.get_attr("begin_mask")
  end_mask = op.get_attr("end_mask")
  ellipsis_mask = op.get_attr("ellipsis_mask")
  new_axis_mask = op.get_attr("new_axis_mask")
  shrink_axis_mask = op.get_attr("shrink_axis_mask")

  def Apply(f, *args):
    # [FUNCTION DOC]
    # HELPER: calls a strided-slice style function with the forward op's five mask attributes.
    # Inputs: the function to call and its positional arguments. Returns the function's result.
    return f(
        *args,
        begin_mask=begin_mask,
        end_mask=end_mask,
        shrink_axis_mask=shrink_axis_mask,
        new_axis_mask=new_axis_mask,
        ellipsis_mask=ellipsis_mask,
    )

  dy = Apply(array_ops.strided_slice, grad, begin, end, strides)
  dx = Apply(
      array_ops.tensor_strided_slice_update,
      grad,
      begin,
      end,
      strides,
      array_ops.zeros_like(dy),
  )

  # The value is potentially broadcast to the shape of the strided slice, so we
  # may need to adjust dy.
  slice_shape = array_ops.shape(dy, out_type=begin.dtype)
  value_shape = array_ops.shape(op.inputs[4], out_type=slice_shape.dtype)

  _, reduction_axes = gen_array_ops.broadcast_gradient_args(
      slice_shape, value_shape
  )
  dy_reshaped = math_ops.reduce_sum(dy, axis=reduction_axes, keepdims=True)
  dy = array_ops.reshape(dy_reshaped, value_shape)

  return dx, None, None, None, dy


@ops.RegisterGradient("Split")
def _SplitGrad(op: ops.Operation, *grads):
  # [FUNCTION DOC]
  # Gradient of the Split op. Concatenates the per-output gradients along the split dimension
  # (the op's first input). Returns null for the dimension input and the concatenated gradient
  # for the value input.
  return None, array_ops.concat(list(grads), op.inputs[0])


@ops.RegisterGradient("SplitV")
def _SplitVGrad(op: ops.Operation, *grads):
  # [FUNCTION DOC]
  # Gradient of the SplitV op. Concatenates the per-output gradients along the axis given by the
  # op's third input. Returns a list whose first entry is the concatenated gradient and whose
  # remaining entries (size splits and axis) are null.
  returnval = array_ops.concat(list(grads), op.inputs[2])
  returnval = [returnval] + [
      None,
  ] * (len(op.inputs) - 1)
  return returnval


ops.NotDifferentiable("Const")


@ops.RegisterGradient("Diag")
def _DiagGrad(_, grad):
  # [FUNCTION DOC]
  # Gradient of the Diag op: the diagonal part of the incoming gradient. The forward op is
  # unused.
  return array_ops.diag_part(grad)


@ops.RegisterGradient("DiagPart")
def _DiagPartGrad(_, grad):
  # [FUNCTION DOC]
  # Gradient of the DiagPart op: builds a diagonal tensor from the incoming gradient. The forward
  # op is unused.
  return array_ops.diag(grad)


@ops.RegisterGradient("MatrixDiag")
def _MatrixDiagGrad(_, grad):
  # [FUNCTION DOC]
  # Gradient of the MatrixDiag op: the batched matrix-diagonal part of the incoming gradient.
  # The forward op is unused.
  return array_ops.matrix_diag_part(grad)


@ops.RegisterGradient("MatrixDiagV2")
def _MatrixDiagV2Grad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of MatrixDiagV2: extracts the diagonal band given by the diagonal-index input from
  # the incoming gradient. Returns that tensor followed by four null entries for the remaining
  # inputs (k, num_rows, num_cols, padding value).
  return (
      array_ops.matrix_diag_part(grad, k=op.inputs[1]),
      None,
      None,
      None,
      None,
  )


@ops.RegisterGradient("MatrixDiagV3")
def _MatrixDiagV3Grad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of MatrixDiagV3: like the V2 version but also passes the op's alignment attribute
  # when extracting the diagonal band from the incoming gradient. Returns that tensor followed by
  # four null entries for the remaining inputs.
  return (
      array_ops.matrix_diag_part(
          grad, k=op.inputs[1], align=op.get_attr("align")
      ),
      None,
      None,
      None,
      None,
  )


@ops.RegisterGradient("MatrixDiagPart")
def _MatrixDiagPartGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of MatrixDiagPart. If the last two input dimensions are statically known and equal
  # (square matrices) it builds a matrix from the gradient's diagonal; otherwise it sets the
  # diagonal of a zeros tensor shaped like the input to the gradient. Returns that single tensor.
  matrix_shape = op.inputs[0].get_shape()[-2:]
  if matrix_shape.is_fully_defined() and matrix_shape[0] == matrix_shape[1]:
    return array_ops.matrix_diag(grad)
  else:
    return array_ops.matrix_set_diag(array_ops.zeros_like(op.inputs[0]), grad)


@ops.RegisterGradient("MatrixDiagPartV2")
def _MatrixDiagPartV2Grad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of MatrixDiagPartV2. With a fully defined matrix shape it rebuilds a matrix from the
  # gradient using the diagonal index and the static row and column counts; otherwise it sets the
  # diagonals of a zeros tensor shaped like the input. Returns the gradient plus two null entries
  # (diagonal index and padding value).
  """Gradient for MatrixDiagPartV2."""
  matrix_shape = op.inputs[0].get_shape()[-2:]
  if matrix_shape.is_fully_defined():
    return (
        array_ops.matrix_diag(
            grad,
            k=op.inputs[1],
            num_rows=matrix_shape[0],
            num_cols=matrix_shape[1],
        ),
        None,
        None,
    )
  else:
    return (
        array_ops.matrix_set_diag(
            array_ops.zeros_like(op.inputs[0]), grad, k=op.inputs[1]
        ),
        None,
        None,
    )


@ops.RegisterGradient("MatrixDiagPartV3")
def _MatrixDiagPartV3Grad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of MatrixDiagPartV3: same strategy as the V2 gradient but also forwards the op's
  # alignment attribute. Returns the gradient plus two null entries.
  """Gradient for MatrixDiagPartV3."""
  matrix_shape = op.inputs[0].get_shape()[-2:]
  align = op.get_attr("align")
  if matrix_shape.is_fully_defined():
    return (
        array_ops.matrix_diag(
            grad,
            k=op.inputs[1],
            num_rows=matrix_shape[0],
            num_cols=matrix_shape[1],
            align=align,
        ),
        None,
        None,
    )
  else:
    return (
        array_ops.matrix_set_diag(
            array_ops.zeros_like(op.inputs[0]),
            grad,
            k=op.inputs[1],
            align=align,
        ),
        None,
        None,
    )


@ops.RegisterGradient("MatrixSetDiag")
def _MatrixSetDiagGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of MatrixSetDiag. Merges the static shapes of the input and gradient to find the
  # diagonal's shape (computed dynamically, colocated with the gradient, if not fully known).
  # Returns a pair: the gradient with its diagonal zeroed (for the matrix input) and the
  # gradient's diagonal (for the diagonal input).
  """Gradient for MatrixSetDiag."""
  input_shape = op.inputs[0].get_shape().merge_with(grad.get_shape())
  diag_shape = op.inputs[1].get_shape()
  batch_shape = input_shape[:-2].merge_with(diag_shape[:-1])
  matrix_shape = input_shape[-2:]
  if batch_shape.is_fully_defined() and matrix_shape.is_fully_defined():
    diag_shape = batch_shape.as_list() + [min(matrix_shape.as_list())]
  else:
    with ops.colocate_with(grad):
      grad_shape = array_ops.shape(grad)
      grad_rank = array_ops.rank(grad)
      batch_shape = array_ops.slice(grad_shape, [0], [grad_rank - 2])
      matrix_shape = array_ops.slice(grad_shape, [grad_rank - 2], [2])
      min_dim = math_ops.reduce_min(matrix_shape)
      diag_shape = array_ops.concat([batch_shape, [min_dim]], 0)
  grad_input = array_ops.matrix_set_diag(
      grad, array_ops.zeros(diag_shape, dtype=grad.dtype)
  )
  grad_diag = array_ops.matrix_diag_part(grad)
  return (grad_input, grad_diag)


@ops.RegisterGradient("MatrixSetDiagV2")
def _MatrixSetDiagGradV2(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of MatrixSetDiagV2. When the diagonal input's shape is not fully defined, infers it
  # from the gradient's shape and the lower/upper diagonal indices using conditional ops. Returns
  # the gradient with the specified diagonals zeroed, the extracted diagonals of the gradient,
  # and null for the diagonal-index input.
  """Gradient for MatrixSetDiagV2."""
  diag_shape = op.inputs[1].get_shape()
  if not diag_shape.is_fully_defined():
    # Need to know the values of `d_lower` and `d_upper` to infer diag_shape.
    grad_shape = array_ops.shape(grad)
    batch_shape = grad_shape[:-2]
    matrix_shape = grad_shape[-2:]
    diag_index = array_ops.reshape(op.inputs[2], [-1])  # Converts to vector.
    d_lower = diag_index[0]
    d_upper = diag_index[-1]  # Works both when len(diag_index) is 1 and 2.
    y_offset = cond.cond(math_ops.less(d_upper, 0), lambda: d_upper, lambda: 0)
    x_offset = cond.cond(
        math_ops.greater(d_lower, 0), lambda: -d_lower, lambda: 0
    )

    max_diag_len = math_ops.minimum(
        matrix_shape[0] + y_offset, matrix_shape[1] + x_offset
    )
    # pylint: disable=g-long-lambda
    # pyformat: disable
    postfix = cond.cond(
        math_ops.equal(d_lower, d_upper),
        lambda: ops.convert_to_tensor([max_diag_len]),
        lambda: ops.convert_to_tensor([d_upper - d_lower + 1,
                                       max_diag_len]))
    # pyformat: enable
    # pylint: enable=g-long-lambda
    diag_shape = array_ops.concat([batch_shape, postfix], 0)

  grad_input = array_ops.matrix_set_diag(
      grad, array_ops.zeros(diag_shape, dtype=grad.dtype), k=op.inputs[2]
  )
  grad_diag = array_ops.matrix_diag_part(grad, k=op.inputs[2])
  return (grad_input, grad_diag, None)


@ops.RegisterGradient("MatrixSetDiagV3")
def _MatrixSetDiagGradV3(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the diagonal-shape inference block is a verbatim duplicate of the one in the V2
  # gradient function; only the alignment attribute handling differs.
  # Gradient of MatrixSetDiagV3. Same as the V2 gradient but forwards the op's alignment
  # attribute to both the set-diagonal and diagonal-part calls. Returns the zeroed-diagonal
  # gradient, the extracted diagonals, and null for the diagonal-index input.
  """Gradient for MatrixSetDiagV3."""
  diag_shape = op.inputs[1].get_shape()
  align = op.get_attr("align")
  if not diag_shape.is_fully_defined():
    # Need to know the values of `d_lower` and `d_upper` to infer diag_shape.
    grad_shape = array_ops.shape(grad)
    batch_shape = grad_shape[:-2]
    matrix_shape = grad_shape[-2:]
    diag_index = array_ops.reshape(op.inputs[2], [-1])  # Converts to vector.
    d_lower = diag_index[0]
    d_upper = diag_index[-1]  # Works both when len(diag_index) is 1 and 2.
    y_offset = cond.cond(math_ops.less(d_upper, 0), lambda: d_upper, lambda: 0)
    x_offset = cond.cond(
        math_ops.greater(d_lower, 0), lambda: -d_lower, lambda: 0
    )

    max_diag_len = math_ops.minimum(
        matrix_shape[0] + y_offset, matrix_shape[1] + x_offset
    )
    # pylint: disable=g-long-lambda
    # pyformat: disable
    postfix = cond.cond(
        math_ops.equal(d_lower, d_upper),
        lambda: ops.convert_to_tensor([max_diag_len]),
        lambda: ops.convert_to_tensor([d_upper - d_lower + 1,
                                       max_diag_len]))
    # pyformat: enable
    # pylint: enable=g-long-lambda
    diag_shape = array_ops.concat([batch_shape, postfix], 0)

  grad_input = array_ops.matrix_set_diag(
      grad,
      array_ops.zeros(diag_shape, dtype=grad.dtype),
      k=op.inputs[2],
      align=align,
  )
  grad_diag = array_ops.matrix_diag_part(grad, k=op.inputs[2], align=align)
  return (grad_input, grad_diag, None)


@ops.RegisterGradient("MatrixBandPart")
def _MatrixBandPartGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of MatrixBandPart: applies the same band mask (same lower/upper counts from the
  # forward inputs) to the gradient. Returns the masked gradient and null for both counts.
  num_lower = op.inputs[1]
  num_upper = op.inputs[2]
  return (array_ops.matrix_band_part(grad, num_lower, num_upper), None, None)


# Edit Distance has no gradient (but can be used to eval seq2seq or CTC).
ops.NotDifferentiable("EditDistance")


@ops.RegisterGradient("Fill")
def _FillGrad(_, grad):
  # [FUNCTION DOC]
  # Gradient of Fill: null for the dims input and the sum of all gradient elements for the
  # scalar value input. The forward op is unused.
  return None, math_ops.reduce_sum(grad)


ops.NotDifferentiable("ZerosLike")
ops.NotDifferentiable("OnesLike")


@ops.RegisterGradient("PreventGradient")
def _PreventGradientGrad(op: ops.Operation, _):
  # [FUNCTION DOC]
  # Gradient of PreventGradient. Never returns: always raises a lookup error whose message
  # includes the reason string stored in the forward op's message attribute, signalling that
  # differentiation through this op was deliberately disabled.
  raise LookupError(
      "Gradient explicitly disabled. Reason: %s" % op.get_attr("message")
  )


def _IndexedSlicesToTensorNoWarning(indexed_slices):
  # [FUNCTION DOC]
  # HELPER: used by the gather, reshape and patch-extraction gradients to densify gradients.
  # Converts sparse indexed-slices to a dense tensor via an unsorted segment sum over the first
  # dimension, bypassing the sparse-to-dense conversion warning. Non-indexed-slices input is
  # returned unchanged. Raises a value error if the dense shape is missing.
  """Converts an IndexedSlices to a Tensor without sparse->dense warnings."""
  if not isinstance(indexed_slices, indexed_slices_lib.IndexedSlices):
    # If it is not IndexedSlices, it's better be a tensor.
    return indexed_slices
  if indexed_slices.dense_shape is None:
    raise ValueError(
        "Tensor conversion requested for IndexedSlices without dense_shape: %s"
        % str(indexed_slices)
    )
  return math_ops.unsorted_segment_sum(
      indexed_slices.values,
      indexed_slices.indices,
      indexed_slices.dense_shape[0],
  )


@ops.RegisterGradient("Gather")
def _GatherGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of Gather (axis 0). Computes the params shape colocated with params, reshapes the
  # densified gradient to one row per index, and returns sparse indexed-slices (values, flattened
  # indices, params shape) for params plus null for indices.
  """Gradient for Gather op."""
  # params can be large, so colocate the shape calculation with it.
  params = op.inputs[0]
  with ops.colocate_with(params):
    params_shape = array_ops.shape(params)

  # Build appropriately shaped IndexedSlices
  indices = op.inputs[1]
  size = array_ops.expand_dims(array_ops.size(indices), 0)
  values_shape = array_ops.concat([size, params_shape[1:]], 0)
  values = array_ops.reshape(
      _IndexedSlicesToTensorNoWarning(grad), values_shape
  )
  indices = array_ops.reshape(indices, size)
  return [indexed_slices_lib.IndexedSlices(values, indices, params_shape), None]


def _GetBatchIndices(params_shape, indices, batch_dims):
  # [FUNCTION DOC]
  # HELPER: used by the batched gather gradient helper.
  # KNOWN ISSUE: the docstring contains a typo ("Addds"); behavior is as described.
  # Adds per-batch offsets to the indices so that batch dimensions can be flattened: for each
  # batch dimension (from innermost batch dim outward) it adds a range scaled by the product of
  # the following parameter dimensions, reshaped to broadcast along that dimension. Inputs:
  # parameter shape, indices tensor, number of batch dimensions. Returns the offset indices.
  """Addds the batch offsets to the given indices and returns the results."""
  batch_indices = indices
  indices_dtype = indices.dtype.base_dtype
  casted_params_shape = math_ops.cast(params_shape, indices_dtype)
  accum_dim_value = array_ops.ones((), dtype=indices_dtype)
  for dim in range(batch_dims, 0, -1):
    dim_value = casted_params_shape[dim - 1]
    accum_dim_value *= casted_params_shape[dim]
    start = array_ops.zeros((), dtype=indices_dtype)
    step = array_ops.ones((), dtype=indices_dtype)
    dim_indices = math_ops.range(start, dim_value, step)
    dim_indices *= accum_dim_value
    dim_shape = array_ops.concat(
        [
            array_ops.tile([1], [dim - 1]),
            [dim_value],
            array_ops.tile([1], [array_ops.rank(indices) - dim]),
        ],
        axis=0,
    )
    batch_indices += array_ops.reshape(dim_indices, dim_shape)

  return batch_indices


def _BatchGatherGrad(
    params_shape, values, indices, batch_dims, gather_dim_size
):
  # [FUNCTION DOC]
  # HELPER: used by the GatherV2 gradient for non-zero axes and batch dimensions.
  # Computes the dense gradient of a gather whose axis is the first non-batch dimension. If there
  # are batch dimensions, indices get batch offsets, values are flattened and the segment count
  # is multiplied by the batch size. It then sums values into the gather dimension with an
  # unsorted segment sum and restores the batch dimensions by reshaping. Inputs: params shape,
  # gradient values, indices, number of batch dims, size of the gathered dimension. Returns the
  # dense gradient for params.
  """Returns the gradient of GatherV2 with batch dimensions."""

  # Axis is the first non-batch dimension.
  indices_size = array_ops.expand_dims(array_ops.size(indices), 0)
  if batch_dims:
    values_shape = array_ops.shape(values)
    # Add the batch offsets to indices and flatten the batch dimensions.
    outer_shape = values_shape[:batch_dims]
    inner_shape = values_shape[batch_dims:][1:]
    batch_size = gen_math_ops.prod(outer_shape, [0], False)
    flat_values_shape = array_ops.concat([[-1], inner_shape], 0)
    gather_dim_size *= batch_size

    indices = _GetBatchIndices(params_shape, indices, batch_dims)
    values = array_ops.reshape(
        _IndexedSlicesToTensorNoWarning(values), flat_values_shape
    )

  indices = array_ops.reshape(indices, indices_size)
  params_grad = math_ops.unsorted_segment_sum(values, indices, gather_dim_size)

  if batch_dims:
    # Put back the batch dimensions.
    params_grad = array_ops.reshape(
        params_grad, array_ops.concat([outer_shape, flat_values_shape], 0)
    )

  return params_grad


@ops.RegisterGradient("GatherV2")
def _GatherV2Grad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of GatherV2 (params, indices, axis, batch_dims attribute). Negative batch_dims is
  # normalized using the indices rank (value error if that rank is unknown). For a statically
  # zero axis it returns sparse indexed-slices; otherwise it transposes the axis to the first
  # non-batch position, uses the batched gather gradient helper, and transposes back. A dense
  # result is reshaped to the params shape to avoid zero-size mismatches. Returns the params
  # gradient plus null for indices and axis.
  """Gradient for GatherV2 op."""
  # params can be large, so colocate the shape calculation with it.
  #
  # params can be very large for sparse model, array_ops.shape raises
  # exception on the Windows platform when any dimension is larger than
  # int32. params_shape is not used in optimizer apply_sparse gradients,
  # so it's fine to convert it back to int32 regardless of truncation.
  params = op.inputs[0]
  with ops.colocate_with(params):
    params_shape = array_ops.shape(params)

  indices = op.inputs[1]
  indices_size = array_ops.expand_dims(array_ops.size(indices), 0)
  axis = op.inputs[2]
  axis_static = tensor_util.constant_value(axis)
  batch_dims = int(op.get_attr("batch_dims"))

  if batch_dims < 0:
    if indices.shape.ndims is None:
      raise ValueError(
          "Currently, it is unsupported to take the gradient of tf.gather "
          "when batch_dims < 0 and the rank of the indices is unknown. Please "
          "pass a positive batch_dims or use tf.ensure_shape to update the "
          "shape of indices when calling tf.gather. Got "
          f"batch_dims={batch_dims} and indices={indices}"
      )
    batch_dims += indices.shape.ndims

  # For axis 0 gathers, build an appropriately shaped IndexedSlices.
  if axis_static == 0:
    if context.executing_eagerly():
      with ops.device(indices_size.device):
        params_tail_shape = array_ops.identity(params_shape)[1:]
    else:
      params_tail_shape = params_shape[1:]
    values_shape = array_ops.concat([indices_size, params_tail_shape], 0)
    values = array_ops.reshape(
        _IndexedSlicesToTensorNoWarning(grad), values_shape
    )
    indices = array_ops.reshape(indices, indices_size)
    params_grad = indexed_slices_lib.IndexedSlices(
        values, indices, params_shape
    )
  else:
    # Handle axis by transposing the axis dimension to be the first non-batch
    # dimension, compute the gradient and transpose the result back.
    outer_shape = params_shape[:axis]
    inner_shape = params_shape[axis:][1:]
    values_shape = array_ops.concat([outer_shape, [-1], inner_shape], 0)

    values_dims = array_ops.size(values_shape)
    axis_dims = array_ops.size(outer_shape)

    outer_batches_indices = math_ops.range(batch_dims)
    batch_axis_indices = math_ops.range(batch_dims, axis_dims)
    inner_axes_indices = math_ops.range(axis_dims + 1, values_dims)

    values = array_ops.reshape(
        _IndexedSlicesToTensorNoWarning(grad), values_shape
    )

    # Move values[axis] up to values[batch_dims]
    transpose_dims = array_ops.concat(
        [
            outer_batches_indices,
            [axis_dims],
            batch_axis_indices,
            inner_axes_indices,
        ],
        0,
    )
    values_transpose = array_ops.transpose(values, transpose_dims)
    params_shape_transpose = array_ops.gather(params_shape, transpose_dims)

    params_grad = _BatchGatherGrad(
        params_shape_transpose,
        values_transpose,
        indices,
        batch_dims,
        params_shape[axis],
    )

    # Inverts the above transpose by moving dimension batch_dims back to its
    # original position.
    invert_transpose_dims = array_ops.concat(
        [
            outer_batches_indices,
            batch_axis_indices + 1,
            [batch_dims],
            inner_axes_indices,
        ],
        0,
    )
    params_grad = array_ops.transpose(params_grad, invert_transpose_dims)

  if not isinstance(params_grad, indexed_slices_lib.IndexedSlices):
    # Prevents mismatches in shapes when some tensor dimensions are zero.
    params_grad = array_ops.reshape(params_grad, array_ops.shape(params))

  return [params_grad, None, None]


@ops.RegisterGradient("GatherNd")
def _GatherNdGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of GatherNd. If indices are a rank-2 tensor with a statically single-element last
  # dimension it returns sparse indexed-slices; otherwise it scatters the gradient into a zeros
  # tensor of the ref shape using the op's bad-indices policy. Returns the ref gradient and null
  # for indices.
  ref = op.inputs[0]
  indices = op.inputs[1]
  ref_shape = array_ops.shape(ref, out_type=indices.dtype)
  if indices.shape.ndims == 2 and indices.shape.dims[-1].value == 1:
    ref_grad = indexed_slices_lib.IndexedSlices(
        grad, array_ops.squeeze(indices, axis=-1), ref_shape
    )
  else:
    ref_grad = array_ops.scatter_nd(
        indices,
        grad,
        ref_shape,
        bad_indices_policy=op.get_attr("bad_indices_policy"),
    )
  return [ref_grad, None]


@ops.RegisterGradient("ResourceGatherNd")
def _ResourceGatherNdGrad(op: ops.Operation, grad):  # pylint: disable=missing-docstring
  # [FUNCTION DOC]
  # KNOWN ISSUE: unlike the GatherNd gradient, the scatter here does not pass the op's
  # bad-indices policy, so that setting is ignored for resource variables.
  # Gradient of ResourceGatherNd. The shape comes from the variable resource. Returns sparse
  # indexed-slices for single-column rank-2 indices, otherwise a scattered dense gradient, plus
  # null for indices.
  ref = op.inputs[0]
  indices = op.inputs[1]
  ref_shape = gen_resource_variable_ops.variable_shape(ref, indices.dtype)
  if indices.shape.ndims == 2 and indices.shape.dims[-1].value == 1:
    ref_grad = indexed_slices_lib.IndexedSlices(
        grad, array_ops.squeeze(indices, axis=-1), ref_shape
    )
  else:
    ref_grad = array_ops.scatter_nd(
        indices,
        grad,
        ref_shape,
    )
  return [ref_grad, None]


@ops.RegisterGradient("CheckNumerics")
def _CheckNumericsGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of CheckNumerics: applies a numerics check to the incoming gradient, with an error
  # message that mentions NaN or infinity in the gradient and appends the forward op's message.
  # Returns the checked gradient (the op fails at run time if bad values are found).
  """Gradient for check_numerics op."""
  return array_ops.check_numerics(
      grad,
      "Not a number (NaN) or infinity (Inf) values detected in gradient. %s"
      % op.get_attr("message"),
  )


@ops.RegisterGradient("CheckNumericsV2")
def _CheckNumericsV2Grad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says "check_numerics op" but this is the gradient for the V2 op.
  # Gradient of CheckNumericsV2: same behavior as the V1 gradient but uses the V2 check op.
  # Returns the checked gradient.
  """Gradient for check_numerics op."""
  return array_ops.check_numerics_v2(
      grad,
      "Not a number (NaN) or infinity (Inf) values detected in gradient. %s"
      % op.get_attr("message"),
  )


@ops.RegisterGradient("PlaceholderWithDefault")
@ops.RegisterGradient("Identity")
def _IdGrad(_, grad):
  # [FUNCTION DOC]
  # Gradient shared by Identity and PlaceholderWithDefault: passes the incoming gradient through
  # unchanged.
  return grad


@ops.RegisterGradient("_EagerConst")
def _EagerConstGrad(_, grad):
  # [FUNCTION DOC]
  # Gradient of the internal eager-constant op. Must never be invoked: always raises an
  # assertion error asking the user to file a bug.
  raise AssertionError(
      "This op should never interact with gradient APIs. Please file a bug."
  )


@ops.RegisterGradient("RefIdentity")
def _RefIdGrad(_, grad):
  # [FUNCTION DOC]
  # Gradient of RefIdentity: passes the incoming gradient through unchanged.
  return grad


@ops.RegisterGradient("IdentityN")
def _IdNGrad(_, *grad):
  # [FUNCTION DOC]
  # Gradient of IdentityN: returns the tuple of incoming gradients unchanged, one per input.
  return grad


ops.NotDifferentiable("StopGradient")


@ops.RegisterGradient("Reshape")
def _ReshapeGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of Reshape. Densifies the gradient and reshapes it back to the input's shape. If the
  # input rank is known and exactly one dimension is unknown (and no dimension is zero) it uses a
  # wildcard dimension to avoid a dynamic shape dependency; otherwise it uses the dynamic input
  # shape. Returns the input gradient and null for the shape input.
  """Defines the gradient for `array_ops.reshape()`."""
  input_shape = op.inputs[0].shape
  if input_shape.rank is not None and not input_shape.is_fully_defined():
    # If only one dimension is undefined, we can use a wildcard dimension in
    # the argument to `reshape()` to avoid creating a data dependency via
    # a dynamic `shape()` operation.
    input_shape_as_list = input_shape.as_list()
    undefined_dims = []
    has_zero_dim = False
    for i, dim in enumerate(input_shape_as_list):
      if dim is None:
        undefined_dims.append(i)
      elif dim == 0:
        # When the tensor has zero elements, the wildcard dimension
        # is underspecified, and `reshape()` will arbitrarily set the unknown
        # to `1`, triggering shape errors downstream.
        has_zero_dim = True
    if len(undefined_dims) == 1 and not has_zero_dim:
      input_shape_as_list[undefined_dims[0]] = -1
      return [
          array_ops.reshape(
              _IndexedSlicesToTensorNoWarning(grad), input_shape_as_list
          ),
          None,
      ]
  return [
      array_ops.reshape(
          _IndexedSlicesToTensorNoWarning(grad), array_ops.shape(op.inputs[0])
      ),
      None,
  ]


ops.NotDifferentiable("InvertPermutation")


def _ReshapeToInput(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # HELPER: shared by the ExpandDims and Squeeze gradients.
  # Densifies the gradient and reshapes it to the dynamic shape of the op's first input.
  # Returns the reshaped tensor.
  """Reshapes the gradient to the shape of the original input."""
  return array_ops.reshape(
      _IndexedSlicesToTensorNoWarning(grad), array_ops.shape(op.inputs[0])
  )


@ops.RegisterGradient("ExpandDims")
def _ExpandDimsGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of ExpandDims: reshapes the gradient to the input shape; returns it with null for
  # the dimension input.
  return [_ReshapeToInput(op, grad), None]


@ops.RegisterGradient("Squeeze")
def _SqueezeGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of Squeeze: reshapes the gradient back to the original (unsqueezed) input shape.
  return _ReshapeToInput(op, grad)


@ops.RegisterGradient("Transpose")
def _TransposeGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of Transpose: transposes the gradient by the inverse of the forward permutation.
  # Returns the input gradient and null for the permutation input.
  """Returns unshuffle(grad)."""
  p = op.inputs[1]
  return [array_ops.transpose(grad, array_ops.invert_permutation(p)), None]


@ops.RegisterGradient("ConjugateTranspose")
def _ConjugateTransposeGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of ConjugateTranspose: conjugate-transposes the gradient by the inverse of the
  # forward permutation. Returns the input gradient and null for the permutation input.
  """Returns conj(unshuffle(grad))."""
  p = op.inputs[1]
  return [
      array_ops.transpose(
          grad, array_ops.invert_permutation(p), conjugate=True
      ),
      None,
  ]


ops.NotDifferentiable("Shape")

ops.NotDifferentiable("ShapeN")

ops.NotDifferentiable("Rank")

ops.NotDifferentiable("Size")


@ops.RegisterGradient("Tile")
def _TileGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of Tile. Interleaves the multiples and input shape into a split shape, reshapes the
  # gradient to it and sum-reduces over the tiled (even) axes. Sparse gradients are first
  # reduced along the first dimension with an unsorted segment sum using index modulo the first
  # input dimension. In graph mode it sets the static shape of the result to the input's static
  # shape. Returns the input gradient and null for the multiples input.
  """Sum reduces grad along the tiled dimensions."""
  input_shape = array_ops.shape(op.inputs[0], out_type=op.inputs[1].dtype)
  # We interleave multiples and input_shape to get split_shape,
  # reshape grad to split_shape, and reduce along all even
  # dimensions (the tiled dimensions) to get the result
  # with shape input_shape.  For example
  #   input_shape = [20, 30, 40]
  #   multiples = [2, 3, 4]
  #   split_shape = [2, 20, 3, 30, 4, 40]
  #   axes = [0, 2, 4]
  split_shape = array_ops.reshape(
      array_ops.transpose(array_ops_stack.stack([op.inputs[1], input_shape])),
      [-1],
  )
  axes = math_ops.range(0, array_ops.size(split_shape), 2)
  # Sum reduces grad along the first dimension for IndexedSlices
  if isinstance(grad, indexed_slices_lib.IndexedSlices):
    input_shape_0 = math_ops.cast(input_shape[0], grad.indices.dtype)
    grad = math_ops.unsorted_segment_sum(
        grad.values, math_ops.mod(grad.indices, input_shape_0), input_shape_0
    )
    split_shape = array_ops.concat([[1], split_shape[1:]], axis=0)
  input_grad = math_ops.reduce_sum(array_ops.reshape(grad, split_shape), axes)
  # Fix shape inference
  if not context.executing_eagerly():
    input_grad.set_shape(op.inputs[0].get_shape())
  return [input_grad, None]


ops.NotDifferentiable("BroadcastGradientArgs")


def _PadGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # KNOWN ISSUE: a stray closing triple-quote sequence appears inside a code comment in the body;
  # it is harmless but misleading. Registered for both Pad and PadV2.
  # Gradient of Pad/PadV2: slices the original input region out of the gradient, using the first
  # column of the paddings input as the begin offsets and the input shape as sizes. Returns the
  # sliced gradient plus null for paddings (and null for the constant value input when the op
  # has three inputs, i.e. PadV2).
  """Gradient for Pad."""
  # Pad introduces values around the original tensor, so the gradient function
  # slices the original shape out of the gradient."""
  x = op.inputs[0]
  a = op.inputs[1]  # [Rank(x), 2]
  # Takes a slice of a. The 1st column. [Rank(x), 1].
  pad_before = array_ops.slice(
      a, [0, 0], array_ops_stack.stack([array_ops.rank(x), 1])
  )
  # Make it a 1-D tensor.
  begin = array_ops.reshape(pad_before, [-1])
  sizes = array_ops.shape(x, out_type=begin.dtype)
  x_grad = array_ops.slice(grad, begin, sizes)
  if len(op.inputs) == 3:
    return x_grad, None, None
  else:
    return x_grad, None


ops.RegisterGradient("Pad")(_PadGrad)
ops.RegisterGradient("PadV2")(_PadGrad)


# ReverseSequence is just a permutation.  The gradient permutes back.
@ops.RegisterGradient("ReverseSequence")
def _ReverseSequenceGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of ReverseSequence: reverses the gradient again with the same batch axis, sequence
  # axis and sequence lengths. Returns it plus null for the lengths input.
  seq_lengths = op.inputs[1]
  return [
      array_ops.reverse_sequence(
          grad,
          batch_axis=op.get_attr("batch_dim"),
          seq_axis=op.get_attr("seq_dim"),
          seq_lengths=seq_lengths,
      ),
      None,
  ]


@ops.RegisterGradient("Reverse")
def _ReverseGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of Reverse: reverses the gradient along the same dimensions input. Returns it and
  # null for the dimensions input.
  reverse_dims = op.inputs[1]
  return gen_array_ops.reverse(grad, reverse_dims), None


@ops.RegisterGradient("ReverseV2")
def _ReverseV2Grad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of ReverseV2: reverses the gradient along the same axes. Returns it and null for
  # the axis input.
  axis = op.inputs[1]
  return array_ops.reverse_v2(grad, axis), None


@ops.RegisterGradient("SpaceToBatch")
def _SpaceToBatchGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of SpaceToBatch: the inverse rearrangement, batch-to-space, with the same paddings
  # input and block size attribute. Returns it and null for the paddings input.
  # Its gradient is the opposite op: BatchToSpace.
  block_size = op.get_attr("block_size")
  return [
      array_ops.batch_to_space(grad, op.inputs[1], block_size=block_size),
      None,
  ]


@ops.RegisterGradient("SpaceToBatchND")
def _SpaceToBatchNDGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of SpaceToBatchND: batch-to-space-ND using the op's block shape and paddings
  # inputs. Returns it and null for those two inputs.
  # Its gradient is the opposite op: BatchToSpaceND.
  return [
      array_ops.batch_to_space_nd(grad, op.inputs[1], op.inputs[2]),
      None,
      None,
  ]


@ops.RegisterGradient("BatchToSpace")
def _BatchToSpaceGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of BatchToSpace: space-to-batch with the same crops input (used as paddings) and
  # block size attribute. Returns it and null for the crops input.
  # Its gradient is the opposite op: SpaceToBatch.
  block_size = op.get_attr("block_size")
  return [
      array_ops.space_to_batch(grad, op.inputs[1], block_size=block_size),
      None,
  ]


@ops.RegisterGradient("BatchToSpaceND")
def _BatchToSpaceNDGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of BatchToSpaceND: space-to-batch-ND using the op's block shape and crops inputs.
  # Returns it and null for those two inputs.
  # Its gradient is the opposite op: SpaceToBatchND.
  return [
      array_ops.space_to_batch_nd(grad, op.inputs[1], op.inputs[2]),
      None,
      None,
  ]


@ops.RegisterGradient("SpaceToDepth")
def _SpaceToDepthGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of SpaceToDepth: applies depth-to-space with the same block size and data format.
  # Raises a value error if the data format is the vectorized NCHW_VECT_C layout (quantized
  # int8 only, unsupported here). Returns the input gradient.
  # Its gradient is the opposite op: DepthToSpace.
  block_size = op.get_attr("block_size")
  data_format = op.get_attr("data_format")
  if data_format == "NCHW_VECT_C":
    raise ValueError(
        "Cannot compute SpaceToDepth gradient with NCHW_VECT_C. "
        "NCHW_VECT_C requires qint8 data type."
    )
  return array_ops.depth_to_space(grad, block_size, data_format=data_format)


@ops.RegisterGradient("DepthToSpace")
def _DepthToSpaceGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of DepthToSpace: applies space-to-depth with the same block size and data format.
  # Raises a value error for the NCHW_VECT_C data format. Returns the input gradient.
  # Its gradient is the opposite op: SpaceToDepth.
  block_size = op.get_attr("block_size")
  data_format = op.get_attr("data_format")
  if data_format == "NCHW_VECT_C":
    raise ValueError(
        "Cannot compute DepthToSpace gradient with NCHW_VECT_C. "
        "NCHW_VECT_C requires qint8 data type."
    )
  return array_ops.space_to_depth(grad, block_size, data_format=data_format)


ops.NotDifferentiable("OneHot")


@ops.RegisterGradient("MirrorPad")
def _MirrorPadGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of MirrorPad: applies the mirror-pad-grad op with the same paddings and mode.
  # Returns it and null for the paddings input.
  mode = op.get_attr("mode")
  return [gen_array_ops.mirror_pad_grad(grad, op.inputs[1], mode=mode), None]


@ops.RegisterGradient("MirrorPadGrad")
def _MirrorPadGradGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of MirrorPadGrad: applies the mirror-pad op with the same paddings and mode.
  # Returns it and null for the paddings input.
  mode = op.get_attr("mode")
  return [gen_array_ops.mirror_pad(grad, op.inputs[1], mode=mode), None]


ops.NotDifferentiable("FakeQuantWithMinMaxArgsGradient")
ops.NotDifferentiable("FakeQuantWithMinMaxVarsGradient")
ops.NotDifferentiable("FakeQuantWithMinMaxVarsPerChannelGradient")


@ops.RegisterGradient("QuantizeAndDequantize")
def _QuantizeAndDequantizeGrad(_, grad):
  # [FUNCTION DOC]
  # Gradient of QuantizeAndDequantize: straight-through estimator, passes the gradient unchanged.
  return grad


@ops.RegisterGradient("QuantizeAndDequantizeV2")
def _QuantizeAndDequantizeV2Grad(_, grad):
  # [FUNCTION DOC]
  # Gradient of QuantizeAndDequantizeV2: passes the gradient through for the input and returns
  # null for the two range inputs.
  return [grad, None, None]


@ops.RegisterGradient("QuantizeAndDequantizeV3")
def _QuantizeAndDequantizeV3Grad(_, grad):
  # [FUNCTION DOC]
  # Gradient of QuantizeAndDequantizeV3: passes the gradient through for the unquantized input
  # only; returns null for the min, max and number-of-bits inputs.
  # Only propagate the gradient for the unquantized input.
  return [grad, None, None, None]


@ops.RegisterGradient("ExtractImagePatches")
def _ExtractImagePatchesGrad(op: ops.Operation, grad):  # pylint:disable=missing-function-docstring
  # [FUNCTION DOC]
  # Gradient of ExtractImagePatches. Builds an index image numbered from 1 (0 reserved for
  # padding; float32 because the XLA version lacks int64), extracts patches from it with the
  # op's ksizes, strides, rates and padding, and uses the resulting indices (shifted by one so
  # padding becomes -1, ignored) as segment ids to sum the reshaped, transposed gradient back
  # onto input pixels. Returns a one-element list holding the gradient in batch, row, column,
  # channel layout. Assumes a four-dimensional input.
  input_bhwc = array_ops.shape(op.inputs[0], out_type=dtypes.int64)
  batch_size, rows_in, cols_in, channels = array_ops_stack.unstack(input_bhwc)

  output_bhwc = array_ops.shape(op.outputs[0], out_type=dtypes.int64)
  rows_out, cols_out = array_ops_stack.unstack(output_bhwc[1:3])

  _, ksize_r, ksize_c, _ = op.get_attr("ksizes")

  # Create indices matrix for input tensor.
  # Note that 0 is preserved for padding location,
  # so indices for input start from 1 to 1 + rows_in * cols_in.
  input_indices_num = rows_in * cols_in
  # XLA version of extract_image_patches does not support int64,
  # using float32 instead.
  input_idx = array_ops.reshape(
      math_ops.range(1, input_indices_num + 1, dtype=ops.dtypes.float32),
      (1, rows_in, cols_in, 1),
  )
  input_idx_patched = gen_array_ops.extract_image_patches(
      input_idx,
      op.get_attr("ksizes"),
      op.get_attr("strides"),
      op.get_attr("rates"),
      op.get_attr("padding"),
  )
  input_idx_patched = math_ops.cast(input_idx_patched, dtypes.int64)

  grad_expanded = array_ops.transpose(
      array_ops.reshape(
          _IndexedSlicesToTensorNoWarning(grad),
          (batch_size, rows_out, cols_out, ksize_r, ksize_c, channels),
      ),
      (1, 2, 3, 4, 0, 5),
  )
  grad_flat = array_ops.reshape(grad_expanded, (-1, batch_size * channels))

  # Shift all input indices back. Padding locations will have "-1" value
  # which is fortunately ignored by segmented sum.
  segment_ids = array_ops.reshape(input_idx_patched, [-1]) - 1
  grad_out = math_ops.unsorted_segment_sum(
      grad_flat, segment_ids, num_segments=input_indices_num
  )

  grad_out = array_ops.reshape(
      grad_out, (rows_in, cols_in, batch_size, channels)
  )
  grad_out = array_ops.transpose(grad_out, (2, 0, 1, 3))

  return [grad_out]


@ops.RegisterGradient("ExtractVolumePatches")
def _ExtractVolumePatchesGrad(op: ops.Operation, grad):  # pylint:disable=missing-function-docstring
  # [FUNCTION DOC]
  # KNOWN ISSUE: the first unpacking of static batch size and channel dimensions is immediately
  # overwritten by dynamic values for batch size and channels; it also requires the plane, row
  # and column dims (and output dims) to be statically known, failing on unknown dimensions.
  # Gradient of ExtractVolumePatches. Builds input and output index tensors, maps input to
  # output positions in a sparse matrix, drops the padding row, and multiplies this sparse
  # matrix by the reshaped, transposed gradient to accumulate onto input voxels. Returns a
  # one-element list with the gradient in batch, plane, row, column, channel layout.
  batch_size, planes_in, rows_in, cols_in, channels = [
      dim.value for dim in op.inputs[0].shape.dims
  ]
  input_bphwc = array_ops.shape(op.inputs[0])
  batch_size = input_bphwc[0]
  channels = input_bphwc[4]

  # Create indices matrix for input tensor.
  # Note that 0 is preserved for padding location,
  # so indices for input start from 1 to 1 + rows_in * cols_in.
  input_indices_num = 1 + planes_in * rows_in * cols_in
  input_idx = array_ops.reshape(
      math_ops.range(1, input_indices_num, dtype=ops.dtypes.int64),
      (1, planes_in, rows_in, cols_in, 1),
  )
  input_idx_patched = gen_array_ops.extract_volume_patches(
      input_idx,
      op.get_attr("ksizes"),
      op.get_attr("strides"),
      op.get_attr("padding"),
  )

  # Create indices matrix for output tensor.
  _, planes_out, rows_out, cols_out, _ = [
      dim.value for dim in op.outputs[0].shape.dims
  ]
  _, ksize_p, ksize_r, ksize_c, _ = op.get_attr("ksizes")
  # Indices for output start from 0.
  prc_indices_num = planes_out * rows_out * cols_out
  output_indices_num = prc_indices_num * ksize_p * ksize_r * ksize_c
  output_idx = array_ops.reshape(
      math_ops.range(output_indices_num, dtype=ops.dtypes.int64),
      (1, planes_out, rows_out, cols_out, ksize_p * ksize_r * ksize_c),
  )

  # Construct mapping table for indices: (input -> output).
  idx_matrix = array_ops.concat(
      [
          array_ops.expand_dims(input_idx_patched, axis=-1),
          array_ops.expand_dims(output_idx, axis=-1),
      ],
      axis=-1,
  )
  idx_map = array_ops.reshape(idx_matrix, (-1, 2))

  sp_shape = (input_indices_num, output_indices_num)
  sp_mat_full = sparse_tensor.SparseTensor(
      idx_map, array_ops.ones([output_indices_num], dtype=grad.dtype), sp_shape
  )
  # Remove all padding locations [0, :].
  sp_mat = sparse_ops.sparse_slice(
      sp_mat_full, (1, 0), (input_indices_num - 1, output_indices_num)
  )

  grad_expanded = array_ops.transpose(
      array_ops.reshape(
          _IndexedSlicesToTensorNoWarning(grad),
          (
              batch_size,
              planes_out,
              rows_out,
              cols_out,
              ksize_p,
              ksize_r,
              ksize_c,
              channels,
          ),
      ),
      (1, 2, 3, 4, 5, 6, 0, 7),
  )
  grad_flat = array_ops.reshape(grad_expanded, (-1, batch_size * channels))

  jac = sparse_ops.sparse_tensor_dense_matmul(sp_mat, grad_flat)

  grad_out = array_ops.reshape(
      jac, (planes_in, rows_in, cols_in, batch_size, channels)
  )
  grad_out = array_ops.transpose(grad_out, (3, 0, 1, 2, 4))

  return [grad_out]


@ops.RegisterGradient("ScatterNd")
def _ScatterNdGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of ScatterNd: the updates gradient is the gather-nd of the incoming gradient at the
  # indices. Returns null for indices, the updates gradient, and null for the shape input.
  indices = op.inputs[0]
  updates_grad = array_ops.gather_nd(
      grad,
      indices,
  )
  return [None, updates_grad, None]


@ops.RegisterGradient("TensorScatterUpdate")
def _TensorScatterUpdateGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of TensorScatterUpdate. The updates gradient is the gather-nd of the incoming
  # gradient at the indices; the tensor gradient is the incoming gradient with those positions
  # overwritten by zeros. Returns the tensor gradient, null for indices, and the updates gradient.
  indices = op.inputs[1]
  updates_grad = array_ops.gather_nd(
      grad,
      indices,
  )
  tensor_grad = array_ops.tensor_scatter_update(
      array_ops.identity(grad),
      indices,
      array_ops.zeros_like(op.inputs[2], dtype=grad.dtype),
  )
  return [tensor_grad, None, updates_grad]


@ops.RegisterGradient("TensorScatterAdd")
def _TensorScatterAddGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of TensorScatterAdd: the tensor gradient is a copy of the incoming gradient and the
  # updates gradient is the gather-nd at the indices. Returns the tensor gradient, null for
  # indices, and the updates gradient.
  indices = op.inputs[1]
  updates_grad = array_ops.gather_nd(
      grad,
      indices,
  )
  tensor_grad = array_ops.identity(grad)
  return [tensor_grad, None, updates_grad]


def _TensorScatterMinOrMaxGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # HELPER: shared implementation behind the TensorScatterMax and TensorScatterMin gradients.
  # Computes indicators marking where the original tensor and the updates equal the output, and
  # divides the gradient evenly among ties. Uses the op's bad-indices policy for the gather and
  # scatter. Returns the gradient for the tensor, null for indices, and the gradient for updates.
  """Gradient for TensorScatterMin and TensorScatterMax."""
  indices = op.inputs[1]
  x = op.inputs[0]
  y = op.inputs[2]
  output = op.outputs[0]
  x_indicators = math_ops.cast(math_ops.equal(x, output), grad.dtype)
  bad_indices_policy = op.get_attr("bad_indices_policy")
  y_output = array_ops.gather_nd(
      output, indices, bad_indices_policy=bad_indices_policy
  )
  y_indicators = math_ops.cast(math_ops.equal(y, y_output), grad.dtype)
  ys_indicators = array_ops.scatter_nd(
      indices,
      y_indicators,
      array_ops.shape(x, out_type=indices.dtype),
      bad_indices_policy=bad_indices_policy,
  )
  indicators = x_indicators + ys_indicators  # All elements are >= 1.
  # If there are multiple minimum or maximum elements then the gradient will be
  # divided between them.
  x_grad = grad * x_indicators / indicators
  y_grad = (
      array_ops.gather_nd(
          grad / indicators, indices, bad_indices_policy=bad_indices_policy
      )
      * y_indicators
  )
  return [x_grad, None, y_grad]


@ops.RegisterGradient("TensorScatterMax")
def _TensorScatterMaxGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of TensorScatterMax: delegates to the shared min-or-max scatter gradient helper.
  """Gradient for TensorScatterMax op."""
  return _TensorScatterMinOrMaxGrad(op, grad)


@ops.RegisterGradient("TensorScatterMin")
def _TensorScatterMinGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of TensorScatterMin: delegates to the shared min-or-max scatter gradient helper.
  """Gradient for TensorScatterMin op."""
  return _TensorScatterMinOrMaxGrad(op, grad)


@ops.RegisterGradient("TensorScatterSub")
def _TensorScatterSubGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of TensorScatterSub: tensor gradient is a copy of the incoming gradient; updates
  # gradient is the negated gather-nd at the indices. Returns tensor gradient, null, updates
  # gradient.
  indices = op.inputs[1]
  updates_grad = array_ops.gather_nd(
      grad,
      indices,
  )
  tensor_grad = array_ops.identity(grad)
  return [tensor_grad, None, -updates_grad]


@ops.RegisterGradient("ScatterNdNonAliasingAdd")
def _ScatterNdNonAliasingAddGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of ScatterNdNonAliasingAdd: the input gradient is the incoming gradient itself and
  # the updates gradient is the gather-nd at the indices. Returns input gradient, null for
  # indices, and updates gradient.
  indices = op.inputs[1]
  updates_grad = array_ops.gather_nd(
      grad,
      indices,
  )
  return [grad, None, updates_grad]


@ops.RegisterGradient("BroadcastTo")
def _BroadcastToGrad(op: ops.Operation, grad):  # pylint:disable=missing-function-docstring
  # [FUNCTION DOC]
  # Gradient of BroadcastTo. Determines the target shape (folding it to a constant when it can
  # be statically evaluated), finds the broadcast reduction axes, sum-reduces the gradient over
  # them keeping dimensions, and reshapes to the input's dynamic shape. Returns the input
  # gradient and null for the shape input.
  input_value = op.inputs[0]
  broadcast_shape = op.inputs[1]
  shape_dtype = dtypes.int32
  if isinstance(broadcast_shape, tensor.Tensor):
    shape_dtype = broadcast_shape.dtype

  input_value_shape = array_ops.shape(input_value, out_type=shape_dtype)
  if not isinstance(broadcast_shape, ops.EagerTensor):
    broadcast_shape_static = tensor_shape.TensorShape(
        tensor_util.try_evaluate_constant(broadcast_shape)
    )
    if broadcast_shape_static.is_fully_defined():
      broadcast_shape = constant_op.constant(
          broadcast_shape_static.as_list(), dtype=shape_dtype
      )
  _, reduction_axes = gen_array_ops.broadcast_gradient_args(
      broadcast_shape, input_value_shape
  )
  updates_grad_reshaped = math_ops.reduce_sum(
      grad, axis=reduction_axes, keepdims=True
  )
  updates_grad = array_ops.reshape(updates_grad_reshaped, input_value_shape)
  return [updates_grad, None]
