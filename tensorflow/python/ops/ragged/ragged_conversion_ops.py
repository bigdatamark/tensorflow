# [MODULE DOC]
# Conversion helpers between ragged tensors and dense or sparse tensors, plus gradient
# registrations for the ragged-to-dense and ragged-to-variant/from-variant kernels.
# Exposes thin wrappers (from_tensor, to_tensor, ragged_to_dense, to_sparse, from_sparse) that
# delegate to RaggedTensor methods, passing through inputs that are already of the target kind
# where applicable. Import-time side effect: registers gradient functions for the
# RaggedTensorToTensor, RaggedTensorFromVariant and RaggedTensorToVariant ops.

# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Ops to convert between RaggedTensors and other tensor types."""

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_ragged_conversion_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_tensor


def from_tensor(tensor,
                lengths=None,
                padding=None,
                ragged_rank=1,
                row_splits_dtype=dtypes.int64,
                name=None):
  # [FUNCTION DOC]
  # Converts a dense tensor into a ragged tensor. Inputs: tensor (dense or already ragged),
  # lengths (optional per-row lengths), padding (optional padding value to strip), ragged_rank
  # (number of ragged dimensions, default 1), row_splits_dtype (integer type for row splits,
  # default 64-bit integer) and name (optional op name). If the input is already ragged it is
  # returned unchanged and the other arguments are ignored; otherwise returns the ragged tensor
  # built by the RaggedTensor from-tensor constructor. No other side effects.
  if ragged_tensor.is_ragged(tensor):
    return tensor
  else:
    return ragged_tensor.RaggedTensor.from_tensor(
        tensor,
        lengths=lengths,
        padding=padding,
        ragged_rank=ragged_rank,
        row_splits_dtype=row_splits_dtype,
        name=name)


def to_tensor(rt_input, default_value=None, name=None):
  # [FUNCTION DOC]
  # Converts a ragged tensor to a dense tensor. Inputs: rt_input (ragged or already dense
  # tensor), default_value (optional value used to pad short rows) and name (optional op
  # name). If the input is ragged, returns its dense form padded with the default; if it is
  # already dense it is returned unchanged and the other arguments are ignored.
  if ragged_tensor.is_ragged(rt_input):
    return rt_input.to_tensor(default_value, name)
  else:
    return rt_input


def ragged_to_dense(rt_input, default_value=None, shape=None):
  # [FUNCTION DOC]
  # Converts a ragged tensor to a dense one by calling its to-tensor method. Inputs: rt_input
  # (must be a ragged tensor; a dense input would fail since no ragged check is made),
  # default_value (optional padding value) and shape (optional target dense shape used to pad
  # or truncate). Returns the dense tensor.
  """Create a dense tensor from a ragged tensor."""
  return rt_input.to_tensor(default_value=default_value, shape=shape)


@ops.RegisterGradient("RaggedTensorToTensor")
def _ragged_tensor_to_tensor_grad(op, grad):
  # [FUNCTION DOC]
  # HELPER: registered gradient function for the RaggedTensorToTensor op (graph autodiff glue).
  # Inputs: op (the forward operation, whose inputs hold the flat values, default value and
  # row partition tensors, and whose attribute lists the partition types) and grad (gradient
  # of the dense output). It maps each output position back to the index of the flat value
  # that produced it (or -1 for the default), builds a mask of positions that came from flat
  # values, and returns a sparse-slices gradient for the flat values. Positions that came
  # from the default are summed over any dimensions the default was broadcast across, then
  # reshaped to the default's shape. Returns a list: nothing for the shape input, the flat
  # values gradient, the default value gradient, and nothing for each row partition tensor.
  """Gradient for RaggedToTensor op."""
  # Extract inputs from the op.
  flat_values = op.inputs[1]
  default_value = op.inputs[2]
  row_partition_tensors = op.inputs[3:]
  row_partition_types = op.get_attr("row_partition_types")
  flat_value_shape = array_ops.shape(flat_values)
  ragged_rank = sum(
      1 for typ in row_partition_types if typ != b"FIRST_DIM_SIZE")

  # Create two tensors that correspond 1:1 with grad (and op.output):
  # * indices[i1...iN] is the index in `flat_values` of the value used to
  #   populate output[i1...iN] (if the value came from `flat_values`) or
  #   -1 (if the value came from `default_value`).
  # * mask[i1...iN] is true if output[i1...iN] came from `flat_values`, or
  #   false if it came from `default_value`.
  indices = gen_ragged_conversion_ops.ragged_tensor_to_tensor(
      shape=array_ops.shape(grad)[:1 + ragged_rank],
      values=math_ops.range(flat_value_shape[0]),
      default_value=-1,
      row_partition_types=row_partition_types,
      row_partition_tensors=row_partition_tensors)
  mask = math_ops.not_equal(indices, -1)

  # Select out the gradients & indices that came from `flat_values`, and use
  # those to construct the gradient for `flat_values` (as an IndexedSlices).
  values_grad = indexed_slices.IndexedSlices(
      values=array_ops.boolean_mask(grad, mask),
      indices=array_ops.boolean_mask(indices, mask),
      dense_shape=flat_value_shape)

  # Select out the gradients that came from `default_value`, and sum them to
  # get the gradient for the default.  Note that the default_value may have
  # been broadcast as part of the RaggedTensorToTensor operation, so we also
  # need to reduce any dimensions that might have been broadcast.
  default_grads = array_ops.boolean_mask(grad, ~mask)
  dims_to_reduce = math_ops.range(
      array_ops.rank(default_grads) -
      _rank_ignoring_leading_dims_with_size_1(default_value))
  default_grad = math_ops.reduce_sum(default_grads, axis=dims_to_reduce)

  # Restore any leading dims with size one.
  default_grad = array_ops.reshape(default_grad, array_ops.shape(default_value))

  return ([None, values_grad, default_grad] +
          [None for _ in row_partition_tensors])


def _rank_ignoring_leading_dims_with_size_1(value):
  # [FUNCTION DOC]
  # HELPER: used by the RaggedTensorToTensor gradient to reduce broadcast default values.
  # Computes the rank of a tensor after discarding leading dimensions of size one. Input:
  # value, a tensor. Uses static shape when fully informative up to the first non-one
  # dimension (returns a Python integer); otherwise computes it dynamically from the runtime
  # shape (leading ones counted via a cumulative product) and returns a scalar tensor.
  """Returns `rank(value)`, ignoring any leading dimensions with size 1."""
  # Compute the result using static shape, if possible.
  if value.shape.rank is not None:
    ndims = value.shape.rank
    for dim in value.shape.dims:
      if dim.value == 1:
        ndims -= 1
      elif dim.value is None:
        ndims = None  # Can't compute the result using static shape.
        break
      else:
        break
    if ndims is not None:
      return ndims

  # Otherwise, we need to compute the result dynamically.  The math we use to
  # do this is a bit round-about, so here's an example to illustrate:
  #              shape = [1, 1, 3, 5, 1, 4]  # shape(value)
  #         dim_is_one = [1, 1, 0, 0, 1, 0]  # equal(shape, 1)
  #       leading_ones = [1, 1, 0, 0, 0, 0]  # cumprod(dim_is_one)
  #   num_leading_ones = 2                   # reduce_sum(leading_ones)
  #             result = 4                   # rank(value) - num_leading_ones
  shape = array_ops.shape(value)
  dim_is_one = math_ops.cast(math_ops.equal(shape, 1), dtypes.int32)
  leading_ones = math_ops.cumprod(dim_is_one)
  num_leading_ones = math_ops.reduce_sum(leading_ones)
  return array_ops.rank(value) - num_leading_ones


def to_sparse(rt_input, name=None):
  # [FUNCTION DOC]
  # Converts a ragged tensor to a sparse tensor by calling its to-sparse method. Inputs:
  # rt_input (a ragged tensor; dense input is not accepted) and name (optional op name).
  # Returns the sparse tensor.
  return rt_input.to_sparse(name)


def from_sparse(st_input, name=None):
  # [FUNCTION DOC]
  # Converts a two-dimensional sparse tensor to a ragged tensor via the RaggedTensor
  # from-sparse constructor. Inputs: st_input (sparse tensor) and name (optional op name).
  # Returns the ragged tensor; invalid sparse inputs raise errors from the constructor.
  return ragged_tensor.RaggedTensor.from_sparse(st_input, name)


@ops.RegisterGradient("RaggedTensorFromVariant")
def _ragged_tensor_from_variant_grad(op, *grads):
  # [FUNCTION DOC]
  # HELPER: registered gradient function for the RaggedTensorFromVariant op.
  # Inputs: op (the forward operation; the rank of its variant input and its output ragged
  # rank attribute decide whether the encoding was batched) and grads (gradients of the
  # outputs; only the last, the flat values gradient, is used). Re-encodes the gradient as a
  # variant using the ragged-to-variant kernel with the op's row splits outputs. Returns a
  # one-element list. Raises a value error when the variant input has rank above one.
  """Gradient for RaggedTensorFromVariant op."""

  variant_rank = op.inputs[0].shape.rank
  if variant_rank == 0:
    batched_input = False
  elif variant_rank == 1:
    batched_input = True
  elif variant_rank is None:
    batched_input = (op.get_attr("output_ragged_rank") > 0)
  else:
    # TODO(edloper): Add a batch_dims argument to RaggedTensorToVariant, so
    # we can support this.
    raise ValueError("Unable to compute gradient: RaggedTensorToVariant "
                     "can currently only generate 0D or 1D output.")
  return [
      gen_ragged_conversion_ops.ragged_tensor_to_variant(
          rt_nested_splits=op.outputs[:-1],
          rt_dense_values=grads[-1],
          batched_input=batched_input)
  ]


@ops.RegisterGradient("RaggedTensorToVariant")
def _ragged_tensor_to_variant_grad(op, encoded_ragged_grad):
  # [FUNCTION DOC]
  # HELPER: registered gradient function for the RaggedTensorToVariant op.
  # Inputs: op (the forward operation; its last input is the dense flat values and the earlier
  # inputs are row splits) and encoded_ragged_grad (the variant-encoded gradient). Decodes it
  # with the variant-gradient kernel into a gradient for the flat values. Returns a list with
  # nothing for each row splits input followed by the flat values gradient.
  """Gradient for RaggedTensorToVariant op."""
  dense_values = op.inputs[-1]
  ragged_rank = len(op.inputs) - 1
  row_splits = 0 if ragged_rank == 0 else op.inputs[0]
  values_grad = gen_ragged_conversion_ops.ragged_tensor_to_variant_gradient(
      encoded_ragged_grad=encoded_ragged_grad,
      row_splits=row_splits,
      dense_values_shape=array_ops.shape(dense_values),
      Tvalues=op.inputs[-1].dtype)
  result = [None] * ragged_rank + [values_grad]
  return result
