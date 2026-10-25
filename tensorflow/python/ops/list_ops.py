# [MODULE DOC]
# Python wrappers and gradient registrations for TensorList ops (lists of tensors used by
# TensorArray v2, control flow and autograph). Re-exports everything from the generated list
# ops through a wildcard import and adds friendlier wrappers (reserve, from tensor, get/set
# item, pop back, gather, scatter, stack, concat, split) that normalize the element shape
# argument and attach type/handle metadata to eager results. Also registers gradient
# functions for the list ops and marks length, element shape, concat-lists and batched push
# back as not differentiable. Import-time side effects: gradient registrations in the global
# registry. Used mainly by TensorArray and while-loop code rather than called by end users.
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
"""Ops to manipulate lists of tensors."""

# pylint: disable=g-bad-name
import numpy as np

from tensorflow.core.framework import full_type_pb2
from tensorflow.python.framework import cpp_shape_inference_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_list_ops
from tensorflow.python.ops import handle_data_util
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_list_ops import *
# pylint: enable=wildcard-import


ops.NotDifferentiable("TensorListConcatLists")
ops.NotDifferentiable("TensorListElementShape")
ops.NotDifferentiable("TensorListLength")
ops.NotDifferentiable("TensorListPushBackBatch")


def empty_tensor_list(element_shape,
                      element_dtype,
                      max_num_elements=None,
                      name=None):
  # [FUNCTION DOC]
  # Creates an empty tensor list. Inputs: element_shape (none, tensor, tensor shape or list of
  # dimensions; unknown shape is allowed), element_dtype (data type of the elements),
  # max_num_elements (optional limit; none means unlimited and is sent as minus one), name
  # (optional). Returns the list handle tensor. Unlike the reserve and from-tensor wrappers it
  # does not attach handle metadata to eager results. No other side effects.
  if max_num_elements is None:
    max_num_elements = -1

  return gen_list_ops.empty_tensor_list(
      element_shape=_build_element_shape(element_shape),
      element_dtype=element_dtype,
      max_num_elements=max_num_elements,
      name=name)


def _set_handle_data(list_handle, element_shape, element_dtype):
  # [FUNCTION DOC]
  # HELPER: makes eager list handles carry the same type information that graph tensors get.
  # Only acts when list_handle is an eager tensor; otherwise does nothing. Converts the
  # element shape to a tensor shape (unknown rank if it is itself a tensor), builds handle
  # data marked as set with one entry holding the shape, the element dtype and the array full
  # type, and stores it on the handle's private handle-data attribute. Returns nothing.
  # Mutates list_handle. Inputs: list_handle (tensor), element_shape (shape-like or tensor),
  # element_dtype (data type object).
  """Sets type information on `list_handle` for consistency with graphs."""
  # TODO(b/169968286): It would be better if we had a consistent story for
  # creating handle data from eager operations (shared with VarHandleOp).
  if isinstance(list_handle, ops.EagerTensor):
    if tensor_util.is_tf_type(element_shape):
      element_shape = tensor_shape.TensorShape(None)
    elif not isinstance(element_shape, tensor_shape.TensorShape):
      element_shape = tensor_shape.TensorShape(element_shape)
    handle_data = cpp_shape_inference_pb2.CppShapeInferenceResult.HandleData()
    handle_data.is_set = True
    # TODO(b/191472076): This duplicates type inference. Clean up.
    handle_data.shape_and_type.append(
        cpp_shape_inference_pb2.CppShapeInferenceResult.HandleShapeAndType(
            shape=element_shape.as_proto(),
            dtype=element_dtype.as_datatype_enum,
            type=full_type_pb2.FullTypeDef(type_id=full_type_pb2.TFT_ARRAY)))
    list_handle._handle_data = handle_data  # pylint: disable=protected-access


def tensor_list_reserve(element_shape, num_elements, element_dtype, name=None):
  # [FUNCTION DOC]
  # Creates a tensor list pre-sized to num_elements uninitialized slots. Inputs: element_shape
  # (none, tensor, tensor shape or list of dimensions), num_elements (integer or scalar
  # tensor), element_dtype, name (optional). Calls the raw reserve op with a normalized
  # element shape, then attaches eager handle metadata. Returns the list handle tensor.
  result = gen_list_ops.tensor_list_reserve(
      element_shape=_build_element_shape(element_shape),
      num_elements=num_elements,
      element_dtype=element_dtype,
      name=name)
  # TODO(b/169968286): gen_ops needs to ensure the metadata is properly
  # populated for eager operations.
  _set_handle_data(result, element_shape, element_dtype)
  return result


def tensor_list_from_tensor(tensor, element_shape, name=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the eager handle metadata uses the shape of the whole tensor (including the
  # leading dimension) as the element shape, which does not describe a single element.
  # Creates a tensor list whose elements are the slices of the input tensor along its first
  # dimension. Inputs: tensor (anything convertible to a tensor), element_shape (shape of each
  # element, normalized before use), name (optional). Returns the list handle. The eager
  # handle metadata is set using the full shape of the input tensor rather than the given
  # element shape.
  tensor = ops.convert_to_tensor(tensor)
  result = gen_list_ops.tensor_list_from_tensor(
      tensor=tensor,
      element_shape=_build_element_shape(element_shape),
      name=name)
  _set_handle_data(result, tensor.shape, tensor.dtype)
  return result


def tensor_list_get_item(input_handle, index, element_dtype, element_shape=None,
                         name=None):
  # [FUNCTION DOC]
  # Reads one element of a tensor list. Inputs: input_handle (list tensor), index (integer
  # scalar tensor), element_dtype, element_shape (optional, unknown if omitted), name
  # (optional). Returns the element tensor. Thin wrapper over the raw op with normalized
  # element shape. Out-of-range or uninitialized access errors come from the op at run time.
  return gen_list_ops.tensor_list_get_item(
      input_handle=input_handle,
      index=index,
      element_shape=_build_element_shape(element_shape),
      element_dtype=element_dtype,
      name=name)


def tensor_list_pop_back(input_handle, element_dtype, name=None):
  # [FUNCTION DOC]
  # Removes the last element of a list. Inputs: input_handle (list tensor), element_dtype,
  # name (optional). Calls the raw pop-back op with an unknown element shape. Returns the
  # pair produced by the op: the shortened list handle and the removed element. Fails at run
  # time on an empty list.
  return gen_list_ops.tensor_list_pop_back(
      input_handle=input_handle,
      element_shape=-1,
      element_dtype=element_dtype,
      name=name)


def tensor_list_gather(input_handle,
                       indices,
                       element_dtype,
                       element_shape=None,
                       name=None):
  # [FUNCTION DOC]
  # Gathers the elements at the given positions into one stacked tensor. Inputs: input_handle
  # (list tensor), indices (integer tensor of positions), element_dtype, element_shape
  # (optional, normalized), name (optional). Returns a tensor whose first dimension follows
  # indices. Thin wrapper over the raw gather op.
  return gen_list_ops.tensor_list_gather(
      input_handle=input_handle,
      indices=indices,
      element_shape=_build_element_shape(element_shape),
      element_dtype=element_dtype,
      name=name)


def tensor_list_scatter(tensor,
                        indices,
                        element_shape=None,
                        input_handle=None,
                        name=None):
  # [FUNCTION DOC]
  # Scatters the slices of a tensor into a list at the given positions. Inputs: tensor
  # (convertible, sliced along first dimension), indices (integer tensor), element_shape
  # (used only when creating a new list; optional), input_handle (optional existing list to
  # update), name (optional). If input_handle is given, writes into that list with the
  # scatter-into-existing op and copies the input handle's type metadata to the output;
  # otherwise creates a new list with unlimited size and sets eager handle metadata. Returns
  # the output list handle.
  """Returns a TensorList created or updated by scattering `tensor`."""
  tensor = ops.convert_to_tensor(tensor)
  if input_handle is not None:
    output_handle = gen_list_ops.tensor_list_scatter_into_existing_list(
        input_handle=input_handle, tensor=tensor, indices=indices, name=name)
    handle_data_util.copy_handle_data(input_handle, output_handle)
    return output_handle
  else:
    output_handle = gen_list_ops.tensor_list_scatter_v2(
        tensor=tensor,
        indices=indices,
        element_shape=_build_element_shape(element_shape),
        num_elements=-1,
        name=name)
    _set_handle_data(output_handle, element_shape, tensor.dtype)
    return output_handle


def tensor_list_stack(input_handle,
                      element_dtype,
                      num_elements=-1,
                      element_shape=None,
                      name=None):
  # [FUNCTION DOC]
  # Stacks all list elements into one tensor with a new leading dimension. Inputs:
  # input_handle (list tensor), element_dtype, num_elements (expected length, default minus
  # one meaning unknown), element_shape (optional, normalized), name (optional). Returns the
  # stacked tensor. Elements must have compatible shapes, otherwise the op fails at run time.
  return gen_list_ops.tensor_list_stack(
      input_handle=input_handle,
      element_shape=_build_element_shape(element_shape),
      element_dtype=element_dtype,
      num_elements=num_elements,
      name=name)


def tensor_list_concat(input_handle, element_dtype, element_shape=None,
                       name=None):
  # [FUNCTION DOC]
  # Concatenates all list elements along their first dimension. Inputs: input_handle (list
  # tensor), element_dtype, element_shape (optional, normalized), name (optional). Calls the
  # version 2 concat op with an empty leading-dimensions argument and returns only its first
  # output (the concatenated tensor); the lengths output is discarded because only gradient
  # code needs it.
  # Ignore the lengths output of TensorListConcat. It is only used during
  # gradient computation.
  return gen_list_ops.tensor_list_concat_v2(
      input_handle=input_handle,
      element_dtype=element_dtype,
      element_shape=_build_element_shape(element_shape),
      leading_dims=ops.convert_to_tensor([], dtype=dtypes.int64),
      name=name)[0]


def tensor_list_split(tensor, element_shape, lengths, name=None):
  # [FUNCTION DOC]
  # Splits a tensor along its first dimension into list elements of the given sizes (inverse
  # of concat). Inputs: tensor, element_shape (normalized), lengths (1-D integer tensor whose
  # sum should equal the first dimension), name (optional). Returns the list handle. Does not
  # set eager handle metadata.
  return gen_list_ops.tensor_list_split(
      tensor=tensor,
      element_shape=_build_element_shape(element_shape),
      lengths=lengths,
      name=name)


def tensor_list_set_item(input_handle,
                         index,
                         item,
                         resize_if_index_out_of_bounds=False,
                         name=None):
  # [FUNCTION DOC]
  # Returns a new list equal to the input with the element at index replaced by item. Inputs:
  # input_handle (list tensor), index (integer scalar), item (tensor), resize_if_index_out_of_
  # bounds (boolean, default false; when true a too-large index grows the list instead of
  # failing), name (optional). Copies the type metadata of the input handle onto the output.
  """Sets `item` at `index` in input list."""
  output_handle = gen_list_ops.tensor_list_set_item(
      input_handle=input_handle,
      index=index,
      item=item,
      name=name,
      resize_if_index_out_of_bounds=resize_if_index_out_of_bounds,
  )
  handle_data_util.copy_handle_data(input_handle, output_handle)
  return output_handle


@ops.RegisterGradient("TensorListPushBack")
def _PushBackGrad(op: ops.Operation, dresult):
  # [FUNCTION DOC]
  # HELPER: registered gradient for the push-back op.
  # Pops the last element off the incoming list gradient, using the shape of the pushed
  # tensor (second input) as element shape and the op's element type attribute. Returns the
  # pair of popped list and element gradient, matching the list and tensor inputs.
  return gen_list_ops.tensor_list_pop_back(
      dresult,
      element_shape=array_ops.shape(op.inputs[1]),
      element_dtype=op.get_attr("element_dtype"))


@ops.RegisterGradient("TensorListPopBack")
def _PopBackGrad(op: ops.Operation, dlist, delement):
  # [FUNCTION DOC]
  # HELPER: registered gradient for the pop-back op.
  # KNOWN ISSUE: when the list gradient is missing the code reads the dtype of the element
  # gradient, so if both gradients are missing it fails with an attribute error on a null value.
  # A missing list gradient becomes an empty list with the op's output element shape; a
  # missing element gradient becomes zeros shaped like the popped element. Pushes the element
  # gradient back onto the list gradient. Returns the new list gradient and no gradient for
  # the second input.
  if dlist is None:
    dlist = empty_tensor_list(
        element_dtype=delement.dtype,
        element_shape=gen_list_ops.tensor_list_element_shape(
            op.outputs[0], shape_type=dtypes.int32))
  if delement is None:
    delement = array_ops.zeros_like(op.outputs[1])
  return gen_list_ops.tensor_list_push_back(dlist, delement), None


@ops.RegisterGradient("TensorListStack")
def _TensorListStackGrad(unused_op: ops.Operation, dtensor):
  # [FUNCTION DOC]
  # HELPER: registered gradient for the stack op. Turns the gradient tensor back into a list
  # whose elements are its slices (element shape is the gradient shape without the leading
  # dimension). Returns that list gradient and no gradient for the element shape input.
  return tensor_list_from_tensor(dtensor, element_shape=dtensor.shape[1:]), None


@ops.RegisterGradient("TensorListConcat")
@ops.RegisterGradient("TensorListConcatV2")
def _TensorListConcatGrad(op: ops.Operation, dtensor, unused_dlengths):
  # [FUNCTION DOC]
  # HELPER: registered gradient for both versions of the concat op.
  # Splits the gradient tensor into a list using the forward op's element shape and its
  # recorded lengths output. For the version 2 op returns the list gradient plus two null
  # gradients (element shape and leading dimensions inputs); for the first version returns
  # only the list gradient. The lengths gradient is ignored.
  """Gradient function for TensorListConcat."""
  dlist = tensor_list_split(
      dtensor,
      element_shape=gen_list_ops.tensor_list_element_shape(
          op.inputs[0], shape_type=dtypes.int32),
      lengths=op.outputs[1])
  if op.type == "TensorListConcatV2":
    return dlist, None, None
  else:
    return dlist


@ops.RegisterGradient("TensorListSplit")
def _TensorListSplitGrad(op: ops.Operation, dlist):
  # [FUNCTION DOC]
  # HELPER: registered gradient for the split op. Computes an element shape of unknown first
  # dimension followed by the trailing dimensions of the forward input tensor, then
  # concatenates the list gradient using the forward lengths as leading dimensions. Returns
  # the concatenated tensor gradient and null gradients for the element shape and lengths.
  tensor, _, lengths = op.inputs
  element_shape = array_ops.slice(array_ops.shape(tensor), [1], [-1])
  element_shape = array_ops.concat([[-1], element_shape], axis=0)
  return gen_list_ops.tensor_list_concat_v2(
      dlist,
      element_shape=element_shape,
      leading_dims=lengths,
      element_dtype=op.inputs[0].dtype)[0], None, None


@ops.RegisterGradient("TensorListFromTensor")
def _TensorListFromTensorGrad(op: ops.Operation, dlist):
  # [FUNCTION DOC]
  # HELPER: registered gradient for the from-tensor op. Determines the element count when the
  # input's first dimension is statically known. A missing list gradient is replaced by an
  # empty list. Stacks the list gradient back into a tensor with the per-element shape of the
  # forward input. Returns that tensor gradient and no gradient for the element shape.
  """Gradient for TensorListFromTensor."""
  t = op.inputs[0]
  if t.shape.dims and t.shape.dims[0].value is not None:
    num_elements = t.shape.dims[0].value
  else:
    num_elements = None
  if dlist is None:
    dlist = empty_tensor_list(
        element_dtype=t.dtype,
        element_shape=gen_list_ops.tensor_list_element_shape(
            op.outputs[0], shape_type=dtypes.int32))
  tensor_grad = gen_list_ops.tensor_list_stack(
      dlist,
      element_shape=array_ops.slice(array_ops.shape(t), [1], [-1]),
      element_dtype=t.dtype,
      num_elements=num_elements)
  shape_grad = None
  return tensor_grad, shape_grad


@ops.RegisterGradient("TensorListGetItem")
def _TensorListGetItemGrad(op: ops.Operation, ditem):
  # [FUNCTION DOC]
  # HELPER: registered gradient for the get-item op. Builds a list of the same length and
  # element shape as the forward input list (reserved, uninitialized slots) and sets the
  # element at the read index to the item gradient. Returns the list gradient plus null
  # gradients for the index and element shape inputs.
  """Gradient for TensorListGetItem."""
  list_size = gen_list_ops.tensor_list_length(op.inputs[0])
  list_grad = gen_list_ops.tensor_list_set_item(
      gen_list_ops.tensor_list_reserve(
          gen_list_ops.tensor_list_element_shape(op.inputs[0],
                                                 shape_type=dtypes.int32),
          list_size, element_dtype=ditem.dtype),
      index=op.inputs[1],
      item=ditem)
  index_grad = None
  element_shape_grad = None
  return list_grad, index_grad, element_shape_grad


@ops.RegisterGradient("TensorListSetItem")
def _TensorListSetItemGrad(op: ops.Operation, dlist):
  # [FUNCTION DOC]
  # HELPER: registered gradient for the set-item op. The list gradient is the incoming one
  # with the written slot zeroed; the element gradient is the incoming gradient read at the
  # index (with the item's shape). If the forward op was allowed to resize, the list
  # gradient is resized back to the original input list length. Returns list gradient, no
  # index gradient and the element gradient.
  """Gradient function for TensorListSetItem."""
  input_list, index, item = op.inputs
  list_grad = gen_list_ops.tensor_list_set_item(
      dlist, index=index, item=array_ops.zeros_like(item)
  )
  index_grad = None
  element_grad = tensor_list_get_item(
      dlist,
      index,
      element_shape=array_ops.shape(item),
      element_dtype=item.dtype,
  )
  if op.get_attr(
      "resize_if_index_out_of_bounds"
  ):
    input_list_size = gen_list_ops.tensor_list_length(input_list)
    list_grad = gen_list_ops.tensor_list_resize(list_grad, input_list_size)
  return list_grad, index_grad, element_grad


@ops.RegisterGradient("TensorListResize")
def _TensorListResizeGrad(op: ops.Operation, dlist):
  # [FUNCTION DOC]
  # HELPER: registered gradient for the resize op. Resizes the list gradient back to the
  # original input list length. Returns it together with no gradient for the size input.
  input_list, _ = op.inputs
  input_list_size = gen_list_ops.tensor_list_length(input_list)
  return gen_list_ops.tensor_list_resize(dlist, input_list_size), None


@ops.RegisterGradient("TensorListGather")
def _TensorListGatherGrad(op: ops.Operation, dtensor):
  # [FUNCTION DOC]
  # HELPER: registered gradient for the gather op. Reserves a list with the forward list's
  # length and element shape and scatters the gradient tensor into it at the gathered
  # indices. Returns the list gradient and null gradients for indices and element shape.
  """Gradient function for TensorListGather."""
  input_list, indices, _ = op.inputs
  element_shape = gen_list_ops.tensor_list_element_shape(
      input_list, shape_type=dtypes.int32)
  num_elements = gen_list_ops.tensor_list_length(input_list)
  dlist = tensor_list_reserve(element_shape, num_elements, dtensor.dtype)
  dlist = tensor_list_scatter(
      tensor=dtensor, indices=indices, input_handle=dlist)
  return dlist, None, None


@ops.RegisterGradient("TensorListScatter")
@ops.RegisterGradient("TensorListScatterV2")
def _TensorListScatterGrad(op: ops.Operation, dlist):
  # [FUNCTION DOC]
  # HELPER: registered gradient for both versions of the scatter op. Gathers the list
  # gradient at the scatter indices into a tensor with the forward input's element shape.
  # Returns the tensor gradient followed by null gradients for the remaining inputs (two for
  # the first version, three for the second version).
  """Gradient function for TensorListScatter."""
  tensor = op.inputs[0]
  indices = op.inputs[1]
  dtensor = gen_list_ops.tensor_list_gather(
      dlist,
      indices,
      element_shape=array_ops.slice(array_ops.shape(tensor), [1], [-1]),
      element_dtype=tensor.dtype)
  if op.type == "TensorListScatterV2":
    return dtensor, None, None, None
  else:
    return dtensor, None, None


@ops.RegisterGradient("TensorListScatterIntoExistingList")
def _TensorListScatterIntoExistingListGrad(op: ops.Operation, dlist):
  # [FUNCTION DOC]
  # HELPER: registered gradient for the scatter-into-existing-list op.
  # KNOWN ISSUE: the call that zeroes the scattered slots passes the indices tensor as the
  # element shape argument; this is harmless only because the shape is ignored when an input
  # list is supplied, but it looks like a mistake.
  # The tensor gradient is the list gradient gathered at the indices. The list gradient is
  # the incoming one with those slots overwritten by zeros. Returns list gradient, tensor
  # gradient and no gradient for the indices.
  """Gradient function for TensorListScatterIntoExistingList."""
  _, tensor, indices = op.inputs
  dtensor = gen_list_ops.tensor_list_gather(
      dlist,
      indices,
      element_shape=array_ops.slice(array_ops.shape(tensor), [1], [-1]),
      element_dtype=tensor.dtype)
  zeros = array_ops.zeros_like(tensor)
  dlist = tensor_list_scatter(zeros, indices, indices, input_handle=dlist)
  return dlist, dtensor, None


def _build_element_shape(shape):
  # [FUNCTION DOC]
  # HELPER: normalizes element shape arguments for all list op wrappers.
  # Tensors pass through unchanged. A tensor shape of unknown rank, or none, becomes minus
  # one. Numpy arrays, numpy scalars and empty or scalar shapes become an int32 tensor. Any
  # other sequence becomes a list in which unknown dimensions are minus one, tensors are kept
  # and dimension objects are replaced by their value. Returns the converted shape; raises
  # nothing itself and has no side effects.
  """Converts shape to a format understood by list_ops for element_shape.

  If `shape` is already a `Tensor` it is returned as-is. We do not perform a
  type check here.

  If shape is None or a TensorShape with unknown rank, -1 is returned.

  If shape is a scalar, an int32 tensor with empty list is returned. Note we
  do directly return an empty list since ops.convert_to_tensor would convert it
  to a float32 which is not a valid type for element_shape.

  If shape is a sequence of dims, None's in the list are replaced with -1. We
  do not check the dtype of the other dims.

  Args:
    shape: Could be None, Tensor, TensorShape or a list of dims (each dim could
      be a None, scalar or Tensor).

  Returns:
    A None-free shape that can be converted to a tensor.
  """
  if isinstance(shape, tensor_lib.Tensor):
    return shape
  if isinstance(shape, tensor_shape.TensorShape):
    # `TensorShape.as_list` requires rank to be known.
    shape = shape.as_list() if shape else None
  # Shape is unknown.
  if shape is None:
    return -1
  # Shape is numpy array or a scalar.
  if isinstance(shape, (np.ndarray, np.generic)) or not shape:
    return ops.convert_to_tensor(shape, dtype=dtypes.int32)
  # Shape is a sequence of dimensions. Convert None dims to -1.
  def convert(val):
    # [FUNCTION DOC]
    # HELPER: converts one dimension of an element shape list. Unknown (null) becomes minus
    # one, tensors are returned as is, dimension objects give their value (minus one when
    # unknown), and anything else is returned unchanged.
    if val is None:
      return -1
    if isinstance(val, tensor_lib.Tensor):
      return val
    if isinstance(val, tensor_shape.Dimension):
      return val.value if val.value is not None else -1
    return val

  return [convert(d) for d in shape]
