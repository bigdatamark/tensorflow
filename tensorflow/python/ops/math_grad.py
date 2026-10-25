# [MODULE DOC]
# Gradient (backprop) functions for the math operators defined in math_ops: reductions (sum, mean,
# min, max, prod, euclidean norm), segment and sparse-segment reductions, unary elementwise math
# (exp, log, trig, hyperbolic, special functions such as erf, gamma family, bessel family),
# binary elementwise math (add, sub, mul, div, pow, min/max, atan2, xlogy family), matrix products
# (dense, sparse, batched), complex-number ops, casts and cumulative ops.
# Each gradient function takes the forward operation and the incoming gradient(s) with respect to
# its output(s) and returns the gradient(s) with respect to each input (None for inputs that are not
# differentiable). They are attached to op type names by the gradient registry decorator, and a
# few op types are declared non-differentiable. Importing the module therefore has the import-time
# side effect of populating the global gradient registry; nothing is exported for direct use except
# the broadcast-gradient helper. Broadcast-aware helpers at the top reduce gradients back to the
# shapes of broadcasting binary-op inputs, and several functions honor the optional per-op list of
# input indices whose gradients are not needed.
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
"""Gradients for operators defined in math_ops.py."""
import numpy as np

from tensorflow.python.compat import compat
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices as indexed_slices_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops


@ops.RegisterGradient("ArgMax")
def _ArgMaxGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for the ArgMax op. The op is not differentiable: both arguments (the
  # incoming gradient and the op) are discarded and a list of two null values is returned, one for
  # the input tensor and one for the dimension argument. No side effects.
  del op, grad
  return [None, None]


@ops.RegisterGradient("ArgMin")
def _ArgMinGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for the ArgMin op. The op is not differentiable: arguments are discarded and
  # a list of two null values is returned (input tensor, dimension). No side effects.
  del op, grad
  return [None, None]


@ops.RegisterGradient("EuclideanNorm")
def _EuclideanNormGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for the EuclideanNorm reduction op. Inputs: the forward op (input tensor and
  # reduction axes) and the gradient of the reduced output. If the op did not keep reduced
  # dimensions, both the output and the gradient are reshaped to the kept-dims shape so they broadcast
  # against the input. Returns a pair: the input divided by the output divided by the gradient (that
  # is, input times gradient over norm) and a null value for the axes. No side effects.
  """Gradient for EuclideanNorm."""

  output = op.outputs[0]

  if not op.get_attr("keep_dims"):
    output_shape_kept_dims = math_ops.reduced_shape(
        array_ops.shape(op.inputs[0]), op.inputs[1])
    output = array_ops.reshape(output, output_shape_kept_dims)
    grad = array_ops.reshape(grad, output_shape_kept_dims)

  return math_ops.truediv(op.inputs[0], output / grad), None


def SmartBroadcastGradientArgs(x, y, grad=None):
  # [FUNCTION DOC]
  # HELPER: used by the gradient functions of broadcasting binary ops to learn how to sum gradients back.
  # Variant of the broadcast-gradient-args op that avoids creating a dynamic op when shapes are
  # partially known. Inputs: x and y, the two operands of a broadcasting binary op; grad is
  # deprecated and ignored. In graph mode with both operands being symbolic tensors, it tries to
  # infer statically which axes were broadcast; if inference fails (unknown rank or data-dependent
  # dimensions) or in eager mode it falls back to the dynamic broadcast-gradient-args op.
  # Returns two triples, one per operand: the dynamic shape tensor of the operand, the axes to sum
  # over (a list or tensor), and a flag saying whether a reduction is needed at all (always true on the
  # dynamic fallback; on the static path true when there are axes or the operand has lower rank than
  # the other). Creates shape ops in the current graph; raises nothing itself.
  """Version of `BroadcastGradientArgs` optimized for partially-known shapes.

  Args:
    x: The first argument of a broadcasting binary op.
    y: The second argument of a broadcasting binary op.
    grad: Deprecated.

  Returns:
    A pair of triples, one per argument with
      * Shape of the argument (tensor);
      * Reduction axes for the argument (list or tensor);
      * Boolean indicating whether the reduction must be applied.
  """
  del grad
  x_shape = array_ops.shape(x)
  y_shape = array_ops.shape(y)

  if (not context.executing_eagerly() and
      isinstance(x, tensor.Tensor) and
      isinstance(y, tensor.Tensor)):
    x_axes, y_axes = _InferGradientReductionAxes(x.shape, y.shape)
  else:
    x_axes, y_axes = None, None

  if x_axes is None or y_axes is None:
    # NOTE: In graph mode, this is never exercised for statically known shapes.
    x_axes, y_axes = gen_array_ops.broadcast_gradient_args(x_shape, y_shape)
    x_must_reduce = True
    y_must_reduce = True
  else:
    x_must_reduce = x_axes or x.shape.rank < y.shape.rank
    y_must_reduce = y_axes or y.shape.rank < x.shape.rank

  return (x_shape, x_axes, x_must_reduce), (y_shape, y_axes, y_must_reduce)


def _InferGradientReductionAxes(x_shape, y_shape):
  # [FUNCTION DOC]
  # HELPER: static-shape part of SmartBroadcastGradientArgs.
  # Given the static shapes of the two operands of a broadcasting op, computes which axes of the
  # broadcast result each operand was stretched along. Right-aligns the shapes, treating missing
  # leading dimensions as size one; an axis is recorded for the operand whose dimension is one while the
  # other's is not. Returns a pair of lists of axes, or a pair of null values if either rank is
  # unknown or a dimension where both could be broadcast is unknown (decision is data dependent).
  # Pure function, no side effects.
  """Infers the sets of axes that might have been broadcasted."""
  x_rank = x_shape.rank
  y_rank = y_shape.rank
  if x_rank is None or y_rank is None:
    return None, None

  # Convert shapes for V1 compatibility, can be omitted in V2.
  x_shape = x_shape.as_list()
  y_shape = y_shape.as_list()

  b_rank = max(x_rank, y_rank)
  x_axes = []
  y_axes = []
  for axis in range(b_rank):
    x_dim = 1 if axis < b_rank - x_rank else x_shape[axis - (b_rank - x_rank)]
    y_dim = 1 if axis < b_rank - y_rank else y_shape[axis - (b_rank - y_rank)]
    if x_dim == 1 and y_dim != 1:
      # It's safe to assume that x_dim was broadcasted.
      x_axes.append(axis)
    elif y_dim == 1 and x_dim != 1:
      # It's safe to assume that y_dim was broadcasted.
      y_axes.append(axis)
    elif x_dim is None or y_dim is None:
      # Broadcasting decision is dynamic (data-dependent).
      return None, None

  return x_axes, y_axes


def _ReduceGradientArg(grad, shape_axes_must_reduce):
  # [FUNCTION DOC]
  # HELPER: reduces the gradient for one operand of a broadcasting binary op.
  # Input: the gradient (may be null) and a triple of shape, axes and must-reduce flag as produced by
  # the broadcast-args helper. If the gradient is present and a reduction is needed, sums it over the
  # axes keeping dimensions (which helps later optimizations when axes are unknown) and reshapes
  # to the operand's shape. Returns the possibly reduced gradient, or the input unchanged.
  """Reduces gradients of one of the arguments of a broadcasting binary op."""
  shape, axes, must_reduce = shape_axes_must_reduce
  if grad is not None and must_reduce:
    # Applying keepdims=True in presence of unknown axes opens up some
    # opportunities for optimizations. For example, _SumGrad below won't have to
    # emit extra ops to recover reduced indices for broadcasting.
    grad = math_ops.reduce_sum(grad, axes, keepdims=True)
    grad = array_ops.reshape(grad, shape)
  return grad


def _ReduceGradientArgs(x, y, gx, gy):
  # [FUNCTION DOC]
  # HELPER: reduces the gradients of both operands of a broadcasting binary op to operand shapes.
  # Inputs: the two operands x and y and their raw gradients gx and gy (either may be null). If at
  # least one gradient is present, computes the broadcast reduction info once and applies it to each
  # gradient. Returns the pair of gradients shaped like x and y respectively. Returns inputs untouched
  # if both gradients are null.
  """Reduces gradients of both arguments of a broadcasting binary op."""
  if gx is not None or gy is not None:
    bx, by = SmartBroadcastGradientArgs(x, y)
    gx = _ReduceGradientArg(gx, bx)
    gy = _ReduceGradientArg(gy, by)
  return gx, gy


_EMPTY_TUPLE = ()


def _IsScalar(x):
  # [FUNCTION DOC]
  # HELPER: small predicate for the fast paths of binary-op gradients.
  # Returns true if the tensor's static shape tuple is the shared empty tuple object, i.e. the tensor is
  # statically known to be a scalar. It relies on identity of the shape tuple rather than equality, so
  # it returns false for unknown shapes. Uses a protected tensor method; no side effects.
  return x._shape_tuple() is _EMPTY_TUPLE  # pylint: disable=protected-access


def _SafeShapeDiv(x, y):
  # [FUNCTION DOC]
  # HELPER: used by the Sum gradient to compute tile multiples.
  # Elementwise integer division of x by y for non-negative shapes, where a zero divisor is
  # replaced by one so that zero divided by zero yields zero. Inputs are shapes (tuples or tensors) of
  # non-negative integers; returns the floor-division result. No side effects.
  """Divides `x / y` assuming `x, y >= 0`, treating `0 / 0 = 0`."""
  return x // math_ops.maximum(y, 1)


@ops.RegisterGradient("Sum")
def _SumGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for the Sum reduction op. Inputs: the forward op (input tensor, reduction
  # axes, keep_dims attribute) and the gradient of the output. Returns a list: the gradient expanded
  # back to the input shape, and null for the axes.
  # Fast path when the input rank is known and the axes are constant: if all dimensions are
  # reduced, the gradient is reshaped to all ones (using a per-context cache of ones-shape
  # constants in eager mode) and tiled to the input shape (static or dynamic). In graph mode with a
  # fully known shape, the kept-dims shape and the tile multiples are computed once by constant
  # evaluation and cached in a per-graph cache, then used to reshape and tile.
  # General path: dynamic input shape; if keep_dims was false, reshape the gradient to the kept-dims
  # shape (computed colocated with the shape op), then broadcast to the input shape.
  # Side effects: reads and writes the per-graph reduced-shape cache and the eager context cache.
  """Gradient for Sum."""
  # Fast path for when reducing to a scalar and ndims is known: adds only
  # Reshape and Tile ops (and possibly a Shape).
  input_0_shape = op.inputs[0]._shape_tuple()  # pylint: disable=protected-access
  if input_0_shape is not None:
    axes = tensor_util.constant_value(op.inputs[1])
    if axes is not None:
      rank = len(input_0_shape)
      if np.array_equal(axes, np.arange(rank)):  # Reduce all dims.
        if context.executing_eagerly():
          ctx = context.context()
          new_shape = ctx.ones_rank_cache().get(rank)
          if new_shape is None:
            new_shape = constant_op.constant([1] * rank, dtype=dtypes.int32)
            ctx.ones_rank_cache().put(rank, new_shape)
        else:
          new_shape = [1] * rank
        grad = array_ops.reshape(grad, new_shape)
        # If shape is not fully defined (but rank is), we use Shape.
        if None not in input_0_shape:
          input_shape = constant_op.constant(input_0_shape, dtype=dtypes.int32)
        else:
          input_shape = array_ops.shape(op.inputs[0])
        return [array_ops.tile(grad, input_shape), None]
      elif None not in input_0_shape and not context.executing_eagerly():
        # The shape and reduction indices are statically known, so we use a
        # graph-level cache to avoid recomputing `reduced_shape()` for each
        # invocation.
        graph = ops.get_default_graph()

        # Canonicalize `axes` to be a tuple of indices. The incoming
        # value may be a scalar or a vector, and may include negative indices.
        axes = tuple(axes.reshape(-1))

        try:
          output_shape_kept_dims, tile_scaling = graph._reduced_shape_cache[  # pylint: disable=protected-access
              (input_0_shape, axes)]
        except KeyError:

          # Compute and cache `output_shape_kept_dims` and `tile_scaling`.
          def EvaluateAsTuple(t):
            # [FUNCTION DOC]
            # HELPER: nested in the Sum gradient to turn a shape-like value into a plain tuple.
            # Input t is either a tensor or an already concrete sequence. For tensors it tries to constant-fold
            # the value and asserts that folding succeeded (an assertion error is raised otherwise); other
            # values are used as-is. Returns the values as a tuple of numbers. No other side effects.
            if tensor_util.is_tf_type(t):
              value = tensor_util.try_evaluate_constant(t)
              assert value is not None
            else:
              value = t
            return tuple(value)

          output_shape_kept_dims = EvaluateAsTuple(
              math_ops.reduced_shape(input_0_shape, axes))
          tile_scaling = EvaluateAsTuple(
              _SafeShapeDiv(input_0_shape, output_shape_kept_dims))
          graph._reduced_shape_cache[(input_0_shape, axes)] = (  # pylint:disable=protected-access
              output_shape_kept_dims, tile_scaling)

        grad = array_ops.reshape(grad, output_shape_kept_dims)
        return [array_ops.tile(grad, tile_scaling), None]

  input_shape = array_ops.shape(op.inputs[0])

  if not op.get_attr("keep_dims"):
    with ops.colocate_with(input_shape):
      # TODO(apassos) remove this once device placement for eager ops makes
      # more sense.
      output_shape_kept_dims = math_ops.reduced_shape(input_shape,
                                                      op.inputs[1])
    grad = array_ops.reshape(grad, output_shape_kept_dims)
  return [array_ops.broadcast_to(grad, input_shape), None]


def _MinOrMaxGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # HELPER: shared implementation for the Min and Max gradients.
  # Inputs: forward op (input, axes, keep_dims) and output gradient. Reshapes the output and
  # gradient to the kept-dims shape if reductions dropped dimensions, marks which input elements equal
  # the reduced result, counts the selected elements per reduction group, and divides the gradient
  # evenly among tied elements. Returns a list: gradient shaped like the input and null for the axes.
  """Gradient for Min or Max. Amazingly it's precisely the same code."""
  input_shape = array_ops.shape(op.inputs[0])
  y = op.outputs[0]
  if not op.get_attr("keep_dims"):
    output_shape_kept_dims = math_ops.reduced_shape(input_shape, op.inputs[1])
    y = array_ops.reshape(y, output_shape_kept_dims)
    grad = array_ops.reshape(grad, output_shape_kept_dims)
  else:
    output_shape_kept_dims = array_ops.shape(y)

  # Compute the number of selected (maximum or minimum) elements in each
  # reduction dimension. If there are multiple minimum or maximum elements
  # then the gradient will be divided between them.
  indicators = math_ops.cast(math_ops.equal(y, op.inputs[0]), grad.dtype)
  num_selected = array_ops.reshape(
      math_ops.reduce_sum(indicators, op.inputs[1]), output_shape_kept_dims)

  return [math_ops.divide(indicators, num_selected) * grad, None]


@ops.RegisterGradient("Max")
def _MaxGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for the Max reduction op; simply delegates to the shared min-or-max gradient
  # helper with the same op and gradient and returns its result (input gradient, null for axes).
  """Gradient for Max."""
  return _MinOrMaxGrad(op, grad)


@ops.RegisterGradient("Min")
def _MinGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for the Min reduction op; delegates to the shared min-or-max gradient helper
  # and returns its result (input gradient, null for axes). No side effects.
  return _MinOrMaxGrad(op, grad)


@ops.RegisterGradient("Mean")
def _MeanGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for the Mean reduction op. Computes the Sum gradient and divides it by the
  # number of elements averaged per output element. If input and output shapes are fully static the
  # factor is the input size divided (integer division, minimum divisor one) by the output size and
  # embedded as a constant; otherwise it is the product of the input dimensions along the reduced
  # axes, with negative axes wrapped by the rank. Returns a pair: the scaled input gradient and a null
  # value for the axes.
  """Gradient for Mean."""
  sum_grad = _SumGrad(op, grad)[0]
  input_shape = op.inputs[0]._shape_tuple()  # pylint: disable=protected-access
  output_shape = op.outputs[0]._shape_tuple()  # pylint: disable=protected-access
  if (input_shape is not None and output_shape is not None and
      None not in input_shape and None not in output_shape):
    input_size = np.prod(input_shape)
    output_size = np.prod(output_shape)
    factor = input_size // max(output_size, 1)
    factor = constant_op.constant(factor, dtype=sum_grad.dtype)
  else:
    input_shape = array_ops.shape(op.inputs[0])
    input_rank = array_ops.size(input_shape)
    axes = math_ops.cast(op.inputs[1], input_rank.dtype)
    axes = (axes + input_rank) % input_rank
    factor = math_ops.reduce_prod(array_ops.gather(input_shape, axes))
  return math_ops.truediv(sum_grad, math_ops.cast(factor, sum_grad.dtype)), None


@ops.RegisterGradient("Prod")
def _ProdGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for the Prod reduction op. Handles zeros in the input by avoiding division:
  # the gradient of each element is the product of all other elements in its reduction group, computed
  # as the product of an exclusive forward cumulative product and an exclusive reverse cumulative
  # product after transposing the reduced axes to the front and flattening to two dimensions. Shape
  # related ops are pinned to the CPU. Complex values use conjugates. The result is multiplied by the
  # broadcast output gradient, transposed back and reshaped. Returns the input gradient and a null
  # value for the axes. Side effect: places some ops on the first CPU device.
  """Gradient for Prod."""
  # The gradient can be expressed by dividing the product by each entry of the
  # input tensor, but this approach can't deal with zeros in the input.
  # Here, we avoid this problem by composing the output as a product of two
  # cumprod operations.

  input_shape = array_ops.shape(op.inputs[0])
  # Reshape reduction indices for the case where the parameter is a scalar
  reduction_indices = array_ops.reshape(op.inputs[1], [-1])

  # Expand grad to full input shape
  if not op.get_attr("keep_dims"):
    output_shape_kept_dims = math_ops.reduced_shape(input_shape, op.inputs[1])
    grad = array_ops.reshape(grad, output_shape_kept_dims)

  grad = array_ops.broadcast_to(grad, input_shape)

  # Pack all reduced dimensions into a single one, so we can perform the
  # cumprod ops. If the reduction dims list is empty, it defaults to float32,
  # so we need to cast here.  We put all the shape-related ops on CPU to avoid
  # copying back and forth, and since listdiff is CPU only.
  with ops.device("/cpu:0"):
    rank = array_ops.rank(op.inputs[0])
    reduction_indices = math_ops.cast(reduction_indices, rank.dtype)
    reduced = (reduction_indices + rank) % rank
    idx = math_ops.range(0, rank)
    other, _ = gen_array_ops.list_diff(idx, reduced, reduced.dtype)
    perm = array_ops.concat([reduced, other], 0)
    reduced_num = math_ops.reduce_prod(array_ops.gather(input_shape, reduced))
    other_num = math_ops.reduce_prod(array_ops.gather(input_shape, other))
  permuted = array_ops.transpose(op.inputs[0], perm)
  permuted_shape = array_ops.shape(permuted)
  reshaped = array_ops.reshape(permuted, (reduced_num, other_num))

  # Calculate product, leaving out the current entry
  left = math_ops.cumprod(reshaped, axis=0, exclusive=True)
  right = math_ops.cumprod(reshaped, axis=0, exclusive=True, reverse=True)
  # For complex inputs, the gradient is in the conjugate direction.
  y = array_ops.reshape(
      math_ops.conj(left) * math_ops.conj(right), permuted_shape)

  # Invert the transpose and reshape operations.
  # Make sure to set the statically known shape information through a reshape.
  out = grad * array_ops.transpose(y, array_ops.invert_permutation(perm))
  return array_ops.reshape(out, input_shape), None


@ops.RegisterGradient("SegmentSum")
def _SegmentSumGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for SegmentSum. Gathers the output gradient by the segment ids so each input
  # row receives the gradient of its segment. Returns the pair of the data gradient and a null value
  # for segment ids.
  """Gradient for SegmentSum."""
  return array_ops.gather(grad, op.inputs[1]), None


@ops.RegisterGradient("SegmentMean")
def _SegmentMeanGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for SegmentMean. Builds a tensor of ones shaped like the data, sums it per
  # segment to get segment sizes, divides the output gradient by the sizes and gathers the result by
  # segment id. Returns the data gradient and a null value for segment ids.
  """Gradient for SegmentMean."""
  data_rank = array_ops.rank(op.inputs[0])
  segment_ids_shape = array_ops.shape(op.inputs[1])
  remaining_shape = array_ops.ones(
      array_ops.expand_dims(data_rank - 1, 0), dtype=segment_ids_shape.dtype
  )
  ones_shape = array_ops.concat([segment_ids_shape, remaining_shape], 0)
  ones = array_ops.ones(ones_shape, dtype=grad.dtype)
  scaled_grad = math_ops.divide(grad, math_ops.segment_sum(ones, op.inputs[1]))
  return array_ops.gather(scaled_grad, op.inputs[1]), None


def _SparseSegmentReduceGradV2(op, grad, norm=None):
  # [FUNCTION DOC]
  # HELPER: shared by the sparse segment sum/mean/sqrt-n gradients when the sparse gradient attribute is set.
  # Inputs: the forward op (data, indices, segment ids), the output gradient and norm, which is null
  # for sum, or the text "mean" or "sqrtn". Asserts that norm has one of those values. Picks the matching
  # version-2 sparse segment gradient kernel, calls it with the dense first dimension of the data,
  # and returns a sparse indexed-slices gradient made of the gradient values, the sorted unique
  # indices and the dense shape of the data.
  """Sparse gradient for SparseSegment(Sum|Mean|SqrtN)[WithNumSegments]."""
  assert norm is None or norm == "mean" or norm == "sqrtn"
  indices = op.inputs[1]
  segment_ids = op.inputs[2]
  data_shape = array_ops.shape(op.inputs[0])
  dense_output_dim0 = data_shape[0]
  if norm == "mean":
    grad_fn = math_ops.sparse_segment_mean_grad_v2
  elif norm == "sqrtn":
    grad_fn = math_ops.sparse_segment_sqrt_n_grad_v2
  else:
    grad_fn = math_ops.sparse_segment_sum_grad_v2
  grad_values, sorted_unique_indices = grad_fn(
      grad, indices, segment_ids, dense_output_dim0
  )
  return indexed_slices_lib.IndexedSlices(
      grad_values, sorted_unique_indices, data_shape
  )


def _GetOpAttrOrNone(op, name):
  # [FUNCTION DOC]
  # HELPER: safe attribute lookup for gradient functions.
  # Returns the attribute of the op with the given name, or a null value if the op has no such
  # attribute (the lookup error is swallowed). No other side effects.
  """Returns the value of the attr of `op` with the given `name`, or None if no

  such attr exists.
  """
  try:
    return op.get_attr(name)
  except ValueError:
    return None


@ops.RegisterGradient("SparseSegmentSum")
def _SparseSegmentSumGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the forward-compatibility check against a 2021 date is always true today, so the
  # fallback branch (gather plus unsorted segment sum) is effectively dead code.
  # Gradient registered for SparseSegmentSum. If the op's sparse-gradient attribute is set, returns a
  # sparse indexed-slices gradient from the shared v2 helper. Otherwise uses the dense sparse-segment
  # sum gradient kernel with the first dimension of the data as output rows. Returns a triple: data
  # gradient, then null values for indices and segment ids.
  """Gradient for SparseSegmentSum."""
  if _GetOpAttrOrNone(op, "sparse_gradient"):
    return _SparseSegmentReduceGradV2(op, grad), None, None
  dim0 = array_ops.shape(op.inputs[0])[0]
  if compat.forward_compatible(2021, 6, 10):
    return (math_ops.sparse_segment_sum_grad(grad, op.inputs[1], op.inputs[2],
                                             dim0), None, None)
  else:
    return (math_ops.unsorted_segment_sum(
        array_ops.gather(grad, op.inputs[2]), op.inputs[1], dim0), None, None)


@ops.RegisterGradient("SparseSegmentSumWithNumSegments")
def _SparseSegmentSumWithNumSegmentsGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the forward-compatibility check against a 2021 date is always true today, so the
  # fallback branch (gather plus unsorted segment sum) is effectively dead code.
  # Gradient registered for SparseSegmentSumWithNumSegments. Same as the plain sparse segment sum
  # gradient but returns four values: the data gradient (sparse or dense depending on the
  # sparse-gradient attribute) and null values for indices, segment ids and number of segments.
  """Gradient for SparseSegmentSumWithNumSegments."""
  if _GetOpAttrOrNone(op, "sparse_gradient"):
    return _SparseSegmentReduceGradV2(op, grad), None, None, None
  dim0 = array_ops.shape(op.inputs[0])[0]
  if compat.forward_compatible(2021, 6, 10):
    return (math_ops.sparse_segment_sum_grad(grad, op.inputs[1], op.inputs[2],
                                             dim0), None, None, None)
  else:
    return (math_ops.unsorted_segment_sum(
        array_ops.gather(grad, op.inputs[2]), op.inputs[1],
        dim0), None, None, None)


@ops.RegisterGradient("SparseSegmentMean")
def _SparseSegmentMeanGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for SparseSegmentMean. If the sparse-gradient attribute is set returns an
  # indexed-slices gradient using the mean normalization; otherwise uses the dense sparse-segment
  # mean gradient kernel. Returns data gradient plus null values for indices and segment ids.
  """Gradient for SparseSegmentMean."""
  if _GetOpAttrOrNone(op, "sparse_gradient"):
    return _SparseSegmentReduceGradV2(op, grad, "mean"), None, None
  dim0 = array_ops.shape(op.inputs[0])[0]
  return (math_ops.sparse_segment_mean_grad(grad, op.inputs[1], op.inputs[2],
                                            dim0), None, None)


@ops.RegisterGradient("SparseSegmentMeanWithNumSegments")
def _SparseSegmentMeanWithNumSegmentsGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for SparseSegmentMeanWithNumSegments. Same as the mean variant but returns
  # four values: data gradient and null values for indices, segment ids and number of segments.
  """Gradient for SparseSegmentMeanWithNumSegments."""
  if _GetOpAttrOrNone(op, "sparse_gradient"):
    return _SparseSegmentReduceGradV2(op, grad, "mean"), None, None, None
  dim0 = array_ops.shape(op.inputs[0])[0]
  return (math_ops.sparse_segment_mean_grad(grad, op.inputs[1], op.inputs[2],
                                            dim0), None, None, None)


@ops.RegisterGradient("SparseSegmentSqrtN")
def _SparseSegmentSqrtNGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for SparseSegmentSqrtN. If the sparse-gradient attribute is set returns an
  # indexed-slices gradient using square-root-of-count normalization; otherwise uses the dense kernel.
  # Returns data gradient plus null values for indices and segment ids.
  """Gradient for SparseSegmentSqrtN."""
  if _GetOpAttrOrNone(op, "sparse_gradient"):
    return _SparseSegmentReduceGradV2(op, grad, "sqrtn"), None, None
  dim0 = array_ops.shape(op.inputs[0])[0]
  return (math_ops.sparse_segment_sqrt_n_grad(grad, op.inputs[1], op.inputs[2],
                                              dim0), None, None)


@ops.RegisterGradient("SparseSegmentSqrtNWithNumSegments")
def _SparseSegmentSqrtNWithNumSegmentsGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for SparseSegmentSqrtNWithNumSegments. Same as the sqrt-n variant but returns
  # four values: data gradient and null values for indices, segment ids and number of segments.
  """Gradient for SparseSegmentSqrtNWithNumSegments."""
  if _GetOpAttrOrNone(op, "sparse_gradient"):
    return _SparseSegmentReduceGradV2(op, grad, "sqrtn"), None, None, None
  dim0 = array_ops.shape(op.inputs[0])[0]
  return (math_ops.sparse_segment_sqrt_n_grad(grad, op.inputs[1], op.inputs[2],
                                              dim0), None, None, None)


def _SegmentMinOrMaxGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # HELPER: shared by the SegmentMin and SegmentMax gradients.
  # Marks which input elements equal their segment's result, counts them per segment, divides the
  # segment gradient evenly among the ties, gathers back by segment id and keeps the value only at
  # selected positions (zeros elsewhere). Returns the data gradient and a null value for segment ids.
  """ Gradient for SegmentMin and SegmentMax. """
  zeros = array_ops.zeros_like(op.inputs[0], dtype=op.inputs[0].dtype)
  # Get the number of selected (minimum or maximum) elements in each segment.
  gathered_outputs = array_ops.gather(op.outputs[0], op.inputs[1])
  is_selected = math_ops.equal(op.inputs[0], gathered_outputs)
  num_selected = math_ops.segment_sum(
      math_ops.cast(is_selected, grad.dtype), op.inputs[1])
  # Compute the gradient for each segment. The gradient for the ith segment is
  # divided evenly among the selected elements in that segment.
  weighted_grads = math_ops.divide(grad, num_selected)
  gathered_grads = array_ops.gather(weighted_grads, op.inputs[1])
  return array_ops.where_v2(is_selected, gathered_grads, zeros), None


@ops.RegisterGradient("SegmentMin")
def _SegmentMinGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for SegmentMin; delegates to the shared segment min-or-max helper and returns
  # its result (data gradient, null for segment ids).
  """Gradient for SegmentMin."""
  return _SegmentMinOrMaxGrad(op, grad)


@ops.RegisterGradient("SegmentMax")
def _SegmentMaxGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for SegmentMax; delegates to the shared segment min-or-max helper and returns
  # its result (data gradient, null for segment ids).
  """Gradient for SegmentMax."""
  return _SegmentMinOrMaxGrad(op, grad)


# pylint: disable=g-doc-args
@ops.RegisterGradient("SegmentProd")
def _SegmentProdGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for SegmentProd. Handles zeros without dividing by them: segments with two or
  # more zeros get zero gradient; segments with exactly one zero give the product of the other
  # entries to the zero element and zero to the rest; segments with no zeros use the segment product
  # divided by the element. Implemented by replacing zeros with ones for a second segment product,
  # gathering both products by segment id and selecting between them elementwise. Returns the data
  # gradient and a null value for segment ids.
  """Gradient for SegmentProd.

  The gradient can be expressed for each segment by dividing the segment's
  product by each element of the segment input tensor, but this approach can't
  deal with zeros in the input.
  Unlike reduce_prod we can't use cumsum here as individual segments may have
  a different number of elements. Therefore we consider three cases:
  1) A segment input contains no zeros and we can safely divide by the input
     tensor.
  2) A segment contains exactly one zero. Then the gradient of each input of
     the segment is zero except for the 0-input, there the gradient is
     the product of the remaining segment entries.
  3) A segment contains at least two zeros. The gradient is zero for all
     segment inputs.
  """
  data = op.inputs[0]
  segment_ids = op.inputs[1]
  is_zero = math_ops.equal(data, 0)
  num_zeros = gen_math_ops.segment_sum(
      math_ops.cast(is_zero, dtype=dtypes.int32), segment_ids)
  # handle case 3 and set the gradient to 0 for segments with more than one
  # 0 as input
  grad = array_ops.where_v2(
      math_ops.greater(num_zeros, 1), array_ops.zeros_like(grad), grad)
  # replace all zeros with ones and compute the segment_prod
  non_zero_data = array_ops.where_v2(is_zero, array_ops.ones_like(data), data)
  non_zero_prod = gen_math_ops.segment_prod(non_zero_data, segment_ids)
  gathered_prod = array_ops.gather(op.outputs[0], segment_ids)
  gathered_non_zero_prod = array_ops.gather(non_zero_prod, segment_ids)
  prod_divided_by_el = gathered_prod / non_zero_data
  # Now fetch the individual results for segments containing 0 and those that
  # don't.
  partial_derivative = array_ops.where_v2(is_zero, gathered_non_zero_prod,
                                          prod_divided_by_el)
  gathered_grad = array_ops.gather(grad, segment_ids)
  return gathered_grad * partial_derivative, None


def _GatherDropNegatives(params,
                         ids,
                         zero_clipped_indices=None,
                         is_positive=None):
  # [FUNCTION DOC]
  # HELPER: shared by the unsorted segment gradients.
  # Gathers rows of params by ids, returning zeros for rows whose id is negative. Inputs: params,
  # ids, and optionally precomputed clipped indices and a positivity mask (to reuse across calls; when
  # the mask is omitted it is computed and broadcast to the shape of the gathered result). If ids is
  # null, both reusable values must be supplied. Returns a triple: gathered values with negatives
  # zeroed, the indices clipped at zero, and the boolean mask of non-negative ids.
  """ Helper function for unsorted segment ops.

  Gathers params for
      positive segment ids and gathers 0 for inputs with negative segment id.
      Also returns the clipped indices and a boolean mask with the same shape
      as ids where a positive id is masked as true. With this, the latter two
      can be passed as arguments to this function to reuse them.
  """
  if zero_clipped_indices is None:
    zero_clipped_indices = math_ops.maximum(ids, array_ops.zeros_like(ids))
  gathered = array_ops.gather(params, zero_clipped_indices)
  if is_positive is None:
    is_positive = math_ops.greater_equal(ids, 0)
    # tf.where(condition, x, y) requires condition to have the same shape as x
    # and y.
    is_positive_shape = array_ops.shape(is_positive)
    broadcastable_shape = array_ops.concat(
        [
            is_positive_shape,
            array_ops.ones(
                [array_ops.rank(gathered) - array_ops.rank(is_positive)],
                dtype=is_positive_shape.dtype,
            ),
        ],
        axis=0,
    )
    is_positive = array_ops.reshape(is_positive, broadcastable_shape)
    is_positive = is_positive & array_ops.ones_like(gathered, dtype=dtypes.bool)
  # replace gathered params of negative indices with 0
  zero_slice = array_ops.zeros_like(gathered)
  return (
      array_ops.where_v2(is_positive, gathered, zero_slice),
      zero_clipped_indices,
      is_positive,
  )


def _UnsortedSegmentMinOrMaxGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # HELPER: shared by the UnsortedSegmentMin and UnsortedSegmentMax gradients.
  # Gathers the segment outputs for each element (zero for negative ids), marks elements equal to their
  # segment result and with a non-negative id, counts them per segment, divides the gradient among
  # ties, gathers back and keeps values only at selected positions. Returns a triple: data
  # gradient and null values for segment ids and number of segments.
  """Gradient for UnsortedSegmentMin and UnsortedSegmentMax."""
  # Get the number of selected (minimum or maximum) elements in each segment.
  gathered_outputs, zero_clipped_indices, is_positive = _GatherDropNegatives(
      op.outputs[0], op.inputs[1]
  )
  is_selected = math_ops.equal(op.inputs[0], gathered_outputs)
  is_selected = math_ops.logical_and(is_selected, is_positive)
  num_selected = math_ops.unsorted_segment_sum(
      math_ops.cast(is_selected, grad.dtype), op.inputs[1], op.inputs[2]
  )
  # Compute the gradient for each segment. The gradient for the ith segment is
  # divided evenly among the selected elements in that segment.
  weighted_grads = math_ops.divide(grad, num_selected)
  gathered_grads, _, _ = _GatherDropNegatives(
      weighted_grads, None, zero_clipped_indices, is_positive
  )
  zeros = array_ops.zeros_like(gathered_grads)
  return array_ops.where_v2(is_selected, gathered_grads, zeros), None, None


@ops.RegisterGradient("UnsortedSegmentSum")
def _UnsortedSegmentSumGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for UnsortedSegmentSum. Gathers the output gradient by segment id, producing
  # zeros for negative ids. Returns a triple: data gradient, null for segment ids, null for number
  # of segments.
  """Gradient for UnsortedSegmentSum."""
  return _GatherDropNegatives(grad, op.inputs[1])[0], None, None


@ops.RegisterGradient("UnsortedSegmentMax")
def _UnsortedSegmentMaxGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for UnsortedSegmentMax; delegates to the shared unsorted min-or-max helper
  # and returns its triple (data gradient, null, null).
  """ Gradient for UnsortedSegmentMax. """
  return _UnsortedSegmentMinOrMaxGrad(op, grad)


@ops.RegisterGradient("UnsortedSegmentMin")
def _UnsortedSegmentMinGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for UnsortedSegmentMin; delegates to the shared unsorted min-or-max helper
  # and returns its triple (data gradient, null, null).
  """ Gradient for UnsortedSegmentMin. """
  return _UnsortedSegmentMinOrMaxGrad(op, grad)


@ops.RegisterGradient("UnsortedSegmentProd")
def _UnsortedSegmentProdGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for UnsortedSegmentProd. Same strategy as the SegmentProd gradient: segments
  # with two or more zeros get zero gradient, a segment with exactly one zero gives the product of
  # the other entries to that element, others use product divided by element. Segment ids are clipped
  # at zero for gathering and negative ids are masked out when gathering the gradient. The division by
  # the element may produce not-a-number or infinity values, but those positions are overridden by the
  # selection. Returns the data gradient and null values for segment ids and number of segments.
  """ Gradient for UnsortedSegmentProd.

  The gradient can be expressed for each segment by dividing the segment's
  product by each element of the segment input tensor, but this approach can't
  deal with zeros in the input.
  Unlike reduce_prod we can't use cumsum here as individual segments may have
  a different number of elements. Therefore we consider three cases:
  1) A segment input contains no zeros and we can safely divide by the input
     tensor.
  2) A segment contains exactly one zero. Then the gradient of each input of
     the segment is zero except for the 0-input, there the gradient is
     the product of the remaining segment entries.
  3) A segment contains at least two zeros. The gradient is zero for all
     segment inputs.
  """
  # Note that unsorted_segment_sum will filter out the negative indices,
  # so we don't need to do a logical_and with is_positive here
  is_zero = math_ops.equal(op.inputs[0], 0)
  num_zeros = gen_math_ops.unsorted_segment_sum(
      math_ops.cast(is_zero, dtype=dtypes.int32), op.inputs[1], op.inputs[2])
  # handle case 3 and set the gradient to 0 for segments with more than one
  # 0 as input
  grad = array_ops.where_v2(
      math_ops.greater(num_zeros, 1), array_ops.zeros_like(grad), grad)
  # replace all zeros with ones and compute the unsorted_segment_prod
  non_zero_data = array_ops.where_v2(is_zero, array_ops.ones_like(op.inputs[0]),
                                     op.inputs[0])
  non_zero_prod = gen_math_ops.unsorted_segment_prod(non_zero_data,
                                                     op.inputs[1], op.inputs[2])
  # clip the indices for gather to be positive
  zero_clipped_indices = math_ops.maximum(op.inputs[1],
                                          array_ops.zeros_like(op.inputs[1]))
  gathered_prod = array_ops.gather(op.outputs[0], zero_clipped_indices)
  gathered_non_zero_prod = array_ops.gather(non_zero_prod, zero_clipped_indices)
  prod_divided_by_el = gathered_prod / op.inputs[0]  # May contain nan/inf.
  # Now fetch the individual results for segments containing 0 and those that
  # don't. is_zero will also fetch results for entries with negative index
  # but the following gather_drop_negatives sets the corresponding entry in
  # grad to 0 for these
  partial_derivative = array_ops.where_v2(is_zero, gathered_non_zero_prod,
                                          prod_divided_by_el)
  gathered_grad = _GatherDropNegatives(grad, op.inputs[1],
                                       zero_clipped_indices)[0]
  return gathered_grad * partial_derivative, None, None


@ops.RegisterGradient("Abs")
def _AbsGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Abs: the incoming gradient multiplied by the sign of the input. Returns a
  # single tensor. No side effects.
  x = op.inputs[0]
  return grad * math_ops.sign(x)


@ops.RegisterGradient("Neg")
def _NegGrad(_, grad):
  # [FUNCTION DOC]
  # Gradient registered for Neg: returns the negated incoming gradient. The op argument is ignored.
  """Returns -grad."""
  return -grad


@ops.RegisterGradient("Inv")
def _InvGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Inv (legacy name of reciprocal). Calls the reciprocal-grad kernel with
  # the forward output and incoming gradient, giving the negated gradient times the square of the
  # output. Returns a single tensor.
  """Returns -grad * (1 / x^2)."""
  y = op.outputs[0]  # y = 1 / x
  return gen_math_ops.reciprocal_grad(y, grad)


@ops.RegisterGradient("Reciprocal")
def _ReciprocalGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Reciprocal. Uses the reciprocal-grad kernel with the forward output and
  # the incoming gradient (negative gradient times output squared). Returns a single tensor.
  """Returns -grad * (1 / x^2)."""
  y = op.outputs[0]  # y = 1 / x
  return gen_math_ops.reciprocal_grad(y, grad)


@ops.RegisterGradient("InvGrad")
def _InvGradGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Second-order gradient registered for InvGrad. Inputs: op with the forward inputs (output y, its
  # gradient) and the incoming gradient. Under a control dependency on the gradient, uses conjugates
  # and returns a pair: the gradient for the first input (minus two times conjugated gradient times
  # second input times conjugated first input) and the reciprocal-grad of the conjugated first
  # input and the gradient for the second input.
  b = op.inputs[1]
  # op.output[0]: y = -b * conj(a)^2
  with ops.control_dependencies([grad]):
    ca = math_ops.conj(op.inputs[0])
    cg = math_ops.conj(grad)
    return cg * -2.0 * b * ca, gen_math_ops.reciprocal_grad(ca, grad)


@ops.RegisterGradient("ReciprocalGrad")
def _ReciprocalGradGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Second-order gradient registered for ReciprocalGrad; identical computation to the one for InvGrad.
  # Under a control dependency on the gradient, returns a pair: gradient for the first input (minus two
  # times conjugated gradient times second input times conjugated first input) and the reciprocal-grad
  # of the conjugated first input and the gradient for the second input.
  b = op.inputs[1]
  # op.output[0]: y = -b * conj(a)^2
  with ops.control_dependencies([grad]):
    ca = math_ops.conj(op.inputs[0])
    cg = math_ops.conj(grad)
    return cg * -2.0 * b * ca, gen_math_ops.reciprocal_grad(ca, grad)


@ops.RegisterGradient("Square")
def _SquareGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Square: gradient times two times the conjugated input. A control
  # dependency on the gradient delays computing the factor until the gradient is available. Returns a
  # single tensor.
  x = op.inputs[0]
  # Added control dependencies to prevent 2*x from being computed too early.
  with ops.control_dependencies([grad]):
    x = math_ops.conj(x)
    y = constant_op.constant(2.0, dtype=x.dtype)
    return math_ops.multiply(grad, math_ops.multiply(x, y))


@ops.RegisterGradient("Sqrt")
def _SqrtGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Sqrt: uses the sqrt-grad kernel with the forward output and incoming
  # gradient (half the gradient divided by the output). Returns a single tensor.
  y = op.outputs[0]  # y = x^(1/2)
  return gen_math_ops.sqrt_grad(y, grad)


@ops.RegisterGradient("SqrtGrad")
def _SqrtGradGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Second-order gradient registered for SqrtGrad. Inputs: op (first input a, second input, output y)
  # and incoming gradient. Under a control dependency, computes the gradient divided by a and returns
  # a pair: the negated conjugate of that quotient times the output y, and half the quotient.
  a = op.inputs[0]
  y = op.outputs[0]  # y = 0.5 * b / conj(a)
  with ops.control_dependencies([grad]):
    ga = grad / a
    return -math_ops.conj(ga) * y, 0.5 * ga  # pylint: disable=invalid-unary-operand-type


@ops.RegisterGradient("Rsqrt")
def _RsqrtGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Rsqrt: uses the rsqrt-grad kernel with the forward output and incoming
  # gradient (minus half the gradient times the cube of the conjugated output). Returns a single tensor.
  """Returns -0.5 * grad * conj(y)^3."""
  y = op.outputs[0]  # y = x^(-1/2)
  return gen_math_ops.rsqrt_grad(y, grad)


@ops.RegisterGradient("RsqrtGrad")
def _RsqrtGradGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Second-order gradient registered for RsqrtGrad. Returns a pair for the two inputs a and b: for a,
  # minus one and a half times conjugate of the gradient times b times square of conjugated a; for b,
  # the rsqrt-grad of the conjugated a and the gradient. Runs under a control dependency on the gradient.
  """Returns backprop gradient for f(a,b) = -0.5 * b * conj(a)^3."""
  a = op.inputs[0]  # a = x^{-1/2}
  b = op.inputs[1]  # backprop gradient for a
  with ops.control_dependencies([grad]):
    ca = math_ops.conj(a)
    cg = math_ops.conj(grad)
    grad_a = -1.5 * cg * b * math_ops.square(ca)
    grad_b = gen_math_ops.rsqrt_grad(ca, grad)
    return grad_a, grad_b


@ops.RegisterGradient("Exp")
def _ExpGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Exp: gradient times the conjugated forward output. Evaluated under a
  # control dependency on the gradient. Returns a single tensor.
  """Returns grad * exp(x)."""
  y = op.outputs[0]  # y = e^x
  with ops.control_dependencies([grad]):
    y = math_ops.conj(y)
    return grad * y


@ops.RegisterGradient("Expm1")
def _Expm1Grad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Expm1: gradient times the exponential of the conjugated input. Evaluated
  # under a control dependency on the gradient. Returns a single tensor.
  """Returns grad * exp(x)."""
  x = op.inputs[0]
  with ops.control_dependencies([grad]):
    x = math_ops.conj(x)
    y = math_ops.exp(x)
    return grad * y


@ops.RegisterGradient("Log")
def _LogGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Log: gradient times the reciprocal of the conjugated input. Evaluated
  # under a control dependency on the gradient. Returns a single tensor.
  """Returns grad * (1/x)."""
  x = op.inputs[0]
  with ops.control_dependencies([grad]):
    x = math_ops.conj(x)
    return grad * math_ops.reciprocal(x)


@ops.RegisterGradient("Log1p")
def _Log1pGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Log1p: gradient times the reciprocal of one plus the conjugated input.
  # Evaluated under a control dependency on the gradient. Returns a single tensor.
  """Returns grad * (1/(1 + x))."""
  x = op.inputs[0]
  with ops.control_dependencies([grad]):
    x = math_ops.conj(x)
    return grad * math_ops.reciprocal(1 + x)


@ops.RegisterGradient("Xlogy")
def _XLogyGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Xlogy (x times log y, defined as zero where x is zero). Computes the
  # partial derivative with respect to x as xlogy of an indicator of nonzero x and y, and with respect
  # to y as x divided by y (safe division). Each is multiplied by the gradient, summed over the
  # broadcast axes found by the dynamic broadcast-gradient-args op and reshaped to the operand shape.
  # Returns a pair of gradients for x and y.
  """Returns gradient of xlogy(x, y) with respect to x and y."""
  x = op.inputs[0]
  y = op.inputs[1]
  sx = array_ops.shape(x)
  sy = array_ops.shape(y)
  rx, ry = gen_array_ops.broadcast_gradient_args(sx, sy)
  with ops.control_dependencies([grad]):
    not_zero_x = math_ops.cast(
        math_ops.not_equal(x, math_ops.cast(0., dtype=x.dtype)), dtype=x.dtype)
    partial_x = gen_math_ops.xlogy(not_zero_x, y)
    partial_y = gen_math_ops.xdivy(x, y)
    return (array_ops.reshape(math_ops.reduce_sum(partial_x * grad, rx), sx),
            array_ops.reshape(math_ops.reduce_sum(partial_y * grad, ry), sy))


@ops.RegisterGradient("Xlog1py")
def _XLog1pyGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Xlog1py (x times log of one plus y, zero where x is zero). Partial for x
  # is xlog1py of a nonzero-x indicator and y; partial for y is x divided by one plus y (safe
  # division). Each is multiplied by the gradient, summed over broadcast axes and reshaped. Returns a
  # pair of gradients for x and y.
  """Returns gradient of xlog1py(x, y) with respect to x and y."""
  x = op.inputs[0]
  y = op.inputs[1]
  sx = array_ops.shape(x)
  sy = array_ops.shape(y)
  rx, ry = gen_array_ops.broadcast_gradient_args(sx, sy)
  with ops.control_dependencies([grad]):
    not_zero_x = math_ops.cast(
        math_ops.not_equal(x, math_ops.cast(0., dtype=x.dtype)), dtype=x.dtype)
    partial_x = gen_math_ops.xlog1py(not_zero_x, y)
    partial_y = gen_math_ops.xdivy(x, y + 1.)
    return (array_ops.reshape(math_ops.reduce_sum(partial_x * grad, rx), sx),
            array_ops.reshape(math_ops.reduce_sum(partial_y * grad, ry), sy))


@ops.RegisterGradient("Xdivy")
def _XDivyGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Xdivy (x divided by y, zero where x is zero). Partial for x is xdivy of a
  # nonzero-x indicator and y; partial for y is negative x divided by y squared (safe division). Each
  # is multiplied by the gradient, summed over broadcast axes and reshaped. Returns a pair for x and y.
  """Returns gradient of xdivy(x, y) with respect to x and y."""
  x = op.inputs[0]
  y = op.inputs[1]
  sx = array_ops.shape(x)
  sy = array_ops.shape(y)
  rx, ry = gen_array_ops.broadcast_gradient_args(sx, sy)
  with ops.control_dependencies([grad]):
    not_zero_x = math_ops.cast(
        math_ops.not_equal(x, math_ops.cast(0., dtype=x.dtype)), dtype=x.dtype)
    partial_x = gen_math_ops.xdivy(not_zero_x, y)
    partial_y = gen_math_ops.xdivy(math_ops.negative(x), y**2)
    return (array_ops.reshape(math_ops.reduce_sum(partial_x * grad, rx), sx),
            array_ops.reshape(math_ops.reduce_sum(partial_y * grad, ry), sy))


@ops.RegisterGradient("Sinh")
def _SinhGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Sinh: gradient times the hyperbolic cosine of the conjugated input, under a
  # control dependency on the gradient. Returns a single tensor.
  """Returns grad * cosh(x)."""
  x = op.inputs[0]
  with ops.control_dependencies([grad]):
    x = math_ops.conj(x)
    return grad * math_ops.cosh(x)


@ops.RegisterGradient("Cosh")
def _CoshGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Cosh: gradient times the hyperbolic sine of the conjugated input, under a
  # control dependency on the gradient. Returns a single tensor.
  """Returns grad * sinh(x)."""
  x = op.inputs[0]
  with ops.control_dependencies([grad]):
    x = math_ops.conj(x)
    return grad * math_ops.sinh(x)


@ops.RegisterGradient("Tanh")
def _TanhGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Tanh: uses the tanh-grad kernel with the conjugated forward output and the
  # incoming gradient. Runs under a control dependency on the gradient. Returns a single tensor.
  """Returns grad * (1 - tanh(x) * tanh(x))."""
  y = op.outputs[0]  # y = tanh(x)
  with ops.control_dependencies([grad]):
    y = math_ops.conj(y)
    return gen_math_ops.tanh_grad(y, grad)


@ops.RegisterGradient("Asinh")
def _AsinhGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Asinh: gradient divided by the hyperbolic cosine of the conjugated forward
  # output. Under a control dependency on the gradient. Returns a single tensor.
  """Returns grad * 1/cosh(y)."""
  y = op.outputs[0]
  with ops.control_dependencies([grad]):
    y = math_ops.conj(y)
    return grad / math_ops.cosh(y)


@ops.RegisterGradient("Acosh")
def _AcoshGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Acosh: gradient divided by the hyperbolic sine of the conjugated forward
  # output. Under a control dependency on the gradient. Returns a single tensor.
  """Returns grad * 1/sinh(y)."""
  y = op.outputs[0]
  with ops.control_dependencies([grad]):
    y = math_ops.conj(y)
    return grad / math_ops.sinh(y)


@ops.RegisterGradient("Atanh")
def _AtanhGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Atanh: gradient times the reciprocal of one minus the square of the
  # conjugated input. Under a control dependency on the gradient. Returns a single tensor.
  """Returns grad * 1/ (1 - x^2)."""
  x = op.inputs[0]
  with ops.control_dependencies([grad]):
    x = math_ops.conj(x)
    x2 = math_ops.square(x)
    one = constant_op.constant(1, dtype=grad.dtype)
    inv = math_ops.reciprocal(math_ops.subtract(one, x2))
    return grad * inv


@ops.RegisterGradient("TanhGrad")
def _TanhGradGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Second-order gradient registered for TanhGrad. Under a control dependency on the gradient and with
  # both inputs conjugated, returns a pair: the gradient for the first input (gradient times minus two
  # times the second input times the first input) and the tanh-grad of the first input and gradient for
  # the second input.
  with ops.control_dependencies([grad]):
    a = math_ops.conj(op.inputs[0])
    b = math_ops.conj(op.inputs[1])
    return grad * -2.0 * b * a, gen_math_ops.tanh_grad(a, grad)


@ops.RegisterGradient("Erf")
def _ErfGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Erf: gradient times two over the square root of pi times the exponential
  # of minus the square of the conjugated input. The constant is created in the gradient's dtype. Under
  # a control dependency on the gradient. Returns a single tensor.
  """Returns grad * 2/sqrt(pi) * exp(-x**2)."""
  x = op.inputs[0]
  two_over_root_pi = constant_op.constant(2 / np.sqrt(np.pi), dtype=grad.dtype)
  with ops.control_dependencies([grad]):
    x = math_ops.conj(x)
    return grad * two_over_root_pi * math_ops.exp(-math_ops.square(x))


@ops.RegisterGradient("Erfc")
def _ErfcGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Erfc: like the Erf gradient but with the opposite sign (minus two over the
  # square root of pi). Under a control dependency on the gradient. Returns a single tensor.
  """Returns -grad * 2/sqrt(pi) * exp(-x**2)."""
  x = op.inputs[0]
  minus_two_over_root_pi = constant_op.constant(
      -2 / np.sqrt(np.pi), dtype=grad.dtype)
  with ops.control_dependencies([grad]):
    x = math_ops.conj(x)
    return grad * minus_two_over_root_pi * math_ops.exp(-math_ops.square(x))


@ops.RegisterGradient("Erfinv")
def _ErfinvGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Erfinv: gradient times half the square root of pi times the exponential
  # of the square of the forward output. Under a control dependency on the gradient. Returns a single
  # tensor.
  """Returns grad * sqrt(pi) / 2 * exp(erfinv(x)**2)."""
  root_pi_over_two = constant_op.constant(np.sqrt(np.pi) / 2, dtype=grad.dtype)
  with ops.control_dependencies([grad]):
    return grad * root_pi_over_two * math_ops.exp(
        math_ops.square(op.outputs[0]))


@ops.RegisterGradient("Ndtri")
def _NdtriGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Ndtri (inverse normal CDF): gradient times the square root of two pi times
  # the exponential of half the square of the forward output. Under a control dependency on the
  # gradient. Returns a single tensor.
  """Returns grad * sqrt(2 * pi) * exp(ndtri(x)**2 / 2)."""
  root_two_pi = constant_op.constant(np.sqrt(2 * np.pi), dtype=grad.dtype)
  with ops.control_dependencies([grad]):
    return grad * root_two_pi * math_ops.exp(
        math_ops.square(op.outputs[0]) / 2.)


@ops.RegisterGradient("Lgamma")
def _LgammaGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Lgamma: gradient times the digamma of the conjugated input, under a control
  # dependency on the gradient. Returns a single tensor.
  """Returns grad * digamma(x)."""
  x = op.inputs[0]
  with ops.control_dependencies([grad]):
    x = math_ops.conj(x)
    return grad * math_ops.digamma(x)


@ops.RegisterGradient("Digamma")
def _DigammaGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Digamma: gradient times the trigamma (order-one polygamma) of the
  # conjugated input, under a control dependency on the gradient. Returns a single tensor.
  """Compute gradient of the digamma function with respect to its argument."""
  x = op.inputs[0]
  with ops.control_dependencies([grad]):
    x = math_ops.conj(x)
    partial_x = math_ops.polygamma(array_ops.constant(1, dtype=x.dtype), x)
    return grad * partial_x


@ops.RegisterGradient("Dawsn")
def _DawsnGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Dawsn (Dawson integral): gradient times one minus two times the input times
  # the forward output. Under a control dependency on the gradient. Returns a single tensor.
  """Compute gradient of dawsn(x) with respect to its argument."""
  x = op.inputs[0]
  y = op.outputs[0]
  with ops.control_dependencies([grad]):
    return grad * (1. - 2 * x * y)


@ops.RegisterGradient("Expint")
def _ExpintGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Expint (exponential integral): gradient times the exponential of the input
  # divided by the input. Under a control dependency on the gradient. Returns a single tensor.
  """Compute gradient of expint(x) with respect to its argument."""
  x = op.inputs[0]
  with ops.control_dependencies([grad]):
    return grad * math_ops.exp(x) / x


@ops.RegisterGradient("FresnelCos")
def _FresnelCosGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for FresnelCos: gradient times the cosine of half pi times the square of the
  # input. Under a control dependency on the gradient. Returns a single tensor.
  """Compute gradient of fresnel_cos(x) with respect to its argument."""
  x = op.inputs[0]
  with ops.control_dependencies([grad]):
    return grad * math_ops.cos((np.pi  / 2.) * math_ops.square(x))


@ops.RegisterGradient("FresnelSin")
def _FresnelSinGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for FresnelSin: gradient times the sine of half pi times the square of the
  # input. Under a control dependency on the gradient. Returns a single tensor.
  """Compute gradient of fresnel_sin(x) with respect to its argument."""
  x = op.inputs[0]
  with ops.control_dependencies([grad]):
    return grad * math_ops.sin((np.pi  / 2.) * math_ops.square(x))


@ops.RegisterGradient("Spence")
def _SpenceGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Spence (dilogarithm): partial derivative is the log of the input divided by
  # one minus the input, with the removable singularity at one replaced by minus one. Multiplied by the
  # gradient under a control dependency on the gradient. Returns a single tensor.
  """Compute gradient of spence(x) with respect to its argument."""
  x = op.inputs[0]
  with ops.control_dependencies([grad]):
    partial_x = math_ops.log(x) / (1 - x)
    partial_x = array_ops.where(
        math_ops.equal(x, 1.), -array_ops.ones_like(x), partial_x)  # pylint: disable=invalid-unary-operand-type
    return grad * partial_x


@ops.RegisterGradient("BesselI0")
def _BesselI0Grad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for BesselI0: gradient times bessel_i1 of the input, under a control
  # dependency on the gradient. Returns a single tensor.
  """Compute gradient of bessel_i0(x) with respect to its argument."""
  x = op.inputs[0]
  with ops.control_dependencies([grad]):
    partial_x = special_math_ops.bessel_i1(x)
    return grad * partial_x


@ops.RegisterGradient("BesselI0e")
def _BesselI0eGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for BesselI0e (exponentially scaled): gradient times bessel_i1e of the input
  # minus sign of the input times the forward output. Under a control dependency on the gradient.
  # Returns a single tensor.
  """Compute gradient of bessel_i0e(x) with respect to its argument."""
  x = op.inputs[0]
  y = op.outputs[0]
  with ops.control_dependencies([grad]):
    partial_x = (special_math_ops.bessel_i1e(x) - math_ops.sign(x) * y)
    return grad * partial_x


@ops.RegisterGradient("BesselI1")
def _BesselI1Grad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for BesselI1. The derivative is bessel_i0 minus output over input, which is
  # undefined at zero, so a manual value of one is substituted where the input equals zero. Multiplied
  # by the gradient under a control dependency on the gradient. Returns a single tensor.
  """Compute gradient of bessel_i1(x) with respect to its argument."""
  x = op.inputs[0]
  y = op.outputs[0]
  with ops.control_dependencies([grad]):
    # For x = 0, the correct gradient is 1.0.
    # However, the main branch gives NaN because of the division by x, so
    # we impute the gradient manually.
    # An alternative solution is to express the gradient via bessel_i0 and
    # bessel_i2, but the latter is not yet implemented in Eigen.
    dy_dx = array_ops.where_v2(
        math_ops.equal(x, 0.), math_ops.cast(1., x.dtype),
        special_math_ops.bessel_i0(x) - math_ops.div(y, x))
    return grad * dy_dx


@ops.RegisterGradient("BesselI1e")
def _BesselI1eGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for BesselI1e. Derivative is bessel_i0e minus output times the sum of the sign
  # and the reciprocal of the input; at zero the value is imputed as one half to avoid not-a-number.
  # Multiplied by the gradient under a control dependency on the gradient. Returns a single tensor.
  """Compute gradient of bessel_i1e(x) with respect to its argument."""
  x = op.inputs[0]
  y = op.outputs[0]
  with ops.control_dependencies([grad]):
    # For x = 0, the correct gradient is 0.5.
    # However, the main branch gives NaN because of the division by x, so
    # we impute the gradient manually.
    # An alternative solution is to express the gradient via bessel_i0e and
    # bessel_i2e, but the latter is not yet implemented in Eigen.
    dy_dx = array_ops.where_v2(
        math_ops.equal(x, 0.), math_ops.cast(0.5, x.dtype),
        special_math_ops.bessel_i0e(x) - y *
        (math_ops.sign(x) + math_ops.reciprocal(x)))
    return grad * dy_dx


@ops.RegisterGradient("BesselK0")
def _BesselK0Grad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for BesselK0: gradient times minus bessel_k1 of the input, under a control
  # dependency on the gradient. Returns a single tensor.
  """Compute gradient of bessel_k0(x) with respect to its argument."""
  x = op.inputs[0]
  with ops.control_dependencies([grad]):
    partial_x = -special_math_ops.bessel_k1(x)
    return grad * partial_x


@ops.RegisterGradient("BesselK0e")
def _BesselK0eGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for BesselK0e: gradient times the forward output minus bessel_k1e of the input.
  # Under a control dependency on the gradient. Returns a single tensor.
  """Compute gradient of bessel_k0e(x) with respect to its argument."""
  x = op.inputs[0]
  y = op.outputs[0]
  with ops.control_dependencies([grad]):
    partial_x = (y - special_math_ops.bessel_k1e(x))
    return grad * partial_x


@ops.RegisterGradient("BesselK1")
def _BesselK1Grad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for BesselK1: gradient times minus bessel_k0 minus output over input. The
  # derivative is undefined (not-a-number) at zero, which is accepted. Under a control dependency on
  # the gradient. Returns a single tensor.
  """Compute gradient of bessel_k1(x) with respect to its argument."""
  x = op.inputs[0]
  y = op.outputs[0]
  with ops.control_dependencies([grad]):
    # At 0., this is NaN which is fine since the derivative is undefined
    # at 0.
    partial_x = -special_math_ops.bessel_k0(x) - math_ops.div(y, x)
    return grad * partial_x


@ops.RegisterGradient("BesselK1e")
def _BesselK1eGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for BesselK1e: gradient times output times one minus the reciprocal of the
  # input, minus bessel_k0e of the input. Not-a-number at zero is accepted. Under a control dependency
  # on the gradient. Returns a single tensor.
  """Compute gradient of bessel_k1e(x) with respect to its argument."""
  x = op.inputs[0]
  y = op.outputs[0]
  with ops.control_dependencies([grad]):
    # At 0., this is NaN which is fine since the derivative is undefined
    # at 0.
    partial_x = (
        y * (1. - math_ops.reciprocal(x)) - special_math_ops.bessel_k0e(x))
    return grad * partial_x


@ops.RegisterGradient("BesselJ0")
def _BesselJ0Grad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for BesselJ0: gradient times minus bessel_j1 of the input, under a control
  # dependency on the gradient. Returns a single tensor.
  """Compute gradient of bessel_j0(x) with respect to its argument."""
  x = op.inputs[0]
  with ops.control_dependencies([grad]):
    partial_x = -special_math_ops.bessel_j1(x)
    return grad * partial_x


@ops.RegisterGradient("BesselJ1")
def _BesselJ1Grad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the inline comment about the zero-input case talks about bessel_i0e and bessel_i2e
  # (copied from the I1e gradient) although the code uses bessel_j0; behavior is unaffected.
  # Gradient registered for BesselJ1: derivative is bessel_j0 minus output over input, with the value
  # one half imputed where the input is zero. Multiplied by the gradient under a control dependency on
  # the gradient. Returns a single tensor.
  """Compute gradient of bessel_j1(x) with respect to its argument."""
  x = op.inputs[0]
  y = op.outputs[0]
  with ops.control_dependencies([grad]):
    # For x = 0, the correct gradient is 0.5.
    # However, the main branch gives NaN because of the division by x, so
    # we impute the gradient manually.
    # An alternative solution is to express the gradient via bessel_i0e and
    # bessel_i2e, but the latter is not yet implemented in Eigen.
    dy_dx = array_ops.where_v2(
        math_ops.equal(x, 0.), math_ops.cast(0.5, x.dtype),
        special_math_ops.bessel_j0(x) - math_ops.div(y, x))
    return grad * dy_dx


@ops.RegisterGradient("BesselY0")
def _BesselY0Grad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for BesselY0: gradient times minus bessel_y1 of the input, under a control
  # dependency on the gradient. Returns a single tensor.
  """Compute gradient of bessel_y0(x) with respect to its argument."""
  x = op.inputs[0]
  with ops.control_dependencies([grad]):
    partial_x = -special_math_ops.bessel_y1(x)
    return grad * partial_x


@ops.RegisterGradient("BesselY1")
def _BesselY1Grad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for BesselY1: gradient times bessel_y0 minus output over input; not-a-number at
  # zero is accepted. Under a control dependency on the gradient. Returns a single tensor.
  """Compute gradient of bessel_y1(x) with respect to its argument."""
  x = op.inputs[0]
  y = op.outputs[0]
  with ops.control_dependencies([grad]):
    # At 0., this is NaN which is fine since the derivative is undefined
    # at 0.
    partial_x = special_math_ops.bessel_y0(x) - math_ops.div(y, x)
    return grad * partial_x


@ops.RegisterGradient("Igamma")
def _IgammaGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Igamma (regularized lower incomplete gamma). Partial for a comes from the
  # igamma-grad-a kernel; partial for x is computed in log space as the exponential of minus x plus
  # (a minus one) times log x minus lgamma of a, to avoid overflow. Each partial is multiplied by the
  # gradient, summed over broadcast axes and reshaped to the operand shape. Returns a pair for a and x.
  """Returns gradient of igamma(a, x) with respect to a and x."""
  a = op.inputs[0]
  x = op.inputs[1]
  sa = array_ops.shape(a)
  sx = array_ops.shape(x)
  ra, rx = gen_array_ops.broadcast_gradient_args(sa, sx)

  with ops.control_dependencies([grad]):
    partial_a = gen_math_ops.igamma_grad_a(a, x)
    # Perform operations in log space before summing, because Gamma(a)
    # and Gamma'(a) can grow large.
    partial_x = math_ops.exp(-x + (a - 1) * math_ops.log(x) -
                             math_ops.lgamma(a))
    return (array_ops.reshape(math_ops.reduce_sum(partial_a * grad, ra), sa),
            array_ops.reshape(math_ops.reduce_sum(partial_x * grad, rx), sx))


@ops.RegisterGradient("Igammac")
def _IgammacGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Igammac (upper incomplete gamma, one minus Igamma). Computes the Igamma
  # gradients with the same op and gradient and returns both negated, as a pair for a and x.
  """Returns gradient of igammac(a, x) = 1 - igamma(a, x) w.r.t. a and x."""
  igamma_grad_a, igamma_grad_x = _IgammaGrad(op, grad)
  return (-igamma_grad_a, -igamma_grad_x)


@ops.RegisterGradient("Betainc")
def _BetaincGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Betainc. Only the gradient with respect to x is implemented: the density
  # of the beta distribution, computed in log space with xlog1py and xlogy and the log beta function,
  # multiplied by the gradient, summed over broadcast axes and reshaped. Returns a triple with null
  # values for a and b and the gradient for x. The input shapes are compared against the shape of a only.
  """Returns gradient of betainc(a, b, x) with respect to x."""
  # TODO(ebrevdo): Perhaps add the derivative w.r.t. a, b
  a, b, x = op.inputs

  # two cases: x is a scalar and a/b are same-shaped tensors, or vice
  # versa; so its sufficient to check against shape(a).
  sa = array_ops.shape(a)
  sx = array_ops.shape(x)
  _, rx = gen_array_ops.broadcast_gradient_args(sa, sx)

  # Perform operations in log space before summing, because terms
  # can grow large.
  log_beta = (
      gen_math_ops.lgamma(a) + gen_math_ops.lgamma(b) -
      gen_math_ops.lgamma(a + b))
  # We use xlog1py and xlogy since the derivatives should tend to
  # zero one of the tails when a is 1. or b is 1.
  partial_x = math_ops.exp(math_ops.xlog1py(b - 1, -x) +
                           math_ops.xlogy(a - 1, x) - log_beta)

  return (
      None,  # da
      None,  # db
      array_ops.reshape(math_ops.reduce_sum(partial_x * grad, rx), sx))


@ops.RegisterGradient("Zeta")
def _ZetaGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Zeta (Hurwitz zeta). Only the derivative with respect to q is implemented:
  # minus x times zeta of x plus one and q (conjugated inputs). It is multiplied by the gradient, summed
  # over broadcast axes and reshaped. Returns a pair: null for x, gradient for q.
  """Returns gradient of zeta(x, q) with respect to x and q."""
  # TODO(tillahoffmann): Add derivative with respect to x
  x = op.inputs[0]
  q = op.inputs[1]
  # Broadcast gradients
  sx = array_ops.shape(x)
  sq = array_ops.shape(q)
  unused_rx, rq = gen_array_ops.broadcast_gradient_args(sx, sq)
  # Evaluate gradient
  with ops.control_dependencies([grad]):
    x = math_ops.conj(x)
    q = math_ops.conj(q)
    partial_q = -x * math_ops.zeta(x + 1, q)  # pylint: disable=invalid-unary-operand-type
    return (None,
            array_ops.reshape(math_ops.reduce_sum(partial_q * grad, rq), sq))


@ops.RegisterGradient("Polygamma")
def _PolygammaGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Polygamma. Only the derivative with respect to x is implemented: polygamma
  # of n plus one and x (conjugated inputs). Multiplied by the gradient, summed over broadcast axes and
  # reshaped. Returns a pair: null for n, gradient for x.
  """Returns gradient of psi(n, x) with respect to n and x."""
  # TODO(tillahoffmann): Add derivative with respect to n
  n = op.inputs[0]
  x = op.inputs[1]
  # Broadcast gradients
  sn = array_ops.shape(n)
  sx = array_ops.shape(x)
  unused_rn, rx = gen_array_ops.broadcast_gradient_args(sn, sx)
  # Evaluate gradient
  with ops.control_dependencies([grad]):
    n = math_ops.conj(n)
    x = math_ops.conj(x)
    partial_x = math_ops.polygamma(n + 1, x)
    return (None,
            array_ops.reshape(math_ops.reduce_sum(partial_x * grad, rx), sx))


@ops.RegisterGradient("Sigmoid")
def _SigmoidGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Sigmoid: uses the sigmoid-grad kernel with the conjugated forward output
  # and the incoming gradient, under a control dependency on the gradient. Returns a single tensor.
  """Returns grad * sigmoid(x) * (1 - sigmoid(x))."""
  y = op.outputs[0]  # y = sigmoid(x)
  with ops.control_dependencies([grad]):
    y = math_ops.conj(y)
    return gen_math_ops.sigmoid_grad(y, grad)


@ops.RegisterGradient("SigmoidGrad")
def _SigmoidGradGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Second-order gradient registered for SigmoidGrad. With both inputs conjugated and under a control
  # dependency on the gradient, returns a pair: gradient times second input times (one minus two times
  # the first input), and the sigmoid-grad of the first input and the gradient.
  with ops.control_dependencies([grad]):
    a = math_ops.conj(op.inputs[0])
    b = math_ops.conj(op.inputs[1])
    gb = grad * b
    return gb - 2.0 * gb * a, gen_math_ops.sigmoid_grad(a, grad)


@ops.RegisterGradient("Sign")
def _SignGrad(op: ops.Operation, _):
  # [FUNCTION DOC]
  # Gradient registered for Sign: always zero. Returns a zeros tensor shaped like the input; the
  # incoming gradient is ignored.
  """Returns 0."""
  x = op.inputs[0]
  return array_ops.zeros_like(x)


@ops.RegisterGradient("Sin")
def _SinGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Sin: gradient times cosine of the conjugated input, under a control
  # dependency on the gradient. Returns a single tensor.
  """Returns grad * cos(x)."""
  x = op.inputs[0]
  with ops.control_dependencies([grad]):
    x = math_ops.conj(x)
    return grad * math_ops.cos(x)


@ops.RegisterGradient("Cos")
def _CosGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Cos: minus gradient times sine of the conjugated input, under a control
  # dependency on the gradient. Returns a single tensor.
  """Returns grad * -sin(x)."""
  x = op.inputs[0]
  with ops.control_dependencies([grad]):
    x = math_ops.conj(x)
    return -grad * math_ops.sin(x)


@ops.RegisterGradient("Tan")
def _TanGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says the result is the gradient times one over secant squared, but the
  # code multiplies the gradient by secant squared (the correct derivative of tangent).
  # Gradient registered for Tan. Computes secant as the reciprocal of the cosine of the conjugated
  # input, squares it and multiplies by the gradient, under a control dependency on the gradient.
  # Returns a single tensor.
  """Returns grad * 1/sec^2(x)."""
  x = op.inputs[0]
  with ops.control_dependencies([grad]):
    x = math_ops.conj(x)
    secx = math_ops.reciprocal(math_ops.cos(x))
    secx2 = math_ops.square(secx)
    return secx2 * grad


@ops.RegisterGradient("Asin")
def _AsinGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Asin: gradient times the reciprocal of the square root of one minus the
  # square of the conjugated input, under a control dependency on the gradient. Returns a single tensor.
  """Returns grad * 1/sqrt(1-x^2)."""
  x = op.inputs[0]
  with ops.control_dependencies([grad]):
    x = math_ops.conj(x)
    x2 = math_ops.square(x)
    one = constant_op.constant(1, dtype=grad.dtype)
    den = math_ops.sqrt(math_ops.subtract(one, x2))
    inv = math_ops.reciprocal(den)
    return grad * inv


@ops.RegisterGradient("Acos")
def _AcosGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Acos: like the Asin gradient but negated (minus gradient over the square
  # root of one minus x squared), under a control dependency on the gradient. Returns a single tensor.
  """Returns grad * -1/sqrt(1-x^2)."""
  x = op.inputs[0]
  with ops.control_dependencies([grad]):
    x = math_ops.conj(x)
    x2 = math_ops.square(x)
    one = constant_op.constant(1, dtype=grad.dtype)
    den = math_ops.sqrt(math_ops.subtract(one, x2))
    inv = math_ops.reciprocal(den)
    return -grad * inv


@ops.RegisterGradient("Atan")
def _AtanGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Atan: gradient times the reciprocal of one plus the square of the
  # conjugated input, under a control dependency on the gradient. Returns a single tensor.
  """Returns grad * 1/ (1 + x^2)."""
  x = op.inputs[0]
  with ops.control_dependencies([grad]):
    x = math_ops.conj(x)
    x2 = math_ops.square(x)
    one = constant_op.constant(1, dtype=grad.dtype)
    inv = math_ops.reciprocal(math_ops.add(one, x2))
    return grad * inv


@ops.RegisterGradient("Atan2")
def _Atan2Grad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Atan2 with inputs y and x. Divides the gradient by the sum of squares of
  # y and x, giving the gradient for y as x times that and for x as minus y times that. The pair is
  # reduced to the operand shapes with the broadcast reduction helper; the argument order passed to it
  # is deliberately swapped relative to the helper's parameter names. Returns a pair for y and x.
  """Returns grad * x / (y^2 + x^2), grad * -y / (y^2 + x^2)."""
  y = op.inputs[0]
  x = op.inputs[1]
  with ops.control_dependencies([grad]):
    grad_inv = grad / (math_ops.square(y) + math_ops.square(x))
    gy = x * grad_inv
    gx = -y * grad_inv
    # pylint: disable=arguments-out-of-order
    return _ReduceGradientArgs(y, x, gy, gx)
    # pylint: enable=arguments-out-of-order


@ops.RegisterGradient("AddN")
def _AddNGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for AddN: every input receives the same incoming gradient (no broadcasting).
  # Returns a list with one entry per input, all referring to the same gradient.
  """Copies the gradient to all inputs."""
  # Not broadcasting.
  return [grad] * len(op.inputs)


def _ShapesFullySpecifiedAndEqual(x, y, grad):
  # [FUNCTION DOC]
  # HELPER: fast-path check used by Add, Sub, Mul, MulNoNan and SquaredDifference gradients.
  # Returns true when the static shapes of x, y and grad are known, fully defined and identical, so no
  # broadcast reduction is needed. Pure check using protected shape accessors; no side effects.
  # pylint: disable=protected-access
  x_shape = x._shape_tuple()
  y_shape = y._shape_tuple()
  grad_shape = grad._shape_tuple()
  # pylint: enable=protected-access
  return (x_shape == y_shape and x_shape == grad_shape and
          x_shape is not None and None not in x_shape)


@ops.RegisterGradient("Add")
@ops.RegisterGradient("AddV2")
def _AddGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Add and AddV2. If the op can skip input gradients and the second input is a
  # scalar and its gradient is skipped, returns the gradient and null. When the gradient is a tensor and
  # x, y and gradient have the same fully known shape, returns the gradient for both. Otherwise
  # passes the gradient (or null for skipped inputs) through the broadcast reduction helper to match
  # operand shapes. Ops lacking the skip-indices attribute are treated as skipping nothing. Returns a
  # pair of gradients for x and y.
  """Gradient for Add."""
  y = op.inputs[1]
  try:
    skip_input_indices = op.skip_input_indices or ()
    if 1 in skip_input_indices and _IsScalar(y):
      return grad, None
  except AttributeError:
    # No gradient skipping, so do the full gradient computation
    skip_input_indices = ()

  x = op.inputs[0]
  if isinstance(grad, tensor.Tensor) and _ShapesFullySpecifiedAndEqual(
      x, y, grad
  ):
    return grad, grad

  gx = None if 0 in skip_input_indices else grad
  gy = None if 1 in skip_input_indices else grad
  return _ReduceGradientArgs(x, y, gx, gy)


@ops.RegisterGradient("Sub")
def _SubGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Sub. Same structure as the Add gradient but the second input receives the
  # negated gradient: scalar fast path (gradient, null), same-static-shape fast path (gradient,
  # negated gradient), otherwise the broadcast-reduced gradients honoring skipped inputs. Returns a pair
  # for x and y.
  """Gradient for Sub."""
  y = op.inputs[1]
  try:
    skip_input_indices = op.skip_input_indices or ()
    if 1 in skip_input_indices and _IsScalar(y):
      return grad, None
  except AttributeError:
    # No gradient skipping, so do the full gradient computation
    skip_input_indices = ()

  x = op.inputs[0]
  if isinstance(grad, tensor.Tensor) and _ShapesFullySpecifiedAndEqual(
      x, y, grad
  ):
    return grad, -grad

  gx = None if 0 in skip_input_indices else grad
  gy = None if 1 in skip_input_indices else -grad
  return _ReduceGradientArgs(x, y, gx, gy)


@ops.RegisterGradient("Mul")
def _MulGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Mul (elementwise multiply). Fast path when the second input is a scalar and
  # its gradient is skipped: gradient times conjugated y, null. Fast path for same fully known shapes and
  # float32 or int32 gradients: gradient times y and gradient times x. Otherwise asserts both inputs have
  # the same base dtype (assertion error if not), computes gradient times conjugated y for x and
  # conjugated x times gradient for y (null for skipped inputs), and reduces over broadcast axes.
  # Returns a pair for x and y.
  """The gradient of scalar multiplication."""
  y = op.inputs[1]
  try:
    skip_input_indices = op.skip_input_indices or ()
    if 1 in skip_input_indices and _IsScalar(y):
      return gen_math_ops.mul(grad, math_ops.conj(y)), None
  except AttributeError:
    # No gradient skipping, so do the full gradient computation
    skip_input_indices = ()

  x = op.inputs[0]
  if (
      isinstance(grad, tensor.Tensor)
      and _ShapesFullySpecifiedAndEqual(x, y, grad)
      and grad.dtype in (dtypes.int32, dtypes.float32)
  ):
    return gen_math_ops.mul(grad, y), gen_math_ops.mul(grad, x)
  assert x.dtype.base_dtype == y.dtype.base_dtype, (x.dtype, " vs. ", y.dtype)

  if 0 in skip_input_indices:
    gx = None
  else:
    gx = gen_math_ops.mul(grad, math_ops.conj(y))

  if 1 in skip_input_indices:
    gy = None
  else:
    gy = gen_math_ops.mul(math_ops.conj(x), grad)

  return _ReduceGradientArgs(x, y, gx, gy)


@ops.RegisterGradient("MulNoNan")
def _MulNoNanGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for MulNoNan. Computes the gradients with the mul-no-nan kernel (gradient with
  # y for x, x with gradient for y). If the shapes are fully known and equal, returns them directly;
  # otherwise asserts both inputs have the same base dtype and reduces them over broadcast axes.
  # Returns a pair for x and y. Does not honor skipped input indices.
  """The gradient of scalar multiplication with NaN-suppression."""
  x = op.inputs[0]
  y = op.inputs[1]
  if isinstance(grad, tensor.Tensor) and _ShapesFullySpecifiedAndEqual(
      x, y, grad
  ):
    return gen_math_ops.mul_no_nan(grad, y), gen_math_ops.mul_no_nan(x, grad)

  assert x.dtype.base_dtype == y.dtype.base_dtype, (x.dtype, " vs. ", y.dtype)
  gx = gen_math_ops.mul_no_nan(grad, y)
  gy = gen_math_ops.mul_no_nan(x, grad)
  return _ReduceGradientArgs(x, y, gx, gy)


@ops.RegisterGradient("Div")
def _DivGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Div. Gradient for x is the gradient divided by the conjugated y; gradient for
  # y is the gradient times minus conjugated x over conjugated y squared. Both are reduced over broadcast
  # axes. Returns a pair for x and y.
  """The gradient for the Div operator."""
  x = op.inputs[0]
  y = op.inputs[1]
  cx = math_ops.conj(x)
  cy = math_ops.conj(y)
  gx = math_ops.divide(grad, cy)
  gy = grad * math_ops.divide(math_ops.divide(-cx, cy), cy)
  return _ReduceGradientArgs(x, y, gx, gy)


@ops.RegisterGradient("FloorDiv")
def _FloorDivGrad(_, unused_grad):
  # [FUNCTION DOC]
  # Gradient registered for FloorDiv: not differentiable, returns a pair of null values and ignores
  # its arguments.
  """The gradient for the FloorDiv operator."""
  return None, None


@ops.RegisterGradient("FloorMod")
def _FloorModGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for FloorMod: the gradient for x is the incoming gradient and for y is the
  # gradient times minus floor division of conjugated x by conjugated y. Both are reduced over broadcast
  # axes. Returns a pair for x and y.
  """Returns grad * (1, -floor(x/y))."""
  x = math_ops.conj(op.inputs[0])
  y = math_ops.conj(op.inputs[1])
  floor_xy = math_ops.floor_div(x, y)
  gx = grad
  gy = grad * math_ops.negative(floor_xy)
  return _ReduceGradientArgs(x, y, gx, gy)


@ops.RegisterGradient("TruncateDiv")
def _TruncateDivGrad(_, unused_grad):
  # [FUNCTION DOC]
  # Gradient registered for TruncateDiv: not differentiable, returns a pair of null values and ignores
  # its arguments.
  return None, None


@ops.RegisterGradient("RealDiv")
def _RealDivGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for RealDiv: same math as the Div gradient but using real division. Gradient for x
  # is the gradient over conjugated y, for y it is the gradient times minus conjugated x over conjugated y
  # squared; both reduced over broadcast axes. Returns a pair for x and y.
  """RealDiv op gradient."""
  x = op.inputs[0]
  y = op.inputs[1]
  cx = math_ops.conj(op.inputs[0])
  cy = math_ops.conj(op.inputs[1])
  gx = math_ops.realdiv(grad, cy)
  gy = grad * math_ops.realdiv(math_ops.realdiv(-cx, cy), cy)
  return _ReduceGradientArgs(x, y, gx, gy)


@ops.RegisterGradient("DivNoNan")
def _DivNoNanGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for DivNoNan: like the Div gradient but with safe division (zero where the
  # divisor is zero). Gradient for x is gradient over conjugated y, for y the gradient times minus
  # conjugated x over conjugated y squared, both reduced over broadcast axes. Returns a pair.
  """DivNoNan op gradient."""
  x = math_ops.conj(op.inputs[0])
  y = math_ops.conj(op.inputs[1])
  gx = math_ops.div_no_nan(grad, y)
  gy = grad * math_ops.div_no_nan(math_ops.div_no_nan(-x, y), y)
  return _ReduceGradientArgs(x, y, gx, gy)


@ops.RegisterGradient("Pow")
def _PowGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring mentions z, which is not defined; it presumably denotes the forward
  # output of the op.
  # Gradient registered for Pow. Gradient for x is gradient times y times x raised to y minus one
  # (conjugated). Gradient for y is gradient times conjugated output times log of x, where the log is
  # replaced by zero for non-positive x (real case) or for zero x (complex case) to avoid false
  # singularities. Honors skipped input indices (with a fast path for a scalar exponent), then reduces
  # over broadcast axes. Returns a pair for x and y.
  """Returns grad * (y*x^(y-1), z*log(x))."""
  x = op.inputs[0]
  y = op.inputs[1]
  cx = math_ops.conj(x)
  cy = math_ops.conj(y)
  try:
    skip_input_indices = op.skip_input_indices or ()
    if 1 in skip_input_indices and _IsScalar(y):
      return grad * cy * math_ops.pow(cx, cy - 1), None
  except AttributeError:
    # No gradient skipping, so do the full gradient computation
    skip_input_indices = ()

  if 0 in skip_input_indices:
    gx = None
  else:
    gx = grad * cy * math_ops.pow(cx, cy - 1)

  if 1 in skip_input_indices:
    gy = None
  else:
    # Avoid false singularity at x = 0
    if x.dtype.is_complex:
      # real(x) < 0 is fine for the complex case
      mask = math_ops.not_equal(cx, 0)
    else:
      # There's no sensible real value to return if x < 0, so return 0
      mask = cx > 0
    safe_x = array_ops.where(mask, cx, array_ops.ones_like(x))
    log_x = array_ops.where(mask, math_ops.log(safe_x), array_ops.zeros_like(x))
    gy = grad * math_ops.conj(op.outputs[0]) * log_x

  return _ReduceGradientArgs(x, y, gx, gy)


def _MaximumMinimumGradInputOnly(op: ops.Operation, grad, selector_op):
  # [FUNCTION DOC]
  # HELPER: fast path of the Maximum/Minimum gradient when only the first input's gradient is needed.
  # Inputs: the op, the gradient and a comparison function. Builds a mask by applying the comparison to x
  # and y, and keeps the gradient where the mask is true and zero elsewhere. Returns a pair: that
  # gradient and null for y (the second input is a scalar whose gradient is skipped).
  x = op.inputs[0]
  y = op.inputs[1]
  zeros = array_ops.zeros_like(grad)
  xmask = selector_op(x, y)
  xgrad = array_ops.where_v2(xmask, grad, zeros)
  ygrad = None  # Return None for ygrad since the config allows that.
  return (xgrad, ygrad)


def _MaximumMinimumGrad(op: ops.Operation, grad, selector_op):
  # [FUNCTION DOC]
  # HELPER: shared by the Maximum and Minimum gradients.
  # Inputs: the op, the gradient, and a comparison function (greater-equal for maximum, less-equal for
  # minimum) producing a mask of where x is selected. If the second input is a scalar whose gradient is
  # skipped, uses the input-only fast path. Otherwise the gradient for x is the gradient where the mask
  # is true (zeros elsewhere) and for y the gradient where it is false, with null for skipped inputs;
  # both are reduced over broadcast axes. Returns a pair for x and y.
  """Factor out the code for the gradient of Maximum or Minimum."""
  y = op.inputs[1]
  try:
    skip_input_indices = op.skip_input_indices or ()
    if 1 in skip_input_indices and _IsScalar(y):
      # When we want to get gradients for the first input only, and the second
      # input tensor is a scalar, we can do a much simpler calculation
      return _MaximumMinimumGradInputOnly(op, grad, selector_op)
  except AttributeError:
    # No gradient skipping, so do the full gradient computation
    skip_input_indices = ()
  x = op.inputs[0]
  zeros = array_ops.zeros_like(grad)
  xmask = selector_op(x, y)
  if 0 in skip_input_indices:
    gx = None
  else:
    gx = array_ops.where_v2(xmask, grad, zeros)
  if 1 in skip_input_indices:
    gy = None
  else:
    gy = array_ops.where_v2(xmask, zeros, grad)
  return _ReduceGradientArgs(x, y, gx, gy)


@ops.RegisterGradient("Maximum")
def _MaximumGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Maximum: delegates to the shared helper using the greater-or-equal
  # comparison, so x receives the gradient where it is at least y (ties go to x). Returns a pair.
  """Returns grad*(x >= y, x < y) with type of grad."""
  return _MaximumMinimumGrad(op, grad, math_ops.greater_equal)


@ops.RegisterGradient("Minimum")
def _MinimumGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Minimum: delegates to the shared helper using the less-or-equal comparison,
  # so x receives the gradient where it is at most y (ties go to x). Returns a pair.
  """Returns grad*(x <= y, x > y) with type of grad."""
  return _MaximumMinimumGrad(op, grad, math_ops.less_equal)


@ops.RegisterGradient("SquaredDifference")
def _SquaredDifferenceGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for SquaredDifference. Computes two times the gradient times the difference of
  # x and y, under a control dependency on the gradient (scalar multiplication keeps sparse gradients
  # convertible). The gradient for x is that value and for y its negation. With equal fully known shapes
  # returns them directly; otherwise honors skipped input indices and reduces over broadcast axes.
  # Returns a pair for x and y.
  """Returns the gradient for (x-y)^2."""
  x = op.inputs[0]
  y = op.inputs[1]
  try:
    skip_input_indices = op.skip_input_indices or ()
  except AttributeError:
    # No gradient skipping, so do the full gradient computation
    skip_input_indices = ()

  with ops.control_dependencies([grad]):
    # The parens ensure that if grad is IndexedSlices, it'll get multiplied by
    # Tensor (not a number like 2.0) which causes it to convert to Tensor.
    x_grad = math_ops.scalar_mul(2.0, grad) * (x - y)

  if isinstance(grad, tensor.Tensor) and _ShapesFullySpecifiedAndEqual(
      x, y, grad
  ):
    return x_grad, -x_grad

  gx = None if 0 in skip_input_indices else x_grad
  gy = None if 1 in skip_input_indices else -x_grad
  return _ReduceGradientArgs(x, y, gx, gy)


# Logical operations have no gradients.
ops.NotDifferentiable("Less")
ops.NotDifferentiable("LessEqual")
ops.NotDifferentiable("Greater")
ops.NotDifferentiable("GreaterEqual")
ops.NotDifferentiable("Equal")
ops.NotDifferentiable("ApproximateEqual")
ops.NotDifferentiable("NotEqual")
ops.NotDifferentiable("LogicalAnd")
ops.NotDifferentiable("LogicalOr")
ops.NotDifferentiable("LogicalNot")


@ops.RegisterGradient("Select")
def _SelectGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Select (legacy condition/then/else op). The condition gets null; the then
  # branch gets the gradient where the condition is true and zeros elsewhere; the else branch the
  # opposite. Returns a triple for condition, x and y.
  c = op.inputs[0]
  x = op.inputs[1]
  zeros = array_ops.zeros_like(x)
  return (
      None,
      array_ops.where(c, grad, zeros),
      array_ops.where(c, zeros, grad),
  )


@ops.RegisterGradient("SelectV2")
def _SelectGradV2(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for SelectV2, which allows broadcasting. Creates a scalar zero in the gradient
  # dtype, selects the gradient for the then and else branches using the condition, and reduces each
  # over broadcast axes against the output shape. Returns a triple: null for the condition, then
  # gradient for x, and else gradient for y.
  c = op.inputs[0]
  x = op.inputs[1]
  y = op.inputs[2]
  z = op.outputs[0]
  zeros = array_ops.zeros([], dtype=grad.dtype.base_dtype)
  gx = array_ops.where_v2(c, grad, zeros)
  gy = array_ops.where_v2(c, zeros, grad)
  gx, _ = _ReduceGradientArgs(x, z, gx, None)
  gy, _ = _ReduceGradientArgs(y, z, gy, None)
  return None, gx, gy


def _MatMulGradAgainstFirstOnly(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # HELPER: MatMul gradient when only the first input needs a gradient.
  # Using the conjugated second input and the transpose attributes of the op, produces the first
  # input's gradient with the appropriate mat-mul transposition for the four transpose combinations,
  # marking it as a gradient for the first operand. Returns a pair: that gradient and null.
  """Gradient for MatMul, only for the first input."""
  t_a = op.get_attr("transpose_a")
  t_b = op.get_attr("transpose_b")
  b = math_ops.conj(op.inputs[1])
  if not t_a and not t_b:
    grad_a = gen_math_ops.mat_mul(grad, b, transpose_b=True, grad_a=True)
  elif not t_a and t_b:
    grad_a = gen_math_ops.mat_mul(grad, b, grad_a=True)
  elif t_a and not t_b:
    grad_a = gen_math_ops.mat_mul(b, grad, transpose_b=True, grad_a=True)
  elif t_a and t_b:
    grad_a = gen_math_ops.mat_mul(
        b, grad, transpose_a=True, transpose_b=True, grad_a=True
    )
  return grad_a, None


def _MatMulGradAgainstSecondOnly(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # HELPER: MatMul gradient when only the second input needs a gradient.
  # Using the conjugated first input and the transpose attributes, produces the second input's gradient
  # with the right operand order and transposition for each of the four transpose combinations.
  # Returns a pair: null and that gradient.
  """Gradient for MatMul, only for the second input."""
  t_a = op.get_attr("transpose_a")
  t_b = op.get_attr("transpose_b")
  a = math_ops.conj(op.inputs[0])
  if not t_a and not t_b:
    grad_b = gen_math_ops.mat_mul(a, grad, transpose_a=True, grad_b=True)
  elif not t_a and t_b:
    grad_b = gen_math_ops.mat_mul(grad, a, transpose_a=True, grad_b=True)
  elif t_a and not t_b:
    grad_b = gen_math_ops.mat_mul(a, grad, grad_b=True)
  elif t_a and t_b:
    grad_b = gen_math_ops.mat_mul(
        grad, a, transpose_a=True, transpose_b=True, grad_b=True
    )
  return None, grad_b


@ops.RegisterGradient("MatMul")
def _MatMulGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for MatMul. If the op reports skipped input indices, uses the single-input
  # helpers; ops without that attribute fall through to the full computation. Otherwise reads the
  # transpose attributes and computes both gradients with mat-mul calls on the conjugated inputs,
  # choosing operand order and transposition for each of the four transpose combinations. Returns a pair
  # of gradients for the first and second input.
  """Gradient for MatMul."""
  try:
    skip_input_indices = op.skip_input_indices
    if skip_input_indices is not None:
      if 1 in skip_input_indices:
        return _MatMulGradAgainstFirstOnly(op, grad)
      elif 0 in skip_input_indices:
        return _MatMulGradAgainstSecondOnly(op, grad)
  except AttributeError:
    # No gradient skipping, so do the full gradient computation
    pass

  t_a = op.get_attr("transpose_a")
  t_b = op.get_attr("transpose_b")
  a = math_ops.conj(op.inputs[0])
  b = math_ops.conj(op.inputs[1])
  if not t_a and not t_b:
    grad_a = gen_math_ops.mat_mul(grad, b, transpose_b=True, grad_a=True)
    grad_b = gen_math_ops.mat_mul(a, grad, transpose_a=True, grad_b=True)
  elif not t_a and t_b:
    grad_a = gen_math_ops.mat_mul(grad, b, grad_a=True)
    grad_b = gen_math_ops.mat_mul(grad, a, transpose_a=True, grad_b=True)
  elif t_a and not t_b:
    grad_a = gen_math_ops.mat_mul(b, grad, transpose_b=True, grad_a=True)
    grad_b = gen_math_ops.mat_mul(a, grad, grad_b=True)
  elif t_a and t_b:
    grad_a = gen_math_ops.mat_mul(
        b, grad, transpose_a=True, transpose_b=True, grad_a=True
    )
    grad_b = gen_math_ops.mat_mul(
        grad, a, transpose_a=True, transpose_b=True, grad_b=True
    )
  return grad_a, grad_b


@ops.RegisterGradient("SparseMatMul")
def _SparseMatMulGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for SparseMatMul. Reads the transpose attributes and the sparsity hints of both
  # inputs, and guesses whether the gradient is sparse (in graph mode when its producer is a relu
  # gradient op). Computes both gradients with a nested helper that calls the general matmul with those
  # sparsity hints, for each of the four transpose combinations. Returns a pair of gradients for the
  # first and second input, cast back to each input's dtype.
  """Gradient for SparseMatMul."""

  t_a = op.get_attr("transpose_a")
  t_b = op.get_attr("transpose_b")
  is_sparse = {}
  is_sparse[op.inputs[0].ref()] = op.get_attr("a_is_sparse")
  is_sparse[op.inputs[1].ref()] = op.get_attr("b_is_sparse")
  # Use heuristic to figure out if grad might be sparse
  is_sparse[grad.ref()] = not context.executing_eagerly() and (
      grad.op.type == "ReluGrad")

  def _SparseMatMul(t1, t2, out_dtype, transpose_a=False, transpose_b=False):
    # [FUNCTION DOC]
    # HELPER: nested helper that builds one matmul with sparsity hints for the SparseMatMul gradient.
    # Inputs: two tensors, the output dtype, and transpose flags for each (default false). Asserts both
    # tensors are known in the sparsity table of the enclosing function. Transposes the second tensor
    # explicitly if requested, then calls the general matmul with the sparsity hints, and casts the
    # product to the output dtype if it differs. Returns the product tensor.
    """Helper function to create SparseMatMul op."""

    assert t1.ref() in is_sparse and t2.ref() in is_sparse
    t1_sparse = is_sparse[t1.ref()]
    t2_sparse = is_sparse[t2.ref()]
    if transpose_b:
      t2 = array_ops.transpose(t2)
      transpose_b = False
    prod = math_ops.matmul(
        t1,
        t2,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        a_is_sparse=t1_sparse,
        b_is_sparse=t2_sparse)
    if prod.dtype != out_dtype:
      prod = math_ops.cast(prod, out_dtype)
    return prod

  dtype_a = op.inputs[0].dtype
  dtype_b = op.inputs[1].dtype
  if not t_a and not t_b:
    return (_SparseMatMul(grad, op.inputs[1], dtype_a, transpose_b=True),
            _SparseMatMul(op.inputs[0], grad, dtype_b, transpose_a=True))
  elif not t_a and t_b:
    return (_SparseMatMul(grad, op.inputs[1], dtype_a),
            _SparseMatMul(grad, op.inputs[0], dtype_b, transpose_a=True))
  elif t_a and not t_b:
    return (_SparseMatMul(op.inputs[1], grad, dtype_a, transpose_b=True),
            _SparseMatMul(op.inputs[0], grad, dtype_b))
  elif t_a and t_b:
    return (_SparseMatMul(
        op.inputs[1], grad, dtype_a, transpose_a=True, transpose_b=True),
            _SparseMatMul(
                grad, op.inputs[0], dtype_b, transpose_a=True,
                transpose_b=True))


@ops.RegisterGradient("Floor")
def _FloorGrad(_, unused_grad):
  # [FUNCTION DOC]
  # Gradient registered for Floor: not differentiable, returns a list with one null value.
  return [None]


@ops.RegisterGradient("Ceil")
def _CeilGrad(_, unused_grad):
  # [FUNCTION DOC]
  # Gradient registered for Ceil: not differentiable, returns a list with one null value.
  return [None]


@ops.RegisterGradient("Round")
def _RoundGrad(_, unused_grad):
  # [FUNCTION DOC]
  # Gradient registered for Round: not differentiable, returns a list with one null value.
  return [None]


@ops.RegisterGradient("Rint")
def _RintGrad(_, unused_grad):
  # [FUNCTION DOC]
  # Gradient registered for Rint: returns a list with one null value (treated as no gradient flow).
  # the gradient of Rint is zero
  return [None]


@ops.RegisterGradient("BatchMatMul")
def _BatchMatMul(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for BatchMatMul (version one, no broadcasting of batch dimensions). Reads the
  # adjoint attributes of the op and computes the gradients for x and y with batched matmul calls whose
  # operand order and adjoint flags depend on the four adjoint combinations. Returns a pair of
  # gradients for x and y.
  """Returns the gradient of x and y given the gradient of x * y."""
  x = op.inputs[0]
  y = op.inputs[1]
  adj_x = op.get_attr("adj_x")
  adj_y = op.get_attr("adj_y")

  if not adj_x:
    if not adj_y:
      grad_x = math_ops.matmul(grad, y, adjoint_a=False, adjoint_b=True)
      grad_y = math_ops.matmul(x, grad, adjoint_a=True, adjoint_b=False)
    else:
      grad_x = math_ops.matmul(grad, y, adjoint_a=False, adjoint_b=False)
      grad_y = math_ops.matmul(grad, x, adjoint_a=True, adjoint_b=False)
  else:
    if not adj_y:
      grad_x = math_ops.matmul(y, grad, adjoint_a=False, adjoint_b=True)
      grad_y = math_ops.matmul(x, grad, adjoint_a=False, adjoint_b=False)
    else:
      grad_x = math_ops.matmul(y, grad, adjoint_a=True, adjoint_b=True)
      grad_y = math_ops.matmul(grad, x, adjoint_a=True, adjoint_b=True)

  return grad_x, grad_y


@ops.RegisterGradient("BatchMatMulV2")
@ops.RegisterGradient("BatchMatMulV3")
def _BatchMatMulV2(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for BatchMatMulV2 and BatchMatMulV3. Computes the gradients like the version
  # one function, additionally marking the matmul calls as gradient calls. If the static batch shapes
  # could differ (higher-rank inputs whose batch dimensions are not fully known and equal), the
  # gradients are summed over the broadcast batch dimensions and reshaped to the inputs' shapes using
  # the dynamic broadcast-gradient-args op. Returns a pair of gradients for x and y.
  """Returns the gradient of x and y given the gradient of x * y."""
  x = op.inputs[0]
  y = op.inputs[1]
  adj_x = op.get_attr("adj_x")
  adj_y = op.get_attr("adj_y")

  if not adj_x:
    if not adj_y:
      grad_x = math_ops.matmul(
          grad, y, adjoint_a=False, adjoint_b=True, grad_a=True
      )
      grad_y = math_ops.matmul(
          x, grad, adjoint_a=True, adjoint_b=False, grad_b=True
      )
    else:
      grad_x = math_ops.matmul(
          grad, y, adjoint_a=False, adjoint_b=False, grad_a=True
      )
      grad_y = math_ops.matmul(
          grad, x, adjoint_a=True, adjoint_b=False, grad_b=True
      )
  else:
    if not adj_y:
      grad_x = math_ops.matmul(
          y, grad, adjoint_a=False, adjoint_b=True, grad_a=True
      )
      grad_y = math_ops.matmul(
          x, grad, adjoint_a=False, adjoint_b=False, grad_b=True
      )
    else:
      grad_x = math_ops.matmul(
          y, grad, adjoint_a=True, adjoint_b=True, grad_a=True
      )
      grad_y = math_ops.matmul(
          grad, x, adjoint_a=True, adjoint_b=True, grad_b=True
      )

  # Possibly reduce along the broadcasted batch dimensions, if broadcasting
  # is required.
  shape_x_static = x.get_shape()
  shape_y_static = y.get_shape()
  output_may_have_non_empty_batch_shape = (
      (shape_x_static.rank is None or shape_x_static.rank > 2) or
      (shape_y_static.rank is None or shape_y_static.rank > 2))
  batch_shapes_match = (
      shape_x_static[:-2].is_fully_defined() and
      shape_y_static[:-2].is_fully_defined() and
      shape_x_static[:-2] == shape_y_static[:-2])
  if (not output_may_have_non_empty_batch_shape) or batch_shapes_match:
    return grad_x, grad_y

  sx = array_ops.shape(x)
  sy = array_ops.shape(y)
  rx, ry = gen_array_ops.broadcast_gradient_args(sx[:-2], sy[:-2])
  grad_x = array_ops.reshape(math_ops.reduce_sum(grad_x, rx), sx)
  grad_y = array_ops.reshape(math_ops.reduce_sum(grad_y, ry), sy)
  return grad_x, grad_y


ops.NotDifferentiable("Range")
ops.NotDifferentiable("LinSpace")


@ops.RegisterGradient("Complex")
def _ComplexGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Complex: the gradient for the real part is the real component of the
  # incoming gradient, and for the imaginary part its imaginary component, reduced over broadcast
  # axes. Returns a pair for the two inputs.
  """Returns the real and imaginary components of 'grad', respectively."""
  x = op.inputs[0]
  y = op.inputs[1]
  gx = math_ops.real(grad)
  gy = math_ops.imag(grad)
  return _ReduceGradientArgs(x, y, gx, gy)


@ops.RegisterGradient("Real")
def _RealGrad(_, grad):
  # [FUNCTION DOC]
  # Gradient registered for Real: builds a complex gradient with the incoming real gradient as the real
  # part and zero as the imaginary part. Returns a single complex tensor.
  """Returns 'grad' as the real part and set the imaginary part 0."""
  zero = constant_op.constant(0, dtype=grad.dtype)
  return math_ops.complex(grad, zero)


@ops.RegisterGradient("Imag")
def _ImagGrad(_, grad):
  # [FUNCTION DOC]
  # Gradient registered for Imag: builds a complex gradient with zero real part and the incoming
  # gradient as the imaginary part. Returns a single complex tensor.
  """Returns 'grad' as the imaginary part and set the real part 0."""
  zero = constant_op.constant(0, dtype=grad.dtype)
  return math_ops.complex(zero, grad)


@ops.RegisterGradient("Angle")
def _AngleGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Angle: minus the (complexified) gradient times the reciprocal of a complex
  # number whose real part is the input's imaginary part and whose imaginary part is the input's real
  # part. Evaluated under a control dependency on the gradient. Returns a single complex tensor.
  """Returns `-grad / (Im(x) + i Re(x))`."""
  x = op.inputs[0]
  with ops.control_dependencies([grad]):
    re = math_ops.real(x)
    im = math_ops.imag(x)
    z = math_ops.reciprocal(math_ops.complex(im, re))
    zero = constant_op.constant(0, dtype=grad.dtype)
    complex_grad = math_ops.complex(grad, zero)
    return -complex_grad * z


@ops.RegisterGradient("Conj")
def _ConjGrad(_, grad):
  # [FUNCTION DOC]
  # Gradient registered for Conj: returns the complex conjugate of the incoming gradient.
  """Returns the complex conjugate of grad."""
  return math_ops.conj(grad)


@ops.RegisterGradient("ComplexAbs")
def _ComplexAbsGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for ComplexAbs: the real gradient promoted to complex times the input, safely
  # divided by the forward output promoted to complex (zero where the magnitude is zero). Returns a
  # single complex tensor.
  """Returns the gradient of ComplexAbs."""
  return math_ops.div_no_nan(
      math_ops.complex(
          grad, array_ops.zeros_like(grad)) * op.inputs[0],
      math_ops.complex(
          op.outputs[0], array_ops.zeros_like(op.outputs[0])))


@ops.RegisterGradient("Cast")
def _CastGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Cast. If both the source and destination dtypes are floating point or
  # complex types (float16, float32, float64, bfloat16, complex64, complex128), casts the gradient back
  # to the source dtype; otherwise returns a null value (no gradient). Returns one tensor or null.
  t = [
      dtypes.float16, dtypes.float32, dtypes.float64, dtypes.bfloat16,
      dtypes.complex64, dtypes.complex128
  ]
  src_type = op.inputs[0].dtype.base_dtype
  dst_type = grad.dtype.base_dtype
  if src_type in t and dst_type in t:
    return math_ops.cast(grad, src_type)
  else:
    return None


@ops.RegisterGradient("Cross")
def _CrossGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Cross (three-element cross product). Returns a pair: the cross product of
  # the second input with the gradient for the first input, and the cross product of the gradient with
  # the first input for the second input.
  u = op.inputs[0]
  v = op.inputs[1]
  return (math_ops.cross(v, grad), math_ops.cross(grad, u))


@ops.RegisterGradient("Cumsum")
def _CumsumGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Cumsum. The gradient is a cumulative sum of the incoming gradient along the
  # same axis with the same exclusive flag but the reverse flag flipped. Returns a list: that tensor and
  # null for the axis.
  axis = op.inputs[1]
  exclusive = op.get_attr("exclusive")
  reverse = op.get_attr("reverse")
  return [
      math_ops.cumsum(grad, axis, exclusive=exclusive, reverse=not reverse),
      None
  ]


@ops.RegisterGradient("Cumprod")
def _CumprodGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for Cumprod. Recomputes the cumulative product of the input, multiplies it by
  # the gradient, takes a reverse-direction cumulative sum with the same exclusive flag and divides by
  # the input with safe division (zero where the input is zero, so gradients at zeros are dropped,
  # which is not the exact derivative). Returns a list: input gradient and null for the axis.
  x = op.inputs[0]
  axis = op.inputs[1]
  exclusive = op.get_attr("exclusive")
  reverse = op.get_attr("reverse")

  prod = math_ops.cumprod(x, axis, exclusive=exclusive, reverse=reverse)
  out = math_ops.cumsum(
      prod * grad, axis, exclusive=exclusive, reverse=not reverse
  )
  return [math_ops.div_no_nan(out, x), None]


# pylint: disable=missing-function-docstring
@ops.RegisterGradient("CumulativeLogsumexp")
def _CumulativeLogsumexpGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for CumulativeLogsumexp. Splits the gradient into positive and negative parts,
  # takes the log of each (using the dtype minimum where the part is empty) for numerical stability,
  # runs a reverse-direction cumulative logsumexp of each log gradient minus the forward output with
  # the same exclusive flag, adds the input, exponentiates and subtracts the negative part from the
  # positive one. Returns a list: input gradient and null for the axis.
  x = op.inputs[0]
  axis = op.inputs[1]
  cumulative_logsumexp = op.outputs[0]

  exclusive = op.get_attr("exclusive")
  reverse = op.get_attr("reverse")

  # Split the incoming gradient into positive and negative part
  # in order to take logs. This is required for stable results.
  log_grad_positive = array_ops.where_v2(
      math_ops.greater(grad, 0),
      math_ops.log(grad),
      grad.dtype.min)

  log_grad_negative = array_ops.where_v2(
      math_ops.less(grad, 0),
      math_ops.log(-grad),
      grad.dtype.min)

  output_pos = math_ops.exp(
      math_ops.cumulative_logsumexp(
          log_grad_positive - cumulative_logsumexp,
          axis=axis, reverse=not reverse, exclusive=exclusive) + x)

  output_neg = math_ops.exp(
      math_ops.cumulative_logsumexp(
          log_grad_negative - cumulative_logsumexp,
          axis=axis, reverse=not reverse, exclusive=exclusive) + x)

  return [output_pos - output_neg, None]


@ops.RegisterGradient("NextAfter")
def _NextAfterGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient registered for NextAfter. Treats the derivative with respect to the first input as one and
  # with respect to the second as zero; each is multiplied by the gradient, summed over broadcast axes
  # found dynamically, and reshaped to the input shape. Under a control dependency on the gradient.
  # Returns a pair for the two inputs.
  """Returns gradient of nextafter(x1, x2) with respect to x1 and x2."""
  x1 = op.inputs[0]
  x2 = op.inputs[1]
  s_x1 = array_ops.shape(x1)
  s_x2 = array_ops.shape(x2)
  r_x1, r_x2 = gen_array_ops.broadcast_gradient_args(s_x1, s_x2)
  with ops.control_dependencies([grad]):
    partial_x1 = array_ops.ones(s_x1, dtype=x1.dtype)
    partial_x2 = array_ops.zeros(s_x2, dtype=x2.dtype)
    return (array_ops.reshape(
        math_ops.reduce_sum(partial_x1 * grad, r_x1), s_x1),
            array_ops.reshape(
                math_ops.reduce_sum(partial_x2 * grad, r_x2), s_x2))
