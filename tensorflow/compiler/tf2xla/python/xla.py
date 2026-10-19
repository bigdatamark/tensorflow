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
# [MODULE DOC]
# Public Python facade over the XLA operators available in TensorFlow graphs. Most names are
# either aliases of generated XLA op wrappers, aliases of ordinary TensorFlow ops, or thin
# wrapper functions that adapt the signature (for example adding XLA-style explicit broadcast
# dimensions to binary operators, serializing configuration protocol buffers into strings, or
# choosing between op versions). Also registers gradients for the sharding and SPMD
# shard/full-shape ops. Operators here are used by the JAX-to-TensorFlow converter, so their
# behavior must stay backward compatible.
# Import-time side effects: saves the builtin max, min and slice under private names, then
# deliberately rebinds many builtin names (absolute value, max, min, pow, round, complex, slice,
# reduce and others) at module level; marks the variadic reduce op as non-differentiable; and
# registers three gradient functions. Importing requires the generated XLA op library.
# ==============================================================================
"""Experimental library that exposes XLA operations directly in TensorFlow.

It is sometimes useful to be able to build HLO programs directly from
TensorFlow. This file provides Tensorflow operators that mirror the semantics of
HLO operators as closely as possible.

Note: Most of the operators defined in this module are used by the jax2tf
converter (see go/jax2tf for details) and are used in SavedModel produced
by jax2tf. Hence, we need to maintain backwards compatibility for these
operators. Please reach out to the JAX team if you want to make changes.
"""

from tensorflow.compiler.tf2xla.ops import gen_xla_ops
from xla import xla_data_pb2
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import bitwise_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import random_ops_util
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops.numpy_ops import np_utils

# TODO(phawkins): provide wrappers for all XLA operators. Currently the missing
# ops include:
# infeed/outfeed (available via tf.contrib.tpu)
# collectives, e.g., cross-replica-sum (available via tf.contrib.tpu)
# conditional
# gather/scatter
# collapse

# This file reuses builtin names (following XLA's names, so we can call things
# like xla.max), so we capture the builtin versions here.
# pylint: disable=redefined-builtin
_max = max
_min = min
_slice = slice  # pylint: disable=invalid-name

constant = constant_op.constant

# Unary operators.

# For most arithmetic operators there is a TensorFlow operator
# that exactly corresponds to each XLA operator. Rather than defining
# XLA-specific variants, we reuse the corresponding TensorFlow operator.
# TODO(phawkins): It would be even better to have TensorFlow operators that 1:1
# wrap every HLO operator, because that would allow us to be confident that the
# semantics match.


def _unary_op(fn):
  # [FUNCTION DOC]
  # HELPER: factory used to define the module's unary operator aliases.
  # Input: fn, a TensorFlow function taking one tensor and an optional name. Returns a new
  # function accepting exactly a tensor and an optional name (default null) that forwards to fn,
  # thereby hiding any extra parameters of fn. No side effects.
  """Wrapper that restricts `fn` to have the correct signature."""

  def unary_op_wrapper(x, name=None):
    # [FUNCTION DOC]
    # HELPER: the function produced by the unary-op factory. Inputs: x (tensor), name (optional
    # operation name). Returns the result of the wrapped function applied to x.
    return fn(x, name=name)

  return unary_op_wrapper


abs = _unary_op(math_ops.abs)
# TODO(phawkins): implement clz.
conj = _unary_op(math_ops.conj)
cos = _unary_op(math_ops.cos)
ceil = _unary_op(math_ops.ceil)
digamma = _unary_op(math_ops.digamma)
erf = _unary_op(math_ops.erf)
erfc = _unary_op(math_ops.erfc)
erfinv = _unary_op(math_ops.erfinv)
ndtri = _unary_op(math_ops.ndtri)
exp = _unary_op(math_ops.exp)
expm1 = _unary_op(math_ops.expm1)
floor = _unary_op(math_ops.floor)
imag = _unary_op(math_ops.imag)
is_finite = _unary_op(math_ops.is_finite)
lgamma = _unary_op(math_ops.lgamma)
log = _unary_op(math_ops.log)
log1p = _unary_op(math_ops.log1p)
logical_not = _unary_op(math_ops.logical_not)
neg = _unary_op(math_ops.neg)
real = _unary_op(math_ops.real)
# TODO(phawkins): unlike xla::Round, this rounds to even instead of zero for
# numbers halfway between two integers.
round = _unary_op(math_ops.round)
sin = _unary_op(math_ops.sin)
sign = _unary_op(math_ops.sign)
tan = _unary_op(math_ops.tan)
tanh = _unary_op(math_ops.tanh)

# Bessel
bessel_i0e = _unary_op(special_math_ops.bessel_i0e)
bessel_i1e = _unary_op(special_math_ops.bessel_i1e)

# Binary operators

# The main difference between TensorFlow and XLA binary ops is the broadcasting
# semantics. TensorFlow uses Numpy-style broadcasting semantics, whereas XLA
# requires an explicit specification of which dimensions to broadcast if the
# arguments have different ranks.


def _broadcasting_binary_op(fn):
  # [FUNCTION DOC]
  # HELPER: factory used to define the module's binary operator aliases.
  # Input: fn, a TensorFlow function taking two tensors and an optional name. Returns a function
  # that first aligns the operands using the XLA broadcast-helper op and then applies fn.
  """Wraps a binary Tensorflow operator and performs XLA-style broadcasting."""

  def broadcasting_binary_op_wrapper(x, y, broadcast_dims=None, name=None):
    # [FUNCTION DOC]
    # HELPER: the function produced by the broadcasting binary-op factory.
    # Inputs: x and y (tensors, possibly of different ranks), broadcast_dims (optional list of
    # integers naming which dimensions of the larger operand the smaller one maps to; null means
    # an empty list), name (optional operation name). Converts broadcast_dims to a 64-bit
    # integer tensor, lets the XLA broadcast-helper op compute compatible shapes at compile
    # time, then returns the wrapped function applied to the broadcast operands.
    """Inner wrapper function."""
    broadcast_dims = broadcast_dims or []
    broadcast_dims = ops.convert_to_tensor(broadcast_dims, dtypes.int64)
    # Rather than relying on having static shape information in the TensorFlow
    # graph, we use an XlaBroadcastHelper op that can compute the correct shapes
    # at JIT compilation time.
    x, y = gen_xla_ops.xla_broadcast_helper(x, y, broadcast_dims)
    return fn(x, y, name=name)

  return broadcasting_binary_op_wrapper


# Map from TF signed types to TF unsigned types.
_SIGNED_TO_UNSIGNED_TABLE = {
    dtypes.int8: dtypes.uint8,
    dtypes.int16: dtypes.uint16,
    dtypes.int32: dtypes.uint32,
    dtypes.int64: dtypes.uint64,
}

# Map from TF unsigned types to TF signed types.
_UNSIGNED_TO_SIGNED_TABLE = {
    dtypes.uint8: dtypes.int8,
    dtypes.uint16: dtypes.int16,
    dtypes.uint32: dtypes.int32,
    dtypes.uint64: dtypes.int64,
}


def _shift_right_logical_helper(x, y, name=None):
  # [FUNCTION DOC]
  # HELPER: implementation behind the logical right-shift operator.
  # Inputs: x (integer tensor), y (shift amounts, same dtype as x, asserted), name (optional).
  # Signed inputs are cast to the unsigned type of equal width, shifted, and cast back, so the
  # shift fills with zeros; unsigned inputs are shifted directly. Returns the shifted tensor of
  # the original dtype. Raises an assertion error if dtypes differ.
  """Performs an integer right logical shift irrespective of input type."""
  assert y.dtype == x.dtype
  dtype = x.dtype
  signed = dtype in _SIGNED_TO_UNSIGNED_TABLE
  if signed:
    unsigned_dtype = _SIGNED_TO_UNSIGNED_TABLE[dtype]
    x = math_ops.cast(x, unsigned_dtype)
    y = math_ops.cast(y, unsigned_dtype)
  output = bitwise_ops.right_shift(x, y, name=name)
  if signed:
    output = math_ops.cast(output, dtype)
  return output


def _shift_right_arithmetic_helper(x, y, name=None):
  # [FUNCTION DOC]
  # HELPER: implementation behind the arithmetic right-shift operator.
  # Inputs: x (integer tensor), y (shift amounts, same dtype as x, asserted), name (optional).
  # Unsigned inputs are cast to the signed type of equal width, shifted, and cast back, so the
  # sign bit is preserved in the signed view; signed inputs are shifted directly. Returns the
  # shifted tensor of the original dtype. Raises an assertion error if dtypes differ.
  """Performs an integer right arithmetic shift irrespective of input type."""
  assert y.dtype == x.dtype
  dtype = x.dtype
  unsigned = dtype in _UNSIGNED_TO_SIGNED_TABLE
  if unsigned:
    signed_dtype = _UNSIGNED_TO_SIGNED_TABLE[dtype]
    x = math_ops.cast(x, signed_dtype)
    y = math_ops.cast(y, signed_dtype)
  output = bitwise_ops.right_shift(x, y, name=name)
  if unsigned:
    output = math_ops.cast(output, dtype)
  return output


add = _broadcasting_binary_op(math_ops.add)
sub = _broadcasting_binary_op(math_ops.sub)
mul = _broadcasting_binary_op(math_ops.mul)
div = _broadcasting_binary_op(math_ops.div)
rem = _broadcasting_binary_op(gen_math_ops.mod)
max = _broadcasting_binary_op(math_ops.maximum)
min = _broadcasting_binary_op(math_ops.minimum)
atan2 = _broadcasting_binary_op(math_ops.atan2)
complex = _broadcasting_binary_op(math_ops.complex)
logical_and = _broadcasting_binary_op(math_ops.logical_and)
logical_or = _broadcasting_binary_op(math_ops.logical_or)
logical_xor = _broadcasting_binary_op(math_ops.logical_xor)
eq = _broadcasting_binary_op(math_ops.equal)
ne = _broadcasting_binary_op(math_ops.not_equal)
ge = _broadcasting_binary_op(math_ops.greater_equal)
gt = _broadcasting_binary_op(math_ops.greater)
le = _broadcasting_binary_op(math_ops.less_equal)
lt = _broadcasting_binary_op(math_ops.less)
pow = _broadcasting_binary_op(math_ops.pow)
shift_left = _broadcasting_binary_op(bitwise_ops.left_shift)
shift_right_logical = _broadcasting_binary_op(_shift_right_logical_helper)
shift_right_arithmetic = _broadcasting_binary_op(_shift_right_arithmetic_helper)

igamma = _broadcasting_binary_op(math_ops.igamma)
igamma_grad_a = _broadcasting_binary_op(gen_math_ops.igamma_grad_a)
random_gamma_grad = _broadcasting_binary_op(gen_random_ops.random_gamma_grad)
igammac = _broadcasting_binary_op(math_ops.igammac)
polygamma = _broadcasting_binary_op(math_ops.polygamma)
zeta = _broadcasting_binary_op(math_ops.zeta)


def _binary_op(fn):
  # [FUNCTION DOC]
  # HELPER: factory used to define the transpose and reverse aliases.
  # Input: fn, a TensorFlow function taking two arguments and an optional name. Returns a
  # function accepting exactly two positional arguments and an optional name that forwards to fn.
  # No broadcasting is performed.
  """Wrapper that restricts `fn` to have the correct signature."""

  def binary_op_wrapper(x, y, name=None):
    # [FUNCTION DOC]
    # HELPER: the function produced by the binary-op factory. Inputs: x, y (the two arguments,
    # for example a tensor and a permutation or axes list), name (optional). Returns the result
    # of the wrapped function.
    return fn(x, y, name=name)

  return binary_op_wrapper


transpose = _binary_op(array_ops.transpose)
rev = _binary_op(array_ops.reverse)

bitcast_convert_type = array_ops.bitcast


def broadcast(x, dims, name=None):
  # [FUNCTION DOC]
  # XLA-style broadcast that prepends dimensions. Inputs: x (tensor or value convertible to
  # one), dims (list of sizes to prepend), name (optional). Builds the target shape as dims
  # followed by the shape of x and returns x broadcast to it.
  x = ops.convert_to_tensor(x)
  shape = array_ops.concat(
      [constant_op.constant(dims), array_ops.shape(x)], axis=0
  )
  return array_ops.broadcast_to(x, shape, name=name)


def clamp(a, x, b, name=None):
  # [FUNCTION DOC]
  # Clamps x between a (lower bound) and b (upper bound). Inputs: a, x, b (tensors broadcast
  # with XLA-style broadcasting using the module's maximum and minimum), name (optional).
  # Returns the elementwise minimum of b and the maximum of a and x.
  return min(max(a, x, name=name), b, name=name)


concatenate = array_ops.concat


def conv(
    lhs,
    rhs,
    window_strides,
    padding,
    lhs_dilation,
    rhs_dilation,
    dimension_numbers,
    feature_group_count=1,
    precision_config=None,
    preferred_element_type=None,
    name=None,
    use_v2=False,
    batch_group_count=1,
):
  # [FUNCTION DOC]
  # Wraps the XLA general dilated convolution operator.
  # Inputs: lhs (input tensor), rhs (kernel tensor), window_strides, padding, lhs_dilation,
  # rhs_dilation (all passed through to the op), dimension_numbers (a convolution dimension
  # numbers protocol buffer message, serialized here), feature_group_count (default 1),
  # precision_config (optional precision protocol buffer message, serialized; otherwise an empty
  # string), preferred_element_type (optional result dtype), name (optional), use_v2 (default
  # false; forces the second op version), batch_group_count (default 1).
  # The second op version is used if preferred_element_type was given, the operand dtypes differ,
  # batch_group_count exceeds one, or use_v2 is set; in that case a missing preferred element
  # type is derived by numpy-style promotion of the operand dtypes. Otherwise the first version
  # is used (without batch groups or element type). Returns the output tensor.
  # KNOWN ISSUE: the preferred element type is computed even on the first-version path, where
  # it is unused. The docstring also omits that the second op version may be chosen implicitly.
  """Wraps the XLA ConvGeneralDilated operator.

  ConvGeneralDilated is the most general form of XLA convolution and is
  documented at
  https://www.tensorflow.org/performance/xla/operation_semantics#conv_convolution

  Args:
    lhs: the input tensor
    rhs: the kernel tensor
    window_strides: the inter-window strides
    padding: the padding to apply at the start and end of each input dimensions
    lhs_dilation: dilation to apply between input elements
    rhs_dilation: dilation to apply between kernel elements
    dimension_numbers: a `ConvolutionDimensionNumbers` proto.
    feature_group_count: number of feature groups for grouped convolution.
    precision_config: a `xla.PrecisionConfig` proto.
    preferred_element_type: the result `dtype`.
    name: an optional name for the operator.
    use_v2: an optional request to use the XlaConvV2 op even if not necessary.
    batch_group_count: number of batch groups or grouped filters.

  Returns:
    A tensor representing the output of the convolution.
  """
  precision_config_proto = ""
  if precision_config:
    precision_config_proto = precision_config.SerializeToString()
  needs_v2 = (
      preferred_element_type
      or (lhs.dtype != rhs.dtype)
      or batch_group_count > 1
  )
  if preferred_element_type is None:
    preferred_element_type = np_utils.result_type(lhs.dtype, rhs.dtype)
  if needs_v2 or use_v2:
    return gen_xla_ops.xla_conv_v2(
        lhs,
        rhs,
        window_strides=window_strides,
        padding=padding,
        lhs_dilation=lhs_dilation,
        rhs_dilation=rhs_dilation,
        feature_group_count=feature_group_count,
        batch_group_count=batch_group_count,
        dimension_numbers=dimension_numbers.SerializeToString(),
        precision_config=precision_config_proto,
        preferred_element_type=preferred_element_type,
        name=name,
    )
  return gen_xla_ops.xla_conv(
      lhs,
      rhs,
      window_strides=window_strides,
      padding=padding,
      lhs_dilation=lhs_dilation,
      rhs_dilation=rhs_dilation,
      feature_group_count=feature_group_count,
      dimension_numbers=dimension_numbers.SerializeToString(),
      precision_config=precision_config_proto,
      name=name,
  )


convert_element_type = math_ops.cast


def dot(lhs, rhs, name=None):
  # [FUNCTION DOC]
  # Dot product of lhs and rhs via tensor contraction over one axis (last axis of lhs with the
  # first of rhs). Inputs: lhs, rhs (tensors), name (optional). Returns the contracted tensor.
  return math_ops.tensordot(lhs, rhs, axes=1, name=name)


DotDimensionNumbers = xla_data_pb2.DotDimensionNumbers
PrecisionConfig = xla_data_pb2.PrecisionConfig


def dot_general(
    lhs,
    rhs,
    dimension_numbers,
    precision_config=None,
    preferred_element_type=None,
    name=None,
    use_v2=False,
):
  # [FUNCTION DOC]
  # Wraps the XLA general dot operator. Inputs: lhs and rhs (tensors), dimension_numbers (a dot
  # dimension numbers protocol buffer message, serialized here), precision_config (optional
  # precision message, serialized; otherwise empty string), preferred_element_type (optional
  # result dtype), name (optional), use_v2 (default false).
  # Uses the second op version if a preferred element type was given, the operand dtypes
  # differ, or use_v2 is set; a missing preferred element type is then derived by promotion of
  # the operand dtypes. Otherwise uses the first version. Returns the result tensor.
  precision_config_proto = ""
  if precision_config:
    precision_config_proto = precision_config.SerializeToString()
  needs_v2 = preferred_element_type or (lhs.dtype != rhs.dtype)
  if preferred_element_type is None:
    preferred_element_type = np_utils.result_type(lhs.dtype, rhs.dtype)
  if needs_v2 or use_v2:
    return gen_xla_ops.xla_dot_v2(
        lhs,
        rhs,
        dimension_numbers=dimension_numbers.SerializeToString(),
        precision_config=precision_config_proto,
        preferred_element_type=preferred_element_type,
        name=name,
    )
  return gen_xla_ops.xla_dot(
      lhs,
      rhs,
      dimension_numbers=dimension_numbers.SerializeToString(),
      precision_config=precision_config_proto,
      name=name,
  )


def self_adjoint_eig(a, lower, max_iter, epsilon):
  # [FUNCTION DOC]
  # Eigen-decomposition of self-adjoint (symmetric) matrices. Inputs: a (batch of matrices),
  # lower (boolean: use the lower triangle), max_iter (maximum sweeps), epsilon (convergence
  # tolerance). Returns what the XLA op returns (eigenvalues and eigenvectors).
  return gen_xla_ops.xla_self_adjoint_eig(a, lower, max_iter, epsilon)


def svd(a, max_iter, epsilon, precision_config=None):
  # [FUNCTION DOC]
  # Singular value decomposition via the XLA op. Inputs: a (batch of matrices), max_iter
  # (maximum iterations), epsilon (tolerance), precision_config (optional precision message,
  # serialized; otherwise an empty string). Returns what the XLA op returns (singular values and
  # the left and right singular vectors).
  precision_config_proto = ""
  if precision_config:
    precision_config_proto = precision_config.SerializeToString()
  return gen_xla_ops.xla_svd(a, max_iter, epsilon, precision_config_proto)


dynamic_slice = gen_xla_ops.xla_dynamic_slice
dynamic_update_slice = gen_xla_ops.xla_dynamic_update_slice
einsum = gen_xla_ops.xla_einsum

# TODO(phawkins): generalize tf.pad to support interior padding, and then remove
# the XLA-specific pad operator.
pad = gen_xla_ops.xla_pad


def random_normal(mu, sigma, dims, name=None):
  # [FUNCTION DOC]
  # Samples normally distributed values. Inputs: mu (mean; converted to a tensor, whose dtype is
  # the output dtype), sigma (standard deviation), dims (output shape), name (optional). Returns
  # a tensor of shape dims. Uses TensorFlow's stateful random generator (randomness side effect).
  mu = ops.convert_to_tensor(mu)
  return random_ops.random_normal(
      dims, mean=mu, stddev=sigma, dtype=mu.dtype, name=name
  )


def random_uniform(minval, maxval, dims, name=None):
  # [FUNCTION DOC]
  # Samples uniformly distributed values. Inputs: minval (lower bound; converted to a tensor,
  # whose dtype is the output dtype), maxval (upper bound), dims (output shape), name
  # (optional). Returns a tensor of shape dims. Uses TensorFlow's random generator.
  minval = ops.convert_to_tensor(minval)
  return random_ops.random_uniform(
      dims, minval, maxval, dtype=minval.dtype, name=name
  )


def rng_bit_generator(algorithm, initial_state, shape, dtype):
  # [FUNCTION DOC]
  # Stateless pseudo-random bit generation through the XLA op. Inputs: algorithm (generator
  # algorithm enumeration or integer), initial_state (state tensor), shape (output shape),
  # dtype (output element type). The algorithm is converted to an integer code. Returns a pair
  # of the new state and the generated data.
  """Stateless PRNG bit generator.

  Wraps the XLA RngBitGenerator operator, documented at
    https://www.tensorflow.org/performance/xla/operation_semantics#rngbitgenerator.

  Args:
    algorithm: The PRNG algorithm to use, one of tf.random.Algorithm.{PHILOX,
      THREEFRY, AUTO_SELECT}.
    initial_state: Initial state for the PRNG algorithm. For THREEFRY, it should
      be a u64[2] and for PHILOX a u64[3].
    shape: The output shape of the generated data.
    dtype: The type of the tensor.

  Returns:
    a tuple with a new state and generated data of the given shape.
  """
  alg_int = random_ops_util.convert_alg_to_int(algorithm)
  return gen_xla_ops.xla_rng_bit_generator(
      alg_int, initial_state, shape, dtype=dtype
  )


recv = gen_xla_ops.xla_recv
reduce = gen_xla_ops.xla_reduce
variadic_reduce = gen_xla_ops.xla_variadic_reduce_v2

ops.no_gradient("XlaVariadicReduce")


def reduce_window(
    operand,
    init,
    reducer,
    window_dimensions,
    window_strides=None,
    base_dilations=None,
    window_dilations=None,
    padding=None,
    name=None,
):
  # [FUNCTION DOC]
  # Wraps the XLA windowed reduction. Inputs: operand (input tensor), init (scalar initial
  # value), reducer (computation combining two scalars), window_dimensions (list of window
  # sizes), window_strides, base_dilations, window_dilations (optional lists; default to ones
  # per window dimension), padding (optional list of low/high pairs; default no padding), name
  # (optional). Returns the reduced tensor.
  # KNOWN ISSUE: the docstring does not document base_dilations and window_dilations.
  """Wraps the XLA ReduceWindow operator.

  ReduceWindow is documented at
  https://www.tensorflow.org/performance/xla/operation_semantics#reducewindow .

  Args:
    operand: the input tensor
    init: a scalar tensor representing the initial value for the reduction
    reducer: a reduction function that combines a pair of scalars.
    window_dimensions: shape of the window, as a list of integers
    window_strides: inter-window strides, as a list of integers. Optional; if
      omitted, defaults to strides of 1.
    padding: padding to apply to 'operand'. List of (low, high) pairs of
      integers that specify the padding to apply before and after each
      dimension. Optional; if omitted, defaults to no padding.
    name: the operator name, or None.

  Returns:
    A tensor that represents the output of the reduce_window operator.
  """
  window_strides = window_strides or [1] * len(window_dimensions)
  base_dilations = base_dilations or [1] * len(window_dimensions)
  window_dilations = window_dilations or [1] * len(window_dimensions)
  padding = padding or [(0, 0)] * len(window_dimensions)
  return gen_xla_ops.xla_reduce_window(
      input=operand,
      init_value=init,
      window_dimensions=window_dimensions,
      window_strides=window_strides,
      base_dilations=base_dilations,
      window_dilations=window_dilations,
      padding=padding,
      computation=reducer,
      name=name,
  )


replica_id = gen_xla_ops.xla_replica_id

# Set a static bound for the given input value as a hint to Xla compiler,
# returns the same value.
# Usage:
# def f(t, p):
#   p = xla.set_bound(p, 3) # Tells xla the constraint that p <= 3.
#   return t[:p]            # xla knows the bound of the slice is 3.
set_bound = gen_xla_ops.xla_set_bound

# Make a static dimension into a xla bounded dynamic dimension. The current
# static dimension size will become the bound and the second operand becomes the
# dynamic size of the dimension.
#
# This should mostly be used for testing.
#
# def f():
#   array = tf.convert_to_tensor([[1, 2, 3, 4, 5]])
#   # Tells xla the valid size of the array is 3.
#   dim = 0
#   p = xla_set_dynamic_dimension_size(array, dim, 3)
#   assert(reduce_sum(p) == 6) # xla knows only the first 3 elements are valid.
set_dynamic_dimension_size = gen_xla_ops.xla_set_dynamic_dimension_size

# Inverse of xla_set_dynamic_dimension_size. Make an xla bounded dynamic
# dimension into a static dimension. The bound of the size of dimension
# `dim_index` becomes the static dimension size.
remove_dynamic_dimension_size = gen_xla_ops.xla_remove_dynamic_dimension_size


def reshape(x, new_sizes, dimensions=None, name=None):
  # [FUNCTION DOC]
  # XLA-style reshape. Inputs: x (tensor), new_sizes (target shape), dimensions (optional
  # permutation applied by transposing before reshaping), name (optional). Returns the reshaped
  # tensor.
  if dimensions is not None:
    x = array_ops.transpose(x, dimensions)
  x = array_ops.reshape(x, new_sizes, name=name)
  return x


def select(condition, x, y, name=None):
  # [FUNCTION DOC]
  # Elementwise selection. Inputs: condition (boolean tensor), x (values where true), y (values
  # where false), name (optional). Returns the chosen values.
  return array_ops.where(condition, x, y, name)


select_and_scatter = gen_xla_ops.xla_select_and_scatter
send = gen_xla_ops.xla_send


def slice(x, start_dims, limit_dims, strides):
  # [FUNCTION DOC]
  # XLA-style strided slice. Inputs: x (tensor), start_dims, limit_dims, strides (equal-length
  # sequences, one entry per leading dimension). Builds one slice object per dimension and
  # returns the indexed sub-tensor. This function shadows the builtin slice name in this module.
  spec = [
      _slice(start, limit, stride)
      for (start, limit, stride) in zip(start_dims, limit_dims, strides)
  ]
  return x[tuple(spec)]


sharding = gen_xla_ops.xla_sharding


@ops.RegisterGradient("XlaSharding")
def _sharding_grad(op, grad):
  # [FUNCTION DOC]
  # HELPER: gradient function registered for the sharding op (registration happens at import).
  # Inputs: op (the forward operation), grad (incoming gradient). Applies the same sharding
  # annotation to the gradient, copying the forward sharding attribute and the unspecified
  # dimensions, and additionally sets an internal sharding attribute on the new operation so it
  # is recognized later. Returns a one-element list with the sharded gradient.
  """Gradient for XlaSharding op."""
  sharding_attr = op.get_attr("sharding")
  grad_sharding = gen_xla_ops.xla_sharding(
      grad,
      sharding=sharding_attr,
      unspecified_dims=op.get_attr("unspecified_dims"),
  )
  # pylint: disable=protected-access
  grad_sharding.op._set_attr(
      "_XlaSharding", attr_value_pb2.AttrValue(s=sharding_attr)
  )
  return [grad_sharding]


spmd_full_to_shard_shape = gen_xla_ops.xla_spmd_full_to_shard_shape
spmd_shard_to_full_shape = gen_xla_ops.xla_spmd_shard_to_full_shape


@ops.RegisterGradient("XlaSpmdFullToShardShape")
def _spmd_full_to_shard_shape_grad(op, grad):
  # [FUNCTION DOC]
  # HELPER: gradient function registered for the full-to-shard-shape op (registered at import).
  # Inputs: op (the forward operation), grad (incoming gradient). Applies the inverse op
  # (shard-to-full), reusing the manual sharding, dimension and unspecified-dimension attributes,
  # with the full shape taken from the forward input's static shape. Returns a one-element list.
  # Requires the forward input shape to be fully known.
  s2f = gen_xla_ops.xla_spmd_shard_to_full_shape(
      grad,
      manual_sharding=op.get_attr("manual_sharding"),
      full_shape=op.inputs[0].shape.as_list(),
      dim=op.get_attr("dim"),
      unspecified_dims=op.get_attr("unspecified_dims"),
  )
  return [s2f]


@ops.RegisterGradient("XlaSpmdShardToFullShape")
def _spmd_shard_to_full_shape_grad(op, grad):
  # [FUNCTION DOC]
  # HELPER: gradient function registered for the shard-to-full-shape op (registered at import).
  # Inputs: op (the forward operation), grad (incoming gradient). Applies the inverse op
  # (full-to-shard), reusing the manual sharding, dimension and unspecified-dimension attributes.
  # Returns a one-element list.
  f2s = gen_xla_ops.xla_spmd_full_to_shard_shape(
      grad,
      manual_sharding=op.get_attr("manual_sharding"),
      dim=op.get_attr("dim"),
      unspecified_dims=op.get_attr("unspecified_dims"),
  )
  return [f2s]


sort = gen_xla_ops.xla_sort
key_value_sort = gen_xla_ops.xla_key_value_sort
variadic_sort = gen_xla_ops.xla_variadic_sort
while_loop = gen_xla_ops.xla_while
dequantize = gen_xla_ops.xla_dequantize
custom_call = gen_xla_ops.xla_custom_call


def custom_call_v2(
    call_target_name,
    operands,
    result_specs,
    backend_config=None,
    has_side_effect=None,
    name=None,
):
  # [FUNCTION DOC]
  # Emits an HLO custom call with several outputs. Inputs: call_target_name (name of the
  # registered user function), operands (sequence of tensors), result_specs (sequence of tensor
  # specifications; only dtype and shape are used), backend_config (optional string, empty if
  # null), has_side_effect (optional boolean, false if null), name (optional). Returns a tuple of
  # output tensors.
  """Emits an HLO `CustomCall` operation with multiple outputs.

  See `CustomCall` specification at
    https://tensorflow.org/xla/operation_semantics#customcall,
  and `mhlo.custom_call` specification at
    https://tensorflow.org/mlir/hlo_ops#mhlocustom_call_mlirmhlocustomcallop.

  Args:
    call_target_name: Name of the user function. The function signature must
      conform to version 3 of the API, see
      `API_VERSION_STATUS_RETURNING_UNIFIED`. All operands and results assumed
      to be in the default layout.
    operands: A sequence of tensors with possibly different types.
    result_specs: A sequence of tensor specs for all results.
    backend_config: A string that encodes a metadata for the backend. Empty
      string by default.
    has_side_effect: Indicates whether the custom call has side effects. `False`
      by default.
    name: Optional name of the operation.

  Returns:
    A tuple of output tensors.
  """
  return gen_xla_ops.xla_custom_call_v2(
      operands=operands,
      call_target_name=call_target_name,
      backend_config="" if backend_config is None else backend_config,
      has_side_effect=False if has_side_effect is None else has_side_effect,
      result_dtypes=tuple(spec.dtype for spec in result_specs),
      result_shapes=tuple(spec.shape for spec in result_specs),
      name=name,
  )


# pylint: disable=g-doc-args
# pylint: disable=g-doc-return-or-yield
def call_module(
    args,
    *,
    version=4,
    module,
    Tout,
    Sout,
    platforms=(),
    function_list=(),
    has_token_input_output=False,
    disabled_checks=(),
    use_shardy_partitioner=False,
):
  # [FUNCTION DOC]
  # Wraps the XLA call-module op, which runs a serialized StableHLO module. Inputs: args (list
  # of input tensors), and keyword-only version (default 4), module (serialized module), Tout
  # (output dtypes), Sout (output shapes), platforms (default empty), function_list (default
  # empty), has_token_input_output (default false), disabled_checks (default empty),
  # use_shardy_partitioner (default false). The deprecated dimension-args spec is always empty.
  # Returns the op outputs; if the op returns a bare operation (no outputs, since the op is
  # stateful and would otherwise leak into traced functions), returns an empty tuple instead.
  """See documentation for the XlaCallModule op.

  https://github.com/search?q=repo%3Atensorflow%2Ftensorflow+path%3Axla_ops.cc+xlacallmodule&type=code
  """
  res = gen_xla_ops.xla_call_module(
      args,
      version=version,
      module=module,
      dim_args_spec=(),
      Tout=Tout,
      Sout=Sout,
      platforms=platforms,
      function_list=function_list,
      has_token_input_output=has_token_input_output,
      disabled_checks=disabled_checks,
      use_shardy_partitioner=use_shardy_partitioner,
  )
  # Since XLACallModule op is stateful, zero return function will return the TF
  # op under tf.function. It creates trouble for downstream codes.
  # Here we force it return empty tuple to work around it.
  # TODO(johnqiangzhang): Figure out a better way to handle control dependency.
  if isinstance(res, ops.Operation):
    res = ()
  return res


def call_module_maximum_supported_version():
  # [FUNCTION DOC]
  # Returns the highest version number of the call-module op that this module supports: the
  # fixed integer ten. Takes no inputs.
  """Maximum version of XlaCallModule op supported.

  See versioning details documentation for the XlaCallModule op at:
  https://github.com/search?q=repo%3Atensorflow%2Ftensorflow+path%3Axla_call_module+%22int+kVersionMaximumSupported%22&type=code
  """
  return 10

# pylint: enable=g-doc-args
# pylint: enable=g-doc-return-or-yield


def call_module_disable_check_platform():
  # [FUNCTION DOC]
  # Returns the string naming the platform check, for use in the disabled-checks list of the
  # call-module op. Takes no inputs.
  # For use with xla_call_module.disabled_checks.
  return "platform"


def gather(
    operand,
    start_indices,
    dimension_numbers,
    slice_sizes,
    indices_are_sorted=False,
    name=None,
):
  # [FUNCTION DOC]
  # Wraps the XLA gather operator. Inputs: operand (tensor to gather from), start_indices
  # (indices tensor), dimension_numbers (gather dimension numbers protocol buffer message,
  # serialized here), slice_sizes (sizes of the slices), indices_are_sorted (default false hint),
  # name (optional). Returns the gathered tensor.
  return gen_xla_ops.xla_gather(
      operand,
      start_indices,
      slice_sizes=slice_sizes,
      dimension_numbers=dimension_numbers.SerializeToString(),
      indices_are_sorted=indices_are_sorted,
      name=name,
  )


def scatter(
    operand,
    scatter_indices,
    updates,
    update_computation,
    dimension_numbers,
    indices_are_sorted=False,
    name=None,
):
  # [FUNCTION DOC]
  # Wraps the XLA scatter operator. Inputs: operand (tensor to update), scatter_indices
  # (indices tensor), updates (values to scatter), update_computation (function combining old
  # and new values), dimension_numbers (scatter dimension numbers protocol buffer message,
  # serialized here), indices_are_sorted (default false hint), name (optional). Returns the
  # updated tensor.
  return gen_xla_ops.xla_scatter(
      operand,
      scatter_indices,
      updates,
      update_computation=update_computation,
      dimension_numbers=dimension_numbers.SerializeToString(),
      indices_are_sorted=indices_are_sorted,
      name=name,
  )


def optimization_barrier(*args):
  # [FUNCTION DOC]
  # Wraps the XLA optimization barrier, which prevents the compiler from moving computation
  # across it. Takes any number of tensors and returns them as outputs of the barrier op.
  return gen_xla_ops.xla_optimization_barrier(args)


def reduce_precision(operand, exponent_bits, mantissa_bits):
  # [FUNCTION DOC]
  # Wraps the XLA reduce-precision operator, which rounds floating-point values to a format with
  # fewer exponent and mantissa bits while keeping the original dtype. Inputs: operand (float
  # tensor), exponent_bits and mantissa_bits (integers). Returns the rounded tensor.
  return gen_xla_ops.xla_reduce_precision(operand, exponent_bits, mantissa_bits)
