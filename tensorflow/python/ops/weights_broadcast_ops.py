# [MODULE DOC]
# Weight broadcasting operations used by the loss and metric libraries. Offers
# assert_broadcastable (static or dynamic check that weights are a scalar or have the same rank
# as the values with each dimension equal to 1 or to the values dimension) and
# broadcast_weights (expands weights to the values shape, registered as an internal exported
# API). Two private helpers implement the dynamic checks. Import-time side effect: registers
# the public function with the export decorator.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Weight broadcasting operations.

In `tf.losses` and `tf.metrics`, we support limited weight broadcasting. This
file includes operations for those broadcasting rules.
"""

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sets
from tensorflow.python.util.tf_export import tf_export


def _has_valid_dims(weights_shape, values_shape):
  # [FUNCTION DOC]
  # HELPER: dynamic check used by assert_broadcastable via the nonscalar shape check.
  # KNOWN ISSUE: the name scope is called has_invalid_dims although the function returns whether
  # dims are valid.
  # Takes 1-D integer shape tensors. Builds, for each values dimension, the allowed set
  # {values dim, 1}, computes the set difference of the weights dims against it, and returns a
  # boolean scalar tensor that is true when no weights dim is outside the allowed set. Adds
  # graph ops only.
  with ops.name_scope(
      None, "has_invalid_dims", (weights_shape, values_shape)) as scope:
    values_shape_2d = array_ops.expand_dims(values_shape, -1)
    valid_dims = array_ops.concat(
        (values_shape_2d, array_ops.ones_like(values_shape_2d)), axis=1)
    weights_shape_2d = array_ops.expand_dims(weights_shape, -1)
    invalid_dims = sets.set_difference(weights_shape_2d, valid_dims)
    num_invalid_dims = array_ops.size(
        invalid_dims.values, name="num_invalid_dims")
    return math_ops.equal(0, num_invalid_dims, name=scope)


def _has_valid_nonscalar_shape(
    weights_rank, weights_shape, values_rank, values_shape):
  # [FUNCTION DOC]
  # HELPER: dynamic check used by assert_broadcastable for non-scalar weights.
  # Takes rank scalars and shape vectors of weights and values. Returns a boolean scalar tensor:
  # false when the ranks differ, otherwise the result of the per-dimension validity check
  # (evaluated lazily through a conditional). Adds graph ops only.
  with ops.name_scope(
      None, "has_valid_nonscalar_shape",
      (weights_rank, weights_shape, values_rank, values_shape)) as scope:
    is_same_rank = math_ops.equal(
        values_rank, weights_rank, name="is_same_rank")
    return cond.cond(
        is_same_rank,
        lambda: _has_valid_dims(weights_shape, values_shape),
        lambda: is_same_rank,
        name=scope)


_ASSERT_BROADCASTABLE_ERROR_PREFIX = "weights can not be broadcast to values."


def assert_broadcastable(weights, values):
  # [FUNCTION DOC]
  # Converts weights and values to tensors, then tries static checks: scalar weights (rank 0)
  # succeed with a no-op; differing static ranks, or a static dimension of weights that is
  # neither 1 nor the values dimension, raise a value error with a detailed message. If static
  # shapes are fully known and valid it returns a no-op. Otherwise it builds a dynamic check
  # and returns an assert operation that fails at run time with an invalid-argument error when
  # the weights shape is invalid. Inputs: weights and values, tensor-like. Output: an operation
  # to use as a control dependency. Adds named scopes and ops to the graph.
  """Asserts `weights` can be broadcast to `values`.

  In `tf.losses` and `tf.metrics`, we support limited weight broadcasting. We
  let weights be either scalar, or the same rank as the target values, with each
  dimension either 1, or the same as the corresponding values dimension.

  Args:
    weights: `Tensor` of weights.
    values: `Tensor` of values to which weights are applied.

  Returns:
    `Operation` raising `InvalidArgumentError` if `weights` has incorrect shape.
    `no_op` if static checks determine `weights` has correct shape.

  Raises:
    ValueError:  If static checks determine `weights` has incorrect shape.
  """
  with ops.name_scope(None, "assert_broadcastable", (weights, values)) as scope:
    with ops.name_scope(None, "weights", (weights,)) as weights_scope:
      weights = ops.convert_to_tensor(weights, name=weights_scope)
      weights_shape = array_ops.shape(weights, name="shape")
      weights_rank = array_ops.rank(weights, name="rank")
    weights_rank_static = tensor_util.constant_value(weights_rank)

    with ops.name_scope(None, "values", (values,)) as values_scope:
      values = ops.convert_to_tensor(values, name=values_scope)
      values_shape = array_ops.shape(values, name="shape")
      values_rank = array_ops.rank(values, name="rank")
    values_rank_static = tensor_util.constant_value(values_rank)

    # Try static checks.
    if weights_rank_static is not None and values_rank_static is not None:
      if weights_rank_static == 0:
        return control_flow_ops.no_op(name="static_scalar_check_success")
      if weights_rank_static != values_rank_static:
        raise ValueError(
            f"{_ASSERT_BROADCASTABLE_ERROR_PREFIX} values.rank="
            f"{values_rank_static}. weights.rank={weights_rank_static}. "
            f"values.shape={values.shape}. weights.shape={weights.shape}. "
            f"Received weights={weights}, values={values}")
      weights_shape_static = tensor_util.constant_value(weights_shape)
      values_shape_static = tensor_util.constant_value(values_shape)
      if weights_shape_static is not None and values_shape_static is not None:
        # Sanity check, this should always be true since we checked rank above.
        ndims = len(values_shape_static)
        assert ndims == len(weights_shape_static)

        for i in range(ndims):
          if weights_shape_static[i] not in (1, values_shape_static[i]):
            raise ValueError(
                f"{_ASSERT_BROADCASTABLE_ERROR_PREFIX} Mismatch at dim {i}. "
                f"values.shape={values_shape_static}, weights.shape="
                f"{weights_shape_static}. Received weights={weights}, "
                f"values={values}")
        return control_flow_ops.no_op(name="static_dims_check_success")

    # Dynamic checks.
    is_scalar = math_ops.equal(0, weights_rank, name="is_scalar")
    data = (
        _ASSERT_BROADCASTABLE_ERROR_PREFIX,
        "weights.shape=", weights.name, weights_shape,
        "values.shape=", values.name, values_shape,
        "is_scalar=", is_scalar,
    )
    is_valid_shape = cond.cond(
        is_scalar,
        lambda: is_scalar,
        lambda: _has_valid_nonscalar_shape(  # pylint: disable=g-long-lambda
            weights_rank, weights_shape, values_rank, values_shape),
        name="is_valid_shape")
    return control_flow_assert.Assert(is_valid_shape, data, name=scope)


@tf_export("__internal__.ops.broadcast_weights", v1=[])
def broadcast_weights(weights, values):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring mentions a function named with a leading underscore for
  # broadcasting weights; the actual function is broadcast_weights.
  # Converts values to a tensor and weights to a tensor of the values base dtype. If both static
  # shapes are fully defined and compatible, returns weights unchanged. Inside an XLA context
  # it skips the assertion and returns weights multiplied by ones shaped like values. Otherwise
  # it multiplies weights by ones_like(values) under a control dependency on
  # assert_broadcastable, so invalid shapes raise (value error statically, invalid-argument
  # error at run time). Returns the broadcast weights tensor.
  """Broadcast `weights` to the same shape as `values`.

  This returns a version of `weights` following the same broadcast rules as
  `mul(weights, values)`, but limited to the weights shapes allowed by
  `assert_broadcastable`. When computing a weighted average, use this function
  to broadcast `weights` before summing them; e.g.,
  `reduce_sum(w * v) / reduce_sum(_broadcast_weights(w, v))`.

  Args:
    weights: `Tensor` whose shape is broadcastable to `values` according to the
      rules of `assert_broadcastable`.
    values: `Tensor` of any shape.

  Returns:
    `weights` broadcast to `values` shape according to the rules of
      `assert_broadcastable`.
  """
  with ops.name_scope(None, "broadcast_weights", (weights, values)) as scope:
    values = ops.convert_to_tensor(values, name="values")
    weights = ops.convert_to_tensor(
        weights, dtype=values.dtype.base_dtype, name="weights")

    # Try static check for exact match.
    weights_shape = weights.get_shape()
    values_shape = values.get_shape()
    if (weights_shape.is_fully_defined() and
        values_shape.is_fully_defined() and
        weights_shape.is_compatible_with(values_shape)):
      return weights

    # Skip the assert_broadcastable on TPU/GPU because asserts are not
    # supported so it only causes unnecessary ops. Also skip it because it uses
    # a DenseToDenseSetOperation op that is incompatible with the TPU/GPU when
    # the shape(s) are dynamic.
    if control_flow_ops.get_enclosing_xla_context() is not None:
      return math_ops.multiply(
          weights, array_ops.ones_like(values), name=scope)
    with ops.control_dependencies((assert_broadcastable(weights, values),)):
      return math_ops.multiply(
          weights, array_ops.ones_like(values), name=scope)
