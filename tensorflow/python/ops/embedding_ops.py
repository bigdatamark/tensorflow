# [MODULE DOC]
# Operations for embeddings. Provides dense embedding lookup (with partitioned parameters under
# a modulo or division strategy, optional norm clipping and optional per-partition transform),
# sparse embedding lookup with weights and a combiner (mean, square-root-n or sum), and "safe"
# sparse lookup that prunes invalid ids and fills empty rows. Public functions are exported to
# the neural-network namespace in both the v1 and v2 APIs and support dispatch. Importing it
# also imports the data-flow gradient definitions so that gradients are registered.
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
"""Operations for embeddings."""

from tensorflow.python.compat import compat
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import clip_ops
# Imports gradient definitions.
from tensorflow.python.ops import data_flow_grad  # pylint: disable=unused-import
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables
from tensorflow.python.types import core
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export


def _clip(params, ids, max_norm):
  # [FUNCTION DOC]
  # HELPER: norm clipping step for the embedding lookup functions.
  # Returns params unchanged when max_norm is nothing; otherwise clips each embedding vector so
  # its l2-norm is at most max_norm. Inputs: params, gathered embeddings; ids, the ids that were
  # used for the gather (only their rank matters); max_norm, optional float limit. The clipping
  # axes are all dimensions of params beyond the rank of ids; they are Python values when both
  # ranks are statically known, otherwise a computed range tensor. Output: tensor with the same
  # type as params. No side effects other than adding ops.
  """Helper function for _embedding_lookup_and_transform.

  This function optionally clips embeddings to an l2-norm of max_norm.

  Args:
    params: A `Tensor` of embeddings retrieved by `gather`.
    ids: The `ids` argument that was passed to `gather`.
    max_norm: If not `None`, each embedding is clipped if its l2-norm is larger
      than this value.

  Returns:
    A `Tensor` with the same type as `params`.
  """

  def _rank(x):
    # [FUNCTION DOC]
    # HELPER: rank lookup for the clipping helper.
    # Converts x to a tensor and returns a pair: the rank and a flag saying whether the rank is
    # statically known. If the static rank is known and nonzero, returns it with true; otherwise
    # (unknown or zero rank) returns a dynamic rank tensor with false.
    """Helper function to retrieve the rank of a tensor.

    Args:
      x: Something convertible to `Tensor`.

    Returns:
      Either a pair `(rank, True)` where `rank` is an integer or a pair
      `(rank, False)` where `rank` is an integer `Tensor`. In either case,
      `rank` is the rank of `x`.
    """
    rank = ops.convert_to_tensor(x).get_shape().ndims
    if rank:
      return rank, True
    else:
      return array_ops.rank(x), False

  if max_norm is None:
    return params
  ids_rank, ids_static = _rank(ids)
  params_rank, params_static = _rank(params)
  return clip_ops.clip_by_norm(
      params,
      max_norm,
      axes=(list(range(ids_rank, params_rank)) if ids_static and params_static
            else math_ops.range(ids_rank, params_rank)))


def _colocate_with(param):
  # [FUNCTION DOC]
  # HELPER: device-colocation context for embedding lookups.
  # Returns a context manager that places ops on the same device as param. Inside a function
  # graph with a variable-like param (one having a handle), it colocates with the captured
  # handle tensor so no hard-coded device string is baked in (which would break serving);
  # otherwise it colocates directly with param.
  if ops.inside_function() and hasattr(param, "handle"):
    # The `ops.colocate_with` will hard-code a device string if `param.device`
    # is known, which will then break serving. We capture it here so that it
    # produces a tensor without a device.
    return ops.colocate_with(ops.get_default_graph().capture(param.handle))
  else:
    return ops.colocate_with(param)


def _embedding_lookup_and_transform(params,
                                    ids,
                                    partition_strategy="mod",
                                    name=None,
                                    max_norm=None,
                                    transform_fn=None):
  # [FUNCTION DOC]
  # HELPER: shared core of the dense embedding lookup and sampled-logits computation.
  # Looks up ids in params (a tensor, list of tensors, or partitioned variable), optionally
  # clipping norms and applying transform_fn per retrieved block. Inputs: params; ids (integer
  # tensor-like); partition_strategy ("mod" default, or "div"); name; max_norm; transform_fn
  # (callable mapping a tensor to a tensor with the same first dimension, or nothing).
  # Single-partition case (and no transform or 1-D ids): gather, clip, transform, then identity
  # to drop colocation. Otherwise ids are flattened, each id assigned to a partition and a local
  # id (modulo or division with remainder handling, using the summed first dimensions), split with
  # dynamic partition, gathered per partition under that params colocation with device cleared,
  # clipped/transformed, stitched back together and reshaped to the ids shape plus the element
  # shape (static where possible), with the static shape set explicitly.
  # Returns a tensor shaped like ids followed by the element shape. Raises a value error if
  # params is nothing, an empty list or tuple, or if the partition strategy is unrecognized.
  # Adds graph ops; no other side effects.
  """Helper function for embedding_lookup and _compute_sampled_logits.

  This function is a generalization of embedding_lookup that optionally
  applies a caller-specified transformation to each embedding. This is
  done through the `transform_fn` argument. If provided, the function is
  applied to each partitioned tensor of retrieved embeddings, colocated
  with the embeddings. This function will be called with a single `Tensor`
  argument of the same type as the `params` tensor and should return a
  `Tensor`. The shape of the argument will be the same as `params` except
  for the size of the first dimension. The first dimension of the result's
  shape must be the same size as the argument's.

  Args:
    params: See embedding_lookup.
    ids: See embedding_lookup.
    partition_strategy: See embedding_lookup.
    name: See embedding_lookup.
    max_norm: See embedding_lookup.
    transform_fn: An optional function to apply to each retrieved embedding. If
      max_norm is provided, transform_fn is applied to the norm-limited
      embeddings.

  Returns:
    See embedding_lookup for details.
  Raises:
    ValueError: If `params` is empty.
  """
  if params is None:
    raise ValueError("params must be specified")
  if isinstance(params, (list, tuple)) and not params:
    raise ValueError("Length of params is currently 0. "
                     "Need at least one param.")
  if isinstance(params, variables.PartitionedVariable):
    params = list(params)  # Iterate to get the underlying Variables.
  if not isinstance(params, list):
    params = [params]

  with ops.name_scope(name, "embedding_lookup", params + [ids]) as name:
    np = len(params)  # Number of partitions
    # Preserve the resource variable status to avoid accidental dense reads.
    if not any(
        isinstance(p, resource_variable_ops.BaseResourceVariable)
        for p in params):
      params = indexed_slices.convert_n_to_tensor_or_indexed_slices(
          params, name="params")
    ids = ops.convert_to_tensor(ids, name="ids")
    if np == 1 and (not transform_fn or ids.get_shape().ndims == 1):
      with _colocate_with(params[0]):
        result = _clip(
            array_ops.gather(params[0], ids, name=name), ids, max_norm)
        if transform_fn:
          result = transform_fn(result)
      # Make sure the final result does not have colocation constraints on the
      # params. Similar to the case np > 1 where parallel_dynamic_stitch is
      # outside the scope of all with _colocate_with(params[p]).
      return array_ops.identity(result)
    else:
      # Flatten the ids. There are two cases where we need to do this.
      # - There is more than one params tensor.
      # - There is a transform_fn and ids is not statically known to be 1-D.
      #   We must flatten in this case because transform_fn expects a flat
      #   tensor of embeddings.
      flat_ids = array_ops.reshape(ids, [-1])
      original_indices = math_ops.range(
          array_ops.size(flat_ids, out_type=dtypes.int32)
      )

      # Create p_assignments and set new_ids depending on the strategy.
      if partition_strategy == "mod":
        p_assignments = flat_ids % np
        new_ids = flat_ids // np
      elif partition_strategy == "div":
        # Compute num_total_ids as the sum of dim-0 of params, then assign to
        # partitions based on a constant number of ids per partition. Optimize
        # if we already know the full shape statically.
        dim_0_size = tensor_shape.Dimension(
            tensor_shape.dimension_value(params[0].get_shape()[0]))
        for p in range(1, np):
          dim_0_size += tensor_shape.Dimension(
              tensor_shape.dimension_value(params[p].get_shape()[0]))
        if dim_0_size.value:
          num_total_ids = constant_op.constant(dim_0_size.value, flat_ids.dtype)
        else:
          dim_0_sizes = []
          for p in range(np):
            param_p_dim = tensor_shape.dimension_value(params[p].get_shape()[0])
            if param_p_dim is not None:
              dim_0_sizes.append(param_p_dim)
            else:
              with _colocate_with(params[p]):
                dim_0_sizes.append(array_ops.shape(params[p])[0])
          num_total_ids = math_ops.reduce_sum(
              math_ops.cast(array_ops_stack.stack(dim_0_sizes), flat_ids.dtype))
        ids_per_partition = num_total_ids // np
        extras = num_total_ids % np

        p_assignments = math_ops.maximum(flat_ids // (ids_per_partition + 1),
                                         (flat_ids - extras) //
                                         ids_per_partition)

        # Emulate a conditional using a boolean indicator tensor
        new_ids = array_ops.where(p_assignments < extras,
                                  flat_ids % (ids_per_partition + 1),
                                  (flat_ids - extras) % ids_per_partition)
      else:
        raise ValueError(
            f"Unrecognized partition strategy: {partition_strategy}."
            "Must be one of either `mod` or `div`.")

      # Cast partition assignments to int32 for use in dynamic_partition.
      # There really should not be more than 2^32 partitions.
      p_assignments = math_ops.cast(p_assignments, dtypes.int32)
      # Partition list of ids based on assignments into np separate lists
      gather_ids = data_flow_ops.dynamic_partition(new_ids, p_assignments, np)
      # Similarly, partition the original indices.
      pindices = data_flow_ops.dynamic_partition(original_indices,
                                                 p_assignments, np)
      # Do np separate lookups, finding embeddings for plist[p] in params[p]
      partitioned_result = []
      for p in range(np):
        pids = gather_ids[p]
        with ops.device_v2(None):
          with _colocate_with(params[p]):
            result = array_ops.gather(params[p], pids)
            if transform_fn:
              # If transform_fn is provided, the clip_by_norm precedes
              # the transform and hence must be co-located. See below
              # for the counterpart if transform_fn is not provided.
              result = transform_fn(_clip(result, pids, max_norm))
        partitioned_result.append(result)
      # Stitch these back together
      ret = data_flow_ops.parallel_dynamic_stitch(
          pindices, partitioned_result, name=name)

      # Determine the static element shape.
      if transform_fn is None:
        element_shape_s = params[0].get_shape()[1:]
        for p in params[1:]:
          element_shape_s = element_shape_s.merge_with(p.get_shape()[1:])
      else:
        element_shape_s = ret.get_shape()[1:]

      # Compute the dynamic element shape.
      if element_shape_s.is_fully_defined():
        element_shape_d = element_shape_s
      elif transform_fn is None:
        # It's important that we compute params[0].shape on the right device
        # to avoid data motion.
        with _colocate_with(params[0]):
          params_shape = array_ops.shape(params[0])
        element_shape_d = params_shape[1:]
      else:
        element_shape_d = array_ops.shape(ret)[1:]

      # Reshape to reverse the flattening of ids.
      ret = array_ops.reshape(
          ret, array_ops.concat([array_ops.shape(ids), element_shape_d], 0))

      # Normally the reshape is sufficient, but setting shape explicitly
      # teaches shape inference that params[1:].get_shape() matters
      # (in the case that transform_fn is None).
      ret.set_shape(ids.get_shape().concatenate(element_shape_s))
      if not transform_fn:
        # If transform_fn was provided, the clip_by_norm was done above.
        ret = _clip(ret, ids, max_norm)
      return ret


@tf_export(v1=["nn.embedding_lookup"])
@dispatch.add_dispatch_support
def embedding_lookup(
    params,
    ids,
    partition_strategy="mod",
    name=None,
    validate_indices=True,  # pylint: disable=unused-argument
    max_norm=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the function body contains a second, unattached string literal after the real
  # docstring (describing CPU versus GPU out-of-bounds behavior and a usage example). It is a
  # no-op expression, not part of the docstring, so that text never appears in generated docs.
  # KNOWN ISSUE: the validate_indices argument is accepted but ignored (deprecated).
  # KNOWN ISSUE: the docstring mentions ragged ids, but the code just forwards to the shared
  # lookup helper without any ragged-specific handling.
  # Version 1 public embedding lookup. Looks up rows of params for each id in ids and returns a
  # tensor shaped like ids followed by the trailing dimensions of params. Inputs: params (tensor,
  # list of sharded tensors, or partitioned variable); ids (integer tensor); partition_strategy
  # ("mod" default, or "div"); name; validate_indices (ignored); max_norm (optional clipping
  # limit on each embedding's l2-norm). Raises a value error if params is empty.
  # Delegates to the shared helper with no transform. Adds graph ops.
  """Looks up embeddings for the given `ids` from a list of tensors.

  This function is used to perform parallel lookups on the list of tensors in
  `params`.  It is a generalization of `tf.gather`, where `params` is
  interpreted as a partitioning of a large embedding tensor.  `params` may be
  a `PartitionedVariable` as returned by using `tf.compat.v1.get_variable()`
  with a partitioner.

  If `len(params) > 1`, each element `id` of `ids` is partitioned between
  the elements of `params` according to the `partition_strategy`.
  In all strategies, if the id space does not evenly divide the number of
  partitions, each of the first `(max_id + 1) % len(params)` partitions will
  be assigned one more id.

  If `partition_strategy` is `"mod"`, we assign each id to partition
  `p = id % len(params)`. For instance,
  13 ids are split across 5 partitions as:
  `[[0, 5, 10], [1, 6, 11], [2, 7, 12], [3, 8], [4, 9]]`

  If `partition_strategy` is `"div"`, we assign ids to partitions in a
  contiguous manner. In this case, 13 ids are split across 5 partitions as:
  `[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11, 12]]`

  If the input ids are ragged tensors, partition variables are not supported and
  the partition strategy and the max_norm are ignored.
  The results of the lookup are concatenated into a dense
  tensor. The returned tensor has shape `shape(ids) + shape(params)[1:]`.

  Args:
    params: A single tensor representing the complete embedding tensor, or a
      list of P tensors all of same shape except for the first dimension,
      representing sharded embedding tensors.  Alternatively, a
      `PartitionedVariable`, created by partitioning along dimension 0. Each
      element must be appropriately sized for the given `partition_strategy`.
    ids: A `Tensor` or a 'RaggedTensor' with type `int32` or `int64` containing
      the ids to be looked up in `params`.
      Caution: Out-of-bounds indices will result in undefined behavior, which
        will differ between devices and backends.
    partition_strategy: A string specifying the partitioning strategy, relevant
      if `len(params) > 1`. Currently `"div"` and `"mod"` are supported. Default
      is `"mod"`.
    name: A name for the operation (optional).
    validate_indices: DEPRECATED. If this operation is assigned to CPU, values
      in `indices` are always validated to be within range.  If assigned to GPU,
      out-of-bound indices result in safe but unspecified behavior, which may
      include raising an error.
    max_norm: If not `None`, each embedding is clipped if its l2-norm is larger
      than this value.

  Returns:
    A `Tensor` or a 'RaggedTensor', depending on the input, with the same type
    as the tensors in `params`.

  Raises:
    ValueError: If `params` is empty.
  """

  """
    **Behavior Difference between CPU and GPU**

    Please note that when using `tf.nn.embedding_lookup` on a GPU, if an out-of-bound 
    index is encountered, a value of 0 will be stored in the corresponding output value. 
    On the other hand, when using `tf.nn.embedding_lookup` on a CPU, an error will be 
    returned if an out-of-bound index is found.

    This behavior difference can impact the results of your computation, especially when 
    dealing with indices that may go beyond the bounds of the tensor. 
    Make sure to be mindful of this distinction when using the `tf.nn.embedding_lookup` 
    function in your computations.

    **Usage Example**

    Here's an example demonstrating how to use `tf.nn.embedding_lookup`:

    ```python
    import tensorflow as tf

    # Example embedding matrix and indices
    embedding_matrix = tf.constant([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    indices = tf.constant([1, 0, 2])

    # Perform embedding lookup
    embeddings = tf.nn.embedding_lookup(embedding_matrix, indices)

    # Print the result
    print("Embeddings:")
    print(embeddings.numpy())
    ```
    """

  return _embedding_lookup_and_transform(
      params=params,
      ids=ids,
      partition_strategy=partition_strategy,
      name=name,
      max_norm=max_norm,
      transform_fn=None)


@tf_export("nn.embedding_lookup", v1=[])
@dispatch.add_dispatch_support
def embedding_lookup_v2(params, ids, max_norm=None, name=None):
  # [FUNCTION DOC]
  # Version 2 public embedding lookup. Same as the version 1 function but fixes the partition
  # strategy to division and drops the deprecated arguments. Inputs: params (tensor or list of
  # tensors sharded under division), ids (integer tensor), max_norm (optional l2-norm clipping
  # limit), name. Returns a tensor shaped like ids followed by params trailing dimensions.
  # Raises a value error if params is empty (via the delegate). Adds graph ops.
  """Looks up embeddings for the given `ids` from a list of tensors.

  This function is used to perform parallel lookups on the list of tensors in
  `params`.  It is a generalization of `tf.gather`, where `params` is
  interpreted as a partitioning of a large embedding tensor.

  If `len(params) > 1`, each element `id` of `ids` is partitioned between the
  elements of `params` according to the "div" partition strategy, which means we
  assign ids to partitions in a contiguous manner. For instance, 13 ids are
  split across 5 partitions as:
  `[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11, 12]]`.

  If the id space does not evenly divide the number of partitions, each of the
  first `(max_id + 1) % len(params)` partitions will be assigned one more id.

  The results of the lookup are concatenated into a dense
  tensor. The returned tensor has shape `shape(ids) + shape(params)[1:]`.

  Args:
    params: A single tensor representing the complete embedding tensor, or a
      list of tensors all of same shape except for the first dimension,
      representing sharded embedding tensors following "div" partition strategy.
    ids: A `Tensor` with type `int32` or `int64` containing the ids to be looked
      up in `params`.
    max_norm: If not `None`, each embedding is clipped if its l2-norm is larger
      than this value.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with the same type as the tensors in `params`.

    For instance, if `params` is a 5x2 matrix:

    ```python
    [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    ```

    or a list of matrices:

    ```python
    params[0]: [[1, 2], [3, 4]]
    params[1]: [[5, 6], [7, 8]]
    params[2]: [[9, 10]]
    ```

    and `ids` is:

    ```python
    [0, 3, 4]
    ```

    The output will be a 3x2 matrix:

    ```python
    [[1, 2], [7, 8], [9, 10]]
    ```

  Raises:
    ValueError: If `params` is empty.
  """
  return embedding_lookup(params, ids, "div", name, max_norm=max_norm)


@tf_export(v1=["nn.embedding_lookup_sparse"])
@dispatch.add_dispatch_support
def embedding_lookup_sparse(
    params,
    sp_ids,
    sp_weights,
    partition_strategy="mod",
    name=None,
    combiner=None,
    max_norm=None,
    allow_fast_lookup=False,
):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says ragged tensors are accepted and that a type error is raised
  # only for non-sparse/ragged input, but the code accepts only sparse tensors for both sp_ids
  # and sp_weights and raises a type error for anything else (including ragged tensors).
  # KNOWN ISSUE: the docstring names the weights argument sparse_weights while the real
  # parameter is sp_weights.
  # Version 1 sparse embedding lookup: gathers embeddings for the ids in a rank-2 sparse tensor,
  # multiplies by optional weights and combines per row. Inputs: params (tensor, list or
  # partitioned variable); sp_ids (sparse tensor of integer ids); sp_weights (sparse tensor of
  # the same shape, or nothing for unit weights); partition_strategy ("mod" default or "div");
  # name; combiner (mean by default, or sqrtn or sum); max_norm (optional clipping limit);
  # allow_fast_lookup (boolean, permits the simplified gather path). Output: dense tensor with one
  # row per row of sp_ids. Raises a value error for an unsupported combiner and a type error for
  # wrong sp_ids or sp_weights types. Checks that values, indices and dense shape of the weights
  # are shape-compatible with the ids. Extracts row indices as segment ids and delegates to the
  # shared implementation function. Adds graph ops.
  """Looks up embeddings for the given ids and weights from a list of tensors.

  This op assumes that there is at least one id for each row in the dense tensor
  represented by sp_ids (i.e. there are no rows with empty features), and that
  all the indices of sp_ids are in canonical row-major order.

  `sp_ids` and `sp_weights` (if not None) are `SparseTensor`s or `RaggedTensor`s
  with rank of 2. For `SpareTensor`s with left-aligned non-zero entries which
  can be described as `RaggedTensor`s, use of `RaggedTensor`s can yield higher
  performance.

  It also assumes that all id values lie in the range [0, p0), where p0
  is the sum of the size of params along dimension 0.

  Args:
    params: A single tensor representing the complete embedding tensor, or a
      list tensors all of same shape except for the first dimension,
      representing sharded embedding tensors. Alternatively, a
      `PartitionedVariable`, created by partitioning along dimension 0. Each
      element must be appropriately sized for the given `partition_strategy`.
    sp_ids: N x M `SparseTensor` of int64 ids where N is typically batch size
      and M is arbitrary or a `RaggedTensor` with rank 2.
    sparse_weights: `SparseTensor` or `RaggedTensor` of same type and shape as
      `sparse_ids`, containing float / double weights corresponding to
      `sparse_ids`, or `None` if all weights are assumed to be 1.0.
    partition_strategy: A string specifying the partitioning strategy, relevant
      if `len(params) > 1`. Currently `"div"` and `"mod"` are supported. Default
      is `"mod"`. See `tf.nn.embedding_lookup` for more details.
    name: Optional name for the op.
    combiner: A string specifying the reduction op. Currently "mean", "sqrtn"
      and "sum" are supported. "sum" computes the weighted sum of the embedding
      results for each row. "mean" is the weighted sum divided by the total
      weight. "sqrtn" is the weighted sum divided by the square root of the sum
      of the squares of the weights. Defaults to `mean`.
    max_norm: If not `None`, each embedding is clipped if its l2-norm is larger
      than this value, before combining.
    allow_fast_lookup: An optional boolean specifying whether to allow
      simplified embedding lookups when `params` is a single tensor and
      `max_norm` is `None`. Setting this flag to `True` during training can
      cause the use of dense gradients with increased memory footprint.

  Returns:
    A dense tensor representing the combined embeddings for the
    sparse ids. For each row in the dense tensor represented by `sp_ids`, the op
    looks up the embeddings for all ids in that row, multiplies them by the
    corresponding weight, and combines these embeddings as specified.

    In other words, if

      `shape(combined params) = [p0, p1, ..., pm]`

    and

      `shape(sp_ids) = shape(sp_weights) = [d0, d1]`

    then

      `shape(output) = [d0, p1, ..., pm]`.

    For instance, if params is a 10x20 matrix, and sp_ids / sp_weights are

      ```python
      [0, 0]: id 1, weight 2.0
      [0, 1]: id 3, weight 0.5
      [1, 0]: id 0, weight 1.0
      [2, 3]: id 1, weight 3.0
      ```

    with `combiner`="mean", then the output will be a 3x20 matrix where

      ```python
      output[0, :] = (params[1, :] * 2.0 + params[3, :] * 0.5) / (2.0 + 0.5)
      output[1, :] = (params[0, :] * 1.0) / 1.0
      output[2, :] = (params[1, :] * 3.0) / 3.0
      ```

  Raises:
    TypeError: If `sp_ids` is not a `SparseTensor` or `RaggedTensor`, or if
    `sp_weights` is neither `None` nor of the same type as `sp_ids`.
    ValueError: If `combiner` is not one of {"mean", "sqrtn", "sum"}.
  """
  if combiner is None:
    combiner = "mean"
  if combiner not in ("mean", "sqrtn", "sum"):
    raise ValueError(
        f"combiner must be one of 'mean', 'sqrtn' or 'sum', got {combiner}")
  if isinstance(params, variables.PartitionedVariable):
    params = list(params)  # Iterate to get the underlying Variables.
  if not isinstance(params, list):
    params = [params]
  if not isinstance(sp_ids, sparse_tensor.SparseTensor):
    raise TypeError(f"sp_ids must be SparseTensor, got {type(sp_ids)}")
  ignore_weights = sp_weights is None
  if not ignore_weights:
    if not isinstance(sp_weights, sparse_tensor.SparseTensor):
      raise TypeError(f"sp_weights must be either None or SparseTensor,"
                      f"got {type(sp_weights)}")
    sp_ids.values.get_shape().assert_is_compatible_with(
        sp_weights.values.get_shape())
    sp_ids.indices.get_shape().assert_is_compatible_with(
        sp_weights.indices.get_shape())
    sp_ids.dense_shape.get_shape().assert_is_compatible_with(
        sp_weights.dense_shape.get_shape())
    # TODO(yleon): Add enhanced node assertions to verify that sp_ids and
    # sp_weights have equal indices and shapes.

  with ops.name_scope(name, "embedding_lookup_sparse",
                      params + [sp_ids]) as name:

    segment_ids = sp_ids.indices[:, 0]
    ids = sp_ids.values

    return embedding_lookup_sparse_impl(
        params,
        segment_ids,
        sp_weights,
        ids,
        combiner,
        ignore_weights,
        max_norm,
        allow_fast_lookup,
        partition_strategy,
        name,
    )


@tf_export("nn.embedding_lookup_sparse", v1=[])
@dispatch.add_dispatch_support
def embedding_lookup_sparse_v2(
    params,
    sp_ids,
    sp_weights,
    combiner=None,
    max_norm=None,
    name=None,
    allow_fast_lookup=False,
):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says ragged tensors are accepted and calls the weights argument
  # sparse_weights, but the code only accepts sparse tensors (it delegates to the version 1
  # function, which raises a type error otherwise) and the parameter is named sp_weights.
  # Version 2 sparse embedding lookup. Same as the version 1 function with the partition
  # strategy fixed to division. Inputs: params, sp_ids (sparse tensor of integer ids),
  # sp_weights (sparse tensor of the same shape or nothing), combiner (mean default, sqrtn or
  # sum), max_norm, name, allow_fast_lookup (boolean, default false). Returns a dense tensor
  # with one combined embedding per row of sp_ids. Raises a type error for wrong id or weight
  # types and a value error for an invalid combiner. Adds graph ops.
  """Looks up embeddings for the given ids and weights from a list of tensors.

  `params` is a dense tensor or a list of dense tensors, and `sp_ids` is a 2D
  `tf.SparseTensor` or `tf.RaggedTensor` indicating the indices of `params` to
  gather.

  This op is best described with an example. Suppose `params` is an embedding
  table of size `(4, 2)` and `sp_ids` has 3 rows. Since `sp_ids` is sparse or
  ragged, not every row has the same number of elements. The output has shape
  (3, 2). Each row of `sp_ids` is a list of indices, where each index selects a
  row of `params`. For a given row of `sp_ids`, the rows of `params` are
  gathered based on the indices in `sp_ids`, then combined by taking their sum
  or mean.

  >>> params = tf.constant([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=tf.float32)
  >>> sp_ids = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0], [2, 0]],
  ...                          values=[0, 1, 3, 2], dense_shape=(3, 2))
  >>> tf.nn.embedding_lookup_sparse(params, sp_ids, sp_weights=None,
  ...                               combiner='sum').numpy()
  array([[4., 6.], [7., 8.], [5., 6.]], dtype=float32)

  In this example, `sp_ids` has 3 rows, so the output has 3 rows. Row 0 of
  `sp_ids` has values 0 and 1, so it selects rows 0 and 1 from `params`, which
  are `[1, 2]` and `[3, 4]`. The rows are summed since `combiner='sum'`,
  resulting in the output row of `[4, 6]`.

  Since row 1 and 2 of `sp_ids` only have one value each, they simply select the
  corresponding row from `params` as the output row. Row 1 has value `3` so
  it selects the `params` elements `[7, 8]` and row 2 has the value 2 so it
  selects the `params` elements `[5, 6]`.

  If `sparse_weights` is specified, it must have the same shape as `sp_ids`.
  `sparse_weights` is used to assign a weight to each slice of `params`. For
  example:

  >>> params = tf.constant([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=tf.float32)
  >>> sp_ids = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0], [2, 0]],
  ...                          values=[0, 1, 3, 2], dense_shape=(3, 2))
  >>> sparse_weights = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0], [2, 0]],
  ...                                  values=[0.1, 1.0, 0.5, 2.0],
  ...                                  dense_shape=(3, 2))
  >>> tf.nn.embedding_lookup_sparse(params, sp_ids, sp_weights=sparse_weights,
  ...                               combiner='sum').numpy()
  array([[3.1, 4.2], [3.5, 4.], [10., 12.]], dtype=float32)

  In general, `params` can have shape `(p0, ..., pn)` and `sp_ids` can have `M`
  rows, where each row can have any number of elements. The output has shape
  `(M, p1, ..., pn)`. Each slice of the output `output[i, ...]` is obtained as
  follows: The `combiner` argument is used to combine the values
  `params[sp_ids[i, j], ...] * sparse_weights[i, j]` for each `j` in `range(0,
  len(sp_ids[i]))`, e.g. by taking the sum or mean of the values.

  This op assumes that there is at least one id for each row in the dense tensor
  represented by sp_ids (i.e. there are no rows with empty features), and that
  all the indices of sp_ids are in canonical row-major order.

  `sp_ids` and `sp_weights` (if not None) are `SparseTensor`s or `RaggedTensor`s
  with rank of 2. For `SpareTensor`s with left-aligned non-zero entries which
  can be described as `RaggedTensor`s, use of `RaggedTensor`s can yield higher
  performance.

  This op assumes that all id values lie in the range [0, p0), where p0
  is `params.shape[0]`. If you want a version of this op that prunes id values
  less than 0, see `tf.nn.safe_embedding_lookup_sparse`

  If `len(params) > 1`, each element of `sp_ids` is partitioned between the
  elements of `params` according to the "div" partition strategy, which means we
  assign ids to partitions in a contiguous manner. For instance, 13 ids are
  split across 5 partitions as:
  `[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11, 12]]`.

  If the id space does not evenly divide the number of partitions, each of the
  first `(max_id + 1) % len(params)` partitions will be assigned one more id.

  Args:
    params: A single tensor representing the complete embedding tensor, or a
      list of tensors all of same shape except for the first dimension,
      representing sharded embedding tensors following "div" partition strategy.
    sp_ids: N x M `SparseTensor` of int64 ids where N is typically batch size
      and M is arbitrary or a `RaggedTensor` with rank 2.
    sparse_weights: `SparseTensor` or `RaggedTensor` of same type and shape as
      `sparse_ids`, containing float / double weights corresponding to
      `sparse_ids`, or `None` if all weights are assumed to be 1.0.
    combiner: A string specifying the reduction op. Currently "mean", "sqrtn"
      and "sum" are supported. "sum" computes the weighted sum of the embedding
      results for each row. "mean" is the weighted sum divided by the total
      weight. "sqrtn" is the weighted sum divided by the square root of the sum
      of the squares of the weights. Defaults to `mean`.
    max_norm: If not `None`, each embedding is clipped if its l2-norm is larger
      than this value, before combining.
    name: Optional name for the op.
    allow_fast_lookup: An optional boolean specifying whether to allow
      simplified embedding lookups when `params` is a single tensor and
      `max_norm` is `None`. Setting this flag to `True` during training can
      cause the use of dense gradients with increased memory footprint.

  Returns:
    A dense tensor representing the combined embeddings for the
    sparse ids. For each row in the dense tensor represented by `sp_ids`, the op
    looks up the embeddings for all ids in that row, multiplies them by the
    corresponding weight, and combines these embeddings as specified.

    In other words, if

      `shape(combined params) = [p0, p1, ..., pm]`

    and

      `shape(sp_ids) = shape(sp_weights) = [d0, d1]`

    then

      `shape(output) = [d0, p1, ..., pm]`.

    For instance, if params is a 10x20 matrix, and sp_ids / sp_weights are

      ```python
      [0, 0]: id 1, weight 2.0
      [0, 1]: id 3, weight 0.5
      [1, 0]: id 0, weight 1.0
      [2, 3]: id 1, weight 3.0
      ```

    with `combiner`="mean", then the output will be a 3x20 matrix where

      ```python
      output[0, :] = (params[1, :] * 2.0 + params[3, :] * 0.5) / (2.0 + 0.5)
      output[1, :] = (params[0, :] * 1.0) / 1.0
      output[2, :] = (params[1, :] * 3.0) / 3.0
      ```

  Raises:
    TypeError: If `sp_ids` is not a `SparseTensor`, or if `sp_weights` is
      neither `None` nor `SparseTensor`.
    ValueError: If `combiner` is not one of {"mean", "sqrtn", "sum"}.
  """
  return embedding_lookup_sparse(
      params,
      sp_ids,
      sp_weights,
      "div",
      name,
      combiner,
      max_norm,
      allow_fast_lookup,
  )


@tf_export("nn.safe_embedding_lookup_sparse", v1=[])
@dispatch.add_dispatch_support
def safe_embedding_lookup_sparse_v2(
    embedding_weights,
    sparse_ids,
    sparse_weights=None,
    combiner="mean",
    default_id=None,
    max_norm=None,
    name=None,
    allow_fast_lookup=False,
):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says ragged tensors are accepted, but the underlying code path
  # reshapes and prunes the ids as sparse tensors only.
  # Version 2 safe sparse embedding lookup. Same as the version 1 function but fixes the
  # partition strategy to division. Inputs: embedding_weights (tensor or list of tensors),
  # sparse_ids (sparse tensor of integer ids), sparse_weights (optional matching sparse tensor),
  # combiner (default mean), default_id (id used for rows without features, else zero vector),
  # max_norm, name, allow_fast_lookup. Returns a dense tensor whose shape is the id shape without
  # its last dimension followed by the embedding dimensions. Raises a value error when the weights
  # are missing or empty. Adds graph ops.
  """Lookup embedding results, accounting for invalid IDs and empty features.

  The partitioned embedding in `embedding_weights` must all be the same shape
  except for the first dimension. The first dimension is allowed to vary as the
  vocabulary size is not necessarily a multiple of num of shards.

  This is similar to `tf.nn.embedding_lookup_sparse`, except invalid IDs (< 0)
  are pruned from input IDs and weights, as well as any IDs with non-positive
  weight. For an entry with no features, the embedding vector for `default_id`
  is returned, or the 0-vector if `default_id` is not supplied. See
  `tf.nn.embedding_lookup_sparse` for more information on how sparse embedding
  lookups work in general.

  The ids and weights may be multi-dimensional `SparseTensor`s or
  `RaggedTensor`s with rank of 2. For `SpareTensor`s with left-aligned non-zero
  entries which can be described as `RaggedTensor`s, use of `RaggedTensor`s can
  yield higher performance.

  If `len(embedding_weights) > 1`, each element `id` of `ids` is partitioned
  between the elements of `embedding_weights` according to the "div" partition
  strategy, which means we assign ids to partitions in a contiguous manner. For
  instance, 13 ids are split across 5 partitions as:
  `[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11, 12]]`.

  If the id space does not evenly divide the number of partitions, each of the
  first `(max_id + 1) % len(embedding_weights)` partitions will be assigned one
  more id.

  Args:
    embedding_weights: A single tensor representing the complete embedding
      tensor, or a list of tensors all of same shape except for the first
      dimension, representing sharded embedding tensors following "div"
      partition strategy.
    sparse_ids: `SparseTensor` of shape `[d_0, d_1, ..., d_n]` containing the
      ids, where `d_0` is typically batch size, or a `RaggedTensor` with rank 2.
    sparse_weights: `SparseTensor` or `RaggedTensor` of same type and shape as
      `sparse_ids`, containing float weights corresponding to `sparse_ids`, or
      `None` if all weights are assumed to be 1.0.
    combiner: A string specifying how to combine embedding results for each
      entry. Currently "mean", "sqrtn" and "sum" are supported, with "mean" the
      default.
    default_id: The id to use for an entry with no features. Defaults to
      0-vector.
    max_norm: If not `None`, all embeddings are l2-normalized to max_norm before
      combining.
    name: A name for this operation (optional).
    allow_fast_lookup: An optional boolean specifying whether to allow
      simplified embedding lookups when `params` is a single tensor and
      `max_norm` is `None`. Setting this flag to `True` during training can
      cause the use of dense gradients with increased memory footprint.

  Returns:
    A dense tensor representing the combined embeddings for the
    sparse ids. For each row in the dense tensor represented by `sparse_ids`,
    the op looks up the embeddings for all ids in that row, multiplies them by
    the corresponding weight, and combines these embeddings as specified.

    In other words, if

      `shape(combined embedding_weights) = [p0, p1, ..., pm]`

    and

      `shape(sparse_ids) = shape(sparse_weights) = [d0, d1, ..., dn]`

    then

      `shape(output) = [d0, d1, ... dn-1, p1, ..., pm]`.

    For instance, if params is a 10x20 matrix, and sp_ids / sp_weights are

      ```python
      [0, 0]: id 1, weight 2.0
      [0, 1]: id 3, weight 0.5
      [1, 0]: id -1, weight 1.0
      [2, 3]: id 1, weight 3.0
      ```

    `default_id` is 0.

    with `combiner`="mean", then the output will be a 3x20 matrix where

      ```python
      output[0, :] = (params[1, :] * 2.0 + params[3, :] * 0.5) / (2.0 + 0.5)
      output[1, :] = (params[0, :] * 1.0) / 1.0
      output[2, :] = (params[1, :] * 3.0) / 3.0
      ```

  Raises:
    ValueError: if `embedding_weights` is empty.
  """
  return safe_embedding_lookup_sparse(
      embedding_weights,
      sparse_ids,
      sparse_weights=sparse_weights,
      combiner=combiner,
      default_id=default_id,
      name=name,
      partition_strategy="div",
      max_norm=max_norm,
      allow_fast_lookup=allow_fast_lookup,
  )


@tf_export(v1=["nn.safe_embedding_lookup_sparse"])
@dispatch.add_dispatch_support
def safe_embedding_lookup_sparse(
    embedding_weights,
    sparse_ids,
    sparse_weights=None,
    combiner="mean",
    default_id=None,
    name=None,
    partition_strategy="div",
    max_norm=None,
    allow_fast_lookup=False,
):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says ragged tensors are accepted, but the code treats the ids as
  # a sparse tensor (reshape, prune, fill empty rows).
  # KNOWN ISSUE: the fill value is chosen with a logical-or fallback, so an explicit default_id
  # of zero behaves the same as nothing for filling; yet only a default_id of nothing triggers
  # the zeroing of empty rows, so an explicit zero default returns row zero of the embeddings
  # (not necessarily the zero vector) for empty rows.
  # Version 1 safe sparse embedding lookup. Flattens higher-rank sparse ids (and weights) to
  # two dimensions, prunes ids below zero (and, unless the combiner is sum, entries with
  # non-positive weight), fills empty rows with the default id (weights filled with one), runs
  # the sparse lookup, zeroes rows that had no features when default_id is nothing, and reshapes
  # back to the original leading dimensions plus the embedding dimensions (setting a partial
  # static shape). Inputs: embedding_weights (tensor, list or partitioned variable); sparse_ids;
  # sparse_weights (optional); combiner (default mean); default_id; name; partition_strategy
  # ("div" default); max_norm; allow_fast_lookup. Raises a value error if embedding_weights is
  # nothing or an empty list. Weights are converted to tensors using the dtype of sparse_weights,
  # keeping resource variables as they are. Adds graph ops.
  """Lookup embedding results, accounting for invalid IDs and empty features.

  The partitioned embedding in `embedding_weights` must all be the same shape
  except for the first dimension. The first dimension is allowed to vary as the
  vocabulary size is not necessarily a multiple of `P`.  `embedding_weights`
  may be a `PartitionedVariable` as returned by using
  `tf.compat.v1.get_variable()` with a
  partitioner.

  Invalid IDs (< 0) are pruned from input IDs and weights, as well as any IDs
  with non-positive weight. For an entry with no features, the embedding vector
  for `default_id` is returned, or the 0-vector if `default_id` is not supplied.

  The ids and weights may be multi-dimensional `SparseTensor`s or
  `RaggedTensor`s with rank of 2. For `SpareTensor`s with left-aligned non-zero
  entries which can be described as `RaggedTensor`s, use of `RaggedTensor`s can
  yield higher performance. Embeddings are always aggregated along the last
  dimension.

  Args:
    embedding_weights: A single tensor representing the complete embedding
      tensor, or a list tensors all of same shape except for the first
      dimension, representing sharded embedding tensors. Alternatively, a
      `PartitionedVariable`, created by partitioning along dimension 0. Each
      element must be appropriately sized for the given `partition_strategy`.
    sparse_ids: `SparseTensor` of shape `[d_0, d_1, ..., d_n]` containing the
      ids, where `d_0` is typically batch size, or a `RaggedTensor` with rank 2.
    sparse_weights: `SparseTensor` or `RaggedTensor` of same type and shape as
      `sparse_ids`, containing float weights corresponding to `sparse_ids`, or
      `None` if all weights are assumed to be 1.0.
    combiner: A string specifying how to combine embedding results for each
      entry. Currently "mean", "sqrtn" and "sum" are supported, with "mean" the
      default.
    default_id: The id to use for an entry with no features.
    name: A name for this operation (optional).
    partition_strategy: A string specifying the partitioning strategy. Currently
      `"div"` and `"mod"` are supported. Default is `"div"`.
    max_norm: If not `None`, all embeddings are l2-normalized to max_norm before
      combining.
    allow_fast_lookup: An optional boolean specifying whether to allow
      simplified embedding lookups when `params` is a single tensor and
      `max_norm` is `None`. Setting this flag to `True` during training can
      cause the use of dense gradients with increased memory footprint.

  Returns:
    A dense tensor representing the combined embeddings for the
    sparse ids. For each row in the dense tensor represented by `sp_ids`, the op
    looks up the embeddings for all ids in that row, multiplies them by the
    corresponding weight, and combines these embeddings as specified.

    In other words, if

      `shape(combined embedding_weights) = [p0, p1, ..., pm]`

    and

      `shape(sparse_ids) = shape(sparse_weights) = [d0, d1, ..., dn]`

    then

      `shape(output) = [d0, d1, ... dn-1, p1, ..., pm]`.

    For instance, if params is a 10x20 matrix, and sp_ids / sp_weights are

      ```python
      [0, 0]: id 1, weight 2.0
      [0, 1]: id 3, weight 0.5
      [1, 0]: id -1, weight 1.0
      [2, 3]: id 1, weight 3.0
      ```

    `default_id` is 0.

    with `combiner`="mean", then the output will be a 3x20 matrix where

      ```python
      output[0, :] = (params[1, :] * 2.0 + params[3, :] * 0.5) / (2.0 + 0.5)
      output[1, :] = (params[0, :] * 1.0) / 1.0
      output[2, :] = (params[1, :] * 3.0) / 3.0
      ```

  Raises:
    ValueError: if `embedding_weights` is empty.
  """
  if embedding_weights is None:
    raise ValueError(f"Missing embedding_weights {embedding_weights}.")
  if isinstance(embedding_weights, variables.PartitionedVariable):
    embedding_weights = list(embedding_weights)  # get underlying Variables.
  if not isinstance(embedding_weights, list):
    embedding_weights = [embedding_weights]
  if len(embedding_weights) < 1:
    raise ValueError(f"Missing embedding_weights {embedding_weights}.")

  dtype = sparse_weights.dtype if sparse_weights is not None else None
  embedding_weights = [
      w if (resource_variable_ops.is_resource_variable(w)
            and dtype in (None, w.dtype))
      else ops.convert_to_tensor(w, dtype=dtype)
      for w in embedding_weights
  ]

  with ops.name_scope(name, "embedding_lookup", embedding_weights +
                      [sparse_ids, sparse_weights]) as scope:
    # Reshape higher-rank sparse ids and weights to linear segment ids.
    original_shape = sparse_ids.dense_shape
    original_rank_dim = tensor_shape.dimension_value(
        sparse_ids.dense_shape.get_shape()[0])
    original_rank = (
        array_ops.size(original_shape)
        if original_rank_dim is None else original_rank_dim)
    sparse_ids = sparse_ops.sparse_reshape(sparse_ids, [
        math_ops.reduce_prod(
            array_ops.slice(original_shape, [0], [original_rank - 1])),
        array_ops.gather(original_shape, original_rank - 1)
    ])
    if sparse_weights is not None:
      sparse_weights = sparse_tensor.SparseTensor(sparse_ids.indices,
                                                  sparse_weights.values,
                                                  sparse_ids.dense_shape)

    # Prune invalid ids and weights.
    sparse_ids, sparse_weights = _prune_invalid_ids(sparse_ids, sparse_weights)
    if combiner != "sum":
      sparse_ids, sparse_weights = _prune_invalid_weights(
          sparse_ids, sparse_weights)

    # Fill in dummy values for empty features, if necessary.
    sparse_ids, is_row_empty = sparse_ops.sparse_fill_empty_rows(
        sparse_ids, default_id or 0)
    if sparse_weights is not None:
      sparse_weights, _ = sparse_ops.sparse_fill_empty_rows(sparse_weights, 1.0)

    result = embedding_lookup_sparse(
        embedding_weights,
        sparse_ids,
        sparse_weights,
        combiner=combiner,
        partition_strategy=partition_strategy,
        name=None if default_id is None else scope,
        max_norm=max_norm,
        allow_fast_lookup=allow_fast_lookup,
    )

    if default_id is None:
      # Broadcast is_row_empty to the same shape as embedding_lookup_result,
      # for use in Select.
      is_row_empty = array_ops.tile(
          array_ops.reshape(is_row_empty, [-1, 1]),
          array_ops_stack.stack([1, array_ops.shape(result)[1]]))

      result = array_ops.where(
          is_row_empty, array_ops.zeros_like(result), result, name=scope)

    # Reshape back from linear ids back into higher-dimensional dense result.
    final_result = array_ops.reshape(
        result,
        array_ops.concat([
            array_ops.slice(
                math_ops.cast(original_shape, dtypes.int32), [0],
                [original_rank - 1]),
            array_ops.slice(array_ops.shape(result), [1], [-1])
        ], 0))
    final_result.set_shape(
        tensor_shape.unknown_shape(
            (tensor_shape.Dimension(original_rank_dim) - 1).value
        ).concatenate(result.get_shape()[1:])
    )
    return final_result


def embedding_lookup_sparse_impl(
    params,
    segment_ids,
    sp_weights,
    ids,
    combiner,
    ignore_weights,
    max_norm,
    allow_fast_lookup,
    partition_strategy,
    name,
):
  # [FUNCTION DOC]
  # HELPER: shared implementation behind the sparse embedding lookup functions.
  # Aggregates embeddings per segment. Inputs: params (list of embedding tensors or variables);
  # segment_ids (row index of each id); sp_weights (sparse weights, used when weights are not
  # ignored); ids (flat id values); combiner (mean, sqrtn or sum); ignore_weights (boolean);
  # max_norm; allow_fast_lookup; partition_strategy; name.
  # Fast path: a single non-composite params on the same device as the segment ids, without
  # max_norm, when fast lookup is allowed or weights are ignored (and a forward-compatibility
  # date check passes): uses ids directly, reads variables without copy, and requests a sparse
  # segment gradient only when fast lookup was not explicitly allowed. Otherwise deduplicates the
  # ids and does a regular embedding lookup.
  # With weights: gathers per id, casts half-precision embeddings to float32 for the computation,
  # casts weights to match, reshapes weights for broadcasting, multiplies, then applies segment
  # sum (sum), sum divided by weight sum with safe division (mean), or divided by the square root
  # of the sum of squared weights (sqrtn), casting back to the original dtype.
  # Without weights: uses sparse segment sum, mean or square-root-n ops.
  # An unknown combiner triggers an assertion failure. Returns the combined embeddings tensor.
  """Implementation of sparse embedding aggregation."""
  need_sparse_segment_gradient = False
  # Ensure we can query the devices below.
  segment_ids = ops.convert_to_tensor(segment_ids, name="segment_ids")
  if len(params) == 1 and not isinstance(
      params[0], (core.Tensor, composite_tensor.CompositeTensor)
  ):
    params = [ops.convert_to_tensor(params[0], name="params")]
  # Note that if the params are on a different device (e.g., CPU), we must use
  # embedding_lookup() so that the gather operation is colocated with them.
  if (
      len(params) == 1
      and not isinstance(params[0], composite_tensor.CompositeTensor)
      and params[0].device == segment_ids.device
      and max_norm is None
      and (
          allow_fast_lookup
          or (ignore_weights and compat.forward_compatible(2023, 9, 26))
      )
  ):
    idx = ids
    embeddings = params[0]
    if isinstance(embeddings, resource_variable_ops.BaseResourceVariable):
      # Avoid a redundant copy due to copy-on-read semantics for
      # sparsely-updated variables.
      embeddings = embeddings.read_value_no_copy()
    if not allow_fast_lookup:
      need_sparse_segment_gradient = True
  else:
    ids, idx = array_ops.unique(ids)
    embeddings = embedding_lookup(
        params, ids, partition_strategy=partition_strategy, max_norm=max_norm
    )

  if not ignore_weights:
    if segment_ids.dtype != dtypes.int32:
      segment_ids = math_ops.cast(segment_ids, dtypes.int32)

    weights = sp_weights.values
    embeddings = array_ops.gather(embeddings, idx)

    original_dtype = embeddings.dtype
    if embeddings.dtype in (dtypes.float16, dtypes.bfloat16):
      # Cast low-precision embeddings to float32 during the computation to
      # avoid numerical issues.
      embeddings = math_ops.cast(embeddings, dtypes.float32)
    if weights.dtype != embeddings.dtype:
      weights = math_ops.cast(weights, embeddings.dtype)

    # Reshape weights to allow broadcast
    ones_shape = array_ops.expand_dims(array_ops.rank(embeddings) - 1, 0)
    weights_shape = array_ops.shape(weights)
    ones = array_ops.ones(ones_shape, dtype=weights_shape.dtype)
    bcast_weights_shape = array_ops.concat([weights_shape, ones], 0)

    orig_weights_shape = weights.get_shape()
    weights = array_ops.reshape(weights, bcast_weights_shape)

    # Set the weight shape, since after reshaping to bcast_weights_shape,
    # the shape becomes None.
    if embeddings.get_shape().ndims is not None:
      weights.set_shape(
          orig_weights_shape.concatenate(
              [1 for _ in range(embeddings.get_shape().ndims - 1)]
          )
      )

    embeddings *= weights

    if combiner == "sum":
      embeddings = math_ops.segment_sum(embeddings, segment_ids, name=name)
    elif combiner == "mean":
      embeddings = math_ops.segment_sum(embeddings, segment_ids)
      weight_sum = math_ops.segment_sum(weights, segment_ids)
      embeddings = math_ops.div_no_nan(embeddings, weight_sum, name=name)
    elif combiner == "sqrtn":
      embeddings = math_ops.segment_sum(embeddings, segment_ids)
      weights_squared = math_ops.pow(weights, 2)
      weight_sum = math_ops.segment_sum(weights_squared, segment_ids)
      weight_sum_sqrt = math_ops.sqrt(weight_sum)
      embeddings = math_ops.div_no_nan(embeddings, weight_sum_sqrt, name=name)
    else:
      assert False, "Unrecognized combiner"
    if embeddings.dtype != original_dtype:
      embeddings = math_ops.cast(embeddings, original_dtype)
  else:
    if segment_ids.dtype not in (dtypes.int32, dtypes.int64):
      segment_ids = math_ops.cast(segment_ids, dtypes.int32)
    assert idx is not None
    if combiner == "sum":
      embeddings = math_ops.sparse_segment_sum(
          embeddings,
          idx,
          segment_ids,
          name=name,
          sparse_gradient=need_sparse_segment_gradient,
      )
    elif combiner == "mean":
      embeddings = math_ops.sparse_segment_mean(
          embeddings,
          idx,
          segment_ids,
          name=name,
          sparse_gradient=need_sparse_segment_gradient,
      )
    elif combiner == "sqrtn":
      embeddings = math_ops.sparse_segment_sqrt_n(
          embeddings,
          idx,
          segment_ids,
          name=name,
          sparse_gradient=need_sparse_segment_gradient,
      )
    else:
      assert False, "Unrecognized combiner"

  return embeddings


def _prune_invalid_ids(sparse_ids, sparse_weights):
  # [FUNCTION DOC]
  # HELPER: id pruning step of the safe sparse embedding lookup.
  # KNOWN ISSUE: the weights branch combines the validity mask with a tensor of all true values,
  # so it has no effect; only the ids are actually tested (ids >= 0).
  # Keeps only entries whose id is non-negative, retaining the same entries in the weights when
  # present. Inputs: sparse_ids (sparse tensor), sparse_weights (sparse tensor or nothing).
  # Returns a pair of the pruned ids and pruned weights (nothing if no weights given).
  """Prune invalid IDs (< 0) from the input ids and weights."""
  is_id_valid = math_ops.greater_equal(sparse_ids.values, 0)
  if sparse_weights is not None:
    is_id_valid = math_ops.logical_and(
        is_id_valid,
        array_ops.ones_like(sparse_weights.values, dtype=dtypes.bool))
  sparse_ids = sparse_ops.sparse_retain(sparse_ids, is_id_valid)
  if sparse_weights is not None:
    sparse_weights = sparse_ops.sparse_retain(sparse_weights, is_id_valid)
  return sparse_ids, sparse_weights


def _prune_invalid_weights(sparse_ids, sparse_weights):
  # [FUNCTION DOC]
  # HELPER: weight pruning step of the safe sparse embedding lookup.
  # KNOWN ISSUE: the docstring says it prunes weights below zero, but the code keeps only
  # strictly positive weights, so zero weights are also removed.
  # When sparse_weights is not nothing, keeps only entries with weight greater than zero in both
  # the ids and the weights. Returns a pair of ids and weights (unchanged if no weights).
  """Prune invalid weights (< 0) from the input ids and weights."""
  if sparse_weights is not None:
    is_weights_valid = math_ops.greater(sparse_weights.values, 0)
    sparse_ids = sparse_ops.sparse_retain(sparse_ids, is_weights_valid)
    sparse_weights = sparse_ops.sparse_retain(sparse_weights, is_weights_valid)
  return sparse_ids, sparse_weights
