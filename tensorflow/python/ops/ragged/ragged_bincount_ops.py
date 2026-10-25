# [MODULE DOC]
# KNOWN ISSUE: the license header in this file has a corrupted word ("maxlengthations"
# instead of "limitations"); it is only a comment typo and has no runtime effect.
# Ragged-tensor implementations of the bincount family of ops. Importing the module
# registers, through the API dispatch mechanism, a ragged handler for the dense
# bincount API and one for the sparse bincount API, so that the public functions accept
# ragged inputs. It also defines validate_ragged_weights, a helper that checks or
# fabricates the weights tensor passed to the underlying kernels. The kernels used are
# the ragged and dense bincount ops and the ragged/dense sparse-output count ops.
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
# maxlengthations under the License.
# ==============================================================================
"""bincount ops for RaggedTensors."""

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import bincount_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_count_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import dispatch


@dispatch.dispatch_for_api(bincount_ops.bincount)
def bincount(arr: ragged_tensor.RaggedTensor,
             weights=None,
             minlength=None,
             maxlength=None,
             dtype=dtypes.int32,
             name=None,
             axis=None,
             binary_output=False):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says only an invalid-argument error is raised for negative
  # values, but this code also raises a value error itself when both weights and
  # binary_output are given, or when axis is not 0, -1 or nothing. The docstring also
  # says a vector is returned, but with axis -1 a 2-D dense result is produced.
  # Dispatch handler for the dense bincount API when the input is a ragged tensor.
  # Inputs: arr (ragged tensor of counted values; non-integer dtypes are cast to 32-bit
  # integers), weights (optional ragged tensor with identical row splits), minlength and
  # maxlength (optional bounds on output length), dtype (output dtype when no weights),
  # name (optional scope name, default "bincount"), axis (nothing or 0 flattens all
  # dimensions; -1 counts per row; other values rejected) and binary_output (emit 1
  # instead of counts). The output size is max value plus one (zero for an empty input),
  # raised to minlength and capped at maxlength. For axis 0 nested ragged dimensions are
  # peeled until a dense values tensor remains and a dense bincount kernel runs; for
  # axis -1 with a ragged input the ragged bincount kernel is used with the row splits.
  # Returns the count tensor. Raises a value error for the argument problems above;
  # kernels raise invalid-argument errors for negative values. No other side effects.
  """Counts the number of occurrences of each value in an integer array.

  If `minlength` and `maxlength` are not given, returns a vector with length
  `tf.reduce_max(arr) + 1` if `arr` is non-empty, and length 0 otherwise.

  >>> data = tf.ragged.constant([[1, 1], [2, 3, 2, 4, 4, 5]])
  >>> tf.math.bincount(data)
  <tf.Tensor: ... numpy=array([0, 2, 2, 1, 2, 1], dtype=int32)>

  Vector length = Maximum element in vector `values` is 5. Adding 1, which is 6
                  will be the vector length.

  Each bin value in the output indicates number of occurrences of the particular
  index. Here, index 1 in output has a value 2. This indicates value 1 occurs
  two times in `values`.

  **Bin-counting with weights**

  >>> data = tf.ragged.constant([[1, 1], [2, 3, 2, 4, 4, 5]])
  >>> weights = tf.ragged.constant([[1, 5], [0, 1, 0, 5, 4, 5]])
  >>> tf.math.bincount(data, weights=weights)
  <tf.Tensor: ... numpy=array([0, 6, 0, 1, 9, 5], dtype=int32)>

  When `weights` is specified, bins will be incremented by the corresponding
  weight instead of 1. Here, index 1 in output has a value 6. This is the
  summation of `weights` corresponding to the value in `arr` (i.e. for index
  1, the first two values `arr` are 1 so the first two weights, 1 and 5, are
  summed).

  There is an equivilance between bin-counting with weights and
  `unsorted_segment_sum` where `data` is the weights and `segment_ids` are the
  values.

  >>> data = tf.ragged.constant([[1, 1], [2, 3, 2, 4, 4, 5]])
  >>> weights = tf.ragged.constant([[1, 5], [0, 1, 0, 5, 4, 5]])
  >>> tf.math.unsorted_segment_sum(weights, data, num_segments=6).numpy()
  array([0, 6, 0, 1, 9, 5], dtype=int32)

  On GPU, `bincount` with weights is only supported when XLA is enabled
  (typically when a function decorated with `@tf.function(jit_compile=True)`).
  `unsorted_segment_sum` can be used as a workaround for the non-XLA case on
  GPU.

  **Bin-counting matrix rows independently**

  This example uses `axis=-1` with a 2 dimensional input and returns a
  `Tensor` with bincounting where axis 0 is **not** flattened, i.e. an
  independent bincount for each matrix row.

  >>> data = tf.ragged.constant([[1, 2], [3, 0, 0, 0, 1, 2]], dtype=np.int32)
  >>> tf.math.bincount(data, axis=-1)
  <tf.Tensor: shape=(2, 4), dtype=int32, numpy=
    array([[0, 1, 1, 0],
           [3, 1, 1, 1]], dtype=int32)>

  **Bin-counting with binary_output**

  This example gives binary output instead of counting the occurrence.

  >>> data = tf.ragged.constant([[1, 2], [3, 0, 0, 0, 1, 2]], dtype=np.int32)
  >>> tf.math.bincount(data, axis=-1, binary_output=True)
  <tf.Tensor: shape=(2, 4), dtype=int32, numpy=
    array([[0, 1, 1, 0],
           [1, 1, 1, 1]], dtype=int32)>

  Args:
    arr: A RaggedTensor whose values should be counted. These tensors must have
      a rank of 2 if `axis=-1`.
    weights: If non-None, must be a RaggedTensor with the same row splits as
      `arr`. For each value in `arr`, the bin will be incremented by the
      corresponding weight instead of 1. If non-None, `binary_output` must be
      False.
    minlength: If given, ensures the output has length at least `minlength`,
      padding with zeros at the end if necessary.
    maxlength: If given, skips values in `arr` that are equal or greater than
      `maxlength`, ensuring that the output has length at most `maxlength`.
    dtype: If `weights` is None, determines the type of the output bins.
    name: A name scope for the associated operations (optional).
    axis: The axis to slice over. Axes at and below `axis` will be flattened
      before bin counting. Currently, only `0`, and `-1` are supported. If None,
      all axes will be flattened (identical to passing `0`).
    binary_output: If True, this op will output 1 instead of the number of times
      a token appears (equivalent to one_hot + reduce_any instead of one_hot +
      reduce_add). Defaults to False.

  Returns:
    A vector with the same dtype as `weights` or the given `dtype` containing
    the bincount values.

  Raises:
    `InvalidArgumentError` if negative values are provided as an input.
  """
  name = "bincount" if name is None else name
  with ops.name_scope(name):
    arr = ragged_tensor.convert_to_tensor_or_ragged_tensor(arr, name="arr")
    if weights is not None:
      if not isinstance(weights, sparse_tensor.SparseTensor):
        weights = ragged_tensor.convert_to_tensor_or_ragged_tensor(
            weights, name="weights")

    if weights is not None and binary_output:
      raise ValueError("Arguments `binary_output` and `weights` are mutually "
                       "exclusive. Please specify only one.")

    if not arr.dtype.is_integer:
      arr = math_ops.cast(arr, dtypes.int32)
    if axis is None:
      axis = 0

    if axis not in [0, -1]:
      raise ValueError(f"Unsupported value for argument axis={axis}. Only 0 and"
                       " -1 are currently supported.")

    array_is_nonempty = array_ops.size(arr) > 0
    output_size = math_ops.cast(array_is_nonempty, arr.dtype) * (
        math_ops.reduce_max(arr) + 1)
    if minlength is not None:
      minlength = ops.convert_to_tensor(
          minlength, name="minlength", dtype=arr.dtype)
      output_size = gen_math_ops.maximum(minlength, output_size)
    if maxlength is not None:
      maxlength = ops.convert_to_tensor(
          maxlength, name="maxlength", dtype=arr.dtype)
      output_size = gen_math_ops.minimum(maxlength, output_size)

    if axis == 0:
      # Flatten RaggedTensors with multiple ragged dimensions which use a
      # nested RaggedTensor for the values tensor.
      while isinstance(arr, ragged_tensor.RaggedTensor):
        if weights is not None:
          weights = validate_ragged_weights(arr, weights, dtype)
        arr = arr.values

    if isinstance(arr, ragged_tensor.RaggedTensor):
      weights = validate_ragged_weights(arr, weights, dtype)
      return gen_math_ops.ragged_bincount(
          splits=arr.row_splits,
          values=arr.values,
          size=output_size,
          weights=weights,
          binary_output=binary_output)
    else:
      weights = bincount_ops.validate_dense_weights(arr, weights, dtype)
      return gen_math_ops.dense_bincount(
          input=arr,
          size=output_size,
          weights=weights,
          binary_output=binary_output)


@dispatch.dispatch_for_api(sparse_ops.sparse_bincount)
def sparse_bincount(values: ragged_tensor.RaggedTensor,
                    weights=None,
                    axis=0,
                    minlength=None,
                    maxlength=None,
                    binary_output=False,
                    name=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring mentions that dense and sparse tensors are accepted, but
  # this handler is registered only for ragged inputs. Also, for axis 0 with a nested
  # ragged input only one ragged level is stripped, so the second validation may fail.
  # Dispatch handler for the sparse bincount API with a ragged input.
  # Inputs: values (ragged tensor of integers), weights (optional ragged tensor with the
  # same row splits), axis (0 or -1; nothing is treated as 0, other values raise a value
  # error), minlength and maxlength (optional output length bounds; absent means no
  # bound), binary_output (emit 1 instead of counts; cannot combine with weights, which
  # raises a value error) and name (optional scope name).
  # With axis 0 the values are flattened one level first. The ragged or dense
  # sparse-output count kernel then produces indices, values and dense shape.
  # Returns a sparse tensor built from them. Kernels raise invalid-argument errors on
  # negative values. No other side effects.
  """Count the number of times an integer value appears in a tensor.

  This op takes an N-dimensional `Tensor`, `RaggedTensor`, or `SparseTensor`,
  and returns an N-dimensional int64 SparseTensor where element
  `[i0...i[axis], j]` contains the number of times the value `j` appears in
  slice `[i0...i[axis], :]` of the input tensor.  Currently, only N=0 and
  N=-1 are supported.

  Args:
    values: A RaggedTensor whose values should be
      counted. These tensors must have a rank of 2 if `axis=-1`.
    weights: If non-None, must be a RaggedTensor with the same row splits as
      `values`. For each value in `value`, the bin will be incremented by the
      corresponding weight instead of 1.
    axis: The axis to slice over. Axes at and below `axis` will be flattened
      before bin counting. Currently, only `0`, and `-1` are supported. If None,
      all axes will be flattened (identical to passing `0`).
    minlength: If given, ensures the output has length at least `minlength`,
      padding with zeros at the end if necessary.
    maxlength: If given, skips values in `values` that are equal or greater than
      `maxlength`, ensuring that the output has length at most `maxlength`.
    binary_output: If True, this op will output 1 instead of the number of times
      a token appears (equivalent to one_hot + reduce_any instead of one_hot +
      reduce_add). Defaults to False.
    name: A name for this op.

  Returns:
    A SparseTensor with `output.shape = values.shape[:axis] + [N]`, where `N` is
      * `maxlength` (if set);
      * `minlength` (if set, and `minlength > reduce_max(values)`);
      * `0` (if `values` is empty);
      * `reduce_max(values) + 1` otherwise.

  Raises:
    `InvalidArgumentError` if negative values are provided as an input.

  Examples:

  **Bin-counting every item in individual batches**

  This example takes an input (which could be a Tensor, RaggedTensor, or
  SparseTensor) and returns a SparseTensor where the value of (i,j) is the
  number of times value j appears in batch i.

  >>> data = tf.ragged.constant(
  ...     [[10, 20], [30, 20, 11, 101, 11, 10001]], dtype=np.int64)
  >>> tf.sparse.bincount(data, axis=-1)
  SparseTensor(indices=tf.Tensor(
    [[    0    10]
     [    0    20]
     [    1    11]
     [    1    20]
     [    1    30]
     [    1   101]
     [    1 10001]], shape=(7, 2), dtype=int64),
     values=tf.Tensor([1 1 2 1 1 1 1], shape=(7,), dtype=int64),
     dense_shape=tf.Tensor([    2 10002], shape=(2,), dtype=int64))

  **Bin-counting with defined output shape**

  This example takes an input (which could be a Tensor, RaggedTensor, or
  SparseTensor) and returns a SparseTensor where the value of (i,j) is the
  number of times value j appears in batch i. However, all values of j
  above 'maxlength' are ignored. The dense_shape of the output sparse tensor
  is set to 'minlength'. Note that, while the input is identical to the
  example above, the value '10001' in batch item 2 is dropped, and the
  dense shape is [2, 500] instead of [2,10002] or [2, 102].

  >>> minlength = maxlength = 500
  >>> data = tf.ragged.constant(
  ...     [[10, 20], [30, 20, 11, 101, 11, 10001]], dtype=np.int64)
  >>> tf.sparse.bincount(
  ...    data, axis=-1, minlength=minlength, maxlength=maxlength)
  SparseTensor(indices=tf.Tensor(
  [[  0  10]
   [  0  20]
   [  1  11]
   [  1  20]
   [  1  30]
   [  1 101]], shape=(6, 2), dtype=int64),
   values=tf.Tensor([1 1 2 1 1 1], shape=(6,), dtype=int64),
   dense_shape=tf.Tensor([  2 500], shape=(2,), dtype=int64))

  **Binary bin-counting**

  This example takes an input (which could be a Tensor, RaggedTensor, or
  SparseTensor) and returns a SparseTensor where (i,j) is 1 if the value j
  appears in batch i at least once and is 0 otherwise. Note that, even though
  some values (like 20 in batch 1 and 11 in batch 2) appear more than once,
  the 'values' tensor is all 1s.

  >>> data = tf.ragged.constant(
  ...     [[10, 20], [30, 20, 11, 101, 11, 10001]], dtype=np.int64)
  >>> tf.sparse.bincount(data, binary_output=True, axis=-1)
  SparseTensor(indices=tf.Tensor(
  [[  0  10]
   [  0  20]
   [  1  11]
   [  1  20]
   [  1  30]
   [  1 101]
   [  1 10001]], shape=(7, 2), dtype=int64),
   values=tf.Tensor([1 1 1 1 1 1 1], shape=(7,), dtype=int64),
   dense_shape=tf.Tensor([  2 10002], shape=(2,), dtype=int64))

  **Weighted bin-counting**

  This example takes two inputs - a values tensor and a weights tensor. These
  tensors must be identically shaped, and have the same row splits or indices
  in the case of RaggedTensors or SparseTensors. When performing a weighted
  count, the op will output a SparseTensor where the value of (i, j) is the
  sum of the values in the weight tensor's batch i in the locations where
  the values tensor has the value j. In this case, the output dtype is the
  same as the dtype of the weights tensor.

  >>> data = tf.ragged.constant(
  ...    [[10, 20], [30, 20, 11, 101, 11, 10001]], dtype=np.int64)
  >>> weights = tf.ragged.constant(
  ...    [[2, 0.25], [15, 0.5, 2, 17, 3, 0.9]])
  >>> tf.sparse.bincount(data, weights=weights, axis=-1)
  SparseTensor(indices=tf.Tensor(
  [[    0    10]
   [    0    20]
   [    1    11]
   [    1    20]
   [    1    30]
   [    1   101]
   [    1 10001]], shape=(7, 2), dtype=int64),
   values=tf.Tensor([ 2. 0.25  5. 0.5 15. 17. 0.9 ], shape=(7,), dtype=float32),
   dense_shape=tf.Tensor([    2 10002], shape=(2,), dtype=int64))

  """
  with ops.name_scope(name, "count", [values, weights]):
    values = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        values, name="values")
    if weights is not None:
      if not isinstance(weights, sparse_tensor.SparseTensor):
        weights = ragged_tensor.convert_to_tensor_or_ragged_tensor(
            weights, name="weights")

    if weights is not None and binary_output:
      raise ValueError("Arguments `binary_output` and `weights` are mutually "
                       "exclusive. Please specify only one.")

    if axis is None:
      axis = 0

    if axis not in [0, -1]:
      raise ValueError(f"Unsupported value for argument axis={axis}. Only 0 and"
                       " -1 are currently supported.")

    minlength_value = minlength if minlength is not None else -1
    maxlength_value = maxlength if maxlength is not None else -1

    if axis == 0:
      if weights is not None:
        weights = validate_ragged_weights(values, weights)
      values = values.values

    if isinstance(values, ragged_tensor.RaggedTensor):
      weights = validate_ragged_weights(values, weights)
      c_ind, c_val, c_shape = gen_count_ops.ragged_count_sparse_output(
          values.row_splits,
          values.values,
          weights,
          minlength=minlength_value,
          maxlength=maxlength_value,
          binary_output=binary_output)
    else:
      weights = bincount_ops.validate_dense_weights(values, weights)
      c_ind, c_val, c_shape = gen_count_ops.dense_count_sparse_output(
          values,
          weights=weights,
          minlength=minlength_value,
          maxlength=maxlength_value,
          binary_output=binary_output)

    return sparse_tensor.SparseTensor(c_ind, c_val, c_shape)


def validate_ragged_weights(values, weights, dtype=None):
  # [FUNCTION DOC]
  # HELPER: weights preparation for the ragged bincount handlers in this module.
  # Inputs: values (ragged tensor being counted), weights (ragged tensor or nothing) and
  # dtype (optional dtype for the placeholder). If weights is nothing, returns an empty
  # constant tensor, of the given dtype if provided, else of the flat values dtype of
  # values. Otherwise weights must be a ragged tensor, else a value error is raised. If
  # its row splits are not the very same object as those of values, an equality
  # assertion on the row splits is added (failing at run time with an invalid-argument
  # error if they differ) and the flat values of weights are returned through an
  # identity op under that dependency. Returns the weights' flat values tensor.
  """Validates the passed weight tensor or creates an empty one."""
  if weights is None:
    if dtype:
      return array_ops.constant([], dtype=dtype)
    return array_ops.constant([], dtype=values.values.dtype)

  if not isinstance(weights, ragged_tensor.RaggedTensor):
    raise ValueError(
        "`weights` must be a RaggedTensor if `values` is a RaggedTensor. "
        f"Received argument weights={weights} of type: "
        f"{type(weights).__name__}.")

  checks = []
  if weights.row_splits is not values.row_splits:
    checks.append(
        check_ops.assert_equal(
            weights.row_splits,
            values.row_splits,
            message="'weights' and 'values' must have the same row splits."))
  if checks:
    with ops.control_dependencies(checks):
      weights = array_ops.identity(weights.values)
  else:
    weights = weights.values

  return weights
