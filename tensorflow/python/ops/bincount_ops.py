# [MODULE DOC]
# Implements the bincount operation (counting occurrences of each non-negative integer value),
# exported publicly under the math namespace in its current form and, with a reduced legacy
# signature, as the deprecated top-level and version 1 endpoint. Also contains a small helper that
# validates or creates the weights tensor passed to the dense bincount kernel. The functions are
# thin wrappers over the generated dense bincount kernel that compute the output size from the
# data and the optional minimum and maximum lengths. Import-time side effects: registers the
# public exports, dispatch support and the deprecation notice for the legacy endpoint.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""bincount ops."""

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export


@tf_export("math.bincount", v1=[])
@dispatch.add_dispatch_support
def bincount(arr,
             weights=None,
             minlength=None,
             maxlength=None,
             dtype=dtypes.int32,
             name=None,
             axis=None,
             binary_output=False):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says only an invalid-argument error is raised (for negative input)
  # and that ragged and sparse inputs with rank 2 are supported, but this body also raises value
  # errors (weights together with binary output, or an unsupported axis) and handles only dense
  # conversion itself; the ragged and sparse cases and the negative-value check rely on dispatch
  # and on the kernel. Non-integer inputs are silently cast to 32-bit integers (truncating).
  # Counts occurrences of each integer value. Inputs: arr, values to count (anything convertible
  # to a tensor); weights, optional same-shape tensor of increments (default none, meaning each
  # occurrence counts one); minlength and maxlength, optional lower and upper bounds on the output
  # length (values at or above the maximum are skipped); dtype, output type when no weights are
  # given (default 32-bit integer); name, optional name scope (default "bincount"); axis, either
  # none or 0 (flatten everything) or -1 (count per row, input is not flattened); binary_output,
  # if true emit one instead of counts (default false). Raises a value error if both weights and
  # binary output are requested, if axis is not 0, -1 or none, or if weights is not a tensor
  # after conversion. Computes the output size as the largest value plus one (zero for an empty
  # input), then raises it to minlength and lowers it to maxlength when given; flattens input
  # and weights when axis is 0; and calls the dense bincount kernel. Returns the tensor of counts
  # (dtype of weights if given, else the requested dtype). Adds ops to the current graph only.
  """Counts the number of occurrences of each value in an integer array.

  If `minlength` and `maxlength` are not given, returns a vector with length
  `tf.reduce_max(arr) + 1` if `arr` is non-empty, and length 0 otherwise.

  >>> values = tf.constant([1,1,2,3,2,4,4,5])
  >>> tf.math.bincount(values)
  <tf.Tensor: ... numpy=array([0, 2, 2, 1, 2, 1], dtype=int32)>

  Vector length = Maximum element in vector `values` is 5. Adding 1, which is 6
                  will be the vector length.

  Each bin value in the output indicates number of occurrences of the particular
  index. Here, index 1 in output has a value 2. This indicates value 1 occurs
  two times in `values`.

  **Bin-counting with weights**

  >>> values = tf.constant([1,1,2,3,2,4,4,5])
  >>> weights = tf.constant([1,5,0,1,0,5,4,5])
  >>> tf.math.bincount(values, weights=weights)
  <tf.Tensor: ... numpy=array([0, 6, 0, 1, 9, 5], dtype=int32)>

  When `weights` is specified, bins will be incremented by the corresponding
  weight instead of 1. Here, index 1 in output has a value 6. This is the
  summation of `weights` corresponding to the value in `values` (i.e. for index
  1, the first two values are 1 so the first two weights, 1 and 5, are
  summed).

  There is an equivilance between bin-counting with weights and
  `unsorted_segement_sum` where `data` is the weights and `segment_ids` are the
  values.

  >>> values = tf.constant([1,1,2,3,2,4,4,5])
  >>> weights = tf.constant([1,5,0,1,0,5,4,5])
  >>> tf.math.unsorted_segment_sum(weights, values, num_segments=6).numpy()
  array([0, 6, 0, 1, 9, 5], dtype=int32)

  On GPU, `bincount` with weights is only supported when XLA is enabled
  (typically when a function decorated with `@tf.function(jit_compile=True)`).
  `unsorted_segment_sum` can be used as a workaround for the non-XLA case on
  GPU.

  **Bin-counting matrix rows independently**

  This example uses `axis=-1` with a 2 dimensional input and returns a
  `Tensor` with bincounting where axis 0 is **not** flattened, i.e. an
  independent bincount for each matrix row.

  >>> data = np.array([[1, 2, 3, 0], [0, 0, 1, 2]], dtype=np.int32)
  >>> tf.math.bincount(data, axis=-1)
  <tf.Tensor: shape=(2, 4), dtype=int32, numpy=
    array([[1, 1, 1, 1],
           [2, 1, 1, 0]], dtype=int32)>

  **Bin-counting with binary_output**

  This example gives binary output instead of counting the occurrence.

  >>> data = np.array([[1, 2, 3, 0], [0, 0, 1, 2]], dtype=np.int32)
  >>> tf.math.bincount(data, axis=-1, binary_output=True)
  <tf.Tensor: shape=(2, 4), dtype=int32, numpy=
    array([[1, 1, 1, 1],
           [1, 1, 1, 0]], dtype=int32)>

  **Missing zeros in SparseTensor**

  Note that missing zeros (implict zeros) in SparseTensor are **NOT** counted.
  This supports cases such as `0` in the values tensor indicates that index/id
  `0`is present and a missing zero indicates that no index/id is present.

  If counting missing zeros is desired, there are workarounds.
  For the `axis=0` case, the number of missing zeros can computed by subtracting
  the number of elements in the SparseTensor's `values` tensor from the
  number of elements in the dense shape, and this difference can be added to the
  first element of the output of `bincount`. For all cases, the SparseTensor
  can be converted to a dense Tensor with `tf.sparse.to_dense` before calling
  `tf.math.bincount`.

  Args:
    arr: A Tensor, RaggedTensor, or SparseTensor whose values should be counted.
      These tensors must have a rank of 2 if `axis=-1`.
    weights: If non-None, must be the same shape as arr. For each value in
      `arr`, the bin will be incremented by the corresponding weight instead of
      1. If non-None, `binary_output` must be False.
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
    arr = tensor_conversion.convert_to_tensor_v2_with_dispatch(arr, name="arr")
    if weights is not None:
      weights = tensor_conversion.convert_to_tensor_v2_with_dispatch(
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
      if weights is not None:
        weights = array_ops.reshape(weights, [-1])
      arr = array_ops.reshape(arr, [-1])

    weights = validate_dense_weights(arr, weights, dtype)
    return gen_math_ops.dense_bincount(
        input=arr,
        size=output_size,
        weights=weights,
        binary_output=binary_output)


@tf_export(v1=["math.bincount", "bincount"])
@deprecation.deprecated_endpoints("bincount")
def bincount_v1(arr,
                weights=None,
                minlength=None,
                maxlength=None,
                dtype=dtypes.int32):
  # [FUNCTION DOC]
  # Deprecated legacy endpoint (top-level and version 1 namespaces) of the bincount operation.
  # Inputs: arr (integer tensor of non-negative values), weights (optional same-shape tensor),
  # minlength and maxlength (optional output length bounds), dtype (output type when no weights,
  # default 32-bit integer). Simply forwards these to the current bincount with its other options
  # left at defaults (no name, flatten all axes, counting output) and returns its result. Raises
  # whatever that function raises; no new side effects.
  """Counts the number of occurrences of each value in an integer array.

  If `minlength` and `maxlength` are not given, returns a vector with length
  `tf.reduce_max(arr) + 1` if `arr` is non-empty, and length 0 otherwise.
  If `weights` are non-None, then index `i` of the output stores the sum of the
  value in `weights` at each index where the corresponding value in `arr` is
  `i`.

  Args:
    arr: An int32 tensor of non-negative values.
    weights: If non-None, must be the same shape as arr. For each value in
      `arr`, the bin will be incremented by the corresponding weight instead of
      1.
    minlength: If given, ensures the output has length at least `minlength`,
      padding with zeros at the end if necessary.
    maxlength: If given, skips values in `arr` that are equal or greater than
      `maxlength`, ensuring that the output has length at most `maxlength`.
    dtype: If `weights` is None, determines the type of the output bins.

  Returns:
    A vector with the same dtype as `weights` or the given `dtype`. The bin
    values.
  """
  return bincount(arr, weights, minlength, maxlength, dtype)


def validate_dense_weights(values, weights, dtype=None):
  # [FUNCTION DOC]
  # HELPER: weights preparation step for the bincount functions.
  # KNOWN ISSUE: the error message mentions the values argument being a tensor, but the check
  # does not look at the values at all; any non-tensor weights are rejected.
  # Inputs: values, the tensor being counted (used only for its dtype); weights, a tensor or
  # nothing; dtype, optional output type. If weights is nothing, returns an empty constant tensor
  # of the given dtype (or of the values' dtype when no dtype is given) which tells the kernel to
  # count occurrences. Otherwise raises a value error unless weights is a tensor, and returns the
  # weights unchanged.
  """Validates the passed weight tensor or creates an empty one."""
  if weights is None:
    if dtype:
      return array_ops.constant([], dtype=dtype)
    return array_ops.constant([], dtype=values.dtype)

  if not isinstance(weights, tensor.Tensor):
    raise ValueError(
        "Argument `weights` must be a tf.Tensor if `values` is a tf.Tensor. "
        f"Received weights={weights} of type: {type(weights).__name__}")

  return weights
