# [MODULE DOC]
# AutoGraph tensor-kind predicates: small functions that classify a value as a dense tensor,
# tensor array, tensor list or range tensor. Used by AutoGraph's converted-code runtime to pick
# the right handling for values. No import-time side effects; not a helper-only module but
# consumed by other autograph utilities.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""This module defines tensor utilities not found in TensorFlow.

The reason these utilities are not defined in TensorFlow is because they may
not be not fully robust, although they work in the vast majority of cases. So
we define them here in order for their behavior to be consistently verified.
"""

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import tensor_array_ops


def is_dense_tensor(t):
  # [FUNCTION DOC]
  # Returns true if the value is a TensorFlow tensor-like type that is not a sparse tensor.
  # Input t: any value. Output: boolean. No side effects, no exceptions.
  # TODO(mdan): Resolve this inconsistency.
  return (tensor_util.is_tf_type(t) and
          not isinstance(t, sparse_tensor.SparseTensor))


def is_tensor_array(t):
  # [FUNCTION DOC]
  # Returns true if the value is an instance of the TensorArray class. Input t: any value.
  # Output: boolean.
  return isinstance(t, tensor_array_ops.TensorArray)


def is_tensor_list(t):
  # [FUNCTION DOC]
  # Heuristic check for a TensorFlow list tensor: true if the value is a tensor-like type whose
  # element type is the variant type and whose shape has no dimensions (scalar or unknown rank).
  # Input t: any value. Output: boolean. It can misclassify other scalar variant tensors.
  # TODO(mdan): This is just a heuristic.
  # With TF lacking support for templated types, this is unfortunately the
  # closest we can get right now. A dedicated op ought to be possible to
  # construct.
  return (tensor_util.is_tf_type(t) and t.dtype == dtypes.variant and
          not t.shape.ndims)


def is_range_tensor(t):
  # [FUNCTION DOC]
  # Best-effort check that a value is a tensor produced by a range operation: true if it is a
  # tensor-like type with an op attribute whose op type is Range. Input t: any value.
  # Output: boolean.
  """Returns True if a tensor is the result of a tf.range op. Best effort."""
  return tensor_util.is_tf_type(t) and hasattr(t, 'op') and t.op.type == 'Range'
