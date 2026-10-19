# [MODULE DOC]
# Miscellaneous AutoGraph utilities: alias_tensors, which wraps tensor arguments in identity
# operations, and get_range_len, which computes the number of elements of a range from tensor
# start, limit and delta values. Used by generated code and AutoGraph operators. alias_tensors is
# re-exported by the utils package. No import-time side effects.
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
"""Miscellaneous utilities that don't fit anywhere else."""

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops


def alias_tensors(*args):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says it returns the same as the arguments, but with several
  # arguments it returns a lazy generator rather than a tuple (callers must unpack it).
  # Takes one or more arguments. Each tensor is replaced by an identity operation on it; other
  # values (variables, strings, lists) are returned unchanged. A single argument is returned
  # directly, several are returned as a generator. Raises a value error when called with no
  # arguments.
  """Wraps any Tensor arguments with an identity op.

  Any other argument, including Variables, is returned unchanged.

  Args:
    *args: Any arguments. Must contain at least one element.

  Returns:
    Same as *args, with Tensor instances replaced as described.

  Raises:
    ValueError: If args doesn't meet the requirements.
  """

  def alias_if_tensor(a):
    # [FUNCTION DOC]
    # HELPER: returns an identity of a if it is a tensor, otherwise a unchanged.
    return array_ops.identity(a) if isinstance(a, tensor.Tensor) else a

  # TODO(mdan): Recurse into containers?
  # TODO(mdan): Anything we can do about variables? Fake a scope reuse?
  if len(args) > 1:
    return (alias_if_tensor(a) for a in args)
  elif len(args) == 1:
    return alias_if_tensor(args[0])

  raise ValueError('at least one argument required')


def get_range_len(start, limit, delta):
  # [FUNCTION DOC]
  # Computes the element count of a range with the given start, limit and step (delta), as a
  # tensor. Takes the distance limit minus start as a tensor, floor-divides by delta and adds
  # one if the remainder is nonzero, then clamps at zero. delta must be nonzero (division by zero
  # otherwise); negative steps work through floor-division semantics. Returns a tensor of the same
  # integer type as the distance.
  dist = ops.convert_to_tensor(limit - start)
  unadjusted_len = dist // delta
  adjustment = math_ops.cast(
      gen_math_ops.not_equal(dist % delta,
                             array_ops.zeros_like(unadjusted_len)), dist.dtype)
  final_len = unadjusted_len + adjustment
  return gen_math_ops.maximum(final_len, array_ops.zeros_like(final_len))
