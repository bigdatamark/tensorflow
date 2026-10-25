# [MODULE DOC]
# HELPER: utility module shared by the signal ops (the docstring still mentions the old contrib
# location). Provides gcd, a greatest-common-divisor op used by shape_ops.frame to pick the
# subframe size. Has no import-time side effects beyond imports.
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
"""Utility ops shared across tf.contrib.signal."""

import fractions  # gcd is here for Python versions < 3
import math  # Get gcd here for Python versions >= 3
import sys

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import while_loop


def gcd(a, b, name=None):
  # [FUNCTION DOC]
  # HELPER: greatest common divisor for frame size computations.
  # KNOWN ISSUE: the branch for old Python 2 versions (using the fractions module) is dead code in
  # a Python 3 only codebase. In the dynamic path the two inputs are assumed positive: a zero
  # second value ends the loop immediately returning the first.
  # Inputs: a and b, scalar integer tensors or values. Raises a value error (or shape error) if
  # either is non-scalar or non-integer. If both are statically known, returns a constant tensor
  # computed in Python; otherwise builds a graph loop with Euclid's algorithm (no backprop) and
  # returns the scalar result.
  """Returns the greatest common divisor via Euclid's algorithm.

  Args:
    a: The dividend. A scalar integer `Tensor`.
    b: The divisor. A scalar integer `Tensor`.
    name: An optional name for the operation.

  Returns:
    A scalar `Tensor` representing the greatest common divisor between `a` and
    `b`.

  Raises:
    ValueError: If `a` or `b` are not scalar integers.
  """
  with ops.name_scope(name, 'gcd', [a, b]):
    a = ops.convert_to_tensor(a)
    b = ops.convert_to_tensor(b)

    a.shape.assert_has_rank(0)
    b.shape.assert_has_rank(0)

    if not a.dtype.is_integer:
      raise ValueError('a must be an integer type. Got: %s' % a.dtype)
    if not b.dtype.is_integer:
      raise ValueError('b must be an integer type. Got: %s' % b.dtype)

    # TPU requires static shape inference. GCD is used for subframe size
    # computation, so we should prefer static computation where possible.
    const_a = tensor_util.constant_value(a)
    const_b = tensor_util.constant_value(b)
    if const_a is not None and const_b is not None:
      if sys.version_info.major < 3:
        math_gcd = fractions.gcd
      else:
        math_gcd = math.gcd
      return ops.convert_to_tensor(math_gcd(const_a, const_b))

    cond = lambda _, b: math_ops.greater(b, array_ops.zeros_like(b))
    body = lambda a, b: [b, math_ops.mod(a, b)]
    a, b = while_loop.while_loop(cond, body, [a, b], back_prop=False)
    return a
