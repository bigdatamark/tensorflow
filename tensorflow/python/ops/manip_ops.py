# [MODULE DOC]
# Public wrapper module for tensor manipulation operators; currently exposes only the roll
# operation, a thin wrapper over the generated manip ops. Import-time side effects: exports the
# function to the public API (as roll and, for version 1, also manip.roll) and copies the
# generated op's documentation onto the wrapper.
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
"""Operators for manipulating tensors.

API docstring: tensorflow.manip
"""

from tensorflow.python.ops import gen_manip_ops as _gen_manip_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export


# pylint: disable=protected-access
@tf_export('roll', v1=['roll', 'manip.roll'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('manip.roll')
def roll(input, shift, axis, name=None):  # pylint: disable=redefined-builtin
  # [FUNCTION DOC]
  # KNOWN ISSUE: the function has no docstring of its own; its documentation is assigned after
  # the definition from the generated op's documentation at import time.
  # Rolls a tensor: elements shifted beyond the last position wrap around to the first. Inputs:
  # input, the tensor to roll; shift, integer scalar or 1-D tensor giving the shift amount for
  # each axis (positive moves toward higher indices); axis, integer scalar or 1-D tensor of the
  # same length as shift naming the axes to shift; name, optional operation name. Returns a
  # tensor with the same shape and type as input. Errors for invalid shapes or axes come from
  # the underlying op. Decorated for dispatch support, and the manip.roll endpoint is deprecated.
  return _gen_manip_ops.roll(input, shift, axis, name)


roll.__doc__ = _gen_manip_ops.roll.__doc__
# pylint: enable=protected-access
