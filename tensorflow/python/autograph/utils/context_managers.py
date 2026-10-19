# [MODULE DOC]
# Context managers used by AutoGraph-generated code. Exposes control_dependency_on_returns, which
# turns the return value(s) of a converted function into a TensorFlow control-dependency context
# (or a no-op context when there is nothing to depend on). Re-exported by the utils package.
# No import-time side effects.
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
"""Various context managers."""

import contextlib

from tensorflow.python.framework import ops
from tensorflow.python.ops import tensor_array_ops


def control_dependency_on_returns(return_value):
  # [FUNCTION DOC]
  # Builds a context manager that makes operations created inside it wait for return_value.
  # return_value may be nothing (then a context manager that does nothing is returned), a single
  # value, or a list or tuple of values; tensor arrays are replaced by their flow tensor, other
  # values are used as they are. Returns the TensorFlow control-dependencies context for the
  # tuple of values. Raises whatever TensorFlow raises for values that cannot be dependencies.
  """Create a TF control dependency on the return values of a function.

  If the function had no return value, a no-op context is returned.

  Args:
    return_value: The return value to set as control dependency.

  Returns:
    A context manager.
  """
  def control_dependency_handle(t):
    # [FUNCTION DOC]
    # HELPER: maps a value to something usable as a control dependency: a tensor array becomes its
    # flow tensor, anything else is returned unchanged.
    if isinstance(t, tensor_array_ops.TensorArray):
      return t.flow
    return t

  if return_value is None:
    return contextlib.contextmanager(lambda: (yield))()
  # TODO(mdan): Filter to tensor objects.
  if not isinstance(return_value, (list, tuple)):
    return_value = (return_value,)
  return_value = tuple(control_dependency_handle(t) for t in return_value)
  return ops.control_dependencies(return_value)
