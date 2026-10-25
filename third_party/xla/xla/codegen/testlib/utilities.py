# [MODULE DOC]
# HELPER: boilerplate utilities for kernel tests, shared by the backend test libraries.
# Provides functions that create XLA literals (host-side typed arrays) from numpy values, either
# scalars or arrays with an optional memory layout, and re-exports the native function that
# returns the operand count of an HLO opcode. No import-time side effects other than importing the
# native extension.
# Copyright 2024 The OpenXLA Authors.
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
"""Boilerplate utilities for kernel testing."""

from typing import Optional

import numpy as np

from xla.codegen.testlib import _extension
from xla.python import xla_extension


def create_scalar_literal(value, dtype: np.dtype) -> xla_extension.Literal:
  # [FUNCTION DOC]
  # HELPER: creates a rank-zero literal for tests.
  # Inputs: value, a Python or numpy scalar; dtype, the numpy element type of the literal.
  # Builds a scalar shape of that type, allocates a literal of it and copies the value into the
  # literal's buffer (conversion errors from numpy propagate). Returns the literal.
  shape = xla_extension.Shape.scalar_shape(dtype)
  literal = xla_extension.Literal(shape)
  np.copyto(np.asarray(literal), value)
  return literal


def create_literal_from_np(
    array: np.ndarray, layout: Optional[list[int]] = None
) -> xla_extension.Literal:
  # [FUNCTION DOC]
  # HELPER: converts a numpy array into an XLA literal for tests.
  # KNOWN ISSUE: for zero-dimensional input the optional layout argument is silently ignored.
  # Inputs: array, a numpy array or numpy scalar; layout, optional list of ints giving the
  # minor-to-major dimension order (default none, meaning the default layout). Rank-zero input
  # is delegated to the scalar helper. Otherwise builds an array shape from the dtype, dimensions
  # and layout, allocates a literal and copies the array into it. Returns the literal.
  # Errors from shape creation (for example an invalid layout) propagate.
  if np.ndim(array) == 0:
    return create_scalar_literal(array.item(), array.dtype)

  shape = xla_extension.Shape.array_shape(array.dtype, array.shape, layout)
  literal = xla_extension.Literal(shape)
  np.copyto(np.asarray(literal), array)
  return literal


# Intentionally rexport-ed to be avalable in the public API.
opcode_arity = _extension.opcode_arity
