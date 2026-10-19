# [MODULE DOC]
# HELPER: helper module for forwardprop.py and function-building code (forward accumulators).
# Small utilities for forward-mode automatic differentiation state. It is kept separate from the
# main forward-propagation module so that function-building code can use these utilities without
# a circular dependency. Exposes a named tuple type holding packed accumulator state, a function
# that packs the current jvp state for a list of tensors, and a context manager that temporarily
# pushes fresh transient accumulator state. All real work is delegated to the native
# tensorflow-eager extension. No import-time side effects.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for managing forward accumulators.

A separate file from forwardprop.py so that functions can use these utilities.
"""

import collections
import contextlib

from tensorflow.python import pywrap_tfe


class TangentInfo(
    collections.namedtuple("TangentInfo", ["indices", "tangents"])):
  # [CLASS DOC]
  # Named tuple with two fields, indices and tangents, describing packed forward accumulator
  # state. It is the return type of pack_tangents. Indices is a sequence (one entry per active
  # accumulator) of pairs of integer positions, tangents is a flat list of tangent tensors.
  # Immutable once created; defaults produce empty values.
  """Packed forward accumulator state. The return value of `pack_tangents`."""

  def __new__(cls, indices=None, tangents=None):
    # [FUNCTION DOC]
    # Constructor for the named tuple. Both arguments are optional: a missing indices value
    # becomes an empty tuple and a missing tangents value becomes an empty list. Returns the new
    # instance; raises nothing and has no side effects.
    if indices is None:
      indices = ()
    if tangents is None:
      tangents = []
    return super(TangentInfo, cls).__new__(cls, indices, tangents)


def pack_tangents(tensors):
  # [FUNCTION DOC]
  # Asks the native extension to pack the jvp state of all active forward accumulators for the
  # given flat list of tensors. Returns a TangentInfo holding the indices structure and the
  # list of tangent tensors to append to the inputs. Reads global accumulator state; raises
  # whatever the native call raises.
  """Packs forward accumulator state into a TangentInfo tuple.

  Args:
    tensors: A flat list of Tensors to pack forward accumulator state for.

  Returns:
    A tuple of (indices, tangents):
      indices: A sequence of sequences of two-element tuples. Each forward
        accumulator is represented as a sequence of tuples with (primal_index,
        jvp_index). Both integers index into the concatenated `tensors + jvps`
        array.
      tangents: A flat list of Tensors. Best interpreted as a sequence to be
        appended to `tensors`.
  """
  return TangentInfo(*pywrap_tfe.TFE_Py_PackJVPs(tensors))


@contextlib.contextmanager
def push_forwardprop_state():
  # [FUNCTION DOC]
  # Context manager (takes no arguments, yields nothing). On entry it pushes a fresh transient
  # state onto the native forward accumulator stack, and on exit (even after an error) it pops
  # it. While pushed, accumulators that are mid-computation can process operations they would
  # otherwise ignore, which is needed when building forward-mode versions of functions.
  # Side effect: mutates global native accumulator state.
  """Temporarily push or pop transient state for accumulators in the active set.

  Allows an accumulator which is currently processing an operation to
  temporarily reset its state. This is useful when building forwardprop versions
  of functions, where an accumulator will trigger function building and then
  must process captured symbolic tensors while building it. Without pushing and
  popping, accumulators ignore operations executed as a direct result of their
  own jvp computations.

  Yields:
    None (used for its side effect).
  """
  try:
    pywrap_tfe.TFE_Py_ForwardAccumulatorPushState()
    yield
  finally:
    pywrap_tfe.TFE_Py_ForwardAccumulatorPopState()
