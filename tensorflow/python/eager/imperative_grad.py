# [MODULE DOC]
# Backpropagation entry point for the eager gradient tape. Defines the VSpace named tuple (a bundle
# of vector-space callbacks used by the tape: aggregation, element counting, zeros/ones creation,
# zeros-like/ones-like and graph shape functions) and the imperative_grad function, which
# delegates gradient computation to the native tape extension. No import-time side effects.
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
"""Code for backpropagation using the tape utilities."""

import collections

from tensorflow.python import pywrap_tfe
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
from tensorflow.python.util import compat

VSpace = collections.namedtuple("VSpace", [
    "aggregate_fn", "num_elements_fn", "zeros_fn", "ones_fn",
    "zeros_like_fn", "ones_like_fn", "graph_shape_fn"
])


def imperative_grad(tape,
                    target,
                    sources,
                    output_gradients=None,
                    sources_raw=None,
                    unconnected_gradients=UnconnectedGradients.NONE):
  # [FUNCTION DOC]
  # Computes gradients of target with respect to sources using the tape. Inputs: tape (a gradient
  # tape object whose native tape is used), target (tensor or list of tensors), sources (list of
  # tensors), output_gradients (optional list of upstream gradients, null means use computed
  # ones), sources_raw (optional original source objects, needed only for zero-filling of
  # unconnected gradients), unconnected_gradients (enum or its string value; defaults to none,
  # meaning unconnected results are null; zero means zero tensors). Converts the last argument
  # to the enum, raising a value error if unknown. Returns the list of gradients, one per source.
  """Computes gradients from the imperatively defined tape on top of the stack.

  Works by filtering the tape, computing how many downstream usages are of each
  tensor and entry, and repeatedly applying backward functions until we have
  gradients for all sources.

  Args:
   tape: the gradient tape which stores the trace.
   target: either a Tensor or list of Tensors to be differentiated.
   sources: list of Tensors for which we want gradients
   output_gradients: if not None, a list of gradient provided for each Target,
     or None if we are to use the target's computed downstream gradient.
   sources_raw: if not None, a list of the source python objects from which the
     sources were generated. Should have the same length as sources. Only needs
     to be populated if unconnected_gradients is 'zero'.
   unconnected_gradients: determines the value returned if the target and
     sources are unconnected. When 'none' the value returned is None whereas
     when 'zero' a zero tensor in the same shape as the sources is returned.

  Returns:
   the gradient wrt each of the sources.

  Raises:
    ValueError: if the arguments are invalid.
    RuntimeError: if something goes wrong.
  """
  try:
    unconnected_gradients = UnconnectedGradients(unconnected_gradients)
  except ValueError:
    raise ValueError(
        "Unknown value for unconnected_gradients: %r" % unconnected_gradients)

  return pywrap_tfe.TFE_Py_TapeGradient(
      tape._tape,  # pylint: disable=protected-access
      target,
      sources,
      output_gradients,
      sources_raw,
      compat.as_str(unconnected_gradients.value))
