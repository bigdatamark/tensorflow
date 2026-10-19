# [MODULE DOC]
# HELPER: glue module connecting tf.data iterators to AutoGraph.
# Provides checks that a default value matches an iterator element structure, an override of
# the built-in next for tf iterators, and a function that registers the overrides (next and
# for-loop handling for owned iterators) in AutoGraph registries. Importing has no side effects;
# registration happens only when the register function is called.
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Autograph specific overrides for tf.data.ops."""
import functools

import numpy as np

from tensorflow.python.autograph.operators import control_flow
from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import cond
from tensorflow.python.util import nest


# TODO(mdan): These checks should be easier. Fix the nest API.
def _verify_spec_compatible(input_name, spec_name, input_, spec):
  # [FUNCTION DOC]
  # HELPER: dtype check used by the structure verifier for the default argument of next.
  # KNOWN ISSUE: the none check tests the built-in input function instead of the input_
  # parameter, so it never fires and a none value is not rejected here (it later fails with a
  # dtype mismatch error instead).
  # Inputs: input_name and spec_name (strings for messages); input_ (value to check); spec (a
  # tensor spec, asserted to be one). Python scalars, strings and arrays are converted to
  # tensors first. Raises a TypeError if the value's dtype differs from the spec's dtype.
  # Returns nothing. The docstring says ValueError, but dtype mismatch raises TypeError.
  """Verifies that a symbol has a type compatible vith a given spec.

  Here, compatibility is viewed in the general TensorFlow sense: that the dtypes
  are the same after implicit conversion, if both are tensors.

  This verifier ensures consistent treatment of types across AutoGraph.

  Args:
    input_name: A name to use for `input_` in error messages.
    spec_name: A name to use for `spec` in error messages.
    input_: Any, value to verify.
    spec: TypeSpec that `input_` must be compatible with.

  Raises:
    ValueError if the two types have been determined not to be compatible.
  """
  assert isinstance(spec, tensor_spec.TensorSpec)
  if input is None:
    # TODO(mdan): raise from None when switching to Py3.
    raise ValueError("{} cannot be None".format(input_name))

  # TODO(mdan): Use TensorCompatible when ready.
  if isinstance(input_, (bool, int, float, str, np.ndarray)):
    input_ = tensor_conversion.convert_to_tensor_v2(input_)

  input_dtype = getattr(input_, "dtype", None)

  if input_dtype != spec.dtype:
    input_dtype_str = "no dtype" if input_dtype is None else str(input_dtype)

    raise TypeError(
        "{} must have the same dtype as {}. Expected {}, got {}".format(
            input_name, spec_name, spec.dtype, input_dtype_str
        )
    )


def _verify_structure_compatible(input_name, spec_name, input_, spec):
  # [FUNCTION DOC]
  # HELPER: structure-level check for the default argument of next on tf iterators.
  # KNOWN ISSUE: docstring says it raises ValueError, but structure mismatches are re-raised as
  # TypeError (and dtype mismatches also raise TypeError).
  # Inputs: input_name and spec_name (strings for messages); input_ (possibly nested value);
  # spec (nested structure of specs). Asserts both have the same nested structure (including
  # composite types), then applies the dtype check to each leaf pair. Returns nothing.
  """Verifies that possibly-structured symbol has types compatible vith another.

  See _verify_spec_compatible for a more concrete meaning of "compatible".
  Unspec _verify_spec_compatible, which handles singular Tensor-spec objects,
  verify_structures_compatible can process structures recognized by tf.nest.

  Args:
    input_name: A name to use for `input_` in error messages.
    spec_name: A name to use for `spec` in error messages.
    input_: Any, value to verify. May, but doesn't need to, be a structure.
    spec: Any, value that `input_` must be compatible with. May, but doesn't
      need to, be a structure.

  Raises:
    ValueError if the two types have been determined not to be compatible.
  """
  try:
    nest.assert_same_structure(input_, spec, expand_composites=True)
  except (ValueError, TypeError) as e:
    raise TypeError(
        "{} must have the same element structure as {}.\n\n{}".format(
            input_name, spec_name, str(e)
        )
    ) from e

  nest.map_structure(
      functools.partial(_verify_spec_compatible, input_name, spec_name), input_,
      spec)


def _next_tf_iterator(iterator, default=py_builtins.UNSPECIFIED):
  # [FUNCTION DOC]
  # HELPER: AutoGraph replacement for the built-in next when applied to a tf iterator.
  # Inputs: iterator (an owned tf.data iterator); default (optional; unspecified marker by
  # default). Without a default, calls normal next, which raises a runtime out-of-range style
  # error at the end. With a default, gets the next element as an optional, verifies the default
  # matches the element structure and dtypes (TypeError otherwise), and returns a graph
  # conditional that yields the element if present, else the default.
  if default is py_builtins.UNSPECIFIED:
    # Without a default, fall back to the "normal" behavior which raises
    # a runtime exception.
    return next(iterator)
  opt_iterate = iterator.get_next_as_optional()
  _verify_structure_compatible(
      "the default argument", "the iterate", default, iterator.element_spec
  )
  return cond.cond(
      opt_iterate.has_value(), opt_iterate.get_value, lambda: default
  )


def register_overrides():
  # [FUNCTION DOC]
  # HELPER: installs the AutoGraph overrides for tf.data iterators.
  # Takes no inputs, returns nothing. Side effects: registers the tf-iterator next override in
  # the AutoGraph next registry and the tf-iterator for-loop handler in the for-loop registry,
  # both keyed on the owned iterator class. Mutates global AutoGraph state.
  py_builtins.next_registry.register(
      iterator_ops.OwnedIterator, _next_tf_iterator
  )
  control_flow.for_loop_registry.register(
      iterator_ops.OwnedIterator, control_flow._tf_iterator_for_stmt  # pylint: disable=protected-access
  )
