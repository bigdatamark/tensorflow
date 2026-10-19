# [MODULE DOC]
# Defines the abstract base class for composite tensors, tensor-like objects built from one or
# more dense tensors (for example sparse or ragged tensors), plus two helpers: one that
# recursively replaces composite tensors in a nested structure with their component tensors, and
# one that converts variables embedded in a composite tensor into plain tensors. Also contains
# a comment about the two internal encodings of composite tensors (component encoding and
# batchable tensor list encoding). Import-time side effects: the base class is registered for
# export under the internal public namespace; the wrapped native extension is imported.
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
"""Tensor-like objects that are composed from tf.Tensors."""

import abc

from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export


@tf_export("__internal__.CompositeTensor", v1=[])
class CompositeTensor(metaclass=abc.ABCMeta):
  # [CLASS DOC]
  # Abstract base class for tensor-like values that are composed of ordinary tensors. Subclasses
  # must supply a type specification describing the value; it knows how to decompose the value
  # into component tensors and rebuild it. Holds no state itself. Optional hooks allow shape
  # invariant conversion for loops, listing consumer operations, tracing-type lookup and
  # conversion of embedded variables to tensors. Uses an abstract-base metaclass and is exported
  # under the internal public API namespace.
  """Abstract base class for Tensor-like objects that are composed from Tensors.

  Each `CompositeTensor` can be decomposed into a structured collection of
  component `tf.Tensor`s, and reconstructed from those components.

  The `tensorflow.python.util.nest` module has support for treating composite
  tensors as structure, which makes it easy to flatten and reconstruct
  composite tensors (or larger structures that contain composite tensors).
  E.g.:

  ```python
  ct = ...  # Create a composite tensor.
  flat_list_of_tensors = nest.flatten(ct, expand_composites=True)
  transformed_list_of_tensors = ...  # do something with the flat tensors.
  result = nest.pack_sequence_as(ct, transformed_list_of_tensors,
                                 expand_composites=True)
  ```
  """

  @abc.abstractproperty
  def _type_spec(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: declared with the deprecated abstract-property decorator, and the base
    # implementation does not return a spec; it always raises not-implemented.
    # Abstract property that subclasses override to return the type specification describing
    # this value. Takes no inputs. In this base class it raises a not-implemented error whose
    # message names the concrete class. No side effects.
    """A `TypeSpec` describing the type of this value."""
    raise NotImplementedError(f"{type(self).__name__}._type_spec()")

  def _shape_invariant_to_type_spec(self, shape):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring describes returning a nested structure of shape invariants, but
    # the base implementation never returns; it always raises not-implemented.
    # Backwards-compatibility hook used by while-loops to turn a shape invariant into a type
    # specification. Input: shape, a tensor shape or nothing (meaning use the default invariant
    # derived from the current value). Subclasses may override; this base version raises a
    # not-implemented error naming the concrete class. No side effects.
    """Returns a TypeSpec given a shape invariant (used by `tf.while_loop`).

    Args:
      shape: A `tf.TensorShape` object.  The shape invariant for this
        `CompositeTensor`, or `None` if a default shape invariant should be used
        (based on the value of this `CompositeTensor`).

    Returns:
      A nested structure whose values are `tf.TensorShape` objects, specifying
      the shape invariants for the tensors that comprise this `CompositeTensor`.
    """
    # New TypeSpec subclasses generally do not need to implement this --
    # this method is used for backwards compatibility.  Users of tf.while_loop
    # can specify a type by passing in TypeSpec instead.
    raise NotImplementedError(
        f"{type(self).__name__}._shape_invariant_to_type_spec")

  def _consumers(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it raises a runtime error when executing eagerly, but this
    # code raises nothing itself; any such error would come from the component tensors' own
    # consumer lookup, and components without a graph are silently skipped.
    # Returns a list of the operations that consume any component tensor of this value. Flattens
    # the value into its components, asks each graph-backed component for its consumers, flattens
    # the results and removes duplicates (so order is unspecified). Takes no inputs; no side effects.
    """Returns a list of `Operation`s that consume this `CompositeTensor`.

    Returns:
      A list of `Operation`s.

    Raises:
      RuntimeError: If this method is called while executing eagerly.
    """
    consumers = nest.flatten([
        component.consumers()
        for component in nest.flatten(self, expand_composites=True)
        if getattr(component, "graph", None) is not None
    ])
    return list(set(consumers))

  def __tf_tracing_type__(self, context):
    # [FUNCTION DOC]
    # Protocol hook used by function tracing to obtain this value's tracing type. Input:
    # context, the tracing context object. Delegates to the type specification's own method of
    # the same name and returns its result. No side effects.
    return self._type_spec.__tf_tracing_type__(context)

  def _convert_variables_to_tensors(self):
    # [FUNCTION DOC]
    # Hook that subclasses override to replace embedded resource variables with plain tensors.
    # Takes no inputs. The default implementation returns this same composite tensor unchanged.
    # No side effects.
    """Converts ResourceVariable components to Tensors.

    Override this method to explicitly convert ResourceVariables embedded in the
    CompositeTensor to Tensors. By default, it returns the CompositeTensor
    unchanged.

    Returns:
      A CompositeTensor with all its ResourceVariable components converted to
      Tensors.
    """
    return self


def replace_composites_with_components(structure):
  # [FUNCTION DOC]
  # Recursively replaces every composite tensor inside a nested structure with its components.
  # Input: structure, any nest-compatible structure (or a single value). A composite tensor is
  # decomposed via its type specification and the components are processed recursively (so
  # nested composites are also expanded); a non-nested value is returned as is; other nested
  # structures are mapped element by element without expanding composites at the map level.
  # Returns a new structure free of composite tensors. No side effects.
  """Recursively replaces CompositeTensors with their components.

  Args:
    structure: A `nest`-compatible structure, possibly containing composite
      tensors.

  Returns:
    A copy of `structure`, where each composite tensor has been replaced by
    its components.  The result will contain no composite tensors.
    Note that `nest.flatten(replace_composites_with_components(structure))`
    returns the same value as `nest.flatten(structure)`.
  """
  if isinstance(structure, CompositeTensor):
    return replace_composites_with_components(
        structure._type_spec._to_components(structure))  # pylint: disable=protected-access
  elif not nest.is_nested(structure):
    return structure
  else:
    return nest.map_structure(
        replace_composites_with_components, structure, expand_composites=False)


def convert_variables_to_tensors(composite_tensor):
  # [FUNCTION DOC]
  # Module-level wrapper that calls the composite tensor's own variable-to-tensor conversion
  # hook and returns its result. Input: composite_tensor, an object providing that hook (a
  # composite tensor). Raises an attribute error if the object lacks the hook. No side effects.
  return composite_tensor._convert_variables_to_tensors()  # pylint: disable=protected-access


# @TODO(edloper): Can we replace convert_to_tensor_or_xyz with just
# convert_to_tensor_or_composite?  Alternatively, should composite tensors
# register a dispatch override for tf.convert_to_tensor?

# Note about the internal encoding of composite tensors when they are "lowered"
# from Python objects to tensors. The usual encoding is "component encoding"
# which uses the dense tensors that represent a composite tensor.
# A second encoding, "batchable tensor list encoding", is used by datasets
# and map_fn which in addition to supporting batching also can use ops
# for encoding and decoding, e.g. for encoding/decoding to/from a
# single variant that represents a composite tensor. Some internal properties
# for type specs for composite tensors use `flat` as a nickname for
# "batchable tensor list encoding". (e.g. `flat_tensor_specs`).
