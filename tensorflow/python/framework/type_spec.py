# [MODULE DOC]
# Defines the TypeSpec abstract base class (public type specification API) that describes the
# type of a TensorFlow value: its component encoding into tensors, its serialization, and the
# default equality, hashing, compatibility, subtype/supertype, placeholder and cast behavior
# derived from them. Also defines batching support: an abstract batch encoder interface, a legacy
# encoder that delegates to spec methods, and the batchable spec subclass, plus helper functions
# for encoding and decoding batchable values. Exposes the public function that infers a spec from a
# value and a registry of value-to-spec converters that other modules fill. Import-time side
# effects: exports public names, registers the spec class as serializable, and creates the empty
# converter registry. Core module of the framework package used by tf.function, tf.data and
# composite tensors.
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
"""Type specifications for TensorFlow APIs."""

import abc
import functools
from typing import Any, List, Optional, Sequence, Type
import warnings

import numpy as np

from tensorflow.core.function import trace_type
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import tf_logging as logging
# TODO(b/238903802): Remove dependency on nested_structure_coder.
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import core as core_types
from tensorflow.python.types import internal
from tensorflow.python.types import trace
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls


_CACHED_CMP_KEY = "_cached_cmp_key"  # Used by hashing and equality.
# Cache fixed, derived TypeSpec properties to avoid expensive recompute.
CACHED_FIXED_PROPERTIES = [_CACHED_CMP_KEY]


@tf_export("TypeSpec", v1=["TypeSpec", "data.experimental.Structure"])
class TypeSpec(
    internal.TypeSpec,
    trace.TraceType,
    trace_type.Serializable,
    metaclass=abc.ABCMeta,
):
  # [CLASS DOC]
  # Abstract base class describing a type of TensorFlow value (public name TypeSpec). Combines the
  # internal TypeSpec interface, the trace-type interface (used for function tracing, subtyping
  # and joining) and the serializable interface. Subclasses supply a "component encoding" (the
  # value decomposed into tensors or composite tensors, described by the component specs), a
  # "serialization" of the spec itself (a nested tuple of parameters), and conversion functions
  # between values and components. This base class derives default behavior for equality,
  # hashing, pickling, repr, compatibility, subtype checks, supertype joins, placeholders and
  # casting from those primitives. Uses slots limited to a cached comparison key (cached fixed
  # properties) so instances hold no other state in the base. Registered for the public API at
  # class creation, with a legacy alias for tf.data structure.
  """Specifies a TensorFlow value type.

  A `tf.TypeSpec` provides metadata describing an object accepted or returned
  by TensorFlow APIs.  Concrete subclasses, such as `tf.TensorSpec` and
  `tf.RaggedTensorSpec`, are used to describe different value types.

  For example, `tf.function`'s `input_signature` argument accepts a list
  (or nested structure) of `TypeSpec`s.

  Creating new subclasses of `TypeSpec` (outside of TensorFlow core) is not
  currently supported.  In particular, we may make breaking changes to the
  private methods and properties defined by this base class.

  Example:

  >>> spec = tf.TensorSpec(shape=[None, None], dtype=tf.int32)
  >>> @tf.function(input_signature=[spec])
  ... def double(x):
  ...   return x * 2
  >>> double(tf.constant([[1, 2], [3, 4]]))
  <tf.Tensor: shape=(2, 2), dtype=int32,
      numpy=array([[2, 4], [6, 8]], dtype=int32)>
  """
  # === Subclassing ===
  #
  # Each `TypeSpec` subclass must define:
  #
  #   * A "component encoding" for values.
  #   * A "serialization" for types.
  #
  # The component encoding for a value is a nested structure of `tf.Tensor`
  # or `CompositeTensor` that can be used by the `TypeSpec` to reconstruct
  # the value.  Each individual `TypeSpec` must use the same nested structure
  # for all values -- this structure is defined by the `component_specs`
  # attribute.  Decomposing values into components, and reconstructing them
  # from those components, should be inexpensive.  In particular, it should
  # *not* require any TensorFlow ops.
  #
  # The serialization for a `TypeSpec` is a nested tuple of values that can
  # be used to reconstruct the `TypeSpec`.  See the documentation for
  # `_serialize()` for more information.

  __slots__ = CACHED_FIXED_PROPERTIES

  @abc.abstractproperty
  def value_type(self):
    # [FUNCTION DOC]
    # Abstract property giving the Python class that all values compatible with this spec are
    # instances of. The base implementation raises a not-implemented error naming the class;
    # concrete subclasses must override it. No side effects.
    """The Python type for values that are compatible with this TypeSpec.

    In particular, all values that are compatible with this TypeSpec must be an
    instance of this type.
    """
    raise NotImplementedError("%s.value_type" % type(self).__name__)

  def is_subtype_of(self, other: trace.TraceType) -> bool:
    # [FUNCTION DOC]
    # Default subtype check for the trace-type interface. Input: other, a trace type. Returns false
    # at once if the two objects are not of exactly the same class. Otherwise compares the
    # serializations of both specs structurally: attributes that are themselves trace types must be
    # subtypes of the corresponding attribute, all other attributes must be equal. Returns true
    # only if every attribute passes; returns false if the serializations have mismatching
    # structure (value or type errors from the structure walk are swallowed). No side effects.
    """Returns True if `self` is a subtype of `other`.

    Implements the tf.types.experimental.func.TraceType interface.

    If not overridden by a subclass, the default behavior is to assume the
    TypeSpec is covariant upon attributes that implement TraceType and
    invariant upon rest of the attributes as well as the structure and type
    of the TypeSpec.

    Args:
      other: A TraceType object.
    """
    if type(self) is not type(other):
      return False

    is_subtype = True

    def check_attribute(attribute_self, attribute_other):
      # [FUNCTION DOC]
      # HELPER: per-attribute comparison callback used by the subtype check.
      # Inputs: the attribute from this spec and the matching attribute from the other spec.
      # Returns nothing; it clears the enclosing flag (is_subtype) when the attribute is a trace
      # type that is not a subtype of the other, or a non-trace-type attribute that differs.
      # Does nothing once the flag is already cleared.
      nonlocal is_subtype
      if not is_subtype:
        return

      if isinstance(attribute_self, trace.TraceType):
        if not attribute_self.is_subtype_of(attribute_other):
          is_subtype = False
          return
      else:
        if attribute_self != attribute_other:
          is_subtype = False

    try:
      # TODO(b/217959193): Replace _serialize with parameter decomposition.
      nest.map_structure(check_attribute, self._serialize(), other._serialize())  # pylint: disable=protected-access
    except (ValueError, TypeError):
      return False

    return is_subtype

  def most_specific_common_supertype(
      self, others: Sequence[trace.TraceType]) -> Optional["TypeSpec"]:
    # [FUNCTION DOC]
    # Default join for the trace-type interface. Input: others, a sequence of trace types.
    # Returns nothing/null if any of them is not exactly the same class as this spec. Otherwise
    # walks the serializations of this spec and all others in parallel: trace-type attributes are
    # joined recursively (failing if any join fails), other attributes must all be equal to this
    # one's. On success, rebuilds a spec of this class from the joined serialization and returns
    # it; returns null if no common supertype exists or the structures mismatch (value or type
    # errors are swallowed). No side effects.
    """Returns the most specific supertype TypeSpec  of `self` and `others`.

    Implements the tf.types.experimental.func.TraceType interface.

    If not overridden by a subclass, the default behavior is to assume the
    TypeSpec is covariant upon attributes that implement TraceType and
    invariant upon rest of the attributes as well as the structure and type
    of the TypeSpec.

    Args:
      others: A sequence of TraceTypes.
    """
    if any(type(self) is not type(other) for other in others):
      return None

    has_supertype = True

    def make_supertype_attribute(attribute_self, *attribute_others):
      # [FUNCTION DOC]
      # HELPER: per-attribute join callback used by the supertype computation.
      # Inputs: this spec's attribute and the matching attributes of the other specs.
      # Returns the joined attribute: the recursive supertype for trace-type attributes, or the
      # attribute itself if all others are equal to it. Returns nothing and clears the enclosing
      # flag (has_supertype) when no join exists; does nothing if the flag is already cleared.
      nonlocal has_supertype
      if not has_supertype:
        return

      if isinstance(attribute_self, trace.TraceType):
        attribute_supertype = attribute_self.most_specific_common_supertype(
            attribute_others)
        if attribute_supertype is None:
          has_supertype = False
          return
        return attribute_supertype
      else:
        if not all(attribute_self == attribute_other
                   for attribute_other in attribute_others):
          has_supertype = False
          return
        return attribute_self

    try:
      # TODO(b/217959193): Replace _serialize with parameter decomposition.
      serialized_supertype = nest.map_structure(
          make_supertype_attribute, self._serialize(),
          *(o._serialize() for o in others))  # pylint: disable=protected-access
    except (ValueError, TypeError):
      return None

    return self._deserialize(serialized_supertype) if has_supertype else None

  @classmethod
  def experimental_type_proto(cls) -> Type[struct_pb2.TypeSpecProto]:
    # [FUNCTION DOC]
    # Class method returning the protocol buffer message class used to serialize type specs
    # (the type spec proto). Takes no inputs besides the class; no side effects.
    """Returns the type of proto associated with TypeSpec serialization.

    Do NOT override for custom non-TF types.
    """
    return struct_pb2.TypeSpecProto

  @classmethod
  def experimental_from_proto(cls,
                              proto: struct_pb2.TypeSpecProto) -> "TypeSpec":
    # [FUNCTION DOC]
    # Class method that rebuilds a type spec from its serialized proto. Input: proto, a type spec
    # proto as produced by the as-proto method. Wraps it in a structured value message and decodes
    # it with the nested structure coder; returns the decoded spec. Decoding errors from the coder
    # propagate. The class the method is called on is not used for the result type.
    """Returns a TypeSpec instance based on the serialized proto.

    Do NOT override for custom non-TF types.

    Args:
      proto: Proto generated using 'experimental_as_proto'.
    """
    return nested_structure_coder.decode_proto(
        struct_pb2.StructuredValue(type_spec_value=proto))

  def experimental_as_proto(self) -> struct_pb2.TypeSpecProto:
    # [FUNCTION DOC]
    # Serializes this spec to a type spec proto by encoding it with the nested structure coder and
    # extracting the type spec field of the resulting structured value. Returns the proto; encoding
    # errors (for example for unregistered spec classes) propagate. No other side effects.
    """Returns a proto representation of the TypeSpec instance.

    Do NOT override for custom non-TF types.
    """
    return nested_structure_coder.encode_structure(self).type_spec_value

  @doc_controls.do_not_doc_inheritable
  def placeholder_value(self, placeholder_context):
    # [FUNCTION DOC]
    # Builds the stand-in value used when tracing a function with this spec. Input:
    # placeholder_context, an object carrying context for placeholder creation (including an
    # unnest-only flag). If the flag is set, returns this spec itself. Otherwise creates
    # placeholders for each component spec (recursively, via the structure mapping) and
    # reassembles them into a value using the from-components conversion. Returns that value.
    # May create graph placeholders as a side effect of the component specs. Not to be overridden.
    """Value used for tracing a function signature with this TraceType.

    WARNING: Do not override.

    Args:
      placeholder_context: A class container for context information when
        creating a placeholder value.

    Returns:
      A `CompositeTensor` placeholder whose components are recursively composed
        of placeholders themselves.
    """
    if placeholder_context.unnest_only:
      return self

    component_placeholders = nest.map_structure(
        lambda x: x.placeholder_value(placeholder_context),
        self._component_specs)
    return self._from_components(component_placeholders)

  @doc_controls.do_not_doc_inheritable
  def to_tensors(self, value):
    # [FUNCTION DOC]
    # Decomposes a value compatible with this spec into a flat list of tensors. Input: value.
    # Converts the value to its component structure and, pairing it with the component specs,
    # collects each component spec's tensors in order. Returns the flat list. Errors from the
    # component conversion propagate. Not to be overridden.
    """See TraceType base class for details. Do not override."""

    tensors = []
    nest.map_structure(
        lambda spec, v: tensors.extend(spec.to_tensors(v)),
        self._component_specs,
        self._to_components(value))
    return tensors

  @doc_controls.do_not_doc_inheritable
  def from_tensors(self, tensors):
    # [FUNCTION DOC]
    # Rebuilds a value from tensors. Input: tensors, an iterator/sequence of tensors that is
    # consumed in order by each component spec's own from-tensors conversion; the components are
    # then assembled with the from-components conversion. Returns the reconstructed value. Consumes
    # the passed tensors as a side effect (the same shared sequence is passed to each component).
    # Not to be overridden.
    """See TraceType base class for details. Do not override."""
    components = nest.map_structure(
        lambda spec: spec.from_tensors(tensors),
        self._component_specs
    )
    return self._from_components(components)

  @doc_controls.do_not_doc_inheritable
  def flatten(self):
    # [FUNCTION DOC]
    # Returns a flat list of the leaf specs (tensor specs) of this spec by concatenating the
    # flattened lists of every component spec, in structure order. No inputs, no side effects.
    # Not to be overridden.
    """See TraceType base class for details. Do not override."""
    flat = []
    nest.map_structure(
        lambda spec: flat.extend(spec.flatten()),
        self._component_specs)
    return flat

  @doc_controls.do_not_doc_inheritable
  def cast(self, value, casting_context):
    # [FUNCTION DOC]
    # Casts a value (or, if allowed, a spec) to this spec. Inputs: value, and casting_context,
    # an object with an allow-specs flag. If specs are allowed and value is itself a type spec,
    # asserts it is a subtype of this spec (assertion error with a message otherwise) and returns
    # this spec. Otherwise casts each component of the value with the matching component spec;
    # if any component changed, returns a new value rebuilt from the cast components, else returns
    # the original value object untouched. No other side effects.
    """See TraceType base class for details. Do not override."""
    if casting_context.allow_specs and isinstance(value, TypeSpec):
      assert value.is_subtype_of(self), f"Can not cast {value!r} to {self!r}"
      return self

    did_cast = False
    def cast_fn(spec, v):
      # [FUNCTION DOC]
      # HELPER: per-component cast callback used by the cast method.
      # Inputs: a component spec and the component value. Returns the value cast by the spec;
      # if the cast produced a different object, sets the enclosing did-cast flag.
      casted_v = spec.cast(v, casting_context)
      if casted_v is not v:
        nonlocal did_cast
        did_cast = True
      return casted_v

    cast_components = nest.map_structure(
        cast_fn, self._component_specs, self._to_components(value)
    )

    if did_cast:
      return self._from_components(cast_components)
    else:
      return value

  # TODO(b/225058047): Reconsider semantics.
  def is_compatible_with(self, spec_or_value):
    # [FUNCTION DOC]
    # Default compatibility check (discouraged in favor of the subtype and supertype methods).
    # Input: spec_or_value, a type spec or a value; a value is first converted to its spec.
    # Returns false if the classes differ, otherwise compares the two serializations with the
    # private recursive compatibility helper (shapes, tensor specs and dtypes use their own
    # compatibility test, everything else must be equal). Conversion errors for unsupported
    # values propagate. No side effects.
    """Returns true if `spec_or_value` is compatible with this TypeSpec.

    Prefer using "is_subtype_of" and "most_specific_common_supertype" wherever
    possible.

    Args:
      spec_or_value: A TypeSpec or TypeSpec associated value to compare against.
    """
    # === Subclassing ===
    # If not overridden by subclasses, the default behavior is to convert
    # `spec_or_value` to a `TypeSpec` (if it isn't already); and then to
    # consider two `TypeSpec`s compatible if they have the same type, and
    # the values returned by `_serialize` are compatible (where
    # `tf.TensorShape`, `tf.TensorSpec`, and `tf.DType` are checked for
    # compatibility using their `is_compatible_with` method; and all other
    # types are considered compatible if they are equal).
    if not isinstance(spec_or_value, TypeSpec):
      spec_or_value = type_spec_from_value(spec_or_value)
    if type(self) is not type(spec_or_value):
      return False
    return self.__is_compatible(self._serialize(), spec_or_value._serialize())  # pylint: disable=protected-access

  @deprecation.deprecated(None, "Use most_specific_common_supertype instead.")
  def most_specific_compatible_type(self, other: "TypeSpec") -> "TypeSpec":
    # [FUNCTION DOC]
    # Deprecated wrapper around the supertype join. Input: other, a type spec. Returns the most
    # specific common supertype of this spec and other. Raises a value error if no such spec
    # exists. Calling it emits a deprecation warning through the decorator.
    """Returns the most specific TypeSpec compatible with `self` and `other`.

    Deprecated. Please use `most_specific_common_supertype` instead.
    Do not override this function.

    Args:
      other: A `TypeSpec`.

    Raises:
      ValueError: If there is no TypeSpec that is compatible with both `self`
        and `other`.
    """
    result = self.most_specific_common_supertype([other])
    if result is None:
      raise ValueError("No TypeSpec is compatible with both %s and %s" %
                       (self, other))
    return result

  # TODO(b/226395276): Delete after removing usages.
  def _with_tensor_ranks_only(self) -> "TypeSpec":
    # [FUNCTION DOC]
    # HELPER: legacy utility for shape relaxation (marked for deletion once usages are removed).
    # Returns a spec of the same class whose serialization has every known-rank tensor shape
    # replaced by an unknown-dimension shape of the same rank; nested specs are relaxed
    # recursively and everything else is kept. Unknown-rank shapes are left unchanged.
    """Returns a TypeSpec compatible with `self`, with tensor shapes relaxed.

    Returns:
      A `TypeSpec` that is compatible with `self`, where any `TensorShape`
      information has been relaxed to include only tensor rank (and not
      the dimension sizes for individual axes).
    """

    # === Subclassing ===
    # If not overridden by a subclass, the default behavior is to serialize
    # this TypeSpec, relax any TensorSpec or TensorShape values, and
    # deserialize the result.

    def relax(value):
      # [FUNCTION DOC]
      # HELPER: mapping callback for the rank-only relaxation.
      # Input: one serialization element. Nested specs are relaxed recursively; tensor shapes with
      # known rank become all-unknown shapes of that rank; any other value is returned as is.
      if isinstance(value, TypeSpec):
        return value._with_tensor_ranks_only()  # pylint: disable=protected-access
      elif (isinstance(value, tensor_shape.TensorShape) and
            value.rank is not None):
        return tensor_shape.TensorShape([None] * value.rank)
      else:
        return value

    return self._deserialize(nest.map_structure(relax, self._serialize()))

  # TODO(b/206014848): Helper function to support logic that does not consider
  # Tensor name. Will be removed once load-bearing usages of Tensor name are
  # fixed.
  def _without_tensor_names(self) -> "TypeSpec":
    # [FUNCTION DOC]
    # HELPER: temporary utility for logic that must ignore tensor names.
    # KNOWN ISSUE: the docstring says tensor spec names are set to none, but this base
    # implementation only recurses into nested type specs; it never touches names itself, so the
    # name clearing relies entirely on subclasses (such as tensor specs) overriding this method.
    # Returns a spec of the same class rebuilt from its serialization with nested specs
    # processed recursively and other values unchanged.
    """Returns a TypeSpec compatible with `self`, with tensor names removed.

    Returns:
      A `TypeSpec` that is compatible with `self`, where the name of any
      `TensorSpec` is set to `None`.
    """

    # === Subclassing ===
    # If not overridden by a subclass, the default behavior is to serialize
    # this TypeSpec, set the TensorSpecs' names to None, and deserialize the
    # result.

    def rename(value):
      # [FUNCTION DOC]
      # HELPER: mapping callback for removing tensor names. Nested specs are processed
      # recursively with the same operation; every other value is returned unchanged.
      if isinstance(value, TypeSpec):
        return value._without_tensor_names()  # pylint: disable=protected-access
      return value

    return self._deserialize(nest.map_structure(rename, self._serialize()))

  # === Component encoding for values ===

  @abc.abstractmethod
  def _to_components(self, value):
    # [FUNCTION DOC]
    # Abstract: encodes a value compatible with this spec as a nested structure of tensors or
    # composite tensors matching the component specs. Input: value (compatibility is the caller's
    # responsibility). Must be cheap and must not run TensorFlow ops. The base implementation
    # raises a not-implemented error naming the class.
    """Encodes `value` as a nested structure of `Tensor` or `CompositeTensor`.

    Args:
      value: A value compatible with this `TypeSpec`.  (Caller is responsible
        for ensuring compatibility.)

    Returns:
      A nested structure of `tf.Tensor` or `tf.CompositeTensor` compatible with
      `self._component_specs`, which can be used to reconstruct `value`.
    """
    # === Subclassing ===
    # This method must be inexpensive (do not call TF ops).
    raise NotImplementedError("%s._to_components()" % type(self).__name__)

  @abc.abstractmethod
  def _from_components(self, components):
    # [FUNCTION DOC]
    # Abstract: reconstructs a value from a nested structure of tensors or composite tensors that
    # matches the component specs. Input: components (compatibility is the caller's
    # responsibility). Returns a compatible value. Must be cheap and run no TensorFlow ops. The
    # base implementation raises a not-implemented error naming the class.
    """Reconstructs a value from a nested structure of Tensor/CompositeTensor.

    Args:
      components: A nested structure of `tf.Tensor` or `tf.CompositeTensor`,
        compatible with `self._component_specs`.  (Caller is responsible for
        ensuring compatibility.)

    Returns:
      A value that is compatible with this `TypeSpec`.
    """
    # === Subclassing ===
    # This method must be inexpensive (do not call TF ops).
    raise NotImplementedError("%s._from_components()" % type(self).__name__)

  @abc.abstractproperty
  def _component_specs(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring example uses an assertion inside an anonymous function, which is
    # not valid syntax; it is illustrative only. Also, this is declared as an abstract property but
    # the failure message names it as a method call.
    # Abstract property: nested structure of specs describing what the to-components conversion
    # returns. The base implementation raises a not-implemented error naming the class.
    """A nested structure of TypeSpecs for this type's components.

    Returns:
      A nested structure describing the component encodings that are returned
      by this TypeSpec's `_to_components` method.  In particular, for a
      TypeSpec `spec` and a compatible value `value`:

      ```
      nest.map_structure(lambda t, c: assert t.is_compatible_with(c),
                         spec._component_specs, spec._to_components(value))
      ```
    """
    raise NotImplementedError("%s._component_specs()" % type(self).__name__)

  # === Tensor list encoding for values ===

  def _to_tensor_list(self, value) -> List["core_types.Symbol"]:
    # [FUNCTION DOC]
    # Encodes a value as a flat list of tensors. Input: value (assumed compatible). Default:
    # flattens the component encoding, expanding composite tensors into their parts. Subclasses
    # (notably batchable specs) may override to give a "boxed" encoding that can be batched.
    # Returns the list of tensors. No side effects.
    """Encodes `value` as a flat list of `tf.Tensor`.

    By default, this just flattens `self._to_components(value)` using
    `nest.flatten`.  However, subclasses may override this to return a
    different tensor encoding for values.  In particular, some subclasses
    of `BatchableTypeSpec` override this method to return a "boxed" encoding
    for values, which then can be batched or unbatched.  See
    `BatchableTypeSpec` for more details.

    Args:
      value: A value with compatible this `TypeSpec`.  (Caller is responsible
        for ensuring compatibility.)

    Returns:
      A list of `tf.Tensor`, compatible with `self._flat_tensor_specs`, which
      can be used to reconstruct `value`.
    """
    return nest.flatten(self._to_components(value), expand_composites=True)

  def _from_tensor_list(self, tensor_list: List["core_types.Symbol"]) -> Any:
    # [FUNCTION DOC]
    # Reconstructs a value from a flat tensor list. Input: tensor_list, list of tensors that must
    # match the flat tensor specs. Validates the list with the private check (raising a value error
    # on wrong length, dtype or shape mismatch) and then delegates to the unchecked conversion.
    # Returns the value.
    """Reconstructs a value from a flat list of `tf.Tensor`.

    Args:
      tensor_list: A flat list of `tf.Tensor`, compatible with
        `self._flat_tensor_specs`.

    Returns:
      A value that is compatible with this `TypeSpec`.

    Raises:
      ValueError: If `tensor_list` is not compatible with
      `self._flat_tensor_specs`.
    """
    self.__check_tensor_list(tensor_list)
    return self._from_compatible_tensor_list(tensor_list)

  def _from_compatible_tensor_list(
      self, tensor_list: List["core_types.Symbol"]) -> Any:
    # [FUNCTION DOC]
    # Unchecked reconstruction of a value from a flat tensor list that the caller guarantees is
    # compatible. Default: packs the list back into the structure of the component specs
    # (expanding composites) and calls the from-components conversion. Returns the value.
    """Reconstructs a value from a compatible flat list of `tf.Tensor`.

    Args:
      tensor_list: A flat list of `tf.Tensor`, compatible with
        `self._flat_tensor_specs`.  (Caller is responsible for ensuring
        compatibility.)

    Returns:
      A value that is compatible with this `TypeSpec`.
    """
    return self._from_components(
        nest.pack_sequence_as(
            self._component_specs, tensor_list, expand_composites=True))

  @property
  def _flat_tensor_specs(self):
    # [FUNCTION DOC]
    # Property returning the flat list of tensor specs that describes the tensor-list encoding:
    # the component specs flattened with composites expanded. Overridden by batchable subclasses
    # that use a different (boxed) encoding. No side effects.
    """A list of TensorSpecs compatible with self._to_tensor_list(v)."""
    return nest.flatten(self._component_specs, expand_composites=True)

  # === Serialization for types ===

  @abc.abstractmethod
  def _serialize(self):
    # [FUNCTION DOC]
    # Abstract: returns a nested tuple holding the state of this spec, enough to rebuild it with
    # the deserialize class method. Allowed element types include booleans, numbers, strings,
    # nothing/null, tensor specs, shapes, dtypes, arrays, specs and nested containers of those.
    # Drives default equality, hashing, pickling, repr and compatibility. The base implementation
    # raises a not-implemented error naming the class.
    """Returns a nested tuple containing the state of this TypeSpec.

    The serialization may contain the following value types: boolean,
    integer, string, float, None, `TensorSpec`, `tf.TensorShape`, `tf.DType`,
    `np.ndarray`, `TypeSpec`, and nested tuples, namedtuples, dicts, and
    OrderedDicts of any of the above.

    This method is used to provide default definitions for: equality
    testing (__eq__, __ne__), hashing (__hash__), pickling (__reduce__),
    string representation (__repr__), `self.is_compatible_with()`,
    `self.most_specific_compatible_type()`, and protobuf serialization
    (e.g. TensorInfo and StructuredValue).
    """
    raise NotImplementedError("%s._serialize()" % type(self).__name__)

  @classmethod
  def _deserialize(cls, serialization):
    # [FUNCTION DOC]
    # Class method that rebuilds a spec from the value produced by the serialize method. Input:
    # serialization, the nested tuple (named tuples may be duck-typed copies when loaded from a
    # saved model). Default: calls the class constructor with the serialization elements as
    # positional arguments and returns the new spec; subclasses with other constructor shapes
    # must override it. Constructor errors propagate.
    """Reconstructs a TypeSpec from a value returned by `serialize`.

    Args:
      serialization: A value returned by _serialize.  In some contexts,
        `namedtuple`s in `serialization` may not have the identical type that
        was returned by `_serialize` (but its type will still be a `namedtuple`
        type with the same type name and field names).  For example, the code
        that loads a SavedModel does not have access to the original
        `namedtuple` type, so it dynamically creates a new `namedtuple` type
        with the same type name and field names as the original one.  If
        necessary, you can check `serialization` for these duck-typed
        `nametuple` types, and restore them to the original type. (E.g., this
        would be necessary if you rely on type checks such as `isinstance` for
        this `TypeSpec`'s member variables).

    Returns:
      A `TypeSpec` of type `cls`.
    """
    return cls(*serialization)  # pytype: disable=not-instantiable  # trace-all-classes

  # === Operators ===

  def __eq__(self, other) -> bool:
    # [FUNCTION DOC]
    # Equality: true only if other has exactly the same class as this spec and the cached
    # comparison keys (derived from the serialization) are equal. Returns a boolean. May populate
    # the cached comparison key on both objects.
    # pylint: disable=protected-access
    return (type(other) is type(self) and
            self.__get_cmp_key() == other.__get_cmp_key())

  def __ne__(self, other) -> bool:
    # [FUNCTION DOC]
    # Inequality: the negation of the equality method. Returns a boolean.
    return not self == other

  def __hash__(self) -> int:
    # [FUNCTION DOC]
    # Hash derived from the cached comparison key, consistent with equality. Returns an integer.
    # Raises a type error if the serialization contains unhashable parts that the key builder
    # cannot convert. May populate the cached key.
    return hash(self.__get_cmp_key())

  def __reduce__(self):
    # [FUNCTION DOC]
    # Pickling support: returns a pair of the class and the serialization tuple, so unpickling
    # calls the constructor with the serialization as positional arguments (same assumption as the
    # default deserialize method).
    return type(self), self._serialize()

  def __repr__(self) -> str:
    # [FUNCTION DOC]
    # String form: the class name followed by the repr of the serialization tuple. Returns a
    # string. No side effects.
    return "%s%r" % (type(self).__name__, self._serialize())

  # === Legacy Output ===
  # TODO(b/133606651) Document and/or deprecate the legacy_output methods.
  # (These are used by tf.data.)

  def _to_legacy_output_types(self):
    # [FUNCTION DOC]
    # Legacy tf.data hook returning the old-style output dtypes for this spec. The base
    # implementation always raises a not-implemented error naming the class; data-compatible
    # subclasses override it.
    raise NotImplementedError("%s._to_legacy_output_types()" %
                              type(self).__name__)

  def _to_legacy_output_shapes(self):
    # [FUNCTION DOC]
    # Legacy tf.data hook returning the old-style output shapes for this spec. The base
    # implementation always raises a not-implemented error naming the class; data-compatible
    # subclasses override it.
    raise NotImplementedError("%s._to_legacy_output_shapes()" %
                              type(self).__name__)

  def _to_legacy_output_classes(self):
    # [FUNCTION DOC]
    # Legacy tf.data hook returning the old-style output classes for this spec. Unlike its
    # siblings, the default simply returns the value type of this spec. No side effects.
    return self.value_type

  # === Private Helper Methods ===

  # TODO(b/154541175): Currently this usage is used to represent a Tensor
  # argument not a TensorSpec argument as it should be.
  def __tf_tracing_type__(self,
                          context: trace.TracingContext) -> trace.TraceType:
    # [FUNCTION DOC]
    # Tracing hook used by function tracing to obtain the trace type of an argument. Input:
    # context, the tracing context (unused). Returns this spec itself, since a spec is already a
    # trace type. See the todo above: it currently stands in for a tensor argument.
    return self

  def __check_tensor_list(self, tensor_list):
    # [FUNCTION DOC]
    # HELPER: validation step of the checked tensor-list conversion.
    # Input: tensor_list, a list of tensors or values. Builds a type spec for each item and
    # compares to the flat tensor specs. Returns nothing. Raises a value error if the item count
    # differs from the expected count, or if any item's spec is not compatible with the expected
    # spec (message names the item index). Conversion of items may itself raise a type error.
    """Raises an exception if tensor_list incompatible w/ flat_tensor_specs."""
    expected = self._flat_tensor_specs
    specs = [type_spec_from_value(t) for t in tensor_list]
    if len(specs) != len(expected):
      raise ValueError(f"Cannot create a {self.value_type.__name__} from the "
                       f"tensor list because the TypeSpec expects "
                       f"{len(expected)} items, but the provided tensor list "
                       f"has {len(specs)} items.")
    for i, (s1, s2) in enumerate(zip(specs, expected)):
      if not s1.is_compatible_with(s2):
        raise ValueError(f"Cannot create a {self.value_type.__name__} from the "
                         f"tensor list because item {i} ({tensor_list[i]!r}) "
                         f"is incompatible with the expected TypeSpec {s2}.")

  def __get_cmp_key(self):
    # [FUNCTION DOC]
    # HELPER: supplies the key used by equality and hashing.
    # Returns a pair of the class and a hashable version of the serialization, computing it on
    # first use and caching it on the instance in the cached-key slot. Side effect: sets that
    # attribute. Propagates the value error from the key builder for unsupported serialization
    # contents.
    """Returns a hashable eq-comparable key for `self`."""
    if not hasattr(self, _CACHED_CMP_KEY):
      setattr(self, _CACHED_CMP_KEY, (
          type(self),
          self.__make_cmp_key(self._serialize()),
      ))
    return getattr(self, _CACHED_CMP_KEY)

  def __make_cmp_key(self, value):
    # [FUNCTION DOC]
    # HELPER: converts one serialization element into a hashable, comparable key.
    # Scalars, dtypes, shapes, specs, strings and null pass through; dictionaries become tuples of
    # key/value pairs sorted by key; tuples convert element-wise; lists become a marker plus tuple
    # so they differ from tuples; arrays become a marker, their shape and a nested tuple of
    # contents. Raises a value error naming the spec and the offending type for anything else.
    """Converts `value` to a hashable key."""
    if isinstance(value, (int, float, bool, np.generic, dtypes.DType, TypeSpec,
                          tensor_shape.TensorShape)):
      return value
    if isinstance(value, compat.bytes_or_text_types):
      return value
    if value is None:
      return value
    if isinstance(value, dict):
      return tuple([
          tuple([self.__make_cmp_key(key),
                 self.__make_cmp_key(value[key])])
          for key in sorted(value.keys())
      ])
    if isinstance(value, tuple):
      return tuple([self.__make_cmp_key(v) for v in value])
    if isinstance(value, list):
      return (list, tuple([self.__make_cmp_key(v) for v in value]))
    if isinstance(value, np.ndarray):
      return (np.ndarray, value.shape,
              TypeSpec.__nested_list_to_tuple(value.tolist()))
    raise ValueError(
        f"Cannot generate a hashable key for {self} because "
        "the _serialize() method "
        f"returned an unsupported value of type {type(value)}"
    )

  @staticmethod
  def __nested_list_to_tuple(value):
    # [FUNCTION DOC]
    # HELPER: static utility for building array comparison keys. Recursively turns a nested list
    # into a nested tuple; non-list values are returned unchanged. Pure function.
    """Converts a nested list to a corresponding nested tuple."""
    if isinstance(value, list):
      return tuple(TypeSpec.__nested_list_to_tuple(v) for v in value)
    return value

  @staticmethod
  def __same_types(a, b):
    # [FUNCTION DOC]
    # HELPER: static type-equivalence test used by the private compatibility check.
    # Inputs: two arbitrary objects. If both are named tuples, returns whether they are equivalent
    # (same class name and field names); otherwise returns whether their types are identical.
    # Pure function returning a boolean.
    """Returns whether a and b have the same type, up to namedtuple equivalence.

    Consistent with tf.nest.assert_same_structure(), two namedtuple types
    are considered the same iff they agree in their class name (without
    qualification by module name) and in their sequence of field names.
    This makes namedtuples recreated by nested_structure_coder compatible with
    their original Python definition.

    Args:
      a: a Python object.
      b: a Python object.

    Returns:
      A boolean that is true iff type(a) and type(b) are the same object
      or equivalent namedtuple types.
    """
    if nest.is_namedtuple(a) and nest.is_namedtuple(b):
      return nest.same_namedtuples(a, b)
    else:
      return type(a) is type(b)

  @staticmethod
  def __is_compatible(a, b):
    # [FUNCTION DOC]
    # HELPER: static recursive compatibility test on two serializations.
    # If a is a spec, defers to its compatibility method with b. Returns false when types differ
    # (named-tuple aware). Lists and tuples must have equal length and pairwise compatible
    # elements; dictionaries must have equal key sets and compatible values; shapes and dtypes use
    # their own compatibility method; everything else must be equal. Returns a boolean.
    """Returns true if the given type serializations compatible."""
    if isinstance(a, TypeSpec):
      return a.is_compatible_with(b)
    if not TypeSpec.__same_types(a, b):
      return False
    if isinstance(a, (list, tuple)):
      return (len(a) == len(b) and
              all(TypeSpec.__is_compatible(x, y) for (x, y) in zip(a, b)))
    if isinstance(a, dict):
      return (len(a) == len(b) and sorted(a.keys()) == sorted(b.keys()) and
              all(TypeSpec.__is_compatible(a[k], b[k]) for k in a.keys()))
    if isinstance(a, (tensor_shape.TensorShape, dtypes.DType)):
      return a.is_compatible_with(b)
    return a == b

trace_type.register_serializable(TypeSpec)


class TypeSpecBatchEncoder(object, metaclass=abc.ABCMeta):
  # [CLASS DOC]
  # Abstract interface for encoding composite tensor values into batchable flat tensors and back,
  # so that data pipelines and map-style functions can batch and unbatch them. Concrete encoders
  # implement batching and unbatching of specs, encoding and decoding of values, and describing
  # the encoding specs. Instances are typically attached to a spec class through its batch
  # encoder attribute. Stateless in the base.
  """Class used to encode and decode composite tensor values for batching.

  In order to be batched and unbatched by APIs such as `tf.data.Dataset` and
  `tf.map_fn`, composite tensors must be encoded using flat tensors that can
  themselves be batched or unbatched.  `TypeSpecBatchEncoder`s are
  responsible for implementing this encoding.

  If a composite tensor's shape is a prefix of the shape of all of its
  component tensors, then this encoding can usually be performed by just
  returning those component tensors as a list.  But if the composite tensor
  has components whose shape has a more complex relationship to the shape
  of the composite tensor, then a custom `TypeSpecBatchEncoder` may
  need to be implemented.
  """

  @abc.abstractmethod
  def batch(self, spec, batch_size):
    # [FUNCTION DOC]
    # Abstract: returns the spec describing a batch of values of the given spec. Inputs: spec, the
    # spec of one value; batch_size, an integer or nothing if unknown. The base implementation
    # raises a not-implemented error naming the class.
    """Returns the TypeSpec representing a batch of values described by `spec`.

    Args:
      spec: The `TypeSpec` for an individual value.
      batch_size: An `int` indicating the number of values that are batched
        together, or `None` if the batch size is not known.

    Returns:
      A `TypeSpec` for a batch of values.
    """
    raise NotImplementedError(f"{type(self).__name__}.batch")

  @abc.abstractmethod
  def unbatch(self, spec):
    # [FUNCTION DOC]
    # Abstract: returns the spec of a single element of a batch described by spec. Input: spec of
    # a batch. The base implementation raises a not-implemented error naming the class.
    """Returns the TypeSpec for a single unbatched element in `spec`.

    Args:
      spec: The `TypeSpec` for a batch of values.

    Returns:
      A `TypeSpec` for an individual value.
    """
    raise NotImplementedError(f"{type(self).__name__}.unbatch")

  @abc.abstractmethod
  def encode(self, spec, value, minimum_rank=0):
    # [FUNCTION DOC]
    # Abstract: encodes a value as a nest of batchable tensors, composites or extension values.
    # Inputs: spec of the value; value compatible with it; minimum_rank, default zero, the minimum
    # rank every returned tensor must have so it can be unbatched that many times. Output: the
    # nest, such that stacking/unstacking/concatenating encodings equals encoding the results.
    # The base implementation raises a not-implemented error naming the class.
    """Encodes `value` as a nest of batchable `Tensor` or `CompositeTensor`.

    Args:
      spec: The TypeSpec of the value to encode.
      value: A value compatible with `spec`.
      minimum_rank: The minimum rank for the returned Tensors, CompositeTensors,
        and ExtensionType values.  This can be used to ensure that the encoded
        values can be unbatched this number of times.   If `minimum_rank>0`,
        then `t.shape[:minimum_rank]` must be compatible for all values `t`
        returned by `encode`.

    Returns:
      A nest (as defined by `tf.nest`) of `tf.Tensor`s, batchable
      `tf.CompositeTensor`s, or `tf.ExtensionType`s.  Stacking, unstacking, or
      concatenating these encoded values and then decoding the result must be
      equivalent to stacking, unstacking, or concatenating the original values.
    """
    raise NotImplementedError(f"{type(self).__name__}.encode")

  @abc.abstractmethod
  def decode(self, spec, encoded_value):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says the result is compatible with a name that is not a
    # parameter; the actual parameter is called spec.
    # Abstract: decodes an encoded nest back into a value. Inputs: spec, the spec of the result
    # (the batched or unbatched spec as appropriate); encoded_value, a nest returned by encode or
    # built by stacking, unstacking or concatenating such nests. Output: a compatible value. The
    # base implementation raises a not-implemented error naming the class.
    """Decodes `value` from a batchable tensor encoding.

    Args:
      spec: The TypeSpec for the result value.  If encoded values with spec `s`
        were batched, then `spec` should be `s.batch(batch_size)`; or if encoded
        values with spec `s` were unbatched, then `spec` should be
        `s.unbatch()`.
      encoded_value: A nest of values returned by `encode`; or a nest of values
        that was formed by stacking, unstacking, or concatenating the
        corresponding elements of values returned by `encode`.

    Returns:
      A value compatible with `type_spec`.
    """
    raise NotImplementedError(f"{type(self).__name__}.decode")

  @abc.abstractmethod
  def encoding_specs(self, spec):
    # [FUNCTION DOC]
    # Abstract: returns a nest of batchable specs describing what the encode method produces for
    # the given spec. Input: spec. The base implementation raises a not-implemented error naming
    # the class.
    """Returns a nest of `TypeSpec`(s) describing the encoding for `spec`.

    Args:
      spec: The TypeSpec whose encoding should be described.

    Returns:
      A nest (as defined by `tf.nest) of `tf.TypeSpec`, describing the values
      that are returned by `self.encode(spec, ...)`.  All TypeSpecs in this
      nest must be batchable.
    """
    raise NotImplementedError(f"{type(self).__name__}.encoding_specs")


class LegacyTypeSpecBatchEncoder(TypeSpecBatchEncoder):
  # [CLASS DOC]
  # Batch encoder for legacy composite tensor specs: it forwards every operation to the private
  # batching and tensor-list methods of the spec itself. A single shared instance is installed as
  # the default batch encoder of batchable specs. Stateless.
  """TypeSpecBatchEncoder for legacy composite tensor classes.

  TODO(edloper): Update existing composite tensors to use non-legacy
    CompositeTensorBatchEncoders.
  """

  def batch(self, type_spec, batch_size):
    # [FUNCTION DOC]
    # Returns the batched spec by calling the spec's own batching method. Inputs: type_spec, a
    # batchable spec; batch_size, an integer or nothing. Errors from the spec propagate.
    return type_spec._batch(batch_size)  # pylint: disable=protected-access

  def unbatch(self, type_spec):
    # [FUNCTION DOC]
    # Returns the unbatched spec by calling the spec's own unbatching method. Input: type_spec, a
    # batchable spec. Errors from the spec propagate.
    return type_spec._unbatch()  # pylint: disable=protected-access

  def encode(self, type_spec, value, minimum_rank=0):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the error messages read the name attribute from the spec instance, but spec
    # instances normally have no such attribute, so the intended value error would surface as an
    # attribute error instead.
    # Encodes a value to a flat tensor list. Inputs: type_spec; value; minimum_rank (default zero).
    # With rank zero uses the spec's plain tensor-list encoding; with rank one requires a batchable
    # spec and uses its batched tensor-list encoding (each tensor of rank at least one). Raises a
    # value error for rank one with a non-batchable spec, and for any rank above one.
    if minimum_rank == 0:
      return type_spec._to_tensor_list(value)  # pylint: disable=protected-access
    elif minimum_rank == 1:
      if not isinstance(type_spec, BatchableTypeSpec):
        raise ValueError(f"{type_spec.__name__}.encode does not support "
                         "minimum_rank>0.")
      return type_spec._to_batched_tensor_list(value)  # pylint: disable=protected-access
    else:
      raise ValueError(f"{type_spec.__name__}.encode does not support "
                       "minimum_rank>1.")

  def decode(self, type_spec, encoded_value):
    # [FUNCTION DOC]
    # Decodes a flat tensor list into a value using the spec's checked tensor-list conversion.
    # Inputs: type_spec; encoded_value, a list of tensors. Raises a value error if the list is
    # incompatible with the spec's flat tensor specs.
    return type_spec._from_tensor_list(encoded_value)  # pylint: disable=protected-access

  def encoding_specs(self, spec):
    # [FUNCTION DOC]
    # Returns the flat list of tensor specs of the given spec as its encoding description.
    # Input: spec. No side effects.
    return spec._flat_tensor_specs  # pylint: disable=protected-access


class BatchableTypeSpec(TypeSpec, metaclass=abc.ABCMeta):
  # [CLASS DOC]
  # Abstract spec subclass whose values have a batchable tensor encoding (a flat list of tensors
  # that can be stacked, unstacked and concatenated, possibly via encode/decode ops). Subclasses
  # implement the private batch and unbatch methods; the default tensor-list encoding is built
  # recursively from the component specs using each component's batch encoder. Has no slots of its
  # own and carries a class-level default batch encoder (the legacy encoder instance).
  """TypeSpec with a batchable tensor encoding.

  The batchable tensor encoding is a list of `tf.Tensor`s that supports
  batching and unbatching.  In particular, stacking (or unstacking)
  values with the same `TypeSpec` must be equivalent to stacking (or
  unstacking) each of their tensor lists.  Unlike the component encoding
  (returned by `self._to_components)`, the batchable tensor encoding
  may require using encoding/decoding ops.

  If a subclass's batchable tensor encoding is not simply a flattened version
  of the component encoding, then the subclass must override `_to_tensor_list`,
  `_from_tensor_list`, and _flat_tensor_specs`.
  """

  __slots__ = []

  __batch_encoder__ = LegacyTypeSpecBatchEncoder()

  @abc.abstractmethod
  def _batch(self, batch_size) -> TypeSpec:
    # [FUNCTION DOC]
    # Abstract: returns the spec for a batch of values of this spec. Input: batch_size, an
    # integer or nothing when the batch size varies. The base implementation raises a
    # not-implemented error naming the class.
    """Returns a TypeSpec representing a batch of objects with this TypeSpec.

    Args:
      batch_size: An `int` representing the number of elements in a batch, or
        `None` if the batch size may vary.

    Returns:
      A `TypeSpec` representing a batch of objects with this TypeSpec.
    """
    raise NotImplementedError(f"{type(self).__name__}._batch")

  @abc.abstractmethod
  def _unbatch(self) -> TypeSpec:
    # [FUNCTION DOC]
    # Abstract: returns the spec of a single element of a batch described by this spec. The base
    # implementation raises a not-implemented error naming the class.
    """Returns a TypeSpec representing a single element this TypeSpec.

    Returns:
      A `TypeSpec` representing a single element of objects with this TypeSpec.
    """
    raise NotImplementedError(f"{type(self).__name__}._unbatch")

# LINT.IfChange
  @property
  def _flat_tensor_specs(self) -> List[TypeSpec]:
    # [FUNCTION DOC]
    # Property override giving the flat tensor specs of the batchable encoding: each component
    # spec is expanded with the batchable flat-spec helper (using this spec as context for
    # warnings) and the results are flattened into one list. Must stay consistent with the
    # flat-spec logic in the type utilities module. May emit warnings for non-batchable components.
    """A list of TensorSpecs compatible with self._to_tensor_list(v)."""
    component_flat_tensor_specs = nest.map_structure(
        functools.partial(get_batchable_flat_tensor_specs, context_spec=self),
        self._component_specs)
    return nest.flatten(component_flat_tensor_specs)
# LINT.ThenChange(//tensorflow/python/framework/type_utils.py:_specs_for_flat_tensors)
# Note that _specs_for_flat_tensors in type_utils.py must correspond
# _flat_tensor_specs in this class and any derived classes.

  def _to_tensor_list(
      self, value: composite_tensor.CompositeTensor
      ) -> List["core_types.Symbol"]:
    # [FUNCTION DOC]
    # Override encoding a composite value to a flat tensor list using the batchable encoding.
    # Input: value, a composite tensor compatible with this spec. Converts it to components and,
    # for each component spec, obtains its tensor list through the batchable helper, then flattens.
    # Returns the list of tensors. May create encoding ops.
    """Encodes `value` as a flat list of `core.Symbol`."""
    component_tensor_lists = nest.map_structure(batchable_to_tensor_list,
                                                self._component_specs,
                                                self._to_components(value))
    return nest.flatten(component_tensor_lists)

  def _to_batched_tensor_list(
      self, value: composite_tensor.CompositeTensor
      ) -> List["core_types.Symbol"]:
    # [FUNCTION DOC]
    # Encodes a composite value to a flat tensor list where each tensor has rank at least one, so
    # the list can be unbatched. Input: value. For each component, batchable specs are encoded
    # with minimum rank one and other specs use their plain tensor list. Raises a value error if
    # any resulting tensor has rank zero (a scalar cannot be batched). Returns the list.
    """Encodes `value` as a flat list of `core.Symbol` each with rank>0."""
    get_spec_tensor_list = lambda spec, v: (  # pylint: disable=g-long-lambda
        batchable_to_tensor_list(spec, v, minimum_rank=1)
        if isinstance(spec, BatchableTypeSpec) else spec._to_tensor_list(v))  # pylint: disable=protected-access
    component_batched_tensor_lists = nest.map_structure(
        get_spec_tensor_list, self._component_specs, self._to_components(value))
    tensor_list = nest.flatten(component_batched_tensor_lists)
    if any(t.shape.ndims == 0 for t in tensor_list):
      raise ValueError(
          f"While converting {value} to a list of tensors for batching, "
          f"found a scalar item which cannot be batched.")
    return tensor_list

  def _from_compatible_tensor_list(
      self,
      tensor_list: List["core_types.Symbol"]
      ) -> composite_tensor.CompositeTensor:
    # [FUNCTION DOC]
    # Override that rebuilds a composite value from a compatible flat tensor list in the batchable
    # encoding. Input: tensor_list (compatibility is the caller's responsibility). Packs the list
    # into the nested structure of batchable flat specs, decodes each component with the
    # batchable decode helper, and calls the from-components conversion. Returns the value.
    """Reconstructs a value from a compatible flat list of `core.Symbol`."""
    flat_specs = nest.map_structure(
        functools.partial(get_batchable_flat_tensor_specs, context_spec=self),
        self._component_specs)
    nested_tensor_list = nest.pack_sequence_as(flat_specs, tensor_list)
    components = nest.map_structure_up_to(self._component_specs,
                                          batchable_from_tensor_list,
                                          self._component_specs,
                                          nested_tensor_list)
    return self._from_components(components)


def get_batchable_flat_tensor_specs(spec, context_spec=None):
  # [FUNCTION DOC]
  # HELPER: flat-spec expansion used by batchable specs.
  # Inputs: spec, a spec to expand; context_spec, optional spec used only in warning text.
  # Returns a flat list of tensor specs: a tensor spec yields a one-item list; a spec with a
  # batch encoder yields the flattened encoding specs (expanded recursively); otherwise emits a
  # warning that a batchable type contains a non-batchable component and returns the spec's own
  # flat tensor specs.
  """Returns the flat tensor specs for `spec`."""
  if isinstance(spec, internal.TensorSpec):
    return [spec]
  elif hasattr(spec, "__batch_encoder__"):
    encoding_specs = nest.map_structure(
        functools.partial(
            get_batchable_flat_tensor_specs, context_spec=context_spec),
        spec.__batch_encoder__.encoding_specs(spec))
    return nest.flatten(encoding_specs)
  else:
    # TODO(edloper) Fix existing CompositeTensors that permit this, and
    # then turn this warning into an error.
    warnings.warn(f"Batchable type {context_spec} contains non-batchable "
                  f"field or component with type {spec}.")
    return spec._flat_tensor_specs  # pylint: disable=protected-access


def batchable_to_tensor_list(spec, value, minimum_rank=0):
  # [FUNCTION DOC]
  # HELPER: recursive encoder used by batchable specs.
  # Inputs: spec; value of that spec; minimum_rank, default zero. A tensor spec returns a
  # one-item list holding the value. A spec with a batch encoder encodes the value with it,
  # then recursively converts the encoded nest to tensors using the encoding specs, and returns
  # the flattened list. Otherwise returns the spec's plain tensor-list encoding. Errors propagate.
  """Returns a list of tensors encoding `value`, whose type is `spec`."""
  if isinstance(spec, internal.TensorSpec):
    return [value]
  elif hasattr(spec, "__batch_encoder__"):
    encoded_value = spec.__batch_encoder__.encode(spec, value, minimum_rank)
    encoded_specs = spec.__batch_encoder__.encoding_specs(spec)
    encoded_flats = nest.map_structure(
        functools.partial(batchable_to_tensor_list, minimum_rank=minimum_rank),
        encoded_specs, encoded_value)
    return nest.flatten(encoded_flats)
  else:
    return spec._to_tensor_list(value)  # pylint: disable=protected-access


def batchable_from_tensor_list(spec, tensor_list):
  # [FUNCTION DOC]
  # HELPER: recursive decoder, inverse of the batchable encoder helper.
  # Inputs: spec; tensor_list, the flat tensors for this spec. A tensor spec asserts the list has
  # exactly one item and returns it (assertion error otherwise). A spec with a batch encoder packs
  # the list per the encoding specs, decodes nested parts recursively and decodes with the
  # encoder. Otherwise uses the spec's unchecked tensor-list conversion. Returns the value.
  """Returns a value with type `spec` decoded from `tensor_list`."""
  if isinstance(spec, internal.TensorSpec):
    assert len(tensor_list) == 1
    return tensor_list[0]
  elif hasattr(spec, "__batch_encoder__"):
    encoded_specs = spec.__batch_encoder__.encoding_specs(spec)
    flat_specs = nest.map_structure(get_batchable_flat_tensor_specs,
                                    encoded_specs)
    encoded_flats = nest.pack_sequence_as(flat_specs, tensor_list)
    encoded_value = nest.map_structure_up_to(encoded_specs,
                                             batchable_from_tensor_list,
                                             encoded_specs, encoded_flats)
    return spec.__batch_encoder__.decode(spec, encoded_value)
  else:
    return spec._from_compatible_tensor_list(tensor_list)  # pylint: disable=protected-access


@tf_export("type_spec_from_value")
def type_spec_from_value(value) -> TypeSpec:
  # [FUNCTION DOC]
  # Public function (exported as the type-spec-from-value API) returning the spec that describes
  # a value. Input: value, a tensor-like, composite tensor, or any value convertible to a tensor.
  # First tries the private converter; if that yields nothing, tries converting the value to a
  # tensor (conversion failures are only logged at verbose level 3) and taking that tensor's spec.
  # Raises a type error if no spec can be built.
  """Returns a `tf.TypeSpec` that represents the given `value`.

  Examples:

    >>> tf.type_spec_from_value(tf.constant([1, 2, 3]))
    TensorSpec(shape=(3,), dtype=tf.int32, name=None)
    >>> tf.type_spec_from_value(np.array([4.0, 5.0], np.float64))
    TensorSpec(shape=(2,), dtype=tf.float64, name=None)
    >>> tf.type_spec_from_value(tf.ragged.constant([[1, 2], [3, 4, 5]]))
    RaggedTensorSpec(TensorShape([2, None]), tf.int32, 1, tf.int64)

    >>> example_input = tf.ragged.constant([[1, 2], [3]])
    >>> @tf.function(input_signature=[tf.type_spec_from_value(example_input)])
    ... def f(x):
    ...   return tf.reduce_sum(x, axis=1)

  Args:
    value: A value that can be accepted or returned by TensorFlow APIs. Accepted
      types for `value` include `tf.Tensor`, any value that can be converted to
      `tf.Tensor` using `tf.convert_to_tensor`, and any subclass of
      `CompositeTensor` (such as `tf.RaggedTensor`).

  Returns:
    A `TypeSpec` that is compatible with `value`.

  Raises:
    TypeError: If a TypeSpec cannot be built for `value`, because its type
      is not supported.
  """
  spec = _type_spec_from_value(value)
  if spec is not None:
    return spec

  # Fallback: try converting value to a tensor.
  try:
    tensor = tensor_conversion_registry.convert(value)
    spec = _type_spec_from_value(tensor)
    if spec is not None:
      return spec
  except (ValueError, TypeError) as e:
    logging.vlog(
        3, "Failed to convert %r to tensor: %s" % (type(value).__name__, e))

  raise TypeError(f"Could not build a TypeSpec for {value} of "
                  f"unsupported type {type(value)}.")


def _type_spec_from_value(value) -> TypeSpec:
  # [FUNCTION DOC]
  # HELPER: core lookup behind the public spec-from-value function.
  # KNOWN ISSUE: the return annotation claims a spec is always returned, but the function returns
  # nothing/null when no rule matches.
  # Input: value. Tensor-like symbols use their trace type; composite tensors use their own spec;
  # non-empty lists whose items all map to the same batchable spec are merged and batched by the
  # list length; otherwise the registry of converters is scanned from the most recent entry and
  # the first match (exact type, or subclass if allowed) is used. Returns the spec or null.
  """Returns a `TypeSpec` that represents the given `value`."""
  if isinstance(value, core_types.Symbol):
    # Note: we do not include Tensor names when constructing TypeSpecs.
    return trace_type.from_value(value)

  if isinstance(value, composite_tensor.CompositeTensor):
    return value._type_spec  # pylint: disable=protected-access

  # If `value` is a list and all of its elements can be represented by the same
  # batchable type spec, then we can represent the entire list using a single
  # type spec that captures the type accurately (unlike the `convert_to_tensor`
  # fallback).
  if isinstance(value, list) and value:
    subspecs = [_type_spec_from_value(v) for v in value]
    if isinstance(subspecs[0], BatchableTypeSpec):
      merged_subspec = subspecs[0].most_specific_common_supertype(subspecs[1:])
      if merged_subspec is not None:
        return merged_subspec._batch(len(subspecs))  # pylint: disable=protected-access

  for entry in reversed(_TYPE_CONVERSION_FUNCTION_REGISTRY):
    type_object, converter_fn, allow_subclass = entry
    if ((type(value) is type_object) or  # pylint: disable=unidiomatic-typecheck
        (allow_subclass and isinstance(value, type_object))):
      return converter_fn(value)

  return None


_TYPE_CONVERSION_FUNCTION_REGISTRY = []


def register_type_spec_from_value_converter(type_object,
                                            converter_fn,
                                            allow_subclass=False):
  # [FUNCTION DOC]
  # Registers a converter from values of a given Python type to specs. Inputs: type_object, the
  # type of accepted values (any decorator wrappers are unwrapped first); converter_fn, a callback
  # taking one value and returning a spec; allow_subclass, default false: if true, subclass
  # instances also match, otherwise only exact type matches. Later registrations take precedence.
  # Side effect: appends to the module-level converter registry. Returns nothing.
  """Registers a function for converting values with a given type to TypeSpecs.

  If multiple registered `type_object`s match a value, then the most recent
  registration takes precedence.  Custom converters should not be defined for
  `CompositeTensor`s; use `CompositeTensor._type_spec` instead.

  Args:
    type_object: A Python `type` object representing the type of values accepted
      by `converter_fn`.
    converter_fn: A function that takes one argument (an instance of the type
      represented by `type_object`) and returns a `TypeSpec`.
    allow_subclass: If true, then use `isinstance(value, type_object)` to check
      for matches.  If false, then use `type(value) is type_object`.
  """
  _, type_object = tf_decorator.unwrap(type_object)
  _TYPE_CONVERSION_FUNCTION_REGISTRY.append(
      (type_object, converter_fn, allow_subclass))
