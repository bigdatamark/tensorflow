# [MODULE DOC]
# Defines the types used by tf.function tracing: the abstract trace type that decides subtype
# relations, common supertypes, placeholders and tensor decomposition for dispatching calls to
# cached concrete functions; empty context classes for tracing, placeholders and casting; and a
# runtime-checkable protocol for objects that supply their own trace type. Import-time side
# effects: registers public API exports, including an explicit constant export for the protocol
# (done separately because the export decorator would break runtime-checkable protocols).
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""tf.function tracing types.

See `core.PolymorphicFunction` and `core.ConcreteFunction`.

`PolymorphicFunction` assigns types to call arguments, forming a signature.
Function signatures are used to match arguments to `ConcreteFunction`s.
For example, when a new `ConcreteFunction` is traced, it is assigned a
the signature of the arguments it was traced with. Subsequent call arguments
which match its signature will be dispatched to the same `ConcreteFunction`.
If no `ConcreteFunction` with a matching signature is found, a new one may be
traced (a process known as retracing).
"""

import abc
from typing import Any, Iterator, List, Optional, Sequence

from typing_extensions import Protocol
from typing_extensions import runtime_checkable

from tensorflow.python.types import core
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls


@tf_export("types.experimental.TraceType", v1=[])
class TraceType(metaclass=abc.ABCMeta):
  # [CLASS DOC]
  # Abstract base class describing the type of values for tf.function tracing. Subclasses
  # define subtype checks, most specific common supertypes, placeholder creation, equality and
  # hashing, and may override tensor decomposition, reconstruction, flattening and casting. The
  # function machinery uses it to decide whether a cached concrete function can be reused or
  # a retrace is needed. Exported as a public experimental type.
  """Represents the type of object(s) for tf.function tracing purposes.

  `TraceType` is an abstract class that other classes might inherit from to
  provide information regarding associated class(es) for the purposes of
  tf.function tracing. The typing logic provided through this mechanism will be
  used to make decisions regarding usage of cached concrete functions and
  retracing.

  For example, if we have the following tf.function and classes:
  ```python
  @tf.function
  def get_mixed_flavor(fruit_a, fruit_b):
    return fruit_a.flavor + fruit_b.flavor

  class Fruit:
    flavor = tf.constant([0, 0])

  class Apple(Fruit):
    flavor = tf.constant([1, 2])

  class Mango(Fruit):
    flavor = tf.constant([3, 4])
  ```

  tf.function does not know when to re-use an existing concrete function in
  regards to the `Fruit` class so naively it retraces for every new instance.
  ```python
  get_mixed_flavor(Apple(), Mango()) # Traces a new concrete function
  get_mixed_flavor(Apple(), Mango()) # Traces a new concrete function again
  ```

  However, we, as the designers of the `Fruit` class, know that each subclass
  has a fixed flavor and we can reuse an existing traced concrete function if
  it was the same subclass. Avoiding such unnecessary tracing of concrete
  functions can have significant performance benefits.

  ```python
  class FruitTraceType(tf.types.experimental.TraceType):
    def __init__(self, fruit):
      self.fruit_type = type(fruit)
      self.fruit_value = fruit

    def is_subtype_of(self, other):
       return (type(other) is FruitTraceType and
               self.fruit_type is other.fruit_type)

    def most_specific_common_supertype(self, others):
       return self if all(self == other for other in others) else None

    def placeholder_value(self, placeholder_context=None):
      return self.fruit_value

  class Fruit:

   def __tf_tracing_type__(self, context):
     return FruitTraceType(self)
  ```

  Now if we try calling it again:
  ```python
  get_mixed_flavor(Apple(), Mango()) # Traces a new concrete function
  get_mixed_flavor(Apple(), Mango()) # Re-uses the traced concrete function
  ```
  """

  @abc.abstractmethod
  def is_subtype_of(self, other: "TraceType") -> bool:
    # [FUNCTION DOC]
    # Abstract method. Input: other, a trace type to compare against. Must return true when
    # this type is a subtype of the other, meaning values of this type can be passed to a
    # function traced with the other type. Base body is empty.
    """Returns True if `self` is a subtype of `other`.

    For example, `tf.function` uses subtyping for dispatch:
    if `a.is_subtype_of(b)` is True, then an argument of `TraceType`
    `a` can be used as argument to a `ConcreteFunction` traced with an
    a `TraceType` `b`.

    Args:
     other: A TraceType object to be compared against.

    Example:

    ```python
    class Dimension(TraceType):
      def __init__(self, value: Optional[int]):
        self.value = value

      def is_subtype_of(self, other):
        # Either the value is the same or other has a generalized value that
        # can represent any specific ones.
        return (self.value == other.value) or (other.value is None)
    ```
    """

  @abc.abstractmethod
  def most_specific_common_supertype(
      self, others: Sequence["TraceType"]
  ) -> Optional["TraceType"]:
    # [FUNCTION DOC]
    # Abstract method. Input: others, a sequence of trace types. Must return the most specific
    # trace type that is a supertype of this type and all the others, or a null value if none
    # exists. Base body is empty.
    """Returns the most specific supertype of `self` and `others`, if exists.

    The returned `TraceType` is a supertype of `self` and `others`, that is,
    they are all subtypes (see `is_subtype_of`) of it.
    It is also most specific, that is, there it has no subtype that is also
    a common supertype of `self` and `others`.

    If `self` and `others` have no common supertype, this returns `None`.

    Args:
     others: A sequence of TraceTypes.

    Example:
    ```python
     class Dimension(TraceType):
       def __init__(self, value: Optional[int]):
         self.value = value

       def most_specific_common_supertype(self, other):
          # Either the value is the same or other has a generalized value that
          # can represent any specific ones.
          if self.value == other.value:
            return self.value
          else:
            return Dimension(None)
    ```
    """

  @abc.abstractmethod
  def placeholder_value(self, placeholder_context) -> Any:
    # [FUNCTION DOC]
    # KNOWN ISSUE: docstring has a typo in the function name ("tf.funcion") and an example
    # mixed into the argument section.
    # Abstract method. Input: placeholder_context, a context reserved for internal or future
    # use. Must return a placeholder value standing for possibly many actual values, which the
    # function is traced with. Base body is empty.
    """Creates a placeholder for tracing.

    tf.funcion traces with the placeholder value rather than the actual value.
    For example, a placeholder value can represent multiple different
    actual values. This means that the trace generated with that placeholder
    value is more general and reusable which saves expensive retracing.

    Args:
      placeholder_context: A context reserved for internal/future usage.
    For the `Fruit` example shared above, implementing:

    ```python
    class FruitTraceType:
      def placeholder_value(self, placeholder_context):
        return Fruit()
    ```
    instructs tf.function to trace with the `Fruit()` objects
    instead of the actual `Apple()` and `Mango()` objects when it receives a
    call to `get_mixed_flavor(Apple(), Mango())`. For example, Tensor arguments
    are replaced with Tensors of similar shape and dtype, output from
    a tf.Placeholder op.

    More generally, placeholder values are the arguments of a tf.function,
    as seen from the function's body:
    ```python
    @tf.function
    def foo(x):
      # Here `x` is be the placeholder value
      ...

    foo(x) # Here `x` is the actual value
    ```
    """

  def to_tensors(self, value: Any) -> List[core.Tensor]:
    # [FUNCTION DOC]
    # Default decomposition of a value into tensors. Input: value, a value of this type
    # (ignored). Returns an empty list; subclasses that carry tensors override it and must
    # always produce the same number of tensors for this type.
    """Breaks down a value of this type into Tensors.

    For a TraceType instance, the number of tensors generated for corresponding
    value should be constant.

    Args:
      value: A value belonging to this TraceType

    Returns:
      List of Tensors.
    """
    del value
    return []

  def from_tensors(self, tensors: Iterator[core.Tensor]) -> Any:
    # [FUNCTION DOC]
    # Default reconstruction of a value from tensors. Input: tensors, an iterator of tensors
    # (ignored, none consumed). Returns the placeholder value of this type, built with a fresh
    # placeholder context. Subclasses that use tensors must consume the same count as the
    # tensor decomposition produces.
    """Generates a value of this type from Tensors.

    Must use the same fixed amount of tensors as `to_tensors`.

    Args:
      tensors: An iterator from which the tensors can be pulled.

    Returns:
      A value of this type.
    """
    del tensors
    return self.placeholder_value(PlaceholderContext())

  def flatten(self) -> List["TraceType"]:
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it returns tensor specs, but the annotation and the
    # default implementation use a list of trace types (and the default list is empty).
    # Default flattening: returns an empty list. Subclasses override to list the component
    # types matching the tensor decomposition values.
    """Returns a list of TensorSpecs corresponding to `to_tensors` values."""
    return []

  def cast(self, value, cast_context) -> Any:
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring refers to an overloaded "_cast" method, which does not exist;
    # the actual method is named cast. It also describes the assertion as the raised error
    # while it is the default behavior.
    # Default cast of a value to this type. Inputs: value, a value belonging to this type;
    # cast_context, reserved for internal or future use (ignored). Asserts that the value
    # equals this type's placeholder value (built with a fresh placeholder context) and raises
    # an assertion error with a message naming the value and type otherwise; returns the
    # value unchanged. Subclasses override to perform real conversion.
    """Cast value to this type.

    Args:
      value: An input value belonging to this TraceType.
      cast_context: A context reserved for internal/future usage.

    Returns:
      The value casted to this TraceType.

    Raises:
      AssertionError: When _cast is not overloaded in subclass,
        the value is returned directly, and it should be the same to
        self.placeholder_value().
    """
    del cast_context
    assert value == self.placeholder_value(
        PlaceholderContext()), f"Can not cast {value!r} to type {self!r}"
    return value

  @abc.abstractmethod
  def __hash__(self) -> int:
    # [FUNCTION DOC]
    # Abstract hash method: subclasses must return an integer hash consistent with equality so
    # trace types can key caches. Base body does nothing (returns a null value).
    pass

  @abc.abstractmethod
  def __eq__(self, other) -> bool:
    # [FUNCTION DOC]
    # Abstract equality method: subclasses must return whether the other object is an equal
    # trace type. Base body does nothing (returns a null value).
    pass


class TracingContext(metaclass=abc.ABCMeta):
  # [CLASS DOC]
  # Abstract container for flags and variables that influence tracing while building the trace
  # types of a set of objects; shared across all tracing-type calls for that set. No members.
  """Contains information scoped to the tracing of multiple objects.

  `TracingContext` is a container class for flags and variables that have
  any kind of influence on the tracing behaviour of the class implementing
  the __tf_tracing_type__. This context will be shared across all
  __tf_tracing_type__ calls while constructing the TraceType for a particular
  set of objects.
  """


class PlaceholderContext():
  # [CLASS DOC]
  # Empty context class carrying information used when generating placeholders within a
  # scope. Instantiated with no arguments by default trace type methods.
  """Contains context information for generating placeholders within a scope."""


class CastContext():
  # [CLASS DOC]
  # Empty context class carrying information and rules for casting values to a type spec.
  """Contains context info and rules for casting values to a TypeSpec."""


@runtime_checkable
class SupportsTracingProtocol(Protocol):
  # [CLASS DOC]
  # Runtime-checkable protocol for custom classes that control tf.function retracing by
  # supplying their own trace type. Exported as a public type via an explicit constant export
  # at the end of this module.
  """A protocol allowing custom classes to control tf.function retracing."""

  @doc_controls.doc_private
  @abc.abstractmethod
  def __tf_tracing_type__(self, context: TracingContext) -> TraceType:
    # [FUNCTION DOC]
    # Protocol method hidden from generated docs. Input: context, a tracing context reserved
    # for internal or future use. Must return the trace type of this object, used to build
    # call signatures and match arguments against existing ones. Base body is empty.
    """Returns the tracing type of this object.

    The tracing type is used to build the signature of a tf.function
    when traced, and to match arguments with existing signatures.
    When a Function object is called, tf.function looks at the tracing type
    of the call arguments. If an existing signature of matching type exists,
    it will be used. Otherwise, a new function is traced, and its signature
    will use the tracing type of the call arguments.

    Args:
      context: a context reserved for internal/future usage.

    Returns:
      The tracing type of this object.
    """


# TODO(b/219556836): Direct tf_export decorator adds non-method members to the
# Protocol which breaks @runtime_checkable since it does not support them.
tf_export("types.experimental.SupportsTracingProtocol", v1=[]).export_constant(
    __name__, "SupportsTracingProtocol"
)
