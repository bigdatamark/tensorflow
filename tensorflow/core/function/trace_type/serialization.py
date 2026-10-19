# [MODULE DOC]
# Serialization support for TraceTypes. Defines the Serializable interface (abstract base class)
# that portable TraceTypes implement, a module-level registry mapping protobuf message classes to
# the Python classes that handle them, and the functions to register a class and to convert
# objects to and from the generic serialized TraceType message (which packs the class-specific
# message inside an "any"-style field). Used by default_types and other TraceType modules.
# Import-time state: the registry starts empty and is filled as modules call the registration
# function.
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
"""Utils for serializing and deserializing TraceTypes."""

import abc
from typing import Type

from google.protobuf import message
from tensorflow.core.function.trace_type import serialization_pb2

SerializedTraceType = serialization_pb2.SerializedTraceType

PROTO_CLASS_TO_PY_CLASS = {}


class Serializable(metaclass=abc.ABCMeta):
  # [CLASS DOC]
  # Abstract interface for TraceTypes that can be converted to and from a protocol buffer message.
  # Implementers must provide a class method naming their unique message class, a class method
  # building an instance from such a message, and an instance method producing the message.
  # Classes must additionally be registered via the module's registration function to be found
  # by the deserialization function.
  """TraceTypes implementing this additional interface are portable."""

  @classmethod
  @abc.abstractmethod
  def experimental_type_proto(cls) -> Type[message.Message]:
    # [FUNCTION DOC]
    # Abstract class method: subclasses return the message class uniquely associated with them.
    # The base version raises a not-implemented error.
    """Returns the unique type of proto associated with this class."""
    raise NotImplementedError

  @classmethod
  @abc.abstractmethod
  def experimental_from_proto(cls, proto: message.Message) -> "Serializable":
    # [FUNCTION DOC]
    # Abstract class method: subclasses build and return an instance from the given message of
    # their associated message class. The base version raises a not-implemented error.
    """Returns an instance based on a proto."""
    raise NotImplementedError

  @abc.abstractmethod
  def experimental_as_proto(self) -> message.Message:
    # [FUNCTION DOC]
    # Abstract instance method: subclasses return a message (of the class named by the type
    # method) describing this instance. The base version raises a not-implemented error.
    """Returns a proto representing this instance."""
    raise NotImplementedError


def register_serializable(cls: Type[Serializable]):
  # [FUNCTION DOC]
  # Adds a serializable class to the global registry, keyed by its associated message class.
  # Takes the class to register. Returns nothing. Raises a value error if another class is
  # already registered for the same message class. Side effect: mutates the module-level
  # registry.
  """Registers a Python class to support serialization.

  Only register standard TF types. Custom types should NOT be registered.

  Args:
    cls: Python class to register.
  """
  if cls.experimental_type_proto() in PROTO_CLASS_TO_PY_CLASS:
    raise ValueError(
        "Existing Python class " +
        PROTO_CLASS_TO_PY_CLASS[cls.experimental_type_proto()].__name__ +
        " already has " + cls.experimental_type_proto().__name__ +
        " as its associated proto representation. Please ensure " +
        cls.__name__ + " has a unique proto representation.")

  PROTO_CLASS_TO_PY_CLASS[cls.experimental_type_proto()] = cls


def serialize(to_serialize: Serializable) -> SerializedTraceType:
  # [FUNCTION DOC]
  # Converts a Serializable object into the generic serialized TraceType message. Takes the object.
  # Calls its as-proto method, verifies the result is of the message class the object declares,
  # and packs it into the representation field of a new serialized message, which is returned.
  # Raises a value error if the object is not Serializable or returned a message of a different
  # class than declared. Does not require the class to be registered.
  """Converts Serializable to a proto SerializedTraceType."""

  if not isinstance(to_serialize, Serializable):
    raise ValueError("Can not serialize " + type(to_serialize).__name__ +
                     " since it is not Serializable. For object " +
                     str(to_serialize))
  actual_proto = to_serialize.experimental_as_proto()

  if not isinstance(actual_proto, to_serialize.experimental_type_proto()):
    raise ValueError(
        type(to_serialize).__name__ +
        " returned different type of proto than specified by " +
        "experimental_type_proto()")

  serialized = SerializedTraceType()
  serialized.representation.Pack(actual_proto)
  return serialized


def deserialize(proto: SerializedTraceType) -> Serializable:
  # [FUNCTION DOC]
  # KNOWN ISSUE: the error is raised with separate arguments (message pieces, url, value) instead
  # of one formatted message, so the error text shows as a tuple-like list of parts.
  # Converts a generic serialized TraceType message back into an object. Takes the message.
  # Searches the registry for the message class matching the packed representation, unpacks it
  # and calls that class's from-proto method, returning the result. Raises a value error if no
  # registered class matches.
  """Converts a proto SerializedTraceType to instance of Serializable."""
  for proto_class in PROTO_CLASS_TO_PY_CLASS:
    if proto.representation.Is(proto_class.DESCRIPTOR):
      actual_proto = proto_class()
      proto.representation.Unpack(actual_proto)
      return PROTO_CLASS_TO_PY_CLASS[proto_class].experimental_from_proto(
          actual_proto)

  raise ValueError(
      "Can not deserialize proto of url: ", proto.representation.type_url,
      " since no matching Python class could be found. For value ",
      proto.representation.value)
