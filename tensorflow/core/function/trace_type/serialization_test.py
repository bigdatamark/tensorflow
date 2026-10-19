# [MODULE DOC]
# Unit tests for the TraceType serialization utilities. Defines several fixture Serializable
# classes (a simple one, a composite one that nests others, and a superclass with three
# subclasses sharing one message type) and registers them at import time with the serialization
# registry. Tests cover serializing, deserializing, nesting, rejection of duplicate message
# classes and detection of a wrong message class.
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
"""Tests for serialization."""

from tensorflow.core.function.trace_type import serialization
from tensorflow.core.function.trace_type import serialization_test_pb2
from tensorflow.python.platform import test


class MyCustomClass(serialization.Serializable):
  # [CLASS DOC]
  # HELPER: Fixture Serializable class for the serialization tests.
  # Simple serializable object holding an integer index and a string name, mapped to a custom
  # test message with those two fields.

  def __init__(self, index, name):
    # [FUNCTION DOC]
    # Stores the integer index and string name.
    self.index = index
    self.name = name

  @classmethod
  def experimental_type_proto(cls):
    # [FUNCTION DOC]
    # Returns the custom test message class associated with this fixture.
    return serialization_test_pb2.MyCustomRepresentation

  @classmethod
  def experimental_from_proto(cls, proto):
    # [FUNCTION DOC]
    # Builds an instance from a message by copying its index and name fields.
    return MyCustomClass(proto.index, proto.name)

  def experimental_as_proto(self):
    # [FUNCTION DOC]
    # Returns a custom test message with this object's index and name.
    proto = serialization_test_pb2.MyCustomRepresentation(
        index=self.index, name=self.name)
    return proto


serialization.register_serializable(MyCustomClass)


class MyCompositeClass(serialization.Serializable):
  # [CLASS DOC]
  # HELPER: Fixture Serializable class for the serialization tests.
  # Holds any number of other serializable elements; its message contains the serialized form of
  # each element, so it exercises nested serialization through the registry.

  def __init__(self, *elements):
    # [FUNCTION DOC]
    # Stores the given serializable elements (positional arguments) as a tuple.
    self.elements = elements

  @classmethod
  def experimental_type_proto(cls):
    # [FUNCTION DOC]
    # Returns the composite test message class associated with this fixture.
    return serialization_test_pb2.MyCompositeRepresentation

  @classmethod
  def experimental_from_proto(cls, proto):
    # [FUNCTION DOC]
    # Builds an instance by deserializing every element in the message through the registry.
    return MyCompositeClass(
        *[serialization.deserialize(element) for element in proto.elements])

  def experimental_as_proto(self):
    # [FUNCTION DOC]
    # Returns a composite test message holding the serialized form of each element.
    serialized_elements = [
        serialization.serialize(element) for element in self.elements
    ]
    proto = serialization_test_pb2.MyCompositeRepresentation(
        elements=serialized_elements)
    return proto

serialization.register_serializable(MyCompositeClass)


class MySerializableSuperClass(serialization.Serializable):
  # [CLASS DOC]
  # HELPER: Fixture Serializable class for the serialization tests.
  # Base class registered once for a shared message type; three subclasses are distinguished by
  # an integer id in the message (1, 2 or 3). Instances are equal when they have the same exact
  # type.

  @classmethod
  def experimental_type_proto(cls):
    # [FUNCTION DOC]
    # Returns the multi-class test message class shared by this class and its subclasses.
    return serialization_test_pb2.MyMultiClassRepresentation

  @classmethod
  def experimental_from_proto(cls, proto):
    # [FUNCTION DOC]
    # Builds an instance of subclass one, two or three depending on the id in the message.
    # Raises a not-implemented error for any other id.
    if proto.id == 1:
      return SerializableFromSuperClassOne()

    if proto.id == 2:
      return SerializableFromSuperClassTwo()

    if proto.id == 3:
      return SerializableFromSuperClassThree()

    raise NotImplementedError

  def experimental_as_proto(self):
    # [FUNCTION DOC]
    # Returns the message with id 1, 2 or 3 according to which subclass this instance is. Raises
    # a not-implemented error for an instance of the base class itself.
    if isinstance(self, SerializableFromSuperClassOne):
      return serialization_test_pb2.MyMultiClassRepresentation(id=1)

    if isinstance(self, SerializableFromSuperClassTwo):
      return serialization_test_pb2.MyMultiClassRepresentation(id=2)

    if isinstance(self, SerializableFromSuperClassThree):
      return serialization_test_pb2.MyMultiClassRepresentation(id=3)

    raise NotImplementedError

  def __eq__(self, other):
    # [FUNCTION DOC]
    # True when both objects are of exactly the same class.
    return type(self) is type(other)

serialization.register_serializable(MySerializableSuperClass)


class SerializableFromSuperClassOne(MySerializableSuperClass):
  # [CLASS DOC]
  # HELPER: Fixture subclass; serialized with id 1 of the shared multi-class message.
  pass


class SerializableFromSuperClassTwo(MySerializableSuperClass):
  # [CLASS DOC]
  # HELPER: Fixture subclass; serialized with id 2 of the shared multi-class message.
  pass


class SerializableFromSuperClassThree(MySerializableSuperClass):
  # [CLASS DOC]
  # HELPER: Fixture subclass; serialized with id 3 of the shared multi-class message.
  pass


class SerializeTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the serialization functions, using the module-level fixture classes.

  def testCustomClassSerialization(self):
    # [FUNCTION DOC]
    # Scenario: serialize a simple custom object. Expects the packed representation to be of the
    # custom message type and, once unpacked, to carry the same index and name.
    my_custom = MyCustomClass(1234, "my_name")
    serialized = serialization.serialize(my_custom)

    self.assertTrue(
        serialized.representation.Is(
            serialization_test_pb2.MyCustomRepresentation.DESCRIPTOR))

    proto = serialization_test_pb2.MyCustomRepresentation()
    serialized.representation.Unpack(proto)
    self.assertEqual(proto.index, my_custom.index)
    self.assertEqual(proto.name, my_custom.name)

  def testCustomClassDeserialization(self):
    # [FUNCTION DOC]
    # Scenario: serialize then deserialize a custom object. Expects an instance of the same class
    # with equal index and name.
    original = MyCustomClass(1234, "my_name")
    serialized = serialization.serialize(original)
    deserialized = serialization.deserialize(serialized)

    self.assertIsInstance(deserialized, MyCustomClass)
    self.assertEqual(deserialized.index, original.index)
    self.assertEqual(deserialized.name, original.name)

  def testCompositeClassSerialization(self):
    # [FUNCTION DOC]
    # Scenario: serialize a composite of three custom objects. Expects the composite message type
    # and that each unpacked element equals the independently serialized custom object.
    my_composite = MyCompositeClass(
        MyCustomClass(1, "name_1"), MyCustomClass(2, "name_2"),
        MyCustomClass(3, "name_3"))
    serialized = serialization.serialize(my_composite)

    self.assertTrue(
        serialized.representation.Is(
            serialization_test_pb2.MyCompositeRepresentation.DESCRIPTOR))

    proto = serialization_test_pb2.MyCompositeRepresentation()
    serialized.representation.Unpack(proto)

    self.assertEqual(proto.elements[0],
                     serialization.serialize(MyCustomClass(1, "name_1")))
    self.assertEqual(proto.elements[1],
                     serialization.serialize(MyCustomClass(2, "name_2")))
    self.assertEqual(proto.elements[2],
                     serialization.serialize(MyCustomClass(3, "name_3")))

  def testCompositeClassDeserialization(self):
    # [FUNCTION DOC]
    # Scenario: serialize then deserialize a composite of three custom objects. Expects a
    # composite whose elements keep their indexes (1, 2, 3) and names.
    original = MyCompositeClass(
        MyCustomClass(1, "name_1"), MyCustomClass(2, "name_2"),
        MyCustomClass(3, "name_3"))
    serialized = serialization.serialize(original)
    deserialized = serialization.deserialize(serialized)

    self.assertIsInstance(deserialized, MyCompositeClass)

    self.assertEqual(deserialized.elements[0].index, 1)
    self.assertEqual(deserialized.elements[1].index, 2)
    self.assertEqual(deserialized.elements[2].index, 3)

    self.assertEqual(deserialized.elements[0].name, "name_1")
    self.assertEqual(deserialized.elements[1].name, "name_2")
    self.assertEqual(deserialized.elements[2].name, "name_3")

  def testNonUniqueProto(self):
    # [FUNCTION DOC]
    # Scenario: define a new class that reuses an already registered message class and try to
    # register it. Expects a value error naming both classes and the message class.
    class ClassThatReusesProto(serialization.Serializable):
      # [CLASS DOC]
      # HELPER: Local fixture claiming the same message class as the custom fixture.

      @classmethod
      def experimental_type_proto(cls):
        # [FUNCTION DOC]
        # Returns the message class already used by the custom fixture.
        return serialization_test_pb2.MyCustomRepresentation

      @classmethod
      def experimental_from_proto(cls, proto):
        # [FUNCTION DOC]
        # Unsupported: always raises a not-implemented error.
        raise NotImplementedError

      def experimental_as_proto(self):
        # [FUNCTION DOC]
        # Unsupported: always raises a not-implemented error.
        raise NotImplementedError

    with self.assertRaisesRegex(
        ValueError,
        ("Existing Python class MyCustomClass already has "
         "MyCustomRepresentation as its associated proto representation. "
         "Please ensure ClassThatReusesProto has a unique proto representation."
        )):
      serialization.register_serializable(ClassThatReusesProto)

  def testWrongProto(self):
    # [FUNCTION DOC]
    # Scenario: a local class declares one message class but its as-proto method returns another.
    # Serializing an instance is expected to raise a value error saying a different message type
    # was returned than declared.

    class ClassReturningWrongProto(serialization.Serializable):
      # [CLASS DOC]
      # HELPER: Local fixture whose declared message class disagrees with what it returns.

      @classmethod
      def experimental_type_proto(cls):
        # [FUNCTION DOC]
        # Returns the generic serialized TraceType message class (the declared class).
        return serialization.SerializedTraceType

      @classmethod
      def experimental_from_proto(cls, proto):
        # [FUNCTION DOC]
        # Unsupported: always raises a not-implemented error.
        raise NotImplementedError

      def experimental_as_proto(self):
        # [FUNCTION DOC]
        # Returns a custom test message, which differs from the declared message class.
        return serialization_test_pb2.MyCustomRepresentation()

    with self.assertRaisesRegex(
        ValueError,
        ("ClassReturningWrongProto returned different type of proto than "
         "specified by experimental_type_proto()")):
      serialization.serialize(ClassReturningWrongProto())

  def testSerializableSuperClass(self):
    # [FUNCTION DOC]
    # Scenario: serialize and deserialize an instance of each of the three subclasses of the
    # shared base fixture. Expects each to come back equal (same subclass).
    self.assertEqual(
        serialization.deserialize(
            serialization.serialize(SerializableFromSuperClassOne())),
        SerializableFromSuperClassOne())
    self.assertEqual(
        serialization.deserialize(
            serialization.serialize(SerializableFromSuperClassTwo())),
        SerializableFromSuperClassTwo())
    self.assertEqual(
        serialization.deserialize(
            serialization.serialize(SerializableFromSuperClassThree())),
        SerializableFromSuperClassThree())


if __name__ == "__main__":
  test.main()
