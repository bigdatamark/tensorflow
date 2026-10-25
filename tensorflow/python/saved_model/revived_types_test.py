# [MODULE DOC]
# Unit tests for revived type registration matching. At import time it registers a test type
# with four versions (differing in producer, consumer and bad consumer constraints) and then
# checks which version is saved and which is chosen on load for various protos.
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
"""Tests for revived type matching."""

from tensorflow.core.framework import versions_pb2
from tensorflow.core.protobuf import saved_object_graph_pb2
from tensorflow.python.platform import test
from tensorflow.python.saved_model import revived_types
from tensorflow.python.trackable import autotrackable


class CustomTestClass(autotrackable.AutoTrackable):
  # [CLASS DOC]
  # HELPER: trackable test type whose single attribute, version, records which registered
  # version's object factory created it. Registered below as the type test_type.

  def __init__(self, version):
    # [FUNCTION DOC]
    # Input: version, the integer identifying which registration created this object. Stores it
    # as an attribute. Returns nothing.
    self.version = version


revived_types.register_revived_type(
    "test_type",
    lambda obj: isinstance(obj, CustomTestClass),
    versions=[
        revived_types.VersionedTypeRegistration(
            object_factory=lambda _: CustomTestClass(1),
            version=1, min_producer_version=1,
            min_consumer_version=1),
        revived_types.VersionedTypeRegistration(
            object_factory=lambda _: CustomTestClass(2),
            version=2, min_producer_version=2, min_consumer_version=1),
        revived_types.VersionedTypeRegistration(
            object_factory=lambda _: CustomTestClass(3),
            version=3, min_producer_version=3, min_consumer_version=2),
        revived_types.VersionedTypeRegistration(
            object_factory=lambda _: CustomTestClass(4),
            version=4, min_producer_version=4, min_consumer_version=2,
            bad_consumers=[3]),
    ]
)


class RegistrationMatchingTest(test.TestCase):
  # [CLASS DOC]
  # Test case for version matching in the revived type registry. Relies on the module-level
  # registration of test_type, whose versions are: 1 (producer min 1, consumer min 1), 2
  # (producer min 2, consumer min 1), 3 (producer min 3, consumer min 2) and 4 (producer min 4,
  # consumer min 2, bad consumer 3).

  def test_save_typecheck(self):
    # [FUNCTION DOC]
    # Scenario: serializing a plain auto-trackable object, which matches no registered type,
    # returns nothing (null).
    self.assertIs(revived_types.serialize(autotrackable.AutoTrackable()), None)

  def test_load_identifier_not_found(self):
    # [FUNCTION DOC]
    # Scenario: deserializing a proto with an unregistered identifier returns nothing (null).
    nothing_matches = revived_types.deserialize(
        saved_object_graph_pb2.SavedUserObject(
            identifier="_unregistered_type",
            version=versions_pb2.VersionDef(
                producer=1,
                min_consumer=1,
                bad_consumers=[])))
    self.assertIs(nothing_matches, None)

  def test_most_recent_version_saved(self):
    # [FUNCTION DOC]
    # Scenario: serializes an instance of the test class. Asserts the proto carries the bad
    # consumer list of the newest version (three), and that deserializing it yields an instance
    # created by version four.
    serialized = revived_types.serialize(CustomTestClass(None))
    self.assertEqual([3], serialized.version.bad_consumers)
    deserialized, _ = revived_types.deserialize(serialized)
    self.assertIsInstance(deserialized, CustomTestClass)
    self.assertEqual(4, deserialized.version)

  def test_min_consumer_version(self):
    # [FUNCTION DOC]
    # Scenario: a proto for test_type with producer and minimum consumer version five, higher
    # than every registered version. Asserts deserialization returns nothing (null).
    nothing_matches = revived_types.deserialize(
        saved_object_graph_pb2.SavedUserObject(
            identifier="test_type",
            version=versions_pb2.VersionDef(
                producer=5,
                min_consumer=5,
                bad_consumers=[])))
    self.assertIs(nothing_matches, None)

  def test_bad_versions(self):
    # [FUNCTION DOC]
    # Scenario: a proto for test_type with producer five, minimum consumer one, and bad
    # consumers four and three. Versions four and three are skipped, so asserts the object
    # comes from version two.
    deserialized, _ = revived_types.deserialize(
        saved_object_graph_pb2.SavedUserObject(
            identifier="test_type",
            version=versions_pb2.VersionDef(
                producer=5,
                min_consumer=1,
                bad_consumers=[4, 3])))
    self.assertEqual(2, deserialized.version)

  def test_min_producer_version(self):
    # [FUNCTION DOC]
    # Scenario: a proto for test_type with producer version three, minimum consumer zero and no
    # bad consumers. Version four requires producer four so is skipped; asserts the object
    # comes from version three.
    deserialized, _ = revived_types.deserialize(
        saved_object_graph_pb2.SavedUserObject(
            identifier="test_type",
            version=versions_pb2.VersionDef(
                producer=3,
                min_consumer=0,
                bad_consumers=[])))
    self.assertEqual(3, deserialized.version)


if __name__ == "__main__":
  test.main()
