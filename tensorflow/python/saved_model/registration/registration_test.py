# [MODULE DOC]
# Unit tests for the registration functions: registering serializable classes (default, custom
# package, custom name, subclass, predicate-based) and looking them up by object or name, error
# cases (non-class, non-callable predicate, duplicate names), and registering and validating
# checkpoint savers. Integration tests with real save and load live in the saving test module.
# Import-time side effects: registers four trackable classes in the global class registry.
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
"""Tests the registration functions.

For integration tests that use save and load functions, see
registration_saving_test.py.
"""

from absl.testing import parameterized

from tensorflow.python.eager import test
from tensorflow.python.saved_model import registration
from tensorflow.python.trackable import base


@registration.register_serializable()
class RegisteredClass(base.Trackable):
  # [CLASS DOC]
  # HELPER: empty trackable fixture registered under the default package, named by its class.
  pass


@registration.register_serializable(name="Subclass")
class RegisteredSubclass(RegisteredClass):
  # [CLASS DOC]
  # HELPER: empty subclass of the registered fixture, registered separately under the custom name
  # Subclass in the default package.
  pass


@registration.register_serializable(package="testing")
class CustomPackage(base.Trackable):
  # [CLASS DOC]
  # HELPER: empty trackable fixture registered under the package testing, named by its class.
  pass


@registration.register_serializable(package="testing", name="name")
class CustomPackageAndName(base.Trackable):
  # [CLASS DOC]
  # HELPER: empty trackable fixture registered under the package testing with the explicit name
  # name.
  pass


class SerializableRegistrationTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case for registering serializable classes and looking up their registered
  # names and classes, using the module-level registered fixture classes.

  @parameterized.parameters([
      (RegisteredClass, "Custom.RegisteredClass"),
      (RegisteredSubclass, "Custom.Subclass"),
      (CustomPackage, "testing.CustomPackage"),
      (CustomPackageAndName, "testing.name"),
  ])
  def test_registration(self, expected_cls, expected_name):
    # [FUNCTION DOC]
    # Parameterized over four fixture classes and their expected full registered names. Creates an
    # instance, asserts the registered class name for the instance equals the expected name, and
    # that looking up the name returns the same class.
    obj = expected_cls()
    self.assertEqual(registration.get_registered_class_name(obj), expected_name)
    self.assertIs(
        registration.get_registered_class(expected_name), expected_cls)

  def test_get_invalid_name(self):
    # [FUNCTION DOC]
    # Asserts that looking up a class by an unregistered name returns nothing/null.
    self.assertIsNone(registration.get_registered_class("invalid name"))

  def test_get_unregistered_class(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the object passed to the lookup is the class itself, not an instance, so the
    # test only shows that a class object (rather than an instance) matches no predicate.
    # Scenario: define a local unregistered trackable class and ask for its registered class name.
    # Asserts the result is nothing/null.

    class NotRegistered(base.Trackable):
      # [CLASS DOC]
      # HELPER: empty trackable class deliberately left unregistered.
      pass

    no_register = NotRegistered
    self.assertIsNone(registration.get_registered_class_name(no_register))

  def test_duplicate_registration(self):
    # [FUNCTION DOC]
    # Scenario: register a local class under the default package, then again under another
    # package. Asserts the name lookup prefers the newest registration, both names resolve to the
    # class, and registering a name that already exists (package testing, name CustomPackage)
    # raises a value error saying it has already been registered.

    @registration.register_serializable()
    class Duplicate(base.Trackable):
      # [CLASS DOC]
      # HELPER: empty trackable class registered more than once in this test.
      pass

    dup = Duplicate()
    self.assertEqual(
        registration.get_registered_class_name(dup), "Custom.Duplicate")
    # Registrations with different names are ok.
    registration.register_serializable(package="duplicate")(Duplicate)
    # Registrations are checked in reverse order.
    self.assertEqual(
        registration.get_registered_class_name(dup), "duplicate.Duplicate")
    # Both names should resolve to the same class.
    self.assertIs(
        registration.get_registered_class("Custom.Duplicate"), Duplicate)
    self.assertIs(
        registration.get_registered_class("duplicate.Duplicate"), Duplicate)

    # Registrations of the same name fails
    with self.assertRaisesRegex(ValueError, "already been registered"):
      registration.register_serializable(
          package="testing", name="CustomPackage")(
              Duplicate)

  def test_register_non_class_fails(self):
    # [FUNCTION DOC]
    # Scenario: apply the registration decorator to an instance instead of a class. Expects a
    # type error saying the registered object must be a class.
    obj = RegisteredClass()
    with self.assertRaisesRegex(TypeError, "must be a class"):
      registration.register_serializable()(obj)

  def test_register_bad_predicate_fails(self):
    # [FUNCTION DOC]
    # Scenario: register an existing class with a predicate that is not callable (the number
    # zero). Expects a type error saying the predicate must be callable.
    with self.assertRaisesRegex(TypeError, "must be callable"):
      registration.register_serializable(predicate=0)(RegisteredClass)

  def test_predicate(self):
    # [FUNCTION DOC]
    # Scenario: a local class with a boolean attribute is registered with a predicate that is true
    # only for instances whose flag is set. Asserts the flagged instance gets the registered name
    # and the unflagged one gets nothing. Then registers again with a predicate matching all
    # instances and asserts both instances now resolve to the newer name.

    class Predicate(base.Trackable):
      # [CLASS DOC]
      # HELPER: trackable with a flag attribute used by registration predicates in this test.

      def __init__(self, register_this):
        # [FUNCTION DOC]
        # Stores the boolean flag deciding whether the predicate accepts this instance.
        self.register_this = register_this

    registration.register_serializable(
        name="RegisterThisOnlyTrue",
        predicate=lambda x: isinstance(x, Predicate) and x.register_this)(
            Predicate)

    a = Predicate(True)
    b = Predicate(False)
    self.assertEqual(
        registration.get_registered_class_name(a),
        "Custom.RegisterThisOnlyTrue")
    self.assertIsNone(registration.get_registered_class_name(b))

    registration.register_serializable(
        name="RegisterAllPredicate",
        predicate=lambda x: isinstance(x, Predicate))(
            Predicate)

    self.assertEqual(
        registration.get_registered_class_name(a),
        "Custom.RegisterAllPredicate")
    self.assertEqual(
        registration.get_registered_class_name(b),
        "Custom.RegisterAllPredicate")


class CheckpointSaverRegistrationTest(test.TestCase):
  # [CLASS DOC]
  # Test case for registering checkpoint savers: invalid argument handling and the lookup and
  # validation functions. No shared fixtures.

  def test_invalid_registration(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the third and fourth error cases (invalid name containing a slash) are exact
    # duplicates of each other, so one of them adds no coverage.
    # Scenario: register savers with invalid arguments and assert the errors. A null package or
    # null name gives a type error saying it must be a string; a name with a slash gives a value
    # error about an invalid registered checkpoint saver; a null predicate gives a type error that
    # it must be callable; a null save or restore function gives a type error that it must be
    # callable.
    with self.assertRaisesRegex(TypeError, "must be string"):
      registration.register_checkpoint_saver(
          package=None,
          name="test",
          predicate=lambda: None,
          save_fn=lambda: None,
          restore_fn=lambda: None)
    with self.assertRaisesRegex(TypeError, "must be string"):
      registration.register_checkpoint_saver(
          name=None,
          predicate=lambda: None,
          save_fn=lambda: None,
          restore_fn=lambda: None)
    with self.assertRaisesRegex(ValueError,
                                "Invalid registered checkpoint saver."):
      registration.register_checkpoint_saver(
          package="package",
          name="t/est",
          predicate=lambda: None,
          save_fn=lambda: None,
          restore_fn=lambda: None)
    with self.assertRaisesRegex(ValueError,
                                "Invalid registered checkpoint saver."):
      registration.register_checkpoint_saver(
          package="package",
          name="t/est",
          predicate=lambda: None,
          save_fn=lambda: None,
          restore_fn=lambda: None)
    with self.assertRaisesRegex(
        TypeError,
        "The predicate registered to a checkpoint saver must be callable"
    ):
      registration.register_checkpoint_saver(
          name="test",
          predicate=None,
          save_fn=lambda: None,
          restore_fn=lambda: None)
    with self.assertRaisesRegex(TypeError, "The save_fn must be callable"):
      registration.register_checkpoint_saver(
          name="test",
          predicate=lambda: None,
          save_fn=None,
          restore_fn=lambda: None)
    with self.assertRaisesRegex(TypeError, "The restore_fn must be callable"):
      registration.register_checkpoint_saver(
          name="test",
          predicate=lambda: None,
          save_fn=lambda: None,
          restore_fn=None)

  def test_registration(self):
    # [FUNCTION DOC]
    # Scenario: register a saver (package Testing, name test_predicate) whose predicate checks for
    # an attribute, with trivial save and restore functions returning marker strings. Asserts the
    # attribute-less trackable has no saver name, then after setting the attribute the name is
    # Testing.test_predicate, the lookup functions return the marker functions, validation passes
    # for the matching object, fails with a "cannot be found" error for an unknown name and with a
    # "cannot be used" error for an object that fails the predicate.
    registration.register_checkpoint_saver(
        package="Testing",
        name="test_predicate",
        predicate=lambda x: hasattr(x, "check_attr"),
        save_fn=lambda: "save",
        restore_fn=lambda: "restore")
    x = base.Trackable()
    self.assertIsNone(registration.get_registered_saver_name(x))

    x.check_attr = 1
    saver_name = registration.get_registered_saver_name(x)
    self.assertEqual(saver_name, "Testing.test_predicate")

    self.assertEqual(registration.get_save_function(saver_name)(), "save")
    self.assertEqual(registration.get_restore_function(saver_name)(), "restore")

    registration.validate_restore_function(x, "Testing.test_predicate")
    with self.assertRaisesRegex(ValueError, "saver cannot be found"):
      registration.validate_restore_function(x, "Invalid.name")
    x2 = base.Trackable()
    with self.assertRaisesRegex(ValueError, "saver cannot be used"):
      registration.validate_restore_function(x2, "Testing.test_predicate")


if __name__ == "__main__":
  test.main()
