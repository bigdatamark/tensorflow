# [MODULE DOC]
# Unit tests for the generic registry class in the framework registry module. Covers basic
# registration and lookup for both a class and a function (parameterized), listing of keys, and
# the error raised when a name is registered twice. Defines a trivial function and a nested
# class as registration candidates. Running the file starts the TensorFlow test runner.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for tensorflow.ops.registry."""

from absl.testing import parameterized

from tensorflow.python.framework import registry
from tensorflow.python.platform import test


def bar():
  # [FUNCTION DOC]
  # HELPER: trivial function used as a registration candidate in the tests. Does nothing and
  # returns nothing.
  pass


class RegistryTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case for the registry class. Shares a nested class and a module-level
  # function as candidates; each test builds its own fresh registry so there is no shared state.

  class Foo(object):
    # [CLASS DOC]
    # HELPER: empty class used as a registration candidate in the registry tests.
    pass

  # Test the registry basics on both classes (Foo) and functions (bar).
  @parameterized.parameters([Foo, bar])
  def testRegistryBasics(self, candidate):
    # [FUNCTION DOC]
    # Parameterized over a class and a function (the candidate). Scenario: a fresh registry
    # named for the test. Asserts that lookup of a missing key raises a lookup error; after
    # registering with the default name the candidate is found under its own name; after
    # registering again under an explicit key it is found under that key; and the sorted list of
    # keys equals the explicit key plus the candidate's own name.
    myreg = registry.Registry('testRegistry')
    with self.assertRaises(LookupError):
      myreg.lookup('testKey')
    myreg.register(candidate)
    self.assertEqual(myreg.lookup(candidate.__name__), candidate)
    myreg.register(candidate, 'testKey')
    self.assertEqual(myreg.lookup('testKey'), candidate)
    self.assertEqual(
        sorted(myreg.list()), sorted(['testKey', candidate.__name__]))

  def testDuplicate(self):
    # [FUNCTION DOC]
    # Scenario: register a function under one key, then register it again under the same key.
    # Asserts that a key error is raised whose message names the registry, the duplicated key
    # and the previous registration location (function name, a Python file name and a line
    # number).
    myreg = registry.Registry('testbar')
    myreg.register(bar, 'Bar')
    with self.assertRaisesRegex(
        KeyError, r'Registering two testbar with name \'Bar\'! '
        r'\(Previous registration was in [^ ]+ .*.py:[0-9]+\)'):
      myreg.register(bar, 'Bar')


if __name__ == '__main__':
  test.main()
