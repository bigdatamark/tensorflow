# [MODULE DOC]
# Tests for the shared variable creator utility: name canonicalization rules and sharing of one
# variable across creators for several devices.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for SharedVariableCreator."""

from tensorflow.python.distribute import shared_variable_creator
from tensorflow.python.eager import test
from tensorflow.python.framework import test_util
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1


class CanonicalizeVariableNameTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the private name canonicalization helper of the shared variable creator module.

  def _canonicalize(self, name):
    # [FUNCTION DOC]
    # HELPER: calls the module's canonicalization function on name and returns its result.
    return shared_variable_creator._canonicalize_variable_name(name)

  def testNoName(self):
    # [FUNCTION DOC]
    # A missing name canonicalizes to the default name "Variable".
    self.assertEqual("Variable", self._canonicalize(None))

  def testPatternInMiddle(self):
    # [FUNCTION DOC]
    # Underscore-digit suffixes before slashes are removed from the middle of a name.
    self.assertEqual("foo/bar/baz", self._canonicalize("foo_1/bar_1/baz"))

  def testPatternAtEnd(self):
    # [FUNCTION DOC]
    # An underscore-digit suffix at the end of a name is removed.
    self.assertEqual("foo", self._canonicalize("foo_1"))

  def testWrongPatterns(self):
    # [FUNCTION DOC]
    # Names that do not match the suffix pattern (a trailing colon-zero, no underscore, a
    # non-digit after the underscore) are returned unchanged.
    self.assertEqual("foo_1:0", self._canonicalize("foo_1:0"))
    self.assertEqual("foo1", self._canonicalize("foo1"))
    self.assertEqual("foo_a", self._canonicalize("foo_a"))


class SharedVariableCreatorTest(test.TestCase):
  # [CLASS DOC]
  # Tests that creators made for different devices share variables.

  @test_util.run_in_graph_and_eager_modes
  def testSharedVariable(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. Builds creators for three devices over one shared store, then
    # creates a variable named foo under each device's creator scope; expects the second and third
    # to be the very same object as the first.

    shared_variable_store = {}
    num_devices = 3
    creator_fns = []
    for i in range(num_devices):
      creator_fn = shared_variable_creator.make_fn(shared_variable_store, i)
      creator_fns.append(creator_fn)

    with variable_scope.variable_creator_scope(creator_fns[0]):
      v0 = variable_v1.VariableV1(1.0, name="foo")

    with variable_scope.variable_creator_scope(creator_fns[1]):
      v1 = variable_v1.VariableV1(1.0, name="foo")

    with variable_scope.variable_creator_scope(creator_fns[2]):
      v2 = variable_v1.VariableV1(1.0, name="foo")

    # v1 and v2 should be same as v0
    self.assertIs(v1, v0)
    self.assertIs(v2, v0)


if __name__ == "__main__":
  test.main()
