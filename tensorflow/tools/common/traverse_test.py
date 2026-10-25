# [MODULE DOC]
# Unit tests for the traverse function: termination on cyclic class references, traversal across
# modules via the two fixture modules, member listing for a class, and no visits for a
# non-class, non-module root. Defines a recording visitor and one test case class. Runs under
# the googletest runner when executed as a script. No import-time side effects.
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
"""Tests for Python module traversal."""

from tensorflow.python.platform import googletest
from tensorflow.tools.common import test_module1
from tensorflow.tools.common import test_module2
from tensorflow.tools.common import traverse


class TestVisitor(object):
  # [CLASS DOC]
  # HELPER: Recording visitor shared by the tests. Keeps a list of every visit as a triple of
  # path, parent and children list, in call order.

  def __init__(self):
    # [FUNCTION DOC]
    # HELPER: Initializes an empty call log. No inputs, returns nothing.
    self.call_log = []

  def __call__(self, path, parent, children):
    # [FUNCTION DOC]
    # HELPER: Visitor callback; appends the path, parent and children list (not a copy) to the
    # call log. Returns nothing.
    self.call_log += [(path, parent, children)]


class TraverseTest(googletest.TestCase):
  # [CLASS DOC]
  # Test case covering the traverse function; uses the recording visitor and the two fixture
  # modules.

  def test_cycle(self):
    # [FUNCTION DOC]
    # Scenario: a locally defined class that has a class attribute referencing itself. Action:
    # traverse it with the recording visitor. Expected: traversal terminates (cycle detection);
    # nothing else is asserted.

    class Cyclist(object):
      # [CLASS DOC]
      # HELPER: Empty local class made cyclic by assigning itself as an attribute after the
      # definition.
      pass
    Cyclist.cycle = Cyclist

    visitor = TestVisitor()
    traverse.traverse(Cyclist, visitor)
    # We simply want to make sure we terminate.

  def test_module(self):
    # [FUNCTION DOC]
    # Scenario: traverse the first fixture module. Expected: the visited parents include the
    # class from the first fixture module and the class from the second one, showing that
    # traversal descends through imported modules.
    visitor = TestVisitor()
    traverse.traverse(test_module1, visitor)

    called = [parent for _, parent, _ in visitor.call_log]

    self.assertIn(test_module1.ModuleClass1, called)
    self.assertIn(test_module2.ModuleClass2, called)

  def test_class(self):
    # [FUNCTION DOC]
    # Scenario: traverse the recording visitor class itself. Expected: the first visit has that
    # class as parent and its children names include the constructor and call dunder methods.
    # Other visited members (base class, class attribute) are deliberately not asserted.
    visitor = TestVisitor()
    traverse.traverse(TestVisitor, visitor)
    self.assertEqual(TestVisitor,
                     visitor.call_log[0][1])
    # There are a bunch of other members, but make sure that the ones we know
    # about are there.
    self.assertIn('__init__', [name for name, _ in visitor.call_log[0][2]])
    self.assertIn('__call__', [name for name, _ in visitor.call_log[0][2]])

    # There are more classes descended into, at least __class__ and
    # __class__.__base__, neither of which are interesting to us, and which may
    # change as part of Python version etc., so we don't test for them.

  def test_non_class(self):
    # [FUNCTION DOC]
    # Scenario: traverse an integer root. Expected: the visitor is never called, so its call
    # log is empty.
    integer = 5
    visitor = TestVisitor()
    traverse.traverse(integer, visitor)
    self.assertEqual([], visitor.call_log)


if __name__ == '__main__':
  googletest.main()
