# [MODULE DOC]
# Unit tests for the public API visitor wrapper in the public_api module. Defines one test case
# class that checks that children are forwarded to the wrapped visitor, that private children
# are removed before the visitor is called, and that do-not-descend children are removed after
# it is called. Runs under the googletest runner when executed as a script. No import-time side
# effects.
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
"""Tests for tensorflow.tools.common.public_api."""

from tensorflow.python.platform import googletest
from tensorflow.tools.common import public_api


class PublicApiTest(googletest.TestCase):
  # [CLASS DOC]
  # Test case covering the PublicAPIVisitor wrapper. Shared fixture: a nested recording visitor
  # class that remembers visited paths, the last parent and a copy of the last children list.

  class TestVisitor(object):
    # [CLASS DOC]
    # HELPER: Recording visitor used as the wrapped visitor by the tests in this file.
    # Stores the set of visited paths, the last parent seen and a copy of the last children
    # list, so tests can inspect what the wrapper forwarded.

    def __init__(self):
      # [FUNCTION DOC]
      # HELPER: Test fixture setup. Takes no inputs; initializes an empty set of visited
      # paths and nothing/null for the last parent and last children. Returns nothing.
      self.symbols = set()
      self.last_parent = None
      self.last_children = None

    def __call__(self, path, parent, children):
      # [FUNCTION DOC]
      # HELPER: Visitor callback for the tests. Inputs: path string, parent object, list of
      # (name, child) pairs. Records the path in the symbol set, remembers the parent, and
      # stores a copy of the children list so later in-place removals do not change it.
      # Returns nothing.
      self.symbols.add(path)
      self.last_parent = parent
      self.last_children = list(children)  # Make a copy to preserve state.

  def test_call_forward(self):
    # [FUNCTION DOC]
    # Scenario: two ordinary public children are passed to the wrapper. Action: call the wrapper
    # with path 'test' and a dummy parent. Expected: the wrapped visitor is called exactly for
    # path 'test', receives the dummy parent, and sees both children unchanged.
    visitor = self.TestVisitor()
    children = [('name1', 'thing1'), ('name2', 'thing2')]
    public_api.PublicAPIVisitor(visitor)('test', 'dummy', children)
    self.assertEqual(set(['test']), visitor.symbols)
    self.assertEqual('dummy', visitor.last_parent)
    self.assertEqual([('name1', 'thing1'), ('name2', 'thing2')],
                     visitor.last_children)

  def test_private_child_removal(self):
    # [FUNCTION DOC]
    # Scenario: children include one public name and one underscore-prefixed private name.
    # Action: call the wrapper. Expected: the wrapped visitor only sees the public child, and
    # the caller's original children list was also modified in place to drop the private one.
    visitor = self.TestVisitor()
    children = [('name1', 'thing1'), ('_name2', 'thing2')]
    public_api.PublicAPIVisitor(visitor)('test', 'dummy', children)
    # Make sure the private symbols are removed before the visitor is called.
    self.assertEqual([('name1', 'thing1')], visitor.last_children)
    self.assertEqual([('name1', 'thing1')], children)

  def test_no_descent_child_removal(self):
    # [FUNCTION DOC]
    # Scenario: children include a public name and the name 'mock', which is on the
    # do-not-descend list only for the 'tf.test' parent. Action: call the wrapper with the
    # path 'test' (full path 'tf.test'). Expected: the wrapped visitor still sees both
    # children, but afterwards the caller's list no longer contains the 'mock' entry.
    visitor = self.TestVisitor()
    children = [('name1', 'thing1'), ('mock', 'thing2')]
    public_api.PublicAPIVisitor(visitor)('test', 'dummy', children)
    # Make sure not-to-be-descended-into symbols are removed after the visitor
    # is called.
    self.assertEqual([('name1', 'thing1'), ('mock', 'thing2')],
                     visitor.last_children)
    self.assertEqual([('name1', 'thing1')], children)


if __name__ == '__main__':
  googletest.main()
