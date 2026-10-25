# [MODULE DOC]
# Unit tests for the object_identity module. Two test classes: one for the identity wrapper
# (equality, hashing and use inside nest utilities) and one for the identity set (difference,
# discard and clear). Run through the standard TensorFlow test main. No import-time side effects.

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
"""Unit tests for object_identity."""

from tensorflow.python.platform import test
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity


class ObjectIdentityWrapperTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the internal identity wrapper class: comparison semantics and compatibility with
  # tf.nest flatten and map-structure. No shared fixtures.

  def testWrapperNotEqualToWrapped(self):
    # [FUNCTION DOC]
    # Scenario: an object with a settable hash is wrapped twice. Asserts that wrappers are equal to
    # themselves and each other, that unwrapping returns the original, that comparing the raw object
    # with a wrapper (equality or inequality) raises a type error, that the raw object is not found
    # in a set holding its wrapper, and that after making the object's hash equal to its identity
    # number (forcing a hash collision) membership testing raises a type error.
    class SettableHash(object):
      # [CLASS DOC]
      # HELPER: test-local class whose hash value can be changed, to provoke hash collisions with
      # wrappers. Holds one attribute, the hash value (initially a fixed number).

      def __init__(self):
        # [FUNCTION DOC]
        # HELPER: sets the initial hash value attribute to a fixed number. Returns nothing.
        self.hash_value = 8675309

      def __hash__(self):
        # [FUNCTION DOC]
        # HELPER: returns the current hash value attribute, which tests may change.
        return self.hash_value

    o = SettableHash()
    wrap1 = object_identity._ObjectIdentityWrapper(o)
    wrap2 = object_identity._ObjectIdentityWrapper(o)

    self.assertEqual(wrap1, wrap1)
    self.assertEqual(wrap1, wrap2)
    self.assertEqual(o, wrap1.unwrapped)
    self.assertEqual(o, wrap2.unwrapped)
    with self.assertRaises(TypeError):
      bool(o == wrap1)
    with self.assertRaises(TypeError):
      bool(wrap1 != o)

    self.assertNotIn(o, set([wrap1]))
    o.hash_value = id(o)
    # Since there is now a hash collision we raise an exception
    with self.assertRaises(TypeError):
      bool(o in set([wrap1]))

  def testNestFlatten(self):
    # [FUNCTION DOC]
    # Scenario: three wrappers are placed in a deeply nested list/tuple structure. Asserts that
    # nest flatten returns them as atoms, in order, so wrappers are treated as leaves.
    a = object_identity._ObjectIdentityWrapper('a')
    b = object_identity._ObjectIdentityWrapper('b')
    c = object_identity._ObjectIdentityWrapper('c')
    flat = nest.flatten([[[(a, b)]], c])
    self.assertEqual(flat, [a, b, c])

  def testNestMapStructure(self):
    # [FUNCTION DOC]
    # Scenario: two dictionaries keyed by the same wrapper hold different wrapper values. Maps a
    # pairing function over both with nest map-structure. Asserts the result is a dictionary with
    # the wrapper key mapped to the pair of values, showing wrappers work as sortable dict keys.
    k = object_identity._ObjectIdentityWrapper('k')
    v1 = object_identity._ObjectIdentityWrapper('v1')
    v2 = object_identity._ObjectIdentityWrapper('v2')
    struct = nest.map_structure(lambda a, b: (a, b), {k: v1}, {k: v2})
    self.assertEqual(struct, {k: (v1, v2)})


class ObjectIdentitySetTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the identity set class: difference, discard and clear. No shared fixtures.

  def testDifference(self):
    # [FUNCTION DOC]
    # Scenario: two identity sets built from overlapping objects a, b and b, c. Takes the
    # difference of the first minus the second. Asserts that only a remains (b and c absent).

    class Element(object):
      # [CLASS DOC]
      # HELPER: empty test-local class whose instances serve as distinct set members.
      pass

    a = Element()
    b = Element()
    c = Element()
    set1 = object_identity.ObjectIdentitySet([a, b])
    set2 = object_identity.ObjectIdentitySet([b, c])
    diff_set = set1.difference(set2)
    self.assertIn(a, diff_set)
    self.assertNotIn(b, diff_set)
    self.assertNotIn(c, diff_set)

  def testDiscard(self):
    # [FUNCTION DOC]
    # Scenario: an identity set of two plain objects; one is discarded. Asserts the discarded
    # object is no longer a member and the other still is.
    a = object()
    b = object()
    set1 = object_identity.ObjectIdentitySet([a, b])
    set1.discard(a)
    self.assertIn(b, set1)
    self.assertNotIn(a, set1)

  def testClear(self):
    # [FUNCTION DOC]
    # Scenario: an identity set of two plain objects is cleared. Asserts its length becomes zero.
    a = object()
    b = object()
    set1 = object_identity.ObjectIdentitySet([a, b])
    set1.clear()
    self.assertLen(set1, 0)


if __name__ == '__main__':
  test.main()
