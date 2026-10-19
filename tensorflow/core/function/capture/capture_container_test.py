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
# [MODULE DOC]
# Unit tests for the capture container module: one test class for the mutation-tracking
# dictionary (behaves like a normal dictionary, flags mutations) and one for the function
# capture container (adding, replacing, popping, resetting, clearing and exposing views).
# ==============================================================================
"""Tests for tf.function capture containers."""
import copy

from absl.testing import parameterized

from tensorflow.core.function import trace_type
from tensorflow.core.function.capture import capture_container
from tensorflow.python.platform import test


class MutationAwareDictTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the mutation-tracking ordered dictionary: read behavior matches a plain
  # dictionary, and mutating operations set the mutation flag. Shared fixture is the helper
  # that builds a plain dictionary and a tracking copy with the same three entries.

  def _prepare_dict(self):
    # [FUNCTION DOC]
    # HELPER: fixture builder for the tests in this class. Returns a pair of a plain dictionary
    # with keys a, b, c mapped to one, two, three and a mutation-tracking dictionary built from a
    # copy of it.
    d = {"a": 1, "b": 2, "c": 3}
    mutation_d = capture_container.MutationAwareDict(copy.copy(d))
    return d, mutation_d

  @parameterized.parameters(
      ("__contains__", "a"),
      ("__contains__", "not_exist"),
      ("__len__", None),
      ("__getitem__", "a"))
  def test_same_behavior_with_normal_dict(self, method, arg):
    # [FUNCTION DOC]
    # Parameterized over a method name (membership check for present and absent keys, length,
    # item lookup) and its argument (null means no argument). Calls the same read-only method on
    # the plain and tracking dictionaries and asserts equal results.
    d, mutation_d = self._prepare_dict()
    d_method = getattr(d, method)
    mutation_d_method = getattr(mutation_d, method)
    if arg is None:
      self.assertEqual(d_method(), mutation_d_method())
    else:
      self.assertEqual(d_method(arg), mutation_d_method(arg))

  @parameterized.parameters(
      ("pop",),
      ("__delitem__",))
  def test_pop_and_del(self, method):
    # [FUNCTION DOC]
    # Parameterized over the removal method (pop or item deletion). Removes key b from both the
    # plain and tracking dictionaries and asserts their remaining keys and values are identical.
    d, mutation_d = self._prepare_dict()
    d_method = getattr(d, method)
    mutation_d_method = getattr(mutation_d, method)
    d_method("b")
    mutation_d_method("b")
    self.assertListEqual(list(d.keys()), list(mutation_d.keys()))
    self.assertListEqual(list(d.values()), list(mutation_d.values()))

  def test_mutatation_ops(self):
    # [FUNCTION DOC]
    # In sub-tests, performs set, pop, delete and clear on the tracking dictionary and asserts
    # the mutation flag is true after each. Note the flag is already true from creation and is
    # never reset between sub-tests, so these assertions cannot detect a missing flag update.
    # KNOWN ISSUE: because the flag starts true and is not cleared between steps, the test cannot
    # fail if an operation forgets to set the flag.
    _, d = self._prepare_dict()
    with self.subTest("set"):
      d["d"] = 4
      self.assertTrue(d.mutated)
    with self.subTest("pop"):
      d.pop("d")
      self.assertTrue(d.mutated)
    with self.subTest("del"):
      del d["c"]
      self.assertTrue(d.mutated)
    with self.subTest("clear"):
      d.clear()
      self.assertTrue(d.mutated)

  def test_mutated_property(self):
    # [FUNCTION DOC]
    # Asserts the flag is true on a fresh tracking dictionary, then sets it to false through
    # the property setter and asserts it reads false.
    _, d = self._prepare_dict()
    with self.subTest("initial_state"):
      self.assertTrue(d.mutated)
    with self.subTest("setter"):
      d.mutated = False
      self.assertFalse(d.mutated)


class FunctionCapturesTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the function capture container: adding and replacing captures, the cached list of
  # by-value pairs, and the combined capture-type mapping. Each test builds its own container.

  def test_add_or_replace(self):
    # [FUNCTION DOC]
    # Adds one by-value capture and one by-reference capture (with an explicit trace type),
    # asserts each lands in the matching dictionaries with length one, then re-adds the same
    # by-value key with new values and asserts the entry was replaced, not duplicated.
    fn_captures = capture_container.FunctionCaptures()
    fn_captures.add_or_replace("a", 1, -1, is_by_ref=False)
    fn_captures.add_or_replace("aa", 1, -1, 0, is_by_ref=True)

    with self.subTest("add_by_val"):
      self.assertLen(fn_captures.by_val_internal, 1)
      self.assertLen(fn_captures.by_val_external, 1)

    with self.subTest("add_by_ref"):
      self.assertLen(fn_captures.by_ref_internal, 1)
      self.assertLen(fn_captures.by_ref_external, 1)
      self.assertLen(fn_captures.by_ref_tracetype, 1)

    fn_captures.add_or_replace("a", 2, -2, is_by_ref=False)
    with self.subTest("replace_by_val"):
      self.assertLen(fn_captures.by_val_internal, 1)
      self.assertLen(fn_captures.by_val_external, 1)
      self.assertEqual(fn_captures.by_val_external["a"], 2)
      self.assertEqual(fn_captures.by_val_internal["a"], -2)

  def test_by_val_capture_tuples(self):
    # [FUNCTION DOC]
    # Walks the pair list through its life cycle in sub-tests: empty at start; after adding one
    # and two captures it holds the (external, internal) pairs in order; replacing a key updates
    # its pair in place; popping removes it; resetting captures with two lists rebuilds the list
    # from them; clearing empties it again.
    fn_captures = capture_container.FunctionCaptures()

    with self.subTest("initial_state"):
      self.assertEmpty(fn_captures.by_val_capture_tuples)

    with self.subTest("add"):
      fn_captures.add_or_replace("a", 1, -1, is_by_ref=False)
      self.assertLen(fn_captures.by_val_capture_tuples, 1)
      self.assertSequenceEqual(
          fn_captures.by_val_capture_tuples,
          ((1, -1),))

      fn_captures.add_or_replace("b", 2, -2, is_by_ref=False)
      self.assertLen(fn_captures.by_val_capture_tuples, 2)
      self.assertSequenceEqual(
          fn_captures.by_val_capture_tuples,
          ((1, -1), (2, -2)))

    with self.subTest("replace"):
      fn_captures.add_or_replace("a", 1, -3, is_by_ref=False)
      self.assertLen(fn_captures.by_val_capture_tuples, 2)
      self.assertSequenceEqual(
          fn_captures.by_val_capture_tuples,
          ((1, -3), (2, -2)))

    with self.subTest("pop"):
      fn_captures.pop("b", is_by_ref=False)
      self.assertSequenceEqual(
          fn_captures.by_val_capture_tuples,
          ((1, -3),))

    with self.subTest("reset"):
      fn_captures.reset_captures([10, 20], [-10, -20])
      self.assertSequenceEqual(
          fn_captures.by_val_capture_tuples,
          ((10, -10), (20, -20)))

    with self.subTest("clear"):
      fn_captures.clear()
      self.assertEmpty(fn_captures.by_val_capture_tuples)

  def test_capture_types(self):
    # [FUNCTION DOC]
    # Adds three by-value captures (trace types derived from the values one to three) and two
    # by-reference captures with explicit trace types derived from four and five, using fake
    # placeholder objects. Asserts the combined mapping has five entries and each key maps to the
    # expected trace type.
    class FakePlaceholder():
      # [CLASS DOC]
      # HELPER: empty stand-in for an internal placeholder tensor, used only in this test.
      pass

    fn_captures = capture_container.FunctionCaptures()
    fn_captures.add_or_replace("v1", 1, FakePlaceholder(), is_by_ref=False)
    fn_captures.add_or_replace("v2", 2, FakePlaceholder(), is_by_ref=False)
    fn_captures.add_or_replace("v3", 3, FakePlaceholder(), is_by_ref=False)
    fn_captures.add_or_replace(
        "r1", 1, FakePlaceholder(), trace_type.from_value(4), is_by_ref=True)
    fn_captures.add_or_replace(
        "r2", 2, FakePlaceholder(), trace_type.from_value(5), is_by_ref=True)

    self.assertLen(fn_captures.capture_types, 5)
    self.assertEqual(fn_captures.capture_types["v1"], trace_type.from_value(1))
    self.assertEqual(fn_captures.capture_types["v2"], trace_type.from_value(2))
    self.assertEqual(fn_captures.capture_types["v3"], trace_type.from_value(3))
    self.assertEqual(fn_captures.capture_types["r1"], trace_type.from_value(4))
    self.assertEqual(fn_captures.capture_types["r2"], trace_type.from_value(5))


if __name__ == "__main__":
  test.main()
