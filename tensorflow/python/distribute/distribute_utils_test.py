# [MODULE DOC]
# Unit tests for the regroup and replica-selection utilities in the distribute utils module:
# regrouping nests of per-replica values (tuples, lists, dicts, custom mapping types, named
# tuples wrapped in proxies) into per-replica or mirrored wrappers, undoing that with replica
# selection, identical-object handling, single-device behaviour, and distributed variable
# detection. No import-time side effects.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for utility functions in distribute_utils."""

import collections
import collections.abc

from absl.testing import parameterized
import wrapt

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import values
from tensorflow.python.eager import test
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1


def _nested_value(d):
  # [FUNCTION DOC]
  # HELPER: builds the nested test structure used by several tests.
  # Takes a string suffix and returns a tuple of three items: a string, a list holding a string,
  # a mapping of two strings, and another string, and a final string; every string is a fixed
  # letter followed by the suffix, so different suffixes simulate different replicas' values.
  return ("a" + d, ["b" + d, {"c": "d" + d, "e": "f" + d}, "g" + d], "h" + d)


class RegroupAndSelectDeviceTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests the regroup function and the replica selection functions of the distribute utils
  # module across nested structures, wrapper classes and variables.

  def _is_per_replica(self, result, expected, klass=values.PerReplica):
    # [FUNCTION DOC]
    # HELPER: assertion helper. Asserts the result is an instance of the given wrapper class
    # (default per-replica) and that its per-replica component values equal the expected
    # sequence, in order.
    self.assertIsInstance(result, klass)
    for i, exp in enumerate(expected):
      self.assertEqual(exp, result.values[i])

  def testNested(self):
    # [FUNCTION DOC]
    # Scenario: regroup two replicas' nested values. Asserts the tuple/list/mapping structure is
    # preserved with per-replica wrappers at the leaves, that replica selection restores each
    # original nest, and that mirrored-only selection raises a type error since the wrappers are
    # not mirrored.
    result = distribute_utils.regroup((_nested_value("1"), _nested_value("2")))
    self.assertIsInstance(result, tuple)
    self.assertLen(result, 3)
    self._is_per_replica(result[0], ["a1", "a2"])
    self._is_per_replica(result[2], ["h1", "h2"])

    self.assertIsInstance(result[1], list)
    self.assertLen(result[1], 3)
    self._is_per_replica(result[1][0], ["b1", "b2"])
    self._is_per_replica(result[1][2], ["g1", "g2"])

    self.assertIsInstance(result[1][1], dict)
    self.assertEqual(set(["c", "e"]), set(result[1][1].keys()))
    self._is_per_replica(result[1][1]["c"], ["d1", "d2"])
    self._is_per_replica(result[1][1]["e"], ["f1", "f2"])

    # Also test that we can undo the merge using select_replica()
    self.assertEqual(_nested_value("1"),
                     distribute_utils.select_replica(0, result))
    self.assertEqual(_nested_value("2"),
                     distribute_utils.select_replica(1, result))
    # select_device_mirrored() should fail due to non-mirrored values
    with self.assertRaises(TypeError):
      distribute_utils.select_replica_mirrored(0, result)
    with self.assertRaises(TypeError):
      distribute_utils.select_replica_mirrored(1, result)

  def testRegroupKeepsDictBasedClass(self):
    # [FUNCTION DOC]
    # Scenario: regroup two instances of a dictionary subclass. Asserts the result keeps the
    # subclass type and each entry is a per-replica value of the two inputs.
    class DictBasedClass(dict):
      # [CLASS DOC]
      # HELPER: trivial dictionary subclass used to check the regroup result keeps its type.
      """Dummy class inherited from a dict."""

    result = distribute_utils.regroup(
        (DictBasedClass(a="a1", b="b1"), DictBasedClass(a="a2", b="b2")))
    self.assertIsInstance(result, DictBasedClass)
    self._is_per_replica(result["a"], ["a1", "a2"])
    self._is_per_replica(result["b"], ["b1", "b2"])

  def testRegroupCollectionsMapping(self):
    # [FUNCTION DOC]
    # Scenario: regroup two instances of a custom read-only mapping type (not a dictionary).
    # Asserts the result keeps that type and entries are per-replica values.

    class CollectionsMappingBasedClass(collections.abc.Mapping):
      # [CLASS DOC]
      # HELPER: minimal read-only mapping type backed by an internal dictionary, used to check
      # regroup supports generic mapping types.
      """Class inherited from collections.abc.Mapping."""

      def __init__(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Builds the internal dictionary from the given positional and keyword arguments.
        self._d = dict(*args, **kwargs)

      def __getitem__(self, key):
        # [FUNCTION DOC]
        # Returns the value stored for the key (key-missing error if absent).
        return self._d.__getitem__(key)

      def __iter__(self):
        # [FUNCTION DOC]
        # Returns an iterator over the keys.
        return iter(self._d)

      def __len__(self):
        # [FUNCTION DOC]
        # Returns the number of entries.
        return len(self._d)

    result = distribute_utils.regroup(
        (CollectionsMappingBasedClass(a="a1", b="b1"),
         CollectionsMappingBasedClass(a="a2", b="b2")))
    self.assertIsInstance(result, CollectionsMappingBasedClass)
    self._is_per_replica(result["a"], ["a1", "a2"])
    self._is_per_replica(result["b"], ["b1", "b2"])

  def testWrapClass(self):
    # [FUNCTION DOC]
    # Scenario: regroup two nests with the mirrored wrapper class. Asserts the structure and the
    # mirrored wrappers at leaves, that replica selection restores the originals, and that the
    # mirrored-only selection now succeeds.
    # Normally a mirrored value would be the same across devices, but
    # for a test it is convenient to be able to tell the values apart.
    result = distribute_utils.regroup((_nested_value("1"), _nested_value("2")),
                                      values.Mirrored)
    self.assertIsInstance(result, tuple)
    self.assertLen(result, 3)
    self._is_per_replica(result[0], ["a1", "a2"], values.Mirrored)
    self._is_per_replica(result[2], ["h1", "h2"], values.Mirrored)

    self.assertIsInstance(result[1], list)
    self.assertLen(result[1], 3)
    self._is_per_replica(result[1][0], ["b1", "b2"], values.Mirrored)
    self._is_per_replica(result[1][2], ["g1", "g2"], values.Mirrored)

    self.assertIsInstance(result[1][1], dict)
    self.assertEqual(set(["c", "e"]), set(result[1][1].keys()))
    self._is_per_replica(result[1][1]["c"], ["d1", "d2"], values.Mirrored)
    self._is_per_replica(result[1][1]["e"], ["f1", "f2"], values.Mirrored)

    # Also test that we can undo the merge using select_replica()
    self.assertEqual(_nested_value("1"),
                     distribute_utils.select_replica(0, result))
    self.assertEqual(_nested_value("2"),
                     distribute_utils.select_replica(1, result))
    # Values are marked as mirrored, so select_device_mirrored() is allowed.
    self.assertEqual(_nested_value("1"),
                     distribute_utils.select_replica_mirrored(0, result))
    self.assertEqual(_nested_value("2"),
                     distribute_utils.select_replica_mirrored(1, result))

  def testWrapAListOfTwoTuples(self):
    # [FUNCTION DOC]
    # Scenario: regroup a list of two tuples. Asserts a tuple of two per-replica values holding
    # the first and second elements respectively.
    result = distribute_utils.regroup([("1", "2"), ("3", "4")])
    self.assertIsInstance(result, tuple)
    self.assertLen(result, 2)
    self._is_per_replica(result[0], ("1", "3"), values.PerReplica)
    self._is_per_replica(result[1], ("2", "4"), values.PerReplica)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.mirrored_strategy_with_one_cpu,
          ],
          mode=["graph", "eager"],
      ))
  def testMirroredContainer(self, distribution):
    # [FUNCTION DOC]
    # Scenario (graph and eager, mirrored strategy with GPU-and-CPU or one CPU): create a
    # sum-aggregated variable in scope. Asserts the variable is a distributed variable and that
    # regrouping its component values yields a distributed variable again (the container).
    with distribution.scope():
      v = variable_v1.VariableV1(
          1., aggregation=variable_scope.VariableAggregation.SUM)
    self.assertTrue(distribute_utils.is_distributed_variable(v))
    self.assertTrue(distribute_utils.is_distributed_variable(
        distribute_utils.regroup(v.values)))

  def testSameId(self):
    # [FUNCTION DOC]
    # Scenario: regroup replicas where one element is the identical object on both. Asserts that
    # element is returned unwrapped while the differing element is wrapped, and that replica
    # selection recovers each replica's tuple.
    foo = object()
    result = distribute_utils.regroup((("a", foo), ("b", foo)))
    self.assertIsInstance(result, tuple)
    self.assertLen(result, 2)
    self._is_per_replica(result[0], ["a", "b"])
    self.assertIs(foo, result[1])

    # Test select_replica(), should undo the merge done by regroup().
    result_0 = distribute_utils.select_replica(0, result)
    self.assertIsInstance(result_0, tuple)
    self.assertLen(result_0, 2)
    self.assertEqual("a", result_0[0])
    self.assertIs(foo, result_0[1])
    result_1 = distribute_utils.select_replica(1, result)
    self.assertIsInstance(result_1, tuple)
    self.assertLen(result_1, 2)
    self.assertEqual("b", result_1[0])
    self.assertIs(foo, result_1[1])

  def testOneDevice(self):
    # [FUNCTION DOC]
    # Scenario: regroup a single replica's nest. Asserts the result equals the input and replica
    # selection of replica zero returns it too (both behave like identity).
    result = distribute_utils.regroup((_nested_value("1"),))
    # On one device regroup() and select_replica() are basically identity.
    self.assertEqual(_nested_value("1"), result)
    self.assertEqual(_nested_value("1"),
                     distribute_utils.select_replica(0, result))

  def testWrappedNamedTuple(self):
    # [FUNCTION DOC]
    # Scenario: regroup two named tuples wrapped in transparent object proxies. Asserts the
    # result exposes the named fields, each holding the per-replica values of that field.
    Point = collections.namedtuple("Point", ["x", "y"])
    point1 = Point(x=0, y=2)
    point2 = Point(x=1, y=3)
    wrapped1 = wrapt.ObjectProxy(point1)
    wrapped2 = wrapt.ObjectProxy(point2)
    result = distribute_utils.regroup([wrapped1, wrapped2])
    self.assertEqual(result.x.values, (0, 1))
    self.assertEqual(result.y.values, (2, 3))

if __name__ == "__main__":
  test.main()
