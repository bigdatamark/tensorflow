# [MODULE DOC]
# Unit tests for the trackable data structures module: the append-only List and Mapping
# containers and the automatic wrappers (list, dict, tuple and namedtuple wrappers) applied when
# such Python containers are assigned to attributes of trackable objects. Covers JSON
# serialization, tracking of variables and weights, copying, hashing, in-place operators, the
# refusal to save after unsupported mutations, pickling, nest structure compatibility, caching
# of traced functions and dataset mapping. Run as a script it starts the test runner.
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
import collections
import copy
import json
import os
import pickle

from absl.testing import parameterized

from tensorflow.python.checkpoint import checkpoint as util
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import tensor_shape
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import data_structures
from tensorflow.python.util import nest
from tensorflow.python.util import serialization


class ListTests(test.TestCase):
  # [CLASS DOC]
  # Test case for the append-only List container: type restrictions, absent mutators, nesting,
  # variable and weight aggregation, copying, slicing, hashing and the multiplication operators.

  def testJSONSerialization(self):
    # [FUNCTION DOC]
    # Scenario: a trackable object gets a plain list attribute (so it becomes a list wrapper).
    # Dumps the wrapper to JSON with the library's custom type hook. Passes if no error is raised.
    obj = autotrackable.AutoTrackable()
    obj.l = [1]
    json.dumps(obj.l, default=serialization.get_json_type)

  def testNotTrackable(self):
    # [FUNCTION DOC]
    # Scenario: constructs a List containing an instance of a local class that is not trackable.
    # Asserts a value error is raised.
    class NotTrackable(object):
      # [CLASS DOC]
      # HELPER: local test fixture; a plain class that does not inherit from the trackable base.
      pass

    with self.assertRaises(ValueError):
      data_structures.List([NotTrackable()])

  def testCallNotImplemented(self):
    # [FUNCTION DOC]
    # Scenario: calls an empty List as if it were a function. Asserts a type error whose message
    # says the object is not callable.
    with self.assertRaisesRegex(TypeError, "not callable"):
      data_structures.List()(1.)  # pylint: disable=not-callable

  def testNoPop(self):
    # [FUNCTION DOC]
    # Scenario: tries to call pop on an empty List. Asserts an attribute error, because the
    # container is append-only and offers no removal methods.
    with self.assertRaises(AttributeError):
      data_structures.List().pop()

  def testNesting(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): a List nested inside another List; a small dense-like module with a
    # weight and bias variable is appended to the inner list and called on a ones tensor. Asserts
    # the outer list reports two variables and the first is a resource variable.
    with context.graph_mode():
      inner = data_structures.List()
      outer = data_structures.List([inner])

      class TestDense(module.Module):
        # [CLASS DOC]
        # HELPER: local test fixture; a module holding a weight variable (1.0) and a bias
        # variable (0.0) that computes weight times input plus bias when called.
        def __init__(self):
          # [FUNCTION DOC]
          # Constructor. Creates the weight variable (initial value 1.0) and bias variable
          # (initial value 0.0). Returns nothing.
          self.w = variables.Variable(1.0)
          self.b = variables.Variable(0.0)

        def __call__(self, inputs):
          # [FUNCTION DOC]
          # Input: a tensor. Returns the weight times the input plus the bias.
          return self.w * inputs + self.b

      inner.append(TestDense())
      inner[0](array_ops.ones([2, 3]))
      self.assertLen(outer.variables, 2)
      self.assertIsInstance(
          outer.variables[0],
          resource_variable_ops.ResourceVariable)

  def testNonLayerVariables(self):
    # [FUNCTION DOC]
    # Scenario: a List holding one resource variable (no layers). Asserts it is trainable, has no
    # layers, and lists the variable as variable, trainable weight and not as non-trainable. After
    # setting the trainable flag to false the variable moves to non-trainable and the trainable
    # list is empty. After turning it back on and appending a non-trainable variable, asserts
    # weights holds both, trainable weights only the first and non-trainable weights only the second.
    v = resource_variable_ops.ResourceVariable([1.])
    l = data_structures.List([v])
    self.assertTrue(l.trainable)
    self.assertEqual([], l.layers)
    self.assertEqual([v], l.variables)
    self.assertEqual([v], l.trainable_weights)
    self.assertEqual([], l.non_trainable_variables)
    l.trainable = False
    self.assertEqual([v], l.variables)
    self.assertEqual([], l.trainable_variables)
    self.assertEqual([v], l.non_trainable_variables)
    l.trainable = True
    v2 = resource_variable_ops.ResourceVariable(1., trainable=False)
    l.append(v2)
    self.assertEqual([v, v2], l.weights)
    self.assertEqual([v], l.trainable_weights)
    self.assertEqual([v2], l.non_trainable_weights)

  def testCopy(self):
    # [FUNCTION DOC]
    # Scenario: copies a List of two variables and appends a third to the copy. Asserts the
    # original still has two elements and the copy has three, i.e. the copy is independent.
    v1 = resource_variable_ops.ResourceVariable(1.)
    v2 = resource_variable_ops.ResourceVariable(1.)
    v3 = resource_variable_ops.ResourceVariable(1.)

    l1 = data_structures.List([v1, v2])
    l2 = l1.copy()
    l2.append(v3)
    self.assertEqual(list(l1), [v1, v2])
    self.assertEqual(list(l2), [v1, v2, v3])

  def testSlicing(self):
    # [FUNCTION DOC]
    # Scenario: a List of four variables is sliced from the second element, from the second to
    # the second-last, and up to the second-last. Asserts each slice equals the matching plain list.
    v1 = resource_variable_ops.ResourceVariable(1.)
    v2 = resource_variable_ops.ResourceVariable(1.)
    v3 = resource_variable_ops.ResourceVariable(1.)
    v4 = resource_variable_ops.ResourceVariable(1.)

    l = data_structures.List([v1, v2, v3, v4])
    self.assertEqual(l[1:], [v2, v3, v4])
    self.assertEqual(l[1:-1], [v2, v3])
    self.assertEqual(l[:-1], [v1, v2, v3])

  def testHash(self):
    # [FUNCTION DOC]
    # Scenario: two empty Lists are put in a set. Asserts the set has two members (identity
    # hashing, not value hashing) and that a third new empty List is not a member.
    has_sequences = {data_structures.List(), data_structures.List()}
    self.assertEqual(2, len(has_sequences))
    self.assertNotIn(data_structures.List(), has_sequences)

  def testIMul_zero(self):
    # [FUNCTION DOC]
    # Scenario: in-place multiplication of an empty List by zero. Asserts a value error whose
    # message says the List only supports append.
    l = data_structures.List([])
    with self.assertRaisesRegex(ValueError, "List only supports append"):
      l *= 0

  def testIMul(self):
    # [FUNCTION DOC]
    # Scenario: a List with one variable is multiplied in place by two. Asserts it then holds the
    # same variable twice.
    v = resource_variable_ops.ResourceVariable(1.)
    l = data_structures.List([v])
    l *= 2
    self.assertEqual(list(l), [v] * 2)

  def testMul(self):
    # [FUNCTION DOC]
    # Scenario: a List of three entries (the same variable) is multiplied by two. Asserts the
    # result, as a list, has the three entries repeated twice.
    v = resource_variable_ops.ResourceVariable(1.)
    l = data_structures.List([v, v, v])
    self.assertEqual(list(l * 2), [v, v, v] * 2)

  def testRMul(self):
    # [FUNCTION DOC]
    # Scenario: two multiplied by a List of three entries (reflected multiplication). Asserts the
    # result, as a list, has the three entries repeated twice.
    v = resource_variable_ops.ResourceVariable(1.)
    l = data_structures.List([v, v, v])
    self.assertEqual(list(2 * l), [v, v, v] * 2)


class ListWrapperTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the list wrapper: coverage of the built-in list API, pickling, structure
  # compatibility, comparison semantics, mutation tracking and the refusal to save after
  # unsupported mutations. Class attribute IGNORED lists special method names exempt from the
  # override check.

  IGNORED = ("__new__", "__init__", "__subclasshook__", "__getattribute__")

  def test_overrides_all_list_methods(self):
    # [FUNCTION DOC]
    # Scenario: inspects every callable attribute of the built-in list type (skipping the ignored
    # names and methods inherited unchanged from the base object type). Collects those for which
    # the wrapper's attribute is identical to the built-in one. Fails the test, naming them, if any
    # method is not overridden, guarding against list methods that could bypass mutation tracking.
    not_overridden = []

    for name in dir(list):
      if name in ListWrapperTest.IGNORED:
        continue

      list_method = getattr(list, name)

      if not callable(list_method):
        continue

      object_method = getattr(object, name, None)
      if object_method is not None and object_method == list_method:
        # Skip methods that aren't overridden from object.
        continue

      if list_method == getattr(data_structures.ListWrapper, name):
        not_overridden.append(name)

    if not_overridden:
      self.fail("ListWrapper does not override %s" % (not_overridden))

  def testPickle(self):
    # [FUNCTION DOC]
    # Scenario: pickles a list wrapper of [1, 2], deletes the original, unpickles. Asserts the
    # result equals [1, 2].
    original = data_structures.ListWrapper([1, 2])
    serialized = pickle.dumps(original)
    del original
    deserialized = pickle.loads(serialized)
    self.assertEqual([1, 2], deserialized)

  def testSameStructure(self):
    # [FUNCTION DOC]
    # Scenario: asserts via the nest utility that a plain list and a wrapper around a copy of it
    # have the same structure (no error raised).
    l = [1]
    nest.assert_same_structure(l, data_structures.ListWrapper(copy.copy(l)))

  def testMutateWithoutTrackableComponents(self):
    # [FUNCTION DOC]
    # Scenario: a module gets a list attribute of plain numbers and a value is inserted at the
    # front. Asserts the list equals the three numbers in order and that its trackable children
    # mapping is empty (no error because nothing trackable was mutated).
    m = module.Module()
    m.l = [1, 2]
    m.l.insert(0, 0)
    self.assertEqual(m.l, [0, 1, 2])
    self.assertEqual(m.l._trackable_children(), {})

  def testFunctionCaching(self):
    # [FUNCTION DOC]
    # Scenario: a compiled function that adds one to the first list element is traced once with a
    # plain list of a tensor and once with a list wrapper of a tensor. Asserts both traces are the
    # same concrete function, i.e. the wrapper maps to the same structure signature as a list.
    @def_function.function
    def f(list_input):
      # [FUNCTION DOC]
      # HELPER: function under trace. Input: a list of tensors. Returns the first element plus one.
      return list_input[0] + constant_op.constant(1.)

    first_trace = f.get_concrete_function([constant_op.constant(2.)])
    second_trace = f.get_concrete_function(
        data_structures.ListWrapper([constant_op.constant(3.)]))
    self.assertIs(first_trace, second_trace)

  def testListWrapperBasic(self):
    # [FUNCTION DOC]
    # Scenario: with two distinct trackable objects, compares plain lists and list wrappers in
    # every mixed combination. Asserts equality, inequality, less-than, less-or-equal, greater and
    # greater-or-equal behave like built-in lists; that a wrapper equals a plain list and a List
    # converts to the same list; that concatenation works in both orders; that a wrapper is an
    # instance of the built-in list type; and that adding a tensor shape to a wrapper yields the
    # expected shape list.
    # ListWrapper, unlike List, compares like the built-in list type (since it
    # is used to automatically replace lists).
    a = autotrackable.AutoTrackable()
    b = autotrackable.AutoTrackable()
    self.assertEqual([a, a],
                     [a, a])
    self.assertEqual(data_structures.ListWrapper([a, a]),
                     data_structures.ListWrapper([a, a]))
    self.assertEqual([a, a],
                     data_structures.ListWrapper([a, a]))
    self.assertEqual(data_structures.ListWrapper([a, a]),
                     [a, a])
    self.assertNotEqual([a, a],
                        [b, a])
    self.assertNotEqual(data_structures.ListWrapper([a, a]),
                        data_structures.ListWrapper([b, a]))
    self.assertNotEqual([a, a],
                        data_structures.ListWrapper([b, a]))
    self.assertLess([a], [a, b])
    self.assertLess(data_structures.ListWrapper([a]),
                    data_structures.ListWrapper([a, b]))
    self.assertLessEqual([a], [a, b])
    self.assertLessEqual(data_structures.ListWrapper([a]),
                         data_structures.ListWrapper([a, b]))
    self.assertGreater([a, b], [a])
    self.assertGreater(data_structures.ListWrapper([a, b]),
                       data_structures.ListWrapper([a]))
    self.assertGreaterEqual([a, b], [a])
    self.assertGreaterEqual(data_structures.ListWrapper([a, b]),
                            data_structures.ListWrapper([a]))
    self.assertEqual([a], data_structures.ListWrapper([a]))
    self.assertEqual([a], list(data_structures.List([a])))
    self.assertEqual([a, a], data_structures.ListWrapper([a]) + [a])
    self.assertEqual([a, a], [a] + data_structures.ListWrapper([a]))
    self.assertIsInstance(data_structures.ListWrapper([a]), list)
    self.assertEqual(
        tensor_shape.TensorShape([None, 2]).as_list(),
        (data_structures.ListWrapper([None])
         + tensor_shape.TensorShape([2])).as_list())

  def testAcceptsNonTrackableContent(self):
    # [FUNCTION DOC]
    # Scenario: a wrapper is built from plain numbers. Asserts construction succeeds and the
    # wrapper equals that list.
    l = data_structures.ListWrapper([1, 2, 3])
    self.assertEqual(l, [1, 2, 3])

  def testWrapperChangesList(self):
    # [FUNCTION DOC]
    # Scenario: appends a value through the wrapper. Asserts the original wrapped list sees the
    # change (they share storage).
    l = []
    l_wrapper = data_structures.ListWrapper(l)
    l_wrapper.append(1)
    self.assertEqual([1], l)

  def testListChangesWrapper(self):
    # [FUNCTION DOC]
    # Scenario: appends a value to the original list behind the wrapper. Asserts the wrapper
    # shows the new value (they share storage).
    l = []
    l_wrapper = data_structures.ListWrapper(l)
    l.append(1)
    self.assertEqual([1], l_wrapper)

  def testNotHashable(self):
    # [FUNCTION DOC]
    # Scenario: hashes an empty list wrapper. Asserts a type error is raised.
    with self.assertRaises(TypeError):
      hash(data_structures.ListWrapper())  # pylint: disable=no-value-for-parameter

  def testDelItem(self):
    # [FUNCTION DOC]
    # Scenario: deletes the first element of a wrapper that contains a nested list. Asserts the
    # remaining content and that saving then fails with an error mentioning item deletion.
    l = data_structures.ListWrapper([1, 2, 3, [4]])
    del l[0]
    self.assertEqual(l, [2, 3, [4]])
    self.assertUnableToSave(l, "Unable to save .*__delitem__")

  def testDelSlice(self):
    # [FUNCTION DOC]
    # Scenario: deletes a one-element slice from a wrapper containing a nested list. Asserts the
    # remaining content and that saving fails with an error mentioning slice deletion.
    l = data_structures.ListWrapper([1, 2, 3, [4]])
    del l[2:3]
    self.assertEqual(l, [1, 2, [4]])
    self.assertUnableToSave(l, "Unable to save .*__delslice__")

  def testSetSlice_canSaveForNonTrackableItems(self):
    # [FUNCTION DOC]
    # Scenario: replaces the whole content of a wrapper of plain numbers using a full slice.
    # Asserts the new content and that listing trackable children works and returns none.
    l = data_structures.ListWrapper([1, 2, 3, 4])
    l[:] = 2, 8, 9, 0
    self.assertEqual(l, [2, 8, 9, 0])
    l._maybe_initialize_trackable()  # pylint: disable=protected-access
    self.assertEqual(len(l._trackable_children()), 0)  # pylint: disable=protected-access

  def testSetSlice_cannotSaveIfTrackableModified(self):
    # [FUNCTION DOC]
    # Scenario: a wrapper holding numbers and two variables has a full slice replaced so that a
    # variable position now holds a number. Asserts the new content and that saving fails with an
    # error mentioning slice assignment.
    v1 = resource_variable_ops.ResourceVariable(1.)
    v2 = resource_variable_ops.ResourceVariable(1.)
    l = data_structures.ListWrapper([1, 2, v1, v2])
    l[:] = 2, 8, 9, v2
    self.assertEqual(l, [2, 8, 9, v2])
    self.assertUnableToSave(l, "Unable to save .*__setslice__")

  def testSetSlice_truncate(self):
    # [FUNCTION DOC]
    # Scenario: assigns an empty list to a full slice of a wrapper of numbers. Asserts the wrapper
    # becomes empty.
    l = data_structures.ListWrapper([1, 2, 3, 4])
    l[:] = []
    self.assertEqual(l, [])

  def testSetSlice_extend(self):
    # [FUNCTION DOC]
    # Scenario: assigns four numbers to the slice starting at position two of a four-number
    # wrapper. Asserts the first two numbers are kept and the rest replaced by the new four.
    l = data_structures.ListWrapper([1, 2, 3, 4])
    l[2:] = 1, 2, 3, 4
    self.assertEqual(l, [1, 2, 1, 2, 3, 4])

  def testIMulNegative(self):
    # [FUNCTION DOC]
    # Scenario: in-place multiplication of a wrapper (containing a nested list) by minus one.
    # Asserts the content equals the plain list multiplied by minus one (empty) and that saving
    # then fails.
    l = data_structures.ListWrapper([1, 2, 3, [4]])
    l *= -1
    self.assertEqual(l, [1, 2, 3, [4]] * -1)
    self.assertUnableToSave(l, "Unable to save")

  def testIMulPositive(self):
    # [FUNCTION DOC]
    # Scenario: a wrapper of numbers and one variable is saved in a checkpoint, the variable is
    # changed to 5, then the wrapper is multiplied in place by two. Asserts the content repeats,
    # the trackable children now name the variable at positions four and nine, and restoring the
    # checkpoint brings the variable back to 1.
    v = variables.Variable(1.)
    l = data_structures.ListWrapper([1, 2, 3, 4, v])
    self.assertDictEqual({"4": v}, l._trackable_children())
    root = util.Checkpoint(l=l)
    prefix = os.path.join(self.get_temp_dir(), "ckpt")
    path = root.save(prefix)
    v.assign(5.)
    l *= 2
    self.assertEqual(l, [1, 2, 3, 4, v, 1, 2, 3, 4, v])
    self.assertDictEqual({"4": v, "9": v}, l._trackable_children())
    root.restore(path)
    self.assertAllClose(1., v.numpy())

  def testSort(self):
    # [FUNCTION DOC]
    # Scenario: sorts a wrapper of already ordered nested lists (a no-op on content). Asserts the
    # content is unchanged yet saving still fails with an error mentioning sort, intentionally,
    # so behavior does not depend on the data.
    l = data_structures.ListWrapper([[1], [2], [3], [4]])
    l.sort()
    self.assertAllEqual(l, [[1], [2], [3], [4]])
    # Regardless of being a no-op for the input list, we still refuse to save.
    # This is intentional since otherwise we would end up with a hard to debug
    # case for users (e.g. sometimes sort on a ListWrapper is trackable and
    # other times it is not).
    self.assertUnableToSave(l, "Unable to save .*sort")

  def assertUnableToSave(self, l, msg):
    # [FUNCTION DOC]
    # HELPER: custom assertion shared by the mutation tests of this class.
    # Inputs: l (a list wrapper) and msg (regular expression). Initializes trackable state on the
    # wrapper, then asserts that listing its trackable children raises a value error whose message
    # matches msg. Returns the children if no error was raised (which then fails the assertion).
    l._maybe_initialize_trackable()  # pylint: disable=protected-access
    with self.assertRaisesRegex(ValueError, msg):
      return l._trackable_children()  # pylint: disable=protected-access


class MappingTests(test.TestCase):
  # [CLASS DOC]
  # Test case for the append-only Mapping container and for the automatic dict wrapper (and some
  # list wrapper cases): overwrite rules, key types, hashing, shallow and deep copies of wrappers
  # and of trackable objects holding them, pickling, construction from a sequence, concatenation
  # order, nest structure and traced function caching.

  def testJSONSerialization(self):
    # [FUNCTION DOC]
    # Scenario: a trackable object gets a plain dict attribute (becoming a dict wrapper). Dumps
    # the wrapper to JSON with the library's custom type hook. Passes if no error is raised.
    obj = autotrackable.AutoTrackable()
    obj.d = {"a": 2}
    json.dumps(obj.d, default=serialization.get_json_type)

  def testNoOverwrite(self):
    # [FUNCTION DOC]
    # Scenario: a Mapping gets key "a" bound to a List. Asserts: assigning another List to the same
    # key raises a value error and the original stays; deleting the key raises an attribute error;
    # update with a new key succeeds, but update with an existing key and a different object
    # raises a value error.
    mapping = data_structures.Mapping()
    original = data_structures.List()
    mapping["a"] = original
    with self.assertRaises(ValueError):
      mapping["a"] = data_structures.List()
    self.assertIs(original, mapping["a"])
    with self.assertRaises(AttributeError):
      del mapping["a"]  # pylint: disable=unsupported-delete-operation
    mapping.update(b=data_structures.Mapping())
    with self.assertRaises(ValueError):
      mapping.update({"b": data_structures.Mapping()})

  def testNonStringKeys(self):
    # [FUNCTION DOC]
    # Scenario: assigns a List under the integer key one in a Mapping. Asserts a type error.
    mapping = data_structures.Mapping()
    with self.assertRaises(TypeError):
      mapping[1] = data_structures.List()

  def testHashing(self):
    # [FUNCTION DOC]
    # Scenario: two empty Mappings are put in a set. Asserts two members and that a new empty
    # Mapping is not a member (identity hashing). Then, for a dict wrapper, asserts value-based
    # equality and inequality against plain dicts, and that putting it in a set raises a type error
    # mentioning unhashable.
    has_mappings = set([data_structures.Mapping(),
                        data_structures.Mapping()])
    self.assertEqual(2, len(has_mappings))
    self.assertNotIn(data_structures.Mapping(), has_mappings)
    # In contrast to Mapping, dict wrappers are not hashable
    a = autotrackable.AutoTrackable()
    a.d = {}
    self.assertEqual({}, a.d)
    self.assertFalse({} != a.d)  # pylint: disable=g-explicit-bool-comparison
    self.assertNotEqual({1: 2}, a.d)
    with self.assertRaisesRegex(TypeError, "unhashable"):
      set([a.d])

  def testListShallowCopy(self):
    # [FUNCTION DOC]
    # Scenario: a list wrapper (list of a list) is shallow copied. Asserts equal content, a
    # different wrapper object, and the same inner list object. After the original list is changed
    # behind the wrapper's back, asserts that listing objects raises a value error for both the
    # wrapper and a fresh shallow copy (the dirty state is inherited).
    root = autotrackable.AutoTrackable()
    orig_list = [[1.]]
    root.a = orig_list
    copied = copy.copy(root.a)
    self.assertAllEqual([[1.]], copied)
    self.assertIsNot(root.a, copied)
    self.assertIs(root.a[0], copied[0])

    # Dirtiness should be inherited
    util.list_objects(root.a)
    orig_list.append(1.)
    with self.assertRaises(ValueError):
      util.list_objects(root.a)
    with self.assertRaises(ValueError):
      util.list_objects(copy.copy(root.a))

  def testListDeepCopy(self):
    # [FUNCTION DOC]
    # Scenario: a list wrapper (list of a list) is deep copied. Asserts equal content, a different
    # wrapper and a different inner list object. After the original list is changed behind the
    # wrapper's back, asserts listing objects raises a value error for the wrapper and for a fresh
    # deep copy (the dirty state is inherited).
    root = autotrackable.AutoTrackable()
    orig_list = [[1.]]
    root.a = orig_list
    copied = copy.deepcopy(root.a)
    self.assertAllEqual([[1.]], copied)
    self.assertIsNot(root.a, copied)
    self.assertIsNot(root.a[0], copied[0])

    # Dirtiness should be inherited
    util.list_objects(root.a)
    orig_list.append(1.)
    with self.assertRaises(ValueError):
      util.list_objects(root.a)
    with self.assertRaises(ValueError):
      util.list_objects(copy.deepcopy(root.a))

  def testDictShallowCopy(self):
    # [FUNCTION DOC]
    # Scenario: a dict wrapper is shallow copied both with the copy module and its copy method.
    # Asserts equal content, a different object and the same inner object each time. After the
    # original dict is changed behind the wrapper's back, asserts listing objects raises a value
    # error for the wrapper and a fresh shallow copy (dirty state is inherited).
    root = autotrackable.AutoTrackable()
    orig_dict = {"a": [1.]}
    root.a = orig_dict
    copied = copy.copy(root.a)
    self.assertAllEqual([1.], copied["a"])
    self.assertIsNot(root.a, copied)
    self.assertIs(root.a["a"], copied["a"])

    copied = root.a.copy()
    self.assertAllEqual([1.], copied["a"])
    self.assertIsNot(root.a, copied)
    self.assertIs(root.a["a"], copied["a"])

    # Dirtiness should be inherited
    util.list_objects(root.a)
    orig_dict["b"] = []
    with self.assertRaises(ValueError):
      util.list_objects(root.a)
    with self.assertRaises(ValueError):
      util.list_objects(copy.copy(root.a))

  def testDictDeepCopy(self):
    # [FUNCTION DOC]
    # Scenario: a dict wrapper is deep copied. Asserts equal content, a different object and a
    # different inner object. After the original dict is changed behind the wrapper's back, asserts
    # listing objects raises a value error for the wrapper and a fresh deep copy.
    root = autotrackable.AutoTrackable()
    orig_dict = {"a": [1.]}
    root.a = orig_dict
    copied = copy.deepcopy(root.a)
    self.assertAllEqual([1.], copied["a"])
    self.assertIsNot(root.a, copied)
    self.assertIsNot(root.a["a"], copied["a"])

    # Dirtiness should be inherited
    util.list_objects(root.a)
    orig_dict["b"] = []
    with self.assertRaises(ValueError):
      util.list_objects(root.a)
    with self.assertRaises(ValueError):
      util.list_objects(copy.deepcopy(root.a))

  def testShallowCopyTrackable(self):
    # [FUNCTION DOC]
    # Scenario: a trackable object with a list attribute and a dict attribute holding another
    # trackable is shallow copied. Asserts the shared inner trackable, a different outer object,
    # equal list content, and that the copy's list wrapper, dict wrapper and inner trackable are
    # all among the copy's listed dependencies.
    original = autotrackable.AutoTrackable()
    original_sub = autotrackable.AutoTrackable()
    original.a = [[1.]]
    original.b = {"a": original_sub}
    shallow_copied = copy.copy(original)
    self.assertIs(original_sub, shallow_copied.b["a"])
    self.assertIsNot(original, shallow_copied)
    self.assertEqual([[1.]], shallow_copied.a)
    shallow_deps = util.list_objects(shallow_copied)
    self.assertIn(shallow_copied.a, shallow_deps)
    self.assertIn(shallow_copied.b, shallow_deps)
    self.assertIn(shallow_copied.b["a"], shallow_deps)

  def testDeepCopyTrackable(self):
    # [FUNCTION DOC]
    # Scenario: the same object graph is deep copied. Asserts the dict attribute stays a dict, the
    # outer and inner trackables are new objects of the right type, list content is equal, the copy's
    # list wrapper, dict wrapper and inner object are among its dependencies, and the original inner
    # object is not.
    original = autotrackable.AutoTrackable()
    original_sub = autotrackable.AutoTrackable()
    original.a = [[1.]]
    original.b = {"a": original_sub}
    self.assertIsInstance(original.b, dict)
    deep_copied = copy.deepcopy(original)
    self.assertIsInstance(deep_copied.b, dict)
    self.assertIsNot(original, deep_copied)
    self.assertIsNot(original_sub, deep_copied.b["a"])
    self.assertEqual([[1.]], deep_copied.a)
    self.assertIsInstance(deep_copied.b["a"], autotrackable.AutoTrackable)
    deps = util.list_objects(deep_copied)
    self.assertIn(deep_copied.a, deps)
    self.assertIn(deep_copied.b, deps)
    self.assertIn(deep_copied.b["a"], deps)
    self.assertNotIn(original_sub, deps)

  def testConstructableFromSequence(self):
    # [FUNCTION DOC]
    # Scenario: a dict wrapper is built from a list of key-value pairs. Asserts it is a dict
    # instance and equals the corresponding dictionary.
    result = data_structures._DictWrapper([(1, 2), (3, 4)])
    self.assertIsInstance(result, dict)
    self.assertEqual({1: 2, 3: 4}, result)

  def testPickle(self):
    # [FUNCTION DOC]
    # Scenario: pickles a dict wrapper with two entries, deletes the original, unpickles. Asserts
    # the result equals the same dictionary.
    original = data_structures._DictWrapper(dict(a=1, b=2))
    serialized = pickle.dumps(original)
    del original
    deserialized = pickle.loads(serialized)
    self.assertEqual(dict(a=1, b=2), deserialized)

  def testListAddOrder(self):
    # [FUNCTION DOC]
    # Scenario: concatenates list wrappers with each other and with plain lists on either side.
    # Asserts each result equals the plain list of the numbers in order (left operand first).
    # Note: lives in the mapping test class although it only exercises list wrappers.
    self.assertEqual([1., 2.],
                     data_structures.ListWrapper([1.])
                     + data_structures.ListWrapper([2.]))
    self.assertEqual([1., 2.],
                     data_structures.ListWrapper([1.])
                     + [2.])
    self.assertEqual([1., 2.],
                     [1.]
                     + data_structures.ListWrapper([2.]))

  def testSameStructure(self):
    # [FUNCTION DOC]
    # Scenario: asserts via the nest utility that a plain dict and a dict wrapper around a copy of
    # it have the same structure (no error raised).
    d = {1: "a"}
    nest.assert_same_structure(d, data_structures._DictWrapper(d.copy()))

  def testFunctionCaching(self):
    # [FUNCTION DOC]
    # Scenario: a compiled function adding one to the value under key "x" is traced once with a
    # plain dict of a tensor and once with a dict wrapper of a tensor. Asserts both traces are the
    # same concrete function.
    @def_function.function
    def f(dict_input):
      # [FUNCTION DOC]
      # HELPER: function under trace. Input: a dict with key "x" holding a tensor. Returns that
      # tensor plus one.
      return dict_input["x"] + constant_op.constant(1.)

    first_trace = f.get_concrete_function({"x": constant_op.constant(2.)})
    second_trace = f.get_concrete_function(
        data_structures._DictWrapper({"x": constant_op.constant(3.)}))
    self.assertIs(first_trace, second_trace)


class TupleTests(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the tuple wrapper and namedtuple handling: JSON, variables, copies, slicing,
  # hashing, in-place operators, pickling, namedtuple subclasses, structure compatibility with the
  # nest utility, traced function caching and dataset mapping. Also parameterized-capable.

  def testJSONSerialization(self):
    # [FUNCTION DOC]
    # Scenario: a trackable object gets a tuple attribute holding a number. Dumps it to JSON with
    # the library's custom type hook. Passes if no error is raised.
    obj = autotrackable.AutoTrackable()
    obj.l = (1,)
    json.dumps(obj.l, default=serialization.get_json_type)

  def testNonLayerVariables(self):
    # [FUNCTION DOC]
    # Scenario: a tuple wrapper holding one resource variable. Asserts there are no layers, the
    # variable is listed as variable and trainable weight, and non-trainable variables are empty.
    v = resource_variable_ops.ResourceVariable([1.])
    l = data_structures._TupleWrapper((v,))
    self.assertEqual([], l.layers)
    self.assertEqual([v], l.variables)
    self.assertEqual([v], l.trainable_weights)
    self.assertEqual([], l.non_trainable_variables)

  def testCopy(self):
    # [FUNCTION DOC]
    # Scenario: a tuple wrapper of two variables is shallow and deep copied. Asserts both equal
    # the tuple, the shallow copy shares the first element while the deep copy does not, and the
    # copy has no append method (attribute error).
    v1 = resource_variable_ops.ResourceVariable(1.)
    v2 = resource_variable_ops.ResourceVariable(1.)

    l1 = data_structures._TupleWrapper((v1, v2))
    l2 = copy.copy(l1)
    self.assertEqual(l1, (v1, v2))
    self.assertEqual(l2, (v1, v2))
    self.assertIs(l1[0], l2[0])
    l2_deep = copy.deepcopy(l1)
    self.assertIsNot(l1[0], l2_deep[0])
    with self.assertRaises(AttributeError):
      l2.append(v1)

  def testSlicing(self):
    # [FUNCTION DOC]
    # Scenario: a tuple wrapper of four variables is sliced from the second element, from the
    # second to the second-last, and up to the second-last. Asserts each slice equals the matching
    # plain tuple.
    v1 = resource_variable_ops.ResourceVariable(1.)
    v2 = resource_variable_ops.ResourceVariable(1.)
    v3 = resource_variable_ops.ResourceVariable(1.)
    v4 = resource_variable_ops.ResourceVariable(1.)

    l = data_structures._TupleWrapper((v1, v2, v3, v4))
    self.assertEqual(l[1:], (v2, v3, v4))
    self.assertEqual(l[1:-1], (v2, v3))
    self.assertEqual(l[:-1], (v1, v2, v3))

  def testHash(self):
    # [FUNCTION DOC]
    # Scenario: two empty tuple wrappers are put in a set. Asserts the set has one member (value
    # hashing, unlike the List) and that a new empty tuple wrapper is a member.
    has_sequences = set([data_structures._TupleWrapper(),
                         data_structures._TupleWrapper()])
    self.assertLen(has_sequences, 1)
    self.assertIn(data_structures._TupleWrapper(), has_sequences)

  def testIMul_zero(self):
    # [FUNCTION DOC]
    # Scenario: in-place multiplication of a one-element tuple wrapper by zero. Asserts the result
    # equals the empty tuple (no error, unlike List).
    l = data_structures._TupleWrapper((1,))
    l *= 0
    self.assertEqual((), l)

  def testIMul(self):
    # [FUNCTION DOC]
    # Scenario: in-place multiplication by two of a tuple wrapper holding a variable. Asserts the
    # new left-hand value is the repeated tuple while the original wrapper object is unchanged.
    # Note: tuple behavior differs from list behavior. Lists are mutated by
    # imul/iadd, tuples assign a new object to the left hand side of the
    # expression.
    v = resource_variable_ops.ResourceVariable(1.)
    l = data_structures._TupleWrapper((v,))
    original = l
    l *= 2
    self.assertEqual(l, (v,) * 2)
    self.assertNotEqual(original, (v,) * 2)

  def testIAdd(self):
    # [FUNCTION DOC]
    # Scenario: in-place addition of a one-element tuple to a tuple wrapper holding a variable.
    # Asserts the new left-hand value is the concatenated tuple and the original wrapper is
    # unchanged.
    v = resource_variable_ops.ResourceVariable(1.)
    l = data_structures._TupleWrapper((v,))
    original = l
    l += (1,)
    self.assertEqual(l, (v, 1))
    self.assertNotEqual(original, (v, 1))
    self.assertEqual(original, (v,))

  def testMul(self):
    # [FUNCTION DOC]
    # Scenario: a tuple wrapper of three entries multiplied by two. Asserts the result equals the
    # plain tuple repeated twice.
    v = resource_variable_ops.ResourceVariable(1.)
    l = data_structures._TupleWrapper((v, v, v))
    self.assertEqual(l * 2, (v, v, v) * 2)

  def testRMul(self):
    # [FUNCTION DOC]
    # Scenario: two multiplied by a tuple wrapper of three entries. Asserts the result equals the
    # plain tuple repeated twice.
    v = resource_variable_ops.ResourceVariable(1.)
    l = data_structures._TupleWrapper((v, v, v))
    self.assertEqual(2 * l, (v, v, v) * 2)

  def testPickle(self):
    # [FUNCTION DOC]
    # Scenario: pickles a tuple wrapper of (1, 2), deletes the original, unpickles. Asserts the
    # result equals the tuple (1, 2).
    original = data_structures._TupleWrapper((1, 2))
    serialized = pickle.dumps(original)
    del original
    deserialized = pickle.loads(serialized)
    self.assertEqual((1, 2), deserialized)

  def testNamedTuple(self):
    # [FUNCTION DOC]
    # Scenario: a module gets a namedtuple attribute holding a variable and a number. Asserts the
    # variable is reachable by field name and by index, is among the grandchild dependencies
    # under the field name, and the plain field keeps its value.
    named = collections.namedtuple("Named", ("x", "y"))
    v = variables.Variable(2)
    nt = named(x=v, y=2)
    m = module.Module()
    m.nt = nt
    self.assertIs(v, m.nt.x)
    self.assertIs(v, m.nt[0])
    self.assertIs(
        v, m._trackable_children()["nt"]._trackable_children()["x"])
    self.assertEqual(2, m.nt.y)

  def testNamedTupleConflictingAttributes(self):
    # [FUNCTION DOC]
    # Scenario: a namedtuple has a field called weights, which clashes with a wrapper property.
    # Assigned to a module, asserts reading the field returns the tuple's value (3), showing the
    # wrapped tuple's attributes win over the wrapper's.
    named = collections.namedtuple("Named", ("x", "weights"))
    v = variables.Variable(2)
    nt = named(x=v, weights=3)
    m = module.Module()
    m.nt = nt
    self.assertEqual(3, m.nt.weights)

  def testNamedSubclassing(self):
    # [FUNCTION DOC]
    # Scenario: a namedtuple subclass whose constructor ignores the second argument and always
    # uses 3, with a property summing its fields, holds a variable and is assigned to a module.
    # Asserts the second field is 3, the first is the variable, the variable is among the wrapper's
    # dependencies, both the field name and index keys are present, and the sum evaluates to 5.
    named = collections.namedtuple("Named", ("x", "y"))
    v = variables.Variable(2)

    class NamedSubclass(named):
      # [CLASS DOC]
      # HELPER: local test fixture; a namedtuple subclass of fields x and y that forces y to 3 and
      # offers a summed property (x plus y).

      def __new__(cls, x, y):
        # [FUNCTION DOC]
        # Constructor hook. Inputs: class, x and y; y is ignored. Returns a new instance with x
        # and the fixed value 3.
        del y  # unused
        return super(NamedSubclass, cls).__new__(cls, x, 3)

      @property
      def summed(self):
        # [FUNCTION DOC]
        # Property returning the sum of the two fields.
        return self.x + self.y

    nt = NamedSubclass(x=v, y=2)
    m = module.Module()
    m.nt = nt
    self.assertEqual(3, m.nt.y)
    self.assertIs(v, m.nt.x)
    self.assertIn(v,
                  m._trackable_children()["nt"]._trackable_children().values())
    self.assertIn("x", m.nt._trackable_children())
    self.assertIn("0", m.nt._trackable_children())
    self.assertEqual(5, self.evaluate(m.nt.summed))

  def testUnnamedSubclassing(self):
    # [FUNCTION DOC]
    # Scenario: a plain tuple subclass with a summed property holds a variable and a number and is
    # assigned to a module. Asserts only the index key "0" is tracked (exactly one dependency), the
    # sum evaluates to 4, and the nest utility keeps the same structure when mapping over it.
    v = variables.Variable(2)

    class UnnamedSubclass(tuple):
      # [CLASS DOC]
      # HELPER: local test fixture; a tuple subclass offering a summed property of its first two
      # elements.

      @property
      def summed(self):
        # [FUNCTION DOC]
        # Property returning the sum of the first two elements.
        return self[0] + self[1]

    unt = UnnamedSubclass([v, 2])
    m = module.Module()
    m.unt = unt
    self.assertIn("0", m.unt._trackable_children())
    self.assertLen(m.unt._trackable_children(), 1)
    self.assertEqual(4, self.evaluate(m.unt.summed))
    nest.assert_same_structure(
        [m.unt], nest.map_structure(lambda x: x, [m.unt]))

  def testNamedtupleSubclassWithCustomNew(self):
    # [FUNCTION DOC]
    # Scenario: a namedtuple subclass whose constructor takes no arguments (so it cannot be rebuilt
    # from its fields) holds a list; a variable is appended to the list and the owner module is
    # saved in a checkpoint. Asserts saving raises a value error.
    class SubclassWithDifferentArgs(collections.namedtuple("A", ["x"])):
      # [CLASS DOC]
      # HELPER: local test fixture; a one-field namedtuple subclass with a no-argument constructor
      # that stores an empty list in the field.

      def __new__(cls):
        # [FUNCTION DOC]
        # Constructor hook taking no arguments; returns an instance whose single field is a new
        # empty list.
        return super(SubclassWithDifferentArgs, cls).__new__(cls, [])

    nt = SubclassWithDifferentArgs()
    m = module.Module()
    m.nt = nt
    m.nt.x.append(variables.Variable(1.))
    prefix = os.path.join(self.get_temp_dir(), "ckpt")
    ckpt = util.Checkpoint(m=m)
    with self.assertRaises(ValueError):
      ckpt.save(prefix)

  def testSameStructure(self):
    # [FUNCTION DOC]
    # Scenario: a plain tuple and a namedtuple are assigned to a module. Asserts via the nest
    # utility that each wrapped value has the same structure as the original (both directions for
    # the tuple), and that comparing the wrapped namedtuple to the wrapped tuple raises a type error.
    t = (variables.Variable(1.),)
    m = module.Module()
    m.t = t
    nest.assert_same_structure(t, m.t)
    nest.assert_same_structure(m.t, t)

    nt_type = collections.namedtuple("nt", ["x", "y"])
    nt = nt_type(x=1, y=2)
    m.nt = nt
    nest.assert_same_structure(m.nt, nt)
    with self.assertRaises(TypeError):  # pylint: disable=g-error-prone-assert-raises
      nest.assert_same_structure(m.nt, m.t)

  def testFlatten(self):
    # [FUNCTION DOC]
    # Scenario: nested tuple wrappers and a wrapped namedtuple go through nest utilities. Asserts
    # flattening yields the leaf values, flattening with paths matches the plain tuples, and packing
    # a sequence back as the wrapper's structure yields the plain tuple (a namedtuple keeps its
    # field access).
    t = data_structures._TupleWrapper((1, data_structures._TupleWrapper((2,))))
    self.assertEqual([1, 2], nest.flatten(t))
    self.assertEqual(
        nest.flatten_with_tuple_paths((1, (2,))),
        nest.flatten_with_tuple_paths(t))
    self.assertEqual((3, (4,)),
                     nest.pack_sequence_as(t, [3, 4]))
    nt_type = collections.namedtuple("nt", ["x", "y"])
    nt = nt_type(1., 2.)
    wrapped_nt = data_structures._TupleWrapper(nt)
    self.assertEqual(
        nest.flatten_with_tuple_paths(nt),
        nest.flatten_with_tuple_paths(wrapped_nt))
    self.assertEqual((3, 4,),
                     nest.pack_sequence_as(wrapped_nt, [3, 4]))
    self.assertEqual(3, nest.pack_sequence_as(wrapped_nt, [3, 4]).x)

  def testFunctionCaching(self):
    # [FUNCTION DOC]
    # Scenario: a compiled function adding one to the first tuple element is traced once with a
    # plain tuple of a tensor and once with a tuple wrapper of a tensor. Asserts both traces are the
    # same concrete function.
    @def_function.function
    def f(tuple_input):
      # [FUNCTION DOC]
      # HELPER: function under trace. Input: a tuple of tensors. Returns the first element plus one.
      return tuple_input[0] + constant_op.constant(1.)

    first_trace = f.get_concrete_function((constant_op.constant(2.),))
    second_trace = f.get_concrete_function(
        data_structures._TupleWrapper((constant_op.constant(3.),)))
    self.assertIs(first_trace, second_trace)

  def testPythonMapImpl(self):
    # [FUNCTION DOC]
    # Scenario: a nested tuple wrapper is mapped over with the nest up-to-structure mapper, adding
    # three to each leaf with type checking. Asserts the result is the expected nested plain tuple,
    # and that the shallow structure check against a flat two-slot structure passes.
    t = data_structures._TupleWrapper((1, data_structures._TupleWrapper((2,))))
    self.assertEqual(
        (4, (5,)),
        nest.map_structure_up_to((None, (None,)), lambda x: x + 3, t,
                                 check_types=True))
    nest.assert_shallow_structure((None, None), t)

  def testDatasetMap(self):
    # [FUNCTION DOC]
    # Scenario: a dataset of the numbers 1 to 3 is mapped to one-element tuple wrappers. Iterates
    # it and asserts each element evaluates to a one-element tuple of the matching number.
    dataset = dataset_ops.Dataset.from_tensor_slices(
        constant_op.constant([1, 2, 3]))
    dataset = dataset.map(lambda x: data_structures._TupleWrapper((x,)))
    for index, element in enumerate(dataset):
      self.assertEqual((index + 1,), self.evaluate(element))

  def testDatasetMapNamed(self):
    # [FUNCTION DOC]
    # Scenario: like the plain dataset test but each element is a wrapped one-field namedtuple.
    # Asserts each iterated element evaluates to a one-element tuple of the matching number.
    nt_type = collections.namedtuple("A", ["x"])
    dataset = dataset_ops.Dataset.from_tensor_slices(
        constant_op.constant([1, 2, 3]))
    dataset = dataset.map(lambda x: data_structures._TupleWrapper(nt_type(x)))
    for index, element in enumerate(dataset):
      self.assertEqual((index + 1,), self.evaluate(element))

  def testLoopAssignedModule(self):
    # [FUNCTION DOC]
    # Scenario: a module assigns a tuple containing itself to one of its attributes (a reference
    # loop). Asserts it has exactly one trackable child under that attribute name, that child is
    # the stored tuple wrapper, and trainable variables are an empty tuple.
    m = module.Module()
    m.s = (m,)
    self.assertLen(m._trackable_children(), 1)
    self.assertIn("s", m._trackable_children())
    self.assertIs(m.s, m._trackable_children()["s"])
    self.assertEqual((), m.trainable_variables)


if __name__ == "__main__":
  test.main()
