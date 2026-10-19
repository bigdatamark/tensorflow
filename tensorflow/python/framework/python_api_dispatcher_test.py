# [MODULE DOC]
# Tests for the native Python API dispatcher bindings (the private pywrap dispatcher module).
# Three test cases cover: type checkers (instance, union and list checkers, with their match
# results, costs, string forms and cache behavior, plus registering dispatchable types), the
# signature checker (which checks canonicalized argument tuples against per-position type
# checkers and orders checks by cost), and the dispatcher itself (registering targets with
# signature checkers and dispatching positional and keyword arguments, including the ambiguity
# error). Module-level constants alias the three match-type enum values. All tests run in both
# graph and eager modes. Running the file starts the googletest runner.
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
"""Tests for tensorflow.python.framework.python_api_dispatcher."""

import collections
import numpy as np

from tensorflow.python.framework import _pywrap_python_api_dispatcher as dispatch
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import googletest

# PyTypeChecker::MatchType enum values.
NO_MATCH = dispatch.MatchType.NO_MATCH
MATCH = dispatch.MatchType.MATCH
MATCH_DISPATCHABLE = dispatch.MatchType.MATCH_DISPATCHABLE


@test_util.run_all_in_graph_and_eager_modes
class PythonTypeCheckerTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the native type checker objects (instance, union and list checkers) and for
  # registering dispatchable types. Runs every test in graph and eager modes.

  def testInstanceChecker(self):
    # [FUNCTION DOC]
    # Scenario: build instance checkers for integers, tensors, ragged tensors, an integer-or-float
    # pair, and a small class hierarchy. Asserts the match result of each value (match, no match,
    # or match-dispatchable for ragged tensors), that cost is one per accepted type, that the
    # string form lists the types, and that subclass instances (including through multiple
    # inheritance) match a checker for their base class.
    t = constant_op.constant([1, 2, 3])
    rt = ragged_factory_ops.constant([[1, 2], [3, 4, 5]])

    with self.subTest('int checker'):
      int_checker = dispatch.MakeInstanceChecker(int)
      self.assertEqual(int_checker.Check(3), MATCH)
      self.assertEqual(int_checker.Check(3.0), NO_MATCH)
      self.assertEqual(int_checker.Check(t), NO_MATCH)
      self.assertEqual(int_checker.cost(), 1)
      self.assertEqual(repr(int_checker), '<PyTypeChecker int>')

    with self.subTest('tensor checker'):
      tensor_checker = dispatch.MakeInstanceChecker(tensor.Tensor)
      self.assertEqual(tensor_checker.Check(t), MATCH)
      self.assertEqual(tensor_checker.Check(3), NO_MATCH)
      self.assertEqual(tensor_checker.Check(3.0), NO_MATCH)
      self.assertEqual(tensor_checker.cost(), 1)
      self.assertEqual(repr(tensor_checker), '<PyTypeChecker Tensor>')

    with self.subTest('ragged checker'):
      ragged_checker = dispatch.MakeInstanceChecker(ragged_tensor.RaggedTensor)
      self.assertEqual(ragged_checker.Check(rt), MATCH_DISPATCHABLE)
      self.assertEqual(ragged_checker.Check(3), NO_MATCH)
      self.assertEqual(ragged_checker.Check(t), NO_MATCH)
      self.assertEqual(ragged_checker.cost(), 1)
      self.assertEqual(repr(ragged_checker), '<PyTypeChecker RaggedTensor>')

    with self.subTest('int or float checker'):
      int_checker = dispatch.MakeInstanceChecker(int, float)
      self.assertEqual(int_checker.Check(3), MATCH)
      self.assertEqual(int_checker.Check(3.0), MATCH)
      self.assertEqual(int_checker.Check(t), NO_MATCH)
      self.assertEqual(int_checker.cost(), 2)
      self.assertEqual(repr(int_checker), '<PyTypeChecker int, float>')

    with self.subTest('subclasses'):

      class A(object):
        # [CLASS DOC]
        # HELPER: empty base class for the subclass-matching sub-test.
        pass

      class B(A):
        # [CLASS DOC]
        # HELPER: empty subclass of the first helper class.
        pass

      class C(object):
        # [CLASS DOC]
        # HELPER: empty class unrelated to the first helper class.
        pass

      class D(C, B):
        # [CLASS DOC]
        # HELPER: empty class inheriting from both the unrelated class and the subclass, so it
        # is also a descendant of the base class through multiple inheritance.
        pass

      checker = dispatch.MakeInstanceChecker(A)
      self.assertEqual(checker.Check(A()), MATCH)
      self.assertEqual(checker.Check(B()), MATCH)
      self.assertEqual(checker.Check(C()), NO_MATCH)
      self.assertEqual(checker.Check(D()), MATCH)

  def testInstanceCheckerCache(self):
    # [FUNCTION DOC]
    # Scenario: a checker for the tuple type is called with a series of values. Asserts the
    # match result and the checker's internal cache size after each call: the cache gains an
    # entry for each previously unseen concrete type (misses) and does not grow when a value of an
    # already seen type is checked (hits), including for a named-tuple subclass.
    checker = dispatch.MakeInstanceChecker(tuple)
    MyTuple = collections.namedtuple('MyTuple', ['a', 'b'])  # Subclass of tuple

    self.assertEqual(checker.cache_size(), 0)
    self.assertEqual(checker.Check(5), NO_MATCH)
    self.assertEqual(checker.cache_size(), 1)  # cache miss
    self.assertEqual(checker.Check(12), NO_MATCH)
    self.assertEqual(checker.cache_size(), 1)  # cache hit
    self.assertEqual(checker.Check(1.3), NO_MATCH)
    self.assertEqual(checker.cache_size(), 2)  # cache miss
    self.assertEqual(checker.Check([1]), NO_MATCH)
    self.assertEqual(checker.cache_size(), 3)  # cache miss
    self.assertEqual(checker.Check((1,)), MATCH)
    self.assertEqual(checker.cache_size(), 4)  # cache miss
    self.assertEqual(checker.Check((1, 2, 3)), MATCH)
    self.assertEqual(checker.cache_size(), 4)  # cache hit
    self.assertEqual(checker.Check(MyTuple(1, 2)), MATCH)
    self.assertEqual(checker.cache_size(), 5)  # cache miss
    self.assertEqual(checker.Check(MyTuple(3, 4)), MATCH)
    self.assertEqual(checker.cache_size(), 5)  # cache miss
    self.assertEqual(checker.Check(()), MATCH)
    self.assertEqual(checker.cache_size(), 5)  # cache hit

  def testUnionChecker(self):
    # [FUNCTION DOC]
    # Scenario: union checkers built from instance checkers (integer, float, string, null,
    # tensor, ragged tensor). Asserts that a value matches if any member matches (ragged tensors
    # give match-dispatchable), that the cost is one more than the sum of the member costs, and
    # that the string form lists the member types.
    int_checker = dispatch.MakeInstanceChecker(int)
    float_checker = dispatch.MakeInstanceChecker(float)
    str_checker = dispatch.MakeInstanceChecker(str)
    none_checker = dispatch.MakeInstanceChecker(type(None))
    tensor_checker = dispatch.MakeInstanceChecker(tensor.Tensor)
    ragged_checker = dispatch.MakeInstanceChecker(ragged_tensor.RaggedTensor)

    t = constant_op.constant([1, 2, 3])
    rt = ragged_factory_ops.constant([[1, 2], [3, 4, 5]])

    with self.subTest('Union[int, float, str]'):
      checker = dispatch.MakeUnionChecker(
          [int_checker, float_checker, str_checker])
      self.assertEqual(checker.Check(3), MATCH)
      self.assertEqual(checker.Check(3.0), MATCH)
      self.assertEqual(checker.Check('x'), MATCH)
      self.assertEqual(checker.Check('x'), MATCH)
      self.assertEqual(checker.Check(None), NO_MATCH)
      self.assertEqual(checker.Check(t), NO_MATCH)
      self.assertEqual(checker.cost(), 4)
      self.assertEqual(repr(checker), '<PyTypeChecker Union[int, float, str]>')

    with self.subTest('Optional[int] (aka Union[int, None])'):
      checker = dispatch.MakeUnionChecker([int_checker, none_checker])
      self.assertEqual(checker.Check(3), MATCH)
      self.assertEqual(checker.Check(3.0), NO_MATCH)
      self.assertEqual(checker.Check(None), MATCH)
      self.assertEqual(checker.Check(t), NO_MATCH)
      self.assertEqual(checker.cost(), 3)
      self.assertEqual(repr(checker), '<PyTypeChecker Union[int, NoneType]>')

    with self.subTest('Union[Tensor, RaggedTensor]'):
      checker = dispatch.MakeUnionChecker([tensor_checker, ragged_checker])
      self.assertEqual(checker.Check(3), NO_MATCH)
      self.assertEqual(checker.Check(3.0), NO_MATCH)
      self.assertEqual(checker.Check(None), NO_MATCH)
      self.assertEqual(checker.Check(t), MATCH)
      self.assertEqual(checker.Check(rt), MATCH_DISPATCHABLE)
      self.assertEqual(checker.cost(), 3)
      self.assertEqual(
          repr(checker), '<PyTypeChecker Union[Tensor, RaggedTensor]>')

  def testListChecker(self):
    # [FUNCTION DOC]
    # Scenario: list checkers over integers, tensors, a union of tensor and ragged tensor, and a
    # union of integer and numpy integer. Uses sample values: lists, tuples, an empty list,
    # iterators, a dictionary and a numpy array. Asserts that lists and tuples whose elements all
    # match are accepted (empty ones too), that iterators, dictionaries and numpy arrays never
    # match, that ragged elements produce match-dispatchable, that cost is ten times the element
    # checker's cost, and the string forms.
    int_checker = dispatch.MakeInstanceChecker(int)
    tensor_checker = dispatch.MakeInstanceChecker(tensor.Tensor)
    ragged_checker = dispatch.MakeInstanceChecker(ragged_tensor.RaggedTensor)
    np_int_checker = dispatch.MakeInstanceChecker(np.integer)

    t = constant_op.constant([1, 2, 3])
    rt = ragged_factory_ops.constant([[1, 2], [3, 4, 5]])
    a = [1, 2, 3]
    b = ['a', 2, t]
    c = [t, t * 2, t - 2]
    d = [t, rt]
    e = []
    f = (1, 2, 3)
    g = (rt,)
    h = {1: 2, 3: 4}
    i = np.array([1, 2, 3])

    with self.subTest('List[int]'):
      checker = dispatch.MakeListChecker(int_checker)
      self.assertEqual(checker.Check(a), MATCH)
      self.assertEqual(checker.Check(b), NO_MATCH)
      self.assertEqual(checker.Check(c), NO_MATCH)
      self.assertEqual(checker.Check(d), NO_MATCH)
      self.assertEqual(checker.Check(e), MATCH)
      self.assertEqual(checker.Check(f), MATCH)
      self.assertEqual(checker.Check(iter(a)), NO_MATCH)
      self.assertEqual(checker.Check(iter(b)), NO_MATCH)
      self.assertEqual(checker.Check(reversed(e)), NO_MATCH)
      self.assertEqual(checker.Check(h), NO_MATCH)
      self.assertEqual(checker.Check(i), NO_MATCH)
      self.assertEqual(checker.cost(), 10)
      self.assertEqual(repr(checker), '<PyTypeChecker List[int]>')

    with self.subTest('List[Tensor]'):
      checker = dispatch.MakeListChecker(tensor_checker)
      self.assertEqual(checker.Check(a), NO_MATCH)
      self.assertEqual(checker.Check(b), NO_MATCH)
      self.assertEqual(checker.Check(c), MATCH)
      self.assertEqual(checker.Check(d), NO_MATCH)
      self.assertEqual(checker.Check(e), MATCH)
      self.assertEqual(checker.cost(), 10)
      self.assertEqual(repr(checker), '<PyTypeChecker List[Tensor]>')

    with self.subTest('List[Union[Tensor, RaggedTensor]]'):
      checker = dispatch.MakeListChecker(
          dispatch.MakeUnionChecker([tensor_checker, ragged_checker]))
      self.assertEqual(checker.Check(a), NO_MATCH)
      self.assertEqual(checker.Check(b), NO_MATCH)
      self.assertEqual(checker.Check(c), MATCH)
      self.assertEqual(checker.Check(d), MATCH_DISPATCHABLE)
      self.assertEqual(checker.Check(e), MATCH)
      self.assertEqual(checker.Check(f), NO_MATCH)
      self.assertEqual(checker.Check(g), MATCH_DISPATCHABLE)
      self.assertEqual(checker.cost(), 30)
      self.assertEqual(
          repr(checker), '<PyTypeChecker List[Union[Tensor, RaggedTensor]]>')

    with self.subTest('List[Union[int, np.integer]]'):
      # Note: np.integer is a subtype of int in *some* Python versions.
      checker = dispatch.MakeListChecker(
          dispatch.MakeUnionChecker([int_checker, np_int_checker]))
      self.assertEqual(checker.Check(a), MATCH)
      self.assertEqual(checker.Check(np.array(a)), NO_MATCH)
      self.assertEqual(checker.Check(np.array(a) * 1.5), NO_MATCH)

  def testRegisterDispatchableType(self):
    # [FUNCTION DOC]
    # Scenario: define a new class and register it as a dispatchable type through the decorator.
    # Asserts that an instance checker for that class then reports match-dispatchable for
    # an instance of it.

    @dispatch.register_dispatchable_type
    class A(object):
      # [CLASS DOC]
      # HELPER: empty class registered as a dispatchable type by the test.
      pass

    checker = dispatch.MakeInstanceChecker(A)
    self.assertEqual(checker.Check(A()), MATCH_DISPATCHABLE)

  def testRegisterDispatchableTypeError(self):
    # [FUNCTION DOC]
    # Scenario: invalid registrations. Asserts that registering a non-type value (the number
    # three) raises a value error saying a type object was expected, and that registering the
    # ragged tensor class again raises a value error saying the type was already registered.

    with self.assertRaisesRegex(ValueError, 'Expected a type object'):
      dispatch.register_dispatchable_type(3)
    with self.assertRaisesRegex(ValueError,
                                'Type .* has already been registered'):
      dispatch.register_dispatchable_type(ragged_tensor.RaggedTensor)


@test_util.run_all_in_graph_and_eager_modes
class PythonSignatureCheckerTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the native signature checker, which holds a list of (argument position, type
  # checker) pairs and decides whether a canonicalized argument tuple satisfies all of them.
  # Provides a shared helper that checks many argument tuples. Runs in graph and eager modes.

  def check_signatures(self, checker, canon_expected_pairs):
    # [FUNCTION DOC]
    # HELPER: shared assertion helper for the signature checker tests.
    # Inputs: checker, a signature checker; canon_expected_pairs, a list of pairs of a
    # canonicalized argument tuple and the expected boolean outcome. For each pair runs a sub-test
    # asserting that the checker's result for the tuple equals the expected outcome.
    # Returns nothing.
    for (canon_args, expected) in canon_expected_pairs:
      with self.subTest(f'{canon_args} -> {expected}'):
        self.assertEqual(checker.CheckCanonicalizedArgs(canon_args), expected)

  def testSimpleSignature(self):
    # [FUNCTION DOC]
    # Scenario: a signature checker requiring an integer at position zero and a ragged tensor at
    # position two. Asserts true only for tuples with matching values at those positions (extra
    # trailing and unconstrained values are fine; too short tuples, wrong types and lists fail),
    # and that the string form lists the constrained positions.
    int_checker = dispatch.MakeInstanceChecker(int)
    rt_checker = dispatch.MakeInstanceChecker(ragged_tensor.RaggedTensor)
    checker = dispatch.PySignatureChecker([(0, int_checker), (2, rt_checker)])
    rt = ragged_factory_ops.constant([[1, 2], [3]])

    self.check_signatures(checker, [
        ((1, 2, rt), True),
        ((1, 2, 3), False),
        ((1, 2), False), ((), False),
        ((5, 'x', rt, None), True),
        (([5], 'x', rt, None), False),
        ((5, 'x', [rt], None), False),
    ])  # pyformat: disable

    self.assertEqual(
        repr(checker), '<PySignatureChecker args[0]:int, args[2]:RaggedTensor>')

  def testUnion(self):
    # [FUNCTION DOC]
    # Scenario: a signature checker requiring a ragged-tensor-or-tensor union at positions zero
    # and one. Asserts that argument tuples are accepted only when both positions hold either type
    # and at least one of them is a ragged tensor (two plain tensors do not count, since nothing
    # would need dispatching); a list is rejected; extra arguments are ignored. Also checks the
    # string form.
    rt_checker = dispatch.MakeInstanceChecker(ragged_tensor.RaggedTensor)
    tensor_checker = dispatch.MakeInstanceChecker(tensor.Tensor)
    rt_or_tensor = dispatch.MakeUnionChecker([rt_checker, tensor_checker])
    checker = dispatch.PySignatureChecker([(0, rt_or_tensor),
                                           (1, rt_or_tensor)])

    t = constant_op.constant([[1, 2], [3, 4]])
    rt = ragged_factory_ops.constant([[1, 2], [3]])

    self.check_signatures(checker, [
        ((t, t), False),
        ((t, rt), True),
        ((rt, t), True),
        ((rt, rt), True),
        ((rt, [rt]), False),
        ((rt, rt, 1, 2, None), True),
    ])  # pyformat: disable

    self.assertEqual(
        repr(checker),
        '<PySignatureChecker args[0]:Union[RaggedTensor, Tensor], '
        'args[1]:Union[RaggedTensor, Tensor]>')

  def testList(self):
    # [FUNCTION DOC]
    # Scenario: a signature checker requiring a list of ragged tensors at position zero.
    # Asserts acceptance for non-empty lists of ragged tensors, rejection for an empty list (no
    # ragged tensor to dispatch on), a bare ragged tensor, or a list containing a non-ragged
    # element, and checks the string form.
    rt_checker = dispatch.MakeInstanceChecker(ragged_tensor.RaggedTensor)
    rt_list_checker = dispatch.MakeListChecker(rt_checker)
    checker = dispatch.PySignatureChecker([(0, rt_list_checker)])

    rt = ragged_factory_ops.constant([[1, 2], [3]])

    self.check_signatures(checker, [
        (([rt],), True),
        (([],), False),
        ((rt,), False),
        (([rt, rt+3, rt*2],), True),
        (([rt, rt.values, rt*2],), False),
    ])  # pyformat: disable

    self.assertEqual(
        repr(checker), '<PySignatureChecker args[0]:List[RaggedTensor]>')

  def testSortByCost(self):
    # [FUNCTION DOC]
    # Scenario: a signature checker over four positions with checkers of different cost
    # (integer, union, list of integers, list of union). Asserts that the string form lists the
    # positions in increasing order of cost, which is the order they will be checked.
    a = dispatch.MakeInstanceChecker(int)
    b = dispatch.MakeInstanceChecker(float)
    c = dispatch.MakeUnionChecker([a, b])
    d = dispatch.MakeListChecker(a)
    e = dispatch.MakeListChecker(c)
    checker = dispatch.PySignatureChecker([(0, e), (1, c), (2, d), (3, a)])

    # Note: `repr(checker)` lists the args in the order they will be checked.
    self.assertEqual(
        repr(checker), '<PySignatureChecker '
        'args[3]:int, '                     # a: cost=1
        'args[1]:Union[int, float], '       # c: cost=3
        'args[2]:List[int], '               # d: cost=10
        'args[0]:List[Union[int, float]]>'  # e: cost=30
        )  # pyformat: disable


@test_util.run_all_in_graph_and_eager_modes
class PythonAPIDispatcherTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the native API dispatcher object, which is created for a named API with its
  # parameter names and defaults, has targets registered with signature checkers, and picks a
  # target for a given positional and keyword argument set. Runs in graph and eager modes.

  def testBasicDispatch(self):
    # [FUNCTION DOC]
    # Scenario: a dispatcher for an API with parameters x, y and an optional name, with one
    # target registered for ragged tensor in the first position. Asserts that dispatching returns
    # the target's result when the first argument is ragged, whether passed positionally, by
    # keyword, with or without the optional name, and returns the not-implemented marker
    # otherwise.
    dispatcher = dispatch.PythonAPIDispatcher('tf.foo', ['x', 'y', 'name'],
                                              (None,))

    rt_checker = dispatch.MakeInstanceChecker(ragged_tensor.RaggedTensor)
    f1 = lambda x, y, name=None: 'f1'
    dispatcher.Register(dispatch.PySignatureChecker([(0, rt_checker)]), f1)

    rt = ragged_factory_ops.constant([[1, 2], [3]])
    self.assertEqual(dispatcher.Dispatch((rt, 5), None), 'f1')
    self.assertEqual(dispatcher.Dispatch((rt, 5, 'my_name'), None), 'f1')
    self.assertEqual(dispatcher.Dispatch((), {'x': rt, 'y': 5}), 'f1')
    self.assertEqual(
        dispatcher.Dispatch((), {
            'x': rt,
            'y': 5,
            'name': 'x'
        }), 'f1')
    self.assertEqual(dispatcher.Dispatch(('foo', rt), None), NotImplemented)
    self.assertEqual(dispatcher.Dispatch(('foo', 'bar'), None), NotImplemented)
    self.assertEqual(
        dispatcher.Dispatch(('foo', 'bar', 'baz'), None), NotImplemented)

  def testMultipleDispatchers(self):
    # [FUNCTION DOC]
    # Scenario: two targets registered on one dispatcher, one keyed on a ragged tensor in the
    # first parameter and one on the second. Asserts that each is selected when only its
    # position holds a ragged tensor (positionally or by keyword), that the not-implemented
    # marker is returned when neither matches, and that a value error about multiple matching
    # dispatch targets is raised when both match.
    dispatcher = dispatch.PythonAPIDispatcher('tf.foo', ['x', 'y', 'name'],
                                              (None,))

    rt_checker = dispatch.MakeInstanceChecker(ragged_tensor.RaggedTensor)
    rt_x_checker = dispatch.PySignatureChecker([(0, rt_checker)])
    rt_y_checker = dispatch.PySignatureChecker([(1, rt_checker)])

    f1 = lambda x, y, name=None: 'f1'
    f2 = lambda x, y, name=None: 'f2'

    rt = ragged_factory_ops.constant([[1, 2], [3]])

    dispatcher.Register(rt_x_checker, f1)
    dispatcher.Register(rt_y_checker, f2)

    self.assertEqual(dispatcher.Dispatch((rt, 5), None), 'f1')
    self.assertEqual(dispatcher.Dispatch(('foo', rt), None), 'f2')
    self.assertEqual(dispatcher.Dispatch(('foo',), {'y': rt}), 'f2')
    self.assertEqual(dispatcher.Dispatch(('foo', 'bar'), None), NotImplemented)
    with self.assertRaisesRegex(
        ValueError, 'Multiple dispatch targets .*'
        r'match the arguments to tf\.foo'):
      dispatcher.Dispatch((rt, rt), None)

  def testListAndUnionDispatch(self):
    # [FUNCTION DOC]
    # Scenario: a target registered for a signature using a union of ragged tensor or tensor for
    # the first parameter and a list of that union for the second. Asserts that dispatch succeeds
    # when at least one value across those arguments is ragged (positionally, by keyword, with an
    # optional name, with an empty list), and returns the not-implemented marker when all values
    # are plain tensors or have wrong types.
    dispatcher = dispatch.PythonAPIDispatcher('tf.foo', ['x', 'ys', 'name'],
                                              (None,))

    rt_checker = dispatch.MakeInstanceChecker(ragged_tensor.RaggedTensor)
    tensor_checker = dispatch.MakeInstanceChecker(tensor.Tensor)
    rt_or_t = dispatch.MakeUnionChecker([rt_checker, tensor_checker])
    list_of_rt_or_t = dispatch.MakeListChecker(rt_or_t)
    f1 = lambda x, ys, name=None: 'f1'
    dispatcher.Register(
        dispatch.PySignatureChecker([(0, rt_or_t), (1, list_of_rt_or_t)]), f1)

    rt = ragged_factory_ops.constant([[1, 2], [3]])
    t = constant_op.constant(5)
    self.assertEqual(dispatcher.Dispatch((rt, [t]), None), 'f1')
    self.assertEqual(dispatcher.Dispatch((rt, [rt]), None), 'f1')
    self.assertEqual(dispatcher.Dispatch((t, [rt]), None), 'f1')
    self.assertEqual(dispatcher.Dispatch((rt, []), None), 'f1')
    self.assertEqual(dispatcher.Dispatch((t, [t, t, rt, t]), None), 'f1')
    self.assertEqual(dispatcher.Dispatch((rt, [t], 'my_name'), None), 'f1')
    self.assertEqual(dispatcher.Dispatch((), {'x': rt, 'ys': [t]}), 'f1')
    self.assertEqual(
        dispatcher.Dispatch((), {
            'x': rt,
            'ys': [t],
            'name': 'x'
        }), 'f1')
    self.assertEqual(dispatcher.Dispatch((t, [t]), None), NotImplemented)
    self.assertEqual(dispatcher.Dispatch((t, []), None), NotImplemented)
    self.assertEqual(dispatcher.Dispatch(('foo', [rt]), None), NotImplemented)
    self.assertEqual(dispatcher.Dispatch(('foo', 'bar'), None), NotImplemented)
    self.assertEqual(
        dispatcher.Dispatch(('foo', 'bar', 'baz'), None), NotImplemented)


if __name__ == '__main__':
  googletest.main()
