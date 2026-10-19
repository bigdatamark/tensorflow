# [MODULE DOC]
# Unit tests for the Session class of the client package (session.py): creating sessions, running
# and fetching tensors, operations, sparse tensors, indexed slices and nested structures, feeding
# values, callables, run options and metadata, interactive sessions, default-session stack, error
# reporting, string handling, device attributes and server-backed sessions.
# It defines one test class and, when run as a script, starts the standard test runner. Importing
# it has no global side effects other than optional imports of the attr and frozendict libraries
# (falling back to a missing attr module and the plain dict type). It is a test module, not a helper.
# Some tests only run in graph (v1) mode; local-server tests start in-process servers.
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
"""Tests for tensorflow.python.client.session.Session."""
import collections
import os
import random
import sys
import threading
import time
import warnings

import numpy as np
import six

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.lib.core import error_codes_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as framework_device_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import function
from tensorflow.python.framework import importer
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import stack
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.framework import versions
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gen_control_flow_ops
# Import gradients to resolve circular imports
from tensorflow.python.ops import gradients  # pylint: disable=unused-import
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
# Import resource_variable_ops for the variables-to-tensor implicit conversion.
from tensorflow.python.ops import resource_variable_ops  # pylint: disable=unused-import
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import googletest
from tensorflow.python.training import server_lib
from tensorflow.python.util import compat

try:
  import attr  # pylint:disable=g-import-not-at-top
except ImportError:
  attr = None

try:
  from frozendict import frozendict  # pylint:disable=g-import-not-at-top
except ImportError:
  frozendict = dict  # pylint:disable=invalid-name

defaultdict = collections.defaultdict  # pylint:disable=invalid-name


@test_util.with_eager_op_as_function
class SessionTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case covering the behavior of Session and InteractiveSession against the runtime.
  # Decorated so eager ops run as functions. Tests are independent; each builds its own graph and
  # session. The class also holds one static helper that builds graphs, used by concurrency tests, and
  # two shared runner helpers used by local-server and direct-session variants of a test.

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture: calls the base class setup, then makes the warnings module always report warnings
    # (no deduplication) so that tests relying on warnings see them. Takes nothing, returns nothing.
    # Side effect: changes the global warning filter.
    super(SessionTest, self).setUp()
    warnings.simplefilter('always')

  def testUseExistingGraph(self):
    # [FUNCTION DOC]
    # Builds a matrix multiplication of two 1x1 constants in an explicit graph on the CPU device,
    # creates a session bound to that graph, and checks the result equals 42.
    with ops.Graph().as_default() as g, ops.device('/cpu:0'):
      a = constant_op.constant(6.0, shape=[1, 1])
      b = constant_op.constant(7.0, shape=[1, 1])
      c = math_ops.matmul(a, b, name='matmul')
    with session.Session(graph=g):
      result = self.evaluate(c)
      self.assertAllEqual(result, [[42.0]])

  def testUseDefaultGraph(self):
    # [FUNCTION DOC]
    # Same computation as the explicit graph test, but the session is created without a graph inside
    # a default-graph scope; checks the session picks up the default graph and yields 42.
    with ops.Graph().as_default(), ops.device('/cpu:0'):
      a = constant_op.constant(6.0, shape=[1, 1])
      b = constant_op.constant(7.0, shape=[1, 1])
      c = math_ops.matmul(a, b, name='matmul')
      with session.Session():
        result = self.evaluate(c)
        self.assertAllEqual(result, [[42.0]])

  def testCreate(self):
    # [FUNCTION DOC]
    # Creates a constant named W1, an identity copy of it, and evaluates the copy once with a feed
    # that overrides W1 by name (checking the fed array comes back) and once without a feed (checking
    # the original constant values, a 2x3 array of tens).
    with session.Session():
      inp = constant_op.constant(10.0, shape=[2, 3], name='W1')
      copy = array_ops.identity(inp)
      # Test with feed.
      # TODO(mrry): Investigate why order='F' didn't work.
      arr = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.float32, order='C')
      copy_val = copy.eval({'W1:0': arr})
      self.assertAllEqual(arr, copy_val)
      # Test without feed.
      copy_val = self.evaluate(copy)
      self.assertAllEqual(
          np.asarray(
              [[10.0, 10.0, 10.0], [10.0, 10.0, 10.0]], dtype=np.float32),
          copy_val)

  def testManyCPUs(self):
    # [FUNCTION DOC]
    # Creates a session whose configuration asks for two CPU devices and zero GPUs, and checks the
    # session's device listing contains exactly two CPU devices and no GPU devices.
    with session.Session(
        config=config_pb2.ConfigProto(device_count={
            'CPU': 2, 'GPU': 0
        })) as sess:
      inp = constant_op.constant(10.0, name='W1')
      self.assertAllEqual(inp, 10.0)

      num_cpu_devices = 0
      num_gpu_devices = 0
      for device in sess.list_devices():
        device_type = framework_device_lib.DeviceSpec.from_string(
            device.name).device_type
        if device_type == 'CPU':
          num_cpu_devices += 1
        elif device_type == 'GPU':
          num_gpu_devices += 1
      self.assertEqual(2, num_cpu_devices)
      self.assertEqual(0, num_gpu_devices)

  def testPerSessionThreads(self):
    # [FUNCTION DOC]
    # Creates a session with the per-session threads option enabled and checks a constant evaluates
    # to its value (only verifies that such a session works).
    with session.Session(
        config=config_pb2.ConfigProto(use_per_session_threads=True)):
      inp = constant_op.constant(10.0, name='W1')
      self.assertAllEqual(inp, 10.0)

  def testSessionInterOpThreadPool(self):
    # [FUNCTION DOC]
    # Exercises the session inter-op thread pool settings. Runs a constant with an empty pool entry,
    # then with a pool of one thread, then with a named global pool selected through run options by
    # pool index. Each run must return the constant's value.
    config_pb = config_pb2.ConfigProto()
    pool = config_pb.session_inter_op_thread_pool.add()
    with session.Session(config=config_pb) as s:
      inp = constant_op.constant(10.0, name='W1')
      results = s.run([inp])
      self.assertAllEqual([10.0], results)

    pool = config_pb.session_inter_op_thread_pool.add()
    pool.num_threads = 1
    with session.Session(config=config_pb) as s:
      inp = constant_op.constant(20.0, name='W2')
      results = s.run([inp])
      self.assertAllEqual([20.0], results)

    pool = config_pb.session_inter_op_thread_pool.add()
    pool.num_threads = 1
    pool.global_name = 't1'
    run_options = config_pb2.RunOptions()
    run_options.inter_op_thread_pool = (
        len(config_pb.session_inter_op_thread_pool) - 1)
    with session.Session(config=config_pb) as s:
      inp = constant_op.constant(30.0, name='W2')
      results = s.run([inp], options=run_options)
      self.assertAllEqual([30.0], results)

  def testErrorsReported(self):
    # [FUNCTION DOC]
    # Checks that fetching a name that does not exist in the graph raises a value error.
    with session.Session() as s:
      constant_op.constant(10.0, name='W1')
      with self.assertRaises(ValueError):
        s.run('foo:0')

  def testErrorPayload(self):
    # [FUNCTION DOC]
    # Evaluates a placeholder with no feed and checks that the raised op error carries the op that
    # failed (matches the placeholder's op), using a predicate.
    with session.Session():
      a = array_ops.placeholder(dtypes.float32)
      with self.assertRaisesOpError(lambda e: e.op == a.op):
        self.evaluate(a)

  def testErrorCodeWithNoNodeDef(self):
    # [FUNCTION DOC]
    # Runs a partial run with a bogus handle and checks the resulting op error has no op, no node
    # definition and the invalid-argument error code.
    with session.Session() as s:
      a = array_ops.placeholder(dtypes.float32, shape=[])
      b = array_ops.placeholder(dtypes.float32, shape=[])
      r1 = math_ops.add(a, b)

      def exc_predicate(e):
        # [FUNCTION DOC]
        # HELPER: predicate for the surrounding error assertion.
        # Returns true when the error has no op, no node definition and the invalid-argument code.
        return (e.op is None and e.node_def is None and
                e.error_code == error_codes_pb2.INVALID_ARGUMENT)

      with self.assertRaisesOpError(exc_predicate):
        # Run with a bogus handle.
        s.partial_run('foo', r1, feed_dict={a: 1, b: 2})

  def testErrorBasedOn(self):
    # [FUNCTION DOC]
    # Builds a chain of ops linked via "original op" markers (placeholder derived from identity derived
    # from constant) and checks that the error from evaluating the placeholder exposes the chain of
    # original ops. The original-op relationships are artificial and only used for the check.
    with session.Session() as sess:
      a = constant_op.constant(0.0, shape=[2, 3])
      # NOTE(mrry): The original_op is nonsense, but used here to test that the
      #   errors are reported correctly.
      with sess.graph._original_op(a.op):
        b = array_ops.identity(a, name='id')
      with sess.graph._original_op(b.op):
        c = array_ops.placeholder(dtypes.float32)

      def exc_predicate(e):
        # [FUNCTION DOC]
        # HELPER: predicate for the surrounding error assertion.
        # Returns true when the error's op is the placeholder op and its original-op chain leads to the
        # identity op and then the constant op.
        return (e.op == c.op and e.op._original_op == b.op and
                e.op._original_op._original_op == a.op)

      with self.assertRaisesOpError(exc_predicate):
        self.evaluate(c)

  def testFetchNone(self):
    # [FUNCTION DOC]
    # Checks that fetching nothing/null, in a list or as a mapping value, raises a type error.
    with session.Session() as s:
      a = constant_op.constant(1.0)
      with self.assertRaises(TypeError):
        s.run(None)
      with self.assertRaises(TypeError):
        s.run([None])
      with self.assertRaises(TypeError):
        s.run({'b': None})
      with self.assertRaises(TypeError):
        s.run({'a': a, 'b': None})

  def testFetchSingleton(self):
    # [FUNCTION DOC]
    # Fetches a single tensor (expects its value) and a single operation (expects nothing/null), both
    # by running the session directly and through callables made from the session.
    with session.Session() as sess:
      a = constant_op.constant(42.0)
      res = sess.run(a)
      self.assertEqual(42.0, res)
      res = sess.run(a.op)  # An op, not a tensor.
      self.assertIsNone(res)
      tensor_runner = sess.make_callable(a)
      res = tensor_runner()
      self.assertEqual(42.0, res)
      op_runner = sess.make_callable(a.op)
      res = op_runner()
      self.assertIsNone(res)

  def testFetchSingletonByName(self):
    # [FUNCTION DOC]
    # Fetches a tensor by its name string and expects its value; fetching its op returns nothing.
    with session.Session() as sess:
      a = constant_op.constant(42.0)
      res = sess.run(a.name)
      self.assertEqual(42.0, res)
      res = sess.run(a.op)  # An op, not a tensor.
      self.assertIsNone(res)

  def testFetchList(self):
    # [FUNCTION DOC]
    # Fetches a list mixing tensors, a no-op, a name string and an assign op's operation. Expects a
    # list with values for tensors and nothing for ops, both from run and from a callable.
    with session.Session() as sess:
      a = constant_op.constant(42.0)
      b = control_flow_ops.no_op()  # An op, not a tensor.
      c = constant_op.constant(44.0)
      v = variables.Variable([54.0])
      assign = v.assign([63.0])
      res = sess.run([a, b, c, a.name, assign.op])
      self.assertIsInstance(res, list)
      self.assertEqual([42.0, None, 44.0, 42.0, None], res)
      list_runner = sess.make_callable([a, b, c, a.name, assign.op])
      res = list_runner()
      self.assertIsInstance(res, list)
      self.assertEqual([42.0, None, 44.0, 42.0, None], res)

  def testFetchTuple(self):
    # [FUNCTION DOC]
    # Same as the list test but fetching a tuple; the result must be a tuple with matching values,
    # both from run and from a callable.
    with session.Session() as sess:
      a = constant_op.constant(42.0)
      b = control_flow_ops.no_op()  # An op, not a tensor.
      c = constant_op.constant(44.0)
      res = sess.run((a, b, c, a.name))
      self.assertIsInstance(res, tuple)
      self.assertEqual((42.0, None, 44.0, 42.0), res)
      tuple_runner = sess.make_callable((a, b, c, a.name))
      res = tuple_runner()
      self.assertIsInstance(res, tuple)
      self.assertEqual((42.0, None, 44.0, 42.0), res)

  def testFetchNamedTuple(self):
    # [FUNCTION DOC]
    # Fetches a named tuple of a tensor, a no-op and another tensor. The result must keep the named
    # tuple type with the expected field values, both from run and from a callable.
    # pylint: disable=invalid-name
    ABC = collections.namedtuple('ABC', ['a', 'b', 'c'])
    # pylint: enable=invalid-name
    with session.Session() as sess:
      a = constant_op.constant(42.0)
      b = control_flow_ops.no_op()  # An op, not a tensor.
      c = constant_op.constant(44.0)
      res = sess.run(ABC(a, b, c))
      self.assertIsInstance(res, ABC)
      self.assertEqual(42.0, res.a)
      self.assertIsNone(res.b)
      self.assertEqual(44.0, res.c)
      namedtuple_runner = sess.make_callable(ABC(a, b, c))
      res = namedtuple_runner()
      self.assertIsInstance(res, ABC)
      self.assertEqual(42.0, res.a)
      self.assertIsNone(res.b)
      self.assertEqual(44.0, res.c)

  def testFetchDict(self):
    # [FUNCTION DOC]
    # Fetches a mapping with tensors and a no-op; the result is a mapping with the same keys and
    # values (nothing for the op).
    with session.Session() as sess:
      a = constant_op.constant(42.0)
      b = control_flow_ops.no_op()  # An op, not a tensor.
      c = constant_op.constant(44.0)
      res = sess.run({'a': a, 'b': b, 'c': c})
      self.assertIsInstance(res, dict)
      self.assertEqual(42.0, res['a'])
      self.assertIsNone(res['b'])
      self.assertEqual(44.0, res['c'])

  def testFetchOrderedDict(self):
    # [FUNCTION DOC]
    # Fetches an ordered mapping with integer keys; the result must be an ordered mapping that
    # preserves the key order and holds the expected values.
    with session.Session() as sess:
      a = constant_op.constant(42.0)
      b = control_flow_ops.no_op()  # An op, not a tensor.
      c = constant_op.constant(44.0)
      res = sess.run(collections.OrderedDict([(3, a), (2, b), (1, c)]))
      self.assertIsInstance(res, collections.OrderedDict)
      self.assertEqual([3, 2, 1], list(res.keys()))
      self.assertEqual(42.0, res[3])
      self.assertIsNone(res[2])
      self.assertEqual(44.0, res[1])

  @test_util.run_v1_only('b/120545219')
  def testFetchAttrs(self):
    # [FUNCTION DOC]
    # Graph-mode only test. Skipped when the attrs library is missing. Defines an attrs class with two
    # fields, fetches an instance holding tensors, and checks the result is an instance of that class
    # with array values. Then feeds a value for one field's tensor and checks the override is used.
    if attr is None:
      self.skipTest('attr module is unavailable.')

    @attr.s
    class SampleAttr(object):
      # [CLASS DOC]
      # Local attrs data class with two fields, used as a fetchable structure in the test.
      field1 = attr.ib()
      field2 = attr.ib()

    val1 = np.array([1.2, 3.4, 5.6])
    val2 = np.array([[1, 2], [4, 3]])
    val3 = np.array([10, 20, 30])

    t1 = constant_op.constant(val1)
    t2 = constant_op.constant(val2)

    sample = SampleAttr(t1, t2)
    with session.Session() as sess:
      result = sess.run(sample)
      self.assertIsInstance(result, SampleAttr)
      self.assertAllEqual(val1, result.field1)
      self.assertAllEqual(val2, result.field2)

      result = sess.run(sample, feed_dict={sample.field1: val3})
      self.assertIsInstance(result, SampleAttr)
      self.assertAllEqual(val3, result.field1)
      self.assertAllEqual(val2, result.field2)

  @test_util.run_v1_only('b/120545219')
  def testFetchNestedAttrs(self):
    # [FUNCTION DOC]
    # Graph-mode only test. Skipped when the attrs library is missing. Builds nested attrs instances
    # containing tensors, including inside a mapping and a list, fetches them and checks the nested
    # types (attrs class, array, mapping, list) and all values are preserved.
    if attr is None:
      self.skipTest('attr module is unavailable.')

    @attr.s
    class SampleAttr(object):
      # [CLASS DOC]
      # Local attrs data class with two fields, used for nested fetch structures in the test.
      field0 = attr.ib()
      field1 = attr.ib()

    v1 = 10
    v2 = 20
    v3 = np.float32(1.2)
    v4 = np.float32(3.4)
    v5 = np.float64(100.001)
    v6 = np.float64(-23.451)
    arr1 = np.array([1.2, 6.7, 3.4])
    arr2 = np.array([7, 11, 3])
    sample = SampleAttr(
        SampleAttr(
            SampleAttr(constant_op.constant(v1), constant_op.constant(v2)),
            SampleAttr(constant_op.constant(arr1), constant_op.constant(arr2))),
        {'A': SampleAttr(constant_op.constant(v3), constant_op.constant(v4)),
         'B': [SampleAttr(constant_op.constant(v5), constant_op.constant(v6))]})

    with session.Session() as sess:
      result = sess.run(sample)
      self.assertIsInstance(result, SampleAttr)
      self.assertIsInstance(result.field0, SampleAttr)
      self.assertIsInstance(result.field0.field0, SampleAttr)
      self.assertIsInstance(result.field0.field1, SampleAttr)
      self.assertIsInstance(result.field0.field1.field0, np.ndarray)
      self.assertAllEqual(arr1, result.field0.field1.field0)
      self.assertIsInstance(result.field0.field1.field1, np.ndarray)
      self.assertAllEqual(arr2, result.field0.field1.field1)
      self.assertIsInstance(result.field1, dict)
      self.assertIn('A', result.field1)
      self.assertIn('B', result.field1)
      self.assertIsInstance(result.field1['A'], SampleAttr)
      self.assertAllEqual(
          [v3, v4],
          [result.field1['A'].field0, result.field1['A'].field1])
      self.assertIsInstance(result.field1['B'], list)
      self.assertEqual(1, len(result.field1['B']))
      self.assertIsInstance(result.field1['B'][0], SampleAttr)
      self.assertAllEqual(
          [v5, v6],
          [result.field1['B'][0].field0, result.field1['B'][0].field1])

  def testFetchNestingEmptyOneLevel(self):
    # [FUNCTION DOC]
    # Fetches lists containing empty list, empty tuple and empty mapping (with and without an extra
    # tensor) and checks the result keeps the container types, empties stay empty, and the tensor
    # value is returned.
    with session.Session() as sess:
      a_val = 11.0
      a = constant_op.constant(a_val)

      res = sess.run([[], tuple(), {}])
      self.assertIsInstance(res, list)
      self.assertEqual(3, len(res))
      self.assertIsInstance(res[0], list)
      self.assertEqual(0, len(res[0]))
      self.assertIsInstance(res[1], tuple)
      self.assertEqual(0, len(res[1]))
      self.assertIsInstance(res[2], dict)
      self.assertEqual(0, len(res[2]))

      res = sess.run([[], tuple(), {}, a])
      self.assertIsInstance(res, list)
      self.assertEqual(4, len(res))
      self.assertIsInstance(res[0], list)
      self.assertEqual(0, len(res[0]))
      self.assertIsInstance(res[1], tuple)
      self.assertEqual(0, len(res[1]))
      self.assertIsInstance(res[2], dict)
      self.assertEqual(0, len(res[2]))
      self.assertEqual(a_val, res[3])

  def testFetchNestingOneLevel(self):
    # [FUNCTION DOC]
    # Fetches containers (list, tuple, named tuples, mapping, frozen mapping, default mapping) nested
    # one level inside outer containers: a list, a tuple, a named tuple and a mapping. Mixes tensors,
    # tensor names and a no-op. Checks the outer and inner container types, lengths, values (nothing for
    # the op) and that the default factory of default mappings is preserved.
    with session.Session() as sess:
      # pylint: disable=invalid-name
      ABC = collections.namedtuple('ABC', ['a', 'b', 'c'])
      DEFGHI = collections.namedtuple('DEFGHI', ['d', 'e', 'f', 'g', 'h', 'i'])
      # pylint: enable=invalid-name
      a_val = 42.0
      b_val = None
      c_val = 44.0
      a = constant_op.constant(a_val)
      b = control_flow_ops.no_op()  # An op, not a tensor.
      c = constant_op.constant(c_val)
      test_dct = {'a': a.name, 'c': c, 'b': b}
      test_dct_types = [dict, frozendict, defaultdict]
      # List of lists, tuples, namedtuple, dict, frozendict, and defaultdict
      res = sess.run([
          [a, b, c],
          (a, b, c),
          ABC(a=a, b=b, c=c),
          dict(test_dct),
          frozendict(test_dct),
          defaultdict(str, test_dct),
      ])
      self.assertIsInstance(res, list)
      self.assertEqual(6, len(res))
      self.assertIsInstance(res[0], list)
      self.assertEqual(3, len(res[0]))
      self.assertEqual(a_val, res[0][0])
      self.assertEqual(b_val, res[0][1])
      self.assertEqual(c_val, res[0][2])
      self.assertIsInstance(res[1], tuple)
      self.assertEqual(3, len(res[1]))
      self.assertEqual(a_val, res[1][0])
      self.assertEqual(b_val, res[1][1])
      self.assertEqual(c_val, res[1][2])
      self.assertIsInstance(res[2], ABC)
      self.assertEqual(a_val, res[2].a)
      self.assertEqual(b_val, res[2].b)
      self.assertEqual(c_val, res[2].c)
      for expected_type, r in zip(test_dct_types, res[3:]):
        self.assertIsInstance(r, expected_type)
        self.assertEqual(3, len(r))
        self.assertEqual(a_val, r['a'])
        self.assertEqual(b_val, r['b'])
        self.assertEqual(c_val, r['c'])
      self.assertEqual(res[5].default_factory, str)
      # Tuple of lists, tuples, namedtuple, dict, frozendict, and defaultdict
      res = sess.run(([a, b, c], (a.name, b, c), ABC(a=a, b=b,
                                                     c=c), dict(test_dct),
                      frozendict(test_dct), defaultdict(str, test_dct)))
      self.assertIsInstance(res, tuple)
      self.assertEqual(6, len(res))
      self.assertIsInstance(res[0], list)
      self.assertEqual(3, len(res[0]))
      self.assertEqual(a_val, res[0][0])
      self.assertEqual(b_val, res[0][1])
      self.assertEqual(c_val, res[0][2])
      self.assertIsInstance(res[1], tuple)
      self.assertEqual(3, len(res[1]))
      self.assertEqual(a_val, res[1][0])
      self.assertEqual(b_val, res[1][1])
      self.assertEqual(c_val, res[1][2])
      self.assertIsInstance(res[2], ABC)
      self.assertEqual(a_val, res[2].a)
      self.assertEqual(b_val, res[2].b)
      self.assertEqual(c_val, res[2].c)
      for expected_type, r in zip(test_dct_types, res[3:]):
        self.assertIsInstance(r, expected_type)
        self.assertEqual(3, len(r))
        self.assertEqual(a_val, r['a'])
        self.assertEqual(b_val, r['b'])
        self.assertEqual(c_val, r['c'])
      self.assertEqual(res[5].default_factory, str)

      # Namedtuple of lists, tuples, namedtuples, dict, frozendict, defaultdict
      res = sess.run(
          DEFGHI(
              d=[a, b, c],
              e=(a, b, c),
              f=ABC(a=a.name, b=b, c=c),
              g=dict(test_dct),
              h=frozendict(test_dct),
              i=defaultdict(str, test_dct)))
      self.assertIsInstance(res, DEFGHI)
      self.assertIsInstance(res.d, list)
      self.assertEqual(3, len(res.d))
      self.assertEqual(a_val, res.d[0])
      self.assertEqual(b_val, res.d[1])
      self.assertEqual(c_val, res.d[2])
      self.assertIsInstance(res.e, tuple)
      self.assertEqual(3, len(res.e))
      self.assertEqual(a_val, res.e[0])
      self.assertEqual(b_val, res.e[1])
      self.assertEqual(c_val, res.e[2])
      self.assertIsInstance(res.f, ABC)
      self.assertEqual(a_val, res.f.a)
      self.assertEqual(b_val, res.f.b)
      self.assertEqual(c_val, res.f.c)
      self.assertIsInstance(res.g, dict)
      self.assertEqual(3, len(res.g))
      self.assertEqual(a_val, res.g['a'])
      self.assertEqual(b_val, res.g['b'])
      self.assertEqual(c_val, res.g['c'])
      self.assertIsInstance(res.h, frozendict)
      self.assertEqual(3, len(res.h))
      self.assertEqual(a_val, res.h['a'])
      self.assertEqual(b_val, res.h['b'])
      self.assertEqual(c_val, res.h['c'])
      self.assertIsInstance(res.i, defaultdict)
      self.assertEqual(3, len(res.i))
      self.assertEqual(a_val, res.i['a'])
      self.assertEqual(b_val, res.i['b'])
      self.assertEqual(c_val, res.i['c'])
      self.assertEqual(res.i.default_factory, str)
      # Dict of lists, tuples, namedtuples, dict, frozendict, defaultdict
      res = sess.run({
          'd': [a, b, c],
          'e': (a, b, c),
          'f': ABC(a=a, b=b, c=c),
          'g': dict(test_dct),
          'h': frozendict(test_dct),
          'i': defaultdict(str, test_dct),
      })
      self.assertIsInstance(res, dict)
      self.assertEqual(6, len(res))
      self.assertIsInstance(res['d'], list)
      self.assertEqual(3, len(res['d']))
      self.assertEqual(a_val, res['d'][0])
      self.assertEqual(b_val, res['d'][1])
      self.assertEqual(c_val, res['d'][2])
      self.assertIsInstance(res['e'], tuple)
      self.assertEqual(3, len(res['e']))
      self.assertEqual(a_val, res['e'][0])
      self.assertEqual(b_val, res['e'][1])
      self.assertEqual(c_val, res['e'][2])
      self.assertIsInstance(res['f'], ABC)
      self.assertEqual(a_val, res['f'].a)
      self.assertEqual(b_val, res['f'].b)
      self.assertEqual(c_val, res['f'].c)
      for expected_type, r_key in zip(test_dct_types, ('g', 'h', 'i')):
        r = res[r_key]
        self.assertIsInstance(r, expected_type)
        self.assertEqual(3, len(r))
        self.assertEqual(a_val, r['a'])
        self.assertEqual(b_val, r['b'])
        self.assertEqual(c_val, r['c'])
      self.assertEqual(res['i'].default_factory, str)

  def testFetchTensorObject(self):
    # [FUNCTION DOC]
    # Fetches the product of two constants as a single value, list, via evaluate, multiple fetches,
    # mapping and nested lists, checking all values and that the same tensor appearing twice returns
    # equal results.
    with session.Session() as s:
      a = constant_op.constant(1.0, shape=[1, 2])
      b = constant_op.constant(2.0, shape=[2, 3])
      c = math_ops.matmul(a, b)
      results_with_list = s.run([c])
      self.assertAllEqual([[4.0, 4.0, 4.0]], results_with_list[0])
      results_with_single = s.run(c)
      self.assertAllEqual([[4.0, 4.0, 4.0]], results_with_single)
      results_with_get = self.evaluate(c)
      self.assertAllEqual([[4.0, 4.0, 4.0]], results_with_get)
      a_val, b_val = s.run([a, b])  # Test multiple fetches.
      self.assertAllEqual([[1.0, 1.0]], a_val)
      self.assertAllEqual([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], b_val)
      results_with_dict = s.run({'a': [a], 'b': b, 'z': [a, b]})
      self.assertAllEqual([[1.0, 1.0]], results_with_dict['a'][0])
      self.assertAllEqual([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
                          results_with_dict['b'])
      self.assertAllEqual(results_with_dict['a'][0], results_with_dict['z'][0])
      self.assertAllEqual(results_with_dict['b'], results_with_dict['z'][1])

      # Test nested structures
      results_with_nested_list = s.run([[[a, b], b], a, [a, b]])
      self.assertAllEqual([[1.0, 1.0]], results_with_nested_list[0][0][0])
      self.assertAllEqual([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
                          results_with_nested_list[0][0][1])
      self.assertAllEqual(results_with_nested_list[0][0][0],
                          results_with_nested_list[1])
      self.assertAllEqual(results_with_nested_list[1],
                          results_with_nested_list[2][0])
      self.assertAllEqual(results_with_nested_list[0][0][1],
                          results_with_nested_list[0][1])
      self.assertAllEqual(results_with_nested_list[0][1],
                          results_with_nested_list[2][1])

  def testFetchScalar(self):
    # [FUNCTION DOC]
    # For several numpy scalar types, adds two scalars and fetches the result as a single value, in a
    # list, in a mapping and in nested lists; checks the value and that the numpy scalar type is
    # preserved.
    with session.Session() as s:
      for scalar in np.int32, np.int64, np.float16, np.float32, np.float64:
        x = scalar(7)
        y = scalar(8)
        tf_x = constant_op.constant(x, shape=[])
        tf_y = constant_op.constant(y)
        tf_xy = math_ops.add(tf_x, tf_y)
        # Single fetch
        xy = s.run(tf_xy)
        self.assertEqual(scalar, type(xy))
        self.assertEqual(x + y, xy)
        # List fetch
        xy, = s.run([tf_xy])
        self.assertEqual(scalar, type(xy))
        self.assertEqual(x + y, xy)
        # Dict fetch
        xy = s.run({'xy': tf_xy})['xy']
        self.assertEqual(scalar, type(xy))
        self.assertEqual(x + y, xy)
        # Nested list fetch
        xy = s.run([[[tf_xy]], tf_xy, [tf_xy]])
        self.assertAllEqual(xy, [[[x + y]], x + y, [x + y]])
        self.assertEqual(scalar, type(xy[0][0][0]))
        self.assertEqual(scalar, type(xy[1]))
        self.assertEqual(scalar, type(xy[2][0]))

  def testFetchOperationObject(self):
    # [FUNCTION DOC]
    # Creates a variable, runs its initializer operation, fetches the variable and checks its value.
    with session.Session() as s:
      a = constant_op.constant(1.0, shape=[1, 2])
      v = variables.Variable(a, name='testFetchOperationObject_v')
      s.run(v.initializer)
      v_val = s.run(v)
      self.assertAllEqual([[1.0, 1.0]], v_val)

  def testFetchSparseTensor(self):
    # [FUNCTION DOC]
    # Fetches a sparse tensor in many container shapes (single, list, mapping value, mapping of list,
    # nested lists) and checks the result can be used both as a three-part tuple (indices, values,
    # dense shape) and as a sparse tensor value with those fields.
    with session.Session() as s:
      indices = np.array([[3, 2, 0], [4, 5, 1]]).astype(np.int64)
      values = np.array([1.0, 2.0]).astype(np.float32)
      shape = np.array([7, 9, 2]).astype(np.int64)
      sp = sparse_tensor.SparseTensor(
          constant_op.constant(indices), constant_op.constant(values),
          constant_op.constant(shape))
      # Single fetch, use as tuple
      sp_out = s.run(sp)
      indices_out, values_out, shape_out = sp_out
      self.assertAllEqual(indices_out, indices)
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(shape_out, shape)
      # Single fetch, use as SparseTensorValue
      sp_out = s.run(sp)
      self.assertAllEqual(sp_out.indices, indices)
      self.assertAllEqual(sp_out.values, values)
      self.assertAllEqual(sp_out.dense_shape, shape)
      # Tuple fetch, use as tuple
      indices_out, values_out, shape_out = s.run(sp)
      self.assertAllEqual(indices_out, indices)
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(shape_out, shape)
      # List fetch, use as tuple
      (indices_out, values_out, shape_out), = s.run([sp])
      self.assertAllEqual(indices_out, indices)
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(shape_out, shape)
      # List fetch, use as SparseTensorValue
      sp_out, = s.run([sp])
      self.assertAllEqual(sp_out.indices, indices)
      self.assertAllEqual(sp_out.values, values)
      self.assertAllEqual(sp_out.dense_shape, shape)
      # Dict fetch (single value), use as tuple
      indices_out, values_out, shape_out = s.run({'sp': sp})['sp']
      self.assertAllEqual(indices_out, indices)
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(shape_out, shape)
      # Dict fetch (list value), use as tuple
      (indices_out, values_out, shape_out), = s.run({'sp': [sp]})['sp']
      self.assertAllEqual(indices_out, indices)
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(shape_out, shape)
      # Dict fetch, use as SparseTensorValue
      sp_out = s.run({'sp': sp})['sp']
      self.assertAllEqual(sp_out.indices, indices)
      self.assertAllEqual(sp_out.values, values)
      self.assertAllEqual(sp_out.dense_shape, shape)
      # Nested list fetch use as tuple
      sp_out = s.run([[[sp]], sp])
      indices_out, values_out, shape_out = sp_out[0][0][0]
      self.assertAllEqual(indices_out, indices)
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(shape_out, shape)
      indices_out, values_out, shape_out = sp_out[1]
      self.assertAllEqual(indices_out, indices)
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(shape_out, shape)
      # Nested list fetch, use as SparseTensorValue
      sp_out = s.run([[[sp]], sp])
      self.assertAllEqual(sp_out[0][0][0].indices, indices)
      self.assertAllEqual(sp_out[0][0][0].values, values)
      self.assertAllEqual(sp_out[0][0][0].dense_shape, shape)
      self.assertAllEqual(sp_out[1].indices, indices)
      self.assertAllEqual(sp_out[1].values, values)
      self.assertAllEqual(sp_out[1].dense_shape, shape)

  def testFeedSparseTensor(self):
    # [FUNCTION DOC]
    # Builds a sparse tensor from placeholders and feeds it either as a tuple or as a sparse tensor
    # value, fetching its parts, the sparse tensor itself, or a derived sparse tensor. Checks the
    # fetched indices, values and dense shape match the fed data.
    with session.Session() as s:
      indices = np.array([[3, 2, 0], [4, 5, 1]]).astype(np.int64)
      values = np.array([1.0, 2.0]).astype(np.float32)
      shape = np.array([7, 9, 2]).astype(np.int64)
      sp = sparse_tensor.SparseTensor(
          array_ops.placeholder(dtype=np.int64, shape=(2, 3)),
          array_ops.placeholder(dtype=np.float32, shape=(2,)),
          array_ops.placeholder(dtype=np.int64, shape=(3,)),
      )
      sp_indices = array_ops.identity(sp.indices)
      sp_values = array_ops.identity(sp.values)
      sp_shape = array_ops.identity(sp.dense_shape)
      sp2 = sparse_tensor.SparseTensor(sp_indices, sp_values, sp_shape)
      # Feed with tuple
      indices_out, values_out, shape_out = s.run(
          [sp_indices, sp_values, sp_shape], {
              sp: (indices, values, shape)
          })
      self.assertAllEqual(indices_out, indices)
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(shape_out, shape)
      # Feed with tuple, fetch sp directly
      sp_out = s.run(sp, {sp: (indices, values, shape)})
      self.assertAllEqual(sp_out.indices, indices)
      self.assertAllEqual(sp_out.values, values)
      self.assertAllEqual(sp_out.dense_shape, shape)
      # Feed with SparseTensorValue
      indices_out, values_out, shape_out = s.run(
          [sp_indices, sp_values, sp_shape], {
              sp: sparse_tensor.SparseTensorValue(indices, values, shape)
          })
      self.assertAllEqual(indices_out, indices)
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(shape_out, shape)
      # Feed with SparseTensorValue, fetch SparseTensorValue
      sp2_out = s.run(sp2, {
          sp: sparse_tensor.SparseTensorValue(indices, values, shape)
      })
      self.assertAllEqual(sp2_out.indices, indices)
      self.assertAllEqual(sp2_out.values, values)
      self.assertAllEqual(sp2_out.dense_shape, shape)
      # Feed SparseTensorValue and fetch sp directly.
      sp_out = s.run(sp, {
          sp: sparse_tensor.SparseTensorValue(indices, values, shape)
      })
      self.assertAllEqual(sp_out.indices, indices)
      self.assertAllEqual(sp_out.values, values)
      self.assertAllEqual(sp_out.dense_shape, shape)

  def testFeedSparsePlaceholder(self):
    # [FUNCTION DOC]
    # Uses a sparse placeholder and feeds it as a tuple and as a sparse tensor value, checking the
    # fetched parts and a derived sparse tensor equal the fed data.
    with session.Session() as s:
      indices = np.array([[3, 2, 0], [4, 5, 1]]).astype(np.int64)
      values = np.array([1.0, 2.0]).astype(np.float32)
      shape = np.array([7, 9, 2]).astype(np.int64)
      sp = array_ops.sparse_placeholder(dtype=np.float32, name='placeholder1')
      sp_indices = array_ops.identity(sp.indices)
      sp_values = array_ops.identity(sp.values)
      sp_shape = array_ops.identity(sp.dense_shape)
      sp2 = sparse_tensor.SparseTensor(sp_indices, sp_values, sp_shape)
      # Feed with tuple
      indices_out, values_out, shape_out = s.run(
          [sp_indices, sp_values, sp_shape], {
              sp: (indices, values, shape)
          })
      self.assertAllEqual(indices_out, indices)
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(shape_out, shape)
      # Feed with SparseTensorValue
      indices_out, values_out, shape_out = s.run(
          [sp_indices, sp_values, sp_shape], {
              sp: sparse_tensor.SparseTensorValue(indices, values, shape)
          })
      self.assertAllEqual(indices_out, indices)
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(shape_out, shape)
      # Feed with SparseTensorValue, fetch SparseTensorValue
      sp2_out = s.run(sp2, {
          sp: sparse_tensor.SparseTensorValue(indices, values, shape)
      })
      self.assertAllEqual(sp2_out.indices, indices)
      self.assertAllEqual(sp2_out.values, values)
      self.assertAllEqual(sp2_out.dense_shape, shape)

  def testFeedSparsePlaceholderPartialShape(self):
    # [FUNCTION DOC]
    # Same as the sparse placeholder test but the placeholder has a partially unknown shape;
    # feeding a tuple and a sparse tensor value must work and return the fed data.
    with session.Session() as s:
      indices = np.array([[3, 2, 0], [4, 5, 1]]).astype(np.int64)
      values = np.array([1.0, 2.0]).astype(np.float32)
      shape = np.array([7, 9, 2]).astype(np.int64)
      sp = array_ops.sparse_placeholder(
          shape=[None, 9, 2], dtype=np.float32, name='placeholder1')
      sp_indices = array_ops.identity(sp.indices)
      sp_values = array_ops.identity(sp.values)
      sp_shape = array_ops.identity(sp.dense_shape)
      sp2 = sparse_tensor.SparseTensor(sp_indices, sp_values, sp_shape)
      # Feed with tuple
      indices_out, values_out, shape_out = s.run(
          [sp_indices, sp_values, sp_shape], {
              sp: (indices, values, shape)
          })
      self.assertAllEqual(indices_out, indices)
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(shape_out, shape)
      # Feed with SparseTensorValue
      indices_out, values_out, shape_out = s.run(
          [sp_indices, sp_values, sp_shape], {
              sp: sparse_tensor.SparseTensorValue(indices, values, shape)
          })
      self.assertAllEqual(indices_out, indices)
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(shape_out, shape)
      # Feed with SparseTensorValue, fetch SparseTensorValue
      sp2_out = s.run(sp2, {
          sp: sparse_tensor.SparseTensorValue(indices, values, shape)
      })
      self.assertAllEqual(sp2_out.indices, indices)
      self.assertAllEqual(sp2_out.values, values)
      self.assertAllEqual(sp2_out.dense_shape, shape)

  def testFeedSparsePlaceholderConstantShape(self):
    # [FUNCTION DOC]
    # Uses a sparse placeholder created with a fully known constant shape. Checks the dense shape tensor
    # and the static shape equal that shape, and that feeding only indices and values (a two-part tuple)
    # works and the dense shape comes back as the constant.
    with session.Session() as s:
      indices = np.array([[3, 2, 0], [4, 5, 1]]).astype(np.int64)
      values = np.array([1.0, 2.0]).astype(np.float32)
      shape = np.array([7, 9, 2]).astype(np.int64)
      sp = array_ops.sparse_placeholder(
          dtype=np.float32, shape=shape, name='placeholder1')
      self.assertAllEqual(sp.dense_shape.eval(session=s), shape)
      self.assertAllEqual(tensor_util.constant_value(sp.shape), shape)
      sp_indices = array_ops.identity(sp.indices)
      sp_values = array_ops.identity(sp.values)
      sp_shape = array_ops.identity(sp.dense_shape)
      # Feed with tuple
      indices_out, values_out, shape_out = s.run(
          [sp_indices, sp_values, sp_shape], {
              sp: (indices, values)
          })
      self.assertAllEqual(indices_out, indices)
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(shape_out, shape)

  def testFetchIndexedSlices(self):
    # [FUNCTION DOC]
    # Fetches an indexed slices object (values, indices, dense shape) in the same container shapes as
    # the sparse tensor test and checks it can be used both as a tuple and as an indexed slices value.
    with session.Session() as s:
      indices = np.array([[3, 2, 0], [4, 5, 1]]).astype(np.int64)
      values = np.array([1.0, 2.0]).astype(np.float32)
      dense_shape = np.array([7, 9, 2]).astype(np.int64)
      ind = indexed_slices.IndexedSlices(
          constant_op.constant(values), constant_op.constant(indices),
          constant_op.constant(dense_shape))
      # Single fetch, use as tuple
      ind_out = s.run(ind)
      values_out, indices_out, dense_shape_out = ind_out
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(indices_out, indices)
      self.assertAllEqual(dense_shape_out, dense_shape)
      # Single fetch, use as IndexedSlicesValue
      ind_out = s.run(ind)
      self.assertAllEqual(ind_out.values, values)
      self.assertAllEqual(ind_out.indices, indices)
      self.assertAllEqual(ind_out.dense_shape, dense_shape)
      # Tuple fetch, use as tuple
      values_out, indices_out, dense_shape_out = s.run(ind)
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(indices_out, indices)
      self.assertAllEqual(dense_shape_out, dense_shape)
      # List fetch, use as tuple
      (values_out, indices_out, dense_shape_out), = s.run([ind])
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(indices_out, indices)
      self.assertAllEqual(dense_shape_out, dense_shape)
      # List fetch, use as IndexedSlicesValue
      ind_out, = s.run([ind])
      self.assertAllEqual(ind_out.values, values)
      self.assertAllEqual(ind_out.indices, indices)
      self.assertAllEqual(ind_out.dense_shape, dense_shape)

  def testFeedIndexedSlices(self):
    # [FUNCTION DOC]
    # Builds indexed slices from placeholders and feeds them as a tuple and as an indexed slices
    # value, fetching components or a derived indexed slices; checks values, indices and dense shape.
    with session.Session() as s:
      values = np.array([1.0, 2.0]).astype(np.float32)
      indices = np.array([[3, 2, 0], [4, 5, 1]]).astype(np.int64)
      dense_shape = np.array([7, 9, 2]).astype(np.int64)
      ind = indexed_slices.IndexedSlices(
          array_ops.placeholder(dtype=np.float32, shape=(2,)),
          array_ops.placeholder(dtype=np.int64, shape=(2, 3)),
          array_ops.placeholder(dtype=np.int64, shape=(3,)),
      )
      ind_values = array_ops.identity(ind.values)
      ind_indices = array_ops.identity(ind.indices)
      ind_dense_shape = array_ops.identity(ind.dense_shape)
      ind2 = indexed_slices.IndexedSlices(ind_values, ind_indices,
                                          ind_dense_shape)
      # Feed with tuple
      values_out, indices_out, dense_shape_out = s.run(
          [ind_values, ind_indices, ind_dense_shape], {
              ind: (values, indices, dense_shape)
          })
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(indices_out, indices)
      self.assertAllEqual(dense_shape_out, dense_shape)
      # Feed with IndexedSlicesValue
      values_out, indices_out, dense_shape_out = s.run([
          ind_values, ind_indices, ind_dense_shape
      ], {ind: indexed_slices.IndexedSlicesValue(values, indices, dense_shape)})
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(indices_out, indices)
      self.assertAllEqual(dense_shape_out, dense_shape)
      # Feed with IndexedSlicesValue, fetch IndexedSlicesValue
      ind2_out = s.run(ind2, {
          ind: indexed_slices.IndexedSlicesValue(values, indices, dense_shape)
      })
      self.assertAllEqual(ind2_out.values, values)
      self.assertAllEqual(ind2_out.indices, indices)
      self.assertAllEqual(ind2_out.dense_shape, dense_shape)

  def testFetchIndexedSlicesWithoutDenseShape(self):
    # [FUNCTION DOC]
    # Same as the indexed slices fetch test but with no dense shape (null); the fetched dense shape
    # must be null, and tuple or value access must work.
    with session.Session() as s:
      indices = np.array([[3, 2, 0], [4, 5, 1]]).astype(np.int64)
      values = np.array([1.0, 2.0]).astype(np.float32)
      dense_shape = None
      ind = indexed_slices.IndexedSlices(
          constant_op.constant(values), constant_op.constant(indices), None)
      # Single fetch, use as tuple
      ind_out = s.run(ind)
      values_out, indices_out, dense_shape_out = ind_out
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(indices_out, indices)
      self.assertAllEqual(dense_shape_out, dense_shape)
      # Single fetch, use as IndexedSlicesValue
      ind_out = s.run(ind)
      self.assertAllEqual(ind_out.values, values)
      self.assertAllEqual(ind_out.indices, indices)
      self.assertAllEqual(ind_out.dense_shape, dense_shape)
      # Tuple fetch, use as tuple
      values_out, indices_out, dense_shape_out = s.run(ind)
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(indices_out, indices)
      self.assertAllEqual(dense_shape_out, dense_shape)
      # List fetch, use as tuple
      (values_out, indices_out, dense_shape_out), = s.run([ind])
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(indices_out, indices)
      self.assertAllEqual(dense_shape_out, dense_shape)
      # List fetch, use as IndexedSlicesValue
      ind_out, = s.run([ind])
      self.assertAllEqual(ind_out.values, values)
      self.assertAllEqual(ind_out.indices, indices)
      self.assertAllEqual(ind_out.dense_shape, dense_shape)

  def testFeedIndexedSlicesWithoutDenseShape(self):
    # [FUNCTION DOC]
    # Same as the indexed slices feed test but without a dense shape: feeding a two-part tuple and an
    # indexed slices value with null dense shape must return the fed values and indices.
    with session.Session() as s:
      values = np.array([1.0, 2.0]).astype(np.float32)
      indices = np.array([[3, 2, 0], [4, 5, 1]]).astype(np.int64)
      dense_shape = None
      ind = indexed_slices.IndexedSlices(
          array_ops.placeholder(dtype=np.float32, shape=(2,)),
          array_ops.placeholder(dtype=np.int64, shape=(2, 3)), None)
      ind_values = array_ops.identity(ind.values)
      ind_indices = array_ops.identity(ind.indices)
      ind2 = indexed_slices.IndexedSlices(ind_values, ind_indices)
      # Feed with tuple
      values_out, indices_out = s.run([ind_values, ind_indices], {
          ind: (values, indices)
      })
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(indices_out, indices)
      # Feed with IndexedSlicesValue
      values_out, indices_out = s.run([ind_values, ind_indices], {
          ind: indexed_slices.IndexedSlicesValue(values, indices, dense_shape)
      })
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(indices_out, indices)
      # Feed with IndexedSlicesValue, fetch IndexedSlicesValue
      ind2_out = s.run(ind2, {
          ind: indexed_slices.IndexedSlicesValue(values, indices, dense_shape)
      })
      self.assertAllEqual(ind2_out.values, values)
      self.assertAllEqual(ind2_out.indices, indices)
      self.assertAllEqual(ind2_out.dense_shape, dense_shape)

  def testExtendWithStatelessOperations(self):
    # [FUNCTION DOC]
    # Runs a matrix product, then adds a new stateless op using it and runs again, so the session graph
    # must be extended after the first run; checks both results.
    with session.Session() as s:
      a = constant_op.constant(1.0, shape=[1, 2])
      b = constant_op.constant(2.0, shape=[2, 3])
      c = math_ops.matmul(a, b)
      c_val = s.run(c)
      self.assertAllEqual([[4.0, 4.0, 4.0]], c_val)
      d = constant_op.constant([1.0, 2.0, 3.0], shape=[3, 1])
      e = math_ops.matmul(c, d)
      # Extend will happen here.
      e_val = s.run(e)
      self.assertAllEqual([[24.0]], e_val)

  def testExtendWithStatefulOperations(self):
    # [FUNCTION DOC]
    # Initializes a variable, then adds new ops (including an assign) after the first run so the session
    # graph is extended. Checks evaluating a new op does not run the assign (variable unchanged) and
    # that running the assign afterwards updates the variable.
    with session.Session() as s:
      a = constant_op.constant(1.0, shape=[1, 2])
      b = constant_op.constant(2.0, shape=[2, 3])
      c = math_ops.matmul(a, b)
      v = variables.Variable(c, name='testExtendWithStatefulOperations_v')
      v.initializer.run()
      v_val = self.evaluate(v)
      self.assertAllEqual([[4.0, 4.0, 4.0]], v_val)
      d = constant_op.constant(3.0, shape=[2, 3])
      e = math_ops.matmul(a, d)
      assign_e_to_v = state_ops.assign(v, e)
      # Extend will happen here.
      e_val = self.evaluate(e)
      self.assertAllEqual([[6.0, 6.0, 6.0]], e_val)
      v_val = self.evaluate(v)
      self.assertAllEqual([[4.0, 4.0, 4.0]], v_val)
      s.run(assign_e_to_v)
      v_val = self.evaluate(v)
      self.assertAllEqual([[6.0, 6.0, 6.0]], v_val)

  def testExtendWithGroupBy(self):
    # [FUNCTION DOC]
    # Creates a variable, forces an extension by evaluating a constant, creates another variable and
    # groups both initializers; running the group and fetching both variables must return their values.
    with session.Session() as s:
      a = constant_op.constant(1.0, shape=[1, 2])
      p = variables.Variable(a, name='testExtendWithGroupBy_p')
      a_val = self.evaluate(a)  # Force an Extend after this op.
      self.assertAllEqual([[1.0, 1.0]], a_val)

      b = constant_op.constant(2.0, shape=[1, 2])
      q = variables.Variable(b, name='testExtendWithGroupBy_q')
      # Extend will happen here.
      init = control_flow_ops.group(p.initializer, q.initializer)
      s.run(init)
      p_val, q_val = s.run([p, q])

      self.assertAllEqual([[1.0, 1.0]], p_val)
      self.assertAllEqual([[2.0, 2.0]], q_val)

  def testTensorGetMethod(self):
    # [FUNCTION DOC]
    # Evaluates a product through evaluate and through the tensor's eval method with a feed dictionary
    # keyed by tensor name, checking the fed value changes the result.
    with session.Session():
      a = constant_op.constant(1.0, shape=[1, 2])
      b = constant_op.constant(2.0, shape=[2, 3])
      c = math_ops.matmul(a, b)

      c_val = self.evaluate(c)
      self.assertAllEqual([[4.0, 4.0, 4.0]], c_val)

      fed_c_val = c.eval(feed_dict={a.name: [[4.0, 4.0]]})
      self.assertAllEqual([[16.0, 16.0, 16.0]], fed_c_val)

  @test_util.run_v1_only('b/120545219')
  def testOperationRunMethod(self):
    # [FUNCTION DOC]
    # Graph-mode only. Creates a variable, assigns constants to it by evaluation, and uses the assign
    # op's eval with a feed keyed by name to assign a fed value; checks the variable after each step.
    with session.Session():
      a = constant_op.constant(1.0, shape=[1, 2])
      b = constant_op.constant(2.0, shape=[1, 2], name='b')
      v = variable_v1.VariableV1(a, a.dtype)
      assign_a_to_v = state_ops.assign(v, a)

      self.evaluate(assign_a_to_v)

      v_val = self.evaluate(v)
      self.assertAllEqual([[1.0, 1.0]], v_val)

      assign_b_to_v = state_ops.assign(v, b)

      self.evaluate(assign_b_to_v)
      v_val = self.evaluate(v)
      self.assertAllEqual([[2.0, 2.0]], v_val)

      assign_b_to_v.eval(feed_dict={'b:0': [[3.0, 3.0]]})
      v_val = self.evaluate(v)
      self.assertAllEqual([[3.0, 3.0]], v_val)

  def testDefaultGraph(self):
    # [FUNCTION DOC]
    # Checks the session's graph is the default graph, and that variable creation, initialization and
    # assignment work using the default graph; unrelated assign ops do not take effect until run.
    with session.Session() as s:
      self.assertEqual(ops.get_default_graph(), s.graph)
      a = constant_op.constant(1.0, shape=[1, 2])
      b = constant_op.constant(2.0, shape=[2, 3])
      self.assertEqual(ops.get_default_graph(), a.graph)
      self.assertEqual(ops.get_default_graph(), b.graph)
      c = math_ops.matmul(a, b)
      v = variables.Variable(c, name='testDefaultGraph_v')
      v.initializer.run()
      v_val = self.evaluate(v)
      self.assertAllEqual([[4.0, 4.0, 4.0]], v_val)
      d = constant_op.constant(3.0, shape=[2, 3])
      e = math_ops.matmul(a, d)
      assign_e_to_v = state_ops.assign(v, e)
      e_val = self.evaluate(e)
      self.assertAllEqual([[6.0, 6.0, 6.0]], e_val)
      v_val = self.evaluate(v)
      self.assertAllEqual([[4.0, 4.0, 4.0]], v_val)
      s.run(assign_e_to_v)
      v_val = self.evaluate(v)
      self.assertAllEqual([[6.0, 6.0, 6.0]], v_val)
      self.assertEqual(ops.get_default_graph(), s.graph)

  def _testDefaultGraphInThread(self, constructed_event, continue_event, i):
    # [FUNCTION DOC]
    # HELPER: worker body for the thread-local default graph test.
    # Inputs: two events (one set when this thread has built its graph, one to wait on before
    # continuing) and an integer index used to name a variable. In a new session, builds a product and a
    # variable, signals construction, waits for the continue event, then initializes, assigns and
    # checks values, finally checking the default graph is still the session graph. Returns nothing.
    with session.Session() as s:
      self.assertEqual(ops.get_default_graph(), s.graph)
      a = constant_op.constant(1.0, shape=[1, 2])
      b = constant_op.constant(2.0, shape=[2, 3])
      c = math_ops.matmul(a, b)
      v = variables.Variable(c, name='var_%d' % i)

      # Block here until all threads have constructed their graph.
      constructed_event.set()
      continue_event.wait()

      assign_c_to_v = state_ops.assign(v, c)
      v.initializer.run()
      self.evaluate(assign_c_to_v)
      v_val = self.evaluate(v)
      self.assertAllEqual([[4.0, 4.0, 4.0]], v_val)
      d = constant_op.constant(3.0, shape=[2, 3])
      e = math_ops.matmul(a, d)
      assign_e_to_v = state_ops.assign(v, e)
      e_val = self.evaluate(e)
      self.assertAllEqual([[6.0, 6.0, 6.0]], e_val)
      v_val = self.evaluate(v)
      self.assertAllEqual([[4.0, 4.0, 4.0]], v_val)
      s.run(assign_e_to_v)
      v_val = self.evaluate(v)
      self.assertAllEqual([[6.0, 6.0, 6.0]], v_val)
      self.assertEqual(ops.get_default_graph(), s.graph)

  def testDefaultGraphWithThreads(self):
    # [FUNCTION DOC]
    # Starts ten checked threads that each run the worker body with their own thread-local default
    # graph. All threads build their graphs, are released together, and are joined; assertion failures in
    # threads are surfaced by the checked threads.
    # Fork ten threads that use their thread-local default graph.
    threads = []
    constructed_events = [threading.Event() for _ in range(10)]
    continue_event = threading.Event()
    for i, constructed_event in enumerate(constructed_events):
      t = self.checkedThread(
          target=self._testDefaultGraphInThread,
          args=(constructed_event, continue_event, i))
      threads.append(t)
    for t in threads:
      t.start()
    for constructed_event in constructed_events:
      constructed_event.wait()
    continue_event.set()
    for t in threads:
      t.join()

  def testParallelRun(self):
    # [FUNCTION DOC]
    # Starts 100 threads that all wait on an event and then each evaluate the same constant in one
    # shared session. The event is set and the threads are joined; each must see the value 5.
    with session.Session() as sess:
      c = constant_op.constant(5.0)
      ev = threading.Event()

      def run_step():
        # [FUNCTION DOC]
        # HELPER: thread body for the parallel run test. Waits for the start event, evaluates the constant
        # in the shared session and asserts it equals 5. Returns nothing.
        ev.wait()
        val = c.eval(session=sess)
        self.assertEqual(val, 5.0)

      threads = [self.checkedThread(target=run_step) for _ in range(100)]
      for t in threads:
        t.start()
      ev.set()
      for t in threads:
        t.join()

  @staticmethod
  def _build_graph():
    # [FUNCTION DOC]
    # HELPER: graph-construction workload for concurrency tests (static method, no arguments).
    # Sleeps a random short time, then repeatedly (ten times) builds placeholders with colocation, a
    # while loop on the CPU device, gradients inside an attribute scope, and either serializes the graph
    # or imports the previously serialized graph. Returns nothing. Randomness and sleeping are used to
    # vary thread interleavings.
    time.sleep(random.random() * 0.1)
    # Do some graph construction. Try to exercise non-trivial paths.
    graph = ops.get_default_graph()
    gdef = None
    for _ in range(10):
      x = array_ops.placeholder(dtype=dtypes.float32)
      with ops.colocate_with(x):
        y = array_ops.placeholder(dtype=dtypes.float32)
      with ops.device('/cpu:0'):
        z = while_loop.while_loop(
            lambda x, y: x < 10, lambda x, y: (x + 1, x * y), [x, y])
      with graph._attr_scope({'_a': attr_value_pb2.AttrValue(b=False)}):
        gradients_impl.gradients(z, [x, y])
        if gdef is None:
          gdef = graph.as_graph_def()
        else:
          importer.import_graph_def(gdef, name='import')

  @test_util.run_v1_only('b/120545219')
  def testParallelRunAndSingleBuild(self):
    # [FUNCTION DOC]
    # Graph-mode only. Ten threads repeatedly run a constant in a shared session while the main thread
    # builds graph content with the build helper; then the threads are stopped and joined. Checks runs
    # keep returning 5 while the graph is being built.
    with session.Session() as sess:
      c = constant_op.constant(5.0)
      stop = threading.Event()

      def run_loop():
        # [FUNCTION DOC]
        # HELPER: thread body. Until the stop event is set, sleeps a random short time and asserts running
        # the constant in the shared session returns 5. Returns nothing.
        while not stop.is_set():
          time.sleep(random.random() * 0.1)
          self.assertEqual(sess.run(c), 5.0)

      threads = [self.checkedThread(target=run_loop) for _ in range(10)]
      for t in threads:
        t.start()

      SessionTest._build_graph()

      stop.set()
      for t in threads:
        t.join()

  @test_util.run_v1_only('b/120545219')
  def testParallelRunAndParallelBuild(self):
    # [FUNCTION DOC]
    # Graph-mode only. Like the single-build test, but ten threads each run the graph builder in
    # parallel with ten running threads; build threads are joined first, then runners are stopped.
    with session.Session() as sess:
      c = constant_op.constant(5.0)
      stop = threading.Event()

      def run_loop():
        # [FUNCTION DOC]
        # HELPER: thread body. Until the stop event is set, sleeps a random short time and asserts running
        # the constant in the shared session returns 5. Returns nothing.
        while not stop.is_set():
          time.sleep(random.random() * 0.1)
          self.assertEqual(sess.run(c), 5.0)

      run_threads = [self.checkedThread(target=run_loop) for _ in range(10)]
      for t in run_threads:
        t.start()

      build_threads = [self.checkedThread(target=SessionTest._build_graph)
                       for _ in range(10)]
      for t in build_threads:
        t.start()
      for t in build_threads:
        t.join()

      # Let the run_threads run until the build threads are finished.
      stop.set()
      for t in run_threads:
        t.join()

  def testRunFeedDict(self):
    # [FUNCTION DOC]
    # Feeds a tensor by object and by name, with array and plain list values, and checks results.
    # Also feeds a nested tuple of tensors with a matching nested tuple of arrays as one key and checks
    # each part is doubled correctly.
    with session.Session() as s:
      x = array_ops.zeros([2])

      y = s.run(2 * x, feed_dict={x: np.ones(2).astype(np.float32)})
      self.assertAllEqual(y, 2 * np.ones(2))

      y = s.run(2 * x, feed_dict={x.name: np.ones(2).astype(np.float32)})
      self.assertAllEqual(y, 2 * np.ones(2))

      y = s.run(2 * x, feed_dict={x: [1, 1]})
      assert (y == 2 * np.ones(2)).all()

      # Test nested tuple keys
      z = (((array_ops.zeros([2]),),), array_ops.zeros([2]),
           (array_ops.zeros([2]),))
      result = [z[0][0][0] * 2, z[1] * 2, z[2][0] * 2]
      values = (((np.array([1, 1]),),), np.array([2, 2]), (np.array([3, 3]),))
      result_value = s.run(result, feed_dict={z: values})
      self.assertAllEqual(result_value[0], 2 * np.ones(2))
      self.assertAllEqual(result_value[1], 2 * np.array([2, 2]))
      self.assertAllEqual(result_value[2], 2 * np.array([3, 3]))

  def testGraphDef(self):
    # [FUNCTION DOC]
    # Checks the session's graph definition initially contains only the version information, and that
    # its node count increases as constants are added, without any run in between.
    with session.Session() as sess:
      self.assertProtoEquals('versions { producer: %d min_consumer: %d }' %
                             (versions.GRAPH_DEF_VERSION,
                              versions.GRAPH_DEF_VERSION_MIN_CONSUMER),
                             sess.graph_def)
      c = constant_op.constant(5.0, name='c')
      self.assertEqual(len(sess.graph_def.node), 1)
      d = constant_op.constant(6.0, name='d')
      self.assertEqual(len(sess.graph_def.node), 2)
      self.assertAllEqual(c, 5.0)
      self.assertAllEqual(d, 6.0)
      e = constant_op.constant(7.0, name='e')
      self.assertEqual(len(sess.graph_def.node), 3)
      self.assertAllEqual(e, 7.0)

  def testUseAfterClose(self):
    # [FUNCTION DOC]
    # Closes a session after one successful run and checks that running again raises a runtime error
    # saying a closed session was used.
    with session.Session() as sess:
      c = constant_op.constant(5.0)
      self.assertAllEqual(sess.run(c), 5.0)
    with self.assertRaisesWithPredicateMatch(
        RuntimeError, lambda e: 'Attempted to use a closed Session.' in str(e)):
      sess.run(c)

  def testUseAfterCloseConcurrent(self):
    # [FUNCTION DOC]
    # Starts a thread that keeps running a constant in a session until a runtime error mentioning a
    # closed session occurs; the main thread closes the session after a short wait and joins the thread.
    with session.Session() as sess:
      c = constant_op.constant(5.0)
      self.assertAllEqual(sess.run(c), 5.0)

      def update_thread():
        # [FUNCTION DOC]
        # HELPER: thread body for the close-concurrency test. Loops running the constant forever inside an
        # assertion expecting a closed-session runtime error, so it ends only when the session is closed.
        with self.assertRaisesWithPredicateMatch(
            RuntimeError,
            lambda e: 'Attempted to use a closed Session.' in str(e)):
          while True:
            sess.run(c)

      t = threading.Thread(target=update_thread)
      t.start()
      time.sleep(0.1)
      sess.close()
      t.join()

  def testUseEmptyGraph(self):
    # [FUNCTION DOC]
    # Checks that running an empty list, tuple or mapping on a session whose graph is empty raises a
    # runtime error saying the graph is empty.
    with session.Session() as sess:
      with self.assertRaisesRegex(RuntimeError, 'The Session graph is empty.'):
        sess.run([])
      with self.assertRaisesRegex(RuntimeError, 'The Session graph is empty.'):
        sess.run(())
      with self.assertRaisesRegex(RuntimeError, 'The Session graph is empty.'):
        sess.run({})

  @test_util.run_v1_only('b/120545219')
  def testNotEntered(self):
    # [FUNCTION DOC]
    # Graph-mode only. Checks no default session is registered, creates a session without entering it,
    # runs a constant through it, and checks that evaluating a tensor without a session raises a value
    # error stating no default session is registered.
    # pylint: disable=protected-access
    self.assertIsNone(stack._default_session_stack.get_default())
    # pylint: enable=protected-access
    with ops.device('/cpu:0'):
      sess = session.Session()
      c_1 = constant_op.constant(5.0)
      with sess.graph.as_default():
        c_2 = constant_op.constant(5.0)
      self.assertEqual(c_1.graph, c_2.graph)
      self.assertEqual(sess.run(c_2), 5.0)
      with self.assertRaisesWithPredicateMatch(
          ValueError, lambda e: 'No default session is registered.' in str(e)):
        c_2.eval()

  @test_util.run_v1_only('b/120545219')
  def testInteractive(self):
    # [FUNCTION DOC]
    # Graph-mode only. Creates an interactive session (which becomes the default) and checks tensors
    # can be compared directly via implicit evaluation; closes the session.
    with ops.device('/cpu:0'):
      sess = session.InteractiveSession()
      a = constant_op.constant(1.0, shape=[1, 2])
      b = constant_op.constant(2.0, shape=[2, 3])
      c = math_ops.matmul(a, b)
      self.assertAllEqual([[4.0, 4.0, 4.0]], c)
      d = constant_op.constant([1.0, 2.0, 3.0], shape=[3, 1])
      e = math_ops.matmul(c, d)
      self.assertAllEqual([[24.0]], e)
      sess.close()

  @test_util.run_v1_only('b/120545219')
  def testMultipleInteractiveSessionsError(self):
    # [FUNCTION DOC]
    # Graph-mode only. Resets the class-level count of active interactive sessions, then checks that
    # opening and closing sessions serially logs no error, opening one session while none is active logs
    # none, and opening a second while one is active logs exactly one error containing the
    # warning about an already-active interactive session. Closes both sessions.
    # Reinitialize the global state to ensure that the expected warnings will
    # be emitted.
    session.InteractiveSession._active_session_count = 0  # pylint: disable=protected-access

    sess = session.InteractiveSession()
    sess.run(constant_op.constant(4.0))  # Run so that the session is "opened".
    sess.close()
    # Opening and closing interactive sessions serially should not warn.
    with self.assertNoLogs(level='ERROR'):
      sess = session.InteractiveSession()
      sess.close()

    with self.assertNoLogs(level='ERROR'):
      sess = session.InteractiveSession()

    with self.assertLogs(level='ERROR') as log_output:
      sess2 = session.InteractiveSession()

    self.assertLen(log_output.output, 1)

    self.assertIn(
        'An interactive session is already active. This can cause'
        ' out-of-memory errors or some other unexpected errors (due to'
        ' the unpredictable timing of garbage collection) in some cases.'
        ' You must explicitly call `InteractiveSession.close()` to release'
        ' resources held by the other session(s). Please use `tf.Session()`'
        ' if you intend to productionize.',
        log_output.output[0],
    )
    sess2.close()
    sess.close()

  @test_util.run_v1_only('b/120545219')
  def testInteractivePlacePrunedGraph(self):
    # [FUNCTION DOC]
    # Graph-mode only. With an interactive session, builds an op pinned to a GPU device that has no
    # kernel plus a valid op. Running the valid op works (the graph is pruned before placement), while
    # running the invalid op raises an invalid-argument error. Closes the session.
    sess = session.InteractiveSession()

    # Build a graph that has a bad op in it (no kernel).
    #
    # This test currently does not link in any GPU kernels,
    # which is why placing this is invalid.  If at some point
    # GPU kernels are added to this test, some other different
    # op / device combo should be chosen.
    with ops.device('/device:GPU:0'):
      a = constant_op.constant(1.0, shape=[1, 2])

    b = constant_op.constant(1.0, shape=[1, 2])

    # Only run the valid op, this should work.
    self.evaluate(b)

    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(a)
    sess.close()

  @test_util.run_v1_only('b/120545219')
  def testDefaultSessionPlacePrunedGraph(self):
    # [FUNCTION DOC]
    # Graph-mode only. With a regular session, builds an op pinned to an unsupported GPU device and a
    # valid op. Running only the valid op still raises an invalid-argument error because the whole graph
    # is placed. Closes the session.
    sess = session.Session()

    # Build a graph that has a bad op in it (no kernel).
    #
    # This test currently does not link in any GPU kernels,
    # which is why placing this is invalid.  If at some point
    # GPU kernels are added to this test, some other different
    # op / device combo should be chosen.
    with ops.device('/device:GPU:0'):
      _ = constant_op.constant(1.0, shape=[1, 2])

    b = constant_op.constant(1.0, shape=[1, 2])

    with self.assertRaises(errors.InvalidArgumentError):
      # Even though we don't run the bad op, we place the entire
      # graph, which should fail with a non-interactive session.
      sess.run(b)

    sess.close()

  def testSharedGraph(self):
    # [FUNCTION DOC]
    # Creates two sessions over the same graph and checks they return equal results for the same fetch.
    with ops.Graph().as_default() as g, ops.device('/cpu:0'):
      a = constant_op.constant(1.0, shape=[1, 2])
      b = constant_op.constant(2.0, shape=[2, 3])
      c = math_ops.matmul(a, b)

    with session.Session(graph=g) as sess1:
      with session.Session(graph=g) as sess2:
        self.assertAllEqual(sess1.run(c), sess2.run(c))

  def testDuplicatedInputs(self):
    # [FUNCTION DOC]
    # Fetches a list in which the same tensor appears twice and checks all three results are correct.
    with session.Session() as sess:
      a = constant_op.constant(1.0, shape=[1, 2])
      b = constant_op.constant(2.0, shape=[1, 3])
      a_val, b_val, a2_val = sess.run([a, b, a])
      self.assertAllEqual(a_val, [[1.0, 1.0]])
      self.assertAllEqual(b_val, [[2.0, 2.0, 2.0]])
      self.assertAllEqual(a2_val, [[1.0, 1.0]])

  def testFeedAndFetch(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the chain of dtype branches tests for complex64 twice; the second branch is dead
    # code and complex128 is never handled specifically (it falls to the plain cast).
    # For many dtypes and shapes (including empty shapes), feeds a random array through a placeholder
    # and identity, then checks fetching the output, the fed placeholder, both together and through a
    # callable with a feed list all return the fed array. Booleans and complex types get special input
    # preparation.
    with session.Session() as sess:
      for dtype in [
          dtypes.float16, dtypes.float32, dtypes.float64, dtypes.int32,
          dtypes.uint8, dtypes.int16, dtypes.int8, dtypes.int64, dtypes.bool,
          dtypes.complex64, dtypes.complex128
      ]:
        for shape in [(32, 4, 128), (37,), (2, 0, 6), (0, 0, 0)]:
          np_dtype = dtype.as_numpy_dtype

          feed_t = array_ops.placeholder(dtype=dtype, shape=shape)
          out_t = array_ops.identity(feed_t)

          np_array = np.random.randint(-10, 10, shape)

          if dtype == dtypes.bool:
            np_array = np_array > 0
          elif dtype == dtypes.complex64:
            np_array = np.sqrt(np_array.astype(np_dtype))
          elif dtype == dtypes.complex64:
            np_array = np.sqrt(np_array.astype(np_dtype))
          else:
            np_array = np_array.astype(np_dtype)

          self.assertAllEqual(np_array,
                              sess.run(out_t, feed_dict={
                                  feed_t: np_array
                              }))
          # Check that we can also get the feed back.
          self.assertAllEqual(np_array,
                              sess.run(feed_t, feed_dict={
                                  feed_t: np_array
                              }))
          # Also check that we can get both back.
          out_v, feed_v = sess.run(
              [out_t, feed_t], feed_dict={
                  feed_t: np_array
              })
          self.assertAllEqual(np_array, out_v)
          self.assertAllEqual(np_array, feed_v)

          feed_fetch_runner = sess.make_callable([out_t, feed_t], [feed_t])
          out_v, feed_v = feed_fetch_runner(np_array)
          self.assertAllEqual(np_array, out_v)
          self.assertAllEqual(np_array, feed_v)

  def testMakeCallableOnTensorWithRunOptions(self):
    # [FUNCTION DOC]
    # Makes a callable for a tensor that accepts options, calls it with full tracing and a metadata
    # object, and checks the value and that device statistics were recorded (none before the call).
    with session.Session() as sess:
      a = constant_op.constant(42.0)
      tensor_runner = sess.make_callable(a, accept_options=True)
      run_options = config_pb2.RunOptions(
          trace_level=config_pb2.RunOptions.FULL_TRACE)
      run_metadata = config_pb2.RunMetadata()
      self.assertEqual(0, len(run_metadata.step_stats.dev_stats))
      res = tensor_runner(options=run_options, run_metadata=run_metadata)
      self.assertEqual(42.0, res)
      self.assertGreater(len(run_metadata.step_stats.dev_stats), 0)

  def testMakeCallableOnOperationWithRunOptions(self):
    # [FUNCTION DOC]
    # Makes an options-accepting callable for an increment operation of an initialized variable, calls
    # it with full tracing, and checks the variable incremented and device stats were recorded.
    with session.Session() as sess:
      a = variables.Variable(42.0)
      b = state_ops.assign_add(a, 1.0)
      sess.run(a.initializer)
      tensor_runner = sess.make_callable(b.op, accept_options=True)
      run_options = config_pb2.RunOptions(
          trace_level=config_pb2.RunOptions.FULL_TRACE)
      run_metadata = config_pb2.RunMetadata()
      self.assertEqual(0, len(run_metadata.step_stats.dev_stats))
      tensor_runner(options=run_options, run_metadata=run_metadata)
      self.assertEqual(43.0, sess.run(a))
      self.assertGreater(len(run_metadata.step_stats.dev_stats), 0)

  def testMakeCallableWithFeedListAndRunOptions(self):
    # [FUNCTION DOC]
    # Makes a callable with a feed list naming a placeholder and options enabled, calls it with a fed
    # value, tracing and metadata; checks the incremented result and that device stats were recorded.
    with session.Session() as sess:
      ph = array_ops.placeholder(dtypes.float32)
      a = math_ops.add(ph, 1.0)
      tensor_runner = sess.make_callable(
          a, feed_list=[ph.name], accept_options=True)
      run_options = config_pb2.RunOptions(
          trace_level=config_pb2.RunOptions.FULL_TRACE)
      run_metadata = config_pb2.RunMetadata()
      self.assertEqual(0, len(run_metadata.step_stats.dev_stats))
      self.assertAllClose(42.0,
                          tensor_runner(
                              41.0,
                              options=run_options,
                              run_metadata=run_metadata))
      self.assertGreater(len(run_metadata.step_stats.dev_stats), 0)

  def testOptimizedMakeCallable(self):
    # [FUNCTION DOC]
    # Builds callable options (one feed and one fetch) and creates a callable from them through the
    # session's private method, three times, calling each five times; every call must return the list
    # with the incremented value.
    with session.Session() as sess:
      ph = array_ops.placeholder(dtypes.float32)
      a = math_ops.add(ph, 1.0)
      callable_opts = config_pb2.CallableOptions()
      callable_opts.feed.append(ph.name)
      callable_opts.fetch.append(a.name)
      for _ in range(3):
        callable_fn = sess._make_callable_from_options(callable_opts)
        for _ in range(5):
          self.assertEqual([2.0], callable_fn(np.array(1.0, dtype=np.float32)))

  def testOptimizedMakeCallableWithRunMetadata(self):
    # [FUNCTION DOC]
    # Like the optimized callable test but with full tracing in the callable options and a metadata
    # object passed at call time; checks the result and that device stats were recorded.
    with session.Session() as sess:
      ph = array_ops.placeholder(dtypes.float32)
      a = math_ops.add(ph, 1.0)
      callable_opts = config_pb2.CallableOptions()
      callable_opts.feed.append(ph.name)
      callable_opts.fetch.append(a.name)
      callable_opts.run_options.trace_level = config_pb2.RunOptions.FULL_TRACE
      callable_fn = sess._make_callable_from_options(callable_opts)
      run_metadata = config_pb2.RunMetadata()
      self.assertEqual([2.0], callable_fn(np.array(1.0, dtype=np.float32),
                                          run_metadata=run_metadata))
      self.assertGreater(len(run_metadata.step_stats.dev_stats), 0)

  def testFeedError(self):
    # [FUNCTION DOC]
    # Checks that using a tensor object as a feed value raises a type error (message says it cannot be a
    # tensor) via session run, tensor eval and operation run.
    with session.Session() as sess:
      feed_t = array_ops.placeholder(dtype=dtypes.float32)
      out_t = array_ops.identity(feed_t)
      feed_val = constant_op.constant(5.0)
      with self.assertRaisesRegex(TypeError, 'cannot be a tf.Tensor object'):
        sess.run(out_t, feed_dict={feed_t: feed_val})
      with self.assertRaisesRegex(TypeError, 'cannot be a tf.Tensor object'):
        out_t.eval(feed_dict={feed_t: feed_val})
      with self.assertRaisesRegex(TypeError, 'cannot be a tf.Tensor object'):
        out_t.op.run(feed_dict={feed_t: feed_val})

  def testFeedPrecisionLossError(self):
    # [FUNCTION DOC]
    # Feeds the largest 64-bit integer into integer 32-bit constants (implicit and explicit); each run
    # must raise a type error saying the value is not compatible with the tensor type.
    with session.Session() as sess:
      largest_int64 = np.iinfo(np.int64).max

      feed_int_implicit_int32 = constant_op.constant(1)
      feed_int_explicit_int32 = constant_op.constant(1, dtype=dtypes.int32)

      out_t = constant_op.constant(1.0)

      with self.assertRaisesRegex(TypeError,
                                  'is not compatible with Tensor type'):
        sess.run(out_t, feed_dict={feed_int_implicit_int32: largest_int64})
      with self.assertRaisesRegex(TypeError,
                                  'is not compatible with Tensor type'):
        sess.run(out_t, feed_dict={feed_int_explicit_int32: largest_int64})

  def testStringFetch(self):
    # [FUNCTION DOC]
    # For several shapes (including empty), builds an array of byte strings, makes a constant from it
    # and checks the evaluated constant equals the array. Despite the name, no session fetch beyond
    # evaluation of the constant is done.
    with session.Session():
      for shape in [(32, 4, 128), (37,), (2, 0, 6), (0, 0, 0)]:
        size = 1
        for s in shape:
          size *= s
        c_list = np.array([compat.as_bytes(str(i)) for i in range(size)],
                          dtype=np.object_).reshape(shape) if size > 0 else []
        c = constant_op.constant(c_list)
        self.assertAllEqual(c, c_list)

  def testStringFeed(self):
    # [FUNCTION DOC]
    # For several shapes, feeds arrays of byte strings through a string placeholder and identity, and
    # checks fetching the output, the placeholder, or both returns the fed array.
    with session.Session() as sess:
      for shape in [(32, 4, 128), (37,), (2, 0, 6), (0, 0, 0)]:
        size = 1
        for s in shape:
          size *= s
        c_list = np.array([compat.as_bytes(str(i)) for i in range(size)],
                          dtype=np.object_).reshape(shape)
        feed_t = array_ops.placeholder(dtype=dtypes.string, shape=shape)
        c = array_ops.identity(feed_t)
        self.assertAllEqual(sess.run(c, feed_dict={feed_t: c_list}), c_list)
        self.assertAllEqual(
            sess.run(feed_t, feed_dict={
                feed_t: c_list
            }), c_list)
        c_v, feed_v = sess.run([c, feed_t], feed_dict={feed_t: c_list})
        self.assertAllEqual(c_v, c_list)
        self.assertAllEqual(feed_v, c_list)

  def testStringFeedWithNullCharacters(self):
    # [FUNCTION DOC]
    # Feeds byte strings containing null characters and checks they are returned unchanged.
    with session.Session():
      c_list = [b'\n\x01\x00', b'\n\x00\x01']
      feed_t = array_ops.placeholder(dtype=dtypes.string, shape=[2])
      c = array_ops.identity(feed_t)
      out = c.eval(feed_dict={feed_t: c_list})
      self.assertEqual(c_list[0], out[0])
      self.assertEqual(c_list[1], out[1])

  def testStringFeedWithUnicode(self):
    # [FUNCTION DOC]
    # Feeds text strings with special and non-ASCII characters, as a list and as an object array, and
    # checks the decoded results match the originals.
    with session.Session():
      c_list = [
          u'\n\x01\x00', u'\n\x00\x01', u'\u26a3 unicode',
          u'\U0001f60e deal with it'
      ]
      feed_t = array_ops.placeholder(dtype=dtypes.string, shape=[len(c_list)])
      c = array_ops.identity(feed_t)

      out = c.eval(feed_dict={feed_t: c_list})
      for i in range(len(c_list)):
        self.assertEqual(c_list[i], out[i].decode('utf-8'))

      out = c.eval(feed_dict={feed_t: np.array(c_list, dtype=np.object_)})
      for i in range(len(c_list)):
        self.assertEqual(c_list[i], out[i].decode('utf-8'))

  def testInvalidTargetFails(self):
    # [FUNCTION DOC]
    # Checks creating a session with an unknown target string raises a not-found error saying no
    # session factory is registered for the options.
    with self.assertRaisesRegex(
        errors.NotFoundError,
        'No session factory registered for the given session options'):
      session.Session('INVALID_TARGET')

  def testFetchByNameDifferentStringTypes(self):
    # [FUNCTION DOC]
    # Creates constants named using plain, unicode, byte and raw strings, checks every name is a text
    # string, and fetches each by names given in all four string types, expecting the right values.
    with session.Session() as sess:
      c = constant_op.constant(42.0, name='c')
      d = constant_op.constant(43.0, name=u'd')
      e = constant_op.constant(44.0, name=b'e')
      f = constant_op.constant(45.0, name=r'f')

      self.assertIsInstance(c.name, six.text_type)
      self.assertIsInstance(d.name, six.text_type)
      self.assertIsInstance(e.name, six.text_type)
      self.assertIsInstance(f.name, six.text_type)

      self.assertEqual(42.0, sess.run('c:0'))
      self.assertEqual(42.0, sess.run(u'c:0'))
      self.assertEqual(42.0, sess.run(b'c:0'))
      self.assertEqual(42.0, sess.run(r'c:0'))

      self.assertEqual(43.0, sess.run('d:0'))
      self.assertEqual(43.0, sess.run(u'd:0'))
      self.assertEqual(43.0, sess.run(b'd:0'))
      self.assertEqual(43.0, sess.run(r'd:0'))

      self.assertEqual(44.0, sess.run('e:0'))
      self.assertEqual(44.0, sess.run(u'e:0'))
      self.assertEqual(44.0, sess.run(b'e:0'))
      self.assertEqual(44.0, sess.run(r'e:0'))

      self.assertEqual(45.0, sess.run('f:0'))
      self.assertEqual(45.0, sess.run(u'f:0'))
      self.assertEqual(45.0, sess.run(b'f:0'))
      self.assertEqual(45.0, sess.run(r'f:0'))

  def testIncorrectGraph(self):
    # [FUNCTION DOC]
    # Builds two graphs with identically named constants and sessions bound to each. Running a tensor
    # or op from the other graph must raise a value error, while own tensors run correctly.
    with ops.Graph().as_default() as g_1:
      c_1 = constant_op.constant(1.0, name='c')

    with ops.Graph().as_default() as g_2:
      c_2 = constant_op.constant(2.0, name='c')

    self.assertEqual('c', c_1.op.name)
    self.assertEqual('c', c_2.op.name)

    with session.Session(graph=g_1) as sess_1:
      self.assertEqual(1.0, sess_1.run(c_1))
      with self.assertRaises(ValueError):
        sess_1.run(c_2)
      with self.assertRaises(ValueError):
        sess_1.run(c_2.op)

    with session.Session(graph=g_2) as sess_2:
      with self.assertRaises(ValueError):
        sess_2.run(c_1)
      with self.assertRaises(ValueError):
        sess_2.run(c_1.op)
      self.assertEqual(2.0, sess_2.run(c_2))

  def testFeedDictKeyException(self):
    # [FUNCTION DOC]
    # Checks that a feed mapping whose key is a plain string raises a type error saying the feed
    # mapping cannot be interpreted.
    with session.Session() as sess:
      a = constant_op.constant(1.0, dtypes.float32, name='a')
      with self.assertRaisesRegex(TypeError, 'Cannot interpret feed_dict'):
        sess.run(a, feed_dict={'a': [2.0]})

  def testPerStepTrace(self):
    # [FUNCTION DOC]
    # Runs with no options or metadata (no step stats recorded), with only metadata (still none), and
    # with software tracing options plus metadata; checks step stats exist with one device entry.
    run_options = config_pb2.RunOptions(
        trace_level=config_pb2.RunOptions.SOFTWARE_TRACE)
    run_metadata = config_pb2.RunMetadata()

    with ops.device('/cpu:0'):
      with session.Session() as sess:
        sess.run(constant_op.constant(1.0))
        self.assertFalse(run_metadata.HasField('step_stats'))

        sess.run(constant_op.constant(1.0), run_metadata=run_metadata)
        self.assertFalse(run_metadata.HasField('step_stats'))

        sess.run(
            constant_op.constant(1.0),
            options=run_options,
            run_metadata=run_metadata)

        self.assertTrue(run_metadata.HasField('step_stats'))
        self.assertEqual(len(run_metadata.step_stats.dev_stats), 1)

  def testRunOptionsRunMetadata(self):
    # [FUNCTION DOC]
    # Runs with all combinations of options and metadata being null or set. Metadata gets step
    # statistics only when both tracing options and metadata are given (one device entry expected).
    run_options = config_pb2.RunOptions(
        trace_level=config_pb2.RunOptions.SOFTWARE_TRACE)
    run_metadata = config_pb2.RunMetadata()

    with ops.device('/cpu:0'):
      with session.Session() as sess:
        # all combinations are valid
        sess.run(constant_op.constant(1.0), options=None, run_metadata=None)
        sess.run(
            constant_op.constant(1.0), options=None, run_metadata=run_metadata)
        self.assertFalse(run_metadata.HasField('step_stats'))

        sess.run(
            constant_op.constant(1.0), options=run_options, run_metadata=None)
        self.assertFalse(run_metadata.HasField('step_stats'))

        sess.run(
            constant_op.constant(1.0),
            options=run_options,
            run_metadata=run_metadata)

        self.assertTrue(run_metadata.HasField('step_stats'))
        self.assertEqual(len(run_metadata.step_stats.dev_stats), 1)

  def testFeedShapeCompatibility(self):
    # [FUNCTION DOC]
    # Checks feeding a value with an incompatible shape raises a value error, and that feeding a
    # shape tensor causing a reshape mismatch raises an invalid-argument error with the expected text.
    with session.Session() as sess:
      some_tensor = constant_op.constant([2.0, 2.0, 2.0, 2.0])
      new_shape = constant_op.constant([2, 2])
      reshaped_tensor = array_ops.reshape(some_tensor, new_shape)

      with self.assertRaisesRegex(ValueError, 'Cannot feed value of shape'):
        sess.run(reshaped_tensor, feed_dict={some_tensor: [1.0, 2.0, 3.0]})

      with self.assertRaisesRegex(
          errors.InvalidArgumentError,
          'Input to reshape is a tensor with 4 values, '
          'but the requested shape has 21'):
        sess.run(reshaped_tensor, feed_dict={new_shape: [3, 7]})

  def testInferShapesFalse(self):
    # [FUNCTION DOC]
    # Checks that by default the session's graph definition does not carry output-shape attributes.
    with ops.Graph().as_default(), ops.device('/cpu:0'):
      a = constant_op.constant([[1, 2]])
      sess = session.Session()
      self.assertNotIn('_output_shapes', sess.graph_def.node[0].attr)
      # Avoid lint error regarding 'unused' var a.
      self.assertEqual(a, a)

  def testInferShapesTrue(self):
    # [FUNCTION DOC]
    # Creates a session with the infer shapes graph option and checks the graph definition does carry
    # the output-shape attribute.
    config_pb = config_pb2.ConfigProto(
        graph_options=config_pb2.GraphOptions(infer_shapes=True))
    with ops.Graph().as_default(), ops.device('/cpu:0'):
      a = constant_op.constant([[1, 2]])
      sess = session.Session(config=config_pb)
      self.assertIn('_output_shapes', sess.graph_def.node[0].attr)
      # Avoid lint error regarding 'unused' var a.
      self.assertEqual(a, a)

  def testBuildCostModel(self):
    # [FUNCTION DOC]
    # With a cost model configured to build at step 100 and soft placement allowed, runs a small graph
    # 120 times; the cost graph must appear in metadata only at step index 99.
    run_options = config_pb2.RunOptions()
    config_pb = config_pb2.ConfigProto(
        allow_soft_placement=True,
        graph_options=config_pb2.GraphOptions(build_cost_model=100))
    with session.Session(config=config_pb) as sess:
      with ops.device('/device:GPU:0'):
        a = array_ops.placeholder(dtypes.float32, shape=[])
        b = math_ops.add(a, a)
        c = array_ops.identity(b)
        d = math_ops.multiply(c, c)
      for step in range(120):
        run_metadata = config_pb2.RunMetadata()
        sess.run(
            d,
            feed_dict={a: 1.0},
            options=run_options,
            run_metadata=run_metadata)
        if step == 99:
          self.assertTrue(run_metadata.HasField('cost_graph'))
        else:
          self.assertFalse(run_metadata.HasField('cost_graph'))

  def runTestOutputPartitionGraphs(self, sess):
    # [FUNCTION DOC]
    # HELPER: shared body for the partition graphs tests, taking a session. Runs a constant with the
    # option to output partition graphs and checks the metadata has some; running again without that
    # option must give none. Returns nothing.
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    a = constant_op.constant(1)
    run_metadata = config_pb2.RunMetadata()
    sess.run(a, options=run_options, run_metadata=run_metadata)
    self.assertGreater(len(run_metadata.partition_graphs), 0)
    sess.run(a, run_metadata=run_metadata)
    self.assertEqual(len(run_metadata.partition_graphs), 0)

  @test_util.run_v1_only('b/120545219')
  def testOutputPartitionGraphsDirect(self):
    # [FUNCTION DOC]
    # Graph-mode only. Runs the partition graphs check on a plain in-process session.
    self.runTestOutputPartitionGraphs(session.Session())

  @test_util.run_v1_only('b/120545219')
  def testOutputPartitionGraphsDistributed(self):
    # [FUNCTION DOC]
    # Graph-mode only. Starts a local server and runs the partition graphs check on a session
    # connected to its target.
    server = server_lib.Server.create_local_server()
    self.runTestOutputPartitionGraphs(session.Session(server.target))

  def testNonInteractiveSessionNesting(self):
    # [FUNCTION DOC]
    # Enters two regular sessions as defaults, then exits the outer controller first; checks an
    # assertion error about violated nesting. Finally resets the default session stack (global state).
    sess1 = session.Session()
    sess1_controller = sess1.as_default()
    sess1_controller.__enter__()

    sess2 = session.Session()
    sess2_controller = sess2.as_default()
    sess2_controller.__enter__()

    with self.assertRaisesRegex(AssertionError, 'Nesting violated'):
      sess1_controller.__exit__(None, None, None)

    stack._default_session_stack.reset()

  def testInteractiveSessionNesting(self):
    # [FUNCTION DOC]
    # Creates two interactive sessions and deletes the references. There are no assertions; it only
    # checks that this sequence does not fail.
    sess1 = session.InteractiveSession()
    sess2 = session.InteractiveSession()
    del sess1
    del sess2

  @test_util.run_v1_only('b/120545219')
  def testAsDefault(self):
    # [FUNCTION DOC]
    # Graph-mode only. Evaluates a constant inside a session made default, and again inside an
    # uncaptured temporary session made default, checking both return the value.
    c = constant_op.constant(37)
    sess = session.Session()
    with sess.as_default():
      self.assertEqual(37, self.evaluate(c))

    # Ensure that the session remains valid even when it is not captured.
    with session.Session().as_default():
      self.assertEqual(37, self.evaluate(c))

  def testReentry(self):
    # [FUNCTION DOC]
    # Checks entering the same session twice (nested context use) raises a runtime error saying it is
    # not re-entrant.
    sess = session.Session()
    with self.assertRaisesRegex(RuntimeError, 'not re-entrant'):
      with sess:
        with sess:
          pass

  def testInvalidArgument(self):
    # [FUNCTION DOC]
    # Checks creating a session with a non-string target, a non-config configuration or a non-graph
    # graph argument raises a type error with the corresponding message.
    with self.assertRaisesRegex(TypeError,
                                'Argument `target` must be a string'):
      session.Session(37)
    with self.assertRaisesRegex(TypeError,
                                'Argument `config` must be a tf.ConfigProto'):
      session.Session(config=37)
    with self.assertRaisesRegex(TypeError,
                                'Argument `graph` must be a tf.Graph'):
      session.Session(graph=37)

  @test_util.run_v1_only('b/120545219')
  def testTimeoutWithShortOperations(self):
    # [FUNCTION DOC]
    # Graph-mode only. With a ten-second operation timeout, repeatedly runs a non-blocking enqueue on
    # a queue and checks the queue size afterwards; the timeout must not trigger.
    num_epochs = 5
    q = data_flow_ops.FIFOQueue(capacity=50, dtypes=[dtypes.int32], shapes=[()])
    enqueue_op = q.enqueue_many(constant_op.constant([1, 2]))

    # Use a 10-second timeout, which should be longer than any
    # non-blocking enqueue_many op.
    config_pb = config_pb2.ConfigProto(operation_timeout_in_ms=10000)
    with session.Session(config=config_pb) as sess:
      for _ in range(num_epochs):
        sess.run(enqueue_op)
      self.assertEqual(sess.run(q.size()), num_epochs * 2)

  @test_util.run_v1_only('b/120545219')
  def testRegisterFetchAndFeedConversionFunctions(self):
    # [FUNCTION DOC]
    # Graph-mode only. Registers conversion functions so a custom class can be fetched and fed; a
    # second registration for the same class must raise a value error. Then runs, feeds and partially
    # runs a custom object and checks the squared values.
    # Side effect: the registration is global and persists for the process.

    class SquaredTensor(object):
      # [CLASS DOC]
      # Local custom class wrapping the square of a tensor, used to test registered fetch and feed
      # conversion functions. State: the squared tensor.

      def __init__(self, tensor):
        # [FUNCTION DOC]
        # Constructor taking a tensor (or array-like); stores the elementwise square as an attribute.
        self.sq = math_ops.square(tensor)

    fetch_fn = lambda squared_tensor: ([squared_tensor.sq], lambda val: val[0])
    feed_fn1 = lambda feed, feed_val: [(feed.sq, feed_val)]
    feed_fn2 = lambda feed: [feed.sq]

    session.register_session_run_conversion_functions(SquaredTensor, fetch_fn,
                                                      feed_fn1, feed_fn2)
    with self.assertRaises(ValueError):
      session.register_session_run_conversion_functions(SquaredTensor, fetch_fn,
                                                        feed_fn1, feed_fn2)
    with self.cached_session() as sess:
      np1 = np.array([1.0, 1.5, 2.0, 2.5])
      np2 = np.array([3.0, 3.5, 4.0, 4.5])
      squared_tensor = SquaredTensor(np2)
      squared_eval = sess.run(squared_tensor)
      self.assertAllClose(np2 * np2, squared_eval)
      squared_eval = sess.run(
          squared_tensor, feed_dict={
              squared_tensor: np1 * np1
          })
      self.assertAllClose(np1 * np1, squared_eval)
      partial_run = sess.partial_run_setup([squared_tensor], [])
      squared_eval = sess.partial_run(partial_run, squared_tensor)
      self.assertAllClose(np2 * np2, squared_eval)

  def testDefaultLogDevicePlacement(self):
    # [FUNCTION DOC]
    # Enables device placement logging globally, captures the process error stream from native code
    # while adding two constants (eagerly, or via a local server in graph mode) and checks the log shows
    # two add executions with results 3. Then runs a compiled function with two dependent additions and
    # checks results 3 and 4 and two logged add executions.
    # Side effect: the global logging setting is turned on and not restored by the test.

    class CaptureStderr(str):
      # [CLASS DOC]
      # HELPER: context manager used to capture low-level standard error output (including from native
      # libraries) by redirecting the file descriptor to a pipe. Subclasses the string type; the text
      # returned by string conversion is the captured output. State: escape marker, captured text, saved
      # descriptors and pipe.
      """Class to capture stderr from C++ shared library."""

      def __enter__(self):
        # [FUNCTION DOC]
        # Starts capture: duplicates the error stream's file descriptor, replaces it with the write end of a
        # new pipe, and returns the object itself.
        self._esc = compat.as_str('\b')
        self._output = compat.as_str('')
        self._stderr = sys.stderr
        self._fd = self._stderr.fileno()
        self._out_pipe, in_pipe = os.pipe()
        # Save the original io stream.
        self._dup_fd = os.dup(self._fd)
        # Replace the original io stream with in pipe.
        os.dup2(in_pipe, self._fd)
        return self

      def __exit__(self, *args):
        # [FUNCTION DOC]
        # Ends capture: writes an escape marker to the stream, reads the pipe until the marker (saving
        # the text), closes the read end and restores the original descriptor. Arguments are ignored.
        self._stderr.write(self._esc)
        self._stderr.flush()
        self.read()
        os.close(self._out_pipe)
        # Restore the original io stream.
        os.dup2(self._dup_fd, self._fd)

      def read(self):
        # [FUNCTION DOC]
        # Reads the pipe one character at a time, appending to the captured output, until the escape marker
        # or end of data is seen. Returns nothing.
        while True:
          data = os.read(self._out_pipe, 1)
          if not data or compat.as_str(data) == self._esc:
            break
          self._output += compat.as_str(data)

      def __str__(self):
        # [FUNCTION DOC]
        # Returns the captured output text.
        return self._output

    context.set_log_device_placement(True)
    if context.executing_eagerly():
      with CaptureStderr() as log:
        a = constant_op.constant(1)
        b = constant_op.constant(2)
        c = a + b
        # Ensure if the same kernel with the same arguments is executed then its
        # execution is logged.
        d = a + b
    else:
      # Passing the config to the server, but not the session should still
      # result in logging device placement.
      config_pb = config_pb2.ConfigProto(log_device_placement=True)
      server = server_lib.Server.create_local_server(config=config_pb)
      a = constant_op.constant(1)
      b = constant_op.constant(2)
      c = a + b
      d = a + b
      with session.Session(server.target) as sess:
        with CaptureStderr() as log:
          c, d = sess.run([c, d])

    self.assertEqual(c, 3)
    self.assertEqual(d, 3)

    # Ensure that we did log device placement.
    # We have three modes of execution at the moment:
    # (1) TF1 Graph (2) TF2 eager (3) TF2 eager with function wrapping.
    # The codepaths taken by each are slightly different in all resulting in
    # slightly different logging messages.
    log_msg = ('Executing op AddV2'
               if ops.executing_eagerly_outside_functions() else 'AddV2')
    add_executions = [l for l in str(log).splitlines() if log_msg in l]
    self.assertEqual(len(add_executions), 2)

    @def_function.function
    def fn(a, b):
      # [FUNCTION DOC]
      # Compiled function (decorated) of two tensors: returns the pair of their sum and the sum of the
      # first input with that sum. The second addition uses a different input so an optimizer does not
      # merge duplicate operations.
      c = a + b
      # These two AddV2 cannot use the same argument in tf.function since an
      # optimization pass will remove duplicate ops and only run it once.
      d = a + c
      return c, d

    with CaptureStderr() as log:
      c, d = self.evaluate(fn(constant_op.constant(1), constant_op.constant(2)))
    self.assertEqual(c, 3)
    self.assertEqual(d, 4)
    # Ensure that we did log device placement.
    add_executions = [l for l in str(log).splitlines() if 'AddV2' in l]
    self.assertEqual(len(add_executions), 2)

  @test_util.run_v1_only('b/120545219')
  def testLocalMasterSessionTimeout(self):
    # [FUNCTION DOC]
    # Graph-mode only. Starts a local server, creates a session with an operation timeout of one second
    # in its config and runs a dequeue on an empty queue; expects a deadline-exceeded error.
    # Test that the timeout passed in a config to the session works correctly.
    config_pb = config_pb2.ConfigProto(operation_timeout_in_ms=1000)
    server = server_lib.Server.create_local_server()
    q = data_flow_ops.FIFOQueue(1, dtypes.float32)
    dequeued_t = q.dequeue()

    with session.Session(server.target, config=config_pb) as sess:
      # Intentionally do not run any enqueue_ops so that dequeue will block
      # until operation_timeout_in_ms.
      with self.assertRaises(errors.DeadlineExceededError):
        sess.run(dequeued_t)

  @test_util.run_v1_only('b/120545219')
  def testDefaultServerTimeout(self):
    # [FUNCTION DOC]
    # Graph-mode only. Starts a local server with a one-second operation timeout in its config and a
    # session without any config; a blocking dequeue must raise a deadline-exceeded error.
    # Test that the default server config timeout gets used when no Session
    # config is provided.
    config_pb = config_pb2.ConfigProto(operation_timeout_in_ms=1000)
    server = server_lib.Server.create_local_server(config=config_pb)
    q = data_flow_ops.FIFOQueue(1, dtypes.float32)
    dequeued_t = q.dequeue()

    with session.Session(server.target) as sess:
      # Intentionally do not run any enqueue_ops so that dequeue will block
      # until operation_timeout_in_ms.
      with self.assertRaises(errors.DeadlineExceededError):
        sess.run(dequeued_t)

  def runTestBuildGraphError(self, sess):
    # [FUNCTION DOC]
    # HELPER: shared body for graph build error tests, taking a session. Builds an add of two values
    # entering different control flow frames and checks that running it raises an op error stating inputs
    # come from different frames. Returns nothing.
    # Ensure that errors from building the graph get propagated.
    data = array_ops.placeholder(dtypes.float32, shape=[])
    # pylint: disable=protected-access
    enter_1 = gen_control_flow_ops.enter(data, 'foo_1', False)
    enter_2 = gen_control_flow_ops.enter(data, 'foo_2', False)
    # pylint: enable=protected-access
    res = math_ops.add(enter_1, enter_2)
    with self.assertRaisesOpError('has inputs from different frames'):
      sess.run(res, feed_dict={data: 1.0})

  @test_util.run_v1_only('b/120545219')
  def testBuildGraphErrorDirect(self):
    # [FUNCTION DOC]
    # Graph-mode only. Runs the build error check on a plain session.
    self.runTestBuildGraphError(session.Session())

  @test_util.run_v1_only('b/120545219')
  def testBuildGraphErrorDist(self):
    # [FUNCTION DOC]
    # Graph-mode only. Runs the build error check on a session connected to a local server.
    server = server_lib.Server.create_local_server()
    self.runTestBuildGraphError(session.Session(server.target))

  def testDeviceAttributes(self):
    # [FUNCTION DOC]
    # Creates a private device attributes object from a name, type, memory limit and incarnation; checks
    # the fields and that its text form starts with the class name.
    attrs = session._DeviceAttributes(
        '/job:worker/replica:0/task:3/device:CPU:2', 'TYPE', 1337, 1000000)
    self.assertEqual(1337, attrs.memory_limit_bytes)
    self.assertEqual('/job:worker/replica:0/task:3/device:CPU:2', attrs.name)
    self.assertEqual('TYPE', attrs.device_type)
    self.assertEqual(1000000, attrs.incarnation)
    str_repr = '%s' % attrs
    self.assertTrue(str_repr.startswith('_DeviceAttributes'), str_repr)

  def testDeviceAttributesCanonicalization(self):
    # [FUNCTION DOC]
    # Like the device attributes test but with a short lowercase device name; checks the name is
    # canonicalized to the full device form and other fields are kept.
    attrs = session._DeviceAttributes('/job:worker/replica:0/task:3/cpu:1',
                                      'TYPE', 1337, 1000000)
    self.assertEqual(1337, attrs.memory_limit_bytes)
    self.assertEqual('/job:worker/replica:0/task:3/device:CPU:1', attrs.name)
    self.assertEqual('TYPE', attrs.device_type)
    self.assertEqual(1000000, attrs.incarnation)
    str_repr = '%s' % attrs
    self.assertTrue(str_repr.startswith('_DeviceAttributes'), str_repr)

  def runTestAddFunctionToSession(self, target=''):
    # [FUNCTION DOC]
    # HELPER: shared body for add-function tests. Input: optional session target string (default
    # empty, meaning in-process). Runs a constant, then defines an increment function and uses it on the
    # same graph after the first run; the result must be 2.
    """Add a function to a session after the graph has already been run."""

    @function.Defun(dtypes.float32)
    def foo(x):
      # [FUNCTION DOC]
      # Graph function defined for float32 inputs; returns its input plus one.
      return x + 1

    x = constant_op.constant(1.0)
    with session.Session(target=target) as sess:
      sess.run(x)
      f = foo(x)
      result = sess.run(f)
      self.assertEqual(result, 2.0)

  @test_util.run_v1_only('b/120545219')
  def testAddFunctionToSession(self):
    # [FUNCTION DOC]
    # Graph-mode only. Runs the add-function check on an in-process session.
    self.runTestAddFunctionToSession()

  @test_util.run_v1_only('b/120545219')
  def testAddFunctionToGrpcSession(self):
    # [FUNCTION DOC]
    # Graph-mode only. Starts a local server and runs the add-function check against its target.
    server = server_lib.Server.create_local_server()
    self.runTestAddFunctionToSession(server.target)

  def testOpenAndCloseGrpcSession(self):
    # [FUNCTION DOC]
    # Starts a local server and opens and closes a session connected to it; only checks no error occurs.
    server = server_lib.Server.create_local_server()
    with session.Session(server.target):
      pass

  def testOpenAndCloseSession(self):
    # [FUNCTION DOC]
    # Opens and closes an in-process session; only checks no error occurs.
    with session.Session():
      pass

  @test_util.run_v1_only('b/120545219')
  def testAutoConvertAndCheckData(self):
    # [FUNCTION DOC]
    # Graph-mode only. Feeds an integer into a string placeholder and checks a type error describing
    # the feed value type is raised.
    with self.cached_session() as sess:
      a = array_ops.placeholder(dtype=dtypes.string)
      with self.assertRaisesRegex(
          TypeError, r'Type of feed value 1 with type <(\w+) \'int\'> is not'):
        sess.run(a, feed_dict={a: 1})

  @test_util.run_v1_only('b/120545219')
  def testOptimizerOptions(self):
    # [FUNCTION DOC]
    # Graph-mode only. Sets the experimental optimizer option for the minimum number of graph nodes to
    # minus one globally, creates a session and checks its stored configuration has that value.
    # Side effect: global optimizer options are changed and not restored.
    config.set_optimizer_experimental_options({'min_graph_nodes': -1})

    with ops.Graph().as_default():
      sess = session.Session()
      self.assertEqual(
          sess._config.graph_options.rewrite_options.min_graph_nodes, -1)


if __name__ == '__main__':
  googletest.main()
