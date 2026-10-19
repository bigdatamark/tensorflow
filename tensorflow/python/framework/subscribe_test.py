# [MODULE DOC]
# Unit tests for the tensor subscription feature (subscribe module). Covers moving control
# dependencies to the subscription identity, supported container types, cache freshness between
# calls, subscription identity detection, extending existing subscriptions, variables, tensors of
# resource type, ops with multiple outputs, device placement and control-flow context
# preservation. Most tests run in legacy graph mode. No import-time side effects.
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
"""Tests for tf.subscribe."""

import collections

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import subscribe
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.platform import googletest


class SubscribeTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the subscribe function; each test builds operations in the default graph, applies
  # subscriptions whose side effect records the observed values through a Python callback op, and
  # evaluates in a session.

  def _ExpectSubscribedIdentities(self, container):
    # [FUNCTION DOC]
    # HELPER: Assertion helper. Asserts every element of the container is recognized as a
    # subscription identity tensor.
    """Convenience function to test a container of subscribed identities."""
    self.assertTrue(
        all(subscribe._is_subscribed_identity(x) for x in container))

  @test_util.run_deprecated_v1
  def testSideEffect(self):
    # [FUNCTION DOC]
    # Scenario: subscribes to a sum tensor that another op depends on via a control dependency.
    # Asserts the control dependency moves from the original op to the subscription identity,
    # that evaluating the sum, a negation of it and the dependent constant give 2, minus 2 and
    # 42, and that the recording callback saw the value 2 exactly three times.
    a = constant_op.constant(1)
    b = constant_op.constant(1)
    c = math_ops.add(a, b)
    with ops.control_dependencies([c]):
      d = constant_op.constant(42)
    n = math_ops.negative(c)

    shared = []

    def sub(t):
      # [FUNCTION DOC]
      # HELPER: Side-effect callback run inside the Python callback op. Records the observed
      # value in the shared list and returns it unchanged.
      shared.append(t)
      return t

    c0 = c
    self.assertTrue(c0.op in d.op.control_inputs)
    c = subscribe.subscribe(c,
                            lambda t: script_ops.py_func(sub, [t], [t.dtype]))
    # Verify that control dependencies are correctly moved to the subscription.
    self.assertFalse(c0.op in d.op.control_inputs)
    self.assertTrue(c.op in d.op.control_inputs)

    with self.cached_session() as sess:
      c_out = self.evaluate([c])
      n_out = self.evaluate([n])
      d_out = self.evaluate([d])

    self.assertEqual(n_out, [-2])
    self.assertEqual(c_out, [2])
    self.assertEqual(d_out, [42])
    self.assertEqual(shared, [2, 2, 2])

  @test_util.run_deprecated_v1
  def testSupportedTypes(self):
    # [FUNCTION DOC]
    # Scenario: subscribes to a tuple, list, dictionary and named tuple of tensors. Asserts the
    # returned container type matches the input and every element is a subscription identity;
    # subscribing a plain string raises a type error mentioning an invalid type.
    """Confirm that supported types are correctly detected and handled."""

    a = constant_op.constant(1)
    b = constant_op.constant(1)
    c = math_ops.add(a, b)

    def sub(t):
      # [FUNCTION DOC]
      # HELPER: Do-nothing side-effect callback; returns its input unchanged.
      return t

    # Tuples.
    subscribed = subscribe.subscribe(
        (a, b), lambda t: script_ops.py_func(sub, [t], [t.dtype]))
    self.assertIsInstance(subscribed, tuple)
    self._ExpectSubscribedIdentities(subscribed)

    # Lists.
    subscribed = subscribe.subscribe(
        [a, b], lambda t: script_ops.py_func(sub, [t], [t.dtype]))
    self.assertIsInstance(subscribed, list)
    self._ExpectSubscribedIdentities(subscribed)

    # Dictionaries.
    subscribed = subscribe.subscribe({
        'first': a,
        'second': b
    }, lambda t: script_ops.py_func(sub, [t], [t.dtype]))
    self.assertIsInstance(subscribed, dict)
    self._ExpectSubscribedIdentities(subscribed.values())

    # Namedtuples.
    # pylint: disable=invalid-name
    TensorPair = collections.namedtuple('TensorPair', ['first', 'second'])
    # pylint: enable=invalid-name
    pair = TensorPair(a, b)
    subscribed = subscribe.subscribe(
        pair, lambda t: script_ops.py_func(sub, [t], [t.dtype]))
    self.assertIsInstance(subscribed, TensorPair)
    self._ExpectSubscribedIdentities(subscribed)

    # Expect an exception to be raised for unsupported types.
    with self.assertRaisesRegex(TypeError, 'has invalid type'):
      subscribe.subscribe(c.name,
                          lambda t: script_ops.py_func(sub, [t], [t.dtype]))

  @test_util.run_deprecated_v1
  def testCaching(self):
    # [FUNCTION DOC]
    # Scenario: subscribes to one tensor, then adds a new control dependency on another tensor
    # and subscribes to it. Asserts evaluating both dependents triggers both subscriptions once
    # each (proving the control-output cache is rebuilt per call) and the values are 42 and 11.
    """Confirm caching of control output is recalculated between calls."""
    a = constant_op.constant(1)
    b = constant_op.constant(2)
    with ops.control_dependencies([a]):
      c = constant_op.constant(42)

    shared = {}

    def sub(t):
      # [FUNCTION DOC]
      # HELPER: Side-effect callback that counts how many times each observed value was seen in
      # the shared mapping and returns the value unchanged.
      shared[t] = shared.get(t, 0) + 1
      return t

    a = subscribe.subscribe(a,
                            lambda t: script_ops.py_func(sub, [t], [t.dtype]))

    with ops.control_dependencies([b]):
      d = constant_op.constant(11)

    # If it was using outdated cached control_outputs then
    # evaling would not trigger the new subscription.
    b = subscribe.subscribe(b,
                            lambda t: script_ops.py_func(sub, [t], [t.dtype]))

    with self.cached_session() as sess:
      c_out = self.evaluate([c])
      d_out = self.evaluate([d])

    self.assertEqual(c_out, [42])
    self.assertEqual(d_out, [11])
    self.assertEqual(shared, {2: 1, 1: 1})

  @test_util.run_deprecated_v1
  def testIsSubscribedIdentity(self):
    # [FUNCTION DOC]
    # Scenario: asserts the identity check is false for a constant, a sum and an ordinary
    # identity op, and true for the result of subscribing with an empty side-effect list.
    """Confirm subscribed identity ops are correctly detected."""
    a = constant_op.constant(1)
    b = constant_op.constant(2)
    c = math_ops.add(a, b)
    idop = array_ops.identity(c)
    c_sub = subscribe.subscribe(c, [])

    self.assertFalse(subscribe._is_subscribed_identity(a))
    self.assertFalse(subscribe._is_subscribed_identity(c))
    self.assertFalse(subscribe._is_subscribed_identity(idop))
    self.assertTrue(subscribe._is_subscribed_identity(c_sub))

  @test_util.run_deprecated_v1
  def testSubscribeExtend(self):
    # [FUNCTION DOC]
    # Scenario: subscribes a tensor three times (original, the returned identity, original
    # again), each with a different side-effect callback. Asserts a single subscription identity
    # exists, all returned tensors are the same object, and all three callbacks ran on evaluation.
    """Confirm side effect are correctly added for different input types."""
    a = constant_op.constant(1)
    b = constant_op.constant(2)
    c = math_ops.add(a, b)

    shared = {}

    def sub(t, name):
      # [FUNCTION DOC]
      # HELPER: Side-effect callback taking the value and a label. Counts invocations per label
      # in the shared mapping and returns the value unchanged.
      shared[name] = shared.get(name, 0) + 1
      return t

    # Subscribe with a first side effect graph, passing an unsubscribed tensor.
    sub_graph1 = lambda t: sub(t, 'graph1')
    c_sub = subscribe.subscribe(
        c, lambda t: script_ops.py_func(sub_graph1, [t], [t.dtype]))

    # Add a second side effect graph, passing the tensor returned by the
    # previous call to subscribe().
    sub_graph2 = lambda t: sub(t, 'graph2')
    c_sub2 = subscribe.subscribe(
        c_sub, lambda t: script_ops.py_func(sub_graph2, [t], [t.dtype]))

    # Add a third side effect graph, passing the original tensor.
    sub_graph3 = lambda t: sub(t, 'graph3')
    c_sub3 = subscribe.subscribe(
        c, lambda t: script_ops.py_func(sub_graph3, [t], [t.dtype]))

    # Make sure there's only one identity op matching the source tensor's name.
    graph_ops = ops.get_default_graph().get_operations()
    name_prefix = c.op.name + '/subscription/Identity'
    identity_ops = [op for op in graph_ops if op.name.startswith(name_prefix)]
    self.assertEqual(1, len(identity_ops))

    # Expect the objects returned by subscribe() to reference the same tensor.
    self.assertIs(c_sub, c_sub2)
    self.assertIs(c_sub, c_sub3)

    # Expect the three side effect graphs to have been evaluated.
    with self.cached_session() as sess:
      self.evaluate([c_sub])
    self.assertIn('graph1', shared)
    self.assertIn('graph2', shared)
    self.assertIn('graph3', shared)

  @test_util.run_v1_only('b/120545219')
  def testSubscribeVariable(self):
    # [FUNCTION DOC]
    # Scenario (version-1 only): subscribes to a variable that is added to another and assigned.
    # Asserts the result is a subscription identity, evaluating the sum triggers the side effect
    # (reading the variable), the assign op does not, and the recorded values are 0 then 3.
    """Confirm that variables can be subscribed."""
    v1 = variable_v1.VariableV1(0.0)
    v2 = variable_v1.VariableV1(4.0)
    add = math_ops.add(v1, v2)
    assign_v1 = v1.assign(3.0)

    shared = []

    def sub(t):
      # [FUNCTION DOC]
      # HELPER: Side-effect callback recording each observed value in the shared list and
      # returning it unchanged.
      shared.append(t)
      return t

    v1_sub = subscribe.subscribe(
        v1, lambda t: script_ops.py_func(sub, [t], [t.dtype]))
    self.assertTrue(subscribe._is_subscribed_identity(v1_sub))

    with self.cached_session() as sess:
      # Initialize the variables first.
      self.evaluate([v1.initializer])
      self.evaluate([v2.initializer])

      # Expect the side effects to be triggered when evaluating the add op as
      # it will read the value of the variable.
      self.evaluate([add])
      self.assertEqual(1, len(shared))

      # Expect the side effect not to be triggered when evaluating the assign
      # op as it will not access the 'read' output of the variable.
      self.evaluate([assign_v1])
      self.assertEqual(1, len(shared))

      self.evaluate([add])
      self.assertEqual(2, len(shared))

      # Make sure the values read from the variable match the expected ones.
      self.assertEqual([0.0, 3.0], shared)

  @test_util.run_v1_only('b/120545219')
  def testResourceType(self):
    # [FUNCTION DOC]
    # Scenario (version-1 only): subscribes to a tensor array's handle, whose resource type is
    # not numpy compatible. Asserts the original tensor is returned unchanged, it is not a
    # subscription identity, and the side effect never runs when reading the array.
    """Confirm that subscribe correctly handles tensors with 'resource' type."""
    tensor_array = tensor_array_ops.TensorArray(
        dtype=dtypes.float32,
        tensor_array_name='test',
        size=3,
        infer_shape=False)
    writer = tensor_array.write(0, [[4.0, 5.0]])
    reader = writer.read(0)

    shared = []

    def sub(t):
      # [FUNCTION DOC]
      # HELPER: Side-effect callback recording each observed value in the shared list and
      # returning it unchanged (expected never to be called in this test).
      shared.append(t)
      return t

    # TensorArray's handle output tensor has a 'resource' type and cannot be
    # subscribed as it's not 'numpy compatible' (see dtypes.py).
    # Expect that the original tensor is returned when subscribing to it.
    tensor_array_sub = subscribe.subscribe(
        tensor_array.handle, lambda t: script_ops.py_func(sub, [t], [t.dtype]))
    self.assertIs(tensor_array_sub, tensor_array.handle)
    self.assertFalse(subscribe._is_subscribed_identity(tensor_array.handle))

    with self.cached_session() as sess:
      self.evaluate([reader])
    self.assertEqual(0, len(shared))

  @test_util.run_deprecated_v1
  def testMultipleOutputs(self):
    # [FUNCTION DOC]
    # Scenario: a sparse addition op with three outputs, all of which another op depends on via
    # control dependencies. Subscribes all three outputs at once, evaluates the dependent op and
    # asserts the side effect ran three times (one per output).
    """Handle subscriptions to multiple outputs from the same op."""
    sparse_tensor_1 = sparse_tensor.SparseTensor(
        indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])
    sparse_tensor_2 = sparse_tensor.SparseTensor(
        indices=[[0, 0], [1, 2]], values=[2, 3], dense_shape=[3, 4])

    # This op has three outputs.
    sparse_add = sparse_ops.sparse_add(sparse_tensor_1, sparse_tensor_2)

    self.assertEqual(3, len(sparse_add.op.outputs))

    c1 = constant_op.constant(1)

    with ops.control_dependencies(sparse_add.op.outputs):
      # This op depends on all the three outputs.
      neg = -c1

    shared = []
    def sub(t):
      # [FUNCTION DOC]
      # HELPER: Side-effect callback recording each observed value in the shared list and
      # returning it unchanged.
      shared.append(t)
      return t

    # Subscribe the three outputs at once.
    subscribe.subscribe(sparse_add.op.outputs,
                        lambda t: script_ops.py_func(sub, [t], [t.dtype]))

    with self.cached_session() as sess:
      self.evaluate([neg])

    # All three ops have been processed.
    self.assertEqual(3, len(shared))

  @test_util.run_deprecated_v1
  def test_subscribe_tensors_on_different_devices(self):
    # [FUNCTION DOC]
    # Scenario: two tensors are created on different CPU devices and subscribed. Asserts the
    # resulting identity tensors sit on the same devices as their sources, and that the two
    # devices differ.
    """Side effect ops are added with the same device of the subscribed op."""
    c1 = constant_op.constant(10)
    c2 = constant_op.constant(20)

    with ops.device('cpu:0'):
      add = math_ops.add(c1, c2)

    with ops.device('cpu:1'):
      mul = math_ops.multiply(c1, c2)

    def sub(t):
      # [FUNCTION DOC]
      # HELPER: Do-nothing side-effect callback; returns its input unchanged.
      return t

    add_sub = subscribe.subscribe(
        add, lambda t: script_ops.py_func(sub, [t], [t.dtype]))

    mul_sub = subscribe.subscribe(
        mul, lambda t: script_ops.py_func(sub, [t], [t.dtype]))

    # Expect the identity tensors injected by subscribe to have been created
    # on the same device as their original tensors.
    self.assertNotEqual(add_sub.device, mul_sub.device)
    self.assertEqual(add.device, add_sub.device)
    self.assertEqual(mul.device, mul_sub.device)

  @test_util.run_v1_only('b/120545219')
  def test_subscribe_tensors_within_control_flow_context(self):
    # [FUNCTION DOC]
    # Scenario (version-1 only): subscribes tensors outside a conditional and one inside a
    # conditional branch. Asserts each subscription identity shares the control flow context of
    # its source, that contexts differ between outside and branch tensors, and that all three
    # side effects run when the conditional is evaluated.
    """Side effect ops are added with the same control flow context."""
    c1 = constant_op.constant(10)
    c2 = constant_op.constant(20)
    x1 = math_ops.add(c1, c2)
    x2 = math_ops.multiply(c1, c2)

    cond = tf_cond.cond(
        x1 < x2,
        lambda: math_ops.add(c1, c2, name='then'),
        lambda: math_ops.subtract(c1, c2, name='else'),
        name='cond')

    branch = ops.get_default_graph().get_tensor_by_name('cond/then:0')

    def context(tensor):
      # [FUNCTION DOC]
      # HELPER: Returns the control flow context object of the tensor's producing operation
      # (nothing if it is outside any conditional or loop).
      return tensor.op._get_control_flow_context()

    self.assertIs(context(x1), context(x2))
    self.assertIsNot(context(x1), context(branch))

    results = []
    def sub(tensor):
      # [FUNCTION DOC]
      # HELPER: Side-effect callback recording each observed tensor in the results list and
      # returning it unchanged.
      results.append(tensor)
      return tensor

    tensors = [x1, branch, x2]
    subscriptions = subscribe.subscribe(
        tensors, lambda t: script_ops.py_func(sub, [t], [t.dtype]))

    for tensor, subscription in zip(tensors, subscriptions):
      self.assertIs(context(tensor), context(subscription))

    # Verify that sub(x1) and sub(x2) are in the same context.
    self.assertIs(context(subscriptions[0]), context(subscriptions[2]))

    # Verify that sub(x1) and sub(branch) are not.
    self.assertIsNot(context(subscriptions[0]), context(subscriptions[1]))

    with self.cached_session() as sess:
      self.evaluate(cond)

    self.assertEqual(3, len(results))


if __name__ == '__main__':
  googletest.main()
