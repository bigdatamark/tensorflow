# [MODULE DOC]
# Unit tests for the control flow operations in the TensorFlow python ops package (group,
# with_dependencies, switch, cond, case, switch_case, while loops, tuple, control flow contexts,
# tensor-array and variable interactions, gradients through control flow, and v1 versus v2 control
# flow behavior). The module defines many test case classes derived from the TensorFlow test base
# class, a few module level helper functions for computing nested shapes and building tensor
# arrays, and two module level named tuple types used as structured branch outputs. Many tests are
# decorated to run only in graph mode (v1), only in v1, or in both graph and eager modes.
# Importing the module has no side effects beyond importing the tensor array gradient module for its
# gradient registrations; running the file as a script launches the test runner.
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
"""Tests for control_flow_ops.py."""

import collections
import itertools
import time

from absl.testing import parameterized
import numpy as np

from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python import tf2
from tensorflow.python.autograph.lang import directives
from tensorflow.python.client import session
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import cond_v2
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_case
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_switch_case
from tensorflow.python.ops import control_flow_util_v2
from tensorflow.python.ops import control_flow_v2_toggles
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.ops import while_v2
import tensorflow.python.ops.tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.platform import googletest
from tensorflow.python.training import momentum
from tensorflow.python.util import nest

TestTuple = collections.namedtuple("TestTuple", "a b")
SingletonTestTuple = collections.namedtuple("SingletonTestTuple", "a")


class GroupTestCase(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the group operation: how it builds a no-op node with control dependencies on the
  # grouped operations, with and without device placement. Provides two helper methods that strip
  # graph definitions down to name, op, inputs and device so they can be compared as text protos.

  def _StripNode(self, nd):
    # [FUNCTION DOC]
    # HELPER: reduces node definitions for comparison in the group tests.
    # Takes a node definition and returns a new node definition holding only its name, op type and
    # inputs, plus the device if one is set. The argument is not modified.
    snode = node_def_pb2.NodeDef(name=nd.name, op=nd.op, input=nd.input)
    if nd.device:
      snode.device = nd.device
    return snode

  def _StripGraph(self, gd):
    # [FUNCTION DOC]
    # HELPER: reduces whole graph definitions for comparison in the group tests.
    # Takes a graph definition and returns a new graph definition whose nodes are the stripped
    # versions (name, op, inputs, device only) of the original nodes, in the same order.
    """Copy gd keeping only, node.name, node.op, node.input, and node.device."""
    return graph_pb2.GraphDef(node=[self._StripNode(nd) for nd in gd.node])

  def testGroup_NoDevices(self):
    # [FUNCTION DOC]
    # Scenario: three constants without device assignment are grouped under the name root.
    # Expected: the graph holds the three constants plus one no-op named root that has control
    # inputs on all three constants. Returns nothing.
    with ops.Graph().as_default() as g:
      a = constant_op.constant(0, name="a")
      b = constant_op.constant(0, name="b")
      c = constant_op.constant(0, name="c")
      control_flow_ops.group(a.op, b.op, c.op, name="root")
    gd = g.as_graph_def()
    self.assertProtoEquals(
        """
      node { name: "a" op: "Const"}
      node { name: "b" op: "Const"}
      node { name: "c" op: "Const"}
      node { name: "root" op: "NoOp" input: "^a" input: "^b" input: "^c" }
    """, self._StripGraph(gd))

  def testGroup_OneDevice(self):
    # [FUNCTION DOC]
    # Scenario: two constants are created on task 0 and grouped outside any device scope.
    # Expected: a single no-op named root is created with control inputs on both constants and it
    # inherits the device of task 0. Returns nothing.
    with ops.Graph().as_default() as g:
      with g.device("/task:0"):
        a = constant_op.constant(0, name="a")
        b = constant_op.constant(0, name="b")
      control_flow_ops.group(a.op, b.op, name="root")
    gd = g.as_graph_def()
    self.assertProtoEquals(
        """
      node { name: "a" op: "Const" device: "/task:0" }
      node { name: "b" op: "Const" device: "/task:0" }
      node { name: "root" op: "NoOp" input: "^a" input: "^b" device: "/task:0" }
    """, self._StripGraph(gd))

  def testGroup_MultiDevice(self):
    # [FUNCTION DOC]
    # Scenario: constants live on task 0 and task 1 and the group is created under task 2.
    # Expected: one intermediate no-op per device (root/NoOp on task 0, root/NoOp_1 on task 1)
    # each controlling the constants of that device, and a final no-op root on task 2 that has
    # control inputs on both intermediate no-ops. Returns nothing.
    with ops.Graph().as_default() as g:
      with g.device("/task:0"):
        a = constant_op.constant(0, name="a")
        b = constant_op.constant(0, name="b")
      with g.device("/task:1"):
        c = constant_op.constant(0, name="c")
        d = constant_op.constant(0, name="d")
      with g.device("/task:2"):
        control_flow_ops.group(a.op, b.op, c.op, d.op, name="root")
    gd = g.as_graph_def()
    self.assertProtoEquals(
        """
      node { name: "a" op: "Const" device: "/task:0"}
      node { name: "b" op: "Const" device: "/task:0"}
      node { name: "c" op: "Const" device: "/task:1"}
      node { name: "d" op: "Const" device: "/task:1"}
      node { name: "root/NoOp" op: "NoOp" input: "^a" input: "^b"
             device: "/task:0" }
      node { name: "root/NoOp_1" op: "NoOp" input: "^c" input: "^d"
             device: "/task:1" }
      node { name: "root" op: "NoOp" input: "^root/NoOp" input: "^root/NoOp_1"
             device: "/task:2" }
    """, self._StripGraph(gd))

  def testPassingList(self):
    # [FUNCTION DOC]
    # Scenario: the operations to group are passed as a single list argument instead of
    # separate arguments. Expected: same result as passing them separately, a no-op named root
    # with control inputs on both constants. Returns nothing.
    with ops.Graph().as_default() as g:
      a = constant_op.constant(0, name="a")
      b = constant_op.constant(0, name="b")
      control_flow_ops.group([a.op, b.op], name="root")
    gd = g.as_graph_def()
    self.assertProtoEquals(
        """
      node { name: "a" op: "Const"}
      node { name: "b" op: "Const"}
      node { name: "root" op: "NoOp" input: "^a" input: "^b" }
    """, self._StripGraph(gd))

  @test_util.run_deprecated_v1
  def testPassingNonTensors(self):
    # [FUNCTION DOC]
    # Scenario: group is called with plain integers that are not operations or tensors.
    # Expected: a type error is raised. Runs in v1 graph mode only. Returns nothing.
    with self.assertRaises(TypeError):
      control_flow_ops.group(1, 2)


class ShapeTestCase(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case checking that static shape information is preserved by control flow helpers.

  def testShape(self):
    # [FUNCTION DOC]
    # Scenario: a constant vector of two elements is wrapped with a control dependency on another
    # constant. Expected: the static shape of both the original and the dependency-wrapped tensor
    # is two. Returns nothing.
    tensor = constant_op.constant([1.0, 2.0])
    self.assertEqual([2], tensor.get_shape())
    self.assertEqual([2],
                     control_flow_ops.with_dependencies(
                         [constant_op.constant(1.0)], tensor).get_shape())


class WithDependenciesTestCase(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the with-dependencies operation, which returns a tensor that is only available
  # after a set of other operations or tensors has run. Both tests run in v1 graph mode.

  @test_util.run_deprecated_v1
  def testTupleDependencies(self):
    # [FUNCTION DOC]
    # Scenario: a counter variable is incremented by an operation that is passed, together with
    # a constant, as a tuple of dependencies of a constant tensor. Expected: before evaluating the
    # wrapped tensor the counter is zero, evaluating it yields seven, and afterwards the counter
    # is one, proving the dependency ran. Returns nothing.
    counter = variable_scope.get_variable(
        "my_counter", shape=[], initializer=init_ops.zeros_initializer())
    increment_counter = state_ops.assign_add(counter, 1)
    const_with_dep = control_flow_ops.with_dependencies(
        (increment_counter, constant_op.constant(42)), constant_op.constant(7))

    self.evaluate(variables.global_variables_initializer())
    self.assertEqual(0, self.evaluate(counter))
    self.assertEqual(7, self.evaluate(const_with_dep))
    self.assertEqual(1, self.evaluate(counter))

  @test_util.run_deprecated_v1
  def testListDependencies(self):
    # [FUNCTION DOC]
    # Same scenario as the tuple dependencies test, but the dependencies are given as a list:
    # the increment must run when the wrapped constant is evaluated (counter goes from zero to
    # one while the value seven is returned). Returns nothing.
    counter = variable_scope.get_variable(
        "my_counter", shape=[], initializer=init_ops.zeros_initializer())
    increment_counter = state_ops.assign_add(counter, 1)
    const_with_dep = control_flow_ops.with_dependencies(
        [increment_counter, constant_op.constant(42)], constant_op.constant(7))

    self.evaluate(variables.global_variables_initializer())
    self.assertEqual(0, self.evaluate(counter))
    self.assertEqual(7, self.evaluate(const_with_dep))
    self.assertEqual(1, self.evaluate(counter))


class SwitchTestCase(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the switch operation and its interplay with sparse (indexed slices) values,
  # gradients inside while loops and resource variables.

  @test_util.run_deprecated_v1
  def testIndexedSlicesWithDenseShape(self):
    # [FUNCTION DOC]
    # Scenario: an indexed slices value with a dense shape is routed through switch using a
    # predicate that is true. Expected: the true output keeps the original values and indices.
    # Returns nothing.
    with self.cached_session():
      data = indexed_slices.IndexedSlices(
          constant_op.constant([1, 2, 3]),
          constant_op.constant([0, 1, 2]),
          dense_shape=constant_op.constant([3]))
      zero = constant_op.constant(0)
      one = constant_op.constant(1)
      less_op = math_ops.less(zero, one)
      _, switch_true = control_flow_ops.switch(data, less_op)
      self.assertAllEqual([1, 2, 3], switch_true.values)
      self.assertAllEqual([0, 1, 2], switch_true.indices)

  @test_util.run_deprecated_v1
  def testIndexedSlicesGradient(self):
    # [FUNCTION DOC]
    # Scenario: a while loop accumulates the sum of an embedding lookup row from a randomly
    # initialized matrix for five iterations, and a momentum optimizer minimizes that cost.
    # Expected: ten training steps run without error (sparse gradients flow through the loop).
    # Returns nothing.
    embedding_matrix = variable_scope.get_variable(
        "embedding_matrix", [5, 5],
        initializer=init_ops.random_normal_initializer())

    def cond(it, _):
      # [FUNCTION DOC]
      # Loop condition: returns whether the iteration counter is below five. The second loop
      # variable is ignored.
      return it < 5

    def body(it, cost):
      # [FUNCTION DOC]
      # Loop body: looks up row zero of the embedding matrix, adds its sum to the cost and
      # increments the counter. Returns the updated counter and cost.
      embedding = embedding_ops.embedding_lookup(embedding_matrix + 0.0, [0])
      cost += math_ops.reduce_sum(embedding)
      return it + 1, cost

    _, cost = while_loop.while_loop(
        cond, body, [constant_op.constant(0),
                     constant_op.constant(0.0)])
    optimizer = momentum.MomentumOptimizer(0.1, 0.9)
    train_op = optimizer.minimize(cost)
    with self.cached_session():
      self.evaluate(variables.global_variables_initializer())
      for _ in range(10):
        self.evaluate([train_op])

  def testResourceReadInLoop(self):
    # [FUNCTION DOC]
    # Scenario: a resource variable holding two rows is read by an embedding lookup of row zero
    # inside a while loop that runs five times. Expected: the accumulated cost equals ten (five
    # times the value two). Returns nothing.
    embedding_matrix = variable_scope.get_variable(
        "embedding_matrix", initializer=[[2.0], [3.0]], use_resource=True)

    def cond(it, _):
      # [FUNCTION DOC]
      # Loop condition: true while the counter is below five; second variable ignored.
      return it < 5

    def body(it, cost):
      # [FUNCTION DOC]
      # Loop body: adds the sum of the looked-up row to the cost and increments the counter.
      # Returns the new counter and cost.
      embedding = embedding_ops.embedding_lookup(embedding_matrix, [0])
      cost += math_ops.reduce_sum(embedding)
      return it + 1, cost

    _, cost = while_loop.while_loop(
        cond, body, [constant_op.constant(0),
                     constant_op.constant(0.0)])
    with self.cached_session():
      self.evaluate(variables.global_variables_initializer())
      self.assertAllEqual(10.0, self.evaluate(cost))

  def doTestIndexedSlicesGradientInCondInWhileLoop(self, use_resource=False):
    # [FUNCTION DOC]
    # HELPER: shared implementation for the two cond-in-while-loop gradient tests.
    # KNOWN ISSUE: everything after the return statement of the nested body function (the while
    # loop construction, both gradient computations and the final equality assertion) is indented
    # inside the body function after its return, so it is unreachable dead code. The helper
    # therefore only defines functions and never runs any assertion.
    # Intended behavior: build a loop whose body uses a conditional that squares the cost on the
    # fourth iteration, compare the sparse gradient of the dynamic computation with the gradient
    # of an equivalent static expression. Parameter use_resource (default false) selects whether
    # the embedding variable is a resource variable. Returns nothing.
    embedding_matrix = variable_scope.get_variable(
        "embedding_matrix", [5, 5],
        initializer=init_ops.random_normal_initializer(),
        use_resource=use_resource)

    def cond(it, _):
      # [FUNCTION DOC]
      # Loop condition: true while the counter is below five; second variable ignored.
      return it < 5

    def body(it, cost):
      # [FUNCTION DOC]
      # Loop body: looks up row zero of the embedding matrix, then uses a conditional that squares
      # the cost when the counter equals three and otherwise adds the row sum. Returns the
      # incremented counter and the new cost.
      embedding = embedding_ops.embedding_lookup(embedding_matrix, [0])
      cost = tf_cond.cond(
          math_ops.equal(it, 3), lambda: math_ops.square(cost),
          (lambda: cost + math_ops.reduce_sum(embedding)))
      return it + 1, cost

      _, cost = while_loop.while_loop(
          cond, body, [constant_op.constant(0),
                       constant_op.constant(0.0)])

      dynamic_grads = gradients_impl.gradients(cost, [embedding_matrix])[0]
      dynamic_grads = math_ops.segment_sum(dynamic_grads.values,
                                           dynamic_grads.indices)

      embedding = embedding_ops.embedding_lookup(embedding_matrix, [0])
      static = math_ops.square(
          math_ops.reduce_sum(embedding) + math_ops.reduce_sum(embedding) +
          math_ops.reduce_sum(embedding)) + math_ops.reduce_sum(embedding)
      static_grads = gradients_impl.gradients(static, [embedding_matrix])[0]
      static_grads = math_ops.segment_sum(static_grads.values,
                                          static_grads.indices)

      with self.cached_session():
        self.evaluate(variables.global_variables_initializer())
        self.assertAllEqual(*self.evaluate([static_grads, dynamic_grads]))

  def testIndexedSlicesGradientInCondInWhileLoop(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: delegates to a helper whose assertions are unreachable, so this test passes
    # without checking anything.
    # Runs the shared helper using a non-resource (reference style) variable. Returns nothing.
    self.doTestIndexedSlicesGradientInCondInWhileLoop(use_resource=False)

  def testIndexedSlicesGradientInCondInWhileLoopResource(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: delegates to a helper whose assertions are unreachable, so this test passes
    # without checking anything.
    # Runs the shared helper using a resource variable. Returns nothing.
    self.doTestIndexedSlicesGradientInCondInWhileLoop(use_resource=True)

  @test_util.run_v1_only("b/120545219")
  def testIndexedSlicesWithShapeGradientInWhileLoop(self):
    # [FUNCTION DOC]
    # Scenario (v1 only): for float32 and float64, a while loop reads elements of a fixed-length
    # placeholder by index and writes them into a tensor array; the summed stacked result is
    # differentiated with respect to the placeholder. Expected: the sum for the fed values equals
    # twenty and the gradient is a vector of ones with one entry per step. Returns nothing.
    for dtype in [dtypes.float32, dtypes.float64]:
      with self.cached_session() as sess:
        num_steps = 9

        inputs = array_ops.placeholder(dtype=dtype, shape=[num_steps])
        initial_outputs = tensor_array_ops.TensorArray(
            dtype=dtype, size=num_steps)
        initial_i = constant_op.constant(0, dtype=dtypes.int32)

        def cond(i, _):
          # [FUNCTION DOC]
          # Loop condition: true while the index is below the fixed number of steps.
          return i < num_steps  # pylint: disable=cell-var-from-loop

        def body(i, outputs):
          # [FUNCTION DOC]
          # Loop body: gathers the element at the current index from the placeholder, writes it
          # to the tensor array at that index and increments the index. Returns index and array.
          x = array_ops.gather(inputs, i)  # pylint: disable=cell-var-from-loop
          outputs = outputs.write(i, x)
          return i + 1, outputs

        _, outputs = while_loop.while_loop(cond, body,
                                           [initial_i, initial_outputs])

        outputs = math_ops.reduce_sum(outputs.stack())
        r = gradients_impl.gradients([outputs], [inputs])[0]
        grad_wr_inputs = ops.convert_to_tensor(r)
        o, grad = sess.run([outputs, grad_wr_inputs],
                           feed_dict={inputs: [4, 6, 0, 7, 0, 0, 1, 2, 0]})
        self.assertEqual(o, 20)
        self.assertAllEqual(grad, [1] * num_steps)

  @test_util.run_v1_only("b/120545219")
  def testIndexedSlicesWithDynamicShapeGradientInWhileLoop(self):
    # [FUNCTION DOC]
    # Scenario (v1 only): like the fixed-size variant, but the placeholder has unknown shape and
    # the tensor array is dynamically sized; the loop runs for the runtime size of the input.
    # Expected: for the fed values one, three, two the sum is six and the gradient is a vector of
    # three ones. Returns nothing.
    for dtype in [dtypes.float32, dtypes.float64]:
      with self.cached_session() as sess:
        inputs = array_ops.placeholder(dtype=dtype)
        initial_outputs = tensor_array_ops.TensorArray(
            dtype=dtype, dynamic_size=True, size=1)
        initial_i = constant_op.constant(0, dtype=dtypes.int32)

        def cond(i, _):
          # [FUNCTION DOC]
          # Loop condition: true while the index is below the runtime size of the input.
          return i < array_ops.size(inputs)  # pylint: disable=cell-var-from-loop

        def body(i, outputs):
          # [FUNCTION DOC]
          # Loop body: gathers the element at the current index, writes it to the dynamic tensor
          # array and increments the index. Returns the new index and array.
          x = array_ops.gather(inputs, i)  # pylint: disable=cell-var-from-loop
          outputs = outputs.write(i, x)
          return i + 1, outputs

        _, outputs = while_loop.while_loop(cond, body,
                                           [initial_i, initial_outputs])

        outputs = math_ops.reduce_sum(outputs.stack())
        r = gradients_impl.gradients([outputs], [inputs])[0]
        grad_wr_inputs = ops.convert_to_tensor(r)
        o, grad = sess.run([outputs, grad_wr_inputs],
                           feed_dict={inputs: [1, 3, 2]})
        self.assertEqual(o, 6)
        self.assertAllEqual(grad, [1] * 3)

  @test_util.run_deprecated_v1
  def testGradientThroughSingleBranchOutsideOfContext(self):
    # [FUNCTION DOC]
    # Scenario: a constant is passed through switch with a true predicate, and gradients are taken
    # of each branch output with respect to the input outside any control flow context.
    # Expected: the gradient through the true branch is one and through the false branch is zero.
    # Runs in v1 graph mode. Returns nothing.
    x = constant_op.constant(2.)
    s = constant_op.constant(True)
    x_false, x_true = control_flow_ops.switch(x, s)
    grad_x_true = gradients_impl.gradients(x_true, x)[0]
    grad_x_false = gradients_impl.gradients(x_false, x)[0]
    self.assertEqual(self.evaluate(grad_x_true), 1.)
    self.assertEqual(self.evaluate(grad_x_false), 0.)


class CondTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the conditional operation: branch selection, legacy argument names, argument
  # validation, gradients, interaction with summaries and with optional-based gradient plumbing.

  def testCondTrue(self):
    # [FUNCTION DOC]
    # Scenario: predicate two less than five is true, so the first branch (two times seventeen)
    # is taken. Expected: the result equals thirty-four. Returns nothing.
    x = constant_op.constant(2)
    y = constant_op.constant(5)
    z = tf_cond.cond(
        math_ops.less(x, y), lambda: math_ops.multiply(x, 17),
        lambda: math_ops.add(y, 23))
    self.assertEqual(self.evaluate(z), 34)

  def testCondFalse(self):
    # [FUNCTION DOC]
    # Scenario: predicate two less than one is false, so the second branch (one plus twenty-three)
    # is taken. Expected: the result equals twenty-four. Returns nothing.
    x = constant_op.constant(2)
    y = constant_op.constant(1)
    z = tf_cond.cond(
        math_ops.less(x, y), lambda: math_ops.multiply(x, 17),
        lambda: math_ops.add(y, 23))
    self.assertEqual(self.evaluate(z), 24)

  def testCondTrueLegacy(self):
    # [FUNCTION DOC]
    # Same as the true-branch test but the branches are passed by the legacy keyword names for the
    # first and second function. Expected: the result equals thirty-four. Returns nothing.
    x = constant_op.constant(2)
    y = constant_op.constant(5)
    z = tf_cond.cond(
        math_ops.less(x, y),
        fn1=lambda: math_ops.multiply(x, 17),
        fn2=lambda: math_ops.add(y, 23))
    self.assertEqual(self.evaluate(z), 34)

  def testCondFalseLegacy(self):
    # [FUNCTION DOC]
    # Same as the false-branch test but with the legacy keyword names for the branch functions.
    # Expected: the result equals twenty-four. Returns nothing.
    x = constant_op.constant(2)
    y = constant_op.constant(1)
    z = tf_cond.cond(
        math_ops.less(x, y),
        fn1=lambda: math_ops.multiply(x, 17),
        fn2=lambda: math_ops.add(y, 23))
    self.assertEqual(self.evaluate(z), 24)

  @test_util.run_v1_only("Exercises Ref variables")
  def testCondModifyBoolPred(self):
    # [FUNCTION DOC]
    # Scenario (v1 only, forced onto the GPU if available): a boolean reference variable is the
    # predicate and the true branch assigns false to it. Expected: the first evaluation returns
    # false (the assigned value) and the second evaluation takes the false branch and returns
    # true. Verifies boolean reference variables can be updated on the device. Returns nothing.
    # We want to use the GPU here because we want to ensure that we can update
    # a boolean ref variable on the GPU.
    with test_util.use_gpu():
      bool_var = variable_scope.get_variable(
          "bool_var", dtype=dtypes.bool, initializer=True)
      cond_on_bool_var = tf_cond.cond(
          pred=bool_var,
          true_fn=lambda: state_ops.assign(bool_var, False),
          false_fn=lambda: True)
      self.evaluate(bool_var.initializer)
      self.assertEqual(self.evaluate(cond_on_bool_var), False)
      self.assertEqual(self.evaluate(cond_on_bool_var), True)

  def testCondMissingArg1(self):
    # [FUNCTION DOC]
    # Scenario: cond is called with only the false branch supplied by keyword. Expected: a type
    # error is raised because the true branch is missing. Returns nothing.
    x = constant_op.constant(1)
    with self.assertRaises(TypeError):
      tf_cond.cond(True, false_fn=lambda: x)

  def testCondMissingArg2(self):
    # [FUNCTION DOC]
    # Scenario: cond is called with only the true branch supplied. Expected: a type error is
    # raised because the false branch is missing. Returns nothing.
    x = constant_op.constant(1)
    with self.assertRaises(TypeError):
      tf_cond.cond(True, lambda: x)

  def testCondDuplicateArg1(self):
    # [FUNCTION DOC]
    # Scenario: the true branch is given both positionally and through the legacy keyword.
    # Expected: a type error is raised. Returns nothing.
    x = constant_op.constant(1)
    with self.assertRaises(TypeError):
      tf_cond.cond(True, lambda: x, lambda: x, fn1=lambda: x)

  def testCondDuplicateArg2(self):
    # [FUNCTION DOC]
    # Scenario: the false branch is given both positionally and through the legacy keyword.
    # Expected: a type error is raised. Returns nothing.
    x = constant_op.constant(1)
    with self.assertRaises(TypeError):
      tf_cond.cond(True, lambda: x, lambda: x, fn2=lambda: x)

  @test_util.enable_control_flow_v2
  @test_util.run_in_graph_and_eager_modes
  def testCond_gradient(self):
    # [FUNCTION DOC]
    # Scenario (control flow v2, graph and eager): two conditionals, one with a constant true
    # predicate and one with a constant false predicate, square either a one or a five. Gradients
    # are taken with an upstream gradient of three. Expected: the taken branch gives three times
    # twice the input; the untaken branch gradient is nothing in eager mode and zero in graph
    # mode. Returns nothing.
    true_in, false_in = array_ops.constant(1.), array_ops.constant(5.)
    with backprop.GradientTape(persistent=True) as tape:
      tape.watch(true_in)
      tape.watch(false_in)
      cond_true = tf_cond.cond(
          array_ops.constant(True), lambda: true_in**2., lambda: false_in**2.)
      cond_false = tf_cond.cond(
          array_ops.constant(False), lambda: true_in**2., lambda: false_in**2.)
    grads_true = tape.gradient(
        cond_true, [true_in, false_in], output_gradients=3.)
    grads_false = tape.gradient(
        cond_false, [true_in, false_in], output_gradients=3.)
    self.assertEqual(3. * 2. * 1., self.evaluate(grads_true[0]))
    self.assertEqual(None if context.executing_eagerly() else 0.,
                     self.evaluate(grads_true[1]))
    self.assertEqual(3. * 2. * 5., self.evaluate(grads_false[1]))
    self.assertEqual(None if context.executing_eagerly() else 0.,
                     self.evaluate(grads_false[0]))

  def testCondWithGroupAndSummaries(self):
    # [FUNCTION DOC]
    # Scenario: in a fresh graph with a file summary writer and always-record enabled, a
    # conditional whose true branch groups a scalar summary write and whose false branch is a
    # no-op is evaluated after initializing variables and the writer. Expected: evaluating the
    # conditional returns true and no error occurs. Writes to the test temporary directory.
    # Returns nothing.
    with ops.Graph().as_default():
      writer = summary_ops_v2.create_file_writer(self.get_temp_dir())
      with writer.as_default(), summary_ops_v2.always_record_summaries():
        op = tf_cond.cond(
            constant_op.constant(1) >= 0,
            lambda: control_flow_ops.group(summary_ops_v2.scalar("loss", 0.2)),
            control_flow_ops.no_op)
        self.evaluate(variables.global_variables_initializer())
        self.evaluate(summary_ops_v2.summary_writer_initializer_op())
        self.assertEqual(self.evaluate(op), True)

  def _makeGradient(self, use_case=False):
    # [FUNCTION DOC]
    # HELPER: builds the gradient graph used by the optionals tests.
    # Creates two variables, then under a gradient tape builds either an indexed case (when
    # use_case is true, default false) or a v2 conditional over two simple branch functions, and
    # returns the gradients of the result with respect to the two variables.
    inputs = [variables.Variable(2.0), variables.Variable(5.0)]
    with backprop.GradientTape() as tape:
      for x in inputs:
        tape.watch(x)
      f1 = lambda: math_ops.multiply(inputs[0], 17)
      f2 = lambda: math_ops.add(inputs[1], 23)
      if use_case:
        z = cond_v2.indexed_case(variables.Variable(1), [f1, f2])
      else:
        z = cond_v2.cond_v2(math_ops.less(inputs[0], inputs[1]), f1, f2)
    return tape.gradient(z, inputs)

  def testCondWithOptionals(self):
    # [FUNCTION DOC]
    # Scenario (graph mode only; returns immediately in eager mode): the gradients of a v2
    # conditional are computed with default settings. Expected: every gradient tensor's graph
    # contains optional-typed operations. Returns nothing.
    if context.executing_eagerly():
      return
    grads = self._makeGradient()
    assert all("Optional" in str(g.graph.as_graph_def()) for g in grads)

  def testCondWithoutOptionals(self):
    # [FUNCTION DOC]
    # Scenario (graph mode only): optionals are temporarily disabled in gradients, then the
    # conditional gradients are built. Expected: two gradients are returned and none of their
    # graphs contains optional-typed operations. Returns nothing.
    if context.executing_eagerly():
      return
    with context.temporarily_disable_optionals_in_gradients():
      grads = self._makeGradient()
      self.assertEqual(len(grads), 2)
      assert not any("Optional" in str(g.graph.as_graph_def()) for g in grads)

  def testCaseWithOptionals(self):
    # [FUNCTION DOC]
    # Scenario (graph mode only): same as the conditional-with-optionals test but using an
    # indexed case. Expected: all gradient graphs contain optional-typed operations. Returns
    # nothing.
    if context.executing_eagerly():
      return
    grads = self._makeGradient(use_case=True)
    assert all("Optional" in str(g.graph.as_graph_def()) for g in grads)

  def testCaseWithoutOptionals(self):
    # [FUNCTION DOC]
    # Scenario (graph mode only): optionals disabled in gradients, indexed case gradients built.
    # Expected: two gradients and none of their graphs contain optional-typed operations.
    # Returns nothing.
    if context.executing_eagerly():
      return
    with context.temporarily_disable_optionals_in_gradients():
      grads = self._makeGradient(use_case=True)
      self.assertEqual(len(grads), 2)
      assert not any("Optional" in str(g.graph.as_graph_def()) for g in grads)


class ContextTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests that control flow contexts (conditional and while loop) survive a round trip through
  # their protocol buffer representation, including import scope handling.

  @test_util.run_deprecated_v1
  def testCondContext(self):
    # [FUNCTION DOC]
    # Scenario: build a conditional in a graph and for every operation that has a control flow
    # context convert the context to a proto, rebuild it from the proto and convert again.
    # Expected: both protos are equal. Returns nothing.
    with self.cached_session() as sess:
      x = constant_op.constant(2)
      y = constant_op.constant(5)
      tf_cond.cond(
          math_ops.less(x, y), lambda: math_ops.multiply(x, 17),
          lambda: math_ops.add(y, 23))
      for op in sess.graph.get_operations():
        c = op._get_control_flow_context()
        if c:
          self.assertProtoEquals(
              c.to_proto(),
              control_flow_ops.CondContext.from_proto(c.to_proto()).to_proto())

  def _testWhileContextHelper(self, maximum_iterations=None):
    # [FUNCTION DOC]
    # HELPER: shared body of the while context round trip tests.
    # Builds a while loop counting to ten, with an optional maximum iterations value (default
    # none), and checks every operation's control flow context survives a proto round trip
    # unchanged. Returns nothing.
    with self.cached_session() as sess:
      i = constant_op.constant(0)
      c = lambda i: math_ops.less(i, 10)
      b = lambda i: math_ops.add(i, 1)
      while_loop.while_loop(c, b, [i], maximum_iterations=maximum_iterations)
      for op in sess.graph.get_operations():
        control_flow_context = op._get_control_flow_context()
        if control_flow_context:
          self.assertProtoEquals(
              control_flow_context.to_proto(),
              control_flow_ops.WhileContext.from_proto(
                  control_flow_context.to_proto()).to_proto())

  @test_util.run_deprecated_v1
  def testWhileContext(self):
    # [FUNCTION DOC]
    # Runs the while context round trip check without a maximum iterations value. Returns nothing.
    self._testWhileContextHelper()

  @test_util.run_deprecated_v1
  def testWhileContextWithMaximumIterations(self):
    # [FUNCTION DOC]
    # Runs the while context round trip check with a maximum iterations value of ten.
    # Returns nothing.
    self._testWhileContextHelper(maximum_iterations=10)

  @test_util.run_deprecated_v1
  def testControlContextImportScope(self):
    # [FUNCTION DOC]
    # Scenario: a minimal concrete control flow context is given a set of value names and an
    # external value mapping, serialized to a values definition, and reloaded with an import scope
    # of test_scope. Expected: the reloaded context has the scope prefixed to its value names and
    # external value keys (mapped to the correspondingly scoped tensors), and exporting it back
    # with the same export scope removes the prefix, giving a definition equal to the original.
    # Returns nothing.

    class NoABCControlFlowContext(control_flow_ops.ControlFlowContext):
      # [CLASS DOC]
      # HELPER: test-only concrete subclass of the abstract control flow context base class,
      # defined only so the base class behavior can be instantiated. It adds no behavior; its only
      # override forwards to the base implementation.
      """A noop wrapper around `ControlFlowContext`.

      `ControlFlowContext` is an ABC and therefore cannot be instantiated.
      """

      # pylint: disable=useless-super-delegation

      def to_control_flow_context_def(self, context_def, export_scope=None):
        # [FUNCTION DOC]
        # Forwards to the base class serialization of the context into the given context
        # definition, passing the optional export scope. Returns nothing (the base result is
        # discarded).
        super(NoABCControlFlowContext,
              self).to_control_flow_context_def(context_def, export_scope)

    with self.cached_session():
      constant_op.constant(0, name="a")
      constant_op.constant(2, name="test_scope/a")
      b1 = constant_op.constant(1, name="b")
      b2 = constant_op.constant(3, name="test_scope/b")

      c = NoABCControlFlowContext()
      c._values = ["a", "b"]
      c._external_values = {"a": b1}

      c_with_scope = NoABCControlFlowContext(
          values_def=c._to_values_def(), import_scope="test_scope")

      # _values and _external_values should be have scope prepended.
      self.assertEqual(c_with_scope._values,
                       set(["test_scope/a", "test_scope/b"]))
      self.assertEqual(c_with_scope._external_values, {"test_scope/a": b2})

      # Calling _to_proto() with export_scope should remove "test_scope".
      self.assertProtoEquals(
          c._to_values_def(),
          c_with_scope._to_values_def(export_scope="test_scope"))


def _get_nested_shape(nested):
  # [FUNCTION DOC]
  # HELPER: shape extraction for the data type tests.
  # Maps over a nested structure of tensors and returns a structure of the same layout holding,
  # for each leaf, its static shape. Tensor arrays are replaced by the tensor array type itself
  # and indexed slices by their dense shape.

  def _get_shape(tensor):
    # [FUNCTION DOC]
    # HELPER: per-leaf shape rule for the nested shape extraction. Returns the tensor array type
    # for tensor arrays, the dense shape for indexed slices, and the static shape otherwise.
    if isinstance(tensor, tensor_array_ops.TensorArray):
      return tensor_array_ops.TensorArray
    elif isinstance(tensor, indexed_slices.IndexedSlices):
      return tensor.dense_shape
    else:
      return tensor.get_shape()

  return nest.map_structure(_get_shape, nested)


def _create_tensor_array(size, shape):
  # [FUNCTION DOC]
  # HELPER: builds a tensor array for the data type tests.
  # Creates a float32 tensor array of the given size that does not clear elements after reading,
  # and fills every slot with a zero tensor of the given element shape. Returns the array.
  ta = tensor_array_ops.TensorArray(
      dtype=dtypes.float32, size=size, clear_after_read=False)
  for i in range(size):
    ta = ta.write(i, array_ops.zeros(shape))
  return ta


def _raw_nested_shape(nested_shape):
  # [FUNCTION DOC]
  # HELPER: converts nested shape structures to plain comparable lists for the data type tests.
  # Maps over a nested structure of shapes and returns a structure of the same layout in which
  # each tensor shape of known rank becomes a list of its dimension sizes (unknown sizes become
  # nothing) and anything else, including unknown rank shapes, becomes nothing.

  def _raw_shape(shape):
    # [FUNCTION DOC]
    # HELPER: per-leaf rule. Returns a list of dimension sizes for a tensor shape with known
    # rank, otherwise nothing.
    if isinstance(shape, tensor_shape.TensorShape) and shape.ndims is not None:
      return [x.value for x in shape.dims]
    else:
      return None

  return nest.map_structure(_raw_shape, nested_shape)


# TODO(yori): Add tests for indexed slices.
class DataTypesTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests that conditional and case operations handle many kinds of branch outputs (scalars,
  # strings, variables, tensors with known, unknown and partial shapes, sparse tensors, tensor
  # arrays, lists, tuples, named tuples and nested structures) in both strict and non-strict
  # mode, checking the inferred static shapes and the evaluated values. Shared helpers build and
  # verify both a conditional and a case operation for the same pair of branch functions.

  def assertAllEqualNested(self, a, b):
    # [FUNCTION DOC]
    # HELPER: assertion helper that compares possibly nested results.
    # If the first argument is a list or tuple, compares it element by element (recursively) with
    # the second argument; otherwise asserts the two values are equal as arrays. Fails the test on
    # any mismatch. Returns nothing.
    if isinstance(a, (list, tuple)):
      for entry_a, entry_b in zip(a, b):
        self.assertAllEqualNested(entry_a, entry_b)
    else:
      self.assertAllEqual(a, b)

  def _testShape(self, fn_true, fn_false, expected_shape, strict=False):
    # [FUNCTION DOC]
    # HELPER: shared shape check for the data type tests.
    # Builds a conditional and a case operation on a boolean placeholder using the two branch
    # functions and the strict flag (default false), and asserts that the static shapes of both
    # outputs (as plain nested lists) equal those of the expected shape structure. Returns nothing.
    condition = array_ops.placeholder(dtypes.bool)
    output_cond = tf_cond.cond(
        condition, fn_true, fn_false, strict=strict)
    self.assertEqual(
        _raw_nested_shape(_get_nested_shape(output_cond)),
        _raw_nested_shape(expected_shape))

    output_case = control_flow_case.case([(condition, fn_true)],
                                         fn_false,
                                         strict=strict)
    self.assertEqual(
        _raw_nested_shape(_get_nested_shape(output_case)),
        _raw_nested_shape(expected_shape))

  def _testReturnValues(self,
                        fn_true,
                        fn_false,
                        expected_value_true,
                        expected_value_false,
                        strict=False,
                        check_cond=True,
                        feed_dict=None):
    # [FUNCTION DOC]
    # HELPER: shared value check for the data type tests.
    # Builds a conditional and a case operation on a boolean placeholder from the two branch
    # functions (strict flag default false), initializes variables, then runs them once with the
    # condition true and once false, merging in extra feed values (feed_dict, optional, default
    # empty). Asserts the conditional results equal the expected true and false values; the case
    # results are also checked unless check_cond is false (default true). Returns nothing.
    if feed_dict is None:
      feed_dict = {}

    condition = array_ops.placeholder(dtypes.bool)
    output_cond = tf_cond.cond(
        condition, fn_true, fn_false, strict=strict)
    output_case = control_flow_case.case([(condition, fn_true)],
                                         fn_false,
                                         strict=strict)

    with self.cached_session() as sess:
      self.evaluate(variables.global_variables_initializer())
      true_feed_dict = {condition: True}
      true_feed_dict.update(feed_dict)
      result_cond, result_case = sess.run([output_cond, output_case],
                                          feed_dict=true_feed_dict)
      self.assertAllEqualNested(result_cond, expected_value_true)
      if check_cond:
        self.assertAllEqualNested(result_case, expected_value_true)
      false_feed_dict = {condition: False}
      false_feed_dict.update(feed_dict)
      result_cond, result_case = sess.run([output_cond, output_case],
                                          feed_dict=false_feed_dict)
      self.assertAllEqualNested(result_cond, expected_value_false)
      if check_cond:
        self.assertAllEqualNested(result_case, expected_value_false)

  @test_util.run_deprecated_v1
  def test_int(self):
    # [FUNCTION DOC]
    # Scenario: branches return plain integers one and two. Expected: scalar static shape and
    # values one and two, in both non-strict and strict mode. Returns nothing.
    shape = tensor_shape.TensorShape([])
    fn_true = lambda: 1
    fn_false = lambda: 2
    self._testShape(fn_true, fn_false, shape)
    self._testReturnValues(fn_true, fn_false, 1, 2)
    self._testShape(fn_true, fn_false, shape, strict=True)
    self._testReturnValues(fn_true, fn_false, 1, 2, strict=True)

  @test_util.run_deprecated_v1
  def test_float(self):
    # [FUNCTION DOC]
    # Scenario: branches return plain floats one and two. Expected: scalar shape and the matching
    # values (non-strict mode). Returns nothing.
    shape = tensor_shape.TensorShape([])
    fn_true = lambda: 1.0
    fn_false = lambda: 2.0
    self._testShape(fn_true, fn_false, shape)
    self._testReturnValues(fn_true, fn_false, 1.0, 2.0)

  @test_util.run_deprecated_v1
  def test_noop(self):
    # [FUNCTION DOC]
    # Scenario: both branches are the no-op operation. Expected: the output has unknown shape and
    # the conditional evaluates to true and false for the respective conditions (the case output is
    # not value-checked). Returns nothing.
    shape = tensor_shape.TensorShape(None)
    self._testShape(control_flow_ops.no_op, control_flow_ops.no_op, shape)
    self._testReturnValues(
        control_flow_ops.no_op,
        control_flow_ops.no_op,
        True,
        False,
        check_cond=False)

  @test_util.run_deprecated_v1
  def test_string(self):
    # [FUNCTION DOC]
    # Scenario: branches return the strings abc and xyz. Expected: scalar shape and byte string
    # values abc and xyz for true and false. Returns nothing.
    shape = tensor_shape.TensorShape([])
    fn_true = lambda: "abc"
    fn_false = lambda: "xyz"
    self._testShape(fn_true, fn_false, shape)
    self._testReturnValues(fn_true, fn_false, b"abc", b"xyz")

  @test_util.run_v1_only("b/138741991")
  def test_variable(self):
    # [FUNCTION DOC]
    # Scenario (v1 only): branches create variables with initial values three and four. Expected:
    # scalar shape and values three and four. Returns nothing.
    shape = tensor_shape.TensorShape([])
    fn_true = lambda: variables.Variable(3.0)
    fn_false = lambda: variables.Variable(4.0)
    self._testShape(fn_true, fn_false, shape)
    self._testReturnValues(fn_true, fn_false, 3.0, 4.0)

  @test_util.run_v1_only("b/120553181")
  def test_none(self):
    # [FUNCTION DOC]
    # Scenario (v1 only): one branch returns nothing and the other returns a tensor, in either
    # order. Expected: a value error is raised both times. Returns nothing.
    fn_none = lambda: None
    fn_tensor = lambda: constant_op.constant(1)

    with self.assertRaises(ValueError):
      tf_cond.cond(constant_op.constant(True), fn_none, fn_tensor)

    with self.assertRaises(ValueError):
      tf_cond.cond(constant_op.constant(True), fn_tensor, fn_none)

  @test_util.run_deprecated_v1
  def test_tensors(self):
    # [FUNCTION DOC]
    # Scenario: for several dtypes (half float, int8, int32, uint8) the true branch returns a pair
    # of zeros (2x2) and ones (3x3) and the false branch the opposite. Expected: static shapes
    # (2x2 and 3x3) and the matching numeric values for both conditions. Returns nothing.

    def _build_true_branch(dtype):
      # [FUNCTION DOC]
      # Returns a branch function producing a 2x2 zeros tensor and a 3x3 ones tensor of the
      # given dtype.

      def _build():
        # [FUNCTION DOC]
        # Branch function: returns the pair of zeros (2x2) and ones (3x3) tensors.
        return (array_ops.zeros([2, 2],
                                dtype=dtype), array_ops.ones([3, 3],
                                                             dtype=dtype))

      return _build

    def _build_false_branch(dtype):
      # [FUNCTION DOC]
      # Returns a branch function producing a 2x2 ones tensor and a 3x3 zeros tensor of the
      # given dtype.

      def _build():
        # [FUNCTION DOC]
        # Branch function: returns the pair of ones (2x2) and zeros (3x3) tensors.
        return (array_ops.ones([2, 2],
                               dtype=dtype), array_ops.zeros([3, 3],
                                                             dtype=dtype))

      return _build

    for dtype in (dtypes.float16, dtypes.int8, dtypes.int32, dtypes.uint8):
      shape = (tensor_shape.TensorShape([2,
                                         2]), tensor_shape.TensorShape([3, 3]))
      fn_true = _build_true_branch(dtype)
      fn_false = _build_false_branch(dtype)
      self._testShape(fn_true, fn_false, shape)
      self._testReturnValues(fn_true, fn_false,
                             (np.zeros([2, 2]), np.ones([3, 3])),
                             (np.ones([2, 2]), np.zeros([3, 3])))

  @test_util.run_deprecated_v1
  def test_tensors_unknown_shape(self):
    # [FUNCTION DOC]
    # Scenario: for several dtypes each branch returns a placeholder of unknown shape. Expected:
    # the output shape is unknown and, when fed zeros (true) or ones (false) of shape 2x2, the
    # outputs equal the fed values. Returns nothing.

    def _build_true_branch(dtype):
      # [FUNCTION DOC]
      # Creates an unknown-shape placeholder of the dtype and returns a branch function returning
      # it, together with the placeholder (so it can be fed).
      tensor = array_ops.placeholder(dtype=dtype, shape=None)

      def _build():
        # [FUNCTION DOC]
        # Branch function: returns the captured placeholder.
        return tensor

      return _build, tensor

    def _build_false_branch(dtype):
      # [FUNCTION DOC]
      # Same as the true-branch builder: creates an unknown-shape placeholder and returns a branch
      # function returning it, plus the placeholder.
      tensor = array_ops.placeholder(dtype=dtype, shape=None)

      def _build():
        # [FUNCTION DOC]
        # Branch function: returns the captured placeholder.
        return tensor

      return _build, tensor

    for dtype in (dtypes.float16, dtypes.int8, dtypes.int32, dtypes.uint8):
      shape = tensor_shape.TensorShape(None)
      fn_true, true_tensor = _build_true_branch(dtype)
      fn_false, false_tensor = _build_false_branch(dtype)
      self._testShape(fn_true, fn_false, shape)
      self._testReturnValues(
          fn_true,
          fn_false,
          np.zeros([2, 2]),
          np.ones([2, 2]),
          feed_dict={
              true_tensor: np.zeros([2, 2]),
              false_tensor: np.ones([2, 2])
          })

  @test_util.run_deprecated_v1
  def test_sparse_tensors(self):
    # [FUNCTION DOC]
    # Scenario: branches return a one-element list holding a 3x4 sparse tensor with different
    # indices and values. Expected: the dense shape is 3x4 and the evaluated sparse values match
    # the corresponding branch; the non-strict form is only checked when the v2 behavior is not
    # enabled, the strict form (list-wrapped) always. Returns nothing.
    shape = tensor_shape.TensorShape([3, 4])

    def true_fn():
      # [FUNCTION DOC]
      # True branch: returns a list with one sparse tensor with entries at (0,0) and (1,2).
      return [
          sparse_tensor.SparseTensor(
              indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])
      ]

    def false_fn():
      # [FUNCTION DOC]
      # False branch: returns a list with one sparse tensor with entries at (0,0) and (2,1).
      return [
          sparse_tensor.SparseTensor(
              indices=[[0, 0], [2, 1]], values=[3, 4], dense_shape=[3, 4])
      ]

    value1 = sparse_tensor.SparseTensorValue(
        indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])
    value2 = sparse_tensor.SparseTensorValue(
        indices=[[0, 0], [2, 1]], values=[3, 4], dense_shape=[3, 4])
    # Non-strict cond is only available in v1
    if not tf2.enabled():
      self._testShape(true_fn, false_fn, shape)
      self._testReturnValues(true_fn, false_fn, value1, value2)
    self._testShape(true_fn, false_fn, [shape], strict=True)
    self._testReturnValues(true_fn, false_fn, [value1], [value2], strict=True)

  @test_util.run_deprecated_v1
  def test_tensors_with_partially_specified_shapes(self):
    # [FUNCTION DOC]
    # Scenario: for several dtypes both branches return three placeholders whose shapes are
    # partially unknown. Expected: the static output shapes keep the partial specification and the
    # evaluated values equal the fed arrays. Returns nothing.

    def _build_branch(dtype, shape):
      # [FUNCTION DOC]
      # Creates three placeholders of the dtype with the three given shapes and returns a branch
      # function returning them as a tuple, plus the tuple of placeholders for feeding.
      a = array_ops.placeholder(dtype=dtype, shape=shape[0])
      b = array_ops.placeholder(dtype=dtype, shape=shape[1])
      c = array_ops.placeholder(dtype=dtype, shape=shape[2])

      def _build():
        # [FUNCTION DOC]
        # Branch function: returns the three captured placeholders as a tuple.
        return a, b, c

      return _build, (a, b, c)

    for dtype in (dtypes.float16, dtypes.int8, dtypes.int32, dtypes.uint8):
      shape = (tensor_shape.TensorShape([None,
                                         2]), tensor_shape.TensorShape([None]),
               tensor_shape.TensorShape([3, None]))
      fn_true, true_tensors = _build_branch(dtype, shape)
      fn_false, false_tensors = _build_branch(dtype, shape)
      self._testShape(fn_true, fn_false, shape)
      self._testReturnValues(
          fn_true,
          fn_false, (np.zeros([2, 2]), np.zeros(5), np.ones([3, 3])),
          (np.zeros([2, 2]), np.zeros(5), np.ones([3, 3])),
          feed_dict={
              true_tensors[0]: np.zeros([2, 2]),
              false_tensors[0]: np.zeros([2, 2]),
              true_tensors[1]: np.zeros([5]),
              false_tensors[1]: np.zeros([5]),
              true_tensors[2]: np.ones([3, 3]),
              false_tensors[2]: np.ones([3, 3])
          })

  @test_util.run_deprecated_v1
  def test_tensor_arrays(self):
    # [FUNCTION DOC]
    # Scenario: branches return two pre-filled tensor arrays. Expected: the inferred output type is
    # the tensor array type (only the shape check is performed). Returns nothing.
    element_shape = tensor_shape.TensorShape([2])
    ta1 = _create_tensor_array(4, element_shape)
    ta2 = _create_tensor_array(4, element_shape)
    shape = tensor_array_ops.TensorArray
    fn_true = lambda: ta1
    fn_false = lambda: ta2
    self._testShape(fn_true, fn_false, shape)

  @test_util.run_deprecated_v1
  def test_tensor_array_reads(self):
    # [FUNCTION DOC]
    # Scenario: branches read different elements of a filled tensor array. Expected: the output
    # shape equals the element shape of two (only the shape check is performed). Returns nothing.
    shape = tensor_shape.TensorShape([2])
    ta = _create_tensor_array(4, shape)
    fn_true = lambda: ta.read(0)
    fn_false = lambda: ta.read(1)
    self._testShape(fn_true, fn_false, shape)

  @test_util.run_v1_only("b/138741991")
  def test_list(self):
    # [FUNCTION DOC]
    # Scenario (v1 only): branches return lists mixing a tensor, a Python integer and a variable.
    # Expected: a list of three scalar shapes and values [1, 2, 3.0] or [3, 4, 5.0]. Returns
    # nothing.
    shape = [
        tensor_shape.TensorShape([]),
        tensor_shape.TensorShape([]),
        tensor_shape.TensorShape([])
    ]
    fn_true = lambda: [constant_op.constant(1), 2, variables.Variable(3.0)]
    fn_false = lambda: [constant_op.constant(3), 4, variables.Variable(5.0)]
    self._testShape(fn_true, fn_false, shape)
    self._testReturnValues(fn_true, fn_false, [1, 2, 3.0], [3, 4, 5.0])

  @test_util.run_v1_only("Non-strict cond is only available in v1")
  def test_non_strict(self):
    # [FUNCTION DOC]
    # Scenario (v1 only): in non-strict mode branches may return a tensor, a one-element list or a
    # one-element tuple interchangeably. Expected: all pairings yield a scalar shape and the values
    # of the single elements (1 versus 2, 1 versus 3, 2 versus 3). Returns nothing.
    shape = tensor_shape.TensorShape([])
    fn_tensor = lambda: constant_op.constant(1)
    fn_list = lambda: [constant_op.constant(2)]
    fn_tuple = lambda: (constant_op.constant(3),)
    self._testShape(fn_tensor, fn_list, shape)
    self._testShape(fn_tensor, fn_tuple, shape)
    self._testShape(fn_list, fn_tuple, shape)
    self._testReturnValues(fn_tensor, fn_list, 1, 2)
    self._testReturnValues(fn_tensor, fn_tuple, 1, 3)
    self._testReturnValues(fn_list, fn_tuple, 2, 3)

  @test_util.run_v1_only("b/120553181")
  def test_singleton_strict(self):
    # [FUNCTION DOC]
    # Scenario (v1 only): in strict mode, mixing a tensor with a list branch raises a value error
    # and mixing a list with a tuple raises a type error, for both cond and case. Returns nothing.
    fn_tensor = lambda: constant_op.constant(1)
    fn_list = lambda: [constant_op.constant(2)]
    fn_tuple = lambda: (constant_op.constant(3),)

    with self.assertRaises(ValueError):
      tf_cond.cond(
          constant_op.constant(True), fn_tensor, fn_list, strict=True)

    with self.assertRaises(TypeError):
      tf_cond.cond(
          constant_op.constant(True), fn_list, fn_tuple, strict=True)

    with self.assertRaises(ValueError):
      control_flow_case.case([(constant_op.constant(True), fn_tensor)],
                             fn_list,
                             strict=True)

    with self.assertRaises(TypeError):
      control_flow_case.case([(constant_op.constant(True), fn_list)],
                             fn_tuple,
                             strict=True)

  @test_util.run_deprecated_v1
  def test_singleton_list(self):
    # [FUNCTION DOC]
    # Scenario: branches return one-element lists. Expected: in non-strict mode (when v2 is not
    # enabled) the result is unwrapped to a scalar; in strict mode the list structure is kept.
    # Shapes and values are verified. Returns nothing.
    shape = tensor_shape.TensorShape([])
    fn_true = lambda: [constant_op.constant(1)]
    fn_false = lambda: [constant_op.constant(3)]
    # Non-strict cond is only available in v1
    if not tf2.enabled():
      self._testShape(fn_true, fn_false, shape)
      self._testReturnValues(fn_true, fn_false, 1, 3)
    self._testShape(fn_true, fn_false, [shape], strict=True)
    self._testReturnValues(fn_true, fn_false, [1], [3], strict=True)

  @test_util.run_deprecated_v1
  def test_singleton_tuple(self):
    # [FUNCTION DOC]
    # Scenario: branches return one-element tuples. Expected: non-strict mode (when v2 is not
    # enabled) unwraps to a scalar; strict mode keeps the tuple structure. Returns nothing.
    shape = tensor_shape.TensorShape([])
    fn_true = lambda: (constant_op.constant(1),)
    fn_false = lambda: (constant_op.constant(3),)
    # Non-strict cond is only available in v1
    if not tf2.enabled():
      self._testShape(fn_true, fn_false, shape)
      self._testReturnValues(fn_true, fn_false, 1, 3)
    self._testShape(fn_true, fn_false, (shape,), strict=True)
    self._testReturnValues(fn_true, fn_false, (1,), (3,), strict=True)

  @test_util.run_deprecated_v1
  def test_singleton_namedtuple(self):
    # [FUNCTION DOC]
    # Scenario: branches return a one-field named tuple. Expected: non-strict mode (when v2 is not
    # enabled) unwraps to a scalar; strict mode keeps the named tuple structure in shapes and
    # values. Returns nothing.
    shape = tensor_shape.TensorShape([])
    fn_true = lambda: SingletonTestTuple(constant_op.constant(1))
    fn_false = lambda: SingletonTestTuple(constant_op.constant(3))
    # Non-strict cond is only available in v1
    if not tf2.enabled():
      self._testShape(fn_true, fn_false, shape)
      self._testReturnValues(fn_true, fn_false, 1, 3)
    self._testShape(fn_true, fn_false, SingletonTestTuple(shape), strict=True)
    self._testReturnValues(
        fn_true,
        fn_false,
        SingletonTestTuple(1),
        SingletonTestTuple(3),
        strict=True)

  @test_util.run_deprecated_v1
  def test_tuple(self):
    # [FUNCTION DOC]
    # Scenario: branches return two-element tuples mixing a tensor and an integer. Expected:
    # tuple of two scalar shapes and values (1, 2) or (3, 4). Returns nothing.
    shape = (tensor_shape.TensorShape([]), tensor_shape.TensorShape([]))
    fn_true = lambda: (constant_op.constant(1), 2)
    fn_false = lambda: (constant_op.constant(3), 4)
    self._testShape(fn_true, fn_false, shape)
    self._testReturnValues(fn_true, fn_false, (1, 2), (3, 4))

  @test_util.run_deprecated_v1
  def test_namedtuple(self):
    # [FUNCTION DOC]
    # Scenario: branches return a two-field named tuple mixing a tensor and an integer. Expected:
    # the structure and values are preserved. Returns nothing.
    shape = TestTuple(
        tensor_shape.TensorShape([]), tensor_shape.TensorShape([]))
    fn_true = lambda: TestTuple(constant_op.constant(1), 2)
    fn_false = lambda: TestTuple(constant_op.constant(3), 4)
    self._testShape(fn_true, fn_false, shape)
    self._testReturnValues(fn_true, fn_false, TestTuple(1, 2), TestTuple(3, 4))

  @test_util.run_deprecated_v1
  def test_nested(self):
    # [FUNCTION DOC]
    # Scenario: branches return a list containing a scalar, a named tuple holding a scalar and a
    # list, a 5x5 zeros or ones tensor and an integer. Expected: the nested static shapes and
    # values are preserved for both conditions. Returns nothing.
    shape = [
        tensor_shape.TensorShape([]),
        TestTuple(
            tensor_shape.TensorShape([]),
            [tensor_shape.TensorShape([]),
             tensor_shape.TensorShape([])]),
        tensor_shape.TensorShape([5, 5]),
        tensor_shape.TensorShape([])
    ]

    def true_fn():
      # [FUNCTION DOC]
      # True branch: returns the nested structure with values 1, (2, [3, 4]), zeros 5x5 and 6.
      return [
          constant_op.constant(1),
          TestTuple(constant_op.constant(2), [3, 4]),
          array_ops.zeros([5, 5]), 6
      ]

    def false_fn():
      # [FUNCTION DOC]
      # False branch: returns the nested structure with values 11, (12, [13, 14]), ones 5x5, 16.
      return [
          constant_op.constant(11),
          TestTuple(constant_op.constant(12), [13, 14]),
          array_ops.ones([5, 5]), 16
      ]

    self._testShape(true_fn, false_fn, shape)
    self._testReturnValues(
        true_fn, false_fn,
        [1, TestTuple(2, [3, 4]), np.zeros([5, 5]), 6],
        [11, TestTuple(12, [13, 14]),
         np.ones([5, 5]), 16])

  @test_util.run_deprecated_v1
  def test_cond_inside_while_loop(self):
    # [FUNCTION DOC]
    # Scenario: a while loop of ten iterations whose body runs a conditional returning a named
    # tuple of scaled matrices plus the matrix. Expected: the loop outputs keep a scalar counter
    # shape and a 2x2 matrix shape. Returns nothing.

    def body(i, matrix):
      # [FUNCTION DOC]
      # Loop body: a conditional (constant true predicate) returns a named tuple of the matrix
      # scaled by two and four (swapped in the other branch) plus the matrix. Returns the
      # incremented counter and the first field of the tuple.
      result_tuple, unused_matrix = tf_cond.cond(
          constant_op.constant(True), lambda:
          (TestTuple(matrix * 2, matrix * 4), matrix), lambda:
          (TestTuple(matrix * 4, matrix * 2), matrix))
      return [i + 1, result_tuple.a]

    iteration, matrix = while_loop.while_loop(
        lambda i, matrix: i < 10,
        body,
        loop_vars=[constant_op.constant(0),
                   array_ops.ones([2, 2])])

    self.assertEqual(iteration.get_shape(), tensor_shape.TensorShape([]))
    self.assertEqual(matrix.get_shape(), tensor_shape.TensorShape([2, 2]))


@test_util.run_all_in_graph_and_eager_modes
class IndexedCaseTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized tests for the index-based switch case operation: branch selection by integer
  # index, default branches for out-of-range indices, dictionary branches, and gradients with
  # and without switch/merge lowering. Runs in both graph and eager modes.

  def make_name(self):
    # [FUNCTION DOC]
    # HELPER: derives a legal operation name from the current test id.
    # Takes the last dot-separated part of the test id, replaces opening parentheses and spaces by
    # underscores and removes closing parentheses. Returns the resulting string.
    name = self.id().split(".")[-1].replace("(", "_").replace(")", "")
    return name.replace(" ", "_")

  def disabled_testCase_ticklesGpuVsHostMemoryIssueWithInt32(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name does not start with the test prefix, so the test runner never
    # collects or runs it; it is effectively disabled dead code.
    # Intended scenario: for each of five branches that return int32 constants (ten times the
    # branch number), evaluate switch case with that index and expect the matching value. Checks
    # that int32 results work across host and device memory. Returns nothing.
    nbranches = 5

    def make_func(bi):
      # [FUNCTION DOC]
      # Returns a branch function that produces an int-typed constant ten times bi.
      return lambda: array_ops.constant(bi * 10, name="br{}_out".format(bi))

    branches = [(i, make_func(i)) for i in range(nbranches)]
    for bi in range(nbranches):
      branch_index = array_ops.placeholder_with_default(bi, [])
      case_out = control_flow_switch_case.switch_case(branch_index, branches)
      self.assertEqual(bi * 10, self.evaluate(case_out))

  @parameterized.parameters((0,), (2,), (3,))
  def testCase(self, bi):
    # [FUNCTION DOC]
    # Parameterized by the branch index bi (0, 2 or 3). Five branches return float constants of
    # ten times their index; switch case with the index is evaluated. Expected: result equals ten
    # times bi. Returns nothing.
    nbranches = 5

    def make_func(bi):
      # [FUNCTION DOC]
      # Returns a branch function producing a float constant equal to ten times bi.
      return lambda: array_ops.constant(bi * 10., name="br{}_out".format(bi))

    branches = [(i, make_func(i)) for i in range(nbranches)]
    branch_index = array_ops.placeholder_with_default(bi, [])
    case_out = control_flow_switch_case.switch_case(
        branch_index, branches, name=self.make_name())
    self.assertEqual(bi * 10., self.evaluate(case_out))

  @parameterized.parameters((-1,), (2,), (4,), (5,), (6,))
  def testCase_withDefault(self, bi):
    # [FUNCTION DOC]
    # Parameterized by index bi (negative, in range, or beyond the range). Five list-style branches
    # plus a default branch returning sixty. Expected: out-of-range indices (negative or at least
    # five) yield sixty, in-range ones yield ten times bi. Returns nothing.
    nbranches = 5

    def make_func(bi):
      # [FUNCTION DOC]
      # Returns a branch function producing a float constant equal to ten times bi.
      return lambda: array_ops.constant(bi * 10., name="br{}_out".format(bi))

    branches = [(i, make_func(i)) for i in range(nbranches)]
    branch_index = array_ops.placeholder_with_default(bi, [])
    case_out = control_flow_switch_case.switch_case(
        branch_index, branches, default=make_func(6), name=self.make_name())
    if bi < 0 or bi >= nbranches:
      expected = 60.
    else:
      expected = bi * 10.
    self.assertEqual(expected, self.evaluate(case_out))

  @parameterized.parameters((-1,), (0,), (3,), (5,))
  def testCase_dictWithDefault(self, bi):
    # [FUNCTION DOC]
    # Same as the default-branch test but the branches are given as a mapping from index to
    # function. Expected: sixty for out-of-range indices, otherwise ten times bi. Returns nothing.
    nbranches = 5

    def make_func(bi):
      # [FUNCTION DOC]
      # Returns a branch function producing a float constant equal to ten times bi.
      return lambda: array_ops.constant(bi * 10., name="br{}_out".format(bi))

    branches = {i: make_func(i) for i in range(nbranches)}
    branch_index = array_ops.placeholder_with_default(bi, [])
    case_out = control_flow_switch_case.switch_case(
        branch_index, branches, default=make_func(6), name=self.make_name())
    if bi < 0 or bi >= nbranches:
      expected = 60.
    else:
      expected = bi * 10.
    self.assertEqual(expected, self.evaluate(case_out))

  @parameterized.parameters((-1,), (1,), (4,), (5,))
  def testCase_gradient_disable_lowering(self, bi):
    # [FUNCTION DOC]
    # Parameterized by index bi. Runs the shared gradient check with switch/merge lowering
    # disabled. Returns nothing.
    self._testCase_gradient(True, bi)

  @parameterized.parameters((-1,), (1,), (4,), (5,))
  def testCase_gradient_enable_lowering(self, bi):
    # [FUNCTION DOC]
    # Parameterized by index bi. Runs the shared gradient check with switch/merge lowering left
    # enabled. Returns nothing.
    self._testCase_gradient(False, bi)

  def _testCase_gradient(self, disable_lowering, bi):
    # [FUNCTION DOC]
    # HELPER: shared gradient check for the lowering tests.
    # Sets the module-wide switch controlling whether case lowering to switch/merge is disabled
    # (side effect on global state of the control flow utility module), builds five branches that
    # square separate float inputs, takes the gradient of the case output with an upstream gradient
    # of three, and verifies that only the used branch (the last one for out-of-range indices) has
    # a gradient equal to the upstream gradient times twice its input, while the others are nothing
    # in eager mode or zero in graph mode. Restores the previous global setting at the end only if
    # no assertion failed before. Returns nothing.
    default_lowering = control_flow_util_v2._DISABLE_LOWER_USING_SWITCH_MERGE
    control_flow_util_v2._DISABLE_LOWER_USING_SWITCH_MERGE = disable_lowering
    nbranches = 5
    inputs = [
        array_ops.constant(float(bi), name="br{}_in".format(bi))
        for bi in range(nbranches)
    ]

    def make_func(bi):
      # [FUNCTION DOC]
      # Returns a branch function that squares the bi-th input.
      return lambda: inputs[bi]**2.

    branches = {bi: make_func(bi) for bi in range(nbranches)}

    branch_index = array_ops.placeholder_with_default(bi, [])
    with backprop.GradientTape() as tape:
      for x in inputs:
        tape.watch(x)
      case_out = control_flow_switch_case.switch_case(branch_index, branches)
    out_grad = 3.
    actual_grads = tape.gradient(case_out, inputs, output_gradients=out_grad)
    expected_grads = [None if context.executing_eagerly() else 0.] * nbranches
    used_branch_idx = nbranches - 1 if bi < 0 or bi >= nbranches - 1 else bi
    expected_grads[used_branch_idx] = out_grad * 2. * used_branch_idx
    self.assertEqual(len(expected_grads), len(actual_grads))
    for expected, actual in zip(expected_grads, actual_grads):
      self.assertEqual(expected, self.evaluate(actual))
    # reset to default value
    control_flow_util_v2._DISABLE_LOWER_USING_SWITCH_MERGE = default_lowering

  @parameterized.parameters((-2,), (2,), (5,))
  def testCase_gradient_diffShapedIntermediates(self, bi):
    # [FUNCTION DOC]
    # Parameterized by index bi. Five branches take vector inputs of different lengths and compute
    # a scalar through intermediates of different shapes. The gradient of the case output with an
    # upstream gradient of three is compared, for the used branch, with the gradient of directly
    # calling that branch under its own tape, and for the other inputs with nothing (eager) or
    # zeros (graph). Returns nothing.
    nbranches = 5
    inputs = [
        array_ops.constant(
            float(bi), shape=[bi + 1], name="br{}_in".format(bi))
        for bi in range(nbranches)
    ]

    def make_func(bi):
      # [FUNCTION DOC]
      # Returns a branch function computing the sum of the squared bi-th input times its leading
      # slice, a scalar built from intermediates of branch-specific shapes.

      def f():
        # [FUNCTION DOC]
        # Branch function: multiplies the squared input elementwise (broadcast) by the leading
        # slice of the input and returns the total sum.
        x = inputs[bi]**2 * inputs[bi][:bi + 1, None]
        return math_ops.reduce_sum(x)

      return f

    branches = {bi: make_func(bi) for bi in range(nbranches)}

    branch_index = array_ops.placeholder_with_default(bi, [])
    with backprop.GradientTape() as tape:
      for x in inputs:
        tape.watch(x)
      case_out = control_flow_switch_case.switch_case(
          branch_index, branches, name=self.make_name())
    out_grad = 3.
    actual_grads = tape.gradient(case_out, inputs, output_gradients=out_grad)
    used_bi = (nbranches - 1) if (bi < 0 or bi >= nbranches - 1) else bi
    expected_grads = []
    for input_idx in range(nbranches):
      if used_bi == input_idx:
        with backprop.GradientTape() as tape:
          tape.watch(inputs[used_bi])
          y = make_func(used_bi)()
        expected_grads.append(
            self.evaluate(
                tape.gradient(y, inputs[used_bi], output_gradients=out_grad)))
      else:
        expected_grads.append(None if context.executing_eagerly() else [0.] *
                              (input_idx + 1))

    self.assertEqual(len(expected_grads), len(actual_grads))
    for expected, actual in zip(expected_grads, actual_grads):
      if expected is None:
        self.assertIsNone(actual)
      else:
        self.assertAllEqual(expected, self.evaluate(actual))

  @test_util.run_gpu_only
  @test_util.disable_xla("Wants RunMetadata")
  def testParallelExecution(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test unconditionally skips itself (flaky bug reference) at its first
    # statement, so the remaining body, including all assertions, is never executed.
    # Intended scenario (GPU only, XLA disabled, skipped under control flow v2): a while loop
    # of seven iterations selects, by index, among seven branches each computing a Cholesky
    # factorization on the GPU, with the loop on the CPU, run with full tracing and soft placement
    # off. Expected: seven Cholesky node stats are recorded and at least one of the others
    # overlaps in time with the first, showing disjoint branches ran in parallel. Returns nothing.
    """Verify disjoint branches across while iterations are run in parallel."""
    self.skipTest("b/210666081: Flaky")
    if control_flow_v2_toggles.control_flow_v2_enabled():
      self.skipTest("b/138870290")

    with ops.Graph().as_default() as g:
      nbranches = 7
      matrices = array_ops_stack.unstack(  # Ensure all are ready before while.
          array_ops.matrix_diag(
              random_ops.random_uniform([nbranches, 8, 512]) + 1e-3))

      def make_branch(i, mat, name):
        # [FUNCTION DOC]
        # Returns a branch function bound to the counter i, matrix mat and op name prefix name.

        def branch_fn():
          # [FUNCTION DOC]
          # Branch function: increments the counter and, on the first GPU, returns it together
          # with the sum of the Cholesky factor of the matrix (op named with the Cholesky suffix).
          next_i = i + 1
          with ops.device("gpu:0"):
            return next_i, math_ops.reduce_sum(
                linalg_ops.cholesky(mat, name=name + "_Cholesky"))

        return branch_fn

      def make_branches(i):
        # [FUNCTION DOC]
        # Returns the list of seven branch functions, one per matrix, for the counter i.
        return [
            make_branch(i, matrices[bi], "br{}".format(bi))
            for bi in range(nbranches)
        ]

      def cond(i, _):
        # [FUNCTION DOC]
        # Loop condition: true while the counter is below the number of branches.
        return i < nbranches

      def body(i, result):
        # [FUNCTION DOC]
        # Loop body: on the CPU, selects the branch indexed by the counter and returns the next
        # counter and the running result plus the branch output.
        with ops.device("cpu:0"):
          next_i, branch_out = control_flow_switch_case.switch_case(
              i, make_branches(i))
        return next_i, result + branch_out

      _, result = while_loop.while_loop(cond, body, [0, 0.])

      run_metadata = config_pb2.RunMetadata()
      run_options = config_pb2.RunOptions(
          trace_level=config_pb2.RunOptions.FULL_TRACE)
      config = config_pb2.ConfigProto(
          allow_soft_placement=False, log_device_placement=True)

      with session.Session(config=config, graph=g) as sess:
        _ = sess.run(result, options=run_options, run_metadata=run_metadata)
    chol_node_stats = []
    for dev_stats in run_metadata.step_stats.dev_stats:
      for node_stats in dev_stats.node_stats:
        if (node_stats.node_name.endswith("Cholesky") and
            node_stats.all_start_nanos > 0):
          chol_node_stats.append(node_stats)

    self.assertLen(chol_node_stats, nbranches)

    chol_node_stats = sorted(chol_node_stats, key=lambda stats: stats.node_name)
    op_start_nanos = [stats.all_start_nanos for stats in chol_node_stats]
    op_end_nanos = [
        stats.all_start_nanos + stats.op_end_rel_nanos
        for stats in chol_node_stats
    ]

    def overlap(range1, range2):
      # [FUNCTION DOC]
      # Takes two (start, end) time ranges and returns the length of their overlap, zero if they
      # are disjoint.
      s1, e1 = range1
      s2, e2 = range2
      if s1 < s2:
        return 0 if s2 > e1 else e1 - s2
      return 0 if s1 > e2 else e2 - s1

    timespans = list(zip(op_start_nanos, op_end_nanos))
    overlaps_chol0 = [overlap(timespans[0], r2) for r2 in timespans[1:]]
    # There are nbranches-1 overlaps, sometimes all nonzero, but we
    # conservatively check for at least one here, to avoid test flakiness.
    self.assertGreater(np.count_nonzero(overlaps_chol0), 0)

  def testCase_validateIndicesContiguous(self):
    # [FUNCTION DOC]
    # Scenario: branches mapped from indices zero, two and four (gaps). Expected: switch case
    # raises a value error saying the indices must form a contiguous range. Returns nothing.

    def make_func(bi):
      # [FUNCTION DOC]
      # Returns a branch function producing a float constant equal to ten times bi.
      return lambda: array_ops.constant(bi * 10., name="br{}_out".format(bi))

    branches = {i: make_func(i) for i in range(0, 6, 2)}
    with self.assertRaisesRegex(ValueError, "must form contiguous"):
      control_flow_switch_case.switch_case(array_ops.constant(0), branches)

  def testCase_validateIndicesDup(self):
    # [FUNCTION DOC]
    # Scenario: a list of index and function pairs with indices zero, two, four plus a duplicate
    # zero. Expected: switch case raises a value error about the indices needing to be
    # contiguous. Returns nothing.

    def make_func(bi):
      # [FUNCTION DOC]
      # Returns a branch function producing a float constant equal to ten times bi.
      return lambda: array_ops.constant(bi * 10., name="br{}_out".format(bi))

    branches = [(i, make_func(i)) for i in range(0, 6, 2)]
    branches.append((0, make_func(7)))
    with self.assertRaisesRegex(ValueError, "must form contiguous"):
      control_flow_switch_case.switch_case(array_ops.constant(0), branches)

  def testCase_validateBranchIndex(self):
    # [FUNCTION DOC]
    # Scenario: switch case is given a plain integer rather than a tensor as the branch index.
    # Expected: a type error mentioning the branch index and tensor is raised. Returns nothing.

    def make_func(bi):
      # [FUNCTION DOC]
      # Returns a branch function producing a float constant equal to ten times bi.
      return lambda: array_ops.constant(bi * 10., name="br{}_out".format(bi))

    branches = {i: make_func(i) for i in range(5)}
    with self.assertRaisesRegex(TypeError, "branch_index.*Tensor"):
      control_flow_switch_case.switch_case(1, branches)

  def testCase_validateNonIntKeys(self):
    # [FUNCTION DOC]
    # Scenario: the branch keys are tensors instead of Python integers. Expected: switch case
    # raises a type error saying keys must be Python integers. Returns nothing.

    def make_func(bi):
      # [FUNCTION DOC]
      # Returns a branch function producing a float constant equal to ten times bi.
      return lambda: array_ops.constant(bi * 10., name="br{}_out".format(bi))

    branches = [(array_ops.constant(i), make_func(i)) for i in range(5)]
    with self.assertRaisesRegex(TypeError, "must be a Python `int`"):
      control_flow_switch_case.switch_case(array_ops.constant(1), branches)


class ExecuteFnForDeviceTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for execute-fn-for-device, which picks a function by the device type (CPU, GPU, TPU)
  # the computation is placed on, falling back to a default function. Covers eager and compiled
  # functions, gradients and fallback behavior.

  # The same test can run with and without XLA compilation.
  # In non-XLA gpu case, it exercises gpu branch.
  # In XLA gpu cases, it exercises the default case.
  # This test is to test the non-XLA case so that we disable XLA.
  @test_util.disable_xla("xla has different execution branch")
  def testCommonCases(self):
    # [FUNCTION DOC]
    # Scenario (XLA disabled): a function doubles its input on CPU and squares it on GPU, with the
    # CPU version as default. It is run directly and inside a traced function with a gradient tape
    # for input three, under CPU placement, GPU placement (if a GPU exists) and no placement.
    # Expected: CPU gives six with gradient two, GPU gives nine with gradient six; with no placement
    # the GPU result is used when a GPU is available, otherwise the CPU result. Returns nothing.

    def cpu_fn(x):
      # [FUNCTION DOC]
      # CPU variant: returns the input added to itself.
      return x + x

    def gpu_fn(x):
      # [FUNCTION DOC]
      # GPU variant: returns the input multiplied by itself.
      return x * x

    def flexible_fn(a):
      # [FUNCTION DOC]
      # Selects between the CPU and GPU variants by device type, defaulting to the CPU variant,
      # and returns the result for input a.
      branches = {"CPU": lambda: cpu_fn(a), "GPU": lambda: gpu_fn(a)}
      return control_flow_switch_case.execute_fn_for_device(
          branches, lambda: cpu_fn(a))

    @def_function.function
    def flexible_defun(a):
      # [FUNCTION DOC]
      # Traced wrapper: returns the device-dependent function applied to a.
      return flexible_fn(a)

    def run_defun_and_tape(a):
      # [FUNCTION DOC]
      # Runs the traced function under a gradient tape and also the plain function. Returns a
      # tuple of the plain result, the traced result and the gradient of the traced result.
      with backprop.GradientTape() as tape:
        tape.watch(a)
        result = flexible_defun(a)
      grad = tape.gradient(result, a)
      r = flexible_fn(a)
      return r, result, grad

    a = array_ops.constant(3.)
    with ops.device("cpu:0"):
      r, result, grad = run_defun_and_tape(a)
      self.assertEqual(6., self.evaluate(r))
      self.assertEqual(6., self.evaluate(result))
      self.assertEqual([2.], self.evaluate(grad))

    if test_util.is_gpu_available():
      with ops.device("gpu:0"):
        r, result, grad = run_defun_and_tape(a)
        self.assertEqual(9., self.evaluate(r))
        self.assertEqual(9., self.evaluate(result))
        self.assertEqual([6.], self.evaluate(grad))

    # no device annotation
    r, result, grad = run_defun_and_tape(a)
    if test_util.is_gpu_available():
      self.assertEqual(9., self.evaluate(r))
      self.assertEqual(9., self.evaluate(result))
      self.assertEqual([6.], self.evaluate(grad))
    else:
      self.assertEqual(6., self.evaluate(r))
      self.assertEqual(6., self.evaluate(result))
      self.assertEqual([2.], self.evaluate(grad))

  def testCompile(self):
    # [FUNCTION DOC]
    # Scenario: only when a GPU is available (otherwise returns immediately), a jit-compiled
    # function selects between doubling (CPU) and squaring (GPU) with doubling as default.
    # Expected: under compilation the default branch is used, so input three gives six.
    # Returns nothing.
    if not test_util.is_gpu_available():
      return

    def cpu_fn(x):
      # [FUNCTION DOC]
      # CPU variant: returns the input added to itself.
      return x + x

    def gpu_fn(x):
      # [FUNCTION DOC]
      # GPU variant: returns the input multiplied by itself.
      return x * x

    @def_function.function(jit_compile=True)
    def flexible_defun(a):
      # [FUNCTION DOC]
      # Compiled function: picks a variant by device type with the CPU variant as default and
      # returns the result for input a.
      branches = {"CPU": lambda: cpu_fn(a), "GPU": lambda: gpu_fn(a)}
      return control_flow_switch_case.execute_fn_for_device(
          branches, lambda: cpu_fn(a))

    # Always execute the default branch in xla compilation case.
    a = array_ops.constant(3.)
    r = flexible_defun(a)
    self.assertEqual(6., self.evaluate(r))

  def testFallBack(self):
    # [FUNCTION DOC]
    # Scenario: only a TPU variant (cubing) is registered, with an identity default. The function
    # is run on CPU and, if available, on GPU, both traced and directly. Expected: the default is
    # used everywhere, so input three yields three. Returns nothing.

    def default_fn(x):
      # [FUNCTION DOC]
      # Default variant: returns the input unchanged.
      return x

    def tpu_fn(x):
      # [FUNCTION DOC]
      # TPU variant: returns the cube of the input.
      return x * x * x

    def flexible_fn(a):
      # [FUNCTION DOC]
      # Selects the TPU variant if on a TPU, otherwise the default, and returns the result for a.
      branches = {"TPU": lambda: tpu_fn(a)}
      return control_flow_switch_case.execute_fn_for_device(
          branches, default_fn=lambda: default_fn(a))

    @def_function.function
    def flexible_defun(a):
      # [FUNCTION DOC]
      # Traced wrapper: returns the device-dependent function applied to a.
      return flexible_fn(a)

    a = array_ops.constant(3.)
    with ops.device("cpu:0"):
      result_defun = flexible_defun(a)
      result_defun = flexible_fn(a)
      self.assertEqual(3., self.evaluate(result_defun))
      # execute_fn_for_device is not inside defun_function.
      result = flexible_fn(a)
      self.assertEqual(3., self.evaluate(result))

    if test_util.is_gpu_available():
      with ops.device("gpu:0"):
        result_defun = flexible_defun(a)
        self.assertEqual(3., self.evaluate(result_defun))
        # execute_fn_for_device is not inside defun_function.
        result = flexible_fn(a)
        self.assertEqual(3., self.evaluate(result))


class CaseTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the predicate-based case operation: default branch, exclusive versus non-exclusive
  # evaluation, missing default behavior and eager execution. Most tests run in v1 graph mode.

  @test_util.run_deprecated_v1
  def testCase_withDefault(self):
    # [FUNCTION DOC]
    # Scenario: an integer placeholder with two predicates (equals one, equals two) and a default,
    # in exclusive mode. Expected: values 1, 2, 3 produce 2, 4, 6 respectively. Returns nothing.
    x = array_ops.placeholder(dtype=dtypes.int32, shape=[])
    conditions = [(math_ops.equal(x, 1), lambda: constant_op.constant(2)),
                  (math_ops.equal(x, 2), lambda: constant_op.constant(4))]
    default = lambda: constant_op.constant(6)
    output = control_flow_case.case(conditions, default, exclusive=True)
    with self.cached_session() as sess:
      self.assertEqual(sess.run(output, feed_dict={x: 1}), 2)
      self.assertEqual(sess.run(output, feed_dict={x: 2}), 4)
      self.assertEqual(sess.run(output, feed_dict={x: 3}), 6)

  @test_util.run_deprecated_v1
  def testCase_multiple_matches_exclusive(self):
    # [FUNCTION DOC]
    # Scenario: two predicates are true for the value two, exclusive mode, with a default of
    # eight. Expected: values one and three give two and eight; value two raises an invalid
    # argument error ("Input error") at run time because several predicates hold. Returns nothing.
    x = array_ops.placeholder(dtype=dtypes.int32, shape=[])
    conditions = [(math_ops.equal(x, 1), lambda: constant_op.constant(2)),
                  (math_ops.equal(x, 2), lambda: constant_op.constant(4)),
                  (math_ops.equal(x, 2), lambda: constant_op.constant(6))]
    default = lambda: constant_op.constant(8)
    output = control_flow_case.case(conditions, default, exclusive=True)
    with self.cached_session() as sess:
      self.assertEqual(sess.run(output, feed_dict={x: 1}), 2)
      self.assertEqual(sess.run(output, feed_dict={x: 3}), 8)
      with self.assertRaisesRegex(errors.InvalidArgumentError, "Input error:"):
        sess.run(output, feed_dict={x: 2})

  @test_util.run_deprecated_v1
  def testCase_multiple_matches_non_exclusive(self):
    # [FUNCTION DOC]
    # Scenario: same predicates as the exclusive test but non-exclusive mode. Expected: the first
    # matching predicate wins, so values one, two, three give two, four, eight with no error.
    # Returns nothing.
    x = array_ops.placeholder(dtype=dtypes.int32, shape=[])
    conditions = [(math_ops.equal(x, 1), lambda: constant_op.constant(2)),
                  (math_ops.equal(x, 2), lambda: constant_op.constant(4)),
                  (math_ops.equal(x, 2), lambda: constant_op.constant(6))]
    default = lambda: constant_op.constant(8)
    output = control_flow_case.case(conditions, default, exclusive=False)
    with self.cached_session() as sess:
      self.assertEqual(sess.run(output, feed_dict={x: 1}), 2)
      self.assertEqual(sess.run(output, feed_dict={x: 2}), 4)
      self.assertEqual(sess.run(output, feed_dict={x: 3}), 8)

  @test_util.run_deprecated_v1
  def testCase_withoutDefault(self):
    # [FUNCTION DOC]
    # Scenario: three predicates and no default, exclusive mode. Expected: values one to three give
    # two, four, six; value four (no predicate true) raises an invalid argument error at run time.
    # Returns nothing.
    x = array_ops.placeholder(dtype=dtypes.int32, shape=[])
    conditions = [(math_ops.equal(x, 1), lambda: constant_op.constant(2)),
                  (math_ops.equal(x, 2), lambda: constant_op.constant(4)),
                  (math_ops.equal(x, 3), lambda: constant_op.constant(6))]
    output = control_flow_case.case(conditions, exclusive=True)
    with self.cached_session() as sess:
      self.assertEqual(sess.run(output, feed_dict={x: 1}), 2)
      self.assertEqual(sess.run(output, feed_dict={x: 2}), 4)
      self.assertEqual(sess.run(output, feed_dict={x: 3}), 6)
      with self.assertRaisesRegex(errors.InvalidArgumentError, "Input error:"):
        sess.run(output, feed_dict={x: 4})

  @test_util.run_deprecated_v1
  def testCase_withoutDefault_oneCondition(self):
    # [FUNCTION DOC]
    # Scenario: a single predicate and no default. Expected: value one gives two; value four raises
    # an invalid argument error at run time. Returns nothing.
    x = array_ops.placeholder(dtype=dtypes.int32, shape=[])
    conditions = [(math_ops.equal(x, 1), lambda: constant_op.constant(2))]
    output = control_flow_case.case(conditions, exclusive=True)
    with self.cached_session() as sess:
      self.assertEqual(sess.run(output, feed_dict={x: 1}), 2)
      with self.assertRaisesRegex(errors.InvalidArgumentError, "Input error:"):
        sess.run(output, feed_dict={x: 4})

  @test_util.run_in_graph_and_eager_modes
  def testCase_dict(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name suggests the predicate and function pairs are passed as a mapping, but
    # the test passes a list of pairs, so no dictionary input is exercised.
    # Scenario (graph and eager): value two with predicates for one and two, no default,
    # exclusive. Expected: the result is four. Returns nothing.
    x = constant_op.constant(2)
    conditions = [(math_ops.equal(x, 1), lambda: constant_op.constant(2)),
                  (math_ops.equal(x, 2), lambda: constant_op.constant(4))]
    output = control_flow_case.case(conditions, exclusive=True)
    self.assertEqual(4, self.evaluate(output))


class WhileLoopTestCase(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the while loop operation: structure of the returned value (single variable versus
  # list, return-same-structure flag, maximum iterations) and gradient capture behavior.

  @test_util.run_in_graph_and_eager_modes
  def testWhileLoopWithSingleVariable(self):
    # [FUNCTION DOC]
    # Scenario: loop a counter from zero while below ten, adding one. Expected: result is ten.
    # Returns nothing.
    i = constant_op.constant(0)
    c = lambda i: math_ops.less(i, 10)
    b = lambda i: math_ops.add(i, 1)
    r = while_loop.while_loop(c, b, [i])

    self.assertEqual(self.evaluate(r), 10)

  @test_util.run_in_graph_and_eager_modes
  def testEagerWhileLoopWithSingleVariable_bodyReturnsTuple(self):
    # [FUNCTION DOC]
    # Scenario: same counter loop but the body returns a one-element tuple. Expected: the loop
    # result is the tuple containing ten, matching the body's structure. Returns nothing.
    i = constant_op.constant(0)
    c = lambda i: math_ops.less(i, 10)
    b = lambda i: (math_ops.add(i, 1),)
    r = while_loop.while_loop(c, b, [i])

    # Expect a tuple since that is what the body returns.
    self.assertEqual(self.evaluate(r), (10,))

  @test_util.run_v1_only("Unsupported in cfv2")
  def testWhileLoopSameReturnShape_False(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the existing comment admits the second result is incorrect: with a maximum
    # iterations value the loop returns the full list of the counter and an empty list, while the
    # first form returns only the counter; the test asserts this inconsistent behavior.
    # Scenario (v1 only): body returns a list of the counter and an empty list, default structure
    # handling. Expected: ten without a maximum, and the pair of ten and empty list with one.
    # Returns nothing.
    i = constant_op.constant(0)
    c = lambda i, _: math_ops.less(i, 10)

    # Body returns a [tensor, []]
    b = lambda i, _: [math_ops.add(i, 1), []]

    # Should only return the tensor.
    r = while_loop.while_loop(c, b, [i, []])
    self.assertEqual(self.evaluate(r), 10)

    # Adding maximum_iterations should yield the same result.
    r = while_loop.while_loop(c, b, [i, []], maximum_iterations=50)
    # Note: this result is still incorrect - it should be just 10.
    self.assertEqual(self.evaluate(r), [10, []])

  def testWhileLoopSameReturnShape_FalseSingleLoopVar(self):
    # [FUNCTION DOC]
    # Scenario: single loop variable, body returns a bare tensor, same-structure flag off.
    # Expected: result is ten with and without a maximum iterations value. Returns nothing.
    i = constant_op.constant(0)
    c = lambda i: math_ops.less(i, 10)

    # Body return must be unpacked in this case.
    b = lambda i: math_ops.add(i, 1)

    # Should only return the tensor.
    r = while_loop.while_loop(c, b, [i])
    self.assertEqual(self.evaluate(r), 10)

    # Adding maximum_iterations should yield the same result.
    r = while_loop.while_loop(c, b, [i], maximum_iterations=50)
    self.assertEqual(self.evaluate(r), 10)

  def testWhileLoopSameReturnShape_True(self):
    # [FUNCTION DOC]
    # Scenario: body returns a list of the counter and an empty list with the same-structure flag
    # on. Expected: the result keeps the structure, a list of ten and an empty list, with and
    # without a maximum iterations value. Returns nothing.
    i = constant_op.constant(0)
    c = lambda i, _: math_ops.less(i, 10)

    # Body returns a [tensor, []]
    b = lambda i, _: [math_ops.add(i, 1), []]

    # Should only return the original structure.
    r = while_loop.while_loop(c, b, [i, []], return_same_structure=True)
    self.assertEqual(self.evaluate(r), [10, []])

    # Adding maximum_iterations should yield the same result.
    r = while_loop.while_loop(
        c, b, [i, []], return_same_structure=True, maximum_iterations=50)
    self.assertEqual(self.evaluate(r), [10, []])

  def testWhileLoopSameReturnShape_TrueSingleLoopVar(self):
    # [FUNCTION DOC]
    # Scenario: single loop variable, body returns a one-element list, same-structure flag on.
    # Expected: the result remains a one-element list holding ten, with and without a maximum
    # iterations value. Returns nothing.
    i = constant_op.constant(0)
    c = lambda i: math_ops.less(i, 10)

    b = lambda i: [math_ops.add(i, 1)]

    # Should not unpack the single variable
    r = while_loop.while_loop(c, b, [i], return_same_structure=True)
    self.assertEqual(self.evaluate(r), [10])

    # Adding maximum_iterations should yield the same result.
    r = while_loop.while_loop(
        c, b, [i], return_same_structure=True, maximum_iterations=50)
    self.assertEqual(self.evaluate(r), [10])

  @test_util.enable_control_flow_v2
  @test_util.run_in_graph_and_eager_modes
  def testSkipsUnnecessaryCaptureGradients(self):
    # [FUNCTION DOC]
    # Scenario (control flow v2, graph and eager): a loop body adds twice a watched variable and a
    # value passed through a custom-gradient trap whose gradient would assert failure if computed.
    # Expected: differentiating the loop output with respect to the watched variable yields twenty
    # (ten iterations of two) and the trap gradient, for a captured value not requested, is never
    # computed. Returns nothing.

    @custom_gradient.custom_gradient
    def gradient_trap(t):
      # [FUNCTION DOC]
      # Custom-gradient identity: returns its input and a gradient function that would fail the
      # test (via an equality assertion) if ever evaluated.

      def grad(w):
        # [FUNCTION DOC]
        # Gradient function: asserts that zero equals one (which should never run) and returns
        # the incoming gradient.
        # Computing this gradient should fail the test
        check_ops.assert_equal(0, 1)
        return w

      return t, grad

    x = array_ops.constant(0.0, name="x")
    y = array_ops.constant(1.0, name="y")

    def cond(s):
      # [FUNCTION DOC]
      # Loop condition: true while the loop value is below ten.
      return s < 10.0

    def body(s):
      # [FUNCTION DOC]
      # Loop body: adds twice the watched variable and the trap-wrapped captured value to the loop
      # value and returns it.
      return s + 2 * x + gradient_trap(y)

    with backprop.GradientTape() as tape:
      tape.watch(x)
      out = while_loop.while_loop(cond, body, (array_ops.constant(0.0),))

    grad = tape.gradient(out, x)
    self.assertAllEqual(grad, 20.0)


class WhileLoopParallelismTest(test_util.TensorFlowTestCase,
                               parameterized.TestCase):
  # [CLASS DOC]
  # Tests that parallel iterations in a traced while loop still respect ordering of stateful
  # resource reads and writes. Both tests only run when the v2 behavior is enabled.

  @parameterized.parameters(*itertools.product(
      (False, True),
      (False, True),
      (False, True),
      (False, True),
      (False, True),
  ))
  def testResourceHandlingInLoop(self, read_before, read_after, modify_in_loop,
                                 modify_before, modify_after):
    # [FUNCTION DOC]
    # Parameterized over all combinations of five boolean flags: whether to read the counter
    # variable before the loop, read it after, modify it inside the loop, before it, or after it.
    # Skipped unless v2 behavior is enabled. A traced function resets a counter variable, optionally
    # reads/modifies it around a loop of parallel iterations that records the counter each
    # iteration. Expected: the maximum recorded value is the pre-loop modification plus three times
    # the in-loop modification; the pre-loop read is zero; the post-loop read matches the value
    # after the loop when enabled (else zero); and the final variable value includes all
    # modifications. Returns nothing.

    if not tf2.enabled():
      self.skipTest("V2-only test.")

    ticker = variables.Variable(0)

    @def_function.function
    def run_loop(n):
      # [FUNCTION DOC]
      # Traced function: runs the loop for n iterations according to the enclosing test's flags and
      # returns the stacked per-iteration counter values, the pre-loop read and the post-loop read.
      ticker.assign(0)
      i = constant_op.constant(0)
      t_acc = tensor_array_ops.TensorArray(
          dtypes.int32, size=0, dynamic_size=True)

      if read_before:
        rb = ticker.read_value()
      else:
        rb = constant_op.constant(0)
      if modify_before:
        ticker.assign_add(1)

      while i < n:
        directives.set_loop_options(parallel_iterations=10)
        if modify_in_loop:
          ticker.assign_add(1)
        t_acc = t_acc.write(i, ticker.read_value())
        i += 1

      if read_after:
        ra = ticker.read_value()
      else:
        ra = constant_op.constant(0)
      if modify_after:
        ticker.assign_add(1)

      return t_acc.stack(), rb, ra

    # Warm-up.
    self.evaluate(run_loop(1))

    self.evaluate(ticker.assign(123))
    acc, rb, ra = run_loop(3)
    self.assertEqual(
        self.evaluate(math_ops.reduce_max(acc)),
        int(modify_before) + 3 * int(modify_in_loop))

    # Double check variable reads are still sequenced.
    self.assertEqual(self.evaluate(rb), 0)

    if read_after:
      expected_ra = int(modify_before) + 3 * int(modify_in_loop)
    else:
      expected_ra = 0
    self.assertEqual(self.evaluate(ra), expected_ra)

    # Double-check that the loop ran completely.
    self.assertEqual(
        self.evaluate(ticker.read_value()),
        int(modify_before) + 3 * int(modify_in_loop) + int(modify_after))

  def testStatefulParallelism(self):
    # [FUNCTION DOC]
    # Scenario (skipped unless v2 behavior is enabled): a traced loop of parallel iterations calls
    # an opaque Python function that sleeps one second, increments a counter variable and records
    # its argument, and each iteration also reads the counter. Expected: because stateful calls are
    # sequenced, reads see the effect of earlier iterations (maximum read is two), the counter ends
    # at three and the Python calls ran in order one, two, three. Timing dependent (reads must be
    # much faster than one second). Returns nothing.

    if not tf2.enabled():
      self.skipTest("V2-only test.")

    ticker = variables.Variable(0)
    # Secondary state for the pyfunc that lets us verify that things ran in
    # the correct relative order.
    ticker_state = []

    def wait_then_tick(i):
      # [FUNCTION DOC]
      # Wraps an eager Python function call that TensorFlow cannot see through and returns its
      # integer output for the loop index i.
      # The contents of py_funcs is opaque, so TF doesn't see this variable
      # assignment. In turn, this allows us to run it in parallel with
      # the variable read.
      def wait_then_tick_py_fn(i):
        # [FUNCTION DOC]
        # Python side effect function: sleeps one second, increments the counter variable, appends
        # the numeric value of i to the shared order list and returns one.
        time.sleep(1)
        ticker.assign_add(1)
        ticker_state.append(i.numpy().item())
        return 1

      return script_ops.eager_py_func(wait_then_tick_py_fn, [i],
                                      [dtypes.int32])[0]

    @def_function.function
    def run_loop(n):
      # [FUNCTION DOC]
      # Traced function: resets the counter, then for n iterations calls the sleeping Python
      # function and records the counter read. Returns the stacked recorded values.
      ticker.assign(0)
      i = constant_op.constant(0)
      t_acc = tensor_array_ops.TensorArray(
          dtypes.int32, size=0, dynamic_size=True)

      while i < n:
        directives.set_loop_options(parallel_iterations=10)
        wait_then_tick(i + 1)
        # The read is expected to run in much less than `wait_then_tick`,
        # which sleeps for 1s. Hence all reads should complete before the first
        # `wait_then_tick` increments the `ticker` variable.
        t_acc = t_acc.write(i, ticker.read_value())
        i += 1

      return t_acc.stack()

    # Warm-up.
    self.evaluate(run_loop(1))

    # This test is deterministic so long as the runtime is fast enough to
    # execute `t_acc = t_acc.write(i, ticker.read_value())` in much less than
    # one second.
    self.evaluate(ticker.assign(123))
    ticker_state.clear()
    acc = run_loop(3)
    # Because the loop runs entirely sequentially, the reads in each iteration
    # see the effects of the pyfunc from the previous iteration.
    self.assertEqual(self.evaluate(math_ops.reduce_max(acc)), 2)

    # Double-check that the loop ran completely.
    self.assertEqual(self.evaluate(ticker.read_value()), 3)
    # Double check that the pyfuncs ran in order.
    self.assertListEqual(ticker_state, [1, 2, 3])


class AssertTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the assert operation, in graph mode and inside traced functions.

  @test_util.run_deprecated_v1
  def testAssert(self):
    # [FUNCTION DOC]
    # Scenario: assert that a counter is below ten. Expected: with zero evaluation succeeds; with
    # ten evaluation raises an invalid argument error. Returns nothing.
    i = constant_op.constant(0)
    c = control_flow_assert.Assert(i < 10, [i, [10], [i + 1]])
    self.evaluate(c)

    i = constant_op.constant(10)
    c = control_flow_assert.Assert(i < 10, [i, [10], [i + 1]])
    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(c)

  @test_util.run_in_graph_and_eager_modes
  def testAssertInFunction(self):
    # [FUNCTION DOC]
    # Scenario (skipped when a GPU is available): a traced function asserts its boolean argument
    # and returns five. Expected: calling with false raises an invalid argument error and calling
    # with true returns five. Returns nothing.
    # TODO(fishx): Re-enable this test for GPU.
    # NOTE(fishx): Disable this test for now because, in GPU, multiple errors
    # will be thrown. But since the root cause error is marked as "derived"
    # error. So it might be ignored.
    if test_util.is_gpu_available():
      self.skipTest("Skip GPU Test")

    @def_function.function
    def whiny(value):
      # [FUNCTION DOC]
      # Traced function: asserts the value is true with a message and returns the constant five.
      control_flow_assert.Assert(value, ["Raised false"])
      return constant_op.constant(5)

    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(whiny(False))

    self.assertAllEqual(whiny(True), 5)


class AsyncNoopTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for the asynchronous no-op operation of the v2 while loop module.

  def testAsyncNoop(self):
    # [FUNCTION DOC]
    # Scenario: a traced function uses the async no-op as a control dependency of an addition.
    # Expected: the function still returns four (two plus two). Returns nothing.

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # Traced function: adds two to the constant two under a control dependency on the async
      # no-op and returns the sum.
      x = constant_op.constant(2)
      with ops.control_dependencies([while_v2.async_noop()]):
        y = x + 2
      return y

    self.assertEqual(self.evaluate(f()), 4)


if __name__ == "__main__":
  googletest.main()
