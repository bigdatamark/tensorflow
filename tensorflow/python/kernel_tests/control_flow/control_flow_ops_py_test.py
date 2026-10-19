# [MODULE DOC]
# KNOWN ISSUE: the Apache license header at the top has a typo in the warranty line (a stray
# letter inside the word for "OR"), which differs from the standard license text.
# Large unit-test module for TensorFlow control flow: cond, case, switch_case, while_loop,
# the low-level Switch/Merge/Enter/Exit/NextIteration primitives, gradients through them, nested
# and XLA contexts, TensorArray, variables and resources inside branches, device placement,
# automatic control dependencies, and related behaviors. Most tests run under both the legacy
# v1 control flow and control flow v2 (via decorators), in graph and eager modes.
# Exposes small module-level helper functions (graph consumer checks, fetchable/feedable tensor
# listing, session config builder, a summing loop, print capture helpers) and the large
# ControlFlowTest class plus further test classes defined later in the file.
# No import-time side effects beyond importing gradient and TensorArray modules for registration.
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
# WITHOUT WARRANTIES OiR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# pylint: disable=g-long-lambda
"""Tests for tensorflow.ops.control_flow_ops."""

import collections
import math
import re
import sys
import time

from absl.testing import parameterized
import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import tf2
from tensorflow.python.client import device_lib
from tensorflow.python.client import session
from tensorflow.python.data.experimental.ops import cardinality
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function as eager_def_function
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import function
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_case
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gen_logging_ops
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_grad  # pylint: disable=unused-import
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop as while_loop_tf
from tensorflow.python.ops import while_v2  # pylint: disable=unused-import
# pylint: disable=unused-import
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
import tensorflow.python.ops.tensor_array_grad
# pylint: enable=unused-import
from tensorflow.python.platform import test
from tensorflow.python.training import adam
from tensorflow.python.training import gradient_descent
from tensorflow.python.util import nest
from tensorflow.python.ops import control_flow_switch_case
from tensorflow.python.ops import variable_v1


def check_consumers(graph):
  # [FUNCTION DOC]
  # HELPER: sanity-check utility used by tests to validate graph consumer bookkeeping.
  # Takes a graph. Counts, for every tensor, how many operation inputs in the graph use it, then
  # compares that count with the number of consumers the tensor itself reports.
  # Returns true if all counts agree, false at the first mismatch. No side effects.
  """Sanity check on the consumer list of the tensors."""

  consumer_count = {}
  for op in graph.get_operations():
    for v in op.inputs:
      cnt = consumer_count.get(v, 0)
      consumer_count[v] = cnt + 1
  for k, v in consumer_count.items():
    if len(k.consumers()) != v:
      return False
  return True


def all_fetchables():
  # [FUNCTION DOC]
  # HELPER: test utility that lists which tensors the default graph allows to be fetched.
  # Takes no arguments; walks every operation output in the current default graph and collects the
  # names of those marked fetchable. Returns the list of tensor names. No side effects.
  tensor_names = []
  graph = ops.get_default_graph()
  for op in graph.get_operations():
    for t in op.outputs:
      if graph.is_fetchable(t):
        tensor_names.append(t.name)
  return tensor_names


def all_feedables():
  # [FUNCTION DOC]
  # HELPER: test utility that lists which tensors the default graph allows to be fed.
  # Takes no arguments; walks every operation input in the current default graph and collects the
  # tensors that are feedable (duplicates are possible when a tensor is used by several ops).
  # Returns the list of tensors. No side effects.
  feedable_tensors = []
  graph = ops.get_default_graph()
  for op in graph.get_operations():
    for t in op.inputs:
      if graph.is_feedable(t):
        feedable_tensors.append(t)
  return feedable_tensors


def opt_cfg(do_constant_folding=True):
  # [FUNCTION DOC]
  # HELPER: builds the session configuration used by tests that need graph optimization.
  # Takes an optional boolean (default true) controlling constant folding. Returns a session
  # configuration with soft placement allowed, optimization level one, function inlining enabled,
  # and constant folding set per the argument. No side effects.
  return config_pb2.ConfigProto(
      allow_soft_placement=True,
      graph_options=config_pb2.GraphOptions(
          optimizer_options=config_pb2.OptimizerOptions(
              opt_level=config_pb2.OptimizerOptions.L1,
              do_function_inlining=True,
              do_constant_folding=do_constant_folding)))


def isum(s, maximum_iterations=None):
  # [FUNCTION DOC]
  # HELPER: builds a small while loop shared by several while-loop tests.
  # Takes an initial accumulator value s (a tensor, possibly a vector) and an optional maximum
  # iteration count (default none). Creates a counter starting at zero and loops while the counter
  # is below ten, incrementing it and adding the counter into the accumulator each iteration.
  # Returns the final accumulator tensor (so the result is the sum of counters, 45 plus the
  # initial value when unbounded). Adds ops to the current graph.
  i = constant_op.constant(0, name="i")
  c = lambda i, s: math_ops.less(i, 10)
  b = lambda i, s: [math_ops.add(i, 1), math_ops.add(i, s)]
  _, r_s = while_loop_tf.while_loop(
      c, b, [i, s], maximum_iterations=maximum_iterations)
  return r_s


def enqueue_print_op(s):
  # [FUNCTION DOC]
  # HELPER: creates a print op whose output can be recognised in captured stderr by the tests.
  # Takes a string s; returns a print operation that writes the fixed test prefix followed by s.
  # Adds an op to the current graph; printing happens only when the op is run.
  """Enqueues an op that prints a message to be captured in the test."""
  return logging_ops.print_v2("ControlFlowOpsTest: " + s)


def filter_test_messages(s):
  # [FUNCTION DOC]
  # HELPER: companion of enqueue_print_op that parses captured output.
  # Takes the captured text, splits it into lines and keeps only lines that start with the test
  # prefix, stripping the prefix. Returns the list of remaining message strings in order.
  """Returns a list of messages printed by enqueue_print_op."""
  prefix = "ControlFlowOpsTest: "
  return [l[len(prefix):] for l in s.split("\n") if l.startswith(prefix)]


def tf_function_in_tf2(f):
  # [FUNCTION DOC]
  # HELPER: conditionally wraps a Python function in a compiled function for TF2 test paths.
  # Takes a callable f. When TF2 behavior is enabled it returns f wrapped as a tf function;
  # otherwise (TF1 mode) returns f unchanged so the v1 control flow code path is exercised.
  if tf2.enabled():
    # In TF1 do not wrap with tf.function so that we can test the v1 control
    # flow code path.
    return eager_def_function.function(f)
  return f


@test_util.with_eager_op_as_function
@test_util.with_control_flow_v2
class ControlFlowTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Main test case for control flow ops, combining the TensorFlow test base class with
  # parameterized testing. Class decorators run it with eager ops-as-functions and with both
  # control flow v1 and v2. Covers the raw control flow primitives, cond, switch_case, while_loop,
  # gradients, device placement and many regression tests. Has no instance state of its own;
  # each test builds its own graph or runs eagerly.

  @test_util.run_v1_only("b/120545219")
  def testRefIdentity(self):
    # [FUNCTION DOC]
    # Test (v1 only): creates a reference variable, takes a ref identity of it, assigns 9 through the
    # identity, and wraps it with a dependency on the assign. Expects the result to be a tensor and
    # to evaluate to 9 after variable initialization.
    with self.cached_session():
      v = variable_v1.VariableV1(7)

      v = control_flow_ops._Identity(v)
      op = state_ops.assign(v, 9)
      v2 = control_flow_ops.with_dependencies([op], v)

      self.assertTrue(isinstance(v2, tensor_lib.Tensor))
      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(9, self.evaluate(v2))

  @test_util.run_v1_only("b/120545219")
  def testRefEnter(self):
    # [FUNCTION DOC]
    # Test (v1 only): enters a reference variable into a loop frame as a constant, assigns a value
    # (also entered into the frame) to it, exits the frame, and expects the exited value to be 9
    # after initializing variables.
    with self.cached_session():
      v = variable_v1.VariableV1(7)

      enter_v = control_flow_ops._Enter(v, "foo_1", is_constant=True)
      nine = constant_op.constant(9)
      enter_nine = gen_control_flow_ops.enter(nine, "foo_1")
      op = state_ops.assign(enter_v, enter_nine)
      v2 = control_flow_ops.with_dependencies([op], enter_v)
      v3 = control_flow_ops.exit(v2)
      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(9, self.evaluate(v3))

  @test_util.run_v1_only("b/120545219")
  def testRefSwitch(self):
    # [FUNCTION DOC]
    # Test (v1 only): switches a variable reference on a true predicate, assigns 9 to the true-side
    # output and expects evaluating the assignment to give 9 after variable initialization.
    with self.cached_session():
      v = variable_v1.VariableV1(7)

      p = constant_op.constant(True)
      v1 = control_flow_ops._SwitchRefOrTensor(v._ref(), p)  # pylint: disable=protected-access
      v2 = state_ops.assign(v1[1], 9)
      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(9, self.evaluate(v2))

  def testEnterMulExit(self):
    # [FUNCTION DOC]
    # Test: enters a vector and a scalar into the same frame, multiplies them, exits the frame and
    # expects every element of the vector to be multiplied by five.
    with self.cached_session():
      data = constant_op.constant([1, 2, 3, 4, 5, 6], name="data")
      enter_data = gen_control_flow_ops.enter(data, "foo_1", False)
      five = constant_op.constant(5)
      enter_five = gen_control_flow_ops.enter(five, "foo_1", False)
      mul_op = math_ops.multiply(enter_data, enter_five)
      exit_op = control_flow_ops.exit(mul_op)

      result = self.evaluate(exit_op)
    self.assertAllEqual(np.array([x * 5 for x in [1, 2, 3, 4, 5, 6]]), result)

  @test_util.run_deprecated_v1
  def testEnterShapePropagation(self):
    # [FUNCTION DOC]
    # Test (deprecated v1 mode): enters a variable into a frame with the constant flag true and
    # expects the static shape to be propagated (two elements); with the constant flag false it
    # expects the shape to be unknown.
    with self.cached_session():
      v = variables.Variable([0.0, 0.0], dtype=dtypes.float32)

      # If is_constant=True, the shape information should be propagated.
      enter_v_constant = gen_control_flow_ops.enter(
          v, "frame1", is_constant=True)
      self.assertEqual(enter_v_constant.shape, [2])

      # Otherwise, the shape should be unknown.
      enter_v_non_constant = gen_control_flow_ops.enter(
          v, "frame2", is_constant=False)
      self.assertEqual(enter_v_non_constant.shape, None)

  @test_util.run_v1_only("b/120545219")
  def testSwitchMergeIndexedSlices(self):
    # [FUNCTION DOC]
    # Test (v1 only): switches an indexed-slices value on a true predicate, merges the outputs and
    # expects the values and indices to pass through unchanged.
    with self.cached_session():
      values = constant_op.constant([1, 2, 3, 4, 5, 6])
      indices = constant_op.constant([0, 2, 4, 6, 8, 10])
      data = indexed_slices.IndexedSlices(values, indices)
      pred = ops.convert_to_tensor(True)
      switch_op = control_flow_ops.switch(data, pred)
      merge_op = control_flow_ops.merge(switch_op)[0]

      val = merge_op.values
      ind = merge_op.indices
    self.assertAllEqual(np.arange(1, 7), val)
    self.assertAllEqual(np.arange(0, 12, 2), ind)

  @test_util.run_v1_only("b/120545219")
  def testSwitchDeadBranch(self):
    # [FUNCTION DOC]
    # Test (v1 only): switches data on a true predicate and evaluates an identity of the false
    # (dead) output. Expects an invalid-argument error saying the return value has no value.
    with self.cached_session():
      data = constant_op.constant([1, 2, 3, 4, 5, 6], name="data")
      ports = ops.convert_to_tensor(True, name="ports")
      switch_op = control_flow_ops.switch(data, ports)
      dead_branch = array_ops.identity(switch_op[0])

      with self.assertRaisesWithPredicateMatch(
          errors_impl.InvalidArgumentError,
          lambda e: "Retval[0] does not have value" in str(e)):
        self.evaluate(dead_branch)

  @test_util.run_v1_only("b/120545219")
  def testSwitchMergeLess(self):
    # [FUNCTION DOC]
    # Test (v1 only): switches data on the result of comparing zero less than one (true), merges the
    # branches and expects the original data back.
    with self.cached_session():
      data = constant_op.constant([1, 2, 3, 4, 5, 6], name="data")
      zero = ops.convert_to_tensor(0)
      one = ops.convert_to_tensor(1)
      less_op = math_ops.less(zero, one)
      switch_op = control_flow_ops.switch(data, less_op)
      merge_op = control_flow_ops.merge(switch_op)[0]

      result = self.evaluate(merge_op)
    self.assertAllEqual(np.arange(1, 7), result)

  @test_util.run_v1_only("b/120545219")
  def testSwitchMergeAddIdentity(self):
    # [FUNCTION DOC]
    # Test (v1 only): switches on a false predicate; the false branch adds one, the true branch is an
    # identity; merges them. Expects the data incremented by one.
    with self.cached_session():
      data = constant_op.constant([1, 2, 3, 4, 5, 6], name="data")
      ports = ops.convert_to_tensor(False, name="ports")
      switch_op = control_flow_ops.switch(data, ports)
      one = constant_op.constant(1)
      add_op = math_ops.add(switch_op[0], one)
      id_op = array_ops.identity(switch_op[1])
      merge_op = control_flow_ops.merge([add_op, id_op])[0]

      result = self.evaluate(merge_op)
    self.assertAllEqual(np.array([x + 1 for x in [1, 2, 3, 4, 5, 6]]), result)

  @test_util.run_v1_only("b/120545219")
  def testSwitchMergeAddMul(self):
    # [FUNCTION DOC]
    # Test (v1 only): switches on a true predicate; false branch adds one, true branch multiplies by
    # five; merges. Expects the data multiplied by five.
    with self.cached_session():
      data = constant_op.constant([1, 2, 3, 4, 5, 6], name="data")
      ports = ops.convert_to_tensor(True, name="ports")
      switch_op = control_flow_ops.switch(data, ports)
      one = constant_op.constant(1)
      add_op = math_ops.add(switch_op[0], one)
      five = constant_op.constant(5)
      mul_op = math_ops.multiply(switch_op[1], five)
      merge_op = control_flow_ops.merge([add_op, mul_op])[0]

      result = self.evaluate(merge_op)
    self.assertAllEqual(np.array([x * 5 for x in [1, 2, 3, 4, 5, 6]]), result)

  @test_util.run_v1_only("b/120545219")
  def testLoop_false(self):
    # [FUNCTION DOC]
    # Test (v1 only): hand-built loop with Enter, Merge, Switch, Exit and NextIteration where the loop
    # condition is false from the start. Expects the loop to exit immediately returning the initial
    # value 10.
    with self.cached_session():
      false = ops.convert_to_tensor(False)
      n = constant_op.constant(10)

      enter_false = gen_control_flow_ops.enter(false, "foo_1", False)
      enter_n = gen_control_flow_ops.enter(n, "foo_1", False)

      merge_n = control_flow_ops.merge([enter_n, enter_n], name="merge_n")[0]
      switch_n = control_flow_ops.switch(merge_n, enter_false)
      exit_n = control_flow_ops.exit(switch_n[0])
      next_n = control_flow_ops.next_iteration(switch_n[0])
      merge_n.op._update_input(1, next_n)

      result = self.evaluate(exit_n)
    self.assertAllEqual(10, result)

  @test_util.run_deprecated_v1
  def testLoop_1(self):
    # [FUNCTION DOC]
    # Test (deprecated v1 mode): hand-builds a counting loop from Enter, Merge, LoopCond, Switch,
    # Add, NextIteration and Exit primitives, placing the merge on the GPU device name. Expects
    # the loop to count from zero and exit with 10.
    with self.cached_session():
      zero = constant_op.constant(0)
      one = constant_op.constant(1)
      n = constant_op.constant(10)

      enter_i = gen_control_flow_ops.enter(zero, "foo", False)
      enter_one = gen_control_flow_ops.enter(one, "foo", True)
      enter_n = gen_control_flow_ops.enter(n, "foo", True)

      with ops.device(test.gpu_device_name()):
        merge_i = control_flow_ops.merge([enter_i, enter_i])[0]

      less_op = math_ops.less(merge_i, enter_n)
      cond_op = control_flow_ops.loop_cond(less_op)
      switch_i = control_flow_ops.switch(merge_i, cond_op)

      add_i = math_ops.add(switch_i[1], enter_one)

      next_i = control_flow_ops.next_iteration(add_i)
      merge_i.op._update_input(1, next_i)

      exit_i = control_flow_ops.exit(switch_i[0])
      result = self.evaluate(exit_i)
    self.assertAllEqual(10, result)

  @test_util.run_v1_only("b/120545219")
  def testLoop_2(self):
    # [FUNCTION DOC]
    # Test (v1 only): same manual counting loop as the previous test but the NextIteration op is
    # placed on the GPU device name. Expects the exit value 10.
    with self.cached_session():
      zero = constant_op.constant(0)
      one = constant_op.constant(1)
      n = constant_op.constant(10)

      enter_i = gen_control_flow_ops.enter(zero, "foo", False)
      enter_one = gen_control_flow_ops.enter(one, "foo", True)
      enter_n = gen_control_flow_ops.enter(n, "foo", True)

      merge_i = control_flow_ops.merge([enter_i, enter_i])[0]

      less_op = math_ops.less(merge_i, enter_n)
      cond_op = control_flow_ops.loop_cond(less_op)
      switch_i = control_flow_ops.switch(merge_i, cond_op)

      add_i = math_ops.add(switch_i[1], enter_one)

      with ops.device(test.gpu_device_name()):
        next_i = control_flow_ops.next_iteration(add_i)
      merge_i.op._update_input(1, next_i)

      exit_i = control_flow_ops.exit(switch_i[0])
      result = self.evaluate(exit_i)
    self.assertAllEqual(10, result)

  @test_util.run_v1_only("b/120545219")
  def testDifferentFrame(self):
    # [FUNCTION DOC]
    # Test (v1 only): enters one placeholder into two different frames and adds the results.
    # Expects running the sum to fail with an error that the op has inputs from different frames.
    with self.cached_session():
      data = array_ops.placeholder(dtypes.float32, shape=[])
      enter_1 = gen_control_flow_ops.enter(data, "foo_1", False)
      enter_2 = gen_control_flow_ops.enter(data, "foo_2", False)
      res = math_ops.add(enter_1, enter_2)
      with self.assertRaisesOpError("has inputs from different frames"):
        res.eval(feed_dict={data: 1.0})

  @test_util.run_deprecated_v1
  def testCondBool(self):
    # [FUNCTION DOC]
    # Test (deprecated v1 mode): passes a plain Python boolean as the cond predicate. Expects a type
    # error saying the predicate must not be a Python bool.
    values = constant_op.constant(10)
    fn1 = lambda: math_ops.add(values, 1)
    fn2 = lambda: math_ops.subtract(values, 1)
    with self.assertRaisesRegex(TypeError, "must not be a Python bool"):
      _ = tf_cond.cond(False, fn1, fn2)

  @test_util.run_deprecated_v1
  def testCondInt(self):
    # [FUNCTION DOC]
    # Test (deprecated v1 mode): builds a cond on a placeholder predicate whose branches use a
    # constant; takes the gradient with respect to a different constant. Expects the gradient result
    # to be a list containing nothing (no path).
    p = array_ops.placeholder(dtypes.bool, shape=[])
    v = constant_op.constant(10)
    fn1 = lambda: math_ops.add(v, 1)
    fn2 = lambda: math_ops.subtract(v, 1)
    y = tf_cond.cond(p, fn1, fn2)
    grad = gradients_impl.gradients(y, [v])
    self.assertAllEqual([None], grad)

  def testCondOutputShape(self):
    # [FUNCTION DOC]
    # Test: cond on a constant true predicate returning a squared scalar or a subtraction. Expects
    # the static output shape to be scalar.
    x = constant_op.constant(1.0)
    b = tf_cond.cond(
        constant_op.constant(True), lambda: math_ops.square(x),
        lambda: math_ops.subtract(x, 1.))
    self.assertEqual(b.shape, tensor_shape.TensorShape([]))

  @test_util.run_v1_only("b/120545219")
  def testFetchable(self):
    # [FUNCTION DOC]
    # Test (v1 only): builds a cond over a placeholder, then for each op input in the graph runs
    # it with a feed. Inputs from fetchable ops must run; others must raise a value error saying
    # the tensor was marked as not fetchable.
    with self.cached_session() as sess:
      x = array_ops.placeholder(dtypes.float32)
      tf_cond.cond(
          constant_op.constant(True), lambda: x + 2, lambda: x + 0)
      graph = ops.get_default_graph()
      for op in graph.get_operations():
        for t in op.inputs:
          if graph.is_fetchable(t.op):
            sess.run(t, feed_dict={x: 3})
          else:
            with self.assertRaisesRegex(ValueError,
                                        "has been marked as not fetchable"):
              sess.run(t, feed_dict={x: 3})

  @test_util.disable_control_flow_v2("Not relevant")
  @test_util.run_v1_only("b/120545219")
  def testFeedable(self):
    # [FUNCTION DOC]
    # Test (v1 only, legacy control flow only): builds a while loop to 1000 and checks it can be
    # fed. Every feedable tensor can be fed; integer tensors that are not feedable must raise a
    # value error saying they may not be fed.
    with self.cached_session() as sess:
      c = constant_op.constant(2)
      i0 = constant_op.constant(0)
      r = while_loop_tf.while_loop(
          lambda i: i < 1000, lambda i: math_ops.square(c) + i, [i0])
      self.assertEqual(1000, r.eval(feed_dict={i0: 0}))
      feedable_tensors = all_feedables()
      for t in feedable_tensors:
        sess.run(r, feed_dict={t: 3})
      graph = ops.get_default_graph()
      for op in graph.get_operations():
        for t in op.inputs:
          if t not in feedable_tensors and t.dtype is dtypes.int32:
            with self.assertRaisesRegex(ValueError, "may not be fed"):
              sess.run(r, feed_dict={t: 3})

  @test_util.run_v1_only("b/120545219")
  def testCondIndexedSlices(self):
    # [FUNCTION DOC]
    # Test (v1 only): cond whose branches both return indexed slices; true branch adds one to the
    # values. Expects values [11] and indices [0].
    with self.cached_session():
      values = constant_op.constant([10])
      indices = constant_op.constant([0])
      x = indexed_slices.IndexedSlices(values, indices)
      pred = math_ops.less(1, 2)
      fn1 = lambda: indexed_slices.IndexedSlices(
          math_ops.add(x.values, 1), indices)
      fn2 = lambda: indexed_slices.IndexedSlices(
          math_ops.subtract(x.values, 1), indices)
      r = tf_cond.cond(pred, fn1, fn2)

      val = r.values
      ind = r.indices
    self.assertAllEqual([11], val)
    self.assertAllEqual([0], ind)

  def testCondMismatchedIndexedSlices(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the second branch lambda is followed by an extra positional argument (the indices
    # tensor) in the cond call, so indices is actually passed as an extra cond parameter rather than
    # being part of a branch; the call looks like a typo.
    # Test: inside a compiled function, a cond whose branches return an indexed-slices value and a
    # dense tensor. Expects a type error saying the zeroth outputs cannot be reconciled. The compiled
    # function is then called.
    @eager_def_function.function
    def foo():
      # [FUNCTION DOC]
      # HELPER: nested compiled function of the mismatched indexed-slices test; builds the values,
      # indices and indexed slices and runs the cond inside an assertion for the type error.
      # Returns nothing.
      values = constant_op.constant([10])
      indices = constant_op.constant([0])
      x = indexed_slices.IndexedSlices(values, indices)
      with self.assertRaisesRegex(TypeError,
                                  "Cannot reconcile tf.cond 0-th outputs"):
        tf_cond.cond(
            constant_op.constant(True), lambda: indexed_slices.IndexedSlices(
                math_ops.add(x.values, 1), indices),
            lambda: math_ops.add(x.values, 1), indices)
    foo()

  def testCondSparseTensor(self):
    # [FUNCTION DOC]
    # Test: cond whose branches return sparse tensors (true branch shifts indices and adds one to
    # values). Expects values [3, 5], indices [[1], [4]] and a static values shape of two.
    values = constant_op.constant([2.0, 4.0], name="values")
    indices = constant_op.constant([[0], [3]],
                                   dtype=dtypes.int64,
                                   name="indices")
    shape = constant_op.constant([10], dtype=dtypes.int64, name="dense_shape")
    x = sparse_tensor.SparseTensor(indices, values, dense_shape=shape)
    pred = math_ops.less(1, 2)
    fn1 = lambda: sparse_tensor.SparseTensor(
        indices + 1, x.values + 1, dense_shape=shape)
    fn2 = lambda: sparse_tensor.SparseTensor(
        indices, x.values - 1, dense_shape=shape)
    r = tf_cond.cond(pred, fn1, fn2)
    self.assertAllEqual([3.0, 5.0], r.values)
    self.assertAllEqual([[1], [4]], r.indices)
    self.assertAllEqual(r.values.get_shape(), (2,))

  def testCondRaggedTensor(self):
    # [FUNCTION DOC]
    # Test: cond whose branches return ragged tensors (true branch concatenates, false branch slices).
    # Expects the flat values and row splits of the true branch result.
    rt = ragged_factory_ops.constant([[1, 2], [3], [4, 5, 6]])
    pred = math_ops.less(1, 2)
    fn1 = lambda: array_ops.concat([rt + 2, [[100]]], axis=0)
    fn2 = lambda: rt[:2] - 2
    result = tf_cond.cond(pred, fn1, fn2)
    self.assertAllEqual([3, 4, 5, 6, 7, 8, 100], result.values)
    self.assertAllEqual([0, 2, 3, 6, 7], result.row_splits)

  @test_util.run_v1_only("b/120545219")
  def testCondResource(self):
    # [FUNCTION DOC]
    # Test (v1 only): cond whose predicate is a resource boolean variable and whose true branch
    # assigns the variable false under a control dependency. Expects the result 1.0.

    with self.cached_session():
      rv = resource_variable_ops.ResourceVariable(True)
      self.evaluate(variables.global_variables_initializer())
      t = ops.convert_to_tensor(1.0)

      def case():
        # [FUNCTION DOC]
        # HELPER: true branch of the resource cond test; assigns False to the resource variable and returns
        # an identity of the captured tensor under a control dependency on the assignment.
        assign = resource_variable_ops.assign_variable_op(rv.handle, False)
        with ops.control_dependencies([assign]):
          return array_ops.identity(t)

      self.assertEqual(
          1.0, self.evaluate(tf_cond.cond(rv, case, lambda: t)))

  @test_util.run_deprecated_v1
  def testCondResourceGradShape(self):
    # [FUNCTION DOC]
    # Test (deprecated v1 mode): cond choosing between two resource variables; gradients of the result
    # with respect to both variables must each have static shape of two elements.
    rv1 = resource_variable_ops.ResourceVariable([1.0, 2.0])
    rv2 = resource_variable_ops.ResourceVariable([3.0, 4.0])
    pred = constant_op.constant(True)
    result = tf_cond.cond(pred, lambda: rv1, lambda: rv2)
    grads = gradients_impl.gradients(result, [rv1, rv2])
    self.assertAllEqual(grads[0].shape.as_list(), [2])
    self.assertAllEqual(grads[1].shape.as_list(), [2])

  @test_util.run_v1_only("b/120545219")
  def testCondWithTensorArrayGrad(self):
    # [FUNCTION DOC]
    # Test (v1 only): cond whose true branch runs a map over a vector (multiplying by 2) on the GPU
    # device scope and whose false branch is a constant of ones. Checks the gradient is twos when the
    # predicate is true and zeros when false.
    with self.cached_session() as sess:
      with ops.device(test.gpu_device_name()):
        pred = array_ops.placeholder(dtypes.bool, [])
        x = constant_op.constant([1.0, 2.0, 3.0])
        y = tf_cond.cond(
            pred, lambda: map_fn.map_fn(lambda z: z * 2.0, x),
            lambda: constant_op.constant([1.0, 1.0, 1.0]))
        g = gradients_impl.gradients(y, x)[0]

      self.assertAllEqual(sess.run(g, {pred: True}), [2.0, 2.0, 2.0])
      self.assertAllEqual(sess.run(g, {pred: False}), [0.0, 0.0, 0.0])

  @test_util.run_v1_only("b/120545219")
  def testCondIndexedSlicesDifferentTypes(self):
    # [FUNCTION DOC]
    # Test (v1 only): cond whose branches return indexed slices with int32 and int64 indices.
    # Expects values [11], indices [0] and the output index dtype promoted to int64.
    with self.cached_session():
      values = constant_op.constant([10])
      i_32 = ops.convert_to_tensor([0], name="one", dtype=dtypes.int32)
      i_64 = ops.convert_to_tensor([0], name="one", dtype=dtypes.int64)
      x = indexed_slices.IndexedSlices(values, i_32)
      pred = math_ops.less(1, 2)
      fn1 = lambda: indexed_slices.IndexedSlices(
          math_ops.add(x.values, 1), i_32)
      fn2 = lambda: indexed_slices.IndexedSlices(
          math_ops.subtract(x.values, 1), i_64)
      r = tf_cond.cond(pred, fn1, fn2)

      val = r.values
      ind = r.indices
    self.assertAllEqual([11], val)
    self.assertAllEqual([0], ind)
    self.assertTrue(ind.dtype == np.int64)

  @test_util.run_v1_only("b/120545219")
  def testCondColocation(self):
    # [FUNCTION DOC]
    # Test (v1 only): variable pinned on CPU used in a cond branch. For any Switch op created for it
    # in the graph, expects its device to be the CPU.
    with self.session():
      with ops.device("/cpu:0"):
        v = variables.Variable(7.0)

      x = constant_op.constant(10.0)
      pred = math_ops.less(1.0, 2.0)
      fn1 = lambda: math_ops.add(v, 1.0)
      fn2 = lambda: math_ops.subtract(x, 1.0)
      r = tf_cond.cond(pred, fn1, fn2)

      for op in x.graph.get_operations():
        if op.name == "cond/Add/Switch":
          self.assertDeviceEqual(op.device, "/cpu:0")

  def _testCond_1(self, use_gpu):
    # [FUNCTION DOC]
    # HELPER: shared body of testCond_1, parameterized by whether to use the GPU. Builds a simple cond
    # with a true predicate (add one to 10) and expects 11.
    with self.cached_session(use_gpu=use_gpu):
      x = constant_op.constant(10)
      pred = math_ops.less(1, 2)
      fn1 = lambda: math_ops.add(x, 1)
      fn2 = lambda: math_ops.subtract(x, 1)
      r = tf_cond.cond(pred, fn1, fn2)

      result = self.evaluate(r)
    self.assertAllEqual(11, result)

  def testCond_1(self):
    # [FUNCTION DOC]
    # Test: calls the shared helper with the GPU disabled. The GPU variant is commented out pending a
    # referenced TODO about enabling GPU tests.

    self._testCond_1(use_gpu=False)
    # TODO(b/116526896): Enable GPU tests.
    # self._testCond_1(use_gpu=True)

  def testCond_2(self):
    # [FUNCTION DOC]
    # Test: cond with a false predicate choosing subtract-one from 10. Expects 9.

    with self.cached_session():
      x = constant_op.constant(10)
      r = tf_cond.cond(
          math_ops.less(1, 0), lambda: math_ops.add(x, 1),
          lambda: math_ops.subtract(x, 1))
      result = self.evaluate(r)
    self.assertAllEqual(9, result)

  def testCond_3(self):
    # [FUNCTION DOC]
    # Test: nested cond where the true branch itself contains a cond; expects 12.

    with self.cached_session():
      x = constant_op.constant(10)
      pred = math_ops.less(1, 2)
      fn1 = lambda: math_ops.add(x, 1)
      fn2 = lambda: math_ops.subtract(x, 1)
      fn3 = lambda: math_ops.add(tf_cond.cond(pred, fn1, fn2), 1)
      r = tf_cond.cond(pred, fn3, fn2)

      result = self.evaluate(r)
    self.assertAllEqual(12, result)

  @test_util.run_in_graph_and_eager_modes
  def testCondPruning(self):
    # [FUNCTION DOC]
    # Test (graph and eager modes): cond with branches performing variable assignments. In graph mode
    # only the fetched second output is run so the first branch assignment is skipped (v1 stays 7,
    # v2 becomes 2). As a compiled function both assignments of the taken branch run (v1 becomes 1).
    v1 = variables.Variable(7)
    v2 = variables.Variable(7)
    v3 = variables.Variable(7)

    def f():
      # [FUNCTION DOC]
      # HELPER: builds the cond used by the pruning test; the true branch assigns two variables, the false
      # branch assigns one and returns a constant. Asserts two outputs and returns the second.
      age = constant_op.constant(3)
      max_age = constant_op.constant(2)
      pred = math_ops.greater(age, max_age)
      fn1 = lambda: [state_ops.assign(v1, 1).op, state_ops.assign(v2, 2).op]
      fn2 = lambda: [state_ops.assign(v3, 3).op, constant_op.constant(10).op]
      r = tf_cond.cond(pred, fn1, fn2)
      self.assertEqual(len(r), 2)
      return r[1]

    f_defun = eager_def_function.function(f)

    if not context.executing_eagerly():
      with self.cached_session():
        self.evaluate(variables.global_variables_initializer())
        result = self.evaluate(f())
        self.assertEqual(True, result)
        # Only second cond result was fetched, so v1 assign shouldn't run.
        self.assertEqual(7, self.evaluate(v1))
        self.assertEqual(2, self.evaluate(v2))
        self.assertEqual(7, self.evaluate(v3))

    result = f_defun()
    self.assertEqual(True, self.evaluate(result))
    # Both v1 and v2 branch assignments should be run in defun.
    self.assertEqual(1, self.evaluate(v1))
    self.assertEqual(2, self.evaluate(v2))
    self.assertEqual(7, self.evaluate(v3))

  def testCond_5(self):
    # [FUNCTION DOC]
    # Test: loops in Python ten times calling a cond that increments a count while alive. Expects the
    # final count to be 4.
    with self.cached_session():
      alive = constant_op.constant(True, name="alive")
      count = constant_op.constant(0, name="count")

      def body(i):
        # [FUNCTION DOC]
        # HELPER: cond over alive; if true returns whether i is below three and count plus one, otherwise
        # returns the unchanged alive and count.
        return tf_cond.cond(
            alive, lambda: [math_ops.less(i, 3), math_ops.add(count, 1)],
            lambda: [alive, count])

      for i in range(10):
        alive, count = body(i)
      self.assertAllEqual(4, self.evaluate(count))

  @test_util.run_v1_only("b/120545219")
  def testCond_6(self):
    # [FUNCTION DOC]
    # Test (v1 only): cond with false predicate returning a variable of value [7]. Expects [7].
    with self.cached_session():
      v1 = variables.Variable([7])

      age = constant_op.constant(3)
      pred = math_ops.greater(age, 4)
      fn1 = lambda: age
      fn2 = lambda: v1
      r = tf_cond.cond(pred, fn1, fn2)

      self.evaluate(variables.global_variables_initializer())
      result = self.evaluate(r)
      self.assertAllEqual(np.array([7]), result)

  def testCond_7(self):
    # [FUNCTION DOC]
    # Test: cond returning a list of two tensors from the true branch; expects [11, 12].
    with self.cached_session() as sess:
      x = constant_op.constant(10)
      y = constant_op.constant(200)
      pred = math_ops.less(1, 2)
      fn1 = lambda: [math_ops.add(x, 1), math_ops.add(x, 2)]
      fn2 = lambda: [y, y]
      r = tf_cond.cond(pred, fn1, fn2)
      self.assertAllEqual([11, 12], self.evaluate(r))

  @parameterized.parameters(dtypes.float32, dtypes.float64)
  @test_util.run_v1_only("Uses tf.gradients")
  def testCondResourceGrad(self, dtype):
    # [FUNCTION DOC]
    # Test (v1 only; parameterized over float32 and float64): cond choosing a constant or a variable
    # for a false predicate; the gradient with respect to the variable should be one.
    init = constant_op.constant([7.], dtype=dtype)
    v1 = variables.Variable(init)

    age = constant_op.constant(3., dtype=dtype)
    pred = math_ops.greater(age, 4.)
    fn1 = lambda: age
    fn2 = lambda: v1
    r = tf_cond.cond(pred, fn1, fn2)

    grad = gradients_impl.gradients(r, v1)[0]
    self.evaluate(variables.global_variables_initializer())
    self.assertAllEqual(grad, [1.])

  @test_util.run_gpu_only
  @test_util.run_deprecated_v1
  def testCond_Device(self):
    # [FUNCTION DOC]
    # Test (GPU only, deprecated v1 mode): cond defined under CPU device scope whose true branch
    # is defined outside it. Expects the output device to be CPU and that the run produces exactly one
    # partitioned graph (everything stays on CPU).
    x = constant_op.constant(-10.)

    # True branch function defined outside of device scope
    def true_fn():
      # [FUNCTION DOC]
      # HELPER: true branch defined outside the device scope; returns the exponential of the captured
      # tensor.
      return math_ops.exp(x)

    with ops.device("CPU:0"):
      r = tf_cond.cond(
          constant_op.constant(True), true_fn, lambda: 0.)
      self.assertIn("cpu", r.device.lower())

    with session.Session() as sess:
      options = config_pb2.RunOptions(output_partition_graphs=True)
      run_metadata = config_pb2.RunMetadata()
      sess.run(r, options=options, run_metadata=run_metadata)
      # We expect that everything runs on CPU, even if GPU is available.
      self.assertEqual(len(run_metadata.partition_graphs), 1)

  def _count_matching_switch_nodes_on_device(self, run_metadata, device_str,
                                             dtype):
    # [FUNCTION DOC]
    # HELPER: inspects run metadata partition graphs. Takes the run metadata, a device substring and a
    # data type. Asserts that exactly one partition graph is on that device and returns the number of
    # Switch nodes of that type in it.
    # Returns the number of Switch nodes with type dtype placed on
    # `device_str`.
    device_graphs = [
        g for g in run_metadata.partition_graphs
        if device_str in g.node[0].device
    ]
    self.assertLen(device_graphs, 1)
    switch_nodes = [
        n for n in device_graphs[0].node
        if n.op == "Switch" and n.attr["T"].type == dtype.as_datatype_enum
    ]
    return len(switch_nodes)

  @test_util.run_gpu_only
  @test_util.run_deprecated_v1
  def testCondSwitchColocatedWithInputWhenInputExplicitlyPlacedOnCPU(self):
    # [FUNCTION DOC]
    # Test (GPU only): input tensor explicitly placed on CPU and used in the true branch; the loop
    # optimizer is turned off. Expects two partition graphs and that the one Switch of float type
    # ends up on CPU and none on GPU.
    x = array_ops.placeholder(dtypes.float32)

    # `arg` is used in the cond then branch so a Switch node is created for it.
    # We test that the Switch node gets placed on the same device as `arg`.
    # We force `arg` to be on CPU here.
    with ops.device("CPU:0"):
      arg = x + 10.

    def true_fn():
      # [FUNCTION DOC]
      # HELPER: true branch; returns the captured tensor plus one, built under the CPU device scope.
      with ops.device("CPU:0"):
        return arg + 1

    r = tf_cond.cond(constant_op.constant(True), true_fn, lambda: 0.)

    # Disable Loop_optimizer grappler pass for this test because it replaces
    # Switch with Identity when it's part of a dead branch.
    config = config_pb2.ConfigProto()
    config.graph_options.rewrite_options.loop_optimization = (
        rewriter_config_pb2.RewriterConfig.OFF)

    with self.session(config=config) as sess:
      run_metadata = config_pb2.RunMetadata()
      options = config_pb2.RunOptions(output_partition_graphs=True)
      sess.run(
          r, feed_dict={x: -10.}, options=options, run_metadata=run_metadata)
      self.assertLen(run_metadata.partition_graphs, 2)
      # Check that the Switch for `arg` gets placed on CPU.
      self.assertEqual(
          self._count_matching_switch_nodes_on_device(run_metadata, "CPU",
                                                      dtypes.float32), 1)
      self.assertEqual(
          self._count_matching_switch_nodes_on_device(run_metadata, "GPU",
                                                      dtypes.float32), 0)

  @test_util.run_gpu_only
  @test_util.run_deprecated_v1
  def testCondSwitchColocatedWithInputWhenInputPlacedOnCPU(self):
    # [FUNCTION DOC]
    # Test (GPU only): the cond input is a dataset (CPU-only kernel); its cardinality is used in the
    # true branch. Expects two partition graphs, one variant-type Switch on CPU and none on GPU.
    x = array_ops.placeholder(dtypes.float32)

    # `arg` is used in the cond then branch so a Switch node is created for it.
    # We test that the Switch node gets placed on the same device as `arg`.
    # Since arg is a dataset (and only has a CPU kernel), it gets placed on CPU
    # by placer.
    arg = dataset_ops.Dataset.range(8)

    def true_fn():
      # [FUNCTION DOC]
      # HELPER: true branch; returns the cardinality of the captured dataset.
      return cardinality.cardinality(arg)

    r = tf_cond.cond(
        constant_op.constant(True), true_fn,
        lambda: constant_op.constant(0, dtypes.int64))

    # Disable Loop_optimizer grappler pass for this test because it replaces
    # Switch with Identity when it's part of a dead branch.
    config = config_pb2.ConfigProto()
    config.graph_options.rewrite_options.loop_optimization = (
        rewriter_config_pb2.RewriterConfig.OFF)

    with session.Session(config=config) as sess:
      run_metadata = config_pb2.RunMetadata()
      options = config_pb2.RunOptions(output_partition_graphs=True)
      sess.run(
          r, feed_dict={x: -10.}, options=options, run_metadata=run_metadata)
      self.assertLen(run_metadata.partition_graphs, 2)
      # Check that the Switch for `arg` gets placed on CPU.
      self.assertEqual(
          self._count_matching_switch_nodes_on_device(run_metadata, "CPU",
                                                      dtypes.variant), 1)
      self.assertEqual(
          self._count_matching_switch_nodes_on_device(run_metadata, "GPU",
                                                      dtypes.variant), 0)

  @test_util.run_gpu_only
  @test_util.run_deprecated_v1
  def testCondSwitchColocatedWithInputWhenInputOnGPU(self):
    # [FUNCTION DOC]
    # Test (GPU only): the cond input is placed on GPU by default. Expects two partition graphs, no
    # float Switch on CPU and one on GPU.
    x = array_ops.placeholder(dtypes.float32)

    # `arg` is used in the cond then branch so a Switch node is created for it.
    # We test that the Switch node gets placed on the same device as `arg`.
    # Note: `arg` gets placed on GPU by default by the placer.
    arg = x + 10.

    def true_fn():
      # [FUNCTION DOC]
      # HELPER: true branch; returns the captured tensor plus one, built under the CPU device scope.
      with ops.device("CPU:0"):
        return arg + 1

    r = tf_cond.cond(constant_op.constant(True), true_fn, lambda: 0.)

    # Disable Loop_optimizer grappler pass for this test because it replaces
    # Switch with Identity when it's part of a dead branch.
    config = config_pb2.ConfigProto()
    config.graph_options.rewrite_options.loop_optimization = (
        rewriter_config_pb2.RewriterConfig.OFF)

    with session.Session(config=config) as sess:
      run_metadata = config_pb2.RunMetadata()
      options = config_pb2.RunOptions(output_partition_graphs=True)
      sess.run(
          r, feed_dict={x: -10.}, options=options, run_metadata=run_metadata)
      self.assertEqual(len(run_metadata.partition_graphs), 2)
      # Check that the Switch for `arg` gets placed on GPU.
      self.assertEqual(
          self._count_matching_switch_nodes_on_device(run_metadata, "CPU",
                                                      dtypes.float32), 0)
      self.assertEqual(
          self._count_matching_switch_nodes_on_device(run_metadata, "GPU",
                                                      dtypes.float32), 1)

  def testCondAccessTrueBranchTensorInFalseBranchRaises(self):
    # [FUNCTION DOC]
    # Test: inside a compiled function, the false branch reads a tensor created in the true branch.
    # Expects a value error saying the tensor in the true function is accessed from the false function.

    @eager_def_function.function
    def f():
      # [FUNCTION DOC]
      # HELPER: compiled function building the cond; the true branch names an op in a shared dictionary
      # and the false branch reads that same tensor.
      c = constant_op.constant(1.)
      inputs = {"c": c}

      def true_fn(inputs):
        # [FUNCTION DOC]
        # HELPER: replaces the dictionary entry with a named identity and returns it (leaks a branch-local
        # tensor).
        inputs["c"] = array_ops.identity(inputs["c"], name="true_branch")
        return inputs["c"]

      def false_fn(inputs):
        # [FUNCTION DOC]
        # HELPER: returns an identity of the dictionary entry (the tensor leaked from the true branch).
        return array_ops.identity(inputs["c"])

      pred = constant_op.constant(True)
      return tf_cond.cond(
          pred, lambda: true_fn(inputs), lambda: false_fn(inputs))

    prefix = "cond/" if context.executing_eagerly() else ""

    with self.assertRaisesRegex(
        ValueError,
        "Tensor %strue_branch:0 in true_fn is accessed from false_fn." %
        prefix):
      f()

  def testSwitchCaseAccessBranch1TensorInBranch4Raises(self):
    # [FUNCTION DOC]
    # Test: switch_case in a compiled function where branch 4 reads a tensor created in branch 1.
    # Expects a value error that the tensor in branch 1 is accessed from branch 4.

    @eager_def_function.function
    def f():
      # [FUNCTION DOC]
      # HELPER: compiled function building the switch_case over five branches with index two.
      c = constant_op.constant(1.)
      inputs = {"c": c}

      def br1_fn(inputs):
        # [FUNCTION DOC]
        # HELPER: branch one; stores a named identity in a shared dictionary and returns it.
        inputs["c"] = array_ops.identity(inputs["c"], name="br1_identity")
        return inputs["c"]

      def br4_fn(inputs):
        # [FUNCTION DOC]
        # HELPER: branch four; returns an identity of the dictionary entry made in branch one.
        return array_ops.identity(inputs["c"])

      def other_fn():
        # [FUNCTION DOC]
        # HELPER: neutral branch; returns an identity of the captured constant.
        return array_ops.identity(c)

      return control_flow_switch_case.switch_case(
          constant_op.constant(2), [
              other_fn, lambda: br1_fn(inputs), other_fn, other_fn,
              lambda: br4_fn(inputs)
          ])

    prefix = "switch_case/indexed_case/" if context.executing_eagerly() else ""
    with self.assertRaisesRegex(
        ValueError, "Tensor %sbr1_identity:0 in branch 1 is "
        "accessed from branch 4." % prefix):
      f()

  def testCondListOutput(self):
    # [FUNCTION DOC]
    # Test: cond whose branches return lists of tensors. Expects the list [210, 210].
    with self.cached_session() as sess:
      x = constant_op.constant(10)
      y = constant_op.constant(200)
      pred = math_ops.less(1, 2)
      fn1 = lambda: [math_ops.add(x, y), math_ops.add(x, y)]
      fn2 = lambda: [y, y]
      r = tf_cond.cond(pred, fn1, fn2)
      test_result = self.evaluate(r)
      self.assertListEqual([210, 210], test_result)

  def testTupleOutput(self):
    # [FUNCTION DOC]
    # Test: cond branches returning tuples. Expects the tuple (210, 210).
    with self.cached_session() as sess:
      x = constant_op.constant(10)
      y = constant_op.constant(200)
      pred = math_ops.less(1, 2)
      fn1 = lambda: (math_ops.add(x, y), math_ops.add(x, y))
      fn2 = lambda: (y, y)
      r = tf_cond.cond(pred, fn1, fn2)
      test_result = self.evaluate(r)
      self.assertTupleEqual((210, 210), test_result)

  def testDictOutput(self):
    # [FUNCTION DOC]
    # Test: cond branches returning dictionaries. Expects a dictionary with both values 210.
    with self.cached_session() as sess:
      x = constant_op.constant(10)
      y = constant_op.constant(200)
      pred = math_ops.less(1, 2)
      fn1 = lambda: {"a": math_ops.add(x, y), "b": math_ops.add(x, y)}
      fn2 = lambda: {"a": y, "b": y}
      r = tf_cond.cond(pred, fn1, fn2)
      test_result = self.evaluate(r)
      self.assertDictEqual({"a": 210, "b": 210}, test_result)

  def testEmbeddedListOutput(self):
    # [FUNCTION DOC]
    # Test: cond with strict mode on, branches returning a list nested in a list. Expects the nested
    # structure [[210, 210]].
    x = constant_op.constant(10)
    y = constant_op.constant(200)
    pred = math_ops.less(1, 2)
    fn1 = lambda: [[math_ops.add(x, y), math_ops.add(x, y)]]
    fn2 = lambda: [[y, y]]
    # Pass strict=True flag as cond_v2 allows for tensors to be
    # in nested output structures as singletons
    r = tf_cond.cond(pred, fn1, fn2, strict=True)
    test_result = self.evaluate(r)
    self.assertListEqual([[210, 210]], test_result)

  def testEmbeddedTupleOutput(self):
    # [FUNCTION DOC]
    # Test: cond branches return doubly-parenthesized tuples (which are just one flat tuple).
    # Expects the tuple (210, 210).
    with self.cached_session() as sess:
      x = constant_op.constant(10)
      y = constant_op.constant(200)
      pred = math_ops.less(1, 2)
      fn1 = lambda: ((math_ops.add(x, y), math_ops.add(x, y)))
      fn2 = lambda: ((y, y))
      r = tf_cond.cond(pred, fn1, fn2)
      test_result = self.evaluate(r)
      self.assertTupleEqual(((210, 210)), test_result)

  def testEmbeddedDictOutput(self):
    # [FUNCTION DOC]
    # Test: cond branches return dictionaries of dictionaries. Expects the nested dictionary of 210s.
    with self.cached_session() as sess:
      x = constant_op.constant(10)
      y = constant_op.constant(200)
      pred = math_ops.less(1, 2)
      fn1 = lambda: {"a": {"c": math_ops.add(x, y)},
                     "b": {"d": math_ops.add(x, y)}}
      fn2 = lambda: {"a": {"c": y},
                     "b": {"d": y}}
      r = tf_cond.cond(pred, fn1, fn2)
      test_result = self.evaluate(r)
      self.assertDictEqual({"a": {"c": 210}, "b": {"d": 210}}, test_result)

  @test_util.run_v1_only("b/120545219")
  def testCheckNestedOutputStruct(self):
    # [FUNCTION DOC]
    # Test (v1 only): cond branches return dictionaries with different keys. Expects a type error under
    # control flow v2 or a value error under v1, with the respective structure-mismatch message.
    with self.cached_session() as sess:
      x = constant_op.constant(10)
      y = constant_op.constant(200)
      pred = math_ops.less(1, 2)
      fn1 = lambda: {"a": math_ops.add(x, y), "b": math_ops.add(x, y)}
      fn2 = lambda: {"c": y, "d": y}
      v1_msg = "The two structures don't have the same nested structure"
      v2_msg = ("true_fn and false_fn arguments to tf.cond must have the same "
                "number, type, and overall structure of return values.")
      with self.assertRaisesRegex(
          TypeError if control_flow_util.ENABLE_CONTROL_FLOW_V2 else ValueError,
          v2_msg if control_flow_util.ENABLE_CONTROL_FLOW_V2 else v1_msg):
        tf_cond.cond(pred, fn1, fn2)

  @test_util.run_v1_only("b/120545219")
  def testCondWithControl(self):
    # [FUNCTION DOC]
    # Test (v1 only): true branch has a control dependency on a fed placeholder. Feeding 5 expects the
    # branch result 5 (a plus two).
    with self.cached_session() as sess:
      control_holder = array_ops.placeholder(dtypes.float32, shape=())
      a = constant_op.constant(3)

      def true_branch():
        # [FUNCTION DOC]
        # HELPER: adds one to the captured value under a control dependency on the placeholder, then
        # returns that value plus two.
        with ops.control_dependencies([control_holder]):
          _ = a + 1
        return a + 2

      r = tf_cond.cond(
          constant_op.constant(True), true_branch,
          lambda: constant_op.constant(1))
      result = sess.run(r, feed_dict={control_holder: 5.})
      self.assertEqual(5, result)

  @test_util.run_v1_only("b/120545219")
  def testUninitializedRefIdentity(self):
    # [FUNCTION DOC]
    # Test (v1 only): ref-switches an uninitialized variable on whether it is initialized and
    # applies ref identity on both outputs. Checks that identity accepts an uninitialized reference
    # and the merged result is the assigned value [1.0].
    with self.cached_session() as sess:
      v = gen_state_ops.variable(
          shape=[1],
          dtype=dtypes.float32,
          name="v",
          container="",
          shared_name="")
      inited = state_ops.is_variable_initialized(v)
      v_f, v_t = control_flow_ops.ref_switch(v, inited)
      # Both v_f and v_t are uninitialized references. However, an actual use
      # of the reference in the 'true' branch in the 'tf.identity' op will
      # not 'fire' when v is uninitialized, so this is a valid construction.
      # This test tests that ref_identity allows uninitialized ref as input
      # so that this construction is allowed.
      v_f_op = gen_array_ops.ref_identity(v_f)
      v_t_op = gen_array_ops.ref_identity(v_t)
      with ops.control_dependencies([v_f_op]):
        assign_v = state_ops.assign(v, [1.0])
      with ops.control_dependencies([v_t_op]):
        orig_v = array_ops.identity(v)
      merged_op = control_flow_ops.merge([assign_v, orig_v])
      self.assertAllEqual([1.0], self.evaluate(merged_op.output))

  def testCondSwitchIdentity(self):
    # [FUNCTION DOC]
    # Test: with optimization enabled, cond on a true predicate whose false branch has a failing
    # assert. Expects evaluation to succeed, ensuring the receive identity is not optimized away.
    # Make sure the recv identity is not removed by optimization.
    with session.Session(config=opt_cfg()) as sess:
      pred = constant_op.constant(True)

      def fn1():
        # [FUNCTION DOC]
        # HELPER: returns a no-op.
        return control_flow_ops.no_op()

      def fn2():
        # [FUNCTION DOC]
        # HELPER: returns an assert on false with the message "Wrong branch" (must not run).
        return control_flow_assert.Assert(False, ["Wrong branch!!!"])

      r = tf_cond.cond(pred, fn1, fn2)
      self.evaluate(r)

  def testCondRecvIdentity(self):
    # [FUNCTION DOC]
    # Test: like the switch identity test but the predicate is placed on the GPU device name and the
    # false branch is on CPU. Expects evaluation to succeed.
    # Make sure the switch identity is not removed by optimization.
    with session.Session(config=opt_cfg()) as sess:
      with ops.device(test.gpu_device_name()):
        pred = constant_op.constant(True)

      def fn1():
        # [FUNCTION DOC]
        # HELPER: returns a no-op.
        return control_flow_ops.no_op()

      def fn2():
        # [FUNCTION DOC]
        # HELPER: under the CPU device scope returns a failing assert (must not run).
        with ops.device("/cpu:0"):
          return control_flow_assert.Assert(False, ["Wrong branch!!!"])

      r = tf_cond.cond(pred, fn1, fn2)
      self.evaluate(r)

  @test_util.run_deprecated_v1
  @test_util.enable_control_flow_v2
  def testDisableLoweringSwitchMerge(self):
    # [FUNCTION DOC]
    # Test (deprecated v1 mode, with control flow v2 on): runs a cond under a normal executor, then
    # under the single-threaded executor which disables lowering. Both must return true. The first
    # trace's node stats must contain a node with "switch" in its name (lowered), the second none.
    # Skipped when a GPU exists.
    if test_util.is_gpu_available():
      self.skipTest(
          "Single threaded executor doesn't support partitioned graphs.  "
          "Skipping GPU test.")
    # Make pred feedable to ensure we don't constant-fold it out.
    run_opts = config_pb2.RunOptions(
        trace_level=config_pb2.RunOptions.FULL_TRACE)
    run_metadata_no_lowering = config_pb2.RunMetadata()
    run_metadata_with_lowering = config_pb2.RunMetadata()

    config = opt_cfg(do_constant_folding=False)

    pred = array_ops.placeholder_with_default(
        constant_op.constant(True), shape=())
    r = tf_cond.cond(pred, lambda: True, lambda: False)

    with session.Session(config=config) as sess:
      r_value = sess.run(
          r, options=run_opts, run_metadata=run_metadata_with_lowering)
      self.assertEqual(r_value, True)

    # Use the single threaded executor, which disables control flow lowering.
    config.experimental.executor_type = "SINGLE_THREADED_EXECUTOR"
    with session.Session(config=config) as sess:
      r_value = sess.run(
          r, options=run_opts, run_metadata=run_metadata_no_lowering)
      self.assertEqual(r_value, True)

    self.assertTrue(  # pylint: disable=g-complex-comprehension
        any("switch" in ns.node_name
            for dev_stat in run_metadata_with_lowering.step_stats.dev_stats
            for ns in dev_stat.node_stats))

    self.assertTrue(  # pylint: disable=g-complex-comprehension
        all("switch" not in ns.node_name
            for dev_stat in run_metadata_no_lowering.step_stats.dev_stats
            for ns in dev_stat.node_stats))

  @test_util.run_v1_only("b/120545219")
  def testCondGrad_1(self):
    # [FUNCTION DOC]
    # Test (v1 only): gradient of a cond with identity branches with respect to its input is 1.0.
    with self.cached_session():
      x = constant_op.constant(10.0, name="x")
      pred = math_ops.less(1, 2)
      fn1 = lambda: array_ops.identity(x)
      fn2 = lambda: array_ops.identity(x)
      r = tf_cond.cond(pred, fn1, fn2)

      grad = gradients_impl.gradients(r, [x])[0]
      self.assertAllEqual(1.0, self.evaluate(grad))

  @test_util.run_deprecated_v1
  @test_util.enable_control_flow_v2
  def testCondComputeGradAfterSessRunFails(self):
    # [FUNCTION DOC]
    # Test (deprecated v1 mode, control flow v2): runs the cond first and then asks for the gradient.
    # Expects an invalid-argument error that the gradient connects to an invalid output of the cond
    # node (suggesting the output-all-intermediates option).
    with self.cached_session():
      x = constant_op.constant(10.0, name="x")
      pred = math_ops.less(1, 2)

      def true_fn():
        # [FUNCTION DOC]
        # HELPER: returns x to the fourth power via two products.
        a = x * x
        return a * a

      def false_fn():
        # [FUNCTION DOC]
        # HELPER: returns x squared.
        return x * x

      r = tf_cond.cond(pred, true_fn, false_fn)

      self.assertAllEqual(r, 10000.)
      grad = gradients_impl.gradients(r, [x])[0]
      with self.assertRaisesRegex(
          errors_impl.InvalidArgumentError,
          r"Connecting to invalid output 1 of source node cond which has 1 "
          r"outputs. Try using "
          "tf.compat.v1.experimental.output_all_intermediates\(True\)."):
        self.evaluate(grad)

  @test_util.run_deprecated_v1
  @test_util.enable_output_all_intermediates
  def testCondComputeGradAfterSessRun(self):
    # [FUNCTION DOC]
    # Test (deprecated v1 mode, output-all-intermediates enabled): runs the cond (10000) then
    # computes the gradient and expects 4000.
    with self.cached_session():
      x = constant_op.constant(10.0, name="x")
      pred = math_ops.less(1, 2)

      def true_fn():
        # [FUNCTION DOC]
        # HELPER: returns x to the fourth power.
        a = x * x
        return a * a

      def false_fn():
        # [FUNCTION DOC]
        # HELPER: returns x squared.
        return x * x

      r = tf_cond.cond(pred, true_fn, false_fn)

      self.assertAllEqual(r, 10000.)
      grad = gradients_impl.gradients(r, [x])[0]
      self.assertAllEqual(grad, 4000.)

  @test_util.run_deprecated_v1
  @test_util.enable_output_all_intermediates
  def testNestedCondComputeGradAfterSessRun(self):
    # [FUNCTION DOC]
    # Test (deprecated v1 mode, output-all-intermediates enabled): same as above but the true branch
    # contains a nested cond. Expects value 10000 and gradient 4000.
    with self.cached_session():
      x = constant_op.constant(10.0, name="x")
      pred = math_ops.less(1, 2)

      def true_fn():
        # [FUNCTION DOC]
        # HELPER: returns the result of an inner cond on a true constant between two inner branches.

        def inner_true_fn():
          # [FUNCTION DOC]
          # HELPER: returns x to the fourth power.
          a = x * x
          return a * a

        def inner_false_fn():
          # [FUNCTION DOC]
          # HELPER: returns x squared.
          return x * x

        return tf_cond.cond(
            constant_op.constant(True), inner_true_fn, inner_false_fn)

      def false_fn():
        # [FUNCTION DOC]
        # HELPER: returns x squared.
        return x * x

      r = tf_cond.cond(pred, true_fn, false_fn)

      self.assertAllEqual(r, 10000.)
      grad = gradients_impl.gradients(r, [x])[0]
      self.assertAllEqual(grad, 4000.)

  @test_util.run_deprecated_v1
  def testCondGrad_2(self):
    # [FUNCTION DOC]
    # Test (deprecated v1 mode): cond on a fed integer comparison, branches multiply x by 42 or 3.
    # Expects gradient 42 for input 1 and 3 for input 3.
    with self.cached_session():
      c = array_ops.placeholder(dtypes.int32, shape=[])
      x = constant_op.constant(10.0)
      pred = math_ops.less(c, 2)
      fn1 = lambda: math_ops.multiply(x, 42.0)
      fn2 = lambda: math_ops.multiply(x, 3.0)
      r = tf_cond.cond(pred, fn1, fn2)

      grad = gradients_impl.gradients(r, [x])[0]
      self.assertAllEqual(42.0, grad.eval(feed_dict={c: 1}))
      self.assertAllEqual(3.0, grad.eval(feed_dict={c: 3}))

  @test_util.disable_control_flow_v2(
      "b/110550782 (gradient w.r.t external variable)")
  @test_util.run_deprecated_v1
  def testCondGrad_3(self):
    # [FUNCTION DOC]
    # Test (legacy control flow, deprecated v1): gradient computed inside a cond branch with respect
    # to an outer tensor. Expects 980 when the true branch runs and 30 otherwise.
    with self.cached_session():
      c = array_ops.placeholder(dtypes.int32, shape=[])
      ox = constant_op.constant(10.0)
      pred = math_ops.less(c, 2)

      def fn1(x):
        # [FUNCTION DOC]
        # HELPER: takes x, squares it and returns its gradient with respect to the captured outer tensor.
        m = x * x
        return gradients_impl.gradients(m, [ox])[0]

      fn2 = lambda: math_ops.multiply(ox, 3.0)
      y = math_ops.multiply(7.0, ox)
      r = tf_cond.cond(pred, lambda: fn1(y), fn2)

      self.assertAllEqual(980.0, r.eval(feed_dict={c: 1}))
      self.assertAllEqual(30.0, r.eval(feed_dict={c: 3}))

  @test_util.run_deprecated_v1
  def testCondGradMultiDevice(self):
    # [FUNCTION DOC]
    # Test (deprecated v1 mode): cond on cpu 0, gradient on cpu 1 and second gradient on cpu 0, with
    # two CPU devices and soft placement. Expects gradient 4 or 0 by predicate; under v1 the second
    # derivative is none, under v2 it is zero.
    config = config_pb2.ConfigProto(device_count={"CPU": 2},
                                    allow_soft_placement=True)
    with self.cached_session(config=config) as sess:
      pred = array_ops.placeholder(dtypes.bool, [])
      x = array_ops.placeholder(dtypes.float32)
      y = array_ops.placeholder(dtypes.float32)

      with ops.device("/cpu:0"):
        z = tf_cond.cond(pred, lambda: x * y * 2.0, lambda: 2.0)

      with ops.device("/cpu:1"):
        grad = gradients_impl.gradients(z, x)[0]

      with ops.device("/cpu:0"):
        grad_grad = gradients_impl.gradients(grad, x)[0]

      self.assertEqual(sess.run(grad, {pred: True, x: 1.0, y: 2.0}), 4.0)
      self.assertEqual(sess.run(grad, {pred: False, x: 1.0, y: 2.0}), 0.0)

      # v1 control flow gets None second derivative for some reason.
      if not control_flow_util.ENABLE_CONTROL_FLOW_V2:
        self.assertIsNone(grad_grad)
        return

      self.assertEqual(sess.run(grad_grad, {pred: True, x: 1.0, y: 2.0}), 0.0)
      self.assertEqual(sess.run(grad_grad, {pred: False, x: 1.0, y: 2.0}), 0.0)

  @test_util.run_v1_only("b/120545219")
  def testNestedCond_Simple(self):
    # [FUNCTION DOC]
    # Test (v1 only): gradient with respect to x of nested conds for true and false outer predicates.
    # Both expect 1.0.
    with self.cached_session():
      x = constant_op.constant(0., name="X")
      y = tf_cond.cond(
          constant_op.constant(True), lambda: x,
          lambda: tf_cond.cond(x < 1., lambda: x, lambda: x))
      result = gradients_impl.gradients(y, x)[0]
      self.assertEqual(1.0, self.evaluate(result))

      z = tf_cond.cond(
          constant_op.constant(False), lambda: x,
          lambda: tf_cond.cond(x < 1., lambda: x, lambda: x))
      result = gradients_impl.gradients(z, x)[0]
      self.assertEqual(1.0, self.evaluate(result))

  @test_util.run_v1_only("b/120545219")
  def testCondGrad_Gather(self):
    # [FUNCTION DOC]
    # Test (v1 only): cond between an identity and a gather of a variable; checks gradients (as dense
    # values via a helper) are [1, 1] for the first branch and [0, 2] for the second, and that with v2
    # the gradient is an indexed slices.
    with self.cached_session() as sess:
      v1 = variables.Variable([1.0, 42.0])
      c = array_ops.placeholder(dtypes.int32, shape=[])
      pred = math_ops.less(c, 2)
      fn1 = lambda: array_ops.identity(v1)
      fn2 = lambda: array_ops.gather(v1, [1, 1])
      r = tf_cond.cond(pred, fn1, fn2)
      # The following `grad` is a Tensor since it is the aggregation of an
      # IndexedSlice and a Tensor. It is an `IndexedSlices` with control flow
      # v2.
      grad = gradients_impl.gradients(r, [v1])[0]
      self.evaluate(variables.global_variables_initializer())

      if control_flow_util.ENABLE_CONTROL_FLOW_V2:
        self.assertIsInstance(grad, indexed_slices.IndexedSlices)

      grad_value = sess.run(grad, feed_dict={c: 1})
      self.assertAllEqual(gradient_checker_v2._to_numpy(grad_value), [1.0, 1.0])

      grad_value = sess.run(grad, feed_dict={c: 3})
      self.assertAllEqual(gradient_checker_v2._to_numpy(grad_value), [0.0, 2.0])

  @test_util.run_deprecated_v1
  def testCondGrad_ResourceVarSparseRead(self):
    # [FUNCTION DOC]
    # Test (deprecated v1 mode): cond using sparse read of a resource variable. Expects the gradient
    # to be an indexed-slices value representing ones in rows one and two.
    # NOTE(skyewm): this test is interesting because the
    # ResourceVariable.sparse_read gradient function returns IndexedSlices.
    var = resource_variable_ops.ResourceVariable(
        np.ones((4, 2), dtype=np.float32))
    x = constant_op.constant(1.0)
    r = tf_cond.cond(
        constant_op.constant(True),
        lambda: x * math_ops.reduce_sum(var.sparse_read([1, 2])),
        lambda: constant_op.constant(np.zeros((2, 3)),
                                     dtype=dtypes.float32))
    grad = gradients_impl.gradients(r, var)[0]

    self.evaluate(variables.global_variables_initializer())
    grad_val = self.evaluate(grad)
    self.assertIsInstance(grad_val, indexed_slices.IndexedSlicesValue)
    self.assertAllEqual(gradient_checker_v2._to_numpy(grad_val), [[0., 0.],
                                                                  [1., 1.],
                                                                  [1., 1.],
                                                                  [0., 0.]])

  def testCondGrad_MultiGather(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the final check repeats the assertion on the x1 gradient type instead of checking
    # the x2 gradient (a dense scalar), so the type check for the x2 gradient is missing.
    # Test: cond branches combine sparse read and gather to produce indexed-slices gradients; in a
    # compiled function computes gradients for the variable and two constants and checks each value.
    # NOTE(skyewm): this test is interesting because the array_ops.gather and
    # ResourceVariable.sparse_read gradient functions returns IndexedSlices.
    var = resource_variable_ops.ResourceVariable(
        np.ones((4, 2), dtype=np.float32))
    x1 = constant_op.constant(np.ones((3, 3), dtype=np.float32))
    x2 = constant_op.constant(2.0)

    def true_fn():
      # [FUNCTION DOC]
      # HELPER: returns the sparse read of rows one and two, a gathered row times x2 and x2 times ones.
      y1 = var.sparse_read([1, 2])
      y2 = array_ops.gather(x1, [2]) * x2
      y3 = x2 * [1., 1., 1.]
      return y1, y2, y3

    def false_fn():
      # [FUNCTION DOC]
      # HELPER: returns zeros, a gathered row times x2 and a gathered row.
      y1 = np.zeros((2, 2), dtype=np.float32)
      y2 = array_ops.gather(x1, [2]) * x2
      y3 = array_ops.gather(x1, [2])
      return y1, y2, y3

    @eager_def_function.function
    def foo():
      # [FUNCTION DOC]
      # HELPER: compiled function running the cond and returning the gradients of the result with
      # respect to the variable and both constants.
      r = tf_cond.cond(constant_op.constant(True), true_fn, false_fn)
      return gradients_impl.gradients(r, [var, x1, x2])

    grad = foo()
    self.evaluate(variables.global_variables_initializer())
    var_grad, x1_grad, x2_grad = self.evaluate(grad)
    self.assertIsInstance(var_grad, indexed_slices.IndexedSlicesValue)
    self.assertAllEqual(gradient_checker_v2._to_numpy(var_grad), [[0., 0.],
                                                                  [1., 1.],
                                                                  [1., 1.],
                                                                  [0., 0]])
    self.assertIsInstance(x1_grad, indexed_slices.IndexedSlicesValue)
    self.assertAllEqual(gradient_checker_v2._to_numpy(x1_grad), [[0., 0., 0.],
                                                                 [0., 0., 0.],
                                                                 [2., 2., 2.]])
    self.assertIsInstance(x1_grad, indexed_slices.IndexedSlicesValue)
    self.assertEqual(gradient_checker_v2._to_numpy(x2_grad), 6.)

  @test_util.run_v1_only("b/120545219")
  def testCondPredicateTensor(self):
    # [FUNCTION DOC]
    # Test (v1 only): regression test for lowering a predicate taken from a non-first output of an op.
    # Expects cond to return 1.0.
    """Regression test for lowering predicate from non-first output of an op."""

    @eager_def_function.function
    def foo():
      # [FUNCTION DOC]
      # HELPER: compiled function returning a string constant and a true boolean constant.
      return constant_op.constant("foo"), constant_op.constant(True)

    r = tf_cond.cond(foo()[1], lambda: 1.0, lambda: 2.0)
    self.assertEqual(self.evaluate(r), 1.0)

  @test_util.run_v1_only("Tests Session.run() pruning logic.")
  def testCondFeedConstantPredicate(self):
    # [FUNCTION DOC]
    # Test (v1 only): feeding a constant predicate (false or true) to a cond changes the branch run.
    # Expects 37 when fed false, 0 when fed true or not fed.
    with self.cached_session() as sess:
      value = constant_op.constant(37.0)
      predicate = constant_op.constant(True)
      cond_output = tf_cond.cond(
          predicate, lambda: constant_op.constant(0.0), lambda: value)
      result = array_ops.identity(cond_output)
      self.assertEqual(37.0, sess.run(result, feed_dict={predicate: False}))
      self.assertEqual(0.0, sess.run(result, feed_dict={predicate: True}))
      self.assertEqual(0.0, sess.run(result))

  @test_util.run_v1_only("Tests Session.run() pruning logic.")
  def testCondFeedPlaceholderWithDefaultPredicate(self):
    # [FUNCTION DOC]
    # Test (v1 only): same as above but the predicate is a placeholder with a true default.
    with self.cached_session() as sess:
      value = constant_op.constant(37.0)
      predicate = array_ops.placeholder_with_default(
          constant_op.constant(True), [])
      cond_output = tf_cond.cond(
          predicate, lambda: constant_op.constant(0.0), lambda: value)
      result = array_ops.identity(cond_output)
      self.assertAllEqual(37.0, sess.run(result, feed_dict={predicate: False}))
      self.assertAllEqual(0.0, sess.run(result, feed_dict={predicate: True}))
      self.assertAllEqual(0.0, sess.run(result))

  def testCondTensorDeps(self):
    # [FUNCTION DOC]
    # Test: a compiled function creating a control dependency on an outer eager/graph tensor must
    # trace (get a concrete function) without raising.
    t = array_ops.identity(1.)

    @eager_def_function.function
    def f():
      # [FUNCTION DOC]
      # HELPER: compiled function returning a constant under a control dependency on the outer tensor.
      with ops.control_dependencies([t]):
        return array_ops.identity(2.)

    f.get_concrete_function()

  @test_util.run_in_graph_and_eager_modes
  def testCondAutoControlDeps(self):
    # [FUNCTION DOC]
    # Test (graph and eager modes; skipped on GPU): branch functions emit print ops A, B, C (C inside
    # a control dependency). Expects in v1 graph mode that only C prints; in compiled functions (v2)
    # all of A, B, C print in order, and with wrap_function only C prints. Covers nested conds too.
    # Captures stderr to check the output.
    if test_util.is_gpu_available():
      self.skipTest("b/128676188 causes OOM on opensource gpu tests")

    print_prefix = "testCondAutoControlDeps: "

    def branch_fn():
      # [FUNCTION DOC]
      # HELPER: enqueues prints A and B, then C as a control dependency, and returns constant 10.
      enqueue_print_op("A")
      enqueue_print_op("B")
      with ops.control_dependencies([enqueue_print_op("C")]):
        return constant_op.constant(10)

    def build_cond():
      # [FUNCTION DOC]
      # HELPER: builds a cond on a true constant between branch_fn and zero.
      return tf_cond.cond(
          constant_op.constant(True), branch_fn, lambda: 0)

    def build_nested_cond():
      # [FUNCTION DOC]
      # HELPER: builds a cond whose true branch is the result of build_cond.
      return tf_cond.cond(
          constant_op.constant(True), build_cond, lambda: 0)

    # In v1 graph mode, pruning should make only "C" print.
    if not context.executing_eagerly():
      with self.cached_session():
        with self.captureWritesToStream(sys.stderr) as printed:
          self.assertEqual(self.evaluate(build_cond()), 10)
        self.assertEqual(["C"], filter_test_messages(printed.contents()))

        with self.captureWritesToStream(sys.stderr) as printed:
          self.assertEqual(self.evaluate(build_nested_cond()), 10)
        self.assertEqual(["C"], filter_test_messages(printed.contents()))

    # In defuns, all prints should execute in program order.
    # This doesn't work with legacy control flow.
    if control_flow_util.ENABLE_CONTROL_FLOW_V2:

      @eager_def_function.function
      def cond():
        # [FUNCTION DOC]
        # HELPER: compiled function that returns the built cond.
        return build_cond()

      with self.captureWritesToStream(sys.stderr) as printed:
        self.assertEqual(self.evaluate(cond()), 10)
      self.assertEqual(["A", "B", "C"],
                       filter_test_messages(printed.contents()))

      @eager_def_function.function
      def nested_cond():
        # [FUNCTION DOC]
        # HELPER: compiled function that returns the built nested cond.
        return build_nested_cond()

      with self.captureWritesToStream(sys.stderr) as printed:
        self.assertEqual(self.evaluate(nested_cond()), 10)
      self.assertEqual(["A", "B", "C"],
                       filter_test_messages(printed.contents()))

    # wrap_function should prune.
    def pruned_cond():
      # [FUNCTION DOC]
      # HELPER: returns the built cond; later rebound to a wrapped function that prunes.
      return build_cond()
    pruned_cond = wrap_function.wrap_function(pruned_cond, [])

    with self.captureWritesToStream(sys.stderr) as printed:
      self.assertEqual(self.evaluate(pruned_cond()), 10)
    self.assertEqual(["C"], filter_test_messages(printed.contents()))

    def pruned_nested_cond():
      # [FUNCTION DOC]
      # HELPER: returns the built nested cond; later rebound to a pruning wrapped function.
      return build_nested_cond()
    pruned_nested_cond = wrap_function.wrap_function(pruned_nested_cond, [])

    with self.captureWritesToStream(sys.stderr) as printed:
      self.assertEqual(self.evaluate(pruned_nested_cond()), 10)
    self.assertEqual(["C"], filter_test_messages(printed.contents()))


  @test_util.run_in_graph_and_eager_modes
  @test_util.disable_tfrt("b/179459136")
  def testWhileAutoControlDeps(self):
    # [FUNCTION DOC]
    # Test (graph and eager modes, TFRT disabled; only under control flow v2): while loop with prints A
    # (condition), B and C/D (body). Expects v1 graph mode to print only D per iteration, compiled
    # functions to print everything in program order, and wrap_function to prune. Checks via
    # captured stderr. Returns early under legacy control flow.
    # Legacy while_loop fails this test because it produces deprecation notices
    # in stderr.
    if not control_flow_util.ENABLE_CONTROL_FLOW_V2: return

    def cond(i, unused_x):
      # [FUNCTION DOC]
      # HELPER: loop condition; enqueues print A and returns whether the counter is below two.
      enqueue_print_op("A")
      return i < 2

    def body(i, x):
      # [FUNCTION DOC]
      # HELPER: loop body; enqueues B, then identity of x under control dependency on print C, then
      # returns incremented counter and x under control dependency on print D.
      enqueue_print_op("B")
      with ops.control_dependencies([enqueue_print_op("C")]):
        x = array_ops.identity(x)
      with ops.control_dependencies([enqueue_print_op("D")]):
        return i + 1, x

    def build_while():
      # [FUNCTION DOC]
      # HELPER: builds a while loop with two zero loop variables using the cond and body above.
      return while_loop_tf.while_loop(
          cond, body, [constant_op.constant(0),
                       constant_op.constant(0)])

    def build_nested_while():
      # [FUNCTION DOC]
      # HELPER: builds a cond whose true branch is the while loop and false branch returns zeros.
      return tf_cond.cond(
          constant_op.constant(True), build_while, lambda: [0, 0])

    # In v1 graph mode, pruning should make only "D" print.
    if not context.executing_eagerly():
      with self.cached_session():
        with self.captureWritesToStream(sys.stderr) as printed:
          self.assertEqual(self.evaluate(build_while()[0]), 2)
        self.assertEqual(["D", "D"], filter_test_messages(printed.contents()))

        with self.captureWritesToStream(sys.stderr) as printed:
          self.assertEqual(self.evaluate(build_nested_while()[0]), 2)
        self.assertEqual(["D", "D"], filter_test_messages(printed.contents()))

    # In defuns, all prints should execute in program order.
    @eager_def_function.function
    def while_loop():
      # [FUNCTION DOC]
      # HELPER: compiled function returning the first output of the built while loop.
      return build_while()[0]

    with self.captureWritesToStream(sys.stderr) as printed:
      self.assertEqual(self.evaluate(while_loop()), 2)
    self.assertEqual(["A", "B", "C", "D", "A", "B", "C", "D", "A"],
                     filter_test_messages(printed.contents()))

    @eager_def_function.function
    def nested_while_loop():
      # [FUNCTION DOC]
      # HELPER: compiled function returning the first output of the nested while loop.
      return build_nested_while()[0]

    with self.captureWritesToStream(sys.stderr) as printed:
      self.assertEqual(self.evaluate(nested_while_loop()), 2)
    self.assertEqual(["A", "B", "C", "D", "A", "B", "C", "D", "A"],
                     filter_test_messages(printed.contents()))

    # wrap_function should prune.
    def pruned_while():
      # [FUNCTION DOC]
      # HELPER: returns first output of built while loop; later rebound to a pruning wrapped function.
      return build_while()[0]
    pruned_while = wrap_function.wrap_function(pruned_while, [])

    with self.captureWritesToStream(sys.stderr) as printed:
      self.assertEqual(self.evaluate(pruned_while()), 2)
    self.assertEqual(["D", "D"], filter_test_messages(printed.contents()))

    def pruned_nested_while():
      # [FUNCTION DOC]
      # HELPER: returns first output of the nested while loop; later rebound to a pruning wrapped function.
      return build_nested_while()[0]
    pruned_nested_while = wrap_function.wrap_function(pruned_nested_while, [])

    with self.captureWritesToStream(sys.stderr) as printed:
      self.assertEqual(self.evaluate(pruned_nested_while()), 2)
    self.assertEqual(["D", "D"], filter_test_messages(printed.contents()))

  # Microbenchmark: 256,000 iterations/s.
  def testWhile_1(self):
    # [FUNCTION DOC]
    # Test: while loop incrementing from zero to 10000 with 20 parallel iterations. Expects 10000.
    # (A comment notes a microbenchmark speed.)
    with self.cached_session():
      n = constant_op.constant(0)
      c = lambda x: math_ops.less(x, 10000)
      b = lambda x: math_ops.add(x, 1)
      r = while_loop_tf.while_loop(c, b, [n], parallel_iterations=20)
      self.assertEqual(10000, self.evaluate(r))

  @test_util.run_v1_only("b/120545219")
  def testWhileExternalControlDependencies(self):
    # [FUNCTION DOC]
    # Test (v1 only): loop body has a control dependency on an external increment of a variable.
    # Expects the result 2 and the variable incremented once (1.0), meaning external dependencies run
    # only once.
    with self.cached_session():
      v = variables.Variable(0.0)
      self.evaluate(v.initializer)
      increment = v.assign_add(1.0).read_value()

      def body_fn(i):
        # [FUNCTION DOC]
        # HELPER: returns i plus one under a control dependency on the outer increment op.
        with ops.control_dependencies([increment]):
          return i + 1

      result = while_loop_tf.while_loop(
          cond=lambda i: i < 2, body=body_fn, loop_vars=[1])
      self.assertAllEqual(result, 2)
      self.assertAllEqual(v.read_value(), 1.0)

  @test_util.run_v1_only("b/120545219")
  def testWhileExternalControlDependenciesNoInput(self):
    # [FUNCTION DOC]
    # Test (v1 only): loop body that ignores its input but depends on an external increment.
    # Expects the variable to be 1.0 after running the loop.
    with self.cached_session():
      v = variables.Variable(0.0)
      self.evaluate(v.initializer)
      # TODO(apassos): figure out why the reading is necessary here.
      increment = v.assign_add(1.0).read_value()

      def body_fn(unused_i):
        # [FUNCTION DOC]
        # HELPER: ignores its argument and returns a constant five under a control dependency on the
        # outer increment op.
        with ops.control_dependencies([increment]):
          return constant_op.constant(5, name="five")

      result = while_loop_tf.while_loop(
          cond=lambda i: i < 5, body=body_fn, loop_vars=[0])
      self.evaluate(result)
      self.assertAllEqual(self.evaluate(v), 1.0)

  @test_util.disable_control_flow_v2("b/113324949 (RefVariable)")
  @test_util.run_v1_only("b/120545219")
  def testWhileWithRefs_1(self):
    # [FUNCTION DOC]
    # Test (legacy control flow, v1 only): while loop carrying a reference variable. Checks reference
    # dtype is kept through the loop; the result is the counter 100 and the variable value 0.
    with self.cached_session() as sess:
      x = variable_v1.VariableV1(0)._ref()  # pylint: disable=protected-access
      i = constant_op.constant(0)
      c = lambda i, x: math_ops.less(i, 100)

      self.assertEqual(x.dtype, dtypes.int32_ref)

      def b(i, x):
        # [FUNCTION DOC]
        # HELPER: loop body; asserts that x has a reference int32 type, returns i plus one and ref identity of x.
        self.assertEqual(x.dtype, dtypes.int32_ref)
        return (i + 1, gen_array_ops.ref_identity(x))

      r = while_loop_tf.while_loop(c, b, [i, x], parallel_iterations=5)

      self.evaluate(variables.global_variables_initializer())

      self.assertEqual(r[0].dtype, dtypes.int32)
      self.assertEqual(r[1].dtype, dtypes.int32_ref)

      value_i, value_x = self.evaluate(r)

    self.assertEqual(100, value_i)
    self.assertEqual(0, value_x)

  def testWhile_2(self):
    # [FUNCTION DOC]
    # Test: runs the summing loop helper from zero; expects 45.
    with self.cached_session():
      s = constant_op.constant(0)
      r = isum(s)
      self.assertAllEqual(45, self.evaluate(r))

  def testWhileWithMaximumIterations(self):
    # [FUNCTION DOC]
    # Test: summing loop helper over a vector with maximum iterations three. Expects each element plus 3.
    with self.cached_session():
      s = constant_op.constant([1, 2, 3, 4, 5])
      r = isum(s, maximum_iterations=3)
      self.assertAllEqual([1 + 3, 2 + 3, 3 + 3, 4 + 3, 5 + 3], self.evaluate(r))

  @test_util.run_v1_only("b/120545219")
  def testWhileWithMaximumIterationsAndSingleArgument(self):
    # [FUNCTION DOC]
    # Test (v1 only): single-variable loop with maximum iterations one. Expects 1.
    with self.cached_session():
      r = while_loop_tf.while_loop(
          lambda i: i < 3, lambda i: i + 1, [0], maximum_iterations=1)
      self.assertEqual(1, self.evaluate(r))

  @test_util.run_v1_only("b/120545219")
  def testXLAGradInLoop(self):
    # [FUNCTION DOC]
    # Test (v1 only): inside an XLA control flow context, builds a loop on dynamic-shape inputs and
    # takes a gradient. Expects the Shape op to feed BroadcastGradientArgs directly rather than being
    # pushed on a stack. The context is exited at the end.
    # We have an optimization that moves certain reduction ops, this test makes
    # sure we don't do that for XLA ops.

    # Use dynamic inputs, which triggers the creation of "BroadcastGradientArgs"
    # and "Shape" op.
    input1 = array_ops.placeholder(dtype=dtypes.float32, shape=[None, None])
    input2 = array_ops.placeholder(dtype=dtypes.float32, shape=[None, None])
    def cond(i1, i2):
      # [FUNCTION DOC]
      # HELPER: loop condition that always returns false.
      return False

    def body(i1, i2):
      # [FUNCTION DOC]
      # HELPER: loop body returning the sum of the two inputs twice.
      return math_ops.add(i1, i2), math_ops.add(i1, i2)

    xla_context = control_flow_ops.XLAControlFlowContext()
    xla_context.Enter()

    out1, _ = while_loop_tf.while_loop(
        cond, body, (input1, input2), maximum_iterations=2)
    g = gradients_impl.gradients(out1, [input1])

    for op in out1.graph.get_operations():
      # Test that the "Shape" is directly passed to BroadcastGradientArgs
      # instead of being pushed to the stack.
      if op.type == "BroadcastGradientArgs":
        self.assertEqual(op.inputs[0].op.type, "Shape")
        self.assertEqual(op.inputs[1].op.type, "Shape")
    xla_context.Exit()


  @test_util.disable_control_flow_v2("b/115776323 (max_iters)")
  @test_util.run_v1_only("b/120545219")
  def testSingleNestedMaximumIterationsWhileLoopGradientInXLAContext(self):
    # [FUNCTION DOC]
    # Test (legacy control flow, v1 only): in an XLA context, an outer loop whose body creates an inner
    # loop bounded by the outer counter and takes a gradient of it. Expects the whole thing to
    # execute and the outer loop to return 3.
    v = constant_op.constant(1.0)

    def training_loop_with_gradient(i):
      # [FUNCTION DOC]
      # HELPER: builds an inner while loop with maximum iterations equal to the argument, takes its
      # gradient with respect to the captured constant and returns the argument plus one under a control
      # dependency on the gradient.
      out = while_loop_tf.while_loop(
          lambda i_, _: i_ < 3,
          lambda i_, j: [i_ + 1, j * v], [0, 1.0],
          maximum_iterations=i)
      g = gradients_impl.gradients(out, v)
      with ops.control_dependencies(g):
        return i + 1

    xla_context = control_flow_ops.XLAControlFlowContext()
    xla_context.Enter()
    # Create training loop, ensure we can call gradient() of
    # while_loop inside the training loop.
    loop = while_loop_tf.while_loop(
        lambda i: i < 3, training_loop_with_gradient, [0])
    xla_context.Exit()

    loop_execute = array_ops.identity(loop)  # Because loop is not fetchable.

    # Should execute without issue.
    self.assertEqual(3, self.evaluate(loop_execute))

  @test_util.run_v1_only("b/120545219")
  def testInvalidMaximumIterationsWhileLoopGradientInXLAContext(self):
    # [FUNCTION DOC]
    # Test (v1 only; skipped under v2): in an XLA context, taking gradients of a loop without a
    # maximum-iterations argument, or with one that is not statically known, must raise value errors
    # describing why the gradient accumulator cannot be created. Outside XLA it works.
    if control_flow_util.ENABLE_CONTROL_FLOW_V2:
      self.skipTest("WhileV2 does lazy evaluation of maximum_iterations")
    v = constant_op.constant(1.0)

    def inner_body(i, x):
      # [FUNCTION DOC]
      # HELPER: runs an inner while loop with maximum iterations equal to the outer counter.
      out = while_loop_tf.while_loop(
          lambda i, _: i < 3,
          lambda i, j: [i + 1, j * v], [0, x],
          maximum_iterations=i)
      return out

    def create_while_loop(maximum_iterations=None):
      # [FUNCTION DOC]
      # HELPER: builds an outer while loop with the given optional maximum iterations (default none).
      return while_loop_tf.while_loop(
          lambda i, _: i < 3,
          inner_body, [0, 1.0],
          maximum_iterations=maximum_iterations)

    loop_no_xla = create_while_loop(maximum_iterations=5)
    # maximum_iterations is fine outside of an XLA scope
    gs = gradients_impl.gradients(loop_no_xla, v)
    self.evaluate(gs)  # This should execute without error.

    xla_context = control_flow_ops.XLAControlFlowContext()
    xla_context.Enter()
    loop_no_maxiter = create_while_loop()
    loop_with_maxiter = create_while_loop(maximum_iterations=2)
    xla_context.Exit()

    with self.assertRaisesRegex(
        ValueError,
        r"Cannot create a gradient accumulator for tensor '.+' inside "
        r"XLA while_loop because maximum_iterations was not passed to "
        r"the tf.while_loop call \('.+'\)."):
      _ = gradients_impl.gradients(loop_no_maxiter, v)

    with self.assertRaisesRegex(
        ValueError,
        r"Cannot create a gradient accumulator for tensor '.+' inside XLA "
        r"while_loop. maximum_iterations tensor '.+' for while_loop context "
        r"'.+' must be statically known \(e.g. a constant value or known "
        r"shape dimension\), or be defined at or outside the while loop "
        r"context '.*' \(currently defined in '.*'\)"):
      _ = gradients_impl.gradients(loop_with_maxiter, v)

  @test_util.run_v1_only("b/120545219")
  def testInvalidMaximumIterationsFromSiblingContextWhileLoopInXLAContext(self):
    # [FUNCTION DOC]
    # Test (v1 only): maximum iterations tensor created in a sibling cond context. Under v2 expects a
    # value error about being from a different graph at construction; under v1 builds fine and the
    # gradient fails because the tensor is defined in a cond context.
    v = constant_op.constant(1.0)

    def create_while_loop():
      # [FUNCTION DOC]
      # HELPER: creates a placeholder inside both branches of a cond to obtain a maximum iterations tensor
      # from another context, then builds a loop using it. Returns the loop.
      max_iter_holder = []

      def create_mi():
        # [FUNCTION DOC]
        # HELPER: appends a new int32 scalar placeholder to the shared list and returns 1.0.
        max_iter_holder.append(array_ops.placeholder(dtypes.int32, shape=()))
        return 1.0

      _ = tf_cond.cond(
          constant_op.constant(True), create_mi, create_mi)

      return while_loop_tf.while_loop(
          lambda i, _: i < 3,
          lambda i, x: (i + 1, v * x), (0, 1.0),
          maximum_iterations=max_iter_holder[0])

    if control_flow_util.ENABLE_CONTROL_FLOW_V2:
      xla_context = control_flow_ops.XLAControlFlowContext()
      xla_context.Enter()
      with self.assertRaisesRegex(ValueError, r"must be from the same graph.*"):
        loop = create_while_loop()
      xla_context.Exit()
    else:
      xla_context = control_flow_ops.XLAControlFlowContext()
      xla_context.Enter()
      loop = create_while_loop()
      xla_context.Exit()
      with self.assertRaisesRegex(
          ValueError,
          r"Cannot create a gradient accumulator for tensor '.+' inside XLA "
          r"while_loop. maximum_iterations tensor '.*Placeholder:0' for "
          r"while_loop context '.+' must be statically known \(e.g. a constant "
          r"value or known shape dimension\), or be defined at or outside the "
          r"while loop context '' \(currently defined in 'cond/.+'\)"):
        _ = gradients_impl.gradients(loop, v)

  @test_util.run_v1_only("b/120545219")
  def testNestedWhileLoopWithMaxItersFromOuterContextInXLAContext(self):
    # [FUNCTION DOC]
    # Test (v1 only; skipped if a GPU is available): nested while loops in an XLA context where the
    # inner loop's maximum iterations comes from an outer context placeholder.
    if test_util.is_gpu_available():
      self.skipTest("b/128646372, b/128645947 fails in opensource build")

    v = constant_op.constant(1.0)

    p = array_ops.placeholder(dtype=dtypes.int32)

    def mid_body_builder(iterations):
      # [FUNCTION DOC]
      # HELPER: given an iteration limit, returns the mid-level loop body that runs an inner while loop
      # bounded by that limit.

      def mid_body(i, x):
        # [FUNCTION DOC]
        # HELPER: middle loop body; runs an inner while loop (always-true condition, bounded by the captured
        # iterations value) multiplying by the captured constant, and returns the incremented counter plus
        # the gradient of x plus the inner result with respect to the constant.
        r = while_loop_tf.while_loop(
            lambda *_: True,
            lambda i, x: (i + 1, v * x), (0, x),
            maximum_iterations=iterations,
            name="inner")
        return (i + 1, gradients_impl.gradients(x + r[1], v)[0])

      return mid_body

    def outer_body(i, x):
      # [FUNCTION DOC]
      # HELPER: outer loop body; computes the iteration count as the size of the placeholder, runs a mid
      # loop bounded by it, and returns the incremented counter and x plus the mid loop result.
      iterations = array_ops.size(p, name="iterations")
      return (i + 1, x + while_loop_tf.while_loop(
          lambda *_: True,
          mid_body_builder(iterations), (0, x),
          maximum_iterations=iterations,
          name="mid")[1])

    def create_while_loop():
      # [FUNCTION DOC]
      # HELPER: under the CPU device scope builds the outer while loop (maximum iterations five) and
      # returns an identity of its second output.
      with ops.device("/cpu:0"):
        r = while_loop_tf.while_loop(
            lambda *_: True,
            outer_body, (0, 1.0),
            maximum_iterations=5,
            name="outer")
        return array_ops.identity(r[1])

    xla_context = control_flow_ops.XLAControlFlowContext()
    xla_context.Enter()
    final_with_xla_context = create_while_loop()
    xla_context.Exit()

    final_without_xla_context = create_while_loop()

    with self.session(use_gpu=False) as sess:
      opts = config_pb2.RunOptions(trace_level=config_pb2.RunOptions.FULL_TRACE)
      run_metadata_without_xla_context = config_pb2.RunMetadata()
      run_metadata = config_pb2.RunMetadata()

      final_value_without_xla_context = sess.run(
          final_without_xla_context,
          feed_dict={p: [0, 0, 0]},
          options=opts,
          run_metadata=run_metadata_without_xla_context)

      final_value_with_xla_context = sess.run(
          final_with_xla_context,
          feed_dict={p: [0, 0, 0]},
          options=opts,
          run_metadata=run_metadata)

      if control_flow_util.ENABLE_CONTROL_FLOW_V2:
        # With while_v2 on xla, run_metadata only contains the unlowered While
        # op so node_stats does not have statistics for the pushes. So as a
        # loose check we check the pushes in the lowered version.
        for dev in run_metadata_without_xla_context.step_stats.dev_stats:
          if "/device:CPU" in dev.device:
            node_stats = dev.node_stats
        stack_push_count = len([
            x for x in node_stats
            if re.match(r".*TensorListPushBack_?\d*", x.node_name)
        ])
      else:
        for dev in run_metadata.step_stats.dev_stats:
          if "/device:CPU" in dev.device:
            node_stats = dev.node_stats
        stack_push_op = "StackPushV2"
        stack_push_count = len(
            [x for x in node_stats if x.node_name.endswith("StackPushV2")])
      # Pushes to the stack = product of maximum_iterations values;
      # the last two "3"s comes from size(p), when p == [0, 0, 0].
      self.assertEqual(stack_push_count, 5 * 3 * 3, str(node_stats))

      self.assertAllClose(final_value_with_xla_context,
                          final_value_without_xla_context)

  # Have more than 10 parallel iterations and hence exercise k-bound
  # most of the time.
  @test_util.run_deprecated_v1
  def testWhile_3(self):
    # [FUNCTION DOC]
    # Test (deprecated v1 mode): a loop with four loop variables and many parallel iterations (default),
    # exercising the k-bound. Runs 100 iterations accumulating counters; expects 10100.
    with self.cached_session():

      def compute(i, m, c, o):
        # [FUNCTION DOC]
        # HELPER: loop body; increments m and c, adds both into o, increments i and returns all four.
        m, c = [math_ops.add(m, 1), math_ops.add(c, 1)]
        o = math_ops.add(o, m)
        o = math_ops.add(o, c)
        i = math_ops.add(i, 1)
        return [i, m, c, o]

      i = ops.convert_to_tensor(0)
      m = ops.convert_to_tensor(0)
      c = ops.convert_to_tensor(0)
      o = ops.convert_to_tensor(0)
      d = ops.convert_to_tensor(100)
      r = while_loop_tf.while_loop(
          lambda i, m, c, o: math_ops.less(i, d), compute, [i, m, c, o])
      result = r[3]
    self.assertAllEqual(10100, result)

  @test_util.run_deprecated_v1
  def testWhile_4(self):
    # [FUNCTION DOC]
    # Test (deprecated v1 mode): loop body gathers the i-th element of a vector twice and accumulates
    # both into o. Expects the doubled sum, 42.
    with self.cached_session():

      def compute(i, m, c, o):
        # [FUNCTION DOC]
        # HELPER: loop body; gathers the i-th element of the captured vector twice, adds both into o,
        # increments i and returns the four values.
        m, c = [array_ops.gather(x, i), array_ops.gather(x, i)]
        o = math_ops.add(o, m)
        o = math_ops.add(o, c)
        i = math_ops.add(i, 1)
        return [i, m, c, o]

      i = ops.convert_to_tensor(0)
      m = ops.convert_to_tensor(0)
      c = ops.convert_to_tensor(0)
      o = ops.convert_to_tensor(0)
      x = ops.convert_to_tensor([1, 2, 3, 4, 5, 6])
      s = array_ops.size(x)
      r = while_loop_tf.while_loop(
          lambda i, m, c, o: math_ops.less(i, s), compute, [i, m, c, o])
      result = r[3]
    self.assertAllEqual(42, result)

  @test_util.run_v1_only("b/120545219")
  def testWhile_5(self):
    # [FUNCTION DOC]
    # Test (v1 only): loop with unknown-shape variables that concatenates strided slices of a vector.
    # Expects the output [0, 1, 2, 3, 4, 5, 6].
    with self.cached_session():

      def compute(i, c, o):
        # [FUNCTION DOC]
        # HELPER: loop body; takes a one element slice at i from the captured vector, appends it to o,
        # increments i and returns the three values.
        c = array_ops.strided_slice(x, array_ops.expand_dims(i, 0),
                                    [1] + array_ops.expand_dims(i, 0))
        o = array_ops.concat([o, c], 0)
        i = math_ops.add(i, 1)
        return [i, c, o]

      i = ops.convert_to_tensor(0)
      c = ops.convert_to_tensor([0])
      o = ops.convert_to_tensor([0])
      x = ops.convert_to_tensor([1, 2, 3, 4, 5, 6])
      s = array_ops.size(x)
      r = while_loop_tf.while_loop(
          lambda i, c, o: math_ops.less(i, s),
          compute,
          [i, c, o],
          [
              i.get_shape(),
              tensor_shape.unknown_shape(),
              tensor_shape.unknown_shape(),
          ],
      )
      result = r[2]
    self.assertAllEqual(np.array([0, 1, 2, 3, 4, 5, 6]), result)

  @test_util.run_gpu_only
  @test_util.run_deprecated_v1
  def testWhile_Device(self):
    # [FUNCTION DOC]
    # Test (GPU only, deprecated v1 mode): while loop defined under a CPU device scope with a body
    # defined outside it. Expects the result device to be CPU and a single partition graph.

    # Body function defined outside of device scope
    def body(x):
      # [FUNCTION DOC]
      # HELPER: loop body defined outside the device scope; returns the exponential of x.
      return math_ops.exp(x)

    with ops.device("CPU:0"):
      r = while_loop_tf.while_loop(
          lambda x: x < 10, body, [constant_op.constant(-10.0)])
      self.assertIn("cpu", r.device.lower())

    with session.Session() as sess:
      options = config_pb2.RunOptions(output_partition_graphs=True)
      run_metadata = config_pb2.RunMetadata()
      sess.run(r, options=options, run_metadata=run_metadata)
      # We expect that everything runs on CPU, even if GPU is available.
      self.assertEqual(len(run_metadata.partition_graphs), 1)

  @test_util.disable_control_flow_v2("b/116338794 (buffer_reuse)")
  @test_util.run_v1_only("b/120545219")
  def testBufferForwarding(self):
    # [FUNCTION DOC]
    # Test (legacy control flow, v1 only): runs a 1000 iteration loop on CPU with full tracing.
    # Expects the result 1000 and that the number of unique output buffer allocations is below 756
    # (far less than the roughly 1005 seen before buffer forwarding).
    run_options = config_pb2.RunOptions(
        trace_level=config_pb2.RunOptions.FULL_TRACE)
    run_metadata = config_pb2.RunMetadata()

    with self.cached_session() as sess:
      with ops.device("/cpu:0"):
        c = constant_op.constant(2)
        i0 = constant_op.constant(0)
        r = while_loop_tf.while_loop(
            lambda i: i < 1000, lambda i: math_ops.square(c) + i, [i0])
      r_val = sess.run(r, options=run_options, run_metadata=run_metadata)
      self.assertEqual(1000, r_val)
      self.assertTrue(run_metadata.HasField("step_stats"))
      unique_allocs = set()
      for node_stat in run_metadata.step_stats.dev_stats[0].node_stats:
        for output in node_stat.output:
          unique_allocs.add(
              output.tensor_description.allocation_description.ptr)
      # Prior to cl/147536680, the number of unique allocations was about 1005.
      self.assertLess(len(unique_allocs), 756)

  def _testWhile_Gpu_1(self, use_gpu):
    # [FUNCTION DOC]
    # HELPER: shared body for testWhile_Gpu_1; takes a use-GPU flag, loops a float from 1 to 10 and
    # checks the result is close to 10.
    with self.cached_session(use_gpu=use_gpu):
      n = constant_op.constant(1.0)
      c = lambda x: math_ops.less(x, 10.0)
      b = lambda x: math_ops.add(x, 1.0)
      r = while_loop_tf.while_loop(c, b, [n])
      self.assertAllClose(10.0, self.evaluate(r))

  def testWhile_Gpu_1(self):
    # [FUNCTION DOC]
    # Test: runs the shared helper without and then with the GPU.
    self._testWhile_Gpu_1(use_gpu=False)
    self._testWhile_Gpu_1(use_gpu=True)

  def _testWhile_Gpu_2(self, use_gpu):
    # [FUNCTION DOC]
    # HELPER: shared body for testWhile_Gpu_2; like the previous helper but the body is pinned to
    # the CPU. Checks the result is close to 10.
    with self.cached_session(use_gpu=use_gpu):
      n = constant_op.constant(1.0)
      c = lambda x: math_ops.less(x, 10.0)

      def b(x):
        # [FUNCTION DOC]
        # HELPER: loop body; adds one under a CPU device scope.
        with ops.device("/cpu:0"):
          return math_ops.add(x, 1.0)

      r = while_loop_tf.while_loop(c, b, [n])
      self.assertAllClose(10.0, self.evaluate(r))

  def testWhile_Gpu_2(self):
    # [FUNCTION DOC]
    # Test: runs the shared helper without and then with the GPU.
    self._testWhile_Gpu_2(use_gpu=False)
    self._testWhile_Gpu_2(use_gpu=True)

  def testWhileShape(self):
    # [FUNCTION DOC]
    # Test: loop tiles a 2x2 matrix by 2x2 twice with an unknown shape invariant. The result times a
    # 8x8 ones matrix is expected to be all ones.
    with self.cached_session():
      i = constant_op.constant(0)
      m = array_ops.ones([2, 2])
      c = lambda i, j: math_ops.less(i, 2)

      def _b(i, j):
        # [FUNCTION DOC]
        # HELPER: loop body; increments i and tiles j by two in both dimensions.
        new_i = math_ops.add(i, 1)
        new_j = array_ops.tile(j, [2, 2])
        return [new_i, new_j]

      r = while_loop_tf.while_loop(
          c, _b, [i, m],
          [i.get_shape(), tensor_shape.unknown_shape()])
      r = r[1] * array_ops.ones([8, 8])
      self.assertAllEqual(np.ones((8, 8)), self.evaluate(r))

  @test_util.disable_control_flow_v2("b/131265085")
  @test_util.run_v1_only("b/131265085")
  def testWhileBadShape(self):
    # [FUNCTION DOC]
    # Test (legacy control flow, v1 only): shape invariant of [5] given for a vector of 2 elements.
    # Expects a value error saying the shapes are not compatible.
    x = constant_op.constant([2.0, 4.0], name="values")
    i = constant_op.constant(0)
    c = lambda i, _: math_ops.less(i, 10)
    b = lambda i, x: [i + 1, x + 1]
    with self.assertRaisesRegex(ValueError, "is not compatible with"):
      # Shape of x is [2], but we specify a shape of [5].
      while_loop_tf.while_loop(
          c, b, [i, x], [i.shape, tensor_shape.TensorShape([5])])

  @test_util.run_in_graph_and_eager_modes
  def testWhileBadBodyReturn(self):
    # [FUNCTION DOC]
    # Test (graph and eager modes): the body takes N variables and returns N plus one values.
    # Expects a value error about different nested structure.
    x = constant_op.constant([2.0, 4.0], name="values")
    i = constant_op.constant(0)
    c = lambda i, *x: math_ops.less(i, 10)

    # body accepts N values and returns N+1 values.
    b = lambda i, *x: (i, i) + x

    with self.assertRaisesRegex(
        ValueError, "The two structures don't have the same nested structure."):
      while_loop_tf.while_loop(c, b, [i, x])

  @test_util.run_deprecated_v1
  def testWhileWithNonTensorInput_Scalar(self):
    # [FUNCTION DOC]
    # Test (deprecated v1 mode): loop variable is a plain Python integer rather than a tensor,
    # loops up to 10000 with 20 parallel iterations. Expects 10000.
    with self.cached_session():
      n = 0
      c = lambda x: x < 10000
      b = lambda x: x + 1
      r = while_loop_tf.while_loop(c, b, [n], parallel_iterations=20)
      self.assertEqual(10000, self.evaluate(r))

  def testWhileWithNonTensorInput_Vector(self):
    # [FUNCTION DOC]
    # Test: loop variable is a NumPy array (a plain list would not work); body stacks the incremented
    # first element. Expects [10000].
    with self.cached_session():
      n = np.array([0])  # Note, [0] would not work here; that is a list
      c = lambda x: x[0] < 10000
      b = lambda x: array_ops_stack.stack([x[0] + 1])
      r = while_loop_tf.while_loop(c, b, [n], parallel_iterations=20)
      self.assertEqual([10000], self.evaluate(r))

  def testWhileShapeInference(self):
    # [FUNCTION DOC]
    # Test: body concatenates the matrix with itself each iteration with a shape invariant of unknown
    # rows and two columns. The result shape must be compatible with 8 by 2.
    with self.cached_session():
      i = constant_op.constant(0)
      m = array_ops.ones([2, 2])
      c = lambda i, j: math_ops.less(i, 2)

      def b(i, j):
        # [FUNCTION DOC]
        # HELPER: loop body; increments i and concatenates j with itself along the first axis.
        new_i = math_ops.add(i, 1)
        new_j = array_ops.concat([j, j], 0)
        return [new_i, new_j]

      r = while_loop_tf.while_loop(
          c, b, [i, m],
          [i.get_shape(), tensor_shape.TensorShape([None, 2])])
      self.assertTrue(r[1].shape.is_compatible_with([8, 2]))

  @test_util.run_v1_only("b/120545219")
  def testWhileShapeInferenceBadShape(self):
    # [FUNCTION DOC]
    # Test (v1 only): body grows the first dimension without shape invariants. Expects a value error
    # mentioning shapes (2, 2) and (4, 2) after one iteration and advising shape invariants.
    with self.cached_session():
      i = constant_op.constant(0)
      m = array_ops.ones([2, 2])
      c = lambda i, j: math_ops.less(i, 2)
      b = lambda i, j: [i + 1, array_ops.concat([j, j], 0)]
      with self.assertRaisesRegex(
          ValueError,
          r".*\(2, 2\).*\(4, 2\) after one iteration\. To allow the shape to "
          r"vary across iterations, use the `shape_invariants` argument of "
          r"tf.while_loop to specify a less-specific shape\."):
        while_loop_tf.while_loop(c, b, [i, m])

  def testWhileShapeInferenceSparseTensor(self):
    # [FUNCTION DOC]
    # Test: shape inference of while loops over sparse tensors for three bodies (modify values, add
    # values, change rank) under default and explicit shape invariants (including a null-rank
    # invariant allowed for backward compatibility). Checks the static shapes of indices, values and
    # dense shape through a helper.
    values = constant_op.constant([2.0, 4.0], name="values")
    indices = constant_op.constant([[0], [3]],
                                   dtype=dtypes.int64,
                                   name="indices")
    shape = constant_op.constant([10], dtype=dtypes.int64, name="dense_shape")
    i = constant_op.constant(0)
    x = sparse_tensor.SparseTensor(indices, values, dense_shape=shape)

    def c(i, _):
      # [FUNCTION DOC]
      # HELPER: loop condition; true while the counter is below ten.
      return i < 10

    def b1(i, x):  # modifies values.  (shape of components is not changed.)
      # [FUNCTION DOC]
      # HELPER: body that increments i and doubles the values of the sparse tensor (component shapes
      # unchanged).
      return [
          i + 1,
          sparse_tensor.SparseTensor(x.indices, x.values * 2.0, x.dense_shape)
      ]

    def b2(i, x):  # adds new values.  (shape of components is changed.)
      # [FUNCTION DOC]
      # HELPER: body that increments i and adds a new single entry to the sparse tensor, changing the
      # shape of its components.
      return [
          i + 1,
          sparse_ops.sparse_add(
              x,
              sparse_tensor.SparseTensor(
                  indices=math_ops.cast(
                      array_ops.fill([1, 1], i), dtypes.int64),
                  values=array_ops.fill([1], 1.0),
                  dense_shape=x.dense_shape))
      ]

    def b3(i, x):  # modifies rank.  (shape of all components is changed.)
      # [FUNCTION DOC]
      # HELPER: body that increments i and increases the sparse tensor rank by appending an index column
      # and dimension.
      return [
          i + 1,
          sparse_tensor.SparseTensor(
              array_ops.concat([x.indices, [[i], [i]]], axis=1), x.values * 2.0,
              array_ops.concat([x.dense_shape, [10]], axis=0))
      ]

    def check_shapes(r, indices, values, dense_shape):
      # [FUNCTION DOC]
      # HELPER: asserts that the given sparse tensor's indices, values and dense shape have static
      # shapes compatible with the three expected shapes passed in.
      self.assertTrue(r.indices.shape.is_compatible_with(indices))
      self.assertTrue(r.values.shape.is_compatible_with(values))
      self.assertTrue(r.dense_shape.shape.is_compatible_with(dense_shape))

    # Default shape invariant; b1 only modifies values.
    _, r = while_loop_tf.while_loop(c, b1, [i, x])
    check_shapes(r, indices=[None, 1], values=[None], dense_shape=[1])

    # Default shape invariant; b2 adds new values
    _, r = while_loop_tf.while_loop(c, b2, [i, x])
    check_shapes(r, indices=[None, 1], values=[None], dense_shape=[1])

    # Explicit shape invariant, allowing any rank; b1 only modifies values.
    _, r = while_loop_tf.while_loop(
        c, b1, [i, x],
        [i.get_shape(), tensor_shape.TensorShape([None])])
    check_shapes(r, indices=[None, None], values=[None], dense_shape=[None])

    # Explicit shape invariant, allowing any rank; b3 modifies rank.
    _, r = while_loop_tf.while_loop(
        c, b3, [i, x],
        [i.get_shape(), tensor_shape.TensorShape([None])])
    check_shapes(r, indices=[None, None], values=[None], dense_shape=[None])

    # Shape invariant with ndims=None.  Technically, this isn't supported
    # according to the docs, but we support it for backwards compatibility.
    _, r = while_loop_tf.while_loop(
        c, b1, [i, x],
        [i.get_shape(), tensor_shape.TensorShape(None)])
    check_shapes(r, indices=[None, None], values=[None], dense_shape=[None])
    _, r = while_loop_tf.while_loop(
        c, b3, [i, x],
        [i.get_shape(), tensor_shape.TensorShape(None)])
    check_shapes(r, indices=[None, None], values=[None], dense_shape=[None])

  @test_util.disable_control_flow_v2("b/131265085")
  @test_util.run_v1_only("b/131265085")
  def testWhileBadShapeSparseTensor(self):
    # [FUNCTION DOC]
    # Test (legacy control flow, v1 only): with a shape invariant of [5] for a sparse tensor expects a
    # value error (not compatible); with default invariants a body that changes the rank must raise a
    # value error.
    values = constant_op.constant([2.0, 4.0], name="values")
    indices = constant_op.constant([[0], [3]],
                                   dtype=dtypes.int64,
                                   name="indices")
    shape = constant_op.constant([10], dtype=dtypes.int64, name="dense_shape")
    i = constant_op.constant(0)
    x = sparse_tensor.SparseTensor(indices, values, dense_shape=shape)
    c = lambda i, _: i < 10
    b1 = lambda i, x: [i+1, x]
    def b2(i, x):  # modifies rank.  (shape of all components is changed.)
      # [FUNCTION DOC]
      # HELPER: body that increments i and increases the sparse tensor's rank (changes the shape of all
      # components).
      return [
          i + 1,
          sparse_tensor.SparseTensor(
              array_ops.concat([x.indices, [[i], [i]]], axis=1), x.values * 2.0,
              array_ops.concat([x.dense_shape, [10]], axis=0))
      ]

    # Explicit shape invariant, with a specific (incompatible) rank.
    with self.assertRaisesRegex(ValueError, "is not compatible with"):
      while_loop_tf.while_loop(
          c, b1, [i, x],
          [i.get_shape(), tensor_shape.TensorShape([5])])

    # Default shape invariant, but b2 modifies rank (which is not allowed).
    with self.assertRaises(ValueError):
      while_loop_tf.while_loop(c, b2, [i, x])

  def testWhileShapeInferenceIndexedSlices(self):
    # [FUNCTION DOC]
    # Test: loop over indexed slices whose values are doubled. With default invariants the dense shape
    # has two entries and values have shape 2 by 2; with a shape invariant of unknown rows the values
    # shape is compatible with unknown by 2.
    with self.cached_session():
      values = constant_op.constant([[2.0, 4.0], [3.0, 5.0]], name="values")
      indices = constant_op.constant([0, 3], name="indices")
      shape = constant_op.constant([10, 2], name="dense_shape")
      i = constant_op.constant(0)
      x = indexed_slices.IndexedSlices(values, indices, dense_shape=shape)

      def c(i, _):
        # [FUNCTION DOC]
        # HELPER: loop condition; true while the counter is below ten.
        return i < 10

      def b(i, x):
        # [FUNCTION DOC]
        # HELPER: body that increments i and doubles the values of the indexed slices.
        return [
            i + 1,
            indexed_slices.IndexedSlices(x.values * 2.0, x.indices,
                                         x.dense_shape)
        ]

      _, r = while_loop_tf.while_loop(c, b, [i, x])
      self.assertEqual(r.dense_shape.get_shape()[0], 2)
      self.assertEqual(r.values.get_shape(), tensor_shape.TensorShape([2, 2]))

      _, r = while_loop_tf.while_loop(
          c, b, [i, x],
          [i.get_shape(), tensor_shape.TensorShape([None, 2])])
      self.assertEqual(r.dense_shape.get_shape()[0], 2)
      self.assertTrue(r.values.get_shape().is_compatible_with([None, 2]))

  @test_util.disable_control_flow_v2("b/131265085")
  @test_util.run_v1_only("b/131265085")
  def testWhileBadShapeIndexedSlices(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: despite the name, the test builds a sparse tensor, not an indexed-slices value, and
    # the loop condition returns the constant 10 rather than a boolean; it only checks the shape
    # invariant error, which is raised during loop construction.
    # Test (legacy control flow, v1 only): shape invariant of [5] for a tensor with a different shape
    # must raise a value error that the shapes are not compatible.
    values = constant_op.constant([2.0, 4.0], name="values")
    indices = constant_op.constant([[0], [3]],
                                   dtype=dtypes.int64,
                                   name="indices")
    shape = constant_op.constant([10], dtype=dtypes.int64, name="dense_shape")
    i = constant_op.constant(0)
    x = sparse_tensor.SparseTensor(indices, values, dense_shape=shape)
    c = lambda i, _: 10
    b = lambda i, x: [i+1, x]

    # Explicit shape invariant, with a specific (incompatible) rank.
    with self.assertRaisesRegex(ValueError, "is not compatible with"):
      while_loop_tf.while_loop(
          c, b, [i, x],
          [i.get_shape(), tensor_shape.TensorShape([5])])

  def testWhileShapeInferenceRaggedTensor(self):
    # [FUNCTION DOC]
    # Test: shape inference of loops over ragged tensors. Default invariants allow adding values to
    # rows but adding rows raises a value error (graph mode only). Explicit shape invariants (a plain
    # shape, deprecated, or a ragged tensor spec) allow both changes; checks values and row split
    # static shapes.
    i = constant_op.constant(0)
    x = ragged_factory_ops.constant([[1, 2], [3], [4, 5, 6]])
    c = lambda i, _: i < 10

    def b1(i, x):  # Adds new values to rows (but doesn't create new rows)
      # [FUNCTION DOC]
      # HELPER: body that increments i and doubles each row by concatenating along axis one.
      return [
          i + 1,
          array_ops.concat([x, x], axis=1)
      ]

    def b2(i, x):  # Adds new rows.
      # [FUNCTION DOC]
      # HELPER: body that increments i and doubles the number of rows by concatenating along axis zero.
      return [
          i + 1,
          array_ops.concat([x, x], axis=0)
      ]

    def check_shapes(r, values, splits):
      # [FUNCTION DOC]
      # HELPER: asserts the ragged tensor's flat values and row splits have static shapes compatible with
      # the given expected shapes.
      self.assertTrue(r.values.shape.is_compatible_with(values))
      self.assertTrue(r.row_splits.shape.is_compatible_with(splits))

    # Default shape invariant; b1 adds new values to rows.
    _, r = while_loop_tf.while_loop(c, b1, [i, x])
    check_shapes(r, values=[None], splits=[4])

    # Default shape invariant; b2 adds new rows (not allowed).
    if not context.executing_eagerly():
      with self.assertRaises(ValueError):
        _, r = while_loop_tf.while_loop(c, b2, [i, x])

    # Explicit shape invariant; b1 adds new values to rows.
    # (deprecated: use TensorShape instead of RaggedTensorSpec)
    _, r = while_loop_tf.while_loop(
        c, b1, [i, x],
        [i.get_shape(), tensor_shape.TensorShape([None, None])])
    check_shapes(r, values=[None], splits=[None])

    # Explicit shape invariant; b1 adds new values to rows.
    _, r = while_loop_tf.while_loop(c, b1, [i, x], [
        i.get_shape(),
        ragged_tensor.RaggedTensorSpec([None, None], dtypes.int32)
    ])
    check_shapes(r, values=[None], splits=[None])

    # Explicit shape invariant; b2 adds new rows.
    _, r = while_loop_tf.while_loop(c, b2, [i, x], [
        i.get_shape(),
        ragged_tensor.RaggedTensorSpec([None, None], dtypes.int32)
    ])
    check_shapes(r, values=[None], splits=[None])

  def testWhileShapeInferenceRaggedTensorRaggedRank2(self):
    # [FUNCTION DOC]
    # Test: loop over a ragged tensor with ragged rank two, appending a slice each iteration. Checks
    # the outer row splits has shape 3 and nested splits and values have the expected sizes (known or
    # unknown).
    i = constant_op.constant(0)
    x = ragged_factory_ops.constant([[[1, 2], [3], [4, 5, 6]],
                                     [[], [8, 9, 10]]])
    c = lambda i, _: i < 10
    def b(i, x):
      # [FUNCTION DOC]
      # HELPER: body that increments i and appends the i-th column slice along the last axis.
      return [
          i + 1,
          array_ops.concat([x, x[..., i:i+1]], axis=-1)
      ]

    _, r = while_loop_tf.while_loop(c, b, [i, x])
    self.assertEqual(r.row_splits.shape.as_list(), [3])
    self.assertIn(r.values.row_splits.shape.as_list(), ([6], [None]))
    self.assertIn(r.values.values.shape.as_list(), ([49], [None]))

  def testWhileShapeInvariantTensorSpec(self):
    # [FUNCTION DOC]
    # Test: loop whose body stacks a tensor, with shape invariants given as tensor specs (scalar int,
    # unknown-shape int). Expects the loop to build without errors; nothing is asserted beyond that.
    i = constant_op.constant(0)
    x = constant_op.constant([1])
    c = lambda i, _: i < 10
    b = lambda i, x: (i + 1, array_ops_stack.stack([x, x]))
    shape_invariants = [
        tensor_lib.TensorSpec([], dtype=dtypes.int32),
        tensor_lib.TensorSpec(None, dtype=dtypes.int32)]
    while_loop_tf.while_loop(c, b, [i, x], shape_invariants)

  # TODO(b/131265085) Remove this decorator when bug is fixed.
  @test_util.build_as_function_and_v1_graph
  def testWhileShapeInvariantWrongTypeSpecType(self):
    # [FUNCTION DOC]
    # Test (built as function and v1 graph): a sparse tensor spec as invariant works for a sparse
    # loop variable but raises a type error for a dense tensor, a ragged tensor or an integer type
    # mismatch for the counter.
    c = lambda i, _: i < 10
    b = lambda i, x: (i + 1, x)
    i = constant_op.constant(0)
    x = sparse_tensor.SparseTensor([[0]], [1.0], [10])
    shape_invariants = [
        tensor_lib.TensorSpec([], dtype=dtypes.int32),
        sparse_tensor.SparseTensorSpec([None])]
    while_loop_tf.while_loop(c, b, [i, x], shape_invariants)

    x2 = constant_op.constant([1])
    with self.assertRaises(TypeError):
      while_loop_tf.while_loop(c, b, [i, x2], shape_invariants)

    x3 = ragged_factory_ops.constant([[1, 2], [3]])
    with self.assertRaises(TypeError):
      while_loop_tf.while_loop(c, b, [i, x3], shape_invariants)

    i2 = constant_op.constant(0.0)
    with self.assertRaises(TypeError):
      while_loop_tf.while_loop(c, b, [i2, x], shape_invariants)

  # TODO(b/131265085) Remove this decorator when bug is fixed.
  @test_util.build_as_function_and_v1_graph
  def testWhileShapeInvariantBadType(self):
    # [FUNCTION DOC]
    # Test (built as function and v1 graph): shape invariants given as strings must raise a value
    # error or a type error.
    i = constant_op.constant(0)
    x = constant_op.constant([1])
    c = lambda i, _: i < 10
    b = lambda i, x: (i + 1, x)
    with self.assertRaises((ValueError, TypeError)):
      while_loop_tf.while_loop(c, b, [i, x], ["foo", "bar"])

  def _testNestedWhile_1(self, use_gpu):
    # [FUNCTION DOC]
    # HELPER: shared body for testNestedWhile_1; takes a use-GPU flag. Outer loop adds an inner CPU-pinned
    # sum to a counter until at least 200 and expects 225.
    with self.cached_session(use_gpu=use_gpu):
      n = constant_op.constant(0)

      def cpu_sum(s):
        # [FUNCTION DOC]
        # HELPER: inner loop; sums integers below ten with the add pinned to the CPU. Returns the sum.
        c = lambda i, s: math_ops.less(i, 10)

        def b(i, s):
          # [FUNCTION DOC]
          # HELPER: inner loop body; increments i and adds i to s on the CPU. Returns both.
          i1 = math_ops.add(i, 1)
          with ops.device("/cpu:0"):
            s1 = math_ops.add(i, s)
          return i1, s1

        _, r_s = while_loop_tf.while_loop(c, b, [n, s])
        return r_s

      c = lambda x: math_ops.less(x, 200)
      b = lambda x: math_ops.add(x, cpu_sum(n))
      r = while_loop_tf.while_loop(c, b, [n])
      self.assertEqual(225, self.evaluate(r))

  def testNestedWhile_1(self):
    # [FUNCTION DOC]
    # Test: runs the shared helper without and then with the GPU.
    self._testNestedWhile_1(use_gpu=False)
    self._testNestedWhile_1(use_gpu=True)

  def _testNestedWhile_2(self, use_gpu):
    # [FUNCTION DOC]
    # HELPER: shared body for testNestedWhile_2; takes a use-GPU flag. Nested loops partitioned across
    # devices (print into enter, exit into square on the CPU). Expects 1048576.
    # Test the cases that A -> Enter and Exit -> A are partitioned.
    with self.cached_session(use_gpu=use_gpu):
      s0 = constant_op.constant(2.0)

      def inner_loop(s):
        # [FUNCTION DOC]
        # HELPER: inner while loop; doubles s until at least 20 with a single parallel iteration.
        c = lambda s: math_ops.less(s, 20.0)

        def b(s):
          # [FUNCTION DOC]
          # HELPER: inner loop body; returns s plus s.
          s1 = math_ops.add(s, s)
          return s1

        r_s = while_loop_tf.while_loop(c, b, [s], parallel_iterations=1)
        return r_s

      outer_c = lambda x: math_ops.less(x, 3000.0)

      def outer_b(x):
        # [FUNCTION DOC]
        # HELPER: outer body; prints x (adds a print op), runs the inner loop and squares the result on CPU.
        x = logging_ops.Print(x, [x])  # Edge "Print -> Enter" is partitioned
        x = inner_loop(x)
        with ops.device("/cpu:0"):
          x = math_ops.square(x)  # Edge "Exit -> Square" is partitioned
        return x

      r = while_loop_tf.while_loop(
          outer_c, outer_b, [s0], parallel_iterations=1)
      self.assertEqual(1048576.0, self.evaluate(r))

  def testNestedWhile_2(self):
    # [FUNCTION DOC]
    # Test: runs the shared helper without and then with the GPU.
    self._testNestedWhile_2(use_gpu=False)
    self._testNestedWhile_2(use_gpu=True)

  @test_util.run_v1_only("b/120545219")
  def testWhileWithControl_1(self):
    # [FUNCTION DOC]
    # Test (v1 only): loop body creates a constant under a control dependency on the loop variable.
    # Expects the final value 12.
    with self.cached_session():
      n = constant_op.constant(0)
      r = constant_op.constant(0)
      condition = lambda n_, r_: math_ops.less(n_, 10)

      def body(n_, r_):
        # [FUNCTION DOC]
        # HELPER: increments n and returns a constant 12 created under a control dependency on r.
        n_ = math_ops.add(n_, 1)
        with r_.graph.control_dependencies([r_]):
          r_ = constant_op.constant(12)
        return [n_, r_]

      res = while_loop_tf.while_loop(
          condition, body, [n, r], parallel_iterations=1)
      self.assertAllEqual(12, res[1])

  @test_util.run_deprecated_v1
  def testWhileWithControl_2(self):
    # [FUNCTION DOC]
    # Test (deprecated v1 mode): same but a single loop variable. Expects 12.
    with self.cached_session():
      r = constant_op.constant(0)
      condition = lambda r_: math_ops.less(r_, 10)

      def body(r_):
        # [FUNCTION DOC]
        # HELPER: returns a constant 12 created under a control dependency on the loop variable.
        with r_.graph.control_dependencies([r_]):
          r_ = constant_op.constant(12)
        return [r_]

      res = while_loop_tf.while_loop(
          condition, body, [r], parallel_iterations=1)
      self.assertAllEqual(12, self.evaluate(res))

  @test_util.run_v1_only("b/120545219")
  def testWhileWithControl_3(self):
    # [FUNCTION DOC]
    # Test (v1 only): while loop created under a control dependency on a placeholder. Feeding true
    # expects result 10.
    with self.cached_session() as sess:
      b = array_ops.placeholder(dtypes.bool)
      c = constant_op.constant(1)
      x0 = constant_op.constant(0)
      with ops.control_dependencies([b]):
        r = while_loop_tf.while_loop(lambda x: x < 10, lambda x: x + c, [x0])
      self.assertEqual(10, sess.run(r, {b: True}))

  @test_util.run_v1_only("b/120545219")
  def testWhileWithControl_4(self):
    # [FUNCTION DOC]
    # Test (v1 only): like the previous test but the body uses an identity of a constant. Expects 10.
    with self.cached_session() as sess:
      b = array_ops.placeholder(dtypes.bool)
      c = constant_op.constant(1)
      x0 = constant_op.constant(0)
      with ops.control_dependencies([b]):
        r = while_loop_tf.while_loop(
            lambda x: x < 10, lambda x: x + array_ops.identity(c), [x0])
      self.assertEqual(10, sess.run(r, {b: True}))

  @test_util.run_v1_only("b/120545219")
  def testWhileWithControl_5(self):
    # [FUNCTION DOC]
    # Test (v1 only): the body itself has a control dependency on the placeholder. Expects 10.
    with self.cached_session() as sess:
      b = array_ops.placeholder(dtypes.bool)
      c = constant_op.constant(1)
      x0 = constant_op.constant(0)

      def body(x):
        # [FUNCTION DOC]
        # HELPER: returns x plus the captured constant under a control dependency on the placeholder.
        with ops.control_dependencies([b]):
          return x + c

      r = while_loop_tf.while_loop(lambda x: x < 10, body, [x0])
      self.assertEqual(10, sess.run(r, {b: True}))

  def testWhileCondWithControl(self):
    # [FUNCTION DOC]
    # Test: ensures an outer control dependency context adds no control edges to nodes inside the
    # cond and while contexts. A countdown loop with cond inside runs under a no-op dependency and
    # returns 0.
    # Ensure that no control edges by an outer control dependency context are
    # added to nodes inside cond/while contexts.
    with self.cached_session() as sess:
      const_true = lambda: constant_op.constant(True)
      const_false = lambda: constant_op.constant(False)
      cond = lambda i: tf_cond.cond(i > 0, const_true, const_false)
      body = lambda i: tf_cond.cond(i > 0, lambda: i - 1, lambda: i)

      with ops.control_dependencies([control_flow_ops.no_op()]):
        loop = while_loop_tf.while_loop(cond, body, (constant_op.constant(5),))
      self.assertEqual(0, self.evaluate(loop))

  @test_util.disable_control_flow_v2("b/113324949 (ref vars)")
  @test_util.run_v1_only("b/120545219")
  def testWhileCondWithControl_1(self):
    # [FUNCTION DOC]
    # Test (legacy control flow, v1 only): loop whose body has a cond that squares a variable under a
    # control dependency. After four iterations expects the loop result 4 and the variable 65536.
    with self.cached_session():
      v = variable_scope.get_variable(
          "v", [], initializer=init_ops.constant_initializer(2))
      i0 = constant_op.constant(0)
      with ops.control_dependencies([i0]):

        def loop_condition(i):
          # [FUNCTION DOC]
          # HELPER: loop condition; true while i is below four.
          return i < 4

        def loop_body(i):
          # [FUNCTION DOC]
          # HELPER: runs a cond that assigns the square of the variable (true branch) and returns i plus one
          # under a control dependency on the cond.
          some_cond = tf_cond.cond(
              constant_op.constant(True),
              lambda: state_ops.assign(v, math_ops.square(v)), lambda: v)
          with ops.control_dependencies([some_cond]):
            return i + 1

      r = while_loop_tf.while_loop(loop_condition, loop_body, (i0,))
      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(4, self.evaluate(r))
      self.assertAllClose(65536.0, self.evaluate(v))

  @test_util.disable_control_flow_v2("b/113324949 (ref vars)")
  @test_util.run_v1_only("b/120545219")
  def testWhileCondExitControl(self):
    # [FUNCTION DOC]
    # Test (legacy control flow, v1 only): a loop in the false branch of a cond, with a control edge
    # from the loop Exit to a constant. Expects 6.0 as result and the variable ending at 99.

    with self.cached_session():
      v = variables.Variable(1)

      def false_branch():
        # [FUNCTION DOC]
        # HELPER: runs a loop to 100 assigning the counter into a variable, then returns a constant 6.0
        # under a control dependency on the loop.
        cond = lambda i: i < 100

        def body(i):
          # [FUNCTION DOC]
          # HELPER: assigns i to the variable and returns the assigned value plus one.
          x = state_ops.assign(v, i)
          return x + 1

        loop = while_loop_tf.while_loop(cond, body, [0])
        # Make sure to handle correctly control edge from Exit to a node.
        with ops.control_dependencies([loop]):
          return constant_op.constant(6.0)

      r = tf_cond.cond(
          constant_op.constant(False), lambda: constant_op.constant(1.0),
          false_branch)
      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(6.0, self.evaluate(r))
      self.assertEqual(99, self.evaluate(v))

  def testCondWhile_1(self):
    # [FUNCTION DOC]
    # Test: cond whose true branch is a while loop counting to 10. Expects 10.

    with self.cached_session():
      n = ops.convert_to_tensor(0, name="n")
      c = lambda x: math_ops.less(x, 10)
      b = lambda x: math_ops.add(x, 1)
      r = tf_cond.cond(
          math_ops.less(0, 1), lambda: while_loop_tf.while_loop(c, b, [n]),
          lambda: n)
      self.assertAllEqual(10, self.evaluate(r))

  def testCondWhile_2(self):
    # [FUNCTION DOC]
    # Test: cond whose false branch is a while loop counting to 10. Expects 10.

    with self.cached_session():
      n = ops.convert_to_tensor(0)
      c = lambda x: math_ops.less(x, 10)
      b = lambda x: math_ops.add(x, 1)
      r = tf_cond.cond(
          math_ops.less(1, 0), lambda: math_ops.add(n, 1),
          lambda: while_loop_tf.while_loop(c, b, [n]))
      self.assertAllEqual(10, self.evaluate(r))

  def _testCondWhile_3(self, use_gpu):
    # [FUNCTION DOC]
    # HELPER: shared body for testCondWhile_3; takes a use-GPU flag. Cond on a placeholder choosing a
    # loop (to 10) or multiplication by two; checks values and gradients (10 and 1.0 when true, 0 and
    # 2.0 when false).
    with self.cached_session(use_gpu=use_gpu) as sess:
      p = array_ops.placeholder(dtypes.bool)
      n = constant_op.constant(0.0)

      def c(x):
        # [FUNCTION DOC]
        # HELPER: loop condition; true while x is below 10.
        return math_ops.less(x, 10.0)

      def b(x):
        # [FUNCTION DOC]
        # HELPER: loop body; adds one on the CPU.
        with ops.device("/cpu:0"):
          x1 = math_ops.add(x, 1.0)
        return x1

      r = tf_cond.cond(
          p,
          lambda: while_loop_tf.while_loop(c, b, [n]),
          lambda: math_ops.multiply(n, 2.0),
      )
      r1 = gradients_impl.gradients(r, [n])
      self.assertEqual(10., sess.run(r, {p: True}))
      self.assertEqual([1.0], sess.run(r1, {p: True}))
      self.assertEqual(0.0, sess.run(r, {p: False}))
      self.assertEqual([2.0], sess.run(r1, {p: False}))

  @test_util.run_deprecated_v1
  def testCondWhile_3(self):
    # [FUNCTION DOC]
    # Test (deprecated v1 mode): runs the shared helper without and then with the GPU.
    self._testCondWhile_3(use_gpu=False)
    self._testCondWhile_3(use_gpu=True)

  def testWhileCond_1(self):
    # [FUNCTION DOC]
    # Test: while loop whose body is a cond with a true predicate adding one. Expects 10.

    with self.cached_session():
      i = ops.convert_to_tensor(0, name="i")
      n = ops.convert_to_tensor(10, name="n")
      one = ops.convert_to_tensor(1, name="one")
      c = lambda x: math_ops.less(x, n)
      # pylint: disable=undefined-variable
      # for OSS build
      b = lambda x: tf_cond.cond(
          constant_op.constant(True),
          lambda: math_ops.add(x, one), lambda: math_ops.subtract(x, one))
      # pylint: enable=undefined-variable
      r = while_loop_tf.while_loop(c, b, [i])
      self.assertAllEqual(10, self.evaluate(r))

  def testWhileCond_2(self):
    # [FUNCTION DOC]
    # Test: while loop whose body is a cond with true predicate adding one (else branch returns the
    # outer constant). Expects 10.

    with self.cached_session():
      n = ops.convert_to_tensor(0, name="n")
      c = lambda x: math_ops.less(x, 10)
      b = lambda x: tf_cond.cond(
          constant_op.constant(True), lambda: math_ops.add(x, 1), lambda: n)
      r = while_loop_tf.while_loop(c, b, [n])
      self.assertAllEqual(10, self.evaluate(r))

  def testWhileCond_3(self):
    # [FUNCTION DOC]
    # Test: while loop with a cond body using a computed predicate (zero less than one). Expects 10.

    with self.cached_session():
      n = ops.convert_to_tensor(0)
      c = lambda x: math_ops.less(x, 10)
      # pylint: disable=undefined-variable
      # for OSS build
      b = lambda x: tf_cond.cond(math_ops.less(0, 1),
                                 lambda: math_ops.add(x, 1),
                                 lambda: math_ops.subtract(x, 1))
      # pylint: enable=undefined-variable
      r = while_loop_tf.while_loop(c, b, [n])
      self.assertAllEqual(10, self.evaluate(r))

  @test_util.run_deprecated_v1
  def testWhileCondGradMultiDevice(self):
    # [FUNCTION DOC]
    # Test (deprecated v1 mode): loop whose body is a cond, gradient computed on cpu 1 and second
    # gradient on cpu 0 with two CPUs. Gradient is 8 for true and 0 for false; second derivative
    # zero (checked only under control flow v2).
    config = config_pb2.ConfigProto(device_count={"CPU": 2},
                                    allow_soft_placement=True)
    with self.cached_session(config=config) as sess:
      pred = array_ops.placeholder(dtypes.bool, [])
      x_init = constant_op.constant(1.0)

      with ops.device("/cpu:0"):
        z = while_loop_tf.while_loop(
            lambda i, _: i < 3, lambda i, x:
            (i + 1, tf_cond.cond(pred, lambda: x * 2.0, lambda: 10.0)),
            [0, x_init])

      with ops.device("/cpu:1"):
        grad = gradients_impl.gradients(z, x_init)[0]

      with ops.device("/cpu:0"):
        grad_grad = gradients_impl.gradients(grad, x_init)[0]

      self.assertEqual(sess.run(grad, {pred: True}), 8.0)
      self.assertEqual(sess.run(grad, {pred: False}), 0.0)

      if not control_flow_util.ENABLE_CONTROL_FLOW_V2:
        return

      self.assertEqual(sess.run(grad_grad, {pred: True}), 0.0)
      self.assertEqual(sess.run(grad_grad, {pred: False}), 0.0)

  # NOTE: It is ok to have parallel_iterations > 1
  @test_util.disable_control_flow_v2("b/113324949 (RefVariable)")
  @test_util.run_deprecated_v1
  def testWhileUpdateVariable_1(self):
    # [FUNCTION DOC]
    # Test (legacy control flow, deprecated v1 mode): loop updating variable entries by scatter update
    # with a grouped dependency, parallel iterations one. Expects counter 3 and the variable [10, 10, 10].
    with self.cached_session():
      select = variables.Variable([3.0, 4.0, 5.0])
      n = constant_op.constant(0)

      def loop_iterator(j):
        # [FUNCTION DOC]
        # HELPER: loop condition; true while j is below three.
        return math_ops.less(j, 3)

      def loop_body(j):
        # [FUNCTION DOC]
        # HELPER: scatter-updates entry j of the variable to 10, groups the update and returns j plus one
        # dependent on it.
        ns = state_ops.scatter_update(select, j, 10.0)
        nj = math_ops.add(j, 1)
        op = control_flow_ops.group(ns)
        nj = control_flow_ops.with_dependencies([op], nj)
        return [nj]

      r = while_loop_tf.while_loop(
          loop_iterator, loop_body, [n], parallel_iterations=1)
      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(3, self.evaluate(r))
      result = self.evaluate(select)
      self.assertAllClose(np.array([10.0, 10.0, 10.0]), result)

  @test_util.disable_control_flow_v2("b/113324949 (RefVariable)")
  @test_util.run_v1_only("b/120545219")
  def testWhileUpdateVariable_2(self):
    # [FUNCTION DOC]
    # Test (legacy control flow, v1 only): like the above with two variables updated in the loop.
    # Expects counter 3 and both variables [10, 10, 10].
    with self.cached_session():
      select1 = variables.Variable([3.0, 4.0, 5.0])
      select2 = variables.Variable([3.0, 4.0, 5.0])
      n = constant_op.constant(0)

      def loop_iterator(j):
        # [FUNCTION DOC]
        # HELPER: loop condition; true while j is below three.
        return math_ops.less(j, 3)

      def loop_body(j):
        # [FUNCTION DOC]
        # HELPER: scatter-updates entry j of both variables, groups the updates and returns j plus one
        # dependent on them.
        ns1 = state_ops.scatter_update(select1, j, 10.0)
        ns2 = state_ops.scatter_update(select2, j, 10.0)
        nj = math_ops.add(j, 1)
        op = control_flow_ops.group(ns1, ns2)
        nj = control_flow_ops.with_dependencies([op], nj)
        return [nj]

      r = while_loop_tf.while_loop(
          loop_iterator, loop_body, [n], parallel_iterations=1)
      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(3, self.evaluate(r))
      result1 = self.evaluate(select1)
      self.assertAllClose(np.array([10.0, 10.0, 10.0]), result1)
      result2 = self.evaluate(select2)
      self.assertAllClose(np.array([10.0, 10.0, 10.0]), result2)

  @test_util.disable_control_flow_v2("b/113324949 (RefVariable)")
  @test_util.run_v1_only("b/120545219")
  def testWhileUpdateVariable_3(self):
    # [FUNCTION DOC]
    # Test (legacy control flow, v1 only): the loop carries the scatter update result as a loop
    # variable. Expects the final result [10, 10, 10].
    with self.cached_session():
      select = variables.Variable([3.0, 4.0, 5.0])
      n = constant_op.constant(0)

      def loop_iterator(j, _):
        # [FUNCTION DOC]
        # HELPER: loop condition; true while j is below three (ignores the second variable).
        return math_ops.less(j, 3)

      def loop_body(j, _):
        # [FUNCTION DOC]
        # HELPER: scatter-updates entry j of the variable and returns the new counter and the update result.
        ns = state_ops.scatter_update(select, j, 10.0)
        nj = math_ops.add(j, 1)
        return [nj, ns]

      r = while_loop_tf.while_loop(
          loop_iterator,
          loop_body, [n, array_ops.identity(select)],
          parallel_iterations=1)
      self.evaluate(variables.global_variables_initializer())
      result = r[1]
    self.assertAllClose(np.array([10.0, 10.0, 10.0]), result)

  @test_util.disable_control_flow_v2("b/113324949 (RefVariable)")
  @test_util.run_v1_only("b/120545219")
  def testWhileUpdateVariable_4(self):
    # [FUNCTION DOC]
    # Test (legacy control flow, v1 only): the loop body adds a precomputed assign-add of one variable
    # into another; expects that the outer assign runs once so var_b goes from 0 to 10 only when the
    # loop is run.
    with self.cached_session():
      var_a = variables.Variable(0, name="a")
      var_b = variables.Variable(0, name="b")
      self.evaluate(variables.global_variables_initializer())

      c = constant_op.constant(0, name="c")
      asn1 = state_ops.assign_add(var_a, 1, name="a_add")

      # Loop condition
      def pred(i):
        # [FUNCTION DOC]
        # HELPER: loop condition; true while i is below ten.
        return math_ops.less(i, 10)

      # Loop body
      def loop_body(i):
        # [FUNCTION DOC]
        # HELPER: adds the outer assign-add result into var_b and returns i plus one dependent on it.
        asn2 = state_ops.assign_add(var_b, asn1, name="b_add")
        with ops.control_dependencies([asn2]):
          ni = math_ops.add(i, 1, name="i_add")
        return ni

      lpa = while_loop_tf.while_loop(
          pred, loop_body, [c], parallel_iterations=1)

      self.assertEqual(0, self.evaluate(var_b))
      self.evaluate(lpa)  # Run the loop
      self.assertEqual(10, self.evaluate(var_b))

  @test_util.disable_control_flow_v2("b/113324949 (RefVariable)")
  @test_util.run_v1_only("b/120545219")
  def testWhileUpdateVariable_5(self):
    # [FUNCTION DOC]
    # Test (legacy control flow, v1 only): the loop condition reads a variable that the body
    # increments. Expects both variables to end at 10.
    with self.cached_session():
      # Create some variables.
      var_a = variables.Variable(0, name="a")
      var_b = variables.Variable(0, name="b")
      self.evaluate(variables.global_variables_initializer())

      # Change condition to check var_b
      def pred(_):
        # [FUNCTION DOC]
        # HELPER: loop condition ignoring its argument; true while var_b is below ten.
        return math_ops.less(var_b, 10)

      # Change body to increment var_b
      def loop_body(i):
        # [FUNCTION DOC]
        # HELPER: increments both variables and returns an identity of var_b dependent on both updates.
        asn1 = state_ops.assign_add(
            var_a, constant_op.constant(1), name="a_add")
        asn2 = state_ops.assign_add(
            var_b, constant_op.constant(1), name="b_add")
        with ops.control_dependencies([asn1, asn2]):
          inc_b = array_ops.identity(var_b)
        return inc_b

      lpa = while_loop_tf.while_loop(
          pred, loop_body, [var_b], parallel_iterations=1, name="loop")

      self.assertEqual(0, self.evaluate(var_b))
      self.evaluate(lpa)  # Run the loop
      self.assertEqual(10, self.evaluate(var_a))
      self.assertEqual(10, self.evaluate(var_b))

  @test_util.disable_control_flow_v2("b/113324949 (RefVariable)")
  @test_util.run_v1_only("b/120545219")
  def testWhileUpdateVariable_6(self):
    # [FUNCTION DOC]
    # Test (legacy control flow, v1 only): chained control dependencies increment var_a and add it to
    # var_b each iteration. Expects var_b 55 and var_a 10.
    with self.cached_session():
      # Create some variables.
      var_a = variables.Variable(0, name="a")
      var_b = variables.Variable(0, name="b")
      c = constant_op.constant(0)
      self.evaluate(variables.global_variables_initializer())

      # Loop condition
      def pred(i):
        # [FUNCTION DOC]
        # HELPER: loop condition; true while i is below ten.
        return math_ops.less(i, 10)

      # Loop body
      def loop_body(i):
        # [FUNCTION DOC]
        # HELPER: increments var_a, adds var_a into var_b after it, and returns i plus one after that.
        asn1 = state_ops.assign_add(var_a, 1, name="a_add")
        with ops.control_dependencies([asn1]):
          asn2 = state_ops.assign_add(var_b, var_a, name="b_add")
        with ops.control_dependencies([asn2]):
          ni = math_ops.add(i, 1, name="i_add")
          return ni

      lpa = while_loop_tf.while_loop(
          pred, loop_body, [c], parallel_iterations=1, name="loop")

      self.assertEqual(0, self.evaluate(var_b))
      self.evaluate(lpa)  # Run the loop
      self.assertEqual(55, self.evaluate(var_b))
      self.assertEqual(10, self.evaluate(var_a))

  @test_util.run_v1_only("b/120545219")
  def testWhileQueue_1(self):
    # [FUNCTION DOC]
    # Test (v1 only): loop enqueues the counter into a FIFO queue each iteration. Expects the loop
    # result [10] and dequeued values 0 to 9 in order.
    with self.cached_session():
      q = data_flow_ops.FIFOQueue(-1, dtypes.int32)
      i = constant_op.constant(0)

      def c(i):
        # [FUNCTION DOC]
        # HELPER: loop condition; true while i is below ten.
        return math_ops.less(i, 10)

      def b(i):
        # [FUNCTION DOC]
        # HELPER: increments i with a dependency on enqueueing the old i.
        ni = math_ops.add(i, 1)
        ni = control_flow_ops.with_dependencies([q.enqueue((i,))], ni)
        return ni

      r = while_loop_tf.while_loop(c, b, [i], parallel_iterations=1)
      self.assertEqual([10], self.evaluate(r))
      for i in range(10):
        self.assertEqual([i], self.evaluate(q.dequeue()))

  @test_util.run_v1_only("b/120545219")
  def testWhileTimeOut(self):
    # [FUNCTION DOC]
    # Test (v1 only): an infinite loop is run with a one millisecond timeout. Expects a
    # deadline-exceeded error.
    run_options = config_pb2.RunOptions(timeout_in_ms=1)
    with self.cached_session() as sess:
      n = constant_op.constant(0)
      c = lambda x: True
      b = lambda x: math_ops.add(x, 1)
      r = while_loop_tf.while_loop(c, b, [n])
      with self.assertRaises(errors_impl.DeadlineExceededError):
        sess.run(r, options=run_options)

  @test_util.disable_control_flow_v2("b/117119329 (stack)")
  @test_util.run_v1_only("b/120545219")
  def testWhileStack_1(self):
    # [FUNCTION DOC]
    # Test (legacy control flow, v1 only): a loop pushes the counter onto a stack, a second loop pops
    # and sums. Expects 45.
    with self.cached_session():
      s = gen_data_flow_ops.stack_v2(-1, dtypes.int32, stack_name="foo")
      i = constant_op.constant(0)

      def c(i):
        # [FUNCTION DOC]
        # HELPER: loop condition; true while i is below ten.
        return math_ops.less(i, 10)

      def b(i):
        # [FUNCTION DOC]
        # HELPER: increments i with a dependency on pushing the old i onto the stack.
        ni = math_ops.add(i, 1)
        ni = control_flow_ops.with_dependencies(
            [gen_data_flow_ops.stack_push_v2(s, i)], ni)
        return ni

      r = while_loop_tf.while_loop(c, b, [i], parallel_iterations=1)

      x = constant_op.constant(0)

      def c1(i, _):
        # [FUNCTION DOC]
        # HELPER: second loop condition; true while i is greater than zero.
        return math_ops.greater(i, 0)

      def b1(i, x):
        # [FUNCTION DOC]
        # HELPER: second loop body; decrements i and adds a value popped from the stack to x.
        ni = math_ops.subtract(i, 1)
        nx = x + gen_data_flow_ops.stack_pop_v2(s, dtypes.int32)
        return [ni, nx]

      _, rx = while_loop_tf.while_loop(
          c1,
          b1, [r, x],
          [r.get_shape(), tensor_shape.unknown_shape()],
          parallel_iterations=1)
      self.assertEqual(45, self.evaluate(rx))

  def _testWhileGrad_ColocateGradients(self, colocate):
    # [FUNCTION DOC]
    # HELPER: shared body for the colocation test; takes a boolean flag for colocating gradients with
    # ops. Builds a loop squaring a value on the GPU device name (or CPU if no GPU), computes the
    # gradient and checks which ops have the GPU device: only the forward Square op when not
    # colocating, both forward and gradient Square ops when colocating, none of the others. Evaluates
    # the gradient expecting 1024.
    gpu_dev_name = test.gpu_device_name() if test.is_gpu_available(
    ) else "/device:CPU:0"

    graph = ops.Graph()
    with graph.as_default():
      v = constant_op.constant(2.0, name="v")
      c = lambda v: math_ops.less(v, 100.0)

      def b(x):
        # [FUNCTION DOC]
        # HELPER: loop body; squares x under the chosen device scope.
        with ops.device(gpu_dev_name):
          return math_ops.square(x)

      loop = while_loop_tf.while_loop(c, b, [v], parallel_iterations=1)
      r = gradients_impl.gradients(
          loop, v, colocate_gradients_with_ops=colocate)[0]

    r_ops = graph.get_operations()
    r_devices = [(op.name, op.device) for op in r_ops]

    self.assertTrue(any("Square" in op.name for op in r_ops))

    for (name, dev) in r_devices:
      if not colocate and name.endswith("Square"):
        # Only forward graph contain gpu in Square device
        self.assertTrue(gpu_dev_name in dev)
      elif colocate and "Square" in name:
        # Forward and backward graphs contain gpu in Square/Square_grad devices
        self.assertTrue(gpu_dev_name in dev)
      else:
        self.assertFalse(gpu_dev_name in dev)

    with self.session(graph=graph) as sess:
      self.assertAllClose(1024.0, self.evaluate(r))

  @test_util.disable_control_flow_v2("b/116351701 (colocation)")
  @test_util.run_v1_only("b/120545219")
  def testWhileGrad_ColocateGradients(self):
    # [FUNCTION DOC]
    # Test (legacy control flow, v1 only): calls the shared helper with colocation off and on.
    self._testWhileGrad_ColocateGradients(colocate=False)
    self._testWhileGrad_ColocateGradients(colocate=True)

  @test_util.run_v1_only("b/120545219")
  def testWhileGrad_Square(self):
    # [FUNCTION DOC]
    # Test (v1 only): loop squaring 2 until at least 100 then wrapped in a cond; gradient with respect
    # to the start is 1024.
    with self.cached_session():
      v = constant_op.constant(2.0, name="v")
      c = lambda v: math_ops.less(v, 100.0)
      b = math_ops.square
      r = while_loop_tf.while_loop(c, b, [v], parallel_iterations=1)
      r = tf_cond.cond(math_ops.less(1, 2), lambda: r, lambda: v)

      r = gradients_impl.gradients(r, v)[0]
      self.assertAllClose(1024.0, self.evaluate(r))

  @test_util.run_v1_only("b/120545219")
  def testWhileGrad_Shape(self):
    # [FUNCTION DOC]
    # Test (v1 only): loop multiplying by a placeholder with unknown shape five times; gradient with
    # respect to the placeholder has unknown static shape and values [810, 2560] for inputs [3, 4].
    with self.cached_session():
      x = array_ops.placeholder(dtypes.float32, shape=[None])
      v = constant_op.constant([2.0], name="v")
      n = constant_op.constant(0, name="n")
      c = lambda i, v: math_ops.less(i, 5)
      b = lambda i, v: [i + 1, math_ops.multiply(x, v)]
      r = while_loop_tf.while_loop(
          c,
          b, [n, v],
          [n.get_shape(), tensor_shape.unknown_shape()],
          parallel_iterations=1)

      r = gradients_impl.gradients(r[1], x)[0]
      self.assertEqual([None], r.get_shape().as_list())
      self.assertAllClose([810.0, 2560.0], r.eval(feed_dict={x: [3.0, 4.0]}))

  @test_util.run_deprecated_v1
  def testWhileGrad_BaseShape(self):
    # [FUNCTION DOC]
    # Test (deprecated v1 mode): loop whose condition is always false and body multiplies by a
    # placeholder; gradients of loop result and a square of the placeholder. Expects [2, 4] for
    # inputs [1, 2].
    with self.cached_session() as sess:
      x = array_ops.placeholder(dtypes.float32, [None])
      v0 = constant_op.constant([2.0, 2.0], name="v")
      c = lambda v: constant_op.constant(False)
      b = lambda v: math_ops.multiply(v, x)
      r = while_loop_tf.while_loop(c, b, [v0])
      y = math_ops.square(x)

      r = gradients_impl.gradients([r, y], x)[0]
      self.assertAllClose([2.0, 4.0], sess.run(r, feed_dict={x: [1.0, 2.0]}))

  @test_util.run_deprecated_v1
  @test_util.enable_output_all_intermediates
  def testWhileGradAfterSessionRun(self):
    # [FUNCTION DOC]
    # Test (deprecated v1 mode, output-all-intermediates enabled): runs a loop squaring 2 three times
    # (256) and only afterwards computes its gradient. Expects value 256 and gradient 1024.
    v0 = constant_op.constant(2.)
    r = while_loop_tf.while_loop(
        lambda _: True, lambda v: v * v, [v0], maximum_iterations=3)

    self.assertAllEqual(r, 256.)
    grad = gradients_impl.gradients(r, v0)[0]
    self.assertAllClose(grad, 1024.)

  @test_util.run_deprecated_v1
  @test_util.enable_output_all_intermediates
  def testNestedWhileGradAfterSessionRun(self):
    # [FUNCTION DOC]
    # Test (deprecated v1 mode, output-all-intermediates enabled): nested loops (inner loop multiplies
    # by the outer variable); evaluates 256 first, then the gradient, expecting 1024.
    v0 = constant_op.constant(2.)

    def body(v):
      # [FUNCTION DOC]
      # HELPER: outer loop body; runs an inner bounded loop multiplying an inner start of one by the
      # captured outer variable twice, and returns the result.
      inner_v0 = constant_op.constant(1.)
      return while_loop_tf.while_loop(
          lambda _: True, lambda x: x * v, [inner_v0], maximum_iterations=2)

    r = while_loop_tf.while_loop(
        lambda _: True, body, [v0], maximum_iterations=3)

    self.assertAllEqual(r, 256.)
    grad = gradients_impl.gradients(r, v0)[0]
    self.assertAllClose(grad, 1024.)

  @test_util.run_v1_only("b/120545219")
  def testWhileGrad_MultipleUses(self):
    # [FUNCTION DOC]
    # Test (v1 only): loop output used twice (squared again); gradient with respect to the start is
    # 524288.
    with self.cached_session():
      v = constant_op.constant(2.0, name="v")
      c = lambda v: math_ops.less(v, 100.0)
      b = math_ops.square
      r = while_loop_tf.while_loop(c, b, [v], parallel_iterations=1)
      r = math_ops.multiply(r, r)

      r = gradients_impl.gradients(r, v)[0]
      self.assertEqual(524288.0, self.evaluate(r))

  @test_util.run_v1_only("b/120545219")
  def testWhileGrad_LoopAdd(self):
    # [FUNCTION DOC]
    # Test (v1 only): loop output added to itself; gradient with respect to the start is 2048.
    with self.cached_session():
      v = constant_op.constant(2.0, name="v")
      c = lambda v: math_ops.less(v, 100.0)
      b = math_ops.square
      r = while_loop_tf.while_loop(c, b, [v], parallel_iterations=1)
      r = math_ops.add(r, r)

      r = gradients_impl.gradients(r, v)[0]
      self.assertAllClose(2048.0, self.evaluate(r))

  def _testWhileGrad_Mul(self, use_gpu, p_iters):
    # [FUNCTION DOC]
    # HELPER: shared body for testWhileGrad_Mul; takes a use-GPU flag and a parallel-iterations count.
    # Loop multiplies by a constant a until at least 100; checks gradients are 216 for a and 81 for v.
    with self.cached_session(use_gpu=use_gpu) as sess:
      a = constant_op.constant(3.0, name="a")
      v = constant_op.constant(2.0, name="v")
      c = lambda v: math_ops.less(v, 100.0)
      b = lambda v: math_ops.multiply(v, a)
      r = while_loop_tf.while_loop(c, b, [v], parallel_iterations=p_iters)

      grad_a, grad_v = gradients_impl.gradients(r, [a, v])
      grad_a_val, grad_v_val = self.evaluate([grad_a, grad_v])
      self.assertAllClose(216.0, grad_a_val)
      self.assertAllClose(81.0, grad_v_val)

  @test_util.run_deprecated_v1
  def testWhileGrad_Mul(self):
    # [FUNCTION DOC]
    # Test (deprecated v1 mode): runs the shared helper for CPU and GPU with one and ten parallel
    # iterations.
    self._testWhileGrad_Mul(use_gpu=False, p_iters=1)
    self._testWhileGrad_Mul(use_gpu=False, p_iters=10)
    self._testWhileGrad_Mul(use_gpu=True, p_iters=1)
    self._testWhileGrad_Mul(use_gpu=True, p_iters=10)

  def testWhileGradInControlDeps(self):
    # [FUNCTION DOC]
    # Test: in a compiled function, computes the gradient of a squaring loop inside a control
    # dependency on the loop output. Expects the gradient 4 times the cube of the start (32).

    @eager_def_function.function
    def f():
      # [FUNCTION DOC]
      # HELPER: compiled function; runs two loop iterations squaring 2 and returns the gradient computed
      # under a control dependency on the loop result.
      x_init = constant_op.constant(2.)
      loop_cond = lambda i, x: math_ops.less(i, 2)
      loop_body = lambda i, x: [i + 1, x**2]
      _, x = while_loop_tf.while_loop(loop_cond, loop_body, [0, x_init])
      with ops.control_dependencies([x]):
        (grad,) = gradients_impl.gradients(x, x_init)
        return grad

    self.assertAllEqual(f(), 4. * 2.**3)  # 4 * x_init ^ 3

  @test_util.run_deprecated_v1
  def testTfFunctionInV1WhileLoop(self):
    # [FUNCTION DOC]
    # Test (deprecated v1 mode): with function inlining enabled in the optimizer config (asserted),
    # a compiled function that creates a constant is used as body of a v1 while loop with maximum
    # iterations five. Expects 5.

    # This test specifically tests that creating a Const node inside a
    # tf.function inside a v1 while_loop while inlining is turned on works.
    config = opt_cfg()
    assert config.graph_options.optimizer_options.do_function_inlining
    with session.Session(config=config):

      @eager_def_function.function
      def loop_body(i):
        # [FUNCTION DOC]
        # HELPER: compiled function returning i plus one (creates a constant inside).
        # Here we create the const.
        return i + 1.

      loop_cond = lambda i: True
      x = while_loop_tf.while_loop(
          loop_cond, loop_body, [0.], maximum_iterations=5)
      self.assertAllEqual(x, 5.)

  def _testNestedWhileCondWhileGrad(self, use_gpu):
    # [FUNCTION DOC]
    # HELPER: shared body for the nested while/cond/while tests; takes a use-GPU flag. Outer loop body is
    # a cond whose true branch squares the result of an inner four-step doubling loop. Expects gradient 512.

    with self.cached_session(use_gpu=use_gpu):
      v = constant_op.constant(1.0)

      def inner_loop(s):
        # [FUNCTION DOC]
        # HELPER: runs four iterations doubling s; returns the loop outputs (counter and value).
        z = constant_op.constant(0)
        c = lambda i, x: math_ops.less(i, 4)
        b = lambda i, x: [math_ops.add(i, 1), math_ops.multiply(x, 2.0)]
        return while_loop_tf.while_loop(c, b, [z, s])

      c = lambda x: math_ops.less(x, 128.0)

      def b(x):
        # [FUNCTION DOC]
        # HELPER: outer loop body; cond on a true constant, true branch squares the inner loop value.
        return tf_cond.cond(
            constant_op.constant(True),
            lambda: math_ops.square(inner_loop(x)[1]),
            lambda: math_ops.multiply(x, 2.0))

      r = while_loop_tf.while_loop(c, b, [v])
      r = gradients_impl.gradients(r, v)[0]
      self.assertAllClose(512.0, self.evaluate(r))

  @test_util.run_deprecated_v1
  def testNestedWhileCondWhileGrad(self):
    # [FUNCTION DOC]
    # Test (deprecated v1 mode): CPU variant of the shared helper.
    self._testNestedWhileCondWhileGrad(use_gpu=False)

  @test_util.run_deprecated_v1
  def testNestedWhileCondWhileGradGpu(self):
    # [FUNCTION DOC]
    # Test (deprecated v1 mode): GPU variant of the shared helper.
    self._testNestedWhileCondWhileGrad(use_gpu=True)

  @test_util.run_v1_only("b/120545219")
  def testWhileGrad_Variable(self):
    # [FUNCTION DOC]
    # Test (v1 only): gradient of a loop multiplying by a reference variable with respect to that
    # variable. Expects 216.
    with self.cached_session():
      a = variables.Variable(3.0)
      v = constant_op.constant(2.0, name="v")
      c = lambda v: math_ops.less(v, 100.0)
      b = lambda v: math_ops.multiply(v, a)
      r = while_loop_tf.while_loop(c, b, [v], parallel_iterations=1)

      r = gradients_impl.gradients(r, a)
      self.evaluate(variables.global_variables_initializer())
      self.assertAllClose(216.0, r[0])

  @test_util.run_deprecated_v1
  def testWhileGrad_ResourceVariable(self):
    # [FUNCTION DOC]
    # Test (deprecated v1 mode): same with a resource variable. Expects 216.
    with self.cached_session():
      a = resource_variable_ops.ResourceVariable(3.0)
      v = constant_op.constant(2.0, name="v")
      c = lambda v: math_ops.less(v, 100.0)
      b = lambda v: math_ops.multiply(v, a)
      r = while_loop_tf.while_loop(c, b, [v], parallel_iterations=1)

      g = gradients_impl.gradients(r, a)
      self.evaluate(variables.global_variables_initializer())
      self.assertAllClose(216.0, g[0])

  def testWhileGrad_EagerResourceVariable(self):
    # [FUNCTION DOC]
    # Test: in eager mode, a compiled function runs a two-step loop using the sum of a resource
    # variable and takes the gradient with respect to a constant. Expects 32.
    with context.eager_mode():
      a = resource_variable_ops.ResourceVariable(
          np.ones([2, 2], dtype=np.float32))
      v = constant_op.constant(1.0)

      @eager_def_function.function
      def fn():
        # [FUNCTION DOC]
        # HELPER: compiled function; loop multiplying by the sum of the variable and the constant, returns
        # the gradient of the result with respect to the constant.
        r = while_loop_tf.while_loop(
            lambda i, _: i < 2, lambda i, x:
            (i + 1, x * math_ops.reduce_sum(a) * v), [0, 1.0])[1]
        return gradients_impl.gradients(r, [v])[0]

      self.assertEqual(self.evaluate(fn()), 32.)

  def testWhileGrad_ResourceVarInFunctionCall(self):
    # [FUNCTION DOC]
    # Test: loop body calls a compiled function that reads sparse rows of a resource variable;
    # gradient with respect to the variable is [0, 2, 0, 2].

    @eager_def_function.function
    def foo(x, var):
      # [FUNCTION DOC]
      # HELPER: compiled function; adds the sum of rows one and three of the variable to x.
      return x + math_ops.reduce_sum(var.sparse_read([1, 3]))

    @eager_def_function.function
    def bar(var):
      # [FUNCTION DOC]
      # HELPER: compiled function; runs a two-step loop calling foo and returns the gradient with respect
      # to the variable.
      r = while_loop_tf.while_loop(
          lambda i, _: i < 2, lambda i, x: (i + 1, foo(x, var)), [0, 0.0])[1]
      return gradients_impl.gradients(r, var)[0]

    var = resource_variable_ops.ResourceVariable([1., 2., 3., 4.])
    self.evaluate(variables.global_variables_initializer())
    grad = self.evaluate(bar(var))
    self.assertAllEqual(gradient_checker_v2._to_numpy(grad), [0., 2., 0., 2.])

  def testWhileGrad_ResourceVarInNestedFunctionCall(self):
    # [FUNCTION DOC]
    # Test: same as above but the loop body calls a compiled function that calls another one.
    # Expects [0, 2, 0, 2].

    @eager_def_function.function
    def foo(x, var):
      # [FUNCTION DOC]
      # HELPER: compiled function; adds the sum of rows one and three of the variable to x.
      return x + math_ops.reduce_sum(var.sparse_read([1, 3]))

    @eager_def_function.function
    def foo2(x, var):
      # [FUNCTION DOC]
      # HELPER: compiled function wrapping foo.
      return foo(x, var)

    @eager_def_function.function
    def bar(var):
      # [FUNCTION DOC]
      # HELPER: compiled function; runs a two-step loop calling foo2 and returns the gradient with
      # respect to the variable.
      r = while_loop_tf.while_loop(
          lambda i, _: i < 2, lambda i, x: (i + 1, foo2(x, var)), [0, 0.0])[1]
      return gradients_impl.gradients(r, var)[0]

    var = resource_variable_ops.ResourceVariable([1., 1., 1., 1.])
    self.evaluate(variables.global_variables_initializer())
    grad = self.evaluate(bar(var))
    self.assertAllEqual(gradient_checker_v2._to_numpy(grad), [0., 2., 0., 2.])

  def testWhileGrad_ResourceVarInLoopInFunctionCall(self):
    # [FUNCTION DOC]
    # Test (skipped when a GPU is available): a compiled function itself contains a loop reading
    # variable rows, called from an outer loop. Expects gradient [0, 6, 6, 0].
    if test.is_gpu_available():
      self.skipTest("b/128635252")

    @eager_def_function.function
    def foo(x, var):
      # [FUNCTION DOC]
      # HELPER: compiled function; runs a three-step loop adding the sum of rows one and two of the
      # variable to y, returns the final y.
      return while_loop_tf.while_loop(
          lambda j, _: j < 3,
          lambda j, y: (j + 1,
                        y + math_ops.reduce_sum(var.sparse_read([1, 2]))),
          [0, x])[1]

    @eager_def_function.function
    def bar(var):
      # [FUNCTION DOC]
      # HELPER: compiled function; runs a two-step loop calling foo and returns the gradient with respect
      # to the variable.
      r = while_loop_tf.while_loop(
          lambda i, _: i < 2,
          lambda i, x: (i + 1, foo(x, var)),
          [0, 0.0])[1]
      return gradients_impl.gradients(r, var)[0]

    var = resource_variable_ops.ResourceVariable([1., 1., 1., 1.])
    self.evaluate(variables.global_variables_initializer())
    grad = self.evaluate(bar(var))
    self.assertAllEqual(gradient_checker_v2._to_numpy(grad), [0., 6., 6., 0.])

  def testWhileCondGrad_ResourceVarInFunctionCall(self):
    # [FUNCTION DOC]
    # Test: loop whose body picks between two compiled-function calls using different resource
    # variables by iteration parity. Expects gradients [0, 2, 0] and [0, 2].

    @eager_def_function.function
    def foo(x, var):
      # [FUNCTION DOC]
      # HELPER: compiled function; adds the first sparse row element (row one) of the variable to x.
      return x + var.sparse_read([1])[0]

    def body(i, x):
      # [FUNCTION DOC]
      # HELPER: loop body; cond on whether the counter is even choosing foo with the first or second
      # variable of the enclosing test (captured from the outer scope).
      return (i + 1, tf_cond.cond(
          math_ops.equal(i % 2, 0),
          lambda: foo(x, var1),
          lambda: foo(x, var2)))

    @eager_def_function.function
    def bar(var1, var2):
      # [FUNCTION DOC]
      # HELPER: compiled function; runs a four-step loop and returns gradients with respect to its two
      # variable arguments.
      r = while_loop_tf.while_loop(lambda i, _: i < 4, body, [0, 0.0])
      return gradients_impl.gradients(r, [var1, var2])

    var1 = resource_variable_ops.ResourceVariable([1., 2., 3.])
    var2 = resource_variable_ops.ResourceVariable([4., 5.])
    self.evaluate(variables.global_variables_initializer())
    grads = self.evaluate(bar(var1, var2))
    self.assertAllEqual(gradient_checker_v2._to_numpy(grads[0]), [0., 2., 0.])
    self.assertAllEqual(gradient_checker_v2._to_numpy(grads[1]), [0., 2.])

  @test_util.run_deprecated_v1
  def testWhileGrad_ResourceVarSparseRead(self):
    # [FUNCTION DOC]
    # Test (deprecated v1 mode): gradient is an aggregation of indexed slices and tensors; a loop
    # multiplies by the sum of two sparse rows. Expects [0, 12, 0, 12, 0].
    # NOTE(skyewm): this test is interesting because the gradient is the
    # aggregation result of IndexedSlices and Tensors.
    var = resource_variable_ops.ResourceVariable(np.ones(5),
                                                 dtype=dtypes.float32)
    r = while_loop_tf.while_loop(
        lambda i, _: i < 3, lambda i, x:
        (i + 1, x * math_ops.reduce_sum(var.sparse_read([1, 3]))),
        [0, constant_op.constant(1.0)])[1]
    grad = gradients_impl.gradients(r, var)[0]

    self.evaluate(variables.global_variables_initializer())
    grad_val = self.evaluate(grad)
    arr = gradient_checker_v2._to_numpy(grad_val)
    self.assertAllEqual(arr, [0., 12., 0., 12., 0.])

  @test_util.run_deprecated_v1
  def testWhileGrad_MultiResourceVarSparseRead(self):
    # [FUNCTION DOC]
    # Test (deprecated v1 mode): loop with several loop variables reading two resource variables
    # sparsely. Expects gradients [0, 1, 0, 1, 0] and [3, 0, 0].
    # NOTE(skyewm): this test is interesting because the gradient is the
    # aggregation result of IndexedSlices and Tensors.
    var1 = resource_variable_ops.ResourceVariable(np.ones(5),
                                                  dtype=dtypes.float32)
    var2 = resource_variable_ops.ResourceVariable(np.ones(3),
                                                  dtype=dtypes.float32)
    x1_init = constant_op.constant([0., 0.])
    x2_init = constant_op.constant(1.)
    x3_init = constant_op.constant(1.)

    def body(i, unused_x1, x2, x3):
      # [FUNCTION DOC]
      # HELPER: reads rows one and three of the first variable, doubles x2, and multiplies x3 by the
      # sparse read of the second; returns the counter plus one and the three results.
      y1 = var1.sparse_read([1, 3])
      y2 = x2 * 2
      y3 = x3 * math_ops.reduce_sum(var2.sparse_read([0]))
      return i + 1, y1, y2, y3

    r = while_loop_tf.while_loop(
        lambda i, x1, x2, x3: i < 3, body,
        [0, x1_init, x2_init, x3_init])[1:]
    var1_grad, var2_grad = gradients_impl.gradients(r, [var1, var2])

    self.evaluate(variables.global_variables_initializer())
    var1_grad_val = self.evaluate(var1_grad)
    var2_grad_val = self.evaluate(var2_grad)
    self.assertAllEqual(gradient_checker_v2._to_numpy(var1_grad_val),
                        [0., 1., 0., 1., 0.])
    self.assertAllEqual(gradient_checker_v2._to_numpy(var2_grad_val),
                        [3., 0., 0.])

  def testWhileGrad_Gather(self):
    # [FUNCTION DOC]
    # Test: loop whose body adds a gathered element, gradient of a scaled result. Expects the loop
    # output 8 and gradient [24, 0, 0, 0, 0]. Runs as a compiled function under TF2.
    # NOTE(skyewm): this test is interesting because the gather gradient
    # function returns an IndexedSlices.
    @tf_function_in_tf2
    def fn():
      # [FUNCTION DOC]
      # HELPER: builds the loop and returns the final y plus the gradient of three times y.
      x = constant_op.constant([1., 1., 1., 1., 1.])
      y = while_loop_tf.while_loop(
          lambda i, _: i < 3, lambda i, x:
          (i + 1, x + array_ops.gather(x, [0])), [0, x[:1]])[1]
      z = y * 3.0
      grad = gradients_impl.gradients(z, x)[0]
      return y, grad
    y, grad = fn()
    self.assertEqual(self.evaluate(y), 8.)
    self.assertAllEqual(self.evaluate(grad), [24., 0., 0., 0., 0.])

  def testWhileGrad_GatherNoFanOut(self):
    # [FUNCTION DOC]
    # Test: like above but the body only gathers. Expects output 1 and gradient [3, 0, 0, 0, 0].
    # NOTE(skyewm): this test is interesting because the gather gradient
    # function returns an IndexedSlices.
    @tf_function_in_tf2
    def fn():
      # [FUNCTION DOC]
      # HELPER: builds the loop and returns the final y plus the gradient of three times y.
      x = constant_op.constant([1., 1., 1., 1., 1.])
      y = while_loop_tf.while_loop(
          lambda i, _: i < 3,
          lambda i, x: (i + 1, array_ops.gather(x, [0])),
          [0, x[:1]])[1]
      z = y * 3.0
      grad = gradients_impl.gradients(z, x)[0]
      return y, grad
    y, grad = fn()
    self.assertEqual(self.evaluate(y), 1.)
    self.assertAllEqual(self.evaluate(grad), [3., 0., 0., 0., 0.])

  @test_util.run_v1_only("b/120545219")
  def testWhileGradInCond(self):
    # [FUNCTION DOC]
    # Test (v1 only): gradient of a while loop computed inside a cond branch. With feed 1.0
    # expects 9.0.

    with self.cached_session():
      n = ops.convert_to_tensor(1.0, name="n")
      x = array_ops.placeholder(dtypes.float32, shape=None)
      c = lambda n: math_ops.less(n, 10.0)
      b = lambda n: math_ops.add(n, x)

      def fn1():
        # [FUNCTION DOC]
        # HELPER: runs the while loop with unknown shape invariant and returns its gradient with respect
        # to x.
        r = while_loop_tf.while_loop(c, b, [n], [tensor_shape.unknown_shape()])
        return gradients_impl.gradients(r, x)[0]

      r = tf_cond.cond(math_ops.less(1, 2), fn1, lambda: x)
      self.assertAllClose(9.0, r.eval(feed_dict={x: 1.0}))

  @test_util.disable_control_flow_v2("b/116340060")
  @test_util.run_v1_only("b/120545219")
  def testGradInWhileWrtInitialLoopVal(self):
    # [FUNCTION DOC]
    # Test (legacy control flow, v1 only): computing a gradient inside a loop body with respect to
    # the initial value of a loop variable. Expects a value error saying this is unsupported.
    with self.cached_session():
      x = array_ops.placeholder(dtypes.float32, shape=(), name="x")
      y = x + 1

      def body(i, v):
        # [FUNCTION DOC]
        # HELPER: doubles v and returns i plus one and the gradient with respect to the outer placeholder.
        z = v * 2
        return i + 1, gradients_impl.gradients(z, x)[0]

      with self.assertRaisesRegex(
          ValueError,
          "Cannot compute gradient inside while loop with respect to op 'x'. "
          "We do not support taking the gradient wrt or through the initial "
          "value of a loop variable. Gradients can be computed through "
          "loop invariants or wrt the input parameters to the loop body."):
        while_loop_tf.while_loop(lambda i, x: i < 3, body, [0, y])

  @test_util.run_v1_only("b/120545219")
  def testWhileGradInWhile(self):
    # [FUNCTION DOC]
    # Test (v1 only): gradient of an inner loop computed inside an outer loop body. With feed 1.0
    # expects 9.0.
    with self.cached_session():
      n = ops.convert_to_tensor(1.0, name="n")
      x = array_ops.placeholder(dtypes.float32, shape=None)
      c = lambda n: math_ops.less(n, 10.0)
      b = lambda n: math_ops.add(n, x)

      def b1(n):
        # [FUNCTION DOC]
        # HELPER: outer loop body; runs the inner loop and returns its gradient with respect to x.
        r = while_loop_tf.while_loop(c, b, [n], [tensor_shape.unknown_shape()])
        return gradients_impl.gradients(r, x)

      r = while_loop_tf.while_loop(
          lambda n: n < 6.0, b1, [n], [tensor_shape.unknown_shape()])
      self.assertAllClose(9.0, r.eval(feed_dict={x: 1.0}))

  @test_util.run_v1_only("b/120545219")
  def testCondGradInNestedWhiles(self):
    # [FUNCTION DOC]
    # Test (v1 only): gradient of a cond is taken inside the inner body of two nested loops. Expects
    # counter 3 and value 1.0.

    def outer_body(i, x):
      # [FUNCTION DOC]
      # HELPER: runs the inner loop three times and returns the incremented counter and its result.
      _, x = while_loop_tf.while_loop(lambda j, x: j < 3, inner_body, [0, 0.0])
      return i + 1, x

    def inner_body(j, x):
      # [FUNCTION DOC]
      # HELPER: cond on x less than one (double or identity) and returns j plus one and the gradient
      # of that cond with respect to x.
      y = tf_cond.cond(math_ops.less(x, 1), lambda: 2 * x, lambda: x)
      return j + 1, gradients_impl.gradients(y, x)[0]

    i, x = while_loop_tf.while_loop(lambda i, x: i < 3, outer_body, [0, 0.0])

    with self.cached_session() as sess:
      i_val, x_val = self.evaluate([i, x])
      self.assertEqual(i_val, 3)
      self.assertAllClose(x_val, 1.0)

  @test_util.run_gpu_only
  def testGpuResourceAccess(self):
    # [FUNCTION DOC]
    # Test (GPU only): a resource variable on the GPU used inside a cond inside a loop in a compiled
    # function. Expects 9.0.
    with ops.device(test.gpu_device_name()):
      var = resource_variable_ops.ResourceVariable(constant_op.constant(3.0))

    @eager_def_function.function
    def foo():
      # [FUNCTION DOC]
      # HELPER: compiled function; three iterations adding the variable inside a cond on a true constant.
      return while_loop_tf.while_loop(
          lambda i, _: i < 3, lambda i, x:
          (i + 1,
           tf_cond.cond(
               constant_op.constant(True), lambda: x + var, lambda: x)),
          [0, 0.0])[1]

    self.evaluate(variables.global_variables_initializer())
    self.assertEqual(self.evaluate(foo()), 9.0)

  def testNestedResourceAccess(self):
    # [FUNCTION DOC]
    # Test: resource variable squared in an inner loop nested in a cond nested in an outer loop.
    # Expects value 81; under control flow v2 expects gradient 4.0 (v1 returns a wrong answer, as noted
    # in the comments).
    var = resource_variable_ops.ResourceVariable(constant_op.constant(3.0))

    @eager_def_function.function
    def test_fn():
      # [FUNCTION DOC]
      # HELPER: compiled function; builds the nested loops/cond and returns the result with its gradient
      # with respect to the initial value.
      x = constant_op.constant(0.0)
      r = while_loop_tf.while_loop(
          # Outer loop condition
          lambda i, y: i < 2,
          # Outer loop body
          lambda i, y: (
              i + 1,
              y + tf_cond.cond(
                  constant_op.constant(True),
                  # True branch
                  lambda: while_loop_tf.while_loop(
                      # Inner loop condition
                      lambda j, z: j < 3,
                      # Inner loop body
                      lambda j, z: (j + 1, z + math_ops.square(var)),
                      # Inner initial loop value
                      [0, y])[1],
                  # False branch
                  lambda: (0.0))),
          # Outer initial loop value
          [0, x])[1]

      grad = gradients_impl.gradients(r, x)[0]
      return r, grad

    self.evaluate(variables.global_variables_initializer())
    r, grad = self.evaluate(test_fn())
    # 2 * 3 * 3^2
    self.assertEqual(r, 81.0)
    # v1 control flow gets the wrong answer!!!
    # Gradient computation:
    #   f(x) = x + 3^2
    #   inner_loop(x) = f(f(f(x))) = x + 3*3^2 = x + 27
    #   g(x) = x + inner_loop(x) = 2x + 27
    #   outer_loop(x) = g(g(x)) = 4x + 81
    #   outer_loop'(x) = 4
    # Note that v1 control flow gets 4.0 as well if the cond is removed.
    if control_flow_util.ENABLE_CONTROL_FLOW_V2:
      self.assertEqual(grad, 4.0)

  def testWhile_NestedInput(self):
    # [FUNCTION DOC]
    # Test: loop variables are a list of a named tuple, a tuple and a tensor. Expects the result to be
    # a list preserving those types and flattened values [100, 1, 102, 3, 204].
    with self.cached_session() as sess:
      named = collections.namedtuple("named", ("a", "b"))
      loop_vars = [
          named(a=constant_op.constant(0.0), b=constant_op.constant(1.0)),
          (constant_op.constant(2.0), constant_op.constant(3.0)),
          constant_op.constant(4.0)
      ]
      c = lambda lv0, _1, _2: lv0.a < 100.0

      def b(lv0, lv1, lv2):
        # [FUNCTION DOC]
        # HELPER: loop body; increments the named tuple's first field, the tuple's first element, and
        # adds two to the tensor.
        lv0 = named(a=lv0.a + 1, b=lv0.b)
        lv1 = (lv1[0] + 1, lv1[1])
        lv2 += 2
        return [lv0, lv1, lv2]

      r = while_loop_tf.while_loop(c, b, loop_vars)

      self.assertTrue(isinstance(r, list))
      self.assertTrue(isinstance(r[0], named))
      self.assertTrue(isinstance(r[1], tuple))
      self.assertTrue(isinstance(r[2], tensor_lib.Tensor))

      r_flattened = nest.flatten(r)
      self.assertEqual([100.0, 1.0, 102.0, 3.0, 4.0 + 100 * 2.0],
                       self.evaluate(r_flattened))

  @test_util.run_v1_only("b/120545219")
  def testWhile_NestedBadArityFails(self):
    # [FUNCTION DOC]
    # Test (v1 only): body returns two values for three loop variables. Expects a value error that
    # the number of elements must be the same.
    with self.cached_session():
      named = collections.namedtuple("named", ("a", "b"))
      loop_vars = [
          named(a=constant_op.constant(0.0), b=constant_op.constant(1.0)),
          (constant_op.constant(2.0), constant_op.constant(3.0)),
          constant_op.constant(4.0)
      ]
      c = lambda lv0, _1, _2: lv0.a < 100.0

      def b(lv0, lv1, _):
        # [FUNCTION DOC]
        # HELPER: ignores its third input and returns only two values.
        return [lv0, lv1]

      with self.assertRaisesRegex(ValueError, "the same number of elements"):
        while_loop_tf.while_loop(c, b, loop_vars)

  @test_util.run_v1_only("b/120545219")
  def testWhileGrad_ys_xs(self):
    # [FUNCTION DOC]
    # Test (v1 only): gradients of loop outputs (both or one) with respect to either input. Expects 304
    # and 124 for both outputs, 295 and 120 for the first only.
    with self.cached_session():
      x = constant_op.constant(3.0, name="x")
      y = constant_op.constant(2.0, name="y")

      c = lambda x, y: math_ops.less(x, 100.0)

      def b(x, y):
        # [FUNCTION DOC]
        # HELPER: loop body; y1 is x plus y, x1 is x times y1; returns x1 and y1.
        y1 = math_ops.add(x, y)
        x1 = math_ops.multiply(x, y1)
        return x1, y1

      rx, ry = while_loop_tf.while_loop(c, b, [x, y], parallel_iterations=1)

      r = gradients_impl.gradients([rx, ry], x)
      self.assertAllClose(304.0, r[0])
      r = gradients_impl.gradients([rx, ry], y)
      self.assertAllClose(124.0, r[0])
      r = gradients_impl.gradients([rx], x)
      self.assertAllClose(295.0, r[0])
      r = gradients_impl.gradients([rx], y)
      self.assertAllClose(120.0, r[0])

  @test_util.run_deprecated_v1
  def testWhileGrad_Dependency(self):
    # [FUNCTION DOC]
    # Test (deprecated v1 mode): gradients where one output is an integer counter; expects 1024 for
    # both gradient sets.
    with self.cached_session():
      i = constant_op.constant(0, name="i")
      x = constant_op.constant(2.0, name="x")

      c = lambda i, x: math_ops.less(i, 10)

      def b(i, x):
        # [FUNCTION DOC]
        # HELPER: loop body; doubles x and increments i.
        x = math_ops.multiply(x, 2.0)
        i = math_ops.add(i, 1)
        return i, x

      ri, rx = while_loop_tf.while_loop(c, b, [i, x], parallel_iterations=1)

      r = gradients_impl.gradients([ri, rx], x)
      self.assertAllClose(1024.0, r[0])
      r = gradients_impl.gradients([rx], x)
      self.assertAllClose(1024.0, r[0])

  @test_util.run_v1_only("b/120545219")
  def testWhileGrad_NoGradient(self):
    # [FUNCTION DOC]
    # Test (v1 only): loop with back propagation disabled; its output plus the start has gradient 1.0
    # (only the direct term).
    with self.cached_session():
      v = constant_op.constant(2.0, name="v")
      c = lambda v: math_ops.less(v, 100.0)
      b = math_ops.square
      r = while_loop_tf.while_loop(c, b, [v], back_prop=False)
      r = math_ops.add(r, v)
      r = gradients_impl.gradients(r, v)
      self.assertAllClose(1.0, r[0])

  @test_util.disable_control_flow_v2("b/113324949 (RefVariable)")
  @test_util.run_v1_only("b/120545219")
  def testWhileGrad_NoDependency(self):
    # [FUNCTION DOC]
    # Test (legacy control flow, v1 only): loop returns a variable twice unchanged; gradient of the
    # sum of its last output with respect to the variable is ones of shape 2 by 3.
    with self.cached_session() as sess:
      variable = variables.Variable(array_ops.ones([2, 3]))
      duration = array_ops.zeros([], dtype=dtypes.int32)

      def cond(duration, tensor, _):
        # [FUNCTION DOC]
        # HELPER: loop condition; true while duration is below ten (deletes its unused second argument).
        del tensor
        return duration < 10

      def body(duration, tensor, _):
        # [FUNCTION DOC]
        # HELPER: loop body; increments the duration and passes the tensor through in both slots.
        return (duration + 1, tensor, tensor)

      loop_vars = [duration, variable, variable]
      tensors = while_loop_tf.while_loop(
          cond=cond, body=body, loop_vars=loop_vars)
      cost = math_ops.reduce_sum(tensors[2])
      grad = gradients_impl.gradients(cost, [variable])
      self.evaluate(variables.global_variables_initializer())
      self.assertAllClose(np.ones([2, 3]), sess.run(grad[0]))

  @test_util.run_deprecated_v1
  def testWhileGrad_Const(self):
    # [FUNCTION DOC]
    # Test (deprecated v1 mode): the loop body returns a constant unrelated to the initial value;
    # gradient with respect to the initial constant is 0.0.
    with self.cached_session() as sess:
      c0 = constant_op.constant(0.0, name="c0")
      c1 = constant_op.constant(1.0, name="c1")
      duration = constant_op.constant(0, name="t")

      def cond(duration, _):
        # [FUNCTION DOC]
        # HELPER: loop condition; true while duration is below one.
        return duration < 1

      def body(duration, _):
        # [FUNCTION DOC]
        # HELPER: loop body; increments the duration and returns a constant.
        return duration + 1, c1

      loop_vars = [duration, c0]
      tensors = while_loop_tf.while_loop(
          cond=cond, body=body, loop_vars=loop_vars)
      cost = math_ops.reduce_sum(tensors[1])
      grad = gradients_impl.gradients(cost, [c0])
      self.assertAllClose(0.0, sess.run(grad[0]))

  @test_util.run_v1_only("b/120545219")
  def testWhileGrad_SerialTwoLoops(self):
    # [FUNCTION DOC]
    # Test (v1 only): two loops in series (five doublings each); gradient 1024.
    with self.cached_session():
      i = constant_op.constant(0, name="i")
      x = constant_op.constant(2.0, name="x")

      c = lambda i, x: math_ops.less(i, 5)

      def b(i, x):
        # [FUNCTION DOC]
        # HELPER: loop body; doubles x and increments i.
        x = math_ops.multiply(x, 2.0)
        i = math_ops.add(i, 1)
        return i, x

      _, rx = while_loop_tf.while_loop(c, b, [i, x], parallel_iterations=1)
      _, rx = while_loop_tf.while_loop(c, b, [i, rx], parallel_iterations=1)

      r = gradients_impl.gradients([rx], x)
      self.assertAllClose(1024.0, r[0])

  @test_util.run_v1_only("b/120545219")
  def testWhileGrad_ParallelTwoLoops(self):
    # [FUNCTION DOC]
    # Test (v1 only): two parallel loops of five doublings added together; gradient 64.
    with self.cached_session():
      i = constant_op.constant(0, name="i")
      x = constant_op.constant(2.0, name="x")

      c = lambda i, x: math_ops.less(i, 5)

      def b(i, x):
        # [FUNCTION DOC]
        # HELPER: loop body; doubles x and increments i.
        x = math_ops.multiply(x, 2.0)
        i = math_ops.add(i, 1)
        return i, x

      _, r1 = while_loop_tf.while_loop(c, b, [i, x], parallel_iterations=1)
      _, r2 = while_loop_tf.while_loop(c, b, [i, x], parallel_iterations=1)
      rx = math_ops.add(r1, r2)

      r = gradients_impl.gradients([rx], x)
      self.assertAllClose(64.0, r[0])

  @test_util.run_v1_only("b/120545219")
  def testWhileGrad_OneOutputWithControlDependencyOnSecond(self):
    # [FUNCTION DOC]
    # Test (v1 only): identity of one loop output made under a control dependency on another; value
    # 2.0 and gradient 1.0 expected, and the gradient must not be none.
    with self.cached_session():
      i = constant_op.constant(0, name="i")
      x = constant_op.constant(1.0, name="x")
      y = constant_op.constant(1.0, name="y")
      c = lambda i, *_: math_ops.less(i, 1, name="cond_less")

      def b(i, xi, yi):
        # [FUNCTION DOC]
        # HELPER: loop body; increments the counter, passes xi through an identity and returns xi plus yi.
        # return (i + 1, xi, xi + yi)
        return (math_ops.add(i, 1, name="inc"), array_ops.identity(
            xi, name="xi"), math_ops.add(xi, yi, name="xi_plus_yi"))

      _, x_f, y_f = while_loop_tf.while_loop(c, b, [i, x, y])
      with ops.control_dependencies([x_f]):
        y_f_d = array_ops.identity(y_f, name="y_f_d")

      self.assertAllClose(2.0, self.evaluate(y_f_d))  # y_f_d = 1.0 + 1.0
      g = gradients_impl.gradients([y_f_d], [x])[0]
      self.assertTrue(g is not None)
      self.assertAllClose(1.0,
                          self.evaluate(g))  # y_f_d = x + 1.0, dy_f_d/dx = 1.0

  def _testNestedWhileGrad_Simple(self, use_gpu):
    # [FUNCTION DOC]
    # HELPER: shared body for testNestedWhileGrad_Simple; takes a use-GPU flag. Outer loop doubles the
    # inner loop result (inner loop doubles until at least four). Expects gradient 8.
    with self.cached_session(use_gpu=use_gpu):
      v = constant_op.constant(1.0)

      def inner_loop(s):
        # [FUNCTION DOC]
        # HELPER: loop doubling s while below four; returns its result.
        c = lambda x: math_ops.less(x, 4.0)
        b = lambda x: math_ops.multiply(x, 2.0)
        return while_loop_tf.while_loop(c, b, [s])

      c = lambda x: math_ops.less(x, 2.0)
      b = lambda x: math_ops.multiply(inner_loop(x), 2.0)
      r = while_loop_tf.while_loop(c, b, [v])

      r = gradients_impl.gradients(r, v)[0]
      self.assertAllClose(8.0, self.evaluate(r))

  @test_util.run_deprecated_v1
  def testNestedWhileGrad_Simple(self):
    # [FUNCTION DOC]
    # Test (deprecated v1 mode): runs the shared helper on CPU then GPU.
    self._testNestedWhileGrad_Simple(use_gpu=False)
    self._testNestedWhileGrad_Simple(use_gpu=True)

  @test_util.run_v1_only("b/120545219")
  def testNestedWhileGrad_SerialInner(self):
    # [FUNCTION DOC]
    # Test (v1 only): outer loop body runs two inner loops one after another. Expects gradient 256.
    with self.cached_session():
      v = constant_op.constant(1.0)

      def inner_loop1(s):
        # [FUNCTION DOC]
        # HELPER: inner loop of four doublings; returns counter and value.
        z = constant_op.constant(0)
        c = lambda i, x: math_ops.less(i, 4)
        b = lambda i, x: [math_ops.add(i, 1), math_ops.multiply(x, 2.0)]
        return while_loop_tf.while_loop(c, b, [z, s])

      def inner_loop2(s):
        # [FUNCTION DOC]
        # HELPER: second identical inner loop of four doublings; returns counter and value.
        z = constant_op.constant(0)
        c = lambda i, x: math_ops.less(i, 4)
        b = lambda i, x: [math_ops.add(i, 1), math_ops.multiply(x, 2.0)]
        return while_loop_tf.while_loop(c, b, [z, s])

      c = lambda x: math_ops.less(x, 128.0)
      b = lambda x: inner_loop2(inner_loop1(x)[1])[1]
      r = while_loop_tf.while_loop(c, b, [v])

      r = gradients_impl.gradients(r, v)[0]
      self.assertAllClose(256.0, self.evaluate(r))

  @test_util.run_deprecated_v1
  def testNestedWhileGrad_ParallelInner(self):
    # [FUNCTION DOC]
    # Test (deprecated v1 mode): outer loop body multiplies the results of two independent inner loops.
    # Expects gradient 512.
    with self.cached_session():
      v = constant_op.constant(1.0)

      def inner_loop1(s):
        # [FUNCTION DOC]
        # HELPER: inner loop of four doublings; returns counter and value.
        z = constant_op.constant(0)
        c = lambda i, x: math_ops.less(i, 4)
        b = lambda i, x: [math_ops.add(i, 1), math_ops.multiply(x, 2.0)]
        return while_loop_tf.while_loop(c, b, [z, s])

      def inner_loop2(s):
        # [FUNCTION DOC]
        # HELPER: second identical inner loop of four doublings; returns counter and value.
        z = constant_op.constant(0)
        c = lambda i, x: math_ops.less(i, 4)
        b = lambda i, x: [math_ops.add(i, 1), math_ops.multiply(x, 2.0)]
        return while_loop_tf.while_loop(c, b, [z, s])

      c = lambda x: math_ops.less(x, 128.0)
      b = lambda x: math_ops.multiply(inner_loop1(x)[1], inner_loop2(x)[1])
      r = while_loop_tf.while_loop(c, b, [v])

      r = gradients_impl.gradients(r, v)[0]
      self.assertAllClose(512.0, self.evaluate(r))

  @test_util.run_v1_only("b/120545219")
  def testNestedWhileGrad_ParallelIterations(self):
    # [FUNCTION DOC]
    # Test (v1 only): nested map operations with ten parallel iterations each, over a variable;
    # trains with Adam for one step. Checks the stack pushes and pops of inner loops execute in order
    # by expecting the variable to become 2.999.
    # Make sure the stack pushes and pops of an inner loop are executed in
    # the sequential order of the iterations of its outer loop.
    with self.cached_session() as sess:

      def inner_loop(t):
        # [FUNCTION DOC]
        # HELPER: maps a function adding the square of the variable over a tensor.
        fn = lambda n: n + math_ops.square(var)
        return map_fn.map_fn(fn=fn, elems=t, parallel_iterations=10)

      def outer_loop(inp):
        # [FUNCTION DOC]
        # HELPER: maps the inner loop over each row of the input.
        return map_fn.map_fn(
            fn=inner_loop, elems=inp, parallel_iterations=10)

      var = variables.Variable(constant_op.constant(3.0))
      inp = constant_op.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
      res = outer_loop(inp)
      optimizer = adam.AdamOptimizer(learning_rate=0.001)
      train_op = optimizer.minimize(math_ops.reduce_mean(math_ops.square(res)))
      self.evaluate(variables.global_variables_initializer())
      self.evaluate(train_op)
      self.assertAllClose(2.999, var.read_value())

  def _testWhileCondGrad_Simple(self, use_gpu):
    # [FUNCTION DOC]
    # HELPER: shared body for testWhileCondGrad_Simple; takes a use-GPU flag. Loop body is a cond
    # squaring or subtracting; loops until at least 100 from 2. Expects gradient 1024.
    with self.cached_session(use_gpu=use_gpu):
      v = ops.convert_to_tensor(2.0, name="v")
      n = ops.convert_to_tensor(100.0, name="n")
      one = ops.convert_to_tensor(1.0, name="one")
      c = lambda x: math_ops.less(x, n)
      # pylint: disable=undefined-variable
      # for OSS build
      b = lambda x: tf_cond.cond(constant_op.constant(True),
                                 lambda: math_ops.square(x),
                                 lambda: math_ops.subtract(x, one))
      # pylint: enable=undefined-variable
      r = while_loop_tf.while_loop(c, b, [v])
      r = gradients_impl.gradients(r, v)[0]
      self.assertAllClose(1024.0, self.evaluate(r))

  @test_util.run_deprecated_v1
  def testWhileCondGrad_Simple(self):
    # [FUNCTION DOC]
    # Test (deprecated v1 mode): runs the shared helper on CPU then GPU.
    self._testWhileCondGrad_Simple(use_gpu=False)
    self._testWhileCondGrad_Simple(use_gpu=True)

  @test_util.run_deprecated_v1
  def testWhileCondGrad_UnknownShape(self):
    # [FUNCTION DOC]
    # Test (deprecated v1 mode): same loop with a placeholder start of unknown shape fed 2.0.
    # Expects gradient 1024.
    with self.cached_session() as sess:
      v = array_ops.placeholder(dtypes.float32)
      n = ops.convert_to_tensor(100.0, name="n")
      one = ops.convert_to_tensor(1.0, name="one")
      c = lambda x: math_ops.less(x, n)
      # pylint: disable=undefined-variable
      # for OSS build
      b = lambda x: tf_cond.cond(constant_op.constant(True),
                                 lambda: math_ops.square(x),
                                 lambda: math_ops.subtract(x, one))
      # pylint: enable=undefined-variable
      r = while_loop_tf.while_loop(c, b, [v])
      r = gradients_impl.gradients(r, v)[0]
      r = sess.run(r, feed_dict={v: 2.0})
      self.assertAllClose(1024.0, r)

  @test_util.run_deprecated_v1
  def testWhileGrad_Concat(self):
    # [FUNCTION DOC]
    # Test (deprecated v1 mode): loop concatenates a variable to a growing matrix (shape invariant
    # unknown rows); minimizes the sum with gradient descent at 0.01. Expects the variable [0.98, 1.98].
    with self.cached_session() as sess:
      x = variable_scope.get_variable("x", initializer=[[1., 2.]])
      i0 = constant_op.constant(0)
      h0 = array_ops.zeros([0, 2])

      def condition(i, _):
        # [FUNCTION DOC]
        # HELPER: loop condition; true while i is below two.
        return i < 2

      def body(i, h):
        # [FUNCTION DOC]
        # HELPER: loop body; increments i and appends the variable as a new row.
        return i + 1, array_ops.concat([h, x], 0)

      _, h = while_loop_tf.while_loop(
          condition, body, [i0, h0],
          [i0.get_shape(), tensor_shape.TensorShape([None, 2])])
      s = math_ops.reduce_sum(h)

      optimizer = gradient_descent.GradientDescentOptimizer(0.01)
      op = optimizer.minimize(s)

      self.evaluate(variables.global_variables_initializer())
      self.evaluate(op)
      self.assertAllClose([[0.98000002, 1.98000002]], self.evaluate(x))

  @test_util.disable_control_flow_v2("b/113324949 (RefVariable)")
  @test_util.run_v1_only("b/120545219")
  def testWhileWithRefsWithGradients_1(self):
    # [FUNCTION DOC]
    # Test (legacy control flow, v1 only): loop carrying a reference variable with gradient seeded
    # by a reference variable. Expects counter 10, value 0 and gradient 73.
    with self.cached_session() as sess:
      x = variable_v1.VariableV1(0.)._ref()  # pylint: disable=protected-access
      i = constant_op.constant(0)
      c = lambda i, x: math_ops.less(i, 10)

      self.assertEqual(x.dtype, dtypes.float32_ref)

      def body(i, x):
        # [FUNCTION DOC]
        # HELPER: loop body; asserts x has a float reference type and returns i plus one and ref identity of x.
        self.assertEqual(x.dtype, dtypes.float32_ref)
        return [i + 1, gen_array_ops.ref_identity(x)]

      r = while_loop_tf.while_loop(c, body, [i, x], parallel_iterations=5)

      grad_ys = [variable_v1.VariableV1(73)._ref()]  # pylint: disable=protected-access
      grad = gradients_impl.gradients([r[1]], [x], grad_ys=grad_ys)

      self.evaluate(variables.global_variables_initializer())

      self.assertEqual(r[0].dtype, dtypes.int32)
      self.assertEqual(r[1].dtype, dtypes.float32_ref)

      value_i, value_x, value_x_grad = sess.run(r + grad)

    self.assertEqual(10, value_i)
    self.assertEqual(0, value_x)
    self.assertEqual(73, value_x_grad)

  @test_util.deprecated_graph_mode_only
  def testWhileGrad_IndexedSlices(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode only): loop doubling indexed slices ten times; gradient of the
    # values with respect to the starting values is [1024, 1024].
    with self.cached_session():
      values = constant_op.constant([2.0, 4.0], name="values")
      indices = constant_op.constant([0, 3], name="indices")
      shape = constant_op.constant([10], name="dense_shape")
      i = constant_op.constant(0)
      x = indexed_slices.IndexedSlices(values, indices, dense_shape=shape)

      def c(i, _):
        # [FUNCTION DOC]
        # HELPER: loop condition; true while the counter is below ten.
        return i < 10

      def b(i, x):
        # [FUNCTION DOC]
        # HELPER: increments i and doubles the indexed slices values.
        return [
            i + 1,
            indexed_slices.IndexedSlices(x.values * 2.0, x.indices,
                                         x.dense_shape)
        ]

      _, r = while_loop_tf.while_loop(c, b, [i, x])
      r = gradients_impl.gradients(r.values, values)[0]
      self.assertAllClose(np.array([1024.0, 1024.0]), self.evaluate(r))

  @test_util.deprecated_graph_mode_only
  def testWhileGrad_SparseTensor(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode only): same as above with a sparse tensor. Expects [1024, 1024].
    with self.cached_session():
      values = constant_op.constant([2.0, 4.0], name="values")
      indices = constant_op.constant(
          [[0], [3]], dtype=dtypes.int64, name="indices")
      shape = constant_op.constant([10], dtype=dtypes.int64, name="dense_shape")
      i = constant_op.constant(0)
      x = sparse_tensor.SparseTensor(indices, values, dense_shape=shape)

      def c(i, _):
        # [FUNCTION DOC]
        # HELPER: loop condition; true while the counter is below ten.
        return i < 10

      def b(i, x):
        # [FUNCTION DOC]
        # HELPER: increments i and doubles the sparse tensor values.
        return [
            i + 1,
            sparse_tensor.SparseTensor(x.indices, x.values * 2.0, x.dense_shape)
        ]

      _, r = while_loop_tf.while_loop(c, b, [i, x])
      r = gradients_impl.gradients(r.values, values)[0]
      self.assertAllClose(np.array([1024.0, 1024.0]), self.evaluate(r))

  @test_util.deprecated_graph_mode_only
  def testCallGradInLoop(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode only): a gradient is computed in each loop iteration and
    # accumulated. Expects the final accumulated value 600.
    with self.cached_session() as sess:
      i0 = constant_op.constant(0)
      params = constant_op.constant(5.0)
      params_1 = math_ops.square(params)

      def c(i, _):
        # [FUNCTION DOC]
        # HELPER: loop condition; true while the counter is below ten.
        return i < 10

      def b(i, x):
        # [FUNCTION DOC]
        # HELPER: builds a vector, scales it by params squared, adds the gradient with respect to params
        # to x and increments i.
        data = constant_op.constant([1.0, 2.0, 3.0])
        data = math_ops.multiply(data, params_1)
        x1 = x + gradients_impl.gradients(data, params)[0]
        return i + 1, x1

      output_grad = while_loop_tf.while_loop(
          c, b, [i0, constant_op.constant(0.0)])
      self.assertAllClose(600.0, self.evaluate(output_grad)[1])

  @test_util.run_deprecated_v1
  def testWhileAndTensorArray(self):
    # [FUNCTION DOC]
    # Test (deprecated v1 mode): loop body maps a multiplication by a parameter over a vector (uses
    # tensor arrays); gradient with respect to the parameter is 107520.
    with self.cached_session() as sess:
      param = constant_op.constant(2.0)
      n0 = constant_op.constant(0)
      y0 = constant_op.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name="elems")

      def c(i, _):
        # [FUNCTION DOC]
        # HELPER: loop condition; true while the counter is below ten.
        return i < 10

      def b(i, y):
        # [FUNCTION DOC]
        # HELPER: increments i and maps a multiplication by the captured parameter over y.
        return [
            i + 1,
            map_fn.map_fn(lambda x: math_ops.multiply(x, param), y)
        ]

      r = while_loop_tf.while_loop(c, b, [n0, y0], parallel_iterations=1)
      r = gradients_impl.gradients(r, param)[0]
      self.assertAllClose(107520.0, self.evaluate(r))

  @test_util.run_deprecated_v1
  def testNestedWhileAndTensorArray(self):
    # [FUNCTION DOC]
    # Test (deprecated v1 mode): nested loops write the products of row and column into a tensor array.
    # Expects the 3 by 3 multiplication table. A gradient check is commented out pending a new tensor
    # array implementation.
    n = constant_op.constant(3.0)

    def Body(row, ta):
      # [FUNCTION DOC]
      # HELPER: outer loop body; runs the inner loop over columns and returns the next row plus the
      # updated tensor array.

      def InnerBody(row, col, ta):
        # [FUNCTION DOC]
        # HELPER: inner loop body; writes row times column at a computed flat index (row and column are
        # one-based) and returns row, next column and the array.
        # Note: row and col are 1-based.
        ta = ta.write(
            math_ops.cast(n * (row - 1.) + col - 1., dtypes.int32), row * col)
        return row, col + 1., ta

      ta = while_loop_tf.while_loop(
          lambda _, col, _1: col <= n,
          InnerBody, [row, constant_op.constant(1.), ta],
          return_same_structure=False)[2]
      return row + 1., ta

    ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=9)
    ta = while_loop_tf.while_loop(
        lambda row, _: row <= n,
        Body, [constant_op.constant(1.), ta],
        return_same_structure=False)[1]

    output = array_ops.reshape(ta.stack(), [3, 3])
    self.assertAllEqual(
        self.evaluate(output), [[1., 2., 3.], [2., 4., 6.], [3., 6., 9.]])
    # TODO(b/117675481): This does not work with current TA. Enable with new TA.
    # grad = gradients_impl.gradients(output, [n])
    # self.assertEqual(self.evaluate(grad), 3.5)

  @test_util.run_deprecated_v1
  def testWhileGrad_StopGrad(self):
    # [FUNCTION DOC]
    # Test (deprecated v1 mode): checks gradients through loop outputs with stop-gradient applied at
    # different places. Expected values include 136, 32, 168, and none for stopped outputs.
    with self.cached_session():
      x = constant_op.constant(3.0, name="x")
      y = constant_op.constant(2.0, name="y")

      c = lambda x, y: math_ops.less(x, 100.0)

      def b(x, y):
        # [FUNCTION DOC]
        # HELPER: loop body; squares y and sets x to the square of x plus the new y.
        y1 = math_ops.square(y)
        x1 = math_ops.add(math_ops.square(x), y1)
        return x1, y1

      rx, ry = while_loop_tf.while_loop(c, b, [x, y])

      r = gradients_impl.gradients(rx, y)[0]
      self.assertEqual(136.0, self.evaluate(r))
      r = gradients_impl.gradients(ry, y)[0]
      self.assertEqual(32.0, self.evaluate(r))

      r = gradients_impl.gradients(array_ops.stop_gradient(rx), y)[0]
      self.assertEqual(r, None)
      r = gradients_impl.gradients(array_ops.stop_gradient(ry), y)[0]
      self.assertEqual(r, None)

      r = gradients_impl.gradients(
          array_ops.stop_gradient(math_ops.square(rx)), y)[0]
      self.assertEqual(r, None)
      r = gradients_impl.gradients(
          array_ops.stop_gradient(math_ops.add(rx, ry)), x)[0]
      self.assertEqual(r, None)
      r = gradients_impl.gradients(
          array_ops.stop_gradient(math_ops.add(rx, ry)), y)[0]
      self.assertEqual(r, None)

      r = gradients_impl.gradients(math_ops.add(rx, ry), y)[0]
      self.assertEqual(168.0, self.evaluate(r))
      r = gradients_impl.gradients(
          math_ops.add(rx, array_ops.stop_gradient(ry)), y)[0]
      self.assertEqual(136.0, self.evaluate(r))
      r = gradients_impl.gradients(
          math_ops.add(array_ops.stop_gradient(rx), ry), y)[0]
      self.assertEqual(32.0, self.evaluate(r))

  @test_util.run_deprecated_v1
  def testWhileGrad_StopGradInside(self):
    # [FUNCTION DOC]
    # Test (deprecated v1 mode): stop-gradient inside the loop body. Gradient with respect to y is 0
    # and with respect to x is 156.
    with self.cached_session():
      x = constant_op.constant(3.0, name="x")
      y = constant_op.constant(2.0, name="y")

      c = lambda x, y: math_ops.less(x, 100.0)

      def b(x, y):
        # [FUNCTION DOC]
        # HELPER: loop body; y1 is a stop-gradient of y squared and x1 is x squared plus y1.
        y1 = array_ops.stop_gradient(math_ops.square(y))
        x1 = math_ops.add(math_ops.square(x), y1)
        return x1, y1

      rx, _ = while_loop_tf.while_loop(c, b, [x, y])

      r = gradients_impl.gradients(rx, y)[0]
      self.assertAllClose(0.0, self.evaluate(r))
      r = gradients_impl.gradients(rx, x)[0]
      self.assertAllClose(156.0, self.evaluate(r))

  @test_util.run_deprecated_v1
  def testWhileGrad_StopGradInsideNoShape(self):
    # [FUNCTION DOC]
    # Test (deprecated v1 mode): same with placeholders of unknown shape; feeds vectors. Expects
    # gradients [0, 0] and [156, 400] and that no gradient op for the stopped tensor exists in the graph.
    with self.cached_session() as sess:
      x = array_ops.placeholder(dtypes.float32)
      y = array_ops.placeholder(dtypes.float32)

      c = lambda x, y: math_ops.less(math_ops.reduce_sum(x), 100.0)

      def b(x, y):
        # [FUNCTION DOC]
        # HELPER: loop body; y1 is a named stop-gradient of y squared and x1 is x squared plus y1.
        y1 = array_ops.stop_gradient(math_ops.square(y, name="stopped"))
        x1 = math_ops.add(math_ops.square(x), y1)
        return x1, y1

      rx, _ = while_loop_tf.while_loop(c, b, [x, y])

      grad_y = gradients_impl.gradients(rx, y)[0]
      grad_x = gradients_impl.gradients(rx, x)[0]
      feed_dict = {x: [3.0, 4.0], y: [2.0, 3.0]}
      self.assertAllClose([0.0, 0.0], sess.run(grad_y, feed_dict=feed_dict))
      self.assertAllClose([156.0, 400.0], sess.run(grad_x, feed_dict=feed_dict))
      name = "gradients/while/stopped_grad"
      all_ops = x.graph.get_operations()
      self.assertFalse(any(name in op.name for op in all_ops))

  @test_util.run_deprecated_v1
  def testWhileGradGradFail(self):
    # [FUNCTION DOC]
    # Test (deprecated v1 mode): gradient of a scan result with respect to a variable. Under legacy
    # control flow a second-order gradient raises a type error mentioning second-order gradients;
    # a stop-gradient on the first gradient then allows differentiating without error.
    theta = variables.Variable(initial_value=1.)

    def fn(prev, x):
      # [FUNCTION DOC]
      # HELPER: scan function; returns the previous value plus x times the captured variable.
      return prev + x * theta

    result = functional_ops.scan(fn, np.array([1., 2., 3.], dtype=np.float32))
    grad_theta = gradients_impl.gradients(result, theta)
    if not control_flow_util.ENABLE_CONTROL_FLOW_V2:
      with self.assertRaisesRegex(TypeError, "Second-order gradient"):
        gradients_impl.gradients(grad_theta, theta)
    grad_theta_stopped = array_ops.stop_gradient(grad_theta)
    gradients_impl.gradients(grad_theta_stopped, theta)

  @test_util.run_deprecated_v1
  def testStopGradOnWhileGrad(self):
    # [FUNCTION DOC]
    # Test (deprecated v1 mode): takes the gradient of a loop output with respect to y, applies a
    # stop-gradient to it, and combines it with other terms. Expects the final gradient 388.
    with self.cached_session():
      x = constant_op.constant(2.0, name="x")
      y = constant_op.constant(2.0, name="y")

      c = lambda x: math_ops.less(x, 100.0)
      b = lambda x: math_ops.multiply(x, y)
      rx = while_loop_tf.while_loop(c, b, [x])

      rg = gradients_impl.gradients(rx, y)[0]
      rg = array_ops.stop_gradient(rg)
      r = math_ops.add(math_ops.square(y), rx)
      r = math_ops.add(r, rg)
      r = gradients_impl.gradients(r, y)[0]
      self.assertEqual(388.0, self.evaluate(r))

  @test_util.disable_control_flow_v2("b/113324949 (RefVariable)")
  @test_util.run_deprecated_v1
  def testWhileGradientWithNontrainablePath1(self):
    # [FUNCTION DOC]
    # Test (legacy control flow, deprecated v1 mode): the loop condition is always false and the loop
    # variable is an integer argmin of a variable, so the path to the variable is non-differentiable.
    # Expects the gradient with respect to the variable to be non-null and [0, 0].
    q = variables.Variable([7., 8.])

    def cond(_, y):
      # [FUNCTION DOC]
      # HELPER: loop condition that ignores its inputs (deletes its second argument) and returns false.
      del y
      return False

    def body(x, _):
      # [FUNCTION DOC]
      # HELPER: loop body returning x and the cast of x plus the sum of the captured variable.
      return x, math_ops.cast(x, dtypes.float32) + math_ops.reduce_sum(q)

    _, y = while_loop_tf.while_loop(cond, body, (math_ops.argmin(q), 0.))
    dy_dq, = gradients_impl.gradients(y, q)
    self.assertIsNotNone(dy_dq)
    with self.cached_session() as sess:
      self.evaluate(q.initializer)
      self.assertAllClose([0., 0.], self.evaluate(dy_dq))

  @test_util.disable_control_flow_v2("b/113324949 (RefVariable)")
  @test_util.run_v1_only("b/120545219")
  def testWhileGradientWithNontrainablePath2(self):
    # [FUNCTION DOC]
    # Test (legacy control flow, v1 only): the loop runs once through an integer non-trainable path.
    # Expects the gradient with respect to the variable to be non-null and [1, 1].
    q = variables.Variable([7., 8.])

    def cond(_, y):
      # [FUNCTION DOC]
      # HELPER: loop condition; true while y equals zero.
      return math_ops.equal(y, 0.)

    def body(x, _):
      # [FUNCTION DOC]
      # HELPER: loop body returning integer zero and the cast of x plus the sum of the captured variable.
      zero = constant_op.constant(0, dtype=dtypes.int64)
      return zero, math_ops.cast(x, dtypes.float32) + math_ops.reduce_sum(q)

    _, y = while_loop_tf.while_loop(cond, body, (math_ops.argmin(q), 0.))
    dy_dq, = gradients_impl.gradients(y, q)
    self.assertIsNotNone(dy_dq)
    with self.cached_session() as sess:
      self.evaluate(q.initializer)
      self.assertAllClose([1., 1.], self.evaluate(dy_dq))

  @test_util.run_v1_only("b/120545219")
  def testIssue16504(self):
    # [FUNCTION DOC]
    # Test (v1 only): regression test for issue 16504. A loop with a gradient computed inside the
    # body (exponentiated-gradient style update) followed by an outer gradient of the result with
    # respect to a constant. Only checks that the gradient is not none.
    c = constant_op.constant(np.arange(100), dtype=dtypes.float32)
    w = variables.Variable(
        initial_value=np.ones(100), dtype=dtypes.float32) / 100
    k = variables.Variable(0, dtype=dtypes.int32)
    chg_w = constant_op.constant(np.inf, dtype=dtypes.float32)

    def cond(k, _, chg_w):
      # [FUNCTION DOC]
      # HELPER: loop condition; true while the counter is below ten and the weight change exceeds 0.001.
      return math_ops.logical_and(k < 10, chg_w > 1e-3)

    def body(k, w, chg_w):
      # [FUNCTION DOC]
      # HELPER: computes the gradient of the negative weighted sum with respect to w, applies a
      # multiplicative update, normalizes it, computes the relative change and returns counter plus one,
      # the new weights and the change.
      grad, = gradients_impl.gradients(-math_ops.reduce_sum(w * c), w)
      w_n = w * math_ops.exp(-0.1 * grad)
      w_n /= math_ops.reduce_sum(w_n)
      chg_w = (
          math_ops.reduce_sum(math_ops.abs(w_n - w)) / math_ops.reduce_sum(
              math_ops.abs(w)))
      return k + 1, w_n, chg_w

    _, w, _ = while_loop_tf.while_loop(cond, body, [k, w, chg_w])
    grad, = gradients_impl.gradients(w, c)
    self.assertIsNotNone(grad)

  @test_util.run_v1_only("b/120545219")
  def testStopGradMultiFlows(self):
    # [FUNCTION DOC]
    # Test (v1 only): loop body creates a variable and multiplies by it; the gradient norm goes through
    # a stop-gradient in a later expression. Expects the final gradient of 5.0.
    with self.cached_session():

      def body(i, y, r):
        # [FUNCTION DOC]
        # HELPER: gets or creates a scalar variable of ones, scales y by it, returns the incremented
        # counter, y and the running sum.
        x = variable_scope.get_variable(
            "x",
            shape=(),
            dtype=dtypes.float32,
            initializer=init_ops.ones_initializer())
        y *= x
        return [i + 1, y, r + math_ops.reduce_sum(y)]

      i0 = constant_op.constant(0)
      y0 = array_ops.ones(5)
      r0 = constant_op.constant(0.0)
      cond = lambda i, y, r: i < 1
      _, _, r = while_loop_tf.while_loop(
          cond, body, [i0, y0, r0], back_prop=True)

      vars_ = variables.global_variables()
      grads = linalg_ops.norm(gradients_impl.gradients(r, vars_)[0])
      z = math_ops.add(r, array_ops.stop_gradient(math_ops.reduce_sum(grads)))
      result = gradients_impl.gradients(z, vars_)[0]
      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(5.0, self.evaluate(result))

  @test_util.run_v1_only("b/120545219")
  def testOneValueCond(self):
    # [FUNCTION DOC]
    # Test (v1 only): cond choosing between two constants on a fed integer. Expects the tensor result
    # 1 when the input is 2 and 2 when the input is 0.

    with self.cached_session():
      c = array_ops.placeholder(dtypes.int32, shape=[])
      one = ops.convert_to_tensor(1, name="one")
      two = ops.convert_to_tensor(2, name="two")
      p = math_ops.greater_equal(c, 1)
      i = tf_cond.cond(p, lambda: one, lambda: two)
      self.assertTrue(isinstance(i, tensor_lib.Tensor))

      # True case: c = 2 is >= 1
      self.assertEqual([1], i.eval(feed_dict={c: 2}))

      # False case: c = 0 is not >= 1
      self.assertEqual([2], i.eval(feed_dict={c: 0}))

  @test_util.run_deprecated_v1
  def testExampleCond(self):
    # [FUNCTION DOC]
    # Test (deprecated v1 mode): example from the cond documentation; chooses between L2 and L1 norm
    # based on a fed dimension. Expects 4.0 for one and 2 times square root of two for two.

    with self.cached_session():
      x = ops.convert_to_tensor([-2.0, 2.0], name="x")
      d = array_ops.placeholder(dtypes.int32, shape=[])

      def l2():
        # [FUNCTION DOC]
        # HELPER: returns the L2 norm of the captured vector.
        return math_ops.sqrt(math_ops.reduce_sum(math_ops.square(x)))

      def l1():
        # [FUNCTION DOC]
        # HELPER: returns the L1 norm of the captured vector.
        return math_ops.reduce_sum(math_ops.abs(x))

      i = tf_cond.cond(math_ops.equal(d, 2), l2, l1)
      self.assertAllClose(4.0, i.eval(feed_dict={d: 1}))
      self.assertAllClose(2.0 * math.sqrt(2), i.eval(feed_dict={d: 2}))

  @test_util.run_v1_only("b/120545219")
  def testCase(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the list that records whether each branch ran is created and filled by the helper
    # but never asserted on; the comment says branches run exactly once but the check is missing.
    # Test (v1 only): exercises the case operation: dictionary and list of predicate-function pairs,
    # exclusive mode, duplicate predicates (first selected; error at run time if exclusive), default
    # used when no predicate holds. Expects 17, 23, 17, an op error, -1 and finally 0.
    with self.cached_session():
      x = constant_op.constant(1)
      y = constant_op.constant(2)
      z = constant_op.constant(3)
      f1 = lambda: constant_op.constant(17)
      f2 = lambda: constant_op.constant(23)
      f3 = lambda: constant_op.constant(-1)

      r1 = control_flow_case.case({
          x < y: f1,
          x > z: f2
      },
                                  default=f3,
                                  exclusive=True)
      self.assertAllEqual(r1, 17)

      r2 = control_flow_case.case([(y > z, f1), (y > x, f2)], default=f3)
      self.assertAllEqual(r2, 23)

      # Duplicate events can happen, first one is selected
      r3 = control_flow_case.case([(x < y, f1), (x < y, f2)], default=f3)
      self.assertAllEqual(r3, 17)

      # Duplicate events cause an error if exclusive = True
      r4 = control_flow_case.case([(x < y, f1), (x < y, f2)],
                                  default=f3,
                                  exclusive=True)
      with self.assertRaisesOpError("Input error:"):
        self.evaluate(r4)

      # Check that the default is called if none of the others are
      r5 = control_flow_case.case({x > y: f1}, default=f3)
      self.assertAllEqual(r5, -1)

      ran_once = [False, False, False]

      def break_run_twice(ix):
        # [FUNCTION DOC]
        # HELPER: given an index, returns a branch function that records it ran in the shared list and
        # returns that index as a constant.

        def _break():
          # [FUNCTION DOC]
          # HELPER: marks the index as run in the shared list and returns the index as a constant tensor.
          ran_once[ix] = True
          return constant_op.constant(ix)

        return _break

      # Should not fail - each conditional gets called exactly once
      # except default.  Default gets called twice: once to create an
      # empty output and once for the actual cond switch.
      r6 = control_flow_case.case([(x < y, break_run_twice(0)),
                                   (x > y, break_run_twice(1))],
                                  default=lambda: constant_op.constant(2))

      self.assertAllEqual(r6, 0)

  @test_util.run_v1_only("b/120545219")
  def testCaseSideEffects(self):
    # [FUNCTION DOC]
    # Test (v1 only): case branches assign to separate variables. For three different predicate
    # setups, running the result must only perform the assignment of the selected branch.
    # Expects results 2, 1, 0 and variable states [-1,-1,2], [-1,1,-1], [0,-1,-1].
    with self.cached_session() as sess:
      v0 = variables.Variable(-1)
      v1 = variables.Variable(-1)
      v2 = variables.Variable(-1)

      a = lambda: control_flow_ops.with_dependencies([state_ops.assign(v0, 0)], 0)
      b = lambda: control_flow_ops.with_dependencies([state_ops.assign(v1, 1)], 1)
      c = lambda: control_flow_ops.with_dependencies([state_ops.assign(v2, 2)], 2)

      x = constant_op.constant(1)
      y = constant_op.constant(2)

      r0 = control_flow_case.case(((x < y, a), (x > y, b)),
                                  default=c,
                                  exclusive=True)
      r1 = control_flow_case.case(((x > y, a), (x < y, b)),
                                  default=c,
                                  exclusive=True)
      r2 = control_flow_case.case(((x > y, a), (x > y, b)),
                                  default=c,
                                  exclusive=True)

      self.evaluate(variables.global_variables_initializer())
      self.assertAllEqual(self.evaluate([v0, v1, v2]), [-1] * 3)
      self.assertEqual(2, self.evaluate(r2))
      self.assertAllEqual(self.evaluate([v0, v1, v2]), [-1, -1, 2])

      self.evaluate(variables.global_variables_initializer())
      self.assertAllEqual(self.evaluate([v0, v1, v2]), [-1] * 3)
      self.assertEqual(1, self.evaluate(r1))
      self.assertAllEqual(self.evaluate([v0, v1, v2]), [-1, 1, -1])

      self.evaluate(variables.global_variables_initializer())
      self.assertAllEqual(self.evaluate([v0, v1, v2]), [-1] * 3)
      self.assertEqual(0, self.evaluate(r0))
      self.assertAllEqual(self.evaluate([v0, v1, v2]), [0, -1, -1])

  @test_util.disable_control_flow_v2("b/113324949 (ref vars)")
  @test_util.run_v1_only("b/120545219")
  def testOneOpCond(self):
    # [FUNCTION DOC]
    # Test (legacy control flow, v1 only): cond whose branches are single assignment ops. Feeding the
    # predicate input selects which assignment runs: 1 for the true branch and 2 for the false.
    with self.cached_session():
      v = variables.Variable(0)
      c = ops.convert_to_tensor(0)
      one = ops.convert_to_tensor(1)
      two = ops.convert_to_tensor(2)
      p = math_ops.greater_equal(c, 1)

      def a():
        # [FUNCTION DOC]
        # HELPER: true branch; assigns one to the variable and returns the result.
        return state_ops.assign(v, one)

      def b():
        # [FUNCTION DOC]
        # HELPER: false branch; assigns two to the variable and returns the result.
        return state_ops.assign(v, two)

      i = tf_cond.cond(p, a, b)
      self.assertTrue(isinstance(i, tensor_lib.Tensor))
      self.evaluate(variables.global_variables_initializer())

      self.assertEqual(0, self.evaluate(v))

      # True case: c = 2 is >= 1, v is set to 1.
      self.assertEqual(1, i.eval(feed_dict={c.name: 2}))
      self.assertEqual(1, self.evaluate(v))

      # False case: c = 0 is not >= 1, v is set to 2.
      self.assertEqual(2, i.eval(feed_dict={c.name: 0}))
      self.assertEqual(2, self.evaluate(v))

  @test_util.run_v1_only("b/120545219")
  def testWithOpsDependencies(self):
    # [FUNCTION DOC]
    # Test (v1 only): fetching an uninitialized variable fails with an op error; wrapping its
    # reference with a dependency on its initializer makes fetching succeed. Checks values 10 and 0.0.
    with self.cached_session() as sess:
      v = variable_v1.VariableV1(0.0)
      c = constant_op.constant(10)

      # Fetching v directly will result in an uninitialized error
      with self.assertRaisesOpError("Attempting to use uninitialized value"):
        self.evaluate([c, v])

      # Use a control dependency to ensure init_variable is run
      # while asking for c
      real_v = control_flow_ops.with_dependencies(
          name="real_tensor",
          output_tensor=v._ref(),  # pylint: disable=protected-access
          dependencies=[v.initializer])
      c_val, real_v_val = self.evaluate([c, real_v])

    # Ensure the result of 'real_c' is the same as 'c'
    self.assertAllEqual(10, c_val)

    # Ensure that 'v' is initialized
    self.assertAllClose(0.0, real_v_val)

  @test_util.run_v1_only("b/120545219")
  def testWithTensorDependencies(self):
    # [FUNCTION DOC]
    # Test (v1 only): chained dependencies where one tensor depends on a variable initializer and
    # another depends on that tensor. Fetching the last initializes the variable. Expects 20 and 0.0.
    with self.cached_session():
      v = variable_v1.VariableV1(0.0)
      c1 = constant_op.constant(10)
      c2 = constant_op.constant(20)

      # c1_with_init_v depends on the init op for v
      c1_with_init_v = control_flow_ops.with_dependencies(
          name="c1_with_init_v", output_tensor=c1, dependencies=[v.initializer])
      # c2_with_c1 depends on the value of c1_with_init_v
      c2_with_c1_dep = control_flow_ops.with_dependencies(
          name="c2_with_c1_dep",
          output_tensor=c2,
          dependencies=[c1_with_init_v])

      # Fetching v directly will result in an uninitialized error
      with self.assertRaisesOpError("Attempting to use uninitialized value"):
        self.evaluate(v)

      # Get the value of 'c2_with_c1_dep', which should cause 'v'
      # to be initialized.
      self.assertAllEqual(20, self.evaluate(c2_with_c1_dep))

      # Ensure that 'v' is initialized
      self.assertAllClose(0.0, self.evaluate(v))

  @test_util.run_v1_only("b/120545219")
  def testWithIndexedSlicesDependencies(self):
    # [FUNCTION DOC]
    # Test (v1 only): wraps an indexed slices with a dependency on a variable initializer. Without it
    # fetching fails (uninitialized); with it gather returns [[10, 11]] and the variable is initialized.
    with self.cached_session():
      v = variable_v1.VariableV1(
          np.array([[0.0, 1.0], [10.0, 11.0], [20.0, 21.0]]).astype(np.float32))
      v_at_1 = indexed_slices.IndexedSlices(v, constant_op.constant([1]))
      gather_v_at_1 = array_ops.gather(v_at_1.values, v_at_1.indices)
      v_at_1_after_init = control_flow_ops.with_dependencies([v.initializer],
                                                             v_at_1)
      gather_v_at_1_after_init = array_ops.gather(v_at_1_after_init.values,
                                                  v_at_1_after_init.indices)

      # Fetching gather_v_at_1 will result in an uninitialized error
      with self.assertRaisesOpError("Attempting to use uninitialized value"):
        self.evaluate(gather_v_at_1)

      # Getting gather_v_at_1_after_init will work, and initialize v.
      self.assertAllEqual([[10.0, 11.0]],
                          self.evaluate(gather_v_at_1_after_init))

      # Double check that 'v' is initialized
      self.assertAllClose([[0.0, 1.0], [10.0, 11.0], [20.0, 21.0]],
                          self.evaluate(v))

  def testDependenciesDevice(self):
    # [FUNCTION DOC]
    # Test: checks which device a with-dependencies output gets. If the tensor is on a job device
    # the dependency output is on that device; with none it has no device; under a different default
    # device it has an empty device but keeps the colocation constraint with the variable.
    with ops.Graph().as_default():
      # device set on tensor => same device on dep.
      with ops.device("/job:ps"):
        vd = variable_v1.VariableV1([0.0])
      with_vd_dep = control_flow_ops.with_dependencies([vd.initializer], vd)
      self.assertTrue("/job:ps" in with_vd_dep.device)

      # No device set on tensor => no device on dep.
      vnod = variable_v1.VariableV1([0.0])
      with_vnod_dep = control_flow_ops.with_dependencies([vnod.initializer],
                                                         vnod)
      self.assertDeviceEqual(None, with_vnod_dep.device)

      # device set on tensor, default device on graph => default device on dep.
      vdef = variable_v1.VariableV1([0.0], name="vdef")
      with ops.device("/job:worker/device:GPU:1"):
        with_vdef_dep = control_flow_ops.with_dependencies([vdef.initializer],
                                                           vdef)
        # The device is empty, but the colocation constraint is set.
        self.assertDeviceEqual("", with_vdef_dep.device)
        self.assertEqual([b"loc:@vdef"], with_vdef_dep.op.colocation_groups())

  @test_util.run_v1_only("b/120545219")
  def testGroup(self):
    # [FUNCTION DOC]
    # Test (v1 only): groups two initializers; fetching an uninitialized variable fails, running the
    # group initializes both. Expects values [0.0] and [1.0].
    with self.cached_session() as sess:
      v1 = variable_v1.VariableV1([0.0])
      v2 = variable_v1.VariableV1([1.0])

      # Group init1 and init2 and run.
      init = control_flow_ops.group(v1.initializer, v2.initializer)
      # Fetching v1 directly will result in an uninitialized error
      with self.assertRaisesOpError("Attempting to use uninitialized value"):
        self.evaluate(v1)

      # Runs "init" before fetching v1 and v2.
      init.run()
      v1_val, v2_val = self.evaluate([v1, v2])

    # Ensure that v1 and v2 are initialized
    self.assertAllClose([0.0], v1_val)
    self.assertAllClose([1.0], v2_val)

  @test_util.run_v1_only("b/120545219")
  def testGroupEmpty(self):
    # [FUNCTION DOC]
    # Test (v1 only): grouping nothing creates a no-op with no control inputs.
    op = control_flow_ops.group()
    self.assertEqual(op.type, "NoOp")
    self.assertEqual(op.control_inputs, [])

  @test_util.run_deprecated_v1
  def testMergeShapes(self):
    # [FUNCTION DOC]
    # Test (deprecated v1 mode): static shape inference of merge across inputs: unknown inputs and
    # different ranks give unknown rank; different dimensions become unknown per dimension; equal
    # shapes stay known. The index output is always a scalar shape.
    # All inputs unknown.
    p1 = array_ops.placeholder(dtypes.float32)
    p2 = array_ops.placeholder(dtypes.float32)
    p3 = array_ops.placeholder(dtypes.float32)
    m, index = control_flow_ops.merge([p1, p2, p3])
    self.assertIs(None, m.get_shape().ndims)
    self.assertEqual([], index.get_shape())

    # All inputs known with different ranks.
    p1 = array_ops.placeholder(dtypes.float32, shape=[1, 2])
    p2 = array_ops.placeholder(dtypes.float32, shape=[1, 2, 3])
    m, index = control_flow_ops.merge([p1, p2])
    self.assertIs(None, m.get_shape().ndims)
    self.assertEqual([], index.get_shape())

    # All inputs known with some dimensions different.
    p1 = array_ops.placeholder(dtypes.float32, shape=[1, 2])
    p2 = array_ops.placeholder(dtypes.float32, shape=[2, 1])
    m, index = control_flow_ops.merge([p1, p2])
    self.assertEqual([None, None], m.get_shape().as_list())
    self.assertEqual([], index.get_shape())

    p1 = array_ops.placeholder(dtypes.float32, shape=[1, 2])
    p2 = array_ops.placeholder(dtypes.float32, shape=[None, 2])
    m, index = control_flow_ops.merge([p1, p2])
    self.assertEqual([None, 2], m.get_shape().as_list())
    self.assertEqual([], index.get_shape())

    p1 = array_ops.placeholder(dtypes.float32, shape=[1, 2])
    p2 = array_ops.placeholder(dtypes.float32, shape=[2, 2])
    m, index = control_flow_ops.merge([p1, p2])
    self.assertEqual([None, 2], m.get_shape().as_list())
    self.assertEqual([], index.get_shape())

    # All inputs known with same dimensions.
    p1 = array_ops.placeholder(dtypes.float32, shape=[1, 2])
    p2 = array_ops.placeholder(dtypes.float32, shape=[1, 2])
    m, index = control_flow_ops.merge([p1, p2])
    self.assertEqual([1, 2], m.get_shape().as_list())
    self.assertEqual([], index.get_shape())

    p1 = array_ops.placeholder(dtypes.float32, shape=[None, 2])
    p2 = array_ops.placeholder(dtypes.float32, shape=[None, 2])
    m, index = control_flow_ops.merge([p1, p2])
    self.assertEqual([None, 2], m.get_shape().as_list())
    self.assertEqual([], index.get_shape())

    p1 = array_ops.placeholder(dtypes.float32, shape=[None, None])
    p2 = array_ops.placeholder(dtypes.float32, shape=[None, None])
    m, index = control_flow_ops.merge([p1, p2])
    self.assertEqual([None, None], m.get_shape().as_list())
    self.assertEqual([], index.get_shape())

  @test_util.run_v1_only("b/120545219")
  def testRefSelect(self):
    # [FUNCTION DOC]
    # Test (v1 only): static shape inference of ref_select over reference variables: unknown shape,
    # different shapes (unknown rank), identical shapes (kept), and possibly-equal shapes (unknown).
    index = array_ops.placeholder(dtypes.int32)

    # All inputs unknown.
    p1 = array_ops.placeholder(dtypes.float32)
    p2 = array_ops.placeholder(dtypes.float32)
    p3 = array_ops.placeholder(dtypes.float32)
    v1 = variable_v1.VariableV1(p1, validate_shape=False)
    v2 = variable_v1.VariableV1(p2, validate_shape=False)
    v3 = variable_v1.VariableV1(p3, validate_shape=False)
    self.assertIs(None, v1.get_shape().ndims)
    s = control_flow_ops.ref_select(index, [v1, v2, v3])
    self.assertIs(None, s.get_shape().ndims)

    # All inputs known but different.
    v1 = variable_v1.VariableV1([[1, 2]])
    v2 = variable_v1.VariableV1([[2], [1]])
    s = control_flow_ops.ref_select(index, [v1, v2])
    self.assertIs(None, s.get_shape().ndims)

    # All inputs known and same.
    v1 = variable_v1.VariableV1([[1, 2]])
    v2 = variable_v1.VariableV1([[1, 2]])
    s = control_flow_ops.ref_select(index, [v1, v2])
    self.assertEqual([1, 2], s.get_shape())

    # Possibly the same but not guaranteed.
    v1 = variable_v1.VariableV1([[1., 2.]])
    p2 = array_ops.placeholder(dtypes.float32, shape=[None, 2])
    v2 = variable_v1.VariableV1(p2, validate_shape=False)
    s = control_flow_ops.ref_select(index, [v1, v2])
    self.assertEqual(None, s.get_shape())

  @test_util.run_deprecated_v1
  def testRunLoopTensor(self):
    # [FUNCTION DOC]
    # Test (deprecated v1 mode): a tensor created inside a loop body is saved outside. Loop returns
    # 10; running the saved tensor directly must raise a value error.
    with self.cached_session() as sess:
      tensor_list = []

      def condition(t):
        # [FUNCTION DOC]
        # HELPER: loop condition; true while t is below five.
        return t < constant_op.constant(5)

      def body(_):
        # [FUNCTION DOC]
        # HELPER: loop body; appends a constant five to the outer list (leaking it) and returns ten.
        tensor_list.append(constant_op.constant(5))
        return constant_op.constant(10)

      result = while_loop_tf.while_loop(
          condition, body, [constant_op.constant(4)])
      self.assertEqual(10, self.evaluate(result))

      # Ensure that we cannot run a tensor that escapes the loop body
      # accidentally.
      with self.assertRaises(ValueError):
        sess.run(tensor_list[0])

  @test_util.run_v1_only("b/120545219")
  def testWhilePyFuncBasic(self):
    # [FUNCTION DOC]
    # Test (v1 only): loop body calls a Python function (squaring) through a python-function op,
    # four times from 2.0. Expects 65536.

    def func(x):
      # [FUNCTION DOC]
      # HELPER: Python function returning the NumPy square of its input.
      return np.square(x)

    with self.cached_session():
      r = while_loop_tf.while_loop(
          lambda i, v: i < 4, lambda i, v:
          [i + 1, script_ops.py_func(func, [v], [dtypes.float32])[0]],
          [constant_op.constant(0),
           constant_op.constant(2.0, dtypes.float32)],
          [tensor_shape.unknown_shape(),
           tensor_shape.unknown_shape()])
      self.assertEqual(self.evaluate(r[1]), 65536.0)

  @test_util.run_v1_only("b/120545219")
  def testWhileFuncBasic(self):
    # [FUNCTION DOC]
    # Test (v1 only): loop body calls a defined graph function (fourth power) twice from 2.0. Expects
    # 65536 and gradient 524288; with legacy control flow expects exactly one stack op in the graph.

    @function.Defun(dtypes.float32)
    def func(x):
      # [FUNCTION DOC]
      # HELPER: graph function over float32 returning the fourth power of its input.
      return math_ops.square(math_ops.square(x))

    with self.cached_session():
      x = constant_op.constant(2.0, dtypes.float32)
      r = while_loop_tf.while_loop(
          lambda i, v: i < 2, lambda i, v: [i + 1, func(v)],
          [constant_op.constant(0), x],
          [tensor_shape.unknown_shape(),
           tensor_shape.unknown_shape()])
      grad = gradients_impl.gradients(r, x)[0]
      self.assertEqual(self.evaluate(r[1]), 65536.0)
      self.assertEqual(self.evaluate(grad), 524288.0)
      # while_v2 does not have stacks.
      if not control_flow_util.ENABLE_CONTROL_FLOW_V2:
        self.assertEqual(
            len([op for op in x.graph.get_operations() if op.type == "StackV2"
                ]), 1)


  @test_util.run_v1_only("b/120545219")
  def testQIntSwitchMerge(self):
    # [FUNCTION DOC]
    # Test (v1 only): switches and merges a quantized int8 tensor; just must evaluate without error.
    with self.cached_session(force_gpu=test.is_gpu_available()) as sess:
      constant_qint = constant_op.constant(np.array([42]), dtypes.qint8)
      cond = constant_op.constant(True, dtypes.bool)
      v_f, v_t = control_flow_ops.switch(constant_qint, cond)
      result = control_flow_ops.merge([v_f, v_t])
      self.evaluate(result)

  @test_util.run_v1_only("b/120545219")
  def testQIntRefSwitchMerge(self):
    # [FUNCTION DOC]
    # Test (v1 only): reference variable of quantized int8 type assigned 42, then ref_switch and
    # ref_merge; must evaluate without error.
    with self.cached_session(use_gpu=test.is_gpu_available()) as sess:
      var_qint = gen_state_ops.variable(
          shape=[1], dtype=dtypes.qint8, name="v", container="", shared_name="")
      assign_op = state_ops.assign(
          var_qint, constant_op.constant(np.array([42]), dtypes.qint8))
      self.evaluate(assign_op)

      cond = constant_op.constant(True, dtypes.bool)
      v_f, v_t = control_flow_ops.ref_switch(var_qint, cond)
      result = control_flow_ops.ref_merge([v_f, v_t])
      self.evaluate(result)

  @test_util.run_v1_only("b/120545219")
  def testUInt64SwitchMerge(self):
    # [FUNCTION DOC]
    # Test (v1 only): switches and merges an unsigned 64-bit tensor; must evaluate without error.
    with self.cached_session(force_gpu=test.is_gpu_available()) as sess:
      constant_uint64 = constant_op.constant(np.array([42]), dtypes.uint64)
      cond = constant_op.constant(True, dtypes.bool)
      v_f, v_t = control_flow_ops.switch(constant_uint64, cond)
      result = control_flow_ops.merge([v_f, v_t])
      self.evaluate(result)

  def testSwitchEagerMode(self):
    # [FUNCTION DOC]
    # Test: only in eager mode (otherwise returns immediately). Switching a Python list on false
    # gives the list as the false output and an empty list as the true output.
    if not context.executing_eagerly():
      return
    input_data = [1, 2, 3, 4]
    vf, vt = control_flow_ops.switch(input_data, False)
    self.assertAllEqual(vf, input_data)
    self.assertAllEqual(vt, [])

  @test_util.run_deprecated_v1
  def testQIntArgAndRet(self):
    # [FUNCTION DOC]
    # Test (deprecated v1 mode): a defined function taking and returning a quantized int8 value
    # must run without error.

    @function.Defun(dtypes.qint8)
    def func(x):
      # [FUNCTION DOC]
      # HELPER: graph function over quantized int8 that returns its input unchanged.
      return x

    with self.cached_session(force_gpu=test.is_gpu_available()) as sess:
      qint = constant_op.constant(np.array([42]), dtypes.qint8)
      result = func(qint)
      self.evaluate(result)

  def testSparseIdentity(self):
    # [FUNCTION DOC]
    # Test: the internal identity applied to a sparse tensor (string values) preserves indices,
    # values and dense shape.
    st1 = sparse_tensor.SparseTensor([[0, 5]], ['x'], [10, 10])
    st2 = control_flow_ops._Identity(st1)
    self.assertAllEqual(st1.indices, st2.indices)
    self.assertAllEqual(st1.values, st2.values)
    self.assertAllEqual(st1.dense_shape, st2.dense_shape)

  def testSparseEnterExit(self):
    # [FUNCTION DOC]
    # Test: entering a sparse tensor into a frame and exiting it preserves indices, values and dense
    # shape.
    st1 = sparse_tensor.SparseTensor([[0, 5]], ['x'], [10, 10])
    st2 = control_flow_ops._Enter(st1, "foo_1")
    st3 = control_flow_ops.exit(st2)
    self.assertAllEqual(st1.indices, st3.indices)
    self.assertAllEqual(st1.values, st3.values)
    self.assertAllEqual(st1.dense_shape, st3.dense_shape)

  def _buildWhileWithShapeInvariants(self, shape_invariants):
    # [FUNCTION DOC]
    # HELPER: builds a while loop over a two-element vector whose body returns a one-element vector,
    # using the given list of shape invariants. Returns the loop output. The condition is always false.
    r = constant_op.constant([1, 2])

    def cond(_):
      # [FUNCTION DOC]
      # HELPER: loop condition ignoring its input; always false.
      return False

    def body(_):
      # [FUNCTION DOC]
      # HELPER: loop body ignoring its input; returns a one-element constant.
      return constant_op.constant([1])

    return while_loop_tf.while_loop(
        cond, body, [r], shape_invariants=shape_invariants)

  def testWhileOutputShapeWithShapeInvariantsUnknownRank(self):
    # [FUNCTION DOC]
    # Test: in a compiled function, a loop with an unknown-rank shape invariant must give an output
    # whose rank is none.
    @eager_def_function.function
    def runTest():
      # [FUNCTION DOC]
      # HELPER: compiled function that builds the loop and asserts on the output rank; called once.
      while_output = self._buildWhileWithShapeInvariants(
          [tensor_shape.TensorShape(None)])
      self.assertIsNone(while_output.shape.rank)
    runTest()

  def testWhileOutputShapeWithShapeInvariantsPartialShape(self):
    # [FUNCTION DOC]
    # Test: in a compiled function, a loop with an unknown-length vector invariant must give an
    # output of shape unknown-length.
    @eager_def_function.function
    def runTest():
      # [FUNCTION DOC]
      # HELPER: compiled function that builds the loop and asserts on the output shape; called once.
      while_output = self._buildWhileWithShapeInvariants(
          [tensor_shape.TensorShape([None])])
      self.assertAllEqual(while_output.shape.as_list(), [None])
    runTest()

  def testFunctionInWhile(self):
    # [FUNCTION DOC]
    # Test: a compiled function used as loop body while counting to five from zero. Expects 5.

    @eager_def_function.function
    def body(x):
      # [FUNCTION DOC]
      # HELPER: compiled function that returns x plus one.
      return x + 1

    r = while_loop_tf.while_loop(lambda x: x < 5, body, [0])
    self.assertAllEqual(r, 5.)


class ControlFlowContextCheckTest(test.TestCase):
  # [CLASS DOC]
  # Test case checking the validity rules for using tensors across control flow contexts (v1 only):
  # using a tensor from a while loop outside it, in a cond, or in another loop is illegal;
  # tensors from cond contexts and nested whiles are allowed. Uses helper methods that leak a tensor
  # from a while loop or a cond via a list.

  def _getWhileTensor(self):
    # [FUNCTION DOC]
    # HELPER: builds a while loop whose body leaks a constant into a list (once) and returns that
    # tensor created inside the while context.
    """Creates and returns a tensor from a while context."""
    tensor = []

    def body(i):
      # [FUNCTION DOC]
      # HELPER: on first call creates a constant one and saves it; returns i plus that tensor.
      if not tensor:
        tensor.append(constant_op.constant(1))
      return i + tensor[0]

    while_loop_tf.while_loop(lambda i: i < 10, body, [0])
    return tensor[0]

  def _getCondTensor(self):
    # [FUNCTION DOC]
    # HELPER: builds a cond whose true branch leaks a constant into a list (once) and returns that
    # tensor created inside the cond context.
    cond_tensor = []

    def true_fn():
      # [FUNCTION DOC]
      # HELPER: on first call creates a constant one and saves it; returns the saved tensor.
      if not cond_tensor:
        cond_tensor.append(constant_op.constant(1))
      return cond_tensor[0]

    tf_cond.cond(
        math_ops.less(1, 2), true_fn, lambda: constant_op.constant(0))
    return cond_tensor[0]

  @test_util.run_v1_only("b/120545219")
  def testInvalidContext(self):
    # [FUNCTION DOC]
    # Test (v1 only): using a while-loop tensor as an input to an op outside any loop raises a value
    # error saying the tensor is in a while loop.
    # Accessing a while loop tensor outside of control flow is illegal.
    while_tensor = self._getWhileTensor()
    with self.assertRaisesRegex(
        ValueError,
        "Cannot use 'while/Const_1' as input to 'Add' because 'while/Const_1' "
        "is in a while loop. See info log for more details."):
      math_ops.add(1, while_tensor)

  @test_util.run_v1_only("b/120545219")
  def testInvalidContextInCond(self):
    # [FUNCTION DOC]
    # Test (v1 only): using a while-loop tensor inside a cond branch op raises a value error. A TODO
    # notes it passes if the tensor is returned directly instead.
    # Accessing a while loop tensor in cond is illegal.
    while_tensor = self._getWhileTensor()
    with self.assertRaisesRegex(
        ValueError, "Cannot use 'while/Const_1' as input to 'cond/Add' because "
        "'while/Const_1' is in a while loop. See info log for more details."):
      # TODO(skyewm): this passes if we return while_tensor directly instead
      # of using it as input to another op.
      tf_cond.cond(
          math_ops.less(1, 2), lambda: math_ops.add(1, while_tensor),
          lambda: constant_op.constant(0))

  @test_util.run_v1_only("b/120545219")
  def testInvalidContextInWhile(self):
    # [FUNCTION DOC]
    # Test (v1 only): using a tensor of one loop in another loop raises a value error about different
    # while loops, both as an op input and as a returned next-iteration value.
    # Accessing a while loop tensor in a different while loop is illegal.
    while_tensor = self._getWhileTensor()
    with self.assertRaisesRegex(
        ValueError,
        "Cannot use 'while/Const_1' as input to 'while_1/Add' because they are "
        "in different while loops. See info log for more details."):
      while_loop_tf.while_loop(
          lambda i: i < 10, lambda x: math_ops.add(1, while_tensor), [0])

    with self.assertRaisesRegex(
        ValueError,
        "Cannot use 'while/Const_1' as input to 'while_2/NextIteration' "
        "because they are in different while loops. See info log for more "
        "details."):
      while_loop_tf.while_loop(lambda i: i < 10, lambda i: while_tensor, [0])

  def testValidCondContext(self):
    # [FUNCTION DOC]
    # Test: using a tensor from a cond context outside is allowed (though dangerous); no error.
    # Accessing a tensor from a cond context is OK (although dangerous).
    cond_tensor = self._getCondTensor()
    math_ops.add(1, cond_tensor)

  def testValidCondContextBranches(self):
    # [FUNCTION DOC]
    # Test: both cond branches are the same function, so the second branch uses a tensor created in the
    # first branch's context; allowed, no error.
    # Accessing a tensor from a cond context from the other branch's cond
    # context is OK (although dangerous).
    cond_tensor = []

    def branch_fn():
      # [FUNCTION DOC]
      # HELPER: creates and saves a constant on first call and returns the saved tensor.
      if not cond_tensor:
        cond_tensor.append(constant_op.constant(1))
      return cond_tensor[0]

    tf_cond.cond(math_ops.less(1, 2), branch_fn, branch_fn)

  @test_util.run_v1_only("b/120545219")
  def testValidWhileContext(self):
    # [FUNCTION DOC]
    # Test (v1 only): a tensor from an outer loop used inside a nested loop is allowed.
    # Accessing a tensor in a nested while is OK.
    def body(_):
      # [FUNCTION DOC]
      # HELPER: outer loop body; creates a constant and runs a nested loop adding it.
      c = constant_op.constant(1)
      return while_loop_tf.while_loop(lambda i: i < 3, lambda i: i + c, [0])

    while_loop_tf.while_loop(lambda i: i < 5, body, [0])

  @test_util.run_v1_only("b/120545219")
  def testValidNestedContexts(self):
    # [FUNCTION DOC]
    # Test (v1 only): a tensor from a cond context used in a loop within a cond within an outer loop
    # is allowed.
    # Accessing a tensor from a cond context in a while context, all inside an
    # outer while context, is OK.
    def body(_):
      # [FUNCTION DOC]
      # HELPER: outer loop body; gets a cond tensor and uses it in a loop inside a cond.
      cond_tensor = self._getCondTensor()
      # Create another cond containing the while loop for good measure
      return tf_cond.cond(
          math_ops.less(1, 2),
          lambda: while_loop_tf.while_loop(lambda i: i < 3,
                                           lambda i: i + cond_tensor, [0]),
          lambda: constant_op.constant(0))

    while_loop_tf.while_loop(lambda i: i < 5, body, [0])

  @test_util.run_v1_only("b/120545219")
  def testInvalidNestedContexts(self):
    # [FUNCTION DOC]
    # Test (v1 only): inside a cond, a tensor from one while loop is used in another while loop.
    # Expects a value error about different while loops (naming the cond-prefixed op names).
    # Accessing a tensor from a while context in a different while context, all
    # inside a cond context, is illegal.
    def true_fn():
      # [FUNCTION DOC]
      # HELPER: gets a while-loop tensor and uses it as an increment in another loop; returns that loop.
      while_tensor = self._getWhileTensor()
      return while_loop_tf.while_loop(
          lambda i: i < 3, lambda i: i + while_tensor, [0])

    with self.assertRaisesRegex(
        ValueError,
        "Cannot use 'cond/while/Const_1' as input to 'cond/while_1/add' because"
        " they are in different while loops. See info log for more details."):
      tf_cond.cond(
          math_ops.less(1, 2), true_fn, lambda: constant_op.constant(0))


class TupleTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the tuple operation, which groups tensors so that using any of them first runs
  # the control dependencies of all of them. Uses uninitialized variables as the observable effect.

  @test_util.run_v1_only("b/120545219")
  def testTensors(self):
    # [FUNCTION DOC]
    # Test (v1 only): tuple of two tensors (and a none entry) whose variable initializers are
    # dependencies. Fetching either tuple member initializes the other variable too, in either order.
    # Both variables start uninitialized (fetching fails).
    for v1_first in [True, False]:
      with self.cached_session():
        v1 = variable_v1.VariableV1([1.0])
        add1 = math_ops.add(
            control_flow_ops.with_dependencies([v1.initializer], v1._ref()),  # pylint: disable=protected-access
            2.0)
        v2 = variable_v1.VariableV1([10.0])
        add2 = math_ops.add(
            control_flow_ops.with_dependencies([v2.initializer], v2._ref()),  # pylint: disable=protected-access
            20.0)
        t1, _, t2 = control_flow_ops.tuple([add1, None, add2])

        # v1 is not initialized.
        with self.assertRaisesOpError("Attempting to use uninitialized value"):
          self.evaluate(v1)

        # v2 is not initialized.
        with self.assertRaisesOpError("Attempting to use uninitialized value"):
          self.evaluate(v2)

        if v1_first:
          # Getting t1 initializes v2.
          self.assertAllClose([3.0], self.evaluate(t1))
          self.assertAllClose([10.0], self.evaluate(v2))
        else:
          # Getting t2 initializes v1.
          self.assertAllClose([30.0], self.evaluate(t2))
          self.assertAllClose([1.0], self.evaluate(v1))

  @test_util.run_v1_only("b/120545219")
  def testIndexedSlices(self):
    # [FUNCTION DOC]
    # Test (v1 only): same as above but with indexed slices; fetching a gather of one initializes the
    # other variable too, in either order.
    for v1_first in [True, False]:
      with self.cached_session():
        v1 = variable_v1.VariableV1(
            np.array([[0.0, 1.0], [10.0, 11.0], [20.0,
                                                 21.0]]).astype(np.float32))
        v1_at_1 = indexed_slices.IndexedSlices(
            control_flow_ops.with_dependencies([v1.initializer], v1._ref()),  # pylint: disable=protected-access
            constant_op.constant([1]))

        v2 = variable_v1.VariableV1(
            np.array([[0.1, 1.1], [10.1, 11.1], [20.1,
                                                 21.1]]).astype(np.float32))
        v2_at_1 = indexed_slices.IndexedSlices(
            control_flow_ops.with_dependencies([v2.initializer], v2._ref()),  # pylint: disable=protected-access
            constant_op.constant([1]))

        st1, st2 = control_flow_ops.tuple([v1_at_1, v2_at_1])
        g1 = array_ops.gather(st1.values, st1.indices)
        g2 = array_ops.gather(st2.values, st2.indices)

        # v1 is not initialized.
        with self.assertRaisesOpError("Attempting to use uninitialized value"):
          self.evaluate(v1)

        # v2 is not initialized.
        with self.assertRaisesOpError("Attempting to use uninitialized value"):
          self.evaluate(v2)

        if v1_first:
          # Getting g1 initializes v2.
          self.assertAllClose([[10.0, 11.0]], self.evaluate(g1))
          self.assertAllClose([[0.1, 1.1], [10.1, 11.1], [20.1, 21.1]],
                              self.evaluate(v2))
        else:
          # Getting g2 initializes v1.
          self.assertAllClose([[10.1, 11.1]], self.evaluate(g2))
          self.assertAllClose([[0.0, 1.0], [10.0, 11.0], [20.0, 21.0]],
                              self.evaluate(v1))

  def testAcceptTensorsAsControlInputs(self):
    # [FUNCTION DOC]
    # Test: the tuple operation accepts an assign output as an extra control input; evaluating the
    # tuple member triggers the assignment (variable becomes 1).
    with self.cached_session():
      var = variable_v1.VariableV1(0)
      assign = state_ops.assign(var, 1)
      t, = control_flow_ops.tuple(
          [constant_op.constant(0)], control_inputs=[assign])

      # Should trigger the assign.
      self.evaluate(t)

      self.assertEqual(1, self.evaluate(var))


class AssertTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the guarded Assert operation in control flow.

  @test_util.run_deprecated_v1
  def testGuardedAssertDoesNotCopyWhenTrue(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test is skipped whenever a GPU is available, yet the assertion that the
    # unguarded assert causes a device-to-host copy is only meaningful when a GPU exists, so the
    # GPU-specific branch is effectively unreachable in normal runs.
    # Test (deprecated v1 mode): a guarded assert (control flow assert) and an unguarded raw assert on a
    # CPU true predicate with a value on the GPU name device. With full tracing, the guarded assert must
    # not trigger a device to host copy; with a GPU the unguarded one must.
    if test_util.is_gpu_available():
      self.skipTest("b/128646478 fails in opensource")

    with self.session() as sess:
      with ops.device(test.gpu_device_name()):
        value = constant_op.constant(1.0)
      with ops.device("/cpu:0"):
        true = constant_op.constant(True)
        guarded_assert = control_flow_assert.Assert(
            true, [value], name="guarded")
        unguarded_assert = gen_logging_ops._assert(
            true, [value], name="unguarded")
      opts = config_pb2.RunOptions(trace_level=config_pb2.RunOptions.FULL_TRACE)
      guarded_metadata = config_pb2.RunMetadata()
      sess.run(guarded_assert, options=opts, run_metadata=guarded_metadata)
      unguarded_metadata = config_pb2.RunMetadata()
      sess.run(unguarded_assert, options=opts, run_metadata=unguarded_metadata)
      guarded_nodestat_names = [
          n.node_name
          for d in guarded_metadata.step_stats.dev_stats
          for n in d.node_stats
      ]
      unguarded_nodestat_names = [
          n.node_name
          for d in unguarded_metadata.step_stats.dev_stats
          for n in d.node_stats
      ]
      guarded_memcpy_nodestat_names = [
          n for n in guarded_nodestat_names if "MEMCPYDtoH" in n
      ]
      unguarded_memcpy_nodestat_names = [
          n for n in unguarded_nodestat_names if "MEMCPYDtoH" in n
      ]
      if "GPU" in [d.device_type for d in device_lib.list_local_devices()]:
        # A copy was performed for the unguarded assert
        self.assertLess(0, len(unguarded_memcpy_nodestat_names),
                        str(unguarded_nodestat_names))
      # No copy was performed for the guarded assert
      self.assertEqual([], guarded_memcpy_nodestat_names)


class WhileOpBenchmark(test.Benchmark):
  # [CLASS DOC]
  # Benchmark class evaluating the performance of while_loop on a convolution loop body, compared
  # with static unrolling, on same or cross-device placement. Reports wall time per run.
  """Evaluate the performance of while_loop op."""

  def _getInitVariables(self):
    # [FUNCTION DOC]
    # HELPER: creates the benchmark inputs. Returns the initial step constant (minus one), a random
    # normal image variable of batch 10, 256 by 256, depth 16, and a truncated-normal kernel variable
    # of 3 by 3 by depth by depth. Creates variables in the current graph.
    batch_size = 10
    image_size = 256
    kernel_size = 3
    depth = 16

    init_step = constant_op.constant(-1)
    image = variable_scope.get_variable(
        "image",
        initializer=random_ops.random_normal(
            [batch_size, image_size, image_size, depth],
            dtype=dtypes.float32,
            stddev=1e-1))
    kernel = variable_scope.get_variable(
        "weights",
        initializer=random_ops.truncated_normal(
            [kernel_size, kernel_size, depth, depth],
            dtype=dtypes.float32,
            stddev=1e-1))
    return init_step, image, kernel

  def _runOneBenchmark(self,
                       default_device,
                       num_iters=10,
                       static_unroll=False,
                       steps=10):
    # [FUNCTION DOC]
    # HELPER: runs one benchmark configuration. Takes a default device for all ops except the loop
    # body (which is always on GPU), number of timed iterations (default 10), a flag to use static
    # unrolling instead of a while loop (default false) and the number of steps (default 10). Resets
    # the default graph, builds the conv loop, takes gradients, groups them to avoid fetching, runs
    # three warm-up evaluations and times the iterations. Returns average seconds per iteration.
    """Evaluate the while loop performance.

    Args:
      default_device: The default device to run all ops except the loop_body.
        loop_body is always run on GPU.
      num_iters: Number of iterations to run.
      static_unroll: If true, run unrolled version; otherwise, run while_loop.
      steps: Total number of repeated steps to run the loop.

    Returns:
      The duration of the run in seconds.
    """

    def loop_body(i, x):
      # [FUNCTION DOC]
      # HELPER: loop body; applies a same-padded 3 by 3 convolution to x with the kernel on GPU 0 and
      # increments i; returns both.
      with ops.device("/gpu:0"):
        # Always put loop body on GPU.
        nx = nn_ops.conv2d(
            input=x,
            filter=kernel,
            strides=[1, 1, 1, 1],
            padding="SAME",
            data_format="NHWC",
            name="conv2d")
        ni = math_ops.add(i, 1)
        return ni, nx

    ops.reset_default_graph()
    with session.Session() as sess, ops.device(default_device):
      # Get the initial id i, input x, and kernel.
      i, x, kernel = self._getInitVariables()
      self.evaluate(variables.global_variables_initializer())

      if static_unroll:
        for _ in range(steps):
          i, x = loop_body(i, x)
      else:
        i, x = while_loop_tf.while_loop(
            lambda i, _: i < steps,
            loop_body, [i, x],
            parallel_iterations=steps,
            swap_memory=True)

      r = math_ops.reduce_sum(x)
      dx, dk = gradients_impl.gradients(r, [x, kernel])
      # Use group to avoid fetching back results.
      r = control_flow_ops.group(dx, dk)

      for _ in range(3):
        # exclude warm up time
        self.evaluate(r)

      start_time = time.time()
      for _ in range(num_iters):
        self.evaluate(r)
      return (time.time() - start_time) / num_iters

  def benchmarkWhileOpCrossDevicePlacement(self):
    # [FUNCTION DOC]
    # Benchmark: while loop with other ops on CPU and body on GPU; reports wall time under the name
    # while_op_cross_device.
    iters = 10
    # Run loop body on GPU, but other ops on CPU.
    duration = self._runOneBenchmark("cpu", iters, static_unroll=False)
    self.report_benchmark(
        name="while_op_cross_device", iters=iters, wall_time=duration)

  def benchmarkWhileOpSameDevicePlacement(self):
    # [FUNCTION DOC]
    # Benchmark: while loop with all ops on GPU; reports wall time under the name while_op_same_device.
    iters = 10
    # Run all ops on the same GPU device.
    duration = self._runOneBenchmark("gpu", iters, static_unroll=False)
    self.report_benchmark(
        name="while_op_same_device", iters=iters, wall_time=duration)

  def benchmarkWhileOpUnrollCrossDevicePlacement(self):
    # [FUNCTION DOC]
    # Benchmark: statically unrolled loop with other ops on CPU; reports under unroll_cross_device_cpu.
    iters = 10
    # Run loop body on GPU, but other ops on CPU.
    duration = self._runOneBenchmark("cpu", iters, static_unroll=True)
    self.report_benchmark(
        name="unroll_cross_device_cpu", iters=iters, wall_time=duration)

  def benchmarkWhileOpUnrollSameDevicePlacement(self):
    # [FUNCTION DOC]
    # Benchmark: statically unrolled loop with all ops on GPU; reports under unroll_same_device.
    iters = 10
    # Run all ops on GPU.
    duration = self._runOneBenchmark("gpu", iters, static_unroll=True)
    self.report_benchmark(
        name="unroll_same_device", iters=iters, wall_time=duration)


@test_util.with_control_flow_v2
class EagerTest(test.TestCase):
  # [CLASS DOC]
  # Test case for control flow ops executed in eager mode, run with control flow v2. Each test
  # enters eager mode explicitly.

  def testCond(self):
    # [FUNCTION DOC]
    # Test: eager cond with true predicate returning a one-element list; expects the value 10 and the
    # result not being a list.
    with context.eager_mode():
      pred = math_ops.less(1, 2)
      fn1 = lambda: [constant_op.constant(10)]
      fn2 = lambda: [constant_op.constant(20)]
      r = tf_cond.cond(pred, fn1, fn2)

      self.assertAllEqual(r.numpy(), 10)
      self.assertFalse(isinstance(r, list))

  # TODO(b/117279927): Re-enable once msan failure is fixed.
  def DISABLED_testCondInDefun(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name prefix disables this test (it is not collected) pending a memory sanitizer
    # fix noted in a TODO.
    # Test (disabled): cond inside a compiled function taking a Python bool, returning pairs. Expects
    # first element 10 for true and 20 for false and the result not being a list.
    with context.eager_mode():

      @eager_def_function.function
      def foo(pred):
        # [FUNCTION DOC]
        # HELPER: compiled function; cond on a constant of the bool argument returning two-element tuples.
        # TODO(b/111124878): this only needs to output one element.
        fn1 = lambda: (constant_op.constant(10), constant_op.constant(100))
        fn2 = lambda: (constant_op.constant(20), constant_op.constant(200))
        return tf_cond.cond(constant_op.constant(pred), fn1, fn2)

      r = foo(True)
      self.assertAllEqual(r[0].numpy(), 10)
      self.assertNotIsInstance(r, list)

      r = foo(False)
      self.assertAllEqual(r[0].numpy(), 20)
      self.assertFalse(isinstance(r, list))

  def testWhileLoop(self):
    # [FUNCTION DOC]
    # Test: eager run of the summing helper loop on a vector; expects [46, 47, 48, 49, 50].
    with context.eager_mode():
      tensor = constant_op.constant([1, 2, 3, 4, 5])
      self.assertAllEqual(isum(tensor).numpy(), [46, 47, 48, 49, 50])

  def testWhileLoopWithMaxIterations(self):
    # [FUNCTION DOC]
    # Test: eager summing helper with maximum iterations three; each element plus three.
    with context.eager_mode():
      tensor = constant_op.constant([1, 2, 3, 4, 5])
      self.assertAllEqual(
          isum(tensor, maximum_iterations=3).numpy(),
          [1 + 3, 2 + 3, 3 + 3, 4 + 3, 5 + 3])

  @test_util.run_v1_only("b/120545219")
  def testWhileWithMaximumIterationsAndSingleArgument(self):
    # [FUNCTION DOC]
    # Test (v1 only): eager single-variable loop with maximum iterations one; expects 1.
    with context.eager_mode():
      tensor = constant_op.constant(0)
      r = while_loop_tf.while_loop(
          lambda i: i < 3, lambda i: i + 1, [tensor], maximum_iterations=1)
      self.assertEqual(1, r.numpy())

  def testWithDependencies(self):
    # [FUNCTION DOC]
    # Test: eager with_dependencies returns the second tensor unchanged (value 2).
    with context.eager_mode():
      t1 = constant_op.constant(1)
      t2 = constant_op.constant(2)
      t3 = control_flow_ops.with_dependencies(t1, t2)
      self.assertAllEqual(t2.numpy(), t3.numpy())

  def testTuple(self):
    # [FUNCTION DOC]
    # Test: eager tuple of two tensors returns the same values.
    with context.eager_mode():
      t1 = constant_op.constant(1)
      t2 = constant_op.constant(2)
      tup1, tup2 = control_flow_ops.tuple([t1, t2])
      self.assertAllEqual(t1.numpy(), tup1.numpy())
      self.assertAllEqual(t2.numpy(), tup2.numpy())

  @test_util.run_v1_only("b/120545219")
  def testCase(self):
    # [FUNCTION DOC]
    # Test (v1 only): eager case with two predicate-function pairs in exclusive mode and a default.
    # Expects 17.
    with context.eager_mode():
      x = constant_op.constant(1)
      y = constant_op.constant(2)
      z = constant_op.constant(3)
      f1 = lambda: constant_op.constant(17)
      f2 = lambda: constant_op.constant(23)
      f3 = lambda: constant_op.constant(-1)

      r1 = control_flow_case.case([(x < y, f1), (x > z, f2)],
                                  default=f3,
                                  exclusive=True)
      self.assertAllEqual(r1.numpy(), 17)


if __name__ == "__main__":
  test.main()
