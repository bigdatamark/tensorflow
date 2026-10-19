# [MODULE DOC]
# Unit tests for the control flow utility module (graph-inspection predicates for switch, merge,
# loop enter/exit and conditional versus loop classification of ops). Defines one test case class and
# a graph-building helper; registered through the standard test main when run as a script. No import-time
# side effects beyond imports. Not a helper module.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for tensorflow.python.ops.control_flow_util."""

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import while_loop as while_loop_tf
from tensorflow.python.platform import test


class ControlFlowUtilTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the graph-inspection predicates in the control flow utility module: whether an
  # operation is a switch, loop enter, loop constant enter, loop exit, or a conditional/loop flavor of
  # switch and merge. Version 1 style graph-mode control flow ops are built either directly through
  # low-level op constructors or via cond and while loop helpers inside a private graph. There is no
  # shared fixture except a graph-building helper method. Derives from the framework test case.

  @test_util.run_v1_only("b/120545219")
  def testIsSwitch(self):
    # [FUNCTION DOC]
    # Test (graph mode, version 1 only) that the switch predicate recognizes both a plain switch op and
    # a reference-typed switch op, and rejects an unrelated op. Builds a switch on a constant and a
    # boolean, a reference switch on a reference-output test op, and asserts the predicate is true for
    # both and false for a plain integer-output test op. Raises assertion failures on mismatch.
    switch_false, _ = control_flow_ops.switch(1, True)
    switch = switch_false.op
    self.assertTrue(control_flow_util.IsSwitch(switch))

    ref_switch_false, _ = control_flow_ops.ref_switch(test_ops.ref_output(),
                                                      True)
    ref_switch = ref_switch_false.op
    self.assertTrue(control_flow_util.IsSwitch(ref_switch))

    self.assertFalse(control_flow_util.IsSwitch(test_ops.int_output().op))

  @test_util.run_v1_only("b/120545219")
  def testIsLoopEnter(self):
    # [FUNCTION DOC]
    # Test (graph mode, version 1 only) of the loop-enter predicates. Creates a regular enter op, a
    # reference enter op and an enter op flagged as constant, all in a frame named "name". Asserts the
    # loop-enter predicate is true for all three, the loop-constant-enter predicate is true only for the
    # constant one, and loop-enter is false for an unrelated integer-output test op.
    enter = gen_control_flow_ops.enter(1, frame_name="name").op
    self.assertTrue(control_flow_util.IsLoopEnter(enter))
    self.assertFalse(control_flow_util.IsLoopConstantEnter(enter))

    ref_enter = gen_control_flow_ops.ref_enter(test_ops.ref_output(),
                                               frame_name="name").op
    self.assertTrue(control_flow_util.IsLoopEnter(ref_enter))
    self.assertFalse(control_flow_util.IsLoopConstantEnter(ref_enter))

    const_enter = gen_control_flow_ops.enter(1, frame_name="name",
                                             is_constant=True).op
    self.assertTrue(control_flow_util.IsLoopEnter(const_enter))
    self.assertTrue(control_flow_util.IsLoopConstantEnter(const_enter))

    self.assertFalse(control_flow_util.IsLoopEnter(test_ops.int_output().op))

  @test_util.run_v1_only("b/120545219")
  def testIsLoopExit(self):
    # [FUNCTION DOC]
    # Test (graph mode, version 1 only) of the loop-exit predicate. Builds an exit op on a constant and
    # an exit op on a reference-output op and asserts both are recognized; asserts that a plain
    # integer-output test op is not.
    exit_op = control_flow_ops.exit(1).op
    self.assertTrue(control_flow_util.IsLoopExit(exit_op))

    ref_exit = control_flow_ops.exit(test_ops.ref_output()).op
    self.assertTrue(control_flow_util.IsLoopExit(ref_exit))

    self.assertFalse(control_flow_util.IsLoopExit(test_ops.int_output().op))

  def build_test_graph(self) -> ops.Graph:
    # [FUNCTION DOC]
    # HELPER: builds the shared graph used by the switch/merge classification tests.
    # Creates and returns a new graph containing a placeholder integer input and an outer conditional
    # (name scope OuterCond). Its true branch runs a while loop (name scope OuterWhile) whose body holds
    # a nested conditional (name scope NestedCond) adding 1 or 2 depending on a threshold of 100; the
    # loop continues while the value is below 10000; the false branch adds 2. Takes no inputs besides
    # self, returns the graph, and the only side effect is operations created in that new graph.
    g = ops.Graph()
    with g.as_default():

      def while_loop(x):
        # [FUNCTION DOC]
        # HELPER: nested builder for the loop part of the test graph in build_test_graph.
        # Takes a tensor x and returns the result of a while loop (under the name scope OuterWhile) with
        # loop variable x, condition "less than 10000", and a body that contains a nested conditional
        # adding 1 when the value is under 100 and 2 otherwise. Adds ops to the current default graph.

        def b(x):
          # [FUNCTION DOC]
          # HELPER: loop body for the while loop built in the nested helper of build_test_graph.
          # Takes the loop variable x and, inside the name scope NestedCond, returns a conditional that adds 1
          # to x if x is less than 100 and adds 2 otherwise. Adds conditional ops to the current graph.
          with ops.name_scope("NestedCond"):
            return cond.cond(
                math_ops.less(x, 100), lambda: math_ops.add(x, 1),
                lambda: math_ops.add(x, 2))

        c = lambda x: math_ops.less(x, 10000)
        with ops.name_scope("OuterWhile"):
          return while_loop_tf.while_loop(c, b, [x])

      x = array_ops.placeholder(dtypes.int32)
      with ops.name_scope("OuterCond"):
        cond.cond(
            math_ops.less(x, 1000), lambda: while_loop(x),
            lambda: math_ops.add(x, 2))
    return g

  def testIsCondSwitch(self):
    # [FUNCTION DOC]
    # Test of the conditional-switch predicate on the shared test graph. Every switch op in the graph
    # must be classified as exactly one of conditional switch or loop switch. Ops whose names are in a
    # fixed list of six expected conditional switch names must be switches, conditional switches and
    # not loop switches; every other op must not be a conditional switch. Mismatches report the op name.
    g = self.build_test_graph()

    cond_switch = [
        "OuterCond/cond/Switch",
        "OuterCond/cond/OuterWhile/while/Switch",
        "OuterCond/cond/OuterWhile/while/NestedCond/cond/Switch",
        "OuterCond/cond/OuterWhile/while/NestedCond/cond/Add/Switch",
        "OuterCond/cond/OuterWhile/while/NestedCond/cond/Add_1/Switch",
        "OuterCond/cond/Add/Switch",
    ]
    for n in g.get_operations():
      if control_flow_util.IsSwitch(n):
        self.assertTrue(
            control_flow_util.IsCondSwitch(n) != control_flow_util.IsLoopSwitch(
                n))
      if n.name in cond_switch:
        self.assertTrue(control_flow_util.IsSwitch(n))
        self.assertTrue(
            control_flow_util.IsCondSwitch(n),
            msg="Mismatch for {}".format(n.name))
        self.assertFalse(
            control_flow_util.IsLoopSwitch(n),
            msg="Mismatch for {}".format(n.name))
      else:
        self.assertFalse(
            control_flow_util.IsCondSwitch(n),
            msg="Mismatch for {}".format(n.name))

  def testIsLoopSwitch(self):
    # [FUNCTION DOC]
    # Test of the loop-switch predicate on the shared test graph. Every switch must be exactly one of
    # conditional or loop switch. The single op named as the switch for the second loop variable of the
    # while loop must be a switch, not a conditional switch, and a loop switch; every other op must not
    # be a loop switch. Mismatches report the op name.
    g = self.build_test_graph()

    loop_switch = ["OuterCond/cond/OuterWhile/while/Switch_1"]
    for n in g.get_operations():
      if control_flow_util.IsSwitch(n):
        self.assertTrue(
            control_flow_util.IsCondSwitch(n) != control_flow_util.IsLoopSwitch(
                n))
      if n.name in loop_switch:
        self.assertTrue(control_flow_util.IsSwitch(n))
        self.assertFalse(
            control_flow_util.IsCondSwitch(n),
            msg="Mismatch for {}".format(n.name))
        self.assertTrue(
            control_flow_util.IsLoopSwitch(n),
            msg="Mismatch for {}".format(n.name))
      else:
        self.assertFalse(
            control_flow_util.IsLoopSwitch(n),
            msg="Mismatch for {}".format(n.name))

  def testIsCondMerge(self):
    # [FUNCTION DOC]
    # Test of the conditional-merge predicate on the shared test graph. The two merges belonging to the
    # outer and the nested conditional must be merges, conditional merges and not loop merges. For all
    # other ops, the conditional-merge predicate must be false and the op must either not be a merge or
    # be a loop merge.
    g = self.build_test_graph()
    cond_merges = [
        "OuterCond/cond/OuterWhile/while/NestedCond/cond/Merge",
        "OuterCond/cond/Merge"
    ]
    for n in g.get_operations():
      if n.name in cond_merges:
        self.assertTrue(control_flow_util.IsMerge(n))
        self.assertTrue(control_flow_util.IsCondMerge(n))
        self.assertFalse(control_flow_util.IsLoopMerge(n))
      else:
        self.assertFalse(control_flow_util.IsCondMerge(n))
        self.assertTrue(not control_flow_util.IsMerge(n) or
                        control_flow_util.IsLoopMerge(n))

  def testIsLoopMerge(self):
    # [FUNCTION DOC]
    # Test of the loop-merge predicate on the shared test graph. The one merge of the while loop must be
    # a merge, not a conditional merge, and a loop merge. For every other op the loop-merge predicate
    # must be false and the op must either not be a merge or be a conditional merge.
    g = self.build_test_graph()
    loop_merges = [
        "OuterCond/cond/OuterWhile/while/Merge",
    ]
    for n in g.get_operations():
      if n.name in loop_merges:
        self.assertTrue(control_flow_util.IsMerge(n))
        self.assertFalse(control_flow_util.IsCondMerge(n))
        self.assertTrue(control_flow_util.IsLoopMerge(n))
      else:
        self.assertFalse(control_flow_util.IsLoopMerge(n))
        self.assertTrue(not control_flow_util.IsMerge(n) or
                        control_flow_util.IsCondMerge(n))


if __name__ == "__main__":
  test.main()
