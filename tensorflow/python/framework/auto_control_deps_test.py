# [MODULE DOC]
# Tests for the automatic control dependency context manager and its function decorator. Covers
# ordering of reads and writes on resource variables, parallel execution of order-insensitive and
# independent ops, handling of reads inside functional control flow (while, cond, case, nested
# functions), conditionals and switch/merge handling, collective-op ordering, stateful and
# stateless op classification, and interaction with gradients and optimizers. Tests run in graph
# mode or inside traced functions and inspect control inputs of built ops, or check final values.
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

import itertools

from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import auto_control_deps as acd
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_switch_case
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gen_sendrecv_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import test
from tensorflow.python.training import adam
from tensorflow.python.training import momentum


class AutomaticControlDependenciesTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the automatic control dependency context manager and decorator. The shared
  # fixture saves the module-level set of must-run order-insensitive stateful op types before each
  # test and restores it afterwards, because one test extends it. Tests build ops in graph mode
  # or traced functions and then assert on control inputs, must-run sets or computed values.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture: calls the base setup and remembers the current module-level set of must-run
    # order-insensitive stateful op types so that tearDown can restore it.
    super().setUp()
    self.must_run_order_insensitive_stateful_ops = (
        acd.MUST_RUN_ORDER_INSENSITIVE_STATEFUL_OPS)

  def tearDown(self):
    # [FUNCTION DOC]
    # Fixture: restores the module-level must-run order-insensitive op set saved in setUp (undoing
    # any extension made by a test), then calls the base teardown.
    acd.MUST_RUN_ORDER_INSENSITIVE_STATEFUL_OPS = (
        self.must_run_order_insensitive_stateful_ops)
    super().tearDown()

  def testBasic(self):
    # [FUNCTION DOC]
    # Scenario: in graph mode, an initialized variable is incremented, then doubled, then read
    # inside the context, and the read is marked as a return value. Asserts the evaluated result
    # is 4.0, showing the writes and read executed in program order.
    with context.graph_mode(), self.cached_session():
      v = resource_variable_ops.ResourceVariable(1.0)
      self.evaluate(variables.global_variables_initializer())
      with acd.AutomaticControlDependencies() as c:
        v.assign(v + 1)
        v.assign(2 * v)
        val = v.read_value()
        val = c.mark_as_return(val)
      self.assertAllEqual(val, 4.0)

  def testUnorderedOpsRunInParallel(self):
    # [FUNCTION DOC]
    # Scenario: the eager Python function op type is added to the module's must-run
    # order-insensitive set (restored in tearDown). A traced function calls two such ops, each
    # appending a distinct marker to a shared list. Asserts that both markers are present after
    # running, i.e. both ops ran even though no ordering was imposed.
    acd.MUST_RUN_ORDER_INSENSITIVE_STATEFUL_OPS |= frozenset(("EagerPyFunc",))

    side_effects = []

    def side_effect_one(x):
      # [FUNCTION DOC]
      # Callback for the first Python function op: records marker one in the shared list and
      # returns its input unchanged.
      side_effects.append(1)
      return x

    def side_effect_two(x):
      # [FUNCTION DOC]
      # Callback for the second Python function op: records marker two in the shared list and
      # returns its input unchanged.
      side_effects.append(2)
      return x

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # Traced function that invokes the two Python function ops (outputs discarded) and returns
      # the constant one.
      script_ops.eager_py_func(side_effect_one, [1], [dtypes.int32])
      script_ops.eager_py_func(side_effect_two, [1], [dtypes.int32])
      return 1

    side_effects = []
    self.evaluate(f())

    self.assertSetEqual(set(side_effects), set((1, 2)))

  def testIndependentOpsRunInParallel(self):
    # [FUNCTION DOC]
    # Scenario: a traced function assigns 1 to a variable normally and then assigns 2 through an
    # op registered as independent. The function is run many times, reading the variable after
    # each run. Asserts that both values 1 and 2 are observed over the runs, showing that the
    # independent assignment is not ordered relative to the first (with ordering only 2 would
    # appear). Statistical by nature: relies on a data race across 10000 runs.
    v = resource_variable_ops.ResourceVariable(1)
    self.evaluate(variables.global_variables_initializer())

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # Traced function: assigns 1 to the variable, then assigns 2 and marks that second assign
      # as independent via the graph's automatic control dependency manager. Returns nothing.
      gen_resource_variable_ops.assign_variable_op(v.handle, 1)
      ops.get_default_graph().experimental_acd_manager.run_independently(
          gen_resource_variable_ops.assign_variable_op(v.handle, 2))

    # A function with two identical ops, should cause a data race in most
    # conditions.
    var_values = set()
    for _ in range(10000):
      self.evaluate(f())
      var_values.add(
          self.evaluate(
              resource_variable_ops.read_variable_op(v.handle, dtypes.int32)))
    # With regular control dependencies, the function should always run the
    # first assign first, and the value 1 should never be seen.
    # With run_independently, assign 1 and 2 are run in parallel. Thus, when f
    # is run large number of times, we see both 1 and 2 values assigned to
    # variable v.
    self.assertSetEqual(var_values, set((1, 2)))

  def testIndependentOpsInLoop(self):
    # [FUNCTION DOC]
    # Scenario: a traced function loops over a range and, on each iteration, assigns the loop
    # index to a variable, marking the assign as independent. After running once, asserts the
    # variable equals 2 (the last index). A TODO notes this is not a robust way to test loops.
    v = resource_variable_ops.ResourceVariable(0)
    self.evaluate(variables.global_variables_initializer())

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # Traced function: loops over a range of three, assigning each index to the variable and
      # registering each assign op as independent. Returns nothing.
      for i in math_ops.range(3):
        ops.get_default_graph().experimental_acd_manager.run_independently(
            gen_resource_variable_ops.assign_variable_op(v.handle, i))

    self.evaluate(f())
    # TODO(mdan): Find a more robust way to test in loops.
    self.assertEqual(
        self.evaluate(
            resource_variable_ops.read_variable_op(v.handle, dtypes.int32)), 2)

  def testNoControlDepsBetweenVariableReads(self):
    # [FUNCTION DOC]
    # Scenario: in graph mode, two reads of the same variable followed by an assign inside the
    # context. Asserts neither read has the other among its control inputs (reads are unordered).
    with context.graph_mode(), self.cached_session():
      v = resource_variable_ops.ResourceVariable(1.0)
      self.evaluate(variables.global_variables_initializer())
      with acd.AutomaticControlDependencies():
        read_op1 = gen_resource_variable_ops.read_variable_op(
            v.handle, v.dtype).op
        read_op2 = gen_resource_variable_ops.read_variable_op(
            v.handle, v.dtype).op
        gen_resource_variable_ops.assign_variable_op(v.handle, v + 1)
      self.assertNotIn(read_op1, read_op2.control_inputs)
      self.assertNotIn(read_op2, read_op1.control_inputs)

  def testVariableReadThenWrite(self):
    # [FUNCTION DOC]
    # Scenario: two reads then an assign of the same variable inside the context. Asserts the
    # assign has both reads as control inputs and the reads have no control edges between them.
    with context.graph_mode(), self.cached_session():
      v = resource_variable_ops.ResourceVariable(1.0)
      self.evaluate(variables.global_variables_initializer())
      with acd.AutomaticControlDependencies():
        read_op1 = gen_resource_variable_ops.read_variable_op(
            v.handle, v.dtype).op
        read_op2 = gen_resource_variable_ops.read_variable_op(
            v.handle, v.dtype).op
        assign_op = gen_resource_variable_ops.assign_variable_op(
            v.handle, v + 1)
      # Writes should have control deps from "all" reads since last write
      # or start of the code block.
      self.assertIn(read_op1, assign_op.control_inputs)
      self.assertIn(read_op2, assign_op.control_inputs)
      # There should be no control deps between reads.
      self.assertNotIn(read_op1, read_op2.control_inputs)
      self.assertNotIn(read_op2, read_op1.control_inputs)

  def testVariableWriteThenRead(self):
    # [FUNCTION DOC]
    # Scenario: an assign followed by two reads of the same variable inside the context. Asserts
    # both reads have the assign as a control input and have no control edges between each other.
    with context.graph_mode(), self.cached_session():
      v = resource_variable_ops.ResourceVariable(1.0)
      self.evaluate(variables.global_variables_initializer())
      with acd.AutomaticControlDependencies():
        assign_op = gen_resource_variable_ops.assign_variable_op(
            v.handle, v + 1)
        read_op1 = gen_resource_variable_ops.read_variable_op(
            v.handle, v.dtype).op
        read_op2 = gen_resource_variable_ops.read_variable_op(
            v.handle, v.dtype).op
      # Reads should have a control dep from the last write.
      self.assertIn(assign_op, read_op1.control_inputs)
      self.assertIn(assign_op, read_op2.control_inputs)
      # There should be no control deps between reads.
      self.assertNotIn(read_op1, read_op2.control_inputs)
      self.assertNotIn(read_op2, read_op1.control_inputs)

  def testIdentityPassThrough(self):
    # [FUNCTION DOC]
    # Scenario: assign, then pass the variable handle through an identity op, assign again, then
    # read using the identity output. Asserts the read has the second assign as a control input,
    # showing identity on a resource handle is resolved to the underlying resource.
    with context.graph_mode(), self.cached_session():
      v = resource_variable_ops.ResourceVariable(1.0)
      self.evaluate(variables.global_variables_initializer())
      with acd.AutomaticControlDependencies():
        gen_resource_variable_ops.assign_variable_op(v.handle, v + 1)
        identity_handle = gen_array_ops.identity(v.handle)
        assign_op2 = gen_resource_variable_ops.assign_variable_op(
            v.handle, v + 1)
        read_op = gen_resource_variable_ops.read_variable_op(
            identity_handle, v.dtype).op
      # Read should have a control dep from second last write even
      # with Identity applied to resource.
      self.assertIn(assign_op2, read_op.control_inputs)

  def testVariableReadsInOpsWithMustRun(self):
    # [FUNCTION DOC]
    # Scenario: a single variable read whose output is marked as a return value (so it has a
    # consumer). Asserts the read op is in the context's must-run set.
    with context.graph_mode(), self.cached_session():
      v = resource_variable_ops.ResourceVariable(1.0)
      self.evaluate(variables.global_variables_initializer())
      with acd.AutomaticControlDependencies() as c:
        read_op = gen_resource_variable_ops.read_variable_op(v.handle,
                                                             v.dtype).op
        # Read ops get added to control outputs only if they have consumers.
        c.mark_as_return(read_op.outputs[0])
      self.assertIn(read_op, c.ops_which_must_run)

  def testVariableMultipleReadsAndWrites(self):
    # [FUNCTION DOC]
    # Scenario: a sequence of two reads, two writes, two reads and two writes on one variable,
    # with all reads marked as returns. Asserts that each group of writes depends on the preceding
    # reads, writes chain in order, later reads depend on the last earlier write, reads have no
    # edges between them, and all reads plus the final write are in the must-run set.
    with context.graph_mode(), self.cached_session():
      v = resource_variable_ops.ResourceVariable(1.0)
      self.evaluate(variables.global_variables_initializer())
      with acd.AutomaticControlDependencies() as c:
        # 2 reads -> 2 writes -> 2 reads -> 2 writes.
        read_op1 = gen_resource_variable_ops.read_variable_op(
            v.handle, v.dtype).op
        read_op2 = gen_resource_variable_ops.read_variable_op(
            v.handle, v.dtype).op
        assign_op1 = gen_resource_variable_ops.assign_variable_op(
            v.handle, v + 1)
        assign_op2 = gen_resource_variable_ops.assign_variable_op(
            v.handle, v + 1)
        read_op3 = gen_resource_variable_ops.read_variable_op(
            v.handle, v.dtype).op
        read_op4 = gen_resource_variable_ops.read_variable_op(
            v.handle, v.dtype).op
        assign_op3 = gen_resource_variable_ops.assign_variable_op(
            v.handle, v + 1)
        assign_op4 = gen_resource_variable_ops.assign_variable_op(
            v.handle, v + 1)
        # Read ops get added to control outputs only if they have consumers.
        c.mark_as_return(read_op1.outputs[0])
        c.mark_as_return(read_op2.outputs[0])
        c.mark_as_return(read_op3.outputs[0])
        c.mark_as_return(read_op4.outputs[0])

      # Verify the control edges.
      self.assertIn(read_op1, assign_op1.control_inputs)
      self.assertIn(read_op2, assign_op1.control_inputs)
      self.assertIn(assign_op1, assign_op2.control_inputs)
      self.assertIn(assign_op2, read_op3.control_inputs)
      self.assertIn(assign_op2, read_op4.control_inputs)
      self.assertIn(read_op3, assign_op3.control_inputs)
      self.assertIn(read_op4, assign_op3.control_inputs)
      self.assertIn(assign_op3, assign_op4.control_inputs)

      # There should be no control deps between reads.
      read_ops = [read_op1, read_op2, read_op3, read_op4]
      for src_op, tgt_op in itertools.product(read_ops, read_ops):
        self.assertNotIn(src_op, tgt_op.control_inputs)

      # Reads must be in `ops_which_must_run`.
      self.assertIn(read_op1, c.ops_which_must_run)
      self.assertIn(read_op2, c.ops_which_must_run)
      self.assertIn(read_op3, c.ops_which_must_run)
      self.assertIn(read_op4, c.ops_which_must_run)
      # Last write must be in `ops_which_must_run`.
      self.assertIn(assign_op4, c.ops_which_must_run)

  def testSendInOpsWithMustRun(self):
    # [FUNCTION DOC]
    # Scenario: a send op is created inside the context. Asserts it is in the must-run set (send
    # ops are not excluded as asynchronous stateful ops so they are not pruned).
    with context.graph_mode(), self.cached_session():
      v = resource_variable_ops.ResourceVariable(1.0)
      self.evaluate(variables.global_variables_initializer())
      with acd.AutomaticControlDependencies() as c:
        send_op = gen_sendrecv_ops.send(v, "x", "/", 0, "/")

      # Send must be in `ops_which_must_run`.
      self.assertIn(send_op, c.ops_which_must_run)

  def _testVariableReadInFunctionalOp(self, build_functional_op, op_type):
    # [FUNCTION DOC]
    # HELPER: shared body for the tests where a functional op only reads a variable.
    # Inputs: a builder callback that takes the variable and creates the functional op (while,
    # cond, case or function call) returning its result, and the expected op type name. Traces a
    # function with a read named read1, the functional op, a read named read2 and an assign, then
    # asserts: the functional op has no control edge from read1, read2 has none from the
    # functional op, and the assign has read1, read2 and the functional op as control inputs.
    # Also asserts the function graph has a single input.
    v = resource_variable_ops.ResourceVariable(1.0)
    self.evaluate(variables.global_variables_initializer())

    @def_function.function
    def read_var_in_while():
      # [FUNCTION DOC]
      # Traced function: read variable (named read1), build the functional op, read again (named
      # read2), assign the variable, and return the functional op result.
      gen_resource_variable_ops.read_variable_op(
          v.handle, v.dtype, name="read1")

      result = build_functional_op(v)
      gen_resource_variable_ops.read_variable_op(
          v.handle, v.dtype, name="read2")
      gen_resource_variable_ops.assign_variable_op(v.handle, v + 1)
      return result

    func_graph = read_var_in_while.get_concrete_function().graph
    assert len(func_graph.inputs) == 1

    def get_op(op_type, sub_name):
      # [FUNCTION DOC]
      # HELPER: finds the single op in the traced function graph with the given type whose name
      # contains the given substring (empty matches any). Asserts exactly one match and returns it.
      operations = [
          op for op in func_graph.get_operations()
          if op.type == op_type and sub_name in op.name
      ]
      assert len(operations) == 1
      return operations[0]

    read1 = get_op("ReadVariableOp", "read1")
    functional_op = get_op(op_type, "")
    read2 = get_op("ReadVariableOp", "read2")
    assign_op = get_op("AssignVariableOp", "")
    # Since the functional op only has reads, previous reads e.g. read1 do not\
    # have a control edge to it and next future reads e.g. read2 do not have a
    # control edge from it.
    self.assertNotIn(read1, functional_op.control_inputs)
    self.assertNotIn(functional_op, read2.control_inputs)
    self.assertIn(read1, assign_op.control_inputs)
    self.assertIn(read2, assign_op.control_inputs)
    self.assertIn(functional_op, assign_op.control_inputs)

  def testVariableReadInWhileLoop(self):
    # [FUNCTION DOC]
    # Scenario: the functional op is a while loop (one iteration) whose body only reads the
    # variable. Delegates to the shared read helper expecting a While op, which asserts reads
    # outside the loop are not ordered against it but the later assign depends on it.

    def build_functional_op(v):
      # [FUNCTION DOC]
      # Builder: creates a single-iteration while loop over a float whose body reads the variable;
      # returns the loop result.

      def body(_):
        # [FUNCTION DOC]
        # Loop body: ignores the loop variable and returns a read of the variable.
        return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)

      return while_loop.while_loop(
          lambda i: True, body, [0.0], maximum_iterations=1)

    self._testVariableReadInFunctionalOp(build_functional_op, "While")

  def testVariableReadInCondTrueBranch(self):
    # [FUNCTION DOC]
    # Scenario: a conditional with a constant true predicate whose true branch reads the variable
    # and whose false branch returns zeros. Delegates to the shared read helper expecting an If
    # op (reads are not ordered against the conditional; later write depends on it).

    def build_functional_op(v):
      # [FUNCTION DOC]
      # Builder: creates the conditional described above and returns its result.

      def then_branch():
        # [FUNCTION DOC]
        # True branch: returns a read of the variable.
        return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)

      def else_branch():
        # [FUNCTION DOC]
        # False branch: returns a scalar zero of the variable's type.
        return array_ops.zeros([], v.dtype)

      return cond.cond(
          constant_op.constant(True), then_branch, else_branch)

    self._testVariableReadInFunctionalOp(build_functional_op, "If")

  def testVariableReadInCondFalseBranch(self):
    # [FUNCTION DOC]
    # Scenario: a conditional with a constant false predicate whose false branch reads the
    # variable and whose true branch returns zeros. Delegates to the shared read helper expecting
    # an If op.

    def build_functional_op(v):
      # [FUNCTION DOC]
      # Builder: creates the conditional described above and returns its result.

      def then_branch():
        # [FUNCTION DOC]
        # True branch: returns a scalar zero of the variable's type.
        return array_ops.zeros([], v.dtype)

      def else_branch():
        # [FUNCTION DOC]
        # False branch: returns a read of the variable.
        return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)

      return cond.cond(
          constant_op.constant(False), then_branch, else_branch)

    self._testVariableReadInFunctionalOp(build_functional_op, "If")

  def testVariableReadInCaseBranch0(self):
    # [FUNCTION DOC]
    # Scenario: a switch-case over index zero where branch zero reads the variable and branch one
    # returns zeros. Delegates to the shared read helper expecting a Case op.

    def build_functional_op(v):
      # [FUNCTION DOC]
      # Builder: creates the switch-case described above and returns its result.

      def branch0():
        # [FUNCTION DOC]
        # Branch zero: returns a read of the variable.
        return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)

      def branch1():
        # [FUNCTION DOC]
        # Branch one: returns a scalar zero of the variable's type.
        return array_ops.zeros([], v.dtype)

      return control_flow_switch_case.switch_case(
          constant_op.constant(0), [branch0, branch1])

    self._testVariableReadInFunctionalOp(build_functional_op, "Case")

  def testVariableReadInCaseBranch1(self):
    # [FUNCTION DOC]
    # Scenario: a switch-case where branch one reads the variable and branch zero returns zeros
    # (the index constant is still zero). Delegates to the shared read helper expecting a Case op.

    def build_functional_op(v):
      # [FUNCTION DOC]
      # Builder: creates the switch-case described above and returns its result.

      def branch0():
        # [FUNCTION DOC]
        # Branch zero: returns a scalar zero of the variable's type.
        return array_ops.zeros([], v.dtype)

      def branch1():
        # [FUNCTION DOC]
        # Branch one: returns a read of the variable.
        return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)

      return control_flow_switch_case.switch_case(
          constant_op.constant(0), [branch0, branch1])

    self._testVariableReadInFunctionalOp(build_functional_op, "Case")

  def testVariableReadInFunction(self):
    # [FUNCTION DOC]
    # Scenario: the functional op is a call to a traced function that reads the variable.
    # Delegates to the shared read helper expecting a stateful partitioned call op.

    def build_functional_op(v):
      # [FUNCTION DOC]
      # Builder: defines a traced function that reads the variable, calls it, and returns the
      # call result.

      @def_function.function
      def fn_with_read():
        # [FUNCTION DOC]
        # Traced function: returns a read of the captured variable.
        return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)

      return fn_with_read()

    self._testVariableReadInFunctionalOp(build_functional_op,
                                         "StatefulPartitionedCall")

  def testVariableReadInNestedFunction(self):
    # [FUNCTION DOC]
    # Scenario: a traced function that calls another traced function which reads the variable.
    # Delegates to the shared read helper expecting a stateful partitioned call op.

    def build_functional_op(v):
      # [FUNCTION DOC]
      # Builder: defines the nested traced functions, calls the outer one and returns its result.

      @def_function.function
      def fn_with_read():
        # [FUNCTION DOC]
        # Outer traced function: calls the inner function and returns its result.

        @def_function.function
        def inner_fn():
          # [FUNCTION DOC]
          # Inner traced function: returns a read of the captured variable.
          return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)

        return inner_fn()

      return fn_with_read()

    self._testVariableReadInFunctionalOp(build_functional_op,
                                         "StatefulPartitionedCall")

  def testVariableReadInWhileInInnerFunc(self):
    # [FUNCTION DOC]
    # Scenario: nested traced functions where the inner one runs a one-iteration while loop whose
    # body reads the variable. Delegates to the shared read helper expecting a stateful
    # partitioned call op.

    def build_functional_op(v):
      # [FUNCTION DOC]
      # Builder: defines the nested traced functions, calls the outer one and returns its result.

      @def_function.function
      def fn_with_read():
        # [FUNCTION DOC]
        # Outer traced function: calls the inner function and returns its result.

        @def_function.function
        def inner_fn():
          # [FUNCTION DOC]
          # Inner traced function: runs the while loop and returns its result.

          def body(_):
            # [FUNCTION DOC]
            # Loop body: ignores the loop variable and returns a read of the variable.
            return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)

          return while_loop.while_loop(
              lambda i: True, body, [0.0], maximum_iterations=1)

        return inner_fn()

      return fn_with_read()

    self._testVariableReadInFunctionalOp(build_functional_op,
                                         "StatefulPartitionedCall")

  def testVariableReadInCondInInnerFunc(self):
    # [FUNCTION DOC]
    # Scenario: nested traced functions where the inner one runs a conditional (constant true)
    # whose true branch reads the variable. Delegates to the shared read helper expecting a
    # stateful partitioned call op.

    def build_functional_op(v):
      # [FUNCTION DOC]
      # Builder: defines the nested traced functions, calls the outer one and returns its result.

      @def_function.function
      def fn_with_read():
        # [FUNCTION DOC]
        # Outer traced function: calls the inner function and returns its result.

        @def_function.function
        def inner_fn():
          # [FUNCTION DOC]
          # Inner traced function: runs the conditional and returns its result.

          def then_branch():
            # [FUNCTION DOC]
            # True branch: returns a read of the variable.
            return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)

          def else_branch():
            # [FUNCTION DOC]
            # False branch: returns a scalar zero of the variable's type.
            return array_ops.zeros([], v.dtype)

          return cond.cond(
              constant_op.constant(True), then_branch, else_branch)

        return inner_fn()

      return fn_with_read()

    self._testVariableReadInFunctionalOp(build_functional_op,
                                         "StatefulPartitionedCall")

  def _testVariableWriteInFunctionalOp(self, build_functional_op, op_type):
    # [FUNCTION DOC]
    # HELPER: shared body for the tests where a functional op writes a variable.
    # Inputs: a builder callback taking the variable and creating the functional op (returning
    # its result), and the expected op type name. Traces a function with a read named read1, the
    # functional op, a read named read2 and an assign, then asserts: the functional op has read1
    # as a control input, read2 has the functional op as control input, and the assign has read2
    # and the functional op as control inputs. Also asserts the function graph has a single input.
    v = resource_variable_ops.ResourceVariable(1.0)
    self.evaluate(variables.global_variables_initializer())

    @def_function.function
    def write_var_in_while():
      # [FUNCTION DOC]
      # Traced function: read variable (named read1), build the functional op, read again (named
      # read2), assign the variable, and return the functional op result.
      gen_resource_variable_ops.read_variable_op(
          v.handle, v.dtype, name="read1")

      result = build_functional_op(v)
      gen_resource_variable_ops.read_variable_op(
          v.handle, v.dtype, name="read2")
      gen_resource_variable_ops.assign_variable_op(v.handle, v + 1)
      return result

    func_graph = write_var_in_while.get_concrete_function().graph
    assert len(func_graph.inputs) == 1

    def get_op(op_type, sub_name):
      # [FUNCTION DOC]
      # HELPER: finds the single op in the traced function graph with the given type whose name
      # contains the given substring (empty matches any). Asserts exactly one match and returns it.
      operations = [
          op for op in func_graph.get_operations()
          if op.type == op_type and sub_name in op.name
      ]
      assert len(operations) == 1
      return operations[0]

    read1 = get_op("ReadVariableOp", "read1")
    functional_op = get_op(op_type, "")
    read2 = get_op("ReadVariableOp", "read2")
    assign_op = get_op("AssignVariableOp", "")
    # Since the While has writes, it has control edges from previous reads
    # e.g. `read1` and to future reads(`read2`) and writes(`assign_op`).
    self.assertIn(read1, functional_op.control_inputs)
    self.assertIn(functional_op, read2.control_inputs)
    self.assertIn(read2, assign_op.control_inputs)
    self.assertIn(functional_op, assign_op.control_inputs)

  def testVariableWriteInWhileLoop(self):
    # [FUNCTION DOC]
    # Scenario: the functional op is a one-iteration while loop whose body assigns and then reads
    # the variable. Delegates to the shared write helper expecting a While op, which asserts it is
    # ordered after earlier reads and before later reads and writes.

    def build_functional_op(v):
      # [FUNCTION DOC]
      # Builder: creates the single-iteration while loop and returns its result.

      def body(_):
        # [FUNCTION DOC]
        # Loop body: assigns the variable its value plus one, then returns a read of it.
        gen_resource_variable_ops.assign_variable_op(v.handle, v + 1)
        return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)

      return while_loop.while_loop(
          lambda i: True, body, [0.0], maximum_iterations=1)

    self._testVariableWriteInFunctionalOp(build_functional_op, "While")

  def testVariableWriteInCondTrueBranch(self):
    # [FUNCTION DOC]
    # Scenario: a conditional (constant true) whose true branch assigns and reads the variable and
    # whose false branch returns zeros. Delegates to the shared write helper expecting an If op.

    def build_functional_op(v):
      # [FUNCTION DOC]
      # Builder: creates the conditional described above and returns its result.

      def then_branch():
        # [FUNCTION DOC]
        # True branch: assigns the variable its value plus one and returns a read of it.
        gen_resource_variable_ops.assign_variable_op(v.handle, v + 1)
        return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)

      def else_branch():
        # [FUNCTION DOC]
        # False branch: returns a scalar zero of the variable's type.
        return array_ops.zeros([], v.dtype)

      return cond.cond(
          constant_op.constant(True), then_branch, else_branch)

    self._testVariableWriteInFunctionalOp(build_functional_op, "If")

  def testVariableWriteInCondFalseBranch(self):
    # [FUNCTION DOC]
    # Scenario: a conditional (constant false) whose false branch assigns and reads the variable
    # and whose true branch returns zeros. Delegates to the shared write helper expecting an If op.

    def build_functional_op(v):
      # [FUNCTION DOC]
      # Builder: creates the conditional described above and returns its result.

      def then_branch():
        # [FUNCTION DOC]
        # True branch: returns a scalar zero of the variable's type.
        return array_ops.zeros([], v.dtype)

      def else_branch():
        # [FUNCTION DOC]
        # False branch: assigns the variable its value plus one and returns a read of it.
        gen_resource_variable_ops.assign_variable_op(v.handle, v + 1)
        return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)

      return cond.cond(
          constant_op.constant(False), then_branch, else_branch)

    self._testVariableWriteInFunctionalOp(build_functional_op, "If")

  def testVariableWriteInCaseBranch0(self):
    # [FUNCTION DOC]
    # Scenario: a switch-case (index zero) where branch zero assigns and reads the variable and
    # branch one returns zeros. Delegates to the shared write helper expecting a Case op.

    def build_functional_op(v):
      # [FUNCTION DOC]
      # Builder: creates the switch-case described above and returns its result.

      def branch0():
        # [FUNCTION DOC]
        # Branch zero: assigns the variable its value plus one and returns a read of it.
        gen_resource_variable_ops.assign_variable_op(v.handle, v + 1)
        return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)

      def branch1():
        # [FUNCTION DOC]
        # Branch one: returns a scalar zero of the variable's type.
        return array_ops.zeros([], v.dtype)

      return control_flow_switch_case.switch_case(
          constant_op.constant(0), [branch0, branch1])

    self._testVariableWriteInFunctionalOp(build_functional_op, "Case")

  def testVariableWriteInCaseBranch1(self):
    # [FUNCTION DOC]
    # Scenario: a switch-case where branch one assigns and reads the variable and branch zero
    # returns zeros (the index constant is still zero). Delegates to the shared write helper
    # expecting a Case op.

    def build_functional_op(v):
      # [FUNCTION DOC]
      # Builder: creates the switch-case described above and returns its result.

      def branch0():
        # [FUNCTION DOC]
        # Branch zero: returns a scalar zero of the variable's type.
        return array_ops.zeros([], v.dtype)

      def branch1():
        # [FUNCTION DOC]
        # Branch one: assigns the variable its value plus one and returns a read of it.
        gen_resource_variable_ops.assign_variable_op(v.handle, v + 1)
        return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)

      return control_flow_switch_case.switch_case(
          constant_op.constant(0), [branch0, branch1])

    self._testVariableWriteInFunctionalOp(build_functional_op, "Case")

  def testVariableWriteInFunction(self):
    # [FUNCTION DOC]
    # Scenario: the functional op is a call to a traced function that assigns and reads the
    # variable. Delegates to the shared write helper expecting a stateful partitioned call op.

    def build_functional_op(v):
      # [FUNCTION DOC]
      # Builder: defines the traced function, calls it and returns the call result.

      @def_function.function
      def fn_with_write():
        # [FUNCTION DOC]
        # Traced function: assigns the captured variable its value plus one and returns a read.
        gen_resource_variable_ops.assign_variable_op(v.handle, v + 1)
        return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)

      return fn_with_write()

    self._testVariableWriteInFunctionalOp(build_functional_op,
                                          "StatefulPartitionedCall")

  def testVariableWriteInNestedFunction(self):
    # [FUNCTION DOC]
    # Scenario: a traced function calling another traced function that assigns and reads the
    # variable. Delegates to the shared write helper expecting a stateful partitioned call op.

    def build_functional_op(v):
      # [FUNCTION DOC]
      # Builder: defines the nested traced functions, calls the outer one, returns its result.

      @def_function.function
      def fn_with_write():
        # [FUNCTION DOC]
        # Outer traced function: calls the inner function and returns its result.

        @def_function.function
        def inner_fn():
          # [FUNCTION DOC]
          # Inner traced function: assigns the variable its value plus one and returns a read.
          gen_resource_variable_ops.assign_variable_op(v.handle, v + 1)
          return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)

        return inner_fn()

      return fn_with_write()

    self._testVariableWriteInFunctionalOp(build_functional_op,
                                          "StatefulPartitionedCall")

  def testVariableWriteInWhileInInnerFunc(self):
    # [FUNCTION DOC]
    # Scenario: nested traced functions where the inner one runs a one-iteration while loop whose
    # body assigns and reads the variable. Delegates to the shared write helper expecting a
    # stateful partitioned call op.

    def build_functional_op(v):
      # [FUNCTION DOC]
      # Builder: defines the nested traced functions, calls the outer one, returns its result.

      @def_function.function
      def fn_with_write():
        # [FUNCTION DOC]
        # Outer traced function: calls the inner function and returns its result.

        @def_function.function
        def inner_fn():
          # [FUNCTION DOC]
          # Inner traced function: runs the while loop and returns its result.

          def body(_):
            # [FUNCTION DOC]
            # Loop body: assigns the variable its value plus one and returns a read of it.
            gen_resource_variable_ops.assign_variable_op(v.handle, v + 1)
            return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)

          return while_loop.while_loop(
              lambda i: True, body, [0.0], maximum_iterations=1)

        return inner_fn()

      return fn_with_write()

    self._testVariableWriteInFunctionalOp(build_functional_op,
                                          "StatefulPartitionedCall")

  def testVariableWriteInCondInInnerFunc(self):
    # [FUNCTION DOC]
    # Scenario: nested traced functions where the inner one runs a conditional (constant true)
    # whose true branch assigns and reads the variable. Delegates to the shared write helper
    # expecting a stateful partitioned call op.

    def build_functional_op(v):
      # [FUNCTION DOC]
      # Builder: defines the nested traced functions, calls the outer one, returns its result.

      @def_function.function
      def fn_with_write():
        # [FUNCTION DOC]
        # Outer traced function: calls the inner function and returns its result.

        @def_function.function
        def inner_fn():
          # [FUNCTION DOC]
          # Inner traced function: runs the conditional and returns its result.

          def then_branch():
            # [FUNCTION DOC]
            # True branch: assigns the variable its value plus one and returns a read of it.
            gen_resource_variable_ops.assign_variable_op(v.handle, v + 1)
            return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)

          def else_branch():
            # [FUNCTION DOC]
            # False branch: returns a scalar zero of the variable's type.
            return array_ops.zeros([], v.dtype)

          return cond.cond(
              constant_op.constant(True), then_branch, else_branch)

        return inner_fn()

      return fn_with_write()

    self._testVariableWriteInFunctionalOp(build_functional_op,
                                          "StatefulPartitionedCall")

  @test_util.run_v1_only("b/120545219")
  def testCondMustRun(self):
    # [FUNCTION DOC]
    # Scenario (v1 graph mode only): a conditional on a boolean placeholder adds 1 to a variable in
    # the true branch and 4 in the false branch, then the variable is read and marked as a return.
    # Asserts the read gives 5.0 when the predicate is false and 6.0 when true (starting from 1
    # each time the state accumulates), showing the conditional's updates are forced to run.
    with context.graph_mode(), self.cached_session():
      v = resource_variable_ops.ResourceVariable(1.0)
      self.evaluate(variables.global_variables_initializer())
      p = array_ops.placeholder(dtype=dtypes.bool)
      with acd.AutomaticControlDependencies() as c:

        def true_fn():
          # [FUNCTION DOC]
          # True branch: increments the variable by one; returns 0.0.
          v.assign(v + 1)
          return 0.0

        def false_fn():
          # [FUNCTION DOC]
          # False branch: increments the variable by four; returns 1.0.
          v.assign(v + 4)
          return 1.0

        cond.cond(p, true_fn, false_fn)
        val = v.read_value()
        val = c.mark_as_return(val)
      self.assertAllEqual(val.eval(feed_dict={p: False}), 5.0)
      self.assertAllEqual(val.eval(feed_dict={p: True}), 6.0)

  @test_util.run_v1_only("b/120545219")
  def testCondMustRunSeparateRead(self):
    # [FUNCTION DOC]
    # Scenario (v1 graph mode only): same conditional updates as the previous test, but the marked
    # return value is an unrelated constant and the variable is read outside the context. Asserts
    # that running the returned constant with a false predicate leaves the variable at 5.0 and
    # then with a true predicate at 6.0, i.e. the conditional's updates run even though no return
    # value consumes them.
    with context.graph_mode(), self.cached_session():
      v = resource_variable_ops.ResourceVariable(1.0)
      self.evaluate(variables.global_variables_initializer())
      p = array_ops.placeholder(dtype=dtypes.bool)
      with acd.AutomaticControlDependencies() as c:

        def true_fn():
          # [FUNCTION DOC]
          # True branch: increments the variable by one; returns 0.0.
          v.assign(v + 1)
          return 0.0

        def false_fn():
          # [FUNCTION DOC]
          # False branch: increments the variable by four; returns 1.0.
          v.assign(v + 4)
          return 1.0

        cond.cond(p, true_fn, false_fn)
        one = constant_op.constant(1.0)
        one = c.mark_as_return(one)
      one.eval(feed_dict={p: False})
      self.assertAllEqual(v.read_value(), 5.0)
      one.eval(feed_dict={p: True})
      self.assertAllEqual(v.read_value(), 6.0)

  @test_util.run_v1_only("b/120545219")
  def testCondNested(self):
    # [FUNCTION DOC]
    # Scenario (v1 graph mode only): an outer conditional on one placeholder; its true branch adds
    # one to a variable, its false branch contains an inner conditional on a second placeholder
    # that multiplies the variable by two or by three. The final read is marked as a return.
    # Asserts the results for all four predicate combinations: (false,false)=3, (false,true)=6,
    # (true,true)=7, (true,false)=8 (each evaluation sees state starting from the prior value
    # as described by the assertion order).
    with context.graph_mode(), self.cached_session():
      v = resource_variable_ops.ResourceVariable(1.0)
      self.evaluate(variables.global_variables_initializer())
      p = array_ops.placeholder(dtype=dtypes.bool)
      q = array_ops.placeholder(dtype=dtypes.bool)
      with acd.AutomaticControlDependencies() as c:

        def true_fn():
          # [FUNCTION DOC]
          # Outer true branch: adds one to the variable (op named "true"); returns 1.0.
          v.assign(v + 1, name="true")
          return 1.0

        def false_fn():
          # [FUNCTION DOC]
          # Outer false branch: defines and runs an inner conditional on the second placeholder;
          # returns 1.0.

          def inner_true_fn():
            # [FUNCTION DOC]
            # Inner true branch: doubles the variable (op named "false_true"); returns 2.0.
            v.assign(v * 2, name="false_true")
            return 2.0

          def inner_false_fn():
            # [FUNCTION DOC]
            # Inner false branch: triples the variable (op named "false_false"); returns 3.0.
            v.assign(v * 3, name="false_false")
            return 3.0

          cond.cond(q, inner_true_fn, inner_false_fn)
          return 1.0

        cond.cond(p, true_fn, false_fn)
        with ops.name_scope("final"):
          val = v.read_value()
        val = c.mark_as_return(val)
      self.assertAllEqual(val.eval(feed_dict={p: False, q: False}), 3.0)
      self.assertAllEqual(val.eval(feed_dict={p: False, q: True}), 6.0)
      self.assertAllEqual(val.eval(feed_dict={p: True, q: True}), 7.0)
      self.assertAllEqual(val.eval(feed_dict={p: True, q: False}), 8.0)

  @test_util.run_v1_only("b/120545219")
  def testCondOneBranch(self):
    # [FUNCTION DOC]
    # Scenario (v1 graph mode only): a conditional where only the false branch adds four to the
    # variable; then the variable is read and marked as a return. Asserts 5.0 for a false
    # predicate and 5.0 for a true predicate (the second evaluation's expectation shows the
    # update from the first run persists).
    with context.graph_mode(), self.cached_session():
      v = resource_variable_ops.ResourceVariable(1.0)
      self.evaluate(variables.global_variables_initializer())
      p = array_ops.placeholder(dtype=dtypes.bool)
      with acd.AutomaticControlDependencies() as c:

        def true_fn():
          # [FUNCTION DOC]
          # True branch: does nothing to the variable; returns 0.0.
          return 0.0

        def false_fn():
          # [FUNCTION DOC]
          # False branch: adds four to the variable; returns 1.0.
          v.assign(v + 4)
          return 1.0

        cond.cond(p, true_fn, false_fn)
        val = v.read_value()
        val = c.mark_as_return(val)
      self.assertAllEqual(val.eval(feed_dict={p: False}), 5.0)
      self.assertAllEqual(val.eval(feed_dict={p: True}), 5.0)

  @test_util.run_v1_only("b/120545219")
  def testCondOneBranchUpdateBefore(self):
    # [FUNCTION DOC]
    # Scenario (v1 graph mode only): the variable is doubled before a conditional whose false
    # branch adds four; then it is read and marked as a return. Asserts 6.0 for a false predicate
    # and 12.0 for a true predicate, checking the pre-conditional update is ordered before the
    # conditional and the read.
    with context.graph_mode(), self.cached_session():
      v = resource_variable_ops.ResourceVariable(1.0)
      self.evaluate(variables.global_variables_initializer())
      p = array_ops.placeholder(dtype=dtypes.bool)
      with acd.AutomaticControlDependencies() as c:
        v.assign(v * 2)

        def true_fn():
          # [FUNCTION DOC]
          # True branch: does nothing to the variable; returns 0.0.
          return 0.0

        def false_fn():
          # [FUNCTION DOC]
          # False branch: adds four to the variable; returns 1.0.
          v.assign(v + 4)
          return 1.0

        cond.cond(p, true_fn, false_fn)
        val = v.read_value()
        val = c.mark_as_return(val)
      self.assertAllEqual(val.eval(feed_dict={p: False}), 6.0)
      self.assertAllEqual(val.eval(feed_dict={p: True}), 12.0)

  @test_util.run_v1_only("b/120545219")
  def testCondOneBranchUpdateAfter(self):
    # [FUNCTION DOC]
    # Scenario (v1 graph mode only): a conditional whose false branch adds four, followed by
    # doubling the variable and reading it as the marked return. Asserts 10.0 for a false
    # predicate and 20.0 for a true predicate, checking the post-conditional update is ordered
    # after the conditional.
    with context.graph_mode(), self.cached_session():
      v = resource_variable_ops.ResourceVariable(1.0)
      self.evaluate(variables.global_variables_initializer())
      p = array_ops.placeholder(dtype=dtypes.bool)
      with acd.AutomaticControlDependencies() as c:

        def true_fn():
          # [FUNCTION DOC]
          # True branch: does nothing to the variable; returns 0.0.
          return 0.0

        def false_fn():
          # [FUNCTION DOC]
          # False branch: adds four to the variable; returns 1.0.
          v.assign(v + 4)
          return 1.0

        cond.cond(p, true_fn, false_fn)
        v.assign(v * 2)
        val = v.read_value()
        val = c.mark_as_return(val)
      self.assertAllEqual(val.eval(feed_dict={p: False}), 10.0)
      self.assertAllEqual(val.eval(feed_dict={p: True}), 20.0)

  def testFunctionWhileLoopWithCapturedLoopVars(self):
    # [FUNCTION DOC]
    # Scenario: a traced function runs a while loop incrementing a counter and adding one to each
    # element of a captured three-element tensor until the counter reaches three. Asserts the
    # counter ends at 3 and the tensor equals 3, 4, 5.
    n = 3
    x = constant_op.constant(list(range(n)))

    @def_function.function
    def loop():
      # [FUNCTION DOC]
      # Traced function: runs the loop described above and returns the final counter and tensor.
      c = lambda i, x: i < n
      b = lambda i, x: (i + 1, x + 1)
      i, out = while_loop.while_loop(c, b, (0, x))
      return i, out

    i, out = loop()
    self.assertEqual(int(i), 3)
    self.assertAllEqual(out, [3, 4, 5])

  def testDecorator(self):
    # [FUNCTION DOC]
    # Scenario: in graph mode, a function decorated with the automatic control dependency
    # decorator increments a variable, doubles it and returns a read. Asserts calling it
    # evaluates to 4.0.
    with context.graph_mode(), self.cached_session():
      v = resource_variable_ops.ResourceVariable(1.0)
      self.evaluate(variables.global_variables_initializer())

      @acd.automatic_control_dependencies
      def f():
        # [FUNCTION DOC]
        # Decorated function: adds one to the captured variable, doubles it, returns a read.
        v.assign(v + 1)
        v.assign(2 * v)
        return v.read_value()

      self.assertAllEqual(f(), 4.0)

  def testOptimizerInFunction(self):
    # [FUNCTION DOC]
    # Scenario: a traced training function computes the implicit gradient of a squared loss for a
    # variable stored on the test case and applies it with a momentum optimizer (learning rate 1,
    # momentum 1), returning a read. Asserts the returned value is -1.0 (1 minus gradient 2).
    def loss(v):
      # [FUNCTION DOC]
      # Loss: the square of the variable.
      return v**2

    optimizer = momentum.MomentumOptimizer(learning_rate=1.0, momentum=1.0)

    @def_function.function
    def train():
      # [FUNCTION DOC]
      # Traced function: computes the gradient, applies it with the optimizer, returns a read.
      grad = backprop.implicit_grad(loss)(self.v)
      optimizer.apply_gradients(grad)
      return self.v.read_value()

    self.v = resource_variable_ops.ResourceVariable(1.0)
    value = train()
    self.assertEqual(value.numpy(), -1.0)

  def testReturningNonTensorRaisesError(self):
    # [FUNCTION DOC]
    # Scenario: the optimizer's apply-gradients method is wrapped as a traced function, which
    # returns an operation rather than tensors. Asserts that calling it with gradients raises a
    # type error whose message says the function must return zero or more tensors. A TODO notes
    # returning operations might be allowed in the future.
    optimizer = momentum.MomentumOptimizer(learning_rate=1.0, momentum=1.0)
    optimizer.apply_gradients = def_function.function(optimizer.apply_gradients)
    v = resource_variable_ops.ResourceVariable(1.0)
    grad = backprop.implicit_grad(lambda v: v**2)(v)

    with self.assertRaisesRegex(TypeError,
                                ".*must return zero or more Tensors.*"):
      # TODO(akshayka): We might want to allow defun-ing Python functions
      # that return operations (and just execute the op instead of running it).
      optimizer.apply_gradients(grad)

  # TODO(b/111663004): This should work when the outer context is graph
  # building.
  def testOptimizerNonSlotVarsInFunctionNoError(self):
    # [FUNCTION DOC]
    # Scenario: a traced training function applies Adam optimizer gradients to a variable stored on
    # the test case (the optimizer creates non-slot variables inside the function). Asserts only
    # that running it raises no error.
    def loss(v):
      # [FUNCTION DOC]
      # Loss: the square of the variable.
      return v**2

    optimizer = adam.AdamOptimizer(learning_rate=1.0)

    @def_function.function
    def train():
      # [FUNCTION DOC]
      # Traced function: computes the gradient, applies it with the optimizer, returns a read.
      grad = backprop.implicit_grad(loss)(self.v)
      optimizer.apply_gradients(grad)
      return self.v.read_value()

    self.v = resource_variable_ops.ResourceVariable(1.0)
    train()

  def testOptimizerInFunctionWithCapturedVariable(self):
    # [FUNCTION DOC]
    # Scenario: a traced training function applies a momentum optimizer step to a variable it
    # captures from the enclosing scope, using a loss that takes no arguments. Asserts the variable
    # ends at -1.0 after one call.
    v = resource_variable_ops.ResourceVariable(1.0)
    def loss():
      # [FUNCTION DOC]
      # Loss: the square of the captured variable.
      return v**2

    optimizer = momentum.MomentumOptimizer(learning_rate=1.0, momentum=1.0)

    @def_function.function
    def train():
      # [FUNCTION DOC]
      # Traced function: computes the gradient of the loss and applies it; returns nothing.
      grad = backprop.implicit_grad(loss)()
      optimizer.apply_gradients(grad)

    train()
    self.assertEqual(v.numpy(), -1.0)

  def testRepeatedResourceInput(self):
    # [FUNCTION DOC]
    # Scenario: an inner traced function takes two resource handles and sums reads of both; the
    # outer function passes the same variable handle twice. Asserts the call evaluates to 2.0 and
    # does not fail (an op receiving the same resource twice must not depend on itself).
    var = resource_variable_ops.ResourceVariable(1.0)

    @def_function.function
    def inner(var1, var2):
      # [FUNCTION DOC]
      # Traced function: returns the sum of reads of the two given resource handles.
      return (resource_variable_ops.read_variable_op(var1, dtypes.float32) +
              resource_variable_ops.read_variable_op(var2, dtypes.float32))

    @def_function.function
    def outer():
      # [FUNCTION DOC]
      # Traced function: calls the inner function with the variable handle as both arguments.
      return inner(var.handle, var.handle)

    self.assertEqual(self.evaluate(outer()), 2.0)

  def testManualControlDepMonitoringAttrNotAdded(self):
    # [FUNCTION DOC]
    # Scenario: two reads then an assign on one variable inside the context (all control edges
    # added automatically). Asserts the assign has both reads as control inputs, and that none of
    # the three ops carries the attribute marking manual control dependencies (reading it raises a
    # value error).
    with context.graph_mode(), self.cached_session():
      v = resource_variable_ops.ResourceVariable(1.0)
      self.evaluate(variables.global_variables_initializer())
      with acd.AutomaticControlDependencies():
        read_op1 = gen_resource_variable_ops.read_variable_op(
            v.handle, v.dtype).op
        read_op2 = gen_resource_variable_ops.read_variable_op(
            v.handle, v.dtype).op
        assign_op = gen_resource_variable_ops.assign_variable_op(
            v.handle, v + 1)
      # Writes should have control deps automatically added from "all" reads
      # since last write or start of the code block.
      self.assertIn(read_op1, assign_op.control_inputs)
      self.assertIn(read_op2, assign_op.control_inputs)
      # But, we shouldn't add the monitoring attribute in this case.
      with self.assertRaises(ValueError):
        assign_op.get_attr("_has_manual_control_dependencies")
      with self.assertRaises(ValueError):
        read_op1.get_attr("_has_manual_control_dependencies")
      with self.assertRaises(ValueError):
        read_op2.get_attr("_has_manual_control_dependencies")


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
