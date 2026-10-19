# [MODULE DOC]
# Unit tests for the critical section op wrappers (CriticalSection and its execute method).
# Covers concurrent execution with mutual exclusion on shared variables, use inside conditionals,
# loops and dataset map functions, deadlock avoidance for recursive or captured access, error
# propagation, graph collections (TF1 only) and exclusive resource access checks.
# Defines one test class and runs the TensorFlow test main when executed as a script.
# No import-time side effects beyond imports.
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
"""critical section tests."""

import itertools

from absl.testing import parameterized

from tensorflow.python.data.experimental.ops import prefetching_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_v2_toggles
from tensorflow.python.ops import critical_section_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
# TODO(ebrevdo): Re-enable once CriticalSection is in core.
# from tensorflow.python.training import saver as saver_lib


@test_util.with_control_flow_v2
class CriticalSectionTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case (also parameterized) covering CriticalSection behavior in graph and eager modes,
  # with control flow v2 enabled by the class decorator. Tests share no fixtures; each builds its
  # own critical section and resource variable.

  @test_util.run_in_graph_and_eager_modes
  def testCreateCriticalSection(self):
    # [FUNCTION DOC]
    # Runs many executions of a read-then-add function through one critical section and checks
    # that they were serialized: the values read, once sorted, are exactly the arithmetic
    # progression of partial sums. Runs in graph and eager modes. Takes no inputs, returns nothing.
    cs = critical_section_ops.CriticalSection(shared_name="cs")
    v = resource_variable_ops.ResourceVariable(0.0, name="v")

    def fn(a, b):
      # [FUNCTION DOC]
      # HELPER: body executed inside the critical section by testCreateCriticalSection.
      # Reads the variable, then (after the read) adds the product of a and b to it, and returns
      # a copy of the value that was read. Inputs are two numbers; output is the old value.
      c = v.value()
      with ops.control_dependencies([c]):
        nv = v.assign_add(a * b)
        with ops.control_dependencies([nv]):
          return array_ops.identity(c)

    num_concurrent = 100
    r = [cs.execute(lambda: fn(1.0, 2.0)) for _ in range(num_concurrent)]
    self.evaluate(v.initializer)
    r_value = self.evaluate(r)
    self.assertAllClose([2.0 * i for i in range(num_concurrent)],
                        sorted(r_value))

  @parameterized.named_parameters(
      ("Inner%sOuter%s" % (inner, outer), inner, outer)
      for (inner, outer) in itertools.product(*([(False, True)] * 2)))
  @test_util.run_in_graph_and_eager_modes
  @test_util.xla_allow_fallback("b/128495870")
  def testCriticalSectionWithControlFlow(self, outer_cond, inner_cond):
    # [FUNCTION DOC]
    # Parameterized over all four combinations of two booleans. Many critical-section executions
    # are placed inside an outer conditional whose predicate is outer_cond, and the executed
    # function itself contains an inner conditional on inner_cond. Asserts that when both are true
    # the serialized reads form the expected progression, otherwise every result is zero.
    # Skipped in graph mode when control flow v2 is enabled. Returns nothing.
    if (not context.executing_eagerly() and
        control_flow_v2_toggles.control_flow_v2_enabled()):
      self.skipTest("b/135070612")
    cs = critical_section_ops.CriticalSection(shared_name="cs")
    v = resource_variable_ops.ResourceVariable(0.0, name="v")
    num_concurrent = 100

    # pylint: disable=cell-var-from-loop
    def fn(a, b):
      # [FUNCTION DOC]
      # HELPER: function run inside the critical section by the control-flow test.
      # Reads the variable, then conditionally (on the captured inner predicate) adds a times b to
      # it; returns the value read, or the plain read when the predicate is false.
      c = v.read_value()
      def true_fn():
        # [FUNCTION DOC]
        # HELPER: true branch of the inner conditional. After the read, adds the product of the
        # enclosing a and b to the variable and returns a copy of the earlier read value.
        with ops.control_dependencies([c]):
          nv = v.assign_add(a * b)
          with ops.control_dependencies([nv]):
            return array_ops.identity(c)
      return cond.cond(
          array_ops.identity(inner_cond), true_fn, lambda: c)

    def execute():
      # [FUNCTION DOC]
      # HELPER: branch taken when the outer predicate is true; runs fn with fixed inputs one and
      # two through the critical section and returns the result tensor.
      return cs.execute(lambda: fn(1.0, 2.0))

    r = [
        cond.cond(array_ops.identity(outer_cond),
                  execute,
                  v.read_value)
        for _ in range(num_concurrent)
    ]
    # pylint: enable=cell-var-from-loop

    self.evaluate(v.initializer)
    r_value = self.evaluate(r)
    if inner_cond and outer_cond:
      self.assertAllClose([2.0 * i for i in range(num_concurrent)],
                          sorted(r_value))
    else:
      self.assertAllClose([0] * num_concurrent, r_value)

  @test_util.run_v1_only("b/123990562 Sees CancelledError on some calls")
  def testCriticalSectionInParallelDoesntDeadlockOnError(self):
    # [FUNCTION DOC]
    # Graph-mode only. Runs two parallel critical-section executions, one of which fails an
    # assertion (odd/even index), a hundred times, and asserts each run raises an op error
    # containing "Error" instead of deadlocking. Returns nothing.
    # No eager mode execution of this test because eager does not
    # run fn() in parallel, which is where the deadlock could
    # potentially occur (in graph mode).
    cs = critical_section_ops.CriticalSection(shared_name="cs")
    v = resource_variable_ops.ResourceVariable(0.0, name="v")

    def fn(i):
      # [FUNCTION DOC]
      # HELPER: executed in the critical section; asserts that the integer i is odd (failing with
      # the message "Error" otherwise) and then returns a read of the variable after the assert.
      error = control_flow_assert.Assert((i % 2) == 1, ["Error"])
      with ops.control_dependencies([error]):
        return v.read_value()

    num_concurrent = 2

    @def_function.function(autograph=False)
    def run_concurrently():
      # [FUNCTION DOC]
      # HELPER: compiled function (autograph off) that launches num_concurrent critical-section
      # executions of fn and returns the list of their results.
      return [cs.execute(lambda: fn(i)) for i in range(num_concurrent)]

    if not context.executing_eagerly():
      run_concurrently = run_concurrently()

    self.evaluate(v.initializer)
    for _ in range(100):
      with self.assertRaisesOpError("Error"):
        if context.executing_eagerly():
          run_concurrently()
        else:
          self.evaluate(run_concurrently)

  @test_util.run_in_graph_and_eager_modes
  def testCreateCriticalSectionFnReturnsOp(self):
    # [FUNCTION DOC]
    # Checks that a function which returns an operation (no tensor) can be executed in a critical
    # section: a hundred executions each add two to the variable and the final value must equal
    # two times the count. Graph and eager modes. Returns nothing.
    cs = critical_section_ops.CriticalSection(shared_name="cs")
    v = resource_variable_ops.ResourceVariable(0.0, name="v")

    def fn_return_op(a, b):
      # [FUNCTION DOC]
      # HELPER: reads the variable, then adds a times b to it and returns a no-op that depends
      # on the update, rather than a tensor.
      c = v.read_value()
      with ops.control_dependencies([c]):
        nv = v.assign_add(a * b)
        with ops.control_dependencies([nv]):
          return control_flow_ops.no_op()

    num_concurrent = 100
    r = [cs.execute(lambda: fn_return_op(1.0, 2.0))
         for _ in range(num_concurrent)]
    self.evaluate(v.initializer)
    self.evaluate(r)
    final_v = self.evaluate(v)
    self.assertAllClose(2.0 * num_concurrent, final_v)

  @test_util.run_v1_only("Collections don't exist in TF2")
  def testCollection(self):
    # [FUNCTION DOC]
    # Graph-mode (TF1) only. Verifies that a new CriticalSection is registered in the critical
    # sections graph collection and that the lock operation created by execute (found by name and
    # type) appears among the signatures in the critical-section-executions collection.
    cs = critical_section_ops.CriticalSection(shared_name="cs")
    self.assertIn(
        cs, ops.get_collection(critical_section_ops.CRITICAL_SECTIONS))
    add = lambda x: x + 1
    execute = cs.execute(lambda: add(1.0), name="my_execute")
    execute_op = [
        x for x in execute.graph.get_operations()
        if "my_execute" in x.name and "MutexLock" in x.type
    ][0]
    self.assertIn(
        execute_op,
        [signature.op for signature in
         ops.get_collection(critical_section_ops.CRITICAL_SECTION_EXECUTIONS)])

  def testRecursiveCriticalSectionAccessIsIllegal(self):
    # [FUNCTION DOC]
    # Asserts that executing on a critical section from inside a function already running in that
    # same section raises a value error about attempting to lock a section already held.
    # Returns nothing.
    # This does not work properly in eager mode.  Eager users will
    # just hit a deadlock if they do this.  But at least it'll be easier
    # to debug.
    cs = critical_section_ops.CriticalSection()
    add = lambda y: y + 1
    def fn(x):
      # [FUNCTION DOC]
      # HELPER: inner function that re-enters the same critical section with an add-one function,
      # triggering the illegal recursive access. Input is a number; returns the nested result.
      return cs.execute(lambda: add(x))

    with self.assertRaisesRegex(
        ValueError, r"Attempting to lock a CriticalSection .* in which we are"):
      cs.execute(lambda: fn(1.0))

  def testRecursiveCriticalSectionAccessViaCapturedTensorIsProtected(self):
    # [FUNCTION DOC]
    # Verifies that executions capturing tensors produced by earlier executions of the same
    # critical section (directly, via identity copies, or with explicit control dependencies) do
    # not deadlock because execute adds control dependencies on captured ops. Evaluates four
    # results and asserts each equals two. Returns nothing.
    # This one is subtle; and we're being overly cautious here.  The
    # deadlock we are ensuring we catch is:
    #
    # to_capture = CS[lambda x: x + 1](1.0)
    # deadlocked = CS[lambda x: x + to_capture](1.0)
    #
    # This would have caused a deadlock because executing `deadlocked` will
    # lock the mutex on CS; but then due to dependencies, will attempt
    # to compute `to_capture`.  This computation requires locking CS,
    # but that is not possible now because CS is already locked by
    # `deadlocked`.
    #
    # We check that CriticalSection.execute properly inserts new
    # control dependencies to its lock to ensure all captured
    # operations are finished before anything runs within the critical section.
    cs = critical_section_ops.CriticalSection(shared_name="cs")
    fn = array_ops.identity
    to_capture = cs.execute(lambda: fn(1.0))
    fn_captures = lambda x: x + to_capture
    to_capture_too = array_ops.identity(to_capture)

    ex_0 = cs.execute(lambda: fn_captures(1.0))

    with ops.control_dependencies([to_capture]):
      # This is OK because to_capture will execute before this next call
      ex_1 = cs.execute(lambda: fn_captures(1.0))

    dependency = array_ops.identity(to_capture)

    fn_captures_dependency = lambda x: x + dependency

    ex_2 = cs.execute(lambda: fn_captures_dependency(1.0))

    with ops.control_dependencies([to_capture_too]):
      ex_3 = cs.execute(lambda: fn_captures_dependency(1.0))

    # Ensure there's no actual deadlock on to_execute.
    self.assertEqual(2.0, self.evaluate(ex_0))
    self.assertEqual(2.0, self.evaluate(ex_1))
    self.assertEqual(2.0, self.evaluate(ex_2))
    self.assertEqual(2.0, self.evaluate(ex_3))

  def testRecursiveCriticalSectionAccessWithinLoopIsProtected(self):
    # [FUNCTION DOC]
    # Runs three while loops (1000 iterations, 25 parallel) whose bodies call the critical
    # section with the loop variable captured implicitly, captured with a manual control
    # dependency, or passed as an argument. Logs start/finish markers and asserts each loop ends
    # with both counters at 1000, i.e. no deadlock. Returns nothing.
    cs = critical_section_ops.CriticalSection(shared_name="cs")

    def body_implicit_capture(i, j):
      # [FUNCTION DOC]
      # HELPER: loop body. Increments i and returns it with the result of executing, in the
      # critical section, a function that implicitly captures j and adds one.
      # This would have caused a deadlock if not for logic in execute
      # that inserts additional control dependencies onto the lock op:
      #   * Loop body argument j is captured by fn()
      #   * i is running in parallel to move forward the execution
      #   * j is not being checked by the predicate function
      #   * output of cs.execute() is returned as next j.
      fn = lambda: j + 1
      return (i + 1, cs.execute(fn))

    (i_n, j_n) = while_loop.while_loop(
        lambda i, _: i < 1000,
        body_implicit_capture, [0, 0],
        parallel_iterations=25)
    # For consistency between eager and graph mode.
    i_n = array_ops.identity(i_n)
    logging.warn(
        "\n==============\nRunning "
        "'testRecursiveCriticalSectionAccessWithinLoopDoesNotDeadlock "
        "body_implicit_capture'\n"
        "==============\n")
    self.assertEqual((1000, 1000), self.evaluate((i_n, j_n)))
    logging.warn(
        "\n==============\nSuccessfully finished running "
        "'testRecursiveCriticalSectionAccessWithinLoopDoesNotDeadlock "
        "body_implicit_capture'\n"
        "==============\n")

    def body_implicit_capture_protected(i, j):
      # [FUNCTION DOC]
      # HELPER: loop body like the implicit-capture one, but wraps the execute call in a control
      # dependency on j so the captured value is safely ordered. Returns incremented i and the
      # execute result.
      # This version is ok because we manually add a control
      # dependency on j, which is an argument to the while_loop body
      # and captured by fn.
      fn = lambda: j + 1
      with ops.control_dependencies([j]):
        return (i + 1, cs.execute(fn))

    (i_n, j_n) = while_loop.while_loop(
        lambda i, _: i < 1000,
        body_implicit_capture_protected, [0, 0],
        parallel_iterations=25)
    # For consistency between eager and graph mode.
    i_n = array_ops.identity(i_n)
    logging.warn(
        "\n==============\nRunning "
        "'testRecursiveCriticalSectionAccessWithinLoopDoesNotDeadlock "
        "body_implicit_capture_protected'\n"
        "==============\n")
    self.assertEqual((1000, 1000), self.evaluate((i_n, j_n)))
    logging.warn(
        "\n==============\nSuccessfully finished running "
        "'testRecursiveCriticalSectionAccessWithinLoopDoesNotDeadlock "
        "body_implicit_capture_protected'\n"
        "==============\n")

    def body_args_capture(i, j):
      # [FUNCTION DOC]
      # HELPER: loop body that passes j as an explicit argument to the function run in the
      # critical section. Returns incremented i and the execute result.
      # This version is ok because j is an argument to fn and we can
      # ensure there's a control dependency on j.
      fn = lambda x: x + 1
      return (i + 1, cs.execute(lambda: fn(j)))

    (i_n, j_n) = while_loop.while_loop(
        lambda i, _: i < 1000,
        body_args_capture, [0, 0],
        parallel_iterations=25)
    # For consistency between eager and graph mode.
    i_n = array_ops.identity(i_n)
    logging.warn(
        "\n==============\nRunning "
        "'testRecursiveCriticalSectionAccessWithinLoopDoesNotDeadlock "
        "body_args_capture'\n"
        "==============\n")
    self.assertEqual((1000, 1000), self.evaluate((i_n, j_n)))
    logging.warn(
        "\n==============\nSuccessfully finished running "
        "'testRecursiveCriticalSectionAccessWithinLoopDoesNotDeadlock "
        "body_args_capture'\n"
        "==============\n")

  def testRecursiveCriticalSectionAccessIsIllegalSameSharedName(self):
    # [FUNCTION DOC]
    # Like the plain recursion test, but the inner access uses a second CriticalSection object
    # created with the same shared name. Asserts a value error about locking a section already
    # held is raised. Returns nothing.
    # This does not work properly in eager mode.  Eager users will
    # just hit a deadlock if they do this.  But at least it'll be easier
    # to debug.
    cs = critical_section_ops.CriticalSection(shared_name="cs")
    cs_same = critical_section_ops.CriticalSection(shared_name="cs")
    add = lambda x: x + 1
    def fn(x):
      # [FUNCTION DOC]
      # HELPER: inner function that executes an add-one function on the second critical section
      # object sharing the same name; input is a number, returns the nested result.
      return cs_same.execute(lambda: add(x))

    with self.assertRaisesRegex(
        ValueError, r"Attempting to lock a CriticalSection .* in which we are"):
      cs.execute(lambda: fn(1.0))

  @test_util.run_v1_only(
      "b/123955885 Can't identify consumed resources in eager mode")
  def testMultipleCSExecutionsRequestSameResource(self):
    # [FUNCTION DOC]
    # Graph-mode only. Checks exclusive resource access rules: one section may reuse a variable,
    # a different section touching it raises a value error ("requested exclusive resource
    # access") whether or not the second call asks for exclusivity, two sections may share a
    # resource when neither is exclusive, but raise if the later one requires exclusivity.
    cs0 = critical_section_ops.CriticalSection()
    cs1 = critical_section_ops.CriticalSection()
    v = resource_variable_ops.ResourceVariable(0.0, name="v")
    cs0.execute(lambda: v + 1)
    # It's OK for the same CriticalSection to access this resource.
    cs0.execute(lambda: v - 1)
    # It's *not* OK for a different CriticalSection to access it by
    # default.
    with self.assertRaisesRegex(ValueError,
                                "requested exclusive resource access"):
      cs1.execute(lambda: v + 1)
    # It's not even OK if the second call doesn't request exclusive access.
    with self.assertRaisesRegex(ValueError,
                                "requested exclusive resource access"):
      cs1.execute(lambda: v + 1, exclusive_resource_access=False)

    v2 = resource_variable_ops.ResourceVariable(0.0, name="v2")
    cs0.execute(lambda: v2 + 1, exclusive_resource_access=False)
    # It's OK if neither requests exclusive resource access.
    cs1.execute(lambda: v2 + 1, exclusive_resource_access=False)

    # It's not OK if the second request requires exclusive resource
    # access.
    with self.assertRaisesRegex(ValueError,
                                "requested exclusive resource access"):
      cs1.execute(lambda: v2 + 1)

  def testControlDependencyFromOutsideWhileLoopMixedWithInsideLoop(self):
    # [FUNCTION DOC]
    # Runs a ten-iteration while loop whose body executes in a critical section a function
    # reading a resource variable created outside the loop, ensuring outside control dependencies
    # do not break the lock op's automatic dependency handling. Asserts the result is ten.
    cs = critical_section_ops.CriticalSection()
    v = resource_variable_ops.ResourceVariable(0, name="v")
    # Make sure that the control dependencies on v do not cause issues
    # in the lock_op's automatic control dependency adder.
    #
    # Note, here v must be a resource variable (or something similar),
    # otherwise it gets hoisted into the while_loop by the time we add
    # control dependencies to the lock_op.
    def body(i):
      # [FUNCTION DOC]
      # HELPER: while-loop body; executes in the critical section a function adding the outer
      # variable plus the loop value plus one, and returns that result as the next loop value.
      add_j = lambda j: v + j + 1
      return cs.execute(lambda: add_j(i))

    out = while_loop.while_loop(lambda i: i < 10, body, [0])
    self.evaluate(v.initializer)
    self.assertEqual(10, self.evaluate(out))

  @test_util.run_in_graph_and_eager_modes
  def testInsideFunction(self):
    # [FUNCTION DOC]
    # Uses a critical section inside a dataset map function (a TensorFlow function), placing the
    # variable on GPU when available (the test is skipped on GPU because of a colocation bug).
    # Fetches the first dataset element and asserts it equals one. Graph and eager modes.
    if test_util.is_gpu_available():
      self.skipTest(
          "b/123899495: Colocation errors for critical sections in map on GPU")
    cs = critical_section_ops.CriticalSection()
    with ops.device("/gpu:0" if test_util.is_gpu_available() else "/cpu:0"):
      v = resource_variable_ops.ResourceVariable(1)
    def fn():
      # [FUNCTION DOC]
      # HELPER: function run in the critical section by the dataset map; returns a read of the
      # variable. Takes no inputs.
      return v.read_value()

    # map() creates a TensorFlow function.
    ds = dataset_ops.Dataset.range(1)
    if test_util.is_gpu_available():
      ds = (ds.apply(prefetching_ops.copy_to_device("/gpu:0"))
            .apply(prefetching_ops.map_on_gpu(lambda _: cs.execute(fn))))
    else:
      ds = ds.map(lambda _: cs.execute(fn))

    def get_first():
      # [FUNCTION DOC]
      # HELPER: returns the first element of the dataset as a concrete value. In eager mode uses
      # a one-shot iterator; in graph mode creates an initializable iterator, runs it and the
      # variable initializers, then evaluates the next element.
      if context.executing_eagerly():
        return self.evaluate(dataset_ops.make_one_shot_iterator(ds).get_next())
      itr = dataset_ops.make_initializable_iterator(ds)
      self.evaluate([v.initializer, itr.initializer])
      return self.evaluate(itr.get_next())

    self.assertEqual(1, get_first())


if __name__ == "__main__":
  test.main()
