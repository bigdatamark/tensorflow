# [MODULE DOC]
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

"""Tests for cond_v2."""

import os

from absl.testing import parameterized

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import remote
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_ops
from tensorflow.python.framework import test_util
from tensorflow.python.module import module as module_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import cond_v2
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import gen_optional_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import test
from tensorflow.python.saved_model import load as load_lib
from tensorflow.python.saved_model import save as save_lib
from tensorflow.python.training import saver
from tensorflow.python.util import compat


_OPTIONAL_OPS = frozenset([
    "OptionalFromValue", "OptionalNone", "OptionalHasValue", "OptionalGetValue"
])


class CondV2Test(test.TestCase):
  # [CLASS DOC]
  # Main test class for cond_v2. Holds most behavioral tests: output equality and gradient equality
  # against the legacy cond via a shared helper, nesting of cond and function calls, gradients taken
  # inside and outside functions, stateful versus stateless branch handling and control dependencies,
  # lowering and XLA behavior, structured outputs, tensor arrays inside conds, forward-pass rewriting
  # for gradients, and optional values. Has no per-instance state; uses the standard test base class.

  def _testCond(self, true_fn, false_fn, train_vals, feed_dict=None):
    # [FUNCTION DOC]
    # HELPER: Shared checker used by many tests in this class to compare cond_v2 against the legacy cond.
    # Builds a boolean placeholder predicate, then creates two conds from the same branch callables: the
    # legacy cond (the predicate squeezed to a scalar) and cond_v2. Also computes gradients of both with
    # respect to the given list of variables. Runs a session four times with the predicate set to true,
    # a nested true value, false and a nested false value (nested means a one-by-one matrix), merging
    # in the optional extra feed mapping each time. Asserts the cond outputs are equal and the gradient
    # values are equal. Inputs: true_fn and false_fn (branch callables), train_vals (tensors to
    # differentiate with respect to), feed_dict (optional mapping of extra feeds, defaults to empty).
    # Returns nothing; fails with an assertion error on mismatch. Uses the default graph.
    if not feed_dict:
      feed_dict = {}
    with self.session(graph=ops.get_default_graph()) as sess:
      pred = array_ops.placeholder(dtypes.bool, name="pred")

      expected = tf_cond.cond(
          array_ops.squeeze_v2(pred), true_fn, false_fn, name="expected")
      actual = cond_v2.cond_v2(pred, true_fn, false_fn, name="actual")

      expected_grad = gradients_impl.gradients(expected, train_vals)
      actual_grad = gradients_impl.gradients(actual, train_vals)

      sess_run_args = {pred: True}
      sess_run_args.update(feed_dict)
      expected_val, actual_val, expected_grad_val, actual_grad_val = sess.run(
          (expected, actual, expected_grad, actual_grad), sess_run_args)
      self.assertEqual(expected_val, actual_val)
      self.assertEqual(expected_grad_val, actual_grad_val)

      sess_run_args = {pred: [[True]]}
      sess_run_args.update(feed_dict)
      expected_val, actual_val, expected_grad_val, actual_grad_val = sess.run(
          (expected, actual, expected_grad, actual_grad), sess_run_args)
      self.assertEqual(expected_val, actual_val)
      self.assertEqual(expected_grad_val, actual_grad_val)

      sess_run_args = {pred: False}
      sess_run_args.update(feed_dict)
      expected_val, actual_val, expected_grad_val, actual_grad_val = sess.run(
          (expected, actual, expected_grad, actual_grad), sess_run_args)
      self.assertEqual(expected_val, actual_val)
      self.assertEqual(expected_grad_val, actual_grad_val)

      sess_run_args = {pred: [[False]]}
      sess_run_args.update(feed_dict)
      expected_val, actual_val, expected_grad_val, actual_grad_val = sess.run(
          (expected, actual, expected_grad, actual_grad), sess_run_args)
      self.assertEqual(expected_val, actual_val)
      self.assertEqual(expected_grad_val, actual_grad_val)

  @test_util.run_deprecated_v1
  def testBasic(self):
    # [FUNCTION DOC]
    # Test: builds a cond with simple scalar branches (multiply x by two, multiply y by three) and
    # calls the comparison helper three times, differentiating with respect to x, to both x and y, and
    # to y. Expects cond_v2 values and gradients to equal the legacy cond. Runs in graph mode only.
    x = constant_op.constant(1.0, name="x")
    y = constant_op.constant(2.0, name="y")

    def true_fn():
      # [FUNCTION DOC]
      # Test branch callable: returns the captured tensor x multiplied by two. Used as the true branch.
      return x * 2.0

    def false_fn():
      # [FUNCTION DOC]
      # Test branch callable: returns the captured tensor y multiplied by three. Used as the false branch.
      return y * 3.0

    self._testCond(true_fn, false_fn, [x])
    self._testCond(true_fn, false_fn, [x, y])
    self._testCond(true_fn, false_fn, [y])

  def testReturnsIndexedSlicesAndNones(self):
    # [FUNCTION DOC]
    # Test: inside a traced function, builds a cond_v2 whose both branches return an indexed-slices
    # value (made from a constant vector) together with a null value. Asserts the null second output
    # stays null and, after converting the first output to a dense tensor, that the true branch value
    # (one) is produced. Verifies cond_v2 supports indexed-slices outputs and null outputs.
    @def_function.function
    def build_cond_with_indexed_slices():
      # [FUNCTION DOC]
      # Helper traced function for the test above: creates a constant true predicate, defines two branches
      # returning an indexed-slices value plus a null value, runs cond_v2, asserts the second result is
      # null and returns the first result converted to a dense tensor.
      pred = constant_op.constant(True)
      def true_fn():
        # [FUNCTION DOC]
        # Branch callable: returns a pair of an indexed-slices version of the constant one and a null value.
        return math_ops._as_indexed_slices(constant_op.constant([1.])), None
      def false_fn():
        # [FUNCTION DOC]
        # Branch callable: returns a pair of an indexed-slices version of the constant two and a null value.
        return math_ops._as_indexed_slices(constant_op.constant([2.])), None
      result = cond_v2.cond_v2(pred, true_fn, false_fn)
      self.assertIsNone(result[1])
      return ops.convert_to_tensor(result[0])
    output = build_cond_with_indexed_slices()
    self.assertAllEqual(output, [1.])

  def testReturnsNonesAndIndexedSlices(self):
    # [FUNCTION DOC]
    # Test: like the previous indexed-slices test but the nulls come first. The branches return three
    # null values followed by an indexed-slices value. Asserts the first three results are null and
    # that the converted fourth result equals the true-branch value (one).

    @def_function.function
    def build_cond_with_indexed_slices():
      # [FUNCTION DOC]
      # Helper traced function for the test above: builds a constant true predicate and a cond_v2 whose
      # branches return three nulls and an indexed-slices value, asserts the first three outputs are
      # null, and returns the fourth converted to a dense tensor.
      pred = constant_op.constant(True)

      def true_fn():
        # [FUNCTION DOC]
        # Branch callable: returns three null values followed by an indexed-slices version of constant one.
        return (None, None, None,
                math_ops._as_indexed_slices(constant_op.constant([1.])))

      def false_fn():
        # [FUNCTION DOC]
        # Branch callable: returns three null values followed by an indexed-slices version of constant two.
        return (None, None, None,
                math_ops._as_indexed_slices(constant_op.constant([2.])))

      result = cond_v2.cond_v2(pred, true_fn, false_fn)
      self.assertIsNone(result[0])
      self.assertIsNone(result[1])
      self.assertIsNone(result[2])
      return ops.convert_to_tensor(result[3])

    output = build_cond_with_indexed_slices()
    self.assertAllEqual(output, [1.])

  def testCondNestedFunctionGradientWithSavedModel(self):
    # [FUNCTION DOC]
    # Test: checks that gradients through nested functions inside a cond_v2 survive saving and loading
    # a saved model. Defines a module whose method calls cond_v2 with two nested function branches, the
    # true branch doing an einsum whose gradient must be found by the Python gradient registry. Computes
    # the gradient of the output with respect to the module variable and expects all twos in the
    # two-by-two result. Saves the module into a temporary directory, loads it, repeats the gradient
    # and expects the same result. Side effects: writes a temporary saved model on disk.
    class Model(module_lib.Module):
      # [CLASS DOC]
      # Local module class used by the saved-model gradient test. Holds one two-by-two resource variable
      # of ones and has a traced method that runs cond_v2 on a condition. Subclass of the framework
      # module base class so it can be saved and loaded.

      def __init__(self):
        # [FUNCTION DOC]
        # Constructor: creates and stores the resource variable of ones (two by two) as the attribute v.
        # Takes no arguments besides self and does not call the parent constructor.
        self.v = resource_variable_ops.ResourceVariable([[1., 1.], [1., 1.]])

      @def_function.function
      def call(self, x, cond):
        # [FUNCTION DOC]
        # Traced method: given an input matrix x and a scalar cond tensor, returns cond_v2 selecting between
        # two nested traced functions based on whether cond is greater than zero. The true function returns
        # the einsum of x with the variable; the false function returns x unchanged.

        @def_function.function
        def true_fn():
          # [FUNCTION DOC]
          # Nested traced function: returns the einsum of the captured input x and the module variable. The
          # comment notes einsum has no symbolic gradient, so loading from a saved model must keep the
          # Python gradient function reachable.
          # Einsum doesn't have a symbolic gradient op registered.
          # Taking gradient of an einsum op will fail if its python gradient
          # function is not found after loaded from a SavedModel.
          return gen_linalg_ops.einsum([x, self.v], "ab,bc->ac")

        @def_function.function
        def false_fn():
          # [FUNCTION DOC]
          # Nested traced function: returns the captured input x unchanged.
          return x

        return cond_v2.cond_v2(cond > 0, true_fn, false_fn)

    model = Model()
    x = constant_op.constant([[1., 1.], [1., 1.]])
    cond = constant_op.constant(1.)
    with backprop.GradientTape() as tape:
      y = tape.gradient(model.call(x, cond), model.v)

    self.assertAllEqual(y, [[2., 2.], [2., 2.]])

    saved_model_dir = os.path.join(self.create_tempdir(), "saved_model")
    save_lib.save(model, saved_model_dir)
    loaded_model = load_lib.load(saved_model_dir)
    with backprop.GradientTape() as tape:
      y = tape.gradient(loaded_model.call(x, cond), loaded_model.v)

    self.assertAllEqual(y, [[2., 2.], [2., 2.]])

  def testCondNestedFunctionGradientWithXlaDynamicCondition(self):
    # [FUNCTION DOC]
    # Test: checks gradient of a cond_v2 under XLA compilation with a dynamically shaped input. Creates a
    # two-by-two resource variable, then a compiled traced function with an input signature of a matrix
    # with unknown first dimension and a scalar condition, using cond_v2 over two nested functions.
    # Computes the gradient with respect to the variable with a two-row input and expects all twos, then
    # with a three-row input (forcing recompilation) and expects all threes. Intended to confirm XLA
    # handles the bounded dynamic shape.

    v = resource_variable_ops.ResourceVariable([[1., 1.], [1., 1.]])

    @def_function.function(
        jit_compile=True,
        input_signature=[
            tensor_spec.TensorSpec([None, 2]),
            tensor_spec.TensorSpec([]),
        ],
    )
    def f(x, cond):
      # [FUNCTION DOC]
      # Traced (XLA compiled) function under test: takes matrix x and scalar condition cond, and returns
      # cond_v2 selecting between a nested function computing einsum of x with the captured variable v and
      # a nested function returning x, depending on whether cond is positive.

      @def_function.function
      def true_fn():
        # [FUNCTION DOC]
        # Nested traced function: returns the einsum of the captured input x and the captured variable v.
        return gen_linalg_ops.einsum([x, v], "ab,bc->ac")

      @def_function.function
      def false_fn():
        # [FUNCTION DOC]
        # Nested traced function: returns the captured input x unchanged.
        return x

      return cond_v2.cond_v2(cond > 0, true_fn, false_fn)

    x = constant_op.constant([[1., 1.], [1., 1.]])
    cond = constant_op.constant(1.)
    with backprop.GradientTape() as tape:
      # Shape of x in HLO graph should be [<=2, 2].
      y = tape.gradient(f(x, cond), v)

    self.assertAllEqual(y, [[2., 2.], [2., 2.]])

    x = constant_op.constant([[1., 1.], [1., 1.], [1., 1.]])
    with backprop.GradientTape() as tape:
      # HLO graph should be re-compiled to handle x with shape [<=3, 2].
      y = tape.gradient(f(x, cond), v)

    self.assertAllEqual(y, [[3., 3.], [3., 3.]])

  def testExternalControlDependencies(self):
    # [FUNCTION DOC]
    # Test: checks external control dependencies are honored inside a cond_v2 branch. In a fresh graph,
    # creates a variable initialized to one and an assign-add operation. The true branch runs under a
    # control dependency on that operation and returns a constant. Evaluates cond_v2 with a predicate
    # placeholder defaulting to false, but expects after evaluation that the variable equals two,
    # meaning the dependency executed even though the true branch was not selected. Uses a test session.
    with ops.Graph().as_default(), self.test_session():
      v = variables.Variable(1.0)
      self.evaluate(v.initializer)
      op = v.assign_add(1.0)

      def true_branch():
        # [FUNCTION DOC]
        # Branch callable: enters a control dependency on the captured assign-add operation and returns the
        # float one. Used as the true branch to test external control dependencies.
        with ops.control_dependencies([op]):
          return 1.0

      cond_v2.cond_v2(array_ops.placeholder_with_default(False, None),
                      true_branch,
                      lambda: 2.0).eval()
      self.assertAllEqual(self.evaluate(v), 2.0)

  @test_util.run_deprecated_v1
  def testMultipleOutputs(self):
    # [FUNCTION DOC]
    # Test: cond_v2 with branches returning pairs of tensors (one branch returns a product and y, the
    # other x and y times three). Uses the comparison helper three times with gradients with respect to
    # x, both, and y. Graph mode only.
    x = constant_op.constant(1.0, name="x")
    y = constant_op.constant(3.0, name="y")

    def true_fn():
      # [FUNCTION DOC]
      # Branch callable: returns a pair of the product x times y, and y.
      return x * y, y

    def false_fn():
      # [FUNCTION DOC]
      # Branch callable: returns a pair of x and y multiplied by three.
      return x, y * 3.0

    self._testCond(true_fn, false_fn, [x])
    self._testCond(true_fn, false_fn, [x, y])
    self._testCond(true_fn, false_fn, [y])

  @test_util.run_deprecated_v1
  def testBasic2(self):
    # [FUNCTION DOC]
    # Test: another basic comparison where the true branch returns x times y times two and the false
    # branch returns the constant two (not a tensor). Runs the helper with gradients with respect to x,
    # to both and to y. Graph mode only.
    x = constant_op.constant(1.0, name="x")
    y = constant_op.constant(2.0, name="y")

    def true_fn():
      # [FUNCTION DOC]
      # Branch callable: returns x multiplied by y multiplied by two.
      return x * y * 2.0

    def false_fn():
      # [FUNCTION DOC]
      # Branch callable: returns the plain float two (a Python constant rather than a tensor).
      return 2.0

    self._testCond(true_fn, false_fn, [x])
    self._testCond(true_fn, false_fn, [x, y])
    self._testCond(true_fn, false_fn, [y])

  @test_util.run_deprecated_v1
  def testNoInputs(self):
    # [FUNCTION DOC]
    # Test: cond_v2 where the branches capture no outside tensors. Builds a boolean placeholder
    # predicate and branches returning constants one and two. Runs with predicate true and expects a
    # one-element tuple holding one, then false and expects a one-element tuple holding two. Graph mode.
    with self.cached_session() as sess:
      pred = array_ops.placeholder(dtypes.bool, name="pred")

      def true_fn():
        # [FUNCTION DOC]
        # Branch callable: returns the constant float one created inside the branch.
        return constant_op.constant(1.0)

      def false_fn():
        # [FUNCTION DOC]
        # Branch callable: returns the constant float two created inside the branch.
        return constant_op.constant(2.0)

      out = cond_v2.cond_v2(pred, true_fn, false_fn)

      self.assertEqual(sess.run(out, {pred: True}), (1.0,))
      self.assertEqual(sess.run(out, {pred: False}), (2.0,))

  def _createCond(self, name, use_fast_cond=False):
    # [FUNCTION DOC]
    # HELPER: Builds a cond for the name and lowering tests. Creates a constant true predicate and a
    # constant x of one. The true branch returns x and the false branch returns x plus one. When
    # use_fast_cond (default false) is set it calls fast_cond_v2 and takes the output's producing op as
    # the cond op; otherwise calls cond_v2 with the given name and finds the cond op through the first
    # input of the output op (the output is wrapped by an identity). Asserts the cond op type is a
    # stateless If. Returns the pair of output tensor and cond operation. Adds ops to the default graph.
    """Creates a cond_v2 call and returns the output tensor and the cond op."""
    pred = constant_op.constant(True, name="pred")
    x = constant_op.constant(1.0, name="x")

    def true_fn():
      # [FUNCTION DOC]
      # Branch callable: returns the captured tensor x.
      return x

    def false_fn():
      # [FUNCTION DOC]
      # Branch callable: returns the captured tensor x plus one.
      return x + 1

    if use_fast_cond:
      output = cond_v2.fast_cond_v2(pred, true_fn, false_fn, name=name)
      cond_op = output.op
    else:
      output = cond_v2.cond_v2(pred, true_fn, false_fn, name=name)
      cond_op = output.op.inputs[0].op
    self.assertEqual(cond_op.type, "StatelessIf")
    return output, cond_op

  def _createNestedCond(self, name):
    # [FUNCTION DOC]
    # HELPER: Like the cond-building helper but creates a nested cond_v2. The true branch itself calls
    # cond_v2 returning x or x plus one; the false branch returns x plus two. The outer cond is created
    # with the given name; the cond operation is found through the output's first input and asserted to
    # be a stateless If. Returns the pair of output tensor and outer cond operation. Adds ops to the
    # default graph.
    """Like _createCond but creates a nested cond_v2 call as well."""
    pred = constant_op.constant(True, name="pred")
    x = constant_op.constant(1.0, name="x")

    def true_fn():
      # [FUNCTION DOC]
      # Branch callable: calls a nested cond_v2 on the captured predicate choosing x or x plus one.
      return cond_v2.cond_v2(pred, lambda: x, lambda: x + 1)

    def false_fn():
      # [FUNCTION DOC]
      # Branch callable: returns the captured x plus two.
      return x + 2

    output = cond_v2.cond_v2(pred, true_fn, false_fn, name=name)
    cond_op = output.op.inputs[0].op
    self.assertEqual(cond_op.type, "StatelessIf")
    return output, cond_op

  def testDefaultName(self):
    # [FUNCTION DOC]
    # Test: checks the default names. In one fresh graph, builds a cond with no name and expects the op
    # name to be cond and branch function names to match true/false patterns with cond prefix and a
    # numeric suffix. In another graph inside a name scope foo, builds a cond with an empty name and
    # expects foo/cond with branch names prefixed foo_cond; a second unnamed cond in that scope is
    # expected to be named foo/cond_1 with branch names prefixed foo_cond_1.
    with ops.Graph().as_default():
      _, cond_op = self._createCond(None)
      self.assertEqual(cond_op.name, "cond")
      self.assertRegex(cond_op.get_attr("then_branch").name, r"cond_true_\d*")
      self.assertRegex(cond_op.get_attr("else_branch").name, r"cond_false_\d*")

    with ops.Graph().as_default():
      with ops.name_scope("foo"):
        _, cond1_op = self._createCond("")
        self.assertEqual(cond1_op.name, "foo/cond")
        self.assertRegex(
            cond1_op.get_attr("then_branch").name, r"foo_cond_true_\d*")
        self.assertRegex(
            cond1_op.get_attr("else_branch").name, r"foo_cond_false_\d*")

        _, cond2_op = self._createCond(None)
        self.assertEqual(cond2_op.name, "foo/cond_1")
        self.assertRegex(
            cond2_op.get_attr("then_branch").name, r"foo_cond_1_true_\d*")
        self.assertRegex(
            cond2_op.get_attr("else_branch").name, r"foo_cond_1_false_\d*")

  @test_util.run_v2_only
  def testInheritParentNameScope(self):
    # [FUNCTION DOC]
    # Test (v2 behavior only): checks that the branch graphs inherit the parent name scope. Inside a
    # traced function and a name scope foo, runs cond_v2 whose branches open their own scopes named then
    # and else. Inside each branch asserts the current name scope is foo/cond/then or foo/cond/else.
    # Calls the function once so the assertions run during tracing.

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # Traced function: opens a name scope foo and returns cond_v2 on a constant true predicate with
      # then and else branch callables.
      with ops.name_scope("foo"):

        def then_branch():
          # [FUNCTION DOC]
          # Branch callable: opens the scope then, asserts the current name scope equals foo/cond/then and
          # returns the float zero.
          with ops.name_scope("then"):
            actual_name_scope = ops.get_name_scope()
            expected_name_scope = "foo/cond/then"
            self.assertEqual(actual_name_scope, expected_name_scope)
          return 0.

        def else_branch():
          # [FUNCTION DOC]
          # Branch callable: opens the scope else, asserts the current name scope equals foo/cond/else and
          # returns the float zero.
          with ops.name_scope("else"):
            actual_name_scope = ops.get_name_scope()
            expected_name_scope = "foo/cond/else"
            self.assertEqual(actual_name_scope, expected_name_scope)
          return 0.

        return cond_v2.cond_v2(
            constant_op.constant(True), then_branch, else_branch)

    f()

  @test_util.run_v1_only("b/120545219")
  def testFunctionInCond(self):
    # [FUNCTION DOC]
    # Test (graph mode only, tracked by a bug reference): the true branch calls a traced function that
    # multiplies captured x and y by two. The false branch returns two. Runs the comparison helper with
    # gradients with respect to x, to both and to y.
    with ops.Graph().as_default():
      x = constant_op.constant(1.0, name="x")
      y = constant_op.constant(2.0, name="y")

      def true_fn():
        # [FUNCTION DOC]
        # Branch callable: defines a traced function multiplying captured x and y by two and returns its
        # result.

        @def_function.function
        def fn():
          # [FUNCTION DOC]
          # Traced function: returns the captured x times y times two.
          return x * y * 2.0

        return fn()

      def false_fn():
        # [FUNCTION DOC]
        # Branch callable: returns the plain float two.
        return 2.0

      self._testCond(true_fn, false_fn, [x])
      self._testCond(true_fn, false_fn, [x, y])
      self._testCond(true_fn, false_fn, [y])

  @test_util.run_deprecated_v1
  def testNestedFunctionInCond(self):
    # [FUNCTION DOC]
    # Test: false branch calls a traced function that calls another traced function (two levels of
    # function nesting), the inner multiplying captured x and y by two; true branch returns two. Runs
    # the comparison helper with gradients with respect to x, both, and y. Graph mode only.
    x = constant_op.constant(1.0, name="x")
    y = constant_op.constant(2.0, name="y")

    def true_fn():
      # [FUNCTION DOC]
      # Branch callable: returns the plain float two.
      return 2.0

    def false_fn():
      # [FUNCTION DOC]
      # Branch callable: defines and calls a traced function which in turn calls a nested traced function
      # and returns the result.

      @def_function.function
      def fn():
        # [FUNCTION DOC]
        # Traced function: defines a nested traced function and returns its result.

        @def_function.function
        def nested_fn():
          # [FUNCTION DOC]
          # Traced function (innermost): returns captured x times y times two.
          return x * y * 2.0

        return nested_fn()

      return fn()

    self._testCond(true_fn, false_fn, [x])
    self._testCond(true_fn, false_fn, [x, y])
    self._testCond(true_fn, false_fn, [y])

  @test_util.run_deprecated_v1
  def testDoubleNestedFunctionInCond(self):
    # [FUNCTION DOC]
    # Test: three levels of nested traced functions in the true branch; innermost computes x times y
    # times two; the false branch returns two. Runs the comparison helper with gradients with respect to
    # x, both, and y. Graph mode only.
    x = constant_op.constant(1.0, name="x")
    y = constant_op.constant(2.0, name="y")

    def true_fn():
      # [FUNCTION DOC]
      # Branch callable: defines and calls a traced function that nests two more levels of traced
      # functions and returns the result.

      @def_function.function
      def fn():
        # [FUNCTION DOC]
        # Traced function: defines a nested traced function and returns its result.

        @def_function.function
        def nested_fn():
          # [FUNCTION DOC]
          # Traced function: defines an innermost traced function and returns its result.

          @def_function.function
          def nested_nested_fn():
            # [FUNCTION DOC]
            # Traced function (innermost): returns captured x times y times two.
            return x * y * 2.0

          return nested_nested_fn()

        return nested_fn()

      return fn()

    def false_fn():
      # [FUNCTION DOC]
      # Branch callable: returns the plain float two.
      return 2.0

    self._testCond(true_fn, false_fn, [x])
    self._testCond(true_fn, false_fn, [x, y])
    self._testCond(true_fn, false_fn, [y])

  def testNestedCond(self):
    # [FUNCTION DOC]
    # Test: nested conds where the false branch holds another cond. Defines a local runner that, for a
    # given predicate value, builds a graph and runs the comparison helper with gradients with respect
    # to both, x and y, feeding the predicate placeholder. Runs it with true and false values. The inner
    # cond is created by the module's private cond helper, which picks the legacy or v2 cond depending
    # on the active control flow context.

    def run_test(pred_value):
      # [FUNCTION DOC]
      # Helper local to the nested cond test: takes the boolean pred_value fed for the predicate and in a
      # fresh default graph builds the graph, then runs the comparison helper three times with gradients
      # with respect to both, x alone, and y alone, feeding the predicate. Returns nothing.

      def build_graph():
        # [FUNCTION DOC]
        # Helper: creates a predicate placeholder and constants x (one) and y (two), defines outer branches
        # (true returns two, false returns an inner cond) and returns x, y, the predicate and both branch
        # callables.
        pred = array_ops.placeholder(dtypes.bool, name="pred")
        x = constant_op.constant(1.0, name="x")
        y = constant_op.constant(2.0, name="y")

        def true_fn():
          # [FUNCTION DOC]
          # Branch callable: returns the plain float two.
          return 2.0

        def false_fn():
          # [FUNCTION DOC]
          # Branch callable: returns the inner cond named inside_false_fn, choosing between two inner branch
          # functions using the captured predicate.

          def false_true_fn():
            # [FUNCTION DOC]
            # Branch callable: returns x times y times two.
            return x * y * 2.0

          def false_false_fn():
            # [FUNCTION DOC]
            # Branch callable: returns x times five.
            return x * 5.0

          return _cond(pred, false_true_fn, false_false_fn, "inside_false_fn")

        return x, y, pred, true_fn, false_fn

      with ops.Graph().as_default():
        x, y, pred, true_fn, false_fn = build_graph()
        self._testCond(true_fn, false_fn, [x, y], {pred: pred_value})
        self._testCond(true_fn, false_fn, [x], {pred: pred_value})
        self._testCond(true_fn, false_fn, [y], {pred: pred_value})

    run_test(True)
    run_test(False)

  def testNestedCondBothBranches(self):
    # [FUNCTION DOC]
    # Test: both branches of the outer cond contain an inner cond on the same predicate. Defines a local
    # runner for a given predicate value that builds the graph and runs the comparison helper with
    # gradients with respect to both, x and y. Runs with true and false.

    def run_test(pred_value):
      # [FUNCTION DOC]
      # Helper local to the test: for the given predicate value, builds a graph and calls the comparison
      # helper three times (gradients with respect to both, x, y), feeding the predicate. Returns nothing.

      def build_graph():
        # [FUNCTION DOC]
        # Helper: builds a predicate placeholder and constants x and y; true branch is an inner cond choosing
        # the sum or the square of x; false branch is an inner cond choosing the difference or the square of
        # y. Returns x, y, predicate and the two branch callables.
        pred = array_ops.placeholder(dtypes.bool, name="pred")
        x = constant_op.constant(1.0, name="x")
        y = constant_op.constant(2.0, name="y")

        def true_fn():
          # [FUNCTION DOC]
          # Branch callable: returns an inner cond (name unspecified) choosing x plus y or x times x.
          return _cond(pred, lambda: x + y, lambda: x * x, name=None)

        def false_fn():
          # [FUNCTION DOC]
          # Branch callable: returns an inner cond (name unspecified) choosing x minus y or y times y.
          return _cond(pred, lambda: x - y, lambda: y * y, name=None)

        return x, y, pred, true_fn, false_fn

      with ops.Graph().as_default():
        x, y, pred, true_fn, false_fn = build_graph()
        self._testCond(true_fn, false_fn, [x, y], {pred: pred_value})
        self._testCond(true_fn, false_fn, [x], {pred: pred_value})
        self._testCond(true_fn, false_fn, [y], {pred: pred_value})

    run_test(True)
    run_test(False)

  def testDoubleNestedCond(self):
    # [FUNCTION DOC]
    # Test: two levels of nesting inside the false branch with two predicates. For each pair of predicate
    # values, a local runner builds the graph (rebuilding it for each gradient target) and runs the
    # comparison helper with gradients with respect to both, x and y, feeding both predicates. Runs the
    # four combinations of true and false.

    def run_test(pred1_value, pred2_value):
      # [FUNCTION DOC]
      # Helper local to the test: for the two given predicate values, builds the graph and runs the
      # comparison helper with gradients with respect to both x and y, then rebuilds and repeats for x
      # alone and then y alone, feeding both predicates. Returns nothing.

      def build_graph():
        # [FUNCTION DOC]
        # Helper: builds two predicate placeholders and constants x and y; the true branch returns two and
        # the false branch is a doubly nested cond. Returns x, y, both predicates and the two branches.
        pred1 = array_ops.placeholder(dtypes.bool, name="pred1")
        pred2 = array_ops.placeholder(dtypes.bool, name="pred2")
        x = constant_op.constant(1.0, name="x")
        y = constant_op.constant(2.0, name="y")

        def true_fn():
          # [FUNCTION DOC]
          # Branch callable: returns the plain float two.
          return 2.0

        def false_fn():
          # [FUNCTION DOC]
          # Branch callable: returns a cond named inside_false_fn on the second predicate choosing between the
          # two inner branches.

          def false_true_fn():
            # [FUNCTION DOC]
            # Branch callable: returns a cond named inside_false_true_fn on the first predicate choosing between
            # two innermost branches.

            def false_true_true_fn():
              # [FUNCTION DOC]
              # Branch callable (innermost): returns x times y times two.
              return x * y * 2.0

            def false_true_false_fn():
              # [FUNCTION DOC]
              # Branch callable (innermost): returns x times ten.
              return x * 10.0

            return _cond(
                pred1,
                false_true_true_fn,
                false_true_false_fn,
                name="inside_false_true_fn")

          def false_false_fn():
            # [FUNCTION DOC]
            # Branch callable: returns x times five.
            return x * 5.0

          return _cond(
              pred2, false_true_fn, false_false_fn, name="inside_false_fn")

        return x, y, pred1, pred2, true_fn, false_fn

      with ops.Graph().as_default():
        x, y, pred1, pred2, true_fn, false_fn = build_graph()
        self._testCond(true_fn, false_fn, [x, y], {
            pred1: pred1_value,
            pred2: pred2_value
        })
        x, y, pred1, pred2, true_fn, false_fn = build_graph()
        self._testCond(true_fn, false_fn, [x], {
            pred1: pred1_value,
            pred2: pred2_value
        })
        x, y, pred1, pred2, true_fn, false_fn = build_graph()
        self._testCond(true_fn, false_fn, [y], {
            pred1: pred1_value,
            pred2: pred2_value
        })

    run_test(True, True)
    run_test(True, False)
    run_test(False, False)
    run_test(False, True)

  def testGradientFromInsideFunction(self):
    # [FUNCTION DOC]
    # Test: computes gradients inside a traced function. Builds nested outer and inner cond_v2 (outer true
    # returns two; outer false is an inner cond of x times y times two or x times five), then calls a
    # traced function that takes the gradient of the outer cond output with respect to x and y. Runs the
    # gradients for the four predicate combinations and expects: zeros for outer true (either inner),
    # four and two for outer false with inner true, five and zero for outer false with inner false.

    def build_graph():
      # [FUNCTION DOC]
      # Helper local to the test: creates predicate placeholders for the outer and inner conds, constants x
      # and y, the nested cond_v2 graph, and a traced function that returns the gradients of the outer cond
      # with respect to x and y. Returns the gradients, outer predicate and inner predicate.
      pred_outer = array_ops.placeholder(dtypes.bool, name="pred_outer")
      pred_inner = array_ops.placeholder(dtypes.bool, name="pred_inner")
      x = constant_op.constant(1.0, name="x")
      y = constant_op.constant(2.0, name="y")

      def true_fn():
        # [FUNCTION DOC]
        # Branch callable: returns the float two.
        return 2.0

      def false_fn():
        # [FUNCTION DOC]
        # Branch callable: returns an inner cond_v2 named inner_cond using the inner predicate.

        def inner_true_fn():
          # [FUNCTION DOC]
          # Branch callable: returns x times y times two.
          return x * y * 2.0

        def inner_false_fn():
          # [FUNCTION DOC]
          # Branch callable: returns x times five.
          return x * 5.0

        return cond_v2.cond_v2(
            pred_inner, inner_true_fn, inner_false_fn, name="inner_cond")

      cond_outer = cond_v2.cond_v2(
          pred_outer, true_fn, false_fn, name="outer_cond")

      # Compute grads inside a tf function.
      @def_function.function
      def nesting_fn():
        # [FUNCTION DOC]
        # Traced function: returns the gradients of the captured outer cond output with respect to x and y,
        # computed inside the function.
        return gradients_impl.gradients(cond_outer, [x, y])

      grads = nesting_fn()

      return grads, pred_outer, pred_inner

    with ops.Graph().as_default():
      grads, pred_outer, pred_inner = build_graph()
      with self.session(graph=ops.get_default_graph()) as sess:
        self.assertSequenceEqual(
            sess.run(grads, {
                pred_outer: True,
                pred_inner: True
            }), [0., 0.])
        self.assertSequenceEqual(
            sess.run(grads, {
                pred_outer: True,
                pred_inner: False
            }), [0., 0.])
        self.assertSequenceEqual(
            sess.run(grads, {
                pred_outer: False,
                pred_inner: True
            }), [4., 2.])
        self.assertSequenceEqual(
            sess.run(grads, {
                pred_outer: False,
                pred_inner: False
            }), [5., 0.])

  def testGradientFromInsideNestedFunction(self):
    # [FUNCTION DOC]
    # Test: like the previous gradient-inside-function test but the gradient call is inside a function
    # nested within another traced function. Expects the same gradient values for the four predicate
    # combinations (zeros, zeros, four and two, five and zero).

    def build_graph():
      # [FUNCTION DOC]
      # Helper local to the test: builds predicate placeholders, constants and nested cond_v2 graph, then
      # a traced function containing another traced function that computes the gradients of the outer cond
      # with respect to x and y. Returns the gradients and both predicates.
      pred_outer = array_ops.placeholder(dtypes.bool, name="pred_outer")
      pred_inner = array_ops.placeholder(dtypes.bool, name="pred_inner")
      x = constant_op.constant(1.0, name="x")
      y = constant_op.constant(2.0, name="y")

      def true_fn():
        # [FUNCTION DOC]
        # Branch callable: returns the float two.
        return 2.0

      def false_fn():
        # [FUNCTION DOC]
        # Branch callable: returns an inner cond_v2 named inner_cond.

        def inner_true_fn():
          # [FUNCTION DOC]
          # Branch callable: returns x times y times two.
          return x * y * 2.0

        def inner_false_fn():
          # [FUNCTION DOC]
          # Branch callable: returns x times five.
          return x * 5.0

        return cond_v2.cond_v2(
            pred_inner, inner_true_fn, inner_false_fn, name="inner_cond")

      cond_outer = cond_v2.cond_v2(
          pred_outer, true_fn, false_fn, name="outer_cond")

      # Compute grads inside a tf function.
      @def_function.function
      def nesting_fn():
        # [FUNCTION DOC]
        # Traced function: defines and calls a nested traced function and returns its result.

        @def_function.function
        def inner_nesting_fn():
          # [FUNCTION DOC]
          # Traced function (inner): returns the gradients of the captured outer cond with respect to x and y.
          return gradients_impl.gradients(cond_outer, [x, y])

        return inner_nesting_fn()

      grads = nesting_fn()

      return grads, pred_outer, pred_inner

    with ops.Graph().as_default():
      grads, pred_outer, pred_inner = build_graph()
      with self.session(graph=ops.get_default_graph()) as sess:
        self.assertSequenceEqual(
            sess.run(grads, {
                pred_outer: True,
                pred_inner: True
            }), [0., 0.])
        self.assertSequenceEqual(
            sess.run(grads, {
                pred_outer: True,
                pred_inner: False
            }), [0., 0.])
        self.assertSequenceEqual(
            sess.run(grads, {
                pred_outer: False,
                pred_inner: True
            }), [4., 2.])
        self.assertSequenceEqual(
            sess.run(grads, {
                pred_outer: False,
                pred_inner: False
            }), [5., 0.])

  def testBuildCondAndGradientInsideFunction(self):
    # [FUNCTION DOC]
    # Test: builds the cond and its gradient entirely inside a traced function. Under a graph and a
    # session, the helper creates predicates and constants, and returns the gradients of a nested
    # cond_v2 computed inside a traced function. Runs the four combinations of the two predicates and
    # expects zeros, zeros, four and two, and five and zero.

    def build_graph():
      # [FUNCTION DOC]
      # Helper local to the test: creates predicate placeholders and constants, defines a traced function
      # that builds the nested cond_v2 and returns the gradients with respect to x and y, calls it and
      # returns the gradients with both predicates.
      pred_outer = array_ops.placeholder(dtypes.bool, name="pred_outer")
      pred_inner = array_ops.placeholder(dtypes.bool, name="pred_inner")
      x = constant_op.constant(1.0, name="x")
      y = constant_op.constant(2.0, name="y")

      # Build cond and its gradient inside a tf function.
      @def_function.function
      def fn():
        # [FUNCTION DOC]
        # Traced function: defines branch callables, builds the outer cond_v2 named outer_cond and returns
        # the gradients of it with respect to x and y.

        def true_fn():
          # [FUNCTION DOC]
          # Branch callable: returns the float two.
          return 2.0

        def false_fn():
          # [FUNCTION DOC]
          # Branch callable: returns an inner cond_v2 named inner_cond choosing between two inner branches.

          def inner_true_fn():
            # [FUNCTION DOC]
            # Branch callable: returns x times y times two.
            return x * y * 2.0

          def inner_false_fn():
            # [FUNCTION DOC]
            # Branch callable: returns x times five.
            return x * 5.0

          return cond_v2.cond_v2(
              pred_inner, inner_true_fn, inner_false_fn, name="inner_cond")

        cond_outer = cond_v2.cond_v2(
            pred_outer, true_fn, false_fn, name="outer_cond")
        return gradients_impl.gradients(cond_outer, [x, y])

      grads = fn()

      return grads, pred_outer, pred_inner

    with ops.Graph().as_default(), self.session(
        graph=ops.get_default_graph()) as sess:
      grads, pred_outer, pred_inner = build_graph()
      self.assertSequenceEqual(
          sess.run(grads, {
              pred_outer: True,
              pred_inner: True
          }), [0., 0.])
      self.assertSequenceEqual(
          sess.run(grads, {
              pred_outer: True,
              pred_inner: False
          }), [0., 0.])
      self.assertSequenceEqual(
          sess.run(grads, {
              pred_outer: False,
              pred_inner: True
          }), [4., 2.])
      self.assertSequenceEqual(
          sess.run(grads, {
              pred_outer: False,
              pred_inner: False
          }), [5., 0.])

  @test_util.run_deprecated_v1
  def testSecondDerivative(self):
    # [FUNCTION DOC]
    # Test: second derivative of a cond_v2 (graph mode only). With x three, the true branch is x cubed
    # and the false is x. Computes first and second gradients and expects first derivative 27 (true) or
    # 1 (false) and second derivative 18 (true) or 0 (false).
    with self.cached_session() as sess:
      pred = array_ops.placeholder(dtypes.bool, name="pred")
      x = constant_op.constant(3.0, name="x")

      def true_fn():
        # [FUNCTION DOC]
        # Branch callable: returns captured x raised to the power three.
        return math_ops.pow(x, 3)

      def false_fn():
        # [FUNCTION DOC]
        # Branch callable: returns captured x.
        return x

      cond = cond_v2.cond_v2(pred, true_fn, false_fn, name="cond")
      cond_grad = gradients_impl.gradients(cond, [x])
      cond_grad_grad = gradients_impl.gradients(cond_grad, [x])

      # d[x^3]/dx = 3x^2
      true_val = sess.run(cond_grad, {pred: True})
      self.assertEqual(true_val, [27.0])
      # d[x]/dx = 1
      false_val = sess.run(cond_grad, {pred: False})
      self.assertEqual(false_val, [1.0])

      true_val = sess.run(cond_grad_grad, {pred: True})
      # d2[x^3]/dx2 = 6x
      self.assertEqual(true_val, [18.0])
      false_val = sess.run(cond_grad_grad, {pred: False})
      # d2[x]/dx2 = 0
      self.assertEqual(false_val, [0.0])

  def testGradientOfDeserializedCond(self):
    # [FUNCTION DOC]
    # Test: gradients of a cond restored from a serialized meta graph. Builds a cond_v2 in one graph
    # storing x, predicate and the cond output in collections and exports a meta graph. In a new graph
    # imports it, fetches the collection items, builds first and second gradients and checks first
    # derivative 27 or 1 and second derivative 18 or 0 for predicate true or false.
    with ops.Graph().as_default():
      pred = array_ops.placeholder(dtypes.bool, name="pred")
      x = constant_op.constant(3.0, name="x")
      ops.add_to_collection("x", x)

      def true_fn():
        # [FUNCTION DOC]
        # Branch callable: returns captured x cubed.
        return math_ops.pow(x, 3)

      def false_fn():
        # [FUNCTION DOC]
        # Branch callable: returns captured x.
        return x

      ops.add_to_collection("pred", pred)
      cond = cond_v2.cond_v2(pred, true_fn, false_fn, name="cond")
      ops.add_to_collection("cond", cond)
      meta_graph = saver.export_meta_graph()

    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        saver.import_meta_graph(meta_graph)
        x = ops.get_collection("x")[0]
        pred = ops.get_collection("pred")[0]
        cond = ops.get_collection("cond")
        cond_grad = gradients_impl.gradients(cond, [x], name="cond_grad")
        cond_grad_grad = gradients_impl.gradients(
            cond_grad, [x], name="cond_grad_grad")
        # d[x^3]/dx = 3x^2
        true_val = sess.run(cond_grad, {pred: True})
        self.assertEqual(true_val, [27.0])
        # d[x]/dx = 1
        false_val = sess.run(cond_grad, {pred: False})
        self.assertEqual(false_val, [1.0])

        true_val = sess.run(cond_grad_grad, {pred: True})
        # d2[x^3]/dx2 = 6x
        self.assertEqual(true_val, [18.0])
        false_val = sess.run(cond_grad_grad, {pred: False})
        # d2[x]/dx2 = 0
        self.assertEqual(false_val, [0.0])

  @test_util.run_deprecated_v1
  def testFuncCond(self):
    # [FUNCTION DOC]
    # Test (graph mode only): a traced function containing two stateless cond_v2 calls named cond_1 and
    # cond_2. Gets the concrete function and checks both ops are stateless If ops and the second has no
    # control inputs. Calls it and expects the output op to be a plain partitioned call with value one.

    @def_function.function
    def fn_with_cond():
      # [FUNCTION DOC]
      # Traced function: creates a cond_v2 named cond_1 (true) with zero or one branches, then returns a
      # cond_v2 named cond_2 (false) with zero or one branches. Used to check statelessness.
      cond_v2.cond_v2(
          constant_op.constant(True),
          lambda: array_ops.zeros([]),
          lambda: array_ops.ones([]),
          name="cond_1")
      return cond_v2.cond_v2(
          constant_op.constant(False),
          lambda: array_ops.zeros([]),
          lambda: array_ops.ones([]),
          name="cond_2")

    concrete_fn = fn_with_cond.get_concrete_function()
    cond_1 = concrete_fn.graph.get_operation_by_name("cond_1")
    cond_2 = concrete_fn.graph.get_operation_by_name("cond_2")
    # Verify that all functional ops are stateless and cond_2 does not have
    # any control inputs.
    self.assertEqual(cond_1.type, "StatelessIf")
    self.assertEqual(cond_2.type, "StatelessIf")
    self.assertLen(cond_2.control_inputs, 0)
    fn_output = concrete_fn()
    self.assertEqual(fn_output.op.type, "PartitionedCall")
    self.assertAllEqual(fn_output, 1.0)

  @test_util.run_deprecated_v1
  def testFuncCondFunc(self):
    # [FUNCTION DOC]
    # Test (graph mode only): like the previous test but the true branch of the second cond is itself a
    # traced function. Checks both conds are stateless If ops, the second has no control inputs, that
    # the second true graph has no stateful partitioned call and exactly one partitioned call, and that
    # running the function yields a partitioned-call output with value three.

    @def_function.function
    def fn_with_cond():
      # [FUNCTION DOC]
      # Traced function: creates cond_1 with constant branches and cond_2 whose true branch is a nested
      # traced function returning three and false branch returns four. Returns cond_2.
      cond_v2.cond_v2(
          constant_op.constant(True),
          lambda: constant_op.constant(1.),
          lambda: constant_op.constant(2.),
          name="cond_1")

      @def_function.function
      def true_branch():
        # [FUNCTION DOC]
        # Nested traced function: returns the constant three.
        return constant_op.constant(3.)

      return cond_v2.cond_v2(
          constant_op.constant(True),
          true_branch,
          lambda: constant_op.constant(4.),
          name="cond_2")

    concrete_fn = fn_with_cond.get_concrete_function()
    cond_1 = concrete_fn.graph.get_operation_by_name("cond_1")
    cond_2 = concrete_fn.graph.get_operation_by_name("cond_2")
    # Verify that all functional ops are stateless and cond_2 does not have
    # any control inputs.
    self.assertEqual(cond_1.type, "StatelessIf")
    self.assertEqual(cond_2.type, "StatelessIf")
    self.assertLen(cond_2.control_inputs, 0)
    cond_2_true_graph, _ = cond_v2.get_func_graphs(cond_2)
    cond_2_true_graph_operations = cond_2_true_graph.get_operations()
    self.assertEmpty([
        op for op in cond_2_true_graph_operations
        if op.type == "StatefulPartitionedCall"
    ])
    self.assertLen([
        op for op in cond_2_true_graph_operations
        if op.type == "PartitionedCall"
    ], 1)
    fn_output = concrete_fn()
    self.assertEqual(fn_output.op.type, "PartitionedCall")
    self.assertAllEqual(fn_output, 3.0)

  @test_util.run_deprecated_v1
  def testFuncCondWithVariable(self):
    # [FUNCTION DOC]
    # Test (graph mode only): stateful conds touching variables. Creates two variables initialized by the
    # global initializer, plus branches assigning to each. In a traced function creates four stateful
    # conds (cond_1 to cond_4) and a stateless one. Checks stateful ones are If ops, stateless is a
    # stateless If, control inputs chain cond_2 after cond_1 and cond_4 after cond_3, the stateless cond
    # has none, and that the function output evaluates to two, four and six with a stateful call op.
    v1 = variables.Variable(2.)
    v2 = variables.Variable(4.)

    self.evaluate(variables.global_variables_initializer())

    def update_v1():
      # [FUNCTION DOC]
      # Branch callable: assigns the captured variable v1 to itself (a side effect) and returns v1.
      v1.assign(v1)
      return v1

    def update_v2():
      # [FUNCTION DOC]
      # Branch callable: assigns the captured variable v2 to itself and returns v2.
      v2.assign(v2)
      return v2

    @def_function.function
    def fn_with_cond():
      # [FUNCTION DOC]
      # Traced function: builds the cond_1 to cond_4 and stateless conds referencing v1 and v2 and returns
      # cond_2, cond_4 and the stateless cond. Used to check automatic control dependencies.
      cond_v2.cond_v2(
          constant_op.constant(True),
          update_v1,
          lambda: constant_op.constant(0.),
          name="cond_1")
      cond_2 = cond_v2.cond_v2(
          constant_op.constant(False),
          lambda: constant_op.constant(0.),
          update_v1,
          name="cond_2")
      cond_v2.cond_v2(
          constant_op.constant(True),
          update_v2,
          lambda: constant_op.constant(0.),
          name="cond_3")
      cond_4 = cond_v2.cond_v2(
          constant_op.constant(False),
          lambda: constant_op.constant(0.),
          lambda: v2,
          name="cond_4")
      stateless_cond = cond_v2.cond_v2(
          constant_op.constant(False),
          lambda: constant_op.constant(5.),
          lambda: constant_op.constant(6.),
          name="stateless_cond")
      return cond_2, cond_4, stateless_cond

    concrete_fn = fn_with_cond.get_concrete_function()
    cond_1 = concrete_fn.graph.get_operation_by_name("cond_1")
    cond_2 = concrete_fn.graph.get_operation_by_name("cond_2")
    cond_3 = concrete_fn.graph.get_operation_by_name("cond_3")
    cond_4 = concrete_fn.graph.get_operation_by_name("cond_4")
    stateless_cond = concrete_fn.graph.get_operation_by_name("stateless_cond")
    self.assertEqual(cond_1.type, "If")
    self.assertEqual(cond_2.type, "If")
    self.assertEqual(cond_3.type, "If")
    self.assertEqual(cond_4.type, "If")
    self.assertEqual(stateless_cond.type, "StatelessIf")
    self.assertEmpty(cond_1.control_inputs)
    self.assertLen(cond_2.control_inputs, 1)
    self.assertIs(cond_2.control_inputs[0], cond_1)
    self.assertEmpty(cond_3.control_inputs)
    self.assertLen(cond_4.control_inputs, 1)
    self.assertIs(cond_4.control_inputs[0], cond_3)
    # Does not touch any variable so should not have any control inputs.
    self.assertEmpty(stateless_cond.control_inputs)
    fn_output = concrete_fn()
    self.assertEqual(fn_output[0].op.type, "StatefulPartitionedCall")
    self.assertAllEqual(self.evaluate(fn_output), [2.0, 4.0, 6.0])

  @test_util.run_deprecated_v1
  def testFuncCondFuncWithVariable(self):
    # [FUNCTION DOC]
    # Test (graph mode only): like the previous test but the update functions are defined inside the
    # traced function and cond_4's false branch is itself a traced function assigning v2. Checks all conds
    # are If ops, control inputs chain cond_2 to cond_1 and cond_4 to cond_3, the nested false graph has
    # a stateful partitioned call (and no plain partitioned call), and the output evaluates to two and
    # four with a stateful call op.
    v1 = variables.Variable(2.)
    v2 = variables.Variable(4.)

    self.evaluate(variables.global_variables_initializer())

    @def_function.function
    def fn_with_cond():
      # [FUNCTION DOC]
      # Traced function: defines update_v1 and update_v2 and the nested false branch, builds four stateful
      # conds and returns cond_2 and cond_4.

      def update_v1():
        # [FUNCTION DOC]
        # Branch callable: assigns the captured variable v1 to itself and returns it.
        v1.assign(v1)
        return v1

      def update_v2():
        # [FUNCTION DOC]
        # Branch callable: assigns the captured variable v2 to itself and returns it.
        v2.assign(v2)
        return v2

      cond_v2.cond_v2(
          constant_op.constant(True),
          update_v1,
          lambda: constant_op.constant(0.),
          name="cond_1")
      cond_2 = cond_v2.cond_v2(
          constant_op.constant(False),
          lambda: constant_op.constant(0.),
          update_v1,
          name="cond_2")
      cond_v2.cond_v2(
          constant_op.constant(True),
          update_v2,
          lambda: constant_op.constant(0.),
          name="cond_3")

      @def_function.function
      def cond_4_false_branch():
        # [FUNCTION DOC]
        # Nested traced function: assigns the captured variable v2 to itself and returns it. Used as the
        # false branch of cond_4.
        v2.assign(v2)
        return v2

      cond_4 = cond_v2.cond_v2(
          constant_op.constant(False),
          lambda: constant_op.constant(0.),
          cond_4_false_branch,
          name="cond_4")
      return cond_2, cond_4

    concrete_fn = fn_with_cond.get_concrete_function()
    cond_1 = concrete_fn.graph.get_operation_by_name("cond_1")
    cond_2 = concrete_fn.graph.get_operation_by_name("cond_2")
    cond_3 = concrete_fn.graph.get_operation_by_name("cond_3")
    cond_4 = concrete_fn.graph.get_operation_by_name("cond_4")
    self.assertEqual(cond_1.type, "If")
    self.assertEqual(cond_2.type, "If")
    self.assertEqual(cond_3.type, "If")
    self.assertEqual(cond_4.type, "If")
    self.assertEmpty(cond_1.control_inputs)
    self.assertLen(cond_2.control_inputs, 1)
    self.assertIs(cond_2.control_inputs[0], cond_1)
    self.assertEmpty(cond_3.control_inputs)
    self.assertLen(cond_4.control_inputs, 1)
    self.assertIs(cond_4.control_inputs[0], cond_3)
    _, cond_4_false_graph = cond_v2.get_func_graphs(cond_4)
    cond_4_false_graph_operations = cond_4_false_graph.get_operations()
    self.assertEmpty([
        op for op in cond_4_false_graph_operations
        if op.type == "PartitionedCall"
    ])
    self.assertLen([
        op for op in cond_4_false_graph_operations
        if op.type == "StatefulPartitionedCall"
    ], 1)
    fn_output = concrete_fn()
    self.assertEqual(fn_output[0].op.type, "StatefulPartitionedCall")
    self.assertAllEqual(self.evaluate(fn_output), [2.0, 4.0])

  def testGradientTapeOfCondWithResourceVariableInFunction(self):
    # [FUNCTION DOC]
    # Test: gradient tape around a cond_v2 reading a resource variable inside a traced function. The
    # variable has value two, the true branch is the variable cubed and the false the variable itself,
    # predicate true. Expects the gradient with respect to the variable to be twelve.
    v = variables.Variable(2.)

    @def_function.function
    def fn_with_cond():
      # [FUNCTION DOC]
      # Traced function: records a gradient tape, builds a cond_v2 over a true predicate choosing the
      # variable cubed or the variable, and returns the gradient of the cond with respect to the variable.
      with backprop.GradientTape() as tape:
        pred = constant_op.constant(True, dtype=dtypes.bool)

        def true_fn():
          # [FUNCTION DOC]
          # Branch callable: returns the captured variable raised to the power three.
          return math_ops.pow(v, 3)

        def false_fn():
          # [FUNCTION DOC]
          # Branch callable: returns the captured variable.
          return v

        cond = cond_v2.cond_v2(pred, true_fn, false_fn, name="cond")
      return tape.gradient(cond, v)

    self.assertAllEqual(fn_with_cond(), 12.0)

  def _CheckIteratedCosGradients(self, func):
    # [FUNCTION DOC]
    # HELPER: Checks higher-order derivatives of a function up to five times. Given a callable func
    # mapping a scalar tensor to a scalar tensor, it wraps it repeatedly with a gradient transformation.
    # Iterates over the expected functions cos, negative sin, negative cos, sin and cos and asserts that
    # the traced function applied to one is close to the expected value at one, then differentiates the
    # function again. Returns nothing.

    def _grad(f):
      # [FUNCTION DOC]
      # HELPER: Local function that takes a function f and returns a new function computing the derivative of
      # f using a gradient tape.
      def _grad_function(primal):
        # [FUNCTION DOC]
        # HELPER: Returned function: watches the given primal tensor on a tape, applies the captured function
        # to it and returns the gradient of the result with respect to the primal.
        with backprop.GradientTape() as tape:
          tape.watch(primal)
          primal_out = f(primal)
        return tape.gradient(primal_out, primal)
      return _grad_function

    f = func
    one = constant_op.constant(1.)
    for expected in [math_ops.cos,
                     lambda x: -math_ops.sin(x),
                     lambda x: -math_ops.cos(x),
                     math_ops.sin,
                     math_ops.cos]:
      self.assertAllClose(expected(one), def_function.function(f)(one))
      f = _grad(f)

  def testIteratedGradientsCond(self):
    # [FUNCTION DOC]
    # Test: iterated gradients of cond_v2. The function is cond_v2 on a true constant choosing the cosine
    # or sine of x. Uses the iterated cosine gradient checker, expecting repeated derivatives of cosine.
    def _func(x):
      # [FUNCTION DOC]
      # Helper local to the test: takes x and returns cond_v2 (constant true predicate) choosing the cosine
      # of an identity of x or the sine of an identity of x.
      return cond_v2.cond_v2(
          constant_op.constant(True),
          lambda: math_ops.cos(array_ops.identity(x)),
          lambda: math_ops.sin(array_ops.identity(x)))
    self._CheckIteratedCosGradients(_func)

  def testIteratedGradientsCase(self):
    # [FUNCTION DOC]
    # Test: iterated gradients of the indexed case op. The function selects with index one between
    # sine and cosine branches, so the cosine branch is taken; checked with the iterated cosine gradient
    # checker.
    def _func(x):
      # [FUNCTION DOC]
      # Helper local to the test: takes x and returns indexed_case with constant index one over two
      # branches (sine and cosine of an identity of x).
      return cond_v2.indexed_case(
          constant_op.constant(1),
          [lambda: math_ops.sin(array_ops.identity(x)),
           lambda: math_ops.cos(array_ops.identity(x))])
    self._CheckIteratedCosGradients(_func)

  def testLowering(self):
    # [FUNCTION DOC]
    # Test: checks that cond_v2 is lowered to switch/merge by default. Builds a graph with the loop
    # optimization pass disabled in the config (otherwise it would replace Switch with Identity), creates
    # a cond via the helper, runs it requesting partition graphs and asserts a Switch node exists and no
    # stateless If node exists in the executed graphs.
    with ops.Graph().as_default() as g:
      # Disable Loop_optimizer grappler pass for this test because it replaces
      # Switch with Identity when it's part of a dead branch.
      config = config_pb2.ConfigProto()
      config.graph_options.rewrite_options.loop_optimization = (
          rewriter_config_pb2.RewriterConfig.OFF)
      with self.session(graph=g, config=config) as sess:
        cond_output, _ = self._createCond("cond")

        run_options = config_pb2.RunOptions(output_partition_graphs=True)
        run_metadata = config_pb2.RunMetadata()
        sess.run(cond_output, options=run_options, run_metadata=run_metadata)

        # If lowering was enabled, there should be a `Switch` node
        self.assertTrue(
            _has_node_with_op(run_metadata, "Switch"),
            "A `Switch` op should exist if the graph was lowered.")

        # If lowering was enabled, there should be no `If` node
        self.assertFalse(
            _has_node_with_op(run_metadata, "StatelessIf"),
            "An `If` op was found, but it should be lowered.")

  @test_util.run_deprecated_v1
  def testLoweringDisabledInXLA(self):
    # [FUNCTION DOC]
    # Test (graph mode only): builds a cond_v2 inside an XLA control flow context and checks lowering is
    # disabled. Asserts reading the lowering attribute of the cond op raises a value error (attribute
    # unset). Runs it, asserts no Switch node in the partition graphs. If XLA is enabled, expects the
    # stateless If to be clustered (no If node, with XLA compile and run nodes present); otherwise
    # expects the If node to remain.
    with self.session(graph=ops.Graph()) as sess:
      # Build the cond_v2 in an XLA context
      xla_context = control_flow_ops.XLAControlFlowContext()
      xla_context.Enter()
      cond_output, cond_op = self._createCond("cond")
      xla_context.Exit()

      # Check lowering attr is not set.
      with self.assertRaises(ValueError):
        cond_op.get_attr("_lower_using_switch_merge")

      # Check the actual graph that is run.
      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      run_metadata = config_pb2.RunMetadata()
      sess.run(cond_output, options=run_options, run_metadata=run_metadata)

      # Lowering disabled in XLA, there should be no `Switch` node
      self.assertFalse(
          _has_node_with_op(run_metadata, "Switch"),
          "A `Switch` op exists, but the graph should not be lowered.")

      if test_util.is_xla_enabled():
        # If XLA is actually enabled then we expect the StatelessIf to have been
        # put inside an XLA cluster.
        self.assertFalse(
            _has_node_with_op(run_metadata, "StatelessIf"),
            ("A `StatelessIf` op was found, but the node should have been " +
             "clustered."))
        self.assertTrue(
            _has_node_with_op(run_metadata, "_XlaCompile"),
            ("An `_XlaCompile` op was not found, but the `StatelessIf` (at " +
             "least) op should have been clustered."))
        self.assertTrue(
            _has_node_with_op(run_metadata, "_XlaRun"),
            ("An `_XlaRun` op was not found, but the `StatelessIf` (at " +
             "least) op should have been clustered."))
      else:
        # Lowering disabled in XLA, there should still be an `If` node
        self.assertTrue(
            _has_node_with_op(run_metadata, "StatelessIf"),
            ("A `StatelessIf` op was not found, but the graph should not be " +
             "lowered."))

  @test_util.run_deprecated_v1
  def testNestedLoweringDisabledInXLA(self):
    # [FUNCTION DOC]
    # Test (graph mode only): nested cond built inside an XLA context via the helper. Asserts that the
    # lowering attribute is not set on the outer cond op (raises a value error) and that exactly one
    # nested stateless If op exists in the graph's registered functions, also without the attribute.
    # The trailing note says the executed graphs are not checked yet (a to-do).
    # Build the cond_v2 in an XLA context
    xla_context = control_flow_ops.XLAControlFlowContext()
    xla_context.Enter()
    _, cond_op = self._createNestedCond("cond")
    xla_context.Exit()

    # Check lowering attr is not set for either If node.
    with self.assertRaises(ValueError):
      cond_op.get_attr("_lower_using_switch_merge")

    nested_if_ops = []
    for func in ops.get_default_graph()._functions.values():
      nested_if_ops.extend(
          op for op in func.graph.get_operations() if op.type == "StatelessIf")
    self.assertEqual(len(nested_if_ops), 1)
    with self.assertRaises(ValueError):
      nested_if_ops[0].get_attr("_lower_using_switch_merge")

    # TODO(skyewm): check the actual graphs that are run once we have a way to
    # programmatically access those graphs.

  @test_util.run_deprecated_v1
  def testLoweringDisabledWithFastCond(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The comment before the cond creation says "Build the cond_v2 in an XLA context" but no
    # XLA context is entered; this test uses fast_cond_v2 instead (comment is stale or misleading).
    # Test (graph mode only): builds a cond with fast_cond_v2 and checks lowering is disabled. Reads
    # the lowering attribute and asserts it is false, runs the cond requesting partition graphs and
    # asserts there is no Switch node. If XLA is enabled expects the stateless If to be clustered with
    # compile and run nodes; otherwise expects the stateless If to remain.
    with self.session(graph=ops.Graph()) as sess:
      # Build the cond_v2 in an XLA context
      cond_output, cond_op = self._createCond("cond", use_fast_cond=True)

      # Check lowering attr is not set.
      lowering_attr = cond_op.get_attr("_lower_using_switch_merge")
      self.assertFalse(
          lowering_attr, f"Lowering attr is not set as expected. {cond_op}"
      )

      # Check the actual graph that is run.
      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      run_metadata = config_pb2.RunMetadata()
      sess.run(cond_output, options=run_options, run_metadata=run_metadata)

      # Lowering disabled for `fast_cond_v2`, there should be no `Switch` node
      self.assertFalse(
          _has_node_with_op(run_metadata, "Switch"),
          "A `Switch` op exists, but the graph should not be lowered.",
      )

      if test_util.is_xla_enabled():
        # If XLA is actually enabled then we expect the StatelessIf to have been
        # put inside an XLA cluster.
        self.assertFalse(
            _has_node_with_op(run_metadata, "StatelessIf"),
            (
                "A `StatelessIf` op was found, but the node should have been "
                + "clustered."
            ),
        )
        self.assertTrue(
            _has_node_with_op(run_metadata, "_XlaCompile"),
            (
                "An `_XlaCompile` op was not found, but the `StatelessIf` (at "
                + "least) op should have been clustered."
            ),
        )
        self.assertTrue(
            _has_node_with_op(run_metadata, "_XlaRun"),
            (
                "An `_XlaRun` op was not found, but the `StatelessIf` (at "
                + "least) op should have been clustered."
            ),
        )
      else:
        # Lowering disabled for `fast_cond_v2`, there should still be a
        # `StatelessIf` node.
        self.assertTrue(
            _has_node_with_op(run_metadata, "StatelessIf"),
            (
                "A `StatelessIf` op was not found, but the graph should not be "
                + "lowered."
                + str(run_metadata)
            ),
        )

  # b/131355614
  @test_util.run_deprecated_v1
  def testNoOptionalsInXla(self):
    # [FUNCTION DOC]
    # Test (graph mode only, linked to a bug): checks no optional ops (from value, none, has value, get
    # value) appear in cond branches when gradients are built under an XLA context. Builds a traced
    # function with a cond_v2 and its gradient and verifies the branch functions of both forward and
    # gradient If ops have no optional ops, then calls it with an XLA context entered and exited.

    @def_function.function
    def func_with_cond():
      # [FUNCTION DOC]
      # Traced function: creates cond_v2 over a true predicate with branches using an intermediate value,
      # takes the gradient with respect to x, and checks no optional ops in then and else branch functions
      # of both the forward and gradient If ops. Returns the gradient.
      pred = constant_op.constant(True, name="pred")
      x = constant_op.constant(1.0, name="x")

      def true_fn():
        # [FUNCTION DOC]
        # Branch callable: computes x plus one and returns it multiplied by x (an intermediate that would be
        # saved for the gradient).
        intermediate = x + 1
        return intermediate * x

      def false_fn():
        # [FUNCTION DOC]
        # Branch callable: returns x plus one.
        return x + 1

      output = cond_v2.cond_v2(pred, true_fn, false_fn)
      grad = gradients_impl.gradients(output, x)[0]

      forward_if_op = output.op.inputs[0].op
      gradient_if_op = grad.op.inputs[0].op

      def verify_no_optional_ops(op, branch_name):
        # [FUNCTION DOC]
        # Helper local to the test: takes an If op and an attribute name for a branch, looks up the branch
        # function in the default graph, and asserts none of its node definitions use an optional op type.
        branch_function = ops.get_default_graph()._get_function(
            op.get_attr(branch_name).name)
        function_def = branch_function.cached_definition
        for node_def in function_def.node_def:
          self.assertNotIn(node_def.op, _OPTIONAL_OPS)

      verify_no_optional_ops(forward_if_op, "then_branch")
      verify_no_optional_ops(forward_if_op, "else_branch")
      verify_no_optional_ops(gradient_if_op, "then_branch")
      verify_no_optional_ops(gradient_if_op, "else_branch")

      return grad

    xla_context = control_flow_ops.XLAControlFlowContext()
    xla_context.Enter()
    func_with_cond()
    xla_context.Exit()

  @test_util.run_deprecated_v1
  def testLoweringDisabledWithSingleThreadedExecutorContext(self):
    # [FUNCTION DOC]
    # Test (graph mode only): checks lowering is disabled under the single-threaded executor. Within a
    # session without GPU (the single-threaded executor does not support partitioned graphs), defines a
    # traced function with a cond_v2 and calls it with the executor type set to single threaded on a
    # placeholder, then runs it with value one. A successful run means lowering was disabled, since that
    # executor cannot run v1 cond ops.
    # Single threaded executor does not support partitioned graphs, so we can't
    # run on GPUs (running on GPU requires a mixed CPU/GPU graph).
    with self.session(graph=ops.Graph(), use_gpu=False) as sess:

      @def_function.function
      def _add_cond(x):
        # [FUNCTION DOC]
        # Traced function: returns cond_v2 on a constant true predicate choosing x or x plus one.
        return cond_v2.cond_v2(
            constant_op.constant(True, name="pred"),
            lambda: x,
            lambda: x + 1)

      x = array_ops.placeholder(shape=None, dtype=dtypes.float32)
      with context.function_executor_type("SINGLE_THREADED_EXECUTOR"):
        out_cond = _add_cond(x)

      # The fact that sess.run() succeeds means lowering is disabled, because
      # the single threaded executor does not support cond v1 ops.
      sess.run(out_cond, feed_dict={x: 1.0})

  @test_util.enable_control_flow_v2
  def testStructuredOutputs(self):
    # [FUNCTION DOC]
    # Test (control flow v2 enabled): the public cond with branches returning nested structures (a tuple
    # of tuple and a tensor). With a false predicate, expects the first nested value one and the second
    # value nine.
    x = constant_op.constant(1.0, name="x")
    y = constant_op.constant(3.0, name="y")

    def true_fn():
      # [FUNCTION DOC]
      # Branch callable: returns a nested tuple holding x times y in an inner tuple and y.
      return ((x * y,), y)

    def false_fn():
      # [FUNCTION DOC]
      # Branch callable: returns a nested tuple holding x in an inner tuple and y times three.
      return ((x,), y * 3.0)

    output = tf_cond.cond(
        constant_op.constant(False), true_fn, false_fn)
    self.assertEqual(self.evaluate(output[0][0]), 1.)
    self.assertEqual(self.evaluate(output[1]), 9.)

  @test_util.enable_control_flow_v2
  @test_util.run_deprecated_v1
  def testRaisesOutputStructuresMismatch(self):
    # [FUNCTION DOC]
    # Test (control flow v2 enabled, graph mode only): checks the public cond raises a type error when the
    # branch outputs have different structures (a flat pair versus a nested pair). Expects the message
    # that both arguments must have the same number, type and overall structure of return values.
    x = constant_op.constant(1.0, name="x")
    y = constant_op.constant(3.0, name="y")

    def true_fn():
      # [FUNCTION DOC]
      # Branch callable: returns a flat pair of x times y and y.
      return x * y, y

    def false_fn():
      # [FUNCTION DOC]
      # Branch callable: returns a nested pair with x in a tuple and y times three.
      return ((x,), y * 3.0)

    with self.assertRaisesRegex(
        TypeError, "true_fn and false_fn arguments to tf.cond must have the "
        "same number, type, and overall structure of return values."):
      tf_cond.cond(constant_op.constant(False), true_fn, false_fn)

  @test_util.enable_control_flow_v2
  def testCondAndTensorArray(self):
    # [FUNCTION DOC]
    # Test (control flow v2 enabled): a while loop whose body writes into a tensor array inside a cond.
    # Over the range minus five to four, writes the square for positive entries and the value otherwise,
    # then stacks the array and expects the values minus five through zero, one, four, nine, sixteen.
    x = math_ops.range(-5, 5)
    output = tensor_array_ops.TensorArray(dtype=dtypes.int32, size=x.shape[0])

    def loop_body(i, output):
      # [FUNCTION DOC]
      # Helper local to the test: loop body taking the index and the tensor array; builds the cond that
      # writes x squared or x at the index and returns the next index and updated array.

      def if_true():
        # [FUNCTION DOC]
        # Branch callable: writes the square of the element at the loop index into the tensor array and returns
        # the new array.
        return output.write(i, x[i]**2)

      def if_false():
        # [FUNCTION DOC]
        # Branch callable: writes the element at the loop index into the tensor array and returns the array.
        return output.write(i, x[i])

      output = tf_cond.cond(x[i] > 0, if_true, if_false)
      return i + 1, output

    _, output = while_loop.while_loop(
        lambda i, arr: i < x.shape[0],
        loop_body,
        loop_vars=(constant_op.constant(0), output))
    output_t = output.stack()
    self.assertAllEqual(
        self.evaluate(output_t), [-5, -4, -3, -2, -1, 0, 1, 4, 9, 16])

  @test_util.enable_control_flow_v2
  def testCondAndTensorArrayInFunction(self):
    # [FUNCTION DOC]
    # Test (control flow v2 enabled): same tensor-array-in-cond while loop but built inside a traced
    # function. Returns the stacked array and expects minus five through zero, one, four, nine, sixteen.

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # Traced function: builds the range, an int tensor array and a while loop whose body uses a cond to
      # write squares of positive values and raw other values, and returns the stacked result.
      x = math_ops.range(-5, 5)
      output = tensor_array_ops.TensorArray(dtype=dtypes.int32, size=x.shape[0])

      def loop_body(i, output):
        # [FUNCTION DOC]
        # Helper local to the traced function: loop body taking index and array and writing a value via a cond;
        # returns the incremented index and updated array.

        def if_true():
          # [FUNCTION DOC]
          # Branch callable: writes the square of the element at the index into the array and returns the array.
          return output.write(i, x[i]**2)

        def if_false():
          # [FUNCTION DOC]
          # Branch callable: writes the element at the index into the array and returns the array.
          return output.write(i, x[i])

        output = tf_cond.cond(x[i] > 0, if_true, if_false)
        return i + 1, output

      _, output = while_loop.while_loop(
          lambda i, arr: i < x.shape[0],
          loop_body,
          loop_vars=(constant_op.constant(0), output))
      return output.stack()

    output_t = f()
    self.assertAllEqual(output_t, [-5, -4, -3, -2, -1, 0, 1, 4, 9, 16])

  @test_util.run_deprecated_v1
  def testForwardPassRewrite(self):
    # [FUNCTION DOC]
    # Test (graph mode only): checks the forward If op is rewritten once when a gradient is requested.
    # Builds a cond_v2 with an intermediate value (y plus one) in the true branch and asserts the If op
    # has one output. After computing gradients it expects two outputs (the intermediate is exposed),
    # and after computing gradients again it still has two (no repeated rewrite).
    x = constant_op.constant(1.0, name="x")
    y = constant_op.constant(1.0, name="y")

    def true_fn():
      # [FUNCTION DOC]
      # Branch callable: computes y plus one and returns x times it.
      y_plus_one = y + 1.
      return x * y_plus_one

    output = cond_v2.cond_v2(constant_op.constant(True), true_fn, lambda: x)
    if_op = output.op.inputs[0].op
    self.assertEqual(if_op.type, "StatelessIf")
    # pylint: disable=g-deprecated-assert
    self.assertEqual(len(if_op.outputs), 1)

    gradients_impl.gradients(output, x)
    # if_op should have been rewritten to output `y_plus_one`.
    self.assertEqual(len(if_op.outputs), 2)

    gradients_impl.gradients(output, x)
    # Computing the gradient again shouldn't rewrite if_op again.
    self.assertEqual(len(if_op.outputs), 2)
    # pylint: enable=g-deprecated-assert

  @test_util.run_deprecated_v1
  def testDoNotAccumulateConstants(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The inline comment says the number of outputs "does change" but the assertions expect
    # the count to stay at one (it does not change). The comment is misleading.
    # Test (graph mode only): checks constants and loop inputs are not accumulated as extra outputs. Builds
    # a cond_v2 with branches x times two and x, asserts the If op has one output, computes gradients
    # and asserts it still has one output (x is a captured input and two is a constant), and repeats the
    # gradient computation with the same expectation.
    x = constant_op.constant(1.0, name="x")
    output = cond_v2.cond_v2(
        constant_op.constant(True), lambda: x * 2.0, lambda: x)
    if_op = output.op.inputs[0].op
    self.assertEqual(if_op.type, "StatelessIf")
    # pylint: disable=g-deprecated-assert
    self.assertEqual(len(if_op.outputs), 1)

    gradients_impl.gradients(output, x)
    # Number of outputs does change because
    # 1. `x` is a loop input so does not need to be accumulated.
    # 2. 2.0 is a constant so it is not accumulated.
    self.assertEqual(len(if_op.outputs), 1)

    gradients_impl.gradients(output, x)
    # Computing the gradient again shouldn't rewrite if_op again.
    self.assertEqual(len(if_op.outputs), 1)
    # pylint: enable=g-deprecated-assert

  def testIsControlFlowGraph(self):
    # [FUNCTION DOC]
    # Test: checks branch graphs are marked as control flow graphs. In a traced function with a condition
    # argument, both branches compute x plus one and assert the graph of the result is flagged as a
    # control flow graph. Calls with true then false and expects the value two both times.
    x = constant_op.constant(1.0, name="x")

    @def_function.function
    def f(c):
      # [FUNCTION DOC]
      # Traced function: takes a condition tensor and returns cond_v2 with two branches.

      def then_branch():
        # [FUNCTION DOC]
        # Branch callable: computes x plus one, asserts its graph is a control flow graph and returns it.
        i = x + 1
        self.assertTrue(i.graph.is_control_flow_graph)
        return i

      def else_branch():
        # [FUNCTION DOC]
        # Branch callable: computes x plus one, asserts its graph is a control flow graph and returns it.
        i = x + 1
        self.assertTrue(i.graph.is_control_flow_graph)
        return i

      return cond_v2.cond_v2(c, then_branch, else_branch)

    i = f(constant_op.constant(True))
    self.assertEqual(self.evaluate(i), 2.0)

    i = f(constant_op.constant(False))
    self.assertEqual(self.evaluate(i), 2.0)

  def testGradientOfMixedOptionals(self):
    # [FUNCTION DOC]
    # Test: gradient of a cond whose branches also output optional values of different types (an integer
    # optional in one and a float optional in the other). The traced function takes the condition,
    # builds the cond, and returns the gradient of the first output with respect to x. With true
    # condition expects two.

    @def_function.function
    def f(c):
      # [FUNCTION DOC]
      # Traced function: takes a condition tensor, creates constant x of one, runs cond_v2 whose branches
      # return a power of x plus an optional value, ignores the optional and returns the gradient of the
      # first output with respect to x.
      x = constant_op.constant(1., name="x")

      def then_branch():
        # [FUNCTION DOC]
        # Branch callable: returns x squared and an optional wrapping the integer constant one.
        return x**2.0, gen_optional_ops.optional_from_value(
            [constant_op.constant(1)]
        )

      def else_branch():
        # [FUNCTION DOC]
        # Branch callable: returns x cubed and an optional wrapping the float constant one.
        return x**3.0, gen_optional_ops.optional_from_value(
            [constant_op.constant(1.0)]
        )

      y, _ = cond_v2.cond_v2(c, then_branch, else_branch)
      return gradients_impl.gradients(y, x)
    self.assertAllClose([2.], f(constant_op.constant(True)))


class CondV2CollectionTest(test.TestCase):
  # [CLASS DOC]
  # Test class for reading and writing graph collections inside cond_v2 branches. Each test builds a
  # fresh graph and session, with no shared fixtures.

  def testCollectionIntValueAccessInCond(self):
    # [FUNCTION DOC]
    # Test: ints stored in collections outside the cond are read inside the branch. Adds two and five to
    # collections x and y, and the branch builds constants from them and adds. Both branches are the
    # same. Expects the result seven.
    """Read values from graph collections inside of cond_v2."""
    with ops.Graph().as_default() as g:
      with self.session(graph=g):
        x = 2
        y = 5
        ops.add_to_collection("x", x)
        ops.add_to_collection("y", y)
        def fn():
          # [FUNCTION DOC]
          # Branch callable: reads values from collections x and y, makes constants and returns their sum.
          x_const = constant_op.constant(ops.get_collection("x")[0])
          y_const = constant_op.constant(ops.get_collection("y")[0])
          return math_ops.add(x_const, y_const)

        cnd = cond_v2.cond_v2(constant_op.constant(True), fn, fn)
        self.assertEqual(self.evaluate(cnd), 7)

  def testCollectionTensorValueAccessInCond(self):
    # [FUNCTION DOC]
    # Test: tensors stored in collections are read inside a cond. Constants two and five are put in
    # collections; the predicate is a comparison of them; the branch reads them from the collections and
    # adds them. Expects seven.
    """Read tensors from collections inside of cond_v2 & use them."""
    with ops.Graph().as_default() as g:
      with self.session(graph=g):
        x = constant_op.constant(2)
        y = constant_op.constant(5)
        ops.add_to_collection("x", x)
        ops.add_to_collection("y", y)

        def fn():
          # [FUNCTION DOC]
          # Branch callable: reads the tensors from collections x and y and returns their sum.
          x_read = ops.get_collection("x")[0]
          y_read = ops.get_collection("y")[0]
          return math_ops.add(x_read, y_read)

        cnd = cond_v2.cond_v2(math_ops.less(x, y), fn, fn)
        self.assertEqual(self.evaluate(cnd), 7)

  def testCollectionIntValueWriteInCond(self):
    # [FUNCTION DOC]
    # Test: a collection write inside a cond branch is visible outside. The true branch adds an integer
    # seven to collection z and returns x times the sum of x and y. Expects the result fourteen and the
    # collection z to equal a list holding seven afterwards.
    """Make sure Int writes to collections work inside of cond_v2."""
    with ops.Graph().as_default() as g:
      with self.session(graph=g):
        x = constant_op.constant(2)
        y = constant_op.constant(5)
        def true_fn():
          # [FUNCTION DOC]
          # Branch callable: computes the sum of x and y, adds seven to collection z (side effect) and returns x
          # times the sum.
          z = math_ops.add(x, y)
          ops.add_to_collection("z", 7)
          return math_ops.mul(x, z)

        def false_fn():
          # [FUNCTION DOC]
          # Branch callable: computes the sum of x and y and returns x times the sum, without touching the
          # collection.
          z = math_ops.add(x, y)
          return math_ops.mul(x, z)

        cnd = cond_v2.cond_v2(constant_op.constant(True), true_fn, false_fn)
        self.assertEqual(self.evaluate(cnd), 14)

        read_z_collection = ops.get_collection("z")
        self.assertEqual(read_z_collection, [7])


class CondV2ContainerTest(test.TestCase):
  # [CLASS DOC]
  # Test class for container handling in cond_v2 (resource container scopes for variables and queues).
  # Contains a single test that is currently skipped.

  def testContainer(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The first statement skips the test (bug reference), so everything after it never runs;
    # the body is dead code. The docstring promises container checks that are not actually performed.
    # Intended test: sets containers outside and inside cond_v2 and checks variables and FIFO queues get
    # the right container names: outside any scope empty, inside the cond branch the enclosing container
    # l1 unless a nested container scope (l2t or l2f) is used, and after leaving the scope the container
    # is empty again.
    """Set containers outside & inside of cond_v2.

    Make sure the containers are set correctly for both variable creation
    (tested by variables.Variable) and for stateful ops (tested by FIFOQueue)
    """
    self.skipTest("b/113048653")
    with ops.Graph().as_default() as g:
      with self.session(graph=g):

        v0 = variables.Variable([0])
        q0 = data_flow_ops.FIFOQueue(1, dtypes.float32)

        def container(node):
          # [FUNCTION DOC]
          # Helper local to the (skipped) test: returns the container attribute of the op that produced the
          # given tensor or queue reference.
          return node.op.get_attr("container")

        self.assertEqual(compat.as_bytes(""), container(v0))
        self.assertEqual(compat.as_bytes(""), container(q0.queue_ref))

        def true_fn():
          # [FUNCTION DOC]
          # Branch callable (inside the skipped test): creates variables and queues in the inherited l1
          # container and in a nested container l2t and asserts the container names are l1, l2t and l1.
          # Returns the constant two.
          # When this branch is created in cond below,
          # the container should begin with 'l1'
          v1 = variables.Variable([1])
          q1 = data_flow_ops.FIFOQueue(1, dtypes.float32)

          with ops.container("l2t"):
            v2 = variables.Variable([2])
            q2 = data_flow_ops.FIFOQueue(1, dtypes.float32)

          v3 = variables.Variable([1])
          q3 = data_flow_ops.FIFOQueue(1, dtypes.float32)

          self.assertEqual(compat.as_bytes("l1"), container(v1))
          self.assertEqual(compat.as_bytes("l1"), container(q1.queue_ref))
          self.assertEqual(compat.as_bytes("l2t"), container(v2))
          self.assertEqual(compat.as_bytes("l2t"), container(q2.queue_ref))
          self.assertEqual(compat.as_bytes("l1"), container(v3))
          self.assertEqual(compat.as_bytes("l1"), container(q3.queue_ref))

          return constant_op.constant(2.0)

        def false_fn():
          # [FUNCTION DOC]
          # Branch callable (inside the skipped test): same as the true branch but using nested container l2f.
          # Asserts the container names and returns the constant six.
          # When this branch is created in cond below,
          # the container should begin with 'l1'
          v1 = variables.Variable([1])
          q1 = data_flow_ops.FIFOQueue(1, dtypes.float32)

          with ops.container("l2f"):
            v2 = variables.Variable([2])
            q2 = data_flow_ops.FIFOQueue(1, dtypes.float32)

          v3 = variables.Variable([1])
          q3 = data_flow_ops.FIFOQueue(1, dtypes.float32)

          self.assertEqual(compat.as_bytes("l1"), container(v1))
          self.assertEqual(compat.as_bytes("l1"), container(q1.queue_ref))
          self.assertEqual(compat.as_bytes("l2f"), container(v2))
          self.assertEqual(compat.as_bytes("l2f"), container(q2.queue_ref))
          self.assertEqual(compat.as_bytes("l1"), container(v3))
          self.assertEqual(compat.as_bytes("l1"), container(q3.queue_ref))

          return constant_op.constant(6.0)

        with ops.container("l1"):
          cnd_true = cond_v2.cond_v2(
              constant_op.constant(True), true_fn, false_fn)
          self.assertEqual(self.evaluate(cnd_true), 2)

          cnd_false = cond_v2.cond_v2(
              constant_op.constant(False), true_fn, false_fn)
          self.assertEqual(self.evaluate(cnd_false), 6)

          v4 = variables.Variable([3])
          q4 = data_flow_ops.FIFOQueue(1, dtypes.float32)
        v5 = variables.Variable([4])
        q5 = data_flow_ops.FIFOQueue(1, dtypes.float32)

      self.assertEqual(compat.as_bytes("l1"), container(v4))
      self.assertEqual(compat.as_bytes("l1"), container(q4.queue_ref))
      self.assertEqual(compat.as_bytes(""), container(v5))
      self.assertEqual(compat.as_bytes(""), container(q5.queue_ref))


@test_util.disable_tfrt("b/171412104: This test requires distributed support.")
class CondV2ColocationGroupAndDeviceTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test class for colocation groups and device placement inside cond_v2 branches. Setup creates two
  # logical CPU devices and connects to a local one-worker cluster. Disabled for the TFRT runtime
  # because it needs distributed support. Also supports parameterized tests.

  def setUp(self):
    # [FUNCTION DOC]
    # Setup: resets the eager context, calls the parent setup, splits the first physical CPU into two
    # logical devices, creates a local cluster with one worker and no parameter servers and connects to
    # the remote worker host. Side effects: modifies global eager context state and starts servers.
    context._reset_context()
    super(CondV2ColocationGroupAndDeviceTest, self).setUp()
    cpus = context.context().list_physical_devices("CPU")
    context.context().set_logical_device_configuration(
        cpus[0], [
            context.LogicalDeviceConfiguration(),
            context.LogicalDeviceConfiguration()
        ])
    workers, _ = test_util.create_local_cluster(num_workers=1, num_ps=0)
    remote.connect_to_remote_host(workers[0].target)

  def testColocateWithBeforeCond(self):
    # [FUNCTION DOC]
    # Test: colocation requested before the cond is inherited in branches. Under colocate with constant a,
    # the branch constant must be in colocation group a; then under both a and b the group list has both
    # names. Expects the cond to evaluate to three each time.
    with ops.Graph().as_default() as g:
      with self.session(graph=g):

        a = constant_op.constant([2.0], name="a")
        b = constant_op.constant([2.0], name="b")

        def fn():
          # [FUNCTION DOC]
          # Branch callable: creates constant three, asserts its colocation groups equal group a, returns it.
          c = constant_op.constant(3.0)
          self.assertEqual([b"loc:@a"], c.op.colocation_groups())
          return c

        with ops.colocate_with(a.op):
          self.assertEqual(
              cond_v2.cond_v2(constant_op.constant(True), fn, fn).eval(), 3)

        def fn2():
          # [FUNCTION DOC]
          # Branch callable: creates constant three, asserts its colocation groups equal groups a and b, returns it.
          c = constant_op.constant(3.0)
          self.assertEqual([b"loc:@a", b"loc:@b"], c.op.colocation_groups())
          return c

        with ops.colocate_with(a.op):
          with ops.colocate_with(b.op):
            self.assertEqual(
                cond_v2.cond_v2(constant_op.constant(True), fn2, fn2).eval(), 3)

  def testColocateWithInAndOutOfCond(self):
    # [FUNCTION DOC]
    # Test: colocation inside and outside the cond. Under colocate with a, a branch that also uses
    # colocate with b asserts both groups for its constant; the cond evaluates to three. A constant made
    # afterwards outside the cond must only be in group a.
    with ops.Graph().as_default() as g:
      with self.session(graph=g):

        a = constant_op.constant([2.0], name="a")
        b = constant_op.constant([2.0], name="b")

        def fn2():
          # [FUNCTION DOC]
          # Branch callable: under colocate with b, creates constant three, asserts groups a and b, returns it.
          with ops.colocate_with(b.op):
            c = constant_op.constant(3.0)
            self.assertEqual([b"loc:@a", b"loc:@b"], c.op.colocation_groups())
            return c

        with ops.colocate_with(a.op):
          self.assertEqual(
              cond_v2.cond_v2(constant_op.constant(True), fn2, fn2).eval(), 3)

          d = constant_op.constant([2.0], name="d")
          self.assertEqual([b"loc:@a"], d.op.colocation_groups())

  def testColocateWithInCondGraphPartitioning(self):
    # [FUNCTION DOC]
    # Test: colocation forces graph partitioning. With two CPU devices, constant a is placed on CPU 0
    # and b on CPU 1. The branch adds a to itself colocated with b. Runs the cond requesting partition
    # graphs and expects at least two partitions.
    with ops.Graph().as_default() as g:
      with self.session(
          graph=g,
          config=config_pb2.ConfigProto(device_count={"CPU": 2})
      ) as sess:

        with ops.device("/device:CPU:0"):
          a = constant_op.constant([2.0], name="a")
        with ops.device("/device:CPU:1"):
          b = constant_op.constant([2.0], name="b")

        def fn():
          # [FUNCTION DOC]
          # Branch callable: adds a to itself under colocation with b and returns the result named c.
          with ops.colocate_with(b.op):
            c = math_ops.add(a, a, name="c")
          return c
        out_cond_2 = cond_v2.cond_v2(constant_op.constant(True), fn, fn)

        run_options = config_pb2.RunOptions(output_partition_graphs=True)
        run_metadata = config_pb2.RunMetadata()
        sess.run(out_cond_2, options=run_options, run_metadata=run_metadata)

        # We expect there to be two partitions because of the
        # colocate_with. We are only running the cond, which has a data
        # dependency on `a` but not on `b`. So, without the colocate_with
        # we would expect execution on just one device.
        self.assertTrue(len(run_metadata.partition_graphs) >= 2)

  def testDeviceBeforeCond(self):
    # [FUNCTION DOC]
    # Test: device placement before the cond is respected. A function builds a device placement op on the
    # CPU 0 and another in a nested device scope for CPU 1, asserting the device strings. A traced
    # wrapper calls cond_v2 under the CPU 0 device and the results contain CPU:0, CPU:1 and the job name.
    # Then a second branch checks the GPU device; if a GPU is available checks placement contains GPU:0
    # and the job, otherwise skips the test.

    def fn():
      # [FUNCTION DOC]
      # Branch callable: creates a device placement op and asserts it is on job localhost CPU 0, then under
      # device CPU 1 another one and asserts CPU 1. Returns both.
      cpu_zero_op = test_ops.device_placement_op()
      self.assertEqual("/job:localhost/device:CPU:0", cpu_zero_op.device)
      with ops.device("CPU:1"):
        cpu_one_op = test_ops.device_placement_op()
        self.assertEqual("/job:localhost/device:CPU:1", cpu_one_op.device)
      return cpu_zero_op, cpu_one_op

    @def_function.function
    def _cond_wrapper():
      # [FUNCTION DOC]
      # Traced wrapper: under the device localhost CPU 0 returns a cond_v2 with the same branch for both.
      with ops.device("/job:localhost/device:CPU:0"):
        return cond_v2.cond_v2(constant_op.constant(True), fn, fn)

    zero_expected, one_expected = self.evaluate(_cond_wrapper())
    self.assertIn(compat.as_bytes("CPU:0"), zero_expected)
    self.assertIn(compat.as_bytes("CPU:1"), one_expected)
    self.assertIn(compat.as_bytes("job:localhost"), zero_expected)
    self.assertIn(compat.as_bytes("job:localhost"), one_expected)

    def fn2():
      # [FUNCTION DOC]
      # Branch callable: asserts a constant gets device localhost GPU 0 and returns a device placement op.
      self.assertEqual("/job:localhost/device:GPU:0",
                       constant_op.constant(3.0).op.device)
      return test_ops.device_placement_op()

    @def_function.function
    def _cond_wrapper2():
      # [FUNCTION DOC]
      # Traced wrapper: under device localhost GPU 0 returns cond_v2 over the second branch.
      with ops.device("/job:localhost/device:GPU:0"):
        return cond_v2.cond_v2(constant_op.constant(True), fn2, fn2)

    if test_util.is_gpu_available():
      self.assertIn(compat.as_bytes("GPU:0"), self.evaluate(_cond_wrapper2()))
      self.assertIn(
          compat.as_bytes("job:localhost"), self.evaluate(_cond_wrapper2()))
    else:
      self.skipTest("Test requires a GPU to check GPU device placement.")

  @parameterized.named_parameters([
      dict(
          testcase_name="Function",
          functional_op_to_test=lambda fn: def_function.function(fn)()),
      dict(
          testcase_name="Cond",
          functional_op_to_test=
          lambda fn: cond_v2.cond_v2(constant_op.constant(True), fn, fn))
  ])
  def testDeviceBeforeRemote(self, functional_op_to_test):
    # [FUNCTION DOC]
    # Parameterized test (named cases Function and Cond; the parameter is a callable that applies a
    # function-calling mechanism, either a traced function or a cond_v2, to a function). Enables device
    # placement logging. Verifies that an inner device scope for the worker wins over the outer localhost
    # scope and that swapping the roles (outer worker, inner localhost) behaves the same. Checks the result
    # byte strings contain the expected job names. Side effect: sets context logging on.
    context.context().log_device_placement = True

    def _fn():
      # [FUNCTION DOC]
      # Helper local to the test: creates a device placement op, then another under the worker CPU device,
      # and returns both.
      local_op = test_ops.device_placement_op()
      with ops.device("/job:worker/CPU:0"):
        worker_op = test_ops.device_placement_op()
      return local_op, worker_op

    @def_function.function
    def _wrapper():
      # [FUNCTION DOC]
      # Traced function: under device localhost applies the functional op under test to the local helper.
      with ops.device("/job:localhost"):
        return functional_op_to_test(_fn)

    local_expected, worker_expected = self.evaluate(_wrapper())
    self.assertIn(compat.as_bytes("job:localhost"), local_expected)
    self.assertIn(compat.as_bytes("job:worker"), worker_expected)

    del _fn, _wrapper

    # There's nothing special about localhost; if we swap roles (functional op
    # on worker, op on localhost) the inner placement still wins.
    def _fn2():
      # [FUNCTION DOC]
      # Helper local to the test: creates a device placement op, then one under the localhost CPU device, and
      # returns both. Used with roles swapped.
      local_op = test_ops.device_placement_op()
      with ops.device("/job:localhost/CPU:0"):
        worker_op = test_ops.device_placement_op()
      return local_op, worker_op

    @def_function.function
    def _wrapper2():
      # [FUNCTION DOC]
      # Traced function: under device worker applies the functional op under test to the second helper.
      with ops.device("/job:worker"):
        return functional_op_to_test(_fn2)

    worker_expected, local_expected = self.evaluate(_wrapper2())
    self.assertIn(compat.as_bytes("job:worker"), worker_expected)
    self.assertIn(compat.as_bytes("job:localhost"), local_expected)

  def testColocationBeforeCond(self):
    # [FUNCTION DOC]
    # Test: colocation requested outside the cond is respected inside. A branch asserts its device
    # placement op is colocated with the op named colocation_test_op. A traced function (autograph off)
    # creates one op on CPU 0 and one on CPU 1, then runs the cond under colocation with each, and the
    # results must contain CPU:0 and CPU:1 respectively.

    def _fn():
      # [FUNCTION DOC]
      # Branch callable: creates a device placement op and asserts its first colocation group contains
      # colocation_test_op. Returns the result.
      result = test_ops.device_placement_op()
      self.assertIn("colocation_test_op",
                    result.op.colocation_groups()[0].decode())
      return result

    @def_function.function(autograph=False)
    def _cond_wrapper():
      # [FUNCTION DOC]
      # Traced function (autograph disabled): creates ops on CPU 0 and CPU 1 and runs the same cond twice,
      # colocated with each op, returning both results.
      with ops.device("/device:CPU:0"):
        op_on_cpu_0 = test_ops.device_placement_op(name="colocation_test_op")
      with ops.device("/device:CPU:1"):
        op_on_cpu_1 = test_ops.device_placement_op(name="colocation_test_op_1")
      condition = constant_op.constant(True)
      with ops.colocate_with(op_on_cpu_0.op):
        zero_expected = cond_v2.cond_v2(condition, _fn, _fn)
      with ops.colocate_with(op_on_cpu_1.op):
        one_expected = cond_v2.cond_v2(condition, _fn, _fn)
      return zero_expected, one_expected

    zero_expected, one_expected = self.evaluate(_cond_wrapper())
    self.assertIn(compat.as_bytes("CPU:0"), zero_expected)
    self.assertIn(compat.as_bytes("CPU:1"), one_expected)

  def testDeviceInAndOutOfCond(self):
    # [FUNCTION DOC]
    # Test: device scopes inside and outside the cond. With two CPU devices, a branch under device CPU 1
    # makes a constant and asserts its device. The cond is created under device CPU 0 and evaluates to
    # three, and a constant created afterwards must be on CPU 0.
    with ops.Graph().as_default() as g:
      with self.session(
          graph=g, config=config_pb2.ConfigProto(device_count={"CPU": 2})):

        def fn2():
          # [FUNCTION DOC]
          # Branch callable: under device CPU 1 creates constant three, asserts its device is CPU 1 and returns it.
          with ops.device("/device:CPU:1"):
            c = constant_op.constant(3.0)
            self.assertEqual("/device:CPU:1", c.op.device)
            return c

        with ops.device("/device:CPU:0"):
          self.assertEqual(
              cond_v2.cond_v2(constant_op.constant(True), fn2, fn2).eval(), 3)

          d = constant_op.constant(4.0)
          self.assertEqual("/device:CPU:0", d.op.device)

  def testDeviceInCondGraphPartitioning(self):
    # [FUNCTION DOC]
    # Test: device scopes force partitioning. With two CPU devices, the branch adds a to itself under
    # CPU 1 while a lives on CPU 0 with the cond. Runs requesting partition graphs and asserts at least
    # two partitions.
    with ops.Graph().as_default() as g:
      with self.session(
          graph=g,
          config=config_pb2.ConfigProto(device_count={"CPU": 2})
      ) as sess:

        def fn():
          # [FUNCTION DOC]
          # Branch callable: adds a to itself under device CPU 1 and returns the result named c.
          with ops.device("/device:CPU:1"):
            c = math_ops.add(a, a, name="c")
          return c

        with ops.device("/device:CPU:0"):
          a = constant_op.constant([2.0], name="a")
          out_cond_2 = cond_v2.cond_v2(constant_op.constant(True), fn, fn)

        run_options = config_pb2.RunOptions(output_partition_graphs=True)
        run_metadata = config_pb2.RunMetadata()
        sess.run(out_cond_2, options=run_options, run_metadata=run_metadata)

        self.assertGreaterEqual(len(run_metadata.partition_graphs), 2)


class CaseTest(test.TestCase):
  # [CLASS DOC]
  # Test class for the indexed case operation (cond_v2 module). Each test builds a graph with an index
  # and a list of branch callables and checks the produced op type and value.

  def testCase(self):
    # [FUNCTION DOC]
    # Test: indexed_case with a stateful branch (one that prints). Index zero selects the first branch and
    # the result is one. The case op is found through the output and asserted to be of the stateful type
    # named Case.

    def branch1(x):
      # [FUNCTION DOC]
      # Branch callable: prints the string one with the logging print op and returns x.
      logging_ops.print_v2("1")
      return x

    def branch2(x):
      # [FUNCTION DOC]
      # Branch callable: returns x plus one.
      return x + 1

    with ops.Graph().as_default():
      x = array_ops.constant(1)
      output = cond_v2.indexed_case(
          array_ops.constant(0), [lambda: branch1(x), lambda: branch2(x)])
      cond_op = output.op.inputs[0].op
      self.assertEqual(cond_op.type, "Case")
      self.assertEqual(1., self.evaluate(output))

  def testStatelessCase(self):
    # [FUNCTION DOC]
    # Test: indexed_case with stateless branches and index zero. The op type is the stateless case and the
    # result is two.

    def branch1(x):
      # [FUNCTION DOC]
      # Branch callable: returns x plus one.
      return x + 1

    def branch2(x):
      # [FUNCTION DOC]
      # Branch callable: returns x plus two.
      return x + 2

    with ops.Graph().as_default():
      x = array_ops.constant(1)
      output = cond_v2.indexed_case(
          array_ops.constant(0), [lambda: branch1(x), lambda: branch2(x)])
      cond_op = output.op.inputs[0].op
      self.assertEqual(cond_op.type, "StatelessCase")
      self.assertEqual(2., self.evaluate(output))


def _cond(pred, true_fn, false_fn, name):
  # [FUNCTION DOC]
  # HELPER: Test helper that creates a conditional with the given predicate, true and false callables
  # and name. Uses the legacy cond when the active graph is inside a legacy cond context, otherwise
  # cond_v2. Returns the cond result. Used by nested cond tests.
  if _is_old_cond():
    return tf_cond.cond(pred, true_fn, false_fn, name=name)
  else:
    return cond_v2.cond_v2(pred, true_fn, false_fn, name=name)


def _is_old_cond():
  # [FUNCTION DOC]
  # HELPER: Returns true if the default graph's current control flow context is a legacy cond context
  # (so a legacy cond is being built), false otherwise.
  return isinstance(ops.get_default_graph()._get_control_flow_context(),
                    control_flow_ops.CondContext)


def _has_node_with_op(run_metadata, op_type):
  # [FUNCTION DOC]
  # HELPER: Test helper that checks whether any node in any partition graph of the run metadata has the
  # given op type name. Takes the run metadata and the op type string; loops over partition graphs and
  # their nodes and returns true on the first match, otherwise false.
  """Whether any node in `run_metadata.partition_graphs` matches `op_type`."""
  for graph in run_metadata.partition_graphs:
    for node in graph.node:
      if node.op == op_type:
        return True
  return False


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
