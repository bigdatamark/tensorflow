# [MODULE DOC]
# KNOWN ISSUE: the module docstring says tests for tf.py, which does not describe the content.
# Tests for TensorFlow variables: the Variable and VariableV1 classes, partitioned variables,
# initialization reporting, containers and aggregation enums. Test classes run under the standard
# TensorFlow test runner, many restricted to legacy graph mode or version 2 only. Contains a
# small shared helper for initialized values. No import-time side effects.
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
"""Tests for tf.py."""

import functools
import operator

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent
from tensorflow.python.util import compat


def initialized_value(var):
  # [FUNCTION DOC]
  # HELPER: shared test utility that mimics the removed initialized-value accessor of variables.
  # Input: var, a graph-mode variable. Builds a conditional op returning the variable's current value
  # if the variable is already initialized, otherwise its initial value tensor. Returns that tensor.
  # Adds ops to the default graph; no other side effects.
  return tf_cond.cond(
      variable_v1.is_variable_initialized(var), var.read_value,
      lambda: var.initial_value)


class VariablesTestCase(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Main test case for the Variable and VariableV1 classes: initialization and naming, assignment,
  # collections, operator overloads, device placement, proto round trips, trainable flags and
  # save-slice info parsing. Mixes graph-only (legacy) tests, v2-only tests and parameterized tests;
  # most create variables inside a cached session and evaluate them.

  @test_util.run_deprecated_v1
  def testDistributeStrategy(self):
    # [FUNCTION DOC]
    # Legacy graph test: a freshly created V1 variable has no distribution strategy; asserts the
    # private strategy attribute is null.
    v = variable_v1.VariableV1(0.0)
    self.assertIsNone(v._distribute_strategy)

  @test_util.run_v1_only("b/120545219")
  def testInitialization(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: several shape assertions are duplicated verbatim (get_shape is checked twice).
    # Legacy graph test: creates two scalar variables and checks default names, shared names, scalar
    # shapes, that evaluating before initialization raises an op error about uninitialized value,
    # and that after running the global initializer the values equal the initial values.
    with self.cached_session():
      var0 = variable_v1.VariableV1(0.0)
      self.assertEqual("Variable:0", var0.name)
      self.assertEqual("Variable", var0._shared_name)
      self.assertEqual([], var0.get_shape())
      self.assertEqual([], var0.get_shape())
      self.assertEqual([], var0.shape)

      var1 = variable_v1.VariableV1(1.1)
      self.assertEqual("Variable_1:0", var1.name)
      self.assertEqual("Variable_1", var1._shared_name)
      self.assertEqual([], var1.get_shape())
      self.assertEqual([], var1.get_shape())
      self.assertEqual([], var1.shape)

      with self.assertRaisesOpError("Attempting to use uninitialized value"):
        self.evaluate(var0)

      with self.assertRaisesOpError("Attempting to use uninitialized value"):
        self.evaluate(var1)

      self.evaluate(variables.global_variables_initializer())

      self.assertAllClose(0.0, self.evaluate(var0))
      self.assertAllClose(1.1, self.evaluate(var1))

  @test_util.run_v1_only("b/120545219")
  def testInitializationOrder(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: shape assertions are duplicated (get_shape checked twice for each variable).
    # Legacy graph test: a random variable, a variable initialized from its initialized value, and a
    # third built from a sum of both; checks names and shapes, then after global initialization
    # that the dependent variables hold the expected values (equal copy, and sum plus two).
    with self.cached_session():
      rnd = variables.Variable(random_ops.random_uniform([3, 6]), name="rnd")
      self.assertEqual("rnd:0", rnd.name)
      self.assertEqual([3, 6], rnd.get_shape())
      self.assertEqual([3, 6], rnd.get_shape())
      self.assertEqual([3, 6], rnd.shape)

      dep = variables.Variable(initialized_value(rnd), name="dep")
      self.assertEqual("dep:0", dep.name)
      self.assertEqual([3, 6], dep.get_shape())
      self.assertEqual([3, 6], dep.get_shape())
      self.assertEqual([3, 6], dep.shape)

      # Currently have to set the shape manually for Add.
      added_val = initialized_value(rnd) + initialized_value(dep) + 2.0
      added_val.set_shape(rnd.get_shape())

      depdep = variables.Variable(added_val, name="depdep")
      self.assertEqual("depdep:0", depdep.name)
      self.assertEqual([3, 6], depdep.get_shape())
      self.assertEqual([3, 6], depdep.get_shape())
      self.assertEqual([3, 6], depdep.shape)

      self.evaluate(variables.global_variables_initializer())

      self.assertAllClose(self.evaluate(rnd), self.evaluate(dep))
      self.assertAllClose(
          self.evaluate(rnd) + self.evaluate(dep) + 2.0, self.evaluate(depdep))

  @test_util.run_deprecated_v1
  def testCyclicInitializer(self):
    # [FUNCTION DOC]
    # Legacy graph test: a while loop output used as initial value is returned unchanged by the
    # private guard against uninitialized dependencies, i.e. no cycle error or rewriting.
    with self.cached_session():
      cyclic = while_loop.while_loop(
          cond=lambda i: i < 10,
          body=lambda i: i + 1,
          loop_vars=(constant_op.constant(0),))
      initial_value = variables._try_guard_against_uninitialized_dependencies(
          "test", cyclic)
      self.assertIs(initial_value, cyclic)

  @test_util.run_deprecated_v1
  def testIterableV1(self):
    # [FUNCTION DOC]
    # Legacy graph test: iterating over a scalar or vector variable in graph mode raises a type
    # error saying iteration is not allowed in graph mode.
    with self.assertRaisesRegex(TypeError, "not allowed.*Graph mode"):
      for _ in variables.Variable(0.0):
        pass
    with self.assertRaisesRegex(TypeError, "not allowed.*Graph mode"):
      for _ in variables.Variable([0.0, 1.0]):
        pass

  @test_util.run_v2_only
  def testIterableV2(self):
    # [FUNCTION DOC]
    # Version 2 only test: iterating over a scalar variable raises a type error mentioning scalar
    # tensor; iterating over a two-element vector variable yields its values, compared to the list.
    with self.assertRaisesRegex(TypeError, "scalar tensor"):
      for _ in variables.Variable(0.0):
        pass
    values = []
    for v in variables.Variable([0.0, 1.0]):
      values.append(v)
    self.assertAllClose([0., 1.], values)

  @test_util.run_deprecated_v1
  def testAssignments(self):
    # [FUNCTION DOC]
    # Legacy graph test: add, subtract and assign ops on a scalar variable; after initialization
    # checks each op's returned value and the variable value (0, then 1, -1, 4).
    with self.cached_session():
      var = variables.Variable(0.0)
      plus_one = var.assign_add(1.0)
      minus_one = var.assign_sub(2.0)
      four = var.assign(4.0)
      self.evaluate(variables.global_variables_initializer())
      self.assertAllClose(0.0, self.evaluate(var))

      self.assertAllClose(1.0, self.evaluate(plus_one))
      self.assertAllClose(1.0, self.evaluate(var))

      self.assertAllClose(-1.0, self.evaluate(minus_one))
      self.assertAllClose(-1.0, self.evaluate(var))

      self.assertAllClose(4.0, self.evaluate(four))
      self.assertAllClose(4.0, self.evaluate(var))

  @test_util.run_deprecated_v1
  def testResourceAssignments(self):
    # [FUNCTION DOC]
    # Legacy graph test: same add/subtract/assign sequence as the plain variable test but with a
    # resource variable, running each op through the session and checking the variable value.
    with self.session():
      var = resource_variable_ops.ResourceVariable(0.0)
      plus_one = var.assign_add(1.0)
      minus_one = var.assign_sub(2.0)
      four = var.assign(4.0)
      self.evaluate(variables.global_variables_initializer())
      self.assertAllClose(0.0, self.evaluate(var))

      self.evaluate(plus_one)
      self.assertAllClose(1.0, self.evaluate(var))

      self.evaluate(minus_one)
      self.assertAllClose(-1.0, self.evaluate(var))

      self.evaluate(four)
      self.assertAllClose(4.0, self.evaluate(var))

  def testAssignDifferentShapesEagerNotAllowed(self):
    # [FUNCTION DOC]
    # Eager mode: assigning an array of a different shape to a fixed-shape variable raises a value
    # error about incompatible shapes.
    with context.eager_mode():
      var = variables.Variable(np.zeros(shape=[1, 1]))
      with self.assertRaisesRegex(ValueError, "shape.*and.*are incompatible"):
        var.assign(np.zeros(shape=[2, 2]))

  @test_util.run_in_graph_and_eager_modes
  def testAssignDifferentShapesAllowed(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes: a variable created with an unknown shape accepts assignment
    # of a differently shaped array; the value is read back after each assignment and compared.
    var = variables.Variable(np.zeros(shape=[1, 1]),
                             shape=tensor_shape.TensorShape(None))
    self.evaluate(variables.global_variables_initializer())
    self.assertAllEqual(np.zeros(shape=[1, 1]), var.read_value())
    self.evaluate(var.assign(np.zeros(shape=[2, 2])))
    self.assertAllEqual(np.zeros(shape=[2, 2]), var.read_value())

  def testZeroSizeStringAssign(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the session bound as sess is never used (evaluation goes through the test helper).
    # Creates a non-trainable local string variable of size zero, runs the local initializer,
    # assigns it its own value and asserts the assignment result is an empty list.
    with self.cached_session() as sess:
      array = variable_v1.VariableV1(
          initial_value=array_ops.zeros((0,), dtype=dtypes.string),
          name="foo",
          trainable=False,
          collections=[ops.GraphKeys.LOCAL_VARIABLES])
      self.evaluate(variables.local_variables_initializer())
      old_value = array.value()
      copy_op = array.assign(old_value)
      self.assertEqual([], list(self.evaluate(copy_op)))

  def _countUpToTest(self, dtype):
    # [FUNCTION DOC]
    # HELPER: shared body for the count-up-to tests, parameterized by data type.
    # Input: dtype, the integer type of the variable. Builds a variable starting at zero and a
    # count-up-to-3 op; asserts the op returns 0, 1, 2 while the variable increments to 3, and that
    # further runs raise an op error about reaching the limit while the variable stays at 3.
    with self.cached_session():
      zero = constant_op.constant(0, dtype=dtype)
      var = variables.Variable(zero)
      count_up_to = var.count_up_to(3)

      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(0, self.evaluate(var))

      self.assertEqual(0, self.evaluate(count_up_to))
      self.assertEqual(1, self.evaluate(var))

      self.assertEqual(1, self.evaluate(count_up_to))
      self.assertEqual(2, self.evaluate(var))

      self.assertEqual(2, self.evaluate(count_up_to))
      self.assertEqual(3, self.evaluate(var))

      with self.assertRaisesOpError("Reached limit of 3"):
        self.evaluate(count_up_to)
      self.assertEqual(3, self.evaluate(var))

      with self.assertRaisesOpError("Reached limit of 3"):
        self.evaluate(count_up_to)
      self.assertEqual(3, self.evaluate(var))

  @test_util.run_deprecated_v1
  def testCountUpToInt32(self):
    # [FUNCTION DOC]
    # Legacy graph test: runs the count-up-to helper with 32-bit integers.
    self._countUpToTest(dtypes.int32)

  @test_util.run_deprecated_v1
  def testCountUpToInt64(self):
    # [FUNCTION DOC]
    # Legacy graph test: runs the count-up-to helper with 64-bit integers.
    self._countUpToTest(dtypes.int64)

  @test_util.run_v1_only("b/120545219")
  def testControlDepsNone(self):
    # [FUNCTION DOC]
    # Legacy graph test: inside a control dependency scope an ordinary constant gets the dependency,
    # but a variable's initializer, value op and reference op have no control inputs.
    with self.cached_session():
      c = constant_op.constant(1.0)
      with ops.control_dependencies([c]):
        # d get the control dep.
        d = constant_op.constant(2.0)
        # variables do not.
        var_x = variable_v1.VariableV1(2.0)
      self.assertEqual([c.op], d.op.control_inputs)
      self.assertEqual([], var_x.initializer.control_inputs)
      self.assertEqual([], var_x.value().op.control_inputs)
      self.assertEqual([], var_x._ref().op.control_inputs)  # pylint: disable=protected-access

  @test_util.run_v1_only("b/120545219")
  def testControlFlow(self):
    # [FUNCTION DOC]
    # Legacy graph test: variables created inside both branches of a conditional can be initialized
    # and read without initializing an outer variable; running the conditional result fails with an
    # uninitialized error until the outer variable is initialized, after which it succeeds.
    with self.cached_session() as sess:
      v0 = variables.Variable(0, name="v0")
      var_dict = {}

      # Call get_variable in each of the cond clauses.
      def var_in_then_clause():
        # [FUNCTION DOC]
        # HELPER: true-branch callback for the conditional in the control flow test.
        # Creates variable v1 (initial value 1), stores it in the enclosing mapping under its key and
        # returns it added to the outer variable v0.
        v1 = variables.Variable(1, name="v1")
        var_dict["v1"] = v1
        return v1 + v0

      def var_in_else_clause():
        # [FUNCTION DOC]
        # HELPER: false-branch callback for the conditional in the control flow test.
        # Creates variable v2 (initial value 2), stores it in the enclosing mapping under its key and
        # returns it added to the outer variable v0.
        v2 = variables.Variable(2, name="v2")
        var_dict["v2"] = v2
        return v2 + v0

      add = tf_cond.cond(
          math_ops.less(v0, 10), var_in_then_clause, var_in_else_clause)
      v1 = var_dict["v1"]
      v2 = var_dict["v2"]
      # We should be able to initialize and run v1 and v2 without initializing
      # v0, even if the variable was created with a control dep on v0.
      self.evaluate(v1.initializer)
      self.assertEqual([1], self.evaluate(v1))
      self.evaluate(v2.initializer)
      self.assertEqual([2], self.evaluate(v2))
      # v0 should still be uninitialized.
      with self.assertRaisesRegex(errors_impl.OpError, "uninitialized"):
        self.evaluate(v0)
      # We should not be able to run 'add' yet.
      with self.assertRaisesRegex(errors_impl.OpError, "uninitialized"):
        self.evaluate(add)
      # If we initialize v0 we should be able to run 'add'.
      self.evaluate(v0.initializer)
      self.evaluate(add)

  @test_util.run_v1_only("b/120545219")
  def testControlFlowInitialization(self):
    # [FUNCTION DOC]
    # Legacy graph test: creating a variable inside a while loop body raises a value error stating
    # that the initializer is inside a control-flow construct.
    """Expects an error if an initializer is in a control-flow scope."""
    def cond(i, _):
      # [FUNCTION DOC]
      # HELPER: loop condition callback; returns whether the counter (first argument) is below 10;
      # the second loop variable is ignored.
      return i < 10

    def body(i, _):
      # [FUNCTION DOC]
      # HELPER: loop body callback. Creates a variable from a zero scalar integer (the forbidden
      # action under test) and returns the incremented counter with the variable's read value;
      # the second loop variable input is ignored.
      zero = array_ops.zeros([], dtype=dtypes.int32)
      v = variables.Variable(initial_value=zero)
      return (i + 1, v.read_value())

    with self.assertRaisesRegex(ValueError, "inside a control-flow"):
      while_loop.while_loop(cond, body, [0, 0])

  @test_util.run_deprecated_v1
  def testUseVariableAsTensor(self):
    # [FUNCTION DOC]
    # Legacy graph test: two scalar variables passed directly to the add op give their sum (5).
    with self.cached_session():
      var_x = variables.Variable(2.0)
      var_y = variables.Variable(3.0)
      self.evaluate(variables.global_variables_initializer())
      self.assertAllClose(2.0, self.evaluate(var_x))
      self.assertAllClose(3.0, self.evaluate(var_y))
      self.assertAllClose(5.0, self.evaluate(math_ops.add(var_x, var_y)))

  @test_util.run_deprecated_v1
  def testZeroSizeVarSameAsConst(self):
    # [FUNCTION DOC]
    # Legacy graph test: a matrix product with a zero-size variable gives the same result as with a
    # zero-size constant, and equals a two-by-two matrix of zeros.
    with self.cached_session():
      zero_size_var = variables.Variable(array_ops.zeros([0, 2]))
      zero_size_const = array_ops.ones([2, 0])
      variable_mul = math_ops.matmul(zero_size_const, zero_size_var)
      const_mul = math_ops.matmul(
          zero_size_const, zero_size_const, transpose_b=True)
      self.evaluate(variables.global_variables_initializer())
      variable_output = self.evaluate(variable_mul)
      self.assertAllClose(self.evaluate(const_mul), variable_output)
      self.assertAllClose([[0., 0.], [0., 0.]], variable_output)

  @test_util.run_deprecated_v1
  def testCachingDevice(self):
    # [FUNCTION DOC]
    # Legacy graph test: a variable with a caching device keeps its own device but its value tensor is
    # placed on the caching device; a variable without it has the same device as its initialized value.
    with self.cached_session():
      var = variables.Variable(2.0)
      self.assertEqual(var.device, initialized_value(var).device)

      var_cached = variables.Variable(2.0, caching_device="/job:foo")
      self.assertFalse(var_cached.device.startswith("/job:foo"))
      self.assertTrue(var_cached.value().device.startswith("/job:foo"))

  @test_util.run_deprecated_v1
  def testCollections(self):
    # [FUNCTION DOC]
    # Legacy graph test: checks which variables show up in the global and trainable collections for
    # default, non-trainable, trainable and explicitly specified collections (no duplicates).
    with self.cached_session():
      var_x = variable_v1.VariableV1(2.0)
      var_y = variable_v1.VariableV1(2.0, trainable=False)
      var_z = variable_v1.VariableV1(2.0, trainable=True)
      var_t = variable_v1.VariableV1(
          2.0,
          trainable=True,
          collections=[
              ops.GraphKeys.TRAINABLE_VARIABLES, ops.GraphKeys.GLOBAL_VARIABLES
          ])
      self.assertEqual([var_x, var_y, var_z, var_t],
                       variables.global_variables())
      self.assertEqual([var_x, var_z, var_t], variables.trainable_variables())

  @test_util.run_deprecated_v1
  def testCollectionsWithScope(self):
    # [FUNCTION DOC]
    # Legacy graph test: variables created in two name scopes are filtered by scope name when listing
    # global and trainable variables.
    with self.cached_session():
      with ops.name_scope("scope_1"):
        var_x = variable_v1.VariableV1(2.0)
      with ops.name_scope("scope_2"):
        var_y = variable_v1.VariableV1(2.0)

      self.assertEqual([var_x, var_y], variables.global_variables())
      self.assertEqual([var_x], variables.global_variables("scope_1"))
      self.assertEqual([var_y], variables.global_variables("scope_2"))

      self.assertEqual([var_x, var_y], variables.trainable_variables())
      self.assertEqual([var_x], variables.trainable_variables("scope_1"))
      self.assertEqual([var_y], variables.trainable_variables("scope_2"))

  def testOperatorWrapping(self):
    # [FUNCTION DOC]
    # Checks that each wrapped-function attribute (name, doc and similar) of the variable addition
    # operator equals that of the tensor addition operator.
    for attr in functools.WRAPPER_ASSIGNMENTS:
      self.assertEqual(
          getattr(variables.Variable.__add__, attr),
          getattr(tensor.Tensor.__add__, attr))

  @test_util.run_deprecated_v1
  def testOperators(self):
    # [FUNCTION DOC]
    # Legacy graph test: applies arithmetic, reflected arithmetic, comparison, negation, absolute
    # value, modulo, boolean logic, invert, slicing and matrix multiplication operators to variables
    # and checks the evaluated results against hand-computed values.
    with self.cached_session():
      var_f = variables.Variable([2.0])
      add = var_f + 0.0
      radd = 1.0 + var_f
      sub = var_f - 1.0
      rsub = 1.0 - var_f
      mul = var_f * 10.0
      rmul = 10.0 * var_f
      div = var_f / 10.0
      rdiv = 10.0 / var_f
      lt = var_f < 3.0
      rlt = 3.0 < var_f
      le = var_f <= 2.0
      rle = 2.0 <= var_f
      gt = var_f > 3.0
      rgt = 3.0 > var_f
      ge = var_f >= 2.0
      rge = 2.0 >= var_f
      neg = -var_f
      abs_v = abs(var_f)

      var_i = variables.Variable([20])
      mod = var_i % 7
      rmod = 103 % var_i

      var_b = variables.Variable([True, False])
      and_v = operator.and_(var_b, [True, True])
      or_v = operator.or_(var_b, [False, True])
      xor_v = operator.xor(var_b, [False, False])
      invert_v = ~var_b

      rnd = np.random.rand(4, 4).astype("f")
      var_t = variables.Variable(rnd)
      slice_v = var_t[2, 0:0]

      var_m = variables.Variable([[2.0, 3.0]])
      matmul = var_m.__matmul__([[10.0], [20.0]])
      rmatmul = var_m.__rmatmul__([[10.0], [20.0]])

      self.evaluate(variables.global_variables_initializer())
      self.assertAllClose([2.0], self.evaluate(add))
      self.assertAllClose([3.0], self.evaluate(radd))
      self.assertAllClose([1.0], self.evaluate(sub))
      self.assertAllClose([-1.0], self.evaluate(rsub))
      self.assertAllClose([20.0], self.evaluate(mul))
      self.assertAllClose([20.0], self.evaluate(rmul))
      self.assertAllClose([0.2], self.evaluate(div))
      self.assertAllClose([5.0], self.evaluate(rdiv))
      self.assertAllClose([-2.0], self.evaluate(neg))
      self.assertAllClose([2.0], self.evaluate(abs_v))
      self.assertAllClose([True], self.evaluate(lt))
      self.assertAllClose([False], self.evaluate(rlt))
      self.assertAllClose([True], self.evaluate(le))
      self.assertAllClose([True], self.evaluate(rle))
      self.assertAllClose([False], self.evaluate(gt))
      self.assertAllClose([True], self.evaluate(rgt))
      self.assertAllClose([True], self.evaluate(ge))
      self.assertAllClose([True], self.evaluate(rge))

      self.assertAllClose([6], self.evaluate(mod))
      self.assertAllClose([3], self.evaluate(rmod))

      self.assertAllClose([True, False], self.evaluate(and_v))
      self.assertAllClose([True, True], self.evaluate(or_v))
      self.assertAllClose([True, False], self.evaluate(xor_v))
      self.assertAllClose([False, True], self.evaluate(invert_v))

      self.assertAllClose(rnd[2, 0:0], self.evaluate(slice_v))

      self.assertAllClose([[80.0]], self.evaluate(matmul))
      self.assertAllClose([[20.0, 30.0], [40.0, 60.0]], self.evaluate(rmatmul))

  @test_util.run_deprecated_v1
  def testSession(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name suggests session behavior but it only initializes a variable and checks
    # its value; the session bound as sess is unused.
    # Legacy graph test: creates a vector variable, initializes and evaluates it.
    with self.cached_session() as sess:
      var = variables.Variable([1, 12])
      self.evaluate(variables.global_variables_initializer())
      self.assertAllClose([1, 12], self.evaluate(var))

  @test_util.run_v1_only("b/120545219")
  def testColocation(self):
    # [FUNCTION DOC]
    # Legacy graph test: an assign op created under a different device scope than its variable is
    # colocated with the variable (device and colocation group of the variable).
    with ops.device("/job:ps"):
      var = variable_v1.VariableV1(0, name="v")
    with ops.device("/job:worker/task:7"):
      assign_op = var.assign(1)
    self.assertDeviceEqual("/job:ps", assign_op.device)
    self.assertEqual([b"loc:@v"], assign_op.op.colocation_groups())

  @test_util.run_v1_only("b/120545219")
  def testInitializerFunction(self):
    # [FUNCTION DOC]
    # Legacy graph test: a variable created from an initializer callable has the right shape and
    # initial value but fails with a failed-precondition error until initialized; a second variable
    # initialized from the negation of the first's initialized value matches shape and negated value.
    value = [[-42], [133.7]]
    shape = [2, 1]
    with self.cached_session():
      initializer = lambda: constant_op.constant(value)

      v1 = variables.Variable(initializer, dtype=dtypes.float32)
      self.assertEqual(shape, v1.get_shape())
      self.assertEqual(shape, v1.shape)
      self.assertAllClose(value, self.evaluate(v1.initial_value))
      with self.assertRaises(errors_impl.FailedPreconditionError):
        self.evaluate(v1)

      v2 = variables.Variable(
          math_ops.negative(initialized_value(v1)), dtype=dtypes.float32)
      self.assertEqual(v1.get_shape(), v2.get_shape())
      self.assertEqual(v1.shape, v2.shape)
      self.assertAllClose(np.negative(value), self.evaluate(v2.initial_value))

      with self.assertRaises(errors_impl.FailedPreconditionError):
        self.evaluate(v2)
      self.evaluate(variables.global_variables_initializer())
      self.assertAllClose(np.negative(value), self.evaluate(v2))

  def testConstraintArg(self):
    # [FUNCTION DOC]
    # A callable constraint is stored on the variable; a non-callable constraint (zero) raises a
    # value error.
    constraint = lambda x: x
    v = variables.Variable(
        lambda: constant_op.constant(1.),
        constraint=constraint)
    self.assertEqual(v.constraint, constraint)

    constraint = 0
    with self.assertRaises(ValueError):
      v = variables.Variable(
          lambda: constant_op.constant(1.),
          constraint=constraint)

  @test_util.run_v1_only("b/120545219")
  def testNoRefDataRace(self):
    # [FUNCTION DOC]
    # Legacy graph test: chained variables each initialized from the previous one's initialized value
    # plus two get the expected values after one global initialization.
    with self.cached_session():
      a = variables.Variable([1, 2, 3], dtype=dtypes.float32)
      b = variables.Variable(initialized_value(a) + 2)
      c = variables.Variable(initialized_value(b) + 2)
      self.evaluate(variables.global_variables_initializer())
      self.assertAllEqual(self.evaluate(a), [1, 2, 3])
      self.assertAllEqual(self.evaluate(b), [3, 4, 5])
      self.assertAllEqual(self.evaluate(c), [5, 6, 7])

  @test_util.run_deprecated_v1
  def testInitializerFunctionDevicePlacement(self):
    # [FUNCTION DOC]
    # Legacy graph test: a variable created under a device scope has that device and is colocated
    # with itself; its initializer inputs are colocated too, as also for one created without a scope.
    with self.cached_session():
      initializer = lambda: constant_op.constant(42.0)
      with ops.device("/cpu:100"):
        v1 = variables.Variable(initializer, dtype=dtypes.float32, name="v1")
      expected_device = "/device:CPU:100"
      expected_group_v1 = [b"loc:@v1"]
      self.assertEqual(expected_device, v1.op.device)
      self.assertEqual(expected_group_v1, v1.op.colocation_groups())
      for i in v1.initializer.inputs:
        self.assertEqual(expected_group_v1, i.op.colocation_groups())

      v2 = variables.Variable(initializer, dtype=dtypes.float32, name="v2")
      expected_group_v2 = [b"loc:@v2"]
      self.assertEqual(expected_group_v2, v2.op.colocation_groups())
      for i in v2.initializer.inputs:
        self.assertEqual(expected_group_v2, i.op.colocation_groups())

  @test_util.run_v1_only("b/120545219")
  def testVariableDefInitializedInstances(self):
    # [FUNCTION DOC]
    # Legacy graph test: a variable restored from a variable definition message without an initial
    # value can still compute initialized value (3), does not re-run the initializer once assigned,
    # and when the initial-value-name field is cleared it round trips to the same message but
    # using its initialized value raises a value error.
    with ops.Graph().as_default(), self.cached_session() as sess:
      v_def = variables.Variable(
          initial_value=constant_op.constant(3.0)).to_proto()

    with ops.Graph().as_default(), self.cached_session() as sess:
      # v describes a VariableDef-based variable without an initial value.
      v = variables.Variable(variable_def=v_def)
      self.assertEqual(3.0, self.evaluate(initialized_value(v)))

      # initialized_value should not rerun the initializer_op if the variable
      # has already been initialized elsewhere.
      self.evaluate(v.assign(1.0))
      self.assertEqual(1.0, self.evaluate(initialized_value(v)))

    v_def.ClearField("initial_value_name")
    with ops.Graph().as_default(), self.cached_session() as sess:
      # Restoring a legacy VariableDef proto that does not have
      # initial_value_name set should still work.
      v = variables.Variable(variable_def=v_def)
      # We should also be able to re-export the variable to a new meta graph.
      self.assertProtoEquals(v_def, v.to_proto())
      # But attempts to use initialized_value will result in errors.
      with self.assertRaises(ValueError):
        self.evaluate(initialized_value(v))

  def testTrainableInProto(self):
    # [FUNCTION DOC]
    # Checks the trainable flag survives a round trip through the variable definition message for
    # both trainable and non-trainable variables.
    with ops.Graph().as_default():
      non_trainable_variable = variables.Variable(
          trainable=False,
          initial_value=constant_op.constant(10.0))
      self.assertEqual(
          False,
          variables.Variable(variable_def=non_trainable_variable.to_proto())
          .trainable)
      trainable_variable = variables.Variable(
          trainable=True,
          initial_value=constant_op.constant(10.0))
      self.assertEqual(
          True,
          variables.Variable(variable_def=trainable_variable.to_proto())
          .trainable)

  def testSynchronizationAndAggregationSaved(self):
    # [FUNCTION DOC]
    # Checks synchronization (none) and aggregation (only first replica) are kept on the variable
    # and again after a round trip through the variable definition message.
    with ops.Graph().as_default():
      original_variable = variables.Variable(
          initial_value=constant_op.constant(10.0),
          synchronization=variables.VariableSynchronization.NONE,
          aggregation=variables.VariableAggregationV2.ONLY_FIRST_REPLICA)
      self.assertEqual(variables.VariableSynchronization.NONE,
                       original_variable.synchronization)
      self.assertEqual(variables.VariableAggregation.ONLY_FIRST_REPLICA,
                       original_variable.aggregation)

      laundered = variables.Variable(
          variable_def=original_variable.to_proto())
      self.assertEqual(
          variables.VariableSynchronization.NONE,
          laundered.synchronization)
      self.assertEqual(variables.VariableAggregationV2.ONLY_FIRST_REPLICA,
                       laundered.aggregation)

  @test_util.run_deprecated_v1
  def testLoad(self):
    # [FUNCTION DOC]
    # Legacy graph test: the load method overwrites a zero matrix variable with ones; verified by
    # evaluation.
    with self.cached_session():
      var = variables.Variable(np.zeros((5, 5), np.float32))
      self.evaluate(variables.global_variables_initializer())
      var.load(np.ones((5, 5), np.float32))

      self.assertAllClose(np.ones((5, 5), np.float32), self.evaluate(var))

  @test_util.run_v1_only("b/120545219")
  def testRepr(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the expected string includes a reference-dtype suffix, which only holds for
    # legacy reference variables (hence the v1-only restriction).
    # Legacy graph test: checks the exact text representation of a named V1 matrix variable.
    var = variable_v1.VariableV1(np.zeros((5, 5), np.float32), name="noop")
    self.assertEqual(
        "<tf.Variable 'noop:0' shape=(5, 5) dtype=float32_ref>",
        repr(var))

  def testVariableNamesPreserveNameScopesWithFunction(self):
    # [FUNCTION DOC]
    # Creates a variable the first time a traced function is run inside a name scope and checks the
    # variable name keeps the scope prefix (foo/bar:0), invoked inside the default graph context.
    v = None

    @def_function.function
    def create_variable():
      # [FUNCTION DOC]
      # HELPER: traced function used by the name scope test. Inside a name scope, creates the
      # enclosing variable only on the first call, then asserts its name has the scope prefix.
      # Takes no input and returns nothing; mutates the enclosing variable binding.
      with ops.name_scope("foo"):
        nonlocal v
        if v is None:
          v = variables.Variable(0.0, name="bar")
      self.assertEqual(v.name, "foo/bar:0")
    with ops.get_default_graph().as_default():
      create_variable()

  @parameterized.parameters(variable_v1.VariableV1, variables.Variable)
  def testTrainableVariable(self, cls):
    # [FUNCTION DOC]
    # Parameterized over the V1 and V2 variable classes: a default variable is trainable; with
    # on-read synchronization it defaults to non-trainable; an explicit trainable flag overrides that
    # in either direction. Input cls is the variable class under test.
    v1 = cls(1.0)
    self.assertEqual(True, v1.trainable)

    v2 = cls(1.0, synchronization=variables.VariableSynchronization.ON_READ)
    self.assertEqual(False, v2.trainable)

    v3 = cls(1.0, synchronization=variables.VariableSynchronization.ON_READ,
             trainable=True)
    self.assertEqual(True, v3.trainable)

    v4 = cls(1.0, synchronization=variables.VariableSynchronization.ON_READ,
             trainable=False)
    self.assertEqual(False, v4.trainable)

  def testSaveSliceInfoFromSpecPasses(self):
    # [FUNCTION DOC]
    # Round trip: parsing the spec string of a save slice info yields an object with the same spec.
    save_slice_info = variables.Variable.SaveSliceInfo(
        full_name="foo",
        full_shape=[2, 3, 4],
        var_offset=[0, 2, 0],
        var_shape=[1, 1, 3])

    save_slice_info_from_spec = variables.Variable.SaveSliceInfo.from_spec(
        save_slice_info.spec)

    self.assertEqual(save_slice_info.spec, save_slice_info_from_spec.spec)

  @parameterized.parameters(
      dict(spec="0", error_message="contain space-separated full_shape info"),
      dict(spec="0:0", error_message="contain space-separated full_shape info"),
      dict(spec="a b", error_message="full_shape must be a sequence of int"),
      dict(spec="0 0:0", error_message="comma-separated pairs of offsets and"),
      dict(spec="0 a,0:0:0", error_message="var_offset must be an integer"),
      dict(spec="0 0,a:0:0", error_message="var_shape must be an integer"))
  def testSaveSliceInfoFromSpecFails(self, spec, error_message):
    # [FUNCTION DOC]
    # Parameterized over malformed spec strings: each makes the spec parser raise a value error whose
    # message matches the given pattern. Inputs: spec, the bad string; error_message, expected text.
    with self.assertRaisesRegex(ValueError, error_message):
      variables.Variable.SaveSliceInfo.from_spec(spec)


class IsInitializedTest(test.TestCase):
  # [CLASS DOC]
  # Tests for reporting uninitialized variables and training with zero-size variables, each in its
  # own graph with a session.

  def testNoVars(self):
    # [FUNCTION DOC]
    # With no variables, the uninitialized-variable report is empty.
    with ops.Graph().as_default(), self.cached_session() as sess:
      uninited = variables.report_uninitialized_variables()
      self.assertEqual(0, self.evaluate(uninited).size)

  def testAssertVariablesInitialized(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: name says assert but it tests the report of uninitialized variable names.
    # Two variables are reported by name (v, w) before the global initializer and none afterwards.
    with ops.Graph().as_default(), self.cached_session() as sess:
      v = variables.Variable([1, 2], name="v")
      w = variables.Variable([3, 4], name="w")
      _ = v, w
      uninited = variables.report_uninitialized_variables()
      self.assertAllEqual(np.array([b"v", b"w"]), self.evaluate(uninited))
      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(0, self.evaluate(uninited).size)

  @test_util.run_v1_only("b/120545219")
  def testVariableList(self):
    # [FUNCTION DOC]
    # Legacy graph test: report lists both names, then only v after initializing w, then nothing
    # after initializing v.
    with ops.Graph().as_default(), self.cached_session() as sess:
      v = variable_v1.VariableV1([1, 2], name="v")
      w = variable_v1.VariableV1([3, 4], name="w")
      uninited = variables.report_uninitialized_variables()
      self.assertAllEqual(np.array([b"v", b"w"]), self.evaluate(uninited))
      self.evaluate(w.initializer)
      self.assertAllEqual(np.array([b"v"]), self.evaluate(uninited))
      self.evaluate(v.initializer)
      self.assertEqual(0, self.evaluate(uninited).size)

  def testZeroSizeVarInitialized(self):
    # [FUNCTION DOC]
    # A zero-size variable is not reported as uninitialized after its initializer runs.
    with ops.Graph().as_default(), self.cached_session() as sess:
      v = variables.Variable(array_ops.zeros([0, 2]), name="v")
      uninited = variables.report_uninitialized_variables()
      self.evaluate(v.initializer)  # not strictly necessary
      self.assertEqual(0, self.evaluate(uninited).size)

  def testTrainingWithZeroSizeVar(self):
    # [FUNCTION DOC]
    # Gradient descent (rate 0.1) on an objective using a zero-size variable runs; the ones matrix
    # variable is updated from ones to 0.9 everywhere.
    with ops.Graph().as_default(), self.cached_session() as sess:
      a = variables.Variable(array_ops.zeros([0, 2]))
      b = variables.Variable(array_ops.ones([2, 2]))
      objective = math_ops.reduce_sum(b + math_ops.matmul(
          a, a, transpose_a=True))
      self.evaluate(variables.global_variables_initializer())
      do_opt = gradient_descent.GradientDescentOptimizer(0.1).minimize(
          objective)
      self.evaluate([do_opt])
      self.assertAllClose([[0.9, 0.9], [0.9, 0.9]], self.evaluate(b))


@test_util.run_v1_only("b/120545219")
class ObsoleteIsInitializedTest(test.TestCase):
  # [CLASS DOC]
  # Legacy-graph-only tests (class decorated) for the obsolete assert-variables-initialized helper.

  def testNoVars(self):
    # [FUNCTION DOC]
    # With no variables in the graph the obsolete helper returns a null value.
    with ops.Graph().as_default():
      self.assertEqual(None, variables.assert_variables_initialized())

  def testVariables(self):
    # [FUNCTION DOC]
    # The helper's result raises an op error about uninitialized value before the global initializer
    # and runs cleanly afterwards.
    with ops.Graph().as_default(), self.cached_session() as sess:
      v = variable_v1.VariableV1([1, 2])
      w = variable_v1.VariableV1([3, 4])
      _ = v, w
      inited = variables.assert_variables_initialized()
      with self.assertRaisesOpError("Attempting to use uninitialized value"):
        self.evaluate(inited)
      self.evaluate(variables.global_variables_initializer())
      self.evaluate(inited)

  def testVariableList(self):
    # [FUNCTION DOC]
    # The helper restricted to variable v: running its op fails while v is uninitialized, even if
    # w is initialized, and succeeds once v is initialized.
    with ops.Graph().as_default(), self.cached_session() as sess:
      v = variable_v1.VariableV1([1, 2])
      w = variable_v1.VariableV1([3, 4])
      inited = variables.assert_variables_initialized([v])
      with self.assertRaisesOpError("Attempting to use uninitialized value"):
        inited.op.run()
      self.evaluate(w.initializer)
      with self.assertRaisesOpError("Attempting to use uninitialized value"):
        inited.op.run()
      self.evaluate(v.initializer)
      inited.op.run()


class PartitionedVariableTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the partitioned variable wrapper: construction, validation errors and assignments.

  def testPartitionedVariable(self):
    # [FUNCTION DOC]
    # Builds a partitioned variable from two single-element variables given in reversed order and
    # checks they are re-sorted by slice offset, length is 2, iteration yields them in order, and
    # shapes of the wrapper and its concatenated tensor are two.
    with ops.Graph().as_default():
      v0 = variables.Variable([0])
      v1 = variables.Variable([1])
      v0._set_save_slice_info(
          variables.Variable.SaveSliceInfo(v0.name, [2], [0], [1]))
      v1._set_save_slice_info(
          variables.Variable.SaveSliceInfo(v0.name, [2], [1], [1]))
      partitions = [2]

      # Pass variable_list as [v1, v0] to ensure they are properly
      # re-sorted to [v0, v1] based on their slice info offsets.
      partitioned_variable = variables.PartitionedVariable(
          name="two_vars",
          shape=[2],
          dtype=v0.dtype,
          variable_list=[v1, v0],
          partitions=partitions)

      concatenated = ops.convert_to_tensor(partitioned_variable)
      num_partitions = len(partitioned_variable)
      iterated_partitions = list(partitioned_variable)
      self.assertEqual(2, num_partitions)
      self.assertEqual([v0, v1], iterated_partitions)
      self.assertEqual([2], partitioned_variable.get_shape())
      self.assertEqual([2], partitioned_variable.shape)
      self.assertEqual([2], concatenated.get_shape())
      self.assertEqual([2], concatenated.shape)

  def testPartitionedVariableFailures(self):
    # [FUNCTION DOC]
    # Checks that constructing a partitioned variable raises value errors for: empty variable list,
    # variables lacking save slice info, mismatched full shapes, and a non-positive partition count.
    with ops.Graph().as_default():
      with self.assertRaisesRegex(ValueError, "empty"):
        variables.PartitionedVariable(
            name="fail",
            shape=2,
            dtype=dtypes.int32,
            variable_list=[],
            partitions=[])

      with self.assertRaisesRegex(ValueError, "must have a save_slice_info"):
        v0 = variables.Variable([0])
        partitions = [1]
        variables.PartitionedVariable(
            name="two_vars",
            shape=[1],
            dtype=v0.dtype,
            variable_list=[v0],
            partitions=partitions)

      with self.assertRaisesRegex(ValueError, "full shapes must match"):
        v0 = variables.Variable([0])
        v1 = variables.Variable([1])
        v0._set_save_slice_info(
            variables.Variable.SaveSliceInfo(v0.name, [2], [0], [1]))
        v1._set_save_slice_info(
            variables.Variable.SaveSliceInfo(v0.name, [2], [1], [1]))
        partitions = [2]

        variables.PartitionedVariable(
            name="two_vars",
            shape=[3],
            dtype=v0.dtype,
            variable_list=[v1, v0],
            partitions=partitions)

      with self.assertRaisesRegex(ValueError, "must be positive"):
        v0 = variables.Variable([0])
        v0._set_save_slice_info(
            variables.Variable.SaveSliceInfo(v0.name, [2], [0], [1]))
        partitions = [0]

        variables.PartitionedVariable(
            name="two_vars",
            shape=[2],
            dtype=v0.dtype,
            variable_list=[v0],
            partitions=partitions)

  def testPartitionedVariableAssignments(self):
    # [FUNCTION DOC]
    # Builds two partitioned variables and checks that assign, add and subtract work from tensors,
    # lists of tensors and from other partitioned variables, returning per-partition results that
    # match both the returned values and the underlying variables.
    with ops.Graph().as_default(), self.cached_session():
      v0 = variables.Variable(initial_value=[0.0])
      v1 = variables.Variable(initial_value=[1.0])
      v2 = variables.Variable(initial_value=[20.0])
      v3 = variables.Variable(initial_value=[30.0])
      v0._set_save_slice_info(
          variables.Variable.SaveSliceInfo(v0.name, [2], [0], [1]))
      v1._set_save_slice_info(
          variables.Variable.SaveSliceInfo(v1.name, [2], [1], [1]))
      v2._set_save_slice_info(
          variables.Variable.SaveSliceInfo(v2.name, [2], [0], [1]))
      v3._set_save_slice_info(
          variables.Variable.SaveSliceInfo(v3.name, [2], [1], [1]))

      partitions = [2]

      # Pass variable_list as [v1, v0] to ensure they are properly
      # re-sorted to [v0, v1] based on their slice info offsets.
      pv_0 = variables.PartitionedVariable(
          name="two_vars",
          shape=[2],
          dtype=v0.dtype,
          variable_list=[v0, v1],
          partitions=partitions)

      pv_1 = variables.PartitionedVariable(
          name="two_vars",
          shape=[2],
          dtype=v0.dtype,
          variable_list=[v2, v3],
          partitions=partitions)

      deltas_a = constant_op.constant([1.0, 2.0])
      deltas_b = constant_op.constant([3.0, 4.0])
      ones = array_ops.ones([2])
      plus_delta = pv_0.assign_add(deltas_a)
      minus_delta = pv_0.assign_sub(deltas_b)
      assign_ones = pv_0.assign(ones)

      c_0 = constant_op.constant([2.0])
      c_1 = constant_op.constant([3.0])
      assign_list = pv_1.assign([c_0, c_1])
      assign_part_value = pv_1.assign_add(assign_ones)
      assign_part_var = pv_1.assign_sub(pv_0)
      self.evaluate(variables.global_variables_initializer())

      self.assertEqual([1.0], self.evaluate(plus_delta[0]))
      self.assertEqual([1.0], self.evaluate(v0))
      self.assertEqual([3.0], self.evaluate(plus_delta[1]))
      self.assertEqual([3.0], self.evaluate(v1))

      self.assertEqual([-2.0], self.evaluate(minus_delta[0]))
      self.assertEqual([-2.0], self.evaluate(v0))
      self.assertEqual([-1.0], self.evaluate(minus_delta[1]))
      self.assertEqual([-1.0], self.evaluate(v1))

      self.assertEqual([1.0], self.evaluate(assign_ones[0]))
      self.assertEqual([1.0], self.evaluate(v0))
      self.assertEqual([1.0], self.evaluate(assign_ones[1]))
      self.assertEqual([1.0], self.evaluate(v1))

      self.assertEqual([2.0], self.evaluate(assign_list[0]))
      self.assertEqual([2.0], self.evaluate(v2))
      self.assertEqual([3.0], self.evaluate(assign_list[1]))
      self.assertEqual([3.0], self.evaluate(v3))

      self.assertEqual([3.0], self.evaluate(assign_part_value[0]))
      self.assertEqual([3.0], self.evaluate(v2))
      self.assertEqual([4.0], self.evaluate(assign_part_value[1]))
      self.assertEqual([4.0], self.evaluate(v3))

      self.assertEqual([2.0], self.evaluate(assign_part_var[0]))
      self.assertEqual([2.0], self.evaluate(v2))
      self.assertEqual([3.0], self.evaluate(assign_part_var[1]))
      self.assertEqual([3.0], self.evaluate(v3))


class VariableContainerTest(test.TestCase):
  # [CLASS DOC]
  # Tests for container scopes applied to variables.

  def testContainer(self):
    # [FUNCTION DOC]
    # Checks the container attribute of variables created under nested container scopes: empty
    # outside scopes, innermost scope name inside, and an explicit container overriding the scope.
    with ops.Graph().as_default():
      v0 = variables.Variable([0])
      with ops.container("l1"):
        v1 = variables.Variable([1])
        with ops.container("l2"):
          v2 = variables.Variable([2])
          special_v = gen_state_ops.variable(
              shape=[1],
              dtype=dtypes.float32,
              name="VariableInL3",
              container="l3",
              shared_name="")
        v3 = variables.Variable([3])
      v4 = variables.Variable([4])
    self.assertEqual(compat.as_bytes(""), v0.op.get_attr("container"))
    self.assertEqual(compat.as_bytes("l1"), v1.op.get_attr("container"))
    self.assertEqual(compat.as_bytes("l2"), v2.op.get_attr("container"))
    self.assertEqual(compat.as_bytes("l3"), special_v.op.get_attr("container"))
    self.assertEqual(compat.as_bytes("l1"), v3.op.get_attr("container"))
    self.assertEqual(compat.as_bytes(""), v4.op.get_attr("container"))


class AggregationModesTest(test.TestCase):
  # [CLASS DOC]
  # Tests that the version 1 and version 2 aggregation enums agree.

  def testV1V2Equal(self):
    # [FUNCTION DOC]
    # Checks each aggregation mode (none, sum, mean, only first replica and the legacy only-first-tower
    # alias) is equal in both enum versions in both comparison directions and has equal hashes.
    v1 = variables.VariableAggregation
    v2 = variables.VariableAggregationV2

    self.assertEqual(v1.NONE, v2.NONE)
    self.assertEqual(v1.SUM, v2.SUM)
    self.assertEqual(v1.MEAN, v2.MEAN)
    self.assertEqual(v1.ONLY_FIRST_REPLICA, v2.ONLY_FIRST_REPLICA)
    self.assertEqual(v1.ONLY_FIRST_TOWER, v2.ONLY_FIRST_REPLICA)

    self.assertEqual(v2.NONE, v1.NONE)
    self.assertEqual(v2.SUM, v1.SUM)
    self.assertEqual(v2.MEAN, v1.MEAN)
    self.assertEqual(v2.ONLY_FIRST_REPLICA, v1.ONLY_FIRST_REPLICA)
    self.assertEqual(v2.ONLY_FIRST_REPLICA, v1.ONLY_FIRST_TOWER)

    self.assertEqual(hash(v1.NONE), hash(v2.NONE))
    self.assertEqual(hash(v1.SUM), hash(v2.SUM))
    self.assertEqual(hash(v1.MEAN), hash(v2.MEAN))
    self.assertEqual(hash(v1.ONLY_FIRST_REPLICA), hash(v2.ONLY_FIRST_REPLICA))
    self.assertEqual(hash(v1.ONLY_FIRST_TOWER), hash(v2.ONLY_FIRST_REPLICA))

if __name__ == "__main__":
  test.main()
