# [MODULE DOC]
# Unit tests for the variable store and variable scope machinery in the variable scope module
# (default variable store, get_variable, variable_scope, partitioned variables, custom getters,
# variable creator scopes, the eager variable store and the partition info helper).
# Most tests run in both graph and eager modes; in eager mode the body is wrapped in a wrapped
# graph function by the helper decorator defined here. A few older tests are graph-only and use
# the deprecated-v1 decorator. Defines three partitioner helper functions used by the
# partitioning tests. Has no import-time side effects beyond imports; running the file as a
# program starts the test runner.
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
"""Tests for variable store."""

import gc
import threading

import numpy

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import resource_variables_toggle
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test
from tensorflow.python.util import compat
from tensorflow.python.util import tf_inspect


def run_inside_wrap_function_in_eager_mode(graph_function):
  # [FUNCTION DOC]
  # HELPER: test decorator letting the same graph-style test body run in graph and eager modes.
  # Takes graph_function, a test method (taking only the test case instance) containing graph
  # code. Returns a new method: in eager mode it wraps the original with the wrap-function
  # utility (passing the test instance as the only argument) and calls the wrapped result; in
  # graph mode it simply calls the original with the instance. Raises nothing itself; any
  # exception from the test body propagates. The returned method does not preserve the name or
  # docstring of the original.
  """Decorator to execute the same graph code in eager and graph modes.

  In graph mode, we just execute the graph_function passed as argument. In eager
  mode, we wrap the function using wrap_function and then execute the wrapped
  result.

  Args:
    graph_function: python function containing graph code to be wrapped

  Returns:
    decorated function
  """
  def wrap_and_execute(self):
    # [FUNCTION DOC]
    # HELPER: inner wrapper returned by the decorator above.
    # Given the test case instance, runs the captured graph function directly when graph mode
    # is active, or via a wrapped graph function when executing eagerly. Returns nothing.
    if context.executing_eagerly():
      wrapped = wrap_function.wrap_function(graph_function, [self])
      # use the wrapped graph function
      wrapped()
    else:
      # use the original function
      graph_function(self)
  return wrap_and_execute


class VariableScopeTest(test.TestCase):
  # [CLASS DOC]
  # Main test case for the variable scope module: default variable store behavior (reuse and
  # existence checks), scope initializers, constraints, dtypes, regularizers, caching devices,
  # name scopes, reuse semantics, auxiliary name scope handling, collections accessors, and the
  # eager variable store. Holds no state of its own; tearDown checks for leaked garbage.

  def tearDown(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: overrides tearDown without calling the parent implementation, so base-class
    # cleanup is skipped.
    # Runs garbage collection and asserts that the list of uncollectable objects (reference
    # cycles involving finalizers) is empty. Takes no inputs, returns nothing, fails the test
    # with an assertion error if garbage remains. Side effect: forces a collection.
    gc.collect()
    # This will only contain uncollectable garbage, i.e. reference cycles
    # involving objects with __del__ defined.
    self.assertEqual(0, len(gc.garbage))

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testGetVar(self):
    # [FUNCTION DOC]
    # Test: requesting the same name and shape twice from the default variable store returns
    # the identical variable object (no new variable created). Runs in graph and eager modes.
    vs = variable_scope._get_default_variable_store()
    v = vs.get_variable("v", [1])
    v1 = vs.get_variable("v", [1])
    self.assertIs(v, v1)

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testResource(self):
    # [FUNCTION DOC]
    # Test: asking the default variable store for a variable with the resource option enabled
    # yields a resource variable instance. Runs in graph and eager modes.
    vs = variable_scope._get_default_variable_store()
    v1 = vs.get_variable("v", [1], use_resource=True)
    self.assertTrue(isinstance(v1, resource_variable_ops.ResourceVariable))

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testNameExists(self):
    # [FUNCTION DOC]
    # Test: with no reuse flag the store both creates and returns existing names; with reuse
    # explicitly false, requesting an existing name raises a value error; with reuse true,
    # requesting a name that does not exist raises a value error while an existing one succeeds.
    vs = variable_scope._get_default_variable_store()
    # No check by default, so we can both create and get existing names.
    v = vs.get_variable("v", [1])
    v1 = vs.get_variable("v", [1])
    self.assertIs(v, v1)

    # When reuse is False, we fail when variables are already there.
    vs.get_variable("w", [1], reuse=False)  # That's ok.
    with self.assertRaises(ValueError):
      vs.get_variable("v", [1], reuse=False)  # That fails.
    # When reuse is True, we fail when variables are new.
    vs.get_variable("v", [1], reuse=True)  # That's ok.
    with self.assertRaises(ValueError):
      vs.get_variable("u", [1], reuse=True)  # That fails.

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testNamelessStore(self):
    # [FUNCTION DOC]
    # Test: after creating two variables directly in the default store, the store's internal
    # variable mapping holds exactly those two variables, named with the output suffix zero.
    vs = variable_scope._get_default_variable_store()
    vs.get_variable("v1", [2])
    vs.get_variable("v2", [2])
    expected_names = ["%s:0" % name for name in ["v1", "v2"]]
    self.assertEqual(
        set(expected_names), set(v.name for v in vs._vars.values()))

  # Not converted to use wrap_function because of
  # TypeError: Expected tf.group() expected Tensor arguments not 'None' with
  # type '<type 'NoneType'>'
  @test_util.run_in_graph_and_eager_modes
  def testVarScopeInitializer(self):
    # [FUNCTION DOC]
    # Test: an initializer given to a variable scope (by name or by re-entering a scope object)
    # is inherited by variables created inside, so they are initialized to the constant 0.3.
    # Graph-and-eager test not wrapped because the group operation rejects null arguments.
    init = init_ops.constant_initializer(0.3)
    with variable_scope.variable_scope("tower0") as tower:
      with variable_scope.variable_scope("foo", initializer=init):
        v = variable_scope.get_variable("v", [])
        self.evaluate(variables_lib.variables_initializer([v]))
        self.assertAllClose(self.evaluate(v.value()), 0.3)
      with variable_scope.variable_scope(tower, initializer=init):
        w = variable_scope.get_variable("w", [])
        self.evaluate(variables_lib.variables_initializer([w]))
        self.assertAllClose(self.evaluate(w.value()), 0.3)

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVarScopeConstraint(self):
    # [FUNCTION DOC]
    # Test: a constraint function given to a scope (by name or via a scope object) is attached
    # to variables created inside and is available as the variable's constraint attribute.
    constraint = lambda x: 0. * x
    with variable_scope.variable_scope("tower1") as tower:
      with variable_scope.variable_scope("foo", constraint=constraint):
        v = variable_scope.get_variable("v", [])
        self.assertEqual(v.constraint, constraint)
      with variable_scope.variable_scope(tower, constraint=constraint):
        w = variable_scope.get_variable("w", [])
        self.assertEqual(w.constraint, constraint)

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVarScopeNestingError(self):
    # [FUNCTION DOC]
    # Test: manually entering a scope, then entering another scope, and then exiting the first
    # one out of order raises a runtime error; exiting afterwards in proper order succeeds.
    with variable_scope.variable_scope("aa"):
      scope = variable_scope.variable_scope("bb")
      scope.__enter__()
      with variable_scope.variable_scope("cc"):
        with self.assertRaises(RuntimeError):
          scope.__exit__(None, None, None)
      scope.__exit__(None, None, None)

  # Not converted to use wrap_function because of
  # TypeError: Fetch argument <tf.Variable 'string:0' shape=() dtype=string>
  # has invalid type <class '...ResourceVariable'>, must be a string or Tensor.
  # (Can not convert a ResourceVariable into a Tensor or Operation.)
  @test_util.run_deprecated_v1
  def testStringDefaultInitializer(self):
    # [FUNCTION DOC]
    # Test (graph mode only): a string-typed scalar variable with no initializer defaults to the
    # empty byte string after global initialization.
    with self.cached_session():
      v = variable_scope.get_variable("string", shape=[], dtype=dtypes.string)
      variables_lib.global_variables_initializer().run()
      self.assertAllEqual(compat.as_bytes(self.evaluate(v)), b"")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVarScopeDType(self):
    # [FUNCTION DOC]
    # Test: a dtype given to a scope (by name or scope object) becomes the default dtype of
    # variables created inside; checks the base dtype is half precision.
    with variable_scope.variable_scope("tower2") as tower:
      with variable_scope.variable_scope("foo", dtype=dtypes.float16):
        v = variable_scope.get_variable("v", [])
        self.assertEqual(v.dtype.base_dtype, dtypes.float16)
      with variable_scope.variable_scope(tower, dtype=dtypes.float16):
        w = variable_scope.get_variable("w", [])
        self.assertEqual(w.dtype.base_dtype, dtypes.float16)

  def testGetVariableInGraphNestedUnderEagerContext(self):
    # [FUNCTION DOC]
    # Test: in eager mode, a variable obtained via get_variable and then passed into a graph
    # function is still a resource variable (checked inside the function).
    with context.eager_mode():

      @def_function.function
      def f(v):
        # [FUNCTION DOC]
        # HELPER: body of the traced function for the enclosing test.
        # Takes a variable v and asserts its exact type is the resource variable class. Returns
        # nothing.
        self.assertEqual(type(v), resource_variable_ops.ResourceVariable)

      var = variable_scope.get_variable("should_be_resource", [])
      f(var)

  def testEagerVariableStore(self):
    # [FUNCTION DOC]
    # Test (eager mode): variables created inside an eager variable store are reported by its
    # variables, trainable and non-trainable accessors correctly; copying the store gives
    # equal initial values but independent variable instances (assigning to one store's
    # variables does not change the other's).
    with context.eager_mode():
      store = variable_scope.EagerVariableStore()
      with store.as_default():
        v = variable_scope.get_variable("v", shape=(), trainable=True)
        w = variable_scope.get_variable("w", shape=(), trainable=False)

      self.assertTrue(v in store.variables())
      self.assertTrue(w in store.variables())
      self.assertTrue(v in store.trainable_variables())
      self.assertFalse(w in store.trainable_variables())
      self.assertFalse(v in store.non_trainable_variables())
      self.assertTrue(w in store.non_trainable_variables())

      # Test copying.
      new_store = store.copy()
      with new_store.as_default():
        new_v = variable_scope.get_variable("v")
        new_w = variable_scope.get_variable("w")
      self.assertEqual(new_v.numpy(), v.numpy())
      self.assertEqual(new_w.numpy(), w.numpy())
      self.assertTrue(new_v in new_store.variables())
      self.assertTrue(new_w in new_store.variables())
      self.assertTrue(new_v in new_store.trainable_variables())
      self.assertFalse(new_w in new_store.trainable_variables())
      self.assertFalse(new_v in new_store.non_trainable_variables())
      self.assertTrue(new_w in new_store.non_trainable_variables())

      # Check that variables are separate instances.
      for v in store.variables():
        v.assign(-1)
      for v in new_store.variables():
        v.assign(1)
      for v in store.variables():
        self.assertEqual(v.numpy(), -1)
      for v in new_store.variables():
        self.assertEqual(v.numpy(), 1)

  # Not converted to use wrap_function because of
  # obtaining different results in the eager case compared to the graph one
  @test_util.run_in_graph_and_eager_modes
  def testEagerVariablesStoreAddsToCollections(self):
    # [FUNCTION DOC]
    # Test: variables created within an eager variable store are added to the global, trainable
    # and custom concatenated collections as requested (non-trainable ones excluded from the
    # trainable collection). Not wrapped because eager and graph results differ.
    store = variable_scope.EagerVariableStore()
    with store.as_default():
      trainable = variable_scope.get_variable("v1", [], trainable=True)
      not_trainable = variable_scope.get_variable("v2", [], trainable=False)
      concat = variable_scope.get_variable(
          "v3", [], collections=[ops.GraphKeys.CONCATENATED_VARIABLES])
      self.assertEqual(
          ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES),
          [trainable, not_trainable])
      self.assertEqual(
          ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES),
          [trainable, concat])
      self.assertEqual(
          ops.get_collection(ops.GraphKeys.CONCATENATED_VARIABLES), [concat])

  def testEagerVariablesOutsideStoreNotAddedToCollections(self):
    # [FUNCTION DOC]
    # Test (eager mode): variables created without an eager variable store do not appear in the
    # global or trainable collections.
    with context.eager_mode():
      variable_scope.get_variable("v1", [], trainable=True)
      variable_scope.get_variable("v2", [], trainable=False)
      self.assertFalse(ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES))
      self.assertFalse(ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES))

  # Not converted to use wrap_function because of
  # TypeError: Expected tf.group() expected Tensor arguments not 'None' with
  # type '<type 'NoneType'>'.
  @test_util.run_in_graph_and_eager_modes
  def testInitFromNonTensorValue(self):
    # [FUNCTION DOC]
    # Test: variables can be initialized from a plain number or an array with explicit dtype and
    # take the given values; an unsupported initializer (an empty mapping) raises a value error
    # when eager and a type error in graph mode (noted in the test as a quirk).
    v = variable_scope.get_variable("v4", initializer=4, dtype=dtypes.int32)
    self.evaluate(variables_lib.variables_initializer([v]))
    self.assertAllClose(self.evaluate(v.value()), 4)

    w = variable_scope.get_variable(
        "w4", initializer=numpy.array([1, 2, 3]), dtype=dtypes.int64)
    self.evaluate(variables_lib.variables_initializer([w]))
    self.assertAllClose(self.evaluate(w.value()), [1, 2, 3])

    # A quirk to be revisited?
    error = ValueError if context.executing_eagerly() else TypeError
    with self.assertRaises(error):
      variable_scope.get_variable("x4", initializer={})

  # Not converted to use wrap_function because of
  # InvalidArgumentError=: You must feed a value for placeholder tensor
  # 'ReadVariableOp/resource' with dtype resource
  @test_util.run_in_graph_and_eager_modes
  def testInitFromNonInitializer(self):
    # [FUNCTION DOC]
    # Test: for several integer and boolean dtypes, a variable with no initializer holds the
    # same values as one initialized explicitly with zeros (the default is zeros).
    # Test various dtypes with zeros initializer as following:
    types = [
        dtypes.int8, dtypes.uint8, dtypes.int16, dtypes.uint16, dtypes.int32,
        dtypes.int64, dtypes.bool
    ]

    # Use different variable_name to distinguish various dtypes
    for (i, dtype) in enumerate(types):
      x = variable_scope.get_variable(
          name="xx%d" % i, shape=(3, 4), dtype=dtype)
      y = variable_scope.get_variable(
          name="yy%d" % i,
          shape=(3, 4),
          dtype=dtype,
          initializer=init_ops.zeros_initializer(dtype=dtype))

      self.evaluate(variables_lib.global_variables_initializer())
      self.assertAllEqual(self.evaluate(x.value()), self.evaluate(y.value()))

  # Not converted to use wrap_function because of
  # InvalidArgumentError: /job:moo/replica:0/task:0/device:CPU:0 unknown device.
  @test_util.run_deprecated_v1
  def testVarScopeCachingDevice(self):
    # [FUNCTION DOC]
    # Test (graph mode): the caching device set on a scope is inherited by child scopes, can be
    # cleared with an empty string, can be a callable returning the op device (which is then not
    # the scope device), can be changed via the scope setter, and does not leak outside.
    with self.cached_session():
      caching_device = "/job:moo"
      with variable_scope.variable_scope("tower"):
        with variable_scope.variable_scope(
            "caching", caching_device=caching_device):
          v = variable_scope.get_variable("v", [])
          self.assertTrue(v.value().device.startswith(caching_device))

          with variable_scope.variable_scope("child"):
            v2 = variable_scope.get_variable("v", [])
            self.assertTrue(v2.value().device.startswith(caching_device))

          with variable_scope.variable_scope("not_cached", caching_device=""):
            v2_not_cached = variable_scope.get_variable("v", [])
            self.assertFalse(
                v2_not_cached.value().device.startswith(caching_device))

          with variable_scope.variable_scope(
              "not_cached_identity_device",
              caching_device=lambda op: op.device):
            v2_identity_device = variable_scope.get_variable("v", [])
            self.assertFalse(
                v2_identity_device.value().device.startswith(caching_device))

          with variable_scope.variable_scope("we_will_do_it_live") as vs_live:
            vs_live.set_caching_device("/job:live")
            v_live = variable_scope.get_variable("v", [])
            self.assertTrue(v_live.value().device.startswith("/job:live"))

        v_tower = variable_scope.get_variable("v", [])
        self.assertFalse(v_tower.value().device.startswith(caching_device))

  # Not converted to use wrap_function because of
  # AttributeError: Tensor.name is meaningless when eager execution is enabled.
  @test_util.run_in_graph_and_eager_modes
  def testVarScopeRegularizer(self):
    # [FUNCTION DOC]
    # Test: regularizers set on scopes apply to variables created in them and add entries to the
    # regularization loss collection with expected values; setting a new regularizer on a scope
    # object, passing the no-regularizer marker per variable or per scope, and resetting it
    # disable regularization; reusing a variable adds no new loss (graph mode only).
    init = init_ops.constant_initializer(0.3)

    def regularizer1(v):
      # [FUNCTION DOC]
      # HELPER: regularizer for the enclosing test. Returns the mean of variable v plus 0.1.
      return math_ops.reduce_mean(v) + 0.1

    def regularizer2(v):
      # [FUNCTION DOC]
      # HELPER: regularizer for the enclosing test. Returns the mean of variable v plus 0.2.
      return math_ops.reduce_mean(v) + 0.2

    with variable_scope.variable_scope(
        "tower3", regularizer=regularizer1) as tower:
      with variable_scope.variable_scope("foo", initializer=init):
        v = variable_scope.get_variable("v", [])
        self.evaluate(variables_lib.variables_initializer([v]))
        losses = ops.get_collection(ops.GraphKeys.REGULARIZATION_LOSSES)
        self.assertEqual(1, len(losses))
        self.assertAllClose(self.evaluate(losses[0]), 0.4)
      with variable_scope.variable_scope(tower, initializer=init) as vs:
        u = variable_scope.get_variable("u", [])
        vs.set_regularizer(regularizer2)
        w = variable_scope.get_variable("w", [])
        # Next 3 variable not regularized to test disabling regularization.
        x = variable_scope.get_variable(
            "x", [], regularizer=variable_scope.no_regularizer)
        with variable_scope.variable_scope(
            "baz", regularizer=variable_scope.no_regularizer):
          y = variable_scope.get_variable("y", [])
        vs.set_regularizer(variable_scope.no_regularizer)
        z = variable_scope.get_variable("z", [])
        # Check results.
        losses = ops.get_collection(ops.GraphKeys.REGULARIZATION_LOSSES)
        self.assertEqual(3, len(losses))
        self.evaluate(variables_lib.variables_initializer([u, w, x, y, z]))
        self.assertAllClose(self.evaluate(losses[0]), 0.4)
        self.assertAllClose(self.evaluate(losses[1]), 0.4)
        self.assertAllClose(self.evaluate(losses[2]), 0.5)
      with variable_scope.variable_scope("foo", reuse=True):
        # reuse=True is for now only supported when eager execution is disabled.
        if not context.executing_eagerly():
          v = variable_scope.get_variable("v",
                                          [])  # "v" is already there, reused
          losses = ops.get_collection(ops.GraphKeys.REGULARIZATION_LOSSES)
          self.assertEqual(3, len(losses))  # No new loss added.

  # Not converted to use wrap_function because of
  # ValueError: Tensor-typed variable initializers must either be wrapped in an
  # init_scope or callable...
  @test_util.run_in_graph_and_eager_modes
  def testInitializeFromValue(self):
    # [FUNCTION DOC]
    # Test: a constant tensor can be the initializer (value 0.1); passing an explicit shape with
    # it raises a value error mentioning shape; it can be inherited through a scope; an integer
    # tensor gives an integer variable; a mismatching explicit dtype raises a value error saying
    # the types do not match.
    init = constant_op.constant(0.1)
    w = variable_scope.get_variable("v", initializer=init)
    self.evaluate(variables_lib.variables_initializer([w]))
    self.assertAllClose(self.evaluate(w.value()), 0.1)

    with self.assertRaisesRegex(ValueError, "shape"):
      # We disallow explicit shape specification when initializer is constant.
      variable_scope.get_variable("u", [1], initializer=init)

    with variable_scope.variable_scope("foo", initializer=init):
      # Constant initializer can be passed through scopes if needed.
      v = variable_scope.get_variable("v")
      self.evaluate(variables_lib.variables_initializer([v]))
      self.assertAllClose(self.evaluate(v.value()), 0.1)

    # Check that non-float32 initializer creates a non-float32 variable.
    init = constant_op.constant(1, dtype=dtypes.int32)
    t = variable_scope.get_variable("t", initializer=init)
    self.assertEqual(t.dtype.base_dtype, dtypes.int32)

    # Raise error if `initializer` dtype and `dtype` are not identical.
    with self.assertRaisesRegex(ValueError, "don't match"):
      variable_scope.get_variable("s", initializer=init, dtype=dtypes.float64)

  # Not converted to use wrap_function because of
  # TypeError: Fetch argument <tf.Variable 'v0:0' shape=(1,) dtype=float32> has
  # invalid type <class '...ops.resource_variable_ops.ResourceVariable'>, must
  # be a string or Tensor. (Can not convert a ResourceVariable into a Tensor or
  # Operation.)
  @test_util.run_deprecated_v1
  def testControlDeps(self):
    # [FUNCTION DOC]
    # Test (graph mode): a variable created under a control dependency on another variable can
    # be initialized and evaluated without initializing the dependency; reading the dependency
    # or an add of both fails with an uninitialized error until the dependency is initialized.
    with self.cached_session() as sess:
      v0 = variable_scope.get_variable(
          "v0", [1], initializer=init_ops.constant_initializer(0))
      with ops.control_dependencies([v0.value()]):
        v1 = variable_scope.get_variable(
            "v1", [1], initializer=init_ops.constant_initializer(1))
        add = v1 + v0
      # v0 should be uninitialized.
      with self.assertRaisesRegex(errors.OpError, "uninitialized"):
        self.evaluate(v0)
      # We should be able to initialize and run v1 without initializing
      # v0, even if the variable was created with a control dep on v0.
      self.evaluate(v1.initializer)
      self.assertEqual(1, self.evaluate(v1))
      # v0 should still be uninitialized.
      with self.assertRaisesRegex(errors.OpError, "uninitialized"):
        self.evaluate(v0)
      with self.assertRaisesRegex(errors.OpError, "uninitialized"):
        self.evaluate(add)
      # If we initialize v0 we should be able to run 'add'.
      self.evaluate(v0.initializer)
      self.evaluate(add)

  # Not converted to use wrap_function because of
  # AssertionError: True is not false (last assertFalse)
  @test_util.run_deprecated_v1
  def testEnableResourceVariables(self):
    # [FUNCTION DOC]
    # Test (graph mode): enabling resource variables makes the V1 variable constructor return
    # resource variables; disabling makes it return something else. The global default flag is
    # saved first and restored in a finally block.
    old = resource_variables_toggle._DEFAULT_USE_RESOURCE
    try:
      resource_variables_toggle.enable_resource_variables()
      self.assertIsInstance(
          variable_v1.VariableV1(1.0),
          resource_variable_ops.ResourceVariable)
      resource_variables_toggle.disable_resource_variables()
      self.assertNotIsInstance(
          variable_v1.VariableV1(1.0),
          resource_variable_ops.ResourceVariable)
    finally:
      resource_variables_toggle._DEFAULT_USE_RESOURCE = old

  # Not converted to use wrap_function because of
  # TypeError: Fetch argument None has invalid type <type 'NoneType'>
  @test_util.run_deprecated_v1
  def testControlFlow(self):
    # [FUNCTION DOC]
    # Test (graph mode): variables created inside both branches of a conditional via
    # get_variable can be initialized and read independently of the outside variable, while the
    # conditional result fails as uninitialized until that outside variable is initialized.
    with self.cached_session() as sess:
      v0 = variable_scope.get_variable(
          "v0", [], initializer=init_ops.constant_initializer(0))
      var_dict = {}

      # Call get_variable in each of the cond clauses.
      def var_in_then_clause():
        # [FUNCTION DOC]
        # HELPER: true branch of the conditional in the enclosing test. Creates variable v1 with
        # initial value 1, records it in the shared mapping and returns v1 plus the outer v0.
        v1 = variable_scope.get_variable(
            "v1", [1], initializer=init_ops.constant_initializer(1))
        var_dict["v1"] = v1
        return v1 + v0

      def var_in_else_clause():
        # [FUNCTION DOC]
        # HELPER: false branch of the conditional in the enclosing test. Creates variable v2 with
        # initial value 2, records it in the shared mapping and returns v2 plus the outer v0.
        v2 = variable_scope.get_variable(
            "v2", [1], initializer=init_ops.constant_initializer(2))
        var_dict["v2"] = v2
        return v2 + v0

      add = cond.cond(
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
      with self.assertRaisesRegex(errors.OpError, "uninitialized"):
        self.evaluate(v0)
      # We should not be able to run 'add' yet.
      with self.assertRaisesRegex(errors.OpError, "uninitialized"):
        self.evaluate(add)
      # If we initialize v0 we should be able to run 'add'.
      self.evaluate(v0.initializer)
      self.evaluate(add)

  # Not converted to use wrap_function because of
  # TypeError: Expected tf.group() expected Tensor arguments not 'None' with
  # type '<type 'NoneType'>'.
  @test_util.run_in_graph_and_eager_modes
  def testGetVariableScope(self):
    # [FUNCTION DOC]
    # Test: the current-scope accessor returns a modifiable scope; the initializer set on it
    # applies to new variables inside the scope and is gone after exit; in graph mode marking
    # the scope for reuse makes creating a new name raise a value error.
    # Test the get_variable_scope() function and setting properties of result.
    init = init_ops.constant_initializer(0.3)
    with variable_scope.variable_scope("bar"):
      new_init1 = variable_scope.get_variable_scope().initializer
      self.assertEqual(new_init1, None)
      # Check that we can set initializer like this.
      variable_scope.get_variable_scope().set_initializer(init)
      v = variable_scope.get_variable("v", [])
      self.evaluate(variables_lib.variables_initializer([v]))
      self.assertAllClose(self.evaluate(v.value()), 0.3)
      if not context.executing_eagerly():
        # Check that we can set reuse.
        variable_scope.get_variable_scope().reuse_variables()
        with self.assertRaises(ValueError):  # Fail, w does not exist yet.
          variable_scope.get_variable("w", [1])
    # Check that the set initializer goes away.
    new_init = variable_scope.get_variable_scope().initializer
    self.assertEqual(new_init, None)

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVarScope(self):
    # [FUNCTION DOC]
    # Test: scope names and the name scopes they open follow nesting, and re-entering a scope
    # object inside another scope prefixes the name scope but keeps the scope's own name.
    with variable_scope.variable_scope("tower4") as tower:
      self.assertEqual(tower.name, "tower4")
      with ops.name_scope("scope") as sc:
        self.assertEqual(sc, "tower4/scope/")

    with variable_scope.variable_scope("tower5"):
      with variable_scope.variable_scope("bar") as bar:
        self.assertEqual(bar.name, "tower5/bar")
        with ops.name_scope("scope") as sc:
          self.assertEqual(sc, "tower5/bar/scope/")

    with variable_scope.variable_scope("tower6"):
      with variable_scope.variable_scope(tower, reuse=True) as tower_shared:
        self.assertEqual(tower_shared.name, "tower4")
        with ops.name_scope("scope") as sc:
          self.assertEqual(sc, "tower6/tower4/scope/")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVarScopeNameScope(self):
    # [FUNCTION DOC]
    # Test: name scopes opened inside variable scopes get expected prefixes; in graph mode
    # re-entering the same scope (by object or string) yields unique suffixed name scopes; the
    # root scope adds no prefix.
    with ops.name_scope("testVarScopeNameScope1"):
      with variable_scope.variable_scope("tower") as tower:
        with ops.name_scope("scope2") as sc2:
          self.assertEqual(sc2, "testVarScopeNameScope1/tower/scope2/")
      if not context.executing_eagerly():
        with variable_scope.variable_scope(
            tower):  # Re-entering acts like another "tower".
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "testVarScopeNameScope1/tower_1/scope2/")
        with variable_scope.variable_scope(
            "tower"):  # Re-entering by string acts the same.
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "testVarScopeNameScope1/tower_2/scope2/")

    with ops.name_scope("testVarScopeNameScope2"):
      with variable_scope.variable_scope("tower"):
        with ops.name_scope("scope2") as sc2:
          self.assertEqual(sc2, "testVarScopeNameScope2/tower/scope2/")
      if not context.executing_eagerly():
        with variable_scope.variable_scope(tower):
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "testVarScopeNameScope2/tower_1/scope2/")

    root_var_scope = variable_scope.get_variable_scope()
    with ops.name_scope("testVarScopeNameScope3"):
      with variable_scope.variable_scope(root_var_scope):
        with ops.name_scope("scope2") as sc2:
          self.assertEqual(sc2, "testVarScopeNameScope3/scope2/")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVarScopeOriginalNameScope(self):
    # [FUNCTION DOC]
    # Test: the scope's original name scope is recorded at first entry and preserved on
    # re-entry; entering a name scope with that string recreates the original prefix, and a
    # second such entry gets a unique suffix.
    with self.cached_session():
      with ops.name_scope("scope1"):
        with variable_scope.variable_scope("tower") as tower:
          self.assertEqual(tower.original_name_scope, "scope1/tower/")
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "scope1/tower/scope2/")
      with ops.name_scope("scope2"):
        with variable_scope.variable_scope(tower) as tower1:
          # Re-entering preserves original name scope.
          self.assertEqual(tower1.original_name_scope, "scope1/tower/")
          with ops.name_scope("foo") as sc2:
            self.assertEqual(sc2, "scope2/tower/foo/")
        # Test re-entering original name scope.
        with ops.name_scope(tower.original_name_scope):
          with ops.name_scope("bar") as sc3:
            self.assertEqual(sc3, "scope1/tower/bar/")
      with ops.name_scope("scope2"):
        with variable_scope.variable_scope(tower):
          with ops.name_scope(tower.original_name_scope):
            with ops.name_scope("bar") as sc3:
              self.assertEqual(sc3, "scope1/tower/bar_1/")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVarScopeObjectReuse(self):
    # [FUNCTION DOC]
    # Test: re-entering a scope object inherits its reuse flag; once reuse is true it cannot be
    # turned off by re-entering with reuse false, but a scope created with reuse false can be
    # re-entered with reuse true.
    with self.cached_session():
      vs = None
      with variable_scope.variable_scope("jump", reuse=True) as scope:
        vs = scope

      with variable_scope.variable_scope(vs) as jump:
        self.assertTrue(jump.reuse)

      with variable_scope.variable_scope(vs, reuse=True) as jump_reuse:
        self.assertTrue(jump_reuse.reuse)

      with variable_scope.variable_scope(vs, reuse=False) as jump_no_reuse:
        self.assertTrue(jump_no_reuse.reuse)  # Inherited, cannot be undone.

      with variable_scope.variable_scope("jump", reuse=False) as scope:
        vs = scope

      with variable_scope.variable_scope(vs) as jump:
        self.assertFalse(jump.reuse)

      with variable_scope.variable_scope(vs, reuse=True) as jump_reuse:
        self.assertTrue(jump_reuse.reuse)

      with variable_scope.variable_scope(vs, reuse=False) as jump_no_reuse:
        self.assertFalse(jump_no_reuse.reuse)

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVarScopeGetOrCreateReuse(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the final assertion compares the input value with the evaluated constant
    # tensor, not the variable, so it is always true and does not check variable reuse.
    # Test: using the automatic-reuse mode, repeatedly creating or fetching the same named
    # variable and assigning to it works across calls with several values.
    with self.cached_session():

      def test_value(value):
        # [FUNCTION DOC]
        # HELPER: inner routine of the enclosing test. Takes a number, builds a constant, assigns
        # it to the auto-reuse variable, fetches the variable again and asserts the number
        # equals the evaluated constant. Returns nothing.
        x = constant_op.constant(value)
        with variable_scope.variable_scope(
            "testVarScopeGetOrCreateReuse_bar",
            reuse=variable_scope.AUTO_REUSE):
          _ = state_ops.assign(variable_scope.get_variable("var", []), x)
        with variable_scope.variable_scope(
            "testVarScopeGetOrCreateReuse_bar",
            reuse=variable_scope.AUTO_REUSE):
          _ = variable_scope.get_variable("var", [])
        self.assertEqual(value, self.evaluate(x))

      test_value(42.)  # Variable is created.
      test_value(13.)  # Variable is reused hereafter.
      test_value(17.)

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVarOpScope(self):
    # [FUNCTION DOC]
    # Test: scopes entered with a name or a default name: the explicit name wins; re-creating a
    # variable with the same name in a re-entered scope raises a value error; default names
    # are made unique with numeric suffixes and name scopes follow.
    with self.cached_session():
      with ops.name_scope("testVarOpScope1"):
        with variable_scope.variable_scope("tower", "default", []):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "tower/w:0")
          with ops.name_scope("testVarOpScope2") as sc2:
            self.assertEqual(sc2, "testVarOpScope1/tower/testVarOpScope2/")
        with variable_scope.variable_scope("tower", "default", []):
          with self.assertRaises(ValueError):
            variable_scope.get_variable("w", [])
          with ops.name_scope("testVarOpScope2") as sc2:
            self.assertEqual(sc2, "testVarOpScope1/tower_1/testVarOpScope2/")

      with ops.name_scope("testVarOpScope2"):
        with variable_scope.variable_scope(None, "default", []):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "default/w:0")
          with ops.name_scope("testVarOpScope2") as sc2:
            self.assertEqual(sc2, "testVarOpScope2/default/testVarOpScope2/")
        with variable_scope.variable_scope(None, "default", []):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "default_1/w:0")
          with ops.name_scope("testVarOpScope2") as sc2:
            self.assertEqual(sc2, "testVarOpScope2/default_1/testVarOpScope2/")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVarOpScopeUniqueNamesInterleavedSubstringScopes(self):
    # [FUNCTION DOC]
    # Test: unique default-name numbering treats names that are prefixes of one another
    # independently (default scope one is numbered separately from default scope).
    with self.cached_session():
      with variable_scope.variable_scope(None, "defaultScope1"):
        with variable_scope.variable_scope(None, "layer"):
          self.assertEqual(
              variable_scope.get_variable("w", []).name,
              "defaultScope1/layer/w:0")
      with variable_scope.variable_scope(None, "defaultScope1"):
        with variable_scope.variable_scope(None, "layer"):
          self.assertEqual(
              variable_scope.get_variable("w", []).name,
              "defaultScope1_1/layer/w:0")
      with variable_scope.variable_scope(None, "defaultScope"):
        with variable_scope.variable_scope(None, "layer"):
          self.assertEqual(
              variable_scope.get_variable("w", []).name,
              "defaultScope/layer/w:0")
      with variable_scope.variable_scope(None, "defaultScope1"):
        with variable_scope.variable_scope(None, "layer"):
          self.assertEqual(
              variable_scope.get_variable("w", []).name,
              "defaultScope1_2/layer/w:0")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVarOpScopeUniqueNamesWithJump(self):
    # [FUNCTION DOC]
    # Test: unique numbering of default scope names continues even when another scope object is
    # re-entered between uses.
    with self.cached_session():
      with variable_scope.variable_scope("default") as default:
        with variable_scope.variable_scope(None, "layer"):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "default/layer/w:0")
        with variable_scope.variable_scope(None, "layer"):
          self.assertEqual(
              variable_scope.get_variable("w", []).name,
              "default/layer_1/w:0")
        with variable_scope.variable_scope(default):
          pass
        # No matter the jump in the middle, unique numbering continues.
        with variable_scope.variable_scope(None, "layer"):
          self.assertEqual(
              variable_scope.get_variable("w", []).name,
              "default/layer_2/w:0")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVarOpScopeReuse(self):
    # [FUNCTION DOC]
    # Test: variables under an outer scope with a default name; re-entering the outer scope with
    # reuse true reuses the same variables while name scopes get new suffixes.
    with self.cached_session():
      with variable_scope.variable_scope("outer") as outer:
        with variable_scope.variable_scope("tower", "default", []):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/tower/w:0")
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer/tower/scope2/")
        with variable_scope.variable_scope(None, "default", []):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/default/w:0")
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer/default/scope2/")

      with variable_scope.variable_scope(outer, reuse=True) as outer:
        with variable_scope.variable_scope("tower", "default", []):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/tower/w:0")
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer_1/tower/scope2/")
        with variable_scope.variable_scope(None, "default", []):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/default/w:0")
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer_1/default/scope2/")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVarScopeGetVar(self):
    # [FUNCTION DOC]
    # Test: variable lookup by scope: reuse through a scope object returns the same variable,
    # new scopes get separate variables, duplicate creation raises a value error, reuse with a
    # different scope fails, and reuse with a wrong shape or dtype raises errors mentioning them.
    with self.cached_session():
      with variable_scope.variable_scope("root"):
        with variable_scope.variable_scope("towerA") as tower_a:
          va = variable_scope.get_variable("v", [1])
          self.assertEqual(va.name, "root/towerA/v:0")

        with variable_scope.variable_scope(tower_a, reuse=True):
          va2 = variable_scope.get_variable("v", [1])
          self.assertIs(va2, va)

        with variable_scope.variable_scope("towerB"):
          vb = variable_scope.get_variable("v", [1])
          self.assertEqual(vb.name, "root/towerB/v:0")

        with self.assertRaises(ValueError):
          with variable_scope.variable_scope("towerA"):
            va2 = variable_scope.get_variable("v", [1])

        with variable_scope.variable_scope("towerA", reuse=True):
          va2 = variable_scope.get_variable("v", [1])
          self.assertIs(va2, va)

        with variable_scope.variable_scope("foo"):
          with variable_scope.variable_scope("bar"):
            v = variable_scope.get_variable("v", [1])
            self.assertEqual(v.name, "root/foo/bar/v:0")
            with variable_scope.variable_scope(tower_a, reuse=True):
              va3 = variable_scope.get_variable("v", [1])
              self.assertIs(va, va3)

        with self.assertRaises(ValueError):
          with variable_scope.variable_scope(tower_a, reuse=True):
            with variable_scope.variable_scope("baz"):
              variable_scope.get_variable("v", [1])

        with self.assertRaises(ValueError) as exc:
          with variable_scope.variable_scope(tower_a, reuse=True):
            variable_scope.get_variable("v", [2])  # Different shape.
        self.assertEqual("shape" in str(exc.exception), True)

        with self.assertRaises(ValueError) as exc:
          with variable_scope.variable_scope(tower_a, reuse=True):
            variable_scope.get_variable("v", [1], dtype=dtypes.int32)
        self.assertEqual("dtype" in str(exc.exception), True)

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVarScopeOuterScope(self):
    # [FUNCTION DOC]
    # Test: re-entering an outer scope object after exit keeps variable names but opens new
    # suffixed name scopes, with and without reuse, and for child scopes.
    with self.cached_session():
      with variable_scope.variable_scope("outer") as outer:
        pass
      with variable_scope.variable_scope(outer):
        self.assertEqual(
            variable_scope.get_variable("w", []).name, "outer/w:0")
        with ops.name_scope("scope2") as sc2:
          self.assertEqual(sc2, "outer_1/scope2/")
        with variable_scope.variable_scope("default"):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/default/w:0")
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer_1/default/scope2/")

      with variable_scope.variable_scope(outer, reuse=True):
        self.assertEqual(
            variable_scope.get_variable("w", []).name, "outer/w:0")
        with ops.name_scope("scope2") as sc2:
          self.assertEqual(sc2, "outer_2/scope2/")
        with variable_scope.variable_scope("default", reuse=True):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/default/w:0")
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer_2/default/scope2/")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVarScopeNestedOuterScope(self):
    # [FUNCTION DOC]
    # Test: entering the outer scope object while inside itself nests names correctly in name
    # scopes (repeated outer prefix), with and without reuse.
    with self.cached_session():
      with variable_scope.variable_scope("outer") as outer:
        with variable_scope.variable_scope(outer):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/w:0")
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer/outer/scope2/")
        with variable_scope.variable_scope("default"):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/default/w:0")
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer/default/scope2/")

        with variable_scope.variable_scope(outer, reuse=True):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/w:0")
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer/outer_1/scope2/")
        with variable_scope.variable_scope("default", reuse=True):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/default/w:0")
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer/default_1/scope2/")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVarOpScopeReuseParam(self):
    # [FUNCTION DOC]
    # Test: the reuse argument and the reuse-variables call on a scope make later variables with
    # same names resolve to existing ones, including with a default name.
    with self.cached_session():
      with variable_scope.variable_scope("outer") as outer:
        with variable_scope.variable_scope("tower", "default", []):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/tower/w:0")
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer/tower/scope2/")
        with variable_scope.variable_scope(None, "default", []):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/default/w:0")
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer/default/scope2/")

      with variable_scope.variable_scope(outer) as outer:
        with variable_scope.variable_scope("tower", "default", reuse=True):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/tower/w:0")
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer_1/tower/scope2/")
        outer.reuse_variables()
        with variable_scope.variable_scope(None, "default", []):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/default/w:0")
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer_1/default/scope2/")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVarOpScopeReuseError(self):
    # [FUNCTION DOC]
    # Test: entering an unnamed scope with a default name and reuse true then requesting a new
    # variable raises a value error.
    with self.cached_session():
      with self.assertRaises(ValueError):
        with variable_scope.variable_scope(None, "default", reuse=True):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/tower/w:0")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVarOpScopeOuterScope(self):
    # [FUNCTION DOC]
    # Test: passing both a scope object and default name uses the scope object (default name is
    # ignored), checking names, name scopes and reuse handling.
    with self.cached_session():
      with variable_scope.variable_scope("outer") as outer:
        pass
      with variable_scope.variable_scope(outer, "default", []):
        self.assertEqual(
            variable_scope.get_variable("w", []).name, "outer/w:0")
        with ops.name_scope("scope2") as sc2:
          self.assertEqual(sc2, "outer_1/scope2/")
        with variable_scope.variable_scope(None, "default", []):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/default/w:0")
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer_1/default/scope2/")

      with variable_scope.variable_scope(outer, "default", reuse=True):
        self.assertEqual(
            variable_scope.get_variable("w", []).name, "outer/w:0")
        with ops.name_scope("scope2") as sc2:
          self.assertEqual(sc2, "outer_2/scope2/")
        outer.reuse_variables()
        with variable_scope.variable_scope(None, "default", []):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/default/w:0")
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer_2/default/scope2/")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVarOpScopeNestedOuterScope(self):
    # [FUNCTION DOC]
    # Test: a scope object plus default name entered inside the same scope nests names and name
    # scopes as expected, with reuse in the second pass.
    with self.cached_session():
      with variable_scope.variable_scope("outer") as outer:
        with variable_scope.variable_scope(outer, "default", []):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/w:0")
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer/outer/scope2/")
        with variable_scope.variable_scope(None, "default", []):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/default/w:0")
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer/default/scope2/")

      with variable_scope.variable_scope(outer, "default", reuse=True):
        self.assertEqual(
            variable_scope.get_variable("w", []).name, "outer/w:0")
        with ops.name_scope("scope2") as sc2:
          self.assertEqual(sc2, "outer_1/scope2/")
        with variable_scope.variable_scope(None, "default", []):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/default/w:0")
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer_1/default/scope2/")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testBasicWhenAuxiliaryNameScopeIsFalse(self):
    # [FUNCTION DOC]
    # Test: with the auxiliary name scope disabled no new name scope is opened: operations get
    # no scope prefix, variables are still prefixed by scope name, and the original name scope
    # is empty or the parent's, for fresh and re-entered scopes, top-level and nested.
    with self.cached_session():
      with variable_scope.variable_scope(
          "scope", auxiliary_name_scope=False) as scope:
        self.assertEqual(scope.original_name_scope, "")
        self.assertEqual(
            variable_scope.get_variable("w", []).name, "scope/w:0")
        self.assertEqual(constant_op.constant([], name="c").name, "c:0")
      with variable_scope.variable_scope(scope, auxiliary_name_scope=False):
        self.assertEqual(scope.original_name_scope, "")
        self.assertEqual(
            variable_scope.get_variable("w1", []).name, "scope/w1:0")
        self.assertEqual(constant_op.constant([], name="c1").name, "c1:0")
      # Recheck: new name scope is NOT created before
      with ops.name_scope("scope"):
        self.assertEqual(constant_op.constant([], name="c").name, "scope/c:0")

      with variable_scope.variable_scope("outer"):
        with variable_scope.variable_scope(
            "inner", auxiliary_name_scope=False) as inner:
          self.assertEqual(inner.original_name_scope, "outer/")
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/inner/w:0")
          self.assertEqual(
              constant_op.constant([], name="c").name, "outer/c:0")
        with variable_scope.variable_scope(
            inner, auxiliary_name_scope=False) as inner1:
          self.assertEqual(inner1.original_name_scope, "outer/")
          self.assertEqual(
              variable_scope.get_variable("w1", []).name, "outer/inner/w1:0")
          self.assertEqual(
              constant_op.constant([], name="c1").name, "outer/c1:0")
        # Recheck: new name scope is NOT created before
        with ops.name_scope("inner"):
          self.assertEqual(
              constant_op.constant([], name="c").name, "outer/inner/c:0")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testCreatedByDefaultNameWhenAuxiliaryNameScopeIsFalse(self):
    # [FUNCTION DOC]
    # Test: same as the basic auxiliary-name-scope-disabled test, but scopes are created from a
    # default name instead of an explicit name.
    with self.cached_session():
      with variable_scope.variable_scope(
          None, default_name="default", auxiliary_name_scope=False) as scope:
        self.assertEqual(scope.original_name_scope, "")
        self.assertEqual(
            variable_scope.get_variable("w", []).name, "default/w:0")
        self.assertEqual(constant_op.constant([], name="c").name, "c:0")
      # Recheck: new name scope is NOT created before
      with ops.name_scope("default"):
        self.assertEqual(
            constant_op.constant([], name="c").name, "default/c:0")

      with variable_scope.variable_scope("outer"):
        with variable_scope.variable_scope(
            None, default_name="default",
            auxiliary_name_scope=False) as inner:
          self.assertEqual(inner.original_name_scope, "outer/")
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/default/w:0")
          self.assertEqual(
              constant_op.constant([], name="c").name, "outer/c:0")
        # Recheck: new name scope is NOT created before
        with ops.name_scope("default"):
          self.assertEqual(
              constant_op.constant([], name="c").name, "outer/default/c:0")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testReenterRootScopeWhenAuxiliaryNameScopeIsFalse(self):
    # [FUNCTION DOC]
    # Test: re-entering the root scope with the auxiliary name scope disabled, at top level and
    # nested in another scope, leaves variable names unprefixed and ops under only the outer
    # name scope.
    with self.cached_session():
      root_scope = variable_scope.get_variable_scope()
      with variable_scope.variable_scope(
          root_scope, auxiliary_name_scope=False) as scope:
        self.assertEqual(scope.original_name_scope, "")
        self.assertEqual(variable_scope.get_variable("w", []).name, "w:0")
        self.assertEqual(constant_op.constant([], name="c").name, "c:0")

      with variable_scope.variable_scope("outer"):
        with variable_scope.variable_scope(
            root_scope, auxiliary_name_scope=False) as inner:
          self.assertEqual(inner.original_name_scope, "")
          self.assertEqual(variable_scope.get_variable("w1", []).name, "w1:0")
          self.assertEqual(
              constant_op.constant([], name="c1").name, "outer/c1:0")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testAuxiliaryNameScopeIsInvalid(self):
    # [FUNCTION DOC]
    # Test: a non-boolean auxiliary name scope value raises a type error mentioning the argument
    # for a default-name scope, a named scope, and a re-entered scope object.
    with self.cached_session():
      with self.assertRaisesRegex(TypeError, "auxiliary_name_scope"):
        with variable_scope.variable_scope(
            None, default_name="scope", auxiliary_name_scope="invalid"):
          pass

      with self.assertRaisesRegex(TypeError, "auxiliary_name_scope"):
        with variable_scope.variable_scope(
            "scope", auxiliary_name_scope="invalid"):
          pass

      with variable_scope.variable_scope("scope") as scope:
        pass
      with self.assertRaisesRegex(TypeError, "auxiliary_name_scope"):
        with variable_scope.variable_scope(
            scope, auxiliary_name_scope="invalid"):
          pass

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testReuseScopeWithoutNameScopeCollision(self):
    # [FUNCTION DOC]
    # Test (regression for a public issue): re-entering a scope without its auxiliary name
    # scope and then entering its original name scope gives expected variable and op names
    # without collisions with a differently nested name scope of the same string.
    # Github issue: #13429
    with self.cached_session():
      with variable_scope.variable_scope("outer"):
        with variable_scope.variable_scope("inner") as inner:
          pass

      with variable_scope.variable_scope(
          inner, auxiliary_name_scope=False) as scope:
        with ops.name_scope(scope.original_name_scope):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/inner/w:0")
          self.assertEqual(
              constant_op.constant([], name="c").name, "outer/inner/c:0")
        with ops.name_scope("inner"):
          self.assertEqual(
              constant_op.constant([], name="c").name, "inner/c:0")

      with variable_scope.variable_scope("another"):
        with variable_scope.variable_scope(
            inner, auxiliary_name_scope=False) as scope1:
          with ops.name_scope(scope1.original_name_scope):
            self.assertEqual(
                variable_scope.get_variable("w1", []).name,
                "outer/inner/w1:0")
            self.assertEqual(
                constant_op.constant([], name="c1").name, "outer/inner/c1:0")
          with ops.name_scope("inner"):
            self.assertEqual(
                constant_op.constant([], name="c").name, "another/inner/c:0")

  # Not converted to use wrap_function because of
  # obtaining different results in the eager case compared to the graph one
  # (different assertions failing after wrapping, in both execution modes)
  @test_util.run_in_graph_and_eager_modes
  def testGetLocalVar(self):
    # [FUNCTION DOC]
    # Test: local variables respect scope naming; in graph mode they are in the local and custom
    # collections but not the trainable collection, and local variable lookup respects reuse.
    # Check that local variable respects naming.
    with variable_scope.variable_scope("outer") as outer:
      with variable_scope.variable_scope(outer, "default", []):
        local_var = variable_scope.get_local_variable(
            "w", [], collections=["foo"])
        self.assertEqual(local_var.name, "outer/w:0")

    if not context.executing_eagerly():
      # Since variable is local, it should be in the local variable collection
      # but not the trainable collection.
      self.assertIn(local_var,
                    ops.get_collection(ops.GraphKeys.LOCAL_VARIABLES))
      self.assertIn(local_var, ops.get_collection("foo"))
      self.assertNotIn(local_var,
                       ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES))
      # Check that local variable respects `reuse`.
      with variable_scope.variable_scope(outer, "default", reuse=True):
        self.assertEqual(
            variable_scope.get_local_variable("w", []).name, "outer/w:0")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testSignatureGetVarVsGetLocalVar(self):
    # [FUNCTION DOC]
    # Test: the functions for variable and local variable creation have the same argument names.
    """get_{local,}variable() must take the same list of args."""
    arg_names = tf_inspect.getargspec(variable_scope.get_variable)[0]
    local_arg_names = tf_inspect.getargspec(
        variable_scope.get_local_variable)[0]
    self.assertEqual(arg_names, local_arg_names)

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testGetVarWithDevice(self):
    # [FUNCTION DOC]
    # Test: a device function applied while building a separate graph is called for variable
    # creation ops, observing the right names and element types for a float and an int variable.
    g = ops.Graph()
    varname_type = []

    def device_func(op):
      # [FUNCTION DOC]
      # HELPER: device function for the enclosing test. Given an operation, records name and
      # dtype attribute of variable-creating ops in a shared list and always returns the first
      # GPU device string. No other side effects.
      if op.type in ["Variable", "VariableV2", "VarHandleOp"]:
        varname_type.append((op.name, op.get_attr("dtype")))
      return "/device:GPU:0"

    with g.as_default():
      with ops.device(device_func):
        _ = variable_scope.get_variable("x", (100, 200))
        _ = variable_scope.get_variable(
            "y", dtype=dtypes.int64, initializer=numpy.arange(73))
    self.assertEqual(varname_type[0], ("x", dtypes.float32))
    self.assertEqual(varname_type[1], ("y", dtypes.int64))

  # Not converted to use wrap_function because of
  # obtaining different results in the eager case compared to the graph one
  @test_util.run_deprecated_v1
  def testGetCollection(self):
    # [FUNCTION DOC]
    # Test (graph mode): the scope's collection accessor returns only variables under that
    # scope's prefix (scope names that are prefixes of each other are separated), for trainable
    # and global collections, and the root scope returns all.
    with self.cached_session():
      _ = variable_scope.get_variable("testGetCollection_a", [])
      _ = variable_scope.get_variable(
          "testGetCollection_b", [], trainable=False)
      with variable_scope.variable_scope("testGetCollection_foo_") as scope1:
        _ = variable_scope.get_variable("testGetCollection_a", [])
        _ = variable_scope.get_variable(
            "testGetCollection_b", [], trainable=False)
        self.assertEqual([
            v.name
            for v in scope1.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
        ], ["testGetCollection_foo_/testGetCollection_a:0"])
        self.assertEqual([
            v.name
            for v in scope1.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
        ], [
            "testGetCollection_foo_/testGetCollection_a:0",
            "testGetCollection_foo_/testGetCollection_b:0"
        ])
      with variable_scope.variable_scope("testGetCollection_foo") as scope2:
        _ = variable_scope.get_variable("testGetCollection_a", [])
        _ = variable_scope.get_variable(
            "testGetCollection_b", [], trainable=False)
        self.assertEqual([
            v.name
            for v in scope2.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
        ], ["testGetCollection_foo/testGetCollection_a:0"])
        self.assertEqual([
            v.name
            for v in scope2.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
        ], [
            "testGetCollection_foo/testGetCollection_a:0",
            "testGetCollection_foo/testGetCollection_b:0"
        ])
      scope = variable_scope.get_variable_scope()
      self.assertEqual([
          v.name for v in scope.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      ], [
          "testGetCollection_a:0", "testGetCollection_b:0",
          "testGetCollection_foo_/testGetCollection_a:0",
          "testGetCollection_foo_/testGetCollection_b:0",
          "testGetCollection_foo/testGetCollection_a:0",
          "testGetCollection_foo/testGetCollection_b:0"
      ])
      self.assertEqual([
          v.name
          for v in scope.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
      ], [
          "testGetCollection_a:0",
          "testGetCollection_foo_/testGetCollection_a:0",
          "testGetCollection_foo/testGetCollection_a:0"
      ])

  # Not converted to use wrap_function because of
  # obtaining different results in the eager case compared to the graph one
  @test_util.run_deprecated_v1
  def testGetTrainableVariablesWithGetVariable(self):
    # [FUNCTION DOC]
    # Test (graph mode): trainable variables listing for a scope when created via get_variable:
    # non-trainable ones excluded; read-synchronized ones default to non-trainable unless
    # trainable is forced true; other synchronization modes stay trainable.
    with self.cached_session():
      _ = variable_scope.get_variable("testGetTrainableVariables_a", [])
      with variable_scope.variable_scope(
          "testGetTrainableVariables_foo") as scope:
        _ = variable_scope.get_variable("testGetTrainableVariables_b", [])
        _ = variable_scope.get_variable(
            "testGetTrainableVariables_c", [], trainable=False)

        # sync `ON_READ` sets trainable=False
        _ = variable_scope.get_variable(
            "testGetTrainableVariables_d", [],
            synchronization=variable_scope.VariableSynchronization.ON_READ)
        self.assertEqual(
            [v.name for v in scope.trainable_variables()],
            ["testGetTrainableVariables_foo/testGetTrainableVariables_b:0"])

        _ = variable_scope.get_variable(
            "testGetTrainableVariables_e", [],
            synchronization=variable_scope.VariableSynchronization.ON_READ,
            trainable=True)
        self.assertEqual([v.name for v in scope.trainable_variables()], [
            "testGetTrainableVariables_foo/testGetTrainableVariables_b:0",
            "testGetTrainableVariables_foo/testGetTrainableVariables_e:0",
        ])

        # All other sync values sets trainable=True
        _ = variable_scope.get_variable(
            "testGetTrainableVariables_f", [],
            synchronization=variable_scope.VariableSynchronization.ON_WRITE)
        self.assertEqual([v.name for v in scope.trainable_variables()], [
            "testGetTrainableVariables_foo/testGetTrainableVariables_b:0",
            "testGetTrainableVariables_foo/testGetTrainableVariables_e:0",
            "testGetTrainableVariables_foo/testGetTrainableVariables_f:0",
        ])

  # Not converted to use wrap_function because of
  # obtaining different results in the eager case compared to the graph one
  @test_util.run_deprecated_v1
  def testGetTrainableVariablesWithVariable(self):
    # [FUNCTION DOC]
    # Test (graph mode): same scenario as the get_variable version, but variables are created
    # via the V1 variable constructor with names.
    with self.cached_session():
      _ = variable_v1.VariableV1(1.0, name="testGetTrainableVariables_a")
      with variable_scope.variable_scope(
          "testGetTrainableVariables_foo") as scope:
        _ = variable_v1.VariableV1(1.0, name="testGetTrainableVariables_b")
        _ = variable_v1.VariableV1(
            1.0, name="testGetTrainableVariables_c", trainable=False)

        # sync `ON_READ` sets trainable=False
        _ = variable_v1.VariableV1(
            1.0,
            name="testGetTrainableVariables_d",
            synchronization=variable_scope.VariableSynchronization.ON_READ)
        self.assertEqual(
            [v.name for v in scope.trainable_variables()],
            ["testGetTrainableVariables_foo/testGetTrainableVariables_b:0"])

        _ = variable_v1.VariableV1(
            1.0,
            name="testGetTrainableVariables_e",
            synchronization=variable_scope.VariableSynchronization.ON_READ,
            trainable=True)
        self.assertEqual([v.name for v in scope.trainable_variables()], [
            "testGetTrainableVariables_foo/testGetTrainableVariables_b:0",
            "testGetTrainableVariables_foo/testGetTrainableVariables_e:0",
        ])

        # All other sync values sets trainable=True
        _ = variable_v1.VariableV1(
            1.0,
            name="testGetTrainableVariables_f",
            synchronization=variable_scope.VariableSynchronization.ON_WRITE)
        self.assertEqual([v.name for v in scope.trainable_variables()], [
            "testGetTrainableVariables_foo/testGetTrainableVariables_b:0",
            "testGetTrainableVariables_foo/testGetTrainableVariables_e:0",
            "testGetTrainableVariables_foo/testGetTrainableVariables_f:0",
        ])

  # Not converted to use wrap_function because of
  # obtaining different results in the eager case compared to the graph one
  @test_util.run_deprecated_v1
  def testGetGlobalVariables(self):
    # [FUNCTION DOC]
    # Test (graph mode): the scope's global variables listing only includes variables created
    # inside that scope.
    with self.cached_session():
      _ = variable_scope.get_variable("testGetGlobalVariables_a", [])
      with variable_scope.variable_scope("testGetGlobalVariables_foo") as scope:
        _ = variable_scope.get_variable("testGetGlobalVariables_b", [])
        self.assertEqual(
            [v.name for v in scope.global_variables()],
            ["testGetGlobalVariables_foo/"
             "testGetGlobalVariables_b:0"])

  # Not converted to use wrap_function because of
  # obtaining different results in the eager case compared to the graph one
  @test_util.run_deprecated_v1
  def testGetLocalVariables(self):
    # [FUNCTION DOC]
    # Test (graph mode): the scope's local variables listing only includes local-collection
    # variables created inside that scope.
    with self.cached_session():
      _ = variable_scope.get_variable(
          "a", [], collections=[ops.GraphKeys.LOCAL_VARIABLES])
      with variable_scope.variable_scope("foo") as scope:
        _ = variable_scope.get_variable(
            "b", [], collections=[ops.GraphKeys.LOCAL_VARIABLES])
        _ = variable_scope.get_variable("c", [])
        self.assertEqual([v.name for v in scope.local_variables()], ["foo/b:0"])

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testGetVariableWithRefDtype(self):
    # [FUNCTION DOC]
    # Test: get_variable accepts the dtype of an existing variable as dtype argument.
    v = variable_scope.get_variable("v", shape=[3, 4], dtype=dtypes.float32)
    # Ensure it is possible to do get_variable with a _ref dtype passed in.
    _ = variable_scope.get_variable("w", shape=[5, 6], dtype=v.dtype)

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testGetVariableWithInitializerWhichTakesNoArgs(self):
    # [FUNCTION DOC]
    # Test: a callable initializer with no arguments is accepted and the variable is named as
    # requested.
    v = variable_scope.get_variable("foo", initializer=lambda: [2])
    self.assertEqual(v.name, "foo:0")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testGetVariableWithInitializerWhichTakesOptionalArgs(self):
    # [FUNCTION DOC]
    # Test: a callable initializer whose arguments all have defaults is accepted.
    v = variable_scope.get_variable("foo", initializer=lambda x=True: [2])
    self.assertEqual(v.name, "foo:0")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testGetVariableWithInitializerWhichTakesUnprovidedArgsAndNoShape(self):
    # [FUNCTION DOC]
    # Test: a callable initializer with a required argument and no shape raises a value error
    # describing the valid initializer forms.
    with self.assertRaisesRegex(
        ValueError,
        "The initializer passed is not valid. It should be a callable with no "
        "arguments and the shape should not be provided or an instance of "
        "`tf.keras.initializers.*' and `shape` should be fully defined."):
      variable_scope.get_variable("foo", initializer=lambda x: [2])

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testTwoGraphs(self):
    # [FUNCTION DOC]
    # Test: entering a scope named with a single underscore, while two nested graphs are
    # default, raises a value error saying it is not a valid scope name.

    def f():
      # [FUNCTION DOC]
      # HELPER: callable passed to the error-assertion in the enclosing test. Creates two graphs,
      # makes both default in nesting and enters an invalid scope name, expecting the failure.
      g1 = ops.Graph()
      g2 = ops.Graph()
      with g1.as_default():
        with g2.as_default():
          with variable_scope.variable_scope("_"):
            pass

    self.assertRaisesRegex(ValueError,
                           "'_' is not a valid (?:root )?scope name", f)


def axis0_into1_partitioner(shape=None, **unused_kwargs):
  # [FUNCTION DOC]
  # HELPER: partitioner for the partitioning tests that does not split. Takes a shape (list of
  # dimension sizes, required in practice) and ignores other keyword arguments; returns a list
  # of ones, one per dimension. Fails with a type error if shape is missing.
  part = [1] * len(shape)
  return part


def axis0_into2_partitioner(shape=None, **unused_kwargs):
  # [FUNCTION DOC]
  # HELPER: partitioner for the partitioning tests that splits into two along the first axis.
  # Takes a shape (required in practice), ignores other keywords; returns partitions per axis:
  # two on axis zero, one elsewhere. Fails with a type error if shape is missing.
  part = [1] * len(shape)
  part[0] = 2
  return part


def axis0_into3_partitioner(shape=None, **unused_kwargs):
  # [FUNCTION DOC]
  # HELPER: partitioner for the partitioning tests that splits into three along the first axis.
  # Takes a shape (required in practice), ignores other keywords; returns partitions per axis:
  # three on axis zero, one elsewhere. Fails with a type error if shape is missing.
  part = [1] * len(shape)
  part[0] = 3
  return part


class VariableScopeWithPartitioningTest(test.TestCase):
  # [CLASS DOC]
  # Tests partitioned variables created through scopes with partitioner functions: naming of
  # parts, reuse and changing partitioners, scalar handling, eager behavior, concatenation axis.

  # Not converted to use wrap_function because of
  # obtaining different results in the eager case compared to the graph one
  @test_util.run_deprecated_v1
  def testResultNameMatchesRequested(self):
    # [FUNCTION DOC]
    # Test (graph mode): a partitioned variable keeps the requested name, its concatenated
    # tensor name has the output suffix, and exactly two part variables exist for a
    # two-way partition.
    with variable_scope.variable_scope(
        "scope0", partitioner=axis0_into2_partitioner):
      v = variable_scope.get_variable("name0", shape=(3, 1, 1))
      self.assertEqual(v.name, "scope0/name0")
      v_concat = v.as_tensor()
      self.assertEqual(v_concat.name, "scope0/name0:0")
      variables = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      self.assertIn("scope0/name0/part_0:0", [x.name for x in variables])
      self.assertIn("scope0/name0/part_1:0", [x.name for x in variables])
      self.assertNotIn("scope0/name0/part_2:0", [x.name for x in variables])

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testBreaksIfPartitioningChanges(self):
    # [FUNCTION DOC]
    # Test: reusing a partitioned variable with a different partitioner (three-way or no split)
    # raises a value error saying the specified and found partitions differ.
    with variable_scope.variable_scope(
        "scope0", partitioner=axis0_into2_partitioner):
      variable_scope.get_variable("name0", shape=(3, 1, 1))

    with variable_scope.variable_scope(
        "scope0", partitioner=axis0_into3_partitioner, reuse=True):
      with self.assertRaisesRegex(
          ValueError,
          "Trying to reuse partitioned variable .* but specified partitions "
          ".* and found partitions .*"):
        variable_scope.get_variable("name0", shape=(3, 1, 1))

    with variable_scope.variable_scope(
        "scope0", partitioner=axis0_into1_partitioner, reuse=True):
      with self.assertRaisesRegex(
          ValueError,
          "Trying to reuse partitioned variable .* but specified partitions "
          ".* and found partitions .*"):
        variable_scope.get_variable("name0", shape=(3, 1, 1))

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testReturnsExistingConcatenatedValueIfReuse(self):
    # [FUNCTION DOC]
    # Test: after marking the scope for reuse, fetching the partitioned variable again returns
    # an equal object.
    with variable_scope.variable_scope(
        "scope0", partitioner=axis0_into2_partitioner):
      v_concat = variable_scope.get_variable("name0", shape=(3, 1, 1))
      variable_scope.get_variable_scope().reuse_variables()
      v_concat_2 = variable_scope.get_variable("name0", shape=(3, 1, 1))
      self.assertEqual(v_concat, v_concat_2)

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testAllowsReuseWithoutPartitioner(self):
    # [FUNCTION DOC]
    # Test: a partitioned variable can be reused in a scope that has no partitioner and gives
    # the identical object.
    with variable_scope.variable_scope(
        "scope0", partitioner=axis0_into2_partitioner):
      v = variable_scope.get_variable("name0", shape=(3, 1, 1))
    with variable_scope.variable_scope("scope0", reuse=True):
      v_reused = variable_scope.get_variable("name0")
    self.assertIs(v, v_reused)

  def testNoReuseInEagerByDefault(self):
    # [FUNCTION DOC]
    # Test (eager mode): requesting the same partitioned variable name twice gives two distinct
    # objects because eager mode does not reuse by default.
    with context.eager_mode():
      with variable_scope.variable_scope(
          "scope0", partitioner=axis0_into2_partitioner):
        v1 = variable_scope.get_variable("name0", shape=(3, 1, 1))
        v2 = variable_scope.get_variable("name0", shape=(3, 1, 1))
        self.assertIsNot(v1, v2)

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testPropagatePartitionerOnReopening(self):
    # [FUNCTION DOC]
    # Test: a scope's partitioner is kept when its scope object is re-entered.
    with variable_scope.variable_scope(
        "scope0", partitioner=axis0_into2_partitioner) as vs:
      self.assertEqual(axis0_into2_partitioner, vs.partitioner)
      with variable_scope.variable_scope(vs) as vs1:
        self.assertEqual(axis0_into2_partitioner, vs1.partitioner)

  # Not converted to use wrap_function because of
  # obtaining different results in the eager case compared to the graph one
  @test_util.run_deprecated_v1
  def testScalarIgnoresPartitioner(self):
    # [FUNCTION DOC]
    # Test (graph mode): a scalar variable is not partitioned despite a partitioner, keeping its
    # plain name in the global collection.
    with variable_scope.variable_scope(
        "scope0", partitioner=axis0_into2_partitioner):
      v = variable_scope.get_variable("name0", shape=())
      self.assertEqual(v.name, "scope0/name0:0")
      variables = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      self.assertIn("scope0/name0:0", [x.name for x in variables])

  def _testPartitionConcatenatesAlongCorrectAxis(self, use_resource):
    # [FUNCTION DOC]
    # HELPER: shared body of the partition-axis tests. Takes use_resource (boolean, whether
    # partitions are resource variables). Creates two variables of shape two by two by two with
    # partitioners splitting axis zero and axis one, and asserts the overall shapes and the
    # shapes of the first two parts. Returns nothing.
    def _part_axis_0(**unused_kwargs):
      # [FUNCTION DOC]
      # HELPER: partitioner splitting two ways on the first axis of three; ignores arguments.
      return (2, 1, 1)

    def _part_axis_1(**unused_kwargs):
      # [FUNCTION DOC]
      # HELPER: partitioner splitting two ways on the second axis of three; ignores arguments.
      return (1, 2, 1)

    with variable_scope.variable_scope("root", use_resource=use_resource):
      v0 = variable_scope.get_variable(
          "n0", shape=(2, 2, 2), partitioner=_part_axis_0)
      v1 = variable_scope.get_variable(
          "n1", shape=(2, 2, 2), partitioner=_part_axis_1)

    self.assertEqual(v0.get_shape(), (2, 2, 2))
    self.assertEqual(v1.get_shape(), (2, 2, 2))

    n0_0 = list(v0)[0]
    n0_1 = list(v0)[1]
    self.assertEqual(n0_0.get_shape(), (1, 2, 2))
    self.assertEqual(n0_1.get_shape(), (1, 2, 2))

    n1_0 = list(v1)[0]
    n1_1 = list(v1)[1]
    self.assertEqual(n1_0.get_shape(), (2, 1, 2))
    self.assertEqual(n1_1.get_shape(), (2, 1, 2))

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testPartitionConcatenatesAlongCorrectAxis(self):
    # [FUNCTION DOC]
    # Test: runs the shared partition-axis check with non-resource variables.
    self._testPartitionConcatenatesAlongCorrectAxis(use_resource=False)

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testPartitionConcatenatesAlongCorrectAxisResource(self):
    # [FUNCTION DOC]
    # Test: runs the shared partition-axis check with resource variables.
    self._testPartitionConcatenatesAlongCorrectAxis(use_resource=True)

  def testPartitionConcatenatesAlongCorrectAxisResourceInEager(self):
    # [FUNCTION DOC]
    # Test (eager mode): runs the shared partition-axis check with resource variables.
    with context.eager_mode():
      self._testPartitionConcatenatesAlongCorrectAxis(use_resource=True)


class VariableScopeWithCustomGetterTest(test.TestCase):
  # [CLASS DOC]
  # Tests custom getters on scopes and calls, and variable creator scopes.

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testNonCallableGetterFails(self):
    # [FUNCTION DOC]
    # Test: a non-callable custom getter, on a scope or per call, raises a value error.
    with self.assertRaisesRegex(ValueError, r"custom_getter .* not callable:"):
      with variable_scope.variable_scope("scope0", custom_getter=3):
        variable_scope.get_variable("name0")
    with self.assertRaisesRegex(ValueError, r"custom_getter .* not callable:"):
      variable_scope.get_variable("name0", custom_getter=3)

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testNoSideEffectsWithIdentityCustomGetter(self):
    # [FUNCTION DOC]
    # Test: a pass-through custom getter changes nothing about reuse: counts invocations (three,
    # one skipped for the reuse scope without getter) and verifies identity of reused variables.
    called = [0]

    def custom_getter(getter, *args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: counting pass-through getter. Increments the shared call counter and returns the
      # result of the underlying getter called with all arguments unchanged.
      called[0] += 1
      return getter(*args, **kwargs)

    with variable_scope.variable_scope(
        "scope", custom_getter=custom_getter) as scope:
      v = variable_scope.get_variable("v", [1])
    with variable_scope.variable_scope(scope, reuse=True):
      v2 = variable_scope.get_variable("v", [1])
    with variable_scope.variable_scope("new_scope") as new_scope:
      v3 = variable_scope.get_variable("v3", [1])
    with variable_scope.variable_scope(
        new_scope, reuse=True, custom_getter=custom_getter):
      v4 = variable_scope.get_variable("v3", [1])

    self.assertIs(v, v2)
    self.assertIs(v3, v4)
    self.assertEqual(3, called[0])  # skipped one in the first new_scope

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testSynchronizationAndAggregationWithCustomGetter(self):
    # [FUNCTION DOC]
    # Test: custom getters receive the synchronization and aggregation keyword arguments with
    # defaults (automatic, none) or the values explicitly passed; counts two calls.
    called = [0]
    synchronization = variable_scope.VariableSynchronization.AUTO
    aggregation = variable_scope.VariableAggregation.NONE

    def custom_getter(getter, *args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: getter that counts calls, asserts the synchronization and aggregation keywords
      # equal the currently expected values from the enclosing test, then calls the underlying
      # getter with all arguments and returns its result.
      called[0] += 1

      # Verify synchronization and aggregation kwargs are as expected.
      self.assertEqual(kwargs["synchronization"], synchronization)
      self.assertEqual(kwargs["aggregation"], aggregation)
      return getter(*args, **kwargs)

    with variable_scope.variable_scope("scope", custom_getter=custom_getter):
      variable_scope.get_variable("v", [1])
    self.assertEqual(1, called[0])

    with variable_scope.variable_scope("scope", custom_getter=custom_getter):
      synchronization = variable_scope.VariableSynchronization.ON_READ
      aggregation = variable_scope.VariableAggregation.MEAN
      variable_scope.get_variable(
          "v1", [1], synchronization=synchronization, aggregation=aggregation)

    self.assertEqual(2, called[0])

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testCustomGetterWithReuse(self):
    # [FUNCTION DOC]
    # Test: a custom getter sees the reuse keyword and may wrap results differently, giving
    # distinct tensor names for first creation and reuse.
    # Custom getter can choose to behave differently on reused variables.
    def custom_getter(getter, *args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: getter that gets the variable, then returns an identity of it named reused when
      # the reuse keyword is truthy and not_reused otherwise.
      var = getter(*args, **kwargs)
      if kwargs["reuse"]:
        # This can be used, e.g., for changing the caching device if needed.
        return array_ops.identity(var, name="reused")
      else:
        return array_ops.identity(var, name="not_reused")

    with variable_scope.variable_scope(
        "scope", custom_getter=custom_getter) as scope:
      v = variable_scope.get_variable("v", [1])
    with variable_scope.variable_scope(scope, reuse=True):
      v2 = variable_scope.get_variable("v", [1])

    self.assertEqual(v.name, "not_reused:0")
    self.assertEqual(v2.name, "reused:0")

  # Not converted to use wrap_function because of
  # ValueError: Fetch argument <tf.Tensor 'custom_getter/add:0' shape=(1, 2, 3)
  # dtype=float32> cannot be interpreted as a Tensor. (Tensor
  # Tensor("custom_getter/add:0", shape=(1, 2, 3), dtype=float32) is not an
  # element of this graph.)
  @test_util.run_deprecated_v1
  def testGetterThatCreatesTwoVariablesAndSumsThem(self):
    # [FUNCTION DOC]
    # Test (graph mode): a getter creating two variables and returning their sum gives a
    # tensor of the right shape and name, two trainable variables, and a value equal to the sum.

    def custom_getter(getter, name, *args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: getter for the enclosing test. Gets two underlying variables named with slash
      # zero and one suffixes and returns their sum inside a name scope. Forwards all extra args.
      g_0 = getter("%s/0" % name, *args, **kwargs)
      g_1 = getter("%s/1" % name, *args, **kwargs)
      with ops.name_scope("custom_getter"):
        return g_0 + g_1

    with variable_scope.variable_scope("scope", custom_getter=custom_getter):
      v = variable_scope.get_variable("v", [1, 2, 3])

    self.assertEqual([1, 2, 3], v.get_shape())
    true_vars = variables_lib.trainable_variables()
    self.assertEqual(2, len(true_vars))
    self.assertEqual("scope/v/0:0", true_vars[0].name)
    self.assertEqual("scope/v/1:0", true_vars[1].name)
    self.assertEqual("custom_getter/add:0", v.name)
    with self.cached_session() as sess:
      variables_lib.global_variables_initializer().run()
      np_vars, np_v = self.evaluate([true_vars, v])
      self.assertAllClose(np_v, sum(np_vars))

  # Not converted to use wrap_function because of
  # ValueError: Fetch argument <tf.Tensor 'sum_getter_2/add:0' shape=(1, 2, 3)
  # dtype=float32> cannot be interpreted as a Tensor. (Tensor
  # Tensor("sum_getter_2/add:0", shape=(1, 2, 3), dtype=float32) is not an
  # element of this graph.)
  @test_util.run_deprecated_v1
  def testNestedCustomGetters(self):
    # [FUNCTION DOC]
    # Test (graph mode): nested scopes with sum and product getters create eight variables with
    # expected nested names, and the final tensor equals the product-of-sums-of-products value.

    def sum_getter(getter, name, *args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: getter that returns the sum of two variables fetched under the sum names.
      g_0 = getter("%s/sum_0" % name, *args, **kwargs)
      g_1 = getter("%s/sum_1" % name, *args, **kwargs)
      with ops.name_scope("sum_getter"):
        return g_0 + g_1

    def prod_getter(getter, name, *args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: getter that returns the product of two variables fetched under the product names.
      g_0 = getter("%s/prod_0" % name, *args, **kwargs)
      g_1 = getter("%s/prod_1" % name, *args, **kwargs)
      with ops.name_scope("prod_getter"):
        return g_0 * g_1

    with variable_scope.variable_scope("prod_scope", custom_getter=prod_getter):
      with variable_scope.variable_scope("sum_scope", custom_getter=sum_getter):
        with variable_scope.variable_scope(
            "inner_sum_scope", custom_getter=sum_getter):
          # take sums of sums of products
          v = variable_scope.get_variable("v", [1, 2, 3])

    self.assertEqual([1, 2, 3], v.get_shape())
    true_vars = variables_lib.trainable_variables()
    self.assertEqual(8, len(true_vars))
    template = (
        "prod_scope/sum_scope/inner_sum_scope/v/sum_%d/sum_%d/prod_%d:0")
    self.assertEqual(template % (0, 0, 0), true_vars[0].name)
    self.assertEqual(template % (0, 0, 1), true_vars[1].name)
    self.assertEqual(template % (0, 1, 0), true_vars[2].name)
    self.assertEqual(template % (0, 1, 1), true_vars[3].name)
    self.assertEqual(template % (1, 0, 0), true_vars[4].name)
    self.assertEqual(template % (1, 0, 1), true_vars[5].name)
    self.assertEqual(template % (1, 1, 0), true_vars[6].name)
    self.assertEqual(template % (1, 1, 1), true_vars[7].name)

    with self.cached_session() as sess:
      variables_lib.global_variables_initializer().run()
      np_vars, np_v = self.evaluate([true_vars, v])
      # take products of sums of products
      self.assertAllClose(
          np_v, (((np_vars[0] * np_vars[1]) + (np_vars[2] * np_vars[3])) + (
              (np_vars[4] * np_vars[5]) + (np_vars[6] * np_vars[7]))))

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVariableCreator(self):
    # [FUNCTION DOC]
    # Test: nested variable creator scopes chain (the inner forces a name, the outer observes
    # it), and creators receive synchronization and aggregation keywords from get_variable.
    variable_names = []

    def creator_a(next_creator, **kwargs):
      # [FUNCTION DOC]
      # HELPER: records the name keyword (empty if missing) then defers to the next creator.
      variable_names.append(kwargs.get("name", ""))
      return next_creator(**kwargs)

    def creator_b(next_creator, **kwargs):
      # [FUNCTION DOC]
      # HELPER: overrides the name keyword with a fixed forced name then defers to the next one.
      kwargs["name"] = "forced_name"
      return next_creator(**kwargs)

    with variable_scope.variable_creator_scope(creator_a):
      with variable_scope.variable_creator_scope(creator_b):
        variable_v1.VariableV1(1.0, name="one_name")

    self.assertEqual(variable_names[0], "forced_name")

    called = [False]

    def creater_c(next_creator, **kwargs):
      # [FUNCTION DOC]
      # HELPER: sets a called flag, asserts on-write synchronization and mean aggregation, then
      # defers to the next creator.
      called[0] = True
      self.assertEqual(kwargs["synchronization"],
                       variable_scope.VariableSynchronization.ON_WRITE)
      self.assertEqual(kwargs["aggregation"],
                       variable_scope.VariableAggregation.MEAN)
      return next_creator(**kwargs)

    with variable_scope.variable_creator_scope(creater_c):
      variable_scope.get_variable(
          "v", [],
          synchronization=variable_scope.VariableSynchronization.ON_WRITE,
          aggregation=variable_scope.VariableAggregation.MEAN)
    self.assertTrue(called[0])

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVariableCreatorNestingError(self):
    # [FUNCTION DOC]
    # Test: exiting a creator scope out of order raises a runtime error; the graph's creator
    # stack is saved and restored in a finally block to clean up.

    def creator(next_creator, **kwargs):
      # [FUNCTION DOC]
      # HELPER: pass-through creator for the nesting-error test.
      return next_creator(**kwargs)

    # Save the state so we can clean up at the end.
    graph = ops.get_default_graph()
    old_creator_stack = graph._variable_creator_stack

    try:
      scope = variable_scope.variable_creator_scope(creator)
      scope.__enter__()
      with variable_scope.variable_creator_scope(creator):
        with self.assertRaises(RuntimeError):
          scope.__exit__(None, None, None)
    finally:
      graph._variable_creator_stack = old_creator_stack


class PartitionInfoTest(test.TestCase):
  # [CLASS DOC]
  # Tests the internal partition information helper (full shape and offset of a shard).

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testConstructorChecks(self):
    # [FUNCTION DOC]
    # Test: constructing partition info with non-sequence arguments raises a type error; with
    # different lengths or an offset not below the shape raises a value error.
    # Invalid arg types.
    with self.assertRaises(TypeError):
      variable_scope._PartitionInfo(full_shape=None, var_offset=[0, 1])
    with self.assertRaises(TypeError):
      variable_scope._PartitionInfo(full_shape=[0, 1], var_offset=None)
    with self.assertRaises(TypeError):
      variable_scope._PartitionInfo(full_shape="foo", var_offset=[0, 1])
    with self.assertRaises(TypeError):
      variable_scope._PartitionInfo(full_shape=[0, 1], var_offset="foo")

    # full_shape and var_offset must have same length.
    with self.assertRaises(ValueError):
      variable_scope._PartitionInfo(full_shape=[0, 1], var_offset=[0])
    # Offset must always be less than shape.
    with self.assertRaises(ValueError):
      variable_scope._PartitionInfo(full_shape=[1, 1], var_offset=[0, 1])

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testSingleOffset(self):
    # [FUNCTION DOC]
    # Test: the single-offset query returns the offset along the sliced axis (four) for a
    # partitioned case and zero when unpartitioned.
    partition_info = variable_scope._PartitionInfo(
        full_shape=[9, 3], var_offset=[4, 0])
    self.assertEqual(4, partition_info.single_offset([1, 3]))

    # Tests when the variable isn't partitioned at all.
    partition_info = variable_scope._PartitionInfo(
        full_shape=[9, 3], var_offset=[0, 0])
    self.assertEqual(0, partition_info.single_offset([9, 3]))

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testSingleSliceDim(self):
    # [FUNCTION DOC]
    # Test: the single-slice-dimension query raises a type error for null shape and value errors
    # for wrong rank, overflow, or ambiguity; otherwise returns the sliced axis index.
    partition_info = variable_scope._PartitionInfo(
        full_shape=[9, 3], var_offset=[4, 0])
    # Invalid shape.
    with self.assertRaises(TypeError):
      partition_info.single_slice_dim(None)

    # Rank of shape differs from full_shape.
    with self.assertRaises(ValueError):
      partition_info.single_slice_dim([1, 2, 3])

    # Shape is too large given var_offset (4+6 > 9).
    with self.assertRaises(ValueError):
      partition_info.single_slice_dim([6, 3])

    # Multiple possible slice dim from shape.
    with self.assertRaises(ValueError):
      partition_info.single_slice_dim([1, 1])

    partition_info = variable_scope._PartitionInfo(
        full_shape=[9, 3], var_offset=[0, 0])
    self.assertEqual(1, partition_info.single_slice_dim([9, 2]))
    partition_info = variable_scope._PartitionInfo(
        full_shape=[9, 3], var_offset=[4, 0])
    self.assertEqual(0, partition_info.single_slice_dim([2, 3]))


class VariableScopeMultithreadedTest(test.TestCase):
  # [CLASS DOC]
  # Tests scope behavior across threads sharing one graph.

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testTwoThreadsDisjointScopeEntry(self):
    # [FUNCTION DOC]
    # Test: two threads run one after the other; the first creates a variable and the second
    # fails with a value error on the same name since the store is shared. Both threads joined.

    def thread_fn(i, graph):
      # [FUNCTION DOC]
      # HELPER: thread body. In the given graph enters a scope; thread index 0 creates a
      # variable and checks its name, others expect a value error. Returns nothing.
      with graph.as_default():
        with variable_scope.variable_scope("foo"):
          if i == 0:
            v = variable_scope.get_variable("v", [])
            self.assertEqual("foo/v:0", v.name)
          else:
            # Any thread after the first one should fail to create variable
            # with the same name.
            with self.assertRaises(ValueError):
              variable_scope.get_variable("v", [])

    graph = ops.get_default_graph()
    threads = [
        threading.Thread(target=thread_fn, args=(
            i,
            graph,
        )) for i in range(2)
    ]

    threads[0].start()
    # Allow thread 0 to finish before starting thread 1.
    threads[0].join()
    threads[1].start()
    threads[1].join()

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testTwoThreadsNestedScopeEntry(self):
    # [FUNCTION DOC]
    # Test: two threads hold the same scope open concurrently using events; the second cannot
    # create the same variable; then both are released and joined.

    def thread_fn(i, graph, run_event, pause_event):
      # [FUNCTION DOC]
      # HELPER: thread body. Same as the disjoint-thread version, but afterwards signals the
      # pause event and blocks until the run event is set. Returns nothing.
      with graph.as_default():
        with variable_scope.variable_scope("foo"):
          if i == 0:
            v = variable_scope.get_variable("v", [])
            self.assertEqual("foo/v:0", v.name)
          else:
            # Any thread after the first one should fail to create variable
            # with the same name.
            with self.assertRaises(ValueError):
              variable_scope.get_variable("v", [])
          pause_event.set()
          run_event.wait()

    graph = ops.get_default_graph()
    run_events = [threading.Event() for _ in range(2)]
    pause_events = [threading.Event() for _ in range(2)]
    threads = [
        threading.Thread(
            target=thread_fn, args=(i, graph, run_events[i], pause_events[i]))
        for i in range(2)
    ]

    # Start first thread.
    threads[0].start()
    pause_events[0].wait()
    # Start next thread once the first thread has paused.
    threads[1].start()
    pause_events[1].wait()
    # Resume both threads.
    run_events[0].set()
    run_events[1].set()
    threads[0].join()
    threads[1].join()

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testReenterMainScope(self):
    # [FUNCTION DOC]
    # Test: a worker thread re-entering the main thread's scope object gets the main prefix,
    # while a fresh scope in that thread gets none.

    def thread_fn(graph, main_thread_scope):
      # [FUNCTION DOC]
      # HELPER: thread body. Takes the graph and the main scope; creates a variable under the
      # re-entered scope and one under a fresh scope and checks both names. Returns nothing.
      with graph.as_default():
        # Variable created with main scope will have prefix "main".
        with variable_scope.variable_scope(main_thread_scope):
          with variable_scope.variable_scope("foo"):
            v = variable_scope.get_variable("v", [])
            self.assertEqual("main/foo/v:0", v.name)

        # Variable created outside main scope will not have prefix "main".
        with variable_scope.variable_scope("bar"):
          v = variable_scope.get_variable("v", [])
          self.assertEqual("bar/v:0", v.name)

    graph = ops.get_default_graph()
    with variable_scope.variable_scope("main") as main_thread_scope:
      thread = threading.Thread(
          target=thread_fn, args=(graph, main_thread_scope))
      thread.start()
      thread.join()


if __name__ == "__main__":
  test.main()
