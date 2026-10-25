# [MODULE DOC]
# Tests for resource variables (resource_variable_ops.ResourceVariable and the raw resource ops such
# as var handle creation, assign, read, scatter updates, gather, destroy) covering both eager and
# graph execution. Contains a small composite-tensor variable wrapper with a custom gradient
# protocol, a helper that creates variable handles safely in eager mode, and the large
# ResourceVariableOpsTest case. Tests check values, dtypes, shapes, device placement, error
# types and messages, gradients, protobuf round trips, pickling, deep copying and garbage collection.
# Not a helper module; no import-time side effects beyond defining the classes and test case.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.resource_variable_ops."""
import copy
import gc
import os
import pickle
import re

from absl.testing import parameterized
import numpy as np

from tensorflow.core.framework import full_type_pb2
from tensorflow.core.framework import tensor_pb2
from tensorflow.python.compat import compat as forward_compat
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import cpp_shape_inference_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import extension_type
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import test
from tensorflow.python.training import momentum
from tensorflow.python.training import saver
from tensorflow.python.training import training_util
from tensorflow.python.util import compat
from tensorflow.python.util import nest


class CompositeVariableGradient(
    composite_tensor_gradient.CompositeTensorGradient):
  # [CLASS DOC]
  # HELPER: gradient protocol used by the CompositeVariable test type.
  # Implements the composite-tensor gradient interface by splitting a CompositeVariable value into its
  # components and rebuilding one from component gradients, both via its type spec. Used by
  # testGradientCompositeVariable so gradients flow to the wrapped variable.
  """Gradient protocol for CompositeVariable."""

  def get_gradient_components(self, value):
    # [FUNCTION DOC]
    # HELPER: part of the composite gradient protocol for CompositeVariable.
    # Takes a composite value and returns the list of components that gradients are computed for, by
    # asking the value's type spec to convert the value to components. No side effects.
    return value._type_spec._to_components(value)

  def replace_gradient_components(self, value, component_grads):
    # [FUNCTION DOC]
    # HELPER: part of the composite gradient protocol for CompositeVariable.
    # Takes the composite value and a structure of component gradients and returns a new composite value
    # rebuilt from those gradients through the type spec. The original value is not changed.
    return value._type_spec._from_components(component_grads)


class CompositeVariable(extension_type.ExtensionType):
  # [CLASS DOC]
  # HELPER: extension type used only by the gradient test.
  # Extension type with a single field holding a resource variable. Its composite gradient attribute is
  # set to the gradient protocol class above so the variable receives gradients when the wrapper is
  # differentiated. Instances are constructed from a variable in testGradientCompositeVariable.
  v: resource_variable_ops.ResourceVariable

  __composite_gradient__ = CompositeVariableGradient()


def _eager_safe_var_handle_op(*args, **kwargs):
  # [FUNCTION DOC]
  # HELPER: wrapper around the variable handle op for the tests in this file.
  # Forwards all positional and keyword arguments to the var handle op. When running eagerly and no
  # shared name was given, it supplies the runtime's reserved anonymous name as the shared name so the
  # runtime generates a unique name and handles do not accidentally share state. Returns the resource
  # handle tensor. Reads the current execution mode; creates a resource handle.
  # When running in eager mode the `shared_name` should be set to the
  # `anonymous_name` to avoid spurious sharing issues. The runtime generates a
  # unique name on our behalf when the reserved `anonymous_name` is used as the
  # `shared_name`.
  if context.executing_eagerly() and "shared_name" not in kwargs:
    kwargs["shared_name"] = context.anonymous_name()
  return resource_variable_ops.var_handle_op(*args, **kwargs)


@test_util.with_eager_op_as_function
@test_util.with_control_flow_v2
class ResourceVariableOpsTest(test_util.TensorFlowTestCase,
                              parameterized.TestCase):
  # [CLASS DOC]
  # Test case for resource variables and the resource variable ops. It is decorated to run ops as
  # functions and with control flow v2, and mixes in parameterized testing. Most tests run in eager
  # mode, graph mode, or both; a few are deprecated-v1 graph tests. A tearDown check ensures no
  # uncollectable garbage is left behind after each test.

  def tearDown(self):
    # [FUNCTION DOC]
    # Runs a garbage collection pass, asserts the list of uncollectable objects (reference cycles with
    # finalizers) is empty, then calls the parent tearDown. Takes no arguments, returns nothing; fails the
    # test if garbage remains.
    gc.collect()
    # This will only contain uncollectable garbage, i.e. reference cycles
    # involving objects with __del__ defined.
    self.assertEmpty(gc.garbage)
    super(ResourceVariableOpsTest, self).tearDown()

  def testLocalVariables(self):
    # [FUNCTION DOC]
    # Defines a compiled-off traced function that creates a variable with variable lifting disabled,
    # adds 5 and reads it. Calls it three times and asserts each result is 8 and the function was traced
    # exactly once, showing the local variable is re-created per call, not lifted out.
    num_traces = 0

    # TODO(b/210930091): Test jit_compile=True when the bridge work is done.
    @def_function.function(jit_compile=False)
    def f():
      # [FUNCTION DOC]
      # HELPER: traced function body for testLocalVariables.
      # Increments a trace counter in the enclosing test, creates a variable initialized to 3 with variable
      # lifting disabled, adds 5 and returns its value. Takes no arguments; mutates the enclosing counter.
      nonlocal num_traces
      num_traces += 1
      v = variables.Variable(3, experimental_enable_variable_lifting=False)
      v.assign_add(5)
      return v.read_value()

    self.assertEqual(num_traces, 0)
    for _ in range(3):
      self.assertAllClose(f(), 8)
      self.assertEqual(num_traces, 1)

  @test_util.run_deprecated_v1
  def testHandleDtypeShapeMatch(self):
    # [FUNCTION DOC]
    # Graph-mode test. Creates an int32 scalar handle, then asserts that assigning a float value or an
    # int32 vector raises a value error, while assigning an int32 scalar succeeds.
    with self.cached_session():
      handle = _eager_safe_var_handle_op(dtype=dtypes.int32, shape=[])
      with self.assertRaises(ValueError):
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant(0.0, dtype=dtypes.float32)).run()
      with self.assertRaises(ValueError):
        resource_variable_ops.assign_variable_op(handle,
                                                 constant_op.constant(
                                                     [0],
                                                     dtype=dtypes.int32)).run()
      resource_variable_ops.assign_variable_op(handle,
                                               constant_op.constant(
                                                   0,
                                                   dtype=dtypes.int32)).run()

  @parameterized.parameters(dtypes.int4, dtypes.uint4)
  @test_util.disable_xla("b/183567451: XLA doesn't yet support int4")
  def testInt4(self, dtype):
    # [FUNCTION DOC]
    # Parameterized over int4 and uint4 (XLA disabled). In eager mode creates a variable with value 1,
    # asserts it reads 1, assigns 2 and asserts 2. If a GPU is available also creates one on the first
    # GPU, asserting the device name and value.
    with context.eager_mode():
      v = resource_variable_ops.ResourceVariable(1, dtype=dtype)
      self.assertAllEqual(1, v.numpy())
      v.assign(2)
      self.assertAllEqual(2, v.numpy())

      if test_util.is_gpu_available():
        with ops.device("gpu:0"):
          v = resource_variable_ops.ResourceVariable(3, dtype=dtype)
          self.assertEqual(
              "/job:localhost/replica:0/task:0/device:GPU:0", v.device
          )
          self.assertAllEqual(3, v.numpy())

  @test_util.run_gpu_only
  def testGPUBfloat16(self):
    # [FUNCTION DOC]
    # Runs only when a GPU is present. In eager mode creates a bfloat16 variable on the first GPU and
    # asserts its device string and that its value reads 1.
    with context.eager_mode(), ops.device("gpu:0"):
      v = resource_variable_ops.ResourceVariable(1, dtype=dtypes.bfloat16)
      self.assertEqual("/job:localhost/replica:0/task:0/device:GPU:0",
                       v.device)
      self.assertAllEqual(1, v.numpy())

  @parameterized.parameters(
      dtypes.int8, dtypes.uint8, dtypes.int16, dtypes.uint16, dtypes.uint32,
      dtypes.int64, dtypes.uint64)
  @test_util.run_gpu_only
  def testGPUInteger(self, dtype):
    # [FUNCTION DOC]
    # Parameterized over several integer dtypes, GPU only, eager mode. Exercises a variable on the first
    # GPU: asserts device and initial value, assign_add and assign_sub, then on a two element vector
    # checks scatter add, scatter update, scatter max, scatter nd add/update/max and gather nd, asserting
    # the resulting contents after each operation.
    with context.eager_mode(), ops.device("gpu:0"):
      v = resource_variable_ops.ResourceVariable(1, dtype=dtype)
      self.assertEqual("/job:localhost/replica:0/task:0/device:GPU:0", v.device)
      self.assertAllEqual(1, v.numpy())
      v.assign_add(1)
      self.assertAllEqual(2, v.numpy())
      v.assign_sub(1)
      self.assertAllEqual(1, v.numpy())
      v = resource_variable_ops.ResourceVariable([1, 2], dtype=dtype)
      self.evaluate(
          v.scatter_add(
              indexed_slices.IndexedSlices(
                  indices=[1],
                  values=constant_op.constant([2], dtype=dtype))))
      self.assertAllEqual([1, 4], v.numpy())
      self.evaluate(
          v.scatter_update(
              indexed_slices.IndexedSlices(
                  indices=[1],
                  values=constant_op.constant([5], dtype=dtype))))
      self.assertAllEqual([1, 5], v.numpy())
      self.evaluate(
          v.scatter_max(
              indexed_slices.IndexedSlices(
                  indices=[0, 1],
                  values=constant_op.constant([2, 2], dtype=dtype))))
      self.assertAllEqual([2, 5], v.numpy())
      self.evaluate(v.scatter_nd_add(indices=[[1]], updates=[2]))
      self.assertAllEqual([2, 7], v.numpy())
      self.evaluate(v.scatter_nd_update(indices=[[1]], updates=[2]))
      self.assertAllEqual([2, 2], v.numpy())
      self.evaluate(v.scatter_nd_max(indices=[[1]], updates=[3]))
      self.assertAllEqual([2, 3], v.numpy())
      self.assertAllEqual(v.gather_nd([1]), 3)

  def testEagerNameNotIdentity(self):
    # [FUNCTION DOC]
    # In eager mode creates two variables with the same name and different values; asserts each keeps its
    # own value, i.e. a name does not make variables identical.
    with context.eager_mode():
      v0 = resource_variable_ops.ResourceVariable(1.0, name="a")
      v1 = resource_variable_ops.ResourceVariable(2.0, name="a")
      self.assertAllEqual(v0.numpy(), 1.0)
      self.assertAllEqual(v1.numpy(), 2.0)

  def testEagerNameNotNeeded(self):
    # [FUNCTION DOC]
    # In eager mode creates an unnamed variable with value 1.0 and asserts it reads back 1.0.
    with context.eager_mode():
      v0 = resource_variable_ops.ResourceVariable(1.0)
      self.assertAllEqual(v0.numpy(), 1.0)

  def testReadVariableDtypeMismatchEager(self):
    # [FUNCTION DOC]
    # Eager mode: creates an int32 handle, assigns 1, then reads it requesting float32. Asserts an
    # invalid-argument error whose message mentions reading with the wrong dtype (either kernel or shape
    # inference wording is accepted).
    with context.eager_mode():
      handle = _eager_safe_var_handle_op(
          dtype=dtypes.int32, shape=[1], name="foo")
      resource_variable_ops.assign_variable_op(handle, 1)
      # The error message varies depending on whether it is being raised
      # by the kernel or shape inference. The shape inference code path can
      # be reached when running in eager op as function mode where each op
      # is wrapped in a tf.function.
      with self.assertRaisesRegex(
          errors.InvalidArgumentError,
          r"Trying to read variable with wrong dtype. "
          r"Expected (float|int32) got (int32|float)"):
        _ = resource_variable_ops.read_variable_op(handle, dtype=dtypes.float32)

  def testEagerInitializedValue(self):
    # [FUNCTION DOC]
    # In eager mode creates a variable with value 1.0 and asserts both its numpy value and read_value
    # result equal 1.0.
    with context.eager_mode():
      variable = resource_variable_ops.ResourceVariable(1.0, name="eager-init")
      self.assertAllEqual(variable.numpy(), 1.0)
      self.assertAllEqual(variable.read_value().numpy(), 1.0)

  def testInitializeVariableUsingInitializedValue(self):
    # [FUNCTION DOC]
    # Creates one variable and then another whose initial value is a conditional that reads the first
    # variable if it is initialized, else its initial value. Asserts a similar conditional on the second
    # variable evaluates to 1.0.
    var1 = resource_variable_ops.ResourceVariable(1.0, name="var1")
    var2 = resource_variable_ops.ResourceVariable(
        tf_cond.cond(
            variable_v1.is_variable_initialized(var1), var1.read_value,
            lambda: var1.initial_value),
        name="var2")
    self.assertAllEqual(
        tf_cond.cond(
            variable_v1.is_variable_initialized(var2), var2.read_value,
            lambda: var2.initial_value), 1.0)

  def testEagerBool(self):
    # [FUNCTION DOC]
    # In eager mode asserts that converting a variable holding false to a bool gives false.
    with context.eager_mode():
      v = resource_variable_ops.ResourceVariable(False, name="bool_test")
      self.assertAllEqual(bool(v), False)

  def testEagerDeepCopy(self):
    # [FUNCTION DOC]
    # Eager mode: creates a 4x4x4 variable with on-read synchronization and sum aggregation, deep-copies
    # it, asserts name, shape, device, synchronization and aggregation match and values are equal, then
    # assigns new values to the copy and asserts the original is unchanged.
    with context.eager_mode():
      init_value = np.ones((4, 4, 4))
      variable = resource_variable_ops.ResourceVariable(
          init_value,
          name="init",
          synchronization=variables.VariableSynchronization.ON_READ,
          aggregation=variables.VariableAggregation.SUM)

      copied_variable = copy.deepcopy(variable)
      self.assertEqual(variable.name, copied_variable.name)
      self.assertEqual(variable.shape, copied_variable.shape)
      self.assertEqual(variable.device, copied_variable.device)
      self.assertEqual(variable.synchronization,
                       copied_variable.synchronization)
      self.assertEqual(variable.aggregation, copied_variable.aggregation)

      # The copied variable should have the same value as the original.
      self.assertAllEqual(variable.numpy(), copied_variable.numpy())

      # Updates to the copy should not be reflected in the original.
      copied_variable.assign(4 * np.ones((4, 4, 4)))
      self.assertNotAllEqual(variable.numpy(), copied_variable.numpy())

  @test_util.run_deprecated_v1
  def testGraphDeepCopy(self):
    # [FUNCTION DOC]
    # Graph-mode test: asserts that deep copying a resource variable raises a not-implemented error.
    with self.cached_session():
      init_value = np.ones((4, 4, 4))
      variable = resource_variable_ops.ResourceVariable(init_value,
                                                        name="init")
      with self.assertRaises(NotImplementedError):
        copy.deepcopy(variable)

  @test_util.run_in_graph_and_eager_modes
  def testStridedSliceAssign(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. Assigns 2.0 to the first element of a two element variable through
    # a slice and asserts the variable becomes both elements 2.0.
    v = resource_variable_ops.ResourceVariable([1.0, 2.0])
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(v[0].assign(2.0))
    self.assertAllEqual(self.evaluate(v), [2.0, 2.0])

  @test_util.run_in_graph_and_eager_modes
  def testVariableShape(self):
    # [FUNCTION DOC]
    # Runs in both modes. Asserts the variable shape op on a variable handle yields constant value [2],
    # and in graph mode that the shape is produced by a constant op.
    v = resource_variable_ops.ResourceVariable([1., 1.])
    vshape = resource_variable_ops.variable_shape(v.handle)
    self.assertAllEqual(
        tensor_util.constant_value(vshape),
        [2])
    if not context.executing_eagerly():
      self.assertEqual("Const", vshape.op.type)

  @test_util.run_deprecated_v1
  def testDifferentAssignGraph(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: there is no assertion; the test only passes if assign after the default graph is
    # reset does not raise, so its intent (assign must not convert in the wrong graph) is checked
    # implicitly.
    # Graph-mode test: creates a variable in a private graph, resets the default graph, then assigns 2.0.
    with ops.Graph().as_default():
      v = resource_variable_ops.ResourceVariable(1.0)
    ops.reset_default_graph()
    v.assign(2.0)  # Note: this fails if we run convert_to_tensor on not the
    # variable graph.

  @test_util.run_deprecated_v1
  def testFetchHandle(self):
    # [FUNCTION DOC]
    # Graph-mode test: creates an int32 handle and asserts that evaluating it returns a non-empty value.
    with self.cached_session():
      handle = _eager_safe_var_handle_op(
          dtype=dtypes.int32, shape=[1], name="foo")
      self.assertNotEmpty(self.evaluate(handle))

  @test_util.run_deprecated_v1
  def testCachedValueReadBeforeWrite(self):
    # [FUNCTION DOC]
    # Graph-mode test: creates a variable with a CPU caching device, initializes it, and in one run
    # reads it and adds 1.0; asserts the value read is the old 0.0 (read happens before the write).
    with self.cached_session() as sess:
      v = resource_variable_ops.ResourceVariable(0.0, caching_device="cpu:0")
      self.evaluate(v.initializer)
      value, _ = sess.run([v, v.assign_add(1.0)])
      self.assertAllEqual(value, 0.0)

  def testAssignVariableDtypeMismatchEager(self):
    # [FUNCTION DOC]
    # Eager mode: creates an int32 handle and assigns an integer, then assigns a float32 value. Asserts
    # an invalid-argument error saying the variable has a wrong dtype (expected int32 got float).
    with context.eager_mode():
      handle = _eager_safe_var_handle_op(
          dtype=dtypes.int32, shape=[1], name="foo")
      resource_variable_ops.assign_variable_op(
          handle, constant_op.constant([1]))
      # The error message varies depending on whether it is being raised
      # by the kernel or shape inference. The shape inference code path can
      # be reached when running in eager op as function mode where each op
      # is wrapped in a tf.function.
      with self.assertRaisesRegex(
          errors.InvalidArgumentError, r"Trying to .* variable with wrong "
          r"dtype. Expected int32 got float"):
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant([1.], dtype=dtypes.float32))

  def testRepr(self):
    # [FUNCTION DOC]
    # Eager mode: asserts the printed representation of an int variable with value 1 equals the
    # expected text naming the variable, shape, dtype and numpy value.
    with context.eager_mode():
      v = resource_variable_ops.ResourceVariable(1)
      text = "%r" % v
      error_msg = "<tf.Variable 'Variable:0' shape=() dtype=int32, numpy=1>"
      self.assertEqual(error_msg, text)

  def testReprUnavailable(self):
    # [FUNCTION DOC]
    # Eager mode: replaces the variable's read method with one that raises a value error and asserts the
    # representation prints the value as unavailable instead of failing.
    with context.eager_mode():
      v = resource_variable_ops.ResourceVariable(1)

      # Monkey-patch this variable to not have an available value
      def broken_read():
        # [FUNCTION DOC]
        # HELPER: stand-in for the variable's read method in testReprUnavailable.
        # Always raises a value error saying it does not work. Takes no arguments.
        raise ValueError("This doesn't work")

      v.read_value = broken_read
      text = "%r" % v
      self.assertEqual("<tf.Variable 'Variable:0' shape=() dtype=int32,"
                       " numpy=<unavailable>>", text)

  def testFormatResourceHandle(self):
    # [FUNCTION DOC]
    # Eager mode: asserts that both str and repr of a resource handle contain the resource handle label.
    with context.eager_mode():
      handle = _eager_safe_var_handle_op(
          dtype=dtypes.int32, shape=[1], name="foo")
      self.assertIn("<ResourceHandle", str(handle))
      self.assertIn("<ResourceHandle", repr(handle))

  @test_util.run_in_graph_and_eager_modes
  def testDtypeSurvivesIdentity(self):
    # [FUNCTION DOC]
    # Runs in both modes. Applies identity to an int32 handle and assigns an int32 value through the
    # copy, asserting that the assignment succeeds (the dtype is preserved).
    handle = _eager_safe_var_handle_op(dtype=dtypes.int32, shape=[])
    id_handle = array_ops.identity(handle)
    self.evaluate(resource_variable_ops.assign_variable_op(
        id_handle, constant_op.constant(0, dtype=dtypes.int32)))

  def testUnreadOpName(self):
    # [FUNCTION DOC]
    # Asserts that the name of the operation returned by assign_add differs from the variable's name.
    v = resource_variable_ops.ResourceVariable(1.0)
    self.assertNotEqual(v.name, v.assign_add(1.0).name)

  @test_util.run_in_graph_and_eager_modes
  def testCreateRead(self):
    # [FUNCTION DOC]
    # Runs in both modes. Assigns 1 to an int32 handle, reads it back and asserts the value is 1.
    handle = _eager_safe_var_handle_op(dtype=dtypes.int32, shape=[])
    self.evaluate(resource_variable_ops.assign_variable_op(
        handle, constant_op.constant(1, dtype=dtypes.int32)))
    value = self.evaluate(
        resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32))
    self.assertAllEqual(1, value)

  @test_util.run_in_graph_and_eager_modes
  def testManyAssigns(self):
    # [FUNCTION DOC]
    # Runs in both modes. Chains, with control dependencies: assign 1, read, assign 2, read. Asserts the
    # first read is 1 and the second is 2.
    handle = _eager_safe_var_handle_op(dtype=dtypes.int32, shape=[])
    create = resource_variable_ops.assign_variable_op(
        handle, constant_op.constant(1, dtype=dtypes.int32))
    with ops.control_dependencies([create]):
      first_read = resource_variable_ops.read_variable_op(
          handle, dtype=dtypes.int32)
    with ops.control_dependencies([first_read]):
      write = resource_variable_ops.assign_variable_op(
          handle, constant_op.constant(2, dtype=dtypes.int32))
    with ops.control_dependencies([write]):
      second_read = resource_variable_ops.read_variable_op(
          handle, dtype=dtypes.int32)
    f, s = self.evaluate([first_read, second_read])
    self.assertEqual(f, 1)
    self.assertEqual(s, 2)

  @test_util.run_in_graph_and_eager_modes
  def testAssignAdd(self):
    # [FUNCTION DOC]
    # Runs in both modes. Assigns 1 then adds 1 using the raw ops and asserts the read value is 2.
    handle = _eager_safe_var_handle_op(dtype=dtypes.int32, shape=[])
    self.evaluate(resource_variable_ops.assign_variable_op(
        handle, constant_op.constant(1, dtype=dtypes.int32)))
    self.evaluate(resource_variable_ops.assign_add_variable_op(
        handle, constant_op.constant(1, dtype=dtypes.int32)))
    read = self.evaluate(
        resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32))
    self.assertEqual(read, 2)

  @test_util.run_in_graph_and_eager_modes
  def testScatterAdd(self):
    # [FUNCTION DOC]
    # Runs in both modes. Assigns [[1]] to an int32 handle, scatter-adds [[2]] at index 0 and asserts
    # the value is [[3]].
    handle = _eager_safe_var_handle_op(dtype=dtypes.int32, shape=[1, 1])
    self.evaluate(
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant([[1]], dtype=dtypes.int32)))
    self.evaluate(
        resource_variable_ops.resource_scatter_add(
            handle, [0], constant_op.constant([[2]], dtype=dtypes.int32)))
    read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
    self.assertEqual(self.evaluate(read), [[3]])

  @test_util.run_in_graph_and_eager_modes
  def testGradientGatherNd(self):
    # [FUNCTION DOC]
    # Runs in both modes. Computes the gradient of the summed gather_nd of a 2x2 variable at [1,1] with
    # a gradient tape and asserts it is a matrix of zeros with a 1 in the lower right entry.
    v = resource_variable_ops.ResourceVariable(
        np.random.uniform(size=[2, 2]), dtype=dtypes.float32)

    with backprop.GradientTape() as tape:
      l = array_ops.gather_nd(v, [[1, 1]])
      l = math_ops.reduce_sum(l)

    grads = tape.gradient(l, v)
    self.evaluate(variables.global_variables_initializer())
    self.assertAllEqual(self.evaluate(grads), [[0., 0.], [0., 1.]])

  @test_util.run_deprecated_v1
  def testDefaultGradientDtype(self):
    # [FUNCTION DOC]
    # Graph-mode test: takes the identity of a constant and a float64 variable handle, copies handle
    # data, computes gradients and asserts the handle gradient has dtype float64 and is zeros of shape
    # two by two.
    v = resource_variable_ops.ResourceVariable(
        np.random.uniform(size=[2, 2]), dtype=dtypes.float64)

    c = constant_op.constant(1.)
    identity = array_ops.identity_n([c, v.handle])
    # TODO(b/137403775): Remove this.
    handle_data_util.copy_handle_data(v.handle, identity[1])

    g = gradients_impl.gradients(identity[0], [c, v.handle])
    self.assertEqual(g[1].dtype, dtypes.float64)
    self.evaluate(variables.global_variables_initializer())
    self.assertAllEqual(g[1], [[0., 0.], [0., 0.]])

  @test_util.run_deprecated_v1
  def testUnconnectedGradientZeros(self):
    # [FUNCTION DOC]
    # Graph-mode test: asks for the gradient of an unrelated constant with respect to a variable with
    # the unconnected-gradients setting of zero; asserts the result has shape [1, 2].
    b = resource_variable_ops.ResourceVariable(initial_value=[[3., 4.]])
    c = constant_op.constant(0.)
    g = gradients_impl.gradients(c, [b], unconnected_gradients="zero")[0]
    self.assertAllEqual(g.shape.as_list(), [1, 2])

  @test_util.run_deprecated_v1
  def testGradientCondInWhileLoop(self):
    # [FUNCTION DOC]
    # Graph-mode test: builds a while loop whose body contains a conditional using a variable; computes
    # the gradient of the result with respect to the variable (must not raise) and asserts it is 2.0
    # since only the false branch executes.
    v = resource_variable_ops.ResourceVariable(initial_value=1.0)
    def cond(i, unused_x):
      # [FUNCTION DOC]
      # HELPER: loop condition for testGradientCondInWhileLoop.
      # Takes the counter and an unused value; returns whether the counter is below 1.
      return i < 1

    def body(i, x):
      # [FUNCTION DOC]
      # HELPER: loop body for testGradientCondInWhileLoop.
      # Takes the counter and value; returns the incremented counter and a conditional that adds the
      # variable to the value when the counter is positive, else doubles the variable.
      def true():
        # [FUNCTION DOC]
        # HELPER: true branch of the conditional in the loop body; returns the value plus the variable.
        return x + v
      def false():
        # [FUNCTION DOC]
        # HELPER: false branch of the conditional in the loop body; returns twice the variable.
        return 2.0 * v
      return i + 1, tf_cond.cond(i > 0, true, false)

    _, x = while_loop.while_loop(cond, body, [0, 0.0])
    # Computing gradients does not produce an exception:
    g = gradients_impl.gradients(x, v)
    self.evaluate(variables.global_variables_initializer())
    # Only the false branch is taken so the gradient is 2.
    self.assertAllEqual(g[0], 2.0)

  @test_util.run_in_graph_and_eager_modes
  def testGradientGatherNdIndexedSlices(self):
    # [FUNCTION DOC]
    # Runs in both modes. Gathers rows [1] and [1] from a 2x2 variable, sums, and asserts that the
    # gradient is an indexed-slices object whose values are all ones.
    v = resource_variable_ops.ResourceVariable(
        np.random.uniform(size=[2, 2]), dtype=dtypes.float32)

    with backprop.GradientTape() as tape:
      l = array_ops.gather_nd(v, [[1], [1]])
      l = math_ops.reduce_sum(l)

    grads = tape.gradient(l, v)
    self.evaluate(variables.global_variables_initializer())
    self.assertAllEqual(self.evaluate(grads.values), [[1., 1.], [1., 1.]])

  @test_util.run_in_graph_and_eager_modes
  def testGradientCompositeVariable(self):
    # [FUNCTION DOC]
    # Runs in both modes. Wraps a three element variable in the composite type, takes the tape gradient
    # of the composite with respect to its variable and asserts it is all ones.
    composite_variable = CompositeVariable(
        resource_variable_ops.ResourceVariable([1., 2., 3.]))

    self.evaluate(variables.global_variables_initializer())

    with backprop.GradientTape() as tape:
      result = tape.gradient(composite_variable, composite_variable.v)

    self.assertAllEqual(result, [1., 1., 1.])

  @test_util.run_in_graph_and_eager_modes
  def testScatterSub(self):
    # [FUNCTION DOC]
    # Runs in both modes. Assigns [[1]], scatter-subtracts [[2]] at index 0, asserts [[-1]].
    handle = _eager_safe_var_handle_op(dtype=dtypes.int32, shape=[1, 1])
    self.evaluate(
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant([[1]], dtype=dtypes.int32)))
    self.evaluate(
        resource_variable_ops.resource_scatter_sub(
            handle, [0], constant_op.constant([[2]], dtype=dtypes.int32)))
    read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
    self.assertEqual(self.evaluate(read), [[-1]])

  @test_util.run_in_graph_and_eager_modes
  def testScatterMul(self):
    # [FUNCTION DOC]
    # Runs in both modes. Assigns [[1]], scatter-multiplies by [[5]] at index 0, asserts [[5]].
    handle = _eager_safe_var_handle_op(dtype=dtypes.int32, shape=[1, 1])
    self.evaluate(
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant([[1]], dtype=dtypes.int32)))
    self.evaluate(
        resource_variable_ops.resource_scatter_mul(
            handle, [0], constant_op.constant([[5]], dtype=dtypes.int32)))
    read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
    self.assertEqual(self.evaluate(read), [[5]])

  def testEagerPickle(self):
    # [FUNCTION DOC]
    # Eager mode: pickles a named float16 variable to a file in the temp directory, loads it back and
    # asserts name, shape, dtype, trainable flag and value are the same. Writes a temporary file.
    with context.eager_mode():
      tmp_dir = self.get_temp_dir()
      fname = os.path.join(tmp_dir, "var.pickle")
      with open(fname, "wb") as f:
        v = resource_variable_ops.ResourceVariable(
            10.0,
            dtype=dtypes.float16,
            name="v")
        pickle.dump(v, f)

      with open(fname, "rb") as f:
        new_v = pickle.load(f)
        self.assertEqual(new_v.name, v.name)
        self.assertEqual(new_v.shape, v.shape)
        self.assertEqual(new_v.dtype, v.dtype)
        self.assertEqual(new_v.trainable, v.trainable)
        self.assertAllEqual(new_v.numpy(), v.numpy())

  @test_util.run_in_graph_and_eager_modes
  def testScatterDiv(self):
    # [FUNCTION DOC]
    # Runs in both modes. Assigns [[6]], scatter-divides by [[3]] at index 0, asserts [[2]].
    handle = _eager_safe_var_handle_op(dtype=dtypes.int32, shape=[1, 1])
    self.evaluate(
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant([[6]], dtype=dtypes.int32)))
    self.evaluate(
        resource_variable_ops.resource_scatter_div(
            handle, [0], constant_op.constant([[3]], dtype=dtypes.int32)))
    read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
    self.assertEqual(self.evaluate(read), [[2]])

  def testUseResource(self):
    # [FUNCTION DOC]
    # Asserts that creating a v1 variable with the use-resource flag yields a resource variable instance.
    v = variable_v1.VariableV1(1.0, use_resource=True)
    self.assertIsInstance(v, resource_variable_ops.ResourceVariable)

  def testEagerNoUseResource(self):
    # [FUNCTION DOC]
    # Eager mode: asserts that a plain variable is a resource variable instance.
    with context.eager_mode():
      v = variables.Variable(1.0)
      self.assertIsInstance(v, resource_variable_ops.ResourceVariable)

  @test_util.run_in_graph_and_eager_modes
  def testScatterMin(self):
    # [FUNCTION DOC]
    # Runs in both modes on the CPU. Assigns [[6]], scatter-minimum with [[3]] at index 0, asserts [[3]].
    with ops.device("cpu:0"):
      handle = _eager_safe_var_handle_op(dtype=dtypes.int32, shape=[1, 1])
      self.evaluate(
          resource_variable_ops.assign_variable_op(handle,
                                                   constant_op.constant(
                                                       [[6]],
                                                       dtype=dtypes.int32)))
      self.evaluate(
          resource_variable_ops.resource_scatter_min(handle, [0],
                                                     constant_op.constant(
                                                         [[3]],
                                                         dtype=dtypes.int32)))
      read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
      self.assertEqual(self.evaluate(read), [[3]])

  def testMetagraph(self):
    # [FUNCTION DOC]
    # Creates a resource variable and a momentum optimizer minimize step with a global step in a graph,
    # exports a meta graph, imports it into a new graph and exports again; asserts both meta graph
    # definitions are equal.
    with ops.Graph().as_default():
      with variable_scope.variable_scope("foo", use_resource=True):
        a = variable_scope.get_variable("a", initializer=10.0)

      momentum.MomentumOptimizer(
          learning_rate=0.001, momentum=0.1).minimize(
              a,
              colocate_gradients_with_ops=True,
              global_step=training_util.get_or_create_global_step())

      graph = ops.get_default_graph()
      meta_graph_def = saver.export_meta_graph(graph=graph)

    with ops.Graph().as_default():
      saver.import_meta_graph(meta_graph_def, import_scope="")
      meta_graph_two = saver.export_meta_graph(graph=graph)
    self.assertEqual(meta_graph_def, meta_graph_two)

  @test_util.run_in_graph_and_eager_modes
  def testScatterMax(self):
    # [FUNCTION DOC]
    # Runs in both modes. Assigns [[6]], scatter-maximum with [[3]] at index 0, asserts [[6]].
    handle = _eager_safe_var_handle_op(dtype=dtypes.int32, shape=[1, 1])
    self.evaluate(
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant([[6]], dtype=dtypes.int32)))
    self.evaluate(
        resource_variable_ops.resource_scatter_max(
            handle, [0], constant_op.constant([[3]], dtype=dtypes.int32)))
    read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
    self.assertEqual(self.evaluate(read), [[6]])

  @test_util.run_in_graph_and_eager_modes
  def testScatterAddScalar(self):
    # [FUNCTION DOC]
    # Runs in both modes. Like the scatter add test but the update is a scalar 2; asserts [[3]].
    handle = _eager_safe_var_handle_op(dtype=dtypes.int32, shape=[1, 1])
    self.evaluate(
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant([[1]], dtype=dtypes.int32)))
    self.evaluate(
        resource_variable_ops.resource_scatter_add(
            handle, [0], constant_op.constant(2, dtype=dtypes.int32)))
    read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
    self.assertEqual(self.evaluate(read), [[3]])

  @test_util.run_in_graph_and_eager_modes
  def testScatterSubScalar(self):
    # [FUNCTION DOC]
    # Runs in both modes. Scatter-subtract with scalar update 2 from [[1]]; asserts [[-1]].
    handle = _eager_safe_var_handle_op(dtype=dtypes.int32, shape=[1, 1])
    self.evaluate(
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant([[1]], dtype=dtypes.int32)))
    self.evaluate(
        resource_variable_ops.resource_scatter_sub(
            handle, [0], constant_op.constant(2, dtype=dtypes.int32)))
    read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
    self.assertEqual(self.evaluate(read), [[-1]])

  @test_util.run_in_graph_and_eager_modes
  def testScatterMulScalar(self):
    # [FUNCTION DOC]
    # Runs in both modes. Scatter-multiply [[1]] by scalar 5; asserts [[5]].
    handle = _eager_safe_var_handle_op(dtype=dtypes.int32, shape=[1, 1])
    self.evaluate(
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant([[1]], dtype=dtypes.int32)))
    self.evaluate(
        resource_variable_ops.resource_scatter_mul(
            handle, [0], constant_op.constant(5, dtype=dtypes.int32)))
    read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
    self.assertEqual(self.evaluate(read), [[5]])

  @test_util.run_in_graph_and_eager_modes
  def testScatterDivScalar(self):
    # [FUNCTION DOC]
    # Runs in both modes. Scatter-divide [[6]] by scalar 3; asserts [[2]].
    handle = _eager_safe_var_handle_op(dtype=dtypes.int32, shape=[1, 1])
    self.evaluate(
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant([[6]], dtype=dtypes.int32)))
    self.evaluate(
        resource_variable_ops.resource_scatter_div(
            handle, [0], constant_op.constant(3, dtype=dtypes.int32)))
    read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
    self.assertEqual(self.evaluate(read), [[2]])

  @test_util.run_in_graph_and_eager_modes
  def testScatterMinScalar(self):
    # [FUNCTION DOC]
    # Runs in both modes. Scatter-minimum of [[6]] with scalar 3; asserts [[3]].
    handle = _eager_safe_var_handle_op(dtype=dtypes.int32, shape=[1, 1])
    self.evaluate(
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant([[6]], dtype=dtypes.int32)))
    self.evaluate(
        resource_variable_ops.resource_scatter_min(
            handle, [0], constant_op.constant(3, dtype=dtypes.int32)))
    read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
    self.assertEqual(self.evaluate(read), [[3]])

  @test_util.run_in_graph_and_eager_modes
  def testScatterMaxScalar(self):
    # [FUNCTION DOC]
    # Runs in both modes. Scatter-maximum of [[6]] with scalar 3; asserts [[6]].
    handle = _eager_safe_var_handle_op(dtype=dtypes.int32, shape=[1, 1])
    self.evaluate(
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant([[6]], dtype=dtypes.int32)))
    self.evaluate(
        resource_variable_ops.resource_scatter_max(
            handle, [0], constant_op.constant(3, dtype=dtypes.int32)))
    read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
    self.assertEqual(self.evaluate(read), [[6]])

  @parameterized.parameters(dtypes.float16, dtypes.float32, dtypes.float64,
                            dtypes.bfloat16)
  @test_util.run_in_graph_and_eager_modes
  def testScatterAddVariableMethod(self, dtype):
    # [FUNCTION DOC]
    # Parameterized over four float dtypes. Calls scatter_add on a variable [0.0, 1.5] adding 2.5 at
    # index 1 using indexed slices; asserts the result is [0.0, 4.0] within dtype tolerance.
    v = resource_variable_ops.ResourceVariable([0.0, 1.5],
                                               name="add",
                                               dtype=dtype)
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(
        v.scatter_add(
            indexed_slices.IndexedSlices(
                indices=[1], values=constant_op.constant([2.5], dtype=dtype))))
    self.assertAllCloseAccordingToType([0.0, 4.0], self.evaluate(v))

  @parameterized.parameters(dtypes.float16, dtypes.float32, dtypes.float64,
                            dtypes.bfloat16)
  @test_util.run_in_graph_and_eager_modes
  def testScatterSubVariableMethod(self, dtype):
    # [FUNCTION DOC]
    # Parameterized over four float dtypes. Subtracts 1.5 at index 1 from [0.0, 2.5]; asserts [0.0, 1.0].
    v = resource_variable_ops.ResourceVariable([0.0, 2.5],
                                               name="sub",
                                               dtype=dtype)
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(
        v.scatter_sub(
            indexed_slices.IndexedSlices(
                indices=[1], values=constant_op.constant([1.5], dtype=dtype))))
    self.assertAllCloseAccordingToType([0.0, 1.0], self.evaluate(v))

  @parameterized.parameters(dtypes.float16, dtypes.float32, dtypes.float64,
                            dtypes.bfloat16)
  @test_util.run_in_graph_and_eager_modes
  def testScatterMaxVariableMethod(self, dtype):
    # [FUNCTION DOC]
    # Parameterized over four float dtypes. Scatter-max of 5.0 into [0.0, 4.0] gives [0.0, 5.0]; then
    # 2.0 into [0.0, 3.5] leaves [0.0, 3.5]. Asserts both within dtype tolerance.
    v = resource_variable_ops.ResourceVariable([0.0, 4.0],
                                               name="max1",
                                               dtype=dtype)
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(
        v.scatter_max(
            indexed_slices.IndexedSlices(
                indices=[1], values=constant_op.constant([5.0], dtype=dtype))))
    self.assertAllCloseAccordingToType([0.0, 5.0], self.evaluate(v))

    v = resource_variable_ops.ResourceVariable([0.0, 3.5],
                                               name="max2",
                                               dtype=dtype)
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(
        v.scatter_max(
            indexed_slices.IndexedSlices(
                indices=[1], values=constant_op.constant([2.0], dtype=dtype))))
    self.assertAllCloseAccordingToType([0.0, 3.5], self.evaluate(v))

  @parameterized.parameters(dtypes.float16, dtypes.float32, dtypes.float64,
                            dtypes.bfloat16)
  @test_util.run_in_graph_and_eager_modes
  def testScatterMinVariableMethod(self, dtype):
    # [FUNCTION DOC]
    # Parameterized over four float dtypes. Scatter-min of 5.0 into [0.0, 4.0] leaves [0.0, 4.0]; then
    # 2.0 into [0.0, 3.5] gives [0.0, 2.0]. Asserts both within dtype tolerance.
    v = resource_variable_ops.ResourceVariable([0.0, 4.0],
                                               name="min1",
                                               dtype=dtype)
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(
        v.scatter_min(
            indexed_slices.IndexedSlices(
                indices=[1], values=constant_op.constant([5.0], dtype=dtype))))
    self.assertAllCloseAccordingToType([0.0, 4.0], self.evaluate(v))

    v = resource_variable_ops.ResourceVariable([0.0, 3.5],
                                               name="min2",
                                               dtype=dtype)
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(
        v.scatter_min(
            indexed_slices.IndexedSlices(
                indices=[1], values=constant_op.constant([2.0], dtype=dtype))))
    self.assertAllCloseAccordingToType([0.0, 2.0], self.evaluate(v))

  @parameterized.parameters(dtypes.float16, dtypes.float32, dtypes.float64,
                            dtypes.bfloat16)
  @test_util.run_in_graph_and_eager_modes
  def testScatterMulVariableMethod(self, dtype):
    # [FUNCTION DOC]
    # Parameterized over four float dtypes. Multiplies index 1 of [0.0, 4.0] by 3.0; asserts [0.0, 12.0].
    v = resource_variable_ops.ResourceVariable([0.0, 4.0],
                                               name="mul",
                                               dtype=dtype)
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(
        v.scatter_mul(
            indexed_slices.IndexedSlices(
                indices=[1], values=constant_op.constant([3.0], dtype=dtype))))
    self.assertAllCloseAccordingToType([0.0, 12.0], self.evaluate(v))

  @parameterized.parameters(dtypes.float16, dtypes.float32, dtypes.float64,
                            dtypes.bfloat16)
  @test_util.run_in_graph_and_eager_modes
  def testScatterDivVariableMethod(self, dtype):
    # [FUNCTION DOC]
    # Parameterized over four float dtypes. Divides index 1 of [0.0, 6.0] by 2.0; asserts [0.0, 3.0].
    v = resource_variable_ops.ResourceVariable([0.0, 6.0],
                                               name="div",
                                               dtype=dtype)
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(
        v.scatter_div(
            indexed_slices.IndexedSlices(
                indices=[1], values=constant_op.constant([2.0], dtype=dtype))))
    self.assertAllCloseAccordingToType([0.0, 3.0], self.evaluate(v))

  @parameterized.parameters(dtypes.float16, dtypes.float32, dtypes.float64,
                            dtypes.bfloat16)
  @test_util.run_in_graph_and_eager_modes
  def testScatterUpdateVariableMethod(self, dtype):
    # [FUNCTION DOC]
    # Parameterized over four float dtypes. Sets index 1 of [0.0, 6.0] to 3.0; asserts [0.0, 3.0].
    v = resource_variable_ops.ResourceVariable([0.0, 6.0],
                                               name="update",
                                               dtype=dtype)
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(
        v.scatter_update(
            indexed_slices.IndexedSlices(
                indices=[1], values=constant_op.constant([3.0], dtype=dtype))))
    self.assertAllCloseAccordingToType([0.0, 3.0], self.evaluate(v))

  @test_util.run_deprecated_v1
  def testScatterUpdateString(self):
    # [FUNCTION DOC]
    # Graph-mode test: scatter-updates a string variable holding "a" with "b" via raw ops and asserts the
    # bytes read back equal "b".
    handle = _eager_safe_var_handle_op(dtype=dtypes.string, shape=[1, 1])
    self.evaluate(resource_variable_ops.assign_variable_op(
        handle, constant_op.constant([["a"]], dtype=dtypes.string)))
    self.evaluate(resource_variable_ops.resource_scatter_update(
        handle, [0], constant_op.constant([["b"]], dtype=dtypes.string)))
    read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.string)
    self.assertEqual(compat.as_bytes(self.evaluate(read)[0][0]),
                     compat.as_bytes("b"))

  @test_util.run_deprecated_v1
  def testScatterUpdateStringScalar(self):
    # [FUNCTION DOC]
    # Graph-mode test: same as the string scatter update test but the update is a scalar string.
    handle = _eager_safe_var_handle_op(dtype=dtypes.string, shape=[1, 1])
    self.evaluate(
        resource_variable_ops.assign_variable_op(handle,
                                                 constant_op.constant(
                                                     [["a"]],
                                                     dtype=dtypes.string)))
    self.evaluate(
        resource_variable_ops.resource_scatter_update(handle, [0],
                                                      constant_op.constant(
                                                          "b",
                                                          dtype=dtypes.string)))
    read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.string)
    self.assertEqual(
        compat.as_bytes(self.evaluate(read)[0][0]), compat.as_bytes("b"))

  # TODO(alive): get this to work in Eager mode.
  def testGPU(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: despite the name, it only prefers a GPU device if one exists and falls back to CPU;
    # the to-do notes it does not yet work in eager mode.
    # Creates a resource variable through the variable scope on the preferred device, initializes it and
    # asserts the variable-is-initialized op reports true.
    with test_util.use_gpu():
      abc = variable_scope.get_variable(
          "abc",
          shape=[1],
          initializer=init_ops.ones_initializer(),
          use_resource=True)

      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(
          self.evaluate(
              resource_variable_ops.var_is_initialized_op(abc.handle)),
          True)

  def testScatterBool(self):
    # [FUNCTION DOC]
    # Eager mode: scatter-updates a boolean variable at all indices with true using the state ops helper
    # and asserts all values read true.
    with context.eager_mode():
      ref = resource_variable_ops.ResourceVariable(
          [False, True, False], trainable=False)
      indices = math_ops.range(3)
      updates = constant_op.constant([True, True, True])
      state_ops.scatter_update(ref, indices, updates)
      self.assertAllEqual(ref.read_value(), [True, True, True])

  @test_util.run_in_graph_and_eager_modes
  def testConstraintArg(self):
    # [FUNCTION DOC]
    # Runs in both modes. Creates a variable with a callable constraint and asserts it is stored; then
    # asserts a non-callable constraint (the number 0) raises a value error.
    constraint = lambda x: x
    v = resource_variable_ops.ResourceVariable(
        initial_value=lambda: 1, constraint=constraint, name="var0")
    self.assertEqual(v.constraint, constraint)

    constraint = 0
    with self.assertRaises(ValueError):
      v = resource_variable_ops.ResourceVariable(
          initial_value=lambda: 1, constraint=constraint, name="var1")

  # TODO(alive): how should this work in Eager mode?
  @test_util.run_deprecated_v1
  def testInitFn(self):
    # [FUNCTION DOC]
    # Graph-mode test: creates a variable from an initial value function and asserts the handle and the
    # initializer input are in the same colocation groups.
    with self.cached_session():
      v = resource_variable_ops.ResourceVariable(
          initial_value=lambda: 1, dtype=dtypes.float32)
      self.assertEqual(v.handle.op.colocation_groups(),
                       v.initializer.inputs[1].op.colocation_groups())

  def testCountUpTo(self):
    # [FUNCTION DOC]
    # Eager mode: count_up_to with limit 1 returns 0 first, then raises an out-of-range error.
    with context.eager_mode():
      v = resource_variable_ops.ResourceVariable(0, name="upto")
      self.assertAllEqual(v.count_up_to(1), 0)
      with self.assertRaises(errors.OutOfRangeError):
        v.count_up_to(1)

  def testCountUpToFunction(self):
    # [FUNCTION DOC]
    # Eager mode: same behavior as the previous test but through the state ops count_up_to function.
    with context.eager_mode():
      v = resource_variable_ops.ResourceVariable(0, name="upto")
      self.assertAllEqual(state_ops.count_up_to(v, 1), 0)
      with self.assertRaises(errors.OutOfRangeError):
        state_ops.count_up_to(v, 1)

  @test_util.run_in_graph_and_eager_modes
  def testInitFnDtype(self):
    # [FUNCTION DOC]
    # Runs in both modes. Variable from an initial value function with explicit float32 dtype; asserts
    # the value dtype is float32.
    v = resource_variable_ops.ResourceVariable(
        initial_value=lambda: 1, dtype=dtypes.float32, name="var0")
    self.assertEqual(dtypes.float32, v.value().dtype)

  @test_util.run_in_graph_and_eager_modes
  def testInitFnNoDtype(self):
    # [FUNCTION DOC]
    # Runs in both modes. Variable from an initial value function without dtype; asserts the inferred
    # dtype is int32.
    v = resource_variable_ops.ResourceVariable(initial_value=lambda: 1,
                                               name="var2")
    self.assertEqual(dtypes.int32, v.value().dtype)

  @test_util.run_in_graph_and_eager_modes
  def testInitializeAllVariables(self):
    # [FUNCTION DOC]
    # Runs in both modes. Creates an integer 1 variable cast to float32, runs global initializer and
    # asserts the value is 1.0.
    v = resource_variable_ops.ResourceVariable(1, dtype=dtypes.float32,
                                               name="var0")
    self.evaluate(variables.global_variables_initializer())
    self.assertEqual(1.0, self.evaluate(v.value()))

  @test_util.run_in_graph_and_eager_modes
  def testOperatorOverload(self):
    # [FUNCTION DOC]
    # Runs in both modes. Asserts that adding a variable to itself evaluates to 2.0.
    v = resource_variable_ops.ResourceVariable(1.0, name="var0")
    self.evaluate(variables.global_variables_initializer())
    self.assertEqual(2.0, self.evaluate(v + v))

  @test_util.run_in_graph_and_eager_modes
  def testAssignMethod(self):
    # [FUNCTION DOC]
    # Runs in both modes. Assign 2.0 and check; then assign with read_value true returns the new value
    # 3.0; assign with read_value false returns nothing in eager mode and an operation in graph mode;
    # after running it the value is 4.0.
    v = resource_variable_ops.ResourceVariable(1.0, name="var0")
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(v.assign(2.0))
    self.assertEqual(2.0, self.evaluate(v.value()))

    # Tests for the 'read_value' argument:
    assign_with_read = v.assign(3.0, read_value=True)
    self.assertEqual(3.0, self.evaluate(assign_with_read))
    assign_without_read = v.assign(4.0, read_value=False)
    if context.executing_eagerly():
      self.assertIsNone(assign_without_read)
    else:
      self.assertIsInstance(assign_without_read, ops.Operation)
    self.evaluate(assign_without_read)
    self.assertEqual(4.0, self.evaluate(v.value()))

  def testAssignRuntimeShapeCheck(self):
    # [FUNCTION DOC]
    # Under a forward compatibility horizon date, a traced function assigns a tensor of dynamic shape to
    # a two element variable. Asserts calling with shape [3] raises an invalid-argument or value error.
    with forward_compat.forward_compatibility_horizon(2022, 3, 30):
      v = resource_variable_ops.ResourceVariable([1.0, 1.0], name="var0")

      @def_function.function
      def f(shape):
        # [FUNCTION DOC]
        # HELPER: traced function body for testAssignRuntimeShapeCheck.
        # Takes a shape tensor, creates zeros of that shape and assigns them to the enclosing variable.
        t = array_ops.zeros(shape)
        v.assign(t)

      with self.assertRaises((errors.InvalidArgumentError, ValueError)):
        f(constant_op.constant([3]))

  @test_util.run_in_graph_and_eager_modes
  def testLoad(self):
    # [FUNCTION DOC]
    # Runs in both modes. Loads 2.0 into a variable and asserts it reads 2.0.
    v = resource_variable_ops.ResourceVariable(1.0, name="var0")
    self.evaluate(variables.global_variables_initializer())
    v.load(2.0)
    self.assertEqual(2.0, self.evaluate(v.value()))

  def testShapePassedToGradient(self):
    # [FUNCTION DOC]
    # Graph-mode: defines a custom-gradient scatter update on a resource variable handle and gathers
    # from it; computes gradients with respect to the updates. The gradient function asserts the incoming
    # gradient has a statically known dense shape.
    with ops.Graph().as_default():
      @custom_gradient.custom_gradient
      def differentiable_scatter_update(handle, indices, values):
        # [FUNCTION DOC]
        # HELPER: custom-gradient function for testShapePassedToGradient.
        # Scatter-updates the handle under control dependencies and returns an identity of the handle plus a
        # gradient function. Inputs: handle, indices, values.
        with ops.control_dependencies([
            resource_variable_ops.resource_scatter_update(
                handle, indices, values)]):
          new_handle = array_ops.identity(handle)

        def grad(dresult):
          # [FUNCTION DOC]
          # HELPER: gradient function of differentiable_scatter_update.
          # Asserts that the incoming gradient has a constant dense shape and returns it for the handle with
          # nothing for indices and values.
          self.assertIsNotNone(
              tensor_util.constant_value(dresult.dense_shape))
          return [dresult, None, None]

        return new_handle, grad

      var = variable_scope.get_variable(
          "foo", shape=[20], initializer=init_ops.zeros_initializer,
          dtype=dtypes.float64, use_resource=True)

      indices = math_ops.range(10)
      updates = math_ops.range(9, -1, -1, dtype=dtypes.float64)
      new_handle = differentiable_scatter_update(var.handle, indices, updates)
      gathered = resource_variable_ops.resource_gather(
          new_handle, indices, dtype=var.dtype)
      gradients_impl.gradients([gathered], [updates])

  def testCustomGradientVariableOutput(self):
    # [FUNCTION DOC]
    # Eager mode: a custom-gradient function adds 3 to its variable input and returns it. Asserts the
    # output is 5, the gradient is a tensor, and it equals 2.
    with context.eager_mode():
      @custom_gradient.custom_gradient
      def test_func(x):
        # [FUNCTION DOC]
        # HELPER: custom-gradient function under test.
        # Adds 3 to the variable x in place and returns x with a gradient function. Mutates its argument.
        x.assign_add(3.)

        def gradient_func(*grad):
          # [FUNCTION DOC]
          # HELPER: gradient function for test_func; returns twice the first incoming gradient.
          return 2. * grad[0]

        return x, gradient_func

      v = resource_variable_ops.ResourceVariable(2.)
      with backprop.GradientTape() as tape:
        out = test_func(v)
        result = tape.gradient(out, v)

      self.assertAllEqual(out, 5.)
      self.assertIsInstance(result, tensor_lib.Tensor)
      self.assertAllEqual(result, 2.)

  def testToFromProtoCachedValue(self):
    # [FUNCTION DOC]
    # Graph mode: builds a variable, converts it to a definition proto and restores it; asserts the
    # restored variable has no cached value. Repeats with a caching device and asserts a cached value
    # exists after restore.
    with ops.Graph().as_default():
      v_def = resource_variable_ops.ResourceVariable(
          initial_value=constant_op.constant(3.0)).to_proto()
      v_prime = resource_variable_ops.ResourceVariable(variable_def=v_def)
      self.assertIsNone(getattr(v_prime, "_cached_value", None))

      other_v_def = resource_variable_ops.ResourceVariable(
          caching_device="cpu:0",
          initial_value=constant_op.constant(3.0)).to_proto()
      other_v_prime = resource_variable_ops.ResourceVariable(
          variable_def=other_v_def)
      self.assertIsNotNone(other_v_prime._cached_value)

  def testVariableDefInitializedInstances(self):
    # [FUNCTION DOC]
    # Graph mode: restores variables from a variable definition proto in new graphs. Asserts the initial
    # value is restored (3.0), read_value after assign sees 1.0, and for a legacy proto without an
    # initial value name asserts re-export equals the proto and conditional initial reads raise a value
    # error.
    with ops.Graph().as_default(), self.cached_session():
      v_def = resource_variable_ops.ResourceVariable(
          initial_value=constant_op.constant(3.0)).to_proto()

    with ops.Graph().as_default(), self.cached_session():
      # v describes a VariableDef-based variable without an initial value.
      v = resource_variable_ops.ResourceVariable(variable_def=v_def)
      self.assertEqual(3.0, self.evaluate(v.initial_value))

      # read_value should not rerun the initializer_op if the variable
      # has already been initialized elsewhere.
      self.evaluate(v.assign(1.0))
      self.assertEqual(1.0, v.read_value().eval())

    v_def.ClearField("initial_value_name")
    with ops.Graph().as_default(), self.cached_session():
      # Restoring a legacy VariableDef proto that does not have
      # initial_value_name set should still work.
      v = resource_variable_ops.ResourceVariable(variable_def=v_def)
      # We should also be able to re-export the variable to a new meta graph.
      self.assertProtoEquals(v_def, v.to_proto())
      # But attempts to use read_value will result in errors.
      with self.assertRaises(ValueError):
        self.evaluate(
            tf_cond.cond(
                variable_v1.is_variable_initialized(v), v.read_value,
                lambda: v.initial_value))

  def testTrainableInProto(self):
    # [FUNCTION DOC]
    # Graph mode: asserts that the trainable flag (false and true) survives conversion to a variable
    # definition proto and restoration.
    with ops.Graph().as_default():
      non_trainable_variable = resource_variable_ops.ResourceVariable(
          trainable=False,
          initial_value=constant_op.constant(10.0))
      self.assertEqual(
          False,
          resource_variable_ops.ResourceVariable(
              variable_def=non_trainable_variable.to_proto())
          .trainable)
      trainable_variable = resource_variable_ops.ResourceVariable(
          trainable=True,
          initial_value=constant_op.constant(10.0))
      self.assertEqual(
          True,
          resource_variable_ops.ResourceVariable(
              variable_def=trainable_variable.to_proto())
          .trainable)

  @test_util.run_in_graph_and_eager_modes
  def testSparseRead(self):
    # [FUNCTION DOC]
    # Runs in both modes. Sparse reads rows [0,3,1,2] of a 4x4x4 int32 variable and asserts they equal
    # the matching rows of the source array.
    init_value = np.reshape(np.arange(np.power(4, 3)), (4, 4, 4))
    v = resource_variable_ops.ResourceVariable(
        constant_op.constant(init_value, dtype=dtypes.int32), name="var3")
    self.evaluate(variables.global_variables_initializer())

    value = self.evaluate(v.sparse_read([0, 3, 1, 2]))
    self.assertAllEqual(init_value[[0, 3, 1, 2], ...], value)

  @test_util.run_in_graph_and_eager_modes
  def testGatherNd(self):
    # [FUNCTION DOC]
    # Runs in both modes. gather_nd on a 4x4x4 variable with index depth 2 gives shape [3, 4] with the
    # expected rows, and with index depth 3 gives shape [3] with values 0, 27 and 63.
    init_value = np.reshape(np.arange(np.power(4, 3)), (4, 4, 4))
    v = resource_variable_ops.ResourceVariable(
        constant_op.constant(init_value, dtype=dtypes.int32), name="var3")
    self.evaluate(variables.global_variables_initializer())

    value_op = v.gather_nd([[0, 0], [1, 2], [3, 3]])
    self.assertAllEqual([3, 4], value_op.shape)
    value = self.evaluate(value_op)
    self.assertAllEqual([[0, 1, 2, 3], [24, 25, 26, 27], [60, 61, 62, 63]],
                        value)

    value_op = v.gather_nd([[0, 0, 0], [1, 2, 3], [3, 3, 3]])
    self.assertAllEqual([3], value_op.shape)
    value = self.evaluate(value_op)
    self.assertAllEqual([0, 27, 63], value)

  @test_util.run_deprecated_v1
  def testToFromProto(self):
    # [FUNCTION DOC]
    # Graph-mode: converts a variable to a proto and back; asserts the new variable plus 1 evaluates to 2
    # and the handle and graph element are equal to the original's.
    with self.cached_session():
      v = resource_variable_ops.ResourceVariable(1.0)
      self.evaluate(variables.global_variables_initializer())

      w = resource_variable_ops.ResourceVariable.from_proto(v.to_proto())
      self.assertEqual(2, math_ops.add(w, 1).eval())

      self.assertEqual(v._handle, w._handle)
      self.assertEqual(v._graph_element, w._graph_element)

  @test_util.run_in_graph_and_eager_modes
  def testAssignAddMethod(self):
    # [FUNCTION DOC]
    # Runs in both modes. assign_add of 1.0 on 1.0 gives 2.0; with read_value true returns 3.0; with
    # read_value false returns nothing eagerly and an operation in graph mode; final value 4.0.
    v = resource_variable_ops.ResourceVariable(1.0, name="var0")
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(v.assign_add(1.0))
    self.assertEqual(2.0, self.evaluate(v.value()))

    # Tests for the 'read_value' argument:
    assign_with_read = v.assign_add(1.0, read_value=True)
    self.assertEqual(3.0, self.evaluate(assign_with_read))
    assign_without_read = v.assign_add(1.0, read_value=False)
    if context.executing_eagerly():
      self.assertIsNone(assign_without_read)
    else:
      self.assertIsInstance(assign_without_read, ops.Operation)
    self.evaluate(assign_without_read)
    self.assertEqual(4.0, self.evaluate(v.value()))

  @test_util.run_in_graph_and_eager_modes
  def testAssignSubMethod(self):
    # [FUNCTION DOC]
    # Runs in both modes. assign_sub of 1.0 from 3.0 gives 2.0; with read_value true gives 1.0; with
    # read_value false returns nothing eagerly and an operation in graph mode; final value 0.0.
    v = resource_variable_ops.ResourceVariable(3.0, name="var0")
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(v.assign_sub(1.0))
    self.assertEqual(2.0, self.evaluate(v.value()))

    # Tests for the 'read_value' argument:
    assign_with_read = v.assign_sub(1.0, read_value=True)
    self.assertEqual(1.0, self.evaluate(assign_with_read))
    assign_without_read = v.assign_sub(1.0, read_value=False)
    if context.executing_eagerly():
      self.assertIsNone(assign_without_read)
    else:
      self.assertIsInstance(assign_without_read, ops.Operation)
    self.evaluate(assign_without_read)
    self.assertEqual(0.0, self.evaluate(v.value()))

  @test_util.run_in_graph_and_eager_modes
  @test_util.run_v1_only("b/120545219")
  def testDestroyResource(self):
    # [FUNCTION DOC]
    # Runs in both modes, only in v1 mode per the decorator. Destroys the variable resource; in eager
    # mode the value remains 3.0, in graph mode reading raises a failed-precondition error. Destroying a
    # handle that never existed with ignore lookup error set raises nothing.
    v = resource_variable_ops.ResourceVariable(3.0, name="var0")
    self.evaluate(variables.global_variables_initializer())
    self.assertEqual(3.0, self.evaluate(v.value()))
    self.evaluate(resource_variable_ops.destroy_resource_op(v.handle))
    if context.executing_eagerly():
      # eager mode creates ref-counting variable handles unaffected by
      # DestroyResourceOp.
      self.assertEqual(3.0, self.evaluate(v.value()))
    else:
      with self.assertRaises(errors.FailedPreconditionError):
        self.evaluate(v.value())
    # Handle to a resource not actually created.
    handle = _eager_safe_var_handle_op(dtype=dtypes.int32, shape=[])
    # Should raise no exception
    self.evaluate(resource_variable_ops.destroy_resource_op(
        handle, ignore_lookup_error=True))

  @test_util.run_deprecated_v1
  def testAssignDifferentShapes(self):
    # [FUNCTION DOC]
    # Graph-mode: a variable of shape [1,1] is assigned through a placeholder fed with a 2x2 array; the
    # run is expected to succeed (graph mode does not check the value shape at run time).
    with self.cached_session() as sess, variable_scope.variable_scope(
        "foo", use_resource=True):
      var = variable_scope.get_variable("x", shape=[1, 1], dtype=dtypes.float32)
      placeholder = array_ops.placeholder(dtypes.float32)
      assign = var.assign(placeholder)
      sess.run(
          [assign],
          feed_dict={placeholder: np.zeros(shape=[2, 2], dtype=np.float32)})

  def testAssignDifferentShapesEagerNotAllowed(self):
    # [FUNCTION DOC]
    # Eager mode: assigning a 2x2 array to a [1,1] variable raises a value error saying the shapes are
    # incompatible.
    with context.eager_mode():
      with variable_scope.variable_scope("foo"):
        var = variable_scope.get_variable("x", shape=[1, 1],
                                          dtype=dtypes.float32)
        with self.assertRaisesRegex(ValueError,
                                    "shape.*and.*are incompatible"):
          assign = var.assign(np.zeros(shape=[2, 2]))
          self.evaluate(assign)

  @test_util.disable_xla("XLA doesn't allow changing shape at assignment, as "
                         "dictated by tf2xla/xla_resource.cc:SetTypeAndShape")
  @test_util.run_in_graph_and_eager_modes
  def testAssignDifferentShapesAllowed(self):
    # [FUNCTION DOC]
    # Runs in both modes (XLA disabled). A variable declared with unknown shape accepts a 2x2 assignment
    # after being created as [1,1]; asserts values before and after.
    var = resource_variable_ops.ResourceVariable(
        initial_value=np.zeros(shape=[1, 1]),
        shape=tensor_shape.TensorShape(None))
    self.evaluate(variables.global_variables_initializer())
    self.assertAllEqual(np.zeros(shape=[1, 1]), var.read_value())
    self.evaluate(var.assign(np.zeros(shape=[2, 2])))
    self.assertAllEqual(np.zeros(shape=[2, 2]), var.read_value())

  @test_util.run_in_graph_and_eager_modes
  def testAssignReturnsVariable(self):
    # [FUNCTION DOC]
    # Runs in both modes. Asserts assign, assign_add, assign_sub and scatter operations (add, sub, max,
    # min, mul, div, nd update/add/sub, batch scatter update) return the variable so calls can be chained,
    # and checks the resulting values after chains.
    var = resource_variable_ops.ResourceVariable(1.)
    self.evaluate(variables.global_variables_initializer())
    assigned = var.assign(2.)
    self.assertIsInstance(assigned, resource_variable_ops.BaseResourceVariable)
    assigned = assigned.assign(3.)
    self.assertEqual(self.evaluate(assigned), 3.)
    self.assertEqual(self.evaluate(var), 3.)

    self.assertEqual(self.evaluate(var.assign_add(1.).assign_add(1.)), 5)
    self.assertEqual(self.evaluate(var.assign_sub(1.).assign_sub(1.)), 3)

    var = resource_variable_ops.ResourceVariable([1., 2.])
    self.evaluate(variables.global_variables_initializer())
    slices = indexed_slices.IndexedSlices(indices=[1], values=[2])
    def assert_eq(tensor, vals):
      # [FUNCTION DOC]
      # HELPER: evaluates a tensor or variable and asserts it equals the given values.
      self.assertAllEqual(self.evaluate(tensor), vals)
    assert_eq(var.scatter_add(slices).scatter_add(slices), [1., 6.])
    assert_eq(var.scatter_sub(slices).scatter_sub(slices), [1., 2.])
    slices2 = indexed_slices.IndexedSlices(indices=[0], values=[3])
    assert_eq(var.scatter_max(slices2).scatter_add(slices), [3., 4.])
    assert_eq(var.scatter_add(slices).scatter_min(slices), [3., 2.])
    assert_eq(var.scatter_mul(slices).scatter_mul(slices), [3., 8.])
    assert_eq(var.scatter_div(slices).scatter_div(slices), [3., 2.])
    assert_eq(
        var.scatter_nd_update([[1]], [4.]).scatter_nd_add([[0]], [2.])
        .scatter_nd_sub([[1]], [3]),
        [5., 1.])
    assert_eq(var, [5., 1.])

    batch_var = resource_variable_ops.ResourceVariable(array_ops.ones((2, 2)))
    self.evaluate(variables.global_variables_initializer())
    batch_slices1 = indexed_slices.IndexedSlices(
        indices=[[1], [0]], values=[[2], [2]])
    batch_slices2 = indexed_slices.IndexedSlices(
        indices=[[1], [1]], values=[[3], [3]])
    assert_eq(
        batch_var.batch_scatter_update(batch_slices1)
        .batch_scatter_update(batch_slices2),
        [[1, 3], [2, 3]])

  @test_util.run_in_graph_and_eager_modes
  def testInitValueWrongShape(self):
    # [FUNCTION DOC]
    # Runs in both modes. Asserts that an initial value of shape [3] with declared shape [4] raises a
    # value error mentioning incompatibility.
    with self.assertRaisesWithPredicateMatch(
        ValueError, r"not compatible with"):
      var = resource_variable_ops.ResourceVariable(
          initial_value=np.zeros(shape=[3]),
          shape=[4])
      self.evaluate(variables.global_variables_initializer())
      self.evaluate(var.read_value())

  @test_util.run_deprecated_v1
  def testDtypeAfterFromProto(self):
    # [FUNCTION DOC]
    # Graph-mode: restores a variable from its proto; asserts the dtype is a data type object equal to
    # the original.
    v = resource_variable_ops.ResourceVariable(2.0)
    w = resource_variable_ops.ResourceVariable.from_proto(v.to_proto())
    self.assertIsInstance(w.dtype, dtypes.DType)
    self.assertEqual(v.dtype, w.dtype)

  # TODO(alive): get caching to work in eager mode.
  @test_util.run_deprecated_v1
  def testCachingDevice(self):
    # [FUNCTION DOC]
    # Graph-mode: a variable created on a remote device spec with a local caching device has its value on
    # the caching device and no colocation class attribute; repeats under a colocation scope.
    with ops.device("/job:server/task:1"):
      v = resource_variable_ops.ResourceVariable(
          2.0, caching_device="/job:localhost")
      self.assertEqual("/job:localhost", v.value().device)
      with self.assertRaises(ValueError):
        _ = v.value().op.get_attr("_class")

    with ops.colocate_with(v.op):
      w = resource_variable_ops.ResourceVariable(
          2.0, caching_device="/job:localhost")
      self.assertEqual("/job:localhost", w.value().device)
      with self.assertRaises(ValueError):
        _ = w.value().op.get_attr("_class")

  @test_util.run_deprecated_v1
  def testSharedName(self):
    # [FUNCTION DOC]
    # Graph-mode: reads the value of a variable through a second handle with the same shared name (300.0)
    # and asserts that a handle with a different shared name raises an op error about a missing or
    # uninitialized resource.
    with self.cached_session():
      v = resource_variable_ops.ResourceVariable(300.0, name="var4")
      self.evaluate(variables.global_variables_initializer())

      w = _eager_safe_var_handle_op(
          dtype=v.dtype.base_dtype,
          shape=v.get_shape(),
          shared_name="var4",
          # Needed in Eager since we get a unique container name by default.
          container=ops.get_default_graph()._container)
      w_read = resource_variable_ops.read_variable_op(w, v.dtype.base_dtype)
      self.assertEqual(300.0, self.evaluate(w_read))

      x = _eager_safe_var_handle_op(
          dtype=v.dtype.base_dtype,
          shape=v.get_shape(),
          shared_name="var5",
          container=ops.get_default_graph()._container)
      with self.assertRaisesOpError(
          "(Resource .*/var5/.* does not exist|uninitialized)"):
        resource_variable_ops.read_variable_op(x, v.dtype.base_dtype).eval()

  @test_util.run_deprecated_v1
  def testSharedNameWithNamescope(self):
    # [FUNCTION DOC]
    # Graph-mode: a variable created inside a name scope has the scope in its shared name and name; a
    # handle with that shared name reads 300.0.
    with self.cached_session():
      with ops.name_scope("foo"):
        v = resource_variable_ops.ResourceVariable(300.0, name="var6")
        self.assertEqual("foo/var6", v._shared_name)  # pylint: disable=protected-access
        self.assertEqual("foo/var6:0", v.name)
        self.evaluate(variables.global_variables_initializer())

      w = _eager_safe_var_handle_op(
          dtype=v.dtype.base_dtype,
          shape=v.get_shape(),
          shared_name="foo/var6",
          # Needed in Eager since we get a unique container name by default.
          container=ops.get_default_graph()._container)
      w_read = resource_variable_ops.read_variable_op(w, v.dtype.base_dtype)
      self.assertEqual(300.0, self.evaluate(w_read))

  @test_util.run_in_graph_and_eager_modes
  def testShape(self):
    # [FUNCTION DOC]
    # Runs in both modes. Asserts static shape strings for the variable, its value, and sparse reads
    # (shape starts with 3) and, in graph mode, unknown shape for placeholder-index sparse reads.
    v = resource_variable_ops.ResourceVariable(
        name="var4", initial_value=array_ops.ones(shape=[10, 20, 35]))
    self.assertEqual("(10, 20, 35)", str(v.shape))
    self.assertEqual("(10, 20, 35)", str(v.get_shape()))
    self.assertEqual("(10, 20, 35)", str(v.value().shape))
    self.assertEqual("(3, 20, 35)", str(v.sparse_read([0, 1, 2]).shape))
    if not context.executing_eagerly():
      self.assertEqual(
          "<unknown>",
          str(v.sparse_read(array_ops.placeholder(dtypes.int32)).shape))

  @test_util.run_deprecated_v1
  def testSetInitialValue(self):
    # [FUNCTION DOC]
    # Graph-mode: runs the initializer feeding a different initial value (3.0) and asserts the variable
    # reads 3.0.
    with self.cached_session():
      # Initialize variable with a value different from the initial value passed
      # in the constructor.
      v = resource_variable_ops.ResourceVariable(2.0)
      v.initializer.run(feed_dict={v.initial_value: 3.0})
      self.assertEqual(3.0, v.value().eval())

  @test_util.run_v1_only("b/120545219")
  def testControlFlowInitialization(self):
    # [FUNCTION DOC]
    # Graph-mode (v1 only): creating a variable inside a while loop body must raise a value error
    # mentioning the initial value.
    """Expects an error if an initializer is in a control-flow scope."""

    def cond(i, _):
      # [FUNCTION DOC]
      # HELPER: loop condition for testControlFlowInitialization; returns whether the counter is below 10.
      return i < 10

    def body(i, _):
      # [FUNCTION DOC]
      # HELPER: loop body for testControlFlowInitialization; creates a variable inside the loop and returns
      # the incremented counter with the variable read, triggering the expected error.
      zero = array_ops.zeros([], dtype=dtypes.int32)
      v = resource_variable_ops.ResourceVariable(initial_value=zero)
      return (i + 1, v.read_value())

    with self.assertRaisesRegex(ValueError, "initial_value"):
      while_loop.while_loop(cond, body, [0, 0])

  def testVariableEager(self):
    # [FUNCTION DOC]
    # Eager mode: checks properties (dtype, name with scope, shape, handle type, constraint), callable
    # initial values, assign_add, assign_sub, assign, load, conversion to tensor and operator results
    # for a large variable.
    with context.eager_mode():
      init = array_ops.ones(shape=[10, 20, 35], dtype=dtypes.int32)
      constraint = lambda x: x
      with ops.name_scope("foo", skip_on_eager=False):
        v = resource_variable_ops.ResourceVariable(
            name="var7",
            initial_value=init,
            caching_device="cpu:0",
            constraint=constraint)
      # Test properties
      self.assertEqual(dtypes.int32, v.dtype)
      self.assertEqual("foo/var7:0", v.name)
      self.assertAllEqual([10, 20, 35], v.shape.as_list())
      self.assertIsInstance(v.handle, ops.EagerTensor)
      self.assertEqual(constraint, v.constraint)
      self.assertAllEqual(init.numpy(), v.read_value().numpy())
      self.assertAllEqual(init.numpy(), v.value().numpy())

      # Callable init.
      callable_init = lambda: init * 2
      v2 = resource_variable_ops.ResourceVariable(
          initial_value=callable_init, name="var7")
      self.assertEqual("var7:0", v2.name)
      self.assertAllEqual(2 * init.numpy(), v2.read_value().numpy())

      # Test assign_add.
      new_v2_val = v2.assign_add(v.read_value())
      self.assertAllEqual(v.read_value().numpy() * 3, new_v2_val.numpy())

      # Test assign_sub.
      new_v2_val = v2.assign_sub(v.read_value())
      self.assertAllEqual(v.read_value().numpy() * 2, new_v2_val.numpy())

      # Test assign.
      v2.assign(v.read_value())
      self.assertAllEqual(v.read_value().numpy(), v2.read_value().numpy())

      # Test load
      v2.load(2 * v.read_value())
      self.assertAllEqual(2 * v.read_value().numpy(), v2.read_value().numpy())

      # Test convert_to_tensor
      t = ops.convert_to_tensor(v)
      self.assertAllEqual(t.numpy(), v.read_value().numpy())

      # Test operations
      self.assertAllEqual((v * 2).numpy(), (v + v).numpy())

  def testNumpyDotArray(self):
    # [FUNCTION DOC]
    # Eager mode: converting scalar and vector variables to numpy arrays gives the expected values.
    with context.eager_mode():
      # Scalars use a separate code path.
      v1 = resource_variable_ops.ResourceVariable(initial_value=lambda: 1,
                                                  name="v1")
      self.assertEqual(1, np.array(v1))

      v2 = resource_variable_ops.ResourceVariable(initial_value=lambda: [1, 2],
                                                  name="v2")
      self.assertAllEqual(v2.read_value().numpy(), np.array(v2))
      self.assertAllEqual([1, 2], np.array(v2))

  def testContainerEager(self):
    # [FUNCTION DOC]
    # Eager mode: two same-named variables, the second created inside a different container, do not
    # share state; assigning to the second does not change the first.
    with context.eager_mode():
      v1 = resource_variable_ops.ResourceVariable(initial_value=lambda: 1,
                                                  name="same")
      with ops.container("different"):
        v2 = resource_variable_ops.ResourceVariable(initial_value=lambda: 0,
                                                    name="same")
      v2.assign(2)
      self.assertEqual(1, v1.read_value().numpy())
      self.assertEqual(2, v2.read_value().numpy())

  def testDestruction(self):
    # [FUNCTION DOC]
    # Eager mode: creates a weak handle for a variable, deletes the variable, then asserts destroying
    # the handle with lookup errors enabled raises not-found.
    with context.eager_mode():
      var = resource_variable_ops.ResourceVariable(initial_value=1.0,
                                                   name="var8")
      var_handle = test_ops.make_weak_resource_handle(var._handle)
      del var
      with self.assertRaisesRegex(errors.NotFoundError,
                                  r"Resource .* does not exist."):
        resource_variable_ops.destroy_resource_op(var_handle,
                                                  ignore_lookup_error=False)

  def testScatterUpdate(self):
    # [FUNCTION DOC]
    # Eager mode: state ops scatter_update sets index 1 of [1.0, 2.0] to 3.0.
    with context.eager_mode():
      v = resource_variable_ops.ResourceVariable([1.0, 2.0], name="update")
      state_ops.scatter_update(v, [1], [3.0])
      self.assertAllEqual([1.0, 3.0], v.numpy())

  def testScatterAddStateOps(self):
    # [FUNCTION DOC]
    # Eager mode: state ops scatter_add adds 3 at index 1; expects [1.0, 5.0].
    with context.eager_mode():
      v = resource_variable_ops.ResourceVariable([1.0, 2.0], name="add")
      state_ops.scatter_add(v, [1], [3])
      self.assertAllEqual([1.0, 5.0], v.numpy())

  def testScatterSubStateOps(self):
    # [FUNCTION DOC]
    # Eager mode: state ops scatter_sub subtracts 3 at index 1; expects [1.0, -1.0].
    with context.eager_mode():
      v = resource_variable_ops.ResourceVariable([1.0, 2.0], name="sub")
      state_ops.scatter_sub(v, [1], [3])
      self.assertAllEqual([1.0, -1.0], v.numpy())

  def testScatterUpdateVariant(self):
    # [FUNCTION DOC]
    # Eager mode: a variable holding a variant list is scatter-updated with a tensor list built from
    # [1, 2]; asserts the first item of the stored list is 1.
    with context.eager_mode():
      v = resource_variable_ops.ResourceVariable([
          list_ops.empty_tensor_list(
              element_dtype=dtypes.float32, element_shape=[])
      ])
      v.scatter_update(
          indexed_slices.IndexedSlices(
              list_ops.tensor_list_from_tensor([1., 2.], element_shape=[]), 0))
      self.assertAllEqual(
          list_ops.tensor_list_get_item(v[0], 0, element_dtype=dtypes.float32),
          1.)

  def testGroupDoesntForceRead(self):
    # [FUNCTION DOC]
    # Graph mode: grouping an assign_add does not add a read; asserts the group's first control input is
    # an assign-add-variable operation.
    with ops.Graph().as_default():
      v = resource_variable_ops.ResourceVariable(1.0)
      assign = v.assign_add(1.0)
      g = control_flow_ops.group([assign])
      self.assertEqual(g.control_inputs[0].type, "AssignAddVariableOp")

  def testScatterNdAddStateOps(self):
    # [FUNCTION DOC]
    # Eager mode: scatter_nd_add of four updates into eight element float variable; asserts the expected
    # result array.
    with context.eager_mode():
      v = resource_variable_ops.ResourceVariable(
          [1, 2, 3, 4, 5, 6, 7, 8], dtype=dtypes.float32, name="add")
      indices = constant_op.constant([[4], [3], [1], [7]], dtype=dtypes.int32)
      updates = constant_op.constant([9, 10, 11, 12], dtype=dtypes.float32)
      expected = np.array([1, 13, 3, 14, 14, 6, 7, 20])
      state_ops.scatter_nd_add(v, indices, updates)
      self.assertAllClose(expected, v.numpy())

  @test_util.run_in_graph_and_eager_modes
  def testUnreadVariableInsideFunction(self):
    # [FUNCTION DOC]
    # Runs in both modes. A traced function only assigns a variable; asserts the concrete graph has no
    # read variable operations.
    v = resource_variable_ops.ResourceVariable(1.0)

    @def_function.function
    def assign():
      # [FUNCTION DOC]
      # HELPER: traced function body; assigns 1.0 to the enclosing variable and returns nothing.
      v.assign(1.0)

    graph = assign.get_concrete_function().graph
    self.assertTrue(all(x.type != "ReadVariableOp"
                        for x in graph.get_operations()))

  def testScatterNdSubStateOps(self):
    # [FUNCTION DOC]
    # Eager mode: scatter_nd_sub of four updates into an eight element variable; asserts the result.
    with context.eager_mode():
      v = resource_variable_ops.ResourceVariable(
          [1, 2, 3, 4, 5, 6, 7, 8], dtype=dtypes.float32, name="sub")
      indices = constant_op.constant([[4], [3], [1], [7]], dtype=dtypes.int32)
      updates = constant_op.constant([9, 10, 11, 12], dtype=dtypes.float32)
      expected = np.array([1, -9, 3, -6, -4, 6, 7, -4])
      state_ops.scatter_nd_sub(v, indices, updates)
      self.assertAllClose(expected, v.numpy())

  def testScatterUpdateCast(self):
    # [FUNCTION DOC]
    # Eager mode: scatter_update with integer update value is cast to float; asserts [1.0, 3.0].
    with context.eager_mode():
      v = resource_variable_ops.ResourceVariable([1.0, 2.0], name="update")
      state_ops.scatter_update(v, [1], [3])
      self.assertAllEqual([1.0, 3.0], v.numpy())

  @test_util.run_in_graph_and_eager_modes
  def testScatterUpdateInvalidArgs(self):
    # [FUNCTION DOC]
    # Runs in both modes. Scatter update with two indices and three updates raises some exception whose
    # message mentions the shape mismatch or a failed check (kernel and XLA wording both accepted).
    v = resource_variable_ops.ResourceVariable([0, 1, 2, 3], name="update")
    # The exact error and message differ between graph construction (where the
    # error is realized during shape inference at graph construction time),
    # eager execution (where the error is realized during kernel execution),
    # and XLA auto-clustering execution (where the error is realized in the xla
    # op kernel) which is triggered when running in eager op as function mode.
    with self.assertRaisesRegex(Exception, r"shape.*2.*3|RET_CHECK failure"):
      state_ops.scatter_update(v, [0, 1], [0, 1, 2])

  @test_util.disable_xla("b/208334252")  # XLA doesn't have a deterministic impl
  def testScatterAddDeterministic(self):
    # [FUNCTION DOC]
    # Eager mode with deterministic ops (XLA disabled): scatter-adds one million random values into a
    # 1024 element variable repeatedly into fresh variables and asserts all results are identical.
    with context.eager_mode(), test_util.deterministic_ops():
      # Normally a nondeterministic codepath occurs when the variable has at
      # least 1024 elements. Test that op determinism ensures the op is
      # deterministc.
      v = resource_variable_ops.ResourceVariable(array_ops.zeros([1024]))
      delta = indexed_slices.IndexedSlices(
          values=np.random.normal(size=(1_000_000,)),
          indices=array_ops.zeros((1_000_000,), dtype=np.int32),
          dense_shape=(1024,))
      v.scatter_add(delta)
      for _ in range(5):
        v2 = resource_variable_ops.ResourceVariable(array_ops.zeros([1024]))
        v2.scatter_add(delta)
        self.assertAllEqual(v, v2)

  @test_util.run_in_graph_and_eager_modes
  def testAssignIncompatibleShape(self):
    # [FUNCTION DOC]
    # Runs in both modes. assign_add of a scalar-valued update to an integer vector must raise an
    # exception whose message says shapes must be equal (case-insensitive).
    v = resource_variable_ops.ResourceVariable([0, 1, 2, 3])
    self.evaluate(v.initializer)
    pattern = re.compile("shapes must be equal", re.IGNORECASE)
    with self.assertRaisesRegex(Exception, pattern):
      self.evaluate(v.assign_add(1))

  @test_util.run_in_graph_and_eager_modes
  @test_util.run_v1_only("b/120545219")
  def testCopyToGraphUninitialized(self):
    # [FUNCTION DOC]
    # Runs in both modes (v1 only). Copies a variable to a new graph uninitialized; asserts the name is
    # the same and the initializer is absent.
    v = resource_variable_ops.ResourceVariable([0, 1, 2, 3])
    copy_to_graph = ops.Graph()
    with copy_to_graph.as_default():  # Intentionally testing v1 behavior
      copied = resource_variable_ops.copy_to_graph_uninitialized(v)
      self.assertEqual(v.name, copied.name)
      self.assertIsNone(copied.initializer)

  def create_variant_shape_and_type_data(self):
    # [FUNCTION DOC]
    # HELPER: builds handle data for the variant tests.
    # Creates a handle data message marked as set with a single shape and type entry: shape unknown by 4,
    # dtype float32. Returns it. No arguments.
    variant_shape_and_type_data = (
        cpp_shape_inference_pb2.CppShapeInferenceResult.HandleData())
    variant_shape_and_type_data.is_set = True
    stored_shape = tensor_shape.TensorShape([None, 4]).as_proto()
    stored_dtype = dtypes.float32.as_datatype_enum
    # NOTE(ebrevdo): shape_and_type lacks append() in some versions of protobuf.
    variant_shape_and_type_data.shape_and_type.extend([
        cpp_shape_inference_pb2.CppShapeInferenceResult.HandleShapeAndType(
            shape=stored_shape,
            dtype=stored_dtype,
            type=full_type_pb2.FullTypeDef())
    ])
    return variant_shape_and_type_data

  @def_function.function
  def create_constant_variant(self, value):
    # [FUNCTION DOC]
    # HELPER: builds a constant variant tensor, traced as a function.
    # Takes an integer value and returns a variant scalar constant from a tensor proto with type name
    # integer and the value's bytes as metadata.
    value = constant_op.constant(
        tensor_pb2.TensorProto(
            dtype=dtypes.variant.as_datatype_enum,
            tensor_shape=tensor_shape.TensorShape([]).as_proto(),
            variant_val=[
                tensor_pb2.VariantTensorDataProto(
                    # Match registration in variant_op_registry.cc
                    type_name=b"int",
                    metadata=np.array(value, dtype=np.int32).tobytes())
            ]))
    return value

  # TODO(ebrevdo): Add run_in_graph_and_eager_modes once we can create
  # EagerTensor constants with TensorProto inputs.
  @test_util.disable_tfrt("Does not support tf.Const in lowering.")
  @test_util.run_in_graph_and_eager_modes()
  def testVariantInitializer(self):
    # [FUNCTION DOC]
    # Runs in both modes (not on the TFRT runtime). Fills a vector with a variant constant, attaches
    # handle shape and type data, creates a variable and asserts that the identity read and the sparse
    # read preserve the handle data. In graph mode also runs the initializer and reads.
    variant_shape_and_type_data = self.create_variant_shape_and_type_data()
    value = self.create_constant_variant(3)
    initializer = array_ops.fill([3], value)
    resource_variable_ops._set_handle_shapes_and_types(  # pylint: disable=protected-access
        initializer, variant_shape_and_type_data,
        graph_mode=not context.executing_eagerly())
    v = resource_variable_ops.ResourceVariable(initializer)
    read = array_ops.identity(v)
    read_variant_shape_and_type = (
        resource_variable_ops.get_eager_safe_handle_data(read))
    self.assertEqual(
        read_variant_shape_and_type, variant_shape_and_type_data)
    gather = v.sparse_read([0])
    gather_variant_shape_and_type = (
        resource_variable_ops.get_eager_safe_handle_data(gather))
    self.assertEqual(
        gather_variant_shape_and_type, variant_shape_and_type_data)
    # Make sure initializer runs.
    if not context.executing_eagerly():
      self.evaluate(v.initializer)
      self.evaluate(read.op)
      self.evaluate(gather.op)

  @parameterized.parameters([
      # batch_dims=0 (equivalent to tf.gather)
      dict(  # 2D indices
          batch_dims=0,
          params=[6, 7, 8, 9],
          indices=[[2, 1], [0, 3]],
          expected=[[8, 7], [6, 9]]),
      dict(  # 3D indices
          batch_dims=0,
          params=[6, 7, 8, 9],
          indices=[[[3, 1], [2, 0]], [[0, 3], [2, 2]]],
          expected=[[[9, 7], [8, 6]], [[6, 9], [8, 8]]]),
      dict(  # 4D indices
          batch_dims=0,
          params=[8, 9],
          indices=[[[[0, 1], [1, 0]], [[0, 0], [1, 1]]],
                   [[[1, 1], [0, 0]], [[0, 1], [1, 0]]]],
          expected=[[[[8, 9], [9, 8]], [[8, 8], [9, 9]]],
                    [[[9, 9], [8, 8]], [[8, 9], [9, 8]]]]),

      # batch_dims=indices.shape.ndims - 1 (equivalent to
      # tf.compat.v1.batch_gather)
      dict(  # 2D indices (1 batch dim)
          batch_dims=1,
          params=[[10, 11, 12, 13], [20, 21, 22, 23]],
          indices=[[2, 1], [0, 3]],
          expected=[[12, 11], [20, 23]]),
      dict(  # 3D indices (2 batch dims)
          batch_dims=2,
          params=[[[100, 101], [110, 111]], [[200, 201], [210, 211]]],
          indices=[[[0, 1], [1, 0]], [[0, 0], [1, 1]]],
          expected=[[[100, 101], [111, 110]], [[200, 200], [211, 211]]]),
      dict(  # 2D indices (1 batch dim)
          batch_dims=1,
          params=[[10, 11, 12, 13], [20, 21, 22, 23]],
          indices=[[2, 1], [0, 3]],
          expected=[[12, 11], [20, 23]]),
      dict(  # 3D indices (2 batch dims)
          batch_dims=2,
          params=[[[100, 101], [110, 111]], [[200, 201], [210, 211]]],
          indices=[[[0, 1], [1, 0]], [[0, 0], [1, 1]]],
          expected=[[[100, 101], [111, 110]], [[200, 200], [211, 211]]]),

      # 0 < batch_dims < indices.shape.ndims - 1
      dict(  # 3D indices (1 batch dim)
          batch_dims=1,
          params=[[10, 11, 12, 13], [20, 21, 22, 23]],
          indices=[[[3, 1], [2, 0]], [[0, 3], [2, 2]]],
          expected=[[[13, 11], [12, 10]], [[20, 23], [22, 22]]]),
      dict(  # 4D indices (1 batch dim)
          batch_dims=1,
          params=[[6, 7], [8, 9]],
          indices=[[[[0, 1], [1, 0]], [[0, 0], [1, 1]]],
                   [[[1, 1], [0, 0]], [[0, 1], [1, 0]]]],
          expected=[[[[6, 7], [7, 6]], [[6, 6], [7, 7]]],
                    [[[9, 9], [8, 8]], [[8, 9], [9, 8]]]]),
      dict(  # 4D indices (2 batch dims)
          batch_dims=2,
          params=[[[2, 3], [4, 5]], [[6, 7], [8, 9]]],
          indices=[[[[0, 1], [1, 0]], [[0, 0], [1, 1]]],
                   [[[1, 1], [0, 0]], [[0, 1], [1, 0]]]],
          expected=[[[[2, 3], [3, 2]], [[4, 4], [5, 5]]],
                    [[[7, 7], [6, 6]], [[8, 9], [9, 8]]]]),
  ])
  @test_util.run_in_graph_and_eager_modes
  def testGatherWithBatchDims(self, params, indices, batch_dims, expected):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the parameter list contains two pairs of duplicated cases (the 1 batch dim and 2 batch
    # dim cases with 2D and 3D indices appear twice), so those cases run redundantly.
    # Parameterized over params, indices, batch_dims and expected. Creates a variable from params and
    # asserts that resource gather with batch dims returns the expected values.
    var = resource_variable_ops.ResourceVariable(params, name="var0")
    with ops.control_dependencies([var.initializer]):
      result = resource_variable_ops.resource_gather(
          var.handle, indices, dtype=var.dtype, batch_dims=batch_dims)
    self.assertAllEqual(expected, result)

  @parameterized.parameters([
      dict(
          params_shape=[2, 3, 4, 5, 6, 7],
          indices_shape=[2, 3, 8, 9, 10],
          batch_dims=0,
          output_shape=[2, 3, 8, 9, 10, 3, 4, 5, 6, 7]
          # = indices.shape + params.shape[1:]
      ),
      dict(
          params_shape=[2, 3, 4, 5, 6, 7],
          indices_shape=[2, 3, 8, 9, 10],
          batch_dims=1,
          output_shape=[2, 3, 8, 9, 10, 4, 5, 6, 7]
          # = params.shape[:1] + indices.shape[1:] + params.shape[2:]
      ),
      dict(
          params_shape=[2, 3, 4, 5, 6, 7],
          indices_shape=[2, 3, 8, 9, 10],
          batch_dims=2,
          output_shape=[2, 3, 8, 9, 10, 5, 6, 7]
          # = params.shape[:2] + indices.shape[2:] + params.shape[3:]
      ),
      dict(
          params_shape=[2, 3, 4, 5, 6, 7],
          indices_shape=[2, 3, 4, 9, 10],
          batch_dims=3,
          output_shape=[2, 3, 4, 9, 10, 6, 7]
          # = params.shape[:3] + indices.shape[3:] + params.shape[4:]
      ),
      dict(
          params_shape=[2, 3, 4, 5, 6, 7],
          indices_shape=[2, 3, 4, 5, 10],
          batch_dims=4,
          output_shape=[2, 3, 4, 5, 10, 7]
          # = params.shape[:4] + indices.shape[4:] + params.shape[5:]
      ),
  ])
  @test_util.run_in_graph_and_eager_modes
  def testGatherWithBatchDimsMatchesTensor(self, params_shape, indices_shape,
                                           batch_dims, output_shape):
    # [FUNCTION DOC]
    # Parameterized over params shape, indices shape, batch dims and output shape. Builds arrays of
    # counting values, wraps indices into range, then asserts resource gather returns the expected
    # output shape and equals ordinary gather with batch dims.
    """Checks that gather with batch_dims returns the correct shape."""
    # Generate a `params` tensor with the indicated shape.
    params_size = np.prod(params_shape)
    params = np.reshape(np.arange(params_size, dtype=np.int32), params_shape)

    # Generate an `indices` tensor with the indicated shape, where each index
    # is within the appropriate range.
    indices_size = np.prod(indices_shape)
    indices = np.reshape(np.arange(indices_size, dtype=np.int32), indices_shape)
    indices = indices % params_shape[batch_dims]

    var = resource_variable_ops.ResourceVariable(params, name="var0")
    with ops.control_dependencies([var.initializer]):
      expected = array_ops.gather(
          var.read_value(), indices, batch_dims=batch_dims)
      result = resource_variable_ops.resource_gather(
          var.handle, indices, dtype=var.dtype, batch_dims=batch_dims)

    self.assertAllEqual(output_shape, result.shape.as_list())
    self.assertAllEqual(expected, result)

  @parameterized.parameters([
      dict(dtype=dtypes.bool),
      dict(dtype=dtypes.int64),
      dict(dtype=dtypes.half),
      dict(dtype=dtypes.bfloat16),
      dict(dtype=dtypes.float32),
      dict(dtype=dtypes.double),
  ])
  @test_util.run_gpu_only
  @test_util.run_in_graph_and_eager_modes
  def testGatherWithDTypes(self, dtype):
    # [FUNCTION DOC]
    # Parameterized over bool, int64, half, bfloat16, float32, double; GPU only. Gathers [[2,1],[0,3]]
    # from a four element variable and asserts the expected values.
    if dtype == dtypes.bool:
      params = constant_op.constant([False, True, False, True])
      expected = constant_op.constant([[False, True], [False, True]])
    else:
      params = constant_op.constant([6, 7, 8, 9], dtype=dtype)
      expected = constant_op.constant([[8, 7], [6, 9]], dtype=dtype)
    indices = constant_op.constant([[2, 1], [0, 3]])
    var = resource_variable_ops.ResourceVariable(params, name="var0")
    with ops.control_dependencies([var.initializer]):
      result = resource_variable_ops.resource_gather(
          var.handle, indices, dtype=dtype)
    self.assertAllEqual(expected, result)

  @test_util.run_v2_only
  def testIterateVariable(self):
    # [FUNCTION DOC]
    # Version 2 behavior only. Iterating a variable yields its elements; asserts [1.0, 2.0].
    v = variables.Variable([1., 2.])
    self.assertAllClose([1., 2.], list(iter(v)))

  @test_util.run_in_graph_and_eager_modes
  def testCompositeTensorTypeSpec(self):
    # [FUNCTION DOC]
    # Runs in both modes. Asserts the variable is a composite tensor with a variable spec having the
    # right shape, dtype and trainable flag, equal to its type spec and shape invariant spec.
    v = resource_variable_ops.ResourceVariable([1.])
    self.evaluate(v.initializer)
    self.assertIsInstance(v, composite_tensor.CompositeTensor)
    spec = type_spec.type_spec_from_value(v)

    self.assertIsInstance(spec, resource_variable_ops.VariableSpec)
    self.assertAllEqual(spec.shape.as_list(), (1,))
    self.assertEqual(spec.dtype, dtypes.float32)
    self.assertTrue(spec.trainable)
    self.assertEqual(spec, v._type_spec)
    self.assertEqual(spec, v._shape_invariant_to_type_spec((1,)))

  @test_util.run_in_graph_and_eager_modes
  def testVariableInExtensionType(self):
    # [FUNCTION DOC]
    # Runs in both modes. Defines an extension type with a variable and a mask and asserts both fields
    # read back as given.
    class MaskVariable(extension_type.ExtensionType):
      # [CLASS DOC]
      # HELPER: extension type local to testVariableInExtensionType holding a variable and a boolean mask
      # tensor.
      variable: resource_variable_ops.ResourceVariable
      mask: tensor_lib.Tensor

    v = resource_variable_ops.ResourceVariable([1., 2.])
    self.evaluate(v.initializer)
    mask = constant_op.constant([True, False])
    mask_variable = MaskVariable(variable=v, mask=mask)
    self.assertAllEqual(mask_variable.variable, [1., 2.])
    self.assertAllEqual(mask_variable.mask, [True, False])

  @test_util.run_in_graph_and_eager_modes
  def testInitFromHandle(self):
    # [FUNCTION DOC]
    # Runs in both modes. Creates a second variable from the first one's handle; asserts the handles are
    # the same object and value converts to 1.0.
    v = resource_variable_ops.ResourceVariable(1.)
    self.evaluate(v.initializer)
    v2 = resource_variable_ops.ResourceVariable(
        trainable=True, shape=(), dtype=dtypes.float32, handle=v.handle)
    self.assertIs(v2.handle, v.handle)
    self.assertAllEqual(ops.convert_to_tensor(v2), 1.)

  @test_util.run_in_graph_and_eager_modes
  def testFlattenResourceVariable(self):
    # [FUNCTION DOC]
    # Runs in both modes. Flattening with composites expanded returns the variable itself as the first
    # element (to-do notes a future change to a resource tensor).
    v = resource_variable_ops.ResourceVariable(1.)
    self.evaluate(v.initializer)
    result = nest.flatten(v, expand_composites=True)
    # TODO(b/246438937): Update this to dt_resource tensor once we expand
    # ResourceVariables with expand_composites=True.
    self.assertIsInstance(result[0], resource_variable_ops.ResourceVariable)

  @test_util.run_in_graph_and_eager_modes
  def testUniqueIdPreservedThroughPackAndUnpack(self):
    # [FUNCTION DOC]
    # Runs in both modes. Flatten and repack the variable with composites expanded; the unique id equals
    # the original.
    v = resource_variable_ops.ResourceVariable(1.)
    self.evaluate(v.initializer)
    expected_unique_id = v._unique_id
    reconstructed_v = nest.pack_sequence_as(
        v,
        nest.flatten(v, expand_composites=True),
        expand_composites=True)
    self.assertEqual(reconstructed_v._unique_id, expected_unique_id)

  @test_util.run_in_graph_and_eager_modes
  def testHandleNamePreservedThroughPackAndUnpack(self):
    # [FUNCTION DOC]
    # Runs in both modes. Flatten and repack the variable; the handle name equals the original.
    v = resource_variable_ops.ResourceVariable(1.)
    self.evaluate(v.initializer)
    expected_handle_name = v._handle_name
    reconstructed_v = nest.pack_sequence_as(
        v,
        nest.flatten(v, expand_composites=True),
        expand_composites=True)
    self.assertEqual(reconstructed_v._handle_name, expected_handle_name)

  @test_util.run_in_graph_and_eager_modes
  def testGatherBatchDimsNeg(self):
    # [FUNCTION DOC]
    # Runs in both modes. Gathering with batch dims -42 must raise a value error or invalid-argument
    # error saying batch dims is negative or out of range.
    var = resource_variable_ops.ResourceVariable(
        [1], dtype=dtypes.int32, name="var0"
    )
    with ops.control_dependencies([var.initializer]):
      with self.assertRaisesRegex(
          (ValueError, errors.InvalidArgumentError),
          "(batch_dims is negative)|(Expected batch_dims in the range)"
      ):
        result = resource_variable_ops.resource_gather(
            var.handle,
            indices=[1],
            dtype=var.dtype,
            batch_dims=-42,
        )
        self.evaluate(result)

if __name__ == "__main__":
  test.main()
