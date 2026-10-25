# [MODULE DOC]
# Tests for the legacy reference-variable ops (state_ops.variable_op, assign, assign_add, assign_sub,
# temporary variables, is_variable_initialized) in graph mode.
# Defines a mapping from numpy dtypes to TensorFlow dtypes and one test class, VariableOpTest.
# Most tests run with the deprecated-v1 graph decorator. Not a helper module; no import-time side
# effects beyond building the dtype table.
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
"""Tests for tensorflow.ops.tf.variable_op."""

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

_NP_TO_TF = {
    dtypes.bfloat16.as_numpy_dtype: dtypes.bfloat16,
    np.float16: dtypes.float16,
    np.float32: dtypes.float32,
    np.float64: dtypes.float64,
    np.int32: dtypes.int32,
    np.int64: dtypes.int64,
    np.complex64: dtypes.complex64,
    np.complex128: dtypes.complex128,
}


class VariableOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering reference variables created with the variable op and the assign family of
  # state ops: dtype round trips, static shape inference with and without validate_shape/set_shape,
  # temporary variable lifecycle errors, cross-device dependencies and string variables.
  # No shared fixtures other than the standard test case setup.

  def _initFetch(self, x, tftype, use_gpu=None):
    # [FUNCTION DOC]
    # HELPER: dtype round-trip support for the variable op tests.
    # Creates a reference variable of the given dtype with the shape of x, assigns x to it, runs the
    # assignment and returns the evaluated variable value.
    # Inputs: x is a numpy array, tftype the TensorFlow dtype, use_gpu optional device preference
    # (default none) passed to the test session. Returns the fetched numpy value.
    # Side effects: builds ops in the default graph and runs them in a temporary session.
    with self.test_session(use_gpu=use_gpu):
      p = state_ops.variable_op(x.shape, tftype)
      op = state_ops.assign(p, x)
      op.op.run()
      return self.evaluate(p)

  def _testTypes(self, vals):
    # [FUNCTION DOC]
    # HELPER: drives the per-dtype checks used by testBasic.
    # For each of eight dtypes (float16/32/64, complex64/128, int32/64, bfloat16) it calls setUp to reset
    # state, casts vals to that dtype and checks that assigning and fetching returns the same values,
    # first on CPU only and then with GPU allowed. Input: vals, a numpy array. Returns nothing; fails
    # through assertions.
    for dtype in [
        np.float16, np.float32, np.float64, np.complex64, np.complex128,
        np.int32, np.int64, dtypes.bfloat16.as_numpy_dtype
    ]:
      self.setUp()
      x = vals.astype(dtype)
      tftype = _NP_TO_TF[dtype]
      self.assertAllEqual(x, self._initFetch(x, tftype, use_gpu=False))
      # NOTE(touts): the GPU test should pass for all types, whether the
      # Variable op has an implementation for that type on GPU as we expect
      # that Variable and Assign have GPU implementations for matching tf.
      self.assertAllEqual(x, self._initFetch(x, tftype, use_gpu=True))

  @test_util.run_deprecated_v1
  def testBasic(self):
    # [FUNCTION DOC]
    # Runs the dtype round-trip check on a 4x5 array of consecutive integers for all supported dtypes.
    # Expects each fetched value to equal the assigned value.
    self._testTypes(np.arange(0, 20).reshape([4, 5]))

  @test_util.run_deprecated_v1
  def testset_shape(self):
    # [FUNCTION DOC]
    # Checks that the variable op's static shape is [1, 2] by default and unknown when set_shape is
    # false. Asserts on the shape of the returned tensor in each case.
    p = state_ops.variable_op([1, 2], dtypes.float32)
    self.assertEqual([1, 2], p.get_shape())
    p = state_ops.variable_op([1, 2], dtypes.float32, set_shape=False)
    self.assertEqual(tensor_shape.unknown_shape(), p.get_shape())

  @test_util.run_deprecated_v1
  def testAssign(self):
    # [FUNCTION DOC]
    # For several dtypes (float32, int64, uint32, uint8) creates a variable and assigns a 1x2 array.
    # Asserts that the variable and the assign result have the same static shape as the value.
    for dtype in [dtypes.float32, dtypes.int64, dtypes.uint32, dtypes.uint8]:
      value = np.array([[42, 43]])
      var = state_ops.variable_op(value.shape, dtype)
      self.assertShapeEqual(value, var)
      assigned = state_ops.assign(var, value)
      self.assertShapeEqual(value, assigned)

  @test_util.run_deprecated_v1
  def testAssignNoValidateShape(self):
    # [FUNCTION DOC]
    # Assigns a 1x2 float array to a float32 variable with shape validation turned off. Asserts the
    # shapes of the variable and the assign result match the value's shape.
    value = np.array([[42.0, 43.0]])
    var = state_ops.variable_op(value.shape, dtypes.float32)
    self.assertShapeEqual(value, var)
    assigned = state_ops.assign(var, value, validate_shape=False)
    self.assertShapeEqual(value, assigned)

  @test_util.run_deprecated_v1
  def testAssignNoVarShape(self):
    # [FUNCTION DOC]
    # Creates a variable without a static shape and asserts that its shape is unknown, then assigns a
    # value with validation on and asserts the result takes the value's shape.
    value = np.array([[42.0, 43.0]])
    var = state_ops.variable_op(value.shape, dtypes.float32, set_shape=False)
    self.assertEqual(tensor_shape.unknown_shape(), var.get_shape())
    assigned = state_ops.assign(var, value)
    self.assertShapeEqual(value, assigned)

  @test_util.run_deprecated_v1
  def testAssignNoVarShapeNoValidateShape(self):
    # [FUNCTION DOC]
    # Variable has no static shape and assignment has shape validation disabled. Asserts the variable
    # shape is unknown and the assign result takes the shape of the assigned value.
    value = np.array([[42.0, 43.0]])
    var = state_ops.variable_op(value.shape, dtypes.float32, set_shape=False)
    self.assertEqual(tensor_shape.unknown_shape(), var.get_shape())
    assigned = state_ops.assign(var, value, validate_shape=False)
    self.assertShapeEqual(value, assigned)

  def _NewShapelessTensor(self):
    # [FUNCTION DOC]
    # HELPER: builds a value with unknown static shape for the assign tests.
    # Creates a float32 placeholder, asserts its static shape is completely unknown and returns it.
    # Takes no arguments; adds a placeholder to the default graph.
    tensor = array_ops.placeholder(dtypes.float32)
    self.assertEqual(tensor_shape.unknown_shape(), tensor.get_shape())
    return tensor

  @test_util.run_deprecated_v1
  def testAssignNoValueShape(self):
    # [FUNCTION DOC]
    # Assigns a value of unknown static shape to a variable of shape [1, 2] with validation on. Asserts
    # that both the variable and the assign result report shape [1, 2].
    value = self._NewShapelessTensor()
    shape = [1, 2]
    var = state_ops.variable_op(shape, dtypes.float32)
    assigned = state_ops.assign(var, value)
    self.assertEqual(shape, var.get_shape())
    self.assertEqual(shape, assigned.get_shape())

  @test_util.run_deprecated_v1
  def testAssignNoValueShapeNoValidateShape(self):
    # [FUNCTION DOC]
    # Assigns a value of unknown shape with validation off. Asserts that the variable keeps shape [1, 2]
    # while the assign result has unknown shape.
    value = self._NewShapelessTensor()
    shape = [1, 2]
    var = state_ops.variable_op(shape, dtypes.float32)
    self.assertEqual(shape, var.get_shape())
    assigned = state_ops.assign(var, value, validate_shape=False)
    self.assertEqual(tensor_shape.unknown_shape(), assigned.get_shape())

  @test_util.run_deprecated_v1
  def testAssignNoShape(self):
    # [FUNCTION DOC]
    # Within a cached session: neither the variable nor the assigned value has a known shape. Asserts
    # that the variable shape and the assign result shape are both unknown.
    with self.cached_session():
      value = self._NewShapelessTensor()
      var = state_ops.variable_op([1, 2], dtypes.float32, set_shape=False)
      self.assertEqual(tensor_shape.unknown_shape(), var.get_shape())
      self.assertEqual(tensor_shape.unknown_shape(),
                       state_ops.assign(var, value).get_shape())

  @test_util.run_deprecated_v1
  def testAssignNoShapeNoValidateShape(self):
    # [FUNCTION DOC]
    # Same as the no-shape case but with shape validation disabled in the assignment. Asserts that the
    # variable and the assign result both have unknown shape.
    with self.cached_session():
      value = self._NewShapelessTensor()
      var = state_ops.variable_op([1, 2], dtypes.float32, set_shape=False)
      self.assertEqual(tensor_shape.unknown_shape(), var.get_shape())
      self.assertEqual(
          tensor_shape.unknown_shape(),
          state_ops.assign(var, value, validate_shape=False).get_shape())

  @test_util.run_deprecated_v1
  def testAssignUpdate(self):
    # [FUNCTION DOC]
    # For float32, int64, uint32, uint8 and bfloat16 creates a [1, 2] variable and applies assign_add and
    # assign_sub with constant lists. Asserts both results have static shape [1, 2].
    for dtype in [
        dtypes.float32, dtypes.int64, dtypes.uint32, dtypes.uint8,
        dtypes.bfloat16
    ]:
      var = state_ops.variable_op([1, 2], dtype)
      added = state_ops.assign_add(var, [[2, 3]])
      self.assertEqual([1, 2], added.get_shape())
      subbed = state_ops.assign_sub(var, [[12, 13]])
      self.assertEqual([1, 2], subbed.get_shape())

  @test_util.run_deprecated_v1
  def testAssignUpdateNoVarShape(self):
    # [FUNCTION DOC]
    # Variable created without a static shape; applies assign_add and assign_sub with float lists.
    # Asserts both update results have shape [1, 2], taken from the update values.
    var = state_ops.variable_op([1, 2], dtypes.float32, set_shape=False)
    added = state_ops.assign_add(var, [[2.0, 3.0]])
    self.assertEqual([1, 2], added.get_shape())
    subbed = state_ops.assign_sub(var, [[12.0, 13.0]])
    self.assertEqual([1, 2], subbed.get_shape())

  @test_util.run_deprecated_v1
  def testAssignUpdateNoValueShape(self):
    # [FUNCTION DOC]
    # Variable has shape [1, 2]; the added and subtracted values have unknown shape. Asserts both update
    # results report shape [1, 2], inferred from the variable.
    var = state_ops.variable_op([1, 2], dtypes.float32)
    added = state_ops.assign_add(var, self._NewShapelessTensor())
    self.assertEqual([1, 2], added.get_shape())
    subbed = state_ops.assign_sub(var, self._NewShapelessTensor())
    self.assertEqual([1, 2], subbed.get_shape())

  @test_util.run_deprecated_v1
  def testAssignUpdateNoShape(self):
    # [FUNCTION DOC]
    # Neither the variable nor the update values have a known shape. Asserts that the results of
    # assign_add and assign_sub both have unknown shape.
    var = state_ops.variable_op([1, 2], dtypes.float32, set_shape=False)
    added = state_ops.assign_add(var, self._NewShapelessTensor())
    self.assertEqual(tensor_shape.unknown_shape(), added.get_shape())
    subbed = state_ops.assign_sub(var, self._NewShapelessTensor())
    self.assertEqual(tensor_shape.unknown_shape(), subbed.get_shape())

  @test_util.run_deprecated_v1
  def testTemporaryVariable(self):
    # [FUNCTION DOC]
    # On GPU-enabled device scope: creates a temporary variable, assigns 4 and 5, adds 6 and 7, then
    # destroys it by name. Asserts the final fetched value is 10 and 12.
    with test_util.use_gpu():
      var = gen_state_ops.temporary_variable([1, 2],
                                             dtypes.float32,
                                             var_name="foo")
      var = state_ops.assign(var, [[4.0, 5.0]])
      var = state_ops.assign_add(var, [[6.0, 7.0]])
      final = gen_state_ops.destroy_temporary_variable(var, var_name="foo")
      self.assertAllClose([[10.0, 12.0]], self.evaluate(final))

  @test_util.run_deprecated_v1
  def testDestroyNonexistentTemporaryVariable(self):
    # [FUNCTION DOC]
    # Destroys a temporary variable using a name that does not match the one created. Asserts that
    # evaluating the result raises a not-found error.
    with test_util.use_gpu():
      var = gen_state_ops.temporary_variable([1, 2], dtypes.float32)
      final = gen_state_ops.destroy_temporary_variable(var, var_name="bad")
      with self.assertRaises(errors.NotFoundError):
        self.evaluate(final)

  @test_util.run_deprecated_v1
  def testDuplicateTemporaryVariable(self):
    # [FUNCTION DOC]
    # Creates two temporary variables with the same name and adds them. Asserts evaluation raises an
    # already-exists error.
    with test_util.use_gpu():
      var1 = gen_state_ops.temporary_variable([1, 2],
                                              dtypes.float32,
                                              var_name="dup")
      var1 = state_ops.assign(var1, [[1.0, 2.0]])
      var2 = gen_state_ops.temporary_variable([1, 2],
                                              dtypes.float32,
                                              var_name="dup")
      var2 = state_ops.assign(var2, [[3.0, 4.0]])
      final = var1 + var2
      with self.assertRaises(errors.AlreadyExistsError):
        self.evaluate(final)

  @test_util.run_deprecated_v1
  def testDestroyTemporaryVariableTwice(self):
    # [FUNCTION DOC]
    # Destroys the same temporary variable twice under one name and sums the results. Asserts that
    # evaluation raises a not-found error because the second destroy finds nothing.
    with test_util.use_gpu():
      var = gen_state_ops.temporary_variable([1, 2], dtypes.float32)
      val1 = gen_state_ops.destroy_temporary_variable(var, var_name="dup")
      val2 = gen_state_ops.destroy_temporary_variable(var, var_name="dup")
      final = val1 + val2
      with self.assertRaises(errors.NotFoundError):
        self.evaluate(final)

  @test_util.run_deprecated_v1
  def testTemporaryVariableNoLeak(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: despite the name, nothing asserts that no resource leaks; it only checks that taking
    # an identity of an unassigned temporary variable and evaluating it runs without raising.
    # Creates a named temporary variable, applies identity, and evaluates it.
    with test_util.use_gpu():
      var = gen_state_ops.temporary_variable([1, 2],
                                             dtypes.float32,
                                             var_name="bar")
      final = array_ops.identity(var)
      self.evaluate(final)

  @test_util.run_deprecated_v1
  def testTwoTemporaryVariablesNoLeaks(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: despite the name, nothing asserts that no resource leaks; it only checks that adding
    # two distinctly named temporary variables evaluates without raising.
    # Creates two temporary variables and evaluates their sum.
    with test_util.use_gpu():
      var1 = gen_state_ops.temporary_variable([1, 2],
                                              dtypes.float32,
                                              var_name="var1")
      var2 = gen_state_ops.temporary_variable([1, 2],
                                              dtypes.float32,
                                              var_name="var2")
      final = var1 + var2
      self.evaluate(final)

  @test_util.run_deprecated_v1
  def testAssignDependencyAcrossDevices(self):
    # [FUNCTION DOC]
    # Variable and its increment are on the GPU while a multiply is pinned to the CPU under a control
    # dependency on the increment. Asserts the multiply sees the incremented value, so the result is 4.
    with test_util.use_gpu():
      # The variable and an op to increment it are on the GPU.
      var = state_ops.variable_op([1], dtypes.float32)
      self.evaluate(state_ops.assign(var, [1.0]))
      increment = state_ops.assign_add(var, [1.0])
      with ops.control_dependencies([increment]):
        with test_util.force_cpu():
          # This mul op is pinned to the CPU, but reads the variable from the
          # GPU. The test ensures that the dependency on 'increment' is still
          # honored, i.e., the Send and Recv from GPU to CPU should take place
          # only after the increment.
          result = math_ops.multiply(var, var)
      self.assertAllClose([4.0], self.evaluate(result))

  @test_util.run_deprecated_v1
  def testIsVariableInitialized(self):
    # [FUNCTION DOC]
    # For GPU and CPU placement: creates a variable and asserts the initialized check is false, assigns a
    # value, then asserts the check is true. Evaluates in a session per device setting.
    for use_gpu in [True, False]:
      with self.test_session(use_gpu=use_gpu):
        v0 = state_ops.variable_op([1, 2], dtypes.float32)
        self.assertEqual(False, variable_v1.is_variable_initialized(v0).eval())
        state_ops.assign(v0, [[2.0, 3.0]]).eval()
        self.assertEqual(True, variable_v1.is_variable_initialized(v0).eval())


  @test_util.run_deprecated_v1
  def testString(self):
    # [FUNCTION DOC]
    # Creates a non-trainable string reference variable (local-variables collection, no shape
    # validation), assigns a stacked bytes value, initializes local variables in a cached session and
    # asserts the result equals the bytes value.
    data = array_ops_stack.stack([b"data"])
    buffer_var = variable_v1.VariableV1(
        initial_value=array_ops.zeros(shape=(), dtype=dtypes.string),
        trainable=False,
        collections=[ops.GraphKeys.LOCAL_VARIABLES],
        name="buffer",
        dtype=dtypes.string,
        validate_shape=False,
        use_resource=False)
    result = state_ops.assign(buffer_var, data, validate_shape=False)
    with self.cached_session() as sess:
      sess.run(variables.local_variables_initializer())
      self.assertEqual(result.eval(), b"data")


if __name__ == "__main__":
  test.main()
