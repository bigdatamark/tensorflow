# [MODULE DOC]
# Tests for the FTRL (follow the regularized leader) resource-variable apply ops on the XLA
# test device. Exposes one test class that applies a single FTRL step and checks the updated
# variable, accumulator and linear slots. No import-time side effects beyond imports.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Test cases for ftrl ("follow the regularized leader") operations."""

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gen_training_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import googletest


class ResourceApplyFtrlTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # Test case for the resource FTRL apply ops (v1 and v2 with l2 shrinkage). Each test runs one
  # update through a shared evaluation helper and asserts on one of the resulting slots.
  # Holds a flag indicating whether ops must be rewritten for TPU.
  """Test cases for ftrl ops."""

  def setUp(self):
    # [FUNCTION DOC]
    # Calls the parent setup and records whether the device is a TPU running with the MLIR
    # bridge enabled. Takes no arguments, returns nothing.
    super().setUp()
    self.rewrite_ops_for_tpu = ("TPU" in self.device and
                                test_util.is_mlir_bridge_enabled())

  def _eval(self, var, accum, linear, grad, lr, l1, l2, l2_shrinkage=0,
            lr_power=1, multiply_linear_by_lr=False):
    # [FUNCTION DOC]
    # HELPER: runs one FTRL update for the tests in this class.
    # Inputs: initial values of the variable, accumulator, linear slot and gradient (array-like,
    # converted to float32), learning rate, l1 and l2 regularization strengths, optional l2
    # shrinkage (default zero; nonzero selects the v2 op), learning-rate power (default one) and
    # a flag (default false) to multiply the linear slot by the learning rate. Creates resource
    # variables on the test device, runs the apply op in a session and returns a tuple of the
    # final variable, accumulator and linear values reshaped to the input shapes. Asserts that
    # shrinkage and the multiply flag are not used together.
    dtype = np.float32
    var = np.array(var, dtype=dtype)
    accum = np.array(accum, dtype=dtype)
    linear = np.array(linear, dtype=dtype)
    grad = np.array(grad, dtype=dtype)
    use_v2 = bool(l2_shrinkage)
    with self.session() as session:
      with self.test_scope():
        lr = constant_op.constant(lr, dtype=dtype)
        l1 = constant_op.constant(l1, dtype=dtype)
        l2 = constant_op.constant(l2, dtype=dtype)
        l2_shrinkage = constant_op.constant(l2_shrinkage, dtype=dtype)
        lr_power = constant_op.constant(lr_power, dtype=dtype)
        v_var = resource_variable_ops.ResourceVariable(var, dtype=dtype)
        v_accum = resource_variable_ops.ResourceVariable(accum, dtype=dtype)
        v_linear = resource_variable_ops.ResourceVariable(linear, dtype=dtype)
        session.run(v_var.create)
        session.run(v_accum.create)
        session.run(v_linear.create)
        assert not (use_v2 and multiply_linear_by_lr)
        if use_v2:
          session.run(gen_training_ops.resource_apply_ftrl_v2(
              v_var.handle, v_accum.handle, v_linear.handle,
              grad, lr, l1, l2, l2_shrinkage, lr_power,
              multiply_linear_by_lr=multiply_linear_by_lr))
        else:
          session.run(gen_training_ops.resource_apply_ftrl(
              v_var.handle, v_accum.handle, v_linear.handle,
              grad, lr, l1, l2, lr_power,
              multiply_linear_by_lr=multiply_linear_by_lr))
        return (v_var.read_value().eval().reshape(var.shape),
                v_accum.read_value().eval().reshape(accum.shape),
                v_linear.read_value().eval().reshape(linear.shape))

  def testAccum(self):
    # [FUNCTION DOC]
    # Applies one FTRL step with a nonzero gradient and checks the accumulator becomes the old
    # accumulator plus the squared gradient.
    """Test that accum is updated with grad^2."""
    accum = np.array([[[1, 3], [2, 5], [6, 8]]])
    grad = np.array([[[1, 3], [2, 5], [6, 8]]])
    _, new_accum, _ = self._eval(
        var=np.zeros((1, 3, 2)),
        accum=accum,
        linear=np.zeros((1, 3, 2)),
        grad=grad,
        lr=7, l1=3, l2=7, lr_power=2)
    self.assertAllClose(accum + grad*grad, new_accum)

  def testLinearNoGradient(self):
    # [FUNCTION DOC]
    # Applies a step with a zero gradient so the new accumulator equals the old one, and checks
    # the linear slot is unchanged.
    """Test that if accum_new == accum, linear doesn't change."""
    _, _, linear = self._eval(
        var=np.ones((1, 3, 2)),
        accum=[[[1, 3], [2, 5], [6, 8]]],
        linear=[[[1, 2], [3, 4], [5, 6]]],
        grad=np.zeros((1, 3, 2)),  # make accum_new == acum
        lr=1, l1=3, l2=7, lr_power=2)
    self.assertAllClose([[[1, 2], [3, 4], [5, 6]]], linear)

  def testLinear(self):
    # [FUNCTION DOC]
    # Applies a step from unit accumulator, zero linear and unit gradient with learning rate one
    # and power two; checks the linear slot becomes 1.75 everywhere.
    """Test the linear update for new_linear=2 and linear=1."""
    _, _, linear = self._eval(
        var=np.ones((1, 3, 2)),
        accum=np.ones((1, 3, 2)),
        linear=np.zeros((1, 3, 2)),
        grad=np.ones((1, 3, 2)),
        lr=1, l1=3, l2=7, lr_power=2)
    self.assertAllClose(1.75 * np.ones((1, 3, 2)), linear)

  def testLR(self):
    # [FUNCTION DOC]
    # Applies a step with learning rate five and power minus one; checks the linear slot equals
    # 0.8 everywhere, i.e. the update is divided by the learning rate.
    """Test that the linear update is divided by lr."""
    _, _, linear = self._eval(
        var=np.ones((1, 3, 2)),
        accum=np.ones((1, 3, 2)),
        linear=np.zeros((1, 3, 2)),
        grad=np.ones((1, 3, 2)),
        lr=5, l1=3, l2=7, lr_power=-1)
    self.assertAllClose(0.8 * np.ones((1, 3, 2)), linear)

  def testVar(self):
    # [FUNCTION DOC]
    # Applies a step where the linear term exceeds l1 and checks the updated variable equals
    # minus 0.5 everywhere.
    """Test computation of var with linear=1.5, quadratic=1."""
    var, _, _ = self._eval(
        var=np.ones((1, 3, 2)),
        accum=np.ones((1, 3, 2)),
        linear=np.zeros((1, 3, 2)),
        grad=np.ones((1, 3, 2)),
        lr=1, l1=1, l2=0.25, lr_power=1)
    self.assertAllClose(-0.5 * np.ones((1, 3, 2)), var)

  def testVarClipped(self):
    # [FUNCTION DOC]
    # Applies a step with l1 larger than the magnitude of the linear term and checks the
    # variable is clipped to zero everywhere.
    """Test that var becomes 0 if |linear| < l1."""
    var, _, _ = self._eval(
        var=np.ones((1, 3, 2)),
        accum=np.ones((1, 3, 2)),
        linear=np.zeros((1, 3, 2)),
        grad=np.ones((1, 3, 2)),
        lr=1, l1=1.6, l2=0.25, lr_power=1)
    self.assertAllClose(np.zeros((1, 3, 2)), var)

  def testQuadratic(self):
    # [FUNCTION DOC]
    # Applies a step with a negative l2 so the quadratic divisor is negative; checks the
    # variable equals 0.25 everywhere.
    """Test that quadratic (here: -2) is the divisor of var."""
    var, _, _ = self._eval(
        var=np.ones((1, 3, 2)),
        accum=np.ones((1, 3, 2)),
        linear=np.zeros((1, 3, 2)),
        grad=np.ones((1, 3, 2)),
        lr=1, l1=1, l2=-1.25, lr_power=1)
    self.assertAllClose(0.25 * np.ones((1, 3, 2)), var)

  def testL2Shrinkage(self):
    # [FUNCTION DOC]
    # Runs the v2 op (nonzero l2 shrinkage) with zero gradient and accumulator; checks the
    # accumulator stays zero, i.e. the shrinkage term does not enter the squared gradient.
    """Test that 2 * l2_shrinkage * var is *not* added to the gradient."""
    _, accum, _ = self._eval(
        var=np.ones((1, 3, 2)),
        accum=np.zeros((1, 3, 2)),
        linear=np.zeros((1, 3, 2)),
        grad=np.zeros((1, 3, 2)),
        lr=7, l1=3, l2=7, lr_power=2, l2_shrinkage=0.5)
    self.assertAllClose(np.zeros((1, 3, 2)), accum)

  def testL2ShrinkageOnLinear(self):
    # [FUNCTION DOC]
    # Runs the v2 op with shrinkage 11 and zero gradient on a unit variable; checks the linear
    # slot equals 22 everywhere (twice shrinkage times variable).
    """Test that 2 * l2_shrinkage * var is added to linear."""
    _, _, linear = self._eval(
        var=np.ones((1, 3, 2)),
        accum=np.zeros((1, 3, 2)),
        linear=np.zeros((1, 3, 2)),
        grad=np.zeros((1, 3, 2)),
        lr=2, l1=3, l2=7, lr_power=0, l2_shrinkage=11)
    self.assertAllClose(22 * np.ones((1, 3, 2)), linear)

  def testMultiplyLinearByLR(self):
    # [FUNCTION DOC]
    # Runs the op with the multiply-linear-by-learning-rate flag on (learning rate six) and
    # checks the linear slot equals 7 everywhere.
    """Test multiply_linear_by_lr = true for the linear variable."""
    _, _, linear = self._eval(
        var=np.zeros((1, 3, 2)),
        accum=np.zeros((1, 3, 2)),
        linear=np.ones((1, 3, 2)),
        grad=np.ones((1, 3, 2)),
        lr=6, l1=1, l2=-1.25, lr_power=0,
        multiply_linear_by_lr=True)
    self.assertAllClose(7 * np.ones((1, 3, 2)), linear)

  def testMultiplyLinearByLRClipping(self):
    # [FUNCTION DOC]
    # Runs the op with the multiply flag on and learning rate three; checks the variable equals
    # minus 0.25 everywhere, showing the clip margins scale with the learning rate.
    """Test that multiply_linear_by_lr = true scales the clip margins."""
    var, _, _ = self._eval(
        var=np.ones((1, 3, 2)),
        accum=np.ones((1, 3, 2)),
        linear=np.zeros((1, 3, 2)),
        grad=np.ones((1, 3, 2)),
        lr=3, l1=1.0, l2=0.25, lr_power=1,
        multiply_linear_by_lr=True)
    self.assertAllClose(-0.25 * np.ones((1, 3, 2)), var)

  def testMultiplyLinearByLRClipZero(self):
    # [FUNCTION DOC]
    # Runs the op with the multiply flag on and l1 large enough that the linear term is inside
    # the clip margin; checks the variable is zero everywhere.
    """Test that multiply_linear_by_lr = true still clips to 0."""
    var, _, _ = self._eval(
        var=np.ones((1, 3, 2)),
        accum=np.ones((1, 3, 2)),
        linear=np.zeros((1, 3, 2)),
        grad=np.ones((1, 3, 2)),
        lr=3, l1=1.2, l2=0.25, lr_power=1,
        multiply_linear_by_lr=True)
    self.assertAllClose(np.zeros((1, 3, 2)), var)


if __name__ == "__main__":
  googletest.main()
