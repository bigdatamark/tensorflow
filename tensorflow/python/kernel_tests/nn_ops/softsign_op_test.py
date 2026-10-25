# [MODULE DOC]
# Unit tests for the softsign activation (nn_ops.softsign) and its gradient. Covers numeric
# agreement with a NumPy reference for several float dtypes on CPU and GPU requests, a numeric
# gradient check, and rejection of integer inputs. Importing nn_grad registers the gradient
# functions. Running as a script starts the TensorFlow test runner.
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
"""Tests for Softsign and SoftsignGrad."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


class SoftsignTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the softsign op: forward values against a NumPy reference, numeric
  # gradient accuracy, and dtype validation. Stateless; each test builds its own session.

  def _npSoftsign(self, np_features):
    # [FUNCTION DOC]
    # HELPER: NumPy reference implementation used by the softsign value tests.
    # Takes an array of features and returns an array of the same shape and dtype computed as the
    # input divided by one plus its absolute value. No side effects.
    return np_features / (1 + np.abs(np_features))

  def _testSoftsign(self, np_features, atol, rtol, use_gpu=False):
    # [FUNCTION DOC]
    # HELPER: shared checker for the forward-value test.
    # Inputs: a NumPy array of features, absolute and relative tolerances, and a flag asking for a
    # GPU-enabled session (default false). Computes the reference result, evaluates softsign in a
    # cached session, and asserts the values are close within the tolerances and that the static
    # shape of the op output matches the reference. Returns nothing; raises on assertion failure.
    np_softsign = self._npSoftsign(np_features)
    with self.cached_session(use_gpu=use_gpu):
      softsign = nn_ops.softsign(np_features)
      tf_softsign = self.evaluate(softsign)

    self.assertAllClose(np_softsign, tf_softsign, rtol=rtol, atol=atol)
    self.assertShapeEqual(np_softsign, softsign)

  def testNumbers(self):
    # [FUNCTION DOC]
    # Scenario: softsign on a fixed 2x5 array of odd integers cast to float16, float32, float64
    # and bfloat16, each requested once without and once with GPU use. Asserts agreement with the
    # reference using tight tolerances (1e-6) for the standard floats and looser (1e-2) for
    # bfloat16.
    for t, atol, rtol in [
        (np.float16, 1e-6, 1e-6),
        (np.float32, 1e-6, 1e-6),
        (np.float64, 1e-6, 1e-6),
        (dtypes.bfloat16.as_numpy_dtype, 1e-2, 1e-2),
    ]:
      self._testSoftsign(
          np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
          atol,
          rtol,
          use_gpu=False)
      self._testSoftsign(
          np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
          atol,
          rtol,
          use_gpu=True)

  @test_util.run_deprecated_v1
  def testGradient(self):
    # [FUNCTION DOC]
    # Scenario: builds a 2x5 float constant in the range -0.9..0.9, applies softsign, and computes
    # the maximum error between the analytic and numeric Jacobians (using a Fortran-ordered
    # float32 initial value). Prints the error and asserts it is below 1e-4. Graph-mode only
    # (run as deprecated v1).
    with self.cached_session():
      x = constant_op.constant(
          [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9],
          shape=[2, 5],
          name="x")
      y = nn_ops.softsign(x, name="softsign")
      x_init = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float32,
          order="F")
      err = gradient_checker.compute_gradient_error(
          x, [2, 5], y, [2, 5], x_init_value=x_init)
    print("softsign (float) gradient err = ", err)
    self.assertLess(err, 1e-4)

  @test_util.run_deprecated_v1
  def testNoInts(self):
    # [FUNCTION DOC]
    # Scenario: softsign applied to an int32 scalar constant. Expects a type error whose message
    # says the features argument has data type int32 not in the list of allowed values. Graph-mode
    # only (run as deprecated v1).
    with self.cached_session():
      with self.assertRaisesRegex(
          TypeError,
          "'features' has DataType int32 not in list of allowed values"):
        nn_ops.softsign(constant_op.constant(7)).eval()


if __name__ == "__main__":
  test.main()
