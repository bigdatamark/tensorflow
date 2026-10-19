# [MODULE DOC]
# Unit tests for the Identity bijector in the distributions ops package (forward and inverse maps
# are the identity, log determinant of the Jacobian is zero). Contains one test case class with a
# direct behavior test and a scalar congruency test using the shared bijector test utilities.
# No import-time side effects. Runs via the test main entry point.
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
"""Identity Tests."""

from tensorflow.python.framework import test_util
from tensorflow.python.ops.distributions import bijector_test_util
from tensorflow.python.ops.distributions import identity_bijector
from tensorflow.python.platform import test


class IdentityBijectorTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the Identity bijector. No shared fixtures; each test builds its own bijector.
  """Tests correctness of the Y = g(X) = X transformation."""

  def testBijector(self):
    # [FUNCTION DOC]
    # Builds an Identity bijector with argument validation on. Asserts its name is identity, that
    # forward and inverse of a nested list return the input unchanged, and that both the inverse
    # and forward log determinant of the Jacobian (event rank three) evaluate to zero.
    bijector = identity_bijector.Identity(validate_args=True)
    self.assertEqual("identity", bijector.name)
    x = [[[0.], [1.]]]
    self.assertAllEqual(x, self.evaluate(bijector.forward(x)))
    self.assertAllEqual(x, self.evaluate(bijector.inverse(x)))
    self.assertAllEqual(
        0.,
        self.evaluate(
            bijector.inverse_log_det_jacobian(x, event_ndims=3)))
    self.assertAllEqual(
        0.,
        self.evaluate(
            bijector.forward_log_det_jacobian(x, event_ndims=3)))

  @test_util.run_deprecated_v1
  def testScalarCongruency(self):
    # [FUNCTION DOC]
    # Runs the shared scalar congruency check from the bijector test utilities on an Identity
    # bijector over the range from minus two to two, inside a cached session (graph mode only,
    # via the deprecated-v1 decorator). Passes if the utility's assertions hold.
    with self.cached_session():
      bijector = identity_bijector.Identity()
      bijector_test_util.assert_scalar_congruency(
          bijector, lower_x=-2., upper_x=2.)


if __name__ == "__main__":
  test.main()
