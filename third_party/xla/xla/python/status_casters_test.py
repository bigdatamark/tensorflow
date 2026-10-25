# [MODULE DOC]
# Unit tests and usage examples for the status conversion helpers (status_casters) that wrap
# C++ callables returning a status or a status-or-value so that Python sees nothing/the value and
# gets an exception on error. The module imports a small test extension (status_casters_ext)
# that exposes sample free functions, a sample class and methods, and checks the wrapped results.
# Run directly it starts the absltest runner.
# Copyright 2017 The OpenXLA Authors.
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
"""Tests and examples for status_casters.h."""

from absl.testing import absltest

from xla.python import status_casters_ext


class StatusCastersTest(absltest.TestCase):
  # [CLASS DOC]
  # Test case covering the status/status-or wrappers exposed by the test extension. It has no
  # fixtures; each test calls extension functions and methods directly and checks the results.

  def test_status_wrappers(self):
    # [FUNCTION DOC]
    # Test: status-returning wrappers. Calls two wrapped sample functions and the plain and
    # const-qualified methods of the sample class (with arguments 1 and 2) and asserts each
    # returns nothing (the null value), i.e. the OK status is converted to no value.
    self.assertIsNone(status_casters_ext.my_lambda())
    self.assertIsNone(status_casters_ext.my_lambda2())

    self.assertIsNone(status_casters_ext.MyClass().my_method(1, 2))
    self.assertIsNone(status_casters_ext.MyClass().my_method_const(1, 2))

  def test_status_or_wrappers(self):
    # [FUNCTION DOC]
    # Test: status-or-value wrappers. Asserts the wrapped sample function returns 1, the identity
    # function returns its argument (2), and the plain and const methods of the sample class,
    # called with 1 and 2, return 3, i.e. the value inside a successful status-or is unwrapped.
    self.assertEqual(status_casters_ext.my_lambda_statusor(), 1)
    self.assertEqual(status_casters_ext.status_or_identity(2), 2)

    self.assertEqual(status_casters_ext.MyClass().my_method_status_or(1, 2), 3)
    self.assertEqual(
        status_casters_ext.MyClass().my_method_status_or_const(1, 2), 3
    )


if __name__ == "__main__":
  absltest.main()
