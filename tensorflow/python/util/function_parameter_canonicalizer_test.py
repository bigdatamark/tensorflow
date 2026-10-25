# [MODULE DOC]
# Test module for the native function-parameter canonicalizer, reached through a test-only
# binding. Verifies conversion of positional and keyword arguments into a full argument list and
# the error messages for missing, extra, unexpected and duplicated arguments.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for `tensorflow::FunctionParameterCanonicalizer`."""

from tensorflow.python.platform import test
from tensorflow.python.util import _function_parameter_canonicalizer_binding_for_test


class FunctionParameterCanonicalizerTest(test.TestCase):
  # [CLASS DOC]
  # Tests the native function-parameter canonicalizer (exposed through a test-only binding)
  # that turns positional and keyword arguments into a full positional list using defaults.
  # Shared fixture: a canonicalizer modelled on a matrix-multiply signature with nine parameter
  # names where the last seven have defaults.

  def setUp(self):
    # [FUNCTION DOC]
    # Calls the parent setUp, then builds the shared canonicalizer for nine parameters (a, b,
    # transpose_a, transpose_b, adjoint_a, adjoint_b, a_is_sparse, b_is_sparse, name) with seven
    # defaults: six false values and a nothing value for name. Stored on the instance.
    super(FunctionParameterCanonicalizerTest, self).setUp()
    self._matmul_func = (
        _function_parameter_canonicalizer_binding_for_test
        .FunctionParameterCanonicalizer([
            'a', 'b', 'transpose_a', 'transpose_b', 'adjoint_a', 'adjoint_b',
            'a_is_sparse', 'b_is_sparse', 'name'
        ], (False, False, False, False, False, False, None)))

  def testPosOnly(self):
    # [FUNCTION DOC]
    # Scenario: pass only the two required positional arguments.
    # Asserts the result is the full list with defaults filled in for the rest.
    self.assertEqual(
        self._matmul_func.canonicalize(2, 3),
        [2, 3, False, False, False, False, False, False, None])

  def testPosOnly2(self):
    # [FUNCTION DOC]
    # Scenario: pass five positional arguments, overriding the first three of the defaults.
    # Asserts the result reflects the given values and defaults for the remaining parameters.
    self.assertEqual(
        self._matmul_func.canonicalize(2, 3, True, False, True),
        [2, 3, True, False, True, False, False, False, None])

  def testPosAndKwd(self):
    # [FUNCTION DOC]
    # Scenario: two positional arguments plus keyword arguments for transpose_a and name.
    # Asserts keywords are placed at their parameter positions and other defaults are kept.
    self.assertEqual(
        self._matmul_func.canonicalize(
            2, 3, transpose_a=True, name='my_matmul'),
        [2, 3, True, False, False, False, False, False, 'my_matmul'])

  def testPosAndKwd2(self):
    # [FUNCTION DOC]
    # Scenario: one positional argument plus the second required parameter given by keyword.
    # Asserts the required parameter is filled from the keyword and defaults fill the rest.
    self.assertEqual(
        self._matmul_func.canonicalize(2, b=3),
        [2, 3, False, False, False, False, False, False, None])

  def testMissingPos(self):
    # [FUNCTION DOC]
    # Scenario: supply only one of the two required positional arguments.
    # Asserts a type error mentioning a missing required positional argument.
    with self.assertRaisesRegex(TypeError,
                                'Missing required positional argument'):
      self._matmul_func.canonicalize(2)

  def testMissingPos2(self):
    # [FUNCTION DOC]
    # Scenario: supply only optional parameters by keyword, omitting both required ones.
    # Asserts a type error mentioning a missing required positional argument.
    with self.assertRaisesRegex(TypeError,
                                'Missing required positional argument'):
      self._matmul_func.canonicalize(
          transpose_a=True, transpose_b=True, adjoint_a=True)

  def testTooManyArgs(self):
    # [FUNCTION DOC]
    # Scenario: pass ten positional arguments to a nine-parameter canonicalizer.
    # Asserts a type error saying too many arguments were given, expected nine but got ten.
    with self.assertRaisesRegex(TypeError, 'Too many arguments were given.'
                                ' Expected 9 but got 10.'):
      self._matmul_func.canonicalize(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

  def testInvalidKwd(self):
    # [FUNCTION DOC]
    # Scenario: pass a keyword that is not a parameter name.
    # Asserts a type error about an unexpected keyword argument.
    with self.assertRaisesRegex(TypeError,
                                'Got an unexpected keyword argument'):
      self._matmul_func.canonicalize(2, 3, hohoho=True)

  def testDuplicatedArg(self):
    # [FUNCTION DOC]
    # Scenario: give the second parameter both positionally and by keyword.
    # Asserts a type error about multiple values for that parameter.
    with self.assertRaisesRegex(TypeError,
                                "Got multiple values for argument 'b'"):
      self._matmul_func.canonicalize(2, 3, False, b=4)

  def testDuplicatedArg2(self):
    # [FUNCTION DOC]
    # Scenario: give the third parameter both positionally and by keyword.
    # Asserts a type error about multiple values for that parameter.
    with self.assertRaisesRegex(
        TypeError, "Got multiple values for argument 'transpose_a'"):
      self._matmul_func.canonicalize(2, 3, False, transpose_a=True)

  def testKwargNotInterned(self):
    # [FUNCTION DOC]
    # Scenario: build a one-parameter canonicalizer and call it with a keyword whose name string
    # is constructed at runtime (so it is not an interned string).
    # Asserts only that the call succeeds, i.e. matching does not depend on string identity.
    func = (
        _function_parameter_canonicalizer_binding_for_test
        .FunctionParameterCanonicalizer(['long_parameter_name'], ()))
    kwargs = dict([('_'.join(['long', 'parameter', 'name']), 5)])
    func.canonicalize(**kwargs)


if __name__ == '__main__':
  test.main()
