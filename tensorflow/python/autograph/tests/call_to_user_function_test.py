# [MODULE DOC]
# Reference test for autograph handling of calls to user-defined functions that are resolved
# dynamically: functions held in variables, received as parameters, returned from factories,
# called with keyword arguments, and an allowlisted TensorFlow function held in a variable.
# Native and compiled results are compared with Python and tensor inputs. No import-time side
# effects.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Calls to dynamic (i.e. nonglobal) functions.

Examples:
 * function variables
 * function parameters
 * factories
"""

import tensorflow as tf

from tensorflow.python.autograph.tests import reference_test_base


def function_1(x):
  # [FUNCTION DOC]
  # HELPER: sample callee used by the dynamic-call samples. Returns x cubed.
  return x * x * x


def function_2(x):
  # [FUNCTION DOC]
  # HELPER: sample callee used by the dynamic-call samples. Returns eleven minus x.
  return -1 * x + 11


def factory(n):
  # [FUNCTION DOC]
  # HELPER: function factory for the dynamic-call samples. Returns the first sample callee when
  # n equals one, otherwise the second.
  if n == 1:
    return function_1
  return function_2


def static_fn(x):
  # [FUNCTION DOC]
  # Function under test: calls both sample callees directly by name and returns the sum.
  a = function_1(x)
  b = function_2(x)
  return a + b


def factory_dynamic_fn(x):
  # [FUNCTION DOC]
  # Function under test: obtains each callee from the factory (arguments one and two), calls it
  # on x, and returns the sum of both results.
  f = factory(1)
  a = f(x)
  f = factory(2)
  b = f(x)
  return a + b


def param_dynamic_fn(f, x):
  # [FUNCTION DOC]
  # Function under test: calls the callable parameter f with x and returns the result.
  return f(x)


def variable_dynamic_fn(x):
  # [FUNCTION DOC]
  # Function under test: stores each sample callee in a variable in turn, calls it on x, and
  # returns the sum of both results.
  f = function_1
  a = f(x)
  f = function_2
  b = f(x)
  return a + b


def variable_dynamic_whitelisted_fn(x):
  # [FUNCTION DOC]
  # Function under test: stores the TensorFlow identity function in a variable, calls it on x
  # and returns the result (an allowlisted function called dynamically).
  f = tf.identity
  return f(x)


def dynamic_fn_with_kwargs(f, x):
  # [FUNCTION DOC]
  # Function under test: calls the callable f with x passed as the keyword argument named x and
  # returns the result.
  return f(x=x)


class ReferenceTest(reference_test_base.TestCase):
  # [CLASS DOC]
  # Reference test case for calls to dynamically resolved user functions.

  def test_basic(self):
    # [FUNCTION DOC]
    # Asserts all six samples match eager results using plain Python inputs.
    self.assertFunctionMatchesEager(static_fn, 1)
    self.assertFunctionMatchesEager(factory_dynamic_fn, 1)
    self.assertFunctionMatchesEager(param_dynamic_fn, function_1, 1)
    self.assertFunctionMatchesEager(variable_dynamic_fn, 1)
    self.assertFunctionMatchesEager(variable_dynamic_whitelisted_fn, 1)
    self.assertFunctionMatchesEager(dynamic_fn_with_kwargs, function_1, 1)

  def test_basic_tensor(self):
    # [FUNCTION DOC]
    # Same checks as the basic test, but with all inputs converted to tensors first.
    self.all_inputs_tensors = True
    self.assertFunctionMatchesEager(static_fn, 1)
    self.assertFunctionMatchesEager(factory_dynamic_fn, 1)
    self.assertFunctionMatchesEager(param_dynamic_fn, function_1, 1)
    self.assertFunctionMatchesEager(variable_dynamic_fn, 1)
    self.assertFunctionMatchesEager(variable_dynamic_whitelisted_fn, 1)
    self.assertFunctionMatchesEager(dynamic_fn_with_kwargs, function_1, 1)


if __name__ == '__main__':
  tf.test.main()
