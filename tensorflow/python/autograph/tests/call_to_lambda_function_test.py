# [MODULE DOC]
# Reference test for autograph handling of calls to lambda functions, both defined inside the
# converted function and passed in from outside. Native and compiled results are compared for a
# Python number and a tensor input. No import-time side effects.
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
"""Simple call to lambda functions."""

import tensorflow.compat.v1 as tf

from tensorflow.python.autograph.tests import reference_test_base


def inline_lambda(x):
  # [FUNCTION DOC]
  # Function under test: defines a lambda that squares positive inputs and negates others, and
  # returns the lambda applied to x.
  l = lambda x: x * x if x > 0 else -x
  return l(x)


def external_lambda(x, l):
  # [FUNCTION DOC]
  # Function under test: calls the callable l (supplied by the caller) with x and returns the
  # result.
  return l(x)


class ReferenceTest(reference_test_base.TestCase):
  # [CLASS DOC]
  # Reference test case for calls to lambda functions.

  def test_inline(self):
    # [FUNCTION DOC]
    # Asserts the function with an inline lambda matches eager results for a number and a tensor.
    self.assertFunctionMatchesEager(inline_lambda, 1)
    self.assertFunctionMatchesEager(inline_lambda, tf.constant(1))

  def test_external(self):
    # [FUNCTION DOC]
    # Asserts the function calling a lambda passed as an argument (an equals-zero check) matches
    # eager results for a number and a tensor.
    self.assertFunctionMatchesEager(external_lambda, 1, lambda x: x == 0)
    self.assertFunctionMatchesEager(
        external_lambda, tf.constant(1), lambda x: x == 0)

if __name__ == '__main__':
  tf.test.main()
