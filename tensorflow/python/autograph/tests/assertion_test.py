# [MODULE DOC]
# Reference test for autograph conversion of assert statements: a function with an assertion is
# run natively and converted, and results and failure types are compared. Uses the shared
# reference test base class. No import-time side effects.
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
"""Basic assertions."""

import tensorflow as tf

from tensorflow.python.autograph.tests import reference_test_base


def simple_assertion(x):
  # [FUNCTION DOC]
  # Function under test: asserts x is greater than zero and returns x. Raises an assertion error
  # in plain Python, or a graph-time invalid-argument error when x is a tensor and converted.
  assert x > 0
  return x


class ReferenceTest(reference_test_base.TestCase):
  # [CLASS DOC]
  # Reference test case enabling only the assert-statement conversion feature.

  def setUp(self):
    # [FUNCTION DOC]
    # Runs the base setup, then sets the autograph options to enable only conversion of assert
    # statements.
    super(ReferenceTest, self).setUp()
    self.autograph_opts = tf.autograph.experimental.Feature.ASSERT_STATEMENTS

  def test_basic(self):
    # [FUNCTION DOC]
    # Checks the function matches eager behavior for a passing Python number and a passing
    # tensor, then checks a failing Python number raises an assertion error and a failing tensor
    # raises a TensorFlow invalid-argument error when run through the compiled function.
    self.assertFunctionMatchesEager(simple_assertion, 1)
    self.assertFunctionMatchesEager(simple_assertion, tf.constant(1))
    with self.assertRaises(AssertionError):
      self.function(simple_assertion)(0)
    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.function(simple_assertion)(tf.constant(0))


if __name__ == '__main__':
  tf.test.main()
