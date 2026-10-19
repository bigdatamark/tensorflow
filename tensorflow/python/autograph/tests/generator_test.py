# [MODULE DOC]
# Tests that AutoGraph rejects Python generators. Defines three generator functions (plain,
# inside a for loop, inside a while loop) and a test class asserting tf.function raises a
# not-implemented error mentioning generators. No import-time side effects.
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
"""Generators."""

import tensorflow as tf

from tensorflow.python.autograph.tests import reference_test_base


def basic_generator():
  # [FUNCTION DOC]
  # HELPER: test subject for LoopControlFlowTest.
  # Generator yielding the single value one.
  yield 1


def generator_in_for(n):
  # [FUNCTION DOC]
  # HELPER: test subject for LoopControlFlowTest.
  # Generator yielding each integer from zero up to n (exclusive), using a for loop.
  for i in range(n):
    yield i


def generator_in_while(n):
  # [FUNCTION DOC]
  # HELPER: test subject for LoopControlFlowTest.
  # Generator that counts from one up to n inclusive using a while loop, yielding each value.
  i = 0
  while i < n:
    i += 1
    yield i


class LoopControlFlowTest(reference_test_base.TestCase):
  # [CLASS DOC]
  # Test case (named for loops but covering generators) asserting generator functions are
  # unsupported by AutoGraph conversion.

  def test_basic_generator(self):
    # [FUNCTION DOC]
    # Converting the plain generator via tf.function and calling it must raise a not-implemented
    # error mentioning generators.
    with self.assertRaisesRegex(NotImplementedError, 'generators'):
      tf.function(basic_generator)()

  def test_generator_in_for(self):
    # [FUNCTION DOC]
    # Calls the for-loop generator through tf.function with an empty list as n; expects a
    # not-implemented error mentioning generators.
    with self.assertRaisesRegex(NotImplementedError, 'generators'):
      tf.function(generator_in_for)([])

  def test_generator_in_while(self):
    # [FUNCTION DOC]
    # Calls the while-loop generator through tf.function with zero; expects a not-implemented
    # error mentioning generators.
    with self.assertRaisesRegex(NotImplementedError, 'generators'):
      tf.function(generator_in_while)(0)


if __name__ == '__main__':
  tf.test.main()

