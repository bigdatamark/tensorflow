# [MODULE DOC]
# Reference tests for variables created inside AutoGraph while loops (static/dynamic shape and
# rank, plain and nested/dictionary values). Verifies that zero-iteration tensor loops raise
# invalid-argument errors, zero-iteration Python loops raise unbound-local errors, and that
# loops that iterate match eager results. No import-time side effects.
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
"""Loops which create variables, with or without shape invariants."""

import itertools

from absl.testing import parameterized
import tensorflow as tf

from tensorflow.python.autograph.tests import reference_test_base


# TODO(mdan): When syntax allows shape_invariants on created vars, add tests.


def while_creates_var_static_shape(n):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # While i is below n, creates v as a zeros tensor of fixed shape. Returns v.
  i = 0
  while i < n:
    v = tf.zeros([1, 2, 3])
    i += 1
  return v


def while_creates_var_dynamic_shape(n):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Creates v as zeros whose middle dimension is a random tensor value (equal to i). Returns v.
  i = 0
  while i < n:
    v = tf.zeros([1, tf.random.uniform((), i, i + 1, tf.int32), 2])
    i += 1
  return v


def while_creates_var_dynamic_rank(n):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Creates v as zeros whose shape is a range of random length, so the rank varies. Returns v.
  i = 0
  while i < n:
    v = tf.zeros(tf.range(tf.random.uniform((), i, i + 1, tf.int32)))
    i += 1
  return v


def while_creates_var_dynamic_shape_py_init_var(n):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Creates v as a range up to i, whose length changes each iteration. Returns v.
  i = 0
  while i < n:
    v = tf.range(i)
    i += 1
  return v


def while_creates_nested_var_static_shape(n):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Creates a mapping of two fixed-shape tensors (zeros and ones). Returns both values.
  i = 0
  while i < n:
    v = {'a': tf.zeros([1, 2, 3]), 'b': tf.ones([1, 2, 3])}
    i += 1
  return v['a'], v['b']


def while_creates_nested_var_dynamic_shape(n):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Creates a mapping of two tensors with dynamically determined dimensions. Returns both values.
  i = 0
  while i < n:
    v = {
        'a': tf.zeros([1, tf.random.uniform((), i, i + 1, tf.int32)]),
        'b': tf.ones([tf.random.uniform((), i, i + 1, tf.int32), 2])
    }
    i += 1
  return v['a'], v['b']


def while_creates_nested_var_dynamic_rank(n):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Creates a mapping with a tensor of dynamic rank and a fixed one. Returns both values.
  i = 0
  while i < n:
    v = {
        'a': tf.ones(tf.range(tf.random.uniform((), i, i + 1, tf.int32))),
        'b': tf.ones([1, 2, 3])
    }
    i += 1
  return v['a'], v['b']


class ReferenceTest(reference_test_base.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case for loop-created variables. No shared fixtures.

  @parameterized.parameters(
      while_creates_var_static_shape,
      while_creates_var_dynamic_shape,
      while_creates_var_dynamic_rank,
      while_creates_var_dynamic_shape_py_init_var,
      while_creates_nested_var_static_shape,
      while_creates_nested_var_dynamic_shape,
      while_creates_nested_var_dynamic_rank,
  )
  def test_while_creates_var_illegal_tf(self, target):
    # [FUNCTION DOC]
    # Runs each subject with a zero tensor and expects an invalid-argument error.
    with self.assertRaises(tf.errors.InvalidArgumentError):
      tf.function(target)(tf.constant(0))

  @parameterized.parameters(
      while_creates_var_static_shape,
      while_creates_var_dynamic_shape,
      while_creates_var_dynamic_rank,
      while_creates_var_dynamic_shape_py_init_var,
      while_creates_nested_var_static_shape,
      while_creates_nested_var_dynamic_shape,
      while_creates_nested_var_dynamic_rank,
  )
  def test_while_creates_var_illegal_py(self, target):
    # [FUNCTION DOC]
    # Runs each subject with plain zero and expects an unbound-local error.
    with self.assertRaises(UnboundLocalError):
      tf.function(target)(0)

  @parameterized.parameters(*itertools.product(
      (1, 2),
      (int, tf.constant),
      (
          while_creates_var_static_shape,
          while_creates_var_dynamic_shape,
          while_creates_var_dynamic_rank,
          while_creates_var_dynamic_shape_py_init_var,
          while_creates_nested_var_static_shape,
          while_creates_nested_var_dynamic_shape,
          while_creates_nested_var_dynamic_rank,
      ),
  ))
  def test_while_creates_var(self, n, type_, target):
    # [FUNCTION DOC]
    # With n of 1 or 2 as int or tensor, asserts graph and eager results match for every subject.
    n = type_(n)
    self.assertFunctionMatchesEager(target, n)


if __name__ == '__main__':
  tf.test.main()
