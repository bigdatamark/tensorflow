# [MODULE DOC]
# Reference tests for AutoGraph loops whose tensor loop variables legitimately change shape,
# using the loop option that declares shape invariants. Subject functions grow vectors and
# matrices (rows, columns, both), use shape invariants inside conditionals, with break, with
# nested named tuples and with sparse (composite) tensors, in both while and for forms.
# One parameterized test class compares tf.function results with eager execution.
# No import-time side effects; runs the TF test main when executed directly.
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
"""Loops with type changing variables."""

import collections
import itertools

from absl.testing import parameterized
import tensorflow as tf

from tensorflow.python.autograph.tests import reference_test_base


def while_with_variable_shape_growing_vector(n):
  # [FUNCTION DOC]
  # Test subject: while i < n, declares an unknown-length shape invariant for vector v and
  # appends i to v. Returns v (initially two zeros).
  v = tf.constant([0, 0])
  i = 0
  while i < n:
    tf.autograph.experimental.set_loop_options(
        shape_invariants=[(v, tf.TensorShape([None]))])
    v = tf.concat((v, [i]), 0)
    i += 1
  return v


def for_with_variable_shape_growing_vector(l):
  # [FUNCTION DOC]
  # Test subject: for each element of l, declares the unknown-length invariant and appends the
  # element to vector v. Returns v.
  v = tf.constant([0, 0])
  for i in l:
    tf.autograph.experimental.set_loop_options(
        shape_invariants=[(v, tf.TensorShape([None]))])
    v = tf.concat((v, [i]), 0)
  return v


def while_with_variable_shape_growing_matrix_rows(n):
  # [FUNCTION DOC]
  # Test subject: while i < n, declares an invariant of unknown row count and one column and
  # appends a row containing i to matrix m. Returns m.
  m = tf.constant([[0]])
  i = 0
  while i < n:
    tf.autograph.experimental.set_loop_options(
        shape_invariants=[(m, tf.TensorShape([None, 1]))])
    m = tf.concat((m, [[i]]), 0)
    i += 1
  return m


def for_with_variable_shape_growing_matrix_rows(l):
  # [FUNCTION DOC]
  # Test subject: for each element of l, appends a row to matrix m under an unknown-rows
  # invariant. Returns m.
  m = tf.constant([[0]])
  for i in l:
    tf.autograph.experimental.set_loop_options(
        shape_invariants=[(m, tf.TensorShape([None, 1]))])
    m = tf.concat((m, [[i]]), 0)
  return m


def while_with_variable_shape_growing_matrix_cols(n):
  # [FUNCTION DOC]
  # Test subject: while i < n, appends a column holding i to a one-row matrix m under an
  # invariant of one row and unknown columns. Returns m.
  m = tf.constant([[0, 0]])
  i = 0
  while i < n:
    tf.autograph.experimental.set_loop_options(
        shape_invariants=[(m, tf.TensorShape([1, None]))])
    m = tf.concat((m, [[i]]), 1)
    i += 1
  return m


def for_with_variable_shape_growing_matrix_cols(l):
  # [FUNCTION DOC]
  # Test subject: for each element of l, appends a column to a one-row matrix under an
  # unknown-columns invariant. Returns the matrix.
  m = tf.constant([[0, 0]])
  for i in l:
    tf.autograph.experimental.set_loop_options(
        shape_invariants=[(m, tf.TensorShape([1, None]))])
    m = tf.concat((m, [[i]]), 1)
  return m


def while_with_variable_shape_growing_matrix(n):
  # [FUNCTION DOC]
  # Test subject: while i < n, pads the 2x2 matrix m with a border of value i on all sides, under
  # a fully unknown shape invariant. Returns m.
  m = tf.constant([[0, 0], [0, 0]])
  i = 0
  while i < n:
    tf.autograph.experimental.set_loop_options(
        shape_invariants=[(m, tf.TensorShape(None))])
    m = tf.pad(m, [[1, 1], [1, 1]], constant_values=i)
    i += 1
  return m


def for_with_variable_shape_growing_matrix(l):
  # [FUNCTION DOC]
  # Test subject: for each element of l, pads matrix m with a border of that value under a fully
  # unknown shape invariant. Returns m.
  m = tf.constant([[0, 0], [0, 0]])
  for i in l:
    tf.autograph.experimental.set_loop_options(
        shape_invariants=[(m, tf.TensorShape(None))])
    m = tf.pad(m, [[1, 1], [1, 1]], constant_values=i)
  return m


def while_with_variable_shape_inside_if(n):
  # [FUNCTION DOC]
  # Test subject: if n is greater than one, runs a growing-vector while loop; otherwise assigns
  # a fixed three-element vector. Returns v.
  v = tf.constant([0, 0])
  i = 0
  if n > 1:
    while i < n:
      tf.autograph.experimental.set_loop_options(
          shape_invariants=[(v, tf.TensorShape([None]))])
      v = tf.concat((v, [i]), 0)
      i += 1
  else:
    v = tf.constant([1, 2, 3])
  return v


def for_with_variable_shape_inside_if(n):
  # [FUNCTION DOC]
  # Test subject: if n is greater than one, loops over a range of n growing vector v with an
  # unknown-length invariant; otherwise assigns a fixed vector. Returns v. (The statement that
  # increments the loop variable has no effect on the iteration.)
  v = tf.constant([0, 0])
  if n > 1:
    for i in range(n):
      tf.autograph.experimental.set_loop_options(
          shape_invariants=[(v, tf.TensorShape([None]))])
      v = tf.concat((v, [i]), 0)
      i += 1
  else:
    v = tf.constant([1, 2, 3])
  return v


def for_with_nested_variable_shape_inside_if(n):
  # [FUNCTION DOC]
  # Test subject: like the for-inside-if subject, but also updates a named tuple t whose field
  # is incremented per iteration (or replaced in the else branch). Returns only v; t exercises
  # nested structures in the conditional.
  Test = collections.namedtuple('Test', ['var'])
  t = Test(var=tf.constant([0]))
  v = tf.constant([0, 0])
  if n > 1:
    for i in range(n):
      tf.autograph.experimental.set_loop_options(
          shape_invariants=[(v, tf.TensorShape([None]))])
      v = tf.concat((v, [i]), 0)
      t = Test(var=t.var + 1)
      i += 1
  else:
    v = tf.constant([1, 2, 3])
    t = Test(var=tf.constant([3]))
  return v


def while_with_variable_shape_and_break(n):
  # [FUNCTION DOC]
  # Test subject: inside an if on n, a while loop grows vector v and breaks once i exceeds three;
  # otherwise v is a fixed vector. Returns v.
  v = tf.constant([0, 0])
  i = 0
  if n > 1:
    while i < n:
      tf.autograph.experimental.set_loop_options(
          shape_invariants=[(v, tf.TensorShape([None]))])
      v = tf.concat((v, [i]), 0)
      i += 1
      if i > 3:
        break
  else:
    v = tf.constant([1, 2, 3])
  return v


def for_with_variable_shape_and_break(n):
  # [FUNCTION DOC]
  # Test subject: inside an if on n, a for loop over a range grows vector v and breaks once the
  # incremented i exceeds three; otherwise v is fixed. Returns v.
  v = tf.constant([0, 0])
  if n > 1:
    for i in range(n):
      tf.autograph.experimental.set_loop_options(
          shape_invariants=[(v, tf.TensorShape([None]))])
      v = tf.concat((v, [i]), 0)
      i += 1
      if i > 3:
        break
  else:
    v = tf.constant([1, 2, 3])
  return v


def while_with_composite_tensor_shape_invariant(n):
  # [FUNCTION DOC]
  # Test subject: starts with a 3x3 sparse tensor and, while i < n, adds a dimension to it under
  # an unknown-shape invariant. Returns the sparse tensor.
  v = tf.SparseTensor(
      indices=[[0, 0], [1, 1]], values=[1, 2], dense_shape=[3, 3])
  i = 0
  while i < n:
    tf.autograph.experimental.set_loop_options(
        shape_invariants=[(v, tf.TensorShape(None))])
    v = tf.sparse.expand_dims(v)
    i += 1
  return v


def for_with_composite_tensor_shape_invariant(l):
  # [FUNCTION DOC]
  # Test subject: for each element of l, adds a dimension to a sparse tensor under an
  # unknown-shape invariant. Returns the sparse tensor.
  v = tf.SparseTensor(
      indices=[[0, 0], [1, 1]], values=[1, 2], dense_shape=[3, 3])
  for _ in l:
    tf.autograph.experimental.set_loop_options(
        shape_invariants=[(v, tf.TensorShape(None))])
    v = tf.sparse.expand_dims(v)
  return v


def _int_dataset_range(n):
  # [FUNCTION DOC]
  # HELPER: returns a dataset of the integers below n cast to int32; used as an input constructor.
  return tf.data.Dataset.range(n).map(lambda x: tf.cast(x, tf.int32))


class ReferenceTest(reference_test_base.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized tests asserting that loops with declared shape invariants compile with
  # tf.function and give the same results as eager execution, over Python/tensor counters,
  # ranges and datasets.

  @parameterized.parameters(*itertools.product(
      (0, 1, 2),
      (int, tf.constant),
  ))
  def test_while_with_variable_shape_growing_vector(self, n, type_):
    # [FUNCTION DOC]
    # Parameterized over n in 0-2 and converters (int, tf.constant). Asserts the vector-growing
    # while loop matches eager.
    n = type_(n)
    self.assertFunctionMatchesEager(while_with_variable_shape_growing_vector, n)

  @parameterized.parameters(*itertools.product(
      (0, 1, 2),
      (range, tf.range, tf.data.Dataset.range),
  ))
  def test_for_with_variable_shape_growing_vector(self, n, list_type):
    # [FUNCTION DOC]
    # Parameterized over n in 0-2 and iterable constructors (range, tf.range, dataset range).
    # Asserts the vector-growing for loop matches eager.
    l = list_type(n)
    self.assertFunctionMatchesEager(for_with_variable_shape_growing_vector, l)

  @parameterized.parameters(*itertools.product(
      (0, 1, 2),
      (int, tf.constant),
  ))
  def test_while_with_variable_shape_growing_matrix_rows(self, n, type_):
    # [FUNCTION DOC]
    # Parameterized over n and converters (int, tf.constant). Asserts the row-growing while loop
    # matches eager.
    n = type_(n)
    self.assertFunctionMatchesEager(
        while_with_variable_shape_growing_matrix_rows, n)

  @parameterized.parameters(*itertools.product(
      (0, 1, 2),
      (range, tf.range, _int_dataset_range),
  ))
  def test_for_with_variable_shape_growing_matrix_rows(self, l, type_):
    # [FUNCTION DOC]
    # Parameterized over counts and iterable constructors (range, tf.range, int32 dataset range).
    # Asserts the row-growing for loop matches eager.
    l = type_(l)
    self.assertFunctionMatchesEager(
        for_with_variable_shape_growing_matrix_rows, l)

  @parameterized.parameters(*itertools.product(
      (0, 1, 2),
      (int, tf.constant),
  ))
  def test_while_with_variable_shape_growing_matrix_cols(self, n, type_):
    # [FUNCTION DOC]
    # Parameterized over n and converters. Asserts the column-growing while loop matches eager.
    n = type_(n)
    self.assertFunctionMatchesEager(
        while_with_variable_shape_growing_matrix_cols, n)

  @parameterized.parameters(*itertools.product(
      (0, 1, 2),
      (range, tf.range, tf.data.Dataset.range),
  ))
  def test_for_with_variable_shape_growing_matrix_cols(self, l, type_):
    # [FUNCTION DOC]
    # Parameterized over counts and iterable constructors (range, tf.range, dataset range).
    # Asserts the column-growing for loop matches eager.
    l = type_(l)
    self.assertFunctionMatchesEager(
        for_with_variable_shape_growing_matrix_cols, l)

  @parameterized.parameters(*itertools.product(
      (0, 1, 2),
      (int, tf.constant),
  ))
  def test_while_with_variable_shape_growing_matrix(self, n, type_):
    # [FUNCTION DOC]
    # Parameterized over n and converters. Asserts the padding while loop matches eager.
    n = type_(n)
    self.assertFunctionMatchesEager(while_with_variable_shape_growing_matrix, n)

  @parameterized.parameters(*itertools.product(
      (0, 1, 2),
      (range, tf.range, _int_dataset_range),
  ))
  def test_for_with_variable_shape_growing_matrix(self, n, type_):
    # [FUNCTION DOC]
    # Parameterized over counts and iterable constructors (range, tf.range, int32 dataset range).
    # Asserts the padding for loop matches eager.
    l = type_(n)
    self.assertFunctionMatchesEager(for_with_variable_shape_growing_matrix, l)

  @parameterized.parameters(*itertools.product(
      (0, 1, 2),
      (int, tf.constant),
  ))
  def test_while_with_variable_shape_inside_if(self, n, type_):
    # [FUNCTION DOC]
    # Parameterized over n and converters. Asserts the while loop nested in a conditional matches
    # eager.
    n = type_(n)
    self.assertFunctionMatchesEager(while_with_variable_shape_inside_if, n)

  @parameterized.parameters(*itertools.product(
      (0, 1, 2),
      (int, tf.constant),
  ))
  def test_for_with_variable_shape_inside_if(self, n, type_):
    # [FUNCTION DOC]
    # Parameterized over n and converters. Asserts the for loop nested in a conditional matches
    # eager.
    n = type_(n)
    self.assertFunctionMatchesEager(for_with_variable_shape_inside_if, n)

  @parameterized.parameters(*itertools.product(
      (0, 1, 2),
      (int, tf.constant),
  ))
  def test_for_with_nested_variable_shape_inside_if(self, n, type_):
    # [FUNCTION DOC]
    # Parameterized over n and converters. Asserts the subject with a named tuple updated
    # alongside the growing vector matches eager.
    n = type_(n)
    self.assertFunctionMatchesEager(for_with_nested_variable_shape_inside_if, n)

  @parameterized.parameters(*itertools.product(
      (0, 1, 2),
      (int, tf.constant),
  ))
  def test_while_with_variable_shape_and_break(self, n, type_):
    # [FUNCTION DOC]
    # Parameterized over n and converters. Asserts the while loop with break matches eager.
    n = type_(n)
    self.assertFunctionMatchesEager(while_with_variable_shape_and_break, n)

  @parameterized.parameters(*itertools.product(
      (0, 1, 2, 5),
      (int, tf.constant),
  ))
  def test_for_with_variable_shape_and_break(self, n, type_):
    # [FUNCTION DOC]
    # Parameterized over n in 0, 1, 2 and 5 and converters. Asserts the for loop with break
    # matches eager.
    n = type_(n)
    self.assertFunctionMatchesEager(for_with_variable_shape_and_break, n)

  @parameterized.parameters(*itertools.product(
      (0, 1, 2, 5),
      (int, tf.constant),
  ))
  def test_while_with_composite_tensor_shape_invariant(self, n, type_):
    # [FUNCTION DOC]
    # Parameterized over n in 0, 1, 2 and 5 and converters. Asserts the sparse-tensor while loop
    # matches eager.
    n = type_(n)
    self.assertFunctionMatchesEager(
        while_with_composite_tensor_shape_invariant, n)

  @parameterized.parameters(*itertools.product(
      (0, 1, 2),
      (range, tf.range, _int_dataset_range),
  ))
  def test_for_with_composite_tensor_shape_invariant(self, n, type_):
    # [FUNCTION DOC]
    # Parameterized over counts and iterable constructors (range, tf.range, int32 dataset range).
    # Asserts the sparse-tensor for loop matches eager.
    l = type_(n)
    self.assertFunctionMatchesEager(
        for_with_composite_tensor_shape_invariant, l)


if __name__ == '__main__':
  tf.test.main()
