# [MODULE DOC]
# Reference tests for the error cases of AutoGraph loops whose loop variables change type, dtype,
# shape or nested structure between iterations. Defines subject functions (while and for variants)
# that violate loop-variable invariants, helper input constructors (tensor range, datasets,
# dataset iterators, distributed datasets) and a test class asserting that tf.function raises the
# expected error with the expected message. Distributed datasets are built with a mirrored
# strategy. No import-time side effects; runs the TF test main when executed directly.
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

import re

from absl.testing import parameterized
import tensorflow as tf

from tensorflow.python.autograph.tests import reference_test_base


def while_with_variable_py_type():
  # [FUNCTION DOC]
  # Test subject: the loop condition starts as a Python boolean and is replaced by a boolean
  # tensor inside the body; returns an int32 scalar tensor zero. Compilation should reject it.
  n = tf.constant(0, dtype=tf.int32)
  c = True
  while c:
    c = tf.constant(True)
  return n


def while_with_variable_dtype():
  # [FUNCTION DOC]
  # Test subject: an int32 tensor n is rebound to a float32 tensor inside a while loop whose
  # condition is a constant true tensor. Returns n. The dtype change should be rejected.
  n = tf.constant(0, dtype=tf.int32)
  while tf.constant(True):
    n = tf.constant(0, dtype=tf.float32)
  return n


def while_with_variable_dtype_and_early_stopping():
  # [FUNCTION DOC]
  # Test subject: like the dtype-changing while loop but with a break right after the rebinding.
  # Returns n. The dtype change should still be rejected.
  n = tf.constant(0, dtype=tf.int32)
  while tf.constant(True):
    n = tf.constant(0, dtype=tf.float32)
    break
  return n


def for_with_variable_dtype(l):
  # [FUNCTION DOC]
  # Test subject: for loop over l rebinding an int32 tensor n to a float32 tensor each iteration.
  # Returns n.
  n = tf.constant(0, dtype=tf.int32)
  for _ in l:
    n = tf.constant(0, dtype=tf.float32)
  return n


def for_with_variable_dtype_and_early_stopping(l):
  # [FUNCTION DOC]
  # Test subject: for loop over l rebinding an int32 tensor n to float32 then breaking. Returns n.
  n = tf.constant(0, dtype=tf.int32)
  for _ in l:
    n = tf.constant(0, dtype=tf.float32)
    break
  return n


def while_with_variable_shape():
  # [FUNCTION DOC]
  # Test subject: a shape-(1,) tensor t is rebound to a shape-(2,) tensor inside a while loop
  # with a constant true condition. Returns t. The shape change should be rejected.
  t = tf.constant([1])
  while tf.constant(True):
    t = tf.constant([1, 1])
  return t


def for_with_variable_shape(l):
  # [FUNCTION DOC]
  # Test subject: for loop over l rebinding a shape-(1,) tensor t to a shape-(2,) tensor.
  # Returns t.
  t = tf.constant([1])
  for _ in l:
    t = tf.constant([1, 1])
  return t


def while_with_shape_erasure():
  # [FUNCTION DOC]
  # Test subject: a shape-(1,) tensor t is rebound to a range tensor whose length is random and
  # therefore statically unknown, so its shape becomes unknown. Returns t.
  t = tf.constant([1])
  while tf.constant(True):
    t = tf.range(tf.random.uniform((), 2, 3, dtype=tf.int32))
  return t


def for_with_shape_erasure(l):
  # [FUNCTION DOC]
  # Test subject: for loop over l rebinding t to a range tensor of statically unknown length
  # (shape erased to unknown). Returns t.
  t = tf.constant([1])
  for _ in l:
    t = tf.range(tf.random.uniform((), 2, 3, dtype=tf.int32))
  return t


def while_with_shape_invariant_violation():
  # [FUNCTION DOC]
  # Test subject: sets a loop option declaring the shape invariant of t as shape (1,), then
  # rebinds t to a range of unknown length which violates it. Returns t.
  t = tf.constant([1])
  while tf.constant(True):
    tf.autograph.experimental.set_loop_options(
        shape_invariants=((t, tf.TensorShape([1])),))
    t = tf.range(tf.random.uniform((), 2, 3, dtype=tf.int32))
  return t


def for_with_shape_invariant_violation(l):
  # [FUNCTION DOC]
  # Test subject: for loop over l that declares a fixed shape invariant of (1,) for t, then
  # rebinds t to a tensor of unknown length, violating the invariant. Returns t.
  t = tf.constant([1])
  for _ in l:
    tf.autograph.experimental.set_loop_options(
        shape_invariants=((t, tf.TensorShape([1])),))
    t = tf.range(tf.random.uniform((), 2, 3, dtype=tf.int32))
  return t


def while_with_variable_structure():
  # [FUNCTION DOC]
  # Test subject: the loop variable s starts as a dictionary holding a tensor and is rebound to a
  # float scalar tensor, changing its nested structure. Returns s.
  s = {'a': tf.constant(0)}
  while tf.constant(True):
    s = tf.constant(7.0)
  return s


def for_with_variable_structure(l):
  # [FUNCTION DOC]
  # Test subject: s starts as a one-element list of tensors and a longer list is assigned each
  # iteration, changing its structure. Returns s.
  s = [tf.constant(0)]
  for _ in l:
    s = s + [tf.constant(0)]
  return s


def _tf_range(l):
  # [FUNCTION DOC]
  # HELPER: converts a list into a tensor range of its length; used as an input constructor.
  return tf.range(len(l))


def _dataset(l):
  # [FUNCTION DOC]
  # HELPER: builds a tf.data dataset with one element per entry of l; used as an input constructor.
  return tf.data.Dataset.from_tensor_slices(l)


def _dataset_iterator(l):
  # [FUNCTION DOC]
  # HELPER: builds a dataset from l and returns an iterator over it; used as an input constructor.
  return iter(tf.data.Dataset.from_tensor_slices(l))


def _distributed_dataset(l):
  # [FUNCTION DOC]
  # HELPER: builds a dataset of two copies of l and distributes it with a mirrored strategy,
  # returning the distributed dataset; used as an input constructor.
  ds = tf.data.Dataset.from_tensor_slices([l] * 2)
  return tf.distribute.MirroredStrategy().experimental_distribute_dataset(ds)


class ReferenceTest(reference_test_base.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests that compiling the invalid-loop subject functions with tf.function raises the right
  # error type and message, over tensors, ranges, datasets, iterators and distributed datasets.

  def test_while_with_variable_py_type(self):
    # [FUNCTION DOC]
    # Asserts tf.function raises a not-implemented error saying the while condition started as
    # a non-tensor and then changed to a tensor.
    with self.assertRaisesRegex(
        NotImplementedError,
        re.compile(
            r'.*condition of while loop started as non\-Tensor,'
            r' then changed to Tensor.*', re.DOTALL)):
      tf.function(while_with_variable_py_type)()

  def test_while_with_variable_dtype(self):
    # [FUNCTION DOC]
    # Asserts a type error stating n has dtype int32 before the loop but float32 after.
    with self.assertRaisesRegex(
        TypeError,
        "'n' has dtype int32 before the loop, but dtype float32 after"):
      tf.function(while_with_variable_dtype)()

  def test_while_with_variable_dtype_and_early_stopping(self):
    # [FUNCTION DOC]
    # Asserts the same dtype-change type error even when the loop body breaks early.
    with self.assertRaisesRegex(
        TypeError,
        "'n' has dtype int32 before the loop, but dtype float32 after"):
      tf.function(while_with_variable_dtype_and_early_stopping)()

  @parameterized.parameters(
      (tf.constant,),
      (_tf_range,),
      (_dataset,),
      (_dataset_iterator,),
      (_distributed_dataset,),
  )
  def test_for_with_variable_dtype(self, type_):
    # [FUNCTION DOC]
    # Parameterized over input constructors (tensor, tensor range, dataset, dataset iterator,
    # distributed dataset). Builds a three-element input and asserts the dtype-change type error.
    l = type_([1, 2, 3])
    with self.assertRaisesRegex(
        TypeError,
        "'n' has dtype int32 before the loop, but dtype float32 after"):
      tf.function(for_with_variable_dtype)(l)

  # Note: distributed datasets don't allow early stopping.
  @parameterized.parameters(
      (tf.constant,),
      (_tf_range,),
      (_dataset,),
      (_dataset_iterator,),
  )
  def test_for_with_variable_dtype_and_early_stopping(self, type_):
    # [FUNCTION DOC]
    # Parameterized like the dtype test but without distributed datasets (they do not allow
    # early stopping). Asserts the dtype-change type error with a break in the loop.
    l = type_([1, 2, 3])
    with self.assertRaisesRegex(
        TypeError,
        "'n' has dtype int32 before the loop, but dtype float32 after"):
      tf.function(for_with_variable_dtype_and_early_stopping)(l)

  def test_while_with_variable_shape(self):
    # [FUNCTION DOC]
    # Asserts a value error saying t has shape (1,) before the loop but (2,) after.
    with self.assertRaisesRegex(
        ValueError,
        r"'t' has shape \(1,\) before the loop, but shape \(2,\) after"):
      tf.function(while_with_variable_shape)()

  # Note: datasets do allow variable shape.
  @parameterized.parameters(
      (tf.constant,),
      (_tf_range,),
      (_dataset_iterator,),
      (_distributed_dataset,),
  )
  def test_for_with_variable_shape(self, type_):
    # [FUNCTION DOC]
    # Parameterized over tensor, tensor range, dataset iterator and distributed dataset (plain
    # datasets are excluded because they allow variable shape). Asserts the shape-change error.
    l = type_([1, 2, 3])
    with self.assertRaisesRegex(
        ValueError,
        r"'t' has shape \(1,\) before the loop, but shape \(2,\) after"):
      tf.function(for_with_variable_shape)(l)

  def test_while_with_shape_erasure(self):
    # [FUNCTION DOC]
    # Asserts a value error saying t has shape (1,) before the loop but unknown shape after.
    with self.assertRaisesRegex(
        ValueError,
        r"'t' has shape \(1,\) before the loop, but shape \(None,\) after"):
      tf.function(while_with_shape_erasure)()

  # Note: datasets do allow variable shape.
  @parameterized.parameters(
      (tf.constant,),
      (_tf_range,),
      (_dataset_iterator,),
      (_distributed_dataset,),
  )
  def test_for_with_shape_erasure(self, type_):
    # [FUNCTION DOC]
    # Parameterized over tensor, tensor range, dataset iterator and distributed dataset.
    # Asserts the shape-erasure value error.
    l = type_([1, 2, 3])
    with self.assertRaisesRegex(
        ValueError,
        r"'t' has shape \(1,\) before the loop, but shape \(None,\) after"):
      tf.function(for_with_shape_erasure)(l)

  def test_while_with_shape_invariant_violation(self):
    # [FUNCTION DOC]
    # Asserts a value error saying t has unknown shape after one iteration, which does not
    # conform to the declared invariant.
    with self.assertRaisesRegex(
        ValueError,
        r"'t' has shape \(None,\) after one iteration, which does not conform"):
      tf.function(while_with_shape_invariant_violation)()

  # Note: dataset loops ignore shape invariants.
  @parameterized.parameters(
      (tf.constant,),
      (_tf_range,),
      (_dataset_iterator,),
      (_distributed_dataset,),
  )
  def test_for_with_shape_invariant_violation(self, type_):
    # [FUNCTION DOC]
    # Parameterized over tensor, tensor range, dataset iterator and distributed dataset (dataset
    # loops ignore shape invariants and so are excluded). Asserts the invariant-violation error.
    l = type_([1, 2, 3])
    with self.assertRaisesRegex(
        ValueError,
        r"'t' has shape \(None,\) after one iteration, which does not conform"):
      tf.function(for_with_shape_invariant_violation)(l)

  def test_while_with_variable_structure(self):
    # [FUNCTION DOC]
    # Asserts a type error saying s does not have the same nested structure before and after.
    with self.assertRaisesRegex(
        TypeError,
        "'s' does not have the same nested structure"):
      tf.function(while_with_variable_structure)()

  @parameterized.parameters(
      (tf.constant,),
      (_tf_range,),
      (_dataset,),
      (_dataset_iterator,),
      (_distributed_dataset,),
  )
  def test_for_with_variable_structure(self, type_):
    # [FUNCTION DOC]
    # Parameterized over all five input constructors. Asserts the nested-structure type error.
    l = type_([1, 2, 3])
    with self.assertRaisesRegex(
        TypeError,
        "'s' does not have the same nested structure"):
      tf.function(for_with_variable_structure)(l)


if __name__ == '__main__':
  tf.test.main()
