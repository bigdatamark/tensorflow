# [MODULE DOC]
# Unit tests for the AutoGraph control flow operators (for_stmt, while_stmt, if_stmt) defined in
# the control_flow operators module. Tests are grouped into ForLoopTest, WhileLoopTest and
# IfStmtTest, all derived from the AutoGraph test base class which can assert which graph ops
# were (or were not) created. They check both the pure-Python code path (plain iterables and
# Python conditions create no graph ops) and the staged path (tensors, datasets, iterators,
# ragged tensors create While/If/ScanDataset ops), plus error messages for illegal loop
# variables. Also defines two private module-level helpers producing tensors of unknown shape.
# Importing this module has no side effects; running it as a script starts the test runner.
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
"""Tests for control_flow module."""

# Unfortunately pylint has false positives when nonlocal is present.
# pylint:disable=unused-variable

import collections
import io
import re
import sys

import numpy as np

from tensorflow.python.autograph.operators import control_flow
from tensorflow.python.autograph.operators import variables as variable_operators
from tensorflow.python.autograph.utils import ag_logging
from tensorflow.python.autograph.utils import testing
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test


def _unranked_item(value):
  # [FUNCTION DOC]
  # HELPER: test helper producing tensors whose rank is unknown at graph construction time.
  # Takes a scalar value. Draws a random integer that is always 3, builds a vector of ones of
  # that length (so its static shape is unknown) and returns a tensor of the same unknown rank
  # filled with the given value. Creates random and fill ops in the current graph.
  rand_rank = random_ops.random_uniform(
      shape=(), minval=3, maxval=4, dtype=dtypes.int32)
  rand_shape = array_ops.ones([rand_rank], dtype=dtypes.int32)
  return array_ops.fill(rand_shape, value)


def _partial_shaped_bools():
  # [FUNCTION DOC]
  # HELPER: test helper producing a boolean tensor with partially unknown static shape.
  # Takes no arguments. Builds a range whose length is a random value that is always 2, adds a
  # leading axis and compares to zero, so the result is a 1-by-2 tensor of false values whose
  # shape is not statically known. Used to test conditions that are not single-element.
  rand_vect = math_ops.range(
      random_ops.random_uniform(
          shape=(), minval=2, maxval=3, dtype=dtypes.int32))
  return array_ops.expand_dims_v2(rand_vect, 0) < 0


class ForLoopTest(testing.AutoGraphTestCase):
  # [CLASS DOC]
  # Tests for the for-loop operator. Each test supplies a body, state getter/setter and symbol
  # names, runs the operator over a different kind of iterable (tensor, range tensor, Python
  # range or generator, dataset, iterator, ragged tensor) and checks the final loop state and
  # which graph ops were created (staged ops for tensors/datasets, none for plain Python).

  def test_tensor(self):
    # [FUNCTION DOC]
    # Loops over a constant tensor of 1..4, accumulating digits into a state variable
    # (multiply by ten and add). Asserts the state is a one-element tuple holding 1234.
    def body(i):
      # [FUNCTION DOC]
      # HELPER: loop body; updates the enclosing state variable by shifting a decimal digit in.
      nonlocal s
      s = s * 10 + i

    def set_state(loop_vars):
      # [FUNCTION DOC]
      # HELPER: state setter; stores the single loop variable into the enclosing variable.
      nonlocal s
      s, = loop_vars

    s = 0
    control_flow.for_stmt(
        constant_op.constant([1, 2, 3, 4]),
        extra_test=lambda: True,
        body=body,
        get_state=lambda: (s,),
        set_state=set_state,
        symbol_names=('s',),
        opts={})
    self.assertEqual(s, (1234,))

  def test_range_tensor(self):
    # [FUNCTION DOC]
    # Loops over a range tensor of length 5 (iterate name given in options), accumulating
    # digits. Asserts the state equals a one-element tuple holding 1234 and that a stateless
    # while op was created.
    def body(i):
      # [FUNCTION DOC]
      # HELPER: loop body; shifts the iteration value into the enclosing state variable.
      nonlocal s
      s = s * 10 + i

    def set_state(loop_vars):
      # [FUNCTION DOC]
      # HELPER: state setter; stores the single loop variable into the enclosing variable.
      nonlocal s
      s, = loop_vars

    s = 0
    control_flow.for_stmt(
        math_ops.range(5),
        extra_test=lambda: True,
        body=body,
        get_state=lambda: (s,),
        set_state=set_state,
        symbol_names=('s',),
        opts={'iterate_names': 'i'})

    self.assertEqual(s, (1234,))
    self.assertOpCreated('StatelessWhile')

  def test_range_tensor_explicit_limit_delta(self):
    # [FUNCTION DOC]
    # Loops over a range tensor with explicit start -17, limit -3 and step 5 (values -17, -12,
    # -7), accumulating with a factor of 100. Asserts the state equals a one-element tuple with
    # -171207 and that a stateless while op was created.
    def body(i):
      # [FUNCTION DOC]
      # HELPER: loop body; appends the iteration value to the state using base 100.
      nonlocal s
      s = s * 100 + i

    def set_state(loop_vars):
      # [FUNCTION DOC]
      # HELPER: state setter; stores the single loop variable into the enclosing variable.
      nonlocal s
      s, = loop_vars

    s = 0
    control_flow.for_stmt(
        math_ops.range(-17, -3, 5),
        extra_test=lambda: True,
        body=body,
        get_state=lambda: (s,),
        set_state=set_state,
        symbol_names=('s',),
        opts={'iterate_names': 'i'})

    self.assertEqual(s, (-171207,))
    self.assertOpCreated('StatelessWhile')

  def test_range_tensor_explicit_limit_negative_delta(self):
    # [FUNCTION DOC]
    # Loops over a range tensor counting down from 17 to 3 with step -5 (values 17, 12, 7),
    # accumulating with a factor of 100. Asserts the state is a one-element tuple with 171207
    # and that a stateless while op was created.
    def body(i):
      # [FUNCTION DOC]
      # HELPER: loop body; appends the iteration value to the state using base 100.
      nonlocal s
      s = s * 100 + i

    def set_state(loop_vars):
      # [FUNCTION DOC]
      # HELPER: state setter; stores the single loop variable into the enclosing variable.
      nonlocal s
      s, = loop_vars

    s = 0
    control_flow.for_stmt(
        math_ops.range(17, 3, -5),
        extra_test=lambda: True,
        body=body,
        get_state=lambda: (s,),
        set_state=set_state,
        symbol_names=('s',),
        opts={'iterate_names': 'i'})

    self.assertEqual(s, (171207,))
    self.assertOpCreated('StatelessWhile')

  def test_range_tensor_random_delta(self):
    # [FUNCTION DOC]
    # Loops over a range tensor whose step is a random tensor that is always 1 (so the step is
    # unknown at graph construction time). Asserts the digits 0..4 accumulate to a one-element
    # tuple holding 1234 and that a stateless while op was created.
    def body(i):
      # [FUNCTION DOC]
      # HELPER: loop body; shifts the iteration value into the enclosing state variable.
      nonlocal s
      s = s * 10 + i

    def set_state(loop_vars):
      # [FUNCTION DOC]
      # HELPER: state setter; stores the single loop variable into the enclosing variable.
      nonlocal s
      s, = loop_vars

    s = 0
    random_one = random_ops.random_uniform((), 1, 2, dtype=dtypes.int32)
    control_flow.for_stmt(
        math_ops.range(0, 5, random_one),
        extra_test=lambda: True,
        body=body,
        get_state=lambda: (s,),
        set_state=set_state,
        symbol_names=('s',),
        opts={'iterate_names': 'i'})

    self.assertEqual(s, (1234,))
    self.assertOpCreated('StatelessWhile')

  def test_range_tensor_random_negative_delta(self):
    # [FUNCTION DOC]
    # Loops over a range tensor from 17 down to 3 whose step is a random tensor that is always
    # -5 (unknown statically). Asserts the state is a one-element tuple with 171207 and that a
    # stateless while op was created.
    def body(i):
      # [FUNCTION DOC]
      # HELPER: loop body; appends the iteration value to the state using base 100.
      nonlocal s
      s = s * 100 + i

    def set_state(loop_vars):
      # [FUNCTION DOC]
      # HELPER: state setter; stores the single loop variable into the enclosing variable.
      nonlocal s
      s, = loop_vars

    s = 0
    random_neg_five = random_ops.random_uniform((), -5, -4, dtype=dtypes.int32)
    control_flow.for_stmt(
        math_ops.range(17, 3, random_neg_five),
        extra_test=lambda: True,
        body=body,
        get_state=lambda: (s,),
        set_state=set_state,
        symbol_names=('s',),
        opts={'iterate_names': 'i'})

    self.assertEqual(s, (171207,))
    self.assertOpCreated('StatelessWhile')

  def test_tensor_with_extra_test_object_vars(self):
    # [FUNCTION DOC]
    # Loops over a constant tensor 1..4 with loop state held in two attributes of a mutable
    # object (symbol names use dotted attribute names). The body adds to one field and
    # multiplies the other; an extra test stops the loop when the first field reaches 6.
    # Asserts both fields end at 6 and a stateless while op was created.

    class MutableObject:
      # [CLASS DOC]
      # HELPER: small mutable holder with two integer tensor attributes (field_1 starts at 0,
      # field_2 starts at 1) used as loop state in this test.
      field_1 = constant_op.constant(0, dtype=dtypes.int32)
      field_2 = constant_op.constant(1, dtype=dtypes.int32)
    state = MutableObject()

    def body(i):
      # [FUNCTION DOC]
      # HELPER: loop body; adds the iteration value to the first field and multiplies the second.
      state.field_1 += i
      state.field_2 *= i

    def get_state():
      # [FUNCTION DOC]
      # HELPER: state getter; returns the two fields as a tuple.
      return state.field_1, state.field_2

    def set_state(loop_vars):
      # [FUNCTION DOC]
      # HELPER: state setter; assigns the two loop variables back to the object's fields.
      state.field_1, state.field_2 = loop_vars

    control_flow.for_stmt(
        iter_=constant_op.constant([1, 2, 3, 4]),
        body=body,
        extra_test=lambda: state.field_1 < 6,
        get_state=get_state,
        set_state=set_state,
        symbol_names=('state.field_1', 'state.field_2'),
        opts={})

    self.assertEqual((state.field_1, state.field_2), (6, 6))
    self.assertOpCreated('StatelessWhile')

  def test_python(self):
    # [FUNCTION DOC]
    # Loops over a plain Python range of 5 numbers, accumulating digits. Asserts the result is
    # the plain integer 1234 and that no graph ops were created (pure Python execution).
    def body(i):
      # [FUNCTION DOC]
      # HELPER: loop body; shifts the iteration value into the enclosing state variable.
      nonlocal s
      s = s * 10 + i

    def set_state(loop_vars):
      # [FUNCTION DOC]
      # HELPER: state setter; stores the single loop variable into the enclosing variable.
      nonlocal s
      s, = loop_vars

    s = 0
    control_flow.for_stmt(
        range(5),
        extra_test=lambda: True,
        body=body,
        get_state=lambda: (s,),
        set_state=set_state,
        symbol_names=('s',),
        opts={})

    self.assertEqual(s, 1234)
    self.assertNoOpsCreated()

  def test_python_generator_with_extra_test(self):
    # [FUNCTION DOC]
    # Runs the for operator three times over the same Python generator (values 1..4) with an
    # extra test that stops the loop after the first iteration, with no state getter/setter.
    # Asserts each run consumes exactly one element (results (1,1), (2,1), (3,1)), that the
    # generator still yields 4 afterwards (no extra element was consumed), and no graph ops.
    def new_generator():
      # [FUNCTION DOC]
      # HELPER: generator function yielding the integers 1 to 4.
      for i in range(1, 5):
        yield i

    gen = new_generator()
    def run_loop():
      # [FUNCTION DOC]
      # HELPER: runs one for-loop pass over the shared generator that breaks after one
      # iteration; returns the pair (accumulated value, iteration count).
      s = 0
      c = 0

      def body(i):
        # [FUNCTION DOC]
        # HELPER: loop body; accumulates the value and counts the iteration.
        nonlocal s, c
        s = s * 10 + i
        c += 1

      control_flow.for_stmt(
          gen,
          extra_test=lambda: c == 0,  # Break after first iteration
          body=body,
          get_state=None,
          set_state=None,
          symbol_names=('s', 'c'),
          opts={})
      return s, c

    self.assertEqual(run_loop(), (1, 1))
    self.assertEqual(run_loop(), (2, 1))
    self.assertEqual(run_loop(), (3, 1))

    self.assertEqual(next(gen), 4)

    self.assertNoOpsCreated()

  def test_python_generator_with_extra_test_no_iterations(self):
    # [FUNCTION DOC]
    # Runs the for operator twice over a generator (values 0..4) with an extra test that is
    # always false. Asserts the body never runs (result 0 both times), that no element is
    # consumed from the generator (next still gives 0) and that no graph ops were created.
    def new_generator():
      # [FUNCTION DOC]
      # HELPER: generator function yielding the integers 0 to 4.
      for i in range(5):
        yield i

    gen = new_generator()
    def run_loop():
      # [FUNCTION DOC]
      # HELPER: runs one for-loop pass whose extra test is false from the start; returns the
      # untouched accumulator (zero).
      s = 0

      def body(i):
        # [FUNCTION DOC]
        # HELPER: loop body; accumulates the value (never reached in this test).
        nonlocal s
        s = s * 10 + i

      control_flow.for_stmt(
          gen,
          extra_test=lambda: False,  # Break before loop
          body=body,
          get_state=None,
          set_state=None,
          symbol_names=('s',),
          opts={})
      return s

    self.assertEqual(run_loop(), 0)
    self.assertEqual(run_loop(), 0)

    self.assertEqual(next(gen), 0)

    self.assertNoOpsCreated()

  def test_tf_dataset(self):
    # [FUNCTION DOC]
    # Loops over a dataset of range 5 with an int64 tensor state and no extra test. Asserts the
    # state is a one-element tuple holding 1234 and that a scan-dataset op was created.
    def body(i):
      # [FUNCTION DOC]
      # HELPER: loop body; shifts the dataset element into the enclosing state variable.
      nonlocal s
      s = s * 10 + i

    def set_state(loop_vars):
      # [FUNCTION DOC]
      # HELPER: state setter; stores the single loop variable into the enclosing variable.
      nonlocal s
      s, = loop_vars

    s = constant_op.constant(0, dtype=dtypes.int64)
    control_flow.for_stmt(
        dataset_ops.Dataset.range(5),
        extra_test=None,
        body=body,
        get_state=lambda: (s,),
        set_state=set_state,
        symbol_names=('s',),
        opts={})

    self.assertEqual(s, (1234,))
    self.assertOpCreated('ScanDataset')

  def test_dataset_with_extra_test(self):
    # [FUNCTION DOC]
    # Loops over a dataset of range 5 with an extra test that stops when the state reaches 3.
    # Asserts the final state is a one-element tuple holding 12 and a scan-dataset op exists.
    def body(i):
      # [FUNCTION DOC]
      # HELPER: loop body; shifts the dataset element into the enclosing state variable.
      nonlocal s
      s = s * 10 + i

    def set_state(loop_vars):
      # [FUNCTION DOC]
      # HELPER: state setter; stores the single loop variable into the enclosing variable.
      nonlocal s
      s, = loop_vars

    s = constant_op.constant(0, dtype=dtypes.int64)
    control_flow.for_stmt(
        dataset_ops.Dataset.range(5),
        extra_test=lambda: s < 3,
        body=body,
        get_state=lambda: (s,),
        set_state=set_state,
        symbol_names=('s',),
        opts={})

    self.assertEqual(s, (12,))
    self.assertOpCreated('ScanDataset')

  def test_dataset_with_extra_test_collection_vars(self):
    # [FUNCTION DOC]
    # Loops over a dataset of range 5 with two state variables, one being an element of a list
    # (symbol name with an index), and an extra test on the scalar one stopping at 3. Asserts
    # both end at 3 and a scan-dataset op was created.
    def body(i):
      # [FUNCTION DOC]
      # HELPER: loop body; adds the element to both the list entry and the scalar state.
      nonlocal s
      l[0] += i
      s += i

    def set_state(loop_vars):
      # [FUNCTION DOC]
      # HELPER: state setter; assigns the list entry and the scalar from the loop variables.
      nonlocal s
      l[0], s = loop_vars

    s = constant_op.constant(0, dtype=dtypes.int64)
    l = [constant_op.constant(0, dtype=dtypes.int64)]
    control_flow.for_stmt(
        dataset_ops.Dataset.range(5),
        extra_test=lambda: s < 3,
        body=body,
        get_state=lambda: (l[0], s),
        set_state=set_state,
        symbol_names=('l[0]', 's'),
        opts={})

    self.assertEqual((l[0], s), (3, 3))
    self.assertOpCreated('ScanDataset')

  def test_dataset_with_extra_test_iteration_limiting(self):
    # [FUNCTION DOC]
    # Loops over a dataset of range 5 with an extra test (state below 3) while the body
    # asserts, via a control dependency, that the previous state is below 3. Verifies that the
    # extra test actually stops iteration so the assertion never fails: final state is a
    # one-element tuple holding 3 and a scan-dataset op was created.
    def body(it):
      # [FUNCTION DOC]
      # HELPER: loop body; asserts the current state is below 3, then replaces the state with
      # the dataset element.
      nonlocal i
      with ops.control_dependencies((control_flow_assert.Assert(i < 3, (i,)),)):
        i = it

    def set_state(loop_vars):
      # [FUNCTION DOC]
      # HELPER: state setter; stores the single loop variable into the enclosing variable.
      nonlocal i
      i, = loop_vars

    i = constant_op.constant(0, dtype=dtypes.int64)
    control_flow.for_stmt(
        dataset_ops.Dataset.range(5),
        extra_test=lambda: i < 3,
        body=body,
        get_state=lambda: (i,),
        set_state=set_state,
        symbol_names=('i',),
        opts={})

    self.assertEqual(i, (3,))
    self.assertOpCreated('ScanDataset')

  def test_tf_dataset_no_loop_vars(self):
    # [FUNCTION DOC]
    # Loops over a dataset of range 5 with no loop variables; the body instead updates a
    # variable created by the test fixture through assignment. Asserts the variable ends at
    # 1234 and a scan-dataset op was created.
    def body(i):
      # [FUNCTION DOC]
      # HELPER: loop body; assigns the variable its value times ten plus the element.
      v.assign(v.read_value() * 10 + i)

    v = self.variable('v', 0, dtypes.int64)

    control_flow.for_stmt(
        dataset_ops.Dataset.range(5),
        extra_test=None,
        body=body,
        get_state=lambda: (),
        set_state=lambda _: None,
        symbol_names=(),
        opts={})

    self.assertEqual(v.read_value(), 1234)
    self.assertOpCreated('ScanDataset')

  def test_tf_iterator(self):
    # [FUNCTION DOC]
    # Loops over an iterator of a dataset of range 5 with an int64 state. Asserts the final
    # state equals 1234 (a bare value, not a tuple) and that an op fetching the next element as
    # an optional value was created.
    def body(i):
      # [FUNCTION DOC]
      # HELPER: loop body; shifts the iterator element into the enclosing state variable.
      nonlocal s
      s = s * 10 + i

    def set_state(loop_vars):
      # [FUNCTION DOC]
      # HELPER: state setter; stores the single loop variable into the enclosing variable.
      nonlocal s
      s, = loop_vars

    s = constant_op.constant(0, dtype=dtypes.int64)
    control_flow.for_stmt(
        iter(dataset_ops.Dataset.range(5)),
        extra_test=None,
        body=body,
        get_state=lambda: (s,),
        set_state=set_state,
        symbol_names=('s',),
        opts={})

    self.assertEqual(s, 1234)
    self.assertOpCreated('IteratorGetNextAsOptional')

  def test_tf_iterator_shape_invariants(self):
    # [FUNCTION DOC]
    # Loops over a dataset iterator while the state is a growing vector (concatenating each
    # element). A shape invariant option declares the state as a vector of unknown length.
    # Asserts the final state is 0..4 and the next-as-optional op was created.
    def body(i):
      # [FUNCTION DOC]
      # HELPER: loop body; appends the element to the state vector.
      nonlocal s
      s = array_ops.concat([s, [i]], 0)

    def set_state(loop_vars):
      # [FUNCTION DOC]
      # HELPER: state setter; stores the single loop variable into the enclosing variable.
      nonlocal s
      s, = loop_vars

    s = constant_op.constant([], dtype=dtypes.int64)
    control_flow.for_stmt(
        iter(dataset_ops.Dataset.range(5)),
        extra_test=None,
        body=body,
        get_state=lambda: (s,),
        set_state=set_state,
        symbol_names=('s',),
        opts={'shape_invariants': [(s, tensor_shape.TensorShape([None]))]})

    self.assertAllEqual(s, [0, 1, 2, 3, 4])
    self.assertOpCreated('IteratorGetNextAsOptional')

  def test_tf_iterator_shape_invariants_with_nested_structures(self):
    # [FUNCTION DOC]
    # Like the shape invariant iterator test, but the loop state also includes a named tuple
    # with one tensor field that is incremented each iteration. Asserts the growing vector ends
    # as 0..4, the tuple field ends as 5, and the next-as-optional op was created.
    def body(i):
      # [FUNCTION DOC]
      # HELPER: loop body; appends the element to the vector and increments the tuple field.
      nonlocal s
      nonlocal t
      s = array_ops.concat([s, [i]], 0)
      t = Test(var=t.var + 1)

    def set_state(loop_vars):
      # [FUNCTION DOC]
      # HELPER: state setter; stores the two loop variables into the enclosing variables.
      nonlocal s
      nonlocal t
      s, t = loop_vars

    s = constant_op.constant([], dtype=dtypes.int64)
    Test = collections.namedtuple('Test', ['var'])
    t = Test(var=constant_op.constant([0], dtype=dtypes.int64))
    control_flow.for_stmt(
        iter(dataset_ops.Dataset.range(5)),
        extra_test=None,
        body=body,
        get_state=lambda: (s, t),
        set_state=set_state,
        symbol_names=('s', 't'),
        opts={'shape_invariants': [(s, tensor_shape.TensorShape([None]))]})

    self.assertAllEqual(s, [0, 1, 2, 3, 4])
    self.assertEqual(t.var, [5])
    self.assertOpCreated('IteratorGetNextAsOptional')

  def test_tf_iterator_no_loop_vars(self):
    # [FUNCTION DOC]
    # Loops over a dataset iterator with no loop variables; the body updates a fixture-created
    # variable. Asserts the variable ends at 1234 and the next-as-optional op was created.
    def body(i):
      # [FUNCTION DOC]
      # HELPER: loop body; assigns the variable its value times ten plus the element.
      v.assign(v.read_value() * 10 + i)

    v = self.variable('v', 0, dtypes.int64)

    control_flow.for_stmt(
        iter(dataset_ops.Dataset.range(5)),
        extra_test=None,
        body=body,
        get_state=lambda: (),
        set_state=lambda _: None,
        symbol_names=(),
        opts={})

    self.assertEqual(v.read_value(), 1234)
    self.assertOpCreated('IteratorGetNextAsOptional')

  def test_tf_ragged_tensor(self):
    # [FUNCTION DOC]
    # Loops over a ragged tensor with rows [1], [2, 4], [3]; the body uses the first element of
    # each row. Asserts the state is a one-element tuple holding 123 and a stateless while op
    # was created.
    def body(i):
      # [FUNCTION DOC]
      # HELPER: loop body; shifts the first element of the current row into the state.
      nonlocal s
      s = s * 10 + i[0]

    def set_state(loop_vars):
      # [FUNCTION DOC]
      # HELPER: state setter; stores the single loop variable into the enclosing variable.
      nonlocal s
      s, = loop_vars

    s = 0
    control_flow.for_stmt(
        ragged_factory_ops.constant([[1], [2, 4], [3]]),
        extra_test=None,
        body=body,
        get_state=lambda: (s,),
        set_state=set_state,
        symbol_names=('s',),
        opts={})

    self.assertEqual(s, (123,))
    self.assertOpCreated('StatelessWhile')

  def test_tf_ragged_tensor_higher_dimensional(self):
    # [FUNCTION DOC]
    # Loops over a three-dimensional ragged tensor with two outer rows; the body uses the
    # first element of the first inner row. Asserts the state is a one-element tuple holding
    # 12 and a stateless while op was created.
    def body(i):
      # [FUNCTION DOC]
      # HELPER: loop body; shifts the first nested element of the current row into the state.
      nonlocal s
      s = s * 10 + i[0][0]

    def set_state(loop_vars):
      # [FUNCTION DOC]
      # HELPER: state setter; stores the single loop variable into the enclosing variable.
      nonlocal s
      s, = loop_vars

    s = 0
    ragged_3d = [
        [[1], [1, 1], [1]],
        [[2], [2]],
    ]
    control_flow.for_stmt(
        ragged_factory_ops.constant(ragged_3d),
        extra_test=None,
        body=body,
        get_state=lambda: (s,),
        set_state=set_state,
        symbol_names=('s',),
        opts={})

    self.assertEqual(s, (12,))
    self.assertOpCreated('StatelessWhile')

  def test_tf_ragged_tensor_no_loop_vars(self):
    # [FUNCTION DOC]
    # Loops over a ragged tensor with no loop variables; the body updates a fixture-created
    # variable with the first element of each row. Asserts the variable ends at 123 and that a
    # (stateful) while op was created.
    v = self.variable('v', 0, dtypes.int32)

    def body(i):
      # [FUNCTION DOC]
      # HELPER: loop body; assigns the variable its value times ten plus the row's first element.
      v.assign(v.read_value() * 10 + i[0])

    control_flow.for_stmt(
        ragged_factory_ops.constant([[1], [2, 4], [3]]),
        extra_test=None,
        body=body,
        get_state=lambda: (),
        set_state=lambda _: None,
        symbol_names=(),
        opts={})

    # Note: 123 = ((0*10 + 1)*10+2)*10+3 (first element of each row).
    self.assertEqual(v.read_value(), 123)
    self.assertOpCreated('While')

  def _basic_loop(self, init_value, body_fn):
    # [FUNCTION DOC]
    # HELPER: shared by the illegal-input tests of this class. Runs the for operator over a
    # constant tensor 1..4 with one state variable starting at init_value; on each iteration
    # the state becomes body_fn(element, state) (body_fn is a callable of two arguments).
    # Returns the final state. Errors from the operator propagate to the caller.
    def body(i):
      # [FUNCTION DOC]
      # HELPER: loop body; applies the supplied function to the element and current state.
      nonlocal s
      s = body_fn(i, s)

    def set_state(loop_vars):
      # [FUNCTION DOC]
      # HELPER: state setter; stores the single loop variable into the enclosing variable.
      nonlocal s
      s, = loop_vars

    s = init_value
    control_flow.for_stmt(
        constant_op.constant([1, 2, 3, 4]),
        extra_test=lambda: True,
        body=body,
        get_state=lambda: (s,),
        set_state=set_state,
        symbol_names=('s',),
        opts={})
    return s

  def test_tensor_illegal_input(self):
    # [FUNCTION DOC]
    # Asserts a value error is raised when the initial state is nothing/null ("not allowed to
    # be None") and when it is an undefined-variable placeholder ("must be defined").
    with self.assertRaisesRegex(ValueError, '\'s\' is not allowed to be None'):
      self._basic_loop(None, lambda i, s: s)
    with self.assertRaisesRegex(ValueError, '\'s\' must be defined'):
      self._basic_loop(variable_operators.Undefined(''), lambda i, s: s)

  def test_tensor_none_output(self):
    # [FUNCTION DOC]
    # Asserts a value error ("is None at the end") when the loop body sets the state to
    # nothing/null.
    with self.assertRaisesRegex(ValueError, '\'s\' is None at the end'):
      self._basic_loop(0, lambda i, s: None)

  def test_tensor_dtype_change(self):
    # [FUNCTION DOC]
    # Asserts a type error mentioning dtype float32 when the body changes the state from an
    # integer to a float.
    with self.assertRaisesRegex(TypeError, '\'s\'.* dtype float32 after'):
      self._basic_loop(0, lambda i, s: 1.0)

  def test_tensor_shape_change(self):
    # [FUNCTION DOC]
    # Asserts a value error mentioning shape (1,) when the body changes the state from a
    # scalar to a one-element vector.
    with self.assertRaisesRegex(ValueError, r'\'s\'.* shape \(1,\) after'):
      self._basic_loop(0, lambda i, s: np.array([1], dtype=np.int32))


class WhileLoopTest(testing.AutoGraphTestCase):
  # [CLASS DOC]
  # Tests for the while-loop operator. Covers tensor conditions (staged while ops), loop
  # variables created inside the body, side-effecting conditions, plain Python loops, the
  # Python iteration limit and large-unroll warnings, illegal loop variable errors, and
  # validation of the loop condition's dtype and shape.

  def test_tensor(self):
    # [FUNCTION DOC]
    # While loop with tensor limit 5 and two state variables (counter and digit accumulator).
    # Asserts counter 5, accumulator 1234 and that a stateless while op was created.

    def body():
      # [FUNCTION DOC]
      # HELPER: loop body; shifts the counter into the accumulator then increments the counter.
      nonlocal i, s
      s = s * 10 + i
      i += 1

    def set_state(loop_vars):
      # [FUNCTION DOC]
      # HELPER: state setter; stores the counter and accumulator into the enclosing variables.
      nonlocal i, s
      i, s = loop_vars

    i = 0
    n = constant_op.constant(5)
    s = 0
    control_flow.while_stmt(
        test=lambda: i < n,
        body=body,
        get_state=lambda: (i, s),
        set_state=set_state,
        symbol_names=('i', 's'),
        opts={})

    self.assertEqual(i, 5)
    self.assertEqual(s, 1234)
    self.assertOpCreated('StatelessWhile')

  def test_tensor_creating_variable(self):
    # [FUNCTION DOC]
    # While loop where one variable starts undefined and is first created inside the body
    # (a constant 2 and its fifth power). Asserts values 2 and 32, a stateless while op, and
    # that exactly one power op exists, i.e. temporarily staging the body did not leave extra ops.

    def body():
      # [FUNCTION DOC]
      # HELPER: loop body; defines the first variable as 2 and the second as its fifth power.
      nonlocal i, s
      i = constant_op.constant(2)
      s = i ** 5

    def set_state(loop_vars):
      # [FUNCTION DOC]
      # HELPER: state setter; stores the two loop variables into the enclosing variables.
      nonlocal i, s
      i, s = loop_vars

    i = variable_operators.Undefined('i')
    s = constant_op.constant(0)
    control_flow.while_stmt(
        test=lambda: math_ops.equal(s, 0),
        body=body,
        get_state=lambda: (i, s),
        set_state=set_state,
        symbol_names=('i', 's'),
        opts={})

    self.assertEqual(i, 2)
    self.assertEqual(s, 32)
    self.assertOpCreated('StatelessWhile')
    # Check that the temporary staging of the body did not create extra ops.
    # Node naming is inconsistent between V1 and V2.
    self.assertGraphContains(r'(while/)?pow$', 1)

  def test_tensor_creating_dynamic_shape_variable(self):
    # [FUNCTION DOC]
    # While loop where a variable is undefined at the start and created in the body as a random
    # vector whose length depends on the counter (dynamic shape). Asserts the counter ends at 3
    # and the first random value is below 3.

    def body():
      # [FUNCTION DOC]
      # HELPER: loop body; increments the counter and defines a random vector of that length.
      nonlocal i, y
      i += 1
      y = random_ops.random_uniform([i])

    def set_state(loop_vars):
      # [FUNCTION DOC]
      # HELPER: state setter; stores the counter and vector into the enclosing variables.
      nonlocal i, y
      i, y = loop_vars

    i = constant_op.constant(0)
    y = variable_operators.Undefined('y')
    control_flow.while_stmt(
        test=lambda: math_ops.less(i, 3),
        body=body,
        get_state=lambda: (i, y),
        set_state=set_state,
        symbol_names=('i', 'y'),
        opts={})

    self.assertEqual(i, 3)
    self.assertLess(y[0], 3)

  def test_tensor_creating_dynamic_shape_variable_preserves_shape_invar(self):
    # [FUNCTION DOC]
    # Like the dynamic-shape creation test, but the body creates a fixed one-element vector and
    # a shape invariant option declares that shape for the initially undefined variable.
    # Asserts the variable can be evaluated afterwards (the declared invariant is honored).

    def body():
      # [FUNCTION DOC]
      # HELPER: loop body; increments the counter and defines a one-element zero vector.
      nonlocal i, y
      i += 1
      y = array_ops.zeros([1])

    def set_state(loop_vars):
      # [FUNCTION DOC]
      # HELPER: state setter; stores the counter and vector into the enclosing variables.
      nonlocal i, y
      i, y = loop_vars

    i = constant_op.constant(0)
    y = variable_operators.Undefined('y')
    control_flow.while_stmt(
        test=lambda: math_ops.less(i, 3),
        body=body,
        get_state=lambda: (i, y),
        set_state=set_state,
        symbol_names=('i', 'y'),
        opts={'shape_invariants': ((y, tensor_shape.TensorShape([1])),)})

    self.evaluate(y)

  def test_tensor_creating_complex_variable(self):
    # [FUNCTION DOC]
    # While loop where an undefined variable is created in the body as a nested dictionary of
    # tensors. Asserts the dictionary equals the expected nested values, the second variable is
    # 32, a stateless while op exists and only one power op was created.

    def body():
      # [FUNCTION DOC]
      # HELPER: loop body; defines a nested dictionary and a value derived from it.
      nonlocal i, s
      i = {'a': constant_op.constant(2), 'b': {'c': constant_op.constant(1)}}
      s = i['a'] ** 5

    def set_state(loop_vars):
      # [FUNCTION DOC]
      # HELPER: state setter; stores the two loop variables into the enclosing variables.
      nonlocal i, s
      i, s = loop_vars

    i = variable_operators.Undefined('i')
    s = constant_op.constant(0)
    control_flow.while_stmt(
        test=lambda: math_ops.equal(s, 0),
        body=body,
        get_state=lambda: (i, s),
        set_state=set_state,
        symbol_names=('i', 's'),
        opts={})

    self.assertDictEqual(i, {'a': 2, 'b': {'c': 1}})
    self.assertEqual(s, 32)
    self.assertOpCreated('StatelessWhile')
    # Check that the temporary staging of the body did not create extra ops.
    # Node naming is inconsistent between V1 and V2.
    self.assertGraphContains(r'(while/)?pow$', 1)

  def test_tensor_creating_variable_of_dynamic_shape(self):
    # [FUNCTION DOC]
    # While loop where an undefined variable is created in the body as a matrix of ones with a
    # random number of rows (7 columns) and the other variable is its sum. Asserts the first
    # element is 1, the sum is at least 7 and a (stateful) while op was created because of the
    # random op.

    def body():
      # [FUNCTION DOC]
      # HELPER: loop body; defines a random-height matrix of ones and its sum.
      nonlocal i, s
      i = array_ops.ones(
          [random_ops.random_uniform(minval=1, maxval=4, shape=()), 7])
      s = math_ops.reduce_sum(i)

    def set_state(loop_vars):
      # [FUNCTION DOC]
      # HELPER: state setter; stores the two loop variables into the enclosing variables.
      nonlocal i, s
      i, s = loop_vars

    i = variable_operators.Undefined('i')
    s = constant_op.constant(0.0)
    control_flow.while_stmt(
        test=lambda: math_ops.equal(s, 0),
        body=body,
        get_state=lambda: (i, s),
        set_state=set_state,
        symbol_names=('i', 's'),
        opts={})

    self.assertEqual(i[0][0], 1)
    self.assertGreaterEqual(s, 7)
    self.assertOpCreated('While')  # Not stateless because of the random op.

  def test_tensor_with_side_effecting_condition(self):
    # [FUNCTION DOC]
    # While loop whose condition has a side effect: it updates a fixture variable on every
    # evaluation. Asserts the counter ends at 5 (as a one-element tuple), the variable holds
    # 12345 (condition evaluated once per check, including the final one) and a stateful
    # while op was created.
    v = self.variable('v', 0, dtypes.int32)

    def cond():
      # [FUNCTION DOC]
      # HELPER: loop condition; updates the variable with the counter then returns whether the
      # counter is below the limit.
      v.assign(v.read_value() * 10 + i)
      return i < n

    def body():
      # [FUNCTION DOC]
      # HELPER: loop body; increments the counter.
      nonlocal i
      i += 1

    def set_state(loop_vars):
      # [FUNCTION DOC]
      # HELPER: state setter; stores the single loop variable into the enclosing variable.
      nonlocal i
      i, = loop_vars

    i = 0
    n = constant_op.constant(5)
    control_flow.while_stmt(
        test=cond,
        body=body,
        get_state=lambda: (i,),
        set_state=set_state,
        symbol_names=('i',),
        opts={})

    self.assertEqual(i, (5,))
    self.assertEqual(v, (12345,))
    self.assertOpCreated('While')

  def test_tensor_failing_to_determine_placeholder(self):
    # [FUNCTION DOC]
    # While loop where an undefined variable is created in the body as an instance of a
    # user-defined class that cannot be turned into a tensor placeholder. Asserts a value error
    # saying the variable must be defined, that creation was tried, and that the type is
    # unsupported.

    class UserType:
      # [CLASS DOC]
      # HELPER: empty user-defined type used as an unsupported loop variable value.
      pass

    def body():
      # [FUNCTION DOC]
      # HELPER: loop body; defines the variable as a new instance of the unsupported type.
      nonlocal v
      v = UserType()

    def set_state(loop_vars):
      # [FUNCTION DOC]
      # HELPER: state setter; stores the single loop variable into the enclosing variable.
      nonlocal v
      v, = loop_vars

    v = variable_operators.Undefined('v')

    with self.assertRaisesRegex(
        ValueError,
        re.compile('must be defined.*tried to define.*unsupported type',
                   re.DOTALL)):
      control_flow.while_stmt(
          test=lambda: constant_op.constant(True),
          body=body,
          get_state=lambda: (v,),
          set_state=set_state,
          symbol_names=('v',),
          opts={})

  def test_tensor_failing_to_stage_loop_body(self):
    # [FUNCTION DOC]
    # While loop whose body raises an error while the operator is trying to stage it to
    # discover the type of an undefined variable. Asserts a value error saying the variable must
    # be defined, creation was tried, and the original error text ("testing") is included.

    def body():
      # [FUNCTION DOC]
      # HELPER: loop body; defines the first variable, then deliberately raises a value error
      # (the final line is unreachable by design).
      nonlocal i, s
      i = constant_op.constant(2)
      raise ValueError('testing')
      s = i ** 5  # pylint: disable=unreachable

    def set_state(loop_vars):
      # [FUNCTION DOC]
      # HELPER: state setter; stores the two loop variables into the enclosing variables.
      nonlocal i, s
      i, s = loop_vars

    i = variable_operators.Undefined('i')
    s = constant_op.constant(0)

    with self.assertRaisesRegex(
        ValueError,
        re.compile('must be defined.*tried to define.*testing', re.DOTALL)):
      control_flow.while_stmt(
          test=lambda: math_ops.equal(s, 0),
          body=body,
          get_state=lambda: (i, s),
          set_state=set_state,
          symbol_names=('i', 's'),
          opts={})

  def test_tensor_with_python_state(self):
    # [FUNCTION DOC]
    # While loop with a tensor limit where one loop variable is an attribute of a mutable
    # Python object (dotted symbol name). Asserts the counter is 5, the attribute is 1234 and a
    # stateless while op was created.

    class MutableObject:
      # [CLASS DOC]
      # HELPER: small mutable holder with one integer tensor attribute starting at zero, used
      # as loop state in this test.
      field = constant_op.constant(0, dtype=dtypes.int32)
    state = MutableObject()

    def body():
      # [FUNCTION DOC]
      # HELPER: loop body; shifts the counter into the object's field and increments the counter.
      nonlocal i
      state.field = state.field * 10 + i
      i += 1

    def set_state(loop_vars):
      # [FUNCTION DOC]
      # HELPER: state setter; assigns the counter and the object's field from the loop variables.
      nonlocal i
      i, state.field = loop_vars

    i = 0
    n = constant_op.constant(5)
    control_flow.while_stmt(
        test=lambda: i < n,
        body=body,
        get_state=lambda: (i, state.field),
        set_state=set_state,
        symbol_names=('i', 'state.field'),
        opts={})

    self.assertEqual(i, 5)
    self.assertEqual(state.field, 1234)
    self.assertOpCreated('StatelessWhile')

  def test_python(self):
    # [FUNCTION DOC]
    # While loop with a plain Python condition and no state getter/setter. Asserts counter and
    # accumulator results (accumulator 1234) and that no graph ops were created.
    def body():
      # [FUNCTION DOC]
      # HELPER: loop body; shifts the counter into the accumulator then increments the counter.
      nonlocal i, s
      s = s * 10 + i
      i += 1

    i = 0
    s = 0
    n = 5
    control_flow.while_stmt(
        test=lambda: i < n,
        body=body,
        get_state=None,
        set_state=None,
        symbol_names=('i', 's'),
        opts={})

    self.assertEqual(s, 1234)
    self.assertNoOpsCreated()

  def test_python_with_tensor_state(self):
    # [FUNCTION DOC]
    # While loop with a plain Python condition but a tensor accumulator. Asserts the loop runs
    # as Python (counter 5, accumulator 1234) and that no while ops of either flavor appear.
    def body():
      # [FUNCTION DOC]
      # HELPER: loop body; shifts the counter into the accumulator then increments the counter.
      nonlocal i, s
      s = s * 10 + i
      i += 1

    i = 0
    s = constant_op.constant(0)
    n = 5
    control_flow.while_stmt(
        test=lambda: i < n,
        body=body,
        get_state=None,
        set_state=None,
        symbol_names=('i', 's'),
        opts={})

    self.assertEqual(i, 5)
    self.assertEqual(s, 1234)
    self.assertOpsNotCreated(('While', 'StatelessWhile'))

  def test_python_while_infinite(self):
    # [FUNCTION DOC]
    # Skipped in optimized mode. Temporarily lowers the Python iteration limit to 100 and runs
    # a while loop whose condition is always true. Asserts a value error mentioning the
    # iteration limit is raised.
    if not __debug__:
      self.skipTest('Feature disabled in optimized mode.')
    with test.mock.patch.object(control_flow, 'PYTHON_MAX_ITERATIONS', 100):
      with self.assertRaisesRegex(ValueError, 'iteration limit'):
        control_flow.while_stmt(
            test=lambda: True,
            body=lambda: None,
            get_state=None,
            set_state=None,
            symbol_names=(),
            opts={})

  def test_python_for_infinite(self):
    # [FUNCTION DOC]
    # Skipped in optimized mode. Temporarily lowers the Python iteration limit to 100 and runs
    # a for loop over 101 items. Asserts a value error mentioning the iteration limit.
    # (The name says "for" and lives in the while test class.)
    if not __debug__:
      self.skipTest('Feature disabled in optimized mode.')
    with test.mock.patch.object(control_flow, 'PYTHON_MAX_ITERATIONS', 100):
      with self.assertRaisesRegex(ValueError, 'iteration limit'):
        control_flow.for_stmt(
            iter_=range(101),
            extra_test=None,
            body=lambda i: None,
            get_state=None,
            set_state=None,
            symbol_names=(),
            opts={})

  def test_python_while_large_unroll_warning(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: despite its name (while), it actually exercises the for operator over a
    # custom generator; the matching test below named for "for" exercises the while operator.
    # Skipped in optimized mode. Lowers the unroll warning threshold to 10, captures standard
    # output with logging echoed to it, and runs a for loop over a generator yielding 11
    # constant tensors in a fresh graph. Asserts the captured log mentions a large unrolled
    # loop together with a constant op.
    if not __debug__:
      self.skipTest('Feature disabled in optimized mode.')
    with test.mock.patch.object(
        control_flow, 'INEFFICIENT_UNROLL_MIN_ITERATIONS', 10):
      with ops.Graph().as_default():
        out_capturer = io.StringIO()
        with test.mock.patch.object(sys, 'stdout', out_capturer):
          with test.mock.patch.object(ag_logging, 'echo_log_to_stdout', True):
            def custom_iterator():
              # [FUNCTION DOC]
              # HELPER: generator yielding eleven constant tensors with values 0 to 10.
              for i in range(11):
                c = constant_op.constant(i)
                yield c

            i = 0
            control_flow.for_stmt(
                iter_=custom_iterator(),
                extra_test=None,
                body=lambda i: None,
                get_state=None,
                set_state=None,
                symbol_names=(),
                opts={})
        self.assertTrue(re.match(
            r'.* Large unrolled loop.*Const.*', out_capturer.getvalue()))

  def test_python_for_large_unroll_warning(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: despite its name (for), it actually exercises the while operator; the test
    # above named for "while" exercises the for operator.
    # Skipped in optimized mode. Lowers the unroll warning threshold to 10, captures standard
    # output with logging echoed to it, and runs a Python while loop (limit 100) whose body
    # adds tensors in a fresh graph. Asserts the captured log mentions a large unrolled loop
    # together with an add op.
    if not __debug__:
      self.skipTest('Feature disabled in optimized mode.')
    with test.mock.patch.object(
        control_flow, 'INEFFICIENT_UNROLL_MIN_ITERATIONS', 10):
      with ops.Graph().as_default():
        out_capturer = io.StringIO()
        with test.mock.patch.object(sys, 'stdout', out_capturer):
          with test.mock.patch.object(ag_logging, 'echo_log_to_stdout', True):
            def body():
              # [FUNCTION DOC]
              # HELPER: loop body; creates an add op on the counter and increments the counter.
              nonlocal i
              gen_math_ops.add(i, 1)
              i += 1

            i = 0
            control_flow.while_stmt(
                test=lambda: i < 100,
                body=body,
                get_state=None,
                set_state=None,
                symbol_names=('i',),
                opts={})
        self.assertTrue(re.match(
            r'.* Large unrolled loop.*Add.*', out_capturer.getvalue()))

  def _basic_loop(self, init_value, body_fn):
    # [FUNCTION DOC]
    # HELPER: shared by the illegal-input tests of this class. Runs a while loop over a tensor
    # limit of 5 with a counter and one state variable starting at init_value; each iteration
    # sets the state to body_fn(counter, state) (a callable of two arguments) and increments
    # the counter. Returns the final state. Errors from the operator propagate.

    def body():
      # [FUNCTION DOC]
      # HELPER: loop body; applies the supplied function and increments the counter.
      nonlocal i, s
      s = body_fn(i, s)
      i += 1

    def set_state(loop_vars):
      # [FUNCTION DOC]
      # HELPER: state setter; stores the counter and state into the enclosing variables.
      nonlocal i, s
      i, s = loop_vars

    i = 0
    n = constant_op.constant(5)
    s = init_value
    control_flow.while_stmt(
        test=lambda: i < n,
        body=body,
        get_state=lambda: (i, s),
        set_state=set_state,
        symbol_names=('i', 's'),
        opts={})
    return s

  def test_tensor_illegal_input(self):
    # [FUNCTION DOC]
    # Asserts a value error when the initial state is nothing/null and when it is an
    # undefined-variable placeholder never assigned by the body.
    with self.assertRaisesRegex(ValueError, "'s' is not allowed to be None"):
      self._basic_loop(None, lambda i, s: s)
    with self.assertRaisesRegex(ValueError, "'s' must be defined"):
      self._basic_loop(variable_operators.Undefined(''), lambda i, s: s)

  def test_tensor_none_output(self):
    # [FUNCTION DOC]
    # Asserts a value error ("is None at the end") when the body sets the state to nothing/null.
    with self.assertRaisesRegex(ValueError, "'s' is None at the end"):
      self._basic_loop(0, lambda i, s: None)

  def test_tensor_dtype_change(self):
    # [FUNCTION DOC]
    # Asserts a type error mentioning dtype float32 when the body changes an integer state to
    # a float.
    with self.assertRaisesRegex(TypeError, "'s'.* dtype float32 after"):
      self._basic_loop(0, lambda i, s: 1.0)

  def test_tensor_shape_change(self):
    # [FUNCTION DOC]
    # Asserts a value error mentioning shape (1,) when the body changes a scalar state to a
    # one-element vector.
    with self.assertRaisesRegex(ValueError, r"'s'.* shape \(1,\) after"):
      self._basic_loop(0, lambda i, s: np.array([1], dtype=np.int32))

  def _fixed_while_loop(self, cond_fn):
    # [FUNCTION DOC]
    # HELPER: shared by the condition validation tests. Runs a while loop with a tensor
    # counter starting at zero and incremented each iteration, whose condition is
    # cond_fn(counter) (a callable taking the counter). Returns the final counter.
    def test_():
      # [FUNCTION DOC]
      # HELPER: loop condition; evaluates the supplied function on the counter.
      return cond_fn(s)

    def body():
      # [FUNCTION DOC]
      # HELPER: loop body; increments the counter.
      nonlocal s
      s += 1

    def set_state(loop_vars):
      # [FUNCTION DOC]
      # HELPER: state setter; stores the single loop variable into the enclosing variable.
      nonlocal s
      s, = loop_vars

    s = constant_op.constant(0)
    control_flow.while_stmt(
        test=test_,
        body=body,
        get_state=lambda: (s,),
        set_state=set_state,
        symbol_names=('s',),
        opts={})
    return s

  def _assertFixedLoopResult(self, cond, expected):
    # [FUNCTION DOC]
    # HELPER: runs the fixed while loop with the given condition function and asserts the final
    # counter equals the expected value.
    def test_fn():
      # [FUNCTION DOC]
      # HELPER: wrapper calling the fixed while loop helper with the condition.
      return self._fixed_while_loop(cond)
    self.assertEqual(test_fn(), expected)

  def test_tensor_legal_cond_scalar(self):
    # [FUNCTION DOC]
    # Asserts scalar boolean tensor conditions are accepted: a constant false condition leaves
    # the counter at 0, and a counter-below-2 condition ends at 2.
    self._assertFixedLoopResult(lambda s: constant_op.constant(False), 0)
    self._assertFixedLoopResult(lambda s: s < 2, 2)

  def test_tensor_legal_cond_single_element_nd(self):
    # [FUNCTION DOC]
    # Asserts single-element multi-dimensional boolean conditions are accepted, including one
    # of unknown rank; both are false so the counter stays at 0.
    self._assertFixedLoopResult(lambda s: constant_op.constant([[False]]), 0)
    self._assertFixedLoopResult(lambda s: _unranked_item(False), 0)

  def _assertCondCheckFails(self, cond):
    # [FUNCTION DOC]
    # HELPER: asserts that running the fixed while loop with the given condition function
    # raises a value error saying the loop condition must be a boolean tensor.
    with self.assertRaisesRegex(
        ValueError, 'condition of while loop expected to be `tf.bool`'):
      self._fixed_while_loop(cond)

  def test_tensor_illegal_cond_not_bool(self):
    # [FUNCTION DOC]
    # Asserts conditions that are not boolean (an integer constant, and the integer counter
    # itself) are rejected.
    self._assertCondCheckFails(lambda s: constant_op.constant(1))
    self._assertCondCheckFails(lambda s: s)

  def test_tensor_illegal_cond_not_single_element(self):
    # [FUNCTION DOC]
    # Asserts conditions that are not single-element (an integer vector and a boolean vector
    # of two items) are rejected with the boolean condition error.
    self._assertCondCheckFails(lambda s: constant_op.constant([1, 2, 3]))
    self._assertCondCheckFails(lambda s: constant_op.constant([True, False]))

  def test_tensor_illegal_cond_not_single_element_dynamic_shape(self):
    # [FUNCTION DOC]
    # Runs the fixed while loop with a condition of partially unknown shape that actually has
    # two elements; the static check cannot reject it so it fails at run time. Asserts a
    # runtime invalid-argument error mentioning the requested shape. A TODO notes the error
    # message is poor.
    self._fixed_while_loop(lambda s: _partial_shaped_bools())
    # TODO(mdan): This error is quite bad. Measure the cost of an assertion.
    self.assertRaisesRuntime(
        errors_impl.InvalidArgumentError, 'requested shape has 1')


class IfStmtTest(testing.AutoGraphTestCase):
  # [CLASS DOC]
  # Tests for the if-statement operator. Covers tensor conditions (staged stateless if ops)
  # with zero, one or several outputs, plain Python conditions (no ops), branch output errors
  # (None, undefined, dtype mismatch) and validation of the condition's dtype and shape.

  def test_tensor(self):
    # [FUNCTION DOC]
    # Runs the if operator with a tensor condition whose branches assign 1 or -1 to one output
    # variable. Asserts true gives 1, false gives -1 and a stateless if op was created.

    def test_fn(cond):
      # [FUNCTION DOC]
      # HELPER: builds and runs the conditional for the given condition; returns the variable.
      def body():
        # [FUNCTION DOC]
        # HELPER: true branch; sets the variable to 1.
        nonlocal i
        i = constant_op.constant(1)

      def orelse():
        # [FUNCTION DOC]
        # HELPER: false branch; sets the variable to -1.
        nonlocal i
        i = constant_op.constant(-1)

      def set_state(cond_vars):
        # [FUNCTION DOC]
        # HELPER: state setter; stores the single output into the enclosing variable.
        nonlocal i
        i, = cond_vars

      i = None
      control_flow.if_stmt(
          cond=cond,
          body=body,
          orelse=orelse,
          get_state=lambda: (i,),
          set_state=set_state,
          symbol_names=('i',),
          nouts=1)
      return i

    self.assertEqual(test_fn(constant_op.constant(True)), 1)
    self.assertEqual(test_fn(constant_op.constant(False)), -1)
    self.assertOpCreated('StatelessIf')

  def test_tensor_no_outputs(self):
    # [FUNCTION DOC]
    # Runs the if operator with a tensor condition and zero declared outputs, where the
    # branches assign differently typed values (integer vs float). Because no outputs are
    # declared the mismatch is ignored: asserts the variable stays nothing/null for both
    # conditions and a stateless if op was created.

    def test_fn(cond):
      # [FUNCTION DOC]
      # HELPER: builds and runs the conditional with zero outputs; returns the variable.
      def body():
        # [FUNCTION DOC]
        # HELPER: true branch; sets the variable to an integer 1.
        nonlocal i
        i = constant_op.constant(1)

      def orelse():
        # [FUNCTION DOC]
        # HELPER: false branch; sets the variable to a float -1.
        nonlocal i
        i = constant_op.constant(-1.0)

      def set_state(cond_vars):
        # [FUNCTION DOC]
        # HELPER: state setter; stores the single value into the enclosing variable.
        nonlocal i
        i, = cond_vars

      i = None
      control_flow.if_stmt(
          cond=cond,
          body=body,
          orelse=orelse,
          get_state=lambda: (i,),
          set_state=set_state,
          symbol_names=('i',),
          nouts=0)
      return i

    self.assertIsNone(test_fn(constant_op.constant(True)))
    self.assertIsNone(test_fn(constant_op.constant(False)))
    self.assertOpCreated('StatelessIf')

  def test_tensor_multiple_returns(self):
    # [FUNCTION DOC]
    # Runs the if operator with a tensor condition and two output variables. Asserts true gives
    # (1, 2), false gives (-1, -2) and a stateless if op was created.

    def test_fn(cond):
      # [FUNCTION DOC]
      # HELPER: builds and runs the conditional with two outputs; returns both variables.
      def body():
        # [FUNCTION DOC]
        # HELPER: true branch; sets the variables to 1 and 2.
        nonlocal i, j
        i = constant_op.constant(1)
        j = constant_op.constant(2)

      def orelse():
        # [FUNCTION DOC]
        # HELPER: false branch; sets the variables to -1 and -2.
        nonlocal i, j
        i = constant_op.constant(-1)
        j = constant_op.constant(-2)

      def set_state(cond_vars):
        # [FUNCTION DOC]
        # HELPER: state setter; stores the two outputs into the enclosing variables.
        nonlocal i, j
        i, j = cond_vars

      i, j = None, None
      control_flow.if_stmt(
          cond=cond,
          body=body,
          orelse=orelse,
          get_state=lambda: (i, j),
          set_state=set_state,
          symbol_names=('i', 'j'),
          nouts=2)
      return i, j

    self.assertEqual(test_fn(constant_op.constant(True)), (1, 2))
    self.assertEqual(test_fn(constant_op.constant(False)), (-1, -2))
    self.assertOpCreated('StatelessIf')

  def test_python(self):
    # [FUNCTION DOC]
    # Runs the if operator with a plain Python boolean condition and no state getter/setter.
    # Asserts true gives 1, false gives -1 and no graph ops were created.

    def test_fn(cond):
      # [FUNCTION DOC]
      # HELPER: builds and runs the conditional for the given Python condition; returns the
      # variable.
      def body():
        # [FUNCTION DOC]
        # HELPER: true branch; sets the variable to 1.
        nonlocal i
        i = 1

      def orelse():
        # [FUNCTION DOC]
        # HELPER: false branch; sets the variable to -1.
        nonlocal i
        i = -1

      i = None
      control_flow.if_stmt(
          cond=cond,
          body=body,
          orelse=orelse,
          get_state=None,
          set_state=None,
          symbol_names=('i',),
          nouts=1)
      return i

    self.assertEqual(test_fn(True), 1)
    self.assertEqual(test_fn(False), -1)
    self.assertNoOpsCreated()

  def test_python_multiple_returns(self):
    # [FUNCTION DOC]
    # Same as the plain Python condition test but with two output variables. Asserts (1, 2) for
    # true, (-1, -2) for false and that no graph ops were created.

    def test_fn(cond):
      # [FUNCTION DOC]
      # HELPER: builds and runs the conditional with two outputs; returns both variables.
      def body():
        # [FUNCTION DOC]
        # HELPER: true branch; sets the variables to 1 and 2.
        nonlocal i, j
        i = 1
        j = 2

      def orelse():
        # [FUNCTION DOC]
        # HELPER: false branch; sets the variables to -1 and -2.
        nonlocal i, j
        i = -1
        j = -2

      i, j = None, None
      control_flow.if_stmt(
          cond=cond,
          body=body,
          orelse=orelse,
          get_state=None,
          set_state=None,
          symbol_names=('i', 'j'),
          nouts=2)
      return i, j

    self.assertEqual(test_fn(True), (1, 2))
    self.assertEqual(test_fn(False), (-1, -2))
    self.assertNoOpsCreated()

  def _basic_cond(self, body_fn, else_fn):
    # [FUNCTION DOC]
    # HELPER: shared by the branch-output tests. Runs the if operator with a constant true
    # tensor condition and one output variable starting at 0; the main branch sets it to
    # body_fn() and the else branch to else_fn() (both callables with no arguments).
    # Returns the variable. Errors from the operator propagate to the caller.
    def body():
      # [FUNCTION DOC]
      # HELPER: true branch; sets the variable from the supplied function.
      nonlocal x
      x = body_fn()

    def orelse():
      # [FUNCTION DOC]
      # HELPER: false branch; sets the variable from the supplied function.
      nonlocal x
      x = else_fn()

    def set_state(cond_vars):
      # [FUNCTION DOC]
      # HELPER: state setter; stores the single output into the enclosing variable.
      nonlocal x
      x, = cond_vars

    x = 0
    control_flow.if_stmt(
        cond=constant_op.constant(True),
        body=body,
        orelse=orelse,
        get_state=lambda: (x,),
        set_state=set_state,
        symbol_names=('x',),
        nouts=1)
    return x

  def test_tensor_none_output(self):
    # [FUNCTION DOC]
    # Asserts a value error naming the main branch when it leaves the output as nothing/null,
    # and another naming the else branch when that one does.
    with self.assertRaisesRegex(
        ValueError, "'x' is None at the end of the main branch"):
      self._basic_cond(lambda: None, lambda: 1)
    with self.assertRaisesRegex(
        ValueError, "'x' is None at the end of the else branch"):
      self._basic_cond(lambda: 1, lambda: None)

  def test_tensor_undefined_output(self):
    # [FUNCTION DOC]
    # Asserts a value error saying the output must also be initialized in the main branch (or
    # the else branch) when the other branch leaves it as an undefined placeholder.
    with self.assertRaisesRegex(
        ValueError, "'x' must also be initialized in the main branch"):
      self._basic_cond(lambda: variable_operators.Undefined('x'), lambda: 1)
    with self.assertRaisesRegex(
        ValueError, "'x' must also be initialized in the else branch"):
      self._basic_cond(lambda: 1, lambda: variable_operators.Undefined('s'))

  def test_tensor_dtype_change(self):
    # [FUNCTION DOC]
    # Asserts a type error mentioning int32 versus float32 when the two branches produce
    # different dtypes for the same output.
    with self.assertRaisesRegex(
        TypeError, "'x' has dtype int32.*but.*float32"):
      self._basic_cond(lambda: 1, lambda: 1.0)

  def _fixed_cond(self, cond_val):
    # [FUNCTION DOC]
    # HELPER: shared by the condition validation tests. Runs the if operator with the given
    # condition value, one output variable starting at 0, main branch setting 1 and else branch
    # setting -1. Returns the variable.
    def body():
      # [FUNCTION DOC]
      # HELPER: true branch; sets the variable to 1.
      nonlocal x
      x = 1

    def orelse():
      # [FUNCTION DOC]
      # HELPER: false branch; sets the variable to -1.
      nonlocal x
      x = -1

    def set_state(cond_vars):
      # [FUNCTION DOC]
      # HELPER: state setter; stores the single output into the enclosing variable.
      nonlocal x
      x, = cond_vars

    x = 0
    control_flow.if_stmt(
        cond=cond_val,
        body=body,
        orelse=orelse,
        get_state=lambda: (x,),
        set_state=set_state,
        symbol_names=('x',),
        nouts=1)
    return x

  def _assertFixedCondResult(self, cond, expected):
    # [FUNCTION DOC]
    # HELPER: runs the fixed conditional with the given condition and asserts the resulting
    # variable equals the expected value.
    def test_fn():
      # [FUNCTION DOC]
      # HELPER: wrapper calling the fixed conditional helper with the condition.
      return self._fixed_cond(cond)
    self.assertEqual(test_fn(), expected)

  def test_tensor_legal_cond_scalar(self):
    # [FUNCTION DOC]
    # Asserts scalar boolean tensor conditions are accepted: true yields 1 and false yields -1.
    self._assertFixedCondResult(constant_op.constant(True), 1)
    self._assertFixedCondResult(constant_op.constant(False), -1)

  def test_tensor_legal_cond_single_element_nd(self):
    # [FUNCTION DOC]
    # Asserts single-element multi-dimensional boolean conditions (including ones of unknown
    # rank) are accepted: true variants yield 1 and false variants yield -1.
    self._assertFixedCondResult(constant_op.constant([[True]]), 1)
    self._assertFixedCondResult(constant_op.constant([[False]]), -1)
    self._assertFixedCondResult(_unranked_item(True), 1)
    self._assertFixedCondResult(_unranked_item(False), -1)

  def _assertCondCheckFails(self, cond):
    # [FUNCTION DOC]
    # HELPER: asserts that running the fixed conditional with the given condition raises a
    # value error saying the if condition must be a boolean tensor.
    with self.assertRaisesRegex(
        ValueError, 'condition of if statement expected to be `tf.bool`'):
      self._fixed_cond(cond)

  def test_tensor_illegal_cond_not_bool(self):
    # [FUNCTION DOC]
    # Asserts an integer tensor condition is rejected.
    self._assertCondCheckFails(constant_op.constant(1))

  def test_tensor_illegal_cond_not_single_element(self):
    # [FUNCTION DOC]
    # Asserts multi-element conditions (an integer vector and a boolean vector of two items)
    # are rejected.
    self._assertCondCheckFails(constant_op.constant([1, 2, 3]))
    self._assertCondCheckFails(constant_op.constant([True, False]))

  def test_tensor_illegal_cond_not_single_element_dynamic_shape(self):
    # [FUNCTION DOC]
    # Runs the fixed conditional with a boolean condition of partially unknown shape that
    # really has two elements; the static check cannot reject it so it fails at run time.
    # Asserts a runtime invalid-argument error mentioning the requested shape. A TODO notes the
    # error message is poor.
    self._fixed_cond(_partial_shaped_bools())
    # TODO(mdan): This error is quite bad. Measure the cost of an assertion.
    self.assertRaisesRuntime(
        errors_impl.InvalidArgumentError, 'requested shape has 1')

if __name__ == '__main__':
  test.main()
