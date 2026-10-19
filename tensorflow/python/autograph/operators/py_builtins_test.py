# [MODULE DOC]
# Unit tests for the py_builtins operators module: overloads of abs, float, int, len, print, min,
# max, range, enumerate, zip, map, next, filter, any, all and sorted for plain Python values,
# tensors and datasets, plus the evaluators that run eval, locals, globals and super in the
# original function context. Tests run in both graph and eager modes via a class decorator.
# Defines a small base class used by the super tests. Runs the TensorFlow test runner as a script.
#
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
"""Tests for py_builtins module."""

import io
import sys


from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.core import function_wrappers
from tensorflow.python.autograph.operators import data_structures
from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.platform import test


class TestBase:
  # [CLASS DOC]
  # HELPER: base class for the super-related tests.
  # Plain class with one method that adds twenty to its argument; subclasses in the tests
  # override it and check which implementation is reached through the super overload.

  def overridden_method(self, x):
    # [FUNCTION DOC]
    # HELPER: target of the super tests. Returns x plus twenty.
    return x + 20


@test_util.run_all_in_graph_and_eager_modes
class PyBuiltinsTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the builtin overloads. The class decorator runs every test in both graph and
  # eager modes. Shared fixture: a method that builds a function-scope object with a fixed scope
  # name, used by the original-context tests.

  def test_abs(self):
    # [FUNCTION DOC]
    # Checks abs on a Python integer, a scalar tensor (value 1) and a rank-one tensor
    # (values 1, 2, 3), evaluating the tensors in a session.
    self.assertEqual(py_builtins.abs_(-1), 1)
    with self.cached_session() as sess:
      t = py_builtins.abs_(constant_op.constant(-1))
      self.assertEqual(self.evaluate(t), 1)
      t = py_builtins.abs_(constant_op.constant([-1, 2, -3]))
      self.assertAllEqual(self.evaluate(t), [1, 2, 3])

  def test_abs_dataset(self):
    # [FUNCTION DOC]
    # Applies abs to a dataset of integers (-1, 2, 3) via the registered dataset override and
    # reads three items from a one-shot iterator; expects 1, 2, 3.
    dataset = dataset_ops.DatasetV2.from_tensor_slices([-1, 2, 3])
    dataset = py_builtins.abs_(dataset)
    iterator = dataset_ops.make_one_shot_iterator(dataset)
    with self.cached_session() as sess:
      self.assertAllEqual(self.evaluate(iterator.get_next()), 1)
      self.assertAllEqual(self.evaluate(iterator.get_next()), 2)
      self.assertAllEqual(self.evaluate(iterator.get_next()), 3)

  def test_abs_dataset_zipped(self):
    # [FUNCTION DOC]
    # Applies abs to a zip of two integer datasets and expects each pair to have absolute values:
    # (1, 1), (2, 2), (3, 3).
    dataset_1 = dataset_ops.DatasetV2.from_tensor_slices([-1, 2, 3])
    dataset_2 = dataset_ops.DatasetV2.from_tensor_slices([1, -2, 3])
    dataset = dataset_ops.DatasetV2.zip((dataset_1, dataset_2))
    dataset = py_builtins.abs_(dataset)
    iterator = dataset_ops.make_one_shot_iterator(dataset)
    with self.cached_session() as sess:
      self.assertAllEqual(self.evaluate(iterator.get_next()), (1, 1))
      self.assertAllEqual(self.evaluate(iterator.get_next()), (2, 2))
      self.assertAllEqual(self.evaluate(iterator.get_next()), (3, 3))

  def test_abs_dataset_mixed(self):
    # [FUNCTION DOC]
    # Applies abs to a nested dataset (a plain dataset zipped with a zip of two datasets) and
    # expects, for steps one to three, the first component to be the step number and the second
    # component to be the pair of the step number twice.
    dataset_1 = dataset_ops.DatasetV2.from_tensor_slices([-1, 2, 3])
    dataset_2 = dataset_ops.DatasetV2.from_tensor_slices([1, -2, 3])
    dataset_3 = dataset_ops.DatasetV2.from_tensor_slices([-1, -2, -3])
    dataset_4 = dataset_ops.DatasetV2.zip((dataset_1, dataset_2))
    dataset = dataset_ops.DatasetV2.zip((dataset_3, dataset_4))
    dataset = py_builtins.abs_(dataset)
    iterator = dataset_ops.make_one_shot_iterator(dataset)
    with self.cached_session() as sess:
      for i in range(1, 4):
        actual = self.evaluate(iterator.get_next())
        self.assertAllEqual(actual[0], i)
        self.assertAllEqual(actual[1], (i, i))

  def test_float(self):
    # [FUNCTION DOC]
    # Checks float conversion of a Python integer and text, of a 64-bit integer tensor, and of a
    # string tensor holding 1.0; all expected to give 1 or 10 as floating-point values.
    self.assertEqual(py_builtins.float_(10), 10.0)
    self.assertEqual(py_builtins.float_('10.0'), 10.0)
    with self.cached_session() as sess:
      t = py_builtins.float_(constant_op.constant(1, dtype=dtypes.int64))
      self.assertEqual(self.evaluate(t), 1.0)
      st = py_builtins.float_(constant_op.constant('1.0'))
      self.assertEqual(self.evaluate(st), 1.0)

  def test_int(self):
    # [FUNCTION DOC]
    # Checks integer conversion of a Python float, of text in base two, of a 64-bit float tensor,
    # of a string tensor, and of a string tensor with an explicit base of ten.
    self.assertEqual(py_builtins.int_(10.0), 10)
    self.assertEqual(py_builtins.int_('11', 2), 3)
    with self.cached_session() as sess:
      t = py_builtins.int_(constant_op.constant(1, dtype=dtypes.float64))
      self.assertEqual(self.evaluate(t), 1)
      st = py_builtins.int_(constant_op.constant('1'))
      self.assertEqual(self.evaluate(st), 1)
      st = py_builtins.int_(constant_op.constant('1'), 10)
      self.assertEqual(self.evaluate(st), 1)

  def test_int_unsupported_base(self):
    # [FUNCTION DOC]
    # Converting a tensor with a base other than ten must raise a not-implemented error.
    t = constant_op.constant(1, dtype=dtypes.float64)
    with self.assertRaises(NotImplementedError):
      py_builtins.int_(t, 2)

  def test_len(self):
    # [FUNCTION DOC]
    # Checks length of a Python list (3), of a statically shaped tensor (3), of a tensor array of
    # size five (5) and of a tensor list built from three items (3).
    self.assertEqual(py_builtins.len_([1, 2, 3]), 3)
    with self.cached_session() as sess:
      t = py_builtins.len_(constant_op.constant([[1], [2], [3]]))
      self.assertEqual(t, 3)
      ta = py_builtins.len_(tensor_array_ops.TensorArray(dtypes.int32, size=5))
      self.assertEqual(self.evaluate(ta), 5)
      tl = py_builtins.len_(data_structures.tf_tensor_list_new([3, 4, 5]))
      self.assertEqual(self.evaluate(tl), 3)

  def test_len_dataset(self):
    # [FUNCTION DOC]
    # Checks that the length of a three-element dataset evaluates to 3, both called directly and
    # called inside a compiled function with autograph disabled.
    dataset = dataset_ops.DatasetV2.from_tensor_slices([3, 2, 1])
    self.assertEqual(self.evaluate(py_builtins.len_(dataset)), 3)

    # graph mode
    @def_function.function(autograph=False)
    def test_fn():
      # [FUNCTION DOC]
      # Builds a three-element dataset inside the compiled function and returns its length.
      dataset = dataset_ops.DatasetV2.from_tensor_slices([3, 2, 1])
      return py_builtins.len_(dataset)

    self.assertEqual(self.evaluate(test_fn()), 3)

  def test_len_dataset_infinite(self):
    # [FUNCTION DOC]
    # The length of an infinitely repeating batched dataset must fail with an invalid-argument
    # error when evaluated, both eagerly and inside a compiled function.
    dataset = dataset_ops.DatasetV2.range(5).repeat().batch(2)
    with self.assertRaises(errors_impl.InvalidArgumentError):
      _ = self.evaluate(py_builtins.len_(dataset))

    # graph mode
    @def_function.function
    def test_fn():
      # [FUNCTION DOC]
      # Builds an infinite batched dataset inside the compiled function and returns its length.
      dataset = dataset_ops.DatasetV2.range(5).repeat().batch(2)
      return py_builtins.len_(dataset)

    with self.assertRaises(errors_impl.InvalidArgumentError):
      self.evaluate(test_fn())

  def test_len_dataset_unknown(self):
    # [FUNCTION DOC]
    # The length of a dataset with unknown cardinality (after a filter) must fail with an
    # invalid-argument error when evaluated, both eagerly and inside a compiled function.
    dataset = dataset_ops.DatasetV2.range(5).filter(lambda _: True).batch(2)
    with self.assertRaises(errors_impl.InvalidArgumentError):
      _ = self.evaluate(py_builtins.len_(dataset))

    # graph mode
    @def_function.function(autograph=False)
    def test_fn():
      # [FUNCTION DOC]
      # Builds a filtered, batched dataset inside the compiled function and returns its length.
      dataset = dataset_ops.DatasetV2.range(5).filter(lambda _: True).batch(2)
      return py_builtins.len_(dataset)

    with self.assertRaises(errors_impl.InvalidArgumentError):
      self.evaluate(test_fn())

  def test_len_scalar(self):
    # [FUNCTION DOC]
    # The length of a scalar tensor must raise a value error.
    with self.assertRaises(ValueError):
      py_builtins.len_(constant_op.constant(1))

  @test_util.run_deprecated_v1
  def test_len_dynamic_shape(self):
    # [FUNCTION DOC]
    # Uses a placeholder of completely unknown shape. Feeding a three-element vector gives length
    # 3; feeding a scalar must fail at run time with an invalid-argument error.
    with self.cached_session() as sess:
      p = array_ops.placeholder(dtype=dtypes.int32, shape=None)
      t = py_builtins.len_(p)
      self.assertEqual(sess.run(t, {p: [1, 2, 3]}), 3)

      with self.assertRaises(errors_impl.InvalidArgumentError):
        t = py_builtins.len_(p)
        sess.run(t, {p: 1})

  @test_util.run_deprecated_v1
  def test_print_tensors(self):
    # [FUNCTION DOC]
    # Redirects standard output to an in-memory buffer (restored afterwards), runs a print of a
    # string tensor and an integer in a session and expects the output "test message 1" and a
    # newline.
    try:
      out_capturer = io.StringIO()
      sys.stdout = out_capturer
      with self.cached_session() as sess:
        sess.run(py_builtins.print_(constant_op.constant('test message'), 1))
        self.assertEqual(out_capturer.getvalue(), 'test message 1\n')
    finally:
      sys.stdout = sys.__stdout__

  @test_util.run_deprecated_v1
  def test_print_complex(self):
    # [FUNCTION DOC]
    # Same as the tensor print test but prints a string tensor and a list; expects the output
    # "test message [1, 2]" and a newline. Standard output is restored afterwards.
    try:
      out_capturer = io.StringIO()
      sys.stdout = out_capturer
      with self.cached_session() as sess:
        sess.run(
            py_builtins.print_(constant_op.constant('test message'), [1, 2]))
        self.assertEqual(out_capturer.getvalue(), 'test message [1, 2]\n')
    finally:
      sys.stdout = sys.__stdout__

  def test_max(self):
    # [FUNCTION DOC]
    # Checks max of a Python list (3) and of three separate Python numbers (2).
    self.assertEqual(py_builtins.max_([1, 3, 2]), 3)
    self.assertEqual(py_builtins.max_(0, 2, 1), 2)

  def test_max_tensor(self):
    # [FUNCTION DOC]
    # Checks tensor max: a scalar returns itself; a rank-one tensor reduces to its maximum (3);
    # rank-two input raises a value error; three scalar tensors give 8; a mix containing a
    # non-scalar tensor among several arguments raises a value error.
    r = py_builtins.max_(constant_op.constant(2))
    self.assertAllEqual(self.evaluate(r), 2)
    with self.assertRaises(ValueError):
      py_builtins.max_(constant_op.constant([[2]]))
    r = py_builtins.max_(constant_op.constant([1, 3, 2]))
    self.assertAllEqual(self.evaluate(r), 3)
    with self.assertRaises(ValueError):
      py_builtins.max_(constant_op.constant([[1, 3], [3, 4]]))
    r = py_builtins.max_(
        constant_op.constant(6), constant_op.constant(4),
        constant_op.constant(8))
    self.assertAllEqual(self.evaluate(r), 8)
    with self.assertRaises(ValueError):
      py_builtins.max_(
          constant_op.constant([6]), constant_op.constant(4),
          constant_op.constant(8))

  def test_min(self):
    # [FUNCTION DOC]
    # Checks min of a Python list (1) and of three separate Python numbers (0).
    self.assertEqual(py_builtins.min_([2, 1, 3]), 1)
    self.assertEqual(py_builtins.min_(2, 0, 1), 0)

  def test_min_tensor(self):
    # [FUNCTION DOC]
    # Checks tensor min: a scalar returns itself; a rank-one tensor reduces to its minimum (1);
    # rank-two input raises a value error; three scalar tensors give 4; a mix containing a
    # non-scalar tensor among several arguments raises a value error.
    r = py_builtins.min_(constant_op.constant(2))
    self.assertAllEqual(self.evaluate(r), 2)
    with self.assertRaises(ValueError):
      py_builtins.min_(constant_op.constant([[2]]))
    r = py_builtins.min_(constant_op.constant([3, 1, 2]))
    self.assertAllEqual(self.evaluate(r), 1)
    with self.assertRaises(ValueError):
      py_builtins.min_(constant_op.constant([[1, 3], [3, 4]]))
    r = py_builtins.min_(
        constant_op.constant(6), constant_op.constant(4),
        constant_op.constant(8))
    self.assertAllEqual(self.evaluate(r), 4)
    with self.assertRaises(ValueError):
      py_builtins.min_(
          constant_op.constant([6]), constant_op.constant(4),
          constant_op.constant(8))


  def test_range(self):
    # [FUNCTION DOC]
    # Checks Python-valued ranges with one, two and three arguments (the last with a negative
    # step): expects [0, 1, 2], [1, 2] and [2, 1].
    self.assertListEqual(list(py_builtins.range_(3)), [0, 1, 2])
    self.assertListEqual(list(py_builtins.range_(1, 3)), [1, 2])
    self.assertListEqual(list(py_builtins.range_(2, 0, -1)), [2, 1])

  def test_range_tensor(self):
    # [FUNCTION DOC]
    # Checks ranges where a tensor is used as stop (3), as stop with a Python start, and as a
    # negative step; expects [0, 1, 2], [1, 2] and [2, 1] when evaluated.
    with self.cached_session() as sess:
      r = py_builtins.range_(constant_op.constant(3))
      self.assertAllEqual(self.evaluate(r), [0, 1, 2])
      r = py_builtins.range_(1, constant_op.constant(3))
      self.assertAllEqual(self.evaluate(r), [1, 2])
      r = py_builtins.range_(2, 0, constant_op.constant(-1))
      self.assertAllEqual(self.evaluate(r), [2, 1])

  def test_range_tensor_empty_range(self):
    # [FUNCTION DOC]
    # A tensor range with a negative stop, and one whose stop is smaller than its start, must
    # evaluate to an empty result rather than raising an error.
    with self.session() as sess:
      r = py_builtins.range_(constant_op.constant(-3))
      self.assertAllEqual(self.evaluate(r), [])
      r = py_builtins.range_(5, constant_op.constant(2))
      self.assertAllEqual(self.evaluate(r), [])

  def test_enumerate(self):
    # [FUNCTION DOC]
    # Checks enumeration of Python lists with default start, positive start 5 and negative
    # start -3, comparing the produced pairs of index and item.
    self.assertListEqual(
        list(py_builtins.enumerate_([3, 2, 1])), [(0, 3), (1, 2), (2, 1)])
    self.assertListEqual(
        list(py_builtins.enumerate_([3, 2, 1], 5)), [(5, 3), (6, 2), (7, 1)])
    self.assertListEqual(list(py_builtins.enumerate_([-8], -3)), [(-3, -8)])

  def test_enumerate_dataset(self):
    # [FUNCTION DOC]
    # Enumerates a string dataset with a 64-bit tensor start of 20 and reads two items from a
    # one-shot iterator; expects (20, a) and (21, c) as byte strings.
    dataset = dataset_ops.DatasetV2.from_tensor_slices(['a', 'c'])
    start = constant_op.constant(20, dtype=dtypes.int64)
    dataset = py_builtins.enumerate_(dataset, start)
    iterator = dataset_ops.make_one_shot_iterator(dataset)
    with self.cached_session() as sess:
      self.assertAllEqual(self.evaluate(iterator.get_next()), (20, b'a'))
      self.assertAllEqual(self.evaluate(iterator.get_next()), (21, b'c'))

  def test_zip(self):
    # [FUNCTION DOC]
    # Zips two Python lists of equal length and of unequal length; expects pairs, with the longer
    # list truncated.
    self.assertListEqual(
        list(py_builtins.zip_([3, 2, 1], [1, 2, 3])), [(3, 1), (2, 2), (1, 3)])
    self.assertListEqual(
        list(py_builtins.zip_([4, 5, 6], [-1, -2])), [(4, -1), (5, -2)])

  def test_zip_dataset(self):
    # [FUNCTION DOC]
    # Zips two integer datasets through the overload and reads three pairs from a one-shot
    # iterator; expects (-11, -21), (-12, -22), (4, 5).
    ds1 = dataset_ops.DatasetV2.from_tensor_slices([-11, -12, 4])
    ds2 = dataset_ops.DatasetV2.from_tensor_slices([-21, -22, 5])
    ds3 = py_builtins.zip_(ds1, ds2)
    iterator = dataset_ops.make_one_shot_iterator(ds3)
    with self.cached_session() as sess:
      self.assertAllEqual(self.evaluate(iterator.get_next()), (-11, -21))
      self.assertAllEqual(self.evaluate(iterator.get_next()), (-12, -22))
      self.assertAllEqual(self.evaluate(iterator.get_next()), (4, 5))

  def test_map(self):
    # [FUNCTION DOC]
    # Maps an increment function over one list and a two-argument addition over two lists;
    # expects [5, 6, 7] and [2, 0, -2].

    def increment(x):
      # [FUNCTION DOC]
      # Returns x plus one.
      return x + 1

    add_list = lambda x, y: x + y
    self.assertListEqual(
        list(py_builtins.map_(increment, [4, 5, 6])), [5, 6, 7])
    self.assertListEqual(
        list(py_builtins.map_(add_list, [3, 2, 1], [-1, -2, -3])), [2, 0, -2])

  def test_map_dataset(self):
    # [FUNCTION DOC]
    # Maps an increment function over an integer dataset (4, 5, 6) and reads three items from a
    # one-shot iterator; expects 5, 6, 7.

    def increment(x):
      # [FUNCTION DOC]
      # Returns x plus one.
      return x + 1

    ds1 = dataset_ops.DatasetV2.from_tensor_slices([4, 5, 6])
    ds2 = py_builtins.map_(increment, ds1)
    iterator = dataset_ops.make_one_shot_iterator(ds2)
    with self.cached_session() as sess:
      self.assertAllEqual(self.evaluate(iterator.get_next()), 5)
      self.assertAllEqual(self.evaluate(iterator.get_next()), 6)
      self.assertAllEqual(self.evaluate(iterator.get_next()), 7)

  def test_map_multiple_datasets(self):
    # [FUNCTION DOC]
    # Maps a two-argument addition over two integer datasets and reads three items; expects
    # -32, -34 and 9.
    add_list = lambda x, y: x + y
    ds1 = dataset_ops.DatasetV2.from_tensor_slices([-11, -12, 4])
    ds2 = dataset_ops.DatasetV2.from_tensor_slices([-21, -22, 5])
    ds3 = py_builtins.map_(add_list, ds1, ds2)
    iterator = dataset_ops.make_one_shot_iterator(ds3)
    with self.cached_session() as sess:
      self.assertAllEqual(self.evaluate(iterator.get_next()), -32)
      self.assertAllEqual(self.evaluate(iterator.get_next()), -34)
      self.assertAllEqual(self.evaluate(iterator.get_next()), 9)

  def test_next_normal(self):
    # [FUNCTION DOC]
    # Calls next on a Python iterator of three items: yields 1, 2, 3; a fourth call raises a
    # stop-iteration error; with a default of 4 the fourth call returns 4 instead.
    iterator = iter([1, 2, 3])
    self.assertEqual(py_builtins.next_(iterator), 1)
    self.assertEqual(py_builtins.next_(iterator), 2)
    self.assertEqual(py_builtins.next_(iterator), 3)
    with self.assertRaises(StopIteration):
      py_builtins.next_(iterator)
    self.assertEqual(py_builtins.next_(iterator, 4), 4)

  def test_next_tf_iterator(self):
    # [FUNCTION DOC]
    # Exercises next on a dataset iterator inside a compiled function. Three calls give 0, 1, 2.
    # When going out of range with 64-bit defaults, two extra calls give the defaults -3 and -4.
    # When going out of range without a default, evaluation raises an out-of-range error.
    # graph-mode iterators are only supported inside tf.function.
    @def_function.function(autograph=False)
    def test_fn(go_out_of_range, with_default):
      # [FUNCTION DOC]
      # Creates an iterator over a dataset of three integers and takes three items. If
      # go_out_of_range is true, takes further items: two with defaults when with_default is true
      # (returned in the tuple), or one without a default otherwise. Returns the tuple of items.
      iterator = iter(dataset_ops.Dataset.range(3))
      retval = (
          py_builtins.next_(iterator),
          py_builtins.next_(iterator),
          py_builtins.next_(iterator),
      )
      if go_out_of_range:
        if with_default:
          retval += (
              py_builtins.next_(iterator,
                                constant_op.constant(-3, dtype=dtypes.int64)),
              py_builtins.next_(iterator,
                                constant_op.constant(-4, dtype=dtypes.int64)),
          )
        else:
          py_builtins.next_(iterator)
      return retval

    self.assertAllEqual(
        self.evaluate(test_fn(go_out_of_range=False, with_default=None)),
        (0, 1, 2))
    self.assertAllEqual(
        self.evaluate(test_fn(go_out_of_range=True, with_default=True)),
        (0, 1, 2, -3, -4))
    with self.assertRaises(errors_impl.OutOfRangeError):
      self.evaluate(test_fn(go_out_of_range=True, with_default=False))

  def test_next_tf_iterator_error_checking(self):
    # [FUNCTION DOC]
    # Passing a 32-bit default to next on an iterator over 64-bit integers must raise a type error
    # mentioning the default and int64.
    # graph-mode iterators are only supported inside tf.function.
    @def_function.function(autograph=False)
    def test_fn():
      # [FUNCTION DOC]
      # Takes one item from a one-element dataset iterator, then calls next again with an
      # incompatible 32-bit default (expected to raise at trace time).
      iterator = iter(dataset_ops.Dataset.range(1))
      py_builtins.next_(iterator)
      py_builtins.next_(iterator, constant_op.constant(-3))

    # Dataset.range defaults to int64,
    with self.assertRaisesRegex(TypeError, 'default.*int64'):
      self.evaluate(test_fn())

  def test_next_tf_iterator_error_checking_structures(self):
    # [FUNCTION DOC]
    # Uses an iterator over a dataset of dictionaries with two integer fields. Defaults with a
    # different structure (missing a field, or a list-valued field) must raise a type error about
    # the same element structure; a default with a matching structure but a 32-bit float field
    # must raise a type error mentioning float32.
    # graph-mode iterators are only supported inside tf.function.
    @def_function.function(autograph=False)
    def test_fn(default_val):
      # [FUNCTION DOC]
      # Takes one item from the dictionary dataset iterator, then calls next again with
      # default_val as the default (expected to be validated against the element structure).
      ds = dataset_ops.Dataset.range(1)
      ds = ds.map(lambda i: {'a': i + 1, 'b': i + 10})
      iterator = iter(ds)
      py_builtins.next_(iterator)
      py_builtins.next_(iterator, default_val)

    default = {
        'a': constant_op.constant(3, dtype=dtypes.int64),
    }
    with self.assertRaisesRegex(TypeError, 'same element structure'):
      test_fn(default)
    default = {
        'a': constant_op.constant(3.0),
        'b': [constant_op.constant(30), constant_op.constant(300)]
    }
    with self.assertRaisesRegex(TypeError, 'same element structure'):
      test_fn(default)
    default = {
        'a': constant_op.constant(3.0),
        'b': constant_op.constant(30, dtype=dtypes.int64),
    }
    with self.assertRaisesRegex(TypeError, 'float32'):
      test_fn(default)

  def _basic_function_scope(self):
    # [FUNCTION DOC]
    # HELPER: fixture for the original-context tests.
    # Returns a function-scope object with function name, scope name "test_scope" (which must
    # match the variable name used in the with statements of the tests) and default conversion
    # options.
    return function_wrappers.FunctionScope(
        'test_function_name',
        'test_scope',  # Note: this must match the name in the `with` statement.
        converter.ConversionOptions())

  def test_eval_in_original_context(self):
    # [FUNCTION DOC]
    # Evaluates the text naming a local variable inside a function scope; expects the local's value
    # (1) from the enclosing function.

    def test_fn():
      # [FUNCTION DOC]
      # Sets a local to 1, enters a function scope and evaluates that local's name through the
      # original-context evaluator. Returns the result.
      l = 1  # pylint:disable=unused-variable
      with self._basic_function_scope() as test_scope:
        return py_builtins.eval_in_original_context(eval, ('l',), test_scope)

    self.assertEqual(test_fn(), 1)

  def test_eval_in_original_context_inner_function(self):
    # [FUNCTION DOC]
    # Evaluates a local's name from an inner function nested in a function scope where the inner
    # function defines its own local; expects the inner value (2), i.e. the innermost frame.

    def test_fn():
      # [FUNCTION DOC]
      # Sets a local to 1, enters a function scope, defines and calls an inner function; returns
      # the inner function's result.
      l = 1  # pylint:disable=unused-variable
      with self._basic_function_scope() as test_scope:

        def inner_fn():
          # [FUNCTION DOC]
          # Sets its own local to 2 and evaluates that local's name via the original-context
          # evaluator. Returns the result.
          # Note: a user function without a top-level function scope should
          # never be found in user code; it's only possible in generated code.
          l = 2  # pylint:disable=unused-variable
          return py_builtins.eval_in_original_context(eval, ('l',), test_scope)

        return inner_fn()

    self.assertEqual(test_fn(), 2)

  def test_locals_in_original_context(self):
    # [FUNCTION DOC]
    # Fetches the locals mapping through the original-context helper inside a function scope and
    # checks it contains the enclosing function's local with value 1.

    def test_fn():
      # [FUNCTION DOC]
      # Sets a local to 1, enters a function scope and returns the locals of the original context.
      l = 1  # pylint:disable=unused-variable
      with self._basic_function_scope() as test_scope:
        return py_builtins.locals_in_original_context(test_scope)

    locs = test_fn()

    self.assertEqual(locs['l'], 1)

  def test_locals_in_original_context_inner_function(self):
    # [FUNCTION DOC]
    # Fetches the locals mapping from an inner function inside a function scope; expects the inner
    # function's own local (2), not the outer one (1).

    def test_fn():
      # [FUNCTION DOC]
      # Sets a local to 1, enters a function scope, calls an inner function and returns its result.
      l = 1  # pylint:disable=unused-variable
      with self._basic_function_scope() as test_scope:

        def inner_fn():
          # [FUNCTION DOC]
          # Sets its own local to 2 and returns the locals mapping of the original context.
          # Note: a user function without a top-level function scope should
          # never be found in user code; it's only possible in generated code.
          l = 2  # pylint:disable=unused-variable
          return py_builtins.locals_in_original_context(test_scope)

        return inner_fn()

    locs = test_fn()

    self.assertEqual(locs['l'], 2)

  def test_globals_in_original_context(self):
    # [FUNCTION DOC]
    # Fetches the globals mapping through the original-context helper inside a function scope and
    # checks it exposes this module's base test class.

    def test_fn():
      # [FUNCTION DOC]
      # Enters a function scope and returns the globals of the original context.
      with self._basic_function_scope() as test_scope:
        return py_builtins.globals_in_original_context(test_scope)

    globs = test_fn()

    self.assertIs(globs['TestBase'], TestBase)

  def test_globals_in_original_context_inner_function(self):
    # [FUNCTION DOC]
    # Fetches the globals mapping from an inner function inside a function scope and checks it
    # exposes this module's base test class.

    def test_fn():
      # [FUNCTION DOC]
      # Enters a function scope, calls an inner function and returns its result.
      with self._basic_function_scope() as test_scope:

        def inner_fn():
          # [FUNCTION DOC]
          # Returns the globals of the original context.
          # Note: a user function without a top-level function scope should
          # never be found in user code; it's only possible in generated code.
          return py_builtins.globals_in_original_context(test_scope)

        return inner_fn()

    globs = test_fn()

    self.assertIs(globs['TestBase'], TestBase)

  def test_super_in_original_context_unary_call(self):
    # [FUNCTION DOC]
    # Calls the super overload with only the class argument, which yields an unbound super object;
    # binds it to the instance manually and calls the base method with 1; expects 21 (the base
    # implementation, not the failing override).
    test_case_self = self

    class TestSubclass(TestBase):
      # [CLASS DOC]
      # Local subclass of the base test class whose override fails the test if ever reached.

      def overridden_method(self, x):
        # [FUNCTION DOC]
        # Override that fails the test: the base implementation must be used instead.
        test_case_self.fail('This should never be called.')

      def test_method(self):
        # [FUNCTION DOC]
        # Inside a function scope, obtains the unbound super object for this class, binds it to the
        # instance, and returns the result of calling the base method with 1.
        with test_case_self._basic_function_scope() as test_scope:
          test_base_unbound = py_builtins.super_in_original_context(
              super, (TestSubclass,), test_scope)
          test_base = test_base_unbound.__get__(self, TestSubclass)
          return test_base.overridden_method(1)

    tc = TestSubclass()
    self.assertEqual(tc.test_method(), 21)

  def test_super_in_original_context_binary_call(self):
    # [FUNCTION DOC]
    # Calls the super overload with explicit class and instance arguments and calls the base
    # method with 1; expects 21 (the base implementation).
    test_case_self = self

    class TestSubclass(TestBase):
      # [CLASS DOC]
      # Local subclass of the base test class whose override fails the test if ever reached.

      def overridden_method(self, x):
        # [FUNCTION DOC]
        # Override that fails the test: the base implementation must be used instead.
        test_case_self.fail('This should never be called.')

      def test_method(self):
        # [FUNCTION DOC]
        # Inside a function scope, builds the super object from explicit class and instance and
        # returns the result of calling the base method with 1.
        with test_case_self._basic_function_scope() as test_scope:
          test_base = py_builtins.super_in_original_context(
              super, (TestSubclass, self), test_scope)
          return test_base.overridden_method(1)

    tc = TestSubclass()
    self.assertEqual(tc.test_method(), 21)

  def test_super_in_original_context_niladic_call(self):
    # [FUNCTION DOC]
    # Calls the super overload with no arguments (resolved from the calling frame) and calls the
    # base method with 1; expects 21 (the base implementation).
    test_case_self = self

    class TestSubclass(TestBase):
      # [CLASS DOC]
      # Local subclass of the base test class whose override fails the test if ever reached.

      def overridden_method(self, x):
        # [FUNCTION DOC]
        # Override that fails the test: the base implementation must be used instead.
        test_case_self.fail('This should never be called.')

      def test_method(self):
        # [FUNCTION DOC]
        # Inside a function scope, obtains the zero-argument super object and returns the result of
        # calling the base method with 1.
        with test_case_self._basic_function_scope() as test_scope:
          b = py_builtins.super_in_original_context(super, (), test_scope)
          return b.overridden_method(1)

    tc = TestSubclass()
    self.assertEqual(tc.test_method(), 21)

  def test_super_in_original_context_caller_with_locals(self):
    # [FUNCTION DOC]
    # Zero-argument super resolution from a method that has additional local variables, some
    # defined inside the function scope; calls the base method with 1 (x plus y minus z); expects
    # 21.
    test_case_self = self

    class TestSubclass(TestBase):
      # [CLASS DOC]
      # Local subclass of the base test class whose override fails the test if ever reached.

      def overridden_method(self, x):
        # [FUNCTION DOC]
        # Override that fails the test: the base implementation must be used instead.
        test_case_self.fail('This should never be called.')

      def test_method(self, x):
        # [FUNCTION DOC]
        # Defines two locals both equal to 7, enters a function scope and calls the base method
        # (via zero-argument super) with the argument plus the first local minus the second.
        y = 7
        with test_case_self._basic_function_scope() as test_scope:
          z = 7
          return py_builtins.super_in_original_context(
              super, (), test_scope).overridden_method(x + y - z)

    tc = TestSubclass()
    self.assertEqual(tc.test_method(1), 21)

  def test_super_in_original_context_inner_function(self):
    # [FUNCTION DOC]
    # Zero-argument super resolution from an inner function defined inside the method within a
    # function scope; expects the base implementation (result 21 for argument 1).
    test_case_self = self

    class TestSubclass(TestBase):
      # [CLASS DOC]
      # Local subclass of the base test class whose override fails the test if ever reached.

      def overridden_method(self, x):
        # [FUNCTION DOC]
        # Override that fails the test: the base implementation must be used instead.
        test_case_self.fail('This should never be called.')

      def test_method(self, x):
        # [FUNCTION DOC]
        # Enters a function scope, defines an inner function that uses zero-argument super to call
        # the base method with x, calls it and returns the result.
        with test_case_self._basic_function_scope() as test_scope:
          # Oddly, it's sufficient to use `self` in an inner function
          # to gain access to __class__ in this scope.
          # TODO(mdan): Is this true across implementations?
          # Note: normally, it's illegal to use super() in inner functions (it
          # throws an error), but the generated code may create them.
          def inner_fn():
            # [FUNCTION DOC]
            # Returns the base method called with x, using zero-argument super resolved through
            # the function scope.
            return py_builtins.super_in_original_context(
                super, (), test_scope).overridden_method(x)

          return inner_fn()

    tc = TestSubclass()
    self.assertEqual(tc.test_method(1), 21)

  def test_super_in_original_context_inner_lambda(self):
    # [FUNCTION DOC]
    # Zero-argument super resolution from a lambda defined inside the method within a function
    # scope; expects the base implementation (result 21 for argument 1).
    test_case_self = self

    class TestSubclass(TestBase):
      # [CLASS DOC]
      # Local subclass of the base test class whose override fails the test if ever reached.

      def overridden_method(self, x):
        # [FUNCTION DOC]
        # Override that fails the test: the base implementation must be used instead.
        test_case_self.fail('This should never be called.')

      def test_method(self, x):
        # [FUNCTION DOC]
        # Enters a function scope, defines a lambda that uses zero-argument super to call the base
        # method with x, calls it and returns the result.
        with test_case_self._basic_function_scope() as test_scope:
          # Oddly, it's sufficient to use `self` in an inner function
          # to gain access to __class__ in this scope.
          # TODO(mdan): Is this true across implementations?
          # Note: normally, it's illegal to use super() in inner functions (it
          # throws an error), but the generated code may create them.
          l = lambda: py_builtins.super_in_original_context(  # pylint:disable=g-long-lambda
              super, (), test_scope).overridden_method(x)
          return l()

    tc = TestSubclass()
    self.assertEqual(tc.test_method(1), 21)

  def test_filter(self):
    # [FUNCTION DOC]
    # Filters Python lists with an equality predicate and a less-than-three predicate; expects
    # ["b"] and [2, 1].
    self.assertListEqual(
        list(py_builtins.filter_(lambda x: x == 'b', ['a', 'b', 'c'])), ['b'])
    self.assertListEqual(
        list(py_builtins.filter_(lambda x: x < 3, [3, 2, 1])), [2, 1])

  def test_filter_dataset(self):
    # [FUNCTION DOC]
    # Filters an integer dataset (3, 2, 1) keeping values below three and reads two items from a
    # one-shot iterator; expects 2 then 1.
    dataset = dataset_ops.DatasetV2.from_tensor_slices([3, 2, 1])
    dataset = py_builtins.filter_(lambda x: x < 3, dataset)
    iterator = dataset_ops.make_one_shot_iterator(dataset)
    with self.cached_session() as sess:
      self.assertAllEqual(self.evaluate(iterator.get_next()), 2)
      self.assertAllEqual(self.evaluate(iterator.get_next()), 1)

  def test_any(self):
    # [FUNCTION DOC]
    # Checks any on Python boolean lists: true when one item is true, false when none is.
    self.assertEqual(py_builtins.any_([False, True, False]), True)
    self.assertEqual(py_builtins.any_([False, False, False]), False)

  def test_any_dataset(self):
    # [FUNCTION DOC]
    # Checks any on boolean datasets (true for one containing a true, false for all false). Datasets
    # of integers, of zipped tuples, or mixing integer and boolean components must raise a value
    # error.
    dataset_1 = dataset_ops.DatasetV2.from_tensor_slices([False, True, False])
    dataset_2 = dataset_ops.DatasetV2.from_tensor_slices([False, False, False])
    self.assertEqual(self.evaluate(py_builtins.any_(dataset_1)), True)
    self.assertEqual(self.evaluate(py_builtins.any_(dataset_2)), False)

    dataset_3 = dataset_ops.DatasetV2.from_tensor_slices([0, 1, 2])
    with self.assertRaises(ValueError):
      py_builtins.any_(dataset_3)

    dataset_4 = dataset_ops.DatasetV2.from_tensor_slices([False, True, False])
    dataset_zipped = dataset_ops.DatasetV2.zip((dataset_4, dataset_4))
    with self.assertRaises(ValueError):
      py_builtins.any_(dataset_zipped)

    dataset_mixed = dataset_ops.DatasetV2.zip((dataset_3, dataset_4))
    with self.assertRaises(ValueError):
      py_builtins.any_(dataset_mixed)

  def test_all(self):
    # [FUNCTION DOC]
    # Checks all on Python boolean lists: false when one item is false, true when all are true.
    self.assertEqual(py_builtins.all_([False, True, False]), False)
    self.assertEqual(py_builtins.all_([True, True, True]), True)

  def test_all_dataset(self):
    # [FUNCTION DOC]
    # Checks all on boolean datasets (false for one containing a false, true for all true).
    # Datasets of integers, of zipped tuples, or mixing integer and boolean components must raise
    # a value error.
    dataset_1 = dataset_ops.DatasetV2.from_tensor_slices([False, True, False])
    dataset_2 = dataset_ops.DatasetV2.from_tensor_slices([True, True, True])
    self.assertEqual(self.evaluate(py_builtins.all_(dataset_1)), False)
    self.assertEqual(self.evaluate(py_builtins.all_(dataset_2)), True)

    dataset_3 = dataset_ops.DatasetV2.from_tensor_slices([0, 1, 2])
    with self.assertRaises(ValueError):
      py_builtins.all_(dataset_3)

    dataset_4 = dataset_ops.DatasetV2.from_tensor_slices([False, True, False])
    dataset_zipped = dataset_ops.DatasetV2.zip((dataset_4, dataset_4))
    with self.assertRaises(ValueError):
      py_builtins.all_(dataset_zipped)

    dataset_mixed = dataset_ops.DatasetV2.zip((dataset_3, dataset_4))
    with self.assertRaises(ValueError):
      py_builtins.all_(dataset_mixed)

  def test_sorted(self):
    # [FUNCTION DOC]
    # Checks sorted on Python lists: default ordering, with a negating key, with reverse, with
    # both key and reverse, and with a sum key over nested lists.
    self.assertListEqual(py_builtins.sorted_([2, 3, 1]), [1, 2, 3])
    self.assertListEqual(
        py_builtins.sorted_([2, 3, 1], key=lambda x: -x), [3, 2, 1])
    self.assertListEqual(
        py_builtins.sorted_([2, 3, 1], reverse=True), [3, 2, 1])
    self.assertListEqual(
        py_builtins.sorted_([2, 3, 1], key=lambda x: -x, reverse=True),
        [1, 2, 3])
    self.assertAllEqual(
        py_builtins.sorted_([[4, 3], [2, 1]], key=lambda x: sum(x)),
        [[2, 1], [4, 3]])

  def test_sorted_tensor(self):
    # [FUNCTION DOC]
    # Checks sorted on a rank-one integer tensor with default order, key, reverse, and both
    # (results evaluated and compared as lists). For a rank-two tensor, sorting without a key or
    # with a negating key must raise a value error, while a reduce-sum key works and yields rows
    # ordered by sum.
    iterable_1 = constant_op.constant([2, 3, 1])
    self.assertListEqual(
        list(self.evaluate(py_builtins.sorted_(iterable_1))), [1, 2, 3])
    self.assertListEqual(
        list(self.evaluate(py_builtins.sorted_(iterable_1, key=lambda x: -x))),
        [3, 2, 1])
    self.assertListEqual(
        list(self.evaluate(py_builtins.sorted_(iterable_1, reverse=True))),
        [3, 2, 1])
    self.assertListEqual(
        list(
            self.evaluate(
                py_builtins.sorted_(iterable_1, key=lambda x: -x,
                                    reverse=True))), [1, 2, 3])

    iterable_2 = constant_op.constant([[4, 3], [2, 1]])
    with self.assertRaises(ValueError):
      py_builtins.sorted_(iterable_2)
    with self.assertRaises(ValueError):
      py_builtins.sorted_(iterable_2, key=lambda x: -x)
    self.assertAllEqual(
        list(
            self.evaluate(
                py_builtins.sorted_(
                    iterable_2, key=lambda x: math_ops.reduce_sum(x)))),
        [[2, 1], [4, 3]])


if __name__ == '__main__':
  test.main()
