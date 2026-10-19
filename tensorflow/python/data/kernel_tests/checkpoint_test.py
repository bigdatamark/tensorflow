# [MODULE DOC]
# Test module for checkpointing tf.data iterators. Covers graph-mode saving and restoring of
# iterator state through serialize/deserialize operations, eager-mode saving and restoring
# with the object-based checkpoint API (including async checkpointing, multiple iterators,
# exhausted iterators, reshuffled and modified datasets), and checks that datasets with
# stateful functions cannot be checkpointed (or can, under a warn policy for external state).
# No import-time side effects other than defining the test class.
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
"""Tests for checkpointing tf.data iterators."""
import os

from absl.testing import parameterized
from tensorflow.python.checkpoint import checkpoint as trackable_utils
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.data.experimental.ops import grouping
from tensorflow.python.data.experimental.ops import interleave_ops
from tensorflow.python.data.experimental.ops import scan_ops
from tensorflow.python.data.experimental.ops import take_while_ops
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.eager import test
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile


# TODO(jsimsa): Add missing test combinations.
class CheckpointTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for saving and restoring dataset iterator state, in graph mode via explicit save and
  # restore operations writing a file, and in eager mode via the checkpoint API. Also covers
  # non-checkpointable (stateful) dataset transformations. Provides small helpers for the
  # graph-mode save and restore operations and the checkpoint file prefix.

  def tearDown(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the removal call is applied lazily through a mapping object that is never
    # consumed, so the checkpoint files matching the prefix are in fact not deleted.
    # Looks up checkpoint files matching the iterator prefix in the temporary directory, then
    # runs the parent teardown. Returns nothing.
    prefix = self._iterator_checkpoint_prefix()
    pattern = prefix + "*"
    files = gfile.Glob(pattern)
    map(gfile.Remove, files)
    super(CheckpointTest, self).tearDown()

  def _iterator_checkpoint_prefix(self):
    # [FUNCTION DOC]
    # HELPER: returns the file path prefix (inside the test temporary directory) used by the
    # graph-mode save and restore operations.
    return os.path.join(self.get_temp_dir(), "iterator")

  def _save_op(self, iterator_resource):
    # [FUNCTION DOC]
    # HELPER: builds a graph operation that serializes the iterator state (given the iterator
    # resource handle) and writes it to the checkpoint prefix file. Returns the write operation.
    iterator_state_variant = gen_dataset_ops.serialize_iterator(
        iterator_resource)
    save_op = io_ops.write_file(
        self._iterator_checkpoint_prefix(),
        parsing_ops.serialize_tensor(iterator_state_variant))
    return save_op

  def _restore_op(self, iterator_resource):
    # [FUNCTION DOC]
    # HELPER: builds a graph operation that reads the checkpoint prefix file, parses the state
    # and loads it into the iterator resource (given as argument). Returns the restore operation.
    iterator_state_variant = parsing_ops.parse_tensor(
        io_ops.read_file(self._iterator_checkpoint_prefix()), dtypes.variant)
    restore_op = gen_dataset_ops.deserialize_iterator(iterator_resource,
                                                      iterator_state_variant)
    return restore_op

  @combinations.generate(test_base.graph_only_combinations())
  def testSaveRestore(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): iterate a range from 2 to 10 up to 5, save the state, and in a new
    # graph and session restore it; expects the remaining values 5 to 9 and then end of data.
    # Repeats in a single session by saving, re-initializing and restoring.

    def _build_graph(start, stop):
      # [FUNCTION DOC]
      # HELPER: builds an initializable iterator over a range and the save and restore
      # operations. Takes start and stop; returns the initializer, next-element tensor, save
      # operation and restore operation.
      iterator = dataset_ops.make_initializable_iterator(
          dataset_ops.Dataset.range(start, stop))
      init_op = iterator.initializer
      get_next = iterator.get_next()
      save_op = self._save_op(iterator._iterator_resource)
      restore_op = self._restore_op(iterator._iterator_resource)
      return init_op, get_next, save_op, restore_op

    # Saving and restoring in different sessions.
    start = 2
    stop = 10
    break_point = 5
    with ops.Graph().as_default() as g:
      init_op, get_next, save_op, _ = _build_graph(start, stop)
      with self.session(graph=g) as sess:
        sess.run(variables.global_variables_initializer())
        sess.run(init_op)
        for i in range(start, break_point):
          self.assertEqual(i, sess.run(get_next))
        sess.run(save_op)

    with ops.Graph().as_default() as g:
      init_op, get_next, _, restore_op = _build_graph(start, stop)
      with self.session(graph=g) as sess:
        sess.run(init_op)
        sess.run(restore_op)
        for i in range(break_point, stop):
          self.assertEqual(i, sess.run(get_next))
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)

    # Saving and restoring in same session.
    with ops.Graph().as_default() as g:
      init_op, get_next, save_op, restore_op = _build_graph(start, stop)
      with self.session(graph=g) as sess:
        sess.run(variables.global_variables_initializer())
        sess.run(init_op)
        for i in range(start, break_point):
          self.assertEqual(i, sess.run(get_next))
        sess.run(save_op)
        sess.run(init_op)
        sess.run(restore_op)
        for i in range(break_point, stop):
          self.assertEqual(i, sess.run(get_next))
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)

  @combinations.generate(test_base.graph_only_combinations())
  def testInitThenRestore(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): save the state after part of a range, then in a new graph run the
    # initializer before the restore operation. Expects restoring an already initialized
    # iterator to work and produce the remaining values.
    # Note: Calling init_op before restore_op is redundant. This test just makes
    # sure we do not fail if restore is called on an already initialized
    # iterator resource.

    def _build_graph(start, stop):
      # [FUNCTION DOC]
      # HELPER: builds an initializable iterator over a range and its save and restore
      # operations. Returns the initializer, next-element tensor, save and restore operations.
      dataset = dataset_ops.Dataset.range(start, stop)
      iterator = dataset_ops.make_initializable_iterator(dataset)
      init_op = iterator.initializer
      get_next = iterator.get_next()
      save_op = self._save_op(iterator._iterator_resource)
      restore_op = self._restore_op(iterator._iterator_resource)
      return init_op, get_next, save_op, restore_op

    # Saving and restoring in different sessions.
    start = 2
    stop = 10
    break_point = 5
    with ops.Graph().as_default() as g:
      init_op, get_next, save_op, _ = _build_graph(start, stop)
      with self.session(graph=g) as sess:
        sess.run(variables.global_variables_initializer())
        sess.run(init_op)
        for i in range(start, break_point):
          self.assertEqual(i, sess.run(get_next))
        sess.run(save_op)

    with ops.Graph().as_default() as g:
      init_op, get_next, _, restore_op = _build_graph(start, stop)
      with self.session(graph=g) as sess:
        sess.run(init_op)
        sess.run(restore_op)
        for i in range(break_point, stop):
          self.assertEqual(i, sess.run(get_next))
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)

  @combinations.generate(test_base.graph_only_combinations())
  def testMultipleSaves(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): save at element 5, restore in a new graph and read to element 7,
    # save again, and restore in a third graph. Expects the remaining values 7 to 9 and then end
    # of data, showing a restored iterator can be saved again.

    def _build_graph(start, stop):
      # [FUNCTION DOC]
      # HELPER: builds an initializable iterator over a range and its save and restore
      # operations. Returns the initializer, next-element tensor, save and restore operations.
      iterator = dataset_ops.make_initializable_iterator(
          dataset_ops.Dataset.range(start, stop))
      init_op = iterator.initializer
      get_next = iterator.get_next()
      save_op = self._save_op(iterator._iterator_resource)
      restore_op = self._restore_op(iterator._iterator_resource)
      return init_op, get_next, save_op, restore_op

    start = 2
    stop = 10
    break_point1 = 5
    break_point2 = 7

    with ops.Graph().as_default() as g:
      init_op, get_next, save_op, _ = _build_graph(start, stop)
      with self.session(graph=g) as sess:
        sess.run(variables.global_variables_initializer())
        sess.run(init_op)
        for i in range(start, break_point1):
          self.assertEqual(i, sess.run(get_next))
        sess.run(save_op)

    with ops.Graph().as_default() as g:
      init_op, get_next, save_op, restore_op = _build_graph(start, stop)
      with self.session(graph=g) as sess:
        sess.run(init_op)
        sess.run(restore_op)
        for i in range(break_point1, break_point2):
          self.assertEqual(i, sess.run(get_next))
        sess.run(save_op)

    break_point2 = 7
    with ops.Graph().as_default() as g:
      init_op, get_next, save_op, restore_op = _build_graph(start, stop)
      with self.session(graph=g) as sess:
        sess.run(init_op)
        sess.run(restore_op)
        for i in range(break_point2, stop):
          self.assertEqual(i, sess.run(get_next))
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)

  @combinations.generate(test_base.graph_only_combinations())
  def testSaveRestoreWithRepeat(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): a range repeated for 5 epochs. Restoring before any checkpoint
    # exists must raise a not-found error. Then iterate part way into the third epoch, save, and
    # restore in a new graph; expects the rest of the sequence and then end of data.

    def _build_graph(start, stop, num_epochs):
      # [FUNCTION DOC]
      # HELPER: like the plain version but the range is repeated num_epochs times. Returns the
      # initializer, next-element tensor, save operation and restore operation.
      iterator = dataset_ops.make_initializable_iterator(
          dataset_ops.Dataset.range(start, stop).repeat(num_epochs))
      init_op = iterator.initializer
      get_next = iterator.get_next()
      save_op = self._save_op(iterator._iterator_resource)
      restore_op = self._restore_op(iterator._iterator_resource)
      return init_op, get_next, save_op, restore_op

    start = 2
    stop = 10
    num_epochs = 5
    break_range = 5
    break_epoch = 3
    with ops.Graph().as_default() as g:
      init_op, get_next, save_op, restore_op = _build_graph(
          start, stop, num_epochs)
      with self.session(graph=g) as sess:
        sess.run(variables.global_variables_initializer())
        sess.run(init_op)
        # Note: There is no checkpoint saved currently so a NotFoundError is
        # raised.
        with self.assertRaises(errors.NotFoundError):
          sess.run(init_op)
          sess.run(restore_op)
        for _ in range(break_epoch - 1):
          for i in range(start, stop):
            self.assertEqual(i, sess.run(get_next))
        for i in range(start, break_range):
          self.assertEqual(i, sess.run(get_next))
        sess.run(save_op)

    with ops.Graph().as_default() as g:
      init_op, get_next, _, restore_op = _build_graph(start, stop, num_epochs)
      with self.session(graph=g) as sess:
        sess.run(init_op)
        sess.run(restore_op)
        for i in range(break_range, stop):
          self.assertEqual(i, sess.run(get_next))
        for _ in range(break_epoch, num_epochs):
          for i in range(start, stop):
            self.assertEqual(i, sess.run(get_next))
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)

  @combinations.generate(test_base.graph_only_combinations())
  def testSaveRestoreExhaustedIterator(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): restore with no checkpoint raises a not-found error; then consume
    # all 5 epochs, observe end of data and save. In a new graph, restoring must give an iterator
    # that is still exhausted (out-of-range error on the first read).

    def _build_graph(start, stop, num_epochs):
      # [FUNCTION DOC]
      # HELPER: builds an iterator over a range repeated num_epochs times with its save and
      # restore operations. Returns the initializer, next-element tensor, save and restore ops.
      iterator = dataset_ops.make_initializable_iterator(
          dataset_ops.Dataset.range(start, stop).repeat(num_epochs))
      init_op = iterator.initializer
      get_next = iterator.get_next()
      save_op = self._save_op(iterator._iterator_resource)
      restore_op = self._restore_op(iterator._iterator_resource)
      return init_op, get_next, save_op, restore_op

    start = 2
    stop = 10
    num_epochs = 5
    with ops.Graph().as_default() as g:
      init_op, get_next, save_op, restore_op = _build_graph(
          start, stop, num_epochs)
      with self.session(graph=g) as sess:
        sess.run(variables.global_variables_initializer())
        sess.run(init_op)
        # Note: There is no checkpoint saved currently so a NotFoundError is
        # raised.
        with self.assertRaises(errors.NotFoundError):
          sess.run(init_op)
          sess.run(restore_op)
        for _ in range(num_epochs):
          for i in range(start, stop):
            self.assertEqual(i, sess.run(get_next))
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)
        sess.run(save_op)

    with ops.Graph().as_default() as g:
      init_op, get_next, _, restore_op = _build_graph(start, stop, num_epochs)
      with self.session(graph=g) as sess:
        sess.run(init_op)
        sess.run(restore_op)
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)

  @combinations.generate(combinations.times(
      test_base.eager_only_combinations(),
      combinations.combine(enable_async_ckpt=[True, False])
  ))
  def testSaveRestoreOneShotIterator(self, enable_async_ckpt):
    # [FUNCTION DOC]
    # Scenario (eager): iterate a squared, batched dataset, read one batch, save a checkpoint
    # (optionally asynchronous), read two more, restore and re-read. Expects the batches after the
    # save point to be replayed and finally an out-of-range error. Writes files to the temporary
    # directory.
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    dataset = dataset_ops.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6]).map(
        math_ops.square).batch(2)
    iterator = iter(dataset)
    get_next = iterator.get_next
    ckpt_options = checkpoint_options.CheckpointOptions(
        experimental_enable_async_checkpoint=enable_async_ckpt)
    checkpoint = trackable_utils.Checkpoint(iterator=iterator)
    self.assertAllEqual([1, 4], get_next())
    save_path = checkpoint.save(checkpoint_prefix, options=ckpt_options)
    self.assertAllEqual([9, 16], get_next())
    self.assertAllEqual([25, 36], get_next())
    checkpoint.restore(save_path).run_restore_ops()
    self.assertAllEqual([9, 16], get_next())
    self.assertAllEqual([25, 36], get_next())
    with self.assertRaises(errors.OutOfRangeError):
      get_next()

  @combinations.generate(combinations.times(
      test_base.eager_only_combinations(),
      combinations.combine(enable_async_ckpt=[True, False])
  ))
  def testSaveRestoreMultipleIterator(self, enable_async_ckpt):
    # [FUNCTION DOC]
    # Scenario (eager): three iterators (two over one dataset, one over another) are advanced
    # by different amounts and saved in one checkpoint; after more reads, restoring returns each
    # iterator to its saved position. Parameter selects asynchronous checkpointing.
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    dataset = dataset_ops.Dataset.from_tensor_slices(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    dataset = dataset.map(math_ops.square).batch(2)
    iterator_1 = iter(dataset)
    get_next_1 = iterator_1.get_next
    iterator_2 = iter(dataset)
    get_next_2 = iterator_2.get_next
    dataset_2 = dataset_ops.Dataset.range(10)
    iterator_3 = iter(dataset_2)
    get_next_3 = iterator_3.get_next
    ckpt_options = checkpoint_options.CheckpointOptions(
        experimental_enable_async_checkpoint=enable_async_ckpt)
    checkpoint = trackable_utils.Checkpoint(
        iterator_1=iterator_1, iterator_2=iterator_2, iterator_3=iterator_3)
    self.assertAllEqual([1, 4], get_next_1())
    self.assertAllEqual(0, get_next_3())
    self.assertAllEqual(1, get_next_3())
    self.assertAllEqual(2, get_next_3())
    save_path = checkpoint.save(checkpoint_prefix, options=ckpt_options)
    self.assertAllEqual([1, 4], get_next_2())
    self.assertAllEqual([9, 16], get_next_2())
    self.assertAllEqual(3, get_next_3())
    checkpoint.restore(save_path).run_restore_ops()
    self.assertAllEqual([9, 16], get_next_1())
    self.assertAllEqual([1, 4], get_next_2())
    self.assertAllEqual(3, get_next_3())

  @combinations.generate(combinations.times(
      test_base.eager_only_combinations(),
      combinations.combine(enable_async_ckpt=[True, False])
  ))
  def testRestoreExhaustedIterator(self, enable_async_ckpt):
    # [FUNCTION DOC]
    # Scenario (eager): a three-element range is saved after two elements, restored and read to
    # the end, saved again when exhausted and restored; expects the restored iterator to raise an
    # out-of-range error. Parameter selects asynchronous checkpointing.
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    dataset = dataset_ops.Dataset.range(3)
    iterator = iter(dataset)
    get_next = iterator.get_next
    ckpt_options = checkpoint_options.CheckpointOptions(
        experimental_enable_async_checkpoint=enable_async_ckpt)
    checkpoint = trackable_utils.Checkpoint(iterator=iterator)
    self.assertAllEqual(0, get_next())
    self.assertAllEqual(1, get_next())
    save_path = checkpoint.save(checkpoint_prefix, options=ckpt_options)
    self.assertAllEqual(2, get_next())
    checkpoint.restore(save_path).run_restore_ops()
    self.assertAllEqual(2, get_next())
    save_path = checkpoint.save(checkpoint_prefix, options=ckpt_options)
    checkpoint.restore(save_path).run_restore_ops()
    with self.assertRaises(errors.OutOfRangeError):
      get_next()

  @combinations.generate(test_base.eager_only_combinations())
  def testRestoreInReconstructedIteratorInitializable(self):
    # [FUNCTION DOC]
    # Scenario (eager): five rounds that each restore the latest checkpoint (or initialize if
    # none), read two elements and save. Expects the sequence to continue without repeats,
    # i.e. element value 2 times round plus offset.
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    dataset = dataset_ops.Dataset.range(10)
    iterator = iter(dataset)
    get_next = iterator.get_next
    checkpoint = trackable_utils.Checkpoint(iterator=iterator)
    for i in range(5):
      checkpoint.restore(
          checkpoint_management.latest_checkpoint(
              checkpoint_directory)).initialize_or_restore()
      for j in range(2):
        self.assertEqual(i * 2 + j, self.evaluate(get_next()))
      checkpoint.save(file_prefix=checkpoint_prefix)

  @combinations.generate(test_base.eager_only_combinations())
  def testSaveRestoreReshuffleDataset(self):
    # [FUNCTION DOC]
    # Scenario (eager): a range of 10 shuffled with reshuffling each iteration. Reads 5, saves,
    # reads 5 more, restores and reads 5 again. Expects the two later reads to contain the same
    # elements and to differ from the first five.
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.shuffle(10, reshuffle_each_iteration=True)
    iterator = iter(dataset)
    ckpt = trackable_utils.Checkpoint(
        step=variables.Variable(0), iterator=iterator)
    manager = checkpoint_management.CheckpointManager(
        ckpt, self.get_temp_dir(), max_to_keep=3)

    iter1 = [next(iterator).numpy() for _ in range(5)]

    manager.save()
    iter2 = [next(iterator).numpy() for _ in range(5)]

    ckpt.restore(manager.latest_checkpoint)
    iter3 = [next(iterator).numpy() for _ in range(5)]

    self.assertNotEqual(iter1, iter2)
    self.assertCountEqual(iter2, iter3)

  @combinations.generate(test_base.eager_only_combinations())
  def testSaveRestoreModifiedDataset(self):
    # [FUNCTION DOC]
    # Scenario (eager): save the iterator of one dataset, then restore that checkpoint into the
    # iterator of a different dataset definition. Expects a not-found error stating the dataset
    # definition must not change.
    ckpt_dir = self.get_temp_dir()
    dataset = dataset_ops.Dataset.range(10)
    iterator = iter(dataset)
    ckpt = trackable_utils.Checkpoint(iterator=iterator)
    manager = checkpoint_management.CheckpointManager(
        ckpt, ckpt_dir, max_to_keep=3)

    for _ in range(5):
      next(iterator)
    manager.save()

    # Define a different dataset and try to restore into its iterator.
    dataset = dataset_ops.Dataset.from_tensor_slices([1, 2, 3])
    iterator = iter(dataset)
    ckpt = trackable_utils.Checkpoint(iterator=iterator)
    manager = checkpoint_management.CheckpointManager(
        ckpt, ckpt_dir, max_to_keep=3)
    with self.assertRaisesRegex(
        errors.NotFoundError,
        "Make sure the dataset definition has not changed"):
      ckpt.restore(manager.latest_checkpoint)

  def _assertNotCheckpointable(self, dataset):
    # [FUNCTION DOC]
    # HELPER: asserts that a dataset's iterator cannot be saved. Creates an iterator for the
    # dataset, attaches it to a checkpoint with a manager, and expects saving to raise a
    # failed-precondition error.
    iterator = iter(dataset)
    ckpt = trackable_utils.Checkpoint(
        step=variables.Variable(0), iterator=iterator)
    manager = checkpoint_management.CheckpointManager(
        ckpt, self.get_temp_dir(), max_to_keep=3)
    with self.assertRaises(errors.FailedPreconditionError):
      manager.save()

  @staticmethod
  def _statefulInt64Func(_):
    # [FUNCTION DOC]
    # HELPER: a stateful function (uses random numbers) that ignores its input and returns a
    # random scalar 64-bit integer drawn from a range of one value (always zero).
    return random_ops.random_uniform((), 0, 1, dtypes.int64)

  @staticmethod
  def _statefulBoolFunc(_):
    # [FUNCTION DOC]
    # HELPER: a stateful function (uses random numbers) that ignores its input and returns a
    # boolean scalar that is always true (the random integer is zero, which is less than one).
    return random_ops.random_uniform((), 0, 1, dtypes.int64) < 1

  @staticmethod
  def _statefulDatasetFunc(_):
    # [FUNCTION DOC]
    # HELPER: a stateful function that ignores its input, draws a random scalar integer (always
    # zero) and returns an empty range dataset of that size.
    x = random_ops.random_uniform((), 0, 1, dtypes.int64)
    return dataset_ops.Dataset.range(x)

  @combinations.generate(test_base.eager_only_combinations())
  def testStatefulFilterNotCheckpointable(self):
    # [FUNCTION DOC]
    # Scenario (eager): a filter with a stateful predicate; expects the iterator not to be
    # checkpointable.
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.filter(self._statefulBoolFunc)
    self._assertNotCheckpointable(dataset)

  @combinations.generate(test_base.eager_only_combinations())
  def testStatefulFlatMapNotCheckpointable(self):
    # [FUNCTION DOC]
    # Scenario (eager): a flat-map with a stateful function; expects the iterator not to be
    # checkpointable.
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.flat_map(self._statefulDatasetFunc)
    self._assertNotCheckpointable(dataset)

  @combinations.generate(test_base.eager_only_combinations())
  def testStatefulInterleaveNotCheckpointable(self):
    # [FUNCTION DOC]
    # Scenario (eager): an interleave with a stateful function; expects the iterator not to be
    # checkpointable.
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.interleave(self._statefulDatasetFunc)
    self._assertNotCheckpointable(dataset)

  @combinations.generate(test_base.eager_only_combinations())
  def testStatefulMapNotCheckpointable(self):
    # [FUNCTION DOC]
    # Scenario (eager): a map with a stateful function; expects the iterator not to be
    # checkpointable.
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.map(self._statefulBoolFunc)
    self._assertNotCheckpointable(dataset)

  @combinations.generate(test_base.eager_only_combinations())
  def testStatefulParallelInterleaveNotCheckpointable(self):
    # [FUNCTION DOC]
    # Scenario (eager): a parallel interleave (two calls) with a stateful function; expects the
    # iterator not to be checkpointable.
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.interleave(
        self._statefulDatasetFunc, num_parallel_calls=2)
    self._assertNotCheckpointable(dataset)

  @combinations.generate(test_base.eager_only_combinations())
  def testStatefulParallelMapNotCheckpointable(self):
    # [FUNCTION DOC]
    # Scenario (eager): a parallel map (two calls) with a stateful function; expects the iterator
    # not to be checkpointable.
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.map(self._statefulBoolFunc, num_parallel_calls=2)
    self._assertNotCheckpointable(dataset)

  @combinations.generate(test_base.eager_only_combinations())
  def testStatefulGroupByReducerNotCheckpointable(self):
    # [FUNCTION DOC]
    # Scenario (eager): group-by-reducer datasets where exactly one of the key, init, reduce or
    # finalize functions is stateful in each case; expects every resulting dataset to be
    # not checkpointable.
    stateful_key_func = self._statefulInt64Func
    key_func = lambda _: math_ops.cast(0, dtypes.int64)
    stateful_init_func = self._statefulBoolFunc
    init_func = lambda x: True
    stateful_reduce_func = lambda _, x: self._statefulBoolFunc(x)
    reduce_func = lambda _, x: True
    stateful_finalize_func = self._statefulBoolFunc
    finalize_func = lambda x: True

    test_cases = [
        (stateful_key_func, init_func, reduce_func, finalize_func),
        (key_func, stateful_init_func, reduce_func, finalize_func),
        (key_func, init_func, stateful_reduce_func, finalize_func),
        (key_func, init_func, reduce_func, stateful_finalize_func),
    ]
    for key_func, init_func, reduce_func, finalize_func in test_cases:
      dataset = dataset_ops.Dataset.range(10)
      reducer = grouping.Reducer(init_func, reduce_func, finalize_func)
      dataset = dataset.apply(grouping.group_by_reducer(key_func, reducer))
      self._assertNotCheckpointable(dataset)

  @combinations.generate(test_base.eager_only_combinations())
  def testStatefulGroupByWindowNotCheckpointable(self):
    # [FUNCTION DOC]
    # Scenario (eager): group-by-window datasets where exactly one of the key, reduce or window
    # size functions is stateful in each case; expects every dataset to be not checkpointable.
    stateful_key_func = self._statefulInt64Func
    key_func = lambda _: math_ops.cast(0, dtypes.int64)
    stateful_reduce_func = lambda _, x: self._statefulDatasetFunc(x)
    reduce_func = lambda _, x: x
    stateful_window_func = self._statefulInt64Func
    window_func = lambda x: math_ops.cast(0, dtypes.int64)

    test_cases = [
        (stateful_key_func, reduce_func, window_func),
        (key_func, stateful_reduce_func, window_func),
        (key_func, reduce_func, stateful_window_func),
    ]
    for key_func_fn, reduce_func_fn, window_func in test_cases:
      dataset = dataset_ops.Dataset.range(10)
      dataset = dataset.apply(
          grouping.group_by_window(
              key_func_fn, reduce_func_fn, window_size_func=window_func))
      self._assertNotCheckpointable(dataset)

  @combinations.generate(test_base.eager_only_combinations())
  def testStatefulMapAndBatchNotCheckpointable(self):
    # [FUNCTION DOC]
    # Scenario (eager): a stateful map followed by batch (which may fuse into map-and-batch);
    # expects the iterator not to be checkpointable.
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.map(self._statefulBoolFunc)
    dataset = dataset.batch(2)
    self._assertNotCheckpointable(dataset)

  @combinations.generate(test_base.eager_only_combinations())
  def testStatefulExperimentalParallelInterleaveNotCheckpointable(self):
    # [FUNCTION DOC]
    # Scenario (eager): the experimental parallel interleave with a stateful function; expects
    # the iterator not to be checkpointable.
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.apply(
        interleave_ops.parallel_interleave(self._statefulDatasetFunc, 2))
    self._assertNotCheckpointable(dataset)

  @combinations.generate(test_base.eager_only_combinations())
  def testStatefulScanNotCheckpointable(self):
    # [FUNCTION DOC]
    # Scenario (eager): a scan whose function calls a stateful helper; expects the iterator not
    # to be checkpointable.
    dataset = dataset_ops.Dataset.range(10)

    def stateful_scan(state, element):
      # [FUNCTION DOC]
      # HELPER: scan function that keeps the state unchanged and emits the stateful boolean
      # function applied to the element.
      return state, self._statefulBoolFunc(element)

    dataset = dataset.apply(scan_ops.scan(0, stateful_scan))
    self._assertNotCheckpointable(dataset)

  @combinations.generate(test_base.eager_only_combinations())
  def testStatefulTakeWhileNotCheckpointable(self):
    # [FUNCTION DOC]
    # Scenario (eager): take-while with a stateful predicate; expects the iterator not to be
    # checkpointable.
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.apply(take_while_ops.take_while(self._statefulBoolFunc))
    self._assertNotCheckpointable(dataset)

  @combinations.generate(test_base.eager_only_combinations())
  def testStatefulExternalPolicy(self):
    # [FUNCTION DOC]
    # Scenario (eager): a map calling an eager Python function (external state), with the
    # external-state policy set to warn. Expects checkpointing to succeed: elements read after
    # the save are replayed after restoring, followed by an out-of-range error.
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    dataset = dataset_ops.Dataset.range(4)

    def fn(x):
      # [FUNCTION DOC]
      # HELPER: Python function returning the square of its input.
      return x * x

    dataset = dataset.map(
        lambda x: script_ops.eager_py_func(fn, [x], dtypes.int64))

    options = options_lib.Options()
    options.experimental_external_state_policy = (
        options_lib.ExternalStatePolicy.WARN)
    dataset = dataset.with_options(options)

    iterator = iter(dataset)
    get_next = iterator.get_next
    checkpoint = trackable_utils.Checkpoint(iterator=iterator)
    self.assertEqual(0, get_next().numpy())
    self.assertEqual(1, get_next().numpy())
    save_path = checkpoint.save(checkpoint_prefix)
    self.assertEqual(4, get_next().numpy())
    self.assertEqual(9, get_next().numpy())
    checkpoint.restore(save_path).run_restore_ops()
    self.assertEqual(4, get_next().numpy())
    self.assertEqual(9, get_next().numpy())
    with self.assertRaises(errors.OutOfRangeError):
      get_next()

if __name__ == "__main__":
  test.main()
