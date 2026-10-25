# [MODULE DOC]
# Unit tests for the legacy queue-based input pipeline helpers in the training input module
# (imported here under the short alias inp): match_filenames_once, limit_epochs, input_producer,
# string_input_producer, range_input_producer, slice_input_producer, the dictionary conversion
# helpers, and the batching family batch, maybe_batch, batch_join, maybe_batch_join,
# shuffle_batch, maybe_shuffle_batch, shuffle_batch_join and maybe_shuffle_batch_join.
# Each test builds a graph, starts queue runners, drains the resulting queues and checks the
# values, shapes, sparse tensor handling, shared queue names, out-of-range termination and
# argument validation errors.  Queue pipelines are not supported under eager execution (TF2
# uses tf.data), so the tests explicitly create private graphs; a few are restricted to graph
# mode (TF1) by a decorator.
# Importing the module has no side effects beyond defining the test classes; running it as a
# script invokes the standard TensorFlow test runner.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for training.input."""

import itertools
import os

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test as test_lib
from tensorflow.python.platform import tf_logging
from tensorflow.python.training import coordinator
from tensorflow.python.training import input as inp
from tensorflow.python.training import queue_runner_impl
from tensorflow.python.util import compat


# Queue-based input pipelines are not supported when eager execution is enabled.
# Please use tf.data instead in TF2.
class MatchFilenamesOnceTest(test_lib.TestCase):
  # [CLASS DOC]
  # Test case for the helper that lists files matching a pattern once and stores the result in a
  # local variable.  It has no shared fixtures beyond the framework temporary directory.

  def test(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the extra files are opened for writing and never explicitly closed, relying on
    # garbage collection to flush them before the matching variables are evaluated.
    # Scenario: the temporary directory is listed, then three extra files named with a common
    # prefix and a running number are created in it.  Three match_filenames_once ops are built in a
    # fresh graph: one with a wildcard for everything, one with a single-character wildcard on the
    # prefix, and one with an exact file name.  After initializing global and local variables, the
    # test asserts that each variable holds exactly the expected set of file names (as byte
    # strings, order ignored).  Side effects: creates files in the temporary directory; the
    # files are opened for writing without being explicitly closed.
    temp_dir = self.get_temp_dir()
    filenames = [os.path.join(temp_dir, n) for n in os.listdir(temp_dir)]
    additional = [
        os.path.join(self.get_temp_dir(), "match_filenames.%d" % i)
        for i in range(3)
    ]
    for name in additional:
      open(name, "w").write("Some contents")
    filenames = list(set(filenames + additional))
    with ops.Graph().as_default(), self.cached_session():
      star = inp.match_filenames_once(os.path.join(self.get_temp_dir(), "*"))
      question = inp.match_filenames_once(
          os.path.join(self.get_temp_dir(), "match_filenames.?"))
      one = inp.match_filenames_once(additional[1])
      self.evaluate(variables.global_variables_initializer())
      variables.local_variables_initializer().run()
      self.assertItemsEqual(
          map(compat.as_bytes, filenames), self.evaluate(star))
      self.assertItemsEqual(
          map(compat.as_bytes, additional), self.evaluate(question))
      self.assertItemsEqual([compat.as_bytes(additional[1])],
                            self.evaluate(one))


class LimitEpochsTest(test_lib.TestCase):
  # [CLASS DOC]
  # Test case for the helper that lets a tensor be produced a limited number of times (epochs)
  # before raising an out-of-range error.

  def testNoLimit(self):
    # [FUNCTION DOC]
    # Scenario: limit_epochs is applied to a constant seven with no epoch limit given.  The test
    # initializes local variables and fetches the result one hundred times, asserting that the
    # value is always seven, i.e. an unlimited number of epochs is allowed.
    with ops.Graph().as_default(), self.cached_session():
      seven = constant_op.constant(7)
      seven_forever = inp.limit_epochs(seven)
      variables.local_variables_initializer().run()
      for _ in range(100):
        self.assertEqual(7, self.evaluate(seven_forever))

  def testLimit(self):
    # [FUNCTION DOC]
    # Scenario: limit_epochs is applied to a constant string with a limit of two epochs.  The first
    # two evaluations must return the string; the third must raise an out-of-range error.
    # Global and local variables are initialized first because the epoch counter is a local
    # variable.
    with ops.Graph().as_default(), self.cached_session():
      love_me = constant_op.constant("Love Me")
      love_me_two_times = inp.limit_epochs(love_me, num_epochs=2)
      self.evaluate(variables.global_variables_initializer())
      variables.local_variables_initializer().run()
      self.assertEqual(b"Love Me", self.evaluate(love_me_two_times))
      self.assertEqual(b"Love Me", self.evaluate(love_me_two_times))
      with self.assertRaises(errors_impl.OutOfRangeError):
        self.evaluate(love_me_two_times)


class InputProducerTest(test_lib.TestCase):
  # [CLASS DOC]
  # Test case for the generic input_producer, which fills a queue with the slices of a tensor
  # for a number of epochs, optionally shuffled.

  def testNoShuffle(self):
    # [FUNCTION DOC]
    # Scenario: a three by four integer matrix is fed through input_producer for two epochs
    # without shuffling.  After starting the queue runners, a bulk dequeue of all rows must equal
    # the matrix repeated twice in order, and one more dequeue must raise an out-of-range error
    # because the epoch limit is reached.  Finally all runner threads are joined.
    with ops.Graph().as_default(), self.cached_session():
      input_tensor = [[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12]]
      num_epochs = 2
      queue = inp.input_producer(
          input_tensor, num_epochs=num_epochs, shuffle=False)
      dequeue_many = queue.dequeue_many(len(input_tensor) * num_epochs)
      dequeue = queue.dequeue()
      self.evaluate(variables.global_variables_initializer())
      variables.local_variables_initializer().run()
      threads = queue_runner_impl.start_queue_runners()

      # No randomness, so just see repeated copies of the input.
      self.assertAllEqual(input_tensor * num_epochs,
                          self.evaluate(dequeue_many))

      # Reached the limit.
      with self.assertRaises(errors_impl.OutOfRangeError):
        self.evaluate(dequeue)
      for thread in threads:
        thread.join()

  def testNoShapeInference(self):
    # [FUNCTION DOC]
    # Scenario: same as the no-shuffle case, but the input is a placeholder with a default value
    # and unknown shape, so static shape inference is impossible; the element shape is supplied
    # explicitly instead.  Asserts that the rows are produced in order for two epochs, that
    # dequeuing past the limit raises an out-of-range error, and that threads are joined.
    with ops.Graph().as_default(), self.cached_session():
      # Disable shape inference for the input.
      input_value = [[1, 2, 3, 4],
                     [5, 6, 7, 8],
                     [9, 10, 11, 12]]
      input_tensor = array_ops.placeholder_with_default(input_value, shape=None)
      num_epochs = 2
      queue = inp.input_producer(
          input_tensor, element_shape=[4], num_epochs=num_epochs, shuffle=False)
      dequeue_many = queue.dequeue_many(len(input_value) * num_epochs)
      dequeue = queue.dequeue()
      self.evaluate(variables.global_variables_initializer())
      variables.local_variables_initializer().run()
      threads = queue_runner_impl.start_queue_runners()

      # No randomness, so just see repeated copies of the input.
      self.assertAllEqual(input_value * num_epochs, self.evaluate(dequeue_many))

      # Reached the limit.
      with self.assertRaises(errors_impl.OutOfRangeError):
        self.evaluate(dequeue)
      for thread in threads:
        thread.join()

  def testShapeError(self):
    # [FUNCTION DOC]
    # Scenario: input_producer is given a float placeholder whose shape is completely unknown and
    # no element shape.  Asserts that a value error mentioning a fully defined shape is raised at
    # graph construction time.
    with ops.Graph().as_default():
      input_tensor = array_ops.placeholder(dtypes.float32, None)
      with self.assertRaisesRegex(ValueError, "fully defined shape"):
        _ = inp.input_producer(input_tensor)


class StringInputProducerTest(test_lib.TestCase):
  # [CLASS DOC]
  # Test case for string_input_producer, which enqueues a list of strings (typically file names)
  # for a number of epochs, optionally shuffled.

  def testNoShuffle(self):
    # [FUNCTION DOC]
    # Scenario: six strings are produced for three epochs without shuffling.  Asserts that a bulk
    # dequeue returns the list repeated three times in order, that the next dequeue raises an
    # out-of-range error, and joins the queue runner threads.
    with ops.Graph().as_default(), self.cached_session():
      strings = [b"to", b"be", b"or", b"not", b"to", b"be"]
      num_epochs = 3
      queue = inp.string_input_producer(
          strings, num_epochs=num_epochs, shuffle=False)
      dequeue_many = queue.dequeue_many(len(strings) * num_epochs)
      dequeue = queue.dequeue()
      self.evaluate(variables.global_variables_initializer())
      variables.local_variables_initializer().run()
      threads = queue_runner_impl.start_queue_runners()

      # No randomness, so just see repeated copies of the input.
      output = self.evaluate(dequeue_many)
      self.assertAllEqual(strings * num_epochs, output)

      # Reached the limit.
      with self.assertRaises(errors_impl.OutOfRangeError):
        self.evaluate(dequeue)
      for thread in threads:
        thread.join()

  def testShuffle(self):
    # [FUNCTION DOC]
    # Scenario: three strings are produced for 600 epochs with shuffling and a fixed seed.  Each
    # epoch is dequeued as a group, joined into a single byte string, and tallied by permutation.
    # Asserts that every group is one of the six permutations (shuffling happens only within an
    # epoch) and that each permutation occurs within forty percent of the uniform expectation.
    # The observed counts are logged.  Finally asserts the out-of-range error after the last epoch.
    with ops.Graph().as_default(), self.cached_session():
      strings = [b"a", b"b", b"c"]
      num_epochs = 600
      queue = inp.string_input_producer(
          strings, num_epochs=num_epochs, shuffle=True, seed=271828)
      dequeue_many = queue.dequeue_many(len(strings))
      dequeue = queue.dequeue()
      self.evaluate(variables.global_variables_initializer())
      variables.local_variables_initializer().run()
      threads = queue_runner_impl.start_queue_runners()

      # Validate that we only shuffle the strings within an epoch and
      # count how often each possible order appears.
      expected = [b"abc", b"acb", b"bac", b"bca", b"cab", b"cba"]
      frequency = {}
      for e in expected:
        frequency[e] = 0
      for _ in range(num_epochs):
        output = self.evaluate(dequeue_many)
        key = b"".join(output)
        self.assertIn(key, expected)
        frequency[key] += 1

      # Expect an approximately even distribution over all possible orders.
      expected_frequency = num_epochs / len(expected)
      margin = expected_frequency * 0.4
      tf_logging.info("Observed counts: %s", frequency)
      for key in expected:
        value = frequency[key]
        self.assertGreater(value, expected_frequency - margin)
        self.assertLess(value, expected_frequency + margin)

      # Reached the limit.
      with self.assertRaises(errors_impl.OutOfRangeError):
        self.evaluate(dequeue)
      for thread in threads:
        thread.join()

  def testNullStringPython(self):
    # [FUNCTION DOC]
    # Scenario: string_input_producer is given an empty Python list.  Asserts that a value error
    # is raised immediately at graph construction time.  Note that no explicit graph is created
    # here, so the default graph of the cached session is used.
    # Graph-construction time check for empty string list:
    with self.cached_session():
      with self.assertRaises(ValueError):
        _ = inp.string_input_producer([])

  def testNullString(self):
    # [FUNCTION DOC]
    # Scenario: string_input_producer is given an empty string tensor, which cannot be detected at
    # construction.  The queue runner is started with a coordinator; it is expected to die on an
    # assertion about the empty input, so that dequeuing raises an out-of-range error.  The test
    # then asks the coordinator to stop and joins the threads.
    # Runtime check for empty string list.  This is slightly oblique:
    # The queue runner should die with an assertion error on the null
    # input tensor, causing the dequeue to fail with an OutOfRangeError.
    with ops.Graph().as_default(), self.cached_session():
      coord = coordinator.Coordinator()
      queue = inp.string_input_producer(
          constant_op.constant(
              [], dtype=dtypes.string))
      dequeue = queue.dequeue()
      self.evaluate(variables.global_variables_initializer())
      variables.local_variables_initializer().run()
      threads = queue_runner_impl.start_queue_runners(coord=coord)
      with self.assertRaises(errors_impl.OutOfRangeError):
        self.evaluate(dequeue)
      coord.request_stop()
      for thread in threads:
        thread.join()

  def testSharedName(self):
    # [FUNCTION DOC]
    # Scenario: a string input producer is created with an explicit shared name and queue name.
    # Asserts that the underlying queue operation carries the shared-name attribute with the given
    # value.  No queue runners are started.
    with ops.Graph().as_default(), self.cached_session():
      strings = [b"to", b"be", b"or", b"not", b"to", b"be"]
      queue = inp.string_input_producer(
          strings, shared_name="SHARED_NAME_XYZ", name="Q")
      self.assertProtoEquals("s: 'SHARED_NAME_XYZ'",
                             queue.queue_ref.op.node_def.attr["shared_name"])

  def testConstructionRace(self):
    # [FUNCTION DOC]
    # Scenario: queue runners are started on a session while the test still creates a new dequeue
    # operation for every element it reads, which mutates the graph concurrently with the runner
    # threads (a pattern not recommended but commonly used).  Two epochs of six strings without
    # shuffling are read one at a time and each must equal the expected string in order.  The
    # coordinator is then stopped and the threads are joined.  Verifies that this usage is
    # supported.
    with ops.Graph().as_default(), self.cached_session() as sess:
      strings = [b"to", b"be", b"or", b"not", b"to", b"be"]
      queue = inp.string_input_producer(strings, shuffle=False)
      coord = coordinator.Coordinator()
      threads = queue_runner_impl.start_queue_runners(sess=sess, coord=coord)
      for _ in range(2):
        for string in strings:
          # NOTE(mrry): This is not the recommended way to write
          # dequeuing code (instead you should create a single dequeue
          # op before starting the queue runners, and run it
          # repeatedly), because it leads to concurrent reading and
          # writing of the `tf.Graph` object. However, many users
          # write code this way, so we include this test to ensure
          # that we can support it.
          self.assertEqual(string, self.evaluate(queue.dequeue()))
      coord.request_stop()
      coord.join(threads)


class RangeInputProducerTest(test_lib.TestCase):
  # [CLASS DOC]
  # Test case for range_input_producer, which enqueues the integers from zero up to a given
  # size for a number of epochs, optionally shuffled.

  def testNoShuffle(self):
    # [FUNCTION DOC]
    # Scenario: range size five, three epochs, no shuffling.  Asserts that the bulk dequeue equals
    # the range repeated three times, that one more dequeue raises an out-of-range error, and
    # joins the threads.
    with ops.Graph().as_default(), self.cached_session():
      num_epochs = 3
      range_size = 5
      queue = inp.range_input_producer(
          range_size, num_epochs=num_epochs, shuffle=False)
      dequeue_many = queue.dequeue_many(range_size * num_epochs)
      dequeue = queue.dequeue()
      self.evaluate(variables.global_variables_initializer())
      variables.local_variables_initializer().run()
      threads = queue_runner_impl.start_queue_runners()

      # No randomness, so just see repeated copies of the input.
      output = self.evaluate(dequeue_many)
      self.assertAllEqual(list(range(range_size)) * num_epochs, output)

      # Reached the limit.
      with self.assertRaises(errors_impl.OutOfRangeError):
        self.evaluate(dequeue)
      for thread in threads:
        thread.join()

  def testShuffle(self):
    # [FUNCTION DOC]
    # Scenario: range size two for 200 epochs with shuffling and a fixed seed.  Each epoch is
    # encoded as a two-digit key; asserts that only the two valid orders appear and that each
    # occurs within forty percent of the uniform expectation, logs the counts, and finally asserts
    # the out-of-range error after the epoch limit.
    with ops.Graph().as_default(), self.cached_session():
      num_epochs = 200
      range_size = 2
      queue = inp.range_input_producer(
          range_size, num_epochs=num_epochs, shuffle=True, seed=314159)
      dequeue_many = queue.dequeue_many(range_size)
      dequeue = queue.dequeue()
      self.evaluate(variables.global_variables_initializer())
      variables.local_variables_initializer().run()
      threads = queue_runner_impl.start_queue_runners()

      # Validate that we only shuffle the integers within an epoch and
      # count how often each possible order appears.
      expected = [12, 21]
      frequency = {}
      for e in expected:
        frequency[e] = 0
      for _ in range(num_epochs):
        output = self.evaluate(dequeue_many)
        key = 10 * (output[0] + 1) + (output[1] + 1)
        self.assertIn(key, expected)
        frequency[key] += 1

      # Expect an approximately even distribution over all possible orders.
      expected_frequency = num_epochs / len(expected)
      margin = expected_frequency * 0.4
      tf_logging.info("Observed counts: %s", frequency)
      for key in expected:
        value = frequency[key]
        self.assertGreater(value, expected_frequency - margin)
        self.assertLess(value, expected_frequency + margin)

      # Reached the limit.
      with self.assertRaises(errors_impl.OutOfRangeError):
        self.evaluate(dequeue)
      for thread in threads:
        thread.join()

  def testSharedName(self):
    # [FUNCTION DOC]
    # Scenario: range_input_producer is created with an explicit shared name and name.  Asserts
    # that the queue operation's shared-name attribute equals the given value.
    with ops.Graph().as_default(), self.cached_session():
      range_size = 5
      queue = inp.range_input_producer(
          range_size, shared_name="SHARED_NAME_XYZ", name="Q")
      self.assertProtoEquals("s: 'SHARED_NAME_XYZ'",
                             queue.queue_ref.op.node_def.attr["shared_name"])


class SliceInputProducerTest(test_lib.TestCase):
  # [CLASS DOC]
  # Test case for slice_input_producer, which produces one slice (a tuple of the i-th elements)
  # from several parallel tensors per dequeue, for a number of epochs, optionally shuffled.

  def testNoShuffle(self):
    # [FUNCTION DOC]
    # Scenario: a list of four byte strings and a list of four integers are sliced together for
    # three epochs without shuffling.  Twelve slices are fetched one by one and transposed into
    # two sequences; asserts that both equal their sources repeated three times, that fetching
    # again raises an out-of-range error, and joins the threads.
    with ops.Graph().as_default(), self.cached_session():
      num_epochs = 3
      source_strings = [b"Alpha", b"Beta", b"Delta", b"Gamma"]
      source_ints = [2, 3, 5, 7]
      slices = inp.slice_input_producer(
          [source_strings, source_ints], num_epochs=num_epochs, shuffle=False)
      self.evaluate(variables.global_variables_initializer())
      variables.local_variables_initializer().run()
      threads = queue_runner_impl.start_queue_runners()

      # No randomness, so just see repeated copies of the input.
      num_items = len(source_strings) * num_epochs
      output = [self.evaluate(slices) for _ in range(num_items)]
      out_strings, out_ints = zip(*output)
      self.assertAllEqual(source_strings * num_epochs, out_strings)
      self.assertAllEqual(source_ints * num_epochs, out_ints)

      # Reached the limit.
      with self.assertRaises(errors_impl.OutOfRangeError):
        self.evaluate(slices)
      for thread in threads:
        thread.join()

  def testShuffle(self):
    # [FUNCTION DOC]
    # Scenario: four strings and four integers are sliced for 1200 epochs with shuffling and a
    # fixed seed.  Each epoch's four slices are joined into a key string and compared against all
    # 24 permutations of the paired elements, asserting that pairs stay together, that each
    # permutation occurs within forty percent of the uniform expectation, and that the epoch limit
    # raises an out-of-range error afterwards.  Counts are logged.
    with ops.Graph().as_default(), self.cached_session():
      num_epochs = 1200
      source_strings = ["A", "B", "D", "G"]
      source_ints = [7, 3, 5, 2]
      slices = inp.slice_input_producer(
          [source_strings, source_ints],
          num_epochs=num_epochs,
          shuffle=True,
          seed=161803)
      self.evaluate(variables.global_variables_initializer())
      variables.local_variables_initializer().run()
      threads = queue_runner_impl.start_queue_runners()

      # Validate that we only shuffle the integers within an epoch and
      # count how often each possible order appears.
      expected = [
          b",".join(x)
          for x in itertools.permutations([b"A7", b"B3", b"D5", b"G2"])
      ]
      frequency = {}
      for e in expected:
        frequency[e] = 0
      for _ in range(num_epochs):
        output = [self.evaluate(slices) for _ in range(len(source_strings))]
        key = b",".join(s + compat.as_bytes(str(i)) for s, i in output)
        self.assertIn(key, expected)
        frequency[key] += 1

      # Expect an approximately even distribution over all possible orders.
      expected_frequency = num_epochs / len(expected)
      margin = expected_frequency * 0.4
      tf_logging.info("Observed counts: %s", frequency)
      for key in expected:
        value = frequency[key]
        self.assertGreater(value, expected_frequency - margin)
        self.assertLess(value, expected_frequency + margin)

      # Reached the limit.
      with self.assertRaises(errors_impl.OutOfRangeError):
        self.evaluate(slices)
      for thread in threads:
        thread.join()

  def testSharedName(self):
    # [FUNCTION DOC]
    # Scenario: slice_input_producer is created with a shared name and a name.  Asserts that the
    # shared-name attribute is found on the underlying queue operation, reached by walking back
    # through the inputs of the first slice's operation.
    with ops.Graph().as_default(), self.cached_session():
      source_strings = ["A", "B", "D", "G"]
      source_ints = [7, 3, 5, 2]
      slices = inp.slice_input_producer(
          [source_strings, source_ints],
          shared_name="SHARED_NAME_XYZ",
          name="sip")

      self.assertProtoEquals(
          "s: 'SHARED_NAME_XYZ'",
          slices[0].op.inputs[1].op.inputs[0].op.node_def.attr["shared_name"])


class DictHelperTest(test_lib.TestCase):
  # [CLASS DOC]
  # Test case for the private conversion helpers that flatten a list or dictionary of tensors
  # into a list and rebuild the original structure.  No graph is needed.

  def testListInputs(self):
    # [FUNCTION DOC]
    # Scenario: a plain list is converted to a tensor list and back.  Asserts that the tensor list
    # equals the input and that restoring the original type yields the same list.
    l = [1, 2, 3, 11, 22, 33]
    l2 = inp._as_tensor_list(l)
    self.assertEqual(l, l2)
    l3 = inp._as_original_type(l, l2)
    self.assertEqual(l, l3)

  def testDictInputs(self):
    # [FUNCTION DOC]
    # Scenario: a dictionary with string keys is flattened.  Asserts that values are ordered by
    # sorted key (keys with a common prefix ordered naturally) and that rebuilding from the
    # flattened list returns an equal dictionary.
    d = {"a": 1, "b": 2, "c": 3, "aa": 11, "bb": 22, "cc": 33}
    l = inp._as_tensor_list(d)
    self.assertEqual([1, 11, 2, 22, 3, 33], l)
    d2 = inp._as_original_type(d, l)
    self.assertEqual(d, d2)

  def testHeterogeneousKeysDictInputs(self):
    # [FUNCTION DOC]
    # Scenario: a dictionary mixing a string, an integer and a tuple as keys is flattened.
    # Asserts that the order is determined by sorting the keys by their string representation
    # (tuple, integer, string in this case) and that the original dictionary is rebuilt.
    d = {"z": 1, 1: 42, ("a", "b"): 100}
    l = inp._as_tensor_list(d)
    self.assertEqual([100, 42, 1], l)
    d2 = inp._as_original_type(d, l)
    self.assertEqual(d, d2)


class BatchTest(test_lib.TestCase):
  # [CLASS DOC]
  # Test case for the batch and maybe_batch functions, which group single examples from a queue
  # into batches, with support for dictionaries, sparse tensors, dynamic padding, multiple
  # threads, smaller final batches and conditional enqueueing.  Shared helper methods are
  # reused by several test methods.

  def _testOneThreadHelper(self, use_dict):
    # [FUNCTION DOC]
    # HELPER: Shared body for the single-thread batching tests (list and dictionary input).
    # Parameter use_dict selects whether the examples are passed as a dictionary of dense,
    # sparse and string entries or as a list.  A counter variable counts up to thirty, a two
    # element sparse tensor and a constant string are batched with batch size ten.  For each of
    # three batches asserts the dense values, sparse indices, values and dense shape, and the
    # strings; a further fetch must raise an out-of-range error.  Returns nothing.
    with ops.Graph().as_default(), self.cached_session():
      batch_size = 10
      num_batches = 3
      zero64 = constant_op.constant(0, dtype=dtypes.int64)
      examples = variables.Variable(zero64)
      counter = examples.count_up_to(num_batches * batch_size)
      sparse_counter = sparse_tensor.SparseTensor(
          indices=array_ops.reshape(
              array_ops_stack.stack([zero64, zero64 + 1]), [2, 1]),
          values=math_ops.cast(
              array_ops_stack.stack([counter, -counter]), dtypes.float32),
          dense_shape=[2])
      if use_dict:
        batched = inp.batch(
            {
                "c": counter,
                "s": sparse_counter,
                "S": "string"
            },
            batch_size=batch_size)
        batched_fetch = [batched["c"], batched["s"], batched["S"]]
      else:
        batched = inp.batch(
            [counter, sparse_counter, "string"], batch_size=batch_size)
        batched_fetch = batched
      self.evaluate(variables.global_variables_initializer())
      variables.local_variables_initializer().run()
      threads = queue_runner_impl.start_queue_runners()

      for i in range(num_batches):
        results = self.evaluate(batched_fetch)
        self.assertAllEqual(results[0],
                            np.arange(i * batch_size, (i + 1) * batch_size))
        self.assertAllEqual(
            results[1].indices,
            np.vstack((
                np.arange(2 * batch_size) // 2,  # 0, 0, 1, 1, ...
                [0, 1] * batch_size)).T)
        #  [x, -x, x+1, -(x+1), ...]
        expected = np.arange(2 * i * batch_size, 2 * (i + 1) * batch_size) // 2
        expected *= ([1, -1] * batch_size)  # mult by [1, -1, 1, -1, ...]
        self.assertAllEqual(results[1].values, expected)
        self.assertAllEqual(results[1].dense_shape, [batch_size, 2])
        self.assertAllEqual(results[2], [b"string"] * batch_size)

      # Reached the limit.
      with self.assertRaises(errors_impl.OutOfRangeError):
        self.evaluate(batched_fetch)
      for thread in threads:
        thread.join()

  def testOneThread(self):
    # [FUNCTION DOC]
    # Runs the single-thread batching scenario with list input via the shared helper.
    self._testOneThreadHelper(use_dict=False)

  def testOneThreadDict(self):
    # [FUNCTION DOC]
    # Runs the single-thread batching scenario with dictionary input via the shared helper.
    self._testOneThreadHelper(use_dict=True)

  def testUint32DataTypes(self):
    # [FUNCTION DOC]
    # Scenario: a constant of unsigned 32-bit integers is batched with batch size two.  The queue
    # runners are started with a coordinator and the batch is evaluated once; the test only
    # verifies that this dtype is supported (no error), then stops the coordinator and joins
    # the threads.
    with ops.Graph().as_default():
      values = constant_op.constant([0, 1, 2, 3, 4, 5], dtype=dtypes.uint32)
      batched = inp.batch([values], batch_size=2)
      with self.cached_session() as sess:
        coord = coordinator.Coordinator()
        threads = queue_runner_impl.start_queue_runners(sess=sess, coord=coord)
        self.evaluate(batched)
        coord.request_stop()
        for thread in threads:
          thread.join()

  def testUint64DataTypes(self):
    # [FUNCTION DOC]
    # Scenario: same as the unsigned 32-bit case but with unsigned 64-bit integers.  Verifies
    # that batching and evaluation succeed without error for this dtype.
    with ops.Graph().as_default():
      values = constant_op.constant([0, 1, 2, 3, 4, 5], dtype=dtypes.uint64)
      batched = inp.batch([values], batch_size=2)
      with self.cached_session() as sess:
        coord = coordinator.Coordinator()
        threads = queue_runner_impl.start_queue_runners(sess=sess, coord=coord)
        self.evaluate(batched)
        coord.request_stop()
        for thread in threads:
          thread.join()

  def testOneThreadDynamicPad(self):
    # [FUNCTION DOC]
    # Scenario: a counter and a string tensor whose length equals the counter value are batched
    # with dynamic padding on and batch size ten.  For each of three batches asserts the counter
    # values and that every string row is padded with empty strings up to the longest length in
    # the batch, then asserts that the final fetch raises an out-of-range error.
    with ops.Graph().as_default(), self.cached_session():
      batch_size = 10
      num_batches = 3
      zero64 = constant_op.constant(0, dtype=dtypes.int64)
      examples = variables.Variable(zero64)
      counter = examples.count_up_to(num_batches * batch_size)
      string = array_ops.tile(["string"],
                              math_ops.cast(array_ops_stack.stack([counter]),
                                            dtypes.int32))
      self.evaluate(variables.global_variables_initializer())
      variables.local_variables_initializer().run()
      batched = inp.batch(
          [counter, string], batch_size=batch_size, dynamic_pad=True)
      threads = queue_runner_impl.start_queue_runners()

      for i in range(num_batches):
        results = self.evaluate(batched)
        expected_results = np.arange(i * batch_size, (i + 1) * batch_size)
        max_len = expected_results[-1]
        self.assertAllEqual(results[0], expected_results)
        expected_strings = [[b"string"] * rep + [b""] * (max_len - rep)
                            for rep in expected_results]
        self.assertAllEqual(results[1], expected_strings)

      # Reached the limit.
      with self.assertRaises(errors_impl.OutOfRangeError):
        self.evaluate(batched)
      for thread in threads:
        thread.join()

  def testOneThreadEnqueueMany(self):
    # [FUNCTION DOC]
    # Scenario: examples are first batched in groups of two, and that result is batched again with
    # the option that each input already holds a batch of examples to enqueue individually, and a
    # batch size of ten.  For three batches asserts dense values, sparse indices, values and
    # dense shape, and strings; then asserts the out-of-range error and joins the threads.
    with ops.Graph().as_default(), self.cached_session():
      batch_size = 10
      num_batches = 3
      zero64 = constant_op.constant(0, dtype=dtypes.int64)
      examples = variables.Variable(zero64)
      counter = examples.count_up_to(num_batches * batch_size)
      sparse_counter = sparse_tensor.SparseTensor(
          indices=array_ops.reshape(zero64, [1, 1]),
          values=array_ops_stack.stack(
              [math_ops.cast(counter, dtypes.float32)]),
          dense_shape=[1])
      pre_batched = inp.batch([counter, sparse_counter, "string"], batch_size=2)
      batched = inp.batch(pre_batched, enqueue_many=True, batch_size=batch_size)
      self.evaluate(variables.global_variables_initializer())
      variables.local_variables_initializer().run()
      threads = queue_runner_impl.start_queue_runners()

      for i in range(num_batches):
        results = self.evaluate(batched)
        self.assertAllEqual(results[0],
                            np.arange(i * batch_size, (i + 1) * batch_size))
        self.assertAllEqual(
            results[1].indices,
            np.vstack((np.arange(batch_size), np.zeros(batch_size))).T)
        self.assertAllEqual(results[1].values,
                            np.arange(i * batch_size, (i + 1) * batch_size))
        self.assertAllEqual(results[1].dense_shape, [batch_size, 1])
        self.assertAllEqual(results[2], [b"string"] * batch_size)

      # Reached the limit.
      with self.assertRaises(errors_impl.OutOfRangeError):
        self.evaluate(batched)
      for thread in threads:
        thread.join()

  def testManyThreads(self):
    # [FUNCTION DOC]
    # Scenario: batching with four enqueue threads and batch size ten over thirty examples.
    # Because thread interleaving is not deterministic, asserts only per-batch structure (length,
    # sparse values equal dense values, indices, dense shape, strings) and that the union of all
    # counters equals the full range.  Batches are logged.  Then asserts the out-of-range error.
    with ops.Graph().as_default(), self.cached_session():
      batch_size = 10
      num_batches = 3
      zero64 = constant_op.constant(0, dtype=dtypes.int64)

      examples = variables.Variable(zero64)
      counter = examples.count_up_to(num_batches * batch_size)
      sparse_counter = sparse_tensor.SparseTensor(
          indices=array_ops.reshape(zero64, [1, 1]),
          values=array_ops_stack.stack(
              [math_ops.cast(counter, dtypes.float32)]),
          dense_shape=[1])
      batched = inp.batch(
          [counter, sparse_counter, "string"],
          batch_size=batch_size,
          num_threads=4)
      self.evaluate(variables.global_variables_initializer())
      variables.local_variables_initializer().run()
      threads = queue_runner_impl.start_queue_runners()

      all_counts = []
      for i in range(num_batches):
        results = self.evaluate(batched)
        tf_logging.info("Batch %d: %s", i, results[0])
        self.assertEqual(len(results[0]), batch_size)
        self.assertAllEqual(results[0], results[1].values)
        self.assertAllEqual(
            results[1].indices,
            np.vstack((np.arange(batch_size), np.zeros(batch_size))).T)
        self.assertAllEqual(results[1].dense_shape, [batch_size, 1])
        all_counts.extend(results[0])
        self.assertAllEqual(results[2], [b"string"] * batch_size)
      self.assertItemsEqual(all_counts, range(num_batches * batch_size))

      # Reached the limit.
      with self.assertRaises(errors_impl.OutOfRangeError):
        self.evaluate(batched)
      for thread in threads:
        thread.join()

  def testOneThreadSmallerBatch(self):
    # [FUNCTION DOC]
    # Scenario: thirty five examples are batched with batch size ten and the option to allow a
    # smaller final batch.  Asserts the three full batches as in the basic test, then asserts that
    # the final batch holds the five remaining elements with matching sparse indices, dense
    # shape and strings, and that a further fetch raises an out-of-range error.
    with ops.Graph().as_default(), self.cached_session():
      batch_size = 10
      num_batches = 3
      extra_elements = 5
      zero64 = constant_op.constant(0, dtype=dtypes.int64)
      examples = variables.Variable(zero64)
      counter = examples.count_up_to(num_batches * batch_size + extra_elements)
      sparse_counter = sparse_tensor.SparseTensor(
          indices=array_ops.reshape(
              array_ops_stack.stack([zero64, zero64 + 1]), [2, 1]),
          values=math_ops.cast(
              array_ops_stack.stack([counter, -counter]), dtypes.float32),
          dense_shape=[2])
      batched = inp.batch(
          [counter, sparse_counter, "string"],
          batch_size=batch_size,
          allow_smaller_final_batch=True)
      self.evaluate(variables.global_variables_initializer())
      variables.local_variables_initializer().run()
      threads = queue_runner_impl.start_queue_runners()

      for i in range(num_batches):
        results = self.evaluate(batched)
        self.assertAllEqual(results[0],
                            np.arange(i * batch_size, (i + 1) * batch_size))
        self.assertAllEqual(
            results[1].indices,
            np.vstack((
                np.arange(2 * batch_size) // 2,  # 0, 0, 1, 1, ...
                [0, 1] * batch_size)).T)
        #  [x, -x, x+1, -(x+1), ...]
        expected = np.arange(2 * i * batch_size, 2 * (i + 1) * batch_size) // 2
        expected *= ([1, -1] * batch_size)  # mult by [1, -1, 1, -1, ...]
        self.assertAllEqual(results[1].values, expected)
        self.assertAllEqual(results[1].dense_shape, [batch_size, 2])
        self.assertAllEqual(results[2], [b"string"] * batch_size)

      # Reached the final batch with extra_elements.
      results = self.evaluate(batched)
      self.assertAllEqual(results[0],
                          np.arange(num_batches * batch_size,
                                    num_batches * batch_size + extra_elements))
      self.assertAllEqual(
          results[1].indices,
          np.vstack((
              np.arange(2 * extra_elements) // 2,  # 0, 0, 1, 1, ...
              [0, 1] * extra_elements)).T)
      self.assertAllEqual(results[1].dense_shape, [extra_elements, 2])
      self.assertAllEqual(results[2], [b"string"] * extra_elements)

      # Reached the limit.
      with self.assertRaises(errors_impl.OutOfRangeError):
        self.evaluate(batched)
      for thread in threads:
        thread.join()

  def testManyThreadsSmallerBatch(self):
    # [FUNCTION DOC]
    # Scenario: like the many-thread test, but with thirty five examples and a smaller final
    # batch allowed.  Asserts the structure of the three full batches, then the five element
    # final batch, and that all counters seen together equal the full range; finally asserts the
    # out-of-range error.
    with ops.Graph().as_default(), self.cached_session():
      batch_size = 10
      num_batches = 3
      extra_elements = 5
      zero64 = constant_op.constant(0, dtype=dtypes.int64)

      examples = variables.Variable(zero64)
      counter = examples.count_up_to(num_batches * batch_size + extra_elements)
      sparse_counter = sparse_tensor.SparseTensor(
          indices=array_ops.reshape(zero64, [1, 1]),
          values=array_ops_stack.stack(
              [math_ops.cast(counter, dtypes.float32)]),
          dense_shape=[1])
      batched = inp.batch(
          [counter, sparse_counter, "string"],
          batch_size=batch_size,
          num_threads=4,
          allow_smaller_final_batch=True)
      self.evaluate(variables.global_variables_initializer())
      variables.local_variables_initializer().run()
      threads = queue_runner_impl.start_queue_runners()

      all_counts = []
      for i in range(num_batches):
        results = self.evaluate(batched)
        tf_logging.info("Batch %d: %s", i, results[0])
        self.assertEqual(len(results[0]), batch_size)
        self.assertAllEqual(results[0], results[1].values)
        self.assertAllEqual(
            results[1].indices,
            np.vstack((np.arange(batch_size), np.zeros(batch_size))).T)
        self.assertAllEqual(results[1].dense_shape, [batch_size, 1])
        all_counts.extend(results[0])
        self.assertAllEqual(results[2], [b"string"] * batch_size)

      # Reached the final batch with extra_elements.
      results = self.evaluate(batched)
      tf_logging.info("Last Batch: %s", results[0])
      self.assertEqual(len(results[0]), extra_elements)
      self.assertAllEqual(results[0], results[1].values)
      self.assertAllEqual(
          results[1].indices,
          np.vstack((np.arange(extra_elements), np.zeros(extra_elements))).T)
      self.assertAllEqual(results[1].dense_shape, [extra_elements, 1])
      all_counts.extend(results[0])
      self.assertAllEqual(results[2], [b"string"] * extra_elements)
      self.assertItemsEqual(all_counts,
                            range(num_batches * batch_size + extra_elements))

      # Reached the limit.
      with self.assertRaises(errors_impl.OutOfRangeError):
        self.evaluate(batched)
      for thread in threads:
        thread.join()

  def testSharedName(self):
    # [FUNCTION DOC]
    # Scenario: batch is called with a shared name and a name.  Asserts that the shared-name
    # attribute appears on the underlying queue operation found by walking back from the first
    # output tensor.
    with ops.Graph().as_default(), self.cached_session():
      batch_size = 10
      num_batches = 3
      zero64 = constant_op.constant(0, dtype=dtypes.int64)
      examples = variables.Variable(zero64)
      counter = examples.count_up_to(num_batches * batch_size)
      batched = inp.batch(
          [counter, "string"],
          batch_size=batch_size,
          shared_name="SHARED_NAME_XYZ",
          name="Q")

      self.assertProtoEquals(
          "s: 'SHARED_NAME_XYZ'",
          batched[0].op.inputs[0].op.node_def.attr["shared_name"])

  def testCannotInferRankError(self):
    # [FUNCTION DOC]
    # Scenario: batch is called with a placeholder of unknown rank.  Asserts that a value error
    # saying the tensor rank cannot be inferred is raised.
    with ops.Graph().as_default(), self.cached_session():
      x = array_ops.placeholder(dtype=dtypes.int64)
      with self.assertRaisesRegex(ValueError, "Cannot infer Tensor's rank"):
        inp.batch([x], batch_size=2)

  def testBatchedSparseTensorInferredShape(self):
    # [FUNCTION DOC]
    # Scenario: a sparse tensor with a one-element dense shape vector is batched with batch size
    # two.  Asserts the statically inferred shape of the batched dense-shape vector has two
    # elements (one extra batch dimension).
    with ops.Graph().as_default():
      sparse = sparse_tensor.SparseTensor(
          indices=[[0]], values=[1.0], dense_shape=[1])
      self.assertAllEqual((1,), sparse.dense_shape.get_shape().as_list())
      batched = inp.batch([sparse], batch_size=2)
      self.assertAllEqual((2,), batched.dense_shape.get_shape().as_list())

  def testBatchedSparseTensorInferredShapeEnqueueMany(self):
    # [FUNCTION DOC]
    # Scenario: as above but with the enqueue-many option, so the batch dimension already exists.
    # Asserts the batched dense-shape vector keeps one element.
    with ops.Graph().as_default():
      sparse = sparse_tensor.SparseTensor(
          indices=[[0]], values=[1.0], dense_shape=[1])
      self.assertAllEqual((1,), sparse.dense_shape.get_shape().as_list())
      batched = inp.batch([sparse], batch_size=2, enqueue_many=True)
      self.assertAllEqual((1,), batched.dense_shape.get_shape().as_list())

  def testBatchedSparseTensorInferredShapeUnknownRank(self):
    # [FUNCTION DOC]
    # Scenario: a sparse tensor built from placeholders so the number of dimensions is unknown is
    # batched.  Asserts that the batched dense-shape vector also has an unknown number of
    # elements (none).
    with ops.Graph().as_default():
      sparse = sparse_tensor.SparseTensor(
          indices=array_ops.placeholder(dtypes.int64),
          values=array_ops.placeholder(dtypes.float32),
          dense_shape=array_ops.placeholder(dtypes.int64))
      self.assertIs(None, sparse.dense_shape.get_shape().num_elements())
      batched = inp.batch([sparse], batch_size=2)
      self.assertIs(None, batched.dense_shape.get_shape().num_elements())

  def testBatchedSparseTensorInferredShapeUnknownRankEnqueueMany(self):
    # [FUNCTION DOC]
    # Scenario: unknown rank sparse tensor batched with the enqueue-many option.  Asserts the
    # element count of the batched dense-shape vector is still unknown.
    with ops.Graph().as_default():
      sparse = sparse_tensor.SparseTensor(
          indices=array_ops.placeholder(dtypes.int64),
          values=array_ops.placeholder(dtypes.float32),
          dense_shape=array_ops.placeholder(dtypes.int64))
      self.assertIs(None, sparse.dense_shape.get_shape().num_elements())
      batched = inp.batch([sparse], batch_size=2, enqueue_many=True)
      self.assertIs(None, batched.dense_shape.get_shape().num_elements())

  def testSingleElementDict(self):
    # [FUNCTION DOC]
    # Scenario: a dictionary with a single entry holding a two element list is batched with batch
    # size eight.  Asserts that the static shape of the returned entry is eight by two, showing
    # that single-key dictionaries are handled.
    with ops.Graph().as_default():
      x = inp.batch({"c": [12, 12]}, batch_size=8)
      self.assertAllEqual((8, 2), x["c"].get_shape().as_list())

  def _testKeepInputHelper(self, num_threads, enqueue_many,
                           keep_input_vector=False):
    # [FUNCTION DOC]
    # HELPER: Shared body for the conditional (maybe_batch) tests of this class.
    # Parameters: num_threads is the number of enqueue threads; enqueue_many says whether the
    # inputs are pre-batched; keep_input_vector (optional, default false) makes the pre-batch
    # size four so that the keep condition becomes a per-example vector.  A counter up to forty is
    # paired with a sparse tensor and a string; only even counter values are kept.  Asserts that
    # four batches of five contain only even numbers (dense and sparse) and the string, and that
    # a further fetch raises an out-of-range error.  The deprecated all-variables and
    # local-variables initializers are used.  Returns nothing.
    with ops.Graph().as_default(), self.cached_session():
      batch_size = 5
      num_batches = 4
      examples = variables.Variable(0)
      counter = examples.count_up_to(num_batches * batch_size * 2)
      sparse_counter = sparse_tensor.SparseTensor(
          indices=array_ops.zeros(
              [1, 1], dtype=dtypes.int64),
          values=array_ops_stack.stack(
              [math_ops.cast(counter, dtypes.float32)]),
          dense_shape=[1])
      to_batch = [counter, sparse_counter, "string"]
      if enqueue_many:
        to_batch = inp.batch(to_batch, 4 if keep_input_vector else 1)
      keep_input = array_ops.squeeze(
          math_ops.equal(0, math_ops.mod(to_batch[0], 2)))
      batched = inp.maybe_batch(
          to_batch,
          keep_input,
          batch_size,
          num_threads=num_threads,
          enqueue_many=enqueue_many)
      variables.initialize_all_variables().run()
      variables.initialize_local_variables().run()
      threads = queue_runner_impl.start_queue_runners()

      for _ in range(num_batches):
        results = self.evaluate(batched)
        self.assertAllEqual([0] * batch_size, np.mod(results[0], 2))
        self.assertAllEqual([0] * batch_size, np.mod(results[1].values, 2))
        self.assertAllEqual([b"string"] * batch_size, results[2])

      # Reached the limit.
      with self.assertRaises(errors_impl.OutOfRangeError):
        self.evaluate(batched)
      for thread in threads:
        thread.join()

  @test_util.run_v1_only("Input pipelines based on Queues are not supported "
                         "when eager execution is enabled. TF2 uses tf.data "
                         "instead.")
  def testSingleThreadKeepInput(self):
    # [FUNCTION DOC]
    # Runs the conditional batching scenario with one thread and no enqueue-many.  Restricted to
    # graph mode (TF1) because queue pipelines are unsupported in eager mode.
    self._testKeepInputHelper(1, False)

  @test_util.run_v1_only("Input pipelines based on Queues are not supported "
                         "when eager execution is enabled. TF2 uses tf.data "
                         "instead.")
  def testSingleThreadKeepInputEnqueueMany(self):
    # [FUNCTION DOC]
    # Runs the conditional batching scenario with one thread and the enqueue-many option.
    # Restricted to graph mode (TF1).
    self._testKeepInputHelper(1, True)

  @test_util.run_v1_only("Input pipelines based on Queues are not supported "
                         "when eager execution is enabled. TF2 uses tf.data "
                         "instead.")
  def testMultipleThreadKeepInput(self):
    # [FUNCTION DOC]
    # Runs the conditional batching scenario with five threads and no enqueue-many.  Restricted to
    # graph mode (TF1).
    self._testKeepInputHelper(5, False)

  @test_util.run_v1_only("Input pipelines based on Queues are not supported "
                         "when eager execution is enabled. TF2 uses tf.data "
                         "instead.")
  def testMultipleThreadKeepInputEnqueueMany(self):
    # [FUNCTION DOC]
    # Runs the conditional batching scenario with five threads and the enqueue-many option.
    # Restricted to graph mode (TF1).
    self._testKeepInputHelper(5, True)

  def testMaybeEnqueuePerExample(self):
    # [FUNCTION DOC]
    # Runs the conditional batching scenario with one thread, the enqueue-many option and a
    # per-example keep vector.
    self._testKeepInputHelper(1, True, keep_input_vector=True)

  def testMultipleThreadMaybeEnqueuePerExample(self):
    # [FUNCTION DOC]
    # Runs the conditional batching scenario with five threads, the enqueue-many option and a
    # per-example keep vector.
    self._testKeepInputHelper(5, True, keep_input_vector=True)

  def testInvalidKeepInputVector(self):
    # [FUNCTION DOC]
    # Scenario: three invalid uses of maybe_batch are checked, each expected to raise a value
    # error at graph construction: a vector keep condition without enqueue-many ("cannot be a
    # vector"), a keep condition with more than one dimension ("must be 0 or 1 dimensions"),
    # and a keep condition of unknown shape ("must be known at graph construction").
    with ops.Graph().as_default():
      # Can't have vector `keep_input` with `enqueue_many=False`.
      with self.assertRaisesRegex(ValueError,
                                  "`keep_input` cannot be a vector"):
        inp.maybe_batch([array_ops.zeros(5)],
                        keep_input=constant_op.constant([True, False]),
                        batch_size=1,
                        enqueue_many=False)
      # Can't have `keep_input` with more than one dimension.
      with self.assertRaisesRegex(ValueError, "must be 0 or 1 dimensions"):
        inp.maybe_batch([array_ops.zeros(5)],
                        keep_input=constant_op.constant([[True], [False]]),
                        batch_size=1,
                        enqueue_many=True)
      # `keep_input` must have dimensions determined at graph construction.
      with self.assertRaisesRegex(ValueError,
                                  "must be known at graph construction"):
        inp.maybe_batch([array_ops.zeros(5)],
                        keep_input=array_ops.placeholder(dtypes.bool),
                        batch_size=1,
                        enqueue_many=True)

  def testMaybeBatchedSparseTensorInferredShape(self):
    # [FUNCTION DOC]
    # Scenario: maybe_batch with an always-true keep condition on a sparse tensor.  Asserts the
    # batched dense-shape vector gets two elements (batch dimension added).
    with ops.Graph().as_default():
      sparse = sparse_tensor.SparseTensor(
          indices=[[0]], values=[1.0], dense_shape=[1])
      self.assertAllEqual((1,), sparse.dense_shape.get_shape().as_list())
      batched = inp.maybe_batch([sparse], keep_input=True, batch_size=2)
      self.assertAllEqual((2,), batched.dense_shape.get_shape().as_list())

  def testMaybeBatchedSparseTensorInferredShapeEnqueueMany(self):
    # [FUNCTION DOC]
    # Scenario: maybe_batch with enqueue-many and a true keep condition.  Asserts the batched
    # dense-shape vector keeps one element.
    with ops.Graph().as_default():
      sparse = sparse_tensor.SparseTensor(
          indices=[[0]], values=[1.0], dense_shape=[1])
      self.assertAllEqual((1,), sparse.dense_shape.get_shape().as_list())
      batched = inp.maybe_batch([sparse],
                                keep_input=True,
                                batch_size=2,
                                enqueue_many=True)
      self.assertAllEqual((1,), batched.dense_shape.get_shape().as_list())

  def testMaybeBatchedSparseTensorInferredShapeEnqueueManyPerExample(self):
    # [FUNCTION DOC]
    # Scenario: maybe_batch with enqueue-many and a two-element per-example keep list.  Asserts
    # the batched dense-shape vector keeps one element.
    with ops.Graph().as_default():
      sparse = sparse_tensor.SparseTensor(
          indices=[[0], [0]], values=[1.0, 2.0], dense_shape=[2])
      self.assertAllEqual((1,), sparse.dense_shape.get_shape().as_list())
      batched = inp.maybe_batch([sparse],
                                keep_input=[True, False],
                                batch_size=2,
                                enqueue_many=True)
      self.assertAllEqual((1,), batched.dense_shape.get_shape().as_list())

  def testMaybeBatchedSparseTensorInferredShapeUnknownRank(self):
    # [FUNCTION DOC]
    # Scenario: maybe_batch on an unknown-rank sparse tensor.  Asserts the dense-shape vector's
    # element count stays unknown.
    with ops.Graph().as_default():
      sparse = sparse_tensor.SparseTensor(
          indices=array_ops.placeholder(dtypes.int64),
          values=array_ops.placeholder(dtypes.float32),
          dense_shape=array_ops.placeholder(dtypes.int64))
      self.assertIs(None, sparse.dense_shape.get_shape().num_elements())
      batched = inp.maybe_batch([sparse], keep_input=True, batch_size=2)
      self.assertIs(None, batched.dense_shape.get_shape().num_elements())

  def testMaybeBatchedSparseTensorInferredShapeUnknownRankEnqueueMany(self):
    # [FUNCTION DOC]
    # Scenario: maybe_batch with enqueue-many on an unknown-rank sparse tensor.  Asserts the
    # element count of the dense-shape vector stays unknown.
    with ops.Graph().as_default():
      sparse = sparse_tensor.SparseTensor(
          indices=array_ops.placeholder(dtypes.int64),
          values=array_ops.placeholder(dtypes.float32),
          dense_shape=array_ops.placeholder(dtypes.int64))
      self.assertIs(None, sparse.dense_shape.get_shape().num_elements())
      batched = inp.maybe_batch([sparse],
                                keep_input=True,
                                batch_size=2,
                                enqueue_many=True)
      self.assertIs(None, batched.dense_shape.get_shape().num_elements())

  def testMaybeBatchedSparseTensorInferredShapeUnknownRankPerExample(self):
    # [FUNCTION DOC]
    # Scenario: maybe_batch with enqueue-many and a per-example keep list on an unknown-rank
    # sparse tensor.  Asserts the element count of the dense-shape vector stays unknown.
    with ops.Graph().as_default():
      sparse = sparse_tensor.SparseTensor(
          indices=array_ops.placeholder(dtypes.int64),
          values=array_ops.placeholder(dtypes.float32),
          dense_shape=array_ops.placeholder(dtypes.int64))
      self.assertIs(None, sparse.dense_shape.get_shape().num_elements())
      batched = inp.maybe_batch([sparse],
                                keep_input=[True, False],
                                batch_size=2,
                                enqueue_many=True)
      self.assertIs(None, batched.dense_shape.get_shape().num_elements())

  def testMaybeBatchCorrectValues(self):
    # [FUNCTION DOC]
    # Scenario: a two-row sparse tensor is passed to maybe_batch with enqueue-many, a keep vector
    # selecting only the first row, and batch size one.  Runs the queue runners with a
    # coordinator, evaluates the batch, stops and joins.  Asserts that the result contains only
    # the first row (indices, values and a dense shape of one by four).
    with ops.Graph().as_default():
      sparse_t = sparse_tensor.SparseTensor(
          indices=[[0, 1], [0, 2], [1, 0], [1, 3]],
          dense_shape=[2, 4],
          values=[5, 4, 7, 2])
      keep = constant_op.constant([True, False])
      batched = inp.maybe_batch([sparse_t],
                                keep_input=keep,
                                batch_size=1,
                                enqueue_many=True)

      with self.cached_session():
        coord = coordinator.Coordinator()
        threads = queue_runner_impl.start_queue_runners(coord=coord)

        batched_np = self.evaluate(batched)

        coord.request_stop()
        for thread in threads:
          thread.join()

      self.assertAllEqual([[0, 1], [0, 2]], batched_np.indices)
      self.assertAllEqual([5, 4], batched_np.values)
      self.assertAllEqual([1, 4], batched_np.dense_shape)


class BatchJoinTest(test_lib.TestCase):
  # [CLASS DOC]
  # Test case for batch_join and maybe_batch_join, which batch examples coming from several
  # independent input lists (one enqueue thread per list).

  def _testTwoThreadsHelper(self, use_dict):
    # [FUNCTION DOC]
    # HELPER: Shared body for the two-producer join tests (list and dictionary input).
    # KNOWN ISSUE: the assertion that both producers are mixed in at least one batch is commented
    # out because thread scheduling is not fair, so mixing is only logged and never verified.
    # Parameter use_dict selects dictionary or list form.  Producer "a" yields seventy counter
    # values with sparse and string parts; producer "b" yields the value 99 ninety times, limited
    # by epochs.  They are joined with batch size five.  Asserts static shapes, then for each batch
    # the lengths, sparse consistency, and tallies which entries came from which producer.
    # Asserts the order of producer "a" results is preserved, all ninety "b" values are seen,
    # and an out-of-range error follows.  Returns nothing.
    with ops.Graph().as_default(), self.cached_session():
      # Two threads, the first generates (0..69, "a").
      num_a = 70
      zero64 = constant_op.constant(0, dtype=dtypes.int64)
      examples = variables.Variable(zero64)
      counter = examples.count_up_to(num_a)
      sparse_counter = sparse_tensor.SparseTensor(
          indices=array_ops.reshape(zero64, [1, 1]),
          values=array_ops_stack.stack(
              [math_ops.cast(counter, dtypes.float32)]),
          dense_shape=[1])

      # The second generates (99, "b") 90 times and then stops.
      num_b = 90
      ninety_nine = inp.limit_epochs(
          constant_op.constant(
              99, dtype=dtypes.int64), num_b)
      sparse_ninety_nine = sparse_tensor.SparseTensor(
          indices=array_ops.reshape(zero64, [1, 1]),
          values=array_ops_stack.stack(
              [math_ops.cast(ninety_nine, dtypes.float32)]),
          dense_shape=[1])

      # These get joined together and grouped into batches of 5.
      batch_size = 5
      if use_dict:
        batched = inp.batch_join(
            [{
                "c": counter,
                "s": sparse_counter,
                "S": "a"
            }, {
                "c": ninety_nine,
                "s": sparse_ninety_nine,
                "S": "b"
            }],
            batch_size=batch_size)
        batched_fetch = [batched["c"], batched["s"], batched["S"]]
      else:
        batched = inp.batch_join(
            [[counter, sparse_counter, "a"],
             [ninety_nine, sparse_ninety_nine, "b"]],
            batch_size=batch_size)
        batched_fetch = batched

      # Shapes.
      self.assertEqual(3, len(batched_fetch))
      self.assertAllEqual((batch_size,), batched_fetch[0].get_shape().as_list())
      self.assertAllEqual((None, 2),
                          batched_fetch[1].indices.get_shape().as_list())
      self.assertAllEqual((None,),
                          batched_fetch[1].values.get_shape().as_list())
      self.assertAllEqual((2,),
                          batched_fetch[1].dense_shape.get_shape().as_list())
      self.assertAllEqual((batch_size,), batched_fetch[2].get_shape().as_list())

      self.evaluate(variables.global_variables_initializer())
      variables.local_variables_initializer().run()
      threads = queue_runner_impl.start_queue_runners()

      # Should see the "a" and "b" threads mixed together.
      all_a = []
      seen_b = 0
      saw_both = 0
      num_batches = (num_a + num_b) // batch_size
      for i in range(num_batches):
        results = self.evaluate(batched_fetch)
        self.assertEqual(3, len(results))
        self.assertEqual(batch_size, len(results[0]))
        self.assertEqual(batch_size, len(results[2]))
        self.assertAllEqual(results[0], results[1].values)
        self.assertAllEqual(
            results[1].indices,
            np.vstack((np.arange(batch_size), np.zeros(batch_size))).T)
        self.assertAllEqual(results[1].dense_shape, [batch_size, 1])
        which_a = [i for i, s in enumerate(results[2]) if s == b"a"]
        which_b = [i for i, s in enumerate(results[2]) if s == b"b"]
        self.assertEqual(len(which_a) + len(which_b), batch_size)
        if which_a and which_b:
          saw_both += 1
        all_a.extend(results[0][i] for i in which_a)
        seen_b += len(which_b)
        self.assertAllEqual([99] * len(which_b),
                            [results[0][i] for i in which_b])

      # We'd like to see some minimum level of mixing of the results of both
      # threads, but we can't rely on fair thread scheduling, so we just log.
      # self.assertGreater(saw_both, 1)
      tf_logging.info("testTwoThreads%s saw both count: %s",
                      "Dict" if use_dict else "", saw_both)

      # Verify the order of results from "a" were preserved.
      self.assertAllEqual(all_a, np.arange(num_a))
      self.assertEqual(seen_b, num_b)

      # Reached the limit.
      with self.assertRaises(errors_impl.OutOfRangeError):
        self.evaluate(batched_fetch)
      for thread in threads:
        thread.join()

  def testTwoThreads(self):
    # [FUNCTION DOC]
    # Runs the two-producer join scenario with list input.
    self._testTwoThreadsHelper(use_dict=False)

  def testTwoThreadsDict(self):
    # [FUNCTION DOC]
    # Runs the two-producer join scenario with dictionary input.
    self._testTwoThreadsHelper(use_dict=True)

  def testMismatchedDictKeys(self):
    # [FUNCTION DOC]
    # Scenario: batch_join is given two dictionaries with different keys.  Asserts that a value
    # error saying that they must have the same keys is raised at construction time.
    with ops.Graph().as_default(), self.assertRaisesRegex(
        ValueError, "must have the same keys"):
      inp.batch_join(
          [{
              "c": 12,
              "s": 123,
              "S": "a"
          }, {
              "cool": -12,
              "s": 99,
              "S": "b"
          }],
          batch_size=8)

  def testTwoThreadsDynamicPad(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the mixing assertion for the two producers is commented out (unfair thread
    # scheduling), so the mixing is only logged and never verified.
    # Scenario: two producers with dynamic padding: "a" yields a counter with a string repeated
    # counter plus one times, "b" yields 99 with a string repeated 99 times.  Joined with batch
    # size five.  Asserts static shapes (batch by unknown), then per batch lengths and the
    # producer tallies, that "b" strings are 99 long, that the lengths of the "a" strings match
    # their counter plus one in order, that the "a" counters are in order, and the out-of-range
    # error afterwards.  Mixing is only logged because the assertion is commented out.
    with ops.Graph().as_default(), self.cached_session():
      # Two threads, the first generates (0..69, ["a"] * 1..70).
      num_a = 70
      zero64 = constant_op.constant(0, dtype=dtypes.int64)
      examples = variables.Variable(zero64)
      counter = examples.count_up_to(num_a)

      # The second generates (99, ["b"] * 99) 90 times and then stops.
      num_b = 90
      ninety_nine = inp.limit_epochs(
          constant_op.constant(
              99, dtype=dtypes.int64), num_b)

      # These get joined together and grouped into batches of 5.
      batch_size = 5
      a = array_ops.tile(
          ["a"],
          math_ops.cast(array_ops_stack.stack([counter + 1]), dtypes.int32))
      b = array_ops.tile(
          ["b"],
          math_ops.cast(array_ops_stack.stack([ninety_nine]), dtypes.int32))
      batched = inp.batch_join(
          [[counter, a], [ninety_nine, b]],
          batch_size=batch_size,
          dynamic_pad=True)

      # Shapes.
      self.assertEqual(2, len(batched))
      self.assertAllEqual((batch_size,), batched[0].get_shape().as_list())
      self.assertAllEqual((batch_size, None), batched[1].get_shape().as_list())

      self.evaluate(variables.global_variables_initializer())
      variables.local_variables_initializer().run()
      threads = queue_runner_impl.start_queue_runners()

      # Should see the "a" and "b" threads mixed together.
      all_a = []
      count_string_a = []
      seen_b = 0
      saw_both = 0
      num_batches = (num_a + num_b) // batch_size
      for i in range(num_batches):
        results = self.evaluate(batched)
        self.assertEqual(2, len(results))
        self.assertEqual(len(results[0]), batch_size)
        self.assertEqual(len(results[1]), batch_size)
        for s in results[1]:
          if s[0] == b"b":
            self.assertAllEqual(s, [b"b"] * 99)
          else:
            count_string_a.append(sum(x == b"a" for x in s))
        which_a = [i for i, s in enumerate(results[1]) if s[0] == b"a"]
        which_b = [i for i, s in enumerate(results[1]) if s[0] == b"b"]
        self.assertEqual(len(which_a) + len(which_b), batch_size)
        if which_a and which_b:
          saw_both += 1
        all_a.extend(results[0][i] for i in which_a)
        seen_b += len(which_b)
        self.assertAllEqual([99] * len(which_b),
                            [results[0][i] for i in which_b])

      # We'd like to see some minimum level of mixing of the results of both
      # threads, but we can't rely on fair thread scheduling, so we just log.
      # self.assertGreater(saw_both, 1)
      tf_logging.info("testTwoThreadsDynamicPad saw both count: %s", saw_both)

      # Verify the order of results from "a" were preserved.
      self.assertAllEqual(  # tiled "a" with counter + 1
          count_string_a, np.arange(num_a) + 1)
      self.assertAllEqual(all_a, np.arange(num_a))
      self.assertEqual(seen_b, num_b)

      # Reached the limit.
      with self.assertRaises(errors_impl.OutOfRangeError):
        self.evaluate(batched)
      for thread in threads:
        thread.join()

  def testTwoThreadsSmallerBatch(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the mixing assertion for the two producers is commented out (unfair thread
    # scheduling), so the mixing is only logged and never verified.
    # Scenario: like the two-producer test but with two extra elements per producer and a smaller
    # final batch allowed.  Asserts dynamic batch shape, the structure of the full batches, then
    # a final batch with four elements, preserved "a" order, all "b" elements counted, and the
    # out-of-range error.  Mixing is only logged.
    with ops.Graph().as_default(), self.cached_session():
      extra_elements = 2
      # Two threads, the first generates (0..69, "a").
      num_a = 70 + extra_elements
      zero64 = constant_op.constant(0, dtype=dtypes.int64)
      examples = variables.Variable(zero64)
      counter = examples.count_up_to(num_a)
      sparse_counter = sparse_tensor.SparseTensor(
          indices=array_ops.reshape(zero64, [1, 1]),
          values=array_ops_stack.stack(
              [math_ops.cast(counter, dtypes.float32)]),
          dense_shape=[1])

      # The second generates (99, "b") 90 times and then stops.
      num_b = 90 + extra_elements
      ninety_nine = inp.limit_epochs(
          constant_op.constant(
              99, dtype=dtypes.int64), num_b)
      sparse_ninety_nine = sparse_tensor.SparseTensor(
          indices=array_ops.reshape(zero64, [1, 1]),
          values=array_ops_stack.stack(
              [math_ops.cast(ninety_nine, dtypes.float32)]),
          dense_shape=[1])

      # These get joined together and grouped into batches of 5.
      batch_size = 5
      batched = inp.batch_join(
          [[counter, sparse_counter, "a"],
           [ninety_nine, sparse_ninety_nine, "b"]],
          batch_size=batch_size,
          allow_smaller_final_batch=True)

      # Shapes.
      self.assertEqual(3, len(batched))
      self.assertAllEqual((None,), batched[0].get_shape().as_list())
      self.assertAllEqual((None, 2), batched[1].indices.get_shape().as_list())
      self.assertAllEqual((None,), batched[1].values.get_shape().as_list())
      self.assertAllEqual((2,), batched[1].dense_shape.get_shape().as_list())
      self.assertAllEqual((None,), batched[2].get_shape().as_list())

      self.evaluate(variables.global_variables_initializer())
      variables.local_variables_initializer().run()
      threads = queue_runner_impl.start_queue_runners()

      # Should see the "a" and "b" threads mixed together.
      all_a = []
      seen_b = 0
      saw_both = 0
      num_batches = (num_a + num_b) // batch_size
      for i in range(num_batches):
        results = self.evaluate(batched)
        tf_logging.info("Batch %d: %s", i, results[0])
        self.assertEqual(len(results[0]), batch_size)
        self.assertEqual(len(results[2]), batch_size)
        self.assertAllEqual(results[0], results[1].values)
        self.assertAllEqual(
            results[1].indices,
            np.vstack((np.arange(batch_size), np.zeros(batch_size))).T)
        self.assertAllEqual(results[1].dense_shape, [batch_size, 1])
        which_a = [i for i, s in enumerate(results[2]) if s == b"a"]
        which_b = [i for i, s in enumerate(results[2]) if s == b"b"]
        self.assertEqual(len(which_a) + len(which_b), batch_size)
        if which_a and which_b:
          saw_both += 1
        all_a.extend(results[0][i] for i in which_a)
        seen_b += len(which_b)
        self.assertAllEqual([99] * len(which_b),
                            [results[0][i] for i in which_b])

      # Reached the final batch with 2 * extra_elements.
      results = self.evaluate(batched)
      tf_logging.info("Last Batch: %s", results[0])
      self.assertEqual(len(results[0]), 2 * extra_elements)
      self.assertEqual(len(results[2]), 2 * extra_elements)
      self.assertAllEqual(results[0], results[1].values)
      self.assertAllEqual(results[1].indices,
                          np.vstack((np.arange(2 * extra_elements),
                                     np.zeros(2 * extra_elements))).T)
      self.assertAllEqual(results[1].dense_shape, [2 * extra_elements, 1])
      which_a = [i for i, s in enumerate(results[2]) if s == b"a"]
      which_b = [i for i, s in enumerate(results[2]) if s == b"b"]
      self.assertEqual(len(which_a) + len(which_b), 2 * extra_elements)
      if which_a and which_b:
        saw_both += 1
      all_a.extend(results[0][i] for i in which_a)
      seen_b += len(which_b)

      # We'd like to see some minimum level of mixing of the results of both
      # threads, but we can't rely on fair thread scheduling, so we just log.
      # self.assertGreater(saw_both, 1)
      tf_logging.info("testTwoThreadsSmallerBatch saw both count: %s", saw_both)

      # Verify the order of results from "a" were preserved.
      self.assertAllEqual(all_a, np.arange(num_a))
      self.assertEqual(seen_b, num_b)

      # Reached the limit.
      with self.assertRaises(errors_impl.OutOfRangeError):
        self.evaluate(batched)
      for thread in threads:
        thread.join()

  def testTwoThreadsDynamicPadSmallerBatch(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the mixing assertion for the two producers is commented out (unfair thread
    # scheduling), so the mixing is only logged and never verified.
    # Scenario: dynamic padding combined with a smaller final batch for two producers with two
    # extra elements each.  Asserts unknown static shapes, per-batch lengths, "b" string content,
    # padded "a" string lengths, a final batch of four elements, preserved order of "a" values,
    # complete count of "b" values, and the out-of-range error.  Mixing is only logged.
    with ops.Graph().as_default(), self.cached_session():
      extra_elements = 2
      # Two threads, the first generates (0..69, ["a"] * 1..70).
      num_a = 70 + extra_elements
      zero64 = constant_op.constant(0, dtype=dtypes.int64)
      examples = variables.Variable(zero64)
      counter = examples.count_up_to(num_a)

      # The second generates (99, ["b"] * 99) 90 times and then stops.
      num_b = 90 + extra_elements
      ninety_nine = inp.limit_epochs(
          constant_op.constant(
              99, dtype=dtypes.int64), num_b)

      # These get joined together and grouped into batches of 5.
      batch_size = 5
      a = array_ops.tile(
          ["a"],
          math_ops.cast(array_ops_stack.stack([counter + 1]), dtypes.int32))
      b = array_ops.tile(
          ["b"],
          math_ops.cast(array_ops_stack.stack([ninety_nine]), dtypes.int32))
      batched = inp.batch_join(
          [[counter, a], [ninety_nine, b]],
          batch_size=batch_size,
          dynamic_pad=True,
          allow_smaller_final_batch=True)

      # Shapes.
      self.assertEqual(2, len(batched))
      self.assertAllEqual((None,), batched[0].get_shape().as_list())
      self.assertAllEqual((None, None), batched[1].get_shape().as_list())

      self.evaluate(variables.global_variables_initializer())
      variables.local_variables_initializer().run()
      threads = queue_runner_impl.start_queue_runners()

      # Should see the "a" and "b" threads mixed together.
      all_a = []
      count_string_a = []
      seen_b = 0
      saw_both = 0
      num_batches = (num_a + num_b) // batch_size
      for i in range(num_batches):
        results = self.evaluate(batched)
        tf_logging.info("Batch %d: %s", i, results[0])
        self.assertEqual(len(results[0]), batch_size)
        self.assertEqual(len(results[1]), batch_size)
        for s in results[1]:
          if s[0] == b"b":
            self.assertAllEqual(s, [b"b"] * 99)
          else:
            count_string_a.append(sum(x == b"a" for x in s))
        which_a = [i for i, s in enumerate(results[1]) if s[0] == b"a"]
        which_b = [i for i, s in enumerate(results[1]) if s[0] == b"b"]
        self.assertEqual(len(which_a) + len(which_b), batch_size)
        if which_a and which_b:
          saw_both += 1
        all_a.extend(results[0][i] for i in which_a)
        seen_b += len(which_b)
        self.assertAllEqual([99] * len(which_b),
                            [results[0][i] for i in which_b])

      # Reached the final batch with 2 * extra_elements.
      results = self.evaluate(batched)
      tf_logging.info("Last Batch: %s", results[0])
      self.assertEqual(len(results[0]), 2 * extra_elements)
      self.assertEqual(len(results[1]), 2 * extra_elements)
      for s in results[1]:
        if s[0] == b"b":
          self.assertAllEqual(s, [b"b"] * 99)
        else:
          count_string_a.append(sum(x == b"a" for x in s))
      which_a = [i for i, s in enumerate(results[1]) if s[0] == b"a"]
      which_b = [i for i, s in enumerate(results[1]) if s[0] == b"b"]
      self.assertEqual(len(which_a) + len(which_b), 2 * extra_elements)
      if which_a and which_b:
        saw_both += 1
      all_a.extend(results[0][i] for i in which_a)
      seen_b += len(which_b)

      # We'd like to see some minimum level of mixing of the results of both
      # threads, but we can't rely on fair thread scheduling, so we just log.
      # self.assertGreater(saw_both, 1)
      tf_logging.info("testTwoThreadsDynamicPadSmallerBatch saw both count: %s",
                      saw_both)

      # Verify the order of results from "a" were preserved.
      self.assertAllEqual(  # tiled "a" with counter + 1
          count_string_a, np.arange(num_a) + 1)
      self.assertAllEqual(all_a, np.arange(num_a))
      self.assertEqual(seen_b, num_b)

      # Reached the limit.
      with self.assertRaises(errors_impl.OutOfRangeError):
        self.evaluate(batched)
      for thread in threads:
        thread.join()

  def testSharedName(self):
    # [FUNCTION DOC]
    # Scenario: batch_join with a single input list, a shared name and a name.  Asserts the static
    # shape of both outputs equals the batch size and that the shared-name attribute is present on
    # the queue operation.
    with ops.Graph().as_default(), self.cached_session():
      batch_size = 10
      num_batches = 3
      zero64 = constant_op.constant(0, dtype=dtypes.int64)
      examples = variables.Variable(zero64)
      counter = examples.count_up_to(num_batches * batch_size)
      batched = inp.batch_join(
          [[counter, "string"]],
          batch_size=batch_size,
          shared_name="SHARED_NAME_XYZ",
          name="Q")

      # Shapes.
      self.assertEqual(2, len(batched))
      self.assertAllEqual((batch_size,), batched[0].get_shape().as_list())
      self.assertAllEqual((batch_size,), batched[1].get_shape().as_list())

      self.assertProtoEquals(
          "s: 'SHARED_NAME_XYZ'",
          batched[0].op.inputs[0].op.node_def.attr["shared_name"])

  def testCannotInferRankError(self):
    # [FUNCTION DOC]
    # Scenario: batch_join with an unknown-rank placeholder.  Asserts a value error stating that
    # the tensor rank cannot be inferred.
    with ops.Graph().as_default(), self.cached_session():
      x = array_ops.placeholder(dtype=dtypes.int64)
      with self.assertRaisesRegex(ValueError, "Cannot infer Tensor's rank"):
        inp.batch_join([[x]], batch_size=2)

  def testSingleElementDict(self):
    # [FUNCTION DOC]
    # Scenario: batch_join with one dictionary holding a two element list and batch size eight.
    # Asserts the static shape of the entry is eight by two.
    with ops.Graph().as_default():
      x = inp.batch_join([{"c": [12, 12]}], batch_size=8)
      self.assertAllEqual((8, 2), x["c"].get_shape().as_list())

  def _testKeepInputHelper(self, num_threads, enqueue_many,
                           keep_input_vector=False):
    # [FUNCTION DOC]
    # HELPER: Shared body for the conditional join tests of this class.
    # Parameters: num_threads is how many copies of the same input list are joined; enqueue_many
    # says whether inputs are pre-batched; keep_input_vector (optional, default false) selects a
    # per-example keep vector.  Only even counter values are kept.  Asserts that four batches of
    # five contain only even values (dense and sparse) and the string, then an out-of-range
    # error.  Uses the deprecated variable initializers.  Returns nothing.
    with ops.Graph().as_default(), self.cached_session():
      batch_size = 5
      num_batches = 4
      examples = variables.Variable(0)
      counter = examples.count_up_to(num_batches * batch_size * 2)
      sparse_counter = sparse_tensor.SparseTensor(
          indices=array_ops.zeros(
              [1, 1], dtype=dtypes.int64),
          values=array_ops_stack.stack(
              [math_ops.cast(counter, dtypes.float32)]),
          dense_shape=[1])
      to_batch = [counter, sparse_counter, "string"]
      if enqueue_many:
        to_batch = inp.batch(to_batch, 4 if keep_input_vector else 1)
      keep_input = array_ops.squeeze(
          math_ops.equal(0, math_ops.mod(to_batch[0], 2)))
      batched = inp.maybe_batch_join(
          [to_batch] * num_threads,
          keep_input,
          batch_size,
          enqueue_many=enqueue_many)
      variables.initialize_all_variables().run()
      variables.initialize_local_variables().run()
      threads = queue_runner_impl.start_queue_runners()

      for _ in range(num_batches):
        results = self.evaluate(batched)
        self.assertAllEqual(
            [0] * batch_size,
            np.mod(results[0], 2),)
        self.assertAllEqual(
            [0] * batch_size,
            np.mod(results[1].values, 2),)
        self.assertAllEqual([b"string"] * batch_size, results[2])

      # Reached the limit.
      with self.assertRaises(errors_impl.OutOfRangeError):
        self.evaluate(batched)
      for thread in threads:
        thread.join()

  @test_util.run_v1_only("Input pipelines based on Queues are not supported "
                         "when eager execution is enabled. TF2 uses tf.data "
                         "instead.")
  def testSingleThreadKeepInput(self):
    # [FUNCTION DOC]
    # Conditional join with one thread and no enqueue-many; graph mode (TF1) only.
    self._testKeepInputHelper(1, False)

  @test_util.run_v1_only("Input pipelines based on Queues are not supported "
                         "when eager execution is enabled. TF2 uses tf.data "
                         "instead.")
  def testSingleThreadKeepInputEnqueueMany(self):
    # [FUNCTION DOC]
    # Conditional join with one thread and enqueue-many; graph mode (TF1) only.
    self._testKeepInputHelper(1, True)

  @test_util.run_v1_only("Input pipelines based on Queues are not supported "
                         "when eager execution is enabled. TF2 uses tf.data "
                         "instead.")
  def testMultipleThreadKeepInput(self):
    # [FUNCTION DOC]
    # Conditional join with five threads and no enqueue-many; graph mode (TF1) only.
    self._testKeepInputHelper(5, False)

  @test_util.run_v1_only("Input pipelines based on Queues are not supported "
                         "when eager execution is enabled. TF2 uses tf.data "
                         "instead.")
  def testMultipleThreadKeepInputEnqueueMany(self):
    # [FUNCTION DOC]
    # Conditional join with five threads and enqueue-many; graph mode (TF1) only.
    self._testKeepInputHelper(5, True)

  def testSingleThreadKeepInputPerExample(self):
    # [FUNCTION DOC]
    # Conditional join with one thread, enqueue-many and a per-example keep vector.
    self._testKeepInputHelper(1, True, keep_input_vector=True)

  def testMultipleThreadKeepInputPerExample(self):
    # [FUNCTION DOC]
    # Conditional join with five threads, enqueue-many and a per-example keep vector.
    self._testKeepInputHelper(5, True, keep_input_vector=True)

  def testInvalidKeepInputVector(self):
    # [FUNCTION DOC]
    # Scenario: three invalid uses of maybe_batch_join, each expected to raise a value error:
    # vector keep condition without enqueue-many, keep condition with more than one dimension,
    # and keep condition with unknown shape at graph construction.
    with ops.Graph().as_default():
      # Can't have vector `keep_input` with `enqueue_many=False`.
      with self.assertRaisesRegex(ValueError,
                                  "`keep_input` cannot be a vector"):
        inp.maybe_batch_join([[array_ops.zeros(5)]],
                             keep_input=constant_op.constant([True, False]),
                             batch_size=1,
                             enqueue_many=False)
      # Can't have `keep_input` with more than one dimension.
      with self.assertRaisesRegex(ValueError, "must be 0 or 1 dimensions"):
        inp.maybe_batch_join([[array_ops.zeros(5)]],
                             keep_input=constant_op.constant([[True], [False]]),
                             batch_size=1,
                             enqueue_many=True)
      # `keep_input` must have dimensions determined at graph construction.
      with self.assertRaisesRegex(ValueError,
                                  "must be known at graph construction"):
        inp.maybe_batch_join([[array_ops.zeros(5)]],
                             keep_input=array_ops.placeholder(dtypes.bool),
                             batch_size=1,
                             enqueue_many=True)

  def testMaybeBatchedSparseTensorInferredShape(self):
    # [FUNCTION DOC]
    # Scenario: maybe_batch_join with a true keep condition on one sparse input.  Asserts the
    # batched dense-shape vector has two elements.
    with ops.Graph().as_default():
      sparse = sparse_tensor.SparseTensor(
          indices=[[0]], values=[1.0], dense_shape=[1])
      self.assertAllEqual((1,), sparse.dense_shape.get_shape().as_list())
      batched = inp.maybe_batch_join([[sparse]], keep_input=True, batch_size=2)
      self.assertAllEqual((2,), batched.dense_shape.get_shape().as_list())

  def testMaybeBatchedSparseTensorInferredShapeEnqueueMany(self):
    # [FUNCTION DOC]
    # Scenario: maybe_batch_join with enqueue-many.  Asserts the dense-shape vector keeps one
    # element.
    with ops.Graph().as_default():
      sparse = sparse_tensor.SparseTensor(
          indices=[[0]], values=[1.0], dense_shape=[1])
      self.assertAllEqual((1,), sparse.dense_shape.get_shape().as_list())
      batched = inp.maybe_batch_join([[sparse]],
                                     keep_input=True,
                                     batch_size=2,
                                     enqueue_many=True)
      self.assertAllEqual((1,), batched.dense_shape.get_shape().as_list())

  def testMaybeBatchedSparseTensorInferredShapeEnqueueManyPerExample(self):
    # [FUNCTION DOC]
    # Scenario: maybe_batch_join with enqueue-many and a per-example keep list.  Asserts the
    # dense-shape vector keeps one element.
    with ops.Graph().as_default():
      sparse = sparse_tensor.SparseTensor(
          indices=[[0], [0]], values=[1.0, 2.0], dense_shape=[2])
      self.assertAllEqual((1,), sparse.dense_shape.get_shape().as_list())
      batched = inp.maybe_batch_join([[sparse]],
                                     keep_input=[True, False],
                                     batch_size=2,
                                     enqueue_many=True)
      self.assertAllEqual((1,), batched.dense_shape.get_shape().as_list())

  def testMaybeBatchedSparseTensorInferredShapeUnknownRank(self):
    # [FUNCTION DOC]
    # Scenario: maybe_batch_join on an unknown-rank sparse tensor.  Asserts the dense-shape
    # vector's element count is unknown.
    with ops.Graph().as_default():
      sparse = sparse_tensor.SparseTensor(
          indices=array_ops.placeholder(dtypes.int64),
          values=array_ops.placeholder(dtypes.float32),
          dense_shape=array_ops.placeholder(dtypes.int64))
      self.assertIs(None, sparse.dense_shape.get_shape().num_elements())
      batched = inp.maybe_batch_join([[sparse]], keep_input=True, batch_size=2)
      self.assertIs(None, batched.dense_shape.get_shape().num_elements())

  def testMaybeBatchedSparseTensorInferredShapeUnknownRankEnqueueMany(self):
    # [FUNCTION DOC]
    # Scenario: maybe_batch_join with enqueue-many on an unknown-rank sparse tensor.  Asserts the
    # element count stays unknown.
    with ops.Graph().as_default():
      sparse = sparse_tensor.SparseTensor(
          indices=array_ops.placeholder(dtypes.int64),
          values=array_ops.placeholder(dtypes.float32),
          dense_shape=array_ops.placeholder(dtypes.int64))
      self.assertIs(None, sparse.dense_shape.get_shape().num_elements())
      batched = inp.maybe_batch_join([[sparse]],
                                     keep_input=True,
                                     batch_size=2,
                                     enqueue_many=True)
      self.assertIs(None, batched.dense_shape.get_shape().num_elements())

  def testMaybeBatchedSparseTensorInferredShapeUnknownRankPerExample(self):
    # [FUNCTION DOC]
    # Scenario: maybe_batch_join with enqueue-many and a per-example keep list on an unknown-rank
    # sparse tensor.  Asserts the element count stays unknown.
    with ops.Graph().as_default():
      sparse = sparse_tensor.SparseTensor(
          indices=array_ops.placeholder(dtypes.int64),
          values=array_ops.placeholder(dtypes.float32),
          dense_shape=array_ops.placeholder(dtypes.int64))
      self.assertIs(None, sparse.dense_shape.get_shape().num_elements())
      batched = inp.maybe_batch_join([[sparse]],
                                     keep_input=[True, False],
                                     batch_size=2,
                                     enqueue_many=True)
      self.assertIs(None, batched.dense_shape.get_shape().num_elements())

  def testMaybeBatchCorrectValues(self):
    # [FUNCTION DOC]
    # Scenario: a two-row sparse tensor goes through maybe_batch_join with enqueue-many and a keep
    # vector selecting only the first row, batch size one.  Runs queue runners with a coordinator,
    # evaluates, stops and joins.  Asserts the result holds only the first row.
    with ops.Graph().as_default():
      sparse = sparse_tensor.SparseTensor(
          indices=[[0, 1], [0, 2], [1, 0], [1, 3]],
          dense_shape=[2, 4],
          values=[5, 4, 7, 2])
      keep = constant_op.constant([True, False])
      batched = inp.maybe_batch_join([[sparse]],
                                     keep_input=keep,
                                     batch_size=1,
                                     enqueue_many=True)

      with self.cached_session():
        coord = coordinator.Coordinator()
        threads = queue_runner_impl.start_queue_runners(coord=coord)

        batched_np = self.evaluate(batched)

        coord.request_stop()
        for thread in threads:
          thread.join()

      self.assertAllEqual([[0, 1], [0, 2]], batched_np.indices)
      self.assertAllEqual([5, 4], batched_np.values)
      self.assertAllEqual([1, 4], batched_np.dense_shape)


class ShuffleBatchTest(test_lib.TestCase):
  # [CLASS DOC]
  # Test case for shuffle_batch and maybe_shuffle_batch, which batch examples through a
  # shuffling queue with a capacity and a minimum number of elements kept after dequeue.

  def _testOneThreadHelper(self, use_dict):
    # [FUNCTION DOC]
    # HELPER: Shared body for single-thread shuffle batching tests (list and dictionary input).
    # Parameter use_dict selects the input form.  Thirty examples (counter, sparse tensor,
    # string) are batched with batch size ten, capacity 32, minimum after dequeue 16 and a fixed
    # seed.  Asserts per-batch structure, that the overall sequence is not in uniformly
    # increasing order (it was shuffled), that all numbers are present, and that an out-of-range
    # error follows.  Returns nothing.
    with ops.Graph().as_default(), self.cached_session():
      batch_size = 10
      num_batches = 3
      zero64 = constant_op.constant(0, dtype=dtypes.int64)
      examples = variables.Variable(zero64)
      counter = examples.count_up_to(num_batches * batch_size)
      sparse_counter = sparse_tensor.SparseTensor(
          indices=array_ops.reshape(zero64, [1, 1]),
          values=array_ops_stack.stack(
              [math_ops.cast(counter, dtypes.float32)]),
          dense_shape=[1])
      if use_dict:
        batched = inp.shuffle_batch(
            {
                "c": counter,
                "s": sparse_counter,
                "S": "string"
            },
            batch_size=batch_size,
            capacity=32,
            min_after_dequeue=16,
            seed=141421)
        batched_fetch = [batched["c"], batched["s"], batched["S"]]
      else:
        batched = inp.shuffle_batch(
            [counter, sparse_counter, "string"],
            batch_size=batch_size,
            capacity=32,
            min_after_dequeue=16,
            seed=141421)
        batched_fetch = batched
      self.evaluate(variables.global_variables_initializer())
      variables.local_variables_initializer().run()
      threads = queue_runner_impl.start_queue_runners()

      all_counts = []
      for i in range(num_batches):
        results = self.evaluate(batched_fetch)
        self.assertEqual(len(results[0]), batch_size)
        all_counts.extend(results[0])
        self.assertAllEqual(
            results[1].indices,
            np.vstack((np.arange(batch_size), np.zeros(batch_size))).T)
        self.assertAllEqual(results[0], results[1].values)
        self.assertAllEqual(results[1].dense_shape, [batch_size, 1])
        self.assertAllEqual(results[2], [b"string"] * batch_size)
      # Results scrambled, but include all the expected numbers.
      deltas = [
          all_counts[i + 1] - all_counts[i] for i in range(len(all_counts) - 1)
      ]
      self.assertFalse(all(d == deltas[0] for d in deltas))
      self.assertItemsEqual(all_counts, range(num_batches * batch_size))

      # Reached the limit.
      with self.assertRaises(errors_impl.OutOfRangeError):
        self.evaluate(batched_fetch)
      for thread in threads:
        thread.join()

  def testOneThread(self):
    # [FUNCTION DOC]
    # Runs the single-thread shuffle scenario with list input.
    self._testOneThreadHelper(use_dict=False)

  def testOneThreadDict(self):
    # [FUNCTION DOC]
    # Runs the single-thread shuffle scenario with dictionary input.
    self._testOneThreadHelper(use_dict=True)

  def testOneThreadSmallerBatch(self):
    # [FUNCTION DOC]
    # Scenario: thirty five examples, batch size ten, smaller final batch allowed.  Asserts the
    # three full shuffled batches, then a final batch of five elements (dense shape and strings),
    # that the combined output is scrambled yet complete, and that a further fetch raises an
    # out-of-range error.
    with ops.Graph().as_default(), self.cached_session():
      batch_size = 10
      num_batches = 3
      extra_elements = 5
      zero64 = constant_op.constant(0, dtype=dtypes.int64)
      examples = variables.Variable(zero64)
      total_elements = num_batches * batch_size + extra_elements
      counter = examples.count_up_to(total_elements)
      sparse_counter = sparse_tensor.SparseTensor(
          indices=array_ops.reshape(zero64, [1, 1]),
          values=array_ops_stack.stack(
              [math_ops.cast(counter, dtypes.float32)]),
          dense_shape=[1])
      batched = inp.shuffle_batch(
          [counter, sparse_counter, "string"],
          batch_size=batch_size,
          capacity=32,
          min_after_dequeue=16,
          seed=141421,
          allow_smaller_final_batch=True)
      batched_fetch = batched
      self.evaluate(variables.global_variables_initializer())
      variables.local_variables_initializer().run()
      threads = queue_runner_impl.start_queue_runners()

      all_counts = []
      for _ in range(num_batches):
        results = self.evaluate(batched_fetch)
        self.assertEqual(len(results[0]), batch_size)
        all_counts.extend(results[0])
        self.assertAllEqual(
            results[1].indices,
            np.vstack((np.arange(batch_size), np.zeros(batch_size))).T)
        self.assertAllEqual(results[0], results[1].values)
        self.assertAllEqual(results[1].dense_shape, [batch_size, 1])
        self.assertAllEqual(results[2], [b"string"] * batch_size)

      # Reached the final batch with extra elements.
      results = self.evaluate(batched)
      self.assertAllEqual(results[1].dense_shape, [extra_elements, 1])
      self.assertAllEqual(results[2], [b"string"] * extra_elements)
      all_counts.extend(results[0])

      # Results scrambled, but include all the expected numbers.
      deltas = [
          all_counts[i + 1] - all_counts[i] for i in range(len(all_counts) - 1)
      ]
      self.assertFalse(all(d == deltas[0] for d in deltas))
      self.assertItemsEqual(all_counts, range(total_elements))

      # Reached the limit.
      with self.assertRaises(errors_impl.OutOfRangeError):
        self.evaluate(batched_fetch)
      for thread in threads:
        thread.join()

  def testManyThreads(self):
    # [FUNCTION DOC]
    # Scenario: shuffle batching with four threads over thirty examples.  Asserts per-batch
    # structure, that the combined output is scrambled but complete, and the out-of-range error.
    # Batches are logged.
    with ops.Graph().as_default(), self.cached_session():
      batch_size = 10
      num_batches = 3
      zero64 = constant_op.constant(0, dtype=dtypes.int64)
      examples = variables.Variable(zero64)
      counter = examples.count_up_to(num_batches * batch_size)
      sparse_counter = sparse_tensor.SparseTensor(
          indices=array_ops.reshape(zero64, [1, 1]),
          values=array_ops_stack.stack(
              [math_ops.cast(counter, dtypes.float32)]),
          dense_shape=[1])
      batched = inp.shuffle_batch(
          [counter, sparse_counter, "string"],
          batch_size=batch_size,
          capacity=32,
          min_after_dequeue=16,
          seed=173205,
          num_threads=4)
      self.evaluate(variables.global_variables_initializer())
      variables.local_variables_initializer().run()
      threads = queue_runner_impl.start_queue_runners()

      all_counts = []
      for i in range(num_batches):
        results = self.evaluate(batched)
        tf_logging.info("Batch %d: %s", i, results[0])
        self.assertEqual(len(results[0]), batch_size)
        all_counts.extend(results[0])
        self.assertAllEqual(
            results[1].indices,
            np.vstack((np.arange(batch_size), np.zeros(batch_size))).T)
        self.assertAllEqual(results[0], results[1].values)
        self.assertAllEqual(results[1].dense_shape, [batch_size, 1])
        self.assertAllEqual(results[2], [b"string"] * batch_size)
      # Results scrambled, but include all the expected numbers.
      deltas = [
          all_counts[i + 1] - all_counts[i] for i in range(len(all_counts) - 1)
      ]
      self.assertFalse(all(d == deltas[0] for d in deltas))
      self.assertItemsEqual(all_counts, range(num_batches * batch_size))

      # Reached the limit.
      with self.assertRaises(errors_impl.OutOfRangeError):
        self.evaluate(batched)
      for thread in threads:
        thread.join()

  def testManyThreadsSmallerBatch(self):
    # [FUNCTION DOC]
    # Scenario: four threads and thirty five examples with a smaller final batch allowed.
    # Asserts the full batches, the five element final batch, scrambled complete output, and the
    # out-of-range error.
    with ops.Graph().as_default(), self.cached_session():
      batch_size = 10
      num_batches = 3
      extra_elements = 5
      zero64 = constant_op.constant(0, dtype=dtypes.int64)
      examples = variables.Variable(zero64)
      total_elements = num_batches * batch_size + extra_elements
      counter = examples.count_up_to(total_elements)
      sparse_counter = sparse_tensor.SparseTensor(
          indices=array_ops.reshape(zero64, [1, 1]),
          values=array_ops_stack.stack(
              [math_ops.cast(counter, dtypes.float32)]),
          dense_shape=[1])
      batched = inp.shuffle_batch(
          [counter, sparse_counter, "string"],
          batch_size=batch_size,
          capacity=32,
          min_after_dequeue=16,
          seed=173205,
          num_threads=4,
          allow_smaller_final_batch=True)
      self.evaluate(variables.global_variables_initializer())
      variables.local_variables_initializer().run()
      threads = queue_runner_impl.start_queue_runners()

      all_counts = []
      for i in range(num_batches):
        results = self.evaluate(batched)
        tf_logging.info("Batch %d: %s", i, results[0])
        self.assertEqual(len(results[0]), batch_size)
        all_counts.extend(results[0])
        self.assertAllEqual(
            results[1].indices,
            np.vstack((np.arange(batch_size), np.zeros(batch_size))).T)
        self.assertAllEqual(results[0], results[1].values)
        self.assertAllEqual(results[1].dense_shape, [batch_size, 1])
        self.assertAllEqual(results[2], [b"string"] * batch_size)

      # Reached the final batch with extra elements.
      results = self.evaluate(batched)
      self.assertAllEqual(results[0].shape, [extra_elements])
      self.assertAllEqual(results[1].dense_shape, [extra_elements, 1])
      self.assertAllEqual(results[2], [b"string"] * extra_elements)
      all_counts.extend(results[0])

      # Results scrambled, but include all the expected numbers.
      deltas = [
          all_counts[i + 1] - all_counts[i] for i in range(len(all_counts) - 1)
      ]
      self.assertFalse(all(d == deltas[0] for d in deltas))
      self.assertItemsEqual(all_counts, range(total_elements))

      # Reached the limit.
      with self.assertRaises(errors_impl.OutOfRangeError):
        self.evaluate(batched)
      for thread in threads:
        thread.join()

  def testSharedName(self):
    # [FUNCTION DOC]
    # Scenario: shuffle_batch with a shared name and a name.  Asserts the shared-name attribute is
    # present on the underlying queue operation.
    with ops.Graph().as_default(), self.cached_session():
      batch_size = 10
      num_batches = 3
      zero64 = constant_op.constant(0, dtype=dtypes.int64)
      examples = variables.Variable(zero64)
      counter = examples.count_up_to(num_batches * batch_size)
      batched = inp.shuffle_batch(
          [counter, "string"],
          batch_size=batch_size,
          capacity=32,
          min_after_dequeue=10,
          shared_name="SHARED_NAME_XYZ",
          name="Q")

      self.assertProtoEquals(
          "s: 'SHARED_NAME_XYZ'",
          batched[0].op.inputs[0].op.node_def.attr["shared_name"])

  def _testKeepInputHelper(self, num_threads, enqueue_many,
                           keep_input_vector=False):
    # [FUNCTION DOC]
    # HELPER: Shared body for the conditional shuffle batching tests.
    # Parameters: num_threads is the thread count; enqueue_many tells whether the inputs are
    # pre-batched; keep_input_vector (optional, default false) selects a per-example keep vector.
    # Only even counter values are kept; capacity 10 and minimum after dequeue 1 are used.
    # Asserts four batches of five even values plus string, then an out-of-range error.  Uses the
    # deprecated variable initializers.  Returns nothing.
    with ops.Graph().as_default(), self.cached_session():
      batch_size = 5
      num_batches = 4
      examples = variables.Variable(0)
      counter = examples.count_up_to(num_batches * batch_size * 2)
      sparse_counter = sparse_tensor.SparseTensor(
          indices=array_ops.zeros(
              [1, 1], dtype=dtypes.int64),
          values=array_ops_stack.stack(
              [math_ops.cast(counter, dtypes.float32)]),
          dense_shape=[1])
      to_batch = [counter, sparse_counter, "string"]
      if enqueue_many:
        to_batch = inp.batch(to_batch, 4 if keep_input_vector else 1)
      keep_input = array_ops.squeeze(
          math_ops.equal(0, math_ops.mod(to_batch[0], 2)))
      batched = inp.maybe_shuffle_batch(
          to_batch,
          batch_size,
          10,
          1,
          keep_input,
          num_threads=num_threads,
          enqueue_many=enqueue_many)
      variables.initialize_all_variables().run()
      variables.initialize_local_variables().run()
      threads = queue_runner_impl.start_queue_runners()

      for _ in range(num_batches):
        results = self.evaluate(batched)
        self.assertAllEqual([0] * batch_size, np.mod(results[0], 2))
        self.assertAllEqual([0] * batch_size, np.mod(results[1].values, 2))
        self.assertAllEqual([b"string"] * batch_size, results[2])

      # Reached the limit.
      with self.assertRaises(errors_impl.OutOfRangeError):
        self.evaluate(batched)
      for thread in threads:
        thread.join()

  @test_util.run_v1_only("Input pipelines based on Queues are not supported "
                         "when eager execution is enabled. TF2 uses tf.data "
                         "instead.")
  def testSingleThreadKeepInput(self):
    # [FUNCTION DOC]
    # Conditional shuffle batching with one thread and no enqueue-many; graph mode (TF1) only.
    self._testKeepInputHelper(1, False)

  @test_util.run_v1_only("Input pipelines based on Queues are not supported "
                         "when eager execution is enabled. TF2 uses tf.data "
                         "instead.")
  def testSingleThreadKeepInputEnqueueMany(self):
    # [FUNCTION DOC]
    # Conditional shuffle batching with one thread and enqueue-many; graph mode (TF1) only.
    self._testKeepInputHelper(1, True)

  @test_util.run_v1_only("Input pipelines based on Queues are not supported "
                         "when eager execution is enabled. TF2 uses tf.data "
                         "instead.")
  def testMultipleThreadKeepInput(self):
    # [FUNCTION DOC]
    # Conditional shuffle batching with five threads and no enqueue-many; graph mode (TF1) only.
    self._testKeepInputHelper(5, False)

  @test_util.run_v1_only("Input pipelines based on Queues are not supported "
                         "when eager execution is enabled. TF2 uses tf.data "
                         "instead.")
  def testMultipleThreadKeepInputEnqueueMany(self):
    # [FUNCTION DOC]
    # Conditional shuffle batching with five threads and enqueue-many; graph mode (TF1) only.
    self._testKeepInputHelper(5, True)

  def testSingleThreadKeepInputPerExample(self):
    # [FUNCTION DOC]
    # Conditional shuffle batching with one thread, enqueue-many and a per-example keep vector.
    self._testKeepInputHelper(1, True, keep_input_vector=True)

  def testMultipleThreadKeepInputPerExample(self):
    # [FUNCTION DOC]
    # Conditional shuffle batching with five threads, enqueue-many and a per-example keep vector.
    self._testKeepInputHelper(5, True, keep_input_vector=True)

  def testInvalidKeepInputVector(self):
    # [FUNCTION DOC]
    # Scenario: three invalid uses of maybe_shuffle_batch, each expected to raise a value error:
    # vector keep condition without enqueue-many, keep condition with more than one dimension,
    # and keep condition of unknown shape at graph construction.
    with ops.Graph().as_default():
      # Can't have vector `keep_input` with `enqueue_many=False`.
      with self.assertRaisesRegex(ValueError,
                                  "`keep_input` cannot be a vector"):
        inp.maybe_shuffle_batch([array_ops.zeros(5)],
                                1,
                                10,
                                1,
                                keep_input=constant_op.constant([True, False]),
                                enqueue_many=False)
      # Can't have `keep_input` with more than one dimension.
      with self.assertRaisesRegex(ValueError, "must be 0 or 1 dimensions"):
        inp.maybe_shuffle_batch([array_ops.zeros(5)],
                                1,
                                10,
                                1,
                                keep_input=constant_op.constant([[True]]),
                                enqueue_many=True)
      # `keep_input` must have dimensions determined at graph construction.
      with self.assertRaisesRegex(ValueError,
                                  "must be known at graph construction"):
        inp.maybe_shuffle_batch([array_ops.zeros(5)],
                                1,
                                10,
                                1,
                                keep_input=array_ops.placeholder(dtypes.bool),
                                enqueue_many=True)

  def testMaybeBatchedSparseTensorInferredShape(self):
    # [FUNCTION DOC]
    # Scenario: maybe_shuffle_batch with a true keep condition on a sparse tensor.  Asserts the
    # batched dense-shape vector has two elements.
    with ops.Graph().as_default():
      sparse = sparse_tensor.SparseTensor(
          indices=[[0]], values=[1.0], dense_shape=[1])
      self.assertAllEqual((1,), sparse.dense_shape.get_shape().as_list())
      batched = inp.maybe_shuffle_batch([sparse], 2, 10, 1, True)
      self.assertAllEqual((2,), batched.dense_shape.get_shape().as_list())

  def testMaybeBatchedSparseTensorInferredShapeEnqueueMany(self):
    # [FUNCTION DOC]
    # Scenario: maybe_shuffle_batch with enqueue-many.  Asserts the dense-shape vector keeps one
    # element.
    with ops.Graph().as_default():
      sparse = sparse_tensor.SparseTensor(
          indices=[[0]], values=[1.0], dense_shape=[1])
      self.assertAllEqual((1,), sparse.dense_shape.get_shape().as_list())
      batched = inp.maybe_shuffle_batch([sparse],
                                        2,
                                        10,
                                        1,
                                        True,
                                        enqueue_many=True)
      self.assertAllEqual((1,), batched.dense_shape.get_shape().as_list())

  def testMaybeBatchedSparseTensorInferredShapeEnqueueManyPerExample(self):
    # [FUNCTION DOC]
    # Scenario: maybe_shuffle_batch with enqueue-many and a per-example keep list.  Asserts the
    # dense-shape vector keeps one element.
    with ops.Graph().as_default():
      sparse = sparse_tensor.SparseTensor(
          indices=[[0], [0]], values=[1.0, 2.0], dense_shape=[2])
      self.assertAllEqual((1,), sparse.dense_shape.get_shape().as_list())
      batched = inp.maybe_shuffle_batch([sparse],
                                        2,
                                        10,
                                        1, [True, False],
                                        enqueue_many=True)
      self.assertAllEqual((1,), batched.dense_shape.get_shape().as_list())

  def testMaybeBatchedSparseTensorInferredShapeUnknownRank(self):
    # [FUNCTION DOC]
    # Scenario: maybe_shuffle_batch on an unknown-rank sparse tensor.  Asserts the dense-shape
    # vector's element count is unknown.
    with ops.Graph().as_default():
      sparse = sparse_tensor.SparseTensor(
          indices=array_ops.placeholder(dtypes.int64),
          values=array_ops.placeholder(dtypes.float32),
          dense_shape=array_ops.placeholder(dtypes.int64))
      self.assertIs(None, sparse.dense_shape.get_shape().num_elements())
      batched = inp.maybe_shuffle_batch([sparse], 2, 10, 1, True)
      self.assertIs(None, batched.dense_shape.get_shape().num_elements())

  def testMaybeBatchedSparseTensorInferredShapeUnknownRankEnqueueMany(self):
    # [FUNCTION DOC]
    # Scenario: maybe_shuffle_batch with enqueue-many on an unknown-rank sparse tensor.  Asserts
    # the element count stays unknown.
    with ops.Graph().as_default():
      sparse = sparse_tensor.SparseTensor(
          indices=array_ops.placeholder(dtypes.int64),
          values=array_ops.placeholder(dtypes.float32),
          dense_shape=array_ops.placeholder(dtypes.int64))
      self.assertIs(None, sparse.dense_shape.get_shape().num_elements())
      batched = inp.maybe_shuffle_batch([sparse],
                                        2,
                                        10,
                                        1,
                                        True,
                                        enqueue_many=True)
      self.assertIs(None, batched.dense_shape.get_shape().num_elements())

  def testMaybeBatchedSparseTensorInferredShapeUnknownRankPerExample(self):
    # [FUNCTION DOC]
    # Scenario: maybe_shuffle_batch with enqueue-many and a per-example keep list on an
    # unknown-rank sparse tensor.  Asserts the element count stays unknown.
    with ops.Graph().as_default():
      sparse = sparse_tensor.SparseTensor(
          indices=array_ops.placeholder(dtypes.int64),
          values=array_ops.placeholder(dtypes.float32),
          dense_shape=array_ops.placeholder(dtypes.int64))
      self.assertIs(None, sparse.dense_shape.get_shape().num_elements())
      batched = inp.maybe_shuffle_batch([sparse],
                                        2,
                                        10,
                                        1, [True, False],
                                        enqueue_many=True)
      self.assertIs(None, batched.dense_shape.get_shape().num_elements())


class ShuffleBatchJoinTest(test_lib.TestCase):
  # [CLASS DOC]
  # Test case for shuffle_batch_join and maybe_shuffle_batch_join, which combine several input
  # lists through a shuffling queue and batch the result.

  def _testTwoThreadsHelper(self, use_dict):
    # [FUNCTION DOC]
    # HELPER: Shared body for the two-producer shuffle join tests (list and dictionary input).
    # Parameter use_dict selects the input form.  Producer "a" yields 25 counter values with
    # sparse and string parts; producer "b" yields 99 thirty five times.  They are joined with
    # batch size five, capacity 32, minimum after dequeue 16 and a fixed seed.  Asserts static
    # shapes, per-batch structure, that both producers are mixed in more than one batch (this
    # assertion is active here), that all "a" values are seen but scrambled, that all "b" values
    # are counted, and that an out-of-range error follows.  Returns nothing.
    with ops.Graph().as_default(), self.cached_session():
      # Two threads, the first generates (0..24, "a").
      num_a = 25
      zero64 = constant_op.constant(0, dtype=dtypes.int64)
      examples = variables.Variable(zero64)
      counter = examples.count_up_to(num_a)
      sparse_counter = sparse_tensor.SparseTensor(
          indices=array_ops.reshape(zero64, [1, 1]),
          values=array_ops_stack.stack(
              [math_ops.cast(counter, dtypes.float32)]),
          dense_shape=[1])

      # The second generates (99, "b") 35 times and then stops.
      num_b = 35
      ninety_nine = inp.limit_epochs(
          constant_op.constant(
              99, dtype=dtypes.int64), num_b)
      sparse_ninety_nine = sparse_tensor.SparseTensor(
          indices=array_ops.reshape(zero64, [1, 1]),
          values=array_ops_stack.stack(
              [math_ops.cast(ninety_nine, dtypes.float32)]),
          dense_shape=[1])

      # These get joined together and grouped into batches of 5.
      batch_size = 5
      if use_dict:
        batched = inp.shuffle_batch_join(
            [{
                "c": counter,
                "s": sparse_counter,
                "S": "a"
            }, {
                "c": ninety_nine,
                "s": sparse_ninety_nine,
                "S": "b"
            }],
            batch_size=batch_size,
            capacity=32,
            min_after_dequeue=16,
            seed=223607)
        batched_fetch = [batched["c"], batched["s"], batched["S"]]
      else:
        batched = inp.shuffle_batch_join(
            [[counter, sparse_counter, "a"],
             [ninety_nine, sparse_ninety_nine, "b"]],
            batch_size=batch_size,
            capacity=32,
            min_after_dequeue=16,
            seed=223607)
        batched_fetch = batched

      # Shapes.
      self.assertEqual(3, len(batched_fetch))
      self.assertAllEqual((batch_size,), batched_fetch[0].get_shape().as_list())
      self.assertAllEqual((None, 2),
                          batched_fetch[1].indices.get_shape().as_list())
      self.assertAllEqual((None,),
                          batched_fetch[1].values.get_shape().as_list())
      self.assertAllEqual((2,),
                          batched_fetch[1].dense_shape.get_shape().as_list())
      self.assertAllEqual((batch_size,), batched_fetch[2].get_shape().as_list())

      self.evaluate(variables.global_variables_initializer())
      variables.local_variables_initializer().run()
      threads = queue_runner_impl.start_queue_runners()

      # Should see the "a" and "b" threads mixed together.
      all_a = []
      seen_b = 0
      saw_both = 0
      num_batches = (num_a + num_b) // batch_size
      for i in range(num_batches):
        results = self.evaluate(batched_fetch)
        self.assertEqual(3, len(results))
        self.assertEqual(len(results[0]), batch_size)
        self.assertEqual(len(results[2]), batch_size)
        self.assertAllEqual(results[0], results[1].values)
        self.assertAllEqual(
            results[1].indices,
            np.vstack((np.arange(batch_size), np.zeros(batch_size))).T)
        self.assertAllEqual(results[1].dense_shape, [batch_size, 1])
        which_a = [i for i, s in enumerate(results[2]) if s == b"a"]
        which_b = [i for i, s in enumerate(results[2]) if s == b"b"]
        self.assertEqual(len(which_a) + len(which_b), batch_size)
        if which_a and which_b:
          saw_both += 1
        all_a.extend(results[0][i] for i in which_a)
        seen_b += len(which_b)
        self.assertAllEqual([99] * len(which_b),
                            [results[0][i] for i in which_b])

      # Some minimum level of mixing of the results of both threads.
      self.assertGreater(saw_both, 1)

      # Saw all the items from "a", but scrambled.
      self.assertItemsEqual(all_a, range(num_a))
      deltas = [all_a[i + 1] - all_a[i] for i in range(len(all_a) - 1)]
      self.assertFalse(all(d == deltas[0] for d in deltas))
      self.assertEqual(seen_b, num_b)

      # Reached the limit.
      with self.assertRaises(errors_impl.OutOfRangeError):
        self.evaluate(batched_fetch)
      for thread in threads:
        thread.join()

  def testTwoThreads(self):
    # [FUNCTION DOC]
    # Runs the two-producer shuffle join scenario with list input.
    self._testTwoThreadsHelper(use_dict=False)

  def testTwoThreadsDict(self):
    # [FUNCTION DOC]
    # Runs the two-producer shuffle join scenario with dictionary input.
    self._testTwoThreadsHelper(use_dict=True)

  def testTwoThreadsSmallerBatch(self):
    # [FUNCTION DOC]
    # Scenario: two producers with two extra elements each (27 and 37 values), joined with a
    # shuffling queue and a smaller final batch allowed.  Asserts unknown batch dimensions,
    # structure of the full batches, a final batch of four elements, mixing in more than one
    # batch, scrambled but complete "a" values, complete "b" count, and the out-of-range error.
    with ops.Graph().as_default(), self.cached_session():
      # Two threads, the first generates (0..26, "a").
      extra_elements = 2
      num_a = 25 + extra_elements
      zero64 = constant_op.constant(0, dtype=dtypes.int64)
      examples = variables.Variable(zero64)
      counter = examples.count_up_to(num_a)
      sparse_counter = sparse_tensor.SparseTensor(
          indices=array_ops.reshape(zero64, [1, 1]),
          values=array_ops_stack.stack(
              [math_ops.cast(counter, dtypes.float32)]),
          dense_shape=[1])

      # The second generates (99, "b") 37 times and then stops.
      num_b = 35 + extra_elements
      ninety_nine = inp.limit_epochs(
          constant_op.constant(
              99, dtype=dtypes.int64), num_b)
      sparse_ninety_nine = sparse_tensor.SparseTensor(
          indices=array_ops.reshape(zero64, [1, 1]),
          values=array_ops_stack.stack(
              [math_ops.cast(ninety_nine, dtypes.float32)]),
          dense_shape=[1])

      # These get joined together and grouped into batches of 5.
      batch_size = 5
      batched = inp.shuffle_batch_join(
          [[counter, sparse_counter, "a"],
           [ninety_nine, sparse_ninety_nine, "b"]],
          batch_size=batch_size,
          capacity=32,
          min_after_dequeue=16,
          seed=223607,
          allow_smaller_final_batch=True)

      # Shapes.
      self.assertEqual(3, len(batched))
      self.assertAllEqual((None,), batched[0].get_shape().as_list())
      self.assertAllEqual((None, 2), batched[1].indices.get_shape().as_list())
      self.assertAllEqual((None,), batched[1].values.get_shape().as_list())
      self.assertAllEqual((2,), batched[1].dense_shape.get_shape().as_list())
      self.assertAllEqual((None,), batched[2].get_shape().as_list())

      self.evaluate(variables.global_variables_initializer())
      variables.local_variables_initializer().run()
      threads = queue_runner_impl.start_queue_runners()

      # Should see the "a" and "b" threads mixed together.
      all_a = []
      seen_b = 0
      saw_both = 0
      num_batches = (num_a + num_b) // batch_size
      for i in range(num_batches):
        results = self.evaluate(batched)
        tf_logging.info("Batch %d: %s", i, results[0])
        self.assertEqual(len(results[0]), batch_size)
        self.assertEqual(len(results[2]), batch_size)
        self.assertAllEqual(results[0], results[1].values)
        self.assertAllEqual(
            results[1].indices,
            np.vstack((np.arange(batch_size), np.zeros(batch_size))).T)
        self.assertAllEqual(results[1].dense_shape, [batch_size, 1])
        which_a = [i for i, s in enumerate(results[2]) if s == b"a"]
        which_b = [i for i, s in enumerate(results[2]) if s == b"b"]
        self.assertEqual(len(which_a) + len(which_b), batch_size)
        if which_a and which_b:
          saw_both += 1
        all_a.extend(results[0][i] for i in which_a)
        seen_b += len(which_b)
        self.assertAllEqual([99] * len(which_b),
                            [results[0][i] for i in which_b])

      # Reached end with 2 * extra_elements left
      results = self.evaluate(batched)
      self.assertEqual(len(results[0]), 2 * extra_elements)
      self.assertAllEqual(results[1].dense_shape, [2 * extra_elements, 1])
      self.assertEqual(len(results[2]), 2 * extra_elements)
      self.assertAllEqual(results[0], results[1].values)
      self.assertAllEqual(results[1].indices,
                          np.vstack((np.arange(2 * extra_elements),
                                     np.zeros(2 * extra_elements))).T)
      which_a = [i for i, s in enumerate(results[2]) if s == b"a"]
      which_b = [i for i, s in enumerate(results[2]) if s == b"b"]
      self.assertEqual(len(which_a) + len(which_b), 2 * extra_elements)
      if which_a and which_b:
        saw_both += 1
      all_a.extend(results[0][i] for i in which_a)
      seen_b += len(which_b)

      # Some minimum level of mixing of the results of both threads.
      self.assertGreater(saw_both, 1)

      # Saw all the items from "a", but scrambled, including extras.
      self.assertItemsEqual(all_a, range(num_a))
      deltas = [all_a[i + 1] - all_a[i] for i in range(len(all_a) - 1)]
      self.assertFalse(all(d == deltas[0] for d in deltas))
      self.assertEqual(seen_b, num_b)

      # Reached the limit.
      with self.assertRaises(errors_impl.OutOfRangeError):
        self.evaluate(batched)
      for thread in threads:
        thread.join()

  def testMismatchedDictKeys(self):
    # [FUNCTION DOC]
    # Scenario: shuffle_batch_join is given two dictionaries with different keys.  Asserts a value
    # error saying they must have the same keys.
    with ops.Graph().as_default(), self.assertRaisesRegex(
        ValueError, "must have the same keys"):
      inp.shuffle_batch_join(
          [{
              "c": 12,
              "s": 123,
              "S": "a"
          }, {
              "cool": -12,
              "s": 99,
              "S": "b"
          }],
          batch_size=8,
          capacity=32,
          min_after_dequeue=16,
          seed=223607)

  def testSharedName(self):
    # [FUNCTION DOC]
    # Scenario: shuffle_batch_join with a single input list, a shared name and a name.  Asserts
    # the static output shapes equal the batch size and the shared-name attribute is present.
    with ops.Graph().as_default(), self.cached_session():
      batch_size = 10
      num_batches = 3
      zero64 = constant_op.constant(0, dtype=dtypes.int64)
      examples = variables.Variable(zero64)
      counter = examples.count_up_to(num_batches * batch_size)
      batched = inp.shuffle_batch_join(
          [[counter, "string"]],
          batch_size=batch_size,
          capacity=32,
          min_after_dequeue=10,
          shared_name="SHARED_NAME_XYZ",
          name="Q")

      # Shapes.
      self.assertEqual(2, len(batched))
      self.assertAllEqual((batch_size,), batched[0].get_shape().as_list())
      self.assertAllEqual((batch_size,), batched[1].get_shape().as_list())

      self.assertProtoEquals(
          "s: 'SHARED_NAME_XYZ'",
          batched[0].op.inputs[0].op.node_def.attr["shared_name"])

  def _testKeepInputHelper(self, num_threads, enqueue_many,
                           keep_input_vector=False):
    # [FUNCTION DOC]
    # HELPER: Shared body for the conditional shuffle join tests.
    # Parameters: num_threads is how many copies of the input list are joined; enqueue_many tells
    # whether inputs are pre-batched; keep_input_vector (optional, default false) selects a
    # per-example keep vector.  Only even counter values are kept.  Asserts four batches of five
    # even values plus string and then an out-of-range error.  Uses deprecated initializers.
    # Returns nothing.
    with ops.Graph().as_default(), self.cached_session():
      batch_size = 5
      num_batches = 4
      examples = variables.Variable(0)
      counter = examples.count_up_to(num_batches * batch_size * 2)
      sparse_counter = sparse_tensor.SparseTensor(
          indices=array_ops.zeros(
              [1, 1], dtype=dtypes.int64),
          values=array_ops_stack.stack(
              [math_ops.cast(counter, dtypes.float32)]),
          dense_shape=[1])
      to_batch = [counter, sparse_counter, "string"]
      if enqueue_many:
        to_batch = inp.batch(to_batch, 4 if keep_input_vector else 1)
      keep_input = array_ops.squeeze(
          math_ops.equal(0, math_ops.mod(to_batch[0], 2)))
      batched = inp.maybe_shuffle_batch_join(
          [to_batch] * num_threads,
          batch_size,
          10,
          1,
          keep_input,
          enqueue_many=enqueue_many)
      variables.initialize_all_variables().run()
      variables.initialize_local_variables().run()
      threads = queue_runner_impl.start_queue_runners()

      for _ in range(num_batches):
        results = self.evaluate(batched)
        self.assertAllEqual([0] * batch_size, np.mod(results[0], 2))
        self.assertAllEqual([0] * batch_size, np.mod(results[1].values, 2))
        self.assertAllEqual([b"string"] * batch_size, results[2])

      # Reached the limit.
      with self.assertRaises(errors_impl.OutOfRangeError):
        self.evaluate(batched)
      for thread in threads:
        thread.join()

  @test_util.run_v1_only("Input pipelines based on Queues are not supported "
                         "when eager execution is enabled. TF2 uses tf.data "
                         "instead.")
  def testSingleThreadKeepInput(self):
    # [FUNCTION DOC]
    # Conditional shuffle join with one thread and no enqueue-many; graph mode (TF1) only.
    self._testKeepInputHelper(1, False)

  @test_util.run_v1_only("Input pipelines based on Queues are not supported "
                         "when eager execution is enabled. TF2 uses tf.data "
                         "instead.")
  def testSingleThreadKeepInputEnqueueMany(self):
    # [FUNCTION DOC]
    # Conditional shuffle join with one thread and enqueue-many; graph mode (TF1) only.
    self._testKeepInputHelper(1, True)

  @test_util.run_v1_only("Input pipelines based on Queues are not supported "
                         "when eager execution is enabled. TF2 uses tf.data "
                         "instead.")
  def testMultipleThreadKeepInput(self):
    # [FUNCTION DOC]
    # Conditional shuffle join with five threads and no enqueue-many; graph mode (TF1) only.
    self._testKeepInputHelper(5, False)

  @test_util.run_v1_only("Input pipelines based on Queues are not supported "
                         "when eager execution is enabled. TF2 uses tf.data "
                         "instead.")
  def testMultipleThreadKeepInputEnqueueMany(self):
    # [FUNCTION DOC]
    # Conditional shuffle join with five threads and enqueue-many; graph mode (TF1) only.
    self._testKeepInputHelper(5, True)

  def testSingleThreadKeepInputPerExample(self):
    # [FUNCTION DOC]
    # Conditional shuffle join with one thread, enqueue-many and a per-example keep vector.
    self._testKeepInputHelper(1, True, keep_input_vector=True)

  def testMultipleThreadKeepInputPerExample(self):
    # [FUNCTION DOC]
    # Conditional shuffle join with five threads, enqueue-many and a per-example keep vector.
    self._testKeepInputHelper(5, True, keep_input_vector=True)

  def testInvalidKeepInputVector(self):
    # [FUNCTION DOC]
    # Scenario: three invalid uses of maybe_shuffle_batch_join, each expected to raise a value
    # error: vector keep condition without enqueue-many, keep condition with more than one
    # dimension, and keep condition of unknown shape at graph construction.
    with ops.Graph().as_default():
      # Can't have vector `keep_input` with `enqueue_many=False`.
      with self.assertRaisesRegex(ValueError,
                                  "`keep_input` cannot be a vector"):
        inp.maybe_shuffle_batch_join([[array_ops.zeros(5)]],
                                     1,
                                     10,
                                     1,
                                     keep_input=constant_op.constant(
                                         [True, False]),
                                     enqueue_many=False)
      # Can't have `keep_input` with more than one dimension.
      with self.assertRaisesRegex(ValueError, "must be 0 or 1 dimensions"):
        inp.maybe_shuffle_batch_join([[array_ops.zeros(5)]],
                                     1,
                                     10,
                                     1,
                                     keep_input=constant_op.constant([[True]]),
                                     enqueue_many=True)
      # `keep_input` must have dimensions determined at graph construction.
      with self.assertRaisesRegex(ValueError,
                                  "must be known at graph construction"):
        inp.maybe_shuffle_batch_join([[array_ops.zeros(5)]],
                                     1,
                                     10,
                                     1,
                                     keep_input=array_ops.placeholder(
                                         dtypes.bool),
                                     enqueue_many=True)

  def testMaybeBatchedSparseTensorInferredShape(self):
    # [FUNCTION DOC]
    # Scenario: maybe_shuffle_batch_join with a true keep condition on one sparse input.  Asserts
    # the batched dense-shape vector has two elements.
    with ops.Graph().as_default():
      sparse = sparse_tensor.SparseTensor(
          indices=[[0]], values=[1.0], dense_shape=[1])
      self.assertAllEqual((1,), sparse.dense_shape.get_shape().as_list())
      batched = inp.maybe_shuffle_batch_join([[sparse]], 2, 10, 1, True)
      self.assertAllEqual((2,), batched.dense_shape.get_shape().as_list())

  def testMaybeBatchedSparseTensorInferredShapeEnqueueMany(self):
    # [FUNCTION DOC]
    # Scenario: maybe_shuffle_batch_join with enqueue-many.  Asserts the dense-shape vector keeps
    # one element.
    with ops.Graph().as_default():
      sparse = sparse_tensor.SparseTensor(
          indices=[[0]], values=[1.0], dense_shape=[1])
      self.assertAllEqual((1,), sparse.dense_shape.get_shape().as_list())
      batched = inp.maybe_shuffle_batch_join([[sparse]],
                                             2,
                                             10,
                                             1,
                                             True,
                                             enqueue_many=True)
      self.assertAllEqual((1,), batched.dense_shape.get_shape().as_list())

  def testMaybeBatchedSparseTensorInferredShapeEnqueueManyPerExample(self):
    # [FUNCTION DOC]
    # Scenario: maybe_shuffle_batch_join with enqueue-many and a per-example keep list.  Asserts
    # the dense-shape vector keeps one element.
    with ops.Graph().as_default():
      sparse = sparse_tensor.SparseTensor(
          indices=[[0], [0]], values=[1.0, 2.0], dense_shape=[2])
      self.assertAllEqual((1,), sparse.dense_shape.get_shape().as_list())
      batched = inp.maybe_shuffle_batch_join([[sparse]],
                                             2,
                                             10,
                                             1, [True, False],
                                             enqueue_many=True)
      self.assertAllEqual((1,), batched.dense_shape.get_shape().as_list())

  def testMaybeBatchedSparseTensorInferredShapeUnknownRank(self):
    # [FUNCTION DOC]
    # Scenario: maybe_shuffle_batch_join on an unknown-rank sparse tensor.  Asserts the
    # dense-shape vector's element count is unknown.
    with ops.Graph().as_default():
      sparse = sparse_tensor.SparseTensor(
          indices=array_ops.placeholder(dtypes.int64),
          values=array_ops.placeholder(dtypes.float32),
          dense_shape=array_ops.placeholder(dtypes.int64))
      self.assertIs(None, sparse.dense_shape.get_shape().num_elements())
      batched = inp.maybe_shuffle_batch_join([[sparse]], 2, 10, 1, True)
      self.assertIs(None, batched.dense_shape.get_shape().num_elements())

  def testMaybeBatchedSparseTensorInferredShapeUnknownRankEnqueueMany(self):
    # [FUNCTION DOC]
    # Scenario: maybe_shuffle_batch_join with enqueue-many on an unknown-rank sparse tensor.
    # Asserts the element count stays unknown.
    with ops.Graph().as_default():
      sparse = sparse_tensor.SparseTensor(
          indices=array_ops.placeholder(dtypes.int64),
          values=array_ops.placeholder(dtypes.float32),
          dense_shape=array_ops.placeholder(dtypes.int64))
      self.assertIs(None, sparse.dense_shape.get_shape().num_elements())
      batched = inp.maybe_shuffle_batch_join([[sparse]],
                                             2,
                                             10,
                                             1,
                                             True,
                                             enqueue_many=True)
      self.assertIs(None, batched.dense_shape.get_shape().num_elements())

  def testMaybeBatchedSparseTensorInferredShapeUnknownRankPerExample(self):
    # [FUNCTION DOC]
    # Scenario: maybe_shuffle_batch_join with enqueue-many and a per-example keep list on an
    # unknown-rank sparse tensor.  Asserts the element count stays unknown.
    with ops.Graph().as_default():
      sparse = sparse_tensor.SparseTensor(
          indices=array_ops.placeholder(dtypes.int64),
          values=array_ops.placeholder(dtypes.float32),
          dense_shape=array_ops.placeholder(dtypes.int64))
      self.assertIs(None, sparse.dense_shape.get_shape().num_elements())
      batched = inp.maybe_shuffle_batch_join([[sparse]],
                                             2,
                                             10,
                                             1, [True, False],
                                             enqueue_many=True)
      self.assertIs(None, batched.dense_shape.get_shape().num_elements())


if __name__ == "__main__":
  test_lib.main()
