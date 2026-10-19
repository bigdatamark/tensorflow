# [MODULE DOC]
# Unit tests for the priority queue in the data flow ops module. A priority queue takes an
# integer (64-bit) priority as the first component of each element and always dequeues elements
# in ascending priority order. The tests cover round trips with single and batched inserts and
# reads, multi-threaded producers and consumers, and error handling for non-integer or non-scalar
# priorities. They run only under the graph-mode (version 1) test runner because the class is
# not available in version 2 behavior. There are no import-time side effects other than importing
# the neural-network gradient module for registration and the standard test entry point.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.data_flow_ops.PriorityQueue."""

import copy
import random
import threading

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


@test_util.run_v1_only("PriorityQueue removed from v2")
class PriorityQueueTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the priority queue. Restricted to graph-mode test execution. Instances hold no
  # state; each test builds its own queue in a cached session with random data and checks that
  # elements come out ordered by priority with their side values intact (side values for equal
  # priorities may come out in any order, so tests check membership rather than exact pairing).

  def testRoundTripInsertReadOnceSorts(self):
    # [FUNCTION DOC]
    # Test: a queue of capacity two thousand with two string side components. One hundred random
    # small integer priorities with random byte-string side values are enqueued one element at a
    # time, then all are dequeued with a single dequeue-many of one hundred. Asserts the priorities
    # come out sorted, that each dequeued side-value pair was one that was inserted with that
    # priority, and that every inserted pair was dequeued exactly once. Uses random data.
    with self.cached_session() as sess:
      q = data_flow_ops.PriorityQueue(2000, (dtypes.string, dtypes.string), (
          (), ()))
      elem = np.random.randint(-5, 5, size=100).astype(np.int64)
      side_value_0 = np.random.rand(100).astype(bytes)
      side_value_1 = np.random.rand(100).astype(bytes)
      enq_list = [
          q.enqueue((e, constant_op.constant(v0), constant_op.constant(v1)))
          for e, v0, v1 in zip(elem, side_value_0, side_value_1)
      ]
      for enq in enq_list:
        enq.run()

      deq = q.dequeue_many(100)
      deq_elem, deq_value_0, deq_value_1 = self.evaluate(deq)

      allowed = {}
      missed = set()
      for e, v0, v1 in zip(elem, side_value_0, side_value_1):
        if e not in allowed:
          allowed[e] = set()
        allowed[e].add((v0, v1))
        missed.add((v0, v1))

      self.assertAllEqual(deq_elem, sorted(elem))
      for e, dv0, dv1 in zip(deq_elem, deq_value_0, deq_value_1):
        self.assertTrue((dv0, dv1) in allowed[e])
        missed.remove((dv0, dv1))
      self.assertEqual(missed, set())

  def testRoundTripInsertMultiThreadedReadOnceSorts(self):
    # [FUNCTION DOC]
    # Test: same data as the single-threaded round trip but each of the one hundred single enqueues
    # runs in its own thread while the main thread performs one dequeue-many of one hundred. The
    # result priorities must be sorted and every side-value pair must match an inserted pair for that
    # priority with none lost or duplicated. Uses per-thread device scoping and random data.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.PriorityQueue(2000, (dtypes.string, dtypes.string), (
          (), ()))
      elem = np.random.randint(-5, 5, size=100).astype(np.int64)
      side_value_0 = np.random.rand(100).astype(bytes)
      side_value_1 = np.random.rand(100).astype(bytes)

      enqueue_ops = [
          q.enqueue((e, constant_op.constant(v0), constant_op.constant(v1)))
          for e, v0, v1 in zip(elem, side_value_0, side_value_1)
      ]

      # Run one producer thread for each element in elems.
      def enqueue(enqueue_op):
        # [FUNCTION DOC]
        # HELPER: producer thread body for testRoundTripInsertMultiThreadedReadOnceSorts.
        # Evaluates the enqueue operation passed in as the argument. Returns nothing.
        self.evaluate(enqueue_op)

      dequeue_op = q.dequeue_many(100)

      enqueue_threads = [
          self.checkedThread(
              target=enqueue, args=(op,)) for op in enqueue_ops
      ]

      for t in enqueue_threads:
        t.start()

      deq_elem, deq_value_0, deq_value_1 = self.evaluate(dequeue_op)

      for t in enqueue_threads:
        t.join()

      allowed = {}
      missed = set()
      for e, v0, v1 in zip(elem, side_value_0, side_value_1):
        if e not in allowed:
          allowed[e] = set()
        allowed[e].add((v0, v1))
        missed.add((v0, v1))

      self.assertAllEqual(deq_elem, sorted(elem))
      for e, dv0, dv1 in zip(deq_elem, deq_value_0, deq_value_1):
        self.assertTrue((dv0, dv1) in allowed[e])
        missed.remove((dv0, dv1))
      self.assertEqual(missed, set())

  def testRoundTripFillsCapacityMultiThreadedEnqueueAndDequeue(self):
    # [FUNCTION DOC]
    # Test: a small queue of capacity ten with an integer priority component and forty producer
    # threads, each batch-enqueuing a random number (up to nine) of small random integers as both
    # priority and value, while forty consumer threads dequeue-many with a shuffled copy of the same
    # counts. The queue regularly fills, so producers and consumers block on each other. Each consumer
    # checks priorities equal values; at the end the sorted set of everything dequeued must equal the
    # sorted set of everything enqueued. Uses randomness and per-thread device scoping.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.PriorityQueue(10, (dtypes.int64), (()))

      num_threads = 40
      enqueue_counts = np.random.randint(10, size=num_threads)
      enqueue_values = [
          np.random.randint(
              5, size=count) for count in enqueue_counts
      ]
      enqueue_ops = [
          q.enqueue_many((values, values)) for values in enqueue_values
      ]
      shuffled_counts = copy.deepcopy(enqueue_counts)
      random.shuffle(shuffled_counts)
      dequeue_ops = [q.dequeue_many(count) for count in shuffled_counts]
      all_enqueued_values = np.hstack(enqueue_values)

      # Run one producer thread for each element in elems.
      def enqueue(enqueue_op):
        # [FUNCTION DOC]
        # HELPER: producer thread body for testRoundTripFillsCapacityMultiThreadedEnqueueAndDequeue.
        # Evaluates the batched enqueue operation given as the argument; may block while the queue is
        # full. Returns nothing.
        self.evaluate(enqueue_op)

      dequeued = []

      def dequeue(dequeue_op):
        # [FUNCTION DOC]
        # HELPER: consumer thread body for testRoundTripFillsCapacityMultiThreadedEnqueueAndDequeue.
        # Evaluates the dequeue-many operation given as the argument, asserts the returned priorities
        # equal the returned values, and appends the priorities to the enclosing list of dequeued items.
        # Returns nothing.
        (dequeue_indices, dequeue_values) = self.evaluate(dequeue_op)
        self.assertAllEqual(dequeue_indices, dequeue_values)
        dequeued.extend(dequeue_indices)

      enqueue_threads = [
          self.checkedThread(
              target=enqueue, args=(op,)) for op in enqueue_ops
      ]
      dequeue_threads = [
          self.checkedThread(
              target=dequeue, args=(op,)) for op in dequeue_ops
      ]

      # Dequeue and check
      for t in dequeue_threads:
        t.start()
      for t in enqueue_threads:
        t.start()
      for t in enqueue_threads:
        t.join()
      for t in dequeue_threads:
        t.join()

      self.assertAllEqual(sorted(dequeued), sorted(all_enqueued_values))

  def testRoundTripInsertManyMultiThreadedReadManyMultithreadedSorts(self):
    # [FUNCTION DOC]
    # Test: a large queue (capacity two thousand) receives batched enqueues of random sizes from forty
    # threads; once all producers have finished, forty consumer threads run dequeue-many with a
    # shuffled copy of the same counts, guarding the shared list with a lock. Because appends are not
    # atomic with the dequeue, full sorting is not asserted; the test only asserts that the set of
    # dequeued values equals the set of enqueued values. Uses randomness.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.PriorityQueue(2000, (dtypes.int64), (()))

      num_threads = 40
      enqueue_counts = np.random.randint(10, size=num_threads)
      enqueue_values = [
          np.random.randint(
              5, size=count) for count in enqueue_counts
      ]
      enqueue_ops = [
          q.enqueue_many((values, values)) for values in enqueue_values
      ]
      shuffled_counts = copy.deepcopy(enqueue_counts)
      random.shuffle(shuffled_counts)
      dequeue_ops = [q.dequeue_many(count) for count in shuffled_counts]
      all_enqueued_values = np.hstack(enqueue_values)

      dequeue_wait = threading.Condition()

      # Run one producer thread for each element in elems.
      def enqueue(enqueue_op):
        # [FUNCTION DOC]
        # HELPER: producer thread body for testRoundTripInsertManyMultiThreadedReadManyMultithreadedSorts.
        # Evaluates the batched enqueue operation given as the argument. Returns nothing.
        self.evaluate(enqueue_op)

      def dequeue(dequeue_op, dequeued):
        # [FUNCTION DOC]
        # HELPER: consumer thread body for testRoundTripInsertManyMultiThreadedReadManyMultithreadedSorts.
        # Takes a dequeue-many operation and the shared result list. Evaluates the operation, asserts
        # priorities equal values, then under the shared condition lock appends the priorities to the list
        # (acquire and release around the update). Returns nothing.
        (dequeue_indices, dequeue_values) = self.evaluate(dequeue_op)
        self.assertAllEqual(dequeue_indices, dequeue_values)
        dequeue_wait.acquire()
        dequeued.extend(dequeue_indices)
        dequeue_wait.release()

      dequeued = []
      enqueue_threads = [
          self.checkedThread(
              target=enqueue, args=(op,)) for op in enqueue_ops
      ]
      dequeue_threads = [
          self.checkedThread(
              target=dequeue, args=(op, dequeued)) for op in dequeue_ops
      ]

      for t in enqueue_threads:
        t.start()
      for t in enqueue_threads:
        t.join()
      # Dequeue and check
      for t in dequeue_threads:
        t.start()
      for t in dequeue_threads:
        t.join()

      # We can't guarantee full sorting because we can't guarantee
      # that the dequeued.extend() call runs immediately after the
      # self.evaluate() call.  Here we're just happy everything came out.
      self.assertAllEqual(set(dequeued), set(all_enqueued_values))

  def testRoundTripInsertManyMultiThreadedReadOnceSorts(self):
    # [FUNCTION DOC]
    # Test: one hundred random priorities with string side values are split into twenty batches of
    # five, each enqueued by its own thread via batched enqueue, while the main thread performs one
    # dequeue-many of one hundred. Asserts sorted priorities and that every side-value pair matches an
    # inserted pair for its priority with none lost or duplicated. Uses randomness.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.PriorityQueue(2000, (dtypes.string, dtypes.string), (
          (), ()))
      elem = np.random.randint(-5, 5, size=100).astype(np.int64)
      side_value_0 = np.random.rand(100).astype(bytes)
      side_value_1 = np.random.rand(100).astype(bytes)

      batch = 5
      enqueue_ops = [
          q.enqueue_many((elem[i * batch:(i + 1) * batch],
                          side_value_0[i * batch:(i + 1) * batch],
                          side_value_1[i * batch:(i + 1) * batch]))
          for i in range(20)
      ]

      # Run one producer thread for each element in elems.
      def enqueue(enqueue_op):
        # [FUNCTION DOC]
        # HELPER: producer thread body for testRoundTripInsertManyMultiThreadedReadOnceSorts.
        # Evaluates the batched enqueue operation given as the argument. Returns nothing.
        self.evaluate(enqueue_op)

      dequeue_op = q.dequeue_many(100)

      enqueue_threads = [
          self.checkedThread(
              target=enqueue, args=(op,)) for op in enqueue_ops
      ]

      for t in enqueue_threads:
        t.start()

      deq_elem, deq_value_0, deq_value_1 = self.evaluate(dequeue_op)

      for t in enqueue_threads:
        t.join()

      allowed = {}
      missed = set()
      for e, v0, v1 in zip(elem, side_value_0, side_value_1):
        if e not in allowed:
          allowed[e] = set()
        allowed[e].add((v0, v1))
        missed.add((v0, v1))

      self.assertAllEqual(deq_elem, sorted(elem))
      for e, dv0, dv1 in zip(deq_elem, deq_value_0, deq_value_1):
        self.assertTrue((dv0, dv1) in allowed[e])
        missed.remove((dv0, dv1))
      self.assertEqual(missed, set())

  def testRoundTripInsertOnceReadOnceSorts(self):
    # [FUNCTION DOC]
    # Test: one thousand random priorities with two string side values each are enqueued in a single
    # batched operation and read back by one dequeue-many of one thousand. Asserts priorities are
    # sorted and that each side-value pair dequeued belongs to the set inserted for that priority
    # (duplicates of pairs are not tracked). Uses random data.
    with self.cached_session() as sess:
      q = data_flow_ops.PriorityQueue(2000, (dtypes.string, dtypes.string), (
          (), ()))
      elem = np.random.randint(-100, 100, size=1000).astype(np.int64)
      side_value_0 = np.random.rand(1000).astype(bytes)
      side_value_1 = np.random.rand(1000).astype(bytes)
      q.enqueue_many((elem, side_value_0, side_value_1)).run()
      deq = q.dequeue_many(1000)
      deq_elem, deq_value_0, deq_value_1 = self.evaluate(deq)

      allowed = {}
      for e, v0, v1 in zip(elem, side_value_0, side_value_1):
        if e not in allowed:
          allowed[e] = set()
        allowed[e].add((v0, v1))

      self.assertAllEqual(deq_elem, sorted(elem))
      for e, dv0, dv1 in zip(deq_elem, deq_value_0, deq_value_1):
        self.assertTrue((dv0, dv1) in allowed[e])

  def testRoundTripInsertOnceReadManySorts(self):
    # [FUNCTION DOC]
    # Test: one thousand random priorities (also used as values) enqueued in one batch and read back
    # in ten dequeue-many calls of one hundred; the concatenated priorities must equal the sorted
    # input.
    with self.cached_session():
      q = data_flow_ops.PriorityQueue(2000, (dtypes.int64), (()))
      elem = np.random.randint(-100, 100, size=1000).astype(np.int64)
      q.enqueue_many((elem, elem)).run()
      deq_values = np.hstack([q.dequeue_many(100)[0].eval() for _ in range(10)])
      self.assertAllEqual(deq_values, sorted(elem))

  def testRoundTripInsertOnceReadOnceLotsSorts(self):
    # [FUNCTION DOC]
    # Test: one thousand random priorities enqueued in one batch and then read one at a time with one
    # thousand single dequeues of the same dequeue operation; the priorities collected must equal the
    # sorted input.
    with self.cached_session():
      q = data_flow_ops.PriorityQueue(2000, (dtypes.int64), (()))
      elem = np.random.randint(-100, 100, size=1000).astype(np.int64)
      q.enqueue_many((elem, elem)).run()
      dequeue_op = q.dequeue()
      deq_values = np.hstack([dequeue_op[0].eval() for _ in range(1000)])
      self.assertAllEqual(deq_values, sorted(elem))

  def testInsertingNonInt64Fails(self):
    # [FUNCTION DOC]
    # Test: a priority queue is created with a string component and a batched enqueue is given lists
    # of strings for both the priority and the value. Running the enqueue must raise a type error
    # because the priority component must be a 64-bit integer.
    with self.cached_session():
      q = data_flow_ops.PriorityQueue(2000, (dtypes.string), (()))
      with self.assertRaises(TypeError):
        q.enqueue_many((["a", "b", "c"], ["a", "b", "c"])).run()

  def testInsertingNonScalarFails(self):
    # [FUNCTION DOC]
    # Test: priorities must be scalars (or a vector for batched enqueue). Feeding a vector of two
    # priorities to a single enqueue must raise an invalid-argument error saying the expected shape of
    # tuple component zero was scalar but the actual was two. Feeding a two by two matrix of
    # priorities to a batched enqueue must raise an invalid-argument error saying the expected shape
    # was a vector of two but the actual was two by two. Uses placeholders fed at run time.
    with self.cached_session() as sess:
      input_priority = array_ops.placeholder(dtypes.int64)
      input_other = array_ops.placeholder(dtypes.string)
      q = data_flow_ops.PriorityQueue(2000, (dtypes.string,), (()))

      with self.assertRaisesRegex(
          errors_impl.InvalidArgumentError,
          r"Shape mismatch in tuple component 0. Expected \[\], got \[2\]"):
        sess.run([q.enqueue((input_priority, input_other))],
                 feed_dict={
                     input_priority: np.array(
                         [0, 2], dtype=np.int64),
                     input_other: np.random.rand(3, 5).astype(bytes)
                 })

      with self.assertRaisesRegex(
          errors_impl.InvalidArgumentError,
          r"Shape mismatch in tuple component 0. Expected \[2\], got \[2,2\]"):
        sess.run(
            [q.enqueue_many((input_priority, input_other))],
            feed_dict={
                input_priority: np.array(
                    [[0, 2], [3, 4]], dtype=np.int64),
                input_other: np.random.rand(2, 3).astype(bytes)
            })


if __name__ == "__main__":
  test.main()
